"""RFID_2014 case: raster reader, recipe, forcing derivation, audit (mask/orientation/topology), MAPLE case generation +
verification (tamper checks), and the solver-input factory. Real-input tests need /home/okin/MAHLERAN and the MAPLE
environment and skip honestly otherwise. Nothing here was run by its author (file-only tools); Codex records results."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from rfid_helpers import RFID_INPUT, needs_rfid, synthetic_capture

pytest.importorskip("maple")

from maple_syrup.case_import import Plot1ImportError
from maple_syrup.rfid_case import (
    audit_rfid,
    derive_applied_forcing,
    load_rfid_recipe,
    read_legacy_grid,
    rfid_inputs,
    rfid_routing_graph,
    verify_rfid_case,
)

RECIPE = Path(__file__).resolve().parents[2] / "cases" / "rfid" / "recipe.yaml"
HEADER = "ncols {c}\nnrows {r}\nxllcorner 0\nyllcorner 0\ncellsize 0.1\nnodata_value -9999\n"


# ------------------------------------------------------------------------------------------------ reader / forcing / recipe
def test_reader_accepts_one_value_per_line_and_refuses_malformed_bodies(tmp_path):
    good = tmp_path / "g.asc"
    good.write_text(HEADER.format(c=2, r=2) + "1\n2\n3\n4\n")
    header, body = read_legacy_grid(good)
    assert body.tolist() == [[1.0, 2.0], [3.0, 4.0]] and header["nodata_value"] == -9999.0
    for name, text, match in (("short", "1\n2\n3\n", "body values"), ("long", "1\n2\n3\n4\n5\n", "body values"),
                              ("word", "1\n2\nx\n4\n", "non-numeric"), ("nan", "1\n2\nnan\n4\n", "non-finite")):
        path = tmp_path / f"{name}.asc"
        path.write_text(HEADER.format(c=2, r=2) + text)
        with pytest.raises(Plot1ImportError, match=match):
            read_legacy_grid(path)
    dup = tmp_path / "dup.asc"
    dup.write_text("ncols 2\nncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 1\n1\n2\n3\n4\n")
    with pytest.raises(Plot1ImportError, match="duplicate header"):
        read_legacy_grid(dup)


def test_applied_forcing_compresses_to_exact_pieces_and_refuses_bad_captures():
    record = derive_applied_forcing(synthetic_capture(), capture_sha256="0" * 64)
    assert record["edges_s"] == [0.0, 2641.0, 2700.0] and record["rate_mm_s"] == [0.03836299851536751, 0.0]
    assert record["total_depth_mm"] == pytest.approx(0.03836299851536751 * 2641, rel=1e-15)
    text = synthetic_capture()
    with pytest.raises(Plot1ImportError, match="iterations are not 1..n"):
        derive_applied_forcing(text.replace("\n2 2.0 ", "\n3 2.0 ", 1), capture_sha256="0" * 64)
    with pytest.raises(Plot1ImportError, match="complete"):
        derive_applied_forcing(text.replace("SYRUP_HYDRO_CAPTURE_COMPLETE", "TRUNCATED"), capture_sha256="0" * 64)
    with pytest.raises(Plot1ImportError, match="identically zero"):
        derive_applied_forcing(synthetic_capture(n_on=0), capture_sha256="0" * 64)
    with pytest.raises(Plot1ImportError, match="refusing to compress"):
        derive_applied_forcing(text, capture_sha256="0" * 64, max_pieces=1)


def test_recipe_loads_and_unknown_keys_are_refused(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    recipe = load_rfid_recipe(RECIPE)
    assert recipe.interior_shape == (104, 58) and recipe.base.cellsize_m == 0.1
    raw = yaml.safe_load(RECIPE.read_text())
    raw["surprise"] = 1
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(Plot1ImportError, match="exactly the keys"):
        load_rfid_recipe(bad)
    raw = yaml.safe_load(RECIPE.read_text())
    raw["hydrology"]["model"] = "pavement_hawkins"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(Plot1ImportError, match="fixed_ksat"):
        load_rfid_recipe(bad)


# ------------------------------------------------------------------------------------------------------------- audit
@needs_rfid
def test_audit_mask_orientation_pits_outlet_and_placeholder(rfid_audit):
    f, r = rfid_audit.fields, rfid_audit.report
    active, dem = f["active"], f["legacy_full_elevation_m"]
    assert active.shape == (104, 58) and dem.shape == (106, 60) and f["legacy_full_rainfall_scaling_native"].shape == (106, 60)
    assert int(active.sum()) == 5697 and int((~active).sum()) == 335
    # inactive interior == DEM nodata (both sentinel-bearing), and the original sentinel is preserved
    assert np.array_equal(~active, dem[1:-1, 1:-1] == -9999.0) and (dem == -9999.0).any()
    # orientation: MAPLE row 0 is the file's LAST row. The file's last row is the south ring (mostly nodata, valid at cols 23..26)
    _, raw = read_legacy_grid(RFID_INPUT / "rfid_2014_dem.asc")
    assert np.array_equal(dem[0], raw[-1]) and np.array_equal(dem[-1], raw[0])
    assert raw[-1, 23] == pytest.approx(0.0049) and dem[0, 23] == pytest.approx(0.0049)
    # topology: 26 strict pits kept as terminal storage, one ring outlet, no filling
    assert int(f["graph_pit_storage"].sum()) == 26 and int(f["graph_outlet"].sum()) == 1
    assert r["terrain"]["outlets_north_first_0based"] == [[104, 23]]
    assert r["terrain"]["native_ring_scaling_at_outlet_receiver"] == pytest.approx(0.9688)
    # placeholder terrain: ONLY the inactive bed cells, equal to the minimum active elevation; active cells untouched
    bed = f["elevation_bed_placeholder_m"]
    interior = dem[1:-1, 1:-1]
    assert np.all(bed[~active] == interior[active].min()) and np.array_equal(bed[active], interior[active])
    assert r["placeholder_terrain"]["cells"] == 335
    # the ACTUAL supplied maps are uniform [0, 0, 0, 0.092, 0.908, 0] everywhere (the XML per-type defaults are unused:
    # use_map_phi is true); closed
    np.testing.assert_allclose(f["composition_table"][0], [0.0, 0.0, 0.0, 0.092, 0.908, 0.0], atol=1e-15)
    assert f["composition_table"].shape[0] == 1
    assert abs(float(f["phi_final"].sum(-1).max()) - 1.0) <= 1e-15
    # inactive cells receive no rain; active scale in the captured native range
    assert np.all(f["rainfall_scaling"][~active] == 0.0)
    assert 0.8 < f["rainfall_scaling"][active].min() and f["rainfall_scaling"][active].max() < 1.3


@needs_rfid
def test_graph_from_sidecar_matches_the_import_and_keeps_every_pit(rfid_audit):
    graph = rfid_routing_graph(rfid_audit.fields, rfid_audit.report)
    assert graph.policy != "strict" and graph.n_active == 5697 and int(graph.pit_storage.sum()) == 26
    assert graph.summary()["n_pit_storage"] == 26 and graph.summary()["n_outlets"] == 1
    assert max(graph.summary()["level_widths"]) >= 1 and len(graph.level_bounds) - 1 == rfid_audit.report["terrain"]["n_levels"]
    # pits hold no outflow capacity (k = 0), every other active cell conveys; the pits collectively receive donors
    k = np.asarray(graph.conveyance).reshape(graph.shape)
    assert np.all(k[graph.pit_storage] == 0.0) and np.all(k[graph.active & ~graph.pit_storage] > 0.0)
    receiver = np.asarray(graph.receiver).reshape(-1)
    flat_pits = np.flatnonzero(np.asarray(graph.pit_storage).reshape(-1))
    assert any((receiver == p).any() for p in flat_pits)
    # no filling/carving: the original DEM (with its sentinel) is what the graph digest hashes
    assert graph.input_sha256 == rfid_audit.report["terrain"]["graph_input_sha256"]


@needs_rfid
@pytest.mark.parametrize("field, value, match", [
    (("expected_topology", "n_pit_storage"), 25, "topology differs"),
    (("expected_topology", "n_active"), 5698, "active cells"),
    (("hydrology", "ksat_mm_per_s"), 0.01, "native type-1 formula"),
    (("hydrology", "suction_mm"), 0.05, "XML"),
])
def test_audit_refuses_recipe_values_that_disagree_with_the_inputs(rfid_recipe, tmp_path, field, value, match):
    import dataclasses

    raw = copy.deepcopy(rfid_recipe.raw)
    raw[field[0]][field[1]] = value
    with pytest.raises(Plot1ImportError, match=match):
        audit_rfid(dataclasses.replace(rfid_recipe, raw=raw), tmp_path / "case")


# ------------------------------------------------------------------------------------- generate / verify / tamper (MAPLE)
@needs_rfid
def test_generated_case_is_an_actual_maple_case_and_verifies(rfid_case):
    case, b = rfid_case.case, rfid_case.binding
    assert b["status"] == "ok" and b["checks"]["compiled_equals_reloaded"] is True
    g = case.config.geometry
    assert (g.ny, g.nx, g.dx_m, g.voxel_dz_m, g.active_layer_thickness_m) == (104, 58, 0.1, 0.1, 0.002)
    assert g.bulk_density_kg_m3 == 1250.0
    assert [c.particle_density_kg_m3 for c in case.config.grain_classes.classes] == [2650.0] * 6
    assert len(case.config.grain_classes.classes) == 6
    mass = np.asarray(case.voxel_column.mass_kg).sum(axis=2) + np.asarray(case.active_layer.mass_kg)
    np.testing.assert_allclose(mass.sum(-1), rfid_case.fields["expected_mass_by_cell_class_kg"].sum(-1), rtol=1e-6)
    assert rfid_case.checks["compiled_state"]["active_composition_max_abs"] <= 1e-9


@needs_rfid
@pytest.mark.parametrize("victim", ["case.yaml", "syrup/rfid_fields.npz", "syrup/rfid_import_report.json",
                                    "source/syrup_derived/elevation_bed_placeholder.npy",
                                    "source/mahleran_input_rfid/rfid_2014_dem.asc", "syrup/forcing/applied_forcing.json"])
def test_verification_detects_tampering_of_any_bound_file(rfid_case, tmp_path, victim):
    copy_dir = tmp_path / "copy"
    shutil.copytree(rfid_case.case_dir, copy_dir)
    target = copy_dir / victim
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(Plot1ImportError):
        verify_rfid_case(copy_dir, allow_maple_source_change=True)


@needs_rfid
def test_a_failed_import_has_no_binding_and_an_existing_output_is_refused(synthetic_recipe, tmp_path):
    from maple_syrup.rfid_case import generate_rfid_case

    capture = tmp_path / "capture.txt"
    capture.write_text(synthetic_capture())
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(Plot1ImportError, match="existing path"):
        generate_rfid_case(synthetic_recipe, out, applied_forcing_capture=capture)
    fresh = tmp_path / "fresh"
    with pytest.raises(Plot1ImportError, match="not found"):  # refused BEFORE the output directory is created
        generate_rfid_case(synthetic_recipe, fresh, applied_forcing_capture=tmp_path / "missing.txt")
    assert not fresh.exists()


# --------------------------------------------------------------------------------------------------------------- inputs
@needs_rfid
def test_inputs_factory_binds_graph_parameters_forcing_and_leaves_the_bed_frozen(rfid_case):
    from maple_syrup.column_experiment import _bed_digest
    from maple_syrup.experimental_hydrology import CpuHydraulicSolver, HydraulicControl
    from maple_syrup.experimental_storm import ExperimentalControl, evolve_experimental

    before = _bed_digest(rfid_case.case)
    inputs = rfid_inputs(rfid_case, "numpy", with_geometry=True)
    active = np.asarray(inputs.graph.active)
    assert inputs.params.model == "fixed_ksat" and np.array_equal(np.asarray(inputs.params.active_mask), active)
    assert np.all(np.asarray(inputs.soil0)[~active] == 0.0) and np.all(np.asarray(inputs.depth0) == 0.0)
    np.testing.assert_allclose(np.asarray(inputs.soil0)[active], 0.004 * 0.21, rtol=1e-15)
    np.testing.assert_allclose(np.asarray(inputs.params.suction_m), 0.0236, rtol=1e-15)  # the XML value, not the native 0.05 mm bug
    assert float(np.asarray(inputs.params.drainage_parameter).max()) == 0.05
    assert inputs.schedule.edges_s.tolist() == [0.0, 2641.0, 2700.0] and inputs.schedule.provenance.kind == "native_applied_capture"
    assert inputs.geometry is not None and inputs.geometry.summary()["n_open_faces"] == 1
    assert np.all(np.isfinite(inputs.geometry.z))  # placeholders substituted for the sentinel; those faces are closed
    solver = CpuHydraulicSolver("explicit", inputs.graph, inputs.params, control=HydraulicControl())
    result = evolve_experimental(solver, inputs.field, inputs.schedule, solver.initial_state(inputs.depth0, inputs.soil0), 10.0,
                                 ExperimentalControl(max_dt_s=1.0), report_every_s=5.0)
    assert result.n_accepted_steps >= 10 and float(np.sum(result.cumulative_rain_m)) > 0.0
    assert np.all(np.asarray(result.cumulative_rain_m)[~active] == 0.0)  # inactive cells never receive rain
    assert _bed_digest(rfid_case.case) == before  # the actual MAPLE bed is not written by a water-only run
    json.dumps(inputs.parameter_record)
