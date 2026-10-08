"""Actual-MAPLE row-band tiles versus the dense bed, the generated/verified synthetic Chastre case, its refusals and the
runtime bed guard. The synthetic terrain is 10 x 8; the real 33 GB case is never generated here. Nothing here was run by its
author (file-only tools); Codex records results."""
import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from chastre_helpers import NODATA, NX, NY, read_dtm_interior, recipe_dict

from maple_syrup.case_import import Plot1ImportError


def _definition(rfid_source, dtm_path):
    """The global bed definition from the values READ BACK from the generated ESRI fixture (the same values the generated case
    audits), not from the in-memory construction."""
    from maple_syrup.chastre_case import define_bed
    from maple_syrup.rfid_case import load_rfid_recipe

    report = rfid_source.report
    src = load_rfid_recipe(report["recipe"]["recipe_path"], mahleran_root=report["mahleran"]["root"])
    interior = read_dtm_interior(dtm_path)
    active = interior != NODATA
    bed_dem = np.where(active, interior, interior[active].min())
    base = dataclasses.replace(src.base, case_name="synthetic", legacy_nrows=NY + 2, legacy_ncols=NX + 2, cellsize_m=1.0)
    return define_bed(base, bed_dem, rfid_source.fields["composition_table"], report["bed"]["particle_density_kg_m3"])


def _arrays(loaded):
    return {"voxel": loaded.voxel_column.mass_kg, "active": loaded.active_layer.mass_kg,
            "available": loaded.sediment_availability.available_mass_kg, "bound": loaded.sediment_availability.bound_mass_kg,
            "elevation": loaded.topography_result.elevation_m}


def test_tiles_equal_the_dense_bed_bitwise(rfid_source, dtm_path, tmp_path):
    from maple_syrup.chastre_case import compile_tile, tile_bands

    definition = _definition(rfid_source, dtm_path)
    _, dense = compile_tile(definition, 0, 0, NY, tmp_path / "dense", keep=True)
    dense = {k: np.asarray(v) for k, v in _arrays(dense).items()}
    bands = tile_bands(NY, 4)
    assert bands == [(0, 4), (4, 8), (8, 10)]
    parts = []
    for index, (r0, r1) in enumerate(bands):
        record, loaded = compile_tile(definition, index, r0, r1, tmp_path / f"tile_{index}", keep=True)
        assert record["nz"] == definition.bed.nz and record["datum_offset_m"] == definition.bed.datum_offset_m
        parts.append({k: np.asarray(v) for k, v in _arrays(loaded).items()})
    for key, whole in dense.items():
        tiled = np.concatenate([p[key] for p in parts], axis=0)
        assert tiled.dtype == whole.dtype and np.array_equal(tiled, whole), f"{key}: tiled bed differs from the dense bed"
    assert dense["voxel"].shape == (NY, NX, definition.bed.nz, 6)


def test_a_bad_global_datum_or_nz_or_existing_tile_is_refused_before_anything_is_written(rfid_source, dtm_path, tmp_path):
    from maple_syrup.chastre_case import compile_tile

    definition = _definition(rfid_source, dtm_path)
    bad_datum = dataclasses.replace(definition, bed=dataclasses.replace(
        definition.bed, datum_offset_m=definition.bed.datum_offset_m + 0.1))
    with pytest.raises(Plot1ImportError, match="datum offset"):
        compile_tile(bad_datum, 0, 0, 4, tmp_path / "a")
    bad_nz = dataclasses.replace(definition, bed=dataclasses.replace(definition.bed, nz=definition.bed.nz + 1))
    with pytest.raises(Plot1ImportError, match="voxel allocation"):
        compile_tile(bad_nz, 0, 0, 4, tmp_path / "b")
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    (tmp_path / "c").mkdir()
    with pytest.raises(Plot1ImportError, match="existing tile"):
        compile_tile(definition, 0, 0, 4, tmp_path / "c")
    with pytest.raises(Plot1ImportError, match="outside"):
        compile_tile(definition, 0, 0, NY + 1, tmp_path / "d")


def test_plan_only_writes_nothing_and_reports_estimates(synthetic_recipe_file, tmp_path):
    from maple_syrup.chastre_case import load_chastre_recipe, plan_only

    plan = plan_only(load_chastre_recipe(synthetic_recipe_file), tmp_path / "never")
    assert not (tmp_path / "never").exists()
    assert plan["n_bands"] == 3 and plan["interior_shape"] == [NY, NX] and plan["n_active"] == NY * NX - 2
    assert plan["dense_voxel_bytes_NOT_BUILT"] == NY * NX * plan["nz"] * 6 * 8 and plan["refuse"] is False
    assert plan["free_bytes"] > 0 and "ESTIMATES" in plan["note"]


def test_generated_case_verifies_with_the_expected_zero_outlet_topology(chastre_case_dir, rfid_source):
    from maple_syrup.chastre_case import verify_chastre_case

    verified = verify_chastre_case(chastre_case_dir, allow_maple_source_change=True)
    terrain, binding = verified.report["terrain"], verified.binding
    assert (terrain["n_pit_storage"], terrain["n_flat_storage"], terrain["n_strict_pit_storage"], terrain["n_outlets"]) == (5, 4, 1, 0)
    assert terrain["allow_flat_storage"] is True and terrain["summary"]["n_outlets"] == 0
    assert binding["grid"]["ny"] == NY and binding["grid"]["nx"] == NX
    assert [(t["row_start"], t["row_stop"]) for t in binding["tiles"]] == [(0, 4), (4, 8), (8, 10)]
    assert all(t["nz"] == binding["grid"]["nz"] and t["datum_offset_m"] == binding["grid"]["datum_offset_m"]
               for t in binding["tiles"])
    assert all(t["compile_s"] > 0 and t["process_peak_rss_kib_after"] > 0 and t["files"] for t in binding["tiles"])
    pins = binding["source_rfid"]
    assert pins["applied_forcing_sha256"] == rfid_source.binding["applied_forcing_sha256"] == binding["applied_forcing_sha256"]
    assert pins["case_identity_sha256"] == rfid_source.binding["maple_case_identity_sha256"]
    assert verified.checks["tiles"]["reloaded_and_checked"] is True
    # the hydrology fields: exact source hydrology, nodata cells are inactive with zero scale, active cells have scale > 0
    assert verified.report["hydrology"] == rfid_source.report["hydrology"]
    active, scale = verified.fields["active"], verified.fields["rainfall_scaling"]
    assert int(active.sum()) == NY * NX - 2 and np.all(scale[active] > 0.0) and np.all(scale[~active] == 0.0)
    # the case file was copied byte for byte
    src = Path(rfid_source.case_dir) / "syrup" / "forcing" / "applied_forcing.json"
    assert (chastre_case_dir / "syrup" / "forcing" / "applied_forcing.json").read_bytes() == src.read_bytes()


def test_hash_only_tile_verification_still_loads_tile_zero(chastre_case_dir):
    from maple_syrup.chastre_case import verify_chastre_case

    verified = verify_chastre_case(chastre_case_dir, allow_maple_source_change=True, reload_tiles=False)
    assert verified.checks["tiles"]["reloaded_and_checked"] is False
    assert len(verified.case.config.grain_classes.classes) == 6


def test_rainfall_scaling_matches_an_independent_nearest_centre_gap_fill(chastre_case_dir, rfid_source):
    from maple_syrup.chastre_case import verify_chastre_case

    fields = verify_chastre_case(chastre_case_dir, allow_maple_source_change=True, reload_tiles=False).fields
    src_scale, src_valid = rfid_source.fields["rainfall_scaling"], rfid_source.fields["active"]
    ns_r, ns_c = src_scale.shape
    valid = [(r, c) for r in range(ns_r) for c in range(ns_c) if src_valid[r, c]]
    active = fields["active"]
    for r in range(NY):
        for c in range(NX):
            if not active[r, c]:
                continue
            sr, sc = (2 * r + 1) * ns_r // (2 * NY), (2 * c + 1) * ns_c // (2 * NX)
            if not src_valid[sr, sc]:
                sr, sc = min(valid, key=lambda rc: ((rc[0] - sr) ** 2 + (rc[1] - sc) ** 2, rc))
            assert fields["rainfall_scaling"][r, c] == src_scale[sr, sc]
    assert set(np.unique(fields["rainfall_scaling"][active])) <= set(np.unique(src_scale[src_valid]))


def _tile_file(case, index=1):
    return next(p for p in sorted((case / "tiles" / f"tile_{index:03d}" / "processed").glob("*.npz")))


def test_verify_refuses_a_modified_tile_artifact(case_copy):
    from maple_syrup.chastre_case import verify_chastre_case

    path = _tile_file(case_copy)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(Plot1ImportError, match="bound manifest"):
        verify_chastre_case(case_copy, allow_maple_source_change=True)


def test_verify_refuses_missing_extra_or_replaced_files(case_copy):
    from maple_syrup.chastre_case import verify_chastre_case

    (case_copy / "tiles" / "tile_000" / "stray.txt").write_text("x")
    with pytest.raises(Plot1ImportError, match="bound manifest"):
        verify_chastre_case(case_copy, allow_maple_source_change=True)
    (case_copy / "tiles" / "tile_000" / "stray.txt").unlink()
    shutil.rmtree(case_copy / "tiles" / "tile_002")
    with pytest.raises(Plot1ImportError, match="missing"):
        verify_chastre_case(case_copy, allow_maple_source_change=True)


def test_verify_refuses_modified_terrain_report_fields_and_forcing(case_copy):
    from maple_syrup.chastre_case import verify_chastre_case

    for relative in ("source/mahleran_input_chastre/dtm.asc", "syrup/chastre_import_report.json", "syrup/chastre_fields.npz",
                     "syrup/forcing/applied_forcing.json"):
        path = case_copy / relative
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(Plot1ImportError, match="does not match binding"):
            verify_chastre_case(case_copy, allow_maple_source_change=True)
        path.write_bytes(original)


def test_verify_refuses_an_unbound_or_failed_directory(case_copy):
    from maple_syrup.chastre_case import verify_chastre_case

    binding = case_copy / "syrup" / "chastre_binding.json"
    text = binding.read_text()
    (case_copy / "syrup" / "FAILED.json").write_text("{}")
    with pytest.raises(Plot1ImportError, match="import failed"):
        verify_chastre_case(case_copy, allow_maple_source_change=True)
    (case_copy / "syrup" / "FAILED.json").unlink()
    binding.unlink()
    with pytest.raises(Plot1ImportError, match="not a completed"):
        verify_chastre_case(case_copy, allow_maple_source_change=True)
    binding.write_text(text.replace('"status": "ok"', '"status": "partial"'))
    with pytest.raises(Plot1ImportError, match="not a completed"):
        verify_chastre_case(case_copy, allow_maple_source_change=True)


def test_failed_generation_leaves_failed_json_partial_tiles_and_no_binding(rfid_source, synthetic_recipe_file, tmp_path, monkeypatch):
    from maple_syrup import chastre_case
    from maple_syrup.chastre_case import (
        generate_chastre_case,
        load_chastre_recipe,
        verify_chastre_case,
    )

    real = chastre_case.compile_tile

    def failing(definition, index, *args, **kwargs):
        if index == 1:
            raise RuntimeError("injected tile failure")
        return real(definition, index, *args, **kwargs)

    monkeypatch.setattr(chastre_case, "compile_tile", failing)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected tile failure"):
        generate_chastre_case(load_chastre_recipe(synthetic_recipe_file), out, source_case_dir=rfid_source.case_dir,
                              allow_maple_source_change=True)
    failure = json.loads((out / "syrup" / "FAILED.json").read_text())
    assert failure["status"] == "failed" and "injected tile failure" in failure["error"]
    assert not (out / "syrup" / "chastre_binding.json").exists()
    assert (out / "tiles" / "tile_000" / "processed").is_dir()  # the partial tile is preserved
    with pytest.raises(Plot1ImportError):
        verify_chastre_case(out, allow_maple_source_change=True)


def test_wrong_pins_and_existing_output_are_refused(rfid_source, synthetic_recipe_file, dtm_path, tmp_path):
    import yaml

    from maple_syrup.chastre_case import generate_chastre_case, load_chastre_recipe

    recipe = load_chastre_recipe(synthetic_recipe_file)
    with pytest.raises(Plot1ImportError, match="refusing to write into existing"):
        generate_chastre_case(recipe, tmp_path, source_case_dir=rfid_source.case_dir, allow_maple_source_change=True)
    raw = recipe_dict(dtm_path, rfid_source.case_dir, n_active=NY * NX - 2, n_sinks=5, nz=recipe.raw["terrain"]["expected_nz"],
                      sha256="0" * 64)
    bad = tmp_path / "bad_recipe" / "recipe.yaml"
    bad.parent.mkdir()
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(Plot1ImportError, match="terrain sha256"):
        generate_chastre_case(load_chastre_recipe(bad), tmp_path / "out_bad_pin", source_case_dir=rfid_source.case_dir,
                              allow_maple_source_change=True)
    assert not (tmp_path / "out_bad_pin").exists()  # refused before anything was written
    # a wrong expected count is found by the audit: the directory exists, FAILED.json explains, no binding
    raw = recipe_dict(dtm_path, rfid_source.case_dir, n_active=NY * NX, n_sinks=5, nz=recipe.raw["terrain"]["expected_nz"])
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(Plot1ImportError, match="active cells"):
        generate_chastre_case(load_chastre_recipe(bad), tmp_path / "out_bad_count", source_case_dir=rfid_source.case_dir,
                              allow_maple_source_change=True)
    assert (tmp_path / "out_bad_count" / "syrup" / "FAILED.json").is_file()
    assert not (tmp_path / "out_bad_count" / "syrup" / "chastre_binding.json").exists()


def test_recipe_validation_is_strict(synthetic_recipe_file, tmp_path):
    import yaml

    from maple_syrup.chastre_case import load_chastre_recipe

    raw = yaml.safe_load(synthetic_recipe_file.read_text())
    for mutate in (lambda r: r.update(extra=1), lambda r: r["terrain"].update(nodata_value=0),
                   lambda r: r["tiles"].update(rows=0), lambda r: r["rainfall_scaling"].update(resize="bilinear"),
                   lambda r: r["terrain"].update(sha256="abc"), lambda r: r.update(schema="x"),
                   lambda r: r["terrain"].update(sha256="g" * 64), lambda r: r["terrain"].update(sha256="A" * 64),
                   lambda r: r["terrain"].update(expected_nz=0),
                   lambda r: r["terrain"].update(cellsize_m=float("nan")), lambda r: r["terrain"].update(cellsize_m=float("inf")),
                   lambda r: r["terrain"].update(nodata_value=float("nan")),
                   lambda r: r["terrain"].update(nodata_value=float("-inf")),
                   lambda r: r["tiles"].update(max_rss_gib=float("nan")), lambda r: r["tiles"].update(max_rss_gib=float("inf")),
                   lambda r: r["tiles"].update(max_rss_gib=0)):
        broken = json.loads(json.dumps(raw))
        mutate(broken)
        path = tmp_path / "r.yaml"
        path.write_text(yaml.safe_dump(broken))
        with pytest.raises(Plot1ImportError):
            load_chastre_recipe(path)


# --------------------------------------------------------------------------------------------------------------------
# adapter, hydrology on the tiled case, runtime guard
# --------------------------------------------------------------------------------------------------------------------
def _runner(name, verified, end_s=30.0):
    import compare_cases as cc
    import compare_plot1 as cmp

    import maple_syrup
    from maple_syrup.chastre_case import chastre_bed_digest
    from maple_syrup.rfid_case import rfid_inputs

    args = SimpleNamespace(bisection_iterations=64, newton_max_iterations=50, cuda_mode="auto", report_every_s=10.0)
    run, _prep, inputs, _prov = cc.build_syrup_runner(name, 1.0, verified, args, rfid_inputs)
    dirs = (Path(maple_syrup.__file__).parent, Path(verified.maple_dependency.package_dir))
    guard = cmp.RunGuard(inputs, dirs, name, bed_digest=chastre_bed_digest)
    return cc, run, guard, inputs


def test_adapter_holds_no_bed_and_feeds_the_unchanged_input_factory(chastre_case_dir):
    from maple_syrup.chastre_case import verify_chastre_case
    from maple_syrup.rfid_case import rfid_inputs

    verified = verify_chastre_case(chastre_case_dir, allow_maple_source_change=True, reload_tiles=False)
    case = verified.case
    for name in ("voxel_column", "active_layer", "sediment_availability", "sediment_ledger", "topography_result",
                 "committed_topography", "expected_total_mass_by_class_kg"):
        assert not hasattr(case, name), f"the adapter must not carry {name}"
    assert (case.config.geometry.ny, case.config.geometry.nx, case.config.geometry.dx_m) == (NY, NX, 1.0)
    assert case.water.depth_m.shape == (NY, NX) and not case.water.depth_m.any()
    inputs = rfid_inputs(verified, "numpy", with_geometry=False)
    assert inputs.case is case and (inputs.ny, inputs.nx, inputs.n_classes) == (NY, NX, 6)
    assert int(np.sum(inputs.graph.pit_storage)) == 5 and int(inputs.graph.outlet.sum()) == 0
    assert inputs.graph.policy.endswith("flat_storage=True")
    assert case.persisted_digest() == verified.binding["tiles_digest"]


@pytest.mark.parametrize("name", ["bisection_numba", "newton_numba"])
def test_zero_outlet_water_storm_closes_its_budget_on_the_tiled_case(chastre_case_dir, name):
    from maple_syrup.chastre_case import verify_chastre_case

    verified = verify_chastre_case(chastre_case_dir, allow_maple_source_change=True, reload_tiles=False)
    cc, run, guard, _inputs = _runner(name, verified)
    out = cc.timed_validated(run, guard, 30.0)["checked"]
    assert out["budget"]["closed"] is True and out["host"]["export"] == 0.0 and out["host"]["peak_q"] == 0.0
    assert out["guard"]["bed_unchanged"] is True and "persisted" in out["guard"]["bed_digest_kind"]
    assert out["guard"]["bed_digest"] == verified.binding["tiles_digest"]
    assert out["host"]["rejected"] == 0 or out["host"]["accepted"] > 0


def test_runtime_guard_detects_a_modified_persisted_tile(case_copy):
    from maple_syrup.chastre_case import verify_chastre_case

    verified = verify_chastre_case(case_copy, allow_maple_source_change=True, reload_tiles=False)
    _cc, run, guard, _inputs = _runner("bisection_numba", verified)
    raw = run(30.0, [])
    path = _tile_file(case_copy, 0)
    data = bytearray(path.read_bytes())
    data[0] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="bed changed"):
        guard.validate(raw, 30.0)


def test_gpu_forms_close_the_budget_and_track_the_numba_solver(chastre_case_dir, gpu):
    from maple_syrup.chastre_case import verify_chastre_case

    verified = verify_chastre_case(chastre_case_dir, allow_maple_source_change=True, reload_tiles=False)
    finals = {}
    for name in ("bisection_numba", "bisection_cuda", "newton_cuda"):
        cc, run, guard, _inputs = _runner(name, verified)
        out = cc.timed_validated(run, guard, 30.0)["checked"]
        assert out["budget"]["closed"] is True and out["host"]["export"] == 0.0
        finals[name] = np.asarray(out["finals"]["depth_m"])
    assert np.allclose(finals["bisection_cuda"], finals["bisection_numba"], rtol=1e-9, atol=1e-12)
