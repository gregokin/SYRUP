"""XML / hash / vegetation / composition ownership of the legacy case adapters, plus the real-case smoke tests (skipped when
the generated cases or Numba are absent). Written without being run; Codex executes them."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maple_syrup import legacy_case as C
from maple_syrup.sediment_physics import LEGACY_CLASS_RADII_M, plot1_sediment_parameters

ROOT = Path(__file__).resolve().parents[2]
PLOT1_XML = Path("/home/okin/MAHLERAN/mahleran_input.xml")
RFID_XML = Path("/home/okin/MAHLERAN/Input/RFID_2014/mahleran_input.xml")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grain_classes(order=None, density=2650.0, scale=1.0):
    ids = order or [f"phi_{k}" for k in range(1, 7)]
    return SimpleNamespace(classes=[SimpleNamespace(class_id=c, diameter_m=scale * 2.0 * r, particle_density_kg_m3=density)
                                    for c, r in zip(ids, LEGACY_CLASS_RADII_M, strict=True)])


needs_xml = pytest.mark.skipif(not (PLOT1_XML.is_file() and RFID_XML.is_file()), reason="MAHLERAN reference XML not present")


@needs_xml
def test_xml_parameters_are_read_from_the_file_and_recorded_against_plot1():
    params, record = C.sediment_parameters_from_xml(PLOT1_XML, sha(PLOT1_XML), grain_classes())
    assert params.summary() == plot1_sediment_parameters().summary()  # the root XML IS the Plot 1 constant set
    assert record["differs_from_transcribed_plot1_constants"] == []
    rfid, rec = C.sediment_parameters_from_xml(RFID_XML, sha(RFID_XML), grain_classes())
    assert rec["xml_sha256"] == sha(RFID_XML) and rec["ke_model_type_xml"] == "2"
    assert rec["sediment_routing_selector_in_xml"] is None  # absent in the RFID XML: the replay is Crank-Nicolson
    assert isinstance(rec["differs_from_transcribed_plot1_constants"], list)  # recorded, whatever it is
    assert rfid.n_classes == 6


@needs_xml
def test_xml_refusals_hash_class_order_diameter_density(tmp_path):
    with pytest.raises(C.LegacyCaseError, match="hashes to"):
        C.sediment_parameters_from_xml(PLOT1_XML, "0" * 64, grain_classes())
    swapped = [f"phi_{k}" for k in (2, 1, 3, 4, 5, 6)]
    with pytest.raises(C.LegacyCaseError, match="in order"):
        C.sediment_parameters_from_xml(PLOT1_XML, sha(PLOT1_XML), grain_classes(order=swapped))
    with pytest.raises(C.LegacyCaseError, match="radii"):
        C.sediment_parameters_from_xml(PLOT1_XML, sha(PLOT1_XML), grain_classes(scale=1.01))
    with pytest.raises(C.LegacyCaseError, match="densities"):
        C.sediment_parameters_from_xml(PLOT1_XML, sha(PLOT1_XML), grain_classes(density=2700.0))


@needs_xml
def test_a_changed_xml_value_is_used_and_reported_never_replaced_by_plot1_constants(tmp_path):
    text = PLOT1_XML.read_text(encoding="latin-1")
    assert "<phi_1>4.25E-5</phi_1>" in text
    changed = tmp_path / "mahleran_input.xml"
    changed.write_text(text.replace("<phi_1>4.25E-5</phi_1>", "<phi_1>5.25E-5</phi_1>", 1), encoding="latin-1")
    params, record = C.sediment_parameters_from_xml(changed, sha(changed), grain_classes())
    assert params.raindrop_a[0] == 5.25e-5
    assert "raindrop_a" in record["differs_from_transcribed_plot1_constants"]


@needs_xml
def test_time_step_other_than_one_second_is_refused(tmp_path):
    text = PLOT1_XML.read_text(encoding="latin-1")
    assert '<time_step value="1.0" />' in text
    changed = tmp_path / "x.xml"
    changed.write_text(text.replace('<time_step value="1.0" />', '<time_step value="2.0" />', 1), encoding="latin-1")
    with pytest.raises(C.LegacyCaseError, match="time_step"):
        C.sediment_parameters_from_xml(changed, sha(changed), grain_classes())


def _veg_case(tmp_path, body_interior, *, name="veg.asc", xml_name="veg.asc"):
    """A tiny RFID-like source: 4 x 4 north-first raster (ring + 2 x 2 interior) and its XML/report."""
    rows = [[-9999] * 4, [-9999, *body_interior[0], -9999], [-9999, *body_interior[1], -9999], [-9999] * 4]
    case = tmp_path / "case"
    (case / "source").mkdir(parents=True)
    raster = case / "source" / name
    raster.write_text("ncols 4\nnrows 4\nxllcorner 0\nyllcorner 0\ncellsize 0.1\nnodata_value -9999\n"
                      + "\n".join(str(v) for r in rows for v in r) + "\n")
    xml = tmp_path / "in.xml"
    xml.write_text(f'<root><vegetation-cover_map value="{xml_name}" /></root>')
    report = {"mahleran": {"xml_path": str(xml), "xml_sha256": sha(xml)},
              "staged_sources": {"veg.asc": {"relpath": f"source/{name}", "sha256": sha(raster)}}}
    return case, report


def test_vegetation_is_read_from_the_bound_raster_and_flipped_to_maple_order(tmp_path):
    case, report = _veg_case(tmp_path, [[10, 10], [10, 10]])
    active = np.ones((2, 2), dtype=bool)
    veg, record = C.uniform_vegetation_fraction(case, report, active)
    assert veg.shape == (2, 2) and np.all(veg == 0.1) and record["value_percent"] == 10.0
    # a nonuniform value on an INACTIVE cell is ignored (north-first row 0 -> MAPLE row 1)
    case, report = _veg_case(tmp_path / "b", [[10, 77], [10, 10]])
    act = np.array([[True, True], [True, False]])  # MAPLE south-first: (1, 1) is the north-east cell holding 77
    veg, _ = C.uniform_vegetation_fraction(case, report, act)
    assert np.all(veg == 0.1)


def test_chastre_style_target_shape_broadcasts_the_verified_source_value_without_resampling(tmp_path):
    """The SOURCE raster/mask is 2 x 2; the target grid is 5 x 7 (the Chastre path): validated at source, broadcast uniformly."""
    case, report = _veg_case(tmp_path, [[10, 10], [10, 10]])
    source_active = np.ones((2, 2), dtype=bool)
    veg, record = C.uniform_vegetation_fraction(case, report, source_active, (5, 7))
    assert veg.shape == (5, 7) and np.all(veg == 0.1)
    assert record["broadcast_to_target"] and record["source_shape"] == [2, 2] and record["target_shape"] == [5, 7]
    # a non-uniform SOURCE is still refused (nothing is resampled or invented for the larger grid)
    case, report = _veg_case(tmp_path / "x", [[10, 20], [10, 10]])
    with pytest.raises(C.LegacyCaseError, match="uniform"):
        C.uniform_vegetation_fraction(case, report, source_active, (5, 7))
    # the source mask must match the source raster, not the target grid
    case, report = _veg_case(tmp_path / "y", [[10, 10], [10, 10]])
    with pytest.raises(C.LegacyCaseError, match="interior"):
        C.uniform_vegetation_fraction(case, report, np.ones((5, 7), dtype=bool), (5, 7))


def test_vegetation_refusals(tmp_path):
    case, report = _veg_case(tmp_path / "a", [[10, 20], [10, 10]])
    with pytest.raises(C.LegacyCaseError, match="uniform"):
        C.uniform_vegetation_fraction(case, report, np.ones((2, 2), bool))
    case, report = _veg_case(tmp_path / "b", [[150, 150], [150, 150]])
    with pytest.raises(C.LegacyCaseError, match="outside"):
        C.uniform_vegetation_fraction(case, report, np.ones((2, 2), bool))
    case, report = _veg_case(tmp_path / "c", [[10, 10], [10, 10]])
    (case / "source" / "veg.asc").write_text((case / "source" / "veg.asc").read_text().replace("10", "11"))
    with pytest.raises(C.LegacyCaseError, match="modified"):
        C.uniform_vegetation_fraction(case, report, np.ones((2, 2), bool))
    case, report = _veg_case(tmp_path / "d", [[10, 10], [10, 10]])
    report["staged_sources"] = {}
    with pytest.raises(C.LegacyCaseError, match="not among"):
        C.uniform_vegetation_fraction(case, report, np.ones((2, 2), bool))
    case, report = _veg_case(tmp_path / "e", [[10, 10], [10, 10]])
    Path(report["mahleran"]["xml_path"]).write_text('<root><vegetation-cover_map value="other.asc" /></root>')
    with pytest.raises(C.LegacyCaseError, match="XML differs"):
        C.uniform_vegetation_fraction(case, report, np.ones((2, 2), bool))


def test_composition_check_compares_the_actual_active_layer():
    mass = np.broadcast_to(np.array([0.0, 0.0, 0.0, 0.092, 0.908, 0.0]) * 2.5, (3, 4, 6)).copy()
    out = C._check_composition(mass, [0.0, 0.0, 0.0, 0.092, 0.908, 0.0], "t")
    assert out["checked_cells"] == 12 and out["max_abs_fraction_difference"] < 1e-12
    mass[1, 1] = np.array([0.0, 0.0, 0.0, 0.1, 0.9, 0.0])
    with pytest.raises(C.LegacyCaseError, match="differ"):
        C._check_composition(mass, [0.0, 0.0, 0.0, 0.092, 0.908, 0.0], "t")
    mass[1, 1] = 0.0
    with pytest.raises(C.LegacyCaseError, match="empty"):
        C._check_composition(mass, [0.0, 0.0, 0.0, 0.092, 0.908, 0.0], "t")


def test_unknown_case_kind_is_refused():
    with pytest.raises(C.LegacyCaseError, match="unknown case kind"):
        C.legacy_case_for("nope", ".")


# --- real-case smoke tests (skipped when the generated cases / Numba are absent) -------------------------------------
needs_numba = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="Numba not installed")


def _run(kind, case, tmp_path, *extra):
    from maple_syrup import legacy_driver as D

    out = tmp_path / "out"
    code = D.main(["--case-kind", kind, "--case", str(case), "--output", str(out), "--allow-maple-source-change",
                   "--progress-every-s", "0", *extra])
    return code, out


@needs_numba
@pytest.mark.skipif(not (ROOT / "outputs/rfid/case/syrup/rfid_binding.json").is_file(), reason="outputs/rfid/case absent")
def test_rfid_600s_run_crosses_sediment_onset_and_publishes_atomically(tmp_path):
    """600 s: the 30 s smoke was dry (ponding onset estimated ~269 s, an estimate not a measurement)."""
    code, out = _run("rfid", ROOT / "outputs/rfid/case", tmp_path, "--end-s", "600", "--snapshot-times", "300,600")
    assert code == 0 and out.is_dir() and not (tmp_path / "out.partial").exists()
    data = np.load(out / "legacy_ledger.npz")
    summary = json.loads((out / "legacy_summary.json").read_text())
    assert data["ledger"].shape[0] == 600 and summary["identity"]["worst_relative_residual"] < 1e-10
    assert summary["case"]["vegetation"]["value_percent"] >= 0.0
    assert summary["network"]["n_terminal_storage"] == 26 and summary["limitations"]
    assert os.path.isfile(out / "legacy_snapshots.npz")
    totals = summary["totals"]  # non-vacuous: pickup, deposition and a mobile inventory (or clipping if the pool stays 0)
    assert totals["pickup_kg"] > 0.0 and totals["deposition_active_kg"] > 0.0  # `totals` holds class-summed SCALARS
    assert sum(summary["final_mobile_by_class_kg"]) > 0.0 or totals["effective_clip_source_kg"] > 0.0
    onset = summary["onset"]
    assert onset["first_positive_pickup_s"] is not None and onset["wet_law_cell_class_steps"] > 0
    assert summary["water"]["whole_storm_budget"]["closed"] is True
    assert float(np.asarray(data["cumulative_detachment_kg"]).sum()) > 0.0
    print("RFID first positive pickup s:", onset["first_positive_pickup_s"], "law cell-class-steps:",
          onset["wet_law_cell_class_steps"])


@needs_numba
@pytest.mark.skipif(not (ROOT / "outputs/rfid/case/syrup/rfid_binding.json").is_file(), reason="outputs/rfid/case absent")
def test_rfid_30s_adaptation_smoke_is_dry_and_is_not_a_sediment_validation(tmp_path):
    code, out = _run("rfid", ROOT / "outputs/rfid/case", tmp_path, "--end-s", "30")
    assert code == 0
    summary = json.loads((out / "legacy_summary.json").read_text())
    assert summary["identity"]["worst_relative_residual"] < 1e-10 and summary["water"]["whole_storm_budget"]["closed"]


def test_whole_storm_water_guard_uses_the_canonical_bound_and_refuses_extra_water():
    from maple_syrup import legacy_driver as D
    from maple_syrup.conservation import volume_roundoff_bound_m3

    v = {"surface_initial": 0.0, "soil_initial": 1.0, "rain": 10.0, "intake": 4.0, "saturation_return": 1.0,
         "drainage": 1.0, "export": 2.0}
    v["surface_final"] = v["surface_initial"] + v["rain"] - v["intake"] + v["saturation_return"] - v["export"]
    v["soil_final"] = v["soil_initial"] + v["intake"] - v["drainage"] - v["saturation_return"]
    out = D.water_closure(v, 3, 4, 100, 10.0)
    assert out["closed"] and out["bound_m3"] == volume_roundoff_bound_m3(4 * 3 * 4 * 100 + 7, max(abs(x) for x in v.values()))
    assert out["rainfall_bound_m3"] == volume_roundoff_bound_m3(4 * 3 * 4 * 100 + 7, 10.0)
    for key in ("surface_final", "soil_final", "drainage", "export"):  # unaccounted water in any store/flux
        bad = dict(v)
        bad[key] += 1.0e-3
        with pytest.raises(D.DriverError, match="budget residual"):
            D.water_closure(bad, 3, 4, 100, 10.0)
    with pytest.raises(D.DriverError, match="rainfall_integral"):
        D.water_closure(v, 3, 4, 100, 10.5)  # the independent schedule integral disagrees with the accumulated rain
    bad = dict(v)
    bad["rain"] = float("nan")
    with pytest.raises(D.DriverError, match="not finite"):
        D.water_closure(bad, 3, 4, 100, 10.0)


@needs_numba
@pytest.mark.skipif(not (ROOT / "outputs/plot1/syrup/plot1_binding.json").is_file(), reason="outputs/plot1 absent")
def test_existing_plot1_replay_and_native_engine_agree_on_a_short_window(tmp_path, record_property):
    """Same hydrology feeds the accepted replay (legacy_physics_step + legacy_transport_step) and the native engine."""
    from dataclasses import replace

    from maple_syrup import hydrology_numba as hn
    from maple_syrup import legacy_native as N
    from maple_syrup import legacy_transport as L
    from maple_syrup.legacy_native_numba import StepEngine, WetLawRunner
    from maple_syrup.legacy_physics_numba import (
        legacy_physics_step,
        prepare_legacy_physics,
    )
    from maple_syrup.sediment_physics import recession_velocity
    from maple_syrup.storm import StormControl

    applied = ROOT / "outputs/phase7/mahleran_reference_audit/applied_rainfall.csv"  # the default old-CLI forcing
    if not applied.is_file():
        pytest.skip("applied rainfall CSV of the accepted Plot 1 reference is absent")
    case = C.legacy_case_for("plot1", ROOT / "outputs/plot1", allow_maple_source_change=True, end_s=40.0,
                             applied_rainfall=applied)
    ny, nx = case.shape
    nc = case.n_classes
    n = ny * nx
    ctx = prepare_legacy_physics(case.sediment, case.grid, case.vegetation, case.holdings_kg)
    net_new, net_old = N.native_network(case.graph), L.legacy_network(case.graph)
    engine = StepEngine(net_new, WetLawRunner(ctx), dt=1.0)
    control = StormControl(max_dt_s=1.0, min_dt_s=1.0, max_retries=1, implementation="numba").validated()
    hctx = hn.prepare_hydrology(case.graph, case.column)
    storm, prev_depth = case.initial_storm, np.array(case.initial_storm.depth_m, copy=True)
    v_prev = np.zeros((ny, nx, nc))
    M, Q, Qin = np.zeros((n, nc)), np.zeros((n, nc)), np.zeros((n, nc))
    cum_det, cum_dep, cum_clip = np.zeros((n, nc)), np.zeros((n, nc)), np.zeros((n, nc))
    rate = np.empty((ny, nx))
    pairs = (("pickup_kg", lambda s: s.detachment_rate_kg_s.sum(0)), ("deposition_active_kg", lambda s: s.deposition_rate_kg_s.sum(0)),
             ("deposition_ring_kg", lambda s: s.ring_deposition_rate_kg_s), ("effective_clip_source_kg", lambda s: s.clipping_source_kg.sum(0)),
             ("old_mobile_kg", lambda s: s.mobile_before_kg.sum(0)), ("new_mobile_kg", lambda s: s.mobile_after_kg.sum(0)),
             ("cn_export_kg", lambda s: s.cn_export_kg), ("outlet_flux_kg_s", lambda s: s.outlet_flux_kg_s))
    for step_index in range(40):
        t = float(step_index)
        case.field.apply(case.schedule.rate_after_m_per_s(t), out=rate)
        hydro = hn.prepared_coupled_step(hctx, rate, storm, 1.0, control)
        route = hydro.route
        physics = legacy_physics_step(ctx, prev_depth, route.velocity_m_s, rate, v_prev, 1.0)
        v_used = np.where(physics.law_applies, physics.sediment_velocity_m_s, recession_velocity(v_prev, 1.0))
        v_used = np.where(np.asarray(case.graph.active)[..., None], v_used, 0.0)
        old = L.legacy_transport_step(net_old, physics.requested_pickup_kg / 1.0, physics.deposition_rate_per_m,
                                      physics.law_applies, physics.regime, v_used, M, Q, Qin, 1.0)
        new = engine.step(prev_depth, route.velocity_m_s, rate)
        for name, getter in pairs:  # declared sediment bound 2e-11 / 1e-14 (the ledger sums use a different summation order)
            np.testing.assert_allclose(new.ledger_row[engine_col(name)], getter(old), rtol=2e-11, atol=1e-14, err_msg=name)
        cum_det += old.detachment_rate_kg_s * 1.0
        cum_dep += old.deposition_rate_kg_s * 1.0
        cum_clip += old.clipping_source_kg
        M, Q, Qin = old.mobile_after_kg, old.flux_after_kg_s, old.inflow_after_kg_s
        v_prev = v_used
        prev_depth = np.array(route.depth_m, copy=True)
        storm = replace(hydro.state, t_s=t + 1.0)
    for name, mine, theirs in (("mobile", engine.mobile, M), ("cum_det", engine.cum_det, cum_det),
                               ("cum_dep", engine.cum_dep, cum_dep), ("cum_clip", engine.cum_clip, cum_clip)):
        np.testing.assert_allclose(mine, theirs, rtol=2e-11, atol=1e-14, err_msg=name)
        record_property(f"bitwise_{name}", bool(np.array_equal(mine, theirs)))  # same arithmetic: bitwise is expected
    assert engine.total_counts[N.C_TERMINAL] == 0  # Plot 1 has no terminal pits
    # not vacuous: the accepted applied-forcing reference first picks up at 2 s and has ~7.4 kg picked up through 40 s
    assert cum_det.sum() > 0.0 and cum_dep.sum() > 0.0 and engine.cum_det.sum() > 0.0 and engine.cum_dep.sum() > 0.0
    record_property("plot1_40s_cumulative_pickup_kg", float(engine.cum_det.sum()))


def engine_col(name):
    from maple_syrup.legacy_native_numba import LEDGER_COLUMNS

    return LEDGER_COLUMNS.index(name)


@needs_numba
@pytest.mark.skipif(not os.environ.get("SYRUP_RUN_CHASTRE_PILOT"), reason="set SYRUP_RUN_CHASTRE_PILOT=1 (needs outputs/chastre/case_v2)")
def test_chastre_600s_pilot_is_nonvacuous(tmp_path):
    """Expensive: 60 s in-process warm-up (JIT, reported separately) then 600 s. Ponding onset is estimated near 426 s
    (an estimate); the assertions below are the actual qualification."""
    code, out = _run("chastre", ROOT / "outputs/chastre/case_v2", tmp_path, "--end-s", "600", "--warmup-s", "60",
                     "--hash-only-tile-verify", "--snapshot-times", "300,600")
    assert code == 0
    summary = json.loads((out / "legacy_summary.json").read_text())
    assert summary["network"]["n_outlets"] == 0 and summary["totals"]["cn_export_kg"] == 0.0
    assert summary["totals"]["pickup_kg"] > 0.0 and summary["totals"]["deposition_active_kg"] > 0.0
    assert sum(summary["final_mobile_by_class_kg"]) > 0.0 or summary["totals"]["effective_clip_source_kg"] > 0.0
    assert summary["onset"]["first_positive_pickup_s"] is not None and summary["onset"]["wet_law_cell_class_steps"] > 0
    assert summary["water"]["whole_storm_budget"]["closed"] is True
    assert summary["artifact_pins"]["checked_unchanged_before_publish"] and summary["chastre_tiles"]["performed"] is False
    assert summary["performance"]["warmup"]["model_s"] == 60.0 and summary["performance"]["first_step_including_jit"]
    assert summary["artifact_pins"]["checked_unchanged_before_publish"] and summary["chastre_tiles"]["performed"] is False
