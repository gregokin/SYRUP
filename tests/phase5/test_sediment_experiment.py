"""Phase 5b Plot 1 runner (`maple_syrup.sediment_experiment`) on the actual
MAPLE case. A fresh case is generated once per module with the Phase 2
importer; runs are short (`end_s` 30-180 s) so the module stays fast. The
full storm and the dt / grid studies are CLI runs for Codex
(docs/phase5/integration.md). Nothing here compares with executed
MAHLERAN Fortran."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup import routing_numba, sediment_experiment
from maple_syrup.rainfall import parse_legacy_rainfall_file
from maple_syrup.sediment_event import SedimentEventError, sediment_hydrograph_columns
from maple_syrup.sediment_experiment import (
    FINAL_NAME,
    HYDROGRAPH_CSV,
    HYDROGRAPH_NPZ,
    RESOLVED_CONVENTIONS,
    SNAPSHOT_NAME,
    SUMMARY_NAME,
    main,
    plot1_sediment_parameters_from_xml,
    run_plot1_sediment_event,
)
from maple_syrup.sediment_physics import plot1_sediment_parameters
from maple_syrup.storm import HYDROGRAPH_COLUMNS

REPO = Path(__file__).resolve().parents[2]
RECIPE_PATH = REPO / "cases" / "plot1" / "recipe.yaml"
MAHLERAN_ROOT = Path(os.environ.get("MAPLE_SYRUP_MAHLERAN_ROOT", "/home/okin/MAHLERAN"))
PLOT1_DIR = MAHLERAN_ROOT / "Input" / "input_p1"
NUMBA_SKIP = pytest.mark.skipif(not routing_numba.numba_available(),
                                reason="Numba not installed (optional extra maple-syrup[numba]); compiled sweep "
                                       "not exercised, no claim made")
SHORT_END_S = 180.0
SHORT_CADENCE_S = 60.0


@pytest.fixture(scope="module")
def plot1_case(tmp_path_factory):
    if not (MAHLERAN_ROOT / "mahleran_input.xml").is_file() or not PLOT1_DIR.is_dir():
        pytest.skip(f"MAHLERAN reference not available at {MAHLERAN_ROOT}")
    from maple_syrup.case_import import generate_plot1_case, load_recipe

    out = tmp_path_factory.mktemp("plot1_sediment_case") / "case"
    generate_plot1_case(load_recipe(RECIPE_PATH, mahleran_root=MAHLERAN_ROOT), out)
    return out


@pytest.fixture(scope="module")
def short_run(plot1_case, tmp_path_factory):
    out = tmp_path_factory.mktemp("plot1_sediment_run") / "event"
    run = run_plot1_sediment_event(plot1_case, out, max_dt_s=1.0, end_s=SHORT_END_S, implementation="array",
                                   report_every_s=SHORT_CADENCE_S)
    return run, out


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def within(values, tolerance):
    """Elementwise |values| <= tolerance for vector tolerances."""
    values, tolerance = np.asarray(values, dtype=np.float64), np.asarray(tolerance, dtype=np.float64)
    ok = np.abs(values) <= tolerance
    assert np.all(ok), f"residual {values} exceeds tolerance {tolerance}"


def test_xml_parameters_bind_to_the_original_xml_and_the_maple_classes(plot1_case):
    from maple.case_tools.compilers.case_compiler import load_compiled_case

    case = load_compiled_case(plot1_case)
    params, record = plot1_sediment_parameters_from_xml(MAHLERAN_ROOT / "mahleran_input.xml", case.config.grain_classes)
    assert params.summary() == plot1_sediment_parameters(**RESOLVED_CONVENTIONS).summary()
    assert record["xml_sha256"] == _sha(MAHLERAN_ROOT / "mahleran_input.xml") and record["ke_model_type_xml"] == "2"
    assert params.ke_vegetation_form == "legacy_literal" and params.distance_convention == "legacy_literal"
    assert params.reference_interval_s == 1.0 and params.particle_density_kg_m3 == 2650.0
    np.testing.assert_array_equal(params.diameter_m, [c.diameter_m for c in case.config.grain_classes.classes])


def test_short_event_budgets_closure_and_status(short_run, plot1_case):
    run, _ = short_run
    s = run.summary
    b, t, sed = s["budget"], s["time"], s["sediment"]
    schedule = parse_legacy_rainfall_file(plot1_case / "syrup" / "rainfall" / "p1_01_08_06.dat")
    assert b["rain_m3"] == pytest.approx(schedule.depth_m(0.0, SHORT_END_S) * 300.0, rel=1e-12)
    total = b["surface_final_m3"] + b["soil_final_m3"] + b["drainage_m3"] + b["export_m3"]
    assert total == pytest.approx(b["surface_initial_m3"] + b["soil_initial_m3"] + b["rain_m3"], rel=1e-12)
    for key in ("water_residual_m3", "surface_residual_m3", "soil_residual_m3"):
        assert abs(b[key]) <= b["tolerance_m3"] < 1e-7
    assert (t["n_accepted_steps"], t["end_s"], t["n_rejected_attempts"], t["n_maple_water_calls"]) == (180, 180.0, 0, 360)
    assert "NOT an event completion" in s["status"] and "no dry reset" in s["status"] and "no restart" in s["status"]
    closure = sed["closure"]
    assert closure["closed"] and closure["request_reconciled"]
    within(closure["residual_kg"], closure["tolerance_kg"])
    totals = sed["totals"]
    assert totals["requested_pickup"] > 0.0 and 0.0 < totals["actual_pickup"] <= totals["requested_pickup"]
    assert totals["deposition_actual"] > 0.0 and totals["export_actual"] >= 0.0
    within(sed["by_class"]["deposition_unmet"], sed["unmet_tolerance_by_class_kg"])
    assert sed["peak_mobile_kg"] >= sed["final_mobile_kg"] >= 0.0 and 0.0 <= sed["time_of_peak_mobile_s"] <= SHORT_END_S
    assert sed["regime_cell_class_steps"]["diffuse"] > 0
    assert s["routing"]["implementation"] == "array" and s["provenance"]["implementation"] == "array"
    assert s["domain"]["active_cells"] == 1200 and s["domain"]["final_graph"]["n_outlets"] == 10
    assert s["commits"]["n_commits"] >= 1 and s["commits"]["final_commit_count"] == s["commits"]["n_commits"]
    assert 60.0 <= s["commits"]["last_commit_time_s"] <= SHORT_END_S and s["commits"]["trigger_spec"]["commit_interval_s"] == 60.0
    for entry in s["commits"]["log"]:
        assert entry["depth_unchanged"] and entry["displaced_water_volume_m3"] == 0.0
        assert entry["n_outlets"] == 10 and isinstance(entry["graph_rebound"], bool)
    channel = np.asarray(s["commits"]["ledger_water_channel_totals_kg"])
    by_class = {k: np.asarray(v) for k, v in sed["by_class"].items()}
    within(channel - (by_class["deposition_actual"] - by_class["actual_pickup"]),
           np.asarray(closure["tolerance_kg"]) + 1e-12)
    assert s["parameters"]["sediment"]["matches_transcribed_plot1_constants"] is True
    assert s["parameters"]["sediment"]["parameters"]["ke_vegetation_form"] == "legacy_literal"
    assert "2.12" in sed["morphology_note"]
    # morphology from actual inventories, separate from the water class exchange
    m = sed["morphology"]
    assert sed["net_erosion_kg"] == m["net_erosion_kg"] and sed["net_deposition_kg"] == m["net_deposition_kg"]
    assert m["net_erosion_kg"] >= 0.0 and m["net_deposition_kg"] >= 0.0 and "actual" in m["basis"]
    assert m["net_bed_change_kg"] == pytest.approx(sum(closure["net_bed_change_kg"]), rel=1e-9, abs=1e-12)
    assert m["class_sorting_exchange_kg"] >= 0.0 and m["non_water_bed_change_abs_kg"] >= 0.0


def test_override_resolved_config_hash_and_provenance(short_run, plot1_case):
    run, out = short_run
    on_disk = json.loads((out / SUMMARY_NAME).read_text())
    override = on_disk["override"]
    assert override["run_depth_update_rule"] == "constant_depth"
    assert override["case_water_coupling"]["depth_update_rule"] == "constant_free_surface"
    assert override["case_water_coupling"]["enabled"] is False and override["case_water_coupling"]["event_kind"] == "aeolian"
    assert override["adapter_name"] == "maple_syrup/phase5" and "no scheduler wind tables" in override["commit_path"]
    rc = on_disk["resolved_config"]
    assert rc["syrup"]["override"] == override and rc["maple_config"]["water_coupling"]["depth_update_rule"] == "constant_free_surface"
    assert rc["syrup"]["conventions"] == RESOLVED_CONVENTIONS and rc["syrup"]["end_s"] == SHORT_END_S
    recomputed = hashlib.sha256(json.dumps(rc, sort_keys=True).encode("utf-8")).hexdigest()
    assert on_disk["resolved_config_sha256"] == recomputed == run.summary["resolved_config_sha256"]
    src = on_disk["sources"]["mahleran_xml"]
    assert src["sha256_before"] == src["sha256_after"] == _sha(MAHLERAN_ROOT / "mahleran_input.xml") and src["stable"]
    binding = json.loads((plot1_case / "syrup" / "plot1_binding.json").read_text())
    assert on_disk["case"]["maple_case_identity_sha256"] == binding["maple_case_identity_sha256"]
    prov = on_disk["provenance"]
    assert prov["source_stability"]["stable"] is True and prov["maple_matches_import_binding"] is True
    assert on_disk["backend"]["backend"] == "numpy" and "refused" in on_disk["backend"]["transfer_boundaries"]
    assert on_disk["backend"]["loop_transfer_counters"]["host_to_device"] == 0


def test_outputs_snapshot_final_state_and_hydrographs(short_run, plot1_case):
    from maple.case_tools.compilers.case_compiler import load_compiled_case
    from maple.io.outputs.snapshot import load_state_snapshot

    run, out = short_run
    assert sorted(p.name for p in out.iterdir()) == sorted([SUMMARY_NAME, FINAL_NAME, HYDROGRAPH_CSV, HYDROGRAPH_NPZ, SNAPSHOT_NAME])
    on_disk = json.loads((out / SUMMARY_NAME).read_text())
    for name in (FINAL_NAME, HYDROGRAPH_NPZ, HYDROGRAPH_CSV, SNAPSHOT_NAME):
        assert on_disk["outputs"][name] == _sha(out / name)
    final = np.load(out / FINAL_NAME)
    required = {"voxel_mass_kg", "active_mass_kg", "available_mass_kg", "bound_mass_kg", "committed_elevation_m",
                "initial_committed_elevation_m", "pending_bed_mass_change_kg", "depth_m", "mobile_mass_kg",
                "soil_water_m", "discharge_m2_s", "velocity_m_s", "sediment_velocity_m_s", "cumulative_pickup_kg",
                "cumulative_deposition_kg", "cumulative_export_request_kg", "water_exchange_net_kg", "bed_change_kg",
                "final_aspect", "final_slope", "initial_aspect"}
    assert required <= set(final.files)
    assert not np.any(final["pending_bed_mass_change_kg"])  # forced final commit: terrain corresponds to the bed
    # the committed elevation change follows the ACTUAL per-cell bed change (classes summed), and the water
    # class exchange is a separate grid
    dz = final["committed_elevation_m"] - final["initial_committed_elevation_m"]
    cell_change = final["bed_change_kg"].sum(axis=-1)
    moved = np.abs(cell_change) > 1e-8
    assert np.all(np.sign(dz[moved]) == np.sign(cell_change[moved]))
    np.testing.assert_allclose(dz, cell_change / (1250.0 * 0.25), rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(final["water_exchange_net_kg"], final["cumulative_deposition_kg"] - final["cumulative_pickup_kg"])
    assert not np.any(final["mobile_mass_kg"][final["depth_m"] == 0.0])  # dry cells hold no wet mobile mass
    # velocity / discharge outputs are the final accepted state's on the final graph: exactly k sqrt(h)
    # after a final commit, otherwise the routed k sqrt(h_flow) with |h - h_flow| <= the root tolerance
    st = run.state
    k_final = np.asarray(st.graph.conveyance).reshape(st.graph.shape)
    np.testing.assert_allclose(final["velocity_m_s"], np.sqrt(final["depth_m"]) * k_final, rtol=1e-6, atol=1e-12)
    np.testing.assert_array_equal(final["discharge_m2_s"], np.asarray(st.storm.discharge_m2_s))
    if on_disk["commits"]["last_commit_time_s"] == SHORT_END_S:
        np.testing.assert_array_equal(final["velocity_m_s"], np.sqrt(final["depth_m"]) * k_final)
    # the final MAPLE state returned in memory is what was written
    np.testing.assert_array_equal(np.asarray(st.bed.active_layer.mass_kg), final["active_mass_kg"])
    np.testing.assert_array_equal(np.asarray(st.bed.water.depth_m), final["depth_m"])
    np.testing.assert_array_equal(np.asarray(st.storm.depth_m), final["depth_m"])
    # MAPLE's own snapshot of the final bed / availability / water; not a SYRUP restart
    snap = load_state_snapshot(out / SNAPSHOT_NAME)
    assert snap["validated"] is True and snap["time_s"] == SHORT_END_S and snap["step"] == 180
    np.testing.assert_array_equal(snap["active_layer"].mass_kg, final["active_mass_kg"])
    np.testing.assert_array_equal(snap["voxel_column"].mass_kg, final["voxel_mass_kg"])
    if snap.get("water") is not None:
        np.testing.assert_array_equal(snap["water"].depth_m, final["depth_m"])
    assert "not a SYRUP restart" in on_disk["maple_state"]["snapshot"]
    # the loaded case object was never mutated
    fresh = load_compiled_case(plot1_case)
    used = run.verified.case
    np.testing.assert_array_equal(np.asarray(used.voxel_column.mass_kg), np.asarray(fresh.voxel_column.mass_kg))
    np.testing.assert_array_equal(np.asarray(used.active_layer.mass_kg), np.asarray(fresh.active_layer.mass_kg))
    assert not np.any(np.asarray(used.water.depth_m)) and not np.any(np.asarray(used.water.mobile_mass_by_cell_class_kg))
    # hydrographs
    hyd = np.load(out / HYDROGRAPH_NPZ)
    assert set(HYDROGRAPH_COLUMNS) <= set(hyd.files)
    sed_columns = [f"sed_{c}" for c in sediment_hydrograph_columns(6)]
    assert set(sed_columns) <= set(hyd.files) and hyd["t_s"].tolist() == [60.0, 120.0, 180.0]
    np.testing.assert_array_equal(hyd["sed_t_s"], hyd["t_s"])
    assert np.all(np.diff(hyd["sed_cumulative_export_kg"]) >= 0.0) and np.all(hyd["sed_mobile_kg"] >= 0.0)
    assert hyd["sed_cumulative_pickup_kg"][-1] == pytest.approx(on_disk["sediment"]["totals"]["actual_pickup"], rel=1e-12)
    np.testing.assert_array_equal(hyd["sed_t_s"], run.sediment_hydrograph[:, 0])
    # the last row's instantaneous outlet discharge and max velocity are the final state's
    q_final = st.graph.dx_m * float(np.asarray(st.storm.discharge_m2_s)[st.graph.outlet].sum())
    assert hyd["outlet_discharge_m3_s"][-1] == pytest.approx(q_final, rel=1e-12)  # summation order only
    assert hyd["max_velocity_m_s"][-1] == float(final["velocity_m_s"].max())
    with (out / HYDROGRAPH_CSV).open(encoding="ascii") as handle:
        header = handle.readline().strip().split(",")
        rows = [line.strip().split(",") for line in handle if line.strip()]
    assert header[: len(HYDROGRAPH_COLUMNS)] == list(HYDROGRAPH_COLUMNS) and "sed_mobile_kg" in header
    assert len(rows) == 3 and float(rows[-1][0]) == SHORT_END_S


def test_refusals_write_nothing(short_run, plot1_case, tmp_path):
    _run, out = short_run
    with pytest.raises(SedimentEventError, match="existing"):
        run_plot1_sediment_event(plot1_case, out, end_s=60.0, implementation="array")
    for kwargs, match in (
        ({"max_dt_s": 0.0}, "max_dt_s"),
        ({"end_s": -5.0}, "end_s"),
        ({"implementation": "fortran"}, "implementation"),
        ({"backend": "cupy"}, "refused"),
        ({"report_every_s": 0.0}, "report_every_s"),
        ({"max_retries": 1.5}, "max_retries"),
        ({"sediment_courant_max": 2.0}, "sediment_courant_max"),
        ({"max_transport_substeps": 0}, "max_transport_substeps"),
        ({"max_report_rows": 0}, "max_report_rows"),
    ):
        with pytest.raises((SedimentEventError, Exception), match=match):
            run_plot1_sediment_event(plot1_case, tmp_path / "never", **kwargs)
    with pytest.raises(SedimentEventError, match="inside the bound case"):
        run_plot1_sediment_event(plot1_case, plot1_case / "event", end_s=60.0, implementation="array")
    assert not (tmp_path / "never").exists() and not (plot1_case / "event").exists()
    assert not any(p.name.startswith(".never") for p in tmp_path.iterdir())


def test_missing_numba_is_a_clear_error_without_fallback(plot1_case, tmp_path, monkeypatch):
    monkeypatch.setattr(routing_numba, "numba_available", lambda: False)
    with pytest.raises(SedimentEventError, match="no fallback"):
        run_plot1_sediment_event(plot1_case, tmp_path / "out", end_s=60.0, implementation="numba")
    assert not (tmp_path / "out").exists()


def test_source_change_during_the_run_writes_nothing(plot1_case, tmp_path, monkeypatch):
    real = sediment_experiment._source_digests
    monkeypatch.setattr(sediment_experiment, "_source_digests", lambda *a: {**real(*a), "maple_syrup": "0" * 64})
    with pytest.raises(SedimentEventError, match="source changed during the run"):
        run_plot1_sediment_event(plot1_case, tmp_path / "out", end_s=30.0, implementation="array")
    assert not (tmp_path / "out").exists() and not any(p.name.startswith(".out") for p in tmp_path.iterdir())


@NUMBA_SKIP
def test_numba_run_matches_the_array_short_run_with_actual_coupling(short_run, plot1_case, tmp_path):
    """Same 180 s window and reporting controls as the array `short_run`
    (which already carries nonzero pickup, deposition and commits): the
    compiled hydraulic sweep must reproduce the whole coupled state."""
    a, _ = short_run
    # Same (array) transport kernel on both sides: this compares the hydraulic sweep implementations
    # bitwise. The numba transport kernel agrees with the array kernel to round-off only, which through
    # terrain commits perturbs conveyance at the 1e-14 level; that pairing is tested in tests/phase7b.
    b = run_plot1_sediment_event(plot1_case, tmp_path / "numba", max_dt_s=1.0, end_s=SHORT_END_S,
                                 implementation="numba", report_every_s=SHORT_CADENCE_S,
                                 transport_implementation="array")
    totals = a.summary["sediment"]["totals"]
    assert totals["actual_pickup"] > 0.0 and totals["deposition_actual"] > 0.0 and a.summary["commits"]["n_commits"] >= 1
    for name in ("depth_m", "soil_water_m", "discharge_m2_s"):
        np.testing.assert_array_equal(getattr(a.state.storm, name), getattr(b.state.storm, name), err_msg=name)
    for name in ("voxel_column", "active_layer"):
        np.testing.assert_array_equal(getattr(a.state.bed, name).mass_kg, getattr(b.state.bed, name).mass_kg, err_msg=name)
    np.testing.assert_array_equal(a.state.bed.water.mobile_mass_by_cell_class_kg, b.state.bed.water.mobile_mass_by_cell_class_kg)
    np.testing.assert_array_equal(a.state.bed.water.depth_m, b.state.bed.water.depth_m)
    np.testing.assert_array_equal(a.state.bed.sediment_availability.available_mass_kg,
                                  b.state.bed.sediment_availability.available_mass_kg)
    np.testing.assert_array_equal(a.state.bed.committed_topography.elevation_offset_m,
                                  b.state.bed.committed_topography.elevation_offset_m)
    assert a.state.bed.committed_topography.commit_count == b.state.bed.committed_topography.commit_count
    np.testing.assert_array_equal(a.state.sediment_velocity_m_s, b.state.sediment_velocity_m_s)
    np.testing.assert_array_equal(a.state.graph.aspect, b.state.graph.aspect)
    np.testing.assert_array_equal(a.state.graph.slope, b.state.graph.slope)
    np.testing.assert_array_equal(a.hydrograph, b.hydrograph)
    np.testing.assert_array_equal(a.sediment_hydrograph, b.sediment_hydrograph)
    assert a.summary["sediment"]["by_class"] == b.summary["sediment"]["by_class"]
    assert a.summary["budget"]["export_m3"] == b.summary["budget"]["export_m3"]
    assert b.summary["routing"]["implementation"] == "numba" and b.summary["provenance"]["numba"] is not None


def test_cli_short_run(plot1_case, tmp_path, capsys):
    out = tmp_path / "cli"
    code = main(["--case-dir", str(plot1_case), "--output-dir", str(out), "--end-s", "60",
                 "--implementation", "array", "--report-every-s", "30"])
    assert code == 0
    text = capsys.readouterr().out
    printed = json.loads(text[text.index("{"):])
    assert printed["time"]["n_accepted_steps"] == 60 and "NOT an event completion" in printed["status"]
    assert printed["sediment"]["closure"]["closed"] is True
    assert (out / SUMMARY_NAME).is_file() and (out / SNAPSHOT_NAME).is_file()
    assert main(["--case-dir", str(plot1_case), "--output-dir", str(out), "--implementation", "array"]) == 1
    assert main(["--case-dir", str(plot1_case), "--output-dir", str(tmp_path / "gpu"), "--implementation", "array",
                 "--backend", "cupy", "--end-s", "10"]) == 1
    assert not (tmp_path / "gpu").exists()
