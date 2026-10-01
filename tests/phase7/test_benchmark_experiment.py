"""Frozen-geometry benchmark mode on real MAPLE beds, applied-rainfall
forcing, the Plot 1 runner and its refusals. Nothing here executes MAHLERAN
or claims storm-scale agreement; the comparator is tested separately."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")

from test_sediment_event import make_bed, setup, valley_elevation

from maple_syrup import routing_numba
from maple_syrup.benchmark_experiment import (
    FINAL_NAME,
    FORCING_NAME,
    HYDROGRAPH_NPZ,
    SUMMARY_NAME,
    frozen_control,
    frozen_geometry_report,
    load_applied_rainfall,
    main,
    run_frozen_event,
    run_plot1_matched_benchmark,
)
from maple_syrup.case_import import Plot1ImportError
from maple_syrup.rainfall import RainfallError, constant_rainfall
from maple_syrup.sediment_bed import bed_inventory
from maple_syrup.sediment_event import (
    SedimentEventControl,
    SedimentEventError,
    evolve_sediment_event,
)
from maple_syrup.storm import StormControl

REPO = Path(__file__).resolve().parents[2]
APPLIED_CSV = REPO / "outputs" / "phase7" / "mahleran_reference_audit" / "applied_rainfall.csv"
REFERENCE_RUN = REPO / "outputs" / "phase7" / "mahleran_deterministic_ksat_run"
NUMBA_SKIP = pytest.mark.skipif(not routing_numba.numba_available(), reason="Numba not installed; no claim made")


def committing_valley():
    """Valley bed whose MAPLE commit interval (10 s) would refresh the
    hydraulic geometry several times in a 45 s storm in normal mode."""
    from maple.core.parameters.topographic_commit import TopographicCommitSpec

    bed, ctx = make_bed(valley_elevation(4, 5), commit_spec=TopographicCommitSpec(commit_interval_s=10.0))
    return setup(bed, ctx, south_ring=valley_elevation(5, 5)[0] - 0.01)


def run_frozen(inputs, schedule, end, *, control=None, cadence=15.0, **kwargs):
    return run_frozen_event(inputs["state0"], inputs["ctx"], inputs["column"], inputs["field"], schedule,
                            inputs["vegetation"], inputs["sediment"], end, control or frozen_control(),
                            report_every_s=cadence, **kwargs)


# --- frozen hydraulic geometry while MAPLE mass evolves ---------------------------------------------------
def test_frozen_mode_keeps_geometry_while_actual_maple_mass_changes():
    from maple.core.types.topographic_commit import committed_elevation_m

    inputs = committing_valley()
    state0 = inputs["state0"]
    schedule = constant_rainfall(0.0, 45.0, 72.0)
    initial_bed = np.asarray(bed_inventory(state0.bed)).copy()
    initial_active = np.asarray(state0.bed.active_layer.mass_kg).copy()
    z0 = np.asarray(committed_elevation_m(state0.bed.committed_topography)).copy()
    result, checks = run_frozen(inputs, schedule, 45.0)
    st = result.state
    # hydraulic geometry: same objects, same arrays, no commit, no reroute
    assert checks["frozen_geometry"]["frozen"] and all(checks["frozen_geometry"]["checks"].values())
    assert st.graph is state0.graph and st.network is state0.network and st.grid is state0.grid
    assert result.n_commits == 0 and result.n_forced_commits == 0 and result.n_graph_changes == 0
    assert st.bed.committed_topography.commit_count == 0
    np.testing.assert_array_equal(committed_elevation_m(st.bed.committed_topography), z0)
    # ... while actual MAPLE sediment moved and the pending ledger records it
    bc = {k: np.asarray(v) for k, v in result.by_class.items()}
    assert bc["actual_pickup"].sum() > 0.0 and bc["deposition_actual"].sum() > 0.0 and bc["export_actual"].sum() > 0.0
    assert checks["exchange"]["nonzero"]
    pending = np.asarray(st.bed.ledger.pending_bed_mass_change_kg)
    assert np.any(pending != 0.0) and not checks["maple_end_state"]["ledger_empty"]
    assert checks["maple_end_state"]["ledger_reconciled_against"]
    assert not np.array_equal(np.asarray(st.bed.active_layer.mass_kg), initial_active)
    final_bed = np.asarray(bed_inventory(st.bed))
    assert np.any(final_bed != initial_bed)
    # the pending ledger IS the actual bed change (frozen mode never resets it)
    change = np.asarray(result.bed_change_by_cell_class_kg())
    np.testing.assert_allclose(pending.sum(axis=(0, 1)), change.sum(axis=(0, 1)), atol=1e-9)
    closure = result.closure()
    assert closure["closed"] and closure["request_reconciled"]
    # net bed mass change is reported although the elevation is frozen
    assert abs(float(np.sum(closure["net_bed_change_kg"]))) > 0.0
    assert np.array_equal(np.asarray(committed_elevation_m(st.bed.committed_topography)), z0)


def test_normal_evolving_mode_is_unchanged_and_differs_from_frozen_mode():
    inputs = committing_valley()
    state0 = inputs["state0"]
    schedule = constant_rainfall(0.0, 45.0, 72.0)
    default = SedimentEventControl()
    assert default.commit and default.force_final_commit  # Phase 5 defaults untouched
    evolving = evolve_sediment_event(state0, inputs["ctx"], inputs["column"], inputs["field"], schedule,
                                     inputs["vegetation"], inputs["sediment"], 45.0, default, report_every_s=15.0)
    assert evolving.n_commits >= 1 and evolving.state.graph is not state0.graph
    assert evolving.state.bed.committed_topography.commit_count == evolving.n_commits
    assert not np.any(evolving.state.bed.ledger.pending_bed_mass_change_kg)  # forced final commit
    frozen, _ = run_frozen(inputs, schedule, 45.0)
    assert frozen.n_commits == 0 and np.any(frozen.state.bed.ledger.pending_bed_mass_change_kg)
    # the two modes are physically different runs (rerouting and re-initialised discharge in normal mode)
    assert evolving.n_accepted_steps == frozen.n_accepted_steps == 45
    with pytest.raises(SedimentEventError, match="frozen hydraulic geometry violated"):
        frozen_geometry_report(state0, evolving.state, evolving)


def test_frozen_mode_refuses_commit_controls_and_empty_exchange():
    inputs = committing_valley()
    state0 = inputs["state0"]
    before = np.asarray(state0.bed.active_layer.mass_kg).copy()
    with pytest.raises(SedimentEventError, match="commit"):
        frozen_control(commit=True)
    with pytest.raises(SedimentEventError, match="force_final_commit"):
        frozen_control(force_final_commit=False)
    for control in (SedimentEventControl(commit=True, force_final_commit=False),
                    SedimentEventControl(commit=False, force_final_commit=True)):
        with pytest.raises(SedimentEventError, match="commit=False AND force_final_commit=False"):
            run_frozen(inputs, constant_rainfall(0.0, 10.0, 72.0), 10.0, control=control, cadence=10.0)
    np.testing.assert_array_equal(state0.bed.active_layer.mass_kg, before)
    # a dry, rainless window moves nothing: not a sediment benchmark
    with pytest.raises(SedimentEventError, match="no actual MAPLE pickup"):
        run_frozen(inputs, constant_rainfall(0.0, 10.0, 0.0), 10.0, cadence=10.0)
    result, checks = run_frozen(inputs, constant_rainfall(0.0, 10.0, 0.0), 10.0, cadence=10.0, require_exchange=False)
    assert not checks["exchange"]["nonzero"] and result.n_accepted_steps == 10


@NUMBA_SKIP
def test_frozen_mode_array_and_numba_hydraulics_agree():
    """Hydraulic sweep implementations with the SAME (array) transport kernel
    are bitwise equal; the transport kernel implementations are compared to
    round-off in tests/phase7b."""
    inputs = committing_valley()
    schedule = constant_rainfall(0.0, 30.0, 72.0)
    a, _ = run_frozen(inputs, schedule, 30.0, control=frozen_control(StormControl(implementation="array"),
                                                                     transport_implementation="array"), cadence=10.0)
    b, _ = run_frozen(inputs, schedule, 30.0, control=frozen_control(StormControl(implementation="numba"),
                                                                     transport_implementation="array"), cadence=10.0)
    for name in ("depth_m", "soil_water_m", "discharge_m2_s"):
        np.testing.assert_array_equal(getattr(a.state.storm, name), getattr(b.state.storm, name), err_msg=name)
    np.testing.assert_array_equal(a.state.bed.active_layer.mass_kg, b.state.bed.active_layer.mass_kg)
    np.testing.assert_array_equal(a.state.bed.water.mobile_mass_by_cell_class_kg,
                                  b.state.bed.water.mobile_mass_by_cell_class_kg)
    np.testing.assert_array_equal(a.state.bed.ledger.pending_bed_mass_change_kg,
                                  b.state.bed.ledger.pending_bed_mass_change_kg)
    np.testing.assert_array_equal(a.hydrograph, b.hydrograph)
    np.testing.assert_array_equal(a.sediment_hydrograph, b.sediment_hydrograph)


# --- applied rainfall override -------------------------------------------------------------------------------
def write_csv(path: Path, rows, header="start_s,end_s,logged_applied_rain_mm_h"):
    path.write_text(header + "\n" + "\n".join(",".join(repr(float(v)) for v in row) for row in rows) + "\n")


def test_applied_rainfall_csv_becomes_an_exact_compressed_schedule(tmp_path):
    rows = [(0, 1, 15.24), (1, 2, 15.24), (2, 3, 30.48), (3, 4, 0.0), (4, 5, 0.0), (5, 6, 7.62)]
    path = tmp_path / "applied.csv"
    write_csv(path, rows)
    schedule, record = load_applied_rainfall(path)
    assert schedule.edges_s.tolist() == [0.0, 2.0, 3.0, 5.0, 6.0]
    assert schedule.intensity_mm_per_h.tolist() == [15.24, 30.48, 0.0, 7.62]
    expected_mm = sum(i for _, _, i in rows) / 3600.0
    assert schedule.total_depth_m() * 1e3 == pytest.approx(expected_mm, rel=1e-12)
    assert schedule.rate_after_m_per_s(4.5) == 0.0 and schedule.rate_after_m_per_s(6.0) == 0.0
    assert schedule.depth_m(0.0, 6.0) == pytest.approx(schedule.total_depth_m())
    assert record["n_rows"] == 6 and record["n_pieces"] == 4 and record["interval_lengths_s"] == [1.0]
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert schedule.provenance.kind == "mahleran_applied_rainfall_csv" and schedule.provenance.sha256 == record["sha256"]
    assert "one step late" in schedule.provenance.convention


@pytest.mark.parametrize("rows, header, match", [
    ([(0, 1, 1.0), (2, 3, 1.0)], None, "contiguous"),
    ([(0, 1, 1.0), (0.5, 1.5, 1.0)], None, "contiguous"),
    ([(0, 1, -1.0)], None, "negative"),
    ([(0, 1, float("nan"))], None, "non-finite"),
    ([(1, 2, 1.0)], None, "start at 0"),
    ([(0, 0, 1.0)], None, "end_s > start_s"),
    ([(0, 1, 1.0)], "t,end,rain", "expected header"),
    ([], None, "expected header"),
])
def test_malformed_applied_rainfall_is_refused(tmp_path, rows, header, match):
    path = tmp_path / "bad.csv"
    if rows or header:
        write_csv(path, rows, header=header or "start_s,end_s,logged_applied_rain_mm_h")
    else:
        path.write_text("")
    with pytest.raises(RainfallError, match=match):
        load_applied_rainfall(path)


# --- Plot 1 runner --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def applied_csv():
    if not APPLIED_CSV.is_file():
        pytest.skip(f"reference applied rainfall not available at {APPLIED_CSV}")
    return APPLIED_CSV


@pytest.fixture(scope="module")
def short_benchmark(plot1_case, applied_csv, tmp_path_factory):
    out = tmp_path_factory.mktemp("plot1_matched") / "dt1"
    reference = REFERENCE_RUN if (REFERENCE_RUN / "execution.json").is_file() else None
    run = run_plot1_matched_benchmark(plot1_case, out, applied_rainfall_csv=applied_csv, reference_run_dir=reference,
                                      end_s=120.0, implementation="array", report_every_s=1.0)
    return run, out


def test_plot1_short_window_is_frozen_matched_and_closed(short_benchmark, plot1_case):
    run, out = short_benchmark
    s = run.summary
    assert s["mode"]["frozen_hydraulic_geometry"] and not s["mode"]["commit"] and not s["mode"]["force_final_commit"]
    assert all(s["mode"]["checks"]["checks"].values()) and s["mode"]["restart"]["supported"] is False
    assert s["time"]["n_accepted_steps"] == 120 and s["time"]["report_every_s"] == 1.0 and s["hydrograph"]["n_rows"] == 120
    assert s["sediment"]["totals"]["actual_pickup"] > 0.0 and s["sediment"]["totals"]["deposition_actual"] > 0.0
    assert s["sediment"]["closure"]["closed"] and s["sediment"]["closure"]["request_reconciled"]
    assert abs(s["budget"]["water_residual_m3"]) <= s["budget"]["tolerance_m3"]
    assert s["conductivity"]["mm_s"] == 0.00025 and s["forcing"]["applied_rainfall_override"]["n_rows"] == 5400
    assert s["forcing"]["original_case_rainfall"]["binding"] == "unchanged"
    # both forcing depths over the window and their difference are recorded (sign depends on the records)
    assert np.isfinite(s["forcing"]["depth_difference_mm"])
    assert s["forcing"]["applied_rainfall_override"]["sha256"] == hashlib.sha256(APPLIED_CSV.read_bytes()).hexdigest()
    binding = json.loads((plot1_case / "syrup" / "plot1_binding.json").read_text())
    assert s["case"]["maple_case_identity_sha256"] == binding["maple_case_identity_sha256"]
    assert s["mode"]["maple_end_state"]["pending_bed_mass_abs_kg"] > 0.0
    assert s["backend"]["gpu"].startswith("not exercised")
    assert "peak_rss_kib" in s["performance"] and s["performance"]["first_accepted_step"]["wall_s"] >= 0.0
    if s["reference_run"] is not None:
        assert s["reference_run"]["conductivity_distribution"] == "deterministic"
    # files: no MAPLE snapshot, no checkpoint; final grids are the wet diagnostic state on the frozen graph
    assert sorted(p.name for p in out.iterdir()) == sorted([SUMMARY_NAME, FINAL_NAME, FORCING_NAME, HYDROGRAPH_NPZ,
                                                             "hydrograph.csv"])
    final = np.load(out / FINAL_NAME)
    np.testing.assert_array_equal(final["committed_elevation_m"], final["initial_committed_elevation_m"])
    assert np.any(final["pending_bed_mass_change_kg"] != 0.0)
    np.testing.assert_array_equal(final["aspect"], run.state.graph.aspect)
    assert np.any(final["bed_change_kg"] != 0.0)
    with np.load(out / HYDROGRAPH_NPZ) as h:
        assert h["t_s"].tolist() == list(range(1, 121)) and np.all(np.diff(h["sed_cumulative_export_kg"]) >= 0.0)
        assert h["outlet_discharge_m3_s"].max() > 0.0
    with np.load(out / FORCING_NAME) as f:
        assert f["edges_s"][0] == 0.0 and f["edges_s"][-1] == 5400.0


def test_plot1_runner_refusals_write_nothing(short_benchmark, plot1_case, applied_csv, tmp_path):
    _run, out = short_benchmark
    with pytest.raises(SedimentEventError, match="existing"):
        run_plot1_matched_benchmark(plot1_case, out, applied_rainfall_csv=applied_csv, end_s=30.0, implementation="array")
    with pytest.raises(Plot1ImportError, match="inside the case tree"):
        run_plot1_matched_benchmark(plot1_case, plot1_case / "bench", applied_rainfall_csv=applied_csv, end_s=30.0,
                                    implementation="array")
    with pytest.raises(Plot1ImportError, match="applied rainfall"):
        run_plot1_matched_benchmark(plot1_case, applied_csv.parent / "bench", applied_rainfall_csv=applied_csv,
                                    end_s=30.0, implementation="array")
    # Malformed forcing files live in their own directory: the applied-CSV parent is a protected tree,
    # so an output beside the CSV would be refused before the parser ever saw the file.
    forcing_dir = tmp_path / "forcing"
    forcing_dir.mkdir()
    never = tmp_path / "out" / "never"
    bad = forcing_dir / "bad.csv"
    write_csv(bad, [(0, 1, 1.0), (2, 3, 1.0)])
    with pytest.raises(RainfallError, match="contiguous"):
        run_plot1_matched_benchmark(plot1_case, never, applied_rainfall_csv=bad, end_s=30.0, implementation="array")
    short = forcing_dir / "short.csv"
    write_csv(short, [(0, 1, 1.0)])
    with pytest.raises(SedimentEventError, match="before the window end"):
        run_plot1_matched_benchmark(plot1_case, never, applied_rainfall_csv=short, end_s=30.0, implementation="array")
    with pytest.raises(SedimentEventError, match="refused"):
        run_plot1_matched_benchmark(plot1_case, never, applied_rainfall_csv=applied_csv, end_s=30.0,
                                    implementation="array", backend="cupy")
    with pytest.raises(SedimentEventError, match="does not exist"):
        run_plot1_matched_benchmark(plot1_case, never, applied_rainfall_csv=forcing_dir / "missing.csv", end_s=30.0,
                                    implementation="array")
    # the protection itself: an output inside the (malformed) forcing directory is refused before parsing
    with pytest.raises(Plot1ImportError, match="applied rainfall"):
        run_plot1_matched_benchmark(plot1_case, forcing_dir / "inside", applied_rainfall_csv=bad, end_s=30.0,
                                    implementation="array")
    assert not never.exists() and not (tmp_path / "out").exists() and not (plot1_case / "bench").exists()
    assert not (applied_csv.parent / "bench").exists() and not (forcing_dir / "inside").exists()
    assert not any(p.name.startswith(".never") for p in tmp_path.iterdir())


def test_cli_short_window(plot1_case, applied_csv, tmp_path, capsys):
    out = tmp_path / "cli"
    code = main(["--case-dir", str(plot1_case), "--output-dir", str(out), "--applied-rainfall", str(applied_csv),
                 "--end-s", "60", "--implementation", "array",
                 "--transport-scheme", "characteristic"])
    assert code == 0
    text = capsys.readouterr().out
    printed = json.loads(text[text.index("{"):])
    assert printed["mode"] == {"frozen_hydraulic_geometry": True, "commit": False, "force_final_commit": False}
    assert printed["time"]["n_accepted_steps"] == 60 and "FROZEN" in printed["status"]
    assert (out / SUMMARY_NAME).is_file()
    assert main(["--case-dir", str(plot1_case), "--output-dir", str(out), "--applied-rainfall", str(applied_csv),
                 "--end-s", "10", "--implementation", "array",
                 "--transport-scheme", "characteristic"]) == 1
    assert main(["--case-dir", str(plot1_case), "--output-dir", str(tmp_path / "gpu"), "--applied-rainfall",
                 str(applied_csv), "--end-s", "10", "--implementation", "array", "--backend", "cupy",
                 "--transport-scheme", "characteristic"]) == 1
    assert not (tmp_path / "gpu").exists()
    with pytest.raises(SystemExit):  # restart is not a feature of this benchmark: no --resume option exists
        main(["--case-dir", str(plot1_case), "--output-dir", str(tmp_path / "x"), "--applied-rainfall",
              str(applied_csv), "--resume", "anything", "--transport-scheme", "characteristic"])
    assert not (tmp_path / "x").exists()
