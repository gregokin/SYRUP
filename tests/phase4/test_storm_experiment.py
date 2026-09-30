"""Phase 4c Plot 1 coupled storm runner on the actual MAPLE case.

A fresh Plot 1 case is generated once per module with the Phase 2 importer.
Runs are kept short (`end_s` of 60-180 s) so the module stays fast; the
full 5400 s storm + recession and the dt study are CLI runs for Codex
(docs/phase4/storm.md). Nothing here compares with executed MAHLERAN
Fortran; that is a later task."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_routing import NUMBA_SKIP

from maple_syrup import routing_numba, storm_experiment
from maple_syrup.rainfall import parse_legacy_rainfall_file
from maple_syrup.storm import HYDROGRAPH_COLUMNS, StormError
from maple_syrup.storm_experiment import (
    FINAL_NAME,
    HYDROGRAPH_CSV,
    HYDROGRAPH_NPZ,
    SUMMARY_NAME,
    main,
    run_plot1_storm,
)

REPO = Path(__file__).resolve().parents[2]
RECIPE_PATH = REPO / "cases" / "plot1" / "recipe.yaml"
MAHLERAN_ROOT = Path(os.environ.get("MAPLE_SYRUP_MAHLERAN_ROOT", "/home/okin/MAHLERAN"))
PLOT1_DIR = MAHLERAN_ROOT / "Input" / "input_p1"

pytest.importorskip("maple")


@pytest.fixture(scope="module")
def plot1_case(tmp_path_factory):
    if not (MAHLERAN_ROOT / "mahleran_input.xml").is_file() or not PLOT1_DIR.is_dir():
        pytest.skip(f"MAHLERAN reference not available at {MAHLERAN_ROOT}")
    from maple_syrup.case_import import generate_plot1_case, load_recipe

    out = tmp_path_factory.mktemp("plot1_storm_case") / "case"
    generate_plot1_case(load_recipe(RECIPE_PATH, mahleran_root=MAHLERAN_ROOT), out)
    return out


@pytest.fixture(scope="module")
def short_run(plot1_case, tmp_path_factory):
    out = tmp_path_factory.mktemp("plot1_storm_run") / "storm"
    run = run_plot1_storm(plot1_case, out, max_dt_s=1.0, end_s=180.0, implementation="array",
                          report_every_s=60.0)
    return run, out


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_short_storm_budget_forcing_and_status(short_run, plot1_case):
    run, _ = short_run
    s = run.summary
    b, t = s["budget"], s["time"]
    schedule = parse_legacy_rainfall_file(plot1_case / "syrup" / "rainfall" / "p1_01_08_06.dat")
    assert b["rain_m3"] == pytest.approx(schedule.depth_m(0.0, 180.0) * 300.0, rel=1e-12)
    assert b["rain_m3"] < schedule.total_depth_m() * 300.0  # only the simulated window is integrated
    total = b["surface_final_m3"] + b["soil_final_m3"] + b["drainage_m3"] + b["export_m3"]
    assert total == pytest.approx(b["surface_initial_m3"] + b["soil_initial_m3"] + b["rain_m3"], rel=1e-12)
    for key in ("water_residual_m3", "surface_residual_m3", "soil_residual_m3"):
        assert abs(b[key]) <= b["tolerance_m3"] < 1e-7
    assert abs(b["rainfall_integral_residual_m3"]) <= b["rainfall_tolerance_m3"]
    assert b["export_m3"] >= 0.0 and b["surface_final_m3"] > 0.0 and b["soil_initial_m3"] == pytest.approx(22.5)
    assert (t["n_accepted_steps"], t["end_s"], t["max_dt_s"], t["n_rejected_attempts"]) == (180, 180.0, 1.0, 0)
    assert t["recession_included"] is False and t["rainfall_end_s"] == 1620.0 and t["legacy_stormlength_s"] == 5400.0
    assert "NOT an event completion" in s["status"] and "no dry reset" in s["status"]
    assert s["routing"]["implementation"] == "array" and s["provenance"]["implementation"] == "array"
    assert s["routing"]["max_courant_old"] < 1.0 and s["routing"]["max_constitutive_residual_m"] <= 1e-11
    assert s["domain"]["graph"]["n_outlets"] >= 1 and s["domain"]["active_cells"] == 1200
    cells = s["routing"]["cell_steps"]
    assert cells["no_runon"] + cells["partial_runon"] + cells["complete_runon"] == 180 * 1200
    assert s["final_state"]["ponded_cells"] > 0
    fs = s["final_state"]
    assert fs["peak_outlet_discharge_m3_s"] >= fs["sampled_peak_outlet_discharge_m3_s"] >= 0.0
    assert 0.0 <= fs["time_of_peak_outlet_discharge_s"] <= 180.0 and "true numerical peak" in fs["peak_note"]
    assert t["min_dt_s"] == 1.0 / 1024.0


def test_hydrograph_rows_are_bounded_consistent_and_written(short_run):
    run, out = short_run
    hyd = np.load(out / HYDROGRAPH_NPZ)
    assert set(HYDROGRAPH_COLUMNS).issubset(hyd.files)
    assert hyd["t_s"].tolist() == [60.0, 120.0, 180.0]  # knots coincide with the 60 s cadence
    np.testing.assert_array_equal(hyd["t_s"], run.hydrograph[:, 0])
    assert np.all(np.abs(hyd["row_water_residual_m3"]) <= run.summary["budget"]["tolerance_m3"])
    assert hyd["interval_rain_m3"].sum() == pytest.approx(run.summary["budget"]["rain_m3"], rel=1e-12)
    assert hyd["cumulative_export_m3"][-1] == pytest.approx(run.summary["budget"]["export_m3"], rel=1e-12)
    assert np.all(np.diff(hyd["cumulative_export_m3"]) >= 0.0) and np.all(hyd["outlet_discharge_m3_s"] >= 0.0)
    with (out / HYDROGRAPH_CSV).open(encoding="ascii") as handle:
        header = handle.readline().strip().split(",")
        rows = [line.strip().split(",") for line in handle if line.strip()]
    assert header[: len(HYDROGRAPH_COLUMNS)] == list(HYDROGRAPH_COLUMNS) and "interval_export_m3" in header
    assert len(rows) == 3 and float(rows[-1][0]) == 180.0
    assert run.summary["hydrograph"]["n_rows"] == 3 and "instantaneous" in run.summary["budget"]["export_meaning"]


def test_outputs_maple_state_and_provenance(short_run, plot1_case):
    from maple.case_tools.compilers.case_compiler import load_compiled_case
    from maple.core.types.water import WaterState

    run, out = short_run
    assert sorted(p.name for p in out.iterdir()) == sorted([SUMMARY_NAME, FINAL_NAME, HYDROGRAPH_CSV, HYDROGRAPH_NPZ])
    on_disk = json.loads((out / SUMMARY_NAME).read_text())
    for name in (FINAL_NAME, HYDROGRAPH_NPZ, HYDROGRAPH_CSV):
        assert on_disk["outputs"][name] == _sha(out / name)
    final = np.load(out / FINAL_NAME)
    assert {"depth_m", "soil_water_m", "discharge_m2_s", "velocity_m_s", "peak_depth_m", "peak_velocity_m_s",
            "cumulative_rain_m", "cumulative_intake_m", "cumulative_return_m", "cumulative_drainage_m"} == set(final.files)
    assert np.all(final["peak_depth_m"] >= final["depth_m"]) and np.all(final["depth_m"] >= 0.0)
    fresh = load_compiled_case(plot1_case)
    used = run.verified.case
    np.testing.assert_array_equal(np.asarray(used.voxel_column.mass_kg), np.asarray(fresh.voxel_column.mass_kg))
    np.testing.assert_array_equal(np.asarray(used.active_layer.mass_kg), np.asarray(fresh.active_layer.mass_kg))
    np.testing.assert_array_equal(np.asarray(used.topography_result.elevation_m),
                                  np.asarray(fresh.topography_result.elevation_m))
    assert isinstance(run.water, WaterState)
    assert run.water.mobile_mass_by_cell_class_kg is used.water.mobile_mass_by_cell_class_kg
    np.testing.assert_array_equal(run.water.depth_m, final["depth_m"])
    assert not np.any(np.asarray(used.water.depth_m))  # the loaded state itself was not overwritten
    assert on_disk["maple_state"]["sediment_digest_before"] == on_disk["maple_state"]["sediment_digest_after"]
    prov = on_disk["provenance"]
    assert prov["source_stability"]["stable"] is True and prov["maple_matches_import_binding"] is True
    assert prov["implementation"] == "array" and (prov["numba"] is None or set(prov["numba"]) == {"numba", "llvmlite"})
    assert on_disk["case"]["maple_case_identity_sha256"] == json.loads(
        (plot1_case / "syrup" / "plot1_binding.json").read_text())["maple_case_identity_sha256"]
    assert on_disk["backend"]["loop_transfer_counters"]["host_to_device"] == 0
    assert on_disk["timings"]["first_accepted_step"]["wall_s"] >= 0.0


def test_refusals_write_nothing(short_run, plot1_case, tmp_path):
    _run, out = short_run
    with pytest.raises(StormError, match="existing"):
        run_plot1_storm(plot1_case, out, end_s=60.0, implementation="array")
    for kwargs, match in (
        ({"max_dt_s": 0.0}, "max_dt_s"),
        ({"end_s": -5.0}, "end_s"),
        ({"implementation": "fortran"}, "implementation"),
        ({"implementation": "numba", "backend": "cupy"}, "numpy backend only"),
        ({"report_every_s": 0.0}, "report_every_s"),
        ({"report_every_s": True}, "report_every_s"),
        # control guards are validated on the original values, never coerced
        ({"max_retries": 1.5}, "max_retries"),
        ({"max_retries": True}, "max_retries"),
        ({"max_steps": True}, "max_steps"),
        ({"max_steps": 10.0}, "max_steps"),
        ({"min_dt_s": True}, "min_dt_s"),
        ({"max_report_rows": 0}, "max_report_rows"),
        ({"max_report_rows": 2.0}, "max_report_rows"),
    ):
        with pytest.raises(StormError, match=match):
            run_plot1_storm(plot1_case, tmp_path / "never", **kwargs)
    with pytest.raises(StormError, match="inside the bound case"):
        run_plot1_storm(plot1_case, plot1_case / "storm", end_s=60.0, implementation="array")
    assert not (tmp_path / "never").exists() and not (plot1_case / "storm").exists()


def test_missing_numba_is_a_clear_error_without_fallback(plot1_case, tmp_path, monkeypatch):
    monkeypatch.setattr(routing_numba, "numba_available", lambda: False)
    with pytest.raises(StormError, match="no fallback"):
        run_plot1_storm(plot1_case, tmp_path / "out", end_s=60.0, implementation="numba")
    assert not (tmp_path / "out").exists()


def test_source_change_during_the_run_writes_nothing(plot1_case, tmp_path, monkeypatch):
    real = storm_experiment._source_digests
    monkeypatch.setattr(storm_experiment, "_source_digests", lambda *a: {**real(*a), "maple_syrup": "0" * 64})
    with pytest.raises(StormError, match="source changed during the run"):
        run_plot1_storm(plot1_case, tmp_path / "out", end_s=60.0, implementation="array")
    assert not (tmp_path / "out").exists()


@NUMBA_SKIP
def test_numba_and_array_runs_are_identical(plot1_case, tmp_path):
    a = run_plot1_storm(plot1_case, tmp_path / "array", end_s=60.0, implementation="array", report_every_s=30.0)
    b = run_plot1_storm(plot1_case, tmp_path / "numba", end_s=60.0, implementation="numba", report_every_s=30.0)
    np.testing.assert_array_equal(a.water.depth_m, b.water.depth_m)
    np.testing.assert_array_equal(a.soil_water_m, b.soil_water_m)
    np.testing.assert_array_equal(a.hydrograph, b.hydrograph)
    assert a.summary["budget"]["export_m3"] == b.summary["budget"]["export_m3"]
    assert b.summary["provenance"]["numba"] is not None and b.summary["routing"]["implementation"] == "numba"
    assert b.summary["timings"]["first_accepted_step"]["wall_s"] > 0.0


def test_cli_short_run(plot1_case, tmp_path, capsys):
    out = tmp_path / "cli"
    code = main(["--case-dir", str(plot1_case), "--output-dir", str(out), "--end-s", "60",
                 "--implementation", "array", "--report-every-s", "30"])
    assert code == 0
    text = capsys.readouterr().out
    printed = json.loads(text[text.index("{"):])
    assert printed["time"]["n_accepted_steps"] == 60 and "NOT an event completion" in printed["status"]
    assert (out / SUMMARY_NAME).is_file()
    assert main(["--case-dir", str(plot1_case), "--output-dir", str(out), "--implementation", "array"]) == 1
