"""The experimental comparison CLI (`python -m maple_syrup.experimental_experiment`) on the actual Plot 1 case: short runs of both
candidates on the NumPy reference and (with a device) the CUDA form. The full 5400 s comparisons are the root's task, not this
file's. Checks: outputs and budgets, source/bed protection records, solver/boundary/numerics provenance, honest labelling,
CPU/GPU agreement and the counted packet reads. Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup import experimental_experiment as ee

SOLVERS = ("explicit", "local_inertial")
RTOL, ATOL = 2.0e-12, 1.0e-14


def run_cli(case, out, *extra, end="90") -> dict:
    argv = ["--case-dir", str(case), "--output-dir", str(out), "--end-s", end, "--report-every-s", "30",
            "--allow-maple-source-change", *extra]
    assert ee.main(argv) == 0
    return json.loads((out / ee.SUMMARY_NAME).read_text())


@pytest.mark.parametrize("solver", SOLVERS)
def test_short_plot1_run_on_the_numpy_reference_writes_the_documented_outputs(plot1_case, tmp_path, solver):
    out = tmp_path / solver
    summary = run_cli(plot1_case, out, "--solver", solver, "--backend", "numpy", "--snapshot-times-s", "30,45")
    for name in (ee.SUMMARY_NAME, ee.FINAL_NAME, ee.HYDROGRAPH_NPZ, ee.HYDROGRAPH_CSV, ee.SNAPSHOTS_NAME):
        assert (out / name).is_file(), name
    assert "EXPERIMENTAL" in summary["status"] and "not equivalent" in summary["status"] and summary["solver"] == solver
    budget = summary["budget"]
    assert abs(budget["water_residual_m3"]) <= budget["bound_m3"] and abs(budget["surface_residual_m3"]) <= budget["bound_m3"]
    assert abs(budget["soil_residual_m3"]) <= budget["bound_m3"] and "volume_roundoff_bound_m3" in budget["bound_rule"]
    assert summary["maple_state"]["unchanged"] is True and summary["provenance"]["source_stability"]["stable"] is True
    assert summary["hydraulics"]["method"] == solver and summary["backend"]["backend"] == "numpy"
    assert summary["numerics"]["cfl_max"] == 0.5 and summary["numerics"]["limiter"] == "off"
    assert "Darcy" in summary["boundary"] and "no sediment" in summary["status"]
    assert summary["time"]["snapshot_times_s"] == [30.0, 45.0]
    final = np.load(out / ee.FINAL_NAME)
    assert ("qx_m2_s" in final.files) == (solver == "local_inertial") and np.all(final["depth_m"] >= 0.0)
    # the portable numeric state is not only the arrays: the time and the adaptive step cap are saved and reported
    continuation = summary["final_state"]["continuation"]
    assert float(final["state_t_s"]) == 90.0 == continuation["state_t_s"]
    assert 0.0 < float(final["state_next_dt_cap_s"]) <= 1.0
    assert continuation["state_next_dt_cap_s"] == float(final["state_next_dt_cap_s"])
    assert "state_next_dt_cap_s" in continuation["numerical_state"] and "not implemented" in continuation["disk_restart"]
    assert summary["hydraulics"]["qualification"] and "NOT" in summary["hydraulics"]["qualification"]
    snaps =np.load(out / ee.SNAPSHOTS_NAME)
    assert {"t30_depth_m", "t30_velocity_m_s", "t45_depth_m"} <= set(snaps.files)
    rows = np.load(out / ee.HYDROGRAPH_NPZ)
    assert rows["t_s"][-1] == 90.0 and np.all(np.abs(rows["row_water_residual_m3"]) <= budget["bound_m3"])
    assert summary["hydraulics"]["cfl_kind"] and "limited_cells_total" in summary["numerics"]


@pytest.mark.parametrize("solver", SOLVERS)
def test_snapshot_times_do_not_alter_the_forcing_or_the_totals(plot1_case, tmp_path, solver):
    a = run_cli(plot1_case, tmp_path / "a", "--solver", solver, "--backend", "numpy")
    b = run_cli(plot1_case, tmp_path / "b", "--solver", solver, "--backend", "numpy", "--snapshot-times-s", "30,45")
    assert a["budget"]["rain_m3"] == pytest.approx(b["budget"]["rain_m3"], rel=1e-13)  # same schedule integral


def test_the_cli_refuses_an_existing_output_and_a_limiter_for_the_explicit_solver(plot1_case, tmp_path, capsys):
    existing = tmp_path / "existing"
    existing.mkdir()
    assert ee.main(["--case-dir", str(plot1_case), "--output-dir", str(existing), "--solver", "explicit"]) == 1
    assert "refusing to write into existing path" in capsys.readouterr().err
    assert ee.main(["--case-dir", str(plot1_case), "--output-dir", str(tmp_path / "x"), "--solver", "explicit",
                    "--limiter", "donor"]) == 1
    assert not (tmp_path / "x").exists()


@pytest.mark.parametrize("solver", SOLVERS)
def test_short_plot1_run_on_cuda_matches_the_numpy_reference_and_records_actual_metadata(plot1_case, tmp_path, gpu, solver):
    cpu = run_cli(plot1_case, tmp_path / "cpu", "--solver", solver, "--backend", "numpy", "--snapshot-times-s", "45")
    gpu_summary = run_cli(plot1_case, tmp_path / "gpu", "--solver", solver, "--backend", "cupy", "--snapshot-times-s", "45")
    for name in (ee.FINAL_NAME, ee.HYDROGRAPH_NPZ, ee.SNAPSHOTS_NAME):
        a, b = np.load(tmp_path / "cpu" / name), np.load(tmp_path / "gpu" / name)
        assert sorted(a.files) == sorted(b.files), name
        for key in a.files:
            assert a[key].shape == b[key].shape, (name, key)
            np.testing.assert_allclose(b[key], a[key], rtol=RTOL, atol=ATOL, err_msg=f"{name}:{key}")
    for key in ("n_accepted_steps", "n_rejected_attempts", "n_boundaries", "dt_min_accepted_s", "dt_max_accepted_s"):
        assert gpu_summary["time"][key] == cpu["time"][key], key
    assert gpu_summary["final_state"]["time_of_peak_outlet_discharge_s"] == cpu["final_state"]["time_of_peak_outlet_discharge_s"]
    assert gpu_summary["numerics"]["limited_cells_total"] == cpu["numerics"]["limited_cells_total"]
    backend = gpu_summary["backend"]
    assert backend["backend"] == "cupy" and backend["counted_packet_reads_per_attempt"] == 2
    attempts = gpu_summary["time"]["n_accepted_steps"] + gpu_summary["time"]["n_rejected_attempts"]
    loop = backend["loop_transfer_counters"]
    assert 0 <= loop["device_to_host"] - 2 * attempts <= 8 and loop["host_to_device_bytes"] == 0
    assert backend["preparation"]["transfer_counters"]["device_to_host_bytes"] > 0  # one-time static validation download
    assert backend["reporting_transfer_counters"]["device_to_host_bytes"] > 0  # explicit final downloads, counted apart
    context = gpu_summary["hydraulics"]["context"]
    assert context["method"] == solver and context["numba_required"] is False and context["fastmath"] is False
    assert context["column_context"]["mode"] in ("fused", "split")
    assert gpu_summary["hydraulics"]["kernels"]["fastmath"] is False
    assert len(gpu_summary["hydraulics"]["kernels"]["experimental_source_sha256"]) == 64
    assert "experimental" in backend["gpu"] and "no performance claim" in backend["gpu"]
