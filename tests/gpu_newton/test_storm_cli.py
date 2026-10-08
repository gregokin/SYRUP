"""`--root-solver newton` through the water storm CLI on the actual Plot 1 case (short run): CUDA Newton against the CPU
Newton CLI (every saved array within rtol 2e-12 / atol 1e-14), metadata, startup kept out of the loop, the unchanged
default and the refusals before anything is read or written."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup import storm_experiment as se

pytestmark = pytest.mark.usefixtures("gpu")
RTOL, ATOL = 2.0e-12, 1.0e-14


def run_cli(case, out, *extra) -> dict:
    argv = ["--case-dir", str(case), "--output-dir", str(out), "--end-s", "120", "--report-every-s", "60",
            "--allow-maple-source-change", *extra]
    assert se.main(argv) == 0
    return json.loads((out / se.SUMMARY_NAME).read_text())


def test_cuda_newton_cli_matches_the_cpu_newton_cli_and_records_the_solver(plot1_case, tmp_path):
    cpu = run_cli(plot1_case, tmp_path / "cpu", "--implementation", "array", "--backend", "numpy",
                  "--root-solver", "newton")
    gpu = run_cli(plot1_case, tmp_path / "cuda", "--implementation", "cuda", "--backend", "cupy",
                  "--root-solver", "newton", "--newton-max-iterations", "50")
    for name in ("final_water.npz", "hydrograph.npz"):
        a, b = np.load(tmp_path / "cpu" / name), np.load(tmp_path / "cuda" / name)
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            np.testing.assert_allclose(b[key], a[key], rtol=RTOL, atol=ATOL, err_msg=f"{name}:{key}")
    for key in ("n_accepted_steps", "n_rejected_attempts", "n_boundaries"):
        assert gpu["time"][key] == cpu["time"][key]
    assert gpu["routing"]["root_solver"] == cpu["routing"]["root_solver"] == "newton"
    assert gpu["routing"]["newton_max_iterations"] == 50
    for summary in (cpu, gpu):
        method = summary["routing"]["method"]
        assert "safeguarded Newton" in method and "h + c k h^{3/2} = R" in method
        assert "bisection on [0, R]" not in method
    block = gpu["cuda_hydrology"]
    assert block["preparation"]["newton_kernel_load_s"] >= 0.0  # startup recorded apart from the loop
    assert len(block["kernels"]["newton_hydrology_source_sha256"]) == 64
    assert block["kernels"]["fastmath"] is False and "--fmad=false" in block["kernels"]["compile_options"]
    attempts = gpu["time"]["n_accepted_steps"] + gpu["time"]["n_rejected_attempts"]
    loop = block["loop_transfer_counters"]
    assert 0 <= loop["device_to_host"] - attempts <= 8 and loop["host_to_device_bytes"] == 0


def test_default_summaries_have_no_solver_keys_and_default_cuda_is_bisection(plot1_case, tmp_path):
    summary = run_cli(plot1_case, tmp_path / "default", "--implementation", "cuda", "--backend", "cupy")
    assert "root_solver" not in summary["routing"] and "newton_kernel_load_s" not in summary["cuda_hydrology"][
        "preparation"]
    assert summary["routing"]["method"] == (
        "MAHLERAN method 5 (Crank-Nicolson, bisection on [0, R], coherent old inflow)")


def test_invalid_solver_options_are_refused_before_anything_is_written(plot1_case, tmp_path, capsys):
    out = tmp_path / "bad"
    for extra in (["--newton-max-iterations", "0"], ["--newton-max-iterations", "1001"]):
        assert se.main(["--case-dir", str(plot1_case), "--output-dir", str(out), "--end-s", "60",
                        "--implementation", "cuda", "--backend", "cupy", "--root-solver", "newton", *extra]) == 1
        assert "newton_max_iterations" in capsys.readouterr().err and not out.exists()
    with pytest.raises(SystemExit):  # argparse refuses an unknown solver name
        se.main(["--case-dir", str(plot1_case), "--output-dir", str(out), "--root-solver", "brent"])
    # a CPU implementation with the cupy backend and Newton is refused, never solved on the host
    assert se.main(["--case-dir", str(plot1_case), "--output-dir", str(out), "--end-s", "60",
                    "--implementation", "cuda", "--backend", "numpy", "--root-solver", "newton"]) == 1
    assert not out.exists()
