"""Phase 4S Task B: the explicit `--implementation cuda --backend cupy` water CLI on the actual Plot 1 case, short run
(the full 5400 s qualification is done by the root scripts, not here), against the CPU `array` run of the same CLI.

The comparison covers every saved array and hydrograph column within rtol 2e-12 / atol 1e-14, exact step counts, the
unchanged budget checks (the CLI itself refuses to write on any failed budget), the recorded CUDA context/kernel metadata,
and the counted loop transfers (one packet per attempt). Skipped without a device or the Plot 1 case. Nothing here was run
by its author (file-only tools); Codex records results.
"""
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


def test_short_cli_run_cuda_matches_the_cpu_cli_and_records_actual_metadata(plot1_case, tmp_path):
    cpu = run_cli(plot1_case, tmp_path / "cpu", "--implementation", "array", "--backend", "numpy")
    gpu = run_cli(plot1_case, tmp_path / "cuda", "--implementation", "cuda", "--backend", "cupy")
    for name in ("final_water.npz", "hydrograph.npz"):
        a, b = np.load(tmp_path / "cpu" / name), np.load(tmp_path / "cuda" / name)
        assert sorted(a.files) == sorted(b.files), name
        for key in a.files:
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, (name, key)
            np.testing.assert_allclose(b[key], a[key], rtol=RTOL, atol=ATOL, err_msg=f"{name}:{key}")
    for key in ("n_accepted_steps", "n_rejected_attempts", "n_boundaries", "dt_min_accepted_s", "dt_max_accepted_s"):
        assert gpu["time"][key] == cpu["time"][key], key
    for key, value in cpu["budget"].items():
        if isinstance(value, float):
            assert gpu["budget"][key] == pytest.approx(value, rel=1e-9, abs=1e-12), key
    assert gpu["final_state"]["time_of_peak_outlet_discharge_s"] == cpu["final_state"]["time_of_peak_outlet_discharge_s"]
    assert gpu["final_state"]["peak_outlet_discharge_m3_s"] == pytest.approx(
        cpu["final_state"]["peak_outlet_discharge_m3_s"], rel=RTOL, abs=ATOL)
    assert gpu["routing"]["cell_steps"] == cpu["routing"]["cell_steps"]
    assert gpu["provenance"]["implementation"] == "cuda" and gpu["backend"]["backend"] == "cupy"
    # actual context / kernel metadata and separately counted transfers
    block = gpu["cuda_hydrology"]
    ctx = block["context"]
    assert ctx["mode"] in ("fused", "split") and ctx["n_active"] == gpu["domain"]["active_cells"]
    assert ctx["device_resident"] is True and ctx["numba_required"] is False and ctx["fastmath"] is False
    assert block["kernels"]["fastmath"] is False and len(block["kernels"]["hydrology_source_sha256"]) == 64
    assert block["preparation"]["transfer_counters"]["device_to_host_bytes"] > 0  # the one-time static download
    assert block["preparation"]["transfer_counters"]["host_to_device_bytes"] >= ctx["static_bytes"]
    attempts = gpu["time"]["n_accepted_steps"] + gpu["time"]["n_rejected_attempts"]
    loop = block["loop_transfer_counters"]
    assert loop == gpu["backend"]["loop_transfer_counters"]
    assert 0 <= loop["device_to_host"] - attempts <= 8  # one packet per attempt + the entry validation read(s)
    assert loop["device_to_host_bytes"] - attempts * ctx["packet_bytes"] < 4096 and loop["host_to_device_bytes"] == 0
    assert block["reporting_transfer_counters"]["device_to_host_bytes"] > 0  # explicit final downloads, separate
    assert "cuda_hydrology" not in cpu
    assert "water-only" in gpu["backend"]["gpu"] and "no GPU sediment" in gpu["backend"]["gpu"]


def test_cuda_cli_refuses_numpy_backend_and_existing_output_without_writing(plot1_case, tmp_path, capsys):
    out = tmp_path / "refused"
    assert se.main(["--case-dir", str(plot1_case), "--output-dir", str(out), "--end-s", "60",
                    "--implementation", "cuda", "--backend", "numpy"]) == 1
    assert "--backend cupy" in capsys.readouterr().err and not out.exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    assert se.main(["--case-dir", str(plot1_case), "--output-dir", str(existing), "--end-s", "60",
                    "--implementation", "cuda", "--backend", "cupy"]) == 1
    assert "refusing to write into existing path" in capsys.readouterr().err
    assert list(existing.iterdir()) == []


def test_cpu_defaults_are_unchanged_by_the_cuda_option(plot1_case, tmp_path):
    summary = run_cli(plot1_case, tmp_path / "default", "--implementation", "array", "--backend", "numpy")
    assert "cuda_hydrology" not in summary
    assert "two batched flag reads per attempt" in summary["backend"]["validation"]
    assert summary["provenance"]["implementation"] == "array"
