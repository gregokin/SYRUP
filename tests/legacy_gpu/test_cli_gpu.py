"""GPU driver controls (CPU-only) and the actual `python -m maple_syrup.legacy_gpu_driver` against the CPU driver (real GPU; queued for
Codex on a verified-idle device). Written without being run."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from maple_syrup import legacy_gpu_driver as G

from .helpers import gpu_available

ROOT = Path(__file__).resolve().parents[2]
PLOT1 = ROOT / "outputs" / "plot1"
APPLIED = ROOT / "outputs" / "phase7" / "mahleran_reference_audit" / "applied_rainfall.csv"
needs_plot1 = pytest.mark.skipif(not (PLOT1 / "syrup" / "plot1_binding.json").is_file() or not APPLIED.is_file(),
                                 reason="outputs/plot1 or the applied-rainfall CSV is absent")
needs_gpu = pytest.mark.skipif(not gpu_available(), reason="no CuPy / CUDA device")


def argv(tmp_path, *extra, out="out"):
    return ["--case-kind", "plot1", "--case", str(tmp_path / "no_case"), "--output", str(tmp_path / out), *extra]


def test_parser_defaults_follow_the_cpu_driver_and_add_the_gpu_controls():
    args = G.build_parser().parse_args(["--case-kind", "rfid", "--case", "c", "--output", "o"])
    assert args.depth_time_level == "previous" and args.source_order == "index" and not args.legacy_depos_erase
    assert args.check_every_steps == 60 and args.cuda_mode == "auto" and args.cn_mode == "auto" and args.gpu_memory_gib is None
    assert args.root_solver == "bisection"


@pytest.mark.parametrize("extra, match", [
    (["--source-order", "legacy"], "source-order legacy"), (["--legacy-depos-erase", "--source-order", "legacy"], "source-order legacy"),
    (["--allow-python-kernels"], "no CPU fallback"), (["--check-every-steps", "0"], "check-every-steps"),
    (["--gpu-memory-gib", "0"], "gpu-memory-gib"), (["--gpu-memory-gib", "nan"], "gpu-memory-gib"),
    (["--end-s", "10.5"], "whole number"), (["--bisection-iterations", "0"], "hydrology control"),
    (["--max-memory-gib", "-1"], "max-memory-gib"),
])
def test_unsupported_modes_and_bad_controls_are_rejected_before_any_work(tmp_path, monkeypatch, extra, match):
    monkeypatch.setattr(G, "_cupy", lambda: (_ for _ in ()).throw(AssertionError("the device must not be touched")))
    monkeypatch.setattr(G, "legacy_case_for", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the case adapter must not run")))
    args = G.build_parser().parse_args(argv(tmp_path, *extra))
    with pytest.raises(G.D.DriverError, match=match):
        G.validate_gpu_controls(args)
    assert G.main(argv(tmp_path, *extra)) == 1
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists() and not (tmp_path / "out.FAILED").exists()


def test_legacy_erasure_is_never_silently_ignored():
    args = G.build_parser().parse_args(["--case-kind", "rfid", "--case", "c", "--output", "o", "--legacy-depos-erase"])
    with pytest.raises(G.D.DriverError):  # either the CPU rule (needs legacy order) or the GPU rule: always an explicit refusal
        G.validate_gpu_controls(args)


def test_missing_cupy_is_an_explicit_failure_with_failed_evidence_and_no_cpu_fallback(tmp_path, monkeypatch):
    from maple_syrup.routing_cuda import CudaUnavailableError

    def unavailable():
        raise CudaUnavailableError("CuPy is not importable; no fallback")

    monkeypatch.setattr(G, "_cupy", unavailable)
    assert G.main(argv(tmp_path)) == 1
    assert not (tmp_path / "out").exists()
    record = json.loads((tmp_path / "out.FAILED" / "FAILED.json").read_text())
    assert "CudaUnavailableError" in record["error"]


def test_protected_output_is_refused_before_any_write(tmp_path):
    target = ROOT / "src" / "maple_syrup_gpu_should_not_exist_out"
    assert G.main(["--case-kind", "plot1", "--case", str(tmp_path / "c"), "--output", str(target)]) == 1
    assert not target.exists() and not target.with_name(target.name + ".FAILED").exists()


def test_module_digests_include_the_gpu_modules_from_their_files():
    digests = G.module_digests()
    assert {"maple_syrup.legacy_native_cuda", "maple_syrup.legacy_gpu_driver"} <= set(digests)
    assert all(len(v) == 64 for v in digests.values())


def run_cli(module, tmp_path, out, *extra):
    command = [sys.executable, "-m", module, "--case-kind", "plot1", "--case", str(PLOT1), "--output", str(tmp_path / out), "--end-s", "40",
               "--applied-rainfall", str(APPLIED), "--allow-maple-source-change", "--progress-every-s", "0", *extra]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=3600, check=False)


@needs_gpu
@needs_plot1
@pytest.mark.parametrize("depth_mode, solver", [("previous", "bisection"), ("post_infiltration", "bisection"),
                                                ("previous", "newton")])
def test_actual_gpu_cli_matches_the_cpu_cli_on_plot1_within_the_declared_bounds(tmp_path, depth_mode, solver):
    extra = ["--depth-time-level", depth_mode, "--root-solver", solver]  # same solver on both sides (same-solver comparison)
    cpu = run_cli("maple_syrup.legacy_driver", tmp_path, "cpu", *extra)
    gpu = run_cli("maple_syrup.legacy_gpu_driver", tmp_path, "gpu", *extra)
    assert cpu.returncode == 0, cpu.stderr[-2000:]
    assert gpu.returncode == 0, gpu.stderr[-3000:]
    c, g = np.load(tmp_path / "cpu" / "legacy_ledger.npz"), np.load(tmp_path / "gpu" / "legacy_ledger.npz")
    cs = json.loads((tmp_path / "cpu" / "legacy_summary.json").read_text())
    gs = json.loads((tmp_path / "gpu" / "legacy_summary.json").read_text())
    assert gs["backend"] == "cuda" and gs["depth_time_level"] == depth_mode and gs["totals"]["pickup_kg"] > 0.0
    assert gs["water"]["whole_storm_budget"]["closed"] is True and gs["identity"]["worst_relative_residual"] < gs["identity"]["guard_rtol"]
    assert gs["onset"]["first_positive_pickup_s"] == cs["onset"]["first_positive_pickup_s"]
    # same-solver full-field comparison at the predeclared sediment bounds; integer tallies exact
    for key in ("ledger", "cumulative_detachment_kg", "cumulative_deposition_kg", "cumulative_clipping_source_kg", "final_mobile_kg"):
        np.testing.assert_allclose(g[key], c[key], rtol=2e-11, atol=1e-14, err_msg=key)
    assert np.array_equal(g["walk_counts"][:, :9], c["walk_counts"][:, :9]) and np.array_equal(g["regime_counts"], c["regime_counts"])
    for key in ("final_depth_m", "final_soil_water_m", "final_discharge_m2_s"):
        np.testing.assert_allclose(g[key], c[key], rtol=2e-12, atol=1e-14, err_msg=key)  # the water bound, unchanged
    assert gs["hydrology"]["root_solver"] == solver
    assert "previous_sediment_end_to_hydrology_end_s" in gs["performance"]["event_split_s"]  # honest segment labels
    assert gs["gpu"]["transfers"]["d2h_driver_final_water_bytes"] > 0 and "scope" in gs["gpu"]["transfers"]
    assert gs["gpu"]["transfers"]["h2d_static_bytes"] > 0 and gs["gpu"]["memory"]["estimate_bytes"]["total"] > 0
