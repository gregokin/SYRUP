"""The ACTUAL command line (`python -m maple_syrup.legacy_driver`), end to end in a fresh process.

Calling `run()` from a test cannot reach the `__main__` path where the module provenance once failed after a full 600 s Chastre run;
these tests start a real subprocess (written without being run; Codex executes them)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from maple_syrup import legacy_driver as D

ROOT = Path(__file__).resolve().parents[2]
PLOT1 = ROOT / "outputs" / "plot1"
APPLIED = ROOT / "outputs" / "phase7" / "mahleran_reference_audit" / "applied_rainfall.csv"

needs_plot1 = pytest.mark.skipif(
    importlib.util.find_spec("numba") is None or not (PLOT1 / "syrup" / "plot1_binding.json").is_file() or not APPLIED.is_file(),
    reason="Numba, outputs/plot1 or the applied-rainfall CSV is absent")


def run_cli(tmp_path, *extra, case=PLOT1, out="out", end="40", applied=True):
    command = [sys.executable, "-m", "maple_syrup.legacy_driver", "--case-kind", "plot1", "--case", str(case),
               "--output", str(tmp_path / out), "--end-s", end, "--allow-maple-source-change", "--progress-every-s", "0"]
    if applied:
        command += ["--applied-rainfall", str(APPLIED)]
    return subprocess.run([*command, *extra], cwd=ROOT, capture_output=True, text=True, timeout=3600, check=False)


@needs_plot1
def test_actual_python_m_cli_publishes_a_complete_summary_with_the_driver_file_digest(tmp_path):
    done = run_cli(tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    out = tmp_path / "out"
    assert out.is_dir() and not (tmp_path / "out.FAILED").exists() and not (tmp_path / "out.partial").exists()
    summary = json.loads((out / "legacy_summary.json").read_text())
    driver_file = Path(D.__file__).resolve()
    modules = summary["provenance"]["module_sha256"]
    assert modules["maple_syrup.legacy_driver"] == hashlib.sha256(driver_file.read_bytes()).hexdigest()
    assert set(modules) == set(D._PROVENANCE_MODULES) and all(len(v) == 64 for v in modules.values())
    assert summary["totals"]["pickup_kg"] > 0.0 and summary["totals"]["deposition_active_kg"] > 0.0  # nonvacuous sediment
    assert summary["water"]["whole_storm_budget"]["closed"] is True
    assert summary["identity"]["worst_relative_residual"] < summary["identity"]["guard_rtol"]
    assert summary["artifact_pins"]["checked_unchanged_before_publish"] is True
    assert summary["provenance"]["source_digests_before"] == summary["provenance"]["source_digests_after"]
    assert summary["depth_time_level"] == "previous" and summary["law_depth"]["native_original_option"] is False
    data = np.load(out / "legacy_ledger.npz")
    assert data["ledger"].shape[0] == 40 and float(data["cumulative_detachment_kg"].sum()) > 0.0


@needs_plot1
def test_actual_cli_native_post_infiltration_option_is_selectable_and_recorded(tmp_path):
    done = run_cli(tmp_path, "--depth-time-level", "post_infiltration")
    assert done.returncode == 0, done.stderr[-3000:]
    summary = json.loads((tmp_path / "out" / "legacy_summary.json").read_text())
    assert summary["depth_time_level"] == "post_infiltration"
    assert summary["law_depth"]["native_original_option"] is True and "infilt.for" in summary["law_depth"]["meaning"]
    assert any("post-infiltration" in s for s in summary["limitations"])
    assert summary["totals"]["pickup_kg"] > 0.0 and summary["water"]["whole_storm_budget"]["closed"] is True
    assert summary["config"]["depth_time_level"] == "post_infiltration"


def test_actual_cli_failure_is_atomic_and_leaves_failed_evidence_only(tmp_path):
    done = run_cli(tmp_path, case=tmp_path / "no_such_case", applied=False)
    assert done.returncode == 1
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists()
    record = json.loads((tmp_path / "out.FAILED" / "FAILED.json").read_text())
    assert record["error"] and "traceback" in record


def test_actual_cli_refuses_a_protected_output_before_writing_anything(tmp_path):
    target = ROOT / "src" / "maple_syrup_should_not_exist_out"
    done = subprocess.run([sys.executable, "-m", "maple_syrup.legacy_driver", "--case-kind", "plot1", "--case", str(tmp_path / "c"),
                           "--output", str(target), "--allow-python-kernels"],
                          cwd=ROOT, capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 1
    assert not target.exists() and not target.with_name(target.name + ".FAILED").exists()
    assert not target.with_name(target.name + ".partial").exists()
