"""HISTORICAL author-tree SOURCE AUDIT of the GPU lifecycle task (preserved VERBATIM; NOT collected by default: the filename is not `test_*.py`). It compares this tree's production files with the lifecycle baseline manifest
(`event_lifecycle_cuda_baseline.json`: main161 + frozen GPU refresh) and holds only in the lifecycle author tree; after the legitimate composition with the CPU daily bridge (new event_daily_bridge_cuda tree) it is EXPECTED to differ.
Run explicitly only against the matching frozen lifecycle snapshot. The per-interface replacement is `tests/phase2_floating_event_daily_bridge_cuda/test_cuda_bridge_host.py::test_production_diff_against_the_bridge_cuda_baseline`.
Runtime SourcePair/native/kernel guards are never disabled."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def test_production_diff_against_the_lifecycle_baseline():
    """Per-interface replacement of the author-refresh-tree audit: against the lifecycle baseline manifest (main161 + frozen GPU refresh) ONLY the four declared production files differ/are new; the refresh production files and
    every kernel/source module are byte-identical to the baseline (no weakening of the runtime source-pair/kernel guards)."""
    here = Path(__file__).resolve()
    task = here.parents[3]
    manifest = task / "completion_program_20261113" / "event_lifecycle_cuda_baseline.json"
    if not manifest.exists():
        pytest.skip("lifecycle baseline manifest not present")
    src = here.parents[2] / "src" / "maple_syrup"
    rel = {k.split("/")[-1]: v for k, v in json.loads(manifest.read_text()).items() if k.startswith("src/maple_syrup/") and k.endswith(".py")}
    differing = {n for n, h in rel.items() if hashlib.sha256((src / n).read_bytes()).hexdigest() != h}
    present = {p.name for p in src.glob("*.py")}
    assert differing == {"floating_event_lifecycle.py", "floating_pond.py", "floating_pond_cuda.py"} and present - set(rel) == {"floating_event_lifecycle_cuda.py"}
    for name in ("floating_event_refresh_cuda.py", "floating_event_refresh.py", "native_floating_soil_cuda.py", "floating_soil_layers_cuda.py", "floating_checkpoint_cuda.py", "floating_completion.py"):
        assert hashlib.sha256((src / name).read_bytes()).hexdigest() == rel[name]
