"""HISTORICAL author-tree SOURCE AUDIT of the 166-module CUDA event<->daily bridge tree (preserved VERBATIM; NOT collected by default: the filename is not `test_*.py`). It compares the tree with
`event_daily_bridge_cuda_baseline.json` (accepted main164 + CPU_C3 overlay) and holds only in the 166-module author tree; after the legitimate composition with the separately accepted continued-CUDA codec (167 modules) it is
EXPECTED to differ. The per-interface replacement is `test_production_diff_against_the_167_composition_plan`. Runtime SourcePair/native/kernel guards are never disabled."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def test_production_diff_against_the_bridge_cuda_baseline():
    """Per-interface proof against `event_daily_bridge_cuda_baseline.json` (accepted main164 + the exact CPU_C3 overlay): ONLY the four declared owning/bridge modules differ and ONLY the new CUDA bridge helper is added; the CPU_C3
    producer proof (pond_event), the daily state/output, every kernel/provider/native/child-codec module and the accepted GPU refresh are byte-identical."""
    here = Path(__file__).resolve()
    manifest = here.parents[3] / "completion_program_20261113" / "event_daily_bridge_cuda_baseline.json"
    if not manifest.exists():
        pytest.skip("bridge cuda baseline manifest not present")
    src = here.parents[2] / "src" / "maple_syrup"
    rel = {k.split("/")[-1]: v for k, v in json.loads(manifest.read_text())["files"].items() if k.startswith("src/maple_syrup/") and k.endswith(".py")}
    differing = {n for n, h in rel.items() if hashlib.sha256((src / n).read_bytes()).hexdigest() != h}
    present = {p.name for p in src.glob("*.py")}
    assert differing == {"floating_event_lifecycle.py", "floating_event_lifecycle_cuda.py", "floating_pond.py", "floating_event_daily_bridge.py"}
    assert present - set(rel) == {"floating_event_daily_bridge_cuda.py"}
    for name in ("pond_event.py", "floating_daily.py", "floating_daily_output.py", "floating_pond_cuda.py", "floating_event_refresh_cuda.py", "native_floating_soil_cuda.py", "floating_soil_layers_cuda.py",
                 "floating_daily_cuda.py", "floating_completion.py"):
        assert hashlib.sha256((src / name).read_bytes()).hexdigest() == rel[name], name
