"""D3 suite plumbing: reuses the D2 host-compiled CUDA plumbing (NOT hardware) by path."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

D2 = Path(__file__).resolve().parents[1] / "phase2_floating_intake_withdrawal"
sys.path[:0] = [str(D2), str(Path(__file__).resolve().parent)]
_spec = importlib.util.spec_from_file_location("d2_plumbing_conftest", D2 / "conftest.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
libs, cp, needs_host_cuda, Counter = _mod.libs, _mod.cp, _mod.needs_host_cuda, _mod.Counter
