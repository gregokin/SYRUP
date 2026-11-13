"""E5 host-double plumbing (NOT hardware): reuses the E3b native-CUDA host fixtures (`libs`, `cp`) by path."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

E3C = Path(__file__).resolve().parents[1] / "phase2_native_floating_soil_cuda"
sys.path.insert(0, str(E3C))
_spec = importlib.util.spec_from_file_location("e3b_plumbing_conftest", E3C / "conftest.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
libs, cp, needs_host_cuda, Counter = _mod.libs, _mod.cp, _mod.needs_host_cuda, _mod.Counter
