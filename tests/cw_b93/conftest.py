"""Bind the CW-B93 private tree's `maple_syrup` for tests/cw_b93 and expose the existing CUDA CSR harness (cw_b52,
cw_b54) and the actual-MAPLE accounting builders (phase5, conservative_water, cw_b20, conservative_pits, cw_b76) of
THIS tree."""

from __future__ import annotations

import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SRC = CANDIDATE_ROOT / "src"
sys.path.insert(0, str(CANDIDATE_SRC))
for i, sub in enumerate(("cw_b52", "cw_b54", "phase5", "conservative_water", "cw_b20", "conservative_pits", "cw_b76"),
                        start=1):
    sys.path.insert(i, str(CANDIDATE_ROOT / "tests" / sub))

import maple_syrup

_PACKAGE_DIR = Path(maple_syrup.__file__).resolve().parent
if _PACKAGE_DIR != (CANDIDATE_SRC / "maple_syrup").resolve():
    raise RuntimeError(f"tests/cw_b93 must exercise {CANDIDATE_SRC}, but maple_syrup was imported from {_PACKAGE_DIR}")
