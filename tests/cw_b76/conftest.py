"""Bind the CW-B76 private tree's `maple_syrup` for tests/cw_b76 and expose the existing actual-MAPLE builders and the
CW-B20 accounting harness (tests/phase5, tests/conservative_water, tests/cw_b20, tests/conservative_pits) of THIS
tree."""

from __future__ import annotations

import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SRC = CANDIDATE_ROOT / "src"
sys.path.insert(0, str(CANDIDATE_SRC))
for i, sub in enumerate(("phase5", "conservative_water", "cw_b20", "conservative_pits"), start=1):
    sys.path.insert(i, str(CANDIDATE_ROOT / "tests" / sub))

import maple_syrup

_PACKAGE_DIR = Path(maple_syrup.__file__).resolve().parent
if _PACKAGE_DIR != (CANDIDATE_SRC / "maple_syrup").resolve():
    raise RuntimeError(f"tests/cw_b76 must exercise {CANDIDATE_SRC}, but maple_syrup was imported from {_PACKAGE_DIR}")
