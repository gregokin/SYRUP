"""Bind the CW-B63 private tree's `maple_syrup` for tests/cw_b63 and expose the corrected CW-B52 / CW-B54 / CW-B59
harness helpers of THIS tree."""

from __future__ import annotations

import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SRC = CANDIDATE_ROOT / "src"
sys.path.insert(0, str(CANDIDATE_SRC))
for i, suite in enumerate(("cw_b52", "cw_b54", "cw_b59"), start=1):
    sys.path.insert(i, str(CANDIDATE_ROOT / "tests" / suite))

import maple_syrup

_PACKAGE_DIR = Path(maple_syrup.__file__).resolve().parent
if _PACKAGE_DIR != (CANDIDATE_SRC / "maple_syrup").resolve():
    raise RuntimeError(f"tests/cw_b63 must exercise {CANDIDATE_SRC}, but maple_syrup was imported from {_PACKAGE_DIR}")
