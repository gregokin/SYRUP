"""Bind the CW-B54 private tree's `maple_syrup` (never another copy) for tests/cw_b54, and make the corrected CW-B52
harness helpers (tests/cw_b52, same tree) importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SRC = CANDIDATE_ROOT / "src"
sys.path.insert(0, str(CANDIDATE_SRC))
sys.path.insert(1, str(CANDIDATE_ROOT / "tests" / "cw_b52"))

import maple_syrup

_PACKAGE_DIR = Path(maple_syrup.__file__).resolve().parent
if _PACKAGE_DIR != (CANDIDATE_SRC / "maple_syrup").resolve():
    raise RuntimeError(f"tests/cw_b54 must exercise {CANDIDATE_SRC}, but maple_syrup was imported from {_PACKAGE_DIR}")
