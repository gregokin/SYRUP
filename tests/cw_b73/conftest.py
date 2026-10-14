"""Bind the CW-B73 private tree's `maple_syrup` for tests/cw_b73 and expose the existing actual-MAPLE bed builders
and pit fixtures (tests/phase5, tests/conservative_water, tests/conservative_pits) of THIS tree."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SRC = CANDIDATE_ROOT / "src"
sys.path.insert(0, str(CANDIDATE_SRC))
for i, sub in enumerate(("phase5", "conservative_water", "conservative_pits"), start=1):
    sys.path.insert(i, str(CANDIDATE_ROOT / "tests" / sub))

import maple_syrup

_PACKAGE_DIR = Path(maple_syrup.__file__).resolve().parent
if _PACKAGE_DIR != (CANDIDATE_SRC / "maple_syrup").resolve():
    raise RuntimeError(f"tests/cw_b73 must exercise {CANDIDATE_SRC}, but maple_syrup was imported from {_PACKAGE_DIR}")


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip (a skip claims nothing about the device)."""
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the device variant is not exercised")
    return backend.cupy_module()
