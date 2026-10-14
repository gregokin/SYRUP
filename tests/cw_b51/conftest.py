"""Bind this tree's `maple_syrup` (its `src`, placed FIRST) and reuse the existing actual-MAPLE helper suites.

Portable: the check compares the imported package with this tree's own `src` (located relative to this file), never a
hard-coded folder name. GPU tests skip without a CUDA device (no claim).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]
SRC = TREE / "src"
sys.path.insert(0, str(SRC))
for suite in ("phase5", "phase6", "conservative_water", "conservative_pits"):
    sys.path.insert(0, str(TREE / "tests" / suite))

import maple_syrup  # noqa: E402

if Path(maple_syrup.__file__).resolve().parent != (SRC / "maple_syrup").resolve():
    raise RuntimeError(f"tests/cw_b51 must exercise {SRC}, but maple_syrup came from {maple_syrup.__file__}")


@pytest.fixture(scope="session")
def gpu():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the device variant is not exercised (no claim)")
    return backend.cupy_module()
