"""Bind the CW-B5 private candidate tree and reuse the actual-MAPLE event fixtures of the other suites.

The private copy lives beside these tests (outputs/conservative_water/csr_event_candidate_source/src) and is placed
FIRST on `sys.path`, then checked to be the package imported; the Phase 5 / Phase 6 / conservative_water /
conservative_pits test helpers (real MAPLE beds, event setup in either namespace, closure and cross-backend
comparators, pit fixtures) are reused, never copied.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SRC = CANDIDATE_ROOT / "src"
sys.path.insert(0, str(CANDIDATE_SRC))
for suite in ("phase5", "phase6", "conservative_water", "conservative_pits"):
    sys.path.insert(0, str(CANDIDATE_ROOT / "tests" / suite))

import maple_syrup

_PACKAGE_DIR = Path(maple_syrup.__file__).resolve().parent
if _PACKAGE_DIR != (CANDIDATE_SRC / "maple_syrup").resolve():
    raise RuntimeError(
        f"tests/conservative_csr_event must exercise the candidate tree {CANDIDATE_SRC}, but maple_syrup was imported "
        f"from {_PACKAGE_DIR}; a production or installed copy shadows the candidate (check PYTHONPATH / editable "
        "installs)"
    )


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip (a skipped GPU test makes no claim). GPU 0 is the user allocation; select
    the device with CUDA_VISIBLE_DEVICES before the process starts."""
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CSR coupled event on CUDA is not exercised (no claim)")
    return backend.cupy_module()
