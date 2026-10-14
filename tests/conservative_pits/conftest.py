import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Reuse the actual-MAPLE bed constructors of the Phase 5 event tests and the either-namespace helpers, closure and
# bitwise comparators of the CW-A1 / CW-C1 test modules (same candidate tree).
sys.path.insert(0, str(ROOT / "tests" / "phase5"))
sys.path.insert(0, str(ROOT / "tests" / "conservative_water"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip: a skipped GPU test makes no claim. Select the device with
    CUDA_VISIBLE_DEVICES before the process starts (GPU 0 is the user allocation, only when free)."""
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the terminal-storage device event is not exercised")
    return backend.cupy_module()
