import sys
from pathlib import Path

import pytest

# Shared layer/column fixtures of this suite (column_cases.py); the suite is run explicitly (`pytest tests/subsurface`).
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip (a skipped GPU test makes no claim). Select the device with
    CUDA_VISIBLE_DEVICES before the process starts; GPU 0 is the user allocation."""
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA subsurface kernel is not exercised (no claim)")
    return backend.cupy_module()
