import os
import sys
from pathlib import Path

import pytest

# Shared case helpers of this suite (matrix_cases.py); the suite is run explicitly (`pytest tests/matrix_infiltration`).
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip (a skipped GPU test makes no claim). Select the device with
    CUDA_VISIBLE_DEVICES before the process starts; GPU 0 is the user allocation."""
    # PORTABLE-LANE DELTA (test guard only): the default lane never imports or probes the GPU backend. Hardware tests need an explicit opt-in (SYRUP_RELEASE_GPU=1) in addition to an operator-selected device
    # (CUDA_VISIBLE_DEVICES set before the process starts) and hardware authorization; without it this skips with no hardware claim.
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA matrix kernel is not exercised (no claim)")
    return backend.cupy_module()
