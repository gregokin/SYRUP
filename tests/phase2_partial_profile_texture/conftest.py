"""Fixtures of the soil-profile-texture suite. Run from the candidate root with the candidate `src` first and the
IMMUTABLE MAPLE reference second on PYTHONPATH (see the task report for the exact command). Numba / CuPy / a CUDA
device are honest skips, never silent fallbacks."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip (a skipped GPU test makes no claim). Select the device with
    CUDA_VISIBLE_DEVICES before the process starts; GPU 0 is the user allocation."""
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA texture kernel is not exercised (no claim)")
    return backend.cupy_module()


@pytest.fixture(scope="session")
def numba_module():
    return pytest.importorskip("numba", reason="Numba is unavailable; the compiled texture kernel is not exercised")
