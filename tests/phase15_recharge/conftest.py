import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip (a skipped GPU test makes no claim). Hardware tests are opt-in (SYRUP_RELEASE_GPU=1, like the
    portable lane) AND need an operator-selected device (CUDA_VISIBLE_DEVICES before the process starts; GPU0 only under Root's fresh
    guard): the default lane never imports or probes the GPU backend and never allocates another device."""
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA pulse-aware kernel is not exercised (no claim)")
    return backend.cupy_module()
