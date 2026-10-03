import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Phase 4 synthetic terrain/graph builders (make_graph, valley_full, random_full, chain_full).
sys.path.insert(0, str(ROOT / "tests" / "phase4"))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip (no GPU claim is made by a skipped test). Select the device with
    CUDA_VISIBLE_DEVICES BEFORE the process starts."""
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA routing kernel is not exercised")
    return backend.cupy_module()
