import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Reuse the actual-MAPLE bed constructors of the Phase 5 event tests (make_bed, chain/valley elevations) and the
# verified Plot 1 case fixture of the Phase 5 runner tests.
sys.path.insert(0, str(ROOT / "tests" / "phase5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_sediment_experiment import (
    plot1_case,  # noqa: F401  (module fixture: generated verified case)
)


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip: a skipped GPU test makes no claim. Select the device with
    CUDA_VISIBLE_DEVICES before the process starts (GPU 0 is the user allocation)."""
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the coupled CUDA event is not exercised (no claim)")
    return backend.cupy_module()
