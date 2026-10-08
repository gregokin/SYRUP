"""Shared paths of the GPU Newton tests: Phase 4 synthetic graph builders, the Phase 4S device-case helpers
(`hydro_cases`), the RFID pit chain and the CPU Newton helpers. Select the device with CUDA_VISIBLE_DEVICES BEFORE
pytest starts; without CuPy/a device the GPU tests skip explicitly (no GPU claim)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
for sub in ("tests/phase4", "tests/phase4s", "tests/phase5", "tests/phase7h", "tests/rfid", "tests"):
    sys.path.insert(0, str(ROOT / sub))


@pytest.fixture(scope="session")
def gpu():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA Newton kernels are not exercised")
    return backend.cupy_module()


from test_sediment_experiment import (
    plot1_case,  # noqa: F401
)
