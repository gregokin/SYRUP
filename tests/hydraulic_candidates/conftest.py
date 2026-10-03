import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Phase 4 terrain/graph builders; Phase 4S shared case helpers (`hydro_cases.terrain`); this directory's `cand_cases`.
sys.path.insert(0, str(ROOT / "tests" / "phase4"))
sys.path.insert(0, str(ROOT / "tests" / "phase4s"))
sys.path.insert(0, str(ROOT / "tests" / "phase5"))  # the verified Plot 1 case fixture (test_experimental_cli.py only)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tests"))


from test_sediment_experiment import (
    plot1_case,  # noqa: F401  (fixture used by test_experimental_cli.py, as in phase7h/phase4s)
)


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip (no GPU claim is made by a skipped test). Select the device with
    CUDA_VISIBLE_DEVICES BEFORE the process starts."""
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the experimental CUDA solvers are not exercised")
    return backend.cupy_module()
