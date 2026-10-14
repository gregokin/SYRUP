import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Reuse the actual-MAPLE bed constructors of the Phase 5 event tests and the backend / windowed fixtures of the
# conservative-water tests (both directories are plain test modules on sys.path, as their own conftests do).
sys.path.insert(0, str(ROOT / "tests" / "phase5"))
sys.path.insert(0, str(ROOT / "tests" / "conservative_water"))
sys.path.insert(0, str(ROOT / "tests" / "phase6"))


@pytest.fixture(params=["numpy", "cupy"])
def xp(request):
    """The event namespace: NumPy always; CuPy only with a device (an explicit skip claims nothing). Select GPU0 with
    CUDA_VISIBLE_DEVICES before the process starts; CPU runs hide CUDA."""
    from maple.core.backend import BackendUnavailableError, resolve_backend

    try:
        return resolve_backend(request.param).xp
    except BackendUnavailableError as exc:
        pytest.skip(f"{request.param} unavailable: {exc}")
