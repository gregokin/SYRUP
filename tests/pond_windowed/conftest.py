"""Fixtures of the windowed pond-event tests. The dense pond helpers (`tests/pond_event/pond_helpers.py`) are reused as they
are; availability of Numba / CuPy / a CUDA device is an honest skip. Nothing here was run by its author (file-only tools)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pond_event"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pond_helpers import backend_namespace, backends  # noqa: E402


@pytest.fixture(params=backends())
def be(request):
    return backend_namespace(request.param)


@pytest.fixture
def cpu():
    return backend_namespace("numpy")
