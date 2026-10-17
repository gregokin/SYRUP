"""Backend fixtures for the pond-event tests. Availability is decided per backend; a missing Numba / CuPy / CUDA device is an
honest skip (no GPU claim is made without a device). Nothing here was run by its author (file-only tools)."""
import numpy as np
import pytest
from pond_helpers import backend_namespace, backends


@pytest.fixture(params=backends())
def be(request):
    return backend_namespace(request.param)


@pytest.fixture
def cpu():
    return backend_namespace("numpy")


@pytest.fixture
def np_xp():
    return np
