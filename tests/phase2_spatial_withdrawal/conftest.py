"""S2-B1.b fixed-spatial CPU withdrawal/availability trial (author-written, CPU only). Comparison targets are the REAL accepted common operator (`subsurface_transpiration.prepare_transpiration` on an N = 1 common column
built from each cell's explicit parameters) and the dyadic hand case; nothing here re-implements the law. The S2-A.2 and S2-A.3 builders are loaded from the sibling suites."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, TESTS / rel / "conftest.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a2 = _load("a2_builders_b1b", "phase2_spatial_water")
a3 = _load("a3_builders_b1b", "phase2_spatial_numba")

from maple_syrup import spatial_soil_layers as sp  # noqa: E402
from maple_syrup import spatial_transpiration as sw  # noqa: E402
from maple_syrup import subsurface as ss  # noqa: E402
from maple_syrup import subsurface_transpiration as st  # noqa: E402

BIG, AREA = a2.BIG, a2.AREA
LAYOUTS = a3.LAYOUTS


def common_pair(p, c, backend, *, active=True):
    """The accepted COMMON column + operator of cell c's explicit parameters (N = 1), same backend."""
    lay = ss.soil_layers(p["edges"][:, c], theta_residual=p["res"][:, c], theta_field_capacity=p["fc"][:, c], theta_saturation=p["sat"][:, c], theta_evaporation_stress=0.5,
                         diffusivity_m2_s=p["iface"][:, c] if p["iface"].shape[0] else 1e-7)
    col = ss.prepare_column(lay, np.array([active]), AREA, backend=backend)
    return col, st.prepare_transpiration(col)


def spatial_owner(p, backend, mask, **kw):
    return a2.prepare(a2.make_spec(p, **kw), backend=backend, mask=mask)


def bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param
