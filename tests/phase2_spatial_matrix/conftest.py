"""S2-B2.a fixed-spatial CPU matrix infiltration trial (author-written, CPU only). Comparison targets are the REAL accepted common operator (`matrix_infiltration.prepare_matrix` on an N = 1 common
column built from each cell's explicit parameters, same backend), Root's frozen pre-B2 controls and decimal oracles, and dyadic hand cases; nothing here re-implements the law. The S2-A.2/A.3
builders and the Phase 1 matrix case helpers are loaded from their sibling suites."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parents[1]
CANDIDATE = TESTS.parent
TASK = CANDIDATE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a2 = _load("a2_builders_b2a", TESTS / "phase2_spatial_water" / "conftest.py")
a3 = _load("a3_builders_b2a", TESTS / "phase2_spatial_numba" / "conftest.py")
cases = _load("phase1_matrix_cases_b2a", TESTS / "subsurface_release" / "matrix_infiltration" / "matrix_cases.py")

from maple_syrup import matrix_infiltration as mi  # noqa: E402
from maple_syrup import spatial_matrix_infiltration as sm  # noqa: E402
from maple_syrup import spatial_soil_layers as sp  # noqa: E402
from maple_syrup import subsurface as ss  # noqa: E402

BIG, AREA = a2.BIG, a2.AREA
REF_DIR = TASK / "root_s2_b2_reference"
RTOL, ATOL = 2e-12, 1e-14


def bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def common_matrix(p, c, backend, K, psi, g, *, active=True, area=AREA):
    """The accepted COMMON column + matrix operator of cell c's explicit parameters (N = 1), same backend."""
    kw = dict(diffusivity_m2_s=p["iface"][:, c] if p["iface"].shape[0] else 1e-7)
    lay = ss.soil_layers(p["edges"][:, c], theta_residual=p["res"][:, c], theta_field_capacity=p["fc"][:, c], theta_saturation=p["sat"][:, c], theta_evaporation_stress=0.5, **kw)
    col = ss.prepare_column(lay, np.array([active]), area, backend=backend)
    op = mi.prepare_matrix(col, conductivity_m_s=float(K[c]), suction_m=float(psi[c]), vegetation_factor=float(g[c]))
    return col, op


def spatial_pair(p_inputs, N, backend, mask, K, psi, g, *, area=AREA, budget=BIG):
    spec = a3.spec_from(p_inputs, N)
    col = sp.prepare_spatial_column(spec, mask, area, backend=backend, memory_budget_bytes=budget)
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=K, suction_m=psi, vegetation_factor=g, memory_budget_bytes=budget)
    return col, op


def hand_column(n_cells, backend="numpy", mask=None, area=1.0):
    """The Phase 1 hand geometry as a spatial owner: tops 0.5 m (theta_fc 0.25, theta_sat 0.9, theta_res 0.01), a 1 m deeper layer."""
    spec = sp.spatial_layer_spec(np.array([0.0, cases.DZ0, cases.DZ0 + 1.0]), theta_residual=0.01, theta_field_capacity=cases.FC, theta_saturation=cases.SAT, theta_evaporation_stress=0.05,
                                 diffusivity_m2_s=0.0, n_cells=n_cells)
    m = np.ones(n_cells, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return sp.prepare_spatial_column(spec, m, area, backend=backend, memory_budget_bytes=BIG)


def hand_state(col, theta0=cases.THETA0, deeper=0.1, t_s=0.0, water=None):
    if water is None:
        dz = np.array([cases.DZ0, 1.0])
        theta = np.array([theta0, deeper])
        water = np.ascontiguousarray(np.repeat((theta * dz)[:, None], col.n_cells, axis=1))
    return sp.spatial_column_state(col, water, t_s, memory_budget_bytes=BIG)


def full(col, v):
    return np.full(col.n_cells, float(v), dtype=np.float64)


def hand_op(col, *, k=cases.K, psi=cases.PSI, g=cases.G):
    return sm.prepare_spatial_matrix(col, conductivity_m_s=k, suction_m=psi, vegetation_factor=g, memory_budget_bytes=BIG)


def hand_hist(op, w=0.0, c=0.0):
    return sm.spatial_matrix_history(op, wetting_m=w, cumulative_intake_m=c, memory_budget_bytes=BIG)


def step(op, pool, standing, rate, state, hist, dt, budget=BIG):
    return sm.spatial_matrix_step(op, pool, standing, rate, state, hist, dt, memory_budget_bytes=budget)
