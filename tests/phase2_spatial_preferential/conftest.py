"""S2-B2.b fixed-spatial CPU preferential infiltration trial (author-written, CPU only). Comparison targets are the REAL accepted common operator (`preferential_infiltration.prepare_preferential` on an
N = 1 common column built from each cell's explicit parameters, same backend), Root's frozen pre-B2 same-backend controls, and independent hand formulas; nothing here re-implements the law. The S2-A.2 /
S2-A.3 builders are loaded from their sibling suites."""
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


a2 = _load("a2_builders_b2b", TESTS / "phase2_spatial_water" / "conftest.py")
a3 = _load("a3_builders_b2b", TESTS / "phase2_spatial_numba" / "conftest.py")

from maple_syrup import preferential_infiltration as pi  # noqa: E402
from maple_syrup import spatial_preferential_infiltration as spf  # noqa: E402
from maple_syrup import spatial_soil_layers as sp  # noqa: E402
from maple_syrup import subsurface as ss  # noqa: E402

BIG, AREA = a2.BIG, a2.AREA
RTOL, ATOL = 2e-12, 1e-14
REF_NUMPY = TASK / "root_s2_b2_reference"
REF_NUMBA = TASK / "root_s2_b2_reference_numba"
PARAMS = TASK / "root_s2_parameter_controls.json"


def bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def common_pref(p, c, backend, kwargs, contributions, *, active=True, area=AREA):
    """The accepted COMMON column + preferential operator of cell c's explicit parameters (N = 1), same backend. `kwargs` holds per-cell arrays or scalars."""
    kw = dict(diffusivity_m2_s=p["iface"][:, c] if p["iface"].shape[0] else 1e-7)
    lay = ss.soil_layers(p["edges"][:, c], theta_residual=p["res"][:, c], theta_field_capacity=p["fc"][:, c], theta_saturation=p["sat"][:, c], theta_evaporation_stress=0.5, **kw)
    col = ss.prepare_column(lay, np.array([active]), area, backend=backend)
    cell = {k: (float(v[c]) if isinstance(v, np.ndarray) else v) for k, v in kwargs.items() if k in ("a3", "max_intake_rate_m_s")}
    rest = {k: v for k, v in kwargs.items() if k not in cell}
    species = [x[c:c + 1].copy() for x in contributions] if contributions else None
    op = pi.prepare_preferential(col, **cell, **rest, species_contributions=None if species is None else iter(species))
    return col, op


def spatial_pair(inputs, N, backend, mask, kwargs, contributions, *, area=AREA, budget=BIG):
    spec = a3.spec_from(inputs, N)
    col = sp.prepare_spatial_column(spec, mask, area, backend=backend, memory_budget_bytes=budget)
    op = spf.prepare_spatial_preferential(col, **kwargs, species_contributions=None if contributions is None else iter([x.copy() for x in contributions]), memory_budget_bytes=budget)
    return col, op


def hand_column(L, n_cells, backend="numpy", mask=None, area=1.0, thetas=None):
    """Hand geometry as a spatial owner: tops 0.5 m then 1 m layers (theta_res 0.01, fc 0.25, sat 0.9)."""
    edges = np.concatenate([[0.0, 0.5], 0.5 + np.arange(1, L)])
    spec = sp.spatial_layer_spec(edges, theta_residual=0.01, theta_field_capacity=0.25, theta_saturation=0.9, theta_evaporation_stress=0.05, diffusivity_m2_s=0.0, n_cells=n_cells)
    m = np.ones(n_cells, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return sp.prepare_spatial_column(spec, m, area, backend=backend, memory_budget_bytes=BIG)


def hand_depths(L):
    return np.concatenate([[0.5], np.ones(L - 1)])


def hand_state(col, water=None, theta=0.1, t_s=0.0):
    L = col.n_layers
    if water is None:
        water = np.ascontiguousarray(np.repeat((theta * hand_depths(L))[:, None], col.n_cells, axis=1))
    return sp.spatial_column_state(col, np.ascontiguousarray(water), t_s, memory_budget_bytes=BIG)


def full(col, v):
    return np.full(col.n_cells, float(v), dtype=np.float64)


def hand_op(col, *, a3=1.0, maxf=1.0, maxrate=np.inf, background=None, species=(), contributions=None, budget=BIG):
    L = col.n_layers
    bg = background if background is not None else [1.0] + [0.0] * (L - 1)
    return spf.prepare_spatial_preferential(col, a3=a3, max_fraction_per_hour=maxf, max_intake_rate_m_s=maxrate, background_receiver_weights=bg, species_receiver_weights=list(species),
                                            species_contributions=None if contributions is None else iter(contributions), memory_budget_bytes=budget)


def step(op, surface, state, dt, budget=BIG):
    return spf.spatial_preferential_step(op, surface, state, dt, memory_budget_bytes=budget)
