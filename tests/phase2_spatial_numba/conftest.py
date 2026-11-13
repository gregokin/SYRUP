"""Shared builders of the S2-A.3 compiled-CPU spatial-water suite. Author-written. The comparison targets are always REAL accepted implementations: the common Numba column (`subsurface.prepare_column(backend="numba")`),
the NumPy common solver and the explicit NumPy spatial path, each evaluated one column at a time under that column's explicit parameters; nothing here reimplements the water law. The S2-A.2 builders are loaded
from the sibling suite (one source of truth)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_A2 = Path(__file__).resolve().parents[1] / "phase2_spatial_water" / "conftest.py"
_spec = importlib.util.spec_from_file_location("a2_builders", _A2)
a2 = importlib.util.module_from_spec(_spec)
sys.modules["a2_builders"] = a2
_spec.loader.exec_module(a2)

from maple_syrup import spatial_soil_layers as sp  # noqa: E402
from maple_syrup import subsurface as ss  # noqa: E402

BIG, AREA, DTS = a2.BIG, a2.AREA, a2.DTS
PRE_A3 = Path(__file__).resolve().parents[3] / "s2_a3_pre_sources" / "subsurface_numba.py"
STEP_FIELDS = ("water_m", "percolation_m", "diffusion_m", "surface_return_m", "drainage_m", "evaporation_m", "evaporation_unmet_m", "rows")
LAYOUTS = ("scalar", "shared", "cell")


def numba_available() -> bool:
    return a2.numba_ok()


def require_numba():
    return pytest.importorskip("numba")


def bits(a):
    return a2.bits(a)


def layout_case(rng, L, N, *, edges="cell", res="cell", fc="cell", sat="cell", iface="cell", mode="constant"):
    """Valid explicit inputs in the requested LAYOUTS plus the fully expanded per-cell arrays `p` (the A2 `params` shape) that describe exactly the same physics, so the accepted common column of cell
    `c` can be built from `p`. A scalar/shared layout broadcasts one value/row to every cell."""
    widths = rng.uniform(0.04, 0.5, size=(L, N))
    cell_edges = np.vstack([np.zeros((1, N)), np.cumsum(widths, axis=0)])
    base_res, base_fc, base_sat = rng.uniform(0.0, 0.05, (L, N)), rng.uniform(0.1, 0.2, (L, N)), rng.uniform(0.25, 0.45, (L, N))
    base_if = rng.uniform(0.0, 1e-6, (max(L - 1, 0), N)) if mode == "constant" else rng.uniform(0.0, 1e-5, (max(L - 1, 0), N))

    def pick(kind, cell_array):
        if kind == "cell":
            return cell_array.copy(), cell_array.copy()
        if kind == "shared":
            col = cell_array[..., :1].copy().reshape(cell_array.shape[:-1])
            return col, np.broadcast_to(col[..., None], cell_array.shape).copy()
        value = float(cell_array.reshape(-1)[0]) if cell_array.size else 0.0
        return value, np.full(cell_array.shape, value)

    e_in, e_full = pick(edges if edges != "scalar" else "shared", cell_edges)
    r_in, r_full = pick(res, base_res)
    f_in, f_full = pick(fc, base_fc)
    s_in, s_full = pick(sat, base_sat)
    i_in, i_full = pick(iface, base_if)
    if L == 1 and iface != "scalar":
        i_in = np.empty((0,)) if iface == "shared" else np.empty((0, N))
    p = {"edges": e_full, "res": r_full, "fc": f_full, "sat": s_full, "iface": i_full}
    return {"edges": e_in, "res": r_in, "fc": f_in, "sat": s_in, "iface": i_in}, p


def spec_from(inputs, N, *, mode="constant", bottom="open", gravity=True, stress=0.5, factor=1.0, d=0.3):
    extra = dict(diffusivity_m2_s=inputs["iface"]) if mode == "constant" else dict(interface_conductivity_m_s=inputs["iface"], diffusion_coefficient=d)
    return sp.spatial_layer_spec(inputs["edges"], theta_residual=inputs["res"], theta_field_capacity=inputs["fc"], theta_saturation=inputs["sat"], theta_evaporation_stress=stress,
                                 evaporation_factor=factor, gravity_drainage=gravity, diffusion_mode=mode, bottom_drainage=bottom, n_cells=N, **extra)


def common_numba_column(p, c, *, mode="constant", bottom="open", gravity=True, stress=0.5, factor=1.0, d=0.3, active=True):
    """The accepted common-layer NUMBA column of cell c's explicit parameters (N = 1)."""
    kw = dict(diffusivity_m2_s=p["iface"][:, c]) if mode == "constant" else dict(interface_conductivity_m_s=p["iface"][:, c], diffusion_coefficient=d)
    lay = ss.soil_layers(p["edges"][:, c], theta_residual=p["res"][:, c], theta_field_capacity=p["fc"][:, c], theta_saturation=p["sat"][:, c], theta_evaporation_stress=stress,
                         evaporation_factor=factor, gravity_drainage=gravity, diffusion_mode=mode, bottom_drainage=bottom, **kw)
    return ss.prepare_column(lay, np.array([active]), AREA, backend="numba")


def step_args(rng, p, mask, kind="random"):
    L, N = p["res"].shape
    water = a2.water_kind(rng, p, kind)
    added = rng.uniform(0, 1e-3, size=(L, N)) * 0.3 * mask
    pet = rng.uniform(0, 2e-7, size=N) * mask
    return water, added, pet


def run_spatial(column, water, dt, *, added=None, pet, share=None, recharge=None, t0=0.0, budget=BIG):
    state = sp.spatial_column_state(column, water, t0, memory_budget_bytes=budget)
    if recharge is None:
        return sp.spatial_step(column, state, dt, added_water_m=added, potential_evaporation_m_s=pet, evaporation_share=share, memory_budget_bytes=budget)
    return sp.spatial_step_with_recharge(column, state, dt, recharge_depth_m=recharge, potential_evaporation_m_s=pet, evaporation_share=share, memory_budget_bytes=budget)


def result_arrays(step):
    s = step
    return {"water_m": s.state.water_m, "percolation_m": s.percolation_m, "diffusion_m": s.diffusion_m, "surface_return_m": s.surface_return_m, "drainage_m": s.drainage_m,
            "evaporation_m": s.evaporation_m, "evaporation_unmet_m": s.evaporation_unmet_m}


@pytest.fixture(params=["constant", "tietjen_donor_depth"])
def mode(request):
    return request.param


@pytest.fixture(params=["open", "closed"])
def bottom(request):
    return request.param
