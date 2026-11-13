"""Shared builders of the S2-A.2 spatial-water suite. Author-written. The comparison target is always the REAL accepted common solver (`subsurface.soil_layers` + `prepare_column` + `reference_step`/`step`)
evaluated one column at a time under that column's explicit parameters; nothing here reimplements the water law."""
from __future__ import annotations

import numpy as np
import pytest

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss

BIG = 1 << 33
AREA = 4.0
DTS = (0.0, 1e-6, 3600.0, 86400.0, 1e7)


def numba_ok() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def params(rng, L, N, *, mode="constant", dry_res=False):
    """Valid random per-cell parameters: edges (L+1,N) from 0 (different widths per cell), thetas (L,N) with different residual/FC/saturation per cell, interface input (L-1,N)."""
    widths = rng.uniform(0.04, 0.5, size=(L, N))
    edges = np.vstack([np.zeros((1, N)), np.cumsum(widths, axis=0)])
    res = np.zeros((L, N)) if dry_res else rng.uniform(0.0, 0.1, size=(L, N))
    fc = res + rng.uniform(0.05, 0.2, size=(L, N))
    sat = fc + rng.uniform(0.05, 0.3, size=(L, N))
    iface = rng.uniform(0.0, 1e-6, size=(max(L - 1, 0), N)) if mode == "constant" else rng.uniform(0.0, 1e-5, size=(max(L - 1, 0), N))
    return {"edges": edges, "res": res, "fc": fc, "sat": sat, "iface": iface}


def make_spec(p, *, mode="constant", bottom="open", gravity=True, stress=0.5, factor=1.0, d=0.3, iface=None, edges=None, res=None, fc=None, sat=None):
    iface = p["iface"] if iface is None else iface
    edges = p["edges"] if edges is None else edges
    n = p["edges"].shape[1]
    extra = dict(diffusivity_m2_s=iface) if mode == "constant" else dict(interface_conductivity_m_s=iface, diffusion_coefficient=d)
    return sp.spatial_layer_spec(edges, theta_residual=p["res"] if res is None else res, theta_field_capacity=p["fc"] if fc is None else fc,
                                 theta_saturation=p["sat"] if sat is None else sat, theta_evaporation_stress=stress, evaporation_factor=factor, gravity_drainage=gravity,
                                 diffusion_mode=mode, bottom_drainage=bottom, n_cells=n, **extra)


def prepare(spec, *, backend="numpy", mask=None, **kw):
    mask = np.ones(spec.n_cells, dtype=bool) if mask is None else mask
    return sp.prepare_spatial_column(spec, mask, AREA, backend=backend, memory_budget_bytes=BIG, **kw)


def common_column(p, c, *, mode="constant", bottom="open", gravity=True, stress=0.5, factor=1.0, d=0.3, active=True):
    """The accepted common-layer column for cell c's explicit parameters (N = 1)."""
    kw = dict(diffusivity_m2_s=p["iface"][:, c]) if mode == "constant" else dict(interface_conductivity_m_s=p["iface"][:, c], diffusion_coefficient=d)
    lay = ss.soil_layers(p["edges"][:, c], theta_residual=p["res"][:, c], theta_field_capacity=p["fc"][:, c], theta_saturation=p["sat"][:, c], theta_evaporation_stress=stress,
                         evaporation_factor=factor, gravity_drainage=gravity, diffusion_mode=mode, bottom_drainage=bottom, **kw)
    return ss.prepare_column(lay, np.array([active]), AREA)


def sat_depth(p):
    return p["sat"] * np.diff(p["edges"], axis=0)


def fc_depth(p):
    return p["fc"] * np.diff(p["edges"], axis=0)


def water_kind(rng, p, kind):
    """Valid-or-flagged water states: dry (all zero), saturated, above field capacity, random, mixed."""
    L, N = p["res"].shape
    sat, fc = sat_depth(p), fc_depth(p)
    if kind == "dry":
        return np.zeros((L, N))
    if kind == "saturated":
        return sat.copy()
    if kind == "above_fc":
        return fc + (sat - fc) * rng.uniform(0.2, 1.0, size=(L, N))
    if kind == "random":
        return sat * rng.uniform(0.0, 1.0, size=(L, N))
    w = sat * rng.uniform(0.0, 1.0, size=(L, N))  # mixed
    w[:, : N // 3] = 0.0
    w[:, N // 3 : 2 * N // 3] = sat[:, N // 3 : 2 * N // 3]
    return w


def bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint64) if a.dtype == np.float64 else a


FIELDS = ("water_m", "percolation_m", "diffusion_m", "surface_return_m", "drainage_m", "evaporation_m", "evaporation_unmet_m", "rows")


def assert_columns_equal(spatial_arrays, p, water, added, pet, dt, share, recharge, mask, *, mode, bottom, gravity, stress=0.5, factor=1.0):
    """Every StepArrays field of the spatial body equals, BIT FOR BIT, the accepted common reference on that column alone (flags/rows included)."""
    N = water.shape[1]
    for c in range(N):
        col = common_column(p, c, mode=mode, bottom=bottom, gravity=gravity, stress=stress, factor=factor, active=bool(mask[c]))
        ref = ss.reference_step(col, water[:, c : c + 1].copy(), None if recharge is not None else added[:, c : c + 1].copy(), pet[c : c + 1].copy(), dt,
                                None if share is None else share[c : c + 1].copy(), recharge)
        for name in FIELDS:
            got, want = getattr(spatial_arrays, name), getattr(ref, name)
            got = got[..., c : c + 1] if got.ndim == 2 else got[c : c + 1]
            np.testing.assert_array_equal(bits(got), bits(want), err_msg=f"{name} column {c}")


@pytest.fixture(params=["constant", "tietjen_donor_depth"])
def mode(request):
    return request.param


@pytest.fixture(params=["open", "closed"])
def bottom(request):
    return request.param
