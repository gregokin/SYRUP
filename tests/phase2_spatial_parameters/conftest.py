"""Shared helpers of the S2-A.1 spatial-parameter preparation suite. Author-written; the comparison target is the REAL accepted `subsurface.soil_layers`, evaluated column by column."""
from __future__ import annotations

import numpy as np
import pytest

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss

BIG = 1 << 33
RETAINED_BY_MODE = {"constant": ("theta_residual", "theta_field_capacity", "theta_saturation", "dz", "pair_inverse_thickness", "interface_conductance_down"),
                    "tietjen_donor_depth": ("theta_residual", "theta_field_capacity", "theta_saturation", "dz", "pair_inverse_thickness", "interface_conductance_down",
                                            "interface_conductance_up")}


def numba_ok() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.fixture(params=["numpy", "numba"])
def cpu_backend(request):
    if request.param == "numba" and not numba_ok():
        pytest.skip("Numba is not importable (no claim for the numba backend)")
    return request.param


def params(rng, L, N, *, mode="constant", dry=False):
    """Valid random PER-CELL parameters: edges (L+1,N) from 0, thetas (L,N), interface input (L-1,N)."""
    widths = rng.uniform(0.05, 0.5, size=(L, N))
    edges = np.vstack([np.zeros((1, N)), np.cumsum(widths, axis=0)])
    res = np.zeros((L, N)) if dry else rng.uniform(0.0, 0.1, size=(L, N))
    fc = res + rng.uniform(0.05, 0.2, size=(L, N))
    sat = fc + rng.uniform(0.05, 0.3, size=(L, N))
    iface = rng.uniform(0.0, 1e-6, size=(max(L - 1, 0), N)) if mode == "constant" else rng.uniform(0.0, 1e-5, size=(max(L - 1, 0), N))
    return {"edges": edges, "res": res, "fc": fc, "sat": sat, "iface": iface}


def make_spec(p, *, mode="constant", N=None, edges=None, res=None, fc=None, sat=None, iface=None, stress=0.5, d=0.3, **kw):
    arrays = {"depth_edges_m": p["edges"] if edges is None else edges, "theta_residual": p["res"] if res is None else res,
              "theta_field_capacity": p["fc"] if fc is None else fc, "theta_saturation": p["sat"] if sat is None else sat}
    iface = p["iface"] if iface is None else iface
    n = N if N is not None else (p["edges"].shape[1] if p["edges"].ndim == 2 else p["res"].shape[1])
    extra = dict(diffusivity_m2_s=iface) if mode == "constant" else dict(interface_conductivity_m_s=iface, diffusion_coefficient=d)
    return sp.spatial_layer_spec(arrays["depth_edges_m"], theta_residual=arrays["theta_residual"], theta_field_capacity=arrays["theta_field_capacity"],
                                 theta_saturation=arrays["theta_saturation"], theta_evaporation_stress=stress, diffusion_mode=mode, n_cells=n, **extra, **kw)


def prepare(spec, *, backend="numpy", mask=None, area=4.0, budget=BIG, **kw):
    n = spec.n_cells
    mask = np.ones(n, dtype=bool) if mask is None else mask
    return sp.prepare_spatial_column(spec, mask, area, backend=backend, memory_budget_bytes=budget, **kw)


def common_cell(p, c, mode="constant", stress=0.5, d=0.3):
    """The REAL accepted common-layer object for cell c's explicit parameters."""
    edges = p["edges"][:, c] if p["edges"].ndim == 2 else p["edges"]
    col = lambda a: a[:, c] if a.ndim == 2 else a  # noqa: E731
    kw = dict(diffusivity_m2_s=col(p["iface"])) if mode == "constant" else dict(interface_conductivity_m_s=col(p["iface"]), diffusion_coefficient=d)
    return ss.soil_layers(edges, theta_residual=col(p["res"]), theta_field_capacity=col(p["fc"]), theta_saturation=col(p["sat"]), theta_evaporation_stress=stress, diffusion_mode=mode, **kw)


def host(col, name):
    return col.inspect(name, host_memory_budget_bytes=BIG)
