"""Builders of the D1 floating-column suite (author-written)."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from maple_syrup import floating_soil_layers as fl

CONTROLS = Path(__file__).resolve().parent / "controls"
BIG = 1 << 33


def numba_ok() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


BACKENDS = ["numpy"] + (["numba"] if numba_ok() else [])


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def geometry_controls():
    return json.loads((CONTROLS / "root_d1_porous_geometry_controls.json").read_text())


def capacity_controls():
    return json.loads((CONTROLS / "root_capacity_exact_controls.json").read_text())["controls"]


def column(edges, depth, *, L=None, tr=0.0, tf=0.2, ts=0.4, bottom="open", mode="constant", iface=None, d=1.0, stress=0.1, active=None, backend="numpy", area=1.0, factor=1.0):
    edges = np.asarray(edges, dtype=np.float64)
    L = len(edges) - 1
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    N = depth.size
    if mode == "constant":
        kw = dict(diffusivity_m2_s=(np.zeros(L - 1) if iface is None else iface))
    else:
        kw = dict(interface_conductivity_m_s=(np.full(L - 1, 1e-6) if iface is None else iface), diffusion_coefficient=d)
    return fl.prepare_floating_column(edges, theta_residual=tr, theta_field_capacity=tf, theta_saturation=ts, theta_evaporation_stress=stress, diffusion_mode=mode, bottom_drainage=bottom, soil_depth_m=depth,
                                      active_mask=np.ones(N, dtype=bool) if active is None else active, cell_area_m2=area, backend=backend, evaporation_factor=factor, **kw)


def zero_forcing(col):
    return dict(added_water_m=np.zeros((col.n_layers, col.n_cells)), potential_evaporation_m_s=np.zeros(col.n_cells))


def frac(s):
    return Fraction(s)
