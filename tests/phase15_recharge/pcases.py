"""Shared fixtures of the Phase 1.5 pulse-aware column controls. Units: metres of water per cell; the Root known-answer table is in
water-equivalent mm per 100 mm layer (theta_FC = 0.2 -> 20 mm, theta_SAT = 0.4 -> 40 mm, theta_res = 0), no E/diffusion unless stated."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from maple_syrup.subsurface import column_state, prepare_column, soil_layers

BIG = 1 << 40
AREA = 4.0
MM = 1e-3
DAY = 86400.0


def layers(n, *, bottom="open", gravity=True, diffusivity=0.0, dz=0.1, fc=0.2, sat=0.4, res=0.0, stress=0.1, factor=1.0):
    edges = [dz * k for k in range(n + 1)]
    return soil_layers(edges, theta_residual=res, theta_field_capacity=fc, theta_saturation=sat, diffusivity_m2_s=diffusivity,
                       theta_evaporation_stress=stress, evaporation_factor=factor, gravity_drainage=gravity, bottom_drainage=bottom)


def make_column(lay, n_cells, backend="numpy", active=None, available=BIG):
    mask = np.ones(n_cells, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if backend == "cuda":  # the CUDA column requires the mask on the current device (state and PET follow col.xp in the helpers)
        from maple.core.backend import cupy_module
        mask = cupy_module().asarray(mask)
    return prepare_column(lay, mask, AREA, backend=backend, reserve_bytes=0, available_bytes_fn=lambda: available)


def state_mm(col, cells_mm, t_s=0.0):
    """`cells_mm`: list over cells of per-layer storages in mm."""
    water = np.ascontiguousarray(np.array(cells_mm, dtype=np.float64).T * MM)
    if col.xp is not np:
        water = col.xp.asarray(water)
    return column_state(col, water, t_s)


def zeros_pet(col):
    return col.xp.zeros(col.n_cells, dtype=np.float64)


def oracle_cascade(w, fc, sat, closed, gravity, pulse):
    """Independent exact-rational (Fraction) model of ONE pulse + cascade + fit test for one cell: returns (final, drain, fits)."""
    n = len(w)
    w = [Fraction(x) for x in w]
    w[0] += Fraction(pulse)
    drain = Fraction(0)
    if gravity:
        cap = [Fraction(0)] * max(n - 1, 0)
        if closed:
            c = Fraction(0)
            for k in range(n - 2, -1, -1):
                t = Fraction(sat[k + 1]) - w[k + 1]
                c += max(t, Fraction(0))
                cap[k] = c
        carry = Fraction(0)
        for l in range(n):
            x = w[l] + carry
            ex = x - Fraction(fc[l])
            if ex > 0:
                if not closed:
                    fwd = ex
                elif l < n - 1:
                    fwd = min(ex, cap[l])
                else:
                    fwd = Fraction(0)
                x -= fwd
                carry = fwd
            else:
                carry = Fraction(0)
            w[l] = x
        drain = carry
    fits = all(w[l] <= Fraction(sat[l]) for l in range(n))
    return w, drain, fits
