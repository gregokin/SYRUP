"""Shared fixtures of the subsurface column tests. Dyadic layer edges and retention values so every hand-derived
balance is exact in FP64 (the repository's convention for asserted exactness)."""
from __future__ import annotations

import numpy as np

from maple_syrup.subsurface import column_state, prepare_column, soil_layers

EDGES = (0.0, 0.25, 1.0)  # thickness 0.25 m and 0.75 m
RES, FC, SAT, STRESS = 0.0625, 0.25, 0.5, 0.25
# residual_m = [0.015625, 0.046875]; field_capacity_m = [0.0625, 0.1875]; saturation_m = [0.125, 0.375]
AREA = 4.0
BIG = 1 << 40


def layers(*, gravity=True, diffusivity=0.0, factor=1.0, edges=EDGES, res=RES, fc=FC, sat=SAT, stress=STRESS):
    return soil_layers(edges, theta_residual=res, theta_field_capacity=fc, theta_saturation=sat,
                       diffusivity_m2_s=diffusivity, theta_evaporation_stress=stress, evaporation_factor=factor,
                       gravity_drainage=gravity)


def column(lay, n_cells=4, active=None, backend="numpy", area=AREA, reserve=0, available=BIG, xp=np):
    mask = np.ones(n_cells, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if xp is not np:
        mask = xp.asarray(mask)
    return prepare_column(lay, mask, area, backend=backend, reserve_bytes=reserve, available_bytes_fn=lambda: available)


def from_cells(cells) -> np.ndarray:
    """`cells` is a list of per-cell layer lists; returns the (L, N) C-contiguous float64 array."""
    return np.ascontiguousarray(np.array(cells, dtype=np.float64).T)


def state_of(col, cells, t_s=0.0):
    water = from_cells(cells)
    if col.xp is not np:
        water = col.xp.asarray(water)
    return column_state(col, water, t_s)


def forcing(col, add_cells=None, pet=None):
    """Zero forcing unless given; arrays of the column namespace."""
    n_layers, n_cells = col.n_layers, col.n_cells
    add = np.zeros((n_layers, n_cells), dtype=np.float64) if add_cells is None else from_cells(add_cells)
    rate = np.zeros(n_cells, dtype=np.float64) if pet is None else np.array(pet, dtype=np.float64)
    if col.xp is not np:
        add, rate = col.xp.asarray(add), col.xp.asarray(rate)
    return add, rate


def host(array) -> np.ndarray:
    return np.asarray(array.get()) if hasattr(array, "get") else np.asarray(array)


def random_case(seed: int, n_layers: int, n_cells: int, *, gravity: bool, diffusive: bool):
    """A random VALID case: layers, mask (at least one active and one inactive cell), state within [0, saturation],
    added water and evaporation rate (zero on inactive cells), dt. Host arrays."""
    rng = np.random.default_rng(seed)
    thickness = rng.uniform(0.05, 0.5, n_layers)
    edges = np.concatenate([[0.0], np.cumsum(thickness)])
    res = rng.uniform(0.01, 0.08, n_layers)
    fc = res + rng.uniform(0.05, 0.2, n_layers)
    sat = fc + rng.uniform(0.05, 0.3, n_layers)
    diffusivity = rng.uniform(1e-8, 1e-6, max(n_layers - 1, 0)) if diffusive else 0.0
    lay = soil_layers(edges, theta_residual=res, theta_field_capacity=fc, theta_saturation=sat,
                      diffusivity_m2_s=diffusivity if n_layers > 1 else 0.0, theta_evaporation_stress=res[0] + 0.1,
                      evaporation_factor=1.2, gravity_drainage=gravity)
    mask = rng.random(n_cells) < 0.7
    mask[0] = True
    mask[-1] = False
    theta = rng.uniform(0.0, 1.0, (n_layers, n_cells)) * sat[:, None]
    water = np.ascontiguousarray(theta * thickness[:, None])
    add = rng.uniform(0.0, 0.02, (n_layers, n_cells)) * (rng.random((n_layers, n_cells)) < 0.5)
    add[:, ~mask] = 0.0
    pet = rng.uniform(0.0, 5e-7, n_cells)
    pet[~mask] = 0.0
    return lay, mask, water, np.ascontiguousarray(add), pet, 600.0
