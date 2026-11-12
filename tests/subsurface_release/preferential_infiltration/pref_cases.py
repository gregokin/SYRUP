"""Shared fixtures of the preferential infiltration tests. Dyadic layer edges, retention values, covers and fractions
so every hand-derived balance is exact in FP64 (the repository's convention for asserted exactness)."""
from __future__ import annotations

import math

import numpy as np

from maple_syrup import preferential_infiltration as pi
from maple_syrup import subsurface as ss

EDGES = (0.0, 0.25, 1.0)  # thickness 0.25 m and 0.75 m
RES, FC, SAT, STRESS = 0.0625, 0.25, 0.5, 0.25
# residual_m = [0.015625, 0.046875]; field_capacity_m = [0.0625, 0.1875]; saturation_m = [0.125, 0.375]
AREA = 4.0
BIG = 1 << 40  # a declared UNIT-TEST budget for tiny arrays: NOT a free-memory reading, admission or scaling/RSS evidence
HOUR = 3600.0


def layers(*, edges=EDGES, res=RES, fc=FC, sat=SAT, stress=STRESS):
    return ss.soil_layers(edges, theta_residual=res, theta_field_capacity=fc, theta_saturation=sat,
                          diffusivity_m2_s=0.0, theta_evaporation_stress=stress, evaporation_factor=1.0,
                          gravity_drainage=True)


def column(lay=None, n_cells=4, active=None, backend="numpy", area=AREA, reserve=0, available=BIG, xp=np):
    lay = layers() if lay is None else lay
    mask = np.ones(n_cells, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if xp is not np:
        mask = xp.asarray(mask)
    return ss.prepare_column(lay, mask, area, backend=backend, reserve_bytes=reserve, available_bytes_fn=lambda: available)


def operator(col, *, a3=0.5, max_fraction=1.0, max_rate=math.inf, background=(0.0, 1.0), species=(),
             contributions=None, extra_reserve=0):
    """Build with host inputs; contributions (list of host (N,) arrays) are moved to the column namespace."""
    xp = col.xp
    if contributions is not None:
        contributions = [xp.asarray(np.ascontiguousarray(np.asarray(c, dtype=np.float64))) for c in contributions]
    if isinstance(a3, np.ndarray) and xp is not np:
        a3 = xp.asarray(a3)
    if isinstance(max_rate, np.ndarray) and xp is not np:
        max_rate = xp.asarray(max_rate)
    return pi.prepare_preferential(col, a3=a3, max_fraction_per_hour=max_fraction, max_intake_rate_m_s=max_rate,
                                   background_receiver_weights=None if background is None else list(background),
                                   species_receiver_weights=[list(w) for w in species],
                                   species_contributions=contributions, extra_reserve_bytes=extra_reserve)


def from_cells(cells) -> np.ndarray:
    """`cells` is a list of per-cell layer lists; returns the (L, N) C-contiguous float64 array."""
    return np.ascontiguousarray(np.array(cells, dtype=np.float64).T)


def state_of(col, cells, t_s=0.0):
    water = from_cells(cells)
    if col.xp is not np:
        water = col.xp.asarray(water)
    return ss.column_state(col, water, t_s)


def host(array) -> np.ndarray:
    return np.asarray(array.get()) if hasattr(array, "get") else np.asarray(array)


def arrays_of(result):
    return {"surface": result.surface_water_m, "water": result.state.water_m, "intake": result.intake_m,
            "total": result.intake_total_m}


def random_case(seed: int, n_layers: int, n_cells: int, *, n_species: int = 3, extremes: bool = True):
    """A random VALID case on host arrays: layers, mask (at least one active and one inactive cell), explicit
    per-cell a3 map, per-cell rate-cap map, background and species weights (one species disabled), streamed
    contributions, soil state within [0, saturation], surface water and dt. With `extremes`, some cells carry
    beta exactly 0 (zero cover and a3 = 0) and some exactly 1 (a3 = 1 with max fraction 1)."""
    rng = np.random.default_rng(seed)
    thickness = rng.uniform(0.05, 0.5, n_layers)
    edges = np.concatenate([[0.0], np.cumsum(thickness)])
    res = rng.uniform(0.01, 0.08, n_layers)
    fc = res + rng.uniform(0.05, 0.2, n_layers)
    sat = fc + rng.uniform(0.05, 0.3, n_layers)
    lay = ss.soil_layers(edges, theta_residual=res, theta_field_capacity=fc, theta_saturation=sat,
                         diffusivity_m2_s=0.0, theta_evaporation_stress=res[0] + 0.1, evaporation_factor=1.0,
                         gravity_drainage=True)
    mask = rng.random(n_cells) < 0.7
    mask[0] = True
    mask[-1] = False
    a3 = rng.choice(np.array(pi.A3_VALUES), n_cells)
    covers = [rng.uniform(0.0, 1.5, n_cells) * (rng.random(n_cells) < 0.8) for _ in range(n_species)]
    if extremes and n_cells >= 4:
        a3[1] = 0.0
        for c in covers:
            c[1] = 0.0  # beta exactly 0
        a3[2] = 1.0  # beta exactly 1 (max fraction 1 below)
    weights = [rng.uniform(0.0, 2.0, n_layers) * (rng.random(n_layers) < 0.7) for _ in range(n_species)]
    if n_species > 1:
        weights[-1] = np.zeros(n_layers)  # disabled species with positive cover
    for w in weights[:-1] if n_species > 1 else weights:
        if not np.any(w > 0.0):
            w[rng.integers(n_layers)] = 1.0
    background = rng.uniform(0.0, 1.0, n_layers) * (rng.random(n_layers) < 0.6)
    if not np.any(background > 0.0):
        background[-1] = 1.0
    max_rate = rng.uniform(1e-7, 5e-6, n_cells)
    max_rate[rng.random(n_cells) < 0.3] = math.inf
    theta = rng.uniform(0.0, 1.0, (n_layers, n_cells)) * sat[:, None]
    theta[:, 3 % n_cells] = sat  # one fully saturated column: every receiver rejects
    water = np.ascontiguousarray(theta * thickness[:, None])
    surface = rng.uniform(0.0, 0.03, n_cells) * (rng.random(n_cells) < 0.9)
    return lay, mask, a3, 1.0, max_rate, background, weights, covers, water, surface, 600.0


def build_random(seed, n_layers, n_cells, backend, *, xp=np, available=BIG, n_species=3, extremes=True):
    lay, mask, a3, max_fraction, max_rate, background, weights, covers, water, surface, dt = random_case(
        seed, n_layers, n_cells, n_species=n_species, extremes=extremes)
    col = ss.prepare_column(lay, xp.asarray(mask) if xp is not np else mask, AREA, backend=backend, reserve_bytes=0,
                            available_bytes_fn=lambda: available)
    op = operator(col, a3=a3, max_fraction=max_fraction, max_rate=max_rate, background=background, species=weights,
                  contributions=covers)
    return col, op, water, surface, dt
