"""Shared cases of the Phase 1 column-law tests (donor-depth diffusion, closed bottom, E-only forcing). Dyadic layer
edges and retention values so hand-derived balances are exact in FP64. Independent of tests/subsurface/column_cases.py
(same conventions, no import)."""
from __future__ import annotations

import math

import numpy as np

from maple_syrup.subsurface import column_state, prepare_column, soil_layers

AREA = 4.0
BIG = 1 << 40  # a declared UNIT-TEST budget for tiny arrays: NOT a free-memory reading, admission or scaling/RSS evidence
# Two layers, unequal thickness: 0.25 m over 0.75 m (centre separation 0.5 m).
EDGES2 = (0.0, 0.25, 1.0)
# Three layers: 0.25, 0.25, 0.5 m.
EDGES3 = (0.0, 0.25, 0.5, 1.0)
RES, FC, SAT, STRESS = 0.0625, 0.25, 0.5, 0.25


def layers(*, edges=EDGES2, res=RES, fc=FC, sat=SAT, stress=STRESS, factor=1.0, gravity=True, diffusivity=0.0,
           bottom="open"):
    """Constant-D layers (the historical mode)."""
    return soil_layers(edges, theta_residual=res, theta_field_capacity=fc, theta_saturation=sat,
                       diffusivity_m2_s=diffusivity, theta_evaporation_stress=stress, evaporation_factor=factor,
                       gravity_drainage=gravity, bottom_drainage=bottom)


def donor_layers(*, edges=EDGES2, res=RES, fc=FC, sat=SAT, stress=STRESS, factor=1.0, gravity=True, k=1e-6, d=0.05,
                 bottom="open"):
    """Donor-depth layers: K_ij per interface (scalar or sequence) and dimensionless d."""
    return soil_layers(edges, theta_residual=res, theta_field_capacity=fc, theta_saturation=sat,
                       theta_evaporation_stress=stress, evaporation_factor=factor, gravity_drainage=gravity,
                       diffusion_mode="tietjen_donor_depth", interface_conductivity_m_s=k, diffusion_coefficient=d,
                       bottom_drainage=bottom)


def column(lay, n_cells=1, active=None, backend="numpy", area=AREA, reserve=0, available=BIG, xp=np):
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
    n_layers, n_cells = col.n_layers, col.n_cells
    add = np.zeros((n_layers, n_cells), dtype=np.float64) if add_cells is None else from_cells(add_cells)
    rate = np.zeros(n_cells, dtype=np.float64) if pet is None else np.array(pet, dtype=np.float64)
    if col.xp is not np:
        add, rate = col.xp.asarray(add), col.xp.asarray(rate)
    return add, rate


def host(array) -> np.ndarray:
    return np.asarray(array.get()) if hasattr(array, "get") else np.asarray(array)


def donor_pair_amount(h_i, h_j, sep, k, d, w_i, w_j, dt):
    """INDEPENDENT oracle of one donor-depth pair exchange (positive downward), written from the law only:
    D = d K h_donor, G = D / sep, s = 1/h_i + 1/h_j, amount = (theta_i - theta_j) / s x (-expm1(-G s dt)); the donor
    is the layer with the higher theta."""
    theta_i, theta_j = w_i / h_i, w_j / h_j
    s = 1.0 / h_i + 1.0 / h_j
    h_donor = h_i if theta_i > theta_j else h_j
    g = d * k * h_donor / sep
    return (theta_i - theta_j) / s * (-math.expm1(-g * s * dt))


def donor_sequence_oracle(thickness, centres, k, d, water, dt, residual_m, saturation_m):
    """INDEPENDENT sequential oracle over the interfaces from the top, with the SAME bounds (donor above residual,
    receiver below saturation) applied before each transfer. Returns (water_after, signed amounts per interface)."""
    w = [float(v) for v in water]
    amounts = []
    for i in range(len(w) - 1):
        sep = centres[i + 1] - centres[i]
        raw = donor_pair_amount(thickness[i], thickness[i + 1], sep, k[i], d, w[i], w[i + 1], dt)
        if raw > 0.0:
            donor, receiver = i, i + 1
        else:
            donor, receiver = i + 1, i
        avail = max(w[donor] - residual_m[donor], 0.0)
        cap = max(saturation_m[receiver] - w[receiver], 0.0)
        act = min(abs(raw), avail, cap)
        w[donor] -= act
        w[receiver] += act
        amounts.append(act if raw > 0.0 else -act)
    return np.array(w), np.array(amounts)


def random_case(seed: int, n_layers: int, n_cells: int, *, gravity: bool, mode: str, bottom: str):
    """A random VALID case in the requested diffusion mode and bottom option: layers, mask (at least one active and
    one inactive cell), state within [0, saturation], added water and evaporation rate (zero on inactive), dt."""
    rng = np.random.default_rng(seed)
    thickness = rng.uniform(0.05, 0.5, n_layers)
    edges = np.concatenate([[0.0], np.cumsum(thickness)])
    res = rng.uniform(0.01, 0.08, n_layers)
    fc = res + rng.uniform(0.05, 0.2, n_layers)
    sat = fc + rng.uniform(0.05, 0.3, n_layers)
    common = {"theta_residual": res, "theta_field_capacity": fc, "theta_saturation": sat,
                  "theta_evaporation_stress": res[0] + 0.1, "evaporation_factor": 1.2, "gravity_drainage": gravity,
                  "bottom_drainage": bottom}
    if mode == "constant":
        lay = soil_layers(edges, diffusivity_m2_s=rng.uniform(1e-8, 1e-6, max(n_layers - 1, 0)) if n_layers > 1 else 0.0,
                          **common)
    else:
        k = rng.uniform(1e-7, 1e-5, max(n_layers - 1, 0)) if n_layers > 1 else 1e-6
        lay = soil_layers(edges, diffusion_mode="tietjen_donor_depth", interface_conductivity_m_s=k,
                          diffusion_coefficient=float(rng.uniform(0.01, 0.2)), **common)
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


def arrays_of(result):
    return {"water": result.state.water_m, "return": result.surface_return_m, "drainage": result.drainage_m,
            "evaporation": result.evaporation_m, "unmet": result.evaporation_unmet_m,
            "percolation": result.percolation_m, "diffusion": result.diffusion_m}
