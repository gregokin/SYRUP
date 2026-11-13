"""Shared inputs of the S0 evaporation suite (author-written; dyadic coefficients so hand derivations are exact in FP64)."""
from __future__ import annotations

import hashlib

import numpy as np

from maple_syrup import soil_evaporation as se
from maple_syrup import subsurface as ss

EDGES = (0.0, 0.25, 0.5, 1.0)
AREA = 4.0
BIG = 1 << 40  # declared UNIT-TEST budget for tiny arrays (not a free-memory reading)
PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
N = 7


def inputs(n=N):
    rng = np.random.default_rng(20261113)
    pet = np.ascontiguousarray(rng.integers(0, 64, n) / 2.0**24)  # m/s, dyadic
    cover = np.ascontiguousarray(np.array([0.0, 1.0, 0.5, 0.25, 0.75, 0.0, 1.0][:n], dtype=np.float64))
    surf = np.ascontiguousarray(rng.integers(0, 32, n) / 2.0**12)  # m
    return pet, cover, surf


MASK = np.array([True, True, False, True, True, False, True])


def fixed_column(mask=MASK, *, factor=1.0, backend="numpy", area=AREA):
    lay = ss.soil_layers(EDGES, theta_residual=0.0625, theta_field_capacity=0.25, theta_saturation=0.5, diffusivity_m2_s=0.0, theta_evaporation_stress=0.25, evaporation_factor=factor)
    return ss.prepare_column(lay, mask.copy(), area, backend=backend, reserve_bytes=0, available_bytes_fn=lambda: BIG)


def digest(staged) -> str:
    h = hashlib.sha256()
    for a in (staged.surface_water_after_m, staged.surface_evaporation_m, staged.surface_evaporation_unmet_m, staged.potential_combined_soil_rate_m_s, staged.soil_evaporation_share):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr(sorted(staged.ledger.as_dict().items())).encode())
    return h.hexdigest()
