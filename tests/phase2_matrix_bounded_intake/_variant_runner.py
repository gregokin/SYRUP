"""Subprocess helper: run the SPATIAL matrix step body (NumPy or Numba) over deterministic multi-cell scenarios with whichever source tree is first on PYTHONPATH and save the raw arrays. Used by the
conditional-effect accounting test to compare the pre-review02 and the current sources on IDENTICAL data. Not a test module."""
import sys

import numpy as np

from maple_syrup import spatial_matrix_infiltration as sm
from maple_syrup import spatial_soil_layers as sp

out, backend = sys.argv[1], sys.argv[2]
N, L = 60000, 3
arrays = {}
for scen in range(4):
    rng = np.random.default_rng(900 + scen)
    dz = rng.uniform(0.05, 0.4, (L, N))
    edges = np.vstack([np.zeros(N), np.cumsum(dz, axis=0)])
    res = rng.uniform(0.0, 0.05, (L, N))
    fc = res + rng.uniform(0.05, 0.2, (L, N))
    sat = fc + rng.uniform(0.02, 0.3, (L, N))
    mask = rng.random(N) < 0.9
    spec = sp.spatial_layer_spec(edges, theta_residual=res, theta_field_capacity=fc, theta_saturation=sat, theta_evaporation_stress=0.5, diffusivity_m2_s=np.zeros((L - 1, N)), n_cells=N)
    col = sp.prepare_spatial_column(spec, mask, 1.0, backend=backend, memory_budget_bytes=1 << 36)
    K, psi, g = rng.uniform(1e-7, 5e-5, N), rng.uniform(0.0, 0.4, N), rng.uniform(0.0, 1.0, N)
    K[::13], psi[::17], g[::19] = 0.0, 0.0, 0.0
    g[::23] = 1.0
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=K, suction_m=psi, vegetation_factor=g, memory_budget_bytes=1 << 36)
    sat_m = sat * dz
    near_full = rng.random(N) < (0.5 if scen % 2 else 0.1)
    w0 = np.where(near_full, sat_m[0] - rng.uniform(1e-4, 2e-2, N), sat_m[0] * rng.uniform(0.0, 0.9, N))
    water = np.vstack([w0, sat_m[1:] * rng.uniform(0.0, 0.9, (L - 1, N))])
    j0 = np.where(rng.random(N) < 0.4, 0.0, rng.uniform(0.0, 0.03, N))
    cum0 = rng.uniform(0.0, 0.1, N)
    hist = sm.spatial_matrix_history(op, wetting_m=j0, cumulative_intake_m=cum0, memory_budget_bytes=1 << 36)
    state = sp.spatial_column_state(col, np.ascontiguousarray(water), 0.0, memory_budget_bytes=1 << 36)
    dt = (3600.0, 600.0, 3600.0, 7200.0)[scen]
    rate = np.where(mask, rng.uniform(0.0, 6e-5, N), 0.0)
    contact = rng.random(N) < 0.5
    standing = np.where(contact & mask, rng.uniform(1e-3, 0.05, N), 0.0)
    pool = np.where(standing > 0, standing + rate * dt, rate * dt * rng.uniform(0.3, 1.0, N))
    pool = np.where(mask, pool, 0.0)
    r = sm._run(op, np.ascontiguousarray(pool), np.ascontiguousarray(standing), np.ascontiguousarray(rate), state.water_m, hist.wetting_m, hist.cumulative_intake_m, dt)
    room = np.maximum(sat_m[0] - w0, 0.0)
    for name, a in (("pool_in", pool), ("room", room), ("mask", mask), ("water_out", r.water_m), ("pool_out", r.pool_m), ("wet_out", r.wetting_m), ("cum_out", r.cumulative_intake_m),
                    ("intake", r.intake_m), ("rows", r.rows)):
        arrays[f"s{scen}_{name}"] = np.asarray(a)
np.savez(out, **arrays)
