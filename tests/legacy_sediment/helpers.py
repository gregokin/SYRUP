"""Tiny synthetic networks and inputs for the Fortran harness tests (written without being run; Codex executes them)."""
from __future__ import annotations

import numpy as np
import sources as S  # benchmarks/legacy_sediment is on sys.path via conftest

from maple_syrup.infiltration import column_parameters
from maple_syrup.routing import build_routing_graph

NODATA = -9999.0
FRACTIONS = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])  # all six classes carry mass: no class skip


def make_graph(interior, ring_low=None, dx=1.0):
    z = np.asarray(interior, dtype=np.float64)
    ny, nx = z.shape
    full = np.full((ny + 2, nx + 2), NODATA)
    full[1:-1, 1:-1] = z
    for (r, c), value in (ring_low or {}).items():
        full[r, c] = value
    export = np.zeros(full.shape, dtype=bool)
    export[0, :] = export[-1, :] = export[:, 0] = export[:, -1] = True
    return build_routing_graph(full, export, np.full((ny, nx), 40.0), dx, active_mask=z != NODATA, nodata_value=NODATA,
                               allow_masked_nodata=True, allow_pit_storage=True)


def pit_graph():
    """[[3, 2.9, 2.8, 2.9, 3]]: two cells drain into the middle pit from each side; no outlet; gentle slope."""
    return make_graph([[3.0, 2.9, 2.8, 2.9, 3.0]])


def tiny_case_arrays(graph, *, rain_mm_s=0.05, n_steps=60, theta0=0.004, ksat_mm_s=0.001):
    """`(arrays, write_input kwargs)` for a ponding storm on `graph` (all six classes, zero vegetation, Plot 1 XML parameters)."""
    ny, nx = graph.shape
    active = np.asarray(graph.active)
    def full(v):
        return np.full((ny, nx), v)

    column = column_parameters(model="fixed_ksat", ksat_m_per_s=full(ksat_mm_s * 1e-3), suction_m=full(0.0236),
                               drainage_parameter=full(0.05), theta_sat=full(0.36), soil_thickness_m=full(0.21),
                               active_mask=active)
    arrays = S.hydrology_arrays(graph, column, np.where(active, 1.0, 0.0), np.where(active, theta0 * 0.21, 0.0), theta0)
    fractions = np.stack([S.full_north_first(np.full((ny, nx), f), 0.0) for f in FRACTIONS])
    return arrays, column, {"rates_mm_s": [rain_mm_s] * n_steps, "sediment_fractions": fractions,
                            "vegetation_percent": S.full_north_first(np.zeros((ny, nx)), 0.0)}
