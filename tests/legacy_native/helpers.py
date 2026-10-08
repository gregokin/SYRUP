"""Small synthetic networks for the native-walk legacy tests (written without being run; Codex executes them)."""
from __future__ import annotations

import numpy as np

from maple_syrup import legacy_native as N
from maple_syrup.legacy_native_numba import get_kernels
from maple_syrup.routing import build_routing_graph
from maple_syrup.sediment_physics import REGIME_CODES

NODATA = -9999.0
DX = 1.0


def make_graph(interior, ring_low=None, active=None):
    """`interior` (ny, nx) elevations south-first (NODATA on inactive cells); `ring_low` {(row, col) in the full grid: z}."""
    z = np.asarray(interior, dtype=np.float64)
    ny, nx = z.shape
    full = np.full((ny + 2, nx + 2), NODATA)
    full[1:-1, 1:-1] = z
    for (r, c), value in (ring_low or {}).items():
        full[r, c] = value
    export = np.zeros(full.shape, dtype=bool)
    export[0, :] = export[-1, :] = export[:, 0] = export[:, -1] = True
    act = (z != NODATA) if active is None else np.asarray(active, dtype=bool)
    return build_routing_graph(full, export, np.full((ny, nx), 40.0), DX, active_mask=act, nodata_value=NODATA,
                               allow_masked_nodata=True, allow_pit_storage=True)


def channel(n=5):
    """1 x n channel draining east into a valid ring cell: one outlet, no pits."""
    return make_graph([list(range(n, 0, -1))], ring_low={(1, n + 1): 0.5})


def pit_channel():
    """[[3,2,1,2,3]]: two cells drain into the middle pit from each side; no outlet."""
    return make_graph([[3, 2, 1, 2, 3]])


def converging():
    """3 x 3 surface with two donors into the outlet cell (row 0, col 2)."""
    return make_graph([[3, 2, 1], [4, 3, 2], [5, 4, 3]], ring_low={(1, 4): 0.5})


def arrays(net, nc=2, det=1.0, par=1.0, regime="concentrated", law=True):
    n = net.active.size
    d = np.zeros((n, nc))
    d[net.active_idx] = det
    inv_l = np.full((n, nc), par)
    return {"det": d, "inv_l": inv_l, "law": np.full((n, nc), law, dtype=bool),
            "regime": np.full((n, nc), REGIME_CODES[regime], dtype=np.int8)}


def run_walk(net, a, *, limits=None, dt=1.0, erase=None, erase_on=False, order=None, nc=None):
    k = get_kernels(False)
    nc = a["det"].shape[1]
    n = net.active.size
    depos = np.zeros((n, nc))
    ring, inactive_dep, erased = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    counts = np.zeros(N.N_COUNTS, dtype=np.int64)
    lim = N.walk_limits(net.dx_m) if limits is None else np.asarray(limits, dtype=np.int64)
    k.walk(net.source_order_index if order is None else order, a["det"], a["inv_l"], a["law"], a["regime"], lim,
           net.slope_zero, net.walk_first, net.walk_next, net.aspect0, net.inactive,
           np.zeros(n, dtype=bool) if erase is None else erase, erase_on, net.dx_m, dt, depos, ring, inactive_dep, erased,
           counts)
    return depos, ring, inactive_dep, erased, counts
