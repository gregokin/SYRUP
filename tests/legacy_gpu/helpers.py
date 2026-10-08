"""Tiny synthetic networks and a CPU oracle of the GPU walk algorithm (written without being run; Codex executes the tests)."""
from __future__ import annotations

import importlib.util
import math

import numpy as np

from maple_syrup import legacy_native as N
from maple_syrup.legacy_native_cuda import WalkTables
from maple_syrup.routing import build_routing_graph

NODATA = -9999.0
DX = 1.0


def make_graph(interior, ring_low=None):
    z = np.asarray(interior, dtype=np.float64)
    ny, nx = z.shape
    full = np.full((ny + 2, nx + 2), NODATA)
    full[1:-1, 1:-1] = z
    for (r, c), value in (ring_low or {}).items():
        full[r, c] = value
    export = np.zeros(full.shape, dtype=bool)
    export[0, :] = export[-1, :] = export[:, 0] = export[:, -1] = True
    return build_routing_graph(full, export, np.full((ny, nx), 40.0), DX, active_mask=z != NODATA, nodata_value=NODATA,
                               allow_masked_nodata=True, allow_pit_storage=True)


GRAPHS = {
    "channel_to_ring": lambda: make_graph([[5.0, 4.0, 3.0, 2.0, 1.0]], {(1, 6): 0.5}),
    "terminal_pit": lambda: make_graph([[3.0, 2.9, 2.8, 2.9, 3.0]]),
    "pit_in_column_zero": lambda: make_graph([[1.0, 2.0, 3.0, 4.0, 5.0]]),
    "inactive_neighbour": lambda: make_graph([[NODATA, 1.0, 2.0, 3.0]]),
    "converging": lambda: make_graph([[3.0, 2.0, 1.0], [4.0, 3.0, 2.0], [5.0, 4.0, 3.0]], {(1, 4): 0.5}),
}


def gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001 - any failure means "no usable device"
        return False


def physics_for(graph, fractions):
    """Wet-law context of the Plot 1 sediment parameters on `graph` whose composition is `fractions`: a (6,) vector or a full
    (ny, nx, 6) array (the holdings are `2.5 x fractions`, so the context fractions equal them for rows summing to one)."""
    from maple_syrup.legacy_physics_numba import prepare_legacy_physics
    from maple_syrup.sediment_physics import (
        physics_grid_from_graph,
        plot1_sediment_parameters,
    )

    ny, nx = graph.shape
    frac = np.asarray(fractions, dtype=np.float64)
    holdings = (np.broadcast_to(frac * 2.5, (ny, nx, 6)) if frac.ndim == 1 else frac * 2.5).copy()
    return prepare_legacy_physics(plot1_sediment_parameters(), physics_grid_from_graph(graph), np.zeros((ny, nx)), holdings)


#: composition patterns of the record-class tests: name -> (builder of the (ny, nx, 6) fractions from the grid shape)
def fraction_pattern(kind: str, shape) -> np.ndarray:
    ny, nx = shape
    f = np.zeros((ny, nx, 6))
    if kind == "all6":
        f[:] = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    elif kind == "two":  # the Chastre / RFID situation: classes 3 and 4 only
        f[..., 3], f[..., 4] = 0.092, 0.908
    elif kind == "zero":
        pass
    elif kind == "hetero":  # class 4 everywhere; class 0 in ONE cell; class 2 in cell 0 only; class 5 in the last cell only
        f[..., 4] = 0.7
        f[0, nx - 1, 0] = 0.2
        f[0, 0, 2] = 0.3
        f[0, nx - 1, 5] = 0.1
        f[0, :, 3] = np.where(np.arange(nx) % 2 == 0, 0.0, 0.2)
    else:
        raise ValueError(kind)
    return f


def expected_eligible(active, fractions) -> list[int]:
    """Independent restatement of the eligibility rule (explicit loops): some ACTIVE cell has a positive fraction."""
    fr = np.asarray(fractions).reshape(len(active), -1)
    return [k for k in range(fr.shape[1]) if any(bool(active[i]) and fr[i, k] > 0.0 for i in range(len(active)))]


def random_walk_inputs(net: N.NativeNetwork, rng, nc: int = 3):
    n = net.active.size
    act = net.active[:, None]
    det = np.where(act & (rng.random((n, nc)) > 0.3), rng.uniform(0.05, 1.0, (n, nc)), 0.0)
    par = np.where(rng.random((n, nc)) > 0.8, 200.0, rng.uniform(0.1, 6.0, (n, nc)))  # 200 reaches the vge cut
    law = rng.random((n, nc)) > 0.25
    regime = rng.integers(2, 7, (n, nc)).astype(np.int8)  # diffuse .. suspended
    return det, par, law, regime


def emulate(net: N.NativeNetwork, tb: WalkTables, det, par, law, regime, limits, dx: float, dt: float):
    """The GPU algorithm in Python: the record values of `sg_values`, the CPU-order gather of `sg_gather`, the ring sum, and the
    tally codes. NOT a production path; it exists to prove the static tables and the gather order against the CPU walk."""
    n, nc = det.shape
    values = np.zeros((tb.n_records, nc))
    code = np.zeros((tb.src_cells.size, nc), dtype=np.uint8)
    vge = 1.0e-19 * dt
    for s, i in enumerate(tb.src_cells.tolist()):
        base = int(tb.rec_off[s])
        plen = int(tb.rec_off[s + 1]) - base - 1
        for k in range(nc):
            d = det[i, k]
            c, written, rec0 = 0, 0, 0.0
            if d > 0.0:
                if not law[i, k]:
                    if net.slope_zero[i] and regime[i, k] in (2, 3):
                        c = 0x40
                    else:
                        rec0, c = d, 0x20
                else:
                    p = par[i, k]
                    fract = 1.0 - math.exp(-p * dx)
                    rec0, c = fract * d, 0x10
                    step, limit, stopped = 0, int(limits[regime[i, k]]), False
                    while step < limit and fract > vge:
                        assert step < plen, "static path exhausted"
                        u = (step + 2.0) * dx
                        lo = u - dx
                        fract = math.exp(-lo * p) - math.exp(-u * p)
                        values[base + 1 + step, k] = d * fract
                        written = step + 1
                        if step == plen - 1 and tb.end_kind[s] != 0:
                            c |= int(tb.end_kind[s])
                            stopped = True
                            break
                        step += 1
                    if not stopped:
                        c |= 4 if step >= limit else 5
            values[base, k] = rec0
            code[s, k] = c
            assert not values[base + 1 + written:base + 1 + plen, k].any()
    depos = np.zeros((n, nc))
    for cell in range(n):
        for k in range(nc):
            acc = 0.0
            for p in range(int(tb.tgt_ptr[cell]), int(tb.tgt_ptr[cell + 1])):
                acc += values[tb.tgt_rec[p], k]
            depos[cell, k] = acc
    ring = np.zeros(nc)
    for k in range(nc):
        acc = 0.0
        for rec in tb.ring_rec.tolist():
            acc += values[rec, k]
        ring[k] = acc
    return values, code, depos, ring


def tally(net: N.NativeNetwork, tb: WalkTables, code: np.ndarray) -> np.ndarray:
    counts = np.zeros(N.N_COUNTS, dtype=np.int64)
    for s, i in enumerate(tb.src_cells.tolist()):
        for k in range(code.shape[1]):
            c = int(code[s, k])
            if c & 0x10:
                counts[N.C_WALKS] += 1
                counts[N.C_ASPECT0] += int(net.aspect0[i])
                stop = c & 0x0F
                for kind, idx in ((1, N.C_RING), (2, N.C_TERMINAL), (3, N.C_INACTIVE), (4, N.C_LIMIT), (5, N.C_VGE)):
                    counts[idx] += int(stop == kind)
            counts[N.C_LOCAL] += int(bool(c & 0x20))
            counts[N.C_ZERO_SLOPE] += int(bool(c & 0x40))
    return counts
