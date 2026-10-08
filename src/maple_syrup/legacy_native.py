"""MAHLERAN legacy transport with the ORIGINAL `flow_distrib` walk semantics, for benchmarking only (task gpu_sediment, A1).

This is a NEW module. It does not modify or replace `legacy_transport.py` (accepted Plot 1 replay, which refuses terminal pits
and treats every negative receiver as the ring). It reuses that module's compiled Crank-Nicolson kernel unchanged.

Scientific status (never a conservation claim): fixed composition, unlimited supply, explicit artificial clipping source,
ring/inactive deposition are DIAGNOSTICS that the active-pool equation does not debit, no MAPLE bed is read or written.

Two separate topological views of the same routing graph (they are different objects in the original):

* CN DONOR view (`donors`, `order`, `outlet`): `route_sediment_xml.f90` 243-251. Slots south, west, north, east (MAPLE row 0 =
  south). Terminal pit cells and outlets are never donors. Terminal pits are ACTIVE cells whose pool only ever grows (their
  sediment velocity is zero), they are storage, not export.
* WALK view (`walk_first`, `walk_next`): `flow_distrib.for`. Codes: >= 0 cell index, `RING` (-1) the boundary ring outside the
  interior, `STOP_TERMINAL` (-2) active terminal pit (aspect 0), `STOP_INACTIVE` (-3) inactive interior cell (aspect 0).
  Every visited cell is credited BEFORE its aspect is tested; an aspect-0 cell (pit, inactive, ring) is credited once and the
  walk stops (`flow_distrib.for` 145-147). A SOURCE whose own aspect is 0 starts to the WEST (`flow_distrib.for` 75-80: the
  `else` branch), column 0 meaning the ring; this is counted separately (`C_ASPECT0`).
* Deposition into an INACTIVE interior cell is not an active-pool debit: it is returned as `inactive_dep` (diagnostic), like the
  ring tally. Terminal-pit deposition IS a debit of the pit's mobile pool (source-based accounting, as for any active cell).

Native zero-slope diffuse behaviour (`diffuse_flow_transport.for` 16-21, probed by agent_handoffs/tasks/gpu_sediment/
zero_slope_diffuse_probe.json): with slope 0 a wet raining cell neither walks nor deposits (`sed_temp` only, which the
Crank-Nicolson solve does not use), so any detachment there stays in the cell's pool. Every OTHER no-law case (concentrated
flow without excess stream power, `conc_flow_transport.for` 49-54; zero/underflowed distance) deposits locally. The kernel
below distinguishes them (`C_ZERO_SLOPE` versus `C_LOCAL`). With the current positive `spc` the zero-slope rain detachment is
itself zero, so this only matters for forced/diagnostic inputs.

Optional order-dependent legacy erasure (`erase_on`): in `route_sediment_xml.f90` 166-169 a wet cell with no rain and Re <= 500
sets `depos_soil = 0` when the row-major loop reaches it, discarding deposition already credited to it by earlier-processed
sources; the no-splash benchmark patch does the same for dry raining cells. Reproducing this needs the legacy source order
(north row first, then west to east: `source_order_legacy`). It is OFF by default (the accepted Plot 1 replay does not model it)
and its discarded mass is returned as `erased`.

Kernels are plain loops (pure Python for tests, Numba via `legacy_native_numba`). Time levels, units and arithmetic of the walk
and of the pool follow `legacy_transport._walk_py/_cn_py` term by term so that with the same inputs the default path is bitwise
identical for networks without terminal pits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from maple_syrup.routing import DONOR_SLOTS, EXPORT, PIT_STORAGE, RoutingGraph
from maple_syrup.sediment_physics import REGIME_CODES

__all__ = [
    "COUNT_NAMES",
    "C_ASPECT0",
    "C_ERASE_CELLS",
    "C_INACTIVE",
    "C_LIMIT",
    "C_LOCAL",
    "C_RING",
    "C_TERMINAL",
    "C_VGE",
    "C_WALKS",
    "C_ZERO_SLOPE",
    "N_COUNTS",
    "REDUCE_ROWS",
    "RING",
    "STOP_INACTIVE",
    "STOP_TERMINAL",
    "NativeNetwork",
    "NativeNetworkError",
    "native_network",
    "walk_limits",
]

RING = -1
STOP_TERMINAL = -2
STOP_INACTIVE = -3
_VGE_PER_S = 1.0e-19  # flow_distrib.for 83: `vge = 10.d-20 * dt`
_R_DIFFUSE, _R_TRANS_RAIN = REGIME_CODES["diffuse"], REGIME_CODES["transitional_rain"]

(C_WALKS, C_RING, C_TERMINAL, C_INACTIVE, C_LIMIT, C_VGE, C_ASPECT0, C_LOCAL, C_ZERO_SLOPE, C_ERASE_CELLS) = range(10)
N_COUNTS = 10
COUNT_NAMES = ("walks", "ring_credits", "terminal_credits", "inactive_credits", "limit_stops", "vge_stops",
               "aspect0_source_walks", "local_deposits", "zero_slope_no_walk", "erase_cells")
#: rows of the reduce output
REDUCE_ROWS = ("det_rate", "dep_rate", "clip", "old_mobile", "new_mobile", "terminal_mobile", "terminal_dep_rate",
               "outlet_flux")
_N_REDUCE = len(REDUCE_ROWS)


class NativeNetworkError(ValueError):
    pass


def walk_limits(dx_m: float) -> np.ndarray:
    """Walk limit (cells) by regime code, `initialize_values_xml.f90` 261-263 (same helper the accepted replay uses)."""
    from maple_syrup.legacy_transport import walk_limits_by_regime

    return walk_limits_by_regime(dx_m)


@dataclass(frozen=True)
class NativeNetwork:
    """Flat host connectivity (cell index `i = iy * nx + ix`, MAPLE row 0 = south). All arrays are read-only."""

    shape: tuple[int, int]
    dx_m: float
    active: np.ndarray  # (n,) bool
    inactive: np.ndarray  # (n,) bool
    outlet: np.ndarray  # (n,) bool
    terminal: np.ndarray  # (n,) bool: active, not an outlet, no receiver (PIT_STORAGE)
    slope_zero: np.ndarray  # (n,) bool: graph slope == 0
    donors: np.ndarray  # (n, 4) int64 CN donor view, slots S, W, N, E, -1 = none
    order: np.ndarray  # (m,) int64 active cells, donors before receivers
    cn_receiver: np.ndarray  # (n,) int64 graph receiver codes (EXPORT / PIT_STORAGE / INACTIVE / cell); CN does not read it
    walk_first: np.ndarray  # (n,) int64 walk view: first cell credited after the source
    walk_next: np.ndarray  # (n,) int64 walk view: successor after crediting a cell, or a STOP code
    aspect0: np.ndarray  # (n,) bool: sources whose walk starts WEST
    source_order_index: np.ndarray  # (m,) active cells, ascending index (the accepted replay's order)
    source_order_legacy: np.ndarray  # (m,) active cells, north row first then west to east (legacy loop order)
    active_idx: np.ndarray
    terminal_idx: np.ndarray
    outlet_idx: np.ndarray
    graph_input_sha256: str

    def summary(self) -> dict:
        return {"shape": list(self.shape), "dx_m": self.dx_m, "n_cells": int(self.active.size),
                "n_active": int(self.active_idx.size), "n_inactive": int(self.inactive.sum()),
                "n_terminal_storage": int(self.terminal_idx.size), "n_outlets": int(self.outlet_idx.size),
                "n_aspect0_sources": int(self.aspect0.sum()), "n_zero_slope_active": int((self.slope_zero & self.active).sum()),
                "graph_input_sha256": self.graph_input_sha256}


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


def native_network(graph: RoutingGraph) -> NativeNetwork:
    """Vectorised construction from a host `RoutingGraph` (no per-cell Python loops). Refuses any receiver code it cannot
    classify instead of mapping it to export."""
    if not isinstance(graph, RoutingGraph):
        raise NativeNetworkError("graph must be a maple_syrup.routing.RoutingGraph")
    ny, nx = (int(v) for v in graph.shape)
    n = ny * nx
    try:
        active = np.asarray(graph.active, dtype=np.bool_).reshape(-1)
        outlet = np.asarray(graph.outlet, dtype=np.bool_).reshape(-1)
        receiver = np.asarray(graph.receiver, dtype=np.int64).reshape(-1)
        slope = np.asarray(graph.slope, dtype=np.float64).reshape(-1)
        level_order = np.asarray(graph.level_order_host, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError, AttributeError) as exc:
        raise NativeNetworkError(f"the graph must be a host NumPy graph: {exc}") from exc
    if not (active.size == outlet.size == receiver.size == slope.size == n):
        raise NativeNetworkError("graph arrays do not match its shape")
    if np.any(outlet & ~active):
        raise NativeNetworkError("an outlet is not an active cell")
    no_receiver = active & ~outlet & (receiver < 0)
    if np.any(receiver[no_receiver] != PIT_STORAGE):
        raise NativeNetworkError("an active non-outlet cell has a negative receiver other than PIT_STORAGE")
    pit_mask = getattr(graph, "pit_storage", None)
    if pit_mask is not None and not np.array_equal(np.asarray(pit_mask, dtype=np.bool_).reshape(-1), no_receiver):
        raise NativeNetworkError("graph.pit_storage differs from the cells without a receiver")
    if np.any(receiver[outlet] != EXPORT):
        raise NativeNetworkError("an outlet cell does not carry the EXPORT receiver code")
    terminal = no_receiver
    routed = active & ~outlet & ~terminal
    src = np.flatnonzero(routed)
    rec = receiver[src]
    if src.size and (rec.min() < 0 or rec.max() >= n or not np.all(active[rec]) or np.any(rec == src)):
        raise NativeNetworkError("a routed cell has an invalid or inactive receiver")
    # CN donor view: the donor sits at (dr, dc) relative to its receiver.
    donors = np.full((n, 4), -1, dtype=np.int64)
    rd, cd = np.divmod(src, nx)
    rr, cr = np.divmod(rec, nx)
    placed = np.zeros(src.size, dtype=np.bool_)
    for slot, (dr, dc) in enumerate(DONOR_SLOTS):
        m = (rd - rr == dr) & (cd - cr == dc)
        donors[rec[m], slot] = src[m]
        placed |= m
    if not np.all(placed):
        raise NativeNetworkError("a receiver is not a D4 neighbour of its donor")
    # topological order over active cells
    order = level_order[active[level_order]] if level_order.size else level_order
    if order.size != int(active.sum()) or np.unique(order).size != order.size:
        raise NativeNetworkError("graph.level_order_host does not list every active cell exactly once")
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(order.size)
    if src.size and np.any(position[src] >= position[rec]):
        raise NativeNetworkError("the level order does not place every donor before its receiver")
    # walk view
    col = np.arange(n, dtype=np.int64) % nx
    walk_next = np.full(n, STOP_INACTIVE, dtype=np.int64)
    walk_next[terminal] = STOP_TERMINAL
    walk_next[routed] = receiver[routed]
    # An outlet's aspect points at an export receiver: the ring, or (if flagged) an inactive interior cell, which the walk
    # credits once and stops on (STOP_INACTIVE), exactly like any other inactive cell.
    aspect = np.asarray(graph.aspect, dtype=np.int64).reshape(-1)
    o = np.flatnonzero(outlet)
    if o.size:
        if np.any((aspect[o] < 1) | (aspect[o] > 4)):
            raise NativeNetworkError("an outlet cell has no flow direction")
        orr = o // nx + np.array([0, 1, 0, -1, 0])[aspect[o]]
        occ = o % nx + np.array([0, 0, 1, 0, -1])[aspect[o]]
        inside = (orr >= 0) & (orr < ny) & (occ >= 0) & (occ < nx)
        target = np.where(inside, np.clip(orr, 0, ny - 1) * nx + np.clip(occ, 0, nx - 1), RING)
        if np.any(inside & active[np.where(inside, target, 0)]):
            raise NativeNetworkError("an outlet's export receiver is an active cell")
        walk_next[o] = target
    walk_first = walk_next.copy()
    west = np.arange(n, dtype=np.int64) - 1
    walk_first[terminal] = np.where(col[terminal] > 0, west[terminal], RING)
    active_idx = np.flatnonzero(active)
    row = active_idx // nx
    key = (ny - 1 - row) * nx + (active_idx % nx)
    legacy = active_idx[np.argsort(key, kind="stable")]
    return NativeNetwork(
        shape=(ny, nx), dx_m=float(graph.dx_m), active=_ro(active), inactive=_ro(~active), outlet=_ro(outlet),
        terminal=_ro(terminal), slope_zero=_ro(slope == 0.0), donors=_ro(donors), order=_ro(order),
        cn_receiver=_ro(receiver), walk_first=_ro(walk_first), walk_next=_ro(walk_next), aspect0=_ro(terminal),
        source_order_index=_ro(active_idx), source_order_legacy=_ro(legacy), active_idx=_ro(active_idx),
        terminal_idx=_ro(np.flatnonzero(terminal)), outlet_idx=_ro(np.flatnonzero(outlet)),
        graph_input_sha256=str(graph.input_sha256))


# --- kernels (plain loops) ------------------------------------------------------------------------------------------
def _pack_py(active, terminal, rain, requested, law, svel, rate, regime, prev, decay, dt, erase_dry_rain,
             det, v_used, erase):
    """Per-step glue: detachment RATE `requested / dt`, the legacy virtual velocity (law value where a law applies, the 0.9
    recession memory elsewhere, zero on inactive cells) and the per-cell erase flag; returns a validation flag word
    (1 bad detachment, 2 bad velocity, 4 inactive cell with detachment, 8 law with detachment but no positive 1/L, 16 pit
    with a nonzero velocity)."""
    n = det.shape[0]
    nc = det.shape[1]
    flags = 0
    for i in range(n):
        a = active[i]
        erase[i] = False
        if a:
            r0 = regime[i, 0]
            if r0 == 1 or (erase_dry_rain and r0 == 0 and rain[i] > 0.0):
                erase[i] = True
        for k in range(nc):
            d = requested[i, k] / dt
            if law[i, k]:
                v = svel[i, k]
            else:
                v = prev[i, k] * decay
            if not a:
                v = 0.0
            det[i, k] = d
            v_used[i, k] = v
            if not (math.isfinite(d) and d >= 0.0):
                flags |= 1
            if not (math.isfinite(v) and v >= 0.0):
                flags |= 2
            if (not a) and d != 0.0:
                flags |= 4
            if law[i, k] and d > 0.0 and not (math.isfinite(rate[i, k]) and rate[i, k] > 0.0):
                flags |= 8
            if terminal[i] and v != 0.0:
                flags |= 16
    return flags


def _walk_py(src, det, inv_l, law, regime, limits, slope_zero, walk_first, walk_next, aspect0, inactive_cell, erase,
             erase_on, dx, dt, depos, ring, inactive_dep, erased, counts):
    """Source-based deposition walk, `flow_distrib.for` semantics (module docstring). Sources in the order of `src`."""
    nc = det.shape[1]
    vge = _VGE_PER_S * dt
    for s in range(src.shape[0]):
        i = src[s]
        if erase[i]:
            counts[C_ERASE_CELLS] += 1
            if erase_on:
                for k in range(nc):
                    erased[k] += depos[i, k]
                    depos[i, k] = 0.0
        for k in range(nc):
            d = det[i, k]
            if d <= 0.0:
                continue
            if not law[i, k]:
                r = regime[i, k]
                if slope_zero[i] and (r == _R_DIFFUSE or r == _R_TRANS_RAIN):
                    counts[C_ZERO_SLOPE] += 1  # diffuse_flow_transport 16-21: no walk, no deposition
                    continue
                depos[i, k] += d  # conc_flow_transport 49-54 (and a zero distance): deposit locally
                counts[C_LOCAL] += 1
                continue
            counts[C_WALKS] += 1
            if aspect0[i]:
                counts[C_ASPECT0] += 1
            par = inv_l[i, k]
            fract = 1.0 - math.exp(-par * dx)
            depos[i, k] += fract * d
            pos = walk_first[i]
            step = 0
            limit = limits[regime[i, k]]
            stopped = False
            while step < limit and fract > vge:
                u = (step + 2.0) * dx
                lo = u - dx
                fract = math.exp(-lo * par) - math.exp(-u * par)
                if pos < 0:
                    ring[k] += d * fract  # the ring cell is credited, then the walk exits on its zero aspect
                    counts[C_RING] += 1
                    stopped = True
                    break
                x = d * fract
                if inactive_cell[pos]:
                    inactive_dep[k] += x  # credited once; not an active-pool debit
                else:
                    depos[pos, k] += x
                nxt = walk_next[pos]
                if nxt == STOP_TERMINAL:
                    counts[C_TERMINAL] += 1
                    stopped = True
                    break
                if nxt == STOP_INACTIVE:
                    counts[C_INACTIVE] += 1
                    stopped = True
                    break
                pos = nxt
                step += 1
            if not stopped:
                if step >= limit:
                    counts[C_LIMIT] += 1
                else:
                    counts[C_VGE] += 1


CHECK_MESSAGES = ("a detachment rate is not finite and >= 0", "a deposition rate is not finite and >= 0",
                  "a clipping source is not finite and >= 0", "a new pool, flux or inflow is not finite and >= 0",
                  "an old pool is not finite", "a cumulative per-cell map would overflow",
                  "a per-class step sum is not finite")


def _check_py(active_idx, outlet_idx, dt, det, dep, clip, m1, m2, q2, qin2, cum_det, cum_dep, cum_clip, probe):
    """Read-only validation of one finished step BEFORE anything is reduced, accumulated or swapped. The legacy pool is
    clipped at zero (`m < 0` -> 0, reported as the clipping source) but NaN and Infinity pass that test, so every new pool,
    flux, inflow, deposition, detachment and clipping value must be finite and non-negative, the cumulative maps must stay
    finite, and the per-class sums must be finite. Returns a bit word (`CHECK_MESSAGES`)."""
    nc = det.shape[1]
    flags = 0
    for r in range(probe.shape[0]):
        for k in range(nc):
            probe[r, k] = 0.0
    for t in range(active_idx.shape[0]):
        i = active_idx[t]
        for k in range(nc):
            d = det[i, k]
            p = dep[i, k]
            c = clip[i, k]
            m = m2[i, k]
            q = q2[i, k]
            qi = qin2[i, k]
            if not (math.isfinite(d) and d >= 0.0):
                flags |= 1
            if not (math.isfinite(p) and p >= 0.0):
                flags |= 2
            if not (math.isfinite(c) and c >= 0.0):
                flags |= 4
            if not (math.isfinite(m) and m >= 0.0 and math.isfinite(q) and q >= 0.0 and math.isfinite(qi) and qi >= 0.0):
                flags |= 8
            if not math.isfinite(m1[i, k]):
                flags |= 16
            if not (math.isfinite(cum_det[i, k] + d * dt) and math.isfinite(cum_dep[i, k] + p * dt)
                    and math.isfinite(cum_clip[i, k] + c)):
                flags |= 32
            probe[0, k] += d
            probe[1, k] += p
            probe[2, k] += c
            probe[3, k] += m
    for t in range(outlet_idx.shape[0]):
        i = outlet_idx[t]
        for k in range(nc):
            probe[4, k] += q2[i, k]
    for r in range(probe.shape[0]):
        for k in range(nc):
            if not math.isfinite(probe[r, k]):
                flags |= 64
    return flags


def _reduce_py(active_idx, terminal, outlet_idx, dt, det, dep, clip, m1, m2, q2, cum_det, cum_dep, cum_clip, out):
    """Per-class sums of the step (rows `REDUCE_ROWS`), cumulative per-cell maps, then clear `dep` and `clip` over the active
    cells so the next step starts clean (inactive cells are never written)."""
    nc = det.shape[1]
    for k in range(out.shape[1]):
        for r in range(out.shape[0]):
            out[r, k] = 0.0
    for t in range(active_idx.shape[0]):
        i = active_idx[t]
        for k in range(nc):
            d = det[i, k]
            p = dep[i, k]
            c = clip[i, k]
            out[0, k] += d
            out[1, k] += p
            out[2, k] += c
            out[3, k] += m1[i, k]
            out[4, k] += m2[i, k]
            if terminal[i]:
                out[5, k] += m2[i, k]
                out[6, k] += p
            cum_det[i, k] += d * dt
            cum_dep[i, k] += p * dt
            cum_clip[i, k] += c
            dep[i, k] = 0.0
            clip[i, k] = 0.0
    for t in range(outlet_idx.shape[0]):
        i = outlet_idx[t]
        for k in range(nc):
            out[7, k] += q2[i, k]
