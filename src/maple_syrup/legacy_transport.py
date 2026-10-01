"""MAHLERAN legacy sediment transport, ported for BENCHMARKING ONLY (Phase 7e).

This module reproduces the legacy MAHLERAN 1.2.3 transport operator so that a
compiled Python replay can be compared with the actual Fortran reference run.
It is NOT a production transport scheme and must never be wired into the
MAPLE bed exchange: it carries the legacy's non-conservative features
explicitly (below) because the benchmark must reproduce them, not hide them.

Reference routines (read-only, /home/okin/MAHLERAN/src/Subroutines_Sediment):

- `flow_distrib.for`: SOURCE-BASED deposition. The mass detached at a cell
  is deposited at once along its downslope D4 path with an exponential
  step-length law of mean `travel_dist_ave`: fraction `1 - exp(-dx/L)` in
  the source cell (44-46), then for downstream cell n = 0, 1, ...
  `exp(-l/L) - exp(-u/L)` with `u = (n + 2) dx`, `l = u - dx` (102-116),
  while the position is inside the grid INCLUDING the boundary ring,
  `n < nsteps` and the previous fraction exceeds `1e-19 dt` (vge, 86-97).
  A ring cell receives its fraction and the walk exits on its zero aspect
  (118-141). Walk limits (initialize_values_xml 261-263): 10 m / 100 m /
  500 m of path for diffuse / concentrated / suspended transport,
  `max(int(limit / dx + 0.5), 2)` cells.
- `conc_flow_transport.for` 49-54: with no excess stream power the detached
  mass is deposited locally (`depos += detach`) without a walk.
- `route_sediment_xml.f90` 236-299 (Crank-Nicolson, method 2): for every
  active cell in up- to downslope order, inflow `qsedin(2)` is the sum of
  `q_soil(2)` of the D4 neighbours whose aspect points into the cell, in the
  fixed legacy direction order; then
  `d2 = [d1/dt + (0.5/dx)(qin2 - q1 + qin1) + (detach - depos)] / (1/dt + 0.5 v/dx)`,
  `d2 < 0 -> 0` (289-291), `q2 = d2 v`. Time levels: level 1 is the
  previous step's level 2 (update_water_flow.for 33-38).
- Outlet export (output_hydro_data_xml.f90 130-142): the instantaneous
  outlet sediment flux is `sum(q_soil(2) at outlet cells) * dx * density`.

Units. The legacy works in depth equivalents (mm, mm^2/s) with
`uc = dx * density * 1e-6` to kg; with square cells the same equations hold
in mass units with `M = d A` (A = dx dy rho), `Q = M v / dx`:

    M2 = [M1/dt + 0.5 (Qin2 - Q1 + Qin1) + (Det - Dep)] / (1/dt + 0.5 v/dx),
    Q2 = M2 v / dx,

all in kg, kg/s, m/s, m. `Det` and `Dep` are the per-step detachment and
source-based deposition RATES (kg/s), exactly as the legacy's `detach_soil`
and `depos_soil` (mm/s) are per-step rates at dt = 1 s.

Explicit legacy accounting (docs/phase7/legacy_sediment_ledger.md):

- Deposition is SOURCE-based: the walk credits `depos_soil` of the receiving
  cells at the moment of detachment, and that deposition is debited from the
  receiving cell's MOBILE pool in its own Crank-Nicolson source term
  (`detach - depos`) before the detached mass has arrived there as flux, so
  a receiving pool can go negative; the legacy clips it to zero, which
  CREATES mass. The clipped trial depth
  changes the implicit outflow too, so the effective artificial source is
  `(-M2_trial) (1 + 0.5 dt v / dx)` (kg). It is returned as
  `clipping_source_kg` per cell and class and reported, never hidden.
- Ring deposition is a separate walk diagnostic, returned as a RATE (kg/s).
  It is not debited by the active-pool equation below and must not be added
  to CN export as another sink. Truncated walk tails likewise remain in
  that equation through `Det - Dep_active`. A caller integrating the ring
  diagnostic over each step multiplies its rate by `dt`.
- Composition does not evolve (update_sediment_flow.for keeps `sed_propn`
  fixed) and supply is unlimited: detachment is never capped.

Identity per step and class (checked by the tests to FP64 roundoff):

    sum(M2) - sum(M1) = (sum Det - sum Dep_active) dt - CN_export + sum clip,
    CN_export = sum_outlets 0.5 (Q1 + Q2) dt.

Nothing here touches MAPLE state. The kernels are plain loops compiled with
Numba when importable (same code runs in pure Python for tests).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from maple_syrup.routing import DONOR_SLOTS, EXPORT, RoutingGraph
from maple_syrup.sediment_physics import REGIME_CODES

__all__ = [
    "LEGACY_WALK_LIMIT_M",
    "LegacyNetwork",
    "LegacyStep",
    "LegacyTransportError",
    "legacy_network",
    "legacy_transport_step",
    "walk_limit_cells",
    "walk_limits_by_regime",
]

#: initialize_values_xml.f90 261-263: path limits (m) of the deposition walk.
LEGACY_WALK_LIMIT_M = {"diffuse": 10.0, "concentrated": 100.0, "suspended": 500.0}
#: flow_distrib.for 97: lower limit on the previous fraction, `10.d-20 * dt`.
_VGE_PER_S = 1.0e-19


class LegacyTransportError(ValueError):
    pass


def walk_limit_cells(limit_m: float, dx_m: float) -> int:
    """`max(int(limit / dx + 0.5), 2)`, the legacy cell count of a walk."""
    return max(int(limit_m / dx_m + 0.5), 2)


def walk_limits_by_regime(dx_m: float) -> np.ndarray:
    """Walk limit (cells) indexed by `REGIME_CODES` value; zero where no law."""
    out = np.zeros(max(REGIME_CODES.values()) + 1, dtype=np.int64)
    out[REGIME_CODES["diffuse"]] = out[REGIME_CODES["transitional_rain"]] = walk_limit_cells(
        LEGACY_WALK_LIMIT_M["diffuse"], dx_m)
    out[REGIME_CODES["concentrated"]] = out[REGIME_CODES["transitional_dry"]] = walk_limit_cells(
        LEGACY_WALK_LIMIT_M["concentrated"], dx_m)
    out[REGIME_CODES["suspended"]] = walk_limit_cells(LEGACY_WALK_LIMIT_M["suspended"], dx_m)
    return out


@dataclass(frozen=True)
class LegacyNetwork:
    """Host connectivity for the legacy operator on a frozen routing graph.

    `receiver` (n,) int64: flat receiver, `EXPORT` (-1) for outlets (the
    ring); `donors` (n, 4) int64: donor flat index per legacy direction slot
    (south, west, north, east: `route_sediment_xml` `sdirin` l = 1..4), -1
    where none; `order` (n_active,) int64: active cells, every donor before
    its receiver (the legacy `order` array sorted by contributing area is
    one such order; the CN result does not depend on which); `outlet`,
    `active` (n,) bool.
    """

    shape: tuple[int, int]
    dx_m: float
    receiver: np.ndarray
    donors: np.ndarray
    order: np.ndarray
    outlet: np.ndarray
    active: np.ndarray
    graph_input_sha256: str


def legacy_network(graph: RoutingGraph) -> LegacyNetwork:
    if not isinstance(graph, RoutingGraph):
        raise LegacyTransportError("graph must be a maple_syrup.routing.RoutingGraph")
    ny, nx = graph.shape
    n = ny * nx
    receiver = np.asarray(graph.receiver, dtype=np.int64).reshape(-1).copy()
    active = np.asarray(graph.active, dtype=np.bool_).reshape(-1)
    outlet = np.asarray(graph.outlet, dtype=np.bool_).reshape(-1)
    if np.any(active & ~outlet & (receiver < 0)):
        raise LegacyTransportError("an active non-outlet cell has no receiver")
    receiver[outlet] = EXPORT
    receiver[~active] = EXPORT
    donors = np.full((n, 4), -1, dtype=np.int64)
    for slot, (dr, dc) in enumerate(DONOR_SLOTS):
        for r in range(ny):
            for c in range(nx):
                rr, cc = r + dr, c + dc
                if 0 <= rr < ny and 0 <= cc < nx:
                    d = rr * nx + cc
                    if active[d] and receiver[d] == r * nx + c:
                        donors[r * nx + c, slot] = d
    # Topological order (Kahn) over active cells: donors strictly before receivers.
    indeg = np.zeros(n, dtype=np.int64)
    for i in np.flatnonzero(active & ~outlet):
        indeg[receiver[i]] += 1
    order = []
    queue = [int(i) for i in np.flatnonzero(active) if indeg[i] == 0]
    while queue:
        i = queue.pop()
        order.append(i)
        r = receiver[i]
        if r >= 0:
            indeg[r] -= 1
            if indeg[r] == 0:
                queue.append(int(r))
    if len(order) != int(active.sum()):
        raise LegacyTransportError("routing graph is not acyclic over active cells")
    return LegacyNetwork((ny, nx), float(graph.dx_m), receiver, donors, np.asarray(order, dtype=np.int64),
                         outlet, active, graph.input_sha256)


# --- kernels (plain loops; Numba-compiled when available) -----------------------------------------------
def _walk_py(det, inv_L, law, nsteps, receiver, dx, dt, depos, ring):
    n, nc = det.shape
    vge = _VGE_PER_S * dt
    for i in range(n):
        for k in range(nc):
            d = det[i, k]
            if d <= 0.0:
                continue
            if not law[i, k]:
                depos[i, k] += d  # conc_flow_transport 49-54: no capacity, deposit locally
                continue
            par = inv_L[i, k]
            fract = 1.0 - math.exp(-par * dx)
            depos[i, k] += fract * d
            pos = receiver[i]
            step = 0
            limit = nsteps[i, k]
            while step < limit and fract > vge:
                u = (step + 2.0) * dx
                lo = u - dx
                fract = math.exp(-lo * par) - math.exp(-u * par)
                if pos < 0:
                    ring[k] += d * fract  # ring cell receives, then the walk exits on its zero aspect
                    break
                depos[pos, k] += d * fract
                pos = receiver[pos]
                step += 1


def _cn_py(order, donors, receiver, outlet, M1, Q1, Qin1, det, dep, v, dt, dx, M2, Q2, Qin2, clip, cn_export,
           endpoint_export):
    nc = M1.shape[1]
    for idx in range(order.shape[0]):
        i = order[idx]
        for k in range(nc):
            qin = 0.0
            for s in range(4):
                d = donors[i, s]
                if d >= 0 and Q2[d, k] >= 0.0:
                    qin += Q2[d, k]
            Qin2[i, k] = qin
            rhs = M1[i, k] / dt + 0.5 * (qin - Q1[i, k] + Qin1[i, k]) + (det[i, k] - dep[i, k])
            m = rhs / (1.0 / dt + 0.5 * v[i, k] / dx)
            if m < 0.0:
                clip[i, k] = -m * (1.0 + 0.5 * dt * v[i, k] / dx)
                m = 0.0
            M2[i, k] = m
            q = m * v[i, k] / dx
            Q2[i, k] = q
            if outlet[i]:
                cn_export[k] += 0.5 * (Q1[i, k] + q) * dt
                endpoint_export[k] += q * dt


try:  # compiled kernels when Numba is importable; identical code otherwise
    import numba as _numba

    _walk = _numba.njit(cache=False, fastmath=False)(_walk_py)
    _cn = _numba.njit(cache=False, fastmath=False)(_cn_py)
    KERNEL_IMPLEMENTATION = "numba"
except Exception:  # noqa: BLE001 - optional Numba import can fail on incompatible installations
    # pragma: no cover - environment dependent
    _walk, _cn = _walk_py, _cn_py
    KERNEL_IMPLEMENTATION = "python"


@dataclass(frozen=True)
class LegacyStep:
    """One legacy transport step. Cell arrays are flat `(n_cells, n_classes)` kg or kg/s."""

    dt_s: float
    detachment_rate_kg_s: np.ndarray  # Det (uncapped legacy detachment)
    deposition_rate_kg_s: np.ndarray  # Dep RATE into active cells (source-based walk + local no-capacity deposits)
    ring_deposition_rate_kg_s: np.ndarray  # (n_classes,) deposition RATE into the boundary ring (outside the active domain)
    mobile_before_kg: np.ndarray  # M1
    mobile_after_kg: np.ndarray  # M2 (after clipping)
    flux_before_kg_s: np.ndarray  # Q1
    flux_after_kg_s: np.ndarray  # Q2
    inflow_before_kg_s: np.ndarray  # Qin1
    inflow_after_kg_s: np.ndarray  # Qin2
    clipping_source_kg: np.ndarray  # artificial mass created by the legacy clip, per cell and class
    cn_export_kg: np.ndarray  # (n_classes,) sum over outlets of 0.5 (Q1 + Q2) dt
    endpoint_export_kg: np.ndarray  # (n_classes,) sum over outlets of Q2 dt (legacy sedtr convention x dt)
    outlet_flux_kg_s: np.ndarray  # (n_classes,) sum over outlets of Q2 (legacy sedtr001 / seddisch001)

    def identity_residual_kg(self) -> np.ndarray:
        """`new - old - (Det - Dep) dt + CN_export - clip` per class; FP64 roundoff when exact."""
        return (self.mobile_after_kg.sum(0) - self.mobile_before_kg.sum(0)
                - (self.detachment_rate_kg_s.sum(0) - self.deposition_rate_kg_s.sum(0)) * self.dt_s
                + self.cn_export_kg - self.clipping_source_kg.sum(0))


def legacy_transport_step(network: LegacyNetwork, detachment_rate_kg_s, inverse_travel_distance_per_m, law_applies,
                          regime, sediment_velocity_m_s, mobile_before_kg, flux_before_kg_s, inflow_before_kg_s,
                          dt_s: float) -> LegacyStep:
    """One legacy step: source-based deposition walk, then Crank-Nicolson routing.

    Inputs are host `(ny, nx, n_classes)` arrays (or flat `(n, n_classes)`):
    detachment RATE (kg/s, uncapped), `1/L` (per m) where a transport law
    applies, `law_applies` (bool), `regime` (int8 `REGIME_CODES`, selects
    the walk limit), legacy virtual velocity (m/s; the law's value where a
    law applies, the 0.9-per-step recession memory elsewhere, never zeroed
    for 'settled' cells), and the previous step's pool, flux and inflow at
    time level 1. Returns the new level-2 fields and the explicit accounting.
    """
    if not isinstance(network, LegacyNetwork):
        raise LegacyTransportError("network must be a LegacyNetwork")
    n = network.receiver.shape[0]
    ny, nx = network.shape
    if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float, np.integer, np.floating)):
        raise LegacyTransportError("dt_s must be a real number")
    dt = float(dt_s)
    if not (math.isfinite(dt) and dt > 0.0):
        raise LegacyTransportError("dt_s must be finite and positive")

    det_in = np.asarray(detachment_rate_kg_s)
    if det_in.ndim == 3 and det_in.shape[:2] == (ny, nx):
        nc = det_in.shape[2]
        accepted = ((ny, nx, nc), (n, nc))
    elif det_in.ndim == 2 and det_in.shape[0] == n:
        nc = det_in.shape[1]
        accepted = ((ny, nx, nc), (n, nc))
    else:
        raise LegacyTransportError(f"detachment_rate_kg_s must be (ny, nx, n_classes) or (n_cells, n_classes) "
                                   f"for the {ny}x{nx} network, got {det_in.shape}")
    if nc < 1:
        raise LegacyTransportError("at least one grain class is required")

    def flat(a, dtype, name):
        arr = np.asarray(a)
        if tuple(arr.shape) not in accepted:
            raise LegacyTransportError(f"{name} must have shape {accepted[0]} or {accepted[1]}, got {arr.shape}")
        if dtype is np.float64 and arr.dtype.kind not in "fiu":
            raise LegacyTransportError(f"{name} must be real-valued, got dtype {arr.dtype}")
        arr = np.ascontiguousarray(arr, dtype=dtype).reshape(n, nc)
        if dtype is np.float64 and not np.isfinite(arr).all():
            raise LegacyTransportError(f"{name} must be finite")
        return arr

    det = flat(det_in, np.float64, "detachment_rate_kg_s")
    inv_L = flat(inverse_travel_distance_per_m, np.float64, "inverse_travel_distance_per_m")
    law_in = np.asarray(law_applies)
    if law_in.dtype != np.bool_:
        raise LegacyTransportError("law_applies must be a boolean array")
    law = flat(law_in, np.bool_, "law_applies")
    reg_in = np.asarray(regime)
    if reg_in.dtype.kind not in "iu":
        raise LegacyTransportError("regime must be an integer array of REGIME_CODES")
    reg = flat(reg_in, np.int64, "regime")
    limits = walk_limits_by_regime(network.dx_m)
    if reg.min() < 0 or reg.max() >= limits.size:
        raise LegacyTransportError(f"regime codes must lie in [0, {limits.size - 1}]")
    v = flat(sediment_velocity_m_s, np.float64, "sediment_velocity_m_s")
    M1 = flat(mobile_before_kg, np.float64, "mobile_before_kg")
    Q1 = flat(flux_before_kg_s, np.float64, "flux_before_kg_s")
    Qin1 = flat(inflow_before_kg_s, np.float64, "inflow_before_kg_s")
    for name, arr in (("detachment_rate_kg_s", det), ("inverse_travel_distance_per_m", inv_L),
                      ("sediment_velocity_m_s", v), ("mobile_before_kg", M1), ("flux_before_kg_s", Q1),
                      ("inflow_before_kg_s", Qin1)):
        if np.any(arr < 0.0):
            raise LegacyTransportError(f"{name} must be >= 0")
    if np.any(law & (inv_L <= 0.0) & (det > 0.0)):
        raise LegacyTransportError("a cell with detachment and an applicable law must have a positive 1/L")
    if np.any(law & (reg == REGIME_CODES["dry"])) or np.any(law & (reg == REGIME_CODES["wet_no_law"])):
        raise LegacyTransportError("law_applies is set on a cell whose regime has no transport law")
    inactive = ~network.active
    if np.any(det[inactive] != 0.0) or np.any(M1[inactive] != 0.0) or np.any(v[inactive] != 0.0):
        raise LegacyTransportError("inactive cells must carry no detachment, pool or velocity")
    nsteps = limits[reg]
    depos = np.zeros((n, nc))
    ring = np.zeros(nc)
    _walk(det, inv_L, law, nsteps, network.receiver, network.dx_m, dt, depos, ring)
    M2 = np.zeros((n, nc))
    Q2 = np.zeros((n, nc))
    Qin2 = np.zeros((n, nc))
    clip = np.zeros((n, nc))
    cn_export = np.zeros(nc)
    endpoint_export = np.zeros(nc)
    _cn(network.order, network.donors, network.receiver, network.outlet, M1, Q1, Qin1, det, depos, v, dt,
        network.dx_m, M2, Q2, Qin2, clip, cn_export, endpoint_export)
    outlet_flux = Q2[network.outlet].sum(axis=0)
    return LegacyStep(dt, det, depos, ring, M1, M2, Q1, Q2, Qin1, Qin2, clip, cn_export, endpoint_export,
                      outlet_flux)
