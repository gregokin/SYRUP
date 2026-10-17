"""POND HYDRAULIC FOUNDATION, stage 1 -- static topology of the explicit-outlet D4 network (host side, no hydraulics).

This module builds, validates and refreshes the IMMUTABLE description a pond hydraulic solver needs; the solver wrappers and
the step/evolve/ledger/diagnostic helpers are `pond_hydraulics`. Nothing here replaces or changes the accepted method-5
hydrology, the experimental kinematic/local-inertial solvers or any default; no sediment, splash, restart or checkpoint
coupling exists yet (stage 2). Nothing here was run by its author (file-only tools); Codex records results.

WHAT IS REUSED. The staggered face arrays (type, zmax, friction, normal-flow k_b, sign) are produced by the existing
`experimental_hydrology.build_local_inertial_geometry` with its explicit `open_faces` argument; no face formula is copied.
Interior faces exist between every pair of D4-adjacent ACTIVE cells, in both directions (one signed face carries both
directions), so flats and multicell depressions are connected without any legacy flow direction. The legacy `RoutingGraph`
is NOT consulted here at all (see `pond_hydraulics` for its incidental column-metadata role).

EXPLICIT OUTLETS. An outlet is `OutletFace(row, col, d_row, d_col)`: an ACTIVE interior cell and one D4 direction that leaves the
interior onto the one-cell ring. Outlets are never inferred (a flat or ponded cell may have no legacy outlet). The boundary law
is the existing Darcy NORMAL-FLOW closure `q = k_b h^(3/2)`, `k_b = sqrt(8 g S_b / f)`, `S_b = (z_cell - z_ring) / dx`, so the
existing solver REQUIRES a real strictly positive bed drop to the ring cell. This stage therefore REFUSES zero-drop, uphill and
backwater/stage-controlled outlets with a `PondNetworkError`. No minimum slope, ghost slope or bed perturbation is invented; a
future explicit-stage boundary is a separate scope. Flat interiors with a genuinely stepped-down ring cell are valid.

ESCAPE (SPILL) LEVELS. For every active cell c, `escape_level_m[c]` is the D4 MINIMAX level to an outlet:

    escape(c) = min over D4 paths c = p0, ..., pk = outlet cell of  max_i z[p_i]

computed once by a host priority queue (Dijkstra with node weights, ties broken by (level, flat index), O(n log n)) at the
initial build and at a NET-elevation refresh only -- never inside a step. Because the normal-flow outlet discharges for any
positive depth, an outlet cell's level is its own bed `z`. Values are exact bed values (only `max` is applied, no arithmetic).
The cell's depth below the level is `fill_depth_m = escape - z` >= 0. Cells with no path to an outlet belong to a CLOSED
COMPONENT: `escape_level_m = +inf`, `outlet_connected = False`, `closed_component_label >= 0` (connected components of the
active D4 graph; inactive cells and connected cells carry -1). A closed component is refused unless the caller explicitly
designates it with `allow_closed_components=True` (closed basins are legitimate for lake tests; they are never silent).
`escape_path_next[c]` is the flat index of the next cell on one minimax path (-1: the cell is an outlet cell, -2: closed or
inactive); it is a tie-broken spanning-tree pointer, NOT a claimed unique spill face. No basin labels or spill faces are
claimed because none are computed: a single-cell neighbour lip `max(z_c, z_neighbour)` is only a local face lip, not a
multicell spill level, and is never used as one.

IMMUTABILITY. Every array is a private read-only copy; the topology, source elevation, active mask, friction, outlets,
geometry and digests are frozen. Face ACTIVATION during a step depends on water head (`h_face = max(eta) - zmax`), never on a
topology rebuild: the escape levels are diagnostics/topology for coupling, not a switch of the hydraulics.

NET-ELEVATION REFRESH (`update_pond_topology`). The caller passes the ACTUAL new physical bed `(ny+2, nx+2)`, the last held
topology (whose `source_z_full` is the last held routing-source elevation) and the already RESOLVED MAPLE threshold in the
bed's units (m). The gate is exactly

    max over ACTIVE cells |z_new - z_held| >= threshold_m

(per-cell NET drift versus the held source: same-cell deltas that cancel never trigger, deltas in different cells never add or
cancel each other, the held source only moves when a refresh happens, so slow monotone drift accumulates until it triggers).
No accounting, clock, water-level or step-count trigger exists. Below threshold the SAME topology object is returned (exact old
geometry and source). When triggered, a new topology is built and fully validated (including the positive outlet drops) before
being returned; nothing is published by this module -- the caller assigns the result, and must have committed the native bed
BEFORE adopting it (this module neither reads nor writes any native bed store). The source mask, friction, dx and outlet face
identity are retained and cannot change in a refresh (`identity_sha256` is invariant); the ring must equal the held ring. A
refresh changes only `geometry.z/zmax/k_b`, the source digest and the escape levels, so a carried face-flux state (`qx, qy`),
depths and soil water stay valid and are carried by the caller unchanged (`pond_hydraulics.rebind_pond`).

ARRAY CONVENTIONS. Row 0 = south; interior `(ny, nx)`; full bed `(ny+2, nx+2)` with the one-cell ring; x faces `(ny, nx+1)`
positive east, y faces `(ny+1, nx)` positive north. Units: elevations m, `dx_m` m, thresholds m. Float64 throughout.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup.experimental_hydrology import (
    ExperimentalGeometryError,
    LocalInertialGeometry,
    build_local_inertial_geometry,
)

__all__ = [
    "OutletFace",
    "PondNetworkError",
    "PondTopology",
    "TopologyUpdate",
    "build_pond_topology",
    "update_pond_topology",
]

_D4 = ((1, 0), (0, 1), (-1, 0), (0, -1))
_SOURCE_SCHEMA = b"maple_syrup.pond_source.v1"
_IDENTITY_SCHEMA = b"maple_syrup.pond_identity.v1"
_TOPOLOGY_SCHEMA = b"maple_syrup.pond_topology.v1"


class PondNetworkError(ValueError):
    """Invalid or unsupported pond topology request (bed, masks, outlets, threshold). Raised before anything is returned; no
    argument is modified and no previously built topology changes."""


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise PondNetworkError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise PondNetworkError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def _frozen(array: np.ndarray) -> np.ndarray:
    out = np.array(array, copy=True)
    out.flags.writeable = False
    return out


def _host(array: Any, name: str, dtype: Any, ndim: int = 2) -> np.ndarray:
    if type(array) is not np.ndarray or array.dtype != np.dtype(dtype) or array.ndim != ndim:
        got = f"{type(array).__name__} {getattr(array, 'dtype', '')}"
        raise PondNetworkError(f"{name} must be a host {np.dtype(dtype)} numpy array with {ndim} dimensions, got {got}")
    return array


def _digest(schema: bytes, *parts: Any) -> str:
    h = hashlib.sha256(schema)
    for part in parts:
        if isinstance(part, np.ndarray):
            h.update(f"{part.dtype}{part.shape}".encode())
            h.update(np.ascontiguousarray(part).tobytes())
        else:
            h.update(repr(part).encode())
    return h.hexdigest()


@dataclass(frozen=True, order=True)
class OutletFace:
    """Explicit exterior outlet: the face of ACTIVE interior cell `(row, col)` in D4 direction `(d_row, d_col)` (north = +row,
    east = +col) that leads onto the ring."""

    row: int
    col: int
    d_row: int
    d_col: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.row, self.col, self.d_row, self.d_col)


def _outlet_faces(outlets: Any) -> tuple[OutletFace, ...]:
    if isinstance(outlets, (str, bytes)) or not hasattr(outlets, "__iter__"):
        raise PondNetworkError("outlets must be an iterable of OutletFace or (row, col, d_row, d_col)")
    out = []
    for item in outlets:
        parts = item.as_tuple() if isinstance(item, OutletFace) else item
        if not isinstance(parts, (tuple, list)) or len(parts) != 4:
            raise PondNetworkError(f"an outlet must be an OutletFace or a (row, col, d_row, d_col) tuple, got {item!r}")
        out.append(OutletFace(*(_strict_int(v, "outlet entry") for v in parts)))
    if len(set(out)) != len(out):
        raise PondNetworkError("an outlet face is listed twice")
    return tuple(sorted(out))


def _escape(z: np.ndarray, active: np.ndarray, outlet_cells: set[tuple[int, int]]):
    """Node-weighted D4 minimax levels to the outlet cells by a host priority queue. Returns `(level, next, label, n_closed)`."""
    ny, nx = z.shape
    n = ny * nx
    zl, al = z.reshape(-1).tolist(), active.reshape(-1).tolist()
    level = [math.inf] * n
    nxt = [-2] * n
    heap: list[tuple[float, int]] = []
    for r, c in sorted(outlet_cells):
        i = r * nx + c
        level[i], nxt[i] = zl[i], -1
        heap.append((zl[i], i))
    heapq.heapify(heap)
    done = [False] * n
    while heap:
        lv, i = heapq.heappop(heap)
        if done[i]:
            continue
        done[i] = True
        r, c = divmod(i, nx)
        for dr, dc in _D4:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < ny and 0 <= cc < nx):
                continue
            j = rr * nx + cc
            if not al[j] or done[j]:
                continue
            cand = lv if lv > zl[j] else zl[j]
            if cand < level[j]:
                level[j], nxt[j] = cand, i
                heapq.heappush(heap, (cand, j))
    label = [-1] * n
    n_closed = 0
    for start in range(n):
        if not al[start] or done[start] or label[start] >= 0:
            continue
        label[start] = n_closed
        stack = [start]
        while stack:
            r, c = divmod(stack.pop(), nx)
            for dr, dc in _D4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < ny and 0 <= cc < nx:
                    j = rr * nx + cc
                    if al[j] and not done[j] and label[j] < 0:
                        label[j] = n_closed
                        stack.append(j)
        n_closed += 1
    return (np.array(level, dtype=np.float64).reshape(ny, nx), np.array(nxt, dtype=np.int64).reshape(ny, nx),
            np.array(label, dtype=np.int32).reshape(ny, nx), n_closed)


@dataclass(frozen=True, eq=False)
class PondTopology:
    """Immutable pond network. All arrays are read-only host copies.

    shape, dx_m              `(ny, nx)`, cell size m
    source_z_full            `(ny+2, nx+2)` HELD routing-source elevation m (the bed the geometry and escape levels were built from)
    active, friction         `(ny, nx)` bool / float64 (Darcy-Weisbach f, finite > 0 on active cells)
    nodata_value             sentinel or None (only inactive cells and the ring may hold it)
    outlets                  tuple of `OutletFace`, sorted
    geometry                 the existing `LocalInertialGeometry` (faces; `geometry.z` is the held interior bed)
    escape_level_m           `(ny, nx)` D4 minimax level m (+inf on closed components, NaN-free; inactive cells +inf)
    escape_path_next         `(ny, nx)` int64 flat index of the next cell on a minimax path (-1 outlet cell, -2 closed/inactive)
    outlet_connected         `(ny, nx)` bool, active cell with a path to an outlet
    closed_component_label   `(ny, nx)` int32, >= 0 on closed-component cells else -1
    n_closed_components      number of closed components
    generation               0 for the initial build, +1 per triggered refresh
    identity_sha256          invariant across refreshes (mask, friction, dx, outlets, nodata)
    source_sha256            digest of `source_z_full`
    topology_sha256          digest of identity + source + geometry + escape levels
    """

    shape: tuple[int, int]
    dx_m: float
    source_z_full: np.ndarray
    active: np.ndarray
    friction: np.ndarray
    nodata_value: float | None
    outlets: tuple[OutletFace, ...]
    geometry: LocalInertialGeometry
    escape_level_m: np.ndarray
    escape_path_next: np.ndarray
    outlet_connected: np.ndarray
    closed_component_label: np.ndarray
    n_closed_components: int
    generation: int
    identity_sha256: str
    source_sha256: str
    topology_sha256: str

    @property
    def source_bed_m(self) -> np.ndarray:
        """Held interior routing-source elevation `(ny, nx)` (read-only view)."""
        return self.source_z_full[1:-1, 1:-1]

    @property
    def fill_depth_m(self) -> np.ndarray:
        """`escape_level - z` on connected active cells (>= 0), +inf on closed active cells, 0 on inactive cells."""
        with np.errstate(invalid="ignore"):
            depth = self.escape_level_m - self.source_bed_m
        return _frozen(np.where(self.active, depth, 0.0))

    def interior_faces(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Flat cell indices `(cell_a, cell_b)` of every interior (type 1) face: `x`: a = west, b = east; `y`: a = south, b =
        north. Both directions are carried by the one signed face."""
        ny, nx = self.shape
        g = self.geometry
        rx, jx = np.nonzero(g.fx_type == 1)
        ry, jy = np.nonzero(g.fy_type == 1)
        return {"x": (rx * nx + jx - 1, rx * nx + jx), "y": ((jy + (ry - 1) * nx), (jy + ry * nx))}

    def summary(self) -> dict[str, Any]:
        finite = self.escape_level_m[np.isfinite(self.escape_level_m) & self.active]
        return {
            "shape": list(self.shape), "dx_m": self.dx_m, "generation": self.generation,
            "n_active": int(np.count_nonzero(self.active)),
            "n_interior_faces": self.geometry.summary()["n_interior_faces"],
            "outlets": [o.as_tuple() for o in self.outlets], "n_outlets": len(self.outlets),
            "n_outlet_connected": int(np.count_nonzero(self.outlet_connected)),
            "n_closed_components": self.n_closed_components,
            "escape_level_range_m": None if finite.size == 0 else [float(finite.min()), float(finite.max())],
            "identity_sha256": self.identity_sha256, "source_sha256": self.source_sha256,
            "topology_sha256": self.topology_sha256,
            "boundary": "explicit Darcy normal-flow outlets with a strictly positive real bed drop; zero-drop and "
                        "stage/backwater boundaries are refused in this stage",
        }


def _check_bed(bed_full_m: Any, nodata_value: float | None) -> np.ndarray:
    z = _host(bed_full_m, "bed_full_m", np.float64)
    if min(z.shape) < 3:
        raise PondNetworkError(f"bed_full_m needs a ring and at least one interior cell, got {z.shape}")
    if not np.all(np.isfinite(z)):
        raise PondNetworkError("bed_full_m must be finite everywhere (ring included)")
    return z


def _assemble(z_full: np.ndarray, active: np.ndarray, friction: np.ndarray, dx: float, outlets: tuple[OutletFace, ...],
              nodata: float | None, allow_closed: bool, generation: int) -> PondTopology:
    ny, nx = z_full.shape[0] - 2, z_full.shape[1] - 2
    if active.shape != (ny, nx) or friction.shape != (ny, nx):
        raise PondNetworkError(f"active and friction_factor must have shape {(ny, nx)}")
    if not active.any():
        raise PondNetworkError("no active cell")
    z = z_full[1:-1, 1:-1]
    if nodata is not None:
        if np.any(z[active] == nodata):
            raise PondNetworkError("an ACTIVE cell holds the nodata elevation")
    for o in outlets:
        if abs(o.d_row) + abs(o.d_col) != 1:
            raise PondNetworkError(f"outlet {o.as_tuple()} must use one of the four D4 directions")
        if not (0 <= o.row < ny and 0 <= o.col < nx) or not active[o.row, o.col]:
            raise PondNetworkError(f"outlet {o.as_tuple()} is not on an active interior cell")
        tr, tc = o.row + o.d_row, o.col + o.d_col
        if 0 <= tr < ny and 0 <= tc < nx:
            raise PondNetworkError(f"outlet {o.as_tuple()} leads into the interior; interior open holes are unsupported")
        ring = z_full[tr + 1, tc + 1]
        if nodata is not None and ring == nodata:
            raise PondNetworkError(f"outlet {o.as_tuple()} faces a ring cell holding the nodata value; a genuine bed is needed")
        drop = (z_full[o.row + 1, o.col + 1] - ring) / dx
        if not drop > 0.0:
            raise PondNetworkError(
                f"outlet {o.as_tuple()} has no strictly positive bed drop to the ring (drop {drop} m/m): the existing "
                "Darcy normal-flow boundary needs a real one. Zero-drop, uphill and backwater/stage-controlled outlets are "
                "unsupported in this stage; no minimum slope or ghost slope is invented")
    try:
        geometry = build_local_inertial_geometry(np.array(z_full, copy=True), np.array(active, copy=True),
                                                 np.array(friction, copy=True), dx, [o.as_tuple() for o in outlets])
    except ExperimentalGeometryError as exc:
        raise PondNetworkError(f"geometry construction refused: {exc}") from exc
    # nothing may be open through an inactive or nodata cell (the builder closes every face touching an inactive cell)
    touches_x = np.zeros((ny, nx + 1), dtype=bool)
    touches_x[:, :-1] |= ~active
    touches_x[:, 1:] |= ~active
    touches_y = np.zeros((ny + 1, nx), dtype=bool)
    touches_y[:-1, :] |= ~active
    touches_y[1:, :] |= ~active
    if np.any((geometry.fx_type != 0) & touches_x) or np.any((geometry.fy_type != 0) & touches_y):
        raise PondNetworkError("internal error: a face next to an inactive cell is open")
    level, nxt, label, n_closed = _escape(z, active, {(o.row, o.col) for o in outlets})
    level = np.where(active, level, np.inf)
    if n_closed and not allow_closed:
        raise PondNetworkError(
            f"{n_closed} closed component(s) (active cells with no path to an outlet) and allow_closed_components is False; "
            "designate closed basins explicitly or add an outlet")
    identity = _digest(_IDENTITY_SCHEMA, active, friction, dx, tuple(o.as_tuple() for o in outlets), nodata)
    source = _digest(_SOURCE_SCHEMA, z_full)
    topo = _digest(_TOPOLOGY_SCHEMA, identity, source, geometry.input_sha256, level)
    return PondTopology(
        shape=(ny, nx), dx_m=dx, source_z_full=_frozen(z_full), active=_frozen(active), friction=_frozen(friction),
        nodata_value=nodata, outlets=outlets, geometry=geometry, escape_level_m=_frozen(level),
        escape_path_next=_frozen(np.where(np.isfinite(level), nxt, -2)), outlet_connected=_frozen(np.isfinite(level)),
        closed_component_label=_frozen(label), n_closed_components=n_closed, generation=generation,
        identity_sha256=identity, source_sha256=source, topology_sha256=topo)


def build_pond_topology(bed_full_m: np.ndarray, active: np.ndarray, friction_factor: np.ndarray, dx_m: float,
                        outlets: Any, *, nodata_value: float | None = None,
                        allow_closed_components: bool = False) -> PondTopology:
    """Initial build. `bed_full_m (ny+2, nx+2)` float64 finite (ring included, row 0 = south), `active (ny, nx)` bool,
    `friction_factor (ny, nx)` float64 (> 0 on active cells), `dx_m` > 0, `outlets` an iterable of `OutletFace` or
    `(row, col, d_row, d_col)`. Raises `PondNetworkError` on anything unsupported; nothing is returned then."""
    z = _check_bed(bed_full_m, nodata_value)
    act = _host(active, "active", np.bool_)
    fric = _host(friction_factor, "friction_factor", np.float64)
    dx = _real(dx_m, "dx_m")
    if not (math.isfinite(dx) and dx > 0.0 and math.isfinite(dx * dx) and dx * dx > 0.0):
        raise PondNetworkError(f"dx_m must be finite with a finite positive cell area, got {dx_m!r}")
    nodata = None if nodata_value is None else _real(nodata_value, "nodata_value")
    if nodata is not None and not math.isfinite(nodata):
        raise PondNetworkError("nodata_value must be finite")
    if type(allow_closed_components) is not bool:
        raise PondNetworkError("allow_closed_components must be a bool")
    interior = (z.shape[0] - 2, z.shape[1] - 2)
    if act.shape != interior or fric.shape != interior:
        raise PondNetworkError(f"active and friction_factor must have shape {interior}")
    if not (np.all(np.isfinite(fric[act])) and np.all(fric[act] > 0.0)):
        raise PondNetworkError("friction_factor must be finite and > 0 on every active cell")
    return _assemble(z, act, fric, dx, _outlet_faces(outlets), nodata, allow_closed_components, 0)


@dataclass(frozen=True, eq=False)
class TopologyUpdate:
    """Outcome of `update_pond_topology`. `topology` is the SAME object as the input when `refreshed` is False."""

    topology: PondTopology
    refreshed: bool
    max_abs_delta_m: float
    threshold_m: float
    n_cells_at_or_above: int


def update_pond_topology(topology: PondTopology, new_bed_full_m: np.ndarray, threshold_m: float) -> TopologyUpdate:
    """Net-elevation gate and (when triggered) a purely staged rebuild; see the module docstring. `new_bed_full_m` is the
    ACTUAL new physical bed (ring unchanged), `threshold_m` the resolved MAPLE threshold (finite, > 0). The new bed is always
    validated, even below threshold. Raises `PondNetworkError`; the input topology is never modified."""
    if not isinstance(topology, PondTopology):
        raise PondNetworkError(f"topology must be a PondTopology, got {type(topology).__name__}")
    threshold = _real(threshold_m, "threshold_m")
    if not (math.isfinite(threshold) and threshold > 0.0):
        raise PondNetworkError(f"threshold_m must be finite and > 0 (the resolved MAPLE threshold), got {threshold_m!r}")
    z = _check_bed(new_bed_full_m, topology.nodata_value)
    held = topology.source_z_full
    if z.shape != held.shape:
        raise PondNetworkError(f"new bed shape {z.shape} != held shape {held.shape}")
    ring = np.ones(z.shape, dtype=bool)
    ring[1:-1, 1:-1] = False
    if not np.array_equal(z[ring], held[ring]):
        raise PondNetworkError("the ring of the new bed differs from the held ring; outlet identity cannot change in a refresh")
    if topology.nodata_value is not None and np.any(z[1:-1, 1:-1][topology.active] == topology.nodata_value):
        raise PondNetworkError("an ACTIVE cell of the new bed holds the nodata elevation")
    with np.errstate(over="ignore"):
        delta = np.abs(z[1:-1, 1:-1] - held[1:-1, 1:-1])[topology.active]
    if not np.all(np.isfinite(delta)):
        raise PondNetworkError("the bed change overflows FP64")
    max_delta = float(delta.max())
    n_at = int(np.count_nonzero(delta >= threshold))
    if not max_delta >= threshold:
        return TopologyUpdate(topology, False, max_delta, threshold, n_at)
    new = _assemble(z, np.asarray(topology.active), np.asarray(topology.friction), topology.dx_m, topology.outlets,
                    topology.nodata_value, topology.n_closed_components > 0, topology.generation + 1)
    if new.identity_sha256 != topology.identity_sha256:
        raise PondNetworkError("internal error: the refresh changed the topology identity")
    return TopologyUpdate(new, True, max_delta, threshold, n_at)
