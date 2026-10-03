"""D4 routing graph and MAHLERAN method-5 hydraulic step (Phase 4b).

Reference, MAHLERAN 1.2.3 (read-only), line numbers as read 2026-09-29:

- `Subroutines_In_out/topog_attrib.for`: aspect 94-117 (lowest D4
  neighbour, strict `<`, N E S W order, elevation in mm), slope 117,
  masking 190-197, cap 218-224, edge rule 265-293, upstream-first order
  232-303.
- `Subroutines_Water/route_water.for`, `iroute = 5` (527-925) with the
  static friction type 1 (storm_setting 563-568): donors summed in `sdirin`
  order (20, 535-568); Crank-Nicolson balance 572-574; `q = d v`,
  `v = sqrt(8 g d S / ff)` (21, 921-923); ff floored to 0.1 AFTER the root
  (916-920).

The discrete equation per active cell, in SI (depth m, unit discharge
m2/s, `c = dt / (2 dx)`):

    R     = h_start + c (Qin_old + Qin_new - q_old)
    solve   h + c q(h) = R,   q(h) = k h^{3/2},   k = sqrt(8 g S / f)
    h_new = R - c q_new          (storage identity; exact mass balance)
    F     = dx^2 c (q_old + q_new)  (face volume to the receiver or export)

`h_start` is the post rain/infiltration surface storage (legacy
`d(1) + excess dt`), `old_flow_depth_m` is the legacy post-infiltration
`d(1)` and `q_old` the legacy `q(1)`. Departures from the literal routine,
all documented in docs/phase4/routing.md:

- The receiver's old inflow is the sum of the SAME `q_old` values the
  donors use for their own outflow (coherent face flux). The literal legacy
  inflow `qin(1)` is last step's value, stale after run-on infiltration;
  `legacy_stale_inflow_step` reproduces it and reports the water it creates.
- The root is bracketed by the proven interval `[0, R]` and found by
  bisection with a fixed iteration count (no host synchronization per
  level), not the legacy `[0, 100 (d(1) + excess)]` bracket. The storage is
  closed by the identity above; the constitutive residual `h_new - h_flow`
  is checked against `root_tolerance_m` and never clipped.
- A Courant condition `q_old dt / (h_old dx) <= courant_max <= 2`, which
  guarantees `R >= 0`, rejects the step instead of the legacy STOP on a
  negative right-hand side.
- Sinks, flats, receivers that would lose water silently, zero slopes,
  friction below the legacy floor 0.1 and nodata elevations are rejected
  when the graph is built. There is no filling or pit storage.

Two implementations of the ordered sweep share everything else:
`implementation="array"` (default; MAPLE NumPy/CuPy namespace, one Python
loop over dependency levels and bisection iterations) and
`implementation="numba"` (optional, CPU NumPy only, one compiled call; see
routing_numba.py). There is no silent fallback between them.

The root solver is selectable: `root_solver="bisection"` (default, unchanged) or `"newton"` (routing_newton.py: a
safeguarded Newton iteration on the SAME equation with a guaranteed bracket and bisection fallback; CPU NumPy
graphs only, `implementation` "array" or "numba", never "cuda"). Newton is a different root finder for the same
physics, not bitwise equal to bisection; the constitutive, balance and water checks are unchanged.

Every step is pure: inputs are never modified, and a rejected step raises
`RoutingError` before anything is returned. There is never a Python loop
over cells in `route_step`.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

from maple_syrup import routing_newton
from maple_syrup.routing_newton import (
    DEFAULT_NEWTON_MAX_ITERATIONS,
    MAX_NEWTON_ITERATIONS,
    ROOT_SOLVERS,
)

__all__ = [
    "ASPECT_STEPS",
    "BALANCE_RTOL",
    "DEFAULT_BISECTION_ITERATIONS",
    "DEFAULT_COURANT_MAX",
    "DEFAULT_NEWTON_MAX_ITERATIONS",
    "DEFAULT_ROOT_TOLERANCE_M",
    "DONOR_SLOTS",
    "EXPORT",
    "GRAVITY_M_S2",
    "IMPLEMENTATIONS",
    "INACTIVE",
    "LEGACY_FRICTION_FLOOR",
    "PIT_STORAGE",
    "ROOT_SOLVERS",
    "ROUTE_IMPLEMENTATIONS",
    "RouteStep",
    "RoutingError",
    "RoutingGraph",
    "RoutingGraphError",
    "RoutingStepRejected",
    "build_routing_graph",
    "legacy_stale_inflow_step",
    "plot1_routing_graph",
    "route_step",
]

# route_water.for 21: gfconst = 78480 mm/s^2 = 8 g, so g = 9.81 m/s^2.
GRAVITY_M_S2 = 9.81
# Legacy aspect codes (topog_attrib sdir order N, E, S, W) as MAPLE
# (d_row, d_col), row 0 = south: 1 = N (+row), 2 = E, 3 = S (-row), 4 = W.
ASPECT_STEPS = {1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1)}
# route_water.for 20 `sdirin`: donors of cell (i, j) are visited at
# (i+1, j), (i, j-1), (i-1, j), (i, j+1) on the north-first legacy grid, i.e.
# the MAPLE south, west, north and east neighbours, whose aspect must be
# 1, 2, 3, 4 respectively. Inflow is summed in this order.
DONOR_SLOTS = ((-1, 0), (0, -1), (1, 0), (0, 1))
# `receiver` codes besides a flat cell index.
EXPORT = -1
INACTIVE = -2
# OPT-IN terminal storage (build_routing_graph(allow_pit_storage=True) only): an ACTIVE strict D4 sink. It has aspect 0,
# zero slope and zero conveyance, receives its donors' flow and keeps the water (no overtopping, no export). It is
# deliberately distinct from EXPORT (leaves the domain) and INACTIVE (not a computed cell).
PIT_STORAGE = -3
# topog_attrib.for 218-224: slope > 1000 -> 1.
_LEGACY_SLOPE_CAP = 1000.0
# route_water.for 916-920 floors ff to 0.1 after the root is found, so the
# legacy v and q(2) no longer belong to the depth just solved. Below this
# floor the two codes would disagree; the graph refuses such friction.
LEGACY_FRICTION_FLOOR = 0.1

IMPLEMENTATIONS = ("array", "numba")
# `route_step` additionally accepts "cuda" (routing_cuda.py: one RawKernel launch per dependency level, CuPy graph
# only). It is deliberately NOT in IMPLEMENTATIONS, which the storm controls and the experiment CLIs use as their
# choice list: they stay CPU/array only and cannot select a GPU sweep.
ROUTE_IMPLEMENTATIONS = (*IMPLEMENTATIONS, "cuda")
DEFAULT_COURANT_MAX = 1.0
# Bracket width after n halvings is R 2^-n; 40 gives <= 1e-12 R.
DEFAULT_BISECTION_ITERATIONS = 40
# route_water.for 22: the legacy bisection stops at a 1e-8 mm bracket.
DEFAULT_ROOT_TOLERANCE_M = 1.0e-11
_MAX_BISECTION_ITERATIONS = 200

_EPS = float(np.finfo(np.float64).eps)
# Per-cell |dh - c (Qin_old + Qin_new - q_old - q_new)| <= BALANCE_RTOL * scale:
# a handful of roundings of terms no larger than the scale.
BALANCE_RTOL = 32.0 * _EPS
_GRAPH_SCHEMA = b"maple_syrup.routing_graph.v2"


class RoutingGraphError(ValueError):
    """Unsupported or invalid terrain, masks or parameters. Raised while the
    graph is built; nothing is returned."""


class RoutingError(ValueError):
    """Invalid step request, rejected step (Courant, negative right-hand
    side, non-convergence, non-finite output) or failed balance. Raised
    before any result is returned; no caller-owned array is modified."""


class RoutingStepRejected(RoutingError):
    """The step is valid but too large for this state: the old-flux Courant
    number exceeds `courant_max`, or the right-hand side went negative (the
    legacy STOP condition). A caller may retry the SAME state with a smaller
    dt. Every other `RoutingError` (bad inputs, non-convergence, non-finite
    output, failed balance) is not recoverable by shortening the step and is
    deliberately not this class."""


# Messages of the two recoverable rejections; `_route` maps them to
# RoutingStepRejected after the batched flag read.
_COURANT_REJECTION = "Courant number q_old dt / (h_old dx) exceeds"
_NEGATIVE_RHS_REJECTION = "negative right-hand side"


# --- graph -----------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class RoutingGraph:
    """Static D4 routing network on a `(ny, nx)` MAPLE grid (row 0 = south).

    Host NumPy arrays (read-only), `(ny, nx)` unless noted:
      `aspect` int8 legacy codes (0 only on inactive cells); `slope` final
      dimensionless slope; `slope_before_edge_rule`; `edge_rule_applied`;
      `friction_factor`; `active`; `outlet` (active cells exporting across
      the boundary); `receiver` int64 (flat receiver index, `EXPORT` or
      `INACTIVE`); `level` int32 (-1 on inactive cells);
      `level_order_host` int64 flat indices of active cells by (level, index).

    Runtime arrays in namespace `xp` (read-only on NumPy): `conveyance` k
    (flat), `active_flat`, `outlet_flat`, `level_order`, `conveyance_lo`
    (k in level order), `donor_position`/`donor_mask` `(4, n_active)` in
    `DONOR_SLOTS` order, pointing into level-ordered arrays.
    """

    shape: tuple[int, int]
    dx_m: float
    aspect: np.ndarray
    slope: np.ndarray
    slope_before_edge_rule: np.ndarray
    edge_rule_applied: np.ndarray
    friction_factor: np.ndarray
    active: np.ndarray
    outlet: np.ndarray
    receiver: np.ndarray
    level: np.ndarray
    level_order_host: np.ndarray
    level_bounds: tuple[int, ...]
    conveyance: Any
    active_flat: Any
    outlet_flat: Any
    level_order: Any
    conveyance_lo: Any
    donor_position: Any
    donor_mask: Any
    input_sha256: str
    xp: ModuleType
    # Defaults LAST so every existing positional/keyword construction is unchanged. `pit_storage` is the host bool
    # `(ny, nx)` mask of terminal-storage cells (all False for the strict default); `policy` names the build policy.
    pit_storage: Any = None
    policy: str = "strict"

    @property
    def n_active(self) -> int:
        return int(self.level_order_host.size)

    @property
    def n_levels(self) -> int:
        return len(self.level_bounds) - 1

    @property
    def max_level_width(self) -> int:
        return int(max(b - a for a, b in itertools.pairwise(self.level_bounds)))

    def summary(self) -> dict[str, Any]:
        """Host-side description for reports (no device reads)."""
        act = self.active
        raw = self.slope_before_edge_rule[act]
        outlet_cells = [
            {"maple_row": int(r), "maple_col": int(c), "aspect": int(self.aspect[r, c]),
             "slope": float(self.slope[r, c])}
            for r, c in zip(*np.nonzero(self.outlet), strict=True)
        ]
        policy_keys = {}
        if self.policy != "strict":  # absent for the strict default so existing reports are unchanged
            policy_keys = {"policy": self.policy,
                           "n_pit_storage": 0 if self.pit_storage is None else int(np.sum(self.pit_storage))}
        return {
            **policy_keys,
            "shape": list(self.shape),
            "dx_m": self.dx_m,
            "n_active": self.n_active,
            "n_outlets": int(self.outlet.sum()),
            "outlets": outlet_cells,
            "n_levels": self.n_levels,
            "max_level_width": self.max_level_width,
            "level_widths": [b - a for a, b in itertools.pairwise(self.level_bounds)],
            "slope_before_edge_rule_range": [float(raw.min()), float(raw.max())],
            "slope_range": [float(self.slope[act].min()), float(self.slope[act].max())],
            "n_edge_rule_applied": int(self.edge_rule_applied.sum()),
            "aspect_counts": {name: int(np.sum(act & (self.aspect == code)))
                              for code, name in ((1, "N"), (2, "E"), (3, "S"), (4, "W"))},
            "friction_factor_range": [float(self.friction_factor[act].min()),
                                      float(self.friction_factor[act].max())],
            "input_sha256": self.input_sha256,
            "namespace": self.xp.__name__,
        }


def _host_grid(array: Any, name: str, dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise RoutingGraphError(f"{name} must be a host NumPy array (graph setup is host-side), "
                                f"got {type(array).__name__}")
    if array.dtype != dtype:
        raise RoutingGraphError(f"{name} must have dtype {np.dtype(dtype)}, got {array.dtype}")
    if array.ndim != 2:
        raise RoutingGraphError(f"{name} must be 2-D, got shape {array.shape}")
    if shape is not None and array.shape != shape:
        raise RoutingGraphError(f"{name} shape {array.shape} != {shape}")
    return np.array(array, copy=True)


def _cells(mask: np.ndarray, limit: int = 20) -> str:
    """`(row, col)` list for messages, with the legacy 1-based index."""
    ny = mask.shape[0]
    cells = [f"(r={r}, c={c}; legacy i={ny + 1 - r}, j={c + 2})"
             for r, c in itertools.islice(zip(*np.nonzero(mask), strict=True), limit)]
    more = int(mask.sum()) - len(cells)
    return ", ".join(cells) + (f" and {more} more" if more > 0 else "")


def _real(value: Any, name: str, error) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise error(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def build_routing_graph(
    elevation_full_m: np.ndarray,
    export_receiver_full: np.ndarray,
    friction_factor: np.ndarray,
    dx_m: float,
    *,
    dy_m: float | None = None,
    active_mask: np.ndarray | None = None,
    nodata_value: float | None = None,
    xp: ModuleType | None = None,
    allow_masked_nodata: bool = False,
    allow_pit_storage: bool = False,
) -> RoutingGraph:
    """Build the legacy D4 network from host NumPy arrays.

    OPT-IN POLICIES (both default OFF; with both off the behaviour, errors and digest are exactly the strict ones):
    `allow_masked_nodata` (needs `nodata_value`) lets cells equal to the sentinel exist on INACTIVE cells and the ring.
    They are never a receiver (the legacy `topog_attrib.for` skip of `nodata_value_from_topog` neighbours), so no false
    low neighbour arises; an ACTIVE cell holding the sentinel is still refused. `allow_pit_storage` keeps every ACTIVE
    strict sink (no strictly lower neighbour, no equal neighbour) as a terminal STORAGE cell: aspect 0, receiver
    `PIT_STORAGE`, slope 0, conveyance 0, not an outlet. It receives its donors' flow and retains the water; nothing
    is filled, carved or exported and there is no overtopping. Flat sinks (an equal neighbour) are still refused.

    `elevation_full_m` `(ny + 2, nx + 2)` float64, MAPLE orientation (row 0 =
    south), including a one-cell boundary ring; every value finite and, when
    `nodata_value` is given, different from it everywhere (the legacy nodata
    skip in the aspect search is NOT reproduced: a finite sentinel such as
    -9999 would otherwise become a false low receiver). `export_receiver_full`
    bool, same shape: non-active cells (ring or inactive interior) that
    EXPORT what flows into them -- the legacy `rmask < 0` receivers.
    `friction_factor` `(ny, nx)` float64 static Darcy-Weisbach f, finite and
    >= LEGACY_FRICTION_FLOOR (0.1) on active cells. `active_mask` `(ny, nx)`
    bool, default all True (legacy `rmask >= 0` interior).

    Rejects: non-square cells; a cell area dx^2 that overflows or underflows;
    sinks/flats (no strictly lower neighbour); receivers that are neither
    active nor export-flagged (legacy loses that water silently); export
    flags on active cells; an edge rule that would read a slope outside the
    active interior; zero slope after the edge rule.
    """
    from maple.core.backend import freeze, to_device

    z = _host_grid(elevation_full_m, "elevation_full_m", np.float64)
    if min(z.shape) < 3:
        raise RoutingGraphError(f"elevation_full_m needs a ring and >= 1 interior cell, got {z.shape}")
    ny, nx = z.shape[0] - 2, z.shape[1] - 2
    export = _host_grid(export_receiver_full, "export_receiver_full", np.bool_, z.shape)
    ff = _host_grid(friction_factor, "friction_factor", np.float64, (ny, nx))
    active = (np.ones((ny, nx), dtype=np.bool_) if active_mask is None
              else _host_grid(active_mask, "active_mask", np.bool_, (ny, nx)))
    dx = _real(dx_m, "dx_m", RoutingGraphError)
    if not math.isfinite(dx) or dx <= 0.0:
        raise RoutingGraphError(f"dx_m must be finite and > 0, got {dx_m!r}")
    if dy_m is not None and _real(dy_m, "dy_m", RoutingGraphError) != dx:
        raise RoutingGraphError(f"non-square cells (dx {dx} m, dy {dy_m} m) are unsupported: the "
                                "legacy D4 slope and unit-width face flux assume square cells")
    if not (math.isfinite(dx * dx) and dx * dx > 0.0 and math.isfinite(dx * 1000.0)):
        raise RoutingGraphError(f"cell area dx^2 overflows or underflows FP64 for dx_m = {dx!r}; "
                                "face volumes could not be finite")
    if not np.all(np.isfinite(z)):
        raise RoutingGraphError("elevation_full_m must be finite everywhere, ring included "
                                "(legacy nodata elevations are unsupported)")
    nodata_cells = np.zeros(z.shape, dtype=np.bool_)
    if allow_masked_nodata and nodata_value is None:
        raise RoutingGraphError("allow_masked_nodata needs the nodata_value that marks the masked cells")
    if nodata_value is not None:
        sentinel = _real(nodata_value, "nodata_value", RoutingGraphError)
        if np.any(z == sentinel):
            if not allow_masked_nodata:
                raise RoutingGraphError(
                    f"elevation_full_m holds the nodata value {sentinel!r} in {int(np.sum(z == sentinel))} cell(s); "
                    "nodata elevations are unsupported (a finite sentinel would act as a false low receiver)"
                )
            nodata_cells = z == sentinel
            if np.any(nodata_cells[1:-1, 1:-1] & active):
                raise RoutingGraphError("an ACTIVE cell holds the nodata elevation: " + _cells(nodata_cells[1:-1, 1:-1] & active))
    if not active.any():
        raise RoutingGraphError("active_mask has no active cell")
    if not np.all(np.isfinite(ff[active])) or np.any(ff[active] < LEGACY_FRICTION_FLOOR):
        raise RoutingGraphError(
            f"friction_factor must be finite and >= {LEGACY_FRICTION_FLOOR} on every active cell: "
            "route_water.for 916-920 floors ff to 0.1 after the root, so smaller values are outside "
            "the supported legacy-comparison domain"
        )
    if np.any(export[1:-1, 1:-1] & active):
        raise RoutingGraphError("an active cell cannot be an export receiver: "
                                + _cells(export[1:-1, 1:-1] & active))

    # topog_attrib 94-117: strict '<' on elevation in mm, N E S W order.
    zmm = z * 1000.0
    centre = zmm[1:-1, 1:-1]
    zmin = centre.copy()
    aspect = np.zeros((ny, nx), dtype=np.int8)
    has_equal = np.zeros((ny, nx), dtype=np.bool_)
    for code in (1, 2, 3, 4):
        dr, dc = ASPECT_STEPS[code]
        neighbour = zmm[1 + dr: 1 + dr + ny, 1 + dc: 1 + dc + nx]
        # a masked-nodata neighbour is never eligible (topog_attrib.for 104-113); with no such cell this is all True
        eligible = ~nodata_cells[1 + dr: 1 + dr + ny, 1 + dc: 1 + dc + nx]
        better = (neighbour < zmin) & eligible
        aspect[better] = code
        zmin = np.where(better, neighbour, zmin)
        has_equal |= (neighbour == centre) & eligible
    slope = (centre - zmin) / (dx * 1000.0)  # 117, legacy mm / mm
    slope = np.where(active, slope, 0.0)  # 190-197
    slope = np.where(slope > _LEGACY_SLOPE_CAP, 1.0, slope)  # 218-224

    sinks = active & (aspect == 0)
    pit = np.zeros((ny, nx), dtype=np.bool_)
    if sinks.any():
        if allow_pit_storage and not (sinks & has_equal).any():
            pit = sinks.copy()
        else:
            raise RoutingGraphError(
                "unsupported sinks (no strictly lower D4 neighbour; no filling, carving or "
                + ("flat-sink storage" if allow_pit_storage else "pit storage")
                + f"): flats {_cells(sinks & has_equal) or 'none'}; strict pits {_cells(sinks & ~has_equal) or 'none'}"
            )
    aspect = np.where(active, aspect, 0).astype(np.int8)

    rows, cols = np.indices((ny, nx))
    step_r = np.array([0, 1, 0, -1, 0])[aspect]
    step_c = np.array([0, 0, 1, 0, -1])[aspect]
    rr, cc = rows + step_r, cols + step_c
    inside = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
    receiver_active = inside & active[np.clip(rr, 0, ny - 1), np.clip(cc, 0, nx - 1)]
    receiver_export = export[rr + 1, cc + 1]
    lost = active & ~receiver_active & ~receiver_export
    if lost.any():
        raise RoutingGraphError(
            "receiver is neither an active cell nor an export-flagged boundary (legacy would lose this "
            "outflow without reporting it): " + _cells(lost)
        )
    receiver = np.full((ny, nx), INACTIVE, dtype=np.int64)
    moving = active & receiver_active & ~pit  # a pit "receives itself" in the step arithmetic: it is terminal
    receiver[moving] = (rr * nx + cc)[moving]
    outlet = active & ~receiver_active
    receiver[outlet] = EXPORT
    receiver[pit] = PIT_STORAGE

    # 265-293, in the legacy loop order (i = 2..nr north to south, j ascending),
    # in place: a cell whose receiver is masked copies the opposite neighbour's
    # slope if its own is larger or zero.
    slope_before = slope.copy()
    applied = np.zeros((ny, nx), dtype=np.bool_)
    for r, c in sorted(zip(*np.nonzero(outlet), strict=True), key=lambda rc: (-rc[0], rc[1])):
        orr, occ = r - step_r[r, c], c - step_c[r, c]
        if not (0 <= orr < ny and 0 <= occ < nx) or not active[orr, occ]:
            raise RoutingGraphError(
                f"edge rule at (r={r}, c={c}) would read the slope of a ring or inactive cell "
                "(legacy slope 0, leaving an outlet that never drains); unsupported"
            )
        if slope[r, c] > slope[orr, occ] or slope[r, c] == 0.0:
            slope[r, c] = slope[orr, occ]
            applied[r, c] = True
    flat = active & ~(slope > 0.0) & ~pit
    if flat.any():
        raise RoutingGraphError("zero slope on an active cell (no conveyance): " + _cells(flat))

    with np.errstate(divide="ignore", invalid="ignore"):
        conveyance = np.where(active, np.sqrt(8.0 * GRAVITY_M_S2 * slope / np.where(active, ff, 1.0)), 0.0)
    if not np.all(np.isfinite(conveyance)):
        raise RoutingGraphError("internal error: non-finite conveyance")

    # Donors, in DONOR_SLOTS (legacy sdirin) order.
    n = ny * nx
    receiver_flat = receiver.reshape(-1)
    donor_flat = np.full((4, n), -1, dtype=np.int64)
    for s, (dr, dc) in enumerate(DONOR_SLOTS):
        sr, sc = rows + dr, cols + dc
        ok = (sr >= 0) & (sr < ny) & (sc >= 0) & (sc < nx)
        src = np.clip(sr, 0, ny - 1) * nx + np.clip(sc, 0, nx - 1)
        is_donor = active & ok & active.reshape(-1)[src] & (receiver_flat[src] == rows * nx + cols)
        donor_flat[s] = np.where(is_donor, src, -1).reshape(-1)

    # Dependency levels: elevation strictly decreases along every edge, so a
    # descending-elevation sweep sees all donors before their receiver.
    active_idx = np.flatnonzero(active)
    level = np.full(n, -1, dtype=np.int32)
    level[active_idx] = 0
    for cell in active_idx[np.argsort(-centre.reshape(-1)[active_idx], kind="stable")]:
        target = receiver_flat[cell]
        if target >= 0 and level[target] <= level[cell]:
            level[target] = level[cell] + 1
    internal = active_idx[receiver_flat[active_idx] >= 0]
    if np.any(level[receiver_flat[internal]] <= level[internal]):
        raise RoutingGraphError("internal error: dependency levels are not topological")
    order = active_idx[np.lexsort((active_idx, level[active_idx]))]
    counts = np.bincount(level[order])
    bounds = tuple(int(b) for b in np.concatenate(([0], np.cumsum(counts))))
    position = np.full(n, -1, dtype=np.int64)
    position[order] = np.arange(order.size)
    donor_lo = donor_flat[:, order]
    donor_mask = donor_lo >= 0
    donor_position = np.where(donor_mask, position[np.clip(donor_lo, 0, None)],
                              np.arange(order.size)[None, :])

    digest = hashlib.sha256(_GRAPH_SCHEMA)
    for array in (z, export, ff, active):
        digest.update(f"{array.dtype}{array.shape}".encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(repr(dx).encode())
    digest.update(repr(nodata_value).encode())
    policy = "strict"
    if allow_masked_nodata or allow_pit_storage:  # the strict default keeps its historical digest
        policy = f"masked_nodata={bool(allow_masked_nodata)};pit_storage={bool(allow_pit_storage)}"
        digest.update(f"policy:{policy}".encode())

    namespace = np if xp is None else xp

    def host(a):
        return freeze(np.ascontiguousarray(a))

    def runtime(a):
        return freeze(to_device(np.ascontiguousarray(a), namespace))

    return RoutingGraph(
        shape=(ny, nx),
        dx_m=dx,
        aspect=host(aspect),
        slope=host(slope),
        slope_before_edge_rule=host(slope_before),
        edge_rule_applied=host(applied),
        friction_factor=host(ff),
        active=host(active),
        outlet=host(outlet),
        receiver=host(receiver),
        level=host(level.reshape(ny, nx)),
        level_order_host=host(order),
        level_bounds=bounds,
        conveyance=runtime(conveyance.reshape(-1)),
        active_flat=runtime(active.reshape(-1)),
        outlet_flat=runtime(outlet.reshape(-1)),
        level_order=runtime(order),
        conveyance_lo=runtime(conveyance.reshape(-1)[order]),
        donor_position=runtime(donor_position),
        donor_mask=runtime(donor_mask),
        input_sha256=digest.hexdigest(),
        xp=namespace,
        pit_storage=host(pit),
        policy=policy,
    )


def plot1_routing_graph(fields: dict[str, Any], report: dict[str, Any], *, xp: ModuleType | None = None) -> RoutingGraph:
    """Graph for the imported Plot 1 case from its verified sidecar `fields`
    and import `report` (Phase 2). Uses the full 62 x 22 legacy DEM and
    rainfall-scaling map already stored south-first (no flip): export
    receivers are `rmask < 0` cells, active cells the interior `rmask >= 0`.
    Friction is the legacy type-1 per-surface-type mean (storm_setting
    563-568, deterministic whatever the distribution setting). The header's
    nodata value is passed so a sentinel elevation would be rejected.
    Requires flow_direction 4, method 5, friction type 1, no friction map,
    and a D4 aspect identical to the Phase 2 audit."""
    settings = report["legacy_options"]["settings"]
    required = {"flow_direction": 4, "flow_routing_solution_method": 5, "friction_factor_type": 1}
    for key, want in required.items():
        if settings.get(key) != want:
            raise RoutingGraphError(f"Plot 1 graph supports {key} = {want}, got {settings.get(key)!r}")
    if settings["use_flags"].get("use_friction_factor_map"):
        raise RoutingGraphError("a friction factor map is not supported")
    full_z = np.asarray(fields["legacy_full_elevation_m"], dtype=np.float64)
    full_rm = np.asarray(fields["legacy_full_rainfall_scaling"], dtype=np.float64)
    interior = np.asarray(fields["elevation_source_m"], dtype=np.float64)
    if full_z.shape != (interior.shape[0] + 2, interior.shape[1] + 2) or not np.array_equal(full_z[1:-1, 1:-1], interior):
        raise RoutingGraphError("legacy_full_elevation_m interior differs from elevation_source_m "
                                "(orientation or crop mismatch)")
    types = np.asarray(fields["surface_type_resolved"])
    means = settings["by_surface_type"]["friction_factor_mean"]
    ff = np.empty(interior.shape, dtype=np.float64)
    for t in np.unique(types):
        key = f"type_{int(t)}"
        if key not in means:
            raise RoutingGraphError(f"no friction_factor_mean for surface {key}")
        ff[types == t] = float(means[key])
    dx = float(report["grid"]["cellsize_m"])
    nodata = float(report["grid"]["header"]["nodata_value"])
    graph = build_routing_graph(full_z, full_rm < 0.0, ff, dx, active_mask=full_rm[1:-1, 1:-1] >= 0.0,
                                nodata_value=nodata, xp=xp)
    if not np.array_equal(graph.aspect, np.asarray(fields["legacy_d4_aspect"])):
        raise RoutingGraphError("graph aspect differs from the Phase 2 legacy_d4_aspect audit")
    return graph


# --- step --------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class RouteStep:
    """Result of one routing step. Grids are `(ny, nx)` in the graph's
    namespace; scalars are 0-d arrays there (read them with
    `maple.core.backend.to_float`, one counted read each).

    depth_m          storage depth after the step (inactive cells unchanged)
    flow_depth_m     bisection depth h_flow that `discharge` and `velocity`
                     belong to; |depth - flow_depth| <= root_tolerance_m
    discharge_m2_s   q_new = k h_flow^{3/2} (legacy q(2)); velocity k sqrt(h_flow)
    inflow_m2_s      Qin_new (legacy qin(2)); old_inflow_m2_s the Qin_old used
    face_volume_m3   dx^2 c (q_old + q_new) leaving each active cell
    export_m3        face volume through outlets; outlet_discharge_m3_s is the
                     instantaneous sum of q_new dx over outlets (legacy q_plot dx)
    budget_residual_m3  storage change + export (0 up to rounding when
                     `conservative`; the created volume in the literal mode)
    stale_inflow_gain_m3  literal mode only: dx^2 c sum(stale - coherent Qin_old)
    implementation   "array", "numba" or "cuda" (which ordered sweep produced it)
    bisection_iterations  the fixed halving count of the bisection solver; 0 for the Newton solver (no bisection
                     count is configured there: see `newton_max_iterations`)
    root_solver      "bisection" (default) or "newton"; `newton_max_iterations` the Newton pass cap (0 for bisection)
    root_stats       None for bisection; for Newton the `routing_newton.STAT_NAMES` counters of the sweep
                     (host ints: max/total Newton passes, bisection safeguard steps, fallback cells, iterated cells)
    """

    dt_s: float
    depth_m: Any
    flow_depth_m: Any
    discharge_m2_s: Any
    velocity_m_s: Any
    inflow_m2_s: Any
    old_discharge_m2_s: Any
    old_inflow_m2_s: Any
    face_volume_m3: Any
    export_m3: Any
    outlet_discharge_m3_s: Any
    storage_change_m3: Any
    budget_residual_m3: Any
    stale_inflow_gain_m3: Any
    max_courant_old: Any
    max_courant_new: Any
    max_constitutive_residual_m: Any
    max_cell_balance_residual_m: Any
    conservative: bool
    bisection_iterations: int
    implementation: str
    root_solver: str = "bisection"
    newton_max_iterations: int = 0
    root_stats: dict[str, int] | None = None


def _require_arrays(named: dict[str, Any], shape: tuple[int, int], xp: ModuleType) -> None:
    from maple.core.backend import MixedArrayNamespaceError, array_namespace, is_array

    for name, array in named.items():
        if not is_array(array):
            raise RoutingError(f"{name} must be a NumPy/CuPy array, got {type(array).__name__}")
    try:
        namespace = array_namespace(*named.values())
    except MixedArrayNamespaceError as exc:
        raise RoutingError(str(exc)) from None
    if namespace is not xp:
        raise RoutingError(f"arrays must be in the graph's namespace {xp.__name__!r}, got {namespace.__name__!r}")
    for name, array in named.items():
        if tuple(array.shape) != shape:
            raise RoutingError(f"{name} shape {tuple(array.shape)} != {shape}")
        if array.dtype != np.float64:
            raise RoutingError(f"{name} must be float64, got {array.dtype}")


def _donor_sum(values_lo, position, mask, xp):
    """Sum of donor values in DONOR_SLOTS order: 0 + a + b + c + d with
    non-donors contributing an exact 0.0, i.e. the legacy sequential sum."""
    gathered = values_lo[position]
    total = xp.where(mask[0], gathered[0], 0.0)
    for s in (1, 2, 3):
        total = total + xp.where(mask[s], gathered[s], 0.0)
    return total


def _bisect(lo, rhs, k, c, iterations, w, mid, t, below, xp):
    """Largest h in [0, rhs] (to 2^-iterations rhs) with h + c q(h) < rhs,
    written into `lo`. `h + c k h^{3/2}` is increasing and equals rhs at the
    root, so [0, rhs] always brackets it. The flux term is formed exactly as
    in the closure `h_new = rhs - c q(lo)`, so h_new > lo >= 0 follows from
    the accepted comparison, with no clipping. routing_numba._sweep mirrors
    this operation sequence statement for statement."""
    lo.fill(0.0)
    xp.copyto(w, rhs)
    for _ in range(iterations):
        xp.multiply(w, 0.5, out=w)
        xp.add(lo, w, out=mid)
        xp.sqrt(mid, out=t)
        xp.multiply(t, mid, out=t)
        xp.multiply(t, k, out=t)
        xp.multiply(t, c, out=t)
        xp.add(t, mid, out=t)
        xp.less(t, rhs, out=below)
        xp.copyto(lo, mid, where=below)


def _sweep_array(graph, base_lo, c, iterations, xp):
    """Level-ordered sweep in the graph namespace: a Python loop over levels
    (never cells), gathers for donors, slice writes for results."""
    n_active = graph.n_active
    k_lo = graph.conveyance_lo
    qin_new_lo = xp.zeros(n_active, dtype=np.float64)
    q_new_lo = xp.zeros(n_active, dtype=np.float64)
    flow_lo = xp.zeros(n_active, dtype=np.float64)
    rhs_lo = xp.empty(n_active, dtype=np.float64)
    width = graph.max_level_width
    w_buf, mid_buf, t_buf = (xp.empty(width, dtype=np.float64) for _ in range(3))
    below_buf = xp.empty(width, dtype=np.bool_)
    for lev, (b0, b1) in enumerate(itertools.pairwise(graph.level_bounds)):
        sl, m = slice(b0, b1), b1 - b0
        if lev:
            qin_new_lo[sl] = _donor_sum(q_new_lo, graph.donor_position[:, sl], graph.donor_mask[:, sl], xp)
        rhs = rhs_lo[sl]
        xp.multiply(qin_new_lo[sl], c, out=rhs)
        xp.add(base_lo[sl], rhs, out=rhs)
        lo, k_l = flow_lo[sl], k_lo[sl]
        _bisect(lo, rhs, k_l, c, iterations, w_buf[:m], mid_buf[:m], t_buf[:m], below_buf[:m], xp)
        q = q_new_lo[sl]
        xp.sqrt(lo, out=q)
        xp.multiply(q, lo, out=q)
        xp.multiply(q, k_l, out=q)
    return qin_new_lo, q_new_lo, flow_lo, rhs_lo


def _sweep_array_newton(graph, base_lo, c, max_iter):
    """`_sweep_array` with the Newton root, vectorized per level (NumPy graph only): returns the same four
    level-ordered arrays plus the Newton statistics."""
    n_active = graph.n_active
    k_lo = graph.conveyance_lo
    qin_new_lo = np.zeros(n_active, dtype=np.float64)
    q_new_lo = np.zeros(n_active, dtype=np.float64)
    flow_lo = np.zeros(n_active, dtype=np.float64)
    rhs_lo = np.empty(n_active, dtype=np.float64)
    top = total = steps = fallbacks = iterated = 0
    for lev, (b0, b1) in enumerate(itertools.pairwise(graph.level_bounds)):
        sl = slice(b0, b1)
        if lev:
            qin_new_lo[sl] = _donor_sum(q_new_lo, graph.donor_position[:, sl], graph.donor_mask[:, sl], np)
        rhs = rhs_lo[sl]
        np.multiply(qin_new_lo[sl], c, out=rhs)
        np.add(base_lo[sl], rhs, out=rhs)
        flow, passes, bisect_steps, fallback = routing_newton.newton_root_level(rhs, k_lo[sl], c, max_iter)
        flow_lo[sl] = flow
        q = q_new_lo[sl]
        np.sqrt(flow, out=q)
        np.multiply(q, flow, out=q)
        np.multiply(q, k_lo[sl], out=q)
        top = max(top, int(passes.max()))
        total += int(passes.sum())
        steps += int(bisect_steps.sum())
        fallbacks += int(fallback.sum())
        iterated += int(np.count_nonzero(passes))
    stats = routing_newton.stats_dict((top, total, steps, fallbacks, iterated))
    return qin_new_lo, q_new_lo, flow_lo, rhs_lo, stats


def _check_root_solver(root_solver, newton_max_iterations, implementation, xp):
    """Validate the root-solver options before anything is computed. Newton is CPU NumPy only: CUDA (and any
    non-NumPy graph) is refused explicitly, never silently replaced."""
    if not isinstance(root_solver, str) or root_solver not in ROOT_SOLVERS:
        raise RoutingError(f"root_solver must be one of {ROOT_SOLVERS}, got {root_solver!r}")
    if (isinstance(newton_max_iterations, bool) or not isinstance(newton_max_iterations, (int, np.integer))
            or not (1 <= int(newton_max_iterations) <= MAX_NEWTON_ITERATIONS)):
        raise RoutingError(f"newton_max_iterations must be an int in [1, {MAX_NEWTON_ITERATIONS}], "
                           f"got {newton_max_iterations!r}")
    if root_solver == "newton":
        if implementation == "cuda":
            raise RoutingError("root_solver 'newton' is CPU-only: implementation 'cuda' supports the bisection "
                               "solver only (no GPU Newton, no fallback)")
        if xp is not np:
            raise RoutingError(f"root_solver 'newton' runs on host NumPy graphs only; the graph lives in "
                               f"{xp.__name__!r} and no host/device transfer is performed")
    return root_solver, int(newton_max_iterations)


def _check_step_options(dt_s, courant_max, iterations, root_tolerance_m, implementation):
    dt = _real(dt_s, "dt_s", RoutingError)
    if not math.isfinite(dt) or dt <= 0.0:
        raise RoutingError(f"dt_s must be finite and > 0 (dt = 0 is rejected, not an identity), got {dt_s!r}")
    cr_max = _real(courant_max, "courant_max", RoutingError)
    if not (0.0 < cr_max <= 2.0):
        raise RoutingError(f"courant_max must lie in (0, 2] (2 guarantees a non-negative right-hand side), "
                           f"got {courant_max!r}")
    root_tol = _real(root_tolerance_m, "root_tolerance_m", RoutingError)
    if not (math.isfinite(root_tol) and root_tol > 0.0):
        raise RoutingError(f"root_tolerance_m must be finite and > 0, got {root_tolerance_m!r}")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) \
            or not (1 <= int(iterations) <= _MAX_BISECTION_ITERATIONS):
        raise RoutingError(f"bisection_iterations must be an int in [1, {_MAX_BISECTION_ITERATIONS}], "
                           f"got {iterations!r}")
    if implementation not in ROUTE_IMPLEMENTATIONS:
        raise RoutingError(f"implementation must be one of {ROUTE_IMPLEMENTATIONS}, got {implementation!r}")
    return dt, cr_max, int(iterations), root_tol


def _route(graph, depth_start_m, old_flow_depth_m, dt_s, old_discharge_m2_s, stale_old_inflow_m2_s,
           courant_max, bisection_iterations, root_tolerance_m, implementation,
           root_solver="bisection", newton_max_iterations=DEFAULT_NEWTON_MAX_ITERATIONS) -> RouteStep:
    from maple.core.backend import (
        DeferredChecks,
        errstate,
        finite_flag,
        negative_flag,
        true_flag,
    )

    if not isinstance(graph, RoutingGraph):
        raise RoutingError("graph must be a RoutingGraph (use build_routing_graph)")
    dt, cr_max, iterations, root_tol = _check_step_options(dt_s, courant_max, bisection_iterations,
                                                           root_tolerance_m, implementation)
    solver, newton_cap = _check_root_solver(root_solver, newton_max_iterations, implementation,
                                            graph.xp)
    newton = solver == "newton"
    if implementation == "cuda" and stale_old_inflow_m2_s is not None:
        raise RoutingError(
            "implementation 'cuda' does not support the literal-legacy stale-inflow step "
            "(legacy_stale_inflow_step is a non-conservative comparison tool); use 'array' or 'numba'"
        )
    named = {"depth_start_m": depth_start_m, "old_flow_depth_m": old_flow_depth_m}
    if old_discharge_m2_s is not None:
        named["old_discharge_m2_s"] = old_discharge_m2_s
    if stale_old_inflow_m2_s is not None:
        named["stale_old_inflow_m2_s"] = stale_old_inflow_m2_s
    xp = graph.xp
    _require_arrays(named, graph.shape, xp)
    if implementation == "numba" and xp is not np:
        raise RoutingError(
            f"implementation 'numba' runs on host NumPy arrays only; the graph lives in {xp.__name__!r} "
            "and no host/device transfer is performed (build the graph with xp=numpy or use 'array')"
        )
    if implementation == "cuda":
        if xp is np:
            raise RoutingError(
                "implementation 'cuda' runs on CuPy graphs and arrays only; the graph lives in 'numpy' and no "
                "host/device transfer is performed (build the graph with xp=cupy or use 'numba'/'array')"
            )
        from maple_syrup import routing_cuda

        # Structure, current device and the owned static context are checked before any shared arithmetic.
        routing_cuda.require_inputs(graph, named)

    ny, nx = graph.shape
    n_active = graph.n_active
    dx = graph.dx_m
    area = dx * dx
    c = dt / (2.0 * dx)
    active, outlet, k, order = graph.active_flat, graph.outlet_flat, graph.conveyance, graph.level_order
    h_start = depth_start_m.reshape(-1)
    h_old = old_flow_depth_m.reshape(-1)

    checks = DeferredChecks()
    for name, array in named.items():
        checks.require(finite_flag(array), f"{name} must be finite everywhere")
        checks.forbid(negative_flag(array), f"{name} must be >= 0 everywhere")
    checks.forbid(true_flag(active & (h_old > h_start)),
                  "old_flow_depth_m (legacy post-infiltration d(1)) exceeds depth_start_m on an active cell")

    with errstate(xp=xp, all="ignore"):
        if old_discharge_m2_s is None:
            q_old = xp.where(active, (xp.sqrt(h_old) * h_old) * k, 0.0)
        else:
            q_old = old_discharge_m2_s.reshape(-1)
            checks.forbid(true_flag(~active & (q_old != 0.0)), "old_discharge_m2_s must be 0 on inactive cells")
            # Depth implied by q_old through q = k h^{3/2} must match old_flow_depth_m.
            implied = (q_old / xp.where(active, k, 1.0)) ** (2.0 / 3.0)
            mismatch = xp.abs(implied - h_old) > root_tol + 64.0 * _EPS * h_old
            checks.forbid(true_flag(active & mismatch),
                          "old_discharge_m2_s does not satisfy q = k h^{3/2} at old_flow_depth_m "
                          "within root_tolerance_m")
        if stale_old_inflow_m2_s is not None:
            checks.forbid(true_flag(~active & (stale_old_inflow_m2_s.reshape(-1) != 0.0)),
                          "stale_old_inflow_m2_s must be 0 on inactive cells")

        courant_old = xp.where(h_old > 0.0, q_old / xp.where(h_old > 0.0, h_old, 1.0),
                               xp.where(q_old > 0.0, xp.inf, 0.0)) * (dt / dx)
        courant_old = xp.where(active, courant_old, 0.0)
        checks.require(finite_flag(q_old), "old discharge q_old = k h_old^{3/2} overflowed FP64")
        checks.forbid(true_flag(courant_old > cr_max),
                      f"{_COURANT_REJECTION} {cr_max}; step rejected (retry with a smaller dt)")

        # Level-ordered sweep.
        q_old_lo = q_old[order]
        coherent_lo = _donor_sum(q_old_lo, graph.donor_position, graph.donor_mask, xp)
        qin_old_lo = coherent_lo if stale_old_inflow_m2_s is None else stale_old_inflow_m2_s.reshape(-1)[order]
        base_lo = h_start[order] + c * (qin_old_lo - q_old_lo)
        root_stats = None
        if newton:
            if implementation == "numba":
                qin_new_lo, q_new_lo, flow_lo, rhs_lo, root_stats = routing_newton.run_sweep(
                    graph, base_lo, c, newton_cap)
            else:
                qin_new_lo, q_new_lo, flow_lo, rhs_lo, root_stats = _sweep_array_newton(
                    graph, base_lo, c, newton_cap)
            root_stats = routing_newton.stats_dict(root_stats) if not isinstance(root_stats, dict) else root_stats
        elif implementation == "cuda":
            qin_new_lo, q_new_lo, flow_lo, rhs_lo = routing_cuda.run_sweep(graph, base_lo, c, iterations)
        elif implementation == "numba":
            from maple_syrup import routing_numba

            qin_new_lo, q_new_lo, flow_lo, rhs_lo = routing_numba.run_sweep(graph, base_lo, c, iterations)
        else:
            qin_new_lo, q_new_lo, flow_lo, rhs_lo = _sweep_array(graph, base_lo, c, iterations, xp)
        h_new_lo = rhs_lo - q_new_lo * c
        checks.require(finite_flag(rhs_lo), "right-hand side overflowed FP64 (inflow or storage too large)")
        checks.forbid(true_flag(rhs_lo < 0.0),
                      f"{_NEGATIVE_RHS_REJECTION} (legacy STOP condition); step rejected")

        h_new = xp.array(h_start, copy=True)
        h_new[order] = h_new_lo
        q_new = xp.zeros(ny * nx, dtype=np.float64)
        q_new[order] = q_new_lo
        flow = xp.zeros(ny * nx, dtype=np.float64)
        flow[order] = flow_lo
        qin_new = xp.zeros(ny * nx, dtype=np.float64)
        qin_new[order] = qin_new_lo
        qin_old = xp.zeros(ny * nx, dtype=np.float64)
        qin_old[order] = qin_old_lo
        velocity = xp.where(active, xp.sqrt(flow) * k, 0.0)
        face = xp.where(active, area * (c * (q_old + q_new)), 0.0)

        balance = (h_new - h_start) - c * ((qin_old + qin_new) - (q_old + q_new))
        scale = h_start + h_new + c * (qin_old + qin_new + q_old + q_new)
        balance = xp.where(active, balance, 0.0)
        constitutive = xp.where(active, h_new - flow, 0.0)
        storage_change = area * xp.sum(h_new - h_start)
        export = xp.sum(xp.where(outlet, face, 0.0))
        residual = storage_change + export
        global_tol = BALANCE_RTOL * (n_active + 2) * area * xp.sum(xp.where(active, scale, 0.0))
        outlet_discharge = dx * xp.sum(xp.where(outlet, q_new, 0.0))
        stale_gain = None if stale_old_inflow_m2_s is None else area * c * xp.sum(qin_old_lo - coherent_lo)

    # Finite before any tolerance comparison: a NaN would otherwise compare
    # False against every tolerance and pass silently.
    for name, array in (("depth", h_new), ("discharge", q_new), ("flow depth", flow),
                        ("velocity", velocity), ("inflow", qin_new), ("old inflow", qin_old),
                        ("face volume", face), ("balance scale", scale)):
        checks.require(finite_flag(array), f"step produced non-finite {name}")
    for name, array in (("depth", h_new), ("discharge", q_new), ("flow depth", flow)):
        checks.forbid(negative_flag(array), f"step produced negative {name}")
    scalars = [("storage change", storage_change), ("export", export), ("budget residual", residual),
               ("balance tolerance", global_tol), ("outlet discharge", outlet_discharge)]
    if stale_gain is not None:
        scalars.append(("stale inflow gain", stale_gain))
    for name, value in scalars:
        checks.require(finite_flag(xp.asarray(value)), f"step produced non-finite {name} (volume overflow)")
    if newton:
        checks.forbid(true_flag(xp.abs(constitutive) > root_tol),
                      f"Newton root solver did not reach root_tolerance_m = {root_tol} m "
                      f"(newton_max_iterations = {newton_cap}); nothing is clipped")
    else:
        checks.forbid(true_flag(xp.abs(constitutive) > root_tol),
                      f"bisection did not reach root_tolerance_m = {root_tol} m after {iterations} iterations "
                      "(increase bisection_iterations); nothing is clipped")
    checks.forbid(true_flag(xp.abs(balance) > BALANCE_RTOL * scale),
                  "per-cell water balance violated beyond FP64 tolerance")
    if stale_old_inflow_m2_s is None:
        checks.forbid(true_flag(xp.abs(residual) > global_tol),
                      "global water balance (storage change + export) violated beyond FP64 tolerance")
    try:
        checks.resolve()
    except ValueError as exc:
        message = str(exc)
        recoverable = message.startswith((_COURANT_REJECTION, _NEGATIVE_RHS_REJECTION))
        raise (RoutingStepRejected if recoverable else RoutingError)(message) from None

    grid = (ny, nx)
    return RouteStep(
        dt_s=dt,
        depth_m=h_new.reshape(grid),
        flow_depth_m=flow.reshape(grid),
        discharge_m2_s=q_new.reshape(grid),
        velocity_m_s=velocity.reshape(grid),
        inflow_m2_s=qin_new.reshape(grid),
        old_discharge_m2_s=xp.array(q_old, copy=True).reshape(grid),
        old_inflow_m2_s=qin_old.reshape(grid),
        face_volume_m3=face.reshape(grid),
        export_m3=export,
        outlet_discharge_m3_s=outlet_discharge,
        storage_change_m3=storage_change,
        budget_residual_m3=residual,
        stale_inflow_gain_m3=stale_gain,
        max_courant_old=xp.max(courant_old),
        max_courant_new=xp.max(velocity) * (dt / dx),
        max_constitutive_residual_m=xp.max(xp.abs(constitutive)),
        max_cell_balance_residual_m=xp.max(xp.abs(balance)),
        conservative=stale_old_inflow_m2_s is None,
        bisection_iterations=0 if newton else iterations,
        implementation=implementation,
        root_solver=solver,
        newton_max_iterations=newton_cap if newton else 0,
        root_stats=root_stats,
    )


def route_step(
    graph: RoutingGraph,
    depth_start_m: Any,
    old_flow_depth_m: Any,
    dt_s: float,
    *,
    old_discharge_m2_s: Any = None,
    courant_max: float = DEFAULT_COURANT_MAX,
    bisection_iterations: int = DEFAULT_BISECTION_ITERATIONS,
    root_tolerance_m: float = DEFAULT_ROOT_TOLERANCE_M,
    implementation: str = "array",
    root_solver: str = "bisection",
    newton_max_iterations: int = DEFAULT_NEWTON_MAX_ITERATIONS,
) -> RouteStep:
    """One conservative method-5 step (see the module docstring).

    `depth_start_m` storage after rain/infiltration (legacy d(1) + excess dt);
    `old_flow_depth_m` legacy post-infiltration d(1), <= depth_start_m;
    `old_discharge_m2_s` legacy q(1): default `k h_old^{3/2}`, otherwise
    checked against that relation within `root_tolerance_m` in depth. The
    receiver's old inflow is the donor sum of this same q_old.
    `implementation` selects the ordered sweep: "array" (default, graph
    namespace), "numba" (CPU NumPy only; raises `RoutingError` if Numba is
    missing or the graph is on a device -- no fallback) or "cuda" (CuPy graph
    only; one RawKernel launch per dependency level, see routing_cuda.py;
    raises on a NumPy graph, missing CuPy/device or a device mismatch -- no
    fallback or dynamic array conversion; first use prepares owned static data
    with counted transfers (see routing_cuda.py); `RouteStep.implementation` is "cuda").
    `root_solver` "bisection" (default; exactly the historical solver) or "newton" (routing_newton.py; CPU NumPy
    graph with "array" or "numba"; "cuda" is refused; `newton_max_iterations` caps the Newton passes per cell and
    `bisection_iterations` is then validated but unused). One batched flag
    read; any failure raises `RoutingError` and returns nothing."""
    return _route(graph, depth_start_m, old_flow_depth_m, dt_s, old_discharge_m2_s, None,
                  courant_max, bisection_iterations, root_tolerance_m, implementation,
                  root_solver, newton_max_iterations)


def legacy_stale_inflow_step(
    graph: RoutingGraph,
    depth_start_m: Any,
    old_flow_depth_m: Any,
    dt_s: float,
    *,
    stale_old_inflow_m2_s: Any,
    old_discharge_m2_s: Any = None,
    courant_max: float = DEFAULT_COURANT_MAX,
    bisection_iterations: int = DEFAULT_BISECTION_ITERATIONS,
    root_tolerance_m: float = DEFAULT_ROOT_TOLERANCE_M,
    implementation: str = "array",
) -> RouteStep:
    """LITERAL-LEGACY COMPARISON ONLY -- NOT CONSERVATIVE. Identical to
    `route_step` except that each receiver's old inflow is the supplied
    `stale_old_inflow_m2_s` (the legacy `qin(1)`, rolled over by
    update_water_flow and not recomputed after infilt changes q(1)) instead
    of the donor sum of q_old. The global balance is reported, not enforced:
    `budget_residual_m3` is the water the literal update creates (or
    destroys) and `stale_inflow_gain_m3` its expected value. Never use for
    production storage."""
    return _route(graph, depth_start_m, old_flow_depth_m, dt_s, old_discharge_m2_s, stale_old_inflow_m2_s,
                  courant_max, bisection_iterations, root_tolerance_m, implementation)
