"""Direct rain-splash redistribution core (standalone; no event integration).

Reference: MAHLERAN 1.2.3 `Subroutines_Sediment/splash_transport.for`
(read-only), line numbers as read 2026-10-17:

- 47-51: local deposit `1 - exp(-spd dx)` of the detached mass in its own
  cell (`depos_fract`).
- 52-56: ray weights, each applied to the FULL detached mass: cross-slope
  `0.25`, down `0.5 (1 - 0.5 exp(-2.42 S))`, up `0.5 - down`.
- 66-103: flat (`S = 0`) all four `0.25`; else aspect 2/3/4/else(1) = E/S/W/N
  gives the down ray, `splash_mv` slots 1..4 = right (east), down (legacy
  row i + 1 = MAPLE south), left (west), up (legacy i - 1 = MAPLE north).
- 119-239: each ray's cell `n = 1, 2, ...` (distance `n dx` .. `(n+1) dx`)
  receives `w (exp(-n spd dx) - exp(-(n+1) spd dx))`.

The nmax cut (`nmax_splash`, 104/121), the `10e-50 dt` vanishing-tail stop
(116, 123) and the Fortran's edge-cell extra deposits (247-267, which add
mass) are NOT reproduced. Original Fortran was NOT executed by this module's
author; comparison with an execution of the original interior is a separate
root task.

Exact law implemented, per source cell with pickup mass `P` (kg), class
decay `spd` (1/m), `a = spd dx`, `g = exp(-a)`:

    local deposit            P (1 - g)
    ray k, cell n=1..m_k     P w_k g^n (1 - g)       (m_k = reachable cells)
    ray k tail beyond m_k    P w_k g^(m_k + 1)       (never truncated)

The tail is the exact remaining directional mass. What happens to it depends
only on what ends the ray:

- OUTER FACE that is open (per-face `open_x_faces`/`open_y_faces`, default ALL CLOSED; the graph adapter opens
  exactly the faces of `graph.outlet` cells in their aspect direction): the tail is
  EXPORTED across that boundary face (reported per edge cell, per source
  cell, and as the face crossing);
- grid edge declared closed, an inactive cell (`active == False`: nodata
  mask, wall, closed obstruction): the ray STOPS at the last active cell and
  the tail is DEPOSITED THERE. When the first neighbour is blocked that last
  active cell is the source itself, i.e. the tail returns to the source.
  Rationale: droplets thrown against a wall or a masked hole come to rest at
  its foot; nothing jumps a masked hole (a ray ends at the first inactive
  cell, cells beyond it are never reached), nothing is reflected, and no
  mass is lost, so closure is exact without bookkeeping outside the domain.

No array is sized by travel distance. The ray mass is carried by the exact
flux recurrence along a line, `F_out = g (F_in + P w)`, deposit `F_in (1 - g)`
per active cell, whose closed form is the bins above; a `spd -> 0` or an
enormous domain needs no more memory than a few `(ny, nx, nc)` arrays, and
the Python loops run over the domain extent (`2 nx + 2 ny` vectorized steps
across all lines and classes), never over cells or over `1 / spd`. Every
step is a NumPy/CuPy elementwise operation in the grid's namespace
(`maple.core.backend`); nothing is transferred to the host inside the
sweeps (a single batched flag read validates the result, as in
`sediment_transport`). A fused compiled kernel is possible later; none is
claimed here.

Units and shapes (MAPLE conventions, row 0 = south, +x east, +y north):

- `pickup_kg` `(ny, nx, nc)` FP64 kg ACTUAL pickup; >= 0, zero on inactive
  cells. Demand is never treated as removed inventory.
- `spd` `(nc,)` 1/m, finite, >= 0; REQUIRED, no calibrated default exists
  in code (`splash_parameters`).
- `slope` dimensionless m/m, `aspect` legacy codes 1 N, 2 E, 3 S, 4 W (the
  MAPLE routing codes), `dx_m` m.
- Outputs: `deposition_kg`, `export_edge_cell_kg`, `export_source_cell_kg`
  `(ny, nx, nc)`; faces x `(ny, nx + 1, nc)` (index c = face WEST of
  column c), y `(ny + 1, nx, nc)` (index r = face SOUTH of row r); gross
  >= 0, net signed (+ east / north); export crossings are included in the
  face arrays exactly as in `sediment_transport`.

Closure (returned and enforced): per class `sum pickup = sum deposition +
sum export_edge`; per cell and class `deposition - pickup = net inflow
through the four faces` (the export face counts as outflow). The
source-attributed export is an independent closed form and is reported with
its own residual.

Instantaneous redistribution only: mass is retained only in final deposits
and export; no airborne cohort, no water advection, no wind.

Nothing here writes to a caller-owned array; validation precedes any output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

__all__ = [
    "ASPECT_EAST",
    "ASPECT_NORTH",
    "ASPECT_SOUTH",
    "ASPECT_WEST",
    "CROSS_SLOPE_WEIGHT",
    "DOWNSLOPE_DECAY_PER_SLOPE",
    "IMPLEMENTATIONS",
    "RAY_ORDER",
    "SplashError",
    "SplashExchange",
    "SplashExchangeAudit",
    "SplashGrid",
    "SplashParameters",
    "SplashResult",
    "apply_splash_exchange",
    "ray_weights",
    "resolve_implementation",
    "splash_grid",
    "splash_grid_from_graph",
    "splash_parameters",
    "splash_redistribute",
]

# routing.ASPECT_STEPS codes (legacy topog_attrib order N, E, S, W).
ASPECT_NORTH, ASPECT_EAST, ASPECT_SOUTH, ASPECT_WEST = 1, 2, 3, 4
# splash_transport.for 52-56 (the 2.42 replaced the 242 of the superseded JWAug2005 block, 41-43).
CROSS_SLOPE_WEIGHT = 0.25
DOWNSLOPE_DECAY_PER_SLOPE = 2.42
# Ray slots = splash_mv(1..4) = right/east, down/SOUTH, left/west, up/NORTH (south-first as the legacy
# row i + 1 is the MAPLE south); the opposite ray of slot k is (k + 2) % 4.
RAY_ORDER = ("east", "south", "west", "north")
_ASPECT_TO_SLOT = {ASPECT_EAST: 0, ASPECT_SOUTH: 1, ASPECT_WEST: 2, ASPECT_NORTH: 3}
_EPS = float(np.finfo(np.float64).eps)
# Declared operation counts behind the enforced closure bounds (a ray recurrence is algebraically consistent;
# only rounding of a few products per cell and ray differs between the deposit and flux forms).
_CLOSURE_OPS = 256


class SplashError(ValueError):
    """Invalid parameters, grid or inputs, or a failed closure. Raised before any result is returned; no
    caller-owned array is modified."""


# --- parameters ----------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SplashParameters:
    """Validated class splash decay `inverse_distance_per_m` (host `(nc,)`, 1/m) and its `(nc,)` copy in
    namespace `xp`. Build with `splash_parameters`."""

    n_classes: int
    inverse_distance_per_m: np.ndarray
    xp: ModuleType
    d_inverse_distance: Any

    def summary(self) -> dict[str, Any]:
        return {"n_classes": self.n_classes, "inverse_distance_per_m": self.inverse_distance_per_m.tolist(),
                "namespace": self.xp.__name__}


def splash_parameters(inverse_distance_per_m: Any, *, xp: ModuleType | None = None) -> SplashParameters:
    """Explicit per-class `spd` (1/m). Finite and >= 0 (0 = no local deposit, the whole mass travels to the
    boundary or its obstruction). No default: the calibrated MAHLERAN values are not available in code."""
    from maple.core.backend import freeze, to_device

    raw = np.asarray(inverse_distance_per_m)
    if raw.dtype.kind not in "fiu" or raw.ndim != 1 or raw.size < 1:
        raise SplashError("inverse_distance_per_m must be a real 1-D per-class sequence with >= 1 class, got "
                          f"dtype {raw.dtype} shape {raw.shape}")
    spd = np.array(raw, dtype=np.float64)
    if not np.all(np.isfinite(spd)) or np.any(spd < 0.0):
        raise SplashError("inverse_distance_per_m must be finite and >= 0 for every class")
    namespace = np if xp is None else xp
    return SplashParameters(n_classes=int(spd.size), inverse_distance_per_m=freeze(np.ascontiguousarray(spd)),
                            xp=namespace,
                            d_inverse_distance=freeze(to_device(np.ascontiguousarray(spd), namespace)))


# --- weights ---------------------------------------------------------------------------
def ray_weights(slope: Any, aspect: Any) -> np.ndarray:
    """Host reference of the four ray weights `(4, ...)` in `RAY_ORDER` (east, south, west, north) for
    `slope` (>= 0) and legacy `aspect` codes: flat -> 0.25 each; else down ray (the aspect direction)
    `0.5 (1 - 0.5 exp(-2.42 S))`, opposite ray `0.5 - down`, the two cross rays 0.25. Sums to 1."""
    s = np.asarray(slope, dtype=np.float64)
    asp = np.asarray(aspect)
    slot = np.full(asp.shape, -1, dtype=np.int64)
    for code, index in _ASPECT_TO_SLOT.items():
        slot = np.where(asp == code, index, slot)
    down = 0.5 * (1.0 - 0.5 * np.exp(-DOWNSLOPE_DECAY_PER_SLOPE * s))
    up = 0.5 - down
    sloped = (s != 0.0) & (slot >= 0)
    out = np.empty((4,) + s.shape, dtype=np.float64)
    for k in range(4):
        out[k] = np.where(sloped, np.where(slot == k, down, np.where((slot + 2) % 4 == k, up, CROSS_SLOPE_WEIGHT)),
                          CROSS_SLOPE_WEIGHT)
    return out


# --- grid ------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SplashGrid:
    """Static terrain the splash law needs, placed once in namespace `xp`: `weights` `(4, ny, nx)` (0 on
    inactive cells), `active` `(ny, nx)` bool, and per ray slot the static geometry `ray_continues`
    (the next cell is active, or the grid edge is open), `edge_cell` (no neighbour that way), `reach_cells`
    (active cells before the ray ends, int64) and `reach_open` (the ray ends at an OPEN face).

    Exterior openness is per FACE in MAPLE's layout: `open_x_faces` `(ny, nx + 1)` (index c = face WEST of
    column c) and `open_y_faces` `(ny + 1, nx)` (index r = face SOUTH of row r), host read-only bools plus
    their device copies. Only OUTER faces of ACTIVE edge cells can be open (validated); every interior face and
    every face of an inactive edge cell is closed. A ray reaching a closed outer face is an obstruction (tail
    deposited at the last active cell). `ignored_interior_outlets` counts graph outlets whose export receiver is
    an in-grid inactive cell: they are not outer faces and stay closed (obstruction)."""

    shape: tuple[int, int]
    dx_m: float
    open_x_faces: np.ndarray
    open_y_faces: np.ndarray
    weights: Any
    active: Any
    ray_continues: Any
    edge_cell: Any
    reach_cells: Any
    reach_open: Any
    xp: ModuleType
    ignored_interior_outlets: int = 0

    @property
    def n_open_faces(self) -> int:
        return int(self.open_x_faces.sum() + self.open_y_faces.sum())


def _reach(active: np.ndarray, open_line: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host static geometry of rays running along axis 1 on oriented `(L, S)` `active` (other slots are oriented
    views); `open_line` `(L,)` bool = the face beyond the last step is open for that line. Returns `continues`,
    `edge`, `reach` (cells before the ray ends) and `reach_open`; python loop over steps (domain extent)."""
    ny, nx = active.shape
    continues = np.empty((ny, nx), dtype=np.bool_)
    continues[:, : nx - 1] = active[:, 1:]
    continues[:, nx - 1] = open_line
    edge = np.zeros((ny, nx), dtype=np.bool_)
    edge[:, nx - 1] = True
    reach = np.zeros((ny, nx), dtype=np.int64)
    ends_open = np.zeros((ny, nx), dtype=np.bool_)
    ends_open[:, nx - 1] = open_line
    for c in range(nx - 2, -1, -1):
        nxt = active[:, c + 1]
        reach[:, c] = np.where(nxt, reach[:, c + 1] + 1, 0)
        ends_open[:, c] = np.where(nxt, ends_open[:, c + 1], False)
    return continues, edge, reach, ends_open


def _orient(a: Any, slot: int) -> Any:
    """View of a cell-frame array `(ny, nx, ...)` with the ray direction running along axis 1 (increasing)."""
    if slot == 0:
        return a
    if slot == 1:
        return a.swapaxes(0, 1)[:, ::-1]
    if slot == 2:
        return a[:, ::-1]
    return a.swapaxes(0, 1)


def _unorient(a: Any, slot: int) -> Any:
    if slot == 0:
        return a
    if slot == 1:
        return a[:, ::-1].swapaxes(0, 1)
    if slot == 2:
        return a[:, ::-1]
    return a.swapaxes(0, 1)


def _face_masks(active: np.ndarray, open_sides: Any, open_x_faces: Any, open_y_faces: Any
                ) -> tuple[np.ndarray, np.ndarray]:
    """Validated host `(ny, nx + 1)` / `(ny + 1, nx)` bool open-face masks. Default (nothing supplied) = all
    CLOSED. `open_sides` `(west, east, south, north)` is a shorthand that opens each named side's faces of
    ACTIVE edge cells (inactive edge cells are never opened); it cannot be combined with explicit face arrays.
    Explicit arrays are validated strictly: bool, exact shape, interior faces closed, no open face on an
    inactive edge cell."""
    ny, nx = active.shape
    open_x = np.zeros((ny, nx + 1), dtype=np.bool_)
    open_y = np.zeros((ny + 1, nx), dtype=np.bool_)
    if open_sides is not None:
        if open_x_faces is not None or open_y_faces is not None:
            raise SplashError("give either open_sides or explicit open_x_faces/open_y_faces, not both")
        sides = tuple(open_sides)
        if len(sides) != 4 or not all(isinstance(v, (bool, np.bool_)) for v in sides):
            raise SplashError("open_sides must be four bools (west, east, south, north)")
        west, east, south, north = (bool(v) for v in sides)
        open_x[:, 0] = west & active[:, 0]
        open_x[:, nx] = east & active[:, nx - 1]
        open_y[0, :] = south & active[0, :]
        open_y[ny, :] = north & active[ny - 1, :]
        return open_x, open_y
    for name, given, shape in (("open_x_faces", open_x_faces, (ny, nx + 1)), ("open_y_faces", open_y_faces, (ny + 1, nx))):
        if given is None:
            continue
        arr = np.array(given)
        if arr.dtype != np.bool_:
            raise SplashError(f"{name} must be a bool array, got {arr.dtype}")
        if arr.shape != shape:
            raise SplashError(f"{name} shape {arr.shape} != {shape} (MAPLE face layout)")
        if name == "open_x_faces":
            open_x = arr
        else:
            open_y = arr
    if open_x[:, 1:nx].any() or open_y[1:ny, :].any():
        raise SplashError("only OUTER faces can be open: an interior face is marked open")
    if np.any(open_x[:, 0] & ~active[:, 0]) or np.any(open_x[:, nx] & ~active[:, nx - 1]) \
            or np.any(open_y[0, :] & ~active[0, :]) or np.any(open_y[ny, :] & ~active[ny - 1, :]):
        raise SplashError("an open face adjoins an inactive (nodata/closed) edge cell; such faces cannot be opened")
    return open_x, open_y


def splash_grid(slope: Any, aspect: Any, active: Any, dx_m: float, *, open_sides: Any = None,
                open_x_faces: Any = None, open_y_faces: Any = None, xp: ModuleType | None = None,
                ignored_interior_outlets: int = 0) -> SplashGrid:
    """Validate and place the static grid. `slope` (>= 0, finite on active cells), `aspect` integer codes 0..4
    (1..4 required where an active cell is sloped), `active` bool, all `(ny, nx)`; `dx_m` finite > 0.

    Exterior openness (DEFAULT ALL CLOSED: an unspecified boundary never exports): per-face `open_x_faces`
    `(ny, nx + 1)` and `open_y_faces` `(ny + 1, nx)` (MAPLE face layout, True = the outer face exports), or the
    generic shorthand `open_sides=(west, east, south, north)`; see `_face_masks`."""
    from maple.core.backend import freeze, to_device

    s = np.array(slope, dtype=np.float64)
    act = np.array(active)
    asp = np.array(aspect)
    if act.dtype != np.bool_:
        raise SplashError(f"active must be a bool array, got {act.dtype}")
    if s.ndim != 2 or act.shape != s.shape or asp.shape != s.shape or min(s.shape) < 1:
        raise SplashError(f"slope, aspect and active must be equal non-empty 2-D shapes, got {s.shape}, "
                          f"{asp.shape}, {act.shape}")
    if asp.dtype.kind not in "iu":
        raise SplashError(f"aspect must hold integer legacy codes, got {asp.dtype}")
    if not np.all(np.isfinite(s[act])) or np.any(s[act] < 0.0):
        raise SplashError("slope must be finite and >= 0 on active cells")
    if np.any(asp[act] < 0) or np.any(asp[act] > 4):
        raise SplashError("aspect codes must lie in 0..4 on active cells")
    if np.any(act & (s > 0.0) & (asp == 0)):
        raise SplashError("a sloped active cell needs an aspect code 1..4 (0 is only valid with zero slope)")
    if isinstance(dx_m, bool) or not isinstance(dx_m, (int, float, np.integer, np.floating)):
        raise SplashError(f"dx_m must be a real number, got {type(dx_m).__name__}")
    dx = float(dx_m)
    if not (math.isfinite(dx) and dx > 0.0):
        raise SplashError(f"dx_m must be finite and > 0, got {dx_m!r}")
    open_x, open_y = _face_masks(act, open_sides, open_x_faces, open_y_faces)
    namespace = np if xp is None else xp

    s_act = np.where(act, s, 0.0)
    weights = ray_weights(s_act, np.where(act, asp, 0)) * act[None]
    ny, nx = s.shape
    continues = np.empty((4, ny, nx), dtype=np.bool_)
    edge = np.empty((4, ny, nx), dtype=np.bool_)
    reach = np.empty((4, ny, nx), dtype=np.int64)
    ends_open = np.empty((4, ny, nx), dtype=np.bool_)
    for slot, name in enumerate(RAY_ORDER):
        oriented_active = np.ascontiguousarray(_orient(act, slot))
        line_open = (open_x[:, nx], open_y[0, :], open_x[:, 0], open_y[ny, :])[slot]  # east, south, west, north
        c, e, r, o = _reach(oriented_active, line_open)
        continues[slot], edge[slot], reach[slot], ends_open[slot] = (
            _unorient(c, slot), _unorient(e, slot), _unorient(r, slot), _unorient(o, slot))

    def place(v):
        return freeze(to_device(np.ascontiguousarray(v), namespace))

    for flag in (open_x, open_y):
        flag.flags.writeable = False
    return SplashGrid(shape=(ny, nx), dx_m=dx, open_x_faces=np.ascontiguousarray(open_x),
                      open_y_faces=np.ascontiguousarray(open_y), weights=place(weights), active=place(act),
                      ray_continues=place(continues), edge_cell=place(edge), reach_cells=place(reach),
                      reach_open=place(ends_open), xp=namespace, ignored_interior_outlets=int(ignored_interior_outlets))


def splash_grid_from_graph(graph: Any) -> SplashGrid:
    """`SplashGrid` from a `maple_syrup.routing.RoutingGraph` (final slope, aspect, active mask, dx, namespace).

    The graph's own export configuration decides which faces open, and nothing else: an OUTLET cell
    (`graph.outlet`: active, receiver not active but export-flagged) opens exactly the OUTER face in the
    direction of its `graph.aspect` (`routing.ASPECT_STEPS`). A graph with no outlets is fully closed. An outlet
    whose export receiver is an in-grid inactive cell is not an outer face: it stays closed (obstruction) and is
    counted in `ignored_interior_outlets`. Inactive cells are obstructions. Never reads the export ring directly."""
    from maple_syrup.routing import ASPECT_STEPS

    outlet = np.asarray(graph.outlet, dtype=np.bool_)
    aspect = np.asarray(graph.aspect)
    ny, nx = outlet.shape
    open_x = np.zeros((ny, nx + 1), dtype=np.bool_)
    open_y = np.zeros((ny + 1, nx), dtype=np.bool_)
    ignored = 0
    for r, c in zip(*np.nonzero(outlet), strict=True):
        code = int(aspect[r, c])
        if code not in ASPECT_STEPS:
            raise SplashError(f"outlet cell ({r}, {c}) has aspect {code}; an exporting cell must have aspect 1..4")
        dr, dc = ASPECT_STEPS[code]
        rr, cc = int(r) + dr, int(c) + dc
        if rr < 0:
            open_y[0, c] = True
        elif rr >= ny:
            open_y[ny, c] = True
        elif cc < 0:
            open_x[r, 0] = True
        elif cc >= nx:
            open_x[r, nx] = True
        else:
            ignored += 1
    return splash_grid(graph.slope, graph.aspect, graph.active, graph.dx_m, open_x_faces=open_x, open_y_faces=open_y,
                       xp=graph.xp, ignored_interior_outlets=ignored)


# --- result ----------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SplashResult:
    """Instantaneous splash redistribution of actual pickup. All `(ny, nx, nc)` FP64 kg in the grid namespace
    unless noted.

    deposition_kg               total final deposit per cell and class (local + ray bins + obstruction tail)
    local_deposition_kg         the `P (1 - g)` part
    ray_bin_deposition_kg       interior ray bins `P w g^n (1 - g)`
    obstruction_tail_deposition_kg  exact remaining tail of rays stopped by an inactive cell or closed edge,
                                deposited at the last active cell (the source when blocked at once)
    export_edge_cell_kg         tail exported across open edges, attributed to the EDGE cell that holds it
                                when it crosses (the MAPLE `boundary_export_by_cell_class_kg` convention)
    export_source_cell_kg       the same export attributed to the SOURCE cell (independent closed form)
    x_face_gross_kg / x_face_net_kg   `(ny, nx + 1, nc)`; y `(ny + 1, nx, nc)`; net + east / north
    class_*_kg                  `(nc,)` pairwise sums
    class_budget_residual_kg    pickup - deposition - export_edge per class (signed); `class_budget_tolerance_kg`
    cell_balance_residual_kg    deposition - pickup - net face inflow per cell/class; `cell_balance_tolerance_kg`
    source_export_residual_kg   sum(export_source) - sum(export_edge) per class (diagnostic, not enforced);
                                `source_export_tolerance_kg`
    """

    deposition_kg: Any
    local_deposition_kg: Any
    ray_bin_deposition_kg: Any
    obstruction_tail_deposition_kg: Any
    export_edge_cell_kg: Any
    export_source_cell_kg: Any
    x_face_gross_kg: Any
    y_face_gross_kg: Any
    x_face_net_kg: Any
    y_face_net_kg: Any
    class_pickup_kg: Any
    class_deposition_kg: Any
    class_export_kg: Any
    class_export_source_kg: Any
    class_budget_residual_kg: Any
    class_budget_tolerance_kg: Any
    cell_balance_residual_kg: Any
    cell_balance_tolerance_kg: Any
    source_export_residual_kg: Any
    source_export_tolerance_kg: Any

    def face_flux(self):
        """MAPLE `FaceFluxResult` of the splash crossings (export crossings included)."""
        from maple.aeolian.flux.face_flux import FaceFluxResult

        return FaceFluxResult(x_face_crossing_mass_kg=self.x_face_gross_kg, y_face_crossing_mass_kg=self.y_face_gross_kg,
                              x_face_net_crossing_mass_kg=self.x_face_net_kg,
                              y_face_net_crossing_mass_kg=self.y_face_net_kg)


def _sweep(src: Any, keep: Any, g: Any, omg: Any, xp: ModuleType) -> tuple[Any, Any, Any]:
    """One direction on oriented arrays `src` `(L, S, nc)` (ray mass launched at each cell BEFORE its first
    decay), `keep` `(L, S)` bool (the next cell is active or the edge is open). Recurrence per step:
    `F_out = g (F_in + src)`, interior bin `F_in (1 - g)`; where the ray cannot continue the whole `F_out` is
    the tail. Returns `(bins, tail, cross)`, `cross` = flux through each cell's forward face."""
    n_lines, n_steps, nc = src.shape
    bins = xp.zeros((n_lines, n_steps, nc), dtype=np.float64)
    tail = xp.zeros((n_lines, n_steps, nc), dtype=np.float64)
    cross = xp.zeros((n_lines, n_steps, nc), dtype=np.float64)
    flux = xp.zeros((n_lines, nc), dtype=np.float64)
    for s in range(n_steps):
        out = g * (flux + src[:, s])
        bins[:, s] = flux * omg
        passed = xp.where(keep[:, s, None], out, 0.0)
        tail[:, s] = out - passed
        cross[:, s] = passed
        flux = passed
    return bins, tail, cross


def _array_rays(grid: SplashGrid, pickup_kg: Any, g: Any, omg: Any, nc: int, xp: ModuleType):
    """The NumPy/CuPy array reference sweep (python loop over the domain extent, vectorized over lines/classes).
    Returns `(bins_total, tail_total, export_edge, x_east, x_west, y_north, y_south)`."""
    ny, nx = grid.shape
    bins_total = xp.zeros((ny, nx, nc), dtype=np.float64)
    tail_total = xp.zeros((ny, nx, nc), dtype=np.float64)
    export_edge = xp.zeros((ny, nx, nc), dtype=np.float64)
    x_east = xp.zeros((ny, nx + 1, nc), dtype=np.float64)
    x_west = xp.zeros((ny, nx + 1, nc), dtype=np.float64)
    y_north = xp.zeros((ny + 1, nx, nc), dtype=np.float64)
    y_south = xp.zeros((ny + 1, nx, nc), dtype=np.float64)
    for slot in range(4):
        src = pickup_kg * grid.weights[slot][..., None]
        bins, tail, cross = (_unorient(r, slot) for r in _sweep(
            _orient(src, slot), _orient(grid.ray_continues[slot], slot), g, omg, xp))
        bins_total += bins
        tail_total += tail
        export_edge += xp.where(grid.edge_cell[slot][..., None], cross, 0.0)
        if slot == 0:
            x_east[:, 1:] += cross  # east face of column c is face index c + 1
        elif slot == 2:
            x_west[:, :nx] += cross  # west face of column c is face index c
        elif slot == 3:
            y_north[1:] += cross  # north face of row r is face index r + 1
        else:
            y_south[:ny] += cross  # south face of row r is face index r
    return bins_total, tail_total, export_edge, x_east, x_west, y_north, y_south


IMPLEMENTATIONS = ("array", "cuda", "auto")


def resolve_implementation(implementation: str, xp: ModuleType) -> str:
    """`"array"` (reference, any namespace), `"cuda"` (CuPy namespace only; no fallback; errors if unavailable) or
    `"auto"`: `"cuda"` only for a CuPy grid AND when `splash_cuda.CUDA_QUALIFIED` has been set True by a qualified
    run, else `"array"`. Anything else, or `"cuda"` on a NumPy grid, is refused."""
    if implementation not in IMPLEMENTATIONS:
        raise SplashError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
    if implementation == "array":
        return "array"
    if implementation == "cuda":
        if xp is np:
            raise SplashError("implementation 'cuda' needs a CuPy grid; the NumPy namespace is unsupported")
        return "cuda"
    from maple_syrup import splash_cuda

    return "cuda" if (xp is not np and splash_cuda.CUDA_QUALIFIED) else "array"


def _check_grid(grid: SplashGrid) -> None:
    """Metadata corruption guard (no device read): shapes, dtypes and namespace of every placed array."""
    from maple.core.backend import array_namespace

    ny, nx = grid.shape
    expected = {"weights": ((4, ny, nx), np.float64), "active": ((ny, nx), np.bool_),
                "ray_continues": ((4, ny, nx), np.bool_), "edge_cell": ((4, ny, nx), np.bool_),
                "reach_cells": ((4, ny, nx), np.int64), "reach_open": ((4, ny, nx), np.bool_)}
    for name, (shape, dtype) in expected.items():
        array = getattr(grid, name)
        if tuple(array.shape) != shape or array.dtype != dtype:
            raise SplashError(f"grid.{name} is corrupt: shape {tuple(array.shape)} dtype {array.dtype}, "
                              f"expected {shape} {np.dtype(dtype)}")
        if array_namespace(array) is not grid.xp:
            raise SplashError(f"grid.{name} is not in the grid namespace {grid.xp.__name__!r}")
    if grid.open_x_faces.shape != (ny, nx + 1) or grid.open_y_faces.shape != (ny + 1, nx) \
            or grid.open_x_faces.dtype != np.bool_ or grid.open_y_faces.dtype != np.bool_:
        raise SplashError("grid open-face masks are corrupt (shape/dtype)")
    if grid.open_x_faces[:, 1:nx].any() or grid.open_y_faces[1:ny, :].any():
        raise SplashError("grid open-face masks are corrupt: an interior face is open")


def splash_redistribute(params: SplashParameters, grid: SplashGrid, pickup_kg: Any, *,
                        implementation: str = "array") -> SplashResult:
    """Redistribute ACTUAL pickup `pickup_kg` `(ny, nx, nc)` by the exact splash law (see the module docstring).
    `implementation` selects the ray sweep: `"array"` (default reference), `"cuda"` (CuPy only) or `"auto"`
    (see `resolve_implementation`); the closure checks, validation and outputs are identical.
    Pure; raises `SplashError` before returning anything on invalid input or a failed closure."""
    from maple.core.backend import (
        DeferredChecks,
        MixedArrayNamespaceError,
        array_namespace,
        errstate,
        finite_flag,
        is_array,
        negative_flag,
        pairwise_bound_scale_factor,
        pairwise_sum_over_leading_axes,
        true_flag,
    )
    from maple.surface.voxels._numerics import summation_error_bound_kg

    if not isinstance(params, SplashParameters) or not isinstance(grid, SplashGrid):
        raise SplashError("params must be SplashParameters and grid a SplashGrid")
    xp = grid.xp
    if params.xp is not xp:
        raise SplashError("parameters and grid must share one array namespace")
    _check_grid(grid)
    chosen = resolve_implementation(implementation, xp)
    ny, nx = grid.shape
    nc = params.n_classes
    if not is_array(pickup_kg):
        raise SplashError(f"pickup_kg must be a NumPy/CuPy array, got {type(pickup_kg).__name__}")
    try:
        namespace = array_namespace(pickup_kg)
    except MixedArrayNamespaceError as exc:
        raise SplashError(str(exc)) from None
    if namespace is not xp:
        raise SplashError(f"pickup_kg must be in the grid namespace {xp.__name__!r}, got {namespace.__name__!r}")
    if tuple(pickup_kg.shape) != (ny, nx, nc):
        raise SplashError(f"pickup_kg shape {tuple(pickup_kg.shape)} != {(ny, nx, nc)}")
    if pickup_kg.dtype != np.float64:
        raise SplashError(f"pickup_kg must be float64, got {pickup_kg.dtype}")
    with np.errstate(over="ignore"):
        a_host = params.inverse_distance_per_m * grid.dx_m  # a = spd dx, host parameters only, no device read
    if not np.all(np.isfinite(a_host)):
        raise SplashError("spd * dx_m overflowed FP64 for some class")

    checks = DeferredChecks()
    checks.require(finite_flag(pickup_kg), "pickup_kg must be finite everywhere")
    checks.forbid(negative_flag(pickup_kg), "pickup_kg must be >= 0 everywhere")
    checks.forbid(true_flag(~grid.active[..., None] & (pickup_kg != 0.0)), "pickup_kg must be 0 on inactive cells")
    try:
        checks.resolve()
    except ValueError as exc:
        raise SplashError(str(exc)) from None

    a = params.d_inverse_distance * grid.dx_m  # (nc,) in the namespace; the host copy above only validated it
    with errstate(xp=xp, all="ignore"):
        g = xp.exp(-a)
        omg = -xp.expm1(-a)  # 1 - g without cancellation for small a
        local = pickup_kg * omg
        if chosen == "cuda":
            from maple_syrup.splash_cuda import cuda_rays

            bins_total, tail_total, export_edge, x_east, x_west, y_north, y_south = cuda_rays(
                grid, pickup_kg, g, omg, nc)
        else:
            bins_total, tail_total, export_edge, x_east, x_west, y_north, y_south = _array_rays(
                grid, pickup_kg, g, omg, nc, xp)
        # Closed-form export by source: P w g^(m + 1) wherever the ray ends at an open face.
        export_source = xp.zeros((ny, nx, nc), dtype=np.float64)
        for slot in range(4):
            src = pickup_kg * grid.weights[slot][..., None]
            decay = xp.exp(-((grid.reach_cells[slot] + 1).astype(np.float64)[..., None] * a))
            export_source += xp.where(grid.reach_open[slot][..., None], src * decay, 0.0)
        deposition = (local + bins_total) + tail_total
        x_gross = x_east + x_west
        y_gross = y_north + y_south
        x_net = x_east - x_west
        y_net = y_north - y_south
        net_in = (x_net[:, :-1] - x_net[:, 1:]) + (y_net[:-1] - y_net[1:])
        cell_residual = (deposition - pickup_kg) - net_in
        cell_scale = pickup_kg + deposition + x_gross[:, :-1] + x_gross[:, 1:] + y_gross[:-1] + y_gross[1:]
        cell_tol = summation_error_bound_kg(_CLOSURE_OPS, cell_scale)

        before = pairwise_sum_over_leading_axes(pickup_kg)
        dep_total = pairwise_sum_over_leading_axes(deposition)
        exp_total = pairwise_sum_over_leading_axes(export_edge)
        exp_src_total = pairwise_sum_over_leading_axes(export_source)
        residual = (before - dep_total) - exp_total
        pairwise = pairwise_bound_scale_factor(ny * nx)
        scale = xp.max(cell_scale.reshape(-1, nc), axis=0)
        tolerance = summation_error_bound_kg(_CLOSURE_OPS, scale) + 2.0 * pairwise * (before + dep_total + exp_total) \
            + 8.0 * _EPS * before
        src_residual = exp_src_total - exp_total
        src_tol = (2.0 * (ny + nx) + 128.0) * _EPS * before + summation_error_bound_kg(_CLOSURE_OPS, scale)

    checks = DeferredChecks()
    for name, array in (("deposition", deposition), ("export_edge", export_edge), ("export_source", export_source),
                        ("x_face_gross", x_gross), ("y_face_gross", y_gross), ("x_face_net", x_net),
                        ("y_face_net", y_net), ("cell_balance", cell_residual), ("class_budget", residual)):
        checks.require(finite_flag(array), f"splash produced non-finite {name}")
    for name, array in (("deposition", deposition), ("export_edge", export_edge), ("export_source", export_source),
                        ("x_face_gross", x_gross), ("y_face_gross", y_gross)):
        checks.forbid(negative_flag(array), f"splash produced negative {name}")
    checks.forbid(true_flag(xp.abs(cell_residual) > cell_tol),
                  "per-cell splash balance (deposition - pickup = net face inflow) violated beyond FP64 tolerance")
    checks.forbid(true_flag(xp.abs(residual) > tolerance),
                  "per-class splash mass not conserved beyond the declared FP64 bound")
    try:
        checks.resolve()
    except ValueError as exc:
        raise SplashError(str(exc)) from None

    return SplashResult(
        deposition_kg=deposition, local_deposition_kg=local, ray_bin_deposition_kg=bins_total,
        obstruction_tail_deposition_kg=tail_total, export_edge_cell_kg=export_edge,
        export_source_cell_kg=export_source, x_face_gross_kg=x_gross, y_face_gross_kg=y_gross,
        x_face_net_kg=x_net, y_face_net_kg=y_net, class_pickup_kg=before, class_deposition_kg=dep_total,
        class_export_kg=exp_total, class_export_source_kg=exp_src_total, class_budget_residual_kg=residual,
        class_budget_tolerance_kg=tolerance, cell_balance_residual_kg=cell_residual,
        cell_balance_tolerance_kg=cell_tol, source_export_residual_kg=src_residual,
        source_export_tolerance_kg=src_tol)


# --- MAPLE exchange helper ---------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SplashExchangeAudit:
    """Per-class `(nc,)` audit of one native exchange (device arrays; `scalar_totals()` reads them to host).

    requested/actual pickup, availability and holdings shortfall and the pickup numerical residual come from
    MAPLE's pickup result; `splash_*_request` are the redistribution outputs; `native_*` the amounts MAPLE
    actually deposited / exported; `unapplied_*` = request - native (MAPLE's deposition result may differ
    from the capped request by its sub-resolution burial residual, reported in `numerical_residual_scalar_kg`);
    `mobile_change_*` is `mobile after - before`, ~0 because only the splash mass is added to the pool and
    MAPLE removes exactly that (preexisting water-mobile sediment is never taken: refused if it were);
    `net_bed_change_by_class_kg` is MAPLE's independently measured bed inventory change over both calls and
    `conservation_residual_by_class_kg` = net bed change + mobile change + native export, ~0."""

    requested_pickup_by_class_kg: Any
    actual_pickup_by_class_kg: Any
    availability_shortfall_by_class_kg: Any
    holdings_shortfall_by_class_kg: Any
    pickup_numerical_residual_by_class_kg: Any
    splash_deposit_request_by_class_kg: Any
    native_deposited_by_class_kg: Any
    splash_export_request_by_class_kg: Any
    native_export_by_class_kg: Any
    unapplied_deposit_by_class_kg: Any
    unapplied_export_by_class_kg: Any
    mobile_before_by_class_kg: Any
    mobile_after_by_class_kg: Any
    mobile_change_by_class_kg: Any
    net_bed_change_by_class_kg: Any
    conservation_residual_by_class_kg: Any
    numerical_residual_scalar_kg: float
    mobile_change_by_cell_class_kg: Any

    def scalar_totals(self) -> dict[str, float]:
        """Class-summed host floats (a device read); the scalar audit."""
        out = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "mobile_change_by_cell_class_kg":
                continue
            out[name] = float(value) if name == "numerical_residual_scalar_kg" else float(value.sum())
        return out


@dataclass(frozen=True, eq=False)
class SplashExchange:
    """`bed` the new `BedState`; `pickup` / `deposit` MAPLE's genuine `WaterProcessResult`s; `splash` the
    redistribution of the ACTUAL pickup; `audit` a `SplashExchangeAudit`."""

    bed: Any
    pickup: Any
    deposit: Any
    splash: SplashResult
    audit: SplashExchangeAudit


def apply_splash_exchange(state: Any, context: Any, params: SplashParameters, grid: SplashGrid,
                          requested_pickup_kg: Any, *, implementation: str = "array") -> SplashExchange:
    """Native MAPLE splash exchange (NOT event integration): pickup cap -> redistribute the ACTUAL pickup ->
    native deposition/export.

    1. `apply_bed_demand(state, WaterProcessDemand(requested, 0))`: MAPLE alone caps by availability and
       holdings and moves the actual pickup into the mobile pool.
    2. `splash_redistribute` of `actual_removal_by_cell_class_kg` (never of the demand).
    3. The mobile pool is published as `preexisting mobile + deposit + export` (the splash mass is the only
       thing added; the old water-mobile sediment is not moved or read by the law; depth is kept as is),
       then `apply_bed_demand(WaterProcessDemand(0, deposition, face_flux, boundary_export))` lets MAPLE
       deposit and export. Soil, availability, depth and ledger authority stay MAPLE's; nothing is clipped or
       reset here, and no inventory is created.

    `requested_pickup_kg` is the dry-splash demand, e.g. `sediment_physics.dry_raindrop_pickup_demand(...)`;
    pickup is refused on a cell with surface depth > 0 (the wet law owns those) or on an inactive cell. Pure."""
    from maple.core.backend import (
        DeferredChecks,
        MixedArrayNamespaceError,
        finite_flag,
        is_array,
        negative_flag,
        resolve_operand_namespace,
        true_flag,
    )
    from maple.surface.voxels._numerics import summation_error_bound_kg
    from maple.water.interfaces import WaterProcessDemand

    from maple_syrup.sediment_bed import apply_bed_demand, with_water

    xp = grid.xp
    ny, nx = grid.shape
    nc = params.n_classes
    if len(context.grain_classes.classes) != nc:
        raise SplashError(f"params has {nc} classes but the MAPLE case has {len(context.grain_classes.classes)}")
    if not is_array(requested_pickup_kg) or tuple(requested_pickup_kg.shape) != (ny, nx, nc) \
            or requested_pickup_kg.dtype != np.float64:
        raise SplashError(f"requested_pickup_kg must be a float64 array of shape {(ny, nx, nc)}")
    try:
        namespace = resolve_operand_namespace(state, requested_pickup_kg, context.initial_available_fraction)
    except MixedArrayNamespaceError as exc:
        raise SplashError(str(exc)) from None
    if namespace is not xp:
        raise SplashError(f"state and requested_pickup_kg must be in the grid namespace {xp.__name__!r}")
    depth = state.water.depth_m
    old_mobile = state.water.mobile_mass_by_cell_class_kg
    checks = DeferredChecks()
    checks.require(finite_flag(requested_pickup_kg), "requested_pickup_kg must be finite everywhere")
    checks.forbid(negative_flag(requested_pickup_kg), "requested_pickup_kg must be >= 0 everywhere")
    checks.forbid(true_flag((requested_pickup_kg != 0.0) & (depth[..., None] > 0.0)),
                  "splash pickup requested on a wet cell (surface depth > 0); the wet law owns those cells")
    checks.forbid(true_flag((requested_pickup_kg != 0.0) & ~grid.active[..., None]),
                  "splash pickup requested on an inactive cell")
    try:
        checks.resolve()
    except ValueError as exc:
        raise SplashError(str(exc)) from None

    zeros = xp.zeros((ny, nx, nc), dtype=np.float64)
    bed1, pickup = apply_bed_demand(state, context, WaterProcessDemand(requested_pickup_kg, zeros))
    splash = splash_redistribute(params, grid, pickup.actual_removal_by_cell_class_kg, implementation=implementation)
    landed = splash.deposition_kg + splash.export_edge_cell_kg
    bed2 = with_water(bed1, context, bed1.water.depth_m, old_mobile + landed)
    demand = WaterProcessDemand(
        requested_removal_by_cell_class_kg=zeros, requested_deposition_by_cell_class_kg=splash.deposition_kg,
        face_flux=splash.face_flux(), boundary_export_by_cell_class_kg=splash.export_edge_cell_kg)
    bed3, deposit = apply_bed_demand(bed2, context, demand)

    mobile_after = bed3.water.mobile_mass_by_cell_class_kg
    change = mobile_after - old_mobile
    tol = summation_error_bound_kg(8, old_mobile + landed + mobile_after)
    checks = DeferredChecks()
    checks.forbid(true_flag(change < -tol), "native exchange removed preexisting water-mobile sediment")
    try:
        checks.resolve()
    except ValueError as exc:
        raise SplashError(str(exc)) from None

    dep_req = splash.class_deposition_kg
    exp_req = splash.class_export_kg
    native_dep = deposit.deposited_mass_by_class_kg
    native_exp = deposit.boundary_export_by_class_kg
    mobile_before = old_mobile.sum(axis=(0, 1))
    mobile_after_c = mobile_after.sum(axis=(0, 1))
    net_bed = pickup.bed_change_by_class_kg + deposit.bed_change_by_class_kg
    audit = SplashExchangeAudit(
        requested_pickup_by_class_kg=pickup.requested_removal_by_class_kg,
        actual_pickup_by_class_kg=pickup.actual_removal_by_class_kg,
        availability_shortfall_by_class_kg=pickup.availability_shortfall_by_class_kg,
        holdings_shortfall_by_class_kg=pickup.shortfall_by_class_kg,
        pickup_numerical_residual_by_class_kg=pickup.numerical_residual_by_class_kg,
        splash_deposit_request_by_class_kg=dep_req, native_deposited_by_class_kg=native_dep,
        splash_export_request_by_class_kg=exp_req, native_export_by_class_kg=native_exp,
        unapplied_deposit_by_class_kg=dep_req - native_dep, unapplied_export_by_class_kg=exp_req - native_exp,
        mobile_before_by_class_kg=mobile_before, mobile_after_by_class_kg=mobile_after_c,
        mobile_change_by_class_kg=mobile_after_c - mobile_before, net_bed_change_by_class_kg=net_bed,
        conservation_residual_by_class_kg=(net_bed + (mobile_after_c - mobile_before)) + native_exp,
        numerical_residual_scalar_kg=float(deposit.numerical_residual_scalar_kg),
        mobile_change_by_cell_class_kg=change)
    return SplashExchange(bed=bed3, pickup=pickup, deposit=deposit, splash=splash, audit=audit)
