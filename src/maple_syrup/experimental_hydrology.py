"""EXPERIMENTAL water-only hydraulic alternatives: explicit conservative D4 kinematic wave and uniform-grid local inertia
(task gpu_hydrology_candidates). Contracts, static geometry, shared error resolution and the pure CPU NumPy reference
step of both solvers. The GPU form is `experimental_cuda`, the shared driver `experimental_storm`, the CLI
`experimental_experiment`. Nothing here replaces or changes the accepted MAHLERAN-inspired method-5 hydrology
(`routing`, `hydrology_numba`, `hydrology_cuda`, `storm`), whose defaults are untouched; nothing here is a sediment, wind,
splash, evapotranspiration, dry-reset or evolving-terrain model, and no equivalence with the legacy hydraulics is claimed.

Both solvers split a step the same way: (1) the EXISTING accepted column physics (`infiltration.column_step` on the CPU,
`hydrology_cuda.prepared_column_step` on the GPU; no infiltration formula is repeated here) turns the state
`(h, S)` and the rain rate into the post-rain/infiltration surface storage `h_c`; (2) the lateral redistribution below moves
that water. Units: depths m, unit discharge m2/s, volumes m3, time s, FP64.

EXPLICIT KINEMATIC WAVE (`method="explicit"`; fixed D4 graph of `routing`, k = sqrt(8 g S / f) from the legacy slope/friction)

    q_i   = k_i h_c,i^(3/2)                        (Darcy-Weisbach kinematic law, same k as the legacy sweep)
    F_i   = dt dx q_i                              (volume leaving cell i to its receiver or the export, used ONCE)
    h_new = h_c + (sum_donors F - F_i) / dx^2      (donors summed in the fixed `DONOR_SLOTS` order)

First-order Euler, no ordered dependency sweep, no bisection. The characteristic speed of q = k h^(3/2) is
dq/dh = (3/2) k sqrt(h) (not the material speed q/h); the step is admissible iff
`max (3/2) k sqrt(h_c) dt/dx <= cfl_max` (`0 < cfl_max <= 0.5`). POSITIVITY PROOF: that bound gives
`k sqrt(h_c) dt/dx <= cfl_max / 1.5 < 1`, so `F_i / dx^2 = h_c (k sqrt(h_c) dt/dx) <= h_c / 3`, and `h_new >= h_c - F_i/dx^2 > 0`
whatever the inflow. A violating step is REJECTED (a recoverable `HydraulicStepRejected`), never clipped; the driver
halves dt from the unchanged state. Reported velocity is `q(h_new)/h_new` at positive depth (0 when dry), the instantaneous
end-of-step value, distinct from the face volumes that actually moved. No legacy `hpre`/Crank-Nicolson consistency guard is
imposed (the candidate has no old flux).

LOCAL INERTIA (`method="local_inertial"`; uniform square MAPLE grid, row 0 = south, staggered signed faces)

State: `h, S (ny, nx)`, `qx (ny, nx+1)` positive east, `qy (ny+1, nx)` positive north (BOTH are retained for continuation).
Advective acceleration is dropped; local acceleration and the water-surface pressure gradient are kept, friction is
Darcy-Weisbach (NOT Manning). For a face between cells A (west/south) and B (east/north), eta = z + h:

    h_f  = max( max(eta_A, eta_B) - max(z_A, z_B), 0 )                       (dry face: q = 0, no mass is clipped)
    q'   = ( q - g h_f dt (eta_B - eta_A)/dx ) / ( 1 + dt (f_f/8) |q| / h_f^2 )     (q = old face flux, semi-implicit friction)
    h'   = h_c - (dt/dx) ( (qx'_east - qx'_west) + (qy'_north - qy'_south) )   (the SAME face arrays on both sides)

Check: steady uniform flow (eta slope = -S) gives (f/8) q^2/h^2 = g h S, i.e. v^2 = 8 g h S / f, the legacy law.
Friction source: Darcy-Weisbach friction slope f q|q|/(8 g h^3) (Kirstetter et al., arXiv:1609.04711 eq. 10) contributes
(f/8) q|q|/h^2 to the unit-width momentum balance; the Wflow local-inertial documentation (Manning, Deltares) provides the
staggered continuity/gravity-wave CFL pattern. The semi-implicit update and the boundary below are OUR adaptation, not an
exact implementation of either source. 2-D friction DEPARTURE: scalar friction per face using only that face's own normal
component |q| (the tangential component is neglected); the face friction factor is the arithmetic mean of the two cells'.
Gravity-wave CFL: the step is admissible iff `dt sqrt(2 g h_f,max) / dx <= cfl_max` (the sqrt 2 is the 2-D explicit
bound), else REJECTED and halved. Interior faces between two active cells carry momentum in both directions; faces touching an inactive cell or the
ring are CLOSED (q = 0; inactive inventories are retained and exchange nothing). No ring cell is opened automatically: only the faces of the
legacy outlet cells (`graph.outlet`, direction from `graph.aspect`) are open, with a declared DARCY NORMAL-FLOW boundary
`q_out = k_b h_donor^(3/2) >= 0`, `k_b = sqrt(8 g S_b / f_donor)`, `S_b = (z_cell - z_ring)/dx > 0` (the bed normal drop
to the ring cell; no imposed external inflow). That boundary closure differs from the interior hydraulics and from the legacy
edge rule (which copies a neighbouring slope); an outlet draining into an interior inactive cell is UNSUPPORTED and refused.
Negative depth after the update REJECTS the step (limiter "off", the default) -- depth is never clipped.
Optional `limiter="donor"` (documented dynamics change, off by default, activity counted and reported): each cell's total
outflow is limited to its available water `h_c dx^2` by one factor `phi_cell = min(1, avail/out) (1 - 16 eps)`, applied to
every face leaving that cell (the donor), once, so both ends of a face still see the same flux; the limited flux is what
is stored as the new momentum state. The removed volume and the number of limited cells are reported.

Conservation: per-cell and global water balances use the existing `routing.BALANCE_RTOL` coefficient (same form as the
legacy step); the event budget in the CLI uses the MAPLE-derived `volume_roundoff_bound_m3`. Failures raise before a result exists and never
modify an input (`HydraulicStepRejected` is the only recoverable class: retry the SAME state with a smaller dt).

Nothing here was run by its author (file-only tools); Codex records results.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

from maple_syrup.infiltration import (
    ColumnParameters,
    ColumnStep,
    _check_dt,
    column_step,
)
from maple_syrup.routing import (
    ASPECT_STEPS,
    BALANCE_RTOL,
    DONOR_SLOTS,
    GRAVITY_M_S2,
    RoutingGraph,
)

__all__ = [
    "DEFAULT_CFL_MAX",
    "LIMITERS",
    "LIMITER_SAFETY",
    "METHODS",
    "QUALIFICATION_STATUS",
    "TRANSFER_SCOPE",
    "CpuHydraulicSolver",
    "ExperimentalGeometryError",
    "ExperimentalHydrologyError",
    "HydraulicControl",
    "HydraulicState",
    "HydraulicStep",
    "HydraulicStepRejected",
    "LocalInertialGeometry",
    "advance_time",
    "build_local_inertial_geometry",
    "check_dt_cap",
    "check_state_time",
    "donor_cells_by_cell",
    "local_inertial_face_flux",
    "open_faces_from_graph",
    "resolve_flags",
    "stage_face_diagnostics",
    "validate_hydraulic_state",
]

METHODS = ("explicit", "local_inertial")
LIMITERS = ("off", "donor")
DEFAULT_CFL_MAX = 0.5
_EPS = float(np.finfo(np.float64).eps)
# Retained fraction margin of a limited donor: the chain divide/multiply/scale/gather/subtract of the limiter accumulates up
# to ~8 roundings of the cell's own depth, so 16 eps keeps h_new >= 0 without any clipping (the leftover is ~3.6e-15 h_c).
LIMITER_SAFETY = 1.0 - 16.0 * _EPS

# Flag bits (shared by the CPU reference and the CUDA kernels; the lowest set bit of a class is reported first).
EXPLICIT_BITS = {"nonfinite": 1 << 0, "negative": 1 << 1, "cfl": 1 << 2, "balance": 1 << 3}
LOCAL_BITS = {"state_nonfinite": 1 << 0, "closed_face": 1 << 1, "open_inflow": 1 << 2, "cfl": 1 << 3,
              "negative": 1 << 4, "nonfinite": 1 << 5, "balance": 1 << 6}
_SCALAR_NAMES = ("storage change", "export", "budget residual", "balance tolerance", "outlet discharge")
CFL_NAMES = {"explicit": "characteristic CFL (3/2) k sqrt(h) dt/dx",
             "local_inertial": "gravity-wave CFL dt sqrt(2 g h_face) / dx"}
TRANSFER_SCOPE = (
    "MAPLE read_transfer_counters counts only its instrumented helpers (to_host / to_device); a raw cupy.asarray(host value) "
    "bypasses those counters. In CuPy14.2 the Python-scalar path uses a fill kernel; host arrays can incur uncounted copies. "
    "A zero helper counter is not a complete transfer trace. The experimental solvers and the "
    "driver create no device array from a host value per step or per report row (Python floats enter kernels by value). "
    "CUDA loop host reads: two counted small packets per attempted step (column flags, lateral packet). Static uploads: "
    "counted, in preparation. Final reporting downloads: counted apart by the caller. Device-to-device copies, fills and "
    "allocations are not transfers. The baseline column kernels' internals were not independently audited here."
)
QUALIFICATION_STATUS = (
    "EXPERIMENTAL, not adopted, no production equivalence. Verified: short-step and controlled-storm CPU/GPU field agreement at "
    "the unchanged bounds rtol 2e-12 / atol 1e-14. NOT met: the full 5400 s local-inertial Plot 1 storm (root trialC2) closes "
    "water and agrees in final export to ~5e-14 m3 but some fields exceed those bounds (soil water, cumulative intake, "
    "surface storage and export rows, peak and snapshot velocities); the bounds were not widened. The same applies to the compiled "
    "Numba form against the NumPy reference (the full local storm exceeds the bounds in some fields; its lateral stage is bitwise "
    "equal to the reference when fed identical column inputs: case-specific evidence consistent with rounding amplification by "
    "independent trajectories, not a universal cause and not an all-backend identity claim). Velocity is NOT qualified for "
    "detachment or transport: the reconstructed cell speed is unbounded near drying cells and even the stage-consistent face "
    "flows reach Froude ~2 (up to ~0.4% of wet faces above 0.5), beyond the low-Froude range of local inertia. Open: long-storm "
    "backend stability/conditioning, advection, directional damping, wetting/drying and outlet treatment."
)


class ExperimentalHydrologyError(ValueError):
    """Invalid request, state, geometry or a failed check of an experimental solver. Raised before a result exists; no
    caller array is modified."""


class HydraulicStepRejected(ExperimentalHydrologyError):
    """The step is valid but too large for this state (CFL bound or, without the limiter, a negative depth): retry the SAME
    state with a smaller dt. Every other error is deliberately not this class."""


class ExperimentalGeometryError(ExperimentalHydrologyError):
    """Unsupported or invalid geometry/boundary specification. Raised while the geometry is built."""


# --- control, state, step -------------------------------------------------------------------------------------------
def _real(value: Any, name: str, error=ExperimentalHydrologyError) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise error(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


@dataclass(frozen=True)
class HydraulicControl:
    """Numerical options of a hydraulic solver (not of the driver): the CFL bound and the optional positivity limiter."""

    cfl_max: float = DEFAULT_CFL_MAX
    limiter: str = "off"

    def validated(self) -> HydraulicControl:
        cfl = _real(self.cfl_max, "cfl_max")
        if not (math.isfinite(cfl) and 0.0 < cfl <= 0.5):
            raise ExperimentalHydrologyError(
                "cfl_max must lie in (0, 0.5], the conservative CFL range (for the explicit method it implies the "
                "positivity bound proved in the module docstring; for local inertia it is only a gravity-wave stability "
                "limit: positivity is checked every step and handled by rejection or the donor limiter, not guaranteed "
                f"by this value), got {self.cfl_max!r}")
        if self.limiter not in LIMITERS:
            raise ExperimentalHydrologyError(f"limiter must be one of {LIMITERS}, got {self.limiter!r}")
        return self


@dataclass(frozen=True, eq=False)
class HydraulicState:
    """`t_s`, surface storage `depth_m` and retained soil water `soil_water_m` `(ny, nx)` float64, and for the local-inertial
    method the signed face fluxes `qx_m2_s (ny, nx+1)` (positive east) and `qy_m2_s (ny+1, nx)` (positive north). The explicit
    method has no momentum: both face fields must be None.

    `next_dt_cap_s` (default None, LAST so existing positional constructors are unchanged) is the numerical adaptive-step
    history the experimental driver needs to CONTINUE an event exactly: the step cap it would use next (it shrinks to the last
    step that survived a rejection and doubles back toward `max_dt_s` after clean full-size steps). Solvers never read it and
    always return None; `evolve_experimental` sets it on every accepted state and resumes from it (bounded by the new control).
    None means a fresh event (the cap starts at `max_dt_s`). When given it must be a real, finite, positive number."""

    t_s: float
    depth_m: Any
    soil_water_m: Any
    qx_m2_s: Any = None
    qy_m2_s: Any = None
    next_dt_cap_s: Any = None


@dataclass(frozen=True, eq=False)
class HydraulicStep:
    """One accepted step. Arrays in the solver's namespace (fresh), scalars host numbers on the CPU / 0-d device views on CUDA.

    face_volume_m3     explicit: {"out": (ny, nx) volume that left each cell to its receiver/export this step, >= 0};
                       local_inertial: {"x": (ny, nx+1), "y": (ny+1, nx)} SIGNED volumes dt dx q' that crossed each face
    used_flux_m2_s     the unit fluxes those volumes were made of (explicit: k h_c^(3/2); local inertial: the new face state)
    velocity_m_s       instantaneous at the END of the step: explicit q(h_new)/h_new (0 dry); local inertial the cell-centred
                       speed from face-averaged fluxes divided by h_new (0 dry) -- neither is the legacy velocity
    export_m3          volume through the open/outlet faces during the step (face volumes used, not the instantaneous q)
    outlet_discharge_m3_s  instantaneous end-of-step discharge through the outlets (dx x sum of outlet unit fluxes)
    max_cfl            the largest value of the CFL quantity named by `cfl_kind` (admissible iff <= cfl_max)
    limited_cells / limited_volume_m3  donor-limiter activity (0 when the limiter is off)
    face_flow_depth_m  the flow depth each face flux was COMPUTED from (the post-rain/infiltration stage depth, before this
                       step's continuity update; explicit: h_c of the donor cell, inactive cells show their stored depth with zero
                       flux): `{"out"}` / `{"x", "y"}`; with `used_flux_m2_s`
                       it gives the stage-consistent face velocity and Froude number (`stage_face_diagnostics`)

    LIMITATION (local inertia, measured by the root diagnostics, no qualification for erosion): `velocity_m_s` is a RECONSTRUCTED
    cell speed = face-averaged NEW flux / END-of-step depth. A draining cell's END depth can be arbitrarily small while the
    fluxes that emptied it are finite, so this ratio (and its Froude number) is unbounded near drying cells -- the root's
    synchronous snapshots show Fr_max 10.1 / 1.4 / 4e4 at 600 / 1200 / 1620 s (h ~ 1.6e-11 m) against a bulk median ~ 0.14.
    That is a property of the reconstruction, not evidence of mass loss, and nothing is clipped. It must NOT feed a future
    detachment/transport law without a fidelity qualification. The stage-consistent face diagnostics below separate the
    reconstruction from the face flows, but they are NOT a fix: the root's trialC2 stage diagnostics measured face Fr up to ~2
    (600 s) with up to ~0.4% of wet faces above 0.5, so the flows themselves leave the low-Froude range where local inertia is
    accurate. Neither quantity is qualified for erosion.
    """

    method: str
    implementation: str
    dt_s: float
    state: HydraulicState
    column: ColumnStep
    velocity_m_s: Any
    face_volume_m3: dict
    used_flux_m2_s: dict
    export_m3: Any
    outlet_discharge_m3_s: Any
    storage_change_m3: Any
    budget_residual_m3: Any
    max_cfl: Any
    max_cell_balance_residual_m: Any
    limited_cells: Any
    limited_volume_m3: Any
    cfl_kind: str
    face_flow_depth_m: Any = None


def stage_face_diagnostics(step: HydraulicStep) -> dict[str, Any]:
    """Stage-consistent face diagnostics of an accepted step (computed on demand, fresh arrays in the step's namespace).

    Units: `face_flow_depth_m` m, `face_velocity_m_s` m/s (signed, along the face normal: +east/+north for local inertia,
    the downstream direction for the explicit `out` faces), `face_froude` dimensionless. STAGE: the depth is the one the flux
    was computed from (post-rain/infiltration, before this step's continuity update) and the flux is the new face flux that
    used it, so `velocity = q / h_f` and `Fr = |v| / sqrt(g h_f)` pair a flux with ITS OWN depth. Dry faces (h_f = 0) report 0.
    With the donor limiter the stored (limited) flux is used with the unchanged `h_f`. These are diagnostics only; they
    change no state and replace no physics. They are NOT the cell velocity of `HydraulicStep.velocity_m_s`, and they are not
    claimed to be bounded or physically qualified: measured face Froude numbers exceed 0.5 on a small subset of wet faces."""
    from maple.core.backend import array_namespace

    if step.face_flow_depth_m is None:
        raise ExperimentalHydrologyError("this step carries no face flow depths")
    out: dict[str, Any] = {"face_flow_depth_m": {}, "face_velocity_m_s": {}, "face_froude": {},
                           "stage": "post rain/infiltration, before this step's continuity update; flux = the new face flux"}
    for key, depth in step.face_flow_depth_m.items():
        flux = step.used_flux_m2_s[key]
        xp = array_namespace(depth, flux)
        wet = depth > 0.0
        safe = xp.where(wet, depth, 1.0)
        velocity = xp.where(wet, flux / safe, 0.0)
        out["face_flow_depth_m"][key] = depth
        out["face_velocity_m_s"][key] = velocity
        out["face_froude"][key] = xp.where(wet, xp.abs(velocity) / xp.sqrt(GRAVITY_M_S2 * safe), 0.0)
    return out


# --- continuation metadata --------------------------------------------------------------------------------------------
def check_dt_cap(value: Any, name: str = "state.next_dt_cap_s") -> float | None:
    """None (a fresh event) or a real (bool, str, bytes, arrays are refused, never coerced), finite, strictly positive step cap in
    s; returns it as a float."""
    if value is None:
        return None
    cap = _real(value, name)
    if not (math.isfinite(cap) and cap > 0.0):
        raise ExperimentalHydrologyError(f"{name} must be None or finite and > 0, got {cap!r}")
    return cap


# --- strict state time ------------------------------------------------------------------------------------------------
def check_state_time(t_s: Any, name: str = "state.t_s") -> float:
    """A real (bool, str and None are refused, never coerced) finite number >= 0."""
    t = _real(t_s, name)
    if not (math.isfinite(t) and t >= 0.0):
        raise ExperimentalHydrologyError(f"{name} must be finite and >= 0, got {t!r}")
    return t


def advance_time(t_s: float, dt_s: float) -> float:
    """`t + dt` for a validated `t >= 0` and `dt > 0`; refuses an overflowing end time and a step that does not advance
    floating time (`t + dt == t`)."""
    t_new = t_s + dt_s
    if not math.isfinite(t_new):
        raise ExperimentalHydrologyError(f"state.t_s + dt_s overflows FP64 (t = {t_s}, dt = {dt_s})")
    if not t_new > t_s:
        raise ExperimentalHydrologyError(f"dt_s = {dt_s} does not advance floating time at t = {t_s}")
    return t_new


# --- shared error resolution (CPU and CUDA use the same order and messages) --------------------------------------------
def resolve_flags(method: str, flags: int, scalar_nonfinite: tuple[bool, ...], residual_failed: bool, cfl_value: float,
                  cfl_max: float, limiter: str) -> None:
    """Raise the FIRST failure: state-input errors (local inertia), the recoverable CFL rejection, non-finite output, negative
    depth (a rejection when the limiter is off, otherwise an error), the per-cell balance, the non-finite scalars, the global
    balance. Returns None if nothing failed."""
    bits = EXPLICIT_BITS if method == "explicit" else LOCAL_BITS
    if method == "local_inertial":
        if flags & bits["state_nonfinite"]:
            raise ExperimentalHydrologyError("state face fluxes must be finite everywhere")
        if flags & bits["closed_face"]:
            raise ExperimentalHydrologyError("state face flux must be 0 on closed faces (inactive or unopened boundary)")
        if flags & bits["open_inflow"]:
            raise ExperimentalHydrologyError("state face flux on an open outlet face must not point into the domain")
    if flags & bits["cfl"]:
        raise HydraulicStepRejected(f"{CFL_NAMES[method]} = {cfl_value} exceeds cfl_max {cfl_max}; step rejected "
                                    "(retry with a smaller dt)")
    if flags & bits["nonfinite"]:
        raise ExperimentalHydrologyError("step produced non-finite depth, flux, velocity or balance scale")
    if flags & bits["negative"]:
        text = "step produced negative depth"
        if method == "local_inertial" and limiter == "off":
            raise HydraulicStepRejected(f"{text} (water leaving a cell exceeds what it holds); step rejected "
                                        "(retry with a smaller dt, or use limiter='donor')")
        raise ExperimentalHydrologyError(f"{text}; nothing is clipped")
    if flags & bits["balance"]:
        raise ExperimentalHydrologyError("per-cell water balance violated beyond FP64 tolerance")
    for name, bad in zip(_SCALAR_NAMES, scalar_nonfinite, strict=True):
        if bad:
            raise ExperimentalHydrologyError(f"step produced non-finite {name} (volume overflow)")
    if residual_failed:
        raise ExperimentalHydrologyError("global water balance (storage change + export) violated beyond FP64 tolerance")


# --- static geometry ------------------------------------------------------------------------------------------------
def donor_cells_by_cell(graph: RoutingGraph) -> np.ndarray:
    """`(4, n_cells)` int64: flat cell index of the donor in `DONOR_SLOTS` order (the legacy `sdirin` order), -1 = none.
    Built from the HOST receiver/active arrays exactly as the graph builder does."""
    ny, nx = graph.shape
    receiver = np.asarray(graph.receiver).reshape(-1)
    active = np.asarray(graph.active).reshape(-1)
    rows, cols = (a.reshape(-1) for a in np.indices((ny, nx)))
    own = rows * nx + cols
    out = np.full((4, ny * nx), -1, dtype=np.int64)
    for s, (dr, dc) in enumerate(DONOR_SLOTS):
        sr, sc = rows + dr, cols + dc
        ok = (sr >= 0) & (sr < ny) & (sc >= 0) & (sc < nx)
        src = np.clip(sr, 0, ny - 1) * nx + np.clip(sc, 0, nx - 1)
        is_donor = active & ok & active[src] & (receiver[src] == own)
        out[s] = np.where(is_donor, src, -1)
    return out


def open_faces_from_graph(graph: RoutingGraph) -> list[tuple[int, int, int, int]]:
    """`(row, col, d_row, d_col)` of every legacy outlet cell and the direction of its outlet face (from the legacy aspect).
    An outlet whose receiver is an interior (inactive, export-flagged) cell is an interior open hole: refused."""
    ny, nx = graph.shape
    outlet, aspect = np.asarray(graph.outlet), np.asarray(graph.aspect)
    faces = []
    for r, c in zip(*np.nonzero(outlet), strict=True):
        code = int(aspect[r, c])
        if code not in ASPECT_STEPS:
            raise ExperimentalGeometryError(f"outlet cell (r={r}, c={c}) has no flow direction")
        dr, dc = ASPECT_STEPS[code]
        tr, tc = int(r) + dr, int(c) + dc
        if 0 <= tr < ny and 0 <= tc < nx:
            raise ExperimentalGeometryError(
                f"outlet cell (r={r}, c={c}) drains into the interior cell (r={tr}, c={tc}); interior open boundaries are "
                "unsupported (only faces on the outer ring can be opened)")
        faces.append((int(r), int(c), dr, dc))
    return faces


@dataclass(frozen=True, eq=False)
class LocalInertialGeometry:
    """Read-only host description of the local-inertial grid: bed `z (ny, nx)`, `active`, cell friction, and per-face static
    data for the x faces `(ny, nx+1)` and y faces `(ny+1, nx)`: `type` (0 closed, 1 interior active-active, 2 open outlet),
    `zmax` (max bed of the two sides, type 1), `fric` (face friction factor, type 1; donor friction, type 2), `kb` (normal-flow
    k, type 2), `sign` (+1/-1 outward direction, type 2, else 0)."""

    shape: tuple[int, int]
    dx_m: float
    z: np.ndarray
    active: np.ndarray
    friction: np.ndarray
    fx_type: np.ndarray
    fx_zmax: np.ndarray
    fx_fric: np.ndarray
    fx_kb: np.ndarray
    fx_sign: np.ndarray
    fy_type: np.ndarray
    fy_zmax: np.ndarray
    fy_fric: np.ndarray
    fy_kb: np.ndarray
    fy_sign: np.ndarray
    open_faces: tuple
    input_sha256: str

    ARRAYS = ("z", "active", "friction", "fx_type", "fx_zmax", "fx_fric", "fx_kb", "fx_sign", "fy_type", "fy_zmax",
              "fy_fric", "fy_kb", "fy_sign")

    def summary(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dx_m": self.dx_m, "n_open_faces": len(self.open_faces),
                "open_faces": [{"row": r, "col": c, "d_row": dr, "d_col": dc, "bed_drop": s, "k_b": k}
                               for r, c, dr, dc, s, k in self.open_faces],
                "n_interior_faces": int(np.count_nonzero(self.fx_type == 1) + np.count_nonzero(self.fy_type == 1)),
                "input_sha256": self.input_sha256,
                "boundary": "Darcy normal-flow outflow q = k_b h^(3/2) on the legacy outlet faces only; every other ring "
                            "face and every face touching an inactive cell is closed"}


def _frozen(array: np.ndarray) -> np.ndarray:
    out = np.array(array, copy=True)
    out.flags.writeable = False
    return out


def build_local_inertial_geometry(elevation_full_m: np.ndarray, active: np.ndarray, friction_factor: np.ndarray,
                                  dx_m: float, open_faces: list[tuple[int, int, int, int]]) -> LocalInertialGeometry:
    """Validate and build the geometry. `elevation_full_m (ny+2, nx+2)` float64 including the one-cell ring (row 0 = south);
    `active (ny, nx)` bool; `friction_factor (ny, nx)` float64 finite > 0 on active cells; `open_faces` from
    `open_faces_from_graph` (empty = closed domain, used for lake-at-rest/ponding tests). Raises
    `ExperimentalGeometryError` on anything unsupported; nothing is returned then."""
    z_full = np.asarray(elevation_full_m)
    if type(elevation_full_m) is not np.ndarray or z_full.dtype != np.float64 or z_full.ndim != 2 or min(z_full.shape) < 3:
        raise ExperimentalGeometryError("elevation_full_m must be a host float64 (ny+2, nx+2) array with a ring")
    ny, nx = z_full.shape[0] - 2, z_full.shape[1] - 2
    if type(active) is not np.ndarray or active.dtype != np.bool_ or active.shape != (ny, nx):
        raise ExperimentalGeometryError(f"active must be a host bool {(ny, nx)} array")
    if type(friction_factor) is not np.ndarray or friction_factor.dtype != np.float64 or friction_factor.shape != (ny, nx):
        raise ExperimentalGeometryError(f"friction_factor must be a host float64 {(ny, nx)} array")
    dx = _real(dx_m, "dx_m", ExperimentalGeometryError)
    if not (math.isfinite(dx) and dx > 0.0 and math.isfinite(dx * dx) and dx * dx > 0.0):
        raise ExperimentalGeometryError(f"dx_m must be finite with a finite positive cell area, got {dx_m!r}")
    if not np.all(np.isfinite(z_full)):
        raise ExperimentalGeometryError("elevation_full_m must be finite everywhere, ring included")
    if not active.any():
        raise ExperimentalGeometryError("no active cell")
    if not (np.all(np.isfinite(friction_factor[active])) and np.all(friction_factor[active] > 0.0)):
        raise ExperimentalGeometryError("friction_factor must be finite and > 0 on every active cell")
    z = np.array(z_full[1:-1, 1:-1])
    f = np.where(active, friction_factor, 1.0)

    fx_type = np.zeros((ny, nx + 1), dtype=np.int8)
    fy_type = np.zeros((ny + 1, nx), dtype=np.int8)
    fx_zmax, fx_fric, fx_kb = (np.zeros((ny, nx + 1)) for _ in range(3))
    fy_zmax, fy_fric, fy_kb = (np.zeros((ny + 1, nx)) for _ in range(3))
    fx_sign = np.zeros((ny, nx + 1), dtype=np.int8)
    fy_sign = np.zeros((ny + 1, nx), dtype=np.int8)
    both_x = active[:, :-1] & active[:, 1:]
    fx_type[:, 1:-1] = np.where(both_x, 1, 0)
    fx_zmax[:, 1:-1] = np.where(both_x, np.maximum(z[:, :-1], z[:, 1:]), 0.0)
    fx_fric[:, 1:-1] = np.where(both_x, 0.5 * (f[:, :-1] + f[:, 1:]), 0.0)
    both_y = active[:-1, :] & active[1:, :]
    fy_type[1:-1, :] = np.where(both_y, 1, 0)
    fy_zmax[1:-1, :] = np.where(both_y, np.maximum(z[:-1, :], z[1:, :]), 0.0)
    fy_fric[1:-1, :] = np.where(both_y, 0.5 * (f[:-1, :] + f[1:, :]), 0.0)

    described = []
    seen = set()
    for r, c, dr, dc in open_faces:
        if not (0 <= r < ny and 0 <= c < nx) or not active[r, c]:
            raise ExperimentalGeometryError(f"open face of ({r}, {c}): not an active interior cell")
        if abs(dr) + abs(dc) != 1:
            raise ExperimentalGeometryError("an open face must be one of the four D4 directions")
        tr, tc = r + dr, c + dc
        if 0 <= tr < ny and 0 <= tc < nx:
            raise ExperimentalGeometryError(f"open face of ({r}, {c}) leads into the interior; interior open holes are "
                                            "unsupported")
        drop = (z_full[r + 1, c + 1] - z_full[tr + 1, tc + 1]) / dx  # bed normal drop to the ring cell (m/m)
        if not drop > 0.0:
            raise ExperimentalGeometryError(f"open face of ({r}, {c}) has no positive bed drop to the ring ({drop}); a "
                                            "normal-flow outlet needs one")
        kb = float(np.sqrt(8.0 * GRAVITY_M_S2 * drop / f[r, c]))
        if dc != 0:
            j, sign = (c + 1, 1) if dc > 0 else (c, -1)
            axis, index, (ftype, ffric, fkb, fsign) = "x", (r, j), (fx_type, fx_fric, fx_kb, fx_sign)
        else:
            i, sign = (r + 1, 1) if dr > 0 else (r, -1)
            axis, index, (ftype, ffric, fkb, fsign) = "y", (i, c), (fy_type, fy_fric, fy_kb, fy_sign)
        if (axis, index) in seen:
            raise ExperimentalGeometryError(f"face {axis}{index} opened twice")
        seen.add((axis, index))
        ftype[index], ffric[index], fkb[index], fsign[index] = 2, f[r, c], kb, sign
        described.append((int(r), int(c), int(dr), int(dc), float(drop), kb))

    digest = hashlib.sha256(b"maple_syrup.experimental_geometry.v1")
    for array in (z_full, active, friction_factor):
        digest.update(f"{array.dtype}{array.shape}".encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(repr((dx, sorted(described))).encode())
    return LocalInertialGeometry(
        shape=(ny, nx), dx_m=dx, z=_frozen(z), active=_frozen(active), friction=_frozen(friction_factor),
        fx_type=_frozen(fx_type), fx_zmax=_frozen(fx_zmax), fx_fric=_frozen(fx_fric), fx_kb=_frozen(fx_kb),
        fx_sign=_frozen(fx_sign), fy_type=_frozen(fy_type), fy_zmax=_frozen(fy_zmax), fy_fric=_frozen(fy_fric),
        fy_kb=_frozen(fy_kb), fy_sign=_frozen(fy_sign), open_faces=tuple(sorted(described)),
        input_sha256=digest.hexdigest())


# --- state validation (xp-generic: NumPy or CuPy arrays) ------------------------------------------------------------
def validate_hydraulic_state(xp: ModuleType, state: Any, *, method: str, shape: tuple[int, int],
                             faces: tuple | None = None) -> None:
    """Structure, finiteness and sign checks of a state in the namespace `xp` (one batched flag read on a device). `faces` =
    `(fx_type, fx_sign, fy_type, fy_sign)` arrays in `xp` for the local-inertial method. Raises `ExperimentalHydrologyError`."""
    from maple.core.backend import DeferredChecks, finite_flag, negative_flag, true_flag

    if not isinstance(state, HydraulicState):
        raise ExperimentalHydrologyError(f"state must be a HydraulicState, got {type(state).__name__}")
    ny, nx = shape
    wanted = {"depth_m": (state.depth_m, shape), "soil_water_m": (state.soil_water_m, shape)}
    if method == "local_inertial":
        wanted["qx_m2_s"] = (state.qx_m2_s, (ny, nx + 1))
        wanted["qy_m2_s"] = (state.qy_m2_s, (ny + 1, nx))
    elif state.qx_m2_s is not None or state.qy_m2_s is not None:
        raise ExperimentalHydrologyError("the explicit method has no momentum: qx_m2_s and qy_m2_s must be None")
    for name, (array, expected) in wanted.items():
        if type(array) is not xp.ndarray:
            raise ExperimentalHydrologyError(f"state.{name} must be an exact {xp.__name__}.ndarray, got "
                                             f"{type(array).__name__}")
        if tuple(array.shape) != expected or array.dtype != np.float64:
            raise ExperimentalHydrologyError(f"state.{name} must be float64 with shape {expected}, got {array.dtype} "
                                             f"{tuple(array.shape)}")
    t = _real(state.t_s, "state.t_s")
    if not (math.isfinite(t) and t >= 0.0):
        raise ExperimentalHydrologyError(f"state.t_s must be finite and >= 0, got {t!r}")
    check_dt_cap(getattr(state, "next_dt_cap_s", None))
    checks = DeferredChecks()
    for name in ("depth_m", "soil_water_m"):
        checks.require(finite_flag(wanted[name][0]), f"state.{name} must be finite everywhere")
        checks.forbid(negative_flag(wanted[name][0]), f"state.{name} must be >= 0 everywhere")
    if method == "local_inertial":
        fx_type, fx_sign, fy_type, fy_sign = faces
        qx, qy = state.qx_m2_s, state.qy_m2_s
        checks.require(finite_flag(qx), "state.qx_m2_s must be finite everywhere")
        checks.require(finite_flag(qy), "state.qy_m2_s must be finite everywhere")
        checks.forbid(true_flag((fx_type == 0) & (qx != 0.0)), "state.qx_m2_s must be 0 on closed faces")
        checks.forbid(true_flag((fy_type == 0) & (qy != 0.0)), "state.qy_m2_s must be 0 on closed faces")
        checks.forbid(true_flag((fx_type == 2) & (qx * fx_sign < 0.0)), "state.qx_m2_s points into the domain on an open face")
        checks.forbid(true_flag((fy_type == 2) & (qy * fy_sign < 0.0)), "state.qy_m2_s points into the domain on an open face")
    try:
        checks.resolve()
    except ValueError as exc:
        raise ExperimentalHydrologyError(str(exc)) from None


# --- local-inertial face update (vectorised NumPy; the CUDA kernels evaluate the same expressions) -----------------------
def local_inertial_face_flux(ftype: np.ndarray, zmax: np.ndarray, fric: np.ndarray, kb: np.ndarray, sign: np.ndarray,
                             q_old: np.ndarray, h_a: np.ndarray, h_b: np.ndarray, eta_a: np.ndarray, eta_b: np.ndarray,
                             dt: float, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """New signed unit flux and face depth on a set of faces (A = west/south side, B = east/north side).
    type 1: the semi-implicit Darcy-Weisbach local-inertial update; type 2: the normal-flow outlet; type 0: zero."""
    with np.errstate(all="ignore"):
        top = np.where(eta_a > eta_b, eta_a, eta_b)
        hf1 = np.maximum(top - zmax, 0.0)
        grad = (eta_b - eta_a) / dx
        numerator = q_old - (((GRAVITY_M_S2 * hf1) * dt) * grad)
        friction = np.where(q_old != 0.0, ((dt * (fric / 8.0)) * np.abs(q_old)) / (hf1 * hf1), 0.0)
        q1 = np.where(hf1 > 0.0, numerator / (1.0 + friction), 0.0)
        donor = np.where(sign > 0, h_a, h_b)
        q_out = (np.sqrt(donor) * donor) * kb
        q2 = np.where(sign > 0, q_out, -q_out)
        q = np.where(ftype == 1, q1, np.where(ftype == 2, q2, 0.0))
        hf = np.where(ftype == 1, hf1, np.where(ftype == 2, donor, 0.0))
    return q, hf


# --- CPU reference solver --------------------------------------------------------------------------------------------
def _zero_pad_cols(a: np.ndarray, left: bool) -> np.ndarray:
    pad = np.zeros((a.shape[0], 1))
    return np.concatenate([pad, a], axis=1) if left else np.concatenate([a, pad], axis=1)


def _zero_pad_rows(a: np.ndarray, below: bool) -> np.ndarray:
    pad = np.zeros((1, a.shape[1]))
    return np.concatenate([pad, a], axis=0) if below else np.concatenate([a, pad], axis=0)


class CpuHydraulicSolver:
    """Pure NumPy reference of both candidates (the oracle of the CUDA forms). `step(rain, state, dt)` is pure: fresh outputs,
    raises before a result exists, never writes an input; `HydraulicStepRejected` means "retry the same state with a smaller
    dt". The column physics is the accepted `infiltration.column_step` (validated) evaluated on the state first."""

    implementation = "numpy"
    xp = np

    def __init__(self, method: str, graph: RoutingGraph, params: ColumnParameters, *,
                 geometry: LocalInertialGeometry | None = None, control: HydraulicControl | None = None):
        if method not in METHODS:
            raise ExperimentalHydrologyError(f"method must be one of {METHODS}, got {method!r}")
        if not isinstance(graph, RoutingGraph) or not isinstance(params, ColumnParameters):
            raise ExperimentalHydrologyError("graph must be a RoutingGraph and params a ColumnParameters")
        if graph.xp is not np or params.xp is not np:
            raise ExperimentalHydrologyError("the CPU reference needs a NumPy graph and NumPy parameters (no transfer)")
        control = (HydraulicControl() if control is None else control).validated()
        if method == "explicit" and control.limiter != "off":
            raise ExperimentalHydrologyError("the donor limiter belongs to the local-inertial method; the explicit "
                                             "method is positive by its CFL bound")
        if tuple(params.shape) != tuple(graph.shape) or not np.array_equal(params.active_mask, graph.active):
            raise ExperimentalHydrologyError("column parameters must match the graph shape and active mask")
        self.method, self.graph, self.params, self.control = method, graph, params, control
        self.shape = tuple(graph.shape)
        self.dx_m = float(graph.dx_m)
        self.active = np.asarray(graph.active)
        self.geometry = None
        if method == "explicit":
            self.k = np.asarray(graph.conveyance).reshape(self.shape)
            self.outlet = np.asarray(graph.outlet)
            self.donors = donor_cells_by_cell(graph)
        else:
            if not isinstance(geometry, LocalInertialGeometry):
                raise ExperimentalHydrologyError("the local-inertial method needs a LocalInertialGeometry")
            if (geometry.shape != self.shape or geometry.dx_m != self.dx_m or not np.array_equal(geometry.active, self.active)
                    or not np.array_equal(geometry.friction, np.asarray(graph.friction_factor))):
                raise ExperimentalHydrologyError("the geometry does not match the graph (shape, dx, active mask, friction)")
            self.geometry = geometry

    # -- state ------------------------------------------------------------------------------------------------------
    def initial_state(self, depth_m: Any, soil_water_m: Any, *, t_s: float = 0.0) -> HydraulicState:
        """Validated copy of a state at rest (zero face momentum for local inertia)."""
        for name, array in (("depth_m", depth_m), ("soil_water_m", soil_water_m)):
            if type(array) is not np.ndarray or array.dtype != np.float64 or tuple(array.shape) != self.shape:
                raise ExperimentalHydrologyError(f"{name} must be a host float64 array with shape {self.shape}")
        ny, nx = self.shape
        faces = (None, None) if self.method == "explicit" else (np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx)))
        state = HydraulicState(check_state_time(t_s, "t_s"), np.array(depth_m, copy=True), np.array(soil_water_m, copy=True), *faces)
        self.validate_state(state)
        return state

    def _faces(self):
        g = self.geometry
        return None if g is None else (g.fx_type, g.fx_sign, g.fy_type, g.fy_sign)

    def validate_state(self, state: HydraulicState) -> None:
        validate_hydraulic_state(np, state, method=self.method, shape=self.shape, faces=self._faces())

    def describe(self) -> dict[str, Any]:
        return {"method": self.method, "implementation": self.implementation,
                "control": {"cfl_max": self.control.cfl_max, "limiter": self.control.limiter},
                "cfl_kind": CFL_NAMES[self.method], "geometry": None if self.geometry is None else self.geometry.summary(),
                "transfer_scope": TRANSFER_SCOPE, "qualification": QUALIFICATION_STATUS}

    # -- one step ---------------------------------------------------------------------------------------------------
    def step(self, rain_rate_m_per_s: Any, state: HydraulicState, dt_s: float) -> HydraulicStep:
        if not isinstance(state, HydraulicState):
            raise ExperimentalHydrologyError(f"state must be a HydraulicState, got {type(state).__name__}")
        # strict time contract (identical to the CUDA solver): finite non-negative real state time, then for dt > 0 a finite,
        # advancing end time; dt itself is checked by the accepted column stage with its own error class
        t_state = check_state_time(state.t_s, "state.t_s")
        check_dt_cap(state.next_dt_cap_s)  # continuation metadata is never read by a step, but a malformed one is refused
        dt_checked = _check_dt(dt_s)
        t_new = advance_time(t_state, dt_checked) if dt_checked > 0.0 else None
        if self.method == "explicit" and (state.qx_m2_s is not None or state.qy_m2_s is not None):
            raise ExperimentalHydrologyError("the explicit method has no momentum: qx_m2_s and qy_m2_s must be None")
        if self.method == "local_inertial":
            ny, nx = self.shape
            for name, array, expected in (("qx_m2_s", state.qx_m2_s, (ny, nx + 1)), ("qy_m2_s", state.qy_m2_s, (ny + 1, nx))):
                if type(array) is not np.ndarray or array.dtype != np.float64 or tuple(array.shape) != expected:
                    raise ExperimentalHydrologyError(f"state.{name} must be a host float64 array with shape {expected}")
        col = self._column(rain_rate_m_per_s, state, dt_s)  # accepted column physics (hook: see `_column`)
        dt = dt_checked
        if not dt > 0.0:
            raise ExperimentalHydrologyError(f"dt_s must be finite and > 0 (dt = 0 is rejected, not an identity), got {dt_s!r}")
        if self.method == "explicit":
            return self._explicit(col, state, dt, t_new)
        return self._local_inertial(col, state, dt, t_new)

    def _column(self, rain_rate_m_per_s: Any, state: HydraulicState, dt_s: float) -> ColumnStep:
        """The accepted column stage on the state. A SUBCLASS hook (the compiled `experimental_numba.NumbaHydraulicSolver`
        overrides it, `_explicit` and `_local_inertial` and inherits every public check of `step`, `initial_state` and
        `validate_state`); the NumPy arithmetic of this class is unchanged."""
        return column_step(self.params, state.depth_m, state.soil_water_m, rain_rate_m_per_s, dt_s)

    def _explicit(self, col: ColumnStep, state: HydraulicState, dt: float, t_new: float) -> HydraulicStep:
        shape, dx = self.shape, self.dx_m
        area, dtdx_area, dt_over_dx = dx * dx, dt * dx, dt / dx
        active, outlet, k = self.active, self.outlet, self.k
        hc = col.depth_m
        with np.errstate(all="ignore"):
            sqrt_h = np.sqrt(hc)
            q_used = np.where(active, (sqrt_h * hc) * k, 0.0)
            cfl_cell = np.where(active, ((1.5 * k) * sqrt_h) * dt_over_dx, 0.0)
            out_vol = np.where(active, dtdx_area * q_used, 0.0)
            flat = out_vol.reshape(-1)
            idx = np.where(self.donors >= 0, self.donors, 0)
            total = np.where(self.donors[0] >= 0, flat[idx[0]], 0.0)
            for s in (1, 2, 3):
                total = total + np.where(self.donors[s] >= 0, flat[idx[s]], 0.0)
            inflow = total.reshape(shape)
            h_new = hc + ((inflow - out_vol) / area)
            q_inst = np.where(active, (np.sqrt(h_new) * h_new) * k, 0.0)
            wet = h_new > 0.0
            velocity = np.where(wet, q_inst / np.where(wet, h_new, 1.0), 0.0)
            balance = np.where(active, np.abs((h_new - hc) - ((inflow - out_vol) / area)), 0.0)
            scale = np.where(active, (hc + h_new) + ((inflow + out_vol) / area), 0.0)
            storage = area * np.sum(h_new - hc)
            export = np.sum(np.where(outlet, out_vol, 0.0))
            residual = storage + export
            gtol = BALANCE_RTOL * (int(np.count_nonzero(active)) + 2) * area * np.sum(scale)
            outlet_q = dx * np.sum(np.where(outlet, q_inst, 0.0))
            max_cfl = float(np.max(cfl_cell))
        flags = 0
        if max_cfl > self.control.cfl_max:
            flags |= EXPLICIT_BITS["cfl"]
        if not all(np.all(np.isfinite(a)) for a in (h_new, out_vol, q_used, q_inst, velocity, scale, balance)):
            flags |= EXPLICIT_BITS["nonfinite"]
        if np.any(h_new < 0.0):
            flags |= EXPLICIT_BITS["negative"]
        if np.any(balance > BALANCE_RTOL * scale):
            flags |= EXPLICIT_BITS["balance"]
        scalars = tuple(not math.isfinite(float(v)) for v in (storage, export, residual, gtol, outlet_q))
        resolve_flags("explicit", flags, scalars, bool(abs(residual) > gtol), max_cfl, self.control.cfl_max, "off")
        new_state = HydraulicState(t_new, h_new, col.soil_water_m)
        return HydraulicStep(
            method="explicit", implementation=self.implementation, dt_s=dt, state=new_state, column=col,
            velocity_m_s=velocity, face_volume_m3={"out": out_vol}, used_flux_m2_s={"out": q_used},
            export_m3=export, outlet_discharge_m3_s=outlet_q, storage_change_m3=storage, budget_residual_m3=residual,
            max_cfl=max_cfl, max_cell_balance_residual_m=float(np.max(balance)), limited_cells=0, limited_volume_m3=0.0,
            cfl_kind=CFL_NAMES["explicit"], face_flow_depth_m={"out": hc})

    def _local_inertial(self, col: ColumnStep, state: HydraulicState, dt: float, t_new: float) -> HydraulicStep:
        g = self.geometry
        ny, nx = self.shape
        dx = self.dx_m
        area, dtdx_area, dt_over_dx = dx * dx, dt * dx, dt / dx
        active = self.active
        hc = col.depth_m
        qx_old, qy_old = state.qx_m2_s, state.qy_m2_s
        flags = 0
        if not (np.all(np.isfinite(qx_old)) and np.all(np.isfinite(qy_old))):
            flags |= LOCAL_BITS["state_nonfinite"]
        if np.any((g.fx_type == 0) & (qx_old != 0.0)) or np.any((g.fy_type == 0) & (qy_old != 0.0)):
            flags |= LOCAL_BITS["closed_face"]
        if np.any((g.fx_type == 2) & (qx_old * g.fx_sign < 0.0)) or np.any((g.fy_type == 2) & (qy_old * g.fy_sign < 0.0)):
            flags |= LOCAL_BITS["open_inflow"]
        eta = g.z + hc
        qx, hfx = local_inertial_face_flux(
            g.fx_type, g.fx_zmax, g.fx_fric, g.fx_kb, g.fx_sign, qx_old, _zero_pad_cols(hc, True), _zero_pad_cols(hc, False),
            _zero_pad_cols(eta, True), _zero_pad_cols(eta, False), dt, dx)
        qy, hfy = local_inertial_face_flux(
            g.fy_type, g.fy_zmax, g.fy_fric, g.fy_kb, g.fy_sign, qy_old, _zero_pad_rows(hc, True), _zero_pad_rows(hc, False),
            _zero_pad_rows(eta, True), _zero_pad_rows(eta, False), dt, dx)
        limited_cells, limited_volume = 0, 0.0
        with np.errstate(all="ignore"):
            if self.control.limiter == "donor":
                west, east, south, north = qx[:, :-1], qx[:, 1:], qy[:-1, :], qy[1:, :]
                out = ((np.where(-west > 0.0, -west, 0.0) + np.where(east > 0.0, east, 0.0))
                       + (np.where(-south > 0.0, -south, 0.0) + np.where(north > 0.0, north, 0.0)))
                out_v = dtdx_area * out
                avail = hc * area
                limited = out_v > avail
                phi = np.where(limited, (avail / np.where(limited, out_v, 1.0)) * LIMITER_SAFETY, 1.0)
                ones_c, ones_r = np.ones((ny, 1)), np.ones((1, nx))
                phi_a_x, phi_b_x = np.concatenate([ones_c, phi], axis=1), np.concatenate([phi, ones_c], axis=1)
                phi_a_y, phi_b_y = np.concatenate([ones_r, phi], axis=0), np.concatenate([phi, ones_r], axis=0)
                qx = np.where(qx > 0.0, qx * phi_a_x, np.where(qx < 0.0, qx * phi_b_x, qx))
                qy = np.where(qy > 0.0, qy * phi_a_y, np.where(qy < 0.0, qy * phi_b_y, qy))
                limited_cells = int(np.count_nonzero(limited))
                limited_volume = float(np.sum(np.where(limited, out_v * (1.0 - phi), 0.0)))
            west, east, south, north = qx[:, :-1], qx[:, 1:], qy[:-1, :], qy[1:, :]
            div = (east - west) + (north - south)
            h_new = hc - (dt_over_dx * div)
            wet = h_new > 0.0
            safe = np.where(wet, h_new, 1.0)
            ux = np.where(wet, (0.5 * (west + east)) / safe, 0.0)
            uy = np.where(wet, (0.5 * (south + north)) / safe, 0.0)
            speed = np.sqrt(ux * ux + uy * uy)
            balance = np.where(active, np.abs((h_new - hc) + (dt_over_dx * div)), 0.0)
            scale = np.where(active, (hc + h_new) + dt_over_dx * ((np.abs(west) + np.abs(east))
                                                                  + (np.abs(south) + np.abs(north))), 0.0)
            storage = area * np.sum(h_new - hc)
            out_x = np.where(g.fx_type == 2, qx * g.fx_sign, 0.0)
            out_y = np.where(g.fy_type == 2, qy * g.fy_sign, 0.0)
            outward = np.sum(out_x) + np.sum(out_y)
            export = dtdx_area * outward
            residual = storage + export
            gtol = BALANCE_RTOL * (int(np.count_nonzero(active)) + 2) * area * np.sum(scale)
            outlet_q = dx * outward
            hf_max = max(float(np.max(hfx)), float(np.max(hfy)))
            max_cfl = float(dt_over_dx * np.sqrt(2.0 * GRAVITY_M_S2 * hf_max))
        if max_cfl > self.control.cfl_max:
            flags |= LOCAL_BITS["cfl"]
        if np.any(h_new < 0.0):
            flags |= LOCAL_BITS["negative"]
        if not all(np.all(np.isfinite(a)) for a in (qx, qy, h_new, speed, scale, balance)):
            flags |= LOCAL_BITS["nonfinite"]
        if np.any(balance > BALANCE_RTOL * scale):
            flags |= LOCAL_BITS["balance"]
        scalars = tuple(not math.isfinite(float(v)) for v in (storage, export, residual, gtol, outlet_q))
        resolve_flags("local_inertial", flags, scalars, bool(abs(residual) > gtol), max_cfl, self.control.cfl_max,
                      self.control.limiter)
        new_state = HydraulicState(t_new, h_new, col.soil_water_m, qx, qy)
        return HydraulicStep(
            method="local_inertial", implementation=self.implementation, dt_s=dt, state=new_state, column=col,
            velocity_m_s=speed, face_volume_m3={"x": dtdx_area * qx, "y": dtdx_area * qy},
            used_flux_m2_s={"x": qx, "y": qy}, export_m3=export, outlet_discharge_m3_s=outlet_q, storage_change_m3=storage,
            budget_residual_m3=residual, max_cfl=max_cfl, max_cell_balance_residual_m=float(np.max(balance)),
            limited_cells=limited_cells, limited_volume_m3=limited_volume, cfl_kind=CFL_NAMES["local_inertial"],
            face_flow_depth_m={"x": hfx, "y": hfy})
