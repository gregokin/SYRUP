"""POND HYDRAULIC FOUNDATION, stage 1 -- thin wrappers over the EXISTING experimental local-inertial solvers (NumPy reference,
optional compiled Numba, CUDA), a pure single-step API, an explicitly bounded retry/evolve helper, an exact water ledger and
stage-consistent face-speed diagnostics. Topology/outlets/escape levels are `pond_network`. This is a bounded hydraulics and
topology stage: NO sediment, splash, erosion, restart/checkpoint or full-Chastre claim is made (stage 2). Nothing here was run
by its author (file-only tools); Codex records CPU and GPU results.

REUSE VERSUS NEW. Reused unchanged (imported, never copied): `experimental_hydrology.CpuHydraulicSolver` /
`HydraulicState` / `HydraulicStep` / `HydraulicControl` / `stage_face_diagnostics` / `advance_time` / `check_dt_cap` and the
geometry builder (through `pond_network`); `experimental_numba.NumbaHydraulicSolver`; `experimental_cuda.CudaHydraulicSolver`;
the accepted column physics (`infiltration.column_step` on the CPU, `hydrology_cuda.prepared_column_step` on the GPU, called
by those solvers); `experimental_storm.ExperimentalControl` (driver guards) and the same adaptive-cap rule as
`evolve_experimental`; `conservation.volume_roundoff_bound_m3` for every tolerance. NEW here: the wrapper object, the carry of
`next_dt_cap_s` through a single step, the bounded single-interval evolve, the per-cell/global ledger with directional face
volumes, the outflow-volume-weighted cell speed and the topology rebinding. No hydraulic or infiltration equation is
re-implemented.

PHYSICS (inherited, see `experimental_hydrology`). Local-inertial, advection dropped, semi-implicit Darcy-Weisbach friction
(NOT Manning), gravity-wave CFL `dt sqrt(2 g h_face,max) / dx <= cfl_max (<= 0.5)`, negative depth with limiter "off" is a
REJECTION (never clipped), no minimum slope, no depth clipping, no widened tolerance. This is an explicit ALTERNATIVE to the
accepted method-5 / option-6 routing, NOT a numerical identity with it, and its velocity qualification is unchanged
(`QUALIFICATION_STATUS`): face Froude numbers are reported, never clipped. The donor limiter is REFUSED here (a documented
dynamics change that is not part of this stage).

ADAPTATIONS FROM THE LOCAL-INERTIAL ASSUMPTIONS (all of them). (1) The solver's `RoutingGraph` argument is satisfied by a
METADATA graph (or a caller's native graph): the local-inertial path reads only its shape, dx, active mask and friction; it is
never a flow-direction authority and its conveyance/outlet/receiver/level arrays are unused. Without a caller graph the
metadata graph is built from the held bed with every non-active cell raised to `max(active z) + 1` (no legacy outlet, no edge
rule, no tie to an exterior cell) and `allow_pit_storage`/`allow_flat_storage` so flats and pits are accepted; that elevated
surround exists only inside the metadata graph. CONSTRAINT FROM REUSE: the legacy graph builder requires friction >= 0.1
(`LEGACY_FRICTION_FLOOR`) on active cells, so the same is required here even though the geometry only needs > 0. (2) Outlets
are explicit (see `pond_network`); the legacy `graph.outlet` is ignored. (3) The bed in a step is the HELD routing source
`topology.geometry.z`; it changes only through an explicit refresh.

STEP PURITY. `step_once` never writes its inputs, raises before a result exists, and a `HydraulicStepRejected` leaves the
caller's state untouched (retry the SAME state with a smaller dt). `next_dt_cap_s` (and `qx`, `qy`) are retained in the
returned state: the cap is carried through unchanged by a single step (the solvers return None; `step_once` restores the
caller's value) and is updated only by `evolve_pond`.

EVOLVE. `evolve_pond` advances ONE interval with a rain-rate field held constant over it (split at forcing knots yourself; the
rate is never averaged across a discontinuity). Each interval is covered by accepted steps of at most `control.max_dt_s`;
every accepted step runs the accepted column physics with its own dt, so rainfall and infiltration are integrated exactly as
in the solvers (`sum rate*dt`). A rejection halves dt from the unchanged state, bounded by `max_retries` per step, the retry
floor `min_dt_s` and `max_steps`; when a bound is reached the helper STOPS and returns the last accepted state with
`reached_target=False` and a `stop_reason` (no hidden unbounded microsteps, no exception). The result reports the ACTUAL elapsed
time. The adaptive cap rule is that of `evolve_experimental` (cap shrinks to the dt that worked after a rejection, doubles back
toward `max_dt_s` after clean full-size steps) and is carried in `state.next_dt_cap_s`.

LEDGER (`PondLedger`). Per cell and globally, in m3 (area `dx^2`):
    surface: area*(h_end - h_start) = area*(sum rain - sum intake + sum saturation_return) + net face inflow
    soil:    area*(S_end - S_start) = area*(sum intake - sum saturation_return - sum drainage)
    global:  sum(surface + soil change) + drainage + export = rain
Face volumes `V = dt dx q'` (signed, +east/+north) are accumulated per x/y face as directional parts `pos = sum max(V,0)`,
`neg = sum max(-V,0)` and the signed `net = sum V`; gross = pos + neg. All tolerances are
`conservation.volume_roundoff_bound_m3(n_terms, largest operand)` with `n_terms = 4 n_cells n_steps + 7` globally (the existing
event rule) and `16 n_steps + 7` per cell; there is no absolute or mass floor.

FACE-SPEED DIAGNOSTICS (`face_speed_diagnostics`). Face velocity and Froude number come from
`stage_face_diagnostics`: `u = q / h_face` with the stage depth that flux was computed from (dry face: 0, no drying cell-centre
velocity), `Fr = |u| / sqrt(g h_face)`, neither clipped. The cell speed for future erosion coupling is the OUTFLOW-VOLUME-weighted
mean `sum_k V_out,k |u_k| / sum_k V_out,k` over the cell's faces whose signed volume leaves the cell (outlet faces included),
exactly 0 when no volume leaves; being a convex combination it is bounded by the largest face speed of that cell (up to
rounding). SLOPE CAVEAT FOR FUTURE WET LAWS: the topographic bed slope that governs rain impact/splash is NOT the hydraulic
(friction-driving) slope under unsteady or flat flow; shear must use the water-surface/friction slope of the flow, rain/splash
the bed slope. This stage replaces neither by the other and supplies only face speed and Froude.

REFRESH. `rebind_pond(pond, new_topology, state)` builds a solver on a refreshed topology (same identity) and validates the
CARRIED state (`qx, qy`, depth, soil water, `next_dt_cap_s`) against the new faces; the state object is returned unchanged
because a refresh keeps the face types and the open-face signs (only `zmax`, `k_b`, `z` change). The caller must have
committed the native bed before adopting a refresh.

Units: depths m, unit discharge m2/s, volumes m3, speeds m/s, time s, FP64. Arrays live in the solver's namespace (NumPy or
CuPy); host reads happen only in `closure()` and the statistics of `face_speed_diagnostics`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.experimental_hydrology import (
    CFL_NAMES,
    QUALIFICATION_STATUS,
    CpuHydraulicSolver,
    ExperimentalHydrologyError,
    HydraulicControl,
    HydraulicState,
    HydraulicStep,
    HydraulicStepRejected,
    advance_time,
    check_dt_cap,
    check_state_time,
    stage_face_diagnostics,
)
from maple_syrup.experimental_storm import ExperimentalControl
from maple_syrup.infiltration import ColumnParameters
from maple_syrup.pond_network import PondTopology
from maple_syrup.routing import GRAVITY_M_S2, LEGACY_FRICTION_FLOOR, RoutingGraph, build_routing_graph

__all__ = [
    "BACKENDS",
    "POND_DEFAULT_FRICTION_TREATMENT",
    "SLOPE_NOTE",
    "LedgerClosure",
    "PondEvolveResult",
    "PondFaceDiagnostics",
    "PondHydraulics",
    "PondHydraulicsError",
    "PondLedger",
    "PondStep",
    "build_pond_hydraulics",
    "evolve_pond",
    "face_speed_diagnostics",
    "metadata_routing_graph",
    "rebind_pond",
    "step_once",
    "uniform_rain_rate",
]

BACKENDS = ("numpy", "numba", "cuda")
SLOPE_NOTE = (
    "face speed/Froude are stage-consistent flow quantities; the topographic bed slope (rain impact, splash) is not the hydraulic "
    "friction-driving slope under unsteady or flat flow, and this stage substitutes neither for the other")
_REJECTIONS_KEPT = 100
# Pond default when NO control is passed (the solvers' own `HydraulicControl()` default stays "lagged" for old experiments):
# the lagged friction gives a frictionless impulsive first flux on a wetting face (see `experimental_hydrology`).
POND_DEFAULT_FRICTION_TREATMENT = "implicit_quadratic"


class PondHydraulicsError(ValueError):
    """Invalid wrapper request (backend, parameters, graph, state, control). Raised before a result exists. A recoverable
    rejection is `HydraulicStepRejected`, never this class."""


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise PondHydraulicsError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _to_host(array: Any) -> np.ndarray:
    return np.asarray(array.get()) if hasattr(array, "get") else np.asarray(array)


def _namespace(backend: str):
    if backend not in BACKENDS:
        raise PondHydraulicsError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend != "cuda":
        return np
    from maple.core.backend import cupy_module

    cp = cupy_module()
    if cp is None:
        raise PondHydraulicsError("backend 'cuda' needs CuPy and a CUDA device; there is no fallback to the CPU")
    return cp


# --- legacy graph: incidental metadata only ---------------------------------------------------------------------------
def metadata_routing_graph(topology: PondTopology, xp: Any = np) -> RoutingGraph:
    """Legacy `RoutingGraph` carrying only the metadata the local-inertial solver reads (shape, dx, active mask, friction).
    NOT a flow-direction authority. Every non-active cell and the ring is raised to `max(active z) + 1` so the legacy builder
    sees no exterior receiver and no edge rule; flats and pits are accepted (`allow_pit_storage`, `allow_flat_storage`)."""
    active = np.asarray(topology.active)
    if np.any(topology.friction[active] < LEGACY_FRICTION_FLOOR):
        raise PondHydraulicsError(
            f"the reused legacy graph builder needs friction >= {LEGACY_FRICTION_FLOOR} on every active cell; pass friction "
            "inside that supported range (the pond geometry itself needs only > 0)")
    ny, nx = topology.shape
    top = float(np.max(topology.source_bed_m[active])) + 1.0
    full = np.full((ny + 2, nx + 2), top)
    full[1:-1, 1:-1] = np.where(active, topology.source_bed_m, top)
    export = np.zeros(full.shape, dtype=np.bool_)
    export[0, :] = export[-1, :] = export[:, 0] = export[:, -1] = True
    return build_routing_graph(full, export, np.array(topology.friction), topology.dx_m, active_mask=np.array(active),
                               xp=xp, allow_pit_storage=True, allow_flat_storage=True)


def _check_native_graph(topology: PondTopology, graph: Any, xp: Any) -> None:
    if not isinstance(graph, RoutingGraph):
        raise PondHydraulicsError(f"graph must be a RoutingGraph or None, got {type(graph).__name__}")
    if (tuple(graph.shape) != topology.shape or graph.dx_m != topology.dx_m or graph.xp is not xp
            or not np.array_equal(np.asarray(graph.active), topology.active)
            or not np.array_equal(np.asarray(graph.friction_factor), topology.friction)):
        raise PondHydraulicsError("the graph does not match the topology (shape, dx, namespace, active mask, friction)")


@dataclass(frozen=True, eq=False)
class PondHydraulics:
    """A topology bound to one solver. `graph_kind` is "metadata" (built here) or "caller" (native graph, metadata use only)."""

    topology: PondTopology
    solver: Any
    backend: str
    params: ColumnParameters
    control: HydraulicControl
    graph: RoutingGraph
    graph_kind: str

    @property
    def xp(self):
        return self.solver.xp

    @property
    def shape(self) -> tuple[int, int]:
        return self.topology.shape

    def initial_state(self, depth_m: Any, soil_water_m: Any, *, t_s: float = 0.0) -> HydraulicState:
        """Validated copy at rest (zero `qx, qy`, `next_dt_cap_s` None) in the solver's namespace."""
        return self.solver.initial_state(depth_m, soil_water_m, t_s=t_s)

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend, "graph_kind": self.graph_kind, "solver": self.solver.describe(),
                "friction_treatment": self.control.friction_treatment,
                "topology": self.topology.summary(), "slope_note": SLOPE_NOTE,
                "scope": "hydraulics/topology only; no sediment, restart or full-Chastre claim"}


def build_pond_hydraulics(topology: PondTopology, params: ColumnParameters, *, backend: str = "numpy",
                          control: HydraulicControl | None = None, graph: RoutingGraph | None = None) -> PondHydraulics:
    """Bind `topology` to the existing local-inertial solver of `backend`. `params` must be a `ColumnParameters` in the
    backend's namespace (NumPy for "numpy"/"numba", CuPy for "cuda") with `active_mask` equal to the topology's. `control` is
    the existing `HydraulicControl` (default cfl_max 0.5); `limiter` must be "off". FRICTION: `control=None` selects the NEW pond
    default `friction_treatment="implicit_quadratic"` (`POND_DEFAULT_FRICTION_TREATMENT`); an explicitly provided
    `HydraulicControl()` keeps its own field, i.e. "lagged" unless the caller sets otherwise (never silently replaced). A genuine constructor blocker (an
    unavailable backend, friction < 0.1) raises `PondHydraulicsError` or the solver's own error, never a fallback."""
    if not isinstance(topology, PondTopology):
        raise PondHydraulicsError(f"topology must be a PondTopology, got {type(topology).__name__}")
    xp = _namespace(backend)
    control = HydraulicControl(friction_treatment=POND_DEFAULT_FRICTION_TREATMENT) if control is None else control
    try:
        control = control.validated()
    except ExperimentalHydrologyError as exc:
        raise PondHydraulicsError(str(exc)) from exc
    if control.limiter != "off":
        raise PondHydraulicsError("the donor limiter changes the dynamics and is not part of this stage; use limiter 'off'")
    if not isinstance(params, ColumnParameters) or params.xp is not xp:
        raise PondHydraulicsError(f"params must be ColumnParameters in the {xp.__name__} namespace for backend {backend!r}")
    if tuple(params.shape) != topology.shape or not np.array_equal(_to_host(params.active_mask), topology.active):
        raise PondHydraulicsError("params must match the topology shape and active mask")
    if graph is None:
        graph, kind = metadata_routing_graph(topology, xp), "metadata"
    else:
        _check_native_graph(topology, graph, xp)
        kind = "caller"
    geometry = topology.geometry
    if backend == "numpy":
        solver = CpuHydraulicSolver("local_inertial", graph, params, geometry=geometry, control=control)
    elif backend == "numba":
        from maple_syrup.experimental_numba import NumbaHydraulicSolver

        solver = NumbaHydraulicSolver("local_inertial", graph, params, geometry=geometry, control=control)
    else:
        from maple_syrup.experimental_cuda import CudaHydraulicSolver

        solver = CudaHydraulicSolver("local_inertial", graph, params, geometry=geometry, control=control)
    return PondHydraulics(topology=topology, solver=solver, backend=backend, params=params, control=control, graph=graph,
                          graph_kind=kind)


def rebind_pond(pond: PondHydraulics, new_topology: PondTopology, state: HydraulicState) -> tuple[PondHydraulics, HydraulicState]:
    """Adopt a refreshed topology of the SAME identity. Returns `(new pond, state)` where `state` is the very same object after
    being validated against the new faces (carried `qx, qy`, depth, soil water, `next_dt_cap_s`). Pure: `pond` is unchanged."""
    if not isinstance(pond, PondHydraulics):
        raise PondHydraulicsError("pond must be a PondHydraulics")
    if new_topology.identity_sha256 != pond.topology.identity_sha256:
        raise PondHydraulicsError("a refresh may not change the source mask, friction, dx or outlet faces (identity differs)")
    new = build_pond_hydraulics(new_topology, pond.params, backend=pond.backend, control=pond.control,
                                graph=pond.graph if pond.graph_kind == "caller" else None)
    new.solver.validate_state(state)
    return new, state


def uniform_rain_rate(pond: PondHydraulics, rate_m_per_s: float):
    """`(ny, nx)` float64 rain-rate field in the solver's namespace: `rate` on active cells, 0 elsewhere (m/s)."""
    rate = _real(rate_m_per_s, "rate_m_per_s")
    if not (math.isfinite(rate) and rate >= 0.0):
        raise PondHydraulicsError(f"rate_m_per_s must be finite and >= 0, got {rate_m_per_s!r}")
    return pond.xp.asarray(np.where(pond.topology.active, rate, 0.0))


# --- one pure step -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PondStep:
    """One accepted step: the raw `HydraulicStep` and the successor `state` (that step's state with the caller's
    `next_dt_cap_s` carried). `step.face_volume_m3 = {"x", "y"}` are the signed volumes (m3, +east/+north) that crossed each face."""

    step: HydraulicStep
    state: HydraulicState
    dt_s: float


def step_once(pond: PondHydraulics, rain_rate_m_per_s: Any, state: HydraulicState, dt_s: float) -> PondStep:
    """ONE pure step (rain rate constant over `dt_s`). Raises `HydraulicStepRejected` (state untouched: retry the same state with
    a smaller dt) or another error; never mutates `state`, the rate field or the pond."""
    if not isinstance(pond, PondHydraulics):
        raise PondHydraulicsError("pond must be a PondHydraulics")
    pond.solver.validate_state(state)
    step = pond.solver.step(rain_rate_m_per_s, state, dt_s)
    return PondStep(step=step, state=replace(step.state, next_dt_cap_s=state.next_dt_cap_s), dt_s=step.dt_s)


# --- ledger -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class LedgerClosure:
    """Host numbers of a closed ledger window (m3 unless noted). `*_cell_residual_m3` are `(ny, nx)` host arrays."""

    n_steps: int
    elapsed_s: float
    rain_m3: float
    intake_m3: float
    saturation_return_m3: float
    drainage_m3: float
    export_m3: float
    surface_change_m3: float
    soil_change_m3: float
    global_residual_m3: float
    global_bound_m3: float
    surface_cell_residual_m3: np.ndarray
    soil_cell_residual_m3: np.ndarray
    max_cell_residual_m3: float
    cell_bound_m3: float
    inactive_unchanged: bool
    face_pos_m3: dict
    face_neg_m3: dict
    face_net_m3: dict

    @property
    def closes(self) -> bool:
        return (abs(self.global_residual_m3) <= self.global_bound_m3 and self.max_cell_residual_m3 <= self.cell_bound_m3
                and self.inactive_unchanged)


class PondLedger:
    """Accumulates one window of accepted steps (see the module docstring). Mutable by design: `add` after every accepted
    step, `closure(end_state)` to evaluate. Arrays live in the solver's namespace."""

    def __init__(self, pond: PondHydraulics, state: HydraulicState):
        xp, (ny, nx) = pond.xp, pond.shape
        self.xp, self.shape = xp, (ny, nx)
        self.area = pond.topology.dx_m * pond.topology.dx_m
        self.active = np.asarray(pond.topology.active)
        self.t_start = check_state_time(state.t_s)
        self.depth0, self.soil0 = xp.array(state.depth_m, copy=True), xp.array(state.soil_water_m, copy=True)
        z = lambda s: xp.zeros(s, dtype=np.float64)  # noqa: E731
        self.rain, self.intake, self.sat_return, self.drain = (z((ny, nx)) for _ in range(4))
        self.pos = {"x": z((ny, nx + 1)), "y": z((ny + 1, nx))}
        self.neg = {"x": z((ny, nx + 1)), "y": z((ny + 1, nx))}
        self.net = {"x": z((ny, nx + 1)), "y": z((ny + 1, nx))}
        self.export = xp.zeros((), dtype=np.float64)
        self.n_steps = 0
        self.t_end = self.t_start

    def add(self, accepted: PondStep) -> None:
        xp, col = self.xp, accepted.step.column
        self.rain = self.rain + col.rain_m
        self.intake = self.intake + col.intake_m
        self.sat_return = self.sat_return + col.saturation_return_m
        self.drain = self.drain + col.drainage_m
        for axis in ("x", "y"):
            volume = accepted.step.face_volume_m3[axis]
            self.pos[axis] = self.pos[axis] + xp.maximum(volume, 0.0)
            self.neg[axis] = self.neg[axis] + xp.maximum(-volume, 0.0)
            self.net[axis] = self.net[axis] + volume
        self.export = self.export + accepted.step.export_m3
        self.n_steps += 1
        self.t_end = accepted.state.t_s

    def closure(self, end_state: HydraulicState) -> LedgerClosure:
        xp, area = self.xp, self.area
        h0, s0, h1, s1 = self.depth0, self.soil0, end_state.depth_m, end_state.soil_water_m
        net_x, net_y = self.net["x"], self.net["y"]
        inflow = (net_x[:, :-1] - net_x[:, 1:]) + (net_y[:-1, :] - net_y[1:, :])
        surface = area * (h1 - h0) - (area * ((self.rain - self.intake) + self.sat_return) + inflow)
        soil = area * (s1 - s0) - area * ((self.intake - self.sat_return) - self.drain)
        gross = {a: self.pos[a] + self.neg[a] for a in ("x", "y")}
        gross_cell = (gross["x"][:, :-1] + gross["x"][:, 1:]) + (gross["y"][:-1, :] + gross["y"][1:, :])
        operands = xp.maximum(xp.maximum(xp.maximum(area * h0, area * h1), xp.maximum(area * s0, area * s1)),
                              xp.maximum(xp.maximum(area * self.rain, area * self.intake),
                                         xp.maximum(xp.maximum(area * self.drain, area * self.sat_return), gross_cell)))
        n_steps = max(self.n_steps, 1)
        cell_bound = volume_roundoff_bound_m3(16 * n_steps + 7, float(xp.max(operands)))
        rain, intake = float(area * xp.sum(self.rain)), float(area * xp.sum(self.intake))
        sat_return, drain = float(area * xp.sum(self.sat_return)), float(area * xp.sum(self.drain))
        export = float(self.export)
        surface_change = float(area * xp.sum(h1 - h0))
        soil_change = float(area * xp.sum(s1 - s0))
        initial = float(area * xp.sum(h0 + s0))
        final = float(area * xp.sum(h1 + s1))
        residual = (surface_change + soil_change) + drain + export - rain
        scale = max(abs(initial), abs(final), rain, drain, abs(export))
        bound = volume_roundoff_bound_m3(4 * h0.size * n_steps + 7, scale)
        inactive = ~self.active
        h0_h, h1_h, s0_h, s1_h = (_to_host(a) for a in (h0, h1, s0, s1))
        unchanged = bool(np.array_equal(h0_h[inactive], h1_h[inactive]) and np.array_equal(s0_h[inactive], s1_h[inactive]))
        surface_h, soil_h = _to_host(surface), _to_host(soil)
        return LedgerClosure(
            n_steps=self.n_steps, elapsed_s=float(self.t_end - self.t_start), rain_m3=rain, intake_m3=intake,
            saturation_return_m3=sat_return, drainage_m3=drain, export_m3=export, surface_change_m3=surface_change,
            soil_change_m3=soil_change, global_residual_m3=float(residual), global_bound_m3=bound,
            surface_cell_residual_m3=surface_h, soil_cell_residual_m3=soil_h,
            max_cell_residual_m3=float(max(np.max(np.abs(surface_h)), np.max(np.abs(soil_h)))), cell_bound_m3=cell_bound,
            inactive_unchanged=unchanged,
            face_pos_m3={a: _to_host(v) for a, v in self.pos.items()}, face_neg_m3={a: _to_host(v) for a, v in self.neg.items()},
            face_net_m3={a: _to_host(v) for a, v in self.net.items()})


# --- bounded evolve ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PondEvolveResult:
    """Everything one `evolve_pond` call produced. `state` is the last ACCEPTED state (its `t_s` is the actual elapsed end time);
    `stop_reason` is None when the interval was completed, else "max_retries", "min_dt", "max_steps" or "time_resolution"."""

    state: HydraulicState
    ledger: PondLedger
    start_state: HydraulicState
    requested_s: float
    elapsed_s: float
    reached_target: bool
    stop_reason: str | None
    n_accepted_steps: int
    n_rejected_attempts: int
    rejections: tuple
    min_accepted_dt_s: float
    max_accepted_dt_s: float
    max_cfl: Any
    next_dt_cap_s: float

    def closure(self) -> LedgerClosure:
        return self.ledger.closure(self.state)


def evolve_pond(pond: PondHydraulics, rain_rate_m_per_s: Any, state: HydraulicState, duration_s: float,
                control: ExperimentalControl | None = None) -> PondEvolveResult:
    """Advance `state` by up to `duration_s` under a rain-rate field constant over the interval (module docstring). Returns the
    last accepted state, the exact ledger and the actual elapsed time; a bound that stops the helper is reported, not raised.
    Errors other than a recoverable rejection propagate unchanged and no result exists."""
    control = (ExperimentalControl() if control is None else control).validated()
    duration = _real(duration_s, "duration_s")
    if not (math.isfinite(duration) and duration > 0.0):
        raise PondHydraulicsError(f"duration_s must be finite and > 0, got {duration_s!r}")
    xp = pond.xp
    pond.solver.validate_state(state)
    t0 = check_state_time(state.t_s)
    end = advance_time(t0, duration)
    ledger = PondLedger(pond, state)
    resumed = check_dt_cap(state.next_dt_cap_s)
    cap = control.max_dt_s if resumed is None else min(max(resumed, control.min_dt_s), control.max_dt_s)
    start_state, t = state, t0
    n_rejected, rejections, stop = 0, [], None
    dt_min, dt_max = math.inf, 0.0
    max_cfl = xp.zeros((), dtype=np.float64)
    while t < end and stop is None:
        if ledger.n_steps >= control.max_steps:
            stop = "max_steps"
            break
        remaining = end - t
        dt = min(remaining, cap)
        snap = dt == remaining
        retries = 0
        accepted = None
        while accepted is None:
            if not t + dt > t:
                stop = "time_resolution"
                break
            try:
                accepted = step_once(pond, rain_rate_m_per_s, state, dt)
            except HydraulicStepRejected as exc:
                retries += 1
                n_rejected += 1
                if len(rejections) < _REJECTIONS_KEPT:
                    rejections.append({"t_s": t, "dt_tried_s": dt, "reason": str(exc)})
                if retries > control.max_retries:
                    stop = "max_retries"
                    break
                if dt * 0.5 < control.min_dt_s:
                    stop = "min_dt"
                    break
                dt, snap = dt * 0.5, False
        if accepted is None:
            break
        if retries:
            cap = dt  # the largest step that just worked
        elif dt >= cap:
            cap = min(control.max_dt_s, 2.0 * cap)
        t = end if snap else t + dt
        state = replace(accepted.state, t_s=t, next_dt_cap_s=cap)
        ledger.add(replace(accepted, state=state))
        max_cfl = xp.maximum(max_cfl, accepted.step.max_cfl)
        dt_min, dt_max = min(dt_min, dt), max(dt_max, dt)
    return PondEvolveResult(
        state=state, ledger=ledger, start_state=start_state, requested_s=duration, elapsed_s=float(t - t0),
        reached_target=stop is None and t >= end, stop_reason=stop, n_accepted_steps=ledger.n_steps,
        n_rejected_attempts=n_rejected, rejections=tuple(rejections), min_accepted_dt_s=dt_min if ledger.n_steps else 0.0,
        max_accepted_dt_s=dt_max, max_cfl=max_cfl, next_dt_cap_s=cap)


# --- face-speed diagnostics -------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PondFaceDiagnostics:
    """Stage-consistent diagnostics of one accepted step (arrays in the solver's namespace, fresh). `face_*` are
    `{"x": (ny, nx+1), "y": (ny+1, nx)}`; `cell_outflow_volume_m3` and `cell_speed_m_s` are `(ny, nx)`. `stats` holds host
    numbers. Nothing is clipped."""

    face_flow_depth_m: dict
    face_velocity_m_s: dict
    face_froude: dict
    cell_outflow_volume_m3: Any
    cell_speed_m_s: Any
    stats: dict
    slope_note: str = SLOPE_NOTE


def face_speed_diagnostics(pond: PondHydraulics, accepted: PondStep, *, froude_reference: float = 0.5) -> PondFaceDiagnostics:
    """Face velocity/Froude (existing `stage_face_diagnostics`) and the outflow-volume-weighted cell speed of an accepted step.
    `froude_reference` only sets the counted statistic `n_faces_above_froude_reference`; it limits nothing."""
    ref = _real(froude_reference, "froude_reference")
    if not (math.isfinite(ref) and ref > 0.0):
        raise PondHydraulicsError("froude_reference must be finite and > 0")
    xp = pond.xp
    stage = stage_face_diagnostics(accepted.step)
    ux, uy = stage["face_velocity_m_s"]["x"], stage["face_velocity_m_s"]["y"]
    vx, vy = accepted.step.face_volume_m3["x"], accepted.step.face_volume_m3["y"]
    out_w, out_e = xp.maximum(-vx[:, :-1], 0.0), xp.maximum(vx[:, 1:], 0.0)
    out_s, out_n = xp.maximum(-vy[:-1, :], 0.0), xp.maximum(vy[1:, :], 0.0)
    total = (out_w + out_e) + (out_s + out_n)
    weighted = ((out_w * xp.abs(ux[:, :-1]) + out_e * xp.abs(ux[:, 1:])) + (out_s * xp.abs(uy[:-1, :]) + out_n * xp.abs(uy[1:, :])))
    has_flow = total > 0.0
    speed = xp.where(has_flow, weighted / xp.where(has_flow, total, 1.0), 0.0)
    depths, froude = stage["face_flow_depth_m"], stage["face_froude"]
    n_open = int(np.count_nonzero(pond.topology.geometry.fx_type != 0) + np.count_nonzero(pond.topology.geometry.fy_type != 0))
    n_wet = int(xp.count_nonzero(depths["x"] > 0.0)) + int(xp.count_nonzero(depths["y"] > 0.0))
    stats = {
        "n_open_faces": n_open, "n_wet_faces": n_wet, "n_dry_open_faces": n_open - n_wet,
        "max_face_froude": max(float(xp.max(froude["x"])), float(xp.max(froude["y"]))),
        "froude_reference": ref,
        "n_faces_above_froude_reference": int(xp.count_nonzero(froude["x"] > ref)) + int(xp.count_nonzero(froude["y"] > ref)),
        "max_face_speed_m_s": max(float(xp.max(xp.abs(ux))), float(xp.max(xp.abs(uy)))),
        "max_cell_speed_m_s": float(xp.max(speed)),
        "n_cells_with_outflow": int(xp.count_nonzero(has_flow)),
        "cfl_kind": CFL_NAMES["local_inertial"], "gravity_m_s2": GRAVITY_M_S2,
        "qualification": QUALIFICATION_STATUS,
    }
    return PondFaceDiagnostics(face_flow_depth_m=depths, face_velocity_m_s={"x": ux, "y": uy}, face_froude=froude,
                               cell_outflow_volume_m3=total, cell_speed_m_s=speed, stats=stats)
