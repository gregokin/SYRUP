"""Phase 4c: coupled water-only storm evolution (pure helpers).

One accepted step of length `dt` couples the accepted Phase 3 column
(`infiltration.column_step`) with the accepted Phase 4b method-5 routing
(`routing.route_step`) exactly as the MAHLERAN storm loop orders them
(`MAHLERAN_storm_xml.f90` 100-104: infilt, then route_water):

    col   = column_step(params, h, S, rain_rate, dt)        on the ORIGINAL state
    h*    = col.depth_m                                     (legacy d(1) + excess dt)
    hpre  = max(h - max(J - P, 0), 0)  = min(h, h* - O)     (legacy post-infilt d(1))
    q_old = 0                 if J >= h + P  (complete run-on; infilt.for 106-114)
          = q_prev            elif J <= P    (no run-on; 125-131, legacy q(1) unchanged)
          = k hpre^{3/2}      otherwise      (partial run-on; 141-156)
    route = route_step(graph, h*, hpre, dt, old_discharge=q_old)
    state' = (t + dt, route.depth_m, col.soil_water_m, route.discharge_m2_s)

with J = intake, P = rain depth, O = saturation return. The branch
precedence is the legacy `infilt.for` order (complete first, then no
run-on, then partial), so the three masks are mutually exclusive; at the
overlap h = 0, J = P both branches give q_old = 0. The receiver's old
inflow inside `route_step` is the donor sum of this same `q_old` (the
accepted conservation correction); the legacy stale `qin(1)` is never used
for production. Both kernels are pure, so a rejected attempt leaves the
state untouched and is simply recomputed with dt/2 from the same state.

`evolve` walks from `state.t_s` to `end_s` through boundaries that are
always: every rainfall knot inside the window, every reporting time, and
the end. Boundaries that coincide up to floating-point noise are merged,
keeping the exact forcing edge. It halves dt only on `RoutingStepRejected`
(Courant / negative right-hand side); any other exception -- non-
convergence, invalid input, programming error -- propagates unchanged.
`min_dt_s` is a RETRY floor: a forced slice to a boundary may be shorter
than it as long as time advances, but a halving below it after a rejection
is refused. Guards: `max_retries` per step, `max_steps`, and a refusal when
floating time would not advance. No clipping, no fallback, no dry reset, no
sediment.

The initial state is validated once at entry (namespace, shape, dtype,
finiteness, non-negativity, zero discharge on inactive cells, discharge
consistent with depth, column/graph active masks equal). Hydrograph rows
are written into a preallocated device-resident buffer (one row per
boundary, count known in advance and bounded by `max_report_rows`); the
true peak outlet discharge and its time are tracked per accepted step as
0-d device values; cumulative grids and scalars stay in the graph
namespace, so the loop performs no grid transfers and the only host reads
are the two validating flag reads per attempt (column, route).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from maple_syrup.infiltration import ColumnParameters, ColumnStep, column_step
from maple_syrup.rainfall import RainfallField, RainfallSchedule
from maple_syrup.routing import (
    DEFAULT_BISECTION_ITERATIONS,
    DEFAULT_COURANT_MAX,
    DEFAULT_ROOT_TOLERANCE_M,
    IMPLEMENTATIONS,
    RouteStep,
    RoutingGraph,
    RoutingStepRejected,
    route_step,
)

__all__ = [
    "HYDROGRAPH_COLUMNS",
    "CoupledStep",
    "EvolveResult",
    "StormControl",
    "StormError",
    "StormState",
    "coupled_step",
    "evolve",
    "initial_state",
    "plan_boundaries",
]

# Hydrograph buffer columns (one row per boundary). Cumulative quantities are
# since `state.t_s` at the start of `evolve`; volumes in m3.
HYDROGRAPH_COLUMNS = (
    "t_s",
    "cumulative_rain_m3",
    "cumulative_intake_m3",
    "cumulative_saturation_return_m3",
    "cumulative_drainage_m3",
    "cumulative_export_m3",
    "surface_storage_m3",
    "soil_storage_m3",
    "outlet_discharge_m3_s",  # instantaneous, end of the last accepted step (legacy q_plot dx)
    "max_depth_m",
    "max_velocity_m_s",
    "accepted_steps",
    "rejected_attempts",
    "min_accepted_dt_s",
    "max_routing_cell_balance_residual_m",
    "max_constitutive_residual_m",
)
_EPS = float(np.finfo(np.float64).eps)
# Two boundaries closer than this (relative to their magnitude) are floating-
# point noise of the same instant, e.g. 0.1 * 3 versus a 0.3 forcing edge.
_COINCIDENT_RTOL = 16.0 * _EPS


class StormError(RuntimeError):
    """Configuration, validation or guard failure of the coupled evolution
    (invalid state/control/plan, retry budget, retry floor, step count,
    non-advancing time). Nothing is published and no input array is
    modified."""


# --- state and control ---------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class StormState:
    """Surface depth h, retained soil water S and the last routed unit
    discharge q (legacy q(2)), all `(ny, nx)` float64 in the graph
    namespace. `q` is consistent with `h` through q = k h_flow^{3/2} within
    the routing root tolerance; `initial_state` makes it so from h."""

    t_s: float
    depth_m: Any
    soil_water_m: Any
    discharge_m2_s: Any


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise StormError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _strict_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise StormError(f"{name} must be an int >= {minimum}, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class StormControl:
    max_dt_s: float = 1.0
    min_dt_s: float = 1.0 / 1024.0  # retry floor (see module docstring)
    max_retries: int = 10  # halvings of one step before giving up
    max_steps: int = 10_000_000
    courant_max: float = DEFAULT_COURANT_MAX
    bisection_iterations: int = DEFAULT_BISECTION_ITERATIONS
    root_tolerance_m: float = DEFAULT_ROOT_TOLERANCE_M
    implementation: str = "array"

    def validated(self) -> StormControl:
        """Strict: bools and non-integers are refused, never coerced."""
        for name in ("max_dt_s", "min_dt_s"):
            value = _real(getattr(self, name), name)
            if not (math.isfinite(value) and value > 0.0):
                raise StormError(f"{name} must be finite and > 0, got {value!r}")
        if self.min_dt_s > self.max_dt_s:
            raise StormError(f"min_dt_s {self.min_dt_s} exceeds max_dt_s {self.max_dt_s}")
        _strict_int(self.max_retries, "max_retries")
        _strict_int(self.max_steps, "max_steps")
        _strict_int(self.bisection_iterations, "bisection_iterations")
        for name in ("courant_max", "root_tolerance_m"):
            _real(getattr(self, name), name)  # ranges are enforced by route_step
        if self.implementation not in IMPLEMENTATIONS:
            raise StormError(f"implementation must be one of {IMPLEMENTATIONS}, got {self.implementation!r}")
        return self


def _validate_state(graph: RoutingGraph, state: StormState, root_tolerance_m: float,
                    params: ColumnParameters | None = None) -> None:
    """One batched check of a state against the graph (and the column
    parameters if given). Raises `StormError`; reads nothing otherwise."""
    from maple.core.backend import (
        DeferredChecks,
        MixedArrayNamespaceError,
        array_namespace,
        errstate,
        finite_flag,
        is_array,
        negative_flag,
        true_flag,
    )

    xp, shape = graph.xp, graph.shape
    arrays = {"depth_m": state.depth_m, "soil_water_m": state.soil_water_m, "discharge_m2_s": state.discharge_m2_s}
    for name, array in arrays.items():
        if not is_array(array):
            raise StormError(f"state.{name} must be a NumPy/CuPy array, got {type(array).__name__}")
        if tuple(array.shape) != shape or array.dtype != np.float64:
            raise StormError(f"state.{name} must be float64 with shape {shape}, got {array.dtype} {tuple(array.shape)}")
    try:
        namespace = array_namespace(*arrays.values(), graph.conveyance)
    except MixedArrayNamespaceError as exc:
        raise StormError(f"state arrays and graph are in mixed namespaces: {exc}") from None
    if namespace is not xp:
        raise StormError(f"state arrays must be in the graph namespace {xp.__name__!r}")
    t = _real(state.t_s, "state.t_s")
    if not (math.isfinite(t) and t >= 0.0):
        raise StormError(f"state.t_s must be finite and >= 0, got {t!r}")
    if params is not None and (params.shape != shape or params.xp is not xp):
        raise StormError("column parameters must match the graph shape and namespace")
    active = graph.active_flat.reshape(shape)
    checks = DeferredChecks()
    for name, array in arrays.items():
        checks.require(finite_flag(array), f"state.{name} must be finite everywhere")
        checks.forbid(negative_flag(array), f"state.{name} must be >= 0 everywhere")
    q, h = state.discharge_m2_s, state.depth_m
    checks.forbid(true_flag(~active & (q != 0.0)), "state.discharge_m2_s must be 0 on inactive cells")
    k = graph.conveyance.reshape(shape)
    with errstate(xp=xp, all="ignore"):
        implied = (q / xp.where(active, k, 1.0)) ** (2.0 / 3.0)
        mismatch = active & (xp.abs(implied - h) > root_tolerance_m + 64.0 * _EPS * h)
    checks.forbid(true_flag(mismatch), "state.discharge_m2_s is not consistent with state.depth_m through "
                                       "q = k h^{3/2} within root_tolerance_m (use initial_state)")
    if params is not None:
        checks.forbid(true_flag(params.active_mask != active),
                      "column active_mask differs from the routing graph's active cells; unsupported")
    try:
        checks.resolve()
    except ValueError as exc:
        raise StormError(str(exc)) from None


def initial_state(graph: RoutingGraph, depth_m: Any, soil_water_m: Any, *, t_s: float = 0.0) -> StormState:
    """State whose discharge is consistent with the depth: q = k h^{3/2}
    (zero on inactive cells), formed exactly as `route_step` forms its default
    old flux. Inputs are validated (graph namespace, shape, dtype, finite,
    >= 0) and copied."""
    from maple.core.backend import (
        MixedArrayNamespaceError,
        array_namespace,
        errstate,
        is_array,
    )

    xp = graph.xp
    for name, array in (("depth_m", depth_m), ("soil_water_m", soil_water_m)):
        if not is_array(array) or tuple(array.shape) != graph.shape or array.dtype != np.float64:
            raise StormError(f"{name} must be a float64 array with shape {graph.shape}")
    # Check the caller's arrays before xp operations can convert or transfer them.
    try:
        namespace = array_namespace(depth_m, soil_water_m, graph.conveyance)
    except MixedArrayNamespaceError as exc:
        raise StormError(f"initial arrays and graph are in mixed namespaces: {exc}") from None
    if namespace is not xp:
        raise StormError(f"initial arrays must be in the graph namespace {xp.__name__!r}")
    with errstate(xp=xp, all="ignore"):
        k = graph.conveyance.reshape(graph.shape)
        active = graph.active_flat.reshape(graph.shape)
        q = xp.where(active, (xp.sqrt(depth_m) * depth_m) * k, 0.0)
    state = StormState(_real(t_s, "t_s"), xp.array(depth_m, copy=True), xp.array(soil_water_m, copy=True), q)
    _validate_state(graph, state, DEFAULT_ROOT_TOLERANCE_M)
    return state


# --- one coupled attempt -------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class CoupledStep:
    """One accepted coupled step. `column`/`route` are the pure kernel
    results; `state` is the new state; the branch counts are 0-d device
    integers over active cells (complete, no run-on, partial: mutually
    exclusive, legacy precedence)."""

    dt_s: float
    state: StormState
    column: ColumnStep
    route: RouteStep
    n_no_runon: Any
    n_partial_runon: Any
    n_complete_runon: Any


def coupled_step(graph: RoutingGraph, params: ColumnParameters, rain_rate_m_per_s: Any, state: StormState,
                 dt_s: float, control: StormControl) -> CoupledStep:
    """Column then routing on the ORIGINAL `state` (see module docstring).
    Pure: raises before returning anything on any kernel failure;
    `RoutingStepRejected` means "retry with a smaller dt"."""
    from maple.core.backend import errstate

    xp = graph.xp
    shape = graph.shape
    h, soil, q_prev = state.depth_m, state.soil_water_m, state.discharge_m2_s
    col = column_step(params, h, soil, rain_rate_m_per_s, dt_s)
    rain, intake = col.rain_m, col.intake_m
    with errstate(xp=xp, all="ignore"):
        hpre = xp.maximum(h - xp.maximum(intake - rain, 0.0), 0.0)
        k = graph.conveyance.reshape(shape)
        active = graph.active_flat.reshape(shape)
        complete = active & (intake >= h + rain)  # infilt.for 106: tested first
        no_runon = active & ~complete & (intake <= rain)  # 125
        partial = active & ~complete & ~no_runon  # 141
        recomputed = (xp.sqrt(hpre) * hpre) * k
        q_old = xp.where(complete, 0.0, xp.where(no_runon, q_prev, xp.where(partial, recomputed, 0.0)))
    route = route_step(
        graph, col.depth_m, hpre, dt_s, old_discharge_m2_s=q_old, courant_max=control.courant_max,
        bisection_iterations=control.bisection_iterations, root_tolerance_m=control.root_tolerance_m,
        implementation=control.implementation,
    )
    new_state = StormState(state.t_s + float(dt_s), route.depth_m, col.soil_water_m, route.discharge_m2_s)
    return CoupledStep(
        dt_s=float(dt_s), state=new_state, column=col, route=route,
        n_no_runon=xp.sum(no_runon), n_partial_runon=xp.sum(partial), n_complete_runon=xp.sum(complete),
    )


# --- boundaries -----------------------------------------------------------------------------
def plan_boundaries(schedule: RainfallSchedule, start_s: float, end_s: float, report_every_s: float,
                    *, max_report_rows: int = 100_000) -> np.ndarray:
    """Strictly increasing step/reporting boundaries in (start, end]: every
    rainfall knot inside the window, `start + k report_every_s`, and `end`.
    Values that coincide up to floating-point noise (16 eps relative) are
    merged into one boundary, keeping the exact forcing edge or end, so no
    rate change is skipped and no spurious sub-eps slice is created. Refuses
    an empty window and more rows than `max_report_rows` (a strict int)."""
    start, end, every = (_real(v, n) for v, n in ((start_s, "start_s"), (end_s, "end_s"),
                                                  (report_every_s, "report_every_s")))
    if not all(math.isfinite(v) for v in (start, end, every)):
        raise StormError("start_s, end_s and report_every_s must be finite")
    rows_max = _strict_int(max_report_rows, "max_report_rows")
    if start < 0.0 or end <= start or every <= 0.0:
        raise StormError(f"need 0 <= start_s < end_s and report_every_s > 0, got {start}, {end}, {every}")
    n_report = math.ceil((end - start) / every)
    if n_report > rows_max:
        raise StormError(f"{n_report} reporting rows exceed max_report_rows = {rows_max}")
    reports = start + every * np.arange(1, n_report + 1, dtype=np.float64)
    knots = schedule.edges_s[(schedule.edges_s > start) & (schedule.edges_s < end)]
    exact = set(knots.tolist()) | {end}
    merged: list[float] = []
    for b in np.unique(np.concatenate([reports[reports < end], knots, [end]])).tolist():
        if b <= start:
            continue
        if (merged and b - merged[-1] <= _COINCIDENT_RTOL * max(abs(b), 1.0)
                and not (b in exact and merged[-1] in exact)):
            if b in exact:
                merged[-1] = b  # keep the exact forcing edge / end
            continue
        merged.append(b)
    boundaries = np.array(merged, dtype=np.float64)
    if boundaries.size == 0 or boundaries[-1] != end or boundaries.size > rows_max:
        raise StormError("boundary plan is empty, does not end at end_s, or exceeds max_report_rows")
    return boundaries


# --- evolution -------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class EvolveResult:
    """Everything `evolve` accumulated. Grids and 0-d values live in the graph
    namespace; `hydrograph` is `(n_boundaries, len(HYDROGRAPH_COLUMNS))`.
    `peak_outlet_discharge_m3_s` / `time_of_peak_outlet_s` are the true
    per-accepted-step maxima (the hydrograph rows are only samples)."""

    state: StormState
    hydrograph: Any
    boundaries: np.ndarray
    cumulative_rain_m: Any
    cumulative_intake_m: Any
    cumulative_saturation_return_m: Any
    cumulative_drainage_m: Any
    cumulative_export_m3: Any
    peak_depth_m: Any
    peak_velocity_m_s: Any
    last_velocity_m_s: Any
    peak_outlet_discharge_m3_s: Any
    time_of_peak_outlet_s: Any
    n_accepted_steps: int
    n_rejected_attempts: int
    rejections: tuple[dict[str, Any], ...]  # first few (t, dt_tried, reason) for the report
    min_accepted_dt_s: float
    max_accepted_dt_s: float
    cell_steps_no_runon: Any
    cell_steps_partial_runon: Any
    cell_steps_complete_runon: Any
    max_courant_old: Any
    max_courant_new: Any
    first_step_wall_s: float
    first_step_cpu_s: float
    remaining_wall_s: float
    remaining_cpu_s: float


def evolve(
    graph: RoutingGraph,
    params: ColumnParameters,
    field: RainfallField,
    schedule: RainfallSchedule,
    state: StormState,
    end_s: float,
    control: StormControl,
    *,
    report_every_s: float,
    max_report_rows: int = 100_000,
) -> EvolveResult:
    """Advance `state` to `end_s` (see module docstring). Raises `StormError`
    on validation and guard failures; `RoutingStepRejected` never escapes
    (it is retried until the guards trip); every other exception propagates
    untouched. Owns every scratch array it writes; inputs are never
    modified."""
    control = control.validated()
    xp = graph.xp
    shape = graph.shape
    area = graph.dx_m * graph.dx_m
    _validate_state(graph, state, control.root_tolerance_m, params)
    if not isinstance(field, RainfallField) or field.shape != shape or field.xp is not xp:
        raise StormError("rainfall field must match the graph shape and namespace")
    boundaries = plan_boundaries(schedule, state.t_s, end_s, report_every_s, max_report_rows=max_report_rows)

    def zeros():
        return xp.zeros(shape, dtype=np.float64)

    def scalar(value=0.0):
        return xp.asarray(value, dtype=np.float64)

    active = graph.active_flat.reshape(shape)
    k = graph.conveyance.reshape(shape)
    cum_rain, cum_intake, cum_return, cum_drain = zeros(), zeros(), zeros(), zeros()
    peak_depth = xp.array(state.depth_m, copy=True)
    last_velocity = xp.where(active, xp.sqrt(state.depth_m) * k, 0.0)  # v = k sqrt(h_flow) of the initial state
    peak_velocity = xp.array(last_velocity, copy=True)
    outlet_q = graph.dx_m * xp.sum(xp.where(graph.outlet_flat.reshape(shape), state.discharge_m2_s, 0.0))
    peak_q, peak_t = scalar(outlet_q), scalar(state.t_s)
    cum_export = scalar()
    max_balance, max_constitutive, max_cr_old, max_cr_new = scalar(), scalar(), scalar(), scalar()
    cells_no, cells_partial, cells_complete = (xp.zeros((), dtype=np.int64) for _ in range(3))
    hydrograph = xp.zeros((boundaries.size, len(HYDROGRAPH_COLUMNS)), dtype=np.float64)
    rate = xp.empty(shape, dtype=np.float64)  # owned scratch; never aliases a caller array

    n_steps = n_rejected = 0
    rejections: list[dict[str, Any]] = []
    dt_min, dt_max = math.inf, 0.0
    t = state.t_s
    first_wall = first_cpu = 0.0
    wall0, cpu0 = time.perf_counter(), time.process_time()
    for row, boundary in enumerate(boundaries.tolist()):
        # Each boundary interval is split into n equal planned substeps of
        # span / n <= max_dt_s that land exactly on their targets (the last
        # one on the boundary itself), so a 60 s interval at max_dt 0.25 s is
        # 240 identical steps and reporting cadences that share the same
        # integer partition give the same accepted steps. A planned target a
        # forcing/report/end boundary makes shorter than min_dt_s is still
        # stepped (min_dt_s is a RETRY floor, see the module docstring).
        t_row = t
        span = boundary - t_row
        n_sub = max(1, math.ceil(span / control.max_dt_s))
        while span / n_sub > control.max_dt_s:
            n_sub += 1
        for k in range(1, n_sub + 1):
            target = boundary if k == n_sub else t_row + k * (span / n_sub)
            if not target > t:
                raise StormError(f"floating time does not advance: planned substep target {target} s is not "
                                 f"after t = {t} s (time resolution too coarse for max_dt_s = {control.max_dt_s})")
            while t < target:
                remaining = target - t
                dt = remaining
                snap = True
                retries = 0
                while True:
                    if n_steps >= control.max_steps:
                        raise StormError(f"max_steps = {control.max_steps} reached at t = {t} s before end_s = "
                                         f"{end_s} s")
                    if not t + dt > t:
                        raise StormError(f"floating time does not advance: t = {t} s, dt = {dt} s")
                    field.apply(schedule.rate_after_m_per_s(t), out=rate)
                    try:
                        step = coupled_step(graph, params, rate, state, dt, control)
                    except RoutingStepRejected as exc:
                        retries += 1
                        n_rejected += 1
                        if len(rejections) < 100:
                            rejections.append({"t_s": t, "dt_tried_s": dt, "reason": str(exc)})
                        if retries > control.max_retries:
                            raise StormError(f"step at t = {t} s rejected {retries} times (max_retries = "
                                             f"{control.max_retries}); last dt {dt} s: {exc}") from exc
                        halved = dt * 0.5
                        if halved < control.min_dt_s:
                            raise StormError(f"halving dt {dt} s to {halved} s would fall below the retry floor "
                                             f"min_dt_s = {control.min_dt_s} s at t = {t} s after {retries} "
                                             f"rejection(s): {exc}") from exc
                        dt = halved
                        snap = False
                        continue
                    break
                t = target if snap else t + dt
                state = replace(step.state, t_s=t)
                col, route = step.column, step.route
                cum_rain += col.rain_m
                cum_intake += col.intake_m
                cum_return += col.saturation_return_m
                cum_drain += col.drainage_m
                cum_export = cum_export + route.export_m3
                outlet_q = route.outlet_discharge_m3_s
                higher = outlet_q > peak_q
                peak_t = xp.where(higher, scalar(t), peak_t)
                peak_q = xp.where(higher, outlet_q, peak_q)
                xp.maximum(peak_depth, route.depth_m, out=peak_depth)
                xp.maximum(peak_velocity, route.velocity_m_s, out=peak_velocity)
                last_velocity = route.velocity_m_s
                max_balance = xp.maximum(max_balance, route.max_cell_balance_residual_m)
                max_constitutive = xp.maximum(max_constitutive, route.max_constitutive_residual_m)
                max_cr_old = xp.maximum(max_cr_old, route.max_courant_old)
                max_cr_new = xp.maximum(max_cr_new, route.max_courant_new)
                cells_no = cells_no + step.n_no_runon
                cells_partial = cells_partial + step.n_partial_runon
                cells_complete = cells_complete + step.n_complete_runon
                n_steps += 1
                dt_min, dt_max = min(dt_min, dt), max(dt_max, dt)
                if n_steps == 1:
                    first_wall, first_cpu = time.perf_counter() - wall0, time.process_time() - cpu0
                    wall0, cpu0 = time.perf_counter(), time.process_time()
        hydrograph[row] = xp.stack([
            scalar(t),
            area * xp.sum(cum_rain), area * xp.sum(cum_intake), area * xp.sum(cum_return),
            area * xp.sum(cum_drain), scalar(cum_export),
            area * xp.sum(state.depth_m), area * xp.sum(state.soil_water_m),
            scalar(outlet_q),
            xp.max(state.depth_m), xp.max(last_velocity),
            scalar(float(n_steps)), scalar(float(n_rejected)),
            scalar(dt_min if n_steps else 0.0),
            scalar(max_balance), scalar(max_constitutive),
        ])
    remaining_wall, remaining_cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    return EvolveResult(
        state=state, hydrograph=hydrograph, boundaries=boundaries,
        cumulative_rain_m=cum_rain, cumulative_intake_m=cum_intake, cumulative_saturation_return_m=cum_return,
        cumulative_drainage_m=cum_drain, cumulative_export_m3=cum_export,
        peak_depth_m=peak_depth, peak_velocity_m_s=peak_velocity, last_velocity_m_s=last_velocity,
        peak_outlet_discharge_m3_s=peak_q, time_of_peak_outlet_s=peak_t,
        n_accepted_steps=n_steps, n_rejected_attempts=n_rejected, rejections=tuple(rejections),
        min_accepted_dt_s=dt_min, max_accepted_dt_s=dt_max,
        cell_steps_no_runon=cells_no, cell_steps_partial_runon=cells_partial,
        cell_steps_complete_runon=cells_complete,
        max_courant_old=max_cr_old, max_courant_new=max_cr_new,
        first_step_wall_s=first_wall, first_step_cpu_s=first_cpu,
        remaining_wall_s=remaining_wall, remaining_cpu_s=remaining_cpu,
    )
