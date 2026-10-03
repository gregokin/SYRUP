"""Shared bounded evolution of the two EXPERIMENTAL hydraulic alternatives (`experimental_hydrology`, `experimental_cuda`).

ONE driver serves both methods and both backends: it takes a solver object (`CpuHydraulicSolver` or
`experimental_cuda.CudaHydraulicSolver`) and advances a `HydraulicState` through the SAME boundaries as the accepted storm
driver (`storm.plan_boundaries`: every rainfall knot inside the window, every reporting time, the end; coincident values
merged onto the exact forcing edge) with the SAME rainfall schedule and field (`RainfallField.apply`, rate constant over every
step, never averaged across a forcing discontinuity). Requested snapshot times are ADDED as extra boundaries (a step is split
there; the forcing is not altered) and merged onto an existing boundary within 16 eps.

Step control: each boundary interval is split into equal planned substeps no longer than `max_dt_s`. A step is attempted with
`min(remaining, dt_cap)`; a recoverable `HydraulicStepRejected` (CFL bound, or a negative depth without the limiter) halves dt
from the UNCHANGED state (no clipping), bounded by `max_retries` per step, the retry floor `min_dt_s` (a forced slice to a
boundary may be shorter than it) and `max_steps`; the cap `dt_cap` starts at `max_dt_s` (fresh event), becomes the accepted dt
after a rejection and doubles (up to `max_dt_s`) after every clean full-size step. Every other exception propagates unchanged and
no result exists. The initial state is validated once at entry (`solver.validate_state`), every step validates its inputs.

CONTINUATION CONTRACT. The cap is numerical history that shapes the trajectory (the step sequence, hence the physical state), so
it is part of the continuation state: every accepted `HydraulicState` this driver returns carries `next_dt_cap_s` (also
`result.next_dt_cap_s`), and a state passed back in resumes from it, clamped into `[min_dt_s, max_dt_s]` of the NEW control. Given
the same forcing, boundary/report grid, grid and control, a run to a boundary followed by a resumed run is the same step sequence
as one continuous run (state, face momentum and accepted/rejected counts add up). The cap is NOT reset at boundaries. A state
without it (None, e.g. `solver.initial_state`) is a fresh event whose cap starts at `max_dt_s`: fresh-event dynamics are
unchanged. The cap is validated (None or a real, finite, positive number) before any step. No disk restart is implemented: a
caller who persists the state must persist this number with the arrays and the time (the CLI saves `state_t_s` and
`state_next_dt_cap_s` in `final_state.npz` and reports them in the summary).

Accumulation and reporting use the namespace of the solver (NumPy or CuPy): cumulative rain/intake/return/drainage grids,
peak depth and peak velocity grids, the last velocity map, the true peak outlet discharge and its time, cumulative export, the
maximum CFL number and the donor-limiter totals stay on the device; the hydrograph rows are written into a device buffer.

TRANSFER SCOPE (what is and is not claimed). MAPLE's `read_transfer_counters` sees only its instrumented helpers (`to_host`,
`to_device`); raw host conversions bypass these helpers. In CuPy14.2 Python scalars use fill kernels, whereas host arrays can
incur uncounted copies. A zero helper counter is not a complete transfer trace.
This driver therefore creates no device array from a host value at all, per step or per report row or at initialization: new
device scalars come from `xp.zeros` / `xp.full`, and Python floats (the time of a new peak, the row time/counters/smallest dt)
enter kernels BY VALUE (`xp.where`, a one-element `fill`). The loop's only host reads are the solver's small counted packets
(two per attempted step on CUDA); the caller's final reporting downloads are counted apart. Device-to-device copies, fills and
allocations are not transfers and are not counted. The static uploads of preparation happen in the solver constructor.

Peak fields: `peak_depth_m` is the maximum over accepted steps (and the initial state) of the END-of-step depth; `peak_velocity_m_s`
the maximum of the END-of-step instantaneous velocity of the method (see `HydraulicStep`); `last_velocity_m_s` the final map.
There is no pickup, travel-distance or detachment field here (water only). Snapshots hold synchronous copies of depth, soil
water, velocity and (local inertia) the face fluxes at the requested boundaries.

Nothing here was run by its author (file-only tools); Codex records results.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from maple_syrup.experimental_hydrology import (
    HydraulicState,
    HydraulicStepRejected,
    check_dt_cap,
)
from maple_syrup.rainfall import RainfallField, RainfallSchedule
from maple_syrup.storm import StormError, plan_boundaries

__all__ = [
    "EXPERIMENT_HYDROGRAPH_COLUMNS",
    "ExperimentalControl",
    "ExperimentalEvolveResult",
    "evolve_experimental",
]

EXPERIMENT_HYDROGRAPH_COLUMNS = (
    "t_s",
    "cumulative_rain_m3",
    "cumulative_intake_m3",
    "cumulative_saturation_return_m3",
    "cumulative_drainage_m3",
    "cumulative_export_m3",
    "surface_storage_m3",
    "soil_storage_m3",
    "outlet_discharge_m3_s",  # instantaneous, end of the last accepted step
    "max_depth_m",
    "max_velocity_m_s",
    "accepted_steps",
    "rejected_attempts",
    "min_accepted_dt_s",
    "max_cfl",
    "cumulative_limited_volume_m3",
)
_EPS = float(np.finfo(np.float64).eps)
_COINCIDENT_RTOL = 16.0 * _EPS


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise StormError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _strict_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise StormError(f"{name} must be an int >= {minimum}, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class ExperimentalControl:
    """Driver guards (strict: bools and non-integers are refused, never coerced). Numerical options of the solver live in
    `HydraulicControl`."""

    max_dt_s: float = 1.0
    min_dt_s: float = 1.0 / 1024.0  # retry floor
    max_retries: int = 10
    max_steps: int = 10_000_000

    def validated(self) -> ExperimentalControl:
        for name in ("max_dt_s", "min_dt_s"):
            value = _real(getattr(self, name), name)
            if not (math.isfinite(value) and value > 0.0):
                raise StormError(f"{name} must be finite and > 0, got {value!r}")
        if self.min_dt_s > self.max_dt_s:
            raise StormError(f"min_dt_s {self.min_dt_s} exceeds max_dt_s {self.max_dt_s}")
        _strict_int(self.max_retries, "max_retries")
        _strict_int(self.max_steps, "max_steps")
        return self


@dataclass(frozen=True, eq=False)
class ExperimentalEvolveResult:
    """Everything one `evolve_experimental` call accumulated. Grids and 0-d values live in the solver's namespace and are fresh
    outputs; `hydrograph` is `(n_boundaries, len(columns))`; `snapshots` maps each requested time to its synchronous copies."""

    method: str
    implementation: str
    columns: tuple[str, ...]
    state: HydraulicState
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
    max_cfl: Any
    limited_cells_total: Any
    limited_volume_total_m3: Any
    n_accepted_steps: int
    n_rejected_attempts: int
    rejections: tuple[dict[str, Any], ...]
    min_accepted_dt_s: float
    max_accepted_dt_s: float
    snapshots: dict
    first_step_wall_s: float
    first_step_cpu_s: float
    remaining_wall_s: float
    remaining_cpu_s: float
    next_dt_cap_s: float  # the adaptive cap at the end (also on `state`): pass `state` back in to continue exactly


def _with_snapshots(planned: np.ndarray, requested: tuple, start: float, end: float, max_rows: int):
    """Planned boundaries plus the requested snapshot times (inserted or merged within 16 eps); returns the new boundary array
    and `{boundary index: requested time}`."""
    if not isinstance(requested, (tuple, list)):
        raise StormError("snapshot_times_s must be a tuple or list of times")
    times = []
    for value in requested:
        s = _real(value, "snapshot time")
        if not (math.isfinite(s) and start < s <= end):
            raise StormError(f"snapshot time {s!r} must lie in ({start}, {end}]")
        times.append(s)
    boundaries = planned.tolist()
    for s in sorted(set(times)):
        if not any(abs(s - b) <= _COINCIDENT_RTOL * max(abs(b), 1.0) for b in boundaries):
            boundaries.append(s)
    boundaries = sorted(boundaries)
    if len(boundaries) > max_rows:
        raise StormError(f"{len(boundaries)} reporting rows exceed max_report_rows = {max_rows}")
    array = np.array(boundaries, dtype=np.float64)
    rows = {}
    for s in times:
        rows[int(np.argmin(np.abs(array - s)))] = s
    return array, rows


def evolve_experimental(solver: Any, field: RainfallField, schedule: RainfallSchedule, state: HydraulicState, end_s: float,
                        control: ExperimentalControl, *, report_every_s: float, max_report_rows: int = 100_000,
                        snapshot_times_s: tuple = ()) -> ExperimentalEvolveResult:
    """Advance `state` to `end_s` with `solver` (see module docstring). Raises `StormError` on validation and guard failures;
    `HydraulicStepRejected` never escapes (it is retried until the guards trip); every other exception propagates untouched.
    Owns every scratch array it writes; inputs are never modified."""
    control = control.validated()
    xp, shape = solver.xp, tuple(solver.shape)
    area = solver.dx_m * solver.dx_m
    solver.validate_state(state)
    if not isinstance(field, RainfallField) or field.shape != shape or field.xp is not xp:
        raise StormError("rainfall field must match the solver's shape and namespace")
    planned = plan_boundaries(schedule, state.t_s, end_s, report_every_s, max_report_rows=max_report_rows)
    boundaries, snapshot_rows = _with_snapshots(planned, tuple(snapshot_times_s), state.t_s, float(end_s),
                                                _strict_int(max_report_rows, "max_report_rows"))

    def zeros():
        return xp.zeros(shape, dtype=np.float64)

    def resident(value, name):
        """Require a resident value; implicit host conversion can add allocation, launches or array copies."""
        if xp is not np and not isinstance(value, xp.ndarray):
            raise StormError(f"{name} must be device-resident on a CUDA solver, got {type(value).__name__}")
        return value

    # No host scalar is converted into an extra device array (in CuPy14.2 xp.asarray(float) adds allocation/fill overhead): initial
    # values are created on the device by xp.zeros / xp.full (a fill kernel with the value as a kernel argument), and host
    # numbers used later (the time of a new peak, the report-row time and counters) enter kernels by value.
    cum_rain, cum_intake, cum_return, cum_drain = zeros(), zeros(), zeros(), zeros()
    peak_depth = xp.array(state.depth_m, copy=True)
    last_velocity, peak_velocity = zeros(), zeros()
    outlet_q = xp.zeros((), dtype=np.float64)
    peak_q, peak_t = xp.zeros((), dtype=np.float64), xp.full((), state.t_s, dtype=np.float64)
    cum_export, max_cfl, lim_volume = (xp.zeros((), dtype=np.float64) for _ in range(3))
    lim_cells = xp.zeros((), dtype=np.int64)
    hydrograph = xp.zeros((boundaries.size, len(EXPERIMENT_HYDROGRAPH_COLUMNS)), dtype=np.float64)
    rate = xp.empty(shape, dtype=np.float64)  # owned scratch; never aliases a caller array
    snapshots: dict = {}

    n_steps = n_rejected = 0
    rejections: list[dict[str, Any]] = []
    dt_min, dt_max = math.inf, 0.0
    # Continuation: a fresh event (next_dt_cap_s None) starts at max_dt_s exactly as before; a resumed state continues from the
    # adaptive cap its previous run ended with (bounded by THIS control: never above max_dt_s, never below the retry floor).
    resumed_cap = check_dt_cap(state.next_dt_cap_s)
    dt_cap = control.max_dt_s if resumed_cap is None else min(max(resumed_cap, control.min_dt_s), control.max_dt_s)
    t = state.t_s
    first_wall = first_cpu = 0.0
    wall0, cpu0 = time.perf_counter(), time.process_time()
    for row, boundary in enumerate(boundaries.tolist()):
        t_row = t
        span = boundary - t_row
        n_sub = max(1, math.ceil(span / control.max_dt_s))
        while span / n_sub > control.max_dt_s:
            n_sub += 1
        for k in range(1, n_sub + 1):
            target = boundary if k == n_sub else t_row + k * (span / n_sub)
            if not target > t:
                raise StormError(f"floating time does not advance: planned substep target {target} s is not after "
                                 f"t = {t} s (time resolution too coarse for max_dt_s = {control.max_dt_s})")
            while t < target:
                remaining = target - t
                dt = min(remaining, dt_cap)
                snap = dt == remaining
                retries = 0
                while True:
                    if n_steps >= control.max_steps:
                        raise StormError(f"max_steps = {control.max_steps} reached at t = {t} s before end_s = {end_s} s")
                    if not t + dt > t:
                        raise StormError(f"floating time does not advance: t = {t} s, dt = {dt} s")
                    field.apply(schedule.rate_after_m_per_s(t), out=rate)
                    try:
                        step = solver.step(rate, state, dt)
                    except HydraulicStepRejected as exc:
                        retries += 1
                        n_rejected += 1
                        if len(rejections) < 100:
                            rejections.append({"t_s": t, "dt_tried_s": dt, "reason": str(exc)})
                        if retries > control.max_retries:
                            raise StormError(f"step at t = {t} s rejected {retries} times (max_retries = "
                                             f"{control.max_retries}); last dt {dt} s: {exc}") from exc
                        halved = dt * 0.5
                        if halved < control.min_dt_s:
                            raise StormError(f"halving dt {dt} s to {halved} s would fall below the retry floor min_dt_s = "
                                             f"{control.min_dt_s} s at t = {t} s after {retries} rejection(s): {exc}") from exc
                        dt = halved
                        snap = False
                        continue
                    break
                if retries:
                    dt_cap = dt  # the largest step that just worked
                elif dt >= dt_cap:
                    dt_cap = min(control.max_dt_s, 2.0 * dt_cap)  # a clean full-size step: grow back toward max_dt_s
                t = target if snap else t + dt
                state = replace(step.state, t_s=t, next_dt_cap_s=dt_cap)  # the cap the NEXT step will start from
                col = step.column
                cum_rain += col.rain_m
                cum_intake += col.intake_m
                cum_return += col.saturation_return_m
                cum_drain += col.drainage_m
                cum_export = cum_export + step.export_m3
                outlet_q = step.outlet_discharge_m3_s
                higher = outlet_q > peak_q
                peak_t = xp.where(higher, t, peak_t)  # t (a Python float) is a kernel argument, not a device array
                peak_q = xp.where(higher, outlet_q, peak_q)
                xp.maximum(peak_depth, state.depth_m, out=peak_depth)
                xp.maximum(peak_velocity, step.velocity_m_s, out=peak_velocity)
                last_velocity = step.velocity_m_s
                max_cfl = xp.maximum(max_cfl, step.max_cfl)
                lim_cells = lim_cells + step.limited_cells
                lim_volume = lim_volume + step.limited_volume_m3
                n_steps += 1
                dt_min, dt_max = min(dt_min, dt), max(dt_max, dt)
                if n_steps == 1:
                    first_wall, first_cpu = time.perf_counter() - wall0, time.process_time() - cpu0
                    wall0, cpu0 = time.perf_counter(), time.process_time()
        # device values are stacked (device to device); the four host numbers (time, counters, smallest dt) are written by
        # value with a one-element fill, so a report row creates no host-to-device array either
        hydrograph[row, 1:11] = xp.stack([
            area * xp.sum(cum_rain), area * xp.sum(cum_intake), area * xp.sum(cum_return), area * xp.sum(cum_drain),
            resident(cum_export, "cumulative export"),
            area * xp.sum(state.depth_m), area * xp.sum(state.soil_water_m),
            resident(outlet_q, "outlet discharge"),
            xp.max(state.depth_m), xp.max(last_velocity),
        ])
        hydrograph[row, 14:16] = xp.stack([resident(max_cfl, "max_cfl"), resident(lim_volume, "limited volume")])
        for column, value in ((0, t), (11, float(n_steps)), (12, float(n_rejected)), (13, dt_min if n_steps else 0.0)):
            hydrograph[row, column:column + 1].fill(value)
        if row in snapshot_rows:
            snap_arrays = {"t_s": t, "depth_m": xp.array(state.depth_m, copy=True),
                           "soil_water_m": xp.array(state.soil_water_m, copy=True),
                           "velocity_m_s": xp.array(last_velocity, copy=True)}
            if state.qx_m2_s is not None:
                snap_arrays["qx_m2_s"] = xp.array(state.qx_m2_s, copy=True)
                snap_arrays["qy_m2_s"] = xp.array(state.qy_m2_s, copy=True)
            snapshots[snapshot_rows[row]] = snap_arrays
    remaining_wall, remaining_cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    return ExperimentalEvolveResult(
        method=solver.method, implementation=solver.implementation, columns=EXPERIMENT_HYDROGRAPH_COLUMNS, state=state,
        hydrograph=hydrograph, boundaries=boundaries, cumulative_rain_m=cum_rain, cumulative_intake_m=cum_intake,
        cumulative_saturation_return_m=cum_return, cumulative_drainage_m=cum_drain, cumulative_export_m3=cum_export,
        peak_depth_m=peak_depth, peak_velocity_m_s=peak_velocity, last_velocity_m_s=last_velocity,
        peak_outlet_discharge_m3_s=peak_q, time_of_peak_outlet_s=peak_t, max_cfl=max_cfl, limited_cells_total=lim_cells,
        limited_volume_total_m3=lim_volume, n_accepted_steps=n_steps, n_rejected_attempts=n_rejected,
        rejections=tuple(rejections), min_accepted_dt_s=dt_min, max_accepted_dt_s=dt_max, snapshots=snapshots,
        first_step_wall_s=first_wall, first_step_cpu_s=first_cpu, remaining_wall_s=remaining_wall,
        remaining_cpu_s=remaining_cpu, next_dt_cap_s=dt_cap)
