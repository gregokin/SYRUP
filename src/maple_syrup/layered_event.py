"""Opt-in bounded, fixed-bed water-only episodes around the genuine layered step.

Run time is a common relative-seconds axis, separate from calendar metadata.
Only rain and numerical recession occur here: E/T/growth are always zero.
Rainfall edges are transformed once, never inferred by subtracting an origin
at each step. Completion preserves the entire physical state and wetting front.
There is no dry finalizer, sediment operation, scheduler publication or disk
checkpoint in this module. Every returned progress record is a proposal for the
caller to adopt whole. An arbitrary stopping target can alter the timestep mesh;
exact continuation means continuation on the same accepted physical mesh.

Candidate authored by Codex while Claude's explicit usage limit is active.
Independent Claude review and actual guarded CUDA qualification remain required.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from maple_syrup import layered_storm as ls
from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.rainfall import RainfallField, RainfallSchedule, rainfall_field
from maple_syrup.surface_routing import HydraulicStepRejected


class LayeredEventError(ValueError):
    """Invalid input or inconsistent continuation; never an adaptive retry."""


def _real(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        value = ls._real(value, name)
    except (ls.LayeredStormError, OverflowError) as exc:
        raise LayeredEventError(f"invalid {name}: {exc}") from exc
    if value < 0.0 or (positive and value == 0.0):
        raise LayeredEventError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise LayeredEventError(f"{name} must be a non-Boolean integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class LayeredEventControl:
    max_dt_s: float = 1.0
    min_dt_s: float = 1.0 / 1024.0
    max_retries: int = 12
    max_steps: int = 100_000
    max_recession_s: float = 86_400.0

    def validated(self):
        _real(self.max_dt_s, "max_dt_s", positive=True)
        _real(self.min_dt_s, "min_dt_s", positive=True)
        _real(self.max_recession_s, "max_recession_s", positive=True)
        _integer(self.max_retries, "max_retries", minimum=1)
        _integer(self.max_steps, "max_steps", minimum=1)
        if self.min_dt_s > self.max_dt_s:
            raise LayeredEventError("min_dt_s exceeds max_dt_s")
        return self


@dataclass(frozen=True)
class LayeredQuietPolicy:
    """Existing pond flow-end physical gates, distinct from conservation bounds."""
    hold_s: float = 60.0
    max_depth_change_m: float = 1e-8
    max_unit_discharge_m2_s: float = 1e-10
    outlet_discharge_m3_s: float = 1e-9
    max_excess_depth_m: float = 1e-8
    excess_volume_m3: float = 1e-6

    def validated(self):
        for name, value in asdict(self).items():
            _real(value, name, positive=True)
        return self


@dataclass(frozen=True)
class QuietProgress:
    since_s: float | None = None
    window_depth_change_m: float = 0.0


@dataclass(frozen=True, eq=False)
class LayeredEventState:
    water: ls.LayeredWaterState
    event_sha256: str
    anchor_surface_m3: float
    anchor_soil_m3: float
    anchor_ledger: ls.LayeredWaterLedger
    start_generation: int
    next_dt_cap_s: float
    forcing_index: int
    accepted_steps: int = 0
    rejected_attempts: int = 0
    quiet: QuietProgress = QuietProgress()
    completed: bool = False


@dataclass(frozen=True)
class LayeredEventResult:
    state: LayeredEventState
    reason: str  # completed / target / max_steps / max_retries / min_dt / time_resolution / recession_limit
    accepted_steps: int
    rejected_attempts: int
    last_receipt: ls.LayeredStepReceipt | None

    @property
    def complete(self) -> bool:
        return self.state.completed


_SEAL = object()


class LayeredEvent:
    """Factory-only immutable prepared binding, with exclusive owned array use."""

    __slots__ = ("op", "_schedule", "_field", "event_origin_s", "absolute_edges_s", "control", "policy",
                 "_active", "_escape_m", "_held_z_m", "_outlet_x", "_outlet_y", "required_bytes", "sha256", "_seal")

    def __init__(self, seal, **fields):
        if seal is not _SEAL:
            raise LayeredEventError("LayeredEvent is constructed only by prepare_layered_event")
        for name in self.__slots__:
            if name == "_seal":
                continue
            object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise LayeredEventError("unexpected prepared event fields")
        object.__setattr__(self, "_seal", seal)

    def __setattr__(self, name, value):
        raise AttributeError("LayeredEvent is immutable")

    def __delattr__(self, name):
        raise AttributeError("LayeredEvent is immutable")

    @property
    def schedule(self):
        # The canonical host arrays use immutable byte buffers, so callers
        # cannot re-enable writeability and mutate the prepared forcing.
        return self._schedule

    @property
    def field(self):
        if self.op.xp is np:
            return self._field
        _check_event(self)
        _admit(self)
        return RainfallField(self._field.multiplier.copy(), self.op.xp)

    @property
    def active(self):
        if self.op.xp is np:
            return self._active
        _check_event(self)
        _admit(self)
        return self._active.copy()

    @property
    def escape_m(self):
        if self.op.xp is np:
            return self._escape_m
        _check_event(self)
        _admit(self)
        return self._escape_m.copy()

    @property
    def held_z_m(self):
        if self.op.xp is np:
            return self._held_z_m
        _check_event(self)
        _admit(self)
        return self._held_z_m.copy()

    @property
    def rainfall_end_s(self) -> float:
        return float(self.absolute_edges_s[-1])

    @property
    def deadline_s(self) -> float:
        return self.rainfall_end_s + float(self.control.max_recession_s)


def _index(event: LayeredEvent, t_s: float) -> int:
    return int(np.searchsorted(event.absolute_edges_s, t_s, side="right")) - 1


def _admit(event: LayeredEvent) -> None:
    ls._admit(event.op.column, event.op.shape, event.required_bytes, "event working set")


def _immutable_host(array):
    """Owned C-order bytes prevent setflags(write=True), unlike a flag alone."""
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=array.dtype).reshape(array.shape)


def prepare_layered_event(op: ls.LayeredStormOperator, schedule: RainfallSchedule, field: RainfallField, *,
                          event_origin_s: float, control: LayeredEventControl = LayeredEventControl(),
                          policy: LayeredQuietPolicy = LayeredQuietPolicy(),
                          host_available_bytes_fn: Any = None, host_reserve_bytes: int = 0) -> LayeredEvent:
    """Seal forcing and quiet geometry once; existing native material is unchanged.

    Additional bound includes four resident N-double maps, the Boolean mask,
    transformed host edges and twelve N-double temporaries for field validation,
    rainfall, quiet metrics and anchor reductions. No growing timestep history.
    Native/JIT/allocator exclusions and explicit reserves follow the pure owner.
    """
    ls._check_operator(op)
    host_reserve = ls.ss._checked_int(host_reserve_bytes, "host_reserve_bytes")
    if host_available_bytes_fn is not None and not callable(host_available_bytes_fn):
        raise LayeredEventError("host_available_bytes_fn must be callable or None")
    if op.backend != "cuda" and (host_available_bytes_fn is not None or host_reserve != 0):
        raise LayeredEventError("separate host staging controls are CUDA-only; CPU uses the column quota")
    if type(schedule) is not RainfallSchedule or type(field) is not RainfallField:
        raise LayeredEventError("use genuine RainfallSchedule and RainfallField factories")
    if type(control) is not LayeredEventControl or type(policy) is not LayeredQuietPolicy:
        raise LayeredEventError("control and policy must be genuine layered event records")
    control.validated()
    policy.validated()
    control = LayeredEventControl(float(control.max_dt_s), float(control.min_dt_s), int(control.max_retries),
                                   int(control.max_steps), float(control.max_recession_s))
    policy = LayeredQuietPolicy(**{key: float(value) for key, value in asdict(policy).items()})
    origin = _real(event_origin_s, "event_origin_s")
    if field.xp is not op.xp or field.shape != op.shape:
        raise LayeredEventError("rainfall field namespace/shape disagrees with the storm operator")
    ls.ss._check_dynamic(op.column, "rainfall multiplier", field.multiplier, op.shape)
    extra = ((16 * 8 + 1 + op.n_layers * 8) * op.n_cells
             + 4 * schedule.edges_s.nbytes + 2 * schedule.intensity_mm_per_h.nbytes + 2 * schedule.rate_m_per_s.nbytes
             + 4 * op.n_faces + 4096)
    # The call argument retains its original aggregate while the loop advances
    # a newer one. Native mobile/material is shared, so do not charge N*K twice.
    extra += ((op.n_layers + 3) * op.n_cells + op.n_faces) * 8
    if op.bed is not None:
        extra += op.n_cells * 8  # operator-bound original depth can predate this call
    ls._admit(op.column, op.shape, op.memory.required_bytes + extra, "event preparation")
    if op.backend == "cuda":
        # Host staging of this call: transformed edges, immutable schedule copies,
        # the field validation/hash chunk and the mask downloads of the topology
        # sources (escape/held depth and outlet masks are host metadata already,
        # copied to the device one at a time).
        ls._admit_host_build(2 * schedule.edges_s.nbytes + schedule.intensity_mm_per_h.nbytes
                              + schedule.rate_m_per_s.nbytes + min(ls.HASH_CHUNK_BYTES, field.multiplier.nbytes)
                              + 2 * op.n_cells * 8 + 2 * op.n_faces + op.n_cells + host_reserve + 4096,
                              host_available_bytes_fn or ls.ss.default_host_available_bytes)
    schedule = RainfallSchedule(schedule.edges_s, schedule.intensity_mm_per_h, schedule.provenance)
    for name in ("edges_s", "intensity_mm_per_h", "rate_m_per_s"):
        object.__setattr__(schedule, name, _immutable_host(getattr(schedule, name)))
    edges = _immutable_host(np.asarray(origin + schedule.edges_s, dtype=np.float64))
    if not np.all(np.isfinite(edges)) or not np.all(edges[1:] > edges[:-1]):
        raise LayeredEventError("event-relative forcing edges collapse or overflow on the physical clock")
    if not math.isfinite(float(edges[-1]) + float(control.max_recession_s)):
        raise LayeredEventError("recession deadline overflows the physical clock")
    if float(edges[-1]) + float(control.max_recession_s) <= float(edges[-1]):
        raise LayeredEventError("recession duration does not advance the physical clock")
    xp = op.xp
    active = xp.array(op.column.active.reshape(op.shape), copy=True)
    if ls._packet(xp, [xp.any((~active) & (field.multiplier != 0.0))])[0] != 0.0:
        raise LayeredEventError("rainfall field must be exactly zero on inactive cells")
    # Revalidate/copy the actual field once; public dataclass construction does
    # not itself prove finite nonnegative values or exclusive ownership.
    owned_field = rainfall_field(*op.shape, scale=field.multiplier, active_mask=active, xp=xp)
    from maple.core.backend import to_device

    escape = xp.array(to_device(op.topology.escape_level_m, xp), copy=True)
    held = xp.array(to_device(op.topology.source_bed_m, xp), copy=True)
    outlet_x = xp.array(to_device(op.topology.geometry.fx_type == 2, xp), copy=True)
    outlet_y = xp.array(to_device(op.topology.geometry.fy_type == 2, xp), copy=True)
    if xp is np:
        active, escape, held, outlet_x, outlet_y = map(_immutable_host, (active, escape, held, outlet_x, outlet_y))
        owned_field = RainfallField(_immutable_host(owned_field.multiplier), xp)
    h = hashlib.sha256(b"maple_syrup.layered_event.fixed_bed_water/1")
    for array in (schedule.edges_s, schedule.intensity_mm_per_h, edges, owned_field.multiplier):
        ls._stream_hash(h, array)
    h.update(json.dumps({"operator": op.sha256, "origin_s": origin,
                         "control": asdict(control), "policy": asdict(policy)},
                        sort_keys=True, allow_nan=False).encode())
    return LayeredEvent(_SEAL, op=op, _schedule=schedule, _field=owned_field, event_origin_s=origin,
                         absolute_edges_s=edges, control=control, policy=policy, _active=active, _escape_m=escape,
                         _held_z_m=held, _outlet_x=outlet_x, _outlet_y=outlet_y,
                         required_bytes=op.memory.required_bytes + extra, sha256=h.hexdigest())


def _check_event(event: LayeredEvent) -> None:
    if type(event) is not LayeredEvent or event._seal is not _SEAL:
        raise LayeredEventError("event must be prepared by prepare_layered_event")
    ls._check_operator(event.op)


def _volumes(event: LayeredEvent, water: ls.LayeredWaterState) -> tuple[float, float]:
    xp, area = event.op.xp, event.op.column.cell_area_m2
    packet = ls._packet(xp, [xp.sum(xp.where(event._active, water.surface_m, 0.0)) * area,
                             xp.sum(xp.where(event._active.reshape(1, -1), water.soil.water_m, 0.0)) * area])
    if not np.all(np.isfinite(packet)) or np.any(packet < 0.0):
        raise LayeredEventError("non-finite/negative physical anchor volumes")
    return float(packet[0]), float(packet[1])


def initial_event_state(event: LayeredEvent, water: ls.LayeredWaterState) -> LayeredEventState:
    _check_event(event)
    _admit(event)
    t = ls._check_state(event.op, water)
    if t != event.event_origin_s:
        raise LayeredEventError("a fresh event must start at its declared physical origin; use continuation to resume")
    ls._validate_inventories(event.op, water.soil, water.history)
    event.op.router.validate_state(ls.SurfaceState(t, water.surface_m, water.qx_m2_s, water.qy_m2_s))
    h, soil = _volumes(event, water)
    return LayeredEventState(water, event.sha256, h, soil, water.ledger, int(water.generation),
                             float(event.control.max_dt_s), _index(event, t))


def _check_state(event: LayeredEvent, state: LayeredEventState) -> None:
    _check_event(event)
    if type(state) is not LayeredEventState or state.event_sha256 != event.sha256:
        raise LayeredEventError("event state belongs to a different prepared event")
    t = ls._check_state(event.op, state.water)
    start = _integer(state.start_generation, "start_generation")
    steps = _integer(state.accepted_steps, "accepted_steps")
    _integer(state.rejected_attempts, "rejected_attempts")
    if int(state.water.generation) != start + steps:
        raise LayeredEventError("event step count disagrees with the physical generation")
    ls._check_ledger(state.anchor_ledger, start)
    for name in ("anchor_surface_m3", "anchor_soil_m3"):
        _real(getattr(state, name), name)
    if t < event.event_origin_s or t > event.deadline_s:
        raise LayeredEventError("event state clock lies outside its declared episode")
    if (isinstance(state.forcing_index, (bool, np.bool_)) or not isinstance(state.forcing_index, (int, np.integer))
            or state.forcing_index != _index(event, t)):
        raise LayeredEventError("forcing cursor disagrees with the physical clock")
    cap = _real(state.next_dt_cap_s, "next_dt_cap_s", positive=True)
    if cap < event.control.min_dt_s or cap > event.control.max_dt_s:
        raise LayeredEventError("continuation cap lies outside this event's controls")
    if type(state.completed) is not bool or type(state.quiet) is not QuietProgress:
        raise LayeredEventError("invalid event completion/quiet record")
    _real(state.quiet.window_depth_change_m, "quiet.window_depth_change_m")
    if state.quiet.since_s is not None:
        since = _real(state.quiet.since_s, "quiet.since_s")
        if since < event.rainfall_end_s or since > t:
            raise LayeredEventError("quiet window is outside post-rain physical time")
    elif state.quiet.window_depth_change_m != 0.0 or state.completed:
        raise LayeredEventError("quiet progress requires a real post-rain start")
    if state.quiet.window_depth_change_m > event.policy.max_depth_change_m:
        raise LayeredEventError("quiet window exceeds the physical change gate")
    if state.completed and t - float(state.quiet.since_s) < event.policy.hold_s:
        raise LayeredEventError("completion has not met the physical quiet hold")
    elapsed = float(state.water.ledger.elapsed_s) - float(state.anchor_ledger.elapsed_s)
    bound_s = 8.0 * (steps + 2) * np.finfo(np.float64).eps * max(t, event.event_origin_s, abs(elapsed))
    if abs(elapsed - (t - event.event_origin_s)) > bound_s:
        raise LayeredEventError("ledger elapsed time disagrees with the event origin")
    for name in ("rain_m3", "preferential_intake_m3", "matrix_intake_m3", "surface_return_m3",
                 "drainage_m3", "export_m3", "global_residual_bound_m3"):
        if getattr(state.water.ledger, name) < getattr(state.anchor_ledger, name):
            raise LayeredEventError(f"cumulative ledger {name} predates its event anchor")


def _window_bound(event: LayeredEvent, state: LayeredEventState, h: float, soil: float,
                  rain: float, export: float, drain: float) -> float:
    """Anchored closure bound, MAPLE `volume_roundoff_bound_m3` only (derivation: layered_event.md).

    1. stage bound: the bounds the owner published for the window's own steps
       (current minus anchor cumulative; monotone, validated by `_check_state`);
    2. rounding of NEW additions into each LARGE prior cumulative (rain, export,
       drainage and the cumulative bound itself): n_steps_in_event + 1 operations
       at scale max(|current|, |anchor|) of that cumulative, so a tiny new flux
       added to a huge earlier total is still bounded;
    3. fresh physical reductions and the final algebra: 2 (L + 1) N + 12
       operations (two N-term surface sums, two N L-term soil sums, differences)
       at the largest actual inventory or window flux.
    """
    ledger, anchor = state.water.ledger, state.anchor_ledger
    steps = int(state.accepted_steps) + 1
    total = float(ledger.global_residual_bound_m3) - float(anchor.global_residual_bound_m3)
    for name in ("rain_m3", "export_m3", "drainage_m3", "global_residual_bound_m3"):
        scale = max(abs(float(getattr(ledger, name))), abs(float(getattr(anchor, name))))
        total += volume_roundoff_bound_m3(steps, scale)
    fresh = 2 * (event.op.n_layers + 1) * event.op.n_cells + 12
    total += volume_roundoff_bound_m3(fresh, max(h, soil, float(state.anchor_surface_m3),
                                                 float(state.anchor_soil_m3), rain, export, drain))
    if not math.isfinite(total) or total < 0.0:
        raise LayeredEventError("non-finite anchored closure bound")
    return total


def audit_event(event: LayeredEvent, state: LayeredEventState) -> dict:
    """Actual inventories plus anchored external fluxes; never trust a saved pass bit."""
    _check_state(event, state)
    _admit(event)
    h, soil = _volumes(event, state.water)
    ledger, anchor = state.water.ledger, state.anchor_ledger
    rain = float(ledger.rain_m3) - float(anchor.rain_m3)
    export = float(ledger.export_m3) - float(anchor.export_m3)
    drain = float(ledger.drainage_m3) - float(anchor.drainage_m3)
    anchor_h, anchor_soil = float(state.anchor_surface_m3), float(state.anchor_soil_m3)
    residual = ((h - anchor_h) + (soil - anchor_soil)) + export + drain - rain
    bound = _window_bound(event, state, h, soil, rain, export, drain)
    if not math.isfinite(residual) or abs(residual) > bound:
        raise LayeredEventError(f"anchored event water closure failed: {residual} m3 against {bound} m3")
    if state.completed:
        ls._validate_inventories(event.op, state.water.soil, state.water.history)
        event.op.router.validate_state(ls.SurfaceState(float(state.water.t_s), state.water.surface_m,
                                                       state.water.qx_m2_s, state.water.qy_m2_s))
        if not _instant_quiet(event, _hydraulic_packet(event, state.water)):
            raise LayeredEventError("completed state fails the actual instantaneous hydraulic quiet gates")
    return dict(surface_m3=h, soil_m3=soil, rain_m3=rain, export_m3=export, drainage_m3=drain,
                residual_m3=residual, bound_m3=bound, evaporation_m3=0.0)


def _hydraulic_packet(event, water, old=None):
    xp = event.op.xp
    excess = xp.where(event._active, xp.minimum(xp.maximum(event._held_z_m + water.surface_m - event._escape_m, 0.0),
                                               water.surface_m), 0.0)
    change = (xp.max(xp.where(event._active, xp.abs(water.surface_m - old.surface_m), 0.0))
              if old is not None else xp.zeros((), dtype=np.float64))
    outlet = event.op.dx_m * (xp.sum(xp.where(event._outlet_x, xp.abs(water.qx_m2_s), 0.0))
                               + xp.sum(xp.where(event._outlet_y, xp.abs(water.qy_m2_s), 0.0)))
    packet = ls._packet(xp, [change,
                             xp.maximum(xp.max(xp.abs(water.qx_m2_s)), xp.max(xp.abs(water.qy_m2_s))),
                             xp.max(excess), xp.sum(excess) * event.op.column.cell_area_m2, outlet])
    if not np.all(np.isfinite(packet)):
        raise LayeredEventError("non-finite hydraulic quiet diagnostic")
    return packet


def _instant_quiet(event, packet):
    pol = event.policy
    return bool(packet[1] <= pol.max_unit_discharge_m2_s and packet[2] <= pol.max_excess_depth_m
                and packet[3] <= pol.excess_volume_m3 and packet[4] <= pol.outlet_discharge_m3_s)


def _quiet(event: LayeredEvent, before: LayeredEventState, candidate: ls.LayeredStepCandidate) -> QuietProgress:
    pol = event.policy
    water, old = candidate.state, before.water
    packet = _hydraulic_packet(event, water, old)
    qualifies = (old.t_s >= event.rainfall_end_s and _instant_quiet(event, packet)
                 and candidate.receipt.export_m3 / candidate.dt_s <= pol.outlet_discharge_m3_s
                 and candidate.receipt.outlet_discharge_m3_s <= pol.outlet_discharge_m3_s)
    if not qualifies:
        return QuietProgress()
    change = float(before.quiet.window_depth_change_m) + float(packet[0])
    if change > pol.max_depth_change_m:
        return QuietProgress()
    since = float(old.t_s) if before.quiet.since_s is None else float(before.quiet.since_s)
    return QuietProgress(since, change)


def evolve_layered_event(event: LayeredEvent, accepted: LayeredEventState, *, until_s: float,
                         max_accepted_steps: int | None = None, stop_when_quiet: bool = True) -> LayeredEventResult:
    """Advance bounded work, adopting each complete pure candidate locally.

    Only HydraulicStepRejected halves dt, rerunning every vertical/lateral stage
    from the same original accepted water. All other failures propagate. A bound
    returns the last accepted physical state as incomplete, not a dry handoff.
    No list of stage arrays or full candidate is retained between accepted steps.
    """
    audit_event(event, accepted)
    target = _real(until_s, "until_s")
    if target < accepted.water.t_s:
        raise LayeredEventError("until_s precedes the accepted physical clock")
    if type(stop_when_quiet) is not bool:
        raise LayeredEventError("stop_when_quiet must be bool")
    budget = (event.control.max_steps if max_accepted_steps is None
              else _integer(max_accepted_steps, "max_accepted_steps", minimum=1))
    if budget > event.control.max_steps:
        raise LayeredEventError("call step budget cannot exceed the prepared event's bound")
    state = accepted
    n = rejected = 0
    last = None
    end = min(target, event.deadline_s)

    def result(reason):
        audit_event(event, state)
        return LayeredEventResult(state, reason, n, rejected, last)

    if state.completed:
        return result("completed")
    if (stop_when_quiet and state.quiet.since_s is not None
            and float(state.water.t_s) - float(state.quiet.since_s) >= event.policy.hold_s
            and _instant_quiet(event, _hydraulic_packet(event, state.water))):
        state = replace(state, completed=True)
        return result("completed")
    while state.water.t_s < end:
        if n >= budget:
            return result("max_steps")
        _admit(event)  # before allocating the spatial rain rate
        t = float(state.water.t_s)
        k = _index(event, t)
        edge = float(event.absolute_edges_s[k + 1]) if k + 1 < len(event.absolute_edges_s) else math.inf
        boundary = min(end, edge)
        cap0 = float(state.next_dt_cap_s)
        dt = min(boundary - t, cap0)
        # A boundary subtraction can round. Never overstep or restamp either
        # physical clock; a representable smaller dt will leave a final slice.
        if t + dt > boundary:
            dt = math.nextafter(dt, 0.0)
        if not t + dt > t:
            return result("time_resolution")
        rate = float(event.schedule.rate_m_per_s[k]) if 0 <= k < len(event.schedule.rate_m_per_s) else 0.0
        rain = event._field.apply(rate)
        attempts = 0
        initial_dt = dt
        floor = min(float(event.control.min_dt_s), initial_dt)
        while True:
            try:
                candidate = ls.layered_step(event.op, state.water, rain, dt)
                break
            except HydraulicStepRejected:
                attempts += 1
                rejected += 1
                state = replace(state, rejected_attempts=int(state.rejected_attempts) + 1)
                if attempts >= event.control.max_retries:
                    return result("max_retries")
                if dt / 2.0 < floor:
                    return result("min_dt")
                dt /= 2.0
        quiet = _quiet(event, state, candidate)
        if attempts:
            cap = max(float(event.control.min_dt_s), dt)
        elif dt >= cap0:
            cap = min(float(event.control.max_dt_s), 2.0 * cap0)
        else:
            cap = cap0
        completed = bool(stop_when_quiet and quiet.since_s is not None
                         and candidate.state.t_s - quiet.since_s >= event.policy.hold_s)
        last = candidate.receipt  # bounded scalars only, not stage arrays
        state = replace(state, water=candidate.state, next_dt_cap_s=cap,
                        forcing_index=_index(event, candidate.state.t_s), accepted_steps=int(state.accepted_steps) + 1,
                        quiet=quiet, completed=completed)
        n += 1
        del candidate, rain
        if completed:
            return result("completed")
    return result("recession_limit" if state.water.t_s >= event.deadline_s else "target")
