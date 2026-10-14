"""Completion and conservative dry handoff around the existing sediment solver.

Only accepted steps advance the quiet interval. Checkpoint boundaries preserve
that solver's original partition; neither checkpoint I/O nor pausing commits
terrain. The terminal reset is an external inter-event water removal, not ET.

Backends (CW-C1): every physical array stays in the event state's namespace
(`state.graph.xp`, NumPy or CuPy) through the terminal settlement, the forced
final commit and the dry handoff. The completion criteria and the water budget
are namespace reductions read as ONE compact host packet per accepted step /
audit (`event_backend.host_packet`); the only other host operands are the
small per-class terminal inventories of the MAPLE conservation audit and the
reported accounting scalars. The NumPy path computes exactly the reductions it
computed before (the packet merely batches the reads). A CUDA control passes
its caller-owned `CudaHydraulics` lifecycle through to the solver; the
terminal forced commit (CW-B104: it reroutes only when net-elevation driven)
does not rebuild that context (the lifecycle re-prepares it, counted, at the
next water step of a continuation; nothing irreconstructible is discarded).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from maple_syrup.conservation import reservoir_bound_kg, volume_roundoff_bound_m3
from maple_syrup.event_backend import host_packet
from maple_syrup.sediment_bed import (
    BedIntegrationError,
    apply_bed_demand,
    bed_already_committed,
    bed_inventory,
    publish_bed_storage,
    validate_deep_storage,
    with_water,
)
from maple_syrup.sediment_event import (
    CudaHydraulics,
    SedimentEventControl,
    SedimentEventError,
    SedimentEventResult,
    SedimentEventState,
    _commit_and_reroute,
    canonical_phase_for,
    evolve_sediment_event,
    fold_commit_exchange_history,
    phase_for_network,
    routing_gate_m,
)
from maple_syrup.storm import initial_state, plan_boundaries


class IncompleteEventError(SedimentEventError):
    """A limit is not physical completion. No dry state is returned."""


@dataclass(frozen=True)
class CompletionPolicy:
    max_depth_m: float = 1e-8
    surface_volume_m3: float = 1e-6
    max_unit_discharge_m2_s: float = 1e-10
    outlet_discharge_m3_s: float = 1e-9
    hold_s: float = 60.0
    # None uses the actual MAPLE per-element physical significance threshold.
    # This is NOT the floating-point conservation bound.
    mobile_per_cell_class_kg: float | None = None
    # CW-C2 explicit option (default False, LAST so existing constructions / bundles are unchanged): flow may be
    # declared ceased while water is still ponded in the graph's RECOGNISED terminal-storage cells (opt-in pits /
    # flat sinks, zero conveyance). Only the depth and surface-volume criteria exclude those cells; the discharge
    # criteria, the mobile-mass criterion (every cell), the hold and the rain-finished rule are unchanged. The
    # ponded water is physical storage until the external dry reset removes and accounts for it.
    terminal_storage_water: bool = False
    # CW-B51 explicit option (default False, LAST so existing constructions / bundles / identities are unchanged): the
    # first-model dry-again assumption at flow end. When rain has ended and EVERY hydraulic quiet gate (depth,
    # surface volume, unit and outlet discharge; terminal-storage policy unchanged) has held for the unchanged
    # `hold_s`, the remaining finite mobile load need not be below the physical significance threshold: the existing
    # terminal transaction deposits it through MAPLE. The mobile load is still validated (finite, nonnegative) and
    # reported. This is a modeling assumption about the end of the event, not a mass tolerance and not a claim that
    # the legacy physics deposits at that instant; transport during the storm is unchanged.
    deposit_mobile_at_flow_end: bool = False

    def validated(self):
        for name in ('max_depth_m', 'surface_volume_m3', 'max_unit_discharge_m2_s',
                     'outlet_discharge_m3_s', 'hold_s', 'mobile_per_cell_class_kg'):
            value = getattr(self, name)
            if value is None and name == 'mobile_per_cell_class_kg':
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise SedimentEventError(f'{name} must be finite and positive')
        if not isinstance(self.terminal_storage_water, bool):
            raise SedimentEventError('terminal_storage_water must be bool')
        if not isinstance(self.deposit_mobile_at_flow_end, bool):
            raise SedimentEventError('deposit_mobile_at_flow_end must be bool')
        return self


FLOW_END_SETTLEMENT_ASSUMPTION = (
    'explicit first-model dry-again assumption (deposit_mobile_at_flow_end): after rain ended and every hydraulic '
    'quiet gate held continuously for hold_s, the remaining finite mobile load is deposited in place through the '
    'genuine MAPLE terminal deposition; not a mass tolerance, not legacy physics at that instant')


def _outlet_mask(graph):
    """The outlet mask in the graph's namespace: the host evidence mask on NumPy (unchanged selection), the
    device runtime mask reshaped on CuPy (no host index array is applied to a device array)."""
    return graph.outlet if graph.xp is np else graph.outlet_flat.reshape(graph.shape)


def quiet_metrics(state, context, policy, rainfall_end_s):
    """Event-boundary reductions in the state's namespace, read as ONE compact host packet; no implicit
    device-to-host conversion. Same thresholds, same values as the host computation."""
    xp = state.graph.xp
    h, q, soil = state.storm.depth_m, state.storm.discharge_m2_s, state.storm.soil_water_m
    mobile = state.bed.water.mobile_mass_by_cell_class_kg
    threshold = context.mass_resolution_kg if policy.mobile_per_cell_class_kg is None else policy.mobile_per_cell_class_kg
    area = state.graph.dx_m ** 2
    finite = (xp.all(xp.isfinite(h)) & xp.all(xp.isfinite(q)) & xp.all(xp.isfinite(mobile))
              & xp.all(xp.isfinite(soil)))
    negative = xp.any(h < 0) | xp.any(q < 0) | xp.any(mobile < 0) | xp.any(soil < 0)
    scalars = [h.max(), h.sum() * area, q.max(), q[_outlet_mask(state.graph)].sum() * state.graph.dx_m,
               mobile.max(), finite, negative]
    # CW-C2: with the explicit policy option and a grid that carries recognised terminal-storage cells, the depth /
    # volume criteria are evaluated on the MOVING cells only; the totals (physical storage) are still reported and
    # the water budget always uses them. Without the option (or without terminal cells) the packet, the values and
    # the criteria are exactly the strict ones.
    terminal = getattr(state.grid, 'terminal', None)
    excluding = bool(getattr(policy, 'terminal_storage_water', False)) and terminal is not None
    if excluding:
        moving_h = xp.where(terminal, 0.0, h)
        terminal_h = xp.where(terminal, h, 0.0)
        scalars += [moving_h.max(), moving_h.sum() * area, terminal_h.max(), terminal_h.sum() * area,
                    xp.sum(terminal)]
    n_scalars = len(scalars)
    packet = host_packet(xp, scalars, mobile.sum(axis=(0, 1)))
    if packet[5] != 1.0 or packet[6] != 0.0:
        raise SedimentEventError('invalid physical state in completion check')
    values = {
        'max_depth_m': float(packet[0]),
        'surface_volume_m3': float(packet[1]),
        'max_unit_discharge_m2_s': float(packet[2]),
        'outlet_discharge_m3_s': float(packet[3]),
        'max_mobile_cell_class_kg': float(packet[4]),
        'mobile_by_class_kg': packet[n_scalars:].tolist(),
        'mobile_per_element_limit_kg': float(threshold),
        'rainfall_finished': bool(state.t_s >= rainfall_end_s),
    }
    depth_criterion, volume_criterion = values['max_depth_m'], values['surface_volume_m3']
    if excluding:
        values.update({
            'moving_max_depth_m': float(packet[7]), 'moving_surface_volume_m3': float(packet[8]),
            'terminal_max_depth_m': float(packet[9]), 'terminal_surface_volume_m3': float(packet[10]),
            'terminal_storage_cells': int(packet[11]),
            'terminal_storage_water_excluded': 'depth and surface-volume criteria on moving cells only; totals '
                                               'reported; discharge, mobile, hold and rain-finished unchanged',
        })
        depth_criterion, volume_criterion = values['moving_max_depth_m'], values['moving_surface_volume_m3']
    mobile_below = values['max_mobile_cell_class_kg'] <= threshold
    # CW-B51: with the explicit flow-end option the mobile gate alone is not required (the load above was still
    # validated finite / nonnegative and is reported in full); every hydraulic gate and rain-off are unchanged.
    waive_mobile = bool(getattr(policy, 'deposit_mobile_at_flow_end', False))
    if waive_mobile:
        values.update({'mobile_gate': 'waived by deposit_mobile_at_flow_end (deposited by the terminal transaction)',
                       'mobile_below_significance': bool(mobile_below)})
    qualifies = values['rainfall_finished'] and all((
        depth_criterion <= policy.max_depth_m,
        volume_criterion <= policy.surface_volume_m3,
        values['max_unit_discharge_m2_s'] <= policy.max_unit_discharge_m2_s,
        values['outlet_discharge_m3_s'] <= policy.outlet_discharge_m3_s,
        waive_mobile or mobile_below,
    ))
    return bool(qualifies), values


@dataclass(frozen=True)
class QuietProgress:
    since_s: float | None = None
    last_t_s: float | None = None
    ready: bool = False

    def observe(self, state, context, policy, rainfall_end_s):
        if self.last_t_s is not None and state.t_s <= self.last_t_s:
            raise SedimentEventError('quiet monitor requires strictly increasing accepted times')
        qualifies, _ = quiet_metrics(state, context, policy, rainfall_end_s)
        since = (state.t_s if self.since_s is None else self.since_s) if qualifies else None
        ready = since is not None and state.t_s - since >= policy.hold_s
        return QuietProgress(since, state.t_s, bool(ready))


@dataclass(frozen=True)
class EventProgress:
    result: SedimentEventResult
    policy: CompletionPolicy
    quiet: QuietProgress
    control: SedimentEventControl
    planned_boundaries: np.ndarray
    origin_t_s: float
    initial_surface_m3: float
    initial_soil_m3: float
    rainfall_end_s: float
    report_every_s: float
    max_report_rows: int
    status: str = 'wet'


@dataclass(frozen=True)
class DryHandoff:
    # Pre-reset physical result includes terminal settling and final commit.
    progress: EventProgress
    dry_state: SedimentEventState
    pre_reset_surface_m: Any
    pre_reset_soil_m: Any
    terminal_deposition_kg: Any
    accounting: dict[str, Any]
    status: str = 'complete'


def complete_event(state, context, column, field, schedule, vegetation, sediment, *,
                   max_end_s, control, policy=None, report_every_s=60.,
                   max_report_rows=100_000, continuation=None, checkpoint_callback=None,
                   hydraulics=None):
    """Return a dry handoff, or wet EventProgress when a callback requests pause.

    callback(progress) runs only at original reporting/forcing boundaries,
    synchronously. Returning True requests pause; it must not retain mutable
    accumulator references when returning False (write the checkpoint now).
    On a duration/step limit, raise IncompleteEventError; earlier published
    checkpoints remain valid. No terminal action has run on that path.
    `hydraulics` (CUDA control only) is the caller-owned `CudaHydraulics`
    lifecycle handed to the solver (its context is reused / rebuilt there).
    """
    policy = CompletionPolicy() if policy is None else policy
    policy.validated()
    control.validated()
    if not control.commit:
        raise SedimentEventError('complete event requires actual MAPLE commits')
    control = replace(control, force_final_commit=False)
    if hydraulics is not None and not isinstance(hydraulics, CudaHydraulics):
        raise SedimentEventError('hydraulics must be a CudaHydraulics lifecycle (or None)')
    if hydraulics is not None and not control.cuda:
        raise SedimentEventError('hydraulics is only meaningful with storm implementation \'cuda\'')
    if continuation is not None:
        if not isinstance(continuation, EventProgress) or continuation.status != 'wet':
            raise SedimentEventError('only a wet checkpoint can resume; completed reset cannot replay')
        if continuation.policy != policy or continuation.control != control:
            raise SedimentEventError('restart stopping or numerical controls changed')
        if continuation.report_every_s != report_every_s or continuation.max_report_rows != max_report_rows:
            raise SedimentEventError('restart reporting controls changed')
        plan = continuation.planned_boundaries
        if plan[-1] != max_end_s or continuation.rainfall_end_s != schedule.end_s:
            raise SedimentEventError('restart duration or forcing changed')
        if state is not continuation.result.state or state.t_s not in plan:
            raise SedimentEventError('restart must use the checkpoint state at an original boundary')
        origin = continuation.origin_t_s
        surface0, soil0 = continuation.initial_surface_m3, continuation.initial_soil_m3
        quiet = continuation.quiet
        previous = continuation.result
    else:
        plan = plan_boundaries(schedule, state.t_s, max_end_s, report_every_s, max_report_rows=max_report_rows)
        origin = state.t_s
        area = state.graph.dx_m ** 2
        initial = host_packet(state.graph.xp, [state.storm.depth_m.sum() * area, state.storm.soil_water_m.sum() * area])
        surface0, soil0 = float(initial[0]), float(initial[1])
        quiet = QuietProgress()
        previous = None
    if state.t_s >= max_end_s:
        raise IncompleteEventError('maximum event duration reached without a dry handoff')
    remaining_plan = plan[plan > state.t_s]
    # CW-B30: every check above has run and every value the rest of this function needs (plan, origin, initial
    # budgets, quiet monitor) is extracted. Hand the start state and the prior result to the solver through a
    # single-use holder, so this frame no longer pins them while the solver advances its own state (and, on the
    # success path, while the terminal transaction runs). Only this frame's references are dropped; a caller's own
    # references and arrays are untouched.
    handoff = [state, previous]
    del state, previous, continuation

    def observe(candidate):
        nonlocal quiet
        quiet = quiet.observe(candidate, context, policy, schedule.end_s)
        return quiet.ready

    def progress(result):
        return EventProgress(result, policy, quiet, control, plan, origin, surface0, soil0,
                             schedule.end_s, report_every_s, max_report_rows)

    paused = False
    # CW-B102: the quiet monitor reads only resident water / storm / graph fields and keeps floats. Under the opt-in
    # 'boundary' deep publication it observes the (possibly staged) accepted state through the solver's PRIVATE path,
    # so the production quiet path never forces per-step deep writes; the default 'step' path is unchanged (the
    # public hook, which always receives published states).
    observer = ({'_staged_accepted_observer': observe} if control.deep_publication == 'boundary'
                else {'on_accepted_step': observe})

    def boundary(result):
        nonlocal paused
        if checkpoint_callback is not None:
            paused = bool(checkpoint_callback(progress(result)))
        return paused

    try:
        result = evolve_sediment_event(
            handoff.pop(0), context, column, field, schedule, vegetation, sediment, max_end_s, control,
            report_every_s=report_every_s, max_report_rows=max_report_rows,
            continuation=handoff.pop(), planned_boundaries=remaining_plan, on_boundary=boundary,
            hydraulics=hydraulics, **observer)
    except SedimentEventError as exc:
        if 'max_steps =' in str(exc):
            raise IncompleteEventError(str(exc)) from exc
        raise
    item = progress(result)
    if quiet.ready:
        return finalize_event(item, context, column)
    if result.state.t_s >= max_end_s:
        raise IncompleteEventError('maximum event duration reached; water and sediment were not reset')
    if paused:
        return item
    raise IncompleteEventError('solver returned before a completion or checkpoint boundary')


def water_budget(progress):
    r = progress.result
    xp = r.state.graph.xp
    area = r.state.graph.dx_m ** 2
    packet = host_packet(xp, [r.cumulative_rain_m.sum() * area, r.state.storm.depth_m.sum() * area,
                              r.state.storm.soil_water_m.sum() * area, r.cumulative_drainage_m.sum() * area,
                              r.cumulative_export_m3])
    rain, surface, soil, drainage, export = (float(v) for v in packet)
    residual = surface + soil + drainage + export - progress.initial_surface_m3 - progress.initial_soil_m3 - rain
    scale = max(surface, soil, drainage, export, progress.initial_surface_m3, progress.initial_soil_m3, rain)
    # Four accumulated grids plus reservoir reductions and scalar arithmetic.
    n_cells = int(np.prod(r.state.graph.shape))
    bound = volume_roundoff_bound_m3(4 * n_cells * max(r.n_accepted_steps, 1) + 7, scale)
    return {'rain_m3': rain, 'surface_initial_m3': progress.initial_surface_m3,
            'soil_initial_m3': progress.initial_soil_m3, 'surface_final_m3': surface,
            'soil_final_m3': soil, 'drainage_m3': drainage, 'export_m3': export,
            'residual_m3': residual, 'tolerance_m3': bound, 'closed': abs(residual) <= bound}


def finalize_event(progress, context, column):
    """Pure terminal transaction; all failure paths preserve the wet input.

    Terminal settlement is a genuine MAPLE deposition call of the whole
    remaining mobile pool, the final commit a genuine MAPLE commit; nothing
    mobile and no budget residual is erased. The surface and soil water removed
    by the external dry reset are reported, and the pre-reset arrays are kept
    on the handoff for audit."""
    from maple.core.backend import to_bool, to_float, to_host
    from maple.surface.topographic_commit.commit import (
        ledger_is_empty,
    )
    from maple.water import WaterProcessDemand

    r, state = progress.result, progress.result.state
    xp = state.graph.xp
    policy = progress.policy.validated()
    qualifies, before_metrics = quiet_metrics(state, context, policy, progress.rainfall_end_s)
    if (progress.status != 'wet' or not progress.quiet.ready or not qualifies
            or progress.quiet.since_s is None or progress.quiet.last_t_s != state.t_s
            or state.t_s - progress.quiet.since_s < policy.hold_s):
        raise SedimentEventError('terminal reset requires a completed continuous quiet interval')
    if not r.closure()['closed'] or not r.closure()['request_reconciled'] or not water_budget(progress)['closed']:
        raise SedimentEventError('pre-terminal event budgets do not close')
    try:  # CW-B108: the final publication's storage option, refused before the terminal transaction
        validate_deep_storage(context, state.bed)
    except BedIntegrationError as exc:
        raise SedimentEventError(str(exc)) from None
    pool = state.bed.water.mobile_mass_by_cell_class_kg
    initial_total = bed_inventory(state.bed) + pool.sum(axis=(0, 1))
    deposited = xp.zeros_like(pool)
    if to_bool(xp.any(pool)):
        bed, transfer = apply_bed_demand(state.bed, context, WaterProcessDemand(xp.zeros_like(pool), pool))
        if to_bool(xp.any(bed.water.mobile_mass_by_cell_class_kg != 0)):
            raise SedimentEventError('MAPLE refused terminal mobile deposition; no inventory erased')
        deposited = transfer.deposition_by_cell_class_kg
        bc = {k: xp.array(v, copy=True) for k, v in r.by_class.items()}
        bc['deposition_requested'] += pool.sum(axis=(0, 1))
        bc['deposition_actual'] += transfer.deposited_mass_by_class_kg
        bc['deposition_unmet'] += pool.sum(axis=(0, 1)) - transfer.deposited_mass_by_class_kg
        bc['deposit_numerical_residual'] += transfer.numerical_residual_by_class_kg
        # The whole pool went back to the bed: the phase partition is canonical again.
        state = replace(state, bed=bed, phase=canonical_phase_for(state))
        r = replace(r, state=state, by_class=bc, cumulative_deposition_kg=r.cumulative_deposition_kg + deposited,
                    n_maple_water_calls=r.n_maple_water_calls + 1,
                    numerical_residual_scalar_abs_kg=r.numerical_residual_scalar_abs_kg + abs(transfer.numerical_residual_scalar_kg))
    bed = state.bed
    already = bed_already_committed(bed, context, state.t_s)
    if not already:
        # CW-B104: the terminal forced commit is an accounting / geometry commit (reason 'final_forced'); the routing is
        # refreshed ONLY if the post-commit net displacement from the held routing source reaches the threshold.
        gate = routing_gate_m(bed, context, state.graph, state.routing_source_elevation_m)
        commit = _commit_and_reroute(bed, context, state.graph, state.terrain, state.storm, state.t_s, force=True,
                                     routing_source_m=state.routing_source_elevation_m, gate_m=gate)
        if commit is None:
            raise SedimentEventError('final terrain commit was not performed')
        # CW-D2: the terminal forced commit's pre-clear receipt (which includes the terminal MAPLE deposition
        # above and any avalanche fold-in) is folded into the event history exactly like every other commit,
        # BEFORE any local state is advanced: a refused receipt leaves every local unchanged and nothing returned.
        history = (None if r.exchange_history is None
                   else fold_commit_exchange_history(r.exchange_history, commit.maple_result, xp))
        if commit.rerouted:
            # The (canonical) phase follows the refreshed network: dense carried over, CSR explicitly rebound.
            state = replace(state, bed=commit.bed, graph=commit.graph, network=commit.network,
                            grid=commit.grid, storm=commit.storm, phase=phase_for_network(state.phase, commit.network),
                            routing_source_elevation_m=commit.routing_source_m)
            routing = {'n_reroutes': r.n_reroutes + 1, 'routing_source_origin': 'net_reroute',
                       'routing_source_t_s': float(state.t_s)}
        else:
            state = replace(state, bed=commit.bed)
            routing = {'max_unrouted_net_displacement_m': max(r.max_unrouted_net_displacement_m, gate,
                                                              commit.record['net_displacement_m'])}
        routing['n_routing_gate_evaluations'] = r.n_routing_gate_evaluations + 1
        log = r.commit_log + ((commit.record,) if len(r.commit_log) < progress.control.max_commit_log else ())
        r = replace(r, state=state, n_commits=r.n_commits + 1, n_forced_commits=r.n_forced_commits + 1,
                    n_graph_changes=r.n_graph_changes + int(commit.record['graph_changed']),
                    rerouted_cells_total=r.rerouted_cells_total + commit.record['rerouted_cells'], commit_log=log,
                    ledger_process_totals_reset_kg=r.ledger_process_totals_reset_kg + commit.maple_result.pre_commit_process_totals_kg,
                    committed_net_bed_change_kg=r.committed_net_bed_change_kg + commit.maple_result.pre_commit_pending_mass_by_class_kg,
                    exchange_history=history, **routing)
    if not ledger_is_empty(state.bed.ledger):
        raise SedimentEventError('terminal MAPLE ledger is not fully empty')
    qualifies, after_metrics = quiet_metrics(state, context, policy, progress.rainfall_end_s)
    if not qualifies:
        raise SedimentEventError('terminal terrain change invalidated completion; input remains wet')
    final_total = bed_inventory(state.bed) + state.bed.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1))
    # The MAPLE conservation audit works on the two small per-class host inventories (explicit, counted reads).
    initial_host = np.asarray(to_host(initial_total), dtype=np.float64)
    final_host = np.asarray(to_host(final_total), dtype=np.float64)
    terminal_bound = reservoir_bound_kg(1, initial_host, final_host)
    if np.any(np.abs(final_host - initial_host) > terminal_bound):
        raise SedimentEventError('terminal sediment transaction fails MAPLE conservation bound')
    if not r.closure()['closed'] or to_bool(xp.any(state.bed.water.mobile_mass_by_cell_class_kg)):
        raise SedimentEventError('terminal sediment closure or empty-mobile check failed')
    # Refresh terminal-row diagnostics after the real final commit/settlement.
    q = to_float(state.storm.discharge_m2_s[_outlet_mask(state.graph)].sum() * state.graph.dx_m)
    v = xp.sqrt(state.storm.depth_m) * state.graph.conveyance.reshape(state.graph.shape)
    hydro, sed_hydro = xp.array(r.hydrograph, copy=True), xp.array(r.sediment_hydrograph, copy=True)
    hydro[-1, 8] = q
    hydro[-1, 10] = to_float(v.max())
    deposition_actual = np.asarray(to_host(r.by_class['deposition_actual']), dtype=np.float64)
    updates = {'mobile_kg': 0.0, 'cumulative_deposition_kg': float(deposition_actual.sum()),
               'commit_count': float(state.bed.committed_topography.commit_count),
               'graph_changes': float(r.n_graph_changes)}
    for k in range(len(context.grain_classes.classes)):
        updates[f'mobile_c{k + 1}_kg'] = 0.0
        updates[f'cumulative_deposition_c{k + 1}_kg'] = float(deposition_actual[k])
    for name, value in updates.items():
        sed_hydro[-1, r.sediment_columns.index(name)] = value
    # The terminal deposit is an instantaneous inter-event operation, not an
    # extra transport timestep; its per-class amounts are reported separately.
    previous_peak_q = to_float(r.peak_outlet_discharge_m3_s)
    r = replace(r, hydrograph=hydro, last_velocity_m_s=v, sediment_hydrograph=sed_hydro,
                peak_velocity_m_s=xp.maximum(r.peak_velocity_m_s, v),
                peak_outlet_discharge_m3_s=xp.maximum(r.peak_outlet_discharge_m3_s, q),
                time_of_peak_outlet_s=xp.asarray(state.t_s) if q > previous_peak_q else r.time_of_peak_outlet_s)
    progress = replace(progress, result=r)
    budget = water_budget(progress)
    if not budget['closed']:
        raise SedimentEventError('pre-reset water budget failed')
    h, soil = xp.array(state.storm.depth_m, copy=True), xp.array(state.storm.soil_water_m, copy=True)
    dry_bed = with_water(state.bed, context, xp.zeros_like(h))
    dry = replace(state, bed=dry_bed, storm=initial_state(state.graph, xp.zeros_like(h), xp.zeros_like(soil), t_s=state.t_s),
                  sediment_velocity_m_s=xp.zeros_like(state.sediment_velocity_m_s), phase=canonical_phase_for(state))
    dry_water = host_packet(xp, [(dry.storm.depth_m.sum() + dry.storm.soil_water_m.sum()) * state.graph.dx_m ** 2])
    combined_residual = (float(dry_water[0])
                         + budget['surface_final_m3'] + budget['soil_final_m3']
                         + budget['drainage_m3'] + budget['export_m3']
                         - budget['surface_initial_m3'] - budget['soil_initial_m3'] - budget['rain_m3'])
    if abs(combined_residual) > budget['tolerance_m3']:
        raise SedimentEventError('combined storm and dry-reset water budget failed')
    accounting = {'policy': 'external dry-again removal; no evapotranspiration simulated',
                  'storm_water_budget': budget, 'completion_before_terminal': before_metrics,
                  'completion_after_terminal': after_metrics,
                  'surface_removed_m3': budget['surface_final_m3'], 'soil_removed_m3': budget['soil_final_m3'],
                  'water_added_m3': 0.0, 'target_soil_water_m': 0.0,
                  'combined_water_residual_m3': combined_residual,
                  'terminal_sediment_residual_kg': (final_host - initial_host).tolist(),
                  'terminal_sediment_tolerance_kg': terminal_bound,
                  'terminal_deposition_by_class_kg': np.asarray(to_host(deposited.sum(axis=(0, 1))),
                                                                dtype=np.float64).tolist(),
                  'sediment_closure': r.closure()}
    if 'terminal_surface_volume_m3' in after_metrics:
        # CW-C2: the ponded terminal-storage water is part of `surface_removed_m3` (full physical storage); its
        # share and the moving-cell share are reported separately. Nothing else is added for the strict policy.
        accounting.update({
            'terminal_storage_removed_m3': after_metrics['terminal_surface_volume_m3'],
            'moving_surface_removed_m3': budget['surface_final_m3'] - after_metrics['terminal_surface_volume_m3'],
            'terminal_storage_cells': after_metrics['terminal_storage_cells'],
            'terminal_storage_policy': 'external dry reset removes the ponded terminal-storage water with the '
                                       'rest of the surface water; no pond residence, spill or overtopping model',
        })
    if policy.deposit_mobile_at_flow_end:
        # CW-B51: small metadata only (class vectors and scalars, no grid copies). The pre-settlement mobile load is
        # the one validated by the pre-terminal quiet check; the deposition is MAPLE's actual terminal deposition.
        accounting['flow_end_settlement'] = {
            'policy': 'deposit_mobile_at_flow_end', 'assumption': FLOW_END_SETTLEMENT_ASSUMPTION,
            'mobile_before_settlement_by_class_kg': before_metrics['mobile_by_class_kg'],
            'max_mobile_cell_class_kg_before_settlement': before_metrics['max_mobile_cell_class_kg'],
            'mobile_per_element_limit_kg': before_metrics['mobile_per_element_limit_kg'],
            'mobile_below_significance_before_settlement': before_metrics['mobile_below_significance'],
            'terminal_deposition_by_class_kg': accounting['terminal_deposition_by_class_kg'],
        }
    # All settlement, terrain, phase, history, quiet, and water/class budget
    # checks passed. Publish the ONE bed shared by pre-reset and dry states last.
    accepted_bed = publish_bed_storage(state.bed, context)
    state = replace(state, bed=accepted_bed)
    dry = replace(dry, bed=replace(dry.bed, voxel_column=accepted_bed.voxel_column))
    r = replace(r, state=state)
    progress = replace(progress, result=r)
    return DryHandoff(progress, dry, h, soil, deposited, accounting)
