"""Completion and conservative dry handoff around the existing sediment solver.

Only accepted steps advance the quiet interval. Checkpoint boundaries preserve
that solver's original partition; neither checkpoint I/O nor pausing commits
terrain. The terminal reset is an external inter-event water removal, not ET.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from maple_syrup.conservation import reservoir_bound_kg, volume_roundoff_bound_m3
from maple_syrup.sediment_bed import apply_bed_demand, bed_inventory, with_water
from maple_syrup.sediment_event import (
    SedimentEventControl,
    SedimentEventError,
    SedimentEventResult,
    SedimentEventState,
    _commit_and_reroute,
    canonical_phase_for,
    evolve_sediment_event,
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

    def validated(self):
        for name in ('max_depth_m', 'surface_volume_m3', 'max_unit_discharge_m2_s',
                     'outlet_discharge_m3_s', 'hold_s', 'mobile_per_cell_class_kg'):
            value = getattr(self, name)
            if value is None and name == 'mobile_per_cell_class_kg':
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise SedimentEventError(f'{name} must be finite and positive')
        return self


def quiet_metrics(state, context, policy, rainfall_end_s):
    """CPU event-boundary reductions; no implicit device-to-host fallback."""
    if state.graph.xp is not np:
        raise SedimentEventError('complete-event driver currently requires NumPy')
    h, q = state.storm.depth_m, state.storm.discharge_m2_s
    mobile = state.bed.water.mobile_mass_by_cell_class_kg
    if any(not np.isfinite(a).all() or np.any(a < 0) for a in (h, q, mobile, state.storm.soil_water_m)):
        raise SedimentEventError('invalid physical state in completion check')
    threshold = context.mass_resolution_kg if policy.mobile_per_cell_class_kg is None else policy.mobile_per_cell_class_kg
    area = state.graph.dx_m ** 2
    values = {
        'max_depth_m': float(h.max()),
        'surface_volume_m3': float(h.sum() * area),
        'max_unit_discharge_m2_s': float(q.max()),
        'outlet_discharge_m3_s': float(q[state.graph.outlet].sum() * state.graph.dx_m),
        'max_mobile_cell_class_kg': float(mobile.max()),
        'mobile_by_class_kg': mobile.sum(axis=(0, 1)).tolist(),
        'mobile_per_element_limit_kg': float(threshold),
        'rainfall_finished': bool(state.t_s >= rainfall_end_s),
    }
    qualifies = values['rainfall_finished'] and all((
        values['max_depth_m'] <= policy.max_depth_m,
        values['surface_volume_m3'] <= policy.surface_volume_m3,
        values['max_unit_discharge_m2_s'] <= policy.max_unit_discharge_m2_s,
        values['outlet_discharge_m3_s'] <= policy.outlet_discharge_m3_s,
        values['max_mobile_cell_class_kg'] <= threshold,
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
                   max_report_rows=100_000, continuation=None, checkpoint_callback=None):
    """Return a dry handoff, or wet EventProgress when a callback requests pause.

    callback(progress) runs only at original reporting/forcing boundaries,
    synchronously. Returning True requests pause; it must not retain mutable
    accumulator references when returning False (write the checkpoint now).
    On a duration/step limit, raise IncompleteEventError; earlier published
    checkpoints remain valid. No terminal action has run on that path.
    """
    policy = CompletionPolicy() if policy is None else policy
    policy.validated()
    control.validated()
    if not control.commit:
        raise SedimentEventError('complete event requires actual MAPLE commits')
    control = replace(control, force_final_commit=False)
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
        surface0, soil0 = float(state.storm.depth_m.sum() * area), float(state.storm.soil_water_m.sum() * area)
        quiet = QuietProgress()
        previous = None
    if state.graph.xp is not np:
        raise SedimentEventError('complete event currently requires NumPy; no GPU fallback')
    if state.t_s >= max_end_s:
        raise IncompleteEventError('maximum event duration reached without a dry handoff')

    def observe(candidate):
        nonlocal quiet
        quiet = quiet.observe(candidate, context, policy, schedule.end_s)
        return quiet.ready

    def progress(result):
        return EventProgress(result, policy, quiet, control, plan, origin, surface0, soil0,
                             schedule.end_s, report_every_s, max_report_rows)

    paused = False

    def boundary(result):
        nonlocal paused
        if checkpoint_callback is not None:
            paused = bool(checkpoint_callback(progress(result)))
        return paused

    try:
        result = evolve_sediment_event(
            state, context, column, field, schedule, vegetation, sediment, max_end_s, control,
            report_every_s=report_every_s, max_report_rows=max_report_rows,
            continuation=previous, planned_boundaries=plan[plan > state.t_s],
            on_accepted_step=observe, on_boundary=boundary)
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
    area = r.state.graph.dx_m ** 2
    rain = float(r.cumulative_rain_m.sum() * area)
    surface = float(r.state.storm.depth_m.sum() * area)
    soil = float(r.state.storm.soil_water_m.sum() * area)
    drainage, export = float(r.cumulative_drainage_m.sum() * area), float(r.cumulative_export_m3)
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
    """Pure terminal transaction; all failure paths preserve the wet input."""
    from maple.surface.topographic_commit.commit import (
        is_already_committed,
        ledger_is_empty,
    )
    from maple.water import WaterProcessDemand

    r, state = progress.result, progress.result.state
    policy = progress.policy.validated()
    qualifies, before_metrics = quiet_metrics(state, context, policy, progress.rainfall_end_s)
    if (progress.status != 'wet' or not progress.quiet.ready or not qualifies
            or progress.quiet.since_s is None or progress.quiet.last_t_s != state.t_s
            or state.t_s - progress.quiet.since_s < policy.hold_s):
        raise SedimentEventError('terminal reset requires a completed continuous quiet interval')
    if not r.closure()['closed'] or not r.closure()['request_reconciled'] or not water_budget(progress)['closed']:
        raise SedimentEventError('pre-terminal event budgets do not close')
    pool = state.bed.water.mobile_mass_by_cell_class_kg
    initial_total = bed_inventory(state.bed) + pool.sum(axis=(0, 1))
    deposited = np.zeros_like(pool)
    if np.any(pool):
        bed, transfer = apply_bed_demand(state.bed, context, WaterProcessDemand(np.zeros_like(pool), pool))
        if np.any(bed.water.mobile_mass_by_cell_class_kg != 0):
            raise SedimentEventError('MAPLE refused terminal mobile deposition; no inventory erased')
        deposited = transfer.deposition_by_cell_class_kg
        bc = {k: v.copy() for k, v in r.by_class.items()}
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
    already = is_already_committed(bed.ledger, bed.committed_topography, bed.voxel_column,
                                   bed.active_layer, context.geometry, context.mass_resolution_kg, state.t_s)
    if not already:
        commit = _commit_and_reroute(bed, context, state.graph, state.terrain, state.storm, state.t_s, force=True)
        if commit is None:
            raise SedimentEventError('final terrain commit was not performed')
        state = replace(state, bed=commit.bed, graph=commit.graph, network=commit.network,
                        grid=commit.grid, storm=commit.storm)
        log = r.commit_log + ((commit.record,) if len(r.commit_log) < progress.control.max_commit_log else ())
        r = replace(r, state=state, n_commits=r.n_commits + 1, n_forced_commits=r.n_forced_commits + 1,
                    n_graph_changes=r.n_graph_changes + int(commit.record['graph_changed']),
                    rerouted_cells_total=r.rerouted_cells_total + commit.record['rerouted_cells'], commit_log=log,
                    ledger_process_totals_reset_kg=r.ledger_process_totals_reset_kg + commit.maple_result.pre_commit_process_totals_kg,
                    committed_net_bed_change_kg=r.committed_net_bed_change_kg + commit.maple_result.pre_commit_pending_mass_by_class_kg)
    if not ledger_is_empty(state.bed.ledger):
        raise SedimentEventError('terminal MAPLE ledger is not fully empty')
    qualifies, after_metrics = quiet_metrics(state, context, policy, progress.rainfall_end_s)
    if not qualifies:
        raise SedimentEventError('terminal terrain change invalidated completion; input remains wet')
    final_total = bed_inventory(state.bed) + state.bed.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1))
    terminal_bound = reservoir_bound_kg(1, initial_total, final_total)
    if np.any(np.abs(final_total - initial_total) > terminal_bound):
        raise SedimentEventError('terminal sediment transaction fails MAPLE conservation bound')
    if not r.closure()['closed'] or np.any(state.bed.water.mobile_mass_by_cell_class_kg):
        raise SedimentEventError('terminal sediment closure or empty-mobile check failed')
    # Refresh terminal-row diagnostics after the real final commit/settlement.
    q = float(state.storm.discharge_m2_s[state.graph.outlet].sum() * state.graph.dx_m)
    v = np.sqrt(state.storm.depth_m) * state.graph.conveyance.reshape(state.graph.shape)
    hydro, sed_hydro = r.hydrograph.copy(), r.sediment_hydrograph.copy()
    hydro[-1, 8] = q
    hydro[-1, 10] = float(v.max())
    updates = {'mobile_kg': 0.0, 'cumulative_deposition_kg': float(r.by_class['deposition_actual'].sum()),
               'commit_count': float(state.bed.committed_topography.commit_count),
               'graph_changes': float(r.n_graph_changes)}
    for k in range(len(context.grain_classes.classes)):
        updates[f'mobile_c{k + 1}_kg'] = 0.0
        updates[f'cumulative_deposition_c{k + 1}_kg'] = float(r.by_class['deposition_actual'][k])
    for name, value in updates.items():
        sed_hydro[-1, r.sediment_columns.index(name)] = value
    # The terminal deposit is an instantaneous inter-event operation, not an
    # extra transport timestep; its per-class amounts are reported separately.
    r = replace(r, hydrograph=hydro, last_velocity_m_s=v, sediment_hydrograph=sed_hydro,
                peak_velocity_m_s=np.maximum(r.peak_velocity_m_s, v),
                peak_outlet_discharge_m3_s=np.maximum(r.peak_outlet_discharge_m3_s, q),
                time_of_peak_outlet_s=np.asarray(state.t_s) if q > r.peak_outlet_discharge_m3_s else r.time_of_peak_outlet_s)
    progress = replace(progress, result=r)
    budget = water_budget(progress)
    if not budget['closed']:
        raise SedimentEventError('pre-reset water budget failed')
    h, soil = state.storm.depth_m.copy(), state.storm.soil_water_m.copy()
    dry_bed = with_water(state.bed, context, np.zeros_like(h))
    dry = replace(state, bed=dry_bed, storm=initial_state(state.graph, np.zeros_like(h), np.zeros_like(soil), t_s=state.t_s),
                  sediment_velocity_m_s=np.zeros_like(state.sediment_velocity_m_s), phase=canonical_phase_for(state))
    combined_residual = (float((dry.storm.depth_m.sum() + dry.storm.soil_water_m.sum()) * state.graph.dx_m**2)
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
                  'terminal_sediment_residual_kg': (final_total - initial_total).tolist(),
                  'terminal_sediment_tolerance_kg': terminal_bound,
                  'terminal_deposition_by_class_kg': deposited.sum(axis=(0, 1)).tolist(),
                  'sediment_closure': r.closure()}
    return DryHandoff(progress, dry, h, soil, deposited, accounting)
