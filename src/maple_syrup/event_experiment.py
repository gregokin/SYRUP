"""Complete Plot 1 water event, conservative dry handoff, and explicit restart."""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from maple_syrup.case_import import _refuse_output, verify_plot1_case
from maple_syrup.characteristic_transport import (
    DEFAULT_COURANT_MAX as CHARACTERISTIC_COURANT_MAX,
)
from maple_syrup.characteristic_transport import DEFAULT_N_BINS, MAX_N_BINS
from maple_syrup.checkpoint import load_checkpoint, save_checkpoint, sha256_file
from maple_syrup.complete_event import (
    CompletionPolicy,
    DryHandoff,
    EventProgress,
    IncompleteEventError,
    complete_event,
    water_budget,
)
from maple_syrup.provenance import source_tree_digest
from maple_syrup.sediment_event import (
    TRANSPORT_IMPLEMENTATIONS,
    TRANSPORT_SCHEMES,
    SedimentEventControl,
    SedimentEventError,
)
from maple_syrup.sediment_experiment import (
    _jsonable,
    phase_summary,
    prepare_verified_sediment_case,
)
from maple_syrup.storm import HYDROGRAPH_COLUMNS, StormControl, plan_boundaries


@dataclasses.dataclass(frozen=True)
class CompleteRun:
    summary: dict
    state: object
    outcome: EventProgress | DryHandoff
    context: object


def run_complete_plot1(case_dir, output_dir, *, resume=None, checkpoint_dir=None,
                       checkpoint_every_s=600., pause_at_s=None, max_end_s=None,
                       max_dt_s=1., implementation='numba', report_every_s=60.,
                       policy=None, max_steps=10_000_000, max_report_rows=100_000,
                       transport_scheme='characteristic', phase_bins=DEFAULT_N_BINS,
                       transport_implementation='auto', sediment_courant_max=CHARACTERISTIC_COURANT_MAX,
                       mahleran_root=None, expected_maple_root=None):
    from maple.core.parameters import config_to_dict
    from maple.core.types.topographic_commit import committed_elevation_m

    policy = CompletionPolicy() if policy is None else policy
    policy.validated()
    control = SedimentEventControl(storm=StormControl(max_dt_s=max_dt_s, implementation=implementation,
                                                    max_steps=max_steps), force_final_commit=False,
                                   transport_scheme=transport_scheme, phase_bins=phase_bins,
                                   transport_implementation=transport_implementation,
                                   sediment_courant_max=sediment_courant_max).validated()
    if max_end_s is not None and (isinstance(max_end_s, bool) or not isinstance(max_end_s, (int, float))
                                  or not np.isfinite(max_end_s) or max_end_s <= 0):
        raise SedimentEventError('maximum end time must be finite and positive')
    for name, value in (('checkpoint_every_s', checkpoint_every_s), ('report_every_s', report_every_s)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise SedimentEventError(f'{name} must be finite and positive')
    if implementation == 'numba' or (control.characteristic and control.resolved_transport_implementation() == 'numba'):
        from maple_syrup.routing_numba import numba_available
        if not numba_available():
            raise SedimentEventError('Numba requested but unavailable; no implicit fallback')
    start_wall = time.perf_counter()
    output = Path(output_dir).resolve()
    checkpoints = Path(checkpoint_dir).resolve() if checkpoint_dir is not None else output.with_name(output.name + '_checkpoints')
    if output.exists() or checkpoints.exists():
        raise SedimentEventError('output and checkpoint directories must be new')
    if output == checkpoints or output.is_relative_to(checkpoints) or checkpoints.is_relative_to(output):
        raise SedimentEventError('output and checkpoint trees must be separate')
    verified = verify_plot1_case(case_dir, mahleran_root=mahleran_root, expected_maple_root=expected_maple_root)
    recipe = verified.report['recipe']
    roots = {'MAPLE source': verified.maple_dependency.source_root,
             'MAPLE package': verified.maple_dependency.package_dir,
             'case': Path(case_dir).resolve(),
             'MAHLERAN': Path(mahleran_root or recipe['mahleran_root']).resolve(),
             'recorded MAHLERAN': Path(recipe['mahleran_root']).resolve(),
             'recipe': Path(recipe['recipe_path']).resolve().parent,
             'SYRUP package': Path(__file__).parent.resolve()}
    for path in (output, checkpoints):
        _refuse_output(path, roots)
    prepared = prepare_verified_sediment_case(verified, mahleran_root=mahleran_root, end_s=max_end_s, control=control)
    end = float(prepared['end'])
    if not np.isfinite(end) or end <= 0:
        raise SedimentEventError('maximum end time must be finite and positive')
    state, context = prepared['state0'], prepared['context']
    column, field, schedule = prepared['column'], prepared['field'], prepared['schedule']
    sediment, vegetation = prepared['sediment'], prepared['vegetation']
    source_root = Path(__file__).parent
    source_hash = source_tree_digest(source_root).digest_sha256
    maple_hash = verified.maple_provenance['package_source_digest']['digest_sha256']
    identity = {
        'schema': 'maple-syrup-complete-event/1',
        'syrup_source_sha256': source_hash, 'maple_source_sha256': maple_hash,
        'case_identity_sha256': verified.binding['maple_case_identity_sha256'],
        'case_binding': verified.binding,
        'maple_configuration': _jsonable(config_to_dict(verified.case.config)),
        'mahleran_xml_sha256': prepared['xml_sha_before'],
        'rainfall_sha256': schedule.provenance.sha256,
        'sediment_parameters': prepared['sediment_record'],
        'column_parameters': prepared['parameter_record'],
        'depth_update_rule': context.depth_update_rule,
        'control': dataclasses.asdict(control), 'completion_policy': dataclasses.asdict(policy),
        'max_end_s': end, 'report_every_s': report_every_s, 'max_report_rows': max_report_rows,
    }
    # JSON canonical values: no custom types, nonfinite values, or silent casts at resume.
    identity = json.loads(json.dumps(identity, default=_jsonable, allow_nan=False))

    def check_sources():
        if source_tree_digest(source_root).digest_sha256 != source_hash:
            raise SedimentEventError('SYRUP source changed during the event')
        if source_tree_digest(verified.maple_dependency.package_dir).digest_sha256 != maple_hash:
            raise SedimentEventError('MAPLE source changed during the event')
        if sha256_file(prepared['xml_path']) != prepared['xml_sha_before']:
            raise SedimentEventError('MAHLERAN XML changed during the event')
        if sha256_file(verified.rainfall_path) != schedule.provenance.sha256:
            raise SedimentEventError('rainfall source changed during the event')

    continuation = None
    if resume is not None:
        continuation = load_checkpoint(resume, context, column, sediment, identity)
        state = continuation.result.state
    plan = plan_boundaries(schedule, 0., end, report_every_s, max_report_rows=max_report_rows)
    if pause_at_s is not None and (isinstance(pause_at_s, bool) or not np.isfinite(pause_at_s)
                                  or pause_at_s not in plan or pause_at_s <= state.t_s or pause_at_s >= end):
        raise SedimentEventError('pause time must be a future original boundary before the duration limit')
    checkpoint_paths = []
    next_checkpoint = (np.floor(state.t_s / checkpoint_every_s) + 1) * checkpoint_every_s

    def boundary(progress):
        nonlocal next_checkpoint
        t = progress.result.state.t_s
        pause = pause_at_s is not None and t == pause_at_s
        if t >= next_checkpoint or pause:
            check_sources()
            target = checkpoints / f'step_{progress.result.n_accepted_steps:09d}'
            save_checkpoint(target, progress, context, column, sediment, identity)
            checkpoint_paths.append(str(target))
            next_checkpoint = (np.floor(t / checkpoint_every_s) + 1) * checkpoint_every_s
        return pause

    loop_start = time.perf_counter()
    try:
        outcome = complete_event(state, context, column, field, schedule, vegetation, sediment,
                                 max_end_s=end, control=control, policy=policy,
                                 report_every_s=report_every_s, max_report_rows=max_report_rows,
                                 continuation=continuation, checkpoint_callback=boundary)
    except IncompleteEventError as exc:
        last = checkpoint_paths[-1] if checkpoint_paths else resume
        raise IncompleteEventError(f'{exc}; last accepted checkpoint: {last}') from exc
    loop_s = time.perf_counter() - loop_start
    check_sources()
    complete = isinstance(outcome, DryHandoff)
    progress = outcome.progress if complete else outcome
    result = progress.result
    final = outcome.dry_state if complete else result.state
    summary = {
        'schema': identity['schema'], 'status': 'complete' if complete else 'paused',
        'identity': identity, 'time_s': final.t_s,
        'n_accepted_steps': result.n_accepted_steps, 'n_rejected_attempts': result.n_rejected_attempts,
        'n_commits': result.n_commits, 'quiet_progress': dataclasses.asdict(progress.quiet),
        'water_budget_before_reset': water_budget(progress), 'sediment_closure': result.closure(),
        'reset_accounting': outcome.accounting if complete else None,
        'checkpoint_paths': checkpoint_paths, 'resumed_from': None if resume is None else str(Path(resume).resolve()),
        'last_checkpoint': checkpoint_paths[-1] if checkpoint_paths else None,
        'timings': {'invocation_step_loop_and_checkpoints_wall_s': loop_s,
                    'setup_wall_s': loop_start - start_wall,
                    'jit_note': 'first invocation includes lazy compilation; timings are observational'},
        'backend': {'hydraulics': implementation, 'arrays': 'numpy', 'gpu': 'not exercised; CPU driver only'},
        'transport': {**control.transport_record(), **phase_summary(result)},
        'hydrograph_semantics': 'accepted physical storm, final row includes terminal settling/commit before external dry reset',
        'limitations': ['fixed exporting ring; new pits refused', 'no wind event, splash, ecology or ET',
                        'original MAHLERAN equations checked separately; not a full MAHLERAN storm benchmark'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.partial-', dir=output.parent))
    try:
        if complete:
            save_checkpoint(staging / 'handoff', outcome, context, column, sediment, identity)
        np.savez(staging / 'final_state.npz',
                 voxel_mass_kg=final.bed.voxel_column.mass_kg, active_mass_kg=final.bed.active_layer.mass_kg,
                 available_mass_kg=final.bed.sediment_availability.available_mass_kg,
                 bound_mass_kg=final.bed.sediment_availability.bound_mass_kg,
                 committed_elevation_m=committed_elevation_m(final.bed.committed_topography),
                 depth_m=final.storm.depth_m, soil_water_m=final.storm.soil_water_m,
                 discharge_m2_s=final.storm.discharge_m2_s,
                 mobile_mass_kg=final.bed.water.mobile_mass_by_cell_class_kg,
                 sediment_velocity_m_s=final.sediment_velocity_m_s,
                 pre_reset_depth_m=result.state.storm.depth_m, pre_reset_soil_water_m=result.state.storm.soil_water_m,
                 terminal_deposition_kg=outcome.terminal_deposition_kg if complete else np.zeros_like(final.bed.active_layer.mass_kg),
                 **({} if final.phase is None else {'phase_fraction': final.phase.fraction,
                                                    'phase_position_m': final.phase.position_m}))
        rows = {name: result.hydrograph[:, i] for i, name in enumerate(HYDROGRAPH_COLUMNS)}
        rows.update({f'sed_{name}': result.sediment_hydrograph[:, i] for i, name in enumerate(result.sediment_columns)})
        np.savez(staging / 'hydrograph.npz', **rows)
        summary['outputs'] = {str(p.relative_to(staging)): sha256_file(p)
                              for p in sorted(staging.rglob('*')) if p.is_file()}
        summary['timings']['invocation_before_summary_wall_s'] = time.perf_counter() - start_wall
        (staging / 'completion_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        if output.exists():
            raise SedimentEventError('output appeared during run')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return CompleteRun(summary, final, outcome, context)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--checkpoint-every-s', type=float, default=600.)
    parser.add_argument('--pause-at-s', type=float)
    parser.add_argument('--max-end-s', type=float)
    parser.add_argument('--max-dt-s', type=float, default=1.)
    parser.add_argument('--implementation', choices=('numba', 'array'), default='numba')
    parser.add_argument('--report-every-s', type=float, default=60.)
    parser.add_argument('--hold-s', type=float, default=60.)
    parser.add_argument('--max-depth-m', type=float, default=1e-8)
    parser.add_argument('--surface-volume-m3', type=float, default=1e-6)
    parser.add_argument('--max-unit-discharge-m2-s', type=float, default=1e-10)
    parser.add_argument('--outlet-discharge-m3-s', type=float, default=1e-9)
    parser.add_argument('--mobile-per-cell-class-kg', type=float,
                        help='physical stopping significance; default actual MAPLE mass_resolution_kg, not a conservation tolerance')
    parser.add_argument('--max-steps', type=int, default=10_000_000)
    parser.add_argument('--max-report-rows', type=int, default=100_000)
    parser.add_argument('--transport-scheme', choices=TRANSPORT_SCHEMES, default='characteristic',
                        help='characteristic (Phase 7b phase bins, default) or the Phase 5 upwind comparison')
    parser.add_argument('--phase-bins', type=int, default=DEFAULT_N_BINS,
                        help=f'position bins per cell/class for the characteristic scheme (1..{MAX_N_BINS})')
    parser.add_argument('--transport-implementation', choices=TRANSPORT_IMPLEMENTATIONS, default='auto',
                        help='characteristic kernel: auto (follow --implementation), array or numba; no fallback')
    parser.add_argument('--sediment-courant-max', type=float, default=CHARACTERISTIC_COURANT_MAX)
    parser.add_argument('--mahleran-root', type=Path)
    parser.add_argument('--expected-maple-root', type=Path)
    args = vars(parser.parse_args(argv))
    policy = CompletionPolicy(**{name: args.pop(name) for name in (
        'hold_s', 'max_depth_m', 'surface_volume_m3', 'max_unit_discharge_m2_s',
        'outlet_discharge_m3_s', 'mobile_per_cell_class_kg')})
    try:
        run = run_complete_plot1(**args, policy=policy)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(run.summary, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
