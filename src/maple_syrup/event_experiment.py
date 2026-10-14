"""Complete Plot 1 water event, conservative dry handoff, and explicit restart.

Backends (CW-C1): `backend='numpy'` runs the array or numba hydraulics on the host (unchanged default);
`backend='cupy'` runs ONLY the explicit CUDA hydrology (`implementation='cuda'`, bisection or Newton) with the
whole small case prepared in CuPy after the CW-A1 device memory preflight, the caller-owned `CudaHydraulics`
lifecycle handed to the solver, checkpoints written through the host bundle format and resumed in the same
backend only. Every other pairing is refused before the case is touched; there is no fallback in either direction.
Reporting converts device results with MAPLE's counted host helpers; the loop timing is synchronized before it is
read. The optional `allow_maple_source_change` flag is the existing case-verification option (an explicitly
recorded isolated MAPLE source difference), never a mutation of the original inputs.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from maple_syrup.case_import import _refuse_output, verify_plot1_case
from maple_syrup.characteristic_sparse import SparsePhaseState
from maple_syrup.characteristic_transport import (
    DEFAULT_COURANT_MAX as CHARACTERISTIC_COURANT_MAX,
)
from maple_syrup.characteristic_transport import DEFAULT_N_BINS, MAX_N_BINS
from maple_syrup.checkpoint import (
    DEFAULT_MAX_CHECKPOINT_BYTES,
    checkpoint_byte_bound,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
)
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
    CUDA_MODES,
    PHASE_STORAGES,
    TRANSPORT_IMPLEMENTATIONS,
    TRANSPORT_SCHEMES,
    CudaHydraulics,
    SedimentEventControl,
    SedimentEventError,
    routing_refresh_record,
)
from maple_syrup.sediment_experiment import (
    BACKEND_PAIRINGS,
    _jsonable,
    cuda_memory_preflight,
    phase_summary,
    prepare_verified_sediment_case,
)
from maple_syrup.storm import (
    HYDROGRAPH_COLUMNS,
    STORM_IMPLEMENTATIONS,
    StormControl,
    plan_boundaries,
)


@dataclasses.dataclass(frozen=True)
class CompleteRun:
    summary: dict
    state: object
    outcome: EventProgress | DryHandoff
    context: object


def control_identity(control: SedimentEventControl) -> dict:
    """The control as JSON data for a run / checkpoint identity. The CW-B5 keys (`phase_storage`, `phase_max_nnz`)
    are present ONLY for the CSR storage, so the identity of every dense run and the loadability of every existing
    dense checkpoint are unchanged."""
    record = dataclasses.asdict(control)
    if control.phase_storage == 'dense' and control.phase_max_nnz is None:
        record.pop('phase_storage')
        record.pop('phase_max_nnz')
    if not control.record_exchange_history:
        record.pop('record_exchange_history')
    if control.deep_publication == 'step' and control.deep_publication_high_water_records is None:
        # CW-B102: the opt-in deep-publication keys appear only when set, so every default identity is unchanged
        record.pop('deep_publication')
        record.pop('deep_publication_high_water_records')
    return record


def policy_identity(policy: CompletionPolicy) -> dict:
    """The completion policy as JSON data for a run / checkpoint identity. The CW-B51 opt-in key
    (`deposit_mobile_at_flow_end`) is present ONLY when set, so the identity of every default run is unchanged."""
    record = dataclasses.asdict(policy)
    if not policy.deposit_mobile_at_flow_end:
        record.pop('deposit_mobile_at_flow_end')
    return record


def _dense_sink_limitation(graph_policy: dict) -> str:
    """CW-B73: the dense summary's sink limitation, from the run's ACTUAL graph policy (the same policy
    `refresh_routing` rebuilds with after every commit). Strict default: unchanged text."""
    if not graph_policy['allow_pit_storage']:
        return 'fixed exporting ring; new pits refused'
    if graph_policy['allow_flat_storage']:
        return ('fixed exporting ring; new pits and flat sinks become terminal storage (per-commit counts in the '
                'commit log; no spill or overtopping)')
    return ('fixed exporting ring; new pits become terminal storage (per-commit counts in the commit log; no spill '
            'or overtopping); new flat sinks refused')


def run_complete_case(case_dir, output_dir, *, resume=None, checkpoint_dir=None,
                       checkpoint_every_s=600., pause_at_s=None, max_end_s=None,
                       max_dt_s=1., implementation='numba', report_every_s=60.,
                       policy=None, max_steps=10_000_000, max_report_rows=100_000,
                       transport_scheme='characteristic', phase_bins=DEFAULT_N_BINS,
                       transport_implementation='auto', sediment_courant_max=CHARACTERISTIC_COURANT_MAX,
                       mahleran_root=None, expected_maple_root=None,
                       backend='numpy', root_solver='bisection', cuda_mode='auto', bisection_iterations=40,
                       allow_maple_source_change=False, record_exchange_history=False,
                       allow_pit_storage=False, allow_flat_storage=False, allow_masked_nodata=False,
                       nodata_value=None, phase_storage='dense', phase_max_nnz=None, case_setup=None,
                       max_checkpoint_bytes=DEFAULT_MAX_CHECKPOINT_BYTES, deep_publication='step',
                       deep_publication_high_water_records=None):
    from maple.core.backend import (
        BackendUnavailableError,
        read_transfer_counters,
        resolve_backend,
        synchronize,
        to_host,
    )
    from maple.core.parameters import config_to_dict
    from maple.core.types.topographic_commit import committed_elevation_m

    from maple_syrup.sediment_bed import maple_supports_exchange_history
    from maple_syrup.sediment_event import exchange_history_totals

    policy = CompletionPolicy() if policy is None else policy
    policy.validated()
    # CW-F2: the resume byte bound is a resource control, validated before any setup or read and forwarded to the
    # loader. It is NOT part of the run/checkpoint identity (a resource bound does not change the physics, and an
    # existing bundle stays loadable under any sufficient bound).
    max_checkpoint_bytes = checkpoint_byte_bound(max_checkpoint_bytes, 'max_checkpoint_bytes')
    if not isinstance(record_exchange_history, bool):
        raise SedimentEventError('record_exchange_history must be bool')
    if record_exchange_history and not maple_supports_exchange_history():
        # Refused before the case is verified or placed anywhere: no event mutation on an unsupported dependency.
        raise SedimentEventError('the bound MAPLE dependency cannot record the exchange history '
                                 '(commit_topography has no record_exchange_history keyword)')
    control = SedimentEventControl(storm=StormControl(max_dt_s=max_dt_s, implementation=implementation,
                                                    max_steps=max_steps, root_solver=root_solver,bisection_iterations=bisection_iterations),
                                   force_final_commit=False,
                                   transport_scheme=transport_scheme, phase_bins=phase_bins,
                                   transport_implementation=transport_implementation,
                                   sediment_courant_max=sediment_courant_max,
                                   record_exchange_history=record_exchange_history, phase_storage=phase_storage, phase_max_nnz=phase_max_nnz,
                                   deep_publication=deep_publication,
                                   deep_publication_high_water_records=deep_publication_high_water_records).validated()
    if max_end_s is not None and (isinstance(max_end_s, bool) or not isinstance(max_end_s, (int, float))
                                  or not np.isfinite(max_end_s) or max_end_s <= 0):
        raise SedimentEventError('maximum end time must be finite and positive')
    for name, value in (('checkpoint_every_s', checkpoint_every_s), ('report_every_s', report_every_s)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise SedimentEventError(f'{name} must be finite and positive')
    if implementation not in STORM_IMPLEMENTATIONS:
        raise SedimentEventError(f'implementation must be one of {STORM_IMPLEMENTATIONS}, got {implementation!r}')
    if backend not in BACKEND_PAIRINGS:
        raise SedimentEventError(f'backend {backend!r} is refused: must be one of {tuple(BACKEND_PAIRINGS)}')
    if implementation not in BACKEND_PAIRINGS[backend]:
        raise SedimentEventError(
            f'backend {backend!r} with implementation {implementation!r} is refused: numpy pairs with '
            f'{BACKEND_PAIRINGS["numpy"]} (host hydraulics) and cupy ONLY with \'cuda\' (the explicit prepared CUDA '
            'hydrology on a CuPy state); CUDA is not a CPU fallback and the host sweeps make no device transfer')
    if cuda_mode not in CUDA_MODES:
        raise SedimentEventError(f'cuda_mode must be one of {CUDA_MODES}, got {cuda_mode!r}')
    if not isinstance(allow_maple_source_change, bool):
        raise SedimentEventError('allow_maple_source_change must be bool')
    for name, value in (('allow_pit_storage', allow_pit_storage), ('allow_flat_storage', allow_flat_storage),
                        ('allow_masked_nodata', allow_masked_nodata)):
        if not isinstance(value, bool):
            raise SedimentEventError(f'{name} must be bool')
    if allow_flat_storage and not allow_pit_storage:
        raise SedimentEventError('allow_flat_storage extends allow_pit_storage and needs it')
    if allow_masked_nodata and nodata_value is None:
        raise SedimentEventError('allow_masked_nodata needs nodata_value')
    if nodata_value is not None and (isinstance(nodata_value, bool) or not isinstance(nodata_value, (int, float))
                                     or not np.isfinite(nodata_value)):
        raise SedimentEventError('nodata_value must be a finite real number')
    graph_policy = {'allow_masked_nodata': allow_masked_nodata, 'nodata_value': None if nodata_value is None else float(nodata_value),
                    'allow_pit_storage': allow_pit_storage, 'allow_flat_storage': allow_flat_storage}
    opt_in_graph = allow_pit_storage or allow_masked_nodata or nodata_value is not None
    if policy.terminal_storage_water and not allow_pit_storage:
        raise SedimentEventError('terminal_storage_water needs an opt-in pit-storage graph (allow_pit_storage)')
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
    setup = None
    if case_setup is None:
        verified = verify_plot1_case(case_dir, mahleran_root=mahleran_root, expected_maple_root=expected_maple_root,
                                     allow_maple_source_change=allow_maple_source_change)
    else:
        setup = case_setup(case_dir=case_dir, backend=backend, control=control, end_s=max_end_s,
                           expected_maple_root=expected_maple_root, mahleran_root=mahleran_root,
                           allow_maple_source_change=allow_maple_source_change)
        verified = setup['verified']
    recipe = verified.report['recipe']
    roots = {'MAPLE source': verified.maple_dependency.source_root,
             'MAPLE package': verified.maple_dependency.package_dir,
             'case': Path(case_dir).resolve(),
             'MAHLERAN': Path(mahleran_root or recipe.get('mahleran_root',verified.report['mahleran']['root'])).resolve(),
             'recorded MAHLERAN': Path(recipe.get('mahleran_root',verified.report['mahleran']['root'])).resolve(),
             'recipe': Path(recipe['recipe_path']).resolve().parent,
             'SYRUP package': Path(__file__).parent.resolve()}
    for path in (output, checkpoints):
        _refuse_output(path, roots)
    # Resolve the backend (selects the CUDA device; never falls back) and, on a device, run the CW-A1 memory
    # preflight BEFORE the case is placed there.
    try:
        resolved = resolve_backend(backend)
    except BackendUnavailableError as exc:
        raise SedimentEventError(f'backend {backend!r} is refused: {exc}') from exc
    memory_preflight = None
    hydraulics = None
    if backend == 'cupy' and setup is None:
        memory_preflight = cuda_memory_preflight(verified.case, phase_bins=phase_bins, device_id=resolved.device_id,
                                                 transport_scheme=transport_scheme, phase_storage=phase_storage,
                                                 phase_max_nnz=phase_max_nnz)
        hydraulics = CudaHydraulics(mode=cuda_mode)
    if setup is None:
        prepared = prepare_verified_sediment_case(verified, backend=backend, mahleran_root=mahleran_root, end_s=max_end_s,
                                                  control=control)
    else:
        prepared = setup['prepared']
        memory_preflight = setup['memory_preflight']
        hydraulics = CudaHydraulics(mode=cuda_mode) if backend == 'cupy' else None
    xp = prepared['xp']
    end = float(prepared['end'])
    if not np.isfinite(end) or end <= 0:
        raise SedimentEventError('maximum end time must be finite and positive')
    state, context = prepared['state0'], prepared['context']
    if record_exchange_history:
        # Opt-in plumbing to MAPLE's commit (the bed context is per-run metadata, never a case file change).
        context = dataclasses.replace(context, record_exchange_history=True)
    column, field, schedule = prepared['column'], prepared['field'], prepared['schedule']
    sediment, vegetation = prepared['sediment'], prepared['vegetation']
    if opt_in_graph:
        # CW-C2: rebuild the routing graph of the SAME verified DEM / export ring / friction / active mask under the
        # explicit opt-in policy, bind the terrain reference with that policy and restart the initial state from the
        # same MAPLE bed and soil water. The physics, bed and controls are untouched; the graph digest records the
        # policy, so the identity below differs from the strict run.
        from maple_syrup.routing import build_routing_graph
        from maple_syrup.sediment_bed import terrain_reference
        from maple_syrup.sediment_event import initial_event_state

        base = prepared['graph']
        full_z = np.asarray(verified.fields['legacy_full_elevation_m'], dtype=np.float64)
        exports = (np.asarray(verified.fields['legacy_full_rainfall_scaling'], dtype=np.float64) < 0.0
                   if setup is None else np.asarray(state.terrain.export_receiver_full,dtype=np.bool_))
        graph = build_routing_graph(full_z, exports, np.asarray(base.friction_factor, dtype=np.float64), base.dx_m,
                                    active_mask=np.asarray(base.active, dtype=np.bool_), xp=xp, **graph_policy)
        terrain = terrain_reference(full_z, exports, state.bed, graph, nodata_value=graph_policy['nodata_value'])
        state = initial_event_state(graph, terrain, state.bed, context, prepared['soil0'], control=control)
        del base, full_z, exports, graph, terrain  # CW-B30: the new state owns what it needs
    terminal_record = {**graph_policy, 'terminal_storage_water': policy.terminal_storage_water,
                       'graph_policy': state.graph.policy,
                       'n_terminal_storage_cells': 0 if state.graph.pit_storage is None else int(np.sum(np.asarray(state.graph.pit_storage))),
                       'graph_input_sha256': state.graph.input_sha256}
    source_root = Path(__file__).parent
    source_hash = source_tree_digest(source_root).digest_sha256
    maple_hash = verified.maple_provenance['package_source_digest']['digest_sha256']
    control_record = control_identity(control)
    if not record_exchange_history:
        # Default identity unchanged: the opt-in key appears in the identity only when the option is on.
        control_record.pop('record_exchange_history', None)
    identity = {
        'schema': 'maple-syrup-complete-event/1',
        'syrup_source_sha256': source_hash, 'maple_source_sha256': maple_hash,
        'case_identity_sha256': setup['case_identity_sha256'] if setup is not None else verified.binding['maple_case_identity_sha256'],
        'case_binding': verified.binding,
        'maple_configuration': setup['maple_configuration'] if setup is not None else _jsonable(config_to_dict(verified.case.config)),
        'mahleran_xml_sha256': prepared['xml_sha_before'],
        'rainfall_sha256': schedule.provenance.sha256,
        'sediment_parameters': prepared['sediment_record'],
        'column_parameters': prepared['parameter_record'],
        'depth_update_rule': context.depth_update_rule,
        'control': control_identity(control), 'completion_policy': policy_identity(policy),
        'max_end_s': end, 'report_every_s': report_every_s, 'max_report_rows': max_report_rows,
        'backend': backend,
        'graph_policy': graph_policy, 'graph_input_sha256': state.graph.input_sha256,
        # CW-B104: the hydraulic routing-refresh contract (net-elevation only) and its actual threshold
        'routing_refresh': routing_refresh_record(context),
    }
    # CW-B30: the few setup / case values used after this point, read once here (same objects, same expressions),
    # so the hook-returned mappings and the prepared case (which hold the initial bed) are not pinned by this frame
    # through the loop. Nothing is removed from, or written into, any mapping a case-setup hook returned.
    setup_present = setup is not None
    run_configuration = setup['run_configuration'] if setup_present else None
    restart_bound = {'max_checkpoint_uncompressed_bytes': max_checkpoint_bytes,
                     'resolution': ('default of the dense runner (1 GiB)'
                                    if max_checkpoint_bytes == DEFAULT_MAX_CHECKPOINT_BYTES else 'caller-explicit'),
                     'applies_to': 'summed uncompressed payload bytes of a resumed bundle (zip central directories, '
                                   'checked before any array is materialized)',
                     'identity': 'resource control only; excluded from the run and checkpoint identity'}
    if setup_present:
        identity['bed_storage'] = run_configuration
        if setup.get('checkpoint_byte_bound') is not None:
            restart_bound.update(setup['checkpoint_byte_bound'])
    if implementation == 'cuda':
        identity['cuda_mode'] = cuda_mode
    # JSON canonical values: no custom types, nonfinite values, or silent casts at resume.
    identity = json.loads(json.dumps(identity, default=_jsonable, allow_nan=False))

    maple_package_dir = verified.maple_dependency.package_dir
    xml_path, xml_sha_before = prepared['xml_path'], prepared['xml_sha_before']
    rainfall_source = prepared.get('rainfall_source_path', verified.rainfall_path)
    verified_checks = verified.checks

    def check_sources():
        if source_tree_digest(source_root).digest_sha256 != source_hash:
            raise SedimentEventError('SYRUP source changed during the event')
        if source_tree_digest(maple_package_dir).digest_sha256 != maple_hash:
            raise SedimentEventError('MAPLE source changed during the event')
        if sha256_file(xml_path) != xml_sha_before:
            raise SedimentEventError('MAHLERAN XML changed during the event')
        if sha256_file(rainfall_source) != schedule.provenance.sha256:
            raise SedimentEventError('rainfall source changed during the event')

    del prepared, setup, verified  # CW-B30: only this frame's references (see above)

    continuation = None
    if resume is not None:
        continuation = load_checkpoint(resume, context, column, sediment, identity, max_bytes=max_checkpoint_bytes)
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

    synchronize(xp)  # device placement of the prepared case complete before the setup time is read
    from maple_syrup.sediment_bed import deep_storage_record
    deep_record = deep_storage_record(context)
    # CW-B108: MAPLE's process-global store counters, captured around THIS invocation's event loop (checkpoint writes
    # included) for a non-default encoding only; an operational per-invocation delta, never physical/restart state.
    deep_before = _deep_store_counters() if deep_record is not None else None
    counters_before = read_transfer_counters()
    loop_start = time.perf_counter()
    # CW-B30: the start state (initial, or the loaded checkpoint state) and the loaded continuation are handed to
    # `complete_event` through a single-use holder, so this frame does not pin them during the event.
    handoff = [state, continuation]
    del state, continuation
    try:
        outcome = complete_event(handoff.pop(0), context, column, field, schedule, vegetation, sediment,
                                 max_end_s=end, control=control, policy=policy,
                                 report_every_s=report_every_s, max_report_rows=max_report_rows,
                                 continuation=handoff.pop(), checkpoint_callback=boundary, hydraulics=hydraulics)
    except IncompleteEventError as exc:
        last = checkpoint_paths[-1] if checkpoint_paths else resume
        raise IncompleteEventError(f'{exc}; last accepted checkpoint: {last}') from exc
    synchronize(xp)  # every launch complete before the loop time, the transfer counters and the pool are read
    loop_s = time.perf_counter() - loop_start
    transfers = dataclasses.asdict(read_transfer_counters().delta(counters_before))
    deep_delta = None
    if deep_before is not None:
        after = _deep_store_counters()
        deep_delta = {group: {k: after[group][k] - deep_before[group].get(k, 0) for k in after[group]} for group in after}
    device_memory = None
    if backend == 'cupy':
        pool = xp.get_default_memory_pool()
        free_bytes, total_bytes = (int(v) for v in xp.cuda.Device(resolved.device_id).mem_info)
        device_memory = {'pool_used_bytes_after_loop': int(pool.used_bytes()),
                         'pool_total_bytes_after_loop': int(pool.total_bytes()),
                         'device_free_bytes_after_loop': free_bytes, 'device_total_bytes': total_bytes,
                         'note': 'CuPy default pool after the synchronized loop, before host conversion; '
                                 'pool_total is allocated memory retained at this measurement; '
                                 'not a measured peak, and allocations outside this pool are excluded'}
    check_sources()
    complete = isinstance(outcome, DryHandoff)
    progress = outcome.progress if complete else outcome
    result = progress.result
    final = outcome.dry_state if complete else result.state

    def host(value):
        return np.asarray(to_host(value))

    def phase_fields(phase):
        """Final phase for the snapshot: the dense pair, or the CSR arrays themselves (never densified)."""
        if phase is None:
            return {}
        if isinstance(phase, SparsePhaseState):
            return {f'phase_csr_{name}': host(getattr(phase, name))
                    for name in ('row_ptr', 'bin', 'fraction', 'position_m', 'cell_of_slot')}
        return {'phase_fraction': host(phase.fraction), 'phase_position_m': host(phase.position_m)}

    from maple_syrup.sediment_bed import is_windowed_bed

    windowed = is_windowed_bed(final.bed)
    gpu_record = 'not exercised; CPU driver only' if backend == 'numpy' else {
        'executed': True, 'cuda_hydraulics': hydraulics.summary(), 'memory_preflight': memory_preflight,
        'device_memory': device_memory,
                  'claims': 'CuPy water event through the actual MAPLE bed API with host checkpoints; '
                  'reported geometry and completion status define the executed scope; '
                  'same-backend resume only; no speed claim'}
    summary = {
        'schema': identity['schema'], 'status': 'complete' if complete else 'paused',
        'identity': identity, 'time_s': final.t_s,
        'n_accepted_steps': result.n_accepted_steps, 'n_rejected_attempts': result.n_rejected_attempts,
        'n_commits': result.n_commits, 'quiet_progress': dataclasses.asdict(progress.quiet),
        'water_budget_before_reset': water_budget(progress), 'sediment_closure': result.closure(),
        'reset_accounting': outcome.accounting if complete else None,
        'checkpoint_paths': checkpoint_paths, 'resumed_from': None if resume is None else str(Path(resume).resolve()),
        'restart_resource_bound': restart_bound,
        'last_checkpoint': checkpoint_paths[-1] if checkpoint_paths else None,
        'timings': {'invocation_step_loop_and_checkpoints_wall_s': loop_s,
                    'setup_wall_s': loop_start - start_wall,
                    'jit_note': 'first invocation includes lazy compilation; timings are observational'
                                + ('' if backend == 'numpy' else '; device runs: loop time read after an explicit '
                                   'stream synchronization, checkpoint writes included')},
        'backend': {'hydraulics': implementation, 'arrays': backend, 'root_solver': root_solver,
                    'device_id': resolved.device_id,
                    'fingerprint': json.loads(json.dumps(resolved.fingerprint, default=str)),
                    'pairing': control.backend_record(), 'loop_transfer_counters': transfers,
                    'maple_source_change_allowed': allow_maple_source_change,
                    'maple_source_changed_since_import': bool(verified_checks['maple_source_changed_since_import']),
                    'gpu': gpu_record},
        'transport': {**control.transport_record(), **phase_summary(result)},
        'routing_refresh': {**routing_refresh_record(context), 'n_commits': result.n_commits,
                            'n_reroutes': result.n_reroutes, 'n_net_commits': result.n_net_commits,
                            'n_forced_commits': result.n_forced_commits, 'n_graph_changes': result.n_graph_changes,
                            'n_routing_gate_evaluations': result.n_routing_gate_evaluations,
                            'max_unrouted_net_displacement_m': result.max_unrouted_net_displacement_m,
                            'routing_source_origin': result.routing_source_origin,
                            'routing_source_t_s': result.routing_source_t_s},
        'terminal_storage': {**terminal_record,
                             'final_n_terminal_storage_cells': 0 if final.graph.pit_storage is None
                             else int(np.sum(np.asarray(final.graph.pit_storage))),
                             'model': 'opt-in terminal storage: a cell with no directed downslope path deposits its '
                                      'mobile load through MAPLE and keeps its water; no pond residence, spill or '
                                      'overtopping; strict default unchanged'},
        'hydrograph_semantics': 'accepted physical storm, final row includes terminal settling/commit before external dry reset',
        'limitations': [_dense_sink_limitation(graph_policy), 'no wind event, splash, ecology or ET',
                        'original MAHLERAN equations checked separately; not a full MAHLERAN storm benchmark']
                       + ([] if backend == 'numpy' else [('whole small bed on the device; host checkpoints; '
                                                          'same-backend resume only; no speed claim')]),
    }
    if result.deep_publication is not None:
        # CW-B102: operational record of the LAST evolve segment (per execution; never a cumulative restart total)
        summary['deep_publication'] = dict(result.deep_publication)
    if deep_record is not None:
        # CW-B108: present ONLY for a non-default deep-chunk encoding (default summaries unchanged).
        final_store = final.bed.voxel_column.deep_store
        summary['deep_storage'] = {
            **deep_record, 'final_store_format_version': int(final_store.format_version),
            'final_physical_manifest_digest': final_store.manifest_digest,
            'final_logical_manifest_digest': final_store.logical_manifest_digest,
            'invocation_store_counter_delta': deep_delta,
            'scope': 'operational: MAPLE process-global deep-store / overlay counters differenced around THIS '
                     'invocation\'s event loop (checkpoint writes included); not physical state, not cumulative '
                     'across a restart, not a storage-fit or speed claim'}
    if windowed:
        if not setup_present:  # unreachable in practice (the default factory builds a dense bed); kept explicit
            raise SedimentEventError('a resident-window bed needs the case-setup run configuration')
        summary['bed_storage'] = run_configuration
        if backend == 'cupy':
            gpu_record['claims'] = 'resident MAPLE window/compact ledger with immutable host deep storage; same-backend restart'
        summary['limitations'] = ['no wind event, splash, ecology, ET or pit overtopping',
                                  'versioned deep files retained externally for same-source restart',
                                  'native correspondence and large-domain timing require executed case qualification']
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.partial-', dir=output.parent))
    try:
        if complete:
            save_checkpoint(staging / 'handoff', outcome, context, column, sediment, identity)
        voxel_fields = ({name:host(getattr(final.bed.voxel_column,name)) for name in
                         ('window_kg','base','deep_by_class_kg','deep_total_kg','deep_room_kg')} if windowed
                        else {'voxel_mass_kg':host(final.bed.voxel_column.mass_kg)})
        np.savez(staging / 'final_state.npz',
                 **voxel_fields, active_mass_kg=host(final.bed.active_layer.mass_kg),
                 available_mass_kg=host(final.bed.sediment_availability.available_mass_kg),
                 bound_mass_kg=host(final.bed.sediment_availability.bound_mass_kg),
                 committed_elevation_m=host(committed_elevation_m(final.bed.committed_topography)),
                 depth_m=host(final.storm.depth_m), soil_water_m=host(final.storm.soil_water_m),
                 discharge_m2_s=host(final.storm.discharge_m2_s),
                 mobile_mass_kg=host(final.bed.water.mobile_mass_by_cell_class_kg),
                 sediment_velocity_m_s=host(final.sediment_velocity_m_s),
                 pre_reset_depth_m=host(result.state.storm.depth_m),
                 pre_reset_soil_water_m=host(result.state.storm.soil_water_m),
                 terminal_deposition_kg=host(outcome.terminal_deposition_kg) if complete
                 else np.zeros(host(final.bed.active_layer.mass_kg).shape, dtype=np.float64),
                 **phase_fields(final.phase))
        hydrograph, sediment_hydrograph = host(result.hydrograph), host(result.sediment_hydrograph)
        rows = {name: hydrograph[:, i] for i, name in enumerate(HYDROGRAPH_COLUMNS)}
        rows.update({f'sed_{name}': sediment_hydrograph[:, i] for i, name in enumerate(result.sediment_columns)})
        np.savez(staging / 'hydrograph.npz', **rows)
        if record_exchange_history:
            # CW-D2: the event-lifetime exchange history (committed receipts + the live pending ledger ONCE) as
            # host arrays, with the D1 evidence binding. Written only when the option is on: default outputs
            # are unchanged. A completed handoff has an empty ledger, so its pending part is zero.
            totals = exchange_history_totals(result)
            arrays = {name: host(value) for name, value in totals['arrays'].items()}
            np.savez(staging / 'exchange_history.npz', **arrays)
            row_hashes = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()}
            summary['exchange_history'] = {
                'schema': 'maple-syrup-exchange-history/1',
                'n_commits_recorded': totals['n_commits_recorded'], 'n_commits': result.n_commits,
                'history_complete': totals['history_complete'], 'includes_pending_ledger': True,
                'pending_ledger_empty': totals['pending_ledger_empty'],
                'terminal_settlement_and_forced_commit_included': complete,
                'receipt': totals['receipt'], 'touch_semantics_source': 'MAPLE CommitResult.exchange_history_receipt',
                'arrays': {name: {'dtype': value.dtype.str, 'shape': list(value.shape), 'sha256': row_hashes[name]}
                           for name, value in arrays.items()},
                'binding': {'syrup_source_sha256': source_hash, 'maple_source_sha256': maple_hash,
                            'case_identity_sha256': identity['case_identity_sha256'],
                            'rows_sha256': row_hashes, 'backend': backend},
                'qualification': 'diagnostic evidence for the operand-qualified correspondence comparator; not a '
                                 'conservation result and not a change to any gate',
            }
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


def _deep_store_counters():
    """CW-B108: copies of MAPLE's deep-store and overlay operation counters (accepted typed source only)."""
    from maple.surface.voxels.deep_store import read_deep_store_counters, read_overlay_counters

    return {'deep_store': dict(read_deep_store_counters()), 'overlay': dict(read_overlay_counters())}


def run_complete_plot1(case_dir, output_dir, **kwargs):
    """Original Plot1 workflow; shared loop also accepts explicit case setup."""
    return run_complete_case(case_dir, output_dir, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--max-checkpoint-bytes', type=int, default=DEFAULT_MAX_CHECKPOINT_BYTES,
                        help='bound on the summed uncompressed payload bytes of the --resume bundle (positive '
                             'integer, default 1 GiB); a resource control, not part of the run identity')
    parser.add_argument('--checkpoint-every-s', type=float, default=600.)
    parser.add_argument('--pause-at-s', type=float)
    parser.add_argument('--max-end-s', type=float)
    parser.add_argument('--max-dt-s', type=float, default=1.)
    parser.add_argument('--implementation', choices=STORM_IMPLEMENTATIONS, default='numba',
                        help="hydraulic sweep: numba (default) or array on the host; cuda requires --backend cupy")
    parser.add_argument('--backend', choices=tuple(BACKEND_PAIRINGS), default='numpy',
                        help='numpy (array/numba) or cupy (ONLY with --implementation cuda); other pairings refused')
    parser.add_argument('--bisection-iterations',type=int,default=40,
                        help='explicit root iteration budget; accuracy and conservation bounds unchanged')
    parser.add_argument('--root-solver', choices=('bisection', 'newton'), default='bisection')
    parser.add_argument('--cuda-mode', choices=CUDA_MODES, default='auto',
                        help='CUDA hydrology launch structure (cupy backend only)')
    parser.add_argument('--allow-maple-source-change', action='store_true',
                        help='accept an explicitly recorded isolated MAPLE source difference from the import binding')
    parser.add_argument('--allow-pit-storage', action='store_true',
                        help='opt-in: active strict sinks become terminal-storage cells (deposit, keep water, no export)')
    parser.add_argument('--allow-flat-storage', action='store_true',
                        help='opt-in with --allow-pit-storage: flat sinks are terminal storage too')
    parser.add_argument('--allow-masked-nodata', action='store_true',
                        help='opt-in: inactive / ring cells may hold --nodata-value and are never receivers')
    parser.add_argument('--nodata-value', type=float, help='the DEM nodata sentinel for --allow-masked-nodata')
    parser.add_argument('--terminal-storage-water', action='store_true',
                        help='completion option: flow cessation with water ponded ONLY in terminal-storage cells '
                             '(removed and accounted by the external dry reset); needs --allow-pit-storage')
    parser.add_argument('--deposit-mobile-at-flow-end', action='store_true',
                        help='CW-B51 completion option: explicit dry-again assumption; after rain ended and every '
                             'hydraulic quiet gate held for --hold-s, the remaining finite mobile load is deposited '
                             'through MAPLE (reported separately); not a mass tolerance; default off')
    parser.add_argument('--record-exchange-history', action='store_true',
                        help='CW-D2: record the actual per-cell/class MAPLE ledger touch counts and gross activity '
                             'through every commit (exchange_history.npz); diagnostic evidence only, default off')
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
                        help='characteristic kernel: auto (follow --implementation), array or numba (dense storage), '
                             'numba or cuda (csr storage); no fallback')
    parser.add_argument('--phase-storage', choices=PHASE_STORAGES, default='dense',
                        help='phase storage of the characteristic scheme: dense (default) or csr (sparse nonempty-bin '
                             'state, same physics and bins; CSR checkpoints and final arrays)')
    parser.add_argument('--phase-max-nnz', type=int, default=None,
                        help='csr only: predeclared capacity of stored bins, refused before allocation when exceeded; '
                             'never truncates mass')
    parser.add_argument('--sediment-courant-max', type=float, default=CHARACTERISTIC_COURANT_MAX)
    parser.add_argument('--mahleran-root', type=Path)
    parser.add_argument('--expected-maple-root', type=Path)
    args = vars(parser.parse_args(argv))
    policy = CompletionPolicy(**{name: args.pop(name) for name in (
        'hold_s', 'max_depth_m', 'surface_volume_m3', 'max_unit_discharge_m2_s',
        'outlet_discharge_m3_s', 'mobile_per_cell_class_kg', 'terminal_storage_water',
        'deposit_mobile_at_flow_end')})
    try:
        run = run_complete_plot1(**args, policy=policy)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(run.summary, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
