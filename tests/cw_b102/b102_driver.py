"""CW-B102 focused scenario driver (one scenario per process; values-only JSON out).

Fixture: the B102 experiment's seeded resident-window valley (`b102_experiment.build`: 32 buried voxels per cell, one
ULP under capacity on a checkerboard, exact capacity elsewhere; deep store in tmp; commits every 2 s; reports every
6 s; 12 s of event). Tree = a SYRUP source tree (the accepted B100 freeze, or the B102 candidate); MAPLE = frozen M94.

Scenarios:
  run        evolve 0 -> 12 s under --mode step|boundary (--high-water): per-step physical digests, staged flags,
             boundary logical columns, final result, deep files and counters, publication record, previous-state
             liveness at every commit fresh-ledger allocation (B100 probe), host / pool memory, overlay and
             canonicalization cost. The B100 tree runs 'step' only (its public hook).
  callbacks  hook / policy combinations: refusals before any work; what each hook sees.
  flush      bounded flush / redo: injected exact M94 append-cap text on the second attempt (carried > 0) and its
             negatives; a REAL M94 append-cap refusal (small context cap, nothing carried) and the recognizer.
  failures   injected commit / reroute / boundary-publication failures and a real deep-store I/O failure inside
             apply_journal (orphan chunk files recorded, never cleaned).
  complete   complete_event to the dry handoff and a wet pause -> own checkpoint -> restart, under --mode.

Usage: python -B b102_driver.py --tree T --backend numpy|cupy --scenario S --mode step|boundary --tmp D --out J
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import tracemalloc
import traceback
import weakref
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
END_S, REPORT_S = 12.0, 6.0


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boundary_control(control, mode, high_water):
    if mode == 'step':
        return control
    return replace(control, deep_publication='boundary', deep_publication_high_water_records=high_water)


class Env:
    """The bound tree, the fixture builder and shared instrumentation."""

    def __init__(self, a):
        self.a = a
        self.exp = load('_b102_exp', HERE / 'b102_experiment.py')
        self.drv = self.exp.load_b100_driver(a.tree)
        self.drv.bind(a.tree)
        import numpy as np
        from maple.core.backend import resolve_backend
        self.np = np
        self.xp = resolve_backend(a.backend).xp
        import maple_syrup.sediment_event as se
        self.se = se
        self.pool = self.xp.get_default_memory_pool() if a.backend == 'cupy' else None

    def fixture(self, sub='fx'):
        tmp = Path(self.a.tmp) / sub
        tmp.mkdir()
        t0 = time.perf_counter()
        s, ctx, control, schedule, seeded, capacity = self.exp.build(self.drv, self.xp, tmp)
        return {'s': s, 'ctx': ctx, 'control': control, 'schedule': schedule, 'seeded': seeded,
                'store': tmp / 'store', 'setup_s': time.perf_counter() - t0}

    def evolve(self, fx, control, **kw):
        s = fx['s']
        return self.se.evolve_sediment_event(s['state0'], fx['ctx'], s['column'], s['field'], fx['schedule'],
                                             s['vegetation'], s['sediment'], END_S, control, report_every_s=REPORT_S,
                                             **kw)

    def phys(self, state):
        return self.exp.physical_digest(self.drv, state)


def scen_run(env, a):
    np, se = env.np, env.se
    import maple.surface.topographic_commit.windowed_commit as wc
    import maple.surface.voxels.deep_store as ds
    import maple.surface.voxels.windowed_transfer as wt
    import maple.water.windowed_step as wstep
    from maple.core.backend import to_host
    from maple.surface.voxels.windowed_inventory import windowed_to_dense
    fx = env.fixture()
    control = boundary_control(fx['control'], a.mode, a.high_water)
    state0 = fx['s']['state0']
    inputs_before = env.drv.raw(state0, 'state0')
    files_before = env.drv.store_files(fx['store'])
    per_step, boundaries, probes, prev = [], [], [], {}
    cost = {'overlay_calls': 0, 'overlay_s': 0.0, 'canonical_calls': 0, 'canonical_s': 0.0, 'apply_journal': []}

    def used():
        return int(env.pool.used_bytes()) if env.pool is not None else int(tracemalloc.get_traced_memory()[0])

    def observer(state):
        col, b = state.bed.voxel_column, state.bed
        per_step.append({'t': float(state.t_s), 'digest': env.phys(state), 'staged': col.pending_deep is not None,
                         'carried': 0 if col.pending_deep is None else int(col.pending_deep.journal.n_records),
                         'generation': int(col.generation),
                         'room_pos_seeded': int(np.count_nonzero(np.asarray(to_host(col.deep_room_kg))[fx['seeded']] > 0)),
                         'commit_count': int(b.committed_topography.commit_count)})
        prev.clear()
        for name, arr in (('ledger.pending', b.ledger.pending_bed_mass_change_kg),
                          ('ledger.touches', b.ledger.n_physical_touches), ('active', b.active_layer.mass_kg),
                          ('water.mobile', b.water.mobile_mass_by_cell_class_kg),
                          ('availability', b.sediment_availability.available_mass_kg), ('window', col.window_kg),
                          ('velocity', state.sediment_velocity_m_s), ('storm.depth', state.storm.depth_m)):
            prev[name] = weakref.ref(arr)
        if state.phase is not None:
            prev['phase.fraction'] = weakref.ref(state.phase.fraction)
        return False

    def boundary(result):
        col = result.state.bed.voxel_column
        dense = windowed_to_dense(col, max_bytes=10**8).mass_kg  # refuses a staged column
        boundaries.append({'t': float(result.state.t_s), 'published': col.pending_deep is None,
                           'generation': int(col.generation), 'manifest_digest': col.manifest_digest,
                           'dense_sha256': hashlib.sha256(np.ascontiguousarray(dense).tobytes()).hexdigest()})
        return False

    orig = {'zeros': wc.zeros_sediment_ledger_state, 'overlay': wt._apply_pending_records,
            'canon_wt': wt.working_bed_journal, 'canon_ws': wstep.working_bed_journal,
            'apply': ds.DeepStoreVersion.apply_journal}

    def zeros(*args, **kwargs):
        probes.append({'alive': {k: r() is not None for k, r in prev.items()}, 'used_bytes': used()})
        return orig['zeros'](*args, **kwargs)

    def overlay(bed, cells, columns):
        t0 = time.perf_counter()
        n = orig['overlay'](bed, cells, columns)
        cost['overlay_s'] += time.perf_counter() - t0
        cost['overlay_calls'] += 1
        return n

    def canonical(bed):
        t0 = time.perf_counter()
        j = orig['canon_wt'](bed)
        cost['canonical_s'] += time.perf_counter() - t0
        cost['canonical_calls'] += 1
        return j

    def apply(self, journal, geometry, mass_resolution_kg, **kw):
        out = orig['apply'](self, journal, geometry, mass_resolution_kg, **kw)
        if journal.n_records:
            cost['apply_journal'].append({'records': int(journal.n_records),
                                          'touched_chunks': int(np.unique(journal.cell_flat // self.header.chunk_columns).size)})
        return out

    wc.zeros_sediment_ledger_state, wt._apply_pending_records = zeros, overlay
    wt.working_bed_journal = wstep.working_bed_journal = canonical
    ds.DeepStoreVersion.apply_journal = apply
    hook = {'on_accepted_step': observer} if a.mode == 'step' else {'_staged_accepted_observer': observer}
    ds.reset_deep_store_counters()
    if env.pool is None:
        tracemalloc.start()
    t0 = time.perf_counter()
    try:
        result = env.evolve(fx, control, on_boundary=boundary, **hook)
    finally:
        wc.zeros_sediment_ledger_state, wt._apply_pending_records = orig['zeros'], orig['overlay']
        wt.working_bed_journal, wstep.working_bed_journal = orig['canon_wt'], orig['canon_ws']
        ds.DeepStoreVersion.apply_journal = orig['apply']
    event_s = time.perf_counter() - t0
    raw = env.drv.raw(result, 'result')
    after = env.drv.store_files(fx['store'])
    out = {'status': 'ok', 'setup_s': fx['setup_s'], 'event_wall_s': event_s,
           'n_accepted_steps': result.n_accepted_steps, 'n_commits': result.n_commits,
           'n_rejected_attempts': result.n_rejected_attempts,
           'final_published': result.state.bed.voxel_column.pending_deep is None,
           'final_generation': int(result.state.bed.voxel_column.generation),
           'deep_publication': getattr(result, 'deep_publication', 'field absent (B100 tree)'),
           'result_raw': raw, 'per_step': per_step, 'boundaries': boundaries, 'probes': probes,
           'cost': cost, 'deep_store_counters': ds.read_deep_store_counters(),
           'inputs_unchanged': env.drv.raw(state0, 'state0') == inputs_before,
           'store_preexisting_preserved': all(after.get(k) == v for k, v in files_before.items()),
           'store_files': after,
           'host_traced_peak_bytes': None if env.pool is not None else tracemalloc.get_traced_memory()[1]}
    if env.pool is None:
        tracemalloc.stop()
    return out


def scen_callbacks(env, a):
    se = env.se
    fx = env.fixture()
    calls = [0]
    orig = se._sediment_accounting_step

    def counted(*args, **kw):
        calls[0] += 1
        return orig(*args, **kw)
    se._sediment_accounting_step = counted
    files0 = env.drv.store_files(fx['store'])
    out = {}
    try:
        bcontrol = boundary_control(fx['control'], 'boundary', 10**5)
        for label, kw in (('boundary_public_hook', {'on_accepted_step': lambda s: False}),
                          ('both_hooks', {'on_accepted_step': lambda s: False,
                                          '_staged_accepted_observer': lambda s: False}),
                          ('private_not_callable', {'_staged_accepted_observer': 3})):
            calls[0] = 0
            try:
                env.evolve(fx, bcontrol, **kw)
                out[label] = {'refused': False}
            except se.SedimentEventError as exc:
                out[label] = {'refused': True, 'message': str(exc), 'attempts_before_refusal': calls[0],
                              'store_unchanged': env.drv.store_files(fx['store']) == files0}
        for mode in ('step', 'boundary'):
            seen = []

            def look(state, seen=seen):
                seen.append(state.bed.voxel_column.pending_deep is not None)
                return False
            sub = env.fixture(f'fx_{mode}')
            hook = {'on_accepted_step': look} if mode == 'step' else {'_staged_accepted_observer': look}
            bnd = []
            res = env.evolve(sub, boundary_control(sub['control'], mode, 10**5), **hook,
                             on_boundary=lambda r, bnd=bnd: bnd.append(r.state.bed.voxel_column.pending_deep is None))
            out[f'{mode}_hook_saw_staged'] = seen
            out[f'{mode}_boundary_saw_published'] = bnd
            out[f'{mode}_final_published'] = res.state.bed.voxel_column.pending_deep is None
    finally:
        se._sediment_accounting_step = orig
    return out


def scen_flush(env, a):
    np, se = env.np, env.se
    from maple_syrup.sediment_bed import is_carried_append_cap_refusal, resolved_journal_cap
    out = {}
    orig = se._sediment_accounting_step

    def run_case(label, *, inject_at=(), text=None, high_water=10**5, ctx_cap=None, observer=True):
        fx = env.fixture(label)
        ctx = fx['ctx'] if ctx_cap is None else replace(fx['ctx'], window_max_journal_records=ctx_cap)
        fx['ctx'] = ctx
        message = text or f'pending journal exceeds max_journal_records={resolved_journal_cap(ctx)}'
        calls, digests = [0], []

        def injected(*args, **kw):
            calls[0] += 1
            if calls[0] in inject_at:
                raise ValueError(message)
            return orig(*args, **kw)
        se._sediment_accounting_step = injected
        state0 = fx['s']['state0']
        before = env.drv.raw(state0, 'state0')
        rec = {}
        try:
            res = env.evolve(fx, boundary_control(fx['control'], 'boundary', high_water),
                             _staged_accepted_observer=(lambda s: digests.append(env.phys(s)) or False))
            rec.update(status='ok', deep_publication=res.deep_publication, n_rejected=res.n_rejected_attempts,
                       n_accepted=res.n_accepted_steps, digests=digests,
                       final_dense=hashlib.sha256(np.ascontiguousarray(
                           __import__('maple.surface.voxels.windowed_inventory', fromlist=['x']).windowed_to_dense(
                               res.state.bed.voxel_column, max_bytes=10**8).mass_kg).tobytes()).hexdigest())
        except BaseException as exc:  # noqa: BLE001
            rec.update(status='error', type=type(exc).__name__, message=str(exc),
                       recognized_carried1=is_carried_append_cap_refusal(exc, ctx, 1),
                       recognized_carried0=is_carried_append_cap_refusal(exc, ctx, 0),
                       trace_tail=traceback.format_exc()[-1500:])
        finally:
            se._sediment_accounting_step = orig
        rec['attempt_calls'] = calls[0]
        rec['inputs_unchanged'] = env.drv.raw(state0, 'state0') == before
        rec['injected_message'] = message
        return rec

    out['reference'] = run_case('reference')
    out['flush_redo'] = run_case('flush', inject_at=(2,))
    out['wrong_cap_text'] = run_case('wrongcap', inject_at=(2,), text='pending journal exceeds max_journal_records=7')
    out['reworded'] = run_case('reworded', inject_at=(2,), text='pending journal exceeds the max_journal_records cap')
    out['prior_oversized_r1'] = run_case('r1', inject_at=(2,), text='prior pending journal exceeds max_journal_records')
    out['nothing_carried'] = run_case('carry0', inject_at=(1,))
    out['second_refusal'] = run_case('twice', inject_at=(2, 3))
    out['real_m94_single_step_overflow'] = run_case('real', ctx_cap=1000, high_water=999)
    return out


def scen_failures(env, a):
    np, se = env.np, env.se
    import maple.surface.voxels.deep_store as ds
    out = {}

    def case(label, patch):
        fx = env.fixture(label)
        state0 = fx['s']['state0']
        before = env.drv.raw(state0, 'state0')
        files0 = env.drv.store_files(fx['store'])
        undo = patch(fx)
        rec = {}
        try:
            env.evolve(fx, boundary_control(fx['control'], 'boundary', 10**5),
                       _staged_accepted_observer=lambda s: False)
            rec['status'] = 'ok'
        except BaseException as exc:  # noqa: BLE001
            rec.update(status='error', type=type(exc).__name__, message=str(exc)[:500])
        finally:
            undo()
        after = env.drv.store_files(fx['store'])
        new = sorted(set(after) - set(files0))
        rec.update(inputs_unchanged=env.drv.raw(state0, 'state0') == before,
                   preexisting_preserved=all(after.get(k) == v for k, v in files0.items()),
                   new_files=new, new_manifests=[k for k in new if k.startswith('versions/')],
                   orphan_chunks=[k for k in new if k.startswith('chunks/')] if not any(
                       k.startswith('versions/') for k in new) else [])
        return rec

    def commit_failure(fx):
        orig, n = se.commit_bed, [0]

        def bad(*args, **kw):
            res = orig(*args, **kw)
            if res.did_commit:
                n[0] += 1
                if n[0] == 2:
                    raise RuntimeError('B102 injected commit failure (second commit, carried journal)')
            return res
        se.commit_bed = bad
        return lambda: setattr(se, 'commit_bed', orig)

    def reroute_failure(fx):
        orig, n = se.physics_grid_from_graph, [0]

        def bad(graph):
            n[0] += 1
            if n[0] == 2:
                raise RuntimeError('B102 injected reroute failure')
            return orig(graph)
        se.physics_grid_from_graph = bad
        return lambda: setattr(se, 'physics_grid_from_graph', orig)

    def publication_failure(fx):
        orig = se.publish_bed_storage

        def bad(bed, context):
            raise OSError('B102 injected boundary-publication failure')
        se.publish_bed_storage = bad
        return lambda: setattr(se, 'publish_bed_storage', orig)

    def io_failure(fx):
        orig, n = ds._write_bytes_once, [0]

        def bad(path, data, *, expected_sha256):
            n[0] += 1
            if n[0] == 3:  # pass 2 of the first boundary publication: some chunk files already sealed
                raise OSError('B102 injected deep-store write failure')
            return orig(path, data, expected_sha256=expected_sha256)
        ds._write_bytes_once = bad
        return lambda: setattr(ds, '_write_bytes_once', orig)

    for label, patch in (('commit', commit_failure), ('reroute', reroute_failure),
                         ('publication', publication_failure), ('io_orphans', io_failure)):
        out[label] = case(label, patch)
    return out


def scen_complete(env, a):
    np, se = env.np, env.se
    from maple.surface.voxels.windowed_inventory import windowed_to_dense
    from test_exchange_history import IDENTITY

    from maple_syrup.checkpoint import load_checkpoint, save_checkpoint
    from maple_syrup.complete_event import CompletionPolicy, DryHandoff, complete_event
    out = {}

    def chain(sub):
        """The accepted restart test's windowed chain fixture (`test_windowed_storm.window_inputs`: 3-cell chain,
        CSR, deep resident window, interval commits every 2 s, history ON), which reaches the dry handoff within
        180 s (the seeded 64 x 32 valley does not, under either policy, in run1), with ITS 32 buried voxels seeded
        one ULP under capacity in every cell (`b102_experiment.seed_columns`) so a real MAPLE pending journal is
        produced and carried. Same construction steps as `window_inputs`, seeding inserted before windowing."""
        from maple.core.backend import to_device_tree, to_host_tree
        from maple.core.types.sediment_ledger import process_index, zeros_sediment_ledger_state
        from maple.core.types.topographic_commit import committed_elevation_m
        from maple.core.types.voxel import VoxelColumnState
        from maple.surface.topographic_commit.initialization import (
            initial_committed_topography_state_from_physical_state,
        )
        from maple.surface.topographic_commit.windowed_commit import windowed_committed_state_from_dense_baseline
        from maple.surface.voxels.capacity import max_voxel_mass_kg
        from maple.surface.voxels.windowed_inventory import windowed_column_from_dense
        from maple.surface.voxels.windowed_prefix import move_prefix_packets
        from test_exchange_history import inputs

        from maple_syrup.sediment_bed import refresh_routing
        from maple_syrup.sediment_event import initial_event_state
        tmp = Path(a.tmp) / sub
        tmp.mkdir()
        implementation = 'numba' if env.xp is np else 'cuda'
        s, ctx, control, schedule = inputs(env.xp, commit_interval_s=2.0, implementation=implementation)
        control = replace(control, phase_storage='csr')
        state = s['state0']
        host = to_host_tree(state.bed)
        deep = np.repeat(host.voxel_column.mass_kg[:, :, :1, :], 32, axis=2)
        dense = np.concatenate((deep, host.voxel_column.mass_kg), axis=2).copy()
        env.exp.seed_columns(dense, max_voxel_mass_kg(ctx.geometry), np.ones(dense.shape[:2], dtype=bool))
        dense_column = VoxelColumnState(dense)
        baseline = initial_committed_topography_state_from_physical_state(dense_column, host.active_layer,
                                                                         ctx.geometry, ctx.mass_resolution_kg)
        host = replace(host, voxel_column=dense_column, committed_topography=baseline)
        old_ref = state.terrain
        shift = committed_elevation_m(baseline) - old_ref.initial_committed_elevation_m
        full = old_ref.initial_full_elevation_m.copy() + float(shift[0, 0])
        full[1:-1, 1:-1] = old_ref.initial_full_elevation_m[1:-1, 1:-1] + shift
        ref = replace(old_ref, initial_full_elevation_m=full,
                      initial_committed_elevation_m=committed_elevation_m(baseline).copy())
        column = windowed_column_from_dense(host.voxel_column, ctx.geometry, ctx.mass_resolution_kg,
                                            store_dir=tmp / 'store', max_store_bytes=10**8,
                                            class_ids=tuple(g.class_id for g in ctx.grain_classes.classes),
                                            window_voxels=min(4, dense.shape[2]), chunk_columns=64,
                                            with_prefix_packets=True)
        committed, _ = windowed_committed_state_from_dense_baseline(host.committed_topography, column,
                                                                    host.active_layer, ctx.geometry)
        ledger = zeros_sediment_ledger_state(ctx.geometry.ny, ctx.geometry.nx, len(ctx.grain_classes.classes),
                                             present_processes=(process_index('water_erosion_deposition'),))
        bed = replace(host, voxel_column=column, committed_topography=committed, ledger=ledger)
        if env.xp is not np:
            moved = to_device_tree(bed, env.xp)
            bed = replace(moved, voxel_column=move_prefix_packets(moved.voxel_column, column, env.xp))
        graph = refresh_routing(ref, bed, xp=env.xp)
        s['state0'] = initial_event_state(graph, ref, bed, ctx, state.storm.soil_water_m, control=control)
        return {'s': s, 'ctx': ctx, 'control': control, 'schedule': schedule}

    def complete(fx, **kw):
        s = fx['s']
        control = boundary_control(fx['control'], a.mode, 10**5)
        return complete_event(s['state0'], fx['ctx'], s['column'], s['field'], fx['schedule'], s['vegetation'],
                              s['sediment'], max_end_s=180, control=control, policy=CompletionPolicy(hold_s=2),
                              report_every_s=2.0, **kw)

    def dense(col):
        return hashlib.sha256(np.ascontiguousarray(windowed_to_dense(col, max_bytes=10**8).mass_kg).tobytes()).hexdigest()

    fx = chain('ref')
    ref = complete(fx)
    out['reference'] = {'dry': isinstance(ref, DryHandoff), 'published': ref.dry_state.bed.voxel_column.pending_deep is None,
                        'progress_published': ref.progress.result.state.bed.voxel_column.pending_deep is None,
                        'n_accepted': ref.progress.result.n_accepted_steps, 'n_commits': ref.progress.result.n_commits,
                        'closed': ref.progress.result.closure()['closed'], 'dry_dense': dense(ref.dry_state.bed.voxel_column),
                        'accounting_sha256': hashlib.sha256(json.dumps(ref.accounting, sort_keys=True, default=str).encode()).hexdigest(),
                        'deep_publication': ref.progress.result.deep_publication,
                        'dry_raw': env.drv.raw(ref.dry_state, 'dry')}
    fx2 = chain('pause')
    paused = complete(fx2, checkpoint_callback=lambda p: True)
    ckpt = Path(a.tmp) / 'wet'
    save_checkpoint(ckpt, paused, fx2['ctx'], fx2['s']['column'], fx2['s']['sediment'], IDENTITY)
    restored = load_checkpoint(ckpt, fx2['ctx'], fx2['s']['column'], fx2['s']['sediment'], IDENTITY)
    resumed = complete(dict(fx2, s=dict(fx2['s'], state0=restored.result.state)), continuation=restored)
    out['restart'] = {'paused_t': float(paused.result.state.t_s),
                      'paused_published': paused.result.state.bed.voxel_column.pending_deep is None,
                      'restored_control_equal': restored.control == paused.control,
                      'resumed_dry': isinstance(resumed, DryHandoff),
                      'dry_dense_equal_reference': dense(resumed.dry_state.bed.voxel_column) == out['reference']['dry_dense'],
                      'dry_raw_nondeep_equal': {k: v for k, v in env.drv.raw(resumed.dry_state, 'dry').items()
                                                if 'deep_store' not in k and 'generation' not in k and 'manifest' not in k}
                      == {k: v for k, v in out['reference']['dry_raw'].items()
                          if 'deep_store' not in k and 'generation' not in k and 'manifest' not in k},
                      'n_accepted_equal': resumed.progress.result.n_accepted_steps == ref.progress.result.n_accepted_steps,
                      'n_commits_equal': resumed.progress.result.n_commits == ref.progress.result.n_commits,
                      'accounting_equal': resumed.accounting == ref.accounting}
    return out


def scen_retry(env, a):
    """CW-B102C1: the root reproducer concepts on the coupled event path ('boundary'). Attempt 2 (t = 1 s, a carried
    journal) raises M94's exact append-cap text with a failed-attempt payload alive in its frame; attempt 3 is the
    post-flush redo. Probes (weak references only): the failed payload and the active exception at flush entry (the
    first publication) and at redo entry."""
    import sys as _sys

    np, se, xp = env.np, env.se, env.xp
    from maple_syrup.sediment_bed import resolved_journal_cap
    out = {}

    def case(label, second, third):
        fx = env.fixture(label)
        orig, pub = se._sediment_accounting_step, se.publish_bed_storage
        calls, refs, probes, digests = [], [], [], []
        before = env.drv.raw(fx['s']['state0'], 'state0')
        files0 = env.drv.store_files(fx['store'])
        cap_text = f'pending journal exceeds max_journal_records={resolved_journal_cap(fx["ctx"])}'

        def injected(*args, **kw):
            calls.append({'t_s': float(args[0].t_s), 'dt_s': float(args[6]),
                          'failed_payload_alive_at_entry': (refs[-1]() is not None) if refs else None,
                          'exception_active_at_entry': _sys.exception() is not None})
            n = len(calls)
            if n == 2 and second is not None:
                payload = xp.ones(16_384, dtype=np.float64)  # failed-attempt payload (device array on CuPy)
                refs.append(weakref.ref(payload))
                raise ValueError(cap_text) if second == 'cap' else se.RoutingStepRejected('B102C1 first ordinary rejection')
            if n == 3 and third is not None:
                raise {'routing': se.RoutingStepRejected('B102C1 post-flush routing rejection'),
                       'transport': se.TransportStepRejected('B102C1 post-flush transport rejection'),
                       'valueerror': ValueError('B102C1 post-flush plain ValueError')}[third]
            if n >= 4 and third is not None:
                raise RuntimeError('B102C1 sentinel: prohibited fourth attempt reached')
            return orig(*args, **kw)

        def publish(bed, context):
            probes.append({'failed_payload_alive': (refs[-1]() is not None) if refs else None,
                           'exception_active': _sys.exception() is not None})
            return pub(bed, context)
        se._sediment_accounting_step, se.publish_bed_storage = injected, publish
        rec = {}
        try:
            res = env.evolve(fx, boundary_control(fx['control'], 'boundary', 10**5),
                             _staged_accepted_observer=lambda s: digests.append(env.phys(s)) or False)
            rec.update(status='ok', n_rejected=res.n_rejected_attempts, n_accepted=res.n_accepted_steps,
                       deep_publication=res.deep_publication, rejections=list(res.rejections))
            del res
        except BaseException as exc:  # noqa: BLE001
            rec.update(status='error', type=type(exc).__name__, message=str(exc))
        finally:
            se._sediment_accounting_step, se.publish_bed_storage = orig, pub
        after = env.drv.store_files(fx['store'])
        rec.update(calls=calls, publication_probes=probes, digests=digests,
                   failed_payload_alive_after=(refs[-1]() is not None) if refs else None,
                   inputs_unchanged=env.drv.raw(fx['s']['state0'], 'state0') == before,
                   preexisting_files_preserved=all(after.get(k) == v for k, v in files0.items()),
                   new_manifests=sorted(k for k in set(after) - set(files0) if k.startswith('versions/')))
        return rec

    out['reference'] = case('reference', None, None)
    out['flush_then_ok'] = case('flush_ok', 'cap', None)
    for third in ('routing', 'transport', 'valueerror'):
        out[f'post_flush_{third}'] = case(f'post_{third}', 'cap', third)
    out['first_ordinary_rejection'] = case('first_rejection', 'reject', None)
    return out


def scen_entry(env, a):
    """CW-B102C1: invalid hook / policy combinations are refused BEFORE `_check_event_inputs` (the first
    allocation-producing validation) and before any attempt."""
    se = env.se
    fx = env.fixture()
    seen = {'check_event_inputs': 0}
    orig = se._check_event_inputs

    def counted(*args, **kw):
        seen['check_event_inputs'] += 1
        return orig(*args, **kw)
    se._check_event_inputs = counted
    out = {}
    try:
        bcontrol = boundary_control(fx['control'], 'boundary', 10**5)
        for label, control, kw in (
                ('boundary_public_hook', bcontrol, {'on_accepted_step': lambda s: False}),
                ('both_hooks_step', fx['control'], {'on_accepted_step': lambda s: False,
                                                    '_staged_accepted_observer': lambda s: False}),
                ('private_not_callable_step', fx['control'], {'_staged_accepted_observer': 3}),
                ('boundary_on_boundary_not_callable', bcontrol, {'on_boundary': 3,
                                                                 '_staged_accepted_observer': lambda s: False})):
            seen['check_event_inputs'] = 0
            try:
                env.evolve(fx, control, **kw)
                out[label] = {'refused': False}
            except se.SedimentEventError as exc:
                out[label] = {'refused': True, 'message': str(exc),
                              'check_event_inputs_calls_before_refusal': seen['check_event_inputs']}
        seen['check_event_inputs'] = 0
        res = env.evolve(fx, fx['control'])  # default: no hooks -> unchanged path, the validation runs once
        out['default_no_hooks'] = {'check_event_inputs_calls': seen['check_event_inputs'],
                                   'n_accepted': res.n_accepted_steps}
    finally:
        se._check_event_inputs = orig
    return out


def scen_arrays(env, a):
    """CW-B102C1: host copies of every physical array at each report boundary (published states), saved as NPZ for
    the declared-bound CPU / GPU comparison (`cw_b102c1_compare_arrays.py`)."""
    np = env.np
    from maple.core.backend import is_array, to_host
    from maple.surface.voxels.windowed_inventory import windowed_to_dense
    import dataclasses
    fx = env.fixture()
    control = boundary_control(fx['control'], a.mode, 10**5)
    saved = {}

    def walk(v, path):
        if is_array(v):
            saved[path] = np.array(to_host(v), copy=True)
        elif dataclasses.is_dataclass(v) and not isinstance(v, type):
            for f in dataclasses.fields(v):
                if f.name not in ('xp',):
                    walk(getattr(v, f.name), f'{path}.{f.name}')

    def boundary(result):
        st = result.state
        tag = f't{st.t_s:g}'
        col = st.bed.voxel_column
        for name in ('window_kg', 'base', 'deep_by_class_kg', 'deep_total_kg', 'deep_room_kg'):
            saved[f'{tag}/bed.voxel_column.{name}'] = np.array(to_host(getattr(col, name)), copy=True)
        saved[f'{tag}/dense_logical'] = np.array(windowed_to_dense(col, max_bytes=10**8).mass_kg, copy=True)
        for name in ('water', 'active_layer', 'ledger', 'sediment_availability', 'committed_topography'):
            walk(getattr(st.bed, name), f'{tag}/bed.{name}')
        walk(st.storm, f'{tag}/storm')
        saved[f'{tag}/sediment_velocity_m_s'] = np.array(to_host(st.sediment_velocity_m_s), copy=True)
        walk(st.phase, f'{tag}/phase')
        return False
    hook = {'on_accepted_step': lambda s: False} if a.mode == 'step' else {'_staged_accepted_observer': lambda s: False}
    env.evolve(fx, control, on_boundary=boundary, **hook)
    path = Path(a.tmp) / f'arrays_{a.backend}_{a.mode}.npz'
    np.savez(path, **{k.replace('/', '__'): v for k, v in saved.items()})
    return {'npz': str(path), 'npz_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'n_arrays': len(saved)}


SCENARIOS = {'run': scen_run, 'callbacks': scen_callbacks, 'flush': scen_flush, 'failures': scen_failures,
             'complete': scen_complete, 'retry': scen_retry, 'entry': scen_entry, 'arrays': scen_arrays}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--tree', required=True)
    p.add_argument('--backend', choices=('numpy', 'cupy'), required=True)
    p.add_argument('--scenario', choices=sorted(SCENARIOS), required=True)
    p.add_argument('--mode', choices=('step', 'boundary'), default='step')
    p.add_argument('--high-water', type=int, default=10**5)
    p.add_argument('--tmp', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    env = Env(a)
    rec = {'tree': str(Path(a.tree).resolve()), 'backend': a.backend, 'scenario': a.scenario, 'mode': a.mode,
           'high_water': a.high_water}
    if env.pool is not None:
        dev = env.xp.cuda.runtime.getDeviceProperties(0)
        rec['device'] = {'name': dev['name'].decode() if isinstance(dev['name'], bytes) else dev['name'],
                         'uuid': 'GPU-' + bytes(dev['uuid']).hex()}
    rec['result'] = SCENARIOS[a.scenario](env, a)
    with open(a.out, 'x') as f:
        json.dump(rec, f, indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
