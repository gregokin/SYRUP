"""CW-B109 reusable bounded fixture helper: separate-process overlay wet restart with deep publication AFTER the disk
resume (derived from agent_handoffs/tasks/conservative_water/cw_b108c1_restart.py, which stays unchanged).

Declared fixture: tests/cw_b108/b108_fixture.py (12x7 cells, 40 voxels, 3 classes, overlay encoding, the MAPLE
default net threshold / caps, the same labelled B108 stopping criteria). LABELLED TEST CADENCE: pause at the original
1 s report boundary with report_every_s = 1 s (the same fixture publishes deep version 1 at t = 1 s and version 2 at
t = 2 s, so the second publication happens only after the resume). The held routing source may have zero lag at 1 s;
the non-zero-lag proof is the separate B108C1 12 s evidence.

Stages, each its OWN child process (cw_b109_run.py):
  reference BACKEND RUN_DIR   uninterrupted event                          -> reference.{npz,json}
  pause     BACKEND RUN_DIR   paused at 1 s, NEW bundle                    -> pause.{npz,json} (+ every store file hash)
  resume    BACKEND RUN_DIR   NEW process: fixture-reconstructed context, every state array from the bundle and the
                              saved store, continued to the dry handoff    -> resume.{npz,json}
  compare   RUN_DIR OUT.json  host-only exact comparison + post-restart publication checks
  controls  RUN_DIR OUT.json  host-only negative controls on copies of the actual outputs (every one must be detected)

Comparison rules: exact key sets at EVERY level (a missing key never equals a None value), exact JSON types (2 != 2.0),
exact array dtype / shape / bytes, unknown fields compared (never dropped). Only the individually named fields in
QUALIFIED_BOTH / RESUME_ONLY are not compared for equality; RESUME_ONLY facts are checked by
`check_post_restart_publication` instead.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PAUSE_S, MAX_END_S, REPORT_S = 1.0, 300.0, 1.0  # LABELLED test cadence (pause and report at 1 s)
IDENTITY = {'case': 'cw_b109 separate-process post-restart overlay publication', 'source': 'declared 12x7x40x3 fixture'}
#: Present in both metadata records, differ by construction (process identity / paths / wall time), not compared.
QUALIFIED_BOTH = ('roots', 'stage_wall_s', 'store_dir')
#: Present only in the resume record; verified by `check_post_restart_publication`, not by equality.
RESUME_ONLY = ('post_restart',)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def files(root):
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(Path(root).rglob('*')) if p.is_file()}


# --- pure comparison / check functions (host only; imported by tests/cw_b109) ------------------------------------------
def compare_values(a, b, path=''):
    """Exact recursive comparison: identical key sets, identical JSON types, identical values. Returns diff paths."""
    if type(a) is not type(b):
        return [f'{path}: type {type(a).__name__} != {type(b).__name__}']
    if isinstance(a, dict):
        out = [f'{path}/{k}: missing in {"B" if k in a else "A"}' for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += compare_values(a[k], b[k], f'{path}/{k}')
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} != {len(b)}']
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += compare_values(x, y, f'{path}[{i}]')
        return out
    return [] if a == b else [f'{path}: {str(a)[:80]} != {str(b)[:80]}']


def compare_metadata(ref, res):
    """Reference vs resume metadata: every key compared except the named QUALIFIED_BOTH (which must exist on both
    sides) and RESUME_ONLY (which must exist on the resume side only)."""
    diffs = []
    for k in QUALIFIED_BOTH:
        if k not in ref or k not in res:
            diffs.append(f'/{k}: qualified field missing')
    for k in RESUME_ONLY:
        if k in ref or k not in res:
            diffs.append(f'/{k}: resume-only field misplaced')
    strip = set(QUALIFIED_BOTH) | set(RESUME_ONLY)
    diffs += compare_values({k: v for k, v in ref.items() if k not in strip},
                            {k: v for k, v in res.items() if k not in strip})
    return diffs


def compare_arrays(ref, res):
    """`ref` / `res`: mapping name -> ndarray. Exact name set, dtype, shape and bytes."""
    diffs = [f'{k}: missing in {"resume" if k in ref else "reference"}' for k in sorted(set(ref) ^ set(res))]
    for k in sorted(set(ref) & set(res)):
        a, b = ref[k], res[k]
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            diffs.append(k)
    return diffs


def check_paused_binding(paused, loaded, pause_facts, loaded_facts):
    """The state loaded in the resume process equals the paused arrays (incl. the held routing source) byte for byte,
    and names the paused deep version (physical / logical digest, generation, version id, format)."""
    out = compare_arrays(paused, loaded)
    for k in ('physical_manifest_digest', 'logical_manifest_digest', 'generation', 'version_id', 'store_format_version',
              'store_dir'):
        if k not in loaded_facts or loaded_facts[k] != pause_facts[k]:
            out.append(f'binding {k}: {loaded_facts.get(k)!r} != paused {pause_facts[k]!r}')
    return out


def check_post_restart_publication(pause, post):
    """Deep publication AFTER the disk resume: a newer generation / version, new manifest AND new data (chunk or
    overlay) files that the final version references, a v2 final version whose verified lineage reaches the paused
    manifest, every paused store file retained unchanged, and positive continuing transport after the pause."""
    out = []
    f = pause['facts']
    if not post['final_generation'] > f['generation']:
        out.append('final generation is not newer than the paused generation')
    if not post['final_version_id'] > f['version_id']:
        out.append('final version id is not newer than the paused version')
    if post['final_store_format_version'] != 2 or not post['final_reload_ok']:
        out.append('final version is not a valid reloaded v2 version')
    if not post['lineage'] or post['lineage'][-1] != f['physical_manifest_digest']:
        out.append('verified lineage of the final version does not reach the paused manifest')
    new = post['new_store_files']
    if not any(n.startswith('versions/') for n in new):
        out.append('no new version manifest file after the resume')
    if not any(n in new for n in post['referenced_data_files']):
        out.append('no new data (chunk / overlay) file referenced by the final version')
    if post['changed_or_missing_paused_files']:
        out.append('paused store files changed or missing')
    if not set(pause['store_files']) <= set(post['store_files_after']):
        out.append('paused store file list not retained')
    for name in ('cumulative_pickup_kg', 'cumulative_deposition_kg'):
        if not post['after'][name] > f[name]:
            out.append(f'{name} did not increase after the resume')
    if not post['after']['n_accepted_steps'] > f['n_accepted_steps'] or not f['mobile_kg'] > 0:
        out.append('no continuing wet / mobile transport after the resume')
    return out


# --- child stages ----------------------------------------------------------------------------------------------------
def setup_imports():
    syrup_dir = Path(os.environ['CW_B108_EXPECT_SYRUP']).resolve()
    tree = syrup_dir.parents[1]
    sys.path[:0] = [str(tree / 'tests' / d) for d in ('cw_b108', 'phase5', 'conservative_water')]
    import maple
    import maple_syrup

    from maple_syrup.provenance import source_tree_digest
    from maple_syrup.sediment_bed import bound_maple_package_digest

    assert Path(maple.__file__).resolve().parent == Path(os.environ['CW_B108_EXPECT_MAPLE']).resolve()
    assert Path(maple_syrup.__file__).resolve().parent == syrup_dir
    import b108_fixture

    assert Path(b108_fixture.__file__).resolve().parent == tree / 'tests' / 'cw_b108'
    return {'maple_package': str(Path(maple.__file__).resolve().parent), 'syrup_package': str(syrup_dir),
            'fixture': str(Path(b108_fixture.__file__).resolve()), 'helper_sha256': sha(Path(__file__).read_bytes()),
            'maple_package_digest': bound_maple_package_digest(),
            'syrup_package_digest': source_tree_digest(syrup_dir).digest_sha256,
            'fixture_sha256': sha(Path(b108_fixture.__file__).read_bytes()), 'pid': os.getpid(),
            'cupy_imported': 'cupy' in sys.modules}


def namespace(backend):
    from maple.core.backend import resolve_backend

    return resolve_backend(backend).xp


def policy():
    from maple_syrup.complete_event import CompletionPolicy

    # The SAME labelled fixture stopping criteria as tests/cw_b108 (physical quiet gates, not conservation tolerances).
    return CompletionPolicy(hold_s=2.0, max_depth_m=5e-4, surface_volume_m3=1e-2, max_unit_discharge_m2_s=1e-5,
                            outlet_discharge_m3_s=1e-4, deposit_mobile_at_flow_end=True)


def configuration(ctx, control):
    """The run configuration both stages must share (compared exactly between reference and resume)."""
    return {'encoding': ctx.deep_chunk_encoding, 'overlay_max_bytes': ctx.deep_overlay_max_bytes,
            'window_max_journal_records': ctx.window_max_journal_records,
            'pending_elevation_trigger_m': float(ctx.topographic_commit_spec.pending_elevation_trigger_m),
            'commit_interval_s': float(ctx.topographic_commit_spec.commit_interval_s),
            'pause_s': PAUSE_S, 'report_every_s': REPORT_S, 'max_end_s': MAX_END_S, 'policy': repr(policy()),
            'control': repr(control)}


def store_facts(column):
    store = column.deep_store
    return {'physical_manifest_digest': store.manifest_digest, 'logical_manifest_digest': store.logical_manifest_digest,
            'generation': column.generation, 'version_id': store.version_id,
            'store_format_version': store.format_version}


def handoff_record(out):
    """(arrays, metadata) of a dry handoff: every numerical array (host) and parsed physical metadata."""
    import b108_fixture as fx
    import numpy as np

    from maple_syrup.complete_event import DryHandoff, water_budget

    assert isinstance(out, DryHandoff)
    r = out.progress.result
    arrays = {}
    for name, value in fx.result_arrays(r).items():
        arrays[f'result:{name}'] = value
    for name, value in fx.state_arrays(r.state).items():
        arrays[f'pre_reset:{name}'] = value
    for name, value in fx.state_arrays(out.dry_state).items():
        arrays[f'dry:{name}'] = value
    for name in ('pre_reset_surface_m', 'pre_reset_soil_m', 'terminal_deposition_kg'):
        arrays[f'handoff:{name}'] = fx.H(getattr(out, name))
    arrays['dry:dense_voxels'] = np.frombuffer(fx.dense_bytes(out.dry_state.bed.voxel_column), dtype=np.uint8)
    meta = {'fields': {n: getattr(r, n) for n in fx.RESULT_FIELDS_EXACT}, 'commit_log': list(r.commit_log),
            'rejections': list(r.rejections), 'accounting': out.accounting, 'closure': r.closure(),
            'water_budget': water_budget(out.progress), 'deep_publication': r.deep_publication,
            'logical_identity': fx.logical_identity(out.dry_state.bed.voxel_column),
            'final_store': store_facts(out.dry_state.bed.voxel_column),
            'final_t_s': out.dry_state.t_s, 'n_forced_commits': r.n_forced_commits}
    return arrays, json.loads(json.dumps(meta, default=lambda v: np.asarray(v).tolist(), sort_keys=True))


def stage_reference(backend, run_dir):
    roots = setup_imports()
    import b108_fixture as fx
    import numpy as np

    from maple_syrup.complete_event import complete_event

    xp = namespace(backend)
    t0 = time.perf_counter()
    s, ctx, control, schedule = fx.build(xp, run_dir / 'store_reference', encoding='overlay')
    out = complete_event(s['state0'], ctx, s['column'], s['field'], schedule, s['vegetation'], s['sediment'],
                         max_end_s=MAX_END_S, control=control, policy=policy(), report_every_s=REPORT_S)
    arrays, meta = handoff_record(out)
    np.savez(run_dir / 'reference.npz', **arrays)
    meta.update(roots=roots, stage_wall_s=time.perf_counter() - t0, store_dir=str(run_dir / 'store_reference'),
                configuration=configuration(ctx, control))
    (run_dir / 'reference.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'stage': 'reference', 'final_store': meta['final_store']}))


def stage_pause(backend, run_dir):
    roots = setup_imports()
    import b108_fixture as fx
    import numpy as np
    from maple.core.types.topographic_commit import committed_elevation_m

    from maple_syrup.checkpoint import save_checkpoint
    from maple_syrup.complete_event import EventProgress, complete_event

    xp = namespace(backend)
    t0 = time.perf_counter()
    store = run_dir / 'store_resumed'
    s, ctx, control, schedule = fx.build(xp, store, encoding='overlay')
    out = complete_event(s['state0'], ctx, s['column'], s['field'], schedule, s['vegetation'], s['sediment'],
                         max_end_s=MAX_END_S, control=control, policy=policy(), report_every_s=REPORT_S,
                         checkpoint_callback=lambda p: p.result.state.t_s == PAUSE_S)
    assert isinstance(out, EventProgress) and out.result.state.t_s == PAUSE_S
    r, state = out.result, out.result.state
    source, committed = fx.H(state.routing_source_elevation_m), fx.H(committed_elevation_m(state.bed.committed_topography))
    facts = {**store_facts(state.bed.voxel_column), 'store_dir': str(store.resolve()),
             'mobile_kg': float(fx.H(state.bed.water.mobile_mass_by_cell_class_kg).sum()),
             'max_depth_m': float(fx.H(state.storm.depth_m).max()),
             'held_source_lag_max_m': float(np.abs(source - committed).max()),
             'routing_source_origin': r.routing_source_origin, 'n_commits': r.n_commits,
             'n_accepted_steps': r.n_accepted_steps,
             'cumulative_pickup_kg': float(np.sum(fx.H(r.cumulative_pickup_kg))),
             'cumulative_deposition_kg': float(np.sum(fx.H(r.cumulative_deposition_kg)))}
    # The actual wet / mobile pause with a published v2 version (zero held-source lag is allowed at 1 s).
    assert facts['mobile_kg'] > 0 and facts['max_depth_m'] > 0 and facts['store_format_version'] == 2
    assert facts['version_id'] >= 1 and facts['generation'] >= 1
    save_checkpoint(run_dir / 'bundle', out, ctx, s['column'], s['sediment'], IDENTITY)
    np.savez(run_dir / 'pause.npz', **{f'state:{k}': a for k, a in fx.state_arrays(state).items()})
    meta = {'roots': roots, 'facts': facts, 'store_files': files(store), 'bundle_files': files(run_dir / 'bundle'),
            'configuration': configuration(ctx, control), 'stage_wall_s': time.perf_counter() - t0}
    (run_dir / 'pause.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'stage': 'pause', **facts}))


def stage_resume(backend, run_dir):
    roots = setup_imports()
    import b108_fixture as fx
    import numpy as np
    from maple.surface.voxels.deep_store import load_version

    from maple_syrup.checkpoint import load_checkpoint
    from maple_syrup.complete_event import complete_event

    xp = namespace(backend)
    t0 = time.perf_counter()
    pause = json.loads((run_dir / 'pause.json').read_text())
    store = run_dir / 'store_resumed'
    assert files(store) == pause['store_files'] and files(run_dir / 'bundle') == pause['bundle_files']
    # Context / parameters / forcing are reconstructed from the declared fixture in THIS process; its state0 and its
    # throwaway store are never used for the continuation (every state array comes from the bundle + the saved store).
    s, ctx, control, schedule = fx.build(xp, run_dir / 'store_construction_only', encoding='overlay')
    assert configuration(ctx, control) == pause['configuration']
    loaded = load_checkpoint(run_dir / 'bundle', ctx, s['column'], s['sediment'], IDENTITY)
    state = loaded.result.state
    assert state is not s['state0'] and state.t_s == PAUSE_S
    with np.load(run_dir / 'pause.npz') as z:
        paused = {k[len('state:'):]: z[k] for k in z.files}
    binding = check_paused_binding(paused, fx.state_arrays(state), pause['facts'],
                                   {**store_facts(state.bed.voxel_column),
                                    'store_dir': str(Path(state.bed.voxel_column.deep_store.store_dir).resolve())})
    assert not binding, binding
    out = complete_event(state, ctx, s['column'], s['field'], schedule, s['vegetation'], s['sediment'],
                         max_end_s=MAX_END_S, control=control, policy=policy(), report_every_s=REPORT_S,
                         continuation=loaded)
    arrays, meta = handoff_record(out)
    final = out.dry_state.bed.voxel_column
    after = files(store)
    # Verified lineage from the final version back to the paused manifest (each load checks one parent level).
    lineage, digest, reload_ok = [], final.deep_store.manifest_digest, False
    for _ in range(64):
        v = load_version(store, digest, geometry=ctx.geometry, mass_resolution_kg=ctx.mass_resolution_kg,
                         check_parent_lineage=True)
        reload_ok = reload_ok or (v.format_version == 2 and digest == final.deep_store.manifest_digest)
        if digest == pause['facts']['physical_manifest_digest'] or v.parent_manifest_digest is None:
            break
        digest = v.parent_manifest_digest
        lineage.append(digest)
    fv = final.deep_store
    referenced = set()
    for i in range(fv.n_chunks):
        referenced.add(fv.base_path(i).relative_to(store).as_posix())
        if fv.overlay_path(i) is not None:
            referenced.add(fv.overlay_path(i).relative_to(store).as_posix())
    r = out.progress.result
    post = {'final_generation': final.generation, 'final_version_id': fv.version_id,
            'final_store_format_version': fv.format_version, 'final_reload_ok': reload_ok, 'lineage': lineage,
            'new_store_files': sorted(set(after) - set(pause['store_files'])),
            'referenced_data_files': sorted(referenced), 'store_files_after': after,
            'changed_or_missing_paused_files': sorted(k for k, v in pause['store_files'].items() if after.get(k) != v),
            'after': {'n_accepted_steps': r.n_accepted_steps,
                      'cumulative_pickup_kg': float(np.sum(fx.H(r.cumulative_pickup_kg))),
                      'cumulative_deposition_kg': float(np.sum(fx.H(r.cumulative_deposition_kg)))},
            'bundle_unchanged': files(run_dir / 'bundle') == pause['bundle_files'], 'binding_checks': binding}
    problems = check_post_restart_publication(pause, post)
    assert not problems and post['bundle_unchanged'], problems
    np.savez(run_dir / 'resume.npz', **arrays)
    meta.update(roots=roots, stage_wall_s=time.perf_counter() - t0, store_dir=str(store),
                configuration=configuration(ctx, control), post_restart=post)
    (run_dir / 'resume.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'stage': 'resume', 'final_generation': post['final_generation'],
                      'new_store_files': post['new_store_files'], 'lineage': post['lineage']}))


def _load(run_dir):
    import numpy as np

    def arrays(name):
        with np.load(run_dir / f'{name}.npz') as z:
            return {k: z[k] for k in z.files}

    meta = {n: json.loads((run_dir / f'{n}.json').read_text()) for n in ('reference', 'pause', 'resume')}
    paused = {k[len('state:'):]: v for k, v in arrays('pause').items()}  # the resume stage's key convention
    return arrays('reference'), arrays('resume'), paused, meta


def stage_compare(run_dir, out_path):
    ref, res, _, meta = _load(run_dir)
    array_diffs = compare_arrays(ref, res)
    meta_diffs = compare_metadata(meta['reference'], meta['resume'])
    config_diffs = compare_values(meta['pause']['configuration'], meta['reference']['configuration'])
    publication = check_post_restart_publication(meta['pause'], meta['resume']['post_restart'])
    roots = [meta[n]['roots'] for n in ('reference', 'pause', 'resume')]
    record = {'run_dir': str(run_dir), 'n_arrays': len(ref), 'array_differences': array_diffs,
              'metadata_differences': meta_diffs, 'configuration_differences': config_diffs,
              'post_restart_problems': publication, 'qualified_both': {k: [meta['reference'][k], meta['resume'][k]]
                                                                       for k in QUALIFIED_BOTH},
              'resume_only': {k: meta['resume'][k] for k in RESUME_ONLY}, 'pause_facts': meta['pause']['facts'],
              'stage_roots': roots, 'distinct_processes': len({r['pid'] for r in roots}) == 3,
              'same_digests_all_stages': len({(r['maple_package_digest'], r['syrup_package_digest'],
                                                r['fixture_sha256'], r['helper_sha256']) for r in roots}) == 1,
              'reference_closed': meta['reference']['closure']['closed'] and meta['reference']['water_budget']['closed']}
    record['exact'] = not (array_diffs or meta_diffs or config_diffs or publication)
    Path(out_path).write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: record[k] for k in ('n_arrays', 'array_differences', 'metadata_differences',
                                             'configuration_differences', 'post_restart_problems', 'exact',
                                             'distinct_processes', 'same_digests_all_stages', 'reference_closed')}))
    raise SystemExit(0 if record['exact'] and record['distinct_processes'] and record['same_digests_all_stages']
                     and record['reference_closed'] else 1)


def controls(ref, res, paused, meta):
    """Negative controls on COPIES of actual outputs; returns {control: detected}."""
    import numpy as np

    out = {}
    name = next(k for k in sorted(res) if res[k].dtype == np.float64 and res[k].size and np.isfinite(res[k]).all()
                and res[k].reshape(-1)[0] != 0)
    bad = dict(res)
    arr = res[name].copy()
    arr.reshape(-1)[0] = np.nextafter(arr.reshape(-1)[0], np.inf)
    bad[name] = arr
    out['one_ulp_array_perturbation'] = bool(compare_arrays(ref, bad))
    bad = dict(res)
    bad.pop(name)
    out['missing_array'] = bool(compare_arrays(ref, bad))
    m = copy.deepcopy(meta['resume'])
    del m['fields']['n_reroutes']
    out['missing_physical_metadata_key'] = bool(compare_metadata(meta['reference'], m))
    m = copy.deepcopy(meta['resume'])
    assert m['deep_publication'] is None
    del m['deep_publication']  # a missing key must not masquerade as the reference's None
    out['missing_key_vs_none'] = bool(compare_metadata(meta['reference'], m))
    m = copy.deepcopy(meta['resume'])
    m['final_store']['store_format_version'] = float(m['final_store']['store_format_version'])
    out['int_vs_float_type'] = bool(compare_metadata(meta['reference'], m))
    m = copy.deepcopy(meta['resume'])
    m['unexpected_new_field'] = 0
    out['unknown_extra_field'] = bool(compare_metadata(meta['reference'], m))
    m = copy.deepcopy(meta['resume'])
    m['closure']['closed'] = not m['closure']['closed']
    out['flipped_closure_flag'] = bool(compare_metadata(meta['reference'], m))
    held = dict(paused)
    src = held['routing_source_elevation_m'].copy()
    src.reshape(-1)[0] = np.nextafter(src.reshape(-1)[0], np.inf)
    held['routing_source_elevation_m'] = src
    facts = meta['pause']['facts']
    out['wrong_held_source'] = bool(check_paused_binding(paused, held, facts, facts))
    out['wrong_binding_generation'] = bool(check_paused_binding(paused, paused, facts,
                                                                {**facts, 'generation': facts['generation'] + 1}))
    out['wrong_binding_manifest'] = bool(check_paused_binding(paused, paused, facts,
                                                              {**facts, 'physical_manifest_digest': '0' * 64}))
    post = meta['resume']['post_restart']
    for label, edit in (
            ('no_newer_generation', lambda p: p.__setitem__('final_generation', facts['generation'])),
            ('no_newer_version', lambda p: p.__setitem__('final_version_id', facts['version_id'])),
            ('no_new_manifest_file', lambda p: p.__setitem__('new_store_files', [n for n in p['new_store_files']
                                                                                  if not n.startswith('versions/')])),
            ('no_new_referenced_data_file', lambda p: p.__setitem__('referenced_data_files', [])),
            ('lineage_not_reaching_pause', lambda p: p.__setitem__('lineage', p['lineage'][:-1] + ['0' * 64])),
            ('paused_file_changed', lambda p: p.__setitem__('changed_or_missing_paused_files', ['versions/x'])),
            ('final_not_v2', lambda p: p.__setitem__('final_store_format_version', 1)),
            ('no_transport_after_resume', lambda p: p['after'].__setitem__('cumulative_pickup_kg',
                                                                           facts['cumulative_pickup_kg']))):
        p = copy.deepcopy(post)
        edit(p)
        out[label] = bool(check_post_restart_publication(meta['pause'], p))
    out['unperturbed_actual_outputs_pass'] = not (compare_arrays(ref, res) or compare_metadata(meta['reference'],
                                                                                               meta['resume'])
                                                  or check_post_restart_publication(meta['pause'], post))
    return out


def stage_controls(run_dir, out_path):
    ref, res, paused, meta = _load(run_dir)
    result = controls(ref, res, paused, meta)
    Path(out_path).write_text(json.dumps({'run_dir': str(run_dir), 'controls': result}, indent=1, sort_keys=True) + '\n')
    print(json.dumps(result))
    raise SystemExit(0 if all(result.values()) else 1)


if __name__ == '__main__':
    stage, *args = sys.argv[1:]
    if stage in ('compare', 'controls'):
        {'compare': stage_compare, 'controls': stage_controls}[stage](Path(args[0]), args[1])
    else:
        backend, run_dir = args[0], Path(args[1])
        run_dir.mkdir(parents=True, exist_ok=True)
        {'reference': stage_reference, 'pause': stage_pause, 'resume': stage_resume}[stage](backend, run_dir)
