"""CW-B100 subprocess driver: ONE small coupled evolving-water event on an actual MAPLE resident-window bed (deep store
in a tmp directory) with interval topographic commits, executed by the SYRUP tree given on the command line (the
accepted R1 tree or the B100 candidate) against the same frozen M94 MAPLE.

It records, without retaining any state object:
* at every MAPLE commit ledger allocation (`windowed_commit.zeros_sediment_ledger_state`, the R6 OOM site) whether the
  PREVIOUS accepted state's arrays are still alive (weak references only) and the allocator's in-use bytes (CuPy pool
  `used_bytes` / tracemalloc current), plus the in-use peak inside the commit;
* the overall in-use peak of the event (CuPy memory hook / tracemalloc) and where it occurred;
* a complete raw digest of the returned result (arrays: dtype / shape / bytes), the caller's input arrays before and
  after, and the deep-store files.
Optional: a CuPy pool hard limit (`--limit`), an injected commit or reroute failure (`--fail`).

Usage: python -B b100_driver.py --tree <tree root> --backend numpy|cupy --tmp <dir> --out <json>
       [--ny 64 --nx 32 --end 12] [--limit BYTES] [--fail commit:N|reroute:N]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import struct
import sys
import time
import traceback
import tracemalloc
import weakref
from dataclasses import replace
from pathlib import Path


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--tree', required=True)
    p.add_argument('--backend', choices=('numpy', 'cupy'), required=True)
    p.add_argument('--tmp', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ny', type=int, default=64)
    p.add_argument('--nx', type=int, default=32)
    p.add_argument('--end', type=float, default=12.0)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--fail', default=None)
    return p.parse_args(argv)


def bind(tree):
    tree = Path(tree).resolve()
    sys.path.insert(0, str(tree / 'src'))
    for i, sub in enumerate(('phase5', 'conservative_water', 'cw_b20'), start=1):
        sys.path.insert(i, str(tree / 'tests' / sub))
    import maple_syrup
    got = Path(maple_syrup.__file__).resolve().parent
    if got != tree / 'src' / 'maple_syrup':
        raise SystemExit(f'maple_syrup imported from {got}, not {tree}')
    return tree


def window_valley_inputs(xp, tmp_path, ny, nx):
    """`test_windowed_storm.window_inputs` (CSR, W4 deep resident window, interval commits every 2 s) on a valley of
    ny x nx cells instead of the 3-cell chain, exchange history OFF (as R1)."""
    from maple.core.backend import to_device_tree, to_host_tree
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.sediment_ledger import process_index, zeros_sediment_ledger_state
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.core.types.voxel import VoxelColumnState
    from maple.surface.topographic_commit.initialization import (
        initial_committed_topography_state_from_physical_state,
    )
    from maple.surface.topographic_commit.windowed_commit import windowed_committed_state_from_dense_baseline
    from maple.surface.voxels.windowed_inventory import windowed_column_from_dense
    from maple.surface.voxels.windowed_prefix import move_prefix_packets
    from test_coupled_backends import setup_xp
    from test_exchange_history import constant_rainfall
    from test_sediment_event import make_bed, setup, valley_elevation

    from maple_syrup.sediment_bed import refresh_routing
    from maple_syrup.sediment_event import SedimentEventControl, initial_event_state
    from maple_syrup.storm import StormControl

    implementation = 'numba' if xp.__name__ == 'numpy' else 'cuda'
    bed, ctx = make_bed(valley_elevation(ny, nx, sy=0.001, sx=0.002), depth_m=.001,
                        commit_spec=TopographicCommitSpec(commit_interval_s=2.0))
    control = SedimentEventControl(storm=StormControl(max_dt_s=1, implementation=implementation, max_steps=10000),
                                   record_exchange_history=False, force_final_commit=False, phase_storage='csr')
    import numpy as np
    s = setup(bed, ctx, ksat=1e-5, control=control) if xp is np else setup_xp(bed, ctx, xp, ksat=1e-5,
                                                                                    control=control)
    ctx = s['ctx']  # the device context on CuPy (setup_xp moves bed and context), the same object on NumPy
    schedule = constant_rainfall(0, 10, 100)
    state = s['state0']
    host = to_host_tree(state.bed)
    deep = np.repeat(host.voxel_column.mass_kg[:, :, :1, :], 32, axis=2)
    dense_column = VoxelColumnState(np.concatenate((deep, host.voxel_column.mass_kg), axis=2))
    baseline = initial_committed_topography_state_from_physical_state(dense_column, host.active_layer, ctx.geometry,
                                                                     ctx.mass_resolution_kg)
    host = replace(host, voxel_column=dense_column, committed_topography=baseline)
    old_ref = state.terrain
    shift = committed_elevation_m(baseline) - old_ref.initial_committed_elevation_m
    full = old_ref.initial_full_elevation_m.copy() + float(shift[0, 0])
    full[1:-1, 1:-1] = old_ref.initial_full_elevation_m[1:-1, 1:-1] + shift
    ref = replace(old_ref, initial_full_elevation_m=full,
                  initial_committed_elevation_m=committed_elevation_m(baseline).copy())
    column = windowed_column_from_dense(host.voxel_column, ctx.geometry, ctx.mass_resolution_kg,
                                        store_dir=Path(tmp_path) / 'store', max_store_bytes=10**9,
                                        class_ids=tuple(g.class_id for g in ctx.grain_classes.classes),
                                        window_voxels=min(4, host.voxel_column.mass_kg.shape[2]),
                                        chunk_columns=64, with_prefix_packets=True)
    committed, _ = windowed_committed_state_from_dense_baseline(host.committed_topography, column, host.active_layer,
                                                                ctx.geometry)
    ledger = zeros_sediment_ledger_state(ctx.geometry.ny, ctx.geometry.nx, len(ctx.grain_classes.classes),
                                         present_processes=(process_index('water_erosion_deposition'),))
    bed = replace(host, voxel_column=column, committed_topography=committed, ledger=ledger)
    if xp is not np:
        moved = to_device_tree(bed, xp)
        bed = replace(moved, voxel_column=move_prefix_packets(moved.voxel_column, column, xp))
    graph = refresh_routing(ref, bed, xp=xp)
    s['state0'] = initial_event_state(graph, ref, bed, ctx, state.storm.soil_water_m, control=control)
    return s, ctx, control, schedule


# --- raw digests ------------------------------------------------------------------------------------------------------
def host(a):
    from maple.core.backend import to_host
    import numpy as np
    return np.ascontiguousarray(np.asarray(to_host(a)))


def is_array(v):
    return hasattr(v, 'dtype') and hasattr(v, 'shape') and hasattr(v, 'nbytes') and not isinstance(v, type)


def raw(v, path='', out=None, seen=None):
    """Flat {path: digest} of every array / scalar reachable through dataclasses, dicts, lists and tuples."""
    import numpy as np
    out = {} if out is None else out
    seen = set() if seen is None else seen
    if is_array(v):
        h = host(v)
        out[path] = f'{h.dtype.str}{h.shape}:{hashlib.sha256(h.tobytes()).hexdigest()}'
    elif isinstance(v, (bool, int, str)) or v is None:
        out[path] = repr(v)
    elif isinstance(v, float):
        out[path] = struct.pack('<d', v).hex()
    elif isinstance(v, np.generic):
        out[path] = f'{v.dtype.str}:{v.tobytes().hex()}'
    elif dataclasses.is_dataclass(v) and not isinstance(v, type):
        if id(v) in seen:
            out[path] = '<seen>'
            return out
        seen.add(id(v))
        for f in dataclasses.fields(v):
            raw(getattr(v, f.name), f'{path}.{f.name}', out, seen)
    elif isinstance(v, dict):
        for k in sorted(v, key=str):
            raw(v[k], f'{path}[{k!r}]', out, seen)
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            raw(x, f'{path}[{i}]', out, seen)
    elif hasattr(v, 'manifest_digest'):
        out[path] = f'deep:{v.manifest_digest}:{getattr(v, "version_id", None)}'
    elif type(v).__name__ == 'module':
        out[path] = f'module:{v.__name__}'
    else:
        out[path] = f'<{type(v).__module__}.{type(v).__qualname__}>'
    return out


def store_files(store):
    out = {}
    for p in sorted(Path(store).rglob('*')):
        if p.is_file():
            out[p.relative_to(store).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def main(argv=None):
    a = parse(argv)
    tree = bind(a.tree)
    from maple.core.backend import resolve_backend
    xp = resolve_backend(a.backend).xp
    import maple.surface.topographic_commit.windowed_commit as wc

    import maple_syrup.sediment_event as se
    rec = {'tree': str(tree), 'backend': a.backend, 'ny': a.ny, 'nx': a.nx, 'end_s': a.end, 'limit': a.limit,
           'fail': a.fail}
    pool = None
    if a.backend == 'cupy':
        pool = xp.get_default_memory_pool()
        dev = xp.cuda.runtime.getDeviceProperties(0)
        rec['device'] = {'name': dev['name'].decode() if isinstance(dev['name'], bytes) else dev['name'],
                         'uuid': 'GPU-' + bytes(dev['uuid']).hex(), 'total_bytes': int(dev['totalGlobalMem'])}
    s, ctx, control, schedule = window_valley_inputs(xp, a.tmp, a.ny, a.nx)
    state0 = s['state0']
    nc = len(ctx.grain_classes.classes)
    rec['c_grid_bytes'] = a.ny * a.nx * nc * 8
    rec['n_classes'] = nc
    inputs_before = raw(state0, 'state0')
    store = Path(a.tmp) / 'store'
    files_before = store_files(store)

    prev = {}  # weak references ONLY (never a state object)
    probes, inside = [], [False]
    peak = {'overall': 0, 'where': None, 'commit': 0, 'pool_total_max': 0}

    def used():
        if pool is not None:
            return int(pool.used_bytes())
        return int(tracemalloc.get_traced_memory()[0])

    def note(where):
        u = used()
        if u > peak['overall']:
            peak['overall'], peak['where'] = u, where
        if inside[0] and u > peak['commit']:
            peak['commit'] = u
        if pool is not None:
            peak['pool_total_max'] = max(peak['pool_total_max'], int(pool.total_bytes()))

    def accepted(state):
        prev.clear()
        b = state.bed
        for name, arr in (('ledger.pending_bed_mass_change_kg', b.ledger.pending_bed_mass_change_kg),
                          ('ledger.n_physical_touches', b.ledger.n_physical_touches),
                          ('active_layer.mass_kg', b.active_layer.mass_kg),
                          ('water.mobile_mass_by_cell_class_kg', b.water.mobile_mass_by_cell_class_kg),
                          ('availability.available_mass_kg', b.sediment_availability.available_mass_kg),
                          ('voxel_column.window_kg', b.voxel_column.window_kg),
                          ('sediment_velocity_m_s', state.sediment_velocity_m_s),
                          ('storm.depth_m', state.storm.depth_m)):
            prev[name] = weakref.ref(arr)
        if state.phase is not None and getattr(state.phase, 'fraction', None) is not None:
            prev['phase.fraction'] = weakref.ref(state.phase.fraction)
        note('accepted')
        return False

    original_zeros = wc.zeros_sediment_ledger_state

    def probed_zeros(*args, **kwargs):
        probes.append({'n_prev_alive': sum(r() is not None for r in prev.values()),
                       'prev_alive': {k: r() is not None for k, r in prev.items()},
                       'used_bytes_before_fresh_ledger': used(),
                       'pool_total_bytes': None if pool is None else int(pool.total_bytes())})
        out = original_zeros(*args, **kwargs)
        note('commit fresh ledger')
        return out

    original_commit_bed = se.commit_bed
    original_grid = se.physics_grid_from_graph
    counts = {'commit': 0, 'reroute': 0}
    fail_kind, fail_n = (a.fail.split(':')[0], int(a.fail.split(':')[1])) if a.fail else (None, None)

    def wrapped_commit_bed(*args, **kwargs):
        inside[0] = True
        try:
            out = original_commit_bed(*args, **kwargs)
            note('commit_bed return')
        finally:
            inside[0] = False
        if out.did_commit:
            counts['commit'] += 1
            if fail_kind == 'commit' and counts['commit'] == fail_n:
                raise RuntimeError('B100 injected commit failure (after the MAPLE commit, before publication)')
        return out

    def wrapped_grid(graph):
        counts['reroute'] += 1
        if fail_kind == 'reroute' and counts['reroute'] == fail_n:
            raise RuntimeError('B100 injected reroute failure (refreshed physics grid)')
        return original_grid(graph)

    hook = None
    if pool is not None:
        from cupy.cuda import memory_hook

        class Peak(memory_hook.MemoryHook):
            name = 'b100_peak'

            def malloc_postprocess(self, **kw):
                note('malloc')
        hook = Peak()
        if a.limit is not None:
            pool.set_limit(size=a.limit)
    else:
        tracemalloc.start()
    wc.zeros_sediment_ledger_state = probed_zeros
    se.commit_bed = wrapped_commit_bed
    se.physics_grid_from_graph = wrapped_grid
    t0 = time.perf_counter()
    try:
        if hook is not None:
            hook.__enter__()
        try:
            result = se.evolve_sediment_event(state0, ctx, s['column'], s['field'], schedule, s['vegetation'],
                                              s['sediment'], a.end, control, report_every_s=2.0,
                                              on_accepted_step=accepted)
        finally:
            if hook is not None:
                hook.__exit__(None, None, None)
        rec['status'] = 'ok'
        rec['n_accepted_steps'], rec['n_commits'] = result.n_accepted_steps, result.n_commits
        rec['result_raw'] = raw(result, 'result')
        del result
    except BaseException as exc:  # noqa: BLE001 -- recorded; the caller decides
        rec['status'] = 'error'
        rec['exception'] = {'type': type(exc).__name__, 'message': str(exc)[:2000],
                            'oom': ('OutOfMemoryError' in type(exc).__name__ or 'OutOfMemory' in str(exc)
                                    or (isinstance(exc, MemoryError) and 'bad_alloc' in str(exc))),
                            'site': ('commit fresh ledger' if 'zeros_sediment_ledger_state' in traceback.format_exc()
                                     else 'commit' if 'commit_bed' in traceback.format_exc() else 'step'),
                            'trace': traceback.format_exc()[-3000:]}
    finally:
        wc.zeros_sediment_ledger_state = original_zeros
        se.commit_bed = original_commit_bed
        se.physics_grid_from_graph = original_grid
    rec['wall_s'] = time.perf_counter() - t0
    rec['counts'] = counts
    rec['probes'] = probes
    rec['peak'] = peak
    if pool is None:
        rec['tracemalloc_peak'] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    rec['inputs_unchanged'] = raw(state0, 'state0') == inputs_before
    after = store_files(store)
    rec['store_preexisting_preserved'] = all(after.get(k) == v for k, v in files_before.items())
    rec['store_files'] = after
    with open(a.out, 'x') as f:
        json.dump(rec, f, indent=1, sort_keys=True, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
