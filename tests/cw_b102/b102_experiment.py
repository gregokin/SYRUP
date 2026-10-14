"""CW-B102 FIRST step: the ONE bounded fixture experiment of the accepted B101C1 design (harness-level; run BEFORE the
candidate policy exists, on the given SYRUP tree with the frozen M94 MAPLE).

Real MAPLE resident-window bed (the B100 valley fixture, deep store in tmp) whose 32 buried voxels are SEEDED per cell:
checkerboard cells hold totals exactly ONE ULP below voxel capacity (the root sample's 125 - 1 ULP pattern at this
fixture's capacity), the other cells exactly at capacity (controls). Actual rainfall / routing / pickup / deposition and
MAPLE staged commits (every 2 s), reports every 6 s, 12 s.

Modes:
* `step`     -- the tree's own per-step publication (unchanged code);
* `boundary` -- HARNESS-ONLY deferral: the tree's `sediment_event.publish_bed_storage` is wrapped to return the
                staged bed unchanged except for the accepted step that ends exactly on a report boundary (and the end),
                so MAPLE's pending journal is CARRIED through physical steps and staged commits. This exercises the
                actual M94 pending-journal composition, not the B102 implementation.

Records (instrumented actual M94 functions, values only): per writer (spill / miss) appended records and buried changed
voxels; `deep_room` of seeded / control cells per accepted step (first-deposit behaviour); carried pending records per
step; every `apply_journal` (records, touched chunks) and the deep-store counters; `_apply_pending_records` and
`working_bed_journal` call counts and time; per-step raw digests of every physical array; dense logical columns at each
report boundary (published); deep files.

Usage: python -B b102_experiment.py --tree <tree> --backend numpy|cupy --mode step|boundary --tmp <dir> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPORT_S, END_S = 6.0, 12.0


def load_b100_driver(tree):
    spec = importlib.util.spec_from_file_location('_b100_driver', Path(tree) / 'tests/cw_b100/b100_driver.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_columns(dense_mass, capacity, seeded_mask):
    """In-place: every buried voxel (the first 32) of a seeded cell sums to exactly nextafter(capacity, 0), of a control
    cell to exactly capacity, by one-ULP steps on the largest class (MAPLE's own `sum(axis=-1)` reduction)."""
    import numpy as np
    target_low = np.nextafter(capacity, 0.0)
    ny, nx = seeded_mask.shape
    for i in range(ny):
        for j in range(nx):
            for v in range(32):
                vec = dense_mass[i, j, v]
                target = target_low if seeded_mask[i, j] else capacity
                k = int(np.argmax(vec))
                for _ in range(64):
                    s = vec.sum()
                    if s == target:
                        break
                    vec[k] = np.nextafter(vec[k], -np.inf if s > target else np.inf)
                assert vec.sum() == target
    totals = dense_mass[:, :, :32].sum(axis=-1)
    assert np.all(totals[seeded_mask] == target_low) and np.all(totals[~seeded_mask] == capacity)


def build(drv, xp, tmp):
    """`b100_driver.window_valley_inputs` with the seeding inserted before the windowed column is built."""
    import numpy as np
    from maple.core.backend import to_device_tree, to_host_tree
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
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
    from test_coupled_backends import setup_xp
    from test_exchange_history import constant_rainfall
    from test_sediment_event import make_bed, setup, valley_elevation

    from maple_syrup.sediment_bed import refresh_routing
    from maple_syrup.sediment_event import SedimentEventControl, initial_event_state
    from maple_syrup.storm import StormControl

    ny, nx = 64, 32
    implementation = 'numba' if xp is np else 'cuda'
    bed, ctx = make_bed(valley_elevation(ny, nx, sy=0.001, sx=0.002), depth_m=.001,
                        commit_spec=TopographicCommitSpec(commit_interval_s=2.0))
    control = SedimentEventControl(storm=StormControl(max_dt_s=1, implementation=implementation, max_steps=10000),
                                   record_exchange_history=False, force_final_commit=False, phase_storage='csr')
    s = setup(bed, ctx, ksat=1e-5, control=control) if xp is np else setup_xp(bed, ctx, xp, ksat=1e-5,
                                                                                    control=control)
    ctx = s['ctx']
    schedule = constant_rainfall(0, 10, 100)
    state = s['state0']
    host = to_host_tree(state.bed)
    deep = np.repeat(host.voxel_column.mass_kg[:, :, :1, :], 32, axis=2)
    dense = np.concatenate((deep, host.voxel_column.mass_kg), axis=2).copy()
    capacity = max_voxel_mass_kg(ctx.geometry)
    seeded = (np.add.outer(np.arange(ny), np.arange(nx)) % 2) == 0
    seed_columns(dense, capacity, seeded)
    dense_column = VoxelColumnState(dense)
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
                                        store_dir=Path(tmp) / 'store', max_store_bytes=10**9,
                                        class_ids=tuple(g.class_id for g in ctx.grain_classes.classes),
                                        window_voxels=min(4, dense.shape[2]), chunk_columns=64,
                                        with_prefix_packets=True)
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
    return s, ctx, control, schedule, seeded, capacity


def physical_digest(drv, state):
    """Raw digests of every PHYSICAL array of an accepted state (resident bed, ledger, water, availability, committed
    topography, storm, velocity, phase); the deep version handle / generation / pending binding are excluded."""
    b = state.bed
    col = b.voxel_column
    parts = {'water': b.water, 'active_layer': b.active_layer, 'ledger': b.ledger,
             'availability': b.sediment_availability, 'committed': b.committed_topography, 'storm': state.storm,
             'velocity': state.sediment_velocity_m_s, 'phase': state.phase,
             'window_kg': col.window_kg, 'base': col.base, 'deep_by_class_kg': col.deep_by_class_kg,
             'deep_total_kg': col.deep_total_kg, 'deep_room_kg': col.deep_room_kg}
    if col.height_packet is not None:
        parts['packet'] = col.height_packet.packet
    out = {}
    for k, v in parts.items():
        out.update(drv.raw(v, k))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--tree', required=True)
    p.add_argument('--backend', choices=('numpy', 'cupy'), required=True)
    p.add_argument('--mode', choices=('step', 'boundary'), required=True)
    p.add_argument('--tmp', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    drv = load_b100_driver(a.tree)
    drv.bind(a.tree)
    import numpy as np
    from maple.core.backend import resolve_backend, to_host
    xp = resolve_backend(a.backend).xp
    import maple.surface.voxels.deep_store as ds
    import maple.surface.voxels.windowed_transfer as wt
    import maple.water.windowed_step as wstep
    from maple.surface.voxels.windowed_inventory import windowed_to_dense

    import maple_syrup.sediment_event as se

    t_setup = time.perf_counter()
    s, ctx, control, schedule, seeded, capacity = build(drv, xp, a.tmp)
    setup_s = time.perf_counter() - t_setup
    store = Path(a.tmp) / 'store'
    files_before = drv.store_files(store)
    rec = {'tree': str(Path(a.tree).resolve()), 'backend': a.backend, 'mode': a.mode, 'capacity_kg': capacity,
           'seeded_cells': int(seeded.sum()), 'control_cells': int((~seeded).sum()), 'report_s': REPORT_S,
           'end_s': END_S, 'setup_s': setup_s}
    counts = {'spill_calls': 0, 'spill_records': 0, 'miss_calls': 0, 'miss_records': 0, 'miss_cells': 0,
              'miss_buried_changed_voxels': 0, 'miss_buried_changed_seeded': 0, 'miss_buried_changed_control': 0,
              'overlay_calls': 0, 'overlay_s': 0.0, 'overlay_records_applied': 0, 'canonical_calls': 0,
              'canonical_s': 0.0, 'canonical_records_out': 0, 'apply_journal_calls': 0, 'apply_journal_records': [],
              'apply_journal_touched_chunks': [], 'publish_wrapper_calls': 0, 'publish_wrapper_published': 0}
    nx = seeded.shape[1]
    orig = {'spill': wt._spill_before_deposition, 'miss': wt._publish_miss_columns, 'overlay': wt._apply_pending_records,
            'canon_wt': wt.working_bed_journal, 'canon_ws': wstep.working_bed_journal, 'apply': ds.DeepStoreVersion.apply_journal,
            'publish': se.publish_bed_storage, 'step': se._sediment_accounting_step}

    def spill(bed, request_total):
        before = bed.pending_records()
        n = orig['spill'](bed, request_total)
        counts['spill_calls'] += 1
        counts['spill_records'] += bed.pending_records() - before
        return n

    def miss(bed, rows, cols, cells_flat, before, after, old_base):
        n0 = bed.pending_records()
        orig['miss'](bed, rows, cols, cells_flat, before, after, old_base)
        counts['miss_calls'] += 1
        counts['miss_records'] += bed.pending_records() - n0
        counts['miss_cells'] += int(len(cells_flat))
        v = np.arange(after.shape[1])[None, :]
        buried = (after != before).any(axis=-1) & (v < np.asarray(old_base)[:, None])
        per_cell = buried.sum(axis=1)
        seeded_flat = seeded.reshape(-1)[np.asarray(cells_flat)]
        counts['miss_buried_changed_voxels'] += int(per_cell.sum())
        counts['miss_buried_changed_seeded'] += int(per_cell[seeded_flat].sum())
        counts['miss_buried_changed_control'] += int(per_cell[~seeded_flat].sum())

    def overlay(bed, cells_flat, columns):
        t0 = time.perf_counter()
        n = orig['overlay'](bed, cells_flat, columns)
        counts['overlay_s'] += time.perf_counter() - t0
        counts['overlay_calls'] += 1
        counts['overlay_records_applied'] += n
        return n

    def canonical(bed):
        t0 = time.perf_counter()
        j = orig['canon_wt'](bed)
        counts['canonical_s'] += time.perf_counter() - t0
        counts['canonical_calls'] += 1
        counts['canonical_records_out'] = max(counts['canonical_records_out'], j.n_records)
        return j

    def apply(self, journal, geometry, mass_resolution_kg, **kw):
        out = orig['apply'](self, journal, geometry, mass_resolution_kg, **kw)
        if journal.n_records:
            counts['apply_journal_calls'] += 1
            counts['apply_journal_records'].append(int(journal.n_records))
            counts['apply_journal_touched_chunks'].append(int(np.unique(journal.cell_flat // self.header.chunk_columns).size))
        return out

    current_t = [0.0]

    def step(*args, **kwargs):
        out = orig['step'](*args, **kwargs)
        current_t[0] = float(out.storm.state.t_s)
        return out

    def publish(bed, context):
        counts['publish_wrapper_calls'] += 1
        t = current_t[0]
        at_boundary = abs(t / REPORT_S - round(t / REPORT_S)) < 1e-12 or t >= END_S
        if a.mode == 'step' or at_boundary:
            counts['publish_wrapper_published'] += 1
            return orig['publish'](bed, context)
        return bed  # HARNESS deferral: the staged bed (actual MAPLE pending journal) is carried

    per_step, boundaries = [], []

    def accepted(state):
        col = state.bed.voxel_column
        room = np.asarray(to_host(col.deep_room_kg))
        per_step.append({'t': float(state.t_s), 'digest': physical_digest(drv, state),
                         'carried_records': 0 if col.pending_deep is None else int(col.pending_deep.journal.n_records),
                         'generation': int(col.generation),
                         'deep_room_positive_seeded': int(np.count_nonzero(room[seeded] > 0.0)),
                         'deep_room_positive_control': int(np.count_nonzero(room[~seeded] > 0.0)),
                         'commit_count': int(state.bed.committed_topography.commit_count)})
        return False

    def boundary(result):
        st = result.state
        col = st.bed.voxel_column
        dense = windowed_to_dense(col, max_bytes=10**8).mass_kg
        boundaries.append({'t': float(st.t_s), 'published': col.pending_deep is None, 'generation': int(col.generation),
                           'manifest_digest': col.manifest_digest,
                           'dense_sha256': hashlib.sha256(np.ascontiguousarray(dense).tobytes()).hexdigest(),
                           'dense_seeded_buried_sum': float(dense[:, :, :32][seeded].sum()),
                           'buried_room_positive_seeded_voxels': int(np.count_nonzero(
                               (capacity - dense[:, :, :32].sum(axis=-1))[seeded] > 0.0))})
        return False

    wt._spill_before_deposition = spill
    wt._publish_miss_columns = miss
    wt._apply_pending_records = overlay
    wt.working_bed_journal = canonical
    wstep.working_bed_journal = canonical
    ds.DeepStoreVersion.apply_journal = apply
    se.publish_bed_storage = publish
    se._sediment_accounting_step = step
    ds.reset_deep_store_counters()
    t0 = time.perf_counter()
    try:
        result = se.evolve_sediment_event(s['state0'], ctx, s['column'], s['field'], schedule, s['vegetation'],
                                          s['sediment'], END_S, control, report_every_s=REPORT_S,
                                          on_accepted_step=accepted, on_boundary=boundary)
        rec['status'] = 'ok'
        rec['n_accepted_steps'], rec['n_commits'] = result.n_accepted_steps, result.n_commits
        rec['final_published'] = result.state.bed.voxel_column.pending_deep is None
        raw = drv.raw(result, 'result')
        rec['result_raw_nondeep'] = {k: v for k, v in raw.items() if 'deep_store' not in k and 'generation' not in k
                                     and 'manifest_digest' not in k and 'timing' not in k}
        del result
    finally:
        wt._spill_before_deposition, wt._publish_miss_columns = orig['spill'], orig['miss']
        wt._apply_pending_records, wt.working_bed_journal = orig['overlay'], orig['canon_wt']
        wstep.working_bed_journal = orig['canon_ws']
        ds.DeepStoreVersion.apply_journal = orig['apply']
        se.publish_bed_storage, se._sediment_accounting_step = orig['publish'], orig['step']
    rec['event_wall_s'] = time.perf_counter() - t0
    rec['deep_store_counters'] = ds.read_deep_store_counters()
    rec['counts'] = counts
    rec['per_step'] = per_step
    rec['boundaries'] = boundaries
    after = drv.store_files(store)
    rec['store_preexisting_preserved'] = all(after.get(k) == v for k, v in files_before.items())
    rec['store_files_after'] = len(after)
    rec['store_new_chunk_files'] = sum(k.startswith('chunks/') for k in after) - sum(k.startswith('chunks/') for k in files_before)
    rec['store_new_versions'] = sum(k.startswith('versions/') for k in after) - sum(k.startswith('versions/') for k in files_before)
    rec['store_new_bytes'] = sum((store / k).stat().st_size for k in after if k not in files_before)
    with open(a.out, 'x') as f:
        json.dump(rec, f, indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
