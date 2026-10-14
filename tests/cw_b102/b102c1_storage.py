"""CW-B102C1 storage-contract fixture (SYNTHETIC legal exchange sequence; NOT a physical storm).

Actual MAPLE M94 resident-window storage APIs only (no SYRUP event loop, no fake storage authority, no record
dropping): `begin_windowed_transaction`, `windowed_deposit_inplace`, `windowed_extract_inplace`,
`working_bed_journal`, the M94 `windowed_step` section-9 column finishing (`with_window_cache`,
`attach_prefix_packets`, `attach_pending_journal`), the pending overlay inside `gather_working_columns`,
`publish_pending_column` and `windowed_to_dense`.

Bed: 16 x 16 cells, dx = dy = 1 m, voxel 0.1 m, bulk 1250 kg/m3 (capacity 125 kg), 40 voxels, 2 classes, W = 4,
chunk 64 columns (4 chunks), prefix packets. Buried voxels 0..31: cells A (0..3) and B (4..7, chunk 0; 70..73, chunk
1) hold the root sample's [11.5, 113.49999999999999] kg (125 - 1 ULP); every other cell [11.5, 113.5] (exact). Voxel 32
full [62.5, 62.5], voxel 33 [30, 30], empty above.

Transactions (the same calls in both modes):
  T1 deposit [0.01, 0.01] kg on A          -> miss: one-ULP rooms filled, buried records (chunk 0)
  T2 deposit [0.01, 0.01] kg on B          -> staggered records in the SAME chunk 0 and in chunk 1
  T3 extract 1000 kg total on A            -> miss: full-column READ through the pending overlay
  T4 deposit [400, 400] kg on A            -> miss: rebase writes voxels already pending since T1 (OVERWRITE)
Modes: `step` publishes after every transaction; `deferred` carries the MAPLE pending journal and publishes once at
the end. Records per transaction: appended (application) vs canonical journal records, overlay records applied,
resident-array digests, logical columns of every cell (fresh read transaction, version + pending overlay + window),
then the published final dense columns, class inventory, deep-store counters and new chunk / manifest files / bytes.

Usage: python -B b102c1_storage.py --backend numpy|cupy --mode step|deferred --tmp D --out J
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=('numpy', 'cupy'), required=True)
    p.add_argument('--mode', choices=('step', 'deferred'), required=True)
    p.add_argument('--tmp', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    t_setup = time.perf_counter()
    import numpy as np
    from maple.core.backend import freeze, resolve_backend, to_device_tree, to_host
    from maple.core.boundaries import resolve_axis_boundary
    from maple.core.parameters.geometry import GeometrySpec
    from maple.core.parameters.numerics import DEFAULT_MASS_RESOLUTION_KG
    from maple.core.types.voxel import VoxelColumnState, WindowedVoxelColumnState
    from maple.surface.voxels import deep_store as ds
    from maple.surface.voxels import windowed_transfer as wt
    from maple.surface.voxels.windowed_inventory import windowed_column_from_dense, windowed_to_dense, with_window_cache
    from maple.surface.voxels.windowed_pending import attach_pending_journal, publish_pending_column
    from maple.surface.voxels.windowed_prefix import attach_prefix_packets, move_prefix_packets
    xp = resolve_backend(a.backend).xp
    res = DEFAULT_MASS_RESOLUTION_KG
    ny = nx = 16
    nv, nc, width = 40, 2, 4
    g = GeometrySpec(nx=nx, ny=ny, dx_m=1.0, dy_m=1.0, boundary_x=resolve_axis_boundary('periodic'),
                     boundary_y=resolve_axis_boundary('periodic'))
    dense = np.zeros((ny, nx, nv, nc))
    flat = dense.reshape(ny * nx, nv, nc)
    flat[:, :32] = [11.5, 113.5]
    A, B = [0, 1, 2, 3], [4, 5, 6, 7, 70, 71, 72, 73]
    flat[A + B, :32] = [11.5, 113.49999999999999]
    flat[:, 32] = [62.5, 62.5]
    flat[:, 33] = [30.0, 30.0]
    store = Path(a.tmp) / 'store'
    host_column = windowed_column_from_dense(VoxelColumnState(dense), g, res, store_dir=store, max_store_bytes=10**9,
                                             class_ids=('a', 'b'), window_voxels=width, chunk_columns=64,
                                             with_prefix_packets=True)
    column = host_column
    if xp is not np:
        column = move_prefix_packets(to_device_tree(host_column, xp), host_column, xp)

    def files():
        return {q.relative_to(store).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest()
                for q in sorted(store.rglob('*')) if q.is_file()}
    files0 = files()
    counts = {'overlay_records_applied': 0}
    orig_overlay = wt._apply_pending_records

    def overlay(bed, cells, columns):
        n = orig_overlay(bed, cells, columns)
        counts['overlay_records_applied'] += n
        return n
    wt._apply_pending_records = overlay
    setup_s = time.perf_counter() - t_setup

    def finish(bed, before):
        """M94 windowed_step section 9, verbatim order."""
        journal = wt.working_bed_journal(bed)
        freeze(bed.window_kg)
        freeze(bed.window_totals_kg)
        new = WindowedVoxelColumnState(window_kg=bed.window_kg, base=bed.base, deep_by_class_kg=bed.deep_by_class_kg,
                                       deep_total_kg=bed.deep_total_kg, deep_room_kg=bed.deep_room_kg,
                                       n_voxels=bed.n_voxels, deep_store=before.deep_store,
                                       generation=before.generation, manifest_digest=before.manifest_digest)
        new = with_window_cache(new)
        if bed.packet is not None:
            new = attach_prefix_packets(new, bed.packet)
        return attach_pending_journal(new, journal), journal

    def digest(col):
        out = {}
        for name in ('window_kg', 'base', 'deep_by_class_kg', 'deep_total_kg', 'deep_room_kg'):
            h = np.ascontiguousarray(np.asarray(to_host(getattr(col, name))))
            out[name] = hashlib.sha256(h.tobytes()).hexdigest()
        out['packet'] = hashlib.sha256(np.ascontiguousarray(np.asarray(to_host(col.height_packet.packet))).tobytes()).hexdigest()
        return out

    def logical(col):
        """Every cell's TRUE logical column through a fresh read-only transaction (version + pending + window)."""
        bed = wt.begin_windowed_transaction(col, g, res, max_gather_columns=ny * nx, max_journal_records=10**7)
        cols, _ = wt.gather_working_columns(bed, np.arange(ny * nx, dtype=np.int64))
        return cols

    def req3(cells, vec):
        r = np.zeros((ny, nx, nc))
        r.reshape(-1, nc)[cells] = vec
        return xp.asarray(r)

    def req2(cells, total):
        r = np.zeros((ny, nx))
        r.reshape(-1)[cells] = total
        return xp.asarray(r)

    plan = [('T1 deposit A', 'deposit', req3(A, [0.01, 0.01])), ('T2 deposit B', 'deposit', req3(B, [0.01, 0.01])),
            ('T3 extract A', 'extract', req2(A, 1000.0)), ('T4 deposit A', 'deposit', req3(A, [400.0, 400.0]))]
    ds.reset_deep_store_counters()
    rows = []
    t_work = time.perf_counter()
    for label, kind, request in plan:
        before = column
        bed = wt.begin_windowed_transaction(column, g, res, max_journal_records=10**7)
        carried = bed.pending_records()
        overlay0 = counts['overlay_records_applied']
        if kind == 'deposit':
            wt.windowed_deposit_inplace(bed, request)
        else:
            wt.windowed_extract_inplace(bed, request)
        appended_incl_carried = bed.pending_records()
        column, journal = finish(bed, before)
        del bed
        row = {'label': label, 'carried_records_at_begin': carried,
               'appended_records_this_transaction': appended_incl_carried - carried,
               'canonical_journal_records': int(journal.n_records),
               'overlay_records_applied': counts['overlay_records_applied'] - overlay0,
               'resident': digest(column)}
        cols = logical(column)
        row['logical_sha256'] = hashlib.sha256(np.ascontiguousarray(cols).tobytes()).hexdigest()
        row['logical_class_sums'] = cols.sum(axis=(0, 1)).tolist()
        if a.mode == 'step':
            column = publish_pending_column(column, g, res, max_records=10**7)
        row['generation_after'] = int(column.generation)
        row['pending_after'] = 0 if column.pending_deep is None else int(column.pending_deep.journal.n_records)
        rows.append(row)
    if a.mode == 'deferred':
        column = publish_pending_column(column, g, res, max_records=10**7)
    work_s = time.perf_counter() - t_work
    wt._apply_pending_records = orig_overlay
    final = windowed_to_dense(column, max_bytes=10**8).mass_kg
    after = files()
    new = sorted(set(after) - set(files0))
    rec = {'backend': a.backend, 'mode': a.mode, 'label': 'SYNTHETIC storage-contract sequence (not a physical storm)',
           'setup_s': setup_s, 'work_s': work_s, 'rows': rows,
           'final_generation': int(column.generation), 'final_published': column.pending_deep is None,
           'final_dense_sha256': hashlib.sha256(np.ascontiguousarray(final).tobytes()).hexdigest(),
           'final_class_inventory_kg': np.asarray(final).sum(axis=(0, 1, 2)).tolist(),
           'final_class_inventory_hex': [float(x).hex() for x in np.asarray(final).sum(axis=(0, 1, 2))],
           'deep_store_counters': ds.read_deep_store_counters(),
           'preexisting_files_preserved': all(after.get(k) == v for k, v in files0.items()),
           'new_chunk_files': [k for k in new if k.startswith('chunks/')],
           'new_manifests': [k for k in new if k.startswith('versions/')],
           'new_chunk_bytes': sum((store / k).stat().st_size for k in new if k.startswith('chunks/')),
           'new_manifest_bytes': sum((store / k).stat().st_size for k in new if k.startswith('versions/')),
           'total_records_overall_canonical': None}
    with open(a.out, 'x') as f:
        json.dump(rec, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
