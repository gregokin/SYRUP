"""Actual resident snapshots reject corrupt physical state without publication."""
from __future__ import annotations

import json

import numpy as np
import pytest
from test_windowed_storm import window_inputs


def snapshot(tmp_path):
    from maple.surface.voxels.windowed_snapshot import save_windowed_state_snapshot

    s, ctx, _control, _schedule = window_inputs(np, tmp_path)
    b = s['state0'].bed
    path = tmp_path/'snapshot.npz'
    save_windowed_state_snapshot(b.voxel_column, b.active_layer, 0., path, step=0,
                                 geometry=ctx.geometry, mass_resolution_kg=ctx.mass_resolution_kg,
                                 sediment_availability=b.sediment_availability, water=b.water)
    return path, b, ctx


@pytest.mark.parametrize('corruption', ['packet','generation','header','class_order','shape','negative','extra','time'])
def test_corrupt_window_snapshot_is_refused(tmp_path, corruption):
    from maple.surface.voxels.windowed_snapshot import load_windowed_state_snapshot

    path, bed, ctx = snapshot(tmp_path)
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {k:loaded[k].copy() for k in loaded.files}
    meta = json.loads(str(arrays['metadata'].item()))
    if corruption == 'packet':
        arrays['prefix_packet'].flat[0] += 1.
    elif corruption == 'generation':
        meta['generation'] += 1
    elif corruption == 'header':
        meta['header_digest'] = '0'*64
    elif corruption == 'class_order':
        meta['class_ids'] = meta['class_ids'][::-1]
    elif corruption == 'shape':
        arrays['window_kg'] = arrays['window_kg'][:,:-1]
    elif corruption == 'negative':
        arrays['mobile_mass_kg'].flat[0] = -1.
    elif corruption == 'extra':
        arrays['unrecognized'] = np.zeros(1)
    elif corruption == 'time':
        meta['time_s'] = -1.
    arrays['metadata'] = np.array(json.dumps(meta))
    corrupted = tmp_path/'bad.npz'
    np.savez(corrupted, **arrays)
    pinned = {str(p):p.stat().st_size for p in (tmp_path/'store').rglob('*') if p.is_file()}
    with pytest.raises((ValueError, TypeError)):
        load_windowed_state_snapshot(corrupted, geometry=ctx.geometry, grain_classes=ctx.grain_classes,
                                      mass_resolution_kg=ctx.mass_resolution_kg, max_bytes=10**8)
    assert pinned == {str(p):p.stat().st_size for p in (tmp_path/'store').rglob('*') if p.is_file()}
    assert bed.voxel_column.pending_deep is None


def test_snapshot_uncompressed_budget_precedes_numpy_load(tmp_path, monkeypatch):
    from maple.surface.voxels import windowed_snapshot as ws

    path, _bed, ctx = snapshot(tmp_path)
    def forbidden(*a, **kw):
        raise AssertionError('loaded archive before uncompressed size gate')
    monkeypatch.setattr(ws.np, 'load', forbidden)
    with pytest.raises(ValueError, match='bounded uncompressed size'):
        ws.load_windowed_state_snapshot(path, geometry=ctx.geometry, grain_classes=ctx.grain_classes,
                                        mass_resolution_kg=ctx.mass_resolution_kg, max_bytes=1)
