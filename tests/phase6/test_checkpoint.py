import dataclasses
import json

import numpy as np
import pytest

from maple_syrup.checkpoint import load_checkpoint, save_checkpoint, sha256_file
from maple_syrup.complete_event import complete_event
from maple_syrup.sediment_event import SedimentEventError

from .test_complete_event import fixture

IDENTITY = {'case': 'controlled actual MAPLE bed', 'forcing': 'fixed', 'source': 'test', 'options': {'dt': 1}}


def arrays(value, path=''):
    if isinstance(value, np.ndarray):
        yield path, value
    elif dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            yield from arrays(getattr(value, f.name), path + '.' + f.name)
    elif isinstance(value, dict):
        for key, v in value.items():
            yield from arrays(v, path + '.' + key)


def assert_science_equal(a, b):
    first, second = dict(arrays(a.dry_state)), dict(arrays(b.dry_state))
    assert first.keys() == second.keys()
    for name, value in first.items():
        np.testing.assert_array_equal(value, second[name], err_msg=name)
    for key in ('hydrograph', 'sediment_hydrograph', 'cumulative_pickup_kg', 'cumulative_deposition_kg',
                'cumulative_export_request_kg', 'initial_bed_by_cell_class_kg', 'ledger_process_totals_reset_kg'):
        np.testing.assert_array_equal(getattr(a.progress.result, key), getattr(b.progress.result, key), err_msg=key)
    assert a.accounting == b.accounting
    assert a.dry_state.t_s == b.dry_state.t_s
    assert a.progress.quiet == b.progress.quiet
    assert a.progress.result.n_accepted_steps == b.progress.result.n_accepted_steps
    assert a.dry_state.bed.committed_topography.commit_count == b.dry_state.bed.committed_topography.commit_count


@pytest.mark.parametrize('wet', [False, True])
def test_uninterrupted_equals_reloaded_including_wet_bed_and_hold(tmp_path, wet):
    args, kw = fixture(wet=wet)
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    assert paused.result.state.t_s == 2
    if wet:
        assert np.any(paused.result.state.bed.water.mobile_mass_by_cell_class_kg > 0)
        assert np.any(paused.result.state.bed.ledger.pending_bed_mass_change_kg != 0)
    else:
        assert paused.quiet.since_s == 2
    count = paused.result.state.bed.committed_topography.commit_count
    path = tmp_path / 'checkpoint'
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    assert restored.result.state.bed.committed_topography.commit_count == count
    for name, value in arrays(paused.result.state.bed.ledger):
        np.testing.assert_array_equal(value, dict(arrays(restored.result.state.bed.ledger))[name])
    full = complete_event(*args, **kw)
    resumed = complete_event(restored.result.state, *args[1:], **kw, continuation=restored)
    assert_science_equal(full, resumed)
    if wet:
        assert full.progress.result.by_class['actual_pickup'].sum() > 0


def test_complete_handoff_load_is_explicit_and_reset_cannot_replay(tmp_path):
    args, kw = fixture()
    out = complete_event(*args, **kw)
    path = tmp_path / 'handoff'
    save_checkpoint(path, out, args[1], args[2], args[6], IDENTITY)
    with pytest.raises(SedimentEventError, match='completed'):
        load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    loaded = load_checkpoint(path, args[1], args[2], args[6], IDENTITY, allow_complete=True)
    assert_science_equal(out, loaded)
    with pytest.raises(SedimentEventError, match='wet checkpoint'):
        complete_event(loaded.dry_state, *args[1:], **kw, continuation=loaded)


def checkpoint_fixture(path):
    args, kw = fixture(wet=True)
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    return args, kw


@pytest.mark.parametrize('kind', ['schema', 'source', 'forcing', 'options', 'case', 'missing', 'hash', 'dtype', 'negative'])
def test_corrupt_or_incompatible_checkpoint_refused(tmp_path, kind):
    path = tmp_path / 'checkpoint'
    args, _kw = checkpoint_fixture(path)
    metadata = path / 'checkpoint.json'
    m = json.loads(metadata.read_text())
    expected = json.loads(json.dumps(IDENTITY))
    if kind in ('source', 'forcing', 'options', 'case'):
        expected[kind] = 'changed'
    elif kind == 'schema':
        m['schema'] = 'future/version'
    elif kind == 'missing':
        (path / 'continuation.npz').unlink()
    elif kind == 'hash':
        with (path / 'continuation.npz').open('ab') as f:
            f.write(b'corrupt')
    else:
        node = m['payload']['fields']['result']['fields']['state']['state']['sediment_velocity_m_s']
        name = node['array']
        with np.load(path / 'continuation.npz') as a:
            data = {k: a[k].copy() for k in a.files}
        if kind == 'dtype':
            data[name] = data[name].astype(np.float32)
            node['dtype'] = data[name].dtype.str
        else:
            data[name].flat[0] = -1
        np.savez(path / 'continuation.npz', **data)
        m['files']['continuation.npz'] = sha256_file(path / 'continuation.npz')
    metadata.write_text(json.dumps(m))
    with pytest.raises((SedimentEventError, OSError, ValueError)):
        load_checkpoint(path, args[1], args[2], args[6], expected)


def test_existing_checkpoint_not_overwritten(tmp_path):
    path = tmp_path / 'checkpoint'
    args, kw = checkpoint_fixture(path)
    before = (path / 'checkpoint.json').read_bytes()
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    with pytest.raises(SedimentEventError, match='exists'):
        save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    assert (path / 'checkpoint.json').read_bytes() == before


def test_numpy_numba_resume_equivalence(tmp_path):
    """Hydraulic sweep implementations (numba vs array) with the SAME
    transport kernel implementation: bitwise science equality. The numba
    and array transport kernels agree only to round-off and are compared
    separately (tests/phase7b)."""
    pytest.importorskip('numba')
    args, kw = fixture(wet=True, implementation='numba', transport_implementation='array')
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    path = tmp_path / 'compiled'
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    loaded = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    compiled = complete_event(loaded.result.state, *args[1:], **kw, continuation=loaded)
    array_args, array_kw = fixture(wet=True, implementation='array')
    array = complete_event(*array_args, **array_kw)
    assert_science_equal(compiled, array)


@pytest.mark.parametrize('kind', ['grid_shape', 'class_shape', 'counter', 'quiet', 'columns', 'ledger'])
def test_rehashed_structural_corruption_refused(tmp_path, kind):
    path = tmp_path / 'checkpoint'
    args, _kw = checkpoint_fixture(path)
    metadata = path / 'checkpoint.json'
    m = json.loads(metadata.read_text())
    root = m['payload']['fields']
    result = root['result']['fields']
    with np.load(path / 'continuation.npz') as archive:
        data = {k: archive[k].copy() for k in archive.files}
    if kind == 'counter':
        result['n_maple_water_calls'] += 2
    elif kind == 'quiet':
        root['quiet']['fields']['since_s'] = 0.
    elif kind == 'columns':
        result['sediment_columns']['tuple'][1] = 'wrong_units'
    else:
        if kind == 'grid_shape':
            node = result['cumulative_pickup_kg']
        elif kind == 'class_shape':
            node = result['by_class']['dict']['export_actual']
        else:
            node = result['state']['state']['bed']['bed']['ledger']['fields']['pending_bed_mass_change_kg']
        name = node['array']
        if kind == 'ledger':
            data[name].flat[0] += 1.
        else:
            data[name] = data[name][0:1].copy()
            node['shape'] = list(data[name].shape)
    np.savez(path / 'continuation.npz', **data)
    m['files']['continuation.npz'] = sha256_file(path / 'continuation.npz')
    metadata.write_text(json.dumps(m))
    with pytest.raises((SedimentEventError, ValueError)):
        load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
