"""Native tiny wet SAVE/LOAD terrain identity; Source-defined until Root admission.

The model-positive path uses the unchanged native phase6 fixture and real
SAVE/LOAD/completion. Authored metadata and copied children below are negatives,
never source/provenance substitutes. No backend/device is constructed here.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import shutil
import struct
import zipfile

import numpy as np
import pytest
from phase6.test_complete_event import fixture
from phase6.test_checkpoint import arrays, assert_science_equal

from maple_syrup import checkpoint as c
from maple_syrup.complete_event import DryHandoff, EventProgress, complete_event
from maple_syrup.sediment_bed import TerrainReference
from maple_syrup.sediment_event import SedimentEventError


IDENTITY = {'case': 'authored terrain-anchor native wet fixture', 'source': 'qualification', 'options': {'dt': 1}}
ARRAY_BYTES = 8 * 1024**2  # Test observation/copy bound only; no Native cap changes.
CHILD_BYTES = 8 * 1024**2
ARRAY_FIELDS = ('initial_full_elevation_m', 'initial_committed_elevation_m', 'export_receiver_full',
                'active_mask', 'friction_factor')
SCALAR_FIELDS = ('dx_m', 'allow_masked_nodata', 'nodata_value', 'allow_pit_storage', 'allow_flat_storage')


def forbidden(*_args, **_kwargs):
    raise AssertionError('authored negative tripwire: forbidden read/delegate')


def anchor_record(anchor):
    leaves = tuple(getattr(anchor, name) for name in ARRAY_FIELDS)
    assert sum(a.nbytes for a in leaves) <= ARRAY_BYTES
    physical = tuple((name, id(a), a.dtype.str, a.shape, a.strides, a.flags.owndata,
                      a.flags.writeable, a.flags.c_contiguous, id(a.base), a.tobytes(order='C'))
                     for name, a in zip(ARRAY_FIELDS, leaves))
    scalars = tuple((name, type(v), struct.pack('<d', v) if type(v) is float else v)
                    for name in SCALAR_FIELDS for v in (getattr(anchor, name),))
    return id(anchor), physical, scalars


def raw_leaves(item):
    leaves = tuple(arrays(item))
    assert sum(a.nbytes for _, a in leaves) <= ARRAY_BYTES
    return {name: (a.dtype.str, a.shape, a.tobytes(order='C')) for name, a in leaves}


def file_record(path):
    files = tuple(sorted(path.iterdir()))
    assert len(files) <= 3 and all(p.is_file() and not p.is_symlink() for p in files)
    assert sum(p.stat().st_size for p in files) <= CHILD_BYTES
    return tuple((p.name, p.stat().st_size, c.sha256_file(p)) for p in files)


def readonly(a):
    a.flags.writeable = False
    return a


@pytest.fixture(scope='module')
def native_wet(tmp_path_factory):
    args, kw = fixture(wet=True, implementation='array', transport_implementation='array',
                       transport_scheme='upwind')
    anchor = args[0].terrain
    before = anchor_record(anchor)
    progress = complete_event(*args, **kw, checkpoint_callback=lambda _p: True)
    assert type(progress) is EventProgress and progress.status == 'wet'
    assert progress.result.state.terrain is anchor
    path = tmp_path_factory.mktemp('terrain_anchor_native') / 'child'
    c.save_checkpoint(path, progress, args[1], args[2], args[6], IDENTITY)
    assert anchor_record(anchor) == before
    file_record(path)
    return args, kw, progress, path, anchor


def load(case, *, path=None, anchor=None, **kw):
    args, _options, _progress, child, _terrain = case
    return c.load_checkpoint(child if path is None else path, args[1], args[2], args[6], IDENTITY,
                             terrain_anchor=anchor, **kw)


def test_genuine_wet_load_anchor_identity_and_native_resume_parity(native_wet):
    args, kw, progress, path, anchor = native_wet
    before, files = anchor_record(anchor), file_record(path)
    held = load(native_wet, anchor=anchor)
    ordinary = load(native_wet)
    assert type(held) is EventProgress and held.result.state.terrain is anchor
    assert ordinary.result.state.terrain is not anchor
    assert raw_leaves(held) == raw_leaves(ordinary)
    assert c._graph_digest(held.result.state.graph) == c._graph_digest(ordinary.result.state.graph)
    assert held.result.state.graph.input_sha256 == ordinary.result.state.graph.input_sha256
    assert held.result.state.bed.committed_topography.commit_count == progress.result.state.bed.committed_topography.commit_count
    full = complete_event(*args, **kw)
    resumed = complete_event(held.result.state, *args[1:], **kw, continuation=held)
    unanchored = complete_event(ordinary.result.state, *args[1:], **kw, continuation=ordinary)
    assert type(resumed) is DryHandoff and resumed.dry_state.terrain is anchor
    assert_science_equal(full, resumed)
    assert_science_equal(unanchored, resumed)
    assert raw_leaves(full.dry_state) == raw_leaves(resumed.dry_state)
    assert full.progress.result.n_accepted_steps == resumed.progress.result.n_accepted_steps
    assert full.dry_state.t_s == resumed.dry_state.t_s
    assert anchor_record(anchor) == before and file_record(path) == files


def test_omitted_and_explicit_none_keep_default_results_and_errors(native_wet):
    args, _kw, _progress, path, _anchor = native_wet
    omitted = c.load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    explicit = load(native_wet, anchor=None)
    assert raw_leaves(omitted) == raw_leaves(explicit)
    errors = []
    for options in ({}, {'terrain_anchor': None}):
        with pytest.raises(SedimentEventError) as error:
            c.load_checkpoint(path, args[1], args[2], args[6], {'changed': True}, **options)
        errors.append((type(error.value), str(error.value)))
    assert errors[0] == errors[1]


def variant(anchor, kind):
    if kind == 'wrong_type':
        class NoAttributes:
            def __getattribute__(self, _name):
                raise AssertionError('wrong type must refuse before attribute access')
        return NoAttributes()
    if kind == 'subclass':
        class Derived(TerrainReference):
            pass
        return Derived(**vars(anchor))
    if kind == 'extra_attributes':
        changed = dataclasses.replace(anchor)
        for i in range(1024):
            object.__setattr__(changed, f'extra_{i}', None)
        return changed
    if kind == 'dx_type':
        return dataclasses.replace(anchor, dx_m=1)
    if kind == 'dx_nonfinite':
        return dataclasses.replace(anchor, dx_m=float('nan'))
    if kind == 'policy_type':
        return dataclasses.replace(anchor, allow_flat_storage=np.bool_(False))
    if kind == 'nodata_type':
        return dataclasses.replace(anchor, nodata_value=np.float64(1.))
    field = 'initial_full_elevation_m'
    a = getattr(anchor, field)
    if kind == 'writable':
        changed = a.copy()
    elif kind == 'view':
        changed = a.view()
    elif kind == 'fortran':
        changed = readonly(np.array(a, copy=True, order='F'))
        assert changed.flags.f_contiguous and not changed.flags.c_contiguous
    elif kind == 'dtype':
        changed = readonly(a.astype(np.float32))
    elif kind == 'endian':
        changed = readonly(a.astype(a.dtype.newbyteorder('S')))
    elif kind == 'shape':
        changed = readonly(a[:-1].copy())
    elif kind == 'empty':
        field = 'initial_committed_elevation_m'
        changed = readonly(np.empty((0, 1), dtype=np.float64))
    else:
        raise AssertionError(kind)
    return dataclasses.replace(anchor, **{field: changed})


@pytest.mark.parametrize('kind', ['wrong_type', 'subclass', 'extra_attributes', 'dx_type', 'dx_nonfinite',
                                  'policy_type', 'nodata_type', 'writable', 'view', 'fortran', 'dtype',
                                  'endian', 'shape', 'empty'])
def test_invalid_anchor_metadata_refuses_before_child_io(native_wet, tmp_path, monkeypatch, kind):
    anchor = native_wet[4]
    before = anchor_record(anchor)
    changed = variant(anchor, kind)
    monkeypatch.setattr(Path, 'resolve', forbidden)
    monkeypatch.setattr(c, 'memoryview', forbidden, raising=False)
    with pytest.raises(SedimentEventError, match='terrain_anchor'):
        load(native_wet, path=tmp_path / 'never_read', anchor=changed)
    assert anchor_record(anchor) == before


def test_anchor_raw_cap_refuses_before_child_io(native_wet, tmp_path, monkeypatch):
    anchor = native_wet[4]
    total = sum(getattr(anchor, name).nbytes for name in ARRAY_FIELDS)
    monkeypatch.setattr(Path, 'resolve', forbidden)
    monkeypatch.setattr(c, 'memoryview', forbidden, raising=False)
    with pytest.raises(SedimentEventError, match='terrain_anchor exceeds'):
        load(native_wet, path=tmp_path / 'never_read', anchor=anchor, max_bytes=total - 1)


@pytest.mark.parametrize('allow', [True, 0, None, np.bool_(False)])
def test_optional_complete_argument_requires_exact_false(native_wet, tmp_path, monkeypatch, allow):
    monkeypatch.setattr(Path, 'resolve', forbidden)
    with pytest.raises(SedimentEventError, match='allow_complete=False'):
        load(native_wet, path=tmp_path / 'never_read', anchor=native_wet[4], allow_complete=allow)


def test_optional_backend_refusal_is_metadata_only(native_wet, tmp_path, monkeypatch):
    # Authored negative backend result only: no CuPy module, device or positive backend claim.
    monkeypatch.setattr(c, 'backend_name', lambda _xp: 'cupy')
    monkeypatch.setattr(Path, 'resolve', forbidden)
    with pytest.raises(SedimentEventError, match='NumPy target'):
        load(native_wet, path=tmp_path / 'never_read', anchor=native_wet[4])


@pytest.mark.parametrize('bad', [0, -1, True, 1.5, None])
def test_original_byte_bound_error_precedes_optional_anchor_work(native_wet, tmp_path, monkeypatch, bad):
    monkeypatch.setattr(c, '_terrain_anchor_admission', forbidden)
    monkeypatch.setattr(c, 'backend_name', forbidden)
    with pytest.raises(SedimentEventError, match='positive integer byte bound'):
        load(native_wet, path=tmp_path / 'never_read', anchor=object(), max_bytes=bad)


@pytest.mark.parametrize('field', ARRAY_FIELDS)
def test_each_static_array_mismatch_preserves_original_anchor_and_child(native_wet, field):
    anchor, path = native_wet[4], native_wet[3]
    before, files = anchor_record(anchor), file_record(path)
    a = getattr(anchor, field).copy()
    if a.dtype == np.bool_:
        a.flat[0] = not bool(a.flat[0])
    else:
        a.flat[0] = np.nextafter(a.flat[0], np.inf)
    changed = dataclasses.replace(anchor, **{field: readonly(a)})
    with pytest.raises(SedimentEventError, match='differs from terrain_anchor'):
        load(native_wet, anchor=changed)
    assert anchor_record(anchor) == before and file_record(path) == files


@pytest.mark.parametrize('field', SCALAR_FIELDS)
def test_scalar_or_policy_mismatch_precedes_array_byte_reads(native_wet, monkeypatch, field):
    anchor = native_wet[4]
    value = getattr(anchor, field)
    if type(value) is bool:
        value = not value
    elif field == 'dx_m':
        value = float(np.nextafter(value, np.inf))
    else:
        value = 0. if value is None else None
    changed = dataclasses.replace(anchor, **{field: value})
    before, files = anchor_record(anchor), file_record(native_wet[3])
    monkeypatch.setattr(c, 'memoryview', forbidden, raising=False)
    with pytest.raises(SedimentEventError, match='differs from terrain_anchor'):
        load(native_wet, anchor=changed)
    assert anchor_record(anchor) == before and file_record(native_wet[3]) == files


def authored_static():
    # Bounded comparator-only record, not a native model state or fabricated LOAD success.
    return TerrainReference(readonly(np.zeros((3, 3), dtype=np.float64)),
                            readonly(np.zeros((1, 1), dtype=np.float64)),
                            readonly(np.zeros((3, 3), dtype=np.bool_)),
                            readonly(np.ones((1, 1), dtype=np.bool_)),
                            readonly(np.ones((1, 1), dtype=np.float64)), .5)


def test_signed_zero_array_and_scalar_bits_are_not_normalized():
    first = authored_static()
    c._terrain_anchor_compare(first, first, ARRAY_BYTES)
    changed = first.initial_full_elevation_m.copy()
    changed[0, 0] = -0.
    other = dataclasses.replace(first, initial_full_elevation_m=readonly(changed))
    with pytest.raises(SedimentEventError, match='initial_full_elevation_m'):
        c._terrain_anchor_compare(first, other, ARRAY_BYTES)
    with pytest.raises(SedimentEventError, match='nodata_value'):
        c._terrain_anchor_compare(dataclasses.replace(first, nodata_value=0.),
                                  dataclasses.replace(first, nodata_value=-0.), ARRAY_BYTES)


def test_noncanonical_bool_byte_difference_is_not_value_equality():
    first = authored_static()
    changed = first.active_mask.copy()
    changed.view(np.uint8)[0, 0] = 2
    assert bool(changed[0, 0]) is True
    with pytest.raises(SedimentEventError, match='active_mask'):
        c._terrain_anchor_compare(first, dataclasses.replace(first, active_mask=readonly(changed)), ARRAY_BYTES)


def copied_child(native_wet, tmp_path):
    target = tmp_path / 'negative_child'
    shutil.copytree(native_wet[3], target)
    return target


@pytest.mark.parametrize('kind', ['windowed', 'csr', 'deep_key'])
def test_optional_storage_refuses_before_payload_hashes(native_wet, tmp_path, monkeypatch, kind):
    target = copied_child(native_wet, tmp_path)
    metadata = target / 'checkpoint.json'
    data = json.loads(metadata.read_text())
    if kind == 'windowed':
        data['bed_storage'] = 'windowed'
    elif kind == 'csr':
        data['phase_storage'] = 'csr'
    else:
        data['deep_storage'] = None  # Original malformed-record error must also precede payload hashes.
    metadata.write_text(json.dumps(data, allow_nan=False))
    before, files = anchor_record(native_wet[4]), file_record(target)
    expected = 'deep_storage record is malformed' if kind == 'deep_key' else 'terrain_anchor requires dense bed storage'
    with monkeypatch.context() as patch:
        patch.setattr(c, 'sha256_file', forbidden)
        with pytest.raises(SedimentEventError, match=expected):
            load(native_wet, path=target, anchor=native_wet[4])
    assert anchor_record(native_wet[4]) == before and file_record(target) == files


def companion_arrays(target):
    with zipfile.ZipFile(target / 'continuation.npz') as archive:
        infos = archive.infolist()
        assert len(infos) <= 256 and sum(i.file_size for i in infos) <= ARRAY_BYTES
    with np.load(target / 'continuation.npz', allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    assert sum(a.nbytes for a in result.values()) <= ARRAY_BYTES
    return result


def write_negative(target, data, leaves):
    assert len(leaves) <= 256 and sum(a.nbytes for a in leaves.values()) <= ARRAY_BYTES
    raw = json.dumps(data, allow_nan=False)
    assert len(raw.encode()) <= 1024**2
    np.savez(target / 'continuation.npz', **leaves)
    data['files']['continuation.npz'] = c.sha256_file(target / 'continuation.npz')
    (target / 'checkpoint.json').write_text(json.dumps(data, allow_nan=False))


def test_zero_state_optional_root_refuses_after_original_used_array_check(native_wet, tmp_path):
    target = copied_child(native_wet, tmp_path)
    data = json.loads((target / 'checkpoint.json').read_text())
    data['payload'] = None
    # Original arrays are still present: the original error must win over the optional count refusal.
    write_negative(target, data, companion_arrays(target))
    with pytest.raises(SedimentEventError, match='unused or missing checkpoint arrays'):
        load(native_wet, path=target, anchor=native_wet[4])
    write_negative(target, data, {})
    before, files = anchor_record(native_wet[4]), file_record(target)
    with pytest.raises(SedimentEventError, match='exactly one bound state'):
        load(native_wet, path=target, anchor=native_wet[4])
    assert anchor_record(native_wet[4]) == before and file_record(target) == files


def test_one_state_wrong_root_is_not_accepted(native_wet, tmp_path, monkeypatch):
    target = copied_child(native_wet, tmp_path)
    data = json.loads((target / 'checkpoint.json').read_text())
    data['payload'] = {'dict': {'progress': data['payload']}}
    write_negative(target, data, companion_arrays(target))
    calls = []
    actual = c._terrain_anchor_compare
    def counted(*args):
        calls.append(1)
        return actual(*args)
    monkeypatch.setattr(c, '_terrain_anchor_compare', counted)
    with pytest.raises(SedimentEventError, match='exactly one bound state'):
        load(native_wet, path=target, anchor=native_wet[4])
    assert calls == [1]


def test_multiple_states_are_all_compared_then_refused(native_wet, tmp_path, monkeypatch):
    target = copied_child(native_wet, tmp_path)
    data = json.loads((target / 'checkpoint.json').read_text())
    leaves = companion_arrays(target)
    assert len(leaves) * 2 <= 256 and sum(a.nbytes for a in leaves.values()) * 2 <= ARRAY_BYTES
    first = data['payload']
    second = json.loads(json.dumps(first))
    def distinct(value):
        if isinstance(value, dict):
            if set(value) == {'array', 'dtype', 'shape'}:
                old = value['array']
                value['array'] = 'second_' + old
            else:
                for v in value.values():
                    distinct(v)
        elif isinstance(value, list):
            for v in value:
                distinct(v)
    distinct(second)
    doubled = {**leaves, **{'second_' + name: a for name, a in leaves.items()}}
    data['payload'] = {'list': [first, second]}
    write_negative(target, data, doubled)
    calls = []
    actual = c._terrain_anchor_compare
    def counted(*args):
        calls.append(1)
        return actual(*args)
    monkeypatch.setattr(c, '_terrain_anchor_compare', counted)
    before, files = anchor_record(native_wet[4]), file_record(target)
    with pytest.raises(SedimentEventError, match='exactly one bound state'):
        load(native_wet, path=target, anchor=native_wet[4])
    assert calls == [1, 1]
    assert anchor_record(native_wet[4]) == before and file_record(target) == files


def test_existing_payload_hash_error_is_preserved_with_anchor(native_wet, tmp_path):
    target = copied_child(native_wet, tmp_path)
    with (target / 'continuation.npz').open('ab') as handle:
        handle.write(b'authored negative corruption')
    before, files = anchor_record(native_wet[4]), file_record(target)
    for anchor in (None, native_wet[4]):
        with pytest.raises(SedimentEventError, match='checkpoint payload hash mismatch'):
            load(native_wet, path=target, anchor=anchor)
    assert anchor_record(native_wet[4]) == before and file_record(target) == files


def test_completed_bundle_default_is_explicit_and_anchor_cannot_replay_reset(native_wet, tmp_path):
    args, kw = native_wet[:2]
    complete = complete_event(*args, **kw)
    path = tmp_path / 'complete'
    c.save_checkpoint(path, complete, args[1], args[2], args[6], IDENTITY)
    before, files = anchor_record(native_wet[4]), file_record(path)
    ordinary = load(native_wet, path=path, allow_complete=True)
    assert type(ordinary) is DryHandoff
    assert_science_equal(complete, ordinary)
    with pytest.raises(SedimentEventError, match='completed event cannot resume'):
        load(native_wet, path=path, anchor=native_wet[4])
    assert anchor_record(native_wet[4]) == before and file_record(path) == files
