"""SOURCE-authored P1 controls; execution requires a separate owning CPU gate.

All physical inputs below are freshly produced by the existing tiny authored
phase6 fixture. No saved terrain, forcing, result or checkpoint is consumed.
The optional budget is serialization metadata, never physical validation.
"""
from __future__ import annotations

import dataclasses
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pytest
from numpy.lib import _format_impl
from phase6.test_complete_event import fixture

from maple.core.backend import is_array, to_host
from maple_syrup import checkpoint as c
from maple_syrup.complete_event import EventProgress, complete_event
from maple_syrup.sediment_event import SedimentEventError


@pytest.fixture(scope='module')
def wet_case():
    args, kw = fixture(wet=True, transport_scheme='upwind', force_final_commit=False)
    kw = {**kw, 'max_report_rows': 4096}
    progress = complete_event(*args, **kw, checkpoint_callback=lambda _progress: True)
    assert type(progress) is EventProgress and progress.status == 'wet'
    return args, kw, progress


def budget(case, identity=None, limits=None):
    args, _kw, progress = case
    return c.checkpoint_write_budget(progress, args[1], args[2], args[6],
                                     {'case': 'authored-tiny-P1'} if identity is None else identity,
                                     limits=c.CheckpointWriteLimits() if limits is None else limits)


def wire(value, *, preexisting=False):
    registry = c._registry()
    first = {'a0000': np.zeros(())} if preexisting else {}
    second = dict(first)
    write = c._encode_wire(value, None, first, registry, is_array, to_host)
    layout = c._WriteLayout(c.CheckpointWriteLimits(), 0)
    projected = c._encode_wire(value, None, second, registry, None, None,
                               _mode=c._LAYOUT_WIRE, _layout=layout)
    return write, projected, first, second, layout


def read_headers(path):
    rows = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            with archive.open(info) as member:
                version = _format_impl.read_magic(member)
                shape, fortran, dtype = _format_impl._read_array_header(member, version)
                offset = member.tell()
            rows.append((info.filename[:-4], dtype.str, shape, fortran, offset, info.file_size,
                         info.compress_type, info.header_offset))
    return rows


def test_one_shared_occurrence_name_and_reference_dict_even_with_repeated_backing():
    array = np.arange(6, dtype=np.float64).reshape(2, 3)
    value = {'first': array, 'again': [array, np.float64(-0.0), np.int64(7), '\x7f😀']}
    written, projected, first, second, layout = wire(value)
    assert written == projected
    assert tuple(first) == tuple(second) == tuple(m.key for m in layout.members) == ('a0000', 'a0001')
    assert first['a0000'] is first['a0001'] is second['a0001'] is array
    assert layout.members[0].data_bytes == layout.members[1].data_bytes == array.nbytes


def test_layout_uses_shared_wire_counter_not_a_second_member_counter():
    written, projected, first, second, layout = wire(np.zeros((), dtype=np.float64), preexisting=True)
    assert written == projected == {'array': 'a0001', 'dtype': np.dtype(np.float64).str, 'shape': []}
    assert tuple(first) == tuple(second) == ('a0000', 'a0001')
    assert tuple(m.key for m in layout.members) == ('a0001',)


@pytest.mark.parametrize('value', [np.zeros(()), np.zeros(0), np.zeros((1, 6)),
                                  np.zeros((2, 3), dtype=np.int64), np.ones(4, dtype=np.bool_)])
def test_actual_numpy_header_and_forced_local_zip64_formula(value, tmp_path):
    limit = c.CheckpointWriteLimits()
    member = c._write_npy_member('native', value, limit)
    layout = c._write_archive('continuation.npz', (member,), limit)
    target = tmp_path / 'native.npz'
    np.savez(target, native=value)
    rows = read_headers(target)
    assert rows[0][:6] == ('native', value.dtype.str, value.shape, False,
                           member.npy_header_bytes, member.npy_bytes)
    assert rows[0][6] == zipfile.ZIP_STORED
    assert target.stat().st_size == layout.file_bytes == member.npy_bytes + 22 + 96 + 2 * len('native.npy')
    with target.open('rb') as file:
        local = file.read(30)
    assert int.from_bytes(local[28:30], 'little') == 20


@pytest.mark.parametrize('scalar', [np.bytes_(b''), np.bytes_(b'abc'), np.int64(0), np.float64(-0.0), np.bool_(True)])
def test_actual_scalar_normalization_and_header_bytes(scalar, tmp_path):
    normalized = np.asanyarray(scalar)
    # Empty-byte itemsize is observed from the actual dependency, not assumed.
    assert normalized.dtype.itemsize > 0
    member = c._write_npy_member('scalar', scalar, c.CheckpointWriteLimits(), scalar=True)
    target = tmp_path / 'scalar.npz'
    np.savez(target, scalar=scalar)
    assert read_headers(target)[0][:6] == ('scalar', normalized.dtype.str, normalized.shape,
                                         False, member.npy_header_bytes, member.npy_bytes)


@pytest.mark.parametrize('value', [np.asfortranarray(np.zeros((2, 3))), np.zeros((3, 4))[:, ::2]])
def test_unsupported_layout_refuses_without_copying(value):
    assert not value.flags.c_contiguous
    with pytest.raises(SedimentEventError, match='C-contiguous'):
        c._write_npy_member('array', value, c.CheckpointWriteLimits())


@pytest.mark.parametrize('value', [np.zeros(6), np.zeros(()), np.zeros((1, 1))])
def test_simultaneous_c_and_f_flags_follow_native_c_first(value):
    assert value.flags.c_contiguous and value.flags.f_contiguous
    assert c._write_npy_member('array', value, c.CheckpointWriteLimits()).fortran_order is False


@pytest.mark.parametrize('value', [np.zeros(1, dtype=object), np.zeros(1, dtype=[('field', 'f8')]),
                                  np.zeros((1, 1, 1, 1, 1)), np.zeros(1, dtype=np.complex128),
                                  np.zeros(1, dtype=np.dtype('i4', metadata={'label': 'authored'}))])
def test_dtype_and_rank_profiles_refuse(value):
    with pytest.raises(SedimentEventError):
        c._write_npy_member('array', value, c.CheckpointWriteLimits())


@pytest.mark.parametrize('value', [True, 0, -1, 1.0, np.int64(1), 2**4096])
def test_limit_admission_is_plain_bounded_and_has_no_huge_repr(value):
    limit = dataclasses.replace(c.CheckpointWriteLimits(), max_nodes=value)
    with pytest.raises(SedimentEventError) as refusal:
        c._write_limits(limit)
    assert len(str(refusal.value)) < 100


@pytest.mark.parametrize('value', [{'\x7f😀': ['quote"', 'slash\\', '\x00', -0.0, 1.25]},
                                  {'empty': [], 'truth': True, 'none': None}, {'整数': 2**128}])
def test_json_count_matches_actual_native_format_and_newline(value):
    limits = c.CheckpointWriteLimits()
    measured, _scratch = c._write_json_measure(value, limits, indent=2, newline=True)
    actual = json.dumps(value, indent=2, allow_nan=False) + '\n'
    assert measured == len(actual.encode('ascii'))
    count, _ = c._write_json_measure(value, limits, sort_keys=True)
    assert count == len(json.dumps(value, sort_keys=True).encode('ascii'))


def test_raw_metadata_caps_precede_json_encoder(monkeypatch):
    def forbidden(*_args, **_kw):
        raise BaseException('JSON encoder entered before raw cap')
    monkeypatch.setattr(json.JSONEncoder, 'iterencode', forbidden)
    for value in ('x' * 1025, 2**257, [None] * 10, {1: 'not a string key'}, float('inf')):
        with pytest.raises(SedimentEventError):
            c._write_json_measure(value, dataclasses.replace(c.CheckpointWriteLimits(), max_nodes=8))


def test_member_257_refuses_without_truncating_or_constructing_its_header(monkeypatch):
    layout = c._WriteLayout(c.CheckpointWriteLimits(), 0)
    value = np.zeros((), dtype=np.float64)
    for index in range(256):
        layout.array(f'a{index:04d}', value)
    def forbidden(*_args, **_kw):
        raise BaseException('257th header was constructed')
    monkeypatch.setattr(c, '_write_npy_member', forbidden)
    with pytest.raises(SedimentEventError, match='member cap'):
        layout.array('a0256', value)
    assert len(layout.members) == 256


def test_container_children_are_admitted_before_encoding(monkeypatch):
    layout = c._WriteLayout(dataclasses.replace(c.CheckpointWriteLimits(), max_nodes=40), 0)
    def forbidden(*_args, **_kw):
        raise BaseException('child array metadata was entered')
    monkeypatch.setattr(c, '_write_npy_member', forbidden)
    with pytest.raises(SedimentEventError, match='child cap'):
        c._encode_wire([np.zeros(())] * 50, None, {}, c._registry(), None, None,
                        _mode=c._LAYOUT_WIRE, _layout=layout)


def test_genuine_wet_layout_matches_every_native_member_and_json_byte(wet_case, tmp_path):
    args, _kw, progress = wet_case
    identity = {'case': 'authored-P1', 'escaped': '\x7f😀', 'signed_zero': -0.0}
    before = [(a, a.copy(), a.flags.writeable) for a in (progress.result.state.bed.voxel_column.mass_kg,
                                                        progress.result.state.bed.active_layer.mass_kg,
                                                        progress.result.state.storm.soil_water_m)]
    result = budget(wet_case, identity)
    saved = c.save_checkpoint(tmp_path / 'wet', progress, args[1], args[2], args[6], identity)
    for archive in result.archives:
        actual = read_headers(saved / archive.filename)
        expected = [(m.key, m.dtype, m.shape, m.fortran_order, m.npy_header_bytes, m.npy_bytes)
                    for m in archive.members]
        assert [row[:6] for row in actual] == expected
        assert (saved / archive.filename).stat().st_size == archive.file_bytes
    assert (saved / 'checkpoint.json').stat().st_size == result.checkpoint_json_bytes
    with np.load(saved / 'maple_state.npz', allow_pickle=False) as image:
        assert len(image.files) == 22
        provenance = image['provenance_json'].item()
        assert len(provenance) == result.snapshot_provenance_bytes
        assert json.loads(provenance)['identity'] == identity
    assert tuple(m.key for m in result.archives[1].members) == tuple(
        f'a{index:04d}' for index in range(len(result.archives[1].members)))
    for original, copy, writeable in before:
        np.testing.assert_array_equal(original, copy)
        assert original.flags.writeable is writeable
    assert result.host_payload_copy_bytes == 0
    assert result.additional_serialization_upper_bytes == sum((
        result.encode_finite_check_bytes, result.whole_array_serialization_fallback_bytes,
        result.nominal_iterator_buffer_pair_bytes, result.header_scalar_metadata_bytes,
        result.encoded_metadata_upper_bytes, result.json_overlap_upper_bytes,
        result.graph_digest_scratch_bytes))
    assert result.native_child_bytes == result.serialized_payload_bytes + result.checkpoint_json_bytes


def test_helper_does_not_validate_read_array_values_rebuild_hash_or_write(wet_case, monkeypatch):
    def forbidden(*_args, **_kw):
        raise BaseException('forbidden native/data/filesystem call in pure helper')
    for name in ('validate_checkpoint_state', 'validate_deep_storage',
                 'check_graph_matches_routing_source', '_graph_digest', 'sha256_file'):
        monkeypatch.setattr(c, name, forbidden)
    monkeypatch.setattr(np, 'isfinite', forbidden)
    monkeypatch.setattr(np, 'savez', forbidden)
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    result = budget(wet_case)
    assert type(result) is c.CheckpointWriteBudget
    assert dict(result.native_call_multiplicities)['validate_checkpoint_state'] == 1
    assert dict(result.native_call_multiplicities)['SedimentEventResult.closure'] == 2


def test_default_writer_does_not_select_optional_budget(wet_case, tmp_path, monkeypatch):
    args, _kw, progress = wet_case
    def forbidden(*_args, **_kw):
        raise BaseException('old writer selected optional P1 helper')
    monkeypatch.setattr(c, 'checkpoint_write_budget', forbidden)
    target = c.save_checkpoint(tmp_path / 'default', progress, args[1], args[2], args[6], {'case': 'old path'})
    assert target.is_dir()


@pytest.mark.parametrize('change', [dict(transport_scheme='characteristic'), dict(force_final_commit=True),
                                   dict(record_exchange_history=True), dict(deep_publication='boundary')])
def test_optional_profiles_refuse_without_altering_writer_contract(wet_case, change):
    args, kw, progress = wet_case
    changed = dataclasses.replace(progress, control=dataclasses.replace(progress.control, **change))
    with pytest.raises(SedimentEventError, match='optional native profile'):
        budget((args, kw, changed))


def test_counter_and_identity_refusals_are_bounded_before_native_calls(wet_case, monkeypatch):
    def forbidden(*_args, **_kw):
        raise BaseException('native validation entered')
    monkeypatch.setattr(c, 'validate_checkpoint_state', forbidden)
    args, kw, progress = wet_case
    for changed in (dataclasses.replace(progress, max_report_rows=4097),
                    dataclasses.replace(progress, max_report_rows=2**4096)):
        with pytest.raises(SedimentEventError, match='counter policy cap'):
            budget((args, kw, changed))
    for identity in ({1: 'bad key'}, {'huge': 2**4096}, {'bad': float('nan')}):
        with pytest.raises(SedimentEventError) as refusal:
            budget(wet_case, identity)
        assert len(str(refusal.value)) < 100


def test_sum_of_uncompressed_npy_members_is_the_raw_cap(wet_case):
    first = budget(wet_case)
    limit = dataclasses.replace(c.CheckpointWriteLimits(), max_uncompressed_bytes=first.uncompressed_payload_bytes - 1)
    with pytest.raises(SedimentEventError, match='byte cap'):
        budget(wet_case, limits=limit)


def test_serialization_maxima_cover_both_archives_and_finite_check_is_separate(wet_case):
    result = budget(wet_case)
    all_members = tuple(m for archive in result.archives for m in archive.members)
    assert result.whole_array_serialization_fallback_bytes == 2 * max(m.data_bytes for m in all_members)
    assert result.nominal_iterator_buffer_pair_bytes == 2 * max(
        max((16 * 1024**2) // np.dtype(m.dtype).itemsize, 1) * np.dtype(m.dtype).itemsize
        for m in all_members)
    assert result.encode_finite_check_bytes == max(
        math.prod(m.shape) + 1 for m in result.archives[1].members)
    assert result.host_payload_copy_bytes == 0


def test_primary_bed_and_exact_registry_obligations_are_not_forged(wet_case):
    args, _kw, progress = wet_case
    primary = progress.result.state
    layout = c._WriteLayout(c.CheckpointWriteLimits(), 0)
    with pytest.raises(SedimentEventError, match='primary wet bed'):
        c._encode_wire(dataclasses.replace(primary.bed), primary, {}, c._registry(), None, None,
                        _mode=c._LAYOUT_WIRE, _layout=layout)
    with pytest.raises(SedimentEventError, match='exact default Dense wet'):
        c.checkpoint_write_budget(object(), args[1], args[2], args[6], {}, limits=c.CheckpointWriteLimits())


def test_optional_none_omission_operand_refuses_before_array_equality(wet_case):
    args, kw, progress = wet_case
    malformed = dataclasses.replace(progress.control, deep_publication_high_water_records=np.zeros(2))
    changed = dataclasses.replace(progress, control=malformed)
    with pytest.raises(SedimentEventError, match='malformed native omission metadata'):
        budget((args, kw, changed))
