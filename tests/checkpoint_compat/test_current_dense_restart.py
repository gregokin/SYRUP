"""MASY-03 prerequisite A: genuine Dense wet save / load / continuation through BOTH checkpoint codecs on whichever MAPLE is
bound (current `5b1a052e` without `CompactSedimentLedgerState`, or the reference with it). Never skipped and never run under
a synthetic binding (not run by its author; file-only tools).

Fixtures are the existing small authored native ones: `tests/phase6/test_complete_event.fixture` (legacy
`maple-syrup-checkpoint/2`) and `tests/pond_checkpoint/pond_fixture.make_case` (`maple-syrup-pond-checkpoint/1`). Negative
cases edit a real saved checkpoint: an unknown / forged compact ledger node, malformed ledger fields and a source / identity
mismatch are refused on either MAPLE (never coerced to the dense class)."""

from __future__ import annotations

import importlib
import shutil

import numpy as np
import pytest
from checkpoint_compat_support import LEDGER_MODULE, compact_present, rewrite_legacy, rewrite_pond, tree_bytes
from phase6.test_checkpoint import IDENTITY as LEGACY_IDENTITY
from phase6.test_checkpoint import arrays, assert_science_equal
from phase6.test_complete_event import fixture
from pond_fixture import IDENTITY as POND_IDENTITY
from pond_fixture import advance, assert_tree_equal, fingerprint, host, make_case

from maple_syrup.checkpoint import load_checkpoint, save_checkpoint
from maple_syrup.complete_event import complete_event
from maple_syrup.pond_checkpoint import PondCheckpointError, load_pond_checkpoint, save_pond_checkpoint
from maple_syrup.sediment_event import SedimentEventError


#: Key prefix of the derived, never-persisted `VoxelColumnState.surface_cache` subtree in `arrays(result)`.
VOXEL_SURFACE_CACHE = '.state.bed.voxel_column.surface_cache.'


def _dense():
    return vars(importlib.import_module(LEDGER_MODULE))['SedimentLedgerState']


def _committed_type():
    from maple.core.types import topographic_commit

    return vars(topographic_commit)['CommittedTopographyState']


# --- legacy codec ----------------------------------------------------------------------------------------------------
def _legacy_paused():
    args, kw = fixture(wet=True)
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    return args, kw, paused


def test_legacy_wet_dense_round_trip_restores_exact_native_state_and_continues_exactly(tmp_path):
    args, kw, paused = _legacy_paused()
    bed = paused.result.state.bed
    assert type(bed.ledger) is _dense() and type(bed.committed_topography) is _committed_type()
    # the fixture did real wet work: a live pending ledger and a mobile load are carried across the restart
    assert np.any(bed.ledger.pending_bed_mass_change_kg != 0)
    assert np.any(bed.water.mobile_mass_by_cell_class_kg > 0)

    path = save_checkpoint(tmp_path / 'checkpoint', paused, args[1], args[2], args[6], LEGACY_IDENTITY)
    restored = load_checkpoint(path, args[1], args[2], args[6], LEGACY_IDENTITY)
    rbed = restored.result.state.bed
    assert type(rbed.ledger) is _dense() and type(rbed.committed_topography) is _committed_type()

    # exact native state and accounting: every array of the result (bed ledger / committed topography / water, storm,
    # phase, cumulative accumulators, by-class accounting, hydrographs) with dtype, and the scalar ledger / counter fields.
    # The ONLY exclusion is the derived voxel `surface_cache` subtree (`init=False`, `compare=False`, never persisted; a
    # native snapshot restore rebuilds the column from `mass_kg` without it), as the existing pond comparator also does.
    first, second = ({k: v for k, v in dict(arrays(r)).items() if not k.startswith(VOXEL_SURFACE_CACHE)}
                     for r in (paused.result, restored.result))
    assert first and first.keys() == second.keys()
    for name, value in first.items():
        assert value.dtype == second[name].dtype and value.shape == second[name].shape, name
        np.testing.assert_array_equal(value, second[name], err_msg=name)
    assert rbed.ledger.largest_residual_kg == bed.ledger.largest_residual_kg
    assert rbed.committed_topography.commit_count == bed.committed_topography.commit_count
    for name in ('n_accepted_steps', 'n_commits', 'n_forced_commits', 'n_net_commits', 'n_reroutes', 'n_maple_water_calls'):
        assert getattr(restored.result, name) == getattr(paused.result, name), name
    assert restored.result.state.t_s == paused.result.state.t_s
    assert restored.quiet == paused.quiet

    full = complete_event(*args, **kw)
    resumed = complete_event(restored.result.state, *args[1:], **kw, continuation=restored)
    assert_science_equal(full, resumed)
    assert full.progress.result.by_class['actual_pickup'].sum() > 0
    assert type(resumed.dry_state.bed.ledger) is _dense()


@pytest.fixture(scope='module')
def legacy_base(tmp_path_factory):
    args, _kw, paused = _legacy_paused()
    path = save_checkpoint(tmp_path_factory.mktemp('legacy_base') / 'checkpoint', paused, args[1], args[2], args[6],
                           LEGACY_IDENTITY)
    return args, path


@pytest.fixture
def legacy_copy(legacy_base, tmp_path):
    args, base = legacy_base
    target = tmp_path / 'checkpoint'
    shutil.copytree(base, target)
    return args, target


def _legacy_ledger(meta):
    return meta['payload']['fields']['result']['fields']['state']['state']['bed']['bed']['ledger']


def _legacy_load(args, path, identity=LEGACY_IDENTITY):
    return load_checkpoint(path, args[1], args[2], args[6], identity)


def test_legacy_untouched_copy_loads(legacy_copy):
    args, path = legacy_copy
    assert type(_legacy_load(args, path).result.state.bed.ledger) is _dense()


LEGACY_FIELD_EDITS = {
    'compact_type_name': lambda n: n.update(type='CompactSedimentLedgerState'),
    'missing_field': lambda n: n['fields'].pop('largest_residual_kg'),
    'extra_field': lambda n: n['fields'].update(bogus=1),
}


@pytest.mark.parametrize('kind', sorted(LEGACY_FIELD_EDITS))
def test_legacy_unknown_compact_node_and_malformed_ledger_fields_are_refused(legacy_copy, kind):
    """Current MAPLE: the compact name is an unknown type. Reference MAPLE: the dense field set lacks the compact
    `present_processes` field. Either way the node is refused, never decoded as the dense class."""
    args, path = legacy_copy
    rewrite_legacy(path, lambda m: LEGACY_FIELD_EDITS[kind](_legacy_ledger(m)))
    before = tree_bytes(path)
    with pytest.raises(SedimentEventError, match='unknown checkpoint type or fields'):
        _legacy_load(args, path)
    assert tree_bytes(path) == before


def test_legacy_forged_complete_compact_node_is_refused(legacy_copy):
    """A compact node with the compact field set over dense-shaped arrays: unknown type where MAPLE lacks the class; where
    MAPLE has it the genuine class is constructed and the native ledger validation refuses its inconsistent shapes."""
    args, path = legacy_copy
    rewrite_legacy(path, lambda m: (_legacy_ledger(m).update(type='CompactSedimentLedgerState'),
                                    _legacy_ledger(m)['fields'].update(present_processes={'tuple': [3]})))
    if compact_present():
        with pytest.raises((SedimentEventError, ValueError)):
            _legacy_load(args, path)
    else:
        with pytest.raises(SedimentEventError, match='unknown checkpoint type or fields'):
            _legacy_load(args, path)


@pytest.mark.parametrize('key', ['source', 'case'])
def test_legacy_identity_mismatch_is_refused(legacy_copy, key):
    args, path = legacy_copy
    expected = {**LEGACY_IDENTITY, key: 'changed'}
    with pytest.raises(SedimentEventError, match='identity'):
        _legacy_load(args, path, expected)


# --- pond codec ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('pickup', [5e-4, 0.1], ids=['pending_below_trigger', 'committed_and_refreshed'])
def test_pond_wet_dense_round_trip_restores_exact_native_state_and_continues_exactly(tmp_path, pickup):
    case = make_case(pickup_kg_per_class=pickup)
    event = case.event()
    paused = advance(event, event.initial_state(case.bed, case.soil), 6)
    assert type(paused.bed.ledger) is _dense() and type(paused.bed.committed_topography) is _committed_type()
    assert paused.counters.n_accepted_steps == 6 and float(host(paused.hydraulic.depth_m).max()) > 0.0
    if pickup == 0.1:
        assert paused.counters.n_commits >= 1 and paused.topology.generation >= 1
    else:
        assert float(np.abs(host(paused.bed.ledger.pending_bed_mass_change_kg)).max()) > 0.0
    before = fingerprint(paused)

    path = save_pond_checkpoint(tmp_path / 'ck', paused, event, POND_IDENTITY)
    assert fingerprint(paused) == before
    fresh = case.event()
    loaded = load_pond_checkpoint(path, fresh, POND_IDENTITY, expected_bed_storage='dense')
    assert type(loaded.bed.ledger) is _dense() and type(loaded.bed.committed_topography) is _committed_type()
    assert_tree_equal(paused, loaded)  # every array, ledger, counter, topology, quiet and accounting leaf

    full = advance(event, paused, 6)
    resumed = advance(fresh, loaded, 6)
    assert resumed.counters.n_accepted_steps == 12
    assert_tree_equal(full, resumed)


@pytest.fixture(scope='module')
def pond_base(tmp_path_factory):
    case = make_case(pickup_kg_per_class=5e-4)
    event = case.event()
    paused = advance(event, event.initial_state(case.bed, case.soil), 4)
    return case, save_pond_checkpoint(tmp_path_factory.mktemp('pond_base') / 'ck', paused, event, POND_IDENTITY)


@pytest.fixture
def pond_copy(pond_base, tmp_path):
    case, base = pond_base
    target = tmp_path / 'ck'
    shutil.copytree(base, target)
    return case, target


def _pond_refused(case, path, match=None):
    before = tree_bytes(path)
    with pytest.raises(PondCheckpointError, match=match):
        load_pond_checkpoint(path, case.event(), POND_IDENTITY)
    assert tree_bytes(path) == before, 'a refused load modified the checkpoint'


def test_pond_untouched_copy_loads(pond_copy):
    case, path = pond_copy
    assert type(load_pond_checkpoint(path, case.event(), POND_IDENTITY).bed.ledger) is _dense()


POND_FIELD_EDITS = {
    'compact_type_name': (lambda n: n.update(type='CompactSedimentLedgerState'),
                          'unknown native type|native fields of CompactSedimentLedgerState do not match'),
    'missing_field': (lambda n: n['fields'].pop('largest_residual_kg'), 'native fields of .* do not match'),
    'extra_field': (lambda n: n['fields'].update(bogus=1), 'native fields of .* do not match'),
}


@pytest.mark.parametrize('kind', sorted(POND_FIELD_EDITS))
def test_pond_unknown_compact_node_and_malformed_ledger_fields_are_refused(pond_copy, kind):
    case, path = pond_copy
    edit, match = POND_FIELD_EDITS[kind]
    rewrite_pond(path, lambda m: edit(m['payload']['native']['ledger']))
    _pond_refused(case, path, match)


def test_pond_unknown_compact_node_names_the_actual_reason(pond_copy):
    """Where MAPLE lacks the class the node is an UNKNOWN type; where it has it, the dense field set mismatches."""
    case, path = pond_copy
    rewrite_pond(path, lambda m: m['payload']['native']['ledger'].update(type='CompactSedimentLedgerState'))
    _pond_refused(case, path, 'native fields of CompactSedimentLedgerState' if compact_present() else 'unknown native type')


def test_pond_forged_complete_compact_node_is_refused(pond_copy):
    case, path = pond_copy
    rewrite_pond(path, lambda m: (m['payload']['native']['ledger'].update(type='CompactSedimentLedgerState'),
                                  m['payload']['native']['ledger']['fields'].update(present_processes={'tuple': [3]})))
    _pond_refused(case, path, None if compact_present() else 'unknown native type')


@pytest.mark.parametrize('field', ['maple_syrup_tree_sha256', 'maple_package_sha256'])
def test_pond_source_pair_mismatch_is_refused(pond_copy, field):
    """The source-bound contract: a checkpoint written by another SYRUP / MAPLE digest is refused before any payload is
    read (no migration across the prerequisite-A SYRUP digest change)."""
    case, path = pond_copy
    rewrite_pond(path, lambda m: m['source'].update({field: 'f' * 64}))
    _pond_refused(case, path, 'source pair')
