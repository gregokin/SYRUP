"""MASY-02A: the strict `is_windowed_bed` classification at the newer PondEvent / checkpoint gates (bounded API checks).

Dense part (runs on every MAPLE): a real native Dense pond bed passes `PondEvent.initial_state`; a dense
`save_pond_checkpoint` / `load_pond_checkpoint` round trip plus continued steps runs with or without
`CompactSedimentLedgerState` (MASY-03 prerequisite A; see docs/current_maple_checkpoint_compatibility.md); unknown columns and Dense subclasses are refused with the
classifier's exact base `BedIntegrationError` (not the `PondBedError` subclass) by `initial_state` and by the save gate,
which writes nothing.

Windowed part (genuine `WindowedVoxelColumnState` only): skips explicitly when the bound MAPLE lacks the class. It reuses
the builders of `tests/pond_windowed` (a real resident-window bed from the native constructors); the MAPLE class is never
deleted or rebound. The windowed physics, storage and checkpoint behaviour are covered by `tests/pond_windowed*`, not here.

Not a coupled-simulation acceptance test, not a performance or GPU claim."""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
for _sub in ('pond_windowed', 'pond_event', 'pond_checkpoint'):
    sys.path.insert(1, str(_TESTS / _sub))

from pond_fixture import IDENTITY, advance, assert_tree_equal, make_case  # noqa: E402
from windowed_helpers import pair  # noqa: E402

from maple_syrup.pond_bed import PondBedError, require_supported_bed  # noqa: E402
from maple_syrup.pond_checkpoint import load_pond_checkpoint, save_pond_checkpoint  # noqa: E402
from maple_syrup.sediment_bed import BedIntegrationError, is_windowed_bed  # noqa: E402


def _column(kind, column):
    mass = column.mass_kg
    if kind == 'object':
        return object()
    if kind == 'namespace':
        return SimpleNamespace(mass_kg=mass, n_voxels=mass.shape[2], pending_deep=None, deep_store=None)
    if kind == 'dense_subclass':
        return type('DenseSubclass', (type(column),), {})(mass_kg=mass)
    raise AssertionError(kind)


KINDS = ('object', 'namespace', 'dense_subclass')


def _assert_exact_base_error(info):
    # The documented stricter boundary: the classifier's base error, not the pond-specific subclass.
    assert type(info.value) is BedIntegrationError


# --- genuine Dense pond bed -----------------------------------------------------------------------------------
def test_native_dense_pond_bed_is_accepted_by_the_event_gates():
    case = make_case()
    event = case.event()
    assert is_windowed_bed(case.bed) is False
    assert require_supported_bed(case.bed, case.ctx) is None
    assert require_supported_bed(case.bed, case.ctx, allow_windowed=True) is None
    state = event.initial_state(case.bed, case.soil)
    assert is_windowed_bed(state.bed) is False


@pytest.mark.parametrize('kind', KINDS)
def test_initial_state_refuses_unknown_and_subclass_columns(kind):
    case = make_case()
    event = case.event()
    bad = replace(case.bed, voxel_column=_column(kind, case.bed.voxel_column))
    with pytest.raises(BedIntegrationError, match='unsupported bed representation') as info:
        event.initial_state(bad, case.soil)
    _assert_exact_base_error(info)


def test_native_dense_checkpoint_round_trip_and_continuation_under_the_strict_classifier(tmp_path):
    """A real Dense `save_pond_checkpoint` / `load_pond_checkpoint` round trip on EVERY bound MAPLE, never skipped: the
    native ledger registry binds the compact class only where MAPLE has it (`checkpoint._native_ledger_types`), so current
    MAPLE `5b1a052e` (no `CompactSedimentLedgerState`) saves and restores Dense beds. The restored state, and both the
    uninterrupted and the resumed run after further real steps, are compared exactly (arrays, ledgers, counters, accounting)."""
    from maple.core.types import sediment_ledger

    case = make_case(pickup_kg_per_class=5e-4)
    event = case.event()
    paused = advance(event, event.initial_state(case.bed, case.soil), 3)
    assert type(paused.bed.ledger) is vars(sediment_ledger)['SedimentLedgerState']
    path = save_pond_checkpoint(tmp_path / 'ck', paused, event, IDENTITY)
    fresh = case.event()
    loaded = load_pond_checkpoint(path, fresh, IDENTITY, expected_bed_storage='dense')
    assert is_windowed_bed(loaded.bed) is False
    assert type(loaded.bed.ledger) is vars(sediment_ledger)['SedimentLedgerState']
    assert_tree_equal(paused, loaded)
    full = advance(event, paused, 3)
    resumed = advance(fresh, loaded, 3)
    assert resumed.counters.n_accepted_steps == paused.counters.n_accepted_steps + 3
    assert_tree_equal(full, resumed)


@pytest.mark.parametrize('kind', KINDS)
def test_save_gate_refuses_unknown_and_subclass_columns_and_writes_nothing(tmp_path, kind):
    case = make_case()
    event = case.event()
    state = event.initial_state(case.bed, case.soil)
    bad = dataclasses.replace(state, bed=replace(state.bed, voxel_column=_column(kind, state.bed.voxel_column)))
    with pytest.raises(BedIntegrationError, match='unsupported bed representation') as info:
        save_pond_checkpoint(tmp_path / 'ck', bad, event, IDENTITY)
    _assert_exact_base_error(info)
    assert list(tmp_path.iterdir()) == []


# --- genuine Windowed (explicit skip when the bound MAPLE has no such class) --------------------------------------
class _Cpu:
    name, hydraulics = 'numpy', 'numpy'
    xp = np


@pytest.fixture
def genuine_windowed_pair(tmp_path):
    from maple.core.types import voxel

    windowed_type = vars(voxel).get('WindowedVoxelColumnState')
    if windowed_type is None:
        pytest.skip('the bound MAPLE has no WindowedVoxelColumnState (genuine windowed checks are not applicable)')
    return windowed_type, pair(_CPU, tmp_path)


_CPU = _Cpu()


def test_genuine_windowed_and_dense_beds_classify_and_enter_their_declared_storage(genuine_windowed_pair):
    windowed_type, (d_event, d0, w_event, w0) = genuine_windowed_pair
    assert type(w0.bed.voxel_column) is windowed_type
    assert is_windowed_bed(w0.bed) is True
    assert is_windowed_bed(d0.bed) is False
    zeros = np.zeros(d_event.shape)
    # Declared storage and actual bed type must agree (both directions); the matching entries were built by `pair`.
    with pytest.raises(PondBedError):
        d_event.initial_state(w0.bed, zeros)
    with pytest.raises(PondBedError):
        w_event.initial_state(d0.bed, zeros)
    assert require_supported_bed(w0.bed, w_event.context, allow_windowed=True) is None
    with pytest.raises(PondBedError, match='windowed'):
        require_supported_bed(w0.bed, w_event.context)


def test_genuine_windowed_subclass_is_refused_by_the_event_gate(genuine_windowed_pair):
    windowed_type, (_, _, w_event, w0) = genuine_windowed_pair
    if not dataclasses.is_dataclass(windowed_type):
        pytest.skip('the genuine windowed class is not a dataclass; a field-preserving subclass cannot be built here')
    column = w0.bed.voxel_column
    subclass = type('WindowedSubclass', (windowed_type,), {})
    clone = subclass(**{f.name: getattr(column, f.name) for f in dataclasses.fields(windowed_type) if f.init})
    bad = replace(w0.bed, voxel_column=clone)
    with pytest.raises(BedIntegrationError, match='unsupported bed representation') as info:
        w_event.initial_state(bad, np.zeros(w_event.shape))
    _assert_exact_base_error(info)
