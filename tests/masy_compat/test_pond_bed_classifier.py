"""The new upstream caller `pond_bed.require_supported_bed` at the `is_windowed_bed` classification boundary only.

A genuine Dense bed passes the pond configuration gate; unknown / subclass columns are refused by the classifier with
its base `BedIntegrationError` before any native physical entry; a recognized (test-only re-exported) windowed binding
is handled by the existing `PondBedError` guards (see `RECOGNIZED_WINDOWED_REFUSALS`). This is not a PondEvent, pond
physics or windowed physics test; `test_pond_windowed_compat.py` covers the PondEvent and checkpoint gates."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from masy_compat_support import ForeignWindowedColumn, synthetic_voxel_module

from maple_syrup.pond_bed import PondBedError, native_commit, pickup_transport_deposit, require_supported_bed
from maple_syrup.sediment_bed import BedIntegrationError


def _pond_callers(ctx, transport_calls):
    def transport(pickup):
        transport_calls.append(True)
        raise AssertionError('pond transport reached by an invalid representation')

    return {
        'require_supported_bed': lambda s: require_supported_bed(s, ctx),
        'native_commit': lambda s: native_commit(s, ctx, 1.0, force=True),
        'pickup_transport_deposit': lambda s: pickup_transport_deposit(
            s, ctx, np.zeros(s.active_layer.mass_kg.shape[:2]), np.zeros_like(s.active_layer.mass_kg), transport),
    }


POND_CALLERS = ('require_supported_bed', 'native_commit', 'pickup_transport_deposit')


def test_real_dense_bed_passes_the_pond_gate(dense_bed, native_spy):
    state, ctx, _, _ = dense_bed
    assert require_supported_bed(state, ctx) is None
    assert native_spy == []


def _unknown(kind, state):
    mass = state.voxel_column.mass_kg
    if kind == 'object':
        return object()
    if kind == 'namespace':
        return SimpleNamespace(mass_kg=mass, n_voxels=mass.shape[2], pending_deep=None, deep_store=None)
    if kind == 'dense_subclass':
        return type('DenseSubclass', (type(state.voxel_column),), {})(mass_kg=mass)
    raise AssertionError(kind)


@pytest.mark.parametrize('kind', ['object', 'namespace', 'dense_subclass'])
@pytest.mark.parametrize('caller', POND_CALLERS)
def test_unknown_column_refused_by_classifier_before_native_physics(dense_bed, native_spy, caller, kind):
    state, ctx, _, _ = dense_bed
    transport_calls = []
    with pytest.raises(BedIntegrationError, match='unsupported bed representation') as info:
        _pond_callers(ctx, transport_calls)[caller](replace(state, voxel_column=_unknown(kind, state)))
    # The documented stricter boundary: the classifier's base error, not the pond-specific subclass.
    assert type(info.value) is BedIntegrationError
    assert native_spy == []
    assert transport_calls == []


@pytest.mark.parametrize('caller', POND_CALLERS)
def test_windowed_subclass_refused_by_classifier_before_native_physics(dense_bed, native_spy, monkeypatch, caller):
    state, ctx, _, _ = dense_bed
    synthetic_voxel_module(monkeypatch, WindowedVoxelColumnState=ForeignWindowedColumn)
    column = type('WindowedSubclass', (ForeignWindowedColumn,), {})(n_voxels=state.voxel_column.mass_kg.shape[2])
    transport_calls = []
    with pytest.raises(BedIntegrationError, match='unsupported bed representation') as info:
        _pond_callers(ctx, transport_calls)[caller](replace(state, voxel_column=column))
    assert type(info.value) is BedIntegrationError
    assert native_spy == []
    assert transport_calls == []


#: A recognized (test-only re-exported) windowed binding passes the classifier and is then handled by the existing pond
#: guards: `require_supported_bed` (default `allow_windowed=False`) refuses windowed beds; `native_commit` and
#: `pickup_transport_deposit` call it with `allow_windowed=True`, so they reach the windowed height-packet check (the
#: test class carries no packets). Classification identity only: no windowed storage or physics is exercised.
RECOGNIZED_WINDOWED_REFUSALS = {
    'require_supported_bed': 'windowed beds are unqualified for this caller',
    'native_commit': 'needs the maintained height',
    'pickup_transport_deposit': 'needs the maintained height',
}


@pytest.mark.parametrize('caller', POND_CALLERS)
def test_recognized_windowed_binding_is_handled_by_existing_pond_guards(dense_bed, native_spy, monkeypatch, caller):
    state, ctx, _, _ = dense_bed
    synthetic_voxel_module(monkeypatch, WindowedVoxelColumnState=ForeignWindowedColumn)
    column = ForeignWindowedColumn(n_voxels=state.voxel_column.mass_kg.shape[2])
    transport_calls = []
    with pytest.raises(PondBedError, match=RECOGNIZED_WINDOWED_REFUSALS[caller]):
        _pond_callers(ctx, transport_calls)[caller](replace(state, voxel_column=column))
    assert native_spy == []
    assert transport_calls == []
