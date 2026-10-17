"""Audited `sediment_bed` callers of `is_windowed_bed` on a genuine small MAPLE Dense bed (native results unchanged) and
their explicit refusal of unknown / foreign / subclass representations BEFORE any native physical apply, inventory or
commit (pre-mutation water/context validation already present in a caller may run first)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from masy_compat_support import (
    MISSING,
    ForeignWindowedColumn,
    assert_same_arrays,
    synthetic_voxel_module,
)

from maple_syrup.sediment_bed import (
    ADAPTER_NAME,
    BedIntegrationError,
    apply_bed_demand,
    apply_bed_pickup_transport_deposit,
    bed_already_committed,
    bed_by_cell_class,
    bed_inventory,
    bed_voxel_count,
    commit_bed,
    current_routing_surface_m,
    is_windowed_bed,
    pending_journal_records,
    publish_bed_storage,
    validate_deep_storage,
)


def _demand(state, *, pickup_kg=0.0, deposit_kg=0.0):
    from maple.water import WaterProcessDemand

    pickup = np.zeros_like(state.active_layer.mass_kg)
    pickup[1, 1] = pickup_kg
    deposit = np.zeros_like(pickup)
    deposit[1, 1] = deposit_kg
    return WaterProcessDemand(pickup, deposit)


def _native_apply(state, ctx, demand):
    from maple.water import apply_water_process_demand

    g = ctx.geometry
    return apply_water_process_demand(
        state.voxel_column, state.active_layer, state.water, state.ledger, demand,
        g, ctx.grain_classes, ctx.mass_resolution_kg,
        sediment_availability=state.sediment_availability,
        initial_available_fraction=ctx.initial_available_fraction,
        adapter_name=ADAPTER_NAME, detachment_integration='rate_times_dt')


# --- genuine Dense regression -------------------------------------------------------------------------------------------
# Each physical test runs ONCE on the real Dense bed under the intact bound MAPLE; the bound voxel module is never
# altered here (native MAPLE uses its own module globals, e.g. `is_windowed_voxel_column`). The windowed absent /
# present matrix comes from two separately declared runs against different MAPLE sources, not from a fixture
# (docs/masy_dense_compatibility.md). Classifier-only absence is covered in test_dense_classifier.py.
def test_dense_inventory_by_cell_and_voxel_count_match_native(dense_bed):
    from maple.aeolian.flux.step import bed_inventory_by_class_kg

    state = dense_bed[0]
    np.testing.assert_array_equal(bed_inventory(state), bed_inventory_by_class_kg(state.voxel_column, state.active_layer))
    np.testing.assert_array_equal(bed_by_cell_class(state), state.voxel_column.mass_kg.sum(axis=2) + state.active_layer.mass_kg)
    assert bed_voxel_count(state) == state.voxel_column.mass_kg.shape[2] == 3


def test_dense_already_committed_matches_native(dense_bed):
    from maple.surface.topographic_commit.commit import is_already_committed

    state, ctx, _, _ = dense_bed
    for t in (0.0, 1.0):
        expected = is_already_committed(state.ledger, state.committed_topography, state.voxel_column,
                                        state.active_layer, ctx.geometry, ctx.mass_resolution_kg, t)
        assert bool(bed_already_committed(state, ctx, t)) == bool(expected)


def test_dense_pickup_and_deposition_are_the_native_exchange(dense_bed):
    state, ctx, _, _ = dense_bed
    initial = bed_inventory(state)
    picked, pickup_result = apply_bed_demand(state, ctx, _demand(state, pickup_kg=0.001))
    assert_same_arrays(pickup_result, _native_apply(state, ctx, _demand(state, pickup_kg=0.001)))
    assert pickup_result.actual_removal_by_class_kg.sum() > 0
    deposited, deposit_result = apply_bed_demand(picked, ctx, _demand(picked, deposit_kg=1.0))
    assert_same_arrays(deposit_result, _native_apply(picked, ctx, _demand(picked, deposit_kg=1.0)))
    np.testing.assert_allclose(initial, bed_inventory(deposited) + deposited.water.mobile_mass_by_cell_class_kg.sum((0, 1)),
                               atol=1e-10, rtol=0)


def test_dense_routing_surface_and_commit(dense_bed):
    from maple.surface.active_layer.diagnostics import diagnose_combined_surface_state

    state, ctx, graph, ref = dense_bed
    surface = diagnose_combined_surface_state(state.voxel_column, state.active_layer, ctx.geometry,
                                              ctx.mass_resolution_kg).combined_elevation_m
    reference_m = state.committed_topography.reference_elevation_m
    np.testing.assert_array_equal(current_routing_surface_m(state, ctx), reference_m + (surface - reference_m))
    picked, _ = apply_bed_demand(state, ctx, _demand(state, pickup_kg=0.001))
    outcome = commit_bed(picked, ctx, graph, ref, 1.0, force=True)
    assert outcome.did_commit and is_windowed_bed(outcome.state) is False
    np.testing.assert_allclose(bed_inventory(state),
                               bed_inventory(outcome.state) + outcome.state.water.mobile_mass_by_cell_class_kg.sum((0, 1)),
                               atol=1e-10, rtol=0)


def test_dense_storage_journal_and_windowed_only_entries(dense_bed):
    state, ctx, _, _ = dense_bed
    assert pending_journal_records(state) == 0
    assert publish_bed_storage(state, ctx) is state
    assert validate_deep_storage(ctx, state) is None
    calls = []

    def entry(*args, **kwargs):
        calls.append(True)
        raise AssertionError('paired windowed transaction reached by a dense bed')

    with pytest.raises(BedIntegrationError, match='resident-window bed'):
        apply_bed_pickup_transport_deposit(state, ctx, [], lambda view: None, entry=entry)
    assert calls == []


# --- unknown / foreign / subclass refusal across callers ------------------------------------------------------------------
def _callers(ctx, graph, ref, paired_calls):
    def entry(*args, **kwargs):
        paired_calls.append(True)
        raise AssertionError('paired windowed transaction reached by an invalid representation')

    return {
        'is_windowed_bed': lambda s: is_windowed_bed(s),
        'apply_bed_demand': lambda s: apply_bed_demand(s, ctx, _demand(s)),
        'apply_bed_pickup_transport_deposit': lambda s: apply_bed_pickup_transport_deposit(
            s, ctx, [], lambda view: None, entry=entry),
        'bed_inventory': bed_inventory,
        'bed_by_cell_class': bed_by_cell_class,
        'bed_voxel_count': bed_voxel_count,
        'bed_already_committed': lambda s: bed_already_committed(s, ctx, 0.0),
        'commit_bed': lambda s: commit_bed(s, ctx, graph, ref, 1.0, force=True),
        'current_routing_surface_m': lambda s: current_routing_surface_m(s, ctx),
        'publish_bed_storage': lambda s: publish_bed_storage(s, ctx),
        'pending_journal_records': pending_journal_records,
        'validate_deep_storage': lambda s: validate_deep_storage(ctx, s),
    }


CALLERS = ('is_windowed_bed', 'apply_bed_demand', 'apply_bed_pickup_transport_deposit', 'bed_inventory',
           'bed_by_cell_class', 'bed_voxel_count', 'bed_already_committed', 'commit_bed', 'current_routing_surface_m',
           'publish_bed_storage', 'pending_journal_records', 'validate_deep_storage')


def _representation(kind, state, monkeypatch):
    mass = state.voxel_column.mass_kg
    if kind == 'object':
        return object()
    if kind == 'namespace':
        return SimpleNamespace(mass_kg=mass, n_voxels=mass.shape[2], pending_deep=None, deep_store=None)
    if kind == 'dense_subclass':
        return type('DenseSubclass', (type(state.voxel_column),), {})(mass_kg=mass)
    if kind == 'windowed_subclass':
        # A re-exported (test-only) windowed class, so the case exists on a MAPLE without the windowed class.
        synthetic_voxel_module(monkeypatch, WindowedVoxelColumnState=ForeignWindowedColumn)
        return type('WindowedSubclass', (ForeignWindowedColumn,), {})(n_voxels=mass.shape[2])
    if kind == 'unbound_windowed':
        # The windowed-shaped class is NOT what the public module binds (truly absent there).
        synthetic_voxel_module(monkeypatch, WindowedVoxelColumnState=MISSING)
        return ForeignWindowedColumn(n_voxels=mass.shape[2])
    raise AssertionError(kind)


@pytest.mark.parametrize('kind', ['object', 'namespace', 'dense_subclass', 'windowed_subclass', 'unbound_windowed'])
@pytest.mark.parametrize('caller', CALLERS)
def test_invalid_representation_refuses_before_native_physics(dense_bed, native_spy, monkeypatch, caller, kind):
    state, ctx, graph, ref = dense_bed
    invalid = replace(state, voxel_column=_representation(kind, state, monkeypatch))
    paired_calls = []
    call = _callers(ctx, graph, ref, paired_calls)[caller]
    with pytest.raises(BedIntegrationError, match='unsupported bed representation'):
        call(invalid)
    assert native_spy == []
    assert paired_calls == []
