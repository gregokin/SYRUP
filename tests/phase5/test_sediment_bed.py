"""Real MAPLE bed exchange, commit/rerouting, and failure atomicity."""
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest

from maple_syrup.probe import build_minimal_state
from maple_syrup.routing import RoutingGraphError, build_routing_graph
from maple_syrup.sediment_bed import (
    BedContext,
    BedIntegrationError,
    BedState,
    apply_bed_demand,
    bed_inventory,
    commit_bed,
    terrain_reference,
    with_water,
)


def arrays(value):
    if isinstance(value, np.ndarray):
        yield value
    elif is_dataclass(value):
        for f in fields(value):
            yield from arrays(getattr(value, f.name))


@pytest.fixture
def bed():
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.sediment_availability import (
        sediment_availability_state_from_active_layer,
    )
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.surface.topographic_commit import (
        initial_committed_topography_state_from_physical_state,
    )

    p = build_minimal_state()
    committed = initial_committed_topography_state_from_physical_state(
        p.voxel_column, p.active_layer, p.geometry, p.mass_resolution_kg)
    available = sediment_availability_state_from_active_layer(p.active_layer.mass_kg, np.ones_like(p.active_layer.mass_kg))
    state = BedState(p.voxel_column, p.active_layer, p.water, p.ledger, committed, available)
    ctx = BedContext(p.geometry, p.grain_classes, p.mass_resolution_kg, TopographicCommitSpec(),
                     AvalanchingSpec(enabled=False), np.ones_like(p.active_layer.mass_kg))
    z = np.ones((5, 6))
    z[1:-1, 1:-1] = committed_elevation_m(committed)
    z[0, 1:-1] = .1 + .005 * np.arange(4)
    exports = np.zeros_like(z, bool)
    exports[0, 1:-1] = True
    graph = build_routing_graph(z, exports, np.full((3, 4), 21.45), .5)
    ref = terrain_reference(z, exports, state, graph)
    return state, ctx, graph, ref


def pickup(state, ctx, *, cell=(1, 1), amount=.001):
    from maple.water import WaterProcessDemand
    demand = np.zeros_like(state.active_layer.mass_kg)
    demand[cell] = amount
    return apply_bed_demand(state, ctx, WaterProcessDemand(demand, np.zeros_like(demand)))


def test_pickup_is_actual_supply_limited_maple_exchange_and_refill(bed):
    state, ctx, _, _ = bed
    before = [a.copy() for a in arrays(state)]
    after, result = pickup(state, ctx, amount=10.)
    # Every class at the chosen cell is exhausted, then MAPLE replenishes
    # the active layer from the underlying shared voxel column.
    np.testing.assert_allclose(result.actual_removal_by_cell_class_kg[1, 1], state.active_layer.mass_kg[1, 1], atol=1e-10)
    assert result.requested_removal_by_class_kg.sum() == 60.
    assert result.actual_removal_by_class_kg.sum() < 1.
    np.testing.assert_allclose(after.active_layer.mass_kg.sum(axis=-1), state.active_layer.mass_kg.sum(axis=-1), atol=1e-10)
    np.testing.assert_allclose(bed_inventory(state), bed_inventory(after) + after.water.mobile_mass_by_cell_class_kg.sum((0, 1)), atol=1e-10, rtol=0)
    for saved, original in zip(before, arrays(state), strict=True):
        np.testing.assert_array_equal(saved, original)


def test_deposition_export_cannot_exceed_actual_mobile(bed):
    from maple.water import WaterProcessDemand
    state, ctx, _, _ = bed
    loaded, _ = pickup(state, ctx)
    huge = np.full_like(state.active_layer.mass_kg, 100.)
    final, result = apply_bed_demand(loaded, ctx, WaterProcessDemand(np.zeros_like(huge), huge, boundary_export_by_cell_class_kg=huge))
    np.testing.assert_allclose(bed_inventory(final), bed_inventory(state), atol=1e-10, rtol=0)
    assert not np.any(final.water.mobile_mass_by_cell_class_kg)
    assert not np.any(result.boundary_export_by_class_kg)


@pytest.mark.parametrize('force', [False, True])
def test_commit_preserves_depth_publishes_maple_terrain_and_reroutes(bed, force):
    from maple.core.types.topographic_commit import committed_elevation_m
    state, ctx, graph, ref = bed
    initial = bed_inventory(state)
    # Three actual active-layer withdrawals lower this cell by 6 mm and
    # turn its east neighbour west; the changed cell still drains south.
    changed = state
    for _ in range(3):
        changed, _ = pickup(changed, ctx, amount=10.)
    outcome = commit_bed(changed, ctx, graph, ref, 3., force=force)
    assert outcome.did_commit and outcome.maple_result is not None
    assert outcome.trigger.pending_elevation_triggered
    np.testing.assert_array_equal(outcome.state.water.depth_m, state.water.depth_m)
    dz = committed_elevation_m(outcome.state.committed_topography) - committed_elevation_m(state.committed_topography)
    assert dz[1, 1] == pytest.approx(-.006, abs=2e-12)
    assert graph.aspect[1, 2] == 3 and outcome.graph.aspect[1, 2] == 4
    assert outcome.state.committed_topography.last_commit_time_s == 3.
    assert np.abs(outcome.maple_result.pre_commit_process_totals_kg).sum() > 0
    assert not np.any(outcome.state.ledger.process_totals_kg)
    np.testing.assert_allclose(initial, bed_inventory(outcome.state) + outcome.state.water.mobile_mass_by_cell_class_kg.sum((0, 1)), atol=1e-10, rtol=0)


def test_pit_after_commit_fails_without_mutating_any_input(bed):
    state, ctx, graph, ref = bed
    for _ in range(6):
        state, _ = pickup(state, ctx, amount=10.)
    before = [a.copy() for a in arrays(state)]
    with pytest.raises(RoutingGraphError, match='sink|flat'):
        commit_bed(state, ctx, graph, ref, 6., force=True)
    for saved, original in zip(before, arrays(state), strict=True):
        np.testing.assert_array_equal(saved, original)
    assert state.committed_topography.commit_count == 0


def test_no_commit_and_invalid_depth_policy(bed):
    state, ctx, graph, ref = bed
    result = commit_bed(state, ctx, graph, ref, 1.)
    assert not result.did_commit and result.state is state and result.graph is graph
    with pytest.raises(BedIntegrationError, match='constant_depth'):
        commit_bed(state, replace(ctx, depth_update_rule='constant_free_surface'), graph, ref, 1., force=True)
    with pytest.raises(ValueError):
        with_water(state, ctx, np.full_like(state.water.depth_m, -.1))


def test_terrain_reference_refuses_a_different_bed(bed):
    state, _, graph, ref = bed
    wrong = ref.initial_full_elevation_m.copy()
    wrong[2, 2] += .001
    with pytest.raises(BedIntegrationError, match='constant datum'):
        terrain_reference(wrong, ref.export_receiver_full, state, graph)
