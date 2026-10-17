"""C1 regressions: public guards, class/context binding, splash face ledger and balance, excess-depth completion.
Actual native fixtures. Nothing here was run by its author (file-only tools)."""
from dataclasses import replace

import numpy as np
import pytest
from pond_helpers import (
    NC,
    dry_plane,
    east_outlets,
    host,
    initial,
    make_bed,
    make_event,
    make_topology,
    plane,
    wet_plane,
)

from maple_syrup.pond_event import PondCompletionPolicy, PondEventError
from test_pond_event_completion import lake


def test_replaced_held_source_is_refused_by_every_public_entry(cpu):
    event, state = wet_plane(cpu)
    event.quiet_metrics(state)  # warms the cache with the genuine source
    forged = replace(state, held_source_m=state.held_source_m + 1.0)
    for call in (lambda: event.quiet_metrics(forged), lambda: event.audit(forged),
                 lambda: event.step(forged, 0.0, 0.05), lambda: event.evolve(forged, 1.0)):
        with pytest.raises(PondEventError, match="source"):
            call()
    assert event.quiet_metrics(state)  # the genuine state is still accepted


def test_datum_and_held_source_are_frozen_against_mutation(cpu):
    event, state = wet_plane(cpu)
    assert not state.datum.initial_full_m.flags.writeable and not state.datum.initial_committed_m.flags.writeable
    with pytest.raises(ValueError):
        state.datum.initial_committed_m[0, 0] = 0.0
    with pytest.raises(ValueError):
        state.held_source_m[0, 0] = 0.0


def test_restart_state_consistency_is_checked_without_repairing_mass(cpu):
    from maple_syrup.sediment_bed import with_water

    event, state = wet_plane(cpu)
    wrong_depth = replace(state, bed=with_water(state.bed, event.context, host(state.hydraulic.depth_m) + 1e-3))
    with pytest.raises(PondEventError, match="solver depth"):
        event.quiet_metrics(wrong_depth)
    wrong_clock = replace(state, water=replace(state.water, t_end_s=5.0))
    with pytest.raises(PondEventError, match="clock"):
        event.quiet_metrics(wrong_clock)
    with pytest.raises(PondEventError, match="float64"):
        event.quiet_metrics(replace(state, phase_fraction=state.phase_fraction.astype(np.float32)))


def test_inactive_cells_must_hold_no_water_at_initialisation(cpu):
    bed, ctx = make_bed(plane(), depth=0.003)  # uniform depth, including the hole
    active = np.ones((3, 4), bool)
    active[1, 1] = False
    topo = make_topology(bed, east_outlets(3, 4), active=active, drop=0.02)
    event = make_event(bed, ctx, topo)
    with pytest.raises(PondEventError, match="inactive"):
        initial(event, bed)


def test_dry_state_cannot_be_stepped_but_can_be_audited():
    policy = PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)
    event, state = lake(0.215, outlet=False, policy=policy)
    dry = event.complete(state, 20.0).dry_state
    assert dry.status == "dry"
    before = dry.counters.n_accepted_steps
    with pytest.raises(PondEventError, match="dry"):
        event.step(dry, 0.0, 0.05)
    with pytest.raises(PondEventError, match="dry"):
        event.evolve(dry, dry.t_s + 1.0)
    assert dry.counters.n_accepted_steps == before
    audit = event.audit(dry)
    assert audit["water"]["closes"] and audit["sediment"]["closes"]
    assert event.quiet_metrics(dry)[0]


def test_bulk_density_and_class_physics_are_bound_to_the_event():
    from maple.core.parameters.grain_classes import GrainClass, GrainClassSet

    bed, ctx = make_bed(plane(), depth=0.003)
    topo = make_topology(bed, east_outlets(3, 4), drop=0.02)
    base = make_event(bed, ctx, topo)
    state = initial(base, bed)
    heavier = replace(ctx, geometry=replace(ctx.geometry, bulk_density_kg_m3=ctx.geometry.bulk_density_kg_m3 + 1.0))
    other = make_event(bed, heavier, topo)
    assert other.config_sha256 != base.config_sha256
    with pytest.raises(PondEventError, match="configuration"):
        other.quiet_metrics(state)
    classes = ctx.grain_classes.classes
    dense = GrainClassSet(classes=tuple(
        GrainClass(class_id=c.class_id, diameter_m=c.diameter_m, particle_density_kg_m3=2700.0, is_aggregate=False)
        for c in classes))
    with pytest.raises(PondEventError, match="particle density"):
        make_event(bed, replace(ctx, grain_classes=dense), topo)
    reordered = GrainClassSet(classes=tuple(reversed(classes)))
    with pytest.raises(PondEventError, match="diameters"):
        make_event(bed, replace(ctx, grain_classes=reordered), topo)


def test_splash_face_ledger_balance_and_export_crossings_are_actual(cpu):
    event, state = dry_plane(cpu)
    state = event.evolve(state, 1.5).state
    s = state.sediment
    gross_x = host(s.x_gross)
    assert gross_x.sum() > 0.0  # all cells are dry: every face crossing is a SPLASH crossing
    # the net crossing of the east boundary faces (all rows are outlets) is exactly the native splash export
    boundary = host(s.x_net)[:, -1, :].sum(axis=0)
    slack = NC * 12 * event.context.mass_resolution_kg + 1e-12
    np.testing.assert_allclose(boundary, host(s.export_splash), rtol=0.0, atol=slack)
    assert not host(s.y_net)[0].any() and not host(s.y_net)[-1].any()  # south/north faces are closed
    out = event.step(state, 60.0 / 3.6e6, 0.5)
    bal = out.record.balance
    assert bal["max_residual_kg"] <= bal["max_bound_kg"] and bal["n_operations"] > 0
    assert set(out.record.numerical_residuals) == {"pickup", "deposit", "splash_pickup", "splash_deposit"}
    assert event.audit(out.state)["sediment"]["closes"]


def test_wet_balance_uses_the_full_face_divergence(cpu):
    event, state = wet_plane(cpu)
    out = event.step(state, 100.0 / 3.6e6, 0.05)
    assert out.record.balance["max_residual_kg"] <= out.record.balance["max_bound_kg"]


def test_tiny_positive_depth_at_the_outlet_does_not_block_completion():
    policy = PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)
    event, state = lake(0.22 + 1.0e-15, outlet=True, policy=policy, load=0.0, dt=0.2)
    handoff = event.complete(state, 20.0)  # asymptotic recession: no exact-zero water is required
    assert handoff.status == "complete"
    assert handoff.accounting["final_audit"]["water"]["closes"]
    assert handoff.accounting["surface_removed_m3"] > 0.0  # the ponded water is accounted, then removed by the reset


def test_moving_excess_water_still_blocks_completion_and_depth_is_not_altered():
    policy = PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)
    event, state = lake(0.225, outlet=True, policy=policy, load=0.0, dt=0.1)
    ok, values = event.quiet_metrics(state)
    assert not ok and values["max_excess_depth_m"] == pytest.approx(0.005, abs=1e-9)
    assert values["excess_volume_m3"] > policy.surface_volume_m3 * 0 and values["n_cells_above_escape"] > 0
    moving = event.evolve(state, 0.5).state
    assert not event.quiet_metrics(moving)[0] and not moving.quiet.ready
    # the physical depth is never clipped or removed by the diagnostic
    assert host(moving.hydraulic.depth_m).max() > 0.04
