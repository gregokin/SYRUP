"""Net-elevation geometry gate of the pond event with ACTUAL native MAPLE pickups/deposits/commits. Dry domain, zero rain, so
only the manufactured native bed changes move the surface (`move_mass`: a native pickup at one cell and a native deposit of
exactly the removed mass at another, leaving no mobile pool for the dry step to settle back). Nothing here was run by its
author (file-only tools)."""
from dataclasses import replace

import numpy as np
import pytest
from pond_helpers import (
    east_outlets,
    host,
    initial,
    make_bed,
    make_event,
    make_topology,
    move_mass,
    plane,
)

from maple_syrup.pond_bed import gate_displacement_m, native_commit, reroute_threshold_m

THRESHOLD = 0.00075


def quiet_spec(threshold=THRESHOLD):
    """MAPLE's own turnover and elapsed-time triggers off, so only the pending-elevation trigger can commit."""
    from maple.core.parameters.topographic_commit import TopographicCommitSpec

    return TopographicCommitSpec(commit_interval_s=1.0e9, active_layer_turnover_fraction_trigger=0.9,
                                 pending_elevation_trigger_m=threshold)


def dry_event(spec=None, **kw):
    bed, ctx = make_bed(plane(), depth=0.0, spec=spec)
    topo = make_topology(bed, east_outlets(3, 4), drop=0.02)
    event = make_event(bed, ctx, topo, **kw)
    return event, initial(event, bed), ctx


def gate_of(event, state):
    rt = event._check_state(state)
    return gate_displacement_m(state.bed, event.context, state.held_source_m, rt.active)


def test_threshold_is_the_actual_maple_value_and_gate_starts_at_zero():
    event, state, ctx = dry_event()
    assert reroute_threshold_m(ctx) == THRESHOLD
    assert gate_of(event, state) == 0.0


def test_maple_trigger_commit_below_threshold_does_not_refresh():
    event, state, ctx = dry_event()  # default spec: MAPLE's turnover trigger fires first
    s1 = replace(state, bed=move_mass(state.bed, ctx, (1, 1), (1, 2), 0.0003))
    assert gate_of(event, s1) == pytest.approx(0.0003, abs=1e-9) and gate_of(event, s1) < THRESHOLD
    out = event.step(s1, 0.0, 0.1)
    assert out.record.commit is not None and out.record.commit["reason"] == "maple_trigger"
    assert not out.record.refreshed and out.state.topology is s1.topology
    np.testing.assert_array_equal(host(out.state.held_source_m), host(s1.held_source_m))
    c = out.state.counters
    assert c.n_commits == 1 and c.n_refreshes == 0 and out.record.commit["net_after_m"] < THRESHOLD


def test_forced_native_commit_on_its_own_never_moves_geometry():
    event, state, ctx = dry_event(spec=quiet_spec())
    bed = move_mass(state.bed, ctx, (1, 1), (1, 2), 0.0006)
    committed = native_commit(bed, ctx, 0.05, force=True)  # a forced accounting commit
    assert committed.did_commit
    out = event.step(replace(state, bed=committed.state), 0.0, 0.1)
    assert not out.record.refreshed and out.state.topology is state.topology and out.state.counters.n_refreshes == 0


def test_threshold_crossing_forces_a_native_commit_before_the_time_commit_then_refreshes():
    from maple.core.types.topographic_commit import committed_elevation_m

    event, state, ctx = dry_event(spec=quiet_spec())
    bed = move_mass(state.bed, ctx, (1, 1), (1, 2), 0.0006)
    s1 = replace(state, bed=native_commit(bed, ctx, 0.05, force=True).state)  # committed; held source lags by 0.6 mm
    first = event.step(s1, 0.0, 0.1)
    assert first.record.commit is None and not first.record.refreshed  # 0.6 mm < 0.75 mm: nothing yet
    s2 = replace(first.state, bed=move_mass(first.state.bed, ctx, (1, 1), (1, 2), 0.0006))
    # the ledger since the last commit is only 0.6 mm, the NET displacement from the held source is 1.2 mm
    assert gate_of(event, s2) == pytest.approx(0.0012, abs=1e-9)
    out = event.step(s2, 0.0, 0.1)
    rec = out.record.commit
    assert rec["reason"] == "net_elevation" and out.record.refreshed and rec["gate_before_m"] >= THRESHOLD
    assert out.state.t_s < 1.0  # long before the 1e9 s elapsed-time commit
    topo, old = out.state.topology, s2.topology
    assert topo.generation == 1 and topo is not old and topo.identity_sha256 == old.identity_sha256
    assert topo.source_sha256 != old.source_sha256
    # source = stable initial datum + actual committed delta; ring and mask fixed
    committed = host(committed_elevation_m(out.state.bed.committed_topography))
    np.testing.assert_array_equal(host(out.state.held_source_m), committed)
    z0 = state.topology.source_bed_m
    np.testing.assert_allclose(topo.source_bed_m[1, 1], z0[1, 1] - 0.0012, atol=1e-9)
    np.testing.assert_allclose(topo.source_bed_m[1, 2], z0[1, 2] + 0.0012, atol=1e-9)
    np.testing.assert_array_equal(topo.source_z_full[0], state.topology.source_z_full[0])
    np.testing.assert_array_equal(topo.source_z_full[:, [0, -1]], state.topology.source_z_full[:, [0, -1]])
    mask = np.ones((3, 4), bool)
    mask[1, 1:3] = False
    np.testing.assert_allclose(topo.source_bed_m[mask], z0[mask], rtol=0.0, atol=1e-12)
    assert out.state.counters.n_net_elevation_commits == 1 and out.state.counters.n_refreshes == 1
    audit = event.audit(out.state)
    assert audit["geometry"]["generation"] == 1 and audit["geometry"]["gate_displacement_m"] < THRESHOLD


def test_same_cell_gain_and_loss_cancel_and_different_cells_do_not_add():
    event, state, ctx = dry_event(spec=quiet_spec())
    same = replace(state, bed=move_mass(state.bed, ctx, (1, 1), (1, 1), 0.0009))  # alone this would cross
    assert gate_of(event, same) < 1e-9
    out = event.step(same, 0.0, 0.1)
    assert not out.record.refreshed and out.state.topology is state.topology

    moved = replace(state, bed=move_mass(state.bed, ctx, (1, 1), (1, 2), 0.0006))  # -0.6 mm and +0.6 mm elsewhere
    gate = gate_of(event, moved)
    assert gate == pytest.approx(0.0006, abs=1e-9) and gate < THRESHOLD  # NOT 1.2 mm
    assert not event.step(moved, 0.0, 0.1).record.refreshed


@pytest.mark.parametrize("scale, refreshed", [(1.0 - 1.0e-6, True), (1.0 + 1.0e-6, False)])
def test_comparison_is_greater_or_equal_at_the_declared_threshold(scale, refreshed):
    dz = 0.0009
    event, state, ctx = dry_event(spec=quiet_spec(dz * scale))
    out = event.step(replace(state, bed=move_mass(state.bed, ctx, (1, 1), (1, 2), dz)), 0.0, 0.1)
    assert out.record.refreshed is refreshed
    assert (out.state.topology.generation == 1) is refreshed


def test_time_alone_never_refreshes_and_no_bed_change_means_no_geometry_change():
    event, state, _ = dry_event(max_dt_s=65.0)
    result = event.evolve(state, 130.0)  # crosses MAPLE's 60 s elapsed-time trigger
    assert result.reached_target
    assert result.state.counters.n_commits >= 1 and result.state.counters.n_refreshes == 0
    assert result.state.topology is state.topology
    np.testing.assert_array_equal(host(result.state.held_source_m), host(state.held_source_m))


def test_refresh_carries_flux_soil_depth_phase_mobile_and_memory_unchanged(cpu):
    from maple_syrup.pond_hydraulics import step_once
    from maple_syrup.rainfall import constant_rainfall
    # Two genuine events, identical bed / forcing / dt / physics; only the TEST-only gate threshold differs (the default
    # 0.75 mm vs 1 cm, with MAPLE's other triggers off in both). Nothing is bypassed or faked.
    mobile = np.zeros((3, 4, 6))
    mobile[0, 0, 2] = 2e-4
    rain = 100.0 / 3.6e6

    def build(threshold):
        bed, ctx = make_bed(plane(), depth=0.003, mobile=mobile, spec=quiet_spec(threshold))
        topo = make_topology(bed, east_outlets(3, 4), drop=0.02)
        event = make_event(bed, ctx, topo, schedule=constant_rainfall(0.0, 100.0, 100.0))
        state = event.evolve(initial(event, bed), 0.4).state  # moving water: nonzero qx, phase positions, memory
        return event, ctx, state

    ev_a, ctx_a, st_a = build(THRESHOLD)
    ev_b, ctx_b, st_b = build(0.01)
    assert np.abs(host(st_a.hydraulic.qx_m2_s)).max() > 0.0
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):  # identical physical states before the manufactured move
        np.testing.assert_array_equal(host(getattr(st_a.hydraulic, name)), host(getattr(st_b.hydraulic, name)))
    s2a = replace(st_a, bed=move_mass(st_a.bed, ctx_a, (2, 3), (2, 2), 0.0009))
    s2b = replace(st_b, bed=move_mass(st_b.bed, ctx_b, (2, 3), (2, 2), 0.0009))
    out_a, out_b = ev_a.step(s2a, rain, 0.1), ev_b.step(s2b, rain, 0.1)
    assert out_a.record.refreshed and out_a.state.topology.generation == 1
    assert not out_b.record.refreshed and out_b.state.topology.generation == 0 and out_b.state.topology is s2b.topology
    # the refresh happens AFTER the physical step: every physical field is bit-identical
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        np.testing.assert_array_equal(host(getattr(out_a.state.hydraulic, name)), host(getattr(out_b.state.hydraulic, name)))
    for name in ("phase_fraction", "phase_position_m", "sediment_velocity_m_s"):
        np.testing.assert_array_equal(host(getattr(out_a.state, name)), host(getattr(out_b.state, name)))
    np.testing.assert_array_equal(host(out_a.state.bed.water.mobile_mass_by_cell_class_kg),
                                  host(out_b.state.bed.water.mobile_mass_by_cell_class_kg))
    # and equal to the plain hydraulic step on the OLD geometry: the refresh resets nothing
    expected = step_once(ev_a._check_state(s2a).pond, ev_a.rain_scale * rain, s2a.hydraulic, 0.1)
    np.testing.assert_array_equal(host(out_a.state.hydraulic.qx_m2_s), host(expected.state.qx_m2_s))
    np.testing.assert_array_equal(host(out_a.state.hydraulic.soil_water_m), host(expected.state.soil_water_m))
    assert np.abs(host(out_a.state.hydraulic.qx_m2_s)).max() > 0.0
    # both outputs are fully valid states of their own event (the full validation accepts them)
    ev_a.quiet_metrics(replace(out_a.state)), ev_b.quiet_metrics(replace(out_b.state))
