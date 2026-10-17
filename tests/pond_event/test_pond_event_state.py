"""State / API contracts of the pond event on an ACTUAL MAPLE bed. Nothing here was run by its author (file-only tools)."""
from dataclasses import replace

import numpy as np
import pytest
from pond_helpers import (
    DX,
    arrays_of,
    assert_unchanged,
    basin,
    east_outlets,
    host,
    initial,
    is_module,
    leaves,
    make_bed,
    make_event,
    make_topology,
    plane,
    wet_plane,
)

from maple_syrup.experimental_hydrology import HydraulicControl, HydraulicStepRejected
from maple_syrup.pond_bed import PondBedError
from maple_syrup.pond_event import PondEventControl, PondEventError, PondEventState
from maple_syrup.rainfall import constant_rainfall


def test_state_is_arrays_and_host_records_only(cpu):
    event, state = wet_plane(cpu)
    assert isinstance(state, PondEventState)
    for path, value in leaves(state):
        assert not is_module(value), f"module in state field {path}"
        assert not callable(value), f"callable in state field {path}"
    assert state.t_s == 0.0 and state.status == "wet"
    assert state.config_sha256 == event.config_sha256 and state.forcing_sha256 == event.forcing_sha256


def test_initial_state_rests_binds_the_held_source_and_audits_empty(cpu):
    from maple.core.types.topographic_commit import committed_elevation_m

    event, state = wet_plane(cpu)
    assert not np.any(host(state.hydraulic.qx_m2_s)) and not np.any(host(state.hydraulic.qy_m2_s))
    committed = host(committed_elevation_m(state.bed.committed_topography))
    np.testing.assert_array_equal(host(state.held_source_m), committed)
    np.testing.assert_array_equal(state.topology.source_bed_m, committed)  # initial datum: source is the DEM
    assert state.topology.generation == 0
    np.testing.assert_array_equal(state.datum.initial_full_m, state.topology.source_z_full)
    audit = event.audit(state)
    assert audit["water"]["closes"] and audit["sediment"]["closes"]
    assert audit["water"]["n_steps"] == 0 and audit["geometry"]["gate_displacement_m"] == 0.0
    assert audit["geometry"]["threshold_m"] == 0.00075 and audit["geometry"]["comparison"] == ">="


def test_one_step_is_pure_and_audits_close(cpu):
    event, state = wet_plane(cpu)
    before = arrays_of(state)
    out = event.step(state, 100.0 / 3.6e6, 0.05)
    assert_unchanged(before, state)  # the input state is never modified
    assert out.state is not state and out.state.t_s == 0.05
    assert out.state.counters.n_accepted_steps == 1 and out.record.dt_s == 0.05
    audit = event.audit(out.state)
    assert audit["water"]["closes"], audit["water"]
    assert audit["sediment"]["closes"], audit["sediment"]
    assert audit["water"]["rain_m3"] == pytest.approx(12 * DX * DX * 100.0 / 3.6e6 * 0.05, rel=1e-12)


def test_construction_and_state_refusals(cpu):
    from maple.core.parameters.avalanching import AvalanchingSpec

    bed, ctx = make_bed(plane(), depth=0.003)
    topo = make_topology(bed, east_outlets(3, 4), drop=0.02)
    with pytest.raises(PondBedError, match="avalanching"):
        make_event(bed, replace(ctx, avalanching_spec=AvalanchingSpec(enabled=True)), topo)
    with pytest.raises(PondBedError, match="constant_depth"):
        make_event(bed, replace(ctx, depth_update_rule="constant_free_surface"), topo)
    with pytest.raises(PondBedError, match="exchange_history"):
        make_event(bed, replace(ctx, record_exchange_history=True), topo)
    with pytest.raises(PondBedError, match="windowed"):
        make_event(bed, replace(ctx, window_max_journal_records=10), topo)
    with pytest.raises(PondEventError, match="donor limiter"):
        make_event(bed, ctx, topo, control=PondEventControl(hydraulic=HydraulicControl(limiter="donor")))
    with pytest.raises(PondEventError, match="vegetation"):
        make_event(bed, ctx, topo, vegetation_cover_fraction=np.full((3, 4), 1.5))
    event_a = make_event(bed, ctx, topo, schedule=constant_rainfall(0.0, 10.0, 50.0))
    event_b = make_event(bed, ctx, topo, schedule=constant_rainfall(0.0, 10.0, 60.0))
    state = initial(event_a, bed)
    with pytest.raises(PondEventError, match="configuration or forcing"):
        event_b.step(state, 0.0, 0.05)


def test_pond_default_friction_is_implicit_quadratic_and_lagged_stays_selectable(cpu):
    assert PondEventControl().hydraulic.friction_treatment == "implicit_quadratic"
    assert HydraulicControl().friction_treatment == "lagged"  # global / legacy default unchanged
    event, state = wet_plane(cpu)
    rt = event._check_state(state)
    assert rt.pond.control.friction_treatment == "implicit_quadratic"
    lagged = PondEventControl(hydraulic=HydraulicControl())  # explicit original qualified variant
    assert lagged.hydraulic.friction_treatment == "lagged"
    old_event, _ = wet_plane(cpu, control=lagged)
    assert old_event._check_state(initial(old_event, make_bed(plane(), depth=0.003)[0])).pond.control.friction_treatment == "lagged"
    assert old_event.config_sha256 != event.config_sha256  # the treatment is part of the configuration identity


def test_hydraulic_rejection_leaves_the_state_unchanged(cpu):
    bed, ctx = make_bed(basin(), depth=np.maximum(0.225 - basin(), 0.0))
    topo = make_topology(bed, [(1, 4, 0, 1)])
    event = make_event(bed, ctx, topo)
    state = initial(event, bed)
    before = arrays_of(state)
    with pytest.raises(HydraulicStepRejected):
        event.step(state, 0.0, 5.0)  # gravity-wave CFL ~ 9 on a 4.5 cm deep pond
    assert_unchanged(before, state)
    assert state.t_s == 0.0 and state.counters.n_accepted_steps == 0


def test_a_bed_with_a_different_terrain_shape_is_refused(cpu):
    bed, ctx = make_bed(plane(), depth=0.003)
    topo = make_topology(bed, east_outlets(3, 4), drop=0.02)
    event = make_event(bed, ctx, topo)
    other_bed, _ = make_bed(plane(drop=0.03), depth=0.003)  # not a constant datum offset of the topology's bed
    with pytest.raises(PondEventError, match="constant datum"):
        event.initial_state(other_bed, np.zeros((3, 4)))
    shifted, _ = make_bed(plane(base=0.21), depth=0.003)  # a constant datum shift IS accepted
    assert event.initial_state(shifted, np.zeros((3, 4))).topology.generation == 0
