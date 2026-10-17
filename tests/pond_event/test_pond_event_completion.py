"""Quiet completion, safe dry handoff and refusals of the pond event. Nothing here was run by its author (file-only tools).

Still lake: uniform water surface `eta` over the basin terrain (depth = max(eta - z, 0)), no rain, no infiltration, so the
hydraulic state is at rest to round-off and the quiet monitor can run on ACTUAL accepted steps."""
import numpy as np
import pytest
from pond_helpers import (
    arrays_of,
    assert_unchanged,
    basin,
    host,
    initial,
    make_bed,
    make_event,
    make_topology,
)

from maple_syrup.pond_event import PondCompletionPolicy, PondEventIncomplete

LOAD = 2.0e-4


def lake(eta, *, outlet, policy=None, load=LOAD, dt=0.5):
    z = basin()
    mobile = np.zeros((3, 5, 6))
    if load:
        mobile[1, 2, 2] = load
    bed, ctx = make_bed(z, depth=np.maximum(eta - z, 0.0), mobile=mobile)
    topo = make_topology(bed, [(1, 4, 0, 1)] if outlet else [], allow_closed=not outlet)
    event = make_event(bed, ctx, topo, policy=policy, max_dt_s=dt)
    return event, initial(event, bed)


def test_closed_lake_completes_with_explicit_flow_end_settlement_and_a_recorded_dry_reset():
    policy = PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)
    event, state = lake(0.215, outlet=False, policy=policy)
    assert state.topology.n_closed_components == 1
    wet_volume = 0.25 * float(np.maximum(0.215 - basin(), 0.0).sum())
    handoff = event.complete(state, 20.0)
    dry = handoff.dry_state
    assert handoff.status == "complete" and dry.status == "dry" and dry.t_s > 1.0
    assert not np.any(host(dry.hydraulic.depth_m)) and not np.any(host(dry.hydraulic.soil_water_m))
    assert not np.any(host(dry.bed.water.mobile_mass_by_cell_class_kg))  # settled through MAPLE, nothing erased
    acc = handoff.accounting
    assert acc["surface_removed_m3"] == pytest.approx(wet_volume, rel=1e-9) and acc["soil_removed_m3"] == 0.0
    assert acc["terminal_deposition_by_class_kg"][2] == pytest.approx(LOAD, abs=1e-9)
    assert acc["final_audit"]["water"]["closes"] and acc["final_audit"]["sediment"]["closes"]
    assert acc["final_audit"]["water"]["reset_removed_m3"] == pytest.approx(wet_volume, rel=1e-9)
    assert dry.counters.n_final_commits == 1 and dry.counters.n_refreshes == 0  # a final commit never reroutes
    assert not np.any(host(dry.phase_position_m)) and np.all(host(dry.phase_fraction)[..., 0] == 1.0)


def test_policy_refuses_to_finish_with_a_significant_mobile_load():
    event, state = lake(0.215, outlet=False, policy=PondCompletionPolicy(hold_s=1.0))
    result = event.evolve(state, 10.0, stop_when_quiet=True)
    assert result.reached_target and not result.quiet_ready  # 2e-4 kg >> the MAPLE significance threshold
    before = arrays_of(result.state)
    with pytest.raises(PondEventIncomplete, match="quiet interval"):
        event.finalize(result.state)
    with pytest.raises(PondEventIncomplete, match="no quiet completion"):
        event.complete(state, 10.0)
    assert_unchanged(before, result.state)  # nothing was reset or deposited
    assert result.state.status == "wet"


def test_water_stored_below_the_escape_level_is_a_quiet_pond_with_an_outlet():
    policy = PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)
    event, state = lake(0.215, outlet=True, policy=policy)  # crest (escape) level 0.22
    assert state.topology.n_closed_components == 0 and state.topology.escape_level_m.max() == pytest.approx(0.22)
    assert not event.quiet_metrics(state)[0]  # rain is not finished yet at t = 0
    rested = event.evolve(state, 1.5).state
    ok, values = event.quiet_metrics(rested)
    assert ok and values["n_cells_above_escape"] == 0 and values["outlet_discharge_m3_s"] == 0.0
    handoff = event.complete(state, 20.0)
    assert handoff.accounting["final_audit"]["sediment"]["closes"]
    assert handoff.accounting["surface_removed_m3"] > 0.0


def test_no_dry_reset_while_water_flows_or_stands_above_its_crest():
    policy = PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)
    event, state = lake(0.225, outlet=True, policy=policy, load=0.0, dt=0.1)  # 5 mm above the 0.22 crest
    ok, values = event.quiet_metrics(state)
    assert not ok and values["n_cells_above_escape"] > 0
    moving = event.evolve(state, 1.0).state
    ok, values = event.quiet_metrics(moving)
    assert not ok  # genuine flow over the crest: discharge and/or cells above their escape level
    before = arrays_of(moving)
    with pytest.raises(PondEventIncomplete):
        event.finalize(moving)
    assert_unchanged(before, moving)
    with pytest.raises(PondEventIncomplete, match="no quiet completion"):
        event.complete(state, 1.0)  # a bound stop is incomplete, never a completion


def test_slow_drainage_is_not_labelled_quiet():
    # the water is only 1 mm above the crest: it drains slowly but it genuinely drains (depth changes every step)
    policy = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1.0e-12, deposit_mobile_at_flow_end=True)
    event, state = lake(0.221, outlet=True, policy=policy, load=0.0, dt=0.1)
    result = event.evolve(state, 3.0, stop_when_quiet=True)
    assert not result.quiet_ready and result.state.quiet.since_s is None
    assert event.audit(result.state)["water"]["export_m3"] > 0.0
