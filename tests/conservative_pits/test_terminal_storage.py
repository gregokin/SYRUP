"""CW-C2: opt-in conservative terminal storage (strict pits, flat sinks, masked nodata) in the coupled event
and the explicit terminal-water completion option, on small REAL MAPLE beds.

CPU tests (always run): strict defaults unchanged; the transport network recognises only valid opt-in pits and
refuses forged encodings; a donor chain into a pit and a flat basin deposit every incoming grain through MAPLE
with no export and closed class / water budgets, inputs unchanged; completion needs the explicit option (strict
policy stays incomplete) and accounts the ponded water removed by the external dry reset; a reactivated moving
flow refuses the handoff; a terrain commit that moves the pit is recorded and rebuilt, a strict case still
refuses the new sink; masked nodata is never a receiver, the datum check uses the active terrain only, and
avalanching across a mask is refused.

GPU tests (skip honestly without a device): the pit-chain event and its terminal-water completion on CUDA match
the host array event at the declared bounds.

Nothing here was run by its author (file-only tools); the root records the executed results.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("maple")

from pit_helpers import NODATA, masked_inputs, masked_two_column, pit_graph, pit_inputs
from test_complete_backend import compare_handoffs, fixture_xp
from test_coupled_backends import (
    SED_ATOL,
    SED_RTOL,
    H,
    assert_closed_any,
    compare_any,
    compare_events,
    control_for,
    snapshot_any,
)
from test_sediment_event import chain_elevation, make_bed, valley_elevation, within

from maple_syrup import complete_event as module
from maple_syrup.complete_event import (
    CompletionPolicy,
    DryHandoff,
    IncompleteEventError,
    QuietProgress,
    complete_event,
    finalize_event,
    quiet_metrics,
)
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.routing import (
    INACTIVE,
    PIT_STORAGE,
    RoutingGraphError,
    build_routing_graph,
)
from maple_syrup.sediment_bed import (
    BedIntegrationError,
    TerrainReference,
    apply_bed_demand,
    bed_inventory,
    commit_bed,
    graph_policy_flags,
    refresh_routing,
    terrain_reference,
    with_water,
)
from maple_syrup.sediment_event import (
    SedimentEventControl,
    SedimentEventError,
    evolve_sediment_event,
    sediment_coupled_step,
)
from maple_syrup.sediment_physics import (
    SedimentPhysicsError,
    physics_grid,
    physics_grid_from_graph,
)
from maple_syrup.sediment_transport import (
    TransportError,
    transport_network,
    transport_step,
)
from maple_syrup.storm import StormControl


def run(inputs, schedule, end, control=None, *, cadence=10.0, hydraulics=None):
    control = control or SedimentEventControl()
    return evolve_sediment_event(inputs["state0"], inputs["ctx"], inputs["column"], inputs["field"], schedule,
                                 inputs["vegetation"], inputs["sediment"], end, control, report_every_s=cadence,
                                 hydraulics=hydraulics)


def complete(inputs, schedule, *, max_end_s, policy, control=None, **kw):
    control = control or SedimentEventControl(storm=StormControl(max_dt_s=1.0))
    return complete_event(inputs["state0"], inputs["ctx"], inputs["column"], inputs["field"], schedule,
                          inputs["vegetation"], inputs["sediment"], max_end_s=max_end_s, control=control,
                          policy=policy, report_every_s=2.0, **kw)


# =====================================================================================================================
# strict defaults
# =====================================================================================================================
def test_strict_defaults_carry_no_terminal_storage_and_their_objects_are_unchanged():
    from test_coupled_backends import graph_for_xp, setup_xp

    bed, ctx = make_bed(valley_elevation(4, 5))
    graph, terrain = graph_for_xp(bed, np, south_ring=valley_elevation(5, 5)[0] - 0.01)
    network = transport_network(graph)
    assert network.terminal_flat is None and network.terminal_host is None and network.n_terminal == 0
    assert physics_grid_from_graph(graph).terminal is None and physics_grid(graph.slope, graph.active, 0.25).terminal is None
    assert terrain.strict and terrain.build_options() == {"nodata_value": None, "allow_masked_nodata": False,
                                                          "allow_pit_storage": False, "allow_flat_storage": False}
    assert TerrainReference.__dataclass_fields__["allow_pit_storage"].default is False
    assert graph_policy_flags("strict") == {"allow_masked_nodata": False, "allow_pit_storage": False, "allow_flat_storage": False}
    assert CompletionPolicy().terminal_storage_water is False and CompletionPolicy().validated()
    with pytest.raises(Exception, match="terminal_storage_water"):
        CompletionPolicy(terminal_storage_water="yes").validated()
    # the strict event still runs on the unchanged objects and its completion values carry no terminal keys
    inputs = setup_xp(bed, ctx, np, south_ring=valley_elevation(5, 5)[0] - 0.01)
    r = run(inputs, constant_rainfall(0.0, 20.0, 72.0), 30.0, cadence=30.0)
    assert_closed_any(r, inputs["state0"])
    _q, values = quiet_metrics(r.state, ctx, CompletionPolicy(), 20.0)
    assert set(values) == {"max_depth_m", "surface_volume_m3", "max_unit_discharge_m2_s", "outlet_discharge_m3_s",
                           "max_mobile_cell_class_kg", "mobile_by_class_kg", "mobile_per_element_limit_kg",
                           "rainfall_finished"}
    # the option without any terminal cell changes nothing either
    _q2, values2 = quiet_metrics(r.state, ctx, CompletionPolicy(terminal_storage_water=True), 20.0)
    assert values2 == values


def test_quiescent_strict_completion_and_zero_reset_are_unchanged():
    args, kw = fixture_xp(np)
    out = complete_event(*args, **kw)
    assert isinstance(out, DryHandoff) and out.dry_state.t_s == 4
    assert "terminal_storage_removed_m3" not in out.accounting and "terminal_storage_cells" not in out.accounting["completion_after_terminal"]
    assert out.accounting["surface_removed_m3"] == 0 and out.accounting["combined_water_residual_m3"] == 0


# =====================================================================================================================
# network recognition and refusals
# =====================================================================================================================
def test_network_recognises_only_valid_opt_in_pits_and_refuses_forged_encodings():
    inputs = pit_inputs(np)
    graph = inputs["graph"]
    pit = np.asarray(graph.pit_storage)
    assert pit.sum() == 1 and pit[0, 0] and graph.receiver[0, 0] == PIT_STORAGE and graph.outlet.sum() == 0
    network = transport_network(graph)
    assert network.n_terminal == 1 and network.terminal_host[0, 0] and bool(network.terminal_flat[0])
    assert int(network.receiver_index[0]) == 0  # the pit receives itself: terminal
    assert int(network.receiver_index[graph.shape[1]]) == 0  # its donor (row 1) routes into it
    assert 0 not in set(H(network.cells_x).tolist()) and 0 not in set(H(network.cells_y).tolist())  # no out-face
    assert physics_grid_from_graph(graph).terminal is not None and bool(H(physics_grid_from_graph(graph).terminal)[0, 0])
    assert graph_policy_flags(graph.policy) == {"allow_masked_nodata": False, "allow_pit_storage": True, "allow_flat_storage": False}
    assert inputs["terrain"].allow_pit_storage and not inputs["terrain"].strict

    def forged(**changes):
        return dataclasses.replace(graph, **changes)

    with pytest.raises(TransportError, match="disagree"):
        transport_network(forged(pit_storage=np.zeros_like(pit)))  # receiver says pit, mask says no
    with pytest.raises(TransportError, match="policy"):
        transport_network(forged(policy="strict"))
    aspect = np.array(graph.aspect, copy=True)
    aspect[0, 0] = 1
    with pytest.raises(TransportError, match="aspect 0"):
        transport_network(forged(aspect=aspect))
    outlet = np.array(graph.outlet, copy=True)
    outlet[0, 0] = True
    with pytest.raises(TransportError, match="not an outlet"):
        transport_network(forged(outlet=outlet))
    slope = np.array(graph.slope, copy=True)
    slope[0, 0] = 0.01
    with pytest.raises(TransportError, match="zero slope"):
        transport_network(forged(slope=slope))
    # a strict graph whose receiver is forged to PIT_STORAGE is refused (no silent missing receiver)
    from test_coupled_backends import graph_for_xp

    strict, _ = graph_for_xp(make_bed(chain_elevation(4))[0], np)
    receiver = np.array(strict.receiver, copy=True)
    receiver[1, 0] = PIT_STORAGE
    with pytest.raises(TransportError, match="disagree|policy"):
        transport_network(dataclasses.replace(strict, receiver=receiver))
    receiver[1, 0] = -7
    with pytest.raises(TransportError, match="no receiver|disagree|policy"):
        transport_network(dataclasses.replace(strict, receiver=receiver))
    # the physics grid refuses a terminal mask on a sloping or inactive cell
    with pytest.raises(SedimentPhysicsError, match="zero slope"):
        physics_grid(strict.slope, strict.active, 0.25, terminal=np.ones(strict.shape, dtype=bool))
    # the transport operator refuses a caller mask that would keep terminal mass mobile
    nc = 6
    shape3 = (*graph.shape, nc)
    settle = np.ones(shape3, dtype=bool)
    settle[0, 0] = False
    with pytest.raises(TransportError, match="settle cells"):
        transport_step(network, np.zeros(shape3), np.zeros(shape3), np.zeros(shape3), settle, 1.0)


@pytest.mark.parametrize("implementation", ["array", "numba"])
def test_characteristic_kernel_requires_every_terminal_class_to_settle_and_deposits_the_terminal_load(implementation):
    """Same rule as the upwind operator, for both characteristic implementations and the current class count:
    a mask leaving ANY class mobile at a terminal cell is refused before the kernel runs; with the settle mask
    the terminal cell's resident load and its donor's arrivals are deposition requests, nothing crosses out of
    it and nothing is exported."""
    from maple_syrup.characteristic_transport import (
        PhaseState,
        characteristic_step,
        validate_phase_state,
    )

    if implementation == "numba":
        from maple_syrup.routing_numba import numba_available

        if not numba_available():
            pytest.skip("Numba not installed; compiled characteristic kernel not exercised, no claim made")
    inputs = pit_inputs(np)
    network = inputs["state0"].network
    ny, nx = network.shape
    nc, nb = 6, 8
    shape3 = (ny, nx, nc)
    dx = network.dx_m
    # canonical phase everywhere except the donor's resident packets, placed 0.05 m short of its downstream
    # face (bin 7 of 8) so they cross into the pit during the first substep at v = 0.1 m/s, dt_sub = 0.5 s
    fraction = np.zeros((ny, nx, nc, nb))
    fraction[..., 0] = 1.0
    position = np.zeros((ny, nx, nc, nb))
    fraction[1, 0, :, 0], fraction[1, 0, :, 7] = 0.0, 1.0
    position[1, 0, :, 7] = dx - 0.05
    phase = PhaseState(fraction=fraction, position_m=position, dx_m=dx, n_bins=nb, xp=np)
    mobile = np.zeros(shape3)
    mobile[0, 0, :] = 1e-3  # resident load at the pit
    mobile[1, 0, :] = 2e-3  # resident load at its donor
    pickup = np.zeros(shape3)
    pickup[1, 0, 2] = 5e-4  # injected at the donor's upstream face: stays in the donor this step
    velocity = np.full(shape3, 0.1)
    velocity[0, 0] = 0.0  # zero conveyance at the pit
    rate = np.full(shape3, 1.0 / 2.0)
    settle = np.zeros(shape3, dtype=bool)
    settle[0, 0] = True
    validate_phase_state(phase, mobile, network)
    step = characteristic_step(network, mobile, phase, pickup, velocity, rate, settle, 1.0, n_substeps=2,
                               courant_max=0.5, implementation=implementation)
    t = step.transport
    assert not np.any(t.export_request_kg) and not np.any(t.internal_transfer_out_kg[0, 0])
    assert np.all(t.internal_transfer_in_kg[0, 0] > 0.0)  # the donor's crossings arrive at the pit, every class
    within(t.deposition_request_kg[0, 0] - (mobile[0, 0] + t.internal_transfer_in_kg[0, 0]),
           np.asarray(t.budget_tolerance_by_class_kg))
    assert not np.any(step.mobile_remaining_kg[0, 0])  # nothing stays mobile at the pit
    within(t.budget_residual_by_class_kg, t.budget_tolerance_by_class_kg)
    for klass in range(nc):  # any single class left mobile at the terminal cell is refused, before any kernel run
        partial = settle.copy()
        partial[0, 0, klass] = False
        with pytest.raises(TransportError, match="settle cells for every class"):
            characteristic_step(network, mobile, phase, pickup, velocity, rate, partial, 1.0, n_substeps=2,
                                courant_max=0.5, implementation=implementation)


def test_physics_grid_terminal_mask_must_match_the_network_before_any_work(monkeypatch):
    """Adversarial event states: a pit state whose physics grid lost its terminal mask (or carries a wrong one),
    and a strict state given a terminal mask, are refused by the solver's input check before any kernel or
    MAPLE call; the caller's arrays are untouched. The strict pair (None, None) stays valid."""
    from maple_syrup import sediment_event

    def boom(*a, **k):
        raise AssertionError("a kernel or MAPLE call ran after a refused grid/network binding")

    monkeypatch.setattr(sediment_event, "apply_bed_demand", boom)
    monkeypatch.setattr(sediment_event, "coupled_step", boom)
    schedule = constant_rainfall(0.0, 20.0, 72.0)
    pit = pit_inputs(np)
    state0 = pit["state0"]
    saved = snapshot_any(state0)
    assert state0.grid.terminal is not None and state0.network.terminal_flat is not None
    lost = replace(state0, grid=replace(state0.grid, terminal=None))
    with pytest.raises(SedimentEventError, match="terminal-storage mask"):
        run({**pit, "state0": lost}, schedule, 10.0)
    wrong = np.array(state0.grid.terminal, copy=True)
    wrong[0, 0], wrong[2, 0] = False, True
    with pytest.raises(SedimentEventError, match="differs from the transport network"):
        run({**pit, "state0": replace(state0, grid=replace(state0.grid, terminal=wrong))}, schedule, 10.0)
    with pytest.raises(SedimentEventError, match="bool"):
        run({**pit, "state0": replace(state0, grid=replace(state0.grid, terminal=wrong.astype(np.float64)))}, schedule, 10.0)
    compare_any(state0, saved)
    # the direct single-step entry: the kernel guard refuses the lost mask after the (pure, unpublished) pickup
    monkeypatch.undo()
    rate = pit["field"].apply(72.0 / 3.6e6)
    with pytest.raises(TransportError, match="settle cells"):
        sediment_coupled_step(lost, pit["ctx"], pit["column"], rate, pit["vegetation"], pit["sediment"], 1.0,
                              SedimentEventControl())
    compare_any(state0, saved)
    # a strict state given a terminal mask is refused; the strict (None, None) pair is valid
    from test_coupled_backends import setup_xp

    bed, ctx = make_bed(valley_elevation(4, 5))
    valley = setup_xp(bed, ctx, np, south_ring=valley_elevation(5, 5)[0] - 0.01)
    vs = valley["state0"]
    assert vs.grid.terminal is None and vs.network.terminal_flat is None
    injected = replace(vs, grid=replace(vs.grid, terminal=np.zeros(vs.graph.shape, dtype=bool)))
    with pytest.raises(SedimentEventError, match="terminal-storage mask"):
        run({**valley, "state0": injected}, schedule, 10.0)
    r = run(valley, schedule, 10.0, cadence=10.0)
    assert_closed_any(r, vs)


# =====================================================================================================================
# conservative transport into terminal storage
# =====================================================================================================================
def terminal_cells_of(graph):
    """The `(row, col)` set of a graph's terminal-storage cells (host mask)."""
    pit = np.asarray(graph.pit_storage, dtype=bool)
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(pit), strict=True)}


def assert_terminal_event(r, inputs, *, terminal_cells, donor_fed=None, final_terminal=None):
    """Closure, no export anywhere, inputs unchanged, and the terminal-cell claims over the interval in which
    `terminal_cells` WERE terminal (the whole run when no commit rerouted them; `final_terminal`, default the
    same cells, is the terminal set of the CURRENT graph after the run's commits, which the evolving terrain may
    legitimately have changed).

    `donor_fed` names the terminal cells that have at least one donor cell: there the deposition must be
    positive and exceed the cell's own pickup (the donors' load). A terminal cell WITHOUT a donor only returns
    its own pickup, and at zero slope the raindrop law's slope term `(100 S)^c` is exactly zero (every Plot 1
    `raindrop_c` is positive) while flow detachment needs shear: such a cell legitimately picks up and
    deposits NOTHING, and local conservation (pickup = deposition, no bed change) is asserted instead of a
    mistaken positivity. The empty-mobile and ponding claims hold for the cells terminal in the current graph."""
    state0 = inputs["state0"]
    closure, bc = assert_closed_any(r, state0, check_dry_settling=True)
    assert np.all(bc["export_actual"] == 0.0) and np.all(bc["export_requested"] == 0.0)
    assert not np.any(H(r.cumulative_export_request_kg)) and float(H(r.cumulative_export_m3)) == 0.0
    assert bc["actual_pickup"].sum() > 0.0 and bc["deposition_actual"].sum() > 0.0
    mobile = H(r.state.bed.water.mobile_mass_by_cell_class_kg)
    deposition, pickup = H(r.cumulative_deposition_kg), H(r.cumulative_pickup_kg)
    change = H(r.bed_change_by_cell_class_kg())
    tolerance = np.asarray(closure["tolerance_kg"])
    fed = set(terminal_cells if donor_fed is None else donor_fed)
    current = set(terminal_cells) if final_terminal is None else set(final_terminal)
    assert terminal_cells_of(r.state.graph) == current and r.state.network.n_terminal == len(current)
    assert r.state.grid.terminal is not None and set(zip(*np.nonzero(H(r.state.grid.terminal)), strict=True)) == current
    for cell in terminal_cells:
        assert np.all(deposition[cell] >= pickup[cell] * (1 - 1e-12))  # its own pickup always returned
        if cell in fed:
            assert deposition[cell].sum() > pickup[cell].sum()  # a genuine donor load arrived
        elif cell in current:
            # actual local conservation of a still-terminal cell: whatever arrived (possibly nothing) was deposited
            # through MAPLE, so its bed change is deposition - pickup; no claim that nothing ever arrived
            within(change[cell] - (deposition[cell] - pickup[cell]), tolerance)
    depth = H(r.state.storm.depth_m)
    for cell in current:
        assert not np.any(mobile[cell])  # settled through MAPLE every step while terminal
        assert depth[cell] > 0.0  # water stays in the domain: the terminal cells pond
    # total deposition + remaining mobile = total pickup (no export) within the unchanged MAPLE closure bound
    within(bc["deposition_actual"] + H(r.state.bed.water.mobile_mass_by_cell_class_kg).sum(axis=(0, 1))
           - bc["actual_pickup"], np.asarray(closure["tolerance_kg"]))
    return closure, bc


def test_donor_chain_into_a_pit_deposits_everything_through_maple_with_no_export():
    inputs = pit_inputs(np)
    state0 = inputs["state0"]
    saved = snapshot_any(state0)
    r = run(inputs, constant_rainfall(0.0, 20.0, 72.0), 60.0, cadence=20.0)
    _closure, bc = assert_terminal_event(r, inputs, terminal_cells=[(0, 0)])
    deposition, pickup = H(r.cumulative_deposition_kg), H(r.cumulative_pickup_kg)
    assert np.all(pickup[1:].sum(axis=-1) > 0.0)  # the donors erode
    assert deposition[0, 0].sum() > pickup[0, 0].sum()  # the pit receives its donors' load
    depth = H(r.state.storm.depth_m)
    assert depth[0, 0] > depth[1:].max()  # the pit ponds
    change = H(r.bed_change_by_cell_class_kg())
    assert change[0, 0].sum() > 0.0  # the pit's bed grew
    assert int(H(r.regime_cell_steps["diffuse"])) > 0
    compare_any(state0, saved)
    assert bc["holdings_shortfall"].sum() >= 0.0 and r.n_commits >= 1


def test_flat_basin_initial_transport_before_any_commit_keeps_both_flats_terminal():
    """The flat-basin claims (no flow between equal-elevation neighbours, the fed flat deposits its donor's load,
    the unfed flat picks up and deposits nothing at zero slope) hold exactly while the INITIAL graph is in force:
    with commits disabled and no forced final commit the terrain is never refreshed, both flats stay terminal
    for the whole window and the deposited mass sits in the pending ledger of the uncommitted bed."""
    inputs = pit_inputs(np, flat=True)
    graph = inputs["graph"]
    pit = np.asarray(graph.pit_storage)
    assert pit[0, 0] and pit[1, 0] and pit.sum() == 2 and graph.flat_storage is not None and graph.flat_storage.sum() == 2
    control = SedimentEventControl(commit=False, force_final_commit=False)
    r = run(inputs, constant_rainfall(0.0, 20.0, 72.0), 60.0, control, cadence=20.0)
    assert r.n_commits == 0 and r.state.graph is graph  # the initial graph was in force throughout
    assert_terminal_event(r, inputs, terminal_cells=[(0, 0), (1, 0)], donor_fed=[(1, 0)])
    deposition, pickup = H(r.cumulative_deposition_kg), H(r.cumulative_pickup_kg)
    # nothing flows between equal-elevation neighbours: cell 0 has no donor; at zero slope the raindrop law's
    # slope term is zero and there is no shear, so it picks up and deposits exactly nothing (intended law, not a
    # clipped value): local conservation holds trivially and its bed is unchanged
    assert pickup[0, 0].sum() == 0.0 and deposition[0, 0].sum() == 0.0
    assert not np.any(H(r.bed_change_by_cell_class_kg())[0, 0])
    # the fed flat is also at zero slope (no own pickup) yet deposits its donor's load: cell 2's load arrived
    assert pickup[1, 0].sum() == 0.0 and deposition[1, 0].sum() > 0.0
    assert pickup[2:].sum() > 0.0 and H(r.state.storm.depth_m)[0, 0] > 0.0  # the slope erodes; the flat still ponds
    # the real deposition is in the bed and recorded as pending (uncommitted) change; the committed terrain is the initial one
    assert H(r.bed_change_by_cell_class_kg())[1, 0].sum() > 0.0
    assert np.any(H(r.state.bed.ledger.pending_bed_mass_change_kg)[1, 0] != 0.0)
    assert r.state.bed.committed_topography.commit_count == 0
    with pytest.raises(RoutingGraphError, match="flat-sink storage"):
        pit_graph(inputs["bed"], np, flat=False)  # the flat basin needs the explicit flat option


def test_flat_basin_evolving_terrain_fills_the_fed_flat_and_reroutes_it_without_erasing_mass_or_water():
    """CW-E47C1: rerouting needs the NET elevation threshold (CW-B104); the original weak storm is now the
    below-threshold negative control and a stronger physical storm makes the genuine reroute.

    With genuine MAPLE commits the fed flat (1, 0) receives its donor's load, its committed bed rises above its
    equal neighbour, the refreshed graph routes it INTO (0, 0): terminal storage goes 2 -> 1 (intended evolving
    physics, not frozen terrain). Nothing is erased across the reroute: class and water budgets close with zero
    export, every inventory (bed, mobile, water) is accounted before and after, the mass deposited at (1, 0)
    stays in its bed, and the phase partition is kept and conserved through the direction change. Whether any
    load reaches (0, 0) within the window is NOT asserted: the rerouted cell's new slope is tiny and the grain
    velocity / travel distance physics decide that, not the reroute itself."""
    from maple.core.types.topographic_commit import committed_elevation_m

    from maple_syrup.characteristic_transport import (
        fraction_sum_tolerance,
        validate_phase_state,
    )

    # CW-E47C1 NEGATIVE control (the original 40 s / 72 mm/h storm): genuine accounting commits publish the fed flat's
    # deposition, but its net displacement (~3.8e-5 m) stays below MAPLE's net threshold (default 0.00075 m), so the
    # routing is NOT refreshed: both flats stay terminal and nothing is erased
    below_in = pit_inputs(np, flat=True)
    assert below_in["ctx"].topographic_commit_spec.pending_elevation_trigger_m == 0.00075  # MAPLE's default, unchanged
    below = run(below_in, constant_rainfall(0.0, 40.0, 72.0), 120.0, cadence=40.0)
    assert below.n_commits >= 1 and below.n_reroutes == 0 and below.n_graph_changes == 0
    assert all(not e["rerouted"] and e["net_displacement_m"] < e["reroute_threshold_m"] for e in below.commit_log)
    assert below.state.graph is below_in["graph"] and terminal_cells_of(below.state.graph) == {(0, 0), (1, 0)}
    assert_terminal_event(below, below_in, terminal_cells=[(0, 0), (1, 0)], donor_fed=[(1, 0)])
    assert H(below.bed_change_by_cell_class_kg())[1, 0].sum() > 0.0  # committed, kept, not rerouted
    # the GENUINE evolving case: a stronger physical storm (300 mm/h for 120 s, run to 180 s) fills the fed flat
    # until the post-commit net displacement reaches the default threshold, and the refreshed graph routes it into
    # (0, 0)
    inputs = pit_inputs(np, flat=True)
    graph = inputs["graph"]
    state0 = inputs["state0"]
    saved = snapshot_any(state0)
    assert terminal_cells_of(graph) == {(0, 0), (1, 0)}
    initial_bed = H(bed_inventory(state0.bed)).copy()
    initial_mobile = H(state0.bed.water.mobile_mass_by_cell_class_kg).sum(axis=(0, 1))
    storm = constant_rainfall(0.0, 120.0, 300.0)
    full = run(inputs, storm, 180.0, cadence=40.0)
    rerouted = [e for e in full.commit_log if e["rerouted"]]
    assert rerouted and full.n_reroutes == len(rerouted) and full.n_commits > len(rerouted)
    assert all(e["net_displacement_m"] >= e["reroute_threshold_m"] == 0.00075 for e in rerouted)
    assert all(e["net_displacement_m"] < e["reroute_threshold_m"] for e in full.commit_log if not e["rerouted"])
    assert rerouted[0]["terminal_storage_change"]["n_terminal_removed"] == 1  # the first reroute drained the filled flat
    # over the WHOLE run (the filled flat then drains as a moving cell): budgets close, no export, the terrain keeps
    # (0, 0) as the only terminal cell, and the phase partition is valid on the refreshed network
    assert_closed_any(full, state0, check_dry_settling=True)
    assert terminal_cells_of(full.state.graph) == {(0, 0)} and float(H(full.cumulative_export_m3)) == 0.0
    validate_phase_state(full.state.phase, full.state.bed.water.mobile_mass_by_cell_class_kg, full.state.network)
    # the terminal-cell claims hold over the interval in which (1, 0) WAS terminal: the same storm ended at the
    # genuine reroute (after it, (1, 0) is a moving cell and legitimately erodes, so cumulative per-cell
    # "deposition >= own pickup" no longer describes a terminal cell)
    t_reroute = rerouted[0]["t_s"]
    r = run(inputs, storm, t_reroute, cadence=40.0)
    assert r.n_reroutes == 1
    assert [e["t_s"] for e in r.commit_log if e["rerouted"]] == [t_reroute]
    assert r.n_commits >= 1 and r.n_graph_changes >= 1 and r.rerouted_cells_total >= 1
    assert terminal_cells_of(r.state.graph) == {(0, 0)}  # 2 -> 1: the filled flat now drains into (0, 0)
    assert r.state.graph.receiver[1, 0] == 0 and r.state.graph.aspect[1, 0] != 0 and r.state.graph.slope[1, 0] > 0.0
    closure, bc = assert_terminal_event(r, inputs, terminal_cells=[(0, 0), (1, 0)], donor_fed=[(1, 0)],
                                        final_terminal=[(0, 0)])
    deposition, change = H(r.cumulative_deposition_kg), H(r.bed_change_by_cell_class_kg())
    assert change[1, 0].sum() > 0.0 and deposition[1, 0].sum() > 0.0  # the filled flat keeps what it received
    assert H(r.state.storm.depth_m)[0, 0] > 0.0  # the remaining terminal cell ponds; no water left the domain
    assert float(H(r.cumulative_export_m3)) == 0.0
    z = H(committed_elevation_m(r.state.bed.committed_topography))
    assert z[1, 0] > z[0, 0]  # the committed terrain itself explains the reroute
    # every inventory accounted across the reroute: bed + mobile before = bed + mobile after (no export) within
    # the unchanged MAPLE closure bound; the ledger's recognised commits equal the actual net bed change
    final_bed = H(bed_inventory(r.state.bed))
    final_mobile = H(r.state.bed.water.mobile_mass_by_cell_class_kg).sum(axis=(0, 1))
    within((final_bed + final_mobile) - (initial_bed + initial_mobile), np.asarray(closure["tolerance_kg"]))
    within(H(r.committed_net_bed_change_kg) - np.asarray(closure["net_bed_change_kg"]), np.asarray(closure["tolerance_kg"]))
    assert bc["export_actual"].sum() == 0.0 and closure["closed"]
    # the phase partition survived the direction change: it still partitions the ACTUAL mobile pool on the
    # refreshed network, fractions sum to 1 and positions stay inside the cell; steered cells are a subset of the
    # rerouted ones
    phase = r.state.phase
    validate_phase_state(phase, r.state.bed.water.mobile_mass_by_cell_class_kg, r.state.network)
    assert np.all(np.abs(H(phase.fraction).sum(axis=-1) - 1.0) <= fraction_sum_tolerance(phase.n_bins))
    assert np.all(H(phase.position_m) < phase.dx_m) and 0 <= r.phase_steered_cells_total <= r.rerouted_cells_total
    assert r.n_phase_canonicalized >= 0 and float(H(r.max_phase_reconciliation_residual_kg)) >= 0.0
    compare_any(state0, saved)
    # completion honours the policy on the EVOLVED graph: one terminal cell (0, 0) with its ponded water; the
    # rerouted flat (1, 0) is now a moving cell and must drain like the slope
    policy = CompletionPolicy(hold_s=10.0, terminal_storage_water=True)
    rain = constant_rainfall(0.0, 10.0, 100.0)
    strict_inputs = pit_inputs(np, flat=True, ksat=1e-5, pit_ksat=0.0, pit_ksat_cells=[(0, 0)])
    with pytest.raises(IncompleteEventError, match="duration"):
        complete(strict_inputs, rain, max_end_s=900.0, policy=CompletionPolicy(hold_s=10.0))
    explicit = pit_inputs(np, flat=True, ksat=1e-5, pit_ksat=0.0, pit_ksat_cells=[(0, 0)])
    out = complete(explicit, rain, max_end_s=900.0, policy=policy)
    assert isinstance(out, DryHandoff)
    final_terminal = terminal_cells_of(out.dry_state.graph)
    assert final_terminal in ({(0, 0)}, {(0, 0), (1, 0)})  # the reroute depends on how much the short storm deposited
    acc = out.accounting
    assert acc["terminal_storage_cells"] == len(final_terminal) and acc["terminal_storage_removed_m3"] > 0.0
    assert acc["surface_removed_m3"] == pytest.approx(acc["terminal_storage_removed_m3"] + acc["moving_surface_removed_m3"], rel=0, abs=1e-15)
    assert acc["moving_surface_removed_m3"] <= policy.surface_volume_m3 and acc["sediment_closure"]["closed"]
    assert abs(acc["combined_water_residual_m3"]) <= acc["storm_water_budget"]["tolerance_m3"]
    assert not np.any(H(out.dry_state.storm.depth_m)) and not np.any(H(out.dry_state.bed.water.mobile_mass_by_cell_class_kg))


# =====================================================================================================================
# completion with ponded terminal water
# =====================================================================================================================
def test_completion_needs_the_explicit_terminal_water_option_and_accounts_the_removed_pond():
    schedule = constant_rainfall(0.0, 10.0, 100.0)
    strict = pit_inputs(np, ksat=1e-5, pit_ksat=0.0)  # the pond never infiltrates
    with pytest.raises(IncompleteEventError, match="duration"):
        complete(strict, schedule, max_end_s=900.0, policy=CompletionPolicy(hold_s=10.0))
    explicit = pit_inputs(np, ksat=1e-5, pit_ksat=0.0)
    state0 = explicit["state0"]
    saved = snapshot_any(state0)
    policy = CompletionPolicy(hold_s=10.0, terminal_storage_water=True)
    out = complete(explicit, schedule, max_end_s=900.0, policy=policy)
    assert isinstance(out, DryHandoff)
    acc = out.accounting
    assert acc["terminal_storage_cells"] == 1 and acc["terminal_storage_removed_m3"] > 0.0
    assert acc["moving_surface_removed_m3"] <= policy.surface_volume_m3
    assert acc["surface_removed_m3"] == pytest.approx(acc["terminal_storage_removed_m3"] + acc["moving_surface_removed_m3"], rel=0, abs=1e-15)
    assert acc["surface_removed_m3"] == acc["storm_water_budget"]["surface_final_m3"]  # full physical storage
    after = acc["completion_after_terminal"]
    assert after["terminal_storage_water_excluded"] and after["moving_max_depth_m"] <= policy.max_depth_m
    assert after["max_depth_m"] > policy.max_depth_m  # the pond itself is still reported as depth
    assert abs(acc["combined_water_residual_m3"]) <= acc["storm_water_budget"]["tolerance_m3"]
    assert acc["sediment_closure"]["closed"] and not np.any(H(out.dry_state.storm.depth_m))
    assert H(out.pre_reset_surface_m)[0, 0] > 0.0 and not np.any(H(out.dry_state.bed.water.mobile_mass_by_cell_class_kg))
    np.testing.assert_array_equal(H(bed_inventory(out.dry_state.bed)) >= 0.0, True)
    compare_any(state0, saved)
    # the hold is measured on the moving cells only after rain finished; the strict quiet metric never qualified
    assert not quiet_metrics(out.progress.result.state, explicit["ctx"], CompletionPolicy(hold_s=10.0), 10.0)[0]
    assert quiet_metrics(out.progress.result.state, explicit["ctx"], policy, 10.0)[0]


def _ready_for_terminal(inputs, policy):
    """Pause at the FIRST boundary where the moving cells qualify under the explicit option, then mark the quiet hold
    complete: the input of `finalize_event`."""
    ctx = inputs["ctx"]
    paused = complete(inputs, constant_rainfall(0.0, 10.0, 100.0), max_end_s=900.0, policy=policy,
                      checkpoint_callback=lambda p: quiet_metrics(p.result.state, ctx, policy, 10.0)[0])
    assert paused.status == "wet" and quiet_metrics(paused.result.state, ctx, policy, 10.0)[0]
    return replace(paused, rainfall_end_s=10.0,
                   quiet=QuietProgress(since_s=paused.result.state.t_s - 20.0, last_t_s=paused.result.state.t_s, ready=True))


def _record_final_commit(monkeypatch, calls, transform=None):
    """Transparent wrapper on the ACTUAL commit `complete_event.finalize_event` calls (`_commit_and_reroute`), and on
    its post-commit quiet check: records that the forced final commit executed and that the quiet gate was then
    evaluated on that commit's bed. `transform` (refusal test) maps the real commit outcome."""
    original_commit, original_quiet = module._commit_and_reroute, module.quiet_metrics

    def commit(*a, **k):
        result = original_commit(*a, **k)
        calls.append(("commit", k.get("force"), None if result is None else result.record["commit_reason"],
                      None if result is None else result.rerouted, result))
        return result if transform is None else transform(result)

    def quiet(state, *a, **k):
        out = original_quiet(state, *a, **k)
        calls.append(("quiet", state.bed, out[0]))
        return out
    monkeypatch.setattr(module, "_commit_and_reroute", commit)
    monkeypatch.setattr(module, "quiet_metrics", quiet)


def test_terminal_forced_commit_executes_and_the_post_commit_quiet_gate_is_evaluated(monkeypatch):
    """CW-E47C1 positive path: the forced final commit is a genuine MAPLE accounting commit ('final_forced'); below the
    net threshold it does not reroute, so (contract of `_commit_and_reroute`) it keeps the caller's graph and storm;
    the completion quiet gate is evaluated AGAIN on the committed bed before the dry handoff."""
    inputs = pit_inputs(np, ksat=1e-5, pit_ksat=0.0)
    policy = CompletionPolicy(hold_s=10.0, terminal_storage_water=True)
    ready = _ready_for_terminal(inputs, policy)
    saved = snapshot_any(ready.result.state)
    calls = []
    _record_final_commit(monkeypatch, calls)
    out = finalize_event(ready, inputs["ctx"], inputs["column"])
    assert isinstance(out, DryHandoff)
    commits = [c for c in calls if c[0] == "commit"]
    assert len(commits) == 1 and commits[0][1] is True and commits[0][2] == "final_forced" and commits[0][3] is False
    final = commits[0][4]
    assert final.storm is ready.result.state.storm and final.graph is ready.result.state.graph  # no flow change
    post = [c for c in calls[calls.index(commits[0]) + 1:] if c[0] == "quiet"]
    assert post and post[0][1] is final.bed and post[0][2] is True  # re-checked on the committed bed, still quiet
    assert out.progress.result.n_forced_commits == ready.result.n_forced_commits + 1
    compare_any(ready.result.state, saved)


# CW-E47C2: a GENUINE net crossing at the forced final commit (MAPLE's default 0.00075 m threshold), from a coherent
# controlled wet initial condition on the actual pit chain. After rain, a wet moving cell with Re <= 500 has no
# transport law (neither diffuse -- no rain energy -- nor concentrated; `sediment_physics`): no capacity / settle rule
# and a zero deposition rate, so a finite mobile pool there persists. With the explicit flow-end deposition policy the
# mobile quiet gate is waived (`deposit_mobile_at_flow_end`); the moving cell's depth stays below the quiet depth gate
# (zero infiltration there, no rain). `finalize_event` then deposits the whole pool through MAPLE (+1.6 mm at the cell:
# 0.5 kg / (1250 kg m-3 * 0.25 m2)) and the forced final commit genuinely reroutes.
FINAL_CROSSING_CELL, FINAL_CROSSING_DEPTH_M, FINAL_CROSSING_POOL_KG = (2, 0), 5e-9, 0.5


def _final_crossing_case():
    from maple_syrup.sediment_event import initial_event_state

    cell = FINAL_CROSSING_CELL
    inputs = pit_inputs(np, ksat=1e-5, pit_ksat=0.0, pit_ksat_cells=[(0, 0), cell])
    ctx, s0 = inputs["ctx"], inputs["state0"]
    control = SedimentEventControl(storm=StormControl(max_dt_s=1.0))
    policy = CompletionPolicy(hold_s=10.0, terminal_storage_water=True, deposit_mobile_at_flow_end=True)
    assert FINAL_CROSSING_DEPTH_M <= policy.max_depth_m  # the wet moving cell can satisfy the moving depth gate
    depth = np.zeros(s0.graph.shape)
    depth[cell] = FINAL_CROSSING_DEPTH_M
    pool = np.zeros_like(H(s0.bed.water.mobile_mass_by_cell_class_kg))
    pool[cell][0] = FINAL_CROSSING_POOL_KG  # a finite actual initial pool, through MAPLE's water-state constructor
    wet_bed = with_water(s0.bed, ctx, depth, pool)
    state0 = initial_event_state(s0.graph, s0.terrain, wet_bed, ctx, s0.storm.soil_water_m, control=control)
    assert float(H(state0.bed.water.mobile_mass_by_cell_class_kg).sum()) == FINAL_CROSSING_POOL_KG
    assert ctx.topographic_commit_spec.pending_elevation_trigger_m == 0.00075  # MAPLE's default, unchanged
    return inputs, state0, control, policy


def _complete_with_final_crossing(inputs, state0, control, policy):
    return complete_event(state0, inputs["ctx"], inputs["column"], inputs["field"], constant_rainfall(0.0, 1.0, 0.0),
                          inputs["vegetation"], inputs["sediment"], max_end_s=900.0, control=control, policy=policy,
                          report_every_s=2.0)


def _record_genuine_final(monkeypatch, rec, *, inject=None):
    """Transparent wrappers: `finalize_event` (the GENUINE ready progress `complete_event` hands over; checked and
    snapshotted before the terminal transaction), the actual `_commit_and_reroute` (outcome checked BEFORE any
    injection: a genuine reroute at or above the threshold) and the post-commit `quiet_metrics`. `inject` (refusal
    test only) may change ONLY the moving-cell discharge of the genuinely rerouted real result."""
    original_finalize, original_commit, original_quiet = module.finalize_event, module._commit_and_reroute, module.quiet_metrics

    def finalize(progress, *a, **k):
        st = progress.result.state
        closure = progress.result.closure()
        rec["ready"] = {"quiet_ready": progress.quiet.ready, "closed": closure["closed"],
                        "request_reconciled": closure["request_reconciled"],
                        "water_closed": module.water_budget(progress)["closed"], "n_reroutes": progress.result.n_reroutes,
                        "pool": H(st.bed.water.mobile_mass_by_cell_class_kg).copy(),
                        "depth": H(st.storm.depth_m).copy(), "graph": st.graph}
        rec["progress"], rec["saved"] = progress, snapshot_any(st)
        return original_finalize(progress, *a, **k)

    def commit(*a, **k):
        result = original_commit(*a, **k)
        rec.setdefault("commits", []).append((k.get("force"), result))
        if result is not None and inject is not None:
            assert result.rerouted is True and result.record["net_displacement_m"] >= result.record["reroute_threshold_m"]
            return inject(result)
        return result

    def quiet(state, *a, **k):
        out = original_quiet(state, *a, **k)
        if "commits" in rec:
            rec.setdefault("post_quiet", []).append((state, out[0], out[1]))
        return out
    monkeypatch.setattr(module, "finalize_event", finalize)
    monkeypatch.setattr(module, "_commit_and_reroute", commit)
    monkeypatch.setattr(module, "quiet_metrics", quiet)


def _assert_genuine_final_crossing(rec):
    """The pre-terminal progress was genuinely quiet with closed budgets and the finite pool still present; the ACTUAL
    forced final commit ran once and genuinely crossed the default net threshold to a refreshed graph."""
    ready = rec["ready"]
    assert ready["quiet_ready"] and ready["closed"] and ready["request_reconciled"] and ready["water_closed"]
    assert ready["n_reroutes"] == 0  # nothing rerouted before the terminal transaction
    pool = ready["pool"]
    assert pool[FINAL_CROSSING_CELL][0] == FINAL_CROSSING_POOL_KG and pool.sum() == FINAL_CROSSING_POOL_KG
    assert 0.0 < ready["depth"][FINAL_CROSSING_CELL] <= CompletionPolicy().max_depth_m  # wet, below the depth gate
    assert len(rec["commits"]) == 1
    force, final = rec["commits"][0]
    assert force is True and final is not None and final.record["commit_reason"] == "final_forced" and final.record["forced"]
    assert final.rerouted is True and final.record["rerouted"] is True
    assert final.record["reroute_threshold_m"] == 0.00075
    assert final.record["net_displacement_m"] >= final.record["reroute_threshold_m"]
    assert final.record["graph_changed"] is True and final.graph is not ready["graph"]
    assert final.graph.input_sha256 == final.record["graph_input_sha256"] != ready["graph"].input_sha256
    return final


def test_genuine_net_crossing_at_the_terminal_commit_reroutes_and_completes(monkeypatch):
    """CW-E47C2 positive: the actual terminal deposition makes the forced final commit a GENUINE reroute; the refreshed
    graph / network / storm (discharge recomputed from the new conveyance at unchanged depth) are adopted and the
    post-commit quiet gate, evaluated on that rerouted state, still holds -- the dry handoff completes with closed
    accounts and the pond / moving water removed only by the reported reset."""
    inputs, state0, control, policy = _final_crossing_case()
    saved0 = snapshot_any(state0)
    rec = {}
    _record_genuine_final(monkeypatch, rec)
    out = _complete_with_final_crossing(inputs, state0, control, policy)
    assert isinstance(out, DryHandoff)
    final = _assert_genuine_final_crossing(rec)
    (post_state, post_quiet, post_values), = rec["post_quiet"]
    assert post_state.bed is final.bed and post_state.storm is final.storm and post_state.graph is final.graph
    assert post_quiet is True and post_values["max_unit_discharge_m2_s"] <= policy.max_unit_discharge_m2_s
    r = out.progress.result
    assert r.n_reroutes == 1 and r.routing_source_origin == "net_reroute" and out.dry_state.graph is final.graph
    acc = out.accounting
    assert acc["sediment_closure"]["closed"] and abs(acc["combined_water_residual_m3"]) <= acc["storm_water_budget"]["tolerance_m3"]
    assert not np.any(H(out.dry_state.bed.water.mobile_mass_by_cell_class_kg))
    compare_any(state0, saved0)


def test_reactivated_moving_flow_after_a_genuine_terminal_reroute_refuses_the_handoff(monkeypatch):
    """CW-E47C2 (replaces the CW-E47C1 form that FORGED `rerouted=True` on a below-threshold commit): the same genuine
    final net crossing; only AFTER the real outcome is proven a genuine reroute (asserted inside the wrapper, before any
    change) is the moving pool cell's discharge in that genuinely rerouted result raised above the quiet
    unit-discharge gate (a transparent failure injection; no flag, graph, network, grid, digest, source, gate or
    counter is altered). The post-commit quiet check must refuse the dry handoff (`max_unit_discharge_m2_s` gate) and
    the genuine wet input -- the ready progress and the caller's initial state -- stays unchanged."""
    inputs, state0, control, policy = _final_crossing_case()
    saved0 = snapshot_any(state0)
    rec = {}

    def reactivated(result):
        q = np.array(result.storm.discharge_m2_s, copy=True)
        assert not np.asarray(result.graph.pit_storage)[FINAL_CROSSING_CELL]  # a MOVING cell, not the pit
        q[FINAL_CROSSING_CELL] = policy.max_unit_discharge_m2_s * 2
        return replace(result, storm=replace(result.storm, discharge_m2_s=q))
    _record_genuine_final(monkeypatch, rec, inject=reactivated)
    with pytest.raises(SedimentEventError, match="invalidated completion"):
        _complete_with_final_crossing(inputs, state0, control, policy)
    _assert_genuine_final_crossing(rec)
    (post_state, post_quiet, post_values), = rec["post_quiet"]
    assert post_quiet is False and post_values["max_unit_discharge_m2_s"] > policy.max_unit_discharge_m2_s
    assert post_values["moving_max_depth_m"] <= policy.max_depth_m  # only the injected discharge gate fails
    compare_any(rec["progress"].result.state, rec["saved"])  # the genuine wet ready input is unchanged
    compare_any(state0, saved0)


# =====================================================================================================================
# terrain commits: pit creation / removal under the policy, strict refusal
# =====================================================================================================================
def test_commit_that_fills_the_pit_moves_terminal_storage_and_is_recorded_while_strict_refuses():
    from maple.water import WaterProcessDemand

    inputs = pit_inputs(np)
    ctx, graph, terrain = inputs["ctx"], inputs["graph"], inputs["terrain"]
    bed = inputs["bed"]
    g = ctx.geometry
    # deposit enough at the pit (row 0) to raise it above row 1: dz > 0.01 m of bulk-density bed
    mass = 1.5 * 0.01 * g.bulk_density_kg_m3 * g.dx_m * g.dy_m
    deposit = np.zeros_like(np.asarray(bed.active_layer.mass_kg))
    deposit[0, 0] = mass / 6.0
    loaded = with_water(bed, ctx, bed.water.depth_m, deposit)
    settled, _result = apply_bed_demand(loaded, ctx, WaterProcessDemand(np.zeros_like(deposit), deposit))
    assert not np.any(settled.water.mobile_mass_by_cell_class_kg)
    outcome = commit_bed(settled, ctx, graph, terrain, 10.0, force=True)
    assert outcome.did_commit and outcome.terminal_storage_change == {
        "n_terminal_before": 1, "n_terminal_after": 1, "n_terminal_created": 1, "n_terminal_removed": 1}
    new = outcome.graph
    assert new.pit_storage[1, 0] and not new.pit_storage[0, 0] and new.receiver[0, 0] == 1 * new.shape[1]
    assert new.policy == graph.policy and new.input_sha256 != graph.input_sha256
    network = transport_network(new)
    assert network.n_terminal == 1 and network.terminal_host[1, 0]
    assert bool(H(physics_grid_from_graph(new).terminal)[1, 0])
    # the same refresh under the retained policy from the committed state
    again = refresh_routing(terrain, outcome.state)
    assert again.input_sha256 == new.input_sha256 and np.array_equal(again.pit_storage, new.pit_storage)
    # a STRICT terrain reference (south export, same bed before the deposit) refuses the new sink atomically
    from test_coupled_backends import graph_for_xp

    strict_bed, strict_ctx = make_bed(chain_elevation(4))
    strict_graph, strict_terrain = graph_for_xp(strict_bed, np)
    loaded = with_water(strict_bed, strict_ctx, strict_bed.water.depth_m, deposit)
    settled_strict, _ = apply_bed_demand(loaded, strict_ctx, WaterProcessDemand(np.zeros_like(deposit), deposit))
    before = [np.array(a, copy=True) for a in (settled_strict.voxel_column.mass_kg, settled_strict.active_layer.mass_kg)]
    with pytest.raises(RoutingGraphError, match="unsupported sinks"):
        commit_bed(settled_strict, strict_ctx, strict_graph, strict_terrain, 10.0, force=True)
    np.testing.assert_array_equal(before[0], settled_strict.voxel_column.mass_kg)
    np.testing.assert_array_equal(before[1], settled_strict.active_layer.mass_kg)


# =====================================================================================================================
# masked nodata
# =====================================================================================================================
def test_masked_nodata_is_never_a_receiver_and_the_datum_check_uses_the_active_terrain_only():
    bed, ctx, graph, terrain, z, exports, active = masked_two_column(np)
    assert terrain.allow_masked_nodata and terrain.nodata_value == NODATA and terrain.allow_pit_storage
    assert np.all(graph.receiver[:, 1] == INACTIVE) and not graph.active[:, 1].any()
    assert np.all(graph.receiver[:, 0][1:] == np.arange(3) * graph.shape[1]) and graph.receiver[0, 0] == PIT_STORAGE
    # the bed's placeholder elevations in the masked column differ from the sentinel by far more than a datum
    from maple.core.types.topographic_commit import committed_elevation_m

    zc = H(committed_elevation_m(bed.committed_topography))
    assert np.all(zc[:, 1] != NODATA)
    # without the sentinel on the reference the masked graph is refused; a strict build of this DEM is refused
    with pytest.raises(BedIntegrationError, match="nodata_value"):
        terrain_reference(z, exports, bed, graph)
    with pytest.raises(RoutingGraphError, match="nodata"):
        build_routing_graph(z, exports, np.ones((4, 2)), 0.5, active_mask=active, nodata_value=NODATA, allow_pit_storage=True)
    # a refresh keeps the sentinel on the masked column (never rewritten from the bed) and the same receivers
    refreshed = refresh_routing(terrain, bed)
    assert np.all(refreshed.receiver[:, 1] == INACTIVE) and np.array_equal(refreshed.receiver, graph.receiver)
    assert refreshed.input_sha256 == graph.input_sha256
    # an active cell holding the sentinel in the DEM is refused by the reference
    bad = z.copy()
    bad[2, 1] = NODATA
    with pytest.raises((BedIntegrationError, RoutingGraphError), match="nodata|ACTIVE"):
        terrain_reference(bad, exports, bed, graph, nodata_value=NODATA)
    # avalanching across the mask is refused by the commit, not silently applied
    avalanching = replace(ctx, avalanching_spec=replace(ctx.avalanching_spec, enabled=True))
    with pytest.raises(BedIntegrationError, match="avalanching"):
        commit_bed(bed, avalanching, graph, terrain, 1.0, force=True)
    # the event runs on the masked grid: the inactive column never holds water or sediment
    inputs = masked_inputs(np)
    state0 = inputs["state0"]
    r = run(inputs, constant_rainfall(0.0, 20.0, 72.0), 40.0, cadence=20.0)
    assert_closed_any(r, state0)
    assert not np.any(H(r.state.storm.depth_m)[:, 1]) and not np.any(H(r.cumulative_pickup_kg)[:, 1])
    assert not np.any(H(r.state.bed.water.mobile_mass_by_cell_class_kg)[:, 1])
    assert H(r.state.storm.depth_m)[0, 0] > 0.0 and float(H(r.cumulative_export_m3)) == 0.0


# =====================================================================================================================
# GPU (optional)
# =====================================================================================================================
# --- peak-time qualification on a constant-mass plateau (test-local; the shared comparator is untouched) ---------
def mobile_total_curve(inputs, schedule, end, control, *, cadence):
    """Run the event and record, at every accepted step, the EXACT expression the solver uses for its peak
    (`sediment_event`: per-class sum over cells, then the class sum), read through one counted host scalar."""
    from maple.core.backend import to_float

    curve = []

    def record(state):
        xp = state.graph.xp
        curve.append((float(state.t_s), to_float(xp.sum(state.bed.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1))))))
        return False

    r = evolve_sediment_event(inputs["state0"], inputs["ctx"], inputs["column"], inputs["field"], schedule,
                              inputs["vegetation"], inputs["sediment"], end, control, report_every_s=cadence,
                              on_accepted_step=record)
    return r, np.asarray(curve, dtype=np.float64)


def peak_plateau(curve, n_operands):
    """`(peak, bound, times)`: the maximum of the exact-expression totals, MAPLE's summation bound for the ACTUAL
    reduction (`n_operands` additions at the scale of that maximum, `summation_error_bound_kg`; no fitted
    tolerance, no mass-resolution floor) and every accepted time whose total is within that bound of the maximum.
    A single time means a well-defined peak; several mean the mass is constant to reduction round-off there."""
    from maple.surface.voxels._numerics import summation_error_bound_kg

    totals = curve[:, 1]
    peak = float(totals.max())
    bound = float(summation_error_bound_kg(int(n_operands), peak))
    return peak, bound, curve[np.abs(totals - peak) <= bound, 0]


def assert_peak_time_consistent(ref, ref_curve, cand, cand_curve, n_operands):
    """Science: once rainfall has stopped there is no pickup, and mobile mass in wet cells without a law
    (`sediment_physics` regime `wet_no_law`: no rain, Re <= 500 -> `law_defined` False, deposition rate 0,
    velocity = decayed memory) is conserved while it travels: the hazard `v r` is zero, crossings between moving
    cells do not change the total, and the total only drops when a packet reaches a settle (dry / no-capacity /
    terminal) cell. The exact total is therefore constant over such an interval up to the round-off of summing it
    over cells whose shares change as packets move, so the TIME of its maximum is a reduction-order tie, not a
    physical quantity. The peak MASS, every curve value and every other field stay at the original bounds.

    Asserts: both maxima and both reported peak masses within the bound of each other; both curves equal at the
    declared sediment bound step by step; the reported peak time of each backend lies in the other backend's
    plateau. Raises AssertionError otherwise (a non-flat curve with a foreign peak time is rejected)."""
    from maple.core.backend import to_float

    assert ref_curve.shape == cand_curve.shape and np.array_equal(ref_curve[:, 0], cand_curve[:, 0])
    np.testing.assert_allclose(cand_curve[:, 1], ref_curve[:, 1], rtol=SED_RTOL, atol=SED_ATOL, err_msg="mobile total curve")
    ref_peak, ref_bound, ref_times = peak_plateau(ref_curve, n_operands)
    cand_peak, cand_bound, cand_times = peak_plateau(cand_curve, n_operands)
    bound = max(ref_bound, cand_bound)
    assert abs(ref_peak - cand_peak) <= bound
    for result, peak in ((ref, ref_peak), (cand, cand_peak)):
        assert abs(to_float(result.peak_mobile_kg) - peak) <= bound
    ref_time, cand_time = to_float(ref.time_of_peak_mobile_s), to_float(cand.time_of_peak_mobile_s)
    assert ref_time in set(ref_times.tolist()) and cand_time in set(cand_times.tolist())
    assert ref_time in set(cand_times.tolist()), f"reference peak time {ref_time} outside the candidate plateau {cand_times}"
    assert cand_time in set(ref_times.tolist()), f"candidate peak time {cand_time} outside the reference plateau {ref_times}"
    return ref_times, cand_times


def compare_events_with_qualified_peak_time(ref, cand, ref_curve, cand_curve, n_operands):
    """The shared `compare_events` on every field at its original bounds; `time_of_peak_mobile_s` is aligned to
    the reference ONLY after `assert_peak_time_consistent` proved both times lie on a shared constant-mass
    plateau, and only when that plateau has more than one accepted time (a single-time peak stays strict)."""
    ref_times, cand_times = assert_peak_time_consistent(ref, ref_curve, cand, cand_curve, n_operands)
    if len(set(ref_times.tolist()) & set(cand_times.tolist())) > 1:
        cand = replace(cand, time_of_peak_mobile_s=ref.time_of_peak_mobile_s)
    compare_events(ref, cand)
    return ref_times, cand_times


def n_peak_operands(result):
    """Additions of the exact peak expression: cells x classes (per-class sums over cells) plus the class sum."""
    ny, nx = result.state.graph.shape
    nc = int(result.state.bed.water.mobile_mass_by_cell_class_kg.shape[-1])
    return ny * nx * nc + nc


def test_peak_time_qualification_rejects_a_foreign_time_on_a_non_flat_curve():
    """Negative control on the host only: on a strictly rising-then-falling curve the plateau is one time, and a
    result reporting another time is rejected; a self-comparison of the real pit-chain run passes."""
    schedule = constant_rainfall(0.0, 20.0, 72.0)
    inputs = pit_inputs(np, control=control_for("array"))
    r, curve = mobile_total_curve(inputs, schedule, 60.0, control_for("array"), cadence=20.0)
    n_ops = n_peak_operands(r)
    assert_peak_time_consistent(r, curve, r, curve, n_ops)
    peak, bound, times = peak_plateau(curve, n_ops)
    assert bound > 0.0 and peak == float(H(r.peak_mobile_kg)) and float(H(r.time_of_peak_mobile_s)) in set(times.tolist())
    # a non-flat curve: replace the totals by a strict ramp up to the peak and down again
    ramp = curve.copy()
    k = int(np.argmax(curve[:, 1]))
    ramp[:k + 1, 1] = np.linspace(0.0, peak, k + 1)
    ramp[k + 1:, 1] = np.linspace(peak * 0.999, peak * 0.5, len(curve) - k - 1)
    _p, _b, single = peak_plateau(ramp, n_ops)
    assert single.tolist() == [curve[k, 0]]
    foreign = replace(r, time_of_peak_mobile_s=np.asarray(10.0))
    with pytest.raises(AssertionError):
        assert_peak_time_consistent(r, ramp, foreign, ramp, n_ops)
    with pytest.raises(AssertionError):
        compare_events_with_qualified_peak_time(r, foreign, ramp, ramp, n_ops)
    # a curve amplitude outside the declared bound is rejected too
    scaled = curve.copy()
    scaled[:, 1] *= 1.0 + 1e-8
    with pytest.raises(AssertionError):
        assert_peak_time_consistent(r, curve, r, scaled, n_ops)


def test_gpu_pit_chain_event_and_terminal_completion_match_the_host(gpu):
    schedule = constant_rainfall(0.0, 20.0, 72.0)
    host = pit_inputs(np, control=control_for("array"))
    ref, ref_curve = mobile_total_curve(host, schedule, 60.0, control_for("array"), cadence=20.0)
    dev = pit_inputs(gpu, control=control_for("cuda"))
    out, dev_curve = mobile_total_curve(dev, schedule, 60.0, control_for("cuda"), cadence=20.0)
    assert dev["state0"].grid.terminal is not None and dev["state0"].network.terminal_flat is not None
    assert_terminal_event(out, dev, terminal_cells=[(0, 0)])
    # the original strict gate failed only on time_of_peak_mobile_s (CPU 26 s, GPU 20 s) with a 1-ULP peak mass
    # and a constant exact total over 20-33 s on both backends: qualified here as a plateau tie, every other
    # field at its original bound
    ref_times, dev_times = compare_events_with_qualified_peak_time(ref, out, ref_curve, dev_curve, n_peak_operands(ref))
    assert len(ref_times) >= 1 and len(dev_times) >= 1
    policy = CompletionPolicy(hold_s=10.0, terminal_storage_water=True)
    storm = {"max_dt_s": 1.0}
    rain = constant_rainfall(0.0, 10.0, 100.0)
    host_c = pit_inputs(np, ksat=1e-5, pit_ksat=0.0, control=control_for("array", storm=storm))
    dev_c = pit_inputs(gpu, ksat=1e-5, pit_ksat=0.0, control=control_for("cuda", storm=storm))
    a = complete(host_c, rain, max_end_s=900.0, policy=policy, control=control_for("array", storm=storm))
    b = complete(dev_c, rain, max_end_s=900.0, policy=policy, control=control_for("cuda", storm=storm))
    assert isinstance(a, DryHandoff) and isinstance(b, DryHandoff)
    # the same qualification for the completed storms: the exact peak curve is replayed with the identical
    # control (complete_event runs the loop with force_final_commit=False on the same time partition)
    loop_control = {impl: replace(control_for(impl, storm=storm), force_final_commit=False) for impl in ("array", "cuda")}
    _ra, curve_a = mobile_total_curve(host_c, rain, a.dry_state.t_s, loop_control["array"], cadence=2.0)
    _rb, curve_b = mobile_total_curve(dev_c, rain, b.dry_state.t_s, loop_control["cuda"], cadence=2.0)
    ta, tb = assert_peak_time_consistent(a.progress.result, curve_a, b.progress.result, curve_b, n_peak_operands(a.progress.result))
    if len(set(ta.tolist()) & set(tb.tolist())) > 1:
        aligned = replace(b.progress.result, time_of_peak_mobile_s=a.progress.result.time_of_peak_mobile_s)
        b = replace(b, progress=replace(b.progress, result=aligned))
    compare_handoffs(a, b)
    assert b.accounting["terminal_storage_cells"] == 1 and b.accounting["terminal_storage_removed_m3"] > 0.0
    assert a.accounting["terminal_storage_removed_m3"] == pytest.approx(b.accounting["terminal_storage_removed_m3"], rel=2e-12, abs=1e-14)
