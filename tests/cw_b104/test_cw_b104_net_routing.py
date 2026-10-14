"""CW-B104: net-elevation-only hydraulic rerouting on actual MAPLE beds.

Every test runs on host NumPy and, when a CUDA device is visible (GPU0 only), on CuPy (`xp` fixture; an explicit skip
claims nothing). Prescribed exchanges go through MAPLE's actual water demand API (`apply_bed_demand`: supply-limited
pickup into the cell's own mobile pool, deposition from it) and MAPLE's actual commit (`commit_bed`). The threshold is
the actual MAPLE `pending_elevation_trigger_m` (default 0.00075 m, compared `>=`); tests that need another value say
so and use an EXPLICITLY LABELLED fixture value. No tolerance is introduced: equalities are bitwise, displacements
are compared with the threshold exactly as the implementation does.

Mass to elevation in a `make_bed` cell: dz = m / (bulk 1250 kg/m3 * 0.25 m2) = m / 312.5 m per kg, so 0.3 kg is
~0.00096 m (> threshold), 0.2 kg ~0.00064 m and 0.1 kg ~0.00032 m (< threshold).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("maple")

from test_coupled_backends import H, compare_any, setup_xp, snapshot_any
from test_sediment_event import (
    FRACTIONS,
    NC,
    chain_elevation,
    make_bed,
    valley_elevation,
)

from maple_syrup import sediment_bed, sediment_event
from maple_syrup.complete_event import (
    CompletionPolicy,
    DryHandoff,
    complete_event,
)
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_bed import (
    apply_bed_demand,
    commit_bed,
    current_routing_surface_m,
    routing_graph_from_elevation,
    with_water,
)
from maple_syrup.sediment_event import (
    CudaHydraulics,
    SedimentEventControl,
    SedimentEventError,
    _commit_and_reroute,
    evolve_sediment_event,
    routing_gate_m,
)
from maple_syrup.storm import StormControl

DEFAULT_THRESHOLD_M = 0.00075
SOUTH = valley_elevation(5, 5)[0] - 0.01
CELL_A, CELL_B = (2, 1), (2, 3)
#: LABELLED FIXTURE values switching MAPLE's elapsed-time and gross-turnover triggers off, so that only the tests'
#: explicit commits happen (never production values).
QUIET = {"commit_interval_s": 1e9, "active_layer_turnover_fraction_trigger": 1e3}


def spec(**kw):
    from maple.core.parameters.topographic_commit import TopographicCommitSpec

    return TopographicCommitSpec(**kw)


def committed(state):
    from maple.core.types.topographic_commit import committed_elevation_m

    return committed_elevation_m(state.bed.committed_topography)


def valley(xp, commit_spec=None, *, control=None, mobile_at=None, mobile_kg=0.0):
    """4x5 valley on an actual MAPLE bed in `xp`; optional seeded mobile mass at one cell (part of the initial
    inventory; changes no elevation)."""
    bed, ctx = make_bed(valley_elevation(4, 5), commit_spec=commit_spec)
    if mobile_at is not None:
        mobile = np.zeros((4, 5, NC))
        mobile[mobile_at] = mobile_kg * np.asarray(FRACTIONS)
        bed = with_water(bed, ctx, bed.water.depth_m, mobile)
    return setup_xp(bed, ctx, xp, south_ring=SOUTH, control=control)


def exchange(state, ctx, cell, *, pickup=0.0, deposit=0.0):
    """One actual MAPLE water exchange at `cell`, split over the classes by the bed fractions (removal into the cell's
    own mobile pool, deposition from it). Returns the new state and MAPLE's result."""
    xp = state.graph.xp
    removal = xp.zeros((*state.graph.shape, NC), dtype=np.float64)
    deposition = xp.zeros((*state.graph.shape, NC), dtype=np.float64)
    split = xp.asarray(np.asarray(FRACTIONS))
    removal[cell] = pickup * split
    deposition[cell] = deposit * split
    from maple.water import WaterProcessDemand

    bed, result = apply_bed_demand(state.bed, ctx, WaterProcessDemand(removal, deposition))
    return replace(state, bed=bed), result


def gate(state, ctx):
    return routing_gate_m(state.bed, ctx, state.graph, state.routing_source_elevation_m)


def bed_commit(state, ctx, *, force=True):
    return commit_bed(state.bed, ctx, state.graph, state.terrain, 1.0, force=force,
                      routing_source_m=state.routing_source_elevation_m)


# --- contract ---------------------------------------------------------------------------------------------------------
def test_threshold_is_the_actual_maple_pending_trigger_with_its_comparison():
    import inspect

    from maple.surface.topographic_commit import triggers

    assert spec().pending_elevation_trigger_m == DEFAULT_THRESHOLD_M
    # MAPLE's own pending-elevation comparison is `>=` (the water / generic trigger; the dry helper's `>` is not used)
    assert "max_pending_elevation_change_m >= config.pending_elevation_trigger_m" in inspect.getsource(triggers)
    _, ctx = make_bed(valley_elevation(4, 5))
    assert sediment_bed.reroute_threshold_m(ctx) == DEFAULT_THRESHOLD_M
    record = sediment_event.routing_refresh_record(ctx)
    assert record["comparison"] == ">=" and record["threshold_m"] == DEFAULT_THRESHOLD_M
    assert record["contract"] == sediment_event.ROUTING_REFRESH_CONTRACT


def test_refresh_routing_is_the_explicit_source_build_byte_for_byte(xp):
    s = valley(xp)
    state = s["state0"]
    a = sediment_bed.refresh_routing(state.terrain, state.bed, xp=xp)
    b = routing_graph_from_elevation(state.terrain, state.routing_source_elevation_m, xp=xp)
    from maple_syrup.checkpoint import _graph_digest

    assert _graph_digest(a) == _graph_digest(b) and a.input_sha256 == b.input_sha256


def test_initialization_sets_the_routing_source_to_the_committed_elevation(xp):
    s = valley(xp)
    state = s["state0"]
    src = state.routing_source_elevation_m
    assert src.dtype == np.float64 and tuple(src.shape) == state.graph.shape
    from maple.core.backend import array_namespace

    assert array_namespace(src) is xp
    np.testing.assert_array_equal(H(src), H(committed(state)))
    # and it is a fresh array, not a view of the committed state's fields
    assert src is not state.bed.committed_topography.reference_elevation_m


# --- exact current surface ------------------------------------------------------------------------------------------
def test_current_surface_is_bitwise_what_the_commit_publishes_dense(xp):
    s = valley(xp, spec(**QUIET), mobile_at=CELL_B, mobile_kg=0.3)
    state, ctx = s["state0"], s["ctx"]
    np.testing.assert_array_equal(H(current_routing_surface_m(state.bed, ctx)), H(committed(state)))
    state, _ = exchange(state, ctx, CELL_A, pickup=0.3)
    state, _ = exchange(state, ctx, CELL_B, deposit=0.3)
    surface = H(current_routing_surface_m(state.bed, ctx))
    outcome = bed_commit(state, ctx)
    assert outcome.did_commit
    np.testing.assert_array_equal(surface, H(committed(replace(state, bed=outcome.state))))


# --- prescribed exchanges ---------------------------------------------------------------------------------------------
def test_same_cell_gain_and_loss_cancel_gross_turnover_commits_without_reroute(xp):
    """Gross turnover (pickup then equal deposition in ONE cell) is net zero: a commit (MAPLE's turnover trigger
    fires) refreshes nothing; the intermediate state (pickup only) would have reached the threshold."""
    s = valley(xp, spec(commit_interval_s=1e9))  # default turnover trigger 0.035 active-layer fraction
    state, ctx = s["state0"], s["ctx"]
    state, picked = exchange(state, ctx, CELL_A, pickup=0.3)
    assert float(H(picked.actual_removal_by_cell_class_kg).sum()) == pytest.approx(0.3, rel=1e-12)
    assert gate(state, ctx) >= DEFAULT_THRESHOLD_M  # the pickup alone would reroute
    state, _ = exchange(state, ctx, CELL_A, deposit=0.3)
    g = gate(state, ctx)
    assert g < DEFAULT_THRESHOLD_M
    outcome = bed_commit(state, ctx, force=False)
    assert outcome.did_commit and outcome.trigger.active_layer_turnover_triggered
    assert not outcome.trigger.pending_elevation_triggered
    assert not outcome.rerouted and outcome.graph is state.graph
    assert outcome.routing_source_m is state.routing_source_elevation_m
    assert outcome.net_displacement_m < DEFAULT_THRESHOLD_M


@pytest.mark.parametrize("mass_kg, reroutes", [(0.3, True), (0.2, False)])
def test_opposing_cells_never_cancel(xp, mass_kg, reroutes):
    """A loss in cell A and an equal gain in cell B: the signed sum over cells is ~0 but the per-cell maximum decides."""
    s = valley(xp, spec(**QUIET), mobile_at=CELL_B, mobile_kg=mass_kg)
    state, ctx = s["state0"], s["ctx"]
    state, _ = exchange(state, ctx, CELL_A, pickup=mass_kg)
    state, _ = exchange(state, ctx, CELL_B, deposit=mass_kg)
    d = H(current_routing_surface_m(state.bed, ctx)) - H(state.routing_source_elevation_m)
    assert abs(float(d.sum())) < 1e-3 * float(np.abs(d).max())  # opposite signs: they WOULD cancel if summed
    g = gate(state, ctx)
    assert g == float(np.abs(d).max())
    assert (g >= DEFAULT_THRESHOLD_M) == reroutes
    outcome = bed_commit(state, ctx)
    assert outcome.did_commit and outcome.rerouted == reroutes
    if reroutes:
        assert outcome.graph is not state.graph
        np.testing.assert_array_equal(H(outcome.routing_source_m), H(committed(replace(state, bed=outcome.state))))
    else:
        assert outcome.graph is state.graph and outcome.routing_source_m is state.routing_source_elevation_m


def test_subthreshold_reversal_across_an_accounting_commit(xp):
    """+0.6 threshold, an accounting commit (ledger reset), then the reversal: never rerouted; the routing source is
    held across the commit and the displacement returns toward zero."""
    s = valley(xp, spec(**QUIET), mobile_at=CELL_A, mobile_kg=0.15)
    state, ctx = s["state0"], s["ctx"]
    source = state.routing_source_elevation_m
    state, _ = exchange(state, ctx, CELL_A, deposit=0.15)  # +0.00048 m
    first = bed_commit(state, ctx)
    assert first.did_commit and not first.rerouted and first.routing_source_m is source
    state = replace(state, bed=first.state)
    assert 0.0 < gate(state, ctx) < DEFAULT_THRESHOLD_M
    state, _ = exchange(state, ctx, CELL_A, pickup=0.15)  # back
    assert gate(state, ctx) < first.net_displacement_m
    second = bed_commit(state, ctx)
    assert second.did_commit and not second.rerouted and second.routing_source_m is source


def test_cumulative_net_change_across_ledger_resets_forces_a_net_commit_between_maple_triggers(xp):
    """Three +0.1 kg increments in one cell, each followed by an accounting commit while it is below the threshold:
    MAPLE's pending measure restarts at every commit (each increment alone ~0.00032 m < threshold), the held
    routing source does not; the third increment reaches the threshold between MAPLE triggers and forces a genuine
    commit (reason net_elevation) that reroutes from the committed geometry."""
    s = valley(xp, spec(**QUIET), mobile_at=CELL_A, mobile_kg=0.3)
    state, ctx = s["state0"], s["ctx"]
    source0 = state.routing_source_elevation_m
    reasons = []
    for k in range(3):
        state, _ = exchange(state, ctx, CELL_A, deposit=0.1)
        g = gate(state, ctx)
        commit = _commit_and_reroute(state.bed, ctx, state.graph, state.terrain, state.storm, 1.0 + k, force=False,
                                     routing_source_m=state.routing_source_elevation_m,
                                     net_due=g >= DEFAULT_THRESHOLD_M, gate_m=g)
        if k < 2:
            assert g < DEFAULT_THRESHOLD_M and commit is None  # MAPLE does not trigger, net below threshold
            outcome = bed_commit(state, ctx)  # an explicit accounting commit (ledger reset), no reroute
            assert outcome.did_commit and not outcome.rerouted and outcome.routing_source_m is source0
            assert outcome.trigger.max_pending_elevation_change_m < DEFAULT_THRESHOLD_M
            state = replace(state, bed=outcome.state)
            continue
        assert g >= DEFAULT_THRESHOLD_M and commit is not None
        reasons.append(commit.record["commit_reason"])
        assert commit.record["max_pending_elevation_change_m"] < DEFAULT_THRESHOLD_M  # MAPLE alone would not commit
        assert not commit.record["pending_elevation_triggered"] and not commit.record["forced"]
        assert commit.rerouted and commit.record["rerouted"] and commit.graph is not state.graph
        np.testing.assert_array_equal(H(commit.routing_source_m), H(committed(replace(state, bed=commit.bed))))
        assert commit.record["net_displacement_before_commit_m"] == g == commit.record["net_displacement_m"]
    assert reasons == ["net_elevation"]


def test_threshold_boundary_is_inclusive_on_exact_binary_values(xp):
    """The decision is `net >= threshold` on the exact gate value: a threshold EQUAL to it reroutes, the next
    representable value above it does not (labelled fixture thresholds; no tolerance)."""
    s = valley(xp, spec(**QUIET), mobile_at=CELL_A, mobile_kg=0.2)
    state, ctx = s["state0"], s["ctx"]
    state, _ = exchange(state, ctx, CELL_A, deposit=0.2)
    g = gate(state, ctx)
    for threshold, expected in ((g, True), (float(np.nextafter(g, np.inf)), False)):
        c = replace(ctx, topographic_commit_spec=spec(pending_elevation_trigger_m=threshold, **QUIET))
        assert gate(state, c) == g
        outcome = commit_bed(state.bed, c, state.graph, state.terrain, 1.0, force=True,
                             routing_source_m=state.routing_source_elevation_m)
        assert outcome.net_displacement_m == g and outcome.rerouted is expected


# --- the event loop ---------------------------------------------------------------------------------------------------
def counting(monkeypatch):
    """Count the hydraulic refresh operations inside the event loop (graph builds, networks, grids, discharge
    re-initialisations)."""
    counts = {"graph_builds": 0, "networks": 0, "grids": 0, "discharge_reinit": 0}

    def wrap(module, name, key):
        real = getattr(module, name)

        def counted(*a, **k):
            counts[key] += 1
            return real(*a, **k)

        monkeypatch.setattr(module, name, counted)

    wrap(sediment_bed, "build_routing_graph", "graph_builds")
    wrap(sediment_event, "transport_network", "networks")
    wrap(sediment_event, "physics_grid_from_graph", "grids")
    wrap(sediment_event, "initial_state", "discharge_reinit")
    return counts


@pytest.mark.parametrize("trigger", ["elapsed_time", "gross_turnover"])
def test_accounting_commits_from_time_or_turnover_rebuild_nothing(xp, monkeypatch, trigger):
    """A real 45 s storm (max net ~5e-5 m < threshold): MAPLE commits on elapsed time (every 10 s) or on gross
    turnover; no graph build beyond the one entry integrity rebuild, no network / grid / discharge re-initialisation,
    the graph / network / grid / routing source objects are the initial ones, every commit is recorded and accounted;
    under CUDA the context is prepared once and every commit's transfers are recorded."""
    commit_spec = spec(commit_interval_s=10.0) if trigger == "elapsed_time" else spec(commit_interval_s=1e9)
    implementation = "cuda" if xp is not np else "array"
    control = SedimentEventControl(storm=StormControl(implementation=implementation),
                                   transport_implementation="auto" if xp is np else "array")
    s = valley(xp, commit_spec, control=control)
    state0 = s["state0"]
    saved = snapshot_any(state0)
    hydraulics = CudaHydraulics() if xp is not np else None
    counts = counting(monkeypatch)
    r = evolve_sediment_event(state0, s["ctx"], s["column"], s["field"], constant_rainfall(0.0, 45.0, 72.0),
                              s["vegetation"], s["sediment"], 45.0, control, report_every_s=15.0, hydraulics=hydraulics)
    compare_any(state0, saved)  # caller inputs untouched
    assert r.n_commits >= 2 and r.n_reroutes == 0 and r.n_graph_changes == 0 and r.n_net_commits == 0
    reasons = {c["commit_reason"] for c in r.commit_log}
    assert reasons <= {"maple_trigger", "final_forced"} and not any(c["rerouted"] for c in r.commit_log)
    key = "elapsed_time_triggered" if trigger == "elapsed_time" else "active_layer_turnover_triggered"
    assert any(c[key] for c in r.commit_log)
    assert counts == {"graph_builds": 1, "networks": 0, "grids": 0, "discharge_reinit": 0}  # 1 = entry integrity check
    st = r.state
    assert st.graph is state0.graph and st.network is state0.network and st.grid is state0.grid
    assert st.routing_source_elevation_m is state0.routing_source_elevation_m
    assert r.routing_source_origin == "entry" and r.routing_source_t_s is None
    assert 0.0 < r.max_unrouted_net_displacement_m < DEFAULT_THRESHOLD_M
    # one gate per accepted step, plus the evaluation of the loop's forced final commit
    assert r.n_routing_gate_evaluations == r.n_accepted_steps + r.n_forced_commits
    # the committed surface genuinely moved away from the (held) routing source: a deliberate sub-threshold lag
    lag = float(np.abs(H(committed(st)) - H(st.routing_source_elevation_m)).max())
    assert 0.0 < lag < DEFAULT_THRESHOLD_M
    closure = r.closure()
    assert closure["closed"] and closure["request_reconciled"]
    if hydraulics is not None:
        assert hydraulics.n_preparations == 1 and hydraulics.n_commits_recorded == r.n_commits


def test_net_reroutes_in_the_loop_rebuild_coherently(xp, monkeypatch):
    """LABELLED fixture threshold 2e-5 m (also MAPLE's pending trigger): the loop reroutes only when the held-source
    displacement reaches it, and then rebuilds graph, network, grid, discharge and (CUDA) context exactly once per
    reroute; non-rerouting commits rebuild nothing."""
    implementation = "cuda" if xp is not np else "array"
    control = SedimentEventControl(storm=StormControl(implementation=implementation),
                                   transport_implementation="auto" if xp is np else "array")
    s = valley(xp, spec(commit_interval_s=10.0, pending_elevation_trigger_m=2e-5), control=control)
    hydraulics = CudaHydraulics() if xp is not np else None
    counts = counting(monkeypatch)
    r = evolve_sediment_event(s["state0"], s["ctx"], s["column"], s["field"], constant_rainfall(0.0, 45.0, 72.0),
                              s["vegetation"], s["sediment"], 45.0, control, report_every_s=15.0, hydraulics=hydraulics)
    rerouted = [c for c in r.commit_log if c["rerouted"]]
    assert r.n_reroutes == len(rerouted) >= 1 and len(r.commit_log) == r.n_commits
    assert r.n_reroutes < r.n_commits  # some commits did not reroute
    for c in r.commit_log:
        assert c["rerouted"] == (c["net_displacement_m"] >= 2e-5)
    assert counts["graph_builds"] == 1 + r.n_reroutes and counts["networks"] == r.n_reroutes
    assert counts["grids"] == r.n_reroutes and counts["discharge_reinit"] == r.n_reroutes
    assert r.routing_source_origin == "net_reroute" and r.routing_source_t_s == rerouted[-1]["t_s"]
    st = r.state
    from maple_syrup.checkpoint import _graph_digest

    rebuilt = routing_graph_from_elevation(st.terrain, st.routing_source_elevation_m, xp=xp)
    assert _graph_digest(rebuilt) == _graph_digest(st.graph)
    assert r.closure()["closed"]
    if hydraulics is not None:
        assert hydraulics.n_preparations == 1 + r.n_reroutes and hydraulics.n_commits_recorded == r.n_commits


# --- refusals ---------------------------------------------------------------------------------------------------------
def test_malformed_or_inconsistent_routing_source_is_refused_before_any_work(xp):
    s = valley(xp)
    state0 = s["state0"]
    saved = snapshot_any(state0)
    src = state0.routing_source_elevation_m
    bad = {
        "missing": None,
        "shape": xp.zeros((3, 5), dtype=np.float64),
        "dtype": src.astype(np.float32),
        "nonfinite": xp.where(xp.arange(src.size).reshape(src.shape) == 0, xp.nan, src),
        "foreign_graph": src + xp.where(xp.arange(src.size).reshape(src.shape) == 7, -1e-4, 0.0),
        "invalid_graph": src + xp.where(xp.arange(src.size).reshape(src.shape) == 7, 0.05, 0.0),  # creates a pit
    }
    if xp is not np:
        bad["namespace"] = H(src).copy()  # a host source for a device event
    for value in bad.values():
        state = replace(state0, routing_source_elevation_m=value)
        with pytest.raises(SedimentEventError, match="routing"):
            evolve_sediment_event(state, s["ctx"], s["column"], s["field"], constant_rainfall(0.0, 4.0, 72.0),
                                  s["vegetation"], s["sediment"], 4.0, SedimentEventControl(), report_every_s=2.0)
    compare_any(state0, saved)


def test_initial_state_refuses_a_graph_that_is_not_the_committed_bed_graph(xp):
    from maple_syrup.sediment_event import initial_event_state

    s = valley(xp)
    state0, ctx = s["state0"], s["ctx"]
    other = valley(xp, mobile_at=None)["state0"]
    moved, _ = exchange(other, ctx, CELL_A, pickup=0.3)
    outcome = commit_bed(moved.bed, ctx, moved.graph, moved.terrain, 1.0, force=True)  # old form: refreshes
    with pytest.raises(SedimentEventError, match="initial routing"):
        initial_event_state(state0.graph, state0.terrain, outcome.state, ctx, state0.storm.soil_water_m)


# --- completion -------------------------------------------------------------------------------------------------------
def chain_complete(xp, commit_spec=None):
    bed, ctx = make_bed(chain_elevation(3), depth_m=.001, commit_spec=commit_spec)
    control = SedimentEventControl(storm=StormControl(max_dt_s=1, max_steps=10000))
    s = setup_xp(bed, ctx, xp, ksat=1e-5, control=control)
    args = (s["state0"], s["ctx"], s["column"], s["field"], constant_rainfall(0, 10, 100), s["vegetation"],
            s["sediment"])
    kw = {"max_end_s": 180, "control": control, "policy": CompletionPolicy(hold_s=2), "report_every_s": 2.}
    return s, args, kw


def test_dry_completion_terminal_forced_commit_is_accounting_only_below_the_threshold(xp):
    """A real wet storm to the dry handoff (default threshold): the terminal forced commit is an accounting commit
    (reason final_forced); nothing reroutes, the routing source is the initialization one; budgets close."""
    s, args, kw = chain_complete(xp, spec(commit_interval_s=2.0))
    out = complete_event(*args, **kw)
    assert isinstance(out, DryHandoff)
    r = out.progress.result
    forced = [c for c in r.commit_log if c["commit_reason"] == "final_forced"]
    assert all(c["forced"] for c in forced) and r.n_forced_commits == len(forced)
    assert r.n_reroutes == 0 and not any(c["rerouted"] for c in r.commit_log)
    assert out.dry_state.graph is s["state0"].graph
    assert out.dry_state.routing_source_elevation_m is s["state0"].routing_source_elevation_m
    assert r.closure()["closed"] and out.accounting["storm_water_budget"]["closed"]


@pytest.mark.parametrize("pool_kg, reroutes", [(0.1, True), (1e-12, False)])
def test_terminal_deposition_reroutes_the_final_commit_only_when_net_driven(pool_kg, reroutes):
    """`finalize_event` on a ready wet progress carrying a mobile pool of `pool_kg` per cell and class (6 classes):
    0.6 kg per cell deposits ~0.0019 m (>= the DEFAULT threshold) and the terminal forced commit reroutes from the
    committed geometry; 6e-12 kg per cell does not. Host fixture (the phase 6 terminal-deposition construction)."""
    from test_complete_event import fixture

    from maple_syrup.complete_event import QuietProgress, finalize_event
    from maple_syrup.sediment_bed import bed_inventory
    from maple_syrup.storm import initial_state

    args, kw = fixture()
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    r = paused.result
    pool = np.full_like(r.state.bed.active_layer.mass_kg, pool_kg)
    h = np.full(r.state.graph.shape, 1e-10)
    bed = with_water(r.state.bed, args[1], h, pool)
    state = replace(r.state, bed=bed, storm=initial_state(r.state.graph, h, r.state.storm.soil_water_m, t_s=2))
    r = replace(r, state=state, initial_mobile_kg=pool.sum(axis=(0, 1)))
    p = replace(paused, result=r, policy=CompletionPolicy(hold_s=1, deposit_mobile_at_flow_end=True),
                initial_surface_m3=float(h.sum() * .25), rainfall_end_s=0.,
                quiet=QuietProgress(since_s=0., last_t_s=2., ready=True))
    initial_bed = bed_inventory(state.bed)
    out = finalize_event(p, args[1], args[2])
    final = out.progress.result.commit_log[-1]
    assert final["commit_reason"] == "final_forced" and final["rerouted"] is reroutes
    assert out.progress.result.n_reroutes == int(reroutes) and out.progress.result.n_forced_commits >= 1
    if reroutes:
        assert final["net_displacement_m"] >= DEFAULT_THRESHOLD_M
        np.testing.assert_array_equal(out.dry_state.routing_source_elevation_m, committed(out.dry_state))
        assert out.progress.result.routing_source_origin == "net_reroute"
    else:
        assert final["net_displacement_m"] < DEFAULT_THRESHOLD_M
        assert out.dry_state.graph is state.graph
        assert out.dry_state.routing_source_elevation_m is state.routing_source_elevation_m
    np.testing.assert_allclose(bed_inventory(out.dry_state.bed) - initial_bed, pool.sum(axis=(0, 1)),
                               atol=out.accounting["terminal_sediment_tolerance_kg"], rtol=0)
    assert out.accounting["sediment_closure"]["closed"]
    np.testing.assert_array_equal(p.result.state.bed.water.mobile_mass_by_cell_class_kg, pool)  # input untouched
