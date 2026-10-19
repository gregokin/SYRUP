"""CW-C1: the complete event (quiet monitor, terminal settlement, forced final commit, external dry reset) on
small REAL MAPLE beds in either namespace.

CPU tests (always run): the completion reductions equal the direct host expressions (the packet adds no
arithmetic), the water budget equals its direct host computation, a CUDA control on a NumPy state and a
lifecycle with a non-CUDA control are refused before any work.

GPU tests (skip honestly without CuPy and a device; a skip claims nothing): a quiescent storm and a wet storm
with terrain change complete with a dry handoff on the device (bisection and Newton), nonzero terminal
settlement is a genuine MAPLE deposition, a failed terminal deposit / commit / reactivated flow leaves the wet
input unchanged, a duration limit is not a dry handoff, the CUDA lifecycle count matches the ACTUAL net reroutes of the loop (held routing:
none; a native conservative net-elevation precondition: genuine reroutes, including a genuine rerouted terminal commit), and
the device event matches the host array / numba events at the DECLARED bounds (water rtol 2e-12 / atol 1e-14,
sediment rtol 1e-10 / atol 1e-14 predeclared for the coupled event, MAPLE closure unchanged, integer counters
exact).

Nothing here was run by its author (file-only tools); the root records the executed results.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("maple")

from test_coupled_backends import (  # tests/conservative_water via conftest
    NUMBA_SKIP,
    SED_ATOL,
    SED_RTOL,
    H,
    assert_closed_any,
    close,
    compare_any,
    compare_events,
    conservative_net_cross,
    setup_xp,
    snapshot_any,
)
from test_sediment_event import chain_elevation, make_bed

from maple_syrup import complete_event as module
from maple_syrup.complete_event import (
    CompletionPolicy,
    DryHandoff,
    EventProgress,
    IncompleteEventError,
    QuietProgress,
    complete_event,
    finalize_event,
    quiet_metrics,
    water_budget,
)
from maple_syrup.event_backend import host_packet
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_bed import bed_inventory, reroute_threshold_m, with_water
from maple_syrup.sediment_event import (
    CudaHydraulics,
    SedimentEventControl,
    SedimentEventError,
    routing_gate_m,
)
from maple_syrup.storm import StormControl, initial_state

ROOT_SOLVERS = ("bisection", "newton")


CHAIN_NET_CROSS = ((2, 0), (1, 0))  # the highest cell -> its lower neighbour: receivers stay monotone, no pit


def fixture_xp(xp, *, wet=False, implementation="array", root_solver="bisection", max_steps=10000, net_cross=False,
               **control_kw):
    """`tests/phase6/test_complete_event.fixture` in namespace `xp`: a 3-cell chain, ksat 1e-5 m/s, 2 s of no
    rain (quiescent) or 10 s of 100 mm/h (wet), hold 2 s, reporting every 2 s. `net_cross=True` first gives the host
    bed the native conservative net-elevation precondition (`conservative_net_cross`, default False = the held fixture)."""
    bed, ctx = make_bed(chain_elevation(3), depth_m=.001 if wet else 0.)
    if net_cross:
        bed = conservative_net_cross(bed, ctx, *CHAIN_NET_CROSS)
    control = SedimentEventControl(storm=StormControl(max_dt_s=1, max_steps=max_steps, implementation=implementation,
                                                      root_solver=root_solver), **control_kw)
    s = setup_xp(bed, ctx, xp, ksat=1e-5, control=control)
    schedule = constant_rainfall(0, 10 if wet else 2, 100 if wet else 0)
    args = (s["state0"], s["ctx"], s["column"], s["field"], schedule, s["vegetation"], s["sediment"])
    kw = {"max_end_s": 180 if wet else 12, "control": control, "policy": CompletionPolicy(hold_s=2),
          "report_every_s": 2.}
    return args, kw


def assert_dry(handoff):
    d = handoff.dry_state
    for a in (d.storm.depth_m, d.storm.soil_water_m, d.storm.discharge_m2_s,
              d.bed.water.mobile_mass_by_cell_class_kg, d.sediment_velocity_m_s):
        assert not np.any(H(a))
    if d.phase is not None:
        assert np.all(H(d.phase.fraction)[..., 0] == 1.0) and not np.any(H(d.phase.position_m))


def assert_preparation_lifecycle(hyd, r):
    """The CUDA context is prepared once initially and once per ACTUAL net reroute of the event loop -- never per
    accounting commit (`record_commit_transfers` still runs for every commit: `n_commits_recorded`).

    `r` is the handoff's result, which also contains complete_event's terminal forced commit; that commit runs OUTSIDE the
    CUDA event loop (`complete_event._commit_and_reroute` is called directly), so even if it reroutes it prepares no
    context. The loop's reroute count is therefore derived from the FULL, untruncated `commit_log`: every rerouting
    record must match one lifecycle preparation (same `t_s`), and at most the LAST, forced record may be unprepared.
    Returns the number of loop reroutes."""
    assert len(r.commit_log) == r.n_commits and not hyd.summary()["log_truncated"], "commit/lifecycle log truncated"
    rerouted = [c for c in r.commit_log if c["rerouted"]]
    assert r.n_reroutes == len(rerouted)
    prepared = Counter(e["t_s"] for e in hyd.log if e["reason"] == "commit")
    wanted = Counter(c["t_s"] for c in rerouted)
    assert not (prepared - wanted), "a preparation without a rerouting commit"
    unprepared = wanted - prepared
    assert sum(unprepared.values()) <= 1
    if unprepared:  # only the terminal forced commit
        last = r.commit_log[-1]
        assert last["rerouted"] and last["forced"] and last["t_s"] in unprepared
    n_loop = len(rerouted) - sum(unprepared.values())
    assert hyd.n_preparations == 1 + n_loop == len(hyd.log)
    return n_loop


def compare_handoffs(ref, cand):
    """Pre-reset results at the declared bounds, dry states dry on both sides, dry beds at the sediment bound,
    reset accounting scalars equal to round-off."""
    compare_events(ref.progress.result, cand.progress.result)
    assert_dry(ref)
    assert_dry(cand)
    assert ref.dry_state.t_s == cand.dry_state.t_s and ref.progress.quiet == cand.progress.quiet
    for name in ("voxel_column", "active_layer"):
        close(getattr(ref.dry_state.bed, name).mass_kg, getattr(cand.dry_state.bed, name).mass_kg, SED_RTOL, SED_ATOL, name)
    close(ref.terminal_deposition_kg, cand.terminal_deposition_kg, SED_RTOL, SED_ATOL, "terminal_deposition")
    close(ref.pre_reset_surface_m, cand.pre_reset_surface_m, 2e-12, 1e-14, "pre_reset_surface")
    close(ref.pre_reset_soil_m, cand.pre_reset_soil_m, 2e-12, 1e-14, "pre_reset_soil")
    for key in ("surface_removed_m3", "soil_removed_m3", "water_added_m3"):
        assert ref.accounting[key] == pytest.approx(cand.accounting[key], rel=2e-12, abs=1e-14), key
    assert abs(cand.accounting["combined_water_residual_m3"]) <= cand.accounting["storm_water_budget"]["tolerance_m3"]
    assert cand.accounting["sediment_closure"]["closed"] and ref.accounting["sediment_closure"]["closed"]


# =====================================================================================================================
# CPU
# =====================================================================================================================
def test_completion_reductions_equal_the_direct_host_expressions():
    args, kw = fixture_xp(np, wet=True)
    state, ctx = args[0], args[1]
    policy = kw["policy"]
    qualifies, values = quiet_metrics(state, ctx, policy, 10.0)
    h, q, mobile = state.storm.depth_m, state.storm.discharge_m2_s, state.bed.water.mobile_mass_by_cell_class_kg
    area, dx = state.graph.dx_m ** 2, state.graph.dx_m
    assert values["max_depth_m"] == float(h.max()) and values["surface_volume_m3"] == float(h.sum() * area)
    assert values["max_unit_discharge_m2_s"] == float(q.max())
    assert values["outlet_discharge_m3_s"] == float(q[state.graph.outlet].sum() * dx)
    assert values["max_mobile_cell_class_kg"] == float(mobile.max())
    assert values["mobile_by_class_kg"] == mobile.sum(axis=(0, 1)).tolist()
    assert values["mobile_per_element_limit_kg"] == ctx.mass_resolution_kg and values["rainfall_finished"] is False
    assert not qualifies  # rain not finished
    # the packet is the identity on NumPy: the very reductions, no added arithmetic
    packet = host_packet(np, [h.max(), h.sum() * area], mobile.sum(axis=(0, 1)))
    assert packet[0] == h.max() and packet[1] == h.sum() * area and np.array_equal(packet[2:], mobile.sum(axis=(0, 1)))
    # an invalid physical state is refused exactly as before
    bad = replace(state, storm=replace(state.storm, depth_m=np.where(np.arange(3)[:, None] == 0, -1.0, h)))
    with pytest.raises(SedimentEventError, match="invalid physical state"):
        quiet_metrics(bad, ctx, policy, 10.0)
    # water budget equals its direct computation
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    r = paused.result
    budget = water_budget(paused)
    assert budget["rain_m3"] == float(r.cumulative_rain_m.sum() * area)
    assert budget["surface_final_m3"] == float(r.state.storm.depth_m.sum() * area)
    assert budget["soil_final_m3"] == float(r.state.storm.soil_water_m.sum() * area)
    assert budget["drainage_m3"] == float(r.cumulative_drainage_m.sum() * area)
    assert budget["export_m3"] == float(r.cumulative_export_m3) and budget["closed"]


def test_cuda_control_on_a_numpy_state_and_a_lifecycle_with_array_control_are_refused(monkeypatch):
    from maple_syrup import sediment_event

    args, kw = fixture_xp(np, implementation="cuda")
    saved = snapshot_any(args[0])

    def boom(*a, **k):
        raise AssertionError("a kernel or MAPLE call ran after a refused binding")

    # the namespace binding is refused inside the solver's input check, before any kernel or MAPLE call
    monkeypatch.setattr(sediment_event, "apply_bed_demand", boom)
    monkeypatch.setattr(sediment_event, "_prepare_cuda_context", boom)
    monkeypatch.setattr(sediment_event, "coupled_step", boom)
    with pytest.raises(SedimentEventError, match="CuPy"):
        complete_event(*args, **kw)
    compare_any(args[0], saved)
    args, kw = fixture_xp(np)
    with pytest.raises(SedimentEventError, match="hydraulics"):
        complete_event(*args, **kw, hydraulics=CudaHydraulics())
    with pytest.raises(SedimentEventError, match="CudaHydraulics"):
        complete_event(*args, **kw, hydraulics=object())


# =====================================================================================================================
# GPU
# =====================================================================================================================
def test_gpu_quiescent_storm_completes_with_dry_reset_without_sediment_loss(gpu):
    args, kw = fixture_xp(gpu, implementation="cuda")
    before = H(bed_inventory(args[0].bed)).copy()
    hyd = CudaHydraulics()
    out = complete_event(*args, **kw, hydraulics=hyd)
    assert isinstance(out, DryHandoff) and out.dry_state.t_s == 4
    assert out.dry_state.graph.xp is gpu and out.dry_state.phase.xp is gpu
    assert_dry(out)
    np.testing.assert_array_equal(before, H(bed_inventory(out.dry_state.bed)))
    assert out.accounting["soil_removed_m3"] == pytest.approx(.25 * .3 * .25 * 3)
    assert out.accounting["surface_removed_m3"] == 0 and out.accounting["combined_water_residual_m3"] == 0
    assert np.any(H(args[0].storm.soil_water_m) > 0)  # caller still wet, no mutation
    # preparations follow the ACTUAL loop reroutes (an accounting / terminal commit below the threshold prepares nothing)
    n_loop = assert_preparation_lifecycle(hyd, out.progress.result)
    assert n_loop == 0 and hyd.n_preparations == 1
    host_args, host_kw = fixture_xp(np)
    compare_handoffs(complete_event(*host_args, **host_kw), out)


@pytest.mark.parametrize("root_solver", ROOT_SOLVERS)
def test_gpu_wet_storm_completes_with_terrain_change_and_matches_the_host(gpu, root_solver):
    from maple.core.types.topographic_commit import committed_elevation_m

    args, kw = fixture_xp(gpu, wet=True, implementation="cuda", root_solver=root_solver)
    state0 = args[0]
    saved = snapshot_any(state0)
    hyd = CudaHydraulics()
    out = complete_event(*args, **kw, hydraulics=hyd)
    assert isinstance(out, DryHandoff)
    r = out.progress.result
    assert r.n_commits >= 1 and r.n_forced_commits >= 0 and float(H(r.by_class["actual_pickup"]).sum()) > 0.0
    dz = H(committed_elevation_m(out.dry_state.bed.committed_topography)) - state0.terrain.initial_committed_elevation_m
    assert float(np.abs(dz).max()) > 0.0  # the terrain actually changed
    assert_dry(out)
    assert_closed_any(r, state0, check_dry_settling=False)
    assert float(H(out.terminal_deposition_kg).sum()) >= 0.0 and out.accounting["sediment_closure"]["closed"]
    compare_any(state0, saved)  # the wet input is untouched
    # HELD routing (natural fixture): the loop's commits are accounting commits whose net displacement from the held routing
    # source stays below the actual threshold, so they publish a bed but prepare nothing; the preparation count follows the
    # ACTUAL reroutes (the positive net-crossing case is the next test)
    threshold = reroute_threshold_m(args[1])
    assert r.n_commits >= 1 and r.n_reroutes == 0 and r.max_unrouted_net_displacement_m < threshold
    assert all(not c["rerouted"] and c["net_displacement_m"] < c["reroute_threshold_m"] == threshold for c in r.commit_log)
    assert assert_preparation_lifecycle(hyd, r) == 0 and hyd.n_preparations == 1
    assert hyd.n_binding_hits == r.n_accepted_steps + r.n_rejected_attempts
    host_args, host_kw = fixture_xp(np, wet=True, root_solver=root_solver)
    compare_handoffs(complete_event(*host_args, **host_kw), out)


@pytest.mark.parametrize("root_solver", ROOT_SOLVERS)
def test_gpu_wet_storm_with_a_genuine_net_crossing_prepares_once_per_loop_reroute_and_matches_the_host(gpu, root_solver):
    """POSITIVE companion of the held-routing test above: the same wet chain after the native conservative net-elevation
    precondition (identical on host and device). The first accepted step must reroute for real and the CUDA lifecycle must
    prepare exactly once per loop reroute; the CPU/GPU physics, closure and exact-input checks are unchanged."""
    args, kw = fixture_xp(gpu, wet=True, implementation="cuda", root_solver=root_solver, net_cross=True)
    state0 = args[0]
    saved = snapshot_any(state0)
    hyd = CudaHydraulics()
    out = complete_event(*args, **kw, hydraulics=hyd)
    assert isinstance(out, DryHandoff)
    r = out.progress.result
    assert_dry(out)
    assert_closed_any(r, state0, check_dry_settling=False)
    assert out.accounting["sediment_closure"]["closed"]
    compare_any(state0, saved)  # the wet input is untouched
    first = r.commit_log[0]
    assert first["rerouted"] and first["net_displacement_m"] >= reroute_threshold_m(args[1])
    n_loop = assert_preparation_lifecycle(hyd, r)
    assert n_loop >= 1 and hyd.n_preparations == 1 + n_loop
    assert hyd.n_binding_hits == r.n_accepted_steps + r.n_rejected_attempts
    host_args, host_kw = fixture_xp(np, wet=True, root_solver=root_solver, net_cross=True)
    compare_handoffs(complete_event(*host_args, **host_kw), out)


@NUMBA_SKIP
def test_gpu_wet_storm_matches_the_numba_host_event(gpu):
    nb_args, nb_kw = fixture_xp(np, wet=True, implementation="numba", transport_implementation="array")
    nb = complete_event(*nb_args, **nb_kw)
    args, kw = fixture_xp(gpu, wet=True, implementation="cuda")
    dev = complete_event(*args, **kw)
    compare_handoffs(nb, dev)


def test_gpu_nonzero_terminal_settlement_is_a_genuine_maple_deposition(gpu):
    args, kw = fixture_xp(gpu, implementation="cuda")
    ctx, column = args[1], args[2]
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    assert isinstance(paused, EventProgress)
    r = paused.result
    pool = gpu.full(r.state.bed.active_layer.mass_kg.shape, 1e-12, dtype=np.float64)
    h = gpu.full(r.state.graph.shape, 1e-10, dtype=np.float64)
    bed = with_water(r.state.bed, ctx, h, pool)
    state = replace(r.state, bed=bed, storm=initial_state(r.state.graph, h, r.state.storm.soil_water_m, t_s=2))
    r = replace(r, state=state, initial_mobile_kg=pool.sum(axis=(0, 1)))
    p = replace(paused, result=r, policy=CompletionPolicy(hold_s=1), initial_surface_m3=float(H(h).sum() * .25),
                rainfall_end_s=0., quiet=QuietProgress(since_s=0., last_t_s=2., ready=True))
    initial_bed = H(bed_inventory(state.bed)).copy()
    out = finalize_event(p, ctx, column)
    np.testing.assert_array_equal(H(out.terminal_deposition_kg), H(pool))
    np.testing.assert_allclose(H(bed_inventory(out.dry_state.bed)) - initial_bed, H(pool).sum(axis=(0, 1)),
                               atol=out.accounting["terminal_sediment_tolerance_kg"], rtol=0)
    assert out.accounting["surface_removed_m3"] == float(H(h).sum() * .25)
    assert out.accounting["terminal_deposition_by_class_kg"] == H(pool).sum(axis=(0, 1)).tolist()
    assert out.accounting["sediment_closure"]["closed"]
    np.testing.assert_array_equal(H(p.result.state.bed.water.mobile_mass_by_cell_class_kg), H(pool))  # input kept
    assert out.dry_state.graph.xp is gpu and H(out.pre_reset_surface_m).max() == 1e-10


def test_gpu_failed_terminal_deposit_commit_or_reactivated_flow_leaves_the_wet_input_unchanged(gpu, monkeypatch):
    args, kw = fixture_xp(gpu, implementation="cuda")
    ctx, column = args[1], args[2]
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    ready = replace(paused, rainfall_end_s=0., policy=CompletionPolicy(hold_s=1),
                    quiet=QuietProgress(since_s=0., last_t_s=2., ready=True))
    saved = snapshot_any(ready.result.state)
    count = ready.result.state.bed.committed_topography.commit_count
    monkeypatch.setattr(module, "_commit_and_reroute", lambda *a, **k: None)
    with pytest.raises(SedimentEventError, match="commit was not performed"):
        finalize_event(ready, ctx, column)
    compare_any(ready.result.state, saved)
    monkeypatch.undo()
    # Reactivated flow after the terminal commit. A NON-rerouting terminal commit returns the caller's storm (discharge)
    # unchanged by design, so the discharge of its result is never read; the injection is meaningful only on a GENUINE
    # rerouted terminal commit. `ready_cross` is the same ready (quiet, wet-input) state whose bed first received the native
    # conservative net-elevation precondition, still held on the old routing source: the terminal forced commit then sees a
    # real net displacement >= the unchanged threshold and `commit.rerouted` is genuinely True (nothing is flagged by hand).
    state_r = ready.result.state
    cross_bed = conservative_net_cross(state_r.bed, ctx, *CHAIN_NET_CROSS)
    ready_cross = replace(ready, result=replace(ready.result, state=replace(state_r, bed=cross_bed)))
    threshold = reroute_threshold_m(ctx)
    assert routing_gate_m(state_r.bed, ctx, state_r.graph, state_r.routing_source_elevation_m) < threshold
    assert routing_gate_m(cross_bed, ctx, state_r.graph, state_r.routing_source_elevation_m) >= threshold
    saved_cross = snapshot_any(ready_cross.result.state)
    count_cross = cross_bed.committed_topography.commit_count
    original_commit = module._commit_and_reroute
    seen = []

    def commit_with_reactivated_flow(*a, **k):
        result = original_commit(*a, **k)
        seen.append(result.rerouted)
        q = gpu.array(result.storm.discharge_m2_s, copy=True)
        q[0, 0] = ready.policy.max_unit_discharge_m2_s * 2
        return replace(result, storm=replace(result.storm, discharge_m2_s=q))

    monkeypatch.setattr(module, "_commit_and_reroute", commit_with_reactivated_flow)
    with pytest.raises(SedimentEventError, match="invalidated completion"):
        finalize_event(ready_cross, ctx, column)
    assert seen == [True], "the injected terminal commit must be a genuine reroute"
    compare_any(ready_cross.result.state, saved_cross)
    assert ready_cross.result.state.bed.committed_topography.commit_count == count_cross
    monkeypatch.undo()
    # without the injection the very same genuine rerouted terminal commit completes the event
    handoff = finalize_event(ready_cross, ctx, column)
    assert handoff.progress.result.commit_log[-1]["rerouted"] and handoff.progress.result.commit_log[-1]["forced"]
    assert handoff.progress.result.n_reroutes == ready_cross.result.n_reroutes + 1
    compare_any(ready_cross.result.state, saved_cross)  # and still leaves the wet input untouched
    # the held (non-rerouting) terminal commit really ignores the commit result's discharge: a source-backed negative
    monkeypatch.setattr(module, "_commit_and_reroute", commit_with_reactivated_flow)
    seen.clear()
    finalize_event(ready, ctx, column)
    assert seen == [False]
    monkeypatch.undo()
    compare_any(ready.result.state, saved)
    # a refused terminal deposit (injected pool, MAPLE call failing) publishes nothing
    r = ready.result
    pool = gpu.full(r.state.bed.active_layer.mass_kg.shape, 1e-12, dtype=np.float64)
    h = gpu.full(r.state.graph.shape, 1e-10, dtype=np.float64)
    state = replace(r.state, bed=with_water(r.state.bed, ctx, h, pool),
                    storm=initial_state(r.state.graph, h, r.state.storm.soil_water_m, t_s=2))
    loaded = replace(ready, result=replace(r, state=state, initial_mobile_kg=pool.sum(axis=(0, 1))),
                     initial_surface_m3=float(H(h).sum() * .25))
    saved_loaded = snapshot_any(state)

    def refusing_demand(*a, **k):
        raise ValueError("MAPLE refused the terminal deposit (test)")

    monkeypatch.setattr(module, "apply_bed_demand", refusing_demand)
    with pytest.raises(ValueError, match="refused"):
        finalize_event(loaded, ctx, column)
    compare_any(state, saved_loaded)
    assert np.all(H(state.bed.water.mobile_mass_by_cell_class_kg) == 1e-12)


def test_gpu_duration_limit_is_not_a_dry_handoff(gpu):
    args, kw = fixture_xp(gpu, wet=True, implementation="cuda")
    before = H(args[0].bed.water.depth_m).copy()
    with pytest.raises(IncompleteEventError, match="duration"):
        complete_event(*args, **{**kw, "max_end_s": 4})
    np.testing.assert_array_equal(before, H(args[0].bed.water.depth_m))
    args, kw = fixture_xp(gpu, implementation="cuda", max_steps=3)
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    assert isinstance(paused, EventProgress)
    with pytest.raises(IncompleteEventError, match="max_steps"):
        complete_event(paused.result.state, *args[1:], **kw, continuation=paused)
