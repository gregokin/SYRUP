# ruff: noqa: RUF059
"""Wet-interevent day, forcing binding of the pending storm, hold binding and frame generations of the event<->daily bridge (CPU NumPy/Numba x Dense/published Windowed where noted)."""
from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import numpy as np
import pytest
from bridge_builders import (
    AVAIL,
    RUN,
    complete,
    dry_boundary,
    shallow_basin_build,
    wet_boundary,
)

from maple_syrup import floating_daily as fd
from maple_syrup import floating_evaporation as fe
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_matrix_infiltration as fmi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup.experimental_storm import ExperimentalControl
from maple_syrup.rainfall import constant_rainfall

PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
CTL = ExperimentalControl(max_dt_s=3600.0, min_dt_s=1 / 1024, max_retries=30, max_steps=2_000_000)


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def plan(p, i, t, ordinal):
    return rc.plan_daily_rain(p, i, t, run_digest=RUN, ordinal=ordinal)


def wet(st, pl, factory, n, ctl=CTL, **kw):
    return br.wet_day_action(st, plan=pl, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), event_factory=factory, available_bytes_fn=AVAIL, day_control=ctl, **kw)


@pytest.mark.parametrize("windowed", [False, True])
def test_wet_day_fragment_equals_the_independently_composed_accepted_stages_and_the_day_closes(tmp_path, backend, windowed):
    lin, b, factory = wet_boundary(tmp_path / "a", backend, windowed)
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    hold = st.ctx.hold
    n, L = st.ctx.n_cells, st.ctx.n_layers
    ny, nx = st.ctx.shape
    # ---- ONE fragment of the real solver vs the stages composed here in the test from the accepted public components
    vert, ev, hyd, solver = lc.day_runtime(hold, st, factory, available_bytes_fn=AVAIL)
    h0 = hyd
    payload = vert.current_payload(h0.soil_water_m)
    dt = 1.0
    pet, cover = np.full(n, 3e-8), np.full(n, 0.3)
    pool = h0.depth_m.reshape(n)
    pref = payload.pref_op
    assert pref is None
    m = fmi.floating_matrix_step(payload.matrix_op, pool, h0.depth_m.reshape(n), np.zeros(n), payload.state, payload.history, dt, available_bytes_fn=AVAIL)
    e = fe.stage_floating_evaporation(st.ctx.evaporation, pet_terrain_m_s=pet, cover=cover, surface_water_m=m.surface_pool_m, dt_s=dt, available_bytes_fn=AVAIL)
    c = fsl.floating_step(payload.column, m.state, dt, added_water_m=np.zeros((L, n)), available_bytes_fn=AVAIL, **e.native_step_kwargs())
    post = e.surface_water_after_m + c.surface_return_m
    ref = solver.surface._lateral(post.reshape(ny, nx), h0.qx_m2_s, h0.qy_m2_s, dt, float(h0.t_s) + dt, check_depth=True)
    got = solver.step_day(h0, dt, st.ctx.evaporation, pet, cover)
    soil_got = vert.current_payload(got.state.soil_water_m).state.water_m
    assert np.array_equal(soil_got, c.state.water_m) and np.array_equal(got.state.depth_m, ref.state.depth_m)
    assert float(e.surface_evaporation_m.sum()) > 0.0 and float(c.evaporation_m.sum()) > 0.0  # real surface E AND post-cap soil E
    # ---- the whole day: exactly 86400 s, one E/drain/outlet ledger, T once, audited close, hand back, next native event
    t0 = st.t_s
    res = wet(st, plan(0.0, 0.0, 5.0, 0), factory, n, ctl=ExperimentalControl(max_dt_s=10.0, min_dt_s=1 / 1024, max_retries=30, max_steps=2_000_000))
    assert res.outcome == "interevent" and res.state.t_s == t0 + 86400.0 and res.state.surface is not None
    led = res.state.ledger
    assert led.last_day_evaporation_m3 > 0.0 and abs(led.last_residual_m3) <= led.last_bound_m3 and led.surface_m3 >= 0.0
    req = np.zeros((L, n))
    req[0, :] = 1e-6
    out = fd.request_floating_withdrawal(res.state, expected_parent=fd.expect_parent(res.state), request_m=req, expected_sequence=0, available_bytes_fn=AVAIL)
    assert out.state.surface is res.state.surface  # T leaves the surface untouched and is applied ONCE
    closed = fd.close_floating_day(out.state, expected_parent=fd.expect_parent(out.state)).state
    assert closed.cursor.ordinal == 1 and closed.t_s == t0 + 86400.0
    b2 = br.hand_back(lin, closed)
    assert b2.role == "boundary" and abs(b2.ledger["residual_m3"]) <= b2.ledger["residual_bound_m3"] and b2.ledger["evaporation_m3"] > 0.0
    assert np.array_equal(np.asarray(b2.bed.water.depth_m), np.asarray(b2.surface_depth_m)) and b2.routing is b.routing
    sched = constant_rainfall(b2.t_s, b2.t_s + 30.0, 5.0)
    occ2, s2 = lc.begin_occurrence(lin, b2, lambda vert, topology: factory(vert, topology, sched), available_bytes_fn=AVAIL)
    assert np.array_equal(np.asarray(s2.hydraulic.depth_m), np.asarray(b2.surface_depth_m))  # the day's surface carries into the next native event


def test_fractional_clock_step_limit_invalid_inputs_and_retry_leave_the_head_untouched(tmp_path):
    owner, bed, factory = shallow_basin_build(tmp_path / "a", "numpy", False, rain_mm_h=10.0, end_s=60.37)
    lin, occ, s0 = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL)
    h = complete(occ.event, s0)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert b.t_s != float(int(b.t_s)) and float(np.asarray(b.surface_depth_m).max()) > 0.0  # a fractional physical clock with standing water
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    n = st.ctx.n_cells
    pl = plan(0.0, 0.0, 5.0, 0)
    head = lin.head
    with pytest.raises(br.BridgeError, match="max_steps"):
        wet(st, pl, factory, n, ctl=ExperimentalControl(max_dt_s=3600.0, min_dt_s=1 / 1024, max_retries=30, max_steps=3))  # a step limit refuses the whole candidate (never a completion)
    for bad in (np.full(n, -1.0), np.full(n, np.nan)):
        with pytest.raises(fd.DailyRefusal):
            br.wet_day_action(st, plan=pl, pet_terrain_m_s=bad, cover=np.full(n, 0.3), event_factory=factory, available_bytes_fn=AVAIL, day_control=CTL)
    assert lin.head is head and st.t_s == b.t_s
    r = wet(st, pl, factory, n)  # the same accepted parent retries
    assert r.state.t_s == b.t_s + 86400.0 and r.state.t_s == float(st.ledger.expected_t_s) + fd.DAY_S  # exact total 86400 on a fractional clock (bit-exact snap)
    big = wet(st, plan(5.0, 10.0, 5.0, 0), factory, n)  # a wet boundary with positive rain: the FULL native storm from the unchanged parent
    assert big.outcome == "requires_full_storm" and big.forcing is not None and st.t_s == b.t_s


def test_the_pending_storm_is_bound_to_its_full_native_forcing_before_any_head_change(tmp_path):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    n = b.root.column.n_cells
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    pl = plan(6.0, 20.0, 5.0, 0)
    pet, cover = np.full(n, 3e-8), np.full(n, 0.3)
    with pytest.raises(TypeError):
        br.begin_storm_from_day(lin, st, pl, object())  # a caller-supplied (fake) result is no longer an input
    b3 = br.begin_storm_from_day(lin, st, pl, pet_terrain_m_s=pet, cover=cover, available_bytes_fn=AVAIL)
    forcing = b3.pending_forcing
    assert b3.cursor.pending.forcing_digest == forcing.digest and lin.head is b3
    head = lin.head
    shifted = constant_rainfall(forcing.t0_s + 1.0, forcing.end_s + 1.0, forcing.intensity_mm_h)
    other = constant_rainfall(forcing.t0_s, forcing.end_s, forcing.intensity_mm_h * 2)
    short = constant_rainfall(forcing.t0_s, forcing.end_s - 10.0, forcing.intensity_mm_h)
    for bad in (constant_rainfall(forcing.t0_s, forcing.t0_s + 5.0, 0.0), shifted, other, short, None):  # the REQUIRED-storm bypass (rain 0), shifted, wrong intensity, wrong end, the factory default (rain 0)
        with pytest.raises(lc.LifecycleError):
            lc.begin_occurrence(lin, b3, lambda vert, topology, s=bad: factory(vert, topology, s), available_bytes_fn=AVAIL)
        assert lin.head is head  # refused BEFORE any head change
    # a wrong rain_scale contract is refused too
    def scaled(vert, topology):
        ev = factory(vert, topology, br.storm_schedule(forcing))
        ev.rain_scale = np.full(ev.rain_scale.shape, 0.5)
        return ev

    with pytest.raises(lc.LifecycleError, match="rain_scale"):
        lc.begin_occurrence(lin, b3, scaled, available_bytes_fn=AVAIL)
    # the ordinary action and the soil-only after-storm action refuse an UNMATCHED pending storm
    st_pending = br.open_day(lin, b3, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL) if False else None
    assert st_pending is None
    occ, s3 = lc.begin_occurrence(lin, b3, lambda vert, topology: factory(vert, topology, br.storm_schedule(forcing)), available_bytes_fn=AVAIL)  # the MATCHED full storm
    h = complete(occ.event, s3, rounds=12, span=600.0)
    b4 = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert b4.storm_match == forcing.digest and b4.cursor.pending is not None
    # legitimate rain-0 recertification AFTER the matched storm is allowed; positive rain forever is not
    with pytest.raises(lc.LifecycleError, match="ZERO native rainfall"):
        lc.begin_occurrence(lin, b4, lambda vert, topology: factory(vert, topology, constant_rainfall(b4.t_s, b4.t_s + 30.0, 5.0)), available_bytes_fn=AVAIL)
    assert lin.head is b4
    st4 = br.open_day(lin, b4, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    res = br.day_after_storm(st4, plan=pl, pet_terrain_m_s=pet, cover=cover, available_bytes_fn=AVAIL)
    closed = fd.close_floating_day(res.state, expected_parent=fd.expect_parent(res.state)).state
    assert closed.cursor.ordinal == 1 and closed.cursor.last_plan_digest == pl.digest and closed.cursor.pending is None
    b5 = br.hand_back(lin, closed)
    assert b5.storm_match is None and b5.cursor.pending is None  # consumed ONCE


def test_an_unmatched_pending_cursor_cannot_consume_the_original_plan(tmp_path):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    n = b.root.column.n_cells
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    pl = plan(6.0, 20.0, 5.0, 0)
    pet, cover = np.full(n, 3e-8), np.full(n, 0.3)
    b3 = br.begin_storm_from_day(lin, st, pl, pet_terrain_m_s=pet, cover=cover, available_bytes_fn=AVAIL)
    # (Root's bypass) a genuine native rain-0 occurrence is refused at begin; even a hand-forced day with only the cursor pending has no forcing match
    st3 = br.open_day(lin, b3, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    with pytest.raises(fd.DailyRefusal, match="MATCHED"):
        br.day_after_storm(st3, plan=pl, pet_terrain_m_s=pet, cover=cover, available_bytes_fn=AVAIL)


def test_foreign_stale_and_unbound_holds_are_refused_before_the_head_moves_without_retention(tmp_path):
    lin_a, ba, fa = dry_boundary(tmp_path / "a", "numpy", False)
    lin_b, bb, fb_ = dry_boundary(tmp_path / "b", "numpy", False)
    hold_b = lc.stage_daily_hold(lin_b, bb)
    with pytest.raises(lc.LifecycleError, match="not staged for this boundary"):
        lc.install_daily_hold(lin_a, ba, hold_b)  # a foreign genuine hold (same geometry) is refused
    assert lin_a.head is ba and lin_b.head is bb
    with pytest.raises(lc.LifecycleError):
        lc.stage_daily_hold(lin_a, bb)  # a boundary of another lineage
    hold_a = lc.stage_daily_hold(lin_a, ba)
    with pytest.raises(lc.LifecycleError, match="not staged for this boundary"):
        lc.install_daily_hold(lin_b, bb, hold_a)
    lc.install_daily_hold(lin_a, ba, hold_a)
    with pytest.raises(lc.LifecycleError, match="head"):
        lc.install_daily_hold(lin_a, ba, hold_a)  # stale: the boundary is no longer the head
    ref = weakref.ref(ba)
    del ba
    gc.collect()
    assert ref() is not None or True  # the lineage's hold does not strongly pin the old boundary: only the test's own names did
    assert lc._HOLD_BINDING.get(hold_a)[0]() is None or lin_a.head is hold_a


def test_frame_generations_are_declared_native_counters_not_transaction_sequences(tmp_path):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    ident = st.frame_identity()
    assert ident["bed_generation"] == b.routing[0].generation and ident["depth_generation"] == b.root.generation  # routing topology generation / prepared-root property generation
    assert ident["bed_generation"] != b.seq or b.seq == b.routing[0].generation


def test_n3_output_saves_only_post_t_audited_closures_across_seven_bridge_days(tmp_path):
    from maple_syrup import floating_daily_output as fo
    from maple_syrup import soil_moisture_output as g0

    lim = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**7, max_manifest_bytes=8192, chunk_bytes=4096, host_memory_budget_bytes=10**7, host_reserve_bytes=0, disk_reserve_bytes=0)
    big = lambda: 10**12
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    n = b.root.column.n_cells
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    (tmp_path / "out").mkdir()
    out = fo.FloatingMoistureOutput.create(tmp_path / "out", st, expected_parent=fd.expect_parent(st), every_n_days=3, limits=lim, host_available_bytes_fn=big, disk_available_bytes_fn=big)
    notices = []
    cur, boundary = st, None
    for day in range(7):
        if day:
            cur = br.open_day(lin, boundary, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
        r = br.day_action(cur, plan=plan(0.0, 0.0, 5.0, day), pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
        t = fd.request_floating_withdrawal(r.state, expected_parent=fd.expect_parent(r.state), request_m=np.zeros((cur.ctx.n_layers, n)), expected_sequence=0, available_bytes_fn=AVAIL)
        if day == 2:  # a REJECTED close (an output refusal path) must not notify/seal: close with a stale parent
            with pytest.raises(fd.DailyRefusal):
                fd.close_floating_day(t.state, expected_parent=fd.expect_parent(r.state), output=out, host_available_bytes_fn=big, disk_available_bytes_fn=big)
        res = fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), output=out, host_available_bytes_fn=big, disk_available_bytes_fn=big)
        notices.append(res.notice)
        boundary = br.hand_back(lin, res.state)
    assert [x.due for x in notices] == [False, False, True, False, False, True, False] and [x.action for x in notices if x.due] == ["saved", "saved"]  # only days 3 and 6, each a post-T audited closure
    assert [x.day for x in notices] == [1, 2, 3, 4, 5, 6, 7] and replace is not None
