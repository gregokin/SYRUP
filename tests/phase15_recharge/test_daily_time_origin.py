"""Priority timing fix: the daily FULL-storm forcing is LOCAL to the native event (`LayeredEvent` adds `event_origin_s` to every schedule edge) with explicit absolute
physical start/end. INDEPENDENT controls at a NONZERO physical clock on the actual native Dense bed: the expected absolute edges/origin/end are literal analytic numbers
(t0 + D with D = 3600 P / I written out by hand), never derived from the helper under test; the native evolve to the rain end must debit the FULL P x active area (within the
existing derived roundoff bound) with no E/T during the storm and no phantom delay. Before the fix the event started at t0 + t0 (origin counted twice) and rained 0 m3 by t0 + D."""
from __future__ import annotations

import numpy as np
import pytest
from dcases import DAY, MM, RUN, Box, act, build, dw, ev, field_of, pet, rc, run_storm, wl
from storm_cases import layers

from maple_syrup.conservation import volume_roundoff_bound_m3

GRID = np.full((4, 6), 0.2)
T0 = 123.25
P_MM, I_MM_H, D_S = 1.0, 360.0, 10.0  # D = 3600 * 1 / 360 = 10 s, written by hand


def lineage(c):
    from maple_syrup import daily_water as d

    return d.start_daily_lineage(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0), run_digest=RUN)


def case(**kw):
    kw.setdefault("lay", layers(bottom="open"))
    kw.setdefault("theta", (0.1, 0.1))
    return build(GRID, t_s=T0, **kw)


def assert_event_edges(storm, t0):
    ev_ = storm.binding.event
    assert ev_.event_origin_s == t0
    assert [float(x) for x in ev_.absolute_edges_s] == [t0, t0 + D_S]  # literal analytic absolute edges
    assert ev_.rainfall_end_s == t0 + D_S
    assert [float(x) for x in ev_.schedule.edges_s] == [0.0, D_S]  # the schedule itself is LOCAL


def assert_full_p_debited_at_rain_end(c, storm, t0):
    res = wl.evolve_storm(c.lop, storm, expected_parent=wl.expect_parent(storm), until_s=t0 + D_S)
    led = res.state.payload.water.ledger
    expected = P_MM * MM * c.col.n_active * c.col.cell_area_m2
    bound = volume_roundoff_bound_m3(res.accepted_steps + 2, expected) + led.global_residual_bound_m3
    assert res.state.payload.water.t_s == t0 + D_S and abs(led.rain_m3 - expected) <= bound  # FULL P x active area by the native rain end
    life0, life1 = storm.lifetime, res.state.lifetime
    assert (life1.surface_evaporation_m3, life1.soil_evaporation_m3, life1.transpiration_m3) == (life0.surface_evaporation_m3, life0.soil_evaporation_m3, life0.transpiration_m3)  # no E/T during the storm
    assert res.accepted_steps < 200  # no phantom stretch of dry 1 s steps before the rain


def test_forcing_physical_edges_and_digest_at_a_nonzero_clock_are_analytic():
    plan = rc.plan_daily_rain(P_MM, I_MM_H, 0.5, run_digest=RUN, ordinal=0)
    f = plan.bind_forcing(T0)
    assert (f.t0_s, f.end_s, f.duration_s) == (T0, T0 + D_S, D_S) and [float(x) for x in f.schedule.edges_s] == [0.0, D_S]
    assert plan.bind_forcing(T0 + 1.0).digest != f.digest  # the physical start is part of the forcing identity
    assert f.amount_mm == pytest.approx(P_MM, rel=1e-12)


def test_storm_after_nothing_at_a_nonzero_clock_starts_at_t0_and_rains_full_p():
    c = case()
    s0 = lineage(c)
    plan = dw.next_plan(s0, P_MM, I_MM_H, 0.5)
    assert plan.selection == "storm"
    st = dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, control=ev.LayeredEventControl())
    assert_event_edges(st, T0)
    assert_full_p_debited_at_rain_end(c, st, T0)


def test_p_zero_day_then_full_storm_starts_at_the_closed_boundary_clock():
    c = case()
    s0 = lineage(c)
    _, r = act(c, s0, 0.0, pet_rate=2.0 ** -24)
    closed = wl.close_interevent(c.lop, r.state, expected_parent=wl.expect_parent(r.state))
    assert closed.t_s == T0 + DAY  # 86523.25
    plan = dw.next_plan(closed, P_MM, I_MM_H, 0.5)
    st = dw.begin_daily_storm(c.lop, closed, expected_parent=wl.expect_parent(closed), plan=plan, control=ev.LayeredEventControl())
    assert_event_edges(st, 86523.25)
    assert_full_p_debited_at_rain_end(c, st, 86523.25)


def test_recharge_day_then_next_full_storm_and_two_storms_separated_by_a_day():
    c = case()
    s = lineage(c)
    _, r = act(c, s, 0.25, t=0.5, pet_rate=2.0 ** -24)  # a successful recharge day
    closed = wl.close_interevent(c.lop, r.state, expected_parent=wl.expect_parent(r.state))
    t1 = 86523.25
    assert closed.t_s == t1
    plan1 = dw.next_plan(closed, P_MM, I_MM_H, 0.5)
    st1 = dw.begin_daily_storm(c.lop, closed, expected_parent=wl.expect_parent(closed), plan=plan1, control=ev.LayeredEventControl())
    assert_event_edges(st1, t1)
    done1 = run_storm(c, st1).state  # genuine completion under the DEFAULT quiet/completion rules
    assert done1.payload.completed and done1.t_s >= t1 + D_S
    day = dw.finish_daily_storm(c.lop, done1, expected_parent=wl.expect_parent(done1), plan=plan1, pet_terrain_m_s=pet(c, 2.0 ** -24))
    closed2 = wl.close_interevent(c.lop, day.state, expected_parent=wl.expect_parent(day.state))
    assert closed2.t_s == done1.t_s + DAY
    plan2 = dw.next_plan(closed2, P_MM, I_MM_H, 0.5)
    st2 = dw.begin_daily_storm(c.lop, closed2, expected_parent=wl.expect_parent(closed2), plan=plan2, control=ev.LayeredEventControl())
    assert [float(x) for x in st2.binding.event.absolute_edges_s] == [closed2.t_s, closed2.t_s + D_S] and st2.binding.event.rainfall_end_s == closed2.t_s + D_S
    assert_full_p_debited_at_rain_end(c, st2, closed2.t_s)


def test_capacity_fallback_at_a_nonzero_clock_runs_the_full_storm_at_t0_not_at_2_t0():
    c = case(lay=layers(bottom="closed"), theta=(0.5, 0.5))
    s0 = lineage(c)
    plan, r = act(c, s0, P_MM, t=2.0, i=I_MM_H)  # P <= T: recharge attempt; saturated closed column: capacity failure
    assert plan.selection == "recharge" and r.status == "capacity_failure"
    st = dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, fallback_of=r.fallback, control=ev.LayeredEventControl())
    assert_event_edges(st, T0)
    assert_full_p_debited_at_rain_end(c, st, T0)
    done = run_storm(c, st).state
    assert done.payload.completed
    out = dw.finish_daily_storm(c.lop, done, expected_parent=wl.expect_parent(done), plan=plan, pet_terrain_m_s=pet(c, 2.0 ** -23))
    assert out.receipt.occurrence_start_s == T0 and out.state.daily.ordinal == 1


def test_finish_refuses_a_storm_whose_native_edges_are_the_old_double_origin_forcing():
    """The ORIGINAL defect reproduced as a forgery: an absolute schedule [t0, t0 + D] under origin t0 (native edges [2 t0, 2 t0 + D]) must be refused at the daily finish."""
    from maple_syrup.rainfall import constant_rainfall

    c = case()
    s0 = lineage(c)
    plan = dw.next_plan(s0, P_MM, I_MM_H, 0.5)
    good = plan.bind_forcing(T0)
    twin = wl._mint(s0, daily=None)
    bad = wl.begin_storm(c.lop, twin, expected_parent=wl.expect_parent(twin), schedule=constant_rainfall(T0, T0 + D_S, I_MM_H), field=field_of(c), control=ev.LayeredEventControl())
    assert [float(x) for x in bad.binding.event.absolute_edges_s] == [2 * T0, 2 * T0 + D_S]  # the defect, observed on the actual native event
    pending = s0.daily.begin_storm(plan, good, fallback=False, t0_s=T0, event_sha256=bad.payload.event_sha256)
    fake = wl._mint(bad, daily=pending)
    done = run_storm(c, fake).state
    with pytest.raises(dw.DailyRefusal) as e:
        dw.finish_daily_storm(c.lop, done, expected_parent=wl.expect_parent(done), plan=plan, pet_terrain_m_s=pet(c, 0.0))
    assert e.value.code == "occurrence"
