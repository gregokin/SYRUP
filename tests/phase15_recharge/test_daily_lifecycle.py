"""Phase 1.5 S2 lifecycle controls on the ACTUAL native Dense case (CPU): capacity failure and exact discard, full-storm fallback versus a DIRECT
full storm, partial storms, single daily update / single movement, wet and refusal envelopes, ledgers, memory, T after water, bypass guards."""
from __future__ import annotations

import dataclasses
import weakref

import numpy as np
import pytest
from dcases import DAY, MM, RUN, STORM_CONTROL, Box, act, build, dw, ev, field_of, fingerprint, host, lifetime, pet, rc, root_layers, run_storm, se, ss, start, uniform_soil, \
    with_soil, wl
from storm_cases import bowl, layers

from maple_syrup import interevent_transpiration as it
from maple_syrup import interevent_water as iw
from maple_syrup.surface_routing import HydraulicStepRejected

GRID = np.full((4, 6), 0.2)
SATURATED = [50.0, 50.0]  # default layers() sat = 0.5 over 0.25 m / 0.75 m: theta 0.5 -> full


def closed_case(theta=(0.25, 0.25), **kw):
    return build(GRID, lay=layers(bottom="closed"), theta=theta, **kw)


def assert_same(a, b):
    assert fingerprint(a) == fingerprint(b)


# --- capacity failure: distinct outcome, exact discard -----------------------------------------------------------------------------
def test_saturated_closed_column_is_a_capacity_failure_with_the_parent_exactly_unchanged():
    c = closed_case(theta=(0.5, 0.5))
    s0 = start(c)
    before = fingerprint(s0)
    cursor = s0.daily
    plan, r = act(c, s0, 2.0)
    assert r.status == "capacity_failure" and r.state is None and r.receipt is None and type(r.fallback) is dw.CapacityFailure
    assert fingerprint(s0) == before and s0.daily is cursor and s0.daily.ordinal == 0  # nothing adopted, nothing consumed
    # it is a physical outcome, not an error: an identical retry gives the identical outcome (the trial is pure)
    _, r2 = act(c, s0, 2.0)
    assert r2.status == "capacity_failure" and r2.fallback.digest == r.fallback.digest


def test_one_saturated_active_column_forces_the_whole_occurrence_to_fall_back_heterogeneous():
    c = closed_case()
    cells = [[25.0, 25.0]] * c.n  # not valid mm for this profile, rescaled below via theta
    del cells
    full = [0.5 * 0.25 * 1000.0, 0.5 * 0.75 * 1000.0]  # exactly saturated storages in mm for the default layer thicknesses
    mix = [[0.25 * 0.25 * 1000.0, 0.25 * 0.75 * 1000.0]] * c.n
    mix[7] = full
    s0 = with_soil(c, mix)
    s0 = dw.start_daily_lineage(c.lop, s0, expected_parent=wl.expect_parent(s0), run_digest=RUN)
    before = fingerprint(s0)
    _, r = act(c, s0, 0.5)
    assert r.status == "capacity_failure" and fingerprint(s0) == before  # one failing column rejects EVERY cell's rain
    ok = with_soil(c, [mix[0]] * c.n)
    ok = dw.start_daily_lineage(c.lop, ok, expected_parent=wl.expect_parent(ok), run_digest=RUN)
    assert act(c, ok, 0.5)[1].status == "complete"


def test_non_capacity_errors_are_refusals_never_a_fallback():
    c = closed_case(theta=(0.25, 0.25))
    s0 = start(c)
    bad_pet = pet(c, 1.0).copy()
    bad_pet[0] = float("nan")
    plan = dw.next_plan(s0, 2.0, 10.0, 5.0)
    before = fingerprint(s0)
    with pytest.raises(dw.DailyRefusal) as e:
        dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, pet_terrain_m_s=bad_pet)
    assert e.value.code in ("step_refused", "entry") and fingerprint(s0) == before


# --- full-storm fallback == DIRECT full storm ----------------------------------------------------------------------------------------
def test_capacity_fallback_equals_a_direct_full_storm_from_the_same_state_with_full_p():
    cd, cn = closed_case(theta=(0.5, 0.5)), closed_case(theta=(0.5, 0.5))  # identical independent builds: daily lineage and a plain Phase 1 lineage
    s_daily, s_direct = start(cd), cn.s0
    plan, r = act(cd, s_daily, 2.0, t=5.0, i=360.0)  # 2 mm at 360 mm/h = 20 s
    assert r.status == "capacity_failure"
    storm = dw.begin_daily_storm(cd.lop, s_daily, expected_parent=wl.expect_parent(s_daily), plan=plan, fallback_of=r.fallback, control=STORM_CONTROL)
    assert storm.role == "storm" and storm.daily.pending.branch == "storm_fallback" and storm.daily.ordinal == 0
    forcing = plan.bind_forcing(0.0)
    direct = wl.begin_storm(cn.lop, s_direct, expected_parent=wl.expect_parent(s_direct), schedule=forcing.schedule, field=field_of(cn), control=STORM_CONTROL)
    a, b = run_storm(cd, storm), run_storm(cn, direct)
    assert a.complete and b.complete and a.state.payload.event_sha256 == b.state.payload.event_sha256
    sa, sb = a.state, b.state
    ra, rb = raw(sa), raw(sb)
    assert all(ra[k].tobytes() == rb[k].tobytes() if isinstance(ra[k], np.ndarray) else ra[k] == rb[k] for k in ra)  # arrays, time, J, intake, faces
    assert sa.payload.water.ledger == sb.payload.water.ledger and sa.lifetime == sb.lifetime  # every ledger (full P, once)
    led = sa.payload.water.ledger
    assert led.rain_m3 == pytest.approx(2.0 * MM * cd.n * 0.25, rel=1e-9)  # the FULL P, never a partial recharge
    # then the ONE day without recharge == the plain Phase 1 entry + one general 86400 s step
    done = dw.finish_daily_storm(cd.lop, sa, expected_parent=wl.expect_parent(sa), plan=plan, pet_terrain_m_s=pet(cd, 2.0 ** -23))
    ie = wl.enter_interevent_phase(cn.lop, sb, expected_parent=wl.expect_parent(sb))
    cand = it.general_interevent_water_step(ie.binding.gop, ie.payload, pet(cn, 2.0 ** -23), DAY, phase="interevent")
    da, db = raw(done.state), {"t": float(cand.state.t_s), "soil": host(cand.state.soil.water_m), "surface": host(cand.state.surface_m)}
    assert da["t"] == db["t"] and da["soil"].tobytes() == db["soil"].tobytes() and da["surface"].tobytes() == db["surface"].tobytes()
    assert done.state.daily.ordinal == 1 and done.state.daily.last_branch == "storm_fallback" and done.state.daily.pending is None
    assert done.receipt.recharge_m3 == 0.0 and done.receipt.occurrence_start_s == 0.0 and done.receipt.t_start_s == sa.t_s
    assert done.state.t_s == sa.t_s + DAY  # actual storm duration PLUS one day


def raw(s):
    from dcases import raw as _raw
    return _raw(s)


def test_storm_plan_p_above_t_uses_the_full_native_storm_branch_and_one_day():
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    s0 = start(c)
    plan = dw.next_plan(s0, 2.0, 360.0, 1.0)
    assert plan.selection == "storm"
    with pytest.raises(dw.DailyRefusal, match="fallback"):
        dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, fallback_of=object())  # a storm plan has no fallback receipt
    st = dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, control=STORM_CONTROL)
    assert st.daily.pending.branch == "storm" and st.daily.pending.event_sha256 == st.payload.event_sha256
    res = run_storm(c, st)
    assert res.complete
    out = dw.finish_daily_storm(c.lop, res.state, expected_parent=wl.expect_parent(res.state), plan=plan, pet_terrain_m_s=pet(c, 0.0))
    assert out.state.daily.ordinal == 1 and out.state.daily.last_branch == "storm"
    assert out.state.t_s == res.state.t_s + DAY and out.state.payload.ledger.hydrology.rain_m3 == pytest.approx(2.0 * MM * c.n * 0.25, rel=1e-9)
    audit = lifetime(c, out.state)
    assert abs(audit["residual_m3"]) <= audit["bound_m3"]


def test_a_partial_storm_cannot_enter_the_daily_step_and_stays_unfinished():
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    s0 = start(c)
    plan = dw.next_plan(s0, 2.0, 360.0, 1.0)
    st = dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, control=STORM_CONTROL)
    part = wl.evolve_storm(c.lop, st, expected_parent=wl.expect_parent(st), until_s=st.t_s + 40_000.0, max_accepted_steps=1)
    assert not part.complete and part.state.t_s < 40_000.0  # a bound stop is not completion
    t, ledger = part.state.t_s, part.state.payload.water.ledger
    with pytest.raises(dw.DailyRefusal) as e:
        dw.finish_daily_storm(c.lop, part.state, expected_parent=wl.expect_parent(part.state), plan=plan, pet_terrain_m_s=pet(c, 1.0))
    assert e.value.code == "storm_incomplete"
    assert part.state.t_s == t and part.state.payload.water.ledger == ledger and part.state.daily.pending is not None and part.state.daily.ordinal == 0
    with pytest.raises(wl.LifecycleError, match="daily lineage"):
        wl.enter_interevent_phase(c.lop, part.state, expected_parent=wl.expect_parent(part.state))  # the old entry cannot bypass the cursor
    more = run_storm(c, part.state)  # the unfinished storm progresses through the existing bounded native loop and then completes
    assert more.complete


# --- single daily update, single movement --------------------------------------------------------------------------------------------
def test_one_hydro_step_one_column_movement_and_no_subdaily_evolution(monkeypatch):
    c = closed_case(theta=(0.25, 0.25), cover=0.3)
    s0 = start(c)
    calls = {"recharge": 0, "plain": 0, "trial": 0}
    real_r, real_p, real_t = ss.step_with_recharge, ss.step, iw._physical_trial

    def count(name, fn):
        def wrapper(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return wrapper

    monkeypatch.setattr(ss, "step_with_recharge", count("recharge", real_r))
    monkeypatch.setattr(ss, "step", count("plain", real_p))
    monkeypatch.setattr(iw, "_physical_trial", count("trial", real_t))
    monkeypatch.setattr(wl, "evolve_interevent", lambda *a, **k: (_ for _ in ()).throw(AssertionError("evolve_interevent must not run on the daily path")))
    _, r = act(c, s0, 2.0, pet_rate=2.0 ** -24)
    assert calls == {"recharge": 1, "plain": 0, "trial": 1}  # ONE trial, ONE pulse-aware movement, no second pass
    assert r.state.hydro_count == s0.hydro_count + 1 and r.state.payload.generation - r.state.payload.entry_generation == 1


def test_p_zero_runs_the_daily_e_and_soil_update_exactly_like_the_plain_phase1_day(monkeypatch):
    cd, cn = build(GRID, lay=layers(bottom="open"), theta=(0.2, 0.2), cover=0.3), build(GRID, lay=layers(bottom="open"), theta=(0.2, 0.2), cover=0.3)
    plan, r = act(cd, start(cd), 0.0, t=5.0, pet_rate=2.0 ** -22)
    ie = wl.enter_interevent_from_boundary(cn.lop, cn.s0, expected_parent=wl.expect_parent(cn.s0))
    cand = it.general_interevent_water_step(ie.binding.gop, ie.payload, pet(cn, 2.0 ** -22), DAY, phase="interevent")
    assert plan.selection == "no_rain" and r.receipt.recharge_m3 == 0.0 and r.receipt.soil_evaporation_m3 > 0.0  # E ran (no precipitation operation)
    assert host(r.state.payload.soil.water_m).tobytes() == host(cand.state.soil.water_m).tobytes() and r.state.t_s == cand.state.t_s
    assert type(cand.receipt) is iw.InterEventStepReceipt  # the ordinary receipt type and fields are unchanged
    assert r.state.payload.ledger.hydrology.rain_m3 == 0.0


def test_daily_e_diffusion_and_drainage_happen_once_equal_to_the_independent_pulse_column_step():
    lay = layers(bottom="open", diffusivity=1e-9)
    c = build(GRID, lay=lay, theta=(0.3, 0.2), cover=0.4)
    s0 = start(c)
    rate = 2.0 ** -23
    _, r = act(c, s0, 2.0, pet_rate=rate)
    from maple_syrup.water_lifecycle import _private_cover

    e = se.stage_evaporation(c.lop.evaporation, pet_terrain_m_s=pet(c, rate), cover=_private_cover(c.lop), surface_water_m=np.zeros(c.n), dt_s=DAY)
    ref = ss.step_with_recharge(c.col, s0.payload.soil, DAY, recharge_depth_m=2.0 * MM, **e.native_step_kwargs())
    assert host(r.state.payload.soil.water_m).tobytes() == host(ref.state.water_m).tobytes()  # dry surface: stage 1-2 are identities
    assert r.receipt.drainage_m3 == ref.budget.drainage_m3 and r.receipt.soil_evaporation_m3 == ref.budget.evaporation_m3
    # the double-count controls: E and drainage in the ledger are those of the ONE step
    led = r.state.payload.ledger.hydrology
    assert led.soil_evaporation_m3 == ref.budget.evaporation_m3 and led.drainage_m3 == ref.budget.drainage_m3


# --- retained wet state, history, ledgers --------------------------------------------------------------------------------------------
def test_dry_day_leaves_matrix_history_and_intake_bitwise_unchanged_and_recharge_is_not_matrix_intake():
    c = closed_case(theta=(0.25, 0.25))
    s0 = start(c)
    _, r = act(c, s0, 2.0)
    h0, h1 = s0.payload.history, r.state.payload.history
    assert host(h0.wetting_m).tobytes() == host(h1.wetting_m).tobytes() and host(h0.cumulative_intake_m).tobytes() == host(h1.cumulative_intake_m).tobytes()
    assert r.receipt.matrix_intake_m3 == 0.0 and r.receipt.preferential_intake_m3 == 0.0 and r.receipt.recharge_m3 > 0.0


def test_quiet_retained_pond_day_advances_modeled_intake_separately_from_recharge_and_closes():
    c = build(bowl(3, 0.02), (), lay=layers(bottom="open"), depth=0.01, theta=(0.25, 0.25), wetting=0.001, cumulative=0.002)
    s0 = start(c)
    _, r = act(c, s0, 2.0)
    assert r.status == "complete"
    assert r.receipt.preferential_intake_m3 > 0.0 and r.receipt.matrix_intake_m3 > 0.0  # the retained pond infiltrates (modeled intake) ...
    assert r.receipt.recharge_m3 == pytest.approx(2.0 * MM * c.col.n_active * 0.25, rel=1e-12)  # ... and recharge is a separate external source
    h0, h1 = s0.payload.history, r.state.payload.history
    assert host(h1.cumulative_intake_m).sum() > host(h0.cumulative_intake_m).sum()  # actual pond intake may legitimately advance J / intake
    led = r.state.payload.ledger.hydrology
    assert led.rain_m3 == r.receipt.recharge_m3 and led.matrix_intake_m3 == r.receipt.matrix_intake_m3
    audit = lifetime(c, r.state)
    assert abs(audit["residual_m3"]) <= audit["bound_m3"] and audit["rain_m3"] == pytest.approx(r.receipt.recharge_m3, rel=1e-12)


def test_unquiet_boundary_is_an_entry_refusal_not_a_storm_fallback():
    from storm_cases import east_outlets, plane

    c = build(plane(4, 6), east_outlets(4, 6), lay=layers(bottom="open"), depth=0.005, theta=(0.25, 0.25))
    s0 = start(c)
    before = fingerprint(s0)
    plan = dw.next_plan(s0, 2.0, 10.0, 5.0)
    with pytest.raises(dw.DailyRefusal) as e:
        dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, pet_terrain_m_s=pet(c, 0.0))
    assert e.value.code == "entry" and "quiet" in str(e.value) and fingerprint(s0) == before and s0.daily.ordinal == 0


def test_router_rejection_of_the_one_day_step_is_a_refusal_with_no_retry_and_the_parent_exact(monkeypatch):
    """INJECTED control (a quiet-state day step that the actual router rejects was not found in the scanned envelope; see the report)."""
    c = closed_case(theta=(0.25, 0.25))
    s0 = start(c)
    before, calls = fingerprint(s0), []
    router_cls = type(c.lop.router)
    real = router_cls.step

    def reject(self, *a, **k):
        calls.append(a[1] if len(a) > 1 else k)
        raise HydraulicStepRejected("injected late CFL")

    monkeypatch.setattr(router_cls, "step", reject)
    plan = dw.next_plan(s0, 2.0, 10.0, 5.0)
    with pytest.raises(dw.DailyRefusal) as e:
        dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, pet_terrain_m_s=pet(c, 2.0 ** -24))
    assert e.value.code == "router_step_rejected" and len(calls) == 1  # ONE attempt at the full day: no halving, no storm fallback
    assert fingerprint(s0) == before and s0.daily.ordinal == 0 and s0.hydro_count == 0 and s0.t_s == 0.0
    monkeypatch.setattr(router_cls, "step", real)
    assert act(c, s0, 2.0)[1].status == "complete"  # the same parent retries cleanly


# --- occurrence binding ---------------------------------------------------------------------------------------------------------------
def test_every_branch_consumes_exactly_one_ordinal_and_duplicates_gaps_and_foreign_runs_are_refused():
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    s = start(c)
    for k, p in enumerate((0.0, 2.0, 5.0)):
        plan, r = act(c, s, p)
        assert r.state.daily.ordinal == k + 1
        s = wl.close_interevent(c.lop, r.state, expected_parent=wl.expect_parent(r.state))
        assert s.daily.ordinal == k + 1 and s.role == "boundary"
        with pytest.raises(dw.DailyRefusal) as e:  # the SAME plan again: duplicate
            dw.daily_water_action(c.lop, s, expected_parent=wl.expect_parent(s), plan=plan, pet_terrain_m_s=pet(c, 0.0))
        assert e.value.code == "occurrence"
    gap = rc.plan_daily_rain(1.0, 10.0, 5.0, run_digest=RUN, ordinal=5)
    other = rc.plan_daily_rain(1.0, 10.0, 5.0, run_digest="cd" * 32, ordinal=3)
    for bad in (gap, other):
        with pytest.raises(dw.DailyRefusal) as e:
            dw.daily_water_action(c.lop, s, expected_parent=wl.expect_parent(s), plan=bad, pet_terrain_m_s=pet(c, 0.0))
        assert e.value.code == "occurrence"
    forged = dataclasses.replace(dw.next_plan(s, 1.0, 10.0, 5.0), depth_m=0.5)
    with pytest.raises(dw.DailyRefusal):
        dw.daily_water_action(c.lop, s, expected_parent=wl.expect_parent(s), plan=forged, pet_terrain_m_s=pet(c, 0.0))


def test_stale_equal_digest_parent_branch_is_a_different_physical_parent():
    c = closed_case(theta=(0.5, 0.5))
    s0 = start(c)
    twin = wl._mint(s0)  # an alternate physical branch with EVERY scalar (and the digest) equal
    assert twin.digest == s0.digest and twin is not s0
    plan = dw.next_plan(s0, 2.0, 10.0, 5.0)
    with pytest.raises(dw.DailyRefusal) as e:
        dw.daily_water_action(c.lop, twin, expected_parent=wl.expect_parent(s0), plan=plan, pet_terrain_m_s=pet(c, 0.0))
    assert e.value.code == "state"
    _, r = act(c, s0, 2.0)
    assert r.status == "capacity_failure"
    with pytest.raises(dw.DailyRefusal) as e:  # the fallback authority of parent A is not accepted for the equal-digest twin B
        dw.begin_daily_storm(c.lop, twin, expected_parent=wl.expect_parent(twin), plan=plan, fallback_of=r.fallback, control=STORM_CONTROL)
    assert e.value.code == "occurrence"
    with pytest.raises(dw.DailyRefusal):  # a recharge plan has no storm path without its own CapacityFailure
        dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, control=STORM_CONTROL)
    ok = dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, fallback_of=r.fallback, control=STORM_CONTROL)
    assert ok.role == "storm"


def test_capacity_failure_does_not_hold_the_original_boundary_alive():
    import gc

    c = closed_case(theta=(0.5, 0.5))
    s0 = start(c)  # a daily-lineage boundary that nothing else references (the build namespace holds the Phase 1 start state only)
    _, r = act(c, s0, 2.0)
    assert r.fallback._ref() is s0
    held = [getattr(r.fallback, n) for n in r.fallback.__slots__ if n not in ("_ref", "__weakref__")]
    assert not any(isinstance(v, (wl.LifecycleState, np.ndarray)) for v in held)  # scalars only: no arrays, no bed, no parent
    del s0
    gc.collect()
    assert r.fallback._ref() is None  # the receipt never kept the original boundary (or its bed) alive


def test_forged_storm_forcing_is_refused_at_the_daily_step_even_with_a_matching_cursor():
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    s0 = start(c)
    plan = dw.next_plan(s0, 2.0, 360.0, 1.0)
    forced = rc.plan_daily_rain(4.0, 360.0, 1.0, run_digest=RUN, ordinal=0)  # a DIFFERENT storm (2x P) actually run by the engine
    twin = wl._mint(s0, daily=None)
    st = wl.begin_storm(c.lop, twin, expected_parent=wl.expect_parent(twin), schedule=forced.bind_forcing(0.0).schedule, field=field_of(c), control=STORM_CONTROL)
    pending = s0.daily.begin_storm(plan, plan.bind_forcing(0.0), fallback=False, t0_s=0.0, event_sha256=st.payload.event_sha256)
    fake = wl._mint(st, daily=pending)
    done = run_storm(c, fake).state
    with pytest.raises(dw.DailyRefusal) as e:
        dw.finish_daily_storm(c.lop, done, expected_parent=wl.expect_parent(done), plan=plan, pet_terrain_m_s=pet(c, 0.0))
    assert e.value.code == "occurrence" and done.daily.ordinal == 0


# --- old-API bypass guards -----------------------------------------------------------------------------------------------------------
def test_old_free_transitions_are_refused_on_a_daily_lineage_and_unchanged_without_a_cursor():
    from maple_syrup.rainfall import constant_rainfall

    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    s0 = start(c)
    ep = wl.expect_parent(s0)
    sched = constant_rainfall(0.0, 5.0, 60.0)
    with pytest.raises(wl.LifecycleError, match="daily lineage"):
        wl.begin_storm(c.lop, s0, expected_parent=ep, schedule=sched, field=field_of(c), control=STORM_CONTROL)
    with pytest.raises(wl.LifecycleError, match="daily lineage"):
        wl.enter_interevent_from_boundary(c.lop, s0, expected_parent=ep)
    _, r = act(c, s0, 2.0)
    ie = r.state
    with pytest.raises(wl.LifecycleError, match="daily lineage"):
        wl.evolve_interevent(c.lop, ie, expected_parent=wl.expect_parent(ie), until_s=ie.t_s + 10.0, pet_terrain_m_s=pet(c, 0.0), max_dt_s=5.0)
    # a state WITHOUT a cursor takes exactly the Phase 1 path, and the daily API refuses it
    assert c.s0.daily is None
    with pytest.raises(dw.DailyRefusal, match="daily lineage"):
        dw.daily_water_action(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0), plan=rc.plan_daily_rain(0.0, 0.0, 1.0, run_digest=RUN, ordinal=0),
                              pet_terrain_m_s=pet(c, 0.0))
    p1 = wl.enter_interevent_from_boundary(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0))
    assert p1.daily is None and p1.role == "interevent"
    with pytest.raises(dw.DailyRefusal):  # an interevent/closed state of a started lineage is not a boundary parent
        dw.start_daily_lineage(c.lop, s0, expected_parent=ep, run_digest=RUN)  # already a daily lineage


# --- T after the daily water, later moisture, close ----------------------------------------------------------------------------------
def test_t_request_after_daily_water_then_close_and_a_later_full_storm_continue_the_same_lineage():
    c = build(GRID, lay=layers(bottom="open"), theta=(0.3, 0.3))
    s0 = start(c)
    _, r = act(c, s0, 2.0, pet_rate=2.0 ** -24)
    ie = r.state
    q = np.zeros((c.L, c.n))
    q[0] = 2e-4
    s3, fb = wl.request_withdrawal(c.lop, ie, expected_parent=wl.expect_parent(ie), depth_m=q, identity="leaf-1")
    assert s3.daily.ordinal == 1 and s3.hydro_count == ie.hydro_count and s3.n_requests == 1  # zero-time T: no extra day, no ordinal change
    closed = wl.close_interevent(c.lop, s3, expected_parent=wl.expect_parent(s3))
    assert closed.role == "boundary" and closed.daily.ordinal == 1 and closed.t_s == DAY
    audit = lifetime(c, closed)
    assert abs(audit["residual_m3"]) <= audit["bound_m3"] and audit["transpiration_m3"] > 0.0 and audit["rain_m3"] > 0.0
    soil_before = host(closed.payload.soil.water_m).copy()  # the preserved moisture
    plan = dw.next_plan(closed, 3.0, 360.0, 1.0)
    assert plan.selection == "storm"
    st = dw.begin_daily_storm(c.lop, closed, expected_parent=wl.expect_parent(closed), plan=plan, control=STORM_CONTROL)
    assert host(st.payload.water.soil.water_m).tobytes() == soil_before.tobytes() and st.t_s == DAY  # the storm starts from the preserved state
    res = run_storm(c, st)
    assert res.complete
    out = dw.finish_daily_storm(c.lop, res.state, expected_parent=wl.expect_parent(res.state), plan=plan, pet_terrain_m_s=pet(c, 2.0 ** -24))
    assert out.state.daily.ordinal == 2
    audit = lifetime(c, out.state)
    assert abs(audit["residual_m3"]) <= audit["bound_m3"]


# --- memory -------------------------------------------------------------------------------------------------------------------------
def test_whole_call_memory_is_admitted_before_any_entry_and_matches_the_declared_figure(monkeypatch):
    c = closed_case(theta=(0.25, 0.25))
    s0 = start(c)
    box = Box(1 << 42)
    object.__setattr__(c.col, "available_bytes_fn", box)
    need = dw.required_daily_bytes(c.lop, s0)
    m = c.lop.memory
    assert need >= wl.required_enter_from_boundary_bytes(c.lop, s0) and need > m.owned_fixed_bytes + m.call_parent_bytes  # includes the real parent + owners
    entered = []
    real = wl.enter_interevent_from_boundary
    monkeypatch.setattr(wl, "enter_interevent_from_boundary", lambda *a, **k: (entered.append(1), real(*a, **k))[1])
    before = fingerprint(s0)
    box.value = need - 1
    plan = dw.next_plan(s0, 2.0, 10.0, 5.0)
    with pytest.raises(Exception) as e:
        dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, pet_terrain_m_s=pet(c, 0.0))
    assert isinstance(e.value, dw.DailyRefusal) and e.value.code == "memory" and entered == [] and fingerprint(s0) == before  # BEFORE any entry
    box.value = need + (1 << 20)
    assert act(c, s0, 2.0)[1].status == "complete"
    assert dw.required_daily_bytes(c.lop, s0, 1 << 30) > need  # the caller's retained feedback bytes join the step stage


def test_daily_action_allocates_no_deep_bed_copy_or_history_beyond_the_declared_need():
    import tracemalloc

    c = closed_case(theta=(0.25, 0.25))
    s0 = start(c)
    need = dw.required_daily_bytes(c.lop, s0)
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    _, r = act(c, s0, 2.0, pet_rate=2.0 ** -24)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert r.status == "complete" and peak <= need
    # the accepted result shares the parent's deep bed (voxel column, active layer, ledger) by reference: no deep-bed copy
    pb, rb = s0.payload.bed, r.state.payload.bed
    assert rb.voxel_column is pb.voxel_column and rb.active_layer is pb.active_layer and rb.ledger is pb.ledger


# --- checkpoint writers refuse a daily state before any I/O ----------------------------------------------------------------------------
def test_existing_checkpoint_writers_refuse_a_daily_state_before_io(tmp_path):
    from maple_syrup import layered_checkpoint as lc
    from maple_syrup import lifecycle_checkpoint as lk
    from maple_syrup import lifecycle_checkpoint_cuda as lkc

    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    s0 = start(c)
    limits = lc.CheckpointLimits(1 << 30, 1 << 30, 0, 1 << 30)
    path = tmp_path / "ck"
    with pytest.raises(wl.LifecycleError):
        wl.checkpoint_state_record(c.lop, s0)
    with pytest.raises(lk.LifecycleCheckpointError, match="daily lineage"):
        lk.save_lifecycle_checkpoint(path, c.lop, s0, {}, limits=limits, host_available_bytes_fn=lambda: 1 << 40)
    assert not path.exists() and list(tmp_path.iterdir()) == []
    limits_dev = lc.CheckpointLimits(1 << 30, 1 << 30, 1 << 30, 1 << 30)
    with pytest.raises(Exception, match="daily lineage"):
        lkc.save_lifecycle_checkpoint_cuda(tmp_path / "ck2", c.lop, s0, {}, limits=limits_dev, caller_host_reserve_bytes=0, caller_device_reserve_bytes=0)
    assert list(tmp_path.iterdir()) == []
    assert c.s0.daily is None and wl.checkpoint_state_record(c.lop, c.s0)  # a Phase 1 state still records exactly as before


def test_absent_cursor_leaves_the_phase1_state_digest_body_and_receipt_shapes_unchanged():
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    p1 = c.s0
    assert p1.daily is None and "daily" not in str(wl.checkpoint_state_record(c.lop, p1).keys())
    started = start(c)
    assert started.digest != wl._mint(started, daily=None).digest  # the cursor IS part of a daily state's digest ...
    assert wl._mint(p1).digest == p1.digest  # ... and absent it the digest of a Phase 1 state is unchanged by re-minting
    ie = wl.enter_interevent_from_boundary(c.lop, p1, expected_parent=wl.expect_parent(p1))
    cand = it.general_interevent_water_step(ie.binding.gop, ie.payload, pet(c, 0.0), 60.0, phase="interevent")
    assert type(cand.receipt) is iw.InterEventStepReceipt and "recharge_m3" not in cand.receipt.as_dict()
