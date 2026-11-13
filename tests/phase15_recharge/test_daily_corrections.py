"""Phase 1.5 S2 correction01 controls (Root review01 A2/A3/R1-R4) on the constructor-built ACTUAL native Dense bed (CPU)."""
from __future__ import annotations

import gc

import numpy as np
import pytest
from dcases import DAY, MM, RUN, STORM_CONTROL, Box, act, build, dw, field_of, fingerprint, host, pet, rc, run_storm, ss, start, uniform_soil, wl
from storm_cases import layers

from maple_syrup import interevent_transpiration as it
from maple_syrup import interevent_water as iw
from maple_syrup.rainfall import constant_rainfall

GRID = np.full((4, 6), 0.2)


def closed(theta=(0.5, 0.5), **kw):
    return build(GRID, lay=layers(bottom="closed"), theta=theta, **kw)


# --- A2: true-parent provenance, no cursor-free twin -----------------------------------------------------------------------------------
def test_no_cursor_free_twin_exists_and_receipts_name_the_true_daily_parent():
    assert not hasattr(dw, "_cursor_free")
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    s0 = start(c)
    _, r = act(c, s0, 2.0)
    assert r.state.last_phase_receipt.parent_digest == s0.digest and r.receipt.parent_digest == s0.digest  # the ACTUAL accepted Daily parent
    assert s0.digest != wl._mint(s0, daily=None).digest  # ... which is not the cursor-free digest a twin would have recorded


def test_storm_branch_receipts_and_staged_states_use_the_true_parent_and_strict_role_cursor_invariant():
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    s0 = start(c)
    plan = dw.next_plan(s0, 2.0, 360.0, 1.0)
    st = dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, control=STORM_CONTROL)
    assert st.role == "storm" and st.daily.pending is not None and st.daily.ordinal == 0 and s0.daily.pending is None  # staged cursor is pure
    done = run_storm(c, st).state
    out = dw.finish_daily_storm(c.lop, done, expected_parent=wl.expect_parent(done), plan=plan, pet_terrain_m_s=pet(c, 0.0))
    assert out.state.last_phase_receipt.parent_digest == done.digest  # the actual completed-storm DAILY parent (with its pending cursor)
    assert done.daily.pending is not None and done.daily.ordinal == 0  # the parent's pending cursor is unchanged by the whole finish
    wl._check_state(c.lop, out.state)  # the accepted result satisfies the role<->pending invariant


def test_the_engine_permit_is_exact_parent_scoped_and_never_leaks():
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    s0 = start(c)
    twin = wl._mint(s0)  # an equal-digest alternate physical branch
    _, r = act(c, s0, 2.0)
    for parent in (s0, twin):  # after a completed daily call the permit is gone: the old public transitions refuse both
        with pytest.raises(wl.LifecycleError, match="daily lineage"):
            wl.enter_interevent_from_boundary(c.lop, parent, expected_parent=wl.expect_parent(parent))
    with wl._daily_engine(s0, lambda payload: s0.daily):  # a permit for ONE instance does not admit its equal-digest twin
        with pytest.raises(wl.LifecycleError, match="daily lineage"):
            wl.enter_interevent_from_boundary(c.lop, twin, expected_parent=wl.expect_parent(twin))
    bad = c.s0
    with pytest.raises(dw.DailyRefusal):  # a cursor-free state is refused by the daily API
        dw.daily_water_action(c.lop, bad, expected_parent=wl.expect_parent(bad), plan=rc.plan_daily_rain(0.0, 0.0, 1.0, run_digest=RUN, ordinal=0),
                              pet_terrain_m_s=pet(c, 0.0))
    # a REFUSED daily call (injected router rejection) also leaves no permit behind
    from maple_syrup.surface_routing import HydraulicStepRejected

    cls = type(c.lop.router)
    real = cls.step
    cls.step = lambda self, *a, **k: (_ for _ in ()).throw(HydraulicStepRejected("x"))
    try:
        with pytest.raises(dw.DailyRefusal):
            act(c, s0, 2.0)
    finally:
        cls.step = real
    with pytest.raises(wl.LifecycleError, match="daily lineage"):
        wl.enter_interevent_from_boundary(c.lop, s0, expected_parent=wl.expect_parent(s0))


# --- A3: genuine (un-injected) quiet-water CFL refusal ----------------------------------------------------------------------------------
@pytest.mark.parametrize("pet_rate", [0.0, 1e-16])
def test_genuine_quiet_water_day_step_cfl_refusal_leaves_everything_exact_and_credits_no_trial_e(pet_rate):
    """4x6 native bed, closed saturated column (no soil room), 1e-9 m of quiet retained water, P=0: the existing quiet entry ADMITS it and the ONE
    86400 s router trial genuinely raises the CFL bound (NOT injected). A bounded UNSUPPORTED case: no sub-steps, hold law or loosened threshold."""
    c = closed(depth=1e-9)
    s0 = start(c)
    before, cursor, life = fingerprint(s0), s0.daily, s0.lifetime
    with pytest.raises(dw.DailyRefusal) as e:
        act(c, s0, 0.0, t=1.0, i=0.0, pet_rate=pet_rate)
    assert e.value.code == "router_step_rejected" and "CFL" in str(e.value) and "exceeds cfl_max" in str(e.value)
    assert fingerprint(s0) == before and s0.daily is cursor and s0.lifetime == life and s0.t_s == 0.0 and s0.hydro_count == 0
    assert s0.lifetime.soil_evaporation_m3 == 0.0 and s0.lifetime.surface_evaporation_m3 == 0.0  # the refused trial's E was never credited
    assert s0.daily.ordinal == 0 and s0.daily.last_branch is None


# --- R1: admission BEFORE the rainfall field --------------------------------------------------------------------------------------------
def test_begin_daily_storm_whole_call_is_admitted_before_the_n_sized_rain_field(monkeypatch):
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    s0 = start(c)
    plan = dw.next_plan(s0, 2.0, 360.0, 1.0)
    box = Box(1 << 42)
    object.__setattr__(c.col, "available_bytes_fn", box)
    need = dw.required_begin_daily_storm_bytes(c.lop, s0, plan)
    assert need >= wl.required_begin_storm_bytes(c.lop, plan.bind_forcing(0.0).schedule) + 25 * c.n  # the field temporaries are charged
    calls = []
    real = dw.rainfall_field
    monkeypatch.setattr(dw, "rainfall_field", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    before = fingerprint(s0)
    for value in (0, need - 1):
        box.value = value
        with pytest.raises(dw.DailyRefusal) as e:
            dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, control=STORM_CONTROL)
        assert e.value.code == "memory" and calls == [] and fingerprint(s0) == before  # the field was NEVER allocated
    box.value = need + (1 << 20)
    assert dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, control=STORM_CONTROL).role == "storm" and calls == [1]


# --- R2: finish whole call, start audit scratch ----------------------------------------------------------------------------------------
def test_finish_daily_storm_whole_call_is_admitted_before_the_entry_with_the_live_storm_parent(monkeypatch):
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    s0 = start(c)
    plan = dw.next_plan(s0, 2.0, 360.0, 1.0)
    st = dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, control=STORM_CONTROL)
    done = run_storm(c, st).state
    box = Box(1 << 42)
    object.__setattr__(c.col, "available_bytes_fn", box)
    need = dw.required_finish_daily_bytes(c.lop, done)
    assert need >= wl.required_enter_interevent_bytes(c.lop, done) and dw.required_finish_daily_bytes(c.lop, done, 1 << 30) > need  # + live event/feedback
    live = int(done.binding.event.required_bytes) - int(done.binding.op.memory.required_bytes)
    assert need >= c.lop.memory.owned_fixed_bytes + c.lop.memory.call_parent_bytes + max(live, 0)  # the original completed-storm parent stays charged
    entered = []
    real = wl.enter_interevent_phase
    monkeypatch.setattr(wl, "enter_interevent_phase", lambda *a, **k: (entered.append(1), real(*a, **k))[1])
    ledger, t = done.payload.water.ledger, done.t_s
    box.value = need - 1
    with pytest.raises(dw.DailyRefusal) as e:
        dw.finish_daily_storm(c.lop, done, expected_parent=wl.expect_parent(done), plan=plan, pet_terrain_m_s=pet(c, 1.0))
    assert e.value.code == "memory" and entered == [] and done.t_s == t and done.payload.water.ledger == ledger and done.daily.pending is not None
    box.value = need + (1 << 20)
    assert dw.finish_daily_storm(c.lop, done, expected_parent=wl.expect_parent(done), plan=plan, pet_terrain_m_s=pet(c, 1.0)).status == "complete"


def test_start_daily_lineage_admits_the_inventory_audit_scratch_before_validation(monkeypatch):
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    box = Box(1 << 42)
    object.__setattr__(c.col, "available_bytes_fn", box)
    called = []
    real = wl._validate_inventories
    monkeypatch.setattr(wl, "_validate_inventories", lambda *a, **k: (called.append(1), real(*a, **k))[1])
    need = wl.required_audit_bytes(c.lop, c.s0)
    box.value = need - 1
    with pytest.raises(dw.DailyRefusal) as e:
        dw.start_daily_lineage(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0), run_digest=RUN)
    assert e.value.code == "memory" and called == []  # the finite/capacity masks were never built
    box.value = need + (1 << 20)
    assert dw.start_daily_lineage(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0), run_digest=RUN).daily.ordinal == 0 and called == [1]


# --- R3: invalid recharge depth at the new public boundary -----------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [None, True, False, np.bool_(True), float("nan"), float("inf"), -1e-9, "0.001", np.array([0.001]), [0.001], 1j])
def test_recharge_wrapper_rejects_every_invalid_depth_before_admission_or_trial(bad, monkeypatch):
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    ie = wl.enter_interevent_from_boundary(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0))
    calls = []
    monkeypatch.setattr(it, "_general_step_engine", lambda *a, **k: calls.append(1))
    with pytest.raises(it.GeneralIntereventError, match="recharge_depth_m"):
        it.general_interevent_recharge_step(ie.binding.gop, ie.payload, pet(c, 0.0), DAY, recharge_depth_m=bad, phase="interevent")
    assert calls == []  # not even the engine (admission/state checks) ran


def test_recharge_wrapper_accepts_valid_scalars_and_the_ordinary_step_is_unchanged():
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    ie = wl.enter_interevent_from_boundary(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0))
    for ok in (0.0, 0, 1e-3, np.float64(1e-3), np.float32(1e-3)):
        cand = it.general_interevent_recharge_step(ie.binding.gop, ie.payload, pet(c, 0.0), 60.0, recharge_depth_m=ok, phase="interevent")
        assert type(cand.receipt) is iw.RechargeStepReceipt
    plain = it.general_interevent_water_step(ie.binding.gop, ie.payload, pet(c, 0.0), 60.0, phase="interevent")
    assert type(plain.receipt) is iw.InterEventStepReceipt


# --- R4: capacity reason ---------------------------------------------------------------------------------------------------------------
def test_capacity_failure_records_the_bounded_reason_and_it_survives_into_the_fallback_receipt():
    c = closed(theta=(0.5, 0.5))
    s0 = start(c)
    plan, r = act(c, s0, 2.0, t=5.0, i=360.0)
    f = r.fallback
    assert f.reason_bit == 17 and f.failing_columns == c.n and (f.selection, f.p_mm, f.i_mm_h, f.t_mm) == ("recharge", 2.0, 360.0, 5.0)
    storm = dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, fallback_of=f, control=STORM_CONTROL)
    assert storm.daily.pending.capacity_failing_columns == c.n and storm.daily.pending.branch == "storm_fallback"
    done = run_storm(c, storm).state
    out = dw.finish_daily_storm(c.lop, done, expected_parent=wl.expect_parent(done), plan=plan, pet_terrain_m_s=pet(c, 2.0 ** -23))  # (with PET 0 the 2 mm pond of a
    rc_ = out.receipt                                                                                                              # saturated column makes the day CFL-refuse)
    assert (rc_.selection, rc_.branch, rc_.capacity_failing_columns, rc_.p_mm, rc_.i_mm_h, rc_.t_mm) == ("recharge", "storm_fallback", c.n, 2.0, 360.0, 5.0)
    assert rc_.recharge_m3 == 0.0 and rc_.preferential_intake_m3 >= 0.0 and rc_.matrix_intake_m3 >= 0.0  # full storm: recharge and modeled intake stay separate
    assert out.state.daily.pending is None and out.state.daily.last_branch == "storm_fallback"


def test_failing_column_count_is_the_actual_number_of_saturated_active_columns():
    c = closed(theta=(0.25, 0.25))
    full = [0.5 * 0.25 * 1000.0, 0.5 * 0.75 * 1000.0]
    ok = [0.25 * 0.25 * 1000.0, 0.25 * 0.75 * 1000.0]
    from dcases import with_soil

    for k in (1, 3):
        cells = [ok] * c.n
        for j in range(k):
            cells[2 * j + 1] = full
        s = with_soil(c, cells)
        s = dw.start_daily_lineage(c.lop, s, expected_parent=wl.expect_parent(s), run_digest=RUN)
        _, r = act(c, s, 0.5)
        assert r.status == "capacity_failure" and r.fallback.failing_columns == k  # a count, not a mask: bounded and restart-explainable


def test_pending_capacity_reason_is_part_of_the_cursor_digest_and_only_for_fallbacks():
    plan = rc.plan_daily_rain(2.0, 10.0, 5.0, run_digest=RUN, ordinal=0)
    f = plan.bind_forcing(0.0)
    cur = rc.DailyCursor(RUN)
    a = cur.begin_storm(plan, f, fallback=True, t0_s=0.0, event_sha256="cd" * 32, capacity_failing_columns=3)
    b = cur.begin_storm(plan, f, fallback=True, t0_s=0.0, event_sha256="cd" * 32, capacity_failing_columns=4)
    assert a.digest() != b.digest()
    with pytest.raises(rc.RechargeError):
        rc.PendingStorm(0, "ab" * 32, "storm", "ab" * 32, None, 2)  # a plain storm has no capacity reason
    with pytest.raises(rc.RechargeError):
        rc.PendingStorm(0, "ab" * 32, "storm_fallback", "ab" * 32, None, 0)
