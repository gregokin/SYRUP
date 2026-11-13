"""Phase 1.5 S3: the daily lifecycle on the GENUINE native9f CUDA case (WRITTEN, NOT RUN by the author: no GPU launch/probe). Opt-in: SYRUP_RELEASE_GPU=1 (the `gpu`
fixture) plus SYRUP_NCP_CUDA_TARGET; run with that CUDA bundle's `editable`/`source/src` and the candidate `src` on PYTHONPATH in a FRESH process (the CPU NCP profile
and this one never share a process), only the GPU Root authorizes visible. Numerical comparisons use the existing declared contracts (rtol 2e-12 / atol 1e-14 against CPU
facts; CUDA-vs-CUDA runs of the same kernels are compared exactly); no tolerance is raised. UNQUALIFIED until Root records a guarded run."""
from __future__ import annotations

import numpy as np
import pytest
import gcases as g
import ncpcases as nc
from dcases import DAY, MM, RUN, rc
from test_daily_known_answers import TABLE

from maple_syrup import daily_water as dw
from maple_syrup import layered_event as ev
from maple_syrup import water_lifecycle as wl

pytestmark = pytest.mark.usefixtures("gpu")


@pytest.fixture(autouse=True)
def _configured_cuda_bundle(gpu):  # noqa: ARG001  (the `gpu` fixture skips unless SYRUP_RELEASE_GPU=1 and a device exists)
    g.require_cuda_bundle()


def start(c, state=None, run=RUN):
    s = c.s0 if state is None else state
    return dw.start_daily_lineage(c.lop, s, expected_parent=wl.expect_parent(s), run_digest=run)


def act(c, state, p, t=5.0, i=10.0, pet_rate=0.0, **kw):
    plan = dw.next_plan(state, p, i, t)
    return plan, dw.daily_water_action(c.lop, state, expected_parent=wl.expect_parent(state), plan=plan, pet_terrain_m_s=g.pet(c, pet_rate), **kw)


def storm_to_completion(c, state, control):
    res = wl.evolve_storm(c.lop, state, expected_parent=wl.expect_parent(state), until_s=state.t_s + 40_000.0)
    for _ in range(12):
        if res.complete:
            break
        res = wl.evolve_storm(c.lop, res.state, expected_parent=wl.expect_parent(res.state), until_s=res.state.t_s + 40_000.0)
    return res


def with_soil_cells(c, mm_by_cell):
    from maple_syrup import subsurface as ss

    w = c.cp.asarray(np.ascontiguousarray(np.array(mm_by_cell, dtype=np.float64).T * MM))
    soil = ss.column_state(c.col, w, 0.0)
    return wl.start_lifecycle(c.lop, bed=c.s0.payload.bed, soil=soil, history=c.s0.payload.history, t_s=0.0)


# --- planners: separate host/device, metadata only ---------------------------------------------------------------------------------
def test_cuda_daily_plans_are_separate_host_device_metadata_and_never_the_cpu_integer(tmp_path):
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2))
    s0 = start(c)
    plan = dw.next_plan(s0, 2.0, 360.0, 1.0)
    for fn in (dw.required_daily_bytes, dw.required_finish_daily_bytes):
        with pytest.raises(dw.DailyRefusal, match="separate host and device"):
            fn(c.lop, s0)
    with pytest.raises(dw.DailyRefusal, match="separate host and device"):
        dw.required_begin_daily_storm_bytes(c.lop, s0, plan)
    calls = []
    real = c.col.preflight
    object.__setattr__(c.col, "preflight", lambda: (calls.append(1), real())[1])
    try:
        plans = {k: dw.cuda_daily_plan(c.lop, k, s0, plan=plan) for k in ("start", "daily", "begin_storm")}
    finally:
        object.__setattr__(c.col, "preflight", real)
    assert calls == []  # the planner performs NO current-device query / availability read
    for k, p in plans.items():
        assert p["host_bytes"] > 0 and p["device_bytes"] > 0 and p["host_bytes"] != p["device_bytes"], k
    assert plans["daily"]["device_bytes"] >= max(plans["daily"]["device_stage_table"].values())
    assert dw.cuda_daily_plan(c.lop, "daily", s0, retained_feedback_bytes=1 << 20)["device_bytes"] >= plans["daily"]["device_bytes"]


# --- Root's known-answer table on the device ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("initial,rain,bottom,final,drain", TABLE)
def test_root_table_on_cuda(initial, rain, bottom, final, drain, tmp_path):
    # OWNED deeper native column (nz = 8): the 2 x / 3 x 0.1 m profiles exceed the public example's represented depth and the genuine texture factory refuses them
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(len(initial), bottom=bottom, theta=[x / 100.0 for x in initial], root_table=True), nz=nc.NZ_DEEP)
    plan, r = act(c, start(c), rain, t=rain)
    assert plan.selection == "recharge" and r.status == "complete"
    soil = g.host(c.cp, r.state.payload.soil.water_m)
    np.testing.assert_allclose(soil / MM, np.array(final)[:, None] * np.ones((1, c.n)), rtol=0, atol=1e-9)
    led = r.state.payload.ledger.hydrology
    assert led.surface_return_m3 == 0.0 and led.matrix_intake_m3 == 0.0 and led.preferential_intake_m3 == 0.0
    area_total = c.col.n_active * c.col.cell_area_m2
    assert r.receipt.drainage_m3 == pytest.approx(drain * MM * area_total, abs=1e-12)  # the known-table DRAINAGE (open bottoms), not only the soil
    assert led.drainage_m3 == pytest.approx(drain * MM * area_total, abs=1e-12) and led.rain_m3 == pytest.approx(rain * MM * area_total, rel=1e-12)
    audit = wl.audit_lifecycle(c.lop, r.state)
    assert abs(audit["residual_m3"]) <= audit["bound_m3"] and r.state.t_s == DAY


def test_gravity_off_and_p_zero_and_selection_on_cuda(tmp_path):
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2, bottom="closed", theta=[0.0, 0.0], root_table=True, gravity=False), nz=nc.NZ_DEEP)
    _, r = act(c, start(c), 30, t=30)
    np.testing.assert_allclose(g.host(c.cp, r.state.payload.soil.water_m)[:, 0] / MM, [30, 0], atol=1e-9)
    s = start(c)
    for p, sel in ((0.0, "no_rain"), (5.0, "recharge")):
        plan = dw.next_plan(s, p, 10.0, 5.0)
        assert plan.selection == sel
        out = dw.daily_water_action(c.lop, s, expected_parent=wl.expect_parent(s), plan=plan, pet_terrain_m_s=g.pet(c, 1e-9))
        assert out.status == "complete" and out.state.t_s == DAY and out.state.daily.last_branch == sel


# --- capacity failure, exact discard, whole-occurrence fallback --------------------------------------------------------------------
def test_one_saturated_column_forces_the_whole_occurrence_and_the_parent_is_exact(tmp_path):
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2, bottom="closed", theta=[0.25, 0.25], root_table=True), nz=nc.NZ_DEEP)
    full = [40.0, 40.0]
    ok = [20.0, 20.0]
    cells = [ok] * c.n
    cells[5] = full
    s0 = start(c, with_soil_cells(c, cells))
    before = g.fingerprint(c, s0)
    _, r = act(c, s0, 0.5)
    assert r.status == "capacity_failure" and r.fallback.failing_columns == 1 and r.fallback.reason_bit == 17
    assert g.fingerprint(c, s0) == before and s0.daily.ordinal == 0


def test_flat_closed_fallback_equals_a_direct_full_storm_default_policy_then_day_T_and_later_storm(tmp_path):
    raw = g.cuda_yaml(2, bottom="closed", theta=0.5)
    cd, cn = g.build_cuda_case(tmp_path / "a", raw, flat_closed=True), g.build_cuda_case(tmp_path / "b", raw, flat_closed=True)
    s_daily = start(cd)
    plan, r = act(cd, s_daily, 0.01, t=1.0, i=360.0, pet_rate=1e-8)
    assert r.status == "capacity_failure"
    control = ev.LayeredEventControl()  # the defaults: no loosened policy, no cap raised
    st = dw.begin_daily_storm(cd.lop, s_daily, expected_parent=wl.expect_parent(s_daily), plan=plan, fallback_of=r.fallback, control=control)
    from maple_syrup.rainfall import rainfall_field

    ny, nx = cn.shape
    field = rainfall_field(ny, nx, active_mask=cn.col.active.reshape(ny, nx), xp=cn.lop.xp)
    direct = wl.begin_storm(cn.lop, cn.s0, expected_parent=wl.expect_parent(cn.s0), schedule=plan.bind_forcing(0.0).schedule, field=field, control=control)
    a, b = storm_to_completion(cd, st, control).state, storm_to_completion(cn, direct, control).state
    assert a.payload.completed and b.payload.completed and a.payload.event_sha256 == b.payload.event_sha256
    fa, fb = g.fingerprint(cd, a), g.fingerprint(cn, b)
    assert fa[1] == fb[1] and fa[5] == fb[5] and a.payload.water.ledger == b.payload.water.ledger  # same kernels, same device: exact
    done = dw.finish_daily_storm(cd.lop, a, expected_parent=wl.expect_parent(a), plan=plan, pet_terrain_m_s=g.pet(cd, 1e-8))
    assert done.state.daily.ordinal == 1 and done.receipt.capacity_failing_columns == cd.n
    q = cd.cp.zeros((cd.L, cd.n))
    q[0] = 1e-6
    s3, _ = wl.request_withdrawal(cd.lop, done.state, expected_parent=wl.expect_parent(done.state), depth_m=q, identity="leaf-1")
    closed = wl.close_interevent(cd.lop, s3, expected_parent=wl.expect_parent(s3))
    later = dw.next_plan(closed, 0.5, 360.0, 0.1)
    st2 = dw.begin_daily_storm(cd.lop, closed, expected_parent=wl.expect_parent(closed), plan=later, control=control)
    assert storm_to_completion(cd, st2, control).complete


def test_partial_storm_is_not_complete_and_cannot_enter_the_daily_step(tmp_path):
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2), flat_closed=True)
    s0 = start(c)
    plan = dw.next_plan(s0, 2.0, 360.0, 1.0)
    st = dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, control=ev.LayeredEventControl())
    part = wl.evolve_storm(c.lop, st, expected_parent=wl.expect_parent(st), until_s=st.t_s + 40_000.0, max_accepted_steps=1)
    assert not part.complete
    with pytest.raises(dw.DailyRefusal) as e:
        dw.finish_daily_storm(c.lop, part.state, expected_parent=wl.expect_parent(part.state), plan=plan, pet_terrain_m_s=g.pet(c, 1e-8))
    assert e.value.code == "storm_incomplete"
    with pytest.raises(wl.LifecycleError, match="daily lineage"):
        wl.enter_interevent_phase(c.lop, part.state, expected_parent=wl.expect_parent(part.state))


# --- occurrence binding, parent provenance ------------------------------------------------------------------------------------------
def test_repeat_gap_wrong_run_twin_and_true_parent_receipt_on_cuda(tmp_path):
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2, theta=0.2))
    s = start(c)
    plan, r = act(c, s, 0.0)
    assert r.state.last_phase_receipt.parent_digest == s.digest  # the ACTUAL accepted Daily parent, no cursor-free twin
    closed = wl.close_interevent(c.lop, r.state, expected_parent=wl.expect_parent(r.state))
    with pytest.raises(dw.DailyRefusal) as e:
        dw.daily_water_action(c.lop, closed, expected_parent=wl.expect_parent(closed), plan=plan, pet_terrain_m_s=g.pet(c, 0.0))
    assert e.value.code == "occurrence"
    for bad in (rc.plan_daily_rain(1.0, 10.0, 5.0, run_digest=RUN, ordinal=7), rc.plan_daily_rain(1.0, 10.0, 5.0, run_digest="cd" * 32, ordinal=1)):
        with pytest.raises(dw.DailyRefusal):
            dw.daily_water_action(c.lop, closed, expected_parent=wl.expect_parent(closed), plan=bad, pet_terrain_m_s=g.pet(c, 0.0))
    twin = wl._mint(closed)
    with pytest.raises(dw.DailyRefusal):
        dw.daily_water_action(c.lop, twin, expected_parent=wl.expect_parent(closed), plan=dw.next_plan(closed, 0.0, 0.0, 1.0), pet_terrain_m_s=g.pet(c, 0.0))


# --- genuine CFL refusal ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pet_rate", [0.0, 1e-16])
def test_genuine_1e9_quiet_water_saturated_closed_cfl_refusal_is_exact(pet_rate, tmp_path):
    from maple_syrup import sediment_bed as sb

    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2, bottom="closed", theta=0.5))
    bed = sb.with_water(c.s0.payload.bed, c.lop.context, c.cp.full(c.shape, 1e-9))
    state = wl.start_lifecycle(c.lop, soil=c.s0.payload.soil, history=c.s0.payload.history, t_s=0.0, bed=bed)
    parent = start(c, state, "cd" * 32)
    before = g.fingerprint(c, parent)
    with pytest.raises(dw.DailyRefusal) as e:
        dw.daily_water_action(c.lop, parent, expected_parent=wl.expect_parent(parent), plan=dw.next_plan(parent, 0.0, 0.0, 1.0), pet_terrain_m_s=g.pet(c, pet_rate))
    assert e.value.code == "router_step_rejected" and "exceeds cfl_max" in str(e.value) and g.fingerprint(c, parent) == before


# --- namespace/device and host/device admission before allocation ---------------------------------------------------------------------
def test_wrong_namespace_pet_is_refused_before_any_entry(tmp_path, monkeypatch):
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2))
    s0 = start(c)
    entered = []
    real = wl.enter_interevent_from_boundary
    monkeypatch.setattr(wl, "enter_interevent_from_boundary", lambda *a, **k: (entered.append(1), real(*a, **k))[1])
    plan = dw.next_plan(s0, 2.0, 10.0, 5.0)
    with pytest.raises(dw.DailyRefusal) as e:
        dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, pet_terrain_m_s=np.zeros(c.n))  # a NumPy array on a CUDA lifecycle
    assert e.value.code == "step_refused" and entered == []


class Box:
    def __init__(self, value=1 << 42):
        self.value = value

    def __call__(self):
        return self.value


@pytest.mark.parametrize("which", ["host", "device"])
def test_host_short_and_device_short_are_refused_before_allocation_and_enough_runs(which, tmp_path, monkeypatch):
    hbox, dbox = Box(), Box()
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2), flat_closed=True, host_fn=hbox, device_fn=dbox)
    object.__setattr__(c.col, "available_bytes_fn", dbox)
    s0 = start(c)
    plan = dw.next_plan(s0, 2.0, 360.0, 1.0)
    p = dw.cuda_daily_plan(c.lop, "begin_storm", s0, plan=plan)
    spy = []
    real = dw.rainfall_field
    monkeypatch.setattr(dw, "rainfall_field", lambda *a, **k: (spy.append(1), real(*a, **k))[1])
    for box, need in ((hbox, p["host_bytes"]), (dbox, p["device_bytes"])):
        if (box is hbox) != (which == "host"):
            continue
        box.value = need - 1
        with pytest.raises(dw.DailyRefusal) as e:
            dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan)
        assert e.value.code == "memory" and spy == []  # the N-sized device field was never allocated
        box.value = 1 << 42
    assert dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan).role == "storm" and spy == [1]


# --- independent analytic NONZERO-clock controls (physical edges are literal numbers; the event adds its origin to the LOCAL schedule) -----------------------------------
T0 = 123.25
P_MM, I_MM_H, D_S = 1.0, 360.0, 10.0  # D = 3600 * 1 / 360 = 10 s, by hand


def check_event_and_full_p(c, storm, t0):
    from maple_syrup.conservation import volume_roundoff_bound_m3

    e = storm.binding.event
    assert e.event_origin_s == t0 and [float(x) for x in e.absolute_edges_s] == [t0, t0 + D_S] and e.rainfall_end_s == t0 + D_S
    assert [float(x) for x in e.schedule.edges_s] == [0.0, D_S]
    res = wl.evolve_storm(c.lop, storm, expected_parent=wl.expect_parent(storm), until_s=t0 + D_S)
    led = res.state.payload.water.ledger
    expected = P_MM * MM * c.col.n_active * c.col.cell_area_m2
    bound = volume_roundoff_bound_m3(res.accepted_steps + 2, expected) + led.global_residual_bound_m3
    assert res.state.payload.water.t_s == t0 + D_S and abs(led.rain_m3 - expected) <= bound  # FULL P x active area at the intended end
    life0, life1 = storm.lifetime, res.state.lifetime
    assert (life1.surface_evaporation_m3, life1.soil_evaporation_m3, life1.transpiration_m3) == (life0.surface_evaporation_m3, life0.soil_evaporation_m3, life0.transpiration_m3)
    assert res.accepted_steps < 200  # no phantom dry stretch


def test_nonzero_initial_clock_storm_edges_and_full_p_on_cuda(tmp_path):
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2, theta=0.1))
    s = start(c, g.at_clock(c, c.s0, T0))
    plan = dw.next_plan(s, P_MM, I_MM_H, 0.5)
    assert plan.selection == "storm"
    st = dw.begin_daily_storm(c.lop, s, expected_parent=wl.expect_parent(s), plan=plan, control=ev.LayeredEventControl())
    check_event_and_full_p(c, st, T0)


def test_p0_day_then_storm_and_recharge_day_then_storm_edges_on_cuda(tmp_path):
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2, theta=0.1))
    for p, t in ((0.0, 0.5), (0.25, 0.5)):  # a P0 day, then a successful recharge day
        s = start(c, g.at_clock(c, c.s0, T0))
        _, r = act(c, s, p, t=t, pet_rate=2.0 ** -24)
        assert r.status == "complete"
        closed = wl.close_interevent(c.lop, r.state, expected_parent=wl.expect_parent(r.state))
        assert closed.t_s == T0 + DAY  # 86523.25
        plan = dw.next_plan(closed, P_MM, I_MM_H, 0.5)
        st = dw.begin_daily_storm(c.lop, closed, expected_parent=wl.expect_parent(closed), plan=plan, control=ev.LayeredEventControl())
        check_event_and_full_p(c, st, 86523.25)


def test_capacity_fallback_at_a_nonzero_origin_runs_the_full_storm_at_t0_on_cuda(tmp_path):
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2, bottom="closed", theta=0.5))
    s = start(c, g.at_clock(c, c.s0, T0))
    plan, r = act(c, s, P_MM, t=2.0, i=I_MM_H)
    assert plan.selection == "recharge" and r.status == "capacity_failure"
    st = dw.begin_daily_storm(c.lop, s, expected_parent=wl.expect_parent(s), plan=plan, fallback_of=r.fallback, control=ev.LayeredEventControl())
    check_event_and_full_p(c, st, T0)


# --- a real configured active mask: inactive cells are unchanged and carry no recharge -------------------------------------------------------------------------------
def test_configured_active_mask_leaves_inactive_cells_unchanged_and_recharges_only_active_cells(tmp_path):
    inactive = [(1, 2), (2, 3)]
    raw = nc.with_inactive_cells(tmp_path, g.cuda_yaml(2, bottom="closed", theta=[0.1, 0.1]), inactive)
    c = g.build_cuda_case(tmp_path, raw)
    assert c.col.n_active == c.n - len(inactive)
    s = start(c)
    before = g.host(c.cp, s.payload.soil.water_m).copy()
    j0, i0 = g.host(c.cp, s.payload.history.wetting_m).copy(), g.host(c.cp, s.payload.history.cumulative_intake_m).copy()
    _, r = act(c, s, 2.0, pet_rate=2.0 ** -23)
    assert r.status == "complete"
    active = g.host(c.cp, c.col.active).reshape(-1).astype(bool)
    after = g.host(c.cp, r.state.payload.soil.water_m)
    assert after[:, ~active].tobytes() == before[:, ~active].tobytes()  # inactive soil is bit-for-bit unchanged (no recharge, no E)
    assert g.host(c.cp, r.state.payload.history.wetting_m)[~active].tobytes() == j0[~active].tobytes()
    assert g.host(c.cp, r.state.payload.history.cumulative_intake_m)[~active].tobytes() == i0[~active].tobytes()
    assert r.receipt.recharge_m3 == pytest.approx(2.0 * MM * c.col.n_active * c.col.cell_area_m2, rel=1e-12)  # ACTIVE cells only
    audit = wl.audit_lifecycle(c.lop, r.state)
    assert abs(audit["residual_m3"]) <= audit["bound_m3"]


# --- cold initial routing source = the ACTUAL device bed (Root review02) -------------------------------------------------------------------------------------------
def _assert_topology_is_the_device_bed_surface(c):
    from maple_syrup import sediment_bed as sb

    bed, ctx = c.s0.payload.bed, c.lop.context
    surface = g.host(c.cp, sb.current_routing_surface_m(bed, ctx))
    act = np.asarray(c.setup.topology.active)
    assert np.array_equal(np.asarray(c.setup.topology.source_bed_m)[act], surface[act])  # EXACT on every active cell (no 1-ULP host/device gap)
    plan = c.setup.plan
    assert plan["actual"]["cold_surface_transfer_bytes"] == 8 * c.n and "device native bed" in plan["actual"]["cold_surface_source"]
    assert plan["device_terms"]["S_surface_d2h_transfer_bytes"] == 8 * c.n and plan["host_terms"]["Z_downloaded_surface"] == 8 * c.n  # one bounded N-surface transfer
    assert plan["stage_bytes"]["device"]["S"] >= 3 * 8 * c.n and plan["device_required_bytes"] >= plan["stage_bytes"]["device"]["S"]


def test_nz8_irregular_topology_equals_the_actual_device_bed_exactly_and_the_genuine_entry_succeeds(tmp_path):
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2, theta=0.2), nz=nc.NZ_DEEP)  # the deeper IRREGULAR field that exposed a 1-ULP host/device difference on 4/24 cells
    _assert_topology_is_the_device_bed_surface(c)
    plan, r = act(c, start(c), 2.0)
    assert r.status == "complete" and r.state.t_s == DAY


def test_original_nz4_irregular_field_remains_exactly_the_device_bed_surface(tmp_path):
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2, theta=0.2))  # the public example geometry, unchanged
    _assert_topology_is_the_device_bed_surface(c)


@pytest.mark.parametrize("which", ["host", "device"])
def test_low_host_or_device_reading_is_refused_before_the_cold_surface_helper(which, tmp_path, monkeypatch):
    from maple_syrup import layered_case as lc
    from maple_syrup import sediment_bed as sb

    calls = []
    real = sb.current_routing_surface_m
    monkeypatch.setattr(sb, "current_routing_surface_m", lambda bed, ctx: (calls.append(1), real(bed, ctx))[1])
    kw = {"host_fn": Box(0)} if which == "host" else {"device_fn": Box(0)}
    with pytest.raises(lc.LayeredCaseMemoryError):
        g.build_cuda_case(tmp_path, g.cuda_yaml(2), **kw)
    assert calls == []  # the device diagnostic never ran
