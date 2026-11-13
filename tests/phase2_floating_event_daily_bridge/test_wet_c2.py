# ruff: noqa: RUF059
"""Correction02: the wet day runs WITHOUT any PondEventState / full-bed inventory / sediment phase allocation; the real wet preferential path; a wet day ending with standing water and consecutive wet days; component controls for the
router (CFL rejection, outlet export) on an actual opened event; small-rain P = T and the closed-capacity fallback on a bridge boundary; low/exact composed wet admission and a scoped traced peak."""
from __future__ import annotations

import re
import tracemalloc

import numpy as np
import pytest
from bridge_builders import AVAIL, RUN, build, dry_boundary, wet_boundary

from maple_syrup import floating_daily as fd
from maple_syrup import floating_evaporation as fe
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_matrix_infiltration as fmi
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import pond_event as pe
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup.experimental_hydrology import HydraulicStepRejected
from maple_syrup.experimental_storm import ExperimentalControl

PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
FAST = ExperimentalControl(max_dt_s=3600.0, min_dt_s=1 / 1024, max_retries=30, max_steps=2_000_000)
SLOW = ExperimentalControl(max_dt_s=10.0, min_dt_s=1 / 1024, max_retries=30, max_steps=2_000_000)


def pl(p=0.0, i=0.0, t=5.0, ordinal=0):
    return rc.plan_daily_rain(p, i, t, run_digest=RUN, ordinal=ordinal)


def wday(st, factory, ctl=FAST, plan=None, pet=3e-8, **kw):
    n = st.ctx.n_cells
    return br.wet_day_action(st, plan=pl() if plan is None else plan, pet_terrain_m_s=np.full(n, pet), cover=np.full(n, 0.3), event_factory=factory, available_bytes_fn=kw.pop("avail", AVAIL), day_control=ctl, **kw)


def test_a_wet_day_never_builds_a_pond_state_bed_inventory_or_sediment_phase_arrays(tmp_path, monkeypatch):
    """Root's V2 control: the tripwires sit on the full-bed inventory, the sediment phase allocation and the Pond state constructors; the wet day must pass them untouched (and the head must not move on a refusal)."""
    lin, b, factory = wet_boundary(tmp_path / "a", "numpy", False)
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)

    def boom(*a, **k):
        raise AssertionError("a wet day touched the full-bed / sediment initialisation")

    monkeypatch.setattr(pe, "bed_inventory", boom)
    monkeypatch.setattr(pe, "empty_phase_state", boom)
    monkeypatch.setattr(pe.PondEvent, "_assemble_initial", boom)
    monkeypatch.setattr(pe.PondEvent, "continued_state", boom)
    monkeypatch.setattr(pe.PondEvent, "initial_state", boom)
    r = wday(st, factory)
    assert r.outcome == "interevent" and r.state.t_s == b.t_s + 86400.0 and isinstance(lin.head, lc.DailyHold)


def test_wet_day_with_the_preferential_operator_matches_the_independently_composed_stages(tmp_path):
    lin, b, factory = wet_boundary(tmp_path / "a", "numpy", False, pref=True)
    assert b.root.pref_op is not None
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    n, L = st.ctx.n_cells, st.ctx.n_layers
    ny, nx = st.ctx.shape
    vert, ev, h0, solver = lc.day_runtime(st.ctx.hold, st, factory, available_bytes_fn=AVAIL)
    payload = vert.current_payload(h0.soil_water_m)
    assert payload.pref_op is not None
    dt, pet, cover = 1.0, np.full(n, 3e-8), np.full(n, 0.3)
    pool = h0.depth_m.reshape(n)
    p = fpi.floating_preferential_step(payload.pref_op, pool, payload.state, dt, available_bytes_fn=AVAIL)
    assert float(np.asarray(p.intake_total_m).sum()) > 0.0  # the preferential pathway is genuinely active
    m = fmi.floating_matrix_step(payload.matrix_op, p.surface_water_m, h0.depth_m.reshape(n), np.zeros(n), p.state, payload.history, dt, available_bytes_fn=AVAIL)
    e = fe.stage_floating_evaporation(st.ctx.evaporation, pet_terrain_m_s=pet, cover=cover, surface_water_m=m.surface_pool_m, dt_s=dt, available_bytes_fn=AVAIL)
    c = fsl.floating_step(payload.column, m.state, dt, added_water_m=np.zeros((L, n)), available_bytes_fn=AVAIL, **e.native_step_kwargs())
    ref = solver.surface._lateral((e.surface_water_after_m + c.surface_return_m).reshape(ny, nx), h0.qx_m2_s, h0.qy_m2_s, dt, float(h0.t_s) + dt, check_depth=True)
    got = solver.step_day(h0, dt, st.ctx.evaporation, pet, cover)
    assert np.array_equal(vert.current_payload(got.state.soil_water_m).state.water_m, c.state.water_m) and np.array_equal(got.state.depth_m, ref.state.depth_m)
    r = wday(st, factory)  # and the whole day with the operator
    assert r.outcome == "interevent" and abs(r.state.ledger.last_residual_m3) <= r.state.ledger.last_bound_m3


def test_a_wet_day_ending_with_standing_water_and_consecutive_wet_days_carry_depth_and_faces_exactly(tmp_path):
    lin, b, factory = wet_boundary(tmp_path / "a", "numpy", False, level_m=0.01)
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    r = wday(st, factory, ctl=SLOW, pet=1e-10)
    depth, qx, qy = r.state.surface
    assert float(depth.max()) > 0.0 and float(depth.sum()) < float(np.asarray(b.surface_depth_m).sum())  # standing water REMAINS after the day (infiltrated/evaporated partly)
    out = fd.request_floating_withdrawal(r.state, expected_parent=fd.expect_parent(r.state), request_m=np.zeros((st.ctx.n_layers, st.ctx.n_cells)), expected_sequence=0, available_bytes_fn=AVAIL)
    closed = fd.close_floating_day(out.state, expected_parent=fd.expect_parent(out.state)).state
    b2 = br.hand_back(lin, closed)
    assert np.array_equal(np.asarray(b2.surface_depth_m), depth) and np.array_equal(np.asarray(b2.qx_m2_s), qx) and np.array_equal(np.asarray(b2.qy_m2_s), qy)  # carried exactly through the hand-back
    assert np.array_equal(np.asarray(b2.bed.water.depth_m), depth) and abs(b2.ledger["residual_m3"]) <= b2.ledger["residual_bound_m3"]
    st2 = br.open_day(lin, b2, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)  # the SECOND consecutive wet day
    vert, ev, h0, solver = lc.day_runtime(st2.ctx.hold, st2, factory, available_bytes_fn=AVAIL)
    assert np.array_equal(h0.depth_m, depth) and np.array_equal(h0.qx_m2_s, qx) and np.array_equal(h0.qy_m2_s, qy)  # day 2 starts from day 1's exact surface state
    r2 = wday(st2, factory, ctl=SLOW, pet=1e-10, plan=pl(ordinal=1))
    assert r2.state.t_s == b2.t_s + 86400.0 and float(r2.state.surface[0].sum()) <= float(depth.sum())


def test_router_component_controls_cfl_rejection_is_pure_and_outlet_export_accumulates(tmp_path):
    """COMPONENT control (an opened, NOT completed event: a genuine complete fixture with an outlet cannot be produced because a draining film never quiets): the actual solver on the actual opened vertical."""
    owner, bed, factory = build(tmp_path / "a", "numpy", False, depth=0.003, rain_mm_h=0.0, end_s=2.0)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL)
    vert, ev = occ.vertical, occ.event
    solver = ev._runtime(ev._initial_topology, None).pond.solver
    held = vert.held_template()
    op = fe.prepare_floating_evaporation(held.column, PART, available_bytes_fn=AVAIL)
    n = vert.n_cells
    pet, cover = np.full(n, 3e-8), np.full(n, 0.3)
    hyd = st.hydraulic
    reg = len(vert._registry)
    with pytest.raises(HydraulicStepRejected):
        solver.step_day(hyd, 600.0, op, pet, cover)  # a CFL-violating fragment from the UNCHANGED state is rejected ...
    assert len(vert._registry) == reg  # ... and registers nothing (pure)
    export, steps = 0.0, 0
    for _ in range(400):
        step = solver.step_day(hyd, 0.5, op, pet, cover)
        export += solver.last_day["export_m3"]
        hyd = step.state
        steps += 1
    assert export > 0.0 and steps == 400  # the draining plane exports through the outlet faces and the faces (q) carry from fragment to fragment
    assert float(np.abs(hyd.qx_m2_s).max()) > 0.0


def test_small_rain_p_equals_t_recharge_and_the_closed_capacity_fallback_on_a_bridge_boundary(tmp_path):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    n, L = st.ctx.n_cells, st.ctx.n_layers
    plan_eq = pl(5.0, 10.0, 5.0)  # P = T
    assert plan_eq.selection == "recharge"
    j0 = np.array(st.history.wetting_m, copy=True)
    staged = fe.stage_floating_evaporation(st.ctx.evaporation, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), surface_water_m=np.zeros(n), dt_s=86400.0, available_bytes_fn=AVAIL)
    ref = fsl.floating_step_with_recharge(st.ctx.column, st.water, 86400.0, recharge_depth_m=plan_eq.depth_m, available_bytes_fn=AVAIL, **staged.native_step_kwargs())
    r = br.day_action(st, plan=plan_eq, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    assert r.outcome == "interevent" and r.branch == "recharge" and np.array_equal(r.state.water.water_m, ref.state.water_m)  # P = T integrates ONCE with the daily soil update
    assert np.array_equal(r.state.history.wetting_m, j0) and r.state.ledger.last_day_rain_m3 > 0.0  # small-rain history J/I unchanged
    # closed-capacity fallback: a near-saturated CLOSED profile cannot take the pulse -> FULL P from the UNCHANGED parent
    lin2, b2, _f = dry_boundary(tmp_path / "b", "numpy", False, bottom="closed", saturation=0.999999, rain_mm_h=0.0, end_s=2.0)
    st2 = br.open_day(lin2, b2, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    w0 = np.array(st2.water.water_m, copy=True)
    r2 = br.day_action(st2, plan=pl(5.0, 10.0, 5.0), pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    assert r2.outcome == "requires_full_storm" and r2.branch == "storm_fallback" and r2.capacity_failure
    assert r2.forcing.digest == pl(5.0, 10.0, 5.0).bind_forcing(st2.t_s).digest and r2.forcing.p_mm == 5.0  # FULL P at the unchanged parent's clock
    assert np.array_equal(st2.water.water_m, w0) and st2.cursor.ordinal == 0 and st2.t_s == b2.t_s  # nothing applied
    assert L > 0


def test_wet_day_composed_admission_ladder_exact_need_minus_one_and_a_scoped_traced_peak(tmp_path, monkeypatch):
    lin, b, factory = wet_boundary(tmp_path / "a", "numpy", False)
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    calls = []
    real = lc.fp._FloatingSolver.step_day
    monkeypatch.setattr(lc.fp._FloatingSolver, "step_day", lambda self, *a, **k: (calls.append(1), real(self, *a, **k))[1])

    def attempt(limit):
        return wday(st, factory, avail=lambda: limit)

    thr, need = 0, None
    for _ in range(12):
        calls.clear()
        try:
            attempt(thr)
            break
        except Exception as exc:  # noqa: BLE001
            m = re.search(r"estimated (\d+) B", str(exc))
            assert m and calls == [], (exc, calls)  # a refusal comes from the ADMISSION, before any fragment
            need = int(m.group(1))
            assert need > thr
            thr = need
    else:
        raise AssertionError("the admission ladder did not converge")
    assert isinstance(lin.head, lc.DailyHold)
    calls.clear()
    with pytest.raises(Exception, match="estimated"):
        attempt(thr - 1)
    assert calls == []  # need - 1 refuses before any fragment, nothing mutated
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    attempt(thr)  # exactly the composed need passes
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert peak <= 4 * thr, (peak, thr)  # scoped: the declared need is the LARGEST single admitted stage; the traced Python/NumPy peak of the whole day stays within a small multiple (not whole-process memory)
