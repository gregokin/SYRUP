# ruff: noqa: RUF059
"""Event <-> daily bridge (CPU NumPy/Numba x Dense/published Windowed): genuine completed native storm -> boundary -> bridge day (E, drainage, T, audited close) -> next native event; ONE lineage head. Real components, no fakes."""
from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
from bridge_builders import AVAIL, RUN, complete, dry_boundary

from maple_syrup import floating_daily as fd
from maple_syrup import floating_evaporation as fe
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_transpiration as ft
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup.rainfall import constant_rainfall

PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def plan(p, i, t, ordinal):
    return rc.plan_daily_rain(p, i, t, run_digest=RUN, ordinal=ordinal)


def day(st, pl, n):
    return fd.floating_daily_action(st, expected_parent=fd.expect_parent(st), plan=pl, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)


@pytest.mark.parametrize("windowed", [False, True])
def test_event_day_event_cycle_matches_independent_components_and_closes_the_lifetime_ledger(tmp_path, backend, windowed):
    lin, b, factory = dry_boundary(tmp_path / "a", backend, windowed)
    root = b.root
    n, L = root.column.n_cells, root.column.n_layers
    t0 = b.t_s
    # independent reference built from the accepted components on the boundary's own root BEFORE the lineage uses it
    op = fe.prepare_floating_evaporation(root.column, PART, available_bytes_fn=AVAIL)
    staged = fe.stage_floating_evaporation(op, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), surface_water_m=np.zeros(n), dt_s=86400.0, available_bytes_fn=AVAIL)
    ref = fsl.floating_step(root.column, root.water_state, 86400.0, added_water_m=np.zeros((L, n)), available_bytes_fn=AVAIL, **staged.native_step_kwargs())
    request = np.zeros((L, n))
    request[0, :] = 1e-5
    ref_w = ft.floating_withdraw(root.column, ref.state, request, available_bytes_fn=AVAIL)
    j0, i0 = np.array(root.history.wetting_m, copy=True), np.array(root.history.cumulative_intake_m, copy=True)
    w_boundary = np.array(root.water_state.water_m, copy=True)
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    assert isinstance(lin.head, lc.DailyHold) and st.role == "boundary" and st.cursor.ordinal == 0 and st.ctx.profile == br.BRIDGE_PROFILE and st.ctx.bed is None
    assert st.water is root.water_state and st.history is root.history  # the boundary's CURRENT water/history, nothing copied, no owner
    res = day(st, plan(0.0, 0.0, 5.0, 0), n)
    assert res.outcome == "interevent" and np.array_equal(res.state.water.water_m, ref.state.water_m) and res.state.t_s == t0 + 86400.0
    assert np.array_equal(res.state.history.wetting_m, j0) and np.array_equal(res.state.history.cumulative_intake_m, i0)  # J/I exact (no matrix/pref in the dry day)
    out = fd.request_floating_withdrawal(res.state, expected_parent=fd.expect_parent(res.state), request_m=request, expected_sequence=0, available_bytes_fn=AVAIL)
    assert np.array_equal(out.state.water.water_m, ref_w.state.water_m)
    with pytest.raises(fd.DailyRefusal, match="sequence"):
        fd.request_floating_withdrawal(out.state, expected_parent=fd.expect_parent(out.state), request_m=request, expected_sequence=0, available_bytes_fn=AVAIL)  # a duplicate withdrawal
    closed = fd.close_floating_day(out.state, expected_parent=fd.expect_parent(out.state)).state
    assert closed.cursor.ordinal == 1 and closed.cursor.last_branch == "no_rain" and closed.t_s == t0 + 86400.0
    b2 = br.hand_back(lin, closed)
    assert lin.head is b2 and b2.role == "boundary" and b2.t_s == t0 + 86400.0 and b2.cursor.ordinal == 1
    assert b2.ledger["evaporation_m3"] > 0.0 and b2.ledger["transpiration_m3"] > 0.0 and abs(b2.ledger["residual_m3"]) <= b2.ledger["residual_bound_m3"] and b2.ledger["n_days"] == 1
    assert np.array_equal(np.asarray(b2.root.water_state.water_m), ref_w.state.water_m) and b2.routing is b.routing and b2.bed is b.bed and not np.array_equal(w_boundary, np.asarray(b2.root.water_state.water_m))
    # next NATIVE event on the same chain (real rain at the day's physical clock), its own close must still close the cumulative ledger WITH the day's E/T
    sched = constant_rainfall(b2.t_s, b2.t_s + 30.0, 10.0)
    occ2, s2 = lc.begin_occurrence(lin, b2, lambda vert, topology: factory(vert, topology, sched), available_bytes_fn=AVAIL)
    h2 = complete(occ2.event, s2)
    b3 = lc.close_occurrence(lin, occ2, occ2.event.completion_capability(h2, h2.state), h2.state, available_bytes_fn=AVAIL)
    assert b3.ledger["n_occurrences"] == 2 and b3.ledger["evaporation_m3"] == b2.ledger["evaporation_m3"] and abs(b3.ledger["residual_m3"]) <= b3.ledger["residual_bound_m3"]
    assert b3.ledger["rain_m3"] > b2.ledger["rain_m3"]


def test_old_water_and_the_old_boundary_root_are_released_and_stale_heads_are_refused(tmp_path):
    def scope():
        lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
        refs = {"w": weakref.ref(b.root.water_state.water_m), "root": weakref.ref(b.root), "boundary": weakref.ref(b)}
        n = b.root.column.n_cells
        st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
        with pytest.raises(lc.LifecycleError, match="head"):
            br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)  # a second open from the same (now stale) boundary
        res = day(st, plan(0.0, 0.0, 5.0, 0), n)
        closed = fd.close_floating_day(res.state, expected_parent=fd.expect_parent(res.state)).state
        b2 = br.hand_back(lin, closed)
        with pytest.raises(lc.LifecycleError):
            br.hand_back(lin, closed)  # duplicate hand-back
        return refs, lin, b2

    refs, lin, b2 = scope()
    for _ in range(3):
        gc.collect()
    assert {k: r() is None for k, r in refs.items()} == dict.fromkeys(refs, True)  # no obsolete boundary / root / W survives once the day handed back
    assert lin.head is b2


def test_a_wet_boundary_opens_a_surface_carrying_day_state(tmp_path):
    from lifecycle_builders import build as lbuild
    from lifecycle_builders import run_to_complete

    owner, bed, factory, pref, ev0 = lbuild(tmp_path / "a", "numpy", False)  # the closed lake keeps standing water
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL)
    h = run_to_complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert float(np.asarray(b.surface_depth_m).max()) > 0.0
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)  # a wet boundary now opens (the wet-day fragments are in test_wet_day.py)
    assert st.surface is not None and isinstance(lin.head, lc.DailyHold)


def test_open_day_refusals_and_exact_admission_leave_the_head_untouched_and_retry_works(tmp_path):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    with pytest.raises(br.BridgeError):
        br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=lambda: 0)
    for bad in ({"partition": {"x": 1}}, {"run_digest": "xyz"}):
        kw = {"partition": PART, "run_digest": RUN, **bad}
        with pytest.raises(br.BridgeError):
            br.open_day(lin, b, available_bytes_fn=AVAIL, **kw)
    assert lin.head is b
    import re

    with pytest.raises(br.BridgeError) as info:
        br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=lambda: 0, reserve_bytes=0)
    need = int(re.search(r"estimated (\d+) B", str(info.value)).group(1))
    with pytest.raises(br.BridgeError):
        br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=lambda: need - 1)
    with pytest.raises(br.BridgeError) as info2:
        br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=lambda: 0, reserve_bytes=9)
    assert int(re.search(r"estimated (\d+) B", str(info2.value)).group(1)) == need + 9  # reserve once
    assert lin.head is b
    assert br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=lambda: need) is not None  # the exact need admits


def test_full_native_storm_branch_keeps_the_original_plan_pending_and_consumes_it_once(tmp_path):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    n = b.root.column.n_cells
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    pl = plan(6.0, 20.0, 5.0, 0)  # P = 6 mm > T = 5 mm: the full native storm (1080 s at 20 mm/h)
    res = day(st, pl, n)
    assert res.outcome == "requires_full_storm" and res.forcing is not None and st.t_s == b.t_s  # nothing applied to the unchanged parent
    b3 = br.begin_storm_from_day(lin, st, pl, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    assert b3.cursor.pending is not None and b3.cursor.ordinal == 0 and b3.cursor.pending.plan_digest == pl.digest
    sched = br.storm_schedule(res.forcing)
    occ, s3 = lc.begin_occurrence(lin, b3, lambda vert, topology: factory(vert, topology, sched), available_bytes_fn=AVAIL)
    assert occ.cursor is b3.cursor
    h = complete(occ.event, s3, rounds=12, span=600.0)
    b4 = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert b4.cursor.pending is not None and b4.cursor.pending.plan_digest == pl.digest and b4.ledger["rain_m3"] > b.ledger["rain_m3"]  # the storm's native rain is in the lifetime ledger
    assert float(np.asarray(b4.surface_depth_m).max()) == 0.0
    st4 = br.open_day(lin, b4, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    with pytest.raises(fd.DailyRefusal):
        fd.floating_daily_action(st4, expected_parent=fd.expect_parent(st4), plan=pl, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)  # the ordinary action refuses a pending plan
    res4 = br.day_after_storm(st4, plan=pl, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    assert res4.outcome == "interevent" and res4.branch == "storm" and res4.state.t_s == st4.t_s + 86400.0
    closed = fd.close_floating_day(res4.state, expected_parent=fd.expect_parent(res4.state)).state
    assert closed.cursor.ordinal == 1 and closed.cursor.pending is None and closed.cursor.last_plan_digest == pl.digest and closed.cursor.last_branch == "storm"  # the ORIGINAL plan consumed once
    b5 = br.hand_back(lin, closed)
    assert abs(b5.ledger["residual_m3"]) <= b5.ledger["residual_bound_m3"]


def test_the_historical_profile_and_digest_bodies_are_unchanged():
    from dbuild import PART as P0
    from dbuild import make

    owner, bed = make()
    old = fd._physics_config_sha(owner, P0, owner.column.backend, "w" * 64, (3, 4))
    parts = fd._physics_config_sha_parts(P0, owner.column.backend, "w" * 64, (3, 4), config_sha256=owner.config.sha256, root_column_sha256=owner.column.sha256, matrix_op_sha256=owner.matrix_op.sha256,
                                         generation=owner.generation, area=owner.column.cell_area_m2)
    assert old == parts and fd.PROFILE == "native_floating_daily_initialized_soil/1"
    assert fd._physics_config_sha_parts(P0, "numpy", "w" * 64, (3, 4), config_sha256=owner.config.sha256, root_column_sha256=owner.column.sha256, matrix_op_sha256=owner.matrix_op.sha256, generation=0,
                                        area=owner.column.cell_area_m2, profile=br.BRIDGE_PROFILE) != old  # the bridge profile is explicitly distinct
