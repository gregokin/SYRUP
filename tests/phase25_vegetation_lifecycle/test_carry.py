# ruff: noqa: RUF059
"""Phase 2.5 vegetation lifecycle: carry through the ACTUAL closed-event refresh, the next native event (splash cover), the surface-only update, the old file/export profiles and the bridge day (E shading). Real thin-lake storms, real LEAF
projection, no fakes. Expected values are independent (closed-form circle areas, prescribed-cover control events/days)."""
from __future__ import annotations

import numpy as np
import pytest
import veg_lc_builders as vb
from lifecycle_builders import AVAIL, build, run_to_complete

from maple_syrup import floating_daily as fd
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import native_floating_soil as nfs
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup.pond_event import PondEvent
from maple_syrup.rainfall import constant_rainfall

BIG = vb.BIG
PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
RUN = "ab" * 32


def lake_boundary(tmp, backend="numpy", windowed=False):
    owner, bed, factory, pref, ev0 = build(tmp, backend, windowed, splash=True, mobile_kg=0.1, loose=True)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL)
    h = run_to_complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    return lin, b, ev0


def event_builder(ev0, sched):
    def build_event(vert, topology, cover):
        return PondEvent(ev0.context, None, ev0.sediment, sched, topology, control=ev0.control, policy=ev0.policy, splash=ev0.splash, vegetation_cover_fraction=cover, floating_vertical=vert)

    return build_event


def second_storm(lin, nb, ev0, *, scheduled=True):
    sched = constant_rainfall(nb.t_s, nb.t_s + 20.0, 60.0)
    occ, st = lc.begin_occurrence(lin, nb, vl.event_factory_for(nb, event_builder(ev0, sched)), available_bytes_fn=AVAIL)
    return occ, st, run_to_complete(occ.event, st)


def test_the_bound_splash_cover_reaches_the_native_event_and_a_foreign_cover_is_refused_before_the_head_moves(tmp_path, leaf):
    lin, b, ev0 = lake_boundary(tmp_path / "a")
    s = vb.stack(leaf, b)
    nb = vl.apply_vegetation_generation(lin, b, vb.binding(leaf, b, s), memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    ny, nx = nb.surface_depth_m.shape
    cover = vl.bound_splash_cover(nb)
    sched = constant_rainfall(nb.t_s, nb.t_s + 20.0, 60.0)
    # independent: the actual LEAF visible live + dead-standing cover = the closed-form add-and-cap of the circle areas
    assert np.allclose(np.asarray(cover).reshape(-1), np.asarray(s["proj"].splash_cover), rtol=0, atol=0) and np.allclose(np.asarray(cover).reshape(-1), _total(s["dx"], ny, nx), rtol=1e-9, atol=1e-15)
    # a factory that prescribes ANOTHER cover (zero default, or a scaled one) is refused before any head change; the head is retryable
    for wrong in (None, np.asarray(cover) * 0.5):
        with pytest.raises(lc.LifecycleError, match="bound splash cover"):
            lc.begin_occurrence(lin, nb, lambda vert, topology, w=wrong: event_builder(ev0, sched)(vert, topology, w), available_bytes_fn=AVAIL)
        assert lin.head is nb
    occ, st = lc.begin_occurrence(lin, nb, vl.event_factory_for(nb, event_builder(ev0, sched)), available_bytes_fn=AVAIL)
    assert lin.head is occ and occ.vegetation is vl.current_record(nb) and np.array_equal(np.asarray(occ.event.veg), np.asarray(cover))  # the ACTUAL native cover (hashed into the config identity)
    assert occ.event.config_sha256 and occ.vegetation.generation == 1


def _total(dx, ny, nx):
    import math

    f = lambda r: math.pi * (r / dx) ** 2
    t = np.zeros(ny * nx)
    t[0] = f(0.4 * dx)
    t[nx + 1] = min(f(0.3 * dx) + f(0.35 * dx), 1.0)
    return t


def test_the_held_g_and_the_sealed_preferential_law_survive_the_actual_property_refresh(tmp_path, leaf, monkeypatch):
    lin, b, ev0 = lake_boundary(tmp_path / "a")
    nb = vl.apply_vegetation_generation(lin, b, vb.binding(leaf, b, vb.stack(leaf, b)), memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    g0 = np.array(fm.held_cell_vegetation_factor(nb.root.matrix_op), copy=True)
    law0 = [np.array(a, copy=True) for a in nb.root.pref_op._law]
    rec0 = vl.current_record(nb)
    fresh_calls, prep_calls = [], []
    real_fresh, real_prep = nfs._fresh_op, fpi.prepare_floating_preferential
    monkeypatch.setattr(nfs, "_fresh_op", lambda cfg, col, ks=None, vegetation=None: (fresh_calls.append(vegetation is not None), real_fresh(cfg, col, ks, vegetation))[1])
    monkeypatch.setattr(fpi, "prepare_floating_preferential", lambda *a, **k: (prep_calls.append(1), real_prep(*a, **k))[1])
    occ, st, h = second_storm(lin, nb, ev0)
    b2 = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert b2.root.column.sha256 != nb.root.column.sha256 and b2.root.generation == nb.root.generation + 1  # a GENUINE property refresh happened (new prepared root)
    assert fresh_calls == [True] and prep_calls == []  # the held g was passed to the matrix factory; NO species contribution was re-streamed / rebuilt for the ordinary soil refresh
    assert b2.root.matrix_op.layouts["vegetation_factor"] == "cell" and np.array_equal(np.asarray(fm.held_cell_vegetation_factor(b2.root.matrix_op)), g0)
    assert all(np.array_equal(np.asarray(a), c) for a, c in zip(b2.root.pref_op._law, law0, strict=True))  # the sealed preferential law carried to the new column unchanged
    assert b2.root.vegetation is rec0 and b2.root.vegetation.generation == 1  # one record, no counter bump, no history list
    # CURRENT attribution after the refresh: the stable generation identity is unchanged, the CURRENT law identity is read from the actual genuine boundary (the NEW column / operators), the apply-time ids stay explicit provenance
    a0, a2 = vl.vegetation_attribution(nb), vl.vegetation_attribution(b2)
    assert a0["recipe_id"] == a2["recipe_id"] == rec0.recipe_id and a0["generation"] == a2["generation"] == 1
    assert a2["current_column_sha256"] == b2.root.column.sha256 != a0["current_column_sha256"] and a2["current_matrix_op_sha256"] == b2.root.matrix_op.sha256 != a0["current_matrix_op_sha256"]
    assert a2["current_preferential_sha256"] == b2.root.pref_op.sha256 and a2["current_property_generation"] == b2.root.generation and a2["current_law_id"] != a0["current_law_id"]
    assert a2["applied"]["column_sha256"] == nb.root.column.sha256 and a2["applied"]["matrix_op_sha256"] == nb.root.matrix_op.sha256  # provenance, never labelled current
    assert vl.recipe_id_of(rec0.binding_sha256, 1, None) == rec0.recipe_id  # reproducible from the compact binding + counter alone (no discarded property column needed)
    # the next occurrence from the refreshed boundary still runs the bound cover
    occ3, _ = lc.begin_occurrence(lin, b2, vl.event_factory_for(b2, event_builder(ev0, constant_rainfall(b2.t_s, b2.t_s + 10.0, 30.0))), available_bytes_fn=AVAIL)
    assert occ3.vegetation is rec0


def test_a_new_legitimate_surface_with_the_same_biology_is_a_surface_only_update_and_older_generations_are_refused(tmp_path, leaf):
    lin, b, ev0 = lake_boundary(tmp_path / "a")
    s1 = vb.stack(leaf, b)
    bd1 = vb.binding(leaf, b, s1)
    nb = vl.apply_vegetation_generation(lin, b, bd1, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    occ, st, h = second_storm(lin, nb, ev0)
    b2 = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    z1, z2 = vb.surface_of(nb), vb.surface_of(b2)
    assert not np.array_equal(z1, z2)  # the authoritative MAPLE surface genuinely moved (the held routing surface is not what is bound)
    # the OLD binding is now stale against the current surface
    with pytest.raises(vl.VegetationLifecycleError, match="CURRENT committed surface"):
        vl.apply_vegetation_generation(lin, b2, bd1, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    # a projection of the SAME biology (the same initial state) on the NEW surface is a legitimate update: no strict sequence increase is required
    s2 = vb.stack(leaf, b2, state=s1["state"], cat=s1["cat"])
    bd2 = vb.binding(leaf, b2, s2)
    assert bd2.sha256 != bd1.sha256 and bd2.surface_sha256 != bd1.surface_sha256
    nb2 = vl.apply_vegetation_generation(lin, b2, bd2, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    rec = vl.current_record(nb2)
    assert lin.head is nb2 and rec.generation == 2 and rec.parent_recipe_id == vl.current_record(nb).recipe_id and nb2.root.water_state is b2.root.water_state
    # a different biology (another plant population) from an initial-state projection has no ordering proof against the current one
    other = vb.plant_layout(3, 5, s1["dx"])[:2]
    s3 = vb.stack(leaf, b2, plants=other, cat=s1["cat"])
    with pytest.raises(vl.VegetationLifecycleError, match="ordering proof"):
        vl.apply_vegetation_generation(lin, nb2, vb.binding(leaf, b2, s3), memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    # other site controls are refused (fixed for one lineage); the head is untouched
    with pytest.raises(vl.VegetationLifecycleError, match="site controls"):
        vl.apply_vegetation_generation(lin, nb2, vb.binding(leaf, b2, s2, ctl=vb.controls(b2, a3=0.3)), memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    assert lin.head is nb2


def test_old_file_and_held_profiles_refuse_a_vegetation_bound_head_closed(tmp_path, leaf):
    lin, b, ev0 = lake_boundary(tmp_path / "a")
    assert lc.saveable_head(lin)[0] == "boundary"  # without a binding: the old profile is untouched
    nb = vl.apply_vegetation_generation(lin, b, vb.binding(leaf, b, vb.stack(leaf, b)), memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    with pytest.raises(lc.LifecycleError, match="vegetation-bound head is not saveable"):
        lc.saveable_head(lin)
    with pytest.raises(lc.LifecycleError, match="held-context export does not support"):
        lc.export_boundary_held_context(nb, None, available_bytes_fn=AVAIL)


def _dry(tmp, leaf, bound):
    from bridge_builders import dry_boundary

    lin, b, factory = dry_boundary(tmp, "numpy", False)
    if bound:
        b = vl.apply_vegetation_generation(lin, b, vb.binding(leaf, b, vb.stack(leaf, b)), memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    return lin, b


def test_the_bound_shading_cover_drives_the_day_like_a_prescribed_control_and_a_caller_cover_cannot_override_it(tmp_path, leaf):
    lin_c, b_c = _dry(tmp_path / "c", leaf, False)  # control lineage: NO binding, the same live shading cover prescribed explicitly
    lin_v, b_v = _dry(tmp_path / "v", leaf, True)
    n = b_v.root.column.n_cells
    rec = vl.current_record(b_v)
    shade = np.array(rec.shade_cover, copy=True)
    pet = np.full(n, 3e-8)
    pl = rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=0)
    st_c = br.open_day(lin_c, b_c, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    st_v = br.open_day(lin_v, b_v, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    r_c = br.day_action(st_c, plan=pl, pet_terrain_m_s=pet, cover=shade, available_bytes_fn=AVAIL)
    r_v = br.day_action(st_v, plan=pl, pet_terrain_m_s=pet, cover=None, available_bytes_fn=AVAIL)  # the bound live shading cover
    assert r_c.outcome == r_v.outcome == "interevent"
    assert np.array_equal(np.asarray(r_c.state.water.water_m), np.asarray(r_v.state.water.water_m))  # identical E / drainage: same formulas, the bound cover is the prescribed one
    assert r_c.state.ledger.evaporation_m3 == r_v.state.ledger.evaporation_m3 and r_v.state.ledger.evaporation_m3 > 0.0
    assert np.array_equal(np.asarray(r_v.state.history.wetting_m), np.asarray(r_c.state.history.wetting_m)) and np.array_equal(np.asarray(r_v.state.history.cumulative_intake_m), np.asarray(r_c.state.history.cumulative_intake_m))  # zero extra intake
    # a caller-prescribed cover may not silently replace the accepted profile (bridge AND the lower-level daily action)
    with pytest.raises(br.BridgeError, match="bound"):
        br.day_action(st_v, plan=pl, pet_terrain_m_s=pet, cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    with pytest.raises(fd.DailyRefusal, match="bound"):
        fd.floating_daily_action(st_v, expected_parent=fd.expect_parent(st_v), plan=pl, pet_terrain_m_s=pet, cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    # frame attribution: the bound lineage's frame property identity names the vegetation law; an unbound lineage is the accepted value
    assert st_c.frame_identity()["property_sha256"] == b_c.root.column.sha256
    assert st_v.frame_identity()["property_sha256"] == b_v.root.column.sha256  # the OLD frame property meaning is untouched by a vegetation binding
    att = vl.vegetation_attribution(st_v)
    assert att["generation"] == 1 and att["recipe_id"] == rec.recipe_id and vl.vegetation_attribution(st_c) is None  # the explicit attribution for a NEW frame profile
    # the closed day hands back to the SAME head with the vegetation record carried (and the whole day closes the lifetime ledger)
    out = fd.request_floating_withdrawal(r_v.state, expected_parent=fd.expect_parent(r_v.state), request_m=np.zeros((b_v.root.column.n_layers, n)), expected_sequence=0, available_bytes_fn=AVAIL)
    closed = fd.close_floating_day(out.state, expected_parent=fd.expect_parent(out.state)).state
    b3 = br.hand_back(lin_v, closed)
    assert b3.root.vegetation is rec and b3.root.matrix_op is b_v.root.matrix_op and abs(b3.ledger["residual_m3"]) <= b3.ledger["residual_bound_m3"]


def test_negative_control_without_the_held_g_hook_the_refresh_would_revert_to_the_configuration_scalar(tmp_path, leaf, monkeypatch):
    lin, b, ev0 = lake_boundary(tmp_path / "a")
    nb = vl.apply_vegetation_generation(lin, b, vb.binding(leaf, b, vb.stack(leaf, b)), memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    monkeypatch.setattr(fm, "held_cell_vegetation_factor", lambda op: None)  # the pre-hook behaviour of the refresh
    occ, st, h = second_storm(lin, nb, ev0)
    b2 = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert b2.root.matrix_op.layouts["vegetation_factor"] == "scalar"  # the applied per-cell law silently reverted: exactly what the carry hook prevents


def test_a_wet_interevent_day_runs_on_the_bound_law_and_cover_and_closes_the_lifetime_ledger(tmp_path, leaf):
    from bridge_builders import wet_boundary

    from maple_syrup.experimental_storm import ExperimentalControl

    ctl = ExperimentalControl(max_dt_s=3600.0, min_dt_s=1 / 1024, max_retries=30, max_steps=2_000_000)
    lin, b, factory = wet_boundary(tmp_path / "w", "numpy", False)
    assert float(np.asarray(b.surface_depth_m).max()) > 0.0
    nb = vl.apply_vegetation_generation(lin, b, vb.binding(leaf, b, vb.stack(leaf, b)), memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    rec = vl.current_record(nb)
    n = nb.root.column.n_cells
    st = br.open_day(lin, nb, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    assert st.surface is not None and st.ctx.hold.vegetation is rec  # standing water: the wet fragments carry the bound record
    pl = rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=0)
    with pytest.raises(br.BridgeError, match="bound"):
        br.day_action(st, plan=pl, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), event_factory=factory, day_control=ctl, available_bytes_fn=AVAIL)  # a caller cover is refused on the wet path too
    res = br.day_action(st, plan=pl, pet_terrain_m_s=np.full(n, 3e-8), cover=None, event_factory=factory, day_control=ctl, available_bytes_fn=AVAIL)
    assert res.outcome == "interevent" and res.state.ledger.evaporation_m3 > 0.0 and res.state.t_s == nb.t_s + 86400.0
    out = fd.request_floating_withdrawal(res.state, expected_parent=fd.expect_parent(res.state), request_m=np.zeros((nb.root.column.n_layers, n)), expected_sequence=0, available_bytes_fn=AVAIL)
    closed = fd.close_floating_day(out.state, expected_parent=fd.expect_parent(out.state)).state
    b2 = br.hand_back(lin, closed)
    assert b2.root.vegetation is rec and abs(b2.ledger["residual_m3"]) <= b2.ledger["residual_bound_m3"]


def test_the_old_every_n_frame_profile_refuses_a_vegetation_bound_lineage_and_stays_unchanged_otherwise(tmp_path, leaf):
    from maple_syrup import floating_daily_output as fo
    from maple_syrup import soil_moisture_output as g0

    lim = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**7, max_manifest_bytes=8192, chunk_bytes=4096, host_memory_budget_bytes=10**7, host_reserve_bytes=0, disk_reserve_bytes=0)
    (tmp_path / "oc").mkdir()
    (tmp_path / "ov").mkdir()
    lin_c, b_c = _dry(tmp_path / "c", leaf, False)
    st_c = br.open_day(lin_c, b_c, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    out = fo.FloatingMoistureOutput.create(tmp_path / "oc", st_c, expected_parent=fd.expect_parent(st_c), every_n_days=2, limits=lim)  # unbound: the old behaviour
    assert out is not None
    lin_v, b_v = _dry(tmp_path / "v", leaf, True)
    st_v = br.open_day(lin_v, b_v, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    with pytest.raises(fo.MoistureOutputError, match="vegetation generation"):
        fo.FloatingMoistureOutput.create(tmp_path / "ov", st_v, expected_parent=fd.expect_parent(st_v), every_n_days=2, limits=lim)  # never a silently unattributed frame


def test_after_a_full_storm_and_its_closed_day_the_boundary_is_eligible_again_genuine_path(tmp_path, leaf):
    from bridge_builders import RUN as R
    from bridge_builders import complete, dry_boundary

    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    n = b.root.column.n_cells
    st = br.open_day(lin, b, partition=PART, run_digest=R, available_bytes_fn=AVAIL)
    pl = rc.plan_daily_rain(6.0, 20.0, 5.0, run_digest=R, ordinal=0)  # P > T: the full native storm
    res = br.day_action(st, plan=pl, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    assert res.outcome == "requires_full_storm"
    b3 = br.begin_storm_from_day(lin, st, pl, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    bd = vb.binding(leaf, b3, vb.stack(leaf, b3))
    with pytest.raises(vl.VegetationLifecycleError, match="nothing pending"):
        vl.apply_vegetation_generation(lin, b3, bd, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)  # a pending storm is not eligible
    sched = br.storm_schedule(res.forcing)
    occ, s3 = lc.begin_occurrence(lin, b3, lambda vert, topology: factory(vert, topology, sched), available_bytes_fn=AVAIL)
    h = complete(occ.event, s3, rounds=12, span=600.0)
    b4 = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert b4.storm_match is not None and b4.cursor.pending is not None  # matched, genuinely completed, the day not yet taken: still pending
    bd4 = vb.binding(leaf, b4, vb.stack(leaf, b4))
    with pytest.raises(vl.VegetationLifecycleError, match="nothing pending"):
        vl.apply_vegetation_generation(lin, b4, bd4, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    st4 = br.open_day(lin, b4, partition=PART, run_digest=R, available_bytes_fn=AVAIL)
    res4 = br.day_after_storm(st4, plan=pl, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    closed = fd.close_floating_day(res4.state, expected_parent=fd.expect_parent(res4.state)).state
    b5 = br.hand_back(lin, closed)
    assert b5.storm_match is None and b5.cursor.pending is None  # the day's hand-back CLEARS the proof: it was proof, not a permanent block
    nb = vl.apply_vegetation_generation(lin, b5, vb.binding(leaf, b5, vb.stack(leaf, b5)), memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    assert lin.head is nb and vl.current_record(nb).generation == 1 and nb.cursor is b5.cursor


def test_default_frame_property_stays_the_root_column_after_a_refresh_and_attribution_follows_the_current_holder(tmp_path, leaf):
    lin, b, ev0 = lake_boundary(tmp_path / "a")
    nb = vl.apply_vegetation_generation(lin, b, vb.binding(leaf, b, vb.stack(leaf, b)), memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    occ, st, h = second_storm(lin, nb, ev0)
    att_occ = vl.vegetation_attribution(occ)  # an Occurrence reads the vertical's canonical held template
    assert att_occ["current_column_sha256"] == nb.root.column.sha256 and att_occ["recipe_id"] == vl.current_record(nb).recipe_id
    b2 = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert vl.vegetation_attribution(b2)["current_column_sha256"] == b2.root.column.sha256
