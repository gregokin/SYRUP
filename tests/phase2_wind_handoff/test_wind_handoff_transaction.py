"""WH1: the staged wind handoff on REAL native lifecycle states (Dense NumPy). Authored, NOT executed by its author.

The wind result is an instance of the actual native `WindEventFinalizationResult` built directly (see `wind_builders`); the genuine MAPLE finalizer has not been run here. Declared perturbations (the repository's own precedent) monkeypatch only the provider
output so a property change is deterministic."""
from __future__ import annotations

import copy
import gc
import weakref

import numpy as np
import pytest
import wind_builders as wb
from lcp_builders import PART, run_day

from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_wind_handoff as wh
from maple_syrup import native_floating_soil as nfs

AVAIL = wb.AVAIL
CELL = (1, 1)
DST = (1, 2)
FLAT = CELL[0] * 4 + CELL[1]
FLAT_DST = DST[0] * 4 + DST[1]


def test_an_unchanged_wind_carries_every_object_and_value_and_only_advances_the_clock():
    ns = wb.started(depth=0.003)
    b = ns.b
    before = wb.snapshot(b)
    child = wb.child_same(b.bed)
    cand = wb.stage(ns, child, end_s=120.0)
    assert ns.lin.head is b and not cand.consumed and cand.role == "boundary"  # staging adopts nothing
    wb.assert_same(before, wb.snapshot(b))
    rec = cand.receipt()
    rz = rec["water"]["zero_time_reconcile"]
    assert rec["properties"]["n_dirty"] == 0 and rec["properties"]["root_replaced"] is False and rz["surface_return_m3"] == 0.0 and rz["bottom_export_m3"] == 0.0
    assert rec["water"]["wind_water_physics"] == "none" and "physical_update" not in rec["water"] and rz["performed"] is False and rz["n_columns"] == 0  # nothing reconciled: the receipt says exactly that
    assert rec["clocks"]["end_s_hex"] == (120.0).hex() and rec["clocks"]["soil_clock_after_hex"] == (120.0).hex()
    nb = wh.adopt_wind_handoff(ns.lin, b, cand)
    assert ns.lin.head is nb and type(nb) is lc.Boundary and nb.role == "boundary" and nb is not b and cand.consumed and cand._cell["parts"] is None
    assert nb.t_s == 120.0 and nb.root.water_state.t_s == 120.0 and nb.seq == b.seq + 1
    assert np.array_equal(nb.root.water_state.water_m, before["W"]) and nb.root.water_state is not b.root.water_state
    assert nb.root.column is b.root.column and nb.root.matrix_op is b.root.matrix_op and nb.root.history is b.root.history and nb.root.generation == b.root.generation  # constant case: the prepared root objects, bit for bit
    assert np.array_equal(nb.root.history.wetting_m, before["J"]) and np.array_equal(nb.root.history.cumulative_intake_m, before["I"])  # J/I retained without resets
    assert all(x is y for x, y in zip(nb.routing, b.routing, strict=True)) and nb.cursor is b.cursor  # the last hydraulic routing tuple and the nominal cursor are untouched
    assert np.array_equal(nb.surface_depth_m, before["H"]) and np.array_equal(nb.bed.water.depth_m, before["H"]) and nb.qx_m2_s is b.qx_m2_s and nb.qy_m2_s is b.qy_m2_s  # every surface-water leaf carried bitwise
    assert nb.bed is child and nb.root.bed is child and nb.bed is not b.bed
    assert nb.ledger["n_refreshes"] == before["ledger"]["n_refreshes"] + 1
    for k, v in before["ledger"].items():
        if k not in ("n_refreshes", "residual_m3", "residual_bound_m3"):
            assert nb.ledger[k] == v, k  # no rain, drainage, export or return entered the lifecycle ledger
    assert nb.provenance["cold_anchor_witness"] == b.provenance["cold_anchor_witness"] and nb.provenance["property_basis"] == "initial_full"
    assert type(nb.provenance["execution_epoch"]) is str and len(nb.provenance["execution_epoch"]) == 64 and nb.provenance["cold_content_hash"] is None
    lc.saveable_head(ns.lin)


def test_a_native_class_exchange_between_two_columns_refreshes_only_those_columns():
    ns = wb.started()
    b = ns.b
    before = wb.snapshot(b)
    old_fc, old_sat = np.array(b.root.column._maps["theta_field_capacity"], copy=True), np.array(b.root.column._maps["theta_saturation"], copy=True)
    child = wb.child_move(b.bed, ns.ctx, CELL, DST, klass=0)  # ACTUAL native pickup + deposit + ledger: a valid bed whose two columns change composition while the pair's net elevation is conserved
    parent_surface = np.asarray(b.bed.voxel_column.mass_kg).sum(axis=(2, 3))
    child_surface = np.asarray(child.voxel_column.mass_kg).sum(axis=(2, 3))
    assert abs(float(child_surface.sum() - parent_surface.sum())) <= 1e-9 * float(np.abs(parent_surface).sum())  # the exchange conserves mass
    old, nb, cand = wb.wind(ns, child, end_s=75.0)
    rec = cand.receipt()
    assert rec["properties"]["n_dirty"] == 2 and 1 <= rec["properties"]["n_changed"] <= 2 and rec["properties"]["root_replaced"] is True
    new_fc, new_sat = np.asarray(nb.root.column._maps["theta_field_capacity"]), np.asarray(nb.root.column._maps["theta_saturation"])
    differs = np.flatnonzero(np.any(new_fc != old_fc, axis=0) | np.any(new_sat != old_sat, axis=0))
    assert set(differs.tolist()) <= {FLAT, FLAT_DST}  # the provider output of every other column is bit for bit the held one
    assert nb.root.column is not old.root.column and nb.root.column.sha256 != before["col"] and nb.root.matrix_op.sha256 != before["op"] and nb.root.generation == before["gen"] + 1
    assert nb.provenance["property_basis"] == "boundary_selected"
    assert np.array_equal(nb.root.history.wetting_m, before["J"]) and np.array_equal(nb.root.history.cumulative_intake_m, before["I"]) and nb.root.history is not old.root.history  # rebound bit for bit, never reset
    others = np.ones(12, bool)
    others[[FLAT, FLAT_DST]] = False
    assert np.array_equal(nb.root.water_state.water_m[:, others], before["W"][:, others])  # unchanged columns keep their water bit for bit
    assert nb.root.water_state.t_s == 75.0 and nb.t_s == 75.0
    area = float(nb.root.column.cell_area_m2)
    stored = float(nfs.fsl.floating_storage_m3(nb.root.column, nb.root.water_state)) + float(np.asarray(nb.surface_depth_m).sum()) * area
    before_total = float(before["W"][:, np.asarray(old.root.column._maps["active"])].sum() + before["H"].sum()) * area
    exported = nb.ledger["bottom_export_m3"] - before["ledger"]["bottom_export_m3"]
    assert stored + exported == pytest.approx(before_total, rel=1e-12)  # soil + surface closes; any export is counted once
    assert abs(nb.ledger["residual_m3"]) <= nb.ledger["residual_bound_m3"]
    # a second, content-identical wind runs NO cascade and keeps the refreshed root objects (no duplicate refresh)
    root_after = nb.root
    old2, nb2, cand2 = wb.wind(ns, wb.child_same(nb.bed), end_s=100.0, occurrence_id="occ-2", start_s=75.0)
    assert cand2.receipt()["properties"]["n_dirty"] == 0 and nb2.root.column is root_after.column and nb2.root.matrix_op is root_after.matrix_op and nb2.root.generation == root_after.generation
    assert np.array_equal(nb2.root.water_state.water_m, nb.root.water_state.water_m) and nb2.ledger["bottom_export_m3"] == nb.ledger["bottom_export_m3"] and nb2.t_s == 100.0


def test_an_equivalent_depth_change_publishes_the_actual_depth_without_a_cascade(monkeypatch):
    ns = wb.started(base=0.20)  # a soil deeper than H = 0.16 m: a small depth change leaves every slab and the bottom flag equal
    b = ns.b
    before = wb.snapshot(b)
    old_fc, old_sat = np.array(b.root.column._maps["theta_field_capacity"], copy=True), np.array(b.root.column._maps["theta_saturation"], copy=True)
    child = wb.child_move(b.bed, ns.ctx, CELL, DST, 1e-4)  # ACTUAL native pickup + deposit + ledger (a valid bed): the two columns change depth
    monkeypatch.setattr(nfs, "_batch_properties", lambda batch, cfg, K, pb: (old_fc[:, batch.indices], old_sat[:, batch.indices], None))  # DECLARED: the provider returns the held properties, so only the depth differs
    old, nb, cand = wb.wind(ns, child, end_s=60.0)
    monkeypatch.undo()
    rec = cand.receipt()
    assert rec["properties"]["n_dirty"] >= 1 and rec["properties"]["n_changed"] == 0 and rec["properties"]["n_depth_changed"] >= 1 and rec["properties"]["root_replaced"] is True
    rz = rec["water"]["zero_time_reconcile"]
    assert rz["performed"] is False and rz["surface_return_m3"] == 0.0 and rz["bottom_export_m3"] == 0.0 and nb.role == "boundary"  # equivalent geometry: nothing reconciled, returned or exported
    assert np.array_equal(nb.root.water_state.water_m, before["W"]) and np.array_equal(nb.root.history.wetting_m, before["J"]) and np.array_equal(nb.root.history.cumulative_intake_m, before["I"])
    both = [FLAT, FLAT_DST]
    assert np.any(nb.root.depth_m[both] != before["D"][both]) and np.array_equal(np.delete(nb.root.depth_m, both), np.delete(before["D"], both))  # the published canonical depth is the ACTUAL one
    assert nb.ledger["bottom_export_m3"] == before["ledger"]["bottom_export_m3"] and nb.ledger["surface_return_m3"] == before["ledger"]["surface_return_m3"]


def test_a_closed_profile_capacity_return_wakes_the_flow_and_is_added_once(monkeypatch):
    ns = wb.started(bottom="closed", saturation=0.999999)
    b = ns.b
    before = wb.snapshot(b)
    real = nfs._batch_properties
    monkeypatch.setattr(nfs, "_batch_properties", lambda batch, cfg, K, pb: (lambda f, s, k: (f * 0.5, s * 0.5, k))(*real(batch, cfg, K, pb)))  # DECLARED perturbation: the capacity genuinely shrinks
    child = wb.child_move(b.bed, ns.ctx, CELL, DST, klass=0)
    old, nb, cand = wb.wind(ns, child, end_s=90.0)
    monkeypatch.undo()
    rec = cand.receipt()
    area = float(nb.root.column.cell_area_m2)
    rz = rec["water"]["zero_time_reconcile"]
    ret = rz["surface_return_m3"]
    assert rec["water"]["wind_water_physics"] == "none" and rz["performed"] is True and rz["n_columns"] >= 1  # no wind water physics, but the explicit zero-time reconcile really returned water: the receipt names it
    assert ret > 0.0 and rz["bottom_export_m3"] == 0.0 and nb.role == "pending_recession" and not nb.eligible_for_interevent
    assert (float(nb.surface_depth_m.sum()) - float(before["H"].sum())) * area == pytest.approx(ret, rel=1e-9)  # returned to the surface exactly once
    assert np.array_equal(nb.bed.water.depth_m, nb.surface_depth_m) and nb.ledger["surface_return_m3"] == before["ledger"]["surface_return_m3"] + ret
    assert abs(nb.ledger["residual_m3"]) <= nb.ledger["residual_bound_m3"]
    assert np.array_equal(nb.root.history.wetting_m, before["J"]) and np.array_equal(nb.root.history.cumulative_intake_m, before["I"])
    with pytest.raises(wh.WindHandoffError, match="pending recession"):  # a woken boundary is not wind eligible until a real recession completes
        wb.stage(ns, wb.child_same(nb.bed), end_s=200.0, occurrence_id="occ-2", start_s=90.0)
    with pytest.raises(br.BridgeError, match="pending recession"):
        br.open_day(ns.lin, nb, partition=PART, run_digest=wb.RUN, available_bytes_fn=AVAIL)
    assert ns.lin.head is nb


def test_an_open_profile_export_is_counted_once_and_never_doubled(monkeypatch):
    ns = wb.started(base=0.20, bottom="open", saturation=0.9)
    b = ns.b
    before = wb.snapshot(b)
    real = nfs._batch_properties
    monkeypatch.setattr(nfs, "_batch_properties", lambda batch, cfg, K, pb: (lambda f, s, k: (f * 0.5, s * 0.5, k))(*real(batch, cfg, K, pb)))
    old, nb, cand = wb.wind(ns, wb.child_move(b.bed, ns.ctx, CELL, DST, klass=0), end_s=45.0)
    monkeypatch.undo()
    export = cand.receipt()["water"]["zero_time_reconcile"]["bottom_export_m3"]
    assert export > 0.0 and nb.ledger["bottom_export_m3"] == before["ledger"]["bottom_export_m3"] + export
    assert abs(nb.ledger["residual_m3"]) <= nb.ledger["residual_bound_m3"]
    _o, n2, c2 = wb.wind(ns, wb.child_same(nb.bed), end_s=50.0, occurrence_id="occ-2", start_s=45.0)
    assert c2.receipt()["water"]["zero_time_reconcile"]["bottom_export_m3"] == 0.0 and n2.ledger["bottom_export_m3"] == nb.ledger["bottom_export_m3"]  # no double drainage by the next handoff


def test_the_next_event_and_day_accept_the_real_wind_end_clock_and_the_cursor_does_not_move():
    ns = wb.started()
    old, nb, _cand = wb.wind(ns, wb.child_same(ns.b.bed), end_s=137.5)
    assert nb.t_s == 137.5 and nb.cursor is None
    st = br.open_day(ns.lin, nb, partition=PART, run_digest=wb.RUN, available_bytes_fn=AVAIL)
    assert st.water.t_s == 137.5 and st.cursor.ordinal == 0  # the wind consumed no calendar day
    closed = run_day(st, 0).state
    assert closed.water.t_s == 137.5 + 86400.0
    nb2 = br.hand_back(ns.lin, closed)
    ns.b = nb2
    assert nb2.cursor.ordinal == 1
    _o, nb3, _c = wb.wind(ns, wb.child_same(nb2.bed), end_s=137.5 + 86400.0 + 30.0, start_s=137.5 + 86400.0, occurrence_id="occ-2")  # the next wind starts at the day's real end clock
    assert nb3.t_s == 137.5 + 86400.0 + 30.0 and nb3.cursor is nb2.cursor
    ns2 = wb.started()
    _o2, nbs, _c2 = wb.wind(ns2, wb.child_same(ns2.b.bed), end_s=137.5)
    occ, state = lc.begin_occurrence(ns2.lin, nbs, lambda vert, topology: ns2.factory(vert, topology), available_bytes_fn=AVAIL)  # a Pond event accepts the real end clock
    assert float(state.t_s) == 137.5 and np.array_equal(state.hydraulic.soil_water_m, np.asarray(nbs.root.water_state.water_m).sum(axis=0).reshape(3, 4))


def test_stale_duplicate_and_wrong_parent_adoptions_are_refused():
    ns = wb.started()
    old = ns.b
    c1 = wb.stage(ns, wb.child_same(old.bed), end_s=60.0, occurrence_id="a")
    c2 = wb.stage(ns, wb.child_same(old.bed), end_s=70.0, occurrence_id="b")  # two candidates of the same parent
    nb = wh.adopt_wind_handoff(ns.lin, old, c1)
    ns.b = nb
    with pytest.raises(lc.LifecycleError):
        wh.adopt_wind_handoff(ns.lin, old, c1)  # duplicate adoption: the parent is no longer the head
    with pytest.raises(lc.LifecycleError):
        wh.adopt_wind_handoff(ns.lin, old, c2)  # stale: another candidate was already adopted
    with pytest.raises(wh.WindHandoffError, match="already adopted"):
        wh.adopt_wind_handoff(ns.lin, nb, c1)  # the head is right but the candidate is consumed
    with pytest.raises(wh.WindHandoffError, match="another parent"):
        wh.adopt_wind_handoff(ns.lin, nb, c2)
    with pytest.raises(lc.LifecycleError):
        wb.stage(ns, wb.child_same(old.bed), end_s=80.0, occurrence_id="c", parent=old)  # staging against the old parent
    assert ns.lin.head is nb
    with pytest.raises(wh.WindHandoffError, match="duplicate"):
        wb.stage(ns, wb.child_same(nb.bed), end_s=100.0, occurrence_id="a", start_s=60.0)  # the occurrence identity is consumed once per lineage
    ok = wb.stage(ns, wb.child_same(nb.bed), end_s=100.0, occurrence_id="d", start_s=60.0)
    other = wb.started()
    with pytest.raises(lc.LifecycleError):
        wh.adopt_wind_handoff(other.lin, other.b, ok)  # a candidate of another lineage/parent
    with pytest.raises(lc.LifecycleError):
        wh.adopt_wind_handoff(ns.lin, other.b, ok)
    with pytest.raises(wh.WindHandoffError):
        wh.WindHandoffCandidate("x", 1.0, "boundary", "{}", None, None, {})  # no direct construction
    with pytest.raises(lc.LifecycleError):
        copy.copy(ok)


def test_a_mid_day_hold_and_a_running_occurrence_are_not_wind_eligible():
    ns = wb.started()
    old = ns.b
    st = br.open_day(ns.lin, old, partition=PART, run_digest=wb.RUN, available_bytes_fn=AVAIL)
    assert isinstance(ns.lin.head, lc.DailyHold)
    with pytest.raises(lc.LifecycleError):
        wb.stage(ns, wb.child_same(old.bed), end_s=60.0)  # the head is a hold (the boundary is stale): a mid-day wind is refused
    with pytest.raises(lc.LifecycleError):
        wb.stage(ns, wb.child_same(old.bed), end_s=60.0, parent=ns.lin.head)  # a hold is not a Boundary
    assert st is not None
    ns2 = wb.started()
    occ, _state = lc.begin_occurrence(ns2.lin, ns2.b, lambda vert, topology: ns2.factory(vert, topology), available_bytes_fn=AVAIL)
    with pytest.raises(lc.LifecycleError):
        wb.stage(ns2, wb.child_same(ns2.b.bed), end_s=60.0)  # a running occurrence
    with pytest.raises(lc.LifecycleError):
        wb.stage(ns2, wb.child_same(ns2.b.bed), end_s=60.0, parent=occ)


def _adopt_and_release(ns):
    cand = wb.stage(ns, wb.child_same(ns.b.bed), end_s=60.0)
    parent = ns.b
    refs = (weakref.ref(parent), weakref.ref(parent.root))
    nb = wh.adopt_wind_handoff(ns.lin, parent, cand)
    ns.b = nb
    return refs, cand


def test_the_adopted_boundary_and_candidate_retain_no_parent_bed_scheduler_event_or_owner():
    ns = wb.started()
    refs, cand = _adopt_and_release(ns)
    gc.collect()
    assert refs[0]() is None and refs[1]() is None  # the old boundary and its root are released (the old soil state / bed objects may stay alive through the test's own owner and bed references)
    assert cand._cell["parts"] is None and cand._parent_ref() is None
    assert set(ns.b.__dict__) >= {"root", "bed", "ledger"} and not any(k in ns.b.__dict__ for k in ("event", "owner", "final"))
    assert not hasattr(cand, "final") and not hasattr(cand, "child") and "scheduler" not in cand.receipt_json


def test_the_staged_candidate_does_not_hold_the_finalized_result_or_the_parent_strongly():
    ns = wb.started()
    child = wb.child_same(ns.b.bed)
    final = wb.native_result(child, 60.0)
    cand = wb.stage(ns, child, final=final)
    wr_final = weakref.ref(final)
    del final
    gc.collect()
    assert wr_final() is None  # the candidate retains no finalized scheduler/pool
    assert cand._parent_ref() is ns.b
