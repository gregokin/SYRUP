"""WH1: malformed completion proof, clocks, water mismatch, aliasing, limits/memory and failure purity of the staged wind handoff. Authored, NOT executed by its author.

Every refusal must leave the accepted head, the parent boundary's arrays and the child untouched (snapshots compared bit for bit)."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import wind_builders as wb

from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_wind_handoff as wh
from maple_syrup import native_floating_soil as nfs

CELL = (1, 1)
DST = (1, 2)


def _child_snap(child):
    return (np.array(child.voxel_column.mass_kg, copy=True), np.array(child.active_layer.mass_kg, copy=True), np.array(child.water.depth_m, copy=True), np.array(child.water.mobile_mass_by_cell_class_kg, copy=True))


def _refused(ns, child, final, *, match=None, exc=wh.WindHandoffError, **over):
    before, csnap = wb.snapshot(ns.b), _child_snap(child) if hasattr(child, "voxel_column") else None
    head = ns.lin.head
    with pytest.raises(exc, match=match):
        wb.stage(ns, child, final=final, **over)
    assert ns.lin.head is head and head is ns.b
    wb.assert_same(before, wb.snapshot(ns.b))
    if csnap is not None:
        for a, b in zip(csnap, _child_snap(child), strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.fixture
def ns():
    return wb.started(depth=0.003)


@pytest.mark.parametrize("stand_in", [True, {"completed": True, "water_handoff_ready": True}, SimpleNamespace(completed=True, water_handoff_ready=True, status="completed", reasons=(), time_s=10.0)], ids=["bool", "json-stub", "duck"])
def test_only_the_actual_native_finalization_result_is_accepted(ns, stand_in):
    _refused(ns, wb.child_same(ns.b.bed), stand_in, match="actual native")


def test_incomplete_or_not_ready_finalizations_are_refused(ns):
    c = wb.child_same(ns.b.bed)
    _refused(ns, c, wb.native_result(c, 10.0, completed=False, status="incomplete", reasons=("airborne mass left",)), match="not completed")
    _refused(ns, c, wb.native_result(c, 10.0, status="surprise"), match="not completed")
    _refused(ns, c, wb.native_result(c, 10.0, reasons=("x",)), match="not completed")


def test_clock_mismatches_are_refused(ns):
    c = wb.child_same(ns.b.bed)
    _refused(ns, c, wb.native_result(c, 10.0, physical_time_s=9.0), match="terminal time")
    _refused(ns, c, wb.native_result(c, 0.0), match="strictly after")  # a nonzero wind end is required
    _refused(ns, c, wb.native_result(c, 10.0), start_s=1.0, match="physical clock")  # the start must be the accepted boundary's clock
    _refused(ns, c, wb.native_result(c, 10.0), start_s=0, match="float")  # not a float clock
    _refused(ns, c, wb.native_result(c, 10.0), start_s=float("nan"), match="finite")
    _refused(ns, c, wb.native_result(c, 10.0, block_offset_s=1.0), match="terminal block")
    _refused(ns, c, wb.native_result(c, 10.0, block_index=0), match="terminal block")


def test_an_end_before_the_accepted_clock_is_refused_after_a_real_wind(ns):
    wb.wind(ns, wb.child_same(ns.b.bed), end_s=60.0)
    c = wb.child_same(ns.b.bed)
    _refused(ns, c, wb.native_result(c, 50.0), occurrence_id="occ-2", start_s=60.0, match="strictly after")


def test_a_nonempty_wind_mobile_pool_is_refused(ns):
    from maple.core.types.cohort import zeros_mobile_cohort_state

    pool = zeros_mobile_cohort_state(2)
    pool = replace(pool, mass_kg=np.array([1.0e-3, 0.0]), active_mask=np.array([True, False]))
    c = wb.child_same(ns.b.bed)
    _refused(ns, c, wb.native_result(c, 10.0, mobile_state=pool), match="not genuinely empty")


def test_the_result_must_share_the_childs_own_fields(ns):
    a, b = wb.child_same(ns.b.bed), wb.child_same(ns.b.bed)
    _refused(ns, b, wb.native_result(a, 10.0), match="shared fields|own")
    with pytest.raises(wh.WindHandoffError, match="child must be"):
        wb.stage(ns, SimpleNamespace(), final=wb.native_result(a, 10.0))
    assert ns.lin.head is ns.b


def test_surface_water_must_be_carried_bit_for_bit(ns):
    base = wb.child_same(ns.b.bed)
    h = np.array(base.water.depth_m, copy=True)
    h[1, 1] += 1e-3
    wet = replace(base, water=replace(base.water, depth_m=h))
    _refused(ns, wet, wb.native_result(wet, 10.0), match="surface-water leaves differ")
    h2 = np.array(base.water.depth_m, copy=True)
    h2[0, 0] = np.nextafter(h2[0, 0], 1.0)  # even one ulp is a physical water update
    ulp = replace(base, water=replace(base.water, depth_m=h2))
    _refused(ns, ulp, wb.native_result(ulp, 10.0), match="surface-water leaves differ")
    mob = np.array(base.water.mobile_mass_by_cell_class_kg, copy=True)
    mob[1, 1, 2] = 1e-3
    loaded = replace(base, water=replace(base.water, mobile_mass_by_cell_class_kg=mob))
    _refused(ns, loaded, wb.native_result(loaded, 10.0), match="surface-water leaves differ")


def test_class_vectors_must_be_closed_and_leave_nothing_airborne(ns):
    c = wb.child_same(ns.b.bed)
    _refused(ns, c, wb.native_result(c, 10.0, class_kw={"deposited_mass_by_class_kg": np.zeros(3)}), match="class vector")
    _refused(ns, c, wb.native_result(c, 10.0, class_kw={"bed_change_by_class_kg": np.full(6, np.nan)}), match="class vector")
    _refused(ns, c, wb.native_result(c, 10.0, class_kw={"airborne_mass_after_by_class_kg": np.full(6, 1e-3)}), match="airborne")
    _refused(ns, c, wb.native_result(c, 10.0, class_kw={"transfer_identity_gap_by_class_kg": np.full(6, 1.0)}), match="not closed")
    _refused(ns, c, wb.native_result(c, 10.0, bounds=(float("nan"), 1e-12)), match="not closed")


def test_an_aliased_child_and_unordered_commit_stamps_are_refused(ns):
    same = ns.b.bed  # the parent itself: its arrays would be compared with themselves
    _refused(ns, same, wb.native_result(same, 10.0), match="shares native arrays")
    partial = replace(wb.child_same(ns.b.bed), voxel_column=ns.b.bed.voxel_column)
    _refused(ns, partial, wb.native_result(partial, 10.0), match="shares native arrays")
    c = wb.child_same(ns.b.bed)
    late = replace(c, committed_topography=replace(c.committed_topography, last_commit_time_s=999.0))
    _refused(ns, late, wb.native_result(late, 10.0), match="commit stamps")


def test_a_context_of_another_source_is_refused(ns):
    other = wb.started(base=0.15)  # another accepted source (different rows, hence another cold witness; an identical rebuild would legitimately share its context)
    c = wb.child_same(ns.b.bed)
    assert other.wctx.source_witness != ns.wctx.source_witness
    _refused(ns, c, wb.native_result(c, 10.0), context=other.wctx, match="not bound")
    _refused(ns, c, wb.native_result(c, 10.0), context=SimpleNamespace(), match="genuine WindContext")
    _refused(ns, c, wb.native_result(c, 10.0), occurrence_id="", match="occurrence_id")
    _refused(ns, c, wb.native_result(c, 10.0), occurrence_sha256="XYZ", match="occurrence_sha256")


def test_limits_and_memory_are_refused_before_any_scan_or_copy(ns, monkeypatch):
    c = wb.child_same(ns.b.bed)
    monkeypatch.setattr(wh, "_dirty_columns", lambda *a, **k: pytest.fail("the cold scan ran before a refusal"))
    L = wb.LIMITS
    for field, value in (("max_cells", 1), ("max_voxels", 1), ("max_classes", 1), ("max_scan_bytes", 1)):
        bad = replace(L, **{field: value})
        _refused(ns, c, wb.native_result(c, 10.0), limits=bad, exc=wh.WindHandoffLimitError)
    _refused(ns, c, wb.native_result(c, 10.0), batch_cells=L.max_batch_cells + 1, exc=wh.WindHandoffLimitError)
    _refused(ns, c, wb.native_result(c, 10.0), limits=None, exc=wh.WindHandoffLimitError)
    _refused(ns, c, wb.native_result(c, 10.0), available_bytes_fn=lambda: 0, exc=nfs.NativeFloatingMemoryError)  # the REAL reader decides
    _refused(ns, c, wb.native_result(c, 10.0), available_bytes_fn=3.5, match="callable")
    with pytest.raises(wh.WindHandoffError):
        wh.WindHandoffLimits(1, 2, 3, 4, 5, 6, True, 8)
    with pytest.raises(wh.WindHandoffError):
        wh.WindHandoffLimits(1, 2, 3, 4, 5, 6, 7, 0)  # max_cohorts must be >= 1


def test_candidate_and_receipt_limits_refuse_before_the_selected_read_or_adoption(ns, monkeypatch):
    c = wb.child_move(ns.b.bed, ns.ctx, CELL, DST, klass=0)
    calls = []
    monkeypatch.setattr(nfs, "_read_texture", lambda *a, **k: calls.append(1))
    _refused(ns, c, wb.native_result(c, 10.0), limits=replace(wb.LIMITS, max_candidate_bytes=1), exc=wh.WindHandoffLimitError)
    assert calls == []  # nothing was copied or read at the actual dirty count
    monkeypatch.undo()
    _refused(ns, c, wb.native_result(c, 10.0), limits=replace(wb.LIMITS, max_receipt_bytes=10), exc=wh.WindHandoffLimitError)
    reads = []
    ok = wb.stage(ns, c, final=wb.native_result(c, 10.0), available_bytes_fn=lambda: reads.append(1) or 10**13)
    assert len(reads) >= 2 and ok.receipt()["properties"]["n_dirty"] == 2  # scan admission + candidate admission at the actual dirty count


def test_the_dirty_set_is_independent_of_the_batch_size_and_catches_composition_exchanges(ns):
    c = wb.child_move(ns.b.bed, ns.ctx, CELL, DST, klass=0)
    props = []
    for B in (1, 5, 64):
        cand = wb.stage(ns, c, final=wb.native_result(c, 10.0), batch_cells=B)
        props.append(cand.receipt()["properties"])
    assert props[0] == props[1] == props[2] and props[0]["n_dirty"] == 2


def test_an_unrecorded_class_mass_change_is_refused_by_the_native_bed_validity_before_the_scan(ns, monkeypatch):
    c = wb.child_class_swap(ns.b.bed, CELL)  # MECHANICAL NEGATIVE fixture: direct native-class construction with no ledger record
    monkeypatch.setattr(wh, "_dirty_columns", lambda *a, **k: pytest.fail("the cold scan ran on a bed the native validators refuse"))
    _refused(ns, c, wb.native_result(c, 10.0), match="not a valid native bed")


def test_a_failure_after_staging_started_leaves_everything_unchanged_and_retryable(ns, monkeypatch):
    c = wb.child_move(ns.b.bed, ns.ctx, CELL, DST, klass=0)
    final = wb.native_result(c, 10.0)

    def boom(*a, **k):
        raise RuntimeError("injected failure inside the native reconcile")

    monkeypatch.setattr(nfs, "_reconcile_changed", boom)
    _refused(ns, c, final, exc=RuntimeError)
    monkeypatch.undo()
    monkeypatch.setattr(lc, "_balance_check", lambda led, stored: (_ for _ in ()).throw(lc.LifecycleError("injected: the cumulative identity does not close")))
    _refused(ns, c, final, exc=lc.LifecycleError, match="injected")
    monkeypatch.undo()
    old, nb, _cand = wb.wind(ns, c, final=final, end_s=10.0)  # the same inputs still stage and adopt
    assert ns.lin.head is nb and nb.t_s == 10.0


def test_a_staged_candidate_is_pure_and_repeatable(ns):
    c = wb.child_move(ns.b.bed, ns.ctx, CELL, DST, klass=0)
    before = wb.snapshot(ns.b)
    a = wb.stage(ns, c, final=wb.native_result(c, 10.0))
    b = wb.stage(ns, c, final=wb.native_result(c, 10.0))
    wb.assert_same(before, wb.snapshot(ns.b))
    assert ns.lin.head is ns.b and a.receipt() == b.receipt() and a is not b
    assert not a.consumed and not b.consumed
