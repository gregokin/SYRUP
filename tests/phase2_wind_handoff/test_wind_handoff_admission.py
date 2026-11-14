"""WH1: admission order of the staged wind handoff -- limits and the REAL available-bytes reader precede every mobile-pool predicate, array value comparison, native validity and cold scan, on every path (including zero dirty columns). Authored, NOT executed by
its author. The `pytest.fail` monkeypatches are mechanical probes of call ORDER, not physics."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import wind_builders as wb

from maple_syrup import floating_wind_handoff as wh
from maple_syrup import native_floating_soil as nfs


@pytest.fixture
def ns():
    return wb.started(depth=0.003)


def _forbid_value_work(monkeypatch):
    """Every stage that READS array values (after admission): the native predicate, water comparison, native validity, descriptor, cold scan and the native row readers."""
    import maple.aeolian.cohorts.dispatch as dispatch

    def fail(name):
        return lambda *a, **k: pytest.fail(f"{name} ran before the admission completed")

    monkeypatch.setattr(dispatch, "mobile_state_is_genuinely_empty", fail("the native mobile-pool predicate"))
    for name in ("_wind_values", "_same_leaves", "_native_valid", "_descriptor", "_dirty_columns"):
        monkeypatch.setattr(wh, name, fail(name))
    monkeypatch.setattr(nfs, "_rows", fail("a native row read"))


def test_a_one_byte_candidate_cap_refuses_even_when_no_column_is_dirty(ns, monkeypatch):
    c = wb.child_same(ns.b.bed)  # zero dirty columns: the continued soil-state copy, the checks and the receipt still exist
    _forbid_value_work(monkeypatch)
    with pytest.raises(wh.WindHandoffLimitError, match="max_candidate_bytes"):
        wb.stage(ns, c, limits=replace(wb.LIMITS, max_candidate_bytes=1))
    assert ns.lin.head is ns.b


def test_zero_available_bytes_refuse_before_the_mobile_pool_predicate_or_any_value_work(ns, monkeypatch):
    c = wb.child_same(ns.b.bed)
    _forbid_value_work(monkeypatch)
    reads = []
    with pytest.raises(nfs.NativeFloatingMemoryError):
        wb.stage(ns, c, available_bytes_fn=lambda: reads.append(1) or 0)
    assert reads == [1] and ns.lin.head is ns.b  # the real reader was consulted exactly once, and decided


def test_the_cohort_limit_is_judged_from_pool_metadata_before_the_native_predicate(ns, monkeypatch):
    from maple.core.types.cohort import zeros_mobile_cohort_state

    pool = replace(zeros_mobile_cohort_state(2), mass_kg=np.zeros(2), active_mask=np.zeros(2, dtype=bool))  # genuinely empty, 2 slots
    c = wb.child_same(ns.b.bed)
    final = wb.native_result(c, 10.0, mobile_state=pool)
    _forbid_value_work(monkeypatch)
    reads = []
    with pytest.raises(wh.WindHandoffLimitError, match="max_cohorts"):
        wb.stage(ns, c, final=final, limits=replace(wb.LIMITS, max_cohorts=1), available_bytes_fn=lambda: reads.append(1) or 10**13)
    assert reads == []  # not even the reader: a limit refusal precedes it
    monkeypatch.undo()
    cand = wb.stage(ns, c, final=final, limits=replace(wb.LIMITS, max_cohorts=2))
    plan = cand.receipt()["plan"]
    assert plan["pool_slots"] == 2 and plan["pool_bytes"] >= 2 * 8 + 2  # the ACTUAL pool footprint is derived from metadata and enters the admission model


def test_the_pool_footprint_is_part_of_the_scan_admission_model(ns):
    from maple.core.types.cohort import zeros_mobile_cohort_state

    c = wb.child_same(ns.b.bed)
    small = wb.stage(ns, c, final=wb.native_result(c, 10.0)).receipt()["plan"]
    big_pool = zeros_mobile_cohort_state(4096)
    big = wb.stage(ns, c, final=wb.native_result(c, 10.0, mobile_state=big_pool)).receipt()["plan"]
    assert big["scan_required_bytes"] - small["scan_required_bytes"] == big["pool_bytes"] - small["pool_bytes"] > 0
    with pytest.raises(wh.WindHandoffLimitError, match="max_scan_bytes"):
        wb.stage(ns, c, final=wb.native_result(c, 10.0, mobile_state=big_pool), limits=replace(wb.LIMITS, max_scan_bytes=big["scan_required_bytes"] - 1))


def test_class_vectors_are_never_coerced_before_admission(ns, monkeypatch):
    c = wb.child_same(ns.b.bed)
    reads = []
    for bad in ([0.0] * 6, np.zeros(6, dtype=np.float32), np.zeros((6, 1))):
        final = wb.native_result(c, 10.0, class_kw={"deposited_mass_by_class_kg": bad})
        with pytest.raises(wh.WindHandoffError, match="class vector"):
            wb.stage(ns, c, final=final, available_bytes_fn=lambda: reads.append(1) or 10**13)
    assert reads == [] and ns.lin.head is ns.b  # refused from metadata, with no coercion and no reader call


def test_an_unqualified_bed_is_refused_from_metadata_before_the_reader(ns):
    class Foreign(np.ndarray):  # any NumPy subclass / device-like array is not the exact host ndarray the qualified path accepts
        pass

    c = wb.child_same(ns.b.bed)
    bad = replace(c, water=replace(c.water, depth_m=np.array(c.water.depth_m).view(Foreign)))
    reads = []
    with pytest.raises(wh.WindHandoffError, match="host float64"):
        wb.stage(ns, bad, final=wb.native_result(bad, 10.0), available_bytes_fn=lambda: reads.append(1) or 10**13)
    assert reads == []


def test_the_admission_precedes_the_native_validity_and_the_scan_in_the_normal_order(ns, monkeypatch):
    order = []
    real_valid, real_dirty = wh._native_valid, wh._dirty_columns
    monkeypatch.setattr(wh, "_native_valid", lambda *a, **k: (order.append("valid"), real_valid(*a, **k))[1])
    monkeypatch.setattr(wh, "_dirty_columns", lambda *a, **k: (order.append("scan"), real_dirty(*a, **k))[1])
    c = wb.child_same(ns.b.bed)
    wb.stage(ns, c, available_bytes_fn=lambda: (order.append("reader"), wb.AVAIL())[1])
    assert order[:3] == ["reader", "valid", "scan"]
