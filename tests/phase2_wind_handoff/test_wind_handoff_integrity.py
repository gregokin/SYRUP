"""WH1: post-stage integrity, bounded exactly-once bookkeeping, adoption interruption and the weak accepted-boundary reference. Authored, NOT executed by its author.

The mechanical fault probes here are bookkeeping controls of the authored fixtures (in-place tampering, injected allocation failures); they are not physics acceptance."""
from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
import wind_builders as wb

from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_wind_handoff as wh


@pytest.fixture
def ns():
    return wb.started(depth=0.003)


def _staged(ns, oid="occ-1", end_s=60.0):
    child = wb.child_same(ns.b.bed)
    return child, wb.stage(ns, child, end_s=end_s, occurrence_id=oid)


def _assert_retryable(ns, cand, head):
    assert ns.lin.head is head and not cand.consumed and cand._cell["parts"] is not None


def test_a_child_changed_in_place_after_staging_is_not_adopted_and_stays_retryable(ns):
    child, cand = _staged(ns)
    m = child.voxel_column.mass_kg
    old = float(m[1, 1, 0, 0])
    m[1, 1, 0, 0] = old + 1.0
    with pytest.raises(wh.WindHandoffError, match="integrity"):
        wh.adopt_wind_handoff(ns.lin, ns.b, cand)
    _assert_retryable(ns, cand, ns.b)
    assert cand.accepted_boundary_ref is None
    m[1, 1, 0, 0] = old  # the exact bytes are restored: the SAME candidate adopts
    nb = wh.adopt_wind_handoff(ns.lin, ns.b, cand)
    assert ns.lin.head is nb and cand.consumed


def test_a_parent_changed_in_place_after_staging_is_not_adopted(ns):
    _child, cand = _staged(ns)
    p = ns.b.bed.voxel_column.mass_kg
    old = float(p[0, 0, 0, 0])
    p[0, 0, 0, 0] = old + 1.0
    with pytest.raises(wh.WindHandoffError, match="integrity"):
        wh.adopt_wind_handoff(ns.lin, ns.b, cand)
    _assert_retryable(ns, cand, ns.b)
    p[0, 0, 0, 0] = old
    assert ns.lin.head is ns.b and wh.adopt_wind_handoff(ns.lin, ns.b, cand) is ns.lin.head


def test_the_candidates_proposed_state_cannot_be_edited_after_staging(ns):
    _child, cand = _staged(ns)
    parts = cand._cell["parts"]
    parts.ledger["n_refreshes"] += 7  # a caller-reachable mutable field of the proposed state
    with pytest.raises(wh.WindHandoffError, match="integrity"):
        wh.adopt_wind_handoff(ns.lin, ns.b, cand)
    parts.ledger["n_refreshes"] -= 7
    _assert_retryable(ns, cand, ns.b)
    cand._cell["parts"] = None  # clearing the cell is refused as a consumed/emptied candidate, never a silent adoption
    with pytest.raises(wh.WindHandoffError):
        wh.adopt_wind_handoff(ns.lin, ns.b, cand)


def test_the_expected_integrity_is_private_to_the_module_not_in_the_candidate_cell(ns):
    _child, cand = _staged(ns)
    assert set(cand._cell) == {"parts", "consumed", "accepted"} and cand in wh._INTEGRITY and len(wh._INTEGRITY[cand]) == 64
    assert not any(isinstance(v, str) and v == wh._INTEGRITY[cand] for v in cand._cell.values())


def test_before_publish_runs_outside_mint_windows_and_a_failure_preserves_everything(ns):
    _child, cand = _staged(ns)
    seen = []

    def boom():
        seen.append((lc._MINT.ok, wh._MINT.ok))
        raise RuntimeError("injected before_publish failure")

    with pytest.raises(RuntimeError, match="injected"):
        wh.adopt_wind_handoff(ns.lin, ns.b, cand, before_publish=boom)
    assert seen == [(False, False)]
    _assert_retryable(ns, cand, ns.b)
    nb = wh.adopt_wind_handoff(ns.lin, ns.b, cand, before_publish=lambda: None)
    assert ns.lin.head is nb
    with pytest.raises(wh.WindHandoffError, match="callable"):
        wh.adopt_wind_handoff(ns.lin, nb, cand, before_publish=3)


def test_a_before_publish_tamper_is_caught_by_the_final_integrity_audit(ns):
    child, cand = _staged(ns)

    def tamper():
        child.voxel_column.mass_kg[0, 0, 0, 0] += 1.0

    with pytest.raises(wh.WindHandoffError, match="integrity"):
        wh.adopt_wind_handoff(ns.lin, ns.b, cand, before_publish=tamper)
    _assert_retryable(ns, cand, ns.b)


def test_the_adoption_admission_uses_the_real_reader_and_refuses_before_any_change(ns):
    from maple_syrup import native_floating_soil as nfs

    _child, cand = _staged(ns)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        wh.adopt_wind_handoff(ns.lin, ns.b, cand, available_bytes_fn=lambda: 0)
    _assert_retryable(ns, cand, ns.b)
    assert wh.adopt_wind_handoff(ns.lin, ns.b, cand, available_bytes_fn=wb.AVAIL) is ns.lin.head


def test_an_allocation_failure_in_the_bookkeeping_leaves_the_head_and_the_candidate_retryable(ns, monkeypatch):
    _child, cand = _staged(ns)

    class Boom(weakref.WeakKeyDictionary):
        def __setitem__(self, key, value):
            raise MemoryError("injected: the per-lineage record cannot be allocated")

    monkeypatch.setattr(wh, "_LAST", Boom())
    with pytest.raises(MemoryError):
        wh.adopt_wind_handoff(ns.lin, ns.b, cand)
    _assert_retryable(ns, cand, ns.b)  # the candidate is NOT consumed and the head did not move
    monkeypatch.undo()
    calls = []

    def flaky(obj):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError("injected: the weak reference cannot be allocated")
        return weakref.ref(obj)

    monkeypatch.setattr(wh, "_weak", flaky)
    with pytest.raises(MemoryError):
        wh.adopt_wind_handoff(ns.lin, ns.b, cand)
    _assert_retryable(ns, cand, ns.b)
    nb = wh.adopt_wind_handoff(ns.lin, ns.b, cand)  # the SAME candidate adopts on retry
    assert ns.lin.head is nb and cand.consumed and cand.accepted_boundary_ref() is nb


def test_no_per_lineage_set_exists_and_only_the_last_occurrence_id_is_recorded(ns):
    assert not hasattr(wh, "_USED")
    for i in range(8):
        wb.wind(ns, wb.child_same(ns.b.bed), end_s=10.0 * (i + 1), occurrence_id=f"occ-{i:02d}", start_s=10.0 * i)
    rec = wh._LAST[ns.lin]
    assert type(rec) is list and rec == ["occ-07"]  # ONE bounded record per lineage: constant size however many events ran
    with pytest.raises(wh.WindHandoffError, match="duplicate"):
        wb.stage(ns, wb.child_same(ns.b.bed), end_s=100.0, occurrence_id="occ-07", start_s=80.0)
    # honest limit: a historical id is NOT remembered here (it would need unbounded storage); the MASY schedule cursor owns cross-event identity
    cand = wb.stage(ns, wb.child_same(ns.b.bed), end_s=100.0, occurrence_id="occ-00", start_s=80.0)
    assert not cand.consumed


def test_the_accepted_boundary_reference_is_weak_read_only_and_releases_every_state(ns):
    child, cand = _staged(ns)
    assert cand.accepted_boundary_ref is None
    old = ns.b
    refs = (weakref.ref(old), weakref.ref(old.root))
    nb = wh.adopt_wind_handoff(ns.lin, old, cand)
    ref = cand.accepted_boundary_ref
    assert type(ref) is weakref.ref and ref() is nb and ns.lin.head is ref()  # the coordinator reconciles with the actual native head
    with pytest.raises(AttributeError):
        cand.accepted_boundary_ref = None
    assert cand._cell["parts"] is None and cand.consumed and cand._parent_ref() is old
    ns.b = nb
    del old
    gc.collect()
    assert refs[0]() is None and refs[1]() is None and cand._parent_ref() is None  # no strong old state is retained by the candidate
    _o, nb2, _c = wb.wind(ns, wb.child_same(nb.bed), end_s=120.0, occurrence_id="occ-2", start_s=60.0)
    ns.b = nb2
    del nb, _o
    gc.collect()
    assert ref() is None  # and no strong NEW state either: the reference dies with the superseded boundary


def test_the_integrity_table_is_weak_and_bounded_by_the_live_candidates(ns):
    gc.collect()
    n0 = len(wh._INTEGRITY)
    held = [wb.stage(ns, wb.child_same(ns.b.bed), end_s=30.0, occurrence_id=f"c{i}") for i in range(4)]
    assert len(wh._INTEGRITY) == n0 + 4
    del held
    gc.collect()
    assert len(wh._INTEGRITY) == n0


def test_an_unchanged_stage_does_not_alter_any_hashed_leaf(ns):
    """Each candidate's integrity is reproducible; distinct native state tokens may differ."""
    child = wb.child_same(ns.b.bed)
    a = wb.stage(ns, child, end_s=60.0)
    b = wb.stage(ns, child, end_s=60.0)
    for candidate in (a, b):
        assert wh._INTEGRITY[candidate] == wh._integrity(ns.b, candidate._cell["parts"],
            candidate.occurrence_id, candidate.end_s, candidate.receipt_json)
    assert np.array_equal(child.voxel_column.mass_kg, ns.b.bed.voxel_column.mass_kg)
