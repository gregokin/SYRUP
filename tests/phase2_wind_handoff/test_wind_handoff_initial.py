"""WH1: a REAL initial Boundary before the first wind/day (fresh current owner + complete actual BedState), failure purity of the start, and the water-free context record. Authored, NOT executed by its author."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import wind_builders as wb
from lcp_builders import PART, run_day

from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_wind_handoff as wh
from maple_syrup import native_floating_soil as nfs

AVAIL = wb.AVAIL


def _start(ns, **kw):
    args = {"bed_context": ns.ctx, "limits": wb.LIMITS, "available_bytes_fn": AVAIL}
    args.update(kw)
    return wh.start_wind_boundary(ns.owner, ns.bed, ns.factory, **args)


def test_start_publishes_a_real_initial_boundary_and_keeps_the_ponded_water():
    ns = wb.build(depth=0.003)
    owner, depth0 = ns.owner, np.array(ns.bed.water.depth_m, copy=True)
    assert owner.is_current and float(depth0.max()) > 0.0
    lin, b, ctx = _start(ns)
    assert not owner.is_current and lin.head is b and type(b) is lc.Boundary and b.role == "boundary" and b.eligible_for_interevent
    assert b.t_s == 0.0 and b.seq == 0 and b.cursor is None and b.pending_forcing is None and b.storm_match is None
    assert np.array_equal(b.surface_depth_m, depth0) and np.array_equal(b.bed.water.depth_m, depth0)  # the initial stored ponded water is carried untouched (not erased to obtain eligibility)
    assert b.bed is ns.bed and b.root.bed is ns.bed
    assert b.root.water_state is owner.state and b.root.history is owner.matrix_history and b.root.column is owner.column  # no event ran: the owner's initial inventory IS the current water
    assert b.root.source_witness == owner.binding.source_witness == ctx.source_witness == b.provenance["cold_anchor_witness"]
    assert b.provenance["property_basis"] == "initial_full" and b.provenance["execution_epoch"] is None and b.provenance["cold_content_hash"] is None
    led = b.ledger
    assert led["rain_m3"] == 0.0 and led["n_occurrences"] == 0 and led["n_refreshes"] == 0 and led["storage0_m3"] > 0.0  # no fictional zero-rain event and no consumed calendar day
    assert len(b.routing) == 3 and b.routing[0] is not None
    lc.saveable_head(lin)  # a genuine saveable boundary


def test_a_retired_owner_can_never_be_reused():
    ns = wb.build()
    owner = ns.owner
    _start(ns)
    nb = wb.fb.native_bed(ns.bed, ns.ctx)
    with pytest.raises(nfs.NativeStaleError):
        nfs.stage_native_refresh(owner, nb, nb, available_bytes_fn=AVAIL)
    with pytest.raises(wh.WindHandoffError):
        _start(ns)


def test_failed_starts_leave_the_owner_current_and_retryable():
    ns = wb.build()
    with pytest.raises(lc.LifecycleError):
        wh.start_wind_boundary(ns.owner, ns.bed, lambda vert: object(), bed_context=ns.ctx, limits=wb.LIMITS, available_bytes_fn=AVAIL)  # the factory must return the actual native PondEvent
    assert ns.owner.is_current
    with pytest.raises(wh.WindHandoffError):
        wh.start_wind_boundary(ns.owner, wb.fb.native_bed(ns.bed, ns.ctx), ns.factory, bed_context=ns.ctx, limits=wb.LIMITS, available_bytes_fn=AVAIL)  # a NativeBed is not a complete BedState
    assert ns.owner.is_current
    with pytest.raises(wh.WindHandoffLimitError):
        _start(ns, limits=replace(wb.LIMITS, max_cells=1))
    assert ns.owner.is_current
    with pytest.raises(wh.WindHandoffLimitError):
        _start(ns, batch_cells=10**6)
    with pytest.raises(wh.WindHandoffError):
        _start(ns, limits=None)
    with pytest.raises(wh.WindHandoffError, match="bed_context"):
        _start(ns, bed_context=object())
    assert ns.owner.is_current
    lin, b, _ctx = _start(ns)  # the same owner still starts after every refusal
    assert lin.head is b and not ns.owner.is_current


def test_water_borne_sediment_in_the_initial_bed_is_refused_before_retirement():
    mob = np.zeros((3, 4, 6))
    mob[1, 2, 2] = 0.1
    ns = wb.build(mobile=mob)
    with pytest.raises(wh.WindHandoffError, match="water-borne"):
        _start(ns)
    assert ns.owner.is_current


def test_bed_context_is_required_and_the_prelude_is_admitted_before_any_native_object_exists():
    ns = wb.build()
    with pytest.raises(TypeError):
        wh.start_wind_boundary(ns.owner, ns.bed, ns.factory, limits=wb.LIMITS, available_bytes_fn=AVAIL)  # the explicit genuine bed_context keyword is required
    calls = []

    def factory(vert):
        calls.append(1)
        return ns.factory(vert)

    reads = []
    for kw, exc in ((dict(limits=replace(wb.LIMITS, max_scan_bytes=1)), wh.WindHandoffLimitError), (dict(available_bytes_fn=lambda: reads.append(1) or 0), nfs.NativeFloatingMemoryError)):
        args = {"bed_context": ns.ctx, "limits": wb.LIMITS, "available_bytes_fn": AVAIL}
        args.update(kw)
        with pytest.raises(exc):
            wh.start_wind_boundary(ns.owner, ns.bed, factory, **args)
        assert calls == [] and ns.owner.is_current  # the factory (hence FloatingPondVertical and initial_state) never ran
    assert reads  # the REAL reader decided


def test_a_before_publish_failure_leaves_the_owner_current_and_retryable():
    ns = wb.build()
    seen = []

    def boom(*_proposal):
        seen.append((lc._MINT.ok, wh._MINT.ok))  # outside every mint window
        raise RuntimeError("injected before_publish failure")

    with pytest.raises(RuntimeError, match="injected"):
        _start(ns, before_publish=boom)
    assert seen == [(False, False)] and ns.owner.is_current
    lin, b, _ctx = _start(ns, before_publish=lambda *_proposal: None)
    assert lin.head is b and not ns.owner.is_current


def test_the_start_runs_the_actual_native_bed_validity_after_admission(monkeypatch):
    ns = wb.build()
    from maple_syrup import pond_checkpoint as pc

    order = []
    real = pc._native_checks
    monkeypatch.setattr(pc, "_native_checks", lambda *a, **k: (order.append("checks"), real(*a, **k))[1])
    _start(ns, available_bytes_fn=lambda: (order.append("reader"), wb.AVAIL())[1])
    assert order[0] == "reader" and "checks" in order


def test_numba_owners_are_a_typed_refusal_before_work():
    pytest.importorskip("numba")
    ns = wb.build(backend="numba")
    with pytest.raises(wh.WindHandoffError, match="Dense NumPy"):
        _start(ns)
    assert ns.owner.is_current


def test_the_initial_boundary_resumes_into_a_day_and_a_storm_without_reinitialisation():
    ns = wb.started()
    b, lin = ns.b, ns.lin
    W0 = np.array(b.root.water_state.water_m, copy=True)
    st = br.open_day(lin, b, partition=PART, run_digest=wb.RUN, available_bytes_fn=AVAIL)
    assert np.array_equal(st.water.water_m, W0) and st.water.t_s == 0.0
    closed = run_day(st, 0).state
    assert closed.water.t_s == 86400.0  # the day advances the physical clock once; the wind start carried no day
    b2 = br.hand_back(lin, closed)
    assert b2.cursor.ordinal == 1 and lin.head is b2
    ns2 = wb.started()
    occ, state = lc.begin_occurrence(ns2.lin, ns2.b, lambda vert, topology: ns2.factory(vert, topology), available_bytes_fn=AVAIL)
    assert float(state.t_s) == 0.0 and np.array_equal(state.hydraulic.depth_m, ns2.b.surface_depth_m) and ns2.lin.head is occ


def test_context_record_roundtrips_and_rebuilds_from_a_verified_boundary():
    ns = wb.started()
    rec = ns.wctx.record()
    assert rec["schema"] == wh.CONTEXT_SCHEMA and rec["shape"] == [3, 4] and rec["n_classes"] == 6 and rec["source_witness"] == ns.b.root.source_witness
    again = wh.restore_wind_context(ns.b, ns.ctx, copy.deepcopy(rec))
    assert again.record() == rec and again is not ns.wctx
    for key, bad in (("descriptor_sha256", "0" * 64), ("source_witness", "1" * 64), ("config_sha256", "2" * 64), ("shape", [4, 3]), ("n_classes", 5)):
        tampered = copy.deepcopy(rec)
        tampered[key] = bad
        with pytest.raises(wh.WindHandoffError):
            wh.restore_wind_context(ns.b, ns.ctx, tampered)
    with pytest.raises(wh.WindHandoffError):
        wh.restore_wind_context(ns.b, ns.ctx, {**rec, "extra": 1})
    other = wb.build(surface=wb.ph.plane(ny=3, nx=5, base=0.12, drop=0.004))  # another case (grid): its static context cannot rebuild this boundary's record
    with pytest.raises(wh.WindHandoffError):
        wh.restore_wind_context(ns.b, other.ctx, rec)
    with pytest.raises(wh.WindHandoffError):
        wh.WindContext(None, (3, 4), 6, "d", "w", "c")  # no direct construction
