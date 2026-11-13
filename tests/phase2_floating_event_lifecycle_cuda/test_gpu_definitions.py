"""Actual-GPU test DEFINITIONS of the CUDA event lifecycle (WRITTEN, NOT RUN by their author: no GPU/CuPy/NVRTC/nvidia-smi was touched). Opt-in (SYRUP_RELEASE_GPU=1); Root runs them on GPU0 with `.venv-cupy`.

Fixtures: the GENUINE CUDA thin closed lake (E5 `device_event`, Dense and published Windowed) opened through the genuine lifecycle opener, run to a real native completion, closed through the GPU selected refresh. DEPENDENCY: the
GPU refresh is UNACCEPTED; these definitions qualify the lifecycle only against an independently accepted compatible refresh. Declared limits: a tiny closed lake; property shifts via the provider-perturbed control of the
refresh suite only; no large-catchment claim; states a genuine completed Pond cannot produce are not fabricated here."""
# ruff: noqa: RUF059
from __future__ import annotations

import gc
import importlib.util
import weakref
from pathlib import Path

import numpy as np
import pytest

_T = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("refresh_gpu_defs_for_lifecycle", _T / "phase2_floating_event_refresh_cuda" / "test_gpu_definitions.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)

from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup.pond_event import PondEvent
from maple_syrup.rainfall import constant_rainfall

BIG = 1 << 40
# The reopened-occurrence forcing of the strengthened controls: a STRICTLY POSITIVE but light 0.01 mm/h applied by 4 real native steps of 0.1 s. Scope = constructor/lifecycle REOPENING + advancement of the CURRENT payload (W/J/I),
# NOT erosion or a storm benchmark. The earlier 30 mm/h literal drove the exact deep fixture into a genuinely unquiet state (Root matched CPU pair: 8 bounded rounds of 200 s, 800 steps each at dt 0.25, never quiet by t = 1604 s;
# an honest bounded failure, no production defect claimed). Root's matched CPU light probe (same production, physics, deep pair and closed saturated fixture) quiets and completes at 5.15 s (3 native post-rain steps). Completion
# policies/defaults are unchanged and the completion loop stays relative and bounded.
LIGHT_RAIN_MM_H = 0.01
MEM = fsl.FloatingMemoryError
cp = rd.cp
host = rd.host


def opened(tmp, windowed, **kw):
    ev, st, vert, owner = rd.e5.device_event(tmp, windowed=windowed, **rd.lake_kwargs())

    def factory(v, topology=None):
        return PondEvent(ev.context, None, ev.sediment, ev.schedule, ev._initial_topology if topology is None else topology, control=ev.control, policy=ev.policy, splash=ev.splash, floating_vertical=v)

    reader = fcu.device_reader(rd.e5._cp())
    lin, occ, state = lc.open_floating_event(owner, st.bed, factory, available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG, **kw)
    return owner, factory, lin, occ, state, reader


def complete(occ, state, max_rounds=8):
    """Real quiet + terminal settlement + recession loop until a genuine COMPLETE handoff, with RELATIVE advancing physical horizons (the real policy hold starts after the first quiet instant, so no absolute time cap is
    asserted): the bounded loop of the CPU `run_to_complete`. Never a fake quiet and never a time-cap completion; a storm that does not complete in the bounded loop fails the test honestly."""
    ev = occ.event
    cur = state
    for _ in range(max_rounds):
        q = ev.evolve(cur, float(cur.t_s) + 200.0, stop_when_quiet=True)
        cur = q.state
        if q.quiet_ready:
            h = ev.settle_retaining_water(cur)
            cur = h.state
            if h.complete:
                return h
    raise AssertionError("the storm did not reach a genuine complete handoff in the bounded loop")


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_open_storm_close_reopen_current_w_j_i_d_h_q_and_ledger(cp, tmp_path, windowed):
    owner, factory, lin, occ, st, reader = opened(tmp_path, windowed)
    assert not owner.is_current and lin.head is occ and occ.vertical._owner is None
    h = complete(occ, st)
    cap = occ.event.completion_capability(h, h.state)
    b = lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)
    assert lin.head is b and b.root.host_active is not None and b.root.edges_host is not None and b.root.host_depth is not None
    assert b.root.depth_m is b.root.column._maps["soil_depth"]  # the CURRENT device canonical depth, never the held D0
    assert abs(b.ledger["residual_m3"]) <= b.ledger["residual_bound_m3"]
    # carried routing, surface depth and face fluxes (no loss, no rebuild)
    assert b.routing[0] is h.state.topology and b.routing[1] is h.state.held_source_m and b.routing[2] is h.state.datum
    assert np.array_equal(host(b.surface_depth_m), host(h.state.hydraulic.depth_m)) or b.role == "pending_recession"
    occ2, st2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)
    assert lin.head is occ2 and occ2.vertical._owner is None
    assert np.array_equal(host(occ2.vertical.current_payload(st2.hydraulic.soil_water_m).state.water_m), host(b.root.water_state.water_m))
    # REAL native steps after the reopen (the failing call on the first device run was the constructor): rain, then complete and close AGAIN; ONE cumulative ledger, current W/J/I advanced by native physics only
    j0, i0 = host(b.root.history.wetting_m).copy(), host(b.root.history.cumulative_intake_m).copy()
    w0 = host(b.root.water_state.water_m).copy()
    t = float(b.t_s)
    st3 = st2
    for _ in range(4):
        st3 = occ2.event.step(st3, LIGHT_RAIN_MM_H / 3.6e6, 0.1).state
    p = occ2.vertical.current_payload(st3.hydraulic.soil_water_m)
    assert (host(p.history.cumulative_intake_m) >= i0).all() and (host(p.history.wetting_m) >= j0).all() and not np.array_equal(host(p.state.water_m), w0)
    h2 = complete(occ2, st3)
    b2 = lc.close_occurrence(lin, occ2, occ2.event.completion_capability(h2, h2.state), h2.state, available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)
    assert lin.head is b2 and b2.ledger["n_refreshes"] == 2 and b2.ledger["n_occurrences"] == 2 and abs(b2.ledger["residual_m3"]) <= b2.ledger["residual_bound_m3"]
    assert b2.routing[0].topology_sha256 == b.routing[0].topology_sha256 and t < float(b2.t_s)  # the net-elevation reference is the LAST genuine build, carried


def test_wrong_device_zero_budgets_need_minus_one_and_reserve_once_before_any_array(cp, tmp_path):
    ev, st, vert, owner = rd.e5.device_event(tmp_path, windowed=False, **rd.lake_kwargs())
    f = lambda v: PondEvent(ev.context, None, ev.sediment, ev.schedule, ev._initial_topology, control=ev.control, policy=ev.policy, splash=ev.splash, floating_vertical=v)
    for kw in ({"memory_budget_bytes": 0}, {"host_memory_budget_bytes": 0}, {"available_bytes_fn": lambda: 0}):
        k = {"available_bytes_fn": fcu.device_reader(cp), "host_available_bytes_fn": lambda: BIG, **kw}
        with pytest.raises(MEM):
            lc.open_floating_event(owner, st.bed, f, **k)
        assert owner.is_current  # refused before any retirement
    # DEFINITION (Root to size): need-1 refuses and the exact need passes with the reserves counted ONCE per pool, using the vertical's published construction model
    # plus lc.flc.storage_model_cuda; the wrong-device gate is exercised with a second device when present.


def test_pending_recession_wake_is_not_eligible_and_a_stale_or_trial_head_is_refused(cp, tmp_path):
    owner, factory, lin, occ, st, reader = opened(tmp_path, False)
    h = complete(occ, st)
    cap = occ.event.completion_capability(h, h.state)
    with pytest.raises(lc.LifecycleError):
        lc.begin_occurrence(lin, occ, factory, available_bytes_fn=reader)  # an occurrence is not a boundary
    b = lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)
    with pytest.raises(lc.LifecycleError):
        lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=reader)  # a duplicate close: the head is no longer this occurrence
    assert b.eligible_for_interevent == (b.role == "boundary")


def test_canonical_h0_export_of_the_new_root_reconstructs_public_identities(cp, tmp_path):
    owner, factory, lin, occ, st, reader = opened(tmp_path, False)
    h = complete(occ, st)
    cap = occ.event.completion_capability(h, h.state)
    b = lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)
    rec = lc.export_boundary_held_context(b, None, available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)
    assert rec.meta()["identities"]["root_column_sha256"] == b.root.column.sha256 and rec.meta()["facts"]["source_witness"] == b.root.source_witness
    assert b.provenance["cold_content_hash"] is None and b.provenance["execution_epoch"]



def _factory(ev):
    def factory(v, topology=None, schedule=None):
        return PondEvent(ev.context, None, ev.sediment, ev.schedule if schedule is None else schedule, ev._initial_topology if topology is None else topology, control=ev.control, policy=ev.policy, splash=ev.splash,
                         floating_vertical=v)

    return factory


def test_current_device_gate_precedes_every_reader_in_open_close_and_begin(cp, tmp_path, monkeypatch):
    """ONE device suffices: the CURRENT device is reported shifted (`fcu.current_device`), so the explicit gate must refuse BEFORE any reader, vertical, hash or array."""
    ev, st, vert, owner = rd.e5.device_event(tmp_path, windowed=False, **rd.lake_kwargs())
    reads = []
    reader = lambda: reads.append(1) or BIG
    real = fcu.current_device
    monkeypatch.setattr(fcu, "current_device", lambda c: real(c) + 1)
    with pytest.raises((lc.LifecycleError, fsl.FloatingSoilError), match="device"):
        lc.open_floating_event(owner, st.bed, _factory(ev), available_bytes_fn=reader, host_available_bytes_fn=reader)
    assert reads == [] and owner.is_current
    monkeypatch.setattr(fcu, "current_device", real)  # a real boundary, then the shifted gate again for close/begin
    owner2, factory, lin, occ, st2, rdr = opened(tmp_path / "b", False)
    h = complete(occ, st2)
    cap = occ.event.completion_capability(h, h.state)
    monkeypatch.setattr(fcu, "current_device", lambda c: real(c) + 1)
    with pytest.raises((lc.LifecycleError, fsl.FloatingSoilError), match="device"):
        lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=reader, host_available_bytes_fn=reader)
    assert reads == [] and lin.head is occ and h not in occ.event._completions_consumed
    monkeypatch.setattr(fcu, "current_device", real)
    b = lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=rdr, host_available_bytes_fn=lambda: BIG)
    monkeypatch.setattr(fcu, "current_device", lambda c: real(c) + 1)
    with pytest.raises((lc.LifecycleError, fsl.FloatingSoilError), match="device"):
        lc.begin_occurrence(lin, b, factory, available_bytes_fn=reader, host_available_bytes_fn=reader)
    assert reads == [] and lin.head is b


def _boundary(tmp, cp):
    owner, factory, lin, occ, st, rdr = opened(tmp, False)
    h = complete(occ, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=rdr, host_available_bytes_fn=lambda: BIG)
    return factory, lin, b, occ, rdr


def test_boundary_vertical_construction_need_minus_one_exact_reserves_once_both_pools_component_scope(cp, tmp_path):
    """COMPONENT scope only (the pure `FloatingPondVertical.from_boundary` constructor; no lifecycle head is touched, so no rewind exists or is needed): the BOUNDARY constructor's own published model (`construction_bytes` device; `cold_host_bytes` + the 64 KiB allowance host), read from a first pure `from_boundary` call - not the owner vertical's, which includes a retained initial-depth copy this constructor references, reserves counted once per pool. The WHOLE begin additionally admits the caller's factory event, the surface binding (`bind_shape`) and the surface router, each against the same
    reader BEFORE its arrays: those are separate admissions (not summed) and the opaque caller factory setup is EXCLUDED from any exact claim here."""
    from maple_syrup import floating_pond as fp

    factory, lin, b, occ, rdr = _boundary(tmp_path, cp)
    big = lambda: BIG
    # The model of THIS constructor, read from the constructor itself: a first pure, unlimited `from_boundary` on the genuine root (no head is touched; the root is a caller-owned reference input and excluded). The ORIGINAL
    # owner vertical `occ.vertical` is NOT this operation: the owner constructor retains a sealed initial depth copy (`initial_depth_copy=True`, +8 N device) that the boundary constructor REFERENCES (`init_cuda_boundary`).
    probe = fp.FloatingPondVertical.from_boundary(b.root, available_bytes_fn=big, host_available_bytes_fn=big)
    dev_need, host_need = probe.construction_bytes, probe.cold_host_bytes + fcu.PY_ALLOWANCE_BYTES
    assert dev_need == occ.vertical.construction_bytes - 8 * probe.n_cells  # independent arithmetic cross-check of the documented owner-vs-boundary difference (the only term that differs)
    assert probe.cold_host_bytes == probe.host_op_bytes
    del probe
    for side, ok, bad in (("device", {"available_bytes_fn": lambda: dev_need + 7, "host_available_bytes_fn": big}, {"available_bytes_fn": lambda: dev_need + 7 - 1, "host_available_bytes_fn": big}),
                          ("host", {"available_bytes_fn": big, "host_available_bytes_fn": lambda: host_need + 11}, {"available_bytes_fn": big, "host_available_bytes_fn": lambda: host_need + 11 - 1})):
        with pytest.raises(MEM, match=side):
            fp.FloatingPondVertical.from_boundary(b.root, reserve_bytes=7, host_reserve_bytes=11, **bad)
        v = fp.FloatingPondVertical.from_boundary(b.root, reserve_bytes=7, host_reserve_bytes=11, **ok)  # exactly the need passes
        assert v.construction_bytes == dev_need
    for kw in ({"memory_budget_bytes": dev_need + 7 - 1}, {"host_memory_budget_bytes": host_need + 11 - 1}):
        with pytest.raises(MEM):
            fp.FloatingPondVertical.from_boundary(b.root, available_bytes_fn=big, host_available_bytes_fn=big, reserve_bytes=7, host_reserve_bytes=11, **kw)


def test_whole_begin_refusals_leave_the_accepted_boundary_the_head_and_a_retry_succeeds(cp, tmp_path):
    """WHOLE-operation scope through the public API only (no private head access): a zero reader / zero budget refuses at the FIRST admission (the vertical, before its arrays and before the caller's factory runs), a factory
    that fails AFTER the vertical staged its private zero rows leaves the Boundary the head, and the retry with the real readers adopts it. An exact whole-begin need is NOT asserted (opaque factory + router admissions)."""
    factory, lin, b, occ, rdr = _boundary(tmp_path, cp)
    calls = []

    def counted(v, topology=None):
        calls.append(1)
        return factory(v, topology)

    for kw in ({"available_bytes_fn": lambda: 0, "host_available_bytes_fn": lambda: BIG}, {"available_bytes_fn": rdr, "host_available_bytes_fn": lambda: 0}, {"available_bytes_fn": rdr, "memory_budget_bytes": 0, "host_available_bytes_fn": lambda: BIG}):
        with pytest.raises(MEM):
            lc.begin_occurrence(lin, b, counted, **kw)
        assert lin.head is b and calls == []

    def late(v, topology=None):
        calls.append("late")
        raise RuntimeError("injected failure after the private vertical was built")

    with pytest.raises(RuntimeError, match="injected"):
        lc.begin_occurrence(lin, b, late, available_bytes_fn=rdr, host_available_bytes_fn=lambda: BIG)
    assert lin.head is b and calls == ["late"]
    occ2, st2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=rdr, host_available_bytes_fn=lambda: BIG)
    assert lin.head is occ2


def test_close_scalar_and_hprime_working_set_need_minus_one_exact_with_the_refresh_admissions_excluded(cp, tmp_path, monkeypatch):
    """The refresh candidate's OWN two admissions dominate a close; to isolate THIS step's declared working set (H' row + storage packets + scalar reads, `storage_model_cuda`) the real refresh runs with large readers (labelled
    control) while the lifecycle's own admission is given need-1 / exact."""
    from maple_syrup import floating_event_lifecycle_cuda as flc

    owner, factory, lin, occ, st, rdr = opened(tmp_path, False)
    h = complete(occ, st)
    cap = occ.event.completion_capability(h, h.state)
    real_stage = flc.erc.stage_completed_event_refresh_cuda

    def big(*a, **k):
        k.update(available_bytes_fn=lambda: BIG, host_available_bytes_fn=lambda: BIG, reserve_bytes=0, host_reserve_bytes=0, memory_budget_bytes=None, host_memory_budget_bytes=None)
        return real_stage(*a, **k)

    monkeypatch.setattr(flc.erc, "stage_completed_event_refresh_cuda", big)
    L, N = occ.vertical.n_layers, occ.vertical.n_cells
    d, hh = flc.storage_model_cuda(L, N)  # the wake case adds 8 N device (declared); a quiet close needs the base model
    with pytest.raises(MEM):
        lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=lambda: d + 7 - 1, reserve_bytes=7, host_available_bytes_fn=lambda: BIG)
    assert lin.head is occ
    with pytest.raises(MEM):
        lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=lambda: BIG, host_available_bytes_fn=lambda: hh + fcu.PY_ALLOWANCE_BYTES + 11 - 1, host_reserve_bytes=11)
    assert lin.head is occ and h not in occ.event._completions_consumed
    b = lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=lambda: d + 7 + 8 * N, reserve_bytes=7, host_available_bytes_fn=lambda: hh + fcu.PY_ALLOWANCE_BYTES + 11, host_reserve_bytes=11)  # exact (+ the wake row)
    assert lin.head is b


def test_late_factory_failure_and_stand_in_events_leave_the_owner_current_and_native_fakes_are_refused(cp, tmp_path):
    ev, st, vert, owner = rd.e5.device_event(tmp_path, windowed=False, **rd.lake_kwargs())
    reader = fcu.device_reader(cp)
    calls = []

    def bad(v):
        calls.append(1)
        raise RuntimeError("injected late factory failure")

    with pytest.raises(RuntimeError, match="injected"):
        lc.open_floating_event(owner, st.bed, bad, available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)
    assert owner.is_current and calls == [1]
    with pytest.raises(lc.LifecycleError, match="vertical"):
        lc.open_floating_event(owner, st.bed, lambda v: ev, available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)  # an event on ANOTHER vertical
    with pytest.raises(lc.LifecycleError):
        lc.open_floating_event(owner, st.bed, lambda v: type("Duck", (), {"vertical": v})(), available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)  # a duck-typed stand-in is refused
    assert owner.is_current
    lin, occ, st2 = lc.open_floating_event(owner, st.bed, _factory(ev), available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)  # retry
    assert not owner.is_current and occ.vertical._owner is None


def test_old_bed_and_water_objects_are_released_after_the_real_next_step(cp, tmp_path):
    """CuPy array weakref support is UNVERIFIED, so the weak references are to the PYTHON objects that hold the old arrays (the old boundary root's water state and the old bed object). After a real next native step on the new
    bed and dropping the caller's own parents they must die: no retained obsolete state."""
    owner, factory, lin, occ, st, rdr = opened(tmp_path, False)
    h = complete(occ, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=rdr, host_available_bytes_fn=lambda: BIG)
    occ2, st2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=rdr, host_available_bytes_fn=lambda: BIG)
    refs = (weakref.ref(b), weakref.ref(b.root.water_state), weakref.ref(b.root))
    st3 = occ2.event.step(st2, LIGHT_RAIN_MM_H / 3.6e6, 0.1).state
    del b, h, st, st2, occ
    gc.collect()
    assert all(r() is None for r in refs), [r() for r in refs]
    assert occ2.event.audit(st3)["water"]["closes"]


def open_bottom_event(tmp, cp):
    """The Root EB3 open-bottom ledger probe (v4) on the device: explicit SUPPORTED `air_entry_kpa_override` 1 kPa (the real fine-soil provider outcome FC6 == SAT is VALID and drains nothing by default), a 3 x 4 plane,
    three 0.01 m layers, saturation 0.999999, depth 0, no rain, the genuine open bottom."""
    from dataclasses import replace

    import fpond_builders as fb
    from maple.core.backend import to_device, to_device_tree

    from maple_syrup import floating_pond as fp
    from maple_syrup import native_floating_soil as nfs
    from maple_syrup.pond_event import PondEvent as PE
    from maple_syrup.sediment_physics import plot1_sediment_parameters

    wh = fb._windowed_helpers()
    z = fb.ph.plane(base=0.12, drop=0.004)
    dense, ctx = wh.dense_deep_bed(z, depth=0.0, mobile=None)
    topo = fb.ph.make_topology(dense, fb.ph.east_outlets(*z.shape), drop=0.02)
    bed = to_device_tree(dense, cp)
    dctx = replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, cp))
    cfg = fb.config(backend="cuda", bottom="open", layers=(0.01, 0.01, 0.01), overrides={"air_entry_kpa_override": np.ones((3, z.size))})
    owner = nfs.initialize_native_floating_soil(fb.native_bed(bed, dctx), cfg, initial_relative_saturation=0.999999)
    from types import SimpleNamespace

    ctl = wh.control_for(SimpleNamespace(hydraulics="cuda", xp=cp), windowed=False, max_dt_s=1.0)

    def factory(v, topology=None):
        return PE(dctx, None, plot1_sediment_parameters(xp=cp), constant_rainfall(0.0, 2.0, 0.0), topo if topology is None else topology, control=ctl, floating_vertical=v)

    return owner, bed, factory, fp


def test_open_bottom_positive_native_drainage_enters_the_lifecycle_ledger_once(cp, tmp_path):
    owner, bed, factory, _fp = open_bottom_event(tmp_path, cp)
    reader = fcu.device_reader(cp)
    n_open = int(host(owner.column._maps["bottom_open"]).sum())
    assert n_open > 0
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)
    h = complete(occ, st)
    audit = occ.event.audit(h.state)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)
    assert audit["water"]["drainage_m3"] > 0.0 and b.ledger["drainage_m3"] == pytest.approx(audit["water"]["drainage_m3"], rel=0, abs=b.ledger["bound_m3"])  # the native bottom loss counted ONCE
    assert abs(b.ledger["residual_m3"]) <= b.ledger["residual_bound_m3"]


def test_pending_return_is_exercised_hprime_once_ledger_closes_and_the_gate_holds(cp, tmp_path, monkeypatch):
    """Controlled PROVIDER perturbation installed AFTER the genuine open and genuine completion and BEFORE the close (installing it before the open only halves the held FC/SAT too, so the refresh recomputes the SAME values and
    never forces a capacity change). The held values are then the originals, the CURRENT W is positive, the dirty set comes from the real terminal mobile deposit and the new provider FC/SAT are smaller: the real changed-column
    cascade must return water to the surface. Asserted, never a conditional skip. Nothing is forged (no changed set, receipt, generation, capability or ledger). Positive native rain is refused before any head change; a RAIN-0
    continuation is allowed and ONE real native zero-rain step keeps the audit closed. Limit: the full quiet completion of the added-capacity recession is NOT required here (unrun)."""
    owner, factory, lin, occ, st, rdr = opened(tmp_path, False)
    h = complete(occ, st)
    cap = occ.event.completion_capability(h, h.state)
    held_fc = host(occ.vertical.held_template().column._maps["theta_field_capacity"]).copy()
    rd.perturb(monkeypatch, fc_scale=0.5, sat_scale=0.5)
    h_before = host(h.state.hydraulic.depth_m).copy()
    b = lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=rdr, host_available_bytes_fn=lambda: BIG)
    assert (host(b.root.column._maps["theta_field_capacity"]) < held_fc).any()
    assert b.ledger["surface_return_m3"] > 0.0 and b.role == "pending_recession" and not b.eligible_for_interevent
    area = float(h.state.topology.dx_m) ** 2
    returned = float(host(b.surface_depth_m).sum() - h_before.sum()) * area  # H' = H + the returned water, exactly once (device readback in this definition only)
    assert abs(returned - b.ledger["surface_return_m3"]) <= b.ledger["residual_bound_m3"]
    assert abs(b.ledger["residual_m3"]) <= b.ledger["residual_bound_m3"]  # soil + surface cumulative closure
    sched = constant_rainfall(float(b.t_s), float(b.t_s) + 0.4, LIGHT_RAIN_MM_H)
    with pytest.raises(lc.LifecycleError, match="ZERO native rainfall"):
        lc.begin_occurrence(lin, b, lambda v, topology=None: _factory(occ.event)(v, topology, sched), available_bytes_fn=rdr, host_available_bytes_fn=lambda: BIG)
    assert lin.head is b  # refused BEFORE any head change
    occ2, st2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=rdr, host_available_bytes_fn=lambda: BIG)  # rain-0 after the boundary clock
    assert lin.head is occ2
    st3 = occ2.event.step(st2, 0.0, 0.05).state
    assert occ2.event.audit(st3)["water"]["closes"]


# Declared fixture limits (NOT defined here, no claim): a true above-net-threshold routing build, a later carry below the threshold and a SECOND crossing (needs a bounded erosion fixture); CuPy-array weakref support;
# Windowed variants of the quota / open-bottom controls; any large-catchment readiness.
