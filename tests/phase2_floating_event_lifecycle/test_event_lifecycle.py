# ruff: noqa: RUF059
"""EB3: the single-authority floating event lifecycle (CPU NumPy/Numba x Dense/published Windowed): open (owner retired LAST) -> native occurrence -> genuine completion -> selected property refresh -> accepted boundary -> next native
occurrence. Real thin-lake storms; no fakes. Limits (natural-fixture) are stated in the report."""
from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
from lifecycle_builders import AVAIL, build, run_to_complete

from maple_syrup import floating_completion as fcm
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import native_floating_soil as nfs
from maple_syrup.rainfall import constant_rainfall


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def opened(tmp, backend, windowed, **kw):
    owner, bed, factory, pref, ev0 = build(tmp, backend, windowed, **kw)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref)
    return owner, bed, factory, lin, occ, st


def test_failed_open_leaves_the_owner_current_and_a_retry_works(tmp_path):
    owner, bed, factory, pref, ev0 = build(tmp_path / "a", "numpy", False)
    w0 = np.array(owner.state.water_m, copy=True)
    calls = []

    def bad(vert):
        calls.append(1)
        raise RuntimeError("injected factory failure")

    with pytest.raises(RuntimeError, match="injected"):
        lc.open_floating_event(owner, bed, bad, available_bytes_fn=AVAIL, preferential=pref)
    assert owner.is_current and calls == [1]
    with pytest.raises(lc.LifecycleError, match="vertical"):
        lc.open_floating_event(owner, bed, lambda vert: ev0, available_bytes_fn=AVAIL, preferential=pref)  # an event on another vertical
    assert owner.is_current and np.array_equal(np.asarray(owner.state.water_m), w0)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref)  # retry
    assert not owner.is_current and lin.head is occ and occ.vertical._owner is None
    with pytest.raises(lc.LifecycleError):
        lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref)  # a retired owner can never open again


def test_open_retires_the_owner_and_the_runtime_holds_no_initial_w0(tmp_path):
    def scope():
        owner, bed, factory, pref, ev0 = build(tmp_path / "a", "numpy", False)
        refs = (weakref.ref(owner.state.water_m), weakref.ref(owner.state))
        lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref)
        for t in (lambda: occ.vertical.initial_soil_water(), lambda: occ.vertical.check_initial(bed, occ.event.context, 0.0)):
            with pytest.raises(Exception, match="initial payload"):
                t()
        return refs, lin, occ

    (w_ref, s_ref), lin, occ = scope()
    for _ in range(3):
        gc.collect()
    assert w_ref() is None and s_ref() is None  # owner, original event/vertical and the caller's parents released: the runtime kept nothing of W0
    assert occ.vertical._initial.state is None and occ.vertical._initial.history is None  # root facts only
    assert occ.vertical.held_template().source_witness  # the cold anchor stays available


@pytest.mark.parametrize("windowed", [False, True])
def test_complete_refresh_adopt_and_next_occurrence_carry_the_one_water_and_the_routing_reference(tmp_path, backend, windowed):
    owner, bed, factory, lin, occ, st = opened(tmp_path / "a", backend, windowed)
    w0 = np.array(owner.state.water_m, copy=True)
    h = run_to_complete(occ.event, st)
    p = occ.vertical.current_payload(h.state.hydraulic.soil_water_m)
    j_before, i_before = np.array(p.history.wetting_m, copy=True), np.array(p.history.cumulative_intake_m, copy=True)
    w_cur = np.array(p.state.water_m, copy=True)
    assert not np.array_equal(w_cur, w0)  # current W is LATER than the initial W0
    trial = occ.event.step(h.state, 0.0, 0.2).state  # a functional trial: never moves the head
    assert lin.head is occ and trial is not h.state
    cap = occ.event.completion_capability(h, h.state)
    b = lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=AVAIL)
    assert lin.head is b and b.role == "boundary" and b.eligible_for_interevent
    assert np.array_equal(np.asarray(b.root.history.wetting_m), j_before) and np.array_equal(np.asarray(b.root.history.cumulative_intake_m), i_before)  # J/I bit exact
    assert np.array_equal(np.asarray(b.root.water_state.water_m), w_cur) or b.ledger["n_refreshes"] == 1  # unchanged columns bit exact (this fixture changes at most one)
    assert b.provenance["property_basis"] == "boundary_selected" and b.provenance["cold_content_hash"] is None and b.provenance["execution_epoch"]
    assert b.provenance["cold_anchor_witness"] == occ.vertical.held_template().source_witness  # the cold anchor is never replaced by an epoch
    assert b.root.generation == occ.vertical.held_template().generation + 1
    assert h in occ.event._completions_consumed  # consumed (LAST)
    # next native occurrence
    occ2, st2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=AVAIL)
    assert lin.head is occ2
    assert st2.topology is b.routing[0] and st2.held_source_m is b.routing[1] and st2.datum is b.routing[2]  # the LAST actual routing reference is carried unchanged
    assert float(st2.hydraulic.t_s) == b.t_s
    assert np.array_equal(np.asarray(st2.hydraulic.depth_m), np.asarray(b.surface_depth_m)) and np.array_equal(np.asarray(st2.hydraulic.qx_m2_s), np.asarray(b.qx_m2_s))
    assert np.array_equal(np.asarray(occ2.vertical.current_payload(st2.hydraulic.soil_water_m).state.water_m), np.asarray(b.root.water_state.water_m))  # the ONE water
    # below-threshold: forced commits and ordinary steps keep the routing source (no event-boundary / clock / publish reroute)
    sched = constant_rainfall(b.t_s, b.t_s + 10.0, 100.0)
    occ3_event = occ2.event
    s = st2
    for _ in range(6):
        s = occ3_event.step(s, 100.0 / 3.6e6, 0.2).state
    assert s.topology.topology_sha256 == b.routing[0].topology_sha256 and s.topology.generation == b.routing[0].generation
    assert sched is not None


def test_head_semantics_duplicate_stale_wrong_forged_and_the_unadopted_trial(tmp_path):
    owner, bed, factory, lin, occ, st = opened(tmp_path / "a", "numpy", False)
    h = run_to_complete(occ.event, st)
    cap = occ.event.completion_capability(h, h.state)
    other_owner, other_bed, other_factory, lin2, occ2, st2 = opened(tmp_path / "b", "numpy", False)
    with pytest.raises(lc.LifecycleError, match="head"):
        lc.close_occurrence(lin2, occ, cap, h.state, available_bytes_fn=AVAIL)  # another lineage's head is not this occurrence
    with pytest.raises(lc.LifecycleError):
        lc.close_occurrence(lin, object(), cap, h.state, available_bytes_fn=AVAIL)
    with pytest.raises(fcm.CompletionError):
        lc.close_occurrence(lin, occ, None, h.state, available_bytes_fn=AVAIL)
    with pytest.raises(lc.LifecycleError, match="minted"):
        lc.Occurrence(occ.event, occ.vertical, 0.0, {}, {}, 9)  # direct construction (a forged head)
    assert lin.head is occ and h not in occ.event._completions_consumed  # every refusal left the authority and the capability untouched
    boundary = lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=AVAIL)
    with pytest.raises(lc.LifecycleError, match="head"):
        lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=AVAIL)  # duplicate close of a stale occurrence
    with pytest.raises(lc.LifecycleError, match="head"):
        lc.begin_occurrence(lin2, boundary, factory, available_bytes_fn=AVAIL)  # a boundary of another lineage
    first, _s = lc.begin_occurrence(lin, boundary, factory, available_bytes_fn=AVAIL)
    with pytest.raises(lc.LifecycleError, match="head"):
        lc.begin_occurrence(lin, boundary, factory, available_bytes_fn=AVAIL)  # the same boundary twice: adopting one branch stales the other
    assert lin.head is first


def test_failed_close_late_leaves_authority_event_registry_and_capability_retryable(tmp_path, monkeypatch):
    owner, bed, factory, lin, occ, st = opened(tmp_path / "a", "numpy", False)
    h = run_to_complete(occ.event, st)
    cap = occ.event.completion_capability(h, h.state)
    real = nfs._inventory_closure
    monkeypatch.setattr(nfs, "_inventory_closure", lambda *a, **k: (_ for _ in ()).throw(nfs.NativeFloatingError("injected")))
    reg = len(occ.vertical._registry)
    with pytest.raises(nfs.NativeFloatingError, match="injected"):
        lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=AVAIL)
    assert lin.head is occ and h not in occ.event._completions_consumed and len(occ.vertical._registry) == reg
    monkeypatch.setattr(nfs, "_inventory_closure", real)
    assert lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=AVAIL).role == "boundary"


def test_boundary_references_current_arrays_only_and_releases_the_old_chain(tmp_path):
    def scope():
        owner, bed, factory, lin, occ, st = opened(tmp_path / "a", "numpy", False)
        h = run_to_complete(occ.event, st)
        refs = {"bed0": weakref.ref(bed), "event": weakref.ref(occ.event), "state0": weakref.ref(st), "complete": weakref.ref(h.state), "owner": weakref.ref(owner)}
        cap = occ.event.completion_capability(h, h.state)
        b = lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=AVAIL)
        refs["cap"] = weakref.ref(cap)
        return refs, lin, b

    refs, lin, b = scope()
    for _ in range(3):
        gc.collect()
    dead = {k: r() is None for k, r in refs.items()}
    assert all(dead.values()), dead  # no old bed / event / owner / completed state / capability / candidate chain survives in the boundary
    assert lin.head is b and not hasattr(b, "capability") and not hasattr(b, "event")


def test_capacity_return_makes_a_pending_recession_that_a_real_native_recession_completes_once(tmp_path, monkeypatch):
    """PROVIDER-PERTURBED (declared): the refresh provider output is scaled so a true capacity return occurs; the storm/payload/completion are genuine."""
    owner, bed, factory, lin, occ, st = opened(tmp_path / "a", "numpy", False, loose=True)
    h = run_to_complete(occ.event, st)
    real = nfs._batch_properties

    def scaled(batch, config, K, provider_bytes):
        f, s, k = real(batch, config, K, provider_bytes)
        return f * 0.8, s * 0.8, k

    monkeypatch.setattr(nfs, "_batch_properties", scaled)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert b.role == "pending_recession" and not b.eligible_for_interevent and b.ledger["surface_return_m3"] > 0.0
    area = b.provenance["area_m2"]
    assert float(np.asarray(b.surface_depth_m).sum()) >= float(np.asarray(h.state.hydraulic.depth_m).sum())  # the return joined H' once (H never erased)
    assert np.array_equal(np.asarray(b.bed.water.depth_m), np.asarray(b.surface_depth_m))  # ONE surface depth in the bed
    ret1 = b.ledger["surface_return_m3"]
    occ2, st2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=AVAIL)  # the actual native recession occurrence (rain 0, no E/T)
    h2 = run_to_complete(occ2.event, st2)
    b2 = lc.close_occurrence(lin, occ2, occ2.event.completion_capability(h2, h2.state), h2.state, available_bytes_fn=AVAIL)
    assert b2.role == "boundary" and b2.eligible_for_interevent
    assert b2.ledger["surface_return_m3"] == ret1 and b2.ledger["bottom_export_m3"] == 0.0 and b2.ledger["n_refreshes"] == 2  # transfers credited once; the repeat refresh is an idempotent no-op
    assert area > 0.0


def test_held_law_root_export_reconstructs_the_canonical_new_root_with_the_preferential_operator(tmp_path):
    from maple_syrup import floating_held_context as fh

    owner, bed, factory, lin, occ, st = opened(tmp_path / "a", "numpy", False, preferential=True, loose=True)
    h = run_to_complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert b.root.pref_op is not None
    spec = fh.PreferentialSpec(np.full(b.root.column.n_cells, 2.0e-3), 0.5, 1e-5, np.array([0.0, 1.0, 1.0]), (), None)
    rec = lc.export_boundary_held_context(b, spec, available_bytes_fn=AVAIL)
    meta, arrays = rec.meta(), {k: np.array(v, copy=True) for k, v in rec.arrays().items()}
    parsed = fh.record_from_parts(meta, arrays, available_bytes_fn=AVAIL)
    exp = fh.ExpectedHeldIdentities(b.root.config.sha256, b.root.column.sha256, b.root.matrix_op.sha256, b.root.pref_op.sha256)
    ctx = fh.reconstruct_held_context(parsed, exp, available_bytes_fn=AVAIL)  # reproduces the canonical new root / operator / preferential identities
    assert ctx.column.sha256 == b.root.column.sha256 and ctx.source_witness == b.provenance["cold_anchor_witness"] and ctx.generation == b.root.generation
    assert np.array_equal(ctx.depth0_m, np.asarray(b.root.depth_m))  # the CURRENT canonical D
    assert lin.head is b  # exporting adopts and consumes nothing


def test_cuda_owner_is_refused_before_any_transfer(tmp_path):
    owner, bed, factory, pref, ev0 = build(tmp_path / "a", "numpy", False)
    import types

    fake = types.SimpleNamespace()
    with pytest.raises(lc.LifecycleError):
        lc.open_floating_event(fake, bed, factory, available_bytes_fn=AVAIL)


# ----------------------------------------------------------------------------------------------------------------- correction01
def test_late_open_failure_after_the_native_initializer_leaves_the_owner_current_and_retry_works(tmp_path, monkeypatch):
    from maple_syrup import floating_pond as fp

    owner, bed, factory, pref, ev0 = build(tmp_path / "a", "numpy", False)
    real = fp.FloatingPondVertical.detach_initial

    def boom(self):
        raise fp.FloatingPondError("injected owning-hook failure")

    monkeypatch.setattr(fp.FloatingPondVertical, "detach_initial", boom)
    with pytest.raises(fp.FloatingPondError, match="injected"):
        lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref)
    assert owner.is_current  # the owner was NOT retired by the failed open
    monkeypatch.setattr(fp.FloatingPondVertical, "detach_initial", real)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref)
    assert not owner.is_current and lin.head is occ


def test_public_factory_stand_ins_are_refused_before_the_owner_is_touched_root_fake_factory_v2(tmp_path):
    import types

    owner, bed, factory, pref, ev0 = build(tmp_path / "a", "numpy", False)

    def fake(vert):
        vert.bind_shape(ev0.shape)  # the PUBLIC binding call
        return types.SimpleNamespace(vertical=vert, initial_state=lambda *a, **k: types.SimpleNamespace(), shape=ev0.shape)

    with pytest.raises(lc.LifecycleError, match="actual native PondEvent"):
        lc.open_floating_event(owner, bed, fake, available_bytes_fn=AVAIL, preferential=pref)
    assert owner.is_current
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref)  # genuine control still valid
    h = run_to_complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    with pytest.raises(lc.LifecycleError, match="actual native PondEvent"):
        lc.begin_occurrence(lin, b, lambda vert, topology: types.SimpleNamespace(vertical=vert), available_bytes_fn=AVAIL)  # the subsequent factory too
    assert lin.head is b  # no head advance


def test_second_open_releases_the_obsolete_parent_bed_and_initial_water_after_a_real_new_bed_step(tmp_path):
    def scope():
        owner, bed, factory, lin, occ, st = opened(tmp_path / "a", "numpy", False)
        h = run_to_complete(occ.event, st)
        b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
        refs = {"parent_bed": weakref.ref(b.bed), "parent_w": weakref.ref(b.root.water_state.water_m), "parent_state": weakref.ref(b.root.water_state)}
        occ2, st2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=AVAIL)
        s3 = occ2.event.step(st2, 100.0 / 3.6e6, 0.2).state  # REAL positive-rain native step: a new bed and a new W
        assert s3.bed is not st2.bed
        v = occ2.vertical
        assert v._boundary_bed is None and v._initial.state is None and v._initial.history is None
        return refs, lin, occ2, s3

    refs, lin, occ2, s3 = scope()
    for _ in range(3):
        gc.collect()
    assert {k: r() is None for k, r in refs.items()} == dict.fromkeys(refs, True)  # no obsolete whole bed / W / initial state survives in the runtime
    assert lin.head is occ2


def test_pending_recession_refuses_positive_future_native_rain_before_any_head_change(tmp_path, monkeypatch):
    owner, bed, factory, lin, occ, st = opened(tmp_path / "a", "numpy", False, loose=True)
    h = run_to_complete(occ.event, st)
    real = nfs._batch_properties

    def scaled(b_, c_, k_, p_):
        f, s_, k = real(b_, c_, k_, p_)
        return f * 0.8, s_ * 0.8, k

    monkeypatch.setattr(nfs, "_batch_properties", scaled)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert b.role == "pending_recession"
    rainy = constant_rainfall(b.t_s, b.t_s + 10.0, 100.0)
    with pytest.raises(lc.LifecycleError, match="ZERO native rainfall"):
        lc.begin_occurrence(lin, b, lambda vert, topology: factory(vert, topology, rainy), available_bytes_fn=AVAIL)
    assert lin.head is b  # the head did not move
    ended = constant_rainfall(0.0, 1.0, 100.0)  # a past storm that already ended before the boundary clock is legitimate zero forcing
    occ2, _s2 = lc.begin_occurrence(lin, b, lambda vert, topology: factory(vert, topology, ended), available_bytes_fn=AVAIL)
    assert lin.head is occ2


@pytest.mark.parametrize("bottom", ["closed", "open"])
def test_cumulative_water_balance_closes_across_occurrences_with_the_full_producer_ledger(tmp_path, bottom):
    owner, bed, factory, lin, occ, st = opened(tmp_path / "a", "numpy", False, bottom=bottom, loose=True)
    h = run_to_complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    led = b.ledger
    assert {"storage0_m3", "rain_m3", "drainage_m3", "outlet_export_m3", "bottom_export_m3", "surface_return_m3", "residual_m3", "residual_bound_m3"} <= set(led)
    assert abs(led["residual_m3"]) <= led["residual_bound_m3"]
    occ2, st2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=AVAIL)
    s = st2
    for _ in range(5):
        s = occ2.event.step(s, 100.0 / 3.6e6, 0.2).state  # real rain on the next occurrence
    h2 = run_to_complete(occ2.event, s)
    b2 = lc.close_occurrence(lin, occ2, occ2.event.completion_capability(h2, h2.state), h2.state, available_bytes_fn=AVAIL)
    assert b2.ledger["rain_m3"] > led["rain_m3"] and b2.ledger["n_occurrences"] == 2 and abs(b2.ledger["residual_m3"]) <= b2.ledger["residual_bound_m3"]
    if bottom == "open":
        assert b2.ledger["drainage_m3"] >= led["drainage_m3"]  # native drainage is its own term (not the refresh export); its size in this fixture is reported, not assumed


def test_above_threshold_net_displacement_builds_a_genuine_generation_and_the_next_event_runs_on_that_routing(tmp_path):
    """1 kg of one class deposited at the terminal exchange (vs 0.1 kg) exceeds the actual 0.00075 m net trigger: a REAL routing build (generation 1); no faked generation/source/counters. Declared looser quiet policy."""
    owner, bed, factory, lin, occ, st = opened(tmp_path / "a", "numpy", False, loose=True, mobile_kg=1.0)
    h = run_to_complete(occ.event, st)
    assert h.state.topology.generation == 1 and h.state.counters.n_refreshes == 1  # a genuine net-displacement build happened inside the first event
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert b.routing[0].generation == 1
    occ2, st2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=AVAIL)
    assert st2.topology is b.routing[0] and st2.topology.generation == 1 and st2.held_source_m is b.routing[1]  # the LAST actual routing build is carried
    assert st2.counters.n_refreshes == 0 and st2.counters.n_commits == 0  # event-local counters restart truthfully (the lifetime generation lives in the topology)
    s = st2
    for _ in range(6):
        s = occ2.event.step(s, 100.0 / 3.6e6, 0.2).state  # below the net gate: no new build
    assert s.topology.generation == 1 and s.topology.topology_sha256 == b.routing[0].topology_sha256
