# ruff: noqa: RUF059
"""EB2: staged selected property refresh from the CURRENT inventory of a genuinely completed event (CPU NumPy/Numba x Dense/published Windowed). Real wet/splash/quiet/terminal-deposit storms; a FULL COLD child property
oracle (TEST ONLY) checks the selected refresh and the gross-dirty coverage. Physical-fixture limits are stated in the report (a natural storm in these fixtures changes properties in at most one column)."""
from __future__ import annotations

import re
import tracemalloc

import numpy as np
import pytest
from case import AVAIL
from test_terminal import lake, settled

from maple_syrup import floating_completion as fcm
from maple_syrup import floating_event_refresh as fer
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def completed(tmp, backend, windowed, *, splash=False):
    ev, st, vert = lake(tmp, backend, windowed, splash=splash)
    s = settled(ev, st)
    q = ev.evolve(s, 120.0, stop_when_quiet=True)
    assert q.quiet_ready
    h = ev.settle_retaining_water(q.state)
    assert h.complete
    return ev, h, vert


def stage(ev, h, **kw):
    cap = ev.completion_capability(h, h.state)
    return cap, fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=kw.pop("avail", AVAIL), **kw)


def cold_oracle(ev, h, vert):
    """FULL cold child property oracle driven by the CURRENT payload: every active column read; the changed set by the accepted comparison; W by the accepted reconcile on exactly those columns."""
    p = vert.current_payload(h.state.hydraulic.soil_water_m)
    held = vert.held_template()
    cfg = vert.config
    ctx = ev.context
    active = np.asarray(vert._active_host()).reshape(-1)
    ny, nx = ev.shape
    child = nfs.NativeBed(h.state.bed.voxel_column, h.state.bed.active_layer, ctx.geometry, ctx.grain_classes, ctx.mass_resolution_kg, active.reshape(ny, nx).copy())
    allc = np.flatnonzero(active).astype(np.int64)
    maps = held.column._maps
    fc, sat, depth = (np.array(maps["theta_field_capacity"], copy=True), np.array(maps["theta_saturation"], copy=True), np.array(p.depth_m, copy=True))
    model = nfs._estimate(None, child, cfg, 64, 0)
    nfs._read_texture(child, cfg, allc, fc, sat, depth, model=model, batch_cells=int(model["batch_cells"]), update_properties=True, ks=None if held.ks_m_s is None else np.array(held.ks_m_s, copy=True))
    changed, _dc = nfs._changed_columns_parts(cfg, maps, np.asarray(p.depth_m), allc, fc, sat, depth, 64)
    water, surf, exp_, _acc = nfs._reconcile_changed(cfg, maps, np.asarray(p.depth_m), np.asarray(p.state.water_m), fc, sat, depth, changed, nfs._area(child), 64, float(p.state.t_s))
    return changed, water, surf, exp_, fc, sat, depth


@pytest.mark.parametrize("splash", [False, True])
@pytest.mark.parametrize("windowed", [False, True])
def test_selected_refresh_equals_the_full_cold_oracle_and_the_gross_dirty_set_covers_it(tmp_path, backend, windowed, splash):
    ev, h, vert = completed(tmp_path / "a", backend, windowed, splash=splash)
    p = vert.current_payload(h.state.hydraulic.soil_water_m)
    w_before, hist_before = np.array(p.state.water_m, copy=True), (np.array(p.history.wetting_m, copy=True), np.array(p.history.cumulative_intake_m, copy=True))
    reg_before, comp_before = len(vert._registry), len(ev._completions)
    cap, c = stage(ev, h)
    changed, water, surf, exp_, fc, sat, depth = cold_oracle(ev, h, vert)
    assert set(changed.tolist()) <= set(c.dirty.tolist())  # coverage: the changed set of the FULL cold oracle lies inside the gross-ledger dirty set
    assert np.array_equal(np.sort(c.changed), np.sort(changed))
    assert np.array_equal(np.asarray(c.water_state.water_m), water) and np.array_equal(c.surface_return_m, surf) and np.array_equal(c.bottom_export_m, exp_)  # bit-exact vs the accepted laws on the oracle set
    keep = np.setdiff1d(np.arange(w_before.shape[1]), c.changed)
    assert np.array_equal(np.asarray(c.water_state.water_m)[:, keep], w_before[:, keep])  # unchanged / inactive columns bit for bit
    assert np.array_equal(np.asarray(c.history.wetting_m), hist_before[0]) and np.array_equal(np.asarray(c.history.cumulative_intake_m), hist_before[1])  # J/I bit exact
    assert float(c.water_state.t_s) == float(p.state.t_s)  # no time advance
    assert c.pref_op is None and p.pref_op is None  # this fixture has no preferential operator (see the dedicated preferential test)
    assert c.property_basis == "boundary_selected" and c.provenance["cold_content_hash"] is None and c.provenance["kind"] == "execution_epoch/v1"
    assert c.child_bed.voxel_column is h.state.bed.voxel_column and c.state is h.state  # references only: no old bed, no owner
    r = c.receipt
    assert abs(r.whole_residual_m3) <= r.whole_bound_m3
    # pure staging: nothing consumed or registered; repeated staging is valid and identical
    assert len(vert._registry) == reg_before and len(ev._completions) == comp_before and h not in ev._completions_consumed
    c2 = fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=AVAIL)
    assert np.array_equal(np.asarray(c2.water_state.water_m), np.asarray(c.water_state.water_m)) and c2.provenance["digest"] == c.provenance["digest"]
    assert fer.verify_refresh_candidate(c) is c
    fcm.consume_completion(cap, h.state, event=ev)
    with pytest.raises(fcm.CompletionError, match="consumed"):
        fer.verify_refresh_candidate(c)  # the completion is consumed LAST by a future adopter, not by staging


def perturb(monkeypatch, *, fc_scale=1.0, sat_scale=1.0, ks_scale=1.0):
    real = nfs._batch_properties

    def wrapped(batch, config, K, provider_bytes):
        f, s, k = real(batch, config, K, provider_bytes)
        return f * fc_scale, s * sat_scale, (None if k is None else k * ks_scale)

    monkeypatch.setattr(nfs, "_batch_properties", wrapped)


def test_true_fc_sat_shift_returns_surface_water_once_and_conserves_the_current_inventory(tmp_path, monkeypatch):
    """PROVIDER-PERTURBED control (declared): the storm, payload and completion are genuine; the texture->property provider output of the refresh is scaled so a true FC/SAT shift occurs (a natural storm in these fixtures does not
    produce a large one)."""
    ev, h, vert = completed(tmp_path / "a", "numpy", False)
    p = vert.current_payload(h.state.hydraulic.soil_water_m)
    w0 = np.array(p.state.water_m, copy=True)
    perturb(monkeypatch, fc_scale=0.8, sat_scale=0.8)
    _cap, c = stage(ev, h)
    r = c.receipt
    assert c.changed.size == c.dirty.size > 0 and r.surface_return_m3 > 0.0  # closed bottom, full soil: the SAT drop returns water to the surface
    area = nfs._area(c.child_bed)
    assert abs(float(c.surface_return_m.sum()) * area - r.surface_return_m3) <= r.whole_bound_m3 and r.bottom_export_m3 == 0.0
    mask = np.asarray(c.column._maps["active"])
    after = float(np.asarray(c.water_state.water_m)[:, mask].sum()) * area
    assert abs((after - float(w0[:, mask].sum()) * area) + r.surface_return_m3 + r.bottom_export_m3) <= r.whole_bound_m3  # conserved: nothing lost, returned exactly once
    keep = np.setdiff1d(np.arange(w0.shape[1]), c.changed)
    assert np.array_equal(np.asarray(c.water_state.water_m)[:, keep], w0[:, keep])  # no extra cascade elsewhere (all columns sit above FC)


def test_ks_only_change_replaces_the_operator_without_a_cascade_and_keeps_water_and_history(tmp_path, monkeypatch):
    import fpond_builders as fb
    from fpond_builders import ph

    from maple_syrup.pond_event import PondCompletionPolicy

    z = ph.basin() - 0.08
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = 1.0e-1
    kw = {"surface": z, "depth": np.maximum(0.135 - z, 0.0), "mobile": mob, "closed": True, "bottom": "closed", "saturation": 0.999999, "backend": "numpy", "splash": False, "rain_mm_h": 0.0, "end_s": 2.0, "max_dt_s": 1.0,
          "policy": PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True), "matrix_conductivity_mode": "texture_ks", "matrix_conductivity_m_s": None, "latent_saturated_conductivity_m_s": 3e-6}
    ev, st, _b, _c, vert, _o = fb.make(**kw)
    s = settled(ev, st)
    h = ev.settle_retaining_water(ev.evolve(s, 120.0, stop_when_quiet=True).state)
    assert h.complete and vert.held_template().ks_m_s is not None
    p = vert.current_payload(h.state.hydraulic.soil_water_m)
    w0, j0 = np.array(p.state.water_m, copy=True), np.array(p.history.wetting_m, copy=True)
    perturb(monkeypatch, ks_scale=2.0)
    _cap, c = stage(ev, h)
    assert c.ks_changed and c.changed.size == 0  # Ks is not a cascade trigger
    assert np.array_equal(np.asarray(c.water_state.water_m), w0) and np.array_equal(np.asarray(c.history.wetting_m), j0)
    assert c.matrix_op.sha256 != p.matrix_op.sha256 and c.receipt.surface_return_m3 == 0.0 and c.receipt.bottom_export_m3 == 0.0


def test_fake_foreign_incomplete_and_consumed_proofs_are_refused_before_any_work(tmp_path, monkeypatch):
    ev, h, _vert = completed(tmp_path / "a", "numpy", False)
    cap = ev.completion_capability(h, h.state)
    plain = ev.step(h.state, 0.0, 0.2).state
    other, _s, _v = lake(tmp_path / "b", "numpy", False)
    calls = []
    monkeypatch.setattr(nfs, "_estimate", lambda *a, **k: calls.append(1))
    for args, kw in (((object(), h.state), {"event": ev}), ((cap, plain), {"event": ev}), ((cap, h.state), {"event": other}), ((None, h.state), {"event": ev})):
        with pytest.raises(fcm.CompletionError):
            fer.stage_completed_event_refresh(*args, available_bytes_fn=AVAIL, **kw)
    fcm.consume_completion(cap, h.state, event=ev)
    with pytest.raises(fcm.CompletionError, match="consumed"):
        fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=AVAIL)
    assert calls == []  # every refusal precedes the first model/array


def test_cuda_vertical_is_refused_typed_before_any_array_or_reader(tmp_path, monkeypatch):
    ev, h, vert = completed(tmp_path / "a", "numpy", False)
    cap = ev.completion_capability(h, h.state)
    reads = []
    monkeypatch.setattr(vert, "backend", "cuda")
    with pytest.raises(fer.EventRefreshError, match="CPU"):
        fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=lambda: reads.append(1) or 1 << 40)
    assert reads == []


def test_late_failure_leaves_everything_unchanged_and_retry_works(tmp_path, monkeypatch):
    ev, h, vert = completed(tmp_path / "a", "numpy", False)
    cap = ev.completion_capability(h, h.state)
    reg, comp = len(vert._registry), len(ev._completions)
    real = nfs._inventory_closure

    def boom(*a, **k):
        raise nfs.NativeFloatingError("injected closure failure")

    monkeypatch.setattr(nfs, "_inventory_closure", boom)
    with pytest.raises(nfs.NativeFloatingError, match="injected"):
        fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=AVAIL)
    assert len(vert._registry) == reg and len(ev._completions) == comp and h not in ev._completions_consumed
    monkeypatch.setattr(nfs, "_inventory_closure", real)
    assert fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=AVAIL) is not None


def test_no_cold_scan_no_owner_and_the_selected_read_covers_only_the_dirty_columns(tmp_path, monkeypatch):
    ev, h, vert = completed(tmp_path / "a", "numpy", False)
    cap = ev.completion_capability(h, h.state)
    forbidden = []
    for name in ("_scan", "native_source_witness", "stage_native_refresh", "initialize_native_floating_soil"):
        monkeypatch.setattr(nfs, name, lambda *a, _n=name, **k: forbidden.append(_n))
    seen = []
    real = nfs._read_texture
    monkeypatch.setattr(nfs, "_read_texture", lambda child, cfg, idx, *a, **k: (seen.append(np.array(idx)), real(child, cfg, idx, *a, **k))[1])
    c = fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=AVAIL)
    assert forbidden == [] and len(seen) == 1 and np.array_equal(seen[0], c.dirty) and c.dirty.size < ev.shape[0] * ev.shape[1]


def test_admission_stage0_stage1_exact_need_minus_one_reserve_once_and_traced_peak(tmp_path):
    ev, h, vert = completed(tmp_path / "a", "numpy", False)
    cap = ev.completion_capability(h, h.state)

    def need_of(avail, reserve=0):
        with pytest.raises(nfs.NativeFloatingMemoryError) as info:
            fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=lambda: avail, reserve_bytes=reserve)
        return int(re.search(r"estimated (\d+) B", str(info.value)).group(1))

    n0 = need_of(0)
    n1 = need_of(n0)  # stage 0 exactly satisfied: the stage-1 composed need is what remains
    assert n1 > 0
    for bad in (n1 - 1,):
        assert need_of(bad) == n1
    c = fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=lambda: max(n0, n1))
    assert c.model["required_bytes"] == n1
    assert need_of(0, reserve=7) == n0 + 7 and need_of(n0 + 7, reserve=7) == n1 + 7  # the reserve is counted once per stage
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=lambda: max(n0, n1))
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert peak <= max(n0, n1), (peak, n0, n1)


def test_windowed_stage_uses_the_real_native_cache_bound(tmp_path):
    ev, h, vert = completed(tmp_path / "a", "numpy", True)
    cap = ev.completion_capability(h, h.state)
    c = fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=AVAIL)
    assert c.model["required_bytes"] >= int(h.state.bed.voxel_column.deep_store.mapped_bytes_bound)  # the accepted compact model includes the real mapping bound of the windowed store
    assert fsl is not None


def test_preferential_operator_is_rebound_to_the_new_column_sharing_the_raw_law(tmp_path):
    import fpond_builders as fb
    from fpond_builders import ph

    from maple_syrup.pond_event import PondCompletionPolicy

    z = ph.basin() - 0.08
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = 1.0e-1
    kw = {"surface": z, "depth": np.maximum(0.135 - z, 0.0), "mobile": mob, "closed": True, "bottom": "closed", "saturation": 0.999999, "backend": "numpy", "splash": False, "rain_mm_h": 0.0, "end_s": 2.0, "max_dt_s": 1.0,
          "policy": PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-6, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True), "preferential": True}
    ev, st, _b, _c, vert, _o = fb.make(**kw)
    h = None
    state = st
    for _ in range(8):
        q = ev.evolve(state, 200.0, stop_when_quiet=True)
        state = q.state
        if q.quiet_ready:
            h = ev.settle_retaining_water(state)
            state = h.state
            if h.complete:
                break
    assert h is not None and h.complete
    p = vert.current_payload(h.state.hydraulic.soil_water_m)
    assert p.pref_op is not None
    _cap, c = stage(ev, h)
    assert c.pref_op is not None and c.pref_op.column is c.column and c.pref_op.sha256 != p.pref_op.sha256
    assert all(a is b for a, b in zip(c.pref_op._law, p.pref_op._law, strict=True)) and c.pref_op.species is p.pref_op.species and c.pref_op.background is p.pref_op.background
    # CANONICAL identity: equal to the normal public prepare of the SAME raw law on the NEW column, and idempotent under repeated rebinding
    from maple_syrup import floating_preferential_infiltration as fpi

    normal = fpi.prepare_floating_preferential(c.column, a3=np.full(c.column.n_cells, 2.0e-3), max_fraction_per_hour=0.5, max_intake_rate_m_s=1e-5, background_receiver_weights=[0.0, 1.0, 1.0], available_bytes_fn=AVAIL)
    assert c.pref_op.sha256 == normal.sha256 and c.pref_op.column_sha256 == normal.column_sha256 == c.column.sha256
    for a, b in zip(c.pref_op._law, normal._law, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))  # same raw law, bit for bit
    again = fpi.rebind_preferential_column(c.pref_op, c.column)
    assert again.sha256 == c.pref_op.sha256 and all(a is b for a, b in zip(again._law, c.pref_op._law, strict=True))
    c2 = fer.stage_completed_event_refresh(ev.completion_capability(h, h.state), h.state, event=ev, available_bytes_fn=AVAIL)
    assert c2.pref_op.sha256 == c.pref_op.sha256  # repeated staging reproduces the same root identity


def test_no_owned_n_scaled_allocation_precedes_the_first_admission_root_zero_reader_reproducer(tmp_path, monkeypatch):
    """Root's reproducer: a zero reader on a real complete event. The child bed's mask is the vertical's sealed array BY REFERENCE (no copy), and no numpy allocator runs in the module before the first reader call."""
    ev, h, vert = completed(tmp_path / "a", "numpy", False)
    cap = ev.completion_capability(h, h.state)
    log, order = [], []
    real_check = nfs._check_bed
    monkeypatch.setattr(nfs, "_check_bed", lambda bed, *a, **k: (log.append(bed), real_check(bed, *a, **k))[1])

    class Proxy:
        def __getattr__(self, name):
            f = getattr(np, name)
            if name in ("array", "zeros", "empty", "copy", "ones", "ascontiguousarray", "flatnonzero", "full", "any"):
                return lambda *a, **k: (order.append(name), f(*a, **k))[1]
            return f

    monkeypatch.setattr(fer, "np", Proxy())
    reg, comp = len(vert._registry), len(ev._completions)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=lambda: order.append("READ") or 0)
    assert order[:1] == ["READ"] and order.count("READ") == 1  # the first thing that happens is the first read: no numpy allocation before it
    assert len(log) == 1 and np.shares_memory(log[0].active_mask, vert._active_host())  # by reference: no owned N-scaled copy
    assert len(vert._registry) == reg and len(ev._completions) == comp and h not in ev._completions_consumed
    order.clear()
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=lambda: order.append("READ") or 10)  # one byte short of nothing: still the first gate
    assert order[:1] == ["READ"]
    monkeypatch.setattr(fer, "np", np)
    assert fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=AVAIL) is not None  # retry works
