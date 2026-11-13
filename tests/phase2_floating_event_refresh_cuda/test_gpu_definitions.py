"""Actual-GPU test DEFINITIONS of the selected CUDA completed-event property refresh (WRITTEN, NOT RUN by their author: no GPU/CuPy/NVRTC/nvidia-smi was touched). Opt-in (SYRUP_RELEASE_GPU=1); Root runs them on GPU0 with
`.venv-cupy`.

Fixtures: GENUINE completed Pond events (the thin closed lake with a real terminal deposit, Dense and published Windowed device beds, built by the qualified E5 `device_event`) with the CPU EB2 stage on the SAME terrain as the
independent oracle. Declared fixture limits (as the CPU EB2 suite): a natural storm in these fixtures changes properties in at most one column, so true FC/SAT/Ks shifts use the PROVIDER-PERTURBED control (the shared provider
output is scaled; storm, payload and completion stay genuine). States a genuine completed Pond cannot produce (water above FC in untouched columns after an open-bottom event, 65 layers, N = 50003) use COMPONENT controls on
the SHARED owner-free core with a genuine CUDA NativeOwner / public continued state; they are labelled component-only and never presented as completed-Pond states. The real fine-soil provider outcome FC6 == SAT is VALID (not a
zero-Ks bug); real drainage controls use the explicit supported `air_entry_kpa_override` 1 kPa path, never an unsupported soil density."""
# ruff: noqa: RUF059
from __future__ import annotations

import gc
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_T = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e5_gpu_defs_for_refresh", _T / "phase2_floating_pond" / "test_gpu_definitions.py")
e5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e5)

import fpond_builders as fb
import refresh_fixtures as rf
import test_event_refresh as cpu_t
from fpond_builders import ph

from maple_syrup import floating_completion as fcm
from maple_syrup import floating_event_refresh as fer
from maple_syrup import floating_event_refresh_cuda as erc
from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import native_floating_soil as nfs
from maple_syrup import native_floating_soil_cuda as ncuda

BIG = 1 << 40
MEM = fsl.FloatingMemoryError


@pytest.fixture(scope="module")
def cp():
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA refresh is not exercised (no claim)")
    return backend.cupy_module()


def host(a):
    from maple.core.backend import to_host

    return np.asarray(to_host(a))


def lake_kwargs(*, mobile=True):
    return rf.lake_kwargs(mobile=mobile)


def device_completed_full(tmp, windowed, mobile=True):
    """`(event, completed handoff, vertical, initial state)` of the CUDA fixture: `e5.device_event` ALWAYS builds the qualified deep layered bed (`wh.dense_deep_bed`)."""
    ev, st, vert, _o = e5.device_event(tmp, windowed=windowed, **lake_kwargs(mobile=mobile))
    return ev, rf.settle_to_completion(ev, st), vert, st


def device_completed(tmp, windowed, mobile=True):
    return device_completed_full(tmp, windowed, mobile)[:3]


def cpu_twin(tmp, windowed, mobile=True):
    """The MATCHED CPU twin: `fb.make_pair(..., backend="numpy", same kwargs)` (the same deep layered Dense bed / its published-window twin), NOT `fb.make` (V = 3, dz = 0.1, homogeneous: a different soil)."""
    return rf.cpu_matched(tmp, windowed, mobile=mobile)[:3]


def device_oracle(ev, h, vert):
    """The independent cold oracle on the SAME completed CUDA child (host copies; numpy-backend twin of the config; full accepted compact read of every active column)."""
    return rf.cold_oracle(ev, h, vert, fb.config(backend="numpy", bottom="closed"))


def stage(ev, h, **kw):
    cap = ev.completion_capability(h, h.state)
    kw.setdefault("available_bytes_fn", lambda: BIG)
    kw.setdefault("host_available_bytes_fn", lambda: BIG)
    return cap, erc.stage_completed_event_refresh_cuda(cap, h.state, event=ev, **kw)


def perturb(monkeypatch, *, fc_scale=1.0, sat_scale=1.0, ks_scale=1.0):
    real = nfs._batch_properties

    def wrapped(batch, config, K, provider_bytes):
        f, s, k = real(batch, config, K, provider_bytes)
        return f * fc_scale, s * sat_scale, (None if k is None else k * ks_scale)

    monkeypatch.setattr(nfs, "_batch_properties", wrapped)


def snapshot_inputs(vert, h):
    p = vert.current_payload(h.state.hydraulic.soil_water_m)
    return {"W": host(p.state.water_m).copy(), "J": host(p.history.wetting_m).copy(), "I": host(p.history.cumulative_intake_m).copy(), "D": host(p.depth_m).copy(), "column": p.column.sha256, "op": p.matrix_op.sha256,
            "registry": len(vert._registry)}


def assert_inputs_untouched(vert, h, ev, before):
    after = snapshot_inputs(vert, h)
    for k in ("W", "J", "I", "D"):
        assert after[k].tobytes() == before[k].tobytes(), k
    assert after["column"] == before["column"] and after["op"] == before["op"] and after["registry"] == before["registry"] and h not in ev._completions_consumed


# ---- equals the CPU oracle, conserves, never mutates ----------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("windowed", [False, True])
def test_device_refresh_of_a_genuine_completed_event_equals_the_cpu_eb2_oracle_and_mutates_nothing(cp, tmp_path, windowed):
    ev, h, vert, st0 = device_completed_full(tmp_path / "g", windowed)
    cev, ch, cvert, cst0 = rf.cpu_matched(tmp_path / "c", windowed)
    rec = rf.assert_matched(ev, st0, vert, cev, cst0, cvert, windowed)  # MATCHED geometry/strata/initial inventory BEFORE any backend-parity claim
    rf.assert_matched_inventory(vert, h, cvert, ch)  # and the CURRENT inventories within the existing numerical contracts
    print("MATCHED_FIXTURE", rec)
    before = snapshot_inputs(vert, h)
    cap, c = stage(ev, h)
    ccap, cc = cpu_t.stage(cev, ch)
    assert np.array_equal(np.sort(c.dirty), np.sort(cc.dirty)) and np.array_equal(np.sort(c.changed), np.sort(cc.changed)) and np.array_equal(np.sort(c.depth_changed), np.sort(cc.depth_changed))
    np.testing.assert_allclose(host(c.water_state.water_m), np.asarray(cc.water_state.water_m), rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(host(c.surface_return_m), cc.surface_return_m, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(host(c.bottom_export_m), cc.bottom_export_m, rtol=2e-12, atol=1e-14)
    assert c.depth_m.tobytes() == np.asarray(cc.depth_m).tobytes()  # native ordered D (bitwise the CPU oracle)
    keep = np.setdiff1d(np.arange(before["W"].shape[1]), c.changed)
    assert host(c.water_state.water_m)[:, keep].tobytes() == before["W"][:, keep].tobytes()  # unchanged / inactive columns: bit for bit
    assert host(c.history.wetting_m).tobytes() == before["J"].tobytes() and host(c.history.cumulative_intake_m).tobytes() == before["I"].tobytes()  # J/I exact
    assert float(c.water_state.t_s) == float(vert.current_payload(h.state.hydraulic.soil_water_m).state.t_s)
    r = c.receipt
    assert abs(r.whole_residual_m3) <= r.whole_bound_m3
    assert c.column.identity_kind == "content_sha256" and c.column.parent_sha256 is None and c.property_basis == "boundary_selected" and c.provenance["cold_content_hash"] is None  # a canonical NEW root, never a view token
    assert erc.verify_refresh_candidate_cuda(c) is c
    assert_inputs_untouched(vert, h, ev, before)  # pure staging: nothing consumed, registered or mutated
    c2 = erc.stage_completed_event_refresh_cuda(cap, h.state, event=ev, available_bytes_fn=lambda: BIG, host_available_bytes_fn=lambda: BIG)
    assert host(c2.water_state.water_m).tobytes() == host(c.water_state.water_m).tobytes() and c2.provenance["digest"] == c.provenance["digest"]  # repeated staging is valid and identical
    fcm.consume_completion(cap, h.state, event=ev)
    with pytest.raises(fcm.CompletionError, match="consumed"):
        erc.verify_refresh_candidate_cuda(c)  # the completion is consumed LAST by a future adopter, never by staging


def test_the_current_depth_is_the_baseline_not_the_held_d0_and_the_native_depth_equals_the_payload_depth_for_untouched_columns(cp, tmp_path):
    ev, h, vert = device_completed(tmp_path / "g", False)
    p = vert.current_payload(h.state.hydraulic.soil_water_m)
    t = vert.held_template()
    cap, c = stage(ev, h)
    assert c.depth_m.tobytes() == host(p.depth_m).tobytes() or set(np.flatnonzero(c.depth_m != host(p.depth_m)).tolist()) <= set(c.dirty.tolist())  # only dirty columns may differ from the payload D
    print("CUDA_REFRESH_FIXTURE", {"dirty": int(c.dirty.size), "changed": int(c.changed.size), "depth_changed": int(c.depth_changed.size), "payload_D_differs_from_D0": bool(not np.array_equal(host(p.depth_m), host(t.depth0_m)))})


# ---- property gating: provider-perturbed controls, Ks-only, unchanged texture ----------------------------------------------------------------------------------------------------
def test_a_genuinely_completed_zero_mobile_event_has_no_dirty_column_and_bit_exact_water_through_the_same_host_provider(cp, tmp_path):
    """Genuine unchanged stage: a completed event with NO bed write (zero mobile, rain 0), so the gross ledger is genuinely zero. (The deposit lake writes a real class-2 terminal deposit on a stratified bed and DOES change a
    column's properties on the device bed: its `changed` set is checked against the same-child cold oracle below, never asserted empty.)"""
    ev, h, vert = device_completed(tmp_path / "g", False, mobile=False)
    before = snapshot_inputs(vert, h)
    led = h.state.sediment
    assert not bool(host(led.pickup_cell != 0).any()) and not bool(host(led.deposition_cell != 0).any())  # a real zero-gross ledger
    cap, c = stage(ev, h)
    assert c.dirty.size == 0 and c.changed.size == 0 and c.depth_changed.size == 0
    assert host(c.water_state.water_m).tobytes() == before["W"].tobytes() and host(c.history.wetting_m).tobytes() == before["J"].tobytes() and host(c.history.cumulative_intake_m).tobytes() == before["I"].tobytes()
    assert float(host(c.surface_return_m).sum()) == 0.0 and float(host(c.bottom_export_m).sum()) == 0.0
    assert c.column is not vert.held_template().column and c.column.sha256 == before["column"] and c.column.identity_kind == "content_sha256" and c.column.parent_sha256 is None  # identical physical law => identical canonical content digest; new genuine root object, never a view token


@pytest.mark.parametrize("windowed", [False, True])
def test_deposit_lake_changed_set_and_water_equal_the_same_child_cold_oracle(cp, tmp_path, windowed):
    """STRICT changed-set equality against the independent cold oracle run on the SAME completed CUDA child (no second storm involved); water within the existing contracts, return/export within the same, native D bitwise."""
    ev, h, vert = device_completed(tmp_path / "g", windowed)
    before = snapshot_inputs(vert, h)
    cap, c = stage(ev, h)
    changed, depth_changed, water, surf, exp_, fc, sat, depth = device_oracle(ev, h, vert)
    assert set(changed.tolist()) <= set(c.dirty.tolist())  # gross-dirty coverage of the full cold changed set
    assert np.array_equal(np.sort(c.changed), np.sort(changed)) and np.array_equal(np.sort(c.depth_changed), np.sort(depth_changed))
    np.testing.assert_allclose(host(c.water_state.water_m), water, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(host(c.surface_return_m), surf, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(host(c.bottom_export_m), exp_, rtol=2e-12, atol=1e-14)
    assert c.depth_m.tobytes() == depth.tobytes()
    print("DEPOSIT_LAKE_ORACLE", {"windowed": windowed, "dirty": int(c.dirty.size), "changed": int(changed.size)})
    assert_inputs_untouched(vert, h, ev, before)


def test_component_unchanged_provider_read_over_nonzero_selected_indices_gives_an_empty_changed_set_on_the_real_device_child(cp, tmp_path):
    """COMPONENT control (not a forged ledger/completion): on the REAL completed CUDA child, take the full accepted cold read as the 'held' properties and run the stage's selected-tile scatter over ALL active columns (a
    nonzero selected count, the SAME host provider): nothing may differ, so nothing is changed and the reconcile core would not run."""
    ev, h, vert = device_completed(tmp_path / "g", False)
    ctx = ev.context
    ny, nx = ev.shape
    active = np.asarray(vert._active_host()).reshape(-1)
    child = nfs.NativeBed(h.state.bed.voxel_column, h.state.bed.active_layer, ctx.geometry, ctx.grain_classes, ctx.mass_resolution_kg, active.reshape(ny, nx).copy())
    allc = np.flatnonzero(active).astype(np.int64)
    held = vert.held_template()
    fc, sat, depth = host(held.column._maps["theta_field_capacity"]).copy(), host(held.column._maps["theta_saturation"]).copy(), host(vert.current_payload(h.state.hydraulic.soil_water_m).depth_m).copy()
    model = nfs._estimate(None, child, vert.config, 64, 0)  # the ACTUAL resident CUDA bed is read with the vertical's own CUDA config (a numpy config on a device bed is rightly refused by the native guards)
    nfs._read_texture(child, vert.config, allc, fc, sat, depth, model=model, batch_cells=int(model["batch_cells"]), update_properties=True, ks=None)  # the full cold read of THIS child
    from maple.core.backend import to_device

    fc_dev, sat_dev = to_device(np.ascontiguousarray(fc), cp), to_device(np.ascontiguousarray(sat), cp)
    new_fc, new_sat = cp.array(fc_dev, copy=True), cp.array(sat_dev, copy=True)
    depth2 = depth.copy()
    smodel = erc.event_refresh_model_cuda(child, vert.config, n_selected=int(allc.size), batch_cells=64, has_pref=False)
    changed, ks_changed, depth_changed = ncuda._scatter_tiles(cp, child, vert.config, allc, new_fc, new_sat, depth2, None, model=smodel, B=int(smodel["batch_cells"]), update_properties=True,
                                                             old={"depth": depth, "fc": fc_dev, "sat": sat_dev, "ks": None})
    assert allc.size > 0 and changed.size == 0 and depth_changed.size == 0 and ks_changed.size == 0
    assert host(new_fc).tobytes() == fc.tobytes() and host(new_sat).tobytes() == sat.tobytes()


def test_true_fc_sat_shift_returns_surface_water_once_and_conserves_the_current_inventory(cp, tmp_path, monkeypatch):
    """PROVIDER-PERTURBED control (declared; as the CPU EB2 test)."""
    ev, h, vert = device_completed(tmp_path / "g", False)
    before = snapshot_inputs(vert, h)
    perturb(monkeypatch, fc_scale=0.8, sat_scale=0.8)
    cap, c = stage(ev, h)
    r = c.receipt
    assert c.changed.size == c.dirty.size > 0 and r.surface_return_m3 > 0.0 and r.bottom_export_m3 == 0.0
    area = nfs._area(c.child_bed)
    assert abs(float(host(c.surface_return_m).sum()) * area - r.surface_return_m3) <= r.whole_bound_m3
    mask = host(c.column._maps["active"]).astype(bool)
    after = float(host(c.water_state.water_m)[:, mask].sum()) * area
    assert abs((after - float(before["W"][:, mask].sum()) * area) + r.surface_return_m3 + r.bottom_export_m3) <= r.whole_bound_m3
    keep = np.setdiff1d(np.arange(before["W"].shape[1]), c.changed)
    assert host(c.water_state.water_m)[:, keep].tobytes() == before["W"][:, keep].tobytes()
    assert host(c.history.wetting_m).tobytes() == before["J"].tobytes() and host(c.history.cumulative_intake_m).tobytes() == before["I"].tobytes()  # persistent J/I exact through a real changed-only reconcile on the same bed
    assert_inputs_untouched(vert, h, ev, before)


def ks_deposit_event(cp, tmp):
    """A genuinely completed `texture_ks` CUDA event on the deep layered bed with the real class-2 terminal deposit. FIXTURE TRUTH: this stratified deposit legitimately changes column 7's field capacity even with an UNPERTURBED
    provider (Root's diagnostic), so a Ks perturbation here is only ever judged RELATIVE to the unperturbed baseline stage of the same completed event."""
    import dataclasses

    from maple.core.backend import to_device, to_device_tree

    from maple_syrup import floating_pond as fp
    from maple_syrup.pond_event import PondEvent
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters

    kw = lake_kwargs()
    wh = fb._windowed_helpers()
    dense, ctx = wh.dense_deep_bed(kw["surface"], depth=kw["depth"], mobile=kw["mobile"])
    topo = ph.make_topology(dense, [], allow_closed=True)
    bed = to_device_tree(dense, cp)
    dctx = dataclasses.replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, cp))
    cfg = fb.config(backend="cuda", bottom="closed", matrix_conductivity_mode="texture_ks", matrix_conductivity_m_s=None, latent_saturated_conductivity_m_s=3e-6)
    owner = nfs.initialize_native_floating_soil(fb.native_bed(bed, dctx), cfg, initial_relative_saturation=0.999999)
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=fcu.device_reader(cp))
    be = type("B", (), {"hydraulics": "cuda", "xp": cp})()
    ev = PondEvent(dctx, None, plot1_sediment_parameters(xp=cp), constant_rainfall(0.0, 2.0, 0.0), topo, control=wh.control_for(be, windowed=False, max_dt_s=1.0), policy=kw["policy"], floating_vertical=vert)
    st = ev.initial_state(bed, vert.initial_soil_water())
    h = rf.settle_to_completion(ev, st)
    assert vert.held_template().ks_m_s is not None
    return ev, h, vert


def test_ks_perturbation_adds_no_changed_column_no_cascade_and_keeps_history_relative_to_the_unperturbed_baseline(cp, tmp_path, monkeypatch):
    """BASELINE-RELATIVE proof (the fixture's real deposit already changes FC in column 7, so `changed` is NOT empty unperturbed): stage the unperturbed completed event first, then the Ks-perturbed one. The perturbation must
    add NO changed column, no extra water/return/export/drain/cascade, keep J/I exact and change only the Ks record and the operator."""
    ev, h, vert = ks_deposit_event(cp, tmp_path / "g")
    before = snapshot_inputs(vert, h)
    cap = ev.completion_capability(h, h.state)
    base = erc.stage_completed_event_refresh_cuda(cap, h.state, event=ev, available_bytes_fn=lambda: BIG, host_available_bytes_fn=lambda: BIG)
    perturb(monkeypatch, ks_scale=2.0)
    c = erc.stage_completed_event_refresh_cuda(cap, h.state, event=ev, available_bytes_fn=lambda: BIG, host_available_bytes_fn=lambda: BIG)
    assert np.array_equal(np.sort(c.changed), np.sort(base.changed)) and np.array_equal(np.sort(c.depth_changed), np.sort(base.depth_changed)) and c.ks_changed  # Ks is NOT a cascade trigger
    assert host(c.water_state.water_m).tobytes() == host(base.water_state.water_m).tobytes()  # no extra W vs the baseline refresh
    assert host(c.surface_return_m).tobytes() == host(base.surface_return_m).tobytes() and host(c.bottom_export_m).tobytes() == host(base.bottom_export_m).tobytes()
    assert c.receipt.surface_return_m3 == base.receipt.surface_return_m3 and c.receipt.bottom_export_m3 == base.receipt.bottom_export_m3 and c.receipt.moved_percolation_m3 == base.receipt.moved_percolation_m3
    assert host(c.history.wetting_m).tobytes() == before["J"].tobytes() and host(c.history.cumulative_intake_m).tobytes() == before["I"].tobytes()
    assert host(c.ks_m_s).tobytes() != host(base.ks_m_s).tobytes() and c.matrix_op.sha256 != base.matrix_op.sha256 and c.matrix_op.sha256 != before["op"]
    assert_inputs_untouched(vert, h, ev, before)


def test_component_ks_only_perturbation_over_real_unchanged_columns_changes_ks_not_the_changed_set(cp, tmp_path, monkeypatch):
    """ISOLATED Ks-only proof (COMPONENT control, no forged ledger/completion/adoption): on the REAL completed `texture_ks` CUDA child the full accepted cold read (the vertical's own CUDA config) is the 'held' FC/SAT/Ks;
    the stage's selected-tile scatter over all active columns with the Ks provider output scaled must report every selected column Ks-changed and NOT one FC/SAT/depth change."""
    from maple.core.backend import to_device

    ev, h, vert = ks_deposit_event(cp, tmp_path / "g")
    ctx = ev.context
    ny, nx = ev.shape
    active = np.asarray(vert._active_host()).reshape(-1)
    child = nfs.NativeBed(h.state.bed.voxel_column, h.state.bed.active_layer, ctx.geometry, ctx.grain_classes, ctx.mass_resolution_kg, active.reshape(ny, nx).copy())
    allc = np.flatnonzero(active).astype(np.int64)
    held = vert.held_template()
    fc, sat, ks = host(held.column._maps["theta_field_capacity"]).copy(), host(held.column._maps["theta_saturation"]).copy(), host(held.ks_m_s).copy()
    depth = host(vert.current_payload(h.state.hydraulic.soil_water_m).depth_m).copy()
    model = nfs._estimate(None, child, vert.config, 64, 0)
    nfs._read_texture(child, vert.config, allc, fc, sat, depth, model=model, batch_cells=int(model["batch_cells"]), update_properties=True, ks=ks)  # the full cold read of THIS child (actual CUDA config)
    fc_dev, sat_dev, ks_dev = (to_device(np.ascontiguousarray(a), cp) for a in (fc, sat, ks))
    perturb(monkeypatch, ks_scale=2.0)
    new_fc, new_sat, new_ks = cp.array(fc_dev, copy=True), cp.array(sat_dev, copy=True), cp.array(ks_dev, copy=True)
    smodel = erc.event_refresh_model_cuda(child, vert.config, n_selected=int(allc.size), batch_cells=64, has_pref=False)
    changed, ks_changed, depth_changed = ncuda._scatter_tiles(cp, child, vert.config, allc, new_fc, new_sat, depth.copy(), new_ks, model=smodel, B=int(smodel["batch_cells"]), update_properties=True,
                                                             old={"depth": depth, "fc": fc_dev, "sat": sat_dev, "ks": ks_dev})
    assert changed.size == 0 and depth_changed.size == 0 and np.array_equal(np.sort(ks_changed), np.sort(allc))  # Ks differs everywhere, FC/SAT/depth nowhere
    assert host(new_fc).tobytes() == fc.tobytes() and host(new_sat).tobytes() == sat.tobytes()


# ---- failures, retry, ordering ------------------------------------------------------------------------------------------------------------------------------------------------------
def test_a_provider_failure_stages_nothing_leaves_every_input_untouched_and_a_retry_works(cp, tmp_path, monkeypatch):
    ev, h, vert = device_completed(tmp_path / "g", False)
    before = snapshot_inputs(vert, h)
    real = nfs._batch_properties

    def boom(*a, **k):
        raise RuntimeError("injected provider failure")

    monkeypatch.setattr(nfs, "_batch_properties", boom)
    cap = ev.completion_capability(h, h.state)
    with pytest.raises(RuntimeError, match="injected provider failure"):
        erc.stage_completed_event_refresh_cuda(cap, h.state, event=ev, available_bytes_fn=lambda: BIG, host_available_bytes_fn=lambda: BIG)
    assert_inputs_untouched(vert, h, ev, before)
    monkeypatch.setattr(nfs, "_batch_properties", real)
    c = erc.stage_completed_event_refresh_cuda(cap, h.state, event=ev, available_bytes_fn=lambda: BIG, host_available_bytes_fn=lambda: BIG)  # the SAME capability stages after the failure
    assert erc.verify_refresh_candidate_cuda(c) is c


def test_a_cpu_stage_refuses_the_device_event_and_the_cuda_stage_refuses_a_cpu_event_both_typed(cp, tmp_path):
    ev, h, vert = device_completed(tmp_path / "g", False)
    cap = ev.completion_capability(h, h.state)
    with pytest.raises(fer.EventRefreshError, match="CPU"):
        fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=lambda: BIG)
    cev, ch, cvert = cpu_twin(tmp_path / "c", False)
    ccap = cev.completion_capability(ch, ch.state)
    with pytest.raises(erc.EventRefreshCudaError, match="CPU events use floating_event_refresh"):
        erc.stage_completed_event_refresh_cuda(ccap, ch.state, event=cev)


def test_the_current_device_gate_precedes_every_reader_hash_and_allocation(cp, tmp_path, monkeypatch):
    ev, h, vert = device_completed(tmp_path / "g", False)
    cap = ev.completion_capability(h, h.state)
    reads = []
    pool = cp.get_default_memory_pool()
    real = fcu.current_device
    monkeypatch.setattr(fcu, "current_device", lambda c: real(c) + 1)  # ORDERING control with the real gate on one UUID (not a second-device qualification)
    allocs = []

    class Hook:
        def malloc(self, size, stream=None):
            allocs.append(int(size))
            return pool.malloc(size)

    with cp.cuda.using_allocator(Hook().malloc), pytest.raises(erc.EventRefreshCudaError, match="CUDA device gate refused"):
        erc.stage_completed_event_refresh_cuda(cap, h.state, event=ev, available_bytes_fn=lambda: reads.append(1) or BIG, host_available_bytes_fn=lambda: reads.append(2) or BIG)
    assert reads == [] and allocs == []


def test_two_pool_admission_zero_need_minus_one_exact_and_reserve_once_before_any_allocation(cp, tmp_path, monkeypatch):
    ev, h, vert = device_completed(tmp_path / "g", False)
    cap = ev.completion_capability(h, h.state)
    calls = []
    real = erc._admit
    monkeypatch.setattr(erc, "_admit", lambda c_, d_, dv, hh, q, what: calls.append((int(dv), int(hh), what)) or real(c_, d_, dv, hh, q, what))
    erc.stage_completed_event_refresh_cuda(cap, h.state, event=ev, available_bytes_fn=lambda: BIG, host_available_bytes_fn=lambda: BIG)
    (d0, h0, _w0), (d1, h1, _w1) = calls[-2], calls[-1]
    assert (d0, h0) == erc.stage0_model_cuda(ev.shape[0] * ev.shape[1], ev.nc)  # stage 0 is the published dirty-selection model
    pool = cp.get_default_memory_pool()
    allocs = []

    class Hook:
        def malloc(self, size, stream=None):
            allocs.append(int(size))
            return pool.malloc(size)

    cases = (({"available_bytes_fn": lambda: 0, "host_available_bytes_fn": lambda: BIG}, "stage0 device zero"), ({"available_bytes_fn": lambda: BIG, "host_available_bytes_fn": lambda: 0}, "stage0 host zero"),
             ({"available_bytes_fn": lambda: d1 - 1, "host_available_bytes_fn": lambda: BIG}, "stage1 device need-1"),
             ({"available_bytes_fn": lambda: BIG, "host_available_bytes_fn": lambda: h1 + fcu.PY_ALLOWANCE_BYTES - 1}, "stage1 host need-1"),
             ({"available_bytes_fn": lambda: d1, "host_available_bytes_fn": lambda: BIG, "reserve_bytes": 1}, "device reserve once"),
             ({"available_bytes_fn": lambda: BIG, "host_available_bytes_fn": lambda: h1 + fcu.PY_ALLOWANCE_BYTES, "host_reserve_bytes": 1}, "host reserve once"))
    for kw, label in cases:
        allocs.clear()
        before = snapshot_inputs(vert, h)
        with cp.cuda.using_allocator(Hook().malloc), pytest.raises(MEM):
            erc.stage_completed_event_refresh_cuda(cap, h.state, event=ev, **kw)
        if label.startswith("stage0"):
            assert allocs == [], label  # nothing owned before the first admission
        assert_inputs_untouched(vert, h, ev, before)
    c = erc.stage_completed_event_refresh_cuda(cap, h.state, event=ev, available_bytes_fn=lambda: d1, host_available_bytes_fn=lambda: h1 + fcu.PY_ALLOWANCE_BYTES)  # exactly the published need
    assert erc.verify_refresh_candidate_cuda(c) is c  # the exact published stage-1 need loads (stage 0 was admitted first)


# ---- component-only controls on the SHARED owner-free core (declared: not completed-Pond states) ----------------------------------------------------------------------------------
def cuda_owner(cp, tmp, layers=3):
    wh = fb._windowed_helpers()
    z = ph.plane(base=0.12, drop=0.004)
    dense, ctx = wh.dense_deep_bed(z, depth=0.0)
    import dataclasses

    from maple.core.backend import to_device, to_device_tree

    bed = to_device_tree(dense, cp)
    dctx = dataclasses.replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, cp))
    cfg = fb.config(backend="cuda", bottom="open", layers=tuple([0.16 / layers] * layers)) if layers != 3 else fb.config(backend="cuda", bottom="open")
    owner = nfs.initialize_native_floating_soil(fb.native_bed(bed, dctx), cfg, initial_relative_saturation=0.5)
    return owner, cfg, bed, dctx


@pytest.mark.parametrize("layers", [3, 65])
def test_component_unchanged_columns_above_fc_keep_their_water_bit_for_bit_in_the_shared_core(cp, tmp_path, layers):
    owner, cfg, bed, dctx = cuda_owner(cp, tmp_path, layers)
    sat = owner.column._maps["theta_saturation"]
    e = owner.column._maps["edges"]
    L = owner.column.n_layers
    wet = cp.empty_like(owner.state.water_m)
    for k in range(L):
        h = cp.maximum(cp.minimum(owner.column._maps["soil_depth"], e[k + 1]) - e[k], 0.0)
        wet[k] = 0.99 * sat[k] * h  # above field capacity, below saturation (a valid continued state)
    state = fsl.floating_continued_state(owner.column, wet, 0.0)
    w, surf, exp_, acc = ncuda.reconcile_selected_columns(cp, cfg, owner.column, state, owner.depth_m, owner.column._maps["theta_field_capacity"], sat, owner.depth_m, np.zeros(0, dtype=np.int64), 0.25, 8,
                                                         {"reserve_bytes": 0, "host_reserve_bytes": 0})
    assert host(w).tobytes() == host(state.water_m).tobytes() and float(host(surf).sum()) == 0.0 and acc["n_changed_columns"] == 0  # the no-op above FC: not passed through any cascade


def test_component_canonical_preferential_rebind_reproduces_the_prepared_identity_on_a_new_root_for_s0_and_s_positive(cp, tmp_path):
    owner, cfg, bed, dctx = cuda_owner(cp, tmp_path, 3)
    L, N = owner.column.n_layers, owner.column.n_cells
    for species in (0, 2):
        sw = tuple(np.eye(L)[i % L].copy() for i in range(species))
        op = fpi.prepare_floating_preferential(owner.column, a3=cp.full(N, 2e-3), max_fraction_per_hour=0.5, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([0.0, 1.0, 1.0]),
                                               species_receiver_weights=sw, species_contributions=tuple(cp.full(N, 0.5) for _ in range(species)) if species else None)
        maps = owner.column._maps
        new_col = ncuda._device_column(cp, cfg, cp.array(maps["theta_field_capacity"], copy=True), cp.array(maps["theta_saturation"], copy=True), cp.array(maps["soil_depth"], copy=True),
                                       cp.array(maps["active"], copy=True), float(owner.column.cell_area_m2), {"reserve_bytes": 0, "host_reserve_bytes": 0})
        rb = fpi.rebind_preferential_column(op, new_col, available_bytes_fn=lambda: BIG, host_available_bytes_fn=lambda: BIG)
        again = fpi.prepare_floating_preferential(new_col, a3=cp.full(N, 2e-3), max_fraction_per_hour=0.5, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([0.0, 1.0, 1.0]),
                                                  species_receiver_weights=sw, species_contributions=tuple(cp.full(N, 0.5) for _ in range(species)) if species else None)
        assert rb.sha256 == again.sha256 and rb.column is new_col  # the canonical identity a durable new root would reproduce from the raw spec
    reads = []
    with pytest.raises(MEM):
        fpi.rebind_preferential_column(op, new_col, available_bytes_fn=lambda: BIG, host_available_bytes_fn=lambda: reads.append(1) or 0)  # the host staging buffer is admitted BEFORE it exists
    assert reads == [1]


def test_component_matrix_history_rebind_is_exact_on_the_new_root(cp, tmp_path):
    owner, cfg, bed, dctx = cuda_owner(cp, tmp_path, 3)
    maps = owner.column._maps
    new_col = ncuda._device_column(cp, cfg, cp.array(maps["theta_field_capacity"], copy=True), cp.array(maps["theta_saturation"], copy=True), cp.array(maps["soil_depth"], copy=True),
                                   cp.array(maps["active"], copy=True), float(owner.column.cell_area_m2), {"reserve_bytes": 0, "host_reserve_bytes": 0})
    new_op = ncuda._device_op(cfg, new_col, owner.ks_m_s, {"reserve_bytes": 0, "host_reserve_bytes": 0})
    rb = fm.rebind_floating_matrix_history(owner.matrix_op, owner.matrix_history, new_op)
    assert host(rb.history.wetting_m).tobytes() == host(owner.matrix_history.wetting_m).tobytes() and host(rb.history.cumulative_intake_m).tobytes() == host(owner.matrix_history.cumulative_intake_m).tobytes()


# ---- moderate N = 50003 sparse selected count: true allocator peak, pool reuse, counted traffic (COMPONENT control) ------------------------------------------------------------------
def test_moderate_n50003_sparse_selected_component_true_peak_inside_the_model_and_the_pool_is_reused(cp, tmp_path, monkeypatch):
    """COMPONENT control (a genuine completed Pond at N = 50003 is not producible: the 25 km line never drains). A genuine CUDA NativeOwner on the `moderate_bed` (public constructors, NV = 60) supplies the held maps; the
    selected tiles/reconcile/new-root/closure of the stage are run on a sparse selected set with the provider perturbed. Observer = the accepted one: pool sampled at EVERY allocator call + measured rounding; model =
    `event_refresh_model_cuda(...)` device payload; a peak above it is a model underestimate and fails. Traffic is read from MAPLE's counters and attributed; the live owner/bed are excluded and declared."""
    import dataclasses

    import moderate_bed as mbd
    from maple.core import backend as mb
    from maple.core.backend import to_device, to_device_tree

    dense, ctx, _topo, _z = mbd.build(1, 50003, 60)
    bed = to_device_tree(dense, cp)
    dctx = dataclasses.replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, cp))
    cfg = fb.config(backend="cuda", bottom="open")
    owner = nfs.initialize_native_floating_soil(fb.native_bed(bed, dctx), cfg, initial_relative_saturation=0.5)
    child = fb.native_bed(bed, dctx)
    sel = np.arange(0, 50003, 500, dtype=np.int64)
    perturb(monkeypatch, fc_scale=0.9, sat_scale=0.9)
    model = erc.event_refresh_model_cuda(child, cfg, n_selected=int(sel.size), batch_cells=64, has_pref=False)
    pool = cp.get_default_memory_pool()
    maps = owner.column._maps
    N, L = 50003, owner.column.n_layers
    quota = {"reserve_bytes": 0, "host_reserve_bytes": 0}

    class Peak:
        def __init__(self):
            self.base, self.peak, self.rounding = int(pool.used_bytes()), 0, 0

        def __call__(self, size):
            ptr = pool.malloc(size)
            self.peak = max(self.peak, int(pool.used_bytes()) - self.base)
            self.rounding += int(ptr.mem.size) - int(size)
            return ptr

    # Attribute the REAL public static-map hash separately; no counter resets
    # inside a phase and no implementation/transfer suppression.
    hash_records = []
    real_stream_hash = fcu.stream_hash

    def observed_static_hash(cp_, arrays, header):
        before = mb.read_transfer_counters().device_to_host_bytes
        described = [(name, tuple(a.shape), int(a.nbytes)) for name, a in arrays]
        digest = real_stream_hash(cp_, arrays, header)
        downloaded = mb.read_transfer_counters().device_to_host_bytes - before
        expected = sum(n for _name, _shape, n in described)
        assert downloaded == expected, (described, downloaded, expected)
        hash_records.append({"arrays": described, "downloaded": downloaded})
        return digest

    monkeypatch.setattr(fcu, "stream_hash", observed_static_hash)
    plateau = []
    evidence = []
    for _ in range(3):
        gc.collect()
        cp.cuda.Device().synchronize()
        hook = Peak()
        mb.reset_transfer_counters()
        hash_records.clear()
        with cp.cuda.using_allocator(hook):
            new_fc, new_sat = cp.array(maps["theta_field_capacity"], copy=True), cp.array(maps["theta_saturation"], copy=True)
            depth = np.array(owner.depth_m, copy=True)
            changed, _ks, _dc = ncuda._scatter_tiles(cp, child, cfg, sel, new_fc, new_sat, depth, None, model=model, B=int(model["batch_cells"]), update_properties=True,
                                                     old={"depth": owner.depth_m, "fc": maps["theta_field_capacity"], "sat": maps["theta_saturation"], "ks": None})
            water, surf, exp_, acc = ncuda.reconcile_selected_columns(cp, cfg, owner.column, owner.state, owner.depth_m, new_fc, new_sat, depth, changed, float(owner.column.cell_area_m2), int(model["batch_cells"]), quota)
            new_col = ncuda._device_column(cp, cfg, new_fc, new_sat, ncuda._up(cp, depth), cp.array(maps["active"], copy=True), float(owner.column.cell_area_m2), quota)
            new_state = fsl.floating_continued_state(new_col, water, float(owner.state.t_s), **quota)
            ncuda.inventory_closure_cuda(cp, cfg, owner.column, owner.state, new_col, new_state, water, changed, int(model["batch_cells"]), acc, quota, check_unrefreshed=bool(changed.size))
        c = mb.read_transfer_counters()
        assert hook.peak <= model["device_payload_bytes"] + hook.rounding, (hook.peak, model["device_payload_bytes"], hook.rounding)  # TRUE peak vs the declared device model + measured rounding
        # Canonical identity MUST stream static maps, whose sum can exceed
        # one full water array. Attribute actual bytes; never call that a
        # dynamic-water mirror or omit it from the reported total.
        static_hash_bytes = sum(r["downloaded"] for r in hash_records)
        assert hash_records and static_hash_bytes > 0
        # C5 hardening (reviewed): ONLY the static identity maps may be attributed to the hash; a dynamic water/history/depth array hashed through `stream_hash` would otherwise be subtracted unseen
        hashed_names = {n for r in hash_records for n, _shape, _nb in r["arrays"]}
        static_names = set(maps) | {"active", "edges"}  # the ACTUAL owned static arrays of `_device_column` (property maps, soil_depth, mask, edges, ...)
        assert hashed_names <= static_names and not hashed_names & {"water_m", "wetting_m", "cumulative_intake_m", "depth_m", "water", "history", "surface_depth_m"}, hashed_names
        dynamic_and_selected_bytes = c.device_to_host_bytes - static_hash_bytes
        assert 0 <= dynamic_and_selected_bytes < L * N * 8, (c, hash_records)
        assert c.host_to_device_bytes <= 2 * N * 8 + (4 * L + 8) * int(model["batch_cells"]) * 8 * (int(sel.size) // int(model["batch_cells"]) + 2) + 16 * int(sel.size) + 4096, c
        evidence.append({"device_peak": hook.peak, "device_model": model["device_payload_bytes"], "measured_rounding": hook.rounding, "D2H_total": c.device_to_host_bytes, "D2H_static_hash": static_hash_bytes, "D2H_selected_and_reductions": dynamic_and_selected_bytes, "H2D_total": c.host_to_device_bytes, "hashes": list(hash_records)})
        del new_fc, new_sat, water, surf, exp_, new_col, new_state
        gc.collect()
        cp.cuda.Device().synchronize()
        plateau.append(int(pool.used_bytes()))
    assert max(plateau) - min(plateau) <= hook.rounding, plateau  # pool reuse: the released plateau is stable across repetitions

    print("ROOT_REFRESH_MODERATE_EVIDENCE", {"N": N, "L": L, "selected": int(sel.size), "repetitions": evidence, "released_plateau": plateau})
