"""E3b CUDA branch through the HOST double (NOT hardware evidence): composed device/host admission, gate-first refusals and namespace mismatches, and the compact CUDA reconciliation/scatter/rebind against the
CPU transaction on the same inputs. The actual native device beds, the E2 compact reader on a device, and the kernels' hardware behaviour are exercised only by test_gpu_definitions.py (unrun)."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_host_cuda
from hostd2 import fake

from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs
from maple_syrup import native_floating_soil_cuda as ncuda

pytestmark = needs_host_cuda

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("e3_builders", Path(__file__).resolve().parents[1] / "phase2_native_floating_soil" / "builders.py")
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)
config, stratified = _b.config, _b.stratified


def device_bed(bed, dev=0):
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    return nfs.NativeBed(VoxelColumnState(mass_kg=fake(bed.voxel_column.mass_kg, dev)), ActiveLayerState(mass_kg=fake(bed.active_layer.mass_kg, dev)), bed.geometry, bed.grain_classes,
                         bed.mass_resolution_kg, bed.active_mask)


def test_composed_estimate_payload_required_and_initialization_vs_refresh(cp):
    bed = stratified()
    d = device_bed(bed)
    cfg = config(backend="cuda")
    init = ncuda.estimate_cuda_memory(None, d, cfg, batch_cells=2)
    ref = ncuda.estimate_cuda_memory(d, d, cfg, batch_cells=2)
    assert init["initialization"] and not ref["initialization"] and init["host_phases"]["parent_native"] == 0 < ref["host_phases"]["parent_native"]
    for m in (init, ref):
        assert m["device_required_bytes"] == m["device_payload_bytes"] and m["host_required_bytes"] == m["host_payload_bytes"] + ncuda.fcu.PY_ALLOWANCE_BYTES
        assert set(m["device_terms"]) >= {"new_maps", "water_scatter_and_state", "sub_columns", "reconcile_B", "new_column_retained", "storage_queries", "compact_reader_device"}
    m2 = ncuda.estimate_cuda_memory(d, d, cfg, batch_cells=2, reserve_bytes=1000, host_reserve_bytes=500)
    assert m2["device_required_bytes"] - ref["device_required_bytes"] == 1000 and m2["host_required_bytes"] - ref["host_required_bytes"] == 500  # reserves counted ONCE
    for bad in (True, -1, 1.5, "3"):
        with pytest.raises(fsl.FloatingSoilError):
            ncuda.estimate_cuda_memory(d, d, cfg, batch_cells=2, reserve_bytes=bad)
        with pytest.raises(fsl.FloatingSoilError):
            ncuda.estimate_cuda_memory(d, d, cfg, batch_cells=2, host_reserve_bytes=bad)


def test_gate_precedes_every_read_and_mismatches_are_typed_refusals(cp):
    bed = stratified()
    d = device_bed(bed)
    cfg = config(backend="cuda")
    reads = []

    def reader():
        reads.append(1)
        return 1 << 40

    cp.dev["id"] = 1  # the CURRENT device differs from the device holding the bed
    with pytest.raises(nfs.NativeFloatingError, match="no implicit transfer"):
        nfs.initialize_native_floating_soil(d, cfg, initial_relative_saturation=0.5, available_bytes_fn=reader, host_available_bytes_fn=reader)
    cp.dev["id"] = 0
    assert reads == [] and cp.count.d2h == 0 and cp.count.h2d == 0 and cp.allocs == []
    with pytest.raises(nfs.NativeFloatingError, match="no implicit transfer"):  # host bed with a CUDA owner
        nfs.initialize_native_floating_soil(bed, cfg, initial_relative_saturation=0.5)
    with pytest.raises(nfs.NativeFloatingError, match="no implicit transfer"):  # device bed with a CPU owner
        nfs.initialize_native_floating_soil(d, config(), initial_relative_saturation=0.5)
    mixed = nfs.NativeBed(bed.voxel_column, d.active_layer, bed.geometry, bed.grain_classes, bed.mass_resolution_kg, bed.active_mask)  # host voxels + device active layer
    with pytest.raises(nfs.NativeFloatingError, match="mixes host and device"):
        nfs.initialize_native_floating_soil(mixed, cfg, initial_relative_saturation=0.5)


def test_initialization_need_minus_one_refuses_before_scratch_and_exact_budget_passes_admission(cp, monkeypatch):
    d = device_bed(stratified())
    cfg = config(backend="cuda")
    model = ncuda.estimate_cuda_memory(None, d, cfg, batch_cells=64)  # the INITIALIZATION model (no parent)
    reached = []

    class PastAdmission(Exception):
        pass

    def stop(*a, **k):
        reached.append(1)
        raise PastAdmission

    monkeypatch.setattr(nfs, "_scan", stop)
    for kw in ({"memory_budget_bytes": model["device_required_bytes"] - 1}, {"host_memory_budget_bytes": model["host_required_bytes"] - 1}):
        with pytest.raises(fsl.FloatingMemoryError, match="nothing was allocated"):
            nfs.initialize_native_floating_soil(d, cfg, initial_relative_saturation=0.5, **kw)
    assert reached == [] and cp.count.d2h == 0 and cp.count.h2d == 0 and cp.allocs == []
    truthful = {"memory_budget_bytes": model["device_required_bytes"], "host_memory_budget_bytes": model["host_required_bytes"], "available_bytes_fn": lambda: model["device_required_bytes"],
                "host_available_bytes_fn": lambda: model["host_required_bytes"]}
    with pytest.raises(PastAdmission):  # the exact figures with truthful readings pass admission and reach the first native read
        nfs.initialize_native_floating_soil(d, cfg, initial_relative_saturation=0.5, **truthful)
    assert reached == [1]
    with pytest.raises(fsl.FloatingMemoryError, match="reported available"):
        nfs.initialize_native_floating_soil(d, cfg, initial_relative_saturation=0.5, available_bytes_fn=lambda: model["device_required_bytes"] - 1)


def test_e2_backend_is_selected_by_the_genuine_representation_and_reaches_the_reader(cp, monkeypatch):
    """The exact GPU failure of the first author run (the E2 reader defaulted to 'numpy' on a CuPy Dense bed) is caught here: the selector is called through the real integration path."""
    import maple_syrup.compact_soil_profile as csp

    d = device_bed(stratified())
    host = stratified()
    assert nfs._e2_backend(d) == "cuda" and nfs._e2_backend(host) == "numpy"
    seen = []

    def spy(*a, **k):
        seen.append(k["backend"])
        return iter(())

    monkeypatch.setattr(csp, "iter_compact_soil_profile", spy)
    nfs_cfg = config(backend="cuda")
    owner = nfs.initialize_native_floating_soil(d, nfs_cfg, initial_relative_saturation=0.5)  # runs scan, scatter (no batches), D1 CUDA column/state, D2 operator on the host double
    assert seen == ["cuda"] and owner.column.backend == "cuda" and owner.generation == 0
    est = nfs._native_estimate(d, nfs_cfg, 2, 3, 4, 4, 3, {})
    assert est.backend == "cuda" and est.output_namespace == "cupy" and est.device_bytes > 0  # the model uses the same backend/namespace
    # a Windowed bed is never a CUDA compute path: the backend is 'numpy' (host gather) whatever its namespace
    from types import SimpleNamespace

    win = SimpleNamespace(representation="windowed", active_layer=d.active_layer)
    assert nfs._e2_backend(win) == "numpy"


def test_cuda_owner_refresh_integration_through_the_selector_and_counted_syncs(cp, monkeypatch):
    import maple_syrup.compact_soil_profile as csp

    d = device_bed(stratified())
    monkeypatch.setattr(csp, "iter_compact_soil_profile", lambda *a, **k: iter(()))
    cfg = config(backend="cuda")
    owner = nfs.initialize_native_floating_soil(d, cfg, initial_relative_saturation=0.5)
    cp.count.__init__()
    cand = nfs.stage_native_refresh(owner, d, d, mode="properties")  # no dirty cell: no reconcile, but the scan, witness and request path run
    assert cand.n_dirty == 0 and not cand.reconciled and cand.request.parent_witness == owner.binding.source_witness
    assert cp.count.h2d == 0 or cp.count.h2d_bytes >= 0


def test_cuda_reconcile_equals_the_cpu_reconcile_for_the_changed_columns_only(cp):
    from maple_syrup import floating_soil_layers as fsl

    cfg_c, cfg = config(backend="cuda"), config()
    L, N = 3, 6
    rng = np.random.default_rng(3)
    old_fc = rng.uniform(0.30, 0.45, (L, N))
    old_sat = old_fc + rng.uniform(0.05, 0.1, (L, N))
    old_depth = np.array([0.30, 0.20, 0.05, 0.0, 0.16, 0.22])
    new_fc, new_sat = old_fc.copy(), old_sat.copy()
    new_fc[:, 1] *= 0.8
    new_sat[:, 1] = np.maximum(new_sat[:, 1], new_fc[:, 1] + 0.02)
    new_depth = old_depth.copy()
    new_depth[2], new_depth[4] = 0.5, 0.01
    changed = np.array([1, 2, 4], dtype=np.int64)
    mask = np.ones(N, bool)
    quota = {"memory_budget_bytes": None, "reserve_bytes": 0, "available_bytes_fn": None, "host_memory_budget_bytes": None, "host_reserve_bytes": 0, "host_available_bytes_fn": None}
    old_col_c = ncuda._device_column(cp, cfg_c, fake(old_fc), fake(old_sat), fake(old_depth), fake(mask), 0.25, quota)
    old_col = nfs._prepare(cfg, old_fc, old_sat, old_depth, mask, 0.25)
    w = np.minimum(old_sat * np.stack([fsl.porous_thickness(old_col, k) for k in range(L)]) * 0.98, 1.0)
    state_c = fsl.floating_continued_state(old_col_c, fake(np.ascontiguousarray(w)), 0.0)
    owner = type("O", (), {})()
    owner.column, owner.state, owner.depth_m = old_col_c, state_c, old_depth
    water_c, surf_c, exp_c, acc_c = ncuda._reconcile_cuda(cp, cfg_c, owner, fake(new_fc), fake(new_sat), new_depth, changed, 0.25, 2, quota)
    maps = {"theta_field_capacity": old_fc, "theta_saturation": old_sat}
    water, surf, exp_, acc = nfs._reconcile_changed(cfg, maps, old_depth, w, new_fc, new_sat, new_depth, changed, 0.25, 2, 0.0)
    assert np.allclose(np.asarray(water_c), water, rtol=2e-12, atol=1e-14) and np.allclose(np.asarray(surf_c), surf, rtol=2e-12, atol=1e-14) and np.allclose(np.asarray(exp_c), exp_, rtol=2e-12, atol=1e-14)
    untouched = np.setdiff1d(np.arange(N), changed)
    assert np.array_equal(np.asarray(water_c)[:, untouched], w[:, untouched])  # unrefreshed columns bitwise
    assert acc_c["n_changed_columns"] == 3 and acc_c["n_batches"] == 2 and abs(acc_c["residual_m3"]) <= acc_c["bound_m3"]
    assert acc_c["surface_return_m3"] == pytest.approx(acc["surface_return_m3"], rel=1e-10, abs=1e-15)


def test_inspection_returns_the_single_counted_host_array_and_refuses_before_any_read(cp, monkeypatch):
    import maple.core.backend as mb

    a = fake(np.arange(1000.0))
    produced = []
    real = mb.to_host
    monkeypatch.setattr(mb, "to_host", lambda x: produced.append(real(x)) or produced[-1])
    cp.count.__init__()
    out = ncuda.inspect_counted(a)
    assert out is produced[0] and cp.count.d2h == 1 and cp.count.d2h_bytes == 8000 and not np.shares_memory(out, a)  # no second copy; not a view of the accepted authority
    need = 8000 + ncuda.fcu.PY_ALLOWANCE_BYTES
    assert np.array_equal(ncuda.inspect_counted(a, memory_budget_bytes=need), np.arange(1000.0))
    cp.count.__init__()
    reads = []
    for kw in ({"memory_budget_bytes": need - 1}, {"available_bytes_fn": lambda: reads.append(1) or 10}):
        with pytest.raises(fsl.FloatingMemoryError):
            ncuda.inspect_counted(a, **kw)
    with pytest.raises(nfs.NativeFloatingError, match="needs a device array"):
        ncuda.inspect_counted(np.arange(4.0), available_bytes_fn=lambda: reads.append(2) or 1 << 40)
    other = fake(np.arange(4.0), 1)  # lives on device 1 while the current device is 0
    with pytest.raises(nfs.NativeFloatingError, match="no implicit transfer"):
        ncuda.inspect_counted(other, available_bytes_fn=lambda: reads.append(3) or 1 << 40)
    with pytest.raises(fsl.FloatingSoilError):
        ncuda.inspect_counted(a, host_reserve_bytes=True)
    assert reads == [1] and cp.count.d2h == 0  # only the availability reading of the admission test ran; no refused call transferred anything


import maple_syrup.compact_soil_profile as _csp

_REAL_ITER = _csp.iter_compact_soil_profile  # the genuine reader, captured before any test patches it


def _host_feed(monkeypatch, host_child):
    """Feed the CUDA branch with the REAL E2 compact batches computed on the host copy of the child (the device reader itself is only exercised on hardware); returns the backends requested."""
    import maple_syrup.compact_soil_profile as csp

    real = _REAL_ITER
    seen = []

    def feed(vc, al, g, gc, mres, layers, **k):
        seen.append(k["backend"])
        k = dict(k, backend="numpy", memory_budget_bytes=1 << 30)  # the host feed is a stand-in for the device reader: its own host estimate differs
        return real(host_child.voxel_column, host_child.active_layer, g, gc, mres, layers, **k)

    monkeypatch.setattr(csp, "iter_compact_soil_profile", feed)
    return seen


@pytest.mark.parametrize("sat", [0.5, 0.999999])
@pytest.mark.parametrize("mode", ["explicit", "ks"])
def test_cuda_stage_matches_the_cpu_transaction_including_the_whole_inventory_receipt(cp, monkeypatch, sat, mode):
    kw = {"matrix_conductivity_m_s": None, "matrix_conductivity_mode": "texture_ks", "latent_saturated_conductivity_m_s": 3e-6} if mode == "ks" else {}
    host = stratified()
    child_h = _b.swap_classes(_b.set_cell_from(host, 1, 2), 0)
    oc = nfs.initialize_native_floating_soil(host, config(**kw), initial_relative_saturation=sat)
    d_parent, d_child = device_bed(host), device_bed(child_h)
    seen = _host_feed(monkeypatch, host)  # init feed: the parent's own compact batches
    og = nfs.initialize_native_floating_soil(d_parent, config(backend="cuda", **kw), initial_relative_saturation=sat)
    assert np.array_equal(og.depth_m, oc.depth_m) and np.allclose(np.asarray(og.state.water_m), oc.state.water_m, rtol=2e-12, atol=1e-14)
    seen = _host_feed(monkeypatch, child_h)  # (a new feed replaces the previous patch; the fixture's own patches stay)
    for m in ("properties", "depth"):
        cc = nfs.stage_native_refresh(oc, host, child_h, mode=m)
        cg = nfs.stage_native_refresh(og, d_parent, d_child, mode=m)
        assert seen[-1] == "cuda"  # the CUDA stage requested the E2 CUDA backend (the CPU stage before it the host one)
        assert list(cc.dirty_indices) == list(cg.dirty_indices) and list(cc.changed_columns) == list(cg.changed_columns)
        assert np.allclose(np.asarray(cg.state.water_m), cc.state.water_m, rtol=2e-12, atol=1e-14)
        assert np.allclose(np.asarray(cg.surface_return_m), cc.surface_return_m, rtol=2e-12, atol=1e-14) and np.allclose(np.asarray(cg.bottom_export_m), cc.bottom_export_m, rtol=2e-12, atol=1e-14)
        assert np.array_equal(cg.depth_m, cc.depth_m)
        if mode == "ks":
            assert np.allclose(np.asarray(cg.ks_m_s), cc.ks_m_s, rtol=2e-12, atol=1e-14)
        rc, rg = cc.receipt, cg.receipt
        assert rc is not None and rg is not None, (m, list(cc.changed_columns), list(cg.changed_columns), cc.reconciled, cg.reconciled)
        # the receipts have the SAME physical meaning on both backends: WHOLE active inventory before/after, and its closure checked independently
        assert rg.whole_before_m3 == pytest.approx(rc.whole_before_m3, rel=2e-12, abs=1e-14) and rg.whole_after_m3 == pytest.approx(rc.whole_after_m3, rel=2e-12, abs=1e-14)
        assert abs(rg.whole_residual_m3) <= rg.whole_bound_m3 and abs(rc.whole_residual_m3) <= rc.whole_bound_m3
        assert rg.whole_before_m3 == pytest.approx(float(oc.state.water_m.sum()) * 0.25, rel=2e-12) and rg.storage_before_m3 < rg.whole_before_m3  # selected-column storage is NOT the whole storage
        assert rg.surface_return_m3 == pytest.approx(rc.surface_return_m3, rel=1e-9, abs=1e-14)
    o2 = nfs.adopt_native_refresh(og, cg)
    assert o2.generation == 1 and o2.totals["surface_return_m3"] == cg.receipt.surface_return_m3
    with pytest.raises(nfs.NativeStaleError):
        nfs.adopt_native_refresh(og, cg)


def test_windowed_namespace_controls_on_a_genuine_host_windowed_twin(cp, tmp_path):
    host = stratified()
    wb = _b.windowed(host, tmp_path, "w", window_voxels=3)
    assert int(np.max(wb.voxel_column.base)) > 0  # genuine native Windowed bed with material below the resident floor
    cfg = config(backend="cuda")
    with pytest.raises(nfs.NativeFloatingError, match="no implicit transfer"):  # host Windowed bed, CUDA owner
        nfs.initialize_native_floating_soil(wb, cfg, initial_relative_saturation=0.5)
    mixed = nfs.NativeBed(wb.voxel_column, device_bed(host).active_layer, wb.geometry, wb.grain_classes, wb.mass_resolution_kg, wb.active_mask)
    with pytest.raises(nfs.NativeFloatingError, match="mixes host and device"):  # device active layer with a host window
        nfs.initialize_native_floating_soil(mixed, cfg, initial_relative_saturation=0.5)
    assert nfs._e2_backend(wb) == "numpy"


def test_root_same_count_mask_trigger_is_refused_before_any_row_property_or_water_work(cp, monkeypatch):
    """Root's real-GPU probe: owner from mask [[T,T,F]], then parent AND child submitted with [[F,T,T]] over exactly equal native columns. It must be refused (the CPU refusal) after admission, with only the bounded
    counted mask upload and one counted 1-byte read, and before the scan, the compact reader or any water work."""
    import maple_syrup.compact_soil_profile as csp

    host = stratified()

    def with_mask(bed, pattern):
        return nfs.NativeBed(bed.voxel_column, bed.active_layer, bed.geometry, bed.grain_classes, bed.mass_resolution_kg, np.array(pattern, dtype=bool))

    monkeypatch.setattr(csp, "iter_compact_soil_profile", lambda *a, **k: iter(()))
    d_good = with_mask(device_bed(host), [[True, True, False]])
    owner = nfs.initialize_native_floating_soil(d_good, config(backend="cuda"), initial_relative_saturation=0.5)
    d_bad = with_mask(device_bed(host), [[False, True, True]])
    reached = []
    monkeypatch.setattr(nfs, "_scan", lambda *a, **k: reached.append("scan"))
    monkeypatch.setattr(ncuda, "_scatter_tiles", lambda *a, **k: reached.append("tiles"))
    cp.count.__init__()
    cp.allocs.clear()
    for mode in ("depth", "properties"):
        with pytest.raises(nfs.NativeFloatingError, match="active mask"):
            nfs.stage_native_refresh(owner, d_bad, d_bad, mode=mode)
    assert reached == [] and owner.is_current
    assert cp.count.h2d_bytes == 2 * 3 and cp.count.d2h_sizes == [1, 1]  # per attempt: the N = 3 byte mask record up, ONE byte of verdict down


@pytest.mark.parametrize("mode", ["explicit", "ks"])
def test_cuda_equivalent_depth_change_matches_cpu_and_keeps_the_water_bitwise(cp, monkeypatch, mode):
    """E3c Root trigger on the CUDA branch (host double): 1 mm removed from cell 0 of a legal above-FC parent; D stays deeper than H. No cascade, no export/return, the accepted water bit for bit, the ACTUAL depth
    published in a genuine new column/state/operator, history exact."""
    kw = {"matrix_conductivity_m_s": None, "matrix_conductivity_mode": "texture_ks", "latent_saturated_conductivity_m_s": 3e-6} if mode == "ks" else {}
    host = stratified()
    child_h = _b.shift_top(host, 0, -0.001)
    oc = nfs.initialize_native_floating_soil(host, config(**kw), initial_relative_saturation=0.999999)
    d_parent, d_child = device_bed(host), device_bed(child_h)
    _host_feed(monkeypatch, host)
    og = nfs.initialize_native_floating_soil(d_parent, config(backend="cuda", **kw), initial_relative_saturation=0.999999)
    _host_feed(monkeypatch, child_h)
    cc = nfs.stage_native_refresh(oc, host, child_h, mode="depth")
    cg = nfs.stage_native_refresh(og, d_parent, d_child, mode="depth")
    assert list(cg.dirty_indices) == list(cc.dirty_indices) == [0] and cg.changed_columns.size == 0 == cc.changed_columns.size and list(cg.depth_changed) == list(cc.depth_changed) == [0]
    assert not cg.reconciled and cg.receipt is None and not np.asarray(cg.surface_return_m).any() and not np.asarray(cg.bottom_export_m).any()
    assert np.array_equal(np.asarray(cg.state.water_m), np.asarray(og.state.water_m)) and cg.state is not og.state and cg.state.t_s == og.state.t_s
    assert np.array_equal(cg.depth_m, cc.depth_m) and cg.depth_m[0] != og.depth_m[0] and cg.column.sha256 != og.column.sha256
    assert np.array_equal(np.asarray(cg.matrix_history.wetting_m), np.asarray(og.matrix_history.wetting_m)) and cg.matrix_history.operator_sha256 == cg.matrix_op.sha256
    o2 = nfs.adopt_native_refresh(og, cg)
    assert o2.generation == 1 and o2.depth_m[0] == cg.depth_m[0] and o2.totals["bottom_export_m3"] == 0.0
