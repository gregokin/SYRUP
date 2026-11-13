"""Actual-GPU test DEFINITIONS for the E3b CUDA native floating refresh and texture-Ks mode (WRITTEN, NOT RUN by their author: no GPU was launched or probed, no CuPy was imported, no NVRTC compile happened,
no nvidia-smi). Opt-in (SYRUP_RELEASE_GPU=1); Root runs them under its device guard with `.venv-cupy` and the genuine pinned native source (`E3_PINNED_MAPLE_SRC`, which makes a wrong source FAIL).
They compare the NumPy/Numba CPU transaction with the CUDA transaction on the SAME physical native fixtures under the existing 2e-12 / 1e-14 contracts: depth, water, FC/SAT/Ks maps (detached counted
inspection only), return/export, persistent history, dirty indices and the binding behaviours; the composed device/host admission and counted-transfer liveness; the second-device gate. Cold costs are honest:
the scan is a cold O(N V K) boundary read, not a per-step path; no full-storm or large-domain claim."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("e3_builders", Path(__file__).resolve().parents[1] / "phase2_native_floating_soil" / "builders.py")
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)

from maple_syrup import native_floating_soil as nfs
from maple_syrup import native_floating_soil_cuda as ncuda

RT, AT = 2e-12, 1e-14


@pytest.fixture(scope="module")
def cp():
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA native refresh is not exercised (no claim)")
    pin = os.environ.get("E3_PINNED_MAPLE_SRC")
    if pin:
        import maple

        assert os.path.realpath(maple.__file__).startswith(os.path.realpath(pin) + os.sep), "the imported MAPLE is not the configured pin"
    return backend.cupy_module()


def device_bed(cp, bed):
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    return nfs.NativeBed(VoxelColumnState(mass_kg=cp.asarray(bed.voxel_column.mass_kg)), ActiveLayerState(mass_kg=cp.asarray(bed.active_layer.mass_kg)), bed.geometry, bed.grain_classes,
                         bed.mass_resolution_kg, bed.active_mask)


def host(x):
    return ncuda.inspect_counted(x) if hasattr(x, "device") else np.asarray(x)


def cfgs(mode):
    kw = {"matrix_conductivity_m_s": None, "matrix_conductivity_mode": "texture_ks", "latent_saturated_conductivity_m_s": 3e-6} if mode == "ks" else {}
    return b.config(backend="numpy", **kw), b.config(backend="cuda", **kw)


def compare_cpu_cuda(cp, bed, dbed, child, dchild, mode, sat):
    """CPU (NumPy) transaction vs CUDA transaction on the SAME physical native fixture: depth, water, FC/SAT/Ks, return/export, receipts (WHOLE-inventory meaning), history, dirty/changed indices, bindings."""
    cpu_cfg, gpu_cfg = cfgs(mode)
    oc = nfs.initialize_native_floating_soil(bed, cpu_cfg, initial_relative_saturation=sat)
    og = nfs.initialize_native_floating_soil(dbed, gpu_cfg, initial_relative_saturation=sat)
    assert np.array_equal(oc.depth_m, og.depth_m)
    np.testing.assert_allclose(host(og.state.water_m), oc.state.water_m, rtol=RT, atol=AT)
    for m in ("properties", "depth"):
        cc = nfs.stage_native_refresh(oc, bed, child, mode=m)
        cg = nfs.stage_native_refresh(og, dbed, dchild, mode=m)
        assert list(cc.dirty_indices) == list(cg.dirty_indices) and list(cc.changed_columns) == list(cg.changed_columns)
        np.testing.assert_allclose(host(cg.state.water_m), cc.state.water_m, rtol=RT, atol=AT)
        np.testing.assert_allclose(host(cg.surface_return_m), cc.surface_return_m, rtol=RT, atol=AT)
        np.testing.assert_allclose(host(cg.bottom_export_m), cc.bottom_export_m, rtol=RT, atol=AT)
        np.testing.assert_allclose(host(cg.column._maps["theta_field_capacity"]), cc.column._maps["theta_field_capacity"], rtol=RT, atol=AT)
        np.testing.assert_allclose(host(cg.column._maps["theta_saturation"]), cc.column._maps["theta_saturation"], rtol=RT, atol=AT)
        assert np.array_equal(cg.depth_m, cc.depth_m)
        if mode == "ks":
            np.testing.assert_allclose(host(cg.ks_m_s), cc.ks_m_s, rtol=RT, atol=AT)
        assert np.array_equal(host(cg.matrix_history.wetting_m), cc.matrix_history.wetting_m) and np.array_equal(host(cg.matrix_history.cumulative_intake_m), cc.matrix_history.cumulative_intake_m)
        if cc.receipt is not None:
            rc, rg = cc.receipt, cg.receipt
            assert rg.whole_before_m3 == pytest.approx(rc.whole_before_m3, rel=RT, abs=AT) and rg.whole_after_m3 == pytest.approx(rc.whole_after_m3, rel=RT, abs=AT)
            assert abs(rg.whole_residual_m3) <= rg.whole_bound_m3 and abs(rc.whole_residual_m3) <= rc.whole_bound_m3  # closure checked independently of the state arrays
            assert rg.surface_return_m3 == pytest.approx(rc.surface_return_m3, rel=RT, abs=AT) and rg.bottom_export_m3 == pytest.approx(rc.bottom_export_m3, rel=RT, abs=AT)
    cg = nfs.stage_native_refresh(og, dbed, dchild, mode="properties")
    untouched = [c for c in range(3) if c not in set(cg.changed_columns.tolist())]
    assert np.array_equal(host(cg.state.water_m)[:, untouched], host(og.state.water_m)[:, untouched])  # unrefreshed columns bitwise, even above FC
    og2 = nfs.adopt_native_refresh(og, cg)
    for call in (lambda: nfs.adopt_native_refresh(og2, cg), lambda: nfs.adopt_native_refresh(og, cg), lambda: nfs.stage_native_refresh(og, dbed, dchild)):
        with pytest.raises(nfs.NativeStaleError):
            call()
    assert og2.totals["surface_return_m3"] == cg.receipt.surface_return_m3


@pytest.mark.parametrize("mode", ["explicit", "ks"])
@pytest.mark.parametrize("sat", [0.5, 0.999999])
def test_dense_cuda_matches_cpu_for_erosion_composition_and_untouched_above_fc(cp, mode, sat):
    bed = b.stratified()
    child = b.swap_classes(b.set_cell_from(bed, 1, 2), 0)  # erosion of cell 1 to rock + zero-net composition exchange of cell 0
    compare_cpu_cuda(cp, bed, device_bed(cp, bed), child, device_bed(cp, child), mode, sat)


def device_windowed(cp, host_dense, tmp_path, name):
    """A GENUINE device Windowed bed: MAPLE's native `windowed_column_from_dense` twin (resident window 3 of 4 voxels, so base > 0 and material lies below the resident floor) moved with `to_device_tree`
    (the exact construction of Root's E2 oracle), together with the device active layer."""
    from maple.core.backend import to_device_tree

    wb = b.windowed(host_dense, tmp_path, name, window_voxels=3)
    assert int(np.max(wb.voxel_column.base)) > 0, "the fixture must hold material below the resident floor"
    return wb, nfs.NativeBed(to_device_tree(wb.voxel_column, cp), to_device_tree(wb.active_layer, cp), wb.geometry, wb.grain_classes, wb.mass_resolution_kg, wb.active_mask)


@pytest.mark.parametrize("mode", ["explicit", "ks"])
def test_windowed_cuda_matches_cpu_windowed_and_dense_and_runs_the_full_native_validator(cp, tmp_path, monkeypatch, mode):
    from maple.surface.voxels import windowed_inventory as wi

    calls = []
    real = wi.validate_windowed_column_full
    monkeypatch.setattr(wi, "validate_windowed_column_full", lambda *a, **k: calls.append(1) or real(*a, **k))
    host_parent = b.stratified()
    host_child = b.swap_classes(b.set_cell_from(host_parent, 1, 2), 0)
    wp, dwp = device_windowed(cp, host_parent, tmp_path, "p")
    wc, dwc = device_windowed(cp, host_child, tmp_path, "c")
    assert nfs._e2_backend(dwp) == "numpy" and dwp.representation == "windowed"  # a Windowed bed is gathered/reduced on the host by MAPLE's own I/O; outputs are placed on the device by E2
    compare_cpu_cuda(cp, wp, dwp, wc, dwc, mode, 0.999999)
    assert len(calls) >= 6  # mandatory full validation: initialization (2) and every staging (parent and child), host and device owners
    # the dense CUDA owner and the Windowed CUDA owner agree on the same physical bed
    og_d = nfs.initialize_native_floating_soil(device_bed(cp, host_parent), b.config(backend="cuda"), initial_relative_saturation=0.5)
    og_w = nfs.initialize_native_floating_soil(dwp, b.config(backend="cuda"), initial_relative_saturation=0.5)
    assert np.array_equal(og_d.depth_m, og_w.depth_m) and np.array_equal(host(og_d.state.water_m), host(og_w.state.water_m))


def test_windowed_mixed_namespace_controls_are_typed_refusals(cp, tmp_path):
    host_parent = b.stratified()
    wp, dwp = device_windowed(cp, host_parent, tmp_path, "m")
    with pytest.raises(nfs.NativeFloatingError, match="no implicit transfer"):
        nfs.initialize_native_floating_soil(wp, b.config(backend="cuda"), initial_relative_saturation=0.5)  # host Windowed bed, CUDA owner
    with pytest.raises(nfs.NativeFloatingError, match="no implicit transfer"):
        nfs.initialize_native_floating_soil(dwp, b.config(), initial_relative_saturation=0.5)  # device Windowed bed, CPU owner
    mixed = nfs.NativeBed(dwp.voxel_column, wp.active_layer, wp.geometry, wp.grain_classes, wp.mass_resolution_kg, wp.active_mask)  # device window + host active layer
    with pytest.raises(nfs.NativeFloatingError, match="mixes host and device"):
        nfs.initialize_native_floating_soil(mixed, b.config(backend="cuda"), initial_relative_saturation=0.5)


def test_need_minus_one_refuses_before_any_transfer_and_wrong_device_is_gated_first(cp):
    from maple.core.backend import read_transfer_counters

    from maple_syrup import floating_soil_layers as fsl

    bed = b.stratified()
    dbed = device_bed(cp, bed)
    cfg = b.config(backend="cuda")
    model = ncuda.estimate_cuda_memory(None, dbed, cfg, batch_cells=64)  # the INITIALIZATION model (no parent)
    t0 = read_transfer_counters()
    with pytest.raises(fsl.FloatingMemoryError, match="nothing was allocated"):
        nfs.initialize_native_floating_soil(dbed, cfg, initial_relative_saturation=0.5, memory_budget_bytes=model["device_required_bytes"] - 1)
    with pytest.raises(fsl.FloatingMemoryError, match="nothing was allocated"):
        nfs.initialize_native_floating_soil(dbed, cfg, initial_relative_saturation=0.5, host_memory_budget_bytes=model["host_required_bytes"] - 1)
    d = read_transfer_counters().delta(t0)
    assert (d.device_to_host, d.host_to_device) == (0, 0)
    owner = nfs.initialize_native_floating_soil(dbed, cfg, initial_relative_saturation=0.5, memory_budget_bytes=model["device_required_bytes"], host_memory_budget_bytes=model["host_required_bytes"],
                                                available_bytes_fn=lambda: model["device_required_bytes"], host_available_bytes_fn=lambda: model["host_required_bytes"])  # exact budgets + truthful readings
    ref = ncuda.estimate_cuda_memory(dbed, dbed, cfg, batch_cells=64)  # the REFRESH model is a different (parent-aware) figure
    with pytest.raises(fsl.FloatingMemoryError, match="nothing was allocated"):
        nfs.stage_native_refresh(owner, dbed, dbed, memory_budget_bytes=ref["device_required_bytes"] - 1)
    if cp.cuda.runtime.getDeviceCount() >= 2:
        reads = []
        with cp.cuda.Device(1), pytest.raises(nfs.NativeFloatingError):
            nfs.initialize_native_floating_soil(dbed, cfg, initial_relative_saturation=0.5, available_bytes_fn=lambda: reads.append(1) or (1 << 40))
        assert not reads
    with pytest.raises(nfs.NativeFloatingError):  # host bed with a CUDA owner
        nfs.initialize_native_floating_soil(bed, cfg, initial_relative_saturation=0.5)


def test_inspection_lifetime_and_refusals_on_a_large_device_array(cp):
    from maple.core.backend import read_transfer_counters

    from maple_syrup import floating_soil_layers as fsl

    big = cp.arange(1 << 20, dtype=cp.float64)
    t0 = read_transfer_counters()
    out = ncuda.inspect_counted(big)
    d = read_transfer_counters().delta(t0)
    assert (d.device_to_host, d.device_to_host_bytes) == (1, big.nbytes) and out.nbytes == big.nbytes and not np.shares_memory(out, np.empty(0))
    with pytest.raises(fsl.FloatingMemoryError):
        ncuda.inspect_counted(big, memory_budget_bytes=big.nbytes)  # the shared admission's allowance is on top of the payload: refused
    with pytest.raises(nfs.NativeFloatingError):
        ncuda.inspect_counted(np.arange(4.0))


def test_component_control_reports_retained_pool_growth_and_counted_transfers_not_a_call_peak(cp):
    """N = 50003 (L = 3, B = 256) component control. It records (a) the retained device-pool growth after the call, (b) the counted MAPLE transfers, (c) the composed estimate. The retained growth is NOT a
    whole-call peak (transient tiles/scratch are released), so it is only compared with the composed payload as a sanity bound, and no peak claim is made."""
    from maple.core.backend import read_transfer_counters

    ny, nx = 1, 50003
    bed = b.bed_from(b.fx.stratified_dense(shallow_cells=(), ny=ny, nx=nx))
    dbed = device_bed(cp, bed)
    cfg = b.config(backend="cuda")
    pool = cp.get_default_memory_pool()
    model = ncuda.estimate_cuda_memory(None, dbed, cfg, batch_cells=256)
    pool.free_all_blocks()
    base = pool.used_bytes()
    t0 = read_transfer_counters()
    owner = nfs.initialize_native_floating_soil(dbed, cfg, initial_relative_saturation=0.5, batch_cells=256)
    d = read_transfer_counters().delta(t0)
    retained = pool.used_bytes() - base
    assert retained <= model["device_payload_bytes"]
    print({"retained_pool_growth": retained, "device_payload_model": model["device_payload_bytes"], "d2h_bytes": d.device_to_host_bytes, "d2h_calls": d.device_to_host, "h2d_bytes": d.host_to_device_bytes,
           "h2d_calls": d.host_to_device})
    del owner


def test_same_count_mask_probe_is_refused_on_hardware(cp):
    """Root's independent real-GPU probe (written, UNRUN): owner from [[T,T,F]]; parent and child submitted with [[F,T,T]] over equal native columns must be refused before any row/property/water work."""
    from maple.core.backend import read_transfer_counters

    host_bed = b.stratified()

    def masked(pattern):
        d = device_bed(cp, host_bed)
        return nfs.NativeBed(d.voxel_column, d.active_layer, d.geometry, d.grain_classes, d.mass_resolution_kg, np.array(pattern, dtype=bool))

    owner = nfs.initialize_native_floating_soil(masked([[True, True, False]]), b.config(backend="cuda"), initial_relative_saturation=0.5)
    bad = masked([[False, True, True]])
    t0 = read_transfer_counters()
    for mode in ("depth", "properties"):
        with pytest.raises(nfs.NativeFloatingError, match="active mask"):
            nfs.stage_native_refresh(owner, bad, bad, mode=mode)
    d = read_transfer_counters().delta(t0)
    assert d.host_to_device_bytes == 2 * 3 and d.device_to_host == 2 and d.device_to_host_bytes == 2
    assert owner.is_current


def _equivalent_depth_case(cp, bed, dbed, child, dchild, mode):
    cpu_cfg, gpu_cfg = cfgs(mode)
    oc = nfs.initialize_native_floating_soil(bed, cpu_cfg, initial_relative_saturation=0.999999)  # legal parent above field capacity
    og = nfs.initialize_native_floating_soil(dbed, gpu_cfg, initial_relative_saturation=0.999999)
    cc = nfs.stage_native_refresh(oc, bed, child, mode="depth")
    cg = nfs.stage_native_refresh(og, dbed, dchild, mode="depth")
    assert cg.changed_columns.size == 0 == cc.changed_columns.size and list(cg.depth_changed) == list(cc.depth_changed) == [0] and not cg.reconciled
    assert np.array_equal(host(cg.state.water_m), host(og.state.water_m)) and np.array_equal(host(cg.state.water_m), cc.state.water_m)  # bitwise: no extra drainage
    assert not host(cg.surface_return_m).any() and not host(cg.bottom_export_m).any() and np.array_equal(cg.depth_m, cc.depth_m)
    assert np.array_equal(host(cg.matrix_history.wetting_m), cc.matrix_history.wetting_m) and cg.column.sha256 != og.column.sha256
    o2 = nfs.adopt_native_refresh(og, cg)
    assert o2.depth_m[0] == cc.depth_m[0] and o2.totals["bottom_export_m3"] == 0.0


@pytest.mark.parametrize("mode", ["explicit", "ks"])
def test_equivalent_depth_change_dense_cuda_matches_cpu(cp, mode):
    bed = b.stratified()
    child = b.shift_top(bed, 0, -0.001)  # Root's E3c trigger: D 0.222 -> 0.221, both deeper than H
    _equivalent_depth_case(cp, bed, device_bed(cp, bed), child, device_bed(cp, child), mode)


def test_equivalent_depth_change_windowed_cuda_matches_cpu(cp, tmp_path):
    host_parent = b.stratified()
    host_child = b.shift_top(host_parent, 0, -0.001)
    wp, dwp = device_windowed(cp, host_parent, tmp_path, "ep")
    wc, dwc = device_windowed(cp, host_child, tmp_path, "ec")
    _equivalent_depth_case(cp, wp, dwp, wc, dwc, "explicit")
