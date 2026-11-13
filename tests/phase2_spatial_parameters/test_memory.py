"""Memory model, pre-allocation admission against quotas/readers, warm peak against the modelled requirement, retained-map counts. Author-written (CPU)."""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from conftest import BIG, make_spec, params, prepare

from maple_syrup import spatial_soil_layers as sp


@pytest.mark.parametrize("mode,per_cell", [("constant", lambda L: 4 * L + 2 * (L - 1)), ("tietjen_donor_depth", lambda L: 4 * L + 3 * (L - 1))])
@pytest.mark.parametrize("L", [2, 3, 5, 8])
def test_retained_doubles_per_cell_for_all_cell_layouts(mode, per_cell, L):
    N = 1000
    layouts = {"depth_edges_m": "cell", "theta_residual": "cell", "theta_field_capacity": "cell", "theta_saturation": "cell",
               "diffusivity_m2_s" if mode == "constant" else "interface_conductivity_m_s": "cell"}
    est = sp.estimate_spatial_memory(L, N, layouts=layouts, diffusion_mode=mode, backend="numpy")
    assert est.retained_bytes == 8 * N * per_cell(L)
    assert len(est.retained) == (6 if mode == "constant" else 7)  # the ONLY retained arrays: 3 thetas, dz, pair_inv, G_down (+ G_up in donor mode)
    assert est.mask_bytes == N


def test_shared_and_scalar_layouts_do_not_scale_with_n_and_edges_drive_pair_and_dz():
    layouts = {"depth_edges_m": "shared", "theta_residual": "scalar", "theta_field_capacity": "shared", "theta_saturation": "scalar", "diffusivity_m2_s": "shared"}
    a, b = sp.estimate_spatial_memory(4, 10, layouts=layouts), sp.estimate_spatial_memory(4, 10_000_000, layouts=layouts)
    assert a.retained_bytes == b.retained_bytes == 8 * (1 + 4 + 1 + 4 + 3 + 3)  # res, fc, sat, dz, pair, G
    mixed = dict(layouts, diffusivity_m2_s="cell")
    m = sp.estimate_spatial_memory(4, 100, layouts=mixed)
    assert m.retained["pair_inverse_thickness"]["kind"] == "shared" and m.retained["interface_conductance_down"]["kind"] == "cell" and m.retained["dz"]["kind"] == "shared"


def test_cuda_sides_are_modelled_separately_and_have_no_workspace_terms():
    layouts = {"depth_edges_m": "cell", "theta_residual": "cell", "theta_field_capacity": "cell", "theta_saturation": "cell", "diffusivity_m2_s": "cell"}
    a = sp.estimate_spatial_memory(3, 1000, layouts=layouts, backend="cuda", reserve_bytes=7, host_reserve_bytes=11, hash_chunk_bytes=4096)
    b = sp.estimate_spatial_memory(3, 100_000, layouts=layouts, backend="cuda", reserve_bytes=7, host_reserve_bytes=11, hash_chunk_bytes=4096)
    assert a.host_required_bytes == b.host_required_bytes == 4096 + sp.PACKET_BYTES + sp.PY_ALLOWANCE_BYTES + 11  # host side independent of N: identity buffer + packet + allowance + host reserve
    assert a.device_required_bytes == a.retained_bytes + a.mask_bytes + sp.PACKET_BYTES + sp.AUX_BYTES + 7 and a.main_required_bytes == a.device_required_bytes
    assert a.terms["cuda_kernel_temporaries"] == 0 and a.terms["no_full_host_mirror"] and "no workspace" in a.terms["excluded"]
    assert sp.estimate_spatial_memory(3, 1000, layouts=layouts, backend="numpy").device_required_bytes == 0


def test_cuda_cell_limit_gate_is_explicit():
    layouts = {"depth_edges_m": "shared", "theta_residual": "scalar", "theta_field_capacity": "scalar", "theta_saturation": "scalar", "diffusivity_m2_s": "scalar"}
    sp.estimate_spatial_memory(1, sp.MAX_CUDA_CELLS, layouts=layouts, backend="cuda")
    with pytest.raises(sp.SpatialLayerError):
        sp.estimate_spatial_memory(1, sp.MAX_CUDA_CELLS + 1, layouts=layouts, backend="cuda")
    assert (sp.MAX_CUDA_CELLS << 8) < 2**64
    sp.estimate_spatial_memory(1, sp.MAX_CUDA_CELLS + 1, layouts=layouts, backend="numpy")


@pytest.mark.parametrize("bad", [dict(layouts={}), dict(layouts={"depth_edges_m": "scalar", "theta_residual": "cell", "theta_field_capacity": "cell", "theta_saturation": "cell", "diffusivity_m2_s": "cell"}),
                                 dict(backend="cupy"), dict(chunk_columns=0), dict(hash_chunk_bytes=7), dict(host_reserve_bytes=1)])
def test_estimator_refusals(bad):
    layouts = {"depth_edges_m": "cell", "theta_residual": "cell", "theta_field_capacity": "cell", "theta_saturation": "cell", "diffusivity_m2_s": "cell"}
    kw = dict(layouts=layouts, backend="numpy")
    kw.update(bad)
    with pytest.raises(sp.SpatialLayerError):
        sp.estimate_spatial_memory(3, 10, **kw)


def test_quotas_reserves_and_readers_are_enforced_before_allocation(cpu_backend):
    p = params(np.random.default_rng(3), 3, 500)
    spec = make_spec(p)
    need = sp.estimate_spatial_memory(3, 500, layouts=spec.layouts(), backend=cpu_backend, input_bytes=spec.input_bytes()).main_required_bytes
    tracemalloc.start()
    with pytest.raises(sp.SpatialLayerMemoryError):
        prepare(spec, backend=cpu_backend, budget=need - 1)
    with pytest.raises(sp.SpatialLayerMemoryError):
        prepare(spec, backend=cpu_backend, budget=need + 10, reserve_bytes=11)  # the reserve is inside the requirement, counted once
    with pytest.raises(sp.SpatialLayerMemoryError):
        prepare(spec, backend=cpu_backend, available_bytes_fn=lambda: need - 1)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 200_000 < need * 10  # nothing of the preparation was allocated by the three refusals (only tiny objects)
    col = prepare(spec, backend=cpu_backend, budget=need, available_bytes_fn=lambda: need)  # exactly enough is admitted
    assert col.n_cells == 500


def test_caller_input_bytes_are_recorded_and_excluded(cpu_backend):
    p = params(np.random.default_rng(4), 3, 64)
    spec = make_spec(p)
    col = prepare(spec, backend=cpu_backend)
    want = {"depth_edges_m": p["edges"].nbytes, "theta_residual": p["res"].nbytes, "theta_field_capacity": p["fc"].nbytes, "theta_saturation": p["sat"].nbytes, "diffusivity_m2_s": p["iface"].nbytes}
    assert col.estimate.input_bytes == want
    assert col.estimate.main_required_bytes < sum(want.values()) + col.estimate.retained_bytes + 10**6  # the requirement does not include the inputs


@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("chunk", [None, 997])
@pytest.mark.parametrize("failing", [False, True])
def test_warm_peak_is_within_the_modelled_requirement(cpu_backend, mode, mixed, chunk, failing):
    L, N = 3, 60_000
    p = params(np.random.default_rng(5), L, N, mode=mode)
    warm = params(np.random.default_rng(6), L, 8, mode=mode)
    prepare(make_spec(warm, mode=mode), backend=cpu_backend)  # warm: imports and first-call costs outside the measurement (the caller's inputs already exist)
    if mixed:
        spec = make_spec({"edges": p["edges"][:, 0].copy(), "res": 0.02, "fc": p["fc"], "sat": p["sat"], "iface": p["iface"]}, mode=mode, N=N, iface=p["iface"])
    else:
        spec = make_spec(p, mode=mode)
    if failing:
        p["sat"][1, N - 3] = np.nan  # the LAST chunk fails: all earlier chunk work and the retained maps were already allocated
        spec = make_spec(p, mode=mode)
    est = sp.estimate_spatial_memory(L, N, layouts=spec.layouts(), diffusion_mode=mode, backend=cpu_backend, chunk_columns=chunk or sp.DEFAULT_CHUNK_COLUMNS)
    mask = np.ones(N, dtype=bool)
    tracemalloc.start()
    try:
        sp.prepare_spatial_column(spec, mask, 1.0, backend=cpu_backend, memory_budget_bytes=BIG, chunk_columns=chunk)
        assert not failing
    except sp.SpatialLayerError:
        assert failing
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= est.main_required_bytes, (peak, est.main_required_bytes, est.terms)


def test_work_vectors_scale_with_the_chunk_not_with_n_or_l():
    a = sp.estimate_spatial_memory(3, 10_000_000, layouts={"depth_edges_m": "cell", "theta_residual": "cell", "theta_field_capacity": "cell", "theta_saturation": "cell",
                                                           "diffusivity_m2_s": "cell"}, chunk_columns=1000)
    b = sp.estimate_spatial_memory(8, 10_000_000, layouts={"depth_edges_m": "cell", "theta_residual": "cell", "theta_field_capacity": "cell", "theta_saturation": "cell",
                                                           "diffusivity_m2_s": "cell"}, chunk_columns=1000)
    assert a.terms["numpy_chunk_work_vectors"] == b.terms["numpy_chunk_work_vectors"] == (sp.REF_VEC_F64 * 8 + sp.REF_VEC_BOOL + sp.REF_VEC_U8) * 1000
