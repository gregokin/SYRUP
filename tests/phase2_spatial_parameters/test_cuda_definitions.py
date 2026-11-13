"""CUDA hardware test DEFINITIONS (written, NOT run by their author: no GPU was launched or probed). Skipped without a usable device; Root runs them under its device guard.
They compare the device preparation with the NumPy reference, check the device-side refusals, the exact packed first cell, the separate host/device admission and the allocation made by the call."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import make_spec, params

from maple_syrup import spatial_soil_layers as sp


@pytest.fixture(scope="module")
def cp():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA spatial-parameter preparation is not exercised (no claim)")
    return backend.cupy_module()


def to_dev(cp, p):
    return {k: cp.asarray(v) for k, v in p.items()}


def dev_spec(cp, p, mode="constant", **kw):
    d = to_dev(cp, p)
    n = p["edges"].shape[1]
    extra = dict(diffusivity_m2_s=d["iface"]) if mode == "constant" else dict(interface_conductivity_m_s=d["iface"], diffusion_coefficient=0.3)
    return sp.spatial_layer_spec(d["edges"], theta_residual=d["res"], theta_field_capacity=d["fc"], theta_saturation=d["sat"], theta_evaporation_stress=0.5, diffusion_mode=mode, n_cells=n, **extra, **kw)


def prep(cp, spec, mask=None, **kw):
    """`mask` is the CALLER's device mask; a test that measures allocations must create it BEFORE its memory hook starts and pass it here."""
    mask = cp.ones(spec.n_cells, dtype=cp.bool_) if mask is None else mask
    return sp.prepare_spatial_column(spec, mask, 4.0, backend="cuda", memory_budget_bytes=1 << 33, **kw)


@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_device_maps_equal_the_numpy_reference_bitwise(cp, mode, L):
    p = params(np.random.default_rng(40 + L), L, 1001, mode=mode)
    dev = prep(cp, dev_spec(cp, p, mode))
    ref = sp.prepare_spatial_column(make_spec(p, mode=mode), np.ones(1001, dtype=bool), 4.0, backend="numpy", memory_budget_bytes=1 << 33)
    assert dev.n_active == 1001 and dev.parameters_sha256 == ref.parameters_sha256 and dev.mask_sha256 == ref.mask_sha256
    for name in ref.retained_names():
        got = dev.inspect(name, host_memory_budget_bytes=1 << 33)
        np.testing.assert_array_equal(got.view(np.uint64), ref.inspect(name, host_memory_budget_bytes=1 << 33).view(np.uint64), err_msg=name)


def test_device_mixed_layouts_and_scalars(cp):
    L, N = 3, 500
    p = params(np.random.default_rng(1), L, N)
    d = cp.asarray(p["edges"][:, 0].copy())
    spec = sp.spatial_layer_spec(d, theta_residual=0.02, theta_field_capacity=cp.asarray(p["fc"]), theta_saturation=0.9, theta_evaporation_stress=0.5, diffusivity_m2_s=2e-7, n_cells=N)
    col = prep(cp, spec)
    assert col.summary()["input_layouts"]["depth_edges_m"] == "shared" and col.n_cells == N


def test_device_refusals_report_the_exact_first_cell_and_flags(cp):
    N = 100_001
    p = params(np.random.default_rng(5), 3, N)
    late = N - 2
    p["res"][0, late] = -1.0
    p["edges"][2, 70_000] = p["edges"][1, 70_000]
    p["fc"][1, 40_000] = np.nan
    with pytest.raises(sp.SpatialLayerError) as info:
        prep(cp, dev_spec(cp, p))
    assert (info.value.first_cell, info.value.n_flagged, info.value.bit) == (40_000, 3, sp.FLAG_THETA)


def test_device_derived_flag_and_stress_flag(cp):
    p = params(np.random.default_rng(6), 3, 64)
    p["edges"][:, 9] = np.array([0.0, 1e-320, 2e-320, 3e-320])
    with pytest.raises(sp.SpatialLayerError) as info:
        prep(cp, dev_spec(cp, p))
    assert info.value.bit == sp.FLAG_DERIVED and info.value.first_cell == 9


def test_host_arrays_on_the_cuda_backend_are_refused_and_host_device_admission_is_separate(cp):
    p = params(np.random.default_rng(7), 2, 32)
    with pytest.raises(sp.SpatialLayerError):  # a numpy spec on the cuda backend: no implicit transfer
        sp.prepare_spatial_column(make_spec(p), cp.ones(32, dtype=cp.bool_), 1.0, backend="cuda", memory_budget_bytes=1 << 30)
    spec = dev_spec(cp, p)
    est = sp.estimate_spatial_memory(2, 32, layouts=spec.layouts(), backend="cuda")
    with pytest.raises(sp.SpatialLayerMemoryError, match="host side"):
        prep(cp, spec, host_available_bytes_fn=lambda: est.host_required_bytes - 1)
    with pytest.raises(sp.SpatialLayerMemoryError, match="host side"):
        prep(cp, spec, host_memory_budget_bytes=est.host_required_bytes - 1)
    with pytest.raises(sp.SpatialLayerMemoryError, match="device side"):
        prep(cp, spec, available_bytes_fn=lambda: est.device_required_bytes - 1)
    prep(cp, spec, host_available_bytes_fn=lambda: est.host_required_bytes, available_bytes_fn=lambda: est.device_required_bytes)  # exactly enough on both sides


def test_device_call_allocates_only_the_modelled_arrays(cp):
    """CuPy memory hook: the REQUESTED device bytes made during `prepare_spatial_column` must not exceed the modelled device requirement (retained maps + mask + diagnostic + auxiliary)."""
    from cupy.cuda import memory_hook

    class Sizes(memory_hook.MemoryHook):
        name = "spatial_prepare_sizes"

        def __init__(self):
            self.sizes = []

        def malloc_postprocess(self, **kwargs):
            self.sizes.append(int(kwargs["size"]))

    p = params(np.random.default_rng(8), 3, 4099)
    spec = dev_spec(cp, p)
    est = sp.estimate_spatial_memory(3, 4099, layouts=spec.layouts(), backend="cuda")
    mask = cp.ones(4099, dtype=cp.bool_)  # the CALLER's mask is created BEFORE the hook: it is not an allocation of the call (the estimator is not inflated to cover it)
    hook = Sizes()
    with hook:
        prep(cp, spec, mask=mask)
    assert sum(hook.sizes) <= est.device_required_bytes, (sum(hook.sizes), est.device_required_bytes, hook.sizes)


def test_device_inspection_is_admitted_and_detached(cp):
    col = prep(cp, dev_spec(cp, params(np.random.default_rng(9), 2, 64)))
    need = int(col._maps["dz"].nbytes)
    with pytest.raises(sp.SpatialLayerMemoryError):
        col.inspect("dz", host_memory_budget_bytes=need - 1)
    a = col.inspect("dz", host_memory_budget_bytes=need)
    a[...] = -1.0
    assert (col.inspect("dz", host_memory_budget_bytes=need) > 0).all()


def test_device_identity_equals_the_cpu_identity_of_the_same_inputs_for_every_hash_chunk(cp):
    p = params(np.random.default_rng(10), 3, 777)
    ref = sp.prepare_spatial_column(make_spec(p), np.ones(777, dtype=bool), 4.0, backend="numpy", memory_budget_bytes=1 << 33)
    for hc in (8, 4096, 1 << 20):
        assert prep(cp, dev_spec(cp, p), hash_chunk_bytes=hc).sha256 == ref.sha256
