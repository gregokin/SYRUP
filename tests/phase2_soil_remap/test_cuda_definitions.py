"""CUDA test DEFINITIONS (not run by their author: no GPU was launched or probed). Skipped without a usable device; Root runs them under its device guard.
They compare the device kernel with the NumPy reference bit for bit and exercise the device-side refusals."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import arrays, random_columns

from maple_syrup import soil_water_remap as sr


@pytest.fixture(scope="module")
def cp():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA remap kernel is not exercised (no claim)")
    return backend.cupy_module()


def device(cp, *a):
    return [cp.asarray(x) for x in a]


def call(cp, a, **kw):
    eo, en, W, up, lo = device(cp, *a)
    return sr.remap_intervals(eo, en, W, upper_theta=up, lower_theta=lo, cell_area_m2=2.0, backend="cuda", memory_budget_bytes=1 << 30, **kw)


@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_cuda_equals_numpy_bitwise(cp, L):
    a = arrays(*random_columns(np.random.default_rng(31 * L), L, 513))
    ref = sr.remap_intervals(*a[:3], upper_theta=a[3], lower_theta=a[4], cell_area_m2=2.0, backend="numpy", memory_budget_bytes=1 << 30)
    dev = call(cp, a)
    np.testing.assert_array_equal(cp.asnumpy(dev.water_m).view(np.uint64), ref.water_m.view(np.uint64))
    for name in ("removed_above_m", "removed_below_m", "added_above_m", "added_below_m"):
        np.testing.assert_array_equal(cp.asnumpy(getattr(dev, name)).view(np.uint64), getattr(ref, name).view(np.uint64))
    np.testing.assert_allclose(dev.closure["global_residual_m3"], ref.closure["global_residual_m3"], rtol=0, atol=ref.closure["global_bound_m3"])


def test_cuda_identity_and_empty(cp):
    eo = np.cumsum(np.random.default_rng(1).uniform(0.1, 1, (4, 9)), axis=0)
    W = (eo[1:] - eo[:-1]) * 0.5
    dev = call(cp, (eo, eo.copy(), W, np.zeros(9), np.zeros(9)))
    np.testing.assert_array_equal(cp.asnumpy(dev.water_m).view(np.uint64), W.view(np.uint64))
    assert not cp.asnumpy(dev.removed_above_m).any()
    z = np.zeros
    empty = call(cp, (z((3, 0)), z((3, 0)), z((2, 0)), z(0), z(0)))
    assert empty.water_m.shape == (2, 0)


def test_cuda_refusals_and_numpy_arrays_refused(cp):
    a = list(arrays(*random_columns(np.random.default_rng(2), 3, 12)))
    a[1][2, 5] = a[1][1, 5]
    with pytest.raises(sr.SoilRemapValidationError) as info:
        call(cp, a)
    assert info.value.bit == sr.FLAG_NEW_EDGES and info.value.first_column == 5
    good = arrays(*random_columns(np.random.default_rng(3), 3, 12))
    with pytest.raises(sr.SoilRemapError):  # host arrays on the cuda backend: no implicit transfer
        sr.remap_intervals(*good[:3], upper_theta=good[3], lower_theta=good[4], cell_area_m2=1.0, backend="cuda", memory_budget_bytes=1 << 30)
    with pytest.raises(sr.SoilRemapMemoryError):
        call(cp, good, available_bytes_fn=lambda: 0)


# ---- correction01 definitions (written, NOT run by their author; Root runs them on the assigned device) -----------------------------------------
def test_cuda_flag_diagnostics_match_the_host_classification(cp):
    """The device atomics (flag OR, count, first column with its flags) replace the unsupported bitwise-OR reduce: several different flags in different columns must be reported like the host."""
    C = 3001
    a = list(arrays(*random_columns(np.random.default_rng(11), 3, C)))
    a[1][2, 2900] = a[1][1, 2900]  # new edges (bit 2) late
    a[2][0, 1500] = -1.0  # water (bit 4) in the middle
    a[3][700] = 2.0  # theta (bit 8) early: the FIRST flagged column is 700
    a[0][1, 2000] = a[0][0, 2000]  # old edges (bit 1) later still
    with pytest.raises(sr.SoilRemapValidationError) as host:
        sr.remap_intervals(*a[:3], upper_theta=a[3], lower_theta=a[4], cell_area_m2=1.0, backend="numpy", memory_budget_bytes=1 << 30)
    with pytest.raises(sr.SoilRemapValidationError) as dev:
        call(cp, a)
    assert (dev.value.bit, dev.value.n_flagged, dev.value.first_column) == (host.value.bit, host.value.n_flagged, host.value.first_column) == (sr.FLAG_THETA, 4, 700)


@pytest.mark.parametrize("negative_zero_value", [-0.0, 5e-324])
def test_cuda_unchanged_geometry_preserves_signed_zero_and_subnormals(cp, negative_zero_value):
    e = np.cumsum(np.ones((3, 5)), axis=0) - 1.0
    W = np.full((2, 5), 0.5)
    W[1, :] = negative_zero_value
    dev = call(cp, (e, e.copy(), W, np.zeros(5), np.zeros(5)))
    np.testing.assert_array_equal(cp.asnumpy(dev.water_m).view(np.uint64), W.view(np.uint64))
    for f in (dev.removed_above_m, dev.removed_below_m, dev.added_above_m, dev.added_below_m):
        assert not np.any(cp.asnumpy(f).view(np.uint64))


def test_cuda_unrepresentable_scale_is_a_typed_closure_error(cp):
    eo = np.array([[-1e308], [0.0], [1e308]])
    with pytest.raises(sr.SoilRemapClosureError):
        call(cp, (eo, eo.copy(), np.zeros((2, 1)), np.zeros(1), np.zeros(1)))
    with pytest.raises(sr.SoilRemapClosureError):  # finite area whose global scale overflows
        eo1 = np.array([[0.0], [1.0]])
        eo_d, en_d, w_d, u_d, l_d = device(cp, eo1, eo1.copy(), np.full((1, 1), 0.5), np.zeros(1), np.zeros(1))
        sr.remap_intervals(eo_d, en_d, w_d, upper_theta=u_d, lower_theta=l_d, cell_area_m2=1e308, backend="cuda", memory_budget_bytes=1 << 30)


def test_cuda_host_and_device_are_admitted_separately_before_allocation(cp):
    a = arrays(*random_columns(np.random.default_rng(12), 2, 64))
    est = sr.estimate_remap_memory(2, 64, backend="cuda")
    with pytest.raises(sr.SoilRemapMemoryError, match="host side"):  # truthful injectable HOST reading too low, device reading fine
        call(cp, a, host_available_bytes_fn=lambda: est.host_required_bytes - 1)
    with pytest.raises(sr.SoilRemapMemoryError, match="host side"):
        call(cp, a, host_memory_budget_bytes=est.host_required_bytes - 1)
    with pytest.raises(sr.SoilRemapMemoryError, match="device side"):
        call(cp, a, available_bytes_fn=lambda: est.device_required_bytes - 1)
    call(cp, a, host_available_bytes_fn=lambda: est.host_required_bytes, available_bytes_fn=lambda: est.device_required_bytes)  # exactly enough on both sides is admitted


# ---- correction02 definitions (written, NOT run by their author; Root runs them on the assigned device) ------------------------------------------
def test_cuda_two_stage_reduction_matches_the_host_sums_over_many_blocks(cp):
    """C spans several column-kernel blocks AND more partials than the finalize block's threads, so both reduction stages and the strided finalize loop are exercised."""
    C = sr.CUDA_FINAL_BLOCK * sr.CUDA_BLOCK + 977  # n_blocks > CUDA_FINAL_BLOCK
    a = arrays(*random_columns(np.random.default_rng(21), 3, C))
    ref = sr.remap_intervals(*a[:3], upper_theta=a[3], lower_theta=a[4], cell_area_m2=2.0, backend="numpy", memory_budget_bytes=1 << 32)
    dev = call(cp, a)
    np.testing.assert_array_equal(cp.asnumpy(dev.water_m).view(np.uint64), ref.water_m.view(np.uint64))
    for key in ("total_old_m3", "total_new_m3", "added_m3", "removed_m3"):
        np.testing.assert_allclose(dev.closure[key], ref.closure[key], rtol=1e-12, atol=0)  # different (but fixed-tree) summation order from NumPy's pairwise sums
    assert dev.closure["max_abs_column_residual_m"] == ref.closure["max_abs_column_residual_m"] or abs(dev.closure["max_abs_column_residual_m"] - ref.closure["max_abs_column_residual_m"]) <= ref.closure["global_bound_m3"]
    assert abs(dev.closure["global_residual_m3"]) <= dev.closure["global_bound_m3"]


def test_cuda_reduction_is_deterministic(cp):
    a = arrays(*random_columns(np.random.default_rng(22), 4, 50_001))
    first, second = call(cp, a), call(cp, a)
    for key in ("total_old_m3", "total_new_m3", "added_m3", "removed_m3", "global_residual_m3", "max_abs_column_residual_m", "max_column_residual_over_bound"):
        assert first.closure[key] == second.closure[key]  # the fixed block tree has no atomic floating-point sums


def test_cuda_call_allocates_only_the_modelled_device_arrays(cp):
    """CuPy memory hook: the sum of the REQUESTED sizes of every device allocation made during the call must not exceed the modelled requirement (outputs + 10 diagnostic words + block partials)."""
    from cupy.cuda import memory_hook

    class Sizes(memory_hook.MemoryHook):
        name = "remap_sizes"

        def __init__(self):
            self.sizes = []

        def malloc_postprocess(self, **kwargs):
            self.sizes.append(int(kwargs["size"]))

    a = arrays(*random_columns(np.random.default_rng(23), 3, 4099))
    eo, en, W, up, lo = device(cp, *a)
    est = sr.estimate_remap_memory(3, 4099, backend="cuda")
    hook = Sizes()
    with hook:
        res = sr.remap_intervals(eo, en, W, upper_theta=up, lower_theta=lo, cell_area_m2=1.0, backend="cuda", memory_budget_bytes=1 << 30)
    assert res.n_columns == 4099
    assert sum(hook.sizes) <= est.device_required_bytes - est.reserve_bytes, (sum(hook.sizes), est.device_required_bytes, hook.sizes)


def test_cuda_raw_integer_diagnostics_exact_for_a_late_column(cp):
    C = 1_000_003
    a = list(arrays(*random_columns(np.random.default_rng(24), 2, C)))
    late = C - 2
    a[2][0, late] = -1.0
    with pytest.raises(sr.SoilRemapValidationError) as info:
        call(cp, a)
    assert (info.value.first_column, info.value.n_flagged, info.value.bit) == (late, 1, sr.FLAG_WATER)
