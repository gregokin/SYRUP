"""CUDA RawKernel backend against the NumPy reference. Every device test uses the `gpu` fixture and skips honestly
without CuPy/a device; the one setup-error test runs ONLY when no GPU is available. Declared bound rtol 2e-12 /
atol 1e-14 (device expm1); bitwise equality where the diffusivity is 0. Nothing here was run by its author; Codex
records results."""
from __future__ import annotations

import numpy as np
import pytest
from column_cases import AREA, layers, random_case

from maple_syrup.subsurface import (
    SubsurfaceBackendUnavailableError,
    SubsurfaceError,
    SubsurfaceMemoryError,
    SubsurfaceStepError,
    column_state,
    prepare_column,
    step,
    uniform_state,
)

RTOL, ATOL = 2e-12, 1e-14


def host(a):
    return np.asarray(a.get()) if hasattr(a, "get") else np.asarray(a)


def columns(gpu, lay, mask):
    ref = prepare_column(lay, mask, AREA, backend="numpy", reserve_bytes=0, available_bytes_fn=lambda: 1 << 40)
    dev = prepare_column(lay, gpu.asarray(mask), AREA, backend="cuda", reserve_bytes=0, available_bytes_fn=lambda: 1 << 40)
    assert dev.backend == "cuda" and dev.xp is gpu and dev.device_id == int(gpu.cuda.Device().id)
    return ref, dev


def arrays_of(result):
    return {"water": result.state.water_m, "return": result.surface_return_m, "drainage": result.drainage_m,
            "evaporation": result.evaporation_m, "unmet": result.evaporation_unmet_m,
            "percolation": result.percolation_m, "diffusion": result.diffusion_m}


@pytest.mark.parametrize("gravity", [True, False])
@pytest.mark.parametrize("n_layers", [1, 2, 5])
def test_cuda_matches_reference_within_declared_bound(gpu, n_layers, gravity):
    lay, mask, water, add, pet, dt = random_case(40 + n_layers, n_layers, 301, gravity=gravity, diffusive=True)
    ref, dev = columns(gpu, lay, mask)
    r = step(ref, column_state(ref, water), dt, added_water_m=add, potential_evaporation_m_s=pet)
    d = step(dev, column_state(dev, gpu.asarray(water)), dt, added_water_m=gpu.asarray(add),
             potential_evaporation_m_s=gpu.asarray(pet))
    for name, a in arrays_of(r).items():
        got = arrays_of(d)[name]
        assert type(got) is gpu.ndarray
        np.testing.assert_allclose(host(got), a, rtol=RTOL, atol=ATOL, err_msg=name)
    for field in ("storage_before_m3", "storage_after_m3", "added_m3", "surface_return_m3", "drainage_m3",
                  "evaporation_m3", "evaporation_unmet_m3"):
        assert getattr(d.budget, field) == pytest.approx(getattr(r.budget, field), rel=RTOL, abs=ATOL)
    assert abs(d.budget.residual_m3) <= d.budget.bound_m3


@pytest.mark.parametrize("n_cells", [1, 127, 128, 129, 257])
def test_cuda_is_bitwise_equal_without_diffusion_at_block_boundaries(gpu, n_cells):
    lay, mask, water, add, pet, dt = random_case(60 + n_cells, 3, n_cells, gravity=True, diffusive=False)
    ref, dev = columns(gpu, lay, mask)
    r = step(ref, column_state(ref, water), dt, added_water_m=add, potential_evaporation_m_s=pet)
    d = step(dev, column_state(dev, gpu.asarray(water)), dt, added_water_m=gpu.asarray(add),
             potential_evaporation_m_s=gpu.asarray(pet))
    for name, a in arrays_of(r).items():
        assert host(arrays_of(d)[name]).tobytes() == a.tobytes(), name


def test_cuda_one_host_read_and_no_upload_per_step(gpu):
    from maple.core.backend import read_transfer_counters

    lay, mask, water, add, pet, dt = random_case(2, 3, 500, gravity=True, diffusive=True)
    _, dev = columns(gpu, lay, mask)
    state = column_state(dev, gpu.asarray(water))
    add_d, pet_d = gpu.asarray(add), gpu.asarray(pet)
    before = read_transfer_counters()
    result = step(dev, state, dt, added_water_m=add_d, potential_evaporation_m_s=pet_d)
    delta = read_transfer_counters().delta(before)
    assert delta.host_to_device == 0 and delta.device_to_host == 1
    assert type(result.state.water_m) is gpu.ndarray and int(result.state.water_m.device.id) == dev.device_id


def test_cuda_failure_bits_match_reference(gpu):
    lay, mask, water, add, pet, dt = random_case(9, 2, 40, gravity=True, diffusive=True)
    water = water.copy()
    water[1, 5] = -1e-3
    pet = pet.copy()
    pet[0] = np.nan
    ref, dev = columns(gpu, lay, mask)
    bits = []
    for col, w, a, p in ((ref, water, add, pet), (dev, gpu.asarray(water), gpu.asarray(add), gpu.asarray(pet))):
        with pytest.raises(SubsurfaceStepError) as info:
            step(col, column_state(col, w), dt, added_water_m=a, potential_evaporation_m_s=p)
        bits.append((info.value.bit, info.value.n_flagged))
    assert bits[0] == bits[1] == (1, 2)


def test_cuda_dt_zero_identity(gpu):
    lay, mask, water, add, pet, _ = random_case(12, 4, 33, gravity=True, diffusive=True)
    _, dev = columns(gpu, lay, mask)
    state = column_state(dev, gpu.asarray(water))
    result = step(dev, state, 0.0, added_water_m=gpu.zeros_like(state.water_m), potential_evaporation_m_s=gpu.asarray(pet))
    assert host(result.state.water_m).tobytes() == water.tobytes()
    with pytest.raises(SubsurfaceStepError) as info:
        step(dev, state, 0.0, added_water_m=gpu.asarray(add), potential_evaporation_m_s=gpu.asarray(pet))
    assert info.value.bit == 9


def test_namespace_mismatches_are_refused_without_transfer(gpu):
    mask = np.array([True, True])
    ref, dev = columns(gpu, layers(), mask)
    with pytest.raises(SubsurfaceError):
        column_state(dev, np.zeros((2, 2)))  # host array on a device column
    with pytest.raises(SubsurfaceError):
        column_state(ref, gpu.zeros((2, 2)))  # device array on a host column
    with pytest.raises(SubsurfaceError):
        prepare_column(layers(), mask, AREA, backend="cuda", reserve_bytes=0, available_bytes_fn=lambda: 1)
    state = uniform_state(dev, [0.25, 0.4])
    assert type(state.water_m) is gpu.ndarray
    with pytest.raises(SubsurfaceError):
        step(dev, state, 1.0, added_water_m=np.zeros((2, 2)), potential_evaporation_m_s=gpu.zeros(2))
    with pytest.raises(SubsurfaceError):
        step(dev, state, 1.0, added_water_m=gpu.zeros((2, 2), dtype=np.float32), potential_evaporation_m_s=gpu.zeros(2))


def test_cuda_default_available_bytes_is_a_positive_int(gpu):
    dev = prepare_column(layers(), gpu.ones(4, dtype=bool), AREA, backend="cuda", reserve_bytes=0)
    value = dev.available_bytes_fn()
    assert isinstance(value, int) and value > 0
    state = uniform_state(dev, [0.25, 0.4])
    result = step(dev, state, 10.0, added_water_m=gpu.zeros((2, 4)), potential_evaporation_m_s=gpu.zeros(4))
    assert abs(result.budget.residual_m3) <= result.budget.bound_m3
    assert result.budget.drainage_m3 > 0.0  # theta 0.4 in layer 1 sits above field capacity 0.25


DT_SEQUENCE = (0.0, 600.0, 0.5, 86400.0, 1e6, 0.0, 3600.0)


@pytest.mark.parametrize("n_layers", [1, 2, 40])
def test_cuda_varying_dt_matches_reference_without_uploads(gpu, n_layers):
    """Successive steps with different dt on ONE device column agree with the reference, upload nothing (the factor
    is prepared on device), read one packet per step, and are bit-identical to a FRESH device column stepped once."""
    from maple.core.backend import read_transfer_counters

    lay, mask, water, add, pet, _ = random_case(120 + n_layers, n_layers, 300, gravity=True, diffusive=True)
    ref, dev = columns(gpu, lay, mask)
    state_ref, state_dev = column_state(ref, water), column_state(dev, gpu.asarray(water))
    add_d, pet_d, zero_d = gpu.asarray(add), gpu.asarray(pet), gpu.zeros_like(gpu.asarray(add))
    for dt in DT_SEQUENCE:
        forcing_h = np.zeros_like(add) if dt == 0.0 else add
        forcing_d = zero_d if dt == 0.0 else add_d
        r = step(ref, state_ref, dt, added_water_m=forcing_h, potential_evaporation_m_s=pet)
        before = read_transfer_counters()
        d = step(dev, state_dev, dt, added_water_m=forcing_d, potential_evaporation_m_s=pet_d)
        delta = read_transfer_counters().delta(before)
        assert delta.host_to_device == 0 and delta.device_to_host == 1
        for name, a in arrays_of(r).items():
            np.testing.assert_allclose(host(arrays_of(d)[name]), a, rtol=RTOL, atol=ATOL, err_msg=f"dt={dt} {name}")
        fresh = prepare_column(lay, gpu.asarray(mask), AREA, backend="cuda", reserve_bytes=0,
                               available_bytes_fn=lambda: 1 << 40)
        f = step(fresh, column_state(fresh, state_dev.water_m), dt, added_water_m=forcing_d, potential_evaporation_m_s=pet_d)
        for name, a in arrays_of(d).items():
            assert host(arrays_of(f)[name]).tobytes() == host(a).tobytes(), f"stale factor? dt={dt} {name}"
        state_ref, state_dev = r.state, d.state
    assert state_dev.t_s == state_ref.t_s == sum(DT_SEQUENCE)
    assert not hasattr(d, "scratch")  # the published step never carries the device scratch


def test_cuda_streaming_matches_reference_streaming(gpu):
    from maple_syrup.subsurface import ForcingRecord, initialize_profile

    lay, mask, water, add, pet, _ = random_case(88, 4, 257, gravity=True, diffusive=True)
    ref, dev = columns(gpu, lay, mask)
    add_d, pet_d = gpu.asarray(add), gpu.asarray(pet)
    records_h = [ForcingRecord(dt, add if dt else np.zeros_like(add), pet) for dt in DT_SEQUENCE]
    records_d = [ForcingRecord(dt, add_d if dt else gpu.zeros_like(add_d), pet_d) for dt in DT_SEQUENCE]
    pr = initialize_profile(ref, column_state(ref, water), records_h)
    pd = initialize_profile(dev, column_state(dev, gpu.asarray(water)), records_d)
    np.testing.assert_allclose(host(pd.state.water_m), pr.state.water_m, rtol=RTOL, atol=ATOL)
    assert pd.cumulative.n_steps == len(DT_SEQUENCE) and pd.cumulative.elapsed_s == pr.cumulative.elapsed_s
    for name in ("added_m3", "surface_return_m3", "drainage_m3", "evaporation_m3", "evaporation_unmet_m3"):
        assert getattr(pd.cumulative, name) == pytest.approx(getattr(pr.cumulative, name), rel=RTOL, abs=ATOL)
    assert abs(pd.residual_m3) <= pd.bound_m3


def test_cuda_provenance_declares_two_kernels_and_the_factor_scratch(gpu):
    from maple_syrup import subsurface_cuda

    dev = prepare_column(layers(), gpu.ones(4, dtype=bool), AREA, backend="cuda", reserve_bytes=0,
                         available_bytes_fn=lambda: 1 << 40)
    info = dev.backend_info
    assert info["launches_per_step"] == subsurface_cuda.LAUNCHES_PER_STEP == 2
    assert info["factor_scratch_device_bytes"] == 8
    assert set(info["provenance"]["kernels"]) == {subsurface_cuda.FACTOR_KERNEL_NAME, subsurface_cuda.KERNEL_NAME}
    one = prepare_column(layers(edges=(0.0, 0.25)), gpu.ones(4, dtype=bool), AREA, backend="cuda", reserve_bytes=0,
                         available_bytes_fn=lambda: 1 << 40)
    assert one.backend_info["launches_per_step"] == 1  # the factor kernel is skipped for one layer
    assert dev.memory.scratch_bytes >= one.memory.scratch_bytes


def test_cuda_uniform_state_uploads_only_per_layer_depths(gpu):
    from maple.core.backend import read_transfer_counters

    dev = prepare_column(layers(), gpu.ones(1000, dtype=bool), AREA, backend="cuda", reserve_bytes=0,
                         available_bytes_fn=lambda: 1 << 40)
    before = read_transfer_counters()
    state = uniform_state(dev, [0.25, 0.4])
    delta = read_transfer_counters().delta(before)
    assert delta.host_to_device == 1 and delta.host_to_device_bytes == 2 * 8 and delta.device_to_host == 0
    assert type(state.water_m) is gpu.ndarray and state.water_m.shape == (2, 1000)
    expected = np.repeat((np.array([0.25, 0.4]) * np.array([0.25, 0.75]))[:, None], 1000, axis=1)
    np.testing.assert_array_equal(host(state.water_m), expected)


def test_cuda_prepare_admits_the_device_side_before_building(gpu):
    from maple_syrup.subsurface import estimate_step_memory

    calls = []

    def zero_budget():
        calls.append(1)
        return 0

    with pytest.raises(SubsurfaceMemoryError):
        prepare_column(layers(), gpu.ones(4, dtype=bool), AREA, backend="cuda", reserve_bytes=0,
                       available_bytes_fn=zero_budget)
    assert calls == [1]
    estimate = estimate_step_memory(2, 4, backend="cuda")
    assert estimate.required_bytes == estimate.device_bytes and estimate.host_bytes > 0
    dev = prepare_column(layers(), gpu.ones(4, dtype=bool), AREA, backend="cuda", reserve_bytes=0,
                         available_bytes_fn=lambda: estimate.required_bytes)
    with pytest.raises(SubsurfaceMemoryError):
        column_state(dev, gpu.zeros((2, 4)))  # the extra state copy exceeds the exact requirement
    state = uniform_state(dev, [0.25, 0.4])
    assert state.water_m.shape == (2, 4)


def test_cuda_unavailable_is_a_setup_error_not_a_fallback():
    backend = pytest.importorskip("maple.core.backend")
    if backend.gpu_execution_available():
        pytest.skip("a GPU is available; the unavailable path is not exercised in this environment")
    with pytest.raises(SubsurfaceBackendUnavailableError):
        prepare_column(layers(), np.ones(2, dtype=bool), AREA, backend="cuda", reserve_bytes=0)
