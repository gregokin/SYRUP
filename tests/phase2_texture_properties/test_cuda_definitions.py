"""Actual-GPU test DEFINITIONS for the texture-property CUDA backend (WRITTEN, NOT RUN by their author: no GPU was launched or probed, no CuPy was initialised, no NVRTC compile happened). Skipped without a usable CuPy
device; Root runs them under its device guard (GPU0 UUID, fresh 0 MiB and 0 %), with `.venv-cupy` and the genuine 9f CUDA materialisation. They compare the device result with the accepted NumPy reference
(bitwise for the outputs without a transcendental; declared bound rtol 2e-12 / atol 1e-14 for `saturated_conductivity_m_s`), flags/first element/counts, layouts, block-edge sizes, the 32-byte transfer budget,
memory growth against the model, release and plateau, and refusal of host arrays/other devices."""
from __future__ import annotations

import gc

import numpy as np
import pytest
from conftest import AT, BIG, RT, call, random_valid

from maple_syrup import soil_texture_properties as tp


@pytest.fixture(scope="module")
def cp():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the texture-property CUDA backend is not exercised (no claim)")
    return backend.cupy_module()


def dev(cp, a):
    return cp.asarray(np.ascontiguousarray(a)) if isinstance(a, np.ndarray) else a


def gpu_call(cp, *arrs, L, N, **kw):
    return call(*[dev(cp, a) for a in arrs], L=L, N=N, backend="cuda", **{k: dev(cp, v) for k, v in kw.items()})


def host(r, name):
    return r.detached(name, memory_budget_bytes=BIG).get()


def mk(rng, L, N):
    return [a.reshape(L, N).copy() for a in random_valid(rng, L * N)]


@pytest.mark.parametrize("L,N", [(1, 1), (3, 127), (2, 128), (2, 129), (3, 257), (4, 5000)])
def test_device_equals_numpy_cell_layout(cp, L, N):
    rng = np.random.default_rng(30 + N)
    args = mk(rng, L, N)
    ov = np.where(rng.random((L, N)) < 0.4, rng.uniform(0.5, 25.0, (L, N)), np.nan)
    g = gpu_call(cp, *args, L=L, N=N, diagnostics=True, air_entry_kpa_override=ov)
    n = call(*args, L=L, N=N, backend="numpy", diagnostics=True, air_entry_kpa_override=ov)
    for name in ("theta_saturation", "theta_field_capacity_6kpa", "theta_1500", "air_entry_kpa", "provenance"):
        assert np.array_equal(host(g, name), n.array(name)), name
    kg, kn = host(g, "saturated_conductivity_m_s"), n.array("saturated_conductivity_m_s")
    assert np.all(np.abs(kg - kn) <= AT + RT * np.abs(kn))
    assert g.sha256 == n.sha256 and g.n_fc_equals_saturation == n.n_fc_equals_saturation and g.device_id is not None


def test_shared_and_scalar_layouts_and_flags(cp):
    rng = np.random.default_rng(31)
    S, Si, C, OM, rho = (a[:, 0].copy() for a in mk(rng, 3, 1))
    g = gpu_call(cp, S, Si, C, OM, rho, L=3, N=10**6)
    assert g.layout == "shared" and g.estimate.output_bytes == 3 * 8 * 3
    n = call(S, Si, C, OM, rho, L=3, N=10**6, backend="numpy")
    assert np.array_equal(host(g, "theta_saturation"), n.array("theta_saturation"))
    L, N = 3, 300
    args = mk(rng, L, N)
    args[3][2, 5], args[3][1, 250] = 9.0, -2.0
    errs = []
    for bk in ("numpy", "cuda"):
        with pytest.raises(tp.TexturePropertyError) as e:
            (gpu_call(cp, *args, L=L, N=N) if bk == "cuda" else call(*args, L=L, N=N, backend="numpy"))
        errs.append((e.value.first_layer, e.value.first_cell, e.value.n_flagged, e.value.flags))
    assert errs[0] == errs[1]


def test_air_entry_branches_on_device(cp):
    for air in (0.0, 5.999999, 6.0, 20.0):
        g = gpu_call(cp, 0.4, 0.4, 0.2, 2.5, 1400.0, L=1, N=1, air_entry_kpa_override=air)
        n = call(0.4, 0.4, 0.2, 2.5, 1400.0, L=1, N=1, backend="numpy", air_entry_kpa_override=air)
        assert np.array_equal(host(g, "theta_field_capacity_6kpa"), n.array("theta_field_capacity_6kpa"))
    for air in (-0.5, 33.0):
        with pytest.raises(tp.TexturePropertyError):
            gpu_call(cp, 0.4, 0.4, 0.2, 2.5, 1400.0, L=1, N=1, air_entry_kpa_override=air)


def test_transfers_memory_model_release_and_plateau(cp):
    """Device pool growth stays within the model; only the 32-byte diagnostic packet and the identity chunks cross to the host; repeated calls plateau and release."""
    rng = np.random.default_rng(32)
    L, N = 4, 100_000
    args = [dev(cp, a) for a in mk(rng, L, N)]
    pool = cp.get_default_memory_pool()
    est = tp.estimate_property_memory(L, N, layouts={k: "cell" for k in tp.INPUTS}, backend="cuda")
    gpu_call(cp, *args, L=L, N=N)  # warm (NVRTC compile, module load)
    cp.cuda.Stream.null.synchronize()
    pool.free_all_blocks()
    base = pool.used_bytes()
    r = gpu_call(cp, *args, L=L, N=N)
    grown = pool.used_bytes() - base
    assert grown <= est.device_required_bytes, (grown, est.device_required_bytes)
    del r
    gc.collect()
    assert pool.used_bytes() <= base
    peaks = []
    for _ in range(4):
        r = gpu_call(cp, *args, L=L, N=N)
        peaks.append(pool.total_bytes())
        del r
        gc.collect()
    assert peaks[-1] <= peaks[1]  # plateau


def test_host_array_and_other_device_refused_without_transfer(cp):
    with pytest.raises(tp.TexturePropertyError):
        call(np.full((1, 1), 0.4), 0.4, 0.2, 2.5, 1400.0, L=1, N=1, backend="cuda")
    with pytest.raises(tp.TexturePropertyError):
        call(cp.full((1, 1), 0.4), 0.4, 0.2, 2.5, 1400.0, L=1, N=1, backend="numpy")


def test_detached_refuses_other_current_device(cp):
    """Gated (Root, GPU0 + a second visible device only): a result owned by device 0 refuses detached() when another device is current, before any reader or copy."""
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs two visible devices (no claim)")
    g = gpu_call(cp, 0.4, 0.4, 0.2, 2.5, 1400.0, L=1, N=1)
    with cp.cuda.Device(1):
        with pytest.raises(tp.TexturePropertyError, match="current device"):
            g.detached("theta_saturation", memory_budget_bytes=BIG)
    assert g.detached("theta_saturation", memory_budget_bytes=BIG).device.id == g.device_id


def test_validation_packet_is_one_counted_32_byte_read(cp):
    """Gated: through the actual MAPLE transfer counters, one device-to-host transfer of 32 bytes is the validation packet; cold hash chunks (`.get(out=)`) are separate and not mirrored maps."""
    from maple.core.backend import read_transfer_counters

    rng = np.random.default_rng(40)
    args = [dev(cp, a) for a in mk(rng, 2, 1000)]
    gpu_call(cp, *args, L=2, N=1000)  # warm: compile and module load
    before = read_transfer_counters()
    gpu_call(cp, *args, L=2, N=1000)
    d = read_transfer_counters().delta(before)
    assert d.device_to_host == 1 and d.device_to_host_bytes == 32
