"""CPU-lane checks of the CUDA component: the kernel SOURCE executed by the host compiler and the Python launcher/plumbing through a CuPy double. NOT hardware evidence (see hostcuda). Skipped without g++."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import AT, BIG, RT, call, random_valid
from hostcuda import FakeArr, FakeCupy, build, fake, have_toolchain

from maple_syrup import soil_texture_properties as tp
from maple_syrup import soil_texture_properties_cuda as cu
from maple_syrup import soil_water_remap_cuda as rc

pytestmark = pytest.mark.skipif(not have_toolchain(), reason="no g++: the host execution of the CUDA source is not run (no claim)")


@pytest.fixture(scope="module")
def lib():
    return build(cu.render_source())


@pytest.fixture()
def cupy(lib, monkeypatch):
    cp = FakeCupy(lib)
    monkeypatch.setattr(tp, "_namespace_type", lambda backend: FakeArr)
    monkeypatch.setattr(rc, "require_cuda", lambda: cp)
    monkeypatch.setattr(rc, "current_device_id", lambda c: 0)
    with cu._LOCK:
        saved = dict(cu._KERNELS)
        cu._KERNELS.clear()
    yield cp
    with cu._LOCK:
        cu._KERNELS.clear()
        cu._KERNELS.update(saved)


def cuda_call(cp, *arrs, L, N, **kw):
    conv = [fake(a) if isinstance(a, np.ndarray) else a for a in arrs]
    kw = {k: (fake(v) if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
    return call(*conv, L=L, N=N, backend="cuda", available_bytes_fn=lambda: 1 << 40, host_available_bytes_fn=lambda: 1 << 40, **kw)


def host_out(r, name):
    return np.asarray(r.detached(name, memory_budget_bytes=BIG, available_bytes_fn=lambda: 1 << 40))


def mk(rng, L, N):
    return [a.reshape(L, N).copy() for a in random_valid(rng, L * N)]


@pytest.mark.parametrize("L,N", [(1, 1), (3, 127), (2, 128), (2, 129), (3, 257)])
def test_kernel_text_equals_numpy_reference_cell_layout(cupy, L, N):
    rng = np.random.default_rng(20 + N)
    args = mk(rng, L, N)
    ov = np.where(rng.random((L, N)) < 0.4, rng.uniform(0.5, 25.0, (L, N)), np.nan)
    ks = np.where(rng.random((L, N)) < 0.2, rng.uniform(1e-7, 1e-4, (L, N)), np.nan)
    g = cuda_call(cupy, *args, L=L, N=N, diagnostics=True, air_entry_kpa_override=ov, saturated_conductivity_override_m_s=ks)
    n = call(*args, L=L, N=N, backend="numpy", diagnostics=True, air_entry_kpa_override=ov, saturated_conductivity_override_m_s=ks)
    for name in ("theta_saturation", "theta_field_capacity_6kpa", "theta_1500", "air_entry_kpa", "provenance"):
        assert np.array_equal(host_out(g, name), n.array(name)), name
    kg, kn = host_out(g, "saturated_conductivity_m_s"), n.array("saturated_conductivity_m_s")
    assert np.all(np.abs(kg - kn) <= AT * 1e-4 + RT * np.abs(kn))
    assert g.n_fc_equals_saturation == n.n_fc_equals_saturation and g.sha256 == n.sha256
    if L * N > 100:
        assert n.n_fc_equals_saturation > 0  # the 6 kPa == saturation outcome is exercised
    grid, block, nargs = cupy.launches[-1]
    assert (grid, block, nargs) == (((N + 127) // 128,), (128,), 43)
    code, options, backend = cupy.modules[-1]
    assert code == cu.render_source() and options == ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false") and backend == "nvrtc"


def test_shared_scalar_and_mixed_layouts(cupy):
    rng = np.random.default_rng(21)
    S, Si, C, OM, rho = (a[:, 0].copy() for a in mk(rng, 3, 1))
    g = cuda_call(cupy, S, Si, C, OM, rho, L=3, N=10**6)  # shared inputs: one thread, O(L) outputs
    n = call(S, Si, C, OM, rho, L=3, N=10**6, backend="numpy")
    assert g.layout == "shared" and cupy.launches[-1][0] == (1,)
    assert np.array_equal(host_out(g, "theta_saturation"), n.array("theta_saturation"))
    g2 = cuda_call(cupy, 0.4, 0.4, 0.2, 2.5, 1400.0, L=2, N=5)
    n2 = call(0.4, 0.4, 0.2, 2.5, 1400.0, L=2, N=5, backend="numpy")
    assert np.array_equal(host_out(g2, "theta_field_capacity_6kpa"), n2.array("theta_field_capacity_6kpa"))
    cell_rho = np.tile(rho[:, None], (1, 6)) * np.linspace(0.97, 1.03, 6)
    g3 = cuda_call(cupy, S, Si, C, OM, cell_rho, L=3, N=6)
    n3 = call(S, Si, C, OM, cell_rho, L=3, N=6, backend="numpy")
    assert g3.layout == "cell" and np.array_equal(host_out(g3, "theta_saturation"), n3.array("theta_saturation"))


def test_flags_first_element_and_counts_match_numpy(cupy):
    rng = np.random.default_rng(22)
    L, N = 3, 200
    args = mk(rng, L, N)
    args[3][2, 5], args[3][1, 150], args[4][0, 199] = 9.0, -2.0, 5000.0
    errs = []
    for bk in ("numpy", "cuda"):
        with pytest.raises(tp.TexturePropertyError) as e:
            (cuda_call(cupy, *args, L=L, N=N) if bk == "cuda" else call(*args, L=L, N=N, backend="numpy"))
        errs.append((e.value.first_layer, e.value.first_cell, e.value.n_flagged, e.value.flags, e.value.bit))
    assert errs[0] == errs[1] and errs[0][:3] == (0, 199, 3)


def test_scalar_override_absent_and_present_through_aux(cupy):
    a = cuda_call(cupy, 0.4, 0.4, 0.2, 2.5, 1400.0, L=2, N=3, diagnostics=True, theta_1500_override=0.07, air_entry_kpa_override=3.0)
    b = call(0.4, 0.4, 0.2, 2.5, 1400.0, L=2, N=3, backend="numpy", diagnostics=True, theta_1500_override=0.07, air_entry_kpa_override=3.0)
    for name in ("theta_1500", "air_entry_kpa", "theta_field_capacity_6kpa", "provenance"):
        assert np.array_equal(host_out(a, name), b.array(name))


def test_cuda_memory_split_and_quotas(cupy):
    args = mk(np.random.default_rng(23), 2, 40)
    est = tp.estimate_property_memory(2, 40, layouts={k: "cell" for k in tp.INPUTS}, backend="cuda", hash_chunk_bytes=64)
    ok = dict(available_bytes_fn=lambda: 1 << 40, host_available_bytes_fn=lambda: 1 << 40)
    fk = [fake(a) for a in args]
    with pytest.raises(tp.TexturePropertyMemoryError, match="device side"):
        call(*fk, L=2, N=40, backend="cuda", memory_budget_bytes=est.device_required_bytes - 1, hash_chunk_bytes=64, **ok)
    with pytest.raises(tp.TexturePropertyMemoryError, match="host side"):
        call(*fk, L=2, N=40, backend="cuda", host_memory_budget_bytes=est.host_required_bytes - 1, hash_chunk_bytes=64, **ok)
    with pytest.raises(tp.TexturePropertyMemoryError, match="device side"):
        call(*fk, L=2, N=40, backend="cuda", hash_chunk_bytes=64, available_bytes_fn=lambda: est.device_required_bytes - 1, host_available_bytes_fn=lambda: 1 << 40)
    assert not cupy.launches  # nothing was launched by a refused call
    call(*fk, L=2, N=40, backend="cuda", memory_budget_bytes=est.device_required_bytes, host_memory_budget_bytes=est.host_required_bytes, hash_chunk_bytes=64, **ok)
    assert len(cupy.launches) == 1


def test_namespace_and_device_mismatch_refused_before_work(cupy):
    with pytest.raises(tp.TexturePropertyError, match="numpy.ndarray|cupy.ndarray"):
        call(np.full((1, 1), 0.4), 0.4, 0.2, 2.5, 1400.0, L=1, N=1, backend="cuda")  # a host array for the device backend: no implicit transfer
    assert not cupy.launches
    with pytest.raises(tp.TexturePropertyError, match="exactly a numpy.ndarray"):
        call(fake(np.full((1, 1), 0.4)), 0.4, 0.2, 2.5, 1400.0, L=1, N=1, backend="numpy")  # a device array for a CPU backend


def test_source_is_generated_from_the_recipe_tuples():
    src = cu.render_source()
    for k, v in enumerate(tp.EQ4_REGRESSION):
        assert f"#define E4R{k} ({float(v)!r})" in src
    assert f"#define LNR ({tp.LN_RATIO!r})" in src and "--fmad" not in src and "fma(" not in src
