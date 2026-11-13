"""Memory model, admission before allocation, warm tracemalloc bound, and the acceptance that provider outputs build the ACTUAL SpatialSoilColumn (no fake class). Author-written."""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from conftest import BIG, call, random_valid

from maple_syrup import soil_texture_properties as tp
from maple_syrup import spatial_soil_layers as sp


def inputs(rng, L, N):
    return [a.reshape(L, N).copy() for a in random_valid(rng, L * N)]


def test_estimate_terms_and_host_device_split():
    lay = {k: "cell" for k in tp.INPUTS}
    e = tp.estimate_property_memory(3, 1000, layouts=lay, backend="numpy", chunk_columns=100)
    assert e.output_bytes == 3 * 8 * 3 * 1000 and e.host_required_bytes == e.main_required_bytes == e.output_bytes + tp.REF_VEC_F64 * 8 * 100 + tp.REF_VEC_BOOL * 100 + tp.PY_ALLOWANCE_BYTES
    assert e.device_required_bytes == 0 and e.terms["no_workspace"]
    d = tp.estimate_property_memory(3, 1000, layouts=lay, backend="numpy", diagnostics=True)
    assert d.output_bytes == e.output_bytes + 2 * 8 * 3000 + 3000
    c = tp.estimate_property_memory(3, 1000, layouts=lay, backend="cuda", hash_chunk_bytes=4096, reserve_bytes=7, host_reserve_bytes=5)
    assert c.device_required_bytes == c.output_bytes + tp.PACKET_BYTES + 8 * len(tp.ALL_INPUTS) + 7 and c.main_required_bytes == c.device_required_bytes
    assert c.host_required_bytes == 4096 + tp.PACKET_BYTES + tp.PY_ALLOWANCE_BYTES + 5 and c.terms["numpy_chunk_work_vectors"] == 0
    shared = tp.estimate_property_memory(3, 10**9, layouts={k: "shared" for k in tp.INPUTS}, backend="cuda")
    assert shared.output_layout == "shared" and shared.output_bytes == 3 * 8 * 3  # O(L), no N
    assert tp.estimate_property_memory(3, 10**9, layouts={k: "shared" for k in tp.INPUTS}, backend="numpy").host_required_bytes < 10**6
    with pytest.raises(tp.TexturePropertyError):
        tp.estimate_property_memory(3, 10, layouts={k: "cell" for k in tp.INPUTS[:4]})  # a mandatory layout is missing
    with pytest.raises(tp.TexturePropertyError):
        tp.estimate_property_memory(3, 10, layouts={**lay, "bogus": "cell"})
    with pytest.raises(tp.TexturePropertyError):
        tp.estimate_property_memory(1, 10, layouts=lay, backend="numpy", host_reserve_bytes=1)
    with pytest.raises(tp.TexturePropertyError):
        tp.estimate_property_memory(10**9, 10**9, layouts=lay, backend="cuda")  # the 64-bit packed element index


def test_admission_is_exact_and_happens_before_allocation(backend, monkeypatch):
    rng = np.random.default_rng(6)
    args = inputs(rng, 2, 50)
    est = tp.estimate_property_memory(2, 50, layouts={k: "cell" for k in tp.INPUTS}, backend=backend, input_bytes={})
    need = est.main_required_bytes
    calls = []
    real = np.zeros
    monkeypatch.setattr(tp.np, "zeros", lambda *a, **k: calls.append(a) or real(*a, **k))
    for budget, reader in ((need - 1, None), (BIG, lambda: need - 1), (BIG, lambda: 0)):
        with pytest.raises(tp.TexturePropertyMemoryError, match="nothing was allocated"):
            call(*args, L=2, N=50, backend=backend, memory_budget_bytes=budget, available_bytes_fn=reader)
    assert not calls  # no output was allocated by any refused call
    with pytest.raises(tp.TexturePropertyMemoryError):
        call(*args, L=2, N=50, backend=backend, memory_budget_bytes=need, reserve_bytes=1)  # the reserve counts
    call(*args, L=2, N=50, backend=backend, memory_budget_bytes=need, available_bytes_fn=lambda: need)  # exactly enough is admitted
    for bad in (lambda: True, lambda: -1, lambda: 1.5):
        with pytest.raises(tp.TexturePropertyError, match="available-bytes reader"):
            call(*args, L=2, N=50, backend=backend, available_bytes_fn=bad)


@pytest.mark.parametrize("N,chunk,diag", [(5000, 5000, False), (5000, 100, True), (30000, 1000, True)])
def test_warm_peak_stays_below_the_model(N, chunk, diag, backend):
    rng = np.random.default_rng(7)
    args = inputs(rng, 1, 64)
    args = [np.tile(a, (1, (N + 63) // 64))[:, :N].copy() for a in args]
    kw = dict(L=1, N=N, backend=backend, chunk_columns=chunk, diagnostics=diag)
    call(*args, **kw)  # warm (compilation, imports, source digests)
    tracemalloc.start()
    tracemalloc.reset_peak()
    r = call(*args, **kw)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= r.estimate.host_required_bytes, (peak, r.estimate.host_required_bytes)


def test_work_vectors_scale_with_chunk_not_with_N():
    lay = {k: "cell" for k in tp.INPUTS}
    a = tp.estimate_property_memory(1, 10**6, layouts=lay, chunk_columns=1000)
    b = tp.estimate_property_memory(1, 10**6, layouts=lay, chunk_columns=2000)
    assert b.terms["numpy_chunk_work_vectors"] == 2 * a.terms["numpy_chunk_work_vectors"] and a.output_bytes == b.output_bytes


def test_cuda_backend_refused_without_cupy_and_nothing_runs(monkeypatch):
    try:
        import cupy  # noqa: F401
        pytest.skip("CuPy is importable here; the unavailable-backend refusal is not exercised (no claim)")
    except ImportError:
        pass
    with pytest.raises(tp.TexturePropertyBackendUnavailableError):
        call(backend="cuda")


# ---- the actual SpatialSoilColumn builds from provider outputs --------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["cell", "shared"])
def test_provider_outputs_build_a_real_spatial_column_and_it_steps(layout):
    rng = np.random.default_rng(8)
    L, N = 3, 9
    if layout == "cell":
        S, Si, C, OM, rho = inputs(rng, L, N)
        r = call(S, Si, C, OM, rho, L=L, N=N, diagnostics=True)
    else:
        S, Si, C, OM, rho = (a[:, :1].reshape(L).copy() for a in inputs(rng, L, 1))
        r = call(S, Si, C, OM, rho, L=L, N=N, diagnostics=True)
    assert r.layout == layout
    edges = np.cumsum(np.r_[0.0, [0.05, 0.1, 0.2]])
    spec = sp.spatial_layer_spec(edges, theta_residual=r.array("theta_1500"), theta_field_capacity=r.array("theta_field_capacity_6kpa"), theta_saturation=r.array("theta_saturation"),
                                 diffusivity_m2_s=1e-7, theta_evaporation_stress=0.6, n_cells=N)  # residual/diffusivity/stress are the caller's explicit choices
    assert spec.layouts()["theta_saturation"] == layout and spec.layouts()["theta_field_capacity"] == layout
    col = sp.prepare_spatial_column(spec, np.ones(N, dtype=bool), 4.0, backend="numpy", memory_budget_bytes=BIG)
    assert isinstance(col, sp.SpatialSoilColumn)
    got = col.inspect("theta_saturation", host_memory_budget_bytes=BIG)
    assert np.array_equal(got.reshape(-1), np.array(r.array("theta_saturation")).reshape(-1))
    st = sp.spatial_uniform_state(col, [0.15, 0.15, 0.15], 0.0, memory_budget_bytes=BIG)
    s1 = sp.spatial_step(col, st, 60.0, added_water_m=np.zeros((L, N)), potential_evaporation_m_s=np.zeros(N), memory_budget_bytes=BIG)
    assert np.all(np.isfinite(np.asarray(s1.state.water_m)))
    # Ks is a separate saturated-conductivity map; it is NOT turned into an interface conductivity or averaged here
    assert "interface" not in " ".join(r.names())
