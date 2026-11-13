"""Actual-GPU test DEFINITIONS for the spatial CUDA withdrawal/availability (WRITTEN, NOT RUN by their author: no GPU was launched or probed). Skipped without a usable CuPy device; Root runs them under its device
guard (GPU0 UUID, fresh 0 MiB and 0 %, `.venv-cupy`, genuine 9f CUDA materialisation). They compare the device operator with the accepted CPU spatial path (arrays bit for bit: no libm term and the products are
unfused under --fmad=false), with the accepted COMMON operator column by column, the dyadic hand case, flags/typed errors on every cell, sizes around the block boundary, memory against the model using real allocator
requests and the used-bytes charge (total pool growth is not a lower bound: reused split chunks), the plateau, transfers, devices and release."""
from __future__ import annotations

import gc

import numpy as np
import pytest
from conftest import BIG, a2, a3, bits, sp, ss, st, sw

RT = 1e-11


@pytest.fixture(scope="module")
def cp():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the spatial CUDA withdrawal is not exercised (no claim)")
    return backend.cupy_module()


def dev(cp, a):
    return cp.asarray(np.ascontiguousarray(a))


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def owners(cp, p, mask, **kw):
    n = p["edges"].shape[1]
    spec = sp.spatial_layer_spec(dev(cp, p["edges"]), theta_residual=dev(cp, p["res"]), theta_field_capacity=dev(cp, p["fc"]), theta_saturation=dev(cp, p["sat"]), theta_evaporation_stress=0.5,
                                 diffusivity_m2_s=dev(cp, p["iface"]), n_cells=n, **kw)
    gpu = sp.prepare_spatial_column(spec, dev(cp, mask), 4.0, backend="cuda", memory_budget_bytes=BIG)
    return gpu, a2.prepare(a2.make_spec(p, **kw), mask=mask)


def wk(rng, p, kind):
    L, N = p["res"].shape
    dz = np.diff(p["edges"], axis=0)
    res, sat = p["res"] * dz, p["sat"] * dz
    return {"zero": np.zeros((L, N)), "below": res * rng.uniform(0, 1, (L, N)), "sat": sat.copy(), "random": sat * rng.uniform(0, 1, (L, N))}[kind]


def run_both(cp, p, mask, w, q):
    gpu, cpu = owners(cp, p, mask)
    gop, cop = sw.prepare_spatial_withdrawal(gpu, memory_budget_bytes=BIG), sw.prepare_spatial_withdrawal(cpu, memory_budget_bytes=BIG)
    g = sw.spatial_withdraw(gop, sp.spatial_column_state(gpu, dev(cp, w), 9.0, memory_budget_bytes=BIG), dev(cp, q), memory_budget_bytes=BIG)
    c = sw.spatial_withdraw(cop, sp.spatial_column_state(cpu, w, 9.0, memory_budget_bytes=BIG), q, memory_budget_bytes=BIG)
    return gpu, gop, g, c


@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("N", [1, 7, 127, 128, 129, 257, 1001])
def test_device_withdrawal_matches_the_cpu_spatial_path_bit_for_bit_on_the_arrays(cp, L, N):
    rng = np.random.default_rng(N + L)
    p = a2.params(rng, L, N)
    mask = rng.random(N) < 0.75
    mask[0] = True
    for kind, scale in (("random", 0.05), ("below", 0.01), ("zero", 0.02), ("sat", 0.5)):
        w = wk(rng, p, kind)
        q = rng.uniform(0, scale, (L, N)) * mask
        gpu, gop, g, c = run_both(cp, p, mask, w, q)
        for f in ("actual_m", "unmet_m", "scale"):
            np.testing.assert_array_equal(bits(host(getattr(g, f))), bits(getattr(c, f)))
        np.testing.assert_array_equal(bits(host(g.state.water_m)), bits(c.state.water_m))
        for f in ("storage_before_m3", "storage_after_m3", "requested_m3", "actual_m3", "unmet_m3"):
            assert getattr(g.budget, f) == pytest.approx(getattr(c.budget, f), rel=RT, abs=1e-15)
        assert g.state.t_s == 9.0 and g.state.namespace == "cupy" and g.budget.n_terms == c.budget.n_terms


def test_each_column_equals_the_accepted_common_operator_and_the_dyadic_example_is_exact(cp):
    rng = np.random.default_rng(3)
    L, N = 3, 24
    p = a2.params(rng, L, N)
    mask = np.ones(N, bool)
    w = wk(rng, p, "random")
    q = rng.uniform(0, 0.05, (L, N))
    gpu, gop, g, _ = run_both(cp, p, mask, w, q)
    for c in range(N):
        lay = ss.soil_layers(p["edges"][:, c], theta_residual=p["res"][:, c], theta_field_capacity=p["fc"][:, c], theta_saturation=p["sat"][:, c], theta_evaporation_stress=0.5, diffusivity_m2_s=p["iface"][:, c])
        col = ss.prepare_column(lay, np.ones(1, bool), 4.0, backend="numpy")
        r = st.withdraw_transpiration(st.prepare_transpiration(col), ss.column_state(col, w[:, c : c + 1].copy(), 9.0), q[:, c : c + 1].copy())
        np.testing.assert_array_equal(bits(host(g.actual_m)[:, c : c + 1]), bits(r.actual_m))
        np.testing.assert_array_equal(bits(host(g.state.water_m)[:, c : c + 1]), bits(r.state.water_m))
    spec = sp.spatial_layer_spec(dev(cp, np.array([0.0, 1.0, 2.0])), theta_residual=0.0, theta_field_capacity=0.25, theta_saturation=dev(cp, np.array([0.5, 0.5])), theta_evaporation_stress=0.5, diffusivity_m2_s=0.0, n_cells=1)
    col = sp.prepare_spatial_column(spec, dev(cp, np.ones(1, bool)), 4.0, backend="cuda", memory_budget_bytes=BIG)
    op = sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
    r = sw.spatial_withdraw(op, sp.spatial_column_state(col, dev(cp, np.array([[25.0], [200.0]]) / 1024.0), 7.0, memory_budget_bytes=BIG), dev(cp, np.array([[50.0], [50.0]]) / 1024.0), memory_budget_bytes=BIG)
    np.testing.assert_array_equal(host(r.state.water_m), np.array([[0.0], [150.0]]) / 1024.0)
    np.testing.assert_array_equal(host(r.scale), np.array([[0.5], [1.0]]))
    assert r.budget.actual_m3 / r.budget.requested_m3 == 0.75


@pytest.mark.parametrize("label,mut", [("nan_water", lambda w, q, m: w.__setitem__((1, 2), np.nan)), ("over_saturated", lambda w, q, m: w.__setitem__((2, 4), 10.0)),
                                       ("nan_water_inactive", lambda w, q, m: w.__setitem__((1, int(np.flatnonzero(~m)[0])), np.nan)), ("negative_request", lambda w, q, m: q.__setitem__((2, 5), -1e-6)),
                                       ("positive_request_inactive", lambda w, q, m: q.__setitem__((1, int(np.flatnonzero(~m)[0])), 1e-6))])
def test_invalid_inputs_give_the_identical_typed_error_and_leave_everything_unchanged(cp, label, mut):
    rng = np.random.default_rng(4)
    N = 260
    p = a2.params(rng, 3, N)
    mask = rng.random(N) < 0.7
    mask[0], mask[-1] = True, False
    w = wk(rng, p, "random")
    q = rng.uniform(0, 0.02, (3, N)) * mask
    mut(w, q, mask)
    gpu, cpu = owners(cp, p, mask)
    gop, cop = sw.prepare_spatial_withdrawal(gpu, memory_budget_bytes=BIG), sw.prepare_spatial_withdrawal(cpu, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceError) as e_c:
        sw.spatial_withdraw(cop, sp.spatial_column_state(cpu, w, 0.0, memory_budget_bytes=BIG), q, memory_budget_bytes=BIG)
    state, req = sp.spatial_column_state(gpu, dev(cp, w), 0.0, memory_budget_bytes=BIG), dev(cp, q)
    w0, q0 = host(state.water_m).copy(), host(req).copy()
    with pytest.raises(ss.SubsurfaceError) as e_g:
        sw.spatial_withdraw(gop, state, req, memory_budget_bytes=BIG)
    assert str(e_g.value) == str(e_c.value)
    np.testing.assert_array_equal(host(state.water_m), w0)
    np.testing.assert_array_equal(host(req), q0)


def test_typed_overflow_availability_and_guards(cp):
    spec = sp.spatial_layer_spec(dev(cp, np.array([0.0, 1.0])), theta_residual=0.0, theta_field_capacity=0.25, theta_saturation=0.5, theta_evaporation_stress=0.375, diffusivity_m2_s=0.0, n_cells=8)
    col = sp.prepare_spatial_column(spec, dev(cp, np.ones(8, bool)), 1e308, backend="cuda", memory_budget_bytes=BIG)
    op = sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
    state = sp.spatial_column_state(col, dev(cp, np.full((1, 8), 0.2)), 0.0, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceError, match="not representable in FP64"):
        sw.spatial_withdraw(op, state, dev(cp, np.full((1, 8), 0.2)), memory_budget_bytes=BIG)
    assert np.all(host(sw.spatial_layer_availability_m(op, state, memory_budget_bytes=BIG)) == 0.2)
    with pytest.raises(sp.SpatialLayerError, match="cupy.ndarray"):
        sw.spatial_withdraw(op, state, np.full((1, 8), 0.2), memory_budget_bytes=BIG)
    if cp.cuda.runtime.getDeviceCount() > 1:
        with cp.cuda.Device(1), pytest.raises(sp.SpatialLayerError, match="current device"):
            sw.spatial_withdraw(op, state, dev(cp, np.full((1, 8), 0.2)), memory_budget_bytes=BIG)


def test_real_allocator_requests_used_bytes_transfers_and_plateau(cp):
    """Requested payload equals the model exactly; the used-bytes delta equals the pool-charged unique outputs; total pool growth keeps only an upper bound; the plateau uses a resident zero request."""
    rng = np.random.default_rng(5)
    L, N = 4, 50_000
    p = a2.params(rng, L, N)
    mask = np.ones(N, bool)
    gpu, _ = owners(cp, p, mask)
    op = sw.prepare_spatial_withdrawal(gpu, memory_budget_bytes=BIG)
    state = sp.spatial_column_state(gpu, dev(cp, wk(rng, p, "random")), 0.0, memory_budget_bytes=BIG)
    q, zero = dev(cp, rng.uniform(0, 0.01, (L, N))), cp.zeros((L, N), dtype=cp.float64)
    sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)  # warm
    cp.cuda.Stream.null.synchronize()
    gc.collect()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    seed = cp.empty(1 << 20, dtype=cp.uint8)
    del seed
    anchor = cp.empty(512, dtype=cp.uint8)
    before = dict(total=pool.total_bytes(), used=pool.used_bytes())
    est = sw.estimate_spatial_withdrawal_memory(L, N, backend="cuda", operation="withdraw")
    requests = []

    def alloc(n):
        requests.append(int(n))
        return pool.malloc(n)

    with cp.cuda.using_allocator(alloc):
        r = sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)
    cp.cuda.Stream.null.synchronize()
    outs = [r.state.water_m, r.actual_m, r.unmet_m, r.scale]
    payload, charged = sum(int(a.nbytes) for a in outs), sum(int(a.data.mem.size) for a in outs)
    assert sum(requests) == est.device_required_bytes and payload == est.output_bytes
    assert pool.used_bytes() - before["used"] == charged and payload <= charged <= payload + 512 * len(outs)
    assert pool.total_bytes() - before["total"] <= est.device_required_bytes + 512 * len(requests)
    del r, outs, anchor
    gc.collect()
    totals = []
    for _ in range(8):
        state = sw.spatial_withdraw(op, state, zero, memory_budget_bytes=BIG).state
        cp.cuda.Stream.null.synchronize()
        totals.append((pool.total_bytes(), pool.used_bytes()))
    assert totals[-1][0] <= (totals[2][0] + (1 << 20)) and totals[-1][1] <= totals[2][1] + (1 << 20)
