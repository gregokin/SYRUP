"""Actual-GPU test DEFINITIONS for the spatial CUDA trial (WRITTEN, NOT RUN by their author: no GPU was launched or probed). Skipped without a usable CuPy device; Root runs them under its device guard
(GPU0 UUID, fresh 0 MiB and 0 %), with `.venv-cupy` and the genuine 9f CUDA materialisation. They compare the device step with the accepted NumPy spatial path and with the accepted COMMON CUDA column
(one column at a time), check flags/errors, mixed layouts, sizes around the block boundary, memory against the model (pool growth), transfers, devices, release and plateau. Declared bound rtol 2e-12 /
atol 1e-14 (device expm1 may differ by an ulp); bitwise with zero diffusivity."""
from __future__ import annotations

import gc

import numpy as np
import pytest
from conftest import BIG, a2, sp, ss

RT, AT = 2e-12, 1e-14


@pytest.fixture(scope="module")
def cp():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the spatial CUDA trial is not exercised (no claim)")
    return backend.cupy_module()


def dev(cp, a):
    return cp.asarray(np.ascontiguousarray(a))


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def owners(cp, p, *, mode="constant", bottom="closed", mask=None, edges=None, iface=None, **kw):
    n = p["edges"].shape[1]
    mask = np.ones(n, bool) if mask is None else mask
    edges = p["edges"] if edges is None else edges
    iface = p["iface"] if iface is None else iface
    extra = dict(diffusivity_m2_s=dev(cp, iface)) if mode == "constant" else dict(interface_conductivity_m_s=dev(cp, iface), diffusion_coefficient=0.3)
    spec = sp.spatial_layer_spec(dev(cp, edges), theta_residual=dev(cp, p["res"]), theta_field_capacity=dev(cp, p["fc"]), theta_saturation=dev(cp, p["sat"]), theta_evaporation_stress=0.5,
                                 diffusion_mode=mode, bottom_drainage=bottom, n_cells=n, **extra, **kw)
    gpu = sp.prepare_spatial_column(spec, dev(cp, mask), 4.0, backend="cuda", memory_budget_bytes=BIG)
    cpu = a2.prepare(a2.make_spec(p, mode=mode, bottom=bottom, edges=edges, iface=iface, **kw), mask=mask)
    return gpu, cpu


def forcing(rng, p, mask):
    L, N = p["res"].shape
    return rng.uniform(0, 1e-3, (L, N)) * 0.3 * mask, rng.uniform(0, 2e-7, N) * mask


def both(cp, gpu, cpu, water, dt, added, pet, share, pulse):
    sg = sp.spatial_column_state(gpu, dev(cp, water), 0.0, memory_budget_bytes=BIG)
    sc = sp.spatial_column_state(cpu, water, 0.0, memory_budget_bytes=BIG)
    out = []
    for owner, st, to in ((gpu, sg, lambda a: dev(cp, a)), (cpu, sc, lambda a: a)):
        try:
            if pulse is None:
                out.append(sp.spatial_step(owner, st, dt, added_water_m=to(added), potential_evaporation_m_s=to(pet), evaporation_share=None if share is None else to(share), memory_budget_bytes=BIG))
            else:
                out.append(sp.spatial_step_with_recharge(owner, st, dt, recharge_depth_m=pulse, potential_evaporation_m_s=to(pet), evaporation_share=None if share is None else to(share), memory_budget_bytes=BIG))
        except ss.SubsurfaceError as exc:
            out.append((type(exc).__name__, getattr(exc, "bit", None), getattr(exc, "n_flagged", None)))
    return out[0], out[1]


def assert_same(g, c, bitwise=False):
    if isinstance(c, tuple):
        assert g == c
        return
    kw = dict(rtol=0, atol=0) if bitwise else dict(rtol=RT, atol=AT)
    for f in ("surface_return_m", "drainage_m", "evaporation_m", "evaporation_unmet_m", "percolation_m", "diffusion_m"):
        np.testing.assert_allclose(host(getattr(g, f)), getattr(c, f), err_msg=f, **kw)
    np.testing.assert_allclose(host(g.state.water_m), c.state.water_m, **kw)
    assert g.state.t_s == c.state.t_s and g.budget.n_terms == c.budget.n_terms


@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("bottom", ["open", "closed"])
@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("N", [1, 127, 128, 129, 257, 1001])
def test_device_step_matches_the_numpy_spatial_path_over_sizes_modes_bottoms(cp, L, N, mode, bottom):
    rng = np.random.default_rng(N + L)
    p = a2.params(rng, L, N, mode=mode)
    mask = rng.random(N) < 0.8
    mask[0] = True
    gpu, cpu = owners(cp, p, mode=mode, bottom=bottom, mask=mask)
    for kind in ("above_fc", "mixed"):
        water = a2.water_kind(rng, p, kind)
        added, pet = forcing(rng, p, mask)
        for dt, share, pulse in ((0.0, None, None), (3600.0, rng.random(N), None), (86400.0, None, 2e-3), (86400.0, rng.random(N), 40.0)):
            g, c = both(cp, gpu, cpu, water, dt, added, pet, share, pulse)
            assert_same(g, c)


def test_zero_diffusivity_is_bitwise_and_an_all_inactive_domain_is_the_identity(cp):
    rng = np.random.default_rng(5)
    N = 300
    p = a2.params(rng, 4, N)
    p["iface"] = np.zeros_like(p["iface"])
    mask = rng.random(N) < 0.7
    gpu, cpu = owners(cp, p, mask=mask)
    water = a2.water_kind(rng, p, "above_fc")
    added, pet = forcing(rng, p, mask)
    g, c = both(cp, gpu, cpu, water, 86400.0, added, pet, rng.random(N), None)
    assert_same(g, c, bitwise=True)
    gpu0, cpu0 = owners(cp, p, mask=np.zeros(N, bool))
    g0, c0 = both(cp, gpu0, cpu0, water, 3600.0, np.zeros((4, N)), np.zeros(N), None, None)
    np.testing.assert_array_equal(host(g0.state.water_m).view(np.uint64), water.view(np.uint64))
    assert g0.budget.storage_before_m3 == g0.budget.storage_after_m3 == 0.0


def test_each_column_equals_the_accepted_common_cuda_column(cp):
    """Independent control: every cell of a heterogeneous spatial device step against the accepted common CUDA column (N = 1) built from that cell's explicit parameters."""
    rng = np.random.default_rng(9)
    L, N = 3, 20
    p = a2.params(rng, L, N)
    gpu, _ = owners(cp, p)
    water = a2.water_kind(rng, p, "above_fc")
    added, pet = forcing(rng, p, np.ones(N))
    sg = sp.spatial_column_state(gpu, dev(cp, water), 0.0, memory_budget_bytes=BIG)
    got = sp.spatial_step(gpu, sg, 3600.0, added_water_m=dev(cp, added), potential_evaporation_m_s=dev(cp, pet), memory_budget_bytes=BIG)
    for c in range(N):
        lay = ss.soil_layers(p["edges"][:, c], theta_residual=p["res"][:, c], theta_field_capacity=p["fc"][:, c], theta_saturation=p["sat"][:, c], theta_evaporation_stress=0.5,
                             diffusivity_m2_s=p["iface"][:, c], bottom_drainage="closed")
        col = ss.prepare_column(lay, cp.ones(1, dtype=cp.bool_), 4.0, backend="cuda")
        st = ss.column_state(col, dev(cp, water[:, c : c + 1]), 0.0)
        ref = ss.step(col, st, 3600.0, added_water_m=dev(cp, added[:, c : c + 1]), potential_evaporation_m_s=dev(cp, pet[c : c + 1]))
        np.testing.assert_allclose(host(got.state.water_m)[:, c : c + 1], host(ref.state.water_m), rtol=RT, atol=AT)
        np.testing.assert_allclose(host(got.diffusion_m)[:, c : c + 1], host(ref.diffusion_m), rtol=RT, atol=AT)


@pytest.mark.parametrize("name,mutate", [
    ("nonfinite_water", lambda w, a, e: (np.put(w, 7, np.nan), a, e)), ("negative_water", lambda w, a, e: (np.put(w, 9, -1e-3), a, e)),
    ("nonfinite_added", lambda w, a, e: (w, np.put(a, 3, np.inf), e)), ("negative_pet", lambda w, a, e: (w, a, np.put(e, 2, -1e-9))),
    ("two_cells", lambda w, a, e: (np.put(w, [1, 200], [np.nan, -1.0]), a, e))])
def test_invalid_inputs_raise_the_identical_typed_error_bit_and_count(cp, name, mutate):
    rng = np.random.default_rng(1)
    N = 260
    p = a2.params(rng, 3, N)
    gpu, cpu = owners(cp, p)
    water = a2.water_kind(rng, p, "random")
    added, pet = forcing(rng, p, np.ones(N))
    water, added, pet = water.copy(), added.copy(), pet.copy()
    mutate(water, added, pet)
    g, c = both(cp, gpu, cpu, water, 3600.0, added, pet, None, None)
    assert isinstance(c, tuple) and g == c


def test_pulse_fit_failure_and_uniform_moisture_storage_helpers(cp):
    rng = np.random.default_rng(2)
    N = 257
    p = a2.params(rng, 3, N)
    mask = rng.random(N) < 0.6
    gpu, cpu = owners(cp, p, mask=mask)
    water = a2.water_kind(rng, p, "random")
    g, c = both(cp, gpu, cpu, water, 86400.0, None, np.zeros(N), None, 50.0)
    assert g == c and g[0] == "SubsurfaceFitFailure"
    ug, uc = sp.spatial_uniform_state(gpu, [0.0, 0.03, 0.05], 2.0, memory_budget_bytes=BIG), sp.spatial_uniform_state(cpu, [0.0, 0.03, 0.05], 2.0, memory_budget_bytes=BIG)
    np.testing.assert_array_equal(host(ug.water_m).view(np.uint64), uc.water_m.view(np.uint64))
    np.testing.assert_array_equal(host(sp.spatial_moisture(gpu, ug, memory_budget_bytes=BIG)).view(np.uint64), sp.spatial_moisture(cpu, uc, memory_budget_bytes=BIG).view(np.uint64))
    assert sp.spatial_storage_m3(gpu, ug, memory_budget_bytes=BIG) == pytest.approx(sp.spatial_storage_m3(cpu, uc, memory_budget_bytes=BIG), rel=1e-12)
    with pytest.raises(ss.SubsurfaceError, match=r"theta_by_layer\[1\]"):
        sp.spatial_uniform_state(gpu, [0.0, 0.99, 0.05], 0.0, memory_budget_bytes=BIG)


def test_actual_pool_requests_used_bytes_and_plateau_account_for_reused_split_chunks(cp):
    """Root correction 2. `pool.total_bytes()` growth is NOT a lower bound of the retained outputs: a deliberately retained SPLIT free chunk is reused, so growth can be smaller than the payload. The model is
    checked against (a) the bytes REQUESTED from the allocator during the call (exactly the model), (b) the `used_bytes` delta, which equals the charged (512-byte-rounded) unique output blocks, and (c) an
    upper bound on growth; the plateau uses a resident zero forcing allocated OUTSIDE the loop and explicit parentheses."""
    rng = np.random.default_rng(4)
    L, N = 4, 50_000
    p = a2.params(rng, L, N)
    for k in ("edges", "iface"):  # identical columns, so that the shared 1-D inputs below describe the same physics as p
        p[k] = np.repeat(p[k][:, :1], N, axis=1)
    gpu, _ = owners(cp, p, edges=p["edges"][:, 0].copy(), iface=p["iface"][:, 0].copy())  # SHARED maps: the hoisted factor kernel (F factor arrays) is part of the model
    water = dev(cp, a2.water_kind(rng, p, "random"))
    added, pet = dev(cp, forcing(rng, p, np.ones(N))[0]), dev(cp, forcing(rng, p, np.ones(N))[1])
    zero = cp.zeros((L, N), dtype=cp.float64)  # resident zero forcing for the plateau loop (no caller allocation inside the loop)
    st = sp.spatial_column_state(gpu, water, 0.0, memory_budget_bytes=BIG)
    sp.spatial_step(gpu, st, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)  # warm: module load, pool
    cp.cuda.Stream.null.synchronize()
    gc.collect()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    seed = cp.empty(1 << 20, dtype=cp.uint8)  # deliberately leave a retained, reusable SPLIT free chunk beside a live anchor
    del seed
    anchor = cp.empty(512, dtype=cp.uint8)
    cp.cuda.Stream.null.synchronize()
    before = dict(total=pool.total_bytes(), used=pool.used_bytes(), free=pool.free_bytes())
    est = sp.estimate_spatial_operation_memory("step", L, N, backend="cuda", hoisted_factor_arrays=sp._factor_arrays(gpu))
    requests = []

    def alloc(n):
        requests.append(int(n))
        return pool.malloc(n)

    with cp.cuda.using_allocator(alloc):
        step = sp.spatial_step(gpu, st, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    cp.cuda.Stream.null.synchronize()
    after = dict(total=pool.total_bytes(), used=pool.used_bytes(), free=pool.free_bytes())
    outputs = [step.state.water_m, step.percolation_m, step.diffusion_m, step.surface_return_m, step.drainage_m, step.evaporation_m, step.evaporation_unmet_m]
    assert len({a.data.ptr for a in outputs}) == len(outputs)
    payload = sum(int(a.nbytes) for a in outputs)
    charged = sum(int(a.data.mem.size) for a in outputs)  # the pool's rounded charge of each unique output block
    assert payload == est.output_bytes
    assert sum(requests) == est.device_required_bytes  # the requested payload is EXACTLY the model (outputs + released scratch + packet + factors)
    assert after["used"] - before["used"] == charged and payload <= charged <= payload + 512 * len(outputs)  # live charge of the unique outputs, plus rounding
    assert after["total"] - before["total"] <= est.device_required_bytes + 512 * len(requests)  # growth has an upper bound only: reused free chunks are not lost model output
    del step, outputs, anchor
    gc.collect()
    totals = []
    for _ in range(8):
        st = sp.spatial_step(gpu, st, 600.0, added_water_m=zero, potential_evaporation_m_s=pet, memory_budget_bytes=BIG).state
        cp.cuda.Stream.null.synchronize()
        totals.append((pool.total_bytes(), pool.used_bytes()))
    assert totals[-1][0] <= (totals[2][0] + (1 << 20)) and totals[-1][1] <= totals[2][1] + (1 << 20)  # plateau (parenthesised): no per-step accumulation


def test_device_guards_other_device_arrays_and_numpy_arrays_are_refused_without_transfer(cp):
    rng = np.random.default_rng(6)
    p = a2.params(rng, 3, 64)
    gpu, _ = owners(cp, p)
    water = a2.water_kind(rng, p, "random")
    with pytest.raises(sp.SpatialLayerError, match="cupy.ndarray"):
        sp.spatial_column_state(gpu, water, 0.0, memory_budget_bytes=BIG)
    if cp.cuda.runtime.getDeviceCount() > 1:
        with cp.cuda.Device(1):
            other = cp.asarray(water)
        with pytest.raises(sp.SpatialLayerError, match="device 1"):
            sp.spatial_column_state(gpu, other, 0.0, memory_budget_bytes=BIG)
        with cp.cuda.Device(1), pytest.raises(sp.SpatialLayerError, match="current device"):
            sp.spatial_column_state(gpu, dev(cp, water), 0.0, memory_budget_bytes=BIG)
