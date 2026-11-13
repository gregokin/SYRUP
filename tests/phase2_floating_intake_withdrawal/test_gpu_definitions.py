"""Actual-GPU test DEFINITIONS for the floating intake/withdrawal kernels (WRITTEN, NOT RUN by their author: no GPU was launched or probed, no CuPy was imported or initialised, no NVRTC compile happened, no nvidia-smi).
Opt-in (SYRUP_RELEASE_GPU=1) and skipped without a usable CuPy device; Root runs them under its device guard (GPU0, fresh 0 MiB / 0 %). They assert, through the ACTUAL MAPLE transfer counters and the CuPy memory pool:
numerical equivalence with the NumPy reference (rtol 2e-12 / atol 1e-14; the withdrawal is exact), zero host-to-device and ONE fixed device-to-host packet per hot call (matrix 136 B, preferential 72 B, withdrawal 64 B,
availability 8 B), allocation = the payload model, pool plateau and release at N = 50003, the compiled-kernel resource attributes (registers, LOCAL bytes, static shared bytes, max threads; written to a JSON report for
Root) independent of the layer count (3/5/65), the device gate with a second device, and the explicit history rebinding. Hardware equivalence of the CUDA definitions with the host double is NOT assumed."""
from __future__ import annotations

import gc
import json
import os

import numpy as np
import pytest

from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_matrix_infiltration_cuda as fmc
from maple_syrup import floating_preferential_infiltration as fp
from maple_syrup import floating_preferential_infiltration_cuda as fpc
from maple_syrup import floating_soil_layers as fl
from maple_syrup import floating_transpiration as ft
from maple_syrup import floating_transpiration_cuda as ftc

INF = float("inf")


@pytest.fixture(scope="module")
def cp():
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the floating intake/withdrawal kernels are not exercised (no claim)")
    return backend.cupy_module()


def dev(cp, a):
    return cp.asarray(np.ascontiguousarray(a)) if isinstance(a, np.ndarray) else a


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def build(cp, edges, depth, *, tr=0.05, tf=0.3, ts=0.45, active=None, backend="cuda", area=1.0):
    depth = np.asarray(depth, np.float64)
    N, L = depth.size, len(edges) - 1
    act = np.ones(N, bool) if active is None else active
    conv = (lambda x: dev(cp, x)) if backend == "cuda" else (lambda x: x)
    return fl.prepare_floating_column(np.asarray(edges, np.float64), theta_residual=conv(tr), theta_field_capacity=conv(tf), theta_saturation=conv(ts), theta_evaporation_stress=0.2,
                                      diffusion_mode="constant", diffusivity_m2_s=conv(np.zeros(L - 1)), bottom_drainage="open", soil_depth_m=conv(depth), active_mask=conv(act), cell_area_m2=area,
                                      backend=backend)


def water_for(cpu, rng, frac=0.8):
    h = np.stack([fl.porous_thickness(cpu, k) for k in range(cpu.n_layers)])
    return h * 0.45 * rng.uniform(0.05, frac, h.shape)


def forcing(rng, N, act):
    pool = rng.uniform(0.0, 0.02, N) * act
    standing = np.where(rng.random(N) > 0.5, rng.uniform(0.0, 0.01, N), 0.0) * act
    rate = np.where(standing > 0.0, rng.uniform(0.0, 1e-5, N), pool / 600.0 * rng.uniform(1.0, 3.0, N)) * act
    return pool, standing, rate


@pytest.mark.parametrize("L", [1, 3, 5, 65])
@pytest.mark.parametrize("N", [1, 127, 128, 129, 257])
def test_all_three_laws_equal_the_numpy_reference(cp, L, N):
    rng = np.random.default_rng(L * 1000 + N)
    edges = np.linspace(0.0, 0.6, L + 1)
    depth = rng.choice([0.0, 1e-6, 0.003, 0.1, 0.31, 0.45, 0.6, 0.9], size=N)
    act = rng.random(N) > 0.15
    cpu, gpu = build(cp, edges, depth, active=act, backend="numpy"), build(cp, edges, depth, active=act)
    w = water_for(cpu, rng, 0.9)
    w[:, ~act] = 3.0
    s_cpu, s_gpu = fl.floating_continued_state(cpu, w), fl.floating_continued_state(gpu, dev(cp, w))
    # withdrawal (exact) and availability
    q = rng.uniform(0.0, 0.01, (L, N)) * act
    a, b = ft.floating_withdraw(cpu, s_cpu, q), ft.floating_withdraw(gpu, s_gpu, dev(cp, q))
    assert np.array_equal(host(b.state.water_m), a.state.water_m) and np.array_equal(host(b.actual_m), a.actual_m)
    assert np.array_equal(host(ft.floating_layer_availability_m(gpu, s_gpu)), ft.floating_layer_availability_m(cpu, s_cpu))
    # matrix
    pool, standing, rate = forcing(rng, N, act)
    oa = fm.prepare_floating_matrix(cpu, conductivity_m_s=2e-6, suction_m=0.15, vegetation_factor=0.8)
    ob = fm.prepare_floating_matrix(gpu, conductivity_m_s=2e-6, suction_m=0.15, vegetation_factor=0.8)
    j0, c0 = rng.uniform(0.0, 0.02, N), rng.uniform(0.0, 0.5, N)
    ra = fm.floating_matrix_step(oa, pool, standing, rate, s_cpu, fm.floating_matrix_history(oa, wetting_m=j0, cumulative_intake_m=c0), 600.0)
    rb = fm.floating_matrix_step(ob, dev(cp, pool), dev(cp, standing), dev(cp, rate), s_gpu, fm.floating_matrix_history(ob, wetting_m=dev(cp, j0), cumulative_intake_m=dev(cp, c0)), 600.0)
    for x, y in ((ra.state.water_m, rb.state.water_m), (ra.surface_pool_m, rb.surface_pool_m), (ra.history.wetting_m, rb.history.wetting_m), (ra.intake_m, rb.intake_m)):
        np.testing.assert_allclose(host(y), x, rtol=2e-12, atol=1e-14)
    # preferential
    weights = np.linspace(0.0, 1.0, L)
    pa = fp.prepare_floating_preferential(cpu, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=weights)
    pb = fp.prepare_floating_preferential(gpu, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=weights)
    surf = rng.uniform(0.0, 0.02, N) * act
    sa = fp.floating_preferential_step(pa, surf, s_cpu, 1800.0)
    sb = fp.floating_preferential_step(pb, dev(cp, surf), s_gpu, 1800.0)
    for x, y in ((sa.state.water_m, sb.state.water_m), (sa.surface_water_m, sb.surface_water_m), (sa.intake_m, sb.intake_m)):
        np.testing.assert_allclose(host(y), x, rtol=2e-12, atol=1e-14)
    assert np.array_equal(host(sb.state.water_m)[:, ~act], w[:, ~act])  # placeholder bits


def _plateau(cp, call, bytes_model, d2h, name):
    from maple.core.backend import read_transfer_counters

    call()  # warm: NVRTC compile, module load
    cp.cuda.Stream.null.synchronize()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    base_used = pool.used_bytes()
    t0 = read_transfer_counters()
    keep = call()
    d = read_transfer_counters().delta(t0)
    grown = pool.used_bytes() - base_used
    assert (d.device_to_host, d.device_to_host_bytes, d.host_to_device) == (1, d2h, 0), name  # ONE fixed packet, no upload
    assert grown <= bytes_model + 6 * 512, name  # the payload model (+ allocator rounding slack per array)
    totals = []
    for _ in range(6):
        keep = call()
        totals.append(pool.total_bytes())
    assert totals[-1] <= totals[1], name  # plateau
    del keep
    gc.collect()
    pool.free_all_blocks()
    assert pool.used_bytes() <= base_used, name  # release


def test_hot_traffic_allocation_plateau_and_release_at_n_50003(cp):
    rng = np.random.default_rng(50003)
    N, L = 50003, 3
    edges = [0.0, 0.1, 0.3, 0.6]
    depth = rng.choice([0.0, 0.05, 0.2, 0.45, 0.9], size=N)
    cpu, gpu = build(cp, edges, depth, backend="numpy"), build(cp, edges, depth)
    w = water_for(cpu, rng, 0.6)
    s = fl.floating_continued_state(gpu, dev(cp, w))
    act = np.ones(N, bool)
    q = dev(cp, rng.uniform(0.0, 0.005, (L, N)))
    _plateau(cp, lambda: ft.floating_withdraw(gpu, s, q), ftc.operation_terms("withdraw", L, N)["outputs"] + ftc.operation_terms("withdraw", L, N)["scratch"], 64, "withdraw")
    pool, standing, rate = (dev(cp, x) for x in forcing(rng, N, act))
    op = fm.prepare_floating_matrix(gpu, conductivity_m_s=2e-6, suction_m=0.15, vegetation_factor=0.8)
    h = fm.floating_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0)
    t = fmc.operation_terms("step", L, N)
    _plateau(cp, lambda: fm.floating_matrix_step(op, pool, standing, rate, s, h, 600.0), t["outputs"] + t["scratch"], fmc.RAW_PACKET_BYTES, "matrix")
    pop = fp.prepare_floating_preferential(gpu, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([0.0, 1.0, 1.0]))
    surf = dev(cp, rng.uniform(0.0, 0.02, N))
    t = fpc.operation_terms("step", L, N)
    _plateau(cp, lambda: fp.floating_preferential_step(pop, surf, s, 1800.0), t["outputs"] + t["scratch"], fpc.RAW_PACKET_BYTES, "preferential")


def test_kernel_resource_attributes_are_reported_and_independent_of_the_layer_count(cp, tmp_path):
    report = {}
    for L in (3, 5, 65):
        edges = np.linspace(0.0, 0.6, L + 1)
        g = build(cp, edges, [0.3, 0.7])
        s = fl.floating_continued_state(g, dev(cp, np.zeros((L, 2))))
        ft.floating_withdraw(g, s, dev(cp, np.zeros((L, 2))))
        op = fm.prepare_floating_matrix(g, conductivity_m_s=2e-6, suction_m=0.1, vegetation_factor=0.5)
        fm.floating_matrix_step(op, dev(cp, np.zeros(2)), dev(cp, np.zeros(2)), dev(cp, np.zeros(2)), s, fm.floating_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0), 10.0)
        pop = fp.prepare_floating_preferential(g, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=np.ones(L))
        fp.floating_preferential_step(pop, dev(cp, np.zeros(2)), s, 10.0)
        keep = ("num_regs", "local_size_bytes", "shared_size_bytes", "max_threads_per_block")
        report[L] = {}
        for tag, ns in (("withdraw", ftc.load_kernels(cp, g.device_id)), ("matrix", fmc.load_kernels(cp, g.device_id)), ("preferential", fpc.load_kernels(cp, g.device_id))):
            report[L][tag] = {n: {k: int(v) for k, v in dict(fn.attributes).items() if k in keep} for n, fn in vars(ns).items()}
    (tmp_path / "floating_intake_withdrawal_kernel_resources.json").write_text(json.dumps({"attributes": report, "withdraw": ftc.kernel_provenance(), "matrix": fmc.kernel_provenance(),
                                                                                         "preferential": fpc.kernel_provenance()}, indent=1))
    print(json.dumps(report[3]))
    assert report[3] == report[5] == report[65]  # the compiled kernels do not depend on L
    for law in report[3].values():
        assert all(v["max_threads_per_block"] >= 128 for v in law.values())


def test_device_gate_with_a_second_device_precedes_readers_and_copies(cp):
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs two visible devices (no claim)")
    g = build(cp, [0.0, 0.1, 0.3, 0.6], [0.5, 0.2])
    s = fl.floating_continued_state(g, dev(cp, np.zeros((3, 2))))
    reads = []
    reader = lambda: reads.append(1) or (1 << 40)
    with cp.cuda.Device(1):
        for call in (lambda: ft.floating_withdraw(g, s, dev(cp, np.zeros((3, 2))), available_bytes_fn=reader), lambda: ft.floating_layer_availability_m(g, s, available_bytes_fn=reader),
                     lambda: fm.prepare_floating_matrix(g, conductivity_m_s=1e-6, suction_m=0.0, vegetation_factor=0.5, available_bytes_fn=reader),
                     lambda: fp.prepare_floating_preferential(g, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=np.ones(3), available_bytes_fn=reader)):
            with pytest.raises(fl.FloatingSoilError, match="current device"):
                call()
    assert not reads


def test_rebinding_on_hardware_retains_exact_history(cp):
    rng = np.random.default_rng(7)
    N = 1000
    edges = [0.0, 0.2, 0.5]
    depth = rng.choice([0.0, 0.05, 0.15, 0.6], size=N)
    o1 = fm.prepare_floating_matrix(build(cp, edges, depth), conductivity_m_s=2e-6, suction_m=0.1, vegetation_factor=0.5)
    o2 = fm.prepare_floating_matrix(build(cp, edges, np.minimum(depth + 0.02, 0.6), ts=0.42), conductivity_m_s=5e-6, suction_m=0.2, vegetation_factor=0.7)
    j, c = rng.uniform(0.0, 0.3, N), rng.uniform(0.0, 2.0, N)
    h1 = fm.floating_matrix_history(o1, wetting_m=dev(cp, j), cumulative_intake_m=dev(cp, c))
    rb = fm.rebind_floating_matrix_history(o1, h1, o2)
    assert not rb.reset and np.array_equal(host(rb.history.wetting_m), j) and np.array_equal(host(rb.history.cumulative_intake_m), c)
    with pytest.raises(fl.FloatingSoilError):
        fm.floating_matrix_step(o2, dev(cp, np.zeros(N)), dev(cp, np.zeros(N)), dev(cp, np.zeros(N)), fl.floating_continued_state(build(cp, edges, depth), dev(cp, np.zeros((2, N)))), h1, 1.0)
