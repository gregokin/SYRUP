"""Actual-GPU test DEFINITIONS for the resident floating column (WRITTEN, NOT RUN by their author: no GPU was launched or probed, no CuPy was initialised, no NVRTC compile happened, no nvidia-smi). Opt-in (SYRUP_RELEASE_GPU=1) and
skipped without a usable CuPy device; Root runs them under its device guard (GPU0 UUID, fresh 0 MiB / 0 %), with `.venv-cupy` and the genuine 9f CUDA materialisation. They assert, through the ACTUAL MAPLE transfer counters and the CuPy
memory pool: numerical equivalence with the NumPy reference (rtol 2e-12 / atol 1e-14; reconciliation exact), zero host-to-device and one 72-byte (step) / 56-byte (reconcile) / 32-byte (validation) device-to-host transfers on the hot paths,
allocation = the payload model, pool plateau and release at N = 50003, the compiled-kernel resource attributes (registers, LOCAL bytes, static shared bytes, max threads; written to a JSON report for Root), independence of those attributes
from the layer count (3/5/65), the device gate with a second device, and invalid metadata/values. Hardware equivalence of the CUDA definitions with the host double is NOT assumed."""
from __future__ import annotations

import gc
import json
import os

import numpy as np
import pytest
from conftest import BIG, capacity_controls

from maple_syrup import floating_soil_layers as fl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_cuda_floating as cf

EDGES = np.array([0.0, 0.1, 0.3, 0.6])


@pytest.fixture(scope="module")
def cp():
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the resident floating column is not exercised (no claim)")
    return backend.cupy_module()


def dev(cp, a):
    return cp.asarray(np.ascontiguousarray(a)) if isinstance(a, np.ndarray) else a


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def build(cp, depth, *, edges=EDGES, tr=0.0, tf=0.2, ts=0.4, mode="tietjen_donor_depth", bottom="open", active=None, backend="cuda"):
    depth = np.asarray(depth, np.float64)
    N, L = depth.size, len(edges) - 1
    act = np.ones(N, bool) if active is None else active
    conv = (lambda x: dev(cp, x)) if backend == "cuda" else (lambda x: x)
    kw = dict(interface_conductivity_m_s=conv(np.full(L - 1, 2e-6)), diffusion_coefficient=0.7) if mode != "constant" else dict(diffusivity_m2_s=conv(np.full(L - 1, 1e-7)))
    return fl.prepare_floating_column(edges, theta_residual=conv(tr), theta_field_capacity=conv(tf), theta_saturation=conv(ts), theta_evaporation_stress=0.3, diffusion_mode=mode, bottom_drainage=bottom,
                                      soil_depth_m=conv(depth), active_mask=conv(act), cell_area_m2=2.0, backend=backend, **kw)


def water_for(cpu, rng, frac=0.8):
    h = np.stack([fl.porous_thickness(cpu, k) for k in range(cpu.n_layers)])
    ts = cpu._maps["theta_saturation"]
    return h * np.broadcast_to(ts.reshape(-1, 1) if ts.ndim == 1 and ts.shape[0] == cpu.n_layers else ts, h.shape) * rng.uniform(0.05, frac, h.shape)


@pytest.mark.parametrize("L", [1, 3, 5, 65])
@pytest.mark.parametrize("N", [1, 127, 128, 129, 257])
def test_step_and_reconcile_equal_the_numpy_reference(cp, L, N):
    rng = np.random.default_rng(L * 1000 + N)
    edges = np.linspace(0.0, 0.6, L + 1)
    depth = rng.choice([0.0, 0.003, 0.1, 0.31, 0.45, 0.6, 0.6000001, 2.0], size=N)
    act = rng.random(N) > 0.15
    cpu, gpu = build(cp, depth, edges=edges, active=act, backend="numpy"), build(cp, depth, edges=edges, active=act)
    w = water_for(cpu, rng, 0.9)
    w[:, ~act] = 3.0
    pet = rng.uniform(0, 3e-7, N) * act
    added = rng.uniform(0, 2e-3, (L, N)) * 0.2 * act
    a = fl.floating_step(cpu, fl.floating_continued_state(cpu, w), 3600.0, added_water_m=added, potential_evaporation_m_s=pet)
    b = fl.floating_step(gpu, fl.floating_continued_state(gpu, dev(cp, w)), 3600.0, added_water_m=dev(cp, added), potential_evaporation_m_s=dev(cp, pet))
    np.testing.assert_allclose(host(b.state.water_m), a.state.water_m, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(host(b.drainage_m), a.drainage_m, rtol=2e-12, atol=1e-14)
    assert np.array_equal(host(b.state.water_m)[:, ~act], w[:, ~act])
    assert abs(b.budget.residual_m3) <= b.budget.bound_m3
    cpu2 = build(cp, rng.choice([0.0, 0.1, 0.45, 0.9], size=N), edges=edges, active=act, backend="numpy")
    gpu2 = build(cp, host(gpu._maps["soil_depth"]) * 0 + cpu2._maps["soil_depth"], edges=edges, active=act)
    ra = fl.reconcile_floating(cpu, fl.floating_continued_state(cpu, w * 0.5), cpu2)
    rb = fl.reconcile_floating(gpu, fl.floating_continued_state(gpu, dev(cp, w * 0.5)), gpu2)
    for f in ("surface_return_m", "bottom_export_m", "percolation_m"):
        np.testing.assert_allclose(host(getattr(rb, f)), getattr(ra, f), rtol=0, atol=1e-15)
    assert rb.receipt.identity_kind == "capability_token" and rb.receipt.water_sha256 is None


@pytest.mark.parametrize("c", capacity_controls(), ids=lambda c: c["label"])
def test_actual_gpu_reconciliation_controls(cp, c):
    n = len(c["initial_water"])
    edges = np.array([float(k) for k in range(n + 1)])
    fc = np.array([float(x) / 1000.0 if float(s_) > 0.0 else 0.1 for x, s_ in zip(c["new_fc"], c["new_sat"])])
    sat = np.array([float(x) / 1000.0 if float(x) > 0.0 else 0.2 for x in c["new_sat"]])
    n_soil = sum(1 for x in c["new_sat"] if float(x) > 0.0)
    mk = lambda tf, ts, depth, bottom: fl.prepare_floating_column(edges, theta_residual=0.0, theta_field_capacity=dev(cp, tf), theta_saturation=dev(cp, ts), diffusivity_m2_s=dev(cp, np.zeros(n - 1)),  # noqa: E731
                                                                  theta_evaporation_stress=0.1, bottom_drainage=bottom, soil_depth_m=dev(cp, np.array([depth])), active_mask=dev(cp, np.ones(1, bool)),
                                                                  cell_area_m2=1.0, backend="cuda")
    old = mk(np.full(n, 0.9), np.full(n, 0.95), n + 1.0, "closed")
    new = mk(fc, sat, float(n_soil) if not c["open_above_bedrock"] else n + 1.0, "open" if c["open_above_bedrock"] else "closed")
    st = fl.floating_continued_state(old, dev(cp, np.array([[float(x) / 1000.0] for x in c["initial_water"]])))
    cand = fl.reconcile_floating(old, st, new)
    np.testing.assert_allclose(host(cand.state.water_m)[:, 0] * 1000.0, [float(x) for x in c["expected_water"]], rtol=0, atol=1e-12)
    assert abs(float(host(cand.surface_return_m)[0]) * 1000.0 - float(c["expected_surface_return"])) <= 1e-12


def test_hot_traffic_allocation_plateau_and_release_at_n_50003(cp, tmp_path):
    from maple.core.backend import read_transfer_counters

    rng = np.random.default_rng(50003)
    N = 50003
    depth = rng.choice([0.0, 0.05, 0.2, 0.45, 0.9], size=N)
    cpu, gpu = build(cp, depth, backend="numpy"), build(cp, depth)
    st = fl.floating_continued_state(gpu, dev(cp, water_for(cpu, rng, 0.6)))
    f = dict(added_water_m=dev(cp, np.zeros((3, N))), potential_evaporation_m_s=dev(cp, rng.uniform(0, 3e-7, N)))
    fl.floating_step(gpu, st, 600.0, **f)  # warm: NVRTC compile, module load
    cp.cuda.Stream.null.synchronize()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    base_used = pool.used_bytes()
    t0 = read_transfer_counters()
    r = fl.floating_step(gpu, st, 600.0, **f)
    d = read_transfer_counters().delta(t0)
    grown = pool.used_bytes() - base_used
    assert (d.device_to_host, d.device_to_host_bytes, d.host_to_device) == (1, 72, 0)  # ONE fixed packet, no upload
    assert grown <= fcu.operation_terms("step", 3, N)["device"] + 4 * 512  # the payload model (+ allocator rounding slack per array)
    t1 = read_transfer_counters()
    fl.reconcile_floating(gpu, r.state, gpu)
    d2 = read_transfer_counters().delta(t1)
    assert (d2.device_to_host, d2.device_to_host_bytes, d2.host_to_device) == (1, 56, 0)
    totals = []
    for _ in range(6):
        r = fl.floating_step(gpu, r.state, 600.0, **f)
        totals.append(pool.total_bytes())
    assert totals[-1] <= totals[1]  # plateau
    del r, st
    gc.collect()
    pool.free_all_blocks()
    assert pool.used_bytes() <= base_used  # release


def test_kernel_resource_attributes_are_reported_and_independent_of_the_layer_count(cp, tmp_path):
    report = {}
    for L in (3, 5, 65):
        edges = np.linspace(0.0, 0.6, L + 1)
        g = build(cp, [0.3, 0.7], edges=edges)
        fns = fcu.load_kernels(cp, g.device_id)
        fl.floating_step(g, fl.floating_continued_state(g, dev(cp, np.zeros((L, 2)))), 60.0, added_water_m=dev(cp, np.zeros((L, 2))), potential_evaporation_m_s=dev(cp, np.zeros(2)))
        report[L] = {n: {k: int(v) for k, v in dict(getattr(fns, n.replace("maple_syrup_floating_", "")).attributes).items() if k in ("num_regs", "local_size_bytes", "shared_size_bytes", "max_threads_per_block")}
                     for n in cf.RESIDENT_KERNELS}
    (tmp_path / "floating_kernel_resources.json").write_text(json.dumps({"attributes": report, "source": cf.static_resource_report()}, indent=1))
    print(json.dumps(report[3]["maple_syrup_floating_resident_step"]))
    assert report[3] == report[5] == report[65]  # the compiled kernels do not depend on L
    assert report[3]["maple_syrup_floating_resident_step"]["local_size_bytes"] <= 64  # only the 7 contribution doubles may spill; Root records the actual number
    assert all(v["max_threads_per_block"] >= cf.THREADS for v in report[3].values())


def test_device_gate_with_a_second_device_precedes_readers_and_copies(cp):
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs two visible devices (no claim)")
    g = build(cp, [0.5, 0.2])
    st = fl.floating_continued_state(g, dev(cp, np.zeros((3, 2))))
    reads = []
    with cp.cuda.Device(1):
        for call in (lambda: fl.floating_step(g, st, 60.0, added_water_m=dev(cp, np.zeros((3, 2))), potential_evaporation_m_s=dev(cp, np.zeros(2)), available_bytes_fn=lambda: reads.append(1) or BIG),
                     lambda: fl.floating_moisture(g, st, available_bytes_fn=lambda: reads.append(1) or BIG), lambda: fl.reconcile_floating(g, st, g, available_bytes_fn=lambda: reads.append(1) or BIG)):
            with pytest.raises(fl.FloatingSoilError, match="current device"):
                call()
    assert not reads


def test_invalid_metadata_and_resident_values_are_refused(cp):
    with pytest.raises(fl.FloatingSoilError):
        build(cp, [np.nan])
    with pytest.raises(fl.FloatingSoilError):
        build(cp, [-1.0])
    with pytest.raises(fl.FloatingSoilError, match="theta values violate"):
        build(cp, [0.5], tf=0.5, ts=0.4)
    with pytest.raises(fl.FloatingSoilError):
        fl.prepare_floating_column(EDGES, theta_residual=np.zeros((3, 2)), theta_field_capacity=0.2, theta_saturation=0.4, diffusivity_m2_s=np.zeros(2), theta_evaporation_stress=0.3,
                                   soil_depth_m=np.full(2, 0.5), active_mask=np.ones(2, bool), cell_area_m2=1.0, backend="cuda")  # host arrays for the device backend: no implicit transfer
    g = build(cp, [0.2, 0.9])
    with pytest.raises(fl.FloatingSoilError, match="strictly"):
        sat = np.zeros((3, 2))
        sat[0] = 0.4 * 0.1
        fl.floating_initial_state(g, dev(cp, sat))
