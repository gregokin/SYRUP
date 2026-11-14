"""REAL CUDA definitions of the S0 vegetation-law preparation: actual CuPy, a genuine device `FloatingSoilColumn`, a genuine LEAF P1 projection, the native CPU law as the control. AUTHORED, NEVER RUN by
its author: every test is skipped unless Root explicitly sets `SYRUP_VEG_BINDING_RELEASE_GPU=1` (and, optionally, `SYRUP_VEG_BINDING_GPU_DEVICE=<id>`, default 0; GPU 0 is the user allocation: check it is
free first). Nothing here imports CuPy at collection time. Run by Root only:

    SYRUP_VEG_BINDING_RELEASE_GPU=1 SYRUP_VEG_BINDING_GPU_DEVICE=0 python -B -m pytest -o pythonpath= -q tests/phase25_vegetation_binding/test_cuda_hardware.py -p no:cacheprovider

The host-compiled double in test_cuda_double.py is NOT hardware evidence; these definitions are the hardware lane."""
from __future__ import annotations

import os

import numpy as np
import pytest
from veg_binding_support import (
    A1,
    A2,
    BIG,
    EDGES,
    full_source,
    helper_pin,
    leaf_pin,
    two_species,
)

from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import water_vegetation_binding as wvb

ERR = wvb.VegetationBindingError
K, PSI = 2e-6, 0.3
FLAG = "SYRUP_VEG_BINDING_RELEASE_GPU"
pytestmark = pytest.mark.skipif(not os.environ.get(FLAG), reason=f"DEFINITION ONLY: Root sets {FLAG}=1 on real hardware (never run by the author; GPU 0 is the user allocation)")


@pytest.fixture(scope="module")
def cp():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.fail(f"{FLAG} is set but CuPy / a CUDA device is unavailable; this is a setup failure, not a skip")
    mod = backend.cupy_module()
    mod.cuda.Device(int(os.environ.get("SYRUP_VEG_BINDING_GPU_DEVICE", "0"))).use()
    return mod


@pytest.fixture()
def traffic(monkeypatch):
    """Counts the MAPLE `to_host` / `to_device` helpers (the transfers the native floating owners and `stream_hash` use); a hot step must add none beyond its declared packet."""
    import maple.core.backend as mb
    t = {"d2h": 0, "d2h_bytes": 0, "h2d": 0, "h2d_bytes": 0}
    real_h, real_d = mb.to_host, mb.to_device

    def to_host(a):
        out = real_h(a)
        t["d2h"] += 1
        t["d2h_bytes"] += int(np.asarray(out).nbytes)
        return out

    def to_device(a, xp):
        t["h2d"] += 1
        t["h2d_bytes"] += int(np.asarray(a).nbytes)
        return real_d(a, xp)

    monkeypatch.setattr(mb, "to_host", to_host)
    monkeypatch.setattr(mb, "to_device", to_device)
    return t


def controls(**kw):
    base = {"a1": A1, "a2": A2, "a3": 0.4, "max_fraction_per_hour": 0.7, "max_intake_rate_m_s": 1e-5, "background_receiver_weights": [0.0, 1.0, 1.0], "n_layers": 3, "n_cells": 9, "max_bytes": BIG}
    base.update(kw)
    return wvb.site_controls(**base)


def bound(leaf, mask=None, **ckw):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 1, 1], mask=mask)
    m = np.ones((3, 3), dtype=bool) if mask is None else mask
    b = wvb.bind_projection(p, controls(**ckw), expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=p.identity_record()["content_sha256"], expected_source=full_source(p),
                            expected_external_helpers=helper_pin(), active_mask=m, current_surface=surf, max_bytes=BIG)
    return p, b, m


def columns(cp, m):
    """The same inputs as a genuine NumPy column and a genuine CUDA column of the current device."""
    edges, depth, L = np.asarray(EDGES, dtype=np.float64), np.full(9, EDGES[-1]), len(EDGES) - 1
    act = m.reshape(-1)
    common = {"theta_evaporation_stress": 0.2, "diffusion_mode": "constant", "bottom_drainage": "open", "cell_area_m2": 1.0}
    cpu = fsl.prepare_floating_column(edges, theta_residual=0.1, theta_field_capacity=0.3, theta_saturation=0.5, soil_depth_m=depth, active_mask=act, diffusivity_m2_s=np.zeros(L - 1), **common)
    gpu = fsl.prepare_floating_column(edges, theta_residual=0.1, theta_field_capacity=0.3, theta_saturation=0.5, soil_depth_m=cp.asarray(depth), active_mask=cp.asarray(act), backend="cuda",
                                      diffusivity_m2_s=cp.asarray(np.zeros(L - 1)), **common)
    return cpu, gpu


def gprep(gpu, b, m, **kw):
    args = {"conductivity_m_s": K, "suction_m": PSI, "memory_budget_bytes": BIG, "host_memory_budget_bytes": BIG, "host_active_mask": m}
    args.update(kw)
    return wvb.prepare_vegetation_law(gpu, b, **args)


def est_for(p, b, **kw):
    args = {"binding_resident_bytes": b.resident_bytes + b.borrowed_bytes, "backend": "cuda", "stream_per_next_held_bytes": p.stream_estimate(previous_held=True)["per_next_bytes"],
            "stream_per_next_free_bytes": p.stream_estimate(previous_held=False)["per_next_bytes"],
            "stream_weights_stage_bytes": p.stream_estimate(previous_held=False)["weights_stage_bytes"]}
    args.update(kw)
    return wvb.estimate_law_bytes(3, 9, len(b.enabled_species), **args)


def test_law_and_native_steps_match_the_cpu_control(leaf, cp):
    _p, b, m = bound(leaf)
    cpu, gpu = columns(cp, m)
    a = wvb.prepare_vegetation_law(cpu, b, conductivity_m_s=K, suction_m=PSI, memory_budget_bytes=BIG)
    r = gprep(gpu, b, m)
    assert r.backend == "cuda" and r.device_id == int(cp.cuda.Device().id) and r.g_sha256 == a.g_sha256 and r.n_contributions == a.n_contributions == 2
    assert np.array_equal(cp.asnumpy(r.matrix_op._law[2]), np.asarray(a.matrix_op._law[2]))
    assert np.allclose(cp.asnumpy(r.preferential_op._law[2]), a.preferential_op._law[2], rtol=1e-12, atol=1e-15)
    assert np.allclose(cp.asnumpy(r.preferential_op._law[0]), a.preferential_op._law[0], rtol=1e-12, atol=1e-18)
    water = np.full((3, 9), 0.01)
    surface, dt = np.full(9, 0.004), 600.0
    s_cpu, s_gpu = fsl.floating_continued_state(cpu, np.ascontiguousarray(water), 0.0), fsl.floating_continued_state(gpu, cp.asarray(water), 0.0)
    pa = fpi.floating_preferential_step(a.preferential_op, surface, s_cpu, dt)
    pb = fpi.floating_preferential_step(r.preferential_op, cp.asarray(surface), s_gpu, dt)
    assert np.allclose(cp.asnumpy(pb.state.water_m), pa.state.water_m, rtol=1e-12, atol=1e-15) and np.allclose(cp.asnumpy(pb.surface_water_m), pa.surface_water_m, rtol=1e-12, atol=1e-15)
    z = np.zeros(9)
    ha = fm.floating_matrix_history(a.matrix_op, wetting_m=z.copy(), cumulative_intake_m=z.copy())
    hb = fm.floating_matrix_history(r.matrix_op, wetting_m=cp.asarray(z), cumulative_intake_m=cp.asarray(z))
    ma = fm.floating_matrix_step(a.matrix_op, surface, surface, np.full(9, 1e-5), s_cpu, ha, dt)
    mb_ = fm.floating_matrix_step(r.matrix_op, cp.asarray(surface), cp.asarray(surface), cp.asarray(np.full(9, 1e-5)), s_gpu, hb, dt)
    assert np.allclose(cp.asnumpy(mb_.intake_m), ma.intake_m, rtol=1e-12, atol=1e-15)


def test_correct_and_wrong_host_mask_against_the_device_identity(leaf, cp):
    _p, b, m = bound(leaf)
    _cpu, gpu = columns(cp, m)
    assert gprep(gpu, b, m).n_contributions == 2
    other = m.copy()
    other[0, 0] = False
    with pytest.raises(ERR, match="host active mask"):
        gprep(gpu, b, other)                                             # not the binding's mask
    _p2, b2, _m2 = bound(leaf, mask=other)
    with pytest.raises(ERR, match="device mask identity"):
        gprep(gpu, b2, other)                                            # the binding's own mask, but the device column carries another


def test_requested_device_mismatch_is_refused_before_any_transfer(leaf, cp, traffic):
    if int(cp.cuda.runtime.getDeviceCount()) < 2:
        pytest.skip("a device-mismatch control needs two visible devices")
    _p, b, m = bound(leaf)
    _cpu, gpu = columns(cp, m)
    own = int(cp.cuda.Device().id)
    other = 1 if own == 0 else 0
    cp.cuda.Device(other).use()
    try:
        before = dict(traffic)
        with pytest.raises(Exception, match="device"):
            gprep(gpu, b, m)
        assert traffic == before                                          # refused by the current-device gate: no transfer, no reader call
    finally:
        cp.cuda.Device(own).use()


def test_dual_pool_need_minus_one_exact_reserve_and_readers(leaf, cp, traffic):
    p, b, m = bound(leaf)
    _cpu, gpu = columns(cp, m)
    est = est_for(p, b)
    host_extra = 65536
    before = dict(traffic)
    with pytest.raises(Exception, match="device side"):
        gprep(gpu, b, m, memory_budget_bytes=est["peak_bytes"] - 1)
    with pytest.raises(Exception, match="host side"):
        gprep(gpu, b, m, host_memory_budget_bytes=est["peak_host_bytes"] + host_extra - 1)
    with pytest.raises(Exception, match="device side"):
        gprep(gpu, b, m, memory_budget_bytes=est["peak_bytes"] + 9, reserve_bytes=10)
    with pytest.raises(Exception, match="host side"):
        gprep(gpu, b, m, host_memory_budget_bytes=est["peak_host_bytes"] + host_extra + 9, host_reserve_bytes=10)
    with pytest.raises(Exception, match="device side"):
        gprep(gpu, b, m, available_bytes_fn=lambda: est["peak_bytes"] - 1)
    with pytest.raises(Exception, match="host side"):
        gprep(gpu, b, m, host_available_bytes_fn=lambda: est["peak_host_bytes"] + host_extra - 1)
    assert traffic == before                                              # every refusal precedes any transfer
    assert gprep(gpu, b, m, memory_budget_bytes=est["peak_bytes"], host_memory_budget_bytes=est["peak_host_bytes"] + host_extra).n_contributions == 2


def test_cold_transfers_are_truthful_and_a_hot_step_transfers_only_its_packet(leaf, cp, traffic):
    _p, b, m = bound(leaf, a3=np.linspace(0.1, 0.9, 9))
    _cpu, gpu = columns(cp, m)
    r = gprep(gpu, b, m)
    est = r.memory["declared"]
    assert "d2h_bytes" not in est and est["d2h_adapter_bytes"] == 9 and traffic["d2h_bytes"] >= est["d2h_adapter_bytes"]    # the mask read-back is a real D2H
    assert est["d2h_native_identity_bytes"] > 0 and est["d2h_modelled_bytes"] == est["d2h_adapter_bytes"] + est["d2h_native_identity_bytes"]
    from maple_syrup import floating_preferential_infiltration_cuda as fpc
    water = np.full((3, 9), 0.01)
    s_gpu = fsl.floating_continued_state(gpu, cp.asarray(water), 0.0)
    surface = cp.asarray(np.full(9, 0.004))
    cold = dict(traffic)
    fpi.floating_preferential_step(r.preferential_op, surface, s_gpu, 600.0)
    assert traffic["h2d"] == cold["h2d"]                                  # no upload in a hot step
    assert traffic["d2h_bytes"] - cold["d2h_bytes"] <= int(fpc.operation_terms("step", 3, 9)["host"])   # only the declared packet is read back
    assert r.step_taken is False and r.water_head_authority is False
