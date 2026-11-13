"""Actual-GPU test DEFINITIONS for the floating evaporation adapter (WRITTEN, NOT RUN by their author: no GPU was launched or probed, no CuPy was initialised, no NVRTC compile happened). Opt-in
(SYRUP_RELEASE_GPU=1) and skipped without a usable CuPy device; Root runs them under its device guard. CuPy is imported only inside the fixture. They define: equivalence with the host reference (the shared law
body is elementwise: rtol 2e-12 / atol 1e-14 allowed for hardware fused operations, nothing assumed), exactly ONE counted device-to-host transfer of (PACKET_LENGTH + 1) * 8 = 96 bytes (the fixed CUDA column's
true baseline is PACKET_LENGTH * 8 = 88 bytes: the extra 8 is the device active count) and ZERO host-to-device, no full-array download, quota refusal BEFORE the law body and before any allocation (device
side then host side), the current-device gate before any reader, and wrong namespace/device/dtype/shape metadata refusals."""
# ruff: noqa: C408
from __future__ import annotations

import os

import numpy as np
import pytest
from cases import EDGES, MASK, PART, inputs

from maple_syrup import floating_evaporation as fe
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import soil_evaporation as se


@pytest.fixture(scope="module")
def cp():
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA adapter is not exercised (no claim)")
    return backend.cupy_module()


def _column(cp, depth):
    """Fixture metadata exactly as `floating_soil_layers._prepare_cuda` requires: the nominal edges are HOST NumPy (shared metadata, uploaded once), the shared retention values are explicit Python scalars (a 0-d array
    would be promoted to a (1,) vector), the interface diffusivity is a device `(L-1,)` vector, the depth and the mask are device `(N,)` arrays."""
    edges = np.ascontiguousarray(np.asarray(EDGES, dtype=np.float64))
    n = len(depth)
    return fsl.prepare_floating_column(edges, theta_residual=0.0625, theta_field_capacity=0.25, theta_saturation=0.5, diffusivity_m2_s=cp.asarray(np.zeros(len(EDGES) - 2)),
                                       theta_evaporation_stress=0.25, diffusion_mode="constant", bottom_drainage="open", soil_depth_m=cp.asarray(np.asarray(depth, np.float64)),
                                       active_mask=cp.asarray(np.ascontiguousarray(MASK[:n])), cell_area_m2=4.0, backend="cuda")


def _device_inputs(cp):
    return tuple(cp.asarray(a) for a in inputs())


def _big():
    return 1 << 40


def test_cuda_equals_the_host_reference_with_one_counted_packet_and_no_upload(cp, monkeypatch):
    from maple.core import backend as mb

    col = _column(cp, [2.0] * 7)
    op = fe.prepare_floating_evaporation(col, PART, available_bytes_fn=_big, host_available_bytes_fn=_big)
    moved = []
    real = mb.to_host

    def counted(x, *a, **k):
        out = real(x, *a, **k)
        moved.append(int(np.asarray(out).nbytes))
        return out

    monkeypatch.setattr(mb, "to_host", counted)
    p, c, s = _device_inputs(cp)
    uploads = []
    real_up = mb.to_device
    monkeypatch.setattr(mb, "to_device", lambda *a, **k: uploads.append(1) or real_up(*a, **k))
    before = [cp.asnumpy(a).copy() for a in (p, c, s)]
    pool = cp.get_default_memory_pool()
    used0 = pool.used_bytes()
    st = fe.stage_floating_evaporation(op, pet_terrain_m_s=p, cover=c, surface_water_m=s, dt_s=3600.0, available_bytes_fn=_big, host_available_bytes_fn=_big)
    assert uploads == [], "no host-to-device transfer by the staging"
    assert all(np.array_equal(cp.asnumpy(a), b) for a, b in zip((p, c, s), before, strict=True)), "inputs are never written"
    outputs = sum(int(a.nbytes) for a in (st.surface_water_after_m, st.surface_evaporation_m, st.surface_evaporation_unmet_m, st.potential_combined_soil_rate_m_s, st.soil_evaporation_share))
    assert pool.used_bytes() - used0 <= outputs + 5 * 512, "after the call only the five fresh outputs stay allocated (scratch released; pool blocks rounded to 512 B)"
    assert moved == [(se.PACKET_LENGTH + 1) * 8]  # ONE packet; the true fixed-CUDA baseline is PACKET_LENGTH * 8 = 88 B (+8: the device active count)
    ph, ch, sh = inputs()
    ref = se.stage_evaporation(se.prepare_evaporation(__import__("cases").fixed_column(), PART), pet_terrain_m_s=ph, cover=ch, surface_water_m=sh, dt_s=3600.0)
    for got, want in ((st.surface_water_after_m, ref.surface_water_after_m), (st.surface_evaporation_m, ref.surface_evaporation_m), (st.surface_evaporation_unmet_m, ref.surface_evaporation_unmet_m),
                      (st.potential_combined_soil_rate_m_s, ref.potential_combined_soil_rate_m_s), (st.soil_evaporation_share, ref.soil_evaporation_share)):
        np.testing.assert_allclose(cp.asnumpy(got), want, rtol=2e-12, atol=1e-14)
    assert st.ledger.n_active == int(MASK.sum())
    n = 7
    base = se.estimate_evaporation_memory(n, backend="cuda")
    assert st.memory.device_bytes == base.device_bytes == (3 * n + 5 * n) * 8 + se.SCRATCH_DOUBLES_PER_CELL * n * 8 + se.SCRATCH_MASK_BYTES_PER_CELL * n + 4 * se.PACKET_LENGTH * 8  # device pool: inputs + outputs + scratch
    assert st.memory.host_bytes == (se.PACKET_LENGTH + 1) * 8 + fe.INTERPRETER_ALLOWANCE_BYTES  # host pool: ONE packet (+ the device count) + the declared allowance, not a function of N
    assert st.memory.required_bytes == st.memory.device_bytes + st.memory.reserve_bytes


def test_cuda_quota_refuses_before_the_law_body_device_then_host(cp, monkeypatch):
    col = _column(cp, [2.0] * 7)
    op = fe.prepare_floating_evaporation(col, PART, available_bytes_fn=_big, host_available_bytes_fn=_big)
    p, c, s = _device_inputs(cp)
    reached = []
    monkeypatch.setattr(se, "_stage_laws", lambda *a, **k: reached.append(1))
    kw = dict(pet_terrain_m_s=p, cover=c, surface_water_m=s, dt_s=1.0)
    m = fe.estimate_floating_evaporation_memory(7, backend="cuda")
    with pytest.raises(fsl.FloatingMemoryError):
        fe.stage_floating_evaporation(op, **kw, available_bytes_fn=lambda: m.device_bytes - 1, host_available_bytes_fn=_big)
    with pytest.raises(fsl.FloatingMemoryError):
        fe.stage_floating_evaporation(op, **kw, available_bytes_fn=_big, host_available_bytes_fn=lambda: m.host_bytes - 1)
    with pytest.raises(fsl.FloatingMemoryError):
        fe.stage_floating_evaporation(op, **kw, available_bytes_fn=lambda: 0, host_available_bytes_fn=lambda: 0)
    assert reached == []


def test_cuda_metadata_and_device_gate_refusals(cp):
    col = _column(cp, [2.0] * 7)
    op = fe.prepare_floating_evaporation(col, PART, available_bytes_fn=_big, host_available_bytes_fn=_big)
    p, c, s = _device_inputs(cp)
    ph, _ch, _sh = inputs()
    for bad in (ph, p.astype(cp.float32), p[:-1].copy()):  # host array, wrong dtype, wrong shape: refused before any admission reading or allocation
        with pytest.raises(fsl.FloatingSoilError):
            fe.stage_floating_evaporation(op, pet_terrain_m_s=bad, cover=c, surface_water_m=s, dt_s=1.0, available_bytes_fn=_big, host_available_bytes_fn=_big)
    if cp.cuda.runtime.getDeviceCount() > 1:
        other = 1 - int(col._maps["active"].device.id)
        reads = []
        with cp.cuda.Device(other), pytest.raises(fsl.FloatingSoilError, match="device"):
            fe.stage_floating_evaporation(op, pet_terrain_m_s=p, cover=c, surface_water_m=s, dt_s=1.0, available_bytes_fn=lambda: reads.append(1) or _big(), host_available_bytes_fn=_big)
        assert reads == []  # the current-device gate precedes every reader
