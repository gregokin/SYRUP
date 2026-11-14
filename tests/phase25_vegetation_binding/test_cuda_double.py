"""The AUTHORED CUDA preparation path through the HOST-compiled plumbing double: NOT hardware evidence. Actual native owners and kernels, the actual `stream_hash` mask recipe, real LEAF projections.
The real-hardware definitions are in test_cuda_hardware.py (skipped unless Root sets the flag; never run by this author)."""
from __future__ import annotations

import numpy as np
import pytest
import veg_cuda_double as vd
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

from maple_syrup import water_vegetation_binding as wvb

ERR = wvb.VegetationBindingError
MEM = wvb.VegetationBindingMemoryError
K, PSI = 2e-6, 0.3
pytestmark = pytest.mark.skipif(not vd.have_toolchain(), reason="no C++20 g++ toolchain: the host double of the CUDA source is not run (no claim)")


@pytest.fixture(scope="module")
def libs():
    return vd.build_libs()


@pytest.fixture()
def cp(libs, monkeypatch):
    return vd.make_cp(libs, monkeypatch)


def controls(**kw):
    base = {"a1": A1, "a2": A2, "a3": 0.4, "max_fraction_per_hour": 0.7, "max_intake_rate_m_s": 1e-5, "background_receiver_weights": [0.0, 1.0, 1.0], "n_layers": 3, "n_cells": 9, "max_bytes": BIG}
    base.update(kw)
    return wvb.site_controls(**base)


def bound(leaf, mask=None, **ckw):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 1, 1], mask=mask)
    m = np.ones((3, 3), dtype=bool) if mask is None else mask
    b = wvb.bind_projection(p, controls(**ckw), expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=p.identity_record()["content_sha256"], expected_source=full_source(p), expected_external_helpers=helper_pin(),
                            active_mask=m, current_surface=surf, max_bytes=BIG)
    return p, b, m


def cols(cp, m):
    return vd.pair(cp, EDGES, [EDGES[-1]] * 9, active=m.reshape(-1))


def prep(col, b, p, m, **kw):
    args = {"conductivity_m_s": K, "suction_m": PSI, "memory_budget_bytes": BIG, "host_memory_budget_bytes": BIG, "host_active_mask": m}
    args.update(kw)
    return wvb.prepare_vegetation_law(col, b, **args)


def test_cuda_law_matches_the_cpu_law_and_uses_the_one_canonical_derivation(leaf, cp):
    p, b, m = bound(leaf)
    cpu, gpu = cols(cp, m)
    a = wvb.prepare_vegetation_law(cpu, b, conductivity_m_s=K, suction_m=PSI, memory_budget_bytes=BIG)
    r = prep(gpu, b, p, m)
    assert r.backend == "cuda" and r.device_id == 0 and r.n_contributions == a.n_contributions == 2 and r.g_sha256 == a.g_sha256
    assert np.array_equal(np.asarray(r.matrix_op._law[2]), np.asarray(a.matrix_op._law[2]))                         # g: the same host values, one H2D copy
    assert np.allclose(np.asarray(r.preferential_op._law[2]), a.preferential_op._law[2], rtol=1e-12, atol=1e-15)    # receiver map
    assert np.allclose(np.asarray(r.preferential_op._law[0]), a.preferential_op._law[0], rtol=1e-12, atol=1e-18)    # aggregate rate
    assert wvb.verify_prepared_law(r) is r and r.step_taken is False and r.water_head_authority is False


def test_cell_maps_and_scalars_follow_the_native_layouts(leaf, cp):
    mr = np.full(9, 5e-6)
    mr[2] = float("inf")
    p, b, m = bound(leaf, a3=np.linspace(0.1, 0.9, 9), max_intake_rate_m_s=mr)
    cpu, gpu = cols(cp, m)
    r = prep(gpu, b, p, m)
    a = wvb.prepare_vegetation_law(cpu, b, conductivity_m_s=K, suction_m=PSI, memory_budget_bytes=BIG)
    assert r.preferential_op.layouts == a.preferential_op.layouts == {"a3": "cell", "max_intake_rate_m_s": "cell"}
    assert np.allclose(np.asarray(r.preferential_op._law[2]), a.preferential_op._law[2], rtol=1e-12, atol=1e-15)


def test_mask_identity_uses_the_actual_stream_hash_recipe(leaf, cp):
    p, b, m = bound(leaf)
    cpu, gpu = cols(cp, m)
    other = m.copy()
    other[0, 0] = False
    with pytest.raises(ERR, match="host active mask"):
        prep(gpu, b, p, other)                                             # not the binding's mask
    p2, b2, _m2 = bound(leaf, mask=other)
    with pytest.raises(ERR, match="device mask identity"):
        prep(gpu, b2, p2, other)                                           # the binding's own mask, but the column was built with another one
    with pytest.raises(ERR):
        prep(gpu, b, p, None)
    with pytest.raises(ERR):
        prep(gpu, b, p, m.astype(np.uint8))
    with pytest.raises(ERR):
        prep(cpu, b, p, m)                                                 # host_active_mask is a cuda-only argument


def test_host_quota_is_required_and_cpu_rejects_host_arguments(leaf, cp):
    p, b, m = bound(leaf)
    cpu, gpu = cols(cp, m)
    with pytest.raises(ERR):
        prep(gpu, b, p, m, host_memory_budget_bytes=None)
    with pytest.raises(ERR):
        wvb.prepare_vegetation_law(cpu, b, conductivity_m_s=K, suction_m=PSI, memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)


def test_device_gate_comes_first_before_any_reader_or_allocation(leaf, cp):
    p, b, m = bound(leaf)
    _cpu, gpu = cols(cp, m)
    reads, before = [], len(cp.allocs)
    cp.dev["id"] = 1
    with pytest.raises(Exception, match="device"):  # the native typed FloatingSoilError of the gate
        prep(gpu, b, p, m, available_bytes_fn=lambda: reads.append(1) or BIG, host_available_bytes_fn=lambda: reads.append(2) or BIG)
    assert reads == [] and len(cp.allocs) == before


def est_for(p, b, **kw):
    held, free = p.stream_estimate(previous_held=True)["per_next_bytes"], p.stream_estimate(previous_held=False)["per_next_bytes"]
    args = {"binding_resident_bytes": b.resident_bytes + b.borrowed_bytes, "stream_per_next_held_bytes": held, "stream_per_next_free_bytes": free, "stream_weights_stage_bytes": p.stream_estimate(previous_held=False)["weights_stage_bytes"], "backend": "cuda"}
    args.update(kw)
    return wvb.estimate_law_bytes(3, 9, len(b.enabled_species), **args)


def test_admission_need_minus_one_on_each_pool_refuses_before_any_allocation_exact_admits(leaf, cp, monkeypatch):
    from maple_syrup import floating_matrix_infiltration as fm
    p, b, m = bound(leaf)
    _cpu, gpu = cols(cp, m)
    est = est_for(p, b)
    calls, before = [], len(cp.allocs)
    real = fm.prepare_floating_matrix
    monkeypatch.setattr(fm, "prepare_floating_matrix", lambda *a, **k: calls.append(1) or real(*a, **k))
    host_extra = 65536  # fcu.admit's declared Python allowance on the host side
    with pytest.raises(Exception, match="device side"):
        prep(gpu, b, p, m, memory_budget_bytes=est["peak_bytes"] - 1)
    with pytest.raises(Exception, match="host side"):
        prep(gpu, b, p, m, host_memory_budget_bytes=est["peak_host_bytes"] + host_extra - 1)
    with pytest.raises(Exception, match="device side"):                      # caller reserves are counted once on each pool
        prep(gpu, b, p, m, memory_budget_bytes=est["peak_bytes"] + 9, reserve_bytes=10)
    with pytest.raises(Exception, match="host side"):
        prep(gpu, b, p, m, host_memory_budget_bytes=est["peak_host_bytes"] + host_extra + 9, host_reserve_bytes=10)
    with pytest.raises(Exception, match="device side"):                      # readers are consulted: a reading below the need refuses
        prep(gpu, b, p, m, available_bytes_fn=lambda: est["peak_bytes"] - 1)
    with pytest.raises(Exception, match="host side"):
        prep(gpu, b, p, m, host_available_bytes_fn=lambda: est["peak_host_bytes"] + host_extra - 1)
    assert calls == [] and len(cp.allocs) == before                       # nothing was allocated or prepared
    r = prep(gpu, b, p, m, memory_budget_bytes=est["peak_bytes"], host_memory_budget_bytes=est["peak_host_bytes"] + host_extra)
    assert calls == [1] and r.n_contributions == 2
    assert prep(gpu, b, p, m, memory_budget_bytes=est["peak_bytes"] + 10, reserve_bytes=10, host_memory_budget_bytes=est["peak_host_bytes"] + host_extra + 10, host_reserve_bytes=10)


def test_cold_transfers_are_measured_and_the_record_is_truthful_not_zero(leaf, cp):
    """The host double COUNTS real traffic: the adapter-owned N-byte mask read-back, the native operators' cold identity hashes and small native packets are D2H; the record must say so."""
    p, b, m = bound(leaf, a3=np.linspace(0.1, 0.9, 9))
    _cpu, gpu = cols(cp, m)
    cp.allocs.clear()
    cp.count.d2h = cp.count.d2h_bytes = cp.count.h2d = cp.count.h2d_bytes = 0
    cp.count.d2h_sizes.clear()
    r = prep(gpu, b, p, m)
    N = 9
    est = r.memory["declared"]
    assert est["d2h_adapter_bytes"] == N and "d2h_bytes" not in est and est["d2h_modelled_bytes"] > N        # NOT a 'no D2H' claim
    assert cp.count.d2h_bytes >= N and N in cp.count.d2h_sizes                                                # the adapter's mask read-back actually happened
    assert cp.count.d2h_bytes >= est["d2h_modelled_bytes"]                                                      # + the native cold identity hashes (`.get()` slices): real D2H too
    small = cp.count.d2h_bytes - est["d2h_modelled_bytes"]
    packets = [x for x in cp.count.d2h_sizes if x in (8, 32)]                                                    # 8-byte range diagnostics and 32-byte cold-reducer packets
    assert 0 <= small <= sum(packets)                                                                           # the unmodelled remainder is only those small native packets
    for size in (N, 8 * N):                                                                                     # the adapter's cold H2D uploads: mask bool, g / a3 / contributions
        assert size in cp.allocs
    assert cp.allocs.count(8 * N) >= 1 + 1 + 2 and est["h2d_adapter_bytes"] == N + 8 * N + 8 * N + 2 * 8 * N
    assert r.step_taken is False
