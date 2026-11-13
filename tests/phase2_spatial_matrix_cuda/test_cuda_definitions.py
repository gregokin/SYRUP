"""Actual-GPU test DEFINITIONS for the spatial CUDA matrix infiltration + history (WRITTEN, NOT RUN by their author: no GPU was launched or probed). Skipped without a usable CuPy device; Root runs them under its
device guard (GPU0 UUID, fresh 0 MiB and 0 %, `.venv-cupy`, genuine 9f CUDA materialisation). They compare the device operator with the accepted CPU spatial path (arrays within the declared rtol 2e-12 / atol
1e-14; bitwise where no logarithm is evaluated), with the frozen common NumPy controls, with the accepted COMMON CUDA matrix operator column by column on the SAME backend (bit for bit: the same body and the same
device math), flags/typed errors on every cell, sizes around the block boundary, one 136-byte read and zero H2D per step, and release. Valid constant-law splits only: `b` is re-evaluated per call."""
from __future__ import annotations

import gc
import json

import numpy as np
import pytest
from conftest import ATOL, AREA, BIG, LAYOUT_SETS, RTOL, HISTORICAL_PRE_REVIEW02_REFUSALS, TASK, a3, bits, forcing, mi, sequence_case, sm, sp, ss, water_for


@pytest.fixture(scope="module")
def cp():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the spatial CUDA matrix is not exercised (no claim)")
    return backend.cupy_module()


def dev(cp, a):
    return cp.asarray(np.ascontiguousarray(a))


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def gpu_owner(cp, inputs, N, mask, area=AREA):
    d = {k: (v if np.isscalar(v) else dev(cp, v)) for k, v in inputs.items()}
    spec = sp.spatial_layer_spec(d["edges"], theta_residual=d["res"], theta_field_capacity=d["fc"], theta_saturation=d["sat"], theta_evaporation_stress=0.5, diffusivity_m2_s=d["iface"], n_cells=N)
    return sp.prepare_spatial_column(spec, dev(cp, mask), area, backend="cuda", memory_budget_bytes=BIG)


def both(cp, rng, L, N, layout, *, scalar_law=False):
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS[layout])
    mask = rng.random(N) < 0.8
    mask[0] = True
    gcol = gpu_owner(cp, inputs, N, mask)
    ccol = sp.prepare_spatial_column(a3.spec_from(inputs, N), mask, AREA, backend="numpy", memory_budget_bytes=BIG)
    K, psi, g, *_ = forcing(rng, mask, 60.0, N)
    mk = (lambda v: float(v[0])) if scalar_law else (lambda v: v)
    gop = sm.prepare_spatial_matrix(gcol, conductivity_m_s=mk(dev(cp, K)) if not scalar_law else float(K[0]), suction_m=dev(cp, psi) if not scalar_law else float(psi[0]),
                                    vegetation_factor=dev(cp, g) if not scalar_law else float(g[0]), memory_budget_bytes=BIG)
    cop = sm.prepare_spatial_matrix(ccol, conductivity_m_s=mk(K), suction_m=mk(psi), vegetation_factor=mk(g), memory_budget_bytes=BIG)
    return p, mask, gcol, ccol, gop, cop


def pair_from_case(cp, c):
    """The device and CPU spatial owners/operators of a `sequence_case` (nothing is altered: the same law arrays feed both)."""
    gcol = gpu_owner(cp, c.inputs, c.N, c.mask)
    ccol = sp.prepare_spatial_column(a3.spec_from(c.inputs, c.N), c.mask, AREA, backend="numpy", memory_budget_bytes=BIG)
    gop = sm.prepare_spatial_matrix(gcol, conductivity_m_s=dev(cp, c.K), suction_m=dev(cp, c.psi), vegetation_factor=dev(cp, c.g), memory_budget_bytes=BIG)
    cop = sm.prepare_spatial_matrix(ccol, conductivity_m_s=c.K, suction_m=c.psi, vegetation_factor=c.g, memory_budget_bytes=BIG)
    return gcol, ccol, gop, cop


@pytest.mark.parametrize("layout", ["all_cell", "mixed", "scalar_thetas", "all_shared"])
@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("N", [1, 7, 127, 128, 129, 257, 1001])
def test_device_matches_the_cpu_spatial_path_over_a_history_sequence(cp, layout, L, N):
    """Every step of every valid sequence must SUCCEED on the device and on the CPU spatial path (review02: the bounded combined intake removed the rounding-level negative pools that the pre-review02 source refused at
    two of these sequences); a refusal of either path fails the test. Physical tolerances: rtol 2e-12 / atol 1e-14."""
    c = sequence_case(L, N, layout)
    gcol, ccol, gop, cop = pair_from_case(cp, c)
    ghist = sm.spatial_matrix_history(gop, wetting_m=dev(cp, c.j0), cumulative_intake_m=0.01, memory_budget_bytes=BIG)
    chist = sm.spatial_matrix_history(cop, wetting_m=c.j0, cumulative_intake_m=0.01, memory_budget_bytes=BIG)
    gst, cst = sp._new_state(gcol, dev(cp, c.water), 55.5), sp.spatial_column_state(ccol, c.water, 55.5, memory_budget_bytes=BIG)
    for dt, pool, standing, rate in c.steps:
        g = sm.spatial_matrix_step(gop, dev(cp, pool), dev(cp, standing), dev(cp, rate), gst, ghist, dt, memory_budget_bytes=BIG)
        c_ = sm.spatial_matrix_step(cop, pool, standing, rate, cst, chist, dt, memory_budget_bytes=BIG)
        assert g.state.t_s == 55.5 and g.state.namespace == "cupy" and g.history.namespace == "cupy"
        for a, b in ((g.state.water_m, c_.state.water_m), (g.surface_pool_m, c_.surface_pool_m), (g.intake_m, c_.intake_m), (g.history.wetting_m, c_.history.wetting_m),
                     (g.history.cumulative_intake_m, c_.history.cumulative_intake_m)):
            np.testing.assert_allclose(host(a), b, rtol=RTOL, atol=ATOL)
        assert (host(g.surface_pool_m) >= 0.0).all() and (host(g.intake_m) <= pool).all()
        assert g.budget.n_onsets == c_.budget.n_onsets and g.budget.n_ponded == c_.budget.n_ponded and g.budget.n_terms == c_.budget.n_terms
        assert abs(g.budget.soil_residual_m3) <= g.budget.bound_m3 and abs(g.budget.pool_residual_m3) <= g.budget.bound_m3
        for f in ("intake_m3", "soil_before_m3", "soil_after_m3", "pool_after_m3", "candidate_m3", "wetting_after_m3"):
            assert getattr(g.budget, f) == pytest.approx(getattr(c_.budget, f), rel=1e-11, abs=1e-15)
        gst, ghist, cst, chist = g.state, g.history, c_.state, c_.history


@pytest.mark.parametrize("case", sorted(HISTORICAL_PRE_REVIEW02_REFUSALS), ids=lambda k: f"L{k[0]}-N{k[1]}-{k[2]}")
def test_the_original_sequences_succeed_on_all_four_stages_and_the_common_cuda_operator_agrees_on_the_recorded_cell(cp, case):
    """Review02 regression definition: the two sequences that the pre-review02 source refused at step 1 now integrate on all four stages on the device, the recorded cell exhausts its pool exactly (pool-out 0, not
    negative), and the genuine COMMON CUDA operator (real cupy mask) on that cell gives bit-identical arrays to the spatial device step (same body, same device math)."""
    L, N, layout = case
    old = HISTORICAL_PRE_REVIEW02_REFUSALS[case]
    c = sequence_case(L, N, layout)
    gcol, ccol, gop, cop = pair_from_case(cp, c)
    ghist = sm.spatial_matrix_history(gop, wetting_m=dev(cp, c.j0), cumulative_intake_m=0.01, memory_budget_bytes=BIG)
    gst = sp._new_state(gcol, dev(cp, c.water), 55.5)
    for i, (dt, pool, standing, rate) in enumerate(c.steps):
        g = sm.spatial_matrix_step(gop, dev(cp, pool), dev(cp, standing), dev(cp, rate), gst, ghist, dt, memory_budget_bytes=BIG)
        if i == old["step"]:
            cell = old["cell"]
            assert host(g.intake_m)[cell] == pool[cell] and host(g.surface_pool_m)[cell] == 0.0
            lay = ss.soil_layers(c.p["edges"][:, cell], theta_residual=c.p["res"][:, cell], theta_field_capacity=c.p["fc"][:, cell], theta_saturation=c.p["sat"][:, cell], theta_evaporation_stress=0.5,
                                 diffusivity_m2_s=c.p["iface"][:, cell])
            col = ss.prepare_column(lay, cp.asarray(np.array([bool(c.mask[cell])])), AREA, backend="cuda")  # the common API needs a REAL cupy bool mask
            law = [float(x.get()[0 if gop.layouts[n] == "scalar" else cell]) for n, x in zip(sm.LAW_NAMES, gop._law, strict=True)]
            op = mi.prepare_matrix(col, conductivity_m_s=law[0], suction_m=law[1], vegetation_factor=law[2])
            oldh = mi.wetting_history(op, wetting_m=dev(cp, host(ghist.wetting_m)[cell:cell + 1]), cumulative_intake_m=dev(cp, host(ghist.cumulative_intake_m)[cell:cell + 1]))
            r = mi.matrix_step(op, dev(cp, pool[cell:cell + 1]), dev(cp, standing[cell:cell + 1]), dev(cp, rate[cell:cell + 1]), ss.column_state(col, dev(cp, host(gst.water_m)[:, cell:cell + 1]), 55.5), oldh, dt)
            for a, b in ((g.state.water_m[:, cell:cell + 1], r.state.water_m), (g.surface_pool_m[cell:cell + 1], r.surface_pool_m), (g.intake_m[cell:cell + 1], r.intake_m),
                         (g.history.wetting_m[cell:cell + 1], r.history.wetting_m), (g.history.cumulative_intake_m[cell:cell + 1], r.history.cumulative_intake_m)):
                np.testing.assert_array_equal(bits(host(a)), bits(host(b)))
        gst, ghist = g.state, g.history


def test_each_column_equals_the_accepted_common_cuda_matrix_operator_bit_for_bit_on_the_same_backend(cp):
    rng = np.random.default_rng(3)
    L, N = 3, 24
    p, mask, gcol, ccol, gop, cop = both(cp, rng, L, N, "all_cell")
    water = water_for(rng, p, N)
    K, psi, g, pool, standing, rate = forcing(rng, mask, 600.0, N)
    gop = sm.prepare_spatial_matrix(gcol, conductivity_m_s=dev(cp, K), suction_m=dev(cp, psi), vegetation_factor=dev(cp, g), memory_budget_bytes=BIG)
    cop = sm.prepare_spatial_matrix(ccol, conductivity_m_s=K, suction_m=psi, vegetation_factor=g, memory_budget_bytes=BIG)
    ghist = sm.spatial_matrix_history(gop, wetting_m=0.002, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    chist = sm.spatial_matrix_history(cop, wetting_m=0.002, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    got = sm.spatial_matrix_step(gop, dev(cp, pool), dev(cp, standing), dev(cp, rate), sp._new_state(gcol, dev(cp, water), 1.0), ghist, 600.0, memory_budget_bytes=BIG)  # must SUCCEED (review02)
    for c in range(N):
        lay = ss.soil_layers(p["edges"][:, c], theta_residual=p["res"][:, c], theta_field_capacity=p["fc"][:, c], theta_saturation=p["sat"][:, c], theta_evaporation_stress=0.5,
                             diffusivity_m2_s=p["iface"][:, c])
        col = ss.prepare_column(lay, cp.asarray(np.array([bool(mask[c])])), AREA, backend="cuda")  # a REAL cupy bool mask (the common API refuses a NumPy mask)
        op = mi.prepare_matrix(col, conductivity_m_s=float(K[c]), suction_m=float(psi[c]), vegetation_factor=float(g[c]))
        hist = mi.wetting_history(op, wetting_m=dev(cp, np.array([0.002])), cumulative_intake_m=dev(cp, np.array([0.0])))
        ref = mi.matrix_step(op, dev(cp, pool[c:c + 1]), dev(cp, standing[c:c + 1]), dev(cp, rate[c:c + 1]), ss.column_state(col, dev(cp, water[:, c:c + 1]), 1.0), hist, 600.0)
        for a, b in ((got.state.water_m[:, c:c + 1], ref.state.water_m), (got.surface_pool_m[c:c + 1], ref.surface_pool_m), (got.intake_m[c:c + 1], ref.intake_m),
                     (got.history.wetting_m[c:c + 1], ref.history.wetting_m), (got.history.cumulative_intake_m[c:c + 1], ref.history.cumulative_intake_m)):
            np.testing.assert_array_equal(bits(host(a)), bits(host(b)), err_msg=f"cell {c}")


FROZEN = TASK / "root_s2_b2_reference"


@pytest.mark.skipif(not (FROZEN / "controls.json").exists(), reason="Root's frozen reference is not present")
def test_frozen_matrix_controls_within_the_declared_bound(cp):
    controls = json.loads((FROZEN / "controls.json").read_text())
    expected = np.load(FROZEN / "expected.npz")
    configs = json.loads((TASK / "root_s2_parameter_controls.json").read_text())["controls"]
    for sc in (s for s in controls["scenarios"] if s["operation"] == "matrix"):
        c = configs[sc["parameters_index"]]
        spec = sp.spatial_layer_spec(dev(cp, np.array(c["depth_edges_m"])), theta_residual=dev(cp, np.array(c["theta_residual"])), theta_field_capacity=dev(cp, np.array(c["theta_field_capacity"])),
                                     theta_saturation=dev(cp, np.array(c["theta_saturation"])), theta_evaporation_stress=c["theta_evaporation_stress"], diffusivity_m2_s=0.0, n_cells=c["cells"])
        col = sp.prepare_spatial_column(spec, dev(cp, np.array(sc["mask"])), sc["area_m2"], backend="cuda", memory_budget_bytes=BIG)
        arr = lambda k: dev(cp, np.array(sc[k], dtype=np.float64))  # noqa: E731
        op = sm.prepare_spatial_matrix(col, conductivity_m_s=arr("conductivity_m_s"), suction_m=arr("suction_m"), vegetation_factor=arr("vegetation_factor"), memory_budget_bytes=BIG)
        hist = sm.spatial_matrix_history(op, wetting_m=arr("wetting_m"), cumulative_intake_m=arr("cumulative_intake_m"), memory_budget_bytes=BIG)
        r = sm.spatial_matrix_step(op, arr("pool_m"), arr("standing_m"), arr("incoming_rate_m_s"), sp._new_state(col, arr("water_m"), sc["t_s"]), hist, sc["dt_s"], memory_budget_bytes=BIG)
        label = sc["label"]
        assert r.state.t_s == sc["t_s"]
        for name, a in (("water", r.state.water_m), ("surface_pool_m", r.surface_pool_m), ("intake_m", r.intake_m), ("wetting_m", r.history.wetting_m), ("cumulative_intake_m", r.history.cumulative_intake_m)):
            np.testing.assert_allclose(host(a), expected[f"{label}_{name}"], rtol=RTOL, atol=ATOL, err_msg=f"{label} {name}")


def test_unponded_supply_limited_cells_are_bit_exact_with_the_cpu_path(cp):
    rng = np.random.default_rng(8)
    L, N = 3, 200
    p, mask, gcol, ccol, gop, cop = both(cp, rng, L, N, "mixed")
    water = water_for(rng, p, N)
    K = rng.uniform(1e-6, 1e-5, N)
    psi, g = rng.uniform(0.0, 0.3, N), rng.uniform(0.0, 1.0, N)
    rate = np.where(mask, K * rng.uniform(0.1, 1.0, N), 0.0)
    pool = np.ascontiguousarray(rate * 600.0 * rng.uniform(0.3, 1.0, N))
    gop = sm.prepare_spatial_matrix(gcol, conductivity_m_s=dev(cp, K), suction_m=dev(cp, psi), vegetation_factor=dev(cp, g), memory_budget_bytes=BIG)
    cop = sm.prepare_spatial_matrix(ccol, conductivity_m_s=K, suction_m=psi, vegetation_factor=g, memory_budget_bytes=BIG)
    gh = sm.spatial_matrix_history(gop, wetting_m=0.002, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    ch = sm.spatial_matrix_history(cop, wetting_m=0.002, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    z = np.zeros(N)
    g_ = sm.spatial_matrix_step(gop, dev(cp, pool), dev(cp, z), dev(cp, rate), sp._new_state(gcol, dev(cp, water), 0.0), gh, 600.0, memory_budget_bytes=BIG)
    c_ = sm.spatial_matrix_step(cop, pool, z, rate, sp.spatial_column_state(ccol, water, 0.0, memory_budget_bytes=BIG), ch, 600.0, memory_budget_bytes=BIG)
    assert g_.budget.n_ponded == 0
    for a, b in ((g_.state.water_m, c_.state.water_m), (g_.surface_pool_m, c_.surface_pool_m), (g_.intake_m, c_.intake_m), (g_.history.wetting_m, c_.history.wetting_m)):
        np.testing.assert_array_equal(bits(host(a)), bits(b))


def test_flags_are_reported_for_every_cell_inactive_included_and_nothing_is_written(cp):
    rng = np.random.default_rng(12)
    L, N = 2, 300
    p, mask, gcol, ccol, gop, cop = both(cp, rng, L, N, "all_cell")
    ghist = sm.spatial_matrix_history(gop, wetting_m=0.001, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    water = water_for(rng, p, N) * 0.5
    water[0, [1, 4, 150, 299]] = -1.0
    water[1, 150] = -2.0
    z = np.zeros(N)
    with pytest.raises(mi.MatrixStepError) as info:
        sm.spatial_matrix_step(gop, dev(cp, z), dev(cp, z), dev(cp, z), sp._new_state(gcol, dev(cp, water), 0.0), ghist, 60.0, memory_budget_bytes=BIG)
    assert info.value.bit == 1 and info.value.n_flagged == 4  # each invalid cell once, inactive included
    pool = np.full(N, 0.01)
    pool[3] = -1.0
    snap = pool.copy()
    d_pool = dev(cp, pool)
    with pytest.raises(mi.MatrixStepError) as info:
        sm.spatial_matrix_step(gop, d_pool, dev(cp, np.full(N, 0.01)), dev(cp, z), sp._new_state(gcol, dev(cp, water * 0 + 0.01), 0.0), ghist, 60.0, memory_budget_bytes=BIG)
    assert info.value.bit == 4
    np.testing.assert_array_equal(host(d_pool), snap)


def test_one_136_byte_read_and_zero_h2d_per_step_and_release(cp):
    backend = pytest.importorskip("maple.core.backend")
    rng = np.random.default_rng(6)
    L, N = 3, 1001
    p, mask, gcol, ccol, gop, cop = both(cp, rng, L, N, "mixed")
    ghist = sm.spatial_matrix_history(gop, wetting_m=0.001, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    water = water_for(rng, p, N)
    _, _, _, pool, standing, rate = forcing(rng, mask, 600.0, N)
    args = (dev(cp, pool), dev(cp, standing), dev(cp, rate), sp._new_state(gcol, dev(cp, water), 0.0), ghist, 600.0)
    sm.spatial_matrix_step(gop, *args, memory_budget_bytes=BIG)  # warm
    before = dict(backend.transfer_counters()) if hasattr(backend, "transfer_counters") else None
    r = sm.spatial_matrix_step(gop, *args, memory_budget_bytes=BIG)
    if before is not None:
        after = backend.transfer_counters()
        assert after.get("device_to_host", 0) - before.get("device_to_host", 0) == 1
        assert after.get("device_to_host_bytes", 0) - before.get("device_to_host_bytes", 0) == 136
        assert after.get("host_to_device", 0) - before.get("host_to_device", 0) == 0
    pool_before = cp.get_default_memory_pool().used_bytes()
    del r
    gc.collect()
    assert cp.get_default_memory_pool().used_bytes() <= pool_before  # the step outputs are released with their owners (scratch was released after the synchronising read)


def test_wrong_device_and_numpy_arrays_are_refused_before_any_launch(cp):
    rng = np.random.default_rng(1)
    p, mask, gcol, ccol, gop, cop = both(cp, rng, 2, 20, "all_cell")
    ghist = sm.spatial_matrix_history(gop, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    z = dev(cp, np.zeros(20))
    state = sp._new_state(gcol, dev(cp, np.zeros((2, 20))), 0.0)
    with pytest.raises(sp.SpatialLayerError):
        sm.spatial_matrix_step(gop, np.zeros(20), z, z, state, ghist, 60.0, memory_budget_bytes=BIG)
    if cp.cuda.runtime.getDeviceCount() > 1:
        with cp.cuda.Device(1):
            with pytest.raises(sp.SpatialLayerError):
                sm.spatial_matrix_step(gop, z, z, z, state, ghist, 60.0, memory_budget_bytes=BIG)
