"""The rendered spatial CUDA MATRIX source EXECUTED on the host (real threads + barrier, hostb2c.py) through the REAL Python launchers and public API on a numpy-backed CuPy plumbing double, compared with the
accepted NumPy spatial operator (arrays within the declared rtol 2e-12 / atol 1e-14 for host libm vs NumPy, bitwise where no logarithm is evaluated; budget sums within the roundoff of a fixed-tree sum), plus
the plumbing contracts (allocations equal the model, one 136-byte read, admission before any allocation/compile on both sides, device/type/namespace refusals, typed OOM/runtime errors, no mutation, release,
bounded cold identity hash). NOT hardware evidence; no fake CUDA-tagged owner is offered as proof of device behaviour. Author-written."""
from __future__ import annotations

import gc
import hashlib
import weakref

import numpy as np
import pytest
from conftest import (ATOL, AREA, BIG, LAYOUT_SETS, RTOL, HISTORICAL_PRE_REVIEW02_REFUSALS, a3, bits, device_owner, dev_state, forcing, fx, mi, needs_host, sequence_case, sm, sp, ss, tc, water_for)

import maple.core.backend as backend

pytestmark = needs_host


def pair(rng, L, N, layout, *, scalar=False, mask=None):
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS[layout])
    mask = ((rng.random(N) < 0.8) if mask is None else mask).copy()
    if mask.any():
        mask[0] = True  # at least the first cell is active unless the caller asked for an all-inactive mask
    np_col = sp.prepare_spatial_column(a3.spec_from(inputs, N), mask, AREA, backend="numpy", memory_budget_bytes=BIG)
    return inputs, p, mask, np_col


def build(cupy, np_col, K, psi, g, *, scalar=False, j0=0.001, c0=0.0, device=0):
    col = device_owner(np_col, device)
    mk = (lambda v: float(v[0])) if scalar else (lambda v: fx(v, device))
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=mk(K), suction_m=mk(psi), vegetation_factor=mk(g), memory_budget_bytes=BIG)
    hist = sm.spatial_matrix_history(op, wetting_m=fx(j0 if isinstance(j0, np.ndarray) else np.full(np_col.n_cells, j0), device), cumulative_intake_m=c0, memory_budget_bytes=BIG)
    return col, op, hist


def cpu_ref(np_col, K, psi, g, j0, c0, scalar=False):
    mk = (lambda v: float(v[0])) if scalar else (lambda v: v)
    op = sm.prepare_spatial_matrix(np_col, conductivity_m_s=mk(K), suction_m=mk(psi), vegetation_factor=mk(g), memory_budget_bytes=BIG)
    j = j0 if isinstance(j0, np.ndarray) else np.full(np_col.n_cells, j0)
    return op, sm.spatial_matrix_history(op, wetting_m=j, cumulative_intake_m=c0, memory_budget_bytes=BIG)


@pytest.mark.parametrize("layout", ["all_cell", "mixed", "scalar_thetas", "all_shared"])
@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("N", [1, 7, 127, 128, 129, 257])
def test_device_steps_match_the_cpu_spatial_path_over_a_history_sequence(cupy, layout, L, N):
    rng = np.random.default_rng(1000 * L + N + len(layout))
    inputs, p, mask, np_col = pair(rng, L, N, layout)
    water = water_for(rng, p, N)
    K, psi, g, *_ = forcing(rng, mask, 60.0, N)
    j0 = np.where(rng.random(N) < 0.5, 0.0, rng.uniform(0.0, 0.05, N))
    col, op, hist = build(cupy, np_col, K, psi, g, j0=j0, c0=0.01)
    cop, chist = cpu_ref(np_col, K, psi, g, j0, 0.01)
    state, cstate = dev_state(col, water, 55.5), sp.spatial_column_state(np_col, water, 55.5, memory_budget_bytes=BIG)
    for dt in (60.0, 3600.0, 0.0, 600.0):
        _, _, _, pool, standing, rate = forcing(rng, mask, dt, N)
        got = sm.spatial_matrix_step(op, fx(pool), fx(standing), fx(rate), state, hist, dt, memory_budget_bytes=BIG)  # every step of every valid sequence must SUCCEED (review02): no refusal is tolerated
        ref = sm.spatial_matrix_step(cop, pool, standing, rate, cstate, chist, dt, memory_budget_bytes=BIG)
        assert got.state.t_s == 55.5 and got.state.namespace == "cupy" and got.history.namespace == "cupy" and got.dt_s == dt  # the clock is preserved
        for a, b in ((got.state.water_m, ref.state.water_m), (got.surface_pool_m, ref.surface_pool_m), (got.intake_m, ref.intake_m), (got.history.wetting_m, ref.history.wetting_m),
                     (got.history.cumulative_intake_m, ref.history.cumulative_intake_m)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=RTOL, atol=ATOL)
        gb, rb = got.budget, ref.budget
        assert gb.n_onsets == rb.n_onsets and gb.n_ponded == rb.n_ponded and abs(gb.max_newton_iterations - rb.max_newton_iterations) <= 1 and gb.n_terms == rb.n_terms
        for f in ("soil_before_m3", "soil_after_m3", "pool_before_m3", "pool_after_m3", "candidate_m3", "intake_m3", "prepond_intake_m3", "postpond_intake_m3", "room_rejected_m3", "unused_capacity_m3",
                  "wetting_before_m3", "wetting_after_m3"):
            assert getattr(gb, f) == pytest.approx(getattr(rb, f), rel=1e-11, abs=1e-15), f
        assert abs(gb.soil_residual_m3) <= gb.bound_m3 and abs(gb.pool_residual_m3) <= gb.bound_m3
        np.testing.assert_array_equal(np.asarray(got.intake_m)[~mask], 0.0)
        state, hist, cstate, chist = got.state, got.history, ref.state, ref.history


def test_without_a_logarithm_the_device_arrays_are_bit_exact(cupy):
    """Unponded supply-limited cells only (r <= K, no contact): no log1p/log is evaluated, so the host-executed kernel and NumPy agree bit for bit (products unfused)."""
    rng = np.random.default_rng(8)
    L, N = 3, 200
    inputs, p, mask, np_col = pair(rng, L, N, "mixed")
    water = water_for(rng, p, N)
    K = rng.uniform(1e-6, 1e-5, N)
    psi, g = rng.uniform(0.0, 0.3, N), rng.uniform(0.0, 1.0, N)
    rate = np.where(mask, K * rng.uniform(0.1, 1.0, N), 0.0)
    pool = np.ascontiguousarray(rate * 600.0 * rng.uniform(0.3, 1.0, N))
    col, op, hist = build(cupy, np_col, K, psi, g, j0=0.002)
    cop, chist = cpu_ref(np_col, K, psi, g, 0.002, 0.0)
    got = sm.spatial_matrix_step(op, fx(pool), fx(np.zeros(N)), fx(rate), dev_state(col, water), hist, 600.0, memory_budget_bytes=BIG)
    ref = sm.spatial_matrix_step(cop, pool, np.zeros(N), rate, sp.spatial_column_state(np_col, water, 0.0, memory_budget_bytes=BIG), chist, 600.0, memory_budget_bytes=BIG)
    assert got.budget.n_ponded == 0 == ref.budget.n_ponded
    for a, b in ((got.state.water_m, ref.state.water_m), (got.surface_pool_m, ref.surface_pool_m), (got.intake_m, ref.intake_m), (got.history.wetting_m, ref.history.wetting_m),
                 (got.history.cumulative_intake_m, ref.history.cumulative_intake_m)):
        np.testing.assert_array_equal(bits(a), bits(b))


@pytest.mark.parametrize("scalar", [False, True])
def test_scalar_and_cell_law_layouts_and_all_inactive(cupy, scalar):
    rng = np.random.default_rng(5)
    L, N = 2, 33
    inputs, p, mask, np_col = pair(rng, L, N, "mixed")
    water = water_for(rng, p, N)
    _, _, _, pool, standing, rate = forcing(rng, mask, 600.0, N)
    K, psi, g = np.full(N, 4e-6), np.full(N, 0.12), np.full(N, 0.6)
    col, op, hist = build(cupy, np_col, K, psi, g, scalar=scalar)
    assert dict(op.layouts) == {k: ("scalar" if scalar else "cell") for k in op.layouts} and op.retained_bytes == (3 * 8 if scalar else 3 * N * 8)
    cop, chist = cpu_ref(np_col, K, psi, g, 0.001, 0.0, scalar=scalar)
    got = sm.spatial_matrix_step(op, fx(pool), fx(standing), fx(rate), dev_state(col, water), hist, 600.0, memory_budget_bytes=BIG)
    ref = sm.spatial_matrix_step(cop, pool, standing, rate, sp.spatial_column_state(np_col, water, 0.0, memory_budget_bytes=BIG), chist, 600.0, memory_budget_bytes=BIG)
    np.testing.assert_allclose(np.asarray(got.state.water_m), ref.state.water_m, rtol=RTOL, atol=ATOL)
    # all inactive: the identity, nothing counted
    dead = np.zeros(N, dtype=bool)
    inputs2, p2 = a3.layout_case(rng, L, N, **LAYOUT_SETS["all_cell"])
    np2 = sp.prepare_spatial_column(a3.spec_from(inputs2, N), dead, AREA, backend="numpy", memory_budget_bytes=BIG)
    c2, o2, h2 = build(cupy, np2, K, psi, g, j0=0.01, c0=0.02)
    w2 = water_for(rng, p2, N)
    r = sm.spatial_matrix_step(o2, fx(np.full(N, 0.003)), fx(np.zeros(N)), fx(np.zeros(N)), dev_state(c2, w2), h2, 600.0, memory_budget_bytes=BIG)
    np.testing.assert_array_equal(bits(r.state.water_m), bits(w2))
    np.testing.assert_array_equal(np.asarray(r.surface_pool_m), 0.003)
    assert r.budget.intake_m3 == 0.0 and r.budget.soil_before_m3 == 0.0 and r.budget.n_ponded == 0
    np.testing.assert_array_equal(np.asarray(r.history.wetting_m), 0.01)


def test_invalid_inputs_on_every_cell_report_the_common_bit_and_count_and_write_nothing(cupy):
    rng = np.random.default_rng(12)
    L, N = 2, 20
    inputs, p, mask, np_col = pair(rng, L, N, "all_cell", mask=np.array([True, True, False, True] * 5))
    water0 = water_for(rng, p, N) * 0.5
    K, psi, g = np.full(N, 3e-6), np.full(N, 0.1), np.full(N, 0.5)
    col, op, hist0 = build(cupy, np_col, K, psi, g, j0=0.002, c0=0.003)
    cop, chist0 = cpu_ref(np_col, K, psi, g, 0.002, 0.003)
    base = dict(pool=np.full(N, 0.01), standing=np.full(N, 0.01), rate=np.zeros(N), wet=np.full(N, 0.002), cum=np.full(N, 0.003))
    cases = {0: ("water", (0, 0), np.nan), 1: ("water", (1, 2), -1e-3), 2: ("water", (0, 3), 50.0), 3: ("pool", 0, np.nan), 4: ("pool", 2, -1.0), 5: ("standing", 0, np.inf), 6: ("standing", 2, -1.0),
             7: ("rate", 0, np.nan), 8: ("rate", 3, -1e-6), 9: ("rate", 2, 1e-6), 10: ("wet", 0, np.nan), 11: ("wet", 2, -1.0), 12: ("cum", 0, np.inf), 13: ("cum", 2, -1.0)}
    for bit, (which, cell, value) in cases.items():
        arrays = {k: v.copy() for k, v in base.items()}
        water = water0.copy()
        if which == "water":
            water[cell] = value
        else:
            arrays[which][cell] = value
        snap = [water.copy(), *(a.copy() for a in arrays.values())]
        if which in ("wet", "cum"):
            with sm._minting():
                hist = sm.SpatialMatrixHistory(fx(arrays["wet"]), fx(arrays["cum"]), op.sha256, N, namespace="cupy", device_id=0)
            chist = None
        else:
            hist = hist0
        args = (fx(arrays["pool"]), fx(arrays["standing"]), fx(arrays["rate"]), dev_state(col, water), hist, 60.0)
        with pytest.raises(mi.MatrixStepError) as info:
            sm.spatial_matrix_step(op, *args, memory_budget_bytes=BIG)
        assert info.value.bit == bit and info.value.n_flagged >= 1 and mi.FLAG_MESSAGES[bit].split(" ")[0] in str(info.value), (bit, info.value)
        for a, b in zip((water, *arrays.values()), snap, strict=True):
            np.testing.assert_array_equal(a, b)  # inputs untouched (NaN-aware)


def test_the_flag_count_counts_each_invalid_cell_once_inactive_included(cupy):
    rng = np.random.default_rng(2)
    L, N = 2, 300
    inputs, p, mask, np_col = pair(rng, L, N, "all_cell", mask=np.array([True, False, True] * 100))
    water = water_for(rng, p, N) * 0.5
    water[0, [1, 4, 150, 299]] = -1.0  # four cells: two inactive (1, 4 is active? 4 % 3 -> 1 inactive; 150 % 3 == 0 active) ; every one counted once
    water[1, 150] = -2.0  # a second invalid layer in an already invalid cell: still one cell
    K, psi, g = np.full(N, 3e-6), np.full(N, 0.1), np.full(N, 0.5)
    col, op, hist = build(cupy, np_col, K, psi, g)
    with pytest.raises(mi.MatrixStepError) as info:
        sm.spatial_matrix_step(op, fx(np.zeros(N)), fx(np.zeros(N)), fx(np.zeros(N)), dev_state(col, water), hist, 60.0, memory_budget_bytes=BIG)
    cop, chist = cpu_ref(np_col, K, psi, g, 0.001, 0.0)
    with pytest.raises(mi.MatrixStepError) as ref:
        sm.spatial_matrix_step(cop, np.zeros(N), np.zeros(N), np.zeros(N), sp._new_state(np_col, water, 0.0), chist, 60.0, memory_budget_bytes=BIG)
    assert (info.value.bit, info.value.n_flagged) == (ref.value.bit, ref.value.n_flagged) == (1, 4)


def test_nonconvergence_inconsistent_split_and_typed_global_overflow(cupy, monkeypatch):
    spec = sp.spatial_layer_spec(np.array([0.0, 0.5, 1.5]), theta_residual=0.01, theta_field_capacity=0.25, theta_saturation=0.9, theta_evaporation_stress=0.05, diffusivity_m2_s=0.0, n_cells=2)
    np_col = sp.prepare_spatial_column(spec, np.ones(2, bool), 1e308, backend="numpy", memory_budget_bytes=BIG)
    col = device_owner(np_col)
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=1e-5, suction_m=0.08, vegetation_factor=0.5, memory_budget_bytes=BIG)
    hist = sm.spatial_matrix_history(op, wetting_m=0.001, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    saturated = np.array([[0.45, 0.45], [0.9, 0.9]])
    with pytest.raises(mi.MatrixStepError, match="non-finite global water budget"):  # typed, before the strict MAPLE bound helper (no raw ValueError)
        sm.spatial_matrix_step(op, fx(np.full(2, 0.01)), fx(np.full(2, 0.01)), fx(np.zeros(2)), dev_state(col, saturated), hist, 60.0, memory_budget_bytes=BIG)
    np2 = sp.prepare_spatial_column(spec, np.ones(2, bool), AREA, backend="numpy", memory_budget_bytes=BIG)
    c2 = device_owner(np2)
    o2 = sm.prepare_spatial_matrix(c2, conductivity_m_s=1e-5, suction_m=0.08, vegetation_factor=0.5, memory_budget_bytes=BIG)
    h2 = sm.spatial_matrix_history(o2, wetting_m=0.002, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    st = dev_state(c2, np.array([[0.0625, 0.0625], [0.1, 0.1]]))
    with pytest.raises(mi.MatrixStepError) as info:  # no contact and a pool above the supply r dt
        sm.spatial_matrix_step(o2, fx(np.full(2, 0.5)), fx(np.zeros(2)), fx(np.full(2, 1e-6)), st, h2, 60.0, memory_budget_bytes=BIG)
    assert info.value.bit == 14 and info.value.n_flagged == 2
    monkeypatch.setattr(mi, "NEWTON_MAX_ITERATIONS", 1)
    with pytest.raises(mi.MatrixStepError) as info:
        sm.spatial_matrix_step(o2, fx(np.full(2, 1.0)), fx(np.full(2, 1.0)), fx(np.zeros(2)), st, h2, 600.0, memory_budget_bytes=BIG)
    assert info.value.bit == 15  # non-convergence is flagged and refused, never clipped


# ---- plumbing: allocations, reads, transfers, launches ------------------------------------------------------------------------------------
def test_step_allocations_reads_and_launches_equal_the_model(cupy, monkeypatch):
    rng = np.random.default_rng(6)
    L, N = 3, 300
    inputs, p, mask, np_col = pair(rng, L, N, "mixed")
    water = water_for(rng, p, N)
    K, psi, g, pool, standing, rate = forcing(rng, mask, 600.0, N)
    col, op, hist = build(cupy, np_col, K, psi, g)
    args = (fx(pool), fx(standing), fx(rate), dev_state(col, water), hist, 600.0)
    reads = []
    real = backend.to_host
    monkeypatch.setattr(backend, "to_host", lambda a: reads.append((a.dtype.name, int(a.nbytes))) or real(a))
    cupy.allocations.clear(), cupy.h2d.clear(), cupy.launches.clear()
    r = sm.spatial_matrix_step(op, *args, memory_budget_bytes=BIG)
    nb = tc.n_blocks(N)
    assert [(k, shape, dt) for k, shape, dt, _ in cupy.allocations] == [("empty", (N,), "float64")] * 4 + [("empty", (L, N), "float64"), ("empty", (15 * nb,), "float64"), ("zeros", (17,), "uint64")]
    est = r.memory
    assert cupy.allocated_bytes == est.output_bytes + est.scratch_bytes == (4 * N + L * N) * 8 + 15 * 8 * nb + 136
    assert est.required_bytes == est.device_required_bytes == est.output_bytes + est.scratch_bytes and est.host_required_bytes == 272 + sp.PY_ALLOWANCE_BYTES
    assert cupy.h2d == []  # zero host-to-device copies per step
    assert reads == [("uint64", 136)]  # ONE counted constant-size read: the raw packet
    assert cupy.launches == [("maple_syrup_spatial_matrix_step", (nb,), (128,)), ("maple_syrup_spatial_matrix_finalize", (1,), (256,))]
    assert est.terms["no_dense_rows"] and est.terms["host_packet_buffers"] == 2 and est.terms["static_shared_bytes_per_block"] == 15 * 8 * 128 + 8 * 128 + 4 * 128


def test_admission_precedes_every_allocation_conversion_and_compile_on_both_sides(cupy):
    rng = np.random.default_rng(1)
    L, N = 2, 400
    inputs, p, mask, np_col = pair(rng, L, N, "all_cell")
    col = device_owner(np_col)
    K = fx(np.full(N, 3e-6))
    kw = dict(conductivity_m_s=K, suction_m=0.1, vegetation_factor=0.5)
    dev_need = sm.estimate_spatial_matrix_memory(L, N, backend="cuda", operation="prepare").device_required_bytes
    host_need = sm.estimate_spatial_matrix_memory(L, N, backend="cuda", operation="prepare").host_required_bytes
    for budget, reader, hb, hrd in ((dev_need - 1, None, None, None), (BIG, lambda: dev_need - 1, None, None), (BIG, None, host_need - 1, None), (BIG, None, None, lambda: host_need - 1)):
        before = len(cupy.allocations)
        with pytest.raises(sp.SpatialLayerMemoryError):
            sm.prepare_spatial_matrix(col, **kw, memory_budget_bytes=budget, available_bytes_fn=reader, host_memory_budget_bytes=hb, host_available_bytes_fn=hrd)
        assert len(cupy.allocations) == before and cupy.modules == [] and cupy.launches == []  # nothing allocated, compiled or launched
    with pytest.raises(sp.SpatialLayerMemoryError):
        sm.prepare_spatial_matrix(col, **kw, memory_budget_bytes=dev_need, reserve_bytes=1)
    with pytest.raises(sp.SpatialLayerMemoryError):
        sm.prepare_spatial_matrix(col, **kw, memory_budget_bytes=BIG, host_memory_budget_bytes=host_need, host_reserve_bytes=1)
    op = sm.prepare_spatial_matrix(col, **kw, memory_budget_bytes=dev_need, host_memory_budget_bytes=host_need)  # exactly the requirements are admitted
    hneed = sm.estimate_spatial_matrix_memory(L, N, backend="cuda", operation="history")
    cupy.allocations.clear()
    with pytest.raises(sp.SpatialLayerMemoryError):
        sm.spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=hneed.device_required_bytes - 1)
    with pytest.raises(sp.SpatialLayerMemoryError):
        sm.spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=BIG, host_memory_budget_bytes=hneed.host_required_bytes - 1)
    assert cupy.allocations == []
    hist = sm.spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=hneed.device_required_bytes, host_memory_budget_bytes=hneed.host_required_bytes)
    state, z = dev_state(col, np.zeros((L, N))), fx(np.zeros(N))
    sneed = sm.estimate_spatial_matrix_memory(L, N, backend="cuda", operation="step")
    cupy.allocations.clear()
    n_launch = len(cupy.launches)
    for kws in (dict(memory_budget_bytes=sneed.device_required_bytes - 1), dict(memory_budget_bytes=BIG, host_memory_budget_bytes=sneed.host_required_bytes - 1),
                dict(memory_budget_bytes=BIG, available_bytes_fn=lambda: sneed.device_required_bytes - 1), dict(memory_budget_bytes=BIG, host_available_bytes_fn=lambda: sneed.host_required_bytes - 1),
                dict(memory_budget_bytes=sneed.device_required_bytes, reserve_bytes=1)):
        with pytest.raises(sp.SpatialLayerMemoryError):
            sm.spatial_matrix_step(op, z, z, z, state, hist, 60.0, **kws)
        assert cupy.allocations == [] and len(cupy.launches) == n_launch  # nothing allocated or launched by a refused step
    sm.spatial_matrix_step(op, z, z, z, state, hist, 60.0, memory_budget_bytes=sneed.device_required_bytes, host_memory_budget_bytes=sneed.host_required_bytes)
    for bad in (True, -1, 2.5):
        with pytest.raises(sp.SpatialLayerError):
            sm.spatial_matrix_step(op, z, z, z, state, hist, 60.0, memory_budget_bytes=BIG, host_available_bytes_fn=lambda b=bad: b)
    with pytest.raises(sp.SpatialLayerError):
        sm.spatial_matrix_step(op, z, z, z, state, hist, 60.0, memory_budget_bytes=BIG, host_available_bytes_fn=3)


def test_cpu_owners_refuse_the_cuda_only_arguments():
    rng = np.random.default_rng(1)
    inputs, p, mask, np_col = pair(rng, 2, 5, "all_cell")
    for kw in (dict(host_memory_budget_bytes=1), dict(host_reserve_bytes=1), dict(host_available_bytes_fn=lambda: 1)):
        with pytest.raises(sp.SpatialLayerError):
            sm.prepare_spatial_matrix(np_col, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG, **kw)
    op = sm.prepare_spatial_matrix(np_col, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        sm.spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=BIG, host_memory_budget_bytes=1)
    with pytest.raises(sp.SpatialLayerError):
        sm.verify_operator_identity(op, host_memory_budget_bytes=1)
    with pytest.raises(sp.SpatialLayerError):
        sm.estimate_spatial_matrix_memory(2, 5, backend="numpy", host_reserve_bytes=1)


def test_type_namespace_device_and_legacy_refusals(cupy):
    rng = np.random.default_rng(3)
    L, N = 2, 20
    inputs, p, mask, np_col = pair(rng, L, N, "all_cell")
    K, psi, g = np.full(N, 3e-6), np.full(N, 0.1), np.full(N, 0.5)
    col, op, hist = build(cupy, np_col, K, psi, g)
    state, z = dev_state(col, np.zeros((L, N))), fx(np.zeros(N))
    cop, chist = cpu_ref(np_col, K, psi, g, 0.001, 0.0)
    def fails(*a, **k):
        with pytest.raises(sp.SpatialLayerError):
            sm.spatial_matrix_step(*a, memory_budget_bytes=BIG, **k)
    before = len(cupy.allocations)
    fails(op, np.zeros(N), z, z, state, hist, 60.0)  # a NumPy array for a CUDA owner: no implicit transfer
    fails(op, fx(np.zeros(N, dtype=np.float32)), z, z, state, hist, 60.0)
    fails(op, fx(np.zeros(N + 1)), z, z, state, hist, 60.0)
    fails(op, fx(np.zeros(2 * N))[::2], z, z, state, hist, 60.0)
    fails(op, z, z, z, sp.spatial_column_state(np_col, np.zeros((L, N)), 0.0, memory_budget_bytes=BIG), hist, 60.0)  # a CPU state for a CUDA owner
    fails(op, z, z, z, state, chist, 60.0)  # a CPU history (other operator identity and namespace)
    with sm._minting():
        wrong_ns = sm.SpatialMatrixHistory(hist.wetting_m, hist.cumulative_intake_m, op.sha256, N)  # right identity, CPU namespace tag
    fails(op, z, z, z, state, wrong_ns, 60.0)
    fails(cop, np.zeros(N), np.zeros(N), np.zeros(N), state, chist, 60.0)  # a CUDA state for a CPU operator
    with pytest.raises(ss.SubsurfaceError):
        mi.matrix_step(op, z, z, z, state, hist, 60.0)  # the legacy entry points still refuse the spatial objects
    assert len(cupy.allocations) == before  # every refusal preceded every allocation
    cupy.rc.current = 1  # the owner's device is 0
    for call in (lambda: sm.spatial_matrix_step(op, z, z, z, state, hist, 60.0, memory_budget_bytes=BIG), lambda: sm.spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=BIG),
                 lambda: sm.prepare_spatial_matrix(col, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)):
        with pytest.raises(sp.SpatialLayerError, match="device"):
            call()
    cupy.rc.current = 0
    assert len(cupy.allocations) == before
    with pytest.raises(sp.SpatialLayerError):  # a wrong-device array
        sm.spatial_matrix_step(op, fx(np.zeros(N), 1), z, z, state, hist, 60.0, memory_budget_bytes=BIG)


def test_law_and_history_validation_on_the_device_covers_every_cell_and_publishes_nothing(cupy):
    rng = np.random.default_rng(4)
    L, N = 2, 40
    inputs, p, mask, np_col = pair(rng, L, N, "all_cell", mask=np.array([True, False] * 20))
    col = device_owner(np_col)
    good = dict(conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5)
    for name, bad in (("conductivity_m_s", -1e-9), ("conductivity_m_s", np.nan), ("conductivity_m_s", np.inf), ("suction_m", -0.1), ("suction_m", np.nan), ("vegetation_factor", 1.0000001),
                      ("vegetation_factor", -0.1), ("vegetation_factor", np.nan)):
        for pos in (0, 1):  # an active and an INACTIVE cell
            arr = np.full(N, good[name])
            arr[pos] = bad
            snap = arr.copy()
            with pytest.raises(sp.SpatialLayerError):
                sm.prepare_spatial_matrix(col, **{**good, name: fx(arr)}, memory_budget_bytes=BIG)
            np.testing.assert_array_equal(arr, snap)
        with pytest.raises(sp.SpatialLayerError):
            sm.prepare_spatial_matrix(col, **{**good, name: bad}, memory_budget_bytes=BIG)
    op = sm.prepare_spatial_matrix(col, **good, memory_budget_bytes=BIG)
    for which in ("wetting_m", "cumulative_intake_m"):
        for bad in (-1.0, np.nan, np.inf, -1e-12):
            for pos in (0, 1):
                arr = np.zeros(N)
                arr[pos] = bad
                with pytest.raises(sp.SpatialLayerError):
                    sm.spatial_matrix_history(op, **{"wetting_m": 0.0, "cumulative_intake_m": 0.0, which: fx(arr)}, memory_budget_bytes=BIG)
        with pytest.raises(sp.SpatialLayerError):
            sm.spatial_matrix_history(op, **{"wetting_m": 0.0, "cumulative_intake_m": 0.0, which: -1.0}, memory_budget_bytes=BIG)
    with pytest.raises(TypeError):
        sm.spatial_matrix_history(op, wetting_m=0.0, memory_budget_bytes=BIG)  # nothing defaults to zero
    src = np.full(N, 0.004)
    h = sm.spatial_matrix_history(op, wetting_m=fx(src), cumulative_intake_m=0.5, memory_budget_bytes=BIG)
    src[:] = 9.0
    np.testing.assert_array_equal(np.asarray(h.wetting_m), 0.004)  # fresh owned arrays, not aliases of the caller
    np.testing.assert_array_equal(np.asarray(h.cumulative_intake_m), 0.5)
    assert h.namespace == "cupy" and h.device_id == 0 and h.summary()["boundary_enforced"] is False


# ---- identity, caches, errors, release --------------------------------------------------------------------------------------------------
def test_the_device_identity_recipe_equals_the_cpu_stream_and_is_bounded(cupy):
    class Rec:
        def __init__(self):
            self.data = bytearray()

        def update(self, b):
            self.data += bytes(b)

    arr = np.linspace(1e-7, 4e-6, 1000)
    cpu = hashlib.sha256()
    sm._hash_array(cpu, "conductivity_m_s", "cell", arr)
    for size in (1, 7, 64, 1000, 4096):
        h = hashlib.sha256()
        tc.hash_device_array(h, "conductivity_m_s", "cell", fx(arr), np.empty(size))
        assert h.hexdigest() == cpu.hexdigest()  # identical bytes for every staging size (chunk-boundary independence)
    rng = np.random.default_rng(1)
    inputs, p, mask, np_col = pair(rng, 2, 3000, "all_cell")
    col = device_owner(np_col)
    gets = []
    base_get = hostb2c_get = type(fx(arr)).get

    def counting_get(self, out=None):
        gets.append(0 if out is None else out.size)
        return hostb2c_get(self, out)

    type(fx(arr)).get = counting_get
    try:
        op = sm.prepare_spatial_matrix(col, conductivity_m_s=fx(np.linspace(1e-7, 1e-5, 3000)), suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)
        n_prepare = len(gets)
        assert max(gets) <= tc.STAGE_ELEMS  # every staged read is at most the counted staging buffer; no full mirror
        gets.clear()
        sm.spatial_matrix_step(op, fx(np.zeros(3000)), fx(np.zeros(3000)), fx(np.zeros(3000)), dev_state(col, np.zeros((2, 3000))), sm.spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0,
                                                                                                                                                       memory_budget_bytes=BIG), 60.0, memory_budget_bytes=BIG)
        assert gets == []  # a step never hashes and never mirrors an array to the host (its only read is the packet)
        assert n_prepare >= 1
    finally:
        type(fx(arr)).get = base_get
    assert op.prepare_memory.host_required_bytes == min(3000, tc.STAGE_ELEMS) * 8 + 8 + sp.PY_ALLOWANCE_BYTES


def test_cold_verification_streams_through_the_admitted_host_buffer_and_detects_tampering(cupy):
    rng = np.random.default_rng(2)
    inputs, p, mask, np_col = pair(rng, 2, 500, "all_cell")
    col = device_owner(np_col)
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=fx(np.linspace(1e-7, 1e-5, 500)), suction_m=0.1, vegetation_factor=fx(np.full(500, 0.5)), memory_budget_bytes=BIG)
    need = 500 * 8 + sp.PY_ALLOWANCE_BYTES
    assert sm.verify_operator_identity(op, host_memory_budget_bytes=need) is True
    with pytest.raises(sp.SpatialLayerMemoryError):
        sm.verify_operator_identity(op, host_memory_budget_bytes=need - 1)
    with pytest.raises(sp.SpatialLayerMemoryError):
        sm.verify_operator_identity(op, host_available_bytes_fn=lambda: need - 1)
    with pytest.raises(sp.SpatialLayerMemoryError):
        sm.verify_operator_identity(op, host_memory_budget_bytes=need, host_reserve_bytes=1)
    object.__setattr__(op, "_law", (fx(np.linspace(2e-7, 1e-5, 500)), op._law[1], op._law[2]))
    assert sm.verify_operator_identity(op) is False


def test_the_kernel_cache_holds_functions_only_per_device_and_compiles_once(cupy):
    rng = np.random.default_rng(2)
    inputs, p, mask, np_col = pair(rng, 2, 10, "all_cell")
    c0, c1 = device_owner(np_col, 0), device_owner(np_col, 1)
    cupy.rc.current = 0
    sm.prepare_spatial_matrix(c0, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)
    sm.prepare_spatial_matrix(c0, conductivity_m_s=2e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)
    assert len(cupy.modules) == 1 and set(tc._FUNCTIONS) == {0}
    cupy.rc.current = 1
    sm.prepare_spatial_matrix(c1, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)
    assert len(cupy.modules) == 2 and set(tc._FUNCTIONS) == {0, 1}
    for fns in tc._FUNCTIONS.values():
        assert sorted(vars(fns)) == ["finalize", "step", "validate"]  # compiled functions only: no array, owner, state or history
    code, options, backend_name = cupy.modules[0]
    assert options == tc.COMPILE_OPTIONS and "--fmad=false" in options and backend_name == "nvrtc" and code == tc.render_source()


def test_failed_compile_and_device_errors_are_typed_and_publish_nothing(cupy, monkeypatch):
    rng = np.random.default_rng(2)
    L, N = 2, 20
    inputs, p, mask, np_col = pair(rng, L, N, "all_cell")
    col = device_owner(np_col)

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("synthetic NVRTC failure")

    real_module = cupy.RawModule
    cupy.RawModule = Boom
    with pytest.raises(sp.SpatialLayerBackendUnavailableError, match="no fallback"):
        sm.prepare_spatial_matrix(col, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)
    cupy.RawModule = real_module
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)
    hist = sm.spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    state, z = dev_state(col, np.zeros((L, N))), fx(np.zeros(N))
    real_empty = cupy.empty

    def oom(shape, dtype=np.float64):
        raise MemoryError("synthetic device out of memory")

    cupy.empty = oom
    with pytest.raises(sp.SpatialLayerMemoryError):
        sm.spatial_matrix_step(op, z, z, z, state, hist, 60.0, memory_budget_bytes=BIG)
    cupy.empty = real_empty

    class CUDARuntimeError(Exception):
        pass

    fns = tc._FUNCTIONS[0]
    real_step = fns.step
    fns.step = lambda *a, **k: (_ for _ in ()).throw(CUDARuntimeError("synthetic launch failure"))
    with pytest.raises(sp.SpatialLayerBackendUnavailableError):
        sm.spatial_matrix_step(op, z, z, z, state, hist, 60.0, memory_budget_bytes=BIG)
    fns.step = real_step
    np.testing.assert_array_equal(np.asarray(state.water_m), 0.0)  # the accepted state is untouched
    ok = sm.spatial_matrix_step(op, z, z, z, state, hist, 60.0, memory_budget_bytes=BIG)
    assert ok.budget.intake_m3 == 0.0


def test_owner_release_and_no_module_level_retention(cupy):
    rng = np.random.default_rng(2)
    inputs, p, mask, np_col = pair(rng, 2, 20, "all_cell")
    col = device_owner(np_col)
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=fx(np.full(20, 1e-5)), suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)
    hist = sm.spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    z = fx(np.zeros(20))
    r = sm.spatial_matrix_step(op, z, z, z, dev_state(col, np.zeros((2, 20))), hist, 60.0, memory_budget_bytes=BIG)
    refs = [weakref.ref(x) for x in (op, hist, r.history, r.state, r.surface_pool_m, r.intake_m, op._law[0])]
    del op, hist, r
    gc.collect()
    assert all(w() is None for w in refs)


def test_estimates_are_pure_arithmetic_and_the_cpu_model_is_unchanged():
    for L, N in ((1, 1), (3, 1000), (5, 257)):
        e = sm.estimate_spatial_matrix_memory(L, N, backend="cuda", operation="step", reserve_bytes=7, host_reserve_bytes=11)
        nb = (N + 127) // 128
        assert e.device_required_bytes == e.required_bytes == (4 * N + L * N) * 8 + 15 * 8 * nb + 136 + 7
        assert e.host_required_bytes == 272 + sp.PY_ALLOWANCE_BYTES + 11 and e.host_reserve_bytes == 11 and e.backend == "cuda"
        h = sm.estimate_spatial_matrix_memory(L, N, backend="cuda", operation="history")
        assert h.device_required_bytes == 2 * N * 8 + 8 and h.host_required_bytes == 8 + sp.PY_ALLOWANCE_BYTES
        pr = sm.estimate_spatial_matrix_memory(L, N, backend="cuda", operation="prepare")
        assert pr.device_required_bytes == 3 * N * 8 + 8 and pr.host_required_bytes == min(N, tc.STAGE_ELEMS) * 8 + 8 + sp.PY_ALLOWANCE_BYTES
    cpu = sm.estimate_spatial_matrix_memory(3, 1000, backend="numba", operation="step", reserve_bytes=7)
    per = mi.KERNEL_SCRATCH_DOUBLES_PER_CELL
    assert cpu.required_bytes == (4 * 1000 + 3000) * 8 + mi.ROW_COUNT * 1000 * 8 + per * 1000 * 8 + mi.SCRATCH_MASK_BYTES_PER_CELL * 1000 + mi._PACKET_SCRATCH_BYTES + 48 + sp.PY_ALLOWANCE_BYTES + 7
    assert cpu.host_required_bytes == cpu.required_bytes and cpu.device_required_bytes == 0
    with pytest.raises(Exception):
        sm.estimate_spatial_matrix_memory(2, 2**60, backend="cuda", operation="step")  # launch-range refusal before any arithmetic is trusted


@pytest.mark.parametrize("case", sorted(HISTORICAL_PRE_REVIEW02_REFUSALS), ids=lambda c: f"L{c[0]}-N{c[1]}-{c[2]}")
def test_the_original_multi_cell_sequences_now_succeed_in_all_four_stages(cupy, case):
    """Review02: the two valid-input sequences that the pre-review02 source refused at their second step now integrate on ALL four stages, on the CPU spatial NumPy path and on the host-executed CUDA text. The
    recorded cells integrate with the intake bounded by the pool (pool-out exactly 0, not negative), the history/ledger consistent, and the device agrees with the CPU within the declared tolerances."""
    L, N, layout = case
    old = HISTORICAL_PRE_REVIEW02_REFUSALS[case]
    c = sequence_case(L, N, layout)
    np_col = sp.prepare_spatial_column(a3.spec_from(c.inputs, N), c.mask, AREA, backend="numpy", memory_budget_bytes=BIG)
    cop, chist = cpu_ref(np_col, c.K, c.psi, c.g, c.j0, 0.01)
    col, op, hist = build(cupy, np_col, c.K, c.psi, c.g, j0=c.j0, c0=0.01)
    cstate, dstate = sp.spatial_column_state(np_col, c.water, 55.5, memory_budget_bytes=BIG), dev_state(col, c.water, 55.5)
    for i, (dt, pool, standing, rate) in enumerate(c.steps):
        snap = [np.array(a) for a in (cstate.water_m, chist.wetting_m, chist.cumulative_intake_m, pool, standing, rate)]
        got = sm.spatial_matrix_step(op, fx(pool), fx(standing), fx(rate), dstate, hist, dt, memory_budget_bytes=BIG)
        ref = sm.spatial_matrix_step(cop, pool, standing, rate, cstate, chist, dt, memory_budget_bytes=BIG)
        for a, b in ((cstate.water_m, snap[0]), (chist.wetting_m, snap[1]), (chist.cumulative_intake_m, snap[2]), (pool, snap[3]), (standing, snap[4]), (rate, snap[5])):
            np.testing.assert_array_equal(a, b)  # inputs unchanged
        for x, y in ((got.state.water_m, ref.state.water_m), (got.surface_pool_m, ref.surface_pool_m), (got.intake_m, ref.intake_m), (got.history.wetting_m, ref.history.wetting_m),
                     (got.history.cumulative_intake_m, ref.history.cumulative_intake_m)):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=RTOL, atol=ATOL)
        assert (ref.surface_pool_m >= 0.0).all() and (np.asarray(got.surface_pool_m) >= 0.0).all()
        assert (ref.intake_m <= pool + 0.0).all() and got.state.t_s == 55.5 and abs(ref.budget.pool_residual_m3) <= ref.budget.bound_m3
        if i == old["step"]:
            cell = old["cell"]
            assert pool[cell] == old["pool"] and ref.intake_m[cell] <= pool[cell] and ref.surface_pool_m[cell] >= 0.0 > old["old_pool_out"]
            assert ref.intake_m[cell] == pool[cell] and ref.surface_pool_m[cell] == 0.0  # the pool-limited cell now exhausts its pool exactly
        cstate, chist, dstate, hist = ref.state, ref.history, got.state, got.history
