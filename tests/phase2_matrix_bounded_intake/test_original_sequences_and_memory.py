"""The original multi-cell seeded history sequences (same seeds and draw order as the gated definitions, loaded unchanged from `tests/phase2_spatial_matrix_cuda/conftest.py`) now succeed on ALL FOUR stages on the
CPU spatial NumPy and Numba paths; plus the NumPy temporary-peak accounting of the bound. Author-written; nothing is skipped or altered."""
from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pytest
from conftest import ATOL, BIG, CANDIDATE, RTOL, _load, backend, mi, sm, sp  # noqa: F401

cuda_conftest = _load("cuda_conftest_review02", CANDIDATE / "tests" / "phase2_spatial_matrix_cuda" / "conftest.py")
CASES = sorted(cuda_conftest.HISTORICAL_PRE_REVIEW02_REFUSALS)


def make(c, backend):
    col = sp.prepare_spatial_column(cuda_conftest.a3.spec_from(c.inputs, c.N), c.mask, cuda_conftest.AREA, backend=backend, memory_budget_bytes=BIG)
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=c.K, suction_m=c.psi, vegetation_factor=c.g, memory_budget_bytes=BIG)
    hist = sm.spatial_matrix_history(op, wetting_m=c.j0, cumulative_intake_m=0.01, memory_budget_bytes=BIG)
    return col, op, hist, sp.spatial_column_state(col, c.water, 55.5, memory_budget_bytes=BIG)


@pytest.mark.parametrize("case", CASES, ids=lambda k: f"L{k[0]}-N{k[1]}-{k[2]}")
def test_the_original_sequences_succeed_on_all_four_stages_on_both_cpu_backends(case):
    pytest.importorskip("numba")
    L, N, layout = case
    c = cuda_conftest.sequence_case(L, N, layout)
    runs = {}
    for b in ("numpy", "numba"):
        col, op, hist, state = make(c, b)
        out = []
        for dt, pool, standing, rate in c.steps:
            r = sm.spatial_matrix_step(op, pool, standing, rate, state, hist, dt, memory_budget_bytes=BIG)  # a refusal would raise: every stage must succeed
            assert r.state.t_s == 55.5 and (r.surface_pool_m >= 0.0).all() and (r.intake_m <= pool).all() and np.isfinite(r.state.water_m).all()
            assert abs(r.budget.soil_residual_m3) <= r.budget.bound_m3 and abs(r.budget.pool_residual_m3) <= r.budget.bound_m3
            out.append(r)
            state, hist = r.state, r.history
        runs[b] = out
    assert len(runs["numpy"]) == 4
    for a, b in zip(runs["numpy"], runs["numba"], strict=True):
        for f in ("surface_pool_m", "intake_m"):
            np.testing.assert_allclose(getattr(b, f), getattr(a, f), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(b.history.wetting_m, a.history.wetting_m, rtol=RTOL, atol=ATOL)
    old = cuda_conftest.HISTORICAL_PRE_REVIEW02_REFUSALS[case]
    cell, r1 = old["cell"], runs["numpy"][old["step"]]
    assert c.steps[old["step"]][1][cell] == old["pool"] and r1.intake_m[cell] == old["pool"] and r1.surface_pool_m[cell] == 0.0  # the recorded cell exhausts its pool exactly


def test_the_numpy_temporary_peak_with_the_bound_is_within_the_unchanged_model():
    """The bound adds two transient boolean masks and keeps `room` alive a few statements longer; the step stays within the unchanged admitted estimate (no quota was raised)."""
    N, L = 50000, 3
    rng = np.random.default_rng(1)
    c = dict(depth_edges_m=[0.0, 0.2, 0.6, 1.0], theta_residual=[0.03] * 3, theta_field_capacity=[0.12] * 3, theta_saturation=[0.36] * 3)
    lay = mi.ss.soil_layers(np.array(c["depth_edges_m"]), theta_residual=np.array(c["theta_residual"]), theta_field_capacity=np.array(c["theta_field_capacity"]), theta_saturation=np.array(c["theta_saturation"]),
                            theta_evaporation_stress=0.5, diffusivity_m2_s=0.0)
    col = mi.ss.prepare_column(lay, np.ones(N, bool), 1.0, backend="numpy")
    op = mi.prepare_matrix(col, conductivity_m_s=rng.uniform(1e-6, 2e-5, N), suction_m=0.15, vegetation_factor=rng.uniform(0.2, 1.0, N))
    hist = mi.wetting_history(op, wetting_m=0.001, cumulative_intake_m=0.0)
    state = mi.ss.column_state(col, np.ascontiguousarray(np.full((L, N), 0.05)), 0.0)
    pool, standing, rate = np.full(N, 0.02), np.full(N, 0.005), np.zeros(N)
    mi.matrix_step(op, pool, standing, rate, state, hist, 600.0)
    gc.collect()
    tracemalloc.start()
    try:
        mi.matrix_step(op, pool, standing, rate, state, hist, 600.0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    est = mi.estimate_matrix_memory(L, N, backend="numpy")
    assert peak <= est.step_allocation_bytes, (peak, est.step_allocation_bytes)
