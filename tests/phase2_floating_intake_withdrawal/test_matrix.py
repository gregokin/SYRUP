"""Floating matrix intake: Root's independent controls, rock/sliver/zero depth, inactive placeholders, history retention and explicit rebinding, common full-soil reference."""
from __future__ import annotations

import numpy as np
import pytest
from builders import REFUSALS, column, state

from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import matrix_infiltration as mi
from maple_syrup import subsurface as ss

EDGES2 = [0.0, 0.2, 0.5]


def col(backend, depth, *, edges=EDGES2, active=None, ts=0.5, tf=0.3, tr=0.1):
    return column(edges, list(depth), tr=tr, tf=tf, ts=ts, backend=backend, active=active, stress=0.2)


def hist(op, j, c):
    return fm.floating_matrix_history(op, wetting_m=j, cumulative_intake_m=c)


def one(x):
    return np.array([x], dtype=np.float64)


def run(c, w, *, pool=0.01, standing=0.01, rate=0.0, k=2e-6, psi=0.0, g=0.5, dt=1000.0, j=0.01, cum=0.02):
    op = fm.prepare_floating_matrix(c, conductivity_m_s=k, suction_m=psi, vegetation_factor=g)
    s = state(c, w)
    return op, s, fm.floating_matrix_step(op, one(pool), one(standing), one(rate), s, hist(op, j, cum), dt)


def test_root_control_partial_depth(backend):
    c = col(backend, [0.1])
    _, s, r = run(c, [[0.01], [0.0]])
    assert np.allclose(r.state.water_m[:, 0], [0.011, 0.0], atol=1e-15)
    assert r.surface_pool_m[0] == pytest.approx(0.009, abs=1e-15)
    assert r.history.wetting_m[0] == pytest.approx(0.012, abs=1e-15)
    assert r.history.cumulative_intake_m[0] == pytest.approx(0.021, abs=1e-15)
    assert r.intake_m[0] == pytest.approx(0.001, abs=1e-15) and r.state.t_s == s.t_s


def test_root_control_zero_depth_changes_nothing(backend):
    c = col(backend, [0.0])
    _, _, r = run(c, np.zeros((2, 1)))
    assert np.all(r.state.water_m == 0.0) and r.intake_m[0] == 0.0 and r.surface_pool_m[0] == 0.01
    assert r.history.wetting_m[0] == 0.01 and r.history.cumulative_intake_m[0] == 0.02


def test_zero_depth_with_suction_has_no_nan(backend):
    c = col(backend, [0.0])
    _, _, r = run(c, np.zeros((2, 1)), psi=0.3, k=1e-5, g=0.0)
    assert np.all(np.isfinite(r.state.water_m)) and r.intake_m[0] == 0.0 and r.history.wetting_m[0] == 0.01 and r.history.cumulative_intake_m[0] == 0.02


def test_sliver_depth_keeps_its_actual_capacity(backend):
    d = 1e-9
    c = col(backend, [d])
    _, _, r = run(c, np.zeros((2, 1)), pool=0.5, standing=0.5, k=1e-3, g=1.0, dt=1000.0, j=0.0, cum=0.0)
    assert r.intake_m[0] == pytest.approx(0.5 * d, rel=1e-9)  # fills (just below) its porous capacity 0.5 * d, never epsilon-to-zero
    assert 0.0 < r.state.water_m[0, 0] <= 0.5 * d * (1 + 1e-12)


def test_inactive_placeholder_bits_and_history_preserved(backend):
    c = col(backend, [0.1, 0.1], active=np.array([True, False]))
    op = fm.prepare_floating_matrix(c, conductivity_m_s=2e-6, suction_m=0.0, vegetation_factor=0.5)
    w = np.array([[0.01, 0.7], [0.0, 0.7]])
    s = state(c, w)
    h = fm.floating_matrix_history(op, wetting_m=np.array([0.01, 0.3]), cumulative_intake_m=np.array([0.02, 0.4]))
    r = fm.floating_matrix_step(op, np.array([0.01, 0.2]), np.array([0.01, 0.2]), np.zeros(2), s, h, 1000.0)
    assert np.array_equal(r.state.water_m[:, 1], w[:, 1]) and r.surface_pool_m[1] == 0.2 and r.intake_m[1] == 0.0
    assert r.history.wetting_m[1] == 0.3 and r.history.cumulative_intake_m[1] == 0.4


def test_zero_time_is_identity(backend):
    c = col(backend, [0.1])
    _op, s, r = run(c, [[0.01], [0.0]], dt=0.0)
    assert np.array_equal(r.state.water_m, s.water_m) and r.surface_pool_m[0] == 0.01 and r.history.wetting_m[0] == 0.01


def test_invalid_inputs_refused_and_unchanged(backend):
    c = col(backend, [0.1])
    op = fm.prepare_floating_matrix(c, conductivity_m_s=2e-6, suction_m=0.0, vegetation_factor=0.5)
    s = state(c, [[0.01], [0.0]])
    h = hist(op, 0.01, 0.02)
    for bad in (np.array([np.nan]), np.array([-1.0])):
        with pytest.raises(REFUSALS):
            fm.floating_matrix_step(op, bad, one(0.01), one(0.0), s, h, 10.0)
    with pytest.raises(REFUSALS, match="nothing was allocated"):
        fm.floating_matrix_step(op, one(0.01), one(0.01), one(0.0), s, h, 10.0, memory_budget_bytes=0)
    with pytest.raises(REFUSALS):
        fm.prepare_floating_matrix(c, conductivity_m_s=-1.0, suction_m=0.0, vegetation_factor=0.5)
    with pytest.raises(REFUSALS):
        fm.prepare_floating_matrix(c, conductivity_m_s=1.0, suction_m=0.0, vegetation_factor=1.5)
    assert np.array_equal(s.water_m, np.array([[0.01], [0.0]])) and h.wetting_m[0] == 0.01


def test_history_is_bound_to_its_operator(backend):
    c = col(backend, [0.1])
    op1 = fm.prepare_floating_matrix(c, conductivity_m_s=2e-6, suction_m=0.0, vegetation_factor=0.5)
    op2 = fm.prepare_floating_matrix(c, conductivity_m_s=3e-6, suction_m=0.0, vegetation_factor=0.5)
    with pytest.raises(REFUSALS, match="different operator identity"):
        fm.floating_matrix_step(op2, one(0.01), one(0.01), one(0.0), state(c, [[0.01], [0.0]]), hist(op1, 0.01, 0.02), 10.0)


def test_rebinding_retains_exact_history_under_changed_parameters(backend):
    c1 = col(backend, [0.1, 0.15])
    c2 = col(backend, [0.12, 0.15], ts=0.45)  # changed depth and properties, same mask/edges/area
    op1 = fm.prepare_floating_matrix(c1, conductivity_m_s=2e-6, suction_m=0.1, vegetation_factor=0.5)
    op2 = fm.prepare_floating_matrix(c2, conductivity_m_s=5e-6, suction_m=0.2, vegetation_factor=0.7)
    j, cum = np.array([0.0123456789, 0.5]), np.array([0.3333333333333333, 1.5])
    h1 = fm.floating_matrix_history(op1, wetting_m=j, cumulative_intake_m=cum)
    rb = fm.rebind_floating_matrix_history(op1, h1, op2)
    assert rb.reset is False and rb.history.operator_sha256 == op2.sha256 and rb.old_operator_sha256 == op1.sha256
    assert np.array_equal(rb.history.wetting_m, j) and np.array_equal(rb.history.cumulative_intake_m, cum)
    assert rb.history.wetting_m is not h1.wetting_m
    # the new history is usable with the new operator, and the old one is not
    fm.floating_matrix_step(op2, np.zeros(2), np.zeros(2), np.zeros(2), state(c2, np.zeros((2, 2))), rb.history, 10.0)
    with pytest.raises(REFUSALS):
        fm.floating_matrix_step(op2, np.zeros(2), np.zeros(2), np.zeros(2), state(c2, np.zeros((2, 2))), h1, 10.0)


def test_rebinding_refusals(backend):
    c1 = col(backend, [0.1, 0.15])
    op1 = fm.prepare_floating_matrix(c1, conductivity_m_s=2e-6, suction_m=0.1, vegetation_factor=0.5)
    h1 = fm.floating_matrix_history(op1, wetting_m=0.1, cumulative_intake_m=0.2)
    other_edges = column([0.0, 0.2, 0.6], [0.1, 0.15], tr=0.1, tf=0.3, ts=0.5, backend=backend, stress=0.2)
    other_mask = col(backend, [0.1, 0.15], active=np.array([True, False]))
    other_area = column(EDGES2, [0.1, 0.15], tr=0.1, tf=0.3, ts=0.5, backend=backend, stress=0.2, area=2.0)
    for bad in (other_edges, other_mask, other_area, col(backend, [0.1, 0.15, 0.2])):
        opx = fm.prepare_floating_matrix(bad, conductivity_m_s=2e-6, suction_m=0.1, vegetation_factor=0.5)
        with pytest.raises(REFUSALS):
            fm.rebind_floating_matrix_history(op1, h1, opx)
    op2 = fm.prepare_floating_matrix(col(backend, [0.1, 0.15]), conductivity_m_s=3e-6, suction_m=0.1, vegetation_factor=0.5)
    with pytest.raises(REFUSALS, match="old the history is bound"):
        fm.rebind_floating_matrix_history(op2, h1, op1)  # history is not bound to the stated old operator
    with pytest.raises(REFUSALS, match="nothing was allocated"):
        fm.rebind_floating_matrix_history(op1, h1, op2, memory_budget_bytes=0)


def test_common_full_soil_reference(backend):
    """Full soil reproduces the common per-cell body (the same shared function) bit for bit on NumPy; the compiled kernel matches to its declared libm bound."""
    edges = [0.0, 0.1, 0.3, 0.6]
    c = column(edges, [0.6, 0.9, 0.7], tr=0.1, tf=0.3, ts=0.5, backend=backend, stress=0.2)
    rng = np.random.default_rng(11)
    dz = np.diff(edges)
    w = rng.uniform(0.05, 0.4, (3, 3)) * dz[:, None]
    pool, standing = rng.uniform(0.0, 0.02, 3), np.array([0.0, 0.01, 0.0])
    rate = pool / 600.0 * np.array([1.5, 1.0, 3.0])  # consistent supply split: a cell without contact never offers more than r dt
    op = fm.prepare_floating_matrix(c, conductivity_m_s=2e-6, suction_m=0.15, vegetation_factor=0.8)
    h = fm.floating_matrix_history(op, wetting_m=np.array([0.0, 0.01, 0.02]), cumulative_intake_m=np.zeros(3))
    r = fm.floating_matrix_step(op, pool, standing, rate, state(c, w), h, 600.0)
    coef = fm.volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    ctx = mi._StepContext(3, 3, np.ones(3, dtype=bool), mi.cell_closure_coefficient(3), mi.newton_tolerance())
    ref = mi._reference_step_body(ctx, np.full(3, 2e-6), np.full(3, 0.15), np.full(3, 0.8), dz, 0.3 * dz, 0.5 * dz, coef * (0.5 * dz), pool, standing, rate, w, h.wetting_m, h.cumulative_intake_m, 600.0)
    pairs = ((r.state.water_m, ref.water_m), (r.surface_pool_m, ref.pool_m), (r.history.wetting_m, ref.wetting_m), (r.history.cumulative_intake_m, ref.cumulative_intake_m), (r.intake_m, ref.intake_m))
    for a, b in pairs:
        if backend == "numpy":
            assert np.array_equal(a, b)
        else:
            assert np.allclose(a, b, rtol=2e-12, atol=1e-14)


def test_old_class_refuses_floating_owner(backend):
    c = col(backend, [0.1])
    with pytest.raises(REFUSALS):
        mi.prepare_matrix(c, conductivity_m_s=1e-6, suction_m=0.0, vegetation_factor=0.5)
