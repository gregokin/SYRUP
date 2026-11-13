"""Adjacent-layer diffusive redistribution: two-layer analytic exchange, direction, donor/receiver limits, stability at
very large dt with thin layers, and convergence against INDEPENDENT oracles (a discrete-ODE matrix exponential for
timestep refinement, a fine explicit finite-difference solution of the continuum diffusion equation for layer
refinement). The oracles are written here from the physical coefficient D only, never from kernel output. Nothing
here was run by its author; Codex records results."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from column_cases import AREA, column, forcing, layers, state_of

from maple_syrup.subsurface import column_state, soil_layers, step


def run(col, state, dt):
    add, pet = forcing(col)
    return step(col, state, dt, added_water_m=add, potential_evaporation_m_s=pet)


def two_layer_amount(dz0, dz1, D, w0, w1, dt):
    """Exact two-compartment exchange (positive downward) from the physical D and the centre separation."""
    G = D / ((dz0 + dz1) / 2.0)
    s = 1.0 / dz0 + 1.0 / dz1
    return (w0 / dz0 - w1 / dz1) / s * (-math.expm1(-G * s * dt))


def test_two_layer_analytic_exchange_downward():
    col = column(layers(gravity=False, diffusivity=2e-6), n_cells=1)
    state = state_of(col, [[0.1, 0.1]])  # theta 0.4 over theta 0.1333: downward
    result = run(col, state, 1e4)
    expected = two_layer_amount(0.25, 0.75, 2e-6, 0.1, 0.1, 1e4)
    assert 0.0 < expected < 0.1 - 0.015625  # inside both bounds: the unbounded formula applies
    assert result.diffusion_m[0, 0] == pytest.approx(expected, rel=1e-12)
    assert result.state.water_m[0, 0] == pytest.approx(0.1 - expected, rel=1e-12)
    assert result.state.water_m[1, 0] == pytest.approx(0.1 + expected, rel=1e-12)
    assert result.percolation_m[0, 0] == 0.0 and result.drainage_m[0] == 0.0
    assert abs(result.budget.residual_m3) <= result.budget.bound_m3


def test_two_layer_analytic_exchange_upward():
    col = column(layers(gravity=False, diffusivity=2e-6), n_cells=1)
    state = state_of(col, [[0.03125, 0.3]])  # theta 0.125 under theta 0.4: upward
    result = run(col, state, 1e4)
    expected = two_layer_amount(0.25, 0.75, 2e-6, 0.03125, 0.3, 1e4)
    assert expected < 0.0
    assert result.diffusion_m[0, 0] == pytest.approx(expected, rel=1e-12)
    assert result.state.water_m[0, 0] == pytest.approx(0.03125 - expected, rel=1e-12)
    assert result.state.water_m[1, 0] == pytest.approx(0.3 + expected, rel=1e-12)


def test_equalisation_at_very_large_dt_conserves_water():
    col = column(layers(gravity=False, diffusivity=2e-6), n_cells=1)
    state = state_of(col, [[0.1, 0.1]])
    result = run(col, state, 1e15)
    theta = result.state.water_m[:, 0] / np.array([0.25, 0.75])
    assert theta[0] == pytest.approx(theta[1], rel=1e-12)
    assert result.state.water_m.sum() == pytest.approx(0.2, rel=1e-14)
    assert abs(result.budget.residual_m3) <= result.budget.bound_m3


def test_donor_limit_leaves_exactly_the_residual():
    col = column(layers(gravity=False, diffusivity=1e-3), n_cells=1)  # G s dt = 1e4: the exchange would equalise
    state = state_of(col, [[0.0234375, 0.0]])  # layer 0 holds 0.0078125 above its residual 0.015625
    result = run(col, state, 1e6)
    unbounded = two_layer_amount(0.25, 0.75, 1e-3, 0.0234375, 0.0, 1e6)
    assert unbounded > 0.0078125
    np.testing.assert_array_equal(result.state.water_m, [[0.015625], [0.0078125]])
    assert result.diffusion_m[0, 0] == 0.0078125


def test_receiver_limit_fills_exactly_to_saturation():
    lay = layers(gravity=False, diffusivity=1e-3, res=[0.0625, 0.0625], fc=[0.25, 0.125], sat=[0.5, 0.25])
    col = column(lay, n_cells=1)
    state = state_of(col, [[0.125, 0.171875]])  # layer 1 saturation depth 0.1875: 0.015625 of room
    result = run(col, state, 1e6)
    unbounded = two_layer_amount(0.25, 0.75, 1e-3, 0.125, 0.171875, 1e6)
    assert unbounded > 0.015625
    np.testing.assert_array_equal(result.state.water_m, [[0.109375], [0.1875]])
    assert result.diffusion_m[0, 0] == 0.015625


def test_thin_layers_and_huge_dt_stay_bounded_and_conservative():
    edges = np.linspace(0.0, 0.01, 11)  # ten 1 mm layers
    lay = soil_layers(edges, theta_residual=0.05, theta_field_capacity=0.3, theta_saturation=0.45,
                      diffusivity_m2_s=1e-6, theta_evaporation_stress=0.2, gravity_drainage=False)
    col = column(lay, n_cells=5)
    rng = np.random.default_rng(3)
    theta = rng.uniform(0.05, 0.45, (10, 5))
    state = column_state(col, np.ascontiguousarray(theta * 0.001))
    total0 = state.water_m.sum(axis=0)
    for _ in range(20):
        result = run(col, state, 1e9)
        state = result.state
        assert np.all(np.isfinite(state.water_m))
        assert np.all(state.water_m >= lay.residual_m[:, None] - 1e-18)
        assert np.all(state.water_m <= lay.saturation_m[:, None] + lay.capacity_tolerance_m[:, None])
        assert abs(result.budget.residual_m3) <= result.budget.bound_m3
    np.testing.assert_allclose(state.water_m.sum(axis=0), total0, rtol=1e-12)


def ode_solution(dz, D, w0, T):
    """Matrix exponential of the linear interface-flux ODE dw/dt = A w (no bounds active), built from D only."""
    L = dz.shape[0]
    G = D / ((dz[:-1] + dz[1:]) / 2.0)
    A = np.zeros((L, L))
    for k in range(L - 1):
        A[k, k] -= G[k] / dz[k]
        A[k, k + 1] += G[k] / dz[k + 1]
        A[k + 1, k] += G[k] / dz[k]
        A[k + 1, k + 1] -= G[k] / dz[k + 1]
    values, vectors = np.linalg.eig(A)
    coefficients = np.linalg.solve(vectors, w0)
    return (vectors @ (np.exp(values * T) * coefficients)).real


def test_timestep_refinement_converges_to_the_discrete_ode_oracle():
    dz = np.full(4, 0.1)
    D, T = 1e-7, 1e5
    lay = soil_layers(np.concatenate([[0.0], np.cumsum(dz)]), theta_residual=0.0, theta_field_capacity=1.0,
                      theta_saturation=1.0, diffusivity_m2_s=D, theta_evaporation_stress=0.5, gravity_drainage=False)
    col = column(lay, n_cells=1)
    theta0 = np.array([0.4, 0.3, 0.2, 0.1])
    w0 = theta0 * dz
    oracle = ode_solution(dz, D, w0, T) / dz
    errors = []
    for n_steps in (1, 4, 16, 64):
        state = column_state(col, np.ascontiguousarray(w0[:, None]))
        for _ in range(n_steps):
            state = run(col, state, T / n_steps).state
        errors.append(float(np.max(np.abs(state.water_m[:, 0] / dz - oracle))))
        assert state.water_m.sum() == pytest.approx(w0.sum(), rel=1e-13)
    assert all(b < a for a, b in itertools.pairwise(errors)), errors
    # first-order (Lie) splitting: a fourfold refinement should cut the error well below 3/4 once dt resolves 1/k
    assert all(b < 0.75 * a for a, b in itertools.pairwise(errors[1:])), errors
    assert errors[-1] < 0.03, errors


def continuum_solution(theta_fine, D, dz_fine, T, n_steps):
    """Explicit finite differences of d theta/dt = D d2 theta/dz2 with no-flux ends (stable: D dt/dz2 < 0.5)."""
    theta = theta_fine.copy()
    r = D * (T / n_steps) / dz_fine ** 2
    assert r < 0.5
    for _ in range(n_steps):
        flux = r * (theta[1:] - theta[:-1])
        theta[:-1] += flux
        theta[1:] -= flux
    return theta


def test_layer_refinement_converges_to_the_continuum_oracle_with_fixed_physical_d():
    D, T, depth, M = 1e-7, 1e5, 1.0, 256
    z = (np.arange(M) + 0.5) * (depth / M)
    theta_fine0 = np.where(z < 0.5, 0.3, 0.1)
    theta_fine = continuum_solution(theta_fine0, D, depth / M, T, 2700)
    errors = []
    for L in (2, 4, 8, 16):
        dz = np.full(L, depth / L)
        lay = soil_layers(np.linspace(0.0, depth, L + 1), theta_residual=0.0, theta_field_capacity=1.0,
                          theta_saturation=1.0, diffusivity_m2_s=D, theta_evaporation_stress=0.5,
                          gravity_drainage=False)
        col = column(lay, n_cells=1)
        theta0 = theta_fine0.reshape(L, -1).mean(axis=1)
        state = column_state(col, np.ascontiguousarray((theta0 * dz)[:, None]))
        for _ in range(1000):
            state = run(col, state, T / 1000).state
        theta = state.water_m[:, 0] / dz
        reference = theta_fine.reshape(L, -1).mean(axis=1)
        errors.append(float(np.sqrt(np.mean((theta - reference) ** 2))))
    assert all(b < a for a, b in itertools.pairwise(errors)), errors
    assert errors[-1] < 0.01, errors


def test_diffusion_budget_is_internal_only():
    col = column(layers(gravity=False, diffusivity=5e-7), n_cells=3)
    state = state_of(col, [[0.1, 0.05], [0.02, 0.3], [0.0625, 0.1875]])
    result = run(col, state, 3600.0)
    b = result.budget
    assert b.added_m3 == b.surface_return_m3 == b.drainage_m3 == b.evaporation_m3 == 0.0
    assert b.storage_after_m3 == pytest.approx(b.storage_before_m3, abs=b.bound_m3)
    assert b.storage_before_m3 == pytest.approx((0.15 + 0.32 + 0.25) * AREA, rel=1e-14)
