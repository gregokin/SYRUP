"""Bare-soil upper-layer evaporation (Tietjen-inspired, MAHLERAN calc_et.f90 form without vegetation): dry, quadratic
and wet regimes, the above-potential factor, the residual cap, and the no-gravity closed form over many steps through
the streaming initialiser. Nothing here was run by its author; Codex records results."""
from __future__ import annotations

import numpy as np
import pytest
from column_cases import column, forcing, layers, state_of

from maple_syrup.subsurface import (
    ForcingRecord,
    initialize_profile,
    soil_layers,
    step,
    uniform_state,
)


def run(col, state, dt, pet):
    add, rate = forcing(col, pet=pet)
    return step(col, state, dt, added_water_m=add, potential_evaporation_m_s=rate)


def test_zero_at_or_below_residual():
    col = column(layers(gravity=False, factor=1.2), n_cells=2)
    state = state_of(col, [[0.015625, 0.1875], [0.0078125, 0.1875]])  # theta0 = residual, below residual
    result = run(col, state, 1000.0, [1e-6, 1e-6])
    np.testing.assert_array_equal(result.evaporation_m, [0.0, 0.0])
    np.testing.assert_array_equal(result.evaporation_unmet_m, [0.0, 0.0])
    assert result.state.water_m.tobytes() == state.water_m.tobytes()


def test_quadratic_regime_hand_value():
    col = column(layers(gravity=False, factor=1.2), n_cells=1)
    state = state_of(col, [[0.03125, 0.1875]])  # theta0 = 0.125 between residual 0.0625 and stress 0.25
    result = run(col, state, 1000.0, [1e-6])
    demand = ((1.2 * 1e-6) * 1000.0) * 0.25  # (0.125 / 0.25)^2 = 0.25
    assert result.evaporation_m[0] == demand
    assert result.evaporation_unmet_m[0] == 0.0
    assert result.state.water_m[0, 0] == 0.03125 - demand
    assert result.state.water_m[1, 0] == 0.1875
    assert result.budget.evaporation_m3 == demand * 4.0


def test_wet_regime_uses_the_factor_above_potential():
    col = column(layers(gravity=False, factor=1.2), n_cells=1)
    state = state_of(col, [[0.125, 0.1875]])  # theta0 = 0.5 >= stress
    result = run(col, state, 1000.0, [1e-6])
    assert result.evaporation_m[0] == (1.2 * 1e-6) * 1000.0
    assert result.evaporation_m[0] > 1e-6 * 1000.0  # no "total <= reference potential" restriction


def test_withdrawal_is_capped_by_water_above_residual():
    col = column(layers(gravity=False, factor=1.0), n_cells=1)
    state = state_of(col, [[0.125, 0.1875]])
    result = run(col, state, 1000.0, [1.0])  # demand 1000 m: far above the 0.109375 m available
    assert result.evaporation_m[0] == 0.125 - 0.015625
    assert result.evaporation_unmet_m[0] == 1000.0 - (0.125 - 0.015625)
    assert result.state.water_m[0, 0] == 0.015625
    assert result.budget.evaporation_unmet_m3 == result.evaporation_unmet_m[0] * 4.0


def test_evaporation_acts_on_layer_zero_only():
    col = column(layers(gravity=False, factor=1.0), n_cells=1)
    state = state_of(col, [[0.015625, 0.375]])  # dry top at residual, saturated layer 1
    result = run(col, state, 1000.0, [1e-6])
    assert result.evaporation_m[0] == 0.0 and result.state.water_m[1, 0] == 0.375


def test_no_gravity_closed_form_over_many_streamed_steps():
    lay = soil_layers([0.0, 0.25], theta_residual=0.0625, theta_field_capacity=0.25, theta_saturation=0.5,
                      diffusivity_m2_s=0.0, theta_evaporation_stress=0.25, evaporation_factor=1.0,
                      gravity_drainage=False)
    col = column(lay, n_cells=2)
    state = uniform_state(col, [0.5])  # 0.125 m, stays above the stress depth 0.0625 throughout
    n, dt, pet = 100, 100.0, 1e-7

    def records():
        for _ in range(n):
            add, rate = forcing(col, pet=[pet, pet])
            yield ForcingRecord(dt, add, rate)

    profile = initialize_profile(col, state, records())
    expected = 0.125 - n * (pet * dt)
    np.testing.assert_allclose(profile.state.water_m, expected, rtol=1e-12)
    assert profile.cumulative.n_steps == n and profile.cumulative.elapsed_s == n * dt
    assert profile.cumulative.evaporation_m3 == pytest.approx(n * pet * dt * 2 * 4.0, rel=1e-12)
    assert profile.cumulative.evaporation_unmet_m3 == 0.0
    assert abs(profile.residual_m3) <= profile.bound_m3
