"""Hand-derived water balances, cascade destinations, identity and refusal contracts of one step (NumPy reference).
All values are dyadic so the expectations are exact. Nothing here was run by its author; Codex records results."""
from __future__ import annotations

import numpy as np
import pytest
from column_cases import AREA, column, forcing, from_cells, layers, state_of

from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.subsurface import (
    FLAG_MESSAGES,
    SoilColumnState,
    SubsurfaceBackendUnavailableError,
    SubsurfaceError,
    SubsurfaceStepError,
    column_state,
    default_host_available_bytes,
    prepare_column,
    step,
    storage_m3,
)


def run(col, state, dt, add=None, pet=None):
    added, rate = forcing(col, add, pet)
    return step(col, state, dt, added_water_m=added, potential_evaporation_m_s=rate)


def test_addition_within_capacity_hand_balance():
    col = column(layers(gravity=False), n_cells=4, active=[True, True, True, False])
    state = state_of(col, [[0.03125, 0.09375]] * 4)
    add = from_cells([[0.03125, 0.0625], [0.0, 0.015625], [0.0078125, 0.0], [0.0, 0.0]])
    state_bytes, add_bytes = state.water_m.tobytes(), add.tobytes()
    result = step(col, state, 60.0, added_water_m=add, potential_evaporation_m_s=np.zeros(4))
    np.testing.assert_array_equal(result.state.water_m, state.water_m + add)
    assert result.state.t_s == 60.0 and result.dt_s == 60.0
    for name in ("surface_return_m", "drainage_m", "evaporation_m", "evaporation_unmet_m"):
        np.testing.assert_array_equal(getattr(result, name), np.zeros(4))
    np.testing.assert_array_equal(result.percolation_m, np.zeros((1, 4)))
    np.testing.assert_array_equal(result.diffusion_m, np.zeros((1, 4)))
    b = result.budget
    assert b.storage_before_m3 == 0.125 * 3 * AREA
    assert b.added_m3 == (0.09375 + 0.015625 + 0.0078125) * AREA
    assert b.storage_after_m3 == b.storage_before_m3 + b.added_m3
    assert b.residual_m3 == 0.0 and b.surface_return_m3 == 0.0 and b.drainage_m3 == 0.0
    assert state.water_m.tobytes() == state_bytes and add.tobytes() == add_bytes


def test_addition_above_saturation_returns_to_surface():
    col = column(layers(gravity=False), n_cells=1)
    state = state_of(col, [[0.0625, 0.25]])
    result = run(col, state, 1.0, add=[[0.125, 0.25]])
    np.testing.assert_array_equal(result.state.water_m, [[0.125], [0.375]])  # exactly saturation
    assert result.surface_return_m[0] == 0.0625 + 0.125
    b = result.budget
    assert b.added_m3 == 0.375 * AREA and b.surface_return_m3 == 0.1875 * AREA
    assert b.storage_after_m3 - b.storage_before_m3 == b.added_m3 - b.surface_return_m3
    assert b.residual_m3 == 0.0


def test_gravity_cascade_destinations():
    col = column(layers(gravity=True), n_cells=4, active=[True, True, False, True])
    state = state_of(col, [[0.0625, 0.0625], [0.0625, 0.1875], [0.03125, 0.25], [0.0, 0.0]])
    result = run(col, state, 30.0, add=[[0.09375, 0.0], [0.0, 0.03125], [0.0, 0.0], [0.5, 0.0]])
    np.testing.assert_array_equal(result.state.water_m, from_cells(
        [[0.0625, 0.125], [0.0625, 0.1875], [0.03125, 0.25], [0.0625, 0.0625]]))
    np.testing.assert_array_equal(result.surface_return_m, [0.03125, 0.0, 0.0, 0.375])
    np.testing.assert_array_equal(result.percolation_m, [[0.0625, 0.0, 0.0, 0.0625]])
    np.testing.assert_array_equal(result.drainage_m, [0.0, 0.03125, 0.0, 0.0])
    np.testing.assert_array_equal(result.evaporation_m, np.zeros(4))
    b = result.budget
    assert b.storage_before_m3 == (0.125 + 0.25 + 0.0) * AREA
    assert b.added_m3 == (0.09375 + 0.03125 + 0.5) * AREA
    assert b.surface_return_m3 == (0.03125 + 0.375) * AREA and b.drainage_m3 == 0.03125 * AREA
    assert b.storage_after_m3 == (0.1875 + 0.25 + 0.125) * AREA
    assert b.residual_m3 == 0.0


def test_state_above_field_capacity_drains_only_with_gravity():
    for gravity, expected, drain in ((True, [[0.0625], [0.1875]], 0.0625), (False, [[0.0625], [0.25]], 0.0)):
        col = column(layers(gravity=gravity), n_cells=1)
        result = run(col, state_of(col, [[0.0625, 0.25]]), 10.0)
        np.testing.assert_array_equal(result.state.water_m, expected)
        assert result.drainage_m[0] == drain and result.percolation_m[0, 0] == 0.0


def test_dt_zero_is_identity_and_rejects_additions():
    col = column(layers(gravity=True), n_cells=2)
    state = state_of(col, [[0.0625, 0.25], [0.125, 0.375]], t_s=7.0)  # above field capacity: a dt > 0 step drains
    result = run(col, state, 0.0)
    assert result.state.water_m.tobytes() == state.water_m.tobytes()
    assert result.state.t_s == 7.0 and result.dt_s == 0.0
    assert result.budget.storage_after_m3 == result.budget.storage_before_m3 == storage_m3(col, state)
    assert result.budget.added_m3 == result.budget.drainage_m3 == result.budget.residual_m3 == 0.0
    assert not np.shares_memory(result.state.water_m, state.water_m)
    with pytest.raises(SubsurfaceStepError) as info:
        run(col, state, 0.0, add=[[0.001, 0.0], [0.0, 0.0]])
    assert info.value.bit == 9 and info.value.n_flagged == 1


def test_inactive_cells_are_unchanged_and_excluded():
    col = column(layers(gravity=True), n_cells=2, active=[True, False])
    state = state_of(col, [[0.0625, 0.25], [0.125, 0.375]])
    result = run(col, state, 5.0)
    np.testing.assert_array_equal(result.state.water_m, [[0.0625, 0.125], [0.1875, 0.375]])
    assert result.drainage_m.tolist() == [0.0625, 0.0]
    assert result.budget.storage_before_m3 == 0.3125 * AREA and result.budget.drainage_m3 == 0.0625 * AREA
    with pytest.raises(SubsurfaceStepError) as info:
        run(col, state, 5.0, add=[[0.0, 0.0], [0.001, 0.0]])
    assert info.value.bit == 5
    with pytest.raises(SubsurfaceStepError) as info:
        run(col, state, 5.0, pet=[0.0, 1e-7])
    assert info.value.bit == 8


@pytest.mark.parametrize("case", [
    ("state_nan", 0), ("state_negative", 1), ("state_above_saturation", 2), ("add_nan", 3), ("add_negative", 4),
    ("pet_inf", 6), ("pet_negative", 7),
])
def test_refusal_reports_the_bit_and_leaves_inputs_unchanged(case):
    name, bit = case
    col = column(layers(), n_cells=3)
    cells = [[0.0625, 0.125]] * 3
    if name == "state_nan":
        cells[1] = [np.nan, 0.125]
    elif name == "state_negative":
        cells[1] = [-1e-9, 0.125]
    elif name == "state_above_saturation":
        cells[1] = [0.125 + 1e-6, 0.125]
    state = state_of(col, cells)
    add, pet = forcing(col)
    if name == "add_nan":
        add[1, 2] = np.nan
    elif name == "add_negative":
        add[0, 0] = -1e-9
    elif name == "pet_inf":
        pet[2] = np.inf
    elif name == "pet_negative":
        pet[0] = -1e-9
    before = (state.water_m.tobytes(), add.tobytes(), pet.tobytes())
    with pytest.raises(SubsurfaceStepError) as info:
        step(col, state, 3.0, added_water_m=add, potential_evaporation_m_s=pet)
    assert info.value.bit == bit and info.value.n_flagged == 1
    assert FLAG_MESSAGES[bit] in str(info.value)
    assert (state.water_m.tobytes(), add.tobytes(), pet.tobytes()) == before


def test_lowest_bit_is_reported_first():
    col = column(layers(), n_cells=3)
    state = state_of(col, [[0.0625, 0.125], [0.0625, 0.125], [np.nan, 0.125]])
    add, pet = forcing(col)
    add[0, 0] = -1.0
    with pytest.raises(SubsurfaceStepError) as info:
        step(col, state, 3.0, added_water_m=add, potential_evaporation_m_s=pet)
    assert info.value.bit == 0 and info.value.n_flagged == 2


@pytest.mark.parametrize("dt", [-1.0, float("nan"), float("inf"), True, "1", None])
def test_invalid_dt_is_refused_before_any_step(dt):
    col = column(layers(), n_cells=1)
    state = state_of(col, [[0.0625, 0.125]])
    add, pet = forcing(col)
    with pytest.raises(SubsurfaceError) as info:
        step(col, state, dt, added_water_m=add, potential_evaporation_m_s=pet)
    assert not isinstance(info.value, SubsurfaceStepError)


def test_source_time_must_be_finite_and_candidate_time_must_not_overflow():
    col = column(layers(gravity=False), n_cells=1)
    water = from_cells([[0.0625, 0.125]])
    add, pet = forcing(col)
    for bad_t in (float("nan"), float("inf"), -float("inf"), True, "0", None):
        state = SoilColumnState(water.copy(), bad_t)  # direct construction bypasses column_state's check
        with pytest.raises(SubsurfaceError) as info:
            step(col, state, 1.0, added_water_m=add, potential_evaporation_m_s=pet)
        assert not isinstance(info.value, SubsurfaceStepError)
    state = column_state(col, water, t_s=1e308)
    before = state.water_m.tobytes()
    with pytest.raises(SubsurfaceError) as info:
        step(col, state, 1e308, added_water_m=add, potential_evaporation_m_s=pet)
    assert not isinstance(info.value, SubsurfaceStepError)
    assert state.water_m.tobytes() == before and state.t_s == 1e308
    result = step(col, state, 1.0, added_water_m=add, potential_evaporation_m_s=pet)
    assert np.isfinite(result.state.t_s) and result.state.t_s == 1e308 + 1.0
    with pytest.raises(SubsurfaceError):
        column_state(col, water, t_s=float("inf"))


def test_time_refusal_queries_no_memory_and_runs_no_kernel():
    calls = []
    col = prepare_column(layers(gravity=False), np.ones(1, dtype=bool), AREA, reserve_bytes=0,
                         available_bytes_fn=lambda: calls.append(1) or (1 << 40))
    state = column_state(col, from_cells([[0.0625, 0.125]]), t_s=1e308)
    n_before = len(calls)
    add, pet = forcing(col)
    with pytest.raises(SubsurfaceError):
        step(col, state, 1e308, added_water_m=add, potential_evaporation_m_s=pet)
    assert len(calls) == n_before


def test_forcing_arrays_require_exact_shape_and_dtype():
    col = column(layers(), n_cells=2)
    state = state_of(col, [[0.0625, 0.125]] * 2)
    add, pet = forcing(col)
    for bad_add, bad_pet in ((add.astype(np.float32), pet), (add[:, :1], pet), (add, pet.reshape(1, 2)),
                             (add, pet.astype(np.float32)), (np.ma.masked_array(add), pet), (add.tolist(), pet)):
        with pytest.raises(SubsurfaceError):
            step(col, state, 1.0, added_water_m=bad_add, potential_evaporation_m_s=bad_pet)


def test_closure_bound_is_the_maple_bound():
    col = column(layers(gravity=True), n_cells=4)
    state = state_of(col, [[0.0625, 0.25]] * 4)
    result = run(col, state, 1.0, add=[[0.01, 0.0]] * 4)
    b = result.budget
    assert b.n_terms == 4 * 2 * 4 + 7
    scale = max(abs(b.storage_before_m3), abs(b.storage_after_m3), b.added_m3, b.surface_return_m3, b.drainage_m3,
                b.evaporation_m3)
    assert b.bound_m3 == volume_roundoff_bound_m3(b.n_terms, scale)
    assert abs(b.residual_m3) <= b.bound_m3


def test_outputs_are_fresh_arrays_not_aliasing_inputs():
    col = column(layers(gravity=True), n_cells=2)
    state = state_of(col, [[0.0625, 0.25]] * 2)
    add, pet = forcing(col)
    result = step(col, state, 1.0, added_water_m=add, potential_evaporation_m_s=pet)
    for out in (result.state.water_m, result.surface_return_m, result.drainage_m, result.evaporation_m,
                result.evaporation_unmet_m, result.percolation_m, result.diffusion_m):
        assert out.flags.c_contiguous and out.dtype == np.float64
        assert not np.shares_memory(out, state.water_m) and not np.shares_memory(out, add)
    assert not np.shares_memory(result.surface_return_m, result.drainage_m)


def test_default_host_available_bytes_is_a_positive_int():
    try:
        value = default_host_available_bytes()
    except SubsurfaceBackendUnavailableError:
        pytest.skip("this platform cannot report available host memory")
    assert isinstance(value, int) and value > 0
