"""Working-array estimate and admission: backend-specific accounting of every owned vector, a MEASURED NumPy
allocation regression (tracemalloc peak of one step against the declared bound), admission BEFORE every bulk
allocation (prepare_column, column_state, uniform_state, step) with inputs untouched, and query/return contracts.
Nothing here was run by its author; Codex records results."""
from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pytest
from column_cases import AREA, forcing, from_cells, layers, prepare_column, state_of

from maple_syrup.subsurface import (
    KERNEL_SCRATCH_DOUBLES_PER_CELL,
    MEMORY_SCOPE,
    NUMPY_SCRATCH_DOUBLES_PER_CELL,
    SubsurfaceError,
    SubsurfaceMemoryError,
    admit_step,
    column_state,
    estimate_step_memory,
    kernel_parameter_doubles,
    layer_vector_doubles,
    soil_layers,
    step,
    uniform_state,
)


def budget_fn(initial):
    box = {"value": initial, "calls": 0}

    def query():
        box["calls"] += 1
        return box["value"]

    return query, box


def test_estimate_accounts_for_every_owned_vector_per_backend():
    L, N = 3, 1000
    e = estimate_step_memory(L, N, backend="numpy")
    assert layer_vector_doubles(L) == (L + 1) + 9 * L + 3 * (L - 1)
    assert kernel_parameter_doubles(L) == 5 * L + 2 * (L - 1)
    assert e.layer_bytes == layer_vector_doubles(L) * 8
    assert e.column_bytes == 2 * N + L * 8
    assert e.input_bytes == (2 * L * N + N) * 8
    assert e.output_bytes == (L + 2 * (L - 1) + 4 + 4) * N * 8
    assert e.scratch_bytes >= NUMPY_SCRATCH_DOUBLES_PER_CELL * N * 8 + 8 * N
    assert e.parameter_bytes == 0 and e.device_bytes == 0
    assert e.host_bytes == e.input_bytes + e.column_bytes + e.layer_bytes + e.output_bytes + e.scratch_bytes
    assert e.required_bytes == e.host_bytes == e.total_bytes
    assert e.step_allocation_bytes == e.output_bytes + e.scratch_bytes
    assert e.scope == MEMORY_SCOPE
    nb = estimate_step_memory(L, N, backend="numba")
    assert nb.parameter_bytes == kernel_parameter_doubles(L) * 8
    assert nb.scratch_bytes >= KERNEL_SCRATCH_DOUBLES_PER_CELL * N * 8 + 8 * N
    assert nb.host_bytes == (nb.input_bytes + nb.column_bytes + nb.layer_bytes + nb.parameter_bytes + nb.output_bytes
                             + nb.scratch_bytes)
    assert nb.required_bytes == nb.host_bytes and nb.device_bytes == 0
    cu = estimate_step_memory(L, N, backend="cuda")
    assert cu.parameter_bytes == 2 * kernel_parameter_doubles(L) * 8
    assert cu.device_bytes >= cu.input_bytes + cu.column_bytes + cu.output_bytes + kernel_parameter_doubles(L) * 8
    assert cu.host_bytes >= cu.layer_bytes + kernel_parameter_doubles(L) * 8
    assert cu.required_bytes == cu.device_bytes
    assert cu.total_bytes == cu.host_bytes + cu.device_bytes
    one = estimate_step_memory(1, N, backend="cuda")
    assert one.parameter_bytes == 2 * kernel_parameter_doubles(1) * 8 == 2 * 7 * 8
    # the compiled backends carry the O(L) per-interface factor scratch (max(L-1, 1) doubles); the reference does not
    assert nb.scratch_bytes - estimate_step_memory(1, N, backend="numba").scratch_bytes == (L - 2) * 8
    assert cu.scratch_bytes - one.scratch_bytes == (L - 2) * 8
    assert e.scratch_bytes == estimate_step_memory(1, N, backend="numpy").scratch_bytes
    assert nb.scratch_bytes >= KERNEL_SCRATCH_DOUBLES_PER_CELL * N * 8 + 8 * N + (L - 1) * 8


def test_estimate_is_monotone_and_includes_reserve():
    small = estimate_step_memory(2, 100, backend="numpy")
    more_cells = estimate_step_memory(2, 200, backend="numpy")
    more_layers = estimate_step_memory(4, 100, backend="numpy")
    reserved = estimate_step_memory(2, 100, backend="numpy", reserve_bytes=12345)
    assert more_cells.required_bytes > small.required_bytes and more_layers.required_bytes > small.required_bytes
    assert reserved.required_bytes == small.required_bytes + 12345 and reserved.reserve_bytes == 12345
    assert reserved.total_bytes == small.total_bytes + 12345
    cuda_reserved = estimate_step_memory(2, 100, backend="cuda", reserve_bytes=7)
    assert cuda_reserved.required_bytes == cuda_reserved.device_bytes + 7


@pytest.mark.parametrize("args", [(0, 10), (2, 0), (-1, 10), (True, 10), (2, 2.5)])
def test_estimate_rejects_invalid_sizes(args):
    with pytest.raises(SubsurfaceError):
        estimate_step_memory(*args, backend="numpy")


def test_estimate_rejects_invalid_backend_and_reserve():
    with pytest.raises(SubsurfaceError):
        estimate_step_memory(2, 10, backend="cpu")
    with pytest.raises(SubsurfaceError):
        estimate_step_memory(2, 10, backend="numpy", reserve_bytes=-5)
    with pytest.raises(SubsurfaceError):
        estimate_step_memory(2, 10, backend="numpy", reserve_bytes=True)


@pytest.mark.parametrize("gravity", [True, False])
def test_reference_step_allocations_stay_within_the_declared_bound(gravity):
    """Measured regression: the tracemalloc peak of NEW allocations during one reference step must not exceed the
    declared per-step allocation bound (outputs + scratch), and the whole estimate must cover that peak plus the
    arrays that already exist (inputs, column, layers)."""
    L, N = 3, 100_000
    lay = soil_layers([0.0, 0.1, 0.3, 1.0], theta_residual=0.02, theta_field_capacity=0.2, theta_saturation=0.4,
                      diffusivity_m2_s=1e-7, theta_evaporation_stress=0.15, evaporation_factor=1.2,
                      gravity_drainage=gravity)
    col = prepare_column(lay, np.ones(N, dtype=bool), 1.0, backend="numpy", reserve_bytes=0,
                         available_bytes_fn=lambda: 1 << 40)
    state = uniform_state(col, [0.25, 0.1, 0.15])
    added = np.zeros((L, N))
    added[0] = 0.03  # drives saturation return and the cascade in every cell
    pet = np.full(N, 1e-7)
    gc.collect()
    tracemalloc.start()
    result = step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    est = col.memory
    assert est.output_bytes <= peak, (est.output_bytes, peak)  # the outputs are genuinely allocated
    assert peak <= est.step_allocation_bytes, (peak, est.step_allocation_bytes)
    assert est.total_bytes >= peak + est.input_bytes + est.column_bytes + est.layer_bytes
    assert result.budget.surface_return_m3 > 0.0 and result.budget.evaporation_m3 > 0.0
    del result


def test_prepare_column_admits_before_building():
    query, box = budget_fn(0)
    mask = np.ones(5, dtype=bool)
    mask_bytes = mask.tobytes()
    with pytest.raises(SubsurfaceMemoryError):
        prepare_column(layers(), mask, AREA, reserve_bytes=0, available_bytes_fn=query)
    assert box["calls"] == 1 and mask.tobytes() == mask_bytes
    box["value"] = 1 << 40
    col = prepare_column(layers(), mask, AREA, reserve_bytes=0, available_bytes_fn=query)
    assert box["calls"] == 2 and col.n_cells == 5


def test_state_constructors_and_step_admit_before_allocating():
    query, box = budget_fn(1 << 40)
    col = prepare_column(layers(), np.ones(3, dtype=bool), AREA, reserve_bytes=0, available_bytes_fn=query)
    water = from_cells([[0.0625, 0.125]] * 3)
    water_bytes = water.tobytes()
    assert box["calls"] == 1
    box["value"] = 0
    with pytest.raises(SubsurfaceMemoryError):
        column_state(col, water)
    assert box["calls"] == 2
    with pytest.raises(SubsurfaceMemoryError):
        uniform_state(col, [0.25, 0.25])
    assert box["calls"] == 3
    assert water.tobytes() == water_bytes
    # the state copy is one extra state-sized allocation: admitted against required + L N 8, the others against required
    box["value"] = col.memory.required_bytes
    with pytest.raises(SubsurfaceMemoryError):
        column_state(col, water)
    state = uniform_state(col, [0.25, 0.25])
    add, pet = forcing(col)
    result = step(col, state, 1.0, added_water_m=add, potential_evaporation_m_s=pet)
    assert result.state.t_s == 1.0
    box["value"] = col.memory.required_bytes + 2 * 3 * 8
    state2 = column_state(col, water)
    assert state2.water_m.tobytes() == water_bytes and not np.shares_memory(state2.water_m, water)


def test_step_refusal_happens_before_allocation_and_leaves_inputs_unchanged():
    query, box = budget_fn(1 << 40)
    col = prepare_column(layers(), np.ones(3, dtype=bool), AREA, reserve_bytes=0, available_bytes_fn=query)
    state = state_of(col, [[0.0625, 0.125]] * 3)
    add, pet = forcing(col)
    before = (state.water_m.tobytes(), add.tobytes(), pet.tobytes())
    box["value"] = 10
    with pytest.raises(SubsurfaceMemoryError):
        step(col, state, 1.0, added_water_m=add, potential_evaporation_m_s=pet)
    assert (state.water_m.tobytes(), add.tobytes(), pet.tobytes()) == before
    with pytest.raises(SubsurfaceMemoryError):
        admit_step(col)


def test_reserve_is_added_to_the_requirement():
    need = estimate_step_memory(2, 3, backend="numpy").required_bytes
    ok = prepare_column(layers(), np.ones(3, dtype=bool), AREA, reserve_bytes=0, available_bytes_fn=lambda: need)
    assert admit_step(ok) == need
    with pytest.raises(SubsurfaceMemoryError):
        prepare_column(layers(), np.ones(3, dtype=bool), AREA, reserve_bytes=1, available_bytes_fn=lambda: need)


def test_query_is_called_once_per_entry_point_and_must_return_an_int():
    query, box = budget_fn(1 << 40)
    col = prepare_column(layers(), np.ones(2, dtype=bool), AREA, reserve_bytes=0, available_bytes_fn=query)
    state = state_of(col, [[0.0625, 0.125]] * 2)
    add, pet = forcing(col)
    step(col, state, 1.0, added_water_m=add, potential_evaporation_m_s=pet)
    step(col, state, 1.0, added_water_m=add, potential_evaporation_m_s=pet)
    assert box["calls"] == 4  # prepare, state copy, two steps
    for bad in (1.5, True, -1, None):
        bad_col = prepare_column(layers(), np.ones(2, dtype=bool), AREA, reserve_bytes=0,
                                 available_bytes_fn=lambda: 1 << 40)
        object.__setattr__(bad_col, "available_bytes_fn", lambda bad=bad: bad)
        with pytest.raises(SubsurfaceError):
            admit_step(bad_col)
        with pytest.raises(SubsurfaceError):
            prepare_column(layers(), np.ones(2, dtype=bool), AREA, reserve_bytes=0, available_bytes_fn=lambda bad=bad: bad)


def test_column_memory_matches_the_estimate():
    col = prepare_column(layers(), np.ones(7, dtype=bool), AREA, reserve_bytes=99, available_bytes_fn=lambda: 1 << 40)
    assert col.memory == estimate_step_memory(2, 7, backend="numpy", reserve_bytes=99)
    assert col.summary()["memory"]["required_bytes"] == col.memory.required_bytes
