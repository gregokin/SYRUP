"""Compiled Numba backend against the NumPy reference (skipped without Numba; a skip makes no claim). Declared bound
rtol 2e-12 / atol 1e-14 (expm1 may differ by an ulp); bitwise equality where the diffusivity is 0. Nothing here was
run by its author; Codex records results."""
from __future__ import annotations

import numpy as np
import pytest
from column_cases import AREA, random_case

from maple_syrup.subsurface import (
    SubsurfaceStepError,
    column_state,
    prepare_column,
    step,
)

numba = pytest.importorskip("numba")

RTOL, ATOL = 2e-12, 1e-14


def both_columns(lay, mask):
    ref = prepare_column(lay, mask, AREA, backend="numpy", reserve_bytes=0, available_bytes_fn=lambda: 1 << 40)
    jit = prepare_column(lay, mask, AREA, backend="numba", reserve_bytes=0, available_bytes_fn=lambda: 1 << 40)
    assert jit.backend == "numba" and jit.backend_info["versions"] is not None
    return ref, jit


def run_both(ref, jit, water, add, pet, dt):
    r = step(ref, column_state(ref, water), dt, added_water_m=add, potential_evaporation_m_s=pet)
    j = step(jit, column_state(jit, water), dt, added_water_m=add, potential_evaporation_m_s=pet)
    return r, j


def arrays_of(result):
    return {"water": result.state.water_m, "return": result.surface_return_m, "drainage": result.drainage_m,
            "evaporation": result.evaporation_m, "unmet": result.evaporation_unmet_m,
            "percolation": result.percolation_m, "diffusion": result.diffusion_m}


@pytest.mark.parametrize("gravity", [True, False])
@pytest.mark.parametrize("n_layers", [1, 2, 5])
def test_numba_matches_reference_within_declared_bound(n_layers, gravity):
    lay, mask, water, add, pet, dt = random_case(7 + n_layers, n_layers, 37, gravity=gravity, diffusive=True)
    ref, jit = both_columns(lay, mask)
    r, j = run_both(ref, jit, water, add, pet, dt)
    for name, a in arrays_of(r).items():
        np.testing.assert_allclose(arrays_of(j)[name], a, rtol=RTOL, atol=ATOL, err_msg=name)
    assert j.state.t_s == r.state.t_s
    for field in ("storage_before_m3", "storage_after_m3", "added_m3", "surface_return_m3", "drainage_m3",
                  "evaporation_m3", "evaporation_unmet_m3"):
        assert getattr(j.budget, field) == pytest.approx(getattr(r.budget, field), rel=RTOL, abs=ATOL)
    assert abs(j.budget.residual_m3) <= j.budget.bound_m3


@pytest.mark.parametrize("n_layers", [1, 3])
def test_numba_is_bitwise_equal_without_diffusion(n_layers):
    lay, mask, water, add, pet, dt = random_case(21 + n_layers, n_layers, 64, gravity=True, diffusive=False)
    ref, jit = both_columns(lay, mask)
    r, j = run_both(ref, jit, water, add, pet, dt)
    for name, a in arrays_of(r).items():
        assert arrays_of(j)[name].tobytes() == a.tobytes(), name
    assert j.budget == r.budget


def test_numba_failure_precedence_matches_reference():
    lay, mask, water, add, pet, dt = random_case(3, 2, 9, gravity=True, diffusive=True)
    water = water.copy()
    water[0, 3] = np.nan
    add = add.copy()
    add[1, 0] = -1.0
    ref, jit = both_columns(lay, mask)
    errors = []
    for col in (ref, jit):
        with pytest.raises(SubsurfaceStepError) as info:
            step(col, column_state(col, water), dt, added_water_m=add, potential_evaporation_m_s=pet)
        errors.append((info.value.bit, info.value.n_flagged))
    assert errors[0] == errors[1] == (0, 2)


def test_numba_step_allocations_stay_within_the_declared_bound():
    """Measured regression: the kernel allocates only the outputs; the packet stage adds one boolean mask."""
    import gc
    import tracemalloc

    from maple_syrup.subsurface import uniform_state

    lay, mask, _, add, pet, dt = random_case(31, 3, 100_000, gravity=True, diffusive=True)
    jit = prepare_column(lay, mask, AREA, backend="numba", reserve_bytes=0, available_bytes_fn=lambda: 1 << 40)
    state = uniform_state(jit, [0.1, 0.1, 0.1])
    gc.collect()
    tracemalloc.start()
    result = step(jit, state, dt, added_water_m=add, potential_evaporation_m_s=pet)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert jit.memory.output_bytes <= peak <= jit.memory.step_allocation_bytes, (peak, jit.memory.step_allocation_bytes)
    del result


DT_SEQUENCE = (0.0, 600.0, 0.5, 86400.0, 1e6, 0.0, 3600.0)


@pytest.mark.parametrize("n_layers", [1, 2, 40])
def test_numba_varying_dt_matches_reference_on_one_column_and_on_fresh_columns(n_layers):
    """The per-interface factor is evaluated per call: successive steps with different dt on ONE column must agree
    with the reference and be bit-identical to a FRESH column stepped once with the same dt from the same state."""
    lay, mask, water, add, pet, _ = random_case(90 + n_layers, n_layers, 53, gravity=True, diffusive=True)
    ref, jit = both_columns(lay, mask)
    state_ref, state_jit = column_state(ref, water), column_state(jit, water)
    zero = np.zeros_like(add)
    for dt in DT_SEQUENCE:
        forcing = zero if dt == 0.0 else add
        r = step(ref, state_ref, dt, added_water_m=forcing, potential_evaporation_m_s=pet)
        j = step(jit, state_jit, dt, added_water_m=forcing, potential_evaporation_m_s=pet)
        for name, a in arrays_of(r).items():
            np.testing.assert_allclose(arrays_of(j)[name], a, rtol=RTOL, atol=ATOL, err_msg=f"dt={dt} {name}")
        fresh = prepare_column(lay, mask, AREA, backend="numba", reserve_bytes=0, available_bytes_fn=lambda: 1 << 40)
        f = step(fresh, column_state(fresh, state_jit.water_m), dt, added_water_m=forcing, potential_evaporation_m_s=pet)
        for name, a in arrays_of(j).items():
            assert arrays_of(f)[name].tobytes() == a.tobytes(), f"stale factor? dt={dt} {name}"
        state_ref, state_jit = r.state, j.state
    assert state_jit.t_s == state_ref.t_s == sum(DT_SEQUENCE)


def test_numba_streaming_matches_reference_streaming():
    from maple_syrup.subsurface import ForcingRecord, initialize_profile

    lay, mask, water, add, pet, _ = random_case(77, 4, 31, gravity=True, diffusive=True)
    ref, jit = both_columns(lay, mask)
    records = [ForcingRecord(dt, add if dt else np.zeros_like(add), pet) for dt in DT_SEQUENCE]
    pr = initialize_profile(ref, column_state(ref, water), records)
    pj = initialize_profile(jit, column_state(jit, water), records)
    np.testing.assert_allclose(pj.state.water_m, pr.state.water_m, rtol=RTOL, atol=ATOL)
    assert pj.cumulative.n_steps == pr.cumulative.n_steps == len(DT_SEQUENCE)
    assert pj.cumulative.elapsed_s == pr.cumulative.elapsed_s
    for name in ("added_m3", "surface_return_m3", "drainage_m3", "evaporation_m3", "evaporation_unmet_m3"):
        assert getattr(pj.cumulative, name) == pytest.approx(getattr(pr.cumulative, name), rel=RTOL, abs=ATOL)
    assert abs(pj.residual_m3) <= pj.bound_m3


def test_numba_dt_zero_identity_and_inputs_untouched():
    lay, mask, water, add, pet, dt = random_case(5, 3, 12, gravity=True, diffusive=True)
    _, jit = both_columns(lay, mask)
    state = column_state(jit, water)
    zero = np.zeros_like(add)
    before = (state.water_m.tobytes(), zero.tobytes(), pet.tobytes())
    result = step(jit, state, 0.0, added_water_m=zero, potential_evaporation_m_s=pet)
    assert result.state.water_m.tobytes() == state.water_m.tobytes()
    assert (state.water_m.tobytes(), zero.tobytes(), pet.tobytes()) == before
    result = step(jit, state, dt, added_water_m=add, potential_evaporation_m_s=pet)
    assert state.water_m.tobytes() == before[0] and pet.tobytes() == before[2]
    assert not np.shares_memory(result.state.water_m, state.water_m)
