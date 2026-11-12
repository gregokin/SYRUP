"""Compiled Numba backend against the NumPy reference (skipped without Numba; a skip makes no claim). Declared bound
rtol 2e-12 / atol 1e-14 (expm1 may differ by an ulp); bitwise equality where every beta is 0 or 1."""
from __future__ import annotations

import gc
import math
import tracemalloc

import numpy as np
import pytest
from pref_cases import AREA, BIG, arrays_of, build_random, column, operator, random_case, state_of

from maple_syrup import preferential_infiltration as pi
from maple_syrup import subsurface as ss

numba = pytest.importorskip("numba")

RTOL, ATOL = 2e-12, 1e-14
DT_SEQUENCE = (0.0, 600.0, 0.5, 86400.0, 1e6, 0.0, 3600.0)


def both(seed, n_layers, n_cells, **kwargs):
    lay, mask, a3, max_fraction, max_rate, background, weights, covers, water, surface, dt = random_case(
        seed, n_layers, n_cells, **kwargs)
    cols, ops = [], []
    for backend in ("numpy", "numba"):
        col = ss.prepare_column(lay, mask, AREA, backend=backend, reserve_bytes=0, available_bytes_fn=lambda: BIG)
        ops.append(operator(col, a3=a3, max_fraction=max_fraction, max_rate=max_rate, background=background,
                            species=weights, contributions=covers))
        cols.append(col)
    assert ops[1].backend == "numba" and ops[1].backend_info["versions"] is not None
    return cols, ops, water, surface, dt


def test_numba_build_maps_are_bitwise_identical_to_the_reference_build():
    _cols, ops, *_ = both(11, 4, 97)
    for name in pi.KernelMaps._fields:
        assert getattr(ops[1].maps, name).tobytes() == getattr(ops[0].maps, name).tobytes(), name
    assert ops[1].beta.tobytes() == ops[0].beta.tobytes()
    assert ops[1].sha256 == ops[0].sha256
    assert ops[1].verify_identity()


@pytest.mark.parametrize("n_layers", [1, 2, 5])
def test_numba_matches_reference_within_declared_bound(n_layers):
    cols, ops, water, surface, dt = both(7 + n_layers, n_layers, 61)
    results = [pi.preferential_step(op, surface, ss.column_state(col, water), dt) for col, op in zip(cols, ops, strict=True)]
    r, j = results
    for name, a in arrays_of(r).items():
        np.testing.assert_allclose(arrays_of(j)[name], a, rtol=RTOL, atol=ATOL, err_msg=name)
    assert j.state.t_s == r.state.t_s and j.dt_s == r.dt_s
    for field in pi.PreferentialBudget.__dataclass_fields__:
        if field in ("n_terms",):
            assert getattr(j.budget, field) == getattr(r.budget, field)
        else:
            assert getattr(j.budget, field) == pytest.approx(getattr(r.budget, field), rel=RTOL, abs=ATOL), field
    assert abs(j.budget.soil_residual_m3) <= j.budget.bound_m3


def test_numba_is_bitwise_equal_where_beta_is_zero_or_one():
    col_r, col_j = column(n_cells=64), column(n_cells=64, backend="numba")
    rng = np.random.default_rng(3)
    a3 = rng.choice(np.array([0.0, 1.0]), 64)
    cover = rng.uniform(0.0, 3.0, 64) * (a3 == 1.0)  # cover only where a3 = 1 (beta stays exactly 1)
    water = np.ascontiguousarray(rng.uniform(0.0, 1.0, (2, 64)) * np.array([0.125, 0.375])[:, None])
    surface = rng.uniform(0.0, 0.05, 64)
    max_rate = rng.uniform(1e-7, 1e-5, 64)
    results = []
    for col in (col_r, col_j):
        op = operator(col, a3=a3, max_fraction=1.0, max_rate=max_rate, background=(0.0, 1.0), species=[(1.0, 1.0)],
                      contributions=[cover])
        assert set(np.unique(op.beta)) <= {0.0, 1.0}
        results.append(pi.preferential_step(op, surface, ss.column_state(col, water), 777.0))
    r, j = results
    for name, a in arrays_of(r).items():
        assert arrays_of(j)[name].tobytes() == a.tobytes(), name
    assert j.budget == r.budget


def test_numba_failure_precedence_matches_reference():
    cols, ops, water, surface, dt = both(3, 2, 9)
    water = water.copy()
    water[0, 3] = np.nan
    surface = surface.copy()
    surface[0] = -1.0
    errors = []
    for col, op in zip(cols, ops, strict=True):
        with pytest.raises(pi.PreferentialStepError) as info:
            pi.preferential_step(op, surface, ss.column_state(col, water), dt)
        errors.append((info.value.bit, info.value.n_flagged))
    assert errors[0] == errors[1] == (0, 2)


def test_numba_step_allocations_stay_within_the_declared_bound():
    col, op, water, surface, dt = build_random(31, 3, 100_000, "numba")
    state = ss.column_state(col, water)
    gc.collect()
    tracemalloc.start()
    result = pi.preferential_step(op, surface, state, dt)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert op.memory.output_bytes <= peak <= op.memory.step_allocation_bytes, (peak, op.memory.step_allocation_bytes)
    del result


@pytest.mark.parametrize("n_layers", [1, 2, 40])
def test_numba_varying_dt_matches_reference_on_one_operator_and_on_fresh_operators(n_layers):
    cols, ops, water, surface, _ = both(90 + n_layers, n_layers, 53)
    (col_r, col_j), (op_r, op_j) = cols, ops
    state_r, state_j = ss.column_state(col_r, water), ss.column_state(col_j, water)
    surface_r = surface_j = surface
    for dt in DT_SEQUENCE:
        r = pi.preferential_step(op_r, surface_r, state_r, dt)
        j = pi.preferential_step(op_j, surface_j, state_j, dt)
        for name, a in arrays_of(r).items():
            np.testing.assert_allclose(arrays_of(j)[name], a, rtol=RTOL, atol=ATOL, err_msg=f"dt={dt} {name}")
        lay, mask, a3, max_fraction, max_rate, background, weights, covers, *_ = random_case(90 + n_layers, n_layers, 53)
        fresh_col = ss.prepare_column(lay, mask, AREA, backend="numba", reserve_bytes=0, available_bytes_fn=lambda: BIG)
        fresh = operator(fresh_col, a3=a3, max_fraction=max_fraction, max_rate=max_rate, background=background,
                         species=weights, contributions=covers)
        f = pi.preferential_step(fresh, surface_j, ss.column_state(fresh_col, state_j.water_m), dt)
        for name, a in arrays_of(j).items():
            assert arrays_of(f)[name].tobytes() == a.tobytes(), f"stale state? dt={dt} {name}"
        state_r, state_j, surface_r, surface_j = r.state, j.state, r.surface_water_m, j.surface_water_m
    assert state_j.t_s == state_r.t_s == 0.0  # time is preserved by every stage


def test_numba_dt_zero_identity_and_inputs_untouched():
    col, op, water, surface, dt = build_random(5, 3, 12, "numba")
    state = ss.column_state(col, water)
    before = (state.water_m.tobytes(), surface.tobytes())
    result = pi.preferential_step(op, surface, state, 0.0)
    assert result.state.water_m.tobytes() == state.water_m.tobytes() and result.surface_water_m.tobytes() == surface.tobytes()
    result = pi.preferential_step(op, surface, state, dt)
    assert (state.water_m.tobytes(), surface.tobytes()) == before
    assert not np.shares_memory(result.state.water_m, state.water_m)
    assert not np.shares_memory(result.surface_water_m, surface)


def test_numba_endpoints_all_or_nothing():
    col = column(n_cells=3, backend="numba")
    state = state_of(col, [[0.03125, 0.09375]] * 3)
    H = np.array([0.01, 0.02, 0.0])
    r = pi.preferential_step(operator(col, a3=1.0, max_fraction=1.0), H, state, 1e-6)
    np.testing.assert_array_equal(r.surface_water_m, 0.0)
    r = pi.preferential_step(operator(col, a3=1.0, max_fraction=0.0), H, state, 1e9)
    assert r.surface_water_m.tobytes() == H.tobytes()
    assert math.isinf(operator(col, a3=1.0, max_fraction=1.0).maps.rate_s[0])


def test_numba_unavailable_is_a_setup_error_not_a_fallback(monkeypatch):
    from maple_syrup import preferential_infiltration_numba as pn

    monkeypatch.setattr(pn, "numba_available", lambda: False)
    col = column(n_cells=2, backend="numba")
    with pytest.raises(pi.PreferentialBackendUnavailableError):
        operator(col)
