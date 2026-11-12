"""Compiled Numba backend against the NumPy reference (skipped without Numba; a skip makes no claim). Declared bound
rtol 2e-12 / atol 1e-14 (log1p/log may differ by an ulp); bitwise equality where no logarithm is evaluated."""
from __future__ import annotations

import numpy as np
import pytest
from matrix_cases import AREA, BIG, arrays_of, column, full, history_of, layers, operator, state_of

from maple_syrup import matrix_infiltration as mi
from maple_syrup import subsurface as ss

numba = pytest.importorskip("numba")

RTOL, ATOL = 2e-12, 1e-14
DT_SEQUENCE = (0.0, 600.0, 0.5, 86400.0, 1e6, 0.0, 3600.0)


def random_case(seed, n_layers, n_cells, *, no_log=False):
    rng = np.random.default_rng(seed)
    lay = layers(deeper=tuple(rng.uniform(0.1, 1.0, n_layers - 1)) if n_layers > 1 else ())
    mask = rng.random(n_cells) < 0.75
    mask[0] = True
    k = rng.uniform(0.0, 3e-5, n_cells)
    psi = rng.uniform(0.0, 0.3, n_cells)
    g = rng.choice([0.0, 0.3, 0.5, 1.0], n_cells)
    thick = np.array(lay.thickness_m)
    water = np.ascontiguousarray(rng.uniform(0.0, 1.0, (n_layers, n_cells)) * (np.array(lay.theta_saturation) * thick)[:, None])
    if no_log:
        water[0] = lay.field_capacity_m[0] + rng.uniform(0.0, 1.0, n_cells) * (lay.saturation_m[0] - lay.field_capacity_m[0])  # b = 0
    water[0, : max(n_cells // 10, 1)] = lay.saturation_m[0]
    rate = rng.uniform(0.0, 6e-5, n_cells) * mask
    standing = rng.uniform(0.0, 0.02, n_cells) * (rng.random(n_cells) < 0.5)
    dt = 450.0
    pool = np.where(standing > 0.0, standing + rate * dt, rate * dt) * (rng.random(n_cells) < 0.95)
    wet = rng.uniform(0.0, 0.02, n_cells) * (rng.random(n_cells) < 0.7)
    cum = rng.uniform(0.0, 0.05, n_cells)
    return lay, mask, k, psi, g, water, pool, standing, rate, wet, cum, dt


def both(seed, n_layers, n_cells, **kwargs):
    lay, mask, k, psi, g, water, pool, standing, rate, wet, cum, dt = random_case(seed, n_layers, n_cells, **kwargs)
    cols, ops, hists = [], [], []
    for backend in ("numpy", "numba"):
        col = ss.prepare_column(lay, mask, AREA, backend=backend, reserve_bytes=0, available_bytes_fn=lambda: BIG)
        op = operator(col, k=k, psi=psi, g=g)
        cols.append(col)
        ops.append(op)
        hists.append(history_of(op, wet, cum))
    assert ops[1].backend == "numba" and ops[1].backend_info["versions"] is not None
    return cols, ops, hists, water, pool, standing, rate, dt


def test_numba_build_maps_are_bitwise_identical_to_the_reference_build():
    _cols, ops, *_ = both(11, 4, 97)
    for name in mi.KernelMaps._fields:
        assert getattr(ops[1].maps, name).tobytes() == getattr(ops[0].maps, name).tobytes(), name
    assert ops[1].sha256 == ops[0].sha256 and ops[1].verify_identity()


@pytest.mark.parametrize("n_layers", [1, 2, 5])
def test_numba_matches_reference_within_declared_bound(n_layers):
    cols, ops, hists, water, pool, standing, rate, dt = both(7 + n_layers, n_layers, 301)
    results = [mi.matrix_step(op, pool, standing, rate, ss.column_state(col, water), h, dt)
               for col, op, h in zip(cols, ops, hists, strict=True)]
    r, j = results
    for name, a in arrays_of(r).items():
        np.testing.assert_allclose(arrays_of(j)[name], a, rtol=RTOL, atol=ATOL, err_msg=name)
    assert j.state.t_s == r.state.t_s and j.dt_s == r.dt_s
    for field in mi.MatrixBudget.__dataclass_fields__:
        if field in ("n_terms", "n_onsets", "n_ponded"):
            assert getattr(j.budget, field) == getattr(r.budget, field), field
        elif field == "max_newton_iterations":
            assert abs(getattr(j.budget, field) - getattr(r.budget, field)) <= 1
        else:
            assert getattr(j.budget, field) == pytest.approx(getattr(r.budget, field), rel=RTOL, abs=ATOL), field
    assert r.budget.n_ponded > 0 and r.budget.max_newton_iterations > 0


def test_numba_is_bitwise_equal_where_no_logarithm_is_evaluated():
    cols, ops, hists, water, pool, standing, rate, dt = both(3, 2, 128, no_log=True)
    results = [mi.matrix_step(op, pool, standing, rate, ss.column_state(col, water), h, dt)
               for col, op, h in zip(cols, ops, hists, strict=True)]
    r, j = results
    assert r.budget.max_newton_iterations == 0
    for name, a in arrays_of(r).items():
        assert arrays_of(j)[name].tobytes() == a.tobytes(), name
    assert j.budget == r.budget


def test_numba_failure_precedence_matches_reference():
    cols, ops, hists, water, pool, standing, rate, dt = both(5, 3, 40)
    bad_water = np.array(water)
    bad_water[1, 3] = -1.0
    bad_water[0, 5] = np.nan
    bad_rate = np.array(rate)
    bad_rate[7] = -1.0
    for col, op, h in zip(cols, ops, hists, strict=True):
        with pytest.raises(mi.MatrixStepError) as info:
            mi.matrix_step(op, pool, standing, bad_rate, ss.column_state(col, bad_water), h, dt)
        assert info.value.bit == 0 and info.value.n_flagged == 3
        bad_pool = np.array(pool)
        bad_pool[2] = pool[2] + 1.0  # no-contact pool above the supply (if cell 2 has no contact), else NaN history
        wet = np.array(h.wetting_m)
        wet[9] = -5.0
        with pytest.raises(mi.MatrixStepError) as info:
            mi.matrix_step(op, pool, standing, rate, ss.column_state(col, water), mi.wetting_history(op, wetting_m=wet, cumulative_intake_m=np.array(h.cumulative_intake_m)), dt)
        assert info.value.bit == 11 and info.value.n_flagged == 1


@pytest.mark.parametrize("n_layers", [1, 3])
def test_numba_varying_dt_on_one_operator_and_history_carry(n_layers):
    cols, ops, hists, water, pool, standing, rate, _ = both(20 + n_layers, n_layers, 200)
    states = [ss.column_state(col, water) for col in cols]
    pools, standings = [pool, pool], [standing, standing]
    for dt in DT_SEQUENCE:
        out = []
        for i in (0, 1):
            consistent_pool = np.where(standings[i] > 0.0, pools[i], np.minimum(pools[i], rate * dt))
            res = mi.matrix_step(ops[i], consistent_pool, standings[i], rate, states[i], hists[i], dt)
            states[i], hists[i], pools[i], standings[i] = res.state, res.history, res.surface_pool_m, res.surface_pool_m
            out.append(res)
        for name, a in arrays_of(out[0]).items():
            np.testing.assert_allclose(arrays_of(out[1])[name], a, rtol=RTOL, atol=ATOL, err_msg=f"dt={dt} {name}")
    assert states[1].t_s == states[0].t_s == 0.0


def test_numba_unavailable_is_a_setup_error_not_a_fallback(monkeypatch):
    from maple_syrup import matrix_infiltration_numba as mn

    monkeypatch.setattr(mn, "numba_available", lambda: False)
    col = column(n_cells=2)
    object.__setattr__(col, "backend", "numba")
    with pytest.raises(mi.MatrixBackendUnavailableError):
        operator(col)
    with pytest.raises(mi.MatrixBackendUnavailableError):
        mn.require_numba()


def test_numba_endpoints_zero_supply_dt_zero_and_measured_allocation():
    col = column(n_cells=64, backend="numba")
    op = operator(col)
    state = state_of(col)
    hist = history_of(op, 0.003, 0.004)
    zero = full(col, 0.0)
    res = mi.matrix_step(op, zero, zero, zero, state, hist, 600.0)
    assert res.history.wetting_m.tobytes() == hist.wetting_m.tobytes() and res.budget.intake_m3 == 0.0
    res0 = mi.matrix_step(op, full(col, 0.01), full(col, 0.01), zero, state, hist, 0.0)
    assert res0.state.water_m.tobytes() == state.water_m.tobytes() and res0.budget.candidate_m3 == 0.0
    ref = operator(column(n_cells=64))
    a = mi.matrix_step(ref, full(col, 0.01), full(col, 0.01), zero, state_of(ref.column), history_of(ref, 0.003, 0.004), 600.0)
    b = mi.matrix_step(op, full(col, 0.01), full(col, 0.01), zero, state, hist, 600.0)
    np.testing.assert_allclose(b.intake_m, a.intake_m, rtol=RTOL, atol=ATOL)
