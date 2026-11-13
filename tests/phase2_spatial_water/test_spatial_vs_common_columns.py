"""Heterogeneous spatial step compared COLUMN BY COLUMN, bit for bit, with the accepted common solver run on that column alone under its own explicit parameters (different widths, residual/FC/sat,
interfaces, masks), for 1/2/3/5 layers, both diffusion modes, open/closed bottom, gravity on/off, dry/saturated/above-FC/random/mixed water, ordinary intake vs uniform recharge, shares, dt boundaries.
Author-written; the oracle is the real common solver, not a mirror of the spatial code."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import DTS, assert_columns_equal, make_spec, params, prepare, water_kind

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss


def body(col, water, added, pet, dt, share, recharge):
    return ss._reference_step_body(sp._SpatialStepParams(col), col._mask, water, added, pet, dt, share, recharge)


@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("kind", ["dry", "saturated", "above_fc", "random", "mixed"])
def test_every_stepaarrays_field_equals_the_common_solver_per_column(mode, bottom, L, kind):
    N = 21
    rng = np.random.default_rng(31 * L + len(kind))
    p = params(rng, L, N, mode=mode)
    mask = rng.uniform(size=N) > 0.25
    col = prepare(make_spec(p, mode=mode, bottom=bottom), mask=mask)
    water = water_kind(rng, p, kind)
    added = rng.uniform(0.0, 2e-3, size=(L, N)) * (rng.uniform(size=(L, N)) > 0.3)
    pet = rng.uniform(0.0, 3e-7, size=N)
    added[:, ~mask] = 0.0
    pet[~mask] = 0.0
    share = rng.uniform(0.0, 1.0, size=N)
    for dt in (0.0, 3600.0, 86400.0):
        for sh in (None, share):
            assert_columns_equal(body(col, water, added, pet, dt, sh, None), p, water, added, pet, dt, sh, None, mask, mode=mode, bottom=bottom, gravity=True)
        if dt > 0:
            for pulse in (0.0, 5e-4, 4e-3):
                assert_columns_equal(body(col, water, None, pet, dt, sh, pulse), p, water, added, pet, dt, sh, pulse, mask, mode=mode, bottom=bottom, gravity=True)


@pytest.mark.parametrize("dt", DTS)
def test_dt_boundaries_including_zero_time_and_extremes(mode, bottom, dt):
    L, N = 3, 17
    rng = np.random.default_rng(3)
    p = params(rng, L, N, mode=mode)
    col = prepare(make_spec(p, mode=mode, bottom=bottom))
    water = water_kind(rng, p, "above_fc")
    added, pet = rng.uniform(0.0, 1e-3, size=(L, N)), rng.uniform(0.0, 1e-6, size=N)
    assert_columns_equal(body(col, water, added, pet, dt, None, None), p, water, added, pet, dt, None, None, np.ones(N, bool), mode=mode, bottom=bottom, gravity=True)


def test_gravity_switch_off_and_non_default_stress_and_factor(mode, bottom):
    L, N = 4, 15
    rng = np.random.default_rng(5)
    p = params(rng, L, N, mode=mode)
    col = prepare(make_spec(p, mode=mode, bottom=bottom, gravity=False, stress=0.62, factor=1.2))
    water = water_kind(rng, p, "above_fc")
    added, pet = rng.uniform(0.0, 1e-3, size=(L, N)), rng.uniform(0.0, 5e-7, size=N)
    assert_columns_equal(body(col, water, added, pet, 7200.0, None, None), p, water, added, pet, 7200.0, None, None, np.ones(N, bool), mode=mode, bottom=bottom, gravity=False, stress=0.62, factor=1.2)


def test_flagged_invalid_states_report_the_same_bits_per_column(mode, bottom):
    L, N = 3, 24
    rng = np.random.default_rng(9)
    p = params(rng, L, N, mode=mode)
    col = prepare(make_spec(p, mode=mode, bottom=bottom))
    water = water_kind(rng, p, "random")
    water[1, 3], water[0, 5], water[2, 7] = np.nan, -1e-3, 5.0  # non-finite, negative, above saturation
    added, pet = rng.uniform(0, 1e-3, size=(L, N)), rng.uniform(0, 1e-7, size=N)
    added[0, 11] = -1.0
    pet[13] = np.nan
    assert_columns_equal(body(col, water, added, pet, 3600.0, None, None), p, water, added, pet, 3600.0, None, None, np.ones(N, bool), mode=mode, bottom=bottom, gravity=True)


def test_public_step_matches_per_column_common_step_budgets_and_clock(mode, bottom):
    L, N = 3, 16
    rng = np.random.default_rng(11)
    p = params(rng, L, N, mode=mode)
    col = prepare(make_spec(p, mode=mode, bottom=bottom))
    water = water_kind(rng, p, "random")
    added, pet = rng.uniform(0, 1e-3, size=(L, N)) * 0.3, rng.uniform(0, 2e-7, size=N)
    st = sp.spatial_column_state(col, water, 12.5, memory_budget_bytes=1 << 30)
    out = sp.spatial_step(col, st, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=1 << 30)
    assert out.state.t_s == 12.5 + 3600.0 and out.dt_s == 3600.0 and out.state.column_sha256 == col.sha256
    from conftest import common_column

    tot_after = tot_ret = tot_drain = tot_evap = 0.0
    for c in range(N):
        cc = common_column(p, c, mode=mode, bottom=bottom)
        o = ss.step(cc, ss.column_state(cc, water[:, c : c + 1].copy(), 12.5), 3600.0, added_water_m=added[:, c : c + 1].copy(), potential_evaporation_m_s=pet[c : c + 1].copy())
        np.testing.assert_array_equal(out.state.water_m[:, c], o.state.water_m[:, 0])
        np.testing.assert_array_equal(out.drainage_m[c : c + 1], o.drainage_m)
        np.testing.assert_array_equal(out.evaporation_m[c : c + 1], o.evaporation_m)
        np.testing.assert_array_equal(out.surface_return_m[c : c + 1], o.surface_return_m)
        tot_after += o.budget.storage_after_m3
        tot_ret += o.budget.surface_return_m3
        tot_drain += o.budget.drainage_m3
        tot_evap += o.budget.evaporation_m3
    b = out.budget
    assert b.storage_after_m3 == pytest.approx(tot_after, rel=1e-13) and b.drainage_m3 == pytest.approx(tot_drain, rel=1e-13) and abs(b.residual_m3) <= b.bound_m3
    assert b.surface_return_m3 == pytest.approx(tot_ret, rel=1e-13) and b.evaporation_m3 == pytest.approx(tot_evap, rel=1e-13)


def test_pulse_step_and_fit_failure_classification_per_column(mode):
    L, N = 3, 18
    rng = np.random.default_rng(13)
    p = params(rng, L, N, mode=mode)
    col = prepare(make_spec(p, mode=mode, bottom="closed"))
    water = water_kind(rng, p, "mixed")  # a third of the cells are saturated: a pulse cannot fit in a closed profile
    pet = rng.uniform(0, 1e-7, size=N)
    assert_columns_equal(body(col, water, None, pet, 86400.0, None, 2e-3), p, water, np.zeros((L, N)), pet, 86400.0, None, 2e-3, np.ones(N, bool), mode=mode, bottom="closed", gravity=True)
    st = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=1 << 30)
    with pytest.raises(ss.SubsurfaceFitFailure):
        sp.spatial_step_with_recharge(col, st, 86400.0, recharge_depth_m=2e-3, potential_evaporation_m_s=pet, memory_budget_bytes=1 << 30)
    open_col = prepare(make_spec(p, mode=mode, bottom="open"))
    ok = sp.spatial_step_with_recharge(open_col, sp.spatial_column_state(open_col, water, 0.0, memory_budget_bytes=1 << 30), 86400.0, recharge_depth_m=2e-3, potential_evaporation_m_s=pet,
                                       memory_budget_bytes=1 << 30)
    assert ok.state.t_s == 86400.0 and abs(ok.budget.residual_m3) <= ok.budget.bound_m3
