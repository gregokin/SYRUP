"""Floating withdrawals: Root's independent controls, rock slots, plants, inactive placeholders, failure behaviour."""
from __future__ import annotations

import numpy as np
import pytest
from builders import REFUSALS, column, state

from maple_syrup import floating_transpiration as ft
from maple_syrup import subsurface as ss

EDGES = [0.0, 0.1, 0.3, 0.6]


def col(backend, depth=(0.15,), active=None):
    return column(EDGES, list(depth), tr=0.1, tf=0.3, ts=0.5, backend=backend, active=active, stress=0.2)


def test_root_control_availability_and_withdrawal(backend):
    c = col(backend)
    s = state(c, [[0.025], [0.020], [0.0]])
    av = ft.floating_layer_availability_m(c, s)
    assert np.allclose(av[:, 0], [0.015, 0.015, 0.0], atol=1e-15)
    q = np.array([[0.020], [0.010], [0.010]])
    r = ft.floating_withdraw(c, s, q)
    assert np.allclose(r.actual_m[:, 0], [0.015, 0.010, 0.0], atol=1e-15)
    assert np.allclose(r.unmet_m[:, 0], [0.005, 0.0, 0.010], atol=1e-15)
    assert np.allclose(r.state.water_m[:, 0], [0.010, 0.010, 0.0], atol=1e-15)
    assert r.actual_m.sum() / q.sum() == pytest.approx(0.625)
    assert r.state.t_s == s.t_s


def test_all_rock_request_is_unmet_without_nan(backend):
    c = col(backend, depth=(0.0,))
    s = state(c, np.zeros((3, 1)))
    r = ft.floating_withdraw(c, s, np.full((3, 1), 0.01))
    assert np.all(r.actual_m == 0.0) and np.allclose(r.unmet_m, 0.01) and np.all(np.isfinite(r.scale)) and np.all(r.scale == 0.0)


def test_demand_proportional_split_from_the_aggregate_scale(backend):
    """LEAF aggregates plant requests and receives ONE `scale`; each plant's accepted amount is `request_p * scale` (no dense per-plant helper exists in production)."""
    c = col(backend)
    s = state(c, [[0.025], [0.020], [0.0]])
    a, b = np.array([[0.010], [0.0], [0.0]]), np.array([[0.020], [0.010], [0.010]])
    r = ft.floating_withdraw(c, s, a + b)
    pa, pb = a * r.scale, b * r.scale
    assert np.allclose(pa + pb, r.actual_m, atol=1e-15)
    assert pa[0, 0] / pb[0, 0] == pytest.approx(0.010 / 0.020)
    assert not hasattr(ft, "floating_withdraw_plants")


def test_inactive_placeholder_bits_preserved_and_positive_request_refused(backend):
    c = col(backend, depth=(0.15, 0.15), active=np.array([True, False]))
    w = np.array([[0.025, 0.9], [0.020, 0.9], [0.0, 0.9]])  # the placeholder holds water beyond any capacity: no capacity law
    s = state(c, w)
    r = ft.floating_withdraw(c, s, np.zeros((3, 2)))
    assert np.array_equal(r.state.water_m[:, 1], w[:, 1])
    bad = np.zeros((3, 2))
    bad[0, 1] = 0.001
    before = s.water_m.copy()
    with pytest.raises(ss.SubsurfaceError):
        ft.floating_withdraw(c, s, bad)
    assert np.array_equal(s.water_m, before)
    assert np.array_equal(ft.floating_layer_availability_m(c, s)[:, 1], np.zeros(3))


def test_invalid_request_refused_and_inputs_unchanged(backend):
    c = col(backend)
    s = state(c, [[0.025], [0.020], [0.0]])
    q = np.array([[np.nan], [0.0], [0.0]])
    q0 = q.copy()
    with pytest.raises(ss.SubsurfaceError):
        ft.floating_withdraw(c, s, q)
    assert np.array_equal(q, q0, equal_nan=True)


def test_admission_refuses_before_allocation(backend):
    c = col(backend)
    s = state(c, [[0.025], [0.020], [0.0]])
    with pytest.raises(REFUSALS, match="nothing was allocated"):
        ft.floating_withdraw(c, s, np.zeros((3, 1)), memory_budget_bytes=0)


def test_common_full_soil_reference_matches_shared_body(backend):
    """Full soil (D >= H) must reproduce the common per-cell body bit for bit (the same `_fused_numpy` with explicit common bounds)."""
    from maple_syrup import subsurface_transpiration as st
    c = column(EDGES, [0.6, 0.9, 0.7], tr=0.1, tf=0.3, ts=0.5, backend=backend, stress=0.2)
    rng = np.random.default_rng(3)
    dz = np.diff(EDGES)
    w = rng.uniform(0.0, 0.49, (3, 3)) * dz[:, None]
    s = state(c, w)
    q = rng.uniform(0.0, 0.02, (3, 3))
    r = ft.floating_withdraw(c, s, q)
    coef = ft.volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    res, satb = 0.1 * dz, 0.5 * dz + coef * (0.5 * dz)
    out_w, actual, unmet, scale = (np.empty((3, 3)) for _ in range(4))
    st._fused_numpy(w, q, res, satb, np.ones(3, dtype=bool), out_w, actual, unmet, scale, np.empty((st._N_ROWS, 3)))
    assert np.array_equal(r.state.water_m, out_w) and np.array_equal(r.actual_m, actual)
