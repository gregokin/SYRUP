"""Step physics on the NumPy reference: the independent audit's hand cases 1-5 under convention B, the root onset
control, the source low-rate branch, b = 0, K = 0, the g limits, standing pond with r = 0, zero supply / zero room
preserving the history, the labelled onset discontinuity, inactive cells, arbitrary layers, dt = 0, time preserved,
the no-contact consistency flag, invalid data refusals and input immutability."""
from __future__ import annotations

import math

import numpy as np
import pytest
from matrix_cases import (
    AREA,
    B,
    G,
    K,
    arrays_of,
    column,
    deficit_b,
    full,
    history_of,
    layers,
    load_root_oracles,
    operator,
    state_of,
)

from maple_syrup import matrix_infiltration as mi
from maple_syrup import subsurface as ss

CASE1_DT = 151.011776539379  # audit case 1 total duration (t_s + tau)


def residuals_ok(b: mi.MatrixBudget) -> bool:
    return abs(b.soil_residual_m3) <= b.bound_m3 and abs(b.pool_residual_m3) <= b.bound_m3


def test_audit_case1_fresh_onset_and_postpond_target():
    col = column()
    op = operator(col)
    state = state_of(col)
    r = 3e-5
    res = mi.matrix_step(op, full(col, r * CASE1_DT), full(col, 0.0), full(col, r), state, history_of(op), CASE1_DT)
    n = col.n_cells
    np.testing.assert_allclose(res.intake_m, 0.00375, rtol=1e-12)
    np.testing.assert_allclose(res.history.wetting_m, 0.005, rtol=1e-12)
    np.testing.assert_allclose(res.history.cumulative_intake_m, 0.00375, rtol=1e-12)
    np.testing.assert_allclose(res.surface_pool_m, 0.00078035329618136, rtol=1e-11)
    b = res.budget
    assert b.n_onsets == n and b.n_ponded == n and b.max_newton_iterations <= mi.NEWTON_MAX_ITERATIONS
    assert b.prepond_intake_m3 == pytest.approx(0.0025 * n * AREA, rel=1e-12)  # F_s, unscaled into J
    assert b.postpond_intake_m3 == pytest.approx(0.00125 * n * AREA, rel=1e-12)  # g x with x = 0.0025
    assert b.room_rejected_m3 == 0.0 and b.unused_capacity_m3 == 0.0 and residuals_ok(b)
    assert res.state.t_s == state.t_s and res.dt_s == CASE1_DT
    np.testing.assert_allclose(res.state.water_m[0], state.water_m[0] + 0.00375, rtol=1e-12)
    np.testing.assert_array_equal(res.state.water_m[1:], state.water_m[1:])


def test_source_onset_time_is_recovered_exactly_at_frozen_deficit():
    """A single interval of exactly t_s from a dry history admits r t_s unscaled with no onset; the next interval
    is ponded from the start (J >= F_s) whatever its length: the threshold is not restarted."""
    col = column(n_cells=2)
    op = operator(col)
    state = state_of(col)
    r = 3e-5
    ts = mi.reference_onset_time_s(r, K, deficit_b(col, state)[0], G)
    assert ts == pytest.approx(250.0 / 3.0, rel=1e-15)
    first = mi.matrix_step(op, full(col, r * ts), full(col, 0.0), full(col, r), state, history_of(op), ts)
    assert first.budget.n_onsets == 0 and first.budget.n_ponded == 0
    np.testing.assert_allclose(first.history.wetting_m, 0.0025, rtol=1e-15)
    np.testing.assert_allclose(first.intake_m, r * ts, rtol=1e-15)
    # one more tiny interval: ponded from the start (no onset counted: the threshold was crossed at the boundary)
    second = mi.matrix_step(op, full(col, r * 1.0), full(col, 0.0), full(col, r), first.state, first.history, 1.0)
    assert second.budget.n_ponded == 2 and second.budget.n_onsets == 0
    assert second.budget.prepond_intake_m3 == 0.0 and second.budget.postpond_intake_m3 > 0.0
    # slightly longer than t_s: the onset lies inside
    third = mi.matrix_step(op, full(col, r * (ts + 1.0)), full(col, 0.0), full(col, r), state, history_of(op), ts + 1.0)
    assert third.budget.n_onsets == 2


def test_audit_case2_changing_rain_does_not_restart_the_threshold():
    col = column(n_cells=3)
    op = operator(col)
    state = state_of(col)
    low = 5e-6  # <= K: unponded, all admitted, J advances unscaled
    first = mi.matrix_step(op, full(col, low * 200.0), full(col, 0.0), full(col, low), state, history_of(op), 200.0)
    np.testing.assert_allclose(first.history.wetting_m, 0.001, rtol=1e-15)
    np.testing.assert_array_equal(first.intake_m, first.history.cumulative_intake_m)
    assert first.budget.n_ponded == 0
    r = 3e-5
    b_now = deficit_b(col, first.state)  # the declared moisture feedback: b re-evaluated from the accepted state
    fs = mi.reference_onset_threshold_m(r, K, b_now, G)
    t_p = (fs - 0.001) / r
    assert np.all(t_p < 50.0) and np.all(t_p > 48.0)  # frozen-b value would be exactly 50 s
    second = mi.matrix_step(op, full(col, r * 60.0), full(col, 0.0), full(col, r), first.state, first.history, 60.0)
    assert second.budget.n_onsets == 3
    np.testing.assert_allclose(second.budget.prepond_intake_m3, float(np.sum(r * t_p)) * AREA, rtol=1e-12)
    x = mi.green_ampt_increment_m(fs, b_now, K, 60.0 - t_p)[0]
    np.testing.assert_allclose(second.budget.postpond_intake_m3, float(np.sum(G * x)) * AREA, rtol=1e-12)
    np.testing.assert_allclose(second.history.wetting_m, fs + x, rtol=1e-12)


def test_audit_case3_standing_pond_recession_with_zero_rain():
    col = column(n_cells=2)
    op = operator(col)
    state = state_of(col)
    # longer than the exhaustion time: the whole pond infiltrates, J = 0.005
    res = mi.matrix_step(op, full(col, 0.0025), full(col, 0.0025), full(col, 0.0), state, history_of(op), 200.0)
    np.testing.assert_array_equal(res.intake_m, 0.0025)
    np.testing.assert_allclose(res.history.wetting_m, 0.005, rtol=1e-15)
    np.testing.assert_array_equal(res.surface_pool_m, 0.0)
    assert res.budget.n_ponded == 2 and res.budget.n_onsets == 0 and res.budget.unused_capacity_m3 > 0.0
    # exactly the exhaustion time: the candidate equals the pond within roundoff
    t_ex = (0.005 - 0.01 * math.log(1.5)) / 1e-5
    assert t_ex == pytest.approx(94.5348918918356, rel=1e-13)
    res2 = mi.matrix_step(op, full(col, 0.0025), full(col, 0.0025), full(col, 0.0), state, history_of(op), t_ex)
    np.testing.assert_allclose(res2.budget.candidate_m3 / (2 * AREA), 0.0025, rtol=1e-12)
    # half the exhaustion time: part of the pond remains, no new rain invented
    res3 = mi.matrix_step(op, full(col, 0.0025), full(col, 0.0025), full(col, 0.0), state, history_of(op), t_ex / 2)
    assert np.all(res3.surface_pool_m > 0.0) and np.all(res3.intake_m < 0.0025)
    np.testing.assert_allclose(res3.history.wetting_m, 0.0 + res3.intake_m / G, rtol=1e-15)


def test_audit_case4_capacity_limited_standing_pond():
    lay = layers(fc=0.126, sat=0.127)  # top room = (0.127 - 0.125) x 0.5 = 0.001 m
    col = column(lay, n_cells=2)
    op = operator(col, psi=10.0)  # b = 10 x (0.126 - 0.125) = 0.01 m (within roundoff of the non-dyadic deficit)
    state = state_of(col)
    room = float(lay.saturation_m[0]) - 0.0625
    assert room == pytest.approx(0.001, rel=1e-12)
    assert deficit_b(col, state, 10.0)[0] == pytest.approx(0.01, rel=1e-12)
    res = mi.matrix_step(op, full(col, 0.02), full(col, 0.02), full(col, 0.0), state, history_of(op, 0.002), 60.0)
    np.testing.assert_allclose(res.intake_m, room, rtol=1e-15)
    np.testing.assert_allclose(res.history.wetting_m, 0.002 + room / G, rtol=1e-15)
    np.testing.assert_allclose(res.surface_pool_m, 0.02 - room, rtol=1e-15)
    np.testing.assert_allclose(res.state.water_m[0], lay.saturation_m[0], rtol=1e-15)
    b = res.budget
    assert b.room_rejected_m3 == pytest.approx((float(load_root_oracles()["cases"][0]["actual_matrix_intake_m"]) - room) * 2 * AREA, rel=1e-9)
    assert b.unused_capacity_m3 == 0.0 and residuals_ok(b)
    # a potential solve for the remaining time must not advance J further: a second step with zero room is inert
    again = mi.matrix_step(op, res.surface_pool_m, res.surface_pool_m, full(col, 0.0), res.state, res.history, 600.0)
    assert again.history.wetting_m.tobytes() == res.history.wetting_m.tobytes()
    assert again.budget.intake_m3 == 0.0 and again.budget.room_rejected_m3 > 0.0


def test_audit_case5_declared_b_split_differs_from_the_continuous_input_value():
    col = column(n_cells=1)
    op = operator(col)
    state = state_of(col)
    t_dry = (0.002 - 0.01 * math.log(7.0 / 6.0)) / 1e-5
    h0 = 0.001 - 2e-6 * t_dry
    dt = t_dry + 50.0
    res = mi.matrix_step(op, full(col, h0 + 2e-6 * dt), full(col, h0), full(col, 2e-6), state, history_of(op, 0.002), dt)
    assert res.intake_m[0] == pytest.approx(0.0011, rel=1e-13)  # both conventions conserve the same water
    assert res.history.wetting_m[0] == pytest.approx(0.0042, rel=1e-13)  # B: the whole pool through the ponded law
    assert res.surface_pool_m[0] == pytest.approx(0.0, abs=1e-18)
    assert res.budget.n_ponded == 1 and res.budget.n_onsets == 0


def test_root_constant_rate_onset_control_is_reproduced():
    oc = load_root_oracles()["constant_rate_onset_control"]
    col = column(layers(fc=0.25, sat=0.9), n_cells=1)
    op = operator(col, k=1e-6, psi=0.16)  # b = 0.16 x 0.125 = 0.02
    state = state_of(col)
    assert deficit_b(col, state, 0.16)[0] == pytest.approx(0.02, rel=1e-15)
    r = float(oc["incoming_rate_m_s"])
    dt = float(oc["dt_s"])
    res = mi.matrix_step(op, full(col, r * dt), full(col, 0.0), full(col, r), state, history_of(op), dt)
    assert res.history.wetting_m[0] == pytest.approx(float(oc["J_final_m"]), rel=1e-14)
    assert res.intake_m[0] == pytest.approx(float(oc["total_matrix_intake_m"]), rel=1e-14)
    assert res.surface_pool_m[0] == pytest.approx(float(oc["surface_final_m"]), rel=1e-13)
    assert res.budget.prepond_intake_m3 == pytest.approx(float(oc["prepond_intake_m"]) * AREA, rel=1e-14)
    assert res.budget.n_onsets == 1


def test_low_rate_branch_admits_the_pool_unscaled_including_g_zero():
    col = column(n_cells=3)
    state = state_of(col)
    for g in (0.5, 1.0, 0.0):
        op = operator(col, g=g)
        pool = np.array([0.003, 0.0, 0.001])
        res = mi.matrix_step(op, pool, full(col, 0.0), full(col, K), state, history_of(op, 0.002, 0.004), 600.0)  # r == K
        np.testing.assert_array_equal(res.intake_m, pool)
        np.testing.assert_array_equal(res.history.wetting_m, 0.002 + pool)  # unscaled advance
        np.testing.assert_array_equal(res.history.cumulative_intake_m, 0.004 + pool)
        np.testing.assert_array_equal(res.surface_pool_m, 0.0)
        assert res.budget.n_ponded == 0 and res.budget.candidate_m3 == pytest.approx(K * 600.0 * 3 * AREA)


def test_zero_deficit_is_gravity_only_green_ampt():
    col = column(n_cells=2)
    op = operator(col)
    state = state_of(col, theta0=0.3)  # above field capacity: b = 0
    assert np.all(deficit_b(col, state) == 0.0)
    # r > K, no contact: immediate ponding (threshold 0), postpond g K dt
    res = mi.matrix_step(op, full(col, 3e-5 * 100.0), full(col, 0.0), full(col, 3e-5), state, history_of(op), 100.0)
    np.testing.assert_array_equal(res.intake_m, G * (K * 100.0))
    np.testing.assert_array_equal(res.history.wetting_m, K * 100.0)  # J advances by intake / g = x
    assert res.budget.n_ponded == 2 and res.budget.n_onsets == 0 and res.budget.max_newton_iterations == 0
    # r <= K, no contact: the low-rate branch still admits the supply up to the room
    res2 = mi.matrix_step(op, full(col, K * 100.0), full(col, 0.0), full(col, K), state, history_of(op), 100.0)
    np.testing.assert_array_equal(res2.intake_m, K * 100.0)
    # contact pond: g K dt as well
    res3 = mi.matrix_step(op, full(col, 0.01), full(col, 0.01), full(col, 0.0), state, history_of(op, 0.3), 100.0)
    np.testing.assert_array_equal(res3.intake_m, G * (K * 100.0))


def test_zero_conductivity_gives_no_intake_and_no_front_advance():
    col = column(n_cells=2)
    op = operator(col, k=0.0)
    state = state_of(col)
    hist = history_of(op, 0.002, 0.005)
    for pool, standing, r in ((0.01, 0.01, 0.0), (3e-5 * 60.0, 0.0, 3e-5), (0.0, 0.0, 0.0)):
        res = mi.matrix_step(op, full(col, pool), full(col, standing), full(col, r), state, hist, 60.0)
        np.testing.assert_array_equal(res.intake_m, 0.0)
        assert res.history.wetting_m.tobytes() == hist.wetting_m.tobytes()
        assert res.history.cumulative_intake_m.tobytes() == hist.cumulative_intake_m.tobytes()
        np.testing.assert_array_equal(res.surface_pool_m, pool)
        assert res.state.water_m.tobytes() == state.water_m.tobytes()
        assert res.budget.candidate_m3 == 0.0 and residuals_ok(res.budget)


def test_vegetation_factor_limits():
    col = column(n_cells=2)
    state = state_of(col)
    # g = 0: contact pond -> zero postpond intake, J unchanged, no division
    op0 = operator(col, g=0.0)
    h = history_of(op0, 0.002)
    res = mi.matrix_step(op0, full(col, 0.01), full(col, 0.01), full(col, 0.0), state, h, 60.0)
    np.testing.assert_array_equal(res.intake_m, 0.0)
    assert res.history.wetting_m.tobytes() == h.wetting_m.tobytes() and res.budget.n_ponded == 2
    # g = 0, no contact, r > K: threshold 0 -> immediate ponding with zero capacity, pool untouched
    res = mi.matrix_step(op0, full(col, 3e-5 * 60.0), full(col, 0.0), full(col, 3e-5), state, h, 60.0)
    np.testing.assert_array_equal(res.intake_m, 0.0)
    np.testing.assert_array_equal(res.surface_pool_m, 3e-5 * 60.0)
    assert res.budget.n_ponded == 2 and res.budget.n_onsets == 0
    # g = 1: M = J within the episode (standard cumulative Green-Ampt), continuous onset capacity
    op1 = operator(col, g=1.0)
    r = 3e-5
    res = mi.matrix_step(op1, full(col, r * 600.0), full(col, 0.0), full(col, r), state, history_of(op1), 600.0)
    assert res.budget.n_onsets == 2
    np.testing.assert_array_equal(res.history.wetting_m, res.history.cumulative_intake_m)
    np.testing.assert_allclose(res.budget.prepond_intake_m3, 2 * AREA * B * K / (r - K), rtol=1e-12)  # F_s with g = 1
    # tiny g: postpond intake tiny but J advances by the full hydraulic increment (root oracle)
    oracle = next(c for c in load_root_oracles()["cases"] if c["name"] == "ponded_small_vegetation_factor")
    op_t = operator(col, g=1e-6)
    res = mi.matrix_step(op_t, full(col, 1.0), full(col, 1.0), full(col, 0.0), state, history_of(op_t, 0.002), 60.0)
    np.testing.assert_allclose(res.intake_m, float(oracle["actual_matrix_intake_m"]), rtol=1e-14)
    np.testing.assert_allclose(res.history.wetting_m, float(oracle["J_final_m"]), rtol=1e-14)
    assert residuals_ok(res.budget)


def test_zero_supply_and_zero_room_preserve_the_history_without_recovery():
    col = column(n_cells=3)
    op = operator(col)
    state = state_of(col)
    hist = history_of(op, np.array([0.0, 0.002, 0.1]), np.array([0.0, 0.003, 0.2]))
    zero = full(col, 0.0)
    for dt in (0.0, 1.0, 3600.0, 1e7):  # dry intervals of any length: no intake, no change, no reset
        res = mi.matrix_step(op, zero, zero, zero, state, hist, dt)
        assert res.history.wetting_m.tobytes() == hist.wetting_m.tobytes()
        assert res.history.cumulative_intake_m.tobytes() == hist.cumulative_intake_m.tobytes()
        assert res.state.water_m.tobytes() == state.water_m.tobytes() and res.budget.intake_m3 == 0.0
        assert res.history.wetting_m is not hist.wetting_m  # fresh arrays, caller's untouched
    # saturated top layer: contact pond, zero room -> nothing admitted, everything stays in the pool
    sat_state = state_of(col, theta0=0.9)
    res = mi.matrix_step(op, full(col, 0.02), full(col, 0.02), zero, sat_state, hist, 600.0)
    np.testing.assert_array_equal(res.intake_m, 0.0)
    assert res.history.wetting_m.tobytes() == hist.wetting_m.tobytes()
    np.testing.assert_array_equal(res.surface_pool_m, 0.02)
    assert res.budget.room_rejected_m3 > 0.0 and res.budget.candidate_m3 == res.budget.room_rejected_m3 + res.budget.unused_capacity_m3


def test_onset_capacity_discontinuity_is_r_minus_one_minus_g_K():
    """Just before the onset the physical intake rate is r (unponded, all supply); just after it the ponded law
    with J = F_s gives g K (1 + b/F_s) = r - (1 - g) K. Measured with short intervals around the crossing."""
    col = column(layers(dz0=1000.0), n_cells=1)  # thick top layer: b stays 0.01 to 1e-7 relative
    op = operator(col)
    state = state_of(col)
    r = 3e-5
    ts = mi.reference_onset_time_s(r, K, B, G)
    before = mi.matrix_step(op, full(col, r * ts), full(col, 0.0), full(col, r), state, history_of(op), ts)
    assert before.intake_m[0] / ts == pytest.approx(r, rel=1e-15)
    h = 1e-3
    after = mi.matrix_step(op, full(col, r * h), full(col, 0.0), full(col, r), before.state, before.history, h)
    rate_after = after.intake_m[0] / h
    _, expected = mi.reference_onset_capacities_m_s(r, K, G)
    assert rate_after == pytest.approx(expected, rel=1e-4)  # O(h) capacity decay + 2e-5 relative drift of b
    assert rate_after < r and expected == pytest.approx(2.5e-5)
    assert rate_after < expected  # the capacity only decreases after the onset


def test_inactive_cells_keep_everything_and_require_zero_rate():
    col = column(n_cells=4, active=[True, False, True, False])
    op = operator(col)
    state = state_of(col)
    hist = history_of(op, np.array([0.0, 0.5, 0.0, 0.7]), np.array([0.0, 0.6, 0.0, 0.8]))
    pool = np.array([0.01, 0.02, 0.01, 0.03])
    res = mi.matrix_step(op, pool, pool, full(col, 0.0), state, hist, 600.0)
    for name, a in arrays_of(res).items():
        got = np.asarray(a)
        if name == "intake":
            assert got[1] == got[3] == 0.0 and got[0] > 0.0 and got[2] > 0.0
        elif name == "water":
            np.testing.assert_array_equal(got[:, [1, 3]], state.water_m[:, [1, 3]])
        else:
            source = {"pool": pool, "wetting": hist.wetting_m, "cumulative": hist.cumulative_intake_m}[name]
            np.testing.assert_array_equal(got[[1, 3]], source[[1, 3]])
    assert res.budget.pool_before_m3 == 0.02 * AREA and res.budget.n_ponded == 2
    with pytest.raises(mi.MatrixStepError) as info:
        mi.matrix_step(op, pool, pool, np.array([0.0, 1e-6, 0.0, 0.0]), state, hist, 600.0)
    assert info.value.bit == 9 and info.value.n_flagged == 1


def test_arbitrary_layers_only_the_top_layer_changes():
    rng = np.random.default_rng(5)
    lay = layers(deeper=(0.3, 0.2, 0.9, 1.5))
    col = column(lay, n_cells=50)
    op = operator(col, k=rng.uniform(1e-6, 5e-5, 50), psi=rng.uniform(0.0, 0.5, 50), g=rng.uniform(0.0, 1.0, 50))
    thick = np.array(lay.thickness_m)
    water = np.ascontiguousarray(rng.uniform(0.0, 0.9, (5, 50)) * thick[:, None])
    state = ss.column_state(col, water)
    pool = rng.uniform(0.0, 0.02, 50)
    standing = pool * (rng.random(50) < 0.5)
    rate = rng.uniform(0.0, 5e-5, 50)
    pool_b = np.where(standing > 0.0, pool, np.minimum(pool, rate * 300.0))  # consistent no-contact pools
    res = mi.matrix_step(op, pool_b, standing, rate, state, history_of(op, rng.uniform(0.0, 0.01, 50)), 300.0)
    np.testing.assert_array_equal(res.state.water_m[1:], water[1:])
    assert np.all(res.state.water_m[0] >= water[0]) and np.all(res.state.water_m[0] <= lay.saturation_m[0] + lay.capacity_tolerance_m[0])
    assert residuals_ok(res.budget) and np.all(res.surface_pool_m >= 0.0)
    np.testing.assert_allclose(res.surface_pool_m + res.intake_m, pool_b, rtol=1e-14, atol=1e-18)


def test_dt_zero_is_the_identity_and_time_is_preserved():
    col = column(n_cells=3)
    op = operator(col)
    state = state_of(col, t_s=123.5)
    hist = history_of(op, 0.004, 0.009)
    pool = np.array([0.01, 0.0, 0.002])
    res = mi.matrix_step(op, pool, pool, full(col, 7e-5), state, hist, 0.0)
    assert res.surface_pool_m.tobytes() == pool.tobytes() and res.state.water_m.tobytes() == state.water_m.tobytes()
    assert res.history.wetting_m.tobytes() == hist.wetting_m.tobytes()
    assert res.budget.intake_m3 == 0.0 and res.budget.soil_after_m3 == res.budget.soil_before_m3
    assert res.state.t_s == 123.5 and res.dt_s == 0.0
    later = mi.matrix_step(op, pool, pool, full(col, 0.0), state, hist, 60.0)
    assert later.state.t_s == 123.5  # the owning step advances time, never this stage


def test_no_contact_pool_larger_than_the_supply_is_refused_without_mutation():
    col = column(n_cells=2)
    op = operator(col)
    state = state_of(col)
    hist = history_of(op)
    pool = np.array([0.01, 0.0])
    snapshot = (pool.tobytes(), state.water_m.tobytes(), hist.wetting_m.tobytes())
    with pytest.raises(mi.MatrixStepError) as info:
        mi.matrix_step(op, pool, full(col, 0.0), full(col, 1e-6), state, hist, 60.0)  # 0.01 > 6e-5
    assert info.value.bit == 14 and info.value.n_flagged == 1
    assert (pool.tobytes(), state.water_m.tobytes(), hist.wetting_m.tobytes()) == snapshot
    with pytest.raises(mi.MatrixStepError) as info:
        mi.matrix_step(op, pool, full(col, 0.0), full(col, 0.0), state, hist, 0.0)  # even at dt = 0
    assert info.value.bit == 14
    # declared contact makes the same pool legitimate
    ok = mi.matrix_step(op, pool, pool, full(col, 1e-6), state, hist, 60.0)
    assert ok.budget.n_ponded == 1


@pytest.mark.parametrize("field, bad, bit", [
    ("pool", np.nan, 3), ("pool", -1e-9, 4), ("standing", np.inf, 5), ("standing", -1.0, 6),
    ("rate", np.nan, 7), ("rate", -1e-9, 8), ("wetting", np.nan, 10), ("wetting", -1e-12, 11),
    ("cumulative", np.inf, 12), ("cumulative", -1.0, 13), ("soil", np.nan, 0), ("soil", -1e-9, 1), ("soil", 1.0, 2),
])
def test_invalid_data_is_flagged_with_the_lowest_bit_and_nothing_is_published(field, bad, bit):
    col = column(n_cells=3)
    op = operator(col)
    state = state_of(col)
    hist = history_of(op)
    pool, standing, rate = full(col, 0.001), full(col, 0.001), full(col, 0.0)
    water = np.array(state.water_m)
    wet, cum = np.array(hist.wetting_m), np.array(hist.cumulative_intake_m)
    {"pool": pool, "standing": standing, "rate": rate, "wetting": wet, "cumulative": cum}.get(field, water[0])[1] = bad
    state = ss.column_state(col, water)
    hist = mi.wetting_history(op, wetting_m=wet, cumulative_intake_m=cum)
    with pytest.raises(mi.MatrixStepError) as info:
        mi.matrix_step(op, pool, standing, rate, state, hist, 60.0)
    assert info.value.bit == bit and info.value.n_flagged == 1
    assert mi.FLAG_MESSAGES[bit] in str(info.value)


def test_budget_accounting_identities_on_a_random_population():
    rng = np.random.default_rng(11)
    n = 400
    lay = layers(deeper=(0.4, 1.0))
    mask = rng.random(n) < 0.8
    mask[0] = True
    col = column(lay, n_cells=n, active=mask)
    op = operator(col, k=rng.uniform(0.0, 3e-5, n), psi=rng.uniform(0.0, 0.3, n), g=rng.choice([0.0, 0.2, 0.5, 1.0], n))
    thick = np.array(lay.thickness_m)
    water = np.ascontiguousarray(rng.uniform(0.0, 1.0, (3, n)) * (np.array(lay.theta_saturation) * thick)[:, None])
    water[0, : n // 10] = lay.saturation_m[0]  # some saturated top layers (zero room)
    state = ss.column_state(col, water)
    rate = rng.uniform(0.0, 6e-5, n) * mask
    standing = rng.uniform(0.0, 0.02, n) * (rng.random(n) < 0.5)
    dt = 450.0
    pool = np.where(standing > 0.0, standing + rate * dt, rate * dt) * (rng.random(n) < 0.95)
    hist = history_of(op, rng.uniform(0.0, 0.02, n) * (rng.random(n) < 0.7), rng.uniform(0.0, 0.05, n))
    res = mi.matrix_step(op, pool, standing, rate, state, hist, dt)
    b = res.budget
    assert residuals_ok(b) and b.max_newton_iterations <= mi.NEWTON_MAX_ITERATIONS
    assert b.candidate_m3 == pytest.approx(b.intake_m3 + b.room_rejected_m3 + b.unused_capacity_m3, rel=1e-12)
    assert b.intake_m3 == pytest.approx(b.prepond_intake_m3 + b.postpond_intake_m3, rel=1e-12)
    assert b.intake_m3 == pytest.approx(float(np.sum(res.intake_m[mask])) * AREA, rel=1e-12)
    assert b.pool_before_m3 - b.pool_after_m3 == pytest.approx(b.intake_m3, rel=1e-12)
    assert b.soil_after_m3 - b.soil_before_m3 == pytest.approx(b.intake_m3, rel=1e-12)
    assert 0 <= b.n_onsets <= b.n_ponded <= int(mask.sum())
    assert b.wetting_after_m3 >= b.wetting_before_m3
    np.testing.assert_allclose(res.history.cumulative_intake_m, hist.cumulative_intake_m + res.intake_m, rtol=1e-15)
    # the inputs were never written and the outputs are fresh
    assert not np.shares_memory(res.surface_pool_m, pool) and not np.shares_memory(res.state.water_m, state.water_m)
    assert not np.shares_memory(res.history.wetting_m, hist.wetting_m)
