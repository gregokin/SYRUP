"""Step physics on the NumPy reference: analytic one-hour response, repeated partitions, beta endpoints, dt = 0,
the ONE aggregate rate cap, capacity rejections staying on the surface, one shared source across species and
receivers, the exact two-layer reference baseline, masks, invalid forcing, immutability and refusals."""
from __future__ import annotations

import math

import numpy as np
import pytest
from pref_cases import AREA, HOUR, arrays_of, column, from_cells, layers, operator, state_of

from maple_syrup import preferential_infiltration as pi
from maple_syrup import subsurface as ss


def test_analytic_one_hour_matches_beta_times_surface():
    col = column()
    cover = np.array([0.0, 1.0, 3.0, 0.5])
    op = operator(col, a3=0.5, max_fraction=0.5, background=(0.0, 1.0), species=[(0.0, 1.0)], contributions=[cover])
    H = np.array([0.01, 0.02, 0.0, 0.004])
    state = state_of(col, [[0.03125, 0.09375]] * 4)  # theta 0.125 in both layers, ample room
    r = pi.preferential_step(op, H, state, HOUR)
    expected = H * op.beta
    np.testing.assert_allclose(r.intake_total_m, expected, rtol=4e-16, atol=0.0)
    np.testing.assert_allclose(r.intake_m[1], expected, rtol=4e-16, atol=0.0)
    np.testing.assert_array_equal(r.intake_m[0], 0.0)
    np.testing.assert_allclose(r.surface_water_m, H - expected, rtol=4e-16, atol=0.0)
    np.testing.assert_array_equal(r.state.water_m[0], state.water_m[0])
    np.testing.assert_allclose(r.state.water_m[1], state.water_m[1] + expected, rtol=4e-16, atol=0.0)
    assert r.state.t_s == state.t_s == 0.0 and r.dt_s == HOUR  # time preserved: the owning step advances it
    b = r.budget
    assert b.intake_m3 == pytest.approx(float(expected.sum()) * AREA, rel=1e-15)
    assert b.isolated_attempt_m3 == b.attempted_m3 == b.intake_m3
    assert b.rate_capped_m3 == 0.0 and b.capacity_rejected_m3 == 0.0
    assert abs(b.soil_residual_m3) <= b.bound_m3 and abs(b.surface_residual_m3) <= b.bound_m3
    assert b.soil_before_m3 == pytest.approx(ss.storage_m3(col, state), rel=1e-15)
    assert b.soil_after_m3 == pytest.approx(ss.storage_m3(col, r.state), rel=1e-15)
    assert b.surface_before_m3 == float(H.sum()) * AREA


@pytest.mark.parametrize("parts", [(600.0,) * 6, (0.5, 1800.0, 1799.5), (1e-3,) * 3 + (3599.997,), (3600.0,)])
def test_repeated_partitions_match_the_isolated_uncapped_hour(parts):
    assert sum(parts) == pytest.approx(HOUR, rel=1e-15)
    col = column()
    op = operator(col, a3=0.5, max_fraction=0.6, background=(0.0, 1.0), species=[(0.0, 1.0)],
                  contributions=[np.array([0.0, 1.0, 3.0, 0.5])])
    H = np.array([0.01, 0.02, 0.03, 0.004])
    state = state_of(col, [[0.03125, 0.09375]] * 4)
    surface, soil = H, state
    removed = np.zeros(4)
    for dt in parts:
        r = pi.preferential_step(op, surface, soil, dt)
        removed += r.intake_total_m
        surface, soil = r.surface_water_m, r.state
    expected = H * op.beta
    np.testing.assert_allclose(removed, expected, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(H - surface, expected, rtol=1e-13, atol=0.0)
    # and a 600 s sub-step alone is NOT beta H: the hourly fraction is never re-applied per short step
    r = pi.preferential_step(op, H, state, 600.0)
    assert np.all(r.intake_total_m < expected)
    np.testing.assert_allclose(r.intake_total_m, H * pi.isolated_intake_fraction(op.beta, 600.0), rtol=4e-16)


def test_beta_zero_and_one_and_dt_zero():
    col = column()
    H = np.array([0.01, 0.02, 0.03, 0.004])
    state = state_of(col, [[0.03125, 0.09375]] * 4)
    zero = operator(col, a3=0.5, max_fraction=0.0)
    r = pi.preferential_step(zero, H, state, 1e6)
    assert r.surface_water_m.tobytes() == H.tobytes() and r.state.water_m.tobytes() == state.water_m.tobytes()
    assert r.budget.intake_m3 == 0.0 and r.budget.isolated_attempt_m3 == 0.0
    one = operator(col, a3=1.0, max_fraction=1.0)
    for dt in (1e-9, 1.0, HOUR, 1e7):
        r = pi.preferential_step(one, H, state, dt)
        np.testing.assert_array_equal(r.surface_water_m, 0.0)  # all of H for any dt > 0 (room is ample)
        np.testing.assert_array_equal(r.intake_m[1], H)
        assert r.budget.intake_m3 == float(H.sum()) * AREA
    r = pi.preferential_step(one, H, state, 0.0)
    assert r.surface_water_m.tobytes() == H.tobytes() and r.state.water_m.tobytes() == state.water_m.tobytes()
    assert r.budget.intake_m3 == 0.0 and r.budget.soil_after_m3 == r.budget.soil_before_m3 and r.dt_s == 0.0
    assert not np.shares_memory(r.surface_water_m, H) and not np.shares_memory(r.state.water_m, state.water_m)


def test_one_aggregate_rate_cap():
    col = column()
    op = operator(col, a3=1.0, max_fraction=1.0, max_rate=np.array([1e-6, 1e-6, math.inf, 0.0]))
    H = np.array([0.01, 0.0005, 0.02, 0.03])
    state = state_of(col, [[0.03125, 0.09375]] * 4)
    r = pi.preferential_step(op, H, state, 600.0)
    cap = 1e-6 * 600.0
    np.testing.assert_array_equal(r.intake_total_m, [cap, 0.0005, 0.02, 0.0])  # min(H f, rate dt), f = 1 here
    np.testing.assert_array_equal(r.surface_water_m, H - np.array([cap, 0.0005, 0.02, 0.0]))
    b = r.budget
    assert b.isolated_attempt_m3 == float(H.sum()) * AREA
    assert b.rate_capped_m3 == pytest.approx((0.01 - cap + 0.03) * AREA, rel=1e-15)
    assert b.capacity_rejected_m3 == 0.0
    # with beta < 1 the cap binds against H f, not against H
    half = operator(col, a3=0.5, max_fraction=1.0, max_rate=np.full(4, 2e-6))
    r = pi.preferential_step(half, H, state, HOUR)
    expected = np.minimum(H * 0.5, 2e-6 * HOUR)
    np.testing.assert_allclose(r.intake_total_m, expected, rtol=4e-16)


def test_capacity_rejections_stay_on_the_surface_and_are_not_redistributed():
    col = column()
    op = operator(col, a3=1.0, max_fraction=1.0, background=(0.0, 1.0))  # everything to layer 1
    # layer 1 saturation depth 0.375: rooms 0.0, 0.001, 0.375 - 0.09375, 0.0 (cell 3 above field capacity but room)
    state = state_of(col, [[0.03125, 0.375], [0.03125, 0.374], [0.03125, 0.09375], [0.0, 0.375]])
    H = np.array([0.01, 0.01, 0.01, 0.01])
    r = pi.preferential_step(op, H, state, HOUR)
    rooms = np.maximum(0.375 - state.water_m[1], 0.0)  # the very FP64 room the kernel forms (0.375 - 0.374 != 0.001)
    expected = np.minimum(H, rooms)
    np.testing.assert_array_equal(r.intake_m[1], expected)
    np.testing.assert_array_equal(r.intake_m[0], 0.0)  # layer 0 has room but is NOT a receiver: no redistribution
    np.testing.assert_array_equal(r.surface_water_m, H - expected)
    np.testing.assert_array_equal(r.state.water_m[0], state.water_m[0])
    b = r.budget
    assert b.capacity_rejected_m3 == pytest.approx(float((H - expected).sum()) * AREA, rel=1e-15)
    assert b.attempted_m3 == float(H.sum()) * AREA
    assert abs(b.soil_residual_m3) <= b.bound_m3 and abs(b.surface_residual_m3) <= b.bound_m3
    # a cell already AT saturation with tolerance headroom is a valid input and rejects everything
    assert float(r.state.water_m[1, 0]) == 0.375 and float(r.surface_water_m[0]) == 0.01


def test_multi_layer_receivers_reject_per_layer_and_keep_the_rest_on_the_surface():
    col = column(layers(edges=(0.0, 0.25, 0.5, 1.0)))  # saturation depths 0.125, 0.125, 0.25
    op = operator(col, a3=1.0, max_fraction=1.0, background=(0.0, 1.0, 1.0))
    state = state_of(col, [[0.0, 0.125, 0.0], [0.0, 0.124, 0.25], [0.0, 0.0, 0.0], [0.0, 0.125, 0.25]])
    H = np.array([0.01, 0.01, 0.01, 0.01])
    r = pi.preferential_step(op, H, state, 1.0)
    room_1 = 0.125 - 0.124  # the FP64 room of cell 1 in layer 1 (not exactly 0.001)
    expected = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, room_1, 0.005, 0.0], [0.005, 0.0, 0.005, 0.0]])
    np.testing.assert_array_equal(r.intake_m, expected)
    np.testing.assert_array_equal(r.surface_water_m, H - expected.sum(axis=0))
    assert float(r.state.water_m[0, 2]) == 0.0  # the un-targeted top layer never receives the rejected share


def test_one_shared_source_across_species_and_receivers():
    col = column(layers(edges=(0.0, 0.25, 0.5, 1.0)))
    c_shrub, c_grass = np.array([1.0, 0.0, 2.0, 0.5]), np.array([0.5, 1.0, 0.0, 0.5])
    op = operator(col, a3=0.5, max_fraction=1.0, background=(1.0, 0.0, 0.0), species=[(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)],
                  contributions=[c_shrub, c_grass])
    H = np.array([0.01, 0.02, 0.03, 0.004])
    state = state_of(col, [[0.0, 0.0, 0.0]] * 4)
    r = pi.preferential_step(op, H, state, HOUR)
    C = c_shrub + c_grass
    beta = (0.5 + C) / (1.0 + C)
    np.testing.assert_allclose(r.intake_total_m, H * beta, rtol=1e-15)
    # the split follows g_l / sum g: background to layer 0, shrubs to layer 2, grass to layer 1
    g0, g1, g2 = 0.5 * np.ones(4), 0.5 / (1.0 + C) * c_grass, 0.5 / (1.0 + C) * c_shrub
    g = g0 + g1 + g2
    np.testing.assert_allclose(r.intake_m, H * np.array([g0, g1, g2]) / g * beta, rtol=1e-14)
    assert np.all(r.intake_total_m <= H)
    # an independent per-species application of the SAME water would remove more than the shared supply allows
    separate = H * ((0.5 + c_shrub) / (1.0 + c_shrub)) + H * ((0.5 + c_grass) / (1.0 + c_grass))
    assert np.all(separate[C > 0.0] > r.intake_total_m[C > 0.0])


def test_exact_two_layer_reference_baseline_one_hour():
    """The reference's hourly FL2 = min(H (a3 + C)/(1 + C) maxFL2, maxAmountFL2) with every vector [0, 1]."""
    col = column()
    a3, max_fraction, max_amount_mm_h = 0.5, 0.5, 3.0
    cover = np.array([0.0, 1.0, 3.0, 0.5])
    op = operator(col, a3=a3, max_fraction=max_fraction, max_rate=pi.mm_per_hour_to_m_s(max_amount_mm_h),
                  background=(0.0, 1.0), species=[(0.0, 1.0), (0.0, 0.0)], contributions=[cover, np.ones(4)])
    H = np.array([0.02, 0.02, 0.02, 0.002])
    state = state_of(col, [[0.03125, 0.09375]] * 4)
    r = pi.preferential_step(op, H, state, HOUR)
    shrub_function = (a3 + cover) / (1.0 + cover)
    fl2 = np.minimum(H * shrub_function * max_fraction, pi.mm_per_hour_to_m_s(max_amount_mm_h) * HOUR)
    np.testing.assert_allclose(r.intake_m[1], fl2, rtol=1e-15)
    np.testing.assert_array_equal(r.intake_m[0], 0.0)
    np.testing.assert_allclose(r.surface_water_m, H - fl2, rtol=1e-15)
    assert r.budget.rate_capped_m3 > 0.0  # the 3 mm/h cap binds on the cover-3 cell: 0.02 x 0.875 x 0.5 = 8.75 mm


def test_inactive_cells_are_identity_and_excluded_from_the_budget():
    col = column(active=[True, False, True, False])
    op = operator(col, a3=1.0, max_fraction=1.0)
    H = np.array([0.01, 0.02, 0.03, 0.04])
    state = state_of(col, [[0.03125, 0.09375]] * 4)
    r = pi.preferential_step(op, H, state, HOUR)
    np.testing.assert_array_equal(r.surface_water_m, [0.0, 0.02, 0.0, 0.04])
    np.testing.assert_array_equal(r.state.water_m[:, 1], state.water_m[:, 1])
    np.testing.assert_array_equal(r.intake_total_m, [0.01, 0.0, 0.03, 0.0])
    b = r.budget
    assert b.surface_before_m3 == 0.04 * AREA and b.intake_m3 == 0.04 * AREA and b.surface_after_m3 == 0.0
    assert b.soil_before_m3 == pytest.approx(2 * 0.125 * AREA, rel=1e-15)


@pytest.mark.parametrize("field,value,bit", [
    ("surface", math.nan, 3), ("surface", -1e-9, 4), ("surface", math.inf, 3),
    ("soil0", math.nan, 0), ("soil0", -1e-9, 1), ("soil1", 0.376, 2), ("soil1", math.inf, 0),
])
def test_invalid_forcing_is_flagged_with_inputs_untouched(field, value, bit):
    col = column()
    op = operator(col)
    H = np.array([0.01, 0.02, 0.03, 0.004])
    water = from_cells([[0.03125, 0.09375]] * 4)
    if field == "surface":
        H[2] = value
    else:
        water[int(field[-1]), 2] = value
    state = ss.column_state(col, water)
    before = (H.tobytes(), state.water_m.tobytes())
    with pytest.raises(pi.PreferentialStepError) as info:
        pi.preferential_step(op, H, state, HOUR)
    assert info.value.bit == bit and info.value.n_flagged == 1
    assert (H.tobytes(), state.water_m.tobytes()) == before
    # dt = 0 still validates the inputs
    with pytest.raises(pi.PreferentialStepError) as info0:
        pi.preferential_step(op, H, state, 0.0)
    assert info0.value.bit == bit


def test_invalid_forcing_on_inactive_cells_is_still_invalid():
    col = column(active=[True, True, False, True])
    op = operator(col)
    state = state_of(col, [[0.03125, 0.09375]] * 4)
    with pytest.raises(pi.PreferentialStepError) as info:
        pi.preferential_step(op, np.array([0.01, 0.02, math.nan, 0.004]), state, HOUR)
    assert info.value.bit == 3
    bad = from_cells([[0.03125, 0.09375]] * 4)
    bad[0, 2] = -1.0
    with pytest.raises(pi.PreferentialStepError) as info:
        pi.preferential_step(op, np.zeros(4), ss.column_state(col, bad), HOUR)
    assert info.value.bit == 1


@pytest.mark.parametrize("dt", [-1.0, math.nan, math.inf, True, "3600", None])
def test_invalid_dt_is_refused_before_any_work(dt):
    col = column()
    op = operator(col)
    H = np.array([0.01, 0.02, 0.03, 0.004])
    state = state_of(col, [[0.03125, 0.09375]] * 4)
    with pytest.raises(pi.SubsurfaceError):
        pi.preferential_step(op, H, state, dt)


def test_array_contract_refusals_and_immutability():
    col = column()
    op = operator(col)
    H = np.array([0.01, 0.02, 0.03, 0.004])
    state = state_of(col, [[0.03125, 0.09375]] * 4)

    class Fake(np.ndarray):
        pass

    for bad in (H.view(Fake), np.ma.masked_array(H), H.astype(np.float32), H[:3], H.reshape(1, 4), H[::2],
                np.ascontiguousarray(H)[None, :][0, :].reshape(2, 2), list(H), 0.01):
        with pytest.raises(pi.SubsurfaceError):
            pi.preferential_step(op, bad, state, HOUR)
    with pytest.raises(pi.SubsurfaceError):
        pi.preferential_step(op, H, ss.SoilColumnState(state.water_m.view(Fake), 0.0), HOUR)
    with pytest.raises(pi.SubsurfaceError):
        pi.preferential_step(op, H, ss.SoilColumnState(np.asfortranarray(state.water_m), 0.0), HOUR)
    with pytest.raises(pi.SubsurfaceError):
        pi.preferential_step(op, H, ss.SoilColumnState(state.water_m, math.nan), HOUR)
    with pytest.raises(pi.SubsurfaceError):
        pi.preferential_step(op, H, state.water_m, HOUR)
    with pytest.raises(pi.PreferentialError):
        pi.preferential_step(object(), H, state, HOUR)
    other = column(n_cells=5)
    with pytest.raises(pi.SubsurfaceError):
        pi.preferential_step(op, np.zeros(5), state_of(other, [[0.0, 0.0]] * 5), HOUR)
    # the caller's buffers are never written, even by a successful step
    before = (H.tobytes(), state.water_m.tobytes())
    r = pi.preferential_step(op, H, state, HOUR)
    assert (H.tobytes(), state.water_m.tobytes()) == before
    for name, a in arrays_of(r).items():
        assert type(a) is np.ndarray and a.flags.c_contiguous and np.all(np.isfinite(a)), name
        assert not np.shares_memory(a, H) and not np.shares_memory(a, state.water_m)
    assert not np.shares_memory(r.intake_m, r.state.water_m)


def test_random_closure_and_bounds_hold_over_a_mixed_population():
    from pref_cases import build_random

    for seed in range(4):
        col, op, water, surface, dt = build_random(100 + seed, 3, 257, "numpy")
        state = ss.column_state(col, water)
        for step_dt in (dt, 0.0, HOUR, 1e-3, 86400.0):
            r = pi.preferential_step(op, surface, state, step_dt)
            b = r.budget
            assert abs(b.soil_residual_m3) <= b.bound_m3 and abs(b.surface_residual_m3) <= b.bound_m3
            assert b.intake_m3 <= b.attempted_m3 + b.bound_m3 <= b.isolated_attempt_m3 + 2 * b.bound_m3
            assert np.all(r.surface_water_m >= 0.0) and np.all(r.state.water_m >= 0.0)
            assert np.all(r.state.water_m <= col.layers.saturation_m[:, None] + col.layers.capacity_tolerance_m[:, None])
            # the per-layer intake sum may exceed the surface debit by FP64 roundoff only (MAPLE cell coefficient)
            excess = r.intake_total_m - surface
            assert np.all(excess <= op.cell_closure_coefficient * (surface + r.intake_total_m))
            assert np.all(r.surface_water_m[r.intake_total_m > surface] == 0.0)
            active = col.active
            np.testing.assert_array_equal(r.surface_water_m[~active], surface[~active])
            np.testing.assert_array_equal(r.state.water_m[:, ~active], water[:, ~active])
            np.testing.assert_array_equal(r.intake_m[:, ~active], 0.0)
            surface, state = r.surface_water_m, r.state
