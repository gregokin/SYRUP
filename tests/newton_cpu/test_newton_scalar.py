"""Safeguarded Newton root of h + c k h^{3/2} = R: scalar/property tests against an independent high-precision
reference, bracket and fallback behaviour, degenerate inputs, and bitwise equality of the three forms (pure
Python specification, NumPy vectorized level form, compiled Numba form)."""
from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from maple_syrup import routing_newton as rn

NUMBA = pytest.mark.skipif(not pytest.importorskip("maple_syrup.routing_numba").numba_available(),
                           reason="Numba not installed; compiled form not exercised, no claim made")
EPS = float(np.finfo(np.float64).eps)


def reference_root(rhs: float, k: float, c: float) -> float:
    """Independent reference: 60-digit Decimal bisection on s = sqrt(h) of s^2 + (c k) s^3 = R (exact products of
    the double inputs); shares no code, bracket, derivative or operation order with the solver."""
    with localcontext() as ctx:
        ctx.prec = 60
        r, a = Decimal(rhs), Decimal(c) * Decimal(k)
        lo, hi = Decimal(0), r.sqrt()
        for _ in range(400):
            mid = (lo + hi) / 2
            if mid * mid + a * mid * mid * mid < r:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2
        return float(s * s)


def trial(x, k, c):
    return ((np.sqrt(x) * x) * k) * c + x


def solve(rhs, k, c, cap=50):
    return rn.newton_root_scalar(np.float64(rhs), np.float64(k), np.float64(c), cap)


def level(rhs, k, c, cap=50, small_level=0):
    """The VECTORIZED level form (small_level=0 disables the cell-by-cell path for narrow levels)."""
    return rn.newton_root_level(np.atleast_1d(np.asarray(rhs, dtype=np.float64)),
                                np.atleast_1d(np.asarray(k, dtype=np.float64)), c, cap, small_level)


def test_random_roots_match_the_independent_decimal_reference():
    rng = np.random.default_rng(20261003)
    worst = 0.0
    for _ in range(300):
        rhs, k, c = 10 ** rng.uniform(-12, 1), 10 ** rng.uniform(-3, 3.5), 10 ** rng.uniform(-3, 1)
        flow, passes, _steps, fallback = solve(rhs, k, c)
        ref = reference_root(rhs, k, c)
        worst = max(worst, abs(flow - ref) / ref)
        assert abs(flow - ref) <= 1e-13 * ref, (rhs, k, c, flow, ref)
        assert 1 <= passes <= 8 and fallback == 0
    assert worst < 1e-13


def test_the_bisection_invariant_and_water_identity_hold_without_clipping():
    rng = np.random.default_rng(7)
    for _ in range(2000):
        rhs, k, c = 10 ** rng.uniform(-14, 1), 10 ** rng.uniform(-3, 3.5), 10 ** rng.uniform(-3, 1)
        flow = solve(rhs, k, c)[0]
        assert 0.0 < flow <= rhs
        assert trial(flow, k, c) < rhs  # the property bisection guarantees, so h_new = R - c q >= flow >= 0
        q = (np.sqrt(flow) * flow) * k
        assert rhs - q * c >= flow * (1.0 - 1e-12)
        assert (rhs - q * c - flow) <= 1e-14 * rhs  # constitutive residual far below root_tolerance_m


WIDE_R = (5e-324, 1e-300, 1e-150, 1e-30, 1e-8, 1.0, 1e3, 1e8)
WIDE_K = (0.0, 1e-300, 1e-12, 1e-3, 1.0, 1e3, 1e9, 1e12)
WIDE_C = (0.0, 1e-9, 0.5, 1e3)


@pytest.mark.parametrize("rhs", WIDE_R)
def test_wide_dynamic_range_matches_the_reference_and_keeps_the_invariants(rhs):
    for k in WIDE_K:
        for c in WIDE_C:
            flow, passes, steps, fallback = solve(rhs, k, c)
            assert math.isfinite(flow) and 0.0 <= flow <= rhs
            if rhs > 0.0 and trial(rhs, k, c) > rhs:
                assert trial(flow, k, c) < rhs, (rhs, k, c)
            ref = reference_root(rhs, k, c)
            assert abs(flow - ref) <= 1e-13 * ref + 5e-324, (rhs, k, c, flow, ref, passes, steps, fallback)
            assert passes <= 50


def test_zero_conveyance_pit_dt_zero_and_negligible_flux_are_analytic():
    for rhs in (1e-3, 0.7, 5e-324):
        assert solve(rhs, 0.0, 0.5) == (rhs, 0, 0, 0)  # pit / zero conveyance: h_flow = R, no iteration
        assert solve(rhs, 12.0, 0.0) == (rhs, 0, 0, 0)  # dt = 0 (c = 0): identity
    assert solve(1e-300, 30.0, 0.5)[1:] == (0, 0, 0)  # the flux term underflows: h_flow = R


@pytest.mark.parametrize("rhs", [0.0, -0.0, -1e-3, -np.inf, np.nan])
def test_non_positive_or_nan_rhs_leaves_zero_exactly_like_the_bisection(rhs):
    flow, passes, steps, fallback = solve(rhs, 3.0, 0.5)
    assert flow == 0.0 and not math.copysign(1.0, flow) < 0 and (passes, steps, fallback) == (0, 0, 0)
    f, p, s, fb = level(rhs, 3.0, 0.5)
    assert f[0] == 0.0 and p[0] == s[0] == 0 and not fb[0]


def test_infinite_rhs_is_left_to_the_shared_overflow_check():
    flow = solve(np.inf, 3.0, 0.5)[0]
    assert flow == np.inf  # the step's own `right-hand side overflowed` check rejects it; nothing is invented here


def test_small_iteration_cap_engages_the_bracketed_bisection_fallback_and_stays_correct():
    rng = np.random.default_rng(11)
    for _ in range(60):
        rhs, k, c = 10 ** rng.uniform(-8, 0), 10 ** rng.uniform(0, 3), 0.5
        flow, passes, steps, fallback = solve(rhs, k, c, cap=1)
        assert passes == 1 and fallback == 1 and steps > 0
        ref = reference_root(rhs, k, c)
        assert abs(flow - ref) <= 1e-13 * ref
        assert trial(flow, k, c) < rhs
        full = solve(rhs, k, c, cap=50)
        assert full[3] == 0 and abs(full[0] - ref) <= 1e-13 * ref


def test_typical_realistic_inputs_need_few_passes_and_no_safeguard():
    rng = np.random.default_rng(5)
    rhs = 10 ** rng.uniform(-6, -0.5, 5000)
    k = 10 ** rng.uniform(0, 2, 5000)
    flow, passes, steps, fallback = level(rhs, k, 0.5)
    assert passes.max() <= 6 and not fallback.any() and steps.sum() == 0
    assert np.all(trial(flow, k, 0.5) < rhs)


def test_root_is_monotone_in_the_right_hand_side_up_to_rounding():
    rhs = np.geomspace(1e-9, 1.0, 2000)
    k = np.full(rhs.size, 40.0)
    flow = level(rhs, k, 0.5)[0]
    assert np.all(np.diff(flow) > -1e-13 * flow[1:])


def test_numpy_level_form_equals_the_python_specification_bitwise():
    rng = np.random.default_rng(3)
    n = 3000
    rhs, k = 10 ** rng.uniform(-14, 1, n), 10 ** rng.uniform(-3, 3.5, n)
    rhs[::53], k[::37] = 0.0, 0.0
    rhs[7::101] = -1.0
    for cap in (1, 2, 50):
        flow, passes, steps, fallback = level(rhs, k, 0.375, cap)
        for i in range(n):
            f, p, s, fb = solve(rhs[i], k[i], 0.375, cap)
            assert (flow[i], passes[i], steps[i], int(fallback[i])) == (f, p, s, fb), (cap, i)


@NUMBA
def test_numba_scalar_form_equals_the_python_specification_bitwise():
    root = rn.compiled_root()
    rng = np.random.default_rng(4)
    n = 3000
    rhs, k = 10 ** rng.uniform(-14, 1, n), 10 ** rng.uniform(-3, 3.5, n)
    rhs[::53], k[::37] = 0.0, 0.0
    for cap in (1, 3, 50):
        for i in range(n):
            assert tuple(root(rhs[i], k[i], 0.375, cap)) == tuple(solve(rhs[i], k[i], 0.375, cap)), (cap, i)


@NUMBA
def test_compiled_sweep_has_no_fastmath_and_stats_names_are_stable():
    sweep = rn.compiled_sweep_newton()
    assert not sweep.targetoptions.get("fastmath", False) and not sweep.targetoptions.get("parallel", False)
    assert rn.STAT_NAMES == ("max_newton_iterations", "total_newton_iterations", "bisection_safeguard_steps",
                             "fallback_cells", "iterated_cells")
    assert rn.stats_dict(np.arange(5)) == dict(zip(rn.STAT_NAMES, range(5), strict=True))


def test_default_level_form_is_fully_vectorized_and_the_optional_small_level_helper_has_parity(monkeypatch):
    rng = np.random.default_rng(12)
    rhs, k = 10 ** rng.uniform(-12, 0, 5), 10 ** rng.uniform(-2, 3, 5)
    rhs[0], k[-1] = 0.0, 0.0
    with monkeypatch.context() as m:  # the default must never reach the Python cell-by-cell specification
        m.setattr(rn, "newton_root_scalar", lambda *a, **kw: pytest.fail("default level form looped over cells"))
        rn.newton_root_level(rhs, k, 0.5, 50)
    for width in (1, 2, 5, rn.SMALL_LEVEL_CELLS, rn.SMALL_LEVEL_CELLS + 1, 200):
        rhs, k = 10 ** rng.uniform(-12, 0, width), 10 ** rng.uniform(-2, 3, width)
        rhs[0], k[-1] = 0.0, 0.0
        for cap in (1, 50):
            vector = rn.newton_root_level(rhs, k, 0.5, cap)
            helper = rn.newton_root_level(rhs, k, 0.5, cap, rn.SMALL_LEVEL_CELLS)
            for a, b in zip(vector, helper, strict=True):
                np.testing.assert_array_equal(a, b)


def test_start_value_is_an_upper_bound_but_its_ratio_to_the_root_is_unbounded():
    ratios = []
    for b in (1e-3, 1.0, 1e3, 1e6, 1e9):  # b = a sqrt(R)
        rhs, c, k = 1e-2, 0.5, 2.0 * b / (0.5 * 0.1)
        a = c * k
        x0 = rhs / (1.0 + a * math.sqrt(rhs / (1.0 + a * math.sqrt(rhs))))
        root = reference_root(rhs, k, c)
        assert root <= x0 * (1 + 1e-12) and x0 <= rhs
        ratios.append(x0 / root)
        flow = solve(rhs, k, c)[0]  # the iteration cap / fallback still deliver the root
        assert abs(flow - root) <= 1e-13 * root
    assert ratios[-1] > 2.0 and ratios == sorted(ratios)
