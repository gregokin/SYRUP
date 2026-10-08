"""Scalar-batch CUDA Newton roots (`routing_newton_cuda.solve_roots`) against the CPU pure-Python specification
`newton_root_scalar` on IDENTICAL inputs (bitwise, including the pass/step/fallback counters), the Numba root when
installed, an independent 60-digit `decimal` reference, extreme/subnormal inputs, k = 0 pits and forced low-cap
fallbacks. Needs a device (explicit skip otherwise)."""
from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pytest
from gn_helpers import SPECIAL

pytest.importorskip("maple")

from maple_syrup import routing_newton as rn
from maple_syrup import routing_newton_cuda as rnc
from maple_syrup.routing import RoutingError

pytestmark = pytest.mark.usefixtures("gpu")
EPS = float(np.finfo(np.float64).eps)


def cupy():
    import cupy as cp

    return cp


def device_roots(rhs, k, c, cap=50):
    cp = cupy()
    flow, passes, steps, fb = rnc.solve_roots(cp.asarray(np.ascontiguousarray(rhs, dtype=np.float64)),
                                              cp.asarray(np.ascontiguousarray(k, dtype=np.float64)), c, cap)
    return flow.get(), passes.get(), steps.get(), fb.get()


def cpu_roots(rhs, k, c, cap=50):
    out = [rn.newton_root_scalar(float(r), float(kk), float(c), cap) for r, kk in zip(rhs, k, strict=True)]
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out], dtype=np.int64),
            np.array([o[2] for o in out], dtype=np.int64), np.array([o[3] for o in out], dtype=np.uint8))


def assert_same(rhs, k, c, cap=50):
    with np.errstate(all="ignore"):
        cpu = cpu_roots(rhs, k, c, cap)
    dev = device_roots(rhs, k, c, cap)
    both_nan = np.isnan(cpu[0]) & np.isnan(dev[0])
    bad = (cpu[0].view(np.uint64) != dev[0].view(np.uint64)) & ~both_nan
    assert not bad.any(), (f"{int(bad.sum())} root bit mismatches, first rhs={rhs[np.flatnonzero(bad)[0]]!r} "
                           f"k={k[np.flatnonzero(bad)[0]]!r}")
    for name, a, b in zip(("passes", "bisection_steps", "fallback"), cpu[1:], dev[1:], strict=True):
        np.testing.assert_array_equal(b, a, err_msg=name)
    return cpu


@pytest.mark.parametrize("cap", [50, 1000, 8])
@pytest.mark.parametrize("c", [0.5 / 3.0, 1e-3, 40.0, 1e-300])
def test_log_uniform_batch_is_bitwise_the_cpu_root(c, cap):
    rng = np.random.default_rng(int(1e6 * c) % 2**31 + cap)
    n = 3000
    rhs = 10.0 ** rng.uniform(-14.0, 2.0, n)
    k = 10.0 ** rng.uniform(-4.0, 5.0, n)
    assert_same(rhs, k, c, cap)


def test_special_values_cross_product_is_bitwise_the_cpu_root():
    special = np.array(SPECIAL + [1.0, 7.5e-6, 0.31, 1e-320, 4.9e-324 * 3])
    rhs = np.repeat(special, special.size)
    k = np.tile(np.abs(special), special.size)
    k = np.where(np.isnan(k), 0.0, k)  # the graph validator keeps conveyance finite and >= 0; zero is a pit
    for c in (0.5 / 3.0, 1e-300, 1e3):
        assert_same(rhs, k, c)


def test_pits_and_tiny_depths_return_the_analytic_root_without_iteration():
    rhs = np.array([0.0, -1.0, 1e-3, 5e-324, 1e-310, 2.0, 1e-8])
    k = np.array([5.0, 5.0, 0.0, 3.0, 1e6, 0.0, 5e-324])
    flow, passes, _steps, fb = assert_same(rhs, k, 0.4)
    assert flow[0] == 0.0 and flow[1] == 0.0
    assert flow[2] == 1e-3 and flow[5] == 2.0 and passes[2] == 0 and passes[5] == 0  # k = 0: h = R, nothing lost
    assert flow[3] == 5e-324 and passes[3] == 0
    assert not fb.any()


def test_the_default_cap_solves_the_physical_range_without_fallback_in_few_passes():
    rng = np.random.default_rng(5)
    rhs = 10.0 ** rng.uniform(-6.0, 0.5, 5000)
    k = rng.uniform(1.0, 60.0, 5000)
    _flow, passes, _steps, fb = assert_same(rhs, k, 0.5 / 3.0)
    assert not fb.any() and passes.max() <= 12 and (passes > 0).all()


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_a_low_cap_forces_the_bracketed_fallback_and_stays_exact_and_safe(cap):
    rng = np.random.default_rng(6)
    rhs = 10.0 ** rng.uniform(-5.0, 0.3, 2000)
    k = rng.uniform(5.0, 200.0, 2000)
    c = 0.5 / 3.0
    flow, passes, _steps, fb = assert_same(rhs, k, c, cap)
    assert fb.sum() > 0 and passes.max() <= cap  # the fallback was really exercised
    # the bisection invariant holds on every cell: trial(h_flow) < R, so h_new = R - c q >= h_flow >= 0 unclipped
    with np.errstate(all="ignore"):
        t = (np.sqrt(flow) * flow * k * c) + flow
    pos = rhs > 0
    assert (t[pos] < rhs[pos]).all() and (flow >= 0).all()


def test_numba_root_is_the_same_function_when_available():
    pytest.importorskip("numba")
    rng = np.random.default_rng(7)
    rhs = 10.0 ** rng.uniform(-10.0, 1.0, 1500)
    k = 10.0 ** rng.uniform(-2.0, 4.0, 1500)
    dev = device_roots(rhs, k, 0.3)
    root = rn.compiled_root()
    ref = [root(float(r), float(kk), 0.3, 50) for r, kk in zip(rhs, k, strict=True)]
    np.testing.assert_array_equal(dev[0].view(np.uint64), np.array([x[0] for x in ref]).view(np.uint64))
    np.testing.assert_array_equal(dev[1], [x[1] for x in ref])
    np.testing.assert_array_equal(dev[2], [x[2] for x in ref])


def _decimal_root(rhs: float, k: float, c: float) -> Decimal:
    """Independent root of h + (c k) h sqrt(h) = rhs in 60-digit decimal arithmetic (bisection on a bracket)."""
    getcontext().prec = 60
    r, a = Decimal(rhs), Decimal(c) * Decimal(k)
    lo, hi = Decimal(0), r
    for _ in range(400):
        mid = (lo + hi) / 2
        if mid + a * mid * mid.sqrt() < r:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def test_roots_agree_with_an_independent_high_precision_reference():
    rng = np.random.default_rng(8)
    rhs = 10.0 ** rng.uniform(-9.0, 0.5, 150)
    k = 10.0 ** rng.uniform(-1.0, 3.0, 150)
    c = 0.5 / 3.0
    flow, *_ = device_roots(rhs, k, c)
    for r, kk, f in zip(rhs, k, flow, strict=True):
        ref = _decimal_root(float(r), float(kk), c)
        # the finalised root lies within ~64 eps of the exact root and below it (trial < R side)
        assert abs(Decimal(float(f)) - ref) <= Decimal(80.0 * EPS * float(r)), (r, kk, f, ref)


def test_root_equation_residual_is_far_below_the_unchanged_tolerance():
    rng = np.random.default_rng(9)
    rhs = 10.0 ** rng.uniform(-8.0, 0.5, 4000)
    k = 10.0 ** rng.uniform(-1.0, 3.0, 4000)
    c = 0.5 / 3.0
    flow = device_roots(rhs, k, c)[0]
    h_new = rhs - ((np.sqrt(flow) * flow) * k) * c  # the storage identity of the sweep
    assert (np.abs(h_new - flow) <= 1e-12).all() and (h_new >= flow).all() and (flow >= 0).all()


@pytest.mark.parametrize("bad", [0, -1, 1001, True, 2.5, "9", None])
def test_invalid_caps_are_refused_before_any_launch(bad):
    cp = cupy()
    x = cp.asarray(np.ones(4))
    with pytest.raises(RoutingError, match="newton_max_iterations"):
        rnc.solve_roots(x, x, 0.3, bad)


@pytest.mark.parametrize("bad_c", [0.0, -1.0, float("nan"), float("inf"), True, "1"])
def test_invalid_c_is_refused(bad_c):
    cp = cupy()
    x = cp.asarray(np.ones(4))
    with pytest.raises(RoutingError, match="c must"):
        rnc.solve_roots(x, x, bad_c, 50)


def test_non_device_inputs_are_refused_without_conversion():
    cp = cupy()
    with pytest.raises(RoutingError, match="cupy"):
        rnc.solve_roots(np.ones(4), cp.ones(4), 0.3)
    with pytest.raises(RoutingError):
        rnc.solve_roots(cp.ones(4), cp.ones(5), 0.3)
    with pytest.raises(RoutingError):
        rnc.solve_roots(cp.ones(4, dtype=np.float32), cp.ones(4), 0.3)
    with pytest.raises(RoutingError):
        rnc.solve_roots(cp.ones(8)[::2], cp.ones(4), 0.3)


def test_the_empty_batch_launches_nothing():
    cp = cupy()
    flow, passes, steps, fb = rnc.solve_roots(cp.empty(0), cp.empty(0), 0.3)
    assert flow.shape == passes.shape == steps.shape == fb.shape == (0,)
