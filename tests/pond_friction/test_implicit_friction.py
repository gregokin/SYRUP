"""Opt-in implicit quadratic (Darcy-Weisbach) face friction of the local-inertial solvers: NumPy / Numba / CUDA tests. Nothing here
was run by its author (file-only tools); Codex records commands and results. Run explicitly (`pytest tests/pond_friction`).

Conventions: references are 80-digit `decimal` evaluations of the friction equation from the exact FP64 inputs, hand-built stable
cases and the verbatim pre-change face function (`friction_cases`), never a mirror of the helper under test. Declared FP64
roundoff only: root relative error `ROOT_REL` (32 eps), whole-equation residual `RESIDUAL_REL |q*|` (64 eps); no physical
tolerance is widened. CUDA tests skip (with a stated reason) only when no CuPy/device is available.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest
from friction_cases import (
    RESIDUAL_REL,
    ROOT_REL,
    col_params,
    exact_residual,
    exact_root,
    face,
    original_face_flux,
    rel_err,
    topo,
)

from maple_syrup.experimental_hydrology import (
    FRICTION_TREATMENTS,
    CpuHydraulicSolver,
    ExperimentalHydrologyError,
    HydraulicControl,
    HydraulicStepRejected,
    implicit_quadratic_friction_flux,
    local_inertial_face_flux,
)
from maple_syrup.experimental_storm import ExperimentalControl
from maple_syrup.pond_hydraulics import (
    POND_DEFAULT_FRICTION_TREATMENT,
    PondHydraulicsError,
    build_pond_hydraulics,
    evolve_pond,
    face_speed_diagnostics,
    metadata_routing_graph,
    rebind_pond,
    step_once,
    uniform_rain_rate,
)
from maple_syrup.pond_hydraulics import _to_host as H
from maple_syrup.routing import GRAVITY_M_S2 as G

IMPLICIT = HydraulicControl(friction_treatment="implicit_quadratic")
RTOL, ATOL = 2e-12, 1e-14  # the repository's existing backend field-agreement bounds (not widened)


@pytest.fixture(scope="session")
def gpu():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA friction treatment is not exercised")
    return backend.cupy_module()


@pytest.fixture(scope="session")
def numba_kernels():
    pytest.importorskip("numba")
    from maple_syrup import experimental_numba as en

    return en._kernels()


def make_pond(topology, backend="numpy", *, control=None, ksat=0.0):
    xp = np
    if backend == "cuda":
        from maple.core.backend import cupy_module

        xp = cupy_module()
    return build_pond_hydraulics(topology, col_params(np.asarray(topology.active), xp, ksat=ksat), backend=backend,
                                 control=control)


def state_of(pond, depth):
    xp = pond.xp
    return pond.initial_state(xp.asarray(np.array(depth, dtype=np.float64)), pond.params.storage_max_m * 0.0)


# --- control field, validation, describe, defaults --------------------------------------------------------------------------
def test_control_default_is_lagged_and_the_old_positional_form_is_unchanged():
    assert FRICTION_TREATMENTS == ("lagged", "implicit_quadratic")
    assert HydraulicControl().friction_treatment == "lagged"
    old = HydraulicControl(0.25, "donor")  # the original two-field positional constructor
    assert (old.cfl_max, old.limiter, old.friction_treatment) == (0.25, "donor", "lagged")
    assert HydraulicControl().validated() == HydraulicControl(0.5, "off", "lagged")
    assert IMPLICIT.validated() is IMPLICIT


@pytest.mark.parametrize("bad", ["implicit", "", "Lagged", None, 1, ["lagged"], b"lagged"])
def test_invalid_friction_treatment_is_refused_before_any_kernel(bad):
    with pytest.raises(ExperimentalHydrologyError, match="friction_treatment"):
        HydraulicControl(friction_treatment=bad).validated()
    topology = topo(np.zeros((2, 3)), [(0, 2, 0, 1)])
    with pytest.raises(PondHydraulicsError, match="friction_treatment"):
        build_pond_hydraulics(topology, col_params(np.asarray(topology.active)), control=HydraulicControl(friction_treatment=bad))
    with pytest.raises(ExperimentalHydrologyError, match="friction_treatment"):
        local_inertial_face_flux(np.array([1], dtype=np.int8), np.zeros(1), np.ones(1), np.zeros(1), np.zeros(1, dtype=np.int8),
                                 np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), 0.1, 1.0, bad)


def test_describe_reports_the_mode_on_the_solver_and_the_pond():
    topology = topo(np.zeros((2, 3)), [(0, 2, 0, 1)])
    for treatment in FRICTION_TREATMENTS:
        pond = make_pond(topology, control=HydraulicControl(friction_treatment=treatment))
        info = pond.describe()
        assert info["friction_treatment"] == treatment
        assert info["solver"]["control"]["friction_treatment"] == treatment


def test_explicit_method_refuses_the_implicit_friction_option():
    topology = topo(np.zeros((2, 3)), [(0, 2, 0, 1)])
    graph = metadata_routing_graph(topology)  # the refusal happens in the constructor, before any explicit-method setup
    params = col_params(np.asarray(topology.active))
    with pytest.raises(ExperimentalHydrologyError, match="friction_treatment"):
        CpuHydraulicSolver("explicit", graph, params, control=IMPLICIT)


def test_pond_default_is_implicit_only_when_no_control_is_given():
    topology = topo(np.zeros((2, 3)), [(0, 2, 0, 1)])
    assert POND_DEFAULT_FRICTION_TREATMENT == "implicit_quadratic"
    assert make_pond(topology, control=None).control.friction_treatment == "implicit_quadratic"
    assert make_pond(topology, control=HydraulicControl()).control.friction_treatment == "lagged"  # explicit control retained
    assert make_pond(topology, control=IMPLICIT).control.friction_treatment == "implicit_quadratic"
    assert make_pond(topology, control=HydraulicControl(0.25)).control.friction_treatment == "lagged"
    assert make_pond(topology, control=None).solver.control.friction_treatment == "implicit_quadratic"
    with pytest.raises(PondHydraulicsError, match="donor"):  # the donor-limiter-OFF pond requirement is unchanged
        make_pond(topology, control=HydraulicControl(limiter="donor", friction_treatment="implicit_quadratic"))


def test_rebind_keeps_the_selected_treatment():
    topology = topo(np.zeros((2, 3)), [(0, 2, 0, 1)])
    for control in (HydraulicControl(), IMPLICIT, None):
        pond = make_pond(topology, control=control)
        state = state_of(pond, np.full((2, 3), 0.01))
        new, _same = rebind_pond(pond, topology, state)
        assert new.control.friction_treatment == pond.control.friction_treatment


# --- old default: bit-exact to the verbatim original formula ------------------------------------------------------------------
def random_faces(rng, n):
    z_a, z_b = rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)
    h_a = np.where(rng.random(n) < 0.2, 0.0, rng.uniform(0.0, 0.05, n) ** 2)
    h_b = np.where(rng.random(n) < 0.2, 0.0, rng.uniform(0.0, 0.05, n) ** 2)
    q_old = np.where(rng.random(n) < 0.3, 0.0, rng.normal(0.0, 1e-3, n))
    return dict(
        ftype=rng.integers(0, 3, n).astype(np.int8), zmax=np.maximum(z_a, z_b), fric=rng.uniform(0.1, 30.0, n),
        kb=rng.uniform(0.0, 2.0, n), sign=rng.choice(np.array([-1, 1], dtype=np.int8), n), q_old=q_old, h_a=h_a, h_b=h_b,
        eta_a=z_a + h_a, eta_b=z_b + h_b)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("how", ["omitted", "lagged"])
def test_lagged_is_bit_exact_to_the_original_formula(seed, how):
    rng = np.random.default_rng(seed)
    f = random_faces(rng, 4000)
    dt, dx = float(rng.choice([0.01, 0.05, 0.25])), float(rng.choice([0.5, 1.0, 2.0]))
    expected_q, expected_h = original_face_flux(**f, dt=dt, dx=dx)
    extra = () if how == "omitted" else ("lagged",)
    q, hf = local_inertial_face_flux(*f.values(), dt, dx, *extra)
    assert np.array_equal(q, expected_q) and np.array_equal(hf, expected_h)
    assert np.array_equal(np.signbit(q), np.signbit(expected_q))


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_old_default_pond_step_is_unchanged_by_an_omitted_or_explicit_lagged_field(backend):
    if backend == "numba":
        pytest.importorskip("numba")
    rng = np.random.default_rng(7)
    bed = rng.uniform(0.0, 0.2, (4, 5))
    depth = rng.uniform(0.0, 0.03, (4, 5))
    topology = topo(bed, [(1, 4, 0, 1)], drop=0.1)
    a = make_pond(topology, backend, control=HydraulicControl())
    b = make_pond(topology, backend, control=HydraulicControl(friction_treatment="lagged"))
    sa, sb = state_of(a, depth), state_of(b, depth)
    rain = uniform_rain_rate(a, 2e-5)
    for _ in range(8):
        ra, rb = step_once(a, rain, sa, 0.01), step_once(b, rain, sb, 0.01)
        for key in ("x", "y"):
            assert np.array_equal(ra.step.used_flux_m2_s[key], rb.step.used_flux_m2_s[key])
        assert np.array_equal(ra.state.depth_m, rb.state.depth_m)
        sa, sb = ra.state, rb.state


# --- the algebra of the root ----------------------------------------------------------------------------------------------
def test_root_matches_an_80_digit_solution_and_the_whole_equation_residual_is_fp64_roundoff():
    rng = np.random.default_rng(11)
    n = 400
    h = 10.0 ** rng.uniform(-12.0, 0.5, n)
    qs = rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(-16.0, 2.0, n)
    fric = 10.0 ** rng.uniform(-1.0, 2.0, n)
    dt = 10.0 ** rng.uniform(-3.0, 3.0, n)
    for i in range(n):
        got = float(implicit_quadratic_friction_flux(qs[i:i + 1], h[i:i + 1], fric[i:i + 1], float(dt[i]))[0])
        ref = exact_root(qs[i], h[i], fric[i], dt[i])
        assert math.isfinite(got) and rel_err(got, ref) <= ROOT_REL, (i, got, ref)
        assert abs(exact_residual(got, qs[i], h[i], fric[i], dt[i])) <= Decimal(RESIDUAL_REL) * abs(Decimal(float(qs[i])))


@pytest.mark.parametrize("h", [1e-30, 1e-100, 1e-150, 1e-200])
def test_physically_thin_depths_down_to_1e_200_are_exact_where_the_flux_is_representable(h):
    # A: z = 0 with depth h, B: dry at z = -0.1 -> hf = h exactly, grad = -0.1 / dx, q_old = 0 (a first wetting face).
    dt, fric = 0.25, 21.45
    q, hf = face(0.0, 0.0, -0.1, h, 0.0, fric, dt, "implicit_quadratic")
    assert hf == h
    q_star = ((G * h) * dt) * 0.1  # the same first-wetting impulse the lagged form would use
    assert q > 0.0 and math.isfinite(q) and q <= q_star
    assert rel_err(q, exact_root(q_star, h, fric, dt)) <= 4 * ROOT_REL  # q* itself carries ~3 roundings
    assert q > 5e-324  # representable, not flushed
    for depth in (h, 3.0 * h):  # no overflow/underflow-induced inf or nan anywhere in the neighbourhood
        out, _ = face(0.0, 0.0, -0.1, depth, 0.0, fric, dt, "implicit_quadratic")
        assert math.isfinite(out) and out > 0.0


def flux1(qs, h, fr, dt):
    return float(implicit_quadratic_friction_flux(np.array([qs]), np.array([h]), np.array([fr]), dt)[0])


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_huge_qstar_with_a_thin_film_is_not_flushed_to_zero(sign):
    # C1 defect: t = h / D underflows (1e-200 / 1e150) before the product with q* = 1e300; the root is still representable.
    got = flux1(sign * 1e300, 1e-200, 21.45, 0.25)
    assert got != 0.0 and math.copysign(1.0, got) == sign
    assert rel_err(got, exact_root(sign * 1e300, 1e-200, 21.45, 0.25)) <= ROOT_REL
    assert got == pytest.approx(sign * 1.2214096331049185e-50, rel=1e-14)  # root-recorded 100-digit value
    small = flux1(sign * 1e-200, 1e-200, 21.45, 0.25)  # the symmetric tiny case keeps its precision
    assert small == pytest.approx(sign * 1.2214e-300, rel=1e-4)
    assert rel_err(small, exact_root(sign * 1e-200, 1e-200, 21.45, 0.25)) <= ROOT_REL


def test_wide_qstar_h_grid_matches_the_80_digit_root_wherever_the_root_is_a_normal_number():
    checked = 0
    for qe in range(-200, 301, 50):
        for he in range(-200, 1, 40):
            for fr, dt in ((21.45, 0.25), (0.1, 1e-3), (30.0, 100.0)):
                for sign in (1.0, -1.0):
                    qs, h = sign * 10.0 ** qe, 10.0 ** he
                    ref = exact_root(qs, h, fr, dt)
                    got = flux1(qs, h, fr, dt)
                    if abs(ref) < Decimal("1e-290"):  # not a normal FP64 number: gradual underflow is allowed there
                        assert math.isfinite(got) and abs(got) <= abs(qs)
                        continue
                    assert rel_err(got, ref) <= ROOT_REL, (qs, h, fr, dt, got, ref)
                    checked += 1
    assert checked > 200


def test_special_values_alpha_zero_qstar_zero_dry_and_sign_flip():
    f = lambda qs, h=0.01, fr=20.0, dt=0.1: float(implicit_quadratic_friction_flux(np.array([qs]), np.array([h]), np.array([fr]), dt)[0])  # noqa: E731
    assert f(0.0) == 0.0 and not math.copysign(1.0, f(0.0)) < 0.0
    assert math.copysign(1.0, f(-0.0)) < 0.0 and f(-0.0) == 0.0  # q* = -0 passes through unchanged
    assert f(0.123, fr=0.0) == 0.123  # alpha = 0 -> q = q* exactly
    assert f(0.123, dt=0.0) == 0.123
    for qs in (1e-9, 3e-4, 1.0, 40.0):  # exact odd symmetry: flipping the sign of q* flips q bitwise
        assert f(-qs) == -f(qs)
    # a dry face carries nothing even with an old momentum and an implicit treatment
    q, hf = face(0.5, 2.0, 2.0, 0.0, 0.0, 20.0, 0.1, "implicit_quadratic")
    assert q == 0.0 and hf == 0.0
    # no friction coefficient needed to be tiny: huge alpha stays finite, small and correctly signed
    big = f(1.0, h=1e-8, fr=30.0, dt=100.0)
    assert 0.0 < big < 1e-3 and rel_err(big, exact_root(1.0, 1e-8, 30.0, 100.0)) <= ROOT_REL


def test_overflow_and_non_finite_inputs_are_refused_as_nan_never_silently_zero():
    nan = np.nan
    for qs, h, fr, dt in [(np.inf, 0.01, 20.0, 0.1), (-np.inf, 0.01, 20.0, 0.1), (nan, 0.01, 20.0, 0.1),
                          (1.0, 0.01, 1e300, 1e10),    # dt f overflows
                          (1e300, 1e-3, 1e300, 1e10)]:  # D overflows
        out = implicit_quadratic_friction_flux(np.array([qs]), np.array([h]), np.array([fr]), dt)
        assert np.isnan(out[0]), (qs, h, fr, dt, out)
    # the largest FP64 flux with moderate friction is still finite and bounded by |q*|
    out = implicit_quadratic_friction_flux(np.array([1.7e308]), np.array([1.0]), np.array([20.0]), 0.1)[0]
    assert math.isfinite(out) and 0.0 < out < 1.7e308


def test_numba_scalar_function_is_bitwise_the_numpy_helper_including_extremes(numba_kernels):
    rng = np.random.default_rng(3)
    cases = [(0.0, 0.01, 20.0, 0.1), (-0.0, 0.01, 20.0, 0.1), (np.nan, 0.01, 20.0, 0.1), (np.inf, 0.01, 20.0, 0.1),
             (1.0, 0.01, 1e300, 1e10), (1e300, 1e-3, 1e300, 1e10), (2.4e-201, 1e-200, 21.45, 0.25),
             (5.8e-8, 2.16014956755e-7, 21.45, 0.25), (0.3, 0.01, 0.0, 0.1), (1.7e308, 1.0, 20.0, 0.1),
             (1e300, 1e-200, 21.45, 0.25), (-1e300, 1e-200, 21.45, 0.25), (1e-200, 1e-200, 21.45, 0.25),
             (1e300, 1e-320, 1e-300, 1.0), (1.7e308, 1e-200, 1e-300, 1e-10)]  # last two: q*/D overflow -> NaN refusal
    for _ in range(500):
        cases.append((rng.choice([-1, 1]) * 10.0 ** rng.uniform(-20, 3), 10.0 ** rng.uniform(-30, 1),
                      10.0 ** rng.uniform(-1, 2), 10.0 ** rng.uniform(-3, 3)))
    for qs, h, fr, dt in cases:
        want = float(implicit_quadratic_friction_flux(np.array([qs]), np.array([h]), np.array([fr]), dt)[0])
        got = float(numba_kernels.implicit_flux(float(qs), float(h), float(fr), float(dt)))
        assert (math.isnan(got) and math.isnan(want)) or (got == want and math.copysign(1.0, got) == math.copysign(1.0, want))


# --- physics of one face ---------------------------------------------------------------------------------------------------
def test_first_wetting_thin_film_is_damped_and_the_lagged_artifact_is_reproduced():
    # root-measured scalars: h = 2.16014956755e-7 m, grad = 0.11, f = 21.45, dt = 0.25, q_old = 0, dx = 1 (A higher than B).
    h0, slope, fric, dt = 2.16014956755e-7, 0.11, 21.45, 0.25
    lag, hf = face(0.0, slope, 0.0, h0, 0.0, fric, dt)
    imp, hf2 = face(0.0, slope, 0.0, h0, 0.0, fric, dt, "implicit_quadratic")
    assert hf == hf2 and hf == pytest.approx(h0, rel=1e-9)
    fr = lambda q: abs(q / hf) / math.sqrt(G * hf)  # noqa: E731
    assert 180.0 < fr(lag) < 190.0  # the lagged first flux is the frictionless impulse g S dt (independent of h)
    grad = (0.0 - (slope + h0)) / 1.0
    q_star = -((G * hf) * dt) * grad
    assert rel_err(imp, exact_root(q_star, hf, fric, dt)) <= 4 * ROOT_REL
    q_eq = math.sqrt(8.0 * G * abs(grad) / fric) * hf ** 1.5
    fr_eq = math.sqrt(8.0 * abs(grad) / fric)
    assert 0.0 < imp < q_eq and imp > 0.99 * q_eq  # from rest the flux approaches the equilibrium from below
    assert 0.19 < fr(imp) < fr_eq  # transient bound: Fr stays below the equilibrium Froude number sqrt(8 S / f)


def test_friction_only_decay_follows_the_ode_and_converges_first_order():
    h, fric, q0, total = 0.01, 21.45, 1e-3, 0.05
    a = fric / (8.0 * h * h)
    exact = q0 / (1.0 + a * q0 * total)

    def run(n):
        q, prev = q0, q0
        for _ in range(n):
            q, _hf = face(q, 0.0, 0.0, h, h, fric, total / n, "implicit_quadratic")  # flat water: no pressure gradient
            assert 0.0 < q < prev  # monotone decay, never sign-changing, never growing
            prev = q
        return q

    errors = {n: abs(run(n) - exact) for n in (10, 20, 40, 80)}
    assert errors[80] < errors[10] / 6.0
    for n in (10, 20, 40):
        assert 1.6 < errors[n] / errors[2 * n] < 2.4  # implicit Euler: first order in dt
    assert errors[80] < 1e-2 * exact  # first-order constant ~ (dt a q0 / 2) ln(1 + a q0 T) / (1 + a q0 T)^2 ~ 0.6% at N = 80


@pytest.mark.parametrize("dt", [0.01, 0.25, 5.0, 1000.0])
@pytest.mark.parametrize("h", [0.005, 0.0625, 0.3])
@pytest.mark.parametrize("fric", [0.7, 21.45])
def test_uniform_darcy_equilibrium_is_an_exact_fixed_point_for_any_dt(dt, h, fric):
    slope = 0.125  # dyadic, dx = 1: A is 0.125 m higher than B, equal depths
    q_eq = math.sqrt(8.0 * G * slope / fric) * h ** 1.5
    q, hf = face(q_eq, slope, 0.0, h, h, fric, dt, "implicit_quadratic")
    assert q == pytest.approx(q_eq, rel=1e-12)
    # and the opposite slope gives the exactly mirrored equilibrium
    q_neg, _ = face(-q_eq, 0.0, slope, h, h, fric, dt, "implicit_quadratic")
    assert q_neg == pytest.approx(-q_eq, rel=1e-12)


def test_iterating_from_rest_climbs_monotonically_to_equilibrium_and_never_overshoots():
    h, fric, slope, dt = 0.01, 21.45, 0.125, 2.0
    q_eq = math.sqrt(8.0 * G * slope / fric) * h ** 1.5
    q, prev = 0.0, 0.0
    for _ in range(300):
        q, _hf = face(q, slope, 0.0, h, h, fric, dt, "implicit_quadratic")
        assert prev * (1.0 - 1e-13) <= q <= q_eq * (1.0 + 1e-13)  # monotone up to the last-ulp noise at convergence
        prev = q
    assert q == pytest.approx(q_eq, rel=1e-9)


def test_new_flux_is_bounded_by_qstar_and_friction_is_energy_dissipative():
    rng = np.random.default_rng(5)
    n = 5000
    h = 10.0 ** rng.uniform(-20.0, 0.5, n)
    qs = rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(-30.0, 3.0, n)
    q = implicit_quadratic_friction_flux(qs, h, 10.0 ** rng.uniform(-1.0, 2.0, n), 0.25)
    assert np.all(np.isfinite(q))
    assert np.all(np.sign(q) == np.sign(qs))  # never reverses the pressure-driven direction
    assert np.all(np.abs(q) <= np.abs(qs))  # |q'| <= |q*| exactly
    assert np.all(q * (q - qs) <= 0.0)  # friction work q' (q' - q*) = -alpha |q'| q'^2 <= 0


# --- solver level: rest, ledgers, mutation, parity -------------------------------------------------------------------------
BED = np.array([[0.5, 0.25, 1.5, 0.75, 1.0],
                [0.0, 0.125, 1.25, 0.5, 1.125],
                [0.75, 0.375, 0.625, 1.375, 0.25]])
FINE = ExperimentalControl(max_dt_s=1.0 / 128.0, min_dt_s=1.0 / 4096.0)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_lake_at_rest_is_exact_with_the_implicit_treatment(backend):
    if backend == "numba":
        pytest.importorskip("numba")
    depth = np.maximum(1.0 - BED, 0.0)
    pond = make_pond(topo(BED), backend, control=IMPLICIT)
    result = evolve_pond(pond, uniform_rain_rate(pond, 0.0), state_of(pond, depth), 0.125, FINE)
    assert result.reached_target and np.array_equal(result.state.depth_m, depth)
    assert not np.any(result.state.qx_m2_s) and not np.any(result.state.qy_m2_s)
    closure = result.closure()
    assert closure.closes and closure.export_m3 == 0.0


def test_rain_ponding_and_a_bump_conserve_water_per_cell_and_globally_through_rejections():
    # Bounded window: a 0.5 m bump on a 0.5 m grid with friction 1.0 is a very-high-Froude dam break; beyond ~0.2 s it leaves the
    # qualified local-inertial regime (root: min_dt stop at 0.2007 s), a documented limitation, not tuned away. Here only the first
    # 0.125 s are asked: the large first step still violates the gravity-wave CFL and is rejected (not clipped, no limiter).
    pond = make_pond(topo(np.zeros((3, 4)), dx=0.5), control=IMPLICIT, ksat=1e-6)
    depth = np.full((3, 4), 1e-7)
    depth[1, 1] += 0.5
    result = evolve_pond(pond, uniform_rain_rate(pond, 3e-5), state_of(pond, depth), 0.125,
                         ExperimentalControl(max_dt_s=0.5, min_dt_s=1.0 / 4096.0))
    closure = result.closure()
    assert result.reached_target and result.n_rejected_attempts >= 1
    assert closure.closes and closure.max_cell_residual_m3 <= closure.cell_bound_m3
    assert abs(closure.global_residual_m3) <= closure.global_bound_m3 and closure.export_m3 == 0.0
    assert H(result.state.depth_m).min() >= 0.0  # nothing was clipped to get here


def test_a_rejected_step_changes_no_input():
    pond = make_pond(topo(np.zeros((3, 4)), dx=0.5), control=IMPLICIT)
    depth = np.full((3, 4), 0.25)
    state = state_of(pond, depth)
    rain = uniform_rain_rate(pond, 1e-5)
    before = (state.depth_m.copy(), state.soil_water_m.copy(), state.qx_m2_s.copy(), state.qy_m2_s.copy(), rain.copy())
    with pytest.raises(HydraulicStepRejected):
        step_once(pond, rain, state, 10.0)
    after = (state.depth_m, state.soil_water_m, state.qx_m2_s, state.qy_m2_s, rain)
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    accepted = step_once(pond, rain, state, 1.0 / 64.0)  # and the accepted step is pure too
    assert np.array_equal(state.qx_m2_s, before[2]) and accepted.state is not state


FILM_M = 2e-7  # explicit first-wetting film (the root's h_face is 2.16e-7 m)


def sloped_pond(control, backend="numpy", *, ny=4, nx=8, slope=0.11, fric=21.45):
    """Plane falling east at `slope`, dx = 1, an east normal-flow outlet on every row; ksat = 0 (the real infiltration physics is
    not altered, it is simply not part of this fixture: with ksat 1e-6 and 0.01 m suction a dry start swallows the whole first
    0.25 s of 120 mm/h rain, so no wet face exists at step 1)."""
    z = np.tile(slope * (nx - 1 - np.arange(nx, dtype=np.float64)), (ny, 1))
    return make_pond(topo(z, [(r, nx - 1, 0, 1) for r in range(ny)], drop=slope, friction=fric, dx=1.0), backend, control=control,
                     ksat=0.0)


def test_first_wet_pond_froude_is_bounded_without_clipping_and_lagged_shows_the_artifact():
    fric, slope, dt = 21.45, 0.11, 0.25
    fr_eq = math.sqrt(8.0 * slope / fric)  # equilibrium Froude of the Darcy-Weisbach law, independent of depth
    lagged, implicit = sloped_pond(HydraulicControl()), sloped_pond(IMPLICIT)
    film = np.full((4, 8), FILM_M)
    rain = uniform_rain_rate(implicit, 0.0)
    first_l = step_once(lagged, rain, state_of(lagged, film), dt)
    first_i = step_once(implicit, rain, state_of(implicit, film), dt)
    dl, di = face_speed_diagnostics(lagged, first_l), face_speed_diagnostics(implicit, first_i)
    # genuinely wet faces carrying flux (interior x faces, q_old = 0): not a vacuous dry-start comparison
    for diag, step in ((dl, first_l), (di, first_i)):
        assert diag.stats["n_wet_faces"] > 0 and np.max(np.abs(H(step.step.used_flux_m2_s["x"])[:, 1:-1])) > 0.0
        assert np.min(H(diag.face_flow_depth_m["x"])[:, 1:-1]) == pytest.approx(FILM_M, rel=1e-9)
    interior = (slice(None), slice(1, -1))
    assert H(dl.face_froude["x"])[interior].max() > 10.0  # lagged: frictionless impulse g S dt on the film
    assert H(di.face_froude["x"])[interior].max() <= fr_eq * (1.0 + 1e-12)  # implicit: below sqrt(8 S / f)
    assert di.stats["max_face_froude"] <= fr_eq * (1.0 + 1e-12)
    state, worst = first_i.state, 0.0
    for _ in range(19):  # five seconds in total; later drying/wetting fronts are not claimed to be bounded
        accepted = step_once(implicit, rain, state, dt)
        worst = max(worst, face_speed_diagnostics(implicit, accepted).stats["max_face_froude"])
        state = accepted.state
    assert worst <= 0.5 and H(state.depth_m).min() >= 0.0


def test_numba_lateral_stage_is_bitwise_numpy_for_both_treatments():
    pytest.importorskip("numba")
    rng = np.random.default_rng(21)
    bed = rng.uniform(0.0, 0.3, (5, 6))
    depth = np.where(rng.random((5, 6)) < 0.5, rng.uniform(0.0, 1e-6, (5, 6)), rng.uniform(0.0, 0.02, (5, 6)))
    topology = topo(bed, [(2, 5, 0, 1), (0, 0, -1, 0)], drop=0.1, friction=15.0)
    for control in (HydraulicControl(), IMPLICIT):
        ref, comp = make_pond(topology, "numpy", control=control), make_pond(topology, "numba", control=control)
        state = state_of(ref, depth)
        rain = uniform_rain_rate(ref, 0.0)  # ksat = 0 and no rain: the column is the identity, so the lateral stage is bitwise
        for _ in range(12):
            a, b = step_once(ref, rain, state, 0.01), step_once(comp, rain, state, 0.01)
            for key in ("x", "y"):
                assert np.array_equal(a.step.used_flux_m2_s[key], b.step.used_flux_m2_s[key])
                assert np.array_equal(a.step.face_flow_depth_m[key], b.step.face_flow_depth_m[key])
            assert np.array_equal(a.state.depth_m, b.state.depth_m)
            state = a.state


def test_cuda_matches_numpy_for_both_treatments(gpu):
    rng = np.random.default_rng(22)
    bed = rng.uniform(0.0, 0.3, (5, 6))
    depth = np.where(rng.random((5, 6)) < 0.5, rng.uniform(0.0, 1e-6, (5, 6)), rng.uniform(0.0, 0.02, (5, 6)))
    topology = topo(bed, [(2, 5, 0, 1), (0, 0, -1, 0)], drop=0.1, friction=15.0)
    for control in (HydraulicControl(), IMPLICIT):
        ref, dev = make_pond(topology, "numpy", control=control), make_pond(topology, "cuda", control=control)
        assert dev.describe()["solver"]["control"]["friction_treatment"] == control.friction_treatment
        state = state_of(ref, depth)
        dstate = state_of(dev, depth)
        rain, drain = uniform_rain_rate(ref, 1e-5), uniform_rain_rate(dev, 1e-5)
        for _ in range(12):
            a, b = step_once(ref, rain, state, 0.01), step_once(dev, drain, dstate, 0.01)
            for key in ("x", "y"):
                np.testing.assert_allclose(H(b.step.used_flux_m2_s[key]), a.step.used_flux_m2_s[key], rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(H(b.state.depth_m), a.state.depth_m, rtol=RTOL, atol=ATOL)
            state, dstate = a.state, b.state  # both continue from the reference so the comparison stays one-step


def test_cuda_first_wet_pond_and_ledger(gpu):
    dev = sloped_pond(IMPLICIT, "cuda")
    rain = uniform_rain_rate(dev, 0.0)
    result = evolve_pond(dev, rain, state_of(dev, np.full((4, 8), FILM_M)), 2.0,
                         ExperimentalControl(max_dt_s=0.25, min_dt_s=1.0 / 4096.0))
    assert result.reached_target and result.closure().closes


# --- kernel argument layout, seals, preparation ----------------------------------------------------------------------------
def test_numba_kernel_signature_seal_and_provenance(numba_kernels):
    names = list(inspect.signature(numba_kernels.local.py_func).parameters)
    assert names.index("implicit_friction") == names.index("donor_limiter") + 1
    topology = topo(np.zeros((2, 3)), [(0, 2, 0, 1)])
    lag, imp = make_pond(topology, "numba", control=HydraulicControl()), make_pond(topology, "numba", control=IMPLICIT)
    assert lag.solver._seal != imp.solver._seal  # the treatment is part of the sealed control
    forged = make_pond(topology, "numba", control=HydraulicControl())
    object.__setattr__(forged.solver, "control", replace(forged.solver.control, friction_treatment="implicit_quadratic"))
    with pytest.raises(ExperimentalHydrologyError, match="sealed"):
        forged.solver.step(uniform_rain_rate(forged, 0.0), state_of(forged, np.full((2, 3), 0.01)), 0.01)


def test_cuda_source_has_the_new_argument_in_both_face_kernels_and_stays_ieee_strict():
    from maple_syrup import experimental_cuda as ec

    src = ec.kernel_source()
    for kernel in ("maple_syrup_li_face_x", "maple_syrup_li_face_y"):
        header = src[src.index(f"void {kernel}("):]
        header = header[:header.index("{")]
        assert header.rstrip().endswith("const int implicit_friction)"), kernel
    assert src.count("implicit_quadratic_flux(") == 2  # definition + the single call in li_face
    assert "hypot" not in src and "__dsqrt_rn(fabs(qs))" in src
    assert len(ec._KERNELS["local_inertial"]) == 7  # no kernel was added: only an argument and a device function


@pytest.mark.parametrize("q_old", [1e300, -1e300, 0.0])
def test_raw_cuda_face_kernel_huge_qstar_thin_film_and_the_lagged_flag_path(gpu, q_old):
    """Launch the REAL x-face kernel on one interior face (ny = 1, nx = 2): A at z = 0 with a 1e-200 m film, B dry at z = -0.1,
    q_old = +/-1e300 (so q* ~ q_old). The implicit root must equal the 80-digit value (not 0); flag 0; lagged arithmetic otherwise."""
    from maple_syrup import experimental_cuda as ec

    cp = gpu
    f64 = np.float64
    z, h = cp.asarray(np.array([0.0, -0.1])), cp.asarray(np.array([1e-200, 0.0]))
    ftype, fsign = cp.asarray(np.array([0, 1, 0], dtype=np.int8)), cp.asarray(np.zeros(3, dtype=np.int8))
    zmax, fric, kb = (cp.asarray(np.array(v, dtype=f64)) for v in ([0.0, 0.0, 0.0], [21.45] * 3, [0.0] * 3))
    qo = cp.asarray(np.array([0.0, q_old, 0.0]))
    out = {}
    for implicit in (1, 0):
        q_new, hf_row, fflag = cp.empty(3, dtype=f64), cp.empty(3, dtype=f64), cp.empty(3, dtype=np.uint8)
        ec._launch("maple_syrup_li_face_x", 1, ec.ELEMENT_THREADS,
                   (np.int64(1), np.int64(2), z, h, ftype, zmax, fric, kb, fsign, qo, q_new, hf_row, fflag, f64(0.25), f64(1.0),
                    f64(G), np.int32(implicit)))
        out[implicit] = (float(q_new.get()[1]), float(hf_row.get()[1]), int(fflag.get()[1]))
    q_imp, hf, flag = out[1]
    q_star = q_old - ((G * 1e-200) * 0.25) * (-0.1)
    assert hf == 1e-200 and flag == 0
    assert q_imp != 0.0 and rel_err(q_imp, exact_root(q_star, 1e-200, 21.45, 0.25)) <= 4 * ROOT_REL
    assert out[0][1] == 1e-200 and out[0][2] == 0  # the lagged branch still runs (its own arithmetic, tested elsewhere)


def test_cuda_source_orders_the_product_so_a_tiny_t_cannot_zero_a_huge_qstar():
    from maple_syrup import experimental_cuda as ec

    body = ec.kernel_source()
    body = body[body.index("double implicit_quadratic_flux("):]
    body = body[:body.index("__device__ __forceinline__ void li_face(")]
    assert "t >= 0.5" in body and "__ddiv_rn(qs, d)" in body and "if (!isfinite(q)) return nan_value();" in body


def test_cuda_seal_contains_the_treatment_and_forgery_is_refused(gpu):
    topology = topo(np.zeros((2, 3)), [(0, 2, 0, 1)])
    lag, imp = make_pond(topology, "cuda", control=HydraulicControl()), make_pond(topology, "cuda", control=IMPLICIT)
    assert lag.solver._seal != imp.solver._seal
    object.__setattr__(lag.solver, "_control", replace(lag.solver.control, friction_treatment="implicit_quadratic"))
    with pytest.raises(ExperimentalHydrologyError, match="sealed"):
        lag.solver.step(uniform_rain_rate(lag, 0.0), state_of(lag, np.full((2, 3), 0.01)), 0.01)
