"""Phase 3b infiltration column checks.

Expected values come from an independent scalar transcription of MAHLERAN
`infilt.for` (inf_type < 5, millimetre units, the legacy three-branch
structure and def1 <= 100 cut-off), from hand-derived closed forms, or from
the water balance itself -- never from the kernel's own intermediate
arithmetic.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from maple_syrup.infiltration import (
    LOCAL_BALANCE_RTOL,
    InfiltrationError,
    column_parameters,
    column_step,
    initial_soil_water_m,
    pavement_lambda_m_per_s,
)

EPS = float(np.finfo(np.float64).eps)


def grid(value, shape=(1, 1)):
    if np.isscalar(value):
        return np.full(shape, float(value), dtype=np.float64)
    return np.array(value, dtype=np.float64)


def params(shape=(1, 1), *, model="fixed_ksat", ksat=1e-6, psi=0.05, drain=0.0, theta_sat=0.4,
           thickness=0.3, pavement=None, mask=None):
    kwargs = {
        "model": model, "ksat_m_per_s": grid(ksat, shape), "suction_m": grid(psi, shape),
        "drainage_parameter": grid(drain, shape), "theta_sat": grid(theta_sat, shape),
        "soil_thickness_m": grid(thickness, shape), "active_mask": mask,
    }
    if model == "pavement_hawkins":
        kwargs["pavement_cover_fraction"] = grid(0.0 if pavement is None else pavement, shape)
    return column_parameters(**kwargs)


def step1(p, h, s, r, dt, **kw):
    """One step on a 1 x 1 grid; returns host scalars."""
    out = column_step(p, grid(h), grid(s), grid(r), dt, **kw)
    return {k: float(np.asarray(getattr(out, k)).item()) for k in
            ("depth_m", "soil_water_m", "rain_m", "intake_m", "saturation_return_m", "drainage_m")}


def assert_balance(before_h, before_s, out):
    lhs = np.asarray(out.depth_m) + np.asarray(out.soil_water_m) + np.asarray(out.drainage_m)
    rhs = np.asarray(before_h) + np.asarray(before_s) + np.asarray(out.rain_m)
    scale = rhs + np.asarray(out.intake_m) + np.asarray(out.drainage_m) + np.asarray(out.saturation_return_m)
    assert np.all(np.abs(lhs - rhs) <= LOCAL_BALANCE_RTOL * scale)


# --- independent legacy transcription (mm, mm/s) -----------------------------------
def legacy_infilt(*, model, r, d1, cum_inf, stmax, theta_sat, ksat, psi, drain_par, pave, dt):
    """infilt.for 38-207 for one cell, inf_type < 5, no routing. Returns
    (surface mm after the step incl. excess * dt, cum_inf mm, drain mm, branch)."""
    theta = cum_inf / stmax * theta_sat  # legacy theta update (line 205)
    if model == 1:
        final_infilt = ksat
    else:
        lam = -0.022891667 * math.log(pave) - 0.098575 if pave > 0.0 else 0.16
        final_infilt = lam * (1 - math.exp(-r / lam)) if r > 0.0 else ksat  # local r, not r2(i, 2)
    c = (psi + d1) * (theta_sat - theta) * final_infilt
    def1 = (cum_inf * final_infilt) / c
    if def1 <= 100.0:
        d = math.exp(def1)
        f = (final_infilt * d) / (d - 1.0)
    else:
        f = final_infilt
    water_in = r + d1 / dt
    drain = (theta / theta_sat) * (ksat * drain_par * dt)
    if f >= water_in:
        branch, excess, d1_new, cum = "complete_runon", 0.0, 0.0, cum_inf + water_in * dt
    elif f <= r:
        branch, excess, d1_new, cum = "rain_excess", r - f, d1, cum_inf + f * dt
    else:
        branch, excess, d1_new, cum = "partial_runon", 0.0, d1 - (f - r) * dt, cum_inf + f * dt
    if cum >= drain:
        cum -= drain
    else:
        drain, cum = cum, 0.0
    if cum > stmax:
        excess += (cum - stmax) / dt
        cum = stmax
        branch += "+saturation"
    return d1_new + excess * dt, cum, drain, branch


@pytest.mark.parametrize("model", ["fixed_ksat", "pavement_hawkins"])
def test_matches_legacy_scalar_transcription(model):
    rng = np.random.default_rng(20260929)
    shape = (24, 25)
    n = shape[0] * shape[1]
    theta_sat = rng.uniform(0.3, 0.5, n)
    thickness_mm = rng.uniform(100.0, 500.0, n)
    stmax = theta_sat * thickness_mm
    fill = np.where(rng.random(n) < 0.25, 1.0 - rng.uniform(1e-7, 1e-4, n), rng.uniform(0.02, 0.95, n))
    cum = fill * stmax
    ksat = rng.uniform(1e-4, 5e-2, n)
    psi = rng.uniform(1.0, 100.0, n)
    drain = rng.uniform(0.0, 0.5, n)
    r = np.where(rng.random(n) < 0.2, 0.0, rng.uniform(1e-4, 5e-2, n))
    d1 = np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0.0, 5.0, n))
    cover = np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0.01, 1.0, n))
    dt = 1.0
    p = column_parameters(
        model=model, ksat_m_per_s=(ksat * 1e-3).reshape(shape), suction_m=(psi * 1e-3).reshape(shape),
        drainage_parameter=drain.reshape(shape), theta_sat=theta_sat.reshape(shape),
        soil_thickness_m=(thickness_mm * 1e-3).reshape(shape),
        **({"pavement_cover_fraction": cover.reshape(shape)} if model == "pavement_hawkins" else {}),
    )
    h0, s0, rate = (d1 * 1e-3).reshape(shape), (cum * 1e-3).reshape(shape), (r * 1e-3).reshape(shape)
    out = column_step(p, h0, s0, rate, dt)
    branches = set()
    for i in range(n):
        h_ref, cum_ref, drain_ref, branch = legacy_infilt(
            model=1 if model == "fixed_ksat" else 2, r=r[i], d1=d1[i], cum_inf=cum[i], stmax=stmax[i],
            theta_sat=theta_sat[i], ksat=ksat[i], psi=psi[i], drain_par=drain[i], pave=cover[i] * 1e-2, dt=dt,
        )
        branches.add(branch.split("+")[0])
        branches.add("saturation" if "saturation" in branch else "unsaturated")
        idx = np.unravel_index(i, shape)
        assert out.depth_m[idx] * 1e3 == pytest.approx(h_ref, rel=1e-9, abs=1e-12)
        assert out.soil_water_m[idx] * 1e3 == pytest.approx(cum_ref, rel=1e-9, abs=1e-12)
        assert out.drainage_m[idx] * 1e3 == pytest.approx(drain_ref, rel=1e-9, abs=1e-15)
    assert branches == {"complete_runon", "rain_excess", "partial_runon", "saturation", "unsaturated"}
    assert_balance(h0, s0, out)


# --- analytic cases -----------------------------------------------------------------
def test_supply_limited_all_rain_infiltrates():
    p = params(ksat=1e-3, psi=0.05)  # capacity >> rain
    r = step1(p, 0.0, 0.03, 2e-6, 10.0)
    assert r["rain_m"] == pytest.approx(2e-5, rel=EPS)  # one rounding of rate * dt
    assert r["intake_m"] == r["rain_m"] and r["depth_m"] == 0.0
    assert r["soil_water_m"] == 0.03 + r["rain_m"] and r["drainage_m"] == 0.0 == r["saturation_return_m"]


def _capacity(k, psi, h, theta_sat, thickness, s):
    x = s / ((psi + h) * (theta_sat - s / thickness))
    return k / (1.0 - math.exp(-x))


def test_rainfall_excess_ponds_the_remainder():
    k, psi, h, s, dt, rain = 1e-6, 0.05, 0.002, 0.06, 5.0, 2e-5
    cap = _capacity(k, psi, h, 0.4, 0.3, s)
    assert cap * dt < rain * dt
    r = step1(params(ksat=k, psi=psi), h, s, rain, dt)
    assert r["intake_m"] == pytest.approx(cap * dt, rel=1e-12)
    assert r["depth_m"] == pytest.approx(h + rain * dt - cap * dt, rel=1e-12)


def test_ponded_runon_infiltrates_without_rain():
    k, psi, h, s, dt = 1e-5, 0.05, 0.01, 0.02, 2.0
    cap = _capacity(k, psi, h, 0.4, 0.3, s)
    partial = step1(params(ksat=k, psi=psi), h, s, 0.0, dt)
    assert partial["intake_m"] == pytest.approx(cap * dt, rel=1e-12) and 0.0 < partial["depth_m"] < h
    complete = step1(params(ksat=k, psi=psi), 1e-6, s, 0.0, dt)  # capacity * dt > h
    assert complete["intake_m"] == 1e-6 and complete["depth_m"] == 0.0


def test_saturation_return_goes_back_to_the_surface():
    smax = 0.4 * 0.3
    s = smax - 1e-4
    r = step1(params(ksat=1e-3, psi=0.05), 0.01, s, 0.0, 1.0)
    assert r["intake_m"] == pytest.approx(1e-3, rel=1e-9)  # deficit tiny: capacity -> K
    assert r["soil_water_m"] == smax
    assert r["saturation_return_m"] == pytest.approx(r["intake_m"] - 1e-4, rel=1e-9)
    assert r["depth_m"] == pytest.approx(0.01 - 1e-4, rel=1e-9)


def test_drainage_is_the_discrete_linear_decay():
    ksat, c, smax, s0, dt = 1e-5, 0.5, 0.12, 0.08, 10.0
    p = params(ksat=ksat, drain=c)
    rate = ksat * c / smax
    s = grid(s0)
    for _ in range(50):
        out = column_step(p, grid(0.0), s, grid(0.0), dt)
        assert_balance(grid(0.0), s, out)
        s = out.soil_water_m
    assert s.item() == pytest.approx(s0 * (1.0 - rate * dt) ** 50, rel=1e-12)


def test_drainage_converges_to_exponential_decay():
    ksat, c, smax, s0, total = 1e-4, 1.0, 0.12, 0.08, 600.0
    exact = s0 * math.exp(-ksat * c / smax * total)
    errors = []
    for dt in (60.0, 30.0, 15.0, 7.5):
        s = grid(s0)
        for _ in range(int(total / dt)):
            s = column_step(params(ksat=ksat, drain=c), grid(0.0), s, grid(0.0), dt).soil_water_m
        errors.append(abs(s.item() - exact))
    assert all(a > b for a, b in itertools.pairwise(errors))
    assert errors[0] / errors[-1] > 6.0  # first order: ~8 over a factor 8 in dt


def test_drainage_cannot_exceed_the_water_present():
    r = step1(params(ksat=1.0, drain=1.0), 0.0, 0.05, 0.0, 1.0)
    assert r["drainage_m"] == 0.05 and r["soil_water_m"] == 0.0


# --- limiting cases -------------------------------------------------------------------
def test_zero_conductivity_blocks_intake_and_drainage():
    r = step1(params(ksat=0.0, drain=0.3), 0.001, 0.05, 1e-5, 10.0)
    assert r["intake_m"] == 0.0 and r["drainage_m"] == 0.0
    assert r["depth_m"] == 0.001 + r["rain_m"] and r["soil_water_m"] == 0.05


def test_zero_deficit_limits_capacity_to_k():
    smax = 0.4 * 0.3
    r = step1(params(ksat=1e-5, psi=0.05), 0.0, smax, 1e-4, 1.0)
    # O = (Smax + J) - Smax: exact up to the rounding of Smax + J, i.e. ~eps * Smax (~4e-18 m here).
    assert r["intake_m"] == 1e-5
    assert r["saturation_return_m"] == pytest.approx(1e-5, rel=0, abs=EPS * smax)
    assert r["soil_water_m"] == smax and r["depth_m"] == pytest.approx(1e-4, rel=0, abs=2 * EPS * smax)


def test_zero_suction_and_depth_limits_capacity_to_k():
    r = step1(params(ksat=1e-6, psi=0.0), 0.0, 0.05, 1e-4, 1.0)
    assert r["intake_m"] == 1e-6 and r["depth_m"] == pytest.approx(1e-4 - 1e-6, rel=1e-14)


def test_empty_column_takes_all_available_water():
    r = step1(params(ksat=1e-7, psi=0.05), 0.002, 0.0, 1e-5, 10.0)  # unbounded capacity
    assert r["intake_m"] == 0.002 + r["rain_m"] and r["depth_m"] == 0.0
    tiny = step1(params(ksat=1e-7, psi=0.05), 0.002, 1e-310, 0.0, 1.0)  # 1 - exp(-x) underflows
    assert tiny["intake_m"] == 0.002 and math.isfinite(tiny["soil_water_m"])
    doubly_zero = step1(params(ksat=1e-6, psi=0.0), 0.0, 0.0, 1e-4, 1.0)  # no capillary term -> K
    assert doubly_zero["intake_m"] == 1e-6


def test_empty_column_overfill_returns_to_surface():
    smax = 0.4 * 0.3
    r = step1(params(ksat=1e-7, psi=0.05), 0.5, 0.0, 0.0, 1.0)
    assert r["soil_water_m"] == smax and r["saturation_return_m"] == pytest.approx(0.5 - smax, rel=1e-15)
    assert r["depth_m"] == pytest.approx(0.5 - smax, rel=1e-15)


# --- model 2: pavement lambda and LOCAL rain -------------------------------------------
def test_pavement_lambda_values():
    cover = np.array([[0.0, 0.005, 0.5, 1.0]])
    expected_mm = [0.16] + [-0.022891667 * math.log(c * 1e-2) - 0.098575 for c in (0.005, 0.5, 1.0)]
    np.testing.assert_allclose(pavement_lambda_m_per_s(cover)[0], np.array(expected_mm) * 1e-3, rtol=1e-14)
    assert np.all(pavement_lambda_m_per_s(cover) > 0.0)


def test_model2_uses_each_cells_own_rain():
    # Full columns without drainage: capacity = K exactly and intake = K dt.
    shape, dt = (1, 4), 1.0
    cover = np.array([[0.3, 0.3, 0.0, 0.3]])
    rain = np.array([[0.0, 1e-5, 1e-5, 4e-5]])
    ksat = 2.5e-7
    p = params(shape, model="pavement_hawkins", ksat=ksat, pavement=cover)
    smax = grid(0.4 * 0.3, shape)
    out = column_step(p, grid(0.5, shape), smax, rain, dt)
    lam = [(-0.022891667 * math.log(c * 1e-2) - 0.098575 if c > 0 else 0.16) * 1e-3 for c in cover[0]]
    expected = [ksat] + [lam[j] * (1.0 - math.exp(-rain[0, j] / lam[j])) for j in (1, 2, 3)]
    np.testing.assert_allclose(out.intake_m[0], np.array(expected) * dt, rtol=1e-12)
    assert len(set(out.intake_m[0].tolist())) == 4  # every column differs
    assert_balance(grid(0.5, shape), smax, out)


# --- masks, rejection, identity ---------------------------------------------------------
def test_inactive_cells_keep_storage_exactly():
    shape = (2, 2)
    mask = np.array([[True, False], [False, True]])
    p = params(shape, ksat=1e-5, drain=0.2, mask=mask)
    h, s = grid([[0.001, 0.004], [0.0, 0.002]]), grid([[0.05, 0.11], [0.0, 0.07]])
    rain = np.where(mask, 1e-5, 0.0)
    out = column_step(p, h, s, rain, 30.0)
    np.testing.assert_array_equal(out.depth_m[~mask], h[~mask])
    np.testing.assert_array_equal(out.soil_water_m[~mask], s[~mask])
    for flux in (out.intake_m, out.drainage_m, out.saturation_return_m, out.rain_m):
        assert not np.any(flux[~mask])
    assert np.any(out.intake_m[mask] > 0.0)
    with pytest.raises(InfiltrationError, match="inactive"):
        column_step(p, h, s, grid(1e-5, shape), 30.0)


def test_dt_zero_is_identity_and_returns_new_arrays():
    h, s = grid(0.003), grid(0.05)
    out = column_step(params(), h, s, grid(1e-5), 0.0)
    assert out.depth_m.item() == 0.003 and out.soil_water_m.item() == 0.05
    assert not np.shares_memory(out.depth_m, h) and not np.shares_memory(out.soil_water_m, s)
    assert not any(np.any(a) for a in (out.rain_m, out.intake_m, out.drainage_m, out.saturation_return_m))


def test_successful_step_does_not_mutate_or_alias_inputs():
    h, s, r = grid(0.001), grid(0.05), grid(1e-5)
    copies = [a.copy() for a in (h, s, r)]
    out = column_step(params(ksat=1e-6, drain=0.1), h, s, r, 5.0)
    for a, b in zip((h, s, r), copies):
        np.testing.assert_array_equal(a, b)
    for new in (out.depth_m, out.soil_water_m):
        assert not any(np.shares_memory(new, a) for a in (h, s, r))


_SMAX = 0.4 * 0.3


@pytest.mark.parametrize(
    "h, s, r, dt, match",
    [
        (-1e-9, 0.05, 0.0, 1.0, "depth_m must be >= 0"),
        (np.nan, 0.05, 0.0, 1.0, "depth_m must be finite"),
        (0.0, -1e-9, 0.0, 1.0, "soil_water_m must be >= 0"),
        (0.0, _SMAX * (1 + 1e-12), 0.0, 1.0, "exceeds"),
        (0.0, 0.05, np.inf, 1.0, "rain_rate_m_per_s must be finite"),
        (0.0, 0.05, -1e-6, 1.0, "rain_rate_m_per_s must be >= 0"),
        (0.0, 0.05, 0.0, -1.0, "dt_s"),
        (0.0, 0.05, 0.0, math.nan, "dt_s"),
        (0.0, 0.05, 0.0, True, "dt_s"),
    ],
)
def test_invalid_state_is_rejected_without_mutation(h, s, r, dt, match):
    arrays = [grid(h), grid(s), grid(r)]
    copies = [a.copy() for a in arrays]
    with pytest.raises(InfiltrationError, match=match):
        column_step(params(), *arrays, dt)
    for a, b in zip(arrays, copies):
        np.testing.assert_array_equal(a, b)


def test_structural_misuse_is_rejected():
    p = params((2, 2))
    ok = grid(0.0, (2, 2))
    with pytest.raises(InfiltrationError, match="shape"):
        column_step(p, grid(0.0, (2, 3)), ok, ok, 1.0)
    with pytest.raises(InfiltrationError, match="float64"):
        column_step(p, ok.astype(np.float32), ok, ok, 1.0)
    with pytest.raises(InfiltrationError, match="array"):
        column_step(p, [[0.0, 0.0], [0.0, 0.0]], ok, ok, 1.0)
    with pytest.raises(InfiltrationError, match="ColumnParameters"):
        column_step(object(), ok, ok, ok, 1.0)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"ksat": -1e-9}, "ksat_m_per_s must be >= 0"),
        ({"ksat": np.nan}, "ksat_m_per_s must be finite"),
        ({"psi": -0.01}, "suction_m must be >= 0"),
        ({"drain": -0.1}, "drainage_parameter must be >= 0"),
        ({"theta_sat": 0.0}, "theta_sat must be > 0"),
        ({"theta_sat": 1.2}, "theta_sat must be <= 1"),
        ({"thickness": 0.0}, "soil_thickness_m must be > 0"),
        ({"model": "pavement_hawkins", "pavement": 1.5}, "pavement_cover_fraction must lie"),
        ({"model": "green_ampt"}, "model must be one of"),
    ],
)
def test_invalid_parameters_are_rejected(overrides, match):
    with pytest.raises(InfiltrationError, match=match):
        params(**overrides)


def test_parameter_structure_is_checked():
    base = {"ksat_m_per_s": grid(1e-6), "suction_m": grid(0.05), "drainage_parameter": grid(0.0),
                "theta_sat": grid(0.4), "soil_thickness_m": grid(0.3)}
    with pytest.raises(InfiltrationError, match="needs pavement"):
        column_parameters(model="pavement_hawkins", **base)
    with pytest.raises(InfiltrationError, match="only used"):
        column_parameters(model="fixed_ksat", pavement_cover_fraction=grid(0.0), **base)
    with pytest.raises(InfiltrationError, match="float64"):
        column_parameters(model="fixed_ksat", **{**base, "suction_m": grid(0.05).astype(np.float32)})
    with pytest.raises(InfiltrationError, match="bool"):
        column_parameters(model="fixed_ksat", active_mask=grid(1.0), **base)
    with pytest.raises(InfiltrationError, match="shape"):
        column_parameters(model="fixed_ksat", **{**base, "theta_sat": grid(0.4, (1, 2))})


def test_parameters_are_private_copies():
    ksat = grid(1e-6)
    p = column_parameters(model="fixed_ksat", ksat_m_per_s=ksat, suction_m=grid(0.05),
                          drainage_parameter=grid(0.0), theta_sat=grid(0.4), soil_thickness_m=grid(0.3))
    ksat[...] = 1.0
    assert p.ksat_m_per_s.item() == 1e-6
    with pytest.raises(ValueError):
        p.ksat_m_per_s[...] = 2.0


def test_initial_soil_water():
    p = params((1, 2))
    np.testing.assert_array_equal(initial_soil_water_m(p, grid([[0.25, 0.4]])), [[0.25 * 0.3, 0.4 * 0.3]])
    with pytest.raises(InfiltrationError, match="theta_sat"):
        initial_soil_water_m(p, grid([[0.25, 0.41]]))
    with pytest.raises(InfiltrationError, match=">= 0"):
        initial_soil_water_m(p, grid([[-0.01, 0.2]]))


def test_validate_false_gives_identical_results():
    p = params((1, 2), ksat=1e-6, drain=0.1)
    args = (grid([[0.001, 0.0]]), grid([[0.05, 0.1]]), grid([[1e-5, 3e-5]]), 7.0)
    a, b = column_step(p, *args), column_step(p, *args, validate=False)
    for name in ("depth_m", "soil_water_m", "intake_m", "drainage_m", "saturation_return_m"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


# --- time-step refinement -----------------------------------------------------------------
def _ponded_storm(dt, total=600.0):
    """Initially ponded, dry soil: intake is capacity-limited throughout, so
    the explicit update is forward Euler of a smooth ODE in (h, S)."""
    p = params(ksat=1e-5, psi=0.05, drain=0.05)
    h, s, drained, rained = grid(0.005), grid(0.01 * 0.3), 0.0, 0.0
    h0s0 = h.item() + s.item()
    for _ in range(round(total / dt)):
        out = column_step(p, h, s, grid(3e-5), dt)
        assert out.depth_m.item() > 0.0  # ponded throughout: intake = capacity * dt
        h, s = out.depth_m, out.soil_water_m
        drained += out.drainage_m.item()
        rained += out.rain_m.item()
    assert h.item() + s.item() + drained == pytest.approx(h0s0 + rained, rel=1e-10)
    return h.item(), s.item()


def test_timestep_refinement_converges_at_first_order():
    reference_h, _ = _ponded_storm(0.05)
    assert reference_h > 0.0
    errors = [abs(_ponded_storm(dt)[0] - reference_h) for dt in (8.0, 4.0, 2.0, 1.0)]
    assert all(a > b for a, b in itertools.pairwise(errors))
    assert errors[0] / errors[-1] > 5.0  # ~8 for first order over a factor 8 in dt


# --- optional CuPy parity ------------------------------------------------------------------
def test_cupy_matches_numpy():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; GPU path not exercised (no GPU claim)")
    cp = backend.cupy_module()
    rng = np.random.default_rng(7)
    shape = (16, 9)
    host = {"ksat_m_per_s": rng.uniform(0, 1e-5, shape), "suction_m": rng.uniform(0, 0.1, shape),
                "drainage_parameter": rng.uniform(0, 0.5, shape), "theta_sat": rng.uniform(0.3, 0.5, shape),
                "soil_thickness_m": rng.uniform(0.1, 0.5, shape), "pavement_cover_fraction": rng.uniform(0, 1, shape)}
    state = (rng.uniform(0, 0.01, shape), host["theta_sat"] * host["soil_thickness_m"] * rng.uniform(0, 1, shape),
             rng.uniform(0, 5e-5, shape))
    p_host = column_parameters(model="pavement_hawkins", **host)
    p_dev = column_parameters(model="pavement_hawkins", **{k: backend.to_device(v, cp) for k, v in host.items()})
    a = column_step(p_host, *state, 3.0)
    b = column_step(p_dev, *(backend.to_device(v, cp) for v in state), 3.0)
    assert backend.is_device_array(b.depth_m)
    for name in ("depth_m", "soil_water_m", "intake_m", "drainage_m", "saturation_return_m"):
        np.testing.assert_allclose(backend.to_host(getattr(b, name)), getattr(a, name), rtol=1e-12, atol=1e-18)
