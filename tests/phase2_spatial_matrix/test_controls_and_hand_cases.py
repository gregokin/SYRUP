"""Root's frozen pre-B2 common NumPy controls (32 matrix scenarios / 224 columns), Root's decimal oracles, the Phase 1 hand cases and the history rule on the SPATIAL operator. Author-written; the controls
are compared, never regenerated or rebound."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from conftest import ATOL, RTOL, TASK, bits, cases, full, hand_column, hand_hist, hand_op, hand_state, mi, sm, sp, step

REF = TASK / "root_s2_b2_reference"
PARAMS = TASK / "root_s2_parameter_controls.json"
needs_ref = pytest.mark.skipif(not (REF / "controls.json").exists() or not PARAMS.exists(), reason="Root's frozen reference is not present")
BIGB = 1 << 33


def _load():
    controls = json.loads((REF / "controls.json").read_text())
    expected = np.load(REF / "expected.npz")
    configs = json.loads(PARAMS.read_text())["controls"]
    return controls, expected, configs


def _scenarios():
    if not (REF / "controls.json").exists():
        return []
    return [s for s in json.loads((REF / "controls.json").read_text())["scenarios"] if s["operation"] == "matrix"]


@needs_ref
@pytest.mark.parametrize("scenario", _scenarios(), ids=lambda s: s["label"])
def test_frozen_common_numpy_controls(backend, scenario):
    controls, expected, configs = _load()
    c = configs[scenario["parameters_index"]]
    edges = np.array(c["depth_edges_m"])
    N = c["cells"]
    spec = sp.spatial_layer_spec(edges, theta_residual=np.array(c["theta_residual"]), theta_field_capacity=np.array(c["theta_field_capacity"]), theta_saturation=np.array(c["theta_saturation"]),
                                 theta_evaporation_stress=c["theta_evaporation_stress"], diffusivity_m2_s=0.0, n_cells=N)
    mask = np.array(scenario["mask"])
    col = sp.prepare_spatial_column(spec, mask, scenario["area_m2"], backend=backend, memory_budget_bytes=BIGB)
    arr = lambda k: np.ascontiguousarray(np.array(scenario[k], dtype=np.float64))  # noqa: E731
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=arr("conductivity_m_s"), suction_m=arr("suction_m"), vegetation_factor=arr("vegetation_factor"), memory_budget_bytes=BIGB)
    state = sp.spatial_column_state(col, arr("water_m"), scenario["t_s"], memory_budget_bytes=BIGB)
    hist = sm.spatial_matrix_history(op, wetting_m=arr("wetting_m"), cumulative_intake_m=arr("cumulative_intake_m"), memory_budget_bytes=BIGB)
    r = sm.spatial_matrix_step(op, arr("pool_m"), arr("standing_m"), arr("incoming_rate_m_s"), state, hist, scenario["dt_s"], memory_budget_bytes=BIGB)
    label = scenario["label"]
    assert r.state.t_s == scenario["t_s"]
    got = {"water": r.state.water_m, "surface_pool_m": r.surface_pool_m, "intake_m": r.intake_m, "wetting_m": r.history.wetting_m, "cumulative_intake_m": r.history.cumulative_intake_m}
    for name, a in got.items():
        want = expected[f"{label}_{name}"]
        if backend == "numpy":
            np.testing.assert_array_equal(bits(a), bits(want), err_msg=f"{label} {name}")  # BIT-exact against the frozen common NumPy result
        else:
            np.testing.assert_allclose(a, want, rtol=RTOL, atol=ATOL, err_msg=f"{label} {name}")
    assert abs(r.budget.soil_residual_m3) <= r.budget.bound_m3 and abs(r.budget.pool_residual_m3) <= r.budget.bound_m3


@needs_ref
def test_the_frozen_set_has_the_declared_size_and_exercises_the_branches():
    sc = _scenarios()
    assert len(sc) == 32 and sum(len(s["mask"]) for s in sc) == 224
    # coverage over the frozen set: ponded, onset-inside, dry-state, saturated-state scenarios all present
    assert {s["label"].split("_")[2] for s in sc} >= {"dry", "below", "above", "saturated"}
    assert {s["label"].split("_")[-1] for s in sc} == {"only", "ponded"}


ORACLES = cases.load_root_oracles()


@pytest.mark.parametrize("oracle", ORACLES["cases"], ids=lambda c: c["name"])
def test_root_decimal_oracles(backend, oracle):
    """Ponded contact intervals with a fixed b (b = 0.01 through psi = PSI, or b = 0), against Root's 90-digit values."""
    b = float(oracle["capillary_b_m"])
    col = hand_column(3, backend)
    op = hand_op(col, k=float(oracle["K_m_s"]), psi=0.0 if b == 0.0 else cases.PSI, g=float(oracle["g"]))
    state = hand_state(col)
    pool = float(oracle["surface_pool_m"])
    r = step(op, full(col, pool), full(col, pool), full(col, 0.0), state, hand_hist(op, float(oracle["J_initial_m"]), 0.0), float(oracle["dt_s"]))
    tol = 1e-13 if backend == "numpy" else 1e-12
    np.testing.assert_allclose(r.intake_m, float(oracle["actual_matrix_intake_m"]), rtol=tol, atol=1e-30)
    np.testing.assert_allclose(r.history.wetting_m, float(oracle["J_final_m"]), rtol=tol)
    np.testing.assert_allclose(r.surface_pool_m, float(oracle["surface_final_m"]), rtol=tol)
    np.testing.assert_allclose(r.history.cumulative_intake_m, float(oracle["actual_matrix_intake_m"]), rtol=tol, atol=1e-30)
    assert abs(r.budget.soil_residual_m3) <= r.budget.bound_m3


def test_root_constant_rate_onset_control(backend):
    oc = ORACLES["constant_rate_onset_control"]
    spec = sp.spatial_layer_spec(np.array([0.0, cases.DZ0, cases.DZ0 + 1.0]), theta_residual=0.01, theta_field_capacity=0.25, theta_saturation=0.9, theta_evaporation_stress=0.05, diffusivity_m2_s=0.0, n_cells=2)
    col = sp.prepare_spatial_column(spec, np.ones(2, bool), 1.0, backend=backend, memory_budget_bytes=BIGB)
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=1e-6, suction_m=0.16, vegetation_factor=0.5, memory_budget_bytes=BIGB)  # b = 0.16 x 0.125 = 0.02
    state = hand_state(col)
    rt, dt = float(oc["incoming_rate_m_s"]), float(oc["dt_s"])
    r = step(op, full(col, rt * dt), full(col, 0.0), full(col, rt), state, hand_hist(op), dt)
    tol = 1e-14 if backend == "numpy" else 1e-12
    np.testing.assert_allclose(r.history.wetting_m, float(oc["J_final_m"]), rtol=tol)
    np.testing.assert_allclose(r.intake_m, float(oc["total_matrix_intake_m"]), rtol=tol)
    np.testing.assert_allclose(r.surface_pool_m, float(oc["surface_final_m"]), rtol=1e-13)
    assert r.budget.prepond_intake_m3 == pytest.approx(float(oc["prepond_intake_m"]) * 2, rel=1e-13) and r.budget.n_onsets == 2


def test_deficit_is_the_exact_evaluation_order_and_room_is_saturation(backend):
    """b = psi * (max(fc_m0 - W0, 0) / dz0) with fc_m0 = theta_fc dz rounded FIRST (a non-dyadic geometry would expose the algebraic reordering `theta_fc - W/dz`); the room is the SATURATION depth minus
    W0 (field capacity is a different, gravity threshold)."""
    dz0, fc, sat, res = 0.3, 0.17, 0.2, 0.02
    spec = sp.spatial_layer_spec(np.array([0.0, dz0, 1.0]), theta_residual=res, theta_field_capacity=fc, theta_saturation=sat, theta_evaporation_stress=0.05, diffusivity_m2_s=0.0, n_cells=1)
    col = sp.prepare_spatial_column(spec, np.ones(1, bool), 1.0, backend=backend, memory_budget_bytes=BIGB)
    psi, k, g = 0.37, 3e-6, 0.8
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=k, suction_m=psi, vegetation_factor=g, memory_budget_bytes=BIGB)
    w0 = 0.043
    state = sp.spatial_column_state(col, np.array([[w0], [0.05]]), 0.0, memory_budget_bytes=BIGB)
    fc_m0 = fc * dz0
    sat_m0 = sat * dz0
    b = psi * (max(fc_m0 - w0, 0.0) / dz0)
    pool = 0.5  # contact, huge pool: ponded from the start, J advanced by the hydraulic increment
    r = step(op, np.array([pool]), np.array([pool]), np.zeros(1), state, hand_hist(op, 0.001), 120.0)
    j_old = 0.001
    # solve x - b log1p(x/(J + b)) = K tau by bisection in plain floats (independent of the library helper)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - b * math.log1p(mid / (j_old + b)) < k * 120.0:
            lo = mid
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    room = sat_m0 - w0
    expect = min(g * x, room)
    assert r.intake_m[0] == pytest.approx(expect, rel=1e-9)
    # the room caps at SATURATION: with a larger step the intake equals the room exactly and W0 + intake == sat_m0
    r2 = step(op, np.array([pool]), np.array([pool]), np.zeros(1), state, hand_hist(op, 0.001), 1e7)
    assert r2.state.water_m[0, 0] == pytest.approx(sat_m0, rel=1e-15) and r2.intake_m[0] == pytest.approx(room, rel=1e-13)
    # J advanced by admitted / g (g > 0), the ledger by the actual intake
    assert r2.history.wetting_m[0] == pytest.approx(0.001 + room / g, rel=1e-13) and r2.history.cumulative_intake_m[0] == pytest.approx(room, rel=1e-13)
    # below-FC vs above-FC water both give a finite b >= 0; above field capacity the deficit is exactly 0 (b = 0)
    above = sp.spatial_column_state(col, np.array([[fc_m0 + 1e-3], [0.05]]), 0.0, memory_budget_bytes=BIGB)
    r3 = step(op, np.array([pool]), np.array([pool]), np.zeros(1), above, hand_hist(op, 0.001), 60.0)
    assert r3.intake_m[0] == pytest.approx(min(g * k * 60.0, sat_m0 - (fc_m0 + 1e-3)), rel=1e-12)  # b = 0: x = K tau exactly


def test_unponded_pool_admits_unscaled_and_g_zero_and_one(backend):
    col = hand_column(2, backend)
    state = hand_state(col)
    for g in (0.0, 0.25, 1.0):
        op = hand_op(col, g=g)
        h = hand_hist(op, 0.002, 0.003)
        # low rate r <= K, no contact: the whole pool (r dt) is admitted, J advances by the admitted amount UNSCALED for every g
        r_lo = cases.K * 0.5
        res = step(op, full(col, r_lo * 600.0), full(col, 0.0), full(col, r_lo), state, h, 600.0)
        np.testing.assert_allclose(res.intake_m, r_lo * 600.0, rtol=1e-15)
        np.testing.assert_allclose(res.history.wetting_m, 0.002 + r_lo * 600.0, rtol=1e-14)
        np.testing.assert_allclose(res.history.cumulative_intake_m, 0.003 + r_lo * 600.0, rtol=1e-14)
        np.testing.assert_array_equal(res.surface_pool_m, 0.0)
        assert res.budget.n_ponded == 0
    # ponded contact with g = 0: zero postpond intake, J unchanged, no division
    op0 = hand_op(col, g=0.0)
    h = hand_hist(op0, 0.002)
    res = step(op0, full(col, 0.01), full(col, 0.01), full(col, 0.0), state, h, 60.0)
    np.testing.assert_array_equal(res.intake_m, 0.0)
    assert bits(res.history.wetting_m).tobytes() == bits(h.wetting_m).tobytes() and res.budget.n_ponded == 2
    # g = 1: J == cumulative intake within one episode (standard cumulative Green-Ampt) and F_s = b K/(r - K)
    op1 = hand_op(col, g=1.0)
    r = 3e-5
    res = step(op1, full(col, r * 600.0), full(col, 0.0), full(col, r), state, hand_hist(op1), 600.0)
    np.testing.assert_allclose(res.history.wetting_m, res.history.cumulative_intake_m, rtol=1e-14)
    np.testing.assert_allclose(res.budget.prepond_intake_m3, 2 * cases.B * cases.K / (r - cases.K), rtol=1e-12)
    assert res.budget.n_onsets == 2


def test_rejected_capacity_never_advances_J_and_history_is_retained(backend):
    """Top layer almost full: the room, not the law, limits the intake; J advances by room / g only. Then a dry interval and a rain-off interval retain J and the ledger bit for bit (no recovery)."""
    col = hand_column(3, backend)
    op = hand_op(col, psi=10.0)
    room = 0.001
    water = np.array([[cases.SAT * cases.DZ0 - room] * 3, [0.1] * 3])
    state = sp.spatial_column_state(col, water, 7.0, memory_budget_bytes=BIGB)
    res = step(op, full(col, 0.02), full(col, 0.02), full(col, 0.0), state, hand_hist(op, 0.002), 600.0)
    np.testing.assert_allclose(res.intake_m, room, rtol=1e-12)
    np.testing.assert_allclose(res.history.wetting_m, 0.002 + room / cases.G, rtol=1e-12)
    assert res.budget.room_rejected_m3 > 0.0
    again = step(op, res.surface_pool_m, res.surface_pool_m, full(col, 0.0), res.state, res.history, 600.0)  # zero room: inert
    assert bits(again.history.wetting_m).tobytes() == bits(res.history.wetting_m).tobytes() and again.budget.intake_m3 == 0.0
    zero = full(col, 0.0)
    h = res.history
    for dt in (0.0, 1.0, 3600.0, 1e7):  # dry intervals of any length: no intake, no change, no reset, no rescaling
        dry = step(op, zero, zero, zero, res.state, h, dt)
        assert bits(dry.history.wetting_m).tobytes() == bits(h.wetting_m).tobytes() and bits(dry.history.cumulative_intake_m).tobytes() == bits(h.cumulative_intake_m).tobytes()
        assert bits(dry.state.water_m).tobytes() == bits(res.state.water_m).tobytes() and dry.state.t_s == 7.0


def test_dt_zero_is_the_identity_only_for_valid_inputs(backend):
    col = hand_column(4, backend)
    op = hand_op(col)
    state = hand_state(col, t_s=3.0)
    h = hand_hist(op, 0.003, 0.004)
    res = step(op, full(col, 0.01), full(col, 0.01), full(col, 0.0), state, h, 0.0)
    assert bits(res.state.water_m).tobytes() == bits(state.water_m).tobytes() and res.budget.candidate_m3 == 0.0 and res.budget.intake_m3 == 0.0
    assert bits(res.history.wetting_m).tobytes() == bits(h.wetting_m).tobytes() and bits(res.surface_pool_m).tobytes() == bits(full(col, 0.01)).tobytes()
    bad = full(col, 0.01)
    bad[2] = -1.0
    with pytest.raises(mi.MatrixStepError) as info:
        step(op, bad, full(col, 0.01), full(col, 0.0), state, h, 0.0)  # an invalid pool is refused even at dt = 0 (contact standing keeps the other cells consistent)
    assert info.value.bit == 4 and info.value.n_flagged == 1


def test_controlled_split_with_constant_b_is_additive(backend):
    """Splitting an interval is exact ONLY where b is constant across the split (the existing method re-evaluates b = psi max(fc_m0 - W0, 0)/dz0 per call): psi = 0 (b = 0) or an unponded supply-limited
    regime. With b held fixed the intake and J sum to the whole-interval values (to rounding)."""
    col = hand_column(2, backend)
    state = hand_state(col)
    # (a) b = 0 contact pond: x = K tau, additive
    op = hand_op(col, psi=0.0)
    big = 1.0
    whole = step(op, full(col, big), full(col, big), full(col, 0.0), state, hand_hist(op, 0.002), 600.0)
    half = step(op, full(col, big), full(col, big), full(col, 0.0), state, hand_hist(op, 0.002), 300.0)
    half2 = step(op, half.surface_pool_m, half.surface_pool_m, full(col, 0.0), half.state, half.history, 300.0)
    # the second half sees a wetter column, but with psi = 0 b stays 0, so the sum is the whole
    np.testing.assert_allclose(half.intake_m + half2.intake_m, whole.intake_m, rtol=1e-13)
    np.testing.assert_allclose(half2.history.wetting_m, whole.history.wetting_m, rtol=1e-12)
    # (b) unponded supply-limited regime (r <= K): dyadic values, exact sums, independent of b
    op2 = hand_op(col)
    r = cases.K / 2
    pool_a, pool_b = r * 256.0, r * 256.0
    a = step(op2, full(col, pool_a), full(col, 0.0), full(col, r), state, hand_hist(op2, 0.0, 0.0), 256.0)
    b2 = step(op2, full(col, pool_b), full(col, 0.0), full(col, r), a.state, a.history, 256.0)
    w = step(op2, full(col, r * 512.0), full(col, 0.0), full(col, r), state, hand_hist(op2, 0.0, 0.0), 512.0)
    np.testing.assert_allclose(a.intake_m + b2.intake_m, w.intake_m, rtol=1e-15)
    np.testing.assert_allclose(b2.history.wetting_m, w.history.wetting_m, rtol=1e-15)


def test_ordinary_refinement_with_changing_b_is_reported_not_forced(backend):
    """Honest note, not a physics fix: with psi > 0 and ponded intake b is re-evaluated from the wetter column at the second call, so two half steps need not equal one whole step. The test only
    records that both conserve water and that J advances monotonically; it does not assert equality."""
    col = hand_column(1, backend)
    op = hand_op(col)
    state = hand_state(col)
    big = 1.0
    whole = step(op, full(col, big), full(col, big), full(col, 0.0), state, hand_hist(op, 0.002), 600.0)
    h1 = step(op, full(col, big), full(col, big), full(col, 0.0), state, hand_hist(op, 0.002), 300.0)
    h2 = step(op, h1.surface_pool_m, h1.surface_pool_m, full(col, 0.0), h1.state, h1.history, 300.0)
    for r in (whole, h1, h2):
        assert abs(r.budget.soil_residual_m3) <= r.budget.bound_m3 and abs(r.budget.pool_residual_m3) <= r.budget.bound_m3
    assert h2.history.wetting_m[0] > h1.history.wetting_m[0] > 0.002 and whole.history.wetting_m[0] > 0.002
