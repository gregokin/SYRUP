"""Root's frozen pre-B2 same-backend preferential controls (48 scenarios / 336 columns) and independent hand controls on the SPATIAL operator: hourly isolated fraction, beta 0/1, the ONE rate cap,
all-zero weights disabled, unnormalised [0,1,1], mixed background / multi-species overlap against an independent closed-form evaluation, a saturating receiver leaving its share on the surface (no
compensation by another layer), dt = 0, repeated sub-steps, inactive cells. Author-written; the controls are compared, never regenerated or rebound."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from conftest import BIG, PARAMS, REF_NUMBA, REF_NUMPY, bits, full, hand_column, hand_depths, hand_op, hand_state, pi, sp, spf, step

BIGB = 1 << 33


def _scenarios(ref):
    if not (ref / "controls.json").exists():
        return []
    return [s for s in json.loads((ref / "controls.json").read_text())["scenarios"] if s["operation"] == "preferential"]


@pytest.mark.skipif(not (REF_NUMPY / "controls.json").exists() or not PARAMS.exists(), reason="Root's frozen reference is not present")
@pytest.mark.parametrize("scenario", _scenarios(REF_NUMPY), ids=lambda s: s["label"])
def test_frozen_same_backend_controls(backend, scenario):
    ref = REF_NUMPY if backend == "numpy" else REF_NUMBA
    expected = np.load(ref / "expected.npz")
    configs = json.loads(PARAMS.read_text())["controls"]
    c = configs[scenario["parameters_index"]]
    spec = sp.spatial_layer_spec(np.array(c["depth_edges_m"]), theta_residual=np.array(c["theta_residual"]), theta_field_capacity=np.array(c["theta_field_capacity"]),
                                 theta_saturation=np.array(c["theta_saturation"]), theta_evaporation_stress=c["theta_evaporation_stress"], diffusivity_m2_s=0.0, n_cells=c["cells"])
    mask = np.array(scenario["mask"])
    col = sp.prepare_spatial_column(spec, mask, scenario["area_m2"], backend=backend, memory_budget_bytes=BIGB)
    op = spf.prepare_spatial_preferential(col, a3=scenario["a3"], max_fraction_per_hour=scenario["max_fraction_per_hour"], max_intake_rate_m_s=scenario["max_intake_rate_m_s"],
                                          background_receiver_weights=np.array(scenario["background_receiver_weights"]), memory_budget_bytes=BIGB)
    state = sp.spatial_column_state(col, np.ascontiguousarray(np.array(scenario["water_m"])), scenario["t_s"], memory_budget_bytes=BIGB)
    r = spf.spatial_preferential_step(op, np.ascontiguousarray(np.array(scenario["surface_m"])), state, scenario["dt_s"], memory_budget_bytes=BIGB)
    assert r.state.t_s == scenario["t_s"]
    label = scenario["label"]
    for name, a in (("water", r.state.water_m), ("intake_m", r.intake_m), ("surface_water_m", r.surface_water_m), ("intake_total_m", r.intake_total_m)):
        want = expected[f"{label}_{name}"]
        np.testing.assert_array_equal(bits(a), bits(want), err_msg=f"{label} {name}")  # BIT-exact against the frozen same-backend common result
    assert abs(r.budget.soil_residual_m3) <= r.budget.bound_m3 and abs(r.budget.surface_residual_m3) <= r.budget.bound_m3


@pytest.mark.skipif(not (REF_NUMPY / "controls.json").exists(), reason="Root's frozen reference is not present")
def test_the_frozen_set_has_the_declared_size():
    sc = _scenarios(REF_NUMPY)
    assert len(sc) == 48 and sum(len(s["mask"]) for s in sc) == 336
    assert {s["a3"] for s in sc} == {0.0, 0.35, 1.0}


# ---- independent hand formulas ---------------------------------------------------------------------------------------------------------
def layer_map(a3, background, species, contributions):
    """g_l = a3 b_l + (1 - a3)/(1 + C) sum_s c_s u_sl with normalised vectors; disabled (all-zero) species contribute nothing."""
    L = len(background) if background is not None else len(species[0])
    b = np.zeros(L) if background is None or sum(background) == 0 else np.array(background) / sum(background)
    g = a3 * b
    cover, acc = 0.0, np.zeros(L)
    for w, c in zip(species, contributions, strict=True):
        if sum(w) > 0:
            cover += c
            acc += c * (np.array(w) / sum(w))
    g = g + (1 - a3) / (1 + cover) * acc
    return g


def expected_intake(g, maxf, H, dt, max_rate, room):
    """beta = maxFL2 sum g, k = -log1p(-beta)/3600, attempted = min(H (-expm1(-k dt)), cap); top-layer-first, bounded by room and the remaining surface, no redistribution."""
    s = g.sum()
    beta = maxf * s
    frac = 1.0 if beta >= 1.0 else -math.expm1(-(-math.log1p(-beta) / 3600.0) * dt)
    attempted = min(H * frac, max_rate * dt)
    r = g / s if s > 0 else np.zeros_like(g)
    left, out = H, np.zeros_like(g)
    for l in range(len(g)):
        a = min(attempted * r[l], max(room[l], 0.0), left)
        out[l] = a
        left -= a
    return out, left


@pytest.mark.parametrize("L", [1, 2, 3])
def test_one_hour_isolated_fraction_and_repeated_sub_steps(backend, L):
    col = hand_column(L, 3, backend)
    op = hand_op(col, a3=0.5, maxf=1.0)  # background [1, 0, ...]: beta = a3 + 0 = 0.5 (no species)
    state = hand_state(col, theta=0.1)
    H = 0.1
    r = step(op, full(col, H), state, 3600.0)
    np.testing.assert_allclose(r.intake_total_m, 0.5 * H, rtol=1e-14)  # one isolated hour reproduces beta H
    np.testing.assert_allclose(r.surface_water_m, H - 0.5 * H, rtol=1e-14)
    s, st = full(col, H), state
    total = np.zeros(3)
    for _ in range(4):  # four 900 s sub-steps: the survival factors multiply
        q = step(op, s, st, 900.0)
        s, st = q.surface_water_m, q.state
        total += q.intake_total_m
    np.testing.assert_allclose(total, 0.5 * H, rtol=1e-13)
    assert q.state.t_s == 0.0  # no implicit time advance


def test_beta_zero_beta_one_and_dt_zero(backend):
    col = hand_column(2, 3, backend)
    state = hand_state(col, theta=0.1)
    H = 0.05
    z = step(hand_op(col, a3=0.0), full(col, H), state, 3600.0)  # a3 = 0 and no species: beta 0
    assert not z.intake_m.any() and z.budget.intake_m3 == 0.0
    np.testing.assert_array_equal(bits(z.surface_water_m), bits(full(col, H)))
    one = step(hand_op(col, a3=1.0, maxf=1.0), full(col, H), state, 1e-3)  # beta 1: k = +inf, ALL of H for any dt > 0
    np.testing.assert_array_equal(one.intake_total_m, H)
    np.testing.assert_array_equal(one.surface_water_m, 0.0)
    ident = step(hand_op(col, a3=1.0), full(col, H), state, 0.0)  # dt = 0: the identity
    np.testing.assert_array_equal(bits(ident.state.water_m), bits(state.water_m))
    np.testing.assert_array_equal(bits(ident.surface_water_m), bits(full(col, H)))
    assert ident.budget.attempted_m3 == 0.0 and ident.state.t_s == state.t_s


def test_one_aggregate_rate_cap_scalar_and_per_cell(backend):
    col = hand_column(2, 3, backend)
    state = hand_state(col, theta=0.1)
    H, dt = 0.2, 3600.0
    capped = step(hand_op(col, a3=1.0, maxrate=1e-6), full(col, H), state, dt)
    np.testing.assert_allclose(capped.intake_total_m, 1e-6 * dt, rtol=1e-14)
    assert capped.budget.rate_capped_m3 == pytest.approx(3 * (H - 1e-6 * dt), rel=1e-12)
    per_cell = step(hand_op(col, a3=1.0, maxrate=np.array([0.0, 1e-6, np.inf])), full(col, H), state, dt)
    np.testing.assert_allclose(per_cell.intake_total_m, [0.0, 1e-6 * dt, H], rtol=1e-14)  # +inf = no cap; 0 = no intake


def test_all_zero_weights_disable_their_pathway(backend):
    col = hand_column(3, 2, backend)
    state = hand_state(col, theta=0.1)
    cover = [np.array([5.0, 5.0]), np.array([1.0, 1.0])]
    # background [0,0,0] and species [0,0,0] (cover ignored): nothing declared -> nothing moves
    none = step(hand_op(col, a3=0.5, background=[0, 0, 0], species=[[0, 0, 0]], contributions=cover[:1]), full(col, 0.05), state, 3600.0)
    assert none.budget.intake_m3 == 0.0
    # the disabled species does not even dilute the enabled one: identical to the one-species operator
    a = step(hand_op(col, a3=0.4, background=[0, 0, 0], species=[[0, 0, 1], [0, 0, 0]], contributions=[cover[1], cover[0]]), full(col, 0.05), state, 3600.0)
    b = step(hand_op(col, a3=0.4, background=[0, 0, 0], species=[[0, 0, 1]], contributions=[cover[1]]), full(col, 0.05), state, 3600.0)
    np.testing.assert_array_equal(bits(a.intake_m), bits(b.intake_m))
    # background disabled: only the species enhancement (1 - a3) C/(1 + C) acts
    want = 0.05 * (1 - 0.4) * 1.0 / 2.0
    np.testing.assert_allclose(a.intake_total_m, want, rtol=1e-13)


def test_unnormalised_lower_two_layers_split_equally(backend):
    col = hand_column(3, 2, backend)
    state = hand_state(col, theta=0.1)
    r = step(hand_op(col, a3=1.0, maxf=0.5, background=[0.0, 1.0, 1.0]), full(col, 0.1), state, 3600.0)  # beta = 0.5, equal shares to layers 2 and 3
    np.testing.assert_allclose(r.intake_m[0], 0.0, atol=0)
    np.testing.assert_allclose(r.intake_m[1], 0.025, rtol=1e-14)
    np.testing.assert_allclose(r.intake_m[2], 0.025, rtol=1e-14)
    scaled = step(hand_op(col, a3=1.0, maxf=0.5, background=[0.0, 7.5, 7.5]), full(col, 0.1), state, 3600.0)  # scaling the vector changes nothing
    np.testing.assert_array_equal(bits(scaled.intake_m), bits(r.intake_m))


@pytest.mark.parametrize("a3", [0.0, 0.3, 0.8, 1.0])
def test_mixed_background_and_overlapping_species_against_the_closed_form(backend, a3):
    L, N = 3, 4
    col = hand_column(L, N, backend)
    state = hand_state(col, theta=0.05)
    bg = [1.0, 2.0, 0.0]
    species = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    contrib = [np.array([0.0, 0.4, 1.5, 3.0]), np.array([2.0, 0.0, 0.5, 1.0]), np.array([9.0, 9.0, 9.0, 9.0])]
    op = hand_op(col, a3=a3, maxf=0.8, maxrate=3e-6, background=bg, species=species, contributions=contrib)
    H, dt = np.array([0.01, 0.02, 0.05, 0.1]), 1800.0
    r = step(op, H, state, dt)
    dz = hand_depths(L)
    room = 0.9 * dz - 0.05 * dz
    for i in range(N):
        g = layer_map(a3, bg, species, [c[i] for c in contrib])
        out, left = expected_intake(g, 0.8, H[i], dt, 3e-6, room)
        np.testing.assert_allclose(r.intake_m[:, i], out, rtol=1e-12, atol=1e-18, err_msg=f"cell {i} a3 {a3}")
        np.testing.assert_allclose(r.surface_water_m[i], left, rtol=1e-12, atol=1e-18)
    assert abs(r.budget.soil_residual_m3) <= r.budget.bound_m3


def test_a_saturating_receiver_keeps_its_share_on_the_surface(backend):
    """Weights [1, 1]: the top layer is nearly full, so its half is limited by ITS room; the other receiver still gets only ITS half and the rest stays on the surface (no compensation)."""
    col = hand_column(2, 2, backend)
    room_top = 1e-3
    water = np.array([[0.9 * 0.5 - room_top] * 2, [0.1] * 2])
    state = hand_state(col, water=water)
    H = 0.2
    r = step(hand_op(col, a3=1.0, maxf=1.0, background=[1.0, 1.0]), full(col, H), state, 1e-3)  # beta 1: attempted = H
    np.testing.assert_allclose(r.intake_m[0], room_top, rtol=1e-12)  # top limited by its room
    np.testing.assert_allclose(r.intake_m[1], H / 2, rtol=1e-14)  # the second receiver gets its own half only
    np.testing.assert_allclose(r.surface_water_m, H - room_top - H / 2, rtol=1e-12)  # the rejected share stays on the surface
    assert r.budget.capacity_rejected_m3 > 0.0 and r.budget.attempted_m3 == pytest.approx(2 * H, rel=1e-14)


def test_saturation_room_not_field_capacity_limits_the_receivers(backend):
    col = hand_column(1, 1, backend)
    fc_depth, sat_depth = 0.25 * 0.5, 0.9 * 0.5
    state = hand_state(col, water=np.array([[fc_depth + 0.01]]))  # above field capacity, far below saturation
    r = step(hand_op(col, a3=1.0), full(col, 0.05), state, 10.0)
    np.testing.assert_allclose(r.intake_total_m, 0.05, rtol=1e-14)  # the field-capacity depth does not cap the intake
    full_state = hand_state(col, water=np.array([[sat_depth]]))
    r2 = step(hand_op(col, a3=1.0), full(col, 0.05), full_state, 10.0)
    assert r2.intake_total_m[0] == 0.0 and r2.surface_water_m[0] == 0.05


def test_inactive_cells_keep_state_and_surface_and_get_no_water(backend):
    col = hand_column(2, 4, backend, mask=[True, False, True, False])
    state = hand_state(col, theta=0.1)
    H = np.array([0.05, 0.07, 0.05, 0.09])
    r = step(hand_op(col, a3=1.0, maxf=1.0), H, state, 3600.0)
    np.testing.assert_array_equal(r.intake_m[:, [1, 3]], 0.0)
    np.testing.assert_array_equal(bits(r.surface_water_m[[1, 3]]), bits(H[[1, 3]]))
    np.testing.assert_array_equal(bits(r.state.water_m[:, [1, 3]]), bits(state.water_m[:, [1, 3]]))
    np.testing.assert_array_equal(r.intake_total_m[[0, 2]], 0.05)
