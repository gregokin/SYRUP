"""Independent analytic / hand-derived cases (the expected numbers are worked out by hand from the stated physics and written as literals or from closed-form formulas; no value is read back from the
implementation or from a mirror of it): field-capacity cascade, capacity-limited closed bottom, saturation return, exact pair relaxation at unequal dz, withdrawal-floor evaporation, zero water with a
nonzero residual, and the pulse step. Per-cell DIFFERENT geometry and properties in every case. Author-written."""
from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import BIG, prepare

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss


def spec_of(edges, res, fc, sat, iface, *, mode="constant", bottom="open", gravity=True, stress=0.5, factor=1.0):
    edges, res, fc, sat = (np.ascontiguousarray(np.asarray(a, dtype=float).T) for a in (edges, res, fc, sat))  # inputs below are written (cells, layers): transpose to (layers, cells)
    iface = np.ascontiguousarray(np.asarray(iface, dtype=float).T)
    N = edges.shape[1]
    extra = dict(diffusivity_m2_s=iface) if mode == "constant" else dict(interface_conductivity_m_s=iface, diffusion_coefficient=0.3)
    return sp.spatial_layer_spec(edges, theta_residual=res, theta_field_capacity=fc, theta_saturation=sat, theta_evaporation_stress=stress, evaporation_factor=factor, gravity_drainage=gravity,
                                 diffusion_mode=mode, bottom_drainage=bottom, n_cells=N, **extra)


def step(col, water_cells, dt=1.0, added=None, pet=None, share=None, recharge=None):
    w = np.ascontiguousarray(np.asarray(water_cells, dtype=float).T)  # (cells, layers) -> (layers, cells)
    N = w.shape[1]
    st = sp.spatial_column_state(col, w, 0.0, memory_budget_bytes=BIG)
    pet = np.zeros(N) if pet is None else np.asarray(pet, dtype=float)
    if recharge is not None:
        return sp.spatial_step_with_recharge(col, st, dt, recharge_depth_m=recharge, potential_evaporation_m_s=pet, evaporation_share=share, memory_budget_bytes=BIG)
    added = np.zeros_like(w) if added is None else np.ascontiguousarray(np.asarray(added, dtype=float).T)
    return sp.spatial_step(col, st, dt, added_water_m=added, potential_evaporation_m_s=pet, evaporation_share=share, memory_budget_bytes=BIG)


# two cells with DIFFERENT widths and thresholds (hand-chosen so every number below is easy to verify)
A = dict(edges=[0.0, 0.1, 0.3, 0.6], res=[0.02] * 3, fc=[0.2] * 3, sat=[0.5] * 3)  # widths 0.1,0.2,0.3 -> FC depths 0.02,0.04,0.06 ; saturation depths 0.05,0.10,0.15
B = dict(edges=[0.0, 0.2, 0.4, 0.6], res=[0.02] * 3, fc=[0.1, 0.3, 0.2], sat=[0.5] * 3)  # widths 0.2 -> FC depths 0.02,0.06,0.04 ; saturation depths 0.10 each
PAIR = {k: np.array([A[k], B[k]]) for k in A}
ZERO_D = np.zeros((2, 2))


def test_field_capacity_cascade_open_bottom_unequal_layers():
    col = prepare(spec_of(PAIR["edges"], PAIR["res"], PAIR["fc"], PAIR["sat"], ZERO_D, bottom="open"))
    out = step(col, [[0.05, 0.03, 0.01], [0.08, 0.05, 0.09]])
    # cell A: top excess 0.05-0.02 = 0.03 -> layer 1 = 0.06 (> FC 0.04) excess 0.02 -> layer 2 = 0.03 (<= FC 0.06): stops. cell B: excess 0.06 -> 0.11 (FC 0.06) excess 0.05 -> 0.14 (FC 0.04) excess 0.10 leaves
    np.testing.assert_allclose(out.state.water_m.T, [[0.02, 0.04, 0.03], [0.02, 0.06, 0.04]], rtol=0, atol=1e-13)
    np.testing.assert_allclose(out.drainage_m, [0.0, 0.10], rtol=0, atol=1e-13)
    np.testing.assert_allclose(out.percolation_m.T, [[0.03, 0.02], [0.06, 0.05]], rtol=0, atol=1e-13)
    assert abs(out.budget.residual_m3) <= out.budget.bound_m3 and out.state.t_s == 1.0


def test_capacity_limited_closed_bottom_keeps_water_above_field_capacity_but_never_above_saturation():
    col = prepare(spec_of(PAIR["edges"], PAIR["res"], PAIR["fc"], PAIR["sat"], ZERO_D, bottom="closed"))
    out = step(col, [[0.05, 0.03, 0.01], [0.08, 0.05, 0.09]])
    # A is unchanged by the closed bottom (room below). B: deeper capacity 0.06 -> forwards 0.06, layer 1 = 0.11, forwards only the remaining room 0.01, ends [0.02, 0.10, 0.10]: nothing leaves
    np.testing.assert_allclose(out.state.water_m.T, [[0.02, 0.04, 0.03], [0.02, 0.10, 0.10]], rtol=0, atol=1e-13)
    np.testing.assert_allclose(out.drainage_m, [0.0, 0.0], atol=0)
    assert (out.state.water_m <= np.array([[0.05, 0.1], [0.1, 0.1], [0.15, 0.1]]) + 1e-13).all()  # saturation depths


def test_saturation_return_goes_to_the_surface_not_into_the_soil():
    edges, res, fc, sat = [[0.0, 0.2], [0.0, 0.1]], [[0.02], [0.02]], [[0.3], [0.3]], [[0.5], [0.9]]  # saturation depths 0.10 and 0.09
    col = prepare(spec_of(edges, res, fc, sat, np.zeros((2, 0)), gravity=False))
    out = step(col, [[0.08], [0.01]], added=[[0.2], [0.05]])
    np.testing.assert_allclose(out.state.water_m[0], [0.10, 0.06], atol=1e-13)
    np.testing.assert_allclose(out.surface_return_m, [0.18, 0.0], atol=1e-13)  # 0.08 + 0.2 - 0.10 returned; 0.01 + 0.05 fits


@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
def test_exact_pair_relaxation_at_unequal_widths_in_both_directions(mode):
    # two layers, water well below FC (no cascade), no evaporation: the pair relaxes exponentially, d(theta_i - theta_j)/dt = -G s (theta_i - theta_j), G = D/centre separation, s = 1/dz_i + 1/dz_j.
    edges = [[0.0, 0.1, 0.5], [0.0, 0.3, 0.4]]  # widths (0.1, 0.4) and (0.3, 0.1)
    res, fc, sat = [[0.0, 0.0]] * 2, [[0.3, 0.3]] * 2, [[0.5, 0.5]] * 2
    theta = np.array([[0.25, 0.05], [0.12, 0.28]])  # cell 0 relaxes downward, cell 1 upward
    dz = np.array([[0.1, 0.4], [0.3, 0.1]])
    D = np.array([[1e-6], [5e-7]])  # constant mode: diffusivity
    d_coef, K = 0.3, np.array([[1e-6], [2e-6]])
    if mode == "constant":
        spec = spec_of(edges, res, fc, sat, D, mode=mode)
        sep = (dz[:, 0] + dz[:, 1]) / 2.0
        G_down = G_up = D[:, 0] / sep
    else:
        spec = spec_of(edges, res, fc, sat, K, mode=mode)
        sep = (dz[:, 0] + dz[:, 1]) / 2.0  # donor depth law: D = d K h_donor, direction from the gradient sign
        G_down, G_up = d_coef * K[:, 0] * dz[:, 0] / sep, d_coef * K[:, 0] * dz[:, 1] / sep
    col = prepare(spec)
    dt = 1.0e5
    out = step(col, theta * dz, dt=dt)
    W = (theta * dz).sum(axis=1)
    for c in range(2):
        s = 1.0 / dz[c, 0] + 1.0 / dz[c, 1]
        G = G_down[c] if theta[c, 0] > theta[c, 1] else G_up[c]
        delta1 = (theta[c, 0] - theta[c, 1]) * math.exp(-G * s * dt)
        tj = (W[c] - dz[c, 0] * delta1) / (dz[c, 0] + dz[c, 1])
        want = np.array([(tj + delta1) * dz[c, 0], tj * dz[c, 1]])
        np.testing.assert_allclose(out.state.water_m[:, c], want, rtol=1e-12, atol=1e-15)
        transfer = (theta[c, 0] - theta[c, 1]) * (1.0 - math.exp(-G * s * dt)) / s
        assert out.diffusion_m[0, c] == pytest.approx(transfer, rel=1e-12)  # positive downward
    assert abs(out.budget.residual_m3) <= out.budget.bound_m3


def test_zero_water_with_a_positive_residual_stays_zero_and_no_residual_water_is_ever_added():
    col = prepare(spec_of(PAIR["edges"], [[0.05] * 3, [0.08] * 3], PAIR["fc"], PAIR["sat"], ZERO_D))
    out = step(col, [[0.0] * 3, [0.0] * 3], dt=3600.0, pet=[1e-6, 5e-7])
    assert not out.state.water_m.any() and not out.evaporation_m.any() and not out.drainage_m.any() and not out.evaporation_unmet_m.any()


def test_evaporation_is_limited_by_the_withdrawal_floor_not_by_a_minimum_storage():
    edges, res, fc, sat = [[0.0, 0.1]] * 4, [[0.02]] * 4, [[0.3]] * 4, [[0.9]] * 4  # residual depth 0.002, saturation depth 0.09
    col = prepare(spec_of(edges, res, fc, sat, np.zeros((4, 0)), gravity=False))
    water = [[0.0031], [0.0], [0.055], [0.055]]
    pet = [0.1, 0.1, 0.1, 1e-6]
    out = step(col, water, dt=100.0, pet=pet)
    mult0 = (0.031 / 0.5) ** 2  # theta0 below the stress threshold 0.5: ratio squared
    demand0 = 0.1 * 100.0 * mult0
    # cell 0: demand >> available (0.0031 - 0.002 = 0.0011): evaporates exactly the available water, ending AT the residual depth, never below
    assert out.evaporation_m[0] == pytest.approx(0.0011, rel=1e-12) and out.state.water_m[0, 0] == pytest.approx(0.002, abs=1e-15)
    assert out.evaporation_unmet_m[0] == pytest.approx(demand0 - 0.0011, rel=1e-10)
    assert out.evaporation_m[1] == 0.0 and out.state.water_m[0, 1] == 0.0  # dry: nothing to withdraw, nothing added
    assert out.evaporation_m[2] == pytest.approx(0.055 - 0.002, rel=1e-12) and out.state.water_m[0, 2] == pytest.approx(0.002, abs=1e-15)
    assert out.evaporation_m[3] == pytest.approx(1e-6 * 100.0, rel=1e-12) and out.state.water_m[0, 3] == pytest.approx(0.055 - 1e-4, rel=1e-12)  # unlimited demand 1e-4


def test_post_cap_share_law_debits_only_limited_times_share():
    edges, res, fc, sat = [[0.0, 0.1]], [[0.02]], [[0.3]], [[0.9]]
    col = prepare(spec_of(edges, res, fc, sat, np.zeros((1, 0)), gravity=False))
    share = np.array([0.5])
    out = step(col, [[0.0031]], dt=100.0, pet=[0.1], share=share)
    demand = 0.1 * 100.0 * (0.031 / 0.5) ** 2
    limited = 0.0031 - 0.002
    assert out.evaporation_m[0] == pytest.approx(limited * 0.5, rel=1e-12)
    assert out.state.water_m[0, 0] == pytest.approx(0.0031 - limited * 0.5, rel=1e-12)  # the (1 - share) remainder stays in the layer
    assert out.evaporation_unmet_m[0] == pytest.approx(demand * 0.5 - limited * 0.5, rel=1e-9)


def test_pulse_step_open_bottom_drains_everything_above_field_capacity_and_closed_bottom_cannot_fit():
    full = [[0.05, 0.10, 0.15], [0.10, 0.10, 0.10]]  # both cells saturated
    p = 1e-3
    open_col = prepare(spec_of(PAIR["edges"], PAIR["res"], PAIR["fc"], PAIR["sat"], ZERO_D, bottom="open"))
    out = step(open_col, full, dt=86400.0, recharge=p)
    np.testing.assert_allclose(out.state.water_m.T, [[0.02, 0.04, 0.06], [0.02, 0.06, 0.04]], rtol=0, atol=1e-13)  # every layer ends at its field capacity
    np.testing.assert_allclose(out.drainage_m, [0.03 + 0.06 + 0.09 + p, 0.08 + 0.04 + 0.06 + p], rtol=0, atol=1e-13)  # sum(saturation - FC) + the pulse
    closed_col = prepare(spec_of(PAIR["edges"], PAIR["res"], PAIR["fc"], PAIR["sat"], ZERO_D, bottom="closed"))
    with pytest.raises(ss.SubsurfaceFitFailure):
        step(closed_col, full, dt=86400.0, recharge=p)


def test_pulse_enters_only_active_cells():
    mask = np.array([True, False])
    col = prepare(spec_of(PAIR["edges"], PAIR["res"], PAIR["fc"], PAIR["sat"], ZERO_D, bottom="open"), mask=mask)
    out = step(col, [[0.02, 0.04, 0.06], [0.02, 0.06, 0.04]], dt=86400.0, recharge=0.01)
    np.testing.assert_array_equal(out.state.water_m[:, 1], [0.02, 0.06, 0.04])  # the inactive cell keeps its water bit for bit and receives nothing
    assert out.drainage_m[1] == 0.0 and out.drainage_m[0] == pytest.approx(0.01, rel=1e-12)
