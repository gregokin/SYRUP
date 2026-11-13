"""Root's independent exact controls: gravity over partial/zero porous slots (Fraction), partial-pair relaxation (Decimal80) and the staged reconciliation capacity controls (exact rationals)."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import BIG, backend, capacity_controls, column, frac, geometry_controls, zero_forcing  # noqa: F401
from decimal import Decimal, localcontext

from maple_syrup import floating_soil_layers as fl

GRAV = geometry_controls()["gravity_controls"]
DIFF = geometry_controls()["diffusion_controls"]


@pytest.mark.parametrize("c", GRAV, ids=lambda c: "D" + c["soil_depth_m"])
def test_gravity_controls_water_percolation_and_bottom_drainage(c, backend):
    edges = [float(x) for x in c["nominal_edges_m"]]
    col = column(edges, [float(c["soil_depth_m"])], tf=0.2, ts=0.4, bottom="open", backend=backend)
    # the effective bottom is open only above rock (strict): the control states it explicitly
    assert bool(col._maps["bottom_open"][0]) == c["effective_open_bottom"]
    w = np.array([[float(x)] for x in c["initial_water_m"]])
    st = fl.floating_continued_state(col, w)
    r = fl.floating_step(col, st, 1.0, **zero_forcing(col))
    exp = [float(x) for x in c["expected_water_after_gravity_m"]]
    np.testing.assert_allclose(r.state.water_m[:, 0], exp, rtol=0, atol=1e-15)
    np.testing.assert_allclose(r.percolation_m[:, 0], [float(x) for x in c["percolation_m"]], rtol=0, atol=1e-15)
    assert abs(float(r.drainage_m[0]) - float(c["bottom_drainage_m"])) <= 1e-15
    assert r.surface_return_m[0] == 0.0 and r.evaporation_m[0] == 0.0
    # exact rational conservation of the control itself, and the float step closes within MAPLE's bound
    assert sum(frac(x) for x in c["initial_water_m"]) == sum(frac(x) for x in c["expected_water_after_gravity_m"]) + frac(c["bottom_drainage_m"])
    assert abs(r.budget.residual_m3) <= r.budget.bound_m3
    assert c["all_rock"] == (float(c["soil_depth_m"]) == 0.0) and (not c["all_rock"] or (r.state.water_m == 0.0).all())


@pytest.mark.parametrize("c", DIFF, ids=lambda c: c["label"])
def test_partial_pair_relaxation_against_decimal80(c, backend):
    th = [float(x) for x in c["thickness_m"]]
    edges = [0.0, th[0], th[0] + th[1]]
    if c["mode"] == "constant":
        col = column(edges, [edges[-1] + 1.0], mode="constant", iface=np.array([float(c["interface_diffusivity_m2_s"])]), tf=float(c["fc_theta"]), ts=float(c["sat_theta"]), tr=float(c["residual_theta"]), backend=backend)
    else:
        col = column(edges, [edges[-1] + 1.0], mode="tietjen_donor_depth", iface=np.array([float(c["interface_conductivity_m_s"])]), d=float(c["diffusion_coefficient"]), tf=float(c["fc_theta"]),
                     ts=float(c["sat_theta"]), tr=float(c["residual_theta"]), backend=backend)
    w = np.array([[float(t) * h] for t, h in zip(c["theta"], th)])
    st = fl.floating_continued_state(col, w)
    r = fl.floating_step(col, st, float(c["dt_s"]), **zero_forcing(col))
    with localcontext() as ctx:
        ctx.prec = 80
        want = [Decimal(x) for x in c["expected_water_m"]]
        signed = Decimal(c["expected_signed_transfer_m"])
    got = r.state.water_m[:, 0]
    for g, wnt in zip(got, want):
        assert abs(Decimal(float(g)) - wnt) <= Decimal("2e-12") * abs(wnt) + Decimal("1e-18")
    assert abs(Decimal(float(r.diffusion_m[0, 0])) - signed) <= Decimal("2e-12") * abs(signed)
    assert c["gravity_no_op"] and float(r.percolation_m[0, 0]) == 0.0 and r.drainage_m[0] == 0.0


def run_capacity(c, backend):
    """Layers of 1 m nominal thickness so that theta = capacity(mm)/1000; rock slots are produced by the soil depth (zero capacity), closed/open by configuration and depth."""
    n = len(c["initial_water"])
    edges = [float(k) for k in range(n + 1)]
    fc_new = [float(x) / 1000.0 if float(s_) > 0.0 else 0.1 for x, s_ in zip(c["new_fc"], c["new_sat"])]  # a rock slot carries no property: any valid theta (its capacity is theta * 0)
    sat_new = [float(x) / 1000.0 if float(x) > 0.0 else 0.2 for x in c["new_sat"]]
    # rock slots: where the new saturation is 0 the layer must hold nothing: it lies below the soil depth
    n_soil = sum(1 for x in c["new_sat"] if float(x) > 0.0)
    depth = float(n_soil) if not c["open_above_bedrock"] else float(n) + 1.0
    # the old column: everything porous (large capacities) so the old water is a valid state of it
    old = column(edges, [float(n) + 1.0], tf=0.9, ts=0.95, bottom="closed", backend=backend)
    w = np.array([[float(x) / 1000.0] for x in c["initial_water"]])
    old_state = fl.floating_continued_state(old, w)
    new = column(edges, [depth], tf=np.array(fc_new), ts=np.array(sat_new), tr=0.0, bottom="open" if c["open_above_bedrock"] else "closed", backend=backend)
    assert bool(new._maps["bottom_open"][0]) == c["open_above_bedrock"]
    return old, old_state, new, fl.reconcile_floating(old, old_state, new)


@pytest.mark.parametrize("c", capacity_controls(), ids=lambda c: c["label"])
def test_staged_reconciliation_matches_the_exact_rational_controls(c, backend):
    old, old_state, new, cand = run_capacity(c, backend)
    np.testing.assert_allclose(cand.state.water_m[:, 0] * 1000.0, [float(x) for x in c["expected_water"]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(cand.percolation_m[:, 0] * 1000.0, [float(x) for x in c["expected_percolation"]], rtol=0, atol=1e-12)
    assert abs(cand.bottom_export_m[0] * 1000.0 - float(c["expected_bottom_export"])) <= 1e-12
    assert abs(cand.surface_return_m[0] * 1000.0 - float(c["expected_surface_return"])) <= 1e-12
    r = cand.receipt
    assert abs(r.residual_m3) <= r.bound_m3 and r.t_s == old_state.t_s == cand.state.t_s  # zero time passes
    assert c["exact_balance"] and sum(frac(x) for x in c["initial_water"]) == sum(frac(x) for x in c["expected_water"]) + frac(c["expected_bottom_export"]) + frac(c["expected_surface_return"])
