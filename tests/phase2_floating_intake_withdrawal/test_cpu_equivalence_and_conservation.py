"""NumPy/Numba equivalence over layer counts (1/2/3/65) and block edges, conservation ledgers, state policies and unchanged-input-on-failure for the three floating laws (CPU/host only)."""
from __future__ import annotations

import numpy as np
import pytest
from builders import REFUSALS, column, state

from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_preferential_infiltration as fp
from maple_syrup import floating_soil_layers as fl
from maple_syrup import floating_transpiration as ft

try:
    import numba  # noqa: F401
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
needs_numba = pytest.mark.skipif(not HAVE_NUMBA, reason="Numba not importable")


def edges_for(L):
    return np.concatenate([[0.0], np.cumsum(np.full(L, 0.6 / L))])


def pair_cpu(L, N, seed, backend):
    rng = np.random.default_rng(seed)
    depth = rng.choice([0.0, 1e-6, 0.03, 0.1, 0.2, 0.45, 0.6, 0.9], size=N)
    act = rng.random(N) > 0.2
    cols = [column(edges_for(L), depth, tr=0.05, tf=0.3, ts=0.45, active=act, backend=b, stress=0.2) for b in ("numpy", backend)]
    h = np.stack([fl.porous_thickness(cols[0], k) for k in range(L)])
    w = h * 0.45 * rng.uniform(0.05, 0.95, h.shape)
    w[:, ~act] = rng.uniform(0.0, 2.0, (L, int((~act).sum())))
    return cols, w, act, rng


@needs_numba
@pytest.mark.parametrize("L,N", [(1, 5), (2, 127), (3, 128), (3, 129), (65, 260)])
def test_numpy_and_numba_agree_for_the_three_laws(L, N):
    (a, b), w, act, rng = pair_cpu(L, N, L * 91 + N, "numba")
    s_a, s_b = state(a, w), state(b, w)
    q = rng.uniform(0.0, 0.01, (L, N)) * act
    ra, rb = ft.floating_withdraw(a, s_a, q), ft.floating_withdraw(b, s_b, q)
    assert np.array_equal(ra.state.water_m, rb.state.water_m) and np.array_equal(ra.actual_m, rb.actual_m)
    pool = rng.uniform(0.0, 0.02, N) * act
    standing = np.where(rng.random(N) > 0.5, rng.uniform(0.0, 0.01, N), 0.0) * act
    rate = np.where(standing > 0.0, rng.uniform(0.0, 1e-5, N), pool / 600.0 * rng.uniform(1.0, 3.0, N)) * act
    oa, ob = (fm.prepare_floating_matrix(c, conductivity_m_s=2e-6, suction_m=0.15, vegetation_factor=0.8) for c in (a, b))
    j0, c0 = rng.uniform(0.0, 0.02, N), rng.uniform(0.0, 0.5, N)
    ma = fm.floating_matrix_step(oa, pool, standing, rate, s_a, fm.floating_matrix_history(oa, wetting_m=j0, cumulative_intake_m=c0), 600.0)
    mb = fm.floating_matrix_step(ob, pool, standing, rate, s_b, fm.floating_matrix_history(ob, wetting_m=j0, cumulative_intake_m=c0), 600.0)
    for x, y in ((ma.state.water_m, mb.state.water_m), (ma.surface_pool_m, mb.surface_pool_m), (ma.history.wetting_m, mb.history.wetting_m), (ma.intake_m, mb.intake_m)):
        assert np.allclose(x, y, rtol=2e-12, atol=1e-14)
    weights = np.linspace(0.0, 1.0, L)
    pa, pb = (fp.prepare_floating_preferential(c, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=weights) for c in (a, b))
    sa_, sb_ = fp.floating_preferential_step(pa, pool, s_a, 1800.0), fp.floating_preferential_step(pb, pool, s_b, 1800.0)
    for x, y in ((sa_.state.water_m, sb_.state.water_m), (sa_.surface_water_m, sb_.surface_water_m), (sa_.intake_m, sb_.intake_m)):
        assert np.allclose(x, y, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("L,N", [(1, 5), (3, 129), (65, 130)])
def test_conservation_ledgers_close_and_exclude_inactive_placeholders(backend, L, N):
    cols, w, act, rng = pair_cpu(L, N, L + N, backend)
    c = cols[1]
    area = float(c.cell_area_m2)
    s = state(c, w)
    active_storage = float(w[:, act].sum()) * area
    pool = rng.uniform(0.0, 0.02, N) * act
    op = fm.prepare_floating_matrix(c, conductivity_m_s=3e-6, suction_m=0.1, vegetation_factor=0.7)
    r = fm.floating_matrix_step(op, pool, np.zeros(N), pool / 600.0 * 2.0, s, fm.floating_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0), 600.0)
    b = r.budget
    assert b.soil_before_m3 == pytest.approx(active_storage, rel=1e-12, abs=1e-18)  # the placeholders (water up to 2 m) are excluded from the active budget
    assert abs(b.soil_residual_m3) <= b.bound_m3 and abs(b.pool_residual_m3) <= b.bound_m3
    assert float(r.intake_m.sum()) * area == pytest.approx(b.intake_m3, rel=1e-12, abs=1e-18)
    q = rng.uniform(0.0, 0.01, (L, N)) * act
    t = ft.floating_withdraw(c, s, q)
    assert abs(t.budget.residual_m3) <= t.budget.bound_m3 and abs(t.budget.request_residual_m3) <= t.budget.bound_m3
    assert t.budget.actual_m3 <= t.budget.requested_m3 + t.budget.bound_m3
    # actual <= request and <= availability, layer by layer
    av = ft.floating_layer_availability_m(c, s)
    assert np.all(t.actual_m <= q + 1e-18) and np.all(t.actual_m <= av + 1e-18)


def test_saturated_continuation_is_accepted_but_unsaturated_initialisation_is_required(backend):
    c = column([0.0, 0.1, 0.3, 0.6], [0.15], tr=0.1, tf=0.3, ts=0.5, backend=backend, stress=0.2)
    full = np.array([[0.05], [0.025], [0.0]])  # layer 0 and 1 exactly at saturation
    fl.floating_continued_state(c, full, 0.0)
    with pytest.raises(fl.FloatingSoilError):
        fl.floating_initial_state(c, full, 0.0)
    s = fl.floating_continued_state(c, full, 0.0)
    r = fp.floating_preferential_step(fp.prepare_floating_preferential(c, a3=1.0, max_fraction_per_hour=1.0, max_intake_rate_m_s=float("inf"), background_receiver_weights=np.array([1.0, 1.0, 1.0])),
                                      np.array([0.01]), s, 100.0)
    assert np.array_equal(r.state.water_m, full) and r.surface_water_m[0] == 0.01  # a full column admits nothing and nothing is lost


def test_wrong_owner_and_state_refusals(backend):
    c1 = column([0.0, 0.1, 0.3, 0.6], [0.15], tr=0.1, tf=0.3, ts=0.5, backend=backend, stress=0.2)
    c2 = column([0.0, 0.1, 0.3, 0.6], [0.2], tr=0.1, tf=0.3, ts=0.5, backend=backend, stress=0.2)
    s2 = state(c2, np.zeros((3, 1)))
    with pytest.raises(fl.FloatingSoilError, match="different floating column"):
        ft.floating_withdraw(c1, s2, np.zeros((3, 1)))
    with pytest.raises(fl.FloatingSoilError):
        ft.floating_withdraw("not a column", s2, np.zeros((3, 1)))
    op = fm.prepare_floating_matrix(c1, conductivity_m_s=1e-6, suction_m=0.0, vegetation_factor=0.5)
    with pytest.raises(REFUSALS):
        fm.floating_matrix_step(op, np.zeros(1), np.zeros(1), np.zeros(1), s2, fm.floating_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0), 1.0)
