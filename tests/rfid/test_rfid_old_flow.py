"""Complete-infiltration branch: the legacy `infilt.for` 112-115 sets d(1) = 0. A tiny depth absorbed by the rain in `h + rain`
must therefore give a zero old-flow depth in the reference, prepared Numba and CUDA paths (root reproducer:
hi = 8.378794223761067e-76). Ordinary complete / partial / no-run-on cells stay at parity. Nothing here was run by its author."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("maple")

from cand_cases import host, make_params, param_arrays
from rfid_helpers import build, chain_full

from maple_syrup import hydrology_numba as hn
from maple_syrup.storm import StormControl, coupled_step, initial_state

TINY = 8.378794223761067e-76
NUMBA = pytest.mark.skipif(not hn.numba_available(), reason="Numba not installed; compiled path not exercised")


def case(ksat, depth, soil_fraction=0.0, xp=np, **param_kw):
    g = build(chain_full(6), xp=xp)
    params = make_params(param_arrays(g.shape, ksat=ksat, **param_kw), np.asarray(g.active), xp=xp)
    d = xp.asarray(np.array(depth, dtype=np.float64).reshape(g.shape))
    s = xp.asarray(soil_fraction * np.asarray(host(params.storage_max_m)))
    return g, params, d, s


TINY_DEPTH = [0.0, TINY, 0.0, TINY, 1e-300, 0.0]


def test_tiny_depth_absorbed_by_rain_gives_zero_old_flow_in_the_reference():
    g, params, d, s = case(1.0e-3, TINY_DEPTH)
    state = initial_state(g, d, s)
    rate = np.full(g.shape, 1.0e-6)
    step = coupled_step(g, params, rate, state, 0.5, StormControl(max_dt_s=0.5))  # used to raise on h_old > h_start
    assert int(step.n_complete_runon) == 6
    assert np.all(np.asarray(step.column.depth_m) == 0.0)  # the column arithmetic is unchanged: nothing ponds
    assert np.all(np.asarray(step.route.old_discharge_m2_s) == 0.0)


def test_the_old_flow_check_still_refuses_a_genuinely_inconsistent_old_depth():
    """The guard is not relaxed: route_step still refuses h_old > h_start."""
    from maple_syrup.routing import RoutingError, route_step

    g, *_ = case(1.0e-3, TINY_DEPTH)
    start = np.zeros(g.shape)
    old = np.zeros(g.shape)
    old[2, 0] = 1.0e-6
    with pytest.raises(RoutingError, match="old_flow_depth_m"):
        route_step(g, start, old, 0.5)


@NUMBA
def test_prepared_numba_matches_the_reference_on_the_tiny_depth_reproducer():
    g, params, d, s = case(1.0e-3, TINY_DEPTH)
    ctx = hn.prepare_hydrology(g, params)
    rate = np.full(g.shape, 1.0e-6)
    ref = coupled_step(g, params, rate, initial_state(g, d, s), 0.5, StormControl(max_dt_s=0.5))
    new = hn.prepared_coupled_step(ctx, rate, initial_state(g, d, s), 0.5, StormControl(max_dt_s=0.5, implementation="numba"))
    assert int(new.n_complete_runon) == int(ref.n_complete_runon) == 6
    np.testing.assert_array_equal(new.state.depth_m, ref.state.depth_m)
    np.testing.assert_array_equal(new.route.old_discharge_m2_s, ref.route.old_discharge_m2_s)


@NUMBA
@pytest.mark.parametrize("ksat, depth, soil", [
    (1.0e-3, [0.0, 1e-9, 1e-4, 1e-3, 2e-3, 3e-3], 0.0),   # complete run-on
    (2.0e-6, [0.0, 1e-9, 1e-4, 1e-3, 2e-3, 3e-3], 0.5),   # partial / no run-on mix
    (5.0e-7, [3e-3, 2e-3, 1e-3, 1e-4, 1e-9, 0.0], 0.9),
])
def test_ordinary_complete_and_partial_cells_keep_parity(ksat, depth, soil):
    g, params, d, s = case(ksat, depth, soil)
    ctx = hn.prepare_hydrology(g, params)
    rate = np.full(g.shape, 1.0e-6)
    ref_state = new_state = initial_state(g, d, s)
    for _ in range(5):
        ref = coupled_step(g, params, rate, ref_state, 0.5, StormControl(max_dt_s=0.5))
        new = hn.prepared_coupled_step(ctx, rate, new_state, 0.5, StormControl(max_dt_s=0.5, implementation="numba"))
        assert (int(new.n_no_runon), int(new.n_partial_runon), int(new.n_complete_runon)) == \
            (int(ref.n_no_runon), int(ref.n_partial_runon), int(ref.n_complete_runon))
        np.testing.assert_allclose(new.state.depth_m, ref.state.depth_m, rtol=2e-12, atol=1e-14)
        np.testing.assert_allclose(new.state.soil_water_m, ref.state.soil_water_m, rtol=2e-12, atol=1e-14)
        ref_state, new_state = ref.state, new.state


def test_cuda_tiny_depth_and_ordinary_parity(gpu):
    cp = gpu
    for ksat, depth, soil in ((1.0e-3, TINY_DEPTH, 0.0), (2.0e-6, [0.0, 1e-9, 1e-4, 1e-3, 2e-3, 3e-3], 0.5)):
        g, params, d, s = case(ksat, depth, soil)
        gd, pd, dd, sd = case(ksat, depth, soil, xp=cp)
        rate = np.full(g.shape, 1.0e-6)
        ref = coupled_step(g, params, rate, initial_state(g, d, s), 0.5, StormControl(max_dt_s=0.5))
        dev = coupled_step(gd, pd, cp.asarray(rate), initial_state(gd, dd, sd), 0.5,
                           StormControl(max_dt_s=0.5, implementation="cuda"))
        assert int(host(dev.n_complete_runon)) == int(ref.n_complete_runon)
        np.testing.assert_allclose(host(dev.state.depth_m), ref.state.depth_m, rtol=2e-12, atol=1e-14)
        np.testing.assert_allclose(host(dev.route.old_discharge_m2_s), ref.route.old_discharge_m2_s, rtol=2e-12, atol=1e-14)


def test_column_depth_and_old_flow_share_one_arithmetic_on_partial_equal_and_tiny_cells():
    """Root reproducer: an ordinary partial cell gave hpre 9.949999999999988e-05 > column depth 9.949999999999987e-05 (1 ulp) because
    h + P - J and h - (J - P) were rounded independently. The column depth now uses the hpre arithmetic. Covers
    partial, complete and tiny-h cells (exact intake == rain has its own tests below); the unchanged per-cell/global balances (validate=True) still pass."""
    from maple_syrup.infiltration import column_step

    g, params, d, s = case(2.0e-6, [0.0, 9.95e-5, 1e-4, 1e-3, TINY, 3e-3], 0.5)
    for rate in (1.0e-6, 3.0e-6, 5.0e-4):
        r = np.full(g.shape, rate)
        col = column_step(params, d, s, r, 0.5)
        state = initial_state(g, d, s)
        coupled_step(g, params, r, state, 0.5, StormControl(max_dt_s=0.5))  # must not raise the old-flow check
        total = np.asarray(col.depth_m) + np.asarray(col.soil_water_m) + np.asarray(col.drainage_m)
        np.testing.assert_allclose(total, np.asarray(d) + np.asarray(s) + np.asarray(col.rain_m), rtol=1e-14, atol=1e-18)


def _equality_case(xp=np):
    """capacity == Ksat exactly (suction 0, a drier-than-full column so x = S/((psi+h) deficit) >> 40 and no saturation return),
    Ksat == rain rate, so intake = min(h + P, Ksat dt) = P bit-for-bit while h > 0 is resolvable: the NO-RUN-ON branch."""
    return case(1.0e-6, [1e-3] * 6, 5.0 / 6.0, xp=xp, suction=0.0)


def test_exact_intake_equals_rain_is_the_no_runon_branch_in_reference_and_numba():
    g, params, d, s = _equality_case()
    rate = np.full(g.shape, 1.0e-6)
    from maple_syrup.infiltration import column_step

    col = column_step(params, d, s, rate, 0.5)
    assert np.all(np.asarray(col.intake_m) == np.asarray(col.rain_m)) and np.all(np.asarray(col.saturation_return_m) == 0.0)
    ref = coupled_step(g, params, rate, initial_state(g, d, s), 0.5, StormControl(max_dt_s=0.5))
    assert int(ref.n_no_runon) == 6 and int(ref.n_complete_runon) == 0 and int(ref.n_partial_runon) == 0
    np.testing.assert_array_equal(np.asarray(col.depth_m), np.asarray(d))  # retained h, excess 0: depth is exactly h
    if hn.numba_available():
        new = hn.prepared_coupled_step(hn.prepare_hydrology(g, params), rate, initial_state(g, d, s), 0.5,
                                       StormControl(max_dt_s=0.5, implementation="numba"))
        assert int(new.n_no_runon) == 6
        np.testing.assert_allclose(new.state.depth_m, ref.state.depth_m, rtol=2e-12, atol=1e-14)


def test_exact_intake_equals_rain_cuda_matches_the_reference(gpu):
    cp = gpu
    g, params, d, s = _equality_case()
    gd, pd, dd, sd = _equality_case(xp=cp)
    rate = np.full(g.shape, 1.0e-6)
    ref = coupled_step(g, params, rate, initial_state(g, d, s), 0.5, StormControl(max_dt_s=0.5))
    dev = coupled_step(gd, pd, cp.asarray(rate), initial_state(gd, dd, sd), 0.5, StormControl(max_dt_s=0.5, implementation="cuda"))
    assert int(host(dev.n_no_runon)) == 6
    np.testing.assert_allclose(host(dev.state.depth_m), ref.state.depth_m, rtol=2e-12, atol=1e-14)
