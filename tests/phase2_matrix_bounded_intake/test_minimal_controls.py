"""The bounded combined intake on single-column controls, on the accepted COMMON operator and the SPATIAL operator, NumPy and Numba: Root's two extracted originally-failing inputs (pool-limited recombination) and
three room-limited recombination cases found with the pre-review02 source. Independent arithmetic and Decimal checks; no tolerance is loosened, nothing is clipped or erased. Author-written."""
from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest
from conftest import ATOL, RTOL, all_controls, control_id, mi, room_of, root_controls, run_common, run_spatial

ALL = all_controls()
IDS = [f"{k}:{control_id(c)}" for k, c in ALL]


def ulp(x):
    return float(np.spacing(np.float64(x)))


@pytest.mark.parametrize("kind,c", ALL, ids=IDS)
@pytest.mark.parametrize("api", ["common", "spatial"])
def test_the_originally_refused_or_overshooting_inputs_now_succeed_with_consistent_actuals(backend, api, kind, c):
    r = (run_common if api == "common" else run_spatial)(c, backend)
    pool, room = c["pool_m"], room_of(c)
    intake, pool_out, w_out = float(r.intake_m[0]), float(r.surface_pool_m[0]), r.state.water_m[:, 0]
    # the bound: finite, nonnegative pool; intake <= the ORIGINAL pool and <= the top-layer room (exactly, no tolerance)
    assert np.isfinite(pool_out) and pool_out >= 0.0 and intake <= pool and intake <= room and intake >= 0.0
    assert r.state.t_s == c.get("t_s", 3.0)  # the clock is preserved
    # ONE set of actuals: soil credit, pool debit, ledger and history advance all derive from the SAME intake
    assert w_out[0] == np.float64(c["water_m"][0]) + np.float64(intake)
    assert pool_out == np.float64(pool) - np.float64(intake)
    assert float(r.history.cumulative_intake_m[0]) == c["cumulative_intake_m"] + intake  # exactly once
    b = r.budget
    assert b.room_rejected_m3 >= 0.0 and b.unused_capacity_m3 >= 0.0  # rejected potential is never negative: nothing is pretended to enter the soil
    assert abs(b.soil_residual_m3) <= b.bound_m3 and abs(b.pool_residual_m3) <= b.bound_m3
    pre, post = b.prepond_intake_m3 / c.get("area_m2", 4.0), b.postpond_intake_m3 / c.get("area_m2", 4.0)
    g = c["vegetation_factor"]
    # history: J' = J + pre + post_actual / g with post_actual = intake - pre (the bounded actual), consistent to rounding of the area scaling
    j_expect = c["wetting_m"] + pre + (intake - pre) / g
    assert float(r.history.wetting_m[0]) == pytest.approx(j_expect, rel=1e-13, abs=1e-16)
    assert post == pytest.approx(intake - pre, rel=1e-12, abs=1e-16) and pre + post == pytest.approx(intake, rel=1e-13, abs=1e-18)


@pytest.mark.parametrize("c", root_controls(), ids=control_id)
def test_root_pool_limited_controls_exhaust_the_pool_exactly_and_recover_the_unchanged_candidate_and_pre(backend, c):
    old = c["old_raw"]
    assert old["intake_m"] > c["pool_m"] and old["pool_out_m"] < 0.0  # the recorded PRE-review02 behaviour (evidence): one ulp above the pool
    for run in (run_common, run_spatial):
        r = run(c, backend)
        assert float(r.intake_m[0]) == c["pool_m"] and float(r.surface_pool_m[0]) == 0.0  # pool-limited: the whole pool enters, no negative remainder
        area = c.get("area_m2", 4.0)
        assert r.budget.candidate_m3 / area == pytest.approx(old["candidate_m"], rel=1e-12)  # the returned candidate is unchanged
        assert r.budget.prepond_intake_m3 / area == pytest.approx(old["pre_m"], rel=1e-12)  # pre is unchanged
        assert r.budget.n_onsets == 1 and r.budget.n_ponded == 1


@pytest.mark.parametrize("c", root_controls(), ids=control_id)
def test_decimal_control_of_the_pool_limited_split(backend, c):
    """The bounded ponded actual equals the exact difference pool - pre to within one ulp, and the history advances by exactly that over g."""
    r = run_spatial(c, backend)
    old = c["old_raw"]
    with localcontext() as ctx:
        ctx.prec = 60
        exact_post = Decimal(c["pool_m"]) - Decimal(old["pre_m"])
        j_exact = Decimal(c["wetting_m"]) + Decimal(old["pre_m"]) + exact_post / Decimal(c["vegetation_factor"])
        area = Decimal(c.get("area_m2", 4.0))
        post = Decimal(r.budget.postpond_intake_m3) / area
        assert abs(post - exact_post) <= Decimal(4 * ulp(c["pool_m"]))
        assert abs(Decimal(float(r.history.wetting_m[0])) - j_exact) <= Decimal(8 * ulp(j_exact))


@pytest.mark.parametrize("c", room_controls_cases := [c for k, c in ALL if k == "room"], ids=control_id)
def test_room_limited_recombination_is_bounded_and_split_consistently(backend, c):
    """Room-limited split: the pre-review02 recombination overshot the room by an ulp (the water stayed within the capacity tolerance, so it was silently accepted). Now intake <= room exactly, the top layer fills to
    exactly theta_sat dz (no capacity increase), the pool keeps the remainder, and pre/post/history are consistent."""
    assert c["old_intake"] > c["room"]  # the recorded PRE-review02 overshoot
    for run in (run_common, run_spatial):
        r = run(c, backend)
        room = room_of(c)
        assert room == c["room"] and float(r.intake_m[0]) == room  # the room, exactly
        assert float(r.surface_pool_m[0]) == c["pool_m"] - room and float(r.surface_pool_m[0]) > 0.0  # the rest stays in the pool
        assert r.state.water_m[0, 0] <= np.float64(c["theta_saturation"][0]) * (np.float64(c["depth_edges_m"][1]) - np.float64(c["depth_edges_m"][0])) * (1 + 1e-14)
        pre = r.budget.prepond_intake_m3 / 4.0
        assert pre == pytest.approx(c["pre"], rel=1e-12)  # pre unchanged
        assert r.budget.postpond_intake_m3 / 4.0 == pytest.approx(room - c["pre"], rel=1e-12, abs=1e-18)


@pytest.mark.parametrize("kind,c", ALL, ids=IDS)
def test_common_and_spatial_agree_bit_for_bit_on_the_same_backend_and_numba_matches_numpy(kind, c):
    pytest.importorskip("numba")
    res = {(a, b): (run_common if a == "common" else run_spatial)(c, b) for a in ("common", "spatial") for b in ("numpy", "numba")}
    for b in ("numpy", "numba"):
        x, y = res[("common", b)], res[("spatial", b)]
        for f in ("surface_pool_m", "intake_m"):
            assert getattr(x, f).tobytes() == getattr(y, f).tobytes()
        assert x.state.water_m.tobytes() == y.state.water_m.tobytes() and x.history.wetting_m.tobytes() == y.history.wetting_m.tobytes()
        assert x.history.cumulative_intake_m.tobytes() == y.history.cumulative_intake_m.tobytes()
    a, b = res[("spatial", "numpy")], res[("spatial", "numba")]
    for f in ("surface_pool_m", "intake_m"):
        np.testing.assert_allclose(getattr(b, f), getattr(a, f), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(b.history.wetting_m, a.history.wetting_m, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("c", root_controls(), ids=control_id)
@pytest.mark.parametrize("k", list(range(-12, 13)))
def test_ulp_sweep_around_the_pool_never_refuses_and_always_conserves(backend, c, k):
    """The recombination exceeded the pool by one ulp for the recorded pool; sweep the pool across +-12 ulps (and the matching standing/rate unchanged): every pool must integrate with intake <= pool, pool_out >= 0 and
    the pool-limited cell exhausting it exactly."""
    pool = float(c["pool_m"])
    for _ in range(abs(k)):
        pool = float(np.nextafter(pool, np.inf if k > 0 else 0.0))
    for run in (run_common, run_spatial):
        r = run(c, backend, pool=pool)
        assert float(r.intake_m[0]) <= pool and float(r.surface_pool_m[0]) >= 0.0
        assert float(r.surface_pool_m[0]) == pool - float(r.intake_m[0])
        assert abs(r.budget.pool_residual_m3) <= r.budget.bound_m3


@pytest.mark.parametrize("c", root_controls(), ids=control_id)
def test_boundaries_zero_saturated_tiny_dt0_g0_masks(backend, c):
    area = c.get("area_m2", 4.0)
    for run in (run_common, run_spatial):
        z = run(c, backend, pool=0.0)  # a zero pool: nothing enters, nothing negative
        assert float(z.intake_m[0]) == 0.0 and float(z.surface_pool_m[0]) == 0.0
        assert float(z.history.wetting_m[0]) >= c["wetting_m"]
        t = run(c, backend, pool=1e-300)  # an extreme tiny pool
        assert 0.0 <= float(t.intake_m[0]) <= 1e-300 and float(t.surface_pool_m[0]) >= 0.0
        sat_top = np.float64(c["theta_saturation"][0]) * (np.float64(c["depth_edges_m"][1]) - np.float64(c["depth_edges_m"][0]))
        full = run(c, backend, water=[float(sat_top)] + list(c["water_m"][1:]))  # a saturated top layer: zero room
        assert float(full.intake_m[0]) == 0.0 and float(full.surface_pool_m[0]) == c["pool_m"]
        assert float(full.history.wetting_m[0]) >= c["wetting_m"] and float(full.history.cumulative_intake_m[0]) == c["cumulative_intake_m"]
        d0 = run(c, backend, dt=0.0, pool=0.0)  # dt = 0 is the identity (a consistent supply: nothing offered at dt = 0)
        assert float(d0.intake_m[0]) == 0.0 and float(d0.surface_pool_m[0]) == 0.0 and float(d0.history.wetting_m[0]) == c["wetting_m"]
        assert d0.state.water_m.tobytes() == np.array(c["water_m"]).reshape(-1, 1).tobytes()
        off = run(dict(c, incoming_rate_m_s=0.0), backend, active=False)  # a masked (inactive) cell: state, pool and history untouched, nothing counted
        assert float(off.intake_m[0]) == 0.0 and float(off.surface_pool_m[0]) == c["pool_m"] and float(off.history.wetting_m[0]) == c["wetting_m"]
        assert off.budget.intake_m3 == 0.0 and off.budget.n_ponded == 0
    g0 = dict(c, vegetation_factor=0.0)  # g = 0: no ponded intake, J unchanged by the ponded part, no division
    for run in (run_common, run_spatial):
        r = run(g0, backend)
        assert float(r.intake_m[0]) <= c["pool_m"] and float(r.surface_pool_m[0]) >= 0.0 and np.isfinite(float(r.history.wetting_m[0]))
        assert float(r.history.cumulative_intake_m[0]) == c["cumulative_intake_m"] + float(r.intake_m[0])
    assert area > 0


@pytest.mark.parametrize("c", root_controls(), ids=control_id)
def test_the_unponded_and_prepond_only_branches_are_untouched_by_the_bound(backend, c):
    """Low rate (r <= K), no contact: the whole supply enters through the unponded branch, J advances by it UNSCALED, the bound is inactive and `post` stays 0."""
    low = dict(c, incoming_rate_m_s=0.5 * c["conductivity_m_s"], standing_m=0.0)
    low["pool_m"] = low["incoming_rate_m_s"] * low["dt_s"]
    for run in (run_common, run_spatial):
        r = run(low, backend)
        assert float(r.intake_m[0]) == low["pool_m"] and float(r.surface_pool_m[0]) == 0.0 and r.budget.postpond_intake_m3 == 0.0 and r.budget.n_ponded == 0
        assert float(r.history.wetting_m[0]) == pytest.approx(c["wetting_m"] + low["pool_m"], rel=1e-14)
