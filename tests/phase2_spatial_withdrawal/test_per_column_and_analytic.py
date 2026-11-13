"""The fixed-spatial withdrawal/availability against the accepted COMMON operator column by column (bit-exact on W', actual, unmet and scale), across layouts, L = 1/2/3/5, floor/saturation/mixed states, masks and
request regimes; the dyadic hand case; availability against its definition; NumPy and Numba agree bitwise on the arrays. Author-written."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import AREA, BIG, LAYOUTS, a2, a3, bits, common_pair, spatial_owner, sp, ss, st, sw


def make_states(rng, p, kind):
    L, N = p["res"].shape
    dz = np.diff(p["edges"], axis=0)
    res, sat = p["res"] * dz, p["sat"] * dz
    if kind == "zero":
        return np.zeros((L, N))
    if kind == "at_floor":
        return res.copy()
    if kind == "below_floor":
        return res * rng.uniform(0.0, 1.0, (L, N))
    if kind == "saturated":
        return sat.copy()
    if kind == "mixed":
        w = sat * rng.uniform(0.0, 1.0, (L, N))
        w[:, 0::3], w[:, 1::3] = 0.0, res[:, 1::3]
        return w
    return sat * rng.uniform(0.0, 1.0, (L, N))


def requests(rng, w, p, mask, regime):
    L, N = w.shape
    res = p["res"] * np.diff(p["edges"], axis=0)
    avail = np.maximum(w - res, 0.0)
    q = {"zero": np.zeros((L, N)), "under": avail * 0.5, "equal": avail.copy(), "over": avail + 0.01, "random": rng.uniform(0, 0.05, (L, N))}[regime]
    return q * mask


def run_common(p, c, backend, w, q, mask, t):
    col, op = common_pair(p, c, backend, active=bool(mask[c]))
    r = st.withdraw_transpiration(op, ss.column_state(col, w[:, c : c + 1].copy(), t), q[:, c : c + 1].copy())
    return r


@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("kind", ["zero", "at_floor", "below_floor", "saturated", "mixed", "random"])
def test_each_column_is_bit_exact_against_the_accepted_common_operator(backend, L, kind):
    rng = np.random.default_rng(L * 7 + len(kind))
    N = 12
    p = a2.params(rng, L, N)
    mask = rng.random(N) < 0.75
    mask[0], mask[1] = True, False
    col = spatial_owner(p, backend, mask)
    op = sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
    w = make_states(rng, p, kind)
    state = sp.spatial_column_state(col, w, 1234.5, memory_budget_bytes=BIG)
    for regime in ("zero", "under", "equal", "over", "random"):
        q = requests(rng, w, p, mask, regime)
        got = sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)
        assert got.state.t_s == 1234.5 and got.state.column_sha256 == col.sha256 and got.state.namespace == "numpy"  # the clock is preserved
        for c in range(N):
            ref = run_common(p, c, backend, w, q, mask, 1234.5)
            for name, a, b in (("water", got.state.water_m, ref.state.water_m), ("actual", got.actual_m, ref.actual_m), ("unmet", got.unmet_m, ref.unmet_m), ("scale", got.scale, ref.scale)):
                np.testing.assert_array_equal(bits(a[:, c : c + 1]), bits(b), err_msg=f"{name} cell {c} {regime}")
        assert abs(got.budget.residual_m3) <= got.budget.bound_m3 and abs(got.budget.request_residual_m3) <= got.budget.bound_m3
        assert got.budget.n_terms == 3 * L * int(mask.sum()) + st._PACKET


@pytest.mark.parametrize("L", [1, 2, 5])
def test_mixed_scalar_shared_and_cell_layouts_are_bit_exact(backend, L):
    rng = np.random.default_rng(50 + L)
    N = 9
    mask = rng.random(N) < 0.8
    mask[0] = True
    combos = [("shared", "scalar", "scalar", "scalar", "shared"), ("cell", "cell", "cell", "cell", "cell"), ("shared", "cell", "shared", "scalar", "shared"), ("cell", "scalar", "cell", "shared", "scalar"),
              ("shared", "shared", "shared", "cell", "cell")]
    for edges, res, fc, sat, iface in combos:
        inputs, p = a3.layout_case(rng, L, N, edges=edges, res=res, fc=fc, sat=sat, iface=iface)
        col = a2.prepare(a3.spec_from(inputs, N), backend=backend, mask=mask)
        op = sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
        w = make_states(rng, p, "mixed")
        state = sp.spatial_column_state(col, w, 0.0, memory_budget_bytes=BIG)
        q = requests(rng, w, p, mask, "random")
        got = sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)
        for c in range(N):
            ref = run_common(p, c, backend, w, q, mask, 0.0)
            np.testing.assert_array_equal(bits(got.actual_m[:, c : c + 1]), bits(ref.actual_m))
            np.testing.assert_array_equal(bits(got.state.water_m[:, c : c + 1]), bits(ref.state.water_m))


def test_numpy_and_numba_owners_agree_bit_for_bit_on_every_array():
    pytest.importorskip("numba")
    rng = np.random.default_rng(3)
    for L in (1, 3, 9):
        N = 40
        p = a2.params(rng, L, N)
        mask = rng.random(N) < 0.8
        w = make_states(rng, p, "random")
        q = requests(rng, w, p, mask, "random")
        outs = []
        for be in ("numpy", "numba"):
            col = spatial_owner(p, be, mask)
            op = sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
            outs.append(sw.spatial_withdraw(op, sp.spatial_column_state(col, w, 0.0, memory_budget_bytes=BIG), q, memory_budget_bytes=BIG))
        for f in ("actual_m", "unmet_m", "scale"):
            np.testing.assert_array_equal(bits(getattr(outs[0], f)), bits(getattr(outs[1], f)))
        np.testing.assert_array_equal(bits(outs[0].state.water_m), bits(outs[1].state.water_m))
        for f in ("storage_before_m3", "storage_after_m3", "requested_m3", "actual_m3", "unmet_m3"):
            assert getattr(outs[0].budget, f) == pytest.approx(getattr(outs[1].budget, f), rel=1e-12, abs=1e-15)


def test_dyadic_delivery_example(backend):
    """W = [25, 200]/1024, q = [50, 50]/1024 with zero residual: actual [25, 50]/1024, remaining [0, 150]/1024, scale [0.5, 1], aggregate delivery 75 %. All values are dyadic: exact."""
    sat = np.array([[0.5], [0.5]])
    spec = sp.spatial_layer_spec(np.array([0.0, 1.0, 2.0]), theta_residual=0.0, theta_field_capacity=0.25, theta_saturation=sat[:, 0], theta_evaporation_stress=0.5, diffusivity_m2_s=0.0, n_cells=1)
    col = sp.prepare_spatial_column(spec, np.ones(1, bool), 4.0, backend=backend, memory_budget_bytes=BIG)
    op = sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
    w = np.array([[25.0], [200.0]]) / 1024.0
    q = np.array([[50.0], [50.0]]) / 1024.0
    r = sw.spatial_withdraw(op, sp.spatial_column_state(col, w, 7.0, memory_budget_bytes=BIG), q, memory_budget_bytes=BIG)
    np.testing.assert_array_equal(r.actual_m, np.array([[25.0], [50.0]]) / 1024.0)
    np.testing.assert_array_equal(r.state.water_m, np.array([[0.0], [150.0]]) / 1024.0)
    np.testing.assert_array_equal(r.unmet_m, np.array([[25.0], [0.0]]) / 1024.0)
    np.testing.assert_array_equal(r.scale, np.array([[0.5], [1.0]]))
    assert r.budget.actual_m3 / r.budget.requested_m3 == 0.75 and r.state.t_s == 7.0
    assert r.fulfillment_fraction is r.scale


def test_zero_request_gives_scale_zero_not_one_and_zero_water_is_legal(backend):
    rng = np.random.default_rng(1)
    p = a2.params(rng, 3, 6)
    col = spatial_owner(p, backend, np.ones(6, bool))
    op = sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
    r = sw.spatial_withdraw(op, sp.spatial_column_state(col, np.zeros((3, 6)), 0.0, memory_budget_bytes=BIG), np.zeros((3, 6)), memory_budget_bytes=BIG)
    assert not np.any(r.scale) and not np.any(r.actual_m) and not np.any(r.state.water_m) and np.all(np.signbit(r.state.water_m) == np.signbit(np.zeros((3, 6))))  # never adds water to reach the floor


def test_availability_equals_its_definition_and_the_actual_of_an_oversized_request(backend):
    rng = np.random.default_rng(4)
    L, N = 4, 30
    p = a2.params(rng, L, N)
    mask = rng.random(N) < 0.7
    col = spatial_owner(p, backend, mask)
    op = sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
    for kind in ("zero", "at_floor", "below_floor", "random", "saturated"):
        w = make_states(rng, p, kind)
        state = sp.spatial_column_state(col, w, 0.0, memory_budget_bytes=BIG)
        av = sw.spatial_layer_availability_m(op, state, memory_budget_bytes=BIG)
        res = p["res"] * np.diff(p["edges"], axis=0)
        expect = np.where(mask[None, :], np.maximum(w - res, 0.0), 0.0)
        np.testing.assert_array_equal(bits(av), bits(expect))
        assert av.shape == (L, N) and av.dtype == np.float64 and av.flags.c_contiguous and np.all(av >= 0.0)
        big = (av + 1.0) * mask[None, :]  # a request above every availability takes exactly the availability
        r = sw.spatial_withdraw(op, state, big, memory_budget_bytes=BIG)
        np.testing.assert_array_equal(bits(r.actual_m), bits(av))
        np.testing.assert_array_equal(bits(state.water_m), bits(w))  # the pure read/withdrawal never writes the input state
