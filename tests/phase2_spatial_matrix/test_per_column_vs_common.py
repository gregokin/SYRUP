"""The fixed-spatial matrix step against the accepted COMMON operator column by column: water, pool, intake, J and ledger are BIT-exact on the same backend (NumPy vs common NumPy, Numba vs common Numba),
across mixed scalar/shared/cell geometry layouts, scalar and per-cell K/psi/g, L = 1/2/3/5, N = 1/7/129, masks and all regimes; Numba agrees with NumPy within the declared bound; the persistent history
is carried over several updates exactly as the common sequence carries it. Author-written."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import ATOL, AREA, BIG, RTOL, a2, a3, bits, common_matrix, mi, sm, sp, spatial_pair, ss

LAYOUT_SETS = {
    "all_cell": dict(edges="cell", res="cell", fc="cell", sat="cell", iface="cell"),
    "all_shared": dict(edges="shared", res="shared", fc="shared", sat="shared", iface="shared"),
    "scalar_thetas": dict(edges="shared", res="scalar", fc="scalar", sat="scalar", iface="scalar"),
    "mixed": dict(edges="cell", res="shared", fc="scalar", sat="cell", iface="shared"),
    "mixed2": dict(edges="shared", res="cell", fc="cell", sat="scalar", iface="cell"),
}


def forcing(rng, p, mask, dt, N):
    """Per-cell regimes: zero rain, low rate (unponded), high rate (onset inside/already), contact ponded, exhausted pool, plus exact K = 0 / psi = 0 / g = 0 / g = 1 cells."""
    K = rng.uniform(1e-8, 2e-5, N)
    psi = rng.uniform(0.0, 0.3, N)
    g = rng.uniform(0.05, 1.0, N)
    K[0::11], psi[1::7], g[2::9], g[3::13] = 0.0, 0.0, 0.0, 1.0
    regime = rng.integers(0, 6, N)
    rate = np.where(regime == 0, 0.0, np.where(regime == 1, K * rng.uniform(0.1, 1.0, N), np.where(regime == 2, K * rng.uniform(1.5, 6.0, N) + 1e-7, rng.uniform(0.0, 5e-5, N))))
    contact = (regime == 3) | (regime == 4)
    standing = np.where(contact, rng.uniform(1e-4, 5e-3, N), 0.0)
    supply = rate * dt
    pool = standing + supply
    pool = np.where(regime == 4, pool * 0.2, pool)  # exhausted / small pools
    pool = np.where((~contact) & (regime == 5), supply * rng.uniform(0.3, 1.0, N), pool)  # a pool that is part of its supply
    rate, standing, pool = (np.where(mask, a, 0.0) for a in (rate, standing, pool))
    pool = np.where(mask, pool, rng.uniform(0.0, 1e-3, N))  # inactive cells may still carry a pool (it must not be touched)
    return K, psi, g, np.ascontiguousarray(pool), np.ascontiguousarray(standing), np.ascontiguousarray(rate)


def water_for(rng, p, N):
    L = p["res"].shape[0]
    dz = np.diff(p["edges"], axis=0)
    fc, sat = p["fc"] * dz, p["sat"] * dz
    w = fc * rng.uniform(0.1, 1.0, (L, N))  # below field capacity
    above = rng.random(N) < 0.35
    w[:, above] = (fc + (sat - fc) * rng.uniform(0.0, 1.0, (L, N)))[:, above]
    full = rng.random(N) < 0.15
    w[0, full] = sat[0, full] * (1.0 - rng.uniform(0.0, 1e-3, full.sum()))  # near saturation: room-limited
    zero = rng.random(N) < 0.1
    w[:, zero] = 0.0
    return np.ascontiguousarray(w)


def common_chain(p, c, backend, K, psi, g, mask, water, steps, j0, c0, t0):
    col, op = common_matrix(p, c, backend, K, psi, g, active=bool(mask[c]))
    state = ss.column_state(col, water[:, c:c + 1].copy(), t0)
    hist = mi.wetting_history(op, wetting_m=j0[c:c + 1].copy(), cumulative_intake_m=c0[c:c + 1].copy())
    out = []
    for pool, standing, rate, dt in steps:
        r = mi.matrix_step(op, pool[c:c + 1].copy(), standing[c:c + 1].copy(), rate[c:c + 1].copy(), state, hist, dt)
        state, hist = r.state, r.history
        out.append(r)
    return out


@pytest.mark.parametrize("name", list(LAYOUT_SETS))
@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("N", [1, 7, 129])
def test_each_column_is_bit_exact_over_a_history_sequence(backend, name, L, N):
    rng = np.random.default_rng(1000 * L + N + len(name))
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS[name])
    mask = rng.random(N) < 0.8
    mask[0] = True
    if N > 1:
        mask[1] = False
    water = water_for(rng, p, N)
    K0, psi0, g0, *_ = forcing(rng, p, mask, 60.0, N)
    j0 = np.where(rng.random(N) < 0.5, 0.0, rng.uniform(0.0, 0.05, N))
    c0 = rng.uniform(0.0, 0.1, N)
    dts = (60.0, 3600.0, 0.0, 600.0)
    steps = []
    for dt in dts:
        _, _, _, pool, standing, rate = forcing(rng, p, mask, dt, N)
        steps.append((pool, standing, rate, dt))
    col, op = spatial_pair(inputs, N, backend, mask, K0, psi0, g0)
    state = sp.spatial_column_state(col, water, 4321.5, memory_budget_bytes=BIG)
    hist = sm.spatial_matrix_history(op, wetting_m=j0, cumulative_intake_m=c0, memory_budget_bytes=BIG)
    chains = [common_chain(p, c, backend, K0, psi0, g0, mask, water, steps, j0, c0, 4321.5) for c in range(N)]
    for s_i, (pool, standing, rate, dt) in enumerate(steps):
        got = sm.spatial_matrix_step(op, pool, standing, rate, state, hist, dt, memory_budget_bytes=BIG)
        assert got.state.t_s == 4321.5 and got.state.column_sha256 == col.sha256 and got.dt_s == dt  # the clock is preserved
        assert got.history.operator_sha256 == op.sha256
        for c in range(N):
            ref = chains[c][s_i]
            for label, a, b in (("water", got.state.water_m[:, c:c + 1], ref.state.water_m), ("pool", got.surface_pool_m[c:c + 1], ref.surface_pool_m), ("intake", got.intake_m[c:c + 1], ref.intake_m),
                                ("J", got.history.wetting_m[c:c + 1], ref.history.wetting_m), ("ledger", got.history.cumulative_intake_m[c:c + 1], ref.history.cumulative_intake_m)):
                np.testing.assert_array_equal(bits(a), bits(b), err_msg=f"{label} cell {c} step {s_i} {name} L={L} N={N}")
        assert got.budget.n_onsets == sum(chains[c][s_i].budget.n_onsets for c in range(N))
        assert got.budget.n_ponded == sum(chains[c][s_i].budget.n_ponded for c in range(N))
        assert got.budget.max_newton_iterations == max(chains[c][s_i].budget.max_newton_iterations for c in range(N))
        assert abs(got.budget.soil_residual_m3) <= got.budget.bound_m3 and abs(got.budget.pool_residual_m3) <= got.budget.bound_m3
        assert got.budget.n_terms == (mi.CELL_OPERATION_TERMS_PER_LAYER * L + mi.CELL_OPERATION_TERMS) * N + mi.ROW_COUNT
        np.testing.assert_array_equal(got.intake_m[~mask], 0.0)
        state, hist = got.state, got.history


def test_numba_agrees_with_numpy_within_the_declared_bound():
    pytest.importorskip("numba")
    rng = np.random.default_rng(77)
    L, N = 3, 200
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS["mixed"])
    mask = rng.random(N) < 0.85
    water = water_for(rng, p, N)
    K, psi, g, pool, standing, rate = forcing(rng, p, mask, 3600.0, N)
    res = {}
    for backend in ("numpy", "numba"):
        col, op = spatial_pair(inputs, N, backend, mask, K, psi, g)
        h = sm.spatial_matrix_history(op, wetting_m=np.zeros(N), cumulative_intake_m=np.zeros(N), memory_budget_bytes=BIG)
        res[backend] = sm.spatial_matrix_step(op, pool, standing, rate, sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG), h, 3600.0, memory_budget_bytes=BIG)
    a, b = res["numpy"], res["numba"]
    for x, y in ((a.state.water_m, b.state.water_m), (a.surface_pool_m, b.surface_pool_m), (a.intake_m, b.intake_m), (a.history.wetting_m, b.history.wetting_m),
                 (a.history.cumulative_intake_m, b.history.cumulative_intake_m)):
        np.testing.assert_allclose(y, x, rtol=RTOL, atol=ATOL)
    assert a.budget.n_onsets == b.budget.n_onsets and a.budget.n_ponded == b.budget.n_ponded


def test_scalar_and_cell_law_inputs_agree(backend):
    """A scalar K/psi/g and the same value expanded per cell give identical arrays (the scalar retains one value, the per-cell one retains N)."""
    rng = np.random.default_rng(5)
    L, N = 2, 33
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS["mixed"])
    mask = np.ones(N, dtype=bool)
    water = water_for(rng, p, N)
    _, _, _, pool, standing, rate = forcing(rng, p, mask, 600.0, N)
    out = []
    for expand in (False, True):
        mk = (lambda v: np.full(N, v)) if expand else float
        col, op = spatial_pair(inputs, N, backend, mask, mk(4e-6), mk(0.12), mk(0.6))
        assert dict(op.layouts) == ({k: "cell" for k in op.layouts} if expand else {k: "scalar" for k in op.layouts})
        assert op.retained_bytes == (3 * N * 8 if expand else 3 * 8)
        h = sm.spatial_matrix_history(op, wetting_m=0.001, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
        out.append(sm.spatial_matrix_step(op, pool, standing, rate, sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG), h, 600.0, memory_budget_bytes=BIG))
    for x, y in ((out[0].state.water_m, out[1].state.water_m), (out[0].surface_pool_m, out[1].surface_pool_m), (out[0].history.wetting_m, out[1].history.wetting_m)):
        np.testing.assert_array_equal(bits(x), bits(y))


def test_all_inactive_mask_is_the_identity(backend):
    rng = np.random.default_rng(9)
    L, N = 2, 6
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS["all_cell"])
    mask = np.zeros(N, dtype=bool)
    water = water_for(rng, p, N)
    col, op = spatial_pair(inputs, N, backend, mask, 1e-5, 0.1, 0.5)
    h = sm.spatial_matrix_history(op, wetting_m=np.full(N, 0.01), cumulative_intake_m=np.full(N, 0.02), memory_budget_bytes=BIG)
    pool = np.full(N, 0.003)
    r = sm.spatial_matrix_step(op, pool, np.zeros(N), np.zeros(N), sp.spatial_column_state(col, water, 1.0, memory_budget_bytes=BIG), h, 600.0, memory_budget_bytes=BIG)
    np.testing.assert_array_equal(bits(r.state.water_m), bits(water))
    np.testing.assert_array_equal(bits(r.surface_pool_m), bits(pool))
    np.testing.assert_array_equal(r.intake_m, 0.0)
    assert r.budget.intake_m3 == 0.0 and r.budget.soil_before_m3 == 0.0 and r.budget.n_ponded == 0
