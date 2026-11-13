"""The fixed-spatial preferential step against the accepted COMMON operator column by column: water, surface, per-receiver intake and total are BIT-exact on the same backend (NumPy vs common NumPy,
Numba vs common Numba), across mixed scalar/shared/cell geometry layouts, scalar and per-cell a3 / max rate (incl. +inf), background/species weight sets (normalised, unnormalised, all-zero, overlapping),
L = 1/2/3/5, N = 1/7/129, masks and room-limited receivers, over repeated stages with the evolving state; Numba agrees with NumPy within the declared bound. Author-written."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import ATOL, BIG, RTOL, a2, a3, bits, common_pref, pi, sp, spatial_pair, spf, ss

LAYOUT_SETS = {
    "all_cell": dict(edges="cell", res="cell", fc="cell", sat="cell", iface="cell"),
    "all_shared": dict(edges="shared", res="shared", fc="shared", sat="shared", iface="shared"),
    "scalar_thetas": dict(edges="shared", res="scalar", fc="scalar", sat="scalar", iface="scalar"),
    "mixed": dict(edges="cell", res="shared", fc="scalar", sat="cell", iface="shared"),
    "mixed2": dict(edges="shared", res="cell", fc="cell", sat="scalar", iface="cell"),
}


def weight_sets(rng, L):
    """background, species weights: positive normalised, unnormalised [0,1,1]-like, all-zero (disabled), overlapping."""
    def vec(kind):
        if kind == "zero":
            return [0.0] * L
        if kind == "top":
            return [1.0] + [0.0] * (L - 1)
        if kind == "lower":
            return [0.0] + [1.0] * (L - 1)
        return list(rng.uniform(0.0, 3.0, L))
    return vec


def make_case(rng, L, N, kind):
    vec = weight_sets(rng, L)
    cases = {
        "no_species": dict(background=vec("top"), species=[]),
        "no_background": dict(background=None, species=[vec("lower")]),
        "zero_background_one_species": dict(background=vec("zero"), species=[vec("random")]),
        "multi_overlap": dict(background=vec("random"), species=[vec("random"), vec("lower"), vec("random")]),
        "disabled_species_with_cover": dict(background=vec("lower"), species=[vec("zero"), vec("random")]),
    }
    return cases[kind]


@pytest.mark.parametrize("kind", ["no_species", "no_background", "zero_background_one_species", "multi_overlap", "disabled_species_with_cover"])
@pytest.mark.parametrize("name", ["all_cell", "mixed", "all_shared"])
@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("N", [1, 7, 129])
def test_each_column_is_bit_exact_over_repeated_stages(backend, kind, name, L, N):
    rng = np.random.default_rng(100 * L + N + len(kind) + len(name))
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS[name])
    mask = rng.random(N) < 0.8
    mask[0] = True
    if N > 1:
        mask[1] = False
    case = make_case(rng, L, N, kind)
    S = len(case["species"])
    contributions = [rng.uniform(0.0, 2.0, N) for _ in range(S)] if S else None
    a3_in = rng.uniform(0.0, 1.0, N)
    a3_in[0::5], a3_in[1::7] = 0.0, 1.0  # exact beta endpoints on some cells
    mr_in = np.where(rng.random(N) < 0.3, np.inf, rng.uniform(1e-8, 5e-6, N))
    scalar_mode = (N % 2 == 1) and name == "all_cell"
    kwargs = dict(a3=0.6 if scalar_mode else a3_in, max_fraction_per_hour=0.85, max_intake_rate_m_s=2e-6 if scalar_mode else mr_in, background_receiver_weights=case["background"],
                  species_receiver_weights=case["species"])
    dz = np.diff(p["edges"], axis=0)
    sat = p["sat"] * dz
    water = np.ascontiguousarray(sat * rng.uniform(0.0, 0.95, (L, N)))
    near_full = rng.random(N) < 0.25
    water[:, near_full] = (sat * (1.0 - rng.uniform(0.0, 5e-3, (L, N))))[:, near_full]  # room-limited receivers
    col, op = spatial_pair(inputs, N, backend, mask, kwargs, contributions)
    assert op.n_species == S and dict(op.layouts) == {"a3": "scalar" if scalar_mode else "cell", "max_intake_rate_m_s": "scalar" if scalar_mode else "cell"}
    state = sp.spatial_column_state(col, water, 777.25, memory_budget_bytes=BIG)
    surface = np.where(mask, rng.uniform(0.0, 0.02, N), rng.uniform(0.0, 0.01, N))
    chains = []
    commons = [common_pref(p, c, backend, kwargs, contributions, active=bool(mask[c])) for c in range(N)]
    cstates = [ss.column_state(commons[c][0], water[:, c:c + 1].copy(), 777.25) for c in range(N)]
    csurf = [surface[c:c + 1].copy() for c in range(N)]
    for dt in (3600.0, 0.0, 600.0, 60.0, 7200.0):
        got = spf.spatial_preferential_step(op, surface, state, dt, memory_budget_bytes=BIG)
        assert got.state.t_s == 777.25 and got.state.column_sha256 == col.sha256 and got.dt_s == dt  # the clock is preserved
        for c in range(N):
            r = pi.preferential_step(commons[c][1], csurf[c], cstates[c], dt)
            for label, a, b in (("water", got.state.water_m[:, c:c + 1], r.state.water_m), ("surface", got.surface_water_m[c:c + 1], r.surface_water_m), ("intake", got.intake_m[:, c:c + 1], r.intake_m),
                                ("total", got.intake_total_m[c:c + 1], r.intake_total_m)):
                np.testing.assert_array_equal(bits(a), bits(b), err_msg=f"{label} cell {c} dt {dt} {kind} {name} L={L} N={N}")
            cstates[c], csurf[c] = r.state, r.surface_water_m
        assert abs(got.budget.soil_residual_m3) <= got.budget.bound_m3 and abs(got.budget.surface_residual_m3) <= got.budget.bound_m3
        assert got.budget.n_terms == (pi.CELL_OPERATION_TERMS_PER_LAYER * L + pi.CELL_OPERATION_TERMS) * N + pi.ROW_COUNT
        np.testing.assert_array_equal(got.intake_m[:, ~mask], 0.0)
        np.testing.assert_array_equal(bits(got.state.water_m[:, ~mask]), bits(state.water_m[:, ~mask]))  # inactive water preserved
        np.testing.assert_array_equal(bits(got.surface_water_m[~mask]), bits(surface[~mask]))  # inactive surface preserved
        state, surface = got.state, got.surface_water_m


def test_numba_agrees_with_numpy_within_the_declared_bound():
    pytest.importorskip("numba")
    rng = np.random.default_rng(31)
    L, N = 3, 200
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS["mixed"])
    mask = rng.random(N) < 0.85
    case = make_case(rng, L, N, "multi_overlap")
    contributions = [rng.uniform(0.0, 2.0, N) for _ in case["species"]]
    kwargs = dict(a3=rng.uniform(0.0, 0.9, N), max_fraction_per_hour=0.7, max_intake_rate_m_s=rng.uniform(1e-7, 4e-6, N), background_receiver_weights=case["background"], species_receiver_weights=case["species"])
    dz = np.diff(p["edges"], axis=0)
    water = np.ascontiguousarray(p["sat"] * dz * rng.uniform(0.0, 0.9, (L, N)))
    surface = rng.uniform(0.0, 0.02, N)
    res = {}
    for backend in ("numpy", "numba"):
        col, op = spatial_pair(inputs, N, backend, mask, kwargs, contributions)
        res[backend] = spf.spatial_preferential_step(op, surface, sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG), 1800.0, memory_budget_bytes=BIG)
    a, b = res["numpy"], res["numba"]
    for x, y in ((a.state.water_m, b.state.water_m), (a.surface_water_m, b.surface_water_m), (a.intake_m, b.intake_m), (a.intake_total_m, b.intake_total_m)):
        np.testing.assert_allclose(y, x, rtol=RTOL, atol=ATOL)


def test_scalar_and_cell_inputs_agree(backend):
    rng = np.random.default_rng(5)
    L, N = 2, 33
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS["mixed"])
    mask = np.ones(N, dtype=bool)
    dz = np.diff(p["edges"], axis=0)
    water = np.ascontiguousarray(p["sat"] * dz * rng.uniform(0.0, 0.9, (L, N)))
    surface = rng.uniform(0.0, 0.02, N)
    out = []
    for expand in (False, True):
        mk = (lambda v: np.full(N, v)) if expand else float
        kw = dict(a3=mk(0.4), max_fraction_per_hour=0.9, max_intake_rate_m_s=mk(1e-6), background_receiver_weights=[1.0, 2.0], species_receiver_weights=[])
        col, op = spatial_pair(inputs, N, backend, mask, kw, None)
        assert op.retained_bytes == (N + L * N) * 8 + (N * 8 if expand else 8)  # rate N + receiver L N + max rate (one value or N)
        out.append(spf.spatial_preferential_step(op, surface, sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG), 900.0, memory_budget_bytes=BIG))
    for x, y in ((out[0].state.water_m, out[1].state.water_m), (out[0].surface_water_m, out[1].surface_water_m), (out[0].intake_m, out[1].intake_m)):
        np.testing.assert_array_equal(bits(x), bits(y))


def test_all_inactive_mask_is_the_identity(backend):
    rng = np.random.default_rng(9)
    L, N = 2, 6
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS["all_cell"])
    mask = np.zeros(N, dtype=bool)
    kw = dict(a3=0.7, max_fraction_per_hour=1.0, max_intake_rate_m_s=np.inf, background_receiver_weights=[1.0, 1.0], species_receiver_weights=[])
    col, op = spatial_pair(inputs, N, backend, mask, kw, None)
    water = np.ascontiguousarray(p["sat"] * np.diff(p["edges"], axis=0) * 0.3)
    surface = np.full(N, 0.01)
    r = spf.spatial_preferential_step(op, surface, sp.spatial_column_state(col, water, 1.0, memory_budget_bytes=BIG), 3600.0, memory_budget_bytes=BIG)
    np.testing.assert_array_equal(bits(r.state.water_m), bits(water))
    np.testing.assert_array_equal(bits(r.surface_water_m), bits(surface))
    assert r.budget.intake_m3 == 0.0 and r.budget.soil_before_m3 == 0.0 and not r.intake_m.any()
