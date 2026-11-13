"""The compiled spatial step against REAL accepted references: bit-for-bit per column against the accepted common Numba column (same kernel body, same libm), within the declared bound against the explicit NumPy
spatial path (and the NumPy common solver through it), bitwise with a zero diffusivity (no libm term), identical typed failures and flag bits, mixed scalar/shared/cell layouts, L = 1/2/3/5, both diffusion modes
and bottoms, ordinary intake vs the uniform pulse, the post-cap share, dt boundaries. Author-written; the oracles are the accepted solvers, not a re-implementation."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import (BIG, DTS, LAYOUTS, STEP_FIELDS, a2, bits, common_numba_column, layout_case, require_numba, result_arrays, run_spatial, sp, spec_from, ss, step_args)

require_numba()
RTOL, ATOL = 2e-12, 1e-14  # the declared Numba-vs-NumPy bound (expm1 may differ by an ulp)


def arrays_of(column, water, dt, added, pet, share, recharge):
    state = sp.spatial_column_state(column, water, 0.0, memory_budget_bytes=BIG)
    return sp._evolve(column, state, added, pet, dt, share, recharge)  # the StepArrays incl. the flag/storage rows


def common_arrays(p, c, water, added, pet, dt, share, recharge, mask, *, mode, bottom, gravity, stress=0.5, factor=1.0):
    col = common_numba_column(p, c, mode=mode, bottom=bottom, gravity=gravity, stress=stress, factor=factor, active=bool(mask[c]))
    return col.runner(col, water[:, c : c + 1].copy(), None if recharge is not None else added[:, c : c + 1].copy(), pet[c : c + 1].copy(), dt,
                      None if share is None else share[c : c + 1].copy(), recharge)


def assert_bitwise_per_column(got, p, water, added, pet, dt, share, recharge, mask, **kw):
    for c in range(water.shape[1]):
        ref = common_arrays(p, c, water, added, pet, dt, share, recharge, mask, **kw)
        for name in STEP_FIELDS:
            g, w = getattr(got, name), getattr(ref, name)
            g = g[..., c : c + 1] if g.ndim == 2 else g[c : c + 1]
            np.testing.assert_array_equal(bits(g), bits(w), err_msg=f"{name} column {c}")


@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_heterogeneous_spatial_numba_step_is_bit_identical_per_column_to_the_accepted_common_numba_column(L, mode, bottom):
    rng = np.random.default_rng(100 + L)
    N = 12
    p = a2.params(rng, L, N, mode=mode)
    mask = rng.random(N) < 0.75
    mask[0], mask[1] = True, False
    for gravity in (True, False):
        col = a2.prepare(a2.make_spec(p, mode=mode, bottom=bottom, gravity=gravity), backend="numba", mask=mask)
        for kind in ("dry", "saturated", "above_fc", "random", "mixed"):
            water, added, pet = step_args(rng, p, mask, kind)
            share = rng.random(N)
            for dt in DTS:
                for sh in (None, share):
                    got = arrays_of(col, water, dt, added, pet, sh, None)
                    assert_bitwise_per_column(got, p, water, added, pet, dt, sh, None, mask, mode=mode, bottom=bottom, gravity=gravity)
                    if dt > 0.0:
                        got = arrays_of(col, water, dt, None, pet, sh, 2.5e-3)
                        assert_bitwise_per_column(got, p, water, added, pet, dt, sh, 2.5e-3, mask, mode=mode, bottom=bottom, gravity=gravity)


@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_spatial_numba_agrees_with_the_explicit_numpy_spatial_path_within_the_declared_bound(L, mode, bottom):
    rng = np.random.default_rng(200 + L)
    N = 40
    p = a2.params(rng, L, N, mode=mode)
    mask = rng.random(N) < 0.8
    spec = a2.make_spec(p, mode=mode, bottom=bottom)
    cn, cb = a2.prepare(spec, backend="numpy", mask=mask), a2.prepare(spec, backend="numba", mask=mask)
    for kind in ("dry", "saturated", "above_fc", "random", "mixed"):
        water, added, pet = step_args(rng, p, mask, kind)
        for dt in DTS:
            for share in (None, rng.random(N)):
                for recharge in (None, 2e-3):
                    if recharge is not None and not dt > 0.0:
                        continue
                    a = arrays_of(cn, water, dt, added if recharge is None else None, pet, share, recharge)
                    b = arrays_of(cb, water, dt, added if recharge is None else None, pet, share, recharge)
                    for name in STEP_FIELDS:
                        x, y = getattr(a, name), getattr(b, name)
                        if name == "rows":  # storage before/added are exact sums; after carries the diffusion ulp; the flag row is exact
                            np.testing.assert_array_equal(x[3], y[3])
                            np.testing.assert_allclose(x[:3], y[:3], rtol=RTOL, atol=ATOL)
                        else:
                            np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL, err_msg=name)


def test_zero_diffusivity_removes_the_only_libm_term_so_numba_equals_numpy_bit_for_bit(mode, bottom):
    rng = np.random.default_rng(7)
    L, N = 4, 30
    p = a2.params(rng, L, N, mode=mode)
    p["iface"] = np.zeros_like(p["iface"])
    mask = rng.random(N) < 0.8
    spec = a2.make_spec(p, mode=mode, bottom=bottom)
    cn, cb = a2.prepare(spec, backend="numpy", mask=mask), a2.prepare(spec, backend="numba", mask=mask)
    for kind in ("above_fc", "random", "mixed"):
        water, added, pet = step_args(rng, p, mask, kind)
        for dt in (3600.0, 86400.0):
            a, b = arrays_of(cn, water, dt, added, pet, rng.random(N), None), None
            share = rng.random(N)
            a = arrays_of(cn, water, dt, added, pet, share, None)
            b = arrays_of(cb, water, dt, added, pet, share, None)
            for name in STEP_FIELDS:
                np.testing.assert_array_equal(bits(getattr(a, name)), bits(getattr(b, name)), err_msg=name)


def _layout_combos():
    for edges, res, fc, sat, iface in itertools.product(("shared", "cell"), LAYOUTS, LAYOUTS, LAYOUTS, LAYOUTS):
        yield edges, res, fc, sat, iface


@pytest.mark.parametrize("mode_", ["constant", "tietjen_donor_depth"])
def test_every_layout_combination_is_bit_identical_per_column_to_the_accepted_common_numba_column(mode_):
    """edges (shared|cell) x each of the three theta maps (scalar|shared|cell) x the interface input (scalar|shared|cell): 2 x 27 x 3 = 162 layouts per mode, L = 3, closed bottom."""
    rng = np.random.default_rng(300)
    N = 4
    mask = np.array([True, True, False, True])
    for edges, res, fc, sat, iface in _layout_combos():
        inputs, p = layout_case(rng, 3, N, edges=edges, res=res, fc=fc, sat=sat, iface=iface, mode=mode_)
        col = a2.prepare(spec_from(inputs, N, mode=mode_, bottom="closed"), backend="numba", mask=mask)
        assert {k: col.layouts[k] for k in ("depth_edges_m", "theta_residual", "theta_field_capacity", "theta_saturation")} == {"depth_edges_m": edges, "theta_residual": res, "theta_field_capacity": fc, "theta_saturation": sat}
        water, added, pet = step_args(rng, p, mask, "above_fc")
        share = rng.random(N)
        for dt, recharge in ((3600.0, None), (86400.0, 2e-3)):
            got = arrays_of(col, water, dt, added if recharge is None else None, pet, share, recharge)
            assert_bitwise_per_column(got, p, water, added, pet, dt, share, recharge, mask, mode=mode_, bottom="closed", gravity=True)


@pytest.mark.parametrize("L", [1, 2, 5])
def test_random_layout_combinations_for_other_stack_heights(L, mode, bottom):
    rng = np.random.default_rng(400 + L)
    N = 6
    mask = rng.random(N) < 0.8
    mask[0] = True
    for seed in range(12):
        kinds = {k: LAYOUTS[int(rng.integers(0, 3))] for k in ("res", "fc", "sat", "iface")}
        inputs, p = layout_case(rng, L, N, edges=("shared", "cell")[int(rng.integers(0, 2))], mode=mode, **kinds)
        col = a2.prepare(spec_from(inputs, N, mode=mode, bottom=bottom), backend="numba", mask=mask)
        water, added, pet = step_args(rng, p, mask, ("random", "above_fc", "mixed")[seed % 3])
        for dt in (3600.0, 86400.0):
            got = arrays_of(col, water, dt, added, pet, None, None)
            assert_bitwise_per_column(got, p, water, added, pet, dt, None, None, mask, mode=mode, bottom=bottom, gravity=True)


def test_shared_scalars_reach_the_kernel(mode):
    """stress / evaporation factor / gravity switch are owner scalars: non-default values change the result exactly as in the common column."""
    rng = np.random.default_rng(9)
    N = 8
    p = a2.params(rng, 3, N, mode=mode)
    mask = np.ones(N, bool)
    for stress, factor, gravity in ((0.3, 0.5, True), (0.9, 1.3, False)):
        col = a2.prepare(a2.make_spec(p, mode=mode, stress=stress, factor=factor, gravity=gravity), backend="numba", mask=mask)
        water, added, pet = step_args(rng, p, mask, "above_fc")
        pet = pet * 1e3  # make the evaporation demand matter
        got = arrays_of(col, water, 86400.0, added, pet, None, None)
        assert_bitwise_per_column(got, p, water, added, pet, 86400.0, None, None, mask, mode=mode, bottom="open", gravity=gravity, stress=stress, factor=factor)


# ---- identical typed failures and flag bits ---------------------------------------------------------------------------------------------------
def _expect_same_failure(L, mutate, *, share=False, recharge=None, dt=3600.0, mode="constant", bottom="open", may_succeed=False):
    rng = np.random.default_rng(11)
    N = 9
    p = a2.params(rng, L, N, mode=mode)
    mask = np.ones(N, bool)
    mask[3] = False
    water, added, pet = step_args(rng, p, mask, "random")
    sh = rng.random(N) if share else None
    water, added, pet, sh = mutate(water.copy(), added.copy(), pet.copy(), sh, p)
    outcomes = []
    for backend in ("numpy", "numba"):
        col = a2.prepare(a2.make_spec(p, mode=mode, bottom=bottom), backend=backend, mask=mask)
        state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
        try:
            if recharge is None:
                sp.spatial_step(col, state, dt, added_water_m=added, potential_evaporation_m_s=pet, evaporation_share=sh, memory_budget_bytes=BIG)
            else:
                sp.spatial_step_with_recharge(col, state, dt, recharge_depth_m=recharge, potential_evaporation_m_s=pet, evaporation_share=sh, memory_budget_bytes=BIG)
            outcomes.append(None)
        except ss.SubsurfaceError as exc:
            outcomes.append((type(exc).__name__, getattr(exc, "bit", None), getattr(exc, "n_flagged", None)))
        assert state.t_s == 0.0 and not state.water_m.flags.writeable  # a refusal leaves the state untouched
    assert outcomes[0] == outcomes[1] and (may_succeed or outcomes[0] is not None), outcomes
    return outcomes[0]


def _set(arr, idx, value):
    if arr.ndim == 2:  # the layer index wraps for a one-layer stack
        idx = (idx[0] % arr.shape[0], idx[1])
    arr[idx] = value
    return arr


@pytest.mark.parametrize("name,bit,mutate,kw", [
    ("nonfinite_water", 0, lambda w, a, e, s, p: (_set(w, (1, 2), np.nan), a, e, s), {}),
    ("negative_water", 1, lambda w, a, e, s, p: (_set(w, (0, 4), -1e-3), a, e, s), {}),
    ("over_saturated_water", 2, lambda w, a, e, s, p: (_set(w, (1, 5), 10.0), a, e, s), {}),
    ("nonfinite_added", 3, lambda w, a, e, s, p: (w, _set(a, (0, 1), np.inf), e, s), {}),
    ("negative_added", 4, lambda w, a, e, s, p: (w, _set(a, (1, 2), -1e-6), e, s), {}),
    ("added_on_inactive", 5, lambda w, a, e, s, p: (w, _set(a, (0, 3), 1e-6), e, s), {}),
    ("nonfinite_pet", 6, lambda w, a, e, s, p: (w, a, _set(e, 2, np.nan), s), {}),
    ("negative_pet", 7, lambda w, a, e, s, p: (w, a, _set(e, 2, -1e-9), s), {}),
    ("pet_on_inactive", 8, lambda w, a, e, s, p: (w, a, _set(e, 3, 1e-9), s), {}),
    ("added_at_dt0", 9, lambda w, a, e, s, p: (w, _set(a, (0, 0), 1e-6), e, s), {"dt": 0.0}),
    ("share_nonfinite", 15, lambda w, a, e, s, p: (w, a, e, _set(s, 4, np.nan)), {"share": True}),
    ("share_out_of_range", 16, lambda w, a, e, s, p: (w, a, e, _set(s, 3, 1.5)), {"share": True}),
])
@pytest.mark.parametrize("L", [1, 3])
def test_invalid_inputs_raise_the_same_typed_step_error_with_the_same_lowest_bit(name, bit, mutate, kw, L):
    outcome = _expect_same_failure(L, mutate, **kw)
    assert outcome[0] == "SubsurfaceStepError" and outcome[1] == bit


def test_the_pulse_fit_failure_is_the_same_typed_trial_outcome_on_both_backends(mode, bottom):
    """A 50 m pulse cannot fit: with a CLOSED bottom the column has nowhere to put it (typed `SubsurfaceFitFailure`, flag 17, nothing clipped); with an OPEN bottom the gravity cascade drains the excess and the
    step succeeds -- identically on both backends."""
    outcome = _expect_same_failure(3, lambda w, a, e, s, p: (_set(w, (0, 2), 0.0), a, e, s), recharge=50.0, mode=mode, bottom=bottom, may_succeed=bottom == "open")
    if bottom == "closed":
        assert outcome[0] == "SubsurfaceFitFailure" and outcome[1] == ss.RECHARGE_FIT_BIT
    else:
        assert outcome is None


# ---- the capacity-tolerance boundary (cap_tol = coef * (theta_sat * dz), derived inline per cell) ------------------------------------------------
@pytest.mark.parametrize("sat_layout,edges_layout", [("cell", "cell"), ("shared", "cell"), ("scalar", "shared"), ("cell", "shared")])
def test_the_inline_capacity_tolerance_flags_exactly_the_cells_the_accepted_solvers_flag(sat_layout, edges_layout, mode):
    """A layer at saturation + f * cap_tol with f < 1 is accepted, f > 1 is flagged (bit 2 = 4.0): the boundary depends on the per-cell `coef * (theta_sat dz)`, so a tolerance derived from the wrong map (e.g.
    from dz) would move it. Compared bit for bit with the accepted common Numba column AND the explicit NumPy spatial path, and against the stated expectation."""
    from maple_syrup.conservation import volume_roundoff_bound_m3

    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    rng = np.random.default_rng(31)
    L, N = 3, 6
    inputs, p = layout_case(rng, L, N, edges=edges_layout, res="cell", fc="cell", sat=sat_layout, iface="cell", mode=mode)
    p["sat"] = np.minimum(p["sat"], 0.6)  # theta_sat < 1 so that coef * dz > coef * sat_m (a tolerance taken from dz would be wrong)
    mask = np.ones(N, bool)
    fs = np.array([0.0, 0.5, 0.99, 1.5, 3.0, 1.2])
    sat_m = a2.sat_depth(p)
    water = np.zeros((L, N))
    water[1] = sat_m[1] + fs * (coef * sat_m[1])  # the middle layer carries the excess (fs[0] = 0: exactly saturated)
    cb = a2.prepare(spec_from(inputs, N, mode=mode, bottom="closed"), backend="numba", mask=mask)
    cn = a2.prepare(spec_from(inputs, N, mode=mode, bottom="closed"), backend="numpy", mask=mask)
    added = np.zeros((L, N))
    pet = np.zeros(N)
    got = arrays_of(cb, water, 3600.0, added, pet, None, None)
    assert_bitwise_per_column(got, p, water, added, pet, 3600.0, None, None, mask, mode=mode, bottom="closed", gravity=True)
    ref = arrays_of(cn, water, 3600.0, added, pet, None, None)
    np.testing.assert_array_equal(got.rows[3], ref.rows[3])
    expected = np.where(fs > 1.0, 4.0, ss.NO_FLAG)
    np.testing.assert_array_equal(got.rows[3], expected)
