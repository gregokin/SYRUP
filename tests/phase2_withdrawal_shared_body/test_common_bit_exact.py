"""The reader-refactored common withdrawal bodies (NumPy, Numba) equal the pre-B1.a bodies BIT FOR BIT on every output array and every reduction row, including signed zeros, long stacks with one cell (reduction
order), masks, flagged invalid inputs; and the public operator API gives identical results/budgets. The only tolerated difference is the sign/payload bit of a NaN produced by an invalid input. Author-written."""
from __future__ import annotations

import numpy as np
import pytest

from maple_syrup import subsurface as ss

NAMES = ("out_w", "actual", "unmet", "scale")


def run(mod, fn_name, w, q, residual, satb, act):
    L, N = w.shape
    out = [np.empty((L, N)) for _ in range(4)] + [np.empty((8, N))]
    getattr(mod, fn_name)(w, q, residual, satb, act, *out)
    return out


def same(a, b, label):
    for x, y in zip(a, b):
        both_nan = np.isnan(x) & np.isnan(y)
        np.testing.assert_array_equal(np.where(both_nan, 0.0, x).view(np.uint64), np.where(both_nan, 0.0, y).view(np.uint64), err_msg=label)


def scenarios(rng):
    for L in (1, 2, 3, 5, 8, 9, 17):
        for N in (1, 2, 7, 64):
            res = np.sort(rng.uniform(0.0, 0.01, L))
            satb = res + rng.uniform(0.05, 0.2, L)
            act = (rng.random(N) < 0.75).astype(np.uint8)
            w = satb[:, None] * rng.random((L, N))
            q = rng.uniform(0.0, 0.1, (L, N)) * act[None, :]
            yield f"L{L}N{N}", w, q, res, satb, act
            w2, q2 = w.copy(), q.copy()  # signed zeros, water at/below residual, zero requests, zero floor
            w2[0, :] = -0.0
            w2[-1, :] = res[-1]
            q2[0, :] = -0.0
            q2[-1, :] = 0.0
            yield f"L{L}N{N}-zeros", w2, q2, np.zeros(L), satb, act
            w3, q3 = w.copy(), q.copy()  # invalid inputs: every flag family, in different cells
            if N >= 4:
                w3[0, 0] = np.nan
                w3[-1, 1] = -1e-3
                w3[0, 2] = satb[0] * 5
                q3[0, 3] = np.inf
                q3[-1, 3 % N] = -1.0
                q3[0, 0] = 0.01
            yield f"L{L}N{N}-invalid", w3, q3, res, satb, act
            act4 = act.copy()
            act4[:] = 0  # all inactive, zero requests
            yield f"L{L}N{N}-inactive", w, np.zeros((L, N)), res, satb, act4


@pytest.mark.parametrize("fn", ["_fused_numpy", "_fused_numba"])
def test_fused_bodies_are_bit_identical_to_the_pre_b1a_bodies(old, new, fn):
    if fn == "_fused_numba" and (new._fused_numba is None or old._fused_numba is None):
        pytest.skip("numba is not importable")
    rng = np.random.default_rng(1)
    n = 0
    for label, w, q, res, satb, act in scenarios(rng):
        with np.errstate(all="ignore"):
            a = run(old, fn, w, q, res, satb, act)
            b = run(new, fn, w, q, res, satb, act)
        same(a, b, f"{fn} {label}")
        n += 1
    assert n >= 100


def make_column(mod_ss, L, N, backend, mask):
    edges = np.concatenate([[0.0], np.cumsum(np.linspace(0.08, 0.3, L))])
    lay = mod_ss.soil_layers(edges, theta_residual=np.linspace(0.01, 0.04, L), theta_field_capacity=np.linspace(0.18, 0.25, L), theta_saturation=np.linspace(0.4, 0.5, L),
                             theta_evaporation_stress=0.5, diffusivity_m2_s=np.linspace(2e-7, 9e-7, max(L - 1, 0)) if L > 1 else 1e-7)
    return mod_ss.prepare_column(lay, mask, 4.0, backend=backend)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("L", [1, 3, 9])
def test_public_operator_results_and_budgets_are_identical(old, new, backend, L):
    rng = np.random.default_rng(L)
    N = 50
    mask = rng.random(N) < 0.8
    col = make_column(ss, L, N, backend, mask)
    w = col.layers.saturation_m[:, None] * rng.random((L, N))
    for requested_scale in (0.0, 0.01, 1.0):
        q = rng.random((L, N)) * requested_scale * mask
        state = ss.column_state(col, w, 12.5)
        ro = old.withdraw_transpiration(old.prepare_transpiration(col), state, q)
        rn = new.withdraw_transpiration(new.prepare_transpiration(col), state, q)
        same([ro.state.water_m, ro.actual_m, ro.unmet_m, ro.scale], [rn.state.water_m, rn.actual_m, rn.unmet_m, rn.scale], "operator")
        assert ro.state.t_s == rn.state.t_s == 12.5 and ro.budget.as_dict() == rn.budget.as_dict()
        assert ro.memory.as_dict() == rn.memory.as_dict()  # the byte model is unchanged
    bad = q.copy()
    bad[0, 0] = np.nan
    for mod in (old, new):  # a refused withdrawal raises the same typed error and publishes nothing
        with pytest.raises(ss.SubsurfaceError) as e:
            mod.withdraw_transpiration(mod.prepare_transpiration(col), state, bad)
        assert "requested_water_m not finite or negative" in str(e.value)
    np.testing.assert_array_equal(state.water_m, w)
