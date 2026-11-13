"""Invariants on random valid columns: closure, permutation equivariance, splitting a coordinate shift, positivity. Author-written; not an independent oracle."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import arrays, random_columns, run


@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_conserved_totals_per_column_and_global(cpu_backend, L):
    rng = np.random.default_rng(100 + L)
    eo, en, W, up, lo = arrays(*random_columns(rng, L, 64))
    res = run(eo, en, W, up, lo, cpu_backend, area=2.5)
    drift = (res.water_m.sum(axis=0) - W.sum(axis=0)) - ((res.added_above_m + res.added_below_m) - (res.removed_above_m + res.removed_below_m))
    scale = (eo[1:] - eo[:-1]).sum(axis=0) + (en[1:] - en[:-1]).sum(axis=0)
    from maple_syrup.conservation import volume_roundoff_bound_m3

    assert np.all(np.abs(drift) <= volume_roundoff_bound_m3(2 * L * L + 9 * L + 8, 1.0) * scale)
    assert res.closure["max_column_residual_over_bound"] <= 1.0
    assert res.closure["global_bound_terms"] == 64 * (2 * L * L + 9 * L + 8) + 8
    assert np.all(res.water_m >= 0.0) and np.all(res.removed_above_m >= 0) and np.all(res.added_below_m >= 0)
    assert np.all(res.water_m <= (en[1:] - en[:-1]) * (1.0 + 1e-12))  # never above a width beyond roundoff


def test_cell_permutation_equivariance_bitwise(cpu_backend):
    rng = np.random.default_rng(7)
    eo, en, W, up, lo = arrays(*random_columns(rng, 3, 31))
    perm = rng.permutation(31)
    a = run(eo, en, W, up, lo, cpu_backend)
    b = run(eo[:, perm], en[:, perm], W[:, perm], up[perm], lo[perm], cpu_backend)
    np.testing.assert_array_equal(b.water_m, a.water_m[:, perm])
    for name in ("removed_above_m", "removed_below_m", "added_above_m", "added_below_m"):
        np.testing.assert_array_equal(getattr(b, name), getattr(a, name)[perm])


def test_columns_are_independent_of_their_neighbours(cpu_backend):
    rng = np.random.default_rng(8)
    eo, en, W, up, lo = arrays(*random_columns(rng, 4, 20))
    full = run(eo, en, W, up, lo, cpu_backend)
    one = run(eo[:, 5:6], en[:, 5:6], W[:, 5:6], up[5:6], lo[5:6], cpu_backend)
    np.testing.assert_array_equal(one.water_m[:, 0], full.water_m[:, 5])
    assert one.removed_below_m[0] == full.removed_below_m[5]


def test_pure_translation_moves_water_with_the_material(cpu_backend):
    # shifting both edge sets by the same offset (old and new in one frame) leaves all amounts unchanged up to the rounding of the shifted coordinates (dyadic offset: exact)
    rng = np.random.default_rng(9)
    eo, en, W, up, lo = arrays(*random_columns(rng, 3, 12))
    shift = 4.0  # exactly representable; coordinates stay well within precision
    a = run(eo, en, W, up, lo, cpu_backend)
    b = run(eo + shift, en + shift, W, up, lo, cpu_backend)
    np.testing.assert_allclose(b.water_m, a.water_m, rtol=0, atol=1e-13)


def test_theta_one_and_zero_extremes(cpu_backend):
    full = run(np.array([[0.0], [1.0]]), np.array([[-1.0], [2.0]]), np.array([[1.0]]), [1.0], [1.0], cpu_backend)
    np.testing.assert_array_equal(full.water_m, [[3.0]])  # saturated everywhere
    dry = run(np.array([[0.0], [1.0]]), np.array([[-1.0], [2.0]]), np.array([[0.0]]), [0.0], [0.0], cpu_backend)
    np.testing.assert_array_equal(dry.water_m, [[0.0]])
