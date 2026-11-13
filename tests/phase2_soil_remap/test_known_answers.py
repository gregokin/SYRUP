"""Hand-computed known answers (dyadic numbers so every value is exact) and exact structural identities. Author-written; not an independent oracle."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import digest, run


def col(*vals):
    return np.array(vals, dtype=np.float64).reshape(-1, 1)


def check(res, water, ra, rb, aa, ab):
    np.testing.assert_array_equal(res.water_m, np.array(water).reshape(-1, 1))
    for got, want in ((res.removed_above_m, ra), (res.removed_below_m, rb), (res.added_above_m, aa), (res.added_below_m, ab)):
        np.testing.assert_array_equal(got, [want])
    new, old = res.water_m.sum(), None
    assert abs(res.closure["global_residual_m3"]) <= res.closure["global_bound_m3"]


def test_one_layer_shift_down_with_lower_entry(cpu_backend):
    # old [0,1] W=0.5; new [0.25,1.25]: overlap 0.75*0.5, new bottom slab 0.25 long at theta 0.25; removed above new top: 0.25*0.5
    res = run(col(0, 1), col(0.25, 1.25), col(0.5), [0.5], [0.25], cpu_backend)
    check(res, [0.4375], ra=0.125, rb=0.0, aa=0.0, ab=0.0625)


def test_two_layers_shift_up_with_upper_entry(cpu_backend):
    # old [0,1,3] W=[0.5,1.0]; new [-0.5,0.5,2.5], upper theta 0.25
    res = run(col(0, 1, 3), col(-0.5, 0.5, 2.5), col(0.5, 1.0), [0.25], [0.75], cpu_backend)
    check(res, [0.375, 1.0], ra=0.0, rb=0.25, aa=0.125, ab=0.0)


def test_three_layers_top_removed_ignores_unused_boundary_thetas(cpu_backend):
    res = run(col(0, 0.5, 1.5, 3.5), col(0.5, 1.0, 2.0, 3.5), col(0.25, 0.5, 1.0), [0.9], [0.8], cpu_backend)
    check(res, [0.25, 0.5, 0.75], ra=0.25, rb=0.0, aa=0.0, ab=0.0)


def test_new_profile_entirely_above_old_with_gap(cpu_backend):
    # old [0,1] W=0.5; new [-3,-2]: all old water is below the new bottom; the new interval is filled by upper theta only
    res = run(col(0, 1), col(-3, -2), col(0.5), [0.25], [0.75], cpu_backend)
    check(res, [0.25], ra=0.0, rb=0.5, aa=0.25, ab=0.0)


def test_new_profile_entirely_below_old_disjoint(cpu_backend):
    res = run(col(0, 1), col(2, 3), col(0.5), [0.25], [0.5], cpu_backend)
    check(res, [0.5], ra=0.5, rb=0.0, aa=0.0, ab=0.5)


def test_dry_and_positive_boundary_water(cpu_backend):
    dry = run(col(0, 1), col(-1, 0), col(0.5), [0.0], [0.0], cpu_backend)
    check(dry, [0.0], ra=0.0, rb=0.5, aa=0.0, ab=0.0)  # dry incoming material: no invented water, no residual floor
    wet = run(col(0, 1), col(-1, 0), col(0.5), [1.0], [0.0], cpu_backend)
    check(wet, [1.0], ra=0.0, rb=0.5, aa=1.0, ab=0.0)  # a full-saturation entering slab is allowed (theta == 1)


@pytest.mark.parametrize("L", [1, 2, 3, 6])
def test_zero_geometry_change_is_bit_exact_and_flux_free(cpu_backend, L):
    rng = np.random.default_rng(10 + L)
    eo = np.cumsum(rng.uniform(0.01, 1.0, size=(L + 1, 7)), axis=0) - 0.37
    W = (eo[1:] - eo[:-1]) * rng.uniform(0.0, 1.0, size=(L, 7))
    res = run(eo, eo.copy(), W, rng.uniform(0, 1, 7), rng.uniform(0, 1, 7), cpu_backend)
    np.testing.assert_array_equal(res.water_m.view(np.uint64), W.view(np.uint64))  # every bit
    for f in (res.removed_above_m, res.removed_below_m, res.added_above_m, res.added_below_m):
        assert not np.any(f) and not np.any(np.signbit(f))


def test_zero_net_exchange_is_not_classified_by_this_module(cpu_backend):
    # removing 0.25 at the top and adding 0.25 at the top leaves the surface unchanged; the caller states edges [-0.25..] vs the same extent via different geometry:
    # old [0,1] -> new [0,1] is identity, but a caller that erodes and redeposits states e.g. old [0.25,1.25] relative to the pre-removal frame and new [0,1]: material entering above.
    res = run(col(0.25, 1.25), col(0.0, 1.0), col(0.5), [0.0], [0.0], cpu_backend)
    # new [0,1]: [0,0.25) is new dry material (0), [0.25,1) overlaps old with 0.75*0.5; old [1,1.25) is removed below
    check(res, [0.375], ra=0.0, rb=0.125, aa=0.0, ab=0.0)


def test_empty_selection_is_a_valid_no_op(cpu_backend):
    z = np.zeros
    res = run(z((3, 0)), z((3, 0)), z((2, 0)), z(0), z(0), cpu_backend)
    assert res.water_m.shape == (2, 0) and res.removed_above_m.shape == (0,)
    assert res.closure["global_residual_m3"] == 0.0 and res.n_columns == 0
    assert res.label == "component_trial_only"


def test_inputs_are_preserved_and_outputs_fresh(cpu_backend):
    rng = np.random.default_rng(5)
    from conftest import random_columns, arrays

    eo, en, W, up, lo = arrays(*random_columns(rng, 3, 9))
    before = digest(eo, en, W, up, lo)
    res = run(eo, en, W, up, lo, cpu_backend)
    assert digest(eo, en, W, up, lo) == before
    for out in (res.water_m, res.removed_above_m, res.removed_below_m, res.added_above_m, res.added_below_m):
        assert all(not np.shares_memory(out, a) for a in (eo, en, W, up, lo))
