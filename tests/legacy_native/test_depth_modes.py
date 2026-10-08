"""The optional native post-infiltration law depth and the exact accepted `previous` / `current` behaviour (written without being run)."""
from __future__ import annotations

import numpy as np
import pytest

from maple_syrup import legacy_driver as D


def hpre_reference(old, rain, intake, active):
    """Independent restatement of `storm.coupled_step`'s hpre (infilt.for 105-148) with explicit branch precedence."""
    out = np.empty_like(old)
    for idx in np.ndindex(old.shape):
        h, r, j = old[idx], rain[idx], intake[idx]
        if active[idx] and j >= h + r:
            out[idx] = 0.0  # complete infiltration (tested first): d(1) is exactly zero
        elif j <= r:
            out[idx] = max(h - 0.0, 0.0)  # no run-on: intake does not exceed the rain, the old depth is untouched
        else:
            out[idx] = max(h - (j - r), 0.0)  # partial: only the net intake leaves the old depth
    return out


def arr(*values):
    return np.array([values], dtype=np.float64)


def test_branches_complete_no_runon_partial_and_no_rain_recession():
    old = arr(0.010, 0.010, 0.010, 0.010)
    rain = arr(0.001, 0.001, 0.001, 0.0)
    intake = arr(0.020, 0.0005, 0.004, 0.003)  # complete (>= 0.011), no run-on (<= rain), partial, recession (rain 0)
    active = np.ones(old.shape, dtype=bool)
    out = D.post_infiltration_depth(old, rain, intake, active)
    assert out[0, 0] == 0.0
    assert out[0, 1] == 0.010
    assert out[0, 2] == pytest.approx(0.010 - (0.004 - 0.001), abs=1e-15)
    assert out[0, 3] == pytest.approx(0.010 - 0.003, abs=1e-15)  # a falling hydrograph with no rain: depth minus intake
    np.testing.assert_array_equal(out, hpre_reference(old, rain, intake, active))


def test_complete_branch_forces_exact_zero_when_roundoff_would_leave_a_positive_depth():
    rng = np.random.default_rng(11)
    found = None
    for _ in range(200000):
        h, r = rng.uniform(0.0, 1.0, 2)
        j = h + r  # intake exactly at the complete threshold
        if j >= h + r and h - max(j - r, 0.0) > 0.0:  # the naive arithmetic leaves a tiny positive leftover
            found = (h, r, j)
            break
    assert found is not None, "no roundoff counter-example found (adjust the search, not the helper)"
    h, r, j = found
    out = D.post_infiltration_depth(arr(h), arr(r), arr(j), np.ones((1, 1), dtype=bool))
    naive = max(h - max(j - r, 0.0), 0.0)
    assert naive > 0.0 and out[0, 0] == 0.0  # the explicit complete branch wins


def test_inactive_cells_are_not_forced_by_the_complete_branch_and_saturation_return_is_not_an_input():
    old = arr(0.0, 0.002)
    rain = arr(0.0, 0.0)
    intake = arr(0.0, 0.0)
    active = np.array([[False, True]])
    out = D.post_infiltration_depth(old, rain, intake, active)
    np.testing.assert_array_equal(out, old)  # overflow/return never raises d(1): only the old depth, rain and intake enter
    import inspect

    assert "saturation" not in " ".join(inspect.signature(D.post_infiltration_depth).parameters)


def test_property_bounds_over_random_states():
    rng = np.random.default_rng(5)
    old = rng.uniform(0.0, 0.02, (40, 50))
    rain = rng.uniform(0.0, 0.003, old.shape)
    intake = rng.uniform(0.0, 0.025, old.shape)
    active = rng.random(old.shape) > 0.2
    out = D.post_infiltration_depth(old, rain, intake, active)
    assert np.all(out >= 0.0) and np.all(out <= old + 1e-18)  # never above the old depth
    np.testing.assert_array_equal(out[intake <= rain], old[intake <= rain])
    np.testing.assert_array_equal(out, hpre_reference(old, rain, intake, active))


def test_real_column_step_reconstruction_matches_the_hydrology_hpre():
    """Uses the actual column kernel quantities (`rain_m`, `intake_m`) exactly as `storm.coupled_step` does."""
    from maple_syrup.infiltration import column_parameters, column_step

    n = (1, 4)

    def full(v):
        return np.full(n, v)

    params = column_parameters(model="fixed_ksat", ksat_m_per_s=np.array([[1e-3, 2e-6, 1e-3, 1e-9]]), suction_m=full(0.0236),
                               drainage_parameter=full(0.05), theta_sat=full(0.4), soil_thickness_m=full(0.1))
    old = arr(0.0, 0.001, 0.0005, 0.002)
    soil = arr(0.0, 0.0, 0.0, 0.0399995)
    col = column_step(params, old, soil, full(1.0e-6), 1.0)
    active = np.ones(n, dtype=bool)
    rain, intake = np.asarray(col.rain_m), np.asarray(col.intake_m)
    out = D.post_infiltration_depth(old, rain, intake, active)
    np.testing.assert_array_equal(out, hpre_reference(old, rain, intake, active))
    assert np.all(out <= old)  # d(1) is never above the old depth (rain/overflow are not added)


@pytest.mark.parametrize("bad", ["shape", "dtype", "negative", "nan", "list", "buffer_alias"])
def test_inputs_are_validated_and_never_mutated(bad):
    old, rain, intake = arr(0.01, 0.02), arr(0.001, 0.001), arr(0.002, 0.0)
    active = np.ones(old.shape, dtype=bool)
    kwargs = {}
    if bad == "shape":
        rain = np.zeros((2, 2))
    elif bad == "dtype":
        intake = intake.astype(np.float32)
    elif bad == "negative":
        old = arr(-0.01, 0.02)
    elif bad == "nan":
        intake = arr(np.nan, 0.0)
    elif bad == "list":
        old = [[0.01, 0.02]]
    else:
        kwargs["out"] = old
    given = (old, rain, intake)
    before = [a.copy() if isinstance(a, np.ndarray) else None for a in given]
    with pytest.raises(D.DriverError):
        D.post_infiltration_depth(old, rain, intake, active, **kwargs)
    for a, b in zip(given, before, strict=True):  # a refusal never changes a caller array
        if b is not None:
            np.testing.assert_array_equal(a, b)  # NaNs in the same places compare equal


def test_selection_returns_the_accepted_objects_for_previous_and_current_and_computes_the_native_depth():
    prev, new = arr(0.01, 0.02), arr(0.03, 0.04)
    assert D.select_law_depth("previous", prev, new) is prev  # exactly the accepted behaviour (the same object)
    assert D.select_law_depth("current", prev, new) is new

    class Col:
        rain_m = arr(0.001, 0.001)
        intake_m = arr(0.004, 0.0)

    out = D.select_law_depth("post_infiltration", prev, new, old_state_depth=prev, column=Col, active=np.ones(prev.shape, bool))
    assert out is not prev and out is not new
    np.testing.assert_allclose(out, [[0.01 - 0.003, 0.02]], atol=1e-15)
    assert prev[0, 0] == 0.01  # the previous array was not modified
    with pytest.raises(D.DriverError, match="unknown depth time level"):
        D.select_law_depth("later", prev, new)


def _alias_case():
    return arr(0.4), arr(0.1), arr(0.2), np.ones((1, 1), dtype=bool)


@pytest.mark.parametrize("which", ["out_is_rain", "out_is_intake", "out_is_old", "out_view_of_old", "scratch_is_rain",
                                   "scratch_is_intake", "scratch_view_of_old", "out_is_scratch", "scratch_view_of_out"])
def test_any_memory_overlap_of_a_buffer_with_an_input_or_the_other_buffer_is_refused_before_any_write(which):
    """Root reproducer: h=.4, rain=.1, intake=.2, out=rain used to return AND MUTATE rain to .3."""
    old, rain, intake, active = _alias_case()
    out, scratch = np.empty_like(old), np.empty_like(old)
    kwargs = {"out": out, "scratch": scratch}
    if which == "out_is_rain":
        kwargs["out"] = rain
    elif which == "out_is_intake":
        kwargs["out"] = intake
    elif which == "out_is_old":
        kwargs["out"] = old
    elif which == "out_view_of_old":
        kwargs["out"] = old[:, :]  # a view of the same memory
    elif which == "scratch_is_rain":
        kwargs["scratch"] = rain
    elif which == "scratch_is_intake":
        kwargs["scratch"] = intake
    elif which == "scratch_view_of_old":
        kwargs["scratch"] = old.reshape(1, 1)
    elif which == "out_is_scratch":
        kwargs["scratch"] = kwargs["out"]
    else:  # scratch_view_of_out
        kwargs["scratch"] = kwargs["out"][:, :]
    before = [a.copy() for a in (old, rain, intake)]
    with pytest.raises(D.DriverError, match="share"):
        D.post_infiltration_depth(old, rain, intake, active, **kwargs)
    for a, b in zip((old, rain, intake), before, strict=True):
        np.testing.assert_array_equal(a, b)  # nothing was written
    assert rain[0, 0] == 0.1  # the reproducer's value stays 0.1, not 0.3


def test_the_root_reproducer_no_longer_mutates_the_input():
    old, rain, intake, active = _alias_case()
    with pytest.raises(D.DriverError):
        D.post_infiltration_depth(old, rain, intake, active, out=rain)
    assert rain[0, 0] == 0.1 and old[0, 0] == 0.4 and intake[0, 0] == 0.2


@pytest.mark.parametrize("bad", ["float32", "int64", "list", "wrong_shape", "readonly", "noncontiguous", "scalar"])
def test_output_buffers_are_validated_before_any_mutation(bad):
    old, rain, intake = arr(0.4, 0.5), arr(0.1, 0.1), arr(0.2, 0.2)
    active = np.ones(old.shape, dtype=bool)
    out, scratch = np.empty_like(old), np.empty_like(old)
    if bad == "float32":
        out = np.empty(old.shape, dtype=np.float32)
    elif bad == "int64":
        scratch = np.empty(old.shape, dtype=np.int64)
    elif bad == "list":
        out = [[0.0, 0.0]]
    elif bad == "wrong_shape":
        out = np.empty((2, 2))
    elif bad == "readonly":
        scratch.flags.writeable = False
    elif bad == "noncontiguous":
        out = np.zeros((1, 4))[:, ::2]  # shape (1, 2) but strided: not C-contiguous
    else:
        out = 1.0
    before = [a.copy() for a in (old, rain, intake)]
    with pytest.raises(D.DriverError):
        D.post_infiltration_depth(old, rain, intake, active, out=out, scratch=scratch)
    for a, b in zip((old, rain, intake), before, strict=True):
        np.testing.assert_array_equal(a, b)


def test_active_mask_shape_is_validated_and_ordinary_preallocated_math_is_unchanged():
    old, rain, intake = arr(0.01, 0.02), arr(0.001, 0.001), arr(0.004, 0.0)
    with pytest.raises(D.DriverError, match="active"):
        D.post_infiltration_depth(old, rain, intake, np.ones((2, 2), dtype=bool))
    fresh = D.post_infiltration_depth(old, rain, intake, np.ones(old.shape, bool))
    buffered = D.post_infiltration_depth(old, rain, intake, np.ones(old.shape, bool), np.empty_like(old), np.empty_like(old))
    np.testing.assert_array_equal(fresh, buffered)  # the arithmetic is the same with or without buffers


def test_reusable_buffers_are_used_without_aliasing_the_inputs():
    old, rain, intake = arr(0.01, 0.02), arr(0.001, 0.001), arr(0.004, 0.0)
    out, scratch = np.empty_like(old), np.empty_like(old)
    result = D.post_infiltration_depth(old, rain, intake, np.ones(old.shape, bool), out, scratch)
    assert result is out
    again = D.post_infiltration_depth(old, rain, intake, np.ones(old.shape, bool), out, scratch)
    np.testing.assert_array_equal(again, result)
