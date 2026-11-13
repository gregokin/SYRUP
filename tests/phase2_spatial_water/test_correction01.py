"""S2-A.2 correction01 regressions: (1) `spatial_uniform_state` checks bounded input METADATA and admits every lifetime (incl. the L-sized conversion) BEFORE converting anything; (2) the global storage
query refuses an unrepresentable/non-finite volume with a typed error; (3) the parameter-only owner flag vs the explicit NumPy trial API is stated in `summary()`. Author-written."""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from conftest import BIG, make_spec, params, prepare

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss


@pytest.fixture
def column():
    p = params(np.random.default_rng(1), 3, 12)
    return prepare(make_spec(p))


@pytest.fixture
def conversion_spy(monkeypatch):
    calls = []
    real = ss._layer_array

    def spy(*a, **k):
        calls.append(a[1] if len(a) > 1 else None)
        return real(*a, **k)

    monkeypatch.setattr(ss, "_layer_array", spy)
    return calls


# ---- 1 uniform state: metadata first, admission second, conversion last ---------------------------------------------------------------------
@pytest.mark.parametrize("theta", [[0.1, 0.1, 0.1], (0.1, 0.1, 0.1), np.array([0.1, 0.1, 0.1]), 0.1, np.float64(0.1), np.array(0.1), np.array([1, 0, 1])])
def test_a_refused_admission_converts_and_allocates_nothing(column, conversion_spy, theta):
    tracemalloc.start()
    with pytest.raises(sp.SpatialLayerMemoryError):
        sp.spatial_uniform_state(column, theta, 0.0, memory_budget_bytes=0)  # Root's repro: budget 0 used to create the owned theta copy first
    with pytest.raises(sp.SpatialLayerMemoryError):
        sp.spatial_uniform_state(column, theta, 0.0, memory_budget_bytes=BIG, available_bytes_fn=lambda: 0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert conversion_spy == [] and peak < 100_000  # the accepted conversion was never reached


def test_the_estimate_counts_the_l_sized_conversion_lifetimes_and_the_budget_boundary_is_exact(conversion_spy):
    L, N = 40, 6
    p = params(np.random.default_rng(2), L, N)
    col = prepare(make_spec(p))
    est = sp.estimate_spatial_operation_memory("uniform_state", L, N)
    assert est.terms["theta_conversion_bytes"] == 3 * 8 * L and est.scratch_bytes == 8 * N + N + 3 * 8 * L + sp.PY_ALLOWANCE_BYTES
    theta = [0.01] * L
    with pytest.raises(sp.SpatialLayerMemoryError):
        sp.spatial_uniform_state(col, theta, 0.0, memory_budget_bytes=est.required_bytes - 1)
    assert conversion_spy == []
    out = sp.spatial_uniform_state(col, theta, 0.0, memory_budget_bytes=est.required_bytes, available_bytes_fn=lambda: est.required_bytes)
    assert out.n_layers == L and len(conversion_spy) == 1  # the accepted conversion ran exactly once, after the admission


def test_estimate_grows_with_the_layer_count_not_only_with_the_cell_rows():
    a, b = sp.estimate_spatial_operation_memory("uniform_state", 10, 100), sp.estimate_spatial_operation_memory("uniform_state", 20, 100)
    assert b.scratch_bytes - a.scratch_bytes == 3 * 8 * 10  # only the conversion term differs in the scratch; the (L, N) result is counted in the outputs


@pytest.mark.parametrize("make", [lambda: [0.1] * 10**6, lambda: tuple([0.1] * 10**6), lambda: np.zeros(2 * 10**6), lambda: np.zeros((3, 3)), lambda: [0.1, 0.1], lambda: np.zeros(2),
                                  lambda: [[0.1, 0.1]] * 3, lambda: [np.zeros(10**5)] * 3])
def test_wrong_length_or_shaped_inputs_are_refused_before_any_conversion(column, conversion_spy, make):
    theta = make()
    tracemalloc.start()
    with pytest.raises(ss.SubsurfaceError):
        sp.spatial_uniform_state(column, theta, 0.0, memory_budget_bytes=BIG)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert conversion_spy == [] and peak < 100_000


def test_objects_with_array_methods_generators_and_other_types_are_never_converted(column, conversion_spy):
    seen = []

    class Sneaky:
        def __array__(self, *a, **k):
            seen.append(1)
            return np.full(10**7, 0.1)

        def __len__(self):
            return 3

        def __getitem__(self, i):
            return 0.1

    for bad in (Sneaky(), (x for x in [0.1, 0.1, 0.1]), {"a": 0.1}, "abc", None, np.ma.masked_array([0.1, 0.1, 0.1]), [True, 0.1, 0.1], [0.1, "x", 0.1], [0.1, None, 0.1],
                np.array(["a", "b", "c"]), np.array([True, False, True]), b"abc", range(3)):
        with pytest.raises(ss.SubsurfaceError):
            sp.spatial_uniform_state(column, bad, 0.0, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceError):  # a numpy.ndarray SUBCLASS is not an exact ndarray
        sp.spatial_uniform_state(column, np.array([0.1, 0.1, 0.1]).view(type(np.ma.masked_array([0.0]))), 0.0, memory_budget_bytes=BIG)
    assert seen == [] and conversion_spy == []


def test_accepted_forms_give_the_same_state_and_values_are_still_checked_after_admission(column):
    ref = sp.spatial_uniform_state(column, [0.04, 0.0, 0.1], 0.0, memory_budget_bytes=BIG).water_m
    for theta in ((0.04, 0.0, 0.1), np.array([0.04, 0.0, 0.1]), [np.float64(0.04), 0, 0.1], np.array([0.04, 0.0, 0.1], dtype=np.float64).reshape(3)):
        np.testing.assert_array_equal(sp.spatial_uniform_state(column, theta, 0.0, memory_budget_bytes=BIG).water_m, ref)
    for bad in ([float("nan"), 0.1, 0.1], [0.1, 0.1, float("inf")], [-1e-9, 0.1, 0.1], [0.1, 0.1, 0.99]):  # values: the accepted conversion/range checks run after admission
        with pytest.raises(ss.SubsurfaceError):
            sp.spatial_uniform_state(column, bad, 0.0, memory_budget_bytes=BIG)


def test_warm_peak_with_many_layers_stays_within_the_l_sized_estimate():
    L, N = 600, 64
    p = params(np.random.default_rng(3), L, N)
    col = prepare(make_spec(p))
    theta = [0.001] * L
    sp.spatial_uniform_state(col, theta, 0.0, memory_budget_bytes=BIG)  # warm
    est = sp.estimate_spatial_operation_memory("uniform_state", L, N)
    tracemalloc.start()
    sp.spatial_uniform_state(col, theta, 0.0, memory_budget_bytes=BIG)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= est.required_bytes, (peak, est.required_bytes)


# ---- 2 the storage query refuses non-finite results ----------------------------------------------------------------------------------
def roots_fixture(area):
    spec = sp.spatial_layer_spec(np.array([0.0, 1.0, 2.0]), theta_residual=0.0, theta_field_capacity=0.25, theta_saturation=0.5, theta_evaporation_stress=0.5, diffusivity_m2_s=1e-7, n_cells=4)
    col = sp.prepare_spatial_column(spec, np.ones(4, dtype=bool), area, backend="numpy", memory_budget_bytes=BIG)
    state = sp.spatial_column_state(col, np.full((2, 4), 0.25), 0.0, memory_budget_bytes=BIG)  # finite, within saturation (0.5 m per layer)
    return col, state


def test_an_unrepresentable_global_volume_is_a_typed_refusal_not_inf():
    col, state = roots_fixture(1e308)
    with pytest.raises(sp.SpatialLayerError, match="not representable") as info:
        sp.spatial_storage_m3(col, state, memory_budget_bytes=BIG)
    assert isinstance(info.value, ss.SubsurfaceError)


def test_representable_large_volumes_and_ordinary_volumes_are_unchanged():
    col, state = roots_fixture(1e307)  # 8 slots x 0.25 m x 1e307 = 2e307 < 1.8e308
    assert sp.spatial_storage_m3(col, state, memory_budget_bytes=BIG) == pytest.approx(8 * 0.25 * 1e307, rel=1e-13)
    col, state = roots_fixture(4.0)
    assert sp.spatial_storage_m3(col, state, memory_budget_bytes=BIG) == pytest.approx(8 * 0.25 * 4.0, rel=1e-13)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_state_with_non_finite_water_has_no_storage_and_is_refused_typed(bad):
    col, _ = roots_fixture(4.0)
    water = np.full((2, 4), 0.25)
    water[1, 2] = bad
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)  # construction stays as documented: values are checked by the next step
    with pytest.raises(sp.SpatialLayerError, match="not representable|non-finite"):
        sp.spatial_storage_m3(col, state, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceStepError):  # ... and the next step reports it with its existing flag
        sp.spatial_step(col, state, 1.0, added_water_m=np.zeros((2, 4)), potential_evaporation_m_s=np.zeros(4), memory_budget_bytes=BIG)


def test_an_inactive_cell_with_huge_water_does_not_poison_the_active_total():
    col = sp.prepare_spatial_column(sp.spatial_layer_spec(np.array([0.0, 1.0]), theta_residual=0.0, theta_field_capacity=0.25, theta_saturation=0.5, theta_evaporation_stress=0.5,
                                                          diffusivity_m2_s=0.0, n_cells=3), np.array([True, False, True]), 2.0, backend="numpy", memory_budget_bytes=BIG)
    water = np.array([[0.1, 1e308, 0.2]])
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    assert sp.spatial_storage_m3(col, state, memory_budget_bytes=BIG) == pytest.approx(0.6, rel=1e-13)


# ---- 3 owner flag vs the explicit NumPy trial API ------------------------------------------------------------------------------------------
def test_summary_states_what_hydrology_step_supported_false_means(column):
    s = column.summary()
    assert s["hydrology_step_supported"] is False and column.hydrology_step_supported is False
    ev = s["water_evolution"]
    assert ev["owner_is_parameter_component_only"] and not ev["legacy_soil_column_compatible"] and not ev["production_or_lifecycle_accepted"]
    assert ev["evolution_backends"] == ["numpy", "numba", "cuda"] and ev["unsupported_backends"] == [] and ev["state_component_tags"] == list(sp.STATE_COMPONENT_TAGS)  # S2-A.4.2: the CUDA trial exists (unqualified)
    assert "asserts no qualification" in ev["cuda_trial_status"] and "Root's source-specific receipts" in ev["cuda_trial_status"] and "unsupported" in ev["cuda_trial_status"]  # static wording: qualification lives in receipts
    for name in ev["explicit_trial_functions"]:
        assert callable(getattr(sp, name))
    assert not hasattr(column, "step") and all(not hasattr(column, n) for n in ev["explicit_trial_functions"])  # the owner itself has no evolution method
    import json

    json.dumps(s)  # the summary stays plain data
    ev["evolution_backends"].append("cuda")  # a copy: the owner's statement cannot be rewritten through the summary
    assert column.summary()["water_evolution"]["evolution_backends"] == ["numpy", "numba", "cuda"]
    assert "owner is parameter-component-only" in sp.SpatialSoilColumn.__doc__.replace("\n", " ") or "parameter-component-only" in sp.SpatialSoilColumn.__doc__


def test_the_trial_state_and_the_owner_use_distinct_tags():
    assert sp.COMPONENT_TAG == "parameter_component_only" and sp.STATE_COMPONENT_TAG == "spatial_explicit_cpu_state"
