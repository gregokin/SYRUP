"""Builder and column validation of the subsurface column (host NumPy only). Nothing here was run by its author;
Codex records results."""
from __future__ import annotations

import numpy as np
import pytest
from column_cases import AREA, BIG, EDGES, FC, RES, SAT, STRESS, column, layers

from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.subsurface import (
    LAYER_OPERATION_TERMS,
    SubsurfaceError,
    column_state,
    moisture,
    prepare_column,
    soil_layers,
    storage_m3,
    uniform_state,
)


def test_one_and_many_layers_build():
    one = soil_layers([0.0, 0.3], theta_residual=0.05, theta_field_capacity=0.2, theta_saturation=0.4,
                      diffusivity_m2_s=0.0, theta_evaporation_stress=0.1)
    assert one.n_layers == 1
    assert one.interface_conductance_m_s.shape == (0,) and one.pair_inverse_thickness.shape == (0,)
    many = soil_layers(np.linspace(0.0, 3.0, 61), theta_residual=0.05, theta_field_capacity=0.2,
                       theta_saturation=0.4, diffusivity_m2_s=1e-6, theta_evaporation_stress=0.1)
    assert many.n_layers == 60
    np.testing.assert_allclose(many.interface_conductance_m_s, 1e-6 / 0.05, rtol=1e-12)


def test_derived_quantities_are_exact_for_dyadic_inputs():
    lay = layers(diffusivity=2e-6)
    np.testing.assert_array_equal(lay.thickness_m, [0.25, 0.75])
    np.testing.assert_array_equal(lay.residual_m, [0.015625, 0.046875])
    np.testing.assert_array_equal(lay.field_capacity_m, [0.0625, 0.1875])
    np.testing.assert_array_equal(lay.saturation_m, [0.125, 0.375])
    np.testing.assert_array_equal(lay.centre_depth_m, [0.125, 0.625])
    assert lay.interface_conductance_m_s[0] == 2e-6 / 0.5  # D / centre separation, exact power-of-two scaling
    assert lay.pair_inverse_thickness[0] == pytest.approx(4.0 + 4.0 / 3.0, rel=1e-15)
    np.testing.assert_array_equal(lay.capacity_tolerance_m,
                                  volume_roundoff_bound_m3(LAYER_OPERATION_TERMS, 1.0) * lay.saturation_m)


def test_conductance_keeps_the_physical_coefficient_under_refinement():
    coarse = soil_layers([0.0, 0.1, 0.3], theta_residual=0.05, theta_field_capacity=0.2, theta_saturation=0.4,
                         diffusivity_m2_s=3e-7, theta_evaporation_stress=0.1)
    fine = soil_layers([0.0, 0.05, 0.1, 0.2, 0.3], theta_residual=0.05, theta_field_capacity=0.2,
                       theta_saturation=0.4, diffusivity_m2_s=3e-7, theta_evaporation_stress=0.1)
    for lay in (coarse, fine):
        separation = lay.centre_depth_m[1:] - lay.centre_depth_m[:-1]
        np.testing.assert_allclose(lay.interface_conductance_m_s * separation, 3e-7, rtol=1e-15)
    assert coarse.interface_conductance_m_s[0] == pytest.approx(3e-7 / 0.15, rel=1e-15)


@pytest.mark.parametrize("edges", [[0.0], [], [0.0, 0.1, 0.1], [0.0, 0.2, 0.1], [0.1, 0.2], [0.0, np.nan],
                                   [0.0, np.inf], [[0.0, 0.1]], [True, False], ["0", "1"], [-0.0, -0.1]])
def test_invalid_depth_edges_are_refused(edges):
    with pytest.raises(SubsurfaceError):
        soil_layers(edges, theta_residual=RES, theta_field_capacity=FC, theta_saturation=SAT, diffusivity_m2_s=0.0,
                    theta_evaporation_stress=STRESS)


@pytest.mark.parametrize("override", [
    {"res": -0.1}, {"res": 0.3, "fc": 0.25}, {"fc": 0.6}, {"sat": 1.5}, {"res": 0.5, "fc": 0.5, "sat": 0.5},
    {"stress": RES}, {"stress": 0.01}, {"stress": -1.0}, {"factor": -1.0}, {"diffusivity": -1e-6},
    {"res": np.nan}, {"sat": np.inf}, {"stress": float("nan")}, {"factor": float("inf")}, {"res": True},
    {"factor": True}, {"diffusivity": True}, {"stress": True}, {"res": [0.05]}, {"diffusivity": [1e-6, 1e-6]},
    {"res": [[0.05, 0.05]]}, {"fc": "0.2"}, {"diffusivity": np.nan},
])
def test_invalid_properties_are_refused(override):
    with pytest.raises(SubsurfaceError):
        layers(**override)


EXTREME = {"theta_residual": 0.02, "theta_field_capacity": 0.2, "theta_saturation": 0.4,
           "theta_evaporation_stress": 0.15}


@pytest.mark.parametrize("edges, diffusivity", [
    ([0.0, 1e-310, 2e-310], 1.0),  # subnormal thickness: inverse thickness and conductance overflow
    ([0.0, 5e-324, 1e-323], 0.0),  # smallest subnormal thickness, no diffusion: inverse thickness still overflows
    ([0.0, 1e-3, 2e-3], 1e308),  # finite D, mm layers: conductance overflows
    ([0.0, 1e-100, 2e-100], 1e200),  # finite conductance 1e300 but exchange rate 2e400 overflows
])
def test_finite_inputs_with_unrepresentable_derived_values_are_refused(edges, diffusivity):
    with pytest.raises(SubsurfaceError):
        soil_layers(edges, diffusivity_m2_s=diffusivity, **EXTREME)


def test_extreme_but_representable_inputs_are_accepted_without_flooring():
    lay = soil_layers([0.0, 1e308, 1.7e308], diffusivity_m2_s=1.0, **EXTREME)
    assert lay.n_layers == 2
    for name in ("thickness_m", "centre_depth_m", "interface_conductance_m_s", "pair_inverse_thickness",
                 "saturation_m", "capacity_tolerance_m"):
        assert np.all(np.isfinite(getattr(lay, name))), name
    assert np.all(lay.saturation_m > 0.0)
    tiny = soil_layers([0.0, 1e-200, 2e-200], diffusivity_m2_s=1e-200, **EXTREME)
    assert np.all(np.isfinite(tiny.interface_conductance_m_s)) and np.all(tiny.saturation_m > 0.0)
    assert tiny.interface_conductance_m_s[0] == pytest.approx(1.0, rel=1e-15)


def test_single_layer_refuses_negative_or_misshaped_diffusivity():
    with pytest.raises(SubsurfaceError):
        soil_layers([0.0, 0.3], theta_residual=0.05, theta_field_capacity=0.2, theta_saturation=0.4,
                    diffusivity_m2_s=-1.0, theta_evaporation_stress=0.1)
    with pytest.raises(SubsurfaceError):
        soil_layers([0.0, 0.3], theta_residual=0.05, theta_field_capacity=0.2, theta_saturation=0.4,
                    diffusivity_m2_s=[1e-6], theta_evaporation_stress=0.1)


def test_per_layer_sequences_and_gravity_flag():
    lay = layers(res=[0.05, 0.0625], fc=[0.2, 0.25], sat=[0.4, 0.5])
    np.testing.assert_array_equal(lay.theta_residual, [0.05, 0.0625])
    with pytest.raises(SubsurfaceError):
        soil_layers(EDGES, theta_residual=RES, theta_field_capacity=FC, theta_saturation=SAT, diffusivity_m2_s=0.0,
                    theta_evaporation_stress=STRESS, gravity_drainage=1)


def test_layer_arrays_are_frozen_owned_copies():
    res = np.array([0.0625, 0.0625])
    lay = layers(res=res)
    res[0] = 0.2
    assert lay.theta_residual[0] == 0.0625
    assert not lay.thickness_m.flags.writeable
    with pytest.raises(ValueError):
        lay.saturation_m[0] = 1.0


@pytest.mark.parametrize("mask", [np.ones(3, dtype=np.int64), np.ones((1, 3), dtype=bool), np.zeros(0, dtype=bool),
                                  [True, True], np.ma.masked_array(np.ones(3, dtype=bool))])
def test_invalid_masks_are_refused(mask):
    with pytest.raises(SubsurfaceError):
        prepare_column(layers(), mask, AREA, reserve_bytes=0, available_bytes_fn=lambda: BIG)


@pytest.mark.parametrize("kwargs", [{"cell_area_m2": 0.0}, {"cell_area_m2": -1.0}, {"cell_area_m2": np.nan},
                                    {"cell_area_m2": True}, {"backend": "torch"}, {"reserve_bytes": -1},
                                    {"reserve_bytes": True}, {"reserve_bytes": 1.5}, {"available_bytes_fn": 3}])
def test_invalid_column_options_are_refused(kwargs):
    options = {"cell_area_m2": AREA, "backend": "numpy", "reserve_bytes": 0, "available_bytes_fn": lambda: BIG}
    options.update(kwargs)
    area = options.pop("cell_area_m2")
    with pytest.raises(SubsurfaceError):
        prepare_column(layers(), np.ones(2, dtype=bool), area, **options)


def test_column_owns_a_mask_copy_and_counts_active_cells():
    mask = np.array([True, False, True])
    col = prepare_column(layers(), mask, AREA, reserve_bytes=0, available_bytes_fn=lambda: BIG)
    mask[0] = False
    assert col.n_active == 2 and col.n_cells == 3
    np.testing.assert_array_equal(col.active, [True, False, True])
    assert col.active_u8.dtype == np.uint8 and col.active_u8.tolist() == [1, 0, 1]


@pytest.mark.parametrize("bad", [
    np.zeros((2, 4), dtype=np.float32), np.zeros((4, 2)), np.zeros((2, 4, 1)), np.asfortranarray(np.zeros((2, 4))),
    np.zeros((2, 8))[:, ::2], np.ma.masked_array(np.zeros((2, 4))), [[0.0] * 4] * 2,
])
def test_state_requires_exact_float64_c_contiguous_ndarray(bad):
    col = column(layers(), n_cells=4)
    with pytest.raises(SubsurfaceError):
        column_state(col, bad)


def test_state_time_must_be_finite_and_array_is_copied():
    col = column(layers(), n_cells=2)
    water = np.zeros((2, 2))
    with pytest.raises(SubsurfaceError):
        column_state(col, water, t_s=float("nan"))
    with pytest.raises(SubsurfaceError):
        column_state(col, water, t_s=True)
    state = column_state(col, water, 5.0)
    water[0, 0] = 1.0
    assert state.water_m[0, 0] == 0.0 and state.t_s == 5.0 and not np.shares_memory(state.water_m, water)


def test_uniform_state_values_and_bounds():
    col = column(layers(), n_cells=3)
    state = uniform_state(col, [0.25, 0.5])
    np.testing.assert_array_equal(state.water_m, [[0.0625] * 3, [0.375] * 3])
    np.testing.assert_array_equal(moisture(col, state), [[0.25] * 3, [0.5] * 3])
    assert storage_m3(col, state) == (0.0625 + 0.375) * 3 * AREA
    for theta in ([0.6, 0.1], [-0.1, 0.1], [0.1], [0.1, 0.1, 0.1], [np.nan, 0.1]):
        with pytest.raises(SubsurfaceError):
            uniform_state(col, theta)


def test_storage_excludes_inactive_cells():
    col = column(layers(), n_cells=2, active=[True, False])
    state = column_state(col, np.array([[0.125, 0.125], [0.375, 0.375]]))
    assert storage_m3(col, state) == 0.5 * AREA
