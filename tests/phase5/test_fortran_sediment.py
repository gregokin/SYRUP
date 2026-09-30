"""Compare wet laws with executed unchanged MAHLERAN equations (not routing)."""
from __future__ import annotations

import numpy as np
import pytest
from sediment_reference import build, locate_toolchain, run

from maple_syrup.sediment_physics import (
    KE_MODELS,
    median_diameter_m,
    physics_grid,
    plot1_sediment_parameters,
    sediment_physics_step,
)


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    if locate_toolchain() is None:
        pytest.skip('original MAHLERAN equation test needs configured gfortran')
    return build(tmp_path_factory.mktemp('sediment_original') / 'build')


def test_original_wet_laws(reference, tmp_path):
    fractions = np.array([.1, .15, .25, .25, .15, .1])
    p = plot1_sediment_parameters(ke_vegetation_form='legacy_literal', distance_convention='legacy_literal')
    d50 = float(median_diameter_m(fractions.reshape(1, 1, 6), p.diameter_m)[0, 0])
    assert d50 == pytest.approx(.000375, rel=0, abs=1e-18)
    # Diffuse, rain transitional with each KE model, concentrated/suspended.
    cases = [[.005, .02, 1e-5, .03, 20., d50, 2, *fractions],
             [.005, .2, 1e-5, .03, 20., d50, 2, *fractions],
             [.005, .2, 1e-5, .03, 0., d50, 1, *fractions],
             [.03, .2, 1e-5, .03, 20., d50, 2, *fractions]]
    values = run(reference, cases, tmp_path / 'run')
    for row, case in enumerate(cases):
        h, v, rain, slope, veg, _, mode = case[:7]
        # KE model selection string comes from the public parameter API.
        params = plot1_sediment_parameters(ke_vegetation_form='legacy_literal', ke_model=KE_MODELS[mode - 1],
                                           distance_convention='legacy_literal')
        grid = physics_grid(np.array([[slope]]), np.ones((1, 1), bool), .25)
        s = sediment_physics_step(params, grid, np.array([[h]]), np.array([[v]]), np.array([[rain]]),
             np.array([[veg / 100]]), fractions.reshape(1, 1, 6), np.zeros((1, 1, 6)), 1.)
        re = v * h / params.kinematic_viscosity_m2_s
        expected_rain = values.rain_rate[row] * .25 if re < 2500 else np.zeros(6)
        expected_flow = values.flow_rate[row] * .25 if re > 500 else np.zeros(6)
        # Original COMMON parameters/literals are largely REAL32. FP64
        # implementation is compared to 20 ppm, not falsely called bitwise.
        np.testing.assert_allclose(s.raindrop_pickup_kg[0, 0], expected_rain, rtol=2e-5, atol=1e-16)
        np.testing.assert_allclose(s.flow_pickup_kg[0, 0], expected_flow, rtol=2e-5, atol=1e-16)
        lengths = values.distance[row]
        speeds = values.speed[row]
        if re < 2500:
            selected = np.zeros(6, dtype=int)
        else:
            # Authored fixture: the van Rijn thresholds put phi1..4 in
            # suspension and phi5..6 in bedload for h=.03m, S=.03.
            np.testing.assert_array_equal(s.regime[0, 0], [6, 6, 6, 6, 5, 5])
            selected = np.array([2, 2, 2, 2, 1, 1])
        expected_lengths = lengths[selected, np.arange(6)]
        assert np.all(expected_lengths > 0), 'selected laws must be within original finite domain'
        np.testing.assert_allclose(s.travel_distance_m()[0, 0], expected_lengths, rtol=2e-5, atol=1e-16)
        np.testing.assert_allclose(s.sediment_velocity_m_s[0, 0], speeds[selected, np.arange(6)], rtol=2e-5, atol=1e-16)


def test_original_skipped_concentrated_law_has_no_stale_velocity(reference, tmp_path):
    result = run(reference, [[.0005, .03, 1e-5, .03, 20., .0004, 2,
                            .1, .15, .25, .25, .15, .1]], tmp_path / 'skipped')
    assert np.all(result.law_applies[:, 0])
    assert not np.any(result.law_applies[:, 1])
    assert np.isnan(result.distance[:, 1]).all()
    assert np.isnan(result.speed[:, 1]).all()


def test_default_plot1_coefficients_match_original_xml():
    import xml.etree.ElementTree as ET

    from sediment_reference import ROOT
    xml = ET.parse(ROOT / 'mahleran_input.xml').getroot()
    params = plot1_sediment_parameters()
    for tag, actual in [('Raindrop_detachment_a_parameter_size', params.raindrop_a),
                        ('Raindrop_detachment_b_parameter_size', params.raindrop_b),
                        ('Raindrop_detachment_c_parameter_size', params.raindrop_c)]:
        values = [float(xml.find('.//' + tag + '/phi_' + str(i)).text) for i in range(1, 7)]
        np.testing.assert_array_equal(actual, values)

    expected_hs = [float(xml.find('.//Raindrop_detachment_max_parameter_size/phi_' + str(i)).text)
                   for i in range(1, 7)]
    np.testing.assert_allclose(params.raindrop_max_depth_per_reference_s_m, np.array(expected_hs) * .001)
    assert params.flow_detachment_depth_scale_m == float(xml.find('.//active_layer_sensitivity').get('value')) * .001
    assert params.particle_density_kg_m3 == float(xml.find('.//particle_density').get('value')) * 1000
    assert params.reference_interval_s == float(xml.find('.//time_step').get('value'))
