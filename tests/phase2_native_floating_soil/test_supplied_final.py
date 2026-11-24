"""Prescribed final properties use actual compact-cell indices, without a PTF."""
import numpy as np
import pytest
from builders import config, stratified
from maple_syrup import native_floating_soil as nfs
from maple_syrup import compact_soil_profile as csp


def maps(n):
    layer, cell = np.indices((3, n))
    return dict(theta_33_override=.15+.01*layer+.001*cell,
        theta_saturation_override=.45+.01*layer+.001*cell,
        saturated_conductivity_override_m_s=1e-5*(1+.1*cell[:1]))


def test_real_compact_initialization_uses_exact_maps_without_estimator(monkeypatch):
    bed = stratified()
    n = bed.geometry.nx*bed.geometry.ny
    supplied = maps(n)
    cfg = config(property_mode='supplied_final', overrides=supplied,
        matrix_conductivity_mode='texture_ks', matrix_conductivity_m_s=None,
        latent_saturated_conductivity_m_s=7e-7)
    sealed_fc = cfg.overrides['theta_33_override'].copy()
    supplied['theta_33_override'][:] = .2
    def forbidden(**kwargs):
        raise AssertionError('unused texture estimator called')
    monkeypatch.setattr(nfs.stp, 'estimate_soil_texture_properties', forbidden)
    owner = nfs.initialize_native_floating_soil(bed, cfg,
        initial_relative_saturation=.25, batch_cells=2)
    seen = 0
    for batch in csp.iter_compact_soil_profile(bed.voxel_column, bed.active_layer,
        bed.geometry, bed.grain_classes, bed.mass_resolution_kg, cfg.layers,
        indices=np.arange(n), active_mask=bed.active_mask, memory_budget_bytes=1<<29,
        batch_cells=2, backend='numpy', coverage='partial_over_base'):
        for layer in range(3):
            for local, cell in enumerate(batch.indices):
                present = batch.layer_total_kg[layer, local] > 0
                expected = sealed_fc[layer, cell] if present else cfg.latent_theta_field_capacity[layer]
                assert owner.column._maps['theta_field_capacity'][layer, cell] == expected
                if layer == 0:
                    expected_ks = cfg.overrides['saturated_conductivity_override_m_s'][0, cell] if present else 7e-7
                    assert owner.ks_m_s[cell] == expected_ks
                seen += 1
    assert seen > 0
    assert not cfg.overrides['theta_33_override'].flags.writeable


@pytest.mark.parametrize('bad', ['missing', 'extra', 'nan', 'dtype', 'shape', 'fc', 'sat', 'ks'])
def test_invalid_final_maps_refuse_before_initialization(bad):
    supplied = maps(3)
    if bad == 'missing': supplied.pop('theta_33_override')
    elif bad == 'extra': supplied['unused'] = np.zeros((3,3))
    elif bad == 'nan': supplied['theta_33_override'][0,0] = np.nan
    elif bad == 'dtype': supplied['theta_33_override'] = supplied['theta_33_override'].astype(np.float32)
    elif bad == 'shape': supplied['theta_33_override'] = supplied['theta_33_override'][:,:2]
    elif bad == 'fc': supplied['theta_33_override'][0,0] = .01
    elif bad == 'sat': supplied['theta_saturation_override'][0,0] = 1.1
    elif bad == 'ks': supplied['saturated_conductivity_override_m_s'][0,0] = 0
    with pytest.raises(nfs.NativeFloatingError):
        config(property_mode='supplied_final', overrides=supplied,
            matrix_conductivity_mode='texture_ks', matrix_conductivity_m_s=None, latent_saturated_conductivity_m_s=7e-7)


def test_default_mode_and_explicit_texture_have_identical_identity():
    assert config().sha256 == config(property_mode='texture').sha256
    assert config().property_mode == 'texture'
