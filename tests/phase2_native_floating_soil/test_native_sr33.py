"""NEW (task sr33_default_20261120): the native floating-soil owner CONSUMES the Saxton-Rawls 33 kPa field-capacity default at initialization and at the event-boundary texture refresh, with exact-rational
independent controls, conservation / once-only export, preserved matrix history and the new config identity. CPU (NumPy owner); no CUDA claim. The existing E3 suite is unchanged (it never compares an
absolute field-capacity value, so it cannot prove adoption)."""
from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
import pytest
from builders import config, stratified, swap_classes

from maple_syrup import compact_soil_profile as csp
from maple_syrup import native_floating_soil as nfs
from maple_syrup import soil_texture_properties as stp


def theta33_exact(S, C, OM, rho_kg):
    """Independent exact-rational S&R2006 Table 1 Eq 2, 3, 6/9 final density-adjusted theta_33 (coefficients typed from the paper, not from the module)."""
    F = Fraction
    S, C, OM, rho = F(S), F(C), F(OM), F(rho_kg) / 1000
    t = -F("0.251") * S + F("0.195") * C + F("0.011") * OM + F("0.006") * S * OM - F("0.027") * C * OM + F("0.452") * S * C + F("0.299")
    th33n = t + F("1.283") * t * t - F("0.374") * t - F("0.015")
    t = F("0.278") * S + F("0.034") * C + F("0.022") * OM - F("0.018") * S * OM - F("0.027") * C * OM - F("0.584") * S * C + F("0.078")
    g = t + F("0.636") * t - F("0.107")
    thsn = th33n + g - F("0.097") * S + F("0.043")
    ths = 1 - rho / F("2.65")
    return float(th33n - F("0.2") * (thsn - ths))


def expected_fc(bed, cfg, cells):
    """`(L, len(cells))` expected final theta_33 per material layer (NaN for a zero-mass slot) from the SAME actual compact layer composition the owner reads, mapped through the explicit class texture."""
    L = cfg.layers.n_layers
    out = np.full((L, len(cells)), np.nan)
    it = csp.iter_compact_soil_profile(bed.voxel_column, bed.active_layer, bed.geometry, bed.grain_classes, bed.mass_resolution_kg, cfg.layers, indices=np.asarray(cells), active_mask=bed.active_mask,
                                       memory_budget_bytes=1 << 30, batch_cells=len(cells), backend="numpy", coverage="partial_over_base")
    for batch in it:
        frac, total = np.asarray(batch.layer_fraction), np.asarray(batch.layer_total_kg)
        tex = np.einsum("lbk,kc->lbc", frac, cfg.grain_texture)
        for l in range(L):
            for b, c in enumerate(batch.indices):
                if total[l, b] > 0.0:
                    out[l, list(cells).index(int(c))] = theta33_exact(tex[l, b, 0], tex[l, b, 2], cfg.organic_matter_percent[l], cfg.bulk_density_kg_m3)
    return out


def test_config_declares_and_binds_the_versioned_fc_method():
    cfg = config()
    assert cfg.field_capacity_method == stp.METHOD_SR33 == "saxton_rawls_33kpa"
    assert cfg.sha256 == config().sha256  # deterministic
    # a pre-version identity (the same inputs hashed without the method tag) cannot equal the new one: re-derive the OLD hash recipe and compare
    import hashlib

    h = hashlib.sha256()
    for a in (cfg.layers.depth_edges_m, cfg.grain_texture, cfg.organic_matter_percent, cfg.latent_theta_field_capacity, cfg.latent_theta_saturation, cfg.diffusivity_m2_s,
              np.array([cfg.bulk_density_kg_m3, cfg.theta_residual, cfg.theta_evaporation_stress, cfg.evaporation_factor, cfg.matrix_conductivity_m_s, cfg.matrix_suction_m,
                        cfg.matrix_vegetation_factor, cfg.initial_wetting_m, cfg.initial_cumulative_intake_m])):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(f"{nfs.SCHEMA}|{cfg.bottom_drainage}|{cfg.backend}".encode())
    assert h.hexdigest() != cfg.sha256


def test_initialization_consumes_the_33_default_and_not_the_6kpa_hybrid():
    bed = stratified()
    cfg = config()
    o = nfs.initialize_native_floating_soil(bed, cfg, initial_relative_saturation=0.5)
    fc = np.asarray(o.column._maps["theta_field_capacity"])
    sat = np.asarray(o.column._maps["theta_saturation"])
    cells = [0, 1, 2]
    exp = expected_fc(bed, cfg, cells)
    mass = ~np.isnan(exp)
    assert mass.sum() >= 7 and np.all(np.abs(fc[mass] - exp[mass]) <= 1e-12)
    assert np.all(fc[~mass] == cfg.latent_theta_field_capacity[0])  # a zero-mass slot keeps the explicit latent value (authoritative prescribed input)
    assert np.all(fc[mass] < sat[mass]) and np.all(fc[mass] < 0.4)
    # the deprecated hybrid would have given strictly larger values for the same actual composition: prove the owner did not use it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stp.FieldCapacityDeprecationWarning)
        legacy_fc = _legacy_fc(bed, cfg, cells)
    assert np.all(fc[mass] < legacy_fc[mass] - 1e-6)


def _legacy_fc(bed, cfg, cells):
    out = np.full((cfg.layers.n_layers, len(cells)), np.nan)
    it = csp.iter_compact_soil_profile(bed.voxel_column, bed.active_layer, bed.geometry, bed.grain_classes, bed.mass_resolution_kg, cfg.layers, indices=np.asarray(cells), active_mask=bed.active_mask,
                                       memory_budget_bytes=1 << 30, batch_cells=len(cells), backend="numpy", coverage="partial_over_base")
    for batch in it:
        frac, total = np.asarray(batch.layer_fraction), np.asarray(batch.layer_total_kg)
        tex = np.einsum("lbk,kc->lbc", frac, cfg.grain_texture)
        for l in range(cfg.layers.n_layers):
            sel = np.flatnonzero(total[l] > 0.0)
            if sel.size:
                r = stp.estimate_soil_texture_properties(sand_fraction=np.ascontiguousarray(tex[l:l + 1, sel, 0]), silt_fraction=np.ascontiguousarray(tex[l:l + 1, sel, 1]),
                                                         clay_fraction=np.ascontiguousarray(tex[l:l + 1, sel, 2]), organic_matter_percent=np.full((1, sel.size), cfg.organic_matter_percent[l]),
                                                         bulk_density_kg_m3=np.full((1, sel.size), cfg.bulk_density_kg_m3), n_layers=1, n_cells=sel.size, memory_budget_bytes=1 << 30,
                                                         field_capacity_method=stp.METHOD_TIETJEN_6KPA_DEPRECATED)
                out[l, batch.indices[sel]] = r.array("theta_field_capacity_6kpa")[0]
    return out


def test_event_boundary_refresh_consumes_the_33_default_conserves_and_preserves_history():
    bed = stratified()
    cfg = config()
    o = nfs.initialize_native_floating_soil(bed, cfg, initial_relative_saturation=0.99)
    child = swap_classes(bed, 0, 1, 2)  # a zero-net-elevation composition change of cell 0 (actual bed texture refresh)
    cand = nfs.stage_native_refresh(o, bed, child, mode="properties")
    assert list(cand.dirty_indices) == [0] and cand.reconciled
    fc_new = np.asarray(cand.column._maps["theta_field_capacity"])
    exp = expected_fc(child, cfg, [0])
    m = ~np.isnan(exp[:, 0])
    assert m.any() and np.all(np.abs(fc_new[m, 0] - exp[m, 0]) <= 1e-12)
    assert not np.array_equal(fc_new[:, 0], np.asarray(o.column._maps["theta_field_capacity"])[:, 0])  # the refresh genuinely changed the property
    # conservation / once-only: stored before = stored after + surface return + bottom export (receipt), untouched cells bitwise
    rc = cand.receipt
    assert abs(rc.residual_m3) <= rc.bound_m3
    w0, w1 = np.asarray(o.state.water_m), np.asarray(cand.state.water_m)
    assert np.array_equal(w0[:, 1:], w1[:, 1:])
    # persistent matrix history preserved exactly (no reset), generation advances once
    assert np.array_equal(cand.matrix_history.wetting_m, o.matrix_history.wetting_m) and np.array_equal(cand.matrix_history.cumulative_intake_m, o.matrix_history.cumulative_intake_m)
    o2 = nfs.adopt_native_refresh(o, cand)
    assert o2.generation == 1 and o2.config.sha256 == cfg.sha256
    assert o2.totals["bottom_export_m3"] == pytest.approx(rc.bottom_export_m3) and o2.totals["surface_return_m3"] == pytest.approx(rc.surface_return_m3)
    with pytest.raises(nfs.NativeStaleError):  # a duplicate adoption can never credit the export/return twice
        nfs.adopt_native_refresh(o, cand)
    # after the zero-time cascade against the (lower) 33 kPa capacity no changed-column layer of the open column stays above its field capacity
    fcol = np.asarray(o2.column._maps["theta_field_capacity"])
    from maple_syrup import floating_soil_layers as fl

    h = np.stack([fl.porous_thickness(o2.column, k) for k in range(cfg.layers.n_layers)])
    assert np.all(np.asarray(o2.state.water_m)[:, 0] <= fcol[:, 0] * h[:, 0] + 1e-12)


def test_final_theta_33_override_remains_authoritative_in_the_native_owner():
    ov = np.full((3, 3), np.nan)
    ov[1, 2] = 0.2  # a measured final theta_33 for layer 1 of cell 2
    bed = stratified()
    cfg = config(overrides={"theta_33_override": ov})
    o = nfs.initialize_native_floating_soil(bed, cfg, initial_relative_saturation=0.5)
    assert np.asarray(o.column._maps["theta_field_capacity"])[1, 2] == 0.2
    assert cfg.sha256 != config().sha256


def test_native_config_cannot_select_another_fc_method_by_replace_or_constructor():
    """Correction01: `field_capacity_method` is a read-only constant of the native config (not a dataclass field), so the identity cannot be kept while the semantics change."""
    import dataclasses

    cfg = config()
    assert not any(f.name == "field_capacity_method" for f in dataclasses.fields(cfg))
    with pytest.raises(TypeError):
        dataclasses.replace(cfg, field_capacity_method=stp.METHOD_TIETJEN_6KPA_DEPRECATED)
    kwargs = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    with pytest.raises(TypeError):
        nfs.NativeFloatingConfig(**kwargs, field_capacity_method=stp.METHOD_TIETJEN_6KPA_DEPRECATED)
    with pytest.raises(AttributeError):
        cfg.field_capacity_method = stp.METHOD_TIETJEN_6KPA_DEPRECATED
    assert cfg.field_capacity_method == stp.METHOD_SR33


def test_native_calls_pass_method_33_to_the_provider_at_initialization_and_refresh(monkeypatch):
    seen = []
    real = stp.estimate_soil_texture_properties

    def spy(**kw):
        seen.append(kw.get("field_capacity_method"))
        return real(**kw)

    monkeypatch.setattr(stp, "estimate_soil_texture_properties", spy)
    bed = stratified()
    o = nfs.initialize_native_floating_soil(bed, config(), initial_relative_saturation=0.9)
    n_init = len(seen)
    nfs.stage_native_refresh(o, bed, swap_classes(bed, 0, 1, 2), mode="properties")
    assert n_init > 0 and len(seen) > n_init and set(seen) == {stp.METHOD_SR33}
