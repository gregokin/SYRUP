"""NEW independent controls for the Saxton-Rawls 33 kPa DEFAULT field capacity (task sr33_default_20261120). The historical 6 kPa oracles stay unchanged and are exercised through the explicit
deprecated method in the adapted suite; nothing here edits an oracle file. CPU lanes only (NumPy, Numba); the CUDA kernel is checked as host-compiled source (test_cuda_source_host_check.py) and is NOT hardware evidence."""
from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
import pytest
from conftest import BIG, FC6, LEGACY, call, call33, random_valid, val

from maple_syrup import soil_texture_properties as tp

FC = "theta_field_capacity"


def mk(rng, L, N):
    """`(S, Si, C, OM, rho_kg)` each `(L, N)`, every element valid under the unchanged domain checks (rejection-sampled against the independent Decimal model)."""
    return [a.reshape(L, N).copy() for a in random_valid(rng, L * N)]


def _frac(x):
    return Fraction(x)


def independent_theta33(S, C, OM, rho_kg):
    """Exact-rational S&R2006 Table 1 Eq 2, 3, 6/9 (final, density-adjusted theta_33) from the paper's published coefficients (written independently of the module's tuples)."""
    S, C, OM = _frac(S), _frac(C), _frac(OM)
    rho = _frac(rho_kg) / 1000
    t = -_frac("0.251") * S + _frac("0.195") * C + _frac("0.011") * OM + _frac("0.006") * S * OM - _frac("0.027") * C * OM + _frac("0.452") * S * C + _frac("0.299")
    th33n = t + _frac("1.283") * t * t - _frac("0.374") * t - _frac("0.015")
    t = _frac("0.278") * S + _frac("0.034") * C + _frac("0.022") * OM - _frac("0.018") * S * OM - _frac("0.027") * C * OM - _frac("0.584") * S * C + _frac("0.078")
    g = t + _frac("0.636") * t - _frac("0.107")
    thsn = th33n + g - _frac("0.097") * S + _frac("0.043")
    ths = 1 - rho / _frac("2.65")
    return float(th33n - _frac("0.2") * (thsn - ths)), float(ths)


@pytest.mark.parametrize("S,Si,C,OM,rho", [(0.4, 0.4, 0.2, 2.5, 1400.0), (0.2, 0.5, 0.3, 3.0, 1300.0), (0.55, 0.3, 0.15, 1.0, 1500.0), (0.3, 0.3, 0.4, 4.0, 1200.0)])
def test_default_fc_is_the_final_density_adjusted_theta33(backend, S, Si, C, OM, rho):
    r = call33(S, Si, C, OM, rho, backend=backend)
    want33, want_sat = independent_theta33(S, C, OM, rho)
    assert r.field_capacity_method == tp.METHOD_SR33 and r.fc_output_name == FC
    assert abs(val(r, FC) - want33) <= 1e-12 and abs(val(r, "theta_saturation") - want_sat) <= 1e-12
    assert FC6 not in r.names() and set(r.names()) == {FC, "theta_saturation", "saturated_conductivity_m_s"}


def test_theta6_is_never_relabelled_theta33(backend):
    d, l = call33(backend=backend), call(backend=backend)
    assert val(d, FC) < val(l, FC6)  # the 33 kPa FC is strictly drier than the hybrid 6 kPa value for this soil
    with pytest.raises(tp.TexturePropertyError, match="no alias"):
        d.array(FC6)
    with pytest.raises(tp.TexturePropertyError):
        l.array(FC)
    assert d.sha256 != l.sha256 and d.summary()["recipe"] == l.summary()["recipe"] == tp.RECIPE_TAG
    assert tp.recipe_record(tp.METHOD_SR33) != tp.recipe_record(LEGACY)
    assert d.summary()["field_capacity_method"] == tp.METHOD_SR33 and l.summary()["field_capacity_method"] == LEGACY


def test_ks_saturation_and_other_outputs_do_not_depend_on_the_fc_method(backend):
    rng = np.random.default_rng(33)
    S, Si, C, OM, rho = mk(rng, 3, 17)
    d = call33(S, Si, C, OM, rho, L=3, N=17, backend=backend, diagnostics=True)
    l = call(S, Si, C, OM, rho, L=3, N=17, backend=backend, diagnostics=True)
    for k in ("saturated_conductivity_m_s", "theta_saturation", "theta_1500", "air_entry_kpa"):
        assert np.array_equal(d.array(k), l.array(k)), k  # Eq 16 already anchors on theta_33: selecting the FC rule must not change Ks
    assert np.array_equal(d.array(FC), (d.array(FC)))
    assert np.all(d.array(FC) <= l.array(FC6)) and np.all(d.array(FC) < d.array("theta_saturation"))


def test_final_theta33_override_is_the_default_fc_and_never_re_adjusted(backend):
    base = call33(backend=backend, diagnostics=True)
    ov = 0.5 * (val(base, FC) + val(base, "theta_1500"))  # a valid measured theta_33 below the estimate
    r = call33(backend=backend, theta_33_override=ov)
    assert val(r, FC) == ov
    l = call(backend=backend, theta_33_override=ov)
    assert val(l, FC6) >= ov


def test_jorn_layer1_default_is_the_root_audited_theta33_and_the_air_entry_guard_still_refuses(backend):
    """Root's accepted JORN audit (jorn_field_capacity_audit_20261120/root_audit.json, layer 1: PSD 86.7/6.2/7.1, OM 0.3448, rho 1419.6): final density-adjusted theta_33 = 0.09746408113338648 (80-digit
    arithmetic), the old hybrid 6 kPa FC = 0.39760410394534984. The raw negative air entry (-0.6079 kPa) STILL refuses without the explicit study override 0 (guard not widened by the FC change)."""
    kw = {"S": 0.867, "Si": 0.062, "C": 0.071, "OM": 0.3448, "rho": 1419.6, "backend": backend}
    with pytest.raises(tp.TexturePropertyError, match="air entry"):
        call33(**kw)
    with pytest.raises(tp.TexturePropertyError, match="air entry"):
        call(**kw)
    d = call33(air_entry_kpa_override=0.0, **kw)
    assert abs(val(d, FC) - 0.09746408113338648) <= 1e-14
    l = call(air_entry_kpa_override=0.0, **kw)
    assert abs(val(l, FC6) - 0.39760410394534984) <= 1e-14  # the retained historical numeric meaning of the 6 kPa name
    assert val(d, "saturated_conductivity_m_s") == val(l, "saturated_conductivity_m_s")


def test_numba_and_numpy_agree_bitwise_on_the_default_fc():
    pytest.importorskip("numba")
    rng = np.random.default_rng(7)
    S, Si, C, OM, rho = mk(rng, 4, 64)
    a = call33(S, Si, C, OM, rho, L=4, N=64, backend="numpy")
    b = call33(S, Si, C, OM, rho, L=4, N=64, backend="numba")
    assert np.array_equal(a.array(FC), b.array(FC)) and np.array_equal(a.array("theta_saturation"), b.array("theta_saturation"))
    assert a.sha256 == b.sha256


def test_legacy_method_warns_once_per_call_not_per_cell_and_default_never_warns():
    rng = np.random.default_rng(1)
    S, Si, C, OM, rho = mk(rng, 2, 50)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tp.estimate_soil_texture_properties(sand_fraction=S, silt_fraction=Si, clay_fraction=C, organic_matter_percent=OM, bulk_density_kg_m3=rho, n_layers=2, n_cells=50, memory_budget_bytes=BIG,
                                            field_capacity_method=LEGACY)
    dep = [x for x in w if issubclass(x.category, tp.FieldCapacityDeprecationWarning)]
    assert len(dep) == 1 and "DEPRECATED" in str(dep[0].message) and issubclass(tp.FieldCapacityDeprecationWarning, FutureWarning)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        call33(S, Si, C, OM, rho, L=2, N=50)
    assert not [x for x in w if issubclass(x.category, tp.FieldCapacityDeprecationWarning)]


@pytest.mark.parametrize("bad", ["", "6kpa", "SAXTON_RAWLS_33KPA", None, 33, b"saxton_rawls_33kpa", ("saxton_rawls_33kpa",)])
def test_method_is_validated_before_allocation_or_backend_probe(bad):
    with pytest.raises(tp.TexturePropertyError, match="field_capacity_method"):
        call33(field_capacity_method=bad)
    with pytest.raises(tp.TexturePropertyError, match="field_capacity_method"):  # even with an unusable backend the method refusal comes first
        call33(field_capacity_method=bad, backend="cuda")
    with pytest.raises(tp.TexturePropertyError, match="field_capacity_method"):
        tp.estimate_property_memory(1, 1, layouts={k: "scalar" for k in tp.INPUTS}, field_capacity_method=bad)


def test_memory_model_has_exactly_one_fc_array_in_either_method():
    lay = {k: "cell" for k in tp.INPUTS}
    a = tp.estimate_property_memory(3, 100, layouts=lay)
    b = tp.estimate_property_memory(3, 100, layouts=lay, field_capacity_method=LEGACY)
    assert a.output_bytes == b.output_bytes == 3 * 8 * 3 * 100 and a.main_required_bytes == b.main_required_bytes
    assert set(a.outputs) == {FC, "theta_saturation", "saturated_conductivity_m_s"} and set(b.outputs) == {FC6, "theta_saturation", "saturated_conductivity_m_s"}


def test_33_never_sets_fc_equals_saturation_provenance(backend):
    # a fine soil whose normal air entry is above 6 kPa: legacy reports fc == saturation, the default must not
    S, Si, C, OM, rho = 0.1, 0.5, 0.4, 2.0, 1300.0
    d = call33(S, Si, C, OM, rho, backend=backend, diagnostics=True)
    l = call(S, Si, C, OM, rho, backend=backend, diagnostics=True)
    assert d.n_fc_equals_saturation == 0 and int(d.array("provenance").reshape(-1)[0]) & tp.PROV_FC_EQ_SAT == 0
    assert l.n_fc_equals_saturation in (0, 1)
    assert l.n_fc_equals_saturation == (1 if val(l, "air_entry_kpa") >= 6.0 else 0)
    assert val(d, FC) < val(d, "theta_saturation")


def test_refusals_and_air_entry_guard_are_unchanged_by_the_method(backend):
    for method in tp.FIELD_CAPACITY_METHODS:
        with pytest.raises(tp.TexturePropertyError):  # clay above the supported domain
            call33(0.2, 0.1, 0.7, field_capacity_method=method, backend=backend)
        with pytest.raises(tp.TexturePropertyError):  # the published sand example: negative normal air entry (guard retained, never widened)
            call33(0.9, 0.05, 0.05, 1.0, 1500.0, field_capacity_method=method, backend=backend)


def test_machine_facing_recipe_metadata_cannot_misrepresent_the_default_pressure():
    import json

    d, l = tp.recipe_record(), tp.recipe_record(LEGACY)
    assert d == tp.recipe_record(tp.METHOD_SR33)
    assert d["field_capacity_method"] == tp.METHOD_SR33 and d["field_capacity_pressure_kpa"] == 33.0 and d["field_capacity_deprecated"] is False
    text = json.dumps(d, sort_keys=True)
    for key in ("fc_pressure_kpa", "fc_span_kpa", "legacy_6kpa", "tietjen"):
        assert key not in text, key  # no 6 kPa / 27 kPa marker anywhere in the default record
    assert not hasattr(tp, "FC_PRESSURE_KPA") and not hasattr(tp, "FC_SPAN_KPA")  # the unqualified 6/27 module constants no longer exist
    assert tp.LEGACY_6KPA_PRESSURE_KPA == 6.0 and tp.LEGACY_6KPA_SPAN_KPA == 27.0
    assert l["field_capacity_pressure_kpa"] == 6.0 and l["field_capacity_deprecated"] is True and l["legacy_6kpa"]["pressure_kpa"] == 6.0 and l["legacy_6kpa"]["span_kpa"] == 27.0
    assert "fc_pressure_kpa" not in l and "fc_span_kpa" not in l
    assert tp.FC_PRESSURE_BY_METHOD_KPA == {tp.METHOD_SR33: 33.0, LEGACY: 6.0}
