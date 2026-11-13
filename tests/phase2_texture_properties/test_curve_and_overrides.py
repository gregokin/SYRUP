"""Air-entry/6 kPa branches, theta_33 (not theta_6) anchoring Ks, measured anchors and Ks overrides, provenance, against the author's Decimal80 model (and Root's controls elsewhere)."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import call, close, random_valid, val
from decimal_model import model

from maple_syrup import soil_texture_properties as tp

LOAM = dict(S=0.40, Si=0.40, C=0.20, OM=2.5)


def rho_loam():
    return 1000.0 / float(model(0.40, 0.40, 0.20, 2.5, 1000.0)[1]["df"])


def run(backend, **kw):
    a = dict(LOAM)
    a["rho"] = kw.pop("rho", rho_loam())
    return call(**a, backend=backend, diagnostics=True, **kw)


def test_estimated_curve_matches_decimal_on_random_elements(backend):
    rng = np.random.default_rng(1)
    S, Si, C, OM, rho = random_valid(rng, 60)
    for k in range(60):
        flags, v = model(S[k], Si[k], C[k], OM[k], rho[k])
        assert not flags
        r = call(S[k], Si[k], C[k], OM[k], rho[k], backend=backend, diagnostics=True)
        assert close(val(r), float(v["sat"])) and close(val(r, "theta_field_capacity_6kpa"), float(v["fc6"]))
        assert close(val(r, "saturated_conductivity_m_s"), float(v["ks"]), 2e-12, 1e-18)
        assert close(val(r, "theta_1500"), float(v["wp"])) and close(val(r, "air_entry_kpa"), float(v["air"]))
        assert int(r.array("provenance")[0]) & tp.PROV_FC_EQ_SAT == (tp.PROV_FC_EQ_SAT if v["eq"] else 0)


@pytest.mark.parametrize("air", [0.0, 1e-12, 5.999999, 6.0, 6.000001, 20.0, 32.999999])
def test_air_entry_branches_and_fc_vs_theta33(air, backend):
    """Exact 0 is a valid continuous curve; psi_e >= 6 gives theta_6 == theta_S (valid, flagged, Ks still > 0 because Eq 16 uses theta_33); below 6 the Eq 12 value."""
    r = run(backend, air_entry_kpa_override=air)
    flags, v = model(0.40, 0.40, 0.20, 2.5, rho_loam(), air_o=repr(air))
    assert not flags
    assert close(val(r, "theta_field_capacity_6kpa"), float(v["fc6"]))
    flag = bool(int(r.array("provenance")[0]) & tp.PROV_FC_EQ_SAT)
    assert flag == (air >= 6.0) == bool(v["eq"])
    if flag:
        assert val(r, "theta_field_capacity_6kpa") == val(r)  # exactly theta_S
    ks = val(r, "saturated_conductivity_m_s")
    assert ks > 0 and close(ks, float(v["ks"]), 2e-12, 1e-18)  # anchored on theta_33, so never 0 when theta_6 == theta_S
    assert r.n_fc_equals_saturation == (1 if air >= 6.0 else 0)
    assert int(r.array("provenance")[0]) & tp.PROV_AIR


@pytest.mark.parametrize("air", [-1e-9, -0.964, 33.0, 40.0, np.inf, -np.inf])
def test_unsupported_air_entry_is_refused_not_clamped(air, backend):
    with pytest.raises(tp.TexturePropertyError) as e:
        run(backend, air_entry_kpa_override=air)
    assert e.value.bit in (tp.FLAG_AIR_ENTRY, tp.FLAG_NONFINITE)


def test_sand_is_rescued_by_a_valid_measured_air_entry_but_not_by_fc6(backend):
    sand = dict(S=0.88, Si=0.07, C=0.05, OM=2.5)
    rho = 1000.0 / float(model(0.88, 0.07, 0.05, 2.5, 1000.0)[1]["df"])
    with pytest.raises(tp.TexturePropertyError) as e:
        call(**sand, rho=rho, backend=backend)
    assert e.value.bit == tp.FLAG_AIR_ENTRY
    r = call(**sand, rho=rho, backend=backend, air_entry_kpa_override=1.0)
    flags, v = model(0.88, 0.07, 0.05, 2.5, rho, air_o="1.0")
    assert not flags and close(val(r, "theta_field_capacity_6kpa"), float(v["fc6"]))
    with pytest.raises(TypeError):  # there is no measured-FC6 override: a bare field capacity claims no curve and belongs to the existing explicit path
        call(**sand, rho=rho, backend=backend, theta_field_capacity_override=0.2)


@pytest.mark.parametrize("which,value", [("theta_1500_override", 0.08), ("theta_33_override", 0.20), ("theta_saturation_override", 0.49), ("air_entry_kpa_override", 2.0)])
def test_single_anchor_override_replaces_final_value(which, value, backend):
    r = run(backend, **{which: value})
    key = {"theta_1500_override": "wp_o", "theta_33_override": "f33_o", "theta_saturation_override": "sat_o", "air_entry_kpa_override": "air_o"}[which]
    flags, v = model(0.40, 0.40, 0.20, 2.5, rho_loam(), **{key: repr(value)})
    assert not flags
    assert close(val(r), float(v["sat"])) and close(val(r, "theta_field_capacity_6kpa"), float(v["fc6"])) and close(val(r, "saturated_conductivity_m_s"), float(v["ks"]), 2e-12, 1e-18)
    out = {"theta_1500_override": "theta_1500", "air_entry_kpa_override": "air_entry_kpa", "theta_saturation_override": "theta_saturation"}.get(which)
    if out:
        assert val(r, out) == value  # a measured anchor is final: not re-adjusted by the density factor
    assert int(r.array("provenance")[0]) & {"theta_1500_override": tp.PROV_THETA_1500, "theta_33_override": tp.PROV_THETA_33, "theta_saturation_override": tp.PROV_THETA_SAT,
                                            "air_entry_kpa_override": tp.PROV_AIR}[which]


def test_measured_ks_replaces_eq16_only(backend):
    base, r = run(backend), run(backend, saturated_conductivity_override_m_s=3.2e-6)
    assert val(r, "saturated_conductivity_m_s") == 3.2e-6 and int(r.array("provenance")[0]) & tp.PROV_KS
    assert val(r) == val(base) and val(r, "theta_field_capacity_6kpa") == val(base, "theta_field_capacity_6kpa")  # the curve is untouched


@pytest.mark.parametrize("kw,bit", [
    (dict(theta_1500_override=0.0), tp.FLAG_CURVE), (dict(theta_1500_override=-0.01), tp.FLAG_CURVE), (dict(theta_33_override=0.05), tp.FLAG_CURVE),
    (dict(theta_33_override=0.60, theta_saturation_override=0.60), tp.FLAG_CURVE), (dict(theta_saturation_override=1.0), tp.FLAG_CURVE),
    (dict(theta_saturation_override=0.2), tp.FLAG_CURVE), (dict(theta_33_override=0.30, theta_saturation_override=0.303), tp.FLAG_DENSITY_GAP),
    (dict(saturated_conductivity_override_m_s=0.0), tp.FLAG_KS_OVERRIDE), (dict(saturated_conductivity_override_m_s=-1e-6), tp.FLAG_KS_OVERRIDE),
    (dict(theta_33_override=np.inf), tp.FLAG_NONFINITE), (dict(saturated_conductivity_override_m_s=np.inf), tp.FLAG_NONFINITE)])
def test_invalid_overrides_are_refused(kw, bit, backend):
    with pytest.raises(tp.TexturePropertyError) as e:
        run(backend, **kw)
    assert e.value.flags & bit


def test_nan_override_means_not_measured(backend):
    a = run(backend)
    b = run(backend, theta_1500_override=float("nan"), theta_33_override=np.full(1, np.nan), saturated_conductivity_override_m_s=float("nan"))
    for name in a.names():
        assert np.array_equal(a.array(name), b.array(name))
    assert a.sha256 != b.sha256  # the identity records what was supplied (an explicit NaN is not an absent argument)


def test_per_element_override_mixture(backend):
    """NaN marks the unmeasured elements of one override array; each element equals the single-element call."""
    ov = np.array([[np.nan, 2.0, np.nan, 10.0]])
    rho = rho_loam()
    r = call(0.40, 0.40, 0.20, 2.5, rho, L=1, N=4, backend=backend, air_entry_kpa_override=ov)
    for j in range(4):
        one = run(backend, **({} if np.isnan(ov[0, j]) else {"air_entry_kpa_override": float(ov[0, j])}))
        assert r.array("theta_saturation")[0, j] == val(one)
        assert r.array("theta_field_capacity_6kpa")[0, j] == val(one, "theta_field_capacity_6kpa")


def test_density_adjustment_and_ks_with_density_factor(backend):
    """Density enters through Eq 7-10 only: sat = 1 - rho/2.65, theta_33 shifted by 0.2 of the saturation change, air entry held at the normal-density value."""
    rn = rho_loam()
    lo, hi = run(backend, rho=0.95 * rn), run(backend, rho=1.15 * rn)
    assert val(lo) > val(hi)  # denser soil holds less water
    assert close(val(lo), 1 - 0.95 * rn / 1000 / 2.65) and close(val(hi), 1 - 1.15 * rn / 1000 / 2.65)
    assert val(lo, "air_entry_kpa") == val(hi, "air_entry_kpa") == val(run(backend), "air_entry_kpa")
    assert val(lo, "theta_1500") == val(hi, "theta_1500")
