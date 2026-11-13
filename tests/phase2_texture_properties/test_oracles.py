"""Root's independent Decimal80 controls (portable copies under ./oracles): the normal-density Table 3 classes and the 30 density cases, on every CPU backend. Equation/numerical verification only."""
from __future__ import annotations

import pytest
from conftest import AT, RT, call, close, density_controls, normal_controls, val

from maple_syrup import soil_texture_properties as tp

FLAG_MAP = {"density_outside_1_to_1p8_g_cm3": tp.FLAG_DENSITY, "DF_outside_0p9_to_1p3": tp.FLAG_DENSITY_FACTOR, "density_gap_below_0p005": tp.FLAG_DENSITY_GAP,
            "invalid_retention_anchor_order": tp.FLAG_CURVE, "invalid_air_entry": tp.FLAG_AIR_ENTRY}


def args(c, rho_key):
    return dict(S=float(c["sand_fraction"]), Si=float(c["silt_fraction"]), C=float(c["clay_fraction"]), OM=float(c["organic_matter_percent"]), rho=float(c[rho_key]) * 1000.0)


@pytest.mark.parametrize("c", normal_controls(), ids=lambda c: c["label"])
def test_normal_density_classes(c, backend):
    """The five printed Table 3 classes at their explicit normal density. The published sand example has a NEGATIVE predicted air entry (about -0.964 kPa): refused, never clamped."""
    if not c["air_entry_nonnegative"]:
        with pytest.raises(tp.TexturePropertyError) as e:
            call(**args(c, "normal_density_g_cm3"), backend=backend)
        assert e.value.bit == tp.FLAG_AIR_ENTRY and float(c["air_entry_kpa"]) < 0
        return
    r = call(**args(c, "normal_density_g_cm3"), backend=backend, diagnostics=True)
    assert close(val(r, "theta_saturation"), float(c["theta_sat"]), RT, AT)
    assert close(val(r, "theta_field_capacity_6kpa"), float(c["theta_6_algebraic"]), RT, AT)
    assert close(val(r, "saturated_conductivity_m_s"), float(c["Ks_m_s"]), RT, 1e-18)
    assert close(val(r, "theta_1500"), float(c["theta_1500"]), RT, AT)
    assert close(val(r, "air_entry_kpa"), float(c["air_entry_kpa"]), RT, AT)
    pub = c["published_rounded"]  # printed Table 3 rounding: 1 vol %, 0.1 mm/h
    assert abs(val(r) * 100 - float(pub["sat_percent"])) <= 0.5
    assert abs(val(r, "saturated_conductivity_m_s") * 3.6e6 - float(pub["Ks_mm_h"])) <= 0.051


def test_density_controls_match_or_refuse(backend):
    """30 density cases (5 classes x 6 density factors): the 20 admissible full curves match Decimal80; the others are refused and every flag the provider reports is one the oracle lists. Exact DF boundaries
    (0.9 and 1.3) are checked on the oracle's own float density; a boundary that lands one ulp outside in FP64 is reported by the test, not hidden."""
    n_ok = n_ref = 0
    boundary_outside = []
    for c in density_controls():
        a = args(c, "bulk_density_g_cm3")
        expect = {FLAG_MAP[f] for f in c["expected_unsupported_flags"]}
        if c["mathematically_admissible_fullcurve"]:
            try:
                r = call(**a, backend=backend)
            except tp.TexturePropertyError as exc:
                assert exc.bit == tp.FLAG_DENSITY_FACTOR and c["density_factor"] in ("0.9", "1.3"), (c["label"], exc)  # FP64 boundary roundoff of the computed DF only
                boundary_outside.append(c["label"])
                continue
            v = c["values"]
            assert close(val(r), float(v["theta_sat"])) and close(val(r, "theta_field_capacity_6kpa"), float(v["theta_6_algebraic"])), c["label"]
            assert close(val(r, "saturated_conductivity_m_s"), float(v["Ks_m_s"]), RT, 1e-18), c["label"]
            n_ok += 1
        else:
            with pytest.raises(tp.TexturePropertyError) as e:
                call(**a, backend=backend)
            mine = {b for b in (1 << k for k in range(11)) if e.value.flags & b}
            assert mine and mine <= expect, (c["label"], mine, expect)
            n_ref += 1
    assert n_ok + len(boundary_outside) == 20 and n_ref == 10
    print("boundary roundoff refusals:", boundary_outside)
