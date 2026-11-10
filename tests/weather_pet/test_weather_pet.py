"""CPU tests of maple_syrup.weather_pet against independent oracles.

Oracles: an inline straight transcription of the downloaded EcoHyD constants (Weather.cpp:141-144,
WaterLandscape.cpp:805-807), the FAO-56 worked examples 8 (Ra) and 20 (Hargreaves) from
https://www.fao.org/4/x0490e/x0490e07.htm (which use slightly different astronomy constants and therefore agree
only to the documented ~1 %), and an explicit Python loop for the terrain factor including the source's < 5 deg
flat classification (WaterLandscape.cpp:1281-1282). Nothing here claims station calibration or coupled acceptance.

Naive datetimes are deliberate: the module's declared contract is the scheduler's own calendar (timezone-aware
instants are refused), so ruff's DTZ001 is silenced for this file."""
# ruff: noqa: DTZ001
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import math
import subprocess
import sys
import tracemalloc
from fractions import Fraction

import numpy as np
import pytest

from maple_syrup import weather_pet as wp

# ---------------------------------------------------------------------------------------------- oracles ---------


def ecohyd_ra_oracle(j: int, lat_deg: float) -> float:
    """Independent transcription of Weather.cpp:141-144 with J = day index + 1 and the 365 denominator; acos domain
    clamped analytically for the oracle as well."""
    phase = 2.0 * math.pi / 365.0 * j
    dr = 1.0 + 0.033 * math.cos(phase)
    decl = 0.4093 * math.sin(phase - 1.405)
    lat = lat_deg * math.pi / 180.0
    arg = -math.tan(lat) * math.tan(decl)
    ws = math.pi if arg <= -1 else (0.0 if arg >= 1 else math.acos(arg))
    return 15.392 * dr * (ws * math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.sin(ws))


def ecohyd_pet_oracle(tmean: float, tmin: float, tmax: float, ra: float) -> float:
    return 0.0023 * (tmean + 17.8) * (tmax - tmin) ** 0.5 * ra


def tetens(t: float) -> float:
    return 0.6108 * math.exp(17.27 * t / (t + 237.3))


# ------------------------------------------------------------------------------------- day of year / dates ------


@pytest.mark.parametrize(
    "date, expected",
    [
        (dt.date(2023, 1, 1), 1),
        (dt.date(2023, 3, 1), 60),
        (dt.date(2024, 2, 29), 60),  # leap day exists
        (dt.date(2024, 3, 1), 61),  # shifted by the leap day
        (dt.date(2023, 12, 31), 365),
        (dt.date(2024, 12, 31), 366),  # documented: phase one day past a full 365 cycle
    ],
)
def test_day_of_year_is_gregorian_with_leap_days(date, expected):
    assert wp.day_of_year(date) == expected


def test_invalid_dates_are_refused():
    with pytest.raises(ValueError):
        dt.date(2023, 2, 29)  # Python itself refuses a non-existent calendar day
    with pytest.raises(wp.WeatherValidationError):
        wp.day_of_year("2023-03-01")
    with pytest.raises(wp.WeatherValidationError):
        wp.day_of_year(dt.datetime(2023, 3, 1, 12))


# ------------------------------------------------------------------------------------------ latitude ------------


@pytest.mark.parametrize("bad", [None, True, False, float("nan"), float("inf"), 90.0001, -91, "35", np.bool_(True)])
def test_latitude_validation_refuses(bad):
    with pytest.raises(wp.WeatherValidationError):
        wp.validate_latitude_deg(bad)


@pytest.mark.parametrize("ok", [0, 35.5, -35.5, 90, -90, np.float64(12.0), np.int32(-7)])
def test_latitude_validation_accepts_signed_range(ok):
    assert wp.validate_latitude_deg(ok) == float(ok)


class _Spec:
    def __init__(self, latitude_deg):
        self.latitude_deg = latitude_deg


def test_maple_location_bridge_requires_authored_latitude_and_keeps_sign():
    assert wp.latitude_from_maple_location(_Spec(-23.5)) == -23.5
    assert wp.latitude_from_maple_location({"latitude_deg": 41.0}) == 41.0
    with pytest.raises(wp.WeatherValidationError, match="unspecified"):
        wp.latitude_from_maple_location(_Spec(None))
    with pytest.raises(wp.WeatherValidationError):
        wp.latitude_from_maple_location({})
    with pytest.raises(wp.WeatherValidationError):
        wp.latitude_from_maple_location(35.0)


# ------------------------------------------------------------------------------------------ astronomy -----------


@pytest.mark.parametrize("lat", [-60.0, -35.0, -20.0, 0.0, 12.5, 35.0, 48.0, 66.0])
@pytest.mark.parametrize("date", [dt.date(2023, 1, 1), dt.date(2023, 3, 21), dt.date(2023, 6, 21),
                                  dt.date(2023, 9, 3), dt.date(2024, 12, 31)])
def test_ra_equiv_matches_literal_ecohyd_transcription(lat, date):
    got = wp.extraterrestrial_radiation_equivalent_mm_day(date, lat)
    assert got == pytest.approx(ecohyd_ra_oracle(wp.day_of_year(date), lat), rel=1e-13, abs=0.0)
    assert got >= 0.0


def test_fao56_example_8_ra_agrees_to_the_documented_constant_difference():
    # FAO-56 Example 8: 20 deg S, 3 September (J = 246): Ra = 32.2 MJ m-2 d-1 = 13.1 mm d-1 with FAO constants.
    got = wp.extraterrestrial_radiation_equivalent_mm_day(dt.date(2023, 9, 3), -20.0)
    assert wp.day_of_year(dt.date(2023, 9, 3)) == 246
    assert got == pytest.approx(13.1, rel=0.01)  # ~13.13 with EcoHyD's 15.392 / 0.4093 / 1.405
    assert got != pytest.approx(13.1, rel=1e-4)  # it is NOT the FAO-56 value; the constants differ
    assert wp.RA_EQUIV_COEFFICIENT_MM_DAY == 15.392
    assert 0.408 * 37.586 == pytest.approx(15.335, abs=1e-3)  # the FAO-56 equivalent coefficient, documented


def test_equator_equinox_sunset_hour_is_quarter_turn_and_ra_near_15_4():
    g = wp.solar_geometry(dt.date(2023, 3, 21), 0.0)
    assert g.sunset_hour_angle_rad == pytest.approx(math.pi / 2, abs=1e-12)
    assert g.polar == "none"
    ra = wp.extraterrestrial_radiation_equivalent_mm_day(dt.date(2023, 3, 21), 0.0)
    assert ra == pytest.approx(15.392 * g.inverse_relative_distance * math.cos(g.declination_rad), rel=1e-13)
    assert 15.3 < ra < 15.6


def test_polar_night_gives_zero_and_polar_day_is_finite_positive():
    night = wp.solar_geometry(dt.date(2023, 12, 21), 80.0)
    assert night.polar == "night" and night.sunset_hour_angle_rad == 0.0
    assert wp.extraterrestrial_radiation_equivalent_mm_day(dt.date(2023, 12, 21), 80.0) == 0.0
    day = wp.solar_geometry(dt.date(2023, 6, 21), 80.0)
    assert day.polar == "day" and day.sunset_hour_angle_rad == math.pi
    ra = wp.extraterrestrial_radiation_equivalent_mm_day(dt.date(2023, 6, 21), 80.0)
    lat = math.radians(80.0)
    expected = 15.392 * day.inverse_relative_distance * math.pi * math.sin(lat) * math.sin(day.declination_rad)
    assert math.isfinite(ra) and ra > 0 and ra == pytest.approx(expected, rel=1e-13)
    # the poles themselves
    assert wp.extraterrestrial_radiation_equivalent_mm_day(dt.date(2023, 6, 21), 90.0) > 0
    assert wp.extraterrestrial_radiation_equivalent_mm_day(dt.date(2023, 12, 21), 90.0) == 0.0
    assert wp.extraterrestrial_radiation_equivalent_mm_day(dt.date(2023, 6, 21), -90.0) == 0.0
    assert wp.extraterrestrial_radiation_equivalent_mm_day(dt.date(2023, 12, 21), -90.0) > 0


def test_latitude_sign_is_honoured_not_absolute_valued():
    june, december = dt.date(2023, 6, 21), dt.date(2023, 12, 21)
    assert wp.extraterrestrial_radiation_equivalent_mm_day(june, -35.0) < wp.extraterrestrial_radiation_equivalent_mm_day(june, 35.0)
    assert wp.extraterrestrial_radiation_equivalent_mm_day(december, -35.0) > wp.extraterrestrial_radiation_equivalent_mm_day(december, 35.0)
    # southern summer ~ northern summer (same |lat|, half a year apart) within the dr asymmetry
    s = wp.extraterrestrial_radiation_equivalent_mm_day(december, -35.0)
    n = wp.extraterrestrial_radiation_equivalent_mm_day(june, 35.0)
    assert s == pytest.approx(n, rel=0.08) and s != n


# ------------------------------------------------------------------------------------------ Hargreaves ----------


def test_fao56_example_20_hargreaves_given_ra():
    # FAO-56 Example 20 (Lyon, July): Tmax 26.6, Tmin 14.8, Tmean 20.7, Ra = 40.6 MJ -> 16.6 mm/day; ET0 = 5.0 mm/day.
    pet, clamped = wp.hargreaves_pet_mm_day(14.8, 20.7, 26.6, 16.6)
    assert not clamped
    assert pet == pytest.approx(ecohyd_pet_oracle(20.7, 14.8, 26.6, 16.6), rel=1e-14)
    assert pet == pytest.approx(5.0, abs=0.06)


def test_hargreaves_zero_range_and_cold_clamp_and_refusals():
    assert wp.hargreaves_pet_mm_day(10.0, 10.0, 10.0, 12.0) == (0.0, False)
    pet, clamped = wp.hargreaves_pet_mm_day(-30.0, -25.0, -20.0, 5.0)
    assert pet == 0.0 and clamped is True
    pet, clamped = wp.hargreaves_pet_mm_day(-20.0, -17.8, -15.0, 5.0)
    assert pet == 0.0 and clamped is False  # exactly the offset: zero without clamping
    with pytest.raises(wp.WeatherValidationError):
        wp.hargreaves_pet_mm_day(20.0, 15.0, 25.0, 10.0)  # tmean < tmin
    with pytest.raises(wp.WeatherValidationError):
        wp.hargreaves_pet_mm_day(20.0, 22.0, 19.0, 10.0)  # tmax < tmin
    with pytest.raises(wp.WeatherValidationError):
        wp.hargreaves_pet_mm_day(10.0, 12.0, 14.0, -1.0)
    with pytest.raises(wp.WeatherValidationError):
        wp.hargreaves_pet_mm_day(10.0, float("nan"), 14.0, 1.0)


def test_potential_demand_ecohyd_hargreaves_end_to_end_matches_oracle():
    rec = wp.DailyWeather(dt.date(2023, 7, 15), 14.8, 20.7, 26.6, provenance="prescribed")
    d = wp.potential_demand(rec, 45.75)
    j = wp.day_of_year(rec.date)
    ra = ecohyd_ra_oracle(j, 45.75)
    assert d.method == "ecohyd_hargreaves"
    assert d.ra_equiv_mm_day == pytest.approx(ra, rel=1e-13)
    assert d.pet_mm_day == pytest.approx(ecohyd_pet_oracle(20.7, 14.8, 26.6, ra), rel=1e-13)
    assert d.clamped is False and d.terrain_applied is False and d.pet_mm_day_cells is None and d.terrain is None
    assert d.detached_cells() is None
    prov = d.provenance()
    assert prov["method"] == "ecohyd_hargreaves" and prov["date"] == "2023-07-15" and prov["latitude_deg"] == 45.75
    assert prov["day_of_year"] == j and prov["days_per_year_reference"] == 365
    assert prov["source"]["commit"].startswith("80b10e6b") and prov["terrain"] is None
    # the FULL weather record travels with the demand
    w = prov["weather"]
    assert (w["tmin_c"], w["tmean_c"], w["tmax_c"]) == (14.8, 20.7, 26.6)
    assert w["provenance"] == "prescribed" and w["pet_mm_day_explicit"] is None and w["humidity_sources"] == {}


def test_hargreaves_requires_daily_extrema():
    rec = wp.DailyWeather(dt.date(2023, 7, 15), None, 20.7, None)
    with pytest.raises(wp.WeatherValidationError, match="tmin_c and tmax_c"):
        wp.potential_demand(rec, 45.75)


def test_cold_day_demand_is_clamped_nonnegative():
    rec = wp.DailyWeather(dt.date(2023, 1, 10), -30.0, -25.0, -18.0)
    d = wp.potential_demand(rec, 60.0)
    assert d.pet_mm_day == 0.0 and d.clamped is True


def test_pet_unit_conversion():
    assert wp.pet_m_s(86.4) == pytest.approx(1e-6, rel=1e-15)
    arr = wp.pet_m_s(np.array([0.0, 8.64]))
    np.testing.assert_allclose(arr, [0.0, 1e-7], rtol=1e-15)


# ------------------------------------------------------------------------------------------ weather record ------


@pytest.mark.parametrize("tmin, tmean, tmax", [(20.0, 15.0, 25.0), (20.0, 22.0, 19.0), (float("nan"), 1.0, 2.0),
                                                (1.0, 2.0, float("inf")), (None, 2.0, 3.0), (1.0, 2.0, None),
                                                (None, float("nan"), None)])
def test_weather_temperature_validation(tmin, tmean, tmax):
    with pytest.raises(wp.WeatherValidationError):
        wp.DailyWeather(dt.date(2023, 1, 1), tmin, tmean, tmax)


def test_weather_refuses_bool_and_bad_provenance_and_negative_optionals():
    with pytest.raises(wp.WeatherValidationError):
        wp.DailyWeather(dt.date(2023, 1, 1), True, 1.0, 2.0)
    with pytest.raises(wp.WeatherValidationError):
        wp.DailyWeather(dt.date(2023, 1, 1), 0.0, 1.0, 2.0, provenance="measured")
    with pytest.raises(wp.WeatherValidationError):
        wp.DailyWeather(dt.date(2023, 1, 1), 0.0, 1.0, 2.0, radiation_mj_m2_day=-1.0)
    with pytest.raises(wp.WeatherValidationError):
        wp.DailyWeather(dt.date(2023, 1, 1), 0.0, 1.0, 2.0, pet_mm_day=-0.1)
    with pytest.raises(wp.WeatherValidationError):
        wp.DailyWeather("2023-01-01", 0.0, 1.0, 2.0)


def test_rh_derives_vpd_once_under_fao56_tetens():
    rec = wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0, relative_humidity_fraction=0.4)
    es = 0.5 * (tetens(30.0) + tetens(15.0))
    assert rec.saturation_vapor_pressure_mean_kpa == pytest.approx(es, rel=1e-14)
    assert rec.vapor_pressure_deficit_kpa == pytest.approx(es * 0.6, rel=1e-14)
    assert rec.rh_fraction == 0.4
    assert rec.humidity_sources == {"rh_source": "supplied", "vpd_source": "derived"}
    assert wp.saturation_vapor_pressure_kpa(20.0) == pytest.approx(2.338, abs=2e-3)  # FAO-56 Table 2.3 value 2.338
    r = rec.record()
    assert r["vpd_kpa_supplied"] is None and r["vapor_pressure_deficit_kpa"] == pytest.approx(es * 0.6, rel=1e-14)
    assert r["humidity_relation"].startswith("FAO-56 Eq. 11") and r["humidity_consistency_kpa"] is None


def test_vpd_derives_rh_and_both_supplied_must_agree():
    es = 0.5 * (tetens(30.0) + tetens(15.0))
    rec = wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0, vpd_kpa=0.5 * es)
    assert rec.rh_fraction == pytest.approx(0.5, rel=1e-13)
    assert rec.humidity_sources == {"rh_source": "derived", "vpd_source": "supplied"}
    ok = wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0, relative_humidity_fraction=0.5, vpd_kpa=0.5 * es)
    assert ok.humidity_sources == {"rh_source": "supplied", "vpd_source": "supplied"}
    assert ok.record()["humidity_consistency_kpa"] == 1e-6
    with pytest.raises(wp.WeatherValidationError, match="implies"):
        wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0, relative_humidity_fraction=0.5, vpd_kpa=0.4 * es)


def test_declared_humidity_tolerance_is_validated_and_honoured():
    es = 0.5 * (tetens(30.0) + tetens(15.0))
    # rounded station data: RH 0.50 vs VPD rounded to 2 decimals disagree beyond 1e-6 kPa ...
    vpd_rounded = round(0.5 * es, 2)
    with pytest.raises(wp.WeatherValidationError, match="tolerance 1e-06"):
        wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0, relative_humidity_fraction=0.5, vpd_kpa=vpd_rounded)
    # ... and pass under an explicitly declared tolerance, which is recorded
    rec = wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0, relative_humidity_fraction=0.5, vpd_kpa=vpd_rounded,
                          humidity_consistency_kpa=0.01)
    assert rec.record()["humidity_consistency_kpa"] == 0.01 and rec.vpd_kpa == vpd_rounded
    for bad in (0.0, -1e-3, 1.5, float("nan"), True):
        with pytest.raises(wp.WeatherValidationError):
            wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0, humidity_consistency_kpa=bad)


def test_humidity_range_refusals_and_extrema_requirement():
    with pytest.raises(wp.WeatherValidationError, match="fraction"):
        wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0, relative_humidity_fraction=55.0)  # percent
    with pytest.raises(wp.WeatherValidationError):
        wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0, vpd_kpa=-0.1)
    with pytest.raises(wp.WeatherValidationError, match="exceeds"):
        wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0, vpd_kpa=50.0)
    with pytest.raises(wp.WeatherValidationError, match="needs tmin_c and tmax_c"):
        wp.DailyWeather(dt.date(2023, 7, 1), None, 22.0, None, relative_humidity_fraction=0.5)  # no e0(tmean) substitute
    plain = wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0)
    assert plain.rh_fraction is None and plain.vapor_pressure_deficit_kpa is None and plain.humidity_sources == {}


def test_radiation_is_shared_but_not_used_by_hargreaves():
    a = wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0)
    b = wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0, radiation_mj_m2_day=25.0)
    assert wp.potential_demand(a, 40.0).pet_mm_day == wp.potential_demand(b, 40.0).pet_mm_day
    assert b.record()["radiation_mj_m2_day"] == 25.0


# ------------------------------------------------------------------------------------------ methods -------------


def test_explicit_pet_requires_declared_value_and_mean_temperature_only():
    rec = wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0, pet_mm_day=6.5, provenance="estimated")
    d = wp.potential_demand(rec, 40.0, method="explicit")
    assert d.pet_mm_day == 6.5 and d.ra_equiv_mm_day is None and d.method == "explicit" and d.clamped is False
    assert d.weather.tmean_c == 22.0  # temperature travels with the explicit PET
    # extrema and latitude are not needed for a standalone explicit PET
    lean = wp.DailyWeather(dt.date(2023, 7, 1), None, 22.0, None, pet_mm_day=6.5, provenance="estimated")
    d2 = wp.potential_demand(lean, None, method="explicit")
    assert d2.pet_mm_day == 6.5 and d2.latitude_deg is None and d2.provenance()["latitude_deg"] is None
    assert d2.provenance()["weather"]["tmin_c"] is None and d2.provenance()["weather"]["pet_mm_day_explicit"] == 6.5
    without = wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0)
    with pytest.raises(wp.WeatherValidationError, match="explicit"):
        wp.potential_demand(without, 40.0, method="explicit")
    with pytest.raises(TypeError):
        wp.DailyWeather(date=dt.date(2023, 7, 1), pet_mm_day=6.5)  # type: ignore[call-arg]  # no temperature-free record
    with pytest.raises(wp.WeatherValidationError):
        wp.potential_demand(rec, 91.0, method="explicit")  # a supplied latitude is still validated
    # the explicit value is NOT consumed by the computed method
    assert wp.potential_demand(rec, 40.0).pet_mm_day != 6.5


def test_explicit_pet_constructor_helper():
    rec = wp.DailyWeather.explicit_pet(dt.date(2023, 7, 1), 22.0, 6.5, provenance="estimated", radiation_mj_m2_day=20.0)
    assert rec == wp.DailyWeather(dt.date(2023, 7, 1), None, 22.0, None, pet_mm_day=6.5, provenance="estimated",
                                  radiation_mj_m2_day=20.0)
    assert not rec.has_extrema and wp.potential_demand(rec, None, method="explicit").pet_mm_day == 6.5
    with pytest.raises(wp.WeatherValidationError):
        wp.DailyWeather.explicit_pet(dt.date(2023, 7, 1), 22.0, None)
    with pytest.raises(wp.WeatherValidationError):
        wp.DailyWeather.explicit_pet(dt.date(2023, 7, 1), float("nan"), 6.5)


def test_unsupported_methods_refuse_without_fake_implementation():
    rec = wp.DailyWeather(dt.date(2023, 7, 1), 15.0, 22.0, 30.0)
    with pytest.raises(wp.UnsupportedPETMethodError, match="penman_monteith"):
        wp.potential_demand(rec, 40.0, method="penman_monteith")
    with pytest.raises(wp.UnsupportedPETMethodError, match="unknown"):
        wp.potential_demand(rec, 40.0, method="priestley_taylor")
    with pytest.raises(wp.WeatherValidationError):
        wp.potential_demand(rec, None)  # hargreaves needs a latitude; None is never the equator
    with pytest.raises(wp.WeatherValidationError):
        wp.potential_demand({"tmean": 1}, 40.0)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------ terrain -------------

NORTH = {"flat": 1.0, "N": 0.90, "NE": 0.95, "E": 0.98, "SE": 1.03, "S": 1.10, "SW": 1.05, "W": 1.02, "NW": 0.97}
SOUTH_EXPECTED = {"flat": 1.0, "N": 1.10, "NE": 1.03, "E": 0.98, "SE": 0.95, "S": 0.90, "SW": 0.97, "W": 1.02, "NW": 1.05}
STEEP = 0.2  # rad, 11.46 deg: above the source's 5 deg flat threshold, so aspect contrast is visible


def test_aspect_enum_order_matches_source_and_north_table_is_literal():
    assert wp.ASPECT_LABELS == ("flat", "N", "NE", "E", "SE", "S", "SW", "W", "NW")
    for label, code in wp.ASPECT_CODES.items():
        assert wp.aspect_factor_table(35.0)[code] == NORTH[label]
    assert wp.aspect_factor_table(0.0) == wp.ASPECT_FACTORS_NORTH  # equator convention: northern table


def test_southern_table_is_north_south_reflection_only():
    table = wp.aspect_factor_table(-35.0)
    for label, code in wp.ASPECT_CODES.items():
        assert table[code] == SOUTH_EXPECTED[label], label
    # E/W and flat untouched; N<->S, NE<->SE, NW<->SW swapped
    assert table[wp.ASPECT_CODES["E"]] == NORTH["E"] and table[wp.ASPECT_CODES["W"]] == NORTH["W"]
    assert table[wp.ASPECT_CODES["S"]] == NORTH["N"] and table[wp.ASPECT_CODES["N"]] == NORTH["S"]
    assert sorted(table) == sorted(wp.ASPECT_FACTORS_NORTH)  # a permutation, no new numbers
    assert wp.aspect_factor_table(-1e-9) == table  # any negative latitude reflects


def test_hemisphere_contrast_on_steep_slopes_through_terrain_factor():
    codes = wp.aspect_codes_from_labels(["N", "S", "E", "W", "NE", "SE", "SW", "NW", "flat"])
    slope = np.full(9, STEEP)
    north = wp.terrain_factor(codes, slope, 37.0).values
    south = wp.terrain_factor(codes, slope, -37.0).values
    c = math.cos(STEEP)
    np.testing.assert_allclose(north, np.array([0.90, 1.10, 0.98, 1.02, 0.95, 1.03, 1.05, 0.97, 1.0]) * c, rtol=1e-15)
    np.testing.assert_allclose(south, np.array([1.10, 0.90, 0.98, 1.02, 1.03, 0.95, 0.97, 1.05, 1.0]) * c, rtol=1e-15)


def test_aspect_label_parsing_and_code_validation():
    codes = wp.aspect_codes_from_labels(np.array([["flat", "n", "North-East"], ["e", "SE", "s"], ["sw", "W", "nw"]]))
    assert codes.dtype == np.int8 and codes.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert wp.aspect_codes_from_labels(np.array([0, 8], dtype=np.int64)).tolist() == [0, 8]
    with pytest.raises(wp.WeatherValidationError, match="unknown aspect label"):
        wp.aspect_codes_from_labels(["NNE"])
    with pytest.raises(wp.WeatherValidationError):
        wp.aspect_codes_from_labels(np.array([9]))
    with pytest.raises(wp.WeatherValidationError):
        wp.aspect_codes_from_labels(np.array([1.5]))


def terrain_oracle(codes: np.ndarray, slope: np.ndarray, lat: float, flat_threshold_deg: float | None = 5.0) -> np.ndarray:
    """Python loop: source flat classification (slope < threshold -> FL, WaterLandscape.cpp:1282) then table x cos."""
    out = np.empty(codes.shape, dtype=np.float64)
    for idx in np.ndindex(codes.shape):
        s = float(slope[idx])
        code = int(codes[idx])
        if flat_threshold_deg is not None and math.degrees(s) < flat_threshold_deg:
            code = 0
        label = wp.ASPECT_LABELS[code]
        factor = NORTH[label] if lat >= 0 else SOUTH_EXPECTED[label]
        out[idx] = factor * math.cos(s)
    return out


@pytest.mark.parametrize("lat", [37.0, 0.0, -37.0])
@pytest.mark.parametrize("threshold", [5.0, None, 2.0])
def test_terrain_factor_numpy_matches_loop_oracle_and_carries_identity(lat, threshold):
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 9, size=(6, 11)).astype(np.int8)
    slope = rng.uniform(0.0, 1.4, size=(6, 11))  # includes gentle (< 5 deg) and steep slopes
    codes_before, slope_before = codes.copy(), slope.copy()
    tf = wp.terrain_factor(codes, slope, lat, flat_threshold_deg=threshold)
    assert isinstance(tf, wp.TerrainFactor)
    got = tf.values
    assert type(got) is np.ndarray and got.dtype == np.float64 and got.shape == codes.shape
    np.testing.assert_allclose(got, terrain_oracle(codes, slope, lat, threshold), rtol=1e-15, atol=0.0)
    assert np.all(got > 0.0)
    # the caller's arrays are read, never modified
    assert codes.tobytes() == codes_before.tobytes() and slope.tobytes() == slope_before.tobytes()
    assert tf.shape == (6, 11) and tf.cell_count == 66 and tf.backend == "numpy" and tf.latitude_deg == lat
    assert tf.hemisphere_table == ("north_literal" if lat >= 0 else "south_reflected")
    assert tf.sha256 == hashlib.sha256(np.ascontiguousarray(got).tobytes()).hexdigest()
    assert tf.verify_identity()
    assert tf.max_value == float(got.max()) and 0.0 < tf.max_value <= 1.10
    ident = tf.identity()
    assert ident["cell_count"] == 66 and ident["shape"] == [6, 11] and ident["sha256"] == tf.sha256
    assert ident["max_value"] == tf.max_value
    if threshold is None:
        assert tf.flat_policy == "caller_classified" and tf.flat_threshold_deg is None
        assert tf.flat_cell_count == int(np.count_nonzero(codes == 0))
    else:
        assert tf.flat_policy == "source_flat_below_threshold" and tf.flat_threshold_deg == threshold
        assert tf.flat_cell_count == int(np.count_nonzero(np.degrees(slope) < threshold))
    assert ident["flat_policy"] == tf.flat_policy and ident["flat_threshold_source"].startswith("WaterLandscape.cpp:1281")


def test_source_flat_classification_default_and_caller_mode():
    one_deg = math.radians(1.0)
    codes = wp.aspect_codes_from_labels(["N", "S", "N", "S", "N"])
    slope = np.array([one_deg, 0.0, math.radians(4.999), math.radians(5.0), STEEP])
    default = wp.terrain_factor(codes, slope, 37.0)
    # < 5 deg: flat (factor cos(slope)); exactly 5 deg and steeper: the table applies
    assert default.values[0] == pytest.approx(math.cos(one_deg), rel=1e-15)  # 0.99985, not 0.90 * cos
    assert default.values[1] == 1.0  # a level "south" cell is flat: 1.0, not 1.10
    assert default.values[2] == pytest.approx(math.cos(math.radians(4.999)), rel=1e-15)
    assert default.values[3] == pytest.approx(1.10 * math.cos(math.radians(5.0)), rel=1e-15)
    assert default.values[4] == pytest.approx(0.90 * math.cos(STEEP), rel=1e-15)
    assert default.flat_cell_count == 3 and default.flat_threshold_deg == 5.0
    caller = wp.terrain_factor(codes, slope, 37.0, flat_threshold_deg=None)
    assert caller.values[0] == pytest.approx(0.90 * math.cos(one_deg), rel=1e-15)
    assert caller.values[1] == 1.10 and caller.flat_cell_count == 0
    assert default.sha256 != caller.sha256  # a different policy is a different identity
    for bad in (-1.0, 90.0, float("nan"), True):
        with pytest.raises(wp.WeatherValidationError):
            wp.terrain_factor(codes, slope, 37.0, flat_threshold_deg=bad)
    assert wp.terrain_factor(codes, slope, 37.0, flat_threshold_deg=0.0).values[1] == 1.10  # threshold 0: nothing flat


def test_terrain_factor_identity_for_flat_level_cells():
    codes = np.zeros(5, dtype=np.int8)
    slope = np.zeros(5, dtype=np.float64)
    assert wp.terrain_factor(codes, slope, 20.0).values.tobytes() == np.ones(5).tobytes()
    assert wp.terrain_factor(codes, slope, -20.0).values.tobytes() == np.ones(5).tobytes()


def test_terrain_factor_validation():
    codes = np.zeros(3, dtype=np.int8)
    ok = np.zeros(3, dtype=np.float64)
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(codes, np.array([0.0, -0.1, 0.0]), 10.0)
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(codes, np.array([0.0, math.pi / 2, 0.0]), 10.0)  # vertical refused
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(codes, np.array([0.0, float("nan"), 0.0]), 10.0)
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(np.array([0, 9, 0], dtype=np.int8), ok, 10.0)
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(codes, ok.astype(np.float32), 10.0)
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(codes, np.zeros(4), 10.0)
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor([0, 0, 0], ok, 10.0)  # list, not an owned ndarray
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(codes, ok, None)
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(codes, ok, 10.0, backend="numba")
    # nearly vertical is accepted
    assert wp.terrain_factor(codes, np.full(3, math.pi / 2 - 1e-9), 10.0).values.shape == (3,)
    # empty terrain is fine
    assert wp.terrain_factor(np.zeros(0, dtype=np.int8), np.zeros(0), 10.0).values.shape == (0,)


def test_terrain_factor_payload_is_sealed_and_read_only():
    codes = wp.aspect_codes_from_labels(["flat", "N", "S"])
    slope = np.array([0.0, STEEP, STEEP])
    tf = wp.terrain_factor(codes, slope, 45.0)
    expected = tf.values.copy()
    # 1. the public view cannot be written, not even with a POSITIVE factor change (Codex finding P1/P2)
    with pytest.raises(ValueError, match="read-only"):
        tf.values[...] *= 0.5
    with pytest.raises(ValueError, match="read-only"):
        tf.values[1] = 2.0
    v = tf.values
    with pytest.raises(ValueError):
        v.flags.writeable = True  # the owned base is read-only, so the view cannot be re-enabled
    assert tf.values.tobytes() == expected.tobytes() and tf.verify_identity()
    # 2. attribute assignment / deletion / direct construction / foreign payload are refused
    with pytest.raises(AttributeError):
        tf.sha256 = "0" * 64  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del tf.latitude_deg
    with pytest.raises(wp.WeatherValidationError):
        wp.TerrainFactor(object(), tf.values, 45.0, "north_literal", tf.sha256, "x", 5.0, 1, 1.0)  # type: ignore[arg-type]
    # 3. detached copies are independent of the owned payload
    copy = tf.detached_values()
    assert copy.flags.writeable and copy is not tf.values
    copy[:] = -10.0
    assert tf.values.tobytes() == expected.tobytes() and tf.verify_identity()
    # 4. the identity records exactly the multiplied payload
    d = wp.potential_demand(wp.DailyWeather(dt.date(2023, 7, 15), 14.8, 20.7, 26.6), 45.0)
    t = wp.apply_terrain(d, tf)
    np.testing.assert_allclose(t.pet_mm_day_cells, d.pet_mm_day * expected, rtol=1e-15)
    assert t.provenance()["terrain"]["sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()


def test_verify_identity_detects_a_stale_payload_through_the_private_handle():
    # Only module-private access can reach the storage; the declared cold check still catches it.
    tf = wp.terrain_factor(np.zeros(3, dtype=np.int8), np.zeros(3), 10.0)
    assert tf.verify_identity()
    storage = tf._payload._internal()
    storage.flags.writeable = True  # deliberate private tampering, outside the public contract
    storage[0] = 2.0
    assert not tf.verify_identity()


def test_apply_terrain_once_and_published_demand_is_read_only():
    rec = wp.DailyWeather(dt.date(2023, 7, 15), 14.8, 20.7, 26.6)
    d = wp.potential_demand(rec, 45.0)
    codes = np.array([0, 1, 5], dtype=np.int8)
    slope = np.array([0.0, STEEP, 0.3])
    tf = wp.terrain_factor(codes, slope, 45.0)
    t = wp.apply_terrain(d, tf)
    assert t.terrain_applied and t.pet_mm_day == d.pet_mm_day and t.terrain is tf
    np.testing.assert_allclose(t.pet_mm_day_cells, d.pet_mm_day * tf.values, rtol=1e-15)
    assert t.pet_mm_day_cells[0] == d.pet_mm_day  # flat level cell: identity
    assert np.all(np.isfinite(t.pet_mm_day_cells)) and np.all(t.pet_mm_day_cells >= 0.0)
    # the published per-cell demand cannot be driven negative / non-finite / rescaled (Codex finding)
    for bad in (-10.0, float("nan"), float("inf"), 0.5):
        with pytest.raises(ValueError, match="read-only"):
            t.pet_mm_day_cells[:] = bad
    with pytest.raises(ValueError, match="read-only"):
        t.pet_mm_day_cells[...] *= 2.0
    copy = t.detached_cells()
    copy[:] = -10.0
    assert np.all(t.pet_mm_day_cells >= 0.0)
    r = wp.integrate_interevent_pet_chunks([(t.date, 43200)], {t.date: t}, "interevent")
    np.testing.assert_allclose(r.depth_mm, 0.5 * d.pet_mm_day * tf.values, rtol=1e-15)
    # refusals: double application, raw arrays, latitude mismatch, foreign payload via dataclasses.replace
    with pytest.raises(wp.WeatherValidationError, match="already applied"):
        wp.apply_terrain(t, tf)
    with pytest.raises(wp.WeatherValidationError, match="TerrainFactor"):
        wp.apply_terrain(d, tf.values)
    with pytest.raises(wp.WeatherValidationError, match="differs from terrain latitude"):
        wp.apply_terrain(d, wp.terrain_factor(codes, slope, -45.0))
    # not a dataclass: replacement cannot swap in a foreign payload, drop the payload or flip the flag
    for change in ({"_cells": np.array([-1.0, -1.0, -1.0])}, {"terrain_applied": False}, {"terrain": None}):
        with pytest.raises(TypeError):
            dataclasses.replace(t, **change)
    with pytest.raises(TypeError):
        dataclasses.replace(d, terrain_applied=True)
    with pytest.raises(AttributeError):
        t.terrain_applied = False  # type: ignore[misc]
    prov = t.provenance()
    assert prov["terrain_applied"] is True and prov["terrain"]["sha256"] == tf.sha256
    assert prov["terrain"]["hemisphere_table"] == "north_literal" and prov["terrain"]["cell_count"] == 3
    assert prov["terrain"]["flat_policy"] == "source_flat_below_threshold" and prov["terrain"]["flat_cell_count"] == 1
    assert d.provenance()["terrain_applied"] is False and d.provenance()["terrain"] is None
    # an explicit-PET demand without latitude may take any site's terrain (latitude recorded by the terrain)
    lean = wp.potential_demand(wp.DailyWeather.explicit_pet(dt.date(2023, 7, 15), 20.0, 3.0), None, "explicit")
    assert wp.apply_terrain(lean, tf).provenance()["terrain"]["latitude_deg"] == 45.0


def test_scalar_demand_is_sealed_and_cannot_be_rebuilt_invalid():
    """Codex correction-02 reproducer: dataclasses.replace(x, pet_mm_day=-4.) used to succeed and integrate -4 mm."""
    x = wp.potential_demand(wp.DailyWeather.explicit_pet(dt.date(2026, 1, 1), 20.0, 4.0), None, "explicit")
    with pytest.raises(TypeError):
        dataclasses.replace(x, pet_mm_day=-4.0)
    other = wp.DailyWeather.explicit_pet(dt.date(2026, 1, 2), 20.0, 4.0)
    for change in ({"date": dt.date(2026, 1, 2)}, {"weather": other}, {"method": "explicit"}, {"latitude_deg": 0.0}):
        with pytest.raises(TypeError):
            dataclasses.replace(x, **change)
    assert not dataclasses.is_dataclass(x)
    for name, value in (("pet_mm_day", -4.0), ("date", dt.date(2026, 1, 2)), ("method", "explicit"), ("clamped", True)):
        with pytest.raises(AttributeError):
            setattr(x, name, value)
    with pytest.raises(AttributeError):
        del x.pet_mm_day
    # direct construction with the public name is refused (factory-only), with or without a valid-looking record
    with pytest.raises(wp.WeatherValidationError, match="constructed only"):
        wp.PotentialDemand(object(), date=x.date, latitude_deg=None, method="explicit", pet_mm_day=-4.0,  # type: ignore[arg-type]
                           ra_equiv_mm_day=None, clamped=False, weather=x.weather)
    with pytest.raises(TypeError):
        wp.PotentialDemand(x.date, None, "explicit", 4.0, None, False, x.weather)  # type: ignore[call-arg, misc]
    # the record itself re-validates on replacement, so a negative declared PET cannot enter through DailyWeather
    with pytest.raises(wp.WeatherValidationError):
        dataclasses.replace(x.weather, pet_mm_day=-4.0)
    assert x.pet_mm_day == 4.0 and x == wp.potential_demand(x.weather, None, "explicit")
    assert repr(x).startswith("PotentialDemand(date=2026-01-01, method='explicit', pet_mm_day=4.0")
    with pytest.raises(TypeError):
        hash(x)
    prov = x.provenance()
    prov["pet_mm_day"] = -4.0  # detached metadata: editing the snapshot changes nothing
    assert x.pet_mm_day == 4.0


def test_accumulation_and_computed_outputs_refuse_overflow_without_clipping():
    """Codex correction-02 reproducer: explicit PET 1e308 over two daily chunks returned inf."""
    huge = wp.potential_demand(wp.DailyWeather.explicit_pet(dt.date(2026, 1, 1), 20.0, 1e308), None, "explicit")
    assert huge.pet_mm_day == 1e308  # finite declared value is accepted as declared, not clipped
    with pytest.raises(wp.WeatherValidationError, match="overflowed"):
        wp.integrate_interevent_pet_chunks([(huge.date, 86400), (huge.date, 86400)], {huge.date: huge}, "interevent")
    assert wp.integrate_interevent_pet_chunks([(huge.date, 43200), (huge.date, 43200)], {huge.date: huge}, "interevent").depth_mm == 1e308
    # terrain application: pet x max factor must stay finite (O(1) guard using the build-time maximum)
    tf = wp.terrain_factor(np.array([0, 5], dtype=np.int8), np.array([0.0, STEEP]), 10.0)
    assert tf.max_value == pytest.approx(1.10 * math.cos(STEEP))
    too_big = wp.potential_demand(wp.DailyWeather.explicit_pet(dt.date(2026, 1, 1), 20.0, 1.7e308), None, "explicit")
    with pytest.raises(wp.WeatherValidationError, match="would overflow"):
        wp.apply_terrain(too_big, tf)
    ok = wp.apply_terrain(huge, tf)  # 1.1e308 is finite
    assert np.all(np.isfinite(ok.pet_mm_day_cells))
    # per-cell accumulation overflow is refused by the single end-of-integration check: choose a PET whose scalar
    # reference sum stays finite (2 x 0.85e308) while the south cell's sum (2 x 1.08e308) overflows
    mid = wp.apply_terrain(wp.potential_demand(wp.DailyWeather.explicit_pet(dt.date(2026, 1, 1), 20.0, 0.85e308), None, "explicit"), tf)
    with pytest.raises(wp.WeatherValidationError, match="per-cell PET overflowed"):
        wp.integrate_interevent_pet_chunks([(mid.date, 86400), (mid.date, 86400)], {mid.date: mid}, "interevent")
    half = wp.integrate_interevent_pet_chunks([(mid.date, 43200), (mid.date, 43200)], {mid.date: mid}, "interevent")
    assert np.all(np.isfinite(half.depth_mm)) and half.depth_mm[0] == 0.85e308
    # Hargreaves refuses a non-finite result rather than publishing inf
    hot = wp.DailyWeather(dt.date(2026, 7, 1), 1e308, 1e308, 1.7e308)
    with pytest.raises(wp.WeatherValidationError, match="overflowed"):
        wp.potential_demand(hot, 35.0)
    with pytest.raises(wp.WeatherValidationError, match="overflowed"):
        wp.hargreaves_pet_mm_day(0.0, 1e308, 1e308, 16.0)
    # the cold clamp remains the only clipping, and it is flagged
    assert wp.hargreaves_pet_mm_day(-30.0, -25.0, -20.0, 5.0) == (0.0, True)


# ------------------------------------------------------------------------------------------ intervals -----------


def _demand(day: int, pet: float) -> wp.PotentialDemand:
    rec = wp.DailyWeather(dt.date(2023, 7, day), 10.0, 15.0, 20.0, pet_mm_day=pet)
    return wp.potential_demand(rec, 30.0, method="explicit")


def test_pet_depth_is_uniform_daily_rate_within_one_date():
    d = _demand(1, 4.8)
    assert wp.pet_depth_mm(d, 43200) == pytest.approx(2.4)
    assert wp.pet_depth_mm(d, 86400) == pytest.approx(4.8)
    assert wp.pet_depth_mm(d, 3600) == pytest.approx(0.2)
    assert wp.pet_depth_mm(d, Fraction(86400, 3)) == pytest.approx(1.6)
    for bad in (0, -1, 86401, float("nan"), Fraction(0), Fraction(86401), True):
        with pytest.raises(wp.WeatherValidationError):
            wp.pet_depth_mm(d, bad)


def test_interevent_integration_splits_at_midnight_and_sums():
    records = {dt.date(2023, 7, 1): _demand(1, 4.8), dt.date(2023, 7, 2): _demand(2, 2.4), dt.date(2023, 7, 3): _demand(3, 9.6)}
    r = wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 18), dt.datetime(2023, 7, 3, 6), records, phase="interevent",
                                    keep_pieces=10)
    assert [(p.date.day, p.seconds) for p in r.pieces] == [(1, 21600.0), (2, 86400.0), (3, 21600.0)]
    assert not r.pieces_truncated and r.chunk_count == 3
    assert r.depth_mm == pytest.approx(4.8 * 0.25 + 2.4 + 9.6 * 0.25)
    assert r.reference_depth_mm == pytest.approx(r.depth_mm) and r.total_seconds == 129600.0
    assert r.terrain_applied is False and r.terrain_sha256_first is None and r.terrain_identity_changes == 0
    assert (r.first_date, r.last_date) == (dt.date(2023, 7, 1), dt.date(2023, 7, 3))
    assert [p.reference_depth_mm for p in r.pieces] == pytest.approx([1.2, 2.4, 2.4])
    assert not hasattr(r.pieces[0], "depth_mm")  # pieces keep scalar metadata only
    # default: no retained history at all
    r0 = wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 18), dt.datetime(2023, 7, 3, 6), records, "interevent")
    assert r0.pieces == () and not r0.pieces_truncated and r0.chunk_count == 3 and r0.depth_mm == pytest.approx(r.depth_mm)
    # callable lookup, single partial day with microseconds
    r2 = wp.integrate_interevent_pet(dt.datetime(2023, 7, 2, 1), dt.datetime(2023, 7, 2, 7, 0, 0, 500000), records.__getitem__, "interevent")
    assert r2.chunk_count == 1 and r2.total_seconds == 21600.5 and r2.depth_mm == pytest.approx(2.4 * 21600.5 / 86400)


def test_exact_chunks_api_accepts_fractions_and_repeated_dates():
    records = {dt.date(2023, 7, 1): _demand(1, 4.8), dt.date(2023, 7, 2): _demand(2, 2.4)}
    chunks = [(dt.date(2023, 7, 1), Fraction(86400, 7)), (dt.date(2023, 7, 1), Fraction(86400, 7)), (dt.date(2023, 7, 2), 3600)]
    r = wp.integrate_interevent_pet_chunks(chunks, records, "interevent", keep_pieces=2)
    assert r.chunk_count == 3 and len(r.pieces) == 2 and r.pieces_truncated
    assert r.depth_mm == pytest.approx(4.8 * 2 / 7 + 2.4 / 24, rel=1e-14)
    assert r.pieces[0].seconds == pytest.approx(86400 / 7)
    with pytest.raises(wp.WeatherValidationError, match="no chunks"):
        wp.integrate_interevent_pet_chunks([], records, "interevent")
    with pytest.raises(wp.WeatherValidationError, match="pair"):
        wp.integrate_interevent_pet_chunks([dt.date(2023, 7, 1)], records, "interevent")  # type: ignore[list-item]
    with pytest.raises(wp.WeatherValidationError):
        wp.integrate_interevent_pet_chunks([(dt.date(2023, 7, 1), 86401)], records, "interevent")
    with pytest.raises(wp.WeatherValidationError):
        wp.integrate_interevent_pet_chunks([("2023-07-01", 60)], records, "interevent")
    with pytest.raises(wp.WeatherValidationError):
        wp.integrate_interevent_pet_chunks([(dt.datetime(2023, 7, 1, 1), 60)], records, "interevent")  # instant, not a date
    for bad in (-1, 1.5, True):
        with pytest.raises(wp.WeatherValidationError, match="keep_pieces"):
            wp.integrate_interevent_pet_chunks(chunks, records, "interevent", keep_pieces=bad)
    with pytest.raises(wp.WeatherValidationError, match="on_chunk"):
        wp.integrate_interevent_pet_chunks(chunks, records, "interevent", on_chunk=5)  # type: ignore[arg-type]


@pytest.mark.parametrize("phase", ["wind", "rain", "recession"])
def test_non_interevent_phases_refuse_without_spending_demand(phase):
    calls = []

    def records(date):
        calls.append(date)
        return _demand(1, 4.8)

    with pytest.raises(wp.WeatherPhaseError):
        wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 0), dt.datetime(2023, 7, 1, 6), records, phase=phase)
    with pytest.raises(wp.WeatherPhaseError):
        wp.integrate_interevent_pet_chunks([(dt.date(2023, 7, 1), 60)], records, phase=phase)
    assert calls == []  # refused before any record was even looked up
    with pytest.raises(wp.WeatherValidationError):
        wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 0), dt.datetime(2023, 7, 1, 6), records, phase="storm")


def test_interval_record_consistency_refusals():
    records = {dt.date(2023, 7, 1): _demand(1, 4.8)}
    with pytest.raises(wp.WeatherValidationError, match="no weather/PET record"):
        wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 12), dt.datetime(2023, 7, 2, 12), records, "interevent")
    with pytest.raises(wp.WeatherValidationError, match="after start"):
        wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 12), dt.datetime(2023, 7, 1, 12), records, "interevent")
    with pytest.raises(wp.WeatherValidationError, match="naive"):
        wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 1, tzinfo=dt.UTC), dt.datetime(2023, 7, 1, 2), records, "interevent")
    with pytest.raises(wp.WeatherValidationError, match="carries date"):
        wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 1), dt.datetime(2023, 7, 1, 2), {dt.date(2023, 7, 1): _demand(2, 1.0)}, "interevent")
    tf = wp.terrain_factor(np.zeros(2, dtype=np.int8), np.zeros(2), 30.0)
    mixed = {dt.date(2023, 7, 1): wp.apply_terrain(_demand(1, 4.8), tf), dt.date(2023, 7, 2): _demand(2, 2.4)}
    with pytest.raises(wp.WeatherValidationError, match="terrain_applied"):
        wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 12), dt.datetime(2023, 7, 2, 12), mixed, "interevent")
    tf3 = wp.terrain_factor(np.zeros(3, dtype=np.int8), np.zeros(3), 30.0)
    reshaped = {dt.date(2023, 7, 1): wp.apply_terrain(_demand(1, 4.8), tf), dt.date(2023, 7, 2): wp.apply_terrain(_demand(2, 2.4), tf3)}
    with pytest.raises(wp.WeatherValidationError, match="no broadcasting"):
        wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 12), dt.datetime(2023, 7, 2, 12), reshaped, "interevent")


def test_terrain_applied_records_integrate_per_cell_with_bounded_memory():
    tf = wp.terrain_factor(np.array([0, 5], dtype=np.int8), np.array([0.0, STEEP]), 30.0)  # flat level, steep south
    south = 1.10 * math.cos(STEEP)
    records = {dt.date(2023, 7, d): wp.apply_terrain(_demand(d, pet), tf) for d, pet in ((1, 4.8), (2, 2.4), (3, 1.2), (4, 0.6))}
    before = {d: rec.pet_mm_day_cells.copy() for d, rec in records.items()}
    r = wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 12), dt.datetime(2023, 7, 4, 12), records, "interevent")
    expected_ref = 2.4 + 2.4 + 1.2 + 0.3
    np.testing.assert_allclose(r.depth_mm, np.array([expected_ref, south * expected_ref]), rtol=1e-14)
    assert r.reference_depth_mm == pytest.approx(expected_ref)
    assert r.terrain_applied and r.terrain_sha256_first == tf.sha256 and r.terrain_identity_changes == 0
    assert type(r.depth_mm) is np.ndarray and r.depth_mm.shape == (2,) and r.depth_mm.flags.writeable  # caller-owned
    assert r.pieces == () and r.chunk_count == 4
    for d, rec in records.items():
        assert rec.pet_mm_day_cells.tobytes() == before[d].tobytes()
    # a second terrain identity (same shape/backend) is counted, not refused
    tf_b = wp.terrain_factor(np.array([0, 1], dtype=np.int8), np.array([0.0, STEEP]), 30.0)
    records[dt.date(2023, 7, 4)] = wp.apply_terrain(_demand(4, 0.6), tf_b)
    r2 = wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 12), dt.datetime(2023, 7, 4, 12), records, "interevent")
    assert r2.terrain_sha256_first == tf.sha256 and r2.terrain_identity_changes == 1


def test_long_callable_stream_keeps_constant_resident_arrays_and_no_history():
    n_cells, n_chunks = 100_000, 400
    tf = wp.terrain_factor(np.zeros(n_cells, dtype=np.int8), np.zeros(n_cells), 30.0)
    base = wp.apply_terrain(wp.potential_demand(wp.DailyWeather.explicit_pet(dt.date(2023, 1, 1), 15.0, 2.4), None, "explicit"), tf)
    seen = []

    def records(date):
        # one record per date, built on the fly: a stream, not a stored history
        rec = wp.potential_demand(wp.DailyWeather.explicit_pet(date, 15.0, 2.4), None, "explicit")
        return wp.apply_terrain(rec, tf)

    def chunks():
        day = dt.date(2023, 1, 1)
        for _ in range(n_chunks):
            yield day, Fraction(43200)
            day += dt.timedelta(days=1)

    del base
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    r = wp.integrate_interevent_pet_chunks(chunks(), records, "interevent", on_chunk=lambda p: seen.append(p.seconds))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cell_bytes = n_cells * 8
    # resident: running sum + scratch + the per-date record's fresh cells + the result; nowhere near n_chunks arrays
    assert peak - baseline < 6 * cell_bytes + 2_000_000, (peak - baseline) / cell_bytes
    assert r.chunk_count == n_chunks and r.pieces == () and not r.pieces_truncated and len(seen) == n_chunks
    assert r.total_seconds == 43200.0 * n_chunks and r.reference_depth_mm == pytest.approx(1.2 * n_chunks)
    np.testing.assert_allclose(r.depth_mm, np.full(n_cells, 1.2 * n_chunks), rtol=1e-12)
    assert (r.first_date, r.last_date) == (dt.date(2023, 1, 1), dt.date(2023, 1, 1) + dt.timedelta(days=n_chunks - 1))
    # a capped history is bounded by the caller's quota
    rc = wp.integrate_interevent_pet_chunks(chunks(), records, "interevent", keep_pieces=5)
    assert len(rc.pieces) == 5 and rc.pieces_truncated and rc.pieces[-1].date == dt.date(2023, 1, 5)


# ------------------------------------------------------------------------------------------ import hygiene ------


def test_weather_replacement_owns_derived_humidity_and_preserves_demand_provenance():
    original = wp.DailyWeather(dt.date(2026, 1, 1), 10.0, 15.0, 20.0, relative_humidity_fraction=0.5)
    demand = wp.potential_demand(original, 35.0)
    before = original.record()
    demand_before = demand.provenance()
    replacement = dataclasses.replace(original, relative_humidity_fraction=0.2)
    assert original.record() == before
    assert demand.provenance() == demand_before
    assert replacement.vapor_pressure_deficit_kpa == pytest.approx(0.8 * replacement.saturation_vapor_pressure_mean_kpa)
    assert original.vapor_pressure_deficit_kpa == pytest.approx(0.5 * original.saturation_vapor_pressure_mean_kpa)
    no_humidity = wp.DailyWeather.explicit_pet(dt.date(2026, 1, 1), 15.0, 4.0)
    with_extrema = dataclasses.replace(no_humidity, tmin_c=10.0, tmax_c=20.0, relative_humidity_fraction=0.3)
    assert no_humidity.rh_fraction is None and no_humidity.vapor_pressure_deficit_kpa is None
    assert with_extrema.rh_fraction == 0.3
    with pytest.raises(TypeError):
        wp.DailyWeather(dt.date(2026, 1, 1), 10.0, 15.0, 20.0, _derived={"vpd_kpa": -5.0, "rh": 7.0})
    with pytest.raises(ValueError):
        dataclasses.replace(original, _derived={"vpd_kpa": -5.0})


def test_import_does_not_load_cupy_or_native_maple():
    code = (
        "import sys, maple_syrup.weather_pet as w; "
        "bad = sorted(m for m in sys.modules if m == 'cupy' or m.startswith('cupy.') or m == 'maple' or m.startswith('maple.')); "
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_cuda_backend_refuses_when_unavailable(gpu_unavailable_reason):
    if not gpu_unavailable_reason:
        pytest.skip("a GPU is available; the refusal path is exercised only without one")
    with pytest.raises(wp.WeatherBackendUnavailableError):
        wp.array_namespace("cuda")
    assert wp.array_namespace("numpy") is np
