"""Opt-in shared weather -> reference potential evapotranspiration (PET) library for SYRUP and LEAF (Phase 0B).

This module turns a DAILY weather record (date, temperatures, optional humidity/radiation, optional explicit PET)
plus one explicit site latitude into a REFERENCE / POTENTIAL evaporative demand. It is a physical weather library:

* not an event scheduler (it never decides which interval is wind, rain or interevent; it only refuses to
  integrate demand for a phase the caller did not declare as interevent),
* not a water authority (it debits nothing; actual soil evaporation, plant withdrawal and their partition stay
  with the SYRUP inventory and LEAF demand interfaces),
* not a metadata owner (the latitude is a narrow explicit input; MAPLE's `location.latitude_deg` stays the
  descriptive metadata it is, and the exact latitude/date/method/weather/terrain used are returned to the
  CONSUMER for its derived forcing provenance / restart record, see `PotentialDemand.provenance()`).

Importing this module imports only the standard library and NumPy. CuPy is imported lazily and only when a caller
explicitly selects `backend="cuda"`; a missing CuPy/device is `WeatherBackendUnavailableError`, never a silent
NumPy fallback. No MAPLE/native model, source-lock bundle or GPU context is touched to compute scalar PET.

Stable public call signatures (correction 01)
---------------------------------------------
    DailyWeather(date, tmin_c, tmean_c, tmax_c, *, relative_humidity_fraction=None, vpd_kpa=None,
                 radiation_mj_m2_day=None, pet_mm_day=None, provenance="prescribed",
                 humidity_consistency_kpa=1e-6)
    DailyWeather.explicit_pet(date, tmean_c, pet_mm_day, *, radiation_mj_m2_day=None, provenance="prescribed")
    potential_demand(weather, latitude_deg, method="ecohyd_hargreaves") -> PotentialDemand
    terrain_factor(aspect_codes, slope_rad, latitude_deg, backend="numpy", flat_threshold_deg=5.0) -> TerrainFactor
    apply_terrain(demand, terrain) -> PotentialDemand
    pet_depth_mm(demand, seconds) -> float | array
    integrate_interevent_pet_chunks(chunks, records, phase, *, keep_pieces=0, on_chunk=None) -> IntereventPET
    integrate_interevent_pet(start, end, records, phase, *, keep_pieces=0, on_chunk=None) -> IntereventPET
    PotentialDemand.provenance() -> dict;  TerrainFactor.identity() -> dict;  TerrainFactor.verify_identity() -> bool

Method `"ecohyd_hargreaves"` (the first computed method)
------------------------------------------------------------
Literal transcription of the downloaded Tietjen-lineage EcoHyD source (Baldauf, Cantón & Tietjen 2023 GitHub
snapshot `80b10e6b6b82ad16e9fd41a31901cc3157d880e9`; the two Dryad EcoHyD versions carry byte-identical constants):

    PET_mm_day = 0.0023 * (Tmean_C + 17.8) * sqrt(Tmax_C - Tmin_C) * Ra_equiv_mm_day
        (WaterLandscape.cpp:805-807, "according to Hargreaves (1974)")
    dr   = 1 + 0.033 cos(2 pi J / 365)                       (Weather.cpp:141)
    decl = 0.4093 sin(2 pi J / 365 - 1.405)                  (Weather.cpp:142)
    ws   = acos(-tan(lat) tan(decl))                         (Weather.cpp:143)
    Ra_equiv_mm_day = 15.392 dr (ws sin(lat) sin(decl) + cos(lat) cos(decl) sin(ws))   (Weather.cpp:144)

The source comment attributes the radiation routine to Maidment (ed.), Handbook of Hydrology, p. 4.31, i.e. the
evaporation-equivalent of extraterrestrial radiation directly in mm/day. These coefficients are PRESERVED
literally; they are NOT the FAO-56 constants. FAO-56 (Allen et al. 1998, https://www.fao.org/4/x0490e/x0490e07.htm)
writes Ra in MJ m-2 day-1 with (24*60/pi) Gsc = 37.586, dr with the same 0.033, decl = 0.409 sin(2 pi J/365 -
1.39) (Eq. 24), ws = arccos(-tan phi tan delta) (Eq. 25), and converts to mm/day with 0.408 (Eq. 20 context), so
the FAO-56 equivalent of 15.392 is 0.408 * 37.586 = 15.335 mm/day (about 0.4 % lower) and its declination
constants differ in the third digit. Hargreaves & Samani (1985) / FAO-56 Eq. 52 is the same 0.0023 (T + 17.8)
sqrt(Tmax - Tmin) Ra form with Ra expressed in mm/day. This module therefore labels the method
`ecohyd_hargreaves` and does not claim FAO-56 identity; the tests quantify the difference on FAO-56 Example 8.

Calendar: J is the Gregorian day of year of the EXPLICIT date supplied by the caller (`date.timetuple().tm_yday`,
leap days included, so 29 February exists and 31 December of a leap year is J = 366). The denominator stays the
reference's 365 (`DAYS_PER_YEAR_REFERENCE`). The downloaded source indexes a fixed 365-day year (`daysPerYear =
365`, `day % daysPerYear + 1`) and has no leap days at all; a dated scheduler cannot do that without drifting, so
the departure is: dated J with the reference's 365 denominator. On J = 366 the phase is 2 pi * 366/365, one day
past a full cycle; this is documented, not hidden.

Polar day / polar night: the acos argument -tan(lat) tan(decl) is handled analytically. Argument <= -1 means the
sun never sets (ws = pi); argument >= 1 means it never rises (ws = 0, Ra = 0). Latitude +-90 deg is accepted
(tan is finite in FP64 at 89.999..., and the clamp absorbs the overflow) and gives polar-day/night results.

Non-negativity: Hargreaves is empirical and goes negative for Tmean < -17.8 C; the physical reference demand is
clamped at zero (`max(0, .)`) and the clamp is reported (`PotentialDemand.clamped`). Tmax == Tmin gives exactly
zero. Tmax < Tmin, Tmean outside [Tmin, Tmax], non-finite or Boolean values are refused.

Temperatures: `tmean_c` is ALWAYS required (LEAF plant processes need temperature regardless of how PET was
obtained). `tmin_c`/`tmax_c` are required TOGETHER by `ecohyd_hargreaves` and by the humidity derivation (FAO-56
Eq. 12 uses the extrema; there is no silent e0(Tmean) substitution), and optional for `explicit`;
`DailyWeather.explicit_pet(date, tmean_c, pet_mm_day)` builds such a record without naming the extrema.

Explicit PET (`method="explicit"`): the record must carry `pet_mm_day` (declared mm/day, provenance declared) and
`tmean_c`; latitude is not needed and may be `None` (recorded as None, never as the equator). The value is returned
as REFERENCE demand only; nothing here splits it into soil evaporation plus plant transpiration, and the consumer
must not spend it twice (see `PotentialDemand` docstring). `method="penman_monteith"` is a declared hook that
refuses with `UnsupportedPETMethodError` (it needs wind speed, net radiation, pressure and resistances that this
record does not define); any other string refuses too. There is no fake implementation.

Humidity: `relative_humidity_fraction` (0..1) and `vpd_kpa` are OPTIONAL shared fields for LEAF. If exactly one is
given, the other is DERIVED once under FAO-56 Eq. 11/12/19: e0(T) = 0.6108 exp(17.27 T / (T + 237.3)) kPa
(Tetens), es = (e0(Tmax) + e0(Tmin)) / 2, ea = RH es, VPD = es - ea. If both are given they must agree within the
record's `humidity_consistency_kpa` (validated, default `HUMIDITY_CONSISTENCY_KPA` = 1e-6 kPa; a caller with
rounded station data may declare a larger tolerance explicitly); a conflict is refused, never silently kept as
two contradictory atmospheres. `radiation_mj_m2_day` is optional daily shortwave energy (MJ m-2 day-1) for LEAF;
Hargreaves does NOT use it, and nothing pretends it does. Every record carries a `provenance` label (`observed`,
`prescribed`, `estimated`); simplified scenarios are expected to declare `prescribed`/`estimated`, never `observed`.

Terrain: `aspect_factor_table(latitude_deg)` is the downloaded `WaterCell::setAspectFactor` table (WaterCell.cpp:
251-281, attributed there to Shevenell 1999): flat 1.0, N 0.90, NE 0.95, E 0.98, SE 1.03, S 1.10, SW 1.05, W 1.02,
NW 0.97, indexed by the source's `Aspects` enum (WaterLandscape.h:22-32, FL=0, NN=1, NE=2, EE=3, SE=4, SS=5, SW=6,
WW=7, NW=8). Non-negative latitude uses the literal northern table (the equator, latitude exactly 0, is the
northern convention by this declared rule). For latitude < 0 the USER EXPLICITLY CHOSE a north-south reflection:
N<->S, NE<->SE, NW<->SW while E, W and flat keep their factors (no east-west swap). Flat classification follows
the source by default: `WaterLandscape.cpp:1281-1282` sets the aspect to FL when the inclination is below 5 deg,
BEFORE the table is applied, while cos(inclination) still multiplies the actual fluxes (WaterCell.cpp:41-89). So
`terrain_factor(..., flat_threshold_deg=5.0)` (default `SOURCE_FLAT_THRESHOLD_DEG`) treats every cell with slope
< 5 deg as flat (factor cos(slope), aspect ignored) without altering the caller's code array; the policy is
recorded in the identity. `flat_threshold_deg=None` is the explicitly named caller-classified mode (codes used
verbatim). Slopes and aspect codes come from the caller (MAPLE owns the slope geometry; this module does not
derive aspect or inclination from a DEM). The per-cell multiplier is aspect_factor * cos(slope_rad) with
0 <= slope < pi/2 (non-vertical). In the reference the aspect factor multiplies potential ET once and
cos(inclination) multiplies each actual flux once; here BOTH are gathered into ONE `TerrainFactor` and
`apply_terrain` marks the demand `terrain_applied=True`, binds the terrain identity into its provenance and refuses
a second application. A consumer that receives a terrain-applied potential must not multiply by slope/aspect again
in the actual E/T computation; a consumer that applies terrain itself must take the unadjusted demand. Aspect
labels (`"N"`, `"SE"`, `"flat"`, ...) are declared compass directions of the DOWNSLOPE direction (the direction the
slope faces), the same convention the reference derives from its runoff direction with the upper map boundary
oriented north. Geographic north is assumed for the labels; the caller must map grid orientation to compass.

Sealed records (corrections 01-02): `TerrainFactor` and `PotentialDemand` are factory-only objects with read-only
fields, constructed only by `terrain_factor`, `potential_demand` and `apply_terrain`. Neither is a dataclass, so
`dataclasses.replace` cannot rebuild one with a negative PET, a different date/weather/method/latitude or a stale
payload; direct construction and attribute assignment refuse. Every published PET is finite and >= 0: Hargreaves
refuses a non-finite result (the cold clamp to 0 for Tmean < -17.8 C is the only clipping), `apply_terrain` refuses
an overflowing product using the O(1) check `pet x max_factor` (the maximum is recorded once at build), and the
integrators refuse a non-finite sum with ONE check at the end (addends are finite and >= 0, so a non-finite sum
is an overflow and persists). No physical formula is evaluated twice and no N-sized host hash occurs on ordinary
use. `TerrainFactor` and a terrain-applied `PotentialDemand` OWN their arrays; public accessors never hand out the
writable owned storage:

  * `TerrainFactor.values` / `PotentialDemand.pet_mm_day_cells`: on NumPy a zero-copy READ-ONLY view (the owned
    base itself is flagged read-only, so the view cannot be made writable); on CuPy, which has no read-only mode,
    a detached COPY (allocates N doubles on the device). Use them for inspection/cold checks, not per step.
  * `.detached_values()` / `.detached_cells()`: always a fresh copy (allocates), e.g. for a checkpoint writer.
  * Internal calculations (`apply_terrain`, integration) read the owned storage zero-copy; nothing is copied per
    date or per step, and no GPU->host transfer happens after the build.
  * `TerrainFactor.verify_identity()`: recomputes the SHA-256 of the owned payload (ONE N-sized host read on CuPy)
    and compares it with the recorded identity; an explicitly declared cold boundary (checkpoint, restart,
    acceptance), never called implicitly.

The identity recorded at build therefore covers exactly the payload every later `apply_terrain` reads.

Dated intervals: `pet_depth_mm(demand, seconds)` is the uniform-rate depth over `seconds` of ONE date's day
(0 < seconds <= 86400). `integrate_interevent_pet_chunks` takes the scheduler's EXACT declared `(date, seconds)`
chunks (int, float or `fractions.Fraction` seconds; the fraction of day is formed exactly for Fraction input and
rounded to FP64 once), looks up each date's demand, and accumulates with O(N) resident work: one running sum and
one scratch array, no per-date array history. Diagnostics are BOUNDED by default: scalar summaries only
(chunk count, total seconds, scalar reference depth, first/last date, first terrain identity and the number of
terrain identity changes). `keep_pieces=k` retains at most the first k `IntervalPiece` records (scalar metadata)
and reports truncation; `on_chunk=callable` receives each piece as it is integrated (the caller decides what to
keep). All terrain-applied records of one integration must share shape and backend (no broadcasting changing the
domain). `integrate_interevent_pet(start, end, ...)` is a convenience wrapper that splits a NAIVE datetime interval
at midnight into such chunks; offset-aware instants are refused, and an adapter with an offset-aware or
rational-subsecond calendar should map its own exact durations to chunks instead. Both refuse any phase that is
not `"interevent"` (wind, rain and recession spend no demand, and there is no catch-up). Hargreaves estimates PET
from daily extrema; there is no invented sub-daily diurnal curve, the fractional day uses the declared uniform
daily rate as the initial approximation.

Units: PET in mm/day (`PotentialDemand.pet_mm_day`), m/s via `pet_m_s` (x 1e-3 / 86400) for the soil column's
`potential_evaporation_m_s` forcing, depths in mm or m as named. Array results are float64 arrays of the selected
namespace; the scalar PET is a Python float.

Nothing here was run by its author against a GPU; Codex records executions. No calibration or validation against
weather-station data is claimed; a provenance dictionary is a metadata snapshot, not a native checkpoint format.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from types import ModuleType
from typing import Any

import numpy as np

__all__ = [
    "ASPECT_CODES",
    "ASPECT_FACTORS_NORTH",
    "ASPECT_LABELS",
    "ASPECT_REFLECTION_SOUTH",
    "BACKENDS",
    "DAYS_PER_YEAR_REFERENCE",
    "HARGREAVES_COEFFICIENT",
    "HARGREAVES_OFFSET_C",
    "HUMIDITY_CONSISTENCY_KPA",
    "INTEREVENT",
    "METHODS",
    "PHASES",
    "RA_EQUIV_COEFFICIENT_MM_DAY",
    "SECONDS_PER_DAY",
    "SOURCE_FLAT_THRESHOLD_DEG",
    "SOURCE_IDENTITY",
    "DailyWeather",
    "IntereventPET",
    "IntervalPiece",
    "PotentialDemand",
    "SolarGeometry",
    "TerrainFactor",
    "UnsupportedPETMethodError",
    "WeatherBackendUnavailableError",
    "WeatherError",
    "WeatherPhaseError",
    "WeatherValidationError",
    "apply_terrain",
    "array_namespace",
    "aspect_codes_from_labels",
    "aspect_factor_table",
    "day_of_year",
    "extraterrestrial_radiation_equivalent_mm_day",
    "hargreaves_pet_mm_day",
    "integrate_interevent_pet",
    "integrate_interevent_pet_chunks",
    "latitude_from_maple_location",
    "pet_depth_mm",
    "pet_m_s",
    "potential_demand",
    "saturation_vapor_pressure_kpa",
    "solar_geometry",
    "terrain_factor",
    "validate_latitude_deg",
]

# --------------------------------------------------------------------------------------------------------------
# Source identity and literal constants
# --------------------------------------------------------------------------------------------------------------

SOURCE_IDENTITY: Mapping[str, str] = {
    "model": "EcoHyD (Tietjen-lineage), Baldauf, Cantón & Tietjen 2023 GitHub snapshot",
    "repository": "https://github.com/selinaZitrone/Baldauf_et_al_Biocrust_Ecohydrology",
    "commit": "80b10e6b6b82ad16e9fd41a31901cc3157d880e9",
    "local_root": "reference_models/ecohyd/github_baldauf_2023/source/model/src/",
    "hargreaves": "WaterLandscape.cpp:799-812 (WaterLandscape::potEvapotranspiration)",
    "radiation": "Weather.cpp:137-144 (Weather::meanWeather)",
    "aspect_table": "WaterCell.cpp:240-281 (WaterCell::setAspectFactor)",
    "aspect_enum": "WaterLandscape.h:20-32 (enum Aspects)",
    "slope_use": "WaterCell.cpp:41-89 (cos(inclination) in actLayerEvapotranspiration)",
    "aspect_derivation": "WaterLandscape.cpp:1225-1298 (setAspectAndInclination)",
    "flat_threshold": "WaterLandscape.cpp:1281-1282 (inclination_deg < 5 -> FL, before the table is applied)",
    "dryad_same_constants": (
        "dryad_4j0zpc8gn .../Model/Source/{Weather.cpp:145-148, WaterLandscape.cpp:501, WaterCell.cpp:228-236}; "
        "dryad_ngf1vhj1t .../Source/EcoHyD/{Weather.cpp:141-144, WaterLandscape.cpp:501, WaterCell.cpp:228-236}"
    ),
    "paper": "Tietjen, Zehe & Jeltsch 2009, Water Resour. Res. 45, W01418, https://doi.org/10.1029/2007WR006589",
    "hargreaves_paper": "Hargreaves & Samani 1985, Applied Engineering in Agriculture 1(2), 96-99",
    "fao56": "Allen, Pereira, Raes & Smith 1998, FAO Irrigation and Drainage Paper 56, Eqs 11,12,19,21-25,52; "
    "https://www.fao.org/4/x0490e/x0490e07.htm",
    "aspect_paper": "Shevenell 1999, Hydrological Processes 13, 577-596",
    "radiation_handbook": "Maidment (ed.) 1993, Handbook of Hydrology, p. 4.31 (as cited by the source comment)",
}

HARGREAVES_COEFFICIENT = 0.0023  # WaterLandscape.cpp:805
HARGREAVES_OFFSET_C = 17.8  # WaterLandscape.cpp:805
RA_EQUIV_COEFFICIENT_MM_DAY = 15.392  # Weather.cpp:144; FAO-56 equivalent would be 0.408 * 37.586 = 15.335
_DR_AMPLITUDE = 0.033  # Weather.cpp:141 (same as FAO-56 Eq. 23)
_DECL_AMPLITUDE = 0.4093  # Weather.cpp:142 (FAO-56 Eq. 24 uses 0.409)
_DECL_PHASE = 1.405  # Weather.cpp:142 (FAO-56 Eq. 24 uses 1.39)
DAYS_PER_YEAR_REFERENCE = 365  # Parameters.h:19 `daysPerYear = 365`; kept as the denominator
SECONDS_PER_DAY = 86400
HUMIDITY_CONSISTENCY_KPA = 1e-6  # default tolerance when RH and VPD are both supplied
_HUMIDITY_CONSISTENCY_MAX_KPA = 1.0  # a declared tolerance above this would accept contradictory atmospheres
SOURCE_FLAT_THRESHOLD_DEG = 5.0  # WaterLandscape.cpp:1282 `if (inclination_deg[i][j] < 5) aspect = FL`
# Tetens / FAO-56 Eq. 11 constants
_TETENS_A_KPA = 0.6108
_TETENS_B = 17.27
_TETENS_C = 237.3

METHODS = ("ecohyd_hargreaves", "explicit", "penman_monteith")
_IMPLEMENTED_METHODS = ("ecohyd_hargreaves", "explicit")
BACKENDS = ("numpy", "cuda")
PHASES = ("interevent", "wind", "rain", "recession")
INTEREVENT = "interevent"
_PROVENANCE = ("observed", "prescribed", "estimated")
_FLAT_POLICY_SOURCE = "source_flat_below_threshold"
_FLAT_POLICY_CALLER = "caller_classified"

# Aspects enum of the source (WaterLandscape.h:22-32). Labels are compass directions of the downslope face.
ASPECT_LABELS: tuple[str, ...] = ("flat", "N", "NE", "E", "SE", "S", "SW", "W", "NW")
ASPECT_CODES: Mapping[str, int] = {label: code for code, label in enumerate(ASPECT_LABELS)}
#: Literal northern-hemisphere factors, WaterCell.cpp:253-279, indexed by code.
ASPECT_FACTORS_NORTH: tuple[float, ...] = (1.0, 0.90, 0.95, 0.98, 1.03, 1.10, 1.05, 1.02, 0.97)
#: User-selected southern reflection (north-south mirror): code -> code whose NORTHERN factor applies.
ASPECT_REFLECTION_SOUTH: Mapping[int, int] = {
    ASPECT_CODES["flat"]: ASPECT_CODES["flat"],
    ASPECT_CODES["N"]: ASPECT_CODES["S"],
    ASPECT_CODES["NE"]: ASPECT_CODES["SE"],
    ASPECT_CODES["E"]: ASPECT_CODES["E"],
    ASPECT_CODES["SE"]: ASPECT_CODES["NE"],
    ASPECT_CODES["S"]: ASPECT_CODES["N"],
    ASPECT_CODES["SW"]: ASPECT_CODES["NW"],
    ASPECT_CODES["W"]: ASPECT_CODES["W"],
    ASPECT_CODES["NW"]: ASPECT_CODES["SW"],
}
_LABEL_ALIASES: Mapping[str, str] = {
    "flat": "flat", "fl": "flat", "f": "flat", "0": "flat",
    "n": "N", "nn": "N", "north": "N",
    "ne": "NE", "northeast": "NE", "north-east": "NE",
    "e": "E", "ee": "E", "east": "E",
    "se": "SE", "southeast": "SE", "south-east": "SE",
    "s": "S", "ss": "S", "south": "S",
    "sw": "SW", "southwest": "SW", "south-west": "SW",
    "w": "W", "ww": "W", "west": "W",
    "nw": "NW", "northwest": "NW", "north-west": "NW",
}

#: Module-private construction seal: only this module can build owned payloads.
_SEAL = object()


# --------------------------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------------------------


class WeatherError(ValueError):
    """Base class of every refusal in this module."""


class WeatherValidationError(WeatherError):
    """An input is of the wrong kind, non-finite, out of range or internally inconsistent."""


class UnsupportedPETMethodError(WeatherError):
    """The declared PET method is a hook without an implementation, or unknown."""


class WeatherBackendUnavailableError(WeatherError):
    """The explicitly selected array backend cannot be used (no CuPy, no device). Never a fallback."""


class WeatherPhaseError(WeatherError):
    """Demand was requested for a phase that is not an interevent interval (no E/T during wind/rain/recession)."""


# --------------------------------------------------------------------------------------------------------------
# Scalar validation helpers
# --------------------------------------------------------------------------------------------------------------


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _finite_float(value: Any, name: str) -> float:
    if not _is_real(value):
        raise WeatherValidationError(f"{name} must be a non-Boolean real number, got {value!r}")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise WeatherValidationError(f"{name} is too large to represent as a finite float") from exc
    if not math.isfinite(numeric):
        raise WeatherValidationError(f"{name} must be finite, got {value!r}")
    return numeric


def validate_latitude_deg(latitude_deg: Any) -> float:
    """A finite, signed, non-Boolean latitude in [-90, 90] degrees (north positive). `None` is refused: a missing
    latitude is never the equator."""
    if latitude_deg is None:
        raise WeatherValidationError(
            "latitude_deg is required and has no default (an absent latitude is not the equator); pass the resolved "
            "MAPLE location.latitude_deg explicitly"
        )
    numeric = _finite_float(latitude_deg, "latitude_deg")
    if numeric < -90.0 or numeric > 90.0:
        raise WeatherValidationError(f"latitude_deg must lie in [-90, 90], got {numeric!r}")
    return numeric


def latitude_from_maple_location(location: Any) -> float:
    """Narrow bridge to MAPLE's resolved `CaseLocationSpec` (duck-typed: any object with a `latitude_deg`
    attribute, or a mapping with that key). Refuses an absent/None latitude. Does not import MAPLE, does not change
    the metadata's descriptive semantics; the consumer records the value it used in its own derived provenance."""
    if isinstance(location, Mapping):
        if "latitude_deg" not in location:
            raise WeatherValidationError("MAPLE location mapping lacks latitude_deg (omitted section: unspecified)")
        value = location["latitude_deg"]
    elif hasattr(location, "latitude_deg"):
        value = location.latitude_deg
    else:
        raise WeatherValidationError(
            f"expected a resolved MAPLE location (object/mapping with latitude_deg), got {type(location).__name__}"
        )
    if value is None:
        raise WeatherValidationError(
            "MAPLE location.latitude_deg is unspecified (section omitted); PET needs an authored latitude, no equator "
            "default is applied"
        )
    return validate_latitude_deg(value)


def _validate_date(date: Any) -> _dt.date:
    if isinstance(date, _dt.datetime):
        raise WeatherValidationError("date must be a datetime.date (a calendar day), not a datetime instant")
    if not isinstance(date, _dt.date):
        raise WeatherValidationError(f"date must be a datetime.date, got {type(date).__name__}")
    return date


def day_of_year(date: _dt.date) -> int:
    """Gregorian day of year J of an explicit date (1 January = 1; 29 February exists; 31 December of a leap year is
    366). The reference indexes a fixed 365-day year without leap days; see the module docstring."""
    return _validate_date(date).timetuple().tm_yday


# --------------------------------------------------------------------------------------------------------------
# Astronomy (scalar, per date and latitude)
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SolarGeometry:
    """Scalar astronomy for one date and latitude (radians except where named)."""

    day_of_year: int
    latitude_deg: float
    inverse_relative_distance: float  # dr
    declination_rad: float
    sunset_hour_angle_rad: float
    polar: str  # "none", "day" or "night"


def solar_geometry(date: _dt.date, latitude_deg: float) -> SolarGeometry:
    """dr, declination and sunset hour angle with the reference's constants and the dated J. The acos domain is
    handled analytically: argument <= -1 -> polar day (ws = pi); argument >= 1 -> polar night (ws = 0)."""
    lat = validate_latitude_deg(latitude_deg)
    j = day_of_year(date)
    phase = 2.0 * math.pi * j / DAYS_PER_YEAR_REFERENCE
    dr = 1.0 + _DR_AMPLITUDE * math.cos(phase)
    decl = _DECL_AMPLITUDE * math.sin(phase - _DECL_PHASE)
    lat_rad = math.radians(lat)
    argument = -math.tan(lat_rad) * math.tan(decl)
    if argument <= -1.0:
        ws, polar = math.pi, "day"
    elif argument >= 1.0:
        ws, polar = 0.0, "night"
    else:
        ws, polar = math.acos(argument), "none"
    return SolarGeometry(j, lat, dr, decl, ws, polar)


def extraterrestrial_radiation_equivalent_mm_day(date: _dt.date, latitude_deg: float) -> float:
    """Ra_equiv in mm/day, Weather.cpp:144 with the reference's 15.392 (NOT the FAO-56 0.408 * 37.586 = 15.335).
    Non-negative by construction; a roundoff-negative value of a polar-night date is clamped to exactly 0."""
    g = solar_geometry(date, latitude_deg)
    lat_rad = math.radians(g.latitude_deg)
    ra = RA_EQUIV_COEFFICIENT_MM_DAY * g.inverse_relative_distance * (
        g.sunset_hour_angle_rad * math.sin(lat_rad) * math.sin(g.declination_rad)
        + math.cos(lat_rad) * math.cos(g.declination_rad) * math.sin(g.sunset_hour_angle_rad)
    )
    if g.polar == "night":
        return 0.0
    return max(0.0, ra)


# --------------------------------------------------------------------------------------------------------------
# Weather record
# --------------------------------------------------------------------------------------------------------------


def saturation_vapor_pressure_kpa(temperature_c: float) -> float:
    """FAO-56 Eq. 11 (Tetens): e0(T) = 0.6108 exp(17.27 T / (T + 237.3)) kPa."""
    t = _finite_float(temperature_c, "temperature_c")
    if t <= -_TETENS_C:
        raise WeatherValidationError(f"temperature_c must exceed {-_TETENS_C} C for the Tetens relation, got {t}")
    return _TETENS_A_KPA * math.exp(_TETENS_B * t / (t + _TETENS_C))


@dataclass(frozen=True)
class DailyWeather:
    """One calendar day of shared weather. Required: `date` and `tmean_c` (finite, non-Boolean). `tmin_c` and
    `tmax_c` are optional but must be given TOGETHER with tmin <= tmean <= tmax; they are required by the
    `ecohyd_hargreaves` method and by any humidity derivation (`DailyWeather.explicit_pet` builds an extrema-free
    record). Optional shared fields: `relative_humidity_fraction` (0..1, NOT percent), `vpd_kpa` (>= 0),
    `radiation_mj_m2_day` (daily shortwave energy, >= 0; not used by Hargreaves), `pet_mm_day` (explicit declared
    reference PET, >= 0). `provenance` is one of observed / prescribed / estimated and labels the WHOLE record; a
    simplified scenario must not call itself observed. Exactly one humidity field given -> the other is derived
    (FAO-56 Eq. 11/12/19) and exposed by the properties; both given -> consistency within
    `humidity_consistency_kpa` (validated, default 1e-6 kPa) is checked."""

    date: _dt.date
    tmin_c: float | None
    tmean_c: float
    tmax_c: float | None
    relative_humidity_fraction: float | None = None
    vpd_kpa: float | None = None
    radiation_mj_m2_day: float | None = None
    pet_mm_day: float | None = None
    provenance: str = "prescribed"
    humidity_consistency_kpa: float = HUMIDITY_CONSISTENCY_KPA
    _derived: dict[str, float] = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def explicit_pet(cls, date: _dt.date, tmean_c: float, pet_mm_day: float, *, radiation_mj_m2_day: float | None = None,
                     provenance: str = "prescribed") -> DailyWeather:
        """A record for `method="explicit"`: declared PET plus the mean temperature LEAF needs, no extrema, no
        humidity (humidity derivation needs the extrema; build the full record for that)."""
        if pet_mm_day is None:
            raise WeatherValidationError("explicit_pet requires pet_mm_day (declared mm/day)")
        return cls(date, None, tmean_c, None, radiation_mj_m2_day=radiation_mj_m2_day, pet_mm_day=pet_mm_day,
                   provenance=provenance)

    def __post_init__(self) -> None:
        _validate_date(self.date)
        tmean = _finite_float(self.tmean_c, "tmean_c")
        object.__setattr__(self, "tmean_c", tmean)
        if (self.tmin_c is None) != (self.tmax_c is None):
            raise WeatherValidationError("tmin_c and tmax_c must be given together (both) or omitted together (both None)")
        if self.tmin_c is not None:
            tmin = _finite_float(self.tmin_c, "tmin_c")
            tmax = _finite_float(self.tmax_c, "tmax_c")
            if not (tmin <= tmean <= tmax):
                raise WeatherValidationError(
                    f"temperatures must satisfy tmin <= tmean <= tmax, got tmin={tmin}, tmean={tmean}, tmax={tmax}"
                )
            object.__setattr__(self, "tmin_c", tmin)
            object.__setattr__(self, "tmax_c", tmax)
        if self.provenance not in _PROVENANCE:
            raise WeatherValidationError(f"provenance must be one of {_PROVENANCE}, got {self.provenance!r}")
        for name in ("radiation_mj_m2_day", "pet_mm_day"):
            value = getattr(self, name)
            if value is not None:
                numeric = _finite_float(value, name)
                if numeric < 0.0:
                    raise WeatherValidationError(f"{name} must be >= 0, got {numeric}")
                object.__setattr__(self, name, numeric)
        tol = _finite_float(self.humidity_consistency_kpa, "humidity_consistency_kpa")
        if not (0.0 < tol <= _HUMIDITY_CONSISTENCY_MAX_KPA):
            raise WeatherValidationError(
                f"humidity_consistency_kpa must satisfy 0 < tol <= {_HUMIDITY_CONSISTENCY_MAX_KPA} kPa, got {tol}"
            )
        object.__setattr__(self, "humidity_consistency_kpa", tol)
        self._resolve_humidity()

    @property
    def has_extrema(self) -> bool:
        return self.tmin_c is not None

    def _resolve_humidity(self) -> None:
        rh, vpd = self.relative_humidity_fraction, self.vpd_kpa
        if rh is None and vpd is None:
            return
        if not self.has_extrema:
            raise WeatherValidationError(
                "humidity derivation (FAO-56 Eq. 12 mean saturation vapour pressure) needs tmin_c and tmax_c; no "
                "e0(tmean) substitution is made silently"
            )
        es = 0.5 * (saturation_vapor_pressure_kpa(self.tmax_c) + saturation_vapor_pressure_kpa(self.tmin_c))
        if rh is not None:
            rh = _finite_float(rh, "relative_humidity_fraction")
            if not (0.0 <= rh <= 1.0):
                raise WeatherValidationError(
                    f"relative_humidity_fraction must lie in [0, 1] (a fraction, not percent), got {rh}"
                )
            object.__setattr__(self, "relative_humidity_fraction", rh)
        if vpd is not None:
            vpd = _finite_float(vpd, "vpd_kpa")
            if vpd < 0.0:
                raise WeatherValidationError(f"vpd_kpa must be >= 0, got {vpd}")
            if vpd > es * (1.0 + 1e-12):
                raise WeatherValidationError(
                    f"vpd_kpa={vpd} exceeds the daily mean saturation vapour pressure es={es:.6f} kPa derived from "
                    "tmin/tmax (FAO-56 Eq. 11/12)"
                )
            object.__setattr__(self, "vpd_kpa", vpd)
        if rh is not None and vpd is not None:
            implied = es * (1.0 - rh)
            if abs(implied - vpd) > self.humidity_consistency_kpa:
                raise WeatherValidationError(
                    f"relative_humidity_fraction={rh} implies vpd={implied:.6f} kPa under FAO-56 Eq. 11/12/19 but "
                    f"vpd_kpa={vpd} was supplied (tolerance {self.humidity_consistency_kpa} kPa); supply one or make "
                    "them consistent"
                )
            self._derived.update({"es_kpa": es, "rh_source": "supplied", "vpd_source": "supplied"})
        elif rh is not None:
            self._derived.update({"es_kpa": es, "vpd_kpa": es * (1.0 - rh), "rh_source": "supplied",
                                  "vpd_source": "derived"})
        else:
            self._derived.update({"es_kpa": es, "rh": 1.0 - vpd / es, "rh_source": "derived",
                                  "vpd_source": "supplied"})

    @property
    def saturation_vapor_pressure_mean_kpa(self) -> float | None:
        """es = (e0(Tmax) + e0(Tmin)) / 2 (FAO-56 Eq. 12), present when any humidity field was given."""
        return self._derived.get("es_kpa")

    @property
    def rh_fraction(self) -> float | None:
        """Consistent RH fraction: supplied, or derived from VPD."""
        if self.relative_humidity_fraction is not None:
            return self.relative_humidity_fraction
        return self._derived.get("rh")

    @property
    def vapor_pressure_deficit_kpa(self) -> float | None:
        """Consistent VPD (kPa): supplied, or derived from RH."""
        if self.vpd_kpa is not None:
            return self.vpd_kpa
        return self._derived.get("vpd_kpa")

    @property
    def humidity_sources(self) -> Mapping[str, str]:
        """Which humidity quantity was supplied and which derived (empty when neither was given)."""
        return {k: v for k, v in self._derived.items() if k.endswith("_source")}

    def record(self) -> dict[str, Any]:
        """Every declared and derived value of this record, for the consumer's provenance."""
        return {
            "date": self.date.isoformat(),
            "tmin_c": self.tmin_c,
            "tmean_c": self.tmean_c,
            "tmax_c": self.tmax_c,
            "relative_humidity_fraction_supplied": self.relative_humidity_fraction,
            "vpd_kpa_supplied": self.vpd_kpa,
            "rh_fraction": self.rh_fraction,
            "vapor_pressure_deficit_kpa": self.vapor_pressure_deficit_kpa,
            "saturation_vapor_pressure_mean_kpa": self.saturation_vapor_pressure_mean_kpa,
            "humidity_sources": dict(self.humidity_sources),
            "humidity_relation": "FAO-56 Eq. 11 (Tetens), Eq. 12 (mean of extrema), Eq. 19" if self.humidity_sources else None,
            "humidity_consistency_kpa": (self.humidity_consistency_kpa
                                         if self._derived.get("rh_source") == self._derived.get("vpd_source") == "supplied"
                                         else None),
            "radiation_mj_m2_day": self.radiation_mj_m2_day,
            "pet_mm_day_explicit": self.pet_mm_day,
            "provenance": self.provenance,
        }


# --------------------------------------------------------------------------------------------------------------
# Array backend and owned payloads
# --------------------------------------------------------------------------------------------------------------


def array_namespace(backend: str) -> ModuleType:
    """The explicitly selected array module: `numpy`, or CuPy for `cuda` (imported lazily here and only here; a
    missing CuPy or a CuPy without a usable device refuses). Never a silent fallback."""
    if backend == "numpy":
        return np
    if backend == "cuda":
        try:
            import cupy  # type: ignore[import-not-found]
        except ImportError as exc:
            raise WeatherBackendUnavailableError("backend 'cuda' selected but CuPy is not importable") from exc
        try:
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise WeatherBackendUnavailableError("backend 'cuda' selected but no CUDA device is visible")
        except WeatherBackendUnavailableError:
            raise
        except Exception as exc:  # CuPy raises its own runtime error types; report, never fall back
            raise WeatherBackendUnavailableError(f"backend 'cuda' selected but the CUDA runtime failed: {exc}") from exc
        return cupy
    raise WeatherValidationError(f"backend must be one of {BACKENDS}, got {backend!r}")


def _own_array(xp: ModuleType, array: Any, name: str) -> Any:
    if type(array) is not xp.ndarray:
        raise WeatherValidationError(
            f"{name} must be a genuine {xp.__name__}.ndarray of the selected backend (no list, subclass or other "
            f"namespace), got {type(array).__name__}"
        )
    return array


def _host_bytes(xp: ModuleType, array: Any) -> bytes:
    """Host bytes of an array for identity hashing: ONE device->host read for CuPy. Cold-boundary use only."""
    host = array.get() if xp is not np else array
    return np.ascontiguousarray(host).tobytes()


class _OwnedArray:
    """A float64 array OWNED by this module. Created only internally (sealed). On NumPy the storage itself is
    flagged read-only, so no public view can become writable; on CuPy (no read-only mode) public access is a
    detached copy. `_internal()` is the zero-copy handle for this module's own calculations."""

    __slots__ = ("_array", "_backend", "_xp")

    def __init__(self, seal: object, xp: ModuleType, backend: str, array: Any) -> None:
        if seal is not _SEAL:
            raise WeatherValidationError("owned array payloads are created only by maple_syrup.weather_pet")
        _own_array(xp, array, "owned payload")
        if array.dtype != xp.float64:
            raise WeatherValidationError("owned payloads are float64")
        if xp is np:
            array = np.ascontiguousarray(array)
            array.flags.writeable = False
        object.__setattr__(self, "_xp", xp)
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_array", array)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("owned array payloads are immutable")

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._array.shape)

    def _internal(self) -> Any:
        """Zero-copy handle for module-internal READ use only."""
        return self._array

    def readonly(self) -> Any:
        """NumPy: zero-copy read-only view (cannot be made writable). CuPy: detached copy (allocates)."""
        if self._xp is np:
            view = self._array.view()
            view.flags.writeable = False
            return view
        return self._array.copy()

    def detached(self) -> Any:
        """Always a fresh writable copy of the payload (allocates N doubles on the payload's backend)."""
        return self._array.copy()

    def sha256(self) -> str:
        """SHA-256 of the host bytes: ONE N-sized device->host read on CuPy. Cold boundary only."""
        return hashlib.sha256(_host_bytes(self._xp, self._array)).hexdigest()


# --------------------------------------------------------------------------------------------------------------
# Terrain
# --------------------------------------------------------------------------------------------------------------


def aspect_factor_table(latitude_deg: float) -> tuple[float, ...]:
    """Factors indexed by aspect code for the SIGNED latitude. latitude >= 0 (equator included by declared
    convention): the literal northern table. latitude < 0: north-south reflection (N<->S, NE<->SE, NW<->SW; E, W,
    flat unchanged), the user's explicit choice."""
    lat = validate_latitude_deg(latitude_deg)
    if lat >= 0.0:
        return ASPECT_FACTORS_NORTH
    return tuple(ASPECT_FACTORS_NORTH[ASPECT_REFLECTION_SOUTH[code]] for code in range(len(ASPECT_LABELS)))


def aspect_codes_from_labels(labels: Any, xp: ModuleType = np) -> Any:
    """Declared compass labels of the downslope face ("N", "NE", ..., "flat"; case-insensitive, common aliases) ->
    int8 code array of `xp` in the source's enum order. Numeric codes 0..8 are accepted as-is after range checks.
    Labels are converted on the host (a label array is authored input, not a per-step array)."""
    host = np.asarray(labels)
    if host.dtype.kind in ("U", "S", "O"):
        flat = [str(x) for x in host.ravel().tolist()]
        codes = []
        for text in flat:
            key = text.strip().lower()
            if key not in _LABEL_ALIASES:
                raise WeatherValidationError(
                    f"unknown aspect label {text!r}; declared labels are {ASPECT_LABELS} (compass direction the slope "
                    "faces, geographic north) or codes 0..8"
                )
            codes.append(ASPECT_CODES[_LABEL_ALIASES[key]])
        out = np.asarray(codes, dtype=np.int8).reshape(host.shape)
    elif host.dtype.kind in ("i", "u"):
        if host.size and (host.min() < 0 or host.max() >= len(ASPECT_LABELS)):
            raise WeatherValidationError(f"aspect codes must lie in 0..{len(ASPECT_LABELS) - 1}")
        out = host.astype(np.int8)
    else:
        raise WeatherValidationError(f"aspect labels must be strings or integer codes, got dtype {host.dtype}")
    return out if xp is np else xp.asarray(out)


class TerrainFactor:
    """Per-cell multiplier aspect_factor * cos(slope) with its identity, built ONLY by `terrain_factor`. The
    payload is owned (see module docstring): `values` is a read-only zero-copy view on NumPy and a detached copy on
    CuPy; `detached_values()` always copies; `verify_identity()` is the explicit cold check that the owned payload
    still hashes to the recorded `sha256`. Attribute assignment and direct construction are refused."""

    __slots__ = ("_payload", "backend", "cell_count", "flat_cell_count", "flat_policy", "flat_threshold_deg",
                 "hemisphere_table", "latitude_deg", "max_value", "sha256", "shape")

    def __init__(self, seal: object, payload: _OwnedArray, latitude_deg: float, hemisphere_table: str, sha256: str,
                 flat_policy: str, flat_threshold_deg: float | None, flat_cell_count: int, max_value: float) -> None:
        if seal is not _SEAL or not isinstance(payload, _OwnedArray):
            raise WeatherValidationError("TerrainFactor is constructed only by terrain_factor()")
        shape = payload.shape
        for name, value in (("_payload", payload), ("backend", payload.backend), ("shape", shape),
                            ("cell_count", int(math.prod(shape)) if shape else 1), ("latitude_deg", latitude_deg),
                            ("hemisphere_table", hemisphere_table), ("sha256", sha256), ("flat_policy", flat_policy),
                            ("flat_threshold_deg", flat_threshold_deg), ("flat_cell_count", flat_cell_count),
                            ("max_value", max_value)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("TerrainFactor is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("TerrainFactor is immutable")

    def __repr__(self) -> str:
        return (f"TerrainFactor(backend={self.backend!r}, shape={self.shape}, latitude_deg={self.latitude_deg}, "
                f"hemisphere_table={self.hemisphere_table!r}, flat_policy={self.flat_policy!r}, sha256={self.sha256[:12]}...)")

    @property
    def values(self) -> Any:
        """NumPy: zero-copy read-only view. CuPy: detached copy (allocates on device). Not for per-step use."""
        return self._payload.readonly()

    def detached_values(self) -> Any:
        """A fresh writable copy of the factor (allocates), e.g. for a checkpoint writer."""
        return self._payload.detached()

    def verify_identity(self) -> bool:
        """Recompute the payload hash (ONE N-sized host read on CuPy) and compare with `sha256`. Cold boundary."""
        return self._payload.sha256() == self.sha256

    def identity(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "shape": list(self.shape),
            "cell_count": self.cell_count,
            "backend": self.backend,
            "latitude_deg": self.latitude_deg,
            "hemisphere_table": self.hemisphere_table,
            "flat_policy": self.flat_policy,
            "flat_threshold_deg": self.flat_threshold_deg,
            "flat_cell_count": self.flat_cell_count,
            "max_value": self.max_value,
            "aspect_table_source": SOURCE_IDENTITY["aspect_table"],
            "flat_threshold_source": SOURCE_IDENTITY["flat_threshold"],
            "slope_source": SOURCE_IDENTITY["slope_use"],
            "definition": "aspect_factor(code, hemisphere) * cos(slope_rad); apply exactly once",
        }


def terrain_factor(aspect_codes: Any, slope_rad: Any, latitude_deg: float, backend: str = "numpy",
                   flat_threshold_deg: float | None = SOURCE_FLAT_THRESHOLD_DEG) -> TerrainFactor:
    """Build the per-cell multiplier aspect_factor(code, hemisphere) * cos(slope_rad) as an owned float64 payload of
    the selected backend, same shape as the inputs. Codes: integer array in 0..8 (see `aspect_codes_from_labels`);
    slopes: radians, finite, 0 <= slope < pi/2 (non-vertical); both are read, never modified. Flat policy: with the
    default `flat_threshold_deg=5.0` every cell whose slope is below the threshold is classified flat as the source
    does (`WaterLandscape.cpp:1281-1282`), so its factor is cos(slope) regardless of its code; `None` selects the
    caller-classified mode (codes used verbatim). Validation is a few bounded reductions and the identity hash is one
    host read; a terrain factor is built once per static terrain, not per step. Apply it exactly ONCE (via
    `apply_terrain` or by the consumer's actual E/T computation, never both)."""
    xp = array_namespace(backend)
    codes = _own_array(xp, aspect_codes, "aspect_codes")
    slope = _own_array(xp, slope_rad, "slope_rad")
    if codes.dtype.kind not in ("i", "u"):
        raise WeatherValidationError(f"aspect_codes must be an integer array, got dtype {codes.dtype}")
    if slope.dtype != xp.float64:
        raise WeatherValidationError(f"slope_rad must be float64, got dtype {slope.dtype}")
    if codes.shape != slope.shape:
        raise WeatherValidationError(f"aspect_codes {codes.shape} and slope_rad {slope.shape} must share a shape")
    if codes.size:
        lo, hi = int(codes.min()), int(codes.max())
        if lo < 0 or hi >= len(ASPECT_LABELS):
            raise WeatherValidationError(f"aspect codes must lie in 0..{len(ASPECT_LABELS) - 1}, got {lo}..{hi}")
        if not bool(xp.all(xp.isfinite(slope) & (slope >= 0.0) & (slope < math.pi / 2.0))):
            raise WeatherValidationError("slope_rad must be finite with 0 <= slope < pi/2 (radians, non-vertical)")
    lat = validate_latitude_deg(latitude_deg)
    if flat_threshold_deg is None:
        policy, threshold = _FLAT_POLICY_CALLER, None
        effective_codes = codes.astype(xp.intp)
        flat_count = int(xp.count_nonzero(codes == 0)) if codes.size else 0
    else:
        threshold = _finite_float(flat_threshold_deg, "flat_threshold_deg")
        if not (0.0 <= threshold < 90.0):
            raise WeatherValidationError(f"flat_threshold_deg must satisfy 0 <= threshold < 90, got {threshold}")
        policy = _FLAT_POLICY_SOURCE
        is_flat = slope < math.radians(threshold)  # build-time temporary; the caller's codes are untouched
        effective_codes = xp.where(is_flat, 0, codes.astype(xp.intp))
        flat_count = int(xp.count_nonzero(is_flat)) if codes.size else 0
    table = xp.asarray(aspect_factor_table(lat), dtype=xp.float64)
    values = xp.take(table, effective_codes) * xp.cos(slope)
    if values.size and not bool(xp.all(xp.isfinite(values) & (values > 0.0))):
        raise WeatherValidationError("terrain factor must be finite and > 0 everywhere (internal check failed)")
    payload = _OwnedArray(_SEAL, xp, backend, values)
    hemisphere = "north_literal" if lat >= 0.0 else "south_reflected"
    max_value = float(values.max()) if values.size else 0.0  # build-time reduction, reused by apply_terrain's O(1) guard
    return TerrainFactor(_SEAL, payload, lat, hemisphere, payload.sha256(), policy, threshold, flat_count, max_value)


# --------------------------------------------------------------------------------------------------------------
# Potential demand
# --------------------------------------------------------------------------------------------------------------


class PotentialDemand:
    """REFERENCE / POTENTIAL evaporative demand for one date: `pet_mm_day` is a uniform daily rate, finite and
    >= 0 by construction. It is not soil evaporation, not transpiration and not their sum spent twice: the consumer
    partitions it ONCE under its own documented convention and SYRUP's inventory decides what is actually
    withdrawn. `terrain_applied` tells whether an aspect/slope multiplier was already folded in; then
    `pet_mm_day_cells` (read-only view on NumPy, detached copy on CuPy) and `detached_cells()` (always a copy)
    expose the owned per-cell array and `terrain` carries the identity.

    Sealed (correction 02): built ONLY by `potential_demand` and `apply_terrain`. It is not a dataclass, so
    `dataclasses.replace` cannot rebuild it with a negative PET, another date/weather/method or a stale payload;
    direct construction and attribute assignment refuse. Every field is read-only; `provenance()` returns detached
    metadata."""

    __slots__ = ("_cells", "clamped", "date", "latitude_deg", "method", "pet_mm_day", "ra_equiv_mm_day", "terrain",
                 "terrain_applied", "weather")

    def __init__(self, seal: object, *, date: _dt.date, latitude_deg: float | None, method: str, pet_mm_day: float,
                 ra_equiv_mm_day: float | None, clamped: bool, weather: DailyWeather,
                 terrain: TerrainFactor | None = None, cells: _OwnedArray | None = None) -> None:
        if seal is not _SEAL:
            raise WeatherValidationError("PotentialDemand is constructed only by potential_demand() and apply_terrain()")
        if not isinstance(weather, DailyWeather) or weather.date != date or method not in _IMPLEMENTED_METHODS:
            raise WeatherValidationError("internal: demand metadata inconsistent with its weather record")
        if not (isinstance(pet_mm_day, float) and math.isfinite(pet_mm_day) and pet_mm_day >= 0.0):
            raise WeatherValidationError(f"internal: reference PET must be finite and >= 0, got {pet_mm_day!r}")
        if (cells is None) != (terrain is None):
            raise WeatherValidationError("internal: terrain payload and identity must be given together")
        if cells is not None:
            if not isinstance(cells, _OwnedArray) or not isinstance(terrain, TerrainFactor):
                raise WeatherValidationError("terrain-applied demands are created only by apply_terrain()")
            if cells.shape != terrain.shape or cells.backend != terrain.backend:
                raise WeatherValidationError("terrain-applied payload does not match its terrain identity")
        for name, value in (("date", date), ("latitude_deg", latitude_deg), ("method", method), ("pet_mm_day", pet_mm_day),
                            ("ra_equiv_mm_day", ra_equiv_mm_day), ("clamped", clamped), ("weather", weather),
                            ("terrain_applied", cells is not None), ("terrain", terrain), ("_cells", cells)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("PotentialDemand is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("PotentialDemand is immutable")

    def __repr__(self) -> str:
        return (f"PotentialDemand(date={self.date.isoformat()}, method={self.method!r}, pet_mm_day={self.pet_mm_day}, "
                f"latitude_deg={self.latitude_deg}, clamped={self.clamped}, terrain_applied={self.terrain_applied})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PotentialDemand):
            return NotImplemented
        return (self.date, self.latitude_deg, self.method, self.pet_mm_day, self.ra_equiv_mm_day, self.clamped,
                self.weather, self.terrain_applied, None if self.terrain is None else self.terrain.sha256) == (
                other.date, other.latitude_deg, other.method, other.pet_mm_day, other.ra_equiv_mm_day, other.clamped,
                other.weather, other.terrain_applied, None if other.terrain is None else other.terrain.sha256)

    __hash__ = None  # type: ignore[assignment]

    @property
    def pet_mm_day_cells(self) -> Any:
        """Per-cell mm/day (terrain applied): NumPy read-only view / CuPy detached copy; None otherwise."""
        return None if self._cells is None else self._cells.readonly()

    def detached_cells(self) -> Any:
        """A fresh writable copy of the per-cell array (allocates); None without terrain."""
        return None if self._cells is None else self._cells.detached()

    def provenance(self) -> dict[str, Any]:
        """A METADATA SNAPSHOT of the exact inputs and results: method, dated calendar, latitude, the FULL weather
        record (declared and derived values), the terrain identity when applied, and the source identity of the
        equations. It is what a consumer should carry in its derived forcing / restart record; it is not itself a
        checkpoint format and does not relax any native source/checkpoint guard."""
        return {
            "method": self.method,
            "date": self.date.isoformat(),
            "day_of_year": day_of_year(self.date),
            "days_per_year_reference": DAYS_PER_YEAR_REFERENCE,
            "latitude_deg": self.latitude_deg,
            "weather": self.weather.record(),
            "pet_mm_day": self.pet_mm_day,
            "ra_equiv_mm_day": self.ra_equiv_mm_day,
            "clamped": self.clamped,
            "terrain_applied": self.terrain_applied,
            "terrain": self.terrain.identity() if self.terrain is not None else None,
            "source": dict(SOURCE_IDENTITY),
        }


def hargreaves_pet_mm_day(tmin_c: float, tmean_c: float, tmax_c: float, ra_equiv_mm_day: float) -> tuple[float, bool]:
    """The literal EcoHyD/Hargreaves law given Ra_equiv in mm/day; returns (PET_mm_day, clamped). Zero when
    Tmax == Tmin; negative empirical values (Tmean < -17.8 C) are clamped to 0 and flagged."""
    tmin = _finite_float(tmin_c, "tmin_c")
    tmean = _finite_float(tmean_c, "tmean_c")
    tmax = _finite_float(tmax_c, "tmax_c")
    ra = _finite_float(ra_equiv_mm_day, "ra_equiv_mm_day")
    if not (tmin <= tmean <= tmax):
        raise WeatherValidationError(f"temperatures must satisfy tmin <= tmean <= tmax, got {tmin}, {tmean}, {tmax}")
    if ra < 0.0:
        raise WeatherValidationError(f"ra_equiv_mm_day must be >= 0, got {ra}")
    raw = HARGREAVES_COEFFICIENT * (tmean + HARGREAVES_OFFSET_C) * math.sqrt(tmax - tmin) * ra
    if not math.isfinite(raw):
        raise WeatherValidationError(f"Hargreaves PET overflowed to {raw!r} for tmean={tmean}, tmax-tmin={tmax - tmin}, ra={ra}")
    if raw < 0.0:
        return 0.0, True  # the documented cold clamp (Tmean < -17.8 C); the only clipping in this module
    return raw, False


def potential_demand(weather: DailyWeather, latitude_deg: float | None, method: str = "ecohyd_hargreaves") -> PotentialDemand:
    """Reference PET for one daily record. `ecohyd_hargreaves` needs tmin/tmax and a latitude and computes from the
    dated astronomy; `explicit` returns the record's declared `pet_mm_day` (tmean still required by the record,
    latitude may be None and is then recorded as None, never 0); `penman_monteith` and unknown methods refuse."""
    if not isinstance(weather, DailyWeather):
        raise WeatherValidationError(f"weather must be a DailyWeather, got {type(weather).__name__}")
    if method not in METHODS:
        raise UnsupportedPETMethodError(f"unknown PET method {method!r}; declared methods are {METHODS}, implemented "
                                        f"{_IMPLEMENTED_METHODS}")
    if method == "penman_monteith":
        raise UnsupportedPETMethodError(
            "penman_monteith is a declared hook without an implementation: it needs wind speed, net radiation, "
            "pressure and surface/aerodynamic resistances that DailyWeather does not define"
        )
    if method == "explicit":
        if weather.pet_mm_day is None:
            raise WeatherValidationError("method 'explicit' requires DailyWeather.pet_mm_day (declared mm/day)")
        lat = None if latitude_deg is None else validate_latitude_deg(latitude_deg)
        return PotentialDemand(_SEAL, date=weather.date, latitude_deg=lat, method=method, pet_mm_day=weather.pet_mm_day,
                               ra_equiv_mm_day=None, clamped=False, weather=weather)
    lat = validate_latitude_deg(latitude_deg)
    if not weather.has_extrema:
        raise WeatherValidationError("method 'ecohyd_hargreaves' requires tmin_c and tmax_c (daily extrema)")
    ra = extraterrestrial_radiation_equivalent_mm_day(weather.date, lat)
    pet, clamped = hargreaves_pet_mm_day(weather.tmin_c, weather.tmean_c, weather.tmax_c, ra)
    return PotentialDemand(_SEAL, date=weather.date, latitude_deg=lat, method=method, pet_mm_day=pet,
                           ra_equiv_mm_day=ra, clamped=clamped, weather=weather)


def pet_m_s(pet_mm_day: Any) -> Any:
    """mm/day -> m/s (x 1e-3 / 86400); works for a float or an array of either namespace."""
    return pet_mm_day * (1e-3 / SECONDS_PER_DAY)


def apply_terrain(demand: PotentialDemand, terrain: TerrainFactor) -> PotentialDemand:
    """Scalar reference demand x owned per-cell terrain factor -> terrain-applied per-cell potential (owned mm/day
    float64 payload of the terrain's backend; one fresh N-array allocation, zero-copy read of the factor, no host
    transfer). Refuses a demand that already has terrain applied (no double adjustment), anything but a
    `TerrainFactor`, and a latitude mismatch between a located demand and the terrain. The factor payload was
    validated finite and > 0 at build and cannot be modified through the public API, so the recorded terrain
    identity covers exactly what is multiplied here; `TerrainFactor.verify_identity()` is the explicit cold check."""
    if not isinstance(demand, PotentialDemand):
        raise WeatherValidationError(f"demand must be a PotentialDemand, got {type(demand).__name__}")
    if demand.terrain_applied:
        raise WeatherValidationError("terrain factor already applied to this demand; a second application would "
                                     "double-count slope/aspect")
    if not isinstance(terrain, TerrainFactor):
        raise WeatherValidationError(f"terrain must be a TerrainFactor built by terrain_factor(), got {type(terrain).__name__}")
    if demand.latitude_deg is not None and demand.latitude_deg != terrain.latitude_deg:
        raise WeatherValidationError(
            f"demand latitude {demand.latitude_deg} differs from terrain latitude {terrain.latitude_deg}; one site"
        )
    xp = array_namespace(terrain.backend)
    # Overflow guard at O(1): the largest cell is pet x max factor (max recorded once at build), so no N-sized
    # reduction or host transfer is needed per date. Inputs are finite and >= 0, so this is the only failure mode.
    if not math.isfinite(demand.pet_mm_day * terrain.max_value):
        raise WeatherValidationError(
            f"terrain-applied PET would overflow: pet_mm_day={demand.pet_mm_day} x max factor {terrain.max_value}"
        )
    cells = terrain._payload._internal() * demand.pet_mm_day  # fresh array; pet_mm_day >= 0 and factor > 0
    payload = _OwnedArray(_SEAL, xp, terrain.backend, cells)
    return PotentialDemand(_SEAL, date=demand.date, latitude_deg=demand.latitude_deg, method=demand.method,
                           pet_mm_day=demand.pet_mm_day, ra_equiv_mm_day=demand.ra_equiv_mm_day, clamped=demand.clamped,
                           weather=demand.weather, terrain=terrain, cells=payload)


# --------------------------------------------------------------------------------------------------------------
# Dated intervals (interevent only)
# --------------------------------------------------------------------------------------------------------------


def _day_fraction(seconds: Any, name: str = "seconds") -> tuple[float, float]:
    """(seconds as float, fraction of day as float). Fraction input is divided exactly and rounded to FP64 once."""
    if isinstance(seconds, Fraction):
        if not (0 < seconds <= SECONDS_PER_DAY):
            raise WeatherValidationError(f"{name} must satisfy 0 < seconds <= {SECONDS_PER_DAY} (one date), got {seconds}")
        return float(seconds), float(seconds / SECONDS_PER_DAY)
    numeric = _finite_float(seconds, name)
    if not (0.0 < numeric <= SECONDS_PER_DAY):
        raise WeatherValidationError(f"{name} must satisfy 0 < seconds <= {SECONDS_PER_DAY} (one date), got {numeric}")
    return numeric, numeric / SECONDS_PER_DAY


def pet_depth_mm(demand: PotentialDemand, seconds: Any) -> Any:
    """Uniform-rate depth (mm) of `demand` over `seconds` WITHIN its date: 0 < seconds <= 86400 (int, float or
    Fraction). Returns a fresh per-cell array (caller-owned) when terrain was applied, else a float. No sub-daily
    curve is invented."""
    if not isinstance(demand, PotentialDemand):
        raise WeatherValidationError(f"demand must be a PotentialDemand, got {type(demand).__name__}")
    _, fraction = _day_fraction(seconds)
    if demand.terrain_applied:
        return demand._cells._internal() * fraction
    return demand.pet_mm_day * fraction


@dataclass(frozen=True)
class IntervalPiece:
    """One (date, seconds) chunk of an integration with its SCALAR reference depth (unadjusted PET x fraction of
    day). No per-cell array is retained per piece."""

    date: _dt.date
    seconds: float
    reference_depth_mm: float


@dataclass(frozen=True)
class IntereventPET:
    """Result of an interevent integration with BOUNDED diagnostics: the summed depth (`depth_mm`: float, or ONE
    fresh caller-owned per-cell array when the records were terrain-applied), scalar totals, first/last date,
    the FIRST terrain identity and how many times the identity changed between consecutive chunks, and at most
    `keep_pieces` retained `IntervalPiece` records (default none)."""

    depth_mm: Any
    reference_depth_mm: float
    total_seconds: float
    chunk_count: int
    first_date: _dt.date
    last_date: _dt.date
    terrain_applied: bool
    terrain_sha256_first: str | None
    terrain_identity_changes: int
    pieces: tuple[IntervalPiece, ...]
    pieces_truncated: bool


def _lookup(records: Mapping[_dt.date, PotentialDemand] | Callable[[_dt.date], PotentialDemand], date: _dt.date) -> PotentialDemand:
    try:
        demand = records[date] if isinstance(records, Mapping) else records(date)
    except KeyError as exc:
        raise WeatherValidationError(f"no weather/PET record for {date.isoformat()}") from exc
    if not isinstance(demand, PotentialDemand):
        raise WeatherValidationError(f"records must yield PotentialDemand, got {type(demand).__name__}")
    if demand.date != date:
        raise WeatherValidationError(f"record for {date.isoformat()} carries date {demand.date.isoformat()}")
    return demand


def _refuse_non_interevent(phase: str) -> None:
    if phase not in PHASES:
        raise WeatherValidationError(f"phase must be one of {PHASES}, got {phase!r}")
    if phase != INTEREVENT:
        raise WeatherPhaseError(f"no evaporation/transpiration demand during phase {phase!r}; only interevent "
                                "intervals integrate PET (no catch-up)")


def integrate_interevent_pet_chunks(
    chunks: Iterable[tuple[_dt.date, Any]],
    records: Mapping[_dt.date, PotentialDemand] | Callable[[_dt.date], PotentialDemand],
    phase: str,
    *,
    keep_pieces: int = 0,
    on_chunk: Callable[[IntervalPiece], None] | None = None,
) -> IntereventPET:
    """Sum the uniform-rate PET depth over the scheduler's EXACT declared `(date, seconds)` chunks (seconds: int,
    float or Fraction, 0 < s <= 86400 each; several chunks of one date are allowed). `records` maps a date to its
    PotentialDemand (mapping or callable; a missing date refuses, never zero-fills). `phase` must be "interevent";
    wind, rain and recession refuse (`WeatherPhaseError`) without spending demand, and nothing is caught up later.
    Resident work is O(N) + bounded diagnostics: one running sum plus one scratch array for terrain-applied records
    (both allocated once, independent of the number of chunks), scalar summaries, at most `keep_pieces` retained
    pieces (default 0), and an optional `on_chunk(piece)` callback for the caller's own bookkeeping. Terrain-applied
    records must all share shape and backend; mixing scalar and terrain-applied records refuses."""
    _refuse_non_interevent(phase)
    if isinstance(keep_pieces, bool) or not isinstance(keep_pieces, (int, np.integer)) or keep_pieces < 0:
        raise WeatherValidationError(f"keep_pieces must be a non-negative integer, got {keep_pieces!r}")
    if on_chunk is not None and not callable(on_chunk):
        raise WeatherValidationError("on_chunk must be callable or None")
    pieces: list[IntervalPiece] = []
    truncated = False
    total: Any = None
    scratch: Any = None
    reference_total = 0.0
    seconds_total = 0.0
    count = 0
    first_date: _dt.date | None = None
    last_date: _dt.date | None = None
    terrain_mode: bool | None = None
    xp: ModuleType | None = None
    shape: tuple[int, ...] | None = None
    first_sha: str | None = None
    previous_sha: str | None = None
    identity_changes = 0
    for chunk in chunks:
        try:
            date, seconds = chunk
        except (TypeError, ValueError) as exc:
            raise WeatherValidationError("each chunk must be a (date, seconds) pair") from exc
        _validate_date(date)
        secs, fraction = _day_fraction(seconds)
        demand = _lookup(records, date)
        if terrain_mode is None:
            terrain_mode = demand.terrain_applied
        elif terrain_mode != demand.terrain_applied:
            raise WeatherValidationError("all records of one integration must agree on terrain_applied")
        reference_depth = demand.pet_mm_day * fraction
        piece = IntervalPiece(date, secs, reference_depth)
        if len(pieces) < keep_pieces:
            pieces.append(piece)
        elif keep_pieces:
            truncated = True
        if on_chunk is not None:
            on_chunk(piece)
        reference_total += reference_depth
        seconds_total += secs
        count += 1
        first_date = date if first_date is None else first_date
        last_date = date
        if not terrain_mode:
            total = reference_depth if total is None else total + reference_depth
            continue
        terrain = demand.terrain
        cells = demand._cells._internal()
        if xp is None:
            xp = array_namespace(terrain.backend)
            shape = terrain.shape
            total = cells * fraction  # the ONE running sum (fresh array, inputs untouched)
            scratch = xp.empty_like(total)
        else:
            if array_namespace(terrain.backend) is not xp:
                raise WeatherValidationError("all terrain-applied records of one integration must share a backend")
            if terrain.shape != shape:
                raise WeatherValidationError(
                    f"terrain shape {terrain.shape} differs from the integration's shape {shape}; no broadcasting"
                )
            xp.multiply(cells, fraction, out=scratch)
            with np.errstate(over="ignore"):  # overflow is detected by the end-of-integration check, not a warning
                total += scratch
        if first_sha is None:
            first_sha = terrain.sha256
        elif terrain.sha256 != previous_sha:
            identity_changes += 1
        previous_sha = terrain.sha256
    if count == 0:
        raise WeatherValidationError("no chunks to integrate")
    # Overflow guard: every addend is finite and >= 0, so a non-finite sum can only be an overflow and it persists;
    # ONE check at the end suffices (O(1) for scalars, one bounded N-reduction for the per-cell sum).
    if not math.isfinite(reference_total):
        raise WeatherValidationError(f"integrated reference PET overflowed to {reference_total!r} over {count} chunks")
    if terrain_mode and not bool(xp.all(xp.isfinite(total))):
        raise WeatherValidationError(f"integrated per-cell PET overflowed over {count} chunks")
    return IntereventPET(total, reference_total, seconds_total, count, first_date, last_date, bool(terrain_mode),
                         first_sha, identity_changes, tuple(pieces), truncated)


def _naive(instant: Any, name: str) -> _dt.datetime:
    if not isinstance(instant, _dt.datetime):
        raise WeatherValidationError(f"{name} must be a datetime.datetime, got {type(instant).__name__}")
    if instant.tzinfo is not None:
        raise WeatherValidationError(
            f"{name} must be a naive datetime in the scheduler's declared calendar; an offset-aware or "
            "rational-subsecond calendar should pass exact (date, seconds) chunks to integrate_interevent_pet_chunks"
        )
    return instant


def integrate_interevent_pet(
    start: _dt.datetime,
    end: _dt.datetime,
    records: Mapping[_dt.date, PotentialDemand] | Callable[[_dt.date], PotentialDemand],
    phase: str,
    *,
    keep_pieces: int = 0,
    on_chunk: Callable[[IntervalPiece], None] | None = None,
) -> IntereventPET:
    """Convenience wrapper: split the NAIVE datetime interval [start, end) at midnight into exact (date, seconds)
    chunks (microsecond resolution of `datetime`) and integrate them with `integrate_interevent_pet_chunks`."""
    s, e = _naive(start, "start"), _naive(end, "end")
    _refuse_non_interevent(phase)
    if not e > s:
        raise WeatherValidationError(f"end must be after start, got start={s.isoformat()} end={e.isoformat()}")

    def chunks():
        cursor = s
        while cursor < e:
            next_midnight = _dt.datetime.combine(cursor.date() + _dt.timedelta(days=1), _dt.time.min)
            piece_end = min(e, next_midnight)
            delta = piece_end - cursor
            # exact: datetime arithmetic is integer microseconds
            yield cursor.date(), Fraction(delta.days * SECONDS_PER_DAY * 1_000_000 + delta.seconds * 1_000_000 + delta.microseconds, 1_000_000)
            cursor = piece_end

    return integrate_interevent_pet_chunks(chunks(), records, phase, keep_pieces=keep_pieces, on_chunk=on_chunk)
