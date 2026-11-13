# Shared weather to reference PET (`maple_syrup.weather_pet`)

Phase 0B candidate, 2026-11-10 (corrections 01 and 02 applied). Opt-in physical library
that turns one daily weather record plus an explicit site latitude into a
REFERENCE / POTENTIAL evaporative demand for SYRUP and LEAF. It is not a
scheduler, not a water authority, not a metadata owner, and nothing in it
changes an existing default. This document describes what the module does,
where every number comes from, where it departs from its reference, and what it
does not claim.

## Stable call signatures

```
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
latitude_from_maple_location(location) -> float
```

Method names: `"ecohyd_hargreaves"` (computed), `"explicit"` (declared PET plus
mean temperature), `"penman_monteith"` (declared hook, refuses). Phases:
`"interevent"` integrates; `"wind"`, `"rain"`, `"recession"` refuse.

## Source of the physics

All equations and constants are transcribed from the downloaded Tietjen-lineage
EcoHyD source described in [ecohyd_reference_sources.md](ecohyd_reference_sources.md):
GitHub snapshot `80b10e6b6b82ad16e9fd41a31901cc3157d880e9`, local root
`reference_models/ecohyd/github_baldauf_2023/source/model/src/`. The two Dryad
EcoHyD packages carry byte-identical constants at the lines listed in
`weather_pet.SOURCE_IDENTITY`. Scientific references:

- Tietjen, Zehe & Jeltsch (2009), *Water Resources Research* 45, W01418,
  <https://doi.org/10.1029/2007WR006589> (the original model paper).
- Hargreaves & Samani (1985), *Applied Engineering in Agriculture* 1(2), 96–99
  (the temperature-based PET law).
- Allen, Pereira, Raes & Smith (1998), FAO Irrigation and Drainage Paper 56,
  <https://www.fao.org/4/x0490e/x0490e07.htm>: Eq. 52 (Hargreaves with Ra in
  mm/day), Eqs. 21–25 (Ra), Eqs. 11, 12, 19 (vapour pressure).
- Shevenell (1999), *Hydrological Processes* 13, 577–596 (aspect factors, as
  cited by the source comment at `WaterCell.cpp:243-246`).
- Maidment (ed.) (1993), *Handbook of Hydrology*, p. 4.31 (radiation routine,
  as cited by the source comment at `Weather.cpp:138`).

| Quantity | Source line | Literal expression |
|---|---|---|
| PET (mm/day) | `WaterLandscape.cpp:805-807` | `0.0023 (Tmean + 17.8) sqrt(Tmax − Tmin) Ra_equiv` |
| dr | `Weather.cpp:141` | `1 + 0.033 cos(2πJ/365)` |
| declination | `Weather.cpp:142` | `0.4093 sin(2πJ/365 − 1.405)` |
| sunset hour angle | `Weather.cpp:143` | `acos(−tan φ tan δ)` |
| Ra_equiv (mm/day) | `Weather.cpp:144` | `15.392 dr (ws sin φ sin δ + cos φ cos δ sin ws)` |
| aspect factors | `WaterCell.cpp:251-281` | flat 1.0, N 0.90, NE 0.95, E 0.98, SE 1.03, S 1.10, SW 1.05, W 1.02, NW 0.97 |
| aspect enum | `WaterLandscape.h:22-32` | FL=0, NN=1, NE=2, EE=3, SE=4, SS=5, SW=6, WW=7, NW=8 |
| flat threshold | `WaterLandscape.cpp:1281-1282` | `inclination_deg < 5` → FL, before the table is applied |
| slope use | `WaterCell.cpp:41-89` | `cos(inclination)` multiplies each actual E/T term |

## Method `ecohyd_hargreaves` and its declared departures

The coefficients 15.392, 0.4093 and 1.405 are preserved literally. They are
**not** the FAO-56 constants: FAO-56 writes Ra in MJ m⁻² day⁻¹ with
(24·60/π)·Gsc = 37.586, declination `0.409 sin(2πJ/365 − 1.39)`, and converts
to mm/day with 0.408, so its equivalent of 15.392 is 0.408 × 37.586 = 15.335
(about 0.4 % lower). The module therefore names the method `ecohyd_hargreaves`
and does not claim FAO-56 identity. Measured difference on FAO-56 Example 8
(20° S, 3 September): FAO-56 gives 13.1 mm/day, this module 13.127 mm/day.
FAO-56 Example 20 (Hargreaves given Ra = 16.6 mm/day) reproduces 5.05 mm/day
against the published 5.0.

Calendar departure: J is the Gregorian day of year of the caller's explicit
date (leap days included), while the denominator stays the reference's 365.
The downloaded source indexes a fixed 365-day year with no leap days
(`Parameters.h:19`, `Weather.cpp:141-142` use `day % daysPerYear + 1`); a dated
scheduler cannot do that without drifting. On 31 December of a leap year J =
366 and the phase runs one day past a full cycle. The source comment
"northern hemisphere: 1st of January (day 0); southern hemisphere: 1st of July"
describes the reference's weather-file start convention; this module takes the
true signed latitude and the true date and needs no such shift.

Polar handling: the `acos` argument is clamped analytically. Argument ≤ −1 is
polar day (ws = π), argument ≥ 1 is polar night (ws = 0, Ra = 0). Latitudes
±90° are accepted and produce polar day/night values.

Non-negativity: Hargreaves goes negative for Tmean < −17.8 °C. The physical
reference demand is clamped at zero and the clamp is reported in
`PotentialDemand.clamped`. Tmax = Tmin gives exactly zero.

## Inputs, units and validation

- `latitude_deg`: finite, signed, non-Boolean, in [−90, 90]; **no default**. A
  `None` latitude is refused by every computation that needs one, never
  treated as the equator. `latitude_from_maple_location` is a duck-typed bridge
  to MAPLE's resolved `CaseLocationSpec` (any object/mapping with
  `latitude_deg`); it imports nothing from MAPLE and does not change the
  metadata's descriptive semantics.
- `DailyWeather(date, tmin_c, tmean_c, tmax_c, ...)`: `datetime.date` (not a
  datetime); `tmean_c` always required (LEAF plant processes need temperature
  however PET was obtained); `tmin_c`/`tmax_c` optional but given together,
  with tmin ≤ tmean ≤ tmax. Hargreaves and the humidity derivation require the
  extrema; an explicit PET does not. `DailyWeather.explicit_pet(date, tmean_c,
  pet_mm_day)` builds the extrema-free record without naming the extrema.
- `relative_humidity_fraction` (0..1, not percent) and `vpd_kpa` (≥ 0) are
  optional shared fields for LEAF. One given → the other is derived once under
  FAO-56 Eq. 11/12/19: e°(T) = 0.6108 exp(17.27 T/(T + 237.3)) kPa, es =
  (e°(Tmax) + e°(Tmin))/2, VPD = es (1 − RH). Both given → they must agree
  within the record's `humidity_consistency_kpa` (validated, 0 < tol ≤ 1 kPa,
  default 1e-6; a caller with rounded station data declares a larger tolerance
  explicitly and it is recorded); a conflict is refused. Without extrema the
  derivation is refused (no silent e°(Tmean) substitute). The derived values
  are exposed by `rh_fraction`, `vapor_pressure_deficit_kpa`, and
  `humidity_sources` says which was supplied and which derived. This is a
  daily-mean approximation, not an hourly humidity model.
- `radiation_mj_m2_day` (daily shortwave energy, MJ m⁻² day⁻¹) is optional and
  shared; Hargreaves does not use it and the module does not pretend otherwise.
- `pet_mm_day`: optional explicit declared reference PET.
- `provenance`: `observed`, `prescribed` or `estimated`, labelling the record.
  Simplified scenarios must declare `prescribed`/`estimated`.
- Units out: `PotentialDemand.pet_mm_day` (mm/day); `pet_m_s` converts with
  1e-3/86400 to the m/s the soil column's `potential_evaporation_m_s` forcing
  expects.

## Methods

- `ecohyd_hargreaves` (default): computed as above; needs extrema and latitude.
- `explicit`: returns the record's `pet_mm_day`. Only `tmean_c` is needed; the
  latitude may be `None` and is then recorded as `None` (never 0). The value is
  REFERENCE demand; it is not split into soil evaporation plus plant
  transpiration here, and a consumer must not spend it as full soil E **and**
  full plant T.
- `penman_monteith`: declared hook that refuses with
  `UnsupportedPETMethodError` (needs wind speed, net radiation, pressure and
  resistances the record does not define). Unknown method strings also refuse.
  No fake implementation exists. The physical Hargreaves and humidity equations
  live only in this module; a scheduler (MASY) should call it rather than
  duplicate the formulas.

## Provenance returned to the consumer

`PotentialDemand.provenance()` returns a METADATA SNAPSHOT: the method, ISO
date, day of year and 365 reference denominator, latitude, the FULL weather
record (declared and derived values, humidity relation, sources and declared
tolerance, radiation, explicit PET, provenance label), the computed PET,
Ra_equiv and clamp flag, the terrain identity when applied (SHA-256, shape,
cell count, backend, latitude, hemisphere table, flat policy and threshold,
flat cell count, source lines), and the source identity of the equations. The
module persists nothing; the consumer's derived forcing or restart record owns
this dictionary. It is not itself a checkpoint format and does not relax any
native source or checkpoint guard.

## Terrain factor

`terrain_factor(aspect_codes, slope_rad, latitude_deg, backend, flat_threshold_deg)`
returns a `TerrainFactor`: the owned per-cell multiplier `aspect_factor ×
cos(slope)` with 0 ≤ slope < π/2 (radians, non-vertical), validated finite and
> 0 at build, plus its identity (SHA-256 of the host bytes, one device→host
read at build time only; shape; backend; latitude; hemisphere table; flat
policy). The caller's code and slope arrays are read, never modified. Aspect
codes follow the source enum; `aspect_codes_from_labels` accepts declared
compass labels (`"N"`, `"SE"`, `"flat"`, case-insensitive, common aliases) of
the **downslope face**, geographic north, the same meaning the reference
derives from its runoff direction with the upper map boundary oriented north
(`WaterLandscape.cpp:1284-1297`). MAPLE owns the slope geometry; this module
does not derive aspect or inclination from a DEM.

Flat classification (source-compatible default): the reference sets the aspect
to FL when the inclination is below 5° (`WaterLandscape.cpp:1281-1282`) before
the table is applied, while `cos(inclination)` still multiplies the actual
fluxes. With the default `flat_threshold_deg=5.0` every cell with slope < 5° is
therefore treated as flat: its factor is `cos(slope)` regardless of its code
(a north-facing 1° cell gives 0.99985, not 0.90 × cos; a level "south" cell
gives 1.0, not 1.10). Exactly 5° is not flat. The policy, threshold and flat
cell count are recorded in the identity. `flat_threshold_deg=None` is the
explicitly named caller-classified mode (codes used verbatim); a different
threshold is allowed in [0, 90) and is recorded.

Hemisphere: latitude ≥ 0 uses the literal northern table (the equator is the
northern convention by this declared rule). For latitude < 0 the user
explicitly chose a north–south reflection: N↔S, NE↔SE, NW↔SW, while E, W and
flat keep their factors. There is no east–west swap. Southern table: flat 1.0,
N 1.10, NE 1.03, E 0.98, SE 0.95, S 0.90, SW 0.97, W 1.02, NW 1.05.

Single application: in the reference the aspect factor multiplies potential ET
once and `cos(inclination)` multiplies each actual flux once. Here both are
gathered into one factor. `apply_terrain(demand, terrain)` accepts only a
`TerrainFactor` (no raw arrays), refuses a latitude mismatch between a located
demand and the terrain, marks the result `terrain_applied=True`, binds the
terrain identity into the provenance, and refuses a second application.
Contract for consumers: a terrain-applied potential must not be multiplied by
slope/aspect again in the actual E/T computation; a consumer that applies
terrain itself must take the unadjusted scalar demand. A flat, level cell has
factor exactly 1.0 (identity).

## Sealed records, finite non-negative results, and what allocates

`TerrainFactor` and `PotentialDemand` are factory-only objects with read-only
fields, built only by `terrain_factor`, `potential_demand` and `apply_terrain`.
Neither is a dataclass, so `dataclasses.replace` cannot rebuild one with a
negative PET, another date/weather/method/latitude or a stale payload (it
raises `TypeError`); direct construction refuses with
`WeatherValidationError`, attribute assignment with `AttributeError`.
`DailyWeather` stays a public dataclass and re-validates on every
construction or replacement, so a negative or non-finite declared PET cannot
enter through it either. `provenance()` and `identity()` return detached
dictionaries; editing them changes nothing.

Every published PET is finite and >= 0 by construction:

- Hargreaves refuses a non-finite result (absurd temperatures); the clamp to 0
  for Tmean < −17.8 °C is the only clipping in the module and is flagged.
- `apply_terrain` refuses a product that would overflow, using the O(1) check
  `pet_mm_day × max_value`, where `max_value` is the factor's maximum recorded
  once at build (no per-date reduction or transfer).
- The integrators refuse a non-finite sum with ONE check at the end: all
  addends are finite and >= 0, so a non-finite sum can only be an overflow and
  it persists (O(1) for scalars, one bounded N-reduction for the per-cell sum;
  NumPy's overflow warning is suppressed inside the accumulation because the
  end check is the detector).

A declared explicit PET of 1e308 mm/day is therefore accepted as declared, but
summing it over two full days is refused rather than published as infinity.
`TerrainFactor` and a terrain-applied `PotentialDemand` OWN their arrays; the
recorded identity covers exactly the payload every later `apply_terrain`
multiplies.

| Accessor | NumPy | CuPy | Allocates |
|---|---|---|---|
| `TerrainFactor.max_value` | scalar recorded at build (one reduction) | same | no |
| `TerrainFactor.values`, `PotentialDemand.pet_mm_day_cells` | zero-copy read-only view; the owned base is read-only, so the view cannot be made writable | detached copy (CuPy has no read-only arrays) | no / yes (N doubles on device) |
| `.detached_values()`, `.detached_cells()` | fresh writable copy | fresh writable copy | yes |
| `TerrainFactor.verify_identity()` | rehash of the owned payload | rehash after ONE N-sized device→host read | host bytes, cold boundary only |
| `apply_terrain` | one fresh N-array (the new demand's owned payload); zero-copy read of the factor | same, on device, no host transfer | yes, once per date |
| `pet_depth_mm` | fresh caller-owned array | same | yes |
| integration | one running sum + one scratch array for the whole integration | same | 2N once, independent of the number of chunks |

Module-internal calculations read the owned storage zero-copy. Nothing is
copied or transferred to the host per date or per step after the build.
`verify_identity()` is for explicitly declared cold boundaries (checkpoint,
restart, acceptance), never called implicitly. Module-private attributes are
outside the contract; a test shows `verify_identity()` detecting a stale
payload reached that way.

Backends: `"numpy"` or `"cuda"` selected explicitly. `"cuda"` imports CuPy
lazily and only then; missing CuPy or no visible device is
`WeatherBackendUnavailableError`, never a NumPy fallback. Input arrays must be
genuine `ndarray` objects of the selected namespace (float64 slopes, integer
codes). The scalar astronomy is host-side; the array work is elementwise table
gather, cosine and multiply.

## Dated intervals (interevent only)

`pet_depth_mm(demand, seconds)` is the uniform-rate depth over
0 < seconds ≤ 86400 within the demand's date (int, float or
`fractions.Fraction`).

`integrate_interevent_pet_chunks(chunks, records, phase, *, keep_pieces=0,
on_chunk=None)` is the primary interface for a scheduler: it takes the
scheduler's EXACT declared `(date, seconds)` chunks (Fraction seconds are
divided exactly and rounded to FP64 once; several chunks of one date are
allowed; dates must be `datetime.date`), looks up each date's
`PotentialDemand` (mapping or callable; a missing date refuses, never
zero-fills), and accumulates with O(N) resident arrays plus BOUNDED
diagnostics. By default no history is kept: the result carries the summed
depth (float, or one fresh caller-owned per-cell array), the scalar reference
total, total seconds, chunk count, first and last date, the first terrain
identity and the number of identity changes between consecutive chunks.
`keep_pieces=k` retains at most the first k `IntervalPiece` records (date,
seconds, scalar reference depth) and sets `pieces_truncated`; `on_chunk`
receives each piece as it is integrated so the caller decides what to keep.
All terrain-applied records of one integration must share shape and backend
(no broadcasting changing the domain); mixing scalar and terrain-applied
records refuses. A test streams 400 chunks over 100 000 cells through a
callable and checks that the traced peak stays within a few cell-arrays.

`integrate_interevent_pet(start, end, records, phase, ...)` is a convenience
wrapper that splits a NAIVE `[start, end)` datetime interval at midnight into
exact microsecond chunks. Offset-aware instants are refused; an adapter with an
offset-aware or rational-subsecond calendar (the MASY calendar is one) should
map its own exact durations to chunks instead of converting to naive datetimes.

`phase` must be `"interevent"`: `"wind"`, `"rain"` and `"recession"` refuse
with `WeatherPhaseError` before any record is looked up, so no demand is spent
and nothing is caught up later. Hargreaves estimates PET from daily extrema, so
there is no invented sub-daily diurnal curve; the fractional day uses the
declared uniform daily rate as the initial approximation. This is a mechanical
gate for a scheduler to call; the module does not decide what phase an interval
is.

## Tests

`tests/weather_pet/test_weather_pet.py` (CPU) checks: Gregorian/leap day of
year; latitude validation (None, Boolean, NaN, out of range, string) and the
MAPLE bridge; Ra against an independent transcription of the source constants
at 8 latitudes × 5 dates; FAO-56 Examples 8 and 20 at the documented tolerance
and the explicit non-identity with FAO-56; equator equinox; polar night zero and
polar day finite positive at 80° and ±90°; signed latitude; cold clamp and
zero-range; temperature/RH/VPD/radiation/provenance validation, the FAO-56
Tetens oracle, the extrema requirement for humidity and the declared humidity
tolerance; the full weather record in provenance; explicit PET with mean
temperature only, without latitude, and via the constructor helper; Hargreaves
refusing missing extrema; unsupported method refusal; literal northern table
and southern reflection, hemisphere contrast on slopes ≥ 5°; label parsing;
NumPy terrain factor against a Python loop oracle at +37°, 0°, −37° for the
5° default, a 2° threshold and caller-classified mode, with caller arrays
untouched and identity hash; the source flat classification cases (1° north,
level south, 4.999°, exactly 5°, steep); identity for flat level cells; sealed
payloads (read-only views refuse positive and negative writes and cannot be
re-enabled, direct construction, attribute assignment and foreign payloads via
`dataclasses.replace` refuse, detached copies are independent, the recorded
hash equals the multiplied payload, `verify_identity` detects private
tampering); single application and refusal of double adjustment and latitude
mismatch; depth with Fraction seconds; midnight-splitting integration with
microseconds; the exact chunks interface and its `keep_pieces`/`on_chunk`
validation; non-interevent refusal before any lookup; record consistency
including shape mismatch across dates; bounded integration with a second
terrain identity counted; a 400-chunk × 100 000-cell callable stream with
tracemalloc; subprocess check that importing the module loads neither CuPy nor
`maple`; CUDA refusal when unavailable.

Correction 02 added: the sealed scalar demand (replacement, direct
construction, attribute assignment and hashing refuse; equality and repr;
detached provenance) and the overflow refusals (two-day sum of 1e308, terrain
product at 1.7e308, per-cell sum where the scalar sum stays finite, Hargreaves
at absurd temperatures) with the cold clamp confirmed as the only clipping.

`tests/weather_pet/test_weather_pet_cuda.py` uses a `gpu` fixture that skips
(no claim) without CuPy or a visible device and otherwise checks the CuPy
terrain (both flat policies), detached-copy accessors, apply/depth/integration
paths against the NumPy reference, a 200-chunk device stream with pool usage,
the overflow guards on device, and the namespace-mismatch, raw-array and
replacement refusals. Device/host equivalence of the terrain factor is
asserted within an explicit 2 ULP bound of the strictly positive reference
(`np.spacing`): CUDA's and libm's `cos` are each within 1 ULP but need not
round identically, and the table multiply adds one rounding; Codex's GPU 0 run
of correction 01 observed exactly 1 ULP on one cell of 91. This is a
floating-point equivalence bound, not a physical or MAPLE water-budget
tolerance. SHA identity equality is asserted only when the bytes are
identical, and inequality when they are not. The author did not execute these
tests on a GPU; see the task reports for the recorded runs.

Run from the candidate root:

```
PYTHONPATH=src python -m pytest -c /dev/null --rootdir . --confcutdir . tests/weather_pet
```

## What this is not

Not acceptance of a coupled simulation, an ET partition, a weather-station fit,
Phase 1 hydrology, or a selected runtime weather schema. No calibration or
validation against observations is claimed. The reference comparisons are
equation-level checks against a transcription of the downloaded source and the
published FAO-56 examples, not a run of the EcoHyD executable. GPU behaviour is
pending Codex's run; a provenance dictionary is a metadata snapshot, not a
native checkpoint acceptance.
