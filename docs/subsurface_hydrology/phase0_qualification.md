# Phase 0 scheduler and interevent foundation qualification

2026-11-10. This publishes a bounded foundation for the shared scheduler and
subsurface program. It does **not** complete the continuous MAPLE–SYRUP–LEAF
simulation or qualify persistent rainfall-to-soil handoffs. Wind, rain/recession
and interevent periods never overlap. Evaporation, transpiration and growth
run only during interevents; calendar time advances throughout without catch-up.

## Delivered and tested

- MASY: dated reusable wind/rain/interevent files, ordered scenarios, exact
  native/global/calendar clocks, event gates, capability checks, source-bound
  weather delegation and replay-validated temporal cursor. 223 independent
  tests and 217 subtests passed against actual adopted SYRUP weather.
- SYRUP: the existing qualified configurable-layer column, conservative
  redistribution/drainage/evaporation and withdrawal accounting, plus the new
  shared weather, explicit PET and EcoHyD Hargreaves calculations. NumPy,
  Numba and CUDA use explicit backends; missing devices never silently fall back.
- Shared weather/PET: 123 CPU tests passed, 10 CUDA tests skipped on CPU;
  hardware GPU0 run passed 132 tests with one no-device-only test skipped.
  Source-compatible flat/aspect coefficients and the user-approved Southern
  Hemisphere reflection are recorded; no station validation is claimed.
- Native interevent driver: 67/67 checks passed on each of NumPy, Numba and
  CUDA. Independent direct-native controls match scheduled results. Final
  water arrays, times and scalar budgets match bit-for-bit across backends.
  Two scenarios exercise explicit PET and Hargreaves, leap-day midnight,
  repeated events without soil reset, positive evaporation/unmet demand,
  redistribution/drainage and unchanged inactive cells. Failure injections
  preserve accepted native state, cursor and ledger.

The 3-layer, 4-cell fixture starts with 3.369903564453125 m3 soil water.
Explicit-PET final storage is 2.8311694407776296 m3; Hargreaves final storage
is 2.846694737606726 m3. Drainage is 0.439453125 m3 in each. Aggregate
water residuals are -2.22e-16 and 4.44e-16 m3, below the existing MAPLE-derived
8.83e-13 m3 bound for 295 terms. No conservation tolerance was relaxed.
These tiny cases do not establish performance or large-domain memory fit.

## Exact development sources

| Component | Identity |
|---|---|
| SYRUP branch base | `deb60c29834a4aad0bf8cb4cfbb611fc28ebe090`, plus explicitly published new files and preserved unrelated dirty work |
| MASY base | `68936bfae87db5243912ea5fc69195c113883bd0`, plus published additive scheduler files |
| Accepted `masy.coordinator` | `338e31e5f478225428f6410ca689a9cd3baf43b9631f1425936407ab4469d009` |
| Accepted `maple_syrup.weather_pet` | `abd04baed539c0d64abffe40c4fd1f47bbe1092884e4fdbd606831cfd82d9169` |
| Existing `maple_syrup.subsurface` | `5e77f14df8af4e473e8a22278700b519ccb2888c722883a9f7da508abdf14ae9` |
| Immutable MAPLE backend | `39bcf79e1e67b6ed960f78a923ddfedf1d173e68`, bundle `37a1bc69bfa034be7cd512d8aa7c30d33682b7c053a53c3b0c91edc34d59a96f` |

The source overlay contains actual same-package copies, not fake native classes.
These are declared development identities; no historical production lock was
rebound or provenance guard bypassed. Live MAPLE `6e457702…` had unrelated
staged/unmerged work and was not adopted. Existing low-level regression
superset passed 317 CPU /350 GPU-enabled tests (35/2 skips); publication-scoped
CPU tests excluding observational XLSX and native case-compiler lanes passed
290/35. The case-compiler lane correctly refused an immutable source path with
metadata identifying a different editable install. It requires its own genuine
locked-runtime setup and is not accepted by this kernel lane.

Native success checks initially used driver `2d454f21…`; final `b89ff311…`
differs only in lint comments and setup-OSError reporting. The final driver
was independently rerun on CPU (67 passed) and its new setup refusal tested.
All numerical paths are unchanged. Raw evidence, author snapshots, failed
attempts and exact module manifests are archived in SYRUP agent_handoffs/tasks/
shared_scheduler_phase0_20261109, shared_weather_pet_phase0_20261110 and
phase0_native_interevent_20261110.

## Still open

Persistent native rain integration belongs with Phase 1's infiltration changes:
current native rain/pond completion resets soil water. The new mode needs one
authoritative layered inventory inside accepted hydraulic timesteps, conservative
surface return/drainage and a newly qualified persistent handoff.

The native interevent helper stages pure column outputs and the test driver
publishes them after temporal acceptance. MASY's coordinator has no combined
native commit hook. Its cursor is metadata, not a soil/bed/plant checkpoint.
Production native bridges, LEAF calendar/growth integration, a new source-locked
three-model runtime and combined physical restart remain unqualified. Evolving
soil remapping/texture-derived hydraulic properties are Phase 2; observational
JORN/MOAB validation is Phase 3. Existing benchmark defaults stay unchanged.
