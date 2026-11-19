# JORN 160-day native SYRUP rainfall + subsurface experiment — protocol (stage 1)

Status: COMPLETE bounded field-column study. Claude implemented the analysis driver and report; Codex independently reviewed, executed and qualified the actual NumPy/Numba 160-day comparisons, timestep control and driver-level restart. See `qualification.md` and `report/SYRUP_JORN_160day_rainfall_comparison_20261119.docx`. Production physics and historical source guards are unchanged; no GPU or coupled-catchment acceptance is claimed.

## Case

160 fixed UTC-7 nominal days, 2024-01-18 to 2024-06-25, from the immutable forcing packet
`jorn_forcing_extension_20261119/preprocessed_160d/` (hashes pinned in the config). All 7680 rain half-hours are observed QA-pass
(25.088 mm, peak 10.034 mm/h); 36 air half-hours are interpolated (days with any are labelled `estimated` in the PET record; values are unchanged).
One 1 m² three-layer column, edges 0/0.1/0.2/0.3 m. FC, saturation, Ks and interface K are those of `selected_config.JORN4_A` in the
joint d/stress harmonic fit; d = 0.0010239461376024856, stress = 0.20170581117963332, E factor 1, T = 0, gravity on, open bottom.
These d/stress values are conditional empirical values, not measured soil properties. No refit. No wind, LEAF, sediment, bed, cover or splash.

## Daily-weather approximation (versioned profile `syrup.jorn160_wet.daily_weather_with_storm_substeps/1`)

Per nominal day k (this is NOT an exact chronological hourly field clock):

1. Wet day: a native `rainfall.RainfallSchedule` of the day's actual 30-minute piecewise rates from the first wet row to the end of the last wet row,
   all zero-rain rows in between kept. It is integrated with `max_dt_s` (1800 production, 900 control), split at schedule edges.
   Each sub-step: `matrix_infiltration.matrix_step` (pool = that sub-step's rain, standing = 0, matrix only, no preferential flow) credits intake once;
   the remaining pool is exported at once as runoff (zero-residence open-surface boundary); then `subsurface.step(dt, added = 0, E = 0)` applies native gravity/diffusion
   and advances the clock once. The matrix stage keeps the clock. Soil `surface_return` is exported once as well. No E, T or growth during rain.
2. Then ONE native 86400 s interevent step with the day's PET (`weather_pet`, `ecohyd_hargreaves`, flat terrain factor 1, no 1.2 canopy factor), E factor 1, T = 0.
3. Dry day: only step 2.

Storm durations are ADDITIONAL physical time. The nominal date index/close (`nominal_close_utc`) and the physical clock (`physical_t_s`, `added_storm_s`,
`added_rain_hours`, `physical_end_t_s`) are reported separately. Rain crossing local midnight is divided between the two nominal days it falls in; wetting history is carried across.
Wetting history J starts at 0 for the run and is never reset or recovered; lifetime matrix intake is a separate ledger. Closure uses
`conservation.volume_roundoff_bound_m3` with the sum of the native per-step term counts (plus the driver's own accumulations), per day and cumulative,
for the soil column and for rain = intake + runoff pool. There is no loose epsilon.

PET air-height disclosure: the packet air temperature is the 030 level; the original calibration used the 020 level. This is disclosed (`site.air_height_note`), the weather height and old fits are not changed.

## Matrix parameters

Matrix effective K = the retained local top conductivity 0.31635738822500753 × Ks_top (38.2117 mm/h; the config refuses any other value).
psi = 0.166 m and g = `reference_vegetation_function(0, a1, a2)` use the downloaded EcoHyD reference values; they are UNMEASURED reference inputs.
The actual peak rain (10.034 mm/h) is below K, so rain should be unponded; the driver records and verifies the actual ponded-step, onset, room-rejected and runoff counts
(`matrix_regime`). If the ponded law turns out active, flag a scientific limit (psi and g then matter).

## Initial state and targets

Initial state: the three QC-passing raw SWC 30-minute means whose record ENDS at 2024-01-18T07:00Z (raw row START 06:30): 0.0711, 0.0921, 0.0658 at the corrected depths 0.058/0.158/0.258 m
(depth table pin; time-valid for JORN HOR 004 through 2024). The earlier 57-day screen listed the START-labelled 07:00 values (0.0709, 0.0921, 0.0659): a support-label error,
never an IC (the old receipt is not edited). The `integrate_initial_profile` layer integral is used; the state must be below saturation.
Each forecast day closes at nominal 07:00 UTC and is matched to the 30-minute record ENDING there. Observations after initialisation never force, stop or reset the model: all 160 days run even if sensors fail.
Missing or failed targets are `null` (no interpolation). Scoring (post hoc, no fit): RAW (model layer mean − point observation) and fixed t0-offset-aligned changes
((model_k − model_0) − (obs_k − obs_0)), with persistence nulls; no re-offset after storms. Point observations vs layer means is a representativeness difference, not hidden.

## Driver / CLI

`python -m benchmarks.subsurface_wet_validation.driver --config CONFIG [--out DIR] [--execute] [--stop-after-day N --export-state P | --resume-state P]`

Plan only is the default. Backends are `numpy` or `numba`, explicit, no fallback; no GPU. Everything (config schema, file hashes, forcing, masks, controls, depths, IC,
expected source pair) is validated before the first native step; the source pair is checked before and after. Outputs are immutable (`result.json`, deterministic
identical retry accepted; `timing.json` separate). Timing (`time.perf_counter` raw seconds): imports + input preparation, operator prepare, JIT warm-up on a DISPOSABLE control state
(both kernels, ponded regime), the actual integration and the output write, with step counts, backend and environment. No per-step arrays are kept.
Midpoint export/import is a DRIVER-level reconstruction (column water, clock, J, lifetime intake, ledgers, nominal day, bindings, hash-sealed); it is not a production checkpoint
qualification. Resume with a changed config/backend/source/input, a tampered state or a duplicate/skipped ordinal is refused.

## Preferential-flow question (diagnosis only; no parameters invented or fitted)

Preferential flow is OFF. Whether it could explain deeper-layer (0.158/0.258 m) discrepancies should be examined only from quantitative observed patterns, e.g. an observed deeper-layer rise
within the storm day or next close that exceeds the matrix model's day-close value after a rain event while the top layer behaves as modelled, with rise size versus rain depth.
Confounds to state with any such diagnosis: the initial point-to-layer offset, sensor representativeness and depth, interpolated air/PET days, d and stress being conditional empirical values,
the approximate daily clock (storm physical time added; no E during rain), and unmeasured psi/g if ponded steps exist. Any later preferential-flow run needs a separate user-approved parameter source.

## Citations

Tietjen, Zehe & Jeltsch (2009), doi:10.1029/2007WR006589; Saxton & Rawls (2006), doi:10.2136/sssaj2005.0117; Hargreaves & Samani (1985), doi:10.13031/2013.26773; NEON RELEASE-2026 data products.

## Tests

`tests/subsurface_wet_validation/` (77 tests, including a PLAN-only import of the actual Root case files): synthetic inputs in the pinned formats, real native physics. They need the Root runtime prefix; no pin is relaxed.
They qualify the driver composition only, not field behaviour.

## Correction01 contract notes

- `daily_air.total_weight` is the sample-count aggregation weight of the 48 half hours (may be < 86400), not elapsed time. Each day is re-derived from its 48 pinned `derived_30min` rows and the receipt; counts must be nonnegative integers summing to 48.
- Closure bounds use the true operand maximum (zero allowed); there is no artificial scale floor.
- PRIMARY overall scores pool every valid pair of all depths (equal weight per observation, as the pooled calibration); per-depth scores are retained; `equal_depth_rmse_*` is a separately labelled optional metric.
- Observations use a narrow tolerant reader: a missing row, unparsable value or invalid flag is a failed (null) target; schema, interval duration, order and duplicates stay strict. The initial record must still pass for all three depths.
- Resume validates every candidate (seal, bindings, rows vs current inputs, water/clock/history vs the last row, initial storage, ledgers folded from rows) before assigning anything. Driver-level contract validation only: no physics replay, no signature claim.
- Timing: `source_checks_and_input_preparation_s` starts inside `execute` after module imports (interpreter start and imports excluded; Root measures fresh-process wall time). `*_this_process` counters are what this invocation integrated; `*_cumulative` include resumed days.
