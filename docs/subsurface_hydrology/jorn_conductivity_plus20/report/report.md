# What happens if every conductivity is 20% larger? JORN, January-June 2024

*A prescribed sensitivity of the native SYRUP column: every K and Ks raised by exactly 1.2, everything else identical to the baseline 160-day run. This is not a recalibration.*

## 1. Summary

- **Small, mixed change.** The pooled error against 282 sensor values after one fixed start-up offset falls from 1.049 to 1.004 percentage points (4.3% lower RMSE). The no-change reference is 1.799.
- **All rain still enters the soil.** The matrix infiltration K rises from 38.2 to 45.9 mm/h, but peak rain is only 10.034 mm/h and the soil never lacks pore room, so intake (25.088 mm) is identical. There is no runoff, ponding, percolation or drainage in either run. The change acts only through faster exchange between layers and, indirectly, evaporation.
- **The surface pulse is not fixed.** On Jan 23 the 0-10 cm layer rises 9.06 points in the baseline and 9.04 with +20%; the 5.8 cm sensor rose 3.06.
- **Middle sensor:** aligned RMSE improves (0.506 to 0.397) but RAW RMSE gets worse (0.534 to 0.580). Both views are shown and explained below.
- **Shallow and deep sensors** remain worse than assuming no change (1.80 vs 0.88; 1.00 vs 0.89). The result does not show that larger K values are physically better.

## 2. What was changed

The baseline is the frozen JORN run described in the original report [7]: one 1 m² column of three 10 cm layers, NEON rain [5] and Hargreaves evaporation demand from NEON air temperature [6] (the native EcoHyD-style family of Hargreaves and Samani [3], not an exact FAO-56 implementation), initial water from the NEON soil sensors [4], diffusion coefficient d = 0.0010239461376024856, evaporation factor 1, stress value 0.2017, transpiration zero, open bottom, gravity on, matrix-only infiltration, no plants, wind, sediment or routing. The soil properties are the approved Saxton-Rawls [2] surrogates (the density and air-entry floors are this project's choices, not published recommendations). The Tietjen et al. [1] gradient exchange scales with d x K, so a 20% larger K acts like a 20% larger d for the layers it joins.

Root scaled ten conductivity entries once by 1.2 and nothing else. I re-compared the two sealed configurations: exactly ten numeric entries differ, each by a ratio of 1.2 (three layer Ks records in mm/h and m/s, the top matrix Ks, the matrix K, and both interface K); the retention values, suction (0.166 m), the local-K factor (0.31635738822500753), d, E, stress, initial water, forcing, clocks and source pins are identical. The physics reads only the matrix K and the two interface K. The configuration check, however, also reads the top matrix Ks and requires matrix K = local factor x top Ks, so the Ks and K entries had to be scaled together; the run would be refused if only one were changed. The layer Ks records are source estimates (user-perturbed here, not new Saxton-Rawls predictions) that act through those derived K values, so there is no second or double 1.2 multiplier; the effective K of layers 2 and 3 are derived for display only. No retention curve was recomputed. Field capacity (0.395-0.399) and saturation are unchanged, and modelled water stays far below field capacity (highest 0.153 in the +20% run), so gravity flow and open-bottom outflow are zero in both runs.

**Table 1. Conductivities (mm/h): source estimates (Ks), values derived from them, and the direct physical inputs of the run. Ks and derived K are scaled together by the same 1.2, once.**

| Quantity | Baseline (mm/h) | +20% (mm/h) | Used by the native run as |
|---|---|---|---|
| Ks, layer 0-10 cm | 120.79 | 144.94 | source estimate; represented by the derived K |
| Ks, layer 10-20 cm | 119.71 | 143.65 | source estimate; represented by the derived K |
| Ks, layer 20-30 cm | 104.33 | 125.20 | source estimate; represented by the derived K |
| Effective K, layer 0-10 cm | 38.21 | 45.85 | derived from Ks; equals the matrix K |
| Effective K, layer 10-20 cm | 37.87 | 45.44 | derived from Ks; used to form interface K |
| Effective K, layer 20-30 cm | 33.01 | 39.61 | derived from Ks; used to form interface K |
| Matrix infiltration K (top layer) | 38.21 | 45.85 | direct physical input (intake law) |
| Interface K, 0-10 / 10-20 cm | 38.04 | 45.65 | direct physical input (exchange law) |
| Interface K, 10-20 / 20-30 cm | 35.27 | 42.33 | direct physical input (exchange law) |

**Clock.** As in the baseline, this is a daily-weather approximation: one 86,400 s inter-event step per nominal day plus the observed storm sub-steps (27.5 h in total, so the physical clock ends at 161.146 days against 160 nominal days). Evaporation and transpiration are off during rain. It is not an exact hourly field clock.

## 3. Water: where the 20% matters

**Table 2. Budget and layer-exchange results (baseline vs +20%).**

| Term (mm over 1 m²) | Baseline | +20% | Change |
|---|---|---|---|
| Initial storage | 22.9424 | 22.9424 | 0 |
| + Rain = intake | 25.088 | 25.088 | 0 |
| - Evaporation | 34.0455 | 34.4686 | +0.4231 |
| - Runoff / percolation / drainage | 0 | 0 | 0 |
| = Final storage | 13.9849 | 13.5618 | -0.4231 |
| Deep-layer peak gain (mm) | 0.5357 | 0.5891 | +0.0534 |
| Deep-layer peak (nominal day) | 2024-03-20 | 2024-03-16 | earlier |
| Deep layer, final (vol. %) | 6.678 | 6.442 | -0.236 |
| Jan 23 top-layer rise (points) | 9.060 | 9.042 | -0.018 |
| Jan 23 middle layer, 72 h (points) | 0.136 | 0.164 | +0.029 |

All 25.088 mm of rain infiltrates in both runs, so the larger matrix K cannot add water. Evaporation rises by 0.423 mm because faster upward exchange from the middle layer re-supplies the dry top layer (whole-run net upward top-middle exchange 3.84 to 4.31 mm), so final storage is 0.423 mm lower. In the deep layer, faster exchange both speeds and strengthens the early downward gain (peak 0.536 to 0.589 mm, reached on 03-20 in the baseline and 03-16 with +20%; nominal day, closing the next day at 07:00 UTC) and the later upward return, so the deep layer ends drier (6.68 to 6.44 vol. %). The same increase works in both directions: whole-run downward and upward middle-deep exchange are 0.536/0.880 mm (baseline) and 0.589/1.170 mm (+20%).

![Figure 1. RAW comparison at the nominal daily closes. Green: baseline; blue dashed: +20%; black open circles: sensor point values; triangles: day-0 values. The two model curves are almost indistinguishable at this scale; missing sensor values are left blank (no interpolation). The last shallow-sensor value used is at the 2024-03-15 07:00 UTC close.](figures/fig1_raw_overlay.png)

![Figure 2. New minus baseline water content, in percentage points (a change in volumetric water content, not a level). The largest absolute changes are about a quarter of a point in the middle and deep layers (maximum 0.232 and 0.236, deep the larger) and 0.071 in the top layer; shaded bands are the nominal days with rain.](figures/fig2_difference.png)

## 4. Comparison with the sensors

Two scorings are given. **RAW** compares model layer means directly with sensor point values. **Aligned** removes one constant start-up offset per depth, fixed at day 0, because a 10 cm layer mean differs from a point reading (offsets of +0.19, -0.59 and +0.44 points at 5.8, 15.8 and 25.8 cm). The offsets are unchanged by this study; model water is never altered by the alignment and nothing is re-aligned after storms.

**Table 3. Scores in percentage points (100 x m³/m³), baseline / +20%. No-change = holding the day-0 sensor value.**

| Depth | n | RAW RMSE base / +20% | Aligned RMSE base / +20% | No-change RMSE | RAW bias base / +20% | Aligned bias base / +20% |
|---|---|---|---|---|---|---|
| 5.8 cm sensor | 56 | 1.865 / 1.852 | 1.817 / 1.803 | 0.881 | 0.57 / 0.57 | 0.38 / 0.38 |
| 15.8 cm sensor | 147 | 0.534 / 0.580 | 0.506 / 0.397 | 2.343 | -0.32 / -0.45 | 0.27 / 0.14 |
| 25.8 cm sensor | 79 | 1.469 / 1.419 | 1.054 / 0.999 | 0.886 | 1.40 / 1.37 | 0.96 / 0.93 |
| All depths pooled | 282 | 1.202 / 1.192 | 1.049 / 1.004 | 1.799 | 0.34 / 0.26 | 0.48 / 0.41 |

The middle sensor shows why both views matter. The model's middle layer starts 0.59 points drier than the point reading; the +20% run lowers the middle layer by a further small amount through faster draw to the top, which removes some of the aligned error (bias +0.27 to +0.14) while moving the RAW bias from -0.32 to -0.45, so RAW RMSE rises. Neither is the more correct scoring; they answer different questions. At the shallow and deep sensors the aligned RMSE drops by only 0.014 and 0.055 points, and both stay worse than no change; the pooled gain comes mostly from the 147 middle values. There is no significance test, no R² and no claim that either set of K values is physically validated. The old d and stress values were fitted jointly to 53 dry-period sensor values from 2025 (JORN and MOAB); the 2024 data remain a test only, and nothing was refitted here.

![Figure 3. Jan 20 - Feb 4 around the 10 mm Jan 23 rain. Changes since day 0 with the fixed day-0 offset (no re-offset after the storm). Rain bars cover the nominal rainfall day; model states and sensor values are plotted at the daily close that follows that day. The 27.5 h of extra physical storm time remains the disclosed daily-weather approximation.](figures/fig3_rain_pulse_zoom.png)

Around the Jan 23 storm (Figure 3) the larger K barely changes the top-layer pulse (9.06 vs 9.04 points against 3.06 observed) and brings the middle-layer response only slightly closer to the observed 72 h rise of +1.06 points (+0.136 baseline, +0.164 with +20%, closer by 0.029); it remains far below the observation, so the larger K does not resolve the lag or the magnitude of the middle-layer response. Increasing every K by 20% therefore does not resolve the large surface pulse. Plausible reasons remain those of the baseline report: a 5.8 cm point versus an instantly mixed 10 cm layer mean, high surrogate field capacity that blocks gravity flow, no transpiration, a 30 cm column with no capillary rise or lateral flow, surrogate properties, and the d x K identifiability (a 20% larger K is similar to a 20% larger d).

## 5. Verification and run time

- NumPy and Numba results are exactly identical (all trajectories and fluxes); I confirmed the row-by-row equality here. A 900 s storm step changes the layer water content by at most 6.91e-06 m³/m³ (0.000691 points).
- Root's independent 90-digit calculation agrees to 4.92e-18 m (0.033 of the native one-step bound; 1.25e-17 m, 0.090 for the 900 s control). Native budgets and the source guards pass; closure residual 1.0e-17 m³ against a bound of 2.74e-13 m³.
- The input-only change did not rerun the earlier 77-test suite, and no new midpoint-restart check was made for this parameter set; the earlier restart qualification belongs to the original case. No GPU was used.

**Table 4. Measured seconds from separate fresh runs on a shared host (cold caches, other jobs running): not a controlled speed comparison.**

| Run | Integration (s) | Operator prep + JIT (s) | Driver call wall (s) | Full fresh command (s) |
|---|---|---|---|---|
| Baseline, Numba | 0.0398 | 4.891 | 6.286 | 7.446 |
| +20%, Numba | 0.0362 | 6.821 | 8.675 | 13.757 |
| +20%, NumPy | 0.1486 | 0.138 | - | 2.727 |
| +20%, Numba, 900 s storm step | 0.0486 | 4.433 | - | 7.006 |

Integration takes about 0.04 s in both runs; the fresh-command time is dominated by one-time compilation and the source-verified bootstrap, so it says nothing about model speed or the effect of the parameter change. One 1 m² column only: no scaling or coupled-model claim.

## 6. Limits

- A prescribed +20% sensitivity: no new calibration, pedotransfer prediction or default change; K, d and the stress value remain conditional, partly degenerate empirical settings.
- Rain was too light to test ponding, suction, gravity drainage or the matrix K as an intake limit.
- No transpiration or plants (a general limitation for spring), point sensors versus layer means, a daily clock, and the 030 vs 020 air-temperature height difference inherited from the baseline.

## References

[1] Tietjen, B., Zehe, E., & Jeltsch, F. (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418. https://doi.org/10.1029/2007WR006589
[2] Saxton, K. E., & Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569-1578. https://doi.org/10.2136/sssaj2005.0117
[3] Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. Applied Engineering in Agriculture 1, 96-99. https://doi.org/10.13031/2013.26773
[4] NEON (RELEASE-2026). Soil water content and water salinity, DP1.00094.001. https://data.neonscience.org/data-products/DP1.00094.001/RELEASE-2026
[5] NEON (RELEASE-2026). Precipitation, DP1.00045.001. https://data.neonscience.org/data-products/DP1.00045.001/RELEASE-2026
[6] NEON (RELEASE-2026). Single aspirated air temperature, DP1.00002.001. https://data.neonscience.org/data-products/DP1.00002.001/RELEASE-2026
[7] Original report of the baseline run: SYRUP on a real 160-day rain record, JORN, January-June 2024 (SYRUP_JORN_160day_rainfall_comparison_20261119.docx).

## Appendix. Identity (compact)

Sources (CPU, Root's source-verified bootstrap): SYRUP package 180 files, SHA-256 37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0; MAPLE package 412 files, SHA-256 fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e; ten frozen helper/test files unchanged. Original main 2591847. Baseline case SHA-256 e7513f545b074da5fc5627e96b79f62ed787863fb32c9df545a9422547e521ee; +20% case SHA-256 507aa743406b543330171cdb57f74c151fb76afc8e785e4aeb2f494f438ecd26.

Every sealed file used is listed with its SHA-256 in tables/provenance_seal.csv and artifact_manifest.json; the parameter differences are in tables/parameter_diff.csv; daily trajectories for both runs are in tables/daily_trajectories.csv. Raw NEON files are not copied.
