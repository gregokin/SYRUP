# Where does the 40% field capacity come from? A worked audit for the JORN soil layers

*Independent recomputation of the Saxton-Rawls inputs and arithmetic behind the field capacity used in the JORN 160-day runs. Arithmetic correctness is separated from whether the value is a credible site field capacity.*

## 1. Short conclusion

- **The arithmetic is correct.** I recomputed every number from the raw NEON files in 60-digit arithmetic with my own transcription of the equations. The results match Root's 80-digit audit, the archived receipt and the field capacities in the actual original run file to better than 1e-45 (to 2e-13 for the run file, which stores doubles).
- **Two different pressures are involved.** At the standard 33 kPa suction of Saxton and Rawls [1] the three layers hold 9.75, 9.94 and 10.91 vol. % of water, about 10-11%; the first two values are slightly below 10%, close to the user's illustrative 10-15 vol. % field-capacity range (an expectation for sandy soil, not a measurement and not an exact match). SYRUP uses the water content at 6 kPa, following the field-capacity pressure of Tietjen et al. [2], and the Saxton-Rawls wet branch gives 39.76, 39.87 and 39.51 vol. % there.
- **The 6 kPa value is our hybrid choice, not a measured or published site value.** Tietjen et al. take field capacity from similar-site field values, not from Saxton-Rawls equations. The Saxton-Rawls wet branch is a straight line between 33 kPa and saturation; with the air entry floored at zero, 6 kPa lies 81.8% of the way to saturation. No measured 6 kPa retention value was used in, or found among, the field inputs I audited.
- **The value is therefore arithmetically right but scientifically unsuitable without further qualification.** Particle-size data alone cannot validate the wet end of the curve. I do not defend 40% as a real field capacity for this sandy soil, and nothing here is adopted or rerun.

## 2. Inputs actually used

The source is the NEON Jornada megapit (pit 1, sampled 2013-03) [3], two horizons: A, 0-18 cm, and Bw, 18-46 cm. They are spatial proxies for the 2024 sensor plot, not proven co-located measurements. Layer 1 uses the A sample, layer 3 the Bw sample, and layer 2 (10-20 cm) mixes 8 cm of A with 2 cm of Bw. The mixing is done on the raw measured properties, before any floors: the texture and carbon are weighted by dry mass (A weight 0.8627 = 0.8 × ρA / (0.8 × ρA + 0.2 × ρBw)) and the density is the thickness-weighted mean.

Table 1 gives the complete particle-size distribution that NEON measured: five sand classes plus silt and clay, as percentages of the mass of the fine earth (material finer than 2 mm), not as volume percentages and not as the MAPLE grain classes. Fine and very fine sand dominate (52.2% in the A sample, 50.9% in Bw), but roughly 40% of the sand (34.5 of 86.7 points in A, 32.8 of 83.7 in Bw) is medium to very coarse. The Saxton-Rawls equations, however, use only the totals of sand, silt and clay; the sand sub-classes do not enter them.

**Table 1. Particle-size distribution, % of fine-earth mass. Layers 1 and 3 equal the measured NEON A and Bw samples; layer 2 is a derived mixture (the raw A and Bw columns are in tables/psd_full.csv).**

| Size fraction (% of the fine-earth mass) | Layer 1, 0-10 cm (= measured A sample) | Layer 2, 10-20 cm (A-Bw mixture) | Layer 3, 20-30 cm (= measured Bw sample) |
|---|---|---|---|
| Very fine sand, 0.047-0.105 mm | 11.300 | 11.176 | 10.400 |
| Fine sand, 0.105-0.25 mm | 40.900 | 40.845 | 40.500 |
| Medium sand, 0.25-0.5 mm | 27.100 | 26.935 | 25.900 |
| Coarse sand, 0.5-1 mm | 6.700 | 6.590 | 5.900 |
| Very coarse sand, 1-2 mm | 0.700 | 0.741 | 1.000 |
| Total sand, 0.047-2 mm | 86.700 | 86.288 | 83.700 |
| Silt, 0.002-0.047 mm | 6.200 | 6.433 | 7.900 |
| Clay, below 0.002 mm | 7.100 | 7.278 | 8.400 |
| Sum of all fractions | 100.000 | 100.000 | 100.000 |
| Very fine + fine sand | 52.200 | 52.022 | 50.900 |

Table 2 lists the other inputs. Organic matter is not measured: NEON reports an estimated organic carbon of 2.0 g/kg (A) and 1.0 g/kg (Bw), converted here with OM = 1.724 × OC / 10 per cent (a provisional conversion). The measured bulk densities are 1.4196 g/cm³ (A) and 0.9035 g/cm³ (Bw). The Bw value is far below the range the equations were built for (density factor 0.57 against 0.9-1.3), so the study floors the density at max(raw, 1.000, 0.901 × normal density), applied after mixing; this raises layer 3 to 1.4346 g/cm³. The metadata of the A density sample is inconsistent (sample ID 0-10 cm and centre 5 cm, but a recorded bottom depth of 110 cm) and the Bw sample comes from 23-33 cm; I preserved both as recorded and did not correct them. The bulk density is NEON's value excluding coarse fragments.

**Table 2. Other inputs (carbon, organic matter, density) and their provenance.**

| Input | Layer 1 | Layer 2 (derived) | Layer 3 |
|---|---|---|---|
| Source | A horizon sample | 0.8627 A + 0.1373 Bw by mass | Bw horizon sample |
| A-horizon mass weight in the mix | 1 | 0.862730 | 0 |
| Estimated organic carbon (g/kg) | 2.0000 | 1.8627 | 1.0000 |
| Organic matter OM = 1.724 x OC / 10 (%) | 0.3448 | 0.3211 | 0.1724 |
| Raw bulk density (measured or mixed) (g/cm³) | 1.4196 | 1.31638 (0.8 A + 0.2 Bw) | 0.9035 |
| Density used after floors (g/cm³) | 1.4196 | 1.4173 | 1.4346 |
| Normal density from Eq. 6 (g/cm³) | 1.5698 | 1.5730 | 1.5923 |
| Density sample metadata | ID 0-10 cm; centre 5.0 cm; bottom depth 110.0 cm | mixture | ID 23-33 cm; centre 28.0 cm |

## 3. Worked calculation for the top layer

The equations are Table 1 of Saxton and Rawls [1]; S and C are the sand and clay fractions (0-1) and OM is organic matter in per cent. Suction ψ is a positive number here: a suction of 33 kPa is a matric potential of -33 kPa. The steps below reproduce layer 1 (S = 0.867, C = 0.071, OM = 0.3448%, raw density 1.4196 g/cm³); every term of the three long sums is listed on its own line, rounded to 10 decimals, so the result can be rebuilt with a calculator. Displayed intermediates are rounded for reading; every later result is computed from the full-precision values and shown to six significant figures. Water contents are fractions (multiply by 100 for vol. %).

**Step 1. Eq. 2 first solution**

    -0.251 S + 0.195 C + 0.011 OM + 0.006 S OM - 0.027 C OM + 0.452 S C + 0.299
    -0.251 S = -0.2176170000
    +0.195 C = +0.0138450000
    +0.011 OM = +0.0037928000
    +0.006 S OM = +0.0017936496
    -0.027 C OM = -0.0006609816
    +0.452 S C = +0.0278237640
    +0.299 = +0.2990000000
    sum of the seven terms = 0.127977 (the printed terms are rounded to 10 decimals; the sum is the full-precision result)

*Result:* 0.127977

**Step 2. Eq. 2 normal-density 33 kPa**

    θ₃₃N = t + (1.283 t² - 0.374 t - 0.015)
    0.127977 + (1.283 x 0.016378 - 0.374 x 0.127977 - 0.015)

*Result:* 0.0861269

**Step 3. Eq. 3 first solution**

    0.278 S + 0.034 C + 0.022 OM - 0.018 S OM - 0.027 C OM - 0.584 S C + 0.078
    +0.278 S = +0.2410260000
    +0.034 C = +0.0024140000
    +0.022 OM = +0.0075856000
    -0.018 S OM = -0.0053809488
    -0.027 C OM = -0.0006609816
    -0.584 S C = -0.0359492880
    +0.078 = +0.0780000000
    sum of the seven terms = 0.287034 (the printed terms are rounded to 10 decimals; the sum is the full-precision result)

*Result:* 0.287034

**Step 4. Eq. 3 gap θ(S-33)**

    gap = t + (0.636 t - 0.107)
    0.287034 + (0.636 x 0.287034 - 0.107)

*Result:* 0.362588

**Step 5. Eq. 5 normal saturation**

    θSN = θ₃₃N + gap - 0.097 S + 0.043
    0.0861269 + 0.362588 - 0.097 x 0.8670 + 0.043

*Result:* 0.407616

**Step 6. Eq. 6 normal density**

    ρN = (1 - θSN) x 2.65
    (1 - 0.407616) x 2.65

*Result:* 1.56982 g/cm³

**Step 7. Project density floors**

    ρ = max(raw, 1.000, 0.901 ρN)
    max(1.4196, 1.000, 1.414405)

*Result:* 1.4196 g/cm³

**Step 8. Eq. 8 saturation**

    θS = 1 - ρ / 2.65
    1 - 1.4196 / 2.65

*Result:* 0.464302

**Step 9. Eq. 9 field capacity at 33 kPa**

    θ₃₃ = θ₃₃N - 0.2 (θSN - θS)
    0.0861269 - 0.2 x (0.407616 - 0.464302)

*Result:* 0.0974641

**Step 10. Eq. 4 first solution**

    -21.67 S - 27.93 C - 81.97 gap + 71.12 S gap + 8.29 C gap + 14.05 S C + 27.16
    -21.67 S = -18.7878900000
    -27.93 C = -1.9830300000
    -81.97 gap = -29.7213587130
    +71.12 S gap = +22.3575684818
    +8.29 C gap = +0.2134158171
    +14.05 S C = +0.8648758500
    +27.16 = +27.1600000000
    sum of the seven terms = 0.103581 (the printed terms are rounded to 10 decimals; the sum is the full-precision result)

*Result:* 0.103581

**Step 11. Eq. 4 air entry (kPa)**

    ψe = t + (0.02 t² - 0.113 t - 0.70)
    0.103581 + (0.02 x 0.010729 - 0.113 x 0.103581 - 0.70)

*Result:* -0.607909 kPa

**Step 12. Project floor on air entry**

    ψe,used = max(0, ψe)
    max(0, -0.607909)

*Result:* 0 kPa

**Step 13. Eq. 12 at 6 kPa (linear wet branch)**

    θ₆ = θ₃₃ + (33 - 6)(θS - θ₃₃) / (33 - ψe,used)
    0.0974641 + 27 x (0.464302 - 0.0974641) / 33

*Result:* 0.397604

*Worked calculation for layer 1 (six significant figures; the same steps are in tables/steps.csv and full precision in tables/steps_layer1.csv).*

Reading the result. The regression for field capacity at 33 kPa gives 9.75 vol. %. Saxton and Rawls define field capacity as the water content at 33 kPa (their Table 2) [1]. Our step 13 evaluates the same printed wet branch at 6 kPa, an interpolation between the 33 kPa value and saturation. Because the study floors the air entry (a raw value of -0.608 kPa is negative and the provider refuses negative air entry), the linear branch runs from θ₃₃ at 33 kPa straight to saturation at 0 kPa, and the point at 6 kPa is 27/33 = 81.8% of the way up: θ₆ = 9.75 + 0.818 × (46.43 - 9.75) = 39.76 vol. %.

## 4. Results for all three layers

**Table 3. Computed hydraulic values (vol. % unless stated). θ₆ is the field capacity written into the original run file.**

| Quantity | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|
| θ₃₃ at normal density (vol. %) | 8.61 | 8.76 | 9.72 |
| θ₃₃ used, density-adjusted (vol. %) | 9.75 | 9.94 | 10.91 |
| θS saturation (vol. %) | 46.43 | 46.52 | 45.86 |
| Air entry, raw (kPa) | -0.608 | -0.591 | -0.461 |
| Air entry used after the zero floor (kPa) | 0 | 0 | 0 |
| θ₆ = SYRUP field capacity (vol. %) | 39.76 | 39.87 | 39.51 |
| Share of the way from θ₃₃ to θS | 81.82% | 81.82% | 81.82% |
| Water added above θ₃₃ (mm per 10 cm) | 30.01 | 29.93 | 28.60 |

![Figure 1. The computed Saxton-Rawls wet-end curves for the three layers (lines) with the values at 33 kPa, 6 kPa and 0 kPa (dots). These are calculated, not observed. The green band is the user's expected field-capacity range of 10-15 vol. %, shown for orientation only (illustrative, not a measurement); the 33 kPa values lie just below or inside its lower edge.](figures/fig1_wet_curves.png)

The check against published numbers is reassuring about the arithmetic but not about the 6 kPa choice: recomputing the published sandy examples of Saxton and Rawls (their Table 3, OM 2.5%, normal density) with my code gives a 33 kPa field capacity of 10.28 and 12.02 vol. %, in line with their listed values of about 10 and 12. The pedotransfer regression itself is uncertain: the paper's Table 1 lists R² 0.63 and a standard error of 0.052 for the 33 kPa water content (read from the text extraction of the local PDF copy), a large error for a value near 0.10.

## 5. Why the value is so high and what it implies

- **Pressure and curve shape dominate.** The step from 33 kPa to 6 kPa is an interpolation along the printed Eq. 12 wet branch, a straight line between the 33 kPa value and saturation, and it adds 30.0/29.9/28.6 mm (layers 1/2/3) of water per 10 cm layer above the 33 kPa value. Saxton and Rawls themselves describe this linear branch as a practical substitute for the often poorly represented low-tension range, and their power-law Eq. 11 applies only from 33 to 1500 kPa; extending Eq. 11 below 33 kPa would be an extrapolation, which I did not do, and I do not offer it as a validated replacement.
- **Particle size alone cannot validate the wet end.** The size classes constrain the texture class but do not by themselves fix the water held at 6 kPa. No measured 6 kPa retention value was used in, or found among, the field inputs I audited (the NEON saturated-extract water fields are mass-based extract measurements, not volumetric water at 6 kPa).
- **The air-entry and density choices are secondary.** Table 4 shows that the zero floor on the air entry changes θ₆ by about 0.5 points (the comparison uses a negative, non-physical air entry and is not a proposed input). Substituting the normal density for the density used, with the air entry kept at zero, lowers θ₆ by about 5 points and still leaves about 35% (layer 1 uses its measured density, so no floor is active there). The curve shape and the pressure choice explain almost all of the difference from 10 vol. %.

**Table 4. Sensitivity of θ₆ to the air-entry and density choices (diagnostics only, not adopted).**

| Quantity | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|
| θ₆ as used (vol. %) | 39.76 | 39.87 | 39.51 |
| Diagnostic A: θ₆ with the raw NEGATIVE air entry (non-physical, not a proposed input) | 39.22 | 39.34 | 39.11 |
| Difference A - used (points) | -0.54 | -0.53 | -0.39 |
| Raw bulk density (measured or mixed) (g/cm³) | 1.4196 | 1.3164 | 0.9035 |
| Density used (g/cm³) | 1.4196 | 1.4173 | 1.4346 |
| Is a density floor active? | no | yes | yes |
| Diagnostic B: θ₆ with the normal density substituted for the density used (air entry kept at 0) | 34.92 | 34.84 | 34.43 |
| Difference B - used (points) | -4.84 | -5.02 | -5.08 |

**Implications for the model runs.** The original 160-day case file contains exactly these field capacities, and the conductivity multipliers (1.2, 2, 3, 4) left them unchanged. In the native model gravity drainage acts only on water above field capacity, so it could not start while the modelled water content stayed far below about 39.5-39.9 vol. %, even at four times the conductivity; water still moved between layers by the moisture-gradient exchange, so the soil was not static. Table 5 is an explicitly hypothetical comparison, not a run: a 10 cm layer holding 15 vol. % (15 mm) would be above the 33 kPa field capacity by about 5 mm, but far below the SYRUP value. If the field capacity were changed later, the diffusion coefficient and evaporation-stress values that were fitted under the present one would have to be reconsidered; the old results should be kept, not rewritten.

**Table 5. Hypothetical storage comparison for layer 1 (illustration only).**

| Hypothetical 10 cm layer 1 (not a model run) | Volumetric water (vol. %) | Stored water (mm) |
|---|---|---|
| Water at S&R field capacity θ₃₃ | 9.75 | 9.75 |
| An example wetted layer | 15.00 | 15.00 |
| Water at SYRUP field capacity θ₆ | 39.76 | 39.76 |
| Example minus θ₃₃ (would exceed 33 kPa field capacity) | 5.25 | 5.25 |
| θ₆ minus example (still below SYRUP field capacity) | 24.76 | 24.76 |

## 6. Suggested later choices (not adopted)

- Offer the 33 kPa value, which is the Saxton-Rawls definition and lies close to the user's expected range, as a transparent sensitivity or control option for the field capacity of the pedotransfer-derived layers. It is not validated at this site and not shown to be right at 6 kPa or for sandy soil in general; the existing d and stress fit is conditional on the present value and would have to be redone under it.
- Keep a 6 kPa pressure only together with a qualified site retention curve or site-specific field capacity values (as Tietjen et al. use), and state that source explicitly; a qualified site curve or a field drainage assessment can then choose the operational field capacity.
- Estimate a site field capacity from the sensors themselves (for example the water content where drainage becomes slow after a wetting event) and label it as a site estimate, not a measurement of a pressure.
- Treat the density and air-entry choices as separate, smaller questions; they do not explain the high value. Nothing here is adopted.

## 7. Reproducibility and limits

Every raw NEON file, the archived receipt, the original run file and the Saxton-Rawls PDF listed in root_audit.json were re-hashed before use (tables/provenance.csv); Root's JSON/CSV copies are in data/ with their hashes in artifact_manifest.json. The builder recomputes all values itself and asserts agreement. Limits: the megapit is a 2013 proxy for 2024 sensors; organic carbon is estimated; the Tietjen paragraph reference and the PDF equation wording were checked by Root and by the PDF text, not re-derived here; no measured 6 kPa retention value was used or found in the audited inputs; and no model was run. No Word or LibreOffice renderer was available, so page layout and equation appearance in Word have not been visually checked.

## References

[1] Saxton, K. E., & Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569-1578. Table 1 (Eq. 2-12), Table 2 (field capacity = 33 kPa), Table 3 (sandy examples). https://doi.org/10.2136/sssaj2005.0117 ; https://www.atmos.illinois.edu/~sshu3/model/saxton2006.pdf
[2] Tietjen, B., Zehe, E., & Jeltsch, F. (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418. Section 3.1 (field capacity, similar-site field values, about -6 kPa). https://doi.org/10.1029/2007WR006589 ; https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006589
[3] NEON (RELEASE-2026). Soil physical and chemical properties, Megapit, DP1.00096.001 (Jornada pit, 2013-03: per-biogeochemistry-sample, per-horizon and per-bulk-sample tables). https://data.neonscience.org/data-products/DP1.00096.001/RELEASE-2026
[4] Earlier SYRUP reports cited by file name only: SYRUP_JORN_160day_rainfall_comparison_20261119.docx and SYRUP_JORN_160day_K_Ks_multipliers_2x_3x_4x_20261120.docx.
