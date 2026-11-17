# Why the soil-property results differ: evaporation, moisture stress, and d and E

*A plain-language companion to the JORN/MOAB interface-comparison report · 2026-11-17 · built from saved results; no new model run, no parameter or default changed*


## 1. The short version

- **Potential evapotranspiration (PET) is the weather's demand for water, not the water actually lost.** It is estimated from the calendar date, the day's air temperatures (lowest, highest, mean) and the site's latitude. The soil can only give up what it holds.
- **Capital E is a plain multiplier with no units.** It scales the PET we hand to the soil. E = 1 keeps the PET, E = 0.5 halves it, E = 0 turns soil evaporation off. E is not a number of millimetres.
- **Lowercase d is a dimensionless tuning knob for how fast neighbouring soil layers trade water; it moves water between layers and does not remove it.** It is not a diffusivity in square metres per second, not a drainage rate, not an amount of water and not a sediment property.
- **The evaporation-stress scale is a different thing again: a moisture level.** It is a volume fraction, so 0.18 means 18 % of the soil volume is water, not 0.18 %. Residual moisture is the floor below which the model never removes water; it is not the stress scale.
- **The big difference between the earlier generic soil and the new site soils, for evaporation, is mostly the stress scale (0.18 versus about 0.04–0.06), not E or d.** A lower scale lets the same dry soil lose more water.


## 2. Terms that are easy to confuse

**Table 1. What each setting is, in everyday terms.**

| Name | What it is | Units | What it does NOT mean |
|---|---|---|---|
| PET (potential evapotranspiration) | Weather-driven demand for water, from date, air temperatures and latitude | mm per day | Actual water lost |
| E (evaporation multiplier) | Scales the PET given to the soil before any soil limit | none (dimensionless) | Millimetres of evaporation; a measured quantity |
| d (exchange coefficient) | Scales how fast neighbouring layers trade water | none (dimensionless) | Physical diffusivity (m² per s), drainage rate, water amount, sediment parameter |
| Evaporation-stress scale | Top-layer moisture at and above which the moisture multiplier is 1 | volume fraction (0.18 = 18 %) | A cut-off where evaporation stops; the residual |
| Residual moisture | Lowest moisture the model lets the layer reach | volume fraction | The stress scale |

A note on units. Moisture is always a fraction of soil volume (a decimal such as 0.05, which is 5 %). Water amounts are millimetres of water depth over the soil surface; a 10 cm layer holding a moisture of 0.05 contains 5 mm of water. PET is in millimetres per day. The test column is one square metre, so millimetres of depth and litres per square metre are the same number.


## 3. How one day of soil evaporation is calculated

In the bare, flat dry-down test the top soil layer is treated like this, in order. No plants, no rain, no slope, no wind:

- **Step 1 – start from the day's PET.** For example 6 mm. (If the PET is zero, nothing is demanded whatever E is.)
- **Step 2 – multiply by E.** This is the demand before the soil has any say. With E = 0.5 and PET 6 mm the demand is 3 mm.
- **Step 3 – apply the moisture multiplier.** Look at the top layer's moisture. At or above the stress scale the multiplier is 1 and the full demand stands. Below the scale, divide the moisture by the scale and multiply that answer by itself. At or below the residual moisture the multiplier is zero and nothing is removed.
- **Step 4 – cap by what the layer can give.** Whatever the demand is, SYRUP cannot take more than the water in the layer above its residual moisture.
- **Step 5 – take the actual loss out of the layer.** SYRUP debits only the actual amount and records any shortfall as unmet demand. No water appears from nowhere to fill it.

Two consequences. At half the scale the multiplier is one quarter (half times half); at a quarter of the scale it is one sixteenth. Both are before the water-stock cap, and only if the moisture is still above the residual: for the new JORN soil a quarter of the scale (0.0099) is already below the residual (0.01), so the actual loss there is zero, not one sixteenth. And **lowering the scale lets MORE evaporate at the same moisture**. The scale is not a moisture cut-off below which evaporation stops. The multiplier itself jumps from zero at the residual to a modest value just above it; the actual loss still changes continuously, because the water available above the residual shrinks to nothing there.

![Figure 1. The moisture multiplier (the share of the E-scaled demand that moisture allows) against top-layer moisture, for three stress scales. This is the multiplier BEFORE the water-stock cap in step 4. The actual loss can be lower, and is zero at or below the residual (0.01). Moisture is a volume fraction (0.05 = 5 %).](figures/fig_moisture_response.png)

*Figure 1. The moisture multiplier (the share of the E-scaled demand that moisture allows) against top-layer moisture, for three stress scales. This is the multiplier BEFORE the water-stock cap in step 4. The actual loss can be lower, and is zero at or below the residual (0.01). Moisture is a volume fraction (0.05 = 5 %).*

**What E means physically.** E is an empirical scaling of the prescribed reference demand, applied BEFORE the moisture multiplier and the water cap. E = 1.2 scales the reference to 120 %, then the soil still decides how much it can give. It is not an energy-balance result, it does not validate the weather-based PET, and it creates no water. Because the soil's own stock limits the answer, **doubling E need not double the actual evaporation**: once the stock is the limit, extra demand goes unmet.

**Why 1.2, and what E is not.** The reference E = 1.2 used in our tests is the combined bare-soil factor of the downloaded EcoHyD example case (its evaporation factor 1 times its vegetation intercept 1.2). It is not a literal factor of 1.2 in that input file, and our bare dry-down driver does not apply a second 1.2 on top. Nor is it the original Tietjen constant a_ET, an additional site or crust factor (section 7). In SYRUP's general vegetation-aware interface the site factor and the cover term are kept separately and the column's own E must be exactly 1, so nothing is applied twice. Do not set 1.2 in several places.


## 4. A worked example with round numbers

This is an illustration, not a new field run. A top layer 10 cm thick holds a moisture of 0.05 (5 %). The residual is 0.01 (1 %). PET is 6 mm and E is 0.5.

- **Water available.** Moisture above residual is 0.05 minus 0.01, which is 0.04. Over 100 mm of layer thickness that is **4 mm** that can be taken.
- **Demand before moisture.** 6 mm times 0.5 is **3 mm**.
- **Generic soil (scale 0.18).** The moisture 0.05 is below 0.18. Divide 0.05 by 0.18 to get 0.2778; multiply that by itself to get the multiplier **0.07716** (7.716 %). Three millimetres times that is **0.2315 mm**, well below the 4 mm available, so that is the actual loss.
- **JORN-derived soil (scale 0.0395).** The moisture 0.05 is above the scale, so the multiplier is **1** and the full 3 mm is demanded. 3 mm is below the 4 mm available, so the actual loss is **3 mm**, about 13 times the generic result.
- **With E = 1.2 instead.** The demand becomes 6 times 1.2, which is 7.2 mm before moisture. The JORN-derived soil lets all of it through the moisture step, but only 4 mm are available, so the actual loss is **4 mm** and 3.2 mm of demand goes unmet. The generic soil would lose 0.5556 mm. So raising E from 0.5 to 1.2 multiplies the JORN-type result by only 4 ÷ 3 = 1.33, because the stock became the limit.

**The same calculation with the real first JORN A day**, using values from the saved results. The top layer starts at a moisture of 0.04759 and the first day's PET is 6.139 mm. With E = 0.5 the demand before moisture is 3.070 mm. Under the earlier generic scale the moisture ratio is 0.2644; times itself the multiplier is **0.06991** (about 7 %), so the first day's loss is 0.2146 mm. Under the new JORN scale the multiplier is **1**, so the loss is 3.070 mm. The layer holds 3.759 mm above residual, so over the ten-day window the new soil gives all of it (3.759 mm) and ends at the residual, 0.01. The same first-day arithmetic matches stored records for all six old and new window cases (an independent check by Root).


## 5. What changed between the earlier generic soil and the new site soils

**Table 2. Earlier generic reference soil against the texture-derived site soils (moisture values are volume fractions).**

| Soil | Stress scale | Residual | Saturated conductivity (mm/h) | Layer-to-layer conductivity (mm/h) |
|---|---|---|---|---|
| Earlier generic reference | 0.18 | 0.01 | 6.9 | 2.1829 (all pairs) |
| New JORN (3 layers) | 0.0395 (top-layer water content at 1,500 kPa) | 0.01 | 120.8 / 119.7 / 104.3 | 38.0 and 35.3 (harmonic) |
| New MOAB (3 layers) | 0.0590 (same rule) | 0.01 | 64.0 / 57.3 / 57.3 | 19.1 and 18.1 (harmonic) |

The new soils have layer-to-layer conductivities roughly 8 to 18 times larger. In these runs that affects only the exchange between layers; gravity drainage is not limited by it here and is exactly zero. Far more important for evaporation, the new stress scale is much lower. It is the top layer's predicted water content at 1,500 kPa suction from the Saxton–Rawls regression. Choosing that as the evaporation scale is a SYRUP experiment choice, in the spirit of Tietjen's stress moisture (section 7); Saxton–Rawls does not calibrate how evaporation responds to moisture, and the choice does not validate the evaporation.

A word on two kinds of averaging. Inside one soil, Tietjen's model and SYRUP take the redistribution conductivity as a fixed fraction of the saturated conductivity (about 31.6 %). Our harmonic-versus-geometric comparison is a different step: it combines the values of two adjacent layers into one conductivity for the boundary between them.


### The controlled comparison: d = 0, E = 0.5

To separate the stress-scale effect from everything else, we compare the two soils with d = 0 (no exchange between layers) and E = 0.5. The saved trajectories of the single one-square-metre column give the total evaporation below. In these controls no exchange, no gravity and no saturation limit acts; the residual cap does act. All states stay below field capacity, so drainage, percolation and surface return are exactly zero in every run (the bottom does not leak).

**Table 3. Total evaporation over each window with d = 0 and E = 0.5, from the saved trajectories (mm of water from a 1 m² column).**

| Window | Days | Earlier generic (mm) | New site soil (mm) | New ÷ old |
|---|---|---|---|---|
| JORN A | 10 | 1.5824 | 3.7593 | 2.38 |
| JORN B | 4 | 0.4131 | 2.3596 | 5.71 |
| MOAB C | 6 | 1.8412 | 5.3409 | 2.90 |

Evaporation is roughly 2.4 to 5.7 times larger with the lower scale. JORN A's new total of 3.759 mm is exactly the water its top layer held above residual, so the new soil simply ran out; the residual cap, not the weather, ended the loss.


### What happens when d is above zero

With d above zero the interface conductivity matters. It rose from 2.182866 mm per hour (earlier soil) to about 35–38 at JORN and 18–19 at MOAB. A drying top layer is then refilled from the layer below, so water can evaporate again on later days. The actual exchange is not 8 to 18 times larger, because the exchange has a finite capacity and the donor and receiver limits cap it (worked example in section 6). Harmonic versus geometric averaging changes the result only trivially.

**Table 4. Old and new outcomes from the saved runs at two settings. Score = equal-layer RMSE against observed moisture (volume fraction, ×10⁻³; lower is better). Evaporation = total over the window.**

| Window | Setting | Evaporation old (mm) | Evaporation new (mm) | Score old (×10⁻³) | Score new (×10⁻³) |
|---|---|---|---|---|---|
| JORN A | d 0, E 0.5 | 1.582 | 3.759 | 4.23 | 17.24 |
| JORN A | d 0.05, E 1.2 | 2.933 | 11.107 | 7.63 | 29.21 |
| JORN B | d 0, E 0.5 | 0.413 | 2.360 | 3.44 | 10.73 |
| JORN B | d 0.05, E 1.2 | 0.927 | 5.963 | 3.77 | 14.50 |
| MOAB C | d 0, E 0.5 | 1.841 | 5.341 | 6.43 | 13.88 |
| MOAB C | d 0.05, E 1.2 | 3.338 | 8.851 | 1.52 | 18.67 |

In this small comparison the new soils score worse than the earlier generic soil at both settings, mainly because they evaporate too much when the same E is used. That says something about these three short windows and a coarse grid, not about which soil is right.


## 6. What d does

Neighbouring layers trade water towards the drier one. In SYRUP (and in the later EcoHyD code) the law uses four ingredients: the conductivity between the layers, the thickness of the layer giving up water, the moisture difference, and the distance between the layer centres. **d scales the strength of that trade.** The transfer moves water from the wetter layer to the drier one, so the profile's total water is unchanged; E, by contrast, removes water from the profile.

**A plain-number illustration (analytical, not a model run).** Take two 10 cm layers at the JORN upper interface, one at moisture 0.04 and one at 0.06, with d = 0.05 and one day. Levelling the two would move **1 mm** of water, which is the most the exchange can ever move here. SYRUP's unchanged solver moves only a share of that in a day. With the new site conductivity (38.04 mm per hour) the share is about **60 %**, which is **0.599 mm**. With the earlier soil's conductivity (2.18 mm per hour) it is only about **5.1 %**, which is **0.051 mm**. The conductivity is about 17 times larger, yet the transfer is only 11.7 times larger, because it cannot exceed the 1 mm available for levelling; the share levels off as it nears 100 %.

Three cautions. First, **d = 0 does not switch off gravity**: water above field capacity still drains downward. **E = 0 does not switch off redistribution or drainage.** Second, this is a simple moisture-gradient approximation, not a Richards-equation solver. Where layers have different retention properties, real water can keep a moisture difference that this law would erase. Third, a given d need not transfer the same amount for other conductivities, layer depths, soils or ways of reading the sensors; it is a calibration constant for a configuration. The exact pair levelling and the top-down sequential update of several layers are SYRUP's own numerical adaptations.


## 7. Where d and E come from

This section separates what the publications say, what later code and downloaded model inputs contain, and what SYRUP chose.


### Tietjen's published model (2009, 2010)

From the publisher's full text of Tietjen et al. (2009) [1] (Sections 2.3 and 3.1, Table 2): the exchange constant d is 0.01 at all four sites, adjusted so the exchange matches the diffusion scale of Bai et al. (2007) and observed moisture series. A constant, a_ET, reduces evaporation for site effects such as soil crusts; it is normally 1 and was calibrated to 0.4 at Sde Boqer only. The authors propose estimating it by comparing measured evaporation from crusted and bare patches. The moisture where stress begins corresponds to about -1500 kPa, taken from similar nearby field sites.

Tietjen et al. (2010, Table I) uses d = 0.05, which it says was estimated according to Bai et al. (2007). The same table lists a saturated conductivity of 17.6 mm per hour, a moisture at the beginning of stomatal closure of about 0.079, a field capacity of about 0.226, a residual of 0.06 and layers 0.2 m and 0.6 m thick, credited to unpublished measurements [2]. The table does not describe an optimiser or calibration procedure, and we do not invent one. It lists no evaporation multiplier. The change from 0.01 to 0.05 shows d is chosen for context; it is not a universal soil constant. The "stomatal closure" moisture is a plant response level, not a general wilting point.

**Our E and the site factor.** In the later EcoHyD code (below) the factor applied to PET is the product of an additional site factor (EP) and a cover term; in our bare dry-down E stands for that whole product. So a site factor of 0.4 would not by itself mean E = 0.4, and we use no such value.


### The downloaded later EcoHyD code and parameter files

The reference code we examined is a later application (Baldauf, Cantón and Tietjen 2023 [4]), not the original 2009 code. Its water-cell routine reduces the upper-layer potential loss by a squared moisture ratio below the wilting-point input, scales it by a vegetation function and an extra evaporation factor (EP, described there as an additional site or crust factor), then caps the result by the water above the residual. Its layer-exchange routine uses the diffusion constant, a geometric conductivity and the donor layer's thickness; SYRUP does the same and we make no claim about the original solver. Its Tabernas soil file has EP = 1, a vegetation intercept of 1.2, a wilting point of 0.18, a residual of 0.01 and a diffusion constant of 0.05. These are inputs for that case; their calibration history is not documented here. With bare soil the combined factor is 1 × 1.2 = 1.2, the origin of our reference E.

Two later applications (Irob and co-authors, data archives) use much lower values: EP 0.16 (bare effective 0.16 × 1.2 = 0.192, if the cover term is the same 1.2), wilting point 0.088, residual 0.075 and a diffusion constant of 0 [5]. The dissertation labels the evaporation factor as calibrated and the wilting point as calculated from the residual; the objective and optimiser are not established. These belong to a different site and application and are not the parameters of our tests.


### Saxton and Rawls (2006)

Saxton and Rawls give regression equations that predict water retention at 1,500 kPa and 33 kPa, saturation and saturated conductivity from sand, clay and organic matter, with density adjustments [6]. The paper has no evaporation multiplier, no squared-ratio evaporation law and no d, and no old evaporation model was replaced by it. Predicting the 1,500 kPa moisture to use as the stress scale is our explicit choice, in the spirit of Tietjen's field-based scale but not calibrated for evaporation. Once the layer's texture and organic matter are fixed, the implemented prediction of that moisture does not depend on our JORN density floor; the floor changes field capacity, saturation and conductivity, not the stress scale. The air-entry and density floors are project bounds, not recommendations of the paper.


### What SYRUP chose and tested

The bare diagnostic sets E and d directly for each scenario and the stress scale from the top layer: 0.18 for the earlier generic soil, 0.0395 for JORN and 0.0590 for MOAB; there is no vegetation or transpiration and the terrain is flat. A coarse grid was tried: d of 0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 1 and E of 0, 0.1, 0.25, 0.5, 1, 1.2. JORN A was the training window; JORN B and MOAB C were scored without refitting. For the generic soil the best coarse-grid pair was d = 0 and E = 0.5; for the new soils both averaging methods chose d = 0 and E = 0. These are the best of a coarse grid under stated assumptions. They are not measured values of d or E and not evidence that evaporation is zero; untested values between 0 and 0.1 might matter.

In SYRUP's general vegetation-aware interface the site factor and the cover function are kept separately, and the column's own E must be exactly 1. The soil evaporation share is applied after the combined moisture and supply cap, SYRUP debits only actual soil evaporation, and the plant model asks for transpiration separately against the water that remains; SYRUP stays the single water authority. Evaporation, transpiration and growth run only between wind and rain events.

**Table 5. Where each value comes from. The site factor (a_ET in 2009, EP in later code) is NOT the same thing as our combined bare E. Moisture values are volume fractions.**

| Source / case | d | Site factor (a_ET or EP) | Combined bare E equivalent | Stress / wilting scale | Residual | How obtained |
|---|---|---|---|---|---|---|
| Tietjen 2009, Table 2 | 0.01 | a_ET 1; 0.4 at one site (calibrated) | Not given | Moisture at about -1500 kPa | See section 7 | See section 7 |
| Tietjen 2010, Table I | 0.05 | No multiplier listed | Not given | About 0.079 (start of stomatal closure) | 0.06 | d estimated according to Bai 2007; soil values unpublished measurements |
| EcoHyD Tabernas input (later code) | 0.05 | EP 1 | 1 × cover intercept 1.2 = 1.2 (flat bare) | 0.18 (wilting point) | 0.01 | Case input; calibration history not established |
| Irob data archives (later, other sites) | 0 | EP 0.16 | 0.16 × 1.2 = 0.192 (flat bare) | 0.088 | 0.075 | Dissertation labels EP calibrated, wilting point calculated; objective unverified |
| Saxton–Rawls 2006 | none | none | none | Predicts water at 1,500 kPa | none | Texture regression; no evaporation or d law |
| SYRUP earlier generic soil | grid; best 0 | not separate | grid; best 0.5 | 0.18 | 0.01 | Coarse grid, JORN A training; not measured |
| SYRUP new JORN / MOAB | grid; best 0 | not separate | grid; best 0 | 0.0395 / 0.0590 (predicted 1,500 kPa water) | 0.01 | Experiment choice, consistent with Tietjen's concept; not calibrated for E |


## 8. How one could estimate d and E, and what this report does not do

Section 7 shows how the cited sources estimated d and the site factor. A general approach for our E would be to compare modelled drying with measured moisture or evaporation over rain-free periods; that is a suggestion of ours. Because d, E and the stress scale all shape the drying curve, one moisture series cannot easily separate them, and a fit for one soil, depth set and sensor arrangement need not carry to another. This report did no new fitting.


## 9. Limits

- Everything quantitative here comes from three short dry-down windows (JORN A, JORN B, MOAB C) of one-square-metre columns. It is a diagnostic, not field validation.
- The worked examples in sections 4 and 6 use round numbers and are illustrations; the 'real first day' numbers are read from saved results.
- The site-derived hydraulic values are pedotransfer surrogates with project bounds, not measured hydraulic properties.
- No soil here exceeds field capacity, so gravity drainage and the saturation limit are untested by these runs.


## Sources

**Table 6. Main sources. Exact file hashes are in evidence/provenance.json.**

| Source | Used for |
|---|---|
| Tietjen et al. 2009 (publisher text); Tietjen et al. 2010 PDF Table I | Origins of d, the site factor and the stress moisture |
| EcoHyD WaterCell.cpp lines 28–112, 475–495; WaterCell.h line 76; soilparameters_tabernas.txt; two Irob soilparameters_EH.txt | Later-code evaporation and exchange laws; case inputs |
| Saxton and Rawls 2006 PDF | Retention prediction at 1,500 kPa |
| SYRUP subsurface.py (step stages 4–5), soil_evaporation.py, drydown_driver.py (candidate worktree) | Law as implemented; bare diagnostic settings |
| Saved results: data/actual_cpu/*_result.json and property_effect.json; ../observed_drydown/data/*_result.json | All numbers in tables and examples |
| Root checks: root_numeric_checks.json, root_d_example_check.json, root_source_evidence.json, primary_2009_update.md | Independent checks and 2009 evidence record |

[1] Tietjen, B., Zehe, E. and Jeltsch, F. (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418. DOI 10.1029/2007WR006589. Publisher full text: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006589 (recovered 2026-11-17).
[2] Tietjen, B., Jeltsch, F., Zehe, E., Classen, N., Groengroeft, A., Schiffers, K. and Oldeland, J. (2010). Effects of climate change on the coupled dynamics of water and vegetation in drylands. Ecohydrology 3, 226–237 (online 2009). DOI 10.1002/eco.70, Table I. Author-hosted copy: https://www.biodiversity-plants.de/downloads/Tietjen_etal_2009_Ecohydro.pdf
[3] Bai, J., Deng, W., Cui, B. and Ouyang, H. (2007). Water diffusion coefficients of horizontal soil columns from natural saline-alkaline wetlands in a semiarid area. Eurasian Soil Science 40, 660–664. DOI 10.1134/S1064229307060075.
[4] Baldauf, S., Cantón, Y. and Tietjen, B. (2023). Biocrusts intensify water redistribution and improve water availability to dryland vegetation: insights from a spatially-explicit ecohydrological model. Frontiers in Microbiology 14, 1179291. DOI 10.3389/fmicb.2023.1179291. Code commit 80b10e6b6b82ad16e9fd41a31901cc3157d880e9. Local copy under reference_models/ecohyd (GPL source, not reproduced).
[5] Irob (2023). Doctoral dissertation, Freie Universität Berlin, Appendix Table A.1.3, p. 138. https://refubium.fu-berlin.de/bitstream/handle/fub188/40230/Dissertation_Irob.pdf
[6] Saxton, K. E. and Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569–1578. DOI 10.2136/sssaj2005.0117.
