# Phase 3: observed shallow-profile validation — plan

Date 2026-11-14. Status: **PLAN, data audit and decision milestone.** Nothing here is a model run, fit, calibrated diffusivity or field validation. Phase 2.5 is complete and published (main `b9a86ebd6eaa5c605562f232657e35922b9d40e7`; exact 180-file production package `37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0`). Data facts live in [phase3_observation_inventory.md](phase3_observation_inventory.md) and its [JSON](phase3_observation_inventory.json); this page is the plan. Integrity evidence (hashes, counts) is kept distinct from scientific admission throughout.

## 1. Goal and honest scope
Ask whether the accepted few-layer, Tietjen-inspired SYRUP water column, driven by declared forcing, reproduces **observed shallow soil moisture** at JORN and MOAB better than simple null controls, with every geometry/boundary/forcing choice declared and sensitivity-tested. The observations are three point probes per plot (5.8/15.8/25.8 cm at JORN plot 4, 5.5/15.5/25.5 cm at MOAB plot 2), not layer averages. With two single-month windows and no measured soil hydraulic properties, this phase can at most **test mechanisms and bound plausible behavior**; it cannot validate fitted physical soil properties or establish a diffusivity. A fit that cannot beat persistence is reported as "no improvement over persistence", not as zero diffusivity. Persistence skill is computed and reported separately (§7), because with evaporation or gravity active a zero exchange can still beat persistence.

## 2. What is already settled (not new questions)
- Architecture and qualified behavior: floating layers on the MAPLE surface, SR2006 properties from explicit sand/silt/clay, organic matter and density (measured overrides), field capacity at −6 kPa, gravity always on, saturated initial states refused at initialization, one water authority (SYRUP), every bottom export counted once ([Phase 2](phase2_qualification.md), [Phase 2.5](phase25_qualification.md)).
- **Lower boundary:** the primary configuration is the existing open lower boundary above actual bedrock, with an explicit closed-bottom sensitivity. This is existing architecture, not a request. The profile bottom is a model choice; **bedrock was not observed there** and must not be implied.
- The Nevada zero-ET/zero-flux experiment was a lysimeter control; it is **not** a JORN prescription.
- PET (when used) is the accepted Hargreaves option and needs **air** daily Tmin/Tmax, date and latitude ([weather_pet.md](weather_pet.md)); soil temperature is not a substitute.
- Vegetation feedback from Phase 2.5 is not field-qualified; the first diagnostics keep plant transpiration off and invent no plant demand, leakage or lateral flux.

## 3. Pending user decisions (Root has asked; NOT adopted until an answer is recorded)
| # | Question | Root's recommendation (not adopted) | Alternatives |
|---|---|---|---|
| D1 | Layer geometry | three 10 cm layers (0–10 / 10–20 / 20–30 cm), putting the three probes in layers 1, 2, 3 | profile bottom at 50 cm or 100 cm |
| D2 | Evaporation / transpiration | a zero-E/T mechanism control plus a weather-driven soil-evaporation sensitivity with plant T off, labelled a partial budget | zero E/T only; wait for plant demand |
| D3 | Initialization | integrate the piecewise-linear first simultaneous passing probe profile over each layer, constant extension above and below | one probe value per layer |
Until answered, every experiment below is a template with these three slots open: **no field-model run has been started**, and the dependent experiments wait for the unresolved physics decisions and inputs. The user's existing automatic-workflow authority is unchanged; this task is plan-only and asks for no extra approval. Experiments blocked by each decision are named in §9.

## 4. Data and QA (summary; details in the inventory)
| Item | JORN plot 4, Apr 2025 | MOAB plot 2, May 2025 |
|---|---|---|
| Rows / passing 501-502-503 | 1,440 / 1,217-1,379-975 | 1,488 / 1,284-1,197-1,330 |
| Simultaneous three-probe passing | 785 | 939 |
| Primary-gauge rain | 5.487 mm | 24.093 mm |
| Soil temperature | retrieved (ST 502-504 ≈ SWC 501-503 depths) | not retrieved |
| Air temperature | **missing** | **missing** |
| Soil texture/OM/density | JORN pit (one pit, 2013) | **missing** |
- RELEASE-2026 files, raw hashes and the corrected-depth table (`876fd5bd…`) are **frozen**. The announced RELEASE-2027 (see [NEON notice, 2026-11-09](https://www.neonscience.org/impact/observatory-blog/soil-water-content-and-water-salinity-dp100094001-algorithm-updates); product page [RELEASE-2026](https://data.neonscience.org/data-products/DP1.00094.001/RELEASE-2026)) would be a separate dataset with its own qualification and comparison, never a silent mixed-release replacement. Its factory-versus-soil-specific uncertainty and temperature-artifact remarks are noted; no thermal correction is invented.
- Integrity versus admission: 746 accepted raw + 49 quarantined metadata = 795 entries; the quarantines stay quarantined. All are hash-checked, not all are scientifically admitted.
- The moisture `sensor_positions` file is shifted; only the pinned corrected table gives depth. Soil-temperature codes differ from moisture codes.
- Soil-moisture QA used for the inventory: finite value, `VSWCFinalQF = 0`, science review blank/0, no filling. The 72 h antecedent, the minimum passing counts and the UTC−7 day are **analytical proposals** for discussion, not accepted science or scheduler rules.
- The reported 0.2135 m³/m³ expanded uncertainty is retained and reported as supplied (original bytes untouched). It is large relative to the changes discussed here; it does not by itself characterize the precision of relative changes, and its magnitude is neither a reason to discard it nor to ignore calibration and temperature drift.

## 5. What can be tested, and what cannot
| Test | Needs | Can say | Cannot say |
|---|---|---|---|
| **A. Zero-E/T interlayer-exchange mechanism control** (rain-free window) | D1, D3 answers; declared lower boundary | whether interlayer exchange adds demonstrated skill beyond persistence and the other active terms (gravity, declared lower boundary) over the window | real dry-down; evaporation; physical diffusivity |
| **B. Weather-driven soil-E sensitivity, T off** (D2 recommendation) | D1–D3, **air Tmin/Tmax**, latitude | how much of the shallow decline a declared soil-E could explain; explicit partial budget; the numerical water ledger of every modeled run closes (see §8) | plant uptake, total ET, or **complete field water-budget closure** (no observed transpiration, deeper or lateral flux) |
| **C. Replicate / independent-window evaluation** | A or B settled, other plots and windows passing QA | whether a behavior transfers across plots and times | independence of 30-min samples; site-property inference |
| **D. Rain pulse** (later) | rain-to-surface coupling qualified at the site, forcing decision | infiltration/redistribution after rain | not planned now |
Only the shallowest probe shows a clear recession in the candidates; the deeper changes (0.0015–0.003 m³/m³) are the same order as their daily swing and as the temperature artifact NEON documents. Expectations must be set accordingly.

## 6. Method choices to review (engineering proposals unless marked)
**Observation operator (proposal; interacts with D1).** Map model state to the probe depth explicitly: the primary operator is the layer-mean θ (layer water / layer thickness) of the layer containing the probe; a declared sensitivity uses a piecewise-linear profile through the layer centers evaluated at the probe depth. A point probe is not a layer average, so the mismatch is carried as representativeness uncertainty, not hidden. No operator is adopted before D1.
**Initialization (D3 pending).** Whatever is chosen, it comes from the first simultaneous passing profile, with the chosen extension and the integration step documented. D3 yields **layer inventories** by integration or assignment, and the capacity check applies to **that candidate profile** with the declared layer properties. A single point reading above an inferred SR2006 layer porosity does not by itself show the integrated candidate violates it and must not automatically invalidate or delete the observation. If the actual candidate violates the production unsaturated-initialization rule, the configuration is flagged/refused; observations are not clipped and properties are not adjusted to make it fit. Because the properties are inferred, investigate representativeness and property uncertainty before concluding the field data are wrong; missing hydraulic inputs block physical interpretation, and no residual content, field capacity or Ks is chosen silently to accommodate a desired curve. Heterogeneity of a point profile against a layer average is a stated uncertainty; run alternative initializations as sensitivities.
**Properties.** SR2006 requires measured primary texture, organic matter and density. JORN has one pit (the A-horizon bulk-density sample depth of 0–110 cm is reported verbatim, not corrected); **MOAB has no soil properties retrieved, which blocks physical-property attribution there.** Properties assigned to model layers from horizons are a declared mapping (D1), and no horizon is a layer by itself.
**Time.** Any UTC−7 grouping means a fixed offset, not civil/daylight time; the eventual protocol must record exact UTC day boundaries. SYRUP advances a dry interevent day as one daily soil update; observations are 30-minute. **Declared methodological proposal (pending the actual protocol, not an extra user question):** the **primary** score compares each native daily model end-state, at a declared clock (UTC or local, same clock for model and data), with the passing observation interval whose END coincides with that state; the representativeness of a 30-minute mean against an instantaneous model state is stated explicitly. A day with no passing target has **no primary score** (no nearest-time substitute or forward fill unless a bounded rule is declared beforehand). Daily medians or similar aggregates are descriptive/secondary and are **not** substitutes for the end-state without an explicit temporal observation operator. No subdaily soil updates and no interpolated daily trajectory are created; diurnal behavior is not scored. Observation days, not 30-minute values, are the unit of evidence: 30-minute values are autocorrelated and are not independent replicate counts.
**Uncertainty.** Carry (i) the supplied sensor uncertainty (retained as is), (ii) the unresolved temperature artifact as a stated allowance, not a correction, (iii) representativeness (point versus layer), (iv) initialization alternatives, (v) forcing gaps. Report intervals and a persistence-relative score, not a single best fit.

## 7. Controls, calibration and validation design
- **Null controls (always run):** persistence (initial state held), zero interlayer exchange, zero-E/T mechanism control, and a no-redistribution + soil-E variant when forcing exists. The literature range of D is a comparison scale, not a prior truth. **Persistence skill is computed and reported separately** from any fitted-exchange skill. A fitted exchange of zero means "no demonstrated added skill from exchange, given the other active terms, this dataset and its uncertainty" — it does not mean the soil has zero physical diffusivity.
- **Production parameter vs diagnostic (so later scripts do not misreport).** The production mode `tietjen_donor_depth` uses the dimensionless Tietjen coefficient `d` (≥ 0) and an explicit interface conductivity `K` [m/s]: the directional diffusivity is `D_direction = d × K × h_donor` with `h_donor` the donor layer thickness (Tietjen et al. 2009, eq. 9). The optional `constant` mode takes `soil.diffusion.diffusivity_m2_s` [m²/s] directly. A constant-D scan is a **diagnostic** only: its fitted D must never be reported as the production parameter `d`. No new infiltration or diffusion law and no invented literature value is introduced here.
- **Calibration vs validation:** declare, before any run, which parameters are free (at most the interlayer exchange and a soil-E scaling), the single window and plot they are fit on (proposal: JORN plot 4, run A in the inventory), and **separate** evaluation targets: other JORN plots over the same period (spatial replicate, but not independent soil), other JORN windows from local months (temporal), and MOAB (different site: needs its own properties, so it tests transfer, not parameters). Scores on the fit window may be reported only as labelled **training** scores and are never called independent validation.
- **No claims from the two candidates alone:** JORN April and MOAB May 2025 cannot validate fitted soil physical properties; they test mechanisms.
- Daily proposals (72 h antecedent, passing counts, UTC−7) are tuned only by recorded rules, never by the outcome.

## 8. Verification requirements for any later run
Pinned sources (production package `37d89f23…`/180 files, MAPLE 9f466d2cfcd52991b24496cf67f4562d6268afa9 package `fbc504bb…`/412, LEAF ff127d29bd1d40f17a1800ffeef87bf080343f67 package `38009872…`/28, helper `333921bb…`) asserted before and after; the observation inputs are consumed only via the frozen RELEASE-2026 hashes and `swc_depthsV3`; water-ledger closure (initial + rain − E − T − bottom export = final) with the bottom export counted once; NumPy/Numba (and, if used, GPU) agreement within a declared tolerance; **every modeled run, including every zero-E/T and soil-E run, must close its numerical water ledger** (conservation is never optional, whatever the field budget can or cannot establish); a mid-window checkpoint/restart equals the uninterrupted run; deterministic repeat. A separate importer test covers depth joins, flag masks, units and the code shifts. Documentation changes alone need none of these.

## 9. Proposed bounded next experiments
1. **E0 (no decision needed):** NEON importer and QA mask with provenance receipts; independent retrieval of air temperature and adjacent months/property metadata (separate follow-ups; credential is configured, nothing to request).
2. **E1 (after D1, D3):** Test A on JORN plot 4 run A with persistence/zero-exchange controls; reported as a mechanism diagnostic.
3. **E2 (after D2 and air temperature):** Test B soil-E sensitivity, T off, partial budget.
4. **E3:** Test C on other plots and local windows; then MOAB once its properties exist.
5. **E4 (later):** rain pulses. Blocked: D1–D3 and forcing for E1–E3; air temperature for E2; MOAB properties for any MOAB physical attribution.

## 10. Decision register
| ID | Item | Status | Recommendation / note |
|---|---|---|---|
| D1 | Layer geometry | **PENDING (user)** | three 10 cm layers recommended, not adopted |
| D2 | E/T treatment | **PENDING (user)** | zero E/T control + soil-E sensitivity, T off, recommended, not adopted |
| D3 | Initialization | **PENDING (user)** | piecewise-linear integrated first passing profile recommended, not adopted |
| A1 | Open lower boundary primary, closed sensitivity | existing architecture | no permission needed; bedrock not observed |
| A2 | RELEASE-2026 frozen; 2027 separate | accepted plan method (Root agreed) | no further agreement needed |
| A3 | Observation operator | engineering proposal | review with D1 |
| A4 | Daily aggregation and QA thresholds | analytical proposal | not accepted science |
| F1 | Air temperature, adjacent months, MOAB/JORN property files | independent retrieval follow-ups | credential configured; no new authority needed |
Not asked again: the exploratory ten-question list; quarantined files stay quarantined; no plant demand, leakage, lateral flux or simulated "observations" are introduced.

## 11. References
NEON, [DP1.00094.001 RELEASE-2026](https://data.neonscience.org/data-products/DP1.00094.001/RELEASE-2026); NEON observatory notice on algorithm updates, [published 2026-11-09](https://www.neonscience.org/impact/observatory-blog/soil-water-content-and-water-salinity-dp100094001-algorithm-updates); NEON air-temperature [quick-start guide](https://data.neonscience.org/api/v0/documents/quick-start-guides/NEON.QSG.DP1.00002.001v1?fallback=html&inline=true); Tietjen, Zehe & Jeltsch (2009), [WRR](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006589); Saxton & Rawls (2006), Soil Sci. Soc. Am. J. 70, 1569–1578; Hargreaves & Samani (1985) as cited in [weather_pet.md](weather_pet.md).
