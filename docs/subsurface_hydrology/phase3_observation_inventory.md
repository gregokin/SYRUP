# Phase 3 observation inventory (JORN and MOAB shallow soil moisture)

Status: plan-stage **data-integrity and QA inventory**, 2026-11-14. It records what exists locally and how it was checked. It is **not** a validation, calibration or model result, and it does not choose any physical parameter. Machine-readable twin: [phase3_observation_inventory.json](phase3_observation_inventory.json) (selected file names, sizes, SHA-256, retrieval manifests, QA counts, candidate intervals, pending decisions). The companion plan is [phase3_validation_plan.md](phase3_validation_plan.md).

## 1. Release and provenance
- Product: NEON DP1.00094.001, soil water content and water salinity, **RELEASE-2026** ([product page](https://data.neonscience.org/data-products/DP1.00094.001/RELEASE-2026)). Retrieved 2026-11-02 into eight local retrieval directories, each with its own `manifest.json` (hashes in the JSON). Selected raw-file paths in the JSON are relative to the JORN raw retrieval root `jorn_three_layer_20261102`; the corrected-depth table is given relative to the task archive (`agent_handoffs/tasks/`); files are identified by site, product id, month, release and file name, so they can be re-obtained from NEON without the local path.
- NEON announced (2026-11-09) corrected depths, calibration and QA changes, with a reprocessed **RELEASE-2027** expected January 2027 ([notice](https://www.neonscience.org/impact/observatory-blog/soil-water-content-and-water-salinity-dp100094001-algorithm-updates)). In our words: depth and calibration handling and the uncertainty reporting (factory versus soil-specific calibration) will change, and temperature artifacts remain a known issue. **Decision of this plan:** RELEASE-2026 files, their raw hashes and the corrected-depth table are frozen; RELEASE-2027 is a separate dataset needing its own qualification, never a silent replacement. It is not downloaded, and revised factory calibration would not make the values absolute water truth.
- Corrected depths: `swc_depthsV3`, sha256 `876fd5bdcfb8b93352399dab20a3b41e01b2ca3f74bb140e3216feb80812827f` (2,217 rows, retrieved 2026-11-01 from NEON's documents API). The moisture `sensor_positions` file of this release is shifted by one level against it and is not used for depth.

## 2. Integrity evidence (not scientific admission)
795 manifest entries across eight retrievals: **746** raw entries pass their listed MD5/SHA-256/size, and **49** metadata entries are **quarantined** (listed MD5 rejected; they differ from the listed checksum). The quarantines are preserved as quarantined: LF-normalized MD5 diagnostics do not rehabilitate them. Root re-hashed all entries independently (`root_independent_data_audit.json`, status `PASS_INDEPENDENT_RAW_PROVENANCE_AND_CORE_QA_COUNTS`) and the preflight probes agree. Passing a checksum shows the bytes are what NEON listed; it says nothing about whether a series is physically usable.

## 3. Probe geometry
| Site / plot | SWC vertical 501 / 502 / 503 (corrected, cm) | Note |
|---|---|---|
| JORN plot 4 | 5.8 / 15.8 / 25.8 | plot 2 is +0.1 cm, plot 4 −0.2 cm relative to plots 1,3,5 |
| MOAB plot 2 | 5.5 / 15.5 / 25.5 | deeper levels 35.5/65.5/95.5/145.5/195.5 cm exist in the table |
These are **point measurements**, not layer averages or soil horizons. Soil-temperature vertical codes differ from moisture codes: ST 502/503/504 (−0.06/−0.16/−0.26 m) are the depth-equivalents of SWC 501/502/503.

## 4. Series, quality and joint availability (rows = 30-minute UTC starts)
Passing = finite value, `VSWCFinalQF = 0`, science review blank or 0. Flags are retained as supplied; nothing is filled or relabeled.
| | MOAB plot 2, May 2025 (1,488 rows) | JORN plot 4, April 2025 (1,440 rows) |
|---|---|---|
| 501 / 502 / 503 passing | 1,284 / 1,197 / 1,330 | 1,217 / 1,379 / 975 |
| Simultaneous three-level passing | **939** | **785** |
| VSWC range (m³/m³) | 0.028–0.094 / 0.049–0.066 / 0.047–0.058 | 0.024–0.281 / 0.058–0.074 / 0.052–0.062 |
| Rain (primary tipping bucket) | all rows pass; 18 non-zero intervals; **24.093 mm** | all rows pass; 9 non-zero; **5.487 mm**; throughfall plots 1,2,4,5 report 5.606–6.938 mm (no plot 3) |
| Soil temperature | not retrieved | retrieved; ST 502/503/504 pass 1,435/1,434/1,435 of 1,440 |
| Air temperature, humidity, radiation | **not retrieved** | **not retrieved** |
The reported expanded uncertainty is 0.2135 m³/m³ in nearly every row; it is a supplied constant, preserved and reported as is (original flag/uncertainty bytes untouched). It is large relative to the changes discussed here and does not by itself characterize the precision of relative changes, nor justify ignoring calibration or temperature drift; its size is not a reason to discard it. Default-calibration fractions are ≈ 100 %.

## 5. Candidate rain-free intervals (a proposed analytical rule, not accepted science)
Rule used for the inventory only: contiguous rain-free half-hours (primary gauge passing and zero), runs ≥ 120 h, evaluable part = run minus its first 72 h; UTC half-hour starts. The 72 h, the passing thresholds and the UTC−7 day are **proposals** for discussion, not scheduler rules. Month-edge runs are **censored** because adjacent months are not retrieved.
| Candidate | Run (UTC) | Evaluable | Simultaneous passing (of evaluable half-hours) | Longest gap | Censored |
|---|---|---|---|---:|---|
| JORN-4 run A | 04-05 21:00 → 04-22 16:00 (403.5 h) | 331.5 h | 431 (65 %) | 13 half-h | no |
| JORN-4 run B | 04-22 17:00 → 04-30 23:30 (199 h) | 127 h | 128 (50 %) | 26 | month end |
| MOAB-2 run C | 05-19 03:00 → 05-31 23:30 (309 h) | 237 h | 324 (68 %) | 9 | month end |
| MOAB-2 run B | 05-08 00:30 → 05-14 17:00 (161 h) | 89 h | 110 (62 %) | 12 | no |
| MOAB-2 run A | 05-01 00:00 → 05-06 23:30 (144 h) | 72 h | 51 (35 %) | 17 | month start |
Descriptive signal in the best intervals (no temperature correction): JORN run A shallow probe median 0.0398 → 0.0292 with mean daily range 0.0079; second and third probes change 0.003 against daily ranges 0.006/0.004. MOAB run C shallow 0.0595 → 0.0306 (range 0.0068); deeper probes change 0.0015–0.003. JORN soil temperature at ≈ 6 cm swings about 15 °C per day. Only the shallowest probe shows a clear downward **trend**; that is an observed trend, not proof of a physical recession, and the deeper changes are the same order as the daily swing and as the temperature artifact NEON warns about.

Machine-readable interval IDs are `JORN-4-A`, `JORN-4-B`, `MOAB-2-A`, `MOAB-2-B` and `MOAB-2-C`, matching the letters in the table. The JSON candidate entries and descriptive-statistics keys use those same IDs; MOAB-2-A has no descriptive-statistics block. The UTC−7 analytical day is a fixed-offset grouping, not a claim about civil/daylight time.

## 6. Soil properties available
JORN megapit (DP1.00096.001, 2013-03, one pit): horizons A 0–18, Bw 18–46, 2Bk 46–78 cm …; texture (sand/silt/clay %): A 86.7/6.2/7.1, Bw 83.7/7.9/8.4, 2Bk 78.8/8.6/12.6; bulk density 0.9035 (Bw), 1.6396 (2Bkk2) and **1.4196 for horizon A with a sample depth of 0–110 cm — reported verbatim and not corrected**. Organic matter/carbon fields are in the biogeochemistry table but were not extracted here. One pit is not the five instrumented plots, and a horizon boundary is not a model layer. **MOAB has no megapit data retrieved**, so any attribution of a MOAB response to soil physical properties is blocked until those files are retrieved and checked.

## 7. Missing for a physical test
Air temperature (Tmin/Tmax) for both sites; adjacent months; MOAB texture/organic matter/density and temperature; hydraulic properties measured at either site; plant demand; lower-boundary flux. Retrieval of air temperature ([reference](https://data.neonscience.org/api/v0/documents/quick-start-guides/NEON.QSG.DP1.00002.001v1?fallback=html&inline=true)) and the adjacent months/property metadata is an independent follow-up, not something this plan performs.

## 8. Not admitted or not claimed
All 795 entries are integrity-checked, not scientifically admitted. Quarantined files stay quarantined. No simulated or prior-model series is labeled an observation; the Nevada zero-flux/zero-ET runs are a Nevada control and not a JORN prescription. Older preflight pointers (hashes in the JSON) are evidence of the earlier inventory and are not copied here.
