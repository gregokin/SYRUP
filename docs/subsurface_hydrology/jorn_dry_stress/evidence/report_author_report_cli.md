The dry-stress DOCX is built and I'm stopping for Root's review. I waited for Root's report-ready file (PASS) and the run seal before the builder read any result. Everything is uncommitted.

**Independent check:** all 85 sealed files matched before and after the build. From the sealed results I recomputed:
- the per-layer and pooled scores,
- the group scores, which match Root's audit to a relative 1e-9,
- the budgets,
- the zero gravity transfer,
- the unchanged historical inputs,
- the preset, case and source identities.

I found no discrepancy. I made no edits to the preset, generator, tests, configs, results or source, and no Git, GPU, network or model runs.

**Results** (aligned RMSE in percentage points, stress 0.2017 against 0.30255):

| Group | n | Stress 0.2017 | Stress 0.30255 |
|---|---|---|---|
| JORN, upper two layers | 26 | 0.424 | 0.275 |
| MOAB, upper two layers | 10 | 0.487 | 1.138 |
| Combined, upper two layers | 36 | 0.442 | 0.644 |
| JORN, all layers | 39 | 0.365 | 0.253 |
| MOAB, all layers | 14 | 0.417 | 0.964 |
| Combined, all layers | 53 | 0.380 | 0.541 |

- By window, the higher stress is lower only in JORN A, so the answer depends on the window and the scoring.
- Cumulative evaporation at the higher stress is 0.483–0.575 of that at 0.2017. The fixed-moisture ratio would be 0.444, because the layers stay wetter and evaporate more later.
- Final storage is higher at the higher stress by 0.33–1.16 mm.
- Gravity transfer, bottom drainage and surface return are exactly zero in all six cases, and every layer stays below its field capacity. The field capacity is therefore not identifiable from these windows, and the report says the dry comparison does not validate the 10 vol. % choice.

**How the report treats your choices**
- It leads with your JORN selection and states that it persists regardless of the ranking, with no suggestion of reversal.
- The third-layer discrepancy stays in the scores but is deferred, with no sensor, wetting-front or root-cause investigation and no fault claim.
- It states that these 2025 windows were part of the earlier d and stress fit, so this is not held-out validation. It also says T = 0 and rain-free weather do not prove no transpiration.
- The restart control is labelled a same-process public-constructor reconstruction, not a fresh-process test or native checkpoint.

The tables have at most 6 columns and match the CSVs cell for cell. I had no Office renderer, so pagination is unchecked, and I viewed the figures as PNGs.

Files are in `analysis_candidate/docs/subsurface_hydrology/jorn_dry_stress/` (the `report_author_report.md` is in the task directory):
- `SYRUP_JORN_MOAB_dry_period_stress_comparison_20261122.docx`
- `report.md`
- `build_report.py`
- `verification.md`
- `artifact_manifest.json`
- `figures/`
- `tables/`
- `data/`