The uniform field-capacity report is built and verified, and Root's test-only tolerance tightening checks out. Everything is uncommitted and stopped for Root's independent review.

I held off until `root_report_ready.json` said PASS and `root_run_seal.json` existed (42 native runs, 14 cases, 423 sealed files). Nothing was read before that. I made no edits to tests, generator, configs or results, and no Git, GPU or network use.

**Root's test tightening**
- I reconstructed the version I authored by reverse-applying five narrow hunks to the current test file. It hashes to exactly the hash in my setup report, so Root's diff is those five hunks and nothing else.
- The old `pytest.approx(..., abs=max(bound, 1e-15))` also admitted pytest's default relative 1e-6 tolerance. The new checks use the native `budget.bound_m3` only, so they are strictly tighter. The one dropped clause (`out[9] > 10 * out[15] + 1e-3`) is implied by the remaining ordering and hand-value checks.
- I re-ran the three native tests and they pass. I found no defect.

**Review of the results**
- The builder re-hashed all sealed files before and after the build.
- It recomputed the scores, period scores and counts, endpoints, budgets, closure, the January deep-layer response, the exact field capacity and stress in every case file, and the unrescaled 22.9424 mm initial storage.
- It also checked NumPy against Numba, the 900 s controls, the 42-run source pins (180 files / 4e38 and 412 files / fbc), and Root's 90-digit and restart receipts.
- Every cross-check agrees with Root's audit files, and I found no discrepancy.

**What the data show** (the text is built from the computed shapes, not from assumed trends)
- **Fit:** pooled aligned RMSE rises to a maximum at FC 11 for both stress values, then falls. The lowest values are 1.158 at FC 15 with stress 0.2017 and 1.665 at FC 9 with stress 0.30255. Five of the 14 cases beat the no-change reference of 1.799. The earlier SR33 depth-specific case scored 2.273 and appears as context only.
- **Deep layer and the January storm:** the model deep layer is within 0.05 points of its field capacity after the January 23 storm in only 4 of 14 cases. It is wetted by more than 0.5 points, up to 10.47 vol. %, in the 8 cases with FC 12 or below. The 25.8 cm sensor never exceeds 6.74 vol. % (6.34 at the January 25 close). In the 6 cases with FC 13 or above the storm raises the deep layer by only 0.03–0.06 points, so the January mismatch exists only for FC 9–12. The report says so rather than carrying over the earlier study's blanket statement, and it makes no probe-fault claim.
- **Water:** bottom drainage is 5.3 mm at FC 9 and zero from FC 11–12 upward at stress 0.2017. It is 9.4 mm at FC 9 with stress 0.30255.
- **Framing:** the report treats field capacity as a drainage threshold, not a storage ceiling. It states that the gravity step is not limited by K, and that Tietjen K, d and Ks are unchanged. The stress provenance, the 2013 megapit as a spatial proxy, the conditional transpiration caveat and the storm-only scope of the 900 s controls are all included.

I had no Office renderer, so pagination is unchecked. The structure is checked instead: tables have at most 7 columns, grid and cell widths match, and DOCX tables equal the CSVs cell for cell. I viewed figures 1, 5 and 7 as PNGs.

Files are in `analysis_candidate/docs/subsurface_hydrology/jorn_fc_sensitivity/` (the `report_author_report.md` is in the task directory):
- `SYRUP_JORN_160day_uniform_FC_sensitivity_20261120.docx`
- `report.md`
- `build_report.py`
- `verification.md`
- `artifact_manifest.json`
- `figures/`
- `tables/`
- `data/`