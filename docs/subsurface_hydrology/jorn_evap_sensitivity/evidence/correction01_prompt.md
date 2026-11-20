# Root report review — bounded corrections only

Reportstage closed0. Root confirms numeric tables, all218seals, Word/CSV/media match so far; source/model/results unchanged. Correct prose/status only and rebuild. Solewriter NEW report docs, no model runs, no generator/config/results edits.

1. Summary headline "No tested setting fits better than baseline" overstates: baseline has lowest FULL160DAY ALL-DEPTH POOLED RMSE, but shallow E.8 and early upper-two .6/.8 improve. Make headline explicitly full-profile/full-period; don't imply no improvement anywhere.

2. Summary effective-coefficient monotonicity direction is REVERSED. Actual coefficient goes1→.111, actual E falls and pooled error RISES. Current text says ordered .111→1 error rises/Efalls. Fix direction; Figure6 actualpoints okay. Derive/assert textualorder from same sorted sequence.

3. "Above-saturation stress only slows the fall of demand with drying" ambiguous/incorrect for our interpretation: at given moisture it reduces demand by constant1/r² in this regime, exponent remains2. Say amplitude changes, no new storage or altered drying-law exponent. Params table header says SAT46.4-46.5 but bottomSAT45.86; because E appliesTOPonly, label relative to TOP-layer saturation46.43%, not wrongprofile range.

4. Root restart receipt EXISTS and is PASS: root_restart_verification.json at taskroot, written22:56:19UTC (S3day80freshprocess, rows/ledger/scores/historycontinuation exact); previous reportpending wordingoutdated. Read and verify it plus NEW root_restart_seal.json (separate late-file seal; original218manifest remainsimmutable). Include it as DRIVER COMPONENT reconstruction only, not nativecheckpoint/coupled acceptance, no newrun. Copyreceipt/seal todata withhashes, update verificationstatus. No guessing/pending left.

5. DeepJanuary filling is practicallysame, not IDENTICAL. Root raw percentages atJan23stormclose: baseline10.90516048%, s3 10.90518308%; Jan25model10.900428-10.903081%, observed6.34%. Write "about10.91%" and "about10.90% at Jan25", "essentially unchanged", not identical or awkward10.90-10.90 range. The mismatch stands; usercorrect that predictedwholebucketwettingincludes25.8sensor. No sensorfaultassumption.

6. Clarify actualE/PET1.6%means160-day summed ratio forstress3 versus3.8%baseline, not instantaneousratioatenddate. User's Marchcontext is assumed onset of TRANSPIRATION, not "March onset of drying". Update that wording. Keep inferenceconditional; no phenologymeasured/implemented.

7. Explicit finer900scontrols refineSTORMsteps ONLY; dryintereventEremainsdaily. Don'tclaimdryEtime-resolutionqualification. Everything scientificallyfixed.
8. Bibliographic citationsnumericcorrect; currentcutdate/effectivenames/endpointtable good. Keep allsame-timestampcomparisons and 13-or-fewer readabletables≤7cols. Ruffclean.

Root has own verifier adapting to your new manifest.files schema and the correct .00023 timestep number; those are Root harness updates, not report/model errors. All modeldata/numericresults fixed.
RebuildDOCX/Markdown/CSV/manifest/verification; preserve olddrafts and all218seals. Save correction01_author_report.md, stop forRootfinalPNG/provenance/numericchecks/publication.

