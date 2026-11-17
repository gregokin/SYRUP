# Verification (static)

- DOCX ZIP integrity OK; 6 tables (expected 6); media ['word/media/image1.png']; PAGE field in footer: True; styles part present; no LaTeX/equation markup.
- About 4056 words; 128688 bytes. Page count NOT measured (no Office/renderer available), so the 5-8 page target is unconfirmed.
- Every worked number is recomputed from saved JSON in the builder and asserted against the task values (old/new totals, fractions, first-day PET and moisture, first-day trajectory evaporation, zero drainage/percolation/return).
- No model, GPU or Git command was run. Original report, manifests, data and figures untouched.
- Original-2009 facts (a_ET, d, wsc) come from the publisher full text recovered by Root (see evidence/provenance.json); 168 words attributed to the 2009 paper in total across body, matrix, estimation note and bibliography (asserted at most 200); no long quotations. The publisher text was used, not the original PDF or code; the 2009 optimiser and Equation 9 donor choice are not claimed.
- First-day arithmetic for all six old/new window cases and the redistribution illustration are asserted against Root's root_numeric_checks.json and root_d_example_check.json.
- Header rows repeat on all 6 tables; every table cell is non-empty; no caret or exponential formula text in the DOCX; figure is 1 PNG.
- NOT verified: Office layout/pagination; original 2009 PDF/code and optimiser details. No publication claim.
