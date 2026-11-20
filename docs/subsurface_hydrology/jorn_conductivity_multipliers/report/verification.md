# Verification record

Before building, every sealed file used was re-hashed against `root_run_seal.json`. From the actual rows of all five runs the builder recomputed: per-depth and pooled RAW, aligned and no-change RMSE/bias/counts (matches Root's comparison.json), budgets, all-zero runoff/percolation/drainage/ponding/onset counts, the 900 s maximum differences, NumPy = Numba row equality for 2x/3x/4x, maximum modelled water content below field capacity, the pooled improvement percentages, and the exact configuration difference of each scaled case from the original (ten numeric entries x the factor, nothing else). Static DOCX checks: zip integrity, one media file per figure (4), table count equals header-row count, footer page field, no placeholder or DRAFT text. No Office renderer was available, so rendered pagination and legibility were not checked by the builder.

```
{
 "zip_ok": true,
 "media": 4,
 "tables": 4,
 "header_rows_repeat": 4,
 "paragraphs": 286,
 "words": 2406,
 "bytes": 665745,
 "sha256": "632d4556902d29cd759c642de457fc050ee801b911f291d4a9892485b43b94af"
}
```
