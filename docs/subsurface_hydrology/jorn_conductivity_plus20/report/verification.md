# Verification record

Before building, every sealed file used was re-hashed against `root_run_seal.json`. From the actual rows the builder recomputed: the per-depth and pooled RAW, aligned and no-change RMSE/bias/counts (matches Root's comparison.json), the budgets, the all-zero runoff/percolation/drainage/ponding/onset counts, exact NumPy = Numba row equality, the 900 s maximum difference, and the exact configuration difference (ten numeric entries x 1.2, nothing else). Static DOCX checks: zip integrity, one media file per figure (3), table count equals header-row count, footer page field, no placeholder or DRAFT text. No Office renderer was available, so rendered pagination and legibility were not checked by the builder.

```
{
 "zip_ok": true,
 "media": 3,
 "tables": 4,
 "header_rows_repeat": 4,
 "paragraphs": 196,
 "words": 2133,
 "bytes": 413092,
 "sha256": "1c8de4489e775eb92c6e43b31030f8a96de593746c3e90a3bc1a99b4626c6e9e"
}
```
