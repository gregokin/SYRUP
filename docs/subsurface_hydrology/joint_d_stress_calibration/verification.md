# Verification record

Static checks of the generated DOCX (zip integrity, one media file per figure, table count equals header-row count, footer page field, no placeholder text). No Office renderer was available, so rendered pagination and figure legibility were not checked by this builder.

```
{
 "zip_ok": true,
 "media": 4,
 "tables": 8,
 "header_rows_repeat": 8,
 "sections": 3,
 "paragraphs": 456,
 "bytes": 738954,
 "sha256": "aae6c2ac93811231b11362396a5414a6e4a989d43f48bf3307057361a2fb5c23"
}
```

`check()` re-verified before building: every frozen data hash against the task's frozen manifest, the previous study's fit files against its own manifest (itself pinned to the immutable e11 sha256), and the reported numbers (joint optima, RMSEs, 53 observations and 28/11/14 split, 5103 + 5017 = 10120 pairs, 30360 window runs, 24 full reruns, 8 successful refinements per fit, per-window and per-site RMSEs, profile excess at d = 0, grid near-region extent below the refined d, and the aligned = raw + offset identity for every sensor). New selected CUDA cases were NOT run (GPU 0 occupied, GPU 1 not authorized).
