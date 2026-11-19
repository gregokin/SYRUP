# Verification record

Before building, every sealed file used was re-hashed against `root_run_seal.json`; the 282 per-depth/pooled RAW, aligned and no-change RMSEs, biases and counts, the water budget, step counts, clocks and all-zero ponding/runoff/drainage were recomputed from the actual rows and matched Root's audit. Static DOCX checks: zip integrity, one media file per figure (4), table count equals header-row count, footer page field, no placeholder text. No Office renderer was available, so rendered pagination and figure legibility were not checked by this builder.

```
{
 "zip_ok": true,
 "media": 4,
 "tables": 7,
 "header_rows_repeat": 7,
 "paragraphs": 285,
 "words": 2838,
 "bytes": 511782,
 "sha256": "c590ec2753e7b4304696f193720b97a539604238849f31521457c5d1be24bd6b"
}
```
