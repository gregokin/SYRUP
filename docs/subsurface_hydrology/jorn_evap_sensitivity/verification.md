# Verification record

Before building, all 218 files of `root_run_seal.json` were re-hashed; they were re-hashed again after the document was built. From the daily rows of all eight cases the builder recomputed per-depth and pooled RAW, aligned and no-change RMSE/bias/counts, the before/after-March-1 period scores (nominal day label < / >= 2024-03-01) and counts, the same-timestamp endpoint values, water budgets and closure, the zero runoff/ponding/onset/room-rejection checks, NumPy = Numba row equality, the 900 s maximum differences, the before/after source pins of all 24 runs (SYRUP 4e38a472.../180, MAPLE fbc504bb.../412), Root's 90-digit controls, the sensor values against Root's independent raw-file audit, exact reproduction of the accepted SR33 1xK baseline, and the monotonic ordering of actual E and pooled error with the effective coefficient E/r². Static DOCX checks: zip integrity, one media file per figure, table count equals repeated-header count, footer page field, no placeholder text. No Office renderer was available, so rendered pagination and legibility were not checked (figures were viewed as PNG).

Root's restart receipt (stress x3, split at nominal day 80, fresh-process resume) was read, its seal (`root_restart_seal.json`, 11 files, separate from the immutable 218-file seal) re-hashed before and after the build, and the resumed rows, ledger and scores compared with the continuous run: exact. It is a driver-level component reconstruction only, not a native checkpoint or coupled acceptance.

```
{
 "zip_ok": true,
 "media": 6,
 "tables": 11,
 "header_rows_repeat": 11,
 "paragraphs": 621,
 "words": 4072,
 "bytes": 1150552,
 "sha256": "8c74728feb30f1e1a3a56e7ae3433b136cd7beb0eb250459eb54556575e3c6dc"
}
```
