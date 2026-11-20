# Verification record

Before building, `root_report_ready.json` was required to say PASS and all files of `root_run_seal.json` were re-hashed; they were re-hashed again after the document was built. From the daily rows of all fourteen cases the builder recomputed per-depth and pooled RAW, aligned and no-change RMSE/bias/counts, the before/after-March-1 period scores (nominal day label < / >= 2024-03-01), the same-timestamp endpoint values, the January 23 deep-layer response and its peak, water budgets and closure, the zero runoff/ponding/onset/room-rejection checks, NumPy = Numba row equality, the 900 s maximum differences, the before/after source pins of all 42 runs (SYRUP 4e38a472.../180, MAPLE fbc504bb.../412), Root's 90-digit controls, the sensor values against Root's independent raw-file audit, the exact uniform field-capacity and stress values in every case file, unchanged conductivities, saturation and physics, and the unrescaled initial storage of 22.9424 mm; all agree with Root's audit files. Static DOCX checks: zip integrity, one media file per figure, table count equals repeated-header count, footer page field, no placeholder text. No Office renderer was available, so rendered pagination and legibility were not checked (figures were viewed as PNG).

Restart: Root's receipt `root_restart_verification.json` (PASS, 2026-11-20 23:25:53 UTC) was read and is covered by the re-hashed seal. Case fc12_s1p5 (field capacity 12 vol. %, stress 0.30255) was split at nominal day 80 and resumed in a fresh process; exact equality of rows, ledger, cumulative_closure, matrix_regime, scores, physical_end_t_s, added_rain_hours, nominal_elapsed_s, complete, days_done is reported. Scope as stated by the receipt: Field-driver public component state reconstruction, not production checkpoint or coupled-simulation acceptance. It is a driver-level component state reconstruction only, not a native checkpoint or coupled-model acceptance, and no new run was made. The receipt is copied to `data/`.

```
{
 "zip_ok": true,
 "media": 7,
 "tables": 12,
 "header_rows_repeat": 12,
 "paragraphs": 975,
 "words": 5109,
 "bytes": 1455010,
 "sha256": "4257d5b8611b01cf94e48704acfd23bb4f057e2ab7dadda8f9aee6c46dde5927"
}
```
