# Verification record

Before building, all 120 files of `root_run_seal.json` (including the old 1x/2x/4x results used for comparison) were re-hashed; they were re-hashed again after the document was built. From the daily rows of all seven runs the builder recomputed per-depth and pooled RAW, aligned and no-change RMSE/bias/counts (matching Root's `root_result_audit.json` to relative 1e-9 for the four new runs and the old 1x run), the water budgets and closure by rain/intake/E/drain/final storage, gravity and diffusive transfers, the zero runoff/ponding/onset/room-rejected checks, NumPy = Numba equality of all rows, the 900 s maximum differences, the before/after source pins (SYRUP 4e38a472.../180, MAPLE fbc504bb.../412), Root's 90-digit controls, and the sensor values against Root's independent raw-file audit. Static DOCX checks: zip integrity, one media file per figure (4), table count equals repeated-header count, footer page field, no placeholder text. No Office renderer was available, so rendered pagination and legibility were not checked by the builder (figures were viewed as PNG).

`root_restart_verification.json` is a Root receipt written after the seal; it is copied (SHA in the manifest) but not part of the 120-file seal.

```
{
 "zip_ok": true,
 "media": 4,
 "tables": 13,
 "header_rows_repeat": 13,
 "paragraphs": 501,
 "words": 3842,
 "bytes": 790256,
 "sha256": "5e6f1d96f449330221b727650f861cd747de6fcede2a15a777d4722065e27760"
}
```
