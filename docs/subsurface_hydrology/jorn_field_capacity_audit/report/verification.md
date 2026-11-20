# Verification record

Before building: every file in root_audit.json['source_sha256'] (raw NEON CSVs, archived receipt, native provider source, original case_numba.json, Saxton-Rawls PDF, published reports) was re-hashed; the builder then recomputed all three layers in 60-digit Decimal arithmetic from the raw NEON CSVs with its own transcription of Saxton-Rawls Table 1 and asserted agreement with Root's 80-digit values (1e-45), the layer mixture, the archived receipt and the original run file (2e-13), and recomputed the published Table 3 sandy examples. The quoted R2 0.63 / standard error 0.052 for the 33 kPa regression was read from the text extraction of the hashed Saxton-Rawls PDF (probes/pdf_raw_text.txt, Table 1 row for theta33: "0.63/0.052"; also 0.36/0.063 and 0.78/2.94 present) and asserted by the builder. Static DOCX checks: zip integrity, one media file, table count equals header-row count, footer page field, no placeholder or raw-LaTeX text. No Word/LibreOffice renderer was available: page layout and equation appearance were not checked visually.

```
{
 "zip_ok": true,
 "media": 1,
 "tables": 5,
 "header_rows_repeat": 5,
 "paragraphs": 289,
 "words": 2961,
 "bytes": 142617,
 "sha256": "d6f0f6ed8a33f0b7529f9bccb48eca921c01bee0705d0443e790a1340c20ba33"
}
```
