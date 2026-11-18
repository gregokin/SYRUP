# Verification record

Static checks of the generated DOCX (zip integrity, one media file per figure, table count equals header-row count, footer page field, no placeholder text). No Office renderer was available, so rendered pagination and figure legibility were not checked by this builder.

```
{
 "zip_ok": true,
 "media": 5,
 "tables": 10,
 "header_rows_repeat": 10,
 "sections": 5,
 "paragraphs": 443,
 "bytes": 995655,
 "sha256": "32cac33e1353489f4d843c477e0d0ed32257b9b26a6f2bdde6a123044fced95b"
}
```

Reported values are asserted against the frozen data in `check()`: selected stresses, pooled RMSEs, zero-change baseline, counts (53; 28/11/14), 208 evaluations and one interior minimum per fit, 624 objective evaluations and 1872 window runs, and the aligned = raw + offset identity for every sensor.
