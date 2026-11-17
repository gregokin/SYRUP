# Lint author report — exact nine Ruff findings, style only

Claude 32819cf0, 2026-11-17. Sole writer of the four named files in `analysis_candidate`; nothing else touched (no frozen manifest, actual outputs, configs, protocol, data, GPU, Git). Uncommitted. Baseline for the diff: Root's `executed_helper_before_lint/`.

## Findings fixed (Ruff on `benchmarks/subsurface_site_comparison` and `tests/subsurface_site_comparison`: 9 found → "All checks passed!")
1. `compare.py` F401: removed unused `WINDOWS` import.
2. `compare.py` PLC0206: skill loop is `for k, hv in sc_h.items()` (same order, same `sc_h[k]` value, same subtraction via `_delta(hv, sc_g.get(k))`).
3–5. `interfaces.py` RUF007 ×3: `from itertools import pairwise`; the edge-order check, the thickness list and the weight list use `pairwise(edges)` / `pairwise(h)` (identical pairs, order and float operations).
6. `study.py` F401: removed unused `typing.Any`.
7. `study.py` PLR0124: `v != v or abs(v) == float("inf")` → `not math.isfinite(v)`, placed after the existing bool/numeric type guards (nan, ±inf still refused; the message is unchanged).
8. `test_site_comparison.py` RUF100: removed the unused `# noqa: E731`.
9. `test_site_comparison.py` PIE810: `name.endswith(("_training_harmonic.json", "_TEMPLATE.json"))`.
Diff line counts against the executed copies: interfaces 7, study 3, compare 6, test 4 changed lines.

## Verification (CPU, sealed runtime prefix, `CUDA_VISIBLE_DEVICES=` empty, existing MAPLE .venv; no water/GPU runs)
* Ruff on the two new directories: all checks passed.
* `pytest tests/subsurface_site_comparison`: 57 passed.
* `study generate` into a temporary directory is byte-identical to all seven committed config templates/coefficient file, so config bytes are unchanged.

## New hashes (sha256)
`interfaces.py 02dfd1bd6f76d14d3a8d13ea7a175e02c71746fa744a7f7dabbf03dfa5f2c09b`, `study.py 5fea3baaa92b4c27d95ba5a16690b4f83c227bd856d16d716c6f82db4e43a056`, `compare.py 460efec5bba599efdca029f243e7500d86c1bc82a2bbfb736d1b8fe29ab7c1e2`, `test_site_comparison.py 02aa4785ad5af41590d9f1a3fd2fd8f8ef0bee313c4b3ab825f3120cfe5a33ab`.

These differ from the helpers that actually executed the water runs (`frozen_helper_manifest.json` and `actual_*` remain historical; not relabelled). I did not re-run the source-guarded comparisons or regeneration against the actual results; Root does that. STOPPED.
