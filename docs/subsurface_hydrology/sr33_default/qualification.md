# SR33 default field capacity — independent qualification

Status: independently accepted bounded native API change, ready for scoped publication.

Scope: source for the final density-adjusted Saxton–Rawls 33 kPa field-capacity default (neutral output `theta_field_capacity`; explicit deprecated `tietjen_6kpa_deprecated` = true 6 kPa). Tietjen K,
`d` (a dimensionless calibration coefficient), stress, residual, saturation unchanged. Air entry is only the retained strict guard/diagnostic in the new mode. Initial water inventory is unchanged by the FC choice;
drainage happens at the next actual update/event-boundary reconciliation. Small rain = uniform recharge + FC cascade (saturation limits storage); matrix uses `max(FC*h-W,0)`.
Conductivity: EcoHyD representative Ku 12.0886 (intermediate) -> effective within-layer K 38.2117 -> neighbor-averaged interface K 38.0403 (JORN example); SR Ku is an unadopted comparison.

## Author CPU evidence (author-run only) (genuine MAPLE 412/fbc + candidate source; logs in `logs/`)
provider 237 passed/12 skipped; native 53; held-context 310; resume 62 (+new continuation control: a real wet checkpoint capture/restore/advance, NOT a completion proof);
closed-lake fixtures (correction03): `phase2_floating_checkpoint` 41 passed, `phase2_floating_completion` 11 passed/2 skipped (GPU), `phase2_floating_context_checkpoint` 29 passed.
Actual CUDA hardware controls are Root's, not author claims. No three-model coupled, scaling or observed-site field acceptance.

## Completion fixture (geometry + test window only)
Certificate fixture = Root-proven isolated pond: same basin, `z[1,1]=z[1,3]=0.14`, exactly one enclosed wet cell [1,2]; saturation `nextafter(1,0)`; 0.1 kg mobile, policy, tolerances, event code unchanged.
First quiet ~3 s, terminal deposit (1 native call, `requires_recession`), second recession actually quiet ~125 s (`stop_when_quiet=True`, quiet asserted, below the 600 s TEST window), empty second settlement complete (still 1 native call).

## Negative control and follow-up (IMPORTANT, unresolved)
`test_original_communicating_lake_remains_not_quiet_and_settlement_is_refused_typed`: the original three-wet-cell lake at 0.999999 is not quiet after 60 s; settlement raises `PondEventIncomplete`; accepted state unchanged.
Correction02 evidence (`correction02_author_report.md`, `diag_quiet.py`, `diag2-5.py`): with the communicating lake, a post-deposition surface oscillation (unit discharge ~1e-7 m2/s, depth +/-5e-7 m) did not damp in 4800 s,
after a 3.2% layer-0 water deficit at the deposit cell (baseline 0.8%) was refilled by slow infiltration; not mass-driven. This is a surface-hydraulic qualification follow-up outside this patch; no damping or quiet-criterion change is adopted.

## Historical-only
Frozen package-digest guards (`phase25_*` `UnknownSource`) still refuse the new source; no source-guard rewrite, no claim that an old profile is qualified.

## Root independently verified evidence (separate from the author evidence above; Root-run, not author claims)
* CPU suites: provider 237 passed / 12 skipped; native 53 passed / 1 skipped; held-context 311 passed; resume 63 passed / 1 skipped = 664 passed, 14 honest skips.
* Root independent CPU controls: 10; actual CUDA controls: 17; actual native CUDA Dense/Windowed refresh/adoption tests: 14 passed.
* All used the exact production package 180 files / `4e38a472e268a56491b788d5b1271cc3c3da23bb1cc3221a8c59d90f3cc864d2` and the genuine MAPLE 412 files / `fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e`, with before/after source guards.
* Final Root checks: checkpoint **41 passed**, CPU completion **11 passed / 2 hardware-only skips**, context checkpoint **29 passed**, actual CUDA completion **2 passed** (Dense and genuine Windowed). All source guards passed. Total unique Root pytest results: **745 CPU passed / 16 skipped**, **16 actual CUDA passed**; additional **10 independent CPU controls and 17 actual CUDA controls**. Duplicate author/style reruns are not counted.
* Final test-only style cleanup was rechecked: provider 237 passed / 12 skipped; held identity and wet continuation controls each passed. Ruff introduces no diagnostics (52 pre-existing versus 53 baseline).
* Exact commands, outputs, before/after guards and source manifest are archived in `evidence/`; machine-readable qualification is `acceptance.json`.

## Compatibility
Old source-locked runtime capsules intentionally require separate new qualification and are not rebound by this patch. This patch accepts the native APIs/components only; it does not accept the MASY three-model
runtime or the historical JORN drivers. No observed-site rerun, refit or scaling claim is made.
