# Phase 1.5 daily recharge: bounded public qualification (source N15)

Status: **bounded whole-phase acceptance recorded (`ACCEPTED_BOUNDED_NATIVE_DAILY_RECHARGE`, [`phase15_acceptance.json`](phase15_acceptance.json)); source commit `06952dbc2f984af8fb75dbf09a9c9f7fdbba23fc` is committed and pushed on `feature/subsurface-daily-recharge`.** Two Root records are distinct: the earlier S4 receipt (`phase15_native_acceptance.json`) accepted the model's native daily lifecycle and V3 physical restart on 82e9…; the final whole-phase receipt additionally records Root's independent review of the final examples, documentation and CPU/CUDA example runs. Neither is a field validation. Publication is recorded separately in [`phase15_publication.json`](phase15_publication.json).
This record identifies the tested sources, reviewers and limits. Binding specification:
[`phase15_daily_recharge.md`](phase15_daily_recharge.md); plan: [`development_phases_20261109.md`](development_phases_20261109.md).

## What was delivered

A native Python capability, not a command-line option, weather/CSV reader, daily scheduler or LEAF change:

- `maple_syrup.recharge` (plan validation, bound forcing, daily cursor, 2^63-1 record limit, clock contract `maple_syrup.daily_clock/1`);
- `maple_syrup.daily_water` (`start_daily_lineage`, `next_plan`, `daily_water_action`, `begin_daily_storm`, `finish_daily_storm`, receipts, typed
  `DailyRefusal` codes `entry, forcing, memory, occurrence, router_step_rejected, state, step_refused, storm_incomplete`);
- the shared column body `subsurface.step_with_recharge` (field capacity triggers gravity movement once; saturation limits storage; capacity failure of any
  active column discards the trial and the whole occurrence is rerun as the full native storm from the unchanged original state with the full P);
- CPU and CUDA backend gates, a true-parent engine permit in `water_lifecycle` (old public transitions refuse daily states);
- V3 physical checkpoints: CPU `lifecycle_checkpoint_v3` (profile `native9f_dense_lifecycle_cpu_v3`) and CUDA `lifecycle_checkpoint_cuda_v3`, with canonical
  plan, occurrence and parent typing and a 63-bit record-count bound. Older C1 V1/V2 CUDA helpers keep their pins and **refuse** N15; a genuine frozen C1 V2
  fixture is only a wrong-version control, not a migration source. No stale `SourceSetup` claim is made.

Selection rule (user decisions of 2026-11-12 UTC): P = 0 no-op (one soil/E day still runs); 0 < P ≤ T attempts recharge (equality included); P > T full native
storm, duration 3600 P / I seconds. T has no default. The nominal ordinal is separate from native physical seconds (new versioned contract; the old dated clocks
and checkpoints are unchanged). Recharge never resets matrix wetting history, J or cumulative intake; modeled matrix/preferential intake is reported separately.

## Source identity

| Item | Identity |
|---|---|
| N15 SYRUP package | `82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107`, 114 files |
| File manifest | `root_s4_correction02_source_manifest.json`, SHA-256 `5cb1b01a7c76b3bb07a8bdc13cb698a65abf96f17f2d12260b6cbddadf8a4986` |
| Native MAPLE | historical `9f466d2cfcd52991b24496cf67f4562d6268afa9`; package `fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e` (actual materialized native Dense bundle, not latest main) |
| Root S4 acceptance record | `root_s4_acceptance.json`, SHA-256 `5d1b991b1ba244e053e4409664dce9c19d50d373d54eef37f0e7fae505c1da08`, status `ACCEPTED_BOUNDED_NATIVE_DAILY_PHYSICAL_RESTART` |
| Historical C1 foundation (unchanged, never rebound) | SYRUP `e40a1b8`, water package `afcad536…`, LEAF `52f39fe`, MASY `a704002` |

This documentation changes no production source: the package digest above was re-verified by the S5 author after the documentation edits.

## Shipped, portable evidence (readable from a clone)

| File | Meaning |
|---|---|
| [`phase15_source_manifest.json`](phase15_source_manifest.json) | the exact scoped 114-file manifest (size and SHA-256 of each `src/maple_syrup` file), byte-identical to Root's `root_s4_correction02_source_manifest.json` (file SHA-256 `5cb1b01a…`) |
| [`phase15_acceptance.json`](phase15_acceptance.json) | the immutable Root whole-phase acceptance receipt (file SHA-256 `4ce37b57f4e009caeb8eb99d30df52ce197bb1c3b4a0947f257b21f9db90deb3`), byte-identical to `root_phase15_acceptance.json`. It binds the 114-file source, the example programs and the scientific contract, **not** the regenerated manuals, metadata or archive (to avoid a hash cycle; those hashes are in the build and archive records) |
| [`phase15_native_acceptance.json`](phase15_native_acceptance.json) | the Root bounded S4 physical acceptance receipt, byte-identical to `root_s4_acceptance.json` (file SHA-256 `5d1b991b…`). It names private task paths for its raw logs; those are additional evidence, not the only record |
| [`verify_phase15_source.py`](verify_phase15_source.py) | `python verify_phase15_source.py CHECKOUT_ROOT` compares the checkout's files with the manifest and prints the package digest; `PASS` only if all 114 hashes and the digest `82e9c011…` match. Author run on this candidate: PASS; with one byte-changed file: FAIL (one `differs`, digest differs) |

Publication is staged and not self-referential: the first phase commit holds the accepted code and both manuals; the metadata field `native_phase15.source_commit` stays null and `published` false until a later documentation-only refresh records that real commit after the actual push.

## Independent controls (Root-run; per source, not summed)

Runs on *different* source snapshots overlap and are not added together.

| Control | Source | Result |
|---|---|---|
| Timing correction, configured MAPLE 9f runtime | `a8f72639…` | 222 passed, 40 deselected, 29.97 s (CPU). Reopened and superseded for forcing-origin by the double-origin fix. |
| Native NumPy/Numba, 1/2/3 layers, recharge/P=0/full completed storm/daily/T+close/later storm | `0a6397b1…` | 30 physical arrays each at the existing 2e-12/1e-14 |
| Actual CUDA native component | `0a6397b1…` | 29 passed, 74.15 s (idle assigned GPU0); CPU affected 36 passed, 70.85 s |
| GPU vs NumPy/Numba continuation | `0a6397b1…` | 1/2/3 layers × 5 stages × 6 physical fields = 90 array comparisons agree at 2e-12/1e-14; exact clocks |
| CPU restart (V3) controls | correction01 snapshot | 42 passed, 203.90 s |
| Final-source CPU metadata/fixture/source-guard/plan controls | `82e9c011…` | 21 passed, 25.78 s |
| Fresh-process NumPy and compiled Numba, uninterrupted vs partial/resume across four later stages incl. completed storm | `82e9c011…` | exact |
| CUDA V3 first group | pre-final | 16 passed (15 fresh-process pauses + scenario control) |
| CUDA V3 tail (tamper/plan/source/counter/duplicate/low host+device) | pre-final | 5 passed, 17 deselected, 46.70 s |
| CUDA wrong-version genuine C1 V2 fixture + fresh partial/after-T-close resumes | `82e9c011…` | 3 passed, 61.58 s |
| Old C1 CUDA helper pin refuses N15 before archive publish | `82e9c011…` | 1 passed, 4.96 s |

The only reviewed change after the first GPU run was an explicit 1-byte partial-false metadata allowance in `lifecycle_checkpoint_cuda_v3_plan.py`. No known open
confirmed defects were recorded by Root at S4 acceptance.

## Author examples (CPU only, corrected in S5 correction01)

`docs/manuals/examples/native_daily_recharge/` runs the branches with the existing MAPLE environment, using the real NCP/admission/Dense/layered factories and
explicit illustrative P/I/T (T is not a recommendation). Author runs are not Root controls and no GPU was launched or probed by the author.
The script now requires a fresh work directory (no deletion or overwrite; `check_preservation.py` shows refusals on sentinel files: 3 refusals, all bytes identical) and uses the real `subsurface.default_host_available_bytes` reader with explicit caller quotas. Results are in its `results/`; all closure residuals are below the printed MAPLE floating-point bound.

| Scenario | Backend | Outcome |
|---|---|---|
| recharge (open, θ=0.20) | NumPy | recharge branch, 1 day, drainage 0; V3 save/load and later full storm ran |
| open_drainage (θ=0.30) | NumPy | recharge, drainage 0.012375 m³ |
| closed_retention (θ=0.25) | NumPy, Numba | recharge, drainage 0; NumPy and Numba ledgers equal |
| fallback (closed saturated flat owned case, P=1 mm, I=360 mm/h, T=2 mm) | NumPy, Numba | capacity failure in 24 active columns, whole-P storm, one day, V3 and later storm; ledgers equal |
| fallback with PET 1e-8 m/s | NumPy | typed `router_step_rejected` refusal (CFL 28720.67 > 0.5), state unchanged — reported, not hidden |

### Output audits and Root's independent CPU example checks

The example summary separates `first_day_audit_before_T` from the final `closure` (native audit of the final accepted state: completed later storm, or the closed boundary after T/checkpoint), so T and the later P are each counted once; the stored results use a positive T of 1e-6 m (accepted 1.5e-6 m³ over the 24 active cells) and `check_results.py` verifies rain = first-day rain + later-storm rain, T once and |residual| within the native bound. Root independently ran the earlier snapshot (four NumPy scenarios, fallback on Numba and four CUDA scenarios, each with V3 checkpoint, a positive T and a later full storm) and all passed; after this reporting-only correction Root re-ran a final snapshot: NumPy recharge, Numba fallback and CUDA recharge, all with checkpoint, positive T, later storm and the final audited closure (recharge: rain 0.0045 m³, T 1.5e-6 m³, residual 1.8e-17 m³ against a bound of 1.45e-10 m³ on NumPy and CUDA alike; fallback on Numba rain 0.003 m³, T 1.5e-6 m³). `check_preservation.py` passed its three refusals with sentinel content unchanged, and Root's static review checked the Word packages (XML/CRC), source identities, the four authentic MAPLE metadata files and the exact shipped manifest and acceptance copies; no whole-page Office rendering was performed. The GPU runs are Root's; the author of this documentation ran no GPU code.

## Limits

Genuine 4×6 Dense native fixtures with fixed layer geometry; no windowed, large-domain or field qualification. Retained quiet water (≈1e-9 m) on a saturated closed
profile can make the one-day router step refuse with CFL > 0.5; it is neither substepped nor loosened. General repeated-storm wetting-front recovery is
unqualified and nothing resets it implicitly. Native API and physical restart only: no complete daily MASY weather scheduler/CSV, no LEAF redesign. The
recharge is a user-chosen approximation and not exact Tietjen or MAHLERAN infiltration. Phases 2, 2.5 and 3 are unchanged and not started.
