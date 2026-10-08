# Legacy sediment replay (CPU Numba, CUDA, original-Fortran reference): current status at the commit of 2026-10-08

This page is the single current statement for the work documented in `docs/legacy_native` (A1, CPU Numba driver), `docs/legacy_sediment`
(A2, original-routine Fortran harness) and `docs/legacy_gpu` (B1 to B3, resident CUDA driver). The other pages keep their executed history;
where they say a run is "in progress" or "pending", this page supersedes them. Everything below was produced by the root (Codex) and
read from the evidence files named in each section. Evidence lives under `agent_handoffs/tasks/gpu_sediment/`, which is git-ignored and
therefore NOT part of the committed tree; the committed docs summarize the results; raw evidence remains local. No full-storm benchmark or GPU test was rerun for this commit. The documentation author ran nothing;
Codex executed the CPU verification recorded below.

## 1. What is committed

* The CPU reference driver (`maple_syrup.legacy_driver`, modules `legacy_native`, `legacy_native_numba`, `legacy_case`, `chastre_case`).
* The resident CUDA driver (`maple_syrup.legacy_gpu_driver`, modules `legacy_native_cuda`, `legacy_water_cuda`, `routing_newton_cuda`)
  in its B3 (C1-corrected) state: compact walk records (`--record-strategy compact`, default) and fused water accounting
  (`--water-accounting fused`, default; `separate` is the reference arithmetic).
* The original-routine Fortran harness (`benchmarks/legacy_sediment/`), the CPU/GPU comparison harness
  (`benchmarks/legacy_gpu/compare_cpu_gpu.py`), the record-strategy microbenchmark and the kernel-event profiler, with their tests.
* The ORIGINAL benchmark helper `benchmarks/legacy_sediment/compare_legacy_sediment.py`, unchanged (see section 7 for its known defects).

NOT committed, NOT adopted (local experimental candidates, outside the tree or git-ignored):

* The P2 lazy flow-detachment-probability CUDA candidate (one changed `sg_laws` block in an isolated package copy under
  `outputs/dependencies/syrup_gpu_sediment_flowprob_candidate/`). Its test `tests/legacy_gpu/test_lazy_flow_probability.py` IS committed and
  skips when the frozen B3C1 snapshot or the candidate copy is absent.
* The corrected golden-helper candidate (`agent_handoffs/tasks/gpu_sediment/golden_candidate/compare_legacy_sediment.py`). Its test
  `tests/legacy_sediment/test_plot1_golden_candidate.py` IS committed and skips at module level when the candidate file is absent.

Adoption of either candidate requires its own regression and review; nothing here is a candidate acceptance.

## 2. Physical scope (unchanged)

MAHLERAN legacy replay for benchmarking only: fixed composition from the verified initial active layer, UNLIMITED supply, an explicit
artificial clipping source, fixed terrain and fixed routing, 1 s steps, no direct dry-cell splash (the wet-cell rain-assisted detachment is
retained), no ET, no dry reset, terminal pits keep their mobile mass, ring and inactive-cell deposition are diagnostics. No MAPLE bed is read
after case verification or written. The default wet-law depth is the previous routed depth with the new velocity; the
native Fortran uses post-infiltration depth. The explicit post-infiltration option is a diagnostic, not a silently changed default. It is NOT an evolving conservative MAPLE bed, NOT a complete-event model, has NO restart and NO wind
handoff, and its legacy behaviour is not MAPLE authority. Chastre has zero outlets, so its identically zero export is a vacuous export
validation; compare maps, per-class ledgers and terminal storage instead.

## 3. Executed backend gates and reference checks (bounds unchanged)

Sediment bounds rtol 2e-11 / atol 1e-14, water 2e-12 / 1e-14, identity guard 1e-10, integer tallies, masks and time axes exact. None was
changed or fitted.

| Gate | Result | Evidence (git-ignored task directory) |
|---|---|---|
| Full Plot 1 5400 s, both solvers, B2 and B3 modes, CPU Numba vs GPU | every saved field, snapshot and series within the unchanged bounds, 0 flags; integers exact; GPU repeats and `separate`/`fused` outputs bitwise | `plot1_full_*_matched_comparison.json`, `plot1_full_*_b3_modes_bitwise_comparison.json` |
| Full Chastre 2700 s, both solvers, same-root CPU Numba vs GPU (B2, B3 separate, B3 fused, B3C1 paired) | all saved fields/snapshots/counters pass unchanged bounds, 0 flags; integers exact; within-mode and between-mode GPU outputs bitwise | `chastre_full_*_matched_comparison.json`, `chastre_*_b3c1_paired_full_matched_comparison.json`, `b3_chastre_full_timing_summary.json` |
| P2 candidate vs committed B3 (Plot 1 and Chastre, both solvers, two runs each) | candidate passes the same CPU bounds; candidate repeats bitwise; candidate and baseline GPU outputs bitwise | `p2_plot1_executed_audit.json`, `p2_chastre_full_executed_audit.json` |
| Full Chastre original-Fortran reference vs both saved CPU Numba references | completed; strict schema/EOF/completion, pinned executable/input/15 sources, case artifacts and tile digest passed; see section 5 | `fortran_chastre_recovery4_results.md`, `recovery4_independent_result_qualification.json` |

Test suites recorded during the task (historical, NOT rerun for this commit): A1 C3 246 passed / 2 skipped; A2 C1 256 passed / 452 skipped
(450 belong to older candidate-specific tests); B3 C1 real-GPU 240 passed / 2 single-visible-device skips, CPU guards 89 passed / 20 GPU skips;
P2 CPU part 4 passed / 24 GPU-skipped, then 28 passed on GPU0 (3.47 s); golden-helper candidate 69 CPU tests passed; Ruff PASS in each. The
root reruns the available CPU suites and Ruff before committing.

## 4. Timings (exploratory, one GTX 1080 Ti as GPU0 under recorded background load; loop seconds exclude case load, startup, JIT/NVRTC and publication)

Committed B3 root, full Chastre 2700 s (the baseline runs of the paired P2 matrix, `p2_chastre_full_executed_audit.json`):

| Solver | B3 loop s, run 1 / run 2 |
|---|---|
| bisection | 290.581879 / 293.328463 |
| Newton | 267.052132 / 183.021517 |

P2 candidate, same matrix, NOT adopted:

| Solver | P2 candidate loop s, run 1 / run 2 |
|---|---|
| bisection | 271.173501 / 270.566621 |
| Newton | 246.100527 / 166.593027 |

The second Newton pair (183.02 and 166.59 s) was rerun after a host crash on a different date and background load from the first pair. These
pairs must not be averaged or quoted as a matched-precision speed-up; the only safe statement is that in each same-session pair the candidate
loop was shorter (about 6.7 to 9.0% per pair) with bitwise-identical outputs, and that this is exploratory.

Earlier committed-root Chastre loops under their own loads (B2: 278.701 / 278.543 s bisection, 257.795 / 258.132 s Newton; the B3 eight-run
fused/separate matrix: bisection 281.416 vs 282.011 s mean, Newton 259.571 vs 260.145 s mean) show no established fusion gain; the
run-to-run spread is as large as the mode difference. Plot 1 (5400 s) GPU loops remain SLOWER than the CPU Numba loop at that size
(`results.md` section 2).

Single-core CPU Numba full Chastre loops (immutable references, one run each): bisection 5736.873890 s, Newton 5673.070882 s.

Original-routine Fortran harness, full Chastre: loop 13839.809219 s (diagnostics 187.895 s, capture 0.684 s), kernel 13651.230445 s,
process 13843.071 s, peak RSS 3,273,724 KiB. This is the controlled 15-source-file harness with ordinary gfortran 13.3 on the source
kind-4/8 globals, NOT the whole MAHLERAN application, and its run differs from the CPU references in date, load and Python wrapper
(NumPy 2.5.3 vs the saved references' 2.5.2; the Fortran executable SHA is unchanged).

## 5. Original-Fortran vs CPU Numba, full Chastre (observations with 1% triggers, not fitted tolerances)

Totals (Fortran, Numba, Numba relative to Fortran):

| Quantity | Fortran | Numba | Difference |
|---|---|---|---|
| pickup, kg | 3,797,456.493 | 3,797,658.349 | +0.0053155% |
| active deposition, kg | 3,797,034.381 | 3,797,236.240 | +0.0053162% |
| pit deposition, kg | 3,926.2453 | 3,926.2493 | +0.000103% |
| clipping source, kg | 1,706.6721 | 1,706.6734 | +0.0000743% |
| final mobile, kg | 2,128.784403 | 2,128.782125 | -0.0001070% |
| final terminal inventory, kg | 289.494221 | 289.447568 | -0.0161% |
| first positive pickup | 193 s | 193 s | equal |

The automatic 1% flags are empty, but they cover class-summed totals, map sums and signs only. The root's independent per-class check found
gram-scale class-5 differences: pit deposition 3.129881 g vs 4.462241 g (+1.332360 g, +42.57% of Fortran) and clipping source 20.210661 g
vs 21.510074 g (+1.299414 g, +6.43%). Per-step mobile-series relative differences are large only at sub-1e-17 kg values. No blanket
per-class or per-time-sample equivalence is claimed.

Water (`recovery4_saved_snapshot_differences.json`): before rain ends at 2641 s the two hydrologies agree closely (2640 s surface volume
difference -0.000234 m3, maximum depth difference 1.12e-7 m). At 2700 s the Fortran surface volume is 28,588.151353 m3 against Numba
28,563.198926 m3, a native surplus of +24.952426 m3 (about 0.0873%), maximum depth difference 0.00115858 m. The derived native whole-storm
residual is -24.952426 m3 (a surplus), whereas the Numba residual is about -4.12e-9 m3 inside its unchanged MAPLE-derived bound of
2.49258 m3. Rain, soil and drainage integrals match. The discrepancy is confined to the recession after rain ends, but the responsible
original routing branch (stale-inflow, bracket or rain-off handling) has NOT been demonstrated. This is an open investigation; SYRUP's
closed water budget is retained and the suspected legacy defect is not copied.

## 6. Environments

Only the user-repaired MAPLE environments are used: `/home/okin/MAPLE/.venv` for CPU (NumPy 2.5.3, JAX 0.11.2, SciPy 1.18.1, PyYAML 6.0.3)
and `/home/okin/MAPLE/.venv-cupy` for GPU (NumPy 2.5.3, CuPy 14.2.0, JAX 0.11.2). There is no SYRUP-local virtual environment and no `/tmp`
package overlay. Numba and llvmlite are ABSENT from both (`repaired_gpu_venv_probe.json`), so the compiled CPU driver, the Numba oracle of the
GPU tests and every Numba-dependent test cannot run in the current environments; the recorded compiled-test results and CPU timings above are
historical, obtained when Numba 0.67.0 / llvmlite 0.49.0 were temporarily present. No fresh GPU test or benchmark was run for this commit.

## 7. Known limitations and open items

* Scientific: everything in section 2; the recession-water surplus of the original routines (section 5); zero-outlet export is vacuous.
* Candidates: P2 and the golden helper are not adopted. The committed original `plot1_golden` has two confirmed defects (it cannot parse the
  real application's omitted-`E` three-digit-exponent tokens, and it sums the mobile inventories over time under a `*_total` label) plus a
  silent reshape and quiet truncation of unequal comparisons; the saved real Plot 1 ledger was reviewed with the isolated strict candidate, not
  with the committed helper. Adoption, regression and the final review remain separate steps.
* Performance: no CPU-vs-GPU speed-up is claimed for Plot 1; the Chastre GPU/CPU loop ratios (about 20 to 22 to one against single-core
  Numba under concurrent load) are exploratory; no fusion gain; the P2 pairs are not a matched-precision speed-up.
* Record compaction is valid only for an immutable event composition; an evolving active layer can expose absent classes
  (`production_GPU_followups.md` in the git-ignored task directory).
* Chastre runs used `--hash-only-tile-verify`; post-run tile immutability is claimed only where `--hash-tiles-after-run` was used.
* Tests in `tests/legacy_native`, `tests/legacy_sediment` and `tests/legacy_gpu` are not in the default `testpaths`; several need the
  MAHLERAN reference tree at `/home/okin/MAHLERAN`, generated cases under `outputs/`, a configured gfortran, Numba or a CUDA device, and may skip or refuse execution when those prerequisites are absent.


## 8. Fresh commit verification (Codex, 2026-10-08)

No full storm or GPU test was rerun, and neither MAPLE environment was modified.

* Current feature suites: 437 passed, 607 skipped, 4 deselected in 19.43 s, using MAPLE's CPU environment and the verified
  Chastre MAPLE dependency. The four deselected tests explicitly require Numba, which is absent. Skips include optional
  compiled/device/reference checks; they are not fresh GPU or compiled-CPU validation. Three overflow warnings came from deliberate
  invalid-input probes.
* Corrected source-path invocation of the subprocess/import-contract suites: 118 passed, 2 skipped in 5.65 s.
* Golden candidate tests with the local, unadopted candidate present: 69 passed in 0.42 s.
* Source-only copy without ignored outputs/candidates: all test suites collect successfully, 4,473 tests collected in 5.76 s.
  The golden candidate module skips rather than raising a missing-file collection error.
* Ruff across source, benchmarks and tests, and Git whitespace validation: passed.
* Claude's bounded review found the missing-candidate collection defect (fixed) and no additional confirmed blocker. Candidate adoption,
  compiled/device regression and the native recession investigation remain separate work.

The initial broad invocation had 1,427 passes, 2,990 skips, 14 failures and 44 fixture errors. All were inspected:
nine subprocess failures lacked an exported SYRUP source path and passed after correction; one receipt assertion used live MAPLE instead
of the recorded Chastre patch and passed with that dependency; four tests required absent Numba. The 44 older Phase 7d fixture errors
explicitly reject a MAPLE revision differing from their pinned dependency. That older optimization suite was not requalified here.
The initial invocation is not reported as a passing full-suite result.

Commands for the passing feature checks (from the SYRUP root):
```bash
export PATH=/home/okin/MAPLE/.venv/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
source benchmarks/chastre/env.sh
python -m pytest tests/chastre tests/gpu_newton tests/legacy_gpu tests/legacy_native tests/legacy_sediment -q   -k 'not test_runguard_with_parallel_hash_detects_a_mutated_persisted_tile_like_the_original and not test_zero_outlet_water_storm_closes_its_budget_on_the_tiled_case and not test_runtime_guard_detects_a_modified_persisted_tile'
python -m ruff check src/maple_syrup benchmarks tests
git diff --check
```
