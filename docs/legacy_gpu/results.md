# Executed results of the legacy sediment GPU work (task gpu_sediment) — executed history up to documentation correction D1

**Current status: see [`current_status.md`](current_status.md).** This page is the executed history as of correction D1 (2026-10-06/07) and
is retained as historical evidence, not as fresh checks. Since then the full Chastre CPU (Numba) references and the full original-Fortran
reference have COMPLETED and been compared, the B3 eight-run matrix and the P2 candidate matrix have been executed, and the environments have
changed (Numba absent). Any "pending", "still running" or "in progress" wording below is superseded by `current_status.md`.

Scope and honesty rules. Everything here is the MAHLERAN **legacy replay** benchmark: fixed composition, unlimited supply, an explicit artificial
clipping source, frozen terrain and routing, 1 s steps, no splash/ET/dry reset, **no MAPLE bed read after case verification or written**, no conservation, restart, wind-handoff or
complete-event claim, and its legacy behaviour is not MAPLE authority. Numbers below were produced by the ROOT (Codex) from the evidence files named in
each row (all under `agent_handoffs/tasks/gpu_sediment/`); the code authors never ran anything. Sediment bounds are rtol 2e-11 / atol 1e-14, water
2e-12 / 1e-14, identity guard 1e-10, integer tallies/masks exact; none was changed or fitted. Timings are exploratory: one device (GTX 1080 Ti, GPU0, a
background memory allocation of about 855-867 MiB and utilization telemetry recorded), CPU and GPU runs co-scheduled with other CPU jobs, two runs per configuration, JIT/NVRTC/startup reported
separately. No grand performance claim is made. The items that were pending at D1 are listed at the end with their current disposition.

## 1. Executed test and gate results (root-verified)

| Stage | Executed result | Evidence |
|---|---|---|
| A1 (CPU native-walk, Numba), after C3 | 246 passed, 2 skipped (the A1 C3 check run) | `a1_c3_checks.log`, `a1_c3_report.md` |
| A2 (original-routine Fortran harness), after C1 | 256 passed, 452 skipped in the combined actual check run (450 skips belong to older candidate tests; an earlier run had 2 harness failures, fixed in C1) | `a2_c1_checks_actual.log`, `a2_checks.log`, `a2_c1_report.md` |
| Fortran kind probe | shared_data globals compiled with gfortran 13 are default REAL (kind 4) for `dt, dx, dx_m, density, sigma, dstar_const, radius, diameter, viscosity, settling_vel, ustar, d50` and kind 8 for `re, ke, p_par, v_soil` | `shared_kind_probe.json` |
| B1 C1 (GPU) | 80 passed + 1 single-device skip; a separate two-idle-GPU cross-device test passed | `b1_c1_report.md` and root records |
| B2 (GPU0) | full suite 191 passed, 1 single-visible-device skip, 21 failed (82.02 s); the 21 were two TEST-FIXTURE defects (reused CPU result buffers; a hook called without `reset()`), fixed in C2 with no model change; corrected compaction tests 31 passed (7.01 s) | `b2_independent_review.md`, `b2_c2_report.md` |
| B3 C1 (GPU) | real-GPU full suite 240 passed, 2 single-visible-device skips (131.50 s); CPU guards/caller/harness 89 passed, 20 GPU skips; Ruff PASS | `b3_c1_gpu_tests.log`, `b3_c1_cpu_tests.log`, `b3_c1_ruff.log` |

The controlled Fortran harness calls original MAHLERAN routines from 15 pinned source files with ordinary gfortran 13.3 on the source kind 4/8 globals; it is not the
MAHLERAN application. Standing qualifications are kept: the wet-law depth time level `previous` (default) vs the native post-infiltration level, the original
erasure option, default REAL kind-4 effects (smooth Fortran probes use the predeclared 2e-6 / 1e-14), and the original water's stale-inflow/bracket behaviour.
Full-storm Fortran differences are observations with 1% investigation triggers, not fitted tolerances.

## 2. Full Plot 1 (5400 s), CPU Numba vs GPU — executed, all comparisons pass the unchanged bounds with 0 flags

Evidence: `plot1_full_<solver>_b2_matched_comparison.json`, `plot1_full_<solver>_<separate|fused>_b3_matched_comparison.json`,
`plot1_full_<solver>_b3_modes_bitwise_comparison.json`, `plot1_full_b1c1_hardened_compare.json`. Integers exact; all saved fields, snapshots and counters of
the GPU repeats are bitwise equal; the `separate` and `fused` accounting outputs are bitwise equal to each other.

Loop seconds (`loop_wall_s_excluding_progress`, startup/JIT excluded), run 1 / run 2 and the mean of the two (mean = (a + b) / 2):

| Solver | CPU Numba (current load) | GPU B2 | GPU B3 `separate` | GPU B3 `fused` |
|---|---|---|---|---|
| bisection | 8.6245 / 8.4833 (8.554) | 14.0828 / 13.8546 (13.969) | 14.4417 / 13.9059 (14.174) | 13.9839 / 13.2013 (13.593) |
| Newton | 9.0667 / 8.7794 (8.923) | 10.6602 / 11.3166 (10.988) | 10.7025 / 10.6337 (10.668) | 10.5807 / 10.6169 (10.599) |

Reading it: at this size the GPU loop is **slower** than the CPU loop (fused/CPU mean = 13.593 / 8.554 = 1.59 for bisection, 10.599 / 8.923 = 1.19
for Newton; the first B1C1 run was 10.408 s vs CPU 7.234 s on an earlier load). Fused vs separate: 1 - 13.593/14.174 = 4.1% (bisection) and
1 - 10.599/10.668 = 0.65% (Newton) lower in the mean, but each mode's two runs differ by 0.54-0.78 s (bisection) and 0.04-0.07 s (Newton), as large as the
mode difference, so no fused gain is established from this data. The accounting change is 7 -> 1 device operations per step (6 fewer launches); it did not
remove any host-to-device copy because the series scalars were already device views of the packet.

## 3. Chastre (1393 x 1604, 1,140,169 active cells, 102,516 terminal-pit cells, ZERO outlets) — full CPU/GPU gates passed; original Fortran completed (see `current_status.md`)

Two full 2700 s B2 GPU runs per solver were published with source/input/33 GB tile hashes (case-verification hashes; post-run tile immutability only where `--hash-tiles-after-run` was used) and output pins verified, and each pair is an independent
GPU run vs a second GPU run (`chastre_full_bisection_b2_GPUrepeat_only.json`, `chastre_full_newton_b2_GPUrepeat_only.json`). All fields, snapshots, counters and summary counters are bitwise equal between the GPU repeats. Both full published Numba references pass every unchanged field and snapshot bound and exact integer check against their same-root B2 GPU repeats (`chastre_full_bisection_b2_matched_comparison.json`, `chastre_full_newton_b2_matched_comparison.json`). The original-Fortran full reference has since completed; its comparison (pickup +0.0053155% Numba relative to Fortran, final mobile -0.0001070%, gram-scale class-5 differences, a +24.952 m3 native surface-water surplus after rain ends that is still under investigation) is in `current_status.md` section 5.

| Quantity (B2, compact records) | Bisection | Newton |
|---|---|---|
| loop seconds, run 1 / run 2 (mean) | 278.701 / 278.543 (278.622) | 257.795 / 258.132 (257.963) |
| CPU Numba loop seconds (one full run) | 5,736.874 | 5,673.071 |
| CPU / GPU mean loop ratio | 20.59 | 21.99 |
| maximum peak host RSS across both runs (KiB) | 3,958,912 | 3,969,972 |
| first positive pickup / deposition | 193 s / 193 s | 193 s / 193 s |

Shared run facts (both solvers; read from the summaries): record values 170,068,224 bytes (all-class reference 510,204,672); estimated context total
2,781,451,419 bytes; allocated by the context 2,283,798,528 bytes; CuPy pool used 3,096,574,976 / total 3,492,774,912 bytes (estimated, allocated and pool
scopes differ); 169 sediment launches per step (456,300 over 2700 steps); 144 B hydrology packet per step; 45 flag/ledger slice reads; cold
costs (outside the loop) about 2.0-2.1 s hydrology prepare, 1.7-1.9 s context build including 1.4-1.6 s kernel compile and tables. Pickup 3,797,658.3485 kg,
active deposition 3,797,236.2398 kg, pit deposition 3,926.2493 kg, artificial clip source 1,706.6734 kg, final mobile 2,128.78212457 kg. These are legacy
benchmark totals with an explicit clip source and unlimited supply, **not** a conservation balance; export is identically zero (no outlets), so compare
maps, per-class and terminal-storage quantities. The B3 eight-run full-Chastre matrix is complete. Both solvers and accounting modes pass the published full same-root Numba reference at unchanged bounds, with exact integers; within-mode repeats and between-mode outputs are bitwise identical (`chastre_full_<solver>_<separate|fused>_b3_matched_comparison.json`, `chastre_<solver>_b3_modes_r1_GPUonly_comparison.json`). **No B3 Chastre gain is established.** Separate/fused means were 281.416 / 282.011 s for bisection and 259.571 / 260.145 s for Newton (fused about 0.21% / 0.22% higher), with two samples per configuration and recorded background load. Full timing evidence: `b3_chastre_full_timing_summary.json`. The record-compaction memory and the 9-12.5% isolated sediment-time reduction (below) are the only measured Chastre optimization numbers.

The published full CPU bisection run passed input/source and post-run tile guards; root independently pinned both CPU archives and summaries (`chastre_cpu_bisection_output_pins.json`, `chastre_cpu_newton_output_pins.json`). Newton loop breakdown was hydrology 985.455 s, wet laws 3,066.212 s and walk/CN/reduction 1,575.814 s, with whole process 6,000.229 s. Its CPU loop breakdown was hydrology 1,081.109 s, wet laws 3,045.228 s, and walk/CN/reduction 1,565.017 s; full process including preparation, warm-up and final verification was 6,070.255 s. The GPU ratio above compares measured loops against single-core Numba under the recorded concurrent load; it is neither a cold-process ratio nor a multicore CPU comparison. Maximum final mobile-map difference was 4.00e-15 kg, detachment-map difference 1.42e-13 kg, and depth-map difference 1.78e-15 m; all saved fields/snapshots passed, not just totals.

## 4. B2 isolated record microbenchmark (Chastre wet state, not a full storm)

`b2_microbench_independent_audit.json`: record values 510,204,672 -> 170,068,224 bytes (340,136,448 saved); every ledger/map/mobile/velocity/count/flag
bitwise equal in both orders. 60-step median sediment time: all 4.0423 s vs compact 3.6783 s (order 1; 1 - 3.6783/4.0423 = 9.0% lower) and all 4.2530 s vs
compact 3.7215 s (reverse order; 12.5% lower). The state is a computed 600 s water state from the old failed-publication CPU run, used as a fixture only.

## 5. Early and recovery diagnostics (NOT accepted benchmarks)

* `chastre600_early_CPUGPU_diagnostic.json`: an old CPU 600 s run computed and passed its guards but failed at publication (no complete receipt); its ledgers
  compared with the GPU 600 s ledgers pass the unchanged physical bounds and all integer controls. Diagnostic only; it does not substitute for the full CPU reference.
* `fortran600_vs_failed_cpu600_ledger_diagnostic.json`: only the complete 600 s ledger blocks parsed strictly; onset 193 s in both, pickup relative difference
  0.0316%, final mobile relative difference 8.08e-8; both publications are unqualified (Fortran maps truncated, CPU publication failed). An observation, not acceptance.
* After the 2026-10-06 environment restart (`recovery_record.md`) the old GPU 600 s NPZ (121,634,816 bytes) and the native Fortran 600 final maps were found
  TRUNCATED despite persisted completion markers: they are never qualified as accepted benchmark results, and markers/timings alone are not accepted. This was an
  environment-migration artifact, not a science, permission or usage-limit failure. It motivated the `output_pins` (SHA-256 and size of the closed archives) now
  written by the GPU driver and verified by the comparison harness; older unpinned outputs are accepted as `unbound`.

## 6. Items pending at D1 and their current disposition (details in `current_status.md`)

* Both full Chastre Numba references published and passed the full GPU comparisons. The original-Fortran 2700 s reference has COMPLETED
  (loop 13,839.809 s, kernel 13,651.230 s, 15-source-file harness, not the whole application); its matched comparison and 1% review were
  executed: no automatic flag, pickup +0.0053155% Numba relative to Fortran, final mobile -0.0001070%, class-5 pit/clip differences of about
  1.3 g (+42.57% / +6.43%), and a native +24.952 m3 surface-water surplus confined to the recession after rain ends (cause not demonstrated;
  open). No full-storm Fortran "acceptance" is claimed; it is an observation set.
* Full Chastre B3 CPU/GPU and bitwise accounting-mode gates passed. Fewer bookkeeping launches do not demonstrate a storm-time gain on this hardware; separate mode remains available.
* The E1 read-only review found no confirmed defect; the P2 candidate (bitwise-identical outputs, shorter loops in each same-session pair) and
  the corrected golden helper remain local candidates, NOT adopted; their adoption, regression and final review are separate steps.
* Not provided and not claimed: an evolving MAPLE bed, conservative event, restart, wind handoff, full-storm Fortran acceptance.
  Class-eligibility compaction is valid only for an event's immutable composition (`production_GPU_followups.md`, git-ignored task directory).
* Environments: Numba/llvmlite are absent from the current MAPLE `.venv` (CPU) and `.venv-cupy` (GPU, CuPy 14.2.0); the compiled-test results
  and CPU timings on this page are historical and were not rerun for the commit.
