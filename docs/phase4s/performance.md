# Resident water-only CUDA storm hydrology

The existing rainfall/infiltration, coherent method-5 routing and storm scheduler now have an explicitly selected CUDA path. The coupled step, cumulative fields, peaks and hydrograph buffer remain on the GPU. This is water-only: sediment/wind GPU events, evolving terrain and disk restart are not qualified here. Existing defaults remain unchanged. See [usage](usage.md).

The narrow-network mode uses one block with unconditional dependency-level barriers. Wider networks use parallel column kernels, an ordered per-level routing sweep and a reduction. Each attempted step reads one 144-byte validation packet through MAPLE; accepted-step accumulation and report rows use resident kernels. Static arrays are validated and copied once. Public step results remain fresh, and failure does not commit state. Forced fused/split modes support comparison. All kernels use FP64 without fast math/FMA contraction; device libm and deterministic reduction order can differ from CPU within the existing 2e-12/1e-14 field bounds.

## Measured complete Plot1 storms

Intel Core i9-7900X and GTX 1080 Ti (physical GPU1, SM6.1), 1200 cells, 5400 seconds, 1-second steps, reporting every 60 seconds plus forcing edges. Frozen geometry and actual MAPLE bed. Best prepared, level-batched Numba hydrology is inserted through a benchmark-only direct closure into the SAME scheduler; the unprepared CPU CLI is not the speed comparator. Three warm full-storm runs per backend, balanced CPU/GPU/GPU/CPU/CPU/GPU order, no task tests or benchmarks concurrently. Other existing user workloads were preserved. Compilation/preparation and diagnostic downloads are outside timers; entry validation, accumulation/reporting and final GPU synchronization are included.

| Inputs | Prepared Numba median | CUDA median | GPU/CPU time |
|---|---:|---:|---:|
| Standard mean conductivity / parsed rainfall | 1.9698 s | 7.6788 s | 3.90 |
| Captured heterogeneous conductivity / applied rainfall / REAL32 initial soil | 1.9387 s | 7.4643 s | 3.85 |

The GPU remains slower for this small grid. These numbers concern water hydrology, not the full sediment model or a new Fortran timing comparison. The captured rainfall schedule compresses unchanged consecutive one-second rates; conversion into schedule intensity adds at most the recorded floating-point rounding, with all rate-change times retained.

Every repeated CPU/GPU storm comparison passes the existing field bounds, including cumulative grids, peaks, step/regime counts and hydrograph rows. Water closes under the unchanged MAPLE-derived volume bound; the bed digest remains unchanged. A separate final-source heterogeneous replay compares every public coupled-step field at each of 5400 steps: maximum normalized error 0.178280 (allowed 1), with conserved water. Source bindings and detailed budgets/timers are in the task evidence.

The standard water CLI was also executed end-to-end on the GPU against its original CPU CLI; all saved final fields and rows pass the existing bounds. Its loop records 5400 packet downloads, 777600 bytes, no uploads, one entry-validation scalar read and one explicit synchronization. Final reporting downloads and static preparation have separate counters. No claim of zero synchronization is made: packet validation synchronizes every attempt.

## Verification and limits

1357 actual-GPU tests passed with no skips, including both execution modes, column laws, refusals, metadata/device/stream guards, fresh results, shared storm scheduling, continuation and CLI checks. After cosmetic test lint fixes, 10 affected GPU checks pass again. CPU regression from the immutable package snapshot: 1006 pass, 11 device-dependent skips; one Git-provenance assertion expects an actual checkout and sees not_a_git_repository in /tmp. That assertion passes separately on the actual checkout (1 test, 6.25s); it is not a physics mismatch. Together these checks cover 1007 passing CPU tests and 11 device-dependent skips. Existing Fortran routine tests are included in the regression where available; whole-GPU/Fortran storm identity is not claimed.

Static context content is immutable by caller contract: pointer/shape/dtype/device/scalar metadata are guarded, in-place content mutation is not hashed every step. Provided contexts must bind to the exact supplied graph/parameter objects and metadata; rebuilding equal arrays requires preparation again. Source changes in MAPLE's live tree are excluded; the actual staged MAPLE package digest is 72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65 (a source digest, not a Git revision), from recorded upstream 74dd3e79 plus preserved dirty files and the accepted isolated bed patch.

GPU compile-cache coldness was not measured: these warm runs follow qualification compilation. Full peak device/host memory and broad whole-storm scaling remain follow-ups. Preliminary immutable TaskA wet-step probes showed GPU benefit at 65792 synthetic cells but not Plot1 or 16512; those probes do not establish a production crossover or complete-storm scaling. Small-grid costs include ordered double-precision roots on a narrow network, packet synchronization, allocation and host scheduling.

Exact commands, source manifests, raw logs, compiler/runtime/context details and scripts are archived under agent_handoffs/tasks/phase4r_gpu_storm. Final acceptance/review records there distinguish the author's unexecuted reports from Codex measurements. Experimental explicit and local-inertial routing remain a separate hydraulic qualification.
