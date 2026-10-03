# RFID water hydrology: measured timings

Completed by Claude implementation and bounded corrections, followed by independent Codex source review, verification, actual original-Fortran runs and CPU/GPU storms. Results below are warm water-only timings, not the full sediment or wind–water model.

RFID_2014: 104 × 58 interior cells, 5697 active and 335 hydraulically inactive; 0.1 m spacing; 2700 s prescribed storm/recession. Actual MAPLE case compiled and reloaded at outputs/rfid/case. One forcing and initial state shared by every method; 26 original D4 pits retained, no filling/carving. Maximum requested dt = 1 s, reports every60 s and forcing edges, no snapshots inside timers. Intel i9-7900X; GTX1080Ti physical GPU3. Unrelated MAPLE jobs continued on the shared host and other GPUs; hardware snapshots are archived. These are shared-machine measurements, not isolated-hardware scaling claims.

## Warm runtime

Each contender ran a complete untimed 2700 s warmup, followed by 3 complete measured storms in forward/reverse/forward order. Preparation/JIT and final field download, case/bed/source/budget validation and post-timer evidence capture are excluded. Evolution, normal public runtime checks, allocations, accumulation, reports and final GPU synchronization are included. Fortran links the unchanged original routines, with -O2; its own loop timer includes forcing, infiltration, routing, original update_water_flow (including dummy 6-class copying), finite-state checks and running totals/report rows. Fortran parsing/output are excluded. The workloads and accepted timesteps differ; a ratio is not a pure language/compiler comparison.

| Solver | Median (s) | Min–max (s) | Accepted steps | Rejected attempts |
|---|---:|---:|---:|---:|
| MAHLERAN Fortran: native Newton–CN (method 2) | 2.454 | 2.428–2.492 | 2700 | 0 |
| MAHLERAN Fortran: bisection–CN (method 5) | 10.866 | 10.747–11.100 | 2700 | 0 |
| SYRUP legacy, Numba CPU | 4.310 | 4.239–4.444 | 2772 | 72 |
| SYRUP legacy, CUDA automatic/split | 16.009 | 15.507–16.152 | 2772 | 72 |
| Explicit kinematic wave, NumPy CPU | 14.546 | 13.682–14.595 | 8926 | 4231 |
| Explicit kinematic wave, Numba CPU | 9.418 | 9.001–9.574 | 8926 | 4231 |
| Explicit kinematic wave, CUDA | 17.077 | 14.593–17.494 | 8926 | 4231 |
| Local inertia, NumPy CPU | 141.663 | 141.620–145.189 | 67336 | 32458 |
| Local inertia, Numba CPU | 82.167 | 81.299–84.312 | 67336 | 32458 |
| Local inertia, CUDA | 156.680 | 155.070–156.914 | 67336 | 32458 |
| SYRUP legacy, CUDA explicitly fused | 14.799 | 14.685–14.889 | 2772 | 72 |

Legacy Numba is **1.76× slower than native Fortran Newton**, but **2.52× faster than original Fortran bisection**, which is the routing solution method it reproduces. It is the fastest current SYRUP form on this case. Explicit Numba takes 2.19× legacy Numba time; local-inertial Numba takes 19.07×. Numba takes 35.3% less time than explicit NumPy and 42.0% less than local NumPy. GPU still loses to the corresponding compiled CPU forms: 3.71× for automatic legacy, 1.81× for explicit, 1.91× for local inertia. Explicit CUDA varies from 14.593 to 17.494 s: 1.55–1.86× the Numba median, so its three-sample median is less stable than the other estimates.

Automatic legacy CUDA selects the split path because maximum level width 1387 exceeds the current 128-cell fused threshold: 150 dependency levels, 153 hydrology launches per attempted step. Testing the EXISTING public fused mode reduces the median by 7.56%, from 16.009 to 14.799 s (separate subsequent launch, not interleaved, on the shared host; the sample ranges do not overlap). Its tested final fields and hydrograph are bitwise identical to automatic/split; it still takes 3.43× compiled CPU time. Kernel count is the hydrology step only: accepted-step accumulation and reporting are additional launches. The default and selector are unchanged. Reduction in launches does not imply proportional end-to-end speedup; a fused block also changes available parallelism. No attribution of the remaining GPU cost from this experiment alone.

Legacy RFID uses 64 bisection halvings rather than Plot1/default 40 to resolve deep zero-conveyance pit roots within the unchanged 1e-11 m root tolerance. This costs every wet root additional work; tolerances were never widened. Analytic zero-conveyance roots or a smaller sufficient iteration count are performance follow-ups, not changes adopted here.

## How timing changes from Plot1

Historical Plot1 measurements used 1200 cells, 0.5 m spacing and 5400 s, with the preceding accepted source. RFID has 4.7475× active cells (5.0267× total interior positions), half the simulated duration, different rain/soil/terrain and many more explicit/local attempts. Its shared surface-water arithmetic was also reassociated to remove documented complete/partial roundoff refusals. The table provides context, not pure grid scaling. Automatic CUDA selects fused for Plot1 and split for RFID. Original Plot1 whole-application Fortran times are NOT water-only comparators.

| Form | Historical Plot1,5400 s (wall s) | RFID,2700 s (wall s) | RFID/Plot1 cost per simulated hour |
|---|---:|---:|---:|
| SYRUP legacy, Numba CPU | 2.071 | 4.310 | 4.16× |
| SYRUP legacy, CUDA automatic | 8.046 | 16.009 | 3.98× |
| Explicit kinematic wave, NumPy CPU | 2.826 | 14.546 | 10.29× |
| Explicit kinematic wave, Numba CPU | 1.577 | 9.418 | 11.95× |
| Explicit kinematic wave, CUDA | 6.618 | 17.077 | 5.16× |
| Local inertia, NumPy CPU | 6.731 | 141.663 | 42.09× |
| Local inertia, Numba CPU | 3.415 | 82.167 | 48.13× |
| Local inertia, CUDA | 12.477 | 156.680 | 25.12× |

Plot1 legacy/explicit used 5400 accepted/no rejected steps; local used 7440 accepted / 2040 rejected. RFID legacy uses 2772 / 72, explicit 8926 / 4231 and local 67336 / 32458. GPUs gain relative to compiled CPU on the new candidates as the case grows, but do not overtake it here. Explicit's extra timesteps also reverse its former whole-event CPU advantage over legacy. Local's total attempt count is about 35.1× legacy's on RFID. The exact cause of the 72 legacy rejected attempts was not isolated; no rejection reason is inferred from the count alone.

## Water output and conservation

| Method | Export (m³) | Export relative to SYRUP legacy | Peak outlet (m³/s) | Peak time (s) |
|---|---:|---:|---:|---:|
| MAHLERAN Fortran: native Newton–CN (method 2) | 0.00157482746916 | 1.00214061× | 2.53454502238e-06 | 2641 |
| MAHLERAN Fortran: bisection–CN (method 5) | 0.00157482753631 | 1.00214066× | 2.53454501692e-06 | 2641 |
| SYRUP legacy, Numba CPU | 0.00157146357462 | 1× | 2.53454497926e-06 | 2641 |
| Explicit kinematic wave, Numba CPU | 0.00157199361373 | 1.00033729× | 2.50951532487e-06 | 2641 |
| Local inertia, Numba CPU | 1.08860081497 | 692.730543× | 0.000964690089659 | 2685 |

Legacy SYRUP exports 0.2136% less than original Fortran on these controlled arrays; peak differs by ~1.7e-8 relative and occurs at 2641 s in both. Final depth relative L2 difference is 0.09455%, discharge 2.739%, soil water 0.001235%; this is NOT field identity with Fortran. Explicit runoff is 0.0337% above legacy but its peak is 0.9875% lower. Differences include the original stale-inflow/conservation behavior and numerical/time-step conventions; their individual contributions were not isolated here.

Every SYRUP warmup/timed sample passes the unchanged MAPLE-derived rainfall, surface, soil and total-water bounds and preserves the actual MAPLE bed/source hashes. Final total-water residual is−2.96e-13 m³ legacy, +7.34e-13 explicit, +4.36e-12 local, for 5.78373273637 m³ applied rain. Original Fortran reports a positive 0.008188 m³ balance residual (~0.1416% of rain), retained/reported rather than copied or corrected. Its qualified timing requires completion, forcing/time/state/index and source/executable/input guards; it is not a conservative reference.

Local inertia exports about 693× legacy on this pit-rich fixed terrain. It uses pressure gradients and can spill out of depressions that store water indefinitely under fixed D4 legacy/explicit routing. Its normal-flow outlet also uses the documented bed-drop coefficient rather than the legacy edge-rule slope. This is a substantial physical-method difference; the full runoff difference has not been attributed solely to either mechanism. It is not a faster numerical replacement with equivalent output.

## Numerical and implementation checks

- Independent broad actual CPU/GPU/Fortran run: 2561 passed, 17 intentional unsupported-limiter/single-visible-device skips, 2 obsolete phase7j expected-refusal failures. After a test-only update preserving triggering inputs and bounds, independent phase7j follow-up 202 passed. Earlier affected tests and RFID checks also pass; no GPU-unavailable or compiler-unavailable skips. Ruff and whitespace checks pass.
- A preflight 600 s pilot completed every form. Main timing has 30 successful measured storms; forced-fused adds 3, each after a complete validated warmup. No failed/relabelled sample entered the medians.
- All tested final depth/soil/discharge or signed-face flux fields and complete reported hydrographs pass backend comparisons at rtol 2e-12 / atol 1e-14. Warmup versus final repetition fields are bitwise stable per form; forced-fused/automatic CUDA fields are bitwise equal; legacy Numba/CUDA fields agree within the stated bounds. This is checked-field qualification on RFID, not every intermediate public field on every terrain. Local-inertial velocity remains unqualified for sediment coupling; earlier Plot1 velocity/conditioning exceptions are not erased by this result.
- New opt-in nodata/pit graph policies preserve default strict Plot1 behavior and checkpoint identity. Deep pit storage is not drained or clipped; remaining storm water is retained at 2700 s. No dry reset, sediment/splash, evolving terrain, wind or restart claim.
- Complete-infiltration branch now sets old-flow depth 0 exactly, as MAHLERAN infilt.for does. Shared column depth uses the SAME retained depth plus rainfall excess/overflow, avoiding independently rounded equivalent expressions in partial intake. Intake, soil, drainage, overflow, physical equations and tolerances are unchanged. Genuine inconsistent old-depth input still fails. Changes apply to reference, compiled CPU and CUDA columns, including candidates.
- Controlled RFID arrays use captured native K but intended XML suction 23.6 mm/drainage0.05 instead of the native setup's overwritten 0.05 mm/zero drainage bug. The ring outlet is explicitly counted as export (the native positive rainfall scale failed to flag it). This compares ORIGINAL Fortran water routines on a matched controlled case, not the unchanged full native application. The importer preserves actual map fractions [0,0,0,0.092,0.908,0]. See README.md for source evidence and exact forcing pin.
- Actual MAPLE bed uses 335 invented finite placeholders excluded hydraulically. Its compiled masks nevertheless label all 6032 cells empirical_core, zero gap_filled. That discrepancy is disclosed; production wind handoff is unqualified. No upstream MAPLE/MAHLERAN file was edited.
- Memory evidence: main Python process maxRSS 575344 KiB (~562 MiB) including all libraries/contenders and imported bed; final CuPy pool 7,543,808 allocated / 5,551,104 used bytes. These are not per-solver peak footprints or a cold-start/memory-scaling comparison. Original Fortran childRSS and process wall are in the raw records.

## Reproduction and evidence

Pinned MAPLE source digest 72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65; original MAHLERAN source revision 305bd95d32123f13708be2f9a88e42ddd45d6f28; SYRUP baseline 031bce75d10371fec0701bf8bc59685e5579f84a plus this task's frozen uncommitted source manifest. Native applied-forcing capture SHA ce078efed2300bfd9ff52273ac15aa071a10eb1f47070c03c117112b002b30cf.

Source the pinned environments in agent_handoffs/tasks/phase7_matched_benchmark/gpu_env.sh and benchmarks/phase7d/candidate_env.sh. Standard harness: benchmarks/rfid/run_rfid_timing.py --case-dir outputs/rfid/case --output-dir <NEW> --end-s 2700 --rounds 3 --fortran-build-dir <NEW> --allow-maple-source-change. Actual launch used the recorded original timing build and Codex post-timer capture wrapper, with CUDA_VISIBLE_DEVICES=3. Forced mode used --legacy-cuda-mode fused --contenders legacy_cuda in that wrapper; context provenance records requested_mode=fused.

Raw timing: outputs/rfid/timing_2700/timing.json and outputs/rfid/timing_2700_fused/timing.json. Saved post-timer fields: the corresponding *_capture directories. Task evidence: agent_handoffs/tasks/rfid_timing/{benchmark_source_manifest.json,benchmark_source.tar.gz,benchmark_helpers_manifest.json,full_field_comparison.json,fused_vs_auto_fields.json,final_numeric_audit.log,independent_final_tests.log,independent_phase7j_followup.log,launches.json,benchmark_processes.json}. Prompts/reports/failures/native diagnostics are retained. Tests/source/helpers guarded and original reference unchanged after timing. No commit/push authorized or performed for this task.
