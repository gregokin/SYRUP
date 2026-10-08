# GPU Newton validation and timings

Measured 2026-10-03 on an Intel i9-7900X and NVIDIA GTX 1080 Ti (physical GPU 3, visible device 0;
SM 6.1), using Python 3.12.3, NumPy 2.5.2, Numba 0.67 and CuPy 14.2 under WSL.
The safeguarded Newton root is now available for the ordered CUDA sweep and complete resident
water hydrology. It solves the same corrected MAHLERAN-derived cell equation as CPU Newton;
rainfall, infiltration, drainage, donor ordering, flux bookkeeping and physical rejection rules
are unchanged. Bisection and the existing automatic launch-selection rule remain defaults.

## Warm complete-storm timings

Seconds below are medians of three measured complete storms after preparation, a first call and
a complete untimed warm-up. Rounds use forward/reverse/forward contender order. Case creation,
JIT/preparation, post-run validation and field capture are outside the evolution timer. The
timer includes the shared storm driver, accumulation, reporting and final synchronization.
This is water-only timing on fixed elevation and routing, without splash or sediment exchange.

| Case | Numba bisection | Numba Newton | GPU bisection, auto | GPU Newton, auto | GPU Newton time saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plot1, 5400 s | 2.130 | 1.871 | 8.464 | 5.950 | 29.7% |
| RFID, 2700 s | 4.521 | 4.009 | 15.785 | 12.668 | 19.7% |

Automatic selection resolves to fused on Plot1 and split on RFID. Both explicit modes were
also validated and measured:

| Case and actual GPU mode | GPU bisection | GPU Newton | Newton time saved |
| --- | ---: | ---: | ---: |
| Plot1 fused (auto) | 8.464 | 5.950 | 29.7% |
| Plot1 split (forced) | 13.139 | 13.093 | 0.4% |
| RFID split (auto) | 15.785 | 12.668 | 19.7% |
| RFID fused (forced) | 15.247 | 6.447 | 57.7% |

Across separate invocations, RFID fused Newton takes 49.1% less time than split Newton and
is 1.61 times the current Numba Newton time. Plot1 fused Newton is 3.18 times Numba Newton. GPU acceleration therefore remains
slower than CPU on these domains, despite the improvement over GPU bisection. Plot1 split's
0.4% median difference is smaller than the observed timing spread; it is not evidence of a
meaningful speed-up. Separate forced-mode invocations are subject to host variability.

| Case/mode | Bisection sample range, s | Newton sample range, s |
| --- | ---: | ---: |
| Plot1 Numba | 2.119–2.597 | 1.737–2.039 |
| Plot1 GPU fused | 8.388–9.096 | 5.515–6.178 |
| Plot1 GPU split | 12.539–13.401 | 12.888–13.320 |
| RFID Numba | 4.502–4.674 | 3.834–4.097 |
| RFID GPU split | 15.461–16.537 | 12.195–13.023 |
| RFID GPU fused | 15.190–15.327 | 6.346–6.533 |

All **42 measured storms** succeeded; the audit checks every sample's status, counters, sources
and applicable water/bed guards. Full warm-ups and first calls are additional runs. No task
writer or other task benchmark ran concurrently. Existing MAPLE jobs on other GPUs were
preserved, so this is a shared-host measurement, not an isolated-machine throughput guarantee.
Raw JSON retains every sample and separate preparation/first-call/warm-up measurements. CUDA
caches were already warm from validation; a truly cold NVRTC compiler cost was not isolated.

Plot1 uses 60×20 cells, dx=0.5 m, actual model-2 pavement/Hawkins infiltration, 40 bisections
for the comparator, and 5400 accepted steps. RFID uses 104×58 array cells (5697 active),
dx=0.1 m, model-1 fixed conductivity, 64 bisections, 2772 accepted steps and 72 rejected attempts.
Both use max dt=1 s, reports every 60 s, and Newton cap 50 with bounded bracket completion.
Forcing-boundary handling and accepted/rejected schedules are shared across SYRUP contenders.

## Independent numerical validation

Codex independently ran the actual-GPU and original-Fortran regression suite: **3168 passed,
17 intentional skips**, zero failures/errors, plus **3 passed** after a metadata-only correction.
Skips were five multiple-device checks and twelve unsupported explicit/donor-limiter combinations;
none was caused by unavailable GPU, Numba or Fortran. Four warnings comprise two known invalid
multiplications in refusal tests and two pytest XML-format warnings. Ruff and whitespace checks
pass. No root, sweep, column or scheduler arithmetic changed after the broad suite.

Scalar roots and ordered sweeps match the CPU specification bitwise on identical inputs,
including extreme/subnormal values, NaN/Inf probe handling, zero conveyance, low caps and forced
bracket completion. Independent high-precision roots provide another check. FP64 round-to-nearest
intrinsics and existing no-fast-math compile options preserve arithmetic order. Untimed full-storm
CPU diagnostics show at most three Newton passes on Plot1 and four on RFID, with zero safeguard
or completion fallbacks. Those branches are validated by scalar/sweep/low-cap tests, rather than
by the measured storm paths; GPU production counters are intentionally absent.

Full-case validation compares all numerical `EvolveResult` fields, named hydrograph columns,
cumulative/peak grids, regime counts, rejection logs and boundaries, excluding four timing fields.
Same-solver CPU/GPU comparisons and opposite GPU launch modes meet the unchanged **rtol=2e-12,
atol=1e-14** bounds: 40 fields on Plot1 and 256 on RFID. Separate report-aligned segmented
continuations compare wet maps at Plot1 600/1200/1620 s and RFID 600/1200/2640 s (166/382 fields).
These are separate full runs, not snapshots from measured trajectories. All full runs close their
water budgets and preserve the bed and sources. Default bisection final fields and all report rows
are bitwise identical to pre-edit CPU and GPU captures on both cases.

One different-solver qualification is retained explicitly. Plot1 Newton versus 40-bisection
has a maximum difference of **1.3793111618143339e-14 m/s** in the reported maximum velocity
column, with relative L2 about 2.974e-13. Some late-recession points exceed the backend allclose
bound; the raw audit keeps `within_backend_bounds: false`. The same difference occurs in CPU
Newton versus CPU bisection and in prior archived CPU results. All other cross-solver fields
pass these bounds on Plot1; all pass on RFID. This does not affect same-solver CPU/GPU agreement.
An initial helper incorrectly enforced backend bounds across different solvers; that failure
and script were archived, the comparison was qualified, and the complete audit was rerun.
No production tolerance was changed and no discrepant field was dropped.

Full-event water residuals are 4.04e-14 m³ (Plot1 bisection), 4.40e-14 m³ (Plot1 Newton) and
−2.96e-13 m³ (RFID, both). MAPLE-derived budget bounds and physical root tolerance 1e-11 m
remain unchanged. Actual CLI selection, invalid controls, failure precedence, device/context
contracts and supported in-memory storm continuation are covered. GPU sediment, evolving
terrain, disk storm restart and alternating wind/water events are outside this qualification.

## Native Fortran reference

Original RFID Fortran water routines (`iroute=2` Newton and `iroute=5` bisection, gfortran 13.3,
−O2) take **2.638 s** and **11.212 s** respectively. Fused GPU Newton takes 2.44 times native
Newton's loop time; Numba Newton takes 1.52 times. These are qualified comparisons, not identical
workloads: Fortran executes 2700 fixed steps, while conservative SYRUP executes 2772 accepted
steps plus 72 rejected attempts. The original routines retain a water gain of about 0.008188 m³
(0.1416% of rain), roughly 0.2141% higher export, and their original inflow bookkeeping.
We did not reproduce that gain to obtain agreement.

Native RFID has 45 reports versus SYRUP's 46 (extra 2641 s forcing boundary); no complete native
versus SYRUP hydrograph norm is claimed. Final fields, export and peaks are compared separately.
No native Plot1 result was produced: the existing native adapter supports model 1/zero pavement,
whereas actual Plot1 uses model 2/pavement. Extending the adapter is a separate task; this is
an adapter limitation, not a limitation of MAHLERAN itself.

## Transfers, memory and next investigation

Newton adds no counted per-step transfer or production statistics. Both root solvers download
one **144-byte diagnostic packet per attempt**: Plot1 5400 packets/777600 bytes, RFID 2844
packets/409536 bytes. Timed evolution has zero counted H2D transfers, two scalar reads and
one final synchronization. These are the existing counter boundaries, not a claim of zero
implicit runtime synchronization. Static preparation and post-timer captures are separate.
Production `root_stats` is `None`; iteration probes are untimed.

No extra per-step diagnostic arrays or persistent water state are introduced; existing fresh
step output allocations remain. Newton uses a separate code module. Reported host RSS is a
process-wide high-water mark, not a per-solver peak; solver-specific GPU peak memory and large
domain throughput were not measured here.

Next, investigate solver-aware fused/split selection and larger-domain scaling without changing
the automatic rule on this evidence alone. RFID split has 153 launches per attempt; its small
levels and packet synchronization are plausible remaining overheads. Profile launch/driver,
packet, allocation and ordered-sweep costs before choosing another optimization. Plot1 split
shows why reducing root arithmetic alone cannot guarantee an event-level improvement.

## Sources, artifacts and reproduction

Baseline SYRUP commit: `d5861b412bf3b6df7c4dd709545b931e05568105`; timed source is uncommitted,
bound by the 381-file verification manifest/archive. SYRUP package digest:
`684e00367be6fbbf98bc657f47fad645f0b5375d1e9a06d2963b009a4c485d11`.
Actual immutable MAPLE source snapshot digest:
`72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65`.
MAHLERAN HEAD: `305bd95d32123f13708be2f9a88e42ddd45d6f28`; JSON additionally pins actual
routine hashes and Fortran executable SHA-256
`2dfefd409bbb469038c8ff37830f51aab9cb29510fe8319c38a823a20d46e4b1`.
Documentation finalized after timings is recorded separately from timed sources.

Local raw comparisons:
[Plot1 auto](../../outputs/gpu_newton/plot1_auto/comparison.json),
[Plot1 split](../../outputs/gpu_newton/plot1_split/comparison.json),
[RFID auto and native](../../outputs/gpu_newton/rfid_auto/comparison.json),
[RFID fused](../../outputs/gpu_newton/rfid_fused/comparison.json).
Full-field captures, pre-edit defaults, numeric audit, XML, logs, commands, manifests and
archives are in `agent_handoffs/tasks/gpu_newton/`. Generated outputs and handoff evidence are
local ignored artifacts; they are not included in a source-only checkout.

```bash
source agent_handoffs/tasks/phase7_matched_benchmark/gpu_env.sh
source benchmarks/phase7d/candidate_env.sh
export PATH=/home/okin/MAPLE/.venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=3
python benchmarks/gpu_newton/compare_cases.py --case plot1 --case-dir outputs/plot1 \
  --output-dir <NEW_PLOT1> --rounds 3 --allow-maple-source-change
python benchmarks/gpu_newton/compare_cases.py --case rfid --case-dir outputs/rfid/case \
  --output-dir <NEW_RFID> --rounds 3 --allow-maple-source-change \
  --contenders bisection_numba,newton_numba,bisection_cuda,newton_cuda,fortran_newton,fortran_bisection \
  --fortran-exe outputs/rfid/fortran_build_final/rfid_water_driver
# For forced-mode tests, use a new output directory and add:
# --contenders bisection_cuda,newton_cuda --cuda-mode fused|split
```

`--allow-maple-source-change` explicitly adopts the recorded immutable MAPLE snapshot instead
of the case's historical binding; case/content and within-run source checks remain enabled.
The actual runs use the task's progress wrapper around the same shared benchmark harness,
with messages outside original timers and exact process/command records in the task folder.
