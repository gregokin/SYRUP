# Phase 4R step 1: CUDA routing qualification and performance

Status: accepted after Codex verification and final Claude review (no material blocker). Prior CPU optimization committed/pushed
as 031bce7. This routing extension is separate and uncommitted.

The CUDA sweep reproduces existing bisection and donor arithmetic. It substantially accelerates the old CuPy array
sweep, but the optimized batched CPU sweep is faster at all three tested sizes on this GTX 1080 Ti. Keep the CPU
default. This qualifies routing, not GPU infiltration, sediment or a complete storm.

## Synchronized wall time

Median milliseconds per warm invocation, alternating candidate order. Four fresh outputs on both backends;
resident inputs. CUDA wall includes cache checks, launches and final stream synchronization. CPU timing includes
no input copies or GPU synchronization.

| Network | Active cells | Requested positive-base fraction | Batched CPU ms | CUDA ms | Old CuPy ms | CUDA speed vs old CuPy | CPU speed vs CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| plot1 | 1,200 | 1.00 | 0.170 | 2.081 | 402.2 | 193.3× | 12.2× |
| plot1 | 1,200 | 0.10 | 0.141 | 1.595 | 431.4 | 270.5× | 11.3× |
| valley_128x129 | 16,512 | 1.00 | 0.787 | 3.889 | 1267.1 | 325.8× | 4.9× |
| valley_128x129 | 16,512 | 0.10 | 0.829 | 5.000 | 1143.5 | 228.7× | 6.0× |
| random_256x257 | 65,792 | 1.00 | 2.881 | 5.416 | 1698.1 | 313.6× | 1.9× |
| random_256x257 | 65,792 | 0.10 | 5.226 | 8.197 | 1650.3 | 201.3× | 1.6× |

Thirty CUDA/CPU samples and ten expensive old-CuPy samples per case, after ten and three warm-ups respectively.
Full samples/IQRs: [measurements.json](measurements.json). Positive-base fraction is a random input selection;
donor-fed cells may become wet. These are synthetic fills on verified Plot1 topology and controlled larger terrains,
not full heterogeneous storms or their measured dry-cell frequencies.

The shared route_step wrapper adds checks, array operations, scattering and budgets:

| Network, all-positive case | CPU wrapper: original serial Numba sweep ms | CUDA wrapper ms | Old CuPy wrapper ms |
|---|---:|---:|---:|
| plot1 | 1.019 | 5.577 | 451.9 |
| valley_128x129 | 8.148 | 9.771 | 1098.5 |
| random_256x257 | 31.152 | 13.602 | 1724.0 |

**CPU wrapper uses the original serial sweep, not optimized prepared/batched production hydrology.** Larger-grid
wrapper gains cannot establish gains over the production default. No whole-event Fortran timing comparison was
run in this task.

## Interpretation

Graph depth/maximum width: 66/86 (Plot1), 192/256 (16,512 cells), 256/257 (65,792 cells). With block size 128, levels use at most one, two and three
blocks respectively. These are relatively deep, narrow networks; wide-frontier GPU throughput is unqualified.
One kernel launch per
dependency level replaces hundreds of small CuPy operations per level. FP64, donor summation order, [0, RHS]
bracket, strict comparison and 40 or caller-selected 1–200 bisections remain unchanged. No physics, conservation
tolerance or fast-math relaxation.

CUDA events time a separate direct launch-loop pass: cache checks/output allocation excluded; gaps between launches
included. They are not isolated device-compute times. Plot1 all-positive event median was 1.517 ms; host enqueue
median 0.807 ms. Enqueue alone exceeds the 0.170 ms CPU invocation, supporting investigation of fewer launches.
Do not subtract separately collected wall/event medians to infer a precise overhead decomposition. Larger-network
event passes sometimes take longer than the earlier wall passes. Instrumented level-boundary events include gaps
and perturb timing.

Next bounded candidates: CUDA graph replay or one-block routing for narrow levels, preserving equations and testing
parity. Coupled GPU infiltration and wrapper optimization need separate qualification. Newton/local-inertial methods
remain separate Phase 4R investigations; none is adopted here.

## Startup, transfers and memory

First kernel compile/load in a separate process and dedicated CuPy cache directory: 0.248 s, excluded from the table.
Previous cache contents were not logged; cold-versus-cached status is not established. Future startup can differ.

| Active cells | Context preparation ms after kernel load | Owned static bytes | One-time D2H bytes | One-time H2D bytes | Fresh output bytes |
|---|---:|---:|---:|---:|---:|
| 1,200 | 2.943 | 52,800 | 74,400 | 52,800 | 38,400 |
| 16,512 | 6.971 | 726,528 | 1,023,744 | 726,528 | 528,384 |
| 65,792 | 56.897 | 2,894,848 | 4,079,104 | 2,894,848 | 2,105,344 |

Preparation downloads/validates data once, uploads owned copies and explicitly synchronizes once. MAPLE counters
count seven downloads/three uploads; this direct CuPy synchronization is not registered in MAPLE's synchronization
counter. Cache-hit sweeps showed no counted transfers or scalar reads. The complete wrapper retains its flag read.

Recorded pool reserve/free deltas after a call include live benchmark inputs/earlier results. These are **not true
peak-used memory**, full-storm memory or host RSS. The raw sweep allocates four outputs, no per-level scratch;
static copies cost 44 bytes per active cell. Contexts expire with graph lifetime or explicit release. Full-event
memory residency and speed remain unqualified.

## Scientific qualification and ownership

Initial real-GPU suite: 366 passed, one device-guard skip, 23.41 s. CPU regression: 438 passed, four device skips,
44.36 s, including executed original-Fortran routing routines using the existing isolated compiler. CPU-only optional
contracts without CuPy: 26 passed, 1.15 s. Final expanded real-GPU suite: **484 passed, no skips**, 23.42 s, with physical GPUs 1 and 3 visible so both
current-device and dynamic-input mismatches were actually checked. Only physical GPU 1 ran routing workloads.
Final CPU regression/contracts: **466 passed, four device skips**, 44.09 s.
After final provenance-diagnostic and wording fixes: 21 targeted optional-backend/device/compiler/route checks passed,
463 deselected, 3.80 s. Compiler-failure recovery, exact widths
1/127/128/129/255/256/257, donor-slot corruption and scope guards passed. Lint and whitespace checks passed.
CUDA compute-sanitizer was unavailable; no memcheck result is claimed.

Four raw outputs matched original serial and batched CPU sweeps bitwise across tested normal/mixed/zero/negative/
subnormal/infinite/NaN states. NaN payloads are outside the contract; positions/signs are checked. Whole-step
physical fields and a 240-step source/recession sequence passed predeclared rtol=2e-12/atol=1e-14; conservation and
rejection tolerances unchanged. This is not a new whole MAHLERAN-versus-GPU storm. CPU/GPU parity and existing executed
Fortran regressions provide distinct evidence.

**Do not mutate graphs after CUDA preparation.** The cache owns copies. Pointer/shape/dtype replacement is detected,
content mutation is not. The wrapper still reads the original graph; mutation can make it inconsistent with the
sweep. CUDA is qualified through the routing API only. Storm controls/CLIs retain existing choices; legacy replay
remains CPU.


## Post-correction confirmation

The six-case table above times the archived pre-correction snapshot. Subsequent changes strengthen preparation
validation, refuse unqualified APIs, bind benchmark closures and improve diagnostics/labels. Kernel source and
launch sequence are unchanged; updated cache metadata checks add host work. Post-correction confirmation, before the final docstring edits, used the
same bases, 30 warm alternating CPU/CUDA samples and fresh output arrays. All four outputs remained bitwise equal
to CPU and the original CuPy sweep; cache-hit transfers/scalar reads stayed zero. It did not re-time old CuPy.

| Network | Requested positive-base fraction | Confirmation batched CPU ms | Confirmation CUDA ms |
|---|---:|---:|---:|
| plot1 | 1.00 | 0.151 | 1.755 |
| plot1 | 0.10 | 0.132 | 1.603 |
| valley_128x129 | 1.00 | 0.760 | 3.900 |
| valley_128x129 | 0.10 | 0.798 | 4.167 |
| random_256x257 | 1.00 | 2.841 | 5.259 |
| random_256x257 | 0.10 | 5.204 | 8.042 |

These observations confirm the recommendation to keep the optimized CPU default. Full source hashes, sample
spreads and verified case/dependency metadata are in the post-correction block of measurements.json. Final documentation
edits change file hashes; the executable run_sweep and helper ASTs match the initially measured implementation
excluding docstrings, and the CUDA source/options remain identical. Raw timings vary between passes (about 16% for
Plot1 CUDA); the consistent conclusion is CPU faster on these networks. The sparse-case slowdown remains unexplained.

## Reproduction and provenance

Baseline 031bce75d10371fec0701bf8bc59685e5579f84a plus separately recorded uncommitted routing patch.
Actual MAPLE package source digest (not a Git revision):
72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65.
This isolated candidate reconstructs upstream Git 74dd3e79d9a7df85244355439e3a4a762abb9273, preserves two
preexisting dirty files (orchestrator/voxel transfer), and applies the accepted Phase 7d bed-allocation patch.
Exact dirty-file and patch hashes are in measurements.json; the source digest is verified before environment selection.
Physical GPU 1 (visible 0), GTX 1080 Ti SM6.1; CuPy14.2.0, NVRTC12.9, runtime12.9/driver API13.0.
Unrelated jobs on other GPUs were preserved. Results do not generalize to other hardware.

Exact measured hashes and original harness snapshot: agent_handoffs/tasks/phase4r_gpu_routing.
Final qualified hashes/post-measurement corrections are recorded in measurements.json. Main-run case binding metadata
was not captured by its earlier harness; verified case/dependency records were captured in the confirmation block. Reference trees/environments unchanged.

```
source agent_handoffs/tasks/phase7_matched_benchmark/gpu_env.sh
source benchmarks/phase7d/candidate_env.sh
export CUDA_VISIBLE_DEVICES=1
export CUPY_CACHE_DIR=/tmp/syrup-cupy-phase4r-qualification
"$SYRUP_PYTHON" -m pytest tests/phase4r -q
"$SYRUP_PYTHON" benchmarks/phase4r/bench_cuda_routing.py \
  --plot1-case outputs/plot1 --allow-maple-source-change \
  --networks plot1,valley_128x129,random_256x257 \
  --repeats 30 --array-repeats 10 --output <new-output>.json
```

Explicit source-change option re-verifies the case against current pinned MAPLE; it does not rewrite the binding.

Technical sources: [CuPy RawKernel](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html),
[CuPy timing guidance](https://docs.cupy.dev/en/stable/user_guide/performance.html),
[NVRTC compiler options](https://docs.nvidia.com/cuda/archive/12.8.0/nvrtc/index.html).
Source uses RN FP64 intrinsics and --fmad=false. Precision flags for division/sqrt document single-precision behavior,
not additional FP64 parity guarantees.
