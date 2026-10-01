# Phase 7c — characteristic transport optimization

This bounded optimization preserves the existing characteristic equations,
pickup position, phase count, timestep, mass authority and validation rules.
It changes only the optional CPU Numba implementation. MAPLE and the array
NumPy/CuPy path are unchanged. Full coupled GPU execution remains open.

## Implementation

The candidate list is compact, sized by the exact number of nonzero source
slots and pickup entries. Settling, export and underflow can only reduce that
count. Candidates remain in the original cell/class/slot order. Their packed
destination keys use int64; no new int32 grid-size limit is introduced.

Narrow, wide and zero-deposition-rate bins share one numerator scratch array
because the branches are mutually exclusive for a destination bin. Logarithmic
weights are calculated only for wide bins, after extrema are known. Positive-
rate bins whose candidate positions all coincide have an exactly zero narrow
numerator, so their redundant expm1(0) evaluations are omitted. Wide-bin
log-sum-exp remains intact, including the extreme-mass-ratio regression.

A completely empty mass/pickup input returns the nine zero output arrays
without allocating merge scratch. This uses exact zero, not a small-mass
threshold. Public validation still occurs before the kernel; nonzero water or
velocity alone does not imply sediment to transport. Every output remains a
new array; no mutable global workspace or caller-owned buffer is introduced.

## Controlled synthetic kernel measurements

`benchmarks/phase7c/characteristic_microbenchmark.py` compares the archived
original kernel and the candidate in one process on identical inputs. Both
are warmed before timing, with 15 interleaved measurements and reversed order
on alternating repetitions. All nine outputs match exactly in every case.
Compilation/first-call time is reported separately. These are CPU kernel
measurements, not storm speedups or GPU measurements.

| Cells | Classes | Bins | Synthetic occupancy | Warm speedup |
|---:|---:|---:|---:|---:|
| 1,200 | 6 | 32 | Empty | 6.41x |
| 1,200 | 6 | 32 | 3% | 3.63x |
| 1,200 | 6 | 32 | 100% | 1.21x |
| 1,200 | 6 | 128 | 3% | 2.08x |
| 4,800 | 6 | 32 | 3% | 2.53x |

For the 1,200-cell 32-bin sparse case, array payload allocated inside the kernel
falls from 22,752,000 to 11,691,936 bytes (49% less). Empty inputs fall to
4,089,600 bytes. Dense inputs require 19,065,600 bytes (16% less). These are
source-derived allocation totals including returned arrays, not measured peak
RSS; they exclude inputs, the wrapper, allocator overhead and JIT. Exact counts
and raw timings are retained in `benchmarks/phase7c/kernel_timings.json`.

## Full-storm verification

The final candidate reproduces all **103 saved numerical fields exactly**,
including forcing, hydrology, final state and the synchronous peak snapshot.
Water and class budgets close under the unchanged MAPLE tolerances. Plot1
export remains 0.01376397764041044 kg. The same frozen 60 x 20-cell, six-class,
20-voxel, 32-bin, dt=1 s, 5,400 s case is used in both runs.

| Component | Baseline (s) | Final candidate (s) |
|---|---:|---:|
| Coupled loop | 312.975 | 287.797 |
| Characteristic transport | 116.584 | 73.608 |
| MAPLE bed transactions, unchanged | 165.401 | 180.684 |
| Rainfall/infiltration/routing, unchanged | 8.075 | 8.818 |

The measured loop reduction is **8.04%** (1.087x speedup); characteristic
transport is **36.86% shorter** (1.584x speedup). A first candidate also
reproduced every field and measured 9.50% less loop time. These are sequential
single-run observations on this machine, not a statistically established
whole-event speedup: unchanged MAPLE and hydrology components varied by about
9%. No other SYRUP storm benchmark ran concurrently. Compilation is included
in the first step and separately reported. Full test suites and microbenchmarks
were run outside the final candidate storm timing.

Peak process RSS was 403,312 KiB baseline and 397,624 KiB candidate (394 versus
388 MiB). This small difference includes Python, JIT, imports and allocator
behavior; the 49% kernel array-allocation reduction is not a 49% full-model
memory reduction. Source identities, per-component timings and exact comparison
are retained in `benchmarks/phase7c/comparison.json`.

The synthetic benchmark exhibits allocation/order sensitivity at 4,800 cells:
candidate median is 12.08 ms when first in a pair and 17.87 ms when second;
baseline is 45.27 and 39.67 ms respectively. Both groups improve, but the
preselected all-sample median remains the reported statistic. Raw samples and
order-stratified medians are retained; best-case samples are not substituted.

The tools-disabled Claude review found no confirmed kernel defect and requested
a permanent bitwise reference regression. The original kernel is now retained
only under tests, with its original source hash, and compared on sparse/dense,
empty, converging, zero-rate, underflow and 1/7/32/128-bin cases. It is never
imported by production. Final full suite: **578 passed, 9 skipped** in 243.71 s. The skipped device
paths were unavailable in that CPU environment; a separate actual CUDA kernel
parity test passed (1 passed, 35 deselected). Lint and whitespace checks pass.
This is GPU kernel compatibility evidence, not a full GPU storm.

## Remaining optimization work

MAPLE transactions remain the dominant cost. They must be optimized in an
isolated, versioned MAPLE candidate and proposed upstream, not reimplemented
as SYRUP bed physics or silently skipped. Preserve active-layer refill and
burial, availability, gross ledger directions/compensation/touch counts,
validation and atomic failure. A zero demand is not generally an identity.

The next useful targets are selective or fused voxel processing and shared
transaction/ledger allocation, followed by coupled GPU execution. CPU gains
here do not qualify full-model GPU speed or memory. The Phase 7b limitations
on raw peaks and legacy sediment timing/composition also remain unchanged.

## Reproducing the checks

Use the existing isolated environment; it imports the recorded MAPLE snapshot.
The test-only frozen reference can reproduce kernel comparisons without `/tmp`
source files (its module hash differs from the original complete module, but
its `_substep` body is the archived original).

```bash
source agent_handoffs/tasks/phase6_complete_event/env.sh
"$SYRUP_PYTHON" -m pytest -q
"$SYRUP_PYTHON" benchmarks/phase7c/characteristic_microbenchmark.py \
  --baseline-module tests/phase7b/characteristic_reference.py \
  --output /tmp/syrup-characteristic-timing.json
```

Full events use the existing `benchmarks/phase7b/run_characteristic_benchmark.py`
with `--bins 32 --dt 1 --output NEW_DIRECTORY`. That script refuses an existing
output directory. `benchmarks/phase7c/compare_runs.py BASELINE CANDIDATE --output
REPORT.json` compares all saved numeric fields and extracts budgets/timings.
Reconstruct the old production package from the task's pre-edit archive for a
baseline event; do not run both storm timings concurrently or use a mutable
live MAPLE checkout.
