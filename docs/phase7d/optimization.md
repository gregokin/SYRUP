# Phase 7d — shared MAPLE bed-transaction optimization

The accepted SYRUP work through Phase 7c was committed and pushed to
`gregokin/SYRUP` main as `a66bc69`. This follow-up optimizes actual MAPLE in an
isolated dependency snapshot; live MAPLE and the original accepted dependency
are untouched. No bed or wind physics has been copied into SYRUP.

## Changes in shared MAPLE

The patch changes only two files:

- `surface/voxels/transfer.py`: use reverse basic-slice views for top-down
  extraction instead of advanced-index copies. Remove the full-column
  provisional proportional divide: full drains are copied exactly, untouched
  voxels take zero, and the sole possible partial terminal take is assigned by
  the same existing corrected proportional-vector calculation afterward.
- `coupling/sediment_ledger/accumulate.py`: reuse the newly allocated
  `t - running_sum` buffer for the final Kahan subtraction. The four arithmetic
  operations retain their order. Input buffers, compensated channels, gross
  accounting, operation counts and zero-increment additions are unchanged.

No zero-demand transaction is skipped. Active-layer refill, burial, availability,
validation, atomic public calls and MAPLE conservation tolerances are unchanged.
The same array code supports NumPy and CuPy; this does not implement a full
GPU storm driver.

## Measured full-storm results

Both sequential CPU runs use frozen Plot1, six classes, 20 voxels, 32 phase bins,
dt=1 s and 5,400 s event duration. SYRUP package source is unchanged from the
accepted Phase 7c implementation. The only model difference is the recorded
MAPLE dependency patch.

| Component | Original MAPLE (s) | Optimized MAPLE (s) |
|---|---:|---:|
| Complete coupled loop | 256.797 | 233.564 |
| MAPLE pickup plus deposition/export | 160.704 | 136.988 |
| Characteristic transport, unchanged | 66.550 | 66.989 |
| Rainfall/infiltration/routing, unchanged | 7.824 | 7.728 |

The loop is **9.05% shorter** (1.099x speedup), and MAPLE transactions take
**14.76% less time**. All **103 saved numerical fields match exactly**; water
and class budgets close with unchanged tolerances. The candidate explicitly
records that MAPLE differs from the original case-import binding. That binding
has not been edited, and before/after source-stability checks pass for both runs.

These are single sequential same-machine runs, not a statistical confidence
interval. No other storm benchmark was run concurrently. The nearly unchanged
transport/hydrology timers provide context for machine variability. Do not add
this percentage to Phase 7c's percentage or compare timings from different
sessions as a controlled combined speedup.

Peak CPU-process RSS is effectively unchanged: 405,920 KiB baseline versus
405,688 KiB candidate (about 396 MiB each). Fewer kernel temporaries are not
evidence of a meaningful full-process peak-memory reduction.

A separate interleaved extraction microbenchmark on Plot1-shaped columns gives
1.66x speedup (4.943 to 2.977 ms median). Input column copies are excluded
symmetrically; public transactions retain their required copies. This is not
a claim that the whole storm is 1.66x faster. The zero-demand public timing in
the probe is candidate-only and is not a baseline/candidate speed comparison.

## Validation and scientific qualifications

The final patch passes 306 existing upstream voxel/active-layer/water/ledger
and conservation tests. A deterministic 120-case differential probe matches
all original extraction result fields and columns exactly. The dedicated
CPU/GPU regression extends this to interior boundaries and adjacent FP64
requests, zero/tiny/large demands, 1/2/6/9/17 classes and 1/2/20/64 voxels,
plus nontrivial Kahan compensation and input immutability.

The complete SYRUP suite against the candidate passed **578 tests**, with nine
device skips in that CPU environment. Separate actual-GPU execution passed all
**44** dedicated original-versus-candidate CPU/GPU checks and **126** upstream
CPU/GPU mass-exchange, adversarial-ledger and water tests, without skips.

On a GTX 1080 Ti, interleaved resident extraction timings improve from **3.723
to 3.309 ms** (1.125x speedup). All saved kernel fields exactly match the original
on the device. CuPy pool allocation growth falls from **7,124,480 to 3,631,616
bytes** (49% less). This measures pool growth after clearing unused blocks with
resident inputs held fixed, not global GPU memory or full-model peak memory.
Placement, input copying and compilation/warmup are excluded from the timing;
stream synchronization bounds each timed call. Raw samples and allocator
counts are in `benchmarks/phase7d/gpu_probe.json`.
Claude's supplied-excerpt reviews found no confirmed numerical defect. The
conditional acceptance requirements are now met. A confirmed setup issue in
Git's discovery ceiling was corrected to use the parent of the source directory;
a direct reproducer and a fresh reconstruction verified the correction. The
preparer now verifies the original snapshot, refuses a no-change patch, and
can require the exact expected final digest. A fresh reconstruction reproduced
the candidate digest; the activation helper independently verifies it before
selecting the dependency.

The reference probe loads original routines by file path but shares unchanged
helper modules with the candidate. The complete package comparison proves
only the two listed files changed. This probe must not be reused to claim
independence if a future patch changes those shared helpers.

Existing zero-request boundary reconciliation can remove a sub-roundoff top
voxel. This preexisting shared behavior is preserved and explicitly regressed;
it is not evidence that zero demands can be skipped. It deserves a separate
upstream scientific review, not a silent change in a performance patch.
Remaining Phase 7b timing/composition/peak qualifications are unchanged.

## Reproducible dependency and adoption

Original MAPLE package:
`d3d007024ff65abc2a0ff179f0f03bcd1c3b27f091493136cce849c34f7a4264`.
Optimized MAPLE package:
`72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65`.
The original snapshot records upstream revision `74dd3e79` plus two preserved
working files; its complete digest, not HEAD alone, identifies the baseline.

The patch and preparer are durable artifacts under `benchmarks/phase7d`.
To recreate the selected dependency at its canonical location, first ensure
the accepted original snapshot is present, then run:

```bash
python3 benchmarks/phase7d/prepare_candidate.py \
  --output outputs/dependencies/maple_72310c49 \
  --patch benchmarks/phase7d/maple_bed_optimization.patch \
  --expected-digest 72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65
```

Preparation refuses an existing target. Select your Python/Numba/CuPy environment,
then source `benchmarks/phase7d/candidate_env.sh`. It verifies package and project
hashes plus editable metadata, then selects the actual isolated MAPLE modules.
It installs nothing into the live MAPLE environment. For this existing matched
case, use the benchmark runner's explicit `--allow-maple-source-change` flag;
new case imports should bind the selected dependency directly.

The patch can be proposed upstream as a small shared optimization. Applying it
to live MAPLE, merging it upstream or switching an active simulation remains a
separate action. Preserve dependency provenance when adopting future updates.


## Acceptance and current state

The isolated optimization is accepted for the tested shared CPU/GPU operations
and matched CPU storm. Runtime selection is explicit through the verified
activation helper; the live upstream checkout remains untouched. The two-file
patch, setup helpers, regression tests and results form the Phase 7d follow-up
to `a66bc69`, approved by the user for commit and publication.

All review, simulation and verification processes have finished. No background
scheduler is running. Future optimizations must retain the same mass, ordering,
compensation and failure guarantees; the documented scientific and full-GPU
limitations remain open.
