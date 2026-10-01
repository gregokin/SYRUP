# Why the current SYRUP benchmark is slower

A full frozen Plot1 Numba run with five component wall timers reproduces all
97 saved numerical arrays exactly. Total process wall time 197.24 s; coupled loop
192.93s. The earlier uninstrumented run was 195.58 s whole process. This is the
unchanged Phase7 source 01709bb52720923aa9b7211409d7ffd19155c4a9bfc87c36d9e1f0180a58d532,
measured before the sediment correction. Pinned MAPLE is unchanged.

| Component | Calls | Wall seconds | Fraction of coupled loop |
|---|---:|---:|---:|
| Actual MAPLE pickup transaction | 5,400 | 80.52 | 41.74% |
| Actual MAPLE deposition/export transaction | 5,400 | 80.96 | 41.96% |
| Rainfall column and compiled water routing | 5,400 | 7.97 | 4.13% |
| Wet sediment laws | 5,400 | 11.49 | 5.96% |
| Lateral sediment transport | 5,400 | 7.63 | 3.95% |
| Water-state publication | 10,800 | 1.09 | 0.57% |
| Remaining event work | — | 3.26 | 1.69% |

Pickup and deposition/export together consume 83.70% of the loop. The timer
wrappers return the original objects without altering computation. This is
low-overhead phase timing, not cProfile. The earlier detailed array cProfile
identifies voxel extraction/refill, burial, inventory reductions and ledger
accumulation inside those MAPLE calls; its nested timings cannot be summed
or substituted for the Numba phase timings above.

## Confirmed source-level causes

Each coupled SYRUP step uses two genuine MAPLE calls so pickup is supply-limited
before the newly mobile mass is transported, and deposition/export follows
transport. Each MAPLE call invokes BOTH erosion/refill and deposition/burial,
even when the caller's opposite-direction request is known to be zero. Each
of those routines copies the full voxel column to preserve atomicity.
For this 20-voxel, 1,200-cell, six-class case, four 1.152 MB column copies per step
amount to 24.9 GB of copied array payload over 5,400 steps, before temporary arrays,
reductions and read/write traffic are counted. This is a source-derived minimum,
not a measured memory-bandwidth or allocation profile.

This does not establish that raw copying alone dominates runtime. The shared
voxel kernels also repeatedly reduce whole columns, construct cumulative sums,
gather terminal voxels, and allocate intermediate arrays. They are already
largely vectorized; the remaining cost is not simply Python loops over cells.
The transaction total is measured directly, while the contribution of each
copy, reduction and validation needs a narrower optimization benchmark.

Each MAPLE call also invokes both gross ledger directions. That yields four
ledger accumulations per coupled step, with full process-shaped temporary
arrays, validation and compensated summation. The output inventory reductions
and repeated shape/value checks add work. Full expensive voxel-partition debug
validation is ALREADY off by default; simply turning off a debug switch does
not remove this cost. Safety and conservation checks must not be discarded.

A separate 100-call probe on the imported initial Plot1 bed measured a median
14.92 ms for a completely zero-demand transaction, comparable to the storm's
14.95 ms average MAPLE call. Even that request performs the shared work.
The resulting bed differs by up to 1.11e-16 kg per voxel entry and 5.56e-17 kg
per active-layer/availability entry through refill/roundoff handling; caller
inputs remain unchanged. These tiny differences are not a conservation defect,
but demonstrate why skipping calls needs a defined equivalence contract rather
than assuming every zero-demand transaction is literally an identity. This
probe is neither an optimization nor a measured achievable speedup; see
`benchmarks/phase7b/measure_zero_demand.py` and its result JSON.

The checked MAHLERAN executable takes about 9.3 s for the whole benchmark. It
updates its legacy arrays largely in place and does not maintain MAPLE's finite
voxel bed, active-layer refill, availability state, pure-return transactions
and gross conservative sediment ledger. Its sediment algorithm also differs.
The roughly 21-fold whole-program contrast is therefore not a controlled
Python-versus-Fortran or routing-only speed ratio.

Peak process RSS was 351,344 KiB (343.1 MiB) for the profiled SYRUP process
and 21,440 KiB (20.9 MiB) for the reference executable. These include the
language runtime, imports and compilation; they are not measurements of bed
storage alone, steady-state allocation, or GPU memory.

## Optimization priorities

1. Profile and optimize the shared MAPLE transaction path: avoid redundant
   zero-direction exchange work, reuse one validated transaction context and
   one owned working column, and reduce duplicate ledger allocations/scans.
   Preserve gross directions, actual-pickup-before-transport ordering, atomic
   failure behavior and MAPLE tolerances. Blindly skipping zero demands is
   unsafe until refill/availability/ledger side effects are accounted for.
2. Investigate compiled or more selective voxel extraction/burial on the actual
   shared MAPLE arrays, retaining exact boundary and roundoff behavior. Keep
   improvements upstream-compatible; do not copy MAPLE physics into SYRUP.
   Local bed exchange has independent cells, making it a useful target for
   fused CPU/GPU kernels that visit only the required voxels. Ledger totals
   still require reductions, and GPU validation synchronization must be measured.
3. Re-measure after the sediment correction. GPU acceleration of only the
   currently measured sediment kernels targets about 10% of the CPU loop;
   a full speedup requires the MAPLE transactions and routing to participate.
   Small kernels may actually be slower on the GPU, as Phase7 measured.

Even hypothetical zero-cost water routing/column work would improve this
loop by only about 4.3%. The current two sediment kernels becoming free would
improve it by about 11%. These are Amdahl bounds for this measured composition,
not projected achievable speedups. Output/setup/JIT account for only a small
part of the whole 197 s; they are not the main problem.

A read-only check of live MAPLE HEAD a3ba694 against its pinned base found no
new transaction optimization in these paths (the voxel-transfer difference
was comment formatting). No live upstream files were changed or adopted.
Raw timer script/logs/results: agent_handoffs/tasks/phase7b_performance.
This is a performance investigation; it does not relax the ongoing sediment
physics correction or authorize unmeasured shared-library substitutions.

## After the sediment correction

The corrected characteristic operator adds substantial work. In the full
32-bin dt=1 trial, the loop took 322 s: about 172 s (53%) in MAPLE exchange
and 119 s (37%) in the new transport operator. At 64/128 bins, transport
becomes the largest component. These trials overlapped other verification,
so their times are observational, not controlled same-work speed comparisons.
The original 83.7% figure applies only to the old operator.

The correction reduces Plot1 export from 218 g to about 13.8 g while preserving
conservation and water results; MAHLERAN exports 9.2 g and has its own
documented accounting differences. The extra cost buys bounded within-cell
position tracking and finite face-arrival timing. It has not yet been optimized.
Investigate the new phase arrays, compiled core and validation/allocation
overhead alongside the shared MAPLE transaction path. See
[bounded correction qualification](../phase7b/acceptance.md) for the convergence
and remaining peak-rate limitations. Full storms in this work were CPU runs;
GPU verification covers kernels only.

## First accepted optimization follow-up

Phase 7c optimizes the characteristic CPU kernel without changing its numerical
results. In sequential Plot1 measurements, the full loop decreased from 313 s
to 288 s and the characteristic stage from 117 s to 74 s. All 103 saved numeric
fields match exactly. The unchanged MAPLE transactions still take most of the
time (181 s in the optimized run). Do not treat the old 84% profile as current.
See [Phase 7c evidence](../phase7c/optimization.md) for source identities,
allocation reductions, variability and final validation status.


## Shared MAPLE allocation follow-up

Phase 7d uses an isolated actual-MAPLE snapshot with a two-file upstream-ready
patch. Reverse extraction views, removal of overwritten provisional work and
Kahan temporary reuse reduce matched-storm bed exchanges from 160.7 to 137.0 s
and the full loop from 256.8 to 233.6 s. All 103 saved numeric fields remain
exactly equal. Actual GPU extraction also improves (3.72 to 3.31 ms) with less
allocation-pool growth. See [Phase 7d evidence](../phase7d/optimization.md).
These are separate same-session comparisons; do not add percentages across
phases or infer a full-GPU storm result.
