# Phase 7e Stage 1 — selective shared MAPLE bed exchange with trusted surface metadata

Status: implemented in an ISOLATED MAPLE candidate built from the accepted
Phase 7d snapshot `72310c49` plus `benchmarks/phase7e/maple_selective_exchange.patch`
(package digest recorded in `benchmarks/phase7e/candidate_digest.txt`). Qualified for explicit opt-in selection;
the prior dependency, live MAPLE and SYRUP production adapter are unchanged.
Numbers in the results section come from the sequential storm set in
`outputs/phase7e/stage1` and the probes under `agent_handoffs/tasks/phase7e_sparse_legacy_bins`.

## What the accepted transaction does per call

Each SYRUP coupled step makes two `apply_water_process_demand` calls (pickup, then
deposition/export). Each call runs BOTH active-layer routines (erosion + refill, deposition +
burial). In the accepted implementation every routine copies the whole
`(ny, nx, n_voxels, n_classes)` voxel array and runs a whole-domain batched kernel over it:
per-voxel totals, cumulative sums, terminal search, per-voxel take arrays and the in-place
update all touch every voxel of every column, with several full-size temporaries. Two
whole-bed inventory reductions bracket the call, and two ledger accumulations each run
Kahan passes over all six process slices of four `(6, ny, nx, n_classes)` channels.

Measured on the Plot1 bed with storm-like dense demands (100 pickup/deposit pairs,
`profile_transactions.py`): 12.6 ms per call; voxel kernels 43%, ledger 32%, result
validation 6%, inventories 3%.

## Design

### Trusted surface metadata (`core/types/voxel.py`)

`VoxelColumnState.surface_cache: VoxelSurfaceCache | None` holds the per-cell voxel totals
`(ny, nx, n_voxels)` of ONE exact mass array, bound by object identity (`ArrayBinding`). The
field is `init=False`: every ordinary construction, `dataclasses.replace`, `to_host_tree` /
`to_device_tree`, snapshot or checkpoint restore and topographic commit yields a column WITHOUT
a cache, so stale metadata is never carried into another state and no old bed is retained
through a conversion. Only `surface/voxels/transfer.py::column_with_surface_cache` attaches
one, to a mass array the exchange routine has just produced, and it FREEZES that array
(read-only on NumPy; `freeze` is an honest no-op on CuPy).

Trust rule (`VoxelColumnState.valid_surface_cache(trusted=False)`): the cache is used only
when it describes this exact array AND either the array is enforceably read-only (NumPy) or the
transaction declares `trusted=True` because it owns the whole chain of states since the
exchange that built the cache. The water step always trusts its internal erosion -> deposition
hand-off (that intermediate is never exposed) and exposes `trust_surface_cache=False` by
default for its own input. An ordinary external caller with a writable array therefore never
has a cache trusted on its behalf: the totals are rebuilt from the mass array (one full read,
counted). This closes the defect Codex reproduced on the first candidate (identity binding on a
writable array could be mutated in place and the stale totals were trusted): on NumPy the
in-place write now raises; a deliberately un-frozen and mutated array is detected as writable
and rebuilt; on CuPy the cache is not used without declared trust. `validate_voxel_surface_cache`
recomputes and compares bitwise for trusted boundaries, `validate_full_state=True`, and tests.

### Selective kernels (`surface/voxels/transfer.py`)

From the totals each call locates every cell's surface and
1. SKIPS cells on which the whole-column kernel provably has no effect (zero extraction request
   with an empty top window voxel; zero deposition request) and reports zeros for them. A zero
   request at a non-empty top ALLOCATED voxel is not skipped, so the established sub-roundoff
   reconciliation is preserved;
2. gathers a top-aligned 3-voxel window (top occupied voxel, one below, one above) for the
   remaining cells that pass a per-cell exactness test and runs the UNCHANGED batched kernel on
   the gathered `(k, 1, 3, n_classes)` array with the full column's summation bound
   (`n_voxels_for_bound`);
3. gathers the WHOLE columns of the cells that fail the test (including every column with
   floating-point room in a nominally full deep voxel, which the whole-column kernel fills) and
   runs the same kernel on them (per-cell independent arithmetic);
4. scatters both groups back and rewrites the totals at exactly the touched voxels with the same
   per-voxel class reduction, so the totals stay bit-identical to `mass.sum(axis=-1)`.

Exactness arguments for the window test are in the module's section comment; they are checked by
the kernel-level differential tests against the original kernels loaded from the accepted snapshot
(`tests/phase7e/test_selective_exchange.py`) and by the two-environment 48-step transaction
sequence (`benchmarks/phase7e/transaction_sequence.py`, 1675 saved fields, Plot1 plus adversarial
multilayer / deep-room / sparse-empty / 1-2-3-voxel beds, zero demand, exhaustion, sub-resolution
and over-capacity refusals).

Fallback cells run FIRST on a gathered copy, so the deposit capacity gate still raises before
`mass` or the totals are touched (atomicity preserved). Nothing is netted or delayed; the active
layer and availability are refreshed every exchange exactly as before.

### Ledger (`coupling/sediment_ledger/accumulate.py`, secondary)

A single-process transfer leaves every other process slice of the four directional channels with
a zero increment; a zero Kahan increment on a slice whose compensation is zero is an exact identity.
When a four-count check proves every foreign compensation slice is zero, the transfer is
Kahan-added on its own slice and the other slices copied; otherwise the original padded path runs.
Bitwise identical to the original in both cases (tests); the upstream instrumentation test that
asserted the old 6x element count is relaxed in the candidate's copy only, with the reason recorded.

### Touched-storage bed change (`water/step.py`, opt-in)

`bed_change_from_touched_storage=True` measures `bed_change_by_class_kg` from the STORED arrays
(active layer after - before over every cell, plus the voxel-mass change over exactly the touched
voxels) instead of the difference of two whole-bed inventory reductions, and gives the validator
the inventory scale from the maintained totals. It stays independent of the transfer totals it is
validated against; it is NOT bit-identical to the whole-inventory difference (different summation),
which is why it is opt-in and reported separately. Default behaviour (two full inventories per
call) is unchanged.

## Counters

`maple.surface.voxels.transfer.selective_statistics` separates: metadata passes over the totals
array (`*_metadata_entries`), kernel voxel-entry accesses on the fast and fallback paths, skipped
cells, whole-array copies for pure state ownership (`whole_mass_copy_entries`), cache rebuilds
(full mass reads), metadata copies, inventory scans and touched-delta bed changes.

## Results

Same-machine sequential Plot1 runs (5400 steps, 32 bins):

| Path | Event loop (s) | MAPLE exchanges (s) |
|---|---:|---:|
| Accepted 72310c49 | 311.967 | 184.010 |
| Selective exchange | 231.545 | 101.519 |
| Selective + touched-storage diagnostic | 217.147 | 92.587 |

The opt-in touched-storage path reduces event time by 30.4% and exchange time by
49.7%. These are single observations, not repeated timing distributions. All 103
saved numeric forcing, hydrograph, final-state and peak-snapshot fields match the
accepted run exactly for both candidates. Existing MAPLE conservation tolerances
are unchanged.

The measured candidate is `1dc70bb4b82c62af70d311d053e7678d06c9d7cabd7f91e25da87ec7a546beab`.
The final candidate `9a4da48c397333eff8a7c8321894843bee7a3df60d7d337c1d16d2408966fb73`
adds rejection of a negative/nonfinite supplied validator inventory scale and
re-reads the touched stored voxels after scatter for the independent diagnostic.
The timed candidate is preserved; timings are not relabeled as final-candidate runs.

Across 10800 erosion/deposition pairs, kernel voxel-entry processing fell from
3,110,400,000 to 125,655,732 (96.0% reduction). Extraction skipped 11,075,602 cells;
deposition skipped 7,865,785 cells. Only 399 deposition cells needed whole-column
fallback. The trusted chain rebuilt totals once and reused them 21599 times.
The opt-in diagnostic eliminated transaction inventory scans, while initial/final
audits remain. Whole mass copies (3,110,400,000 entries) and totals metadata passes
remain: this optimization reduces arithmetic and reads, not all memory traffic.

The normal production adapter is unchanged. Select the isolated dependency and use
`run_storm.py --touched-inventory` to reproduce the fastest path. This explicitly
opts into the trusted state chain and touched-storage diagnostic; it is not an
implicit change to every public MAPLE caller.

### Final reviewed candidate

The post-scatter-audited candidate9a4da48c completed the same storm in **217.246 s**,
with **93.190 s** in MAPLE exchanges (30.36% and 49.36% reductions from the recorded
311.967/184.010 s baseline). All 103 saved numeric fields remain bitwise identical.
Peak RSS was 410540 KiB versus baseline407072 KiB: this change speeds execution,
but does not establish a memory-footprint reduction. Cache validation and class
budgets pass; counters are unchanged from the earlier touched-storage run. See
`benchmarks/phase7e/stage1_reviewed_qualification.json`. Extra touched-index reads
for the post-scatter audit are additional memory traffic beyond kernel-entry
counters. The final runner also now propagates cache audit failures rather than
merely recording an error object; that reporting-only change followed the run.
