# Resident CUDA native-walk legacy sediment storm (task gpu_sediment, stages B1, B2 and B3)

**Current status (commit of 2026-10-08): see [`current_status.md`](current_status.md)**, the single current statement of results, evidence
and limits. It supersedes any "in progress" / "pending" wording left on this page and in `results.md`: the full Chastre CPU (Numba) and
original-Fortran references have COMPLETED and been compared (all same-root CPU/GPU comparisons pass the unchanged bounds with exact
integers and bitwise GPU repeats; the Fortran comparison is an observation set with an open recession-water investigation). The P2 CUDA
candidate and the golden-helper candidate are NOT adopted. No GPU test or full-storm benchmark was rerun for the commit; fresh CPU verification is recorded in `current_status.md`.
Numba is absent from the current environments, so the compiled-test and CPU-timing results below are historical.

Executed status (documentation correction D1, 2026-10-06/07; historical). The code was authored with file-only tools (the author never ran it); the
ROOT (Codex) then compiled, tested and ran it on a GTX 1080 Ti and recorded the results in `agent_handoffs/tasks/gpu_sediment/` (git-ignored). Those
executed results are collected, with their qualifications, in [`results.md`](results.md). In short: the B3C1 real-GPU suite passed
(240 passed, 2 single-visible-device skips; CPU guards 89 passed / 20 GPU skips; Ruff PASS), full Plot 1 CPU/GPU comparisons pass at the
unchanged bounds, and full Chastre 2700 s GPU repeats are bitwise reproducible. Where a section below says what a change "is
meant to" do, the executed outcome (if any) is in `results.md`; timings there are exploratory. The CPU reference
(`maple_syrup.legacy_driver`, the A1 modules and the immutable CPU snapshot) is unchanged.

## Physical scope (read this first)

This is the MAHLERAN **legacy replay** (fixed composition from the verified initial active layer, UNLIMITED supply, an explicit
artificial clipping source, no MAPLE bed read or written). It is a benchmark-only legacy run. It makes **no conservation claim**, is not
a conservative complete-event model, has no restart and no wind handoff, and its legacy behaviour is not MAPLE authority. The wet-law
depth time level is chosen with `--depth-time-level` (`previous` default = the previous step's routed depth; `current`; native
`post_infiltration`); the velocity is always the new step's. Splash, ET, ecohydrology, dry reset, evolving terrain/routing and the
order-dependent legacy erasure / legacy source order are NOT supported.

## What it is, and is not

`python -m maple_syrup.legacy_gpu_driver` runs the SAME benchmark-only MAHLERAN legacy replay as the CPU driver, entirely on the GPU:
rainfall field -> the accepted CUDA hydrology (`hydrology_cuda`, bisection or Newton) -> wet laws -> native detachment-distance walk ->
the accepted Crank-Nicolson pool -> ledger and maps. Fixed composition, UNLIMITED supply, an explicit artificial clipping source, ring and
inactive-cell deposition as diagnostics, terminal pits that keep their mobile mass, frozen terrain/routing, 1 s steps, no splash, no ET,
no dry reset. It is not a conservative MAPLE-bed event, restart or wind handoff; no second wind model exists. No CPU fallback.

## Design (modules)

* `src/maple_syrup/legacy_native_cuda.py`: kernels, the resident context, static walk tables, memory estimate.
* `src/maple_syrup/legacy_gpu_driver.py`: CLI/driver reusing the CPU driver's controls, protection, guards, water closure, identity guard,
  publication, ledger/summary contract.
* Kernels: `sg_laws` (thread per cell: the wet laws ported term for term, `--fmad=false`, no fast math, plus the CPU validation flags),
  `sg_values` (thread per source/class: the `flow_distrib` walk once, writing each credit to its own record), `sg_gather` (thread per
  target/class: sums its records in CPU source order starting from 0.0, so the additions equal the CPU `depos += ...`), `sg_ring_inactive`,
  `sg_cn_level` / `sg_cn_block` (ordered S, W, N, E donor sums; the actual negative trial and the clip source written explicitly),
  `sg_reduce_partial` + `sg_reduce_final` (fixed block partials in index order, no floating point atomics; cumulative maps; the CPU
  `_check_py` flags), `sg_outlet`, `sg_tally` (integer walk and regime tallies), `sg_post_infiltration` (the original `d(1)`).
* Static tables (`build_walk_tables`): per source one local record plus the credited cells along the downstream path (ring, terminal pit
  and inactive stop kinds, aspect-0 sources start WEST), a target-sorted CSR of record ids (ascending = CPU source order), the ring records
  and the inactive targets. Nothing grows with time. For Chastre the census predicted about 10.6 M records: values `records x 6 x 8` bytes
  (about 0.5 GB for six classes) plus about 0.1 GB of tables. `estimate_bytes` runs BEFORE any allocation and the build fails early if it
  exceeds the budget or the free device memory.
* Not supported, rejected before any mutation (never silently ignored): `--source-order legacy`, `--legacy-depos-erase`,
  `--allow-python-kernels`. Zero-mass classes are not skipped in the PHYSICS (laws, regimes, virtual velocities, the CN pool including
  any seeded mobile mass, ledgers and cumulative maps of all classes are evaluated); only the walk record values are compacted (B2 below).
* Device failures: per-step flag words accumulate on the device and are read in slices every `--check-every-steps` (default 60) and at the
  end together with the ledger rows and tallies; a nonzero word poisons the context (`reset()` required), the per-step identity guard (1e-10)
  runs on the rows just read, and nothing is published (`.partial` -> `.FAILED`).
* Depth time levels: `previous` (default), `current`, native `post_infiltration` (device kernel, same formula as the CPU helper).

## B2: compact walk records (`--record-strategy compact|all`, default `compact`; root real-GPU gates executed, see `results.md`)

Only the record values `values[record, class]` are compacted: they hold the `ne` classes that can detach, `n_records x ne x 8` bytes
instead of `n_records x 6 x 8`. Eligibility (`eligible_record_classes`) is static: class `k` is a record class iff some ACTIVE cell has
fraction `> 0` in the immutable composition (host work once, no per-step scan, no GPU field read). Proof that the others carry no value:
in `sg_laws` both detachment terms of a class are forced to exactly 0 where `!(f > 0)`, inactive cells request 0, so `det = +0`; the walk
then takes no branch and every record, local credit, code and deposition of that class is `+0` (never written; those arrays start at
zero and `reset()` zeroes them). Zero composition does NOT imply zero velocity or zero mobile mass: laws, regimes, velocities, the CN pool
(including seeded pools), ledgers and cumulative maps keep all physical classes. With zero record classes no record kernel is launched and
`values` is one placeholder element. `--record-strategy all` is the B1 reference (identity class map). The record-class count is compiled
in (`NE`), part of the kernel cache key `(device, nc, ne)`, the kernel provenance hash, the sealed context scalars and the summary
(`gpu.record_strategy`: strategy, record/excluded classes, estimated and all-class record bytes; `walk_tables.record_values_bytes_allocated`).
`walk_only` (a test hook) refuses forced detachment in a non-record class BEFORE any mutation; use `record_strategy='all'` for that
diagnostic. Effect on Chastre (2 of 6 classes; record classes 3 and 4), as executed by the root: record values 510,204,672 -> 170,068,224 bytes
(340,136,448 bytes saved), allocated equal to estimated; in the isolated wet-state Chastre replay (60 steps, three passes, an actual computed
600 s water state from the old failed-publication CPU run, NOT an accepted full storm) every ledger, map, mobile, recession-velocity, count and
flag was bitwise equal between the two strategies in both orders, and the median sediment time was about 9.0% (all 4.042 s vs compact 3.678 s)
and about 12.5% (reverse order 4.253 s vs 3.722 s) lower: exploratory single-device timings under recorded background load, not a full-storm gain.
The eligibility is valid only for an event's IMMUTABLE composition; an evolving active layer can expose classes absent initially, so this
cannot be reused blindly by a future conservative event (`production_GPU_followups.md` in the git-ignored task directory). A direct-gather
alternative is deliberately not implemented.

## B3: fused water accounting (`--water-accounting fused|separate`, default `fused`; no full-Chastre gain established)

Executed by the root (see `results.md` and `current_status.md`): the B2 GPU gates (full suite 191 passed / 1 single-visible-device skip, corrected
compaction tests 31 passed) and, for B3 after correction C1, the real-GPU full suite (240 passed, 2 single-visible-device skips, 131.50 s; CPU guards
89 passed / 20 GPU skips; Ruff PASS) and full Plot 1 5400 s runs in both modes and both solvers (all saved fields/snapshots/counters bitwise equal
between the two modes; each mode passes the CPU comparison at the unchanged bounds with 0 flags). The full Chastre B3 eight-run matrix has since
COMPLETED: both modes pass the full same-root CPU references at the unchanged bounds, between-mode outputs are bitwise identical, and the fused
means were about 0.2% HIGHER than separate (within run-to-run spread), so **no full-Chastre B3 gain is claimed**. The default stays `fused`
because the two modes are bitwise identical in output; `separate` keeps the exact reference arithmetic. The description below is of the design;
the executed numbers are in `results.md`.

After each hydrology step the driver records, device-only, the series row `[outlet_discharge_m3_s, export_m3, budget_residual_m3]` and four cumulative
maps `cum += x` (rain, intake, saturation return, drainage). These are DEVICE scalar views of the hydrology packet, never host scalars; the single counted
144-byte packet read per step, the absence of any extra host read/H2D copy and the per-step flags/check cadence are unchanged. `separate` is the B1/B2
reference (3 scalar device copies + 4 `cupy.add`, 7 device operations per step). `fused` is ONE launch of `sg_water_account` per step (new module
`src/maple_syrup/legacy_water_cuda.py`: global thread 0 copies the three packet doubles, every thread adds one cell of the four sources into the four maps,
one IEEE addition per cell and map, `--fmad=false`, no atomics, no reduction), so the per-step water accounting drops from 7 to 1 device operation (6 fewer
launches/step; the three series copies and four additions were already DEVICE-to-device scalar-view operations, so no host-to-device copy was saved).
The hydrology `CudaDiagnosticsAccumulator` was not reused: it needs six grids including unused peak maps plus a separate report reduction,
so a narrow helper was written. Series shape `(steps, 3)`, float64, time level, units and bit patterns, and the cumulative maps' per-step additions are
meant to be bit-identical in both modes (a backend implementation difference, never a forcing difference). Guards before any write: creation device,
sealed scalars and array metadata, strict consecutive rows, packet type/size/device and aliasing of the step's scalar views, float64 contiguous grids on
the current device, no source/destination overlap; each of the three step scalars must be EXACTLY a 0-d float64 cupy view of the creation device at its
packet word (a uint64/float32/vector view sharing the address, a host scalar or a pointer impostor is refused); missing `column`/`route` fields are a typed
`WaterAccountingError`; a write-phase failure poisons until `reset()`. Setup is reported outside the loop: `performance.water_accounting_setup_s`,
`water_accounting_compile_s`, `water_accounting_compile_cached` (False = first use in the process, an NVRTC compile; the warm-up loop's accountant makes the
measured loop's a cache hit) and the same keys in `performance.warmup`; they are excluded from `loop_wall_s_excluding_progress`. Counters: `summary["gpu"]["water_accounting"]` (mode,
operations per step, launches, bytes, kernel provenance); `gpu.transfers.launches` / `launches_per_step_sediment` keep counting SEDIMENT launches only
(`launches_scope`). Memory: unchanged, `(steps x 3 + 4 x ny x nx) x 8` bytes, allocated once per loop; no per-step allocation. Event split: the accounting is
enqueued after the sediment-end event, i.e. it lies in the `previous_sediment_end_to_hydrology_end` segment of the NEXT iteration (and the last step's lies in
no segment); that segment remains NOT isolated hydrology.

## Host/device traffic

Per step the hydrology's own 144-byte packet read, plus (not "only") every `--check-every-steps` steps and at the end the sediment
flag/ledger/tally slices; the driver's snapshot downloads (depth, mobile) at the requested times; one set of final water maps and the
volume sums; the sediment maps once. Static data are uploaded once and the hydrology context's own static uploads are not included in the
sediment counters. Counts and bytes are in `summary["gpu"]["transfers"]` (with `scope`, the warm-up counters `before_reset_warmup`, and the
`d2h_driver_*` fields), memory in `summary["gpu"]["memory"]`, launch counts and timings in the summary. `event_split_s` segments are
labelled honestly: the first contains hydrology plus water accounting plus host reads and is NOT isolated hydrology time.

## Commands (Codex; device selection and idle check by UUID are Codex's)

```bash
source agent_handoffs/tasks/phase7_matched_benchmark/gpu_env.sh; source benchmarks/chastre/env.sh
export PATH=/home/okin/MAPLE/.venv/bin:$PATH; export CUDA_VISIBLE_DEVICES=<verified idle device>
python -m pytest tests/legacy_gpu -q                     # CPU-only tests run anywhere; GPU tests skip only without CuPy/device
python -m maple_syrup.legacy_gpu_driver --case-kind plot1 --case outputs/plot1 --output outputs/legacy_gpu/plot1_40 --end-s 40 \
   --applied-rainfall outputs/phase7/mahleran_reference_audit/applied_rainfall.csv --allow-maple-source-change
python -m maple_syrup.legacy_gpu_driver --case-kind rfid --case outputs/rfid/case --output outputs/legacy_gpu/rfid_600 --end-s 600 \
   --allow-maple-source-change --snapshot-times 300,600
python -m maple_syrup.legacy_gpu_driver --case-kind chastre --case outputs/chastre/case_v2 --output outputs/legacy_gpu/chastre_2700 \
   --warmup-s 60 --hash-only-tile-verify --allow-maple-source-change --root-solver newton --snapshot-times 600,1200,1800,2640,2700
python benchmarks/legacy_gpu/compare_cpu_gpu.py --cpu <CPU dir> --gpu <GPU dir> [--gpu-repeat <GPU dir 2>] --output <NEW report.json>
# compact vs all-class record strategy (same coupled storm, each its own new output directory)
python -m maple_syrup.legacy_gpu_driver ... --record-strategy all     --output outputs/legacy_gpu/<name>_all
python -m maple_syrup.legacy_gpu_driver ... --record-strategy compact --output outputs/legacy_gpu/<name>_compact
# B3: the same storm in both water-accounting modes (equal settings; repeat in the opposite order; cold/JIT reported separately)
python -m maple_syrup.legacy_gpu_driver ... --water-accounting separate --output outputs/legacy_gpu/<name>_wsep
python -m maple_syrup.legacy_gpu_driver ... --water-accounting fused    --output outputs/legacy_gpu/<name>_wfus
# full Plot 1: add --end-s omitted/5400 and --root-solver bisection|newton; Chastre 2700: --warmup-s 60 --hash-only-tile-verify ... as above
# all saved arrays of the two GPU outputs must be byte-identical; each is then compared to the CPU with compare_cpu_gpu.py
# isolated sediment microbenchmark on supplied wet inputs (state npz: depth_m, velocity_m_s, rain_m_s); repeat with --order compact,all
python benchmarks/legacy_gpu/record_strategy_microbench.py --case-kind chastre --case outputs/chastre/case_v2 --state-npz <wet_state.npz> \
   --steps 60 --repeats 3 --order all,compact --hash-only-tile-verify --allow-maple-source-change --output <NEW report.json>
```

### Comparison harness contract (`compare_cpu_gpu.py`)

Output integrity (B2 C1): the GPU driver records `output_pins` in the summary (SHA-256, byte count and zip member names of the closed
`legacy_ledger.npz` and optional `legacy_snapshots.npz`, streamed once after they are written and outside the measured loop; the summary
cannot pin itself) and a `qualification` block (case kind, the unchanged bounds, the legacy-replay scope statement). The harness verifies
the pins before reading the archives; a truncated/altered body, a size or hash mismatch, a missing pinned file or an unlisted archive is an
unusable artifact. Receipts without pins (B1, CPU outputs) are accepted and reported as `unbound` - never trusted via a marker alone, and
never rewritten. No CRC re-read of the arrays is made; the pins are what detect a change after publication or an environment migration.

Acceptance is strict and every failure is a flag: unreadable/truncated or incomplete artifacts (a missing required array is a failure,
never a skip); mismatched saved receipts (case kind, steps, dt, end, depth time level, source order, erase convention, snapshot request,
hydrology root solver and iteration controls, graph hash, bound-input hashes, network summary) are refused BEFORE any numeric comparison;
every compared field must be an all-finite float array of the same shape (NaN/Inf never pass); the time series, outlet/export series and
requested snapshots are compared at the same declared bounds (sediment 2e-11/1e-14, water 2e-12/1e-14); integer tallies, masks and the
time axis are exact. The clipped-pool identity residual is a separate cancellation diagnostic: finite, equal to the residual recomputed
bitwise from the run's own ledger, individually within the unchanged `legacy_driver._check_identity` guard (1e-10), and its cross-backend
difference bounded only by the declared ledger bounds propagated through the six unit-coefficient columns plus the standard 5u/(1-5u)
evaluation error; a violation stays flagged. `--gpu-repeat` demands byte-identical repeats of every saved array, snapshot field and saved
summary counter. No tolerance was changed and none is derived from CPU/Fortran noise floors.

Full Plot 1 (5400 s) and Chastre 2700 s bisection/Newton are launched by Codex with the same CLI; timings are only meaningful after the
`--warmup-s` window; cold compile/context costs are reported separately. Timings are exploratory until replicated and order-balanced.

## Limits

At the Plot 1 size the GPU loop is SLOWER than the CPU Numba loop in the recorded runs (small grid, launch-overhead dominated; see `results.md`);
no GPU speed-up is claimed for any case. Chastre has 169 sediment launches per step and 161 routing levels; fewer ordered CN launches are a
profiled follow-up, not done. The fixed-composition legacy replay is not the MAPLE bed: an evolving-bed GPU event needs sparse exchange tracking and a
voxel-backed active-layer cache (a full Chastre voxel bed exceeds the 11 GiB device) and must remove the artificial clip source through an accepted
exchange scheme (`production_GPU_followups.md`, git-ignored task directory). Current results, the completed Fortran comparison and the open
recession-water investigation are in [`current_status.md`](current_status.md).

GTX 1080 Ti (Pascal) has slow FP64 and 11 GiB; the ring accumulation is a sequential single-block loop (fine for the cases here, a large
number of ring records would be slow); the CN launch structure is per level (161 launches/step on Chastre) or one fused block on narrow
networks; the walk records are a time-invariant bounded geometry, but their VALUES are rewritten every step. The comparison with the CPU is
done at the unchanged declared bounds; a CPU root-to-root noise floor is NOT an acceptance criterion.
