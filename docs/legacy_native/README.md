# Native-walk legacy sediment benchmark (CPU, Numba) — task gpu_sediment, stage A1

**Current status (commit of 2026-10-08): see [`../legacy_gpu/current_status.md`](../legacy_gpu/current_status.md).** The Numba full Chastre
2700 s bisection and Newton references have COMPLETED from immutable inputs (single-core loops 5736.873890 s and 5673.070882 s, source/input/post-run
guards passed, archives independently pinned); both pass every same-root GPU comparison at the unchanged bounds and were compared with the completed
original-Fortran reference (observations, including an open recession-water investigation). Numba and llvmlite were restored to both MAPLE environments for adoption regression (see the linked status).
The check runs and large-domain timings below remain historical.

Executed status (documentation correction D1; historical): the code was implemented by a file-only writer and has since been **executed and verified by the
root (Codex)**; the sections titled "not yet executed" below are historical authoring notes, superseded by this paragraph. Root-recorded results
(`agent_handoffs/tasks/gpu_sediment/`, git-ignored): the A1 correction-C3 check run `a1_c3_checks.log` = 246 passed, 2 skipped; the CPU driver is the reference for the full
Plot 1 (5400 s) and full Chastre 2700 s CPU/GPU comparisons. Cumulative GPU/CPU results are in [`../legacy_gpu/results.md`](../legacy_gpu/results.md). The earlier Chastre 600 s CPU run
computed but failed at publication (the A1 records attribute the failed pilot to the module-provenance bug fixed in C3; the failed evidence is preserved,
not relabelled) and its outputs are diagnostic evidence only. Standing scope is unchanged: this is a
fixed-composition, unlimited-supply benchmark replay, not a conservative event, restart or wind handoff, and no MAPLE bed is modified. (A separate CUDA
driver now exists, `maple_syrup.legacy_gpu_driver`; see `docs/legacy_gpu`.)

## What it is, and is not

A reusable driver for the MAHLERAN LEGACY sediment replay on three verified imported cases — Plot 1 (small), RFID_2014 and
the Chastre 1 m case (large) — with the **original `flow_distrib.for` walk semantics** that the accepted Plot 1 replay
(`legacy_transport.py`, `legacy_experiment.py`, both unchanged) cannot express: terminal pits, inactive interior cells, aspect-0
sources and a boundary ring that is not export.

It is a benchmark replay: fixed composition, **unlimited supply**, an explicit artificial clipping source, ring and
inactive-cell deposition kept as diagnostics (never debited), terminal pits that keep their mobile mass, frozen terrain and
routing, 1 s steps, no splash, no ET, no dry reset, a fixed window that ends while flow persists. No MAPLE bed is read after
case verification or written. Not a conservative event, restart or wind handoff (the CUDA port is a separate driver, see `docs/legacy_gpu`). The
physical state comes from the imported, verified MAPLE case (MAPLE dependency provenance is recorded in the summary); wind physics is not copied here.

## Modules

| Module | Role |
|---|---|
| `legacy_native.py` | vectorised `native_network(graph)` (CN donor view and walk view kept separate) and the plain-loop kernels (pack, walk, reduce) |
| `legacy_native_numba.py` | Numba wrappers, `WetLawRunner` (accepted compiled wet-law kernel, preallocated buffers), `StepEngine` (one step on reused buffers) |
| `legacy_case.py` | `legacy_case_for(kind, dir)`: verified case -> hydrology inputs, XML-bound parameters, vegetation, composition |
| `legacy_driver.py` | CLI/driver: step loop, ledgers, maps, snapshots, FAILED record, provenance |

The accepted Crank-Nicolson kernel `legacy_transport._cn` and the accepted wet-law kernel
(`legacy_physics_numba.compiled_kernel`) are reused unchanged. Hydrology is the accepted `hydrology_numba` prepared step
(bisection default, Newton selectable).

## Walk semantics (`flow_distrib.for`)

Every visited cell is credited **before** its aspect is tested; an aspect-0 cell (terminal pit, inactive interior cell, ring) is
credited once and the walk stops. A source whose own aspect is 0 starts to the **west** (the original `else` branch; column 0 =
ring). Tallies: ring, terminal, inactive, aspect-0 source walks, limit stops, `vge` stops, local deposits, zero-slope
no-walks, erase cells. Deposition into an inactive cell and the ring are returned as diagnostics, not subtracted from any
active pool. Zero-slope diffuse cells neither walk nor deposit (probe `zero_slope_diffuse_probe.json`); every other no-law case
deposits locally.

Optional `--legacy-depos-erase` (default OFF, needs `--source-order legacy`): reproduces the order-dependent zeroing of
`depos_soil` at wet no-rain cells (`route_sediment_xml.f90` 166-169) and, as in the no-splash benchmark patch, at dry raining
cells. The accepted replay does not model it; its discarded mass is reported.

## Law-depth time level

`--depth-time-level previous` (default) uses the previous step's depth with the new velocity (as the accepted replay);
`current` is a diagnostic. The Fortran value is the post-infiltration `d(1)`. The A1 text originally did not offer it; since correction
C3 the optional `--depth-time-level post_infiltration` computes the ORIGINAL `infilt.for` `d(1)` natively (see "Corrections C3" below). `previous` stays
the default; the departure of `previous` from the original is a standing qualification and is quantified, not fitted.

## Usage

Environment (this machine; same as the Chastre water benchmarks):

```bash
export PATH=/home/okin/MAPLE/.venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=''
source benchmarks/chastre/env.sh
```

```bash
# Plot 1 (needs outputs/plot1); add --applied-rainfall <csv> to use the legacy applied-rainfall file
python -m maple_syrup.legacy_driver --case-kind plot1 --case outputs/plot1 --output outputs/legacy_native/plot1_60s \
    --end-s 60 --allow-maple-source-change --snapshot-times 30,60
# RFID_2014 full storm
python -m maple_syrup.legacy_driver --case-kind rfid --case outputs/rfid/case --output outputs/legacy_native/rfid_2700 \
    --allow-maple-source-change --snapshot-times 600,1200,1800,2640,2700
# Chastre pilot (60 s, JIT warm-up in-process), then the full window
python -m maple_syrup.legacy_driver --case-kind chastre --case outputs/chastre/case_v2 --output outputs/legacy_native/chastre_60s \
    --end-s 60 --warmup-s 10 --hash-only-tile-verify --allow-maple-source-change --snapshot-times 60
python -m maple_syrup.legacy_driver --case-kind chastre --case outputs/chastre/case_v2 --output outputs/legacy_native/chastre_2700 \
    --warmup-s 60 --hash-only-tile-verify --allow-maple-source-change --snapshot-times 600,1200,1800,2640,2700
```

Use a NEW output path each time. Progress goes to stderr every 60 model seconds (`--progress-every-s 0` disables it) and is
excluded from the timers. Exit status 0 on success; 1 with `<output>.FAILED/FAILED.json` otherwise.

## Outputs

`legacy_ledger.npz`: `ledger (steps, 13, n_classes)` with columns `pickup_kg, deposition_active_kg, deposition_pit_kg,
deposition_ring_kg, deposition_inactive_kg, effective_clip_source_kg, old_mobile_kg, new_mobile_kg, mobile_terminal_kg,
cn_export_kg, endpoint_export_kg, outlet_flux_kg_s, erased_deposition_kg`; per-step walk tallies; water outlet series; per-cell
cumulative detachment / deposition (pits included) / clipping source `(ny, nx, nc)`; final mobile; final depth; masks;
identity residual. `legacy_snapshots.npz` (depth and per-class mobile at requested times, at most 16). `legacy_summary.json`:
limitations, totals by class, terminal-storage census, walk tallies, wet-regime counts, identity guard (per step
`|new - old - (pickup - deposition_active) + cn_export - clip| / sum|terms| <= 1e-10`, otherwise the run is refused and nothing is
published), performance (JIT-inclusive first step, warm-up, component timers, peak RSS), hydrology and kernel provenance, case
record (XML hash and differences from the Plot 1 constants, composition check, vegetation file hash and value), source digests
before/after.

### Corrections C1 (historical authoring note; later executed by the root, see the status paragraph)

* Run it as `python -m maple_syrup.legacy_driver`; no console script is registered.
* Vegetation of RFID/Chastre: the staged raster is validated against the SOURCE RFID mask (104 x 58), then the verified uniform
  value is broadcast to the target grid (Chastre 1393 x 1604). Nothing is resampled.
* Output protection: the output and its `.partial`/`.FAILED` siblings must not equal, lie inside or contain (symlinks resolved)
  the case, project `src/tests/benchmarks/cases`, MAPLE dependency, MAHLERAN reference or any bound input; checked before any
  write, the FAILED marker included.
* `StepEngine.step` accepts only host `numpy.ndarray` float64 of the grid shape (no conversion). After the walk, a compiled
  validation pass refuses NaN/Infinity/negative new pools, fluxes, inflows, depositions, detachments and clipping sources, an
  overflowing cumulative map and a non-finite class sum before anything is accumulated or swapped; any such failure poisons the
  engine (`reset()` or a new engine is required). The run is then abandoned and nothing is published.
* Controls (`--max-memory-gib`, `--progress-every-s`, `--warmup-s`, `--end-s`, iteration counts, snapshot times) are validated
  before the case adapter runs; explicit zeros are validated, not replaced.
* Input artifacts (XML, forcing, vegetation raster, sidecars) are re-hashed after the run and must equal the pre-run hashes.
  Chastre tile immutability is only claimed when `--hash-tiles-after-run` was used (`chastre_tiles.performed`).
* Summary units: `totals` sums only the per-step kg columns; `outlet_flux_kg_s` is a rate (peak reported) and the mobile
  columns are inventories (final values reported). Per-step wet regime counts, the per-step routing budget residual, and the final
  soil water, discharge and velocity are saved. (C1 applied no aggregate whole-storm water bound; **superseded by C2 below**, which adds the
  whole-storm MAPLE-derived volume bound.)

### Corrections C2 (historical authoring note; later executed by the root)

* **Meaningful pilots are 600 s.** The saved 30 s RFID smoke had pickup, deposition and clipping all zero (it is dry; ponding onset
  is only estimated, ~269 s RFID, ~426 s Chastre, from the no-run-on infiltration formula, not measured). RFID pilot: `--end-s 600`;
  Chastre pilot: `--end-s 600 --warmup-s 60`. The summary records `onset.first_positive_pickup_s`,
  `first_positive_deposition_s` and `wet_law_cell_class_steps`; qualify by those, not by the estimate. Short runs (<= 60 s) are
  adaptation checks only. Report cold/JIT (first step, `warmup`), and warm loop times separately; Chastre pilots are expensive.
* **Whole-storm water guard** (`water_budget` in the summary): the exact `compare_plot1.water_budget` rule with the canonical
  `conservation.volume_roundoff_bound_m3(4 n_cells n_steps + 7, largest operand)` for the water, surface and soil residuals and the
  rainfall integral; the expected rain integral is independent of the loop (schedule depth x case-verified rainfall-scale sum x
  cell area). Four per-cell cumulative water depths (rain, intake, saturation return, drainage) are accumulated in place and saved.
  A violation refuses publication. It is a water statement only.

### Helper validation (B1 prerequisite; authoring note, later executed by the root)

`post_infiltration_depth` now validates `out`/`scratch` (exact writable C-contiguous float64 ndarrays of the depth shape) and rejects ANY
memory overlap, including views and read-only inputs, between a buffer and any input or the other buffer (`np.shares_memory`), before
writing anything. The earlier reproducer `out=rain` used to mutate `rain`; it is refused now and covered by tests. The arithmetic is
unchanged; production uses fresh buffers.

### Corrections C3 / T2 (authoring note; the C3 check run `a1_c3_checks.log` = 246 passed, 2 skipped)

* Module provenance is hashed from the actual source files (the driver from its own `__file__`) at the START of the run, so `python -m
  maple_syrup.legacy_driver` no longer fails at publication; the earlier failed Chastre pilot evidence stays as failed evidence.
* `--depth-time-level post_infiltration` (optional, never the default) feeds the wet laws the ORIGINAL `infilt.for` `d(1)`:
  `hpre = max(h_old - max(intake - rain, 0), 0)`, complete branch `intake >= h_old + rain` forced to 0, saturation return excluded.
  `previous` (default) and `current` are unchanged. The summary records the selected convention and whether it is the native option.

Chastre has zero outlets: its export is identically zero and is not evidence. Compare the maps, terminal-storage census and
per-class ledgers.
