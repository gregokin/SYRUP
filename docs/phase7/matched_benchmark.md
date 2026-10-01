# Phase 7 — matched fixed-terrain water/sediment benchmark (design, not results)

Status: implemented 2026-09-30 by Claude for task `phase7_matched_benchmark`
(`src/maple_syrup/benchmark_experiment.py`, `benchmarks/phase7/compare_matched.py`,
`benchmarks/phase7/compare_refinement.py`, `tests/phase7/`, this document).
**Nothing in this task was executed by the author** (no shell): every test,
timing and Plot 1 number is a specification for Codex's independent run. The
results section below is a template to be filled from the executed runs.

## 1. What is compared with what

| Side | Model | Terrain | Splash | Conductivity | Forcing |
|---|---|---|---|---|---|
| Reference | whole MAHLERAN 1.2.3 program, `outputs/phase7/mahleran_deterministic_ksat_run` | `update_topography=n` | dry-cell splash branch disabled by `no_splash.patch` | XML mean 0.00025 mm/s, `deterministic` | rain file, applied with the legacy one-step lag |
| Test | SYRUP Phase 5 step on the actual MAPLE bed, **frozen hydraulic geometry** | `commit=False`, `force_final_commit=False` | none (deferred) | `plot1_parameters` deterministic 0.00025 mm/s, asserted | the reference's logged applied rainfall (`applied_rainfall.csv`) |

Both sides keep rain-assisted wet detachment, method-5 routing, D4 aspects on
the same DEM, the same infiltration model 2 / parameter type 2, friction 21.45,
initial moisture 0.25 and soil thickness 0.3 m (docs/phase7/initialization_audit.md).

## 2. Frozen hydraulic geometry mode

`benchmark_experiment.frozen_control` builds a `SedimentEventControl` with
both commit flags off and refuses any attempt to pass them. `run_frozen_event`
drives the unchanged `evolve_sediment_event` and afterwards proves the freeze
(`frozen_geometry_report`): the routing graph, transport network and physics
grid are the same objects as at t = 0; aspect, slope, receivers, outlets,
active mask, friction and conveyance are array-equal; the committed elevation
and offset are unchanged; commit count and clock are unchanged; zero commits,
zero graph changes, empty commit log. It then runs the actual MAPLE validators
on the wet end state (`validate_frozen_end_state`): active-layer/voxel
partition, availability closure, ledger structure, water state, and the
ledger-to-physical reconciliation against the committed inventory. Because no
commit ran, that inventory is the initial one, so MAPLE's own check confirms
that the pending ledger equals the whole-window physical change per cell and
class within MAPLE's tolerance. A run without actual pickup **and**
deposition is refused as "not a sediment benchmark".

Implications checked in source (pinned MAPLE d3d007024): `apply_water_process_demand`
validates every call independently of commits; the ledger's `n_physical_touches`
is int64; `evaluate_commit_triggers` is never called, so no elapsed-time commit
is scheduled; MAPLE's snapshot writer is not used (a snapshot of an uncommitted
ledger would omit it, and this state is not a handoff). Normal evolving mode is
untouched: `SedimentEventControl()` defaults remain `commit=True,
force_final_commit=True`, and a test runs both modes on the same bed.

Net bed mass change is reported from the actual MAPLE inventory
(`bed_change_by_cell_class_kg`, `morphology_summary`) **separately** from the
frozen elevation; the elevation-equivalent value is a diagnostic only.

## 3. Forcing and conductivity binding

`load_applied_rainfall` reads the audit CSV (`start_s,end_s,logged_applied_rain_mm_h`),
refuses a wrong header, non-finite or negative values, a first interval not
starting at 0 s, non-positive lengths, and any gap or overlap; merges equal
consecutive intensities into one exact piece; and records the file hash,
size, row count, piece count, interval lengths, total depth and the log's
0.01 mm/h precision in a `RainfallProvenance` of kind
`mahleran_applied_rainfall_csv`. The case's original rainfall binding is not
modified; both hashes and both window depths are recorded, together with the
difference (the legacy one-step lag: the first record is applied one extra
second, 15.24 mm/h × 1 s = 0.004233 mm for Plot 1). The run refuses a CSV that
ends before the window and re-hashes the CSV after the loop.

Conductivity: the case's `ksat_m_per_s` array must equal 0.00025 mm/s on every
cell (the value `plot1_parameters` already uses deterministically). With
`--reference-run`, the reference's `ksat_001.asc` interior must be uniformly
that value, its XML must select `deterministic`, its `execution.json` must
record completion, and every `Output/*` hash is recomputed and bound.

## 4. Outputs (new directory, staged then renamed)

- `benchmark_summary.json`: status (wet fixed-window diagnostic), mode and
  freeze checks, MAPLE end-state validation, forcing/conductivity/reference
  binding, resolved config and hash, budgets (water with the MAPLE
  summation coefficient in m³; per-class sediment closure with MAPLE's
  reservoir bound; accumulated unmet and transport bounds), sediment totals
  and net bed mass change, peaks, performance (setup, first step incl. JIT,
  remaining steps, mean per step, `ru_maxrss` at start/end, transfer
  counters), backend (numpy only; GPU unverified), limitations.
- `hydrograph.npz/.csv`: 1 s rows (`--report-every-s 1`): the Phase 4 water
  columns (instantaneous outlet discharge, conservative cumulative export,
  storages, maxima) and the `sed_*` columns (mobile, export rate, cumulative
  pickup/deposition/export, per class).
- `final_state.npz`: voxel/active/available/bound masses, initial active and
  available, initial bed by cell/class, committed elevation (initial and
  final, equal), pending ledger, depth, mobile, soil water, discharge,
  velocity, sediment velocity memory, cumulative pickup/deposition/export
  request, peak depth/velocity, cumulative rain/intake/return/drainage,
  aspect/slope/receiver/outlet/conveyance, water exchange net, bed change.
  These are the WET end-state diagnostics; they are not a restart or a
  handoff and no MAPLE snapshot is written.
- `forcing.npz`: the applied schedule (edges, mm/h) and the original one.

Restart is refused by construction: the CLI has no resume option and the
summary records `restart.supported = false`.

## 5. Comparator (`compare_matched.py`)

Reads only saved outputs; verifies SYRUP output hashes, the MAHLERAN execution
manifest hashes and frozen mode. It reports rainfall equality second by
second and whether reference conductivity is uniform at the SYRUP value;
mismatches are flagged rather than silently treated as matched.

- Water: MAHLERAN `hydro001.dat` column 3 = `q_plot·dx` (mm³/s → ×1e-9 m³/s),
  the outlet set being every active cell whose aspect points into the export
  ring (same definition as SYRUP's outlets); SYRUP `outlet_discharge_m3_s` on
  the same rows. Reported: endpoint-sum integrals for both, SYRUP's
  conservative face ledger separately, peak values, the reference's rounded
  peak plateau, SYRUP's true peak time and its own four-digit plateau, RMSE,
  maximum difference and its time, the rounding floor, Nash–Sutcliffe, and the
  through-1620 s split.
- Sediment: `sedtr001.dat` column 2 (kg/s instantaneous outlet flux) summed
  with dt = 1 s, `seddisch001.dat` per class; SYRUP conservative cumulative
  export total and per class; ratios per class, exported fractions, peak
  rates and timing (reference plateau), cumulative RMSE, cumulative at 1620 s,
  SYRUP pickup/deposition totals. No pass mark: whole-storm sediment agreement
  is a measurement.
- Spatial: ASCII maps cropped of the ring and row-reversed to MAPLE order.
  `depth001` (mm) and `veloc001` (mm/s) are **not** per-cell storm maxima:
  `output_hydro_data_xml.f90` 489-505 copies the whole-domain `d(2)` and `v`
  into `dmax`/`vmax` only when `q_plot` exceeds its running maximum, so the
  files are the synchronous snapshot at the step of the full-precision peak
  outlet discharge (a step inside the rounded plateau; the exact step is not
  recoverable from stock output). They are therefore **excluded** from the
  comparison with SYRUP's per-cell `peak_depth_m`/`peak_velocity_m_s`; the
  comparator reports the definition, labelled statistics of each quantity
  and no error metric or panel. The matching SYRUP quantity is a synchronous
  routed depth/velocity snapshot at SYRUP's own peak-outlet step, recorded by
  a read-only observer sidecar and compared separately.
  `detac001` (kg/cell of accumulated legacy detachment demand, `detach_tot·dx²·ρ·dt`)
  vs SYRUP actual MAPLE pickup; `neter001` (`(detach_tot − depos_tot)·dx²·ρ·dt`)
  vs SYRUP pickup − deposition (water class exchange) and, separately, vs
  minus the actual bed inventory change. `dschg001` has no SYRUP counterpart
  and is not compared. Metrics: sums and ratio, maxima, RMSE, max difference
  and its cell, relative L2, Pearson r.
- Predeclared targets (hydrology at dt = 1 s): integrated outlet ≤ 1 %, peak
  ≤ 1 %, SYRUP peak time within the reference's rounded plateau ± 10 s. A miss
  is reported for investigation, never relaxed.

Preserved equations and departures are those of docs/phase5/physics.md
(legacy-literal conventions, 1 s reference pickup interval, exponential
deposition hazard with upwind advection, particle-density mass with MAPLE bulk
density for elevation, no splash) and docs/phase4/routing.md (coherent old
inflow instead of the legacy stale inflow, proven root bracket).

## 6. Reproduction (Codex)

```bash
cd /home/okin/SYRUP
source agent_handoffs/tasks/phase6_complete_event/env.sh   # pinned MAPLE, Numba, gfortran exports
"$SYRUP_PYTHON" -m pytest -q -rs -p no:cacheprovider tests/phase7
"$SYRUP_PYTHON" -m pytest -q -rs -p no:cacheprovider            # all phases (tests/phase7 is in testpaths)
"$SYRUP_PYTHON" -m ruff check src/maple_syrup tests/phase7 benchmarks/phase7
git diff --check
for dt in 1 0.5 0.25; do
  "$SYRUP_PYTHON" -m maple_syrup.benchmark_experiment --case-dir outputs/plot1 \
    --applied-rainfall outputs/phase7/mahleran_reference_audit/applied_rainfall.csv \
    --reference-run outputs/phase7/mahleran_deterministic_ksat_run \
    --output-dir outputs/phase7/syrup_matched_dt${dt/./p} --max-dt-s $dt --implementation numba
done
"$SYRUP_PYTHON" -m maple_syrup.benchmark_experiment --case-dir outputs/plot1 \
  --applied-rainfall outputs/phase7/mahleran_reference_audit/applied_rainfall.csv \
  --reference-run outputs/phase7/mahleran_deterministic_ksat_run \
  --output-dir outputs/phase7/syrup_matched_dt1_array --max-dt-s 1 --implementation array
"$SYRUP_PYTHON" benchmarks/phase7/compare_matched.py --syrup outputs/phase7/syrup_matched_dt1 \
  --mahleran outputs/phase7/mahleran_deterministic_ksat_run --output outputs/phase7/matched_comparison_dt1
"$SYRUP_PYTHON" benchmarks/phase7/compare_refinement.py \
  --runs outputs/phase7/syrup_matched_dt1 outputs/phase7/syrup_matched_dt0p5 outputs/phase7/syrup_matched_dt0p25 \
  --output outputs/phase7/matched_refinement.json
```

The applied-rainfall CSV was produced from the sampled-conductivity run's log;
the deterministic run applies the identical rainfall (same file, same code
path), and the comparator records its equality second by second from the deterministic
run's own log. Every output directory must be new; the runner refuses paths
inside the case, MAPLE, MAHLERAN, recipe, SYRUP package, applied-CSV and
reference-run trees.

## 7. Executed results

All three full5400s runs (dt1,0.5,0.25), array/Numba parity, actual-device
kernel checks and CPU domain scaling are complete. See [qualification and
limitations](acceptance.md) and [curated evidence](../../benchmarks/phase7/qualification/).
Hydrology meets the original targets; sediment export remains about24times
the reference and is not accepted as faithful. A controlled impulse confirms
a coarse-cell distance-discretization mechanism; exact storm attribution
requires Phase7b. No tolerance was relaxed to pass these runs.

## 7a. Backend note: empty face tables and the pinned MAPLE scatter

Codex's isolated CuPy 14.2 / CUDA 12.9 kernel checks (agent_handoffs/tasks/
phase7_matched_benchmark/gpu_defect.md) found that a pure south-draining plane
has no x-oriented faces, and that the pinned MAPLE `scatter_add`
(`core/backend/scatter.py`, `_flatten_destination` 375-379) reshapes the
block contributions of an empty index over trailing class axes with
`reshape(0, -1)`. MAPLE’s NumPy branch uses `add.at` without that reshape;
the CuPy branch raises "cannot reshape array of
size 0 into shape (0, -1)". `sediment_transport.transport_step` now skips the
x (or y) face-diagnostic scatter when the corresponding static index table
has `size == 0` (a host attribute, no device reduction). No flux is omitted:
a topology without faces of one orientation has no crossing of that
orientation, so the zero-initialised arrays are the correct diagnostics.
`tests/phase7/test_face_topology_backends.py` checks both the pure south and
the pure west chain on CPU (zero absent-orientation arrays of canonical
shape, every crossing and export in the present orientation, per-cell and
per-class balances within the operator's own bounds) and, when an actual
CUDA device is available, field-by-field CPU/CuPy parity. This is a kernel
check; a coupled GPU event remains unsupported and unverified. An upstream
MAPLE change that accepts empty blocks would make the guard redundant, not
wrong; MAPLE itself is unchanged.

## 8. Limitations

See acceptance.md for the executed GPU kernel checks; the following GPU
restriction refers to the complete benchmark runner.

- Wet fixed-window diagnostic: no completion, dry reset, restart, wind or GPU
  (numpy backend only; GPU unverified).
- MAHLERAN stock files are four-significant-digit endpoint samples and are not
  a conservation ledger; SYRUP's internal closure uses MAPLE's bounds.
- Legacy detachment / net erosion maps are demand-based; MAPLE pickup and bed
  change are actual, supply-limited inventories.
- The pending ledger is deliberately left uncommitted; the end state is not a
  MAPLE handoff and must not be fed to a wind event.
- One observational timing per run; not a controlled speed comparison with the
  whole MAHLERAN program.
