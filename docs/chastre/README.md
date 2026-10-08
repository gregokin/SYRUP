# Chastre 1 m terrain with RFID_2014 settings: water-only timing case (EXPERIMENTAL)

Task `chastre_timing`. Case generation and CPU benchmarks completed on 2026-10-03; GPU benchmarks and unchanged-bound CPU/GPU audit completed on 2026-10-05. See [results.md](results.md).

## What it is, and is not

- **Is:** the native 1 m Chastre DTM (`DTM_Chastre_PD4way.asc`, sha256 pinned in `cases/chastre/recipe.yaml`; 1393 x 1604
  interior, 1,140,169 active cells, 102,516 cells with no strictly lower D4 neighbour) carrying the verified RFID_2014 case's hydrology, including its recorded departures from native Fortran,
  with only terrain, cell size, mask and resized rainfall scaling changed (captured applied forcing, fixed-Ksat Smith-Parlange columns, f = 40, theta, soil depth, grains and bed settings),
  for timing the SYRUP water solvers (Numba CPU and CUDA, bisection and Newton) on a large real terrain.
- **Is not:** a Chastre flood prediction (rainfall is the stretched RFID pattern, infiltration/friction/grains are RFID values at
  a 1 m cell), a sediment, splash, wind, evolving-terrain or restart case, a MAPLE `CompiledCase`, or wind/evolving-bed ready.
  No Fortran comparison exists for it.

## Terrain and routing

- Active cells = DTM cells that are not nodata. The one-cell outer ring is nodata (required). No crop, downsample, fill, carve,
  jitter or elevation edit; elevations are the file values.
- `routing.build_routing_graph(..., allow_masked_nodata=True, allow_pit_storage=True, allow_flat_storage=True)`:
  the legacy search (`topog_attrib.for` 94-117: strict `<`, nodata neighbours skipped) leaves aspect 0 / slope 0 where no
  neighbour is strictly lower. Those cells, strict pits AND flats, are terminal storage (`PIT_STORAGE`, conveyance 0, not an
  outlet). Nothing is filled, carved, perturbed or tie-routed, there is no overtopping (iroute 6 excluded) and water never moves
  between equal-elevation neighbours: a plateau is many one-cell stores. A flat cell WITH a strictly lower neighbour still routes.
- `allow_flat_storage` is opt-in, default False, requires `allow_pit_storage`. Without it errors, policy string and digests are
  exactly the previous ones (the policy string gains `;flat_storage=True` only when requested). Kernels are unchanged: the same
  `PIT_STORAGE` receiver/mask/zero-conveyance checks apply.
- **Zero outlets.** The ring is nodata, nodata faces are closed, nothing exports. All storm water ends as infiltration, soil
  drainage or surface storage (pits and in transit). The outlet hydrograph is identically zero. This is a solver timing and
  conservation benchmark.

## Rainfall-scaling map

Source: the verified RFID case's rainfall-scaling interior (104 x 58, south-first). Its 335 masked gap cells take the value of the
nearest valid source cell (squared Euclidean distance between pixel centres; ties go to the first tied valid cell in row-major
order). The gap-filled map is resized to 1393 x 1604 by nearest pixel centre: source index = `(2 i + 1) * n_src // (2 n_dst)` per
axis (integer arithmetic, no interpolation: only source values appear). A target centre exactly on a source boundary takes the
HIGHER index, so the mapping is not mirror-symmetric at ties; the rows 104 -> 1393 have exactly one (row 696), the columns none.
Target nodata cells get 0; every active cell must be > 0. No global renormalization and no K recalculation. The pattern is
stretched ~13.4x in rows and ~27.7x in columns; it is not Chastre rainfall.

## Bed: actual MAPLE tiles, no dense bed

A dense (1393, 1604, 306, 6) FP64 voxel array is ~32.8 GB and is never built. The recipe's `tiles.rows` (default 32) row bands are
each compiled by MAPLE's real `compile_case`, reloaded with `load_compiled_case` and checked with `case_import.check_compiled_plot1`,
one at a time, all sharing the GLOBAL datum offset, nz = 306, 0.1 m voxels, 0.002 m active layer, 1250 kg/m3 bulk density, 2650
particle density, six grain classes and the exact RFID uniform composition. Bed construction is column-local, so the tiles equal
the rows of the dense bed (`tests/chastre/test_tiled_case.py::test_tiles_equal_the_dense_bed_bitwise` proves it on a small grid
for voxel, active layer, availability and elevation). A global datum/nz mismatch is refused before anything is written.
Every band is instantiated, including inactive-only bands, with the RFID placeholder (`min active elevation`, invented, bed only,
not distinguished by MAPLE's compiled masks; the authoritative record is `inactive_interior` in the sidecar). The tile `case.yaml`
header is the shared RFID writer's text and mentions RFID.

`ChastreCase` (the object `rfid_inputs` receives) holds the global `GeometrySpec`, tile 0's class table, a zero water depth and the
tile manifest. It has no voxel/active-layer/availability attributes. The hydrology never receives bed arrays.

## Commands (use NEW output directories)

```
python -m maple_syrup.chastre_case --recipe cases/chastre/recipe.yaml --output-dir <NEW> --plan-only   # writes nothing, exit 3 if disk too small
python -m maple_syrup.chastre_case --recipe cases/chastre/recipe.yaml --output-dir <NEW>               # generate (stderr progress)
python -m maple_syrup.chastre_case --verify-case <case> [--hash-only-tiles]
python benchmarks/chastre/run_chastre_timing.py --case-dir <case> --output-dir <NEW> [--end-s 2700] [--rounds 3] [--contenders ...]
```

## Provenance, failure and verification

`syrup/chastre_binding.json` (written last) pins: recipe, terrain and staged terrain hashes, the RFID source binding/report/fields/
forcing/case-identity hashes, sidecar and forcing hashes, the global grid/datum, per tile the MAPLE case identity, artifact hashes,
every persisted file hash, measured compile and load+check seconds and process peak RSS (`getrusage`, process-wide), the tile
manifest digest, MAPLE/SYRUP provenance and source stability. A failure leaves `syrup/FAILED.json`, preserves partial tiles and
writes no binding. `verify_chastre_case` re-verifies the RFID source, stream-hashes every bound file, RE-COMPUTES the audit
(terrain, gap fill, resize, graph, bed plan) and compares every sidecar array and report section, requires each tile directory to
equal its bound manifest exactly (no extra or missing file) and reloads and checks each tile by default. Estimates printed by
`--plan-only` (state ~1.1 x voxel bytes, compile peak ~10 x one tile's voxel array) are assumptions, not measurements.

## Timing entry and its guard

`benchmarks/chastre/run_chastre_timing.py` reuses `compare_cases.build_syrup_runner/timed_validated/round_orders/compare_finals`
and `compare_plot1.time_sample/sample_record/RunGuard` (only the additive optional `RunGuard(bed_digest=...)` was added; the
default is unchanged). Four explicit contenders; defaults 2700 s, full warm-up, 3 balanced rounds; `--end-s` shorter is a recorded
pilot. The executed single-sample protocol uses the parallel-hash wrapper below; see [results.md](results.md). Progress goes to stderr outside the timers; there is no per-step callback. The bed guard stream-hashes the persisted tiles
before the run and after each validated sample: it reports artifact immutability, not an in-memory bed digest, and costs a full
read of the ~33 GB state each time (outside the timers). `--min-free-gpu-gib` is a freshness proxy, not a utilization check; the
operator/Codex must ensure the GPU is idle. CuPy pool bytes are the allocator pool, not the whole-GPU peak.

## Isolated MAPLE compiler fix (initial-fill receipt evaluation only)

MAPLE's `validate_compiled_case` checked the initial-fill receipt `requested = actual + residual + shortfall` by subtracting four
independently rounded aggregate scalars. On native Chastre tile 1 (32 x 1604 cells, ~1.82e8 kg) the totals were
181988206.2500007, 181988206.25, 6.636e-7 and 0, the ordinary gap was exactly one ULP of the total (2.98e-8 kg) against the
unchanged bound 5.6986e-9 kg, and the compile was refused. The archived per-cell operands
(`agent_handoffs/tasks/chastre_timing/native_receipt_arrays.npz`, `native_receipt_audit.json`) give 0.0 as one
compensated signed sum `math.fsum(requested_cells, -actual_cells, -residual_cells, -shortfall_cells)`. An independent exact-rational sum of the 205,312 captured operands has numerator zero. Classification: a
cancellation false positive of the aggregate-rounding evaluation, not mass loss. This proves only that this receipt balances in
exact arithmetic over the routine's own reported per-cell operands; the independent ground-truth reconciliations (per cell, per
class, per voxel level, inventory, MAPLE's voxel/active-layer validators) are unchanged and still run.

The candidate changes ONLY that gap computation in `case_tools/validators/compiled_case.py` (one compensated signed sum, a local
`import math`); the bound `summation_error_bound_kg(n_cells, capacity_kg)`, all tolerances, operation counts and physical parameters
are the same, and a real imbalance beyond the bound is still refused. `benchmarks/chastre/maple_receipt.patch` is the one-file patch;
`prepare_maple.py` clones the accepted 72310c49 package into a NEW directory, applies it and writes `candidate_manifest.json`;
`verify_maple.py` + `env.sh` (source after the original gpu_env) verify the package, pyproject, editable metadata and manifest, and
refuse unless `candidate_digest.txt` holds the bound, verified candidate digest. The accepted 72310c49 environment, older snapshots and the live
MAPLE checkout are not modified. Shared fix candidate for upstream, not adopted there.

## Verified benchmark environment and parallel artifact guard

On this machine, select the verified isolated MAPLE dependency before any command above:

```bash
source agent_handoffs/tasks/phase7_matched_benchmark/gpu_env.sh
source benchmarks/chastre/env.sh
export PATH=/home/okin/MAPLE/.venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=''  # CPU only
python benchmarks/chastre/run_chastre_parallel.py --hash-workers 8 \
  --case-dir outputs/chastre/case_v2 --output-dir <NEW_OUTPUT> \
  --contenders bisection_numba,newton_numba --rounds 1 \
  --hash-only-tile-verify --allow-maple-source-change
```

The interpreter and third-party packages come from the live MAPLE virtual environment, which can change independently.
The recorded environment in `outputs/chastre/full_cpu/comparison.json` pins the run's executable
(`/home/okin/MAPLE/.venv/bin/python`), `sys.prefix`, and package versions; the source package is the isolated verified candidate.

Replace `<NEW_OUTPUT>` with a new path. `--hash-only-tile-verify` still verifies every file hash and re-audits the entire terrain,
forcing and routing, but reloads tile 0 only; generation already reloaded and checked every tile. The explicit source-change
flag admits the verified receipt-only MAPLE patch relative to the original RFID source binding; within-run source checks remain.
The wrapper changes only file-hash scheduling, not physics, guard frequency or timers. `validation_parallelism.json` binds the
wrapper/runner hashes and worker count; the original comparison file is unchanged. GPU runs require an actually idle device
selected immediately before launch; the free-memory threshold alone does not establish idleness.


To reproduce the completed GPU group, first verify that physical GPU 0 is actually idle, then use the same environment with
`CUDA_VISIBLE_DEVICES=0` and a new output directory:

```bash
export CUDA_VISIBLE_DEVICES=0
python benchmarks/chastre/run_chastre_parallel.py --hash-workers 8 \
  --case-dir outputs/chastre/case_v2 --output-dir <NEW_GPU_OUTPUT> \
  --contenders bisection_cuda,newton_cuda --rounds 1 \
  --hash-only-tile-verify --allow-maple-source-change
```

Check occupancy again after exit. Device choice is an operator decision based on an immediate idle check, not a fixed reservation.
The existing saved CPU/GPU audit is `agent_handoffs/tasks/chastre_timing/full_cpu_gpu_numeric_audit_20261005.json`.
