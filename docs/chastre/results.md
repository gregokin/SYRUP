# Chastre large-domain water benchmark — CPU and GPU complete

CPU verified on 2026-10-03; GPU runs and independent CPU/GPU audit completed on 2026-10-05. This is a water-only,
fixed-terrain benchmark using actual MAPLE infrastructure and compiled MAPLE bed tiles. Physical GPU 0 (GTX 1080 Ti) was
verified idle (0 MiB, 0% utilization) immediately before tests and again before the benchmark, and idle after it exited.
The runner selected `CUDA_VISIBLE_DEVICES=0`; an in-run probe observed physical GPU 0 at 98% utilization/1375 MiB, with
other devices' memory unchanged. The runner did not capture a CUDA UUID/PCI mapping; physical-device identification is supported
by these consistent probes, not a recorded runtime UUID binding. Other GPU jobs were preserved. No model or benchmark
implementation changed between the CPU and GPU runs.

## Case and protocol

Native `DTM_Chastre_PD4way.asc`: 1393 × 1604 interior cells, **1,140,169 active**, 1 m cells, active elevations 130.68–160.76 m.
Terrain SHA-256: `89ce64747abfc7e60282090ee4db80fd0361bae5427000a2e61153de45435677`.
RFID_2014's approved nearest-neighbour rainfall-map extension, captured applied forcing and all other hydrology/bed parameters
are retained. There is no rainfall renormalization or conductivity recalculation. RFID's canonical XML-based suction/drainage
settings and other inherited differences from uncorrected native Fortran are recorded in the import report; this is not a new
native-Fortran benchmark.

The ring is nodata and closed: **zero outlets**. The network has 161 levels, 73,372 strict pits and 29,144 flat storage cells.
They retain water, with no equal-height routing, overtopping, terrain conditioning or bed evolution. This is a solver and
conservation test, **not a Chastre flood prediction**.

Every contender starts from a fresh state and uses the complete **2700 s RFID benchmark window**, max dt 1 s, reporting every
60 s, 64 bisections or Newton cap 50. One complete warm-up precedes **one measured sample**. These are exploratory, single-sample,
fixed-order timings (bisection then Newton), with no measured spread or order balance. CPU and GPU groups were run separately on different dates. Compilation,
case verification, field downloads/capture and all artifact/source/budget checks are outside the evolution timer.

## CPU/GPU results

| Solver | Measured evolution | Complete warm-up | Accepted / rejected steps |
|---|---:|---:|---:|
| Numba bisection | 1075.76 s (17 min 56 s) | 1134.6 s | 2700 / 0 |
| Numba Newton | 944.33 s (15 min 44 s) | 981.2 s | 2700 / 0 |
| GPU bisection (split CUDA) | 80.48 s (1 min 20 s) | 79.98 s | 2700 / 0 |
| GPU Newton (split CUDA) | 64.87 s (1 min 5 s) | 64.64 s | 2700 / 0 |

CPU Newton used **12.22% less time** in this pair of samples. Fixed order and changing host conditions confound this difference;
it is not a definitive solver-speed ranking. Host: Intel Core i9-7900X; Python 3.12.3, NumPy 2.5.2, Numba 0.67.
The recorded starting host load averages were about 4; unrelated MAPLE jobs remained active and were preserved. CPU thread
environment settings are archived; the timed Numba kernels are serial (no `parallel=True` or `prange`), using one application
thread. Numba's configured pool size was not recorded and is unused by these kernels. CPU model evidence is archived in
`agent_handoffs/tasks/chastre_timing/postrun_lscpu.txt`; the ongoing jobs were operator-observed. This larger case changes resolution, terrain/mask, routing levels and storage structure as
well as cell count, so comparison with RFID is not pure grid-size scaling.

Relative to the saved CPU samples, the GPU speedups were **13.37× for bisection** and **14.56× for Newton**.
GPU Newton took 19.39% less time than GPU bisection in this pair. These are exploratory ratios, not balanced performance
estimates: CPU and GPU groups ran on different dates, with different starting host loads (about 4 for CPU; about 0.49 for GPU).
Both GPU contexts resolved `auto` to **split**, with 161 routing levels and 164 kernel launches per attempted step. The timed
Numba kernels are serial; the CUDA path processes independent cells in parallel within each ordered routing level. At this
size the GPU benefits from many cells per level, unlike the [earlier small-domain tests](../gpu_newton/results.md). No Fortran Chastre timing exists.

All recorded report-time physical hydrograph columns and all final fields agree between the CPU roots within the unchanged
backend bounds (`rtol=2e-12`, `atol=1e-14`). Maximum final differences:

| Field | Maximum absolute difference | Relative L2 difference |
|---|---:|---:|
| Surface depth | 5.3291e-15 m | 1.8394e-16 |
| Soil-water depth | 0 (bitwise equal) | 0 |
| Unit discharge | 1.7347e-17 m²/s | 3.8510e-16 |

The table above compares CPU roots. The independent combined audit also **qualified both same-solver CPU/GPU pairs**
under these unchanged bounds, with zero out-of-bound cells in all final fields:

| GPU versus corresponding CPU | Depth max difference (m) | Soil-water max difference (m) | Discharge max difference (m²/s) |
|---|---:|---:|---:|
| Bisection | 1.7764e-15 | 0 (bitwise equal) | 6.0715e-18 |
| Newton | 1.7764e-15 | 0 (bitwise equal) | 5.2042e-18 |

All report-time physical hydrograph columns also meet the unchanged bounds; counters match. This qualification covers final
saved fields and reported times, not the full interior time history. Both GPU and CPU roots share graph, forcing and column
physics; their agreement is not independent validation against MAHLERAN.

All active cells have positive final surface depth; 1,037,653 have positive final discharge. Maximum reported velocity is
0.173404 m/s. Positive flow confirms the routing path was exercised.
The maximum final depth, **23.8235 m**, is accumulated water in terminal storage under the no-overtopping policy; it illustrates
why these results should not be interpreted as flood predictions.

All four samples have the same independently recomputed aggregate ledger (m³): rain 116297.457199, initial soil 957.741960, final surface 28563.198926,
final soil 86196.776400, soil drainage 2495.223832, export 0. Intake and saturation return are internal transfers.
The total water residual is **−4.1182e-9 m³**, or **3.5411e-14 of rain volume**. The unchanged MAPLE-derived whole-run bound is
2.492583 m³: it counts full-grid operations across all steps and is loose; the actual residual is reported separately.
Flow remains positive at 2700 s. This ends the specified benchmark window, not a dry-again event; no ponded water, soil storage
or flow is discarded or reset.

## Bed, memory and validation cost

The ~32.8 GB dense voxel array is never constructed. **44 actual MAPLE row-band tiles** share the global datum and 306 voxel
layers; each was compiled, reloaded and checked. Generation took 1729.85 s, including 1042.98 s compiling and 587.61 s loading/
checking, with **3.84 GiB process peak RSS**. Full benchmark process peak RSS was **2.828 GiB**, including startup verification
and all contenders; this is not a per-solver memory peak. GPU benchmark process peak RSS was **2.104 GiB**. The largest logged CuPy allocator-pool size was
**1.699 GiB**, including retained allocations across both contenders; it is not a per-solver or whole-device peak.

The model never receives voxel/active-layer arrays during hydrology. Every runtime bed guard nevertheless re-hashes all
persisted tile files: it proves **artifact immutability**, not in-memory sediment evolution. A new, additive benchmark entry
streams independent file hashes with eight threads (16 MiB buffer per worker). It retains the identical hashes, tile order,
guard frequency and failure checks. Nine expected full-manifest calls were verified in each group; CPU cumulative hash time
was 140.89 s. GPU cumulative hash time was 113.88 s (12.65 s/check), all outside evolution timers. CPU hashing took
roughly 16 s/check versus the serial pilot's 90.853 s initial manifest check
(`outputs/chastre/pilot_cpu/comparison.json`, `initial_bed_manifest_check_s`). All hashing is outside the storm timer. On a tiny actual-MAPLE fixture, serial and
parallel guard entries preserve both roots' final fields and report hydrographs bitwise; altered/extra/missing files still fail.

MAPLE's original initial-fill receipt aggregation falsely rejected native tile 1 by one ULP. The isolated dependency changes
only that aggregate gap evaluation to a **compensated signed sum**, keeping the exact original tolerance and all independent
per-cell/class/voxel/inventory checks. An independent rational sum proves the captured tile-1 receipt balances exactly over
its 205,312 reported operands; other receipts were not captured independently. All 44 tiles passed the corrected validator.
The live MAPLE checkout and accepted earlier dependency remain untouched.

The persisted bed is authoritative MAPLE storage; the lightweight hydrology adapter is not a single MAPLE `CompiledCase` and
is **not ready for globally coupled wind, sediment or evolving-bed events**. No splash, detachment, sediment transport,
vegetation growth, nutrients or disk restart is qualified here.

## Evidence and next action

- Case: `outputs/chastre/case_v2`; binding SHA `2f7ae3ef6bfa719f8ad8e61a891afd9af2f00eea2bdcfaf628682eca3e02040c`.
- Raw CPU timings/fields/hydrographs: `outputs/chastre/full_cpu/`; GPU: `outputs/chastre/full_gpu_20261005/`.
  The original serial 60 s pilot is `outputs/chastre/pilot_cpu/`.
- Independent combined audit: `agent_handoffs/tasks/chastre_timing/full_cpu_gpu_numeric_audit_20261005.json`
  (valid, no violations, same-solver fields/routing qualified, all artifacts re-hashed). The earlier CPU-only audit is retained.
- Exact 399-file run archive/pin: `full_timed_source.tar.gz` / `full_timed_source_manifest.json` in that task directory;
  all 399 files matched at the CPU post-run check before documentation edits. At GPU resume and post-run, 398 matched;
  the only difference was `docs/chastre/README.md` (documentation). Final result documents were edited separately after the runs.
  Older partial manifests are not the run pin.
- MAPLE package: baseline `72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65`, isolated receipt candidate
  `ed0afc6c448a53358bbc09467d05b8a52e0909e404eda719ae08b95c26d85f7f`; one changed package file.
- Verification logs: expanded CPU suite 405 passed / 7 skipped / one test-oracle failure; after correcting that oracle, the
  five receipt tests passed; that failure was Claude's incorrect test assertion, not a model failure; 14 parallel-hash tests passed. Ruff passed. These are separate focused runs, not a newly run combined
  test suite. On the idle GPU, `python -m pytest -q tests/gpu_newton tests/chastre` passed **526 tests** in 39.58 s; no skips.

The requested four-contender benchmark is complete. Repeated, order-balanced CPU/GPU measurements would be a separate
follow-up for stronger performance estimates. The closed-mask/no-overtopping and tiled-adapter limits above still apply;
no sediment, evolving-bed, wind or restart qualification follows from this water benchmark. No commit or push was performed.
