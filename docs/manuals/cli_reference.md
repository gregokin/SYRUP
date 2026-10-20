## Appendix A. Complete CLI option reference

This reference was generated from the argument parsers in committed SYRUP d05d8c6, using the same captured help and parser actions. Every installed command and the seven module-only utilities listed in the guide are included. Aliases appear together. A False Boolean flag is off until supplied; True/False here is a switch state, not a value to type. A required option must be supplied even if its parser default is null. “Not supplied” frequently resolves from the case or another option; important resolutions are described in the guide and the meaning column. Every command also accepts `-h` / `--help`, which prints help and exits. Quoting string choices is optional in ordinary Bash usage.

The generic benchmark has two interfaces: its default execution delegates to the legacy parser, while generic `--help` shows the explicitly conservative branch. This table distinguishes them. Parser acceptance does not override documented scientific restrictions or validation. In particular, a parsed unsupported legacy option is deliberately refused, and method-5 terminal storage does not implement spill.

The tables reproduce parser defaults, not a claim that every combination is compatible. Resource limits do not become scientific tolerances. Keep all output paths new and use matching source/backend/configuration for restart.

### A1. maple-syrup-probe

Dependency and backend probe.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--backend` | "numpy"; choices: numpy, cupy | Array backend; select the matching implementation where the runner exposes both controls. |
| `--device-id` | 0; int | Visible-device index, after any CUDA_VISIBLE_DEVICES remapping. Default 0. |
| `--expected-maple-root` | Not supplied (runtime resolution) | Refuse to run unless the imported MAPLE resolves to this source root (default: $MAPLE_SYRUP_EXPECTED_MAPLE_ROOT when set). |
| `--output` | Not supplied (runtime resolution) | Also write the JSON report to this NEW file. |
| `--include-file-manifest` | false | Include per-file SHA-256 entries in the provenance record. |

### A2. maple-syrup-plot1-import

Corrected Plot1 importer; compilation is on unless --no-compile is passed.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--recipe` | Required | Authored recipe, e.g. cases/plot1/recipe.yaml |
| `--output-dir` | Required | NEW directory for the case package. |
| `--mahleran-root` | Not supplied (runtime resolution) | Override the recipe's MAHLERAN root (read-only). |
| `--expected-maple-root` | Not supplied (runtime resolution) | Refuse any other MAPLE source root. |
| `--no-compile` | false | Write the audited case package but do not call MAPLE's compiler. |

### A3. maple-syrup-columns

Independent infiltration columns; no routing or sediment.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--case-dir` | Required | Verified Phase 2 case, e.g. outputs/plot1 |
| `--max-dt-s` | Required; float | Largest step (s); knots always split. |
| `--output-dir` | Required | NEW directory for the summary and final grids. |
| `--backend` | "numpy"; choices: numpy, cupy | Array backend; select the matching implementation where the runner exposes both controls. |
| `--mahleran-root` | Not supplied (runtime resolution) | Override the recipe's MAHLERAN root (read-only). |
| `--expected-maple-root` | Not supplied (runtime resolution) | Refuse any other MAPLE source root. |
| `--allow-maple-source-change` | false | Run even if MAPLE's source digest differs from the import's (recorded). |

### A4. maple-syrup-storm

Water-only directed storm to a specified end time; not event completion.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--case-dir` | Required | Verified Phase 2 case, e.g. outputs/plot1 |
| `--output-dir` | Required | NEW directory for summary, grids and hydrograph. |
| `--max-dt-s` | 1.0; float | Largest step (s); knots always split. |
| `--end-s` | Not supplied (runtime resolution); float | Simulated end (s); default the legacy stormlength (5400 s for Plot 1). |
| `--implementation` | "numba"; choices: array, numba, cuda | Hydrology implementation (default numba; missing Numba is an error, no fallback). 'cuda' is the explicit water-only prepared CUDA hydrology: it REQUIRES --backend cupy and a CUDA device and never needs Numba (no fallback). |
| `--backend` | "numpy"; choices: numpy, cupy | Array backend; select the matching implementation where the runner exposes both controls. |
| `--root-solver` | "bisection"; choices: bisection, newton | Cell-equation root solver (default bisection, unchanged). 'newton' is the explicit safeguarded Newton (CPU array/numba, or CUDA with --implementation cuda). |
| `--newton-max-iterations` | 50; int | Newton pass cap per cell, 1..1000 (only meaningful with --root-solver newton). |
| `--report-every-s` | 60.0; float | Hydrograph row cadence (s). |
| `--min-dt-s` | 0.0009765625; float | Retry floor in seconds; a forcing/report boundary may itself require a shorter slice. |
| `--max-retries` | 10; int | Maximum recoverable step retries before a refusal. |
| `--max-steps` | 10000000; int | Guard on accepted steps; exceeding it is not physical completion. |
| `--mahleran-root` | Not supplied (runtime resolution) | Override the recipe's MAHLERAN root (read-only). |
| `--expected-maple-root` | Not supplied (runtime resolution) | Refuse any other MAPLE source root. |
| `--allow-maple-source-change` | false | Run even if MAPLE's source digest differs from the import's (recorded). |

### A5. maple-syrup-sediment

Conservative sediment time window; no complete-event reset or restart.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--case-dir` | Required | Verified Phase 2 case, e.g. outputs/plot1 |
| `--output-dir` | Required | NEW directory for summary, grids, hydrograph, snapshot. |
| `--max-dt-s` | 1.0; float | Largest step (s); knots always split. |
| `--end-s` | Not supplied (runtime resolution); float | Simulated end (s); default the legacy stormlength (5400 s for Plot 1). |
| `--implementation` | "numba"; choices: array, numba, cuda | Hydraulic sweep implementation (default numba; missing Numba is an error, no fallback; 'cuda' is the explicit prepared CUDA hydrology and requires --backend cupy). |
| `--backend` | "numpy"; choices: numpy, cupy | numpy (array/numba hydraulics) or cupy (ONLY with --implementation cuda: whole small bed on the device, memory preflight first); other pairings are refused, no fallback. Select the device with CUDA_VISIBLE_DEVICES before launching. |
| `--root-solver` | "bisection"; choices: bisection, newton | Cell-equation root solver of the hydraulic sweep (default bisection). |
| `--cuda-mode` | "auto"; choices: auto, fused, split | CUDA hydrology launch structure (auto/fused/split; cupy backend only). |
| `--report-every-s` | 60.0; float | Hydrograph row cadence (s). |
| `--min-dt-s` | 0.0009765625; float | Retry floor in seconds; a forcing/report boundary may itself require a shorter slice. |
| `--max-retries` | 10; int | Maximum recoverable step retries before a refusal. |
| `--max-steps` | 10000000; int | Guard on accepted steps; exceeding it is not physical completion. |
| `--sediment-courant-max` | 0.5; float | Sediment Courant cap per substep (<= 0.5 for characteristic, <= 1 for upwind). |
| `--max-transport-substeps` | 64; int | Cap on internal sediment substeps per physical step. Exceeding it rejects the attempt; it never truncates a path or deletes mass. |
| `--transport-scheme` | "characteristic"; choices: characteristic, upwind | Lateral transport: characteristic (Phase 7b phase bins, default) or the Phase 5 upwind operator kept as an explicit comparison. |
| `--phase-bins` | 32; int | Position bins per cell/class for the characteristic scheme (1..128; 32 is a candidate default, not an accepted convergence). |
| `--transport-implementation` | "auto"; choices: auto, array, numba, cuda | Characteristic kernel: auto (follow --implementation), array or numba (dense storage); numba or cuda (csr storage); no fallback. |
| `--phase-storage` | "dense"; choices: dense, csr | Phase storage of the characteristic scheme: dense (default, the accepted (ny,nx,nc,B) arrays) or csr (the accepted sparse nonempty-bin state; same physics and bins). |
| `--phase-max-nnz` | Not supplied (runtime resolution); int | csr only: predeclared capacity (stored bins) refused BEFORE allocation when exceeded; never truncates mass. Required by the CUDA operator for large cases. |
| `--no-commit` | false | Do not evaluate MAPLE commit triggers per step. |
| `--no-final-commit` | false | Do not force a final commit at end_s. |
| `--mahleran-root` | Not supplied (runtime resolution) | Override the recipe's MAHLERAN root (read-only). |
| `--expected-maple-root` | Not supplied (runtime resolution) | Refuse any other MAPLE source root. |
| `--allow-maple-source-change` | false | Run even if MAPLE's source digest differs from the import's (recorded). |

### A6. maple-syrup-event

Complete conservative evolving-bed Plot1 event, Dense storage.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--case-dir` | Required; Path | Verified imported case package directory. |
| `--output-dir` | Required; Path | New output directory. Existing output paths and protected input/source locations are refused. |
| `--resume` | Not supplied (runtime resolution); Path | Existing source/config/backend-bound wet checkpoint bundle, not a completed dry handoff. |
| `--checkpoint-dir` | Not supplied (runtime resolution); Path | Checkpoint destination; default is the output path plus _checkpoints. |
| `--max-checkpoint-bytes` | 1073741824; int | bound on the summed uncompressed payload bytes of the --resume bundle (positive integer, default 1 GiB); a resource control, not part of the run identity |
| `--checkpoint-every-s` | 600.0; float | Checkpoint scheduling interval in model seconds; this does not change hydraulic routing. |
| `--pause-at-s` | Not supplied (runtime resolution); float | Pause at a future original report/forcing boundary before the duration limit; preserve wet checkpoint. |
| `--max-end-s` | Not supplied (runtime resolution); float | Maximum event duration in seconds. Reaching it without physical completion is a refusal, not drying. |
| `--max-dt-s` | 1.0; float | Largest physical timestep in seconds; forcing/report boundaries and accepted retries can make it shorter. |
| `--implementation` | "numba"; choices: array, numba, cuda | hydraulic sweep: numba (default) or array on the host; cuda requires --backend cupy |
| `--backend` | "numpy"; choices: numpy, cupy | numpy (array/numba) or cupy (ONLY with --implementation cuda); other pairings refused |
| `--bisection-iterations` | 40; int | explicit root iteration budget; accuracy and conservation bounds unchanged |
| `--root-solver` | "bisection"; choices: bisection, newton | Bisection or safeguarded Newton for the same directed hydraulic equation. |
| `--cuda-mode` | "auto"; choices: auto, fused, split | CUDA hydrology launch structure (cupy backend only) |
| `--allow-maple-source-change` | false | accept an explicitly recorded isolated MAPLE source difference from the import binding |
| `--allow-pit-storage` | false | opt-in: active strict sinks become terminal-storage cells (deposit, keep water, no export) |
| `--allow-flat-storage` | false | opt-in with --allow-pit-storage: flat sinks are terminal storage too |
| `--allow-masked-nodata` | false | opt-in: inactive / ring cells may hold --nodata-value and are never receivers |
| `--nodata-value` | Not supplied (runtime resolution); float | the DEM nodata sentinel for --allow-masked-nodata |
| `--terminal-storage-water` | false | completion option: flow cessation with water ponded ONLY in terminal-storage cells (removed and accounted by the external dry reset); needs --allow-pit-storage |
| `--deposit-mobile-at-flow-end` | false | CW-B51 completion option: explicit dry-again assumption; after rain ended and every hydraulic quiet gate held for --hold-s, the remaining finite mobile load is deposited through MAPLE (reported separately); not a mass tolerance; default off |
| `--record-exchange-history` | false | CW-D2: record the actual per-cell/class MAPLE ledger touch counts and gross activity through every commit (exchange_history.npz); diagnostic evidence only, default off |
| `--report-every-s` | 60.0; float | Reporting cadence in model seconds. Forcing knots may add rows; it does not write spatial movie frames. |
| `--hold-s` | 60.0; float | Continuous accepted hydraulic quiet period required before completion, in seconds. |
| `--max-depth-m` | 1e-08; float | Maximum moving surface depth allowed by the stopping policy (m); physical quiet gate. |
| `--surface-volume-m3` | 1e-06; float | Maximum moving surface-water volume allowed by the stopping policy (m³). |
| `--max-unit-discharge-m2-s` | 1e-10; float | Physical stopping threshold on unit-width discharge (m²/s). |
| `--outlet-discharge-m3-s` | 1e-09; float | Physical stopping threshold on summed outlet discharge (m³/s). |
| `--mobile-per-cell-class-kg` | Not supplied (runtime resolution); float | physical stopping significance; default actual MAPLE mass_resolution_kg, not a conservation tolerance |
| `--max-steps` | 10000000; int | Guard on accepted steps; exceeding it is not physical completion. |
| `--max-report-rows` | 100000; int | Guard on the number of report rows planned/stored. |
| `--transport-scheme` | "characteristic"; choices: characteristic, upwind | characteristic (Phase 7b phase bins, default) or the Phase 5 upwind comparison |
| `--phase-bins` | 32; int | position bins per cell/class for the characteristic scheme (1..128) |
| `--transport-implementation` | "auto"; choices: auto, array, numba, cuda | characteristic kernel: auto (follow --implementation), array or numba (dense storage), numba or cuda (csr storage); no fallback |
| `--phase-storage` | "dense"; choices: dense, csr | phase storage of the characteristic scheme: dense (default) or csr (sparse nonempty-bin state, same physics and bins; CSR checkpoints and final arrays) |
| `--phase-max-nnz` | Not supplied (runtime resolution); int | csr only: predeclared capacity of stored bins, refused before allocation when exceeded; never truncates mass |
| `--sediment-courant-max` | 0.5; float | Maximum virtual-sediment-speed × substep duration / cell width; characteristic transport requires a value no greater than 0.5. |
| `--mahleran-root` | Not supplied (runtime resolution); Path | External read-only legacy input tree; overrides the recipe root where supported. |
| `--expected-maple-root` | Not supplied (runtime resolution); Path | Refuse an imported MAPLE that does not resolve to this exact source root. |

### A7. maple-syrup-windowed

Complete conservative resident-window event; selects Numba for numpy, CUDA for cupy. Phase storage is CSR. This CLI has no --implementation, --max-dt-s or --expected-maple-root flag; use its actual options and the expected-root environment variable.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--case-dir` | Required | Verified imported case package directory. |
| `--output-dir` | Required | New output directory. Existing output paths and protected input/source locations are refused. |
| `--store-dir` | Required | Native immutable deep-store destination. Keep referenced versions for checkpoint/output use. |
| `--case-kind` | Required; choices: plot1, rfid, chastre | Choose the matching case importer/verification path. |
| `--backend` | "numpy"; choices: numpy, cupy | Array backend; select the matching implementation where the runner exposes both controls. |
| `--bisection-iterations` | 40; int | Root iteration budget; exact default and case-specific resolution described above. |
| `--root-solver` | "bisection"; choices: bisection, newton | Bisection or safeguarded Newton for the same directed hydraulic equation. |
| `--max-end-s` | Not supplied (runtime resolution); float | Maximum event duration in seconds. Reaching it without physical completion is a refusal, not drying. |
| `--pause-at-s` | Not supplied (runtime resolution); float | Pause at a future original report/forcing boundary before the duration limit; preserve wet checkpoint. |
| `--reuse-initial` | false | Restore persisted native initial window assembly instead of creating it again. |
| `--resume` | Not supplied (runtime resolution) | Existing source/config/backend-bound wet checkpoint bundle, not a completed dry handoff. |
| `--checkpoint-every-s` | 600.0; float | Checkpoint scheduling interval in model seconds; this does not change hydraulic routing. |
| `--phase-max-nnz` | 1000000; int | Explicit capacity for sparse phase bins; exceeding it refuses, rather than truncates mass. |
| `--max-journal-records` | 1000000; int | Maximum pending native deep-journal records, not a license to discard changes. |
| `--window-voxels` | 4; int | Requested number of resident subactive voxel levels; native geometry rules still apply. |
| `--max-host-bytes` | 42949672960; int | Host assembly/checkpoint planning allowance in bytes; not an enforced process RSS cap or whole-storm fit guarantee. |
| `--max-checkpoint-bytes` | Not supplied (runtime resolution); int | resume bound on summed uncompressed bundle bytes; default = --max-host-bytes, may only tighten it |
| `--max-store-bytes` | 274877906944; int | Initial native deep-store logical-byte allowance; not a quota covering every retained later version. |
| `--deep-chunk-encoding` | "full"; choices: full, overlay | CW-B108 deep-chunk storage: full (default) or exact overlay (accepted typed MAPLE source only) |
| `--deep-overlay-max-bytes` | Not supplied (runtime resolution); int | overlay only: explicit per-publication overlay byte cap forwarded to MAPLE (default: MAPLE rule) |
| `--nodata-value` | Not supplied (runtime resolution); float | Explicit DEM nodata sentinel; must agree with the case, never inferred as a physical elevation. |
| `--allow-flat-storage` | Not supplied (runtime resolution) | Default unset: enabled automatically for chastre; optional for rfid; refused for plot1, whose CLI path has no pit storage. These are terminal sinks, not overflow hydraulics. |
| `--terminal-storage-water` | false | Allow event quiet gates to exclude recognized terminal-storage water; retain it in water accounting until explicit reset. Requires pit storage: rfid/chastre only; refused for plot1 in this CLI. |
| `--deposit-mobile-at-flow-end` | false | Explicit modeling option: settle remaining finite mobile load through MAPLE after hydraulic quiet gates hold; not a looser conservation tolerance. |
| `--record-exchange-history` | false | Write cumulative native per-cell/class touch/gross-exchange evidence, not temporal spatial frames. |
| `--allow-maple-source-change` | false | Record deliberate MAPLE drift from import binding. Does not waive checkpoint, storage or within-run source guards. |

### A8. maple-syrup-legacy; default maple-syrup-benchmark

Legacy replay, CPU Plot1. The generic benchmark dispatches here unless characteristic/upwind is explicitly selected. Options labeled unsupported are parsed to give a clear refusal; they do not enable the corresponding feature.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--output / --output-dir` | Required; Path | NEW output directory (legacy_ledger.npz, legacy_summary.json) |
| `--case / --case-dir` | "outputs/plot1" | Verified imported case package directory. |
| `--applied-rainfall` | "outputs/phase7/mahleran_reference_audit/applied_rainfall.csv" | Legacy-applied rainfall capture CSV; external benchmark input, not a bundled default dataset. |
| `--depth-time-level` | "previous"; choices: previous, current | legacy evaluates the laws with the PREVIOUS step's depth d(1) and the new velocity |
| `--end-s` | Not supplied (runtime resolution); float | End of the requested time window in seconds; None resolves from the case/driver. This flag alone does not imply completion. |
| `--implementation` | "numba"; choices: numba, array | SYRUP water-routing implementation. numba (default) requires compiled water AND compiled legacy sediment kernels, with no silent Python fallback. array is a declared slow diagnostic of the water sweep only. |
| `--physics-implementation` | Not supplied (runtime resolution); choices: numba, array | wet physical-law evaluation. Default: the same as --implementation. numba = fused compiled CPU kernel on a prepared frozen-composition context (explicit error if Numba is missing, no silent fallback). array = the NumPy sediment_physics_step reference (declared slower; may be combined with compiled hydrology and legacy transport for controlled comparisons). |
| `--hydrology-implementation` | Not supplied (runtime resolution); choices: prepared, reference | rainfall/infiltration/routing step. Default: prepared with --implementation numba, reference with --implementation array. prepared = fused compiled CPU kernels on a prepared fixed-terrain context (same equations, bisection and checks; requires --implementation numba and host NumPy, explicit error if Numba is missing, no silent fallback). reference = the NumPy/CuPy-reference storm.coupled_step (with the numba or array sweep per --implementation); declared slower, kept for controlled comparisons. |
| `--fortran-ledger` | "outputs/phase7b/mahleran_ledger_run_v3/Output/syrup_sediment_ledger.dat"; Path | External Fortran sediment ledger used by the narrow Plot1 legacy comparison. |
| `--reference-run / --reference-run-dir` | "outputs/phase7/mahleran_deterministic_ksat_run"; Path | External native reference run directory; it must contain the expected comparison artifacts. |
| `--allow-maple-source-change` | false | Record deliberate MAPLE drift from import binding. Does not waive checkpoint, storage or within-run source guards. |
| `--transport-scheme` | "legacy"; choices: legacy | Transport model selection. The narrow legacy replay accepts only legacy; choose an explicitly conservative runner for characteristic/upwind. |
| `--max-dt-s` | Not supplied (runtime resolution); float | only 1 is supported |
| `--report-every-s` | Not supplied (runtime resolution); float | only 1 is supported |
| `--backend` | Not supplied (runtime resolution) | only numpy is supported (no GPU) |
| `--phase-bins` | Not supplied (runtime resolution) | unsupported for legacy: multi-bin characteristic transport (deferred to Phase 7g; use --transport-scheme characteristic) |
| `--transport-implementation` | Not supplied (runtime resolution) | unsupported for legacy: characteristic kernel selection |
| `--sediment-courant-max` | Not supplied (runtime resolution) | unsupported for legacy: sediment Courant control (the legacy replay has none) |
| `--max-transport-substeps` | Not supplied (runtime resolution) | unsupported for legacy: transport substeps (the legacy replay has none) |
| `--min-dt-s` | Not supplied (runtime resolution) | unsupported for legacy: adaptive time stepping (the legacy replay is fixed dt = 1 s) |
| `--max-retries` | Not supplied (runtime resolution) | unsupported for legacy: step retries (the legacy program has no retry) |
| `--max-steps` | Not supplied (runtime resolution) | unsupported for legacy: step limit |
| `--max-report-rows` | Not supplied (runtime resolution) | unsupported for legacy: hydrograph report rows |
| `--expected-ksat-mm-s` | Not supplied (runtime resolution) | unsupported for legacy: conductivity check (the matched benchmark only) |
| `--mahleran-root` | Not supplied (runtime resolution) | unsupported for legacy: MAHLERAN root override (the matched benchmark only) |
| `--expected-maple-root` | Not supplied (runtime resolution) | unsupported for legacy: MAPLE root pin (the matched benchmark only) |

### A9. maple-syrup-benchmark --transport-scheme characteristic|upwind

Explicit conservative fixed-geometry comparison only. The parser tables below describe this explicit branch; default legacy execution uses the preceding table. --applied-rainfall is required here, not for generic --help.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--case-dir` | Required | Verified Phase 2 case, e.g. outputs/plot1 |
| `--output-dir` | Required | NEW directory for summary, grids and hydrographs. |
| `--applied-rainfall` | Required | CSV of the reference run's logged applied rainfall (audit_mahleran.py). |
| `--reference-run` | Not supplied (runtime resolution) | Deterministic-conductivity MAHLERAN run directory to bind (execution.json, Output/). |
| `--max-dt-s` | 1.0; float | Largest step (s): 1, 0.5 or 0.25 for the study. |
| `--end-s` | Not supplied (runtime resolution); float | Window end (s); default the legacy 5400 s. |
| `--implementation` | "numba"; choices: array, numba | Hydraulic sweep: numba (default, CPU compiled; missing Numba is an error) or array. |
| `--backend` | "numpy"; choices: numpy, cupy | Only numpy is accepted; cupy is refused explicitly (no GPU claim). |
| `--report-every-s` | 1.0; float | Hydrograph cadence (s); 1 s for matching. |
| `--min-dt-s` | 0.0009765625; float | Retry floor in seconds; a forcing/report boundary may itself require a shorter slice. |
| `--max-retries` | 10; int | Maximum recoverable step retries before a refusal. |
| `--max-steps` | 10000000; int | Guard on accepted steps; exceeding it is not physical completion. |
| `--max-report-rows` | 100000; int | Guard on the number of report rows planned/stored. |
| `--sediment-courant-max` | 0.5; float | Sediment Courant cap per substep (<= 0.5 for characteristic, <= 1 for upwind). |
| `--max-transport-substeps` | 64; int | Cap on internal sediment substeps per physical step. Exceeding it rejects the attempt; it never truncates a path or deletes mass. |
| `--transport-scheme` | "legacy"; choices: legacy, characteristic, upwind | legacy (DEFAULT): MAHLERAN legacy replay, dispatched to maple_syrup.legacy_experiment. characteristic: Phase 7b conservative phase-bin model (explicit). upwind: Phase 5 operator kept as an explicit comparison. |
| `--phase-bins` | 32; int | Position bins per cell/class for the characteristic scheme (1..128; 32 is a candidate default, not an accepted convergence). |
| `--transport-implementation` | "auto"; choices: auto, array, numba, cuda | Characteristic kernel: auto (follow --implementation), array or numba; no fallback. |
| `--expected-ksat-mm-s` | 0.00025; float | Assert the matched benchmark conductivity (mm/s) on every active cell; this checks the selected case instead of changing its Ksat. |
| `--mahleran-root` | Not supplied (runtime resolution) | Override the recipe's MAHLERAN root (read-only). |
| `--expected-maple-root` | Not supplied (runtime resolution) | Refuse any other MAPLE source root. |
| `--allow-maple-source-change` | false | Run even if MAPLE's source digest differs from the import's (recorded). |

### A10. python -m maple_syrup.experimental_experiment

Water-only explicit/local-inertial experiments; snapshots possible. NumPy backend implementation is named numpy, not array.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--case-dir` | Required | Verified Phase 2 case, e.g. outputs/plot1 |
| `--output-dir` | Required | NEW directory for summary, grids, hydrograph and snapshots. |
| `--solver` | Required; choices: explicit, local_inertial | Choose explicit D4 kinematic wave or two-dimensional local inertia. These are different water equations, not alternative roots of method 5. |
| `--backend` | Not supplied (runtime resolution); choices: numpy, cupy | array namespace: numpy (default) or cupy (needs CuPy + device; never Numba; no fallback) |
| `--implementation` | Not supplied (runtime resolution); choices: numpy, numba, cuda | solver form: numpy = NumPy reference, numba = compiled CPU form (needs numpy backend and Numba), cuda = CUDA form (needs cupy). Omitted: numpy for --backend numpy, cuda for --backend cupy. Invalid pairs are refused before the case is read; there is never a silent fallback |
| `--max-dt-s` | 1.0; float | Largest physical timestep in seconds; forcing/report boundaries and accepted retries can make it shorter. |
| `--end-s` | Not supplied (runtime resolution); float | default: the legacy stormlength (5400 s for Plot 1) |
| `--report-every-s` | 60.0; float | Reporting cadence in model seconds. Forcing knots may add rows; it does not write spatial movie frames. |
| `--cfl-max` | 0.5; float | CFL bound in (0, 0.5] |
| `--limiter` | "off"; choices: off, donor | 'donor' = documented conservative positivity limiter (local inertia only); default off |
| `--snapshot-times-s` | Not supplied (runtime resolution) | comma-separated times for synchronous snapshot maps Times must be in (start, end]; zero is refused. |
| `--min-dt-s` | 0.0009765625; float | Retry floor in seconds; a forcing/report boundary may itself require a shorter slice. |
| `--max-retries` | 10; int | Maximum recoverable step retries before a refusal. |
| `--max-steps` | 10000000; int | Guard on accepted steps; exceeding it is not physical completion. |
| `--mahleran-root` | Not supplied (runtime resolution) | Override the recipe's MAHLERAN root (read-only). |
| `--expected-maple-root` | Not supplied (runtime resolution) | Refuse any other MAPLE source root. |
| `--allow-maple-source-change` | false | Run even if MAPLE's source digest differs from the import's (recorded). |

### A11. python -m maple_syrup.rfid_case

RFID-specific import; the shipped recipe requires an external native-applied forcing capture.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--recipe` | Required | Authored case recipe; referenced external data must be present and match its hashes. |
| `--output-dir` | Required | NEW directory for the case package. |
| `--applied-forcing-capture` | Not supplied (runtime resolution) | syrup_hydro_steps.txt of the executed native run |
| `--mahleran-root` | Not supplied (runtime resolution) | External read-only legacy input tree; overrides the recipe root where supported. |
| `--expected-maple-root` | Not supplied (runtime resolution) | Refuse an imported MAPLE that does not resolve to this exact source root. |
| `--no-compile` | false | Audit/write inputs without native compilation; this is not a completed runnable case. |

### A12. python -m maple_syrup.chastre_case

Tiled Chastre plan/generate/verify. Required combinations depend on the chosen mode, although argparse shows them as optional.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--recipe` | Not supplied (runtime resolution) | cases/chastre/recipe.yaml (plan / generate) |
| `--output-dir` | Not supplied (runtime resolution) | NEW case directory (plan / generate) |
| `--plan-only` | false | print bands, estimates and free disk; write nothing |
| `--verify-case` | Not supplied (runtime resolution) | verify a completed case directory |
| `--tile-rows` | Not supplied (runtime resolution); int | Number of rows per compiled Chastre tile; when absent, use the authored recipe. |
| `--source-case-dir` | Not supplied (runtime resolution) | Override the RFID source case for Chastre generation. Plan-only reads source_rfid.case_dir from the recipe; set that path too. |
| `--expected-maple-root` | Not supplied (runtime resolution) | Refuse an imported MAPLE that does not resolve to this exact source root. |
| `--allow-maple-source-change` | false | Record deliberate MAPLE drift from import binding. Does not waive checkpoint, storage or within-run source guards. |
| `--hash-only-tiles` | false | verify: stream-hash tiles but reload only tile 0 |

### A13. python -m maple_syrup.legacy_driver

Extended CPU legacy replay for Plot1/RFID/Chastre; fixed bed, clipping-source diagnostics.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--case-kind` | Required; choices: plot1, rfid, chastre | Choose the matching case importer/verification path. |
| `--case / --case-dir` | Required; Path | verified imported case directory |
| `--output / --output-dir` | Required; Path | NEW output directory |
| `--end-s` | Not supplied (runtime resolution); float | default: the case storm length; whole seconds |
| `--snapshot-times` | "" | comma-separated model seconds (at most 16); depth and mobile maps |
| `--warmup-s` | 0.0; float | whole model seconds run first on throwaway state in-process |
| `--progress-every-s` | 60.0; float | stderr progress cadence in model seconds; 0 = off |
| `--root-solver` | "bisection"; choices: bisection, newton | Bisection or safeguarded Newton for the same directed hydraulic equation. |
| `--bisection-iterations` | Not supplied (runtime resolution); int | default 40 (Plot 1) / 64 (RFID, Chastre) |
| `--newton-max-iterations` | Not supplied (runtime resolution); int | Safeguarded Newton iteration budget; when not supplied, use the case/driver resolution (the root implementation defaults to 50). |
| `--depth-time-level` | "previous"; choices: previous, current, post_infiltration | wet-law depth: previous (default, accepted), current (new depth), post_infiltration (the original's d(1)) |
| `--source-order` | "index"; choices: index, legacy | Legacy walk source processing order: index uses south-first flat indices; legacy uses north-first rows then west to east. CUDA refuses legacy order. |
| `--legacy-depos-erase` | false | reproduce the order-dependent erasure of deposition at wet no-rain cells (needs --source-order legacy) |
| `--applied-rainfall` | Not supplied (runtime resolution); Path | Plot 1 only: applied-rainfall CSV (as legacy_experiment) |
| `--allow-maple-source-change` | false | Record deliberate MAPLE drift from import binding. Does not waive checkpoint, storage or within-run source guards. |
| `--hash-only-tile-verify` | false | Chastre: reload tile 0 only during verification |
| `--hash-tiles-after-run` | false | Chastre: stream-hash every persisted tile after the run (reads the whole ~33 GB bed state once) |
| `--max-memory-gib` | 32.0; float | refuse if the estimated engine buffers exceed this |
| `--allow-python-kernels` | false | TESTS ONLY: pure-Python kernels (very slow) |

### A14. python -m maple_syrup.legacy_gpu_driver

Extended CuPy legacy replay. Inherited CPU diagnostic flags may be explicitly refused: source-order legacy, legacy-depos-erase and allow-python-kernels are unsupported on CUDA.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--case-kind` | Required; choices: plot1, rfid, chastre | Choose the matching case importer/verification path. |
| `--case / --case-dir` | Required; Path | verified imported case directory |
| `--output / --output-dir` | Required; Path | NEW output directory |
| `--end-s` | Not supplied (runtime resolution); float | default: the case storm length; whole seconds |
| `--snapshot-times` | "" | comma-separated model seconds (at most 16); depth and mobile maps |
| `--warmup-s` | 0.0; float | whole model seconds run first on throwaway state in-process |
| `--progress-every-s` | 60.0; float | stderr progress cadence in model seconds; 0 = off |
| `--root-solver` | "bisection"; choices: bisection, newton | Bisection or safeguarded Newton for the same directed hydraulic equation. |
| `--bisection-iterations` | Not supplied (runtime resolution); int | default 40 (Plot 1) / 64 (RFID, Chastre) |
| `--newton-max-iterations` | Not supplied (runtime resolution); int | Safeguarded Newton iteration budget; when not supplied, use the case/driver resolution (the root implementation defaults to 50). |
| `--depth-time-level` | "previous"; choices: previous, current, post_infiltration | wet-law depth: previous (default, accepted), current (new depth), post_infiltration (the original's d(1)) |
| `--source-order` | "index"; choices: index, legacy | Legacy walk source processing order: index uses south-first flat indices; legacy uses north-first rows then west to east. CUDA refuses legacy order. |
| `--legacy-depos-erase` | false | reproduce the order-dependent erasure of deposition at wet no-rain cells (needs --source-order legacy) |
| `--applied-rainfall` | Not supplied (runtime resolution); Path | Plot 1 only: applied-rainfall CSV (as legacy_experiment) |
| `--allow-maple-source-change` | false | Record deliberate MAPLE drift from import binding. Does not waive checkpoint, storage or within-run source guards. |
| `--hash-only-tile-verify` | false | Chastre: reload tile 0 only during verification |
| `--hash-tiles-after-run` | false | Chastre: stream-hash every persisted tile after the run (reads the whole ~33 GB bed state once) |
| `--max-memory-gib` | 32.0; float | refuse if the estimated engine buffers exceed this |
| `--allow-python-kernels` | false | TESTS ONLY: pure-Python kernels (very slow) |
| `--check-every-steps` | 60; int | read the device flags/ledger rows every N steps (and at the end) |
| `--cuda-mode` | "auto"; choices: auto, fused, split | hydrology launch structure |
| `--cn-mode` | "auto"; choices: auto, level, block | Crank-Nicolson launch structure |
| `--gpu-memory-gib` | Not supplied (runtime resolution); float | refuse if the estimated device memory exceeds this |
| `--water-accounting` | "fused"; choices: separate, fused | per-step water series and cumulative maps: 'fused' = one device launch per step, 'separate' = the B1/B2 reference (3 scalar copies + 4 adds); identical arithmetic, device-only either way (default fused: provisional) |
| `--record-strategy` | "compact"; choices: compact, all | walk record values: 'compact' = only classes with a positive composition fraction in an active cell (static eligibility, exactly the same science), 'all' = every class (the B1 reference) |

### A15. python -m maple_syrup.dependency_bundle

Advanced dependency source/metadata delivery utility; does not run physics. A materialized bundle carries native MAPLE source and truthful metadata, not the environment dependencies.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |

### python -m maple_syrup.dependency_bundle export

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--project` | Required | Exact native MAPLE project selected for export. |
| `--distribution-metadata` | Required | Genuine matching native MAPLE dist-info directory; fabricated or stale metadata is refused. |
| `--origin` | Required | Authored origin/provenance JSON required by the export schema. |
| `--output` | Required | New output path; file for the probe, directory for a run. |

### python -m maple_syrup.dependency_bundle verify

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `<bundle>` | Required | Existing verified native dependency bundle directory. |

### python -m maple_syrup.dependency_bundle materialize

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `<bundle>` | Required | Existing verified native dependency bundle directory. |
| `--cache` | Required | Destination content-addressed materialization cache. |

### python -m maple_syrup.dependency_bundle verify-materialized

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `<target>` | Required | Existing materialized native source/metadata directory. |

### python -m maple_syrup.dependency_bundle env

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `<target>` | Required | Existing materialized native source/metadata directory. |
| `--syrup-src` | Required | Selected SYRUP src directory to include in the printed runtime environment. |

### python -m maple_syrup.dependency_bundle check-runtime

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `<target>` | Required | Existing materialized native source/metadata directory. |

### A16. python -m maple_syrup.initial_reimport

Maintainer-only propose/import of a Chastre generation-zero initial assembly under an exact independently reviewed source transition. A proposal is unreviewed and cannot be imported as-is. This is not wet checkpoint migration, a routine source upgrade, or permission to waive any source guard.

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |

### python -m maple_syrup.initial_reimport propose

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--store-dir` | Required | Native immutable deep-store destination. Keep referenced versions for checkpoint/output use. |
| `--producer-package-dir` | Required | Exact original producer MAPLE package directory; its complete file manifest is checked. |
| `--expected-maple-root` | Not supplied (runtime resolution) | Refuse an imported MAPLE that does not resolve to this exact source root. |
| `--out` | Required | New path for an unreviewed source-transition proposal. The proposal itself is not authorization to import. |

### python -m maple_syrup.initial_reimport import

| Option / alias | Default, type and choices | Meaning / constraints |
| --- | --- | --- |
| `--store-dir` | Required | Native immutable deep-store destination. Keep referenced versions for checkpoint/output use. |
| `--producer-package-dir` | Required | Exact original producer MAPLE package directory; its complete file manifest is checked. |
| `--expected-maple-root` | Not supplied (runtime resolution) | Refuse an imported MAPLE that does not resolve to this exact source root. |
| `--case-dir` | Required | Verified imported case package directory. |
| `--metadata-dir` | Required | New directory for initial-assembly metadata; original operands and deep-store versions are preserved. |
| `--transition` | Required | Exact source-transition JSON with an independent review record and matching producer/consumer manifests. |
| `--max-host-bytes` | Required; int | Host assembly/checkpoint planning allowance in bytes; not an enforced process RSS cap or whole-storm fit guarantee. |
| `--allow-case-maple-source-change` | false | Acknowledge a Chastre case-verification source difference only. Does not relax initial-assembly source/transition checks. |

## Appendix B. References and verification scope

Versioned implementation and requirements: https://github.com/gregokin/SYRUP/tree/d05d8c6c66d9fd2969c007681916254e8d1c2c7c and https://github.com/gregokin/MAPLE/tree/39bcf79e1e67b6ed960f78a923ddfedf1d173e68 . The optional CUDA dependency instructions were cross-checked with the official CuPy 14.2 installation documentation: https://docs.cupy.dev/en/stable/install.html . NVIDIA drivers remain a separate host prerequisite.

Documentation checks captured actual --help for all nine console-command modules and the seven module-only utilities, compared every reference-table flag/default/choice with the pinned parser, checked shell/Python example syntax, and exercised the companion plotting example with small synthetic NPZ files and a real CPU FFmpeg encode. No new storm simulation, GPU computation, environment installation or full fresh-machine installation was performed for the manuals. Historical qualification results remain bound to their recorded source pairs; this document does not extend them.
