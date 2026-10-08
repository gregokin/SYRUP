# MAPLE-SYRUP

**Sediment Yield, Runoff, and Uptake by Plants**: a water erosion and transport extension for MAPLE (`/home/okin/MAPLE`). It is not a standalone model. MAPLE owns wind physics and the shared sediment bed (voxel column, active layer, grain classes, availability, ledger, topographic commits, snapshots, backend). MAPLE-SYRUP adds MAHLERAN-like water processes on top of that state and invokes MAPLE for everything else.

Project scope and rules: `AGENTS.md`, `claude.md`, `syrup_implementation_phases.md`.

The CPU legacy replay also prepares fixed hydrology data and compiles infiltration,
runoff branches, and routing setup/checks around the existing ordered sweep.
`--hydrology-implementation reference` selects the original hydrology for comparison.
See [CPU hydrology qualification](docs/phase7h/performance.md). Water-only resident
CUDA storms are now available through an explicit backend choice; see
[GPU hydrology results](docs/phase4s/performance.md). An explicit CUDA safeguarded Newton root solver for the same
equation is available with independently verified complete-storm [GPU Newton results](docs/gpu_newton/results.md). GPU sediment
remains unqualified.

## Default run workflow (Phase 7f)

Runs default to the Python/Numba replay of the MAHLERAN **legacy** sediment
transport on frozen Plot1 terrain (dt = 1 s, no splash). It has fixed
composition, unlimited supply, explicit clipping-source accounting and no
evolving MAPLE bed; it is not a conservative complete-event, restart or
wind-handoff model. With the existing verified dependency and Numba environment:

```bash
cd /home/okin/SYRUP
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/syrup-numba
source benchmarks/phase7d/candidate_env.sh
/home/okin/MAPLE/.venv/bin/python -m maple_syrup.benchmark_experiment \
  --case-dir outputs/plot1 --output-dir NEW_OUTPUT_DIRECTORY --allow-maple-source-change
# After installing this package, maple-syrup-legacy invokes the same legacy module.
```

Unsupported controls (dt other than 1, GPU, bins, Courant/substep options) are
refused, and the numba path requires compiled water and sediment kernels. The
multi-bin conservative model remains available with
`--transport-scheme characteristic` (needs `--applied-rainfall`); its
convergence/performance work is deferred to Phase 7g. Details:
[default workflow](docs/phase7f/default_workflow.md),
[performance diagnosis](docs/phase7f/performance.md).

## Experimental water hydraulics

Two separately selectable CPU/CUDA candidates reuse the existing MAPLE case,
rainfall and MAHLERAN-inspired infiltration: explicit D4 kinematic routing and
uniform-grid local inertia. Use `python -m maple_syrup.experimental_experiment`
with `--solver explicit` or `--solver local_inertial`, `--backend numpy` or
`--backend cupy`, and the verified case and a new output directory.
See [usage, equations and qualification limits](docs/hydraulic_candidates/README.md)
and [measured comparisons](docs/hydraulic_candidates/results.md).
These are water-only experiments; local-inertial velocity and full-storm backend
sensitivity require further work before erosion coupling. Existing defaults remain unchanged.

## Current water workflow

The commands in this and the following historical sections that name
`run_characteristic_benchmark.py` select the explicit multi-bin model.

Storm completion/restart, matched MAHLERAN benchmarking, characteristic sediment
transport and CPU optimizations are implemented. The benchmark retains fixed
terrain/routing and excludes splash. Water/class budgets pass; remaining legacy
sediment timing/composition and raw-peak limitations are documented in
[Phase 7b acceptance](docs/phase7b/acceptance.md). Full coupled GPU storms and
wind–water event integration remain open; individual kernels have actual-GPU
verification.

The selected optimized MAPLE dependency is the isolated package `72310c49…`,
built from accepted `d3d007024…` plus the small shared allocation patch. Live
MAPLE is unchanged. See [Phase 7d setup and results](docs/phase7d/optimization.md)
for reconstruction, exact hashes and validation. After preparing that snapshot:

```bash
cd /home/okin/SYRUP
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
export PYTHONPATH=/tmp/syrup-numba
source benchmarks/phase7d/candidate_env.sh
/home/okin/MAPLE/.venv/bin/python benchmarks/phase7b/run_characteristic_benchmark.py \
  --bins 32 --dt 1 --allow-maple-source-change --output NEW_OUTPUT_DIRECTORY
```

This uses the existing imported Plot1 fixture and records the dependency change
explicitly. New case imports should bind this selected MAPLE snapshot directly.
The runtime helper verifies source hashes before exposing the candidate modules.

## Phase 7e: selective voxels and transport comparisons

The new opt-in shared MAPLE candidate avoids most full-column kernel work and
reuses surface metadata across trusted exchanges. A separate compiled legacy
replay closely matches actual MAHLERAN, while the 1–128 bin study distinguishes
final-yield convergence from peak-flux convergence. See
[qualification](docs/phase7e/acceptance.md) and
[setup/reproduction](benchmarks/phase7e/README.md). The older selected dependency
above remains available; selecting Phase7e is explicit. No wind physics is copied.

## Earlier milestones: Phases 1–5


Implemented:

- `src/maple_syrup/dependency.py`: resolves the installed MAPLE and refuses a mismatched or unidentifiable one;
- `src/maple_syrup/provenance.py`: bounded source digest and read-only git state;
- `src/maple_syrup/probe.py`: builds a tiny real MAPLE state and checks it;
- `docs/phase1/interface_contract.md`: dependency, state ownership, gaps and selected integration approach;
- `src/maple_syrup/case_import.py` (Phase 2): audits MAHLERAN Plot 1 and compiles it into an actual MAPLE case with MAPLE's own `compile_case`/`load_compiled_case`; recipe `cases/plot1/recipe.yaml`; see `docs/phase2/plot1_import.md`.

Phase 3 adds exact rainfall integration and conservative, independent soil-water columns, using actual MAPLE water state and backend helpers. Phase 4 adds coupled spatial routing and storm recession. Phase 5 adds wet sediment physics and actual MAPLE bed exchange; its verification record is in [docs/phase5/acceptance.md](docs/phase5/acceptance.md). See `docs/phase3/infiltration.md` for the equations and deliberate legacy departures. The Plot 1 case is a documented **corrected** fixture (six distinct grain maps instead of the root XML's repeated map), not a reproduction of the root XML. `port_feasibility/` is an older, separate feasibility experiment.

## Dependency

Phase 5 runs use a fixed, read-only installation of actual MAPLE (distribution `maple` 0.0.1), with MAPLE’s existing interpreter and dependencies. See [dependency reconstruction and environment](docs/phase5/dependency.md). The complete package hash matches the imported case; new live upstream changes are adopted only after compatibility checks. It is intentionally not listed in `pyproject.toml`, because the PyPI name `maple` belongs to an unrelated project. Its identity, source root and required API surface are checked at runtime; see interface contract section 1.

## Historical Phase 1–5 development commands

Use MAPLE's interpreter with the prepared fixed dependency below. If it is
missing, follow [reconstruction instructions](docs/phase5/dependency.md).
Set this environment once before the remaining commands:

```bash
cd /home/okin/SYRUP
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
export GIT_CEILING_DIRECTORIES="$PWD/outputs/dependencies/maple_d3d007024/source"
export PYTHONPATH=src:outputs/dependencies/maple_d3d007024/editable:outputs/dependencies/maple_d3d007024/source/src:/tmp/syrup-numba
```

Probe and dependency tests:

```
cd /home/okin/SYRUP
/home/okin/MAPLE/.venv/bin/python -m maple_syrup.probe
/home/okin/MAPLE/.venv/bin/python -m pytest -p no:cacheprovider tests/phase1 -q
```

`PYTHONDONTWRITEBYTECODE=1` keeps the MAPLE import from writing `__pycache__` into the MAPLE tree; `-p no:cacheprovider` avoids a `.pytest_cache` directory.

Phase 2 (Plot 1 import; the output directory must not exist yet):

```
/home/okin/MAPLE/.venv/bin/python \
    -m maple_syrup.case_import --recipe cases/plot1/recipe.yaml --output-dir NEW_DIR \
    --expected-maple-root /home/okin/SYRUP/outputs/dependencies/maple_d3d007024/source
/home/okin/MAPLE/.venv/bin/python -m pytest -p no:cacheprovider tests/phase2 -q
```

Exit status: 0 success (`NEW_DIR/syrup/plot1_binding.json` written), 1 failed audit/check or MAPLE rejection (`NEW_DIR/syrup/FAILED.json` when the directory was created), 2 dependency error. `--no-compile` writes the audited case package without calling MAPLE's compiler.

Probe options:

- `--expected-maple-root /home/okin/SYRUP/outputs/dependencies/maple_d3d007024/source` (or `MAPLE_SYRUP_EXPECTED_MAPLE_ROOT`): refuse any other MAPLE source.
- `--output FILE`: also write the JSON report; refuses an existing file or a path inside the MAPLE tree.
- `--include-file-manifest`: add per-file SHA-256 entries.
- `--backend cupy`: fails with `BackendUnavailableError` when CuPy or a CUDA device is unavailable (true for the current venv); never falls back to CPU.

Exit status: 0 success, 1 failed check or MAPLE source changed during the probe, 2 dependency/backend/usage error.

## Phase 3 column experiment

After importing Plot 1, run rainfall and infiltration through the rainfall window:

```
/home/okin/MAPLE/.venv/bin/python \
    -m maple_syrup.column_experiment --case-dir outputs/plot1 --max-dt-s 1 \
    --output-dir outputs/plot1_columns
/home/okin/MAPLE/.venv/bin/python -m pytest -q -p no:cacheprovider
```

The output directory must be new. The runner verifies the imported case and sources before calculation. It writes a water budget and final grids; it keeps ponded water at rainfall end. This is a **no-routing diagnostic**, not a complete storm: no runoff hydrograph, erosion, wind event, evapotranspiration or dry reset. Conductivity uses the XML's deterministic positive mean instead of its potentially negative normal draw. CuPy-compatible kernels are present; GPU execution remains unverified in the current environment.

Phase 3 validation: 222 tests passed, 2 GPU tests skipped; see `docs/phase3/acceptance.md`. Phase 2 baseline is `951f0db`; Phase 3 implementation and acceptance evidence are tracked together.

## Phase 4 water-only storm

MAHLERAN method-5 routing now couples to rainfall/infiltration on the actual MAPLE case, with an optional Numba CPU sweep. Full storms, timestep refinement and original-Fortran routine comparisons are verified: 366 tests passed, 5 GPU-dependent tests skipped. See [acceptance, timings and reproduction commands](docs/phase4/storm_acceptance.md) and [coupling contract](docs/phase4/storm.md). Those checks preceded GPU qualification. Resident GPU water hydrology is now verified in [Phase 4S](docs/phase4s/performance.md); alternate hydraulic solvers have separate [experimental results and limitations](docs/hydraulic_candidates/results.md). Phase 5 adds wet sediment transport; see its separate acceptance record.

## Phase 5 wet sediment

MAHLERAN-like wet detachment, class-specific travel distance and velocity,
conservative mobile transport, deposition and export now use actual MAPLE
pickup, availability, voxels, active layer, terrain commits and rerouting.
Wind physics remains in MAPLE. After setting the fixed dependency environment:

```bash
/home/okin/MAPLE/.venv/bin/python -m maple_syrup.sediment_experiment \
    --case-dir outputs/plot1 --output-dir NEW_SEDIMENT_RUN \
    --max-dt-s 1 --implementation numba --report-every-s 60
```

Verification: **446 passed, 6 GPU skips**, selected equations checked against
compiled original MAHLERAN, and three complete configured Plot 1 windows at
1, 0.5 and 0.25 seconds. Total sediment export differs by 0.245% between the
coarsest and finest timesteps; water and sediment budgets close.
See [acceptance, performance and limitations](docs/phase5/acceptance.md) and
[comparison figure](docs/phase5/storm_comparison.svg).

This is a CPU milestone. GPU execution, full original-MAHLERAN sediment-storm
qualification, event-stop/dry-reset/restart and wind handoff remain pending.
The saved MAPLE snapshot and diagnostic grids are not a storm restart format.

## GPU legacy benchmarks and large terrain

GPU Newton hydrology and coupled GPU legacy sediment replay now have Plot 1 and large Chastre comparisons against CPU Numba,
with a completed original-routine Fortran reference for Chastre. See [current results, timings and qualifications](docs/legacy_gpu/current_status.md).
This fixed-composition benchmark is separate from the conservative evolving MAPLE-bed model; candidate optimizations and the
Fortran recession-water discrepancy remain follow-ups. Use MAPLE's CPU and CuPy environments; no separate SYRUP environment is needed.
