# Phase 7f default run workflow

Subsequent hydrology optimization is documented in [Phase7h](../phase7h/performance.md).
Measurements below refer to the wet-law/default-workflow stage before that change.

The default now also compiles the wet physical laws; see
[compiled wet physics](compiled_physics.md) and [measurements](compiled_performance.md).
`--physics-implementation array` retains the original NumPy wet laws while water
and legacy sediment transport remain compiled.

## Selection

| Entry | Model |
|---|---|
| `python -m maple_syrup.benchmark_experiment` / `maple-syrup-benchmark` (default) | legacy replay |
| `python -m maple_syrup.legacy_experiment` / `maple-syrup-legacy` | legacy replay |
| `benchmarks/phase7e/run_legacy_benchmark.py` | thin wrapper of the same module |
| `... --transport-scheme characteristic` or `upwind` | conservative MAPLE-exchange model (explicit) |
| `benchmarks/phase7b/run_characteristic_benchmark.py` | opt-in historic/specialist multi-bin tool, unchanged |

Dispatch happens in `benchmark_experiment.main` before its parser runs (local
import, no circular execution). `run_plot1_matched_benchmark` and the other
conservative APIs are unchanged.

## Legacy replay status

`legacy_experiment.py` is the Phase 7e driver moved into the package; the loop
and ledger arrays are unchanged (module-level parity test against the b350d36
script). Dt = 1 s, frozen terrain/routing, no splash, previous-depth law
evaluation, reference bindings as before. The summary `provenance` key
`script_sha256` is now `module_sha256`.

Scientific limits: fixed initial composition, unlimited supply, explicit
clipping source, ring deposition, no evolving MAPLE bed. It is not a
conservative complete-event, restart or wind-handoff model and is not MAPLE
authority. No wind integration is claimed.

## Refusals

Exit code 2 before any output is created: dt other than 1, report cadence other
than 1, `--backend cupy`, `--phase-bins`, `--transport-implementation`,
Courant/substep, retry/step-limit and conservative-only checks, and, for
`--implementation numba`, missing Numba (water) or non-compiled legacy sediment
kernels. `--implementation array` is a declared slow diagnostic selecting array water and, unless overridden, array wet physics; it
does not require Numba. Compiled wet physics explicitly requires Numba. An output directory that already exists
is refused by the replay.

## Deferred

Multi-bin convergence/performance: Phase 7g (see
`syrup_implementation_phases.md`). Performance diagnosis: `performance.md`.

## Original workflow-switch verification (before wet-law optimization)

Codex independently ran the new default/dispatch checks and existing matched
benchmark tests on the verified Phase7e MAPLE candidate: **38 passed** in 55.57 s,
no skips. Lint and whitespace checks pass. The moved scientific timestep loop
is structurally identical to the b350d36 driver.

A full 5400-step run through the generic CLI with neither `--transport-scheme` nor
`--implementation` specified selected **legacy + Numba water + Numba sediment**.
All 10 saved ledger arrays are bitwise identical to the accepted Phase7e legacy
run. Source hashes stayed stable. See [verification](default_run_verification.json).
Its observed loop time 26.343 s is an additional single-run observation, not an
optimization speedup; no numerical optimization was made.

Claude authored the bounded implementation and reviewed Codex's performance
diagnosis; Codex independently inspected the diff, checked the unchanged loop,
and executed the checks above. Live MAPLE/MAHLERAN trees are unchanged. This
workflow change is uncommitted.
