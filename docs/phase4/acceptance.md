# Phase 4b routing and compiled-baseline acceptance

2026-09-29. Accepted as a **routing-kernel milestone**, not completion of Phase 4. Claude authored the graph, array/Numba sweeps and original-Fortran harness in bounded tasks `phase4b_routing_kernel` and `phase4b_numba`; Codex independently reviewed and executed them. Baseline commit `ddf13c1`; Phase 4 changes remain uncommitted. The updated implementation plan keeps Numba in Phase 4 and documents alternative solvers as priority follow-up Phase 4R.

## Verification

- Full Phases 1–4 regression: **330 passed, 4 skipped, 34.82 s**. Four skips require CuPy/GPU execution, unavailable here. No GPU equivalence or performance claim.
- Original MAHLERAN sources were compiled unchanged using isolated GNU Fortran 13.3. Both array and Numba routing paths pass executed `route_water` comparisons on chains, branching networks, wetting and Plot 1, with depth tolerance 1e-11 m. This executes the original routine, not the whole MAHLERAN model or a coupled storm.
- Independent scalar root, analytic recession, steady-plane, conservation, failure-without-mutation and backend tests pass. Tested array/Numba outputs are bitwise equal on the parity cases; this is not a promise for every compiler/platform.
- Legacy stale-inflow water creation and insufficient root-bracket cases are tested separately. The compiled path preserves the selected method-5 equations and documented corrections, not the demonstrated legacy defects.
- Ruff across `src/maple_syrup` and `tests`, and `git diff --check`, pass. Codex corrected two overflow-test message expectations (the solver already rejected the state correctly) and mechanical lint issues after Claude's delivery. No hydraulic changes were needed after delivery.

Executed regression command (Numba installed only under `/tmp/syrup-numba`, leaving MAPLE's environment unchanged):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:/tmp/syrup-numba \
MAPLE_SYRUP_GFORTRAN=/tmp/syrup-fortran/root/usr/bin/gfortran-13 \
MAPLE_SYRUP_GFORTRAN_FLAGS='-B/tmp/syrup-fortran/root/usr/libexec/gcc/x86_64-linux-gnu/13/ -B/usr/lib/gcc/x86_64-linux-gnu/13/' \
MAPLE_SYRUP_GFORTRAN_LDFLAGS='-L/tmp/syrup-fortran/root/usr/lib/gcc/x86_64-linux-gnu/13/' \
/home/okin/MAPLE/.venv/bin/python -m pytest tests/phase1 tests/phase2 tests/phase3 tests/phase4 -p no:cacheprovider -q
```

Numba 0.67.0, llvmlite 0.49.0, NumPy 2.5.2, Python 3.12.3. Original-source hashes and compile commands are enforced/recorded by `tests/phase4/fortran_reference.py`. Prompts, manifests, raw logs and reports are archived locally under `agent_handoffs/tasks/phase4b_numba/`; implementation and tests are reviewable in the source tree.

## Initial CPU cost

Synthetic draining planes, fixed 1 mm initial water depth, 1 s step, 0.5 m spacing, friction 21.45, 40 bisections. Each process runs one first call then three timed repeated calls; table uses their median. These are complete validated **routing steps**, excluding rainfall/infiltration and graph setup, not storm benchmarks or actual Plot 1 terrain. Runs were sequential, not concurrent with the regression suite.

| Grid | Array ms/step | Numba ms/step | Ratio | Numba extra traced MB |
|---|---:|---:|---:|---:|
| 60 × 20 | 11.477 | 0.952 | 12.06× | 0.218 |
| 128 × 128 | 28.346 | 8.001 | 3.54× | 2.906 |
| 512 × 512 | 180.029 | 135.333 | 1.33× | 46.406 |

The first Numba routing call took 0.731 s including lazy import/compilation; later sizes reused that compilation. On-disk caching was off. Extra traced memory excludes the existing graph/state and native compiler allocations; it is **not peak process RSS**. Memory remained similar for both implementations. The shrinking speed advantage on larger grids supports measuring total event cost rather than promising a universal multiplier. All three cases had absolute budget residual below 4e-17 m³, equal between implementations.

Raw measurements and environment: [kernel_cost.json](kernel_cost.json). Reproduce from the repository root with `PYTHONPATH=src:/tmp/syrup-numba /home/okin/MAPLE/.venv/bin/python benchmarks/phase4/kernel_cost.py --implementation array`, then the same command with `--implementation numba`. Point PYTHONPATH at the actual isolated Numba installation if different. Install the project's optional `numba` extra in an appropriately isolated environment for a persistent setup.

## Subsequent acceptance

The coupled storm, refinement, whole-event cost and controlled original-Fortran comparisons are now complete: see [Phase 4 storm acceptance](storm_acceptance.md). This document retains the earlier kernel milestone measurements. GPU execution, sediment exchange, dry reset and restart remain outside acceptance. Phase 4R is the important solver/GPU follow-up.
