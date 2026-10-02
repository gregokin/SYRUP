# Exact batched CPU routing benchmark

The production prepared hydrology now uses the level-batched form of the same
40-iteration bisection. Original serial `compiled_sweep()` remains the oracle.
Results, limits and provenance: [performance](../../docs/phase7j/performance.md).

Use the established pinned environment and actual MAPLE dependency:

```bash
source agent_handoffs/tasks/phase6_complete_event/env.sh
source benchmarks/phase7d/candidate_env.sh
"$SYRUP_PYTHON" -m pytest tests/phase7j -q -p no:cacheprovider
"$SYRUP_PYTHON" benchmarks/phase7j/bench_sweep.py --output /tmp/syrup_sweep.json
"$SYRUP_PYTHON" -m maple_syrup.benchmark_experiment --output-dir <new-output> --allow-maple-source-change
```

`bench_sweep.py` times original serial, a benchmark-only dry-skip serial variant,
and the production batched sweep. It checks bitwise parity before timing,
alternates order, excludes JIT, and reports samples, medians/IQR, versions,
CPU flags and packed/FMA instruction counts. It uses committed Phase4 terrain
builders; pytest is needed for those helpers. Synthetic base-water fractions
are not the same as positive RHS fractions because donors can wet other cells.

The measured before/after storm baseline is revision `7e254af`; retain its
`src/maple_syrup` as an immutable package snapshot and prepend that snapshot's
parent to PYTHONPATH for the baseline process. Run three balanced-order fresh
process pairs with the same pinned MAPLE dependency, forcing and CLI options.
The existing `--hydrology-implementation reference` still selects the original
column/routing wrapper; it is a useful reference but is not the old prepared
implementation used in the before/after table.

Full exact-field water reproduction uses the unchanged Phase7i harness and
capture inputs:

```bash
"$SYRUP_PYTHON" benchmarks/phase7i/run_syrup_hydrology.py \
  --capture-run outputs/phase7i/mahleran_capture_run \
  --output <new-heterogeneous-output> --allow-maple-source-change
```

The production code has no new solver/context/CLI selector. Unit tests build the
old prepared serial variant by binding the retained original sweep in isolated
test-only kernel objects, and compare every public field exactly. Local task
logs additionally include the 5400-step strict prepared-serial/batched replay
and warm prepared-call scaling, distinct from the ordinary Phase7i reference.
This CPU optimization does not implement GPU kernels, changing terrain,
conservative evolving-bed sediment, or wind–water handoff.
