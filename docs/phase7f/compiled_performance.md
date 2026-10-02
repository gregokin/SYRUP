# Compiled legacy wet-law optimization: measured result

Subsequent hydrology optimization is documented in [Phase7h](../phase7h/performance.md).
Measurements below refer to the wet-law/default-workflow stage before that change.

The default legacy replay now uses one serial CPU Numba wet-law kernel and prepares
fixed grain/composition properties once. The original NumPy/CuPy wet-law implementation
remains available; `--physics-implementation array` selects it in the legacy CPU driver.
Hydrology and legacy transport algorithms are unchanged. Multi-bin optimization remains
Phase 7g.

## Matched timings

Frozen-topography, no-splash Plot1: 1200 cells, six classes, 5400 one-second steps.
Three alternating fresh-process baseline/candidate pairs on this machine. Baseline is
an immutable pre-optimization package snapshot, including the qualified legacy default
switch; both use actual MAPLE package digest `72310c49`. No task tests ran during timing;
other user workloads were not stopped. Medians below; raw results, source hashes and
all three physical comparisons are in [measurements](compiled_measurements.json).

| Measurement | Before | Compiled wet laws | Change |
|---|---:|---:|---:|
| True external process, including imports, preparation, JIT and output | 29.46 s | 23.14 s | 21.5% less time; 1.27× speed |
| Storm loop, including first-call JIT | 25.94 s | 19.57 s | 24.6% less time; 1.33× speed |
| Loop minus timed first-step components | 24.72 s | 16.07 s | 35.0% less time; 1.54× speed |
| Wet-law stage excluding first call | 12.93 s | 4.14 s | 68.0% less time; 3.13× speed |
| Hydrology excluding first call | 7.95 s | 8.03 s | unchanged code; timing variation |
| Legacy sediment stage excluding first call | 2.37 s | 2.41 s | unchanged code; timing variation |
| Peak whole-process RSS | 335.1 MiB | 366.5 MiB | 31.5 MiB higher |

External-process ranges were 28.95–29.55 s before and 21.49–23.70 s after.
The first compiled wet-law call takes about 2.32 s (includes compilation and execution).
Static preparation takes about 0.00081 s and retains 212872 bytes (0.203 MiB).
Peak RSS includes imports and JIT compiler memory; it is not a kernel-memory measurement.
Numba disk caching is not enabled for this closure-based kernel. Compilation therefore
recurs in a fresh process; an already-compiled process avoids it. The loop-minus-first
figure subtracts timed component calls, not the entire first driver iteration, and is
not a separately measured warm whole-process run.

The driver field `whole_process_wall_s` starts after module imports. The table instead
uses the external subprocess harness clock so imports and shutdown are included.
Independent component medians need not add to the median loop time.

## Physical agreement and verification

All three full-storm comparisons passed the previously declared output-comparison
bounds (`rtol=2e-11`, `atol=1e-14` for sediment arrays). These are implementation
comparison bounds, not changed scientific conservation tolerances.

- Water/time arrays match bit-for-bit; regime counts and peak time (1261 s) are exact.
- Export remains 0.00921633936926069 kg; relative change is 4.44e-16.
- Largest saved sediment-array difference is 1.683e-16 kg.
- Maximum legacy accounting residual remains 1.651e-15 kg. Clipping-source accounting
  is unchanged; this is not proof of conservative evolving-bed behavior.
- Scoped regression: **138 passed, 1 skipped**, including differential physics,
  invalid/overflow refusal, static-input mutation, independent output ownership,
  selector/default behavior and baseline CLI comparison. The skip needs a usable GPU.
- **Three direct original-MAHLERAN Fortran equation checks passed** using the compiled
  evaluator. Existing 20 ppm tolerance reflects original REAL32 literals; this does
  not claim bitwise Fortran identity.
- Ruff and whitespace checks passed. Claude authored the implementation and bounded
  correction; Codex independently reviewed the equations, callers, tests and runs.

No full GPU storm or new GPU acceleration is claimed. The prepared context is only
valid for frozen composition. Conservative evolving MAPLE-bed calls retain their
original implementation and must never reuse stale frozen holdings.

## Remaining work

Hydrology is now the largest timed component. Next candidates are its validation,
infiltration, state preparation and interfaces around the compiled routing sweep,
followed by legacy transport allocations and driver overhead. Keep dynamic validation
and accounting; do not remove checks to gain speed. Reducing JIT startup or adding a
proper cache is another separate opportunity.

Historical MAHLERAN whole-process timings are 9.30–9.75 s (different instrumentation
builds). SYRUP's new 23.14 s cold-process median remains roughly 2.4–2.5× those times.
MAHLERAN was not rerun in these pairs, so this is historical context, not a new matched
Fortran-versus-Numba speed test. The wet-law optimization improves the measured cost
without claiming that it closes the entire performance gap.

## Reproduction

```
source agent_handoffs/tasks/phase6_complete_event/env.sh
source benchmarks/phase7d/candidate_env.sh
"$SYRUP_PYTHON" -m maple_syrup.benchmark_experiment --output-dir <new-output> --allow-maple-source-change
# Array-law control on the same compiled water/transport:
"$SYRUP_PYTHON" -m maple_syrup.benchmark_experiment --output-dir <other-new-output> --allow-maple-source-change --physics-implementation array
```

Exact harness commands/logs and immutable baseline manifest are local under
`agent_handoffs/tasks/phase7f_compiled_physics`; outputs are under
`outputs/phase7f_optimization`. Neither live upstream tree was edited.
