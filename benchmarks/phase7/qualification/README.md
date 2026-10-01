# Executed Phase 7 evidence

Scientific interpretation, performance qualifications and open requirements are
in [the report](../../../docs/phase7/acceptance.md). This directory is curated
from completed local runs, not proposed targets or a fabricated reference.

- `runs.json`: budgets, class exchanges, timing, frozen checks and source
  provenance for dt1, dt0.5 and dt0.25.
- `comparison_*.json`: corrected matched MAHLERAN comparisons. Legacy
  peak-outlet snapshots are explicitly distinguished from per-cell maxima.
- `refinement.json`: same-input timestep sensitivity; finest dt is not truth.
- `array_parity_and_spatial.json`: 97-array exact CPU parity, plus separately
  observed synchronous peak-outlet depth/velocity comparison.
- `distance_diagnostics.json`: requested-pickup-weighted local mean distances.
- `distance_resolution.json`: actual operator impulse and independent analytic
  cell-exit comparison. Not a full MAHLERAN sediment execution.
- `cpu_scaling.json`: actual MAPLE coupled synthetic windows, three voxels,
  six classes; excludes full-storm spatial convergence.
- `gpu_valley_after_fix.json`: synchronized wet-law/transport kernel parity and
  timing. CuPy pool reservation is not whole-device peak memory; full coupled
  GPU execution and transfer cost are unqualified.
- `array_profile.txt`: instrumented array-run cProfile; cumulative entries
  overlap, and its times are not a controlled Numba comparison.
- `measurement_sources.json`: exact experiment-specific harness source bytes
  and SHA256, including the isolated GPU environment. Main runner/comparator
  code lives one directory above / in `src/maple_syrup`; use the pinned MAPLE
  environment described in the report. Harnesses assume repository-root cwd
  and local input/reference paths; reproduce into new output locations, not
  by overwriting accepted evidence.

Original artifacts remain under `outputs/phase7/` and raw logs/prompts/reviews
under `agent_handoffs/tasks/phase7_matched_benchmark/` (both gitignored). The
initial comparison's mismatched maximum-depth/velocity scores were withdrawn;
only corrected scores are curated here. The reports do not accept the
approximately 24-fold sediment-export discrepancy as scientific agreement.
