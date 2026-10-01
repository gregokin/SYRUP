# Legacy replay performance diagnosis

This diagnosis records the pre-optimization baseline. The subsequent compiled
wet-law implementation and measurements are in [compiled performance](compiled_performance.md).

The baseline legacy replay is partly compiled. Numba compiles the ordered water
sweep, source-based deposition walk and Crank–Nicolson sediment pool. The wet
physical laws, infiltration, validation, array preparation and driver remain
Python/NumPy. Therefore 28 seconds versus MAHLERAN's 9.3–9.75 seconds is not a
controlled comparison of the same complete compiled implementation in two languages.

The unprofiled Phase 7e run took 28.099 s in its loop (30.317 s whole process):

| Component | Seconds | Share of loop |
|---|---:|---:|
| Wet detachment / transport-property calculation | 14.091 | 50.1% |
| Hydrology, including infiltration and routing interface | 9.099 | 32.4% |
| Legacy sediment interface + walk + CN | 3.296 | 11.7% |
| Driver and accounting outside these timers | 1.612 | 5.7% |

We do not yet have a matched per-routine Fortran profile for this replay. The
Python profile identifies its own costs, but does not uniquely apportion the
roughly threefold cross-program difference; solver details and diagnostics also
differ.

Fortran whole-process references are recorded in
`outputs/phase7/mahleran_deterministic_ksat_run/execution.json` (9.297 s) and
`outputs/phase7b/mahleran_ledger_run_v3/execution.json` (9.749 s with the additional
sediment ledger). These are separate cold process executions of checked builds,
including output; they are not matched steady-state kernel timings. Plot1 is
60 by 20 active cells at 0.5 m (1200 cells), as recorded in
[the water-storm case description](../phase4/storm_acceptance.md).

First-step costs include compilation: 0.563 s hydrology, 0.694 s sediment and 0.003 s
physical laws. Compilation is not the dominant explanation for the gap.

## New diagnostic evidence

A full 5400-step cProfile replay used an immutable b350d36 source snapshot, the same
actual MAPLE `72310c49` dependency, and the recorded forcing/reference. All 10 saved ledger
arrays match the unprofiled replay exactly. No numerical implementation changed.
See [curated profile](legacy_profile.json); raw captures are local under
`outputs/phase7f` and `agent_handoffs/tasks/phase7f_legacy_default`.

The instrumented process made 24.19 million function calls and took 38.07 s. These
profiled times include tracing overhead and must not replace the unprofiled timing
above. Cumulative rows overlap; they must not be summed as independent costs.

- `sediment_physics_step`: 5400 calls, 15.54 s cumulative. Source inspection shows
  repeated NumPy expressions allocating intermediate arrays and computing multiple
  regimes before selection with `where`.
- `median_diameter_m`: 5400 calls, 2.00 s cumulative inside those physical laws. Legacy
  composition is fixed, so recomputing fractions and median diameter each step is
  unnecessary for this mode. An evolving MAPLE bed requires invalidation after exchange.
- `finite_flag`: 313223 calls, 2.57 s cumulative; `negative_flag`: 172813 calls, 1.31 s.
  These are totals across the process, not additional times to add to physics/hydrology.
- Routing's Python interface `_route` accounts for 8.17 s cumulative, whereas its
  `run_sweep` call accounts for 3.89 s including first compilation. State preparation,
  shape/namespace checks, diagnostics and array work matter alongside the native solve.
- Legacy's interface allocates several arrays per step, recreates walk limits from
  fixed spacing and checks input arrays before invoking its compiled kernels.

## Ranked optimization opportunities

1. **Compile and combine the wet physical-law calculations.** Keep the public validated
   NumPy/CuPy interface; add a CPU Numba loop with the same formulas, regime selection,
   units, caps, velocity memory and explicit error checks. Avoid fastmath. Reuse output
   workspaces under explicit ownership. This targets the largest measured component.
2. **Precompute invariant legacy properties.** Fractions, median grain diameter,
   grain-only powers/constants, vegetation-dependent constants, routing level bounds
   and walk-limit lookup tables are fixed for this frozen legacy run. Validate and bind
   them to the case/context once. Do not reuse stale values in evolving-bed mode.
3. **Reduce repeated validation and allocation costs without removing checks.** Validate
   immutable shapes/parameters once; combine dynamic finiteness/sign checks with compiled
   calculations. Keep per-step balance/error detection, public-boundary validation and
   failure atomicity. Reuse internal buffers or alternate owned buffers rather than
   allocating all intermediates at every step. Account for arrays retained in outputs.
4. **Compile more of the hydrology and timestep driver.** Retain the accepted hydraulic
   equations, rainfall timing and infiltration. Reduce Python/NumPy hand-offs around
   the existing compiled sweep before considering a different root solver or GPU method.

No achieved speedup is claimed for these opportunities. Even making the 14.1 s
physical-law stage free would leave about 14 s, so several components need attention
to approach the recorded Fortran runtime. CPU profiling does not establish a GPU
speedup; this 1200-cell case is particularly sensitive to launch/dispatch overhead.
A follow-up should separate cold and warm timings and compare identical output and
validation workloads against optimized and checked Fortran builds.

Current user direction makes the Python/Numba legacy replay the default experimental
run workflow. It retains explicit clipping-source accounting, fixed composition and
unlimited supply. Conservative multi-bin code is retained for a deferred phase;
its voxel/copy optimization is not part of the 28-second legacy replay cost.
