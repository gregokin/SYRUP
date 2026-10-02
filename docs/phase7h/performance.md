# CPU hydrology optimization and GPU preparation

The default compiled legacy replay now prepares fixed routing/column data once and
uses fused CPU Numba calculations for infiltration, runoff branches, routing setup
and output checks. It calls the existing ordered routing sweep directly. Equations,
root method, donor order and scientific budget tolerances are unchanged. Select
`--hydrology-implementation reference` to retain the original hydrology path.

## Repeated matched Plot1 measurements

Frozen terrain/routing, no splash, six classes, 1200 cells, 5400 one-second steps.
Three alternating fresh-process baseline/candidate pairs on this machine, with
no concurrent task tests/profiles. Other user workloads were not stopped.
Baseline is committed `f40ee38550b3dddca88c949d26eab0735e25b68b` in an immutable
package snapshot; both use actual MAPLE package digest `72310c49…`. Candidate source
hashes remained unchanged. See [raw measurements and comparisons](measurements.json).

| Measurement, median of three | Before | Prepared hydrology | Change |
|---|---:|---:|---:|
| True external process, imports/preparation/JIT/output included | 19.63 s | 18.49 s | 5.8% less time |
| Storm loop, first-call JIT included | 16.47 s | 15.31 s | 7.0% less time |
| Loop minus timed first-step components | 13.40 s | 10.31 s | 23.0% less time |
| Hydrology excluding first call | 6.68 s | 3.79 s | 43.3% less time; 1.76× speed |
| Hydrology first call, including JIT | 0.501 s | 2.334 s | more compiled work at startup |
| Peak whole-process RSS | 366.2 MiB | 376.6 MiB | 10.4 MiB higher |

External-process ranges: 19.58–19.75 s before, 18.27–18.49 s after. Context
preparation takes about 0.965 ms and retains 132536 bytes (0.126 MiB).
Cold compilation recurs in a fresh process (closure kernels, disk cache disabled).
An already-compiled process avoids that startup work. The loop-minus-first figure
subtracts the timed component calls, not the entire first driver iteration; it is
not a separately timed warm whole process. Independent component medians need not
add to the loop median. RSS includes imports/JIT and is not kernel peak memory.
The existing driver's `whole_process_wall_s` excludes imports, so this table uses
an external subprocess clock instead.

These are new same-machine paired measurements, not a comparison of the current
machine load with the earlier Phase7f 23.14-second observation. MAHLERAN was not
rerun during this task; no new Fortran whole-process speed comparison is claimed.

## Physical comparison and checks

All three full-storm comparisons passed the predeclared water comparison bound
(`rtol=2e-12`, `atol=1e-14`) and previous sediment comparison bound
(`rtol=2e-11`, `atol=1e-14`). These compare implementations; original scientific
conservation tolerances (`LOCAL_BALANCE_RTOL=16eps`, routing `BALANCE_RTOL=32eps`,
MAPLE budgets) are unchanged.

- Largest saved water difference: 6.783e-18 m3/s in outlet discharge, 3.395e-18 m3
  in per-step export. Water arrays agree within the bound; they are not bitwise.
- Largest saved sediment difference: 1.388e-16 kg. Export remains about
  0.00921633936926069 kg (relative change −3.33e-16).
- Regime counts, timestamps and sediment peak time (1261 s) match exactly.
- Maximum legacy accounting residual: 1.614e-15 kg. Clipping source, unlimited
  supply and frozen composition remain explicit legacy limitations; this is not
  an evolving MAPLE-bed conservation claim.
- **351 scoped tests passed**, six device-dependent skips. Coverage includes both
  infiltration formulations, wet/dry/recession and branch ties, saturation/drainage,
  all returned fields, masks, refusal classes/precedence, overflow, input/static
  ownership and prior-result preservation, selectors and actual Plot1 CLI.
- **Six direct original-Fortran routing tests passed** through a hook invoking the
  new compiled routing phase: chains at 1/.5 s, branching networks, wetting and
  Plot1. Eight actual Fortran calls are covered because each branched case runs two
  geometries. Original 1e-11 m depth tolerance applies; no bitwise Fortran claim.
- Ruff and whitespace checks passed. Original reference modules, sediment/wet laws,
  shared MAPLE bed code, and live upstream trees remain unchanged.

Claude authored the bounded implementation and fixture corrections; Codex
independently reviewed code and ran checks/benchmarks. Codex restored two accidentally
removed test newlines and moved runtime provenance capture after execution so
`compiled_in_process` reflects the completed run.

## Larger-grid calls

[Warm scaling results](scaling.json) time repeated calls on the same synthetic
wet state, alternating reference/prepared order in one process after warmup.
They exclude graph/context creation and JIT (50, 15 and 8 repetitions respectively).
Every compared public field is bitwise identical in these cases. The first synthetic
context preparation measured 0.128 s versus 6/35 ms on larger grids; this is a cold
setup observation, not the already-imported Plot1 preparation measurement. These are hydrology calls, not complete storm scaling.

| Cells | Shape | Reference call | Prepared call | Speed |
|---:|---|---:|---:|---:|
| 1260 | 60 × 21 | 1.241 ms | 0.745 ms | 1.67× |
| 16512 | 128 × 129 | 9.280 ms | 8.548 ms | 1.09× |
| 65792 | 256 × 257 | 40.328 ms | 35.486 ms | 1.14× |

The smaller gains on larger grids are consistent with more time in the unchanged
per-cell bisection solve, but these calls were not profiled by component. Sample
ranges overlap; the 1.09× versus 1.14× ordering is not a scaling trend. Further solver/GPU investigation
belongs to Phase4R; it must preserve or qualify hydraulic/erosion fidelity separately.

## GPU preparation and limits

The new context owns validated contiguous static geometry, donors, levels and
column parameters; dynamic depth, soil, rain and discharge are separate flat
arrays. Calculation stages expose cellwise column/runoff work, routing preparation,
ordered levels, output checks, and reductions. Future GPU work can use the same
physics/data contract, execute cells within each level in parallel, and keep state
resident while reading small flags/scalars for acceptance.

No GPU kernels or new GPU performance qualification were added. CPU column packing
uses AoS `(n_cells,6)`; a device backend can pack SoA `(6,n_cells)` once for coalesced
access. Device reductions, level barriers, immutable/static ownership, error priority,
resident coupled events and CPU/GPU comparison still need implementation. Plot1's
prepared graph has 66 levels and at most 86 cells in a level, so many small launches
could be expensive. Phase4R must measure alternatives rather than assuming acceleration.

The prepared path is currently used by the legacy replay; the conservative
`storm.evolve` event/retry path keeps its reference implementation. The prepared
coupled/column functions return existing public dataclasses and are reusable for
later integration. Rebuild context after terrain/routing or soil-parameter changes.
Read-only/noncontiguous dynamic input requires a per-call host copy; ordinary
contiguous writable event state avoids it. CuPy input is explicitly refused by this
CPU path, without transfer; the original CuPy reference remains available.

## Important existing roundoff follow-up

The reference uses `hpre=h−max(J−P,0)` and column depth `(h+P−J)+return`.
With positive rain and partial or complete intake, these mathematically compatible
expressions can differ by an ulp, causing the strict `old_flow_depth > depth_start`
guard to reject a physical state. Two deterministic tests preserve identical
reference/prepared refusal and unchanged inputs. Successful differential trajectories
exercise accepted rain/capacity regimes plus dry recession, while explicit branch-tie
cases cover partial/complete intake. No skip, xfail, error swallowing or conservation
relaxation was used to resolve the fixture failures.

Evaluate a roundoff-consistent expression in a separate correction with unchanged
water budgets and matched storms. This optimization deliberately preserves the
reference behavior; it does not solve that existing limitation.

## Reproduction

```
source agent_handoffs/tasks/phase6_complete_event/env.sh
source benchmarks/phase7d/candidate_env.sh
"$SYRUP_PYTHON" -m maple_syrup.benchmark_experiment --output-dir <new-output> --allow-maple-source-change
"$SYRUP_PYTHON" -m maple_syrup.benchmark_experiment --output-dir <reference-output> --allow-maple-source-change --hydrology-implementation reference
```

Exact commands, raw logs, profile, baseline/candidate manifests and test-hook are
local under `agent_handoffs/tasks/phase7h_hydrology`; generated storm output under
`outputs/phase7h_hydrology`. Design details: [hydrology design](hydrology_design.md).
