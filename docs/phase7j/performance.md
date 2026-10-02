# Exact CPU routing optimization

The prepared CPU hydrology used by the default legacy replay now batches the
existing bisection across independent cells in each routing level and skips its
iterations where the right-hand side is non-positive. It retains the original
per-cell arithmetic, donor order, bracket, timestep and iteration count. Zero is
an exact test: no small positive water depth is discarded. Non-finite/negative
inputs still reach the existing refusals. There is no Newton approximation,
fastmath, clipping or tolerance change.

The original serial compiled sweep and array solver remain available. The
conservative `storm.evolve` path continues to use its reference hydrology;
this change accelerates prepared hydrology and the default legacy benchmark.
[Design](design.md), [measurements and source bindings](measurements.json).

## Repeated Plot1 storm measurements

Frozen elevation/routing, the same no-splash legacy replay, 1200 cells, six grain
classes and 5400 one-second steps. Both sides use the default replay's existing
constant conductivity and rainfall inputs. The heterogeneous exact-input water
qualification is a separate run below; it is not mixed into this timing table.
Baseline: committed `7e254af`, immutable package snapshot verified byte for byte.
Actual MAPLE package: `72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65`.

Three fresh-process pairs, with balanced order: baseline1/candidate1,
candidate2/baseline2, baseline3/candidate3. No task tests or other benchmarks ran
concurrently; other user workloads were not stopped. Intel Core i9-7900X CPU.

| Measurement, median of three | Before | Batched | Change |
|---|---:|---:|---:|
| External process, imports/setup/JIT/output included | 19.38 s | 17.27 s | 10.9% shorter |
| Storm loop, first-call JIT included | 16.10 s | 13.90 s | 13.6% shorter |
| Steady hydrology, first call excluded | 3.97 s | 1.83 s | 54.0% shorter; 2.17× speed |
| First hydrology call, JIT included | 2.516 s | 2.607 s | 0.091 s higher |
| Loop minus timed first-step components | 10.92 s | 8.62 s | 21.1% shorter |
| Peak whole-process RSS | 376.56 MiB | 377.88 MiB | 1.32 MiB higher |

External ranges: 19.06–20.39 s before, 17.20–17.28 s after. The loop-minus-first
row subtracts timed components, not the entire first driver iteration; it is not
a separately timed warm whole process. Component medians need not sum to the
loop median. RSS includes imports/JIT; it is not kernel peak allocation.
There is no new whole-application Fortran speed comparison in this task.

The compiled routing loop uses packed double-precision square roots/multiplies
on this CPU; assembly inspection found two packed square-root instructions and
no fused multiply-add instructions. In the prior default storm, 55.25% of
cell-steps had exactly zero RHS. Batched independent wet cells also benefit:
[warm sweep measurements](measurements.json) separate original serial, serial
with dry skipping only, and batched execution on wet/mixed/dry synthetic grids.
These are isolated kernel timings, not whole-model speedups.

The new scratch is five arrays of the widest level's length, allocated per call
and never shared. Plot1 needs 3440 bytes of this scratch (86 lanes × 5 × 8 bytes).
The prepared immutable context remains 132536 bytes. No new full-domain copy,
retained storm history or shared mutable work buffer was introduced.

## Larger-grid hydrology calls

Warm, fixed synthetic wet states; graph/context setup and JIT excluded, alternating
serial/batched prepared calls in the same process. All compared public fields
are bitwise equal. These are calls, not complete storms or GPU scaling.

| Cells | Shape | Prepared serial | Prepared batched | Speed |
|---:|---|---:|---:|---:|
| 1260 | 60 × 21 | 0.690 ms | 0.252 ms | 2.74× |
| 16512 | 128 × 129 | 9.753 ms | 4.243 ms | 2.30× |
| 65792 | 256 × 257 | 39.192 ms | 16.612 ms | 2.36× |

## Scientific qualification

All three matched legacy storms have **every saved numeric array bit-for-bit
unchanged**, with exact regime counts and peak timing. This preserves the legacy
sediment replay's fixed-composition/unlimited-supply/clipping-accounted behavior;
it does not establish conservation of an evolving MAPLE bed.

Two independent full heterogeneous Plot1 water comparisons used the captured
MAHLERAN conductivity, exact rainfall, REAL32 soil initialization and unchanged
MAPLE dependency from [Phase7i](../phase7i/qualification.md):

- Prepared batched versus prepared original serial: **all 36 public fields at
  every one of 5400 steps are bitwise equal**. The harness explicitly selects
  each dispatcher on every call; arithmetic ASTs match the committed baseline.
- Prepared batched versus unchanged `storm.coupled_step`: the original
  `rtol=2e-12`, `atol=1e-14` comparison passes, with maximum normalized error
  0.1762 (acceptance bound 1), identical to Phase7i.

Saved heterogeneous history, spatial fields and forcing also exactly match the
accepted Phase7i replay. Its MAHLERAN comparison remains runoff −0.782924%,
peak flow −0.026484%, exact peak time 1321 s, peak-depth relative L2 0.023785%
and velocity 0.011120%. Water residual remains −1.776e-15 m³ under the unchanged
5.811e-7 m³ MAPLE-derived bound. Local/routing budget tolerances are unchanged.

Verification: 411 existing regression tests passed and five device-dependent
checks skipped in the CPU environment; 202 new tests pass after two test-only
corrections. The initial combined run had six failures in the test harness:
a renamed used variable and a raw single-cell fixture whose writable donor
arrays differed from production's read-only arrays. Both were corrected without
changing production physics or test targets. Final focused run: 202 passed in
8.79 s; Ruff and whitespace checks pass. Six additional selected tests directly
execute original Fortran against the new prepared routing phase (chains,
branching, dry/wetting and Plot1); all pass at the existing 1e-11 m depth bound.

Claude implemented the bounded change and its first fixture/lint correction;
Codex independently reviewed, corrected the two remaining test defects, executed
qualification and performance runs, and authored this report. Final read-only Claude review found no blocking defect; Codex accepted the bounded
CPU optimization. Claude cross-checked evidence and did not execute tests or timings. Raw task evidence: `agent_handoffs/tasks/phase7j_routing_solver`;
generated storms: `outputs/phase7j_routing_solver`.

## Remaining work

GPU hydrology kernels are still pending. This CPU layout exposes independent
work per level, but does not qualify GPU acceleration; Plot1's 66 narrow routing
levels may make launch/barrier costs significant. Safeguarded Newton and other
numerical departures remain deferred because this gain preserves exact results.
The existing positive-rain `hpre` strict roundoff refusal is preserved.

Steady default-storm wet physical-law evaluation now takes about 3.52 s,
legacy sediment traversal about 2.01 s, and hydrology about 1.83 s. These are
component measurements, not a matched component profile of Fortran. Further CPU
work should reassess these costs; resident GPU hydrology remains a separate task.
Multi-bin optimization remains the previously deferred Phase7g.
