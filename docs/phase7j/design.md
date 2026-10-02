# Phase 7j — level-batched ordered sweep (design and reproduction)

Status: accepted exact CPU optimization. Claude implemented it with file-only tools; Codex independently
verified scientific behavior and measured runtime/memory. Final Claude review found no blocking defect.
The batched sweep is the prepared default. Outcomes and limits: [performance](performance.md).

Baseline: clean `7e254af`. Unchanged and still the oracle: `routing.py`, `storm.py`, `infiltration.py`,
`routing_numba._sweep`, `routing_numba.compiled_sweep()` and `routing_numba.run_sweep()` (the standalone numba
`route_step` path), and the NumPy/CuPy array sweep. `--hydrology-implementation reference` keeps the original
hydrology.

## What changed

| File | Change |
|---|---|
| `src/maple_syrup/routing_numba.py` | NEW `_sweep_batched` + `compiled_sweep_batched()` (same call signature as `_sweep`); `reset_compiled()` drops both dispatchers; small `_require_numba()` shared by both factories. `_sweep`/`compiled_sweep`/`run_sweep` numerically unchanged. |
| `src/maple_syrup/hydrology_numba.py` | The kernel closure binds `compiled_sweep_batched()` instead of `compiled_sweep()`; docstring and `kernel_provenance()["ordered_sweep"]` text. No column/route/check equation touched. No selector, context field or CLI flag. |
| `tests/phase7j/`, `benchmarks/phase7j/bench_sweep.py` | NEW. |

## The kernel

Per dependency level, levels in series (donors are always in earlier levels):

1. every cell: donor sum `((0 + d0) + d1) + d2) + d3` in `DONOR_SLOTS` order (non-donors add `0.0`), raw
   `rhs = base + qin*c` (stored unchanged in `rhs_lo`), `qin_new_lo`, `flow_lo = 0.0`;
2. cells with `rhs > 0.0` are packed into contiguous per-call scratch; the root iterations are the OUTER loop and the
   independent packed cells the INNER loop, with exactly the original multiply sequence
   (`w*=0.5; mid=lo+w; t=((sqrt(mid)*mid)*k)*c+mid; if t < rhs: lo=mid`), the proven `[0, R]` bracket and the
   caller's iteration count (1..200);
3. after all iterations every cell gets `q = (sqrt(lo)*lo)*k` — the same formula for non-positive and invalid RHS.

Exactness of skipping non-positive cells: for `rhs = ±0`, `mid = +0`, `t = +0` and `0 < rhs` is False; for
`rhs < 0` the first `mid` is negative, `t` is NaN (or `+0` once `w` underflows) and the comparison is False; a NaN RHS
gives NaN and False. So `lo` remains `+0.0` exactly as in the original, and the raw RHS is stored so the downstream
non-finite / negative-RHS refusals and their precedence see identical inputs. Tiny, subnormal and `+inf` positive RHS
are NOT skipped. There is no epsilon threshold, no clipping, no fastmath, no reassociation, no prange.

No tolerance, timestep, bracket, iteration semantics or refusal message changed; `bisection_iterations=5` still
produces the non-recoverable "bisection did not reach" `RoutingError` through the unchanged constitutive check.

## Ownership and safety

The scratch (`idx`, `rhs`, `k`, `w`, `lo`, each of the widest level's length) is allocated inside each call; nothing
is shared between calls or threads. Missing Numba raises `NumbaUnavailableError` (no fallback). The existing
`MAPLE_SYRUP_NUMBA_CACHE` contract applies to both dispatchers. Host NumPy only; no transfer is introduced and the
CuPy refusal of the prepared path is unchanged. The new helper is CPU-only and public only for tests/benchmarks.

## Alignment with a later GPU kernel (nothing implemented)

The structure is the intended device form: one launch (or one barrier) per level; one lane per cell; a fixed trip
count with no data-dependent exit; the positivity test is a per-lane predicate (a device kernel can predicate instead
of compacting); no atomics; results independent of cell order within a level; arrays already level-ordered and
donor tables slot-major. Whether many small launches pay off on Plot1 (66 levels, at most 86 cells per level) is the
open Phase 4R question. CPU timings establish nothing about GPU efficiency.

## Deferred

Safeguarded Newton/secant, tight-bracket or decision-replay solvers (they change root bits, see
`agent_handoffs/tasks/phase7j_routing_solver/design.md`); sediment, infiltration and the known positive-rain
`hpre`/`h*` roundoff refusal (preserved identically and tested); GPU kernels.

## Reproduction (Codex)

```
source agent_handoffs/tasks/phase6_complete_event/env.sh
source benchmarks/phase7d/candidate_env.sh
"$SYRUP_PYTHON" -m pytest tests/phase7j tests/phase7h tests/phase4 -q
"$SYRUP_PYTHON" benchmarks/phase7j/bench_sweep.py --output <out>/sweep_bench.json
"$SYRUP_PYTHON" -m maple_syrup.benchmark_experiment --output-dir <new-output> --allow-maple-source-change
"$SYRUP_PYTHON" -m maple_syrup.benchmark_experiment --output-dir <ref-output> --allow-maple-source-change --hydrology-implementation reference
```

Acceptance criteria (both verified; see the performance report): every test above passes with
bitwise equality of all 36 public fields against the original-sweep prepared path, the unchanged reference within
rtol 2e-12 / atol 1e-14, full-storm counts and peak time exact; and a measured steady-hydrology gain on repeated
alternating fresh-process Plot1 runs. If either fails, revert the one-line bind in `hydrology_numba._build_kernels`
(`compiled_sweep_batched` -> `compiled_sweep`).
