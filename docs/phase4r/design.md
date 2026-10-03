# Phase 4R step 1 — CUDA ordered routing sweep (GPU qualified)

Status: Claude implemented with file-only tools; Codex independently tested on actual GPUs and measured performance.
484 GPU tests and 466 CPU regression/contracts passed. Results and scope: [performance](performance.md). Full design
rationale: `agent_handoffs/tasks/phase4r_gpu_routing/design.md`. This CUDA option qualifies the routing sweep; complete GPU hydrology and storms remain separate work.

## What exists

| File | Role |
|---|---|
| `src/maple_syrup/routing_cuda.py` | RawKernel (one thread per cell, one launch per dependency level, current stream, block 128), owned static context with weak identity cache, `run_sweep`, provenance. |
| `src/maple_syrup/routing.py` | `ROUTE_IMPLEMENTATIONS = ("array","numba","cuda")` used only by `_check_step_options`; `_route` cuda branch. `IMPLEMENTATIONS` unchanged, so `StormControl`, the experiment CLIs and the prepared hydrology still cannot select `cuda`. |
| `tests/phase4r/`, `benchmarks/phase4r/bench_cuda_routing.py` | Contract/GPU tests and the performance harness. |

## Contract

* Same per-cell operations and order as `routing_numba._sweep_batched`: donor sum `0+a+b+c+d`, raw `rhs`, `[0, rhs]`
  bracket, 1..200 bisections, strict `<`, exact `rhs > 0` skip, `q = (sqrt(lo)*lo)*k`. Compiled with `--fmad=false`
  and `__dadd_rn/__dmul_rn/__dsqrt_rn`; no fast math. `--prec-div/--prec-sqrt` document intent for single precision
  only and are not claimed to make FP64 exact. The sweep has no division.
* `prepare_cuda_routing(graph)`: validates every runtime array (exact CuPy, dtype, shape, C-contiguity, current
  device) before any launch; downloads the static data once via `maple.core.backend.to_host`; checks index range,
  dependency order, receiver consistency, uniqueness/completeness, bounds and outlets; uploads fresh owned copies;
  synchronizes once. Cache hit: metadata checks only (no transfer, no synchronization).
* **Graph freshness:** the caller must not mutate the graph's host or runtime arrays after preparing. The sweep
  uses the owned copies while the rest of `_route` reads the graph's arrays, so a mutation would make them
  inconsistent; CuPy arrays cannot be frozen. Pointer/shape/dtype changes are detected, content changes are not.
  `release_cuda_routing(graph)` drops a context. The cache holds no reference to the graph.
* `run_sweep(graph, base_lo, c, iterations)` returns four fresh arrays and never writes inputs or static data; base
  values are not inspected, so negative/NaN/Inf reach the unchanged shared refusals.
* Errors: missing CuPy/device or compile/load failure → `CudaUnavailableError`; launch failure and all structural
  problems → `RoutingError`. No fallback and no hidden transfer.

## Additional contract notes (correction 1)

* **Static ownership.** The cached context's device arrays are the sweep's truth and must be treated as immutable.
  Content mutations of the graph (host or CuPy) after preparation are NOT detected on a cache hit (only pointer,
  shape, dtype, contiguity and device are); doing so makes the wrapper's own reads of the graph and the sweep's owned
  copies inconsistent. Replace the graph (or `release_cuda_routing`) instead.
* **Validation.** The preparation checks duplicate the CPU `prepare_hydrology` checks and are STRICTER in one
  respect: slot `s` of a receiver must hold the neighbour at `receiver + DONOR_SLOTS[s]` (the summation order is the
  legacy order). Malformed `shape` or `level_order_host` raise `RoutingError`, not `AttributeError`/`TypeError`. The
  CPU validators are unchanged.
* **Asynchronous errors.** Only enqueue-time failures become `RoutingError`. A device fault inside a kernel surfaces
  at a later synchronization (the flag read in `route_step`, or any later CuPy call) as a CuPy/CUDA runtime error.
* **`c` underflow.** `c = dt/(2 dx)` must be finite and > 0. If it underflows to 0 for an extreme tiny `dt/dx` the GPU
  path refuses (`RoutingError`) while the array/numba paths would run.
* **Scope.** `route_step(implementation="cuda")` only. `legacy_stale_inflow_step` refuses `cuda`; `StormControl.validated()`
  rejects it and `storm.coupled_step` refuses it before any column work (a minimal guard, no full validation). There is
  no qualification of infiltration, sediment, coupled water, storms, restart or CLI execution with this CUDA kernel.
  Existing array-backend capability is separate.
* **Benchmark scope.** See `timing_scope_notes` in the benchmark JSON: event intervals, wall rows and CPU rows have
  different scopes; the CPU `route_step` row is the original serial sweep, not the production CPU default (batched
  prepared hydrology); states are random on the stated topology; memory fields are pool/device-free deltas, not peaks.

## Not done / next bounded steps

CUDA-graph capture of the per-level launches, or a single-block kernel for narrow levels (Plot1: 66 levels, widest 86),
are the candidate launch-overhead remedies; neither is implemented. Measurements show the batched CPU sweep remains faster at all tested sizes on this GTX 1080 Ti;
see [performance](performance.md).
