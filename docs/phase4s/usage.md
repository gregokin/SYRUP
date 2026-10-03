# Phase 4S — water-only CUDA hydrology: usage and contracts

Status: implemented by Claude with file tools only; **nothing in this document was run by its author**. Codex records
test, parity and benchmark results elsewhere (acceptance/performance/measurements are Codex-owned). This page is usage
and contract documentation, not evidence.

## What it is

An explicit, opt-in, **water-only** FP64 CUDA implementation of the frozen-terrain storm hydrology: rainfall scaling,
both column laws (`fixed_ksat`, `pavement_hawkins`: Smith–Parlange capacity, drainage, saturation return), the legacy
old-flux branches, the method-5 ordered routing with every per-cell and global check, retries, cumulative bookkeeping and
hydrograph rows, all on the device. The discrete equations, expression order, tolerances and refusals are those of the
CPU path. It is **not** a GPU sediment, wind, restart-to-disk, evapotranspiration, splash, dry-reset or evolving-terrain
model, and nothing here claims speed. CPU defaults, the legacy replay, sediment and wind code are unchanged.

Requirements: CuPy and a CUDA device. **Numba is not required** and is never imported by this path. There is no
fallback: a missing CuPy/device/compiler is an error (`CudaUnavailableError`).

## Command line

```
python -m maple_syrup.storm_experiment --case-dir outputs/plot1 --output-dir outputs/plot1_storm_cuda \
    --max-dt-s 1 --implementation cuda --backend cupy
```

`--implementation cuda` without `--backend cupy` is refused before anything is read (no automatic backend switch). Existing
choices (`array`, `numba`) and the default (`numba`) are unchanged, as are the output files and every budget check.
The summary gains a `cuda_hydrology` block containing the **actual** context summary (launch structure, widest level,
static bytes, kernel register/local-memory attributes), the kernel provenance (source hash, compile options, device,
toolchain), the preparation timing and counted transfers (the one-time static download/upload), the loop transfers (one
144-byte packet per attempted step) and the explicit final-report downloads, kept separate.

## Python API

```python
from maple_syrup import hydrology_cuda as hc
from maple_syrup.storm import StormControl, evolve

ctx = hc.prepare_cuda_hydrology(graph, params)            # CuPy graph/params; validated once, owned device copies
step = hc.prepared_coupled_step(ctx, rain, state, dt, StormControl(implementation="cuda"))   # one pure step
col = hc.prepared_column_step(ctx, depth, soil, rain, dt)  # the column physics alone (no routing guards)
result = evolve(graph, params, field, schedule, state, end_s,
                StormControl(implementation="cuda"), report_every_s=60.0, cuda_context=ctx)  # ctx optional
```

* `storm.STORM_IMPLEMENTATIONS = ("array", "numba", "cuda")` is separate from `routing.IMPLEMENTATIONS`
  (`("array", "numba")`, unchanged and used by every other module and CLI). The sediment event controls refuse `"cuda"`
  before any physics (`SedimentEventError`); GPU sediment does not exist.
* `storm.evolve` is the **single** host scheduler (boundaries, substeps, retries, floors, guards, exceptions). For `cuda` it
  prepares a context once before the loop (or uses `cuda_context`), calls the prepared step for each attempt and uses fused
  accumulate/report kernels; nothing else differs. There is no second driver.
* `storm.coupled_step` with a CUDA control prepares a context **on every call** (a counted static download): a convenience
  form. Loops should prepare once.
* `prepared_column_step` reuses the very `pre_cell` device function of the coupled step. It applies no old-flux,
  `hpre`/`h*`, Courant or previous-discharge check, so it accepts states the coupled step refuses; same `ColumnStep`, errors,
  messages, `dt = 0` identity as `infiltration.column_step`. It is the reusable column physics for other solvers.

## Contracts

* **Inputs**: exact `cupy.ndarray`, float64, C-contiguous, `(ny, nx)`, on the current device. NumPy arrays, lists,
  subclasses, other devices and non-contiguous views are refused; nothing is converted or transferred. Producers on other
  streams must be ordered before the call (the caller's responsibility; the context's static data is synchronized at
  preparation and may be used on any stream).
* **Outputs**: fresh CuPy arrays every call; result scalars (`export_m3`, branch counts, maxima, …) are 0-d CuPy views of a
  fresh per-call packet (no host read). A failure raises after the one packet read, before any result exists, and never
  modifies an input or an earlier result. `RoutingStepRejected` still means "retry the same state with a smaller dt".
* **Errors/precedence** equal the CPU prepared step: context/state/control types → `dt` → input structure → column failure
  (lowest recorded bit; `dt = 0` keeps the input bits) → scalar-option failure (deferred: with invalid options only the
  column stage runs, so a bad iteration count is never executed) → routing failures in the CPU order → previous-discharge
  failures.
* **Context**: owned device copies of the static graph/column data. The device arrays are immutable by contract. The host
  metadata that reaches kernel launches (counts, extents, dx, model code, launch structure, level bounds, device) is
  **sealed at preparation** and compared before every launch, together with the pointer/shape/dtype/contiguity/device
  fingerprints of the owned arrays; a forged context (for example with `dataclasses.replace`) is refused before any enqueue.
  Arbitrary content mutation of the owned arrays is not detected (no per-step device hashing). Mutating the caller's own
  graph/parameters after preparation does not affect the context. A new graph or parameters need a new context.
  `shape` and `level_bounds` must be tuples (a list equal to them is refused).
* **Binding of an external context**: `evolve(..., cuda_context=ctx)` (and `ctx.is_bound_to(graph, params)`) accepts the
  context only for the very graph and parameter OBJECTS it was prepared from (weak identity, no strong reference is held, a
  recycled address cannot pass) whose array metadata (pointer/shape/dtype/contiguity/device) is unchanged. Same-shape,
  same-model parameters with different values, an equal-data twin, or a `dataclasses.replace` copy are refused before any
  step. An in-place CONTENT change of the source arrays after preparation is not detected (the context owns static copies and
  is immutable by contract): prepare a new context.
* **Accumulator helper**: `CudaStormAccumulator` re-checks the current device, the sealed context metadata and the
  metadata of its own arrays on every public call, and validates the packet (exact C-contiguous `uint64[18]` on the context's
  device), the time, row and count arguments (no bools, finite, in range) before any enqueue.
* **Launch structure** (chosen once at preparation, recorded in the summary): `fused` = one launch of one 128-thread block
  with a barrier after each dependency level (widest level ≤ 128 cells); `split` = parallel cell kernels and one launch per
  level (wider graphs). `prepare_cuda_hydrology(..., mode="fused"|"split")` forces one for testing.
* **Host traffic per attempted step**: one counted device-to-host read of a 144-byte packet; per accepted step one
  accumulate launch; per hydrograph row one report launch. Static data and final grids are downloaded/uploaded
  separately and counted separately.
* **Floating point**: `--fmad=false`, IEEE round-to-nearest intrinsics, no fast math, NaN-propagating min/max. Device
  `expm1`/`pow` may differ from NumPy by an ulp; scalar sums use a fixed, deterministic tree rather than NumPy's pairwise
  order. The declared bound against the CPU is rtol 2e-12 / atol 1e-14; counts and times are compared exactly.

## Known limits

Water only; frozen terrain and routing; no restart to disk; asynchronous device faults surface at the next synchronization
(the packet read) as CuPy/CUDA errors rather than `RoutingError`; no `compute-sanitizer` result; performance and scaling are
unmeasured by the author. The strict roundoff guard of the coupled step (`old_flow_depth_m` above `depth_start_m`) is
intentionally unchanged and refuses the same inputs on the device as on the CPU.
