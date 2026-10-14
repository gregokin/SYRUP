# CW-A1 — conservative CUDA hydrology in the evolving-bed event (small-domain coupling)

Historical task record; pending statements below apply to that task snapshot. See [the current implementation status](implementation.md) for executed results and remaining gates.

Status: implemented by the sole writer of task CW-A1 (file-only tools; nothing here was executed by its author).
Executed results, acceptance and any corrections are recorded by the root under
`agent_handoffs/tasks/conservative_water/`. Baseline `b6b7f234…` plus the root's documentation edits.

## What changed

Two source modules, one new test package, this document.

### `src/maple_syrup/sediment_event.py`

* `SedimentEventControl.validated()` no longer refuses `storm.implementation == "cuda"`. It refuses the pairing
  `cuda` + `transport_implementation="numba"` (the compiled transport kernel is host-only). `"auto"` resolves to
  `"array"` for both the array and the CUDA sweeps and to `"numba"` only for the numba sweep (unchanged logic).
  New: `control.cuda` and `control.backend_record()` (host record of the resolved pairing for runner summaries).
* `_check_backend_binding(xp, control)` runs inside `_bind_phase_to_control`, i.e. at every public entry
  (`sediment_coupled_step`, `evolve_sediment_event` through `_check_event_inputs`, and the checkpoint validator)
  BEFORE any kernel or MAPLE call: `cuda` on a NumPy state, `numba` (sweep or transport) on a CuPy state and
  array + Newton on a CuPy state are refused with `SedimentEventError`. No state is ever transferred between host
  and device: CUDA is not a CPU fallback and the host sweeps are not a device fallback. The generic array sweep
  remains accepted on either namespace.
* `CudaHydraulics`: a caller-owned lifecycle object holding at most one `hydrology_cuda.CudaHydrologyContext`.
  `prepare(graph, column, t_s, reason)` returns a new context unless the held one `is_bound_to` the given graph
  and column (object identity through weak references plus array metadata); `adopt` installs it and accumulates its
  counted costs (`preparation_s`, `kernel_load_s`, host/device bytes); `bind` is the one-phase form;
  `record_commit_transfers` accumulates the counted traffic of commits; `summary()` is the host record.
  Nothing is persisted: a context is reconstructible from the graph and the column. There is no module-level cache.
* `sediment_coupled_step(..., hydraulics=None)`: with a CUDA control, step 1 runs
  `hydrology_cuda.prepared_coupled_step` on the lifecycle's context (bound to `state.graph` / `column`); without a
  lifecycle a context is prepared for THAT CALL ONLY (the documented convenience form, one counted static download
  and upload per invocation; never used by the loop). Steps 2–5 (array wet laws on the CURRENT active holdings,
  actual MAPLE pickup, characteristic array transport, actual MAPLE deposit/export, phase reconciliation) are the
  unchanged namespace-generic code.
* `evolve_sediment_event(..., hydraulics=None)`: with a CUDA control the context is prepared ONCE before any time is
  advanced (`reason="initial"`; a lifecycle already bound to this graph/column is reused) and the Newton kernels are
  loaded eagerly when selected; every attempt and retry of the same graph reuses the context (a retry is the same
  accepted state); at every commit the context of the REFRESHED graph is prepared BEFORE the commit is published
  and adopted together with it, so a preparation failure leaves nothing published (`reason="commit"`). The two
  phase-reconciliation counters are read through `to_int` (counted) instead of bare `int`. No preparation and no
  full-grid static transfer happens per step. A CUDA control without a lifecycle gets a private one.
* `_commit_and_reroute` is unchanged; the loop wrapper `commit_and_reroute` brackets it with transfer counters when
  a lifecycle exists. Commits still rebuild the routing graph on the host from ONE counted download of the committed
  elevation (`sediment_bed.refresh_routing`, unchanged).
* Default CPU behaviour: the array/numba numerical order is untouched; the CPU loop differs only by the function
  indirection of the commit wrapper and the counted `to_int` reads (identical values).

### `src/maple_syrup/sediment_experiment.py`

* Pairing policy `BACKEND_PAIRINGS = {"numpy": ("array", "numba"), "cupy": ("cuda",)}`: the runner executes the
  explicit CUDA form only. `cupy` + `array`/`numba` and `numpy` + `cuda` are refused (message contains "refused")
  BEFORE the case directory is touched; the generic array sweep on CuPy is reachable through the event API, not the
  runner. New keyword arguments at the end of the signature: `root_solver="bisection"`, `cuda_mode="auto"`
  (existing calls are unchanged). CLI: `--implementation` accepts `cuda`, `--backend cupy`, `--root-solver`,
  `--cuda-mode`. `BackendUnavailableError` is converted to a `SedimentEventError` ("refused").
* `cuda_memory_preflight(case, phase_bins, device_id, ...)` (correction 1): validates its arguments first (strict
  ints for bins / device / cap, a supported transport scheme, a finite safety factor >= 1), then counts the ACTUAL
  bytes of every unique array reachable from the six MAPLE state fields `bed_from_case` places on the device
  (`PREFLIGHT_CASE_FIELDS`: voxel column, active layer, water, sediment ledger with its eight
  `(n_process, ny, nx, nc)` arrays, pending / touch grids and process totals, committed topography, availability),
  multiplies by `PREFLIGHT_COEXISTING_STATES = 8` (correction 2: the documented conservative upper bound on the
  full state sets that can be alive at once in the runner's loop, named in the record's `coexisting_state_basis`:
  the retained initial `state0`, the current accepted state, the previous attempt's pickup and post-deposit
  results, which stay referenced while the next attempt is evaluated, the current attempt's pickup and post-deposit
  results, the commit candidate bed, and one within-call generation), adds MAPLE's whole-voxel working copies
  inside an exchange call and the commit's voxel + active snapshots, the characteristic phase (fraction + position)
  times `PREFLIGHT_PHASE_STATE_SETS = 5` (retained initial, current accepted, previous kernel prediction, current
  kernel prediction, current reconciled; `phase_state_basis`), the ~12-array transport working set, the CUDA
  statics / scratch / three storm-state grids and the accumulators, and scales by the safety factor (an additional
  margin, not a substitute for terms). The multipliers are intentionally conservative upper bounds on concurrent
  lifetimes, not an exact peak, and no memory-fit claim is made until the pool is measured. The small-case voxel
  cap `SMALL_CASE_VOXEL_BYTES_MAX` (256 MiB; Plot 1 is ~1.2 MB) is applied BEFORE the device is queried; only then
  is the device's free memory compared with the requirement. An estimate, not a measurement; the CuPy default pool
  is reported after the loop. The later memory-bounded representation (CW-B) gets its own preflight counting its
  production live buffers and owned transactions; this one bounds the whole-small-bed form only.
* On `cupy`: `resolve_backend` selects the device, the preflight runs, a `CudaHydraulics(mode=cuda_mode)` is created
  and passed to the loop; `synchronize(xp)` precedes the setup and loop timings; the summary's `backend` block
  gains `pairing`, and for device runs `gpu = {executed, cuda_hydraulics (lifecycle summary), memory_preflight,
  device_memory, claims}`; `resolved_config.syrup` gains `cuda_mode` and `backend_pairing` ONLY for device runs,
  so the host resolved-config hash is unchanged. Reporting converts device results with MAPLE's host helpers
  (`to_host`, `to_host_tree`, `to_float`) as before. The `routing` block records `root_solver`.
* Host runs: `backend.gpu` stays `"not exercised; no GPU claim"`, the transfer-boundaries note still says the
  cupy backend is refused unless `cuda` is selected, and the limitations say so. The `routing.method` string is the
  unchanged bisection text by default and names the safeguarded Newton truthfully when `root_solver="newton"`.

### `src/maple_syrup/sediment_bed.py` (correction 1, explicit namespace / evidence transfers only)

* `terrain_reference`: every static terrain-evidence array (`fixed(...)`) is brought to the host ONCE through
  MAPLE's counted `to_host` before the NumPy copy (a NumPy input passes through unchanged, so the host path is
  byte-for-byte as before); the rebuilt-versus-initial graph comparison reads the host evidence arrays through
  `to_host` explicitly. `refresh_routing` keeps its single counted committed-elevation download per refresh. No bed
  physics, commit or exchange change. Source fact: `RoutingGraph.aspect/slope/active/outlet/friction_factor` are
  host arrays by contract whatever `xp` is, so these conversions are explicit and counted rather than a changed
  numerical path.
* `sediment_event._commit_and_reroute` likewise reads the old / new graph evidence arrays through `to_host` once and
  forms its counts with `to_int`; `_check_event_inputs` compares the class diameters against `to_host` of the
  parameters' host copy.

## Not changed

Legacy replay (`legacy_*`, `benchmark_experiment`, `maple-syrup-legacy`, the `--transport-scheme legacy` default),
`complete_event`, `checkpoint`, `characteristic_transport`, `sediment_physics`, `hydrology_cuda`, MAPLE, MAHLERAN,
the isolated Phase 7e dependency, bin counts, physics, bounds. No pit storage, no splash, no ecology, no wind, no
restart.

## Rules the implementation follows

| Rule | Where |
|---|---|
| Refuse mixed backends and unsupported pairings before physical mutation | `SedimentEventControl.validated`, `_check_backend_binding` (event); `BACKEND_PAIRINGS` (runner) |
| CUDA is not a CPU fallback; no state transfer either way | `_check_backend_binding`; `hydrology_cuda.prepare_cuda_hydrology` refuses non-CuPy inputs |
| One preparation per graph/column binding; rebuild after commits; none per step or retry | `CudaHydraulics`, `evolve_sediment_event` (`reason` log: `initial`, then one `commit` per commit) |
| No stale context across a changed graph or parameters | `CudaHydrologyContext.is_bound_to` (identity + metadata) checked at every bind; a foreign/stale lifecycle is re-prepared and counted |
| Rejections retry from the unchanged accepted state | unchanged loop; retries reuse the bound context (`n_binding_hits == accepted + rejected`) |
| Device wet laws on CURRENT holdings; all classes can become exposed | unchanged `sediment_physics_step(…, state.bed.active_layer.mass_kg, …)` in either namespace |
| Actual MAPLE pickup/deposit/export, supply/refill/sorting, class budgets, face layouts | unchanged `apply_bed_demand`, `characteristic_step`, accounting |
| No new result fields (older checkpoint bundles stay loadable) | counts/costs live in `CudaHydraulics.summary()`; `SedimentEventResult` and `SedimentEventState` unchanged |
| No mutable global graph cache | lifecycle objects are caller-owned |
| Explicit memory preflight before device allocation; no large-case voxel residency | `cuda_memory_preflight`, `SMALL_CASE_VOXEL_BYTES_MAX`; lifetime multipliers `PREFLIGHT_COEXISTING_STATES = 8`, `PREFLIGHT_PHASE_STATE_SETS = 5` (documented upper bounds) |

## Declared comparison bounds (not fitted)

* Water fields (depth, soil water, discharge, velocity, cumulative water grids, hydrograph, committed elevation,
  slope, conveyance): rtol 2e-12 / atol 1e-14, the accepted CUDA-hydrology bound.
* Sediment fields (cumulative pickup / deposition / export request grids, per-class totals, mobile pool, voxel and
  active masses, availability, pending ledger, sediment velocity memory, sediment hydrograph, peaks, phase
  mass): rtol 1e-10 / atol 1e-14, the bound PREDECLARED for the coupled event by the root (the existing isolated
  T11 kernel test keeps its own rtol 1e-10 / atol 1e-15; the two are distinct and neither is changed); phase
  positions atol 1e-9 as in T11.
* Exact: accepted/rejected step counts, commit count and times, graph-change and reroute counts, branch counts,
  regime counts, aspect and outlet sets, substep counts.
* Round-off diagnostics (`max_transport_cell_residual_kg`, `max_phase_reconciliation_residual_kg`, numerical
  residual accumulators) are NOT compared across backends; each backend satisfies its own unchanged MAPLE bound
  (closure, request reconciliation, transport budget).

## Expected test commands (root executes; author did not run anything)

There is no SYRUP virtual environment: the user's MAPLE environments are `/home/okin/MAPLE/.venv` (CPU) and
`/home/okin/MAPLE/.venv-cupy` (GPU), and the verified isolated Phase 7e MAPLE candidate is selected by sourcing
`benchmarks/phase7e/candidate_env.sh` AFTER the Python environment (it installs and edits nothing).

```bash
cd /home/okin/SYRUP
# CPU (no device needed; GPU tests skip with an explicit message)
export PATH=/home/okin/MAPLE/.venv/bin:$PATH && source benchmarks/phase7e/candidate_env.sh
python -m pytest tests/conservative_water -q
python -m pytest tests/phase5 tests/phase6 tests/phase7b tests/phase7e tests/phase7f tests/phase7h tests/phase4s -q
python -m maple_syrup.sediment_experiment --case-dir outputs/plot1 \
    --output-dir outputs/conservative_water/plot1_array_60s --end-s 60 --implementation array
ruff check src/maple_syrup/sediment_event.py src/maple_syrup/sediment_experiment.py src/maple_syrup/sediment_bed.py \
    tests/conservative_water
# actual GPU 0 (the user allocation; check it is free first) with the CuPy environment
export PATH=/home/okin/MAPLE/.venv-cupy/bin:$PATH && source benchmarks/phase7e/candidate_env.sh
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/conservative_water -q
CUDA_VISIBLE_DEVICES=0 python -m maple_syrup.sediment_experiment --case-dir outputs/plot1 \
    --output-dir outputs/conservative_water/plot1_cuda_60s --end-s 60 --implementation cuda --backend cupy
```

Collected item counts are whatever the root's run reports (the CPU-only tests, the parametrised GPU tests and the
Plot 1 fixture tests skip or run depending on the device and the MAHLERAN root); the author does not assert a count.

## Coverage limits and what is not claimed

* Author-unexecuted: every assertion in `tests/conservative_water/test_coupled_backends.py` and every command above.
  The GPU parity assertions at the declared bounds are predictions; a failure there is a finding for the root, not
  a tolerance to adjust.
* Whole small bed on the device through the genuine MAPLE API: MAPLE still copies the whole voxel array in each
  of its erosion/deposition routines per call and the commit takes whole-bed snapshots; the graph is rebuilt on
  the host at every commit; MAPLE's validation performs its own counted host reads per step. Memory-bounded
  (windowed) beds and a fused CUDA transport remain the CW-B/CW-4/CW-6 work and are required for Chastre scale;
  no Chastre conservative event is possible or claimed.
* No GPU complete event, no GPU checkpoint/restart (host-only by contract; a CUDA control in a host bundle is
  refused by the binding check), no pit storage, no dry-reset option change, no speed claim (Plot 1 on the GTX 1080
  Ti is expected slower than the Numba CPU sweep, as for the water-only drivers).
* Commit-time re-preparation downloads the static graph for validation and uploads owned copies (F4 of the
  architecture report); its cost is now measured per commit (`CudaHydraulics.summary()["log"]`) rather than
  assumed, and an in-place conveyance update of `hydrology_cuda` is the documented mitigation if it dominates.
* The legacy replay comparison identity does not apply to these operators (different transport, evolving bed,
  limited supply); the reference here is the host array/numba coupled event.
