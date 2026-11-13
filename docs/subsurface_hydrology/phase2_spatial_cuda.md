# Phase 2 S2-A4.2 — fixed explicit spatial CUDA water trial component

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_spatial_cuda_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: author-implemented on a **CPU-only host** (no GPU, CuPy or NVRTC was used by the author) and **uncommitted**. This source asserts no qualification; qualification of a specific source is recorded only in Root's source-specific receipts, and native, lifecycle, restart and storm use is unsupported. A `SpatialSoilColumn` prepared with
`backend="cuda"` can be evolved through the same six explicit helpers as the CPU backends. Production/native hydrology admission stays false (`hydrology_step_supported = False` describes the parameter owner).
No geometry, water-fate, texture-to-property, averaging, reset or calendar choice was made; no evolving bed, storm, restart, LEAF or lifecycle integration exists; three user physics questions are unanswered.

## One physical body

`subsurface_cuda._BODY` (accepted in A4.1) is the only stage text. `subsurface_cuda.render_body_source(readers, step_head, tail="")` is a PURE function (no registry, no cache; the dormant A4.1
`register_variant` was removed) and `render_source("common")` still returns the A4.1-accepted source byte for byte (sha256 `46f1307e…`). `subsurface_cuda_spatial.render_source()` renders the same body with:

* **spatial readers** (A1 strided raw maps `p[l*stride_layer + i*stride_cell]`; scalar 0/0, shared 1/0, cell N/1): `DZ = dz`, `RES_M = θres·dz`, `FC_M = θfc·dz`, `SAT_M = θsat·dz`, `CAP_TOL = coef·SAT_M` in the product order of
  `soil_layers`/A1; `FAC(k)`/`FAC_UP(k)` read the hoisted scratch when `hoisted != 0`, otherwise `relax_factor(G, 1/pair, dt)` per cell and interface. No `(L,N)` derived tensor, host mirror, per-thread array (other than 7 contribution doubles) or Python cell loop;
* **the body as a `__device__` function** `spatial_column` whose early `return` is the invalid-lane guard, plus **wrapper kernels** in which EVERY lane (padded or inactive) reaches every barrier: only valid lanes read or write arrays.

Hoisting: when the conductance maps are SHARED the existing factor kernel runs unchanged on the retained `(L-1,)` maps (`F = 1` array in constant mode, `2` in donor mode, `max(L-1,1)` doubles each); per-cell maps evaluate the factor inline. In constant mode the upward conductance IS the downward map; a one-layer stack has no
interface maps (dummy never-read pointers).

## Launches, reduction, packet

Current stream, in order: [factor] → `maple_syrup_spatial_step` (per-lane contributions: before, after, added, return, drainage, evaporation, unmet; a block tree in static shared memory `7·8·128 B`; `7·nb` block partials; integer atomics: OR of the cell flag words and the flagged-cell count) →
`maple_syrup_spatial_step_finalize` (one block, 256 threads) → ONE 72-byte `maple.core.backend.to_host` of the 9-word `uint64` packet `[OR flags, flagged count, 7 total bit patterns]`. The host rebuilds the original nine-value float packet
(lowest set bit of the OR = the minimum over cells of each cell's lowest bit; exact count; totals) and `subsurface._resolve` applies the original closure coefficient, term count and typed errors (`SubsurfaceStepError`, `SubsurfaceFitFailure`). No CUB/CuPy reduction, no scalar array, no `rows` array, no new tolerance,
no first-flagged-cell claim. Totals are summed in a fixed block/tree order, which is roundoff-level different from NumPy's pairwise order and CuPy's; the original `volume_roundoff_bound_m3(n_terms, …)` still applies. Other kernels: uniform init (4-word diagnostic, threshold checked in EVERY cell), moisture, storage (+ the one-row finalize, 8-byte packet).

## API (explicit; CPU semantics unchanged)

Same six functions. For a CUDA owner arrays must be exactly `cupy.ndarray` (float64, C-contiguous, exact shape, `.device.id` = owner device) and the current device must equal the owner's, checked before any copy/launch; NumPy arrays, other devices and CPU states are typed
refusals (no implicit transfer, no fallback, no conversion). `spatial_column_state` is a device-to-device copy; `spatial_uniform_state` keeps the HOST L-vector (metadata preflight, then admission, then conversion, then one counted `to_device` of `8L` bytes and one counted 32-byte read; the threshold failure raises the CPU message and discards the water);
moisture and storage are device queries (`spatial_storage_m3` zeroes inactive cells, never clips a negative finite value and refuses a non-finite or unrepresentable total). The state copy and the moisture query may complete asynchronously on the current stream: later cross-stream ordering is the caller's. Keyword-only
`host_memory_budget_bytes`, `host_reserve_bytes`, `host_available_bytes_fn` are the separate HOST quota/reserve/reading and are refused for CPU owners; for a CUDA owner `memory_budget_bytes`, `reserve_bytes`, `available_bytes_fn` are the DEVICE quota/reserve/reading (default: driver free memory plus the pool's free bytes under the pool limit).
Typed errors: out-of-memory → `SpatialLayerMemoryError`, CUDA runtime/driver/launch/compile → `SpatialLayerBackendUnavailableError`; fresh outputs are dropped, nothing is published, the accepted state, forcing and retained maps are only read. Context recovery after a CUDA error is NOT claimed.

State: `SpatialSoilState` gained defaulted `namespace` (`"numpy"`/`"cupy"`) and `device_id`; CPU states are unchanged (tag `spatial_explicit_cpu_state`, read-only NumPy array); CUDA states carry `spatial_explicit_cuda_state`. CuPy arrays have no read-only flag, so immutability is LOGICAL only
(module-created, `const` kernel arguments, fresh arrays per step, owner-bound identity) and nothing detects an external write. `STATE_COMPONENT_TAGS` lists both; no cross-backend conversion, checkpoint or migration. `summary()["water_evolution"]` lists `["numpy","numba","cuda"]` with a static `cuda_trial_status` that points to Root's receipts and states that native/lifecycle/restart/storm use is unsupported.

## Memory (payload model; bytes; `nb = ceil(N/128)`; `estimate_spatial_operation_memory(..., backend="cuda", host_reserve_bytes=…, hoisted_factor_arrays=…)`)

| op | device (main quota) | host (separate quota) |
|---|---|---|
| state | `8LN` | 0 |
| uniform_state | `8LN` + `8L` theta + 32 diagnostic | `24L` conversion lifetimes + 32 diagnostic read |
| moisture | `8LN` | 0 |
| storage | `8·nb` partials + 8 packet | 8 |
| step, step_with_recharge | outputs `(LN + 2(L-1)N + 4N)·8` + scratch `7·8·nb` + 72 packet + `F·max(L-1,1)·8` | **144** (the 72-byte raw `uint64` array returned by the read AND the 72-byte float packet rebuilt for `_resolve`, live together; wire traffic stays ONE 72-byte read) |

`required_bytes` = `device_required_bytes` = outputs + scratch + device reserve; `host_required_bytes` = host payload + the 64 KiB Python allowance + the host reserve. Caller state/forcing and the owner's retained maps are excluded. Static shared memory (`7168 B` per step block) is reported, not a global allocation.
Scratch (partials, packet, factors, the uniform theta copy) is held until after the synchronising packet read and then released to the pool. Excluded and declared: NVRTC/module load, the CUDA context, allocator pool policy and 512-byte block rounding, interpreter/whole-process memory; estimates are payload models, not reservations. Integer launch widths are validated before any cast
(`nl` ≤ 2³¹−1, grid ≤ 2³¹−1 blocks). CPU estimates (`numpy`, `numba`) are unchanged (compared with the pre-A4.2 module).

## Verification (author, CPU only)

`tests/phase2_spatial_cuda/`: **host execution** of the rendered spatial source by g++ with real threads and a real barrier (`hostgpu.py`), through the REAL Python launchers and public API on a numpy-backed CuPy plumbing double: agreement with the NumPy spatial path within rtol 2e-12/atol 1e-14 (bitwise at zero diffusivity),
identical typed failures/flag bits/flagged counts, N = 1/127/128/129/257 and all-inactive, L = 1/2/3/5, both modes and bottoms, share/pulse/fit-failure, random scalar/shared/cell layouts incl. hoisted and per-cell factors, uniform/moisture/storage; plumbing tests of admission boundaries on both sides, device/namespace/type refusals before allocation,
exact allocation lists equal to the model, one 72-byte read and zero uploads per step, scratch lifetime, typed OOM, no mutation, owner release, cold import, kernel-cache bounds. `test_cuda_definitions.py` holds actual-GPU definitions (106 cases) that are WRITTEN BUT NOT RUN. Its pool test checks the bytes REQUESTED from the allocator (exactly the model), the `used_bytes` delta against the pool-charged unique output blocks, an upper bound on `total_bytes` growth only (a retained split free chunk is reused, so growth is not a lower bound of the outputs), and a plateau with a resident zero forcing outside the loop and explicit parentheses. These are not hardware evidence and use no fake CUDA-tagged owner as proof of device behaviour.

Limits: no register/spill/shared-memory/occupancy measurement, no performance or compile-time claim, no pool-behaviour measurement, no multi-stream or multi-device claim beyond refusals, no large-case, storm, restart or scaling claim.

## Independent fixed-profile component qualification

Root accepted package `c4c0e3122a150db87d5e5f36c1288862ef7bdc40f67f07a5146ffb445acf9f6a` (119 files), using actual locked MAPLE9f and the existing MAPLE CPU/CUDA environments. Independent checks passed: 1,088 CPU tests (121 skips), 106 spatial GPU tests, 23 common GPU tests (one skip); each backend matched 1,224 independent column evaluations bit-exactly. Seven GPU hand-controls and real allocation/plateau checks passed. The host packet peak is 144B while device-to-host traffic remains one72B packet. See [phase2_spatial_cuda_acceptance.json](phase2_spatial_cuda_acceptance.json) for exact sources and limits. This qualifies fixed explicit column trials, not evolving-bed, storm, LEAF, restart or the whole Phase2.

Matched warmed full-wrapper timing on the assigned **NVIDIA GeForce GTX1080Ti**, 200,000 cells (single-thread CPU; milliseconds per call; setup/first-call separate):

| Layers | Diffusion mode | Property layout | NumPy ms | Numba ms | CUDA ms | Numba/CUDA |
|---|---|---|---|---|---|---|
| 3 | constant | shared | 36.440 | 25.981 | 0.838 | 31.02 |
| 3 | constant | cell | 49.361 | 33.654 | 0.956 | 35.21 |
| 3 | tietjen_donor_depth | shared | 33.176 | 20.631 | 0.826 | 24.99 |
| 3 | tietjen_donor_depth | cell | 53.643 | 47.800 | 1.039 | 46.02 |
| 5 | constant | shared | 52.562 | 40.948 | 1.698 | 24.11 |
| 5 | constant | cell | 77.516 | 64.784 | 1.545 | 41.93 |
| 5 | tietjen_donor_depth | shared | 50.640 | 42.055 | 1.278 | 32.91 |
| 5 | tietjen_donor_depth | cell | 83.553 | 75.507 | 4.117 | 18.34 |

The 18–46 times ratio applies only to these fixed-profile component workloads. All outputs were checked against NumPy under the existing numerical contract. The step kernel used96registers,7168Bstatic shared memory and reported0Blocal memory on this device; the finalize kernel uses14336Bstatic shared memory. No whole-storm, Fortran, moving-MAPLE-bed, large-domain or field validation is inferred.
