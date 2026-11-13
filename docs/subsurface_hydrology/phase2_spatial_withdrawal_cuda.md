# Phase 2 S2-B1.c — fixed-spatial CUDA withdrawal and availability trial

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_spatial_withdrawal_cuda_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: author-implemented on a **CPU-only host** (no GPU, CuPy or NVRTC used by the author), **uncommitted**. This source asserts no qualification; qualification of a specific source is recorded only in Root's source-specific receipts. A pure trial component: no exactly-once adoption, stale-proposal, LEAF, lifecycle, storm, moving-soil, root-remap, PTF or restart claim; three user physics questions unanswered. The CPU (NumPy/Numba) methods, numerics, quotas and the typed overflow refusal of B1.b are unchanged.

## API (the SAME public functions)

`prepare_spatial_withdrawal`, `spatial_layer_availability_m`, `spatial_withdraw`, `estimate_spatial_withdrawal_memory(..., backend="cuda", host_reserve_bytes=0)` now accept a real `cuda` `SpatialSoilColumn`/`SpatialSoilState`. Strict checks BEFORE any copy, launch or allocation: exact `cupy.ndarray` float64 C-contiguous shape/device for the state and the request, the current device equal to the owner's, the parent-input-SHA state binding of the CPU path (no owner-object identity, no water generation), no dtype coercion, no namespace guess, no fallback. For a CUDA owner `memory_budget_bytes`/`reserve_bytes`/`available_bytes_fn` are the DEVICE quota/reserve/reading (existing `sp._admit_cuda_operation`) and the new keyword-only `host_memory_budget_bytes`, `host_reserve_bytes`, `host_available_bytes_fn` the separate host ones; CPU owners refuse them (same wording as the spatial column helpers). Results stay on the device (`state` is a CUDA `SpatialSoilState` with the same clock; `actual_m`, `unmet_m`, dimensionless `scale` are CuPy arrays); one constant-size packet is read back; no host copy or bed update. Availability is a fresh `(L, N)` device array (inactive cells 0, every cell's water validated, one 8-byte diagnostic read) and is NOT a generation receipt.

## One physical body, no fusion

`subsurface_transpiration._CUDA_BODY` (byte for byte; the common CUDA text `4bc236cf…` and its `-std=c++11` options are untouched and not broadened) is rendered by the pure `render_withdrawal_source` with spatial readers and a reduction tail in the new `spatial_transpiration_cuda.py`. The body becomes the `__device__` function `spatial_withdraw_cell` (its early `return` is the padded-lane guard); the wrapper kernel lets every lane reach every barrier.

* Raw readers (A1 strides, scalar 0/0, shared 1/0, cell N/1): `WITHDRAW_RESIDUAL = res_at(θres, dz)` (`res = θres·dz`), `WITHDRAW_SATB = satb_at(θsat, dz, coef)` (`sat = θsat·dz`, `satb = sat + coef·sat`). **These products must be rounded before the next operation, exactly as on the CPU**; a contracted FMA (`wv − θres·dz`, `sat + coef·sat`) changes bits (a host test compiles the same text with contraction allowed and shows differing outputs). The NEW module is therefore compiled with `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`; the common kernel keeps its precomputed bounds and its own options.
* Per lane: the eight per-cell quantities of the body in registers (`contrib[8]`: five active-gated sums, minimum water after, the flag word as a double, request maximum; neutral for a padded lane: 0,0,0,0,0,+inf,0,0), a fixed block tree in static shared memory (`8·8·128` B; sum rows 0–4, min row 5, max rows 6–7), `8·nb` block partials, a one-block finalize, ONE `maple.core.backend.to_host` of the 8-double packet (64 B) in the common packet order. No CuPy/CUB reduction, no `8N` rows array, no `(L,N)` bound tensor, no host mirror, zero H2D for resident inputs. The flag semantics are the common ones (per-cell OR → `(double)flags`, maximum over cells), so reason texts are identical to the CPU path including the preserved aggregation limitation.
* Totals are summed in a fixed tree order (roundoff-level different from NumPy/Numba/CuPy sums); arrays are bit-identical to the CPU path (the law has no libm term). Budget closure uses the same `volume_roundoff_bound_m3` coefficient, `n_terms = 3·L·n_active + 8` and area applied once; unrepresentable global volumes or scale are a typed `SubsurfaceError` BEFORE the conservation helper (shared `_publish`).

## Memory (payload model; `nb = ceil(N/128)`)

| op | device (main quota) | host (separate quota) |
|---|---|---|
| prepare | 0 (module load only; no dummy launch or copy) | 64 KiB Python allowance |
| withdraw | outputs `4·L·N·8` + scratch `8·8·nb` partials + `64` device packet | `64` (the ONE packet array `to_host` returns; no raw/reconstructed pair) + allowance |
| availability | outputs `L·N·8` + scratch `8` | `8` + allowance |

Plus the device/host reserves. Caller state/request and the owner's retained maps/mask are excluded; static shared memory (`8192 B` per block) is reported; NVRTC, the context, the pool policy and rounding and the interpreter are excluded; estimates are payload models, not reservations, and no pool-growth lower bound is assumed (reused split chunks). Admission precedes every allocation, conversion, compile and launch; scratch (partials, packet, diagnostic) is held until after the synchronising read and then released to the pool; launches run on the current stream; OOM → `SpatialLayerMemoryError`, CUDA runtime/NVRTC/launch errors → `SpatialLayerBackendUnavailableError` (no NumPy fallback; context recovery not claimed). Functions/modules are cached per device only, compiled at preparation (not at the first call, no physical dummy copy). CPU imports stay lazy (no CuPy probe).

## Checks (author, CPU host; NO hardware)

`tests/phase2_spatial_withdrawal_cuda/` (78 passed, 36 skipped): host source/render checks (common source/options byte-exact, one shared body, no second copy of the law, wrappers reach every barrier, no library reduction, cold import); host-compiled execution of the real rendered text with real threads and barrier through the real launchers/API on a CuPy plumbing double, bit-equal to the CPU spatial path for N = 1/7/127/128/129/257, L = 1/2/3/5, all water/request regimes, masks and all-inactive domains, scalar/shared/cell layouts, the dyadic example, availability, invalid inputs on every cell with identical reason text and no mutation, typed global overflow, the FMA-hazard demonstration; plumbing tests (allocation list equals the model, one 64-byte read and no upload, no dummy launch at preparation, admission on both sides before any allocation or compile, device/namespace/type refusals, scratch lifetime, typed OOM, release). `test_cuda_definitions.py` holds the actual-GPU definitions (36 cases) WRITTEN BUT NOT RUN. The earlier cuda-tagged-owner refusal test of B1.b became an honest guard-path/non-hardware test. Limits: no device compile, register/spill/shared-memory, occupancy, pool or performance measurement; no timing claim.


## Independent hardware qualification

Root accepted the fixed explicit-property trial at package `94be72e37e60f4030206704f62ffdec69f5e313248401058bc879b3b7c0f4961` (121 files). See [phase2_spatial_withdrawal_cuda_acceptance.json](phase2_spatial_withdrawal_cuda_acceptance.json). Independent verification passed 232 CPU checks and the 114-check GPU-enabled suite (36 additional actual-device cases). Each backend matched all saved 336-column reference arrays exactly. Hardware was the authorized GPU0, NVIDIA GeForce GTX 1080 Ti. Withdrawal used one 64-byte device-to-host diagnostic; availability used 8 bytes, both without uploads once resident. The withdrawal kernel used 62 registers, 8192 B static shared memory and zero local bytes; its final reduction used 16384 B shared memory.

Matched fixed-property wrapper timings (identical geometry, soil, masks, water and requests; compilation/setup separate):

| Layers | Columns | NumPy (ms) | Numba (ms) | CUDA (ms) | Numba/CUDA |
|---|---:|---:|---:|---:|---:|
| 3 | 10000 | 0.690 | 0.401 | 0.434 | 0.92 |
| 3 | 200000 | 16.020 | 7.985 | 0.559 | 14.29 |
| 5 | 10000 | 1.009 | 0.533 | 0.442 | 1.21 |
| 5 | 200000 | 24.776 | 11.756 | 0.634 | 18.53 |

These measurements qualify this component and workload; they are not full-storm timing, observational validation, restart, LEAF, moving-bed or large-domain acceptance.
