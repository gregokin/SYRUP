# Phase 2 S2-B2.d — fixed explicit spatial preferential (bypass) infiltration, CUDA (trial component)

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_spatial_preferential_cuda_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: author-written WITHOUT any GPU (no CuPy initialisation, NVRTC or device probe); verified only by pure text checks, host-compiled execution of the rendered source and plumbing doubles. These are NOT hardware evidence; Root qualifies the CUDA path on GPU0 separately. Uncommitted, pure-trial only: no accepted generation, exactly-once, LEAF, lifecycle, restart or storm claim; species cover and receiving weights stay prescribed inputs.

## What changed (four production files)
* `preferential_infiltration_cuda.py`: the kernel text is rendered from shared pieces by the pure, fail-closed `render_source` (declared `@@TOKEN@@` sites: lane guard, `sat_m[l]+cap_tol[l]` ×2, `sat_m[l]`, `rate[i]`, `max_rate[i]`, the stores). The COMMON rendering is byte-identical to the previous text (3 457 characters, sha256 `0d454f1ad0fc4c6d23252687c311802788e3ee2a411f714969a2a08e9c9f72e2`, checked at import and by tests); name, ABI, options and `kernel_provenance()` are unchanged.
* `preferential_infiltration.py`: ONE optional trailing field `reduce` of the private `_DeriveContext` plus the small fail-closed helper `_reduce` (kinds `count_nonfinite`, `min`, `max`, `count_nan`; an unknown kind raises). The derivation's eight cold reductions call `_reduce`. With `reduce=None` (every previous caller) they are exactly the previous expressions, so outputs, errors and allocations are unchanged.
* NEW `spatial_preferential_infiltration_cuda.py`: spatial readers (`sat = theta_sat*dz`, `sat + coef*sat`, strided raw maps), the `__device__` head, fused step/finalize kernels, the bounded cold reducer and validation kernels, launchers, original-shape staged hashing, `operation_terms`.
* `spatial_preferential_infiltration.py`: the same public functions accept a CUDA owner with device quota/reserve/reader plus the separate host ones (refused for CPU owners); CPU interfaces, identity recipe, numerics and quotas are unchanged.

## Law (unchanged, ONE shared body)
ONE aggregate continuous hourly-fraction path `H(−expm1(−k dt))` with `k = −log1p(−beta)/3600` (`+inf` for `beta = 1`), ONE aggregate cap `max_rate dt`, allocation TOP LAYER FIRST bounded by each receiver's saturation room (`sat − W`) and the remaining surface, no remapping of a rejected share, no history effect, `state.t_s` preserved, `dt = 0` the identity. Receiving weights, cover and a3 stay prescribed.

## Hot step
Fused fixed tree: seven sums (soil before/after, surface in/out, isolated, attempted, intake) + flag OR + flagged count in `7·nb` partials, one finalize block, ONE 72-byte raw `uint64` packet (9 words) read with `to_host`, rebuilt into the unchanged 9-float packet for `pi._resolve_core`; raw (72 B) + rebuilt (72 B) host buffers = 144 B counted. No dense `8 N` rows, no CuPy/CUB workspace, no L-sized or per-thread arrays, no host mirror, zero H2D. Every lane reaches every barrier.

## Cold derivation
`_derive_pathway` (reused, not copied) runs on the owner's device (`ctx.xp = cupy`; only elementwise operations) and reaches its reductions through the bounded counted `ColdReducer`: one pass kernel keeps a 32-byte `uint64` diagnostic `[#non-finite, #NaN, min-key, max-key]` (order-preserving integer keys, `atomicMin/Max`); NaN propagates, ±inf and the sign of zero follow the key order. The 32-byte seed is a COLD host-to-device copy and the 32-byte result a COLD synchronising read (host buffers 64 B); a species costs 2 calls (non-finite, min; disabled species included), then cover 1, closed form 1, beta 2, one per receiver layer, rate 1: `2S + L + 5`. No library `sum/min/max`.

## Memory model (payload; not a reservation; NVRTC/context/pool rounding excluded)
* step device `(2N + 2LN)·8` + `7·8·nb` partials + 72 raw packet (+ reserve); host 144 + 65 536 allowance (+ host reserve). Static shared (not global): step `7·8·128 + 8·128 + 4·128 = 8 704` B, finalize `7·8·256 = 14 336` B.
* prepare device `(LN + 6N)·8 + 3N + 64` (+ reserve): retained `(LN + 2N)·8` (rate, receiver, max rate upper bound) + the derivation's live set (`cover`, `beta`, `denominator`, `layer_sum`, masks `inactive`/`positive`/`~positive`, the 32-byte diagnostic and the one-element scalar arrays); host: weight records `2(S+1)L·8` + the record under construction `2L·8` + reducer buffers 64 + hash staging `min(LN, 16 384)·8` + allowance (+ host reserve). Species contributions are caller-owned, streamed one `(N,)` array at a time, never an `S×N` stack.
* Cold identity: the CPU canonical recipe (original-shape header + flat bytes; weights via the unchanged 4096-byte hex stream) streamed from the device through the counted staging buffer; no per-step hashing.

## Limits
Fixed explicit geometry/properties; no policy, provider, native, B3, moving-bed or LEAF work. Kernel compile/load at preparation (function cache per device only). Device `expm1`/`log1p` may differ from NumPy by an ulp (declared rtol 2e-12 / atol 1e-14; bitwise where none is evaluated). Register/occupancy/timing and actual allocator behaviour are unmeasured here: Root's.


## Independent Root qualification — 2026-11-13

Accepted bounded fixed-spatial preferential CUDA component; see [phase2_spatial_preferential_cuda_acceptance.json](phase2_spatial_preferential_cuda_acceptance.json) for exact sources and limits. Root ran 271 source/host-execution checks and 454 actual GPU checks. Each of NumPy, Numba and CUDA matched all four arrays in 84 independent original-reference scenarios (588 columns) bit for bit on these controls; 704 saved common-GPU arrays were unchanged. At L=4,N=50003 the requested device payload was 4,022,208 bytes, exactly the model; the hot call read one72-byte packet and uploaded nothing. Pool use plateaued for eight identity calls, owners were released, and the step kernel used56 registers/no local spills. These are component checks, not complete-storm/native-bed/field/large-domain acceptance. Cold-reducer host64-byte seed/result figures describe array payload; a decoded scalar temporary is also covered by the fixed allowance.
