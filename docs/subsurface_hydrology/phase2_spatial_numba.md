# Phase 2 S2-A.3 — compiled Numba CPU evolution of explicit spatial soil parameters

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_spatial_numba_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: author-implemented candidate, **uncommitted and not independently accepted**. Root freezes and reviews it. This note describes what the
code does and what was run by its author; it claims nothing beyond that.

## Scope

A `SpatialSoilColumn` prepared with `backend="numba"` (S2-A.1) now genuinely evolves water with a compiled CPU kernel, through the same explicit
entry points as the NumPy path of S2-A.2: `spatial_column_state`, `spatial_uniform_state`, `spatial_moisture`, `spatial_storage_m3`,
`spatial_step`, `spatial_step_with_recharge`. Geometry and properties are the fixed explicit inputs the caller prepared. Nothing here chooses a layer
motion frame, a material-water fate, a texture-to-property method, a conductivity averaging rule, a wetting-history reset or a calendar rule (the
three user physics questions remain pending). There is no evolving bed, storm, restart, LEAF/root coupling or source-lock rebinding. A `cuda` owner
is still refused with a typed `SpatialLayerBackendUnavailableError` before any copy or kernel (S2-A.4); there is never a fallback between backends.

## ONE physical body, two compile-time variants

The statements of the step are written once, in `subsurface_numba._build(numba, common)`, and compiled twice (Numba prunes the branches on the closure
constant `common` before typing):

| variant | accessor | caller | parameter source | per-call Numba heap |
|---|---|---|---|---|
| common | `compiled_kernel()` | `prepare_runner(layers)` for a common-layer column (public API unchanged) | the cached `(L,)` metre arrays of `KernelParameters`, read as they are; the bound "locals" ALIAS them | exactly the one (constant) / two (donor) per-interface factor arrays of the pre-A3 kernel = the frozen estimate's `factor_scratch_doubles` |
| spatial | `compiled_spatial_kernel()` | `spatial_launch(...)` for the explicit spatial owner | the owner's retained RAW maps `θres, θfc, θsat, dz, 1/pair, G_down, G_up` as flat read-only views | `5L` bound doubles + `3·max(L−1,1)` interface doubles (see Memory) |

Every map is read as `a[k * stride_layer + i * stride_cell]`. The two integer strides are resolved once per call from the layout kind:

| kind | meaning | stride_layer | stride_cell |
|---|---|---|---|
| 0 scalar | one value | 0 | 0 |
| 1 shared | one value per layer/interface | 1 | 0 |
| 2 cell | C-contiguous `(rows, N)` map | N | 1 |

(An earlier draft branched on the kind inside the element read; per-cell layouts then ran ≈2.4× slower than with the strides because the branch kept
the compiler from optimising the cell loop. Another draft gave the common path the same O(L) bound locals as the spatial path; it was replaced by the
aliasing common variant because those locals exceeded the frozen common memory model for tall stacks with few cells and cost the common path 2–4 %.)

Spatial variant: everything that does not depend on the cell is computed ONCE per call into local arrays before the cell loop — the metre bounds
`res = θres·dz`, `fc = θfc·dz`, `sat = θsat·dz` and `cap_tol = coef·sat` (`coef = volume_roundoff_bound_m3(LAYER_OPERATION_TERMS, 1.0)`, the same IEEE
products as `subsurface.soil_layers` and the NumPy spatial reader), `1/pair`, and the per-interface factor(s) `−expm1(−((G·s)·dt))` (downward and, in the
donor-depth mode, upward). Only maps whose layout is `cell` are re-bound inside the cell loop, with the same expressions, so a per-cell conductance pays one
`expm1` per interface per cell, which is physically required. The common variant keeps the shared per-interface factor hoisting of the pre-A3 kernel. The stage
order, flags, bounds, defaults, the post-cap evaporation share, the closed/open bottom, both diffusion modes and the pulse fit test are the statements of the
previous kernel; they read `sat_c[l]`, `res_c[l]`, … from the bound locals (spatial) or from the cached arrays (common).

There is no `fastmath`, no `prange`, no Python per-cell loop, no per-step expanded `(L, N)` parameter tensor, no copy of the owner's maps or mask and no NumPy
fallback.

### Compile-time proof (tests; both variants)

* `CPUDispatcher`, `nopython=True`, `fastmath=False`, `error_model="numpy"`, `boundscheck=False`; exactly one signature, compiled in nopython mode; the
  compile result is not object mode and has no lifted loops; the LLVM IR contains `expm1` and no fast-math flag (`fast`, `reassoc`, `nnan`, `ninf`,
  `nsz`, `arcp`, `contract`, `afn`).
* The single explicit signature types the 13 input arrays (state, forcing, mask, parameter maps, share) as READ-ONLY and the 8 fresh outputs as
  writable, so Numba rejects at compile time any statement that writes an input; writable and read-only caller arrays both use the one specialisation.
* `ss._reference_step_body` and `_SpatialStepParams` are monkey-patched to raise: numba steps still run; a failing kernel build is a typed error, not a
  NumPy call.

## Memory

Admission precedes every allocation, exactly as in S2-A.2 (`memory_budget_bytes`, optional `reserve_bytes` and `available_bytes_fn`; no quota is raised
silently). `estimate_spatial_operation_memory(op, L, N, ..., backend="numba")` changes only the two step operations:

```
outputs  = (L·N + 2(L−1)·N + 4·N + ROW_COUNT·N) · 8                      (fresh arrays, identical to the NumPy path)
scratch  = N                                                             (the packet's `rows[3] < NO_FLAG` bool row)
         + 8 · (5L + 3·max(L−1, 1))                                      (spatial kernel locals, Numba NRT heap, independent of N)
         + 4 · PACKET_LENGTH · 8                                         (packet scratch, 288 B)
         + 65 536                                                        (Python allowance)
inputs excluded: state (LN·8), added water (LN·8; none for the pulse step), potential rate (N·8), optional share (N·8)
```

The NumPy path keeps its `(16 + 8)·N·8 + 8N + packet + 64 KiB` scratch. The retained maps and the mask are not re-counted and never copied (flat views;
the mask is a `uint8` view of the owner's bool mask). The kernel's never-read pointers (`added` for the pulse step, `share` without a share) alias
arrays that already exist. Numba NRT allocations are invisible to `tracemalloc`; they are bounded by the declared local-array term and verified by
allocation counting (below). The non-step operations are host NumPy operations and are estimated identically for both backends.

Common column: the common variant allocates exactly the per-interface factor arrays that the frozen `estimate_step_memory` counts (`factor_scratch_doubles`);
`backend_info["kernel_local_scratch_host_bytes"]` equals `factor_scratch_host_bytes` for every L (tested up to L = 40 with one cell, where any O(L)
copy would exceed the model). `subsurface.py` is read-only in this slice and was not changed.

## Metadata

`SpatialSoilState.component_tag` is now `"spatial_explicit_cpu_state"` (it was `"spatial_numpy_reference_only"`: a state is backend-neutral host data and
is stepped by whichever CPU backend its owner was prepared for). `summary()["water_evolution"]` reads `evolution_backends = ["numpy", "numba"]`,
`unsupported_backends = ["cuda"]` and lists `explicit_trial_functions` (the key was `explicit_numpy_trial_functions`). The owner's
`hydrology_step_supported = False` still describes the parameter-only owner object, not these module-level functions.

## Numerical contract

* Spatial Numba vs the accepted common Numba column, per column: bit-identical (same body, same libm).
* Spatial Numba vs the explicit NumPy spatial path: the declared Numba bound rtol 2e-12 / atol 1e-14 (only `expm1` may differ by an ulp); bit-identical
  when the diffusivity is zero; identical flag rows, typed error classes, flag bits and flagged counts.
* The new common Numba kernel vs the pre-A3 kernel: bit-identical on every tested finite output. The one tolerated difference is the sign bit of a NaN in
  a flagged, refused cell (e.g. `inf − inf`), which two compilations may order differently.

## Tests (author-written, CPU only, existing MAPLE `.venv`, frozen MAPLE 9f paths)

`tests/phase2_spatial_numba/` (141 tests): compile proof for both variants; per-column bitwise identity to the common Numba column over L = 1/2/3/5, both
modes, both bottoms, gravity on/off, five water states, five dt values including 0, share/no share, ordinary intake vs uniform pulse; all 2 × 27 × 3 = 162
layout combinations per mode; random layouts for L = 1/2/5; NumPy agreement within the bound and bitwise at zero diffusivity; the capacity-tolerance boundary
(`cap_tol = coef·θsat·dz` derived inline); identical typed failures for each flag 0–9, 15, 16 and 17; admission boundary/refusal ordering; warm traced peak
within the modelled requirement; NRT allocation counts (spatial: the 21 argument wrappers + 7 locals, 8 in donor mode; common: wrappers + 1 or 2 factor
arrays for L = 1/2/6/40; equal frees; independent of N); no mutation of the state, forcing or retained maps; owner/state release and no cache holding owners; the
pre-A3 comparison and ledger. A manual mutation pass on a throw-away copy (cell stride, layer stride, tolerance from `dz`, upward factor from the downward
conductance, ignored cell strides, wrong layer-0 residual in either variant, a wrong aliased array in the common variant, skipped pre-bind) was killed by these
tests. One further mutant (`ifaces_cell` ignoring the pair map) is equivalent under the owner's join rule (the conductance maps are per-cell whenever the pair map
is), so the kernel keeps both checks defensively.

## Timings (single thread, warmed medians, small-kernel only)

Numba 0.67.0 / llvmlite 0.49.0. Compile startup, measured in a fresh process: common variant ≈ 2.3 s, spatial variant ≈ 2.2 s (one-off each; at `prepare_column`
/ `prepare_spatial_column(backend="numba")`, never in a step). Warmed `ms/call` (90 % active cells, closed bottom, 3600 s):

| L | cells | common pre-A3 | common new | spatial numba shared | spatial numpy shared | spatial numba per-cell | spatial numpy per-cell |
|---|---|---|---|---|---|---|---|
| 3 | 10 000 | 1.02 | 1.01 | 1.28 | 2.16 | 1.80 | 2.76 |
| 3 | 200 000 | 26.1 | 25.9 | 21.4 | 51.0 | 47.4 | 73.5 |
| 5 | 10 000 | 1.69 | 1.67 | 2.06 | 3.58 | 2.82 | 4.44 |
| 5 | 200 000 | 44.7 | 46.0 | 42.9 | 81.5 | 79.2 | 120.3 |

Interleaved A/B of the common column (400/100/25 alternating calls per size; `s2_a3_common_ab.json`): new/pre-A3 median ratio 0.982–1.038 over nine cases (L = 3/5,
constant and donor mode, 1 000/10 000/200 000 cells); below 200 000 cells it straddles 1 (0.982–1.012), at 200 000 cells with L = 5 it is 1.027 and 1.038. With
25 repeats this does not separate from noise, and no consistent common regression was found. No whole-storm, large-domain, multi-thread or scaling claim is made.

## Limits

CPU only; fixed explicit geometry/properties; no moving bed, remapping, sediment, LEAF, lifecycle, checkpoint or native storm acceptance; no CUDA
evolution (S2-A.4); the S2-A.2 doc's statement that Numba owners refuse evolution is superseded by this note for the Numba backend; the three user
physics decisions are unchanged and unanswered.

## Independent component qualification

Root accepted the fixed explicit NumPy/Numba column component at package `dba2bba1e48a05c701e55432077e347003fbffeba9400ba056c46594cc5aa437` (118 files), using actual MAPLE `9f466d2c` and its locked materialized CPU bundle. See [phase2_spatial_numba_acceptance.json](phase2_spatial_numba_acceptance.json) for exact identities, raw evidence and limits. Independent checks: 1,061 CPU tests passed, 23 skipped; each backend matched 168 reference-column and 1,056 mixed-layout evaluations bit-exactly; 512 common arrays remained exact. The earlier draft's extra common scratch was removed before this tested source.

Independent warmed timings use different forcing from the author table: all cells active, open bottom, shared explicit properties, fixed controlled moisture; at 262,144 cells the validated Numba wrapper was 1.14–1.31 times faster for one/three layers, while NumPy was about 1.18 times faster for five layers. Do not infer uniform acceleration or whole-storm performance. Interleaved common-kernel ratios were 0.967–1.031 versus the accepted old kernel. GPU evolution, actual moving-bed integration and the whole Phase 2 remain unaccepted.
