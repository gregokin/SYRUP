# Phase 2 S2-A4.1 — shared CUDA physical body (common behaviour unchanged)

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_cuda_shared_body_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: author-implemented candidate, **uncommitted and not independently accepted**; author ran **no GPU, CuPy or NVRTC**. First slice of the staged A4 design: only
`src/maple_syrup/subsurface_cuda.py` changed. No spatial CUDA launch, reduction, state or API exists yet; no capability was widened.

## What changed

The common CUDA source is now generated: `_SOURCE = render_source("common")` =
`relax_factor` (shared device function) + the hoisted factor kernel + the common reader macros + the common step head + `_BODY` (the ONE physical stage text).

* `_BODY` is the old step kernel after its lane index, with every parameter read, the invalid-lane guard and the row epilogue behind macros
  (`BODY_MACROS`: `GUARD_INVALID_LANE DZ RES_M FC_M SAT_M CAP_TOL PAIR FAC FAC_UP RES_THETA0 STORE_ROWS`).
* The common reader macros reproduce the old reads exactly (`DZ(l)=dz[l]`, `SAT_M(l)=sat_m[l]`, `FAC(k)=factor[k]`, …, `GUARD_INVALID_LANE = if (i >= n) return;`, `STORE_ROWS` = the four `rows[...]` stores). The kernel
  names, argument lists, launch order, parameter/factor-scratch ledger, `StepArrays.scratch`, preflight, compile options and the CuPy `_read_packet` reduction are untouched.
* `relax_factor(g, s, dt) = { rate = g*s; x = rate*dt; return -expm1(-x); }` is shared by the factor kernel's downward and upward factor (same operations, same order as before).
* `register_variant(name, readers, step_head)` and `render_source(variant)` are the hooks for the later spatial variant. They are placeholders: nothing registers a variant, `render_source("spatial")` is a typed
  `SubsurfaceError`, registration requires every body macro and never replaces `common`, and a variant is rendered with the same `_BODY` object (no second stage text).
* `kernel_provenance()` keeps every old key/value except `source_sha256` (changed openly) and adds `source_variant`, `shared_body_sha256`, `common_readers_sha256`, `source_structure`.

## Source consumers audited

Importers of `subsurface_cuda`: `subsurface.py` (`require_cuda`, `current_device_id`, `default_device_available_bytes_fn`, `prepare_runner`), `layered_case_cuda.py`, `native_dense_cuda_profile.py`
(`require_cuda`, `current_device_id`, `default_device_available_bytes_fn`), `surface_routing.py`, `tests/subsurface/test_cuda.py` (`KERNEL_NAME`, `FACTOR_KERNEL_NAME`, `LAUNCHES_PER_STEP`) and
`tests/phase15_recharge/test_daily_cuda_plan.py` (existence and hashing of the file, no pinned value). None reads `_SOURCE`, `kernel_source` or `kernel_provenance`; `docs/**/phase15_source_manifest.json` pins the historical
file hash of the OLD packages, which are untouched. No guard was disabled.

## Checks (author, CPU only; `tests/phase2_cuda_shared_body/`, 30 tests)

* The rendered common source and the pre-A4.1 source (task-evidence copy, byte-identical to the frozen A3 file) are compiled by `g++` against CUDA-keyword stand-ins (`-ffp-contract=off`) and EXECUTED for every cell plus padded
  lanes: factor kernel and step kernel are bit-identical (NaN sign bits aside) for L = 1/2/3/5, both diffusion modes, both bottoms, gravity on/off, share/none, ordinary/pulse, dt ∈ {0, 1e-6, 3600, 86400, 1e7}, inactive cells and flagged invalid inputs.
* `g++ -E -P` token streams of the macro-expanded step kernels are identical; the factor kernel differs only by the shared relaxation function.
* AST identity of every untouched Python function; constants, `__all__` (superset), provenance values; cold import touches no CuPy and compiles nothing; `load_kernels` through a CuPy plumbing double compiles
  `render_source("common")` once per device with the unchanged options/backend; the common runner launches the same arguments/allocations and reports the same ledger; each stage text and the exchange expression occur exactly once;
  no raw reader or `rows[` remains in the body; hook refusals.

Limits: host execution of the C text is not device execution (no `fmad`, device `expm1`, occupancy, NVRTC or register claim); Root's actual-GPU comparison with the frozen A3 kernel is required. No performance or compile-time claim is made.

## Independent common-path qualification

Root accepted the common-source refactor at package `610ca226b067e74d8ca86b1e41c87dc1d84574bcd1e106b84979b8043bdda500` (118 files): 30 host checks and 23 actual GPU checks passed, one skipped, and 256 arrays across 32 common-column cases match frozen A3 GPU results bit-for-bit. The assigned physical GPU0 was checked unallocated and idle immediately before the child launch. Actual historical MAPLE9f CUDA materialization and the existing MAPLE environments were used. See [phase2_cuda_shared_body_acceptance.json](phase2_cuda_shared_body_acceptance.json) for exact identities and limits. This accepts only the common-source refactor; spatial GPU evolution and the whole Phase2 remain unaccepted.
