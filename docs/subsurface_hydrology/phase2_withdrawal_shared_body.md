# Phase 2 S2-B1.a — common withdrawal physical bodies behind parameter readers

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_withdrawal_shared_body_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: author-implemented, **uncommitted, not independently accepted**, CPU only (no GPU, CuPy or NVRTC used). Only `src/maple_syrup/subsurface_transpiration.py` changed. No spatial operator, API or capability exists; no physics changed; three user physics questions unanswered.

The common withdrawal law is unchanged: `available = max(W − residual, 0)`, `actual = min(requested, available)`, `W' = W − actual`, `unmet = requested − actual`, `scale = actual/requested` (0 where the request is 0; dimensionless). The residual is a withdrawal floor, not a minimum inventory (zero or below-residual water is legal); no clock advance, drainage, evaporation, demand calculation, root redistribution or flag-aggregation change.

## What changed

* **NumPy (`_fused_numpy`)**: the two layer parameters are read only as `residual[l]` / `satb[l]` inside per-layer loops (arithmetic outside the former `errstate` block, flags inside it, as before). Every element operation is the former IEEE operation; every reduction (`sum/min/max(axis=0)`, flag `any`) is untouched and still over the whole `(L, N)` arrays, so sums, reduction order and signed zeros are those of before. The `(L, N)` `avail` and `bad_w` temporaries became `(N,)` per layer: the traced peak fell (e.g. L = 9, N = 200 000: 23.5 MB → 9.3 MB) and the unchanged byte model (6 `(L, N)` temporaries reserved) is therefore never undercounted.
* **Numba (`_fused_numba`)**: the same compiled serial body reads the parameters through two inline readers `_read_residual(residual, l, j)` / `_read_satb(satb, l, j)` (here `residual[l]`, `satb[l]`); no local array, no copy, `fastmath` False, still `cache=True` and a typed lazy missing-numba path. A later spatial variant can supply other readers of the same body.
* **CUDA source**: the physical text is `_CUDA_BODY`; the lane guard, the two parameter reads and the row stores are the macros `WITHDRAW_GUARD`, `WITHDRAW_RESIDUAL(l)`, `WITHDRAW_SATB(l)`, `WITHDRAW_STORE(...)` (`WITHDRAW_MACROS`). The common definitions in `_CUDA_COMMON_READERS` expand to the previous text, so the macro-expanded kernel is token-identical to the pre-change kernel. `render_withdrawal_source(readers, kernel_head, tail="")` is a pure function (no registry/cache/state; typed refusal of incomplete readers); `_CUDA_SRC` is its common rendering. Kernel name, `-std=c++11` compile option, launch arguments, rows, CuPy packet reductions and the native packet are unchanged. The common source hash changes openly.
* Python surface: every function except `_fused_numpy` is AST-identical; the only new function is `render_withdrawal_source`; `estimate_transpiration_memory`, admission, guards (`ss._check_column`, `_check_state`, `_check_dynamic`), operator identity/namespace/device checks are untouched.

## Memory alias proof

The operator still owns exactly two `(L,)` vectors (`residual_xp`, `saturation_xp`); the body receives those same objects (tested by a spy on `_launch`), the Numba body allocates nothing (AST check), and the NumPy body allocates strictly less than before. `estimate_transpiration_memory` is byte-identical to the pre-change function for every backend tested.

## Checks (CPU)

`tests/phase2_withdrawal_shared_body/` (15 tests): NumPy and Numba bodies bit-identical on 112 scenarios (L = 1…17, N = 1…64, signed zeros, zero floor, masks, all-inactive, flagged invalid inputs) and the public operator results/budgets/memory model identical; token-identical preprocessed CUDA text; host-compiled (g++ `-ffp-contract=off`, CUDA keywords stubbed, padded lanes) execution bit-identical; pure/complete rendering, body once, no raw reads left in the body; AST surface; CUDA prepare/launch flags, name and argument count through a CuPy plumbing double (no GPU); genuine nopython Numba without fastmath; alias proof. The only tolerated difference is the sign/payload bit of a NaN from an invalid input.

Timings (single thread, interleaved warmed medians; `s2_b1_a_numba_ab.json`, `s2_b1_a_timings.json`): Numba new/old 0.993–0.998 over L = 3/5/9, N = 10 000/200 000 (600/80 repeats); NumPy 0.96–1.02; cold Numba compile (empty cache) 0.78 s → 0.82 s. No consistent regression; no storm or scaling claim.

Limits: host execution of the C text is not device execution; NVRTC acceptance of the macro form, device arithmetic and the actual-GPU 48-case comparison are Root's. No spatial reader, operator or CUDA capability was added.

## Independent common-path qualification

Root accepted the common-reader refactor at package `649b18e4631b29680741f9c96293bdd20464cff5205ad1d1a4ac44b1fd4a5458` (119 files): 29 focused CPU tests passed, two skipped; NumPy and Numba each match all 336 frozen individual-column reference evaluations bit-for-bit, and 192 actual GPU arrays match the previous CUDA execution exactly. See [phase2_withdrawal_shared_body_acceptance.json](phase2_withdrawal_shared_body_acceptance.json) for exact sources, commands and limits. This does not yet add spatial withdrawal operators or native LEAF/lifecycle acceptance.
