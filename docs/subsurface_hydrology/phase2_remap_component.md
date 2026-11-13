# Phase 2 S1: conservative interval remap of piecewise-uniform water (component, trial only)

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_remap_component_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: **new utility modules and tests, uncommitted; component/trial only.** This is a pure mathematical primitive. It adopts no accepted state, reads no clock or
ledger, binds no parent, selects no cells and carries no restart, storm, integration, performance or CUDA-hardware qualification claim. It does **not** choose the erosion/deposition
order, the layer frame (fixed bottom vs constant thicknesses), where removed water goes, the moisture of entering sediment, groundwater, capillary rules, saturation
redistribution or any wetting-history/provider rule. Those user questions are pending (`agent_handoffs/tasks/phase2_evolving_soil_20261112/pending_physics_questions.json`); a later
caller composes ordered native exchanges and supplies the geometry and the incoming water.

Files: `src/maple_syrup/soil_water_remap.py` (API, validation, NumPy reference, memory model), `soil_water_remap_numba.py` (compiled CPU), `soil_water_remap_cuda.py` (CuPy RawKernel,
written not run by its author), `tests/phase2_soil_remap/`.

## Signature

```python
remap_intervals(old_edges_m, new_edges_m, water_m, *, upper_theta, lower_theta, cell_area_m2, backend="numpy",
                memory_budget_bytes, reserve_bytes=0, available_bytes_fn=None,
                host_memory_budget_bytes=None, host_reserve_bytes=0, host_available_bytes_fn=None) -> RemapResult
```

`memory_budget_bytes`, `reserve_bytes`, `available_bytes_fn` apply to the admitted side of the backend (host for "numpy"/"numba", **device** for "cuda"). The three `host_*` fields (correction01) apply **only** to
"cuda" (they are refused for the CPU backends): the CUDA call also keeps a small host packet, so the host side is admitted separately and before any allocation against `host_memory_budget_bytes`
(None = no separate quota), `host_reserve_bytes` and a truthful host reading (`host_available_bytes_fn`, default the `sysconf` reading; never a fabricated lambda). A refusal names the failing side.

`L >= 1` intervals, `C >= 0` selected columns (the caller gathers active changed columns; the utility has no authority to select cells and never sees the whole domain or a deep bed).
All arrays float64, C-contiguous, exactly of the backend's namespace (NumPy for "numpy"/"numba"; CuPy on the current device for "cuda"; no subclass, no implicit transfer):
`old_edges_m`, `new_edges_m` `(L+1, C)`; `water_m` `(L, C)`; `upper_theta`, `lower_theta` `(C,)`; `cell_area_m2` a finite positive host scalar.
Returns a fresh `RemapResult`: `water_m (L,C)` and the per-column `removed_above_m`, `removed_below_m`, `added_above_m`, `added_below_m` `(C,)`, plus `closure`, `memory`, `label="component_trial_only"`.

## Mathematics

One depth coordinate per column, increasing downward, the same reference for old and new edges (negative values allowed; the new top may differ from the old top, the new bottom from
the old bottom; the two extents may be disjoint). Old interval `i` has width `w_i`, water `W_i` (m of water), volumetric fraction `theta_i = W_i / w_i`, required `0 <= W_i <= w_i`
(no hydraulic residual or other floor is applied or implied; dry material is allowed). `upper_theta`/`lower_theta` in `[0,1]` are explicit (no defaults): the fraction of material entering above the old top / below the old bottom.

```
N_j = sum_i theta_i |old_i ∩ new_j| + upper_theta |new_j ∩ (-inf, eo_0)| + lower_theta |new_j ∩ (eo_L, +inf)|
removed_above = old water above the NEW top      removed_below = old water below the NEW bottom
added_above   = upper_theta * (new span above the OLD top)       added_below = lower_theta * (new span below the OLD bottom)
closure:  sum_j N_j - sum_i W_i = added_above + added_below - removed_above - removed_below
```

The four fluxes are **not** routed, removed or classified as loss/drainage here; there is no saturation or property clipping, no water redistribution, no `dt`. Zero geometry change (numerically equal edges in a column) is an **identity copy**: every water value is returned bit for bit,
including `-0.0` and subnormals, with all fluxes exactly `+0` (the general path would turn `-0.0` into `+0.0`; the identity branch keeps the contract without relaxing it). Zero **net** elevation
change is not classified: if the caller's actual exchange replaced material, the caller must state the geometry and incoming water explicitly.
The exact arithmetic order (ascending sums, `min`/`max`, products) is the definition in the module docstring; Numba (`fastmath=False`) and CUDA (`--fmad=false`, no fast math) repeat it term for term.

## Checks, errors, bounds

Metadata before any allocation (backend, exact types/namespace/device, float64, C-contiguity, shapes, scalars, integer budget/reserve), then the memory admission, then value flags
after the computation and before the result exists: 1 old edges, 2 new edges (not finite, not strictly increasing, or an unrepresentable width), 4 water outside `[0, width]` or not finite,
8 theta outside `[0,1]` or not finite, 16 an unrepresentable output or a new interval above its width plus its derived bound, 32 per-column closure above its bound; then the global closure.
Errors: `SoilRemapError` (base, `ValueError`), `SoilRemapValidationError` / `SoilRemapClosureError` (carry `bit`, `n_flagged`, `first_column`; `bit` is the lowest flag of the first flagged
column), `SoilRemapMemoryError`, `SoilRemapBackendUnavailableError` (Numba or CuPy missing: no fallback). Inputs are never written; failed calls return nothing.

Every derived quantity is checked finite: a non-finite column scale/bound (never accepted by a vacuous `<= inf` comparison), a non-finite total, residual or global scale (for example a finite but huge `cell_area_m2`) is a
typed `SoilRemapClosureError` (`bit = FLAG_CLOSURE`), never a raw `ValueError`; inputs are untouched. Bounds use MAPLE's existing coefficient `conservation.volume_roundoff_bound_m3(n, 1.0)` (= `summation_error_bound_kg(n, 1.0)`), no new epsilon. Column closure: `n_col = 2L^2 + 9L + 8`
(per interval `L` products + `L+2` adds + 2 products `= 2L+4`, times `L`; `L` old-total terms; `2L` in each of the two removed sums; 4 added/removed terms; 4 final combinations), scale
`S_c = sum_i w_i + sum_j wn_j`. Output sanity: `coef(2L+4) * wn_j` above a new width is tolerated and **not clipped** (the computed value is returned). Global (m3): `n = C n_col + 8`, scale `A * sum_c S_c`.
Examples of the term counts: `L=1: 19`, `L=2: 34`, `L=3: 53`, `L=5: 103`.

## Memory (modelled, not measured resident memory)

`estimate_remap_memory(L, C, backend=, reserve_bytes=, host_reserve_bytes=)` returns `required_bytes` (main admitted side), `host_required_bytes`, `device_required_bytes`. Excluded and reported as
`input_bytes_excluded`: the caller's five arrays `8((L+1)C*2 + LC + 2C)` B. Retained outputs `8LC + 32C`.

**CPU backends (host):** `(5,C)` stats `40C` + `C` flag bytes + (numpy only) `REFERENCE_TEMP_F64=12` float64 and `REFERENCE_TEMP_BOOL=6` bool `(C,)` work arrays + the fixed
`HOST_OBJECT_ALLOWANCE_BYTES = 16384 + 3*8*FLAG_SEARCH_CHUNK` (114 688 B: small Python objects and three chunk-sized host temporaries; the first flagged column and the two residual maxima are found in
chunks of 4096, so no domain-sized index, `|residual|` or ratio array exists).

**CUDA (correction02: an explicit two-stage RawKernel reduction, no library reduction at all).** Device: `outputs + 80 B + 5*8*n_blocks + reserve`, where `n_blocks = ceil(C / CUDA_BLOCK)`, `CUDA_BLOCK = 128`:
the 80-byte diagnostic/packet (`DIAG_WORDS = 10` `uint64`: flag OR, flagged count, `first column << 8 | flags`, bit patterns of max|res| and max res/bound, then the five totals' bit patterns) and the
`(5, n_blocks)` float64 block partials. There is no per-column stats/flag/index/mask/`|residual|`/ratio array, no CuPy `sum`/`stack`/`argmax` (so no hidden library workspace or per-reduction scalar). Kernel 1
(column work, one thread per column) reduces the five per-column contributions in **static shared memory** (`5*8*128 = 5120` B per block, fixed binary tree) into the partials; kernel 2 (one block of
`CUDA_FINAL_BLOCK = 256` threads, `5*8*256 = 10240` B static shared memory) reduces the partials with a fixed tree into diagnostic words 5..9. The shared memory is reported in the estimate terms (it is
not a global allocation). Host: the 80 B packet read back with ONE transfer + the fixed allowance + `host_reserve_bytes`; independent of `C`. **Raw integers:** the packet stays `uint64`, so the first-column index and
flags are exact; the explicit range gate `C <= MAX_CUDA_COLUMNS = min(2**56 - 1, (2**31 - 1) * 128)` (so `column << 8` fits 64 bits and the grid fits the launch limit) is enforced in `estimate_remap_memory`
before any allocation. Excluded: NVRTC/module load, the CUDA context, the allocator-pool policy, interpreter/JIT state and whole-process memory. Each side's requirement plus reserve must fit its declared quota
and its available-bytes reading or `SoilRemapMemoryError` ("host side" / "device side") is raised before any allocation; a reading is not a reservation.

CPU-measured checks: warm `tracemalloc` (compile excluded, caller inputs pre-existing, invalid-all-columns and valid paths, L = 1 and 3, C = 100 000, both CPU backends) peak <= requirement; a fake-CuPy test records
every allocation of `run` and asserts they equal `outputs + diagnostic + partials` exactly and that no other CuPy call is made; the CUDA C source is compiled by `g++` against stand-ins and its per-column function
is compared with the NumPy reference bit for bit. Hardware-only definitions (CuPy memory hook, multi-block/finalize-stride reduction, determinism, late-column raw-integer diagnostics) are not run by their author.

## Tests (author-written; not an independent oracle)

`tests/phase2_soil_remap/` (run with `-c tests/phase2_soil_remap/pytest.ini`, the materialized 9f bundle and the candidate `src` on `PYTHONPATH`, `CUDA_VISIBLE_DEVICES=""`):
hand-computed dyadic known answers and a `test_correction01.py` regression file (identity `-0.0`/subnormal bits, typed non-finite errors, warm failure-path memory, chunked flag search, host/device estimate fields), (1, 2 and 3 layers; shift down/up; top removal; disjoint above/below with and without a gap; dry and saturated boundary water), bit-exact identity for
`L=1,2,3,6`, empty selection, input preservation and fresh outputs, conservation per column and globally on random columns, cell-permutation equivariance (bitwise), column independence, pure
translation, `theta` extremes, about 100 collected validation/refusal cases (105 test items in `test_validation.py` across backends), memory admission (budget, reserve, availability, exact fit), an estimate-vs-`tracemalloc` check of the NumPy reference, and
NumPy-vs-Numba bitwise equality with identical refusal classification. `test_cuda_definitions.py` defines the device tests (bitwise vs NumPy, identity incl. signed zero/subnormal, empty, device flag diagnostics vs the host classification, typed overflow, separate host/device admission, two-stage reduction over many blocks, determinism, a CuPy memory-hook allocation check, a late-column raw-integer diagnostic); `test_correction02.py` (fake CuPy, CPU) and `test_cuda_source_host_check.py` (g++ host check of the CUDA C per-column function) run on the CPU; they are **not run by their
author** and skip without a device. Root writes the independent exact-rational (`Fraction`) oracle in a separate file; none of the oracle logic is in the runtime modules.

## Limits

Piecewise-uniform water in every old interval (first-order: moisture fronts are smeared at interval scale). No physical acceptance, ordering, frame, water-fate or wetting-history decision is made.
No storm, integration, restart, performance or whole-process-memory claim. CUDA exact agreement is a declared expectation until Root's hardware run.


## Independent component qualification — 2026-11-12

Codex independently reviewed the final correction02 source and ran 226 CPU checks (15 CUDA definitions skipped on CPU), 15 actual GPU checks and 124 separate exact-rational profiles on each of NumPy, compiled Numba and CUDA. The CUDA workspace is explicitly sized and its allocation-hook check passed; original inputs and identity bits are preserved. Exact sources, commands, evidence and limits are in [phase2_remap_component_acceptance.json](phase2_remap_component_acceptance.json). This accepts the explicit-interval mathematical component only. It does not adopt a soil-motion/water-fate policy, qualify evolving-bed storms or complete Phase 2.
