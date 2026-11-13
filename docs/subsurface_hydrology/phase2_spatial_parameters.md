# Phase 2 S2-A.1: explicit per-cell / per-layer soil parameters, preparation only (`parameter_component_only`)

Status: **new module and tests, uncommitted; component only.** `src/maple_syrup/spatial_soil_layers.py` validates caller-supplied soil-layer inputs for `N` cells and owns detached, backend-resident copies of a minimal retained set. It contains **no
hydrology** (no step, kernel, runner, state evolution, infiltration, evaporation, transpiration, storm or checkpoint) and makes **no physics choice**: no moving frame, texture-to-property method, field-capacity pressure, conductivity averaging,
wetting-history rule or water fate. `theta_residual` is a withdrawal floor, not a minimum storage: dry (zero) states stay allowed and nothing here inspects or invents water. It is a separate class from `subsurface.SoilLayers`/`SoilColumn`
(no subclass). The three pending user physics questions are unchanged; nothing here depends on their answers.

## Interface

```python
spatial_layer_spec(depth_edges_m, *, theta_residual, theta_field_capacity, theta_saturation, diffusivity_m2_s=None, interface_conductivity_m_s=None, diffusion_coefficient=None,
                   theta_evaporation_stress, evaporation_factor=1.0, gravity_drainage=True, diffusion_mode="constant", bottom_drainage="open", n_cells) -> SpatialLayerSpec
prepare_spatial_column(spec, active_mask, cell_area_m2, *, backend="numpy", memory_budget_bytes, reserve_bytes=0, available_bytes_fn=None, host_memory_budget_bytes=None,
                       host_reserve_bytes=0, host_available_bytes_fn=None, chunk_columns=None, hash_chunk_bytes=1<<20) -> SpatialSoilColumn
estimate_spatial_memory(n_layers, n_cells, *, layouts, diffusion_mode, backend, chunk_columns, hash_chunk_bytes, reserve_bytes, host_reserve_bytes, input_bytes) -> SpatialMemoryEstimate
SpatialSoilColumn.summary()      # bounded metadata only
SpatialSoilColumn.inspect(name, *, host_memory_budget_bytes, host_reserve_bytes=0, host_available_bytes_fn=None)   # an ADMITTED detached host copy
```

* **Inputs** (explicit arrays of exactly the backend's namespace, float64, C-contiguous; no conversion, no transfer): `depth_edges_m` `(L+1,)` SHARED or `(L+1, N)` CELL (each column strictly increasing from exactly 0: depth below the local surface; the caller's geometry, no frame is
  chosen); the three thetas as a real scalar, `(L,)` or `(L, N)`; the interface input (`diffusivity_m2_s` in mode `constant`, `interface_conductivity_m_s` + `diffusion_coefficient` in mode `tietjen_donor_depth`: the existing modes only, no averaging) as a scalar, `(L-1,)` or
  `(L-1, N)` (for `L = 1`: a scalar or an empty array, no interface exists); the shared scalars/strings of `soil_layers`. Cells are flattened in C order (`cell = iy*nx + ix`). `N >= 1`. Layouts SCALAR/SHARED/CELL are derived from shapes and recorded; a SHARED or SCALAR array is never materialised per cell.
* **Validation** (every cell, inactive ones included; the stored mask never skips a check): exactly the per-element rules of `soil_layers` with the same expression order (finite edges, `edges[0] == 0`, strictly increasing finite widths; finite thetas with `0 <= res <= fc <= sat <= 1`, `res < sat`; interface input finite `>= 0`;
  `stress > theta_residual[0]` in every cell; every derived quantity finite/representable, separations `> 0`, storage depths `theta*dz` finite, saturation depth `> 0`, tolerance `coef*sat_depth` resolvable). Input flags (1 edges, 2 non-finite theta, 4 theta relations, 8 interface input, 32 stress) come first; the derived flag (16) only for cells whose
  inputs passed. The error (`SpatialLayerError`, a `SubsurfaceError`) carries `bit`, `n_flagged`, `first_cell`; nothing is published and the inputs are untouched. Tested: refusal agrees with the real `soil_layers` on 250 random single-cell corruptions per mode (including `1e308`, `1e-320`, `nan`, `inf`).
* **Retained maps (the only arrays kept):** the three thetas (detached copies, layout kept; a scalar becomes a 1-element array), `dz`, `pair_inverse_thickness`, `interface_conductance_down` (+ `interface_conductance_up` in donor mode; in constant mode the "up" conductance aliases the same map, as in `SoilLayers`). The top residual theta0 is row 0 of
  `theta_residual`. **Not retained:** edges, centres, `D`/`K` inputs, depth-form thresholds (`theta*dz`), capacity-tolerance maps (a later compiled reader recomputes `theta*dz` and `coef*sat_depth` with the same ordered operations, to be qualified then). Retained doubles per cell (all CELL):
  constant `4L + 2(L-1)`, donor `4L + 3(L-1)`:

  | L | constant B/cell | donor B/cell |
  |---|---|---|
  | 1 | 32 | 32 |
  | 2 | 80 | 88 |
  | 3 | 128 | 144 |
  | 5 | 224 | 256 |
  | 8 | 368 | 424 |

  plus `N` bytes of owned mask. (The S2 design's earlier count was `8(7L-1)`; this slice retains 18 % less at `L = 5`.) Per-array layouts mix freely: the layout of `dz` and `pair_inverse_thickness` is that of the edges; `interface_conductance_*` is CELL if the edges or the interface input are CELL.
* **Ownership:** the prepared column owns private copies (NumPy arrays read-only; CUDA arrays private to the column; no public mutable view). Mutating the caller's arrays afterwards changes nothing. `summary()` is bounded metadata (no reduction, no host copy); `inspect()` makes ONE detached host copy after admitting its bytes against a quota and a truthful reading. No security claim against a caller with private access.
  `component_tag = "parameter_component_only"`, `hydrology_step_supported = False`.
* **Identity:** `parameters_sha256` hashes the schema tag, the options and each **input** array (name, layout, shape, bytes row by row in C order, streamed in `hash_chunk_bytes` pieces; NumPy hashes views, CUDA copies each piece into ONE preallocated counted host buffer), and `mask_sha256` hashes the mask; chunk-size invariant (tested for 8 B...1 MiB pieces and
  1...N-column chunks), layout-sensitive, distinct schema from `SoilLayers.identity()`. Because it hashes inputs and not derived maps, CPU/GPU roundoff in derived values can never change it. The summary also records the module source hash; no existing verifier, capsule or receipt is touched.
* **Backends:** "numpy" and "numba" run the NumPy derivation (numba must be importable; this slice has no kernel to compile). "cuda" runs ONE RawKernel (one thread per cell, layer loop, the same expression order, `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`) on inputs that already live on the current device; flags, counts and the first flagged cell travel through
  device atomics into four `uint64` words read back with one 32-byte transfer. No CuPy reduction, no flag/mask/index array, no library workspace, no full-size download; `N <= MAX_CUDA_CELLS = min(2**56-1, (2**31-1)*128)` is an explicit gate. Host NumPy arrays on the cuda backend (or the reverse) are refused: no implicit transfer.

## Memory (modelled; not measured resident memory)

Counted before any allocation, host and device separately, against explicit quotas (`memory_budget_bytes`, `host_memory_budget_bytes`) and truthful injectable readings; the three `host_*` arguments are refused on the CPU backends; nothing is raised silently. Terms: retained maps + mask copy; for the CPU backends `REF_VEC_F64 = 24` float64, `REF_VEC_BOOL = 16` bool and `REF_VEC_U8 = 4` uint8 `(chunk_columns,)` work vectors (default chunk 65 536;
independent of `N` and `L`); a 65 536 B Python-object allowance; for CUDA, device = retained + mask + 32 B diagnostic + 32 B auxiliary array, host = identity buffer (`hash_chunk_bytes`) + 32 B packet + allowance + host reserve (independent of `N`); the kernel uses no temporary arrays. The caller's input arrays are **excluded** and reported with their actual bytes (`estimate.input_bytes`). Excluded:
NVRTC/module load, CUDA context, allocator pool policy, interpreter/JIT state. Warm `tracemalloc` checks (both CPU backends, both modes, all-CELL and mixed layouts, default and small chunks, success and last-chunk failure, `N = 60 000`): peak <= requirement; for the default chunk the measured peak was 15.8 MB (constant) / 18.2 MB (donor) against a modelled 20.4 MB / 21.4 MB.

## Gating of existing APIs (correction01)

The spatial column and spec are not subclasses of `SoilLayers`/`SoilColumn`. **All 19 existing public functions whose first parameter is a soil `column` now refuse the spatial column with a typed `SubsurfaceError` at their first line** (`subsurface.{admit_step, column_state, initialize_profile, moisture, reference_runner, reference_step, step, step_memory, step_with_recharge, storage_m3, summary, uniform_state}`,
`matrix_infiltration.prepare_matrix`, `preferential_infiltration.prepare_preferential`, `soil_evaporation.prepare_evaporation`, `subsurface_transpiration.{estimate_combined_memory, prepare_transpiration}`, `layered_storm.prepare_layered_storm`, `water_lifecycle.prepare_lifecycle`; tested 19/19). Three of them (`subsurface.reference_step`, `subsurface.reference_runner`,
`subsurface_transpiration.estimate_combined_memory`) received exactly ONE added line each, the existing `_check_column(column)` guard (the only edits to existing sources; see the report for the dirty delta from the 117-file S1 state); a proper `SoilColumn` runs the unchanged body (checked bit for bit against the pre-guard sources: step, runner, pulse-aware runner, combined-memory estimate).
`prepare_spatial_column` refuses a `SoilLayers`, a `SoilColumn` and any non-spec object.

## Correction01 contracts

* **Live revalidation:** the spec is only a convenience record. `prepare_spatial_column` rebuilds it from the LIVE arrays/scalars through the same factory (types, dtypes, contiguity, shapes, layouts, scalars, modes) and requires every cached field (L, N, namespace, per-input kind/rows/bytes, options) to agree, BEFORE any admission, allocation or kernel. A caller who changed an array's shape/dtype after `spatial_layer_spec`,
  edited the spec (`dataclasses.replace`), constructed one by hand, subclassed it, or added/removed/forged input records gets a typed `SpatialLayerError` (tested, with a `tracemalloc` bound proving nothing was allocated). Value edits remain legal and are caught by the value checks. `spec.inputs` is a read-only mapping.
* **One layer:** a scalar `diffusivity_m2_s`/`interface_conductivity_m_s` must still be real, finite and `>= 0` (the accepted `soil_layers` rule) even though no interface array results; empty arrays and valid scalars keep producing no interface maps.
* **Read-only public metadata:** `column.layouts`, `column.estimate.{layouts, retained (nested), input_bytes, terms}` and `column._maps` are deeply read-only mappings; attributes are frozen; `estimate_spatial_memory` copies the caller's layout dict; `summary()` returns copies. The owner can be created only inside `prepare_spatial_column` (direct construction, `dataclasses.replace`, `copy`, `pickle` are refused). The private maps are an ownership convention, not a security boundary.
* **No CuPy on the NumPy path:** `spatial_layer_spec` selects the NumPy namespace directly from a `numpy.ndarray`; CuPy is discovered ONLY when a `cupy`-typed array is supplied (or the cuda backend is requested), and an unusable CuPy is a typed `SpatialLayerBackendUnavailableError` carrying the reason, never a broad fallback. A spy test proves the NumPy spec/preparation/estimator never call `cupy_module`.
* **Ownership:** weakref tests show the owner retains neither the spec nor any caller array; the owner's public metadata equals its private maps and estimate (shapes, bytes, kinds) for all-CELL and mixed layouts.
* **GPU test scope:** the device allocation test creates the caller's mask BEFORE its memory hook and passes it in; the estimator, quotas and tolerance are unchanged (the earlier 4 099 B difference was the caller's `cp.ones(N)` mask allocated inside the hook).

## Tests (author-written; CPU only; not an independent oracle)

`tests/phase2_spatial_parameters/` (run with `-c tests/phase2_spatial_parameters/pytest.ini`, the historical 9f bundle and the candidate `src` on `PYTHONPATH`, `CUDA_VISIBLE_DEVICES=""`): derivation equal **bit for bit** to the real `soil_layers` per cell (L = 1, 2, 3, 5; both modes; dry; six mixed SHARED/SCALAR/CELL layout cases; chunk invariance 1...N); the metadata/value/refusal matrix with first-offender reports and
flag hierarchy; the 250-case refusal-parity fuzz; identity determinism/sensitivity; memory model, admission and warm peaks; gating and ownership; and `test_correction01.py` (live-revalidation, one-layer contract, read-only metadata, lazy CuPy, ownership, unchanged-numerics checks), `test_cuda_plumbing_host_check.py`, which compiles the CUDA C text with `g++` against stand-ins, runs the whole kernel body per cell (strides, atomics, flags) and drives the REAL `_derive_cuda` through a fake CuPy (a plumbing double: numpy arrays plus `.get()`) to compare with the NumPy reference bit for bit in
all layout/mode/L combinations, check the exact allocation list (diagnostic + auxiliary only), the flag/first-cell classification and the device-style buffered identity route. **No hardware claim**: `test_cuda_definitions.py` defines the device tests (maps vs NumPy, mixed layouts, exact first cell at N = 100 001, host/device admission, a CuPy memory hook, inspection, identity equality with the CPU) and is not run by its author.

## Limits

Component only; no hydrology, no kernels, no state, no lifecycle/checkpoint/restart. CUDA NVRTC compile and launch are unexecuted until Root's hardware run; CPU/GPU derived roundoff is not claimed identical (identity hashes inputs). Updating changed columns (generations) is not in this slice. Windowed beds, LEAF and texture-derived properties are out of scope.


## Independent component qualification — 2026-11-12

Codex independently ran 469 CPU checks (15 hardware definitions skipped), 15 actual CUDA checks, 56 reference-column controls on each of CPU and CUDA, and mutation/ownership safeguards. All passed on the reviewed correction01 source. The three added common-column entry checks preserve all 512 physical arrays in 64 independent before/after scenarios bit for bit. See [phase2_spatial_parameters_acceptance.json](phase2_spatial_parameters_acceptance.json) for exact sources, commands and limits. This accepts explicit parameter preparation only: no spatial hydrology step, moving-bed event or restart is enabled by this receipt.
