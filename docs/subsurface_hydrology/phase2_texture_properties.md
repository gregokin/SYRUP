# Phase 2 S2-C: pure texture -> hydraulic-property provider (Saxton & Rawls 2006)

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_texture_properties_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status (2026-11-13): AUTHOR-VERIFIED CPU component, uncommitted, NOT independently qualified by Root, NOT adopted by any native path. The CUDA backend is written and unrun (no GPU, CuPy initialisation or NVRTC
compile was performed by the author). This is equation/numerical verification of a published regression recipe, not field validation. Modules: `src/maple_syrup/soil_texture_properties.py` (reference, API,
admission, identity), `soil_texture_properties_numba.py`, `soil_texture_properties_cuda.py`; tests `tests/phase2_texture_properties/`.

## What it does

A pure function of explicit inputs. For `L` fixed layers and `N` cells it returns `theta_field_capacity_6kpa`, `theta_saturation` and `saturated_conductivity_m_s` (optional diagnostics `theta_1500`,
`air_entry_kpa`, a `provenance` uint8). It reads or mutates no soil water, bed, geometry, base/rock datum, clock, checkpoint or history and chooses no remapping, water-fate, wetting-history, averaging or Green-Ampt rule.
Green-Ampt front suction, the residual content and the interface conductivity/diffusivity remain explicit consumer inputs (no `Ks/2`, `K_geom` or interface averaging). The native bed class -> (sand, silt, clay,
OM, density) mapping and the packing/porosity consistency are later required qualification and are NOT accepted here. Fine earth only (no gravel/coarse input exists).

The complete recipe (Table 1 Eq 1-16, 18, equation-numbered), units and refusal bits are in the module docstring of `soil_texture_properties.py`; in short: mandatory sand/silt/clay fractions, organic matter percent
by weight and `bulk_density_kg_m3` (SI; divided by 1000 to the paper's g/cm3 inside); computed density factor `DF = rho/rho_N` (never a second mandatory input); `theta_33` (not `theta_6`) anchors Eq 16;
`theta_6 = theta_S` when `psi_e >= 6 kPa` (a valid, flagged outcome) else the Eq 12 value; the air entry is evaluated from the normal-density Eq 3 input and held fixed under density adjustment (declared
recipe assumption, not a full-SPAW identity claim); exact air entry 0 is valid; negative, non-finite or `>= 33 kPa` is refused unless a valid measured air entry is supplied. Unsupported predictions are
REFUSED with a typed `TexturePropertyError` (first flag, count, first layer/cell), never clamped; fractions must close within MAPLE's own `summation_error_bound_kg(3, 1)`.
Optional measured `theta_1500/theta_33/theta_saturation/air_entry_kpa` overrides replace the FINAL anchors (NaN = not measured; +-inf refused) and a measured Ks (> 0) replaces Eq 16 only; the final set
passes the same checks. There is no second prescribed-mode engine and no measured-FC6 override (a field capacity without a claimed curve belongs to the existing explicit path).

## API (summary)

`estimate_soil_texture_properties(*, sand_fraction, silt_fraction, clay_fraction, organic_matter_percent, bulk_density_kg_m3, n_layers, n_cells, [theta_1500_override, theta_33_override,
theta_saturation_override, air_entry_kpa_override, saturated_conductivity_override_m_s], backend="numpy"|"numba"|"cuda", diagnostics=False, memory_budget_bytes, reserve_bytes, available_bytes_fn,
host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, chunk_columns, hash_chunk_bytes) -> SoilTextureProperties` and `estimate_property_memory(...)`. Every input is a host scalar (SCALAR), an exact
`(L,)` (SHARED) or `(L, N)` (CELL) float64 C-contiguous array in the backend's namespace; no implicit conversion or transfer. All outputs share the join layout (CELL if any input is CELL else SHARED `(L,)`;
shared-only costs O(L) and never expands to N). The result is sealed (not copyable/serialisable), owns read-only NumPy outputs (`array(name)`) or private CuPy outputs (`detached(name, ...)` gives an admitted copy),
and carries a streamed chunk-invariant `sha256` of the inputs, options and recipe plus `source_digests()` of the three module files.

## Memory

Modelled before any allocation, host and device separately, against caller quotas and truthful readings: outputs (O(L) shared / O(LN) cell, plus diagnostics if requested), NumPy chunk work vectors
(`REF_VEC_F64 = 40`, `REF_VEC_BOOL = 24` per chunk column, checked against `tracemalloc` warm), the CUDA host identity buffer and 32-byte packet, one small device auxiliary array, a declared Python allowance and
the caller's reserve. The caller's inputs, JIT/NVRTC compilation, the CUDA context, allocator pool policy and whole-process memory are excluded; no library workspace exists.

## Verification actually run (author, CPU only, existing MAPLE `.venv`, native 9f materialisation, `CUDA_VISIBLE_DEVICES=''`, `NUMBA_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`)

`pytest -c tests/phase2_texture_properties/pytest.ini tests/phase2_texture_properties`: 198 passed, 10 skipped (the 10 are the gated GPU definitions, written and unrun).
Coverage: Root's independent Decimal80 normal-density and 30 density controls (portable copies with attribution under `oracles/`; Table 3 printed rounding; the published sand example refused for its negative
air entry); an author Decimal80 model for measured anchors/overrides (shares the author's reading, a cross-check only); air entry 0 / near 6 / 6 / near 33 / refused; theta_33-anchored Ks when `theta_6 == theta_S`;
override validity and provenance; the refusal matrix and metadata refusals before any kernel; layouts and block-edge sizes (N = 1/127/128/129/257), element equality with single-element calls; NumPy/Numba bitwise
agreement for outputs without a library call and the declared bound (rtol 2e-12, atol 1e-14) for Ks; chunk and hash-chunk invariance; identity and ownership; admission before allocation, exact quotas, warm
`tracemalloc` peak below the model; provider outputs building the ACTUAL `SpatialSoilColumn` (cell and shared layouts) and stepping it; the CUDA kernel SOURCE compiled by `g++` and executed on the host, and the
launcher/plumbing run through a CuPy double (arguments, strides, quotas, flags, diagnostics): NOT hardware evidence (no atomics, launch, device libm or CUDA flags).

## Limits and notes

* Exact density-factor boundaries (0.9, 1.3) are floating-point comparisons: an input exactly at the boundary can land one ulp outside (observed for two of Root's boundary controls). No tolerance was added.
* A read-only versus writeable input array is a distinct Numba type: the first call of each combination compiles once.
* No mask: every cell is validated and must carry valid placeholders.
* Identity is of the explicit inputs and recipe, not of CPU/GPU outputs (roundoff in Ks may differ by an ulp across backends).

## Correction C1 (2026-11-13; API/metadata/counter only, no equation, domain or default change)

* `SoilTextureProperties.detached` on a CUDA result refuses (typed, before any availability reader or copy) when the current device differs from the owner's `device_id`; it never copies across devices.
* The CUDA launcher reads the 32-byte validation packet through MAPLE's counted `maple.core.backend.to_host`; the cold identity chunks remain separate bounded `.get(out=)` reads (no full-map mirror). A gated hardware test asserts one counted 32-byte device-to-host transfer.
* `estimate_property_memory` validates EVERY supplied layout value, including optional overrides (`'not_a_layout'` was previously accepted for an override).
* The CUDA planner enforces the launch grid limit `(2**31 - 1) * CUDA_BLOCK` cells (`MAX_CUDA_GRID_CELLS`) before allocation, in addition to the packed element-index limit; a shared-only call launches one thread and its logical N is not expanded. No other limit was added.
