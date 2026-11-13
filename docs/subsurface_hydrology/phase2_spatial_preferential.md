# Phase 2 S2-B2.b — fixed explicit spatial preferential (bypass) infiltration (CPU trial)

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_spatial_preferential_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: author-written, CPU only (NumPy and compiled Numba), uncommitted, awaiting Root's independent review. No GPU, CUDA, native storm/lifecycle/checkpoint, LEAF or Phase 2.5 claim.

## What it is
`maple_syrup.spatial_preferential_infiltration` runs the EXISTING Phase 1 preferential law (`preferential_infiltration`, see its module note) on a real `SpatialSoilColumn` / `SpatialSoilState`; it adds no physics.
One aggregate pathway: a soil background weight vector and one LEAF receiving-weight vector per species (positive vectors are normalised, all-zero disables, `[0,1,1]` splits equally between layers 2 and 3);
layer map `g_l = a3 b_l + (1 - a3)/(1 + C) sum_s c_s u_sl`; hourly fraction `beta = maxFL2 sum_l g_l`; continuous rate `k = -log1p(-beta)/3600 s^-1` (`+inf` for `beta = 1`); isolated attempt `H (-expm1(-k dt))`;
ONE aggregate cap `max_rate dt`; allocation TOP LAYER FIRST, each receiver bounded by its SATURATION room and the surface still standing. A rejected share stays on the surface (never redistributed); no per-species debit,
no root plasticity, no double count. The matrix history `J` is untouched; daily recharge is not preferential intake. Weights and cover remain prescribed inputs (Phase 2.5 not implemented).

```
op  = prepare_spatial_preferential(column, a3=..., max_fraction_per_hour=..., max_intake_rate_m_s=..., background_receiver_weights=..., species_receiver_weights=..., species_contributions=iter, memory_budget_bytes=...)
res = spatial_preferential_step(op, surface_water_m, state, dt_s, memory_budget_bytes=...)      # SpatialPreferentialStep
```
Inputs/copies: `a3` in [0,1] and `max_intake_rate_m_s >= 0` (`+inf` = no cap) are host scalars or exact `(N,)` float64 C-contiguous arrays validated on EVERY cell; `a3` is read during the build only, a per-cell max rate is copied once into
sealed storage, a scalar keeps one value; contributions are streamed one `(N,)` array at a time and never retained. Units m, s; `dt_s >= 0`; the stage preserves `state.t_s`; `dt = 0` is the identity; inactive water/surface are
preserved and nothing is allocated to inactive cells; a refused trial writes and publishes nothing; non-finite global volumes are refused (typed) before the strict MAPLE bound helper.

## One body per backend; shared derivation
* The derivation of the maps (`rate (N,)`, `receiver (L, N)`, `beta`) is the common `_derive_pathway` (moved textually out of `prepare_preferential`, tested by text equality) through a private `_DeriveContext(xp, L, N, mask, contribution check)`.
* NumPy: `_reference_step_body` (the old `reference_step` body, textually unchanged) with a private `_StepContext(n_layers, n_cells, mask, cell_tol)` and lazy readers of the owner's raw maps (`sat = theta_sat dz`, `cap_tol = coef sat`).
* Numba: one kernel body in `preferential_infiltration_numba._build(numba, common)`, compiled twice by dead-branch pruning; the common variant keeps the historical 16-argument wrapper; the spatial variant has an explicit read-only signature, strided raw-map reads and two `(L,)` local rows.
* Budget: private `_resolve_core(area, L, N, packet)` (finite checks before the strict bound, `n_terms`, area once).

## Retained state, identity, binding
Retained per operator (sealed bytes-backed read-only, counted): rate `(N,)`, receiver `(L, N)`, max rate (one value or `(N,)`), the small `ReceiverWeights` records. No saturation/tolerance bound tensor, mask/geometry copy or host mirror.
`sha256` = digest of (tag `maple_syrup.spatial_preferential.operator.v1`, column input SHA, backend, dimensions, scalar parameters/layouts, raw weight records, the sealed derived maps), streamed in 1 MiB chunks once at preparation;
steps check metadata only; `verify_operator_identity` re-hashes cold. The binding is to INPUTS (twin operators share an identity); it is not a water generation, accepted-transition receipt, exactly-once record, LEAF adoption,
lifecycle, restart or storm claim. Legacy and spatial operators/states refuse each other (typed); a CUDA owner is refused before any copy.

## Memory model (host payload bytes; not a reservation)
step: `(2N + 2LN)8` outputs + `8 N 8` rows + scratch (`48 N 8` numpy | `2 N 8` numba, `+ 16 N` mask bytes + packet scratch) + (numpy `6 N 8` lazy rows | numba `2 L 8` local rows) + 64 KiB allowance + reserve.
prepare: retained `(2N + LN)8` + transients `(4N + LN)8` + masks `8 N` + weights `2(S+1)L 8` + allowance + reserve. Excluded: caller state/surface/contributions, the owner's retained maps, JIT, interpreter, allocator pools.
Admission precedes every allocation, conversion, weight record, species consumption, compile and NumPy lazy temporary.

## Limits
Fixed explicit geometry; no moving frame, water fate, provider, K-averaging, root frame or history policy is chosen. Numba vs NumPy differ only by libm `expm1` (declared rtol 2e-12 / atol 1e-14). CPU author evidence only; CUDA is S2-B2.d.

## Correction01 (2026-11-12, supersedes the identity-hash wording above)
The cold identity digest previously serialised each raw weight vector as one `raw.tobytes().hex()` f-string, whose data buffers scale with L and exceeded the modelled prepare payload at N = 1 and L = 4096 / 8192 (exact-quota NumPy path, traced). The weight records are now streamed: the prefix and then the hexlified raw bytes in sequential slices of `WEIGHT_HASH_CHUNK_BYTES = 4096` raw bytes (one 8192-byte hex buffer at a time, independent of L; counted in the Python allowance). The canonical bytes (`label|hex|NUL`, or `label|none|NUL`) and every identity are unchanged; the cold `verify_operator_identity` uses the same path; estimates, quotas and physics are unchanged. The weight RECORDS themselves (raw and normalised, `2 (S + 1) L` doubles) remain counted in the prepare model as before.


## Independent qualification

Root accepted the fixed explicit-property CPU trial at package `976faaef532ff16b2b88abd5481243a29728f3969c92c6c78de53f8ed4fc1542` (123 files). See [phase2_spatial_preferential_acceptance.json](phase2_spatial_preferential_acceptance.json). The final 601-test suite passed independently; each CPU backend reproduced all four saved arrays over 588 original-reference columns exactly, including background/species overlap and disabled pathways. Existing common CUDA outputs retained all 704 saved arrays; this is not spatial CUDA support.

Root found and corrected an uncounted full-vector weight-hex serialization buffer before acceptance. The corrected helper streams 4096-byte raw chunks while preserving every canonical digest byte. At 8192 layers and one cell, the observed preparation peak fell from 463868 B to 266860 B under the unchanged 327736 B admission estimate; no quota was raised. Compilation/interpreter/allocator exclusions and incomplete compiled-code tracing remain explicit.
