# Phase 2 E1 — explicit partial-coverage soil-profile texture

Status (2026-11-13): independently accepted isolated component, pending patch adoption into the evolving-soil candidate. Root passed 36 CPU checks and 2 actual-GPU checks. Four independent exact-rational material profiles passed on Dense NumPy/Numba/CUDA and genuine Windowed NumPy/Numba, including Numba reduction with GPU-resident storage; all physical array hashes matched. See [the source and verification record](phase2_partial_profile_texture_acceptance.json). This is not acceptance of native water/property/restart integration.

`build_soil_profile_texture(..., coverage="partial_over_base")` (and `soil_profile_texture_from_bed(..., coverage=...)`)
returns the actual material coverage when the fixed nominal layers extend below the bottom of the native voxel column
(the flat full-grid base, elevation 0 of voxel 0). Default `coverage="full"` is the historical strict behaviour, byte
for byte (the refusal, arrays and `source` record are unchanged; no new key is added to the default `source`).

## Semantics under the opt-in
* The numeric batch kernels (NumPy reference, Numba, CUDA) are unchanged. They already emit overlap-weighted actual class
  mass, fractions, totals, coverage and the sequential represented depth before `FLAG_INSUFFICIENT_DEPTH`; the opt-in
  only stops treating that flag as a refusal.
* A layer (or the lower part of one) below the base has exactly zero mass: `layer_total_kg == 0`, zero fractions
  ("texture undefined"), `coverage_m` shorter than the thickness. Emptiness is read directly from `layer_total_kg == 0` (no helper tensor). Nothing is invented or estimated; fractions of covered layers are the actual mass fractions.
* Validation stage (opt-in only, fail closed): for active cells depth and coverage must be finite and >= 0, and `coverage_m[l]`
  must equal `clip(depth - top_l, 0, dz_l)` within the existing operation-derived bound `factor * max(depth, bottom edge)`.
  It runs per chunk of `batch_cells` cells in the outputs' namespace; only that chunk's bool flags are copied to the host
  (MAPLE counted `to_host`) and the HOST mask slice selects/counts active cells: no second mask upload, no reduction
  workspace, no L*N tensor. Model `partial_stage_bytes(B)` = (6*8*B + 8*B namespace, B host); `_admit_partial_stage` checks
  it against the already admitted batch working set (`validation_bytes`, and the host scratch) because the stage runs after
  every batch buffer is released; otherwise `SoilProfileMemoryError`. Default estimates are unchanged. Gated GPU expectation:
  ceil(N/B) device-to-host copies of <= B bytes each, no other transfers.
* Unchanged: top-layer-vs-active-cap refusal, all native source/representation guards, memory admission, bounded
  batching/gather and the always-on full native Windowed validation, inactive cells (all zero, no refusal).
* `source["coverage_policy"]` (only under the opt-in) records the policy, version tag
  `maple_syrup.soil_profile_texture.partial_coverage/1`, the interpretation, the count of short cells and the not-claimed list.
  `SoilProfileTexture.coverage_policy` (default `"full"`) is a new last field.

## Not claimed
`represented_depth_m` is the EXISTING sequential segment-height diagnostic, not the canonical hydraulic bedrock-depth
witness (native NumPy-order mass-sum height); partial output alone does not qualify hydraulic depth classification,
property refresh, water conservation, native Pond/restart or evolving-soil integration. The resident-window floor is never
bedrock: Windowed columns are read through the unchanged full validation and `gather_full_columns`.
