# Phase 2 E2 — compact selected-cell soil-profile batches and cold canonical depth

Status (2026-11-13): independently accepted isolated component; pending main-candidate integration. Root passed 60 CPU checks and 2 actual-GPU checks (one two-device check skipped). Three independent exact material profiles passed in six backend/storage modes; all physical hashes matched and canonical depth matched native bytes. See [exact source and evidence](phase2_compact_soil_profile_acceptance.json). No water/property/native-adoption or restart acceptance is implied.

`compact_soil_profile.iter_compact_soil_profile(voxel_column, active_layer, geometry, grain_classes, mass_resolution_kg, layers,
*, indices, active_mask, memory_budget_bytes, backend, reserve_bytes, batch_cells, coverage)` returns a one-shot iterator of
`CompactBatch` (<= B selected cells): `(L,B,K)` mass/fraction, `(L,B)` total/coverage, S1 sequential `represented_depth_m` and cap,
exact original flat `indices`, host `canonical_depth_m`, and descriptive `source`. The S1 engines (`reference_texture_batch`, Numba,
CUDA via `soil_profile_texture._runner`), MAPLE validators and the native windowed gather are reused unchanged; compact results are
byte-equal to the whole-grid E1 reader's selected columns (tested). Public whole-grid behaviour is unchanged.

* Selection: 1-D int64 sorted unique, in range, all active in the mask; empty = no-op (no validation/gather/hash/allocation).
* Dense validation is selected-batch-only; Windowed always runs the full native trusted-boundary validation of the whole store
  at construction (explicit O(store) cost); an unpublished journal is rejected. The resident window floor is never bedrock.
* Canonical depth: NumPy contiguous `sum(axis=(-2,-1))` over the gathered `(B,V,K)` rows, native `(total/rho)/area` plus native
  `active/(rho*area)`; bitwise equal to `diagnose_combined_surface_state(...).combined_elevation_m` (tests). Dense device: counted
  bounded download of the selected rows per batch (cold). Not the hot near-rock algorithm; never replaced by or clipped to
  the sequential depth.
* Memory: `estimate_compact_memory` replaces S1's `(L,N,K)` output/staging and N-scale index terms with B-sized compact terms
  (+ host mask N bytes) and reuses S1's scratch/validator/gather/cache/fixed/metadata terms at the same B. Default batch shrinks
  and is recorded; explicit batch is never changed; admission precedes any gather.
* Consumer contract: no retained batch/history; consume/release each batch before the next; retained batches are caller-owned and
  excluded from the stated peak. Value-dependent refusals raise at the offending batch; nothing is mutated.
Not claimed: property refresh, water, dirty detection, Pond/restart/CLI, hot depth classification, native integration.

## Correction01 contracts
* Selection ownership: after metadata-only admission the iterator owns an immutable (non-writeable) copy of the int64 selection
  (8C bytes, counted as `selection_bytes = 8C + 4B`); order (also across chunk boundaries), bounds and activity are validated on
  that copy in B-sized chunks while its SHA-256 is streamed from contiguous views (no C-sized bool, no `tobytes` copy). Later
  caller mutation/rebinding of the indices or mask cannot change yielded batches or the source digest. Empty selection returns
  after cheap metadata checks only (type/dtype/rank, geometry grid, representation, mask metadata): no layers, budget,
  validators, gather, hash or allocation.
* Device guard (CuPy only): every participating device array (state, active layer, a device mask) must live on one device that is
  the current device, checked before admission reads, the mask copy, validators, gather or scalar reads; the device is never
  switched; `source["device_id"]` records it (None on host). Windowed on a GPU stays HOST compute with device output placement.
* Canonical depth: exact `numpy.ndarray` inputs only (no subclass/masked array); at most 3 live `(B,)` rows; native NumPy
  reduction on the host (not the fixed-tree GPU reduction; the 32768-lane bound does not apply). Missing
  `numpy_pairwise_order.ORDER_ID` is a typed source failure.
* All transfers use MAPLE `to_device`/`to_host` (windowed active gather index included); no full-N mask upload.

## Correction02
A device-resident `active_mask` is refused (metadata-only, before any transfer) when the native source is CPU resident; supported
masks are host bool arrays, or a device mask on the SAME device as a device-resident source (guarded). Pre-admission wording: only
O(L) layer parsing and O(1) metadata precede admission; no B/C/N-sized array is allocated before it.
