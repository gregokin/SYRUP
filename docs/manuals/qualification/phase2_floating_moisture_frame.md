# Phase 2 G1 — floating soil-moisture diagnostic frame codec

`floating_moisture_frame.write_floating_frame` / `read_floating_frame` store and reconstruct ONE frame of the surface-following
floating profile (schema `maple_syrup.floating_moisture_frame/1`): `water_m (L,N)`, canonical `depth_m (N,)`, `active (N,)`, nominal
`depth_edges_m (L+1,)`, shape/cell area, completed nominal day / cadence N / origin / date, separate physical `physical_t_s`, and a
caller-supplied attribution record (run/source/lineage digests, CURRENT property/bed/depth generations, parent/state digests).
Attribution is recorded verbatim; it does not prove acceptance. Pure codec: no cadence, no DailyCursor, no restart, no property
maps/beds/histories. Layout `<root>/frame_<day:012d>/{water_m.npy, depth_m.npy, active.npy, manifest.json}`.

Reconstruction: `h_l = max(0, min(D, edge_{l+1}) - edge_l)`, `theta = W/h` (0 where `h == 0`, documented). Active cells: finite
`D >= 0`, finite `W >= 0`, `W == 0` in rock slots (refused otherwise). Inactive cells' raw W/D/mask are preserved bit for bit with no
physical bounds. Storage is chunked (c = chunk_bytes // 48 cells: one D piece, one mask piece, L water pieces per chunk) with MAPLE
counted `to_host` only inside a call; positional writes into preallocated `.npy` files; no full host mirror. Admission (argument
checks, device guard, host staging, frame/byte quota scan, disk reserve) precedes any transfer or write. Publication: exclusive
reservation, fsync, re-hash, manifest temp, manifest-LAST `os.link` seal. Replay of a sealed frame verifies identity, checksums and
live bytes and never overwrites; an existing unsealed reservation is refused untouched; post-seal failures raise
`MoistureOutputIndeterminate` with the sealed frame intact. Reader: strict JSON, `max_return_bytes` checked before any read,
read-only arrays, `theta(max_bytes=)` admits its own `16LN` bytes. G0 (`soil_moisture_output`) is unchanged and historical; its
private descriptor/JSON/header/hash helpers and `MoistureOutputLimits`/errors are reused.

## Correction01 contracts
* Streaming: per cell chunk (`c = chunk_bytes // 64`, declared 64 live host bytes per cell) one D piece and one mask piece are kept and the
  L water rows are downloaded, validated and written/compared ONE AT A TIME; at most D + mask + one water row are live downloaded
  buffers (never L rows). The reader validates one row view at a time.
* Reconstruction: `thickness_m` is one `8LN` output computed row by row in place (no ufunc temporaries); `theta` peaks at `8LN + N`
  (thickness divided in place, one boolean row); `max_bytes` is a typed non-negative integer admitted before any allocation.
* Admission order: type/shape/metadata checks (no value scans; caller-identity size from lengths only) -> device guard -> analytic
  O(L) manifest bound (`2560 + 26(L+1) + caller bound`) and host staging admission -> edge values/owned copy/canonical JSON/exact
  manifest size -> root scan, quotas, disk. `bytes_written` is exact (files + real manifest); the quota estimate uses `n_active = N`.
* Replay: first difference raises a typed conflict (no accumulated list); the sealed manifest must equal what THIS codec source would
  write (strict `module_sha256`): a changed codec source refuses replay; any read-compatibility policy belongs to the final G2
  runtime/source guard. Descriptors are opened incrementally inside `finally`.
* After the manifest link any failure (including `MoistureOutputError`) is `MoistureOutputIndeterminate`; sealed files are never
  removed; unknown/unsealed reservations are preserved (owned crash recovery is a later G2 concern).
* `FloatingFrameMeta` catches date overflow/type errors as `MoistureOutputError`.

Root independently accepted the bounded codec against exact MAPLE9f on CPU (39 tests) and GPU0 (1 hardware test; second-device test skipped), plus the large-L actual download-lifetime and reconstruction controls. See phase2_floating_moisture_frame_acceptance.json for immutable evidence and limits. This is not yet the final native daily save hook or a CLI option.
