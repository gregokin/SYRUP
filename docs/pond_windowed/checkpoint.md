# Windowed pond checkpoint (`maple-syrup-pond-checkpoint/2`)

Module: `src/maple_syrup/pond_checkpoint.py`. Tests: `tests/pond_windowed_checkpoint/`. The implementation author used file tools only; independent executed results are recorded in [qualification](qualification.md).

The dense pond checkpoint (`maple-syrup-pond-checkpoint/1`, `docs/splash_and_overflow/pond_checkpoint.md`) is unchanged: same
files, key sets, binding fields and load order. A pond state whose bed is MAPLE's resident-window
`WindowedVoxelColumnState` is written under a second schema.

## API

```python
save_pond_checkpoint(path, state_or_handoff, event, identity, *, max_bytes=DEFAULT_MAX_CHECKPOINT_BYTES) -> Path
load_pond_checkpoint(path, event, expected_identity, *, allow_complete=False,
                     max_bytes=DEFAULT_MAX_CHECKPOINT_BYTES, expected_bed_storage=None)
```

`save` picks the schema from the bed (`is_windowed_bed`); `load` picks the loader from the manifest schema.
`expected_bed_storage` (`None` | `"dense"` | `"windowed"`, new, default `None`) lets a caller demand one storage; any other
checkpoint is refused before a payload is read. Schema constants: `SCHEMA` (`/1`), `WINDOWED_SCHEMA` (`/2`).

## Layout (same three files, same names, a NEW sibling directory, atomic no-overwrite rename)

| file | content |
| --- | --- |
| `pond_checkpoint.json` | manifest, see below |
| `maple_state.npz` | MAPLE's `save_windowed_state_snapshot`: resident `window_kg`, `base`, `deep_by_class_kg`, `deep_total_kg`, `deep_room_kg`, `prefix_packet`, active layer, availability, water, and one Unicode `metadata` scalar (store directory, version digests, generation, class ids, time, step, provenance) |
| `pond_continuation.npz` | the companion exactly as in the dense schema: hydraulic state, phase, held topology SOURCE, datum, water / sediment ledgers, the native pending ledger and committed topography, counters, quiet; for a handoff the pre-reset arrays, terminal deposition and accounting |

The native snapshot is the ONE bed authority. The companion repeats no voxel, window, buried or active inventory. There is no
column densification, no copy of deep chunks, no pickup arithmetic, and no copy of the native snapshot code.

Manifest keys = dense keys + `bed_storage` (`"windowed"`) + `deep_storage`:

```
deep_storage = { deep_chunk_encoding, deep_overlay_max_bytes,            # the event context's option, EXACTLY
                 store_format_version (1 full | 2 overlay),
                 physical_manifest_digest, logical_manifest_digest,       # equal for v1, distinct for v2
                 header_digest, store_dir (absolute, resolved), generation (version id),
                 n_voxels, window_voxels, chunk_columns, n_chunks, class_ids, packet_terms }
```

The binding additionally holds `bed_storage` and `deep_storage_option`, and every binding digest is seeded with the windowed
schema string.

## What the checkpoint references, and what it is not

It references ONE immutable version of an external deep store (content-addressed by its manifest digest). It is **not a portable
stand-alone archive**: the store directory and the referenced version (all its chunk / overlay files and its manifest) must be
retained, unmodified and at the recorded path. Older versions are never copied, rewritten or deleted by save or load; a store
that keeps growing by new versions leaves every earlier checkpoint loadable. A missing or altered chunk / manifest makes the
load refuse (MAPLE's `load_version` verifies every chunk size and SHA-256 and every v2 overlay materialization).

## Publication rule (the integration contract with the core)

MAPLE's `save_windowed_state_snapshot` requires a PUBLISHED column. A column carrying a pending deep journal is refused
before any file is written, any store is read, or any version is published, with a message asking the caller to publish the
accepted storage state explicitly (native path, e.g. `sediment_bed.publish_bed_storage`, or the publication a core `evolve`
return performs at its boundary). The checkpoint never publishes, canonicalizes, deletes or replays a journal, never commits,
reroutes, resets or physically steps, and never writes into the deep store. A completed `PondDryHandoff` is already published
(`finalize` publishes) and is saved as data; it loads only with `allow_complete=True`, and its dry reset / terminal deposition
are never replayed.

## Load order (nothing is trusted before the step that proves it)

1. path, `max_bytes` (positive int), `allow_complete` (bool), `expected_bed_storage`; directory has exactly the three files;
   strict manifest JSON, schema dispatch (a `/1` manifest never reaches the windowed steps and vice versa).
2. before any payload is opened: exact windowed key set, manifest SHA-256, caller identity, source pair (`maple_syrup` tree and
   `maple` package, exact; old-source checkpoints intentionally fail, no migration), array backend (no cross-backend), hydraulic
   backend, status / `allow_complete`, `bed_storage`, the independent event binding (native context, column, sediment, schedule,
   control, ORIGINAL topology, deep-chunk option), and the typed `deep_storage` record against the event's context (no
   cross-encoding or bound change; a `full` context refuses a v2 version; class order equals the context's).
3. bounds from the manifest alone: the grid-derived layout of the native snapshot (shapes and dtypes of every member) is
   computed and must fit `max_bytes`; then payload SHA-256, zip central-directory audit (stored `.npy` members only), the
   companion table, and a header-only audit of EVERY member of BOTH archives: object / structured dtypes refused, declared
   size = `.npy` header + shape x itemsize exactly, native members equal the expected set / dtype / shape, and the Unicode
   `metadata` scalar must be 0-d and at most 8 MiB. Nothing is expanded before this passes.
4. MAPLE's `load_windowed_state_snapshot` with the caller's remaining bound (`max_bytes` minus the companion). It re-validates
   (streamed) and loads the referenced version. The loaded column must be published and equal the recorded version in format,
   physical / logical digests, header digest, store directory, generation, geometry, class ids and packet size, and the
   snapshot provenance must be bound to this manifest.
5. companion decoding through the fixed whitelist, reconstruction in the event namespace (CuPy: MAPLE's `to_device_tree` plus
   `move_prefix_packets`, a counted copy, because the tree walk does not carry the derived height packets), then the SAME
   independent validation the writer runs (below).
6. the verified state is returned; the event rebinds its runtime lazily from the saved held source.

## Validation shared by save and load (`_validate_state`)

All dense checks plus the windowed ones: the bed is a published `WindowedVoxelColumnState`; the actual MAPLE
`validate_windowed_partition` (streamed chunk / prefix rederivation); availability, ledger, committed topography and water
shape / type validators; `validate_reconciliation_context` and `_reconcile_ledger_with_physical_state` (which dispatch to the
windowed fold); the fresh topology build from the SAVED held source with the saved generation; the hydraulic state against a
freshly built solver; the geometry gate and net displacement from the core's exact height-packet functions; phase partition;
counters / clock / quiet; `event.audit`. `PondEvent.step` is never executed to validate.

## Limits and costs (honest)

* `max_bytes` (default 1 GiB, never inferred or raised) bounds the summed uncompressed companion plus resident native
  snapshot. It does NOT bound the externally referenced deep chunks; the store keeps its own `max_store_bytes`.
* Save estimates only resident arrays, counters, packet and metadata bytes (never cells x voxels x classes); the exact
  size is audited again from the written archives before the rename.
* Full partition validation streams the deep store chunk by chunk (reads every chunk; an overlaid v2 slot is materialized in
  memory, never written). Cost: save = one streamed pass in the checkpoint validator plus one inside MAPLE's snapshot writer;
  load = one inside MAPLE's loader (plus `load_version` hashing every chunk) plus one in the checkpoint validator. This is
  O(store size) I/O at checkpoint boundaries only, never per physical step, and it is not cheap for a large domain.
* Same-backend, same-hydraulic-backend exact continuation only. No cross-backend, cross-storage (dense <-> windowed) or
  cross-encoding (full <-> overlay) migration.
