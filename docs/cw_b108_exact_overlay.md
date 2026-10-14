# CW-B108: optional exact-overlay deep storage in conservative SYRUP events (candidate)

Status: isolated candidate `outputs/conservative_water/cw_b108_overlay_candidate` (seeded from the root-frozen B104C2
tree, package `8acabd96…`). Not adopted, not delivered; the default and the active CPU reference are unchanged.

## Option

`BedContext.deep_chunk_encoding` (`'full'` default | `'overlay'`) and `BedContext.deep_overlay_max_bytes`
(`None` default; overlay only: a plain positive `int` forwarded to MAPLE as `overlay_max_bytes`). Both are appended
LAST with defaults, so every existing construction is unchanged. Windowed runner: `run_complete_windowed(...,
deep_chunk_encoding=, deep_overlay_max_bytes=)` and CLI `--deep-chunk-encoding {full,overlay}` /
`--deep-overlay-max-bytes`. No global or benchmark default changes; the legacy replay workflow is untouched.

Storage only. Overlay chunks hold exact final replacement values (MAPLE B106C1/B107 store); the logical stored values,
the physics, the routing threshold (`pending_elevation_trigger_m`, `>=`), schemes / bins and the one MAPLE mass
authority are unchanged. Logical identity (`logical_manifest_digest`, journal digest, version id, generation) is
encoding-independent; the physical manifest / files differ.

## Supported combinations (refused otherwise, before any physical step or store publication; no fallback)

| combination | result |
|---|---|
| default `'full'`, any bound MAPLE (M94, accepted `24c6…`) | unchanged call: no new keyword is forwarded |
| `'overlay'` with the accepted typed source `24c6e147…` and a resident-window bed | supported |
| `'overlay'` with M94 or any other source | refused (source) |
| `'overlay'` on a dense bed | refused (no deep store; not ignored) |
| `'full'` with a bound, unknown encoding, bool / NumPy / non-positive bound | refused |
| default `'full'` on a bed whose deep version is v2 (overlay) | refused (no cross-encoding continuation) |
| `'overlay'` with `n_voxels > 65535` | refused by MAPLE's own `_resolve_encoding` rule (windowed runner: before assembly) |

Validation points: windowed runner (pure values before setup; source and voxel rule before store assembly; the actual
bed/header after assembly), `evolve_sediment_event` entry (after input checks, before any event array),
`finalize_event` (before the terminal transaction), `save_checkpoint` (before writing), `load_checkpoint` (manifest
before any payload read; deep version after the snapshot load; source at the end).

## Publication paths

Every SYRUP windowed publication goes through `sediment_bed.publish_bed_storage`, which forwards the context options
(only when non-default) to MAPLE's `publish_pending_column`: per-step publication, MAPLE accounting / net-reroute commit
publication (`publish_commit`), deferred `'boundary'` publication at report boundaries / high-water / the bounded flush
/ return (`publish_carried`), pause / checkpoint (checkpoints save published states only; the `'boundary'` row end
publishes before `on_boundary`), and the terminal settlement / final forced commit / dry handoff (`finalize_event`).
Staged commits are published through the same function. B100 lifetime / rollback and the B102 one-flush same-t/dt redo
structure are unchanged.

## Source-qualified append-cap recognition

`sediment_bed.journal_recognizer_source()` selects by the bound package digest:

* `eb14da66…` (M94): the unchanged exact-message path (`type(exc) is ValueError`, message with the actual cap);
* `24c6e147…` (accepted typed source): exactly `deep_store.JournalCapacityError` (checked at startup to be the class
  `windowed_transfer` raises) with plain-int `current/new/max_records`, `max_records` = the context's actual cap, the
  unchanged message, `0 < carried <= current <= max < current + new`, and `current - carried + new <= max`
  (otherwise a single-step overflow, which propagates without a flush);
* anything else: `'boundary'` publication is refused at startup; nothing is recognized.

Recognized refusals set a flag only; the flush of the previous accepted state and the one redo of the SAME t / dt run
outside the handler (no traceback alive), with no rejection, halving or counter change. The per-segment publication
record gains `append_cap_recognizer` ONLY for the non-M94 source (M94 records are unchanged).

## Identity, checkpoints and reports

* Run identity: `bed_storage.deep_storage` appears only for a non-default option (default identities unchanged; the
  candidate's own source digest naturally differs from the parent's).
* Checkpoint manifest: optional `deep_storage` = {`deep_chunk_encoding`, `deep_overlay_max_bytes`,
  `store_format_version`, `physical_manifest_digest`, `logical_manifest_digest`} written only for a non-default option.
  A default bundle is byte-compatible (verified against the parent on M94 apart from the named observational timings).
  Resume requires the same encoding AND bound (else refused before any payload read); the loaded deep version must be
  the recorded one (default bundles must name a v1 version). Saved/loaded through MAPLE's v1/v2 version abstraction.
  The held routing source is still restored as saved, never the latest bed. No cross-source wet migration is claimed.
* Summary: `deep_storage` (non-default only) with the final physical / logical digests and MAPLE deep-store / overlay
  counter deltas differenced around THIS invocation's event loop (operational, per invocation, not restart state).

## External store retention

Overlay versions reference earlier base chunks; every referenced file must be retained with the store for restart
(as for full storage). Nothing is deleted or rewritten; GC is out of scope.

## Executed / not executed (see agent_handoffs/tasks/conservative_water/cw_b108_report.md)

Executed on small fixtures only (12×7 cells, 40 voxels, 3 classes): CPU NumPy/Numba and GPU0 CuPy/CUDA full-vs-overlay
complete events (step and boundary publication, keyframe bound), genuine net reroute at the production threshold,
terminal settlement and final forced commit, wet pause / bundle / fresh load / continuation per encoding with a lagging
held routing surface, actual and injected typed cap refusals, failure injection at publication and reroute, refusals;
parent-vs-candidate default-full compatibility on M94; one tiny CPU measurement. Not executed: the windowed runner
entrypoint end-to-end with a real case (Plot1 / RFID / Chastre exceed the fixture limits), CPU↔GPU correspondence,
large stores, full storms, storage-fit or speed qualification, dependency delivery.
