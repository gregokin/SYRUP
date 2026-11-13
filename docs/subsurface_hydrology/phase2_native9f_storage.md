# Phase 2 E4j — native 9f journal publication pin and hot prefix depth

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_native9f_storage_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

## 1. One explicit native 9f package pin (`sediment_bed.py`)
`NATIVE9F_MAPLE_DIGEST = fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e` (the imported package of the pinned materialization `5d850789…` (per the E3 reports),
MAPLE `9f466d2cfcd52991b24496cf67f4562d6268afa9`) was added to `TYPED_JOURNAL_SOURCE_DIGESTS` and `JOURNAL_RECOGNIZER_SOURCES`
(`typed_journal_capacity_error`). Nothing else changed: exact typed class identity (`deep_store.JournalCapacityError`, exact class, no subclass, no plain
`ValueError`), integer counts, the context's actual cap and the carried-journal rules are the existing ones; an unlisted package digest is still refused.

Audit (read-only `git show` in `/home/okin/MAPLE`, 39bcf79 -> 9f466d2):
* FILE-identical: `surface/voxels/deep_store.py` (sha256 `b03d1503…`, also equal to the materialized runtime file), `water/windowed_step.py`
  (`publish_windowed_result`, `pending_deep_journal`), `surface/voxels/windowed_pending.py` (`publish_pending_column`, `require_published_column`,
  `attach_pending_journal`).
* AST-identical: `JournalCapacityError`, `_rebuild_journal_capacity_error`, `_require_journal_room`, `_resolve_encoding`, `_publish_version(_v2)`,
  `publish_staged_windowed_commit`.
* Changed (reviewed, all additive/inert on the paths used): `windowed_transfer.py` (private `_evidence`, `_avalanche_observation`, `_depth_work` taps, default None;
  `_apply_pending_records` delegates to the new `_apply_pending_segments`), `windowed_commit.py` (`_native_transaction=None` parameter, new native-avalanche staging
  helpers), `windowed_inventory.check_windowed_column_cheap` (preamble split into `_check_windowed_column_layout`), `prefix_cell_reductions.py` (3 additions).
Functional evidence (actual pinned 9f CPU runtime): see the report; the publication/cap-refusal/retry/parent-preservation/overlay/checkpoint suites
(`cw_b102`, `cw_b108`, `cw_b109`, `pond_windowed`, `pond_windowed_checkpoint`, `pond_event`, `pond_checkpoint`) with the pin added.
Limits: `storage_admission.py` remains bound to 39bcf79 only (separate guard, 50 `shared_storage_compat` failures unchanged and out of scope); two historical
M94-only tests of `cw_b102` fail by design on any non-M94 source; no upstream edit; GPU not run.

## 2. Hot Windowed canonical depth (no new helper)
The native `windowed_committed_surface_elevation_m(column, active_layer, geometry)` = `windowed_committed_height_total_kg` (NumPy-order decode of the native prefix
packet; compiled Numba on the host) `/ rho / area` `+ compute_active_layer_actual_thickness_m`. `require_prefix_packets` binds the packet to the column's
base object, deep-store manifest digest, generation, shape/namespace; a stale, cross-bed or missing packet is refused before decoding. Tests prove bitwise equality
with the Dense NumPy oracle (`diagnose_combined_surface_state`) at genesis, after native pickups (window refills from the deep store, BASE > 0), after deposition
(spill/miss, UNPUBLISHED pending journal) and after publication: the packet already reflects the pending journal, so the current depth is valid without publishing. No
epsilon, no alternative grouping, no copied decoder, no deep mirror or hot hashing. On a device the decode needs `trusted=True` after the actual validator (GPU definition only).

Root independently accepted this bounded source update after 411 CPU checks, 88 GPU-enabled publication checks, one actual GPU prefix test and a separate real GPU pending/published oracle control. See phase2_native9f_storage_acceptance.json for limits; the published-boundary validator still refuses pending journals.
