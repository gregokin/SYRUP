# Pond event on an actual MAPLE resident-window bed (core)

Opt-in: `PondEventControl(bed_storage="windowed", ...)`. The dense pond event, the legacy and method-5 defaults and every dense
digest are unchanged. The implementation author used file tools only; executed independent checks are recorded in [qualification](qualification.md).

## What runs where

| Stage | Dense (unchanged) | Windowed |
|---|---|---|
| wet pickup -> transport -> deposit/export | two `apply_bed_demand` calls | MAPLE paired transaction (`sediment_bed.apply_bed_pickup_transport_deposit` / `paired_transaction_entry`), post-pickup storage private |
| dry splash pickup -> redistribution -> deposit/export | two calls | same paired entry; `between(view)` redistributes the ACTUAL pickup |
| geometry gate | `diagnose_combined_surface_state` (depth sum) | `windowed_committed_surface_elevation_m(trusted_packet=True)`: height packets + window + active layer; no deep read |
| accounting commit | `commit_topography` | `stage_windowed_topography` + `make_windowed_water_depth_commit_callback`, `trusted_packet=True`, **no D4 graph** |
| geometry refresh | max active \|current - held\| >= `pending_elevation_trigger_m` (0.00075 m, `>=`) | identical, the only trigger |
| deep storage | n/a | `publish_pending_column` via `publish_bed_storage` (context encoding and cap options) |

A missing paired entry (older MAPLE) is an **unsupported dependency** (`PondBedError` at construction); there is no two-call
fallback. Any other import or attribute error from `paired_transaction_entry` propagates. Avalanching stays OFF, square cells,
solver-owned `constant_depth`, exchange history off.

## Control (all `repr=False`, so dense digests are unchanged)

* `bed_storage`: `"dense"` (default) or `"windowed"`. Declared, never inferred; a bed of the other kind is refused (`PondBedError`).
  A dense event still refuses `window_max_journal_records` and non-default deep encodings exactly as before.
* `deep_publication`: `"step"` (default) publishes the journal after every accepted step; `"boundary"` carries it.
* `deep_publication_high_water_records`: required (positive int, strictly below the ACTUAL journal cap `resolved_journal_cap`)
  for `"boundary"`, forbidden otherwise. Construction also requires a recognised append-cap source
  (`journal_recognizer_source()`; the accepted typed `JournalCapacityError` is checked structurally).

## Pending-journal lifecycle (`"boundary"`)

* `PondEvent.step` is pure and may return a state whose `bed.voxel_column.pending_deep` is MAPLE's valid, immutable, unpublished
  journal. It is NOT restartable. `PondStepRecord.storage` = `{"pending_records", "published"}`.
* `PondEvent.evolve(..., publish_at_return=True)` publishes at: the high-water mark (after an accepted step, `n_high_water`),
  the single capacity flush, and return. `PondEvolveResult.storage` reports the counts. `publish_at_return=False` lets a caller
  chain `evolve` calls and take responsibility for publishing later.
* `PondEvent.publish_storage(state)` is the explicit no-physics boundary (for the checkpoint wrapper): returns the **same object**
  when nothing is pending, else a state identical except `bed.voxel_column` binding and the storage counters. Hydraulics,
  topology, held source, ledgers, quiet monitor and clocks are the same objects; it never refreshes geometry.
* `finalize`/`complete`: the terminal deposition, the (forced) final commit and the dry reset run on the possibly pending state;
  the handoff's bed is **always published** (checked; `PondEventError` otherwise).
* Counters (`PondCounters`, int, 0 for dense): `n_storage_publications`, `n_journal_flushes`, `max_carried_records`.

## Capacity retry (evolve and finalize)

Only the native append-cap refusal recognised by `sediment_bed.is_carried_append_cap_refusal` (typed `JournalCapacityError` with
consistent counts, the context's real cap, a step that fits a clean journal; or the pinned M94 message) **while the previous
accepted state carries a journal** triggers: publish that PRIOR accepted state once (`reason="flush"`, distinct from candidate
publication), then redo the same step (same clock, rain rate, dt, snap, retry counters, adaptive cap). The flush never changes
dt. A genuine `HydraulicStepRejected`/`TransportStepRejected` of the redo enters the ordinary bounded dt adaptation (the capacity
refusal can precede the transport); the flush is never repeated for that accepted step. An unrelated `ValueError`, `MemoryError`,
a refusal from a clean journal (a step too large) or a second capacity refusal propagates.

Publication ordering: in `step`, `_publish` and `finalize` every host-side computation (counters, record, state construction,
audits) runs on the unpublished bed first; the deep-store write is the last operation, so a late host error (for example a
malformed publicly supplied counter) leaves no speculative version. The only earlier write is the explicit prior-state flush.

## Splash audit

`SplashExchange.pickup` is MAPLE's `PickupView` on a windowed bed. It has every attribute the event and the audit read except
`bed_change_by_class_kg`, so `SplashExchangeAudit.net_bed_change_by_class_kg` and `conservation_residual_by_class_kg` are
`None` there (never fabricated; `scalar_totals()` skips them). All face and cell accounting is retained.

## Not here

Checkpoint save/load is implemented separately in [checkpoint.md](checkpoint.md). A store version alone is not a restart: the resident window remains authoritative inside its range. Densification and an alternative bed authority are not introduced.
