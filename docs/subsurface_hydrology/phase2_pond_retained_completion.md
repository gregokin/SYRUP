# Phase 2 E4a — shared terminal settlement and retained-water Pond completion

`PondEvent._settle_terminal(state, rt)` is the legacy terminal sediment settlement (deposit the actual mobile pool through MAPLE with the
single recognised append-cap flush, final commit / net-only reroute, empty-ledger check, sediment ledger, counters, canonical phase)
extracted VERBATIM from `finalize`; `finalize` / `complete` call it and keep their dry-reset physics, numerics, counters, audits, exceptions
and native publication order (tested bitwise against the archived pre-edit source, plus the existing pond/windowed/checkpoint suites with
identical results). `PondEvent.settle_retaining_water(state) -> PondRetainedHandoff` is the NEW explicit method: same required initial gates
(rain off, no flow/overcrest, a completed continuous quiet hold; a resource cap is never a completion) and pre-audit, the same settlement,
then NO dry reset: surface depth, soil water, faces, history, `bed.water` and the clock are untouched; no E/T/drainage/growth/time.
Result `status`: `complete` (gates hold again, ledger empty, nothing mobile, no pending deep journal, audits close; a wet retained state, never
dry) or `requires_recession` (the terminal terrain woke the flow: a valid STAGED wet state with only the quiet monitor reset; the caller evolves
rain 0 and calls the method again; storage follows the event's own `deep_publication` policy). Windowed publication is the LAST operation after
every refusable check; any refusal leaves the accepted parent and prior store versions unchanged. `PondRetainedHandoff` records config/forcing
SHA, time and topology generation (a descriptive record, not a restart schema). No field was added to `PondEventControl`/`PondEventState`
(config SHA and old digests unchanged). Not here: floating vertical water, depth/capacity reconcile hook (E4b), checkpoint, LEAF, wind, GPU.

## Correction01
* A genuine net-elevation routing GEOMETRY REFRESH at the terminal commit (`term.refreshed`) invalidates the previous quiet certificate: the
  instantaneous gates use the OLD face fluxes (exactly 0 in a closed pond) and cannot see the new gradient (Root's real probe: a 1 kg deposit left
  qmax 1.1e-5 m2/s after one real step against a 1e-10 tolerance). `settle_retaining_water` now returns `requires_recession` with
  `reason == "routing_geometry_refreshed"` (or `"quiet_gates_not_held"`), `accounting["wake_reason"]`, only the monitor reset; routing is still
  triggered by the existing net-elevation threshold only. The shared settlement worker and legacy `finalize`/`complete` are unchanged.
* `requires_recession` publication receipt: counters are updated and the audit taken on the UNPUBLISHED candidate; publication (step policy) is last and
  only `pending_journal_records` / `deep_store_version_id` of the receipt change, so `accounting["final_audit"]` equals `audit(returned state)`.
* Recorded limits: with the DEFAULT policy (max unit discharge 1e-10 m2/s) real recession on the 1 kg fixture did not recertify within 3600 s of real steps;
  with the explicit policy `max_unit_discharge_m2_s = 2e-6, max_depth_change_m = 1e-4` a fresh hold is reached at t = 250.75 s and the second call completes.
  Windowed qualification is `deep_publication="step"` only; the pinned 9f source refuses the `boundary` policy and the `overlay` encoding (pre-existing, see
  `completion_program_20261113/e4a_baseline_failure_reasons_pre_edit.txt`); that source qualification is a required later E4 follow-up, not claimed here.
