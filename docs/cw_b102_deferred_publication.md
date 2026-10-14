# CW-B102 — opt-in deferred deep publication (candidate; not adopted)

**Base.** The accepted B100 freeze (`12284822…`) plus this change, implementing the root-accepted B101C1 design
(`cw_b101c1_independent_design_acceptance.json`). MAPLE M94 (`eb14da66…`) is unchanged.

## What changes

**New control (opt-in).** `SedimentEventControl.deep_publication` (`"step"` default | `"boundary"`) with
`deep_publication_high_water_records`.

**Default `"step"`.** Every accepted step and commit publishes the actual MAPLE pending deep journal as a new
deep-store version, exactly as before. Default identities, bundles, public hooks and physics are unchanged. The ONE
visible addition is a new `SedimentEventResult.deep_publication` field that is `None` in this mode.

**`"boundary"`.**

*Carrying.* The MAPLE pending journal is carried by the accepted in-memory state across physical steps and actual MAPLE
staged commits. Commits still happen immediately, at the same times; only deep-file publication is deferred.

*When it publishes.* Only the CURRENT accepted state's journal is ever published:

- at every report-boundary row, before the forced-final-commit check and before `on_boundary`, and again after a
  forced commit;
- before return;
- when the carried journal reaches `deep_publication_high_water_records`;
- in ONE bounded flush before a redo.

*Flush / redo.* The redo happens only after M94's exact append-cap refusal
(`pending journal exceeds max_journal_records=<actual cap>`, plain `ValueError`), raised while the PREVIOUS accepted
state carries a journal, under the pinned M94 package digest. The previous accepted (validated) state is published,
then the SAME attempt is redone (same `t` / `dt` / snap). Physical retries, rejections and `dt` are untouched. Any
further refusal (nothing is carried any more), the oversized-prior, canonical and apply caps, and every other error
propagate unchanged. A genuine single-step overflow still refuses.

*Callbacks.*
- The public `on_accepted_step` hook (always published states) is refused before any work.
- `complete_event` observes quiet progress through the PRIVATE `_staged_accepted_observer`, which reads only resident
  water / storm / graph fields.
- Checkpoints, pause, early stop, return, the terminal transaction and the dry handoff always see published states.

## Entry validation (before any allocation or physics)

- callback combinations;
- a resident-window bed;
- `high_water < the actual journal cap`, which is `context.window_max_journal_records`, else M94's
  `DEFAULT_MAX_JOURNAL_RECORDS` = 1,000,000;
- the bound MAPLE package is the pinned M94.

`validated()` checks the policy fields and the CSR storage.

## Fields that intentionally differ between policies

- per-step deep version handle / `generation` / `manifest_digest` / `pending_deep`;
- checkpoint snapshot lineage metadata;
- the identity / control keys (present only when non-default);
- deep-store counters and files;
- `SedimentEventResult.deep_publication`, the per-execution-segment record of the LAST evolve call, also copied into
  the runner summary. It is never a cumulative restart total.

Physical arrays, diagnostics, commits and logical columns are identical.

## Evidence

Exact commands, logs, probes and results are in `agent_handoffs/tasks/conservative_water/cw_b102_*`.

**The fixture experiment (seeded one-ULP rooms):**
- the pending journal carries exactly through steps and two staged commits;
- one-ULP rooms fill once (step 1, 31,744 records, all from the miss writer);
- exact-capacity controls are unchanged;
- one publication either way in this fixture, so **no write saving is demonstrated by this small fixture**.

## Limits

- No large run, fit or storage-saving claim.
- The flush / redo protocol is exercised with an injected exact-text refusal; a REAL M94 single-step overflow verifies
  the recognizer text.
- A natural carried-journal overflow does not occur in the small fixture.
- Relying on pinned M94 message text is a documented interim. A structured MAPLE exception would be a separate
  reviewed task.
- Deep-store I/O failures can leave orphan content-addressed chunk files without a manifest. They are reported and
  never cleaned.

## CW-B102C1 corrections (root closed review)

**1. Flush outside the handler.**
- The recognized carried-journal append-cap refusal now only sets a scalar flag inside `except ValueError`.
- The flush of the PREVIOUS accepted state, and the redo of the same attempt (same `t` / `dt` / snap), run after the handler has exited. Neither the exception nor its traceback (the failed attempt's frames and arrays) is alive during publication or the redo.
- After a flush, ANY exception of the redo propagates unchanged, including `RoutingStepRejected` / `TransportStepRejected`: no rejection bookkeeping, no `dt` halving, no fourth attempt.
- A first ordinary rejection keeps the default retry.
- At most one flush per accepted step.

**2. Entry order.** The pure callback / policy combination checks run before `_check_event_inputs`, the first allocation-producing validation. The state / context checks (windowed bed, actual journal cap, MAPLE binding) follow it, because they need the types it validates.

**3. High-water is a trigger, not a bound.** `deep_publication_high_water_records` is evaluated only AFTER an accepted step's physical work and commit. The carried journal can therefore exceed it during the next step, up to the actual MAPLE journal cap. Host journal memory is bounded by that cap plus MAPLE's legitimate transient copies, not by `h`: `PendingDeepBinding` copies its journal, a staged commit re-attaches a copy, and gathers apply overlays.

**4. Storage contract.** `tests/cw_b102/b102c1_storage.py` is a SYNTHETIC storage sequence on actual M94 APIs: staggered fills in shared chunks, then a later read and overwrite of pending keys. It compares per-transaction publication with a single deferred publication. On that sequence deferred publication wrote fewer version / chunk files and bytes, with bit-identical logical, resident, final and inventory values. It is NOT a storm measurement.

**5. Cross-backend comparison.** CPU / GPU arrays are compared quantitatively with the accepted comparator's declared bounds (B100, step, boundary). Exact cross-backend byte equality is not a MAPLE / F9 acceptance requirement: the accepted comparators use declared bounds. The original B102 bit-exact hash checks remain recorded FAILs.

**6. Runner exposure.** The runner exposure is API-only: `run_complete_case(deep_publication=, deep_publication_high_water_records=)` and `run_complete_windowed(**options)`. No CLI flags were added.
