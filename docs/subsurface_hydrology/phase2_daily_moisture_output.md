# Phase 2 G2a — daily final-close hook for the every-N-days moisture output

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_daily_moisture_output_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

`daily_moisture_output.close_final_day(lop, interevent_state, *, expected_parent, output=None)` calls the EXISTING `water_lifecycle.close_interevent`
(forwarding the CALLER's required `expected_parent` capability; it is never re-minted for the state argument; `output` must be None or an exact `MoistureOutput`) on the interevent of a daily lineage and, only when a G0 `MoistureOutput` is given, notifies it at that accepted final
post-T close; it returns `DailyCloseResult(state=<closed boundary>, notice)`. `create_daily_moisture_output` / `resume_daily_moisture_output` delegate
to G0 `create` / `resume` (explicit positive `every_n_days`, no default; explicit `MoistureOutputLimits`). Off by absence: no output => no read,
transfer or write. Cadence, storage, quotas, publication and replay are G0's; nominal days are the daily cursor ordinals (storm/recession
duration is additional). Output failure: the G0 exception propagates, no closed state is returned, the accepted interevent parent is unchanged and
a retry with it verifies a sealed due day instead of overwriting (a duplicate of an already notified NON-due day is refused: reuse the returned state).
No E/T/growth/drain/clock/debit is added; plants call the actual `request_withdrawal` before the close. LIMITS: fixed-full-porous common-column
profile only (G0 schema); the floating porous-depth hook (W + canonical D + generations via `floating_moisture_frame`) is pending the floating daily
owner. No CLI key, scheduler or weather. Example: `docs/manuals/examples/native_daily_moisture_output/run_example.py` (genuine native file-case factory).

## Correction01: transfer contract and caller authority
* Transfers: the existing native `close_interevent` performs its own audits and small device reads (Root measured 64 B in 3 reads on CUDA). That
  baseline is not a diagnostic regression and is not removed or weakened. The contract is zero EXTRA transfers from the disabled / not-due diagnostics
  RELATIVE to a direct native close of the same accepted parent (all MAPLE counter fields equal), and for a due day exactly `8*L*N` extra
  device-to-host bytes in `L*ceil(N/(chunk_bytes//8))` pieces with no extra host-to-device (gated CUDA definition; unrun by the author).
* `create_daily_moisture_output`, `resume_daily_moisture_output` and `close_final_day` all REQUIRE the caller's `expected_parent`; a wrong-instance,
  stale or alternate-branch capability is refused by the native check before any audit, model work or I/O.
* Restart qualification: the file-case fresh-process test uses the genuine native file-case factory (separate `factory_full` / `factory_resume`
  source-guard paths), NumPy and Numba; the older `fresh_worker.py` uses the manual native Dense fixture with the real V3 codec and is NOT the file-case
  factory.

Root accepted this bounded native fixed-column daily hook after 22 CPU tests, one actual GPU test, and independent genuine file-case NumPy/Numba fresh-process V3 continuation. All replay physical/state/water hashes match. See phase2_daily_moisture_output_acceptance.json; floating integration and CLI configuration remain later work.
