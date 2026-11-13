# Phase 2 G2a — daily final-close hook for the every-N-days moisture output

> **Current status pointer (publication refresh, 2026-11-13).** The text below is the original **G2a record for the Phase 1.5 FIXED-full-porous profile** (`daily_moisture_output`, G0 schema) and is preserved unchanged, including its "author-written / unrun" wording and its statement that the floating hook was pending. It is **not** the floating-profile writer: the Phase 2 floating writer is the separate `floating_daily_output.FloatingMoistureOutput` with the `floating_moisture_frame` format (see `phase2_floating_moisture_frame.md`), and G2a is not claimed to be reused unchanged as that writer. Root's qualification of this fixed-profile component is `phase2_daily_moisture_output_acceptance.json`; where the two differ, that receipt governs. The combined whole-Phase-2 qualification is [phase2_qualification.json](phase2_qualification.json) / [phase2_qualification.md](phase2_qualification.md) and the source publication is [phase2_publication.json](phase2_publication.json) (bounded native API; no field validation; integration into `main` published at merge `6f8bab2ab045a3e6299b33ad2a3b0fc170ab4f06`).

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
