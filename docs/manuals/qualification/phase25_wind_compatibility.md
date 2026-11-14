# Phase 2.5 — wind / vegetation compatibility guard for WH1

Scope: one bounded fail-closed guard in `src/maple_syrup/floating_wind_handoff.py` (WH1, Dense NumPy, no-LEAF). No wind or water physics, no scheduler, no new wind–vegetation support.

## Baseline preserved
Published main `953e6fdf85b73e3fdf5a1ac7b0d97dea4df7dab1` added `floating_wind_handoff.py` (sha256 `a289e35b329bc33ec918768133ada9aef1d37089b110d4085fb8c97c4c09156b`); the 169 older production files are byte-identical to the historical baseline and are not touched. The integration overlay receipt (`integration_initial_overlay_receipt.json`, 180 files) records that baseline; its source identity is **not rebound**. Every non-vegetated WH1 behavior is unchanged (the 68 existing WH1 tests pass unmodified).

## Defect (confirmed with a genuine applied boundary)
`stage_wind_handoff` + `adopt_wind_handoff` accepted a boundary whose root carried an applied vegetation generation and minted the new `BoundaryRoot` without the `vegetation` field (`adopt_wind_handoff`, the `lc.BoundaryRoot(...)` call), so the generation record, bound covers and recipe were silently dropped; the refresh paths use configuration scalars rather than the held per-cell law. Reproduction: `wind_builders.started` -> real LEAF P1 + S0 binding applied by `vegetation_lifecycle.apply_vegetation_generation` -> staging and adoption of an unchanged typed wind child: **adopted, `new.root.vegetation is None`**. The wind result in the tests is the mechanical typed `WindEventFinalizationResult` of the repository's own builder, not a genuine MAPLE wind simulation.

## Guard
`_refuse_vegetation(boundary, what)` raises `WindHandoffError` (a `LifecycleError`) when `boundary.root.vegetation is not None` — the owning lifecycle's field, no parsing or duck typing, no LEAF import. It is called by `stage_wind_handoff` (right after the native head/eligibility check, before limits, occurrence parsing, context, child payload, admission or callbacks), `adopt_wind_handoff` (before `before_publish`, admission and the integrity audit) and `restore_wind_context`. Nothing is read, allocated or mutated. A vegetation-aware wind handoff (carrying generation/recipe, bound covers and the per-cell law, refreshing with the held law) is a later qualified interface; positive wind carry is **not** implemented here.

## Evidence
`tests/phase25_wind_compatibility`: real vegetated parents; zero reader calls and unusable-argument sentinels prove nothing was read; head/W/J/I/bed/time/recipe unchanged; a native dry day and hand-back still work after the refusal and keep the generation; the stale native refusal precedes the guard; plain WH1 path and context restore preserved; source scope (only `floating_wind_handoff.py` differs from the receipt) and LEAF-free import. A combined release needs a new receipt for the changed `floating_wind_handoff.py`.
