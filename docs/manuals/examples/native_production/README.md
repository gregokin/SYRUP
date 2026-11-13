# Production (committed-source) run — example template

Template for `masy.native_water_run.v2` with `source.mode: production_committed`. The case, scenario, weather, layered-water parameters, catalog and plant CSV (+ companion metadata) are REUSED from
`../native_leaf_calendar_cli` (no copied physics); `tools/masy_production.py render-config` copies them outside the source trees and fills the placeholders.

Nothing here names a commit: the recipe/receipt come from the published release metadata (C2) and the store/lock/capsule from running the existing V1 `prepare` and `tools/masy_leaf_capsule.py seal`
at the exact C1 commits. See `docs/native_production.md`. A template is not a qualification; with a draft/self-generated recipe the run refuses (`recipe_unreviewed`).
