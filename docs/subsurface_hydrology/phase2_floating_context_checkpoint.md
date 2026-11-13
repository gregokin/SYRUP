# Phase 2 / H2 — CPU context container (`floating_context_checkpoint.py`)

Profile `maple-syrup-floating-context-checkpoint/1`: directory `floating_context_checkpoint.json` (strict, sealed LAST) + `context_arrays.npz` (H0 record arrays) + `child/` (the UNCHANGED `floating_checkpoint` /5 dense or /6 windowed
directory, saved/loaded as is; Windowed keeps deep-store references only). Outer manifest binds source pair, CPU backend, expected identity, H0 record meta (strict JSON, closed keys) and array table, size+SHA-256 of every file
(outer arrays and every child file), bed storage. Save: stage in a sibling temp dir, `export_held_context` (raw `PreferentialSpec` required when a preferential operator exists; never inferred), write arrays, release the record, then
`fc.save`, hash, seal, atomic no-overwrite rename; failures remove only the own temp dir. Load: bounded manifest read -> source/identity/backend/file table (regular files only, no symlinks/unknown entries) -> streamed hashes ->
zip/npy header cross-check against the table -> `record_from_parts` / `reconstruct_held_context` -> the caller's static `factory(held_context) -> PondEvent` (outside every mint window; must use `from_held_context` of exactly that context, no
owner) -> unchanged `fc.load_floating_pond_checkpoint` (strict child binding refuses a wrong config/source/topology/forcing/Ks/pref/held maps; adoption last).

Static recipe example: `tests/phase2_floating_context_checkpoint/static_case.py` (terrain array, grid, forcing, sediment/splash params via actual public MAPLE constructors; no original bed/owner/provider). Memory: the H0 model is admitted and released before
the child model; held maps O(L N) exist twice only while the record lives. CPU only; no daily/wind/scheduler. Limits: the child's own limits (original-case-equivalent static factory, bucket retained states) apply; no GPU file claim.
