# Phase 2 / R2a — CPU floating FILE checkpoint (Stage 1: original-template target)

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_floating_checkpoint_cuda_reconciliation_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: author-written CPU/host implementation, not independently reviewed, no GPU. **Not a general long-run or daily checkpoint.** The restart TARGET must be built from the genuine ORIGINAL immutable case (original bed -> cold
native owner -> vertical -> event), so its HELD template is the original one. Stage 2 (a held-context factory from the verified stored specification) and the event-end dirty refresh are later work.

## Profile
New module `floating_checkpoint.py`, new schemas `maple-syrup-floating-pond-checkpoint/1` (dense) and `/2` (published resident-window bed), entry points `save_floating_pond_checkpoint` / `load_floating_pond_checkpoint` (explicit
`available_bytes_fn`, `reserve_bytes`, `max_bytes`). The historical `maple-syrup-pond-checkpoint/1|2` codec cannot serve a floating event (its `_validate_state` builds the bucket hydraulics from `event.column`, None for a floating
event, and refuses before any byte is written; its arrays carry no W/J/I/D/held maps); it is untouched and a regression test pins that refusal. Directory: `floating_pond_checkpoint.json`, `maple_state.npz` (native MAPLE snapshot
of the CURRENT bed, a companion with `syrup_floating_pond_checkpoint_schema` provenance; windowed: published deep-store references only), `pond_continuation.npz`.

Saved: all event-state arrays (reused `pond_checkpoint._extract/_specs`), current W `(L,N)`, J, cumulative intake I, current canonical depth D, soil clock, and the HELD facts: prepared FC/SAT maps, D0, layer-0 Ks record, root column /
matrix / preferential CONTENT identities, owner config sha, generation and the native source witness of the ORIGINAL property source. No random view/generation tokens enter any durable digest. The restore never derives a property
from the checkpoint (eroded/deposited) bed and validates W against the CURRENT geometry, never against D0 (a test round-trips a state whose W exceeds the original-D0 capacity after a real terminal deposition).

## Verification chain (all refusable work precedes the single registration)
1. strict manifest reader (own keys; bounded JSON helpers reused), seal, profile, expected identity, source pair, CPU backends (a CUDA event/vertical is refused before any read or transfer), binding (historical static-input
binding plus the vertical identity), floating facts, deep record. 2. streamed file hashes, archive directory, declared table, `.npy` header audits (windowed: the grid-derived native layout; dense: the manifest's native voxel
dimensions cross-validated against the ACTUAL snapshot member headers). 3. composed admission. 4. native snapshot (windowed: MAPLE full published validator, deep-store version bound by a floating-schema copy of the provenance
check), companion arrays through the fixed whitelist. 5. held-template check against the target. 6. typed reconstruction (`_build_state`) of the candidate parent. 7. `VerifiedFloatingInputs` minted (only by the loader, one-shot,
claims no physical acceptance). 8. `floating_resume.prepare_from_verified_inputs` (R1a preparation factored into ONE shared body used by restore too: official view, continued state on the CURRENT geometry, history rebind, one-shot
target-bound prepared record). 9. inside `vertical.resume_scope(prepared)`: cold native canonical-depth check of every active column, the independent floating validation `_validate_floating_state` (the historical validator
verbatim except the declared hydraulic block, guarded by a mechanical AST drift test), scalar equality, `event._check_state`, native water/sediment audits. 10. `adopt_resumed` LAST. Any failure leaves the registry, stats,
accepted payloads and the target untouched.

## Memory
Stage 0 (before the manifest is read): `3 * manifest bytes + hash/header buffers + reserve`. Stage 1 (before any payload array is loaded), from DECLARED then header-VERIFIED dimensions: `3 * companion table bytes + 2 * native snapshot
bytes + (48 N + 4 N nc nb + 8 N nc) 8 (topology/phase/audit working sets) + 8 N (depth scratch) + hash/header buffers + the R1a prepare model + one reserve`; the native canonical-depth reader self-admits its compact need against
the same explicit reader. Exclusions: interpreter/JIT/allocator pool, the already-live target event, whole-process memory. Need 0 refuses before the manifest is read; a need that fails stage 1 refuses before any payload array
is allocated or loaded (the bounded manifest/hash/header reads of stage 0 are not hidden). Save admits the same model from the state's actual dimensions before any host copy.

## Limits
CPU (NumPy/Numba) only; Windowed requires a PUBLISHED column (publish through the event's own storage policy first; no pending trust); no nominal-daily cursor/N policy; no CUDA; the file profile does not claim a fresh target can
be built without the original-case factory; gross dirty tracking and the between-event refresh handoff are not part of this profile (the property-source freshness is a separate fact).
