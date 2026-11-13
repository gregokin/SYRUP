# Phase 2 — CUDA continued-origin child profile (CC2; candidate, uncommitted, GPU-unrun)

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_continued_checkpoint_cuda_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Family: schemas `maple-syrup-floating-continued-cuda-pond-checkpoint/1` (Dense device bed) and `/2` (native Windowed), profile `floating-vertical-continued-origin/2-cuda`; manifest = CUDA `/7|/8` closed keys + `continued_origin`; companion + host array `co_routing_source` (O(N padded), modelled in both pools' declared table bytes). Old CUDA `/7|/8`, CPU `/1-/6` and CPU continued `/1|2` profiles are unchanged and refuse this family (and it them).

API: `floating_continued_checkpoint_cuda.save_/load_floating_continued_checkpoint_cuda(..., original_topology=)` (wrappers over `floating_checkpoint_cuda.save/load_floating_pond_checkpoint_cuda(..., original_topology=None)`), `read_continued_origin_cuda` (host metadata reader: strict manifest, SourcePair, kernel descriptor, admission, companion hash, zip + table + every `.npy` header audit BEFORE decoding the one small array), `origin_topology` (CPU CC1). Rules and validator copy are the CPU CC1 ones (datum attested against the caller's generation-0 original; `n_refreshes == generation - carried generation`).
Unchanged: device gate FIRST, two-pool admission before D2H/H2D/read, SourcePair/kernel guards, Windowed published resident window + external deep reference + mapping-cache bound, `VerifiedFloatingInputsCuda` -> prepare -> resume_scope -> adopt LAST.
Limits: no device pass claimed; GPU fixtures need the composed GPU lifecycle (absent here); no generation-2 fixture; no outer/lifecycle/cursor codec; no quiet/completion proof.


## Independent bounded acceptance — 2026-11-13

Root qualified 199 CPU/host checks and 10 actual GPU controls (7 C1 plus 3 C2 rechecks on identical production). Dense and native published Windowed generation-1 continued states round-trip and continue natively. Strict historical families, source/kernel/deep-store guards and pre-allocation checks remain. Exact sources and limits are in phase2_continued_checkpoint_cuda_acceptance.json. The fresh-process fixture still builds an original case during setup; a recipe avoiding that allocation is a separate qualification. This is not a daily-cursor restart or a complete coupled simulation.


## Fresh restore without constructing the original landscape

A later test-only static recipe passed 37 Root host checks, including four fresh CPU variants, and both actual GPU fresh-process variants. The load worker disables original-bed, owner, texture/refresh and probe constructors before reading the recipe. Genuine public MAPLE context constructors rebuild static parameters; verified H0 data supplies the current soil law and the physical checkpoint supplies current water and landscape. Four subsequent GPU steps match uninterrupted continuation bit for bit. The [supplemental acceptance](phase2_continued_static_recipe_acceptance.json) supersedes the earlier test-setup limitation for this particular closed-lake recipe. It does not qualify a generic case loader, daily-cursor restart or large-domain memory fit.
