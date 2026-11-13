# Phase 2 — continued-origin CPU physical checkpoint (CC1; candidate, uncommitted)

Module: `maple_syrup.floating_continued_checkpoint` (+ bounded hooks in `floating_checkpoint`). CPU NumPy/Numba, Dense and published Windowed. Not GPU, not an outer/boundary/daily container.

Problem. A state continued from a lifecycle boundary (`begin_occurrence` -> `PondEvent.continued_state`) carries the LAST actual routing topology (generation >= 1 after a real reroute), the held source and the GLOBAL original datum, with fresh event-local counters (`n_refreshes == 0`). The historical floating profiles (`maple-syrup-floating-pond-checkpoint/1-6`) demand `datum == event initial topology source` (refused first: "state does not belong to the event's ORIGINAL initial topology / source datum") and `n_refreshes == topology.generation`. Neither is relaxed.

Profile. `maple-syrup-floating-continued-pond-checkpoint/1` (Dense) and `/2` (Windowed), profile `floating-vertical-continued-origin/1`; written/read only with `original_topology=` (generation-0 original case, caller-attested). Bound distinctly: the carried routing origin (`binding.initial_topology`, also stored as `co_routing_source`) and the original datum (`binding.continued_origin`, `continued_origin.datum_full_sha256`). Local counter rule: `n_refreshes == topology.generation - origin.generation`; a state with no local reroute carries exactly the carried topology. Everything else (native bed validators, geometry gate against the CURRENT bed, phase partition, event audit, canonical depth, held template, capability chain, admission) is the historical code.

API. `save_floating_continued_checkpoint(...)`, `read_continued_origin(...)` -> `ContinuedOrigin` (pure value; one small array), `origin_topology(origin, static_topology)`, `load_floating_continued_checkpoint(...)`.

Limits. No quiet/completion proof is claimed for the fresh monitor; no outer H0 container (its factory is one-argument), cursor, scheduler or GPU for this profile; `datum_committed` is carried from the file (shape-attested only); a generation-2 continued state (second real reroute inside a continued event) is not fixtured; CPU-only memory models (deep/wide CPU admission) are the historical ones.


## Independent bounded acceptance — 2026-11-13

The C1 child profile passed 104 Root checks: 39 continued-origin tests (including four fresh-process backend/storage variants), 39 historical checkpoint regressions and 26 preservation guards. An independently reproduced header/manifest mismatch now refuses before array decoding. Exact source and inherited-test limits are in phase2_continued_checkpoint_acceptance.json. This does not qualify a daily-cursor or complete coupled-run restart, CUDA continued origins, or a second local rerouting generation.
