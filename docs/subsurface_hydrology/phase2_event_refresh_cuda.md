# Phase 2 EB3-CUDA — selected completed-event property refresh on the device (author-written, device behaviour UNRUN)

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_event_refresh_cuda_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

`floating_event_refresh_cuda.stage_completed_event_refresh_cuda(cap, state, *, event, <two-pool quota>, batch_cells)` stages (adopts nothing, consumes nothing) the property refresh of a genuinely completed CUDA Pond event from the CURRENT
inventory (device W, J/I, current canonical D) through the accepted pieces: gross-ledger dirty columns formed on the device, the accepted compact reader on only those columns, the EXACT host NumPy tile provider (so unchanged
texture reproduces the held FC/SAT bit for bit; a device-resident provider is a future explicit mode with its own byte-gating proof), the accepted changed-set rule against the CURRENT depth, the owner-free reconcile core
(`native_floating_soil_cuda.reconcile_selected_columns`, factored from `_reconcile_cuda`, which now forwards to it), the shared closure (`inventory_closure_cuda`, factored from `stage_cuda`), a NEW prepared root column/operator
(content identities), exact J/I rebind and the canonical preferential rebind (`fpi.rebind_preferential_column` CUDA branch). Two-pool admission precedes every allocation; the current-device gate precedes every reader/hash/allocation.
The CPU stage (`floating_event_refresh`) is byte-unchanged and still refuses CUDA events. See the task report for traffic, models, tests and limits.


## Independent qualification — 2026-11-13

The staged component is independently accepted on MAPLE 9f466d2 with the exact source manifest in phase2_event_refresh_cuda_acceptance.json. Root ran 32 host controls and 19 actual GPU controls (18 C4 scientific tests plus the C5 moderate memory control, on identical production sources). The 50,003-column component test measured 8,509,440 bytes of incremental device allocation against an 11,180,681-byte payload model; its released pool allocation was identical across three repetitions. Required static-map hashing is reported separately from selected-property/reduction traffic. This is not full-catchment or event/daily lifecycle qualification. The earlier author-only status above describes the implementation report before this independent qualification.
