# Dense CUDA external-wind boundary

`floating_wind_handoff_cuda` is a separate, explicit Dense CUDA interface. It
uses the genuine native MAPLE finalizer and the existing CUDA floating-soil
reconciliation; it adds no transport, retention, infiltration or plant law.
Historical NumPy handoff and source profiles retain their defaults.

The initial transaction uses the actual static Pond initializer to retain a
real graph/source and water state, without inventing a completed rain event.
It retires the current owner last. Stale retirement never transfers authority.

Staging requires a completed native wind result, its separately staged complete
bed, matching geometry/classes/masks/device, an empty native wind pool, native
class identities and a matching physical clock. Wind carries both surface-water
leaves bitwise and changes neither W nor J/I except the existing zero-time
capacity reconcile on genuinely changed columns. The applied vegetation record,
per-cell law and covers are preserved. Wind assumes a dry surface for its wind
physics; no moisture effect on threshold or physical water reset is added.

Dirty selection compares resident Dense column bytes in bounded tiles. Current
selected texture/geometry passes through the existing native host provider and
CUDA reconcile. Surface return joins the surface once; bottom export is a
separate named ledger term. A returned-water boundary is `pending_recession`:
only native water completion can make it eligible again. Routing source/graph
are retained for the actual upstream reroute decision.

Separate host/device caps, reserves and real readers are mandatory. The current
device gate precedes readers and arrays. Limits cover grid, voxels, classes,
cohorts, comparison batch, candidate/scan bytes and scalar receipt. Cold native
SHA256 integrity streams use bounded host buffers at stage/adoption; no full
host bed or soil-water mirror is retained. The vegetation projection is borrowed
and never traversed by the integrity walk.

Adoption checks the exact current parent and post-stage content before moving
the lineage head. It records the accepted weak reference before publication,
then consumes/empties the candidate only if that boundary became the head.
Callbacks, tamper/stale/refusal paths leave the accepted parent retryable.

Current committed MAPLE 1dca has separate explicit source and Dense preparation
names: `native_maple_1dca_case_latitude_cuda_v1` and
`native_maple_1dca_dense_cuda_v1`. They bind the immutable exact package descriptor
also used by its CPU profile. They do not admit arbitrary newer MAPLE sources or
rebind the historical 9f/39 CUDA profiles.

Qualification evidence is owned by MASY's `dense_triplet_gpu_20261115` archive:
actual small Dense CUDA wind/storm/daily E/T/weekly growth, matched CPU fields,
standing-water wind, closed-bottom capacity return and refusal/purity controls.
This is a bounded small Dense integration milestone. It does not qualify
Windowed/large-domain performance, all wind variants or combined persisted
restart; `restore_wind_context` refuses the latter explicitly.

The adverse near-saturated closed-bottom standing-water control verifies real
recession initialization, one accepted native step and wind exclusion while
water is active. Full recession for that control exceeded the external 300 s
CPU guard; that attempt remains incomplete. No tolerance or physical law was
changed to complete it. Long wet recession performance is not qualified here.
