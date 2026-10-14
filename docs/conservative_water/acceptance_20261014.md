# Initial conservative water component — qualified acceptance, 2026-10-14

The reviewed water runtime and reproducible MAPLE dependency are adopted in the **working tree** and independently qualified on CPU/Numba and GPU/CUDA for the configurations below. This closes the current implementation/adoption program within its declared scope. The accepted source, tests, dependency recipe and documentation are included in this commit. Full Chastre storm capacity and universal CPU/GPU equivalence are not established.

## Accepted implementation

Actual MAPLE owns the evolving active layer, voxel bed, availability, water inventory, accounting and immutable deep storage. SYRUP adds rainfall/infiltration, MAHLERAN-like depth/routing and conservative wet detachment, transport and deposition. Water and sediment exchange at physical timesteps. Hydraulic rerouting occurs only when the maximum per-cell net displacement from the last routing source reaches MAPLE's **0.00075 m** trigger (`>=`); time, turnover, checkpoints and forced accounting alone never reroute. The routing source survives and is validated across restart.

Deep storage can use exact cumulative final-value overlays, including tiny changes and signed zeros. Old referenced files remain intact. Full encoding remains available and is the API default; overlays and boundary publication are explicit options. Completion checks rain-off, flow and sediment, accounts for the explicit dry reset and preserves sediment. Terminal-water storage and flow-end mobile deposition remain explicit assumptions. The default benchmark is still the separate legacy replay; conservative evolving-bed runs are explicit. No wind physics is copied into SYRUP.

The accepted runtime is 55 files, package `5f85a989…`; the MAPLE package is `24c6e147…`. The Git-revision-plus-patch recipe reproduces all **341 runtime files**. Runtime-only bundle/cache `7ee10bd8…` is distinct from full-project qualification artifact `b8fa43a3…`, with identical MAPLE package bytes. See [usage](../conservative_water_usage.md) and [recipe](../maple_water_dependency_recipe.md).

## Independent checks of the adopted tree

| Check | Result |
|---|---|
| Four default conservative CPU families plus Phase 7b | 294 passed; 86 explicit device skips |
| CPU net-routing guards | 33 passed; 42 explicit device skips |
| Recipe tests | 58 passed, no skips |
| Reconstruction and both existing-environment imports | 341 files exact; actual ROOT + intended MAPLE cache, no installation |
| GPU storage/event checks | 7 passed; one documented matrix skip (keyframe bound exercised under step publication only) |
| GPU dense/fused lifetime and pit-event completion checks | 3 passed, no skips |
| CPU fresh-process wet restart | 119 numeric arrays and physical metadata exact; actual new deep file/version after resume; 19 controls pass |
| GPU fresh-process wet restart | 140 arrays and physical metadata exact; actual new deep file/version after resume; 19 controls pass |

Every GPU child used freshly idle allocated GPU0 and the original 9,092,045,209-byte pool ceiling. Test corrections use genuine physical net crossings at the normal trigger, preserve weaker storms as negative controls, and retain conservation/phase checks. A forged final-reroute test was rejected and replaced with actual MAPLE deposition producing a 1.6 mm change before testing completion. See [test contracts](current_test_contracts.md).

## Storm and storage evidence

Complete Plot1 CPU/GPU storms and own-wet restarts were qualified on immutable copies with the same accepted runtime/package bytes. Full/overlay comparisons reproduce 908 CPU and 1,089 GPU arrays; own-wet continuations reproduce 549/650 arrays. They finish at 4,819 seconds with closed budgets and explicit dry-reset accounting.

Rotated Chastre (886×1780, 1,140,189 active cells) passed its **300-second wet GPU slice**, and its own 240-to300-second restart matches 184 arrays and strict physical metadata. Public deep-file validation and all 15 manifest links pass. All 291 saved numeric members match the earlier same-GPU reference exactly. These large runs imported the immutable qualification sources; actual-root/import parity is independently verified above.

New deep data fell from **55.25 GB to 6.75 GB** (8.18× less), excluding initial storage and metadata. Earlier storage had historical deduplication. Loop/checkpoint observations were 34.0 versus 28.5 minutes; initialization/cache/source conditions differ, so this is not a controlled end-to-end speedup. No old referenced state was deleted.

## Qualifications retained

- A complete Chastre storm's GPU/host/disk/CSR-capacity fit is unproven; a 300-second slice does not establish it.
- Current Plot1 CPU/GPU comparison retains **89 strict failed records**, and the earlier rotated Chastre comparison retains four/seven at 240/300 seconds. Physical fields and decisions pass the recorded bounds; residual/accounting and some storage placement/version records differ. No tolerance or failed-field waiver was introduced. This is qualified physical agreement, not blanket equivalence.
- Eight off-default historical B20/B54 source-identity/AST proofs remain recorded failures against superseded implementations. Current affected default contracts pass; historical gates were not weakened.
- Terminal storage has no pit overtopping/backwater; flood validity is outside this qualification. If a final commit invalidates quiet, the handoff is refused with the input wet; automatic renewed recession is not implemented.
- The native Fortran recession-water surplus remains documented only. No native-Fortran equivalence is claimed for rotated/resampled terrain.
- Wind handoff, direct dry-cell splash, growth, full ecohydrology/ET and nutrients remain outside this initial component.

Authoritative receipt: `agent_handoffs/tasks/conservative_water/cw_e48_root_primary_adoption_acceptance.json`, with raw process records, JUnit, source/preservation proofs and linked large-run evidence. All failed/interrupted attempts and pre-edit baselines remain preserved.
