# Bounded windowed device-water qualification

This interface permits a separately controlled physical experiment for the
labelled synthetic Small 4 by 4 landscape: 16 cells, one sediment class, six
vertical voxels and a three-voxel resident window. It is an
implementation awaiting independent qualification. It supplies no production,
arbitrary-landscape, restart, calibration or whole-allocation guarantee.

The existing strict device-water interface remains a distinct route. Its
`DeviceWaterRuntimeBounds`, `ExternalRuntimeBound`, primitive constructor
accounting and admission result retain their original meaning. Their missing
contracts cannot be supplied by this experiment. The strict entry rejects a
qualification budget; the qualification entry rejects the strict operand.

## Explicit budget and Source handoff

`DeviceWaterQualificationBudget` has seven required fields:

- `native_source_identity`: the genuine Native identity used by the canonical
  context, independently joined to its selected Source evidence.
- `current_runtime_ref`: the authenticated runtime record's `SourceReference`.
- `factory_source`: the unchanged prescribed physical factory Source reference,
  exactly 20,771 bytes with SHA-256
  `8ab9ad38bfd434989f8c5e3c0727b301df41184aaef01557fb4678cc9e504eaf`.
- `grant`: the complete canonical product-budget JSON reference.
- `source_bindings`: the original ordered 22-module Source roster, with an exact
  file reference for each loaded module.
- `gpu_uuid`: the physical GPU UUID joined by the external controller.
- `source_manifest_sha256`: the authenticated carrier Source-manifest digest.

References retain the original path, size, SHA-256, regular-file, readonly
and currentness checks. The Native identity is not the digest of a
containing metadata file. The owning code checks its equality to
`context.source_identity`; external controls bind the runtime, physical device,
Source carrier, selected factory and grant before transport.

`qualification_budget_record(budget)` returns the complete closed product
record. `qualification_budget_bytes(budget)` encodes it as canonical ASCII JSON
with sorted keys, compact separators and no trailing newline. The record omits
its containing `grant` reference, so it has no self-hash cycle. The separate
external qualification grant authenticates this product record and the actual
actor, Plugin, runtime, Source and device observations. The product record alone
does not authorize execution or establish a fresh device allocation.

The eleven external operation roles remain `None`: Native metadata, CUDA
preparation and attempt, transport, report, writer, query, validation, fixture,
oracle and observer. The eight primitive allocation units and constructor
allocation accounting also remain `None`. No observed peak, available-memory
headroom, reserve or numeric placeholder is promoted to one of those bounds.

## Execution and retained checks

Use the existing canonical `WindowedDeviceWaterContext` with the exact
qualification budget as its `runtime_bounds` operand. Before genesis, call
`reserve_qualification_device_water_oracle_frontier(budget, device_id=...)`.
The complete-event entry is:

```python
handoff, admission = complete_qualification_device_event(
    state, context, column, field, schedule, vegetation, sediment,
    max_end_s=max_end_s, control=control, policy=policy,
    report_every_s=report_every_s,
)
```

This invokes the genuine `CudaHydraulics` and `complete_event` with the existing
controls and completion policy. It preserves actual Native admission profiles,
water demand and paired pickup/deposit, current availability, ledger folds,
wet constant-depth commit, full validation, publication and dry completion.
It uses no CPU transport fallback, copied solver formula, full-volume bed
reconstruction, fabricated journal or forced event completion.

The qualification producer retains the Source-known positive payload and Native
plan terms, actual retained backing occurrences, current/next/partial writer
reservation, journal and version limits, mapped/read/work ceilings, and Native
0.5 available-device-memory check. Its logical HOST/device predicates remain
5 GiB and 2 GiB. The original HOST availability check also remains. These are
known-subset predicates; they exclude the opaque operations and constructors.
They cannot prove total allocation fits within either predicate.

Every `QualificationDeviceWaterPhase` and `QualificationAdmissionSummary`
explicitly reports `known_subset_only=True`, all twenty opaque terms as `None`,
and the claim `bounded physical qualification; known subset only; no allocation
proof`. A summary is emitted only after a genuine complete `DryHandoff`.
Exceptions poison and close the active authority and propagate. Closing that
authority is not deallocation or rollback. Preserve genuine partial files,
store versions and error evidence after failure.

The separate external controller must retain the original execution budgets:

| Control | Budget and enforcement scope |
| --- | --- |
| Ordinary CuPy pool | 2 GiB allocator limit; foreign/nonpool allocations excluded |
| Leader RSS | 5 GiB sampled abort threshold |
| Device driver growth | 9 GiB sampled abort threshold |
| Run storage | 512 MiB sampled retained-tree abort threshold |
| Regular output file | 64 MiB OS file-size limit |
| Combined output streams | 4 MiB monitored threshold |
| CPU time | 480 s soft / 540 s hard OS limit |
| Wall time | 600 s controller abort threshold |

All cold compiler, fixture, oracle, observer and failure activity belongs to the
same run. Sampling is not instantaneous or whole-tree hard enforcement. Report
actual sample gaps and cleanup latency. No extra prewarm allowance, opaque
allocator theorem or new device reservation is implied.

## Focused physical oracle

The single new selector is
`tests/conservative_water/test_windowed_shared_device_qualification.py::test_small_complete_device_water_qualification_matches_dense`.
The Root-owned fixture `device_water_qualification_budget` supplies the exact
authenticated descriptor. Missing or changed Source/grant identities refuse.
The test does not manufacture a grant or fill missing proof fields.

Its inputs, factory and every existing physical assertion match the strict
oracle. It compares a genuine complete windowed GPU water event with the
independent Dense CuPy event, retaining all exact array/scalar checks, existing
Native inventory regroup/reduction bounds, availability, water/sediment budgets,
dry handoff, timing/representation rules and original input/deep-version
immutability. A no-op deep publication may retain the original version. The
additional assertions check only the truthful qualification labels.

Stage A wet Native commit and Stage B direct/paired exchange qualification are
prerequisites. A future passing run qualifies only this labelled physical
experiment under its actual Source/runtime/device/budget controls. Failure is
not completion, and successful metadata tests alone establish no water result.

## Commit-only qualification entry

`commit_qualification_device_entry(state, context, graph, reference,
current_time_s, routing_source_m, *, control)` accepts the adapted seven-field
shared bed and the same qualification context/budget. It creates a short,
nonreentrant owner after the actual type, control, geometry, profile, Source and
runtime checks. One entry gate reserves the original known positive prefix,
then genuine `sediment_bed.commit_bed(..., force=True)` performs the existing
Native trigger, avalanche, constant-depth water copy, accounting, publication,
full validation and net-displacement routing decision. Its trigger, writer and
candidate-validation gates remain independent and unchanged.

The result is the actual `BedCommit` and an actual-counter
`QualificationAdmissionSummary`. This commit-only summary has
`hydraulics=None`; it creates no `CudaHydraulics`. Thus the earlier
complete-handoff summary statement applies to the full-event API, while this
additive entry returns only after its genuine forced commit. The full-event API
and every prior strict/qualification body remain unchanged. All twenty opaque
terms remain `None`, with the same known-subset/no-allocation-proof labels.

An unrouted commit retains the exact graph and routing-source objects. A routed
commit returns the actual refreshed graph/source; the decision uses the actual
Native context threshold and maximum active-cell absolute net displacement,
never a wrapper threshold. Errors poison the short owner and propagate; finally
closes authority without claiming memory release, rollback or completion.

The appended component selector
`test_qualification_device_entry_matches_direct_native_commit_and_routing`
compares two sequential real entry commits with the independent direct Native
commit/routing calls on a separate instance of the same small fixture. It
checks input/old-version immutability, the complete logical column and all six
other authoritative fields, constant-depth/mobile identities, routing evidence,
actual owner counters and opaque labels. It does not execute a water event or
create a hydraulic lifecycle. Root must issue its own actual selector/source/
runtime/grant controls and run it under the same whole-experiment budgets; no
extra oracle, event timeout or allocation allowance is introduced. Metadata
checks or this component alone do not qualify complete water or alternation.


## Qualification DEVICE payload projection

The qualification-only known DEVICE term separates ordinary grid/mask payload
from HOST planning/text/header projections. It keeps the unchanged Source
request/work census. The complete-entry17Q boundary-planning loop is NumPy and
Python metadata; its fixed830 requests remain in the ordinary bank. For the
fixedC16/K1/V6/W3 family, the packet gridP8 gives128 elements; numeric payloads
receive1024B and masks128B per ordinary request. Three full ordinary request
banks retain current, prospective and first-error occurrences. Genuine CuPy
water16-column and sediment14-column report tables retain three full4096-row
pairs for current/copy/partial-error owners. The resulting known DEVICE term is
461010816B, not a universal allocation or backing theorem.

The closed helper refuses changed Source polynomials, Source domain controls or
request counts. It offers no configurable fee, shape or bank multiplier. Generic
HOST/work projections, old HOST report/planner overcoverage, actual Native peaks,
current retained inputs, CUDA context/attempt banks and every reserve/cap/0.5
admission rule remain in place. All20 opaque provider/primitive/constructor terms
still remainNone. The strict proof-oriented producer and generic Source are
unchanged. Root must qualify a newly authenticated Source/runtime/carrier/grant
composition under the original monitors; a smaller known-payload projection is
not fit, physical completion or allocation-proof acceptance.
