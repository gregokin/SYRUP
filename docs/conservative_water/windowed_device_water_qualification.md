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
