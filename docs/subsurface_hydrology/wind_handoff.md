# Native wind handoff of the floating lifecycle (WH1, Dense NumPy CPU)

The source (`src/maple_syrup/floating_wind_handoff.py`) adds the external-wind interface to the published Phase 2
native lifecycle. Codex executed all **68 native interface tests** on Dense NumPy CPU and a genuine CSV/MAPLE wind
occurrence followed by native handoff and checkpoint reload. The MASY coordinator has separate integration tests and
qualification; native component acceptance does not establish a complete scenario CLI, LEAF, GPU or windowed support.

This is a new engineering seam on top of the published Phase 2 source. It adds **no water or wind physics and no parameter
choice**; no kernel, default, historical API or codec semantics changed, and no existing source file was modified.

## What it does

A MASY-style caller runs the actual native wind occurrence on the accepted boundary's bed (MAPLE
`finalize_wind_event`) and hands the transient `WindEventFinalizationResult` plus the complete child `BedState` to SYRUP:

```
Boundary (complete, accepted head)  --actual native wind-->  finalized result + child BedState
        --stage_wind_handoff-->  WindHandoffCandidate (pure)  --adopt_wind_handoff-->  refreshed Boundary (new head)
```

Wind assumes dry regardless of soil or ponded water. Every surface-water leaf is carried bit for bit (no water-borne
sediment, no wind water physics), real soil water and J/I are retained, E/T/growth are absent, there is no extra
hydrology step or replay. Only the actual zero-time approved capacity reconcile of the columns whose properties/geometry
really changed can alter soil water (surface return, open-bottom export). There is no wind moisture suppression or physical reset.

## Public API (`maple_syrup.floating_wind_handoff`)

| name | role |
| --- | --- |
| `WindHandoffLimits` | explicit plain-int limits: cells, voxels, classes, scan batch, modelled scan bytes, modelled candidate bytes, receipt bytes, **mobile-pool cohort slots (`max_cohorts`)** |
| `start_wind_boundary(owner, bed, event_factory, *, bed_context, limits, available_bytes_fn, ..., before_publish=None)` | a REAL initial `Boundary` from a fresh current owner and the complete `BedState`. **`bed_context` (the genuine static `BedContext`) is required.** Returns `(lineage, boundary, wind_context)` |
| `stage_wind_handoff(lineage, parent, final, child, *, context, occurrence_id, start_s, limits, available_bytes_fn, ...)` | pure staging (signature unchanged); returns a `WindHandoffCandidate` with a bounded scalar receipt |
| `adopt_wind_handoff(lineage, parent, candidate, *, available_bytes_fn=None, reserve_bytes=0, before_publish=None)` | integrity audit, preallocation, mint, head install; the candidate is then emptied |
| `WindHandoffCandidate.accepted_boundary_ref` | read-only weak reference to the minted boundary (see below) |
| `WindContext`, `restore_wind_context(boundary, bed_context, expected_record)` | the prepared water-free static context and its fresh-process reconstruction |

Errors: `WindHandoffError` (a `LifecycleError`), `WindHandoffLimitError`. The lifecycle's own `LifecycleError` for a stale,
duplicate, wrong-head or non-`Boundary` argument is raised unchanged; `NativeFloatingMemoryError` is raised unchanged by the admission.

## Admission (all paths, before any value is read)

`stage_wind_handoff` order:

1. head, eligibility (complete boundary only), occurrence-id string, context binding, start clock (scalars);
2. **metadata only**: exact host float64 NumPy leaves and shapes of both beds, class-vector type/shape (nothing is coerced), the
   finalized mobile pool's exact native type (flat or calendar), its actual **array footprint** and **cohort slot count**
   (`max_cohorts`), shared-field identity, clocks, aliasing bounds and commit stamps. Unqualified pools/backends/devices are
   refused here, before any native dispatch or transfer;
3. **aggregate admission**: limits (`max_scan_bytes`, `max_candidate_bytes`, receipt model), then the real `available_bytes_fn`.
   The model is the native `nfs._estimate` scan plus this module's terms (water comparison, the **actual pool footprint**,
   native-validity scratch, hash scratch) and the candidate base (**the continued soil-state copy, the surface array and the
   receipt, which exist even with zero dirty columns**), so `max_candidate_bytes` bounds the zero-dirty path too;
4. only then: the native mobile-pool predicate, class-vector values, water carry, **full native bed validity**, descriptor, the
   cold scan, and (at the actual dirty count, with a second admission) the selected refresh.

`start_wind_boundary` plans the **prelude** from `bed_context` before `FloatingPondVertical`/`PondEvent.initial_state` exist:
metadata, limits (`max_scan_bytes` bounds the whole prelude: initial cold witness, context, validity), the native initialization
model, the real reader; then the water-borne check and native validity; only then the vertical, the static factory and the native
initializer (which keep their own internal admission contract).

**Honest scope:** these are modelled host payloads of this module's own stages (declared terms: validity scratch is one
voxel+active array size, hash scratch one 1 MiB strided chunk plus 64 KiB metadata, receipt `3072 + 112 K` bytes). The
selected candidate and adoption models also reserve this integrity scratch. They are not measured and not a whole-process bound;
the native initializer/factory admission and the interpreter are outside the model.

## Wind proof

`final` must be the actual native class (no boolean, JSON stub or duck type), `completed` and `water_handoff_ready`, status
`completed`/`already_empty`, no reasons. Its finalized scheduler state's shared fields must be the very objects of `child`; its
terminal clock equals the physical clock and the cursor is at a terminal block (offset 0); the wind mobile pool is empty by the
native predicate; the class vectors are exact host float64 `(K,)` arrays, finite, close within their native bounds and leave
nothing airborne. The child must not alias the parent's arrays, its water leaves equal the parent's bitwise, and the commit stamps
are ordered. **Not claimed:** producer-cryptographic provenance beyond what the upstream types supply; MASY additionally verifies
the source pair and the full wind budgets.

## Native bed validity

After admission the child passes the same owning native validators the checkpoint uses (`pond_checkpoint._native_checks`: bed
support, active-layer/voxel partition, availability, ledger/committed/water validators, reconciliation context and the ledger
reconciliation against the physical state at the wind end). A class or mass change that no native ledger records is refused; nothing
is repaired, reset or re-initialised and no formula is copied. A valid wind child therefore comes from the real MAPLE
exchange/ledger (`finalize_wind_event` / `apply_bed_demand`), never from overwriting voxel classes.

## Post-stage integrity

At staging the module hashes (SHA-256, copyless views, 1 MiB chunks) every CPU array leaf and scalar of the **current parent**
(bed, surface/flow, root column maps, operator, history, soil water, config, ledger, provenance, routing) and of everything
**proposed** (child/published bed, new column/operator/history/water/depth/Ks, ledger, provenance, receipt, occurrence, end). The
digest lives in a private weak table keyed by the candidate (not in the caller-mutable cell). Adoption recomputes it after the
`before_publish` callback and refuses before consuming or publishing if the child, the proposed state or the parent changed in
place. This is a tamper/aliasing check of this process's objects, **not** a source-independent cryptographic proof of the producer.

## Exactly-once and boundedness

There is no per-lineage set of identities. Guards: the expected accepted head (the parent stops being the head at adoption) and the
one-shot candidate. One bounded record per live lineage (the last adopted id, at most 200 characters) catches an **immediately
repeated** id only. Historical membership is **not** claimed (it would need unbounded storage): the MASY schedule cursor owns
cross-event and cross-process identity and order (the Root adapter persists the next occurrence index and the resolved schedule
hash).

Adoption order: checks -> admission (the actual host reader is used when omitted) -> `before_publish()` (outside every mint window, before anything irreversible) ->
head re-check and integrity audit -> **preallocation of every bookkeeping slot** (the last-id record) -> mint -> weak accepted
reference -> head install (first commit store) -> consumed / emptied / last-id stores (plain assignments in `finally`). A failure in any step
before the head store leaves the head and the candidate retryable; after the head store the parent is no longer the head, so a
duplicate adoption is refused whatever else is interrupted.

`candidate.accepted_boundary_ref` is a read-only weak reference to the minted boundary, created before the commit (`None` if the
adoption never reached it). A coordinator interrupted right after publication reconciles with
`ref = candidate.accepted_boundary_ref; ok = ref is not None and ref() is not None and lineage.head is ref()`. It retains no strong
old or new state.

Initialization calls `before_publish(lineage, proposed_boundary, wind_context)` before retiring the original owner.
The callback lets the coordinator prepare all binding/cursor metadata, and the native initializer checks the complete
proposal's integrity again afterward. Retirement then precedes head installation; a `finally` block exposes the new
head if interruption occurs immediately after retirement. A failure before retirement leaves the original owner current.
An explicit native stale-owner refusal never publishes the proposed head, including when another action adopted or retired the owner.

## Dirty set and properties

Wind has no accepted gross-coverage proof (unlike a Pond event), so the dirty set is a bounded cold streamed comparison of the
active columns in batches through the actual native readers (`native_floating_soil._rows`, `_changed_rows`): every voxel/class
and the active layer, bitwise. A composition exchange of zero net elevation is dirty. Compact texture is read for the dirty columns
only; the changed set, reconcile, closure, new column, operator, history rebinding and preferential rebinding are the accepted
`native_floating_soil` / `floating_matrix_infiltration` / `floating_preferential_infiltration` helpers (no copied formulas). Cases:

* nothing dirty, or dirty with identical properties and depth: the prepared root objects (column, operator, history, preferential) are reused bit for bit; W/J/I are bitwise unchanged;
* actual depth differs but the hydraulic geometry is equivalent (and/or only Ks): a column/operator bound to the actual depth, no cascade, nothing exported or returned (E3c rule);
* properties or geometry changed: only those columns pass through the zero-time reconcile on the current W. A surface return is added to the surface depth exactly once and makes the boundary `pending_recession` (not wind- or day-eligible until a real recession completes again, and not saveable); an open-bottom export enters the lifecycle ledger once.

J and the cumulative intake are rebound bit for bit, never reset. The native wind end clock and bed stamps, the original routing
tuple/source, the property generation (+1 only if the root is replaced), the cursor and the lifetime ledger are preserved; the
retired owner/W0 are never reused. Quiet retained standing water alone does not block wind.

## Receipt truth

`water.wind_water_physics` is `"none"`: wind runs no water physics. The zero-time capacity reconcile is a separate explicit native
step reported as `water.zero_time_reconcile` = `{performed, n_columns, surface_return_m3, bottom_export_m3, whole_bound_m3}`. The
earlier `physical_update: "none"` was misleading whenever that reconcile returned or exported water and is removed. Native volume
and sediment bounds are used as-is (no ad hoc epsilon).

## Clocks

`start_s` must equal the accepted boundary's physical clock and the soil clock exactly; `end_s` is the finalizer's native
terminal time (finite, strictly after the start). The soil clock is advanced across the real interval with the owning public
`floating_continued_state` on the same W (an explicit event-time carry, not a flux step). No wind/water/solver clock is
rebased or rounded; the weather nominal cursor and the last hydraulic routing tuple are untouched.

## Static context and restart

`WindContext` holds the genuine static `BedContext`, the shape, the native descriptor digest (grid, geometry, ordered class
metadata, mass resolution, active-mask pattern), the original cold source witness and the config identity. It holds no owner, W0,
bed or terrain. `WindContext.record()` is a scalar record the caller keeps; `restore_wind_context` rebuilds the context from a
verified restored boundary, the original static `BedContext` and that record.

The new `Boundary` is a genuine complete lifecycle boundary with the closed provenance/ledger vocabulary, so the **unchanged**
lifecycle checkpoint saves and restores it, and the restored lineage resumes into a storm, a day or another wind.

## Refused (typed, before work)

CUDA and numba owners, Windowed or other representations, device arrays and non-exact NumPy leaves, an unqualified or too-large mobile
pool, a pending recession, pending storm, running occurrence or mid-day hold, a stale/wrong parent or lineage, an immediately repeated
occurrence id, a context of another source, a bed the native validators/ledger reconciliation refuse.

## Tests

`tests/phase2_wind_handoff/` passed 68 native CPU tests. Valid wind children are built with the actual native `apply_bed_demand`
pickup/deposit and ledger (`wind_builders.child_move`). Direct native-class construction (`child_class_swap`) and a lone pickup
are labelled mechanical/negative fixtures; the former is used only to prove the native-validity refusal. Integrity, admission-order,
interruption and weak-reference tests are in `test_wind_handoff_integrity.py` / `test_wind_handoff_admission.py`. Root separately
runs the actual MASY/MAPLE wind.

## Separate qualifications

Claude authored the native interface and corrections; Codex executed the native and actual MAPLE integration checks.
MASY qualifies its coordinator and native fresh-process controls separately. GPU/windowed support, LEAF, Phase 2.5,
the complete scenario CLI/runtime provisioner and physics changes are outside this interface's acceptance.
