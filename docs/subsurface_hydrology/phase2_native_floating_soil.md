# Phase 2 E3 — native bed to floating-soil owner: initialization and refresh transaction

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_native_floating_soil_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: author-implemented (corrections 01-02, E3b, mask binding, E3c equivalent geometry), CPU/host verified only; the CUDA branch is unrun; not independently accepted. Text below is the module contract (`native_floating_soil.py`).
Phase 2 E3: native bed -> floating-soil water owner, initialization and the staged changed-column REFRESH transaction (CPU/host; NumPy and Numba floating owners).

What this is
------------
A narrow orchestration layer that connects four ACCEPTED components without re-implementing any of them: the native MAPLE bed (Dense `VoxelColumnState`/`ActiveLayerState` or genuine resident-window
`WindowedColumn`), the E2 compact reader (`compact_soil_profile.iter_compact_soil_profile`: actual mass-overlap layer texture and the cold canonical depth of explicit cells), the pure Saxton-Rawls property
provider (`soil_texture_properties.estimate_soil_texture_properties`) and the D1/D2 floating column, zero-time reconciliation and persistent matrix history (`floating_soil_layers`,
`floating_matrix_infiltration`). The bed stays the only sediment authority: nothing here stores sediment, and the owner holds ONE water inventory.

    owner = initialize_native_floating_soil(bed, config, initial_relative_saturation=..., memory_budget_bytes=...)
    cand  = stage_native_refresh(owner, parent_bed, child_bed, mode="properties"|"depth", memory_budget_bytes=...)   # PURE: nothing is adopted
    owner2 = adopt_native_refresh(owner, cand)                                                                      # the one mutation point; owner/cand become stale

Explicit inputs (`native_floating_config`)
------------------------------------------
Fixed surface-following nominal layer thicknesses; an EXPLICIT grain-class -> (sand, silt, clay) composition mapping `(K, 3)` (rows sum to 1; NEVER guessed from a grain diameter), organic matter percent
`(L,)`, the bulk density (checked against the actual MAPLE packing: `geometry.bulk_density_kg_m3` must equal it, otherwise a typed `NativePackingError`; nothing is adjusted), explicit latent hydraulic properties
`(L,)` that a zero-mass slot RETAINS (field capacity and saturation; a zero-mass slot never receives a primary soil texture), the explicit residual content, evaporation stress/factor, constant interface
diffusivity `(L-1,)`, the lower-boundary mode, the explicit Green-Ampt matrix inputs (conductivity, suction, vegetation factor) and the initial persistent history, optional measured overrides `(L, N)` (NaN = not
measured, passed through the provider). Thin/zero coverage uses the E1 `partial_over_base` policy (a layer holding any actual material has a texture; a layer with exactly zero mass is latent).

Geometry and depth
------------------
`D` is the COLD canonical native depth of the compact reader (native NumPy order, full flat base 0; NOT the S1 sequential diagnostic). The profile bottom is open only when `D > H` (D == H is closed rock; no epsilon);
partial and zero depths are the D1 semantics. Layers keep their nominal thickness measured from the surface; the per-layer inventory is held through reassignment and the zero-time reconciliation of D1
(field-capacity cascade, saturation return to the surface, open-bottom export once) adds no time, E, T, diffusion or rain.

Binding, dirty proof, staging, adoption
---------------------------------------
There is no caller-supplied dirty list. `stage_native_refresh(owner, parent_bed, child_bed)` (1) streams the actual native rows of the ACCEPTED PARENT bed and the staged CHILD bed in cell batches of `B`
(Windowed: the mandatory native published-journal/cheap/full validation first, then the native gather), compares every active cell EXACTLY (all voxels and classes, plus the active-layer mass; a zero-net-elevation
composition exchange is dirty), and computes a cold SHA-256 SOURCE WITNESS of each side; (2) refuses unless the parent witness equals the witness recorded in the owner's current binding (so the proof is bound to the actual
accepted parent, not a label); (3) reads compact texture/depth ONLY for the dirty cells (batches released, no `(L, N, K)` tensor), computes properties and reconciles; (4) stages every result (new column, state, matrix
operator and rebound history, depth, witness) and mutates nothing. `adopt_native_refresh` checks that the owner is the CURRENT one of its lineage and that the candidate's minted parent binding is that owner's binding;
adopting twice, adopting a stale candidate, or staging against an owner that has been superseded are typed refusals, so a duplicate refresh can never double-debit the export or surface return. The witness
passes are COLD O(N V K) native reads once per refresh (counted in the admission), not a hot-GPU optimisation. The binding token is a run-local minted capability and is NOT a physical checksum.
`mode="depth"` keeps every property and updates only `D` of the dirty cells (physical-depth-only reconciliation, e.g. after actual erosion before publication); `mode="properties"` also refreshes texture-derived
properties. When the staged maps and depth equal the accepted ones nothing is reconciled (no redundant cascade at one generation).

Outputs exposed ONCE: `surface_return_m` `(N,)` (capacity water returned to the surface, an internal transfer the caller must incorporate) and `bottom_export_m` `(N,)` (open-bottom export, a loss), with the D1
receipt. Nothing claims that the returned water already flowed or that an event completed. Bounded accumulated receipts (`maxlen = 8`) plus running totals live on the owner.

Correction01 contract
---------------------
* ONLY the columns whose depth or properties actually changed are reconciled, in compact old/new sub-column batches through the accepted D1 `reconcile_floating`; every unrefreshed column (including one above
  field capacity, a valid continued state) keeps its water bit for bit, and the whole-inventory closure is checked against the FP64 bound. `NativeReconcileReceipt` is the ONE source of the accounting adoption credits.
* Every exposed array and config record is sealed (read-only); the config identity covers every override value; minted owners/candidates/bindings are registered instances that cannot be copied, deep-copied,
  pickled or rebuilt by `dataclasses.replace`; adoption does all refusable work BEFORE consuming the candidate or advancing the lineage authority.
* The witness binds the physical descriptor (grid, geometry, ORDERED grain classes, mass resolution) and canonical rows: newly added or removed EMPTY TOP CAPACITY (different voxel counts) is compared as zero
  padding and is not part of the identity; any other difference is a change.
* The pure provider is called only for slots holding actual material (no fabricated texture, no renormalisation beyond the provider's own declared summation bound).
* The scan/comparison is a COLD O(N V K) boundary pass at a refresh, NOT a post-each-Pond-step optimised path. CUDA integration is pending. The texture-derived saturated conductivity `Ks` is NOT fed back
  into the matrix law (the explicit matrix `K` is retained), so this FC/SAT refresh does not complete the Phase 2 texture-Ks feedback (that needs actual texture `Ks` unless an explicit measured override).

Correction02 contract
---------------------
* Admission is composed from the ACTUAL parent AND child shapes (`estimate_native_refresh_memory(parent_bed, child_bed, config)`; `estimate_native_initialization_memory(bed, config)` for one bed): the native full
  capacity may grow or shrink between them, so the scan rows/gather/canonical-hash/comparison scratch, each bed's own compact-reader/native validators (Windowed: deep-store validator and cache) and the persistent
  candidate are all counted BEFORE the first row is read. Row comparison needs no padding copy (`_changed_rows`); the invariant and closure checks run in bounded column chunks; fresh result arrays are sealed in
  place. A warm tracked-array (tracemalloc) peak is tested inside the estimate for Dense and Windowed at N = 1000, L = 3, B = 16; this is NOT a whole-process bound (interpreter/JIT/pool excluded; no allowance hides a
  known array).
* The minting window is thread-local and bound to ONE constructor call (no public boolean global spans user availability callbacks). `NativeRefreshRequest` on the candidate records the exact parent lineage,
  generation, witness, child witness, mode, batch size, memory requirement and descriptor for a later E4 caller. Late adoption failure stays atomic and the history is never reset.

Actual depth bookkeeping vs hydraulic geometry (E3c)
----------------------------------------------------
The published canonical depth `D` always carries the ACTUAL native value (owner `depth_m`, the column's `soil_depth` map, provenance/snapshots). Only a change of the HYDRAULIC geometry or properties may call
the zero-time cascade: with fixed nominal edges the porous slabs are `max(0, min(D, e_k+1) - e_k)` and the lower boundary is open only when the mode is open AND `D > H` (strict, no epsilon), so a pure depth change
that leaves every slab and the bottom flag bitwise equal (both depths deeper than H with unchanged properties; in an explicit CLOSED profile also `D == H` versus `D > H`) is EQUIVALENT: nothing is
reconciled, nothing exported or returned, yet a genuine new column/state/operator is published bound to the actual depth, carrying the accepted inventory bit for bit (same `W`, time, `J`, cumulative intake) with the
persistent history rebound. An OPEN profile crossing `D == H` -> `D > H` flips the bottom flag and IS reconciled; partial/zero slabs, FC/SAT changes and rock still reconcile as before. `NativeRefreshRequest`
and the candidate now distinguish `n_changed` / `changed_columns` (reconciled) from `n_depth_changed` / `depth_changed` (actual canonical depth differs). This supersedes the earlier C2 behaviour that treated
any raw depth difference as a reconcile trigger (a confirmed defect: an equivalent 1 mm change exported water from an above-field-capacity parent); the old immutable receipts stay evidence of the subset they tested.

Source-binding recipe v2 (E3b correction02)
-------------------------------------------
The physical descriptor, and through it every source witness, now binds the ACTUAL active-mask pattern (versioned digest of the mask shape and every flat cell flag, layout independent) in addition to the grid,
geometry, ordered grain classes, mass resolution and the canonical rows; v1 hashed only the active COUNT, so a different domain with the same count and equal columns could masquerade as the accepted parent.
Both backends also refuse a staged mask different from the owner's with the same typed error (CUDA: after admission, one counted O(N) mask upload and one counted 1-byte verdict, before any row, property or
water work). Witnesses of the two recipes are never equal; accepted C2 receipts/checkpoints are not rewritten and a future checkpoint must record the recipe version.

E3b contract (CUDA branch and texture-Ks mode)
-----------------------------------------------
* `backend="cuda"` configs are served by `native_floating_soil_cuda` (written WITHOUT a GPU; only host-double tests ran): same cold witness/descriptor/canonical depth/provider/changed-column rule/once-only
  adoption; the water, FC/SAT maps and layer-0 Ks stay device-resident (private, logically immutable; admitted counted detached inspection `inspect_counted`); the host holds the sealed O(N) depth and mask
  records and bounded B-tiles; the provider runs on the host per tile (explicit cold path); native device beds are read tile by tile through MAPLE's counted `to_host`; reconciliation is the accepted D1 CUDA
  reconcile over compact device sub-columns and D2 CUDA owns the operator/rebind/history. The CURRENT device is gated first; composed DEVICE and HOST admission use separate budgets, reserves and readings; host/device
  mismatches and other-device beds are typed refusals. The whole-inventory closure of the CUDA branch is the closure of the reconciled columns. The scan is a cold O(N V K) boundary read, not a per-step path;
  nothing here is a full-storm or large-domain claim.
* `matrix_conductivity_mode="texture_ks"` (explicit opt-in; `"explicit"` remains the default and is byte-identical in behaviour and identity): the Green-Ampt matrix conductivity of each cell is the accepted
  provider's CURRENT UPPER-layer (layer 0) saturated conductivity (or a measured override through the provider); a layer 0 with no material keeps the user-supplied latent Ks. Properties mode updates Ks for
  actually affected cells (including a zero-net composition exchange); depth mode holds it. Interface conductivity, diffusivity, suction and residual are NOT derived from it and no averaging exists. A Ks-only
  change builds a new operator over the same column/state (no cascade) and the persistent `J`/cumulative intake are rebound exactly (never reset).

Not in scope: Pond/wind scheduler/checkpoint/restart/every-N output, CUDA owners and device-resident beds (refused), vegetation. Author-side: CPU/host only.

## E3b correction01 (CUDA branch)

See the `native_floating_soil_cuda` module docstring: backend selector by genuine representation, whole-inventory receipts, payload vs required admission, counted syncs, device Windowed qualification definitions (unrun).

## Supported vs deferred (E4 requirements)

Supported and tested (CPU, NumPy and Numba floating owners; native Dense and genuine Windowed beds with material below the resident floor): initialization from the actual bed; exact streamed dirty proof
bound to the accepted parent witness (including zero-net-elevation composition exchange); inactive-cell exclusion; compact texture/depth only for dirty cells; properties through the pure provider with explicit
mapping, measured overrides and latent zero-mass slots; D > H open / D == H closed (no epsilon); staged zero-time reconciliation with returns/export exposed once; matrix history rebind without reset; minted
binding/lineage with stale, duplicate and wrong-parent refusals; composed admission before any scratch; failure (provider, native row, later batch) with no mutation.

Deferred (not claimed): the Pond bridge, wind scheduler, checkpoint/restart codec (the random binding token is run-local; E4 must persist a deterministic
witness/generation), every-N output and the daily driver; hot per-step dirty tracking (the witness passes are COLD O(N V K) reads per refresh, counted in the admission); Windowed pending-journal publication
(refused by the native checks); a hot near-rock classification. E4 must call `mode="depth"` after actual erosion BEFORE publication (physical-depth-only reconciliation) and `mode="properties"` at the
between-events handoff, and must incorporate `surface_return_m` (internal) and `bottom_export_m` (loss) exactly once from the candidate.

Root independently accepted the bounded CPU native transaction after 33 tests and fresh independent controls, including the untouched-above-FC and parent-capacity memory reproducers. Source identity and remaining CUDA/Ks/Pond/restart/output limits are in phase2_native_floating_soil_acceptance.json. This is component acceptance, not whole Phase2.
