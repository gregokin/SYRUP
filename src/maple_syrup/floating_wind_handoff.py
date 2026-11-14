"""WH1 (CPU: Dense NumPy): the native WIND HANDOFF of the single-authority floating lifecycle -- a genuine completed `Boundary` -> an actual native MAPLE wind result -> a refreshed accepted `Boundary`.

SCOPE. One bounded engineering seam, no new water or wind physics and no parameter choice. A MASY-style caller runs the ACTUAL native wind occurrence on the accepted boundary's bed (`maple.aeolian.scheduler.finalization.finalize_wind_event`) and hands the transient
`WindEventFinalizationResult` and the complete child `BedState` to this module. Wind is DRY regardless of the soil or ponded water: every surface-water leaf of the child must equal the parent's bit for bit, there is no water-borne sediment, no E/T/growth, no
extra hydrology step and no replay. Only the actual zero-time approved capacity reconcile of the columns whose properties/geometry really changed may alter soil water (surface return / open-bottom export, each exactly once). J/I are rebound bit for bit.
Dense NumPy CPU only: CUDA, numba owners, Windowed beds and any unqualified representation are typed refusals BEFORE any array work.

PUBLIC TRANSACTIONS (a staged pair; nothing is shared with the historical API/codec semantics)
  `start_wind_boundary(owner, bed, event_factory, *, bed_context, limits, available_bytes_fn, ..., before_publish=None)` -> `(lineage, boundary, wind_context)`: a REAL initial `Boundary` from a fresh genuinely CURRENT `NativeFloatingSoil` and the complete actual `BedState`
      BEFORE the first wind/day. The explicit genuine static `BedContext` is REQUIRED: shape, class count and the static descriptors come from it, so the PRELUDE is planned and admitted from metadata (limits, then the actual available-bytes reader) BEFORE
      `FloatingPondVertical` / `PondEvent.initial_state` exist (the native initializer / factory keep their own internal admission contract). The prelude is then exactly the opener's (actual vertical, caller's static event factory, `PondEvent.initial_state` with all
      callbacks, cold source witness) so the routing tuple `(topology, held_source, datum)` is the actual native static initializer's; no fictional zero-rain event, no consumed calendar day. The initial stored ponded water is carried untouched. The lineage head is
      installed on the not-yet-returned lineage and the owner is retired LAST (the only mutation); every refusable step, and the optional `before_publish` callback, precede it, so a failed start leaves the owner current and retryable.
  `stage_wind_handoff(lineage, parent, final, child, *, context, occurrence_id, start_s, limits, available_bytes_fn, ...)` -> `WindHandoffCandidate`: PURE. ORDER: expected head, eligibility, typed backend/representation/limits and the metadata of every input (exact host
      float64 NumPy leaves, mobile-pool type/footprint/cohort count, class-vector shapes; nothing is coerced, compared or scanned) -> the aggregate admission (limits, then the REAL available-bytes reader) -> ONLY THEN the mobile-pool predicate, the class-vector values, the
      water carry, the full native bed validity (below), the descriptor, the bounded cold row comparison, the selected property read, the native reconcile and closure, and the cumulative lifecycle water identity. Nothing is adopted or consumed; the head and inputs are untouched.
  `adopt_wind_handoff(lineage, parent, candidate, *, available_bytes_fn=None, reserve_bytes=0, before_publish=None)` -> `Boundary`: re-verifies the head, runs the optional `before_publish` callback (outside every mint window, before anything irreversible), audits the
      POST-STAGE INTEGRITY, preallocates every bookkeeping slot, mints the genuine new `Boundary`/`BoundaryRoot` through the lifecycle's own minting window and installs the head; the candidate is then emptied. A second adoption, a stale parent or a candidate staged for another
      lineage is refused (native `LifecycleError`).
  `WindContext` / `restore_wind_context` -- the prepared WATER-FREE static context (see below).

WIND PROOF. The result must be the actual native `WindEventFinalizationResult` (exact class: no duck type), `completed` with `water_handoff_ready`, no reasons, whose finalized scheduler state's shared fields (voxel column, active layer, water, ledger, committed topography,
availability) are the very objects of the child `BedState`, whose native terminal clock equals the physical clock and cursor (offset 0.0), whose wind mobile pool (exact native flat or calendar type, host arrays) is genuinely empty by the native predicate, and whose per-class
finalizer vectors are exact host float64 `(K,)` arrays, finite, closed within their native bounds and leave nothing airborne. The child's water leaves must equal the parent's bit for bit and carry no water-borne sediment. NOT claimed: producer-cryptographic provenance
stronger than these real upstream types supply (the caller, MASY, additionally verifies the source pair and the full wind budgets).

NATIVE BED VALIDITY. After admission the child bed must pass the SAME owning native validators the checkpoint uses (`pond_checkpoint._native_checks`: bed support, active-layer/voxel partition, availability, ledger/committed/water validators, reconciliation context and the
ledger reconciliation with the physical state at the wind end). A class/mass change that no native ledger records is therefore refused; nothing is repaired, reset or re-initialised, and no formula is copied here.

DIRTY / PROPERTIES. Unlike a Pond event, wind has no accepted gross-sediment coverage proof, so the dirty set is a bounded COLD streamed parent/child row comparison of the active columns in batches of `B` through the ACTUAL native readers and comparison
(`native_floating_soil._rows` / `_changed_rows`, every voxel/class and the active layer, bitwise: a composition exchange of zero net elevation is dirty). Compact texture is read for the DIRTY columns only (`_read_texture` with the accepted provider), the changed set is the
accepted `_changed_columns_parts` (porous geometry/bottom flag or FC/SAT; Ks is not a trigger), and only those columns pass through the accepted zero-time `_reconcile_changed` on the CURRENT W with the accepted `_inventory_closure`. Column / operator / rebound history /
preferential rebinding use the accepted `_build_column`, `_fresh_op`, `rebind_floating_matrix_history` and `rebind_preferential_column`. Constant and no-change cases keep the prepared root OBJECTS bit for bit; a depth-only equivalent change publishes a column bound to
the actual depth without any cascade (the accepted E3c rule). A capacity SURFACE RETURN is added to the surface depth exactly once and makes the boundary `pending_recession`; an open-bottom export enters the lifecycle ledger once. J/I and the cumulative intake are never reset.

RECEIPT TRUTH. `water.wind_water_physics` is `"none"`: the wind runs no water physics. The zero-time capacity reconcile is a SEPARATE explicit native step (`water.zero_time_reconcile`: `performed`, the number of columns and the named fluxes `surface_return_m3`,
`bottom_export_m3`, `whole_bound_m3`); it can return water to the surface or export it, so the receipt never says the water was not updated when it was.

CLOCKS. The actual wind start and end come from the caller / MAPLE: `start_s` must equal the accepted boundary's physical clock exactly and the shared-bed commit stamps must be ordered; `end_s` is the finalizer's native terminal time. The soil clock is advanced across the REAL
interval WITHOUT E/T or hydrology by the owning public `floating_continued_state` constructor on the SAME W, an explicit event-time carry and not a flux step. No native solver timestep, wind or water clock is rebased or rounded; the weather nominal cursor and the last
hydraulic routing tuple are untouched.

STATIC CONTEXT (`WindContext`). The water-free prepared record: the genuine static `BedContext` (its immutable case descriptors), the shape, the native descriptor digest (grid, geometry numbers, ordered class metadata, mass resolution and the ACTIVE-MASK PATTERN), the ORIGINAL
cold source witness and the native config identity. It holds no owner, `W0`, bed or terrain. `restore_wind_context(boundary, bed_context, expected_record)` rebuilds it in a fresh process from a verified restored boundary and the kept scalar `WindContext.record()`.

POST-STAGE INTEGRITY. At staging a bounded, COPYLESS, chunked SHA-256 over every CPU array leaf and scalar of (a) the CURRENT parent (bed, surface/flow, root column maps, operator, history, soil water, config, ledger, provenance, routing) and (b) everything proposed
(`_Parts`: the child/published bed, new column/operator/history/water/depth/Ks, ledger, provenance, the receipt, occurrence and end) is stored in a PRIVATE weak table keyed by the candidate (never in the caller-mutable cell). Adoption recomputes it AFTER the `before_publish`
callback and refuses (before consuming or publishing anything) if the child, the candidate parts or the parent changed in place. It reads the arrays in 1 MiB chunks; no bed copy exists. It is a tamper/aliasing check of this process's objects, NOT a source-independent
cryptographic proof of the producer.

EXACTLY-ONCE AND BOUNDEDNESS. There is NO per-lineage set of consumed identities. The guards are the expected accepted head (the parent stops being the head at adoption) and the one-shot candidate; one bounded record per live lineage (`_LAST`: the last adopted
occurrence id, at most `_ID_CHARS` characters) catches an IMMEDIATELY repeated id. Cross-event / cross-process identity and order belong to the MASY schedule cursor (next occurrence index and resolved schedule hash, persisted by the caller); historical membership is
NOT claimed because it would need unbounded storage. Every bookkeeping slot (the `_LAST` record, the accepted weak reference, the minted records) is created BEFORE the head moves; a late bookkeeping failure leaves the head and the candidate retryable. The head
is installed FIRST among the commit stores (the parent is then no longer the head, so a duplicate adoption is refused whatever else is interrupted).

ACCEPTED-BOUNDARY REFERENCE. `candidate.accepted_boundary_ref` is a READ-ONLY weak reference to the minted boundary, created before the commit (None before an adoption reached that point; no strong old or new state is retained). A coordinator interrupted right after
publication reconciles with `ref() is not None and lineage.head is ref()`.

RESTART. The new `Boundary` is a genuine complete lifecycle boundary (closed provenance/ledger vocabulary), so the UNCHANGED lifecycle checkpoint saves/restores it. No codec, profile or historical source semantics are changed or blessed.

VEGETATION (Phase 2.5 compatibility guard). WH1 is a no-LEAF scope: the boundary it mints carries no vegetation record, and its prepared root keeps the carried operator by reference while the refresh paths use configuration scalars. A boundary that carries an applied
vegetation generation (`parent.root.vegetation is not None`, the owning lifecycle's field; never a duck-typed or parsed record) is therefore REFUSED with `WindHandoffError` by `stage_wind_handoff`, `adopt_wind_handoff` and `restore_wind_context` BEFORE the context/pins, the child payload,
any callback, array or admission work, with no mutation. A vegetation-aware wind handoff (carrying the generation/recipe, the bound covers and the per-cell law, and refreshing with the held per-cell law) is a later qualified interface; nothing here is silently reset or dropped.

MEMORY. Explicit typed `WindHandoffLimits` (cells, voxels, classes, mobile-pool cohorts, scan batch, scan bytes, candidate bytes, receipt bytes) and the REAL available-bytes reader are checked before any copy/scan/predicate from the dimension-aware accepted native model
(`nfs._estimate`) plus this module's declared terms (water comparison, the soil-state copy, the actual mobile-pool footprint, native-validity scratch, hash scratch, receipt). The zero-dirty path is covered (the continued soil-state copy, checks and receipt exist even
with no dirty column). A declared modelled host payload of THIS module's own stages, NOT a whole-process bound; the native initializer/factory keep their own admission.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_event_refresh as fer
from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_pond as fp
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs
from maple_syrup.sediment_bed import BedContext, BedState, is_windowed_bed

__all__ = ["CONTEXT_SCHEMA", "PROFILE", "RECEIPT_SCHEMA", "WindContext", "WindHandoffCandidate", "WindHandoffError", "WindHandoffLimitError", "WindHandoffLimits", "adopt_wind_handoff", "restore_wind_context", "stage_wind_handoff",
           "start_wind_boundary"]

PROFILE = "maple-syrup-floating-wind-handoff/2"
RECEIPT_SCHEMA = "maple_syrup.floating_wind_handoff_receipt/2"
CONTEXT_SCHEMA = "maple_syrup.floating_wind_context/1"
_SHARED = ("voxel_column", "active_layer", "water", "ledger", "committed_topography", "sediment_availability")
_STATUSES = ("completed", "already_empty")
_CLASS_FIELDS = ("airborne_mass_before_by_class_kg", "airborne_mass_after_by_class_kg", "deposited_mass_by_class_kg", "boundary_export_mass_by_class_kg", "bed_change_by_class_kg", "transfer_identity_gap_by_class_kg",
                 "reservoir_identity_gap_by_class_kg")
_RECORD_KEYS = frozenset({"schema", "shape", "n_classes", "descriptor_sha256", "source_witness", "config_sha256", "representation", "backend"})
_ID_CHARS = 200
_HASH_CHUNK = 1 << 20  # bytes per hash update of a CPU array leaf (a view: no copy)
_HASH_SCRATCH = _HASH_CHUNK + (1 << 16)  # one strided 1-D chunk copy plus digests and the visited-object table
_ADOPT_BYTES = _HASH_SCRATCH + (1 << 16)  # integrity scratch plus records, weak reference and bookkeeping
_MAX_DEPTH = 12


class WindHandoffError(lc.LifecycleError):
    """Typed refusal of the wind handoff (a `LifecycleError`: the native stale/duplicate/wrong-head refusals of the lifecycle are raised unchanged and are caught by the same handler)."""


def _refuse_vegetation(boundary: Any, what: str) -> None:
    """Fail closed on a vegetation-bound boundary (module docstring, VEGETATION): the owning lifecycle's `BoundaryRoot.vegetation` field is the only authority; the check is a single attribute test and performs no read, allocation or callback."""
    if boundary.root.vegetation is not None:
        raise WindHandoffError(f"{what}: the boundary carries an applied vegetation generation; the vegetation-aware wind handoff is not qualified (WH1 is a no-LEAF scope and would drop the generation record, the bound covers and the per-cell law). Nothing was read or changed")


class WindHandoffLimitError(WindHandoffError):
    """A declared explicit limit is exceeded; raised BEFORE any array copy, scan or native work of the refused stage."""


def _plain_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise WindHandoffError(f"{name} must be a plain int >= 1, got {value!r}")
    return value


@dataclass(frozen=True)
class WindHandoffLimits:
    """Explicit caller resource policy (never physics), checked BEFORE the work it bounds: grid cells, voxels per column, grain classes, the cold scan batch, the modelled scan bytes, the modelled candidate bytes, the scalar receipt bytes and the mobile-pool cohort slots
    (`max_cohorts`: the total mass slots of the finalized wind pool, judged from metadata before the native predicate runs)."""

    max_cells: int
    max_voxels: int
    max_classes: int
    max_batch_cells: int
    max_scan_bytes: int
    max_candidate_bytes: int
    max_receipt_bytes: int
    max_cohorts: int

    def __post_init__(self) -> None:
        for f in fields(self):
            _plain_int(getattr(self, f.name), f.name)

    def as_record(self) -> dict:
        return {f.name: int(getattr(self, f.name)) for f in fields(self)}


# ----------------------------------------------------------------------------------------------------------------------------------------------
# minted records
# ----------------------------------------------------------------------------------------------------------------------------------------------
_MINT = threading.local()
_CONTEXTS: weakref.WeakSet = weakref.WeakSet()
_CANDIDATES: weakref.WeakSet = weakref.WeakSet()
_INTEGRITY: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # candidate -> expected integrity digest (PRIVATE: never in the caller-reachable cell)
_LAST: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # lineage -> ONE one-element list holding the last adopted occurrence id (bounded; process-local)


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


def _weak(obj: Any) -> Any:
    return weakref.ref(obj)


class _Record:
    def __copy__(self):
        raise WindHandoffError(f"a {type(self).__name__} cannot be copied")

    def __deepcopy__(self, memo):
        raise WindHandoffError(f"a {type(self).__name__} cannot be copied")

    def __reduce_ex__(self, protocol):
        raise WindHandoffError(f"a {type(self).__name__} is not serialisable")


@dataclass(frozen=True, eq=False)
class WindContext(_Record):
    """The prepared WATER-FREE static context (module docstring): the genuine static `BedContext`, the shape, the native descriptor digest, the ORIGINAL cold source witness and the config identity. Minted only by `start_wind_boundary` / `restore_wind_context`."""

    bed_context: Any
    shape: tuple
    n_classes: int
    descriptor_sha256: str
    source_witness: str
    config_sha256: str

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise WindHandoffError("a WindContext is minted only by start_wind_boundary / restore_wind_context (no direct construction)")
        _CONTEXTS.add(self)

    @property
    def geometry(self) -> Any:
        return self.bed_context.geometry

    @property
    def grain_classes(self) -> Any:
        return self.bed_context.grain_classes

    @property
    def mass_resolution_kg(self) -> float:
        return float(self.bed_context.mass_resolution_kg)

    def record(self) -> dict:
        """The bounded scalar record the caller keeps to rebuild the context in a fresh process (`restore_wind_context`)."""
        return {"schema": CONTEXT_SCHEMA, "shape": [int(self.shape[0]), int(self.shape[1])], "n_classes": int(self.n_classes), "descriptor_sha256": self.descriptor_sha256, "source_witness": self.source_witness,
                "config_sha256": self.config_sha256, "representation": "dense", "backend": "numpy"}


@dataclass(frozen=True, eq=False)
class _Parts:
    """The CURRENT references the adopted `Boundary` is built from (never the parent, its bed, the finalized scheduler, an event or an owner)."""

    column: Any
    matrix_op: Any
    pref_op: Any
    ks_m_s: Any
    depth_m: Any
    history: Any
    water_state: Any
    config: Any
    generation: int
    source_witness: str
    bed: Any
    surface_depth_m: Any
    qx_m2_s: Any
    qy_m2_s: Any
    routing: tuple
    t_s: float
    ledger: Any
    provenance: Any
    seq: int
    cursor: Any
    role: str


@dataclass(frozen=True, eq=False)
class WindHandoffCandidate(_Record):
    """A STAGED, unadopted wind handoff bound weakly to its exact parent boundary and lineage. Holds the bounded scalar receipt and, until adopted, only the current references of the boundary to be minted; it is emptied by adoption. Its expected integrity digest lives in
    a private weak table, not in this object."""

    occurrence_id: str
    end_s: float
    role: str
    receipt_json: str
    _parent_ref: Any
    _lineage_ref: Any
    _cell: Any

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise WindHandoffError("a WindHandoffCandidate is minted only by stage_wind_handoff (no direct construction)")
        _CANDIDATES.add(self)

    def receipt(self) -> dict:
        return json.loads(self.receipt_json)

    @property
    def consumed(self) -> bool:
        return bool(self._cell["consumed"])

    @property
    def accepted_boundary_ref(self) -> Any:
        """READ-ONLY weak reference to the boundary this candidate minted (created BEFORE the head moved; None until an adoption reached that point). The accepted boundary is `ref()` only if `lineage.head is ref()`."""
        return self._cell["accepted"]


# ----------------------------------------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------------------------------------
def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clock(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.floating)):
        raise WindHandoffError(f"{name} must be a float physical clock, got {type(value).__name__}")
    t = float(value)
    if not math.isfinite(t) or t < 0.0:
        raise WindHandoffError(f"{name} must be finite and >= 0, got {t!r}")
    return t


def _occurrence(occurrence_id: Any, occurrence_sha256: Any) -> tuple:
    if type(occurrence_id) is not str or not 1 <= len(occurrence_id) <= _ID_CHARS or not occurrence_id.isprintable():
        raise WindHandoffError(f"occurrence_id must be a printable str of 1..{_ID_CHARS} characters (the unique occurrence identity)")
    if occurrence_sha256 is not None and (type(occurrence_sha256) is not str or len(occurrence_sha256) != 64 or any(c not in "0123456789abcdef" for c in occurrence_sha256)):
        raise WindHandoffError("occurrence_sha256 must be a 64-hex lower-case digest or None")
    return occurrence_id, occurrence_sha256


def _limits(limits: Any) -> WindHandoffLimits:
    if type(limits) is not WindHandoffLimits:
        raise WindHandoffLimitError("explicit WindHandoffLimits are required")
    return limits


def _callback(fn: Any, name: str) -> Any:
    if fn is not None and not callable(fn):
        raise WindHandoffError(f"{name} must be callable or None")
    return fn


def _native_bed(ctx: Any, bed: Any, active: np.ndarray) -> nfs.NativeBed:
    return nfs.NativeBed(bed.voxel_column, bed.active_layer, ctx.geometry, ctx.grain_classes, ctx.mass_resolution_kg, active)


def _bed_meta(bed: Any, ny: int, nx: int, nc: int, what: str) -> int:
    """METADATA of a complete bed (exact host float64 NumPy leaves, ranks and shapes): no value is read, nothing is coerced. Returns the voxel count."""
    if is_windowed_bed(bed):
        raise WindHandoffError(f"the {what} bed is Windowed: only the Dense NumPy bed is qualified")
    try:
        leaves = (("voxel_column.mass_kg", bed.voxel_column.mass_kg, 4), ("active_layer.mass_kg", bed.active_layer.mass_kg, 3), ("water.depth_m", bed.water.depth_m, 2), ("water.mobile_mass_by_cell_class_kg", bed.water.mobile_mass_by_cell_class_kg, 3))
    except AttributeError as exc:
        raise WindHandoffError(f"the {what} bed is not a complete native Dense BedState: {exc}") from None
    for name, a, rank in leaves:
        if type(a) is not np.ndarray or a.dtype != np.float64 or a.ndim != rank:
            raise WindHandoffError(f"the {what} bed's {name} must be a host float64 NumPy array of rank {rank} (CUDA/device arrays and other types are refused; there is no implicit transfer and nothing is coerced)")
    vox, act, dep, mob = (a for _n, a, _r in leaves)
    if vox.shape[:2] != (ny, nx) or vox.shape[3] != nc or act.shape != (ny, nx, nc) or dep.shape != (ny, nx) or mob.shape != (ny, nx, nc):
        raise WindHandoffError(f"the {what} bed's array shapes are not the case's ({ny}, {nx}, {nc})")
    return int(vox.shape[2])


def _dense_numpy(nb: Any, cfg: Any, what: str) -> tuple:
    """Typed refusal of every non-Dense-NumPy representation BEFORE any array work; returns `(N, V, K, n_active)` (metadata only)."""
    try:
        rep = nb.representation
    except (TypeError, ValueError, AttributeError) as exc:
        raise WindHandoffError(f"the {what} bed is not a native Dense bed: {exc}") from None
    if rep != "dense":
        raise WindHandoffError(f"the {what} bed is {rep!r}: only the Dense NumPy bed is qualified (Windowed and other representations are refused)")
    if type(nb.voxel_column.mass_kg) is not np.ndarray or type(nb.active_layer.mass_kg) is not np.ndarray:
        raise WindHandoffError(f"the {what} bed is not host NumPy (CUDA/device arrays are refused; there is no implicit transfer)")
    try:
        return nfs._check_bed(nb, cfg)
    except nfs.NativeFloatingError as exc:
        raise WindHandoffError(f"the {what} bed is refused by the native checks: {exc}") from None


def _same_bits(a: Any, b: Any) -> bool:
    if type(a) is not np.ndarray or type(b) is not np.ndarray or a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.ndim == 0:
        return a.tobytes() == b.tobytes()
    for i in range(a.shape[0]):  # row by row: bounded scratch of two rows, never a whole-array copy
        if a[i].tobytes() != b[i].tobytes():
            return False
    return True


def _same_leaves(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if is_dataclass(a) and not isinstance(a, type):
        return all(_same_leaves(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same_leaves(x, y) for x, y in zip(a, b, strict=True))
    if type(a) is np.ndarray:
        return _same_bits(a, b)
    return bool(a == b)


def _stamp(bed: Any, name: str) -> float:
    t = getattr(getattr(bed, "committed_topography", None), "last_commit_time_s", None)
    if isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, float, np.integer, np.floating)) or not math.isfinite(float(t)) or float(t) < 0.0:
        raise WindHandoffError(f"the {name} bed's committed-topography commit stamp is not a finite non-negative clock")
    return float(t)


def _descriptor(ctx: Any, nb: nfs.NativeBed, what: str) -> None:
    try:
        d = nfs._descriptor(nb)
    except (TypeError, ValueError, AttributeError, KeyError) as exc:
        raise WindHandoffError(f"the {what} bed's native descriptor cannot be read: {exc}") from None
    if d != ctx.descriptor_sha256:
        raise WindHandoffError(f"the {what} bed's native descriptor (grid, geometry, ordered grain classes, mass resolution, active-mask pattern) is not the accepted source's")


def _native_valid(bed: Any, bed_context: Any, t_s: float, what: str) -> None:
    """FULL actual native bed validity through the owning API the pond checkpoint uses (`pond_checkpoint._native_checks`: bed support, active-layer/voxel partition, availability, ledger / committed topography / water validators, reconciliation context and the ledger
    reconciliation with the physical state at `t_s`). No formula is copied; an unrecorded class/mass change is refused here, never repaired."""
    from maple_syrup import pond_checkpoint as pc

    try:
        pc._native_checks(bed, SimpleNamespace(context=bed_context), float(t_s), "dense")
    except Exception as exc:  # noqa: BLE001 -- every native validator failure (and an allocation failure) is a typed refusal of the handoff
        raise WindHandoffError(f"the {what} bed is not a valid native bed (the actual MAPLE validators / ledger reconciliation at the clock refuse it): {exc}") from None


# ----------------------------------------------------------------------------------------------------------------------------------------------
# post-stage integrity: bounded, copyless, chunked hashes of CPU array leaves and metadata
# ----------------------------------------------------------------------------------------------------------------------------------------------
def _feed_array(h: Any, a: np.ndarray) -> None:
    if a.dtype.hasobject:
        raise WindHandoffError("an object array cannot be part of the integrity digest")
    h.update(f"A{a.dtype.str}{a.shape};".encode())
    if a.size == 0:
        return
    if a.ndim == 0:
        h.update(a.tobytes())
        return
    if a.flags.c_contiguous:
        flat = a.reshape(-1).view(np.uint8)  # a VIEW: no copy
        for off in range(0, flat.size, _HASH_CHUNK):
            h.update(flat[off:off + _HASH_CHUNK])
        return
    if a.ndim == 1:  # strided: copy at most one chunk at a time
        step = max(1, _HASH_CHUNK // a.dtype.itemsize)
        for off in range(0, a.shape[0], step):
            h.update(np.ascontiguousarray(a[off:off + step]).view(np.uint8))
        return
    for i in range(a.shape[0]):
        _feed_array(h, a[i])


def _feed(h: Any, v: Any, seen: dict, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise WindHandoffError("the state is too deeply nested for the integrity digest")
    if v is None:
        h.update(b"N;")
    elif type(v) is np.ndarray:
        if id(v) in seen:
            h.update(f"R{seen[id(v)]};".encode())
            return
        seen[id(v)] = len(seen)
        _feed_array(h, v)
    elif isinstance(v, (bool, int, float, str, bytes, np.generic)):
        h.update(f"{type(v).__name__}:{v!r};".encode())
    elif is_dataclass(v) and not isinstance(v, type):
        if id(v) in seen:
            h.update(f"R{seen[id(v)]};".encode())
            return
        seen[id(v)] = len(seen)
        h.update(f"D{type(v).__qualname__}(".encode())
        for f in fields(v):
            h.update(f.name.encode())
            _feed(h, getattr(v, f.name), seen, depth + 1)
        h.update(b");")
    elif isinstance(v, (tuple, list)):
        h.update(f"T{len(v)}(".encode())
        for x in v:
            _feed(h, x, seen, depth + 1)
        h.update(b");")
    elif isinstance(v, Mapping):
        h.update(f"M{len(v)}(".encode())
        for k in sorted(v, key=repr):
            h.update(repr(k).encode())
            _feed(h, v[k], seen, depth + 1)
        h.update(b");")
    else:  # an opaque static object (topology, geometry, class set, operator law holder): its type and its own content identity when it has one
        h.update(f"O{type(v).__qualname__}".encode())
        for name in ("sha256", "topology_sha256"):
            ident = getattr(v, name, None)
            if isinstance(ident, str):
                h.update(f"|{name}:{ident}".encode())
        h.update(b";")


def _integrity(parent: Any, parts: _Parts, oid: str, end_s: float, text: str) -> str:
    """The expected-integrity digest: the CURRENT parent state and the proposed state (module docstring). Metadata plus copyless chunked reads of CPU leaves; nothing is copied or retained."""
    root = parent.root
    h = hashlib.sha256(b"maple_syrup.floating_wind_handoff.integrity/1")
    seen: dict = {}
    _feed(h, (parent.role, parent.t_s, parent.seq, parent.bed, parent.surface_depth_m, parent.qx_m2_s, parent.qy_m2_s, parent.routing, parent.ledger, parent.provenance, parent.cursor, root.column, root.matrix_op, root.pref_op, root.ks_m_s,
              root.depth0_m, root.depth_m, root.history, root.water_state, root.config, root.generation, root.source_witness), seen)
    _feed(h, (oid, float(end_s).hex(), text), seen)
    _feed(h, parts, seen)
    return h.hexdigest()


def _initial_integrity(boundary: Any, context: Any) -> str:
    h = hashlib.sha256(b"maple_syrup.floating_wind_handoff.initial_integrity/1")
    _feed(h, (boundary, context), {})
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------------------------------------
# the wind proof: metadata (before admission) and values (after admission)
# ----------------------------------------------------------------------------------------------------------------------------------------------
def _pool_footprint(pool: Any) -> tuple:
    """`(bytes, slots)` of the finalized wind mobile pool from METADATA only (exact native flat / calendar type; every array an exact host NumPy array). Nothing is read or compared."""
    try:
        from maple.core.types.cohort import MobileCohortState
        from maple.core.types.cohort_calendar import SegmentedCohortCalendarState
    except ImportError as exc:
        raise WindHandoffError(f"the native MAPLE mobile-pool types are unavailable: {exc}") from None
    if type(pool) not in (MobileCohortState, SegmentedCohortCalendarState):
        raise WindHandoffError(f"the wind mobile pool type {type(pool).__name__} is not a qualified native pool (flat MobileCohortState or SegmentedCohortCalendarState)")
    total, slots = [0, 0], [0]

    def walk(obj: Any, name: str, depth: int) -> None:
        if depth > 6:
            raise WindHandoffError("the wind mobile pool is nested deeper than the qualified native layouts")
        if type(obj) is np.ndarray:
            total[0] += int(obj.nbytes)
            if name == "mass_kg":
                slots[0] += int(obj.size)
        elif is_dataclass(obj) and not isinstance(obj, type):
            for f in fields(obj):
                walk(getattr(obj, f.name), f.name, depth + 1)
        elif isinstance(obj, (tuple, list)):
            for x in obj:
                walk(x, name, depth + 1)
        elif isinstance(obj, Mapping):
            for key, x in obj.items():
                walk(x, str(key), depth + 1)
        elif hasattr(obj, "__cuda_array_interface__") or (hasattr(obj, "nbytes") and not isinstance(obj, np.generic)):
            raise WindHandoffError("the wind mobile pool holds a non-NumPy (device) array: it is refused before any dispatch or transfer")

    walk(pool, "pool", 0)
    return total[0], slots[0]


def _wind_meta(final: Any, child: Any, parent_bed: Any, start_s: float, ctx: WindContext, limits: WindHandoffLimits) -> dict:
    """Everything of the wind proof decidable WITHOUT reading an array value: classes, flags, shared identity, clocks, pool type/footprint/cohort count, class-vector metadata, aliasing bounds and commit stamps. Nothing is mutated."""
    try:
        from maple.aeolian.scheduler.finalization import WindEventFinalizationResult
    except ImportError as exc:
        raise WindHandoffError(f"the native MAPLE wind types are unavailable: {exc}") from None
    if type(final) is not WindEventFinalizationResult:
        raise WindHandoffError("final must be the actual native MAPLE WindEventFinalizationResult (no boolean, JSON stub or duck type)")
    if final.completed is not True or final.water_handoff_ready is not True or final.status not in _STATUSES or tuple(final.reasons) != ():
        raise WindHandoffError(f"the native wind finalization is not completed and water-handoff ready (status {final.status!r}, reasons {tuple(final.reasons)!r})")
    if type(child) is not BedState:
        raise WindHandoffError("child must be the complete native BedState of the finalized wind result")
    sched = final.state
    phys = getattr(sched, "physical", None)
    if phys is None:
        raise WindHandoffError("the finalized scheduler state carries no physical state")
    for name in _SHARED:
        got = getattr(child, name)
        if got is None or getattr(phys, name) is not got:
            raise WindHandoffError(f"child.{name} is not the finalized result's own {name} (the shared fields must be exactly the child's)")
    end_s = _clock(final.time_s, "final.time_s")
    if float(phys.time_s).hex() != end_s.hex():
        raise WindHandoffError("the finalized physical clock differs from the result's terminal time")
    if not end_s > start_s:
        raise WindHandoffError(f"the native wind end {end_s!r} is not strictly after the start {start_s!r}")
    if float(final.block_offset_s) != 0.0 or float(sched.block_offset_s) != 0.0 or int(final.block_index) != int(sched.block_index) or int(sched.block_index) < 1:
        raise WindHandoffError("the native record cursor is not at a completed terminal block (offset 0, at least one block)")
    pool_bytes, slots = _pool_footprint(phys.mobile_state)
    if slots > limits.max_cohorts:
        raise WindHandoffLimitError(f"the wind mobile pool has {slots} cohort slots above max_cohorts {limits.max_cohorts}; no predicate ran")
    nc = int(ctx.n_classes)
    vec: dict = {}
    for name in _CLASS_FIELDS:
        v = getattr(final, name)
        if type(v) is not np.ndarray or v.dtype != np.float64 or v.shape != (nc,):
            raise WindHandoffError(f"final.{name} must be a host float64 ({nc},) class vector (nothing is coerced before admission)")
        vec[name] = v
    if child is parent_bed or child.voxel_column is parent_bed.voxel_column or child.active_layer is parent_bed.active_layer or np.may_share_memory(child.voxel_column.mass_kg, parent_bed.voxel_column.mass_kg) \
            or np.may_share_memory(child.active_layer.mass_kg, parent_bed.active_layer.mass_kg):
        raise WindHandoffError("the child shares native arrays with the parent bed: a wind child must be a separate staged state (an aliased parent could not be compared)")
    p_stamp, c_stamp = _stamp(parent_bed, "parent"), _stamp(child, "child")
    if not (p_stamp <= start_s and p_stamp <= c_stamp <= end_s):
        raise WindHandoffError(f"the shared-bed commit stamps are not ordered (parent {p_stamp!r} <= start {start_s!r}, parent <= child {c_stamp!r} <= end {end_s!r})")
    return {"end_s": end_s, "vec": vec, "pool_bytes": pool_bytes, "slots": slots, "phys": phys}


def _wind_values(final: Any, meta: dict) -> dict:
    """The value checks of the wind proof, run ONLY after the aggregate admission (native predicate, finiteness, closure bounds). Returns the class vectors."""
    from maple.aeolian.cohorts.dispatch import mobile_state_is_genuinely_empty

    try:
        empty = bool(mobile_state_is_genuinely_empty(meta["phys"].mobile_state))
    except (TypeError, ValueError) as exc:
        raise WindHandoffError(f"the wind mobile pool cannot be judged by the native predicate: {exc}") from None
    if not empty:
        raise WindHandoffError("the finalized wind mobile pool is not genuinely empty")
    vec = meta["vec"]
    nc = next(iter(vec.values())).shape[0]
    for name, v in vec.items():
        if not bool(np.isfinite(v).all()):
            raise WindHandoffError(f"final.{name} must be a finite ({nc},) class vector")
    if bool(np.any(vec["airborne_mass_after_by_class_kg"] != 0.0)):
        raise WindHandoffError("the finalizer leaves airborne mass")
    for prefix in ("transfer", "reservoir"):
        bound = float(getattr(final, prefix + "_identity_bound_kg"))
        if not math.isfinite(bound) or bound < 0.0 or not bool(np.all(np.abs(vec[prefix + "_identity_gap_by_class_kg"]) <= bound)):
            raise WindHandoffError(f"the finalizer's {prefix} class identity is not closed within its native bound")
    return vec


def _dirty_columns(parent_nb: nfs.NativeBed, child_nb: nfs.NativeBed, active_flat: np.ndarray, B: int) -> np.ndarray:
    """ONE cold streamed pass over the active columns (batches of `B`): the flat cells whose native rows (every voxel/class, bit for bit, plus the active-layer mass) differ between the parent and the child, through the ACTUAL native readers/comparison."""
    dirty = []
    for b0 in range(0, active_flat.size, B):
        idx = active_flat[b0:b0 + B]
        vc, ac = nfs._rows(child_nb, idx, B)
        vp, ap = nfs._rows(parent_nb, idx, B)
        changed = nfs._changed_rows(vc, vp)
        changed |= np.any(ac.view(np.uint64) != ap.view(np.uint64), axis=1)
        dirty.append(idx[changed])
        del vc, ac, vp, ap
    out = np.concatenate(dirty) if dirty else np.zeros(0, dtype=np.int64)
    return out.astype(np.int64, copy=False)


def _seal(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


# ----------------------------------------------------------------------------------------------------------------------------------------------
# prepared context
# ----------------------------------------------------------------------------------------------------------------------------------------------
def _mint_context(bed_context: Any, nb: nfs.NativeBed, source_witness: str, config_sha256: str) -> WindContext:
    ny, nx = int(bed_context.geometry.ny), int(bed_context.geometry.nx)
    with _minting():
        return WindContext(bed_context, (ny, nx), int(nb.active_layer.mass_kg.shape[2]), nfs._descriptor(nb), source_witness, config_sha256)


def restore_wind_context(boundary: Any, bed_context: Any, expected_record: Any) -> WindContext:
    """Rebuild the water-free context in a fresh process from a VERIFIED restored `Boundary`, the original static `BedContext` and the expected scalar record (`WindContext.record()`). The descriptor is recomputed from the boundary's actual bed and the static case
    (metadata and the active-mask pattern only: no row is hashed and no initial terrain is regenerated); it, the boundary root's original cold witness and the config identity must equal the record, else a typed refusal."""
    lc._record(boundary, lc.Boundary)
    _refuse_vegetation(boundary, "restore_wind_context")
    if type(expected_record) is not dict or set(expected_record) != _RECORD_KEYS or expected_record["schema"] != CONTEXT_SCHEMA or expected_record["representation"] != "dense" or expected_record["backend"] != "numpy":
        raise WindHandoffError("the expected context record is malformed (closed key set, dense numpy)")
    root = boundary.root
    if root.column.backend != "numpy" or root.host_active is not None:
        raise WindHandoffError("only a Dense NumPy boundary has a wind context")
    if not isinstance(bed_context, BedContext):
        raise WindHandoffError("bed_context must be the static native BedContext of the case")
    ny, nx = int(bed_context.geometry.ny), int(bed_context.geometry.nx)
    if root.column.n_cells != ny * nx:
        raise WindHandoffError("the static case grid differs from the boundary's cell count")
    nb = _native_bed(bed_context, boundary.bed, np.asarray(root.column._maps["active"]).reshape(ny, nx))
    _dense_numpy(nb, root.config, "restored")
    ctx = _mint_context(bed_context, nb, root.source_witness, root.config.sha256)
    if ctx.record() != expected_record:
        raise WindHandoffError("the rebuilt wind context differs from the expected verified record (descriptor, source witness, config identity, shape or class count)")
    return ctx


# ----------------------------------------------------------------------------------------------------------------------------------------------
# before the first wind: a real initial boundary
# ----------------------------------------------------------------------------------------------------------------------------------------------
def start_wind_boundary(owner: Any, bed: Any, event_factory: Any, *, bed_context: Any, limits: Any, available_bytes_fn: Any, preferential: Any = None, reserve_bytes: int = 0, batch_cells: int = nfs.DEFAULT_BATCH_CELLS,
                        before_publish: Any = None) -> tuple:
    """Publish a REAL initial `Boundary` from a genuinely CURRENT Dense NumPy `NativeFloatingSoil` and the complete actual `BedState` (module docstring). Returns `(lineage, boundary, wind_context)`. The prelude is planned from `bed_context` and admitted (limits, then the actual
    reader) BEFORE the vertical/event/initial state exist; every refusable step precedes the owner's retirement (LAST)."""
    limits = _limits(limits)
    if type(owner) is not nfs.NativeFloatingSoil or not owner.is_current:
        raise WindHandoffError("owner must be a genuine CURRENT NativeFloatingSoil")
    if owner.column.backend != "numpy":
        raise WindHandoffError(f"the wind handoff serves the Dense NumPy owner; backend {owner.column.backend!r} is refused before any work")
    if not callable(event_factory) or not callable(available_bytes_fn):
        raise WindHandoffError("event_factory and available_bytes_fn must be callable")
    before_publish = _callback(before_publish, "before_publish")
    if type(bed) is not BedState:
        raise WindHandoffError("bed must be the complete native BedState (not a NativeBed or a partial record)")
    if not isinstance(bed_context, BedContext):
        raise WindHandoffError("bed_context must be the genuine static native BedContext of the case")
    reserve = fsl._count(reserve_bytes, "reserve_bytes")
    B0 = fsl._count(batch_cells, "batch_cells", minimum=1)
    if B0 > limits.max_batch_cells:
        raise WindHandoffLimitError(f"batch_cells {B0} exceeds max_batch_cells {limits.max_batch_cells}")
    ny, nx = int(bed_context.geometry.ny), int(bed_context.geometry.nx)
    nc = len(bed_context.grain_classes.classes)
    N = ny * nx
    n_vox = _bed_meta(bed, ny, nx, nc, "initial")
    if int(owner.column.n_cells) != N:
        raise WindHandoffError("the static case grid differs from the owner's cell count")
    if N > limits.max_cells or n_vox > limits.max_voxels or nc > limits.max_classes:
        raise WindHandoffLimitError(f"{N} cells / {n_vox} voxels / {nc} classes exceed the declared limits")
    cfg = owner.config
    nb0 = _native_bed(bed_context, bed, np.asarray(owner.column._maps["active"]).reshape(ny, nx))
    _dense_numpy(nb0, cfg, "initial")
    # ---- the PRELUDE plan (metadata only): the native initialization model (cold scan / witness, compact reader, provider, persistent maps) plus this module's terms
    try:
        model = nfs.estimate_native_initialization_memory(nb0, cfg, batch_cells=B0, reserve_bytes=reserve)
    except nfs.NativeFloatingError as exc:
        raise WindHandoffError(f"the native memory plan refuses the initial bed: {exc}") from None
    check_scratch = int(bed.voxel_column.mass_kg.nbytes) + int(bed.active_layer.mass_kg.nbytes)
    extra = N * nc + N + check_scratch + _HASH_SCRATCH  # water-borne comparison booleans, context descriptor (mask), native-validity scratch
    need = int(model["required_bytes"]) + extra
    if need > limits.max_scan_bytes:
        raise WindHandoffLimitError(f"the start prelude (initial cold witness, context, native validity) needs a modelled {need} B above max_scan_bytes {limits.max_scan_bytes}; nothing was read")
    nfs._admit({"required_bytes": need, "terms": {"native_initialization": int(model["required_bytes"]), "context_and_validity_scratch": extra}}, None, available_bytes_fn, "wind start prelude")
    # ---- values, only after admission
    if bool(np.any(bed.water.mobile_mass_by_cell_class_kg != 0.0)):
        raise WindHandoffError("the initial bed carries water-borne sediment: a complete boundary has none (nothing is erased or reset)")
    t0 = float(owner.state.t_s)
    _native_valid(bed, bed_context, t0, "initial")
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=available_bytes_fn, preferential=preferential, reserve_bytes=reserve, batch_cells=B0)
    event = event_factory(vert)  # the caller's static factory, outside every mint window
    lc._admit_native(event, vert)
    if event.control.bed_storage != "dense" or event.control.backend != "numpy":
        raise WindHandoffError("the static event must be the Dense NumPy event")
    if tuple(event.shape) != (ny, nx):
        raise WindHandoffError("the static event's grid differs from bed_context")
    state = event.initial_state(bed, vert.initial_soil_water(), t_s=t0)  # the actual full validation and cold source witness, owner still current
    lc._admit_state(event, vert, state, bed, t0)
    held = vert.held_template()
    ctx = _mint_context(bed_context, nb0, held.source_witness, cfg.sha256)  # the descriptor of the REQUIRED bed_context (the preflight bed) ...
    nb = _native_bed(event.context, bed, np.array(vert._active_host().reshape(ny, nx)))
    _dense_numpy(nb, cfg, "initial")
    if nfs._descriptor(nb) != ctx.descriptor_sha256:  # ... must be the static event's own case
        raise WindHandoffError("the static event's native descriptor differs from bed_context's (another case)")
    area = float(owner.column.cell_area_m2)
    ledger = {"storage0_m3": lc._storage_m3(state), "rain_m3": 0.0, "drainage_m3": 0.0, "outlet_export_m3": 0.0, "bottom_export_m3": 0.0, "surface_return_m3": 0.0, "bound_m3": 0.0, "n_refreshes": 0, "n_occurrences": 0}
    prov = {"cold_anchor_witness": held.source_witness, "property_basis": "initial_full", "execution_epoch": None, "cold_content_hash": None, "area_m2": area}
    lineage = lc.FloatingEventLineage()
    with lc._minting():
        root = lc.BoundaryRoot(owner.column, owner.matrix_op, preferential, owner.ks_m_s, owner.depth_m, owner.depth_m, owner.matrix_history, owner.state, cfg, int(owner.binding.generation), held.source_witness, bed)
        boundary = lc.Boundary("boundary", root, bed, state.hydraulic.depth_m, state.hydraulic.qx_m2_s, state.hydraulic.qy_m2_s, (state.topology, state.held_source_m, state.datum), t0, ledger, prov, 0, None, None, None)
    proposal = (lineage, boundary, ctx)  # allocate the return carrier before retirement
    expected_initial = _initial_integrity(boundary, ctx)
    if before_publish is not None:
        # The coordinator can stage its own binding/receipt before retirement.
        # It sees a proposal, never a second accepted water authority.
        before_publish(lineage, boundary, ctx)
        if _initial_integrity(boundary, ctx) != expected_initial:
            raise WindHandoffError("initial boundary or context changed during before_publish; nothing was published")
    if not owner.is_current:
        raise WindHandoffError("the owner was superseded during the start")
    stale_refusal = False
    try:
        nfs.retire_native_owner_to_lineage(owner)
    except nfs.NativeStaleError:
        stale_refusal = True
        raise
    finally:
        # A failure before retirement leaves the old authority current. If an
        # interrupt arrives just after retirement, expose the real new owner.
        # A native stale-owner refusal never transfers authority to this start.
        if not stale_refusal and owner._authority.current is None:
            lineage._head = boundary
    return proposal


# ----------------------------------------------------------------------------------------------------------------------------------------------
# the staged transaction
# ----------------------------------------------------------------------------------------------------------------------------------------------
def stage_wind_handoff(lineage: Any, parent: Any, final: Any, child: Any, *, context: Any, occurrence_id: Any, start_s: Any, limits: Any, available_bytes_fn: Any, occurrence_sha256: Any = None, reserve_bytes: int = 0,
                       batch_cells: int = nfs.DEFAULT_BATCH_CELLS) -> WindHandoffCandidate:
    """PURE staging of one wind handoff (module docstring). Order: head/eligibility -> typed backend/representation/limits and input METADATA (no coercion, no scan) -> aggregate admission (limits, then the real reader) -> pool predicate and class-vector values ->
    water carry -> full native bed validity -> descriptor -> cold row comparison -> candidate admission at the actual dirty count -> selected read, changed set, reconcile, closure -> new root/state -> surface return -> cumulative water identity -> bounded receipt ->
    integrity digest. Any refusal leaves the head, the parent, the child and every input untouched."""
    lc._expect(lineage, parent, lc.Boundary)
    _refuse_vegetation(parent, "stage_wind_handoff")
    limits = _limits(limits)
    oid, osha = _occurrence(occurrence_id, occurrence_sha256)
    if type(context) is not WindContext or context not in _CONTEXTS:
        raise WindHandoffError("context must be a genuine WindContext")
    if not callable(available_bytes_fn):
        raise WindHandoffError("available_bytes_fn must be callable (the admission needs a real reader)")
    if not parent.eligible_for_interevent:
        raise WindHandoffError("a pending recession boundary is not wind eligible until a real native recession completes again")
    if parent.pending_forcing is not None or parent.storm_match is not None or (parent.cursor is not None and parent.cursor.pending is not None):
        raise WindHandoffError("a boundary with a pending storm is not wind eligible")
    last = _LAST.get(lineage)
    if last is not None and last[0] == oid:
        raise WindHandoffError("this occurrence identity is the lineage's last adopted one (an immediate duplicate; the schedule cursor owns cross-event identity)")
    root = parent.root
    cfg = root.config
    if root.column.backend != "numpy" or root.host_active is not None:
        raise WindHandoffError(f"the wind handoff serves the Dense NumPy boundary; backend {root.column.backend!r} is refused before any work")
    if parent.provenance.get("cold_anchor_witness") != context.source_witness or root.source_witness != context.source_witness or cfg.sha256 != context.config_sha256:
        raise WindHandoffError("the context is not bound to this boundary's original cold source witness / native config")
    start = _clock(start_s, "start_s")
    if start.hex() != float(parent.t_s).hex() or float(root.water_state.t_s).hex() != float(parent.t_s).hex():
        raise WindHandoffError(f"the wind start {start!r} is not the accepted boundary's physical clock {float(parent.t_s)!r} (the soil clock must equal it too)")
    reserve = fsl._count(reserve_bytes, "reserve_bytes")
    B0 = fsl._count(batch_cells, "batch_cells", minimum=1)
    if B0 > limits.max_batch_cells:
        raise WindHandoffLimitError(f"batch_cells {B0} exceeds max_batch_cells {limits.max_batch_cells}")
    ny, nx = context.shape
    N, nc, L = ny * nx, int(context.n_classes), int(root.column.n_layers)
    if root.column.n_cells != N or N > limits.max_cells or nc > limits.max_classes:
        raise WindHandoffLimitError(f"{N} cells / {nc} classes exceed the declared limits or the context's grid")
    if type(child) is not BedState:
        raise WindHandoffError("child must be the complete native BedState of the finalized wind result")
    # ---- METADATA of every input (no array value read, nothing coerced, no native predicate)
    n_vox = max(_bed_meta(parent.bed, ny, nx, nc, "parent"), _bed_meta(child, ny, nx, nc, "child"))
    if n_vox > limits.max_voxels:
        raise WindHandoffLimitError(f"{n_vox} voxels exceed max_voxels {limits.max_voxels}")
    meta = _wind_meta(final, child, parent.bed, start, context, limits)
    end_s = meta["end_s"]
    active = np.asarray(root.column._maps["active"]).reshape(ny, nx)
    pnb, cnb = _native_bed(context, parent.bed, active), _native_bed(context, child, active)
    _dense_numpy(pnb, cfg, "parent")
    _dense_numpy(cnb, cfg, "child")
    # ---- the aggregate admission (limits, then the REAL reader) of the cold scan, the pool predicate, the native validity, the soil-state copy and the receipt: BEFORE any value is read; the zero-dirty path is covered
    try:
        model0 = nfs._estimate(pnb, cnb, cfg, B0, reserve)
    except nfs.NativeFloatingError as exc:
        raise WindHandoffError(f"the native memory plan refuses the beds: {exc}") from None
    validity = int(child.voxel_column.mass_kg.nbytes) + int(child.active_layer.mass_kg.nbytes)  # declared scratch of the native validators
    scan_need = int(model0["terms"]["scan"]) + N * (nc + 2) + int(meta["pool_bytes"]) + validity + _HASH_SCRATCH + reserve
    base_cand = (L + 1) * N * 8 + 3072 + 112 * nc  # the continued soil-state copy, the surface/return array and the scalar receipt (exist even with no dirty column)
    if scan_need > limits.max_scan_bytes:
        raise WindHandoffLimitError(f"the cold scan, pool predicate and validity need a modelled {scan_need} B above max_scan_bytes {limits.max_scan_bytes}; nothing was read")
    if base_cand > limits.max_candidate_bytes:
        raise WindHandoffLimitError(f"the candidate (soil-state copy, checks and receipt, even with no dirty column) needs a modelled {base_cand} B above max_candidate_bytes {limits.max_candidate_bytes}; nothing was copied")
    nfs._admit({"required_bytes": scan_need + base_cand, "terms": {"scan_pool_validity": scan_need, "candidate_base": base_cand, "reserve": reserve}}, None, available_bytes_fn, "wind handoff cold comparison and candidate")
    B = int(model0["batch_cells"])
    # ---- values (only now)
    cls = _wind_values(final, meta)
    if not _same_leaves(parent.bed.water, child.water):
        raise WindHandoffError("the child's surface-water leaves differ from the parent's: wind must carry every water leaf bit for bit (no wind water physics)")
    if bool(np.any(parent.bed.water.mobile_mass_by_cell_class_kg != 0.0)):
        raise WindHandoffError("the parent carries water-borne sediment: a dry wind entry needs none")
    _native_valid(child, context.bed_context, end_s, "child")
    _descriptor(context, pnb, "parent")
    _descriptor(context, cnb, "child")
    dirty = _dirty_columns(pnb, cnb, np.flatnonzero(active.reshape(-1)).astype(np.int64), B)
    # ---- stage 1: the candidate (selected read, reconcile, rebinding), admitted at the actual dirty count
    maps = root.column._maps
    d_cur = np.asarray(root.depth_m)
    old_water = np.asarray(root.water_state.water_m)
    t_old = float(root.water_state.t_s)
    area = nfs._area(cnb)
    new_column, new_op, new_hist, new_pref, ks_out, depth_out = root.column, root.matrix_op, root.history, root.pref_op, root.ks_m_s, root.depth_m
    water, surf, receipt, n_changed, n_depth, ks_changed, model1 = old_water, None, None, 0, 0, False, None
    if dirty.size:
        try:
            model1 = nfs._estimate(None, cnb, cfg, B0, reserve, n_selected=int(dirty.size))
        except nfs.NativeFloatingError as exc:
            raise WindHandoffError(f"the native memory plan refuses the selected refresh: {exc}") from None
        terms = model1["terms"]
        need1 = int(terms["persistent_candidate"]) + int(terms["child_native_and_properties"]) + 3 * N * 8 + base_cand + _HASH_SCRATCH + reserve
        if need1 > limits.max_candidate_bytes:
            raise WindHandoffLimitError(f"the candidate needs a modelled {need1} B above max_candidate_bytes {limits.max_candidate_bytes}; nothing was copied")
        nfs._admit({"required_bytes": need1, "terms": {"persistent_candidate": terms["persistent_candidate"], "child_native_and_properties": terms["child_native_and_properties"], "dirty_and_index": 3 * N * 8, "candidate_base": base_cand,
                                                        "integrity_scratch": _HASH_SCRATCH, "reserve": reserve}}, None, available_bytes_fn, "wind handoff selected property refresh")
        B1 = int(model1["batch_cells"])
        fc, sat, depth = (np.array(maps["theta_field_capacity"], copy=True), np.array(maps["theta_saturation"], copy=True), np.array(d_cur, copy=True))
        ks = None if root.ks_m_s is None else np.array(root.ks_m_s, copy=True)
        nfs._read_texture(cnb, cfg, dirty, fc, sat, depth, model=model1, batch_cells=B1, update_properties=True, ks=ks)  # the compact reader on ONLY the dirty columns of the CURRENT child
        changed, depth_changed = nfs._changed_columns_parts(cfg, maps, d_cur, dirty, fc, sat, depth, B1)
        n_changed, n_depth = int(changed.size), int(depth_changed.size)
        ks_changed = ks is not None and bool(np.any(ks[dirty] != root.ks_m_s[dirty]))
        if n_changed:
            water, surf_c, exp_c, acc = nfs._reconcile_changed(cfg, maps, d_cur, old_water, fc, sat, depth, changed, area, B1, t_old)
            new_column = nfs._build_column(cfg, fc, sat, depth, cnb)
            mask = np.asarray(new_column._maps["active"])
            try:
                w_before, w_after, resid, bound = nfs._inventory_closure(water, old_water, mask, changed, B1, area, acc, surf_c, exp_c)
            except nfs.NativeFloatingError as exc:
                raise WindHandoffError(f"the whole-inventory closure of the reconcile is violated: {exc}") from None
            if ks is not None:
                ks.flags.writeable = False
            new_op = nfs._fresh_op(cfg, new_column, ks)
            new_hist = fm.rebind_floating_matrix_history(root.matrix_op, root.history, new_op).history
            new_pref = None if root.pref_op is None else fpi.rebind_preferential_column(root.pref_op, new_column)
            for fresh in (depth, surf_c, exp_c):
                _seal(fresh)
            depth_out, surf, ks_out = depth, surf_c, (ks if ks is not None else root.ks_m_s)
            receipt = nfs.NativeReconcileReceipt(whole_before_m3=w_before, whole_after_m3=w_after, whole_residual_m3=resid, whole_bound_m3=bound, old_column_sha256=root.column.sha256, new_column_sha256=new_column.sha256, **acc)
        elif n_depth or ks_changed:  # equivalent hydraulic geometry and/or Ks only: nothing reconciled, exported or returned (the accepted E3c / Ks-only rule)
            if n_depth:
                _seal(depth)
                new_column = nfs._build_column(cfg, np.asarray(maps["theta_field_capacity"]), np.asarray(maps["theta_saturation"]), depth, cnb)
                depth_out = depth
            if ks is not None:
                _seal(ks)
            new_op = nfs._fresh_op(cfg, new_column, ks)
            new_hist = fm.rebind_floating_matrix_history(root.matrix_op, root.history, new_op).history
            if new_column is not root.column:
                new_pref = None if root.pref_op is None else fpi.rebind_preferential_column(root.pref_op, new_column)
            ks_out = ks if ks_changed else root.ks_m_s
    replaced = new_op is not root.matrix_op or new_column is not root.column
    # ---- the SAME W (reconciled columns only) carried across the REAL wind interval: the owning public constructor, an explicit event-time carry (no E/T, no hydrology, no flux step)
    new_state = fsl.floating_continued_state(new_column, np.ascontiguousarray(water), end_s)
    ret_total = 0.0 if surf is None else float(surf.sum())
    bed = child
    surface = child.water.depth_m
    if ret_total != 0.0:  # the capacity return joins the actual surface ONCE
        surface = np.array(child.water.depth_m, copy=True) + surf.reshape(ny, nx)
        bed = replace(child, water=replace(child.water, depth_m=surface))
    role = "pending_recession" if ret_total > 0.0 else "boundary"
    led = dict(parent.ledger)
    led["n_refreshes"] = int(led["n_refreshes"]) + 1
    if receipt is not None:
        led["bottom_export_m3"] += float(receipt.bottom_export_m3)  # exactly once (the native drainage / outlet entries are untouched: wind runs no hydrology)
        led["surface_return_m3"] += float(receipt.surface_return_m3)  # INTERNAL transfer soil -> surface: never a net loss
        led["bound_m3"] += float(receipt.whole_bound_m3)
        soil_after = float(receipt.whole_after_m3)
    else:
        soil_after = float(fsl.floating_storage_m3(new_column, new_state))
    resid, bound = lc._balance_check(led, soil_after + float(np.asarray(surface).sum()) * float(area))  # the native cumulative lifecycle identity; a refusal here precedes every mutation
    led["residual_m3"], led["residual_bound_m3"] = float(resid), float(bound)
    prev_epoch = parent.provenance.get("execution_epoch") or ""
    digest = hashlib.sha256("|".join(("wind", oid, start.hex(), end_s.hex(), new_column.sha256, str(int(dirty.size)), str(n_changed))).encode()).hexdigest()
    epoch = hashlib.sha256(f"{fer.EPOCH_KIND}|{prev_epoch}|{digest}".encode()).hexdigest()
    prov = dict(parent.provenance)
    prov.update({"property_basis": fer.PROPERTY_BASIS if replaced else parent.provenance["property_basis"], "execution_epoch": epoch, "cold_content_hash": None, "area_m2": float(area)})
    generation = int(root.generation) + (1 if replaced else 0)
    hexv = lambda v: [float(x).hex() for x in v]  # noqa: E731
    ret_m3 = 0.0 if receipt is None else float(receipt.surface_return_m3)
    exp_m3 = 0.0 if receipt is None else float(receipt.bottom_export_m3)
    bnd_m3 = 0.0 if receipt is None else float(receipt.whole_bound_m3)
    rec = {"schema": RECEIPT_SCHEMA, "profile": PROFILE, "occurrence": {"id": oid, "sha256": osha},
           "parent": {"seq": int(parent.seq), "role": parent.role, "generation": int(root.generation), "t_s_hex": float(parent.t_s).hex()},
           "clocks": {"start_s_hex": start.hex(), "end_s_hex": end_s.hex(), "elapsed_s": end_s - start, "soil_clock_before_hex": t_old.hex(), "soil_clock_after_hex": float(new_state.t_s).hex()},
           "water": {"wind_water_physics": "none", "surface_leaves_bitwise_equal": True,
                     "zero_time_reconcile": {"performed": bool(n_changed), "n_columns": n_changed, "surface_return_m3": ret_m3, "bottom_export_m3": exp_m3, "whole_bound_m3": bnd_m3},
                     "soil_after_m3": soil_after, "lifecycle_residual_m3": float(resid), "lifecycle_bound_m3": float(bound), "role": role},
           "properties": {"n_dirty": int(dirty.size), "n_changed": n_changed, "n_depth_changed": n_depth, "ks_changed": bool(ks_changed), "root_replaced": bool(replaced), "new_generation": generation, "new_column_sha256": new_column.sha256,
                          "property_basis": prov["property_basis"]},
           "classes": {"n_classes": nc, "airborne_before_hex": hexv(cls["airborne_mass_before_by_class_kg"]), "deposited_hex": hexv(cls["deposited_mass_by_class_kg"]), "boundary_export_hex": hexv(cls["boundary_export_mass_by_class_kg"]),
                       "bed_change_hex": hexv(cls["bed_change_by_class_kg"])},
           "plan": {"scan_required_bytes": scan_need, "candidate_base_bytes": base_cand, "pool_bytes": int(meta["pool_bytes"]), "pool_slots": int(meta["slots"]), "n_cells": N, "n_layers": L, "batch_cells": B,
                    "candidate_required_bytes": 0 if model1 is None else int(model1["required_bytes"])},
           "limits": limits.as_record(),
           "scope": "native Dense NumPy wind handoff; no water physics during wind (the explicit zero-time capacity reconcile is separate), no E/T/hydrology, no reroute; weather cursor and routing tuple untouched"}
    text = _json(rec)
    if len(text.encode()) > limits.max_receipt_bytes:
        raise WindHandoffLimitError(f"the scalar receipt needs {len(text.encode())} B above max_receipt_bytes {limits.max_receipt_bytes}")
    parts = _Parts(new_column, new_op, new_pref, ks_out, depth_out, new_hist, new_state, cfg, generation, root.source_witness, bed, surface, parent.qx_m2_s, parent.qy_m2_s, parent.routing, end_s, led, prov, int(parent.seq) + 1, parent.cursor, role)
    expected = _integrity(parent, parts, oid, end_s, text)  # the post-stage integrity, bound to the CURRENT parent and every proposed field
    with _minting():
        cand = WindHandoffCandidate(oid, end_s, role, text, weakref.ref(parent), weakref.ref(lineage), {"parts": parts, "consumed": False, "accepted": None})
    _INTEGRITY[cand] = expected  # PRIVATE weak table, not the caller-reachable cell
    return cand


def adopt_wind_handoff(lineage: Any, parent: Any, candidate: Any, *, available_bytes_fn: Any = None, reserve_bytes: int = 0, before_publish: Any = None) -> lc.Boundary:
    """Adopt a staged candidate (module docstring). Order: genuine candidate, expected head, one-shot state, parent/lineage binding, immediate-duplicate id -> admission (actual host reader when omitted) -> `before_publish` (outside every mint window) -> head re-check and the POST-STAGE INTEGRITY audit
    -> preallocation of every bookkeeping slot -> mint -> weak accepted reference -> head install (first commit store) -> consumed/emptied/last-id stores (no allocation). Every refusal before the head store leaves the head and the candidate retryable."""
    if type(candidate) is not WindHandoffCandidate or candidate not in _CANDIDATES:
        raise WindHandoffError("not a genuine WindHandoffCandidate")
    lc._expect(lineage, parent, lc.Boundary)
    _refuse_vegetation(parent, "adopt_wind_handoff")
    before_publish = _callback(before_publish, "before_publish")
    if available_bytes_fn is not None and not callable(available_bytes_fn):
        raise WindHandoffError("available_bytes_fn must be callable or None")
    cell = candidate._cell
    if cell["consumed"] or cell["parts"] is None:
        raise WindHandoffError("this wind handoff was already adopted (a candidate is consumed exactly once)")
    if candidate._parent_ref() is not parent or candidate._lineage_ref() is not lineage:
        raise WindHandoffError("the candidate was staged for another parent boundary or lineage")
    oid = candidate.occurrence_id
    last = _LAST.get(lineage)
    if last is not None and last[0] == oid:
        raise WindHandoffError("this occurrence identity is the lineage's last adopted one (an immediate duplicate; the schedule cursor owns cross-event identity)")
    if _INTEGRITY.get(candidate) is None:
        raise WindHandoffError("the candidate has no recorded integrity (not staged by this module)")
    reserve = fsl._count(reserve_bytes, "reserve_bytes")
    nfs._admit({"required_bytes": _ADOPT_BYTES + reserve, "terms": {"adoption": _ADOPT_BYTES, "reserve": reserve}}, None, available_bytes_fn, "wind handoff adoption")
    if before_publish is not None:
        before_publish()  # OUTSIDE every mint window and BEFORE anything irreversible: the coordinator re-checks its immutable source/case here; a failure preserves the head and the unconsumed candidate
        lc._expect(lineage, parent, lc.Boundary)
        if cell["consumed"] or cell["parts"] is None:
            raise WindHandoffError("this wind handoff was already adopted (a candidate is consumed exactly once)")
    p = cell["parts"]
    if _integrity(parent, p, oid, candidate.end_s, candidate.receipt_json) != _INTEGRITY.get(candidate):
        raise WindHandoffError("the post-stage integrity audit failed: the child, the candidate's proposed state or the parent changed in place after staging (nothing was consumed or published)")
    # ---- preallocate EVERY bookkeeping slot before anything is consumed or moved
    if last is None:
        last = [None]
        _LAST[lineage] = last
    with lc._minting():  # nothing refusable runs inside the window
        root = lc.BoundaryRoot(p.column, p.matrix_op, p.pref_op, p.ks_m_s, p.depth_m, p.depth_m, p.history, p.water_state, p.config, p.generation, p.source_witness, p.bed)
        boundary = lc.Boundary(p.role, root, p.bed, p.surface_depth_m, p.qx_m2_s, p.qy_m2_s, p.routing, p.t_s, p.ledger, p.provenance, p.seq, p.cursor, None, None)
    ref = _weak(boundary)
    # ---- the commit: plain stores only (no allocation, no callback). The head moves FIRST: from then on the parent is no longer the head, so the candidate cannot be adopted twice whatever else is interrupted.
    cell["accepted"] = ref
    try:
        lineage._head = boundary
    finally:
        if lineage.head is boundary:
            cell["consumed"] = True
            cell["parts"] = None  # the candidate retains no state
            last[0] = oid
    return boundary
