"""Phase 2.5 vegetation LIFECYCLE (CPU: NumPy/Numba; Dense and published Windowed): the owned ZERO-TIME `apply_vegetation_generation` that gives an accepted lifecycle `Boundary` one CURRENT vegetation generation.

LEAF owns plants; this module does not compute vegetation. It consumes an S0 `water_vegetation_binding.VegetationBinding` (one authentic LEAF projection bound to explicit site controls, the expected LEAF package / projection / source,
the actual mask and the actual current surface) and turns it into the NATIVE operators of the boundary's own root:

    matrix g (R1 -> `reference_vegetation_function`, per cell)   -> a NEW `FloatingMatrixOperator` on the SAME prepared column, J/I rebound BIT FOR BIT by the accepted `rebind_floating_matrix_history`
    preferential contributions (one species at a time)           -> a NEW `FloatingPreferentialOperator` through the one canonical derivation (shared background, one aggregate rate, one cap)
    splash cover / live E-shading cover (compact (N,) arrays)    -> carried on ONE sealed `VegetationRecord`; the splash cover reaches the actual native `PondEvent` (factory + a lifecycle check) and the shading cover the daily E path

ZERO TIME. Nothing steps and no water, history, surface, face flux, routing reference, bed, clock, cursor, ledger, sequence, cold anchor or last-actual-event epoch changes: the new head shares the old head's current W, depth,
bed, surface, routing reference, ledger copy and provenance OBJECTS and replaces only the law (operators) and the vegetation record. There is no fictitious event, no property refresh and no property-generation bump; the new
vegetation generation counter / law identity are SEPARATE fields of the record. ONE current record, never a list. Everything refusable (genuine/forged head, eligibility, binding validity, source / catalog / session / package / controls
ordering, domain, mask, the AUTHORITATIVE current bed surface, memory admission, law preparation, history rebinding, the covers) runs BEFORE the head is replaced, LAST; a refusal leaves the original head retryable.

ELIGIBILITY: only the lineage's genuine COMPLETE head `Boundary` (nothing pending: no recession role, no pending storm / cursor storm / forcing match). An active `Occurrence`, a `DailyHold` or a mid-day state is refused. The SAME binding is
idempotent (the existing head is returned). The SAME biological generation with a new legitimate current surface / mask projection is accepted (surface-only update: no strict sequence increase is required); an OLDER session
sequence, another session / lineage / catalog / LEAF package / site controls, or an initial-state projection after a session generation is refused. CPU is run and tested here; a CUDA boundary is served by `vegetation_lifecycle_cuda` (AUTHORED, NOT RUN: no device
hardware qualification), after the current-device gate.

SURFACE. The binding's surface hash is compared with the AUTHORITATIVE current bed surface read through the existing bounded public MAPLE read (`committed_elevation_m(bed.committed_topography)`), never the routing held surface (which may
lag by design). No deep-bed clone or global refresh is made.

RECORD. The ONE current `VegetationRecord` BORROWS the genuine S0 `VegetationBinding` by reference (which borrows the immutable LEAF projection: derived CSR / areas / root and receiver tables / covers / plain source identities, never a PlantSession,
plant population, S x N or P x N array) and its two covers are immutable VIEWS of the projection's canonical arrays (no copies). It also keeps the binding / controls / LEAF package / external-helper / source / mask / surface / law identities as explicit
scalars. Every field is covered by a producer fingerprint checked O(1) on every use (`verify_record`; the dataclass is frozen but `object.__setattr__` forgery, replaced arrays, a replaced binding and a forged record all refuse); no content is re-hashed.

CARRY. A refreshed property generation keeps the held g (`floating_event_refresh` reads `fm.held_cell_vegetation_factor`) and the sealed preferential law (`rebind_preferential_column`, capacity read at the step): nothing is
re-streamed on an ordinary soil-property refresh; only a genuine new vegetation generation rebuilds. The record rides `BoundaryRoot` / `Occurrence` / `DailyHold`. `begin_occurrence` checks that the event's ACTUAL cover is the bound splash cover
(`event_factory_for` builds it); the bridge passes the bound shading cover (`cover=None`) and refuses a caller cover; the every-N frame `property_sha256` keeps its old meaning (the root column identity); the vegetation generation is exposed by `vegetation_attribution` for a NEW frame profile, and the old frame writer refuses a vegetation-bound state.

OLD PROFILES. The H0 held-context export and the physical lifecycle files do not carry the per-cell law or the covers: a vegetation-bound head is a typed refusal at `saveable_head` and `export_boundary_held_context` (no codec is started).

MEMORY. Declared host arithmetic admitted from metadata before every allocation (`estimate_apply_bytes`), counting the caller reserve ONCE and every simultaneously live term: the held (old) operators, the S0 law peak (with LEAF's public stream
estimate), the rebound history and the new operators; the record OWNS only a small allowance and BORROWS the binding / projection (reported separately). No cap is raised and no per-step projection exists.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import threading
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import water_vegetation_binding as wvb

__all__ = ["VegetationLifecycleError", "VegetationRecord", "apply_vegetation_generation", "bound_shading_cover", "bound_splash_cover", "current_record", "estimate_apply_bytes", "event_factory_for", "restore_vegetation_record", "vegetation_attribution",
           "verify_record"]

_F8 = 8
_MINT = threading.local()
_RECORDS: weakref.WeakSet = weakref.WeakSet()
RECORD_ALLOWANCE_BYTES = 8192


class VegetationLifecycleError(lc.LifecycleError):
    """Typed refusal of the vegetation lifecycle transition; the lineage head is unchanged and the call may be retried."""


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


@dataclass(frozen=True, eq=False)
class VegetationRecord:
    """The ONE current vegetation generation of a lineage head (module docstring). Minted ONLY by `apply_vegetation_generation`; verified by `verify_record` (producer fingerprint of EVERY field, O(1))."""

    generation: int  # vegetation generation counter of THIS lineage (1 = the first application); NOT increased by a property refresh
    recipe_id: str  # STABLE generation identity: sha256(binding sha, generation counter, parent recipe id) -- reproducible from the compact binding + counter, independent of any property column / operator
    binding: Any  # the genuine S0 VegetationBinding, BORROWED by reference (it borrows the immutable LEAF projection)
    binding_sha256: str
    controls_sha256: str
    projection_sha256: str
    leaf_package_sha256: str
    external_helpers_json: str  # the external helper pins (module, sha256, bytes) of the binding
    source_json: str  # the LEAF source identity (session token / sequence / pop / geometry hashes, or the labelled initial-state hash)
    mask_sha256: str
    surface_sha256: str  # the AUTHORITATIVE current bed surface the generation was bound to
    shape: tuple
    n_cells: int
    n_layers: int
    backend: str  # "numpy" | "numba" | "cuda"
    device_id: int | None
    # APPLY-TIME provenance ONLY (what was prepared when the generation was applied): NEVER the current law. After a property refresh the current column / operators differ; read them with `vegetation_attribution`.
    applied_column_sha256: str
    applied_law_sha256: str
    applied_g_sha256: str
    applied_matrix_op_sha256: str
    applied_preferential_sha256: str
    splash_cover: Any  # (N,) read-only VIEW of the projection's canonical total visible live + dead_standing cover (host immutable canonical)
    shade_cover: Any  # (N,) read-only VIEW of the projection's canonical total visible LIVE cover (host immutable canonical)
    splash_cover_2d: Any  # (ny, nx) read-only view of `splash_cover`
    splash_cover_device: Any  # CUDA only: the (ny, nx) device splash cover, prepared COLD ONCE at apply (the event's cover is compared with it resident); None on CPU
    shade_cover_device: Any  # CUDA only: the (N,) device E-shading cover, prepared COLD ONCE at apply and passed to every GPU day; None on CPU
    owned_bytes: int  # host bytes the record owns (allowance)
    borrowed_bytes: int  # the borrowed binding + LEAF projection bytes the record keeps alive
    device_bytes: int = 0  # device bytes the record owns (the two device covers)
    parent_recipe_id: str | None = None

    @property
    def bound_shade(self) -> Any:
        """The E-shading cover the day must use: the resident device array on a CUDA lineage, else the host canonical view."""
        return self.shade_cover_device if self.shade_cover_device is not None else self.shade_cover

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise VegetationLifecycleError("a VegetationRecord is minted only by apply_vegetation_generation")
        _RECORDS.add(self)

    def __copy__(self):
        raise VegetationLifecycleError("a VegetationRecord cannot be copied")

    def __deepcopy__(self, memo):
        raise VegetationLifecycleError("a VegetationRecord cannot be copied")

    def __reduce_ex__(self, protocol):
        raise VegetationLifecycleError("a VegetationRecord is not serialisable (no checkpoint profile carries it yet)")

    @property
    def source(self) -> dict:
        """A fresh detached copy of the recorded LEAF source identity."""
        return json.loads(self.source_json)

    @property
    def external_helpers(self) -> list:
        """A fresh detached copy of the recorded external helper pins."""
        return json.loads(self.external_helpers_json)


_FP: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _chain_ends_in_bytes(a: Any) -> bool:
    while isinstance(a, np.ndarray):
        if a.flags.writeable:
            return False
        a = a.base
    return isinstance(a, bytes)


def _fingerprint(rec: VegetationRecord) -> tuple:
    out = []
    for f in dataclasses.fields(rec):
        v = getattr(rec, f.name)
        if isinstance(v, np.ndarray):
            out.append((f.name, int(v.__array_interface__["data"][0]), tuple(v.shape), v.dtype.str, bool(v.flags.writeable)))
        elif hasattr(v, "__cuda_array_interface__"):  # a device array: pointer / shape / dtype (CuPy has no read-only flag)
            out.append((f.name, int(v.__cuda_array_interface__["data"][0]), tuple(v.shape), str(v.dtype)))
        elif f.name == "binding":
            out.append((f.name, id(v)))
        else:
            out.append((f.name, v))
    return tuple(out)


def verify_record(rec: Any) -> VegetationRecord:
    """Public genuine-factory proof of a `VegetationRecord`: exact type, producer registry, a fingerprint of EVERY field (a replaced field / array / binding, `object.__setattr__` forgery and a hand-built record all refuse), the borrowed binding's own
    public `verify_binding` and the covers still being the projection's canonical immutable arrays. O(1): nothing is re-hashed."""
    if type(rec) is not VegetationRecord or rec not in _RECORDS:
        raise VegetationLifecycleError("not a genuine VegetationRecord")
    fp = _FP.get(rec)
    if fp is None or _fingerprint(rec) != fp:
        raise VegetationLifecycleError("the VegetationRecord was altered after sealing (a field, array or binding was replaced)")
    try:
        b = wvb.verify_binding(rec.binding)
    except wvb.VegetationBindingError as exc:
        raise VegetationLifecycleError(f"the record's borrowed binding is not genuine: {exc}") from None
    if (b.sha256, b.controls.sha256, b.projection_sha256, b.leaf_package_sha256, b.external_helpers_json, b.source_json, b.mask_sha256, b.surface_sha256) != (
            rec.binding_sha256, rec.controls_sha256, rec.projection_sha256, rec.leaf_package_sha256, rec.external_helpers_json, rec.source_json, rec.mask_sha256, rec.surface_sha256):
        raise VegetationLifecycleError("the record's identities differ from its borrowed binding")
    return rec


def _record(rec: Any) -> VegetationRecord:
    return verify_record(rec)


def _current_law(obj: Any) -> tuple:
    """`(column, matrix_op, preferential_op, property_generation)` of the CURRENT genuine holder: the Boundary root / DailyHold / the occurrence vertical's canonical held template / a bridge daily state's hold. Never the apply-time record."""
    if type(obj) is lc.Boundary:
        r = obj.root
        return r.column, r.matrix_op, r.pref_op, int(r.generation)
    if type(obj) is lc.DailyHold:
        return obj.column, obj.matrix_op, obj.pref_op, int(obj.generation)
    if type(obj) is lc.Occurrence:
        t = obj.vertical.held_template()
        return t.column, t.matrix_op, t.preferential, int(t.generation)
    hold = getattr(getattr(obj, "ctx", None), "hold", None)
    if type(hold) is lc.DailyHold:
        return _current_law(hold)
    raise VegetationLifecycleError("obj must be a Boundary, DailyHold, Occurrence or a bridge daily state")


def vegetation_attribution(obj: Any) -> dict | None:
    """Attribution for a NEW frame/output profile of a `Boundary`, `DailyHold`, `Occurrence` or a bridge `FloatingDailyState` (the old frame `property_sha256` keeps its meaning). None without a binding.

    STABLE identity (`generation`, `recipe_id`, binding / controls / projection / package / source / helper / mask / surface ids) comes from the one record and does NOT change on a property refresh. CURRENT law identity
    (`current_column_sha256`, `current_matrix_op_sha256`, `current_preferential_sha256`, `current_property_generation`, `current_law_id`) is read from the actual genuine holder NOW. The apply-time ids are returned only
    under the explicit `applied` provenance key."""
    if type(obj) in (lc.Boundary, lc.DailyHold, lc.Occurrence):
        rec = current_record(obj)
    else:
        hold = getattr(getattr(obj, "ctx", None), "hold", None)
        rec = None if hold is None or getattr(hold, "vegetation", None) is None else _record(hold.vegetation)
    if rec is None:
        return None
    col, mop, pop, gen = _current_law(obj)
    cur = {"column_sha256": col.sha256, "matrix_op_sha256": mop.sha256, "preferential_sha256": None if pop is None else pop.sha256}
    current_law_id = hashlib.sha256(json.dumps({"tag": "maple_syrup.vegetation_lifecycle.current_law.v1", "recipe": rec.recipe_id, **cur}, sort_keys=True).encode()).hexdigest()
    return {"generation": rec.generation, "recipe_id": rec.recipe_id, "parent_recipe_id": rec.parent_recipe_id, "binding_sha256": rec.binding_sha256, "controls_sha256": rec.controls_sha256,
            "projection_sha256": rec.projection_sha256, "leaf_package_sha256": rec.leaf_package_sha256, "surface_sha256": rec.surface_sha256, "mask_sha256": rec.mask_sha256, "backend": rec.backend,
            "source": rec.source, "external_helpers": rec.external_helpers,
            "current_column_sha256": cur["column_sha256"], "current_matrix_op_sha256": cur["matrix_op_sha256"], "current_preferential_sha256": cur["preferential_sha256"], "current_property_generation": gen,
            "current_law_id": current_law_id,
            "applied": {"column_sha256": rec.applied_column_sha256, "law_sha256": rec.applied_law_sha256, "g_sha256": rec.applied_g_sha256, "matrix_op_sha256": rec.applied_matrix_op_sha256,
                        "preferential_sha256": rec.applied_preferential_sha256, "note": "apply-time provenance; NOT the current law"}}


def recipe_id_of(binding_sha256: str, generation: int, parent_recipe_id: str | None) -> str:
    """The stable generation identity from the compact binding identity + counter + parent recipe (no property column / operator involved)."""
    return hashlib.sha256(json.dumps({"tag": "maple_syrup.vegetation_lifecycle.recipe.v1", "binding": binding_sha256, "generation": int(generation), "parent": parent_recipe_id}, sort_keys=True).encode()).hexdigest()


def current_record(head: Any) -> VegetationRecord | None:
    """The current vegetation record of a lineage head (`Boundary`, `DailyHold` or `Occurrence`); None when no vegetation generation is applied."""
    if type(head) is lc.Boundary:
        rec = head.root.vegetation
    elif type(head) in (lc.DailyHold, lc.Occurrence):
        rec = head.vegetation
    else:
        raise VegetationLifecycleError("head must be a Boundary, DailyHold or Occurrence")
    return None if rec is None else _record(rec)


def bound_splash_cover(head: Any) -> np.ndarray:
    """The bound splash cover `(ny, nx)` (read-only) of a head with an applied vegetation generation: pass it as `vegetation_cover_fraction` when building the next native `PondEvent`."""
    rec = current_record(head)
    if rec is None:
        raise VegetationLifecycleError("the head has no applied vegetation generation")
    return rec.splash_cover_2d


def bound_shading_cover(head: Any) -> np.ndarray:
    """The bound live E-shading cover `(N,)` (read-only); the bridge passes it itself when `cover=None`."""
    rec = current_record(head)
    if rec is None:
        raise VegetationLifecycleError("the head has no applied vegetation generation")
    return rec.shade_cover


def event_factory_for(head: Any, build: Any) -> Any:
    """Wrap the caller's static event builder so the native `PondEvent` is CONSTRUCTED with the head's bound splash cover (hashed into its native config identity at construction; the event is never mutated afterwards).
    `build(vertical, topology, vegetation_cover_fraction) -> PondEvent` must pass the cover to the `PondEvent` constructor; the returned factory has the lifecycle's `(vertical, topology)` signature. The cover array captured is the sealed
    read-only record array only (the head, bed and water are not retained)."""
    cover = bound_splash_cover(head)
    if not callable(build):
        raise VegetationLifecycleError("build must be callable")

    def factory(vertical: Any, topology: Any = None) -> Any:
        return build(vertical, topology, cover)

    return factory


# ------------------------------------------------------------------------------------------------------------------------------ memory
def stream_numbers(binding: Any) -> tuple:
    """`(per_next_held, per_next_free, weights_stage)` bytes from LEAF's PUBLIC `projection.stream_estimate(previous_held=)` (live origin; the weights stage is the SEPARATE stage the corrected S0 model takes). A projection
    without these public numbers is a typed refusal (no private arithmetic, no padding). Zero when no species is enabled."""
    if not binding.enabled_species:
        return 0, 0, 0
    try:
        held, free = binding.projection.stream_estimate(previous_held=True), binding.projection.stream_estimate(previous_held=False)
        return int(held["per_next_bytes"]), int(free["per_next_bytes"]), int(held["weights_stage_bytes"])
    except (AttributeError, KeyError, TypeError) as exc:
        raise VegetationLifecycleError(f"the LEAF projection has no public stream_estimate() with per_next_bytes / weights_stage_bytes: {exc!r}") from None


def estimate_apply_bytes(n_layers: int, n_cells: int, n_species_enabled: int, *, binding_resident_bytes: int, borrowed_bytes: int = 0, old_retained_bytes: int, stream_per_next_held_bytes: int = 0,
                         stream_per_next_free_bytes: int = 0, stream_weights_stage_bytes: int = 0, backend: str = "numpy", k_suction_cell_arrays: int = 0, a3_cell_map: bool = False, max_rate_cell_map: bool = False) -> dict:
    """Declared host bytes of ONE `apply_vegetation_generation`, stage by stage from metadata (the maximum stage is `peak_bytes`, never a padded sum). Each stage lists every simultaneously live term; the caller reserve is counted once by the caller's
    admission. `binding_resident_bytes` is the binding's OWN bytes, `borrowed_bytes` the borrowed LEAF projection it keeps alive (both live throughout; the record adds only its small allowance and its covers are views, not copies):
      law      the held (old) operators + the S0 law preparation peak (`wvb.estimate_law_bytes`, with LEAF's public stream estimate);
      rebind   old operators + the NEW matrix and preferential operators + the binding + the native history rebinding (`fm` 'rebind');
      seal     old + new operators + the rebound history (2 N doubles) + the binding + the record allowance."""
    L, N, S = (wvb._count(v, k, mn) for v, k, mn in ((n_layers, "n_layers", 1), (n_cells, "n_cells", 1), (n_species_enabled, "n_species_enabled", 0)))
    old, br, bo = wvb._count(old_retained_bytes, "old_retained_bytes"), wvb._count(binding_resident_bytes, "binding_resident_bytes"), wvb._count(borrowed_bytes, "borrowed_bytes")
    law = wvb.estimate_law_bytes(L, N, S, binding_resident_bytes=br + bo, stream_per_next_held_bytes=stream_per_next_held_bytes, stream_per_next_free_bytes=stream_per_next_free_bytes,
                                 stream_weights_stage_bytes=stream_weights_stage_bytes, backend=backend,
                                 k_suction_cell_arrays=k_suction_cell_arrays, a3_cell_map=a3_cell_map, max_rate_cell_map=max_rate_cell_map)
    mat = fm.estimate_floating_matrix_memory(L, N, backend=backend, operation="prepare")
    pref = fpi.estimate_floating_preferential_memory(L, N, backend=backend, operation="prepare", n_species=S)
    reb = fm.estimate_floating_matrix_memory(L, N, backend=backend, operation="rebind")
    new = int(mat.retained_bytes) + int(pref.retained_bytes)
    stages = {"law": old + int(law["peak_bytes"]), "rebind": old + new + br + bo + int(reb.required_bytes), "seal": old + new + 2 * _F8 * N + br + bo + RECORD_ALLOWANCE_BYTES}
    return {"stages": stages, "peak_bytes": int(max(stages.values())), "new_operator_bytes": new, "owned_record_bytes": RECORD_ALLOWANCE_BYTES, "borrowed_bytes": int(br + bo),
            "scope": "declared host arithmetic of this call; the caller's arrays, the interpreter and JIT caches are excluded; the borrowed binding/projection is counted as live, not as new"}


def _retained(op: Any) -> int:
    return 0 if op is None else int(op.retained_bytes)


# ------------------------------------------------------------------------------------------------------------------------------ ordering of biological generations
def _check_generation(cur: VegetationRecord | None, binding: Any) -> None:
    """Reject foreign / older / conflicting generations against the ONE current record (no history list). Raises VegetationLifecycleError."""
    if cur is None:
        return
    src, cs = binding.source, cur.source
    if src.get("catalog_hash") != cs.get("catalog_hash"):
        raise VegetationLifecycleError("the binding's species catalog differs from the lineage's current vegetation generation (foreign catalog)")
    if binding.leaf_package_sha256 != cur.leaf_package_sha256:
        raise VegetationLifecycleError("the binding's LEAF package differs from the lineage's current vegetation generation (a package change is not a legitimate update of one lineage)")
    if binding.external_helpers_json != cur.external_helpers_json:
        raise VegetationLifecycleError("the binding's external helper pins differ from the lineage's current vegetation generation")
    if binding.controls.sha256 != cur.controls_sha256:
        raise VegetationLifecycleError("the site controls differ from the lineage's current vegetation generation (they are fixed for one lineage in this slice)")
    if (binding.n_cells, binding.n_layers, tuple(binding.shape)) != (cur.n_cells, cur.n_layers, tuple(cur.shape)):
        raise VegetationLifecycleError("the binding's domain / layer count differs from the lineage's current vegetation generation")
    if cs["kind"] == "plant_session":
        if src["kind"] != "plant_session":
            raise VegetationLifecycleError("an initial-state projection cannot follow an accepted session generation")
        if src["session_id"] != cs["session_id"] or src["lineage"] != cs["lineage"]:
            raise VegetationLifecycleError("the binding belongs to another PlantSession / lineage (foreign session)")
        if int(src["sequence"]) < int(cs["sequence"]):
            raise VegetationLifecycleError(f"the binding is an OLDER biological generation (session sequence {src['sequence']} < the current {cs['sequence']}): stale")
        if int(src["sequence"]) == int(cs["sequence"]) and any(src[k] != cs[k] for k in ("token", "pop_sha256", "geometry_sha256")):
            raise VegetationLifecycleError("the binding has the current session sequence but a different token / population / geometry (conflicting biology)")
    else:
        if src["kind"] == "initial_state" and src["state_sha256"] != cs["state_sha256"]:
            raise VegetationLifecycleError("a different initial-state projection has no ordering proof against the current one (refused)")


# ------------------------------------------------------------------------------------------------------------------------------ the transition
def _require_live_leaf(binding: Any) -> None:
    """A LIVE application requires the genuine actual `leaf.water_vegetation.VegetationProjection` in `binding.projection` (lazy actual import, exact type): a cached derived input with the same `VegetationBinding` class can never bypass the
    live biology-generation proof. Refused BEFORE any water, history or head work; no fake class and no getattr fallback."""
    try:
        lw = wvb._load_leaf()
    except wvb.VegetationBindingError as exc:
        raise VegetationLifecycleError(f"a live vegetation application needs the actual LEAF package: {exc}") from None
    try:
        proj = binding.projection
    except AttributeError:  # an uninitialised / forged record
        raise VegetationLifecycleError("the binding record is uninitialised (no projection): not a genuine sealed record") from None
    if type(proj) is not lw.VegetationProjection:
        raise VegetationLifecycleError(f"a live application requires a genuine leaf.water_vegetation.VegetationProjection in binding.projection, got {type(proj).__module__}.{type(proj).__name__}")


def apply_vegetation_generation(lineage: Any, expected: Any, binding: Any, *, memory_budget_bytes: Any, reserve_bytes: Any = 0, available_bytes_fn: Any = None, host_memory_budget_bytes: Any = None,
                                host_reserve_bytes: Any = 0, host_available_bytes_fn: Any = None) -> Any:
    """Apply `binding` to the lineage's head `Boundary` `expected` (module docstring). Returns the new head Boundary (or `expected` itself for the same binding). The head is replaced LAST; any refusal leaves it untouched.
    CPU: `memory_budget_bytes` / `reserve_bytes` / `available_bytes_fn`. CUDA: the CURRENT device is gated FIRST and TWO pools are mandatory: device (`memory_budget_bytes`, `reserve_bytes`, `available_bytes_fn`) and host
    (`host_memory_budget_bytes`, `host_reserve_bytes`, `host_available_bytes_fn`); a CPU boundary refuses the host-pool arguments.
    PUBLIC ERROR POLICY: every typed refusal -- own checks, S0 `VegetationBindingError` / `VegetationBindingMemoryError`, and the native admissions' `FloatingSoilError` / `FloatingMemoryError` -- surfaces as `VegetationLifecycleError`
    with the head untouched; only these named families are wrapped (no broad `Exception`, and a real `MemoryError` from an allocator is never caught or hidden)."""
    from maple.core.types.topographic_commit import committed_elevation_m

    lc._expect(lineage, expected, lc.Boundary)  # a wrong / stale / duplicate / copied / forged expected head (or a non-Boundary head) is refused here
    root = expected.root
    backend = root.column.backend
    if backend not in ("numpy", "numba", "cuda"):
        raise VegetationLifecycleError(f"the vegetation transition serves the numpy / numba / cuda boundaries, got {backend!r}")
    cuda = backend == "cuda"
    cp = q = None
    if cuda:
        from maple_syrup import floating_event_lifecycle_cuda as flc

        try:
            cp, _dev = flc._gate(root.column)  # the CURRENT device FIRST: before any reader, hash, array or transfer
        except lc.LifecycleError as exc:
            raise VegetationLifecycleError(str(exc)) from None
        if host_memory_budget_bytes is None or host_available_bytes_fn is not None and not callable(host_available_bytes_fn):
            raise VegetationLifecycleError("a CUDA boundary requires the explicit host pool: host_memory_budget_bytes (and a callable host_available_bytes_fn or None) besides the device pool; there is no default")
        try:
            q = flc.quota_of(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
        except lc.LifecycleError as exc:
            raise VegetationLifecycleError(str(exc)) from None
    elif host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None:
        raise VegetationLifecycleError("host_* arguments apply only to a CUDA boundary (a meaningless CPU argument is refused)")
    if not expected.eligible_for_interevent or expected.pending_forcing is not None or expected.storm_match is not None or (expected.cursor is not None and expected.cursor.pending is not None):
        raise VegetationLifecycleError("only a genuine COMPLETE boundary with nothing pending (no recession, pending storm, cursor storm or forcing match) accepts a vegetation generation")
    if type(binding) is not wvb.VegetationBinding:
        raise VegetationLifecycleError(f"binding must be a water_vegetation_binding.VegetationBinding, got {type(binding).__name__}")
    if available_bytes_fn is not None and not callable(available_bytes_fn):
        raise VegetationLifecycleError("available_bytes_fn must be callable or None")
    _require_live_leaf(binding)
    if binding.origin != wvb.ORIGIN_LIVE:  # a cached-derived binding is a restored recipe, never a new live generation
        raise VegetationLifecycleError(f"a live application requires a {wvb.ORIGIN_LIVE!r} binding, got {binding.origin!r}")
    try:
        wvb.verify_binding(binding)  # the public genuine-factory proof (registry + every field + the borrowed projection)
    except wvb.VegetationBindingError as exc:
        raise VegetationLifecycleError(f"the binding is not a genuine sealed record: {exc}") from None
    column = root.column
    cur = None if root.vegetation is None else verify_record(root.vegetation)
    L, N = int(column.n_layers), int(column.n_cells)
    if (binding.n_layers, binding.n_cells) != (L, N) or tuple(binding.shape) != tuple(int(v) for v in expected.surface_depth_m.shape):
        raise VegetationLifecycleError(f"the binding's domain ({binding.n_layers} layers, {binding.n_cells} cells, {tuple(binding.shape)}) is not the boundary's ({L}, {N}, {tuple(expected.surface_depth_m.shape)})")
    if float(column.cell_area_m2) != float(binding.cell_area_m2):  # the LEAF cover fractions are fractions of THAT cell area: verified exactly BEFORE any hash or allocation
        raise VegetationLifecycleError(f"the binding's cell_area_m2 {binding.cell_area_m2!r} is not the boundary column's {column.cell_area_m2!r}")
    # ---- the actual mask (canonical host topology mask only: no hidden N copy) and the AUTHORITATIVE current bed surface (never the routing held surface)
    active = expected.routing[0].active
    if type(active) is not np.ndarray or active.dtype != np.bool_ or active.shape != tuple(binding.shape) or not active.flags.c_contiguous:
        raise VegetationLifecycleError("the boundary's routing mask is not a C-contiguous bool (ny, nx) array (refused; no hidden conversion)")
    hm = hashlib.sha256()
    wvb._hash_chunked(hm, active.reshape(-1).view(np.uint8))  # zero-copy chunked view (the same recipe as the binding's mask hash)
    if hm.hexdigest() != binding.mask_sha256 or (not cuda and column.mask_sha256 != binding.mask_sha256):  # a CUDA column's own identity is a device stream hash: S0 verifies it with the sealed host mask, never by comparing it with LEAF's byte hash here
        raise VegetationLifecycleError("the binding's active mask is not the boundary's actual mask (stale or mismatched)")
    surface = committed_elevation_m(expected.bed.committed_topography)
    h = hashlib.sha256()
    if isinstance(surface, np.ndarray):
        if surface.dtype != np.float64 or surface.shape != tuple(binding.shape) or not surface.flags.c_contiguous:
            raise VegetationLifecycleError("the bed's committed elevation is not a C-contiguous float64 (ny, nx) array (refused; no hidden conversion)")
        wvb._hash_chunked(h, surface)
    elif cuda and hasattr(surface, "__cuda_array_interface__"):
        from maple_syrup import vegetation_lifecycle_cuda as vlc

        vlc.hash_device_surface(cp, h, surface, tuple(binding.shape), q, int(binding.resident_bytes + binding.borrowed_bytes))  # bounded COLD device-to-host chunk stream (counted), never a full-N host array
    else:
        raise VegetationLifecycleError(f"the bed's committed elevation has the unsupported type {type(surface).__name__} for a {backend} boundary")
    if h.hexdigest() != binding.surface_sha256:
        raise VegetationLifecycleError("the binding was not built from the boundary bed's CURRENT committed surface (stale surface)")
    _check_generation(cur, binding)
    if cur is not None and cur.binding_sha256 == binding.sha256:
        return expected  # idempotent: the same exact binding is the current generation
    mb, rb = wvb._count(memory_budget_bytes, "memory_budget_bytes"), wvb._count(reserve_bytes, "reserve_bytes")
    if cuda:
        from maple_syrup import vegetation_lifecycle_cuda as vlc

        return vlc.finish_cuda(lineage, expected, binding, cur, cp, q, mb, rb)
    # ---- composed admission of EVERY stage from metadata BEFORE any allocation (caller reserve counted once per stage)
    cfg = root.config
    kc = int(cfg.matrix_conductivity_mode == "texture_ks")
    held, free, wstage = stream_numbers(binding)
    est = estimate_apply_bytes(L, N, len(binding.enabled_species), binding_resident_bytes=binding.resident_bytes, borrowed_bytes=binding.borrowed_bytes, old_retained_bytes=_retained(root.matrix_op) + _retained(root.pref_op),
                               stream_per_next_held_bytes=held, stream_per_next_free_bytes=free, stream_weights_stage_bytes=wstage, backend=column.backend, k_suction_cell_arrays=kc, a3_cell_map=binding.controls.a3_layout == "cell",
                               max_rate_cell_map=binding.controls.max_rate_layout == "cell")
    for stage, need in est["stages"].items():
        try:
            wvb._admit(f"apply_vegetation_generation/{stage}", need, mb, rb, available_bytes_fn)
        except wvb.VegetationBindingError as exc:
            raise VegetationLifecycleError(str(exc)) from None
    cond = root.ks_m_s if cfg.matrix_conductivity_mode == "texture_ks" else cfg.matrix_conductivity_m_s
    old_ret = _retained(root.matrix_op) + _retained(root.pref_op)
    try:
        law = wvb.prepare_vegetation_law(column, binding, conductivity_m_s=cond, suction_m=cfg.matrix_suction_m, memory_budget_bytes=mb, reserve_bytes=rb + old_ret, available_bytes_fn=available_bytes_fn)
        wvb.verify_prepared_law(law)
        if law.binding_sha256 != binding.sha256 or law.column_sha256 != column.sha256:
            raise VegetationLifecycleError("the prepared law is not this binding's / this column's")
        reb = fm.rebind_floating_matrix_history(root.matrix_op, root.history, law.matrix_op, memory_budget_bytes=mb, reserve_bytes=rb + old_ret + int(law.matrix_op.retained_bytes) + _retained(law.preferential_op),
                                                available_bytes_fn=available_bytes_fn)
    except (wvb.VegetationBindingError, fm.fsl.FloatingSoilError, fm.fsl.FloatingMemoryError) as exc:  # public error policy: every typed refusal (including the native `FloatingMemoryError`) is a VegetationLifecycleError; a real MemoryError is never caught
        raise VegetationLifecycleError(f"the vegetation law could not be prepared / the history rebound: {exc}") from None
    old_h, new_h = root.history, reb.history
    if (not np.array_equal(np.asarray(old_h.wetting_m), np.asarray(new_h.wetting_m)) or not np.array_equal(np.asarray(old_h.cumulative_intake_m), np.asarray(new_h.cumulative_intake_m))
            or new_h.operator_sha256 != law.matrix_op.sha256 or reb.reset):
        raise VegetationLifecycleError("the persistent J/I were not rebound bit for bit (refused)")
    return mint_and_install(lineage, expected, binding, cur, law, new_h, None, None, 0)


def host_covers(binding: Any, n_cells: int) -> tuple:
    """The immutable canonical host covers (views of the borrowed projection; their base chain must end in immutable bytes)."""
    proj = binding.projection
    if binding.origin == wvb.ORIGIN_CACHED:  # typed dispatch: a cached-derived (restored) recipe exposes its compact covers through the cache's public view, never a LEAF attribute
        from maple_syrup import water_vegetation_held as wvh

        arrays = wvh.cached_compact(proj)["arrays"]
        splash, shade = arrays["splash_cover"], arrays["shading_cover"]
    else:
        splash, shade = proj.splash_cover, proj.shading_cover
    if not (_chain_ends_in_bytes(splash) and _chain_ends_in_bytes(shade)) or splash.shape != (n_cells,) or shade.shape != (n_cells,):
        raise VegetationLifecycleError("the projection covers are not immutable (N,) arrays (refused)")
    return splash, shade


def mint_and_install(lineage: Any, expected: Any, binding: Any, cur: Any, law: Any, new_history: Any, splash_dev: Any, shade_dev: Any, device_bytes: int) -> Any:
    """Mint the ONE record and the new head from validated parts and replace the head LAST (shared by the CPU and CUDA transitions). Nothing refusable remains except the final head check."""
    root, column = expected.root, expected.root.column
    N, L = int(column.n_cells), int(column.n_layers)
    splash, shade = host_covers(binding, N)
    gen = 1 if cur is None else cur.generation + 1
    rid = recipe_id_of(binding.sha256, gen, None if cur is None else cur.recipe_id)
    with _minting():
        rec = VegetationRecord(gen, rid, binding, binding.sha256, binding.controls.sha256, binding.projection_sha256, binding.leaf_package_sha256, binding.external_helpers_json, binding.source_json, binding.mask_sha256,
                               binding.surface_sha256, tuple(binding.shape), N, L, column.backend, column.device_id, column.sha256, law.law_sha256, law.g_sha256, law.matrix_sha256, law.preferential_sha256, splash, shade,
                               splash.reshape(tuple(binding.shape)), splash_dev, shade_dev, RECORD_ALLOWANCE_BYTES, int(binding.resident_bytes + binding.borrowed_bytes), int(device_bytes),
                               None if cur is None else cur.recipe_id)
    _FP[rec] = _fingerprint(rec)
    with lc._minting():
        new_root = lc.BoundaryRoot(column, law.matrix_op, law.preferential_op, root.ks_m_s, root.depth0_m, root.depth_m, new_history, root.water_state, root.config, root.generation, root.source_witness, expected.bed,
                                   root.host_active, root.edges_host, root.host_depth, vegetation=rec)
        head = lc.Boundary(expected.role, new_root, expected.bed, expected.surface_depth_m, expected.qx_m2_s, expected.qy_m2_s, expected.routing, expected.t_s, dict(expected.ledger), expected.provenance, expected.seq,
                           expected.cursor, expected.pending_forcing, expected.storm_match)
    if lineage.head is not expected:
        raise VegetationLifecycleError("the lineage head changed during the transition (refused)")
    lineage._head = head  # the head replacement is LAST
    return head


# ------------------------------------------------------------------------------------------------------------------------------ restore (file profile)
_APPLIED_KEYS = frozenset({"column_sha256", "law_sha256", "g_sha256", "matrix_op_sha256", "preferential_sha256"})


def _hex64(v: Any, what: str) -> str:
    if type(v) is not str or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
        raise VegetationLifecycleError(f"{what} must be a 64-hex lowercase digest")
    return v


def restore_vegetation_record(token: Any, *, generation: Any, recipe_id: Any, parent_recipe_id: Any, applied: Any, max_bytes: Any, reserve_bytes: Any = 0, available_bytes_fn: Any = None,
                              cuda_quota: Any = None) -> VegetationRecord:
    """THE owning factory of the vegetation record of a RESTORED lineage head (physical checkpoint `maple-syrup-floating-lifecycle-checkpoint-vegetation/1`). It needs the genuine one-shot `VerifiedRestore` token whose event runs the
    held-context vertical that ACTUALLY holds the cached-derived vegetation recipe (`/3` | `/4` context, `HeldVegetation`): the sealed cached record, the law identities and the CURRENT operators / column are those of that context
    (never from the file). The stable generation counter / parent recipe / apply-time provenance are the file's HISTORICAL record: `recipe_id` must reproduce `recipe_id_of(binding, generation, parent)` exactly. The binding is the
    LABELLED cached-derived `VegetationBinding` (`origin == cached_derived`): restoring is NOT a new live biology generation, a later live `apply_vegetation_generation` still needs the actual LEAF producer proof and ordering.
    The record is minted here, nowhere else; the head is minted afterwards by the lifecycle's own `stage_restored_head`, which re-verifies this record against the same token.
    CUDA (explicit, AUTHORED NOT RUN): the SAME checks on a `/4` context; `cuda_quota` (the caller's strict two-pool quota) is REQUIRED, the current device is gated FIRST, the two device covers are uploaded COLD ONCE (admitted from
    metadata) and the event's ACTUAL device cover is compared with the device splash cover resident (one counted scalar read). A CPU restore refuses `cuda_quota`."""
    from maple_syrup import floating_held_context as fhc
    from maple_syrup import water_vegetation_held as wvh

    try:
        lc.require_usable_token(token)  # genuine AND not yet consumed: BEFORE any binding, array or device upload
    except lc.LifecycleError as exc:
        raise VegetationLifecycleError(f"the restore token is not a genuine, unconsumed verified-restore capability: {exc}") from None
    vert = getattr(token.event, "vertical", None)
    held = getattr(vert, "_held", None)
    try:
        fhc.require_held_context(held)
    except fhc.HeldContextError as exc:
        raise VegetationLifecycleError(f"the token's vertical does not hold a genuine held context: {exc}") from None
    hv = held.vegetation
    if hv is None:
        raise VegetationLifecycleError("the token's held context carries no vegetation recipe (an old /1 | /2 context cannot restore a vegetation generation)")
    if type(generation) is not int or generation < 1:
        raise VegetationLifecycleError("generation must be a strict positive int")
    if parent_recipe_id is None:
        if generation != 1:
            raise VegetationLifecycleError("only generation 1 has no parent recipe")
    else:
        _hex64(parent_recipe_id, "parent_recipe_id")
        if generation < 2:
            raise VegetationLifecycleError("generation 1 has no parent recipe")
    rid = _hex64(recipe_id, "recipe_id")
    if type(applied) is not dict or set(applied) != _APPLIED_KEYS:
        raise VegetationLifecycleError("the apply-time provenance must be exactly the closed key set")
    for k in _APPLIED_KEYS:
        _hex64(applied[k], f"applied.{k}")
    if rid != recipe_id_of(hv.binding_sha256, generation, parent_recipe_id):
        raise VegetationLifecycleError("the recorded recipe id is not reproduced by the held recipe's binding, the generation counter and the parent recipe")
    column = held.column
    N, L = int(column.n_cells), int(column.n_layers)
    cuda = column.backend == "cuda"
    if not cuda and cuda_quota is not None:
        raise VegetationLifecycleError("cuda_quota applies only to a CUDA restore (a meaningless CPU argument is refused)")
    cp = None
    if cuda:
        if type(cuda_quota) is not dict:
            raise VegetationLifecycleError("a CUDA restore requires the caller's strict two-pool quota (cuda_quota)")
        from maple_syrup import floating_event_lifecycle_cuda as flc

        try:
            cp, _dev = flc._gate(column)  # the CURRENT device FIRST, before any upload
        except lc.LifecycleError as exc:
            raise VegetationLifecycleError(str(exc)) from None
    try:
        wvh.verify_cached_inputs(hv.cached)
        binding = wvh.cached_binding(hv.cached, expected_binding_sha256=hv.binding_sha256, max_bytes=max_bytes, reserve_bytes=reserve_bytes, available_bytes_fn=available_bytes_fn)
        wvb.verify_binding(binding)
    except wvb.VegetationBindingError as exc:
        raise VegetationLifecycleError(f"the held cached recipe cannot be bound: {exc}") from None
    if (binding.n_cells, binding.n_layers) != (N, L) or float(binding.cell_area_m2) != float(column.cell_area_m2) or (not cuda and column.mask_sha256 != binding.mask_sha256):
        raise VegetationLifecycleError("the held cached recipe's domain / cell area / mask differ from the held column")
    splash, shade = host_covers(binding, N)
    splash_dev = shade_dev = None
    dev_bytes = 0
    if cuda:
        from maple_syrup import floating_soil_layers as fsl
        from maple_syrup import floating_soil_layers_cuda as fcu
        from maple_syrup import native_floating_soil_cuda as ncuda

        need = int(splash.nbytes + shade.nbytes)
        try:
            fcu.admit(cp, column.device_id, need, int(RECORD_ALLOWANCE_BYTES), cuda_quota, "restore_vegetation_record (CUDA covers)")
        except (fsl.FloatingSoilError, fsl.FloatingMemoryError) as exc:
            raise VegetationLifecycleError(str(exc)) from None
        splash_dev = ncuda._up(cp, np.ascontiguousarray(splash).reshape(tuple(binding.shape)))  # cold ONCE: resident covers for every GPU event check and day
        shade_dev = ncuda._up(cp, np.ascontiguousarray(shade))
        dev_bytes = int(splash_dev.nbytes + shade_dev.nbytes)
        if not flc.arrays_equal_cuda(column, token.event.veg, splash_dev):
            raise VegetationLifecycleError("the restored event does not run the held recipe's bound splash cover (the static factory must build the event with the cover it is handed)")
    elif not np.array_equal(np.asarray(getattr(token.event, "veg", None)), np.asarray(splash.reshape(tuple(binding.shape)))):
        raise VegetationLifecycleError("the restored event does not run the held recipe's bound splash cover (the static factory must build the event with the cover it is handed)")
    with _minting():
        rec = VegetationRecord(generation, rid, binding, binding.sha256, binding.controls.sha256, binding.projection_sha256, binding.leaf_package_sha256, binding.external_helpers_json, binding.source_json,
                               binding.mask_sha256, binding.surface_sha256, tuple(binding.shape), N, L, column.backend, column.device_id, applied["column_sha256"], applied["law_sha256"], applied["g_sha256"],
                               applied["matrix_op_sha256"], applied["preferential_sha256"], splash, shade, splash.reshape(tuple(binding.shape)), splash_dev, shade_dev, RECORD_ALLOWANCE_BYTES,
                               int(binding.resident_bytes + binding.borrowed_bytes), dev_bytes, parent_recipe_id)
    _FP[rec] = _fingerprint(rec)
    return rec
