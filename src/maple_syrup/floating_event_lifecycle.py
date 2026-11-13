"""EB3 (CPU: NumPy/Numba; Dense and published Windowed): the single-authority floating EVENT LIFECYCLE -- open -> native occurrence -> genuine completion -> selected property refresh -> accepted boundary -> next native occurrence.

ARCHITECTURE (one water authority, one current bed, one accepted head)
* `open_floating_event(owner, bed, event_factory, ...)` builds the vertical, the caller's static `PondEvent` (factory called OUTSIDE every mint window) and the actual `PondEvent.initial_state` (all callbacks, memory admission and
  validation) while the original `NativeFloatingSoil` is still CURRENT. Only then, LAST: the owner is retired (`nfs.retire_native_owner_to_lineage`) and the vertical detaches it (`FloatingPondVertical.detach_initial`), so no stale
  `W0` stays actionable or referenced by the runtime during the storm; the registered current initial payload is the ONE water. A failed open leaves the owner current and retryable.
* `FloatingEventLineage` holds ONE accepted head (an `Occurrence` or a `Boundary`; immutable, minted here). A functional trial (`event.step`/`evolve`/`settle_retaining_water`) never touches it; only `open`, `close_occurrence` and
  `begin_occurrence` replace it, LAST, after every refusable step. A wrong / stale / duplicate / copied / forged expected head is refused; the EB1 completion proof alone is not lineage authority.
* `close_occurrence(lineage, occurrence, cap, state, ...)`: the expected head must be this occurrence; the EB2 staged selected refresh (CURRENT payload W/J/I/D, gross-ledger dirty columns, accepted zero-time cascade, Ks-only no cascade,
  J/I bit exact) is staged; the event's full audit closes; the boundary is built from CURRENT references only (bed, law, W, H', face fluxes, routing reference, scalar ledgers: never the candidate, capability, event, owner or old
  parent bed -- weak-lifetime tested); the completion capability is consumed LAST, then the head. A capacity SURFACE RETURN is added to H' once (the bed's native surface depth is updated by reference-sharing replace) and makes the
  boundary role `pending_recession`: NOT eligible for wind or a day until a real rain-0 native recession occurrence (the continuation) completes again; a previously COMPLETE retained standing H is an ordinary eligible boundary.
* `begin_occurrence(lineage, boundary, event_factory, ...)`: a vertical from the boundary root (`FloatingPondVertical.from_boundary`: the boundary's current inventory is the initial payload; the bed handed to the event must be the
  boundary's own bed) and `PondEvent.continued_state`, which CARRIES the last actual routing reference (topology, held source, datum): nothing is rebuilt from the current terrain, so the net-displacement gate keeps measuring from the
  last genuine routing build. Surface depth, face fluxes, W, J/I, D carry without loss; event-local ledgers restart truthfully; lifetime transfer totals live in the boundary.
* DURABLE HELD LAW ROOT (strategy): a refreshed property generation is a genuine PUBLIC prepared root (`_build_column` on the CURRENT canonical D, `_fresh_op`, canonical `rebind_preferential_column` identity) whose identities the H0
  export/reconstruction reproduces (`export_boundary_held_context`: a CPU vertical from the boundary root, then the public `export_held_context`). Cold anchor and execution epoch stay SEPARATE: `source_witness` is always the original cold
  anchor (never an epoch); the boundary's `provenance` carries `property_basis="boundary_selected"`, an `execution_epoch/v1` chain and `cold_content_hash=None` (an epoch is not a content hash). A future general file codec versions
  its record; here only the water-free held facts are proven.
The event<->daily bridge (`floating_event_daily_bridge`) extends the same head with `DailyHold` (daily hold / hand-back, wet-day runtime without any Pond state) and binds a pending daily storm to its forcing and to the producer's forcing-history proof.
No physics, kernels, source pins, historical codecs or defaults change; CUDA is a typed refusal before any transfer. Memory: references to the current arrays; the EB2 admission governs the refresh; no NVK clone or bed scan per event.
"""
from __future__ import annotations

import contextlib
import hashlib
import threading
import weakref
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from maple_syrup import floating_completion as fcm
from maple_syrup import floating_event_refresh as fer
from maple_syrup import floating_pond as fp
from maple_syrup import native_floating_soil as nfs
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = ["Boundary", "BoundaryRoot", "FloatingEventLineage", "LifecycleError", "Occurrence", "begin_occurrence", "close_occurrence", "export_boundary_held_context", "open_floating_event"]


class LifecycleError(RuntimeError):
    """Typed refusal of the event lifecycle."""


_MINT = threading.local()
_ROOTS: weakref.WeakSet = weakref.WeakSet()
_RECORDS: weakref.WeakSet = weakref.WeakSet()


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


class _Minted:
    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise LifecycleError(f"a {type(self).__name__} is minted only by the event lifecycle (no direct construction)")
        _RECORDS.add(self)

    def __copy__(self):
        raise LifecycleError(f"a {type(self).__name__} cannot be copied")

    def __deepcopy__(self, memo):
        raise LifecycleError(f"a {type(self).__name__} cannot be copied")

    def __reduce_ex__(self, protocol):
        raise LifecycleError(f"a {type(self).__name__} is not serialisable")


@dataclass(frozen=True, eq=False)
class BoundaryRoot(_Minted):
    """The refreshed property generation's prepared law root and CURRENT payload (references; minted by `close_occurrence`)."""

    column: Any
    matrix_op: Any
    pref_op: Any
    ks_m_s: Any
    depth0_m: Any  # the canonical depth the root was prepared at (= the current D)
    depth_m: Any
    history: Any
    water_state: Any
    config: Any
    generation: int
    source_witness: str  # the ORIGINAL cold anchor, never an epoch
    bed: Any
    host_active: Any = None  # CUDA only (sealed host O(N) mask / O(L) edges / the candidate's host canonical-depth record, by reference from the genuine producer); None on CPU
    edges_host: Any = None
    host_depth: Any = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _ROOTS.add(self)


@dataclass(frozen=True, eq=False)
class Boundary(_Minted):
    """An accepted lifecycle boundary: CURRENT references only (no candidate / capability / event / owner / old bed)."""

    role: str  # "boundary" | "pending_recession"
    root: BoundaryRoot
    bed: Any
    surface_depth_m: Any
    qx_m2_s: Any
    qy_m2_s: Any
    routing: tuple  # (topology, held_source_m, datum): the LAST actual routing reference
    t_s: float
    ledger: Any  # lifetime scalars: dict
    provenance: Any
    seq: int
    cursor: Any = None  # the event<->daily bridge's `recharge.DailyCursor` (a pending storm between a day and its genuine completion); None outside the bridge
    pending_forcing: Any = None  # the FULL native `BoundForcing` of the pending storm (the bridge's begin), until its genuine completion is matched
    storm_match: Any = None  # digest of the pending forcing once a MATCHED full native storm genuinely completed (the only proof `day_after_storm` accepts)

    @property
    def eligible_for_interevent(self) -> bool:
        """Wind / a day may start only from a COMPLETE boundary: a pending recession (a capacity return woke the flow) is not eligible until a real native recession completes again."""
        return self.role == "boundary"


@dataclass(frozen=True, eq=False)
class Occurrence(_Minted):
    """A live native occurrence (the accepted head while a storm / recession runs)."""

    event: Any
    vertical: Any
    t_s: float
    ledger: Any
    provenance: Any
    seq: int
    role: str = "occurrence"
    cursor: Any = None
    pending_forcing: Any = None
    storm_match: Any = None


class FloatingEventLineage:
    """ONE accepted head. Only `open_floating_event`, `close_occurrence` and `begin_occurrence` replace it, LAST."""

    __slots__ = ("__weakref__", "_head")

    def __init__(self) -> None:
        self._head: Any = None

    @property
    def head(self) -> Any:
        return self._head


def _record(obj: Any, kind: type) -> None:
    if type(obj) is not kind or obj not in _RECORDS:
        raise LifecycleError(f"not a genuine {kind.__name__}")


def _expect(lineage: Any, expected: Any, kind: type) -> None:
    if type(lineage) is not FloatingEventLineage:
        raise LifecycleError("lineage must be a FloatingEventLineage")
    _record(expected, kind)
    if lineage._head is not expected:
        raise LifecycleError("the expected parent is not the lineage's accepted head (stale, duplicate or another branch)")


def _admit_native(event: Any, vert: Any) -> None:
    """The factory's return value must be the ACTUAL native `PondEvent` running THIS vertical (no stand-in, subclass or duck type): a public callback return value never bypasses the genuine native initializer."""
    from maple_syrup.pond_event import PondEvent

    if type(event) is not PondEvent or event.vertical is not vert:
        raise LifecycleError("the factory must return the actual native PondEvent running the lifecycle's floating vertical (stand-ins and duck types are refused)")


def _admit_state(event: Any, vert: Any, state: Any, bed: Any, t_s: float) -> None:
    """The state must be the ACTUAL `PondEventState` the native initializer produced for THIS event (producer-registered), on THIS bed and clock, with its soil water registered as the vertical's current payload."""
    from maple_syrup.pond_event import PondEventState

    if (type(state) is not PondEventState or state not in event._blessed or state.config_sha256 != event.config_sha256 or state.forcing_sha256 != event.forcing_sha256
            or state.bed is not bed or float(state.t_s) != float(t_s)):
        raise LifecycleError("the initial state is not the native initializer's state for this event/bed/clock")
    try:
        vert.current_payload(state.hydraulic.soil_water_m)
    except fp.FloatingPondError as exc:
        raise LifecycleError(f"the state's soil water is not the vertical's registered current payload: {exc}") from None


def _storage_m3(state: Any) -> float:
    area = float(state.topology.dx_m) ** 2
    return (float(np.asarray(state.hydraulic.soil_water_m).sum()) + float(np.asarray(state.hydraulic.depth_m).sum())) * area


def _cpu_only_quota(**kw: Any) -> None:
    if any(v is not None and v != 0 for v in kw.values()):
        raise LifecycleError("memory_budget_bytes / host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to the CUDA lifecycle (a meaningless CPU argument is refused)")


def open_floating_event(owner: Any, bed: Any, event_factory: Any, *, available_bytes_fn: Any, preferential: Any = None, reserve_bytes: int = 0, batch_cells: int = nfs.DEFAULT_BATCH_CELLS,
                        memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Any = None) -> tuple:
    """Open the lifecycle from a genuine CURRENT owner and the actual bed (module docstring). Returns `(lineage, occurrence, state)`. Everything refusable precedes the owner's retirement (LAST)."""
    if type(owner) is not nfs.NativeFloatingSoil or not owner.is_current:
        raise LifecycleError("owner must be a genuine CURRENT NativeFloatingSoil")
    if owner.column.backend not in ("numpy", "numba", "cuda"):
        raise LifecycleError("the event lifecycle serves the numpy/numba/cuda owners")
    if not callable(event_factory) or not callable(available_bytes_fn):
        raise LifecycleError("event_factory and available_bytes_fn must be callable")
    if owner.column.backend == "cuda":
        from maple_syrup import floating_event_lifecycle_cuda as flc

        return flc.open_cuda(owner, bed, event_factory, available_bytes_fn=available_bytes_fn, preferential=preferential, reserve_bytes=reserve_bytes, batch_cells=batch_cells, host_available_bytes_fn=host_available_bytes_fn,
                             host_reserve_bytes=host_reserve_bytes, memory_budget_bytes=memory_budget_bytes, host_memory_budget_bytes=host_memory_budget_bytes)
    _cpu_only_quota(memory_budget_bytes=memory_budget_bytes, host_memory_budget_bytes=host_memory_budget_bytes, host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn)
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=available_bytes_fn, preferential=preferential, reserve_bytes=reserve_bytes, batch_cells=batch_cells)
    event = event_factory(vert)  # the caller's static factory, outside every mint window
    _admit_native(event, vert)
    t0 = float(owner.state.t_s)
    state = event.initial_state(bed, vert.initial_soil_water(), t_s=t0)  # the actual full validation, owner still current
    _admit_state(event, vert, state, bed, t0)
    # ---- every refusable / allocating step is staged BEFORE the owner is touched
    storage0 = _storage_m3(state)
    ledger = {"storage0_m3": storage0, "rain_m3": 0.0, "drainage_m3": 0.0, "outlet_export_m3": 0.0, "bottom_export_m3": 0.0, "surface_return_m3": 0.0, "bound_m3": 0.0, "n_refreshes": 0, "n_occurrences": 1}
    lineage = FloatingEventLineage()
    vert.detach_initial()  # on the PRIVATE new vertical (a failure discards it; the owner is untouched)
    prov = {"cold_anchor_witness": vert.held_template().source_witness, "property_basis": "initial_full", "execution_epoch": None, "cold_content_hash": None}
    with _minting():
        occ = Occurrence(event, vert, t0, ledger, prov, 0)
    if not owner.is_current:
        raise LifecycleError("the owner was superseded during the open")
    nfs.retire_native_owner_to_lineage(owner)  # the ONLY owner mutation, LAST: its preconditions were verified just above, nothing after it can raise
    lineage._head = occ
    return lineage, occ, state


def _pending_storm_proof(occurrence: Any, event: Any, state: Any, audit: dict) -> Any:
    """Backend-neutral proof of a pending daily storm at a genuine completion (the C3 CPU body, factored unchanged so the CUDA close runs the SAME proof): the producer's forcing-history conformity for ANY pending cursor (full storm AND
    rain-0 recertification), and for a bound full storm the native end, the unit rain scale, the declared amount and the end clock. Returns the `storm_match` digest to record (the occurrence's own when nothing is pending)."""
    if occurrence.cursor is not None and occurrence.cursor.pending is not None:
        # a pending daily storm (its full native storm OR a rain-0 recertification after it) needs the PRODUCER's proof of the forcing HISTORY: every accepted step of this occurrence's chain integrated exactly the native
        # schedule's rate on a piece-confined interval. A total alone is no proof; an unproven chain (restored from a file) or a step that did not conform (permanent) refuses BEFORE the head/consume.
        proof = event.forcing_conformity(state)
        if proof is not True:
            raise LifecycleError("the pending storm's occurrence has no producer proof of its forcing history: " + ("a step integrated a rate other than the native schedule's (a manual or off-piece step)" if proof is False
                                                                                                                      else "the state's step chain is unproven (restored or foreign): no proof is granted from a total alone"))
    storm_match = occurrence.storm_match
    if occurrence.pending_forcing is not None:  # a matched FULL native storm: its native rain total must be what the forcing integrates (completion is never inferred from totals alone: the genuine completion, the event binding and the end clock are checked too)
        f = occurrence.pending_forcing
        _check_storm_binding(event, f, state.topology.dx_m)
        if float(state.t_s) < float(f.end_s):
            raise LifecycleError("the pending full storm has not reached its native rainfall end")
        area = float(state.topology.dx_m) ** 2
        n_active = int(np.count_nonzero(np.asarray(occurrence.vertical._active_host())))
        declared_m3 = float(f.amount_mm) * 1e-3 * area * n_active
        if abs(float(audit["water"]["rain_m3"]) - declared_m3) > float(audit["water"]["global_bound_m3"]) + float(f.amount_bound_mm) * 1e-3 * area * n_active:
            raise LifecycleError("the completed event's native rain total is not the pending full forcing's (refused)")
        storm_match = f.digest
    return storm_match


def close_occurrence(lineage: Any, occurrence: Any, cap: Any, state: Any, *, available_bytes_fn: Any, batch_cells: int = nfs.DEFAULT_BATCH_CELLS, reserve_bytes: int = 0, memory_budget_bytes: Any = None,
                     host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Any = None) -> Boundary:
    """Close the head occurrence at a genuinely COMPLETE state: stage the EB2 refresh, audit, build the boundary from CURRENT references, consume the completion LAST, replace the head LAST."""
    _expect(lineage, occurrence, Occurrence)
    event = occurrence.event
    if occurrence.vertical.backend == "cuda":
        from maple_syrup import floating_event_lifecycle_cuda as flc

        return flc.close_cuda(lineage, occurrence, cap, state, available_bytes_fn=available_bytes_fn, batch_cells=batch_cells, reserve_bytes=reserve_bytes, host_available_bytes_fn=host_available_bytes_fn,
                              host_reserve_bytes=host_reserve_bytes, memory_budget_bytes=memory_budget_bytes, host_memory_budget_bytes=host_memory_budget_bytes)
    _cpu_only_quota(memory_budget_bytes=memory_budget_bytes, host_memory_budget_bytes=host_memory_budget_bytes, host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn)
    fcm.verify_completion(cap, state, event=event)
    cand = fer.stage_completed_event_refresh(cap, state, event=event, available_bytes_fn=available_bytes_fn, batch_cells=batch_cells, reserve_bytes=reserve_bytes)
    audit = event.audit(state)
    if not (audit["water"]["closes"] and audit["sediment"]["closes"]):
        raise LifecycleError("the completed event's water/sediment audits do not close; the boundary is refused")
    storm_match = _pending_storm_proof(occurrence, event, state, audit)
    held = occurrence.vertical.held_template()
    ny, nx = event.shape
    ret = np.asarray(cand.surface_return_m).reshape(ny, nx)
    h_prime = np.array(state.hydraulic.depth_m, copy=True) + ret if float(ret.sum()) != 0.0 else state.hydraulic.depth_m
    bed = state.bed if h_prime is state.hydraulic.depth_m else replace(state.bed, water=replace(state.bed.water, depth_m=h_prime))
    area = float(nfs._area(cand.child_bed))
    led = dict(occurrence.ledger)
    w = audit["water"]
    led["rain_m3"] += float(w["rain_m3"])
    led["drainage_m3"] += float(w["drainage_m3"])
    led["outlet_export_m3"] += float(w["export_m3"])
    led["bottom_export_m3"] += float(cand.receipt.bottom_export_m3)  # refresh bottom loss, once (native drainage is the separate `drainage_m3`)
    led["surface_return_m3"] += float(cand.receipt.surface_return_m3)  # INTERNAL transfer (soil -> surface): never a net loss
    led["bound_m3"] += float(w["global_bound_m3"]) + float(cand.receipt.whole_bound_m3)
    led["n_refreshes"] += 1
    stored = float(cand.receipt.whole_after_m3) + float(np.asarray(h_prime).sum()) * float(state.topology.dx_m) ** 2
    expected = led["storage0_m3"] + led["rain_m3"] - led["drainage_m3"] - led["outlet_export_m3"] - led["bottom_export_m3"] - led.get("evaporation_m3", 0.0) - led.get("transpiration_m3", 0.0)
    resid = stored - expected
    bound = led["bound_m3"] + volume_roundoff_bound_m3(12, max(abs(stored), abs(expected), led["rain_m3"], led["drainage_m3"], led["outlet_export_m3"], led["bottom_export_m3"], led.get("evaporation_m3", 0.0), led.get("transpiration_m3", 0.0), 1e-30))
    if not (np.isfinite(resid) and abs(resid) <= bound):
        raise LifecycleError(f"the cumulative water balance across the lifecycle does not close: residual {resid!r} m3 exceeds the MAPLE volume bound {bound!r} m3; the boundary is refused")
    led["residual_m3"], led["residual_bound_m3"] = float(resid), float(bound)
    prev_epoch = occurrence.provenance.get("execution_epoch") or ""
    epoch = hashlib.sha256(f"{fer.EPOCH_KIND}|{prev_epoch}|{cand.provenance['digest']}".encode()).hexdigest()
    prov = {"cold_anchor_witness": held.source_witness, "property_basis": fer.PROPERTY_BASIS, "execution_epoch": epoch, "cold_content_hash": None, "area_m2": area}
    role = "pending_recession" if float(ret.sum()) > 0.0 else "boundary"
    with _minting():
        root = BoundaryRoot(cand.column, cand.matrix_op, cand.pref_op, cand.ks_m_s, cand.depth_m, cand.depth_m, cand.history, cand.water_state, occurrence.vertical.config, int(held.generation) + 1, held.source_witness, bed)
        boundary = Boundary(role, root, bed, h_prime, state.hydraulic.qx_m2_s, state.hydraulic.qy_m2_s, (state.topology, state.held_source_m, state.datum), float(state.t_s), led, prov, occurrence.seq + 1, occurrence.cursor,
                            None if storm_match is not None else occurrence.pending_forcing, storm_match)
    fcm.consume_completion(cap, state, event=event)  # the completion is consumed LAST ...
    lineage._head = boundary  # ... and then the head
    return boundary


def begin_occurrence(lineage: Any, boundary: Any, event_factory: Any, *, available_bytes_fn: Any, reserve_bytes: int = 0, batch_cells: int = nfs.DEFAULT_BATCH_CELLS, memory_budget_bytes: Any = None,
                     host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Any = None) -> tuple:
    """Start the NEXT native occurrence from the head boundary (a pending recession is the legitimate continuation). Returns `(occurrence, state)`; the head is replaced LAST."""
    _expect(lineage, boundary, Boundary)
    if not callable(event_factory) or not callable(available_bytes_fn):
        raise LifecycleError("event_factory and available_bytes_fn must be callable")
    vert = fp.FloatingPondVertical.from_boundary(boundary.root, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, batch_cells=batch_cells, memory_budget_bytes=memory_budget_bytes,
                                                 host_memory_budget_bytes=host_memory_budget_bytes, host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn)
    event = event_factory(vert, boundary.routing[0])  # the factory builds the event on the CARRIED routing topology
    _admit_native(event, vert)
    pend = None if boundary.cursor is None else boundary.cursor.pending
    if pend is not None and boundary.storm_match != pend.forcing_digest:
        # a pending daily storm that is not yet matched MUST run its FULL native forcing: absolute edges/intensity/rain_scale/clock bound BEFORE any head mutation
        f = boundary.pending_forcing
        if f is None or f.digest != pend.forcing_digest:
            raise LifecycleError("the boundary carries no forcing matching its pending storm")
        _check_storm_binding(event, f, boundary.routing[0].dx_m)
        if float(boundary.t_s) != float(f.t0_s):
            raise LifecycleError("the pending full storm starts at its bound physical start: the boundary clock differs")
    elif boundary.role == "pending_recession" or pend is not None:
        _require_zero_future_rain(event.schedule, boundary.t_s)  # a recession / a recertification AFTER a matched storm continues with ZERO native rain
    state = event.continued_state(boundary.bed, vert.initial_soil_water(), routing=boundary.routing, depth_m=boundary.surface_depth_m, qx_m2_s=boundary.qx_m2_s, qy_m2_s=boundary.qy_m2_s, t_s=boundary.t_s)
    _admit_state(event, vert, state, boundary.bed, boundary.t_s)
    led = dict(boundary.ledger)
    led["n_occurrences"] += 1
    with _minting():
        occ = Occurrence(event, vert, boundary.t_s, led, boundary.provenance, boundary.seq + 1, role="occurrence", cursor=boundary.cursor,
                         pending_forcing=boundary.pending_forcing if (pend is not None and boundary.storm_match != pend.forcing_digest) else None, storm_match=boundary.storm_match)
    vert.release_boundary_initial()  # the vertical forgets the boundary's bed and initial W/J/I (the registered current payload is the ONE water)
    lineage._head = occ
    return occ, state


def _check_storm_binding(event: Any, f: Any, dx_m: float) -> None:
    """The event's ACTUAL native forcing must be the bound full storm: the absolute rainfall edges `[t0, end]` and the single piece's intensity (exactly the plan's), no other positive rain, and a uniform unit `rain_scale` over
    the active cells (the contract under which `P` is the depth each cell receives)."""
    sch = event.schedule
    edges, inten = np.asarray(sch.edges_s), np.asarray(sch.intensity_mm_per_h)
    if not (edges.shape == (2,) and inten.shape == (1,) and float(edges[0]) == float(f.t0_s) and float(edges[1]) == float(f.end_s) and float(inten[0]) == float(f.intensity_mm_h)):
        raise LifecycleError("the event's native rainfall schedule is not the bound full storm (absolute start/end/intensity)")
    if event.vertical is not None and event.vertical.backend == "cuda":  # resident check on the ACTUAL device map: ONE counted scalar read, no full-N map download
        from maple_syrup import floating_event_lifecycle_cuda as flc

        ok = flc.unit_rain_scale_cuda(event)
    else:
        scale = np.asarray(event.rain_scale)
        act = np.asarray(event._initial_topology.active)
        ok = bool(np.all(scale[act] == 1.0))
    if not ok:
        raise LifecycleError("the event's rain_scale is not the unit contract of the bound storm")


def _require_zero_future_rain(schedule: Any, t_s: float) -> None:
    """A pending recession continues as a real native RAIN-0 recession: every positive-intensity piece of the event's native forcing must already have ended at the boundary clock (zero native forcing over the remaining interval).
    Refused BEFORE any head mutation; no invented intensity threshold."""
    edges, inten = np.asarray(schedule.edges_s), np.asarray(schedule.intensity_mm_per_h)
    if np.any((inten > 0.0) & (edges[1:] > float(t_s))):
        raise LifecycleError("a pending recession continues with ZERO native rainfall over the remaining interval; the event's forcing is positive after the boundary clock (refused before any head change)")


def export_boundary_held_context(boundary: Any, pref_spec: Any = None, *, available_bytes_fn: Any, reserve_bytes: int = 0, memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None,
                                 host_reserve_bytes: Any = 0, host_available_bytes_fn: Any = None) -> Any:
    """Water-free held facts of the refreshed property generation through the PUBLIC H0 export (a temporary CPU vertical from the boundary root; nothing is adopted or consumed). Reconstruction reproduces the canonical new
    root / operator / preferential identities; the cold anchor stays the original source witness."""
    from maple_syrup import floating_held_context as fhc

    _record(boundary, Boundary)
    vert = fp.FloatingPondVertical.from_boundary(boundary.root, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, memory_budget_bytes=memory_budget_bytes,
                                                 host_memory_budget_bytes=host_memory_budget_bytes, host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn)
    if vert.backend == "cuda":
        from maple_syrup import floating_held_context_cuda as fhcc

        return fhcc.export_held_context_cuda(vert, pref_spec, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, host_available_bytes_fn=host_available_bytes_fn, host_reserve_bytes=host_reserve_bytes)
    return fhc.export_held_context(vert, pref_spec, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes)


# ----------------------------------------------------------------------------------------------------------------------------------------------
# event <-> daily bridge hooks (the SAME single lineage head: Occurrence -> Boundary -> DailyHold -> Boundary ...)
# ----------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class DailyHold(_Minted):
    """The lineage head WHILE a day runs: the boundary's STATIC pieces (law root, bed, surface, faces, routing reference, ledger, provenance, cursor) WITHOUT its old W/J/I (those live only in the daily state)."""

    column: Any
    matrix_op: Any
    pref_op: Any
    ks_m_s: Any
    depth_m: Any
    config: Any
    generation: int
    source_witness: str
    bed: Any
    surface_depth_m: Any
    qx_m2_s: Any
    qy_m2_s: Any
    routing: tuple
    ledger: Any
    provenance: Any
    seq: int
    t_s: float
    cursor: Any
    pending_forcing: Any = None
    storm_match: Any = None
    role: str = "daily_hold"
    host_active: Any = None  # CUDA only: the boundary root's sealed host mask / edges / host canonical-depth record, carried by reference so a GPU boundary after the day stays valid (None on CPU)
    edges_host: Any = None
    host_depth: Any = None


_HOLD_BINDING: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # hold -> (weakref(parent boundary), weakref(lineage)): WEAK metadata, no strong retention of the boundary/root/W


def stage_daily_hold(lineage: Any, boundary: Any) -> DailyHold:
    """Mint the hold of the lineage's ELIGIBLE head boundary (not a pending recession), bound to the EXACT parent boundary and THIS lineage (weakly); changes no head and no state."""
    _expect(lineage, boundary, Boundary)
    if not boundary.eligible_for_interevent:
        raise LifecycleError("a pending recession boundary is not eligible for an interevent day until a real native recession completes again")
    r = boundary.root
    with _minting():
        hold = DailyHold(r.column, r.matrix_op, r.pref_op, r.ks_m_s, r.depth_m, r.config, int(r.generation), r.source_witness, boundary.bed, boundary.surface_depth_m, boundary.qx_m2_s, boundary.qy_m2_s,
                         boundary.routing, dict(boundary.ledger), boundary.provenance, boundary.seq + 1, boundary.t_s, boundary.cursor, boundary.pending_forcing, boundary.storm_match,
                         host_active=r.host_active, edges_host=r.edges_host, host_depth=r.host_depth)
    _HOLD_BINDING[hold] = (weakref.ref(boundary), weakref.ref(lineage))
    return hold


def install_daily_hold(lineage: Any, boundary: Any, hold: Any) -> None:
    """Replace the head (boundary -> hold), LAST. The hold must be the one staged for THIS boundary on THIS lineage (a foreign, same-geometry or stale hold is refused BEFORE the head moves)."""
    _expect(lineage, boundary, Boundary)
    _record(hold, DailyHold)
    bound = _HOLD_BINDING.get(hold)
    if bound is None or bound[0]() is not boundary or bound[1]() is not lineage:
        raise LifecycleError("the hold was not staged for this boundary on this lineage (foreign, stale or copied)")
    lineage._head = hold


def rebase_boundary_from_day(lineage: Any, hold: Any, closed: Any, *, cursor: Any = None, pending_forcing: Any = None, cuda_quota: Any = None) -> Boundary:
    """Hand the CLOSED daily state back to the event lifecycle: the head must be this hold and `closed` a genuinely minted closed `FloatingDailyState` of THIS hold (`ctx.hold is hold`, role boundary). The lifetime ledger adds
    the day segment's rain / E / T / drainage (the lifecycle identity must still close within the accumulated MAPLE bounds, else refused BEFORE the head moves); the new Boundary shares the hold's static pieces and carries the
    closed day's W/J/I and cursor. Nothing of the old initial water or the previous boundary's root is retained."""
    from maple_syrup import floating_daily as fd
    from maple_syrup import floating_soil_layers as fsl

    _expect(lineage, hold, DailyHold)
    if type(closed) is not fd.FloatingDailyState or closed not in fd._MINTED or closed.role != "boundary" or closed.ctx.hold is not hold:
        raise LifecycleError("the closed daily state is not a genuine closed boundary of this day's hold")
    d = closed.ledger
    led = dict(hold.ledger)
    for key, val in (("rain_m3", d.rain_m3), ("evaporation_m3", d.evaporation_m3), ("transpiration_m3", d.transpiration_m3), ("drainage_m3", d.drainage_m3)):
        led[key] = led.get(key, 0.0) + float(val)
    led["outlet_export_m3"] += float(d.outlet_m3)  # a wet day's surface-router export
    led["bound_m3"] += float(d.bound_m3)
    led["n_days"] = led.get("n_days", 0) + int(d.days_applied)
    area = float(hold.column.cell_area_m2)
    surf = closed.surface  # a wet day carries its CURRENT surface state (depth, faces); a dry day leaves the hold's (zero) surface
    depth = hold.surface_depth_m if surf is None else surf[0]
    qx, qy = (hold.qx_m2_s, hold.qy_m2_s) if surf is None else (surf[1], surf[2])
    bed = hold.bed if surf is None else replace(hold.bed, water=replace(hold.bed.water, depth_m=depth))
    if hold.column.backend == "cuda":  # resident storage packet with the caller's two-pool quota + ONE counted scalar read of the surface sum (never np on a device array)
        from maple_syrup import floating_event_lifecycle_cuda as flc

        stored = flc.stored_m3_cuda(hold.column, closed.water, depth, area, cuda_quota)
    else:
        if cuda_quota is not None:
            raise LifecycleError("cuda_quota applies only to a CUDA hold (a meaningless CPU argument is refused)")
        stored = float(fsl.floating_storage_m3(hold.column, closed.water)) + float(np.asarray(depth).sum()) * area
    expected = (led["storage0_m3"] + led["rain_m3"] - led["drainage_m3"] - led["outlet_export_m3"] - led["bottom_export_m3"] - led["evaporation_m3"] - led["transpiration_m3"])
    resid = stored - expected
    bound = led["bound_m3"] + volume_roundoff_bound_m3(16, max(abs(stored), abs(expected), led["rain_m3"], led["drainage_m3"], led["outlet_export_m3"], led["bottom_export_m3"], led["evaporation_m3"],
                                                              led["transpiration_m3"], 1e-30))
    if not (np.isfinite(resid) and abs(resid) <= bound):
        raise LifecycleError(f"the cumulative water balance across event and day does not close: residual {resid!r} m3 exceeds the MAPLE volume bound {bound!r} m3; the hand-back is refused")
    led["residual_m3"], led["residual_bound_m3"] = float(resid), float(bound)
    with _minting():
        root = BoundaryRoot(hold.column, hold.matrix_op, hold.pref_op, hold.ks_m_s, hold.depth_m, hold.depth_m, closed.history, closed.water, hold.config, hold.generation, hold.source_witness, bed, hold.host_active, hold.edges_host, hold.host_depth)
        boundary = Boundary("boundary", root, bed, depth, qx, qy, hold.routing, float(closed.water.t_s), led, hold.provenance, hold.seq + 1,
                            closed.cursor if cursor is None else cursor, pending_forcing, hold.storm_match if (closed.cursor if cursor is None else cursor).pending is not None else None)
    lineage._head = boundary
    return boundary


def day_runtime(hold: Any, state: Any, event_factory: Any, *, available_bytes_fn: Any, reserve_bytes: int = 0, batch_cells: int = nfs.DEFAULT_BATCH_CELLS, memory_budget_bytes: Any = None,
                host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Any = None) -> tuple:
    """Native runtime of a WET day: a CPU vertical from the hold's law root and the day state's CURRENT W/J/I, the caller's static event (on the CARRIED routing topology) and the actual `PondEvent.continued_state`
    (full validation; no sediment is run). Returns `(vertical, event, hydraulic_state, solver)` where `solver` is the genuine floating Pond solver (accepted surface router) built by the vertical on the carried topology; `hydraulic_state` is a plain `HydraulicState`. The vertical forgets its initial
    payload; nothing is adopted and no head changes."""
    from maple_syrup import floating_daily as fd

    _record(hold, DailyHold)
    if type(state) is not fd.FloatingDailyState or state not in fd._MINTED or state.ctx.hold is not hold or state.surface is None:
        raise LifecycleError("the daily state is not a genuine wet-surface state of this hold")
    depth, qx, qy = state.surface
    if hold.column.backend == "cuda":  # resident comparison on the device (one counted scalar read), no full-N download
        from maple_syrup import floating_event_lifecycle_cuda as flc

        same = hold.bed.water.depth_m is depth or flc.arrays_equal_cuda(hold.column, hold.bed.water.depth_m, depth)
    else:
        same = np.array_equal(np.asarray(hold.bed.water.depth_m), np.asarray(depth))
    bed = hold.bed if same else replace(hold.bed, water=replace(hold.bed.water, depth_m=depth))
    with _minting():
        root = BoundaryRoot(hold.column, hold.matrix_op, hold.pref_op, hold.ks_m_s, hold.depth_m, hold.depth_m, state.history, state.water, hold.config, hold.generation, hold.source_witness, bed, hold.host_active, hold.edges_host, hold.host_depth)
    vert = fp.FloatingPondVertical.from_boundary(root, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, batch_cells=batch_cells, memory_budget_bytes=memory_budget_bytes,
                                                 host_memory_budget_bytes=host_memory_budget_bytes, host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn)  # CUDA: the full two-pool quota; a CPU hold refuses CUDA-only arguments
    event = event_factory(vert, hold.routing[0])  # the static case: control + context ONLY (no PondEventState, no sediment initialisation, no bed inventory)
    _admit_native(event, vert)
    t = float(state.water.t_s)
    pond = vert.build_hydraulics(hold.routing[0], control=event.control.hydraulic)  # binds the shape; the actual floating solver + accepted surface router
    from maple_syrup import floating_evaporation as fe

    # the COMPOSED wet-fragment admission BEFORE any fragment/array: the vertical's own stage/exchange/surface step model (preferential, matrix, FSL, router), the evaporation staging and the carried surface/face arrays
    N = int(vert.n_cells)
    what = "the wet-interevent day (composed fragment stages, evaporation staging and the carried surface/face arrays)"
    if vert.backend == "cuda":  # explicit TWO-POOL terms (the CPU line below is the unchanged historical arithmetic): the evaporation staging's device compute and its host packet are DIFFERENT pools
        from maple_syrup import floating_event_lifecycle_cuda as flc
        from maple_syrup import floating_pond_cuda as fpc_

        d_need, h_need = flc.wet_runtime_plan_cuda(int(vert.step_bytes), int(vert.host_op_bytes), N, int(qx.size) + int(qy.size))
        fpc_.admit(vert, d_need, what, h_need)  # current device first; reserves and the 64 KiB host allowance counted ONCE by `fcu.admit`
    else:
        vert._read_and_admit(int(vert.step_bytes) + int(fe.estimate_floating_evaporation_memory(N, backend=vert.backend).host_bytes) + 6 * N * 8 + (int(qx.size) + int(qy.size)) * 8, what)
    vert.check_initial(bed, event.context, t)  # the boundary's own bed (reference identity) and clock
    h0 = pond.initial_state(depth, vert.initial_soil_water(), t_s=t)  # the actual solver initializer: registers the CURRENT payload (the ONE water)
    xp = vert.xp  # numpy on the CPU verticals, cupy on a CUDA one (device-to-device copies of the carried faces)
    hyd = replace(h0, qx_m2_s=xp.array(qx, copy=True), qy_m2_s=xp.array(qy, copy=True))  # the carried faces, exactly
    pond.solver.validate_state(hyd)
    vert.release_boundary_initial()
    return vert, event, hyd, pond.solver


# ----------------------------------------------------------------------------------------------------------------------------------------------
# physical-file hooks (`floating_lifecycle_checkpoint`): save-side authority checks and the verified one-shot restore of the head
# ----------------------------------------------------------------------------------------------------------------------------------------------
_LEDGER_BASE = ("storage0_m3", "rain_m3", "drainage_m3", "outlet_export_m3", "bottom_export_m3", "surface_return_m3", "bound_m3", "n_refreshes", "n_occurrences")
_LEDGER_OPTIONAL = ("evaporation_m3", "transpiration_m3", "n_days", "residual_m3", "residual_bound_m3")
_LEDGER_COUNTS = ("n_refreshes", "n_occurrences", "n_days")
_PROVENANCE_KEYS = frozenset({"cold_anchor_witness", "property_basis", "execution_epoch", "cold_content_hash", "area_m2"})
_RESTORED_VERTICALS: weakref.WeakSet = weakref.WeakSet()  # a restored vertical opens exactly ONE lineage


@dataclass(frozen=True, eq=False)
class VerifiedRestore(_Minted):
    """One-shot capability of a file restore that has passed the container's verification: it names the actual restored event, state and the expected sealed manifest digest. Minted only by `verified_restore`; consumed by `stage_restored_head`."""

    event: Any
    state: Any
    seal: str


def saveable_head(lineage: Any) -> tuple:
    """Authority of a SAVE: the lineage's accepted head must be a genuine eligible `Boundary` (nothing pending) or a genuine `DailyHold` whose binding names THIS lineage (the old parent Boundary may be long dead -- it is neither
    required nor retained). Returns `(kind, head)` with kind in {"boundary", "daily_hold"}. Nothing moves; an Occurrence (a running storm, a mid-day trial) is refused."""
    if type(lineage) is not FloatingEventLineage:
        raise LifecycleError("lineage must be a FloatingEventLineage")
    head = lineage._head
    if type(head) is Boundary and head in _RECORDS:
        if head.role != "boundary" or head.pending_forcing is not None or head.storm_match is not None or (head.cursor is not None and head.cursor.pending is not None):
            raise LifecycleError("only a complete boundary with nothing pending is saveable (a pending recession or a pending storm is not carried by this file)")
        return "boundary", head
    if type(head) is DailyHold and head in _RECORDS:
        bound = _HOLD_BINDING.get(head)
        if bound is None or bound[1]() is not lineage:
            raise LifecycleError("the daily hold is not bound to this lineage")
        if head.pending_forcing is not None or head.storm_match is not None or (head.cursor is not None and head.cursor.pending is not None):
            raise LifecycleError("a hold with a pending storm is not saveable by this file")
        return "daily_hold", head
    raise LifecycleError("the lineage head is not a saveable accepted Boundary or DailyHold (an Occurrence / running storm is refused)")


def check_closed_day(hold: Any, closed: Any) -> None:
    """The closed daily state must be a genuinely minted CLOSED state of THIS hold (`ctx.hold is hold`)."""
    from maple_syrup import floating_daily as fd

    if type(closed) is not fd.FloatingDailyState or closed not in fd._MINTED or closed.role != "boundary" or closed.occurrence is not None or closed.ctx.hold is not hold or closed.cursor.pending is not None:
        raise LifecycleError("the daily state is not a genuine closed day of the head hold")


def _scalar_ledger(led: Any) -> dict:
    if type(led) is not dict or not set(_LEDGER_BASE) <= set(led) <= set(_LEDGER_BASE) | set(_LEDGER_OPTIONAL):
        raise LifecycleError("the lifecycle ledger keys are not the closed lifecycle set")
    out: dict = {}
    for k, v in led.items():
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise LifecycleError(f"lifecycle ledger {k} has the wrong type")
        if k in _LEDGER_COUNTS:
            if not isinstance(v, (int, np.integer)) or v < 0:
                raise LifecycleError(f"lifecycle ledger count {k} must be a non-negative integer")
            out[k] = int(v)
        else:
            if not np.isfinite(float(v)):
                raise LifecycleError(f"lifecycle ledger {k} is not finite")
            out[k] = float(v)
    return out


def head_record(head: Any) -> dict:
    """The bounded closed SCALAR record of an accepted `Boundary` / `DailyHold` (no arrays, no references)."""
    from maple_syrup import recharge as rc

    _record(head, type(head))
    return {"seq": int(head.seq), "t_s": float(head.t_s), "ledger": _scalar_ledger(dict(head.ledger)), "provenance": {k: head.provenance.get(k) for k in sorted(head.provenance)},
            "cursor": None if head.cursor is None else rc.cursor_record(head.cursor)}


def verified_restore(event: Any, state: Any, seal: str) -> VerifiedRestore:
    """Mint the one-shot token of a VERIFIED restore. The restored pair must be the ACTUAL native event and a state blessed by it whose soil water is registered as the held-context vertical's CURRENT payload -- a state exists on such a
    vertical only through a verified `floating_resume` adoption (the vertical has no initial payload, no owner and no bed)."""
    from maple_syrup.pond_event import PondEvent, PondEventState

    v = getattr(event, "vertical", None)
    if type(event) is not PondEvent or type(v) is not fp.FloatingPondVertical or v._held is None or v._owner is not None or v.backend not in ("numpy", "numba", "cuda"):
        raise LifecycleError("the restored event must be the actual native PondEvent on a held-context floating vertical (CPU or the explicit CUDA restore)")
    if type(state) is not PondEventState or state not in event._blessed or state.status != "wet" or state.counters.n_refreshes != 0:
        raise LifecycleError("the restored state is not a state the event blessed (a verified restore adoption)")
    try:
        v.current_payload(state.hydraulic.soil_water_m)
    except fp.FloatingPondError as exc:
        raise LifecycleError(f"the restored soil water is not the vertical's current payload: {exc}") from None
    if v in _RESTORED_VERTICALS:
        raise LifecycleError("this restored vertical already opened a lineage (one-shot)")
    if type(seal) is not str or len(seal) != 64:
        raise LifecycleError("the expected seal must be a 64-hex digest")
    with _minting():
        return VerifiedRestore(event, state, seal)


def _check_provenance(prov: Any, witness: str) -> dict:
    if type(prov) is not dict or not set(prov) <= _PROVENANCE_KEYS or not {"cold_anchor_witness", "property_basis", "execution_epoch", "cold_content_hash"} <= set(prov):
        raise LifecycleError("the provenance keys are not the closed lifecycle set")
    epoch = prov["execution_epoch"]
    if (prov["cold_anchor_witness"] != witness or prov["property_basis"] not in ("initial_full", fer.PROPERTY_BASIS) or prov["cold_content_hash"] is not None
            or not (epoch is None or (type(epoch) is str and len(epoch) == 64)) or ("area_m2" in prov and type(prov["area_m2"]) is not float)):
        raise LifecycleError("the provenance is not bound to the restored root's cold anchor / the closed lifecycle vocabulary")
    return dict(prov)


_PROVENANCE_KEYS_CUDA = _PROVENANCE_KEYS | {"backend", "kernel"}


def _check_provenance_cuda(prov: Any, witness: str) -> dict:
    """CUDA twin of `_check_provenance` (the CPU vocabulary and behaviour are unchanged and strict): the EXACT keys a genuine CUDA open/close writes -- the CPU keys plus `backend` (required, exactly "cuda") and `kernel` (the
    CURRENT exact kernel/ABI descriptor; required on a refreshed root, absent on an initial-full root as `open_cuda` writes it) -- unknown keys, a wrong backend and a wrong kernel digest are refused; the ORIGINAL canonical dict is
    returned so the saved head record round-trips unchanged (nothing dropped, nothing re-derived)."""
    from maple_syrup import floating_held_context_cuda as fhc

    if type(prov) is not dict or not set(prov) <= _PROVENANCE_KEYS_CUDA or not {"cold_anchor_witness", "property_basis", "execution_epoch", "cold_content_hash", "backend"} <= set(prov):
        raise LifecycleError("the CUDA provenance keys are not the closed CUDA lifecycle set")
    if prov["backend"] != "cuda":
        raise LifecycleError("the CUDA provenance backend is not exactly 'cuda'")
    if "kernel" in prov:
        if prov["kernel"] != fhc.kernel_descriptor():
            raise LifecycleError("the provenance kernel/ABI descriptor differs from the current exact CUDA kernel descriptor")
    elif prov["property_basis"] != "initial_full":
        raise LifecycleError("a refreshed CUDA root's provenance must carry the kernel/ABI descriptor")
    core = {k: v for k, v in prov.items() if k not in ("backend", "kernel")}
    _check_provenance(core, witness)  # the shared closed vocabulary check on the CPU keys (witness, basis, epoch, hash, area), unchanged
    return dict(prov)


def _balance_check(led: dict, stored_m3: float) -> tuple:
    """The cumulative lifecycle water identity (the hand-back's own formula) must close within the accumulated MAPLE bound, else a typed refusal."""
    expected = (led["storage0_m3"] + led["rain_m3"] - led["drainage_m3"] - led["outlet_export_m3"] - led["bottom_export_m3"] - led.get("evaporation_m3", 0.0) - led.get("transpiration_m3", 0.0))
    resid = float(stored_m3) - expected
    bound = led["bound_m3"] + volume_roundoff_bound_m3(16, max(abs(stored_m3), abs(expected), led["rain_m3"], led["drainage_m3"], led["outlet_export_m3"], led["bottom_export_m3"], led.get("evaporation_m3", 0.0),
                                                              led.get("transpiration_m3", 0.0), 1e-30))
    if not (np.isfinite(resid) and abs(resid) <= bound):
        raise LifecycleError(f"the restored water identity does not close: residual {resid!r} m3 exceeds the MAPLE volume bound {bound!r} m3")
    return resid, bound


def stage_restored_head(token: Any, kind: str, head_rec: Any, hold_rec: Any = None, *, cuda_quota: Any = None) -> tuple:
    """Stage (NOT install) the head of a restored lineage from a verified token: `kind` "boundary" mints a `Boundary` (a fresh lineage, no old parent), "daily_hold" mints the `DailyHold` (record of the hold as it was staged:
    static pieces, ledger, seq, clock, cursor; its surface and bed are the restored state's) bound weakly to a NEW lineage. Returns `(lineage, head, root)`; `install_restored_head` replaces the head LAST.
    CUDA restore (explicit): the SAME staging with resident comparisons/storage (`floating_event_lifecycle_cuda.arrays_equal_cuda` / `stored_m3_cuda`: counted scalar reads, no host water mirror) and the SEALED host
    mask/edges/D0 of the held CUDA context carried on the root/hold; `cuda_quota` (the caller's strict two-pool quota) is its admission; a CPU restore refuses `cuda_quota`."""
    from maple_syrup import floating_soil_layers as fsl
    from maple_syrup import recharge as rc

    _record(token, VerifiedRestore)
    event, state = token.event, token.state
    vert = event.vertical
    if vert in _RESTORED_VERTICALS:
        raise LifecycleError("this restored vertical already opened a lineage (one-shot)")
    if kind not in ("boundary", "daily_hold") or type(head_rec) is not dict or set(head_rec) != {"seq", "t_s", "ledger", "provenance", "cursor"}:
        raise LifecycleError("the head record is malformed (closed key set)")
    if type(head_rec["seq"]) is not int or head_rec["seq"] < 0 or type(head_rec["t_s"]) is not float or not np.isfinite(head_rec["t_s"]):
        raise LifecycleError("the head record seq/clock are malformed")
    payload = vert.current_payload(state.hydraulic.soil_water_m)
    held = vert.held_template()
    cuda = vert.backend == "cuda"
    if not cuda and cuda_quota is not None:
        raise LifecycleError("cuda_quota applies only to a CUDA restore (a meaningless CPU argument is refused)")
    if cuda:
        from maple_syrup import floating_event_lifecycle_cuda as flc

        same_depth = flc.arrays_equal_cuda(held.column, payload.depth_m, held.depth0_m)  # resident comparison: ONE counted scalar read
    else:
        same_depth = np.array_equal(np.asarray(payload.depth_m), np.asarray(held.depth0_m))
    if not same_depth:
        raise LifecycleError("the restored canonical depth differs from the held root's prepared depth")
    led = _scalar_ledger(head_rec["ledger"])
    prov = _check_provenance_cuda(head_rec["provenance"], held.source_witness) if vert.backend == "cuda" else _check_provenance(head_rec["provenance"], held.source_witness)
    try:
        cursor = None if head_rec["cursor"] is None else rc.cursor_from_record(head_rec["cursor"])
    except rc.RechargeError as exc:
        raise LifecycleError(f"the cursor record is refused: {exc}") from None
    if cursor is not None and cursor.pending is not None:
        raise LifecycleError("a pending storm is not carried by this file")
    bed = state.bed
    hyd = state.hydraulic
    routing = (state.topology, state.held_source_m, state.datum)
    area = float(held.column.cell_area_m2)
    host = (vert._active_host(), vert._edges_host, vert._held.host_depth0_m) if cuda else (None, None, None)  # CUDA: the SEALED host mask / edges / D0 of the held context, by reference
    with _minting():
        root = BoundaryRoot(held.column, held.matrix_op, held.preferential, held.ks_m_s, held.depth0_m, held.depth0_m, payload.history, payload.state, vert.config, int(held.generation), held.source_witness, bed, *host)
        lineage = FloatingEventLineage()
        if kind == "boundary":
            if float(head_rec["t_s"]) != float(state.t_s) or float(payload.state.t_s) != float(state.t_s):
                raise LifecycleError("the boundary clock differs from the restored state's / soil's clock")
            if cuda:
                _balance_check(led, flc.stored_m3_cuda(held.column, payload.state, hyd.depth_m, area, cuda_quota))
            else:
                _balance_check(led, float(fsl.floating_storage_m3(held.column, payload.state)) + float(np.asarray(hyd.depth_m).sum()) * area)
            head = Boundary("boundary", root, bed, hyd.depth_m, hyd.qx_m2_s, hyd.qy_m2_s, routing, float(state.t_s), led, prov, int(head_rec["seq"]), cursor, None, None)
        else:
            if type(hold_rec) is not dict or set(hold_rec) != {"closed_t_s"} or float(hold_rec["closed_t_s"]) != float(state.t_s):
                raise LifecycleError("the hold record's closed-day clock differs from the restored state's clock")
            head = DailyHold(held.column, held.matrix_op, held.preferential, held.ks_m_s, held.depth0_m, vert.config, int(held.generation), held.source_witness, bed, hyd.depth_m, hyd.qx_m2_s, hyd.qy_m2_s, routing,
                             led, prov, int(head_rec["seq"]), float(head_rec["t_s"]), cursor, None, None, host_active=host[0], edges_host=host[1], host_depth=host[2])
    if kind == "daily_hold":
        class _Gone:  # the original parent Boundary is NOT restored: its weak slot is dead, exactly as after a parent release
            pass

        _HOLD_BINDING[head] = (weakref.ref(_Gone()), weakref.ref(lineage))
    return lineage, head, root


def install_restored_head(token: Any, lineage: Any, head: Any) -> None:
    """Replace the (None) head of a restored lineage LAST and consume the one-shot token and vertical."""
    _record(token, VerifiedRestore)
    if lineage._head is not None or type(lineage) is not FloatingEventLineage or type(head) not in (Boundary, DailyHold) or head not in _RECORDS:
        raise LifecycleError("a restored head installs on a fresh lineage only")
    vert = token.event.vertical
    if vert in _RESTORED_VERTICALS:
        raise LifecycleError("this restored vertical already opened a lineage (one-shot)")
    _RESTORED_VERTICALS.add(vert)
    lineage._head = head
