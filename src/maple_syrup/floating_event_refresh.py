"""EB2: STAGED selected property refresh from the CURRENT inventory of a genuinely COMPLETED Pond event (CPU: NumPy/Numba; Dense and published Windowed). It stages a candidate; it adopts nothing, consumes nothing and mutates nothing.

Inputs are only a genuine EB1 completion capability, its state and its event. Verified through the producer (`floating_completion.verify_completion`), the CURRENT payload of the floating vertical (W, J/I, current canonical depth D,
held law, preferential operator) comes from `FloatingPondVertical.event_end_bundle`; a caller-built bundle or dirty mask is never accepted. No `NativeFloatingSoil` (its W0 is stale after a storm), no old/parent bed, no cold source
hash and no whole-bed scan are involved. Nothing proportional to N is allocated before the first admission (the host active mask is the vertical's sealed array, reshaped by reference).

Dirty columns: the GROSS per-cell pickup + deposition of the producer's cumulative sediment ledger (`SedimentLedger.pickup_cell` + `deposition_cell`: wet, splash and the terminal deposit; net-zero turnover included; avalanching is refused by
the event) restricted to the active cells. COVERAGE ASSUMPTION (explicit): every bed write of a Pond event goes through those three exchanges, so a column whose composition or height could change is in the dirty set; this is argued
from the closed writer set and checked by a full cold parent/child oracle in the TESTS only (never in production). It says nothing about wind.
Selected read: the accepted compact reader (`nfs._read_texture`) on ONLY the dirty columns of the CURRENT native bed; texture -> FC/SAT/Ks by the accepted provider; properties are frozen during a storm so the held root maps are the
"old" properties. Changed set = accepted `_changed_columns_parts` (porous geometry/bottom flag or FC/SAT; Ks is NOT a trigger): only those columns pass through the accepted zero-time `_reconcile_changed` on the CURRENT W (explicit surface
return and bottom export, each exactly once; unchanged and inactive columns bit for bit). A new root column is prepared by the accepted `_build_column`, a new matrix operator by `_fresh_op`, the persistent J/I are rebound bit for bit by
the accepted `rebind_floating_matrix_history` (no reset, no intake, no time, no rain, no E/T, no diffusion) and the preferential operator by `rebind_preferential_column` (its law depends only on a3/weights). Routing/datum/event-held
state are untouched. The candidate keeps only a reference to the CURRENT child bed.

Provenance: `property_basis = "boundary_selected"` and an `execution_epoch/v1` label derived from the event facts and the new column identity; it is NOT a cold content hash of the bed (`cold_content_hash` is None). The candidate is physical
staging only; a later lifecycle transaction must adopt it and consume the completion capability LAST.
Memory (declared modelled host bytes, not a whole-process bound): stage 0 (the O(N nc) dirty selection) then the accepted compact-reader / provider / persistent-candidate terms of `nfs._estimate` at the actual selected count B x V x K
and the real Windowed cache bound, one reserve, both admitted from the real reader BEFORE the arrays they cover.
"""
from __future__ import annotations

import contextlib
import hashlib
import threading
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import floating_completion as fcm
from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs

__all__ = ["EventRefreshCandidate", "EventRefreshError", "stage_completed_event_refresh", "verify_refresh_candidate"]

PROPERTY_BASIS = "boundary_selected"
EPOCH_KIND = "execution_epoch/v1"
_MINT = threading.local()
_CANDIDATES: weakref.WeakSet = weakref.WeakSet()


class EventRefreshError(nfs.NativeFloatingError):
    """Typed refusal of the event-boundary refresh."""


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


@dataclass(frozen=True, eq=False)
class EventRefreshCandidate:
    """Genuine staged refresh bound to the EXACT completion capability / state / event / payload / current child bed. Uncopyable, unpicklable, minted only by `stage_completed_event_refresh`; not an adopted parent."""

    capability: Any
    state: Any
    event: Any
    payload: Any
    child_bed: Any
    column: Any  # the NEW prepared root column (current canonical D)
    water_state: Any  # fsl.FloatingSoilState on the new column (current W with the changed columns reconciled)
    matrix_op: Any
    history: Any  # J/I rebound bit for bit
    pref_op: Any
    ks_m_s: Any
    depth_m: np.ndarray
    surface_return_m: np.ndarray  # (N,) m, to be added to the surface depth by the adopter, once
    bottom_export_m: np.ndarray  # (N,) m leaving through the open bottom, once
    receipt: Any  # nfs.NativeReconcileReceipt
    dirty: np.ndarray
    changed: np.ndarray
    depth_changed: np.ndarray
    ks_changed: bool
    property_basis: str
    provenance: Any
    model: Any

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise EventRefreshError("an EventRefreshCandidate is minted only by stage_completed_event_refresh")
        _CANDIDATES.add(self)

    def __copy__(self):
        raise EventRefreshError("an EventRefreshCandidate cannot be copied")

    def __deepcopy__(self, memo):
        raise EventRefreshError("an EventRefreshCandidate cannot be copied")

    def __reduce_ex__(self, protocol):
        raise EventRefreshError("an EventRefreshCandidate is not serialisable")


def verify_refresh_candidate(cand: Any) -> EventRefreshCandidate:
    """Genuine candidate whose completion is still valid and unconsumed (the producer is asked again); changes nothing."""
    if type(cand) is not EventRefreshCandidate or cand not in _CANDIDATES:
        raise EventRefreshError("not a genuine EventRefreshCandidate")
    fcm.verify_completion(cand.capability, cand.state, event=cand.event)
    return cand


def _seal(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


def _reader_need(reader: Any, need: int, what: str) -> None:
    nfs._admit({"required_bytes": int(need), "terms": {}}, None, reader, what)


def _gross_dirty(state: Any, active: np.ndarray, shape: tuple) -> np.ndarray:
    """Flat indices of the ACTIVE cells whose gross pickup or deposition in the producer's cumulative ledger is non-zero (net-zero turnover included)."""
    led = state.sediment
    ny, nx = shape
    pick, dep = np.asarray(led.pickup_cell), np.asarray(led.deposition_cell)
    if pick.shape[:2] != (ny, nx) or dep.shape[:2] != (ny, nx):
        raise EventRefreshError("the producer's gross ledger has another grid than the event")
    touched = np.any(pick != 0, axis=2) | np.any(dep != 0, axis=2)
    return np.flatnonzero(touched.reshape(-1) & active.reshape(-1)).astype(np.int64)


def stage_completed_event_refresh(cap: Any, state: Any, *, event: Any, available_bytes_fn: Any, batch_cells: int = nfs.DEFAULT_BATCH_CELLS, reserve_bytes: int = 0) -> EventRefreshCandidate:
    """Stage the selected property refresh of a genuinely completed event (module docstring). Pure: every refusal raises before/without mutating the event, its registries, the vertical or the capability, which stays
    unconsumed; repeated staging of the same pair is valid. Order: completion verified through the producer -> typed CUDA/type refusal (before any array or transfer) -> stage-0 admission -> gross dirty selection ->
    stage-1 composed admission -> selected read -> changed set -> reconciliation on the CURRENT W -> closure -> new column/operator/rebound history/preferential -> sealed candidate."""
    from maple_syrup import floating_pond as fp

    fcm.verify_completion(cap, state, event=event)
    vert = getattr(event, "vertical", None)
    if type(vert) is not fp.FloatingPondVertical:
        raise EventRefreshError("the event does not run a genuine floating vertical (the bucket column has no floating inventory to refresh)")
    if vert.backend not in ("numpy", "numba"):
        raise EventRefreshError(f"the event-boundary refresh serves the CPU verticals; backend {vert.backend!r} is refused before any array or transfer (the CUDA stage is a later slice)")
    if not callable(available_bytes_fn):
        raise EventRefreshError("available_bytes_fn must be callable (the admission needs a real reader)")
    reserve = fsl._count(reserve_bytes, "reserve_bytes")
    B0 = fsl._count(batch_cells, "batch_cells", minimum=1)
    bundle = vert.event_end_bundle(cap, state, event)
    payload, held, cfg = bundle.payload, bundle.held, vert.config
    ny, nx = event.shape
    N, nc = ny * nx, int(event.nc)
    active = np.asarray(vert._active_host()).reshape(-1)
    ctx = event.context
    child = nfs.NativeBed(state.bed.voxel_column, state.bed.active_layer, ctx.geometry, ctx.grain_classes, ctx.mass_resolution_kg, active.reshape(ny, nx))  # the vertical's sealed host mask BY REFERENCE: no owned copy before the first admission
    nfs._check_bed(child, cfg)
    if child.representation == "windowed":
        nfs._windowed_checks(child, "stage_completed_event_refresh")
    _reader_need(available_bytes_fn, 2 * N * nc + 11 * N + reserve, "event-boundary refresh dirty selection")  # stage 0: the bool temporaries and the index/mask arrays
    dirty = _gross_dirty(state, active, event.shape)
    model = nfs._estimate(None, child, cfg, B0, reserve, n_selected=int(dirty.size))
    terms = model["terms"]
    need = int(terms["persistent_candidate"]) + int(terms["child_native_and_properties"]) + 3 * N * 8 + reserve  # + the dirty/index arrays and the sealed host copies
    stage1 = {"required_bytes": need, "terms": {"persistent_candidate": terms["persistent_candidate"], "child_native_and_properties": terms["child_native_and_properties"], "dirty_and_index": 3 * N * 8, "reserve": reserve}}
    nfs._admit(stage1, None, available_bytes_fn, "event-boundary selected property refresh")
    B = int(model["batch_cells"])
    maps = held.column._maps
    d_cur = np.asarray(payload.depth_m)
    fc, sat, depth = (np.array(maps["theta_field_capacity"], copy=True), np.array(maps["theta_saturation"], copy=True), np.array(d_cur, copy=True))
    ks = None if held.ks_m_s is None else np.array(held.ks_m_s, copy=True)
    nfs._read_texture(child, cfg, dirty, fc, sat, depth, model=model, batch_cells=B, update_properties=True, ks=ks)
    changed, depth_changed = nfs._changed_columns_parts(cfg, maps, d_cur, dirty, fc, sat, depth, B)
    ks_changed = ks is not None and bool(np.any(ks[dirty] != held.ks_m_s[dirty]))
    area = nfs._area(child)
    old_water = np.asarray(payload.state.water_m)
    t_s = float(payload.state.t_s)
    water, surf, exp_, acc = nfs._reconcile_changed(cfg, maps, d_cur, old_water, fc, sat, depth, changed, area, B, t_s)
    new_column = nfs._build_column(cfg, fc, sat, depth, child)
    mask = np.asarray(new_column._maps["active"])
    new_state = fsl.floating_continued_state(new_column, np.ascontiguousarray(water), t_s)
    w_before, w_after, resid, bound = nfs._inventory_closure(water, old_water, mask, changed, B, area, acc, surf, exp_)
    if ks is not None:
        ks.flags.writeable = False
    new_op = nfs._fresh_op(cfg, new_column, ks)
    rb = fm.rebind_floating_matrix_history(payload.matrix_op, payload.history, new_op)
    pref = None if payload.pref_op is None else fpi.rebind_preferential_column(payload.pref_op, new_column)
    receipt = nfs.NativeReconcileReceipt(whole_before_m3=w_before, whole_after_m3=w_after, whole_residual_m3=resid, whole_bound_m3=bound, old_column_sha256=held.column.sha256, new_column_sha256=new_column.sha256, **acc)
    for fresh in (depth, surf, exp_, dirty, changed, depth_changed):
        _seal(fresh)
    f = cap.facts
    epoch = hashlib.sha256("|".join((EPOCH_KIND, f.config_sha256, f.forcing_sha256, str(f.topology_generation), str(f.n_accepted_steps), repr(f.t_s), new_column.sha256)).encode()).hexdigest()
    provenance = {"kind": EPOCH_KIND, "digest": epoch, "cold_content_hash": None, "note": "execution-epoch label of this event boundary; NOT a content hash of the bed"}
    with _minting():  # every refusable step has run
        return EventRefreshCandidate(cap, state, event, payload, child, new_column, new_state, new_op, rb.history, pref, ks, depth, surf, exp_, receipt, dirty, changed, depth_changed, ks_changed, PROPERTY_BASIS, provenance,
                                     {"required_bytes": need, "batch_cells": B, "n_dirty": int(dirty.size), "terms": stage1["terms"], "stage0_bytes": 2 * N * nc + 11 * N})
