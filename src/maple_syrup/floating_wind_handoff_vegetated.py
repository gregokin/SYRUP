"""Separate vegetation-aware Dense NumPy wind handoff (V2).

WH1 remains byte-exact and refuses vegetated parents. This owner carries one
verified vegetation generation and held law through real finalized Native
wind and the existing zero-time property reconciliation. It takes no plant,
hydrology, E/T or growth step. Native clocks, routing and all water/conservation
checks retain WH1 semantics. Source/model qualification is Root-owned.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import weakref
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from maple_syrup import floating_wind_handoff as wh
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_event_refresh as fer
from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup import water_vegetation_binding as wvb
from maple_syrup.floating_wind_handoff import (
    _limits, _occurrence, _clock, _bed_meta, _wind_meta, _native_bed,
    _dense_numpy, _wind_values, _same_leaves, _native_valid, _descriptor,
    _dirty_columns, _seal, _json, _callback, _weak,
)

PROFILE = "maple-syrup-floating-wind-handoff-vegetated/1"
RECEIPT_SCHEMA = "maple_syrup.floating_wind_handoff_vegetated_receipt.v1"
WindHandoffError = wh.WindHandoffError
WindHandoffLimitError = wh.WindHandoffLimitError
WindHandoffLimits = wh.WindHandoffLimits
WindContext = wh.WindContext
BedState = wh.BedState
start_wind_boundary = wh.start_wind_boundary  # genuine bare start, BEFORE apply
_HASH_SCRATCH = wh._HASH_SCRATCH
_ADOPT_BYTES = wh._ADOPT_BYTES
_CONTEXTS = wh._CONTEXTS
_Record = wh._Record
_MINT = threading.local()
_CANDIDATES = weakref.WeakSet()
_INTEGRITY = weakref.WeakKeyDictionary()
_QUOTAS = weakref.WeakKeyDictionary()
_LAST = weakref.WeakKeyDictionary()

__all__ = ["PROFILE", "WindHandoffError", "WindHandoffLimitError", "WindHandoffLimits", "WindContext",
           "WindHandoffCandidate", "start_wind_boundary", "restore_wind_context", "stage_wind_handoff",
           "adopt_wind_handoff", "estimate_vegetated_handoff_bytes"]


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


@dataclass(frozen=True, eq=False)
class _Parts:
    base: Any  # exact WH1 field carrier; no plant/projection recursion
    vegetation: Any
    host_active: Any
    edges_host: Any
    host_depth: Any


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


def _vegetation(boundary: Any) -> Any:
    """Genuine current record and current cell-law domain, METADATA ONLY."""
    record = vl.current_record(boundary)
    if record is None:
        raise WindHandoffError("V2 requires an applied vegetation generation; a bare boundary belongs to WH1")
    root = boundary.root
    if (record.backend != "numpy" or record.device_id is not None or record.device_bytes != 0
            or root.column.backend != "numpy" or root.host_active is not None
            or type(boundary.surface_depth_m) is not np.ndarray):
        raise WindHandoffError("vegetated wind handoff is Dense NumPy CPU only")
    if ((record.n_cells, record.n_layers) != (root.column.n_cells, root.column.n_layers)
            or tuple(record.shape) != tuple(boundary.surface_depth_m.shape)):
        raise WindHandoffError("the genuine vegetation record does not describe this boundary domain")
    factor = fm.held_cell_vegetation_factor(root.matrix_op)
    if (type(factor) is not np.ndarray or factor.dtype != np.float64 or factor.shape != (record.n_cells,)
            or not factor.flags.c_contiguous or factor.flags.writeable
            or root.matrix_op.column_sha256 != root.column.sha256
            or root.pref_op is None or root.pref_op.column_sha256 != root.column.sha256):
        raise WindHandoffError("the boundary must hold its genuine per-cell matrix and preferential laws")
    return record


def estimate_vegetated_handoff_bytes(boundary: Any, limits: Any) -> dict:
    """Positive EXTRA model beside unchanged WH1/native terms (metadata only).

    Borrowed record/projection and existing operators remain live. Owning native
    prepare/rebind estimates cover the new law/history, conservatively AGAIN
    beside nfs._estimate's <=3N-law/2N-history allowance. The g/J/I byte budget
    conservatively covers the rowwise byte comparisons. Three bounded receipt text/buffer
    occurrences cover the additional attribution. No allocation/heap theorem.
    """
    limits = _limits(limits)
    record = _vegetation(boundary)
    root = boundary.root
    N, L = int(root.column.n_cells), int(root.column.n_layers)
    prepare = fm.estimate_floating_matrix_memory(L, N, backend="numpy", operation="prepare")
    rebind = fm.estimate_floating_matrix_memory(L, N, backend="numpy", operation="rebind")
    terms = dict(record_owned=int(record.owned_bytes), binding_projection_borrowed=int(record.borrowed_bytes),
        held_matrix=int(root.matrix_op.retained_bytes), held_preferential=int(root.pref_op.retained_bytes),
        native_matrix_prepare=int(prepare.required_bytes), native_history_rebind=int(rebind.required_bytes),
        bitwise_checks=int(fm.held_cell_vegetation_factor(root.matrix_op).nbytes
            + root.history.wetting_m.nbytes + root.history.cumulative_intake_m.nbytes),
        receipt_buffers=3*limits.max_receipt_bytes)
    if any(type(value) is not int or value < 0 for value in terms.values()):
        raise WindHandoffError("invalid owning vegetation memory metadata")
    return dict(terms=terms, required_bytes=sum(terms.values()), known_subset_only=True,
                scope="extra held and temporary vegetation occurrences; original WH1/native terms remain")


def _factor_sha(factor: Any) -> str:
    h = hashlib.sha256()
    wvb._hash_chunked(h, factor)
    return h.hexdigest()


def _integrity(parent: Any, parts: _Parts, oid: str, end_s: float, text: str) -> str:
    if type(parts) is not _Parts or type(parts.base) is not wh._Parts:
        raise WindHandoffError("not this owner's staged field carrier")
    record = _vegetation(parent)
    if parts.vegetation is not record:
        raise WindHandoffError("the staged vegetation record is not the current parent record")
    h = hashlib.sha256(b"maple_syrup.floating_wind_handoff_vegetated.integrity/1")
    h.update(wh._integrity(parent, parts.base, oid, end_s, text).encode())
    # The record's owning verifier fingerprints EVERY field and its immutable
    # borrowed binding/covers. Do not walk a PlantSession/projection as a tree.
    h.update(str(id(record)).encode())
    h.update(_json(vl.vegetation_attribution(parent)).encode())
    h.update(_factor_sha(fm.held_cell_vegetation_factor(parent.root.matrix_op)).encode())
    wh._feed(h, (parts.host_active, parts.edges_host, parts.host_depth), {})
    return h.hexdigest()


def restore_wind_context(boundary: Any, bed_context: Any, expected_record: Any) -> WindContext:
    """Restore only the water-free static context of a genuine vegetated head.

    The Phase2.5 owning checkpoint restores the record/law first. This never
    mints a generation or takes the WH1 guarded vegetated restore route.
    """
    lc._record(boundary, lc.Boundary)
    _vegetation(boundary)
    if (type(expected_record) is not dict or set(expected_record) != wh._RECORD_KEYS
            or expected_record["schema"] != wh.CONTEXT_SCHEMA
            or expected_record["representation"] != "dense" or expected_record["backend"] != "numpy"):
        raise WindHandoffError("the expected static wind context record is malformed")
    if not isinstance(bed_context, wh.BedContext):
        raise WindHandoffError("bed_context must be the static native BedContext of the case")
    root = boundary.root
    ny, nx = int(bed_context.geometry.ny), int(bed_context.geometry.nx)
    if root.column.n_cells != ny*nx:
        raise WindHandoffError("the static case grid differs from the boundary")
    nb = _native_bed(bed_context, boundary.bed, np.asarray(root.column._maps["active"]).reshape(ny, nx))
    _dense_numpy(nb, root.config, "restored vegetated")
    context = wh._mint_context(bed_context, nb, root.source_witness, root.config.sha256)
    if context.record() != expected_record:
        raise WindHandoffError("the restored static wind context differs from the expected record")
    return context


def stage_wind_handoff(lineage: Any, parent: Any, final: Any, child: Any, *, context: Any, occurrence_id: Any, start_s: Any, limits: Any, available_bytes_fn: Any, occurrence_sha256: Any = None, reserve_bytes: int = 0,
                       batch_cells: int = nfs.DEFAULT_BATCH_CELLS) -> WindHandoffCandidate:
    """PURE staging of one wind handoff (module docstring). Order: head/eligibility -> typed backend/representation/limits and input METADATA (no coercion, no scan) -> aggregate admission (limits, then the real reader) -> pool predicate and class-vector values ->
    water carry -> full native bed validity -> descriptor -> cold row comparison -> candidate admission at the actual dirty count -> selected read, changed set, reconcile, closure -> new root/state -> surface return -> cumulative water identity -> bounded receipt ->
    integrity digest. Any refusal leaves the head, the parent, the child and every input untouched."""
    lc._expect(lineage, parent, lc.Boundary)
    vegetation = _vegetation(parent)
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
    vegetation_plan = estimate_vegetated_handoff_bytes(parent, limits)
    vegetation_extra = vegetation_plan["required_bytes"]
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
    base_cand = (L + 1) * N * 8 + 3072 + 112 * nc + vegetation_extra  # the continued soil-state copy, the surface/return array and the scalar receipt (exist even with no dirty column)
    if scan_need > limits.max_scan_bytes:
        raise WindHandoffLimitError(f"the cold scan, pool predicate and validity need a modelled {scan_need} B above max_scan_bytes {limits.max_scan_bytes}; nothing was read")
    if base_cand > limits.max_candidate_bytes:
        raise WindHandoffLimitError(f"the candidate (soil-state copy, checks and receipt, even with no dirty column) needs a modelled {base_cand} B above max_candidate_bytes {limits.max_candidate_bytes}; nothing was copied")
    nfs._admit({"required_bytes": scan_need + base_cand, "terms": {"scan_pool_validity": scan_need, "candidate_base": base_cand, "reserve": reserve}}, None, available_bytes_fn, "wind handoff cold comparison and candidate")
    B = int(model0["batch_cells"])
    # ---- values (only now)
    factor_before = _factor_sha(fm.held_cell_vegetation_factor(root.matrix_op))
    if factor_before != vegetation.applied_g_sha256:
        raise WindHandoffError("the held cell factor differs from this generation's applied factor")
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
            new_op = nfs._fresh_op(cfg, new_column, ks, vegetation=fm.held_cell_vegetation_factor(root.matrix_op))
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
            new_op = nfs._fresh_op(cfg, new_column, ks, vegetation=fm.held_cell_vegetation_factor(root.matrix_op))
            new_hist = fm.rebind_floating_matrix_history(root.matrix_op, root.history, new_op).history
            if new_column is not root.column:
                new_pref = None if root.pref_op is None else fpi.rebind_preferential_column(root.pref_op, new_column)
            ks_out = ks if ks_changed else root.ks_m_s
    if not wh._same_bits(fm.held_cell_vegetation_factor(root.matrix_op), fm.held_cell_vegetation_factor(new_op)):
        raise WindHandoffError("property refresh changed the held per-cell vegetation factor")
    if (not wh._same_bits(root.history.wetting_m, new_hist.wetting_m)
            or not wh._same_bits(root.history.cumulative_intake_m, new_hist.cumulative_intake_m)):
        raise WindHandoffError("property refresh changed persistent J/I")
    factor_after = _factor_sha(fm.held_cell_vegetation_factor(new_op))
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
    rec["vegetation"] = dict(generation=vegetation.generation, recipe_id=vegetation.recipe_id,
        binding_sha256=vegetation.binding_sha256, held_factor_sha256_before=factor_before,
        held_factor_sha256_after=factor_after, covers="unchanged", current_before=vl.vegetation_attribution(parent),
        current_after=dict(column_sha256=new_column.sha256, matrix_op_sha256=new_op.sha256,
            preferential_sha256=new_pref.sha256, property_generation=generation))
    rec["plan"]["vegetation_extra"] = vegetation_plan
    text = _json(rec)
    if len(text.encode()) > limits.max_receipt_bytes:
        raise WindHandoffLimitError(f"the scalar receipt needs {len(text.encode())} B above max_receipt_bytes {limits.max_receipt_bytes}")
    base = wh._Parts(new_column, new_op, new_pref, ks_out, depth_out, new_hist, new_state, cfg, generation, root.source_witness, bed, surface, parent.qx_m2_s, parent.qy_m2_s, parent.routing, end_s, led, prov, int(parent.seq) + 1, parent.cursor, role)
    parts = _Parts(base, vegetation, root.host_active, root.edges_host, root.host_depth)
    expected = _integrity(parent, parts, oid, end_s, text)  # the post-stage integrity, bound to the CURRENT parent and every proposed field
    with _minting():
        cand = WindHandoffCandidate(oid, end_s, role, text, weakref.ref(parent), weakref.ref(lineage), {"parts": parts, "consumed": False, "accepted": None})
    _QUOTAS[cand] = (vegetation_extra, limits.max_receipt_bytes)
    _INTEGRITY[cand] = expected  # PRIVATE weak table, not the caller-reachable cell
    return cand


def adopt_wind_handoff(lineage: Any, parent: Any, candidate: Any, *, available_bytes_fn: Any = None, reserve_bytes: int = 0, before_publish: Any = None) -> lc.Boundary:
    """Adopt a staged candidate (module docstring). Order: genuine candidate, expected head, one-shot state, parent/lineage binding, immediate-duplicate id -> admission (actual host reader when omitted) -> `before_publish` (outside every mint window) -> head re-check and the POST-STAGE INTEGRITY audit
    -> preallocation of every bookkeeping slot -> mint -> weak accepted reference -> head install (first commit store) -> consumed/emptied/last-id stores (no allocation). Every refusal before the head store leaves the head and the candidate retryable."""
    if type(candidate) is not WindHandoffCandidate or candidate not in _CANDIDATES:
        raise WindHandoffError("not a genuine WindHandoffCandidate")
    lc._expect(lineage, parent, lc.Boundary)
    vegetation = _vegetation(parent)
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
    parts = cell["parts"]
    quota = _QUOTAS.get(candidate)
    if (type(parts) is not _Parts or type(parts.base) is not wh._Parts or parts.vegetation is not vegetation
            or type(quota) is not tuple or len(quota) != 2
            or type(candidate.receipt_json) is not str or not candidate.receipt_json.isascii()
            or len(candidate.receipt_json) > quota[1]):
        raise WindHandoffError("the staged vegetation fields or receipt differ from this owner")
    reserve = fsl._count(reserve_bytes, "reserve_bytes")
    nfs._admit({"required_bytes": _ADOPT_BYTES + quota[0] + reserve, "terms": {"adoption": _ADOPT_BYTES, "vegetation_extra": quota[0], "reserve": reserve}}, None, available_bytes_fn, "wind handoff adoption")
    if before_publish is not None:
        before_publish()  # OUTSIDE every mint window and BEFORE anything irreversible: the coordinator re-checks its immutable source/case here; a failure preserves the head and the unconsumed candidate
        lc._expect(lineage, parent, lc.Boundary)
        if cell["consumed"] or cell["parts"] is None:
            raise WindHandoffError("this wind handoff was already adopted (a candidate is consumed exactly once)")
    parts = cell["parts"]
    if _integrity(parent, parts, oid, candidate.end_s, candidate.receipt_json) != _INTEGRITY.get(candidate):
        raise WindHandoffError("the post-stage integrity audit failed: the child, the candidate's proposed state or the parent changed in place after staging (nothing was consumed or published)")
    # ---- preallocate EVERY bookkeeping slot before anything is consumed or moved
    if last is None:
        last = [None]
        _LAST[lineage] = last
    p = parts.base
    with lc._minting():  # nothing refusable runs inside the window
        root = lc.BoundaryRoot(p.column, p.matrix_op, p.pref_op, p.ks_m_s, p.depth_m, p.depth_m, p.history, p.water_state, p.config, p.generation, p.source_witness, p.bed, parts.host_active, parts.edges_host, parts.host_depth, vegetation=parts.vegetation)
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

