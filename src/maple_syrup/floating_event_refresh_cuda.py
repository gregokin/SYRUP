"""EB3-CUDA: STAGED selected property refresh from the CURRENT inventory of a genuinely COMPLETED CUDA Pond event (Dense and published Windowed device beds). AUTHOR-WRITTEN, NOT RUN ON A DEVICE (Root qualifies it on hardware).
Importing this module never imports CuPy.

The CUDA counterpart of `floating_event_refresh` (EB2, CPU; byte-unchanged and still refusing CUDA events). It stages a candidate; it adopts nothing, consumes nothing and mutates nothing (no lifecycle adoption, no daily
cursor, no file integration here).

Inputs are only a genuine EB1 completion capability, its state and its event. The CURRENT payload of the floating vertical (device W, J/I, the current canonical depth `D`, held law, preferential operator) comes from
`FloatingPondVertical.event_end_bundle`; no `NativeFloatingSoil` (its W0 is stale after a storm), no fake owner, no old/parent bed, no cold whole-bed hash and no parent N x V x K scan are involved.

Property gating (the reviewed HYBRID): the property provider is the EXACT host NumPy tile provider the CUDA native owner already uses (`nfs._batch_properties` -> `nfs._provider_backend` == "numpy" for a CUDA owner), through the
accepted compact reader on ONLY the dirty columns. The held FC/SAT of every CUDA vertical are host-provider bytes, so an unchanged texture reproduces them EXACTLY: no tolerance gate, no forced cascade. A device-resident
provider is a future explicit initialization+refresh mode needing its own byte-gating proof; it is never a silent substitute. Selected tiles move through MAPLE's counted `to_host`/`to_device` (cold, bounded B).

Dirty columns: the GROSS per-cell pickup + deposition of the producer's cumulative sediment ledger, formed on the DEVICE (O(N K) bool temporaries, ONE intrinsic `flatnonzero` count read) restricted to the vertical's device
active map; the selected index vector is then downloaded once (cold, counted, chunked) because the compact reader needs host indices. Coverage assumption identical to EB2 (every Pond bed write goes through those three
exchanges), checked only by the full cold oracle in the TESTS, never in production.

Changed set = the accepted `nfs._geometry_changed` slab/bottom rule against the CURRENT depth `D` (NOT the held D0) or an exact FC/SAT difference; Ks is NOT a trigger (a Ks-only change replaces the operator without a cascade);
a depth change that leaves every slab and the strict bottom flag unchanged cascades nothing. Only the changed columns pass through the accepted owner-free D1 body `native_floating_soil_cuda.reconcile_selected_columns`
(explicit CURRENT column/state/depth, compact device sub-columns); unchanged and inactive columns keep W bit for bit; J/I are rebound exactly (`fm.rebind_floating_matrix_history`); the preferential operator is rebound to
the new CANONICAL root (`fpi.rebind_preferential_column`, CUDA branch: the identity `prepare_floating_preferential` would mint). The NEW prepared root column and matrix operator are minted by the public factories
(`fsl.prepare_floating_column`, `fm.prepare_floating_matrix`: content identities by streamed hash, never a geometry-view token). Whole-inventory closure: `inventory_closure_cuda` (two D1 storage queries, FP64 bound).

Order: genuine completion -> typed refusals (metadata only) -> argument validation -> the CURRENT-device gate (before any reader, hash, allocation) -> stage-0 two-pool admission (the dirty-selection temporaries and the O(N)
records) -> device dirty mask, index download, current-D record -> stage-1 composed two-pool admission (compact-reader/native/Windowed deep-store mapping-cache bound, provider tile, new maps/column/operator, sub-columns,
tiles, hash staging, records; reserves once, the 64 KiB host allowance) -> windowed validators -> selected read/scatter -> changed set -> reconcile -> closure -> new root/operator/history/preferential -> sealed candidate.
Nothing is registered or consumed before the sealed candidate is minted; every refusal leaves the event, vertical, capability and state untouched and a retry valid.

Cold traffic (counted through MAPLE's helpers unless stated): the dirty count read (8 B), the dirty index D2H (8 n), the current-D D2H (8 N, chunked), per selected tile the compact tile D2H and property H2D, the old-value
tile reads of the changed-set comparison, the new-depth H2D (8 N), per-chunk 1-byte unrefreshed checks and two 8-byte storage reads. The public canonical identity hashes stream the NEW static maps (FC/SAT/depth/mask,
operator law, preferential law) through bounded host staging buffers: that traffic is mandatory, is modelled (not hidden) and, for the preferential law, goes through the accepted `hash_device_array` which uses a bounded
`get(out=)` buffer that MAPLE's counters do not see. No full (L, N) W/J/I host mirror, no hot per-step transfer.

Memory (declared modelled payload, host and device separate, never summed): the composed `estimate_cuda_memory(None, child, cfg, n_selected=n_dirty)` (the reviewed E3b terms at the ACTUAL selected count, the real
Windowed deep-store mapping bound, the host provider tile path) plus this stage's own terms. Excluded and stated: the live current inputs (payload W/J/I/D, held maps, bed), interpreter/JIT/NVRTC/driver/pool rounding,
whole-process memory.

Limits: the held FC/SAT must be CELL layout (L, N) (a native-owner column; a raw-layout H0 context is refused typed before any array); if real texture changes wake flow, only explicit surface-return/export arrays are staged
(a later lifecycle needs its own recertification); every device behaviour here is unrun."""
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
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import native_floating_soil as nfs
from maple_syrup import native_floating_soil_cuda as ncuda
from maple_syrup import subsurface as ss

__all__ = ["EventRefreshCandidateCuda", "EventRefreshCudaError", "event_refresh_model_cuda", "stage0_model_cuda", "stage_completed_event_refresh_cuda", "verify_refresh_candidate_cuda"]

PROPERTY_BASIS = "boundary_selected"
EPOCH_KIND = "execution_epoch/v1"
CHUNK_BYTES = fcu.HASH_CHUNK_BYTES
_MINT = threading.local()
_CANDIDATES: weakref.WeakSet = weakref.WeakSet()


class EventRefreshCudaError(nfs.NativeFloatingError):
    """Typed refusal of the CUDA event-boundary refresh."""


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


@dataclass(frozen=True, eq=False)
class EventRefreshCandidateCuda:
    """Genuine staged CUDA refresh bound to the EXACT completion capability / state / event / payload / current child bed. Uncopyable, unpicklable, minted only by `stage_completed_event_refresh_cuda`; not an adopted parent.
    Device arrays are logically immutable (CuPy has no read-only flag); the host records are sealed."""

    capability: Any
    state: Any
    event: Any
    payload: Any
    child_bed: Any
    column: Any  # the NEW prepared root column (canonical content identity, current canonical D)
    water_state: Any  # fsl.FloatingSoilState on the new column (current W, changed columns reconciled), device
    matrix_op: Any
    history: Any  # J/I rebound bit for bit
    pref_op: Any
    ks_m_s: Any  # device (N,) or None
    depth_m: np.ndarray  # host sealed (N,) canonical depth record of the new column
    surface_return_m: Any  # device (N,) m, added to the surface depth by the adopter, once
    bottom_export_m: Any  # device (N,) m leaving through the open bottom, once
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
            raise EventRefreshCudaError("an EventRefreshCandidateCuda is minted only by stage_completed_event_refresh_cuda")
        _CANDIDATES.add(self)

    def __copy__(self):
        raise EventRefreshCudaError("an EventRefreshCandidateCuda cannot be copied")

    def __deepcopy__(self, memo):
        raise EventRefreshCudaError("an EventRefreshCandidateCuda cannot be copied")

    def __reduce_ex__(self, protocol):
        raise EventRefreshCudaError("an EventRefreshCandidateCuda is not serialisable")


def verify_refresh_candidate_cuda(cand: Any) -> EventRefreshCandidateCuda:
    """Genuine candidate whose completion is still valid and unconsumed (the producer is asked again); changes nothing."""
    if type(cand) is not EventRefreshCandidateCuda or cand not in _CANDIDATES:
        raise EventRefreshCudaError("not a genuine EventRefreshCandidateCuda")
    fcm.verify_completion(cand.capability, cand.state, event=cand.event)
    return cand


# ---------------------------------------------------------------------------------------------------------------------------------------------
# arguments, gate, models
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _quota(available_bytes_fn: Any, reserve_bytes: Any, host_available_bytes_fn: Any, host_reserve_bytes: Any, memory_budget_bytes: Any, host_memory_budget_bytes: Any) -> dict:
    for name, fn in (("available_bytes_fn", available_bytes_fn), ("host_available_bytes_fn", host_available_bytes_fn)):
        if fn is not None and not callable(fn):
            raise EventRefreshCudaError(f"{name} must be callable or None (the real default reader)")
    try:
        return {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": fsl._count(reserve_bytes, "reserve_bytes"), "available_bytes_fn": available_bytes_fn, "host_memory_budget_bytes": host_memory_budget_bytes,
                "host_reserve_bytes": fsl._count(host_reserve_bytes, "host_reserve_bytes"), "host_available_bytes_fn": host_available_bytes_fn}
    except fsl.FloatingSoilError as exc:
        raise EventRefreshCudaError(str(exc)) from None


def _component(q: dict) -> dict:
    """The SAME readers/budgets with ZERO reserves: the composed admission already counted each pool's reserve once."""
    return {**q, "reserve_bytes": 0, "host_reserve_bytes": 0}


def _gate(column: Any) -> tuple[Any, int]:
    try:
        cp = fcu._rc().require_cuda()
        return cp, fcu.gate_device(cp, column.device_id)
    except (ss.SubsurfaceError, RuntimeError) as exc:
        raise EventRefreshCudaError(f"the CUDA device gate refused: {exc}") from None


def stage0_model_cuda(n_cells: int, n_classes: int) -> tuple[int, int]:
    """`(device, host)` payload of the dirty-selection stage (reserves/allowance added by `fcu.admit`): device = the two `(N, nc)` bool comparison temporaries + the `(N,)` touched mask + the index vector (upper bound 8 N);
    host = the current-D record (8 N), the index vector upper bound (8 N) and one transfer chunk."""
    N, nc = int(n_cells), int(n_classes)
    return 2 * N * nc + N + 8 * N, 16 * N + CHUNK_BYTES


def event_refresh_model_cuda(child: Any, config: Any, *, n_selected: int, batch_cells: int, has_pref: bool, reserve_bytes: Any = 0, host_reserve_bytes: Any = 0) -> dict:
    """The composed stage-1 model: `ncuda.estimate_cuda_memory(None, child, config, n_selected=...)` (E3b terms at the ACTUAL selected count, the real Windowed mapping bound, the host provider tile path) plus this stage's
    own terms. Pure arithmetic from the actual native shapes; reserves are added once by the admission."""
    n = int(n_selected)
    base = ncuda.estimate_cuda_memory(None, child, config, batch_cells=batch_cells, reserve_bytes=reserve_bytes, host_reserve_bytes=host_reserve_bytes, n_selected=n)
    L = config.layers.n_layers
    N = int(child.active_layer.mass_kg.shape[0]) * int(child.active_layer.mass_kg.shape[1])
    extra_device = {"dirty_index": 8 * n}
    extra_host = {"current_depth_record": N * 8, "new_depth_copy": N * 8, "dirty_index": 8 * n, "transfer_chunk": CHUNK_BYTES, "changed_ks_index": 16 * n}
    if has_pref:
        from maple_syrup import floating_preferential_infiltration_cuda as fpc

        extra_host["pref_hash_staging"] = min(L * N, fpc.STAGE_ELEMS) * 8
    model = dict(base)
    model["device_terms"] = {**base["device_terms"], **extra_device}
    model["device_payload_bytes"] = base["device_payload_bytes"] + sum(extra_device.values())
    model["device_required_bytes"] = model["device_payload_bytes"] + int(reserve_bytes)
    model["host_payload_bytes"] = base["host_payload_bytes"] + sum(extra_host.values())
    model["host_required_bytes"] = model["host_payload_bytes"] + fcu.PY_ALLOWANCE_BYTES + int(host_reserve_bytes)
    model["host_extra_terms"] = extra_host
    model["n_selected"] = n
    return model


def _admit(cp: Any, dev: int, device_bytes: int, host_bytes: int, q: dict, what: str) -> None:
    fcu.admit(cp, dev, int(device_bytes), int(host_bytes), q, what)  # typed FloatingMemoryError propagates unchanged


# ---------------------------------------------------------------------------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------------------------------------------------------------------------
def stage_completed_event_refresh_cuda(cap: Any, state: Any, *, event: Any, available_bytes_fn: Any = None, reserve_bytes: Any = 0, host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0,
                                       memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None, batch_cells: int = nfs.DEFAULT_BATCH_CELLS) -> EventRefreshCandidateCuda:
    """Stage the selected property refresh of a genuinely completed CUDA event (module docstring). Pure: every refusal raises before/without mutating the event, its registries, the vertical or the capability, which
    stays unconsumed; repeated staging of the same pair is valid. Real default readers when the readers are None."""
    from maple_syrup import floating_checkpoint_cuda as fcc
    from maple_syrup import floating_pond as fp

    fcm.verify_completion(cap, state, event=event)
    vert = getattr(event, "vertical", None)
    if type(vert) is not fp.FloatingPondVertical:
        raise EventRefreshCudaError("the event does not run a genuine floating vertical (the bucket column has no floating inventory to refresh)")
    if vert.backend != "cuda" or getattr(event.xp, "__name__", None) != "cupy" or event.control.backend != "cuda":
        raise EventRefreshCudaError(f"the CUDA event-boundary refresh needs a CUDA vertical on a CuPy event (vertical {vert.backend!r}, hydraulic {event.control.backend!r}); CPU events use floating_event_refresh")
    q = _quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    comp = _component(q)
    B0 = fsl._count(batch_cells, "batch_cells", minimum=1)
    held0 = vert.held_template()
    for name in ("theta_field_capacity", "theta_saturation"):
        if str(held0.column.layouts[name]) != "cell":
            raise EventRefreshCudaError(f"the held {name} must have the CELL layout (L, N) for a property refresh (got {held0.column.layouts[name]!r}); nothing was read or allocated")
    cp, dev = _gate(held0.column)  # the CURRENT device FIRST: before any reader, hash, allocation or transfer
    bundle = vert.event_end_bundle(cap, state, event)
    payload, held, cfg = bundle.payload, bundle.held, vert.config
    ny, nx = event.shape
    N, nc = ny * nx, int(event.nc)
    active_host = np.asarray(vert._active_host()).reshape(-1)
    ctx = event.context
    child = nfs.NativeBed(state.bed.voxel_column, state.bed.active_layer, ctx.geometry, ctx.grain_classes, ctx.mass_resolution_kg, active_host.reshape(ny, nx))  # the vertical's sealed host mask BY REFERENCE
    nfs._check_bed(child, cfg)
    ncuda._check_devices(child, dev)
    led = state.sediment
    for name, a in (("pickup_cell", led.pickup_cell), ("deposition_cell", led.deposition_cell)):
        if tuple(a.shape[:2]) != (ny, nx) or int(a.shape[2]) != nc or int(a.device.id) != dev:
            raise EventRefreshCudaError(f"the producer's gross ledger {name} does not match the event's grid/classes/device")
    d0, h0 = stage0_model_cuda(N, nc)
    _admit(cp, dev, d0, h0, q, "the CUDA event-boundary refresh dirty selection (device temporaries and O(N) records)")
    touched = (cp.any(led.pickup_cell != 0, axis=2) | cp.any(led.deposition_cell != 0, axis=2)).reshape(-1) & held.column._maps["active"]
    idx_dev = cp.flatnonzero(touched).astype(np.int64)  # ONE intrinsic count read (declared)
    del touched
    dirty = fcc.d2h(idx_dev) if int(idx_dev.shape[0]) else np.zeros(0, dtype=np.int64)  # cold counted chunked index download (the compact reader needs host indices)
    del idx_dev
    d_cur = fcc.d2h(payload.depth_m)  # the CURRENT canonical D record (8 N, counted, chunked): never the stale held D0
    model = event_refresh_model_cuda(child, cfg, n_selected=int(dirty.size), batch_cells=B0, has_pref=payload.pref_op is not None, reserve_bytes=q["reserve_bytes"], host_reserve_bytes=q["host_reserve_bytes"])
    _admit(cp, dev, model["device_payload_bytes"], model["host_payload_bytes"], q, "the CUDA event-boundary selected property refresh (composed)")
    if child.representation == "windowed":
        nfs._windowed_checks(child, "stage_completed_event_refresh_cuda")
    B = int(model["batch_cells"])
    maps = held.column._maps
    new_fc, new_sat = cp.array(maps["theta_field_capacity"], copy=True), cp.array(maps["theta_saturation"], copy=True)  # device-to-device
    new_ks = None if held.ks_m_s is None else cp.array(held.ks_m_s, copy=True)
    depth = np.array(d_cur, copy=True)
    old = {"depth": d_cur, "fc": maps["theta_field_capacity"], "sat": maps["theta_saturation"], "ks": held.ks_m_s}
    changed, ks_changed, depth_changed = ncuda._scatter_tiles(cp, child, cfg, dirty, new_fc, new_sat, depth, new_ks, model=model, B=B, update_properties=True, old=old)
    area = nfs._area(child)
    t_s = float(payload.state.t_s)
    if changed.size:
        water, surf, exp_, acc = ncuda.reconcile_selected_columns(cp, cfg, payload.column, payload.state, d_cur, new_fc, new_sat, depth, changed, area, B, comp)
    else:  # nothing cascades (unchanged / above-FC / Ks-only / slab-equivalent depth): W is a device-to-device copy, bit for bit
        water = cp.array(payload.state.water_m, copy=True)
        surf, exp_ = cp.zeros(N, dtype=cp.float64), cp.zeros(N, dtype=cp.float64)
        acc = {"storage_before_m3": 0.0, "storage_after_m3": 0.0, "surface_return_m3": 0.0, "bottom_export_m3": 0.0, "residual_m3": 0.0, "bound_m3": 0.0, "moved_percolation_m3": 0.0, "n_changed_columns": 0, "n_batches": 0}
    depth_dev, mask_dev = ncuda._up(cp, depth), cp.array(maps["active"], copy=True)  # counted H2D of the 8 N depth record
    new_column = ncuda._device_column(cp, cfg, new_fc, new_sat, depth_dev, mask_dev, area, comp)  # the NEW prepared root: canonical content identity
    new_state = fsl.floating_continued_state(new_column, water, t_s, **comp)
    w_before, w_after, resid, bound = ncuda.inventory_closure_cuda(cp, cfg, payload.column, payload.state, new_column, new_state, water, changed, B, acc, comp, check_unrefreshed=bool(changed.size))
    new_op = ncuda._device_op(cfg, new_column, new_ks, comp)
    rb = fm.rebind_floating_matrix_history(payload.matrix_op, payload.history, new_op, **comp)
    pref = None if payload.pref_op is None else fpi.rebind_preferential_column(payload.pref_op, new_column, **comp)
    receipt = nfs.NativeReconcileReceipt(whole_before_m3=w_before, whole_after_m3=w_after, whole_residual_m3=resid, whole_bound_m3=bound, old_column_sha256=payload.column.sha256, new_column_sha256=new_column.sha256, **acc)
    for fresh in (depth, dirty, changed, depth_changed):
        fresh.flags.writeable = False
    f = cap.facts
    epoch = hashlib.sha256("|".join((EPOCH_KIND, f.config_sha256, f.forcing_sha256, str(f.topology_generation), str(f.n_accepted_steps), repr(f.t_s), new_column.sha256)).encode()).hexdigest()
    provenance = {"kind": EPOCH_KIND, "digest": epoch, "cold_content_hash": None, "note": "execution-epoch label of this event boundary; NOT a content hash of the bed"}
    details = {"required_device_bytes": model["device_required_bytes"], "required_host_bytes": model["host_required_bytes"], "batch_cells": B, "n_dirty": int(dirty.size), "n_changed": int(changed.size),
               "terms": model["device_terms"], "host_extra_terms": model["host_extra_terms"], "stage0_device_bytes": d0, "stage0_host_bytes": h0}
    with _minting():  # every refusable step has run
        return EventRefreshCandidateCuda(cap, state, event, payload, child, new_column, new_state, new_op, rb.history, pref, new_ks, depth, surf, exp_, receipt, dirty, changed, depth_changed, bool(ks_changed.size),
                                         PROPERTY_BASIS, provenance, details)
