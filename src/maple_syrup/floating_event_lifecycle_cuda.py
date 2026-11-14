"""EB3-CUDA: the CUDA branches of the ONE floating event lifecycle (`floating_event_lifecycle`): open -> native occurrence -> genuine completion -> selected CURRENT-W refresh -> accepted boundary -> next native occurrence.
AUTHOR-WRITTEN, NOT RUN ON A DEVICE (Root qualifies it on hardware). Importing this module never imports CuPy.

SAME authority. There is NO second lifecycle class: the genuine `FloatingEventLineage` / `Occurrence` / `Boundary` / `BoundaryRoot` of `floating_event_lifecycle` are minted here through its private window (lazy import: the
lifecycle imports this module only inside its CUDA branches). The CPU bodies are untouched. This module only holds the device-specific numerics of three steps:
  open_cuda   the CPU opener with (a) the CURRENT-device gate before any reader/hash/array, (b) the scalar storage from the actual resident reductions (`floating_storage_m3` packet + ONE counted 8-byte read of the surface
              sum; never a download of W/J/I/H), (c) the owner retired LAST after every refusable step (identical order);
  close_cuda  the CPU closer over the GPU selected refresh candidate (`floating_event_refresh_cuda`): CURRENT device W/J/I/D, the exact capacity SURFACE RETURN (device row) added to H' ONCE, the wake decided from the
              candidate's host receipt scalar (no device read), the cumulative water closure from counted scalars (two 8-byte reads), the boundary root carrying the NEW root column's DEVICE canonical depth (the CURRENT D, never the
              held D0) plus the vertical's SEALED host mask / edges and the candidate's sealed host depth record (explicit O(N) metadata, by reference, from the genuine producer);
  the boundary constructor `FloatingPondVertical.from_boundary` (CUDA branch -> `floating_pond_cuda.init_cuda_boundary`: the water-free setup, no fake HeldContext/owner).
The cumulative expected storage subtracts the OPTIONAL daily keys `evaporation_m3` / `transpiration_m3` (`.get(.., 0.0)`), so a later daily overlay on the same dict keeps its terms.

Dependency (unaccepted): the selected GPU refresh (`floating_event_refresh_cuda`) is consumed only through its sealed candidate; this module never recomputes or overrides its changed set. ADOPTION of this lifecycle must depend on an
independently accepted, compatible refresh. Nothing here is a claim for the refresh, a GPU run or a large catchment.

Memory (declared payload models, two pools, reserves once by `fcu.admit`, 64 KiB host allowance): open = the vertical's own construction model (by the vertical) + the storage packet terms; close = the refresh's own two admissions + this
step's H' row (8 N device) + the storage packet terms + `HOST_OP_BYTES`; begin = `init_cuda_boundary` (two device zero rows + the vertical's stage models; the event factory has its own quota and is excluded). Counted cold
O(N) transfers declared: open 2 scalar reads; close 2 scalar reads; begin: `PondEvent.continued_state` compares the boundary depth with the event's (two counted O(N) D2H of the surface depth). No (L, N) host mirror, no
whole-bed scan, no per-step transfer. Excluded: interpreter/JIT/driver/allocator pool and whole-process memory.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import numpy as np

from maple_syrup import floating_completion as fcm
from maple_syrup import floating_event_refresh_cuda as erc
from maple_syrup import floating_pond as fp
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import native_floating_soil as nfs
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = ["arrays_equal_cuda", "close_cuda", "open_cuda", "storage_model_cuda", "stored_m3_cuda", "unit_rain_scale_cuda"]

HOST_OP_BYTES = 4096


def _lc():
    from maple_syrup import floating_event_lifecycle as lc

    return lc


def quota_of(available_bytes_fn: Any, reserve_bytes: Any, host_available_bytes_fn: Any, host_reserve_bytes: Any, memory_budget_bytes: Any, host_memory_budget_bytes: Any) -> dict:
    lc = _lc()
    for name, fn in (("available_bytes_fn", available_bytes_fn), ("host_available_bytes_fn", host_available_bytes_fn)):
        if fn is not None and not callable(fn):
            raise lc.LifecycleError(f"{name} must be callable or None (the real default reader)")
    try:
        return {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": fsl._count(reserve_bytes, "reserve_bytes"), "available_bytes_fn": available_bytes_fn, "host_memory_budget_bytes": host_memory_budget_bytes,
                "host_reserve_bytes": fsl._count(host_reserve_bytes, "host_reserve_bytes"), "host_available_bytes_fn": host_available_bytes_fn}
    except fsl.FloatingSoilError as exc:
        raise lc.LifecycleError(str(exc)) from None


def _component(q: dict) -> dict:
    """The SAME readers/budgets with ZERO reserves: the composed admission already counted each pool's reserve once."""
    return {**q, "reserve_bytes": 0, "host_reserve_bytes": 0}


def _gate(column: Any) -> tuple[Any, int]:
    lc = _lc()
    try:
        cp = fcu._rc().require_cuda()
        return cp, fcu.gate_device(cp, column.device_id)
    except (fsl.FloatingSoilError, ss.SubsurfaceError, RuntimeError) as exc:
        raise lc.LifecycleError(f"the CUDA device gate refused: {exc}") from None


def storage_model_cuda(n_layers: int, n_cells: int, *, extra_device: int = 0) -> tuple[int, int]:
    """`(device, host)` payload of one scalar storage evaluation: the soil storage packet (`operation_terms("storage")`: partials + diag), the surface-sum partial (8 N upper bound) and the scalar reads' host side."""
    t = fcu.operation_terms("storage", int(n_layers), int(n_cells))
    return int(t["outputs"]) + int(t["scratch"]) + 8 * int(n_cells) + int(extra_device), HOST_OP_BYTES


def unit_rain_scale_cuda(event: Any) -> bool:
    """The unit `rain_scale` contract of a bound full storm on the DEVICE: every ACTIVE cell's scale is exactly 1, from the event's resident `rain_scale` and the vertical's device-prepared active mask through ONE counted scalar read
    (no full-N map download). The current device is gated first."""
    vert = event.vertical
    cp, _dev = _gate(vert.held_template().column)
    act = vert.held_template().column._maps["active"]
    scale = cp.asarray(event.rain_scale).reshape(-1)
    return bool(_scalar(cp.all(scale[act] == 1.0)))


def wet_runtime_plan_cuda(step_bytes: int, host_op_bytes: int, n_cells: int, q_elems: int) -> tuple[int, int]:
    """Pure `(device, host)` payload of the CUDA wet-day runtime admission (module `day_runtime`), per pool, reserves excluded (`fcu.admit` adds each once, plus the 64 KiB host allowance once):
    DEVICE = the vertical's actual per-step model (`step_bytes`: preferential, matrix, FSL, router stage buffers) + the evaporation staging's DEVICE compute (`estimate_floating_evaporation_memory(N, "cuda").device_bytes`) +
    the six `N` working rows and the carried face copies (`q_elems` doubles);
    HOST = the vertical's own small hot-operation host side (`host_op_bytes`) + the evaporation staging's host PACKET only (its `host_bytes` less its own interpreter allowance, which the admission counts once).
    The nested stages keep their own per-stage guards (these can require more than this composed need; the larger governs and is never relabelled as exact)."""
    from maple_syrup import floating_evaporation as fe

    n = int(n_cells)
    est = fe.estimate_floating_evaporation_memory(n, backend="cuda")
    device = int(step_bytes) + int(est.device_bytes) + 6 * n * 8 + int(q_elems) * 8
    host = int(host_op_bytes) + max(int(est.host_bytes) - int(fe.INTERPRETER_ALLOWANCE_BYTES), 0)
    return device, host


def stored_m3_cuda(column: Any, water: Any, depth: Any, area: float, q: Any) -> float:
    """Soil + surface storage (m3) of a CUDA hold's closed day at the hand-back: the resident storage packet and ONE counted scalar read of the surface sum, admitted on the caller's two pools BEFORE any kernel (gate first); `q` is
    the strict two-pool quota (`floating_daily_cuda.quota`) or None for the real default readers."""
    cp, dev = _gate(column)
    from maple_syrup import floating_daily_cuda as fdc

    quota = q if q is not None else fdc.quota(None, 0, None, 0)
    d, h = storage_model_cuda(column.n_layers, column.n_cells)
    fcu.admit(cp, dev, d, h, quota, "the CUDA event<->daily hand-back storage (resident packet, one counted scalar)")
    return float(fsl.floating_storage_m3(column, water, **_component(quota))) + _scalar(cp.sum(depth)) * float(area)


def arrays_equal_cuda(column: Any, a: Any, b: Any) -> bool:
    """Exact resident equality of two device arrays of the column's device: ONE counted scalar read (the current device gated first)."""
    cp, _dev = _gate(column)
    return bool(_scalar(cp.array_equal(a, b)))


def _scalar(x: Any) -> float:
    """ONE counted 8-byte device-to-host read of a 0-d device value."""
    from maple.core.backend import to_host

    return float(np.asarray(to_host(x)))


def _storage_m3(vert: Any, state: Any, depth: Any, q: dict, cp: Any, dev: int, what: str) -> float:
    """Current native storage (soil + surface) in m3 from resident reductions and counted scalar reads; admitted (two pools) BEFORE any kernel; never `np.asarray`/`.get()` of water."""
    payload = vert.current_payload(state.hydraulic.soil_water_m)
    d, h = storage_model_cuda(vert.n_layers, vert.n_cells)
    fcu.admit(cp, dev, d, h, q, what)
    soil = float(fsl.floating_storage_m3(payload.column, payload.state, **_component(q)))
    area = float(state.topology.dx_m) ** 2
    return soil + _scalar(cp.sum(depth)) * area


def open_cuda(owner: Any, bed: Any, event_factory: Any, *, available_bytes_fn: Any, preferential: Any, reserve_bytes: int, batch_cells: int, host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0,
              memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None) -> tuple:
    """CUDA branch of `open_floating_event` (same order as the CPU opener, module docstring). The caller has already verified the genuine CURRENT owner and the callables."""
    lc = _lc()
    cp, dev = _gate(owner.column)  # the CURRENT device FIRST: before any reader, vertical, hash or array
    q = quota_of(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=available_bytes_fn, preferential=preferential, reserve_bytes=reserve_bytes, batch_cells=batch_cells, memory_budget_bytes=memory_budget_bytes,
                                   host_memory_budget_bytes=host_memory_budget_bytes, host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn)
    event = event_factory(vert)  # the caller's static factory, outside every mint window
    lc._admit_native(event, vert)
    t0 = float(owner.state.t_s)
    state = event.initial_state(bed, vert.initial_soil_water(), t_s=t0)  # the actual full validation, owner still current
    lc._admit_state(event, vert, state, bed, t0)
    storage0 = _storage_m3(vert, state, state.hydraulic.depth_m, q, cp, dev, "the CUDA lifecycle open storage (resident reductions, two counted scalar reads)")
    ledger = {"storage0_m3": storage0, "rain_m3": 0.0, "drainage_m3": 0.0, "outlet_export_m3": 0.0, "bottom_export_m3": 0.0, "surface_return_m3": 0.0, "bound_m3": 0.0, "n_refreshes": 0, "n_occurrences": 1}
    lineage = lc.FloatingEventLineage()
    vert.detach_initial()  # on the PRIVATE new vertical (a failure discards it; the owner is untouched)
    prov = {"cold_anchor_witness": vert.held_template().source_witness, "property_basis": "initial_full", "execution_epoch": None, "cold_content_hash": None, "backend": "cuda"}
    with lc._minting():
        occ = lc.Occurrence(event, vert, t0, ledger, prov, 0)
    if not owner.is_current:
        raise lc.LifecycleError("the owner was superseded during the open")
    nfs.retire_native_owner_to_lineage(owner)  # the ONLY owner mutation, LAST
    lineage._head = occ
    return lineage, occ, state


def close_cuda(lineage: Any, occurrence: Any, cap: Any, state: Any, *, available_bytes_fn: Any, batch_cells: int, reserve_bytes: int, host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0,
               memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None) -> Any:
    """CUDA branch of `close_occurrence` (same order as the CPU closer; the caller has verified the expected head is this occurrence)."""
    lc = _lc()
    event = occurrence.event
    vert = occurrence.vertical
    cp, dev = _gate(vert.held_template().column)  # the CURRENT device FIRST
    q = quota_of(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    fcm.verify_completion(cap, state, event=event)
    cand = erc.stage_completed_event_refresh_cuda(cap, state, event=event, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, host_available_bytes_fn=host_available_bytes_fn,
                                                  host_reserve_bytes=host_reserve_bytes, memory_budget_bytes=memory_budget_bytes, host_memory_budget_bytes=host_memory_budget_bytes, batch_cells=batch_cells)
    audit = event.audit(state)
    if not (audit["water"]["closes"] and audit["sediment"]["closes"]):
        raise lc.LifecycleError("the completed event's water/sediment audits do not close; the boundary is refused")
    storm_match = lc._pending_storm_proof(occurrence, event, state, audit)  # the SAME backend-neutral pending-storm proof as the CPU close (producer forcing-history conformity for ANY pending cursor, native end, unit scale, amount)
    held = vert.held_template()
    ny, nx = event.shape
    n = ny * nx
    wake = float(cand.receipt.surface_return_m3) > 0.0  # the receipt's host scalar: the returned volume is non-negative, so this is exactly `sum(return) > 0` without a device read
    d_h, h_h = storage_model_cuda(vert.n_layers, vert.n_cells, extra_device=8 * n if wake else 0)
    fcu.admit(cp, dev, d_h, h_h, q, "the CUDA lifecycle close (H' row, storage packets and scalar reads)")
    h_prime = state.hydraulic.depth_m + cand.surface_return_m.reshape(ny, nx) if wake else state.hydraulic.depth_m  # the capacity return joins the ACTUAL surface ONCE
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
    stored = float(cand.receipt.whole_after_m3) + _scalar(cp.sum(h_prime)) * float(state.topology.dx_m) ** 2
    ev, tr = float(led.get("evaporation_m3", 0.0)), float(led.get("transpiration_m3", 0.0))  # optional daily terms (a later overlay): carried, never invented
    expected = led["storage0_m3"] + led["rain_m3"] - led["drainage_m3"] - led["outlet_export_m3"] - led["bottom_export_m3"] - ev - tr
    resid = stored - expected
    bound = led["bound_m3"] + volume_roundoff_bound_m3(12, max(abs(stored), abs(expected), led["rain_m3"], led["drainage_m3"], led["outlet_export_m3"], led["bottom_export_m3"], ev, tr, 1e-30))
    if not (np.isfinite(resid) and abs(resid) <= bound):
        raise lc.LifecycleError(f"the cumulative water balance across the lifecycle does not close: residual {resid!r} m3 exceeds the MAPLE volume bound {bound!r} m3; the boundary is refused")
    led["residual_m3"], led["residual_bound_m3"] = float(resid), float(bound)
    prev_epoch = occurrence.provenance.get("execution_epoch") or ""
    epoch = hashlib.sha256(f"{erc.EPOCH_KIND}|{prev_epoch}|{cand.provenance['digest']}".encode()).hexdigest()
    from maple_syrup import floating_held_context_cuda as fhc

    prov = {"cold_anchor_witness": held.source_witness, "property_basis": erc.PROPERTY_BASIS, "execution_epoch": epoch, "cold_content_hash": None, "area_m2": area, "backend": "cuda", "kernel": fhc.kernel_descriptor()}
    role = "pending_recession" if wake else "boundary"
    d_dev = cand.column._maps["soil_depth"]  # the NEW root's CURRENT canonical depth (device), never the held D0
    with lc._minting():
        root = lc.BoundaryRoot(cand.column, cand.matrix_op, cand.pref_op, cand.ks_m_s, d_dev, d_dev, cand.history, cand.water_state, vert.config, int(held.generation) + 1, held.source_witness, bed,
                               vert._active_host(), vert._edges_host, cand.depth_m, vegetation=occurrence.vegetation)
        boundary = lc.Boundary(role, root, bed, h_prime, state.hydraulic.qx_m2_s, state.hydraulic.qy_m2_s, (state.topology, state.held_source_m, state.datum), float(state.t_s), led, prov, occurrence.seq + 1, occurrence.cursor,
                               None if storm_match is not None else occurrence.pending_forcing, storm_match)
    fcm.consume_completion(cap, state, event=event)  # the completion is consumed LAST ...
    lineage._head = boundary  # ... and then the head
    return boundary
