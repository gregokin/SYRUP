"""Phase 2 E3b: the CUDA branch of the native floating-soil transaction (`native_floating_soil`, owner backend "cuda"). Written by its author WITHOUT a GPU: nothing here was executed on a device.

The transaction logic is the CPU one (same cold witness `_scan`, same descriptor, same canonical depth, same provider, same changed-column rule, same once-only adoption); only the placement differs:

* The accepted/proposed water `W`, the field-capacity/saturation maps and (texture_ks mode) the layer-0 Ks record stay DEVICE-RESIDENT arrays of the owner's device (private, logically immutable; CuPy has no
  read-only flag, so there is no public mutable duplicate authority and no whole-array host mirror). Host data are exactly: the sealed O(N) canonical depth record, the O(N) active-mask record, O(L) inputs
  uploaded once through MAPLE's counted `to_device`, and bounded B-sized TILES.
* Native device beds are read through MAPLE's counted `to_host` ONE selected tile at a time (`native_floating_soil._rows`): the cold O(N V K) witness/comparison is explicit, counted and NOT a per-step path.
  The E2 compact reader keeps its own mandatory native checks (Windowed: published journal, cheap checks, full validator) and its host canonical-depth witness; this module changes none of that.
* The accepted pure property provider runs on the HOST for each B-tile (an explicit bounded cold path: tile download, host provider, tile upload); the tile movement and the host terms are part of the
  composed admission. No `(L, N)` host property or water mirror exists. Properties are scattered into the resident maps with device index arrays.
* Reconciliation is the accepted D1 CUDA `reconcile_floating` over compact device sub-columns of ONLY the changed columns; D2 CUDA owns the matrix operator, rebind and history. The whole-inventory closure of
  the CUDA branch is the closure of the reconciled columns (untouched columns are unchanged by construction; a bounded chunked device check confirms it); no whole-`W` reduction or hash is performed.
* E2 backend: selected by the GENUINE native representation (`native_floating_soil._e2_backend`): a Dense CuPy bed uses the E2 CUDA compute; a Windowed bed (host or device window) is gathered and reduced on the
  host by MAPLE's own I/O ('numpy') and E2 places the compact outputs in the window's namespace; the E2 guards are untouched. The same selector feeds the composed estimate (backend, output namespace, device part).
* Receipt meaning: `whole_before/after` are the WHOLE active inventory on both backends (CUDA: two accepted D1 storage queries, a fixed block tree + one 8-byte read each); the selected-column storage fields keep
  their changed-columns-only meaning; closure is checked against the FP64 bound for the whole inventory. Cold bounded syncs (one counted 1-byte `to_host` per column chunk for the unrefreshed-column check, the
  storage reads, the validation/diagnostic reads) are declared and counted through MAPLE's helpers.
* Admission: the models publish `*_payload_bytes` (arrays) and `*_required_bytes` (payload + explicit reserve + the declared allowance, host); the shared admission is handed the payload and adds reserves/allowance
  once. `estimate_cuda_memory(None, bed, ...)` is the initialization model, `(parent, child, ...)` the refresh model. Reserves are strict non-negative integers.
* Order: the CURRENT device is gated first (before any availability reader, tile or transfer); metadata (namespace/device agreement of the bed arrays, packing, shapes) next; then the composed DEVICE and HOST
  admission with separate budgets/reserves/readings; only then the windowed validators, scan, tiles and allocations. A host bed with a CUDA owner (or the reverse), or a bed on another device, is a typed
  refusal: no implicit transfer and no backend fallback.
"""

from __future__ import annotations

import secrets
from typing import Any

import numpy as np

from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu

__all__ = ["estimate_cuda_memory", "initialize_cuda", "inspect_counted", "inventory_closure_cuda", "reconcile_selected_columns", "stage_cuda"]

_F8 = 8


def _nfs():
    from maple_syrup import native_floating_soil as nfs

    return nfs


def _cp():
    return fcu._rc().require_cuda()


def _quota(memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn) -> dict:
    return {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": reserve_bytes, "available_bytes_fn": available_bytes_fn, "host_memory_budget_bytes": host_memory_budget_bytes,
            "host_reserve_bytes": host_reserve_bytes, "host_available_bytes_fn": host_available_bytes_fn}


def _check_devices(bed: Any, dev: int) -> None:
    """Every native array of the bed must live on the gated current device (metadata only; no transfer)."""
    nfs = _nfs()
    arrays = [bed.active_layer.mass_kg]
    if bed.representation == "dense":
        arrays.append(bed.voxel_column.mass_kg)
    else:
        arrays += [bed.voxel_column.window_kg, bed.voxel_column.base, bed.voxel_column.deep_by_class_kg, bed.voxel_column.deep_total_kg, bed.voxel_column.deep_room_kg]
    for a in arrays:
        if int(a.device.id) != dev:
            raise nfs.NativeFloatingError(f"a native bed array lives on device {int(a.device.id)} but the current device is {dev}; no implicit transfer")


def estimate_cuda_memory(parent: Any, child: Any, config: Any, *, batch_cells: int, reserve_bytes: Any = 0, host_reserve_bytes: Any = 0, n_selected: int | None = None) -> dict:
    """Composed DEVICE and HOST requirements (metadata only), parent-aware: `parent=None` is the INITIALIZATION model (one bed), a parent bed makes it the REFRESH model (both actual shapes, as in C2).
    `device_payload_bytes` / `host_payload_bytes` are the modelled arrays only; `device_required_bytes` = payload + the explicit reserve and `host_required_bytes` = payload + the shared admission's declared
    interpreter allowance + the explicit host reserve: the figures the shared admission (`fcu.admit`, which adds reserve/allowance exactly ONCE to the payload) compares with the budgets/readings. Host and
    device sides are never summed into one budget. Reserves are strict non-negative integers (a bool is refused). `n_selected` (default None = the historical all-active model) is the SELECTED-refresh column count of the
    event-boundary refresh (`floating_event_refresh_cuda`): the compact-reader terms are modelled at exactly that many columns."""
    nfs = _nfs()
    dres, hres = fcu._count(reserve_bytes, "reserve_bytes"), fcu._count(host_reserve_bytes, "host_reserve_bytes")
    base = nfs._estimate(parent, child, config, batch_cells, 0, n_selected=n_selected)  # host phases from the ACTUAL shapes
    N, L, B = base["n_cells"], config.layers.n_layers, base["batch_cells"]
    lay = {"theta_residual": "scalar", "theta_field_capacity": "cell", "theta_saturation": "cell", "interface": "shared"}
    ret = fcu.retained_terms(L, N, lay)["total"]
    ret_b = fcu.retained_terms(L, B, lay)["total"]
    rec = fcu.operation_terms("reconcile", L, B)
    sto = fcu.operation_terms("storage", L, N)  # two whole-inventory storage queries (old, new): fixed block tree + one 8-byte read each, sequential
    device = {"new_maps": 2 * L * N * _F8, "ks_and_op_law": 2 * N * _F8, "depth_mask_upload": N * _F8 + N, "new_column_retained": ret, "new_column_cold_scratch": fcu.operation_terms("prepare", L, N)["device"],
              "water_scatter_and_state": 2 * L * N * _F8, "return_export": 2 * N * _F8, "sub_columns": 2 * ret_b, "reconcile_B": rec["device"] + rec["outputs"], "tiles": (4 * L + 8) * B * _F8,
              "history_rebind": 2 * N * _F8, "storage_queries": sto["device"], "compact_reader_device": base["native_device_bytes"]}
    t = base["terms"]
    host_tiles = (3 * L + 3) * B * _F8  # property tile, old-value tile, index/upload staging
    host_phase = max(t["scan"], t["parent_native"] - base["native_parent_device_bytes"], t["child_native_and_properties"] - base["native_child_device_bytes"])
    host = host_phase + N * _F8 * 2 + N + host_tiles + 3 * _F8 * base["n_active"] + rec["host"] + fcu.CHECK_HOST_BYTES + fcu.HASH_CHUNK_BYTES + 2 * sto["host"] + B
    return {"batch_cells": B, "device_terms": device, "device_payload_bytes": sum(device.values()), "device_required_bytes": sum(device.values()) + dres, "host_phases": dict(t),
            "host_payload_bytes": host, "host_required_bytes": host + fcu.PY_ALLOWANCE_BYTES + hres, "parent_voxels": base["parent_voxels"], "child_voxels": base["child_voxels"],
            "compact_bytes": base["compact_bytes"], "provider_bytes": base["provider_bytes"], "initialization": parent is None,
            "scope": "modelled payload; host and device sides separate; interpreter/JIT/NVRTC/context/pool rounding excluded; the host provider tile path, the cold counted transfers and the bounded chunk syncs are included"}


def _admit(cp: Any, dev: int, model: dict, quota: dict, what: str) -> None:
    """The shared admission is handed the PAYLOAD only: it adds the explicit reserve and the declared allowance exactly once, i.e. it compares the published `*_required_bytes` with the budgets/readings."""
    fcu.admit(cp, dev, model["device_payload_bytes"], model["host_payload_bytes"], quota, what)


def inspect_counted(array: Any, *, memory_budget_bytes: Any = None, available_bytes_fn: Any = None, host_reserve_bytes: Any = 0) -> np.ndarray:
    """Admitted, counted DETACHED host copy of one resident device result. The accepted authority stays on the device (its logical immutability is a cooperative contract; CuPy has no read-only flag) and the
    caller owns the returned host array: MAPLE's counted `to_host` of a device array already creates the fresh host copy, which is returned directly (the admission counts ONE array, never two). Order: current
    device gate, then the array must be a device array of THAT device (typed refusals before any availability reading or transfer), then admission, then the single counted transfer."""
    from maple.core.backend import is_device_array, to_host

    nfs = _nfs()
    cp = _cp()
    dev = fcu.gate_device(cp, None)
    if not is_device_array(array):
        raise nfs.NativeFloatingError("inspect_counted needs a device array (a host array is already host data and nothing is transferred)")
    if int(array.device.id) != dev:
        raise nfs.NativeFloatingError(f"the array lives on device {int(array.device.id)} but the current device is {dev}; no implicit transfer")
    fcu.admit(cp, dev, 0, int(array.nbytes), {"memory_budget_bytes": None, "reserve_bytes": 0, "available_bytes_fn": None, "host_memory_budget_bytes": memory_budget_bytes,
                                               "host_reserve_bytes": host_reserve_bytes, "host_available_bytes_fn": available_bytes_fn}, "native floating detached inspection")
    return to_host(array)


def _up(cp: Any, host: np.ndarray) -> Any:
    from maple.core.backend import to_device

    return to_device(np.ascontiguousarray(host), cp)  # MAPLE's counted host->device helper


def _device_column(cp, config, fc, sat, depth_dev, mask_dev, area, quota) -> Any:
    diff = _up(cp, config.diffusivity_m2_s) if config.layers.n_layers > 1 else 0.0
    return fsl.prepare_floating_column(config.nominal_edges_m, theta_residual=config.theta_residual, theta_field_capacity=fc, theta_saturation=sat, diffusivity_m2_s=diff,
                                       theta_evaporation_stress=config.theta_evaporation_stress, evaporation_factor=config.evaporation_factor, diffusion_mode="constant",
                                       bottom_drainage=config.bottom_drainage, soil_depth_m=depth_dev, active_mask=mask_dev, cell_area_m2=area, backend="cuda", **quota)


def _device_op(config, column, ks_dev, quota) -> Any:
    cond = ks_dev if config.matrix_conductivity_mode == "texture_ks" else config.matrix_conductivity_m_s
    return fm.prepare_floating_matrix(column, conductivity_m_s=cond, suction_m=config.matrix_suction_m, vegetation_factor=config.matrix_vegetation_factor, **quota)


def _scatter_tiles(cp, child, config, indices, fc, sat, depth, ks, *, model, B, update_properties, old=None) -> list:
    """Stream the compact batches of the selected cells; per tile: host provider properties, device scatter into the NEW resident maps, host depth record. With `old` (stage) also returns the cells whose depth,
    field capacity, saturation or Ks differ from the accepted ones, compared per tile against the old values gathered from the device (B-sized, counted)."""
    from maple.core.backend import to_host

    nfs = _nfs()
    changed, ks_changed, depth_changed = [], [], []
    if indices.size == 0:
        return [np.zeros(0, dtype=np.int64)] * 3
    it = nfs.csp.iter_compact_soil_profile(child.voxel_column, child.active_layer, child.geometry, child.grain_classes, child.mass_resolution_kg, config.layers, indices=indices,
                                           active_mask=child.active_mask, memory_budget_bytes=model["compact_bytes"], batch_cells=max(1, min(B, int(indices.size))), backend=nfs._e2_backend(child),
                                           coverage="partial_over_base")
    K = int(child.active_layer.mass_kg.shape[2])
    for batch in it:
        idx = batch.indices
        idx_dev = _up(cp, idx)
        new_depth = batch.canonical_depth_m
        f = s = k0 = None
        if update_properties:
            f, s, k0 = nfs._batch_properties(batch, config, K, model["provider_bytes"])
        if old is not None:
            od = old["depth"][idx]
            ch = nfs._geometry_changed(config, od, new_depth)  # ONLY a change of the porous slabs / bottom condition (or FC-SAT below) may reconcile
            depth_changed.append(idx[new_depth != od])  # the actual canonical depth is always published
            if f is not None:
                ch |= np.any(f != to_host(old["fc"][:, idx_dev]), axis=0) | np.any(s != to_host(old["sat"][:, idx_dev]), axis=0)
            changed.append(idx[ch])
            if k0 is not None and old["ks"] is not None:
                ks_changed.append(idx[k0 != to_host(old["ks"][idx_dev])])
        depth[idx] = new_depth
        if f is not None:
            fc[:, idx_dev] = _up(cp, f)
            sat[:, idx_dev] = _up(cp, s)
            if ks is not None:
                ks[idx_dev] = _up(cp, k0)
        del batch
    cat = lambda parts: np.concatenate(parts).astype(np.int64, copy=False) if parts else np.zeros(0, dtype=np.int64)
    return [cat(changed), cat(ks_changed), cat(depth_changed)]


def initialize_cuda(bed, config, *, initial_relative_saturation, quota, batch_cells) -> Any:
    """CUDA initialization (module docstring). Gate first, metadata, composed admission, then the cold scan/tiles/allocations."""
    nfs = _nfs()
    cp = _cp()
    dev = fcu.gate_device(cp, None)
    N, _V, _K, _C = nfs._check_bed(bed, config)
    _check_devices(bed, dev)
    r = float(initial_relative_saturation)
    if not (0.0 <= r < 1.0):
        raise nfs.NativeFloatingError("initial_relative_saturation must lie in [0, 1) (strictly unsaturated initialization)")
    model = estimate_cuda_memory(None, bed, config, batch_cells=batch_cells, reserve_bytes=quota["reserve_bytes"], host_reserve_bytes=quota["host_reserve_bytes"])
    _admit(cp, dev, model, quota, "native floating CUDA initialization")
    if bed.representation == "windowed":
        nfs._windowed_checks(bed, "initialize_native_floating_soil")
    active_flat = nfs._flat_active(bed)
    B = model["batch_cells"]
    _, witness, _ = nfs._scan(None, bed, active_flat, B)
    L = config.layers.n_layers
    lat_fc, lat_sat = _up(cp, config.latent_theta_field_capacity), _up(cp, config.latent_theta_saturation)
    fc, sat = cp.empty((L, N), dtype=cp.float64), cp.empty((L, N), dtype=cp.float64)
    fc[:] = lat_fc[:, None]
    sat[:] = lat_sat[:, None]
    ks = cp.full(N, config.latent_saturated_conductivity_m_s, dtype=cp.float64) if config.matrix_conductivity_mode == "texture_ks" else None
    depth = np.zeros(N)
    _scatter_tiles(cp, bed, config, active_flat, fc, sat, depth, ks, model=model, B=B, update_properties=True)
    mask_host = bed.active_mask.reshape(-1)
    depth_dev, mask_dev = _up(cp, depth), _up(cp, mask_host)
    column = _device_column(cp, config, fc, sat, depth_dev, mask_dev, nfs._area(bed), quota)
    e = column._maps["edges"]
    water = cp.empty((L, N), dtype=cp.float64)
    for k in range(L):
        h = cp.maximum(cp.minimum(depth_dev, e[k + 1]) - e[k], 0.0)  # the D1 porous thickness, evaluated on the device
        water[k] = (r * sat[k]) * h
    water[:, ~mask_dev] = 0.0
    state = fsl.floating_initial_state(column, water, 0.0, **quota)
    op = _device_op(config, column, ks, quota)
    hist = fm.floating_matrix_history(op, wetting_m=config.initial_wetting_m, cumulative_intake_m=config.initial_cumulative_intake_m, **quota)
    depth.flags.writeable = False
    authority = nfs._Authority()
    with nfs._minting():
        binding = nfs.NativeBinding(secrets.token_hex(8), 0, witness, secrets.token_hex(16))
        owner = nfs.NativeFloatingSoil(config, column, state, op, hist, depth, ks, binding, (), nfs.MappingProxyType({"refreshes": 0, "cells_refreshed": 0, "surface_return_m3": 0.0, "bottom_export_m3": 0.0}),
                                       authority)
    authority.current = binding
    return owner


def reconcile_selected_columns(cp, config, old_column, old_state, old_depth_host, new_fc, new_sat, new_depth, changed, area, B, quota) -> tuple:
    """OWNER-FREE core of the D1 CUDA `reconcile_floating` over compact device sub-columns of ONLY the changed columns (B at a time), scattered into whole resident arrays. The OLD column/state/depth are EXPLICIT
    (`old_column` its device maps, `old_state` the current W `(L, N)` and clock, `old_depth_host` the host `(N,)` canonical depth the old column corresponds to): a native owner passes its own (`_reconcile_cuda`), the
    event-boundary refresh passes the CURRENT payload (never a stale held D0, never a fake owner). The arithmetic and the order of operations are the historical ones."""
    N = old_column.n_cells
    water = cp.array(old_state.water_m, copy=True)  # device-to-device
    surf, exp_ = cp.zeros(N, dtype=cp.float64), cp.zeros(N, dtype=cp.float64)
    maps = old_column._maps
    acc = {"storage_before_m3": 0.0, "storage_after_m3": 0.0, "surface_return_m3": 0.0, "bottom_export_m3": 0.0, "residual_m3": 0.0, "bound_m3": 0.0, "moved_percolation_m3": 0.0}
    nb = 0
    for b0 in range(0, changed.size, B):
        idx = changed[b0:b0 + B]
        idx_dev = _up(cp, idx)
        ones = cp.ones(idx.size, dtype=cp.bool_)
        old_sub = _device_column(cp, config, cp.ascontiguousarray(maps["theta_field_capacity"][:, idx_dev]), cp.ascontiguousarray(maps["theta_saturation"][:, idx_dev]), _up(cp, old_depth_host[idx]), ones, area,
                                 quota)
        new_sub = _device_column(cp, config, cp.ascontiguousarray(new_fc[:, idx_dev]), cp.ascontiguousarray(new_sat[:, idx_dev]), _up(cp, new_depth[idx]), ones, area, quota)
        sub_state = fsl.floating_continued_state(old_sub, cp.ascontiguousarray(old_state.water_m[:, idx_dev]), float(old_state.t_s), **quota)
        cand = fsl.reconcile_floating(old_sub, sub_state, new_sub, **quota)
        water[:, idx_dev] = cand.state.water_m
        surf[idx_dev] = cand.surface_return_m
        exp_[idx_dev] = cand.bottom_export_m
        r = cand.receipt
        for key, val in (("storage_before_m3", r.storage_before_m3), ("storage_after_m3", r.storage_after_m3), ("surface_return_m3", r.surface_return_m3), ("bottom_export_m3", r.bottom_export_m3),
                         ("residual_m3", r.residual_m3), ("bound_m3", r.bound_m3), ("moved_percolation_m3", r.moved_percolation_m3)):
            acc[key] += float(val)
        nb += 1
        del old_sub, new_sub, sub_state, cand
    acc.update(n_changed_columns=int(changed.size), n_batches=nb)
    return water, surf, exp_, acc


def _reconcile_cuda(cp, config, owner, new_fc, new_sat, new_depth, changed, area, B, quota) -> tuple:
    """The native owner's reconciliation: `reconcile_selected_columns` with the owner's own column, state and canonical depth record (behaviour unchanged)."""
    return reconcile_selected_columns(cp, config, owner.column, owner.state, owner.depth_m, new_fc, new_sat, new_depth, changed, area, B, quota)


def inventory_closure_cuda(cp, config, old_column, old_state, new_column, new_state, water, changed, B, acc, quota, *, check_unrefreshed=True) -> tuple[float, float, float, float]:
    """The whole-inventory closure of one changed-column CUDA refresh, factored UNCHANGED from `stage_cuda` (shared with the event-boundary refresh): a bounded chunked device check that unrefreshed columns kept their water
    (ONE counted 1-byte read per chunk, a declared cold sync; skippable only when nothing was reconciled), the WHOLE active inventory before/after through the accepted D1 storage query (block tree + one 8-byte read each)
    and the FP64 residual bound. Returns `(w_before_m3, w_after_m3, residual_m3, bound_m3)` or raises `NativeFloatingError`."""
    from maple.core.backend import to_host

    nfs = _nfs()
    N, L = old_column.n_cells, config.layers.n_layers
    if check_unrefreshed:
        is_changed = np.zeros(N, dtype=bool)
        is_changed[changed] = True
        for c0 in range(0, N, B):
            sl = slice(c0, min(c0 + B, N))
            keep = _up(cp, ~is_changed[sl])
            same = to_host(cp.array_equal(water[:, sl][:, keep], old_state.water_m[:, sl][:, keep]))  # ONE counted 1-byte read per bounded chunk (a declared cold sync)
            if not bool(same):
                raise nfs.NativeFloatingError("internal: an unrefreshed column's water changed")
    w_before = fsl.floating_storage_m3(old_column, old_state, **quota)
    w_after = fsl.floating_storage_m3(new_column, new_state, **quota)
    resid = (w_after - w_before) + (acc["surface_return_m3"] + acc["bottom_export_m3"])
    bound = nfs.volume_roundoff_bound_m3(4 * L * N + 4, max(abs(w_before), abs(w_after), abs(acc["surface_return_m3"]), abs(acc["bottom_export_m3"])))
    if not (np.isfinite(resid) and abs(resid) <= bound):
        raise nfs.NativeFloatingError(f"whole-inventory closure violated: residual {resid!r} m3 exceeds the FP64 bound {bound!r} m3; nothing is staged")
    return w_before, w_after, resid, bound


def stage_cuda(owner, parent_bed, child_bed, *, mode, quota, batch_cells) -> Any:
    """CUDA staging (module docstring): the same checks, proofs and outcomes as the CPU `stage_native_refresh`."""
    from maple.core.backend import to_host

    nfs = _nfs()
    cp = _cp()
    dev = fcu.gate_device(cp, owner.column.device_id)  # the owner's device must be the CURRENT device, before anything else
    cfg = owner.config
    for bed in (parent_bed, child_bed):
        N, _V, _K, _C = nfs._check_bed(bed, cfg)
        _check_devices(bed, dev)
    if (parent_bed.representation, parent_bed.geometry.ny, parent_bed.geometry.nx) != (child_bed.representation, child_bed.geometry.ny, child_bed.geometry.nx) \
            or not np.array_equal(parent_bed.active_mask, child_bed.active_mask) or N != owner.column.n_cells:
        raise nfs.NativeFloatingError("parent and child beds must share the representation, grid and the owner's active mask")
    model = estimate_cuda_memory(parent_bed, child_bed, cfg, batch_cells=batch_cells, reserve_bytes=quota["reserve_bytes"], host_reserve_bytes=quota["host_reserve_bytes"])
    _admit(cp, dev, model, quota, "native floating CUDA refresh")
    if parent_bed.representation == "windowed":
        nfs._windowed_checks(parent_bed, "stage_native_refresh(parent)")
        nfs._windowed_checks(child_bed, "stage_native_refresh(child)")
    # the ACTIVE MASK of the staged beds must equal the owner's (the CPU owner-mask check, with the same refusal): bounded O(N) upload of the host mask record (counted) and ONE counted 1-byte read, after admission
    # and before any native row, property or water work
    mask_host = np.ascontiguousarray(child_bed.active_mask.reshape(-1))
    if not bool(to_host(cp.array_equal(_up(cp, mask_host), owner.column._maps["active"]))):
        raise nfs.NativeFloatingError("parent and child beds must share the representation, grid and the owner's active mask")
    B = model["batch_cells"]
    active_flat = nfs._flat_active(child_bed)
    dirty, child_witness, parent_witness = nfs._scan(parent_bed, child_bed, active_flat, B)
    if parent_witness != owner.binding.source_witness:
        raise nfs.NativeStaleError("the parent bed is not the native source this owner was accepted against (cold witness mismatch, including the physical descriptor); the dirty set would not be proven")
    maps = owner.column._maps
    new_fc, new_sat = cp.array(maps["theta_field_capacity"], copy=True), cp.array(maps["theta_saturation"], copy=True)
    new_ks = None if owner.ks_m_s is None else cp.array(owner.ks_m_s, copy=True)
    depth = np.array(owner.depth_m, copy=True)
    props = mode == "properties"
    old = {"depth": owner.depth_m, "fc": maps["theta_field_capacity"], "sat": maps["theta_saturation"], "ks": owner.ks_m_s}
    changed, ks_changed, depth_changed = _scatter_tiles(cp, child_bed, cfg, dirty, new_fc, new_sat, depth, new_ks if props else None, model=model, B=B, update_properties=props, old=old)
    request = nfs.NativeRefreshRequest(owner.binding.lineage, owner.binding.generation, owner.binding.source_witness, child_witness, mode, int(B), int(model["host_required_bytes"]) + int(model["device_required_bytes"]),
                                       nfs._descriptor(child_bed), int(dirty.size), int(changed.size), int(depth_changed.size))
    dirty.flags.writeable = False
    changed.flags.writeable = False
    depth_changed.flags.writeable = False
    area = nfs._area(child_bed)
    if changed.size == 0:
        zeros = cp.zeros(N, dtype=cp.float64)
        op, hist, ks_out = owner.matrix_op, owner.matrix_history, owner.ks_m_s
        column, state, depth_out = owner.column, owner.state, owner.depth_m
        if depth_changed.size:
            # hydraulically equivalent geometry: publish the ACTUAL canonical depth in a genuine new column/state/operator; the water is a device-to-device copy of the accepted inventory (same bits, same time,
            # no export/return, no cascade, no physics step); the accepted parent stays untouched until adoption
            depth.flags.writeable = False
            column = _device_column(cp, cfg, new_fc, new_sat, _up(cp, depth), cp.array(owner.column._maps["active"], copy=True), area, quota)
            state = fsl.floating_continued_state(column, cp.array(owner.state.water_m, copy=True), float(owner.state.t_s), **quota)
            depth_out = depth
        if depth_changed.size or ks_changed.size:
            op = _device_op(cfg, column, new_ks, quota)
            hist = fm.rebind_floating_matrix_history(owner.matrix_op, owner.matrix_history, op, **quota).history
            ks_out = new_ks if ks_changed.size else owner.ks_m_s
        with nfs._minting():
            return nfs.NativeRefreshCandidate(owner.binding, child_witness, mode, dirty, changed, depth_changed, column, state, op, hist, depth_out, ks_out, zeros, zeros, None, False, model, request,
                                              owner._authority, [False])
    water, surf, exp_, acc = _reconcile_cuda(cp, cfg, owner, new_fc, new_sat, depth, changed, area, B, quota)
    depth_dev, mask_dev = _up(cp, depth), cp.array(owner.column._maps["active"], copy=True)
    new_column = _device_column(cp, cfg, new_fc, new_sat, depth_dev, mask_dev, area, quota)
    new_state = fsl.floating_continued_state(new_column, water, float(owner.state.t_s), **quota)
    w_before, w_after, resid, bound = inventory_closure_cuda(cp, cfg, owner.column, owner.state, new_column, new_state, water, changed, B, acc, quota)
    new_op = _device_op(cfg, new_column, new_ks, quota)
    rb = fm.rebind_floating_matrix_history(owner.matrix_op, owner.matrix_history, new_op, **quota)
    receipt = nfs.NativeReconcileReceipt(whole_before_m3=w_before, whole_after_m3=w_after, whole_residual_m3=resid, whole_bound_m3=bound,
                                         old_column_sha256=owner.column.sha256, new_column_sha256=new_column.sha256, **acc)
    depth.flags.writeable = False
    with nfs._minting():
        return nfs.NativeRefreshCandidate(owner.binding, child_witness, mode, dirty, changed, depth_changed, new_column, new_state, new_op, rb.history, depth, new_ks, surf, exp_, receipt, True, model, request,
                                          owner._authority, [False])
