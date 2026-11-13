"""E5: CUDA helpers of the floating Pond vertical (`maple_syrup.floating_pond`, backend "cuda"). Dense AND native Windowed device beds. AUTHOR-WRITTEN, NOT RUN ON A DEVICE (Root qualifies it on hardware).
Importing this module never imports CuPy or touches a device.

No new physics and no new numerical body: the vertical stage is the SAME floating matrix/preferential/gravity bodies (CUDA owners) composed with the genuine `CudaSurfaceRouter`; the hot geometry uses the
hardware-qualified D3 helpers (`floating_geometry_view`, zero-copy `rebind_*_geometry_view`, the explicit `reconcile_floating_selected` operation with a COMPLETE actual selection mask); the sediment/exchange/quiet/
publication body is the one `PondEvent.step`.

Touched-column depth (device):
  Dense    the native NumPy-order reduction `maple.core.backend.cell_reductions.sum_per_cell_numpy_order` on bounded contiguous gathered rows of ONLY the gross-exchange cells: voxel `(B, 1, V, K)`
           (`trailing_axes=2`) and active `(B, 1, K)`; `D = T / rho / area + active.sum / (rho * area)` through the native `_elevation_from_total_mass` and `compute_active_layer_actual_thickness_m`
           (bitwise the cold canonical depth). The qualified extent (`V * K <= 32768`) is preflighted ONCE at the cold initial handoff, before any physical mutation; an unsupported shape is a typed refusal,
           never a relaxed qualification, never a raw `cupy.sum`, never a non-contiguous fallback and never a whole-bed host copy.
  Windowed `windowed_committed_surface_elevation_m(..., trusted_packet=True)` (native prefix packets, no resident floor, no deep read), valid along the native producer chain (cold full-validated initial bed,
           then the Pond's own native descendants).
Complete selection: the changed-geometry mask is formed from the actual canonical depths with the SAME floating-point slab formula as the owner (`max(0, min(D, e_{k+1}) - e_k)` per layer and the strict
`D > H` bottom flag), on exactly the cells whose raw depth changed; a raw depth change that leaves every slab and the bottom flag bitwise equal (E3c) selects nothing and drains nothing.

Transfer model (counted through MAPLE's `to_host`): the gross-exchange index list is formed with `cupy.flatnonzero` (ONE intrinsic count read, declared), then ONE counted 8-byte count per `moved`
and `changed` selection; the D3 packets are their own (view 32 B; selected 8 + 8 + 32 + 56 + 32 B); no download of W, J, maps, depth or masks in the hot path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from maple_syrup import floating_soil_layers_cuda as fcu

__all__ = ["admit", "after_exchange_cuda", "dense_canonical_depth", "geometry_changed_device", "init_cuda", "init_cuda_boundary", "init_cuda_held", "preflight_dense", "preflight_depth", "scalar", "windowed_canonical_depth"]

DENSE_MAX_TERMS = 32768  # the native qualified extent of the fixed-order reduction (V * K)
HOST_OP_BYTES = 4096  # the declared small host side of one hot operation (counts and the D3 packets are separately counted by their own admissions)


def _err():
    from maple_syrup.floating_pond import FloatingPondError

    return FloatingPondError


def scalar(x: Any) -> int:
    """ONE counted device-to-host read of a 0-d device integer (the declared intrinsic count of a boolean selection)."""
    from maple.core.backend import to_host

    return int(np.asarray(to_host(x)))


def admit(v: Any, device_bytes: int, what: str, host_bytes: int | None = None) -> None:
    """Current device FIRST, then the DEVICE requirement against `memory_budget_bytes` + the (single) device reserve + a truthful device reading and the HOST requirement (+ allowance + the host reserve) against its
    own budget/reading: two separate pools; the reserves are counted ONCE by `fcu.admit`; the real default readers apply when the caller gave none."""
    fcu.admit(v.cp, v.device_id, int(device_bytes), int(v.host_op_bytes if host_bytes is None else host_bytes), v.quota, what)


def _set_models(v: Any, c: Any, preferential: Any, quota: dict, batch_cells: int, *, initial_depth_copy: bool) -> tuple[Any, int, int, int]:
    """The WATER-FREE common CUDA setup shared by the native-owner constructor and the held-context factory: the CURRENT device gate FIRST (before any reader), the shared attributes, and the composed DEVICE/HOST
    stage models from the actual operation terms. Returns `(cp, dev, L, N)`. `initial_depth_copy` says whether a sealed initial depth copy is retained (owner: yes; held context: D0 is REFERENCED, no copy). No array is
    allocated here and nothing is admitted yet."""
    from maple_syrup import floating_matrix_infiltration_cuda as fmc
    from maple_syrup import floating_preferential_infiltration_cuda as fpc
    from maple_syrup import floating_soil_layers as fsl

    cp = fcu._rc().require_cuda()
    dev = fcu.gate_device(cp, c.device_id)  # the CURRENT device before any reader
    L, N = c.n_layers, c.n_cells
    v.cp, v.device_id, v.xp, v.quota = cp, dev, cp, dict(quota)
    v.reserve_bytes = 0  # the device/host reserves live in `quota` and are counted ONCE by `fcu.admit`
    v.batch_cells = fsl._count(batch_cells, "batch_cells", minimum=1)
    v.host_op_bytes = HOST_OP_BYTES
    mm = fmc.operation_terms("step", L, N)
    pm = None if preferential is None else fpc.operation_terms("step", L, N)
    col = fcu.operation_terms("step", L, N)
    chk = fcu.operation_terms("state_check", L, N)
    rec = fcu.operation_terms("reconcile", L, N)
    payload = (L + 2) * N * 8  # one live payload: W (L, N), canonical depth N, the derived sum N
    v.retained_bytes = (L * N + 2 * N) * 8 if initial_depth_copy else (L * N + N) * 8  # zero_added, zero_pet and (owner only) the sealed initial depth
    v.payload_bytes = payload
    # one step: pool/rain/h0 rows and the ColumnStep fields (about 8 N), the three stage results alive at once and the largest scratch, plus the surface result (added at bind)
    v.stage_step_bytes = 8 * N * 8 + mm["outputs"] + (0 if pm is None else pm["outputs"]) + col["outputs"] + max(mm["scratch"], 0 if pm is None else pm["scratch"], col["scratch"]) + 2 * payload
    # one hot exchange: index/touched/depth/mask rows (about 14 N), the new depth map and view maps (N), the staged water and surface/export rows (L N + 2 N), the selected reconcile (outputs + scratch + the
    # two N-byte selection temporaries), the two state checks and the continued-state copy (L N), plus one live payload; the gathered Dense rows are added from the actual (V, K) at use
    v.exchange_bytes = 14 * N * 8 + (L * N + 2 * N) * 8 + rec["outputs"] + rec["scratch"] + 2 * N + 2 * chk["scratch"] + L * N * 8 + payload
    v.surface_step_bytes = 0
    v.last_exchange_required = 0
    v.step_bytes = v.stage_step_bytes
    v.construction_bytes = v.retained_bytes + 3 * payload + max(v.stage_step_bytes, v.exchange_bytes)
    return cp, dev, L, N


def init_cuda(v: Any, owner: Any, preferential: Any, quota: dict, batch_cells: int) -> None:
    """Fill the attributes of a CUDA `FloatingPondVertical`: composed DEVICE and HOST models from the actual stage operation terms, admitted from the real readers BEFORE any owned array exists; then the owned arrays
    (device zero rows and depth, the sealed host active-mask record, the O(L) edges)."""
    c = owner.column
    cp, _dev, L, N = _set_models(v, c, preferential, quota, batch_cells, initial_depth_copy=True)
    # HOST pool of the COLD constructor (dimension-aware, admitted before any readback): the active-mask download (N bool) and its contiguous staging copy (N), the host depth record handed to the one-off upload (8 N, the
    # owner's array is a view but the counted `to_device` stages a buffer), the retained O(L) edges and their transient copy, plus the declared small hot-operation host side; `fcu.admit` adds its 64 KiB Python allowance and
    # the single host reserve. Retained afterwards: the sealed mask record (N) and the edges ((L + 1) 8). Excluded: interpreter/JIT/allocator pool, the owner's own host arrays (already live), whole-process memory.
    v.cold_host_bytes = 2 * N + 8 * N + 2 * (L + 1) * 8 + v.host_op_bytes
    v.retained_host_bytes = N + (L + 1) * 8
    admit(v, v.construction_bytes, "the CUDA floating Pond vertical (retained stage buffers, live payloads and the operation peak)", host_bytes=v.cold_host_bytes)
    v._owner = owner
    # ---- owned state (after admission)
    from maple.core.backend import to_device, to_host

    v._zero_added = cp.zeros((L, N), dtype=np.float64)
    v._zero_pet = cp.zeros(N, dtype=np.float64)
    depth0 = to_device(np.ascontiguousarray(np.asarray(owner.depth_m, dtype=np.float64)), cp)  # the owner's canonical depth record, uploaded ONCE (cold)
    v._depth_scratch = None
    v._host_active = np.ascontiguousarray(np.asarray(to_host(c._maps["active"]), dtype=np.bool_))  # sealed O(N) host record of the active-mask pattern (cold, counted)
    v._host_active.flags.writeable = False
    v._edges_host = np.array(np.asarray(owner.config.nominal_edges_m), dtype=np.float64, copy=True)
    v._initial_depth = depth0


def init_cuda_held(v: Any, context: Any, quota: dict, batch_cells: int) -> None:
    """Fill a CUDA `FloatingPondVertical` from a genuine water-free CUDA `HeldContext` (caller of `FloatingPondVertical.from_held_context`; the exact class/registry check already ran). The SAME water-free setup as the
    owner constructor (`_set_models`), then ONE two-pool admission BEFORE any owned array; the only owned arrays are the two device zero rows. D0 / mask / edges are REFERENCED from the caller-owned context (device
    `column` maps, sealed host `host_active` and `edges_m`): no D2H, no H2D, no cold payload copy. The context itself is caller-owned and EXCLUDED from the constructor models."""
    c = context.column
    cp, _dev, L, N = _set_models(v, c, context.preferential, quota, batch_cells, initial_depth_copy=False)
    v.cold_host_bytes = v.host_op_bytes  # nothing is read back or copied on the host: only the declared small hot-operation host side
    v.retained_host_bytes = 0
    admit(v, v.construction_bytes, "the CUDA held-context floating Pond vertical (retained stage buffers, live payloads and the operation peak; the caller-owned context excluded)", host_bytes=v.cold_host_bytes)
    v._owner = None
    v._zero_added = cp.zeros((L, N), dtype=np.float64)
    v._zero_pet = cp.zeros(N, dtype=np.float64)
    v._depth_scratch = None
    v._host_active = context.host_active  # sealed host mask pattern of the record (referenced)
    v._edges_host = context.edges_m  # sealed HOST O(L) edges (referenced)
    v._initial_depth = None  # no initial payload exists


def init_cuda_boundary(v: Any, root: Any, quota: dict, batch_cells: int) -> None:
    """Fill a CUDA `FloatingPondVertical` for the NEXT native occurrence from a genuine CUDA `BoundaryRoot` (caller: `FloatingPondVertical.from_boundary`; the exact class/registry check already ran). The SAME water-free setup as
    the held-context factory (`_set_models`: current-device gate FIRST), ONE two-pool admission BEFORE any owned array; the only owned arrays are the two device zero rows. The mask / edges are the boundary's sealed HOST
    references (no D2H, no H2D) and the canonical depth is the boundary's CURRENT device record (never the held D0). The root is caller-owned and EXCLUDED from the constructor models."""
    c = root.column
    if root.host_active is None or root.edges_host is None:
        raise _err()("a CUDA boundary root must carry its sealed host mask and edges (a boundary minted by the CUDA lifecycle close)")
    cp, _dev, L, N = _set_models(v, c, root.pref_op, quota, batch_cells, initial_depth_copy=False)
    v.cold_host_bytes = v.host_op_bytes
    v.retained_host_bytes = 0
    admit(v, v.construction_bytes, "the CUDA boundary floating Pond vertical (retained stage buffers, live payloads and the operation peak; the caller-owned root excluded)", host_bytes=v.cold_host_bytes)
    v._owner = None
    v._zero_added = cp.zeros((L, N), dtype=np.float64)
    v._zero_pet = cp.zeros(N, dtype=np.float64)
    v._depth_scratch = None
    v._host_active = root.host_active
    v._edges_host = root.edges_host
    v._initial_depth = root.depth_m


def preflight_depth(v: Any, bed: Any) -> None:
    """COLD, once, before any physical mutation or witness scan: the native fixed-order reduction must be qualified for the actual extents. Dense: the voxel `(V, K)` extent and the active `(K,)` extent; Windowed: the active
    `(K,)` extent (the column total comes from the native prefix packets). An unsupported shape is refused, never relaxed."""
    from maple.core.backend.numpy_pairwise_order import qualify_numpy_order

    from maple_syrup.sediment_bed import is_windowed_bed

    K = int(bed.active_layer.mass_kg.shape[-1])
    terms = K
    if not is_windowed_bed(bed):
        V = int(bed.voxel_column.mass_kg.shape[2])
        terms = max(V * K, K)
        if V * K > DENSE_MAX_TERMS:
            raise _err()(f"the Dense device canonical depth needs V*K = {V * K} <= {DENSE_MAX_TERMS} (the native qualified fixed-order reduction extent); an unsupported shape is refused, not relaxed")
    if K > DENSE_MAX_TERMS:
        raise _err()(f"the device canonical depth needs K = {K} <= {DENSE_MAX_TERMS} (the native qualified fixed-order reduction extent)")
    try:
        if not is_windowed_bed(bed):
            qualify_numpy_order(V * K, trailing_shape=(V, K))
        qualify_numpy_order(K, trailing_shape=(K,))
    except Exception as exc:
        raise _err()(f"the native NumPy-order reduction is not qualified for the actual extent ({terms} terms: {type(exc).__name__}: {exc}); refused before any mutation") from exc


preflight_dense = preflight_depth  # the earlier name (Dense); it now covers both representations


def active_thickness_ordered(v: Any, bed: Any, context: Any, idx: Any) -> Any:
    """`active.sum / (rho * area)` of the cells `idx` in the NATIVE NumPy reduction order (bounded contiguous gathered `(B, 1, K)` rows through `sum_per_cell_numpy_order`, `trailing_axes=1`): the exact expression of the cold
    canonical depth (`compute_active_layer_actual_thickness_m` on host rows), never a raw `cupy.sum`."""
    from maple.core.backend.cell_reductions import sum_per_cell_numpy_order

    cp = v.cp
    g = context.geometry
    K = int(bed.active_layer.mass_kg.shape[-1])
    flat_a = bed.active_layer.mass_kg.reshape(-1, K)
    n = int(idx.shape[0])
    out = []
    for b0 in range(0, n, v.batch_cells):
        i = idx[b0:b0 + v.batch_cells]
        m = int(i.shape[0])
        rows = cp.ascontiguousarray(flat_a[i]).reshape(m, 1, K)
        total = sum_per_cell_numpy_order(rows, trailing_axes=1)
        out.append((total / (g.bulk_density_kg_m3 * (g.dx_m * g.dy_m))).reshape(m))
        del rows, total
    return out[0] if len(out) == 1 else cp.concatenate(out)


def windowed_canonical_depth(v: Any, bed: Any, context: Any, idx: Any) -> Any:
    """Canonical depth of the cells `idx` of a native Windowed device bed: the column total from the NATIVE prefix packets (`windowed_committed_height_total_kg`, NumPy-order, `trusted=True` along the native producer chain)
    converted with the native `_elevation_from_total_mass`, plus the ordered active thickness above (NOT the native whole-grid wrapper, whose active-layer sum is the raw device sum)."""
    from maple.surface.voxels.diagnostics import _elevation_from_total_mass
    from maple.surface.voxels.windowed_prefix import windowed_committed_height_total_kg

    g = context.geometry
    total = windowed_committed_height_total_kg(bed.voxel_column, trusted=True)
    voxel_m = _elevation_from_total_mass(total.reshape(-1)[idx], g.bulk_density_kg_m3, g.dx_m * g.dy_m)
    del total
    return voxel_m + active_thickness_ordered(v, bed, context, idx)


def dense_canonical_depth(v: Any, bed: Any, context: Any, idx: Any) -> Any:
    """Canonical depth `(n,)` of the flat cells `idx` (device int64) of a Dense device bed, bounded batches of `batch_cells` contiguous gathered rows, native NumPy-order reductions only."""
    from maple.core.backend.cell_reductions import sum_per_cell_numpy_order
    from maple.surface.voxels.diagnostics import _elevation_from_total_mass

    cp = v.cp
    g = context.geometry
    vox, act = bed.voxel_column.mass_kg, bed.active_layer.mass_kg
    V, K = int(vox.shape[2]), int(vox.shape[3])
    flat_v, flat_a = vox.reshape(-1, V, K), act.reshape(-1, K)
    n = int(idx.shape[0])
    out = []
    for b0 in range(0, n, v.batch_cells):
        i = idx[b0:b0 + v.batch_cells]
        m = int(i.shape[0])
        rows_v = cp.ascontiguousarray(flat_v[i]).reshape(m, 1, V, K)
        rows_a = cp.ascontiguousarray(flat_a[i]).reshape(m, 1, K)
        total_v = sum_per_cell_numpy_order(rows_v, trailing_axes=2)
        voxel_m = _elevation_from_total_mass(total_v, g.bulk_density_kg_m3, g.dx_m * g.dy_m)
        del total_v, rows_v
        total_a = sum_per_cell_numpy_order(rows_a, trailing_axes=1)  # NumPy-order active total (the native helper's raw device sum is NOT used)
        thickness = total_a / (g.bulk_density_kg_m3 * (g.dx_m * g.dy_m))
        out.append((voxel_m + thickness).reshape(m))
        del rows_a, voxel_m, thickness, total_a
    return out[0] if len(out) == 1 else cp.concatenate(out)


def geometry_changed_device(v: Any, d_old: Any, d_new: Any) -> Any:
    """Per-cell bool (device): does the HYDRAULIC geometry differ between two actual canonical depths? The SAME floating-point expressions as `native_floating_soil._geometry_changed` (no epsilon)."""
    cp = v.cp
    e = v._edges_host
    H = float(e[-1])
    differ = cp.zeros(d_old.shape, dtype=np.bool_)
    for k in range(len(e) - 1):
        h_old = cp.maximum(cp.minimum(d_old, float(e[k + 1])) - float(e[k]), 0.0)
        h_new = cp.maximum(cp.minimum(d_new, float(e[k + 1])) - float(e[k]), 0.0)
        differ |= h_old != h_new
        del h_old, h_new
    if v.config.bottom_drainage == "open":
        differ |= (d_old > H) != (d_new > H)
    return differ


def after_exchange_cuda(v: Any, bed: Any, context: Any, hyd: Any, touched: Any, active: Any) -> tuple[Any, dict]:
    """Stage the actual depth of the GROSS-touched columns after the real MAPLE exchange (device twin of `FloatingPondVertical.after_exchange`; same contract and receipt). `touched` and `active` are device `(ny, nx)`
    bool arrays. Pure: the accepted payload of `hyd` is never mutated."""
    from maple_syrup import floating_matrix_infiltration as fmi
    from maple_syrup import floating_preferential_infiltration as fpi
    from maple_syrup import floating_soil_layers as fsl
    from maple_syrup.experimental_hydrology import HydraulicState
    from maple_syrup.floating_pond import _Payload
    from maple_syrup.sediment_bed import is_windowed_bed

    cp = v.cp
    v.stats.after_exchange_calls += 1
    payload = v._payload(hyd.soil_water_m)
    windowed = bool(is_windowed_bed(bed))
    N = v.n_cells
    gather = 0
    if not windowed:
        V, K = int(bed.voxel_column.mass_kg.shape[2]), int(bed.voxel_column.mass_kg.shape[3])
        gather = v.batch_cells * (V * K + K + 4) * 8  # the bounded gathered rows of one batch (+ the native reducer's own fixed scratch is declared by the native helper)
    else:
        K = int(bed.active_layer.mass_kg.shape[-1])
        gather = 2 * N * 8 + v.batch_cells * (K + 4) * 8  # the whole-grid prefix total decode (+ its sliced copy) and the bounded active rows
    v.last_exchange_required = v.exchange_bytes + gather
    admit(v, v.last_exchange_required, "the CUDA floating touched-column exchange stage (stage arrays + gathered rows / prefix decode)")
    receipt = {"n_touched": 0, "n_depth_changed": 0, "n_reconciled": 0, "surface_return_m3": 0.0, "bottom_export_m3": 0.0, "surface_return_m": None, "bottom_export_m": None}
    gross = (touched & active).reshape(-1)
    idx = cp.flatnonzero(gross).astype(np.int64)  # ONE intrinsic count read (declared); sparse: only the actual gross-exchange cells, net-zero class exchange included
    n = int(idx.shape[0])
    receipt["n_touched"] = n
    v.stats.touched_columns += n
    if n == 0:
        return hyd, receipt
    if windowed:
        d_new = windowed_canonical_depth(v, bed, context, idx)  # native prefix total + ORDERED active thickness (no raw device active sum, no whole-grid wrapper)
    else:
        d_new = dense_canonical_depth(v, bed, context, idx)
    d_old = payload.depth_m[idx]
    moved = d_new != d_old
    n_moved = scalar(moved.sum(dtype=np.int64))
    receipt["n_depth_changed"] = n_moved
    v.stats.depth_changed_columns += n_moved
    if n_moved == 0:
        return hyd, receipt
    ridx, rnew, rold = idx[moved], d_new[moved], d_old[moved]
    depth_full = cp.array(payload.depth_m, copy=True)
    depth_full[ridx] = rnew
    view = fsl.floating_geometry_view(payload.column, depth_full, **v.quota)
    v.stats.views_built += 1
    differ = geometry_changed_device(v, rold, rnew)
    n_changed = scalar(differ.sum(dtype=np.int64))
    water_state = payload.state
    surf = exp_ = None
    if n_changed:
        mask = cp.zeros(N, dtype=np.bool_)
        mask[ridx[differ]] = True  # the COMPLETE actual selection of this exchange (never a caller-invented mask)
        cand = fsl.reconcile_floating_selected(payload.column, payload.state, view, mask, **v.quota)
        water_state, surf, exp_ = cand.state, cand.surface_return_m, cand.bottom_export_m
        receipt["n_reconciled"] = n_changed
        receipt["surface_return_m3"], receipt["bottom_export_m3"] = cand.receipt.surface_return_m3, cand.receipt.bottom_export_m3
        v.stats.reconciled_columns += n_changed
        v.stats.surface_return_m3 += cand.receipt.surface_return_m3
        v.stats.bottom_export_m3 += cand.receipt.bottom_export_m3
    else:
        water_state = fsl.floating_continued_state(view, payload.state.water_m, payload.state.t_s, **v.quota)  # D-equivalent geometry: no extra zero-time cascade (E3c); validated against the new view
    op, hist = fmi.rebind_matrix_geometry_view(payload.matrix_op, view, payload.history)
    pref = None if payload.pref_op is None else fpi.rebind_preferential_geometry_view(payload.pref_op, view)
    new = _Payload(view, op, hist, pref, water_state, depth_full)
    soil = v._derived(water_state)
    ny, nx = v.shape
    depth = hyd.depth_m
    if n_changed:
        depth = depth + surf.reshape(ny, nx)  # the returned water joins the actual local surface ONCE
        receipt["surface_return_m"], receipt["bottom_export_m"] = surf.reshape(ny, nx), exp_.reshape(ny, nx)
    v._register(soil, new)
    return HydraulicState(hyd.t_s, depth if n_changed else cp.array(depth, copy=True), soil, hyd.qx_m2_s, hyd.qy_m2_s, hyd.next_dt_cap_s), receipt
