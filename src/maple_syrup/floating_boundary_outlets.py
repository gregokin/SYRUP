"""Explicit test-only outlet closure at accepted floating lifecycle boundaries.

No solver or transport law lives here. Observed ring elevations stay fixed. A
face that is no longer downhill is closed before the next water occurrence/day;
its instantaneous flux is zeroed without moving water or advancing time. A real
compatibility rebuild becomes the retained routing reference. Fixed-outlet
paths, displacement-triggered rerouting and active-event guards are unchanged.
See docs/subsurface_hydrology/masy_boundary_followup_20261124.md.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np

MAX_CELLS = 16384
MAX_RECEIPT_BYTES = 64 << 10
POLICY = "close_uphill_at_water_boundary"


def close_uphill_boundary_outlets(lineage, boundary, context, *, available_bytes_fn,
        reserve_bytes=0, host_available_bytes_fn=None, host_reserve_bytes=0,
        memory_budget_bytes=None, host_memory_budget_bytes=None, batch_cells=64,
        before_publish=None):
    """Return ``(accepted_boundary, receipt)``; publish a changed successor LAST.

    Dense numpy/numba and CUDA only, <=16384 cells, explicit finite memory caps.
    Both complete and pending-recession boundaries are accepted; their role is
    preserved. Active occurrences/days and stale parents are refused. Only
    currently declared outlets can close: this approximation never reopens one.
    An unchanged outlet set returns the identical head/routing/source/sequence.

    CUDA gates before readers/array work. Cold O(N) transfers are the current
    committed and previous held surfaces plus the physical mask; W/J/I/deep
    volumes are never downloaded.
    Conservative admission includes native bed-validator scratch, copied faces,
    topology arrays and priority-queue objects, plus external reserves once.
    ``before_publish()`` is outside the mint window; its own effects are its own.
    Callback failure does not publish this operation's candidate.
    """
    from maple.core.backend import freeze, to_host
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple_syrup import floating_event_lifecycle as lc
    from maple_syrup import floating_soil_layers as fsl
    from maple_syrup import floating_soil_layers_cuda as fcu
    from maple_syrup import sediment_bed as sb
    from maple_syrup.pond_bed import require_supported_bed, physical_full_source
    from maple_syrup.pond_event import PondDatum
    from maple_syrup.pond_network import PondTopology, build_pond_topology

    lc._expect(lineage, boundary, lc.Boundary)
    lc._record(boundary.root, lc.BoundaryRoot)
    if boundary.role not in ("boundary", "pending_recession"):
        raise lc.LifecycleError("outlet closure requires an accepted water boundary")
    if type(context) is not sb.BedContext or type(boundary.bed) is not sb.BedState or sb.is_windowed_bed(boundary.bed):
        raise lc.LifecycleError("outlet closure requires a genuine Dense native bed/context")
    if before_publish is not None and not callable(before_publish):
        raise lc.LifecycleError("before_publish must be callable or None")
    column = boundary.root.column
    backend = column.backend
    if backend == "cuda":
        xp = fcu._rc().require_cuda()
        fcu.gate_device(xp, column.device_id)
    elif backend in ("numpy", "numba"):
        xp = np
    else:
        raise lc.LifecycleError("unsupported boundary backend")
    for name, value in (("reserve_bytes", reserve_bytes), ("host_reserve_bytes", host_reserve_bytes),
                        ("batch_cells", batch_cells), ("memory_budget_bytes", memory_budget_bytes)):
        if type(value) is not int or value < (1 if name in ("batch_cells", "memory_budget_bytes") else 0):
            raise lc.LifecycleError(f"{name} must be an explicit plain integer in its valid range")
    if backend == "cuda" and (type(host_memory_budget_bytes) is not int or host_memory_budget_bytes < 1):
        raise lc.LifecycleError("an explicit positive host_memory_budget_bytes is required")
    if host_memory_budget_bytes is not None and (type(host_memory_budget_bytes) is not int or host_memory_budget_bytes < 1):
        raise lc.LifecycleError("host_memory_budget_bytes must be a positive plain integer or None")
    if not callable(available_bytes_fn) or (host_available_bytes_fn is not None and not callable(host_available_bytes_fn)):
        raise lc.LifecycleError("actual available-byte readers are required")
    if type(boundary.routing) is not tuple or len(boundary.routing) != 3:
        raise lc.LifecycleError("native retained routing tuple required")
    topology, held, datum = boundary.routing
    if type(topology) is not PondTopology or type(datum) is not PondDatum:
        raise lc.LifecycleError("native topology/datum required")
    g = context.geometry
    ny, nx = int(g.ny), int(g.nx)
    n = ny * nx
    if not 0 < n <= MAX_CELLS or topology.shape != (ny, nx) or column.n_cells != n or topology.dx_m != g.dx_m:
        raise lc.LifecycleError("boundary topology/column/geometry exceed the admitted Dense shape")
    qbytes = int(boundary.qx_m2_s.nbytes) + int(boundary.qy_m2_s.nbytes)
    device = 2 * int(boundary.bed.voxel_column.mass_kg.nbytes) + 12 * int(boundary.bed.active_layer.mass_kg.nbytes) + 64*n + 2*qbytes
    host = 2048*n + 256*len(topology.outlets) + (1 << 20)
    quota = dict(available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes,
        host_available_bytes_fn=host_available_bytes_fn, host_reserve_bytes=host_reserve_bytes,
        memory_budget_bytes=memory_budget_bytes, host_memory_budget_bytes=host_memory_budget_bytes)
    if backend == "cuda":
        fcu.admit(xp, column.device_id, device, host, quota, "boundary outlet compatibility")
    else:
        need = host + device + reserve_bytes + host_reserve_bytes
        fsl._admit(need, memory_budget_bytes, available_bytes_fn, "boundary outlet compatibility")
        if host_memory_budget_bytes is not None or host_available_bytes_fn is not None:
            fsl._admit(need, host_memory_budget_bytes, host_available_bytes_fn, "boundary outlet host compatibility")
    # Every potentially allocating validator runs AFTER admission.
    require_supported_bed(boundary.bed, context)
    if boundary.root.bed is not boundary.bed:
        raise lc.LifecycleError("boundary law root does not carry this bed")
    commit_time = float(boundary.bed.committed_topography.last_commit_time_s)
    if not math.isfinite(boundary.t_s) or commit_time > boundary.t_s or not sb.bed_already_committed(boundary.bed, context, commit_time):
        raise lc.LifecycleError("boundary bed is not authoritatively committed; no commit or repair is performed here")
    active = np.asarray(topology.active)
    if context.domain is not None:
        domain_mask = np.asarray(to_host(context.domain.valid_mask))
        if not np.array_equal(active, domain_mask):
            raise lc.LifecycleError("topology differs from the physical footprint")
    old_full = physical_full_source(datum.initial_full_m, datum.initial_committed_m,
                                   np.asarray(to_host(held)), active)
    if not np.array_equal(old_full, topology.source_z_full):
        raise lc.LifecycleError("retained routing source/datum do not reproduce their topology")
    committed = committed_elevation_m(boundary.bed.committed_topography)
    if type(committed) is not xp.ndarray or committed.shape != (ny, nx) or committed.dtype != np.float64:
        raise lc.LifecycleError("committed world surface has a foreign backend/shape/dtype")
    full = physical_full_source(datum.initial_full_m, datum.initial_committed_m,
                                np.asarray(to_host(committed)), active)
    keep, removed = [], []
    for o in topology.outlets:
        (keep if full[o.row+1, o.col+1] > full[o.row+o.d_row+1, o.col+o.d_col+1] else removed).append(o)
    receipt = dict(policy=POLICY, changed=bool(removed), reason="outlet_compatibility" if removed else "unchanged_outlets",
        time_s_hex=float(boundary.t_s).hex(), role=boundary.role, outlets_before=len(topology.outlets),
        outlets_after=len(keep), closed_faces=[o.as_tuple() for o in removed], removed_prior_unit_discharge_max_m2_s=0.0,
        topology_before=topology.topology_sha256, topology_after=topology.topology_sha256,
        source_before=topology.source_sha256, source_after=topology.source_sha256,
        generation_before=int(topology.generation), generation_after=int(topology.generation),
        water_exchange_m3=0.0, sediment_exchange_kg=0.0, elapsed_s=0.0,
        admitted_host_scratch_bytes=host, admitted_backend_scratch_bytes=device,
        approximation="fixed observed ring; close only; no backwater, reopening, momentum or restart qualification")
    if not removed:
        return boundary, receipt
    rebuilt = build_pond_topology(full, active, np.asarray(topology.friction), topology.dx_m,
        keep, nodata_value=topology.nodata_value, allow_closed_components=True)
    rebuilt = replace(rebuilt, generation=topology.generation+1)
    for a, shape in ((boundary.qx_m2_s, (ny, nx+1)), (boundary.qy_m2_s, (ny+1, nx))):
        if type(a) is not xp.ndarray or a.dtype != np.float64 or a.shape != shape:
            raise lc.LifecycleError("carried face flux has a foreign backend/shape/dtype")
        if backend == "cuda" and int(a.device.id) != int(column.device_id):
            raise lc.LifecycleError("carried face flux is on another device")
    qx, qy = boundary.qx_m2_s.copy(), boundary.qy_m2_s.copy()
    prior = []
    for o in removed:
        a, index = (qx, (o.row, o.col + (o.d_col > 0))) if o.d_col else (qy, (o.row + (o.d_row > 0), o.col))
        prior.append(abs(float(to_host(a[index]))))
        a[index] = 0.0
    if not all(math.isfinite(v) for v in prior):
        raise lc.LifecycleError("removed outlet has nonfinite carried flux")
    new_held = freeze(xp.array(committed, copy=True))
    receipt.update(removed_prior_unit_discharge_max_m2_s=max(prior, default=0.0),
        topology_after=rebuilt.topology_sha256, source_after=rebuilt.source_sha256,
        generation_after=int(rebuilt.generation))
    raw = json.dumps(receipt, sort_keys=True, allow_nan=False)
    if len(raw.encode()) > MAX_RECEIPT_BYTES:
        raise lc.LifecycleError("boundary outlet receipt exceeds its explicit 64KiB cap")
    if before_publish is not None:
        before_publish()
    lc._expect(lineage, boundary, lc.Boundary)
    with lc._minting():
        successor = lc.Boundary(boundary.role, boundary.root, boundary.bed, boundary.surface_depth_m,
            freeze(qx), freeze(qy), (rebuilt, new_held, datum), boundary.t_s, boundary.ledger,
            boundary.provenance, boundary.seq+1,
            boundary.cursor, boundary.pending_forcing, boundary.storm_match)
    lineage._head = successor
    return successor, receipt
