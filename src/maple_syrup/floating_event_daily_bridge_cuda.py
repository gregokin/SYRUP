"""Event <-> daily BRIDGE, CUDA branches (AUTHOR-WRITTEN, NOT RUN ON A DEVICE; Root qualifies on hardware; importing never imports CuPy). The ONE lineage head, the DailyHold/cursor/storm-match/foreign-hold logic and the dry/wet
day transactions are the shared backend-neutral bridge code (`floating_event_daily_bridge`, `floating_daily`, `floating_event_lifecycle`); this module holds ONLY the device-specific pieces:
  open_day_cuda     the CUDA `open_day`: current-device gate FIRST (before any reader/hash/array), the strict two-pool quota (`floating_daily_cuda.quota`), ONE composed admission BEFORE any owned array (the `(L, N)`/`(N,)` device zero
                    blocks + the storage query on the DEVICE pool; the cold source-pair hash, the O(L) edges and the scalar packets on the HOST pool; reserves once, 64 KiB host allowance), then the context from the boundary root:
                    the root's CURRENT device canonical D and the device-prepared active mask by reference, the root's SEALED host mask/edges (no D2H), the evaporation operator, the initial native storage and the wet probe from counted
                    SCALAR reads (`max(H)`, `sum(H)`); the carried surface/faces are device-to-device copies. The head is replaced LAST. No `NativeFloatingSoil`, no NVK/bed scan, no host W/J/I mirror.
  wet-day helpers   device input validation (one packet), counted scalar sums and the finite/non-negative depth check, so the wet loop of `wet_day_action` never copies an (N,) array to the host per fragment (the fragment terms
                    are the E-ledger / FSL-budget host scalars and the router's export scalar: `_FloatingSolver.step_day` on a CUDA vertical leaves scalars only).
Dry days are the EXISTING qualified CUDA dispatch of `floating_daily_action` (one 86400 s step). T is the existing CUDA withdrawal. Memory models are declared payload models (reserves once per pool; the caller's event factory, the
router's own admission and the current inputs are separate admissions/excluded); no exact whole-day need is claimed. Not claimed: any device run, GPU restart/physical file, large-catchment readiness.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np

from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_cuda as fdc
from maple_syrup import floating_evaporation as fe
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup import subsurface as ss

__all__ = ["admit_wet", "depth_ok_cuda", "inputs_ok_cuda", "open_day_cuda", "open_plan_cuda", "restore_closed_day_cuda", "scalar", "sum_cuda", "wet_plan_cuda", "wet_quota"]


def _bridge():
    from maple_syrup import floating_event_daily_bridge as br

    return br


def scalar(x: Any) -> float:
    """ONE counted 8-byte device-to-host read of a 0-d device value."""
    from maple.core.backend import to_host

    return float(np.asarray(to_host(x)))


def sum_cuda(column: Any, a: Any) -> float:
    from maple_syrup import floating_soil_layers_cuda as fcu

    cp = fcu._rc().require_cuda()
    fcu.gate_device(cp, column.device_id)
    return scalar(cp.sum(a))


def inputs_ok_cuda(column: Any, pet: Any, cover: Any) -> bool:
    """PET finite and >= 0 and cover finite in [0, 1] on every cell, from ONE device reduction packet (metadata was validated by `fd._validate_inputs`)."""
    from maple_syrup import floating_soil_layers_cuda as fcu

    cp = fcu._rc().require_cuda()
    fcu.gate_device(cp, column.device_id)
    ok = cp.all(cp.isfinite(pet) & (pet >= 0.0)) & cp.all(cp.isfinite(cover) & (cover >= 0.0) & (cover <= 1.0))
    return bool(scalar(ok))


def depth_ok_cuda(column: Any, depth: Any) -> bool:
    from maple_syrup import floating_soil_layers_cuda as fcu

    cp = fcu._rc().require_cuda()
    fcu.gate_device(cp, column.device_id)
    return bool(scalar(cp.all(cp.isfinite(depth) & (depth >= 0.0))))


def wet_quota(column: Any, available_bytes_fn: Any, reserve_bytes: Any, host_available_bytes_fn: Any, host_reserve_bytes: Any) -> dict:
    """Current device gate FIRST (before any reader), then the strict two-pool quota (typed refusal of a non-callable reader / non-integer reserve). No budget is invented; a caller zero is a zero."""
    br = _bridge()
    try:
        fdc.gate(column)
        return fdc.quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes)
    except (fsl.FloatingSoilError, ss.SubsurfaceError, RuntimeError) as exc:
        raise br.BridgeError(f"the CUDA device gate / quota was refused: {exc}") from None


def wet_plan_cuda(L: int, N: int) -> tuple[int, int]:
    """`(device, host)` declared payload of THIS transaction's own scalar/probe operations (the vertical's stages and the router admit separately and are excluded): the storage queries (`operation_terms("storage")`), the input
    validation packet (two device reductions over N), the surface sum/finite probes (8 N partials) and the host scalar side."""
    from maple_syrup import floating_soil_layers_cuda as fcu

    t = fcu.operation_terms("storage", int(L), int(N))
    return int(t["device"]) + 2 * 8 * int(N) + 8 * int(N), int(t["host"]) + 4096


def admit_wet(column: Any, q: dict) -> None:
    br = _bridge()
    d, h = wet_plan_cuda(column.n_layers, column.n_cells)
    try:
        fdc.admit(column, d, h, q, "the CUDA wet-day transaction (scalar queries and probes; the stages admit separately)")
    except (fsl.FloatingSoilError, ss.SubsurfaceError, MemoryError) as exc:
        raise br.BridgeError(str(exc)) from None


def open_plan_cuda(L: int, N: int) -> tuple[int, int]:
    """`(device, host)` declared payload of `open_day_cuda` (reserves/allowance added by the admission). The operation's OWN live payload is DEVICE = the zero blocks + the storage query (`fdc.start_plan`) + the carried
    surface/face copies (3 N doubles) + the wet probe partials (8 N upper bound); HOST = the scalar packets, the O(L) edges and the first-use source-pair hash. It also performs
    `fe.prepare_floating_evaporation` on the caller's pools (zero reserve: this admission counted it), whose EXISTING per-staging guard requires `fe.estimate_floating_evaporation_memory(N, backend="cuda")` on each pool
    (the preparation binds the operator only and allocates no array, but it is an admission against the same readers). So at the minimum the outer need must cover it: per pool `max(own live payload, preparation
    requirement)` (the guard is not changed and no arbitrary pad is added; the two are sequential, never summed)."""
    d, h = fdc.start_plan(int(L), int(N))
    own_d = d + 3 * 8 * int(N) + 8 * int(N)
    own_h = h + (int(L) + 1) * 8 + fd._cold_source_bytes() + fd.CONSTRUCTION_ALLOWANCE_BYTES
    prep = fe.estimate_floating_evaporation_memory(int(N), backend="cuda")  # reserves are not part of the per-staging requirement (counted once by the outer admission)
    return max(own_d, int(prep.device_bytes)), max(own_h, int(prep.host_bytes))


def open_day_cuda(lineage: Any, boundary: Any, *, partition: Any, run_digest: Any, available_bytes_fn: Any, reserve_bytes: int, host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0,
                  memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None) -> Any:
    """CUDA `open_day` (module docstring). The shared checks (head, eligibility, partition type) ran in the caller; the order here is gate -> quota -> composed admission -> arrays -> head LAST."""
    from maple_syrup import floating_event_lifecycle as lc

    br = _bridge()
    root = boundary.root
    column = root.column
    if root.host_active is None or root.edges_host is None:
        raise br.BridgeError("a CUDA boundary root must carry its sealed host mask and edges (a boundary minted by the CUDA lifecycle)")
    try:
        fdc.gate(column)  # the CURRENT device FIRST: before any reader, hash or array
        q = fdc.quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    except (fsl.FloatingSoilError, ss.SubsurfaceError, RuntimeError) as exc:
        raise br.BridgeError(f"the CUDA device gate / quota was refused: {exc}") from None
    cursor = boundary.cursor
    if cursor is None:
        try:
            cursor = rc.DailyCursor(rc._hex64(run_digest, "run_digest"))
        except rc.RechargeError as exc:
            raise br.BridgeError(str(exc)) from None
    L, N = int(column.n_layers), int(column.n_cells)
    d_need, h_need = open_plan_cuda(L, N)
    try:
        fdc.admit(column, d_need, h_need, q, "the CUDA event<->daily bridge day opening (zero blocks, storage query, surface copies, source hash)")
    except (fsl.FloatingSoilError, ss.SubsurfaceError, MemoryError) as exc:
        raise br.BridgeError(str(exc)) from None
    comp = fdc.component_quota(q)
    hold = lc.stage_daily_hold(lineage, boundary)
    try:
        from maple_syrup.pond_checkpoint import pond_checkpoint_source_pair

        pair = MappingProxyType(dict(pond_checkpoint_source_pair()))
        evap = fe.prepare_floating_evaporation(column, partition, **comp)
        soil0 = float(fsl.floating_storage_m3(column, root.water_state, **comp))
    except (fsl.FloatingSoilError, ss.SubsurfaceError, ImportError) as exc:
        raise br.BridgeError(f"evaporation operator / source pair / initial native storage refused: {exc}") from None
    zero_added, zero_surface = fdc.zeros(column, (L, N)), fdc.zeros(column, (N,))
    depth, active = root.depth_m, column._maps["active"]  # the CURRENT device canonical D and the device-prepared mask, by reference
    edges = root.edges_host
    shape = tuple(int(x) for x in boundary.surface_depth_m.shape)
    physics = fd._physics_config_sha_parts(partition, column.backend, root.source_witness, shape, config_sha256=root.config.sha256, root_column_sha256=column.sha256, matrix_op_sha256=root.matrix_op.sha256,
                                            generation=root.generation, area=column.cell_area_m2, profile=br.BRIDGE_PROFILE)
    n_active = int(np.asarray(root.host_active).sum())  # HOST mask pattern (never a device reduction)
    ctx = fd._Context(root.config.sha256, column, evap, partition, zero_added, zero_surface, shape, L, N, float(column.cell_area_m2), depth, active, edges, root.source_witness, int(root.generation),
                      column.sha256, cursor.run_digest, None, physics, pair, root.config, root.matrix_op, root.ks_m_s, n_active, int(boundary.routing[0].generation), int(root.generation), hold, br.BRIDGE_PROFILE,
                      fd._stable_physics_sha(partition, column.backend, root.source_witness, shape, config_sha256=root.config.sha256, area=column.cell_area_m2, profile=br.BRIDGE_PROFILE))
    from maple_syrup import floating_soil_layers_cuda as fcu

    cp = fcu._rc().require_cuda()
    h0 = boundary.surface_depth_m
    wet = bool(scalar(cp.max(h0)) > 0.0)  # counted scalar probe
    surf0 = (scalar(cp.sum(h0)) * float(column.cell_area_m2)) if wet else 0.0
    led0 = fd.DailyLedger(initial_storage_m3=soil0 + surf0, expected_t_s=float(root.water_state.t_s), surface_m3=surf0)
    surface = (cp.array(boundary.surface_depth_m, copy=True), cp.array(boundary.qx_m2_s, copy=True), cp.array(boundary.qy_m2_s, copy=True)) if wet else None
    state = fd._mint_state(ctx, "boundary", root.water_state, root.history, cursor, None, None, 0, 0, led0, None, None,
                           {"kind": "bridge_start", "boundary_seq": boundary.seq, "witness": root.source_witness, "physics": physics, "property_generation": int(root.generation), "wet": wet}, surface)
    lc.install_daily_hold(lineage, boundary, hold)  # the head replacement is LAST
    return state


def restore_closed_day_cuda(token: Any, hold: Any, root: Any, state: Any, record: Any, *, partition: Any, q: dict) -> Any:
    """Mint the CLOSED day of a verified CUDA file restore (`floating_lifecycle_checkpoint_cuda`; hook of the file codec, not a public transition). The CUDA twin of `floating_event_daily_bridge.restore_closed_day`: the
    context is built from the staged root/hold EXACTLY as `open_day_cuda` builds it (current device gate FIRST, the caller's strict two-pool quota `q`, ONE composed admission of `open_plan_cuda` BEFORE any owned array, the root's
    CURRENT device canonical D and device-prepared active mask by reference, the sealed host mask/edges, the evaporation operator, resident storage from counted scalar reads), the closed state is the producer-receipt restore
    `floating_daily.restore_closed_state` (execution/state digests RECOMPUTED, revision, request sequence, parent digest, cursor and ledger preserved EXACTLY), and the cumulative lifecycle identity of the future hand-back (hold
    ledger + this day segment, added ONCE) must close. No new day, no reset, no step time; nothing is installed (the caller installs the head LAST)."""
    from maple_syrup import floating_event_lifecycle as lc
    from maple_syrup import floating_event_lifecycle_cuda as flc

    br = _bridge()
    lc._record(token, lc.VerifiedRestore)
    if type(hold) is not lc.DailyHold or hold not in lc._RECORDS or type(root) is not lc.BoundaryRoot or root not in lc._ROOTS or token.state is not state:
        raise br.BridgeError("restore_closed_day_cuda needs the staged hold/root of this verified restore")
    if not isinstance(partition, se.EvaporationPartition):
        raise br.BridgeError(f"partition must be an EvaporationPartition, got {type(partition).__name__}")
    column = root.column
    if column.backend != "cuda" or root.host_active is None or root.edges_host is None:
        raise br.BridgeError("a CUDA restore needs a CUDA root carrying its sealed host mask and edges")
    try:
        fdc.gate(column)  # the CURRENT device FIRST
        d_need, h_need = open_plan_cuda(int(column.n_layers), int(column.n_cells))
        fdc.admit(column, d_need, h_need, q, "the CUDA closed-day restore (zero blocks, storage query, surface copies, evaporation operator, source hash)")
        comp = fdc.component_quota(q)
        cursor = rc.cursor_from_record(record["cursor"])
        from maple_syrup.pond_checkpoint import pond_checkpoint_source_pair

        pair = MappingProxyType(dict(pond_checkpoint_source_pair()))
        evap = fe.prepare_floating_evaporation(column, partition, **comp)
        L, N = int(column.n_layers), int(column.n_cells)
        zero_added, zero_surface = fdc.zeros(column, (L, N)), fdc.zeros(column, (N,))
        hyd = state.hydraulic
        shape = tuple(int(x) for x in hyd.depth_m.shape)
        routing_generation = int(state.topology.generation)
        physics = fd._physics_config_sha_parts(partition, column.backend, root.source_witness, shape, config_sha256=root.config.sha256, root_column_sha256=column.sha256, matrix_op_sha256=root.matrix_op.sha256,
                                                generation=root.generation, area=column.cell_area_m2, profile=br.BRIDGE_PROFILE)
        n_active = int(np.asarray(root.host_active).sum())  # HOST mask pattern (never a device reduction)
        ctx = fd._Context(root.config.sha256, column, evap, partition, zero_added, zero_surface, shape, L, N, float(column.cell_area_m2), root.depth_m, column._maps["active"], root.edges_host, root.source_witness,
                          int(root.generation), column.sha256, cursor.run_digest, None, physics, pair, root.config, root.matrix_op, root.ks_m_s, n_active, routing_generation, int(root.generation), hold, br.BRIDGE_PROFILE,
                          fd._stable_physics_sha(partition, column.backend, root.source_witness, shape, config_sha256=root.config.sha256, area=column.cell_area_m2, profile=br.BRIDGE_PROFILE))
        from maple_syrup import floating_soil_layers_cuda as fcu

        cp = fcu._rc().require_cuda()
        surface = (cp.array(hyd.depth_m, copy=True), cp.array(hyd.qx_m2_s, copy=True), cp.array(hyd.qy_m2_s, copy=True)) if record["has_surface"] else None
        closed = fd.restore_closed_state(ctx, root.water_state, root.history, record, surface)
    except MemoryError:
        raise  # EVERY resource refusal keeps its own class (typed `SubsurfaceMemoryError` subclasses, a plain `MemoryError`, a CuPy `OutOfMemoryError`): a retry with a larger reader never sees a semantic error
    except (rc.RechargeError, KeyError, TypeError, fd.DailyRefusal, fsl.FloatingSoilError, ss.SubsurfaceError) as exc:
        raise br.BridgeError(f"the closed day cannot be restored: {exc}") from None
    d = closed.ledger
    led = dict(hold.ledger)
    for key, val in (("rain_m3", d.rain_m3), ("evaporation_m3", d.evaporation_m3), ("transpiration_m3", d.transpiration_m3), ("drainage_m3", d.drainage_m3)):
        led[key] = led.get(key, 0.0) + float(val)
    led["outlet_export_m3"] += float(d.outlet_m3)
    led["bound_m3"] += float(d.bound_m3)
    depth = hold.surface_depth_m if closed.surface is None else closed.surface[0]
    lc._balance_check(led, flc.stored_m3_cuda(column, closed.water, depth, float(ctx.area), q))
    return closed
