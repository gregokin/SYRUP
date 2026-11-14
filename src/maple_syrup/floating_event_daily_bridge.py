"""Event <-> daily BRIDGE (CPU: NumPy/Numba; Dense and published Windowed beds REFERENCED, never copied): an accepted lifecycle `Boundary` (prepared root, CURRENT W/J/I/D, surface H/q, routing reference, bed) enters a genuine
floating DAILY lineage and its closed day returns to the SAME lifecycle head, so event -> day -> event is one accepted chain.

ONE head. The only authority is `floating_event_lifecycle.FloatingEventLineage`: `Boundary -> DailyHold -> Boundary`. `open_day` stages everything refusable (eligibility, dry surface, memory admission, the source pair, the
evaporation operator, the new-profile `FloatingDailyState`) and replaces the head LAST; a refusal leaves the boundary head untouched and retryable. The day's transactions are the EXISTING accepted ones (`floating_daily_action`,
`request_floating_withdrawal`, `close_floating_day`, optional every-N output) with the bridge profile's context built from the boundary root: no `NativeFloatingSoil`, no cold NVK scan or bed hash (the bed is not even stored in the
context), no old W0/parent bed. `hand_back` rebases the closed day into a new Boundary on the same head (the lifetime ledger adds the day's rain/E/T/drainage and the cumulative identity must still close within the accumulated MAPLE
bounds, else the hand-back is refused before the head moves).

Scope (explicit). IMPLEMENTED (CPU, NumPy/Numba, Dense and published Windowed): dry days (P0 and small-rain recharge integrated ONCE with the daily soil update, ONE 86400 s step, aggregate T once, audited close), wet days (standing
surface water: short hydrology/E fragments totalling exactly 86400 s in the accepted stage order preferential -> matrix -> E -> ONE FSL step -> return -> the accepted surface router, carried routing topology, no sediment, no
Pond state / bed inventory), `P > T` / capacity failure -> `requires_full_storm` with the FULL native forcing from the unchanged parent, `begin_storm_from_day` (the ORIGINAL plan pending on the cursor with its bound forcing), the
genuine full native storm through the lifecycle and `day_after_storm` (the original plan consumed once). A pending storm is bound to its forcing at begin (absolute edges/intensity/rain_scale/clock) AND, at completion, to the
PRODUCER's proof of the forcing HISTORY: `PondEvent.forcing_conformity(state)` is True only if every accepted step of the state's chain integrated exactly the native schedule's rate on a piece-confined interval (a bounded weak
bool per live state, never a caller flag). A total alone is no proof: a same-total manual history, an off-piece step, an unproven chain (a state restored from a file or built elsewhere: the receipt is process-local; a restart
needs a future versioned file field) or a rain-0 recertification with a positive manual step is refused before the head moves. Every-N output continues across accepted events and property refreshes (versioned evolving
profile). NOT implemented: CUDA (refused), a physical daily/boundary file codec, wind, LEAF/growth. Hooks a codec needs: the state's `water/history/cursor/ledger/execution_digest/surface`, the hold's static pieces and the
lifecycle ledger; no skeleton file is provided.
"""
from __future__ import annotations

import math
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import numpy as np

from maple_syrup import floating_daily as fd
from maple_syrup import floating_evaporation as fe
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.experimental_hydrology import HydraulicStepRejected
from maple_syrup.rainfall import constant_rainfall

__all__ = ["BRIDGE_PROFILE", "BridgeError", "begin_storm_from_day", "day_action", "day_after_storm", "hand_back", "open_day", "restore_closed_day", "storm_schedule", "wet_day_action"]

BRIDGE_PROFILE = "native_floating_daily_from_event_boundary/1"


class BridgeError(RuntimeError):
    """Typed refusal of the event<->daily bridge."""


def _need(ctx_like_n: int, L: int, backend: str, layouts: Any, diffusion_mode: Any, reserve: int) -> int:
    """Composed HOST plan of `open_day` from dimensions only (admitted BEFORE any owned array, the first-use source hash or the evaporation operator): the cold source-pair hash when the caches are empty, the evaporation
    operator model, the retained arrays (zero `(L, N)` block, `N` zero surface / mask / depth, `L + 1` edges), one storage-query block, the allowance and the reserve once."""
    N = ctx_like_n
    evap = int(fe.estimate_floating_evaporation_memory(N, backend=backend).host_bytes)
    retained = (L * N + 3 * N + (L + 1)) * 8 + N
    return fd._cold_source_bytes() + evap + retained + L * N * 8 + fd.CONSTRUCTION_ALLOWANCE_BYTES + reserve


def _build_context(root: Any, surface_shape: Any, routing_generation: int, hold: Any, cursor: Any, partition: Any, available_bytes_fn: Any, reserve: int) -> tuple:
    """The bridge profile's `_Context` (and the native storage of the root's CURRENT water) from a root, the surface shape, the carried routing generation, the hold and the cursor: shared by `open_day` and a verified
    file restore (`floating_lifecycle_checkpoint`), so both mint the same context from the same inputs. Returns `(ctx, soil0_m3, physics_sha)`."""
    column = root.column
    L, N = int(column.n_layers), int(column.n_cells)
    try:
        from maple_syrup.pond_checkpoint import pond_checkpoint_source_pair

        pair = MappingProxyType(dict(pond_checkpoint_source_pair()))  # the ACTUAL imported model source pair (cold once, admitted above; warm afterwards)
        evap = fe.prepare_floating_evaporation(column, partition, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
        soil0 = float(fsl.floating_storage_m3(column, root.water_state))
    except (fsl.FloatingSoilError, ss.SubsurfaceError, ImportError) as exc:
        raise BridgeError(f"evaporation operator / source pair / initial native storage refused: {exc}") from None
    zero_added, zero_surface = np.zeros((L, N)), np.zeros(N)
    zero_added.flags.writeable = False
    zero_surface.flags.writeable = False
    depth = np.array(root.depth_m, copy=True)
    depth.flags.writeable = False
    edges = np.array(column._maps["edges"], copy=True)
    edges.flags.writeable = False
    active = np.array(column._maps["active"], copy=True)
    active.flags.writeable = False
    shape = tuple(int(x) for x in surface_shape)
    physics = fd._physics_config_sha_parts(partition, column.backend, root.source_witness, shape, config_sha256=root.config.sha256, root_column_sha256=column.sha256, matrix_op_sha256=root.matrix_op.sha256,
                                            generation=root.generation, area=column.cell_area_m2, profile=BRIDGE_PROFILE)
    ctx = fd._Context(root.config.sha256, column, evap, partition, zero_added, zero_surface, shape, L, N, float(column.cell_area_m2), depth, active, edges, root.source_witness, int(root.generation),
                      column.sha256, cursor.run_digest, None, physics, pair, root.config, root.matrix_op, root.ks_m_s, int(active.sum()), int(routing_generation), int(root.generation), hold, BRIDGE_PROFILE,
                      fd._stable_physics_sha(partition, column.backend, root.source_witness, shape, config_sha256=root.config.sha256, area=column.cell_area_m2, profile=BRIDGE_PROFILE))  # bed_generation = the carried routing topology generation (genuine net-elevation hydraulic builds); depth_generation = the prepared root's property generation (its canonical D)
    return ctx, soil0, physics


def open_day(lineage: Any, boundary: Any, *, partition: Any, run_digest: Any = None, available_bytes_fn: Any, reserve_bytes: int = 0, host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0,
             memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None) -> fd.FloatingDailyState:
    """Open a daily lineage from the head `Boundary` (module docstring). `run_digest` is required when the boundary carries no cursor (a new run); otherwise the boundary's own cursor continues (the nominal ordinal and any
    pending storm). Returns the closed-role daily state; the lineage head is the hold LAST."""
    lc._expect(lineage, boundary, lc.Boundary)
    root = boundary.root
    if root.column.backend not in ("numpy", "numba", "cuda"):
        raise BridgeError("the event<->daily bridge serves the numpy/numba/cuda verticals")
    if not boundary.eligible_for_interevent:
        raise BridgeError("a pending recession boundary is not eligible for a day until a real native recession completes again")
    if not isinstance(partition, se.EvaporationPartition):
        raise BridgeError(f"partition must be an EvaporationPartition, got {type(partition).__name__}")
    if root.column.backend == "cuda":
        from maple_syrup import floating_event_daily_bridge_cuda as bc

        return bc.open_day_cuda(lineage, boundary, partition=partition, run_digest=run_digest, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, host_available_bytes_fn=host_available_bytes_fn,
                                host_reserve_bytes=host_reserve_bytes, memory_budget_bytes=memory_budget_bytes, host_memory_budget_bytes=host_memory_budget_bytes)
    if host_available_bytes_fn is not None or host_reserve_bytes or memory_budget_bytes is not None or host_memory_budget_bytes is not None:
        raise BridgeError("host-pool / budget arguments apply only to a CUDA boundary (a meaningless CPU argument is refused)")
    if not callable(available_bytes_fn):
        raise BridgeError("available_bytes_fn must be callable (the admission needs a real reader)")
    reserve = fsl._count(reserve_bytes, "reserve_bytes")
    cursor = boundary.cursor
    if cursor is None:
        try:
            cursor = rc.DailyCursor(rc._hex64(run_digest, "run_digest"))
        except rc.RechargeError as exc:
            raise BridgeError(str(exc)) from None
    column = root.column
    L, N = int(column.n_layers), int(column.n_cells)
    try:
        fd._read(available_bytes_fn, _need(N, L, column.backend, column.layouts, column.diffusion_mode, reserve), "the event<->daily bridge day opening (first-use source hash, evaporation operator, retained arrays)")
    except fd.DailyRefusal as exc:
        raise BridgeError(str(exc)) from None
    hold = lc.stage_daily_hold(lineage, boundary)
    ctx, soil0, physics = _build_context(root, boundary.surface_depth_m.shape, int(boundary.routing[0].generation), hold, cursor, partition, available_bytes_fn, reserve)
    h0 = np.asarray(boundary.surface_depth_m)
    wet = bool(h0.max(initial=0.0) > 0.0)
    surf0 = float(h0.sum()) * float(column.cell_area_m2)
    led0 = fd.DailyLedger(initial_storage_m3=soil0 + surf0, expected_t_s=float(root.water_state.t_s), surface_m3=surf0)
    state = fd._mint_state(ctx, "boundary", root.water_state, root.history, cursor, None, None, 0, 0, led0, None, None,
                           {"kind": "bridge_start", "boundary_seq": boundary.seq, "witness": root.source_witness, "physics": physics, "property_generation": int(root.generation), "wet": wet},
                           (np.array(boundary.surface_depth_m, copy=True), np.array(boundary.qx_m2_s, copy=True), np.array(boundary.qy_m2_s, copy=True)) if wet else None)
    lc.install_daily_hold(lineage, boundary, hold)  # the head replacement is LAST
    return state


def restore_closed_day(token: Any, hold: Any, root: Any, state: Any, record: Any, *, partition: Any, available_bytes_fn: Any, reserve_bytes: int = 0) -> fd.FloatingDailyState:
    """Mint the CLOSED day of a verified file restore (`floating_lifecycle_checkpoint`; hook of the file codec, not a public transition): the context is built from the staged root/hold exactly as `open_day` builds it, the closed
    state is the producer-receipt restore of `floating_daily.restore_closed_state` (execution digest, revision, request sequence, parent digest, cursor and ledger preserved EXACTLY), and the cumulative lifecycle identity of
    the future hand-back (hold ledger + this day segment, added ONCE) must close. Nothing is installed: the caller replaces the new lineage's head LAST."""
    lc._record(token, lc.VerifiedRestore)
    if type(hold) is not lc.DailyHold or hold not in lc._RECORDS or type(root) is not lc.BoundaryRoot or root not in lc._ROOTS or token.state is not state:
        raise BridgeError("restore_closed_day needs the staged hold/root of this verified restore")
    if not isinstance(partition, se.EvaporationPartition):
        raise BridgeError(f"partition must be an EvaporationPartition, got {type(partition).__name__}")
    try:
        cursor = rc.cursor_from_record(record["cursor"])
        ctx, _soil0, _physics = _build_context(root, tuple(np.asarray(state.hydraulic.depth_m).shape), int(state.topology.generation), hold, cursor, partition, available_bytes_fn, fsl._count(reserve_bytes, "reserve_bytes"))
        hyd = state.hydraulic
        surface = (np.array(hyd.depth_m, copy=True), np.array(hyd.qx_m2_s, copy=True), np.array(hyd.qy_m2_s, copy=True)) if record["has_surface"] else None
        closed = fd.restore_closed_state(ctx, root.water_state, root.history, record, surface)
    except (rc.RechargeError, KeyError, TypeError, fd.DailyRefusal) as exc:
        raise BridgeError(f"the closed day cannot be restored: {exc}") from None
    d = closed.ledger
    led = dict(hold.ledger)
    for key, val in (("rain_m3", d.rain_m3), ("evaporation_m3", d.evaporation_m3), ("transpiration_m3", d.transpiration_m3), ("drainage_m3", d.drainage_m3)):
        led[key] = led.get(key, 0.0) + float(val)
    led["outlet_export_m3"] += float(d.outlet_m3)
    led["bound_m3"] += float(d.bound_m3)
    depth = hold.surface_depth_m if closed.surface is None else closed.surface[0]
    lc._balance_check(led, float(fsl.floating_storage_m3(ctx.column, closed.water)) + float(np.asarray(depth).sum()) * float(ctx.area))
    return closed


def hand_back(lineage: Any, closed: Any, *, available_bytes_fn: Any = None, reserve_bytes: Any = 0, host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0) -> lc.Boundary:
    """Return the CLOSED day (role boundary, no pending storm) to the lifecycle head as a new Boundary (module docstring)."""
    _need_bridge_state(closed)
    if closed.cursor.pending is not None:
        raise BridgeError("a pending storm must be run through the lifecycle (begin_storm_from_day), not handed back as a closed day")
    cuda = closed.ctx.column.backend == "cuda"
    if not cuda and (available_bytes_fn is not None or reserve_bytes or host_available_bytes_fn is not None or host_reserve_bytes):
        raise BridgeError("hand_back reader/reserve arguments apply only to a CUDA lineage (a meaningless CPU argument is refused)")
    q = None
    if cuda:
        from maple_syrup import floating_event_daily_bridge_cuda as bc

        q = bc.wet_quota(closed.ctx.column, available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes)
    return lc.rebase_boundary_from_day(lineage, closed.ctx.hold, closed, cuda_quota=q)


def _need_bridge_state(state: Any) -> None:
    if type(state) is not fd.FloatingDailyState or state not in fd._MINTED or state.ctx.profile != BRIDGE_PROFILE:
        raise BridgeError("not a genuine bridge-profile FloatingDailyState")


def storm_schedule(forcing: Any) -> Any:
    """The native Pond `RainfallSchedule` of a bound full storm: the physical (absolute) edges `[t0, end]` at the plan's intensity (the PondEvent schedule is absolute; the `BoundForcing.schedule` is the local one)."""
    return constant_rainfall(float(forcing.t0_s), float(forcing.end_s), float(forcing.intensity_mm_h))


def _exact_dt(t: float, target: float) -> float:
    """The fragment length `dt` with `t + dt == target` EXACTLY in FP64 (the accepted snap contract: `t_end` is reached to the bit by the native `t + dt`, never by restamping): `target - t` first, then the neighbouring
    representable lengths."""
    dt = target - t
    for _ in range(8):
        if t + dt == target:
            return dt
        dt = math.nextafter(dt, math.inf if t + dt < target else -math.inf)
    raise BridgeError(f"no representable fragment length reaches t_end = {target!r} exactly from t = {t!r}")


def _cuda_args(column: Any, host_available_bytes_fn: Any, host_reserve_bytes: Any) -> dict:
    """The CUDA host-pool arguments of the public day entries (the dry day's `floating_daily_action` exposes exactly the host reader and reserve; no budget is invented here): forwarded for a CUDA column; for a CPU column ANY
    non-default value is a typed refusal (never silently stripped)."""
    if column.backend == "cuda":
        return {"host_available_bytes_fn": host_available_bytes_fn, "host_reserve_bytes": host_reserve_bytes}
    if host_available_bytes_fn is not None or host_reserve_bytes:
        raise BridgeError("host-pool arguments apply only to a CUDA lineage (a meaningless CPU argument is refused)")
    return {}


def _bound_cover(parent: Any, cover: Any) -> Any:
    """The day's E-shading cover. Without an applied vegetation generation (`hold.vegetation is None`) `cover` is returned UNCHANGED (the caller-prescribed historical path). With one, the bound live-shading cover is
    used: `cover=None` (or the bound array itself) selects it; any other caller cover is refused (never silently overrides the accepted profile)."""
    veg = getattr(parent.ctx.hold, "vegetation", None)
    if veg is None:
        return cover
    from maple_syrup import vegetation_lifecycle as vl

    vl.verify_record(veg)  # genuine, unaltered record (O(1) fingerprint)
    if cover is None or cover is veg.bound_shade:
        return veg.bound_shade
    raise BridgeError("this lineage has an applied vegetation generation: the evaporation-shading cover is bound; pass cover=None (a different caller cover is refused)")


def day_action(parent: Any, *, plan: Any, pet_terrain_m_s: Any, cover: Any, available_bytes_fn: Any, reserve_bytes: int = 0, event_factory: Any = None, day_control: Any = None,
               host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0) -> fd.DailyResult:
    """The day's action from the closed bridge state: a DRY boundary is the accepted soil-only action (`floating_daily_action`); a boundary with standing water is the approved wet-interevent day (`wet_day_action`). The
    CUDA host reader/reserve reach BOTH branches (a CPU lineage refuses them)."""
    _need_bridge_state(parent)
    cover = _bound_cover(parent, cover)
    cq = _cuda_args(parent.ctx.column, host_available_bytes_fn, host_reserve_bytes)
    if parent.surface is None:
        return fd.floating_daily_action(parent, expected_parent=fd.expect_parent(parent), plan=plan, pet_terrain_m_s=pet_terrain_m_s, cover=cover, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, **cq)
    return wet_day_action(parent, plan=plan, pet_terrain_m_s=pet_terrain_m_s, cover=cover, event_factory=event_factory, day_control=day_control, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, **cq)


def wet_day_action(parent: Any, *, plan: Any, pet_terrain_m_s: Any, cover: Any, event_factory: Any, available_bytes_fn: Any, reserve_bytes: int = 0, day_control: Any = None,
                   after_storm: bool = False, host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0) -> fd.DailyResult:
    """The approved WET interevent day (standing surface water on the boundary): short hydrology/E fragments totalling EXACTLY one 86400 s day, each the accepted stage order preferential -> matrix(H0, zero rain) -> evaporation
    (surface E on the remaining pool + the post-cap soil share) -> ONE FSL gravity/diffusion step -> the saturation return rejoins the post-E pool -> the accepted surface router (`_FloatingSolver.step_day`), on the carried
    routing topology (no rebuild, no sediment). Fragment lengths follow the accepted adaptive contract (halve from the UNCHANGED fragment on `HydraulicStepRejected`, grow after clean steps, exact snap to `t_end`);
    `max_steps`, `max_retries` and the retry floor refuse the WHOLE candidate (never a completion). `plan` must be a P = 0 day: a wet boundary with positive rain requires the FULL native storm from the unchanged parent
    (explicit bounded eligibility, no intensity threshold, no surface erasure). T is NOT applied here (once, at the day's end, through `request_floating_withdrawal`)."""
    _need_bridge_state(parent)
    cover = _bound_cover(parent, cover)
    if parent.surface is None or parent.role != "boundary":
        raise BridgeError("wet_day_action needs a closed bridge boundary state with standing surface water")
    ctx, hold = parent.ctx, parent.ctx.hold
    fd._check_parent(parent, fd.expect_parent(parent), "boundary")
    branch = "no_rain"
    try:
        rc.verify_plan(plan)
        if after_storm:  # the day of a MATCHED, genuinely completed full native storm: the ORIGINAL pending plan is consumed once at the close
            pend = parent.cursor.pending
            if pend is None or getattr(hold, "storm_match", None) != pend.forcing_digest:
                raise rc.RechargeError("after_storm needs the pending storm to have been completed as a MATCHED full native storm (the lifecycle's recorded forcing match)")
            parent.cursor.check_for_consume(plan, pend.branch)
            branch = pend.branch
        else:
            parent.cursor.check(plan)
    except (rc.RechargeError, AttributeError) as exc:
        raise fd.DailyRefusal("occurrence", str(exc)) from None
    t0 = parent.t_s
    if plan.selection != "no_rain" and not after_storm:
        try:
            forcing = plan.bind_forcing(t0)
        except rc.RechargeError as exc:
            raise fd.DailyRefusal("forcing", str(exc)) from None
        return fd.DailyResult("requires_full_storm", None, plan.digest, "storm", forcing, "wet boundary: positive rain requires the FULL native storm from the unchanged parent (bounded eligibility of this bridge profile)")
    fd._validate_inputs(ctx, pet_terrain_m_s, cover)
    cuda = ctx.column.backend == "cuda"
    if not callable(event_factory) or not callable(available_bytes_fn):
        raise BridgeError("event_factory and available_bytes_fn must be callable")
    vkw: dict = {}
    comp: dict = {}
    if cuda:
        from maple_syrup import floating_event_daily_bridge_cuda as bc

        # CUDA: the CURRENT device FIRST, the strict two-pool quota (typed), then ONE composed admission of this transaction's own scalar/probe working set BEFORE any device operation; the SAME readers/reserve then reach
        # the vertical (from_boundary -> every stage: FE/FM/FPI/FSL/router) and every scalar query: nothing falls back to a default pool
        q = bc.wet_quota(ctx.column, available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes)
        bc.admit_wet(ctx.column, q)
        comp = bc.fdc.component_quota(q)
        vkw = {"host_available_bytes_fn": host_available_bytes_fn, "host_reserve_bytes": host_reserve_bytes}
        inputs_ok = bc.inputs_ok_cuda(ctx.column, pet_terrain_m_s, cover)  # ONE device reduction packet
    else:
        if host_available_bytes_fn is not None or host_reserve_bytes:
            raise BridgeError("host-pool arguments apply only to a CUDA lineage (a meaningless CPU argument is refused)")
        pet_a, cov_a = np.asarray(pet_terrain_m_s), np.asarray(cover)
        inputs_ok = bool(np.isfinite(pet_a).all() and (pet_a >= 0.0).all() and np.isfinite(cov_a).all() and (cov_a >= 0.0).all() and (cov_a <= 1.0).all())
    if not inputs_ok:
        raise fd.DailyRefusal("input", "pet_terrain_m_s must be finite and >= 0 and cover finite in [0, 1] on every cell (refused before any runtime or mutation)")
    vert, event, hyd0, solver = lc.day_runtime(hold, parent, event_factory, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, **vkw)
    ctl = (event.control.time if day_control is None else day_control).validated()
    area = float(ctx.area)
    soil0 = float(fsl.floating_storage_m3(ctx.column, parent.water, **comp))
    surf0 = (bc.sum_cuda(ctx.column, hyd0.depth_m) if cuda else float(np.asarray(hyd0.depth_m).sum())) * area
    t_end = float(parent.ledger.expected_t_s) + fd.DAY_S
    hyd = hyd0
    t = float(hyd.t_s)
    es = eso = drain = outlet = bound = 0.0
    n_terms = n_steps = n_rej = 0
    dt_cap = float(ctl.max_dt_s)
    while t < t_end:
        remaining = t_end - t
        dt = min(remaining, dt_cap)
        snap = dt == remaining
        retries = 0
        while True:
            if n_steps >= ctl.max_steps:
                raise BridgeError(f"the wet day reached max_steps = {ctl.max_steps} at t = {t!r} s before t_end = {t_end!r} s (the whole candidate is refused; this is not a completion)")
            if snap:
                dt = _exact_dt(t, t_end)
            if not t + dt > t:
                raise BridgeError(f"floating time does not advance at t = {t!r} s (dt = {dt!r})")
            try:
                step = solver.step_day(hyd, dt, ctx.evaporation, pet_terrain_m_s, cover)
            except (fsl.FloatingSoilError, ss.SubsurfaceError) as exc:
                raise fd.DailyRefusal("step_refused", f"the wet day's fragment was refused: {exc}") from None
            except HydraulicStepRejected as exc:
                retries += 1
                n_rej += 1
                if retries > ctl.max_retries or dt * 0.5 < ctl.min_dt_s:
                    raise BridgeError(f"the wet day's step at t = {t!r} s was rejected {retries} times (limits max_retries = {ctl.max_retries}, min_dt_s = {ctl.min_dt_s}); the whole candidate is refused: {exc}") from None
                dt, snap = dt * 0.5, False
                continue
            break
        info = solver.last_day
        b = info["soil_budget"]
        es += (float(info["surface_evaporation_m3"]) if cuda else float(info["surface_evaporation_m"].sum()) * area)
        eso += float(b.evaporation_m3)
        drain += float(b.drainage_m3)
        outlet += float(info["export_m3"])
        bound += float(b.bound_m3)
        n_terms += int(b.n_terms)
        n_steps += 1
        t = t_end if snap else t + dt
        hyd = replace(step.state, t_s=t)
        if retries:
            dt_cap = dt
        elif dt >= dt_cap:
            dt_cap = min(float(ctl.max_dt_s), 2.0 * dt_cap)
    payload = vert.current_payload(hyd.soil_water_m)
    if float(payload.state.t_s) != t_end:
        raise fd.DailyRefusal("step_refused", f"the day's soil clock {payload.state.t_s!r} s is not the accepted clock + one day {t_end!r} s (no restamp)")
    soil1 = float(fsl.floating_storage_m3(ctx.column, payload.state, **comp))
    xp = vert.xp
    depth1 = xp.array(hyd.depth_m, copy=True)
    surf1 = (bc.sum_cuda(ctx.column, depth1) if cuda else float(depth1.sum())) * area
    depth_ok = bc.depth_ok_cuda(ctx.column, depth1) if cuda else bool(np.isfinite(depth1).all() and (depth1 >= 0.0).all())
    if not depth_ok:
        raise fd.DailyRefusal("step_refused", "the wet day left a non-finite or negative surface depth")
    e_total = es + eso
    resid = (soil1 + surf1) - (soil0 + surf0) + e_total + drain + outlet
    day_bound = bound + volume_roundoff_bound_m3(n_terms + 2 * ctx.n_layers * ctx.n_active + 8, max(abs(soil0 + surf0), abs(soil1 + surf1), e_total, drain, outlet, 1e-30))
    if not (math.isfinite(resid) and abs(resid) <= day_bound):
        raise fd.DailyRefusal("step_refused", f"wet-day closure violated: residual {resid!r} m3 exceeds the MAPLE bound {day_bound!r} m3 (surface + soil; E, drainage and outlet once)")
    led = parent.ledger
    new_led = fd.DailyLedger(led.days_applied + 1, led.rain_m3, led.evaporation_m3 + e_total, led.drainage_m3 + drain, led.transpiration_m3, 0.0, e_total, drain, 0.0, resid, day_bound, led.initial_storage_m3,
                             led.bound_m3 + day_bound, led.n_terms + n_terms, t_end, soil0 + surf0, day_bound, n_terms, surface_m3=surf1, outlet_m3=led.outlet_m3 + outlet, last_day_outlet_m3=outlet)
    qx, qy = xp.array(hyd.qx_m2_s, copy=True), xp.array(hyd.qy_m2_s, copy=True)
    child = fd._mint_state(ctx, "interevent", payload.state, payload.history, parent.cursor, (plan, branch), parent.digest, parent.revision + 1, 0, new_led, parent.execution_digest, parent.digest,
                           {"kind": "wet_day", "plan": plan.digest, "branch": branch, "steps": n_steps, "rejected": n_rej, "dt": fd.DAY_S}, (depth1, qx, qy))
    return fd.DailyResult("interevent", child, plan.digest, branch)


def begin_storm_from_day(lineage: Any, daily_parent: Any, plan: Any, *, pet_terrain_m_s: Any, cover: Any, available_bytes_fn: Any, reserve_bytes: int = 0, event_factory: Any = None, host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0) -> lc.Boundary:
    """Start the FULL native storm of `plan` from the UNCHANGED closed day parent. The outcome is NOT taken from a caller: the bridge re-runs the accepted day action itself (dry: `floating_daily_action`; wet: the P > 0 eligibility) and
    requires `requires_full_storm`; its `plan`/forcing/branch are then bound onto the cursor (ORIGINAL plan/digest/branch, ordinal not advanced) together with the FULL native `BoundForcing`, and the unchanged state is handed
    back as a Boundary carrying them. The caller runs the genuine storm with `lc.begin_occurrence` and an event whose native schedule is `storm_schedule(forcing)`: the lifecycle refuses a mismatching schedule / clock /
    `rain_scale` BEFORE any head change and only a matched genuinely complete storm records the match that `day_after_storm` needs."""
    _need_bridge_state(daily_parent)
    res = day_action(daily_parent, plan=plan, pet_terrain_m_s=pet_terrain_m_s, cover=cover, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, event_factory=event_factory,
                     host_available_bytes_fn=host_available_bytes_fn, host_reserve_bytes=host_reserve_bytes)
    if res.outcome != "requires_full_storm" or res.plan_digest != plan.digest or res.forcing is None:
        raise BridgeError("the day action did not require a full native storm for this plan")
    if daily_parent.role != "boundary" or daily_parent.cursor.pending is not None:
        raise BridgeError("the storm starts from an unchanged closed day parent with nothing pending")
    try:
        cursor = daily_parent.cursor.begin_storm(plan, res.forcing, fallback=(res.branch == "storm_fallback"), t0_s=daily_parent.t_s)
    except rc.RechargeError as exc:
        raise BridgeError(str(exc)) from None
    return lc.rebase_boundary_from_day(lineage, daily_parent.ctx.hold, daily_parent, cursor=cursor, pending_forcing=res.forcing)


def day_after_storm(state: Any, *, plan: Any, pet_terrain_m_s: Any, cover: Any, available_bytes_fn: Any, reserve_bytes: int = 0, event_factory: Any = None, day_control: Any = None,
                    host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0) -> fd.DailyResult:
    """The day's update after a MATCHED, genuinely completed full native storm (the hold must carry the recorded forcing match of exactly the cursor's pending storm: a cursor alone is not proof): the accepted soil-only
    rain-0 action for a dry boundary, the wet-interevent fragments (`wet_day_action(after_storm=True)`) when standing water remains. The pending occurrence is consumed ONCE at the final close with the ORIGINAL plan. No
    E/T ran during the storm or its recession; the physical clock advances one day on top (nothing is caught up)."""
    _need_bridge_state(state)
    cover = _bound_cover(state, cover)
    cq = _cuda_args(state.ctx.column, host_available_bytes_fn, host_reserve_bytes)
    if state.surface is not None:
        return wet_day_action(state, plan=plan, pet_terrain_m_s=pet_terrain_m_s, cover=cover, event_factory=event_factory, day_control=day_control, available_bytes_fn=available_bytes_fn,
                              reserve_bytes=reserve_bytes, after_storm=True, **cq)
    return fd.floating_daily_action(state, expected_parent=fd.expect_parent(state), plan=plan, pet_terrain_m_s=pet_terrain_m_s, cover=cover, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes,
                                    after_storm=True, **cq)
