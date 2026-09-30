"""Phase 5b: one coupled rainfall / infiltration / routing / wet-sediment
event on the ACTUAL MAPLE bed (pure helpers; the Plot 1 runner is
sediment_experiment.py).

State ownership
---------------
One authority for sediment and water: `sediment_bed.BedState` holds the
MAPLE voxel column, active layer, availability, ledger, committed
topography and `WaterState` (depth + water-borne mobile mass). SYRUP owns
the hydraulic depth, soil water and discharge (`storm.StormState`), the
transport memory (sediment velocity, `(ny, nx, nc)`) and the routing
graph / transport network / physics grid derived from the committed
terrain. No copy of the bed or of the terrain evolves on its own: every
mass change goes through `apply_water_process_demand` and every terrain
change through `commit_topography` (both composed by `sediment_bed`).

One accepted step of length `dt` (all local until published)
--------------------------------------------------------------
1. `storm.coupled_step`: accepted Phase 3 column + Phase 4b method-5
   routing on the ORIGINAL storm state (may raise `RoutingStepRejected`).
2. `sediment_physics_step` on the routed depth and velocity, the step's
   rain rate, the prescribed vegetation cover, the CURRENT active-layer
   holdings (pre-pickup) and the velocity memory -> pickup demand over
   `dt`, sediment velocity, deposition rate `1/L`, settle mask.
3. Publish the solver depth into the MAPLE water state; PICKUP call:
   `apply_water_process_demand` with the demand only (MAPLE caps by
   availability and holdings, refills the active layer from the column
   and moves the ACTUAL removal into the same cell's mobile pool).
4. `transport_step` on the actual post-pickup pool (exact `exp(-v dt/L)`
   reaction half-steps around upwind advection; substeps chosen from the
   sediment Courant number, see below). Newly picked-up mass is
   transported in the same step (no source lag).
5. Publish `WaterState(depth, T(M))`; DEPOSIT/EXPORT call:
   `apply_water_process_demand` with the transport's deposition and
   export requests and face crossings (removal zero). Dry / no-capacity
   cells request their whole pool, so mobile mass reaching a dry cell
   returns to the MAPLE bed; wet residual mobile mass is retained.
6. Commit: `sediment_bed.commit_bed` evaluates MAPLE's own triggers on the
   ledger. When a commit fires, MAPLE publishes the terrain under
   `constant_depth` (depth array-equal, displaced volume exactly 0), the
   routing graph is rebuilt from the newly committed surface with the
   fixed boundary ring, the transport network and physics grid are
   rebuilt from that graph, and the discharge is re-initialised from the
   NEW conveyance and the UNCHANGED depth (`storm.initial_state`: an
   instantaneous kinematic adjustment with no water-volume change). A
   graph failure (new pit, lost receiver) raises before anything is
   published.
7. Only now are time, the storm state, the bed, the velocity memory, the
   graph objects and every accumulator published.

Retry policy: `RoutingStepRejected` (water Courant / negative RHS) and
`TransportStepRejected` (sediment Courant beyond `max_transport_substeps`
substeps) both discard the whole unpublished attempt and halve `dt` from
the SAME state, bounded by `max_retries` and the retry floor `min_dt_s`
(a forced slice to a forcing / reporting boundary may be shorter; halving
below the floor is refused). Sediment substeps are chosen from the
maximum sediment velocity, `ceil(v_max dt / (dx courant_max))`, and
doubled on an FP-boundary rejection. Every other exception propagates
with nothing published.

Accounting (device-resident, bounded)
--------------------------------------
Gross pickup, deposition and export are accumulated per cell and class
from MAPLE's ground-truth results (never from requests); requested,
availability-refused, holdings-refused and numerically-residual amounts
are accumulated per class; the transport operator's own per-class budget
residual and tolerance are summed; the ledger's per-process totals are
captured from every commit before MAPLE resets them; true per-step peaks
(mobile mass, export rate, outlet discharge, depth, velocity) are tracked
independently of the reporting cadence; the water hydrograph keeps the
Phase 4 columns and a sediment hydrograph adds per-class mobile, export
and cumulative pickup / deposition rows. The global closure
`bed_initial + mobile_initial = bed_final + mobile_final + export` per
class is derived from `bed_inventory` (MAPLE's helper) at the two ends
and the accumulated actual export -- independently of the ledger.

Not here: splash, ecology, nutrients, an event-stop criterion, the dry
reset, wind alternation, restart. The run ends at the configured time.
GPU: every loop operation is a namespace operation, but a commit rebuilds
the graph on the host (`sediment_bed.refresh_routing` transfers the
committed elevation) and the numba hydraulics are CPU-only; no GPU event
has been executed and the runner refuses `backend != "numpy"`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from maple_syrup.infiltration import ColumnParameters
from maple_syrup.rainfall import RainfallField, RainfallSchedule
from maple_syrup.routing import RoutingGraph, RoutingStepRejected
from maple_syrup.sediment_bed import (
    BedContext,
    BedIntegrationError,
    BedState,
    TerrainReference,
    apply_bed_demand,
    bed_inventory,
    commit_bed,
    with_water,
)
from maple_syrup.sediment_physics import (
    REGIME_CODES,
    PhysicsGrid,
    SedimentPhysicsParameters,
    SedimentPhysicsStep,
    physics_grid_from_graph,
    sediment_physics_step,
)
from maple_syrup.sediment_transport import (
    TransportNetwork,
    TransportStep,
    TransportStepRejected,
    transport_network,
    transport_step,
    water_demand_from_transport,
)
from maple_syrup.storm import (
    HYDROGRAPH_COLUMNS,
    CoupledStep,
    StormControl,
    StormError,
    StormState,
    coupled_step,
    initial_state,
    plan_boundaries,
)
from maple_syrup.storm import _validate_state as validate_storm_state

__all__ = [
    "SEDIMENT_SCALAR_COLUMNS",
    "SEDIMENT_VECTOR_COLUMNS",
    "SedimentCoupledStep",
    "SedimentEventControl",
    "SedimentEventError",
    "SedimentEventResult",
    "SedimentEventState",
    "evolve_sediment_event",
    "initial_event_state",
    "morphology_summary",
    "sediment_coupled_step",
    "sediment_hydrograph_columns",
]

_EPS = float(np.finfo(np.float64).eps)
# FP64 roundings allowed per accumulated term in the closure tolerance.
_CLOSURE_ROUNDINGS = 64.0

# Sediment hydrograph: scalar columns, then per-class blocks (each block
# has nc columns, suffixed `_c1..c<nc>` in canonical class order).
SEDIMENT_SCALAR_COLUMNS = (
    "t_s",
    "mobile_kg",
    "cumulative_pickup_kg",
    "cumulative_deposition_kg",
    "cumulative_export_kg",
    "export_rate_kg_s",  # last accepted step's actual export / dt
    "commit_count",
    "graph_changes",
    "max_transport_substeps",
)
SEDIMENT_VECTOR_COLUMNS = (
    "mobile",
    "export_rate",
    "cumulative_pickup",
    "cumulative_deposition",
    "cumulative_export",
)


def sediment_hydrograph_columns(n_classes: int) -> tuple[str, ...]:
    names = list(SEDIMENT_SCALAR_COLUMNS)
    for block in SEDIMENT_VECTOR_COLUMNS:
        unit = "kg_s" if block == "export_rate" else "kg"
        names += [f"{block}_c{k + 1}_{unit}" for k in range(n_classes)]
    return tuple(names)


class SedimentEventError(RuntimeError):
    """Configuration, validation, guard or consistency failure of the
    coupled sediment event. Nothing is published and no input array is
    modified."""


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SedimentEventError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _strict_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise SedimentEventError(f"{name} must be an int >= {minimum}, got {value!r}")
    return int(value)


# --- control -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class SedimentEventControl:
    """Hydraulic control (`storm`) plus the sediment / commit controls.

    `sediment_courant_max` bounds `v dt / (n_substeps dx)` for the
    transport operator (explicit upwind positivity requires <= 1);
    `max_transport_substeps` caps the substep count before the whole step
    is halved instead. `commit` evaluates MAPLE's commit triggers after
    every accepted step; `force_final_commit` commits any pending bed
    change at the configured end time (real time, no fabricated elapsed
    time) so the returned terrain corresponds to the returned bed.
    """

    storm: StormControl = field(default_factory=StormControl)
    sediment_courant_max: float = 1.0
    max_transport_substeps: int = 64
    commit: bool = True
    force_final_commit: bool = True
    max_commit_log: int = 256
    max_rejections_recorded: int = 100

    def validated(self) -> SedimentEventControl:
        if not isinstance(self.storm, StormControl):
            raise SedimentEventError("storm must be a StormControl")
        self.storm.validated()
        cr = _real(self.sediment_courant_max, "sediment_courant_max")
        if not (0.0 < cr <= 1.0):
            raise SedimentEventError(f"sediment_courant_max must lie in (0, 1], got {cr!r}")
        _strict_int(self.max_transport_substeps, "max_transport_substeps")
        _strict_int(self.max_commit_log, "max_commit_log", 0)
        _strict_int(self.max_rejections_recorded, "max_rejections_recorded", 0)
        for name in ("commit", "force_final_commit"):
            if not isinstance(getattr(self, name), bool):
                raise SedimentEventError(f"{name} must be bool")
        return self


# --- state ----------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SedimentEventState:
    """Everything an accepted step publishes. `storm.depth_m` is the
    solver depth and equals `bed.water.depth_m` after every accepted step;
    `graph`, `network` and `grid` are derived from the committed terrain
    held in `bed.committed_topography` (rebuilt at every commit)."""

    storm: StormState
    bed: BedState
    sediment_velocity_m_s: Any  # (ny, nx, nc) transport memory
    graph: RoutingGraph
    network: TransportNetwork
    grid: PhysicsGrid
    terrain: TerrainReference

    @property
    def t_s(self) -> float:
        return self.storm.t_s


def _check_event_inputs(state: SedimentEventState, context: BedContext, column: ColumnParameters,
                        sediment: SedimentPhysicsParameters, root_tolerance_m: float) -> None:
    from maple.core.backend import (
        DeferredChecks,
        MixedArrayNamespaceError,
        array_namespace,
        finite_flag,
        is_array,
        negative_flag,
        true_flag,
    )

    graph = state.graph
    xp, shape = graph.xp, graph.shape
    g = context.geometry
    nc = len(context.grain_classes.classes)
    if (g.ny, g.nx) != shape or g.dx_m != g.dy_m or g.dx_m != graph.dx_m:
        raise SedimentEventError("bed geometry, routing graph shape and spacing must agree (square cells)")
    if context.depth_update_rule != "constant_depth":
        raise SedimentEventError("the event requires the constant_depth ownership rule (SYRUP owns depth)")
    if not isinstance(state.network, TransportNetwork) or state.network.graph_input_sha256 != graph.input_sha256:
        raise SedimentEventError("transport network does not belong to the current routing graph")
    if not isinstance(state.grid, PhysicsGrid) or state.grid.shape != shape or state.grid.xp is not xp:
        raise SedimentEventError("physics grid does not match the routing graph")
    if sediment.n_classes != nc or sediment.xp is not xp:
        raise SedimentEventError("sediment parameters must have the bed's class count and the graph namespace")
    diameters = np.array([c.diameter_m for c in context.grain_classes.classes], dtype=np.float64)
    if not np.array_equal(diameters, sediment.diameter_m):
        raise SedimentEventError("sediment parameter diameters differ from the MAPLE grain classes")
    densities = {float(c.particle_density_kg_m3) for c in context.grain_classes.classes}
    if densities != {float(sediment.particle_density_kg_m3)}:
        raise SedimentEventError("sediment particle density differs from the MAPLE grain classes")
    if state.terrain.initial_committed_elevation_m.shape != shape or state.terrain.dx_m != graph.dx_m:
        raise SedimentEventError("terrain reference does not match the routing graph")
    validate_storm_state(graph, state.storm, root_tolerance_m, column)
    v = state.sediment_velocity_m_s
    if not is_array(v) or tuple(v.shape) != (*shape, nc) or v.dtype != np.float64:
        raise SedimentEventError(f"sediment_velocity_m_s must be float64 {(*shape, nc)}")
    try:
        namespace = array_namespace(v, state.bed.water.depth_m, state.bed.active_layer.mass_kg, graph.conveyance)
    except MixedArrayNamespaceError as exc:
        raise SedimentEventError(f"event arrays are in mixed namespaces: {exc}") from None
    if namespace is not xp:
        raise SedimentEventError(f"event arrays must be in the graph namespace {xp.__name__!r}")
    checks = DeferredChecks()
    checks.require(finite_flag(v), "sediment_velocity_m_s must be finite")
    checks.forbid(negative_flag(v), "sediment_velocity_m_s must be >= 0")
    checks.forbid(true_flag(state.bed.water.depth_m != state.storm.depth_m),
                  "bed.water.depth_m must equal the solver depth storm.depth_m")
    active3 = graph.active_flat.reshape(shape)[..., None]
    checks.forbid(true_flag(~active3 & (state.bed.water.mobile_mass_by_cell_class_kg != 0.0)),
                  "mobile mass on an inactive cell")
    try:
        checks.resolve()
    except ValueError as exc:
        raise SedimentEventError(str(exc)) from None


def initial_event_state(graph: RoutingGraph, terrain: TerrainReference, bed: BedState, context: BedContext,
                        soil_water_m: Any, *, t_s: float = 0.0) -> SedimentEventState:
    """Event state at `t_s` from a routing graph, its terrain reference,
    a MAPLE bed (its `water.depth_m` is the initial solver depth, its
    mobile mass the initial load) and the retained soil water. Discharge
    is initialised from the depth (`storm.initial_state`); the transport
    memory starts at zero (no prior sediment motion)."""
    nc = len(context.grain_classes.classes)
    storm = initial_state(graph, bed.water.depth_m, soil_water_m, t_s=t_s)
    velocity = graph.xp.zeros((*graph.shape, nc), dtype=np.float64)
    return SedimentEventState(storm=storm, bed=bed, sediment_velocity_m_s=velocity, graph=graph,
                              network=transport_network(graph), grid=physics_grid_from_graph(graph),
                              terrain=terrain)


# --- one coupled attempt ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SedimentCoupledStep:
    """One accepted (not yet committed / published) coupled attempt."""

    dt_s: float
    storm: CoupledStep
    physics: SedimentPhysicsStep
    pickup: Any  # MAPLE WaterProcessResult of the pickup call
    transport: TransportStep
    deposit: Any  # MAPLE WaterProcessResult of the deposit / export call
    bed: BedState  # after both MAPLE calls; water = (routed depth, mobile after deposit / export)
    n_substeps: int
    n_transport_rejections: int


def sediment_coupled_step(
    state: SedimentEventState,
    context: BedContext,
    column: ColumnParameters,
    rain_rate_m_per_s: Any,
    vegetation_cover_fraction: Any,
    sediment: SedimentPhysicsParameters,
    dt_s: float,
    control: SedimentEventControl,
    *,
    zero_demand: Any = None,
) -> SedimentCoupledStep:
    """Steps 1-5 of the module docstring on the ORIGINAL `state`. Pure:
    every MAPLE call returns new objects and nothing is published; raises
    `RoutingStepRejected` / `TransportStepRejected` when the attempt must
    be retried with a smaller `dt`, and propagates every other failure."""
    from maple.core.backend import to_float
    from maple.water import WaterProcessDemand

    xp = state.graph.xp
    nc = sediment.n_classes
    dt = float(dt_s)
    zeros = xp.zeros((*state.graph.shape, nc), dtype=np.float64) if zero_demand is None else zero_demand

    # 1. water: column + routing (accepted Phase 4 kernels).
    storm = coupled_step(state.graph, column, rain_rate_m_per_s, state.storm, dt, control.storm)
    route = storm.route
    # 2. wet laws on the routed depth / velocity and the CURRENT holdings.
    physics = sediment_physics_step(sediment, state.grid, route.depth_m, route.velocity_m_s, rain_rate_m_per_s,
                                    vegetation_cover_fraction, state.bed.active_layer.mass_kg,
                                    state.sediment_velocity_m_s, dt)
    # 3. publish the solver depth; actual pickup through MAPLE.
    bed = with_water(state.bed, context, route.depth_m)
    bed, pickup = apply_bed_demand(bed, context, WaterProcessDemand(physics.requested_pickup_kg, zeros))
    # 4. transport of the ACTUAL post-pickup pool; substeps from the sediment Courant number.
    velocity = physics.sediment_velocity_m_s
    v_max = to_float(xp.max(velocity))
    dx = state.network.dx_m
    n_sub = max(1, math.ceil(v_max * dt / (dx * control.sediment_courant_max)))
    rejections = 0
    while True:
        if n_sub > control.max_transport_substeps:
            raise TransportStepRejected(
                f"sediment Courant number needs {n_sub} substeps (max_transport_substeps = "
                f"{control.max_transport_substeps}) at dt = {dt} s; step rejected (retry with a smaller dt)")
        try:
            transport = transport_step(state.network, bed.water.mobile_mass_by_cell_class_kg, velocity,
                                       physics.deposition_rate_per_m, physics.settle_mask, dt,
                                       courant_max=control.sediment_courant_max, n_substeps=n_sub)
            break
        except TransportStepRejected:
            rejections += 1
            n_sub *= 2
    # 5. publish T(M); actual deposition / export through MAPLE.
    bed = with_water(bed, context, route.depth_m, transport.mobile_after_transfer_kg)
    bed, deposit = apply_bed_demand(bed, context, water_demand_from_transport(transport, zeros))
    return SedimentCoupledStep(dt_s=dt, storm=storm, physics=physics, pickup=pickup, transport=transport,
                               deposit=deposit, bed=bed, n_substeps=n_sub, n_transport_rejections=rejections)


# --- commit and reroute ------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _Commit:
    bed: BedState
    graph: RoutingGraph
    network: TransportNetwork
    grid: PhysicsGrid
    storm: StormState
    record: dict[str, Any]
    maple_result: Any


def _commit_and_reroute(bed: BedState, context: BedContext, graph: RoutingGraph, terrain: TerrainReference,
                        storm: StormState, t_s: float, *, force: bool) -> _Commit | None:
    """MAPLE commit through `sediment_bed.commit_bed`; on a commit verify
    the depth policy and the water volume, rebuild network / grid from the
    refreshed graph and re-initialise the discharge from the new
    conveyance and the unchanged depth. Nothing is published here."""
    from maple.core.backend import to_float, to_host

    outcome = commit_bed(bed, context, graph, terrain, t_s, force=force)
    if not outcome.did_commit:
        return None
    depth_update = None if outcome.water_result is None else outcome.water_result.depth_update
    if (depth_update is None or depth_update.rule != "constant_depth" or depth_update.displaced_volume_m3 != 0.0
            or depth_update.clamp_added_volume_m3 != 0.0 or depth_update.clamp_cell_count != 0):
        raise SedimentEventError(f"commit at t = {t_s} s changed the water volume or depth policy: {depth_update}")
    new_graph = outcome.graph
    if new_graph.shape != graph.shape or new_graph.dx_m != graph.dx_m or new_graph.xp is not graph.xp:
        raise SedimentEventError("refreshed routing graph changed shape, spacing or namespace")
    if not np.array_equal(new_graph.active, graph.active):
        raise SedimentEventError("refreshed routing graph changed the active cells")
    # The exporting boundary ring and the active set are fixed; WHICH interior
    # cells drain into the ring (the outlet set) may legitimately change with
    # the terrain. `build_routing_graph` has already refused pits, flats and
    # receivers outside the active set or the ring, so a valid reroute is
    # accepted and reported; the transport network is rebuilt on the new
    # outlets below and the caller refreshes its outlet-based diagnostics.
    # `graph_changed` means a PHYSICAL change (slope or receiver); a differing
    # `input_sha256` alone (`graph_rebound`) can come from the refresh omitting
    # the initial graph's nodata metadata and is only the binding.
    geometry_changed = not (np.array_equal(new_graph.aspect, graph.aspect)
                            and np.array_equal(new_graph.slope, graph.slope))
    # Discharge from the NEW conveyance and the UNCHANGED depth (kinematic
    # adjustment, no volume change); validates depth / soil water again.
    new_storm = initial_state(new_graph, outcome.state.water.depth_m, storm.soil_water_m, t_s=t_s)
    result = outcome.maple_result
    trigger = outcome.trigger
    record = {
        "t_s": t_s,
        "forced": bool(force),
        "commit_count": int(outcome.state.committed_topography.commit_count),
        "elapsed_time_triggered": bool(trigger.elapsed_time_triggered),
        "pending_elevation_triggered": bool(trigger.pending_elevation_triggered),
        "active_layer_turnover_triggered": bool(trigger.active_layer_turnover_triggered),
        "max_pending_elevation_change_m": float(trigger.max_pending_elevation_change_m),
        "max_abs_committed_elevation_change_m": to_float(graph.xp.max(graph.xp.abs(result.committed_elevation_change_m))),
        "rerouted_cells": int(np.sum(new_graph.aspect != graph.aspect)),
        "outlets_changed": int(np.sum(new_graph.outlet != graph.outlet)),
        "n_outlets": int(new_graph.outlet.sum()),
        "graph_changed": geometry_changed,
        "graph_rebound": new_graph.input_sha256 != graph.input_sha256,
        "graph_input_sha256": new_graph.input_sha256,
        "avalanche_applied": bool(result.avalanche_applied),
        "depth_unchanged": True,
        "displaced_water_volume_m3": float(depth_update.displaced_volume_m3),
        "pre_commit_pending_mass_by_class_kg": [float(v) for v in
                                                to_host(result.pre_commit_pending_mass_by_class_kg).tolist()],
        "commit_stage_log": list(result.commit_stage_log),
    }
    return _Commit(bed=outcome.state, graph=new_graph, network=transport_network(new_graph),
                   grid=physics_grid_from_graph(new_graph), storm=new_storm, record=record, maple_result=result)


# --- result ----------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SedimentEventResult:
    """Everything `evolve_sediment_event` accumulated. Grids and 0-d
    values live in the graph namespace; `hydrograph` has the Phase 4
    `HYDROGRAPH_COLUMNS`, `sediment_hydrograph` the columns of
    `sediment_hydrograph_columns(nc)`. `by_class` maps names to `(nc,)`
    device arrays; `commit_log` is bounded by `max_commit_log`."""

    state: SedimentEventState
    hydrograph: Any
    sediment_hydrograph: Any
    sediment_columns: tuple[str, ...]
    boundaries: np.ndarray
    # water
    cumulative_rain_m: Any
    cumulative_intake_m: Any
    cumulative_saturation_return_m: Any
    cumulative_drainage_m: Any
    cumulative_export_m3: Any
    peak_depth_m: Any
    peak_velocity_m_s: Any
    last_velocity_m_s: Any
    peak_outlet_discharge_m3_s: Any
    time_of_peak_outlet_s: Any
    max_courant_old: Any
    max_courant_new: Any
    max_routing_cell_balance_residual_m: Any
    max_constitutive_residual_m: Any
    cell_steps_no_runon: Any
    cell_steps_partial_runon: Any
    cell_steps_complete_runon: Any
    # sediment, per cell and class (actual MAPLE ground truth)
    cumulative_pickup_kg: Any
    cumulative_deposition_kg: Any
    cumulative_export_request_kg: Any  # nonzero only at outlets; equals actual export when nothing was refused
    by_class: dict[str, Any]
    initial_bed_by_cell_class_kg: Any  # (ny, nx, nc): voxel.sum(axis=2) + active at the start
    initial_bed_inventory_kg: Any
    initial_mobile_kg: Any
    numerical_residual_scalar_abs_kg: Any
    peak_mobile_kg: Any
    time_of_peak_mobile_s: Any
    peak_export_rate_kg_s: Any
    time_of_peak_export_s: Any
    max_sediment_courant: Any
    max_decay_exponent: Any
    max_transport_cell_residual_kg: Any
    max_transport_substeps_used: int
    n_transport_rejections: int
    regime_cell_steps: dict[str, Any]
    # commits
    ledger_process_totals_reset_kg: Any  # (n_process, nc): sum of pre-commit totals over commits
    committed_net_bed_change_kg: Any  # (nc,): sum of pre-commit pending mass over commits
    n_commits: int
    n_forced_commits: int
    n_graph_changes: int
    rerouted_cells_total: int
    commit_log: tuple[dict[str, Any], ...]
    # steps
    n_accepted_steps: int
    n_rejected_attempts: int
    rejections: tuple[dict[str, Any], ...]
    min_accepted_dt_s: float
    max_accepted_dt_s: float
    n_maple_water_calls: int
    first_step_wall_s: float
    first_step_cpu_s: float
    remaining_wall_s: float
    remaining_cpu_s: float
    commit_wall_s: float
    n_inventory_terms: int

    def bed_change_by_cell_class_kg(self) -> Any:
        """ACTUAL final minus initial MAPLE bed inventory per cell and class
        (`voxel.sum(axis=2) + active`, robust to the voxel count): the
        morphological ground truth, which includes every bed process MAPLE
        ran (water exchange, avalanching inside commits, sub-resolution
        placement). Contrast `cumulative_deposition_kg -
        cumulative_pickup_kg`, the WATER class exchange only."""
        bed = self.state.bed
        return (bed.voxel_column.mass_kg.sum(axis=2) + bed.active_layer.mass_kg) - self.initial_bed_by_cell_class_kg

    def closure(self) -> dict[str, Any]:
        """Per-class sediment closure from ACTUAL inventories (host):
        `bed_final + mobile_final + export_actual - (bed_initial +
        mobile_initial)` against a declared tolerance (FP64 roundings of
        the inventory sums and the MAPLE water calls plus the reported
        numerical residuals). Also the water-independent request
        reconciliation (requested pickup = actual + refused + residual)."""
        from maple.core.backend import to_host

        bc = {k: np.asarray(to_host(v), dtype=np.float64) for k, v in self.by_class.items()}
        initial_bed = np.asarray(to_host(self.initial_bed_inventory_kg), dtype=np.float64)
        initial_mobile = np.asarray(to_host(self.initial_mobile_kg), dtype=np.float64)
        final_bed = np.asarray(to_host(bed_inventory(self.state.bed)), dtype=np.float64)
        final_mobile = np.asarray(to_host(self.state.bed.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1))),
                                  dtype=np.float64)
        export = bc["export_actual"]
        residual = (final_bed + final_mobile + export) - (initial_bed + initial_mobile)
        scalar = float(to_host(self.numerical_residual_scalar_abs_kg))
        n_terms = self.n_inventory_terms + 2 * self.n_maple_water_calls
        scale = np.maximum(initial_bed + initial_mobile, final_bed + final_mobile + export)
        tolerance = (_CLOSURE_ROUNDINGS * _EPS * n_terms * scale + np.abs(bc["pickup_numerical_residual"])
                     + np.abs(bc["deposit_numerical_residual"]) + scalar)
        request = bc["requested_pickup"] - (bc["actual_pickup"] + bc["availability_shortfall"]
                                            + bc["holdings_shortfall"] + np.abs(bc["pickup_numerical_residual"]))
        request_tol = _CLOSURE_ROUNDINGS * _EPS * n_terms * np.maximum(bc["requested_pickup"], 1.0)
        return {
            "initial_bed_kg": initial_bed.tolist(),
            "initial_mobile_kg": initial_mobile.tolist(),
            "final_bed_kg": final_bed.tolist(),
            "final_mobile_kg": final_mobile.tolist(),
            "export_actual_kg": export.tolist(),
            "net_bed_change_kg": (final_bed - initial_bed).tolist(),
            "residual_kg": residual.tolist(),
            "tolerance_kg": tolerance.tolist(),
            "closed": bool(np.all(np.abs(residual) <= tolerance)),
            "request_reconciliation_residual_kg": request.tolist(),
            "request_reconciliation_tolerance_kg": request_tol.tolist(),
            "request_reconciled": bool(np.all(np.abs(request) <= request_tol)),
            "numerical_residual_scalar_abs_kg": scalar,
            "tolerance_rule": f"{_CLOSURE_ROUNDINGS:g} eps (inventory terms + 2 water calls) max(inventory) "
                              "+ |MAPLE per-class numerical residuals| + |class-unaware scalar residual|",
        }


def morphology_summary(bed_change_kg: Any, water_exchange_kg: Any, *, bulk_density_kg_m3: float,
                       cell_area_m2: float) -> dict[str, Any]:
    """Morphological reporting from the ACTUAL per-cell/class bed change
    (`SedimentEventResult.bed_change_by_cell_class_kg()`), kept separate
    from the water CLASS exchange (`cumulative_deposition - cumulative_pickup`).

    Net erosion / deposition sum the classes PER CELL before taking the
    negative / positive parts, so a balanced compositional exchange (one
    class in, another out, same cell) is zero morphology but nonzero
    `class_sorting_exchange_kg`. `non_water_bed_change_abs_kg` is the part of
    the actual bed change that the water exchange does not explain (MAPLE
    avalanching inside commits, sub-resolution placement); it is measured
    from independent inventories, never re-derived from the exchange."""
    from maple.core.backend import to_host

    change = np.asarray(to_host(bed_change_kg), dtype=np.float64)
    exchange = np.asarray(to_host(water_exchange_kg), dtype=np.float64)
    if change.shape != exchange.shape or change.ndim != 3:
        raise SedimentEventError("bed change and water exchange must both be (ny, nx, nc)")
    cell = change.sum(axis=-1)
    exchange_cell = exchange.sum(axis=-1)
    non_water = change - exchange
    per_m = float(bulk_density_kg_m3) * float(cell_area_m2)
    dz = cell / per_m
    return {
        "basis": "actual final - initial MAPLE bed inventory per cell and class (voxel.sum(axis=2) + active); "
                 "classes summed per cell before splitting erosion / deposition; water class exchange reported "
                 "separately",
        "net_erosion_kg": float(-cell[cell < 0.0].sum()),
        "net_deposition_kg": float(cell[cell > 0.0].sum()),
        "net_bed_change_kg": float(cell.sum()),
        "n_cells_lowered": int(np.sum(cell < 0.0)),
        "n_cells_raised": int(np.sum(cell > 0.0)),
        "elevation_change_equivalent_m": {"min": float(dz.min()), "max": float(dz.max()), "mean": float(dz.mean()),
                                          "rule": "cell mass change / (bulk density x cell area)"},
        "class_bed_change_kg": change.sum(axis=(0, 1)).tolist(),
        "class_water_exchange_kg": exchange.sum(axis=(0, 1)).tolist(),
        "water_exchange_net_erosion_kg": float(-exchange_cell[exchange_cell < 0.0].sum()),
        "water_exchange_net_deposition_kg": float(exchange_cell[exchange_cell > 0.0].sum()),
        # mass swapped between classes within cells without changing the cell mass
        "class_sorting_exchange_kg": float(0.5 * (np.abs(exchange).sum(axis=-1) - np.abs(exchange_cell)).sum()),
        "non_water_bed_change_abs_kg": float(np.abs(non_water).sum()),
        "non_water_bed_change_max_abs_kg": float(np.abs(non_water).max()) if non_water.size else 0.0,
    }


def _row(xp, scalars, vectors):
    parts = [xp.stack([xp.asarray(v, dtype=np.float64) for v in scalars])]
    parts += [xp.asarray(v, dtype=np.float64).reshape(-1) for v in vectors]
    return xp.concatenate(parts)


# --- evolution -------------------------------------------------------------------------------------------
def evolve_sediment_event(
    state: SedimentEventState,
    context: BedContext,
    column: ColumnParameters,
    field: RainfallField,
    schedule: RainfallSchedule,
    vegetation_cover_fraction: Any,
    sediment: SedimentPhysicsParameters,
    end_s: float,
    control: SedimentEventControl,
    *,
    report_every_s: float,
    max_report_rows: int = 100_000,
) -> SedimentEventResult:
    """Advance the event from `state.t_s` to `end_s` (module docstring).
    Raises `SedimentEventError` on validation / guard failures; the two
    recoverable rejections are retried until the guards trip; every other
    exception propagates untouched with nothing published."""
    from maple.core.backend import freeze, to_bool, to_float
    from maple.surface.topographic_commit.commit import is_already_committed

    control = control.validated()
    storm_control = control.storm
    _check_event_inputs(state, context, column, sediment, storm_control.root_tolerance_m)
    graph = state.graph
    xp = graph.xp
    shape = graph.shape
    nc = sediment.n_classes
    area = graph.dx_m * graph.dx_m
    if not isinstance(field, RainfallField) or field.shape != shape or field.xp is not xp:
        raise SedimentEventError("rainfall field must match the graph shape and namespace")
    end = _real(end_s, "end_s")
    boundaries = plan_boundaries(schedule, state.t_s, end, report_every_s, max_report_rows=max_report_rows)
    columns = sediment_hydrograph_columns(nc)

    def zeros(*extra):
        return xp.zeros((*shape, *extra), dtype=np.float64)

    def scalar(value=0.0):
        return xp.asarray(value, dtype=np.float64)

    def vector():
        return xp.zeros((nc,), dtype=np.float64)

    active = graph.active_flat.reshape(shape)
    outlet = graph.outlet_flat.reshape(shape)
    zero_demand = freeze(zeros(nc))  # read-only on NumPy; MAPLE never writes into a demand array
    # water accumulators (Phase 4 columns)
    cum_rain, cum_intake, cum_return, cum_drain = zeros(), zeros(), zeros(), zeros()
    peak_depth = xp.array(state.storm.depth_m, copy=True)
    last_velocity = xp.where(active, xp.sqrt(state.storm.depth_m) * graph.conveyance.reshape(shape), 0.0)
    peak_velocity = xp.array(last_velocity, copy=True)
    outlet_q = graph.dx_m * xp.sum(xp.where(outlet, state.storm.discharge_m2_s, 0.0))
    peak_q, peak_t = scalar(outlet_q), scalar(state.t_s)
    cum_export = scalar()
    max_balance, max_constitutive, max_cr_old, max_cr_new = scalar(), scalar(), scalar(), scalar()
    cells_no, cells_partial, cells_complete = (xp.zeros((), dtype=np.int64) for _ in range(3))
    # sediment accumulators
    cum_pickup, cum_deposition, cum_export_request = zeros(nc), zeros(nc), zeros(nc)
    by_class = {name: vector() for name in (
        "requested_pickup", "raindrop_pickup_requested", "flow_pickup_requested", "actual_pickup",
        "availability_shortfall", "holdings_shortfall", "pickup_numerical_residual", "deposition_requested",
        "deposition_actual", "deposition_unmet", "export_requested", "export_actual", "export_unmet",
        "transport_budget_residual", "transport_budget_tolerance", "deposit_numerical_residual")}
    residual_scalar = scalar()
    mobile_by_class = state.bed.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1))
    initial_bed = bed_inventory(state.bed)
    initial_bed_cells = state.bed.voxel_column.mass_kg.sum(axis=2) + state.bed.active_layer.mass_kg
    initial_mobile = xp.array(mobile_by_class, copy=True)
    peak_mobile, peak_mobile_t = scalar(xp.sum(mobile_by_class)), scalar(state.t_s)
    export_rate = vector()
    peak_export, peak_export_t = scalar(), scalar(state.t_s)
    max_sed_courant, max_decay, max_cell_residual = scalar(), scalar(), scalar()
    regime_steps = {name: xp.zeros((), dtype=np.int64) for name in REGIME_CODES}
    ledger_reset = xp.zeros(state.bed.ledger.process_totals_kg.shape, dtype=np.float64)
    committed_net = vector()
    hydrograph = xp.zeros((boundaries.size, len(HYDROGRAPH_COLUMNS)), dtype=np.float64)
    sediment_hydrograph = xp.zeros((boundaries.size, len(columns)), dtype=np.float64)
    rate = xp.empty(shape, dtype=np.float64)  # owned scratch

    n_steps = n_rejected = n_calls = n_transport_rejections = 0
    max_substeps = 0
    n_commits = n_forced = n_graph_changes = rerouted_total = 0
    commit_log: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    dt_min, dt_max = math.inf, 0.0
    commit_wall = 0.0
    t = state.t_s
    first_wall = first_cpu = 0.0
    wall0, cpu0 = time.perf_counter(), time.process_time()

    def publish_commit(commit: _Commit) -> None:
        nonlocal state, n_commits, n_forced, n_graph_changes, rerouted_total, ledger_reset, committed_net
        state = replace(state, bed=commit.bed, graph=commit.graph, network=commit.network, grid=commit.grid,
                        storm=commit.storm)
        n_commits += 1
        n_forced += int(commit.record["forced"])
        n_graph_changes += int(commit.record["graph_changed"])
        rerouted_total += commit.record["rerouted_cells"]
        ledger_reset = ledger_reset + commit.maple_result.pre_commit_process_totals_kg
        committed_net = committed_net + commit.maple_result.pre_commit_pending_mass_by_class_kg
        if len(commit_log) < control.max_commit_log:
            commit_log.append(commit.record)

    def refresh_hydraulics() -> None:
        """Hydraulic diagnostics of the CURRENT accepted graph and state
        after a commit re-initialised the discharge (instantaneous kinematic
        view, no water volume added): velocity k sqrt(h), outlet discharge
        from the re-initialised q, the outlet mask of the refreshed graph.
        Peaks are taken over BOTH the pre-commit routed values and these
        post-commit values; the integrated export stays the routing's."""
        nonlocal last_velocity, outlet_q, peak_q, peak_t, outlet
        g_ = state.graph
        outlet = g_.outlet_flat.reshape(shape)
        last_velocity = xp.where(active, xp.sqrt(state.storm.depth_m) * g_.conveyance.reshape(shape), 0.0)
        outlet_q = g_.dx_m * xp.sum(xp.where(outlet, state.storm.discharge_m2_s, 0.0))
        higher = outlet_q > peak_q
        peak_t = xp.where(higher, scalar(t), peak_t)
        peak_q = xp.where(higher, outlet_q, peak_q)
        xp.maximum(peak_velocity, last_velocity, out=peak_velocity)

    for row, boundary in enumerate(boundaries.tolist()):
        t_row = t
        span = boundary - t_row
        n_sub = max(1, math.ceil(span / storm_control.max_dt_s))
        while span / n_sub > storm_control.max_dt_s:
            n_sub += 1
        for k in range(1, n_sub + 1):
            target = boundary if k == n_sub else t_row + k * (span / n_sub)
            if not target > t:
                raise SedimentEventError(f"floating time does not advance: planned substep target {target} s is "
                                         f"not after t = {t} s (time resolution too coarse for max_dt_s = "
                                         f"{storm_control.max_dt_s})")
            while t < target:
                dt = target - t
                snap = True
                retries = 0
                while True:
                    if n_steps >= storm_control.max_steps:
                        raise SedimentEventError(f"max_steps = {storm_control.max_steps} reached at t = {t} s "
                                                 f"before end_s = {end} s")
                    if not t + dt > t:
                        raise SedimentEventError(f"floating time does not advance: t = {t} s, dt = {dt} s")
                    field.apply(schedule.rate_after_m_per_s(t), out=rate)
                    try:
                        attempt = sediment_coupled_step(state, context, column, rate, vegetation_cover_fraction,
                                                        sediment, dt, control, zero_demand=zero_demand)
                    except (RoutingStepRejected, TransportStepRejected) as exc:
                        retries += 1
                        n_rejected += 1
                        if len(rejections) < control.max_rejections_recorded:
                            rejections.append({"t_s": t, "dt_tried_s": dt, "kind": type(exc).__name__,
                                               "reason": str(exc)})
                        if retries > storm_control.max_retries:
                            raise SedimentEventError(f"step at t = {t} s rejected {retries} times (max_retries = "
                                                     f"{storm_control.max_retries}); last dt {dt} s: {exc}") from exc
                        halved = dt * 0.5
                        if halved < storm_control.min_dt_s:
                            raise SedimentEventError(f"halving dt {dt} s to {halved} s would fall below the retry "
                                                     f"floor min_dt_s = {storm_control.min_dt_s} s at t = {t} s "
                                                     f"after {retries} rejection(s): {exc}") from exc
                        dt = halved
                        snap = False
                        continue
                    break
                t_new = target if snap else t + dt
                # 6. MAPLE commit triggers on the post-step ledger; reroute on commit. Nothing
                #    below publishes until every check passed.
                commit = None
                if control.commit:
                    wall_c = time.perf_counter()
                    commit = _commit_and_reroute(attempt.bed, context, state.graph, state.terrain,
                                                 replace(attempt.storm.state, t_s=t_new), t_new, force=False)
                    commit_wall += time.perf_counter() - wall_c
                # 7. publish.
                t = t_new
                storm_state = replace(attempt.storm.state, t_s=t)
                state = replace(state, storm=storm_state, bed=attempt.bed,
                                sediment_velocity_m_s=attempt.physics.sediment_velocity_m_s)
                col, route = attempt.storm.column, attempt.storm.route
                pickup, deposit, transport, physics = attempt.pickup, attempt.deposit, attempt.transport, attempt.physics
                cum_rain += col.rain_m
                cum_intake += col.intake_m
                cum_return += col.saturation_return_m
                cum_drain += col.drainage_m
                cum_export = cum_export + route.export_m3  # integrated face volume of the accepted routing step
                # pre-commit routed diagnostics (the water that actually moved this step)
                outlet_q = route.outlet_discharge_m3_s
                higher = outlet_q > peak_q
                peak_t = xp.where(higher, scalar(t), peak_t)
                peak_q = xp.where(higher, outlet_q, peak_q)
                xp.maximum(peak_depth, route.depth_m, out=peak_depth)
                xp.maximum(peak_velocity, route.velocity_m_s, out=peak_velocity)
                last_velocity = route.velocity_m_s
                if commit is not None:
                    publish_commit(commit)
                    refresh_hydraulics()  # post-commit graph / q: diagnostics and peaks follow the accepted state
                max_balance = xp.maximum(max_balance, route.max_cell_balance_residual_m)
                max_constitutive = xp.maximum(max_constitutive, route.max_constitutive_residual_m)
                max_cr_old = xp.maximum(max_cr_old, route.max_courant_old)
                max_cr_new = xp.maximum(max_cr_new, route.max_courant_new)
                cells_no = cells_no + attempt.storm.n_no_runon
                cells_partial = cells_partial + attempt.storm.n_partial_runon
                cells_complete = cells_complete + attempt.storm.n_complete_runon
                # sediment ground truth
                cum_pickup += pickup.actual_removal_by_cell_class_kg
                cum_deposition += deposit.deposition_by_cell_class_kg
                cum_export_request += transport.export_request_kg
                by_class["requested_pickup"] += pickup.requested_removal_by_class_kg
                by_class["raindrop_pickup_requested"] += physics.raindrop_pickup_kg.sum(axis=(0, 1))
                by_class["flow_pickup_requested"] += physics.flow_pickup_kg.sum(axis=(0, 1))
                by_class["actual_pickup"] += pickup.actual_removal_by_class_kg
                by_class["availability_shortfall"] += pickup.availability_shortfall_by_class_kg
                by_class["holdings_shortfall"] += pickup.shortfall_by_class_kg
                by_class["pickup_numerical_residual"] += pickup.numerical_residual_by_class_kg
                by_class["deposition_requested"] += transport.deposition_request_by_class_kg
                by_class["deposition_actual"] += deposit.deposited_mass_by_class_kg
                by_class["deposition_unmet"] += transport.deposition_request_by_class_kg - deposit.deposited_mass_by_class_kg
                by_class["export_requested"] += transport.export_request_by_class_kg
                by_class["export_actual"] += deposit.boundary_export_by_class_kg
                by_class["export_unmet"] += transport.export_request_by_class_kg - deposit.boundary_export_by_class_kg
                by_class["transport_budget_residual"] += transport.budget_residual_by_class_kg
                by_class["transport_budget_tolerance"] += transport.budget_tolerance_by_class_kg
                by_class["deposit_numerical_residual"] += deposit.numerical_residual_by_class_kg
                residual_scalar = residual_scalar + abs(pickup.numerical_residual_scalar_kg) \
                    + abs(deposit.numerical_residual_scalar_kg)
                mobile_by_class = state.bed.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1))
                mobile_total = xp.sum(mobile_by_class)
                higher = mobile_total > peak_mobile
                peak_mobile_t = xp.where(higher, scalar(t), peak_mobile_t)
                peak_mobile = xp.where(higher, mobile_total, peak_mobile)
                export_rate = deposit.boundary_export_by_class_kg / dt
                export_total_rate = xp.sum(export_rate)
                higher = export_total_rate > peak_export
                peak_export_t = xp.where(higher, scalar(t), peak_export_t)
                peak_export = xp.where(higher, export_total_rate, peak_export)
                max_sed_courant = xp.maximum(max_sed_courant, transport.max_courant)
                max_decay = xp.maximum(max_decay, transport.max_decay_exponent)
                max_cell_residual = xp.maximum(max_cell_residual, transport.max_cell_balance_residual_kg)
                for name, count in physics.regime_counts.items():
                    regime_steps[name] = regime_steps[name] + count
                max_substeps = max(max_substeps, attempt.n_substeps)
                n_transport_rejections += attempt.n_transport_rejections
                n_calls += 2
                n_steps += 1
                dt_min, dt_max = min(dt_min, dt), max(dt_max, dt)
                if n_steps == 1:
                    first_wall, first_cpu = time.perf_counter() - wall0, time.process_time() - cpu0
                    wall0, cpu0 = time.perf_counter(), time.process_time()
        # Forced final commit at the configured end (real time), only when the
        # ledger records physical activity since the last commit and MAPLE does
        # not already consider this exact state committed (no no-op commits).
        if row == boundaries.size - 1 and control.force_final_commit:
            bed = state.bed
            ledger = bed.ledger
            activity = to_bool(xp.any(ledger.pending_bed_mass_change_kg != 0.0)
                               | xp.any(ledger.n_physical_touches != 0))
            already = not activity or is_already_committed(
                ledger, bed.committed_topography, bed.voxel_column, bed.active_layer,
                context.geometry, context.mass_resolution_kg, t)
            if not already:
                wall_c = time.perf_counter()
                commit = _commit_and_reroute(bed, context, state.graph, state.terrain, state.storm, t, force=True)
                commit_wall += time.perf_counter() - wall_c
                if commit is not None:
                    publish_commit(commit)
                    refresh_hydraulics()
        storm_state = state.storm
        hydrograph[row] = xp.stack([
            scalar(t),
            area * xp.sum(cum_rain), area * xp.sum(cum_intake), area * xp.sum(cum_return),
            area * xp.sum(cum_drain), scalar(cum_export),
            area * xp.sum(storm_state.depth_m), area * xp.sum(storm_state.soil_water_m),
            scalar(outlet_q),
            xp.max(storm_state.depth_m), xp.max(last_velocity),
            scalar(float(n_steps)), scalar(float(n_rejected)),
            scalar(dt_min if n_steps else 0.0),
            scalar(max_balance), scalar(max_constitutive),
        ])
        mobile_by_class = state.bed.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1))
        sediment_hydrograph[row] = _row(xp, [
            scalar(t), xp.sum(mobile_by_class), xp.sum(by_class["actual_pickup"]),
            xp.sum(by_class["deposition_actual"]), xp.sum(by_class["export_actual"]), xp.sum(export_rate),
            scalar(float(state.bed.committed_topography.commit_count)), scalar(float(n_graph_changes)),
            scalar(float(max_substeps)),
        ], [mobile_by_class, export_rate, by_class["actual_pickup"], by_class["deposition_actual"],
            by_class["export_actual"]])
    remaining_wall, remaining_cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    # final consistency: solver depth is the MAPLE depth; discharge consistent with the final graph
    if not to_float(xp.max(xp.abs(state.bed.water.depth_m - state.storm.depth_m))) == 0.0:
        raise SedimentEventError("internal error: final MAPLE water depth differs from the solver depth")
    validate_storm_state(state.graph, state.storm, storm_control.root_tolerance_m, column)
    if state.network.graph_input_sha256 != state.graph.input_sha256:
        raise SedimentEventError("internal error: transport network does not belong to the final graph")
    nz = int(state.bed.voxel_column.mass_kg.shape[2])
    return SedimentEventResult(
        state=state, hydrograph=hydrograph, sediment_hydrograph=sediment_hydrograph, sediment_columns=columns,
        boundaries=boundaries,
        cumulative_rain_m=cum_rain, cumulative_intake_m=cum_intake, cumulative_saturation_return_m=cum_return,
        cumulative_drainage_m=cum_drain, cumulative_export_m3=cum_export,
        peak_depth_m=peak_depth, peak_velocity_m_s=peak_velocity, last_velocity_m_s=last_velocity,
        peak_outlet_discharge_m3_s=peak_q, time_of_peak_outlet_s=peak_t,
        max_courant_old=max_cr_old, max_courant_new=max_cr_new,
        max_routing_cell_balance_residual_m=max_balance, max_constitutive_residual_m=max_constitutive,
        cell_steps_no_runon=cells_no, cell_steps_partial_runon=cells_partial, cell_steps_complete_runon=cells_complete,
        cumulative_pickup_kg=cum_pickup, cumulative_deposition_kg=cum_deposition,
        cumulative_export_request_kg=cum_export_request, by_class=by_class,
        initial_bed_by_cell_class_kg=initial_bed_cells,
        initial_bed_inventory_kg=initial_bed, initial_mobile_kg=initial_mobile,
        numerical_residual_scalar_abs_kg=residual_scalar,
        peak_mobile_kg=peak_mobile, time_of_peak_mobile_s=peak_mobile_t,
        peak_export_rate_kg_s=peak_export, time_of_peak_export_s=peak_export_t,
        max_sediment_courant=max_sed_courant, max_decay_exponent=max_decay,
        max_transport_cell_residual_kg=max_cell_residual, max_transport_substeps_used=max_substeps,
        n_transport_rejections=n_transport_rejections, regime_cell_steps=regime_steps,
        ledger_process_totals_reset_kg=ledger_reset, committed_net_bed_change_kg=committed_net,
        n_commits=n_commits, n_forced_commits=n_forced, n_graph_changes=n_graph_changes,
        rerouted_cells_total=rerouted_total, commit_log=tuple(commit_log),
        n_accepted_steps=n_steps, n_rejected_attempts=n_rejected, rejections=tuple(rejections),
        min_accepted_dt_s=dt_min, max_accepted_dt_s=dt_max, n_maple_water_calls=n_calls,
        first_step_wall_s=first_wall, first_step_cpu_s=first_cpu,
        remaining_wall_s=remaining_wall, remaining_cpu_s=remaining_cpu, commit_wall_s=commit_wall,
        n_inventory_terms=shape[0] * shape[1] * (nz + 1),
    )


def water_channel_index() -> int:
    """MAPLE's ledger process index of the water channel (for reports)."""
    from maple.core.types.sediment_ledger import process_index

    return process_index("water_erosion_deposition")


def event_error_types() -> tuple[type[BaseException], ...]:
    """Exception types a runner treats as a refused / failed event."""
    return (SedimentEventError, StormError, BedIntegrationError)
