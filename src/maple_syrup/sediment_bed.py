"""Narrow water integration with MAPLE's actual bed and commit machinery.

No wind physics or bed-exchange arithmetic is implemented here. BedState
contains MAPLE objects; all physical mass changes go through MAPLE's water
API. TerrainReference is immutable boundary/datum evidence, never a second
evolving bed. Rebuilding routing reads the current MAPLE committed surface.

Water depth remains SYRUP-owned (constant_depth). A geometry refresh resets
kinematic discharge in the event driver, not here. Graph construction is a
host operation at commits; GPU callers incur an explicit elevation transfer.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from maple_syrup.routing import RoutingGraph, build_routing_graph

ADAPTER_NAME = 'maple_syrup/phase5'


class BedIntegrationError(ValueError):
    pass


@dataclass(frozen=True)
class BedState:
    voxel_column: Any
    active_layer: Any
    water: Any
    ledger: Any
    committed_topography: Any
    sediment_availability: Any


@dataclass(frozen=True)
class BedContext:
    geometry: Any
    grain_classes: Any
    mass_resolution_kg: float
    topographic_commit_spec: Any
    avalanching_spec: Any
    initial_available_fraction: Any
    depth_update_rule: str = 'constant_depth'


@dataclass(frozen=True)
class TerrainReference:
    initial_full_elevation_m: np.ndarray
    initial_committed_elevation_m: np.ndarray
    export_receiver_full: np.ndarray
    active_mask: np.ndarray
    friction_factor: np.ndarray
    dx_m: float


@dataclass(frozen=True)
class BedCommit:
    state: BedState
    graph: RoutingGraph
    did_commit: bool
    trigger: Any
    maple_result: Any = None
    water_result: Any = None


def bed_from_case(case, *, xp=None) -> tuple[BedState, BedContext]:
    """Reuse a loaded MAPLE case. Declarative availability builders need
    their original resolved maps and are explicitly outside this constructor.
    Plot 1 uses the declared uniform 100% availability, not a wind restriction.
    The depth-rule override is per-run metadata; no case file is modified.
    """
    from maple.core.backend import to_device, to_device_tree
    from maple.core.parameters.case_initialization import (
        resolve_initial_available_fraction_array,
    )

    if case.water is None:
        raise BedIntegrationError('case must carry an actual MAPLE WaterState')
    spec = case.sediment_availability_spec
    if spec.builder is not None or spec.builder_by_class:
        raise BedIntegrationError('availability builders require their resolved maps; unsupported by this case adapter')
    namespace = np if xp is None else xp
    cfg = case.config
    fractions = resolve_initial_available_fraction_array(
        spec, cfg.geometry, [g.class_id for g in cfg.grain_classes.classes])
    state = BedState(case.voxel_column, case.active_layer, case.water, case.sediment_ledger,
                     case.committed_topography, case.sediment_availability)
    state = to_device_tree(state, namespace)
    context = BedContext(cfg.geometry, cfg.grain_classes, cfg.numerics.mass_resolution_kg,
                         cfg.topographic_commit, cfg.avalanching, to_device(fractions, namespace))
    return state, context


def with_water(state: BedState, context: BedContext, depth_m, mobile_kg=None) -> BedState:
    """Validate a replacement WaterState before returning a new wrapper."""
    from maple.core.backend import resolve_operand_namespace
    from maple.core.types.water import WaterState
    from maple.water import validate_water_state

    water = WaterState(depth_m, state.water.mobile_mass_by_cell_class_kg if mobile_kg is None else mobile_kg)
    resolve_operand_namespace(state, water, context.initial_available_fraction)
    validate_water_state(water, context.geometry.ny, context.geometry.nx, len(context.grain_classes.classes))
    return replace(state, water=water)


def apply_bed_demand(state: BedState, context: BedContext, demand) -> tuple[BedState, Any]:
    """Actual MAPLE pickup/deposition/export, including supply limits,
    availability, stratigraphic refill and gross ledger accounting. Pure.
    The caller retains the returned genuine result for event accounting.
    """
    from maple.water import apply_water_process_demand, validate_water_state

    g = context.geometry
    validate_water_state(state.water, g.ny, g.nx, len(context.grain_classes.classes))
    result = apply_water_process_demand(
        state.voxel_column, state.active_layer, state.water, state.ledger, demand,
        g, context.grain_classes, context.mass_resolution_kg,
        sediment_availability=state.sediment_availability,
        initial_available_fraction=context.initial_available_fraction,
        adapter_name=ADAPTER_NAME, detachment_integration='rate_times_dt')
    return replace(state, voxel_column=result.new_voxel_column, active_layer=result.new_active_layer,
                   water=result.new_water, ledger=result.new_ledger,
                   sediment_availability=result.new_sediment_availability), result


def bed_inventory(state: BedState):
    """Actual bed inventory by class via MAPLE's shared accounting helper."""
    from maple.aeolian.flux.step import bed_inventory_by_class_kg

    return bed_inventory_by_class_kg(state.voxel_column, state.active_layer)


def terrain_reference(full_elevation_m, export_receiver_full, state: BedState,
                      graph: RoutingGraph) -> TerrainReference:
    from maple.core.backend import to_host
    from maple.core.types.topographic_commit import committed_elevation_m

    def fixed(array, dtype):
        result = np.array(array, dtype=dtype, copy=True)
        result.flags.writeable = False
        return result

    z = fixed(full_elevation_m, np.float64)
    exports = fixed(export_receiver_full, np.bool_)
    committed = fixed(to_host(committed_elevation_m(state.committed_topography)), np.float64)
    if z.shape != (graph.shape[0] + 2, graph.shape[1] + 2) or exports.shape != z.shape:
        raise BedIntegrationError('terrain boundary ring does not match the routing grid')
    if committed.shape != graph.shape or not np.isfinite(committed).all():
        raise BedIntegrationError('MAPLE committed elevation does not match the routing grid')
    # A constant datum offset is allowed; a different terrain shape is not.
    offset = committed - z[1:-1, 1:-1]
    tol = 128 * np.finfo(float).eps * max(1., float(np.abs(z).max()), float(np.abs(committed).max()))
    if not np.isfinite(z).all() or np.max(np.abs(offset - offset.flat[0])) > tol:
        raise BedIntegrationError('reference DEM differs from MAPLE terrain by more than a constant datum')
    rebuilt = build_routing_graph(z, exports, graph.friction_factor, graph.dx_m,
                                  active_mask=graph.active, xp=graph.xp)
    if any(not np.array_equal(getattr(rebuilt, name), getattr(graph, name))
           for name in ('aspect', 'slope', 'receiver', 'active', 'outlet')):
        raise BedIntegrationError('terrain reference does not reproduce the initial graph')
    return TerrainReference(z, committed, exports, fixed(graph.active, np.bool_),
                            fixed(graph.friction_factor, np.float64), graph.dx_m)


def refresh_routing(reference: TerrainReference, state: BedState, *, xp=np) -> RoutingGraph:
    """Read authoritative committed terrain; hold the original outer ring
    fixed. Unsupported pits/flats/outlets raise before a new state is exposed.
    """
    from maple.core.backend import to_host
    from maple.core.types.topographic_commit import committed_elevation_m

    current = to_host(committed_elevation_m(state.committed_topography))
    if current.shape != reference.initial_committed_elevation_m.shape:
        raise BedIntegrationError('committed terrain shape changed')
    z = reference.initial_full_elevation_m.copy()
    z[1:-1, 1:-1] += current - reference.initial_committed_elevation_m
    return build_routing_graph(z, reference.export_receiver_full, reference.friction_factor,
                               reference.dx_m, active_mask=reference.active_mask, xp=xp)


def commit_bed(state: BedState, context: BedContext, graph: RoutingGraph,
               reference: TerrainReference, time_s: float, *, force: bool = False) -> BedCommit:
    """Compose MAPLE's actual trigger and commit APIs. A forced final
    commit uses the real current time. No wind scheduler/hop table is needed.

    All candidate outputs, callback boxes and refreshed graphs stay local
    until the full operation succeeds. A late graph failure does not publish
    any bed, ledger, availability or water change into the caller's state.
    """
    from maple.core.backend import array_namespace, to_bool
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.surface.avalanche.callback import (
        make_availability_aware_avalanche_callback,
    )
    from maple.surface.topographic_commit.commit import commit_topography
    from maple.surface.topographic_commit.triggers import evaluate_commit_triggers
    from maple.water.commit_callback import make_water_depth_commit_callback

    g = context.geometry
    if (graph.shape != (g.ny, g.nx) or graph.dx_m != g.dx_m or g.dx_m != g.dy_m
            or reference.dx_m != graph.dx_m
            or reference.initial_committed_elevation_m.shape != graph.shape):
        raise BedIntegrationError('bed context, terrain and routing geometry must agree')
    xp = array_namespace(state.water.depth_m)
    if xp is not graph.xp:
        raise BedIntegrationError('bed water and routing must use the same backend')
    if context.depth_update_rule != 'constant_depth':
        raise BedIntegrationError('SYRUP requires constant_depth for hydraulic ownership')
    if not isinstance(force, bool):
        raise BedIntegrationError('force must be bool')
    trigger = evaluate_commit_triggers(state.ledger, state.committed_topography, context.geometry,
                                      time_s, context.topographic_commit_spec)
    if not force and not trigger.should_commit:
        return BedCommit(state, graph, False, trigger)
    avalanche, avalanche_box = make_availability_aware_avalanche_callback(
        context.avalanching_spec, state.sediment_availability, context.initial_available_fraction)
    water, water_box = make_water_depth_commit_callback(
        state.water, context.geometry, committed_elevation_m(state.committed_topography),
        depth_update_rule=context.depth_update_rule)
    result = commit_topography(state.ledger, state.committed_topography, state.voxel_column,
                               state.active_layer, context.geometry, context.mass_resolution_kg,
                               time_s, avalanche_callback=avalanche, water_callback=water)
    water_box.publish()  # local box only; nothing is returned until graph validation
    xp = array_namespace(state.water.depth_m)
    if (not water_box.ran or water_box.depth_update.rule != 'constant_depth'
            or not to_bool(xp.all(water_box.water.depth_m == state.water.depth_m))):
        raise BedIntegrationError('MAPLE commit changed solver-owned water depth')
    candidate = replace(state, voxel_column=result.new_voxel_column, active_layer=result.new_active_layer,
                        committed_topography=result.new_committed_state, ledger=result.new_ledger,
                        sediment_availability=avalanche_box.new_sediment_availability, water=water_box.water)
    updated_graph = refresh_routing(reference, candidate, xp=graph.xp)
    return BedCommit(candidate, updated_graph, True, trigger, result, water_box)
