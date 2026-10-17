"""Native MAPLE bed adapter for the pond event: accounting commits WITHOUT any D4 graph refresh (dense bed only).

`sediment_bed.commit_bed` conditionally handles the D4 `RoutingGraph` refresh / validation after a commit (with a held
routing source the refresh is gated on the net displacement; without one it refreshes every commit via
`refresh_routing` / `routing_graph_from_elevation`). The pond event must handle none of it: its hydraulic geometry is
the held `PondTopology`, so this adapter omits graph handling entirely. This module is the narrow standalone adapter the assignment prefers over refactoring
`sediment_bed.py`: it composes the SAME native MAPLE APIs `commit_bed` composes (`evaluate_commit_triggers`,
`commit_topography`, the availability-aware avalanche callback, the water depth commit callback) and stops there.
Nothing in `sediment_bed.py` is modified. No bed arithmetic, wind physics or alternative bed lives here; the only
evolving inventory is MAPLE's `BedState`.

Contents
--------
- `require_supported_bed`: dense bed only. A windowed bed, enabled avalanching, a non-`constant_depth` rule or the
  exchange-history option are REFUSED before anything is read or mutated (windowed support is unqualified).
- `native_commit`: MAPLE trigger evaluation and (when triggered or forced) the genuine native commit. The
  solver-owned depth must be unchanged (checked as `commit_bed` does). No routing, topology or source is touched.
- `gate_displacement_m` / `committed_displacement_m`: the geometry gate quantity, max over ACTIVE cells of
  |actual MAPLE elevation - held source| (existing `net_routing_displacement_m`; one counted scalar read, no
  full-surface transfer). `current_routing_surface_m` is MAPLE's exact current surface (dense: its own diagnostic).
- `physical_full_source`: the full `(ny+2, nx+2)` host source = stable initial datum + the ACTUAL committed delta on
  active cells, ring and inactive cells fixed. Only built when a refresh really happens.
- `steepest_d4_descent`: topographic slope / legacy aspect of the HELD real bed (steepest strictly downward D4
  neighbour among active cells and the ring cell across an explicit outlet face; flat -> S = 0, aspect 1).
- `pickup_transport_deposit`: the dense two-call native transaction (pickup, caller transport, deposit/export), pure.

Nothing here was run by its author (file-only tools); the root executes the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from maple_syrup.sediment_bed import (
    BedContext,
    BedIntegrationError,
    BedState,
    apply_bed_demand,
    current_routing_surface_m,
    is_windowed_bed,
    net_routing_displacement_m,
    reroute_threshold_m,
    with_water,
)

__all__ = [
    "NativeCommit",
    "PondBedError",
    "committed_displacement_m",
    "gate_displacement_m",
    "native_commit",
    "physical_full_source",
    "pickup_transport_deposit",
    "require_supported_bed",
    "reroute_threshold_m",
    "steepest_d4_descent",
]

# routing.ASPECT_STEPS: 1 north (+row), 2 east (+col), 3 south, 4 west. Ties keep the earlier code.
_D4_ASPECT = ((1, 0, 1), (0, 1, 2), (-1, 0, 3), (0, -1, 4))


class PondBedError(BedIntegrationError):
    """Unsupported or invalid bed for the pond event. Raised before anything is mutated."""


def require_supported_bed(state: BedState, context: BedContext) -> None:
    """Dense actual-MAPLE bed with disabled avalanching and the solver-owned depth rule; refuse anything else."""
    if not isinstance(state, BedState) or not isinstance(context, BedContext):
        raise PondBedError("state and context must be the actual sediment_bed BedState / BedContext")
    if is_windowed_bed(state):
        raise PondBedError("windowed beds are unqualified for the pond event; use a dense bed (refused before mutation)")
    if bool(getattr(context.avalanching_spec, "enabled", False)):
        raise PondBedError("avalanching must be explicitly disabled for the pond event (its mass exchange over a "
                           "static pond topology is unqualified)")
    if context.depth_update_rule != "constant_depth":
        raise PondBedError("the pond event requires constant_depth (the hydraulic solver owns the depth)")
    if context.record_exchange_history is not False:
        raise PondBedError("record_exchange_history is not supported by the pond event")
    if context.geometry.dx_m != context.geometry.dy_m:
        raise PondBedError("the pond event needs a square cell (dx == dy)")


# --- commit ---------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class NativeCommit:
    """`state` the committed bed (the input object when `did_commit` is False), the MAPLE `trigger` evaluated at
    `time_s` and the genuine MAPLE `CommitResult` (None without a commit)."""

    state: BedState
    did_commit: bool
    trigger: Any
    maple_result: Any = None


def native_commit(state: BedState, context: BedContext, time_s: float, *, force: bool = False) -> NativeCommit:
    """MAPLE's actual commit when its own triggers fire or `force` is True (forced FINAL commits and net-elevation
    commits are the caller's decision). No routing/topology work: this function can never refresh geometry.
    Pure: every candidate stays local until the water-depth check passed."""
    from maple.core.backend import array_namespace, to_bool
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.surface.avalanche.callback import make_availability_aware_avalanche_callback
    from maple.surface.topographic_commit.commit import commit_topography
    from maple.surface.topographic_commit.triggers import evaluate_commit_triggers
    from maple.water.commit_callback import make_water_depth_commit_callback

    require_supported_bed(state, context)
    if not isinstance(force, bool):
        raise PondBedError("force must be bool")
    trigger = evaluate_commit_triggers(state.ledger, state.committed_topography, context.geometry, time_s,
                                       context.topographic_commit_spec)
    if not force and not trigger.should_commit:
        return NativeCommit(state, False, trigger)
    avalanche, avalanche_box = make_availability_aware_avalanche_callback(
        context.avalanching_spec, state.sediment_availability, context.initial_available_fraction)
    water, water_box = make_water_depth_commit_callback(
        state.water, context.geometry, committed_elevation_m(state.committed_topography),
        depth_update_rule=context.depth_update_rule)
    result = commit_topography(state.ledger, state.committed_topography, state.voxel_column, state.active_layer,
                               context.geometry, context.mass_resolution_kg, time_s,
                               avalanche_callback=avalanche, water_callback=water)
    water_box.publish()  # local box only
    xp = array_namespace(state.water.depth_m)
    if (not water_box.ran or water_box.depth_update.rule != "constant_depth"
            or not to_bool(xp.all(water_box.water.depth_m == state.water.depth_m))):
        raise PondBedError("MAPLE commit changed solver-owned water depth")
    depth_update = water_box.depth_update
    if depth_update.displaced_volume_m3 != 0.0 or depth_update.clamp_added_volume_m3 != 0.0 \
            or depth_update.clamp_cell_count != 0:
        raise PondBedError(f"commit at t = {time_s} s changed the water volume: {depth_update}")
    candidate = replace(state, voxel_column=result.new_voxel_column, active_layer=result.new_active_layer,
                        committed_topography=result.new_committed_state, ledger=result.new_ledger,
                        sediment_availability=avalanche_box.new_sediment_availability, water=water_box.water)
    return NativeCommit(candidate, True, trigger, result)


# --- geometry gate ---------------------------------------------------------------------------
def gate_displacement_m(state: BedState, context: BedContext, held_source_m: Any, active: Any) -> float:
    """max over ACTIVE cells of |MAPLE's exact CURRENT surface - held source| (m): per-cell signed (gains and
    losses within a cell cancel), absolute per cell, maximum over cells (different cells never cancel). ONE counted
    scalar read in the bed namespace; the `(ny, nx)` temporaries stay on the device. Evaluated before any commit."""
    return net_routing_displacement_m(current_routing_surface_m(state, context), held_source_m, active)


def committed_displacement_m(state: BedState, held_source_m: Any, active: Any) -> float:
    """The same quantity for the COMMITTED elevation (what a commit just published)."""
    from maple.core.types.topographic_commit import committed_elevation_m

    return net_routing_displacement_m(committed_elevation_m(state.committed_topography), held_source_m, active)


def physical_full_source(initial_full_m: np.ndarray, initial_committed_m: np.ndarray, committed_m: np.ndarray,
                         active: np.ndarray) -> np.ndarray:
    """Full `(ny+2, nx+2)` host elevation: the stable initial datum + the actual committed delta on ACTIVE cells.
    The ring and every inactive cell keep their initial values exactly (fixed halo / mask)."""
    out = np.array(initial_full_m, dtype=np.float64, copy=True)
    interior = out[1:-1, 1:-1]
    interior[active] += (np.asarray(committed_m, dtype=np.float64) - initial_committed_m)[active]
    if not np.isfinite(out).all():
        raise PondBedError("physical source elevation is not finite")
    return out


def steepest_d4_descent(source_z_full: np.ndarray, active: np.ndarray, dx_m: float,
                        outlets: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Topographic `(slope, aspect)` of the HELD real bed, `(ny, nx)` float64 / int8.

    Slope = largest strictly positive drop `(z_c - z_n) / dx` over the D4 neighbours that are ACTIVE cells or the
    ring cell across an explicit outlet face (inactive/hole cells and closed ring faces are not terrain the water
    or splash can use). Aspect: legacy codes 1 N (+row), 2 E, 3 S, 4 W, the first of equal drops in that order.
    A cell with no strictly lower neighbour is FLAT: slope 0, aspect 1 (a flat has no direction; the splash law
    gives all four rays 0.25 there). Inactive cells: slope 0, aspect 0. Host, vectorized, no loop over cells."""
    z = np.asarray(source_z_full, dtype=np.float64)
    ny, nx = active.shape
    zc = z[1:-1, 1:-1]
    padded = np.zeros((ny + 2, nx + 2), dtype=bool)
    padded[1:-1, 1:-1] = active
    best = np.zeros((ny, nx), dtype=np.float64)
    code = np.ones((ny, nx), dtype=np.int8)
    for dr, dc, c in _D4_ASPECT:
        ring_open = np.zeros((ny, nx), dtype=bool)
        for o in outlets:
            if (o.d_row, o.d_col) == (dr, dc):
                ring_open[o.row, o.col] = True
        neighbor = z[1 + dr: 1 + dr + ny, 1 + dc: 1 + dc + nx]
        usable = active & (padded[1 + dr: 1 + dr + ny, 1 + dc: 1 + dc + nx] | ring_open)
        drop = (zc - neighbor) / dx_m
        better = usable & (drop > best)
        best = np.where(better, drop, best)
        code = np.where(better, np.int8(c), code)
    slope = np.where(active, best, 0.0)
    aspect = np.where(active, code, np.int8(0)).astype(np.int8)
    return slope, aspect


# --- transaction -------------------------------------------------------------------------------
def pickup_transport_deposit(state: BedState, context: BedContext, depth_m: Any, requested_pickup_kg: Any, transport):
    """Dense native two-call transaction of one accounting step, pure.

    1. publish the solver `depth_m`, MAPLE pickup of `requested_pickup_kg` (capped by holdings and availability);
    2. `transport(pickup_result)` returns `(mobile_after_transfer_kg, deposit_demand)` from the ACTUAL pickup;
    3. publish `(depth_m, mobile_after_transfer)`, MAPLE deposition/export of `deposit_demand`.
    Returns `(bed, pickup_result, deposit_result)`; an exception anywhere leaves `state` untouched (immutable)."""
    from maple.water import WaterProcessDemand

    require_supported_bed(state, context)
    zeros = state.water.mobile_mass_by_cell_class_kg * 0.0
    bed = with_water(state, context, depth_m)
    bed, pickup = apply_bed_demand(bed, context, WaterProcessDemand(requested_pickup_kg, zeros))
    mobile_after_transfer, demand = transport(pickup)
    bed = with_water(bed, context, depth_m, mobile_after_transfer)
    bed, deposit = apply_bed_demand(bed, context, demand)
    return bed, pickup, deposit
