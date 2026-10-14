"""Namespace helpers for the completion / checkpoint layer (CW-C1).

The coupled event state keeps its physical arrays in ONE namespace (`state.graph.xp`, NumPy or CuPy). The
completion driver and the checkpoint layer need three things from that namespace that the generic MAPLE tree
converters do not provide:

* `host_packet`: ONE counted device-to-host read of a few 0-d reductions (the compact audit packet of the
  completion criteria and the water budget), instead of one implicit synchronizing `float()` per value;
* `arrays_equal`: an equality decision between two arrays of the same namespace read through MAPLE's counted
  `to_bool`, never an implicit NumPy conversion of a device array;
* `item_to_namespace`: an explicit, field-by-field reconstruction of a host continuation (`EventProgress` or
  `DryHandoff`) in a target namespace. `to_device_tree` / `to_host_tree` walk dataclass fields and leave module-valued
  `xp` attributes (`RoutingGraph`, `TransportNetwork`, `PhysicsGrid`, `PhaseState`, `ColumnParameters`) unchanged,
  so those objects are REBUILT here: the routing graph from the host terrain reference and the state's HELD
  hydraulic routing source (CW-B104, `sediment_bed.routing_graph_from_elevation`, same host arithmetic as the
  original build, then placed in `xp`; never the latest committed bed), the
  transport network and physics grid from that graph, the phase state from converted fraction / position arrays.
  Host-by-contract objects stay host: `TerrainReference`, the planned / reporting boundaries, the graph's evidence
  arrays (aspect, slope, active, outlet, ...), the integer counters, the commit / rejection logs, the accounting.

Nothing here changes a value: the NumPy target is the identity (no copy, no rebuild), so the accepted CPU path is
byte-for-byte unchanged; cross-namespace reconstruction is a counted transfer of every physical array.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from types import ModuleType
from typing import Any

import numpy as np

from maple_syrup.characteristic_sparse import (
    SparsePhaseState,
    empty_sparse_phase_state,
    host_network_evidence,
    slot_maps_from_host_evidence,
)
from maple_syrup.characteristic_transport import PhaseState
from maple_syrup.sediment_bed import (
    BedIntegrationError,
    check_graph_evidence_matches,
    check_routing_source,
    routing_graph_from_elevation,
)
from maple_syrup.sediment_event import (
    SedimentEventError,
    SedimentEventResult,
    SedimentEventState,
)
from maple_syrup.sediment_physics import physics_grid_from_graph
from maple_syrup.sediment_transport import transport_network
from maple_syrup.storm import StormState

__all__ = [
    "BACKEND_NAMES",
    "arrays_equal",
    "backend_name",
    "host_packet",
    "item_to_namespace",
    "namespace_of",
    "result_to_namespace",
    "state_to_namespace",
]

BACKEND_NAMES = ("numpy", "cupy")


def backend_name(xp: ModuleType) -> str:
    """`"numpy"` or `"cupy"` for an array namespace; anything else is refused."""
    name = getattr(xp, "__name__", None)
    if name not in BACKEND_NAMES:
        raise SedimentEventError(f"unsupported array namespace {xp!r}; expected one of {BACKEND_NAMES}")
    return name


def namespace_of(state: SedimentEventState) -> ModuleType:
    return state.graph.xp


def arrays_equal(a: Any, b: Any) -> bool:
    """`array_equal` of two arrays of ONE namespace, read through MAPLE's counted `to_bool` (a NumPy pair is the
    plain `np.array_equal`); a mixed pair is refused, never converted implicitly."""
    from maple.core.backend import array_namespace, to_bool

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return bool(np.array_equal(a, b))
    xp = array_namespace(a, b)  # raises MixedArrayNamespaceError on a host/device pair
    return to_bool(xp.array_equal(a, b))


def host_packet(xp: ModuleType, scalars: list[Any], *vectors: Any) -> np.ndarray:
    """ONE counted host read of a few 0-d namespace values followed by optional 1-d vectors, as one float64
    array (bools become 1.0 / 0.0). On NumPy the values are the very reductions (no arithmetic is added)."""
    from maple.core.backend import to_host

    parts = [xp.stack([xp.asarray(v, dtype=np.float64) for v in scalars])] if scalars else []
    parts += [xp.asarray(v, dtype=np.float64).reshape(-1) for v in vectors]
    return np.asarray(to_host(xp.concatenate(parts) if len(parts) > 1 else parts[0]), dtype=np.float64)


def _to(value: Any, xp: ModuleType) -> Any:
    from maple.core.backend import is_array, to_device

    return to_device(value, xp) if is_array(value) else value


def state_to_namespace(state: SedimentEventState, xp: ModuleType) -> SedimentEventState:
    """The event state with every physical array in `xp` and its xp-bearing objects REBUILT there. The NumPy target
    of a NumPy state is the identity. The graph keeps the state's binding (`input_sha256`); its arrays are the same
    host arithmetic as the original build placed in `xp`. CW-B104C1: a state whose graph is not the graph of its
    held routing source is refused (never rebuilt into a different routing under the old binding)."""
    from maple.core.backend import to_device, to_device_tree

    if not isinstance(state, SedimentEventState):
        raise SedimentEventError("state must be a SedimentEventState")
    backend_name(xp)
    if state.graph.xp is xp:
        return state
    storm = StormState(state.storm.t_s, to_device(state.storm.depth_m, xp), to_device(state.storm.soil_water_m, xp),
                       to_device(state.storm.discharge_m2_s, xp))
    bed = to_device_tree(state.bed, xp)  # MAPLE state dataclasses carry no xp attribute (as bed_from_case)
    from maple_syrup.sediment_bed import is_windowed_bed

    if is_windowed_bed(state.bed):
        from maple.surface.voxels.windowed_pending import require_published_column
        from maple.surface.voxels.windowed_prefix import move_prefix_packets

        require_published_column(state.bed.voxel_column, operation='state_to_namespace')
        bed = replace(bed, voxel_column=move_prefix_packets(bed.voxel_column, state.bed.voxel_column, xp))

    velocity = to_device(state.sediment_velocity_m_s, xp)
    # CW-B104: the graph of the HELD routing source (never the latest committed bed); the source moves with the state.
    try:
        check_routing_source(state.routing_source_elevation_m, state.graph.shape, state.graph.xp)
    except BedIntegrationError as exc:
        raise SedimentEventError(str(exc)) from None
    source = to_device(state.routing_source_elevation_m, xp)
    graph = routing_graph_from_elevation(state.terrain, state.routing_source_elevation_m, xp=xp)
    # CW-B104C1: the conversion never repairs routing. The target graph just built from the held source must carry
    # the input graph's host evidence (no second build); a source / graph mismatch is refused before the binding is
    # carried over or anything is returned.
    try:
        check_graph_evidence_matches(graph, state.graph)
    except BedIntegrationError as exc:
        raise SedimentEventError(f"state_to_namespace: {exc}") from None
    graph = replace(graph, input_sha256=state.graph.input_sha256)
    network = transport_network(graph)
    phase = state.phase
    if isinstance(phase, SparsePhaseState):
        # CSR boundary conversion: the four CSR arrays are moved with counted transfers; the slot maps become the
        # target network's own (built once from host evidence, bound to the graph digest the state already carries)
        # ONLY after the source state's maps are shown equal to that host evidence -- a bundle whose maps disagree
        # is refused here, never silently repaired by the rebuild. No dense array is formed.
        from maple.core.backend import to_host

        if phase.network_sha256 != network.graph_input_sha256:
            raise SedimentEventError("CSR phase state is bound to a different graph than the state's network")
        evidence = host_network_evidence(network)
        maps = slot_maps_from_host_evidence(evidence.active, evidence.receiver_index, evidence.outlet)
        if (not np.array_equal(np.asarray(to_host(phase.cell_of_slot)), maps.cell_of_slot)
                or not np.array_equal(np.asarray(to_host(phase.slot_of_cell)), maps.slot_of_cell)):
            raise SedimentEventError("CSR phase slot maps disagree with the network's active cells; refusing the "
                                     "namespace conversion (nothing is repaired implicitly)")
        template = empty_sparse_phase_state(network, int(phase.n_classes), int(phase.n_bins))
        phase = replace(template, row_ptr=to_device(phase.row_ptr, xp), bin=to_device(phase.bin, xp),
                        fraction=to_device(phase.fraction, xp), position_m=to_device(phase.position_m, xp))
    elif phase is not None:
        phase = PhaseState(fraction=to_device(phase.fraction, xp), position_m=to_device(phase.position_m, xp),
                           dx_m=float(phase.dx_m), n_bins=int(phase.n_bins), xp=xp)
    return SedimentEventState(storm=storm, bed=bed, sediment_velocity_m_s=velocity, graph=graph,
                              network=network, grid=physics_grid_from_graph(graph),
                              terrain=state.terrain, phase=phase, routing_source_elevation_m=source)


def result_to_namespace(result: SedimentEventResult, xp: ModuleType) -> SedimentEventResult:
    """Every array-valued accumulator of a result in `xp` (dicts of arrays included), the state rebuilt, the host
    `boundaries` and every non-array field untouched. Identity for a result already in `xp`."""
    if not isinstance(result, SedimentEventResult):
        raise SedimentEventError("result must be a SedimentEventResult")
    if result.state.graph.xp is xp:
        return result
    updates: dict[str, Any] = {"state": state_to_namespace(result.state, xp)}
    for f in dataclasses.fields(result):
        if f.name in ("state", "boundaries"):
            continue
        value = getattr(result, f.name)
        if isinstance(value, dict):
            updates[f.name] = {k: _to(v, xp) for k, v in value.items()}
        else:
            updates[f.name] = _to(value, xp)
    return replace(result, **updates)


def item_to_namespace(item: Any, xp: ModuleType) -> Any:
    """A wet `EventProgress` or a completed `DryHandoff` with every physical array in `xp` (see the module
    docstring). Identity (the very object, no `replace`) for a supported item whose states already live in
    `xp`; the namespace is validated and an unsupported item type is refused before either path. A mixed
    item (progress state and dry state in different namespaces) is not claimed to be in `xp`: it goes through the
    field-by-field conversion and the checkpoint validator refuses it afterwards."""
    from maple_syrup.complete_event import DryHandoff, EventProgress

    backend_name(xp)
    if isinstance(item, DryHandoff):
        if item.progress.result.state.graph.xp is xp and item.dry_state.graph.xp is xp:
            return item
        progress = item_to_namespace(item.progress, xp)
        return replace(item, progress=progress, dry_state=state_to_namespace(item.dry_state, xp),
                       pre_reset_surface_m=_to(item.pre_reset_surface_m, xp),
                       pre_reset_soil_m=_to(item.pre_reset_soil_m, xp),
                       terminal_deposition_kg=_to(item.terminal_deposition_kg, xp))
    if isinstance(item, EventProgress):
        if item.result.state.graph.xp is xp:
            return item
        return replace(item, result=result_to_namespace(item.result, xp))
    raise SedimentEventError("item must be an EventProgress or a DryHandoff")
