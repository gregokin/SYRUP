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
    # CW-D2 opt-in: ask MAPLE's commit to return the pre-clear ledger's per-cell/class touch counter and gross
    # physical activity (`CommitResult.pre_commit_n_physical_touches` / `pre_commit_gross_physical_activity_kg`).
    # Default False: `commit_topography` is called exactly as before (the keyword is not even passed), so an older
    # MAPLE dependency without the keyword keeps working and every existing output is unchanged.
    record_exchange_history: bool = False
    # Window-only hard application-order journal bound; dense paths ignore no options.
    window_max_journal_records: int | None = None
    # CW-B108 (defaults LAST): deep-chunk storage encoding of every windowed publication, forwarded to MAPLE's
    # `publish_pending_column` only when not the default. 'full' (v1 chunks, the default) or 'overlay' (exact v2
    # final-value overlays; needs the accepted typed MAPLE source, see `deep_storage_support`). Storage only: the
    # logical stored values, physics and mass authority are unchanged.
    deep_chunk_encoding: str = 'full'
    deep_overlay_max_bytes: int | None = None


def maple_supports_exchange_history() -> bool:
    """Whether the bound MAPLE dependency's `commit_topography` accepts `record_exchange_history`
    (CW-D2). A static signature check: no commit, no state, no transfer."""
    import inspect

    from maple.surface.topographic_commit.commit import commit_topography

    return 'record_exchange_history' in inspect.signature(commit_topography).parameters


@dataclass(frozen=True)
class TerrainReference:
    """Immutable terrain evidence of one event: the full legacy DEM with its
    ring, the MAPLE committed elevation it was bound to, the export ring, the
    active mask, the friction and the cell size, plus (CW-C2, defaults LAST so
    every existing construction, checkpoint and digest is unchanged) the
    routing-graph build policy the event was started with: masked nodata
    (with its sentinel), opt-in pit storage and flat-sink storage. Every
    refreshed graph is built under exactly this policy."""

    initial_full_elevation_m: np.ndarray
    initial_committed_elevation_m: np.ndarray
    export_receiver_full: np.ndarray
    active_mask: np.ndarray
    friction_factor: np.ndarray
    dx_m: float
    allow_masked_nodata: bool = False
    nodata_value: float | None = None
    allow_pit_storage: bool = False
    allow_flat_storage: bool = False

    @property
    def strict(self) -> bool:
        return not (self.allow_masked_nodata or self.allow_pit_storage or self.allow_flat_storage
                    or self.nodata_value is not None)

    def build_options(self) -> dict[str, Any]:
        """Keyword arguments of `routing.build_routing_graph` for this policy."""
        return {'nodata_value': self.nodata_value, 'allow_masked_nodata': self.allow_masked_nodata,
                'allow_pit_storage': self.allow_pit_storage, 'allow_flat_storage': self.allow_flat_storage}


@dataclass(frozen=True)
class BedCommit:
    state: BedState
    graph: RoutingGraph
    did_commit: bool
    trigger: Any
    maple_result: Any = None
    water_result: Any = None
    # CW-C2: under an opt-in pit-storage policy a commit may legitimately create or remove terminal-storage cells;
    # the counts are recorded here (host ints) with the refreshed graph. None when no commit happened.
    terminal_storage_change: dict[str, int] | None = None
    # CW-B104 (defaults LAST): whether this commit refreshed the hydraulic routing (`graph` is then the new graph,
    # otherwise the caller's graph object, unchanged), the routing source the returned graph is built from (the new
    # committed elevation on a refresh; the caller's source otherwise; None for the source-less compatibility form),
    # and the post-commit max active-cell |committed - routing source| (m) that decided it (None without a source).
    rerouted: bool = True
    routing_source_m: Any = None
    net_displacement_m: float | None = None


_POLICY_TOKENS = {'masked_nodata', 'pit_storage', 'flat_storage'}


def graph_policy_flags(policy: str) -> dict[str, bool]:
    """The three opt-in flags encoded in `RoutingGraph.policy` (`"strict"` or
    `"masked_nodata=<bool>;pit_storage=<bool>[;flat_storage=True]"`); any
    other string is refused."""
    flags = {'allow_masked_nodata': False, 'allow_pit_storage': False, 'allow_flat_storage': False}
    if policy == 'strict':
        return flags
    if not isinstance(policy, str) or not policy:
        raise BedIntegrationError(f'unsupported routing graph policy {policy!r}')
    for token in policy.split(';'):
        name, _, value = token.partition('=')
        if name not in _POLICY_TOKENS or value not in ('True', 'False'):
            raise BedIntegrationError(f'unsupported routing graph policy {policy!r}')
        flags['allow_' + name] = value == 'True'
    if flags['allow_flat_storage'] and not flags['allow_pit_storage']:
        raise BedIntegrationError(f'unsupported routing graph policy {policy!r}')
    return flags


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
    apply = apply_water_process_demand
    window_options = {}
    if is_windowed_bed(state):
        from maple.water.windowed_step import apply_windowed_water_process_demand

        apply = apply_windowed_water_process_demand
        if context.window_max_journal_records is not None:
            window_options["max_journal_records"] = context.window_max_journal_records
    result = apply(
        state.voxel_column, state.active_layer, state.water, state.ledger, demand,
        g, context.grain_classes, context.mass_resolution_kg,
        sediment_availability=state.sediment_availability,
        initial_available_fraction=context.initial_available_fraction,
        adapter_name=ADAPTER_NAME, detachment_integration='rate_times_dt', **window_options)
    return replace(state, voxel_column=result.new_voxel_column, active_layer=result.new_active_layer,
                   water=result.new_water, ledger=result.new_ledger,
                   sediment_availability=result.new_sediment_availability), result


PAIRED_TRANSACTION_MODULE = 'maple.water.windowed_transaction'
PAIRED_TRANSACTION_ENTRY = '_apply_windowed_pickup_transport_deposit'


def paired_transaction_entry():
    """CW-B76: MAPLE's PRIVATE paired pickup -> transport callback -> deposit entry, or None ONLY when that exact
    module does not exist in the bound MAPLE (an older dependency such as M44/M67). Any other failure -- a missing
    module imported BY a present helper, a present module without the entry, an error raised while importing it --
    propagates; nothing is silently downgraded to the two-call path."""
    import importlib

    try:
        module = importlib.import_module(PAIRED_TRANSACTION_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == PAIRED_TRANSACTION_MODULE:
            return None
        raise
    return getattr(module, PAIRED_TRANSACTION_ENTRY)  # AttributeError propagates (present module, no entry)


def apply_bed_pickup_transport_deposit(state: BedState, context: BedContext, pickup_holder: list, between,
                                       *, entry) -> tuple[Any, BedState, Any]:
    """CW-B76 adapter: the two `apply_bed_demand` calls of one accounting step (pickup, then the deposit demand the
    `between(view)` transport callback returns) through MAPLE's private paired windowed transaction `entry`.

    The same caller-side contract as `apply_bed_demand`: the incoming water is validated first; the same geometry,
    classes, resolution, availability, initial fraction, adapter name, detachment integration and (only when set)
    `max_journal_records` are passed; the returned MAPLE state rebuilds the bed with `apply_bed_demand`'s exact
    `replace` (committed topography and every non-MAPLE field from `state`; no placeholder). Returns
    `(PickupView, new_bed, deposit_result)`. Windowed beds only."""
    from maple.water import validate_water_state

    if not is_windowed_bed(state):
        raise BedIntegrationError('the paired transaction needs a resident-window bed')
    g = context.geometry
    validate_water_state(state.water, g.ny, g.nx, len(context.grain_classes.classes))
    window_options = {}
    if context.window_max_journal_records is not None:
        window_options['max_journal_records'] = context.window_max_journal_records
    view, result = entry(
        state.voxel_column, state.active_layer, state.water, state.ledger, pickup_holder,
        g, context.grain_classes, context.mass_resolution_kg, between=between,
        sediment_availability=state.sediment_availability,
        initial_available_fraction=context.initial_available_fraction,
        adapter_name=ADAPTER_NAME, detachment_integration='rate_times_dt', **window_options)
    return view, replace(state, voxel_column=result.new_voxel_column, active_layer=result.new_active_layer,
                         water=result.new_water, ledger=result.new_ledger,
                         sediment_availability=result.new_sediment_availability), result


def bed_inventory(state: BedState):
    """Actual bed inventory by class via MAPLE's shared accounting helper."""
    from maple.aeolian.flux.step import bed_inventory_by_class_kg

    if is_windowed_bed(state):
        from maple.surface.voxels.windowed_inventory import (
            windowed_bed_inventory_by_class_kg,
        )

        return windowed_bed_inventory_by_class_kg(state.voxel_column, state.active_layer)
    return bed_inventory_by_class_kg(state.voxel_column, state.active_layer)


def terrain_reference(full_elevation_m, export_receiver_full, state: BedState,
                      graph: RoutingGraph, *, nodata_value=None) -> TerrainReference:
    """Bind the full legacy DEM to the MAPLE committed terrain and to `graph`.

    The graph's build policy (`graph.policy`: strict, masked nodata, pit /
    flat storage) is retained on the reference so every later refresh is
    built the same way; a masked-nodata graph needs its `nodata_value`
    (the sentinel the DEM holds on inactive / ring cells). The constant-datum
    check compares the MAPLE committed elevation with the DEM on EVERY
    interior cell for a strict graph (unchanged) and ONLY on the genuine
    active terrain for a masked-nodata graph, where inactive interior cells
    may hold the sentinel or a MAPLE placeholder; those cells are never a
    hydraulic receiver (the graph refuses it) and are never rewritten by a
    refresh. The committed elevation and the DEM must still be finite on
    every cell and have the full geometry."""
    from maple.core.backend import to_host
    from maple.core.types.topographic_commit import committed_elevation_m

    def fixed(array, dtype):
        # Terrain evidence is small, static and host-resident by contract; an input that lives on a device is
        # brought over ONCE here through MAPLE's counted `to_host` (a NumPy input is returned unchanged, so the
        # host path is exactly as before). Never an implicit NumPy conversion of a device array.
        result = np.array(to_host(array), dtype=dtype, copy=True)
        result.flags.writeable = False
        return result

    flags = graph_policy_flags(getattr(graph, 'policy', 'strict'))
    sentinel = None
    if nodata_value is not None:
        if isinstance(nodata_value, bool) or not isinstance(nodata_value, (int, float, np.integer, np.floating)):
            raise BedIntegrationError('nodata_value must be a real number')
        sentinel = float(nodata_value)
    if flags['allow_masked_nodata'] and sentinel is None:
        raise BedIntegrationError('a masked-nodata routing graph needs its nodata_value on the terrain reference')
    z = fixed(full_elevation_m, np.float64)
    exports = fixed(export_receiver_full, np.bool_)
    committed = fixed(committed_elevation_m(state.committed_topography), np.float64)
    if z.shape != (graph.shape[0] + 2, graph.shape[1] + 2) or exports.shape != z.shape:
        raise BedIntegrationError('terrain boundary ring does not match the routing grid')
    if committed.shape != graph.shape or not np.isfinite(committed).all():
        raise BedIntegrationError('MAPLE committed elevation does not match the routing grid')
    active_mask = fixed(graph.active, np.bool_)
    friction = fixed(graph.friction_factor, np.float64)
    # A constant datum offset is allowed; a different terrain shape is not.
    offset = committed - z[1:-1, 1:-1]
    tol = 128 * np.finfo(float).eps * max(1., float(np.abs(z).max()), float(np.abs(committed).max()))
    if flags['allow_masked_nodata']:
        if sentinel is not None and np.any((z[1:-1, 1:-1] == sentinel) & active_mask):
            raise BedIntegrationError('an active cell of the reference DEM holds the nodata sentinel')
        compared = offset[active_mask]
        datum_ok = np.isfinite(z).all() and np.max(np.abs(compared - compared.flat[0])) <= tol
    else:
        datum_ok = np.isfinite(z).all() and np.max(np.abs(offset - offset.flat[0])) <= tol
    if not datum_ok:
        raise BedIntegrationError('reference DEM differs from MAPLE terrain by more than a constant datum')
    options = {'nodata_value': sentinel, **{k: v for k, v in flags.items()}}
    rebuilt = build_routing_graph(z, exports, friction, graph.dx_m, active_mask=active_mask, xp=graph.xp, **options)
    # The compared graph fields are the host evidence arrays of RoutingGraph (aspect, slope, receiver, active,
    # outlet, and the terminal-storage mask); the explicit `to_host` keeps the comparison on the host whatever
    # namespace the graph was built in.
    if any(not np.array_equal(to_host(getattr(rebuilt, name)), to_host(getattr(graph, name)))
           for name in ('aspect', 'slope', 'receiver', 'active', 'outlet')):
        raise BedIntegrationError('terrain reference does not reproduce the initial graph')
    if rebuilt.policy != graph.policy or not np.array_equal(_pit_mask(rebuilt), _pit_mask(graph)):
        raise BedIntegrationError('terrain reference does not reproduce the initial graph policy / terminal storage')
    return TerrainReference(z, committed, exports, active_mask, friction, graph.dx_m, **options)


def _pit_mask(graph: RoutingGraph) -> np.ndarray:
    """Host `(ny, nx)` bool terminal-storage mask of a graph (all False when the graph carries none)."""
    pit = getattr(graph, 'pit_storage', None)
    if pit is None:
        return np.zeros(graph.shape, dtype=np.bool_)
    return np.asarray(pit, dtype=np.bool_).reshape(graph.shape)


def refresh_routing(reference: TerrainReference, state: BedState, *, xp=np) -> RoutingGraph:
    """Read authoritative committed terrain; hold the original outer ring
    fixed. Unsupported pits/flats/outlets raise before a new state is exposed.
    The graph is built under the reference's retained policy; under masked
    nodata ONLY the active interior cells follow the committed elevation
    (inactive interior cells keep their reference value, so a placeholder
    bed elevation can never become a low hydraulic receiver); the strict
    default refreshes every interior cell exactly as before.
    """
    from maple.core.types.topographic_commit import committed_elevation_m

    return routing_graph_from_elevation(reference, committed_elevation_m(state.committed_topography), xp=xp)


def routing_graph_from_elevation(reference: TerrainReference, committed_elevation, *, xp=np) -> RoutingGraph:
    """CW-B104: the routing graph built from an EXPLICIT committed-elevation source `(ny, nx)` (host or device;
    the held hydraulic routing source of an event, or the current MAPLE committed elevation), with exactly
    `refresh_routing`'s arithmetic and policy: `refresh_routing(reference, state)` is this function applied to
    `committed_elevation_m(state.committed_topography)`, byte for byte."""
    from maple.core.backend import to_host

    # ONE counted download of the committed elevation per refresh (the only evolving field read here).
    current = np.asarray(to_host(committed_elevation))
    if current.shape != reference.initial_committed_elevation_m.shape:
        raise BedIntegrationError('committed terrain shape changed')
    if not np.isfinite(current).all():
        raise BedIntegrationError('committed terrain is not finite')
    z = reference.initial_full_elevation_m.copy()
    if reference.allow_masked_nodata:
        interior = z[1:-1, 1:-1]
        interior[reference.active_mask] += (current - reference.initial_committed_elevation_m)[reference.active_mask]
    else:
        z[1:-1, 1:-1] += current - reference.initial_committed_elevation_m
    return build_routing_graph(z, reference.export_receiver_full, reference.friction_factor,
                               reference.dx_m, active_mask=reference.active_mask, xp=xp,
                               **reference.build_options())


def reroute_threshold_m(context: BedContext) -> float:
    """CW-B104: the hydraulic rerouting threshold -- the ACTUAL MAPLE `pending_elevation_trigger_m` of the bed
    context's commit spec (default 0.00075 m), a physical geometry threshold (never a conservation tolerance). It is
    compared with MAPLE's own pending-elevation convention, `>=` (`surface/topographic_commit/triggers.py`)."""
    value = float(context.topographic_commit_spec.pending_elevation_trigger_m)
    if not (np.isfinite(value) and value > 0.0):
        raise BedIntegrationError('pending_elevation_trigger_m must be finite and positive')
    return value


def current_routing_surface_m(state: BedState, context: BedContext):
    """CW-B104: the committed elevation `(ny, nx)` that MAPLE's commit would publish for this bed NOW (before any
    commit-stage callback), in the bed namespace, with the commit's own datum and arithmetic:
    `reference + (surface - reference)` where `surface` is the EXACT combined surface MAPLE's commit refresh uses --
    windowed: `windowed_committed_surface_elevation_m(..., trusted_packet=True)` (height packets plus the resident
    window and the active layer; no deep read, no voxel-depth scan; the same trusted call `stage_windowed_topography`
    makes); dense: `diagnose_combined_surface_state(...).combined_elevation_m` (the dense commit's own diagnostic; a
    full partition validation and depth sum -- the small dense path only). With an identity avalanche stage (windowed
    always; dense when avalanching is disabled) a commit of this bed publishes exactly this array. A derived
    geometry diagnostic, never a mass authority."""
    committed = state.committed_topography
    if is_windowed_bed(state):
        from maple.surface.topographic_commit.windowed_commit import (
            windowed_committed_surface_elevation_m,
        )

        surface = windowed_committed_surface_elevation_m(state.voxel_column, state.active_layer, context.geometry,
                                                         trusted_packet=True)
    else:
        from maple.surface.active_layer.diagnostics import (
            diagnose_combined_surface_state,
        )

        surface = diagnose_combined_surface_state(state.voxel_column, state.active_layer, context.geometry,
                                                  context.mass_resolution_kg).combined_elevation_m
    reference_m = committed.reference_elevation_m
    return reference_m + (surface - reference_m)


def net_routing_displacement_m(elevation_m, routing_source_m, active) -> float:
    """CW-B104: max over ACTIVE cells of |elevation - routing source| (m), a host float through ONE counted scalar
    read. Signed per cell (gains and losses within a cell cancel), absolute per cell, maximum over cells (different
    cells never cancel). A non-finite value is refused (a NaN would otherwise compare False and hide a reroute)."""
    from maple.core.backend import array_namespace, to_float

    xp = array_namespace(elevation_m, routing_source_m, active)
    value = to_float(xp.max(xp.where(active, xp.abs(elevation_m - routing_source_m), 0.0)))
    if not np.isfinite(value):
        raise BedIntegrationError('net elevation displacement from the routing source is not finite')
    return value


def check_routing_source(source, shape: tuple[int, int], xp) -> None:
    """CW-B104 cheap metadata/value checks of a held hydraulic routing source: a float64 `(ny, nx)` array of the
    event namespace with finite values (one counted flag read). Nothing is repaired or guessed."""
    from maple.core.backend import array_namespace, is_array, to_bool

    if source is None:
        raise BedIntegrationError('the event state carries no hydraulic routing source (routing_source_elevation_m); '
                                  'build it with initial_event_state or load it from a checkpoint that saved one; '
                                  'no reference is guessed from the current bed')
    if not is_array(source) or tuple(source.shape) != tuple(shape) or source.dtype != np.float64:
        raise BedIntegrationError(f'routing source must be a float64 {tuple(shape)} array')
    if array_namespace(source) is not xp:
        raise BedIntegrationError('routing source is not in the event namespace')
    if not to_bool(xp.all(xp.isfinite(source))):
        raise BedIntegrationError('routing source is not finite')


_GRAPH_EVIDENCE_FIELDS = ('aspect', 'slope', 'slope_before_edge_rule', 'edge_rule_applied', 'friction_factor',
                          'active', 'outlet', 'receiver', 'level', 'level_order_host', 'level_bounds', 'policy',
                          'pit_storage', 'flat_storage', 'dx_m', 'shape')


def check_graph_matches_routing_source(graph: RoutingGraph, reference: TerrainReference, source) -> None:
    """CW-B104 graph integrity: `graph` must be the graph `routing_graph_from_elevation(reference, source)` builds,
    compared on its HOST evidence fields (geometry, levels, policy, terminal storage). One host rebuild (no device
    work; one counted download of a device source); used at public entry / initialization, never per step."""
    from maple_syrup.routing import RoutingGraphError

    try:
        rebuilt = routing_graph_from_elevation(reference, source, xp=np)
    except RoutingGraphError as exc:
        raise BedIntegrationError(f'routing source does not build a valid routing graph ({exc}); refusing') from None
    check_graph_evidence_matches(rebuilt, graph)


def check_graph_evidence_matches(expected: RoutingGraph, graph: RoutingGraph) -> None:
    """CW-B104C1: `graph` must carry exactly the HOST evidence fields of `expected`, a graph already built from the
    held routing source (in any namespace: the evidence fields are host arrays). Lets a caller that has just built
    the source's graph (e.g. a namespace conversion) prove consistency without a second build."""
    for name in _GRAPH_EVIDENCE_FIELDS:
        a, b = getattr(expected, name), getattr(graph, name)
        if a is None or b is None:
            same = a is None and b is None or not np.any(a if a is not None else b)
        elif isinstance(a, np.ndarray):
            same = isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
        else:
            same = a == b
        if not same:
            raise BedIntegrationError(f'routing graph is not the graph of its held routing source ({name} differs); '
                                      'refusing (no refresh or repair is made)')


def commit_bed(state: BedState, context: BedContext, graph: RoutingGraph,
               reference: TerrainReference, time_s: float, *, force: bool = False,
               routing_source_m=None) -> BedCommit:
    """Compose MAPLE's actual trigger and commit APIs. A forced final
    commit uses the real current time. No wind scheduler/hop table is needed.

    All candidate outputs, callback boxes and refreshed graphs stay local
    until the full operation succeeds. A late graph failure does not publish
    any bed, ledger, availability or water change into the caller's state.

    CW-B104: with the event's held hydraulic `routing_source_m` (the committed
    elevation `graph` was built from) the commit is an ACCOUNTING / geometry
    publication only, and the routing is refreshed iff the max active-cell
    |new committed elevation - routing source| reaches `reroute_threshold_m`
    (`>=`); otherwise the caller's graph object is returned unchanged
    (`rerouted=False`). Without a source (None; low-level compatibility form,
    never used by the event driver) every commit refreshes as before.
    """
    from maple.core.backend import array_namespace, to_bool
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.surface.avalanche.callback import (
        make_availability_aware_avalanche_callback,
    )
    from maple.surface.topographic_commit.commit import commit_topography
    from maple.surface.topographic_commit.triggers import evaluate_commit_triggers
    from maple.water.commit_callback import make_water_depth_commit_callback

    if is_windowed_bed(state):
        if bool(getattr(context.avalanching_spec, 'enabled', False)):
            raise BedIntegrationError('windowed commits require explicitly disabled avalanching')
        from maple.surface.topographic_commit.windowed_commit import (
            stage_windowed_topography,
        )
        from maple.water.windowed_commit_callback import (
            make_windowed_water_depth_commit_callback,
        )

        commit_topography = stage_windowed_topography
        make_water_depth_commit_callback = make_windowed_water_depth_commit_callback
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
    if routing_source_m is not None:
        check_routing_source(routing_source_m, graph.shape, graph.xp)
        threshold = reroute_threshold_m(context)
    if reference.allow_masked_nodata and bool(getattr(context.avalanching_spec, 'enabled', False)):
        # MAPLE's avalanching knows the whole grid, not SYRUP's inactive / nodata mask: a slope relaxation could
        # move bed mass into or out of masked cells. Refused until that interaction is qualified separately.
        raise BedIntegrationError('avalanching inside commits is not supported with a masked-nodata terrain '
                                  '(unqualified mass exchange across the mask); disable avalanching or the mask')
    if not isinstance(context.record_exchange_history, bool):
        raise BedIntegrationError('record_exchange_history must be bool')
    history_kwargs = {}
    if context.record_exchange_history:
        # Refused BEFORE the trigger evaluation and the commit: an older MAPLE cannot record the history and
        # nothing may be published on an unsupported dependency when the caller asked for it.
        if not maple_supports_exchange_history():
            raise BedIntegrationError('the bound MAPLE dependency cannot record the exchange history '
                                      '(commit_topography has no record_exchange_history keyword); no commit made')
        history_kwargs['record_exchange_history'] = True
    trigger = evaluate_commit_triggers(state.ledger, state.committed_topography, context.geometry,
                                      time_s, context.topographic_commit_spec)
    if not force and not trigger.should_commit:
        return BedCommit(state, graph, False, trigger)
    avalanche, avalanche_box = make_availability_aware_avalanche_callback(
        context.avalanching_spec, state.sediment_availability, context.initial_available_fraction)
    water, water_box = make_water_depth_commit_callback(
        state.water, context.geometry, committed_elevation_m(state.committed_topography),
        depth_update_rule=context.depth_update_rule,
        **({'trusted_packet': True} if is_windowed_bed(state) else {}))
    result = commit_topography(state.ledger, state.committed_topography, state.voxel_column,
                               state.active_layer, context.geometry, context.mass_resolution_kg,
                               time_s, avalanche_callback=avalanche, water_callback=water,
                               **({'trusted_packet': True} if is_windowed_bed(state) else {}), **history_kwargs)
    if context.record_exchange_history and (
            result.pre_commit_n_physical_touches is None or result.pre_commit_gross_physical_activity_kg is None
            or result.exchange_history_receipt is None):
        raise BedIntegrationError('MAPLE commit returned no exchange history although it was requested; '
                                  'nothing published')
    water_box.publish()  # local box only; nothing is returned until graph validation
    xp = array_namespace(state.water.depth_m)
    if (not water_box.ran or water_box.depth_update.rule != 'constant_depth'
            or not to_bool(xp.all(water_box.water.depth_m == state.water.depth_m))):
        raise BedIntegrationError('MAPLE commit changed solver-owned water depth')
    candidate = replace(state, voxel_column=result.new_voxel_column, active_layer=result.new_active_layer,
                        committed_topography=result.new_committed_state, ledger=result.new_ledger,
                        sediment_availability=avalanche_box.new_sediment_availability, water=water_box.water)
    if routing_source_m is None:
        updated_graph, rerouted, source, net = refresh_routing(reference, candidate, xp=graph.xp), True, None, None
    else:
        # The new authoritative committed elevation (validated, freshly computed: never a view of the state).
        committed_now = committed_elevation_m(candidate.committed_topography)
        net = net_routing_displacement_m(committed_now, routing_source_m, graph.active_flat.reshape(graph.shape))
        rerouted = net >= threshold
        if rerouted:
            updated_graph, source = routing_graph_from_elevation(reference, committed_now, xp=graph.xp), committed_now
        else:
            updated_graph, source = graph, routing_source_m
            del committed_now
    old_pit, new_pit = _pit_mask(graph), _pit_mask(updated_graph)
    change = {'n_terminal_before': int(old_pit.sum()), 'n_terminal_after': int(new_pit.sum()),
              'n_terminal_created': int(np.sum(new_pit & ~old_pit)), 'n_terminal_removed': int(np.sum(old_pit & ~new_pit))}
    return BedCommit(candidate, updated_graph, True, trigger, result, water_box, change,
                     rerouted=rerouted, routing_source_m=source, net_displacement_m=net)


def is_windowed_bed(state: BedState) -> bool:
    from maple.core.types.voxel import WindowedVoxelColumnState

    return isinstance(state.voxel_column, WindowedVoxelColumnState)


def bed_by_cell_class(state: BedState):
    if is_windowed_bed(state):
        from maple.surface.voxels.windowed_inventory import column_mass_by_class_kg

        return column_mass_by_class_kg(state.voxel_column) + state.active_layer.mass_kg
    return state.voxel_column.mass_kg.sum(axis=2) + state.active_layer.mass_kg


def bed_voxel_count(state: BedState) -> int:
    return int(state.voxel_column.n_voxels if is_windowed_bed(state) else state.voxel_column.mass_kg.shape[2])


def bed_already_committed(state: BedState, context: BedContext, time_s: float) -> bool:
    from maple.surface.topographic_commit.commit import is_already_committed

    if is_windowed_bed(state):
        from maple.surface.topographic_commit.windowed_commit import (
            is_windowed_already_committed,
        )

        # A trusted staged bed necessarily needs final publication/commit. Public
        # MAPLE already-committed never blesses an unpublished journal.
        if state.voxel_column.pending_deep is not None:
            return False
        is_already_committed = is_windowed_already_committed
    return is_already_committed(state.ledger, state.committed_topography, state.voxel_column,
                                state.active_layer, context.geometry, context.mass_resolution_kg, time_s)


def publish_bed_storage(state: BedState, context: BedContext) -> BedState:
    """Last operation of an accepted transaction, after all prospective checks. CW-B108: the context's deep-chunk
    encoding options are forwarded ONLY when not the default, so a default-full publication is exactly the call
    every earlier dependency (M94) accepts."""
    if not is_windowed_bed(state):
        return state
    from maple.surface.voxels.windowed_pending import publish_pending_column

    options = {} if context.window_max_journal_records is None else {"max_records": context.window_max_journal_records}
    options.update(deep_storage_publish_options(context))
    column = publish_pending_column(state.voxel_column, context.geometry, context.mass_resolution_kg, **options)
    return replace(state, voxel_column=column)


# --- CW-B108: optional exact-overlay deep storage (storage adapter only; no bed physics) ---------------------------------
DEEP_CHUNK_ENCODINGS = ('full', 'overlay')


def deep_storage_is_default(context: BedContext) -> bool:
    return context.deep_chunk_encoding == 'full' and context.deep_overlay_max_bytes is None


def deep_storage_publish_options(context: BedContext) -> dict[str, Any]:
    """The `publish_pending_column` keywords of the context's encoding: none for the default (`'full'`, no bound)."""
    if deep_storage_is_default(context):
        return {}
    return {'encoding': context.deep_chunk_encoding, 'overlay_max_bytes': context.deep_overlay_max_bytes}


def deep_storage_record(context: BedContext) -> dict[str, Any] | None:
    """JSON identity / metadata record of a NON-default encoding (None for the default, so default identities,
    configurations and checkpoint manifests are unchanged)."""
    if deep_storage_is_default(context):
        return None
    return {'deep_chunk_encoding': context.deep_chunk_encoding, 'deep_overlay_max_bytes': context.deep_overlay_max_bytes}


def check_deep_storage_options(encoding, overlay_max_bytes) -> None:
    """Pure value checks (no MAPLE import): `encoding` in DEEP_CHUNK_ENCODINGS; the bound is None for 'full' and None
    or a positive int (never a bool) for 'overlay'. Nothing is defaulted or coerced."""
    if not isinstance(encoding, str) or encoding not in DEEP_CHUNK_ENCODINGS:
        raise BedIntegrationError(f'deep_chunk_encoding must be one of {DEEP_CHUNK_ENCODINGS}, got {encoding!r}')
    if overlay_max_bytes is None:
        return
    if encoding != 'overlay':
        raise BedIntegrationError('deep_overlay_max_bytes is supported only with deep_chunk_encoding=\'overlay\'')
    # A plain positive int only (it is JSON identity / checkpoint metadata): bools and NumPy scalars are refused.
    if type(overlay_max_bytes) is not int or overlay_max_bytes < 1:
        raise BedIntegrationError(f'deep_overlay_max_bytes must be None or a positive int, got {overlay_max_bytes!r}')


def check_deep_storage_source(encoding, overlay_max_bytes) -> None:
    """Value checks, then (non-default only) the bound MAPLE source: overlay publication is supported ONLY by the
    accepted typed source (`TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST`). The default needs no MAPLE read."""
    check_deep_storage_options(encoding, overlay_max_bytes)
    if (encoding, overlay_max_bytes) == ('full', None):
        return
    if bound_maple_package_digest() != TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST:
        raise BedIntegrationError('deep_chunk_encoding=\'overlay\' is supported only with the accepted MAPLE source '
                                  f'{TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST[:16]}...; the bound MAPLE package differs')


def validate_deep_storage(context: BedContext, state: BedState | None = None) -> None:
    """CW-B108 pre-mutation validation of the context's deep-chunk storage option (read-only; no publication).

    Default ('full', no bound): only the published-version compatibility check below runs (a v2 store version cannot
    take a default-full publication; MAPLE would refuse it at the first publication). 'overlay' additionally requires:
    a resident-window bed (dense beds are refused, not silently ignored), the accepted typed MAPLE source
    (`TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST`, whose publishers accept the options), and the MAPLE store's own option
    parser to accept the encoding for this store header (`deep_store._resolve_encoding`, e.g. its voxel-count rule).
    No fallback encoding is chosen."""
    check_deep_storage_source(context.deep_chunk_encoding, context.deep_overlay_max_bytes)
    windowed = state is not None and is_windowed_bed(state)
    if state is not None and not windowed and not deep_storage_is_default(context):
        raise BedIntegrationError('deep_chunk_encoding=\'overlay\' needs a resident-window MAPLE bed; a dense bed has '
                                  'no deep store (refused, not ignored)')
    if not windowed:
        return
    store = state.voxel_column.deep_store
    if deep_storage_is_default(context):
        if getattr(store, 'format_version', 1) != 1:
            raise BedIntegrationError('the bed\'s deep store version is an overlay (v2) version; the default '
                                      'deep_chunk_encoding=\'full\' cannot publish on it (no cross-encoding continuation)')
        return
    from maple.surface.voxels.deep_store import _resolve_encoding

    try:
        _resolve_encoding(context.deep_chunk_encoding, context.deep_overlay_max_bytes, store.header,
                          operation='SYRUP deep storage option')
    except ValueError as exc:
        raise BedIntegrationError(str(exc)) from None


# --- CW-B102: opt-in deferred deep publication (storage adapters only; no bed physics) ----------------------------------
#: The ONLY MAPLE package whose append-cap refusal text the deferred-publication recognizer is bound to (M94).
JOURNAL_RECOGNIZER_MAPLE_DIGEST = 'eb14da66e0ffee1e11cd37cc98d3379b01b9e8f54e562ad0d12fc22afb11abd6'
_MAPLE_DIGEST_CACHE: dict[str, str] = {}


def pending_journal_records(state: BedState) -> int:
    """Records of the MAPLE pending deep journal a windowed bed carries (0 when published / not windowed)."""
    if not is_windowed_bed(state):
        return 0
    pending = state.voxel_column.pending_deep
    return 0 if pending is None else int(pending.journal.n_records)


def resolved_journal_cap(context: BedContext) -> int:
    """The application-order journal cap MAPLE actually enforces for this context: the context's explicit
    `window_max_journal_records`, else the windowed APIs' own default (`deep_store.DEFAULT_MAX_JOURNAL_RECORDS`)."""
    if context.window_max_journal_records is not None:
        return int(context.window_max_journal_records)
    from maple.surface.voxels.deep_store import DEFAULT_MAX_JOURNAL_RECORDS

    return int(DEFAULT_MAX_JOURNAL_RECORDS)


def bound_maple_package_digest() -> str:
    """SHA-256 tree digest of the IMPORTED `maple` package (computed once per process)."""
    if 'digest' not in _MAPLE_DIGEST_CACHE:
        from pathlib import Path

        import maple

        from maple_syrup.provenance import source_tree_digest

        _MAPLE_DIGEST_CACHE['digest'] = source_tree_digest(Path(maple.__file__).resolve().parent).digest_sha256
    return _MAPLE_DIGEST_CACHE['digest']


#: CW-B108: the independently accepted MAPLE package (B107 review freeze) whose `_require_journal_room` raises the
#: structured `deep_store.JournalCapacityError` (same message, validated record counts). Same digest convention as M94.
TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST = '24c6e1478ed732df22f448eddf8cb16c3b72f2406dc9b37a9414cee9a2d1a9ce'
JOURNAL_RECOGNIZER_SOURCES = {JOURNAL_RECOGNIZER_MAPLE_DIGEST: 'm94_pinned_message',
                              TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST: 'typed_journal_capacity_error'}


def journal_recognizer_source() -> str | None:
    """CW-B108: which append-cap recognizer the BOUND MAPLE package supports -- `'m94_pinned_message'` (M94, exact
    message), `'typed_journal_capacity_error'` (the accepted typed source), or None (unrecognized source: the deferred
    'boundary' publication is refused at startup). Selected by the package digest, never by probing behaviour."""
    return JOURNAL_RECOGNIZER_SOURCES.get(bound_maple_package_digest())


def require_typed_journal_capacity_error() -> type:
    """The accepted source's `JournalCapacityError` class, checked structurally at startup: it must be the exact
    object `windowed_transfer` raises (imported there from `deep_store`) and a `ValueError` subclass."""
    from maple.surface.voxels import deep_store, windowed_transfer

    cls = getattr(deep_store, 'JournalCapacityError', None)
    if (not isinstance(cls, type) or not issubclass(cls, ValueError)
            or getattr(windowed_transfer, 'JournalCapacityError', None) is not cls):
        raise BedIntegrationError('the bound MAPLE source does not provide the typed JournalCapacityError of the '
                                  'accepted source; refusing the typed append-cap recognizer')
    return cls


def _typed_counts_recognized(exc: BaseException, cap: int, carried_records: int) -> bool:
    """The accepted source's refusal with VALID structured counts and the context's actual cap, raised by an attempt
    that started from the carried journal (`carried <= current`) and that one flush can resolve: the step's own
    records `current - carried + new` must fit the cap (otherwise it is a single-step overflow and propagates)."""
    counts = [getattr(exc, name, None) for name in ('current_records', 'new_records', 'max_records')]
    if any(type(value) is not int for value in counts):
        return False
    current, new, maximum = counts
    return (maximum == cap and new >= 1 and 0 < carried_records <= current <= maximum < current + new
            and current - carried_records + new <= maximum
            and str(exc) == f'pending journal exceeds max_journal_records={cap}')


def is_carried_append_cap_refusal(exc: BaseException, context: BedContext, carried_records: int) -> bool:
    """True ONLY for the append-cap refusal (`windowed_transfer._require_journal_room`) raised while the accepted
    state carries pending records, source-qualified by the bound MAPLE package:

    * M94 (`JOURNAL_RECOGNIZER_MAPLE_DIGEST`): exactly `ValueError` with the message
      ``pending journal exceeds max_journal_records=<cap>`` and the context's actual cap (unchanged);
    * the accepted typed source (`TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST`): exactly its `JournalCapacityError` with
      integer `current/new/max_records`, `max_records` the context's actual cap, the unchanged message, counts
      consistent with an attempt started from the carried journal, and a step that fits the cap on its own.

    Everything else -- an unrecognized source, the oversized prior-journal refusal, the canonical / apply caps, other
    ValueErrors or types, malformed / inconsistent counts, a refusal with nothing carried, a single-step overflow -- is
    not recognized and must propagate."""
    if isinstance(carried_records, bool) or not isinstance(carried_records, int) or carried_records <= 0:
        return False
    source = journal_recognizer_source()
    cap = resolved_journal_cap(context)
    if source == 'm94_pinned_message':
        return type(exc) is ValueError and str(exc) == f'pending journal exceeds max_journal_records={cap}'
    if source == 'typed_journal_capacity_error':
        return type(exc) is require_typed_journal_capacity_error() and _typed_counts_recognized(exc, cap,
                                                                                                carried_records)
    return False
