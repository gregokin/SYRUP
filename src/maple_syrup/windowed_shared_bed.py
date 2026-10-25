"""Opt-in K1 NumPy water on the genuine Native windowed bed.

This module constructs no landscape, callback, solver, or restart codec.  The
ordinary six-field SYRUP family does not import it.  A Source tag is metadata;
Root must separately qualify the locked Native/provider/runtime before use.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

from maple_syrup.sediment_bed import BedContext, BedIntegrationError, BedState


@dataclass(frozen=True)
class WindowedSharedBedState(BedState):
    reconciliation_baseline: object


@dataclass(frozen=True)
class WindowedWaterBridgeLimits:
    """Required logical declarations within the existing bounded CPU contract.

    They are refusal ceilings, never runtime/provider/physics success evidence.
    The run directory is the actual owned directory, including old versions,
    partials and other artifacts.  No file is removed or deduplicated here.
    """
    max_host_bytes: int
    max_mapped_bytes: int
    max_logical_read_bytes: int
    max_work: int
    max_store_bytes: int
    max_run_bytes: int
    max_new_versions: int
    max_entries: int
    max_external_backing_bytes: int
    max_event_steps: int
    max_retries: int
    max_transport_substeps: int
    max_report_rows: int
    reserve_host_bytes: int
    run_dir: object

    def __post_init__(self):
        caps = {'max_host_bytes': 4 << 30, 'max_mapped_bytes': 4 << 30,
                'max_store_bytes': 64 << 20, 'max_run_bytes': 128 << 20,
                'max_entries': 4096, 'max_external_backing_bytes': 4 << 30,
                'reserve_host_bytes': 4 << 30}
        for field in fields(type(self)):
            if field.name == 'run_dir':
                continue
            value = getattr(self, field.name)
            if type(value) is not int or not 0 < value <= caps.get(field.name, (1 << 63) - 1):
                _refuse('bridge_limit')


@dataclass(frozen=True, kw_only=True)
class WindowedAvalancheContext(BedContext):
    native_iteration_limits: object
    native_evidence_limits: object
    native_solve_limits: object
    native_validation_limits: object
    native_publication_limits: object
    native_qualification_limits: object
    source_identity: str
    bridge_limits: WindowedWaterBridgeLimits


@dataclass(frozen=True, eq=False)
class _Admission:
    original: object
    storage_plan: object
    provider: object
    array_prepasses: tuple
    payload_terms: tuple
    mapped_bytes: int
    logical_read_bytes: int
    work_terms: tuple
    disk_terms: tuple


_BED_FIELDS = ('voxel_column', 'active_layer', 'water', 'ledger',
               'committed_topography', 'sediment_availability', 'reconciliation_baseline')
_NATIVE_POLICIES = ('native_iteration_limits', 'native_evidence_limits',
                    'native_solve_limits', 'native_validation_limits',
                    'native_publication_limits', 'native_qualification_limits')
_GRAPH_ARRAYS = ('aspect', 'slope', 'slope_before_edge_rule', 'edge_rule_applied',
                 'friction_factor', 'active', 'outlet', 'receiver', 'level',
                 'level_order_host', 'conveyance', 'active_flat', 'outlet_flat',
                 'level_order', 'conveyance_lo', 'donor_position', 'donor_mask',
                 'pit_storage', 'flat_storage')
_TERRAIN_ARRAYS = ('initial_full_elevation_m', 'initial_committed_elevation_m',
                   'export_receiver_full', 'active_mask', 'friction_factor')
_NETWORK_ARRAYS = ('receiver_index', 'outlet_flat', 'active_flat', 'cells_x',
                   'rows_x', 'faces_x', 'sign_x', 'cells_y', 'faces_y', 'cols_y',
                   'sign_y', 'receiver_host', 'outlet_host', 'active_host',
                   'terminal_flat', 'terminal_host')
_COLUMN_ARRAYS = ('ksat_m_per_s', 'suction_m', 'drainage_parameter', 'theta_sat',
                  'soil_thickness_m', 'storage_max_m', 'lambda_m_per_s', 'active_mask')
_SEDIMENT_ARRAYS = ('diameter_m', 'raindrop_a', 'raindrop_b', 'raindrop_c',
                    'raindrop_depth_attenuation_per_cm', 'raindrop_max_depth_per_reference_s_m',
                    'settling_velocity_m_s', 'particle_mass_g', 'd_diameter',
                    'd_raindrop_a', 'd_raindrop_b', 'd_raindrop_c', 'd_attenuation',
                    'd_raindrop_max', 'd_settling', 'd_particle_mass_g')
# Fixed Source-derived, loop-expanded visible request census.  The independent
# Source auditor regenerates these exact polynomials from the ten owning files.
# Variables are C,B,S,L,A,R,Q; payload excludes existing out= output buffers,
# while work retains their byte visits.  Native B/N and opaque providers remain
# separate positive terms/prerequisites.  These are NOT executed-model counts.
_VISIBLE_REQUEST_CENSUS = {'attempt': (((3852, (0, 0, 0, 0, 0, 0, 0)), (1172, (0, 0, 0, 1, 0, 0, 0)), (190, (0, 0, 1, 0, 0, 0, 0)), (190, (0, 0, 1, 1, 0, 0, 0)), (39, (1, 0, 0, 0, 0, 0, 0)), (1, (1, 1, 0, 0, 0, 0, 0))), ((3852, (0, 0, 0, 0, 0, 0, 0)), (1172, (0, 0, 0, 1, 0, 0, 0)), (190, (0, 0, 1, 0, 0, 0, 0)), (190, (0, 0, 1, 1, 0, 0, 0)), (45, (1, 0, 0, 0, 0, 0, 0)), (9, (1, 1, 0, 0, 0, 0, 0)))), 'commit_routing': (((2068, (0, 0, 0, 0, 0, 0, 0)), (130, (1, 0, 0, 0, 0, 0, 0))), ((2068, (0, 0, 0, 0, 0, 0, 0)), (130, (1, 0, 0, 0, 0, 0, 0)))), 'complete_entry': (((830, (0, 0, 0, 0, 0, 0, 0)), (17, (0, 0, 0, 0, 0, 0, 1))), ((830, (0, 0, 0, 0, 0, 0, 0)), (17, (0, 0, 0, 0, 0, 0, 1)))), 'driver': (((4309, (0, 0, 0, 0, 0, 0, 0)), (1013, (0, 0, 0, 0, 0, 0, 1)), (545, (0, 0, 0, 0, 0, 0, 2)), (165, (0, 0, 0, 0, 0, 1, 1)), (165, (0, 0, 0, 0, 0, 1, 2)), (1082, (0, 0, 0, 0, 1, 0, 1)), (537, (0, 0, 0, 0, 1, 0, 2)), (330, (0, 0, 0, 0, 1, 1, 1)), (165, (0, 0, 0, 0, 1, 1, 2)), (537, (0, 0, 0, 0, 2, 0, 1)), (165, (0, 0, 0, 0, 2, 1, 1)), (65, (1, 0, 0, 0, 0, 0, 0))), ((4309, (0, 0, 0, 0, 0, 0, 0)), (1019, (0, 0, 0, 0, 0, 0, 1)), (550, (0, 0, 0, 0, 0, 0, 2)), (167, (0, 0, 0, 0, 0, 1, 1)), (167, (0, 0, 0, 0, 0, 1, 2)), (1092, (0, 0, 0, 0, 1, 0, 1)), (542, (0, 0, 0, 0, 1, 0, 2)), (334, (0, 0, 0, 0, 1, 1, 1)), (167, (0, 0, 0, 0, 1, 1, 2)), (542, (0, 0, 0, 0, 2, 0, 1)), (167, (0, 0, 0, 0, 2, 1, 1)), (65, (1, 0, 0, 0, 0, 0, 0)))), 'initial': (((1727, (0, 0, 0, 0, 0, 0, 0)), (65, (1, 0, 0, 0, 0, 0, 0))), ((1727, (0, 0, 0, 0, 0, 0, 0)), (65, (1, 0, 0, 0, 0, 0, 0)))), 'packed_result': (((47, (0, 0, 0, 0, 0, 0, 0)),), ((47, (0, 0, 0, 0, 0, 0, 0)),)), 'prepared_commit': (((5, (0, 0, 0, 0, 0, 0, 0)),), ((5, (0, 0, 0, 0, 0, 0, 0)),)), 'publish_commit': (((60, (0, 0, 0, 0, 0, 0, 0)),), ((60, (0, 0, 0, 0, 0, 0, 0)),)), 'quiet': (((111, (0, 0, 0, 0, 0, 0, 0)),), ((111, (0, 0, 0, 0, 0, 0, 0)),)), 'refresh_hydraulics': (((13, (0, 0, 0, 0, 0, 0, 0)),), ((14, (0, 0, 0, 0, 0, 0, 0)),)), 'terminal': (((1342, (0, 0, 0, 0, 0, 0, 0)),), ((1342, (0, 0, 0, 0, 0, 0, 0)),))}
_MAX_EXTERNAL_ARRAYS = 256
_MAX_ROOT_RANK = 4
# Source-derived logical counter/frame/header/error banks, NOT object heap sizes.
# Complete fixed seven-counter census is recorded in the owning Source ledger.
_REGIME_COUNTER_LOGICAL_HOST_BYTES = 135003
_REGIME_COUNTER_LOGICAL_WORK = 28932


def _refuse(code):
    raise BedIntegrationError('windowed shared water: ' + code)


def is_shared_pair(state, context):
    new = isinstance(state, WindowedSharedBedState) or isinstance(context, WindowedAvalancheContext)
    if new and (type(state) is not WindowedSharedBedState or type(context) is not WindowedAvalancheContext):
        _refuse('exact_state_context_pair')
    return new


def refuse_checkpoint(*, item=None, context=None, value=None):
    """No seventh-field wire exists.  No payload or filesystem is examined."""
    if isinstance(context, WindowedAvalancheContext) or isinstance(value, WindowedSharedBedState):
        _refuse('checkpoint_codec_unqualified')
    if item is not None:
        from maple_syrup.complete_event import DryHandoff, EventProgress

        if isinstance(item, (EventProgress, DryHandoff)):
            progress = getattr(item, 'progress', None) if isinstance(item, DryHandoff) else item
            result = getattr(progress, 'result', None)
            state = getattr(result, 'state', None)
            if isinstance(getattr(state, 'bed', None), WindowedSharedBedState):
                _refuse('checkpoint_codec_unqualified')
            if isinstance(item, DryHandoff) and isinstance(
                    getattr(getattr(item, 'dry_state', None), 'bed', None), WindowedSharedBedState):
                _refuse('checkpoint_codec_unqualified')


def _native():
    from maple.surface.active_layer import bed_exchange
    from maple.surface.topographic_commit import closed_windowed_commit as closed
    from maple.surface.topographic_commit import commit, windowed_commit
    from maple.surface.voxels import deep_store, windowed_inventory, windowed_pending

    return closed, bed_exchange, commit, windowed_commit, deep_store, windowed_inventory, windowed_pending


def _time(value):
    import math

    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        _refuse('current_time_plain_finite_float')
    return value


def _holder(value, cls, names=None):
    expected = tuple(field.name for field in fields(cls))
    if (type(value) is not cls or (names is not None and expected != names)
            or len(expected) > 128 or set(vars(value)) != set(expected)):
        _refuse('holder_schema')


def _profile(state, context):
    if _positive_context(context):
        return _positive_profile(state, context)
    if not is_shared_pair(state, context):
        _refuse('shared_family_required')
    _holder(state, WindowedSharedBedState, _BED_FIELDS)
    _holder(context, WindowedAvalancheContext)
    closed, bed, *_rest = _native()
    closed._k_holder(context.geometry, closed._KGeometrySpec)
    closed._k_holder(context.avalanching_spec, closed._KAvalanchingSpec)
    closed._k_holder(state.voxel_column, closed._KWindowedVoxelColumnState)
    if context.geometry.dx_m != context.geometry.dy_m:
        _refuse('square_cpu_geometry')
    limits = context.bridge_limits
    _holder(limits, WindowedWaterBridgeLimits)
    WindowedWaterBridgeLimits.__post_init__(limits)
    if (context.depth_update_rule != 'constant_depth' or context.record_exchange_history is not False
            or context.deep_chunk_encoding != 'full' or context.deep_overlay_max_bytes is not None
            or type(context.window_max_journal_records) is not int
            or not 0 < context.window_max_journal_records <= 4096
            or context.avalanching_spec.enabled is not True):
        _refuse('constant_depth_enabled_avalanche_full_step_profile')
    if (type(context.source_identity) is not str or len(context.source_identity) != 64
            or any(ch not in '0123456789abcdef' for ch in context.source_identity)):
        _refuse('source_identity_metadata')
    from maple.core.parameters.grain_classes import GrainClass, GrainClassSet

    _holder(context.grain_classes, GrainClassSet, ('classes',))
    classes = context.grain_classes.classes
    if type(classes) is not tuple or len(classes) != 1:
        _refuse('K1_profile')
    _holder(classes[0], GrainClass)
    ids = (classes[0].class_id,)
    closed._q_check_limits(context.native_qualification_limits)
    closed._q_check_image()
    closed._w_check_image()
    closed._p_check_static_contracts()
    provider = closed._p_provider()  # Genuine held state, never a probe/qualifier.
    prefix = closed._p_modules['maple.core.backend.prefix_cell_reductions']
    identity = closed._p_modules['maple.core.backend.depth_class_reductions']
    kernels = identity._kernel_objects(1)
    if (type(prefix._NUMBA) is not dict or len(kernels) != 6
            or any(prefix._NUMBA.get(name) is not kernels[i + 3]
                   for i, name in enumerate(('encode_cells', 'advance_cells', 'decode_cells')))):
        _refuse('held_E1_dispatchers_required')
    order = closed._w_order_module
    import numpy as np

    layout = (state.voxel_column.n_voxels, 1)
    if type(order._VERIFIED) is not set:
        _refuse('held_numpy_order_required')
    for shape in tuple((n,) for n in sorted(set(order.PROBE_EXTENTS) | {layout[0]})) + (layout,):
        if (np.__version__, shape) not in order._VERIFIED:
            _refuse('held_numpy_order_required')
    if context.native_iteration_limits.max_journal_records != context.window_max_journal_records:
        _refuse('single_journal_authority')
    return closed, bed, ids, provider


def _native_args(state, context, ids):
    return (state.voxel_column, state.active_layer, state.ledger, state.committed_topography,
            context.geometry, ids, context.avalanching_spec,
            *(getattr(context, name) for name in _NATIVE_POLICIES[:5]))


def _native_kwargs(state, context, current_time_s):
    return dict(mass_resolution_kg=context.mass_resolution_kg, current_time_s=current_time_s,
                capture_history=False, reconciliation_baseline=state.reconciliation_baseline,
                availability=state.sediment_availability,
                initial_available_fraction=context.initial_available_fraction)


def _metadata(state, context):
    if _positive_context(context):
        return _positive_metadata(state, context)
    closed, bed, ids, provider = _profile(state, context)
    original = closed._preflight_original_metadata(*_native_args(state, context, ids),
        reconciliation_baseline=state.reconciliation_baseline,
        availability=state.sediment_availability, initial_available_fraction=context.initial_available_fraction)
    for channel in ('water_erosion_deposition', 'avalanching'):
        if closed._k_ledger.process_index(channel) not in state.ledger.present_processes:
            _refuse('both_native_process_channels_required')
    if state.water is None or state.sediment_availability is None or context.initial_available_fraction is None:
        _refuse('tracked_water_and_availability_required')
    # No water value validator or pointer/interface capture is run yet.
    closed._w_holder(state.water, closed._WWaterState, closed._W_WATER_FIELDS)
    water_arrays = tuple(getattr(state.water, name) for name in closed._W_WATER_FIELDS)
    prepass = closed._prepass_rank_chains(water_arrays)
    closed._k_dtype_anchors(prepass)
    closed._w_native_rows(closed._w_graph(state.water)[1],
                         (context.geometry.ny, context.geometry.nx, 1))
    plan = bed._plan_windowed_bed_exchange(state.voxel_column, state.active_layer, context.geometry,
        class_ids=ids, limits=context.native_iteration_limits)
    # The unchanged Native water public entry can gather every cell in its
    # default batch.  B must cover that batch and its two exchange operations;
    # the existing BedExchangeLimits hard maximum G=4 is NOT widened.
    if (context.native_iteration_limits.max_gather_columns < context.geometry.ny * context.geometry.nx
            or context.native_iteration_limits.max_operations < 2):
        _refuse('native_water_batch_not_covered')
    return closed, ids, provider, original, plan, (prepass,)


def _array_rows(closed, arrays, limit):
    """Finite primitive metadata batches; no payload conversion or alias credit."""
    import numpy as np

    if type(arrays) is not tuple or len(arrays) > _MAX_EXTERNAL_ARRAYS:
        _refuse('external_array_count')
    result, roots = [], 0
    for start in range(0, len(arrays), 34):
        batch = arrays[start:start + 34]
        if not batch:
            continue
        prepass = closed._prepass_rank_chains(batch)
        closed._k_dtype_anchors(prepass)  # BEFORE every possible interface capture.
        for chain in prepass.chains:
            if any(type(a) is not np.ndarray or a.ndim > _MAX_ROOT_RANK for a in chain):
                _refuse('external_primitive_rank')
            roots += chain[-1].nbytes
            if roots > limit:
                _refuse('external_backing_occurrences')
        result.append(prepass)
    return tuple(result), roots


def _append(arrays, value, *, shape=None, ndim=None, max_size=None):
    if value is not None:
        import numpy as np

        if (type(value) is not np.ndarray or (shape is not None and value.shape != shape)
                or (ndim is not None and value.ndim != ndim)
                or (max_size is not None and value.size > max_size)):
            _refuse('external_array_metadata_shape')
        if len(arrays) >= _MAX_EXTERNAL_ARRAYS:
            _refuse('external_array_count')
        arrays.append(value)


def _append_regime_counter(arrays, value):
    """Retain the Native CPU count representation; never create an array."""
    import numpy as np

    if type(value) is np.int64:
        if value < 0:
            _refuse('regime_counter_scalar_nonnegative')
        return
    if type(value) is not np.ndarray:
        _refuse('regime_counter_representation')
    _append(arrays, value, shape=())


def _external_arrays(*, graph, terrain, routing_source_m=None, soil_water_m=None,
                     state=None, column=None, sediment=None, field=None, schedule=None,
                     vegetation=None, continuation=None, planned_boundaries=None, max_rows=None,
                     rain_rate=None, zero_demand=None):
    from maple_syrup.routing import RoutingGraph
    from maple_syrup.sediment_bed import TerrainReference

    _holder(graph, RoutingGraph)
    _holder(terrain, TerrainReference)
    if (type(graph.shape) is not tuple or len(graph.shape) != 2
            or any(type(n) is not int or not 0 < n <= 64 for n in graph.shape)
            or graph.shape[0] * graph.shape[1] > 64):
        _refuse('external_geometry_metadata')
    ny, nx = graph.shape
    c, cell_shape = ny * nx, (ny, nx)
    if (type(graph.level_bounds) is not tuple or not 1 <= len(graph.level_bounds) <= c + 1
            or any(type(n) is not int or not 0 <= n <= c for n in graph.level_bounds)
            or type(graph.policy) is not str or len(graph.policy) > 128
            or len(graph.policy.split(';')) > 3 or type(graph.input_sha256) is not str
            or len(graph.input_sha256) != 64):
        _refuse('graph_metadata_domain')
    if max_rows is None or type(max_rows) is not int or max_rows < 1:
        _refuse('report_metadata_domain')
    arrays = []
    for name in _GRAPH_ARRAYS:
        shape = ((4, c) if name in ('donor_position', 'donor_mask') else
                 (c,) if name in ('level_order_host', 'conveyance', 'active_flat', 'outlet_flat',
                                 'level_order', 'conveyance_lo') else cell_shape)
        _append(arrays, getattr(graph, name), shape=shape)
    for name in _TERRAIN_ARRAYS:
        shape = (ny + 2, nx + 2) if name in ('initial_full_elevation_m', 'export_receiver_full') else cell_shape
        _append(arrays, getattr(terrain, name), shape=shape)
    for value in (routing_source_m, soil_water_m, vegetation, rain_rate):
        _append(arrays, value, shape=cell_shape)
    _append(arrays, planned_boundaries, ndim=1, max_size=max_rows)
    _append(arrays, zero_demand, shape=(*cell_shape, 1))
    if state is not None:
        from maple_syrup.sediment_event import SedimentEventState
        from maple_syrup.sediment_physics import PhysicsGrid
        from maple_syrup.sediment_transport import TransportNetwork
        from maple_syrup.storm import StormState

        _holder(state, SedimentEventState)
        _holder(state.storm, StormState)
        _holder(state.network, TransportNetwork)
        _holder(state.grid, PhysicsGrid)
        for name in ('depth_m', 'soil_water_m', 'discharge_m2_s'):
            _append(arrays, getattr(state.storm, name), shape=cell_shape)
        _append(arrays, state.sediment_velocity_m_s, shape=(*cell_shape, 1))
        _append(arrays, state.routing_source_elevation_m, shape=cell_shape)
        for name in _NETWORK_ARRAYS:
            _append(arrays, getattr(state.network, name), max_size=c)
        for name in ('slope', 'active', 'terminal'):
            _append(arrays, getattr(state.grid, name), shape=cell_shape)
        if state.phase is not None:
            from maple_syrup.characteristic_transport import PhaseState

            if type(state.phase) is PhaseState:
                names = ('fraction', 'position_m')
            else:
                _refuse('phase_holder')
            _holder(state.phase, type(state.phase))
            from maple_syrup.characteristic_transport import MAX_N_BINS

            if type(state.phase.n_bins) is not int or not 0 < state.phase.n_bins <= MAX_N_BINS:
                _refuse('phase_bins_metadata')
            for name in names:
                value = getattr(state.phase, name)
                _append(arrays, value, shape=(*cell_shape, 1, state.phase.n_bins))
    if column is not None:
        from maple_syrup.infiltration import ColumnParameters

        _holder(column, ColumnParameters)
        for name in _COLUMN_ARRAYS:
            _append(arrays, getattr(column, name), shape=cell_shape)
    if sediment is not None:
        from maple_syrup.sediment_physics import SedimentPhysicsParameters

        _holder(sediment, SedimentPhysicsParameters)
        for name in _SEDIMENT_ARRAYS:
            _append(arrays, getattr(sediment, name), shape=(1, 1, 1) if name.startswith('d_') else (1,))
    if field is not None:
        from maple_syrup.rainfall import RainfallField

        _holder(field, RainfallField)
        _append(arrays, field.multiplier, shape=cell_shape)
    if schedule is not None:
        from maple_syrup.rainfall import RainfallSchedule

        _holder(schedule, RainfallSchedule)
        from maple_syrup.rainfall import RainfallProvenance

        _holder(schedule.provenance, RainfallProvenance)
        text_size = 0
        for name in ('kind', 'convention', 'path', 'sha256', 'start_clock'):
            value = getattr(schedule.provenance, name)
            if value is not None:
                if type(value) is not str or len(value) > 4096:
                    _refuse('rainfall_metadata_text')
                text_size += len(value)
        if text_size > 5 * 4096:
            _refuse('rainfall_metadata_text')
        for name in ('size_bytes', 'rollover_record', 'n_records', 'blank_lines_skipped'):
            value = getattr(schedule.provenance, name)
            if value is not None and (type(value) is not int or not 0 <= value <= (1 << 63) - 1):
                _refuse('rainfall_metadata_integer')
        value = schedule.provenance.start_clock_s
        if value is not None and (type(value) is not float or not 0.0 <= value <= 86400.0):
            _refuse('rainfall_metadata_clock')
        for name in ('edges_s', 'rate_m_per_s', 'intensity_mm_per_h'):
            _append(arrays, getattr(schedule, name), ndim=1, max_size=max_rows + 1)
    if continuation is not None:
        from maple_syrup.sediment_event import SedimentEventResult
        import numpy as np

        _holder(continuation, SedimentEventResult)
        # All direct retained result arrays and its finite class-array map.
        for member in fields(SedimentEventResult):
            value = getattr(continuation, member.name)
            if type(value) is np.ndarray:
                _append(arrays, value)
        if type(continuation.by_class) is not dict or len(continuation.by_class) != 16:
            _refuse('class_result_map')
        for value in continuation.by_class.values():
            _append(arrays, value, shape=(1,))
        if type(continuation.regime_cell_steps) is not dict or len(continuation.regime_cell_steps) != 7:
            _refuse('regime_result_map')
        for value in continuation.regime_cell_steps.values():
            _append_regime_counter(arrays, value)
        if continuation.exchange_history is not None or continuation.deep_publication is not None:
            _refuse('continuation_history_or_deferred_publication')
        # Its prior event state is separately retained, without alias credit.
        prior = _external_arrays(graph=continuation.state.graph, terrain=continuation.state.terrain,
                                 state=continuation.state, max_rows=max_rows)
        for value in prior:
            _append(arrays, value)
    return tuple(arrays)


def _run_root(context):
    import stat
    from pathlib import Path

    root = context.bridge_limits.run_dir
    if (type(root) is not type(Path()) or not root.is_absolute()
            or type(root._raw_paths) is not list or len(root._raw_paths) > 128
            or any(type(p) is not str or len(p) > 4096 for p in root._raw_paths)
            or sum(len(p) + 1 for p in root._raw_paths) > 4096
            or '..' in root.parts):
        _refuse('owned_run_path')
    # Fresh every call, quiescent owned-path protocol, no hostile ABA claim.
    for path in (root, *root.parents):
        if not stat.S_ISDIR(path.lstat().st_mode):
            _refuse('owned_run_ancestor')
    return root


def _disk_inventory(context):
    import os
    import stat
    from pathlib import Path

    root = _run_root(context)
    total, entries, stack = 0, 0, [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        st = directory.lstat()
        if not stat.S_ISDIR(st.st_mode) or depth > 16:
            _refuse('owned_run_directory')
        with os.scandir(directory) as scan:
            for entry in scan:
                if entries >= context.bridge_limits.max_entries:
                    _refuse('owned_run_entries')
                entries += 1
                st = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(st.st_mode):
                    total += st.st_size
                    if total > context.bridge_limits.max_run_bytes:
                        _refuse('owned_run_bytes')
                elif stat.S_ISDIR(st.st_mode):
                    if len(stack) >= context.bridge_limits.max_entries:
                        _refuse('owned_run_entries')
                    stack.append((Path(entry.path), depth + 1))
                else:
                    _refuse('owned_run_file_type')
    return total, entries


def _visible_requests(cells, bisections, substeps, steps, retries, reports):
    # Source transport loop doubles a positive integer only after a rejection;
    # before EVERY operator call it enforces n_sub<=S.  Hence <=bit_length(S)
    # operator attempts, each with at most S actual substeps.
    raw = (cells, bisections, substeps, steps, retries, reports)
    if any(type(n) is not int or not 0 <= n <= (1 << 63) - 1 for n in raw):
        _refuse('visible_work_metadata')
    variables = (cells, bisections, substeps, substeps.bit_length(), steps, retries, reports)
    values = {}
    for name, both in _VISIBLE_REQUEST_CENSUS.items():
        if type(both) is not tuple or len(both) != 2:
            _refuse('visible_census_source_image')
        totals = []
        for polynomial in both:
            total = 0
            for coefficient, powers in polynomial:
                term = coefficient
                for value, power in zip(variables, powers):
                    term *= value ** power
                total += term
            totals.append(total)
        values[name] = tuple(totals)
    # Driver skeleton already overcharges nested scalar loop bodies; actual
    # transport/commit bodies are priced separately at their owning counts.
    # A successful step: <=R hydraulic attempts, one commit; one final forced
    # commit/terminal.  Each pack/callback remains positive even if unused.
    counts = {'initial': 1, 'attempt': steps * retries,
              'commit_routing': steps + 1, 'driver': 1,
              'complete_entry': 1, 'terminal': 1, 'quiet': steps + 1,
              'packed_result': reports + 2, 'prepared_commit': 2 * (steps + 1),
              'publish_commit': steps + 1, 'refresh_hydraulics': steps + 1}
    return tuple(sum(values[name][i] * count for name, count in counts.items()) for i in (0, 1))


def _event_time_domain(state, context, control, end_s, report_every_s):
    # Scalar eligibility only.  The original planner/loop still receives all
    # exact clocks and creates the actual forcing/report partitions later.
    import math

    if end_s is None:
        return
    for value in (end_s, report_every_s):
        if type(value) is not float or not math.isfinite(value):
            _refuse('event_clock_metadata')
    if end_s <= state.t_s or report_every_s <= 0.0:
        _refuse('event_clock_metadata')
    ratio = (end_s - state.t_s) / control.storm.max_dt_s
    if not math.isfinite(ratio) or math.ceil(ratio) > context.bridge_limits.max_event_steps:
        _refuse('event_scalar_planning_ceiling')
    # Resource admission enforces A well below 2**53 before this planning can
    # execute.  For these finite quotients the original ceil/while correction
    # needs at most one increment; no clock/timestep is modified here.


def _reserve(state, context, metadata, external_arrays=(), *, control=None, max_report_rows=None):
    if _positive_context(context):
        return _positive_reserve(state, context, metadata, external_arrays, control, max_report_rows)
    closed, ids, provider, original, storage, water_prepasses = metadata
    limits, policy = context.bridge_limits, context.native_publication_limits
    ext_prepasses, external = _array_rows(closed, external_arrays, limits.max_external_backing_bytes)
    native_roots = sum(chain[-1].nbytes for chain in original.prepass.chains)
    water_roots = sum(chain[-1].nbytes for p in water_prepasses for chain in p.chains)
    h = state.voxel_column.deep_store.header
    root = _run_root(context)
    store_dir = state.voxel_column.deep_store.store_dir
    if not store_dir.is_relative_to(root):
        _refuse('store_outside_owned_run')
    import stat

    for path in (store_dir, *store_dir.parents):
        if not stat.S_ISDIR(path.lstat().st_mode):
            _refuse('owned_store_ancestor')
    steps = 1 if control is None else control.storm.max_steps
    retries = 1 if control is None else control.storm.max_retries + 1
    substeps = 1 if control is None else control.max_transport_substeps
    reports = limits.max_report_rows if max_report_rows is None else max_report_rows
    if (type(steps) is not int or not 0 < steps <= limits.max_event_steps
            or type(retries) is not int or not 0 < retries <= limits.max_retries + 1
            or type(substeps) is not int or not 0 < substeps <= limits.max_transport_substeps
            or type(reports) is not int or not 0 < reports <= limits.max_report_rows):
        _refuse('event_resource_declaration')
    bins = 1 if control is None or not control.characteristic else control.phase_bins
    # Largest logical output: C*K*V, C*K*bin, D4 donor/faces, full terrain
    # ring, seven-process ledger, report tables (16 water +9+5K sediment).
    # K=1 is already authenticated, and all input ROOT occurrences are separate.
    cells = h.ny * h.nx
    largest = max(cells * max(7, h.n_voxels, bins + 1), (h.ny + 2) * (h.nx + 2),
                  (reports + 1) * (16 + 9 + 5)) * 8
    if any(array.nbytes > largest for array in external_arrays):
        _refuse('external_logical_extent')
    bisections = 0 if control is None else control.storm.bisection_iterations
    payload_sites, work_sites = _visible_requests(cells, bisections, substeps,
                                                steps, retries, reports)
    # Every counted request gets its full logical output AND a rank<=4
    # 17+2r interface/header projection.  No alias, allocator reuse, mutually
    # exclusive branch or last-use credit is taken.  Provider-private scratch
    # and actual Python heap remain the positive reserve/Root prerequisites.
    visible = payload_sites * (largest + 8 * (17 + 2 * _MAX_ROOT_RANK))
    visible_work = work_sites * largest
    attempts = steps * retries
    commits = steps + 1 if control is not None else 1
    # Four entry checks, <=one core+four water/demand checks per attempt,
    # two step-publication checks/accepted step, four checks/commit, eight
    # terminal checks+two direct terminal water/demand checks, three prior
    # metadata issuances.  Every call retains its own positive work bank.
    metadata_calls = 17 + 5 * attempts + 2 * steps + 4 * commits
    native_reference = metadata_calls * policy.max_proof_bytes
    # Two COW attempts, two passes each; all buffers/old roots/error results are
    # additive.  Storage's whole native transaction scratch also overcharges
    # publisher/validation arrays rather than crediting mutually exclusive life.
    chunk = max(ref.nbytes for ref in state.voxel_column.deep_store.chunks)
    payload = (('native_future_wet_policy', policy.max_retained_payload_bytes),
               ('original_root_occurrences', native_roots + water_roots),
               ('external_current_and_prior_root_occurrences', 2 * external),
               ('external_unobserved_caller_ceiling', limits.max_external_backing_bytes),
               ('storage_caller_positive', storage.caller_live_host_bytes),
               ('two_storage_and_validator_envelopes', 2 * storage.new_host_bytes_peak),
               ('four_publication_chunk_buffers', 4 * chunk),
               ('metadata_original_and_water_and_adapter', native_reference + 8 * (256 + 8 * len(external_arrays))),
               ('owned_directory_metadata_and_error_paths', limits.max_entries * (4 * 4096 + 256)),
               ('fixed_rainfall_text_and_scalar_metadata', 5 * 4096 * 4 + 128 * 32),
               ('visible_syrup_requests_including_terminal_error', visible),
               ('seven_continuation_regime_scalar_frame_header_error', _REGIME_COUNTER_LOGICAL_HOST_BYTES))
    mapped = policy.max_mapped_bytes + 2 * storage.mapped_bytes_max + 2 * chunk
    reads = (commits * policy.max_logical_deep_read_bytes
             + (2 * attempts + 2 * commits) * storage.logical_deep_read_bytes
             + (metadata_calls + 8) * h.logical_bytes)
    work = (('future_native_wet_policies', commits * policy.max_work),
            ('additional_metadata_and_native_value_preflights', metadata_calls * policy.max_work),
            ('visible_syrup_loop_expanded_byte_visits', visible_work),
            ('original_and_external_metadata', original.request.logical_work + 256 * len(external_arrays)),
            ('two_owned_directory_scans', 2 * limits.max_entries * (4 * 4096 + 256)),
            ('publication_and_validation_byte_visits', 8 * chunk + 2 * storage.new_host_bytes_peak),
            ('seven_continuation_regime_scalar_metadata_work', _REGIME_COUNTER_LOGICAL_WORK))
    if (sum(value for _, value in payload) + limits.reserve_host_bytes > limits.max_host_bytes
            or mapped > limits.max_mapped_bytes or reads > limits.max_logical_read_bytes
            or sum(value for _, value in work) > limits.max_work):
        _refuse('bridge_logical_resource_ceiling')
    available = closed._p_modules['maple.aeolian.cohorts.capacity'].default_available_memory_bytes()
    if type(available) is not int or available < sum(v for _, v in payload) + mapped + limits.reserve_host_bytes:
        _refuse('genuine_host_headroom')
    # Metadata producer ONLY.  It prices both next generation widths; no chunk
    # is read and no post-write size is used to justify the first write.
    deep = _native()[4]
    publications = steps + 2 * commits + 1 if control is not None else 2
    version = state.voxel_column.deep_store
    if publications > limits.max_new_versions or version.version_id + publications > (1 << 63) - 1:
        _refuse('version_count_ceiling')
    manifest = len(deep._canonical_json_bytes(deep._v1_manifest_object(h, chunks=version.chunks,
        version_id=version.version_id + publications, parent_version_id=version.version_id + publications - 1,
        parent_manifest_digest='0' * 64, journal_digest='0' * 64)))
    increment = h.ny * h.nx * h.n_voxels * h.n_classes * 8 + max(256 + 192 * h.n_chunks, manifest)
    current_run, entries = _disk_inventory(context)
    current_store, _chunks, _versions = deep._scan_store_bytes(version.store_dir)
    # Each future publication and a complete partial attempt coexist; no GC,
    # alias, no-pending, immutable-chunk reuse or hardlink credit is taken.
    future = (publications + 1) * increment
    if current_store + future > limits.max_store_bytes or current_run + future > limits.max_run_bytes:
        _refuse('store_or_run_reservation')
    admission = _Admission(original, storage, provider, water_prepasses + ext_prepasses, payload,
                           mapped, reads, work, (('current_run', current_run), ('current_store', current_store),
                                                ('future_and_partial', future), ('entries', entries)))
    closed._p_check_provider(provider)
    return admission


def _value_validate(state, context, current_time_s, *, published):
    closed, _bed, commit, _wc, _deep, inventory, pending = _native()
    from maple.core.types.topographic_commit import validate_committed_topography_state
    from maple.coupling.sediment_ledger.validation import validate_ledger_state
    from maple.surface.active_layer.validation import check_active_layer_state, _check_partition_totals
    from maple.surface.active_layer.capacity import active_layer_total_mass_kg, compute_active_layer_target_mass_kg
    from maple.surface.availability.validation import check_sediment_availability_state
    from maple.water.validation import validate_water_state

    from maple.core.parameters.grain_classes import validate_grain_classes

    validate_grain_classes(context.grain_classes)
    ids = tuple(g.class_id for g in context.grain_classes.classes)
    pending.validate_pending_binding(state.voxel_column, operation='SYRUP shared water')
    inventory.check_windowed_column_cheap(state.voxel_column, context.geometry, class_ids=ids)
    if published:
        inventory.validate_windowed_column_full(state.voxel_column, context.geometry,
            context.mass_resolution_kg, class_ids=ids)
    check_active_layer_state(state.active_layer, context.geometry, context.mass_resolution_kg)
    active_total = active_layer_total_mass_kg(state.active_layer)
    subactive_total = inventory.column_mass_by_class_kg(state.voxel_column).sum(axis=-1)
    _check_partition_totals(active_total, subactive_total,
        compute_active_layer_target_mass_kg(context.geometry), context.geometry, closed.np)
    validate_ledger_state(state.ledger)
    validate_committed_topography_state(state.committed_topography)
    validate_water_state(state.water, context.geometry.ny, context.geometry.nx, 1)
    check_sediment_availability_state(state.sediment_availability, state.active_layer, context.mass_resolution_kg)
    commit.validate_reconciliation_context(state.ledger, state.voxel_column, state.active_layer,
        context.geometry, committed_state=state.committed_topography,
        reconciliation_baseline=state.reconciliation_baseline, current_time_s=current_time_s,
        operation='SYRUP shared water')
    reference, label = commit.resolve_reconciliation_reference_inventory_kg(
        state.committed_topography, state.reconciliation_baseline)
    commit._reconcile_ledger_with_physical_state(state.ledger, reference, state.voxel_column,
        state.active_layer, context.mass_resolution_kg, reference_label=label)
    closed._q_check_image()
    closed._w_check_image()


def validate_shared_bed(state, context, *, current_time_s, published):
    _time(current_time_s)
    if type(published) is not bool:
        _refuse('published_selector')
    metadata = _metadata(state, context)
    admission = _reserve(state, context, metadata)
    _value_validate(state, context, current_time_s, published=published)
    return admission



def admit_water_operands(state, context, *, depth_m=None, mobile_kg=None, demand=None):
    """Direct public helper admission; the original Native body still runs once.

    Only primitive authentic metadata is examined before the complete reserve.
    Optional diagnostic objects are outside this narrow array-only profile.
    """
    import numpy as np

    shape = (context.geometry.ny, context.geometry.nx, 1)
    arrays = []
    if demand is None:
        supplied = (depth_m, state.water.mobile_mass_by_cell_class_kg if mobile_kg is None else mobile_kg)
        expected = (shape[:2], shape)
        for array, extent in zip(supplied, expected):
            if type(array) is not np.ndarray or array.shape != extent or array.dtype != np.float64:
                _refuse('water_operand_metadata')
            arrays.append(array)
    else:
        from maple.water.interfaces import WaterProcessDemand
        from maple.aeolian.flux.face_flux import FaceFluxResult

        _holder(demand, WaterProcessDemand)
        if (set(vars(demand)) != {'requested_removal_by_cell_class_kg',
                'requested_deposition_by_cell_class_kg', 'hydraulic_diagnostics', 'face_flux',
                'boundary_input_by_cell_class_kg', 'boundary_export_by_cell_class_kg'}
                or demand.hydraulic_diagnostics is not None):
            _refuse('array_only_native_demand_schema')
        for name in ('requested_removal_by_cell_class_kg', 'requested_deposition_by_cell_class_kg',
                     'boundary_input_by_cell_class_kg', 'boundary_export_by_cell_class_kg'):
            array = getattr(demand, name)
            if array is None and name.startswith('boundary_'):
                continue
            if type(array) is not np.ndarray or array.shape != shape or array.dtype != np.float64:
                _refuse('demand_operand_metadata')
            arrays.append(array)
        if demand.face_flux is not None:
            flux = demand.face_flux
            _holder(flux, FaceFluxResult)
            if set(vars(flux)) != {'x_face_crossing_mass_kg', 'y_face_crossing_mass_kg',
                    'x_face_net_crossing_mass_kg', 'y_face_net_crossing_mass_kg'}:
                _refuse('native_face_metadata')
            for name in ('x_face_crossing_mass_kg', 'y_face_crossing_mass_kg',
                         'x_face_net_crossing_mass_kg', 'y_face_net_crossing_mass_kg'):
                array = getattr(flux, name)
                extent = (shape[0], shape[1] + 1, 1) if name.startswith('x_') else (shape[0] + 1, shape[1], 1)
                if type(array) is not np.ndarray or array.shape != extent or array.dtype != np.float64:
                    _refuse('native_face_metadata')
                arrays.append(array)
    metadata = _metadata(state, context)
    admission = _reserve(state, context, metadata, tuple(arrays))
    _value_validate(state, context, storage_reference_time(state),
                    published=state.voxel_column.pending_deep is None)
    return admission


def admit_initial(bed, context, graph, terrain, soil_water_m, current_time_s, control):
    _time(current_time_s)
    _control(context, control)
    arrays = _external_arrays(graph=graph, terrain=terrain, soil_water_m=soil_water_m,
                              max_rows=context.bridge_limits.max_report_rows)
    _geometry(context, graph, terrain)
    admission = _reserve(bed, context, _metadata(bed, context), arrays, control=control)
    _value_validate(bed, context, current_time_s, published=bed.voxel_column.pending_deep is None)
    return admission


def _control(context, control):
    from maple_syrup.sediment_event import SedimentEventControl

    from maple_syrup.storm import StormControl

    _holder(control, SedimentEventControl)
    _holder(control.storm, StormControl)
    control.validated()
    # Existing routing.py hard domain, enforced early only for this family.
    if (type(control.storm.bisection_iterations) is not int
            or not 1 <= control.storm.bisection_iterations <= 200):
        _refuse('native_bisection_domain')
    if (control.storm.implementation != 'array' or control.storm.root_solver != 'bisection'
            or control.phase_storage != 'dense' or control.resolved_transport_implementation() != 'array'):
        _refuse('array_bisection_dense_transport_profile')
    if (control.cuda is not False or control.deep_publication != 'step' or control.record_exchange_history is not False
            or context.record_exchange_history is not False or control.commit is not True):
        _refuse('step_publication_cpu_no_history')


def _geometry(context, graph, terrain):
    import numpy as np

    g = context.geometry
    if (graph.xp is not np or graph.shape != (g.ny, g.nx) or g.dx_m != g.dy_m
            or graph.dx_m != g.dx_m or terrain.dx_m != graph.dx_m
            or terrain.initial_committed_elevation_m.shape != graph.shape
            or terrain.allow_masked_nodata):
        _refuse('square_unmasked_cpu_geometry')


def admit_event(state, context, control, **operands):
    _time(state.t_s)
    _control(context, control)
    _event_time_domain(state, context, control, operands.get('end_s'), operands.get('report_every_s'))
    _geometry(context, state.graph, state.terrain)
    arrays = _external_arrays(graph=state.graph, terrain=state.terrain, state=state,
        max_rows=context.bridge_limits.max_report_rows,
        **{k: v for k, v in operands.items() if k not in ('max_report_rows', 'end_s', 'report_every_s')})
    previous = operands.get('continuation')
    if previous is not None:
        prior_metadata = _metadata(previous.state.bed, context)
        prior_native = tuple(leaf.array for leaf in prior_metadata[3].leaves)
        prior_water = tuple(getattr(previous.state.bed.water, name) for name in prior_metadata[0]._W_WATER_FIELDS)
        if len(arrays) + len(prior_native) + len(prior_water) > _MAX_EXTERNAL_ARRAYS:
            _refuse('external_array_count')
        arrays += prior_native + prior_water
    admission = _reserve(state.bed, context, _metadata(state.bed, context), arrays,
                         control=control, max_report_rows=operands.get('max_report_rows'))
    _value_validate(state.bed, context, state.t_s, published=state.bed.voxel_column.pending_deep is None)
    return admission


def commit_shared_bed(state, context, graph, reference, *, current_time_s, force, routing_source_m):
    if _positive_context(context):
        return _positive_commit(state, context, graph, reference, current_time_s, force, routing_source_m)
    from maple_syrup.sediment_bed import check_routing_source, reroute_threshold_m
    from maple.surface.topographic_commit.triggers import evaluate_commit_triggers

    _time(current_time_s)
    if type(force) is not bool or routing_source_m is None:
        _refuse('force_and_held_routing_source')
    _geometry(context, graph, reference)
    arrays = _external_arrays(graph=graph, terrain=reference, routing_source_m=routing_source_m,
                              max_rows=context.bridge_limits.max_report_rows)
    metadata = _metadata(state, context)
    _reserve(state, context, metadata, arrays)
    _value_validate(state, context, current_time_s, published=state.voxel_column.pending_deep is None)
    check_routing_source(routing_source_m, graph.shape, graph.xp)
    threshold = reroute_threshold_m(context)
    trigger = evaluate_commit_triggers(state.ledger, state.committed_topography, context.geometry,
                                      current_time_s, context.topographic_commit_spec)
    if not force and not trigger.should_commit:
        return None, trigger, None, None, threshold
    published = publish_shared_bed(state, context, current_time_s=current_time_s)
    closed, _bed, ids, _provider = _profile(published, context)
    # The actual stage starts with its own fresh genuine request on the
    # returned published column.  No extra caller-held request is retained.
    args, kwargs = _native_args(published, context, ids), _native_kwargs(published, context, current_time_s)
    wet = closed._stage_bounded_cpu_native_windowed_water_topography(*args, water=published.water,
        qualification_limits=context.native_qualification_limits, **kwargs)
    if type(wet) is not closed._ClosedWaterReturn:
        _refuse('actual_native_wet_return')
    result = wet.result
    if (type(result) is not _native()[3].WindowedCommitResult or wet.depth_update.rule != 'constant_depth'
            or wet.water.depth_m is not wet.depth_update.new_depth_m
            or wet.depth_update.displaced_volume_m3 != 0.0 or wet.depth_update.clamp_added_volume_m3 != 0.0
            or wet.depth_update.clamp_cell_count != 0
            or not closed._p_modules['maple.surface.voxels.windowed_evidence']._same(
                wet.water.depth_m, published.water.depth_m)):
        _refuse('genuine_constant_depth_return')
    candidate = replace(published, voxel_column=result.new_voxel_column,
        active_layer=result.new_active_layer, ledger=result.new_ledger,
        committed_topography=result.new_committed_state, sediment_availability=wet.availability,
        water=wet.water, reconciliation_baseline=result.new_reconciliation_baseline)
    validate_shared_bed(candidate, context, current_time_s=current_time_s, published=False)
    return candidate, trigger, result, wet, threshold


def shared_bed_already_committed(state, context, current_time_s):
    if not is_shared_pair(state, context):
        _refuse('shared_family_required')
    if state.voxel_column.pending_deep is not None:
        return False
    validate_shared_bed(state, context, current_time_s=current_time_s, published=True)
    return _native()[3].is_windowed_already_committed(state.ledger, state.committed_topography,
        state.voxel_column, state.active_layer, context.geometry, context.mass_resolution_kg,
        current_time_s, reconciliation_baseline=state.reconciliation_baseline)


def publish_shared_bed(state, context, *, current_time_s):
    if _positive_context(context):
        return _positive_publish(state, context, current_time_s)
    # Storage-only rebinding; the actual float time is never advanced here.
    validate_shared_bed(state, context, current_time_s=current_time_s, published=False)
    column = _native()[6].publish_pending_column(state.voxel_column, context.geometry,
        context.mass_resolution_kg, max_records=context.window_max_journal_records,
        encoding='full', overlay_max_bytes=None)
    candidate = replace(state, voxel_column=column)
    validate_shared_bed(candidate, context, current_time_s=current_time_s, published=True)
    if candidate.reconciliation_baseline is not state.reconciliation_baseline:
        _refuse('publication_baseline_reference')
    return candidate


def storage_reference_time(state):
    """Storage binding's existing Native reference, never a solver-clock claim.

    Old publish_bed_storage has no time argument.  Actual event/terminal entry
    separately checks the real StormState.t_s.  Rebinding storage does not
    update either Native time or either reconciliation reference.
    """
    time = state.committed_topography.last_commit_time_s
    if state.reconciliation_baseline is not None:
        time = max(time, state.reconciliation_baseline.last_reconciliation_time_s)
    return _time(time)


def require_terminal_pair(wet, dry, context, *, current_time_s):
    if not is_shared_pair(wet, context) or not is_shared_pair(dry, context):
        _refuse('terminal_exact_shared_pair')
    for name in _BED_FIELDS:
        if name != 'water' and getattr(wet, name) is not getattr(dry, name):
            _refuse('terminal_physical_reference_pair')
    validate_shared_bed(wet, context, current_time_s=current_time_s, published=False)
    validate_shared_bed(dry, context, current_time_s=current_time_s, published=False)


def splice_terminal_shared_beds(accepted, dry, context, *, current_time_s):
    if not is_shared_pair(accepted, context) or not is_shared_pair(dry, context):
        _refuse('terminal_exact_shared_pair')
    candidate = replace(dry, voxel_column=accepted.voxel_column,
                        reconciliation_baseline=accepted.reconciliation_baseline)
    validate_shared_bed(accepted, context, current_time_s=current_time_s, published=True)
    validate_shared_bed(candidate, context, current_time_s=current_time_s, published=True)
    for name in _BED_FIELDS:
        if name != 'water' and getattr(accepted, name) is not getattr(candidate, name):
            _refuse('terminal_published_reference_pair')
    return candidate


# Explicit opt-in positive-water lane. Original functions above retain their
# complete old bodies behind only exact new-family dispatch.
@dataclass(frozen=True)
class PositiveWindowedWaterBridgeLimits(WindowedWaterBridgeLimits):
    """Explicit C16/V6/W3/K1 admission lane; never an execution certificate."""

    def __post_init__(self):
        WindowedWaterBridgeLimits.__post_init__(self)
        expected = {'max_host_bytes': 4 << 30, 'max_mapped_bytes': 128 << 20,
                    'max_store_bytes': 64 << 20, 'max_run_bytes': 128 << 20,
                    'max_entries': 4096, 'max_external_backing_bytes': 64 << 20,
                    'max_event_steps': 10000, 'max_retries': 10,
                    'max_transport_substeps': 64, 'max_report_rows': 4096,
                    'reserve_host_bytes': 64 << 20}
        for name, value in expected.items():
            if type(getattr(self, name)) is not int or getattr(self, name) != value:
                _positive_refuse('positive_fixed_limit')


@dataclass(frozen=True)
class PositiveWaterAdmissionSummary:
    """Actual counters only. No claim of completion/fit beyond the real return."""
    gate_count: int
    logical_work_debited: int
    logical_read_bytes_debited: int
    versions_reserved: int
    maximum_predicate_bytes: int


class _PositiveOwner:
    __slots__ = ('context', 'control', 'storm', 'projection', 'inner_control',
                 'birth', 'thread', 'ordinal', 'work', 'reads', 'versions',
                 'maximum', 'poisoned', 'closed')

    def __init__(self, context, control, birth, thread, projection):
        self.context, self.control, self.storm = context, control, control.storm
        self.projection, self.inner_control = projection, None
        self.birth, self.thread = birth, thread
        self.ordinal = self.work = self.reads = self.versions = self.maximum = 0
        self.poisoned = self.closed = False


_POSITIVE_OWNER = None
_POSITIVE_CENSUS_NAMES = ('initial', 'attempt', 'commit_routing', 'driver',
    'complete_entry', 'terminal', 'quiet', 'packed_result', 'prepared_commit',
    'publish_commit', 'refresh_hydraulics')
# Logical Source projections: word8, scalar/header32, rank<=4 interface200.
# They are positive work/owner charges, not CPython/provider heap assertions.
_POSITIVE_WORD = 8
_POSITIVE_HEADER = 32
_POSITIVE_INTERFACE = 200
_POSITIVE_GRID = 1024
_POSITIVE_FLAG = 2440
_POSITIVE_REPORT_ROWS = 4096
_POSITIVE_PLANNER_EXTENT = 8194
# Filled from the final helper call-tree/loop audit, never an incoming fee.
_POSITIVE_HELPER_HOST = 284938012
_POSITIVE_HELPER_WORK = 1851912394512


def _positive_refuse(code):
    # Sticky only for this new admission family. A retry which catches an
    # admission refusal can never obtain another action from the same owner.
    owner = _POSITIVE_OWNER
    if type(owner) is _PositiveOwner:
        owner.poisoned = True
    _refuse(code)


def _positive_context(context):
    if isinstance(context, PositiveWindowedWaterBridgeLimits):
        _positive_refuse('positive_exact_context')
    limits = getattr(context, 'bridge_limits', None)
    if isinstance(limits, PositiveWindowedWaterBridgeLimits):
        if (type(limits) is not PositiveWindowedWaterBridgeLimits
                or type(context) is not WindowedAvalancheContext):
            _positive_refuse('positive_exact_context')
        return True
    return False


def _positive_pair_guard(state, context):
    active = _POSITIVE_OWNER
    selected = _positive_context(context)
    if selected or active is not None:
        if (not selected or type(state) is not WindowedSharedBedState
                or type(context) is not WindowedAvalancheContext
                or (active is not None and context is not active.context)):
            _positive_refuse('positive_exact_active_pair')


def _positive_integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= (1 << 63) - 1:
        _positive_refuse('positive_integer_' + name)
    return value


def _positive_fixed_geometry(state, context):
    h = state.voxel_column.deep_store.header
    if ((context.geometry.ny, context.geometry.nx, state.voxel_column.n_voxels,
            h.ny, h.nx, h.n_voxels, h.window_voxels, h.n_classes)
            != (4, 4, 6, 4, 4, 6, 3, 1)
            or state.voxel_column.deep_store.format_version != 1
            or state.voxel_column.deep_store.encoding != 'full'
            or context.window_max_journal_records != 320
            or context.native_iteration_limits.max_gather_columns != 4
            or context.native_iteration_limits.max_operations != 2
            or context.native_iteration_limits.max_journal_records != 320
            or type(h.class_ids) is not tuple or len(h.class_ids) != 1
            or type(h.class_ids[0]) is not str or not 0 < len(h.class_ids[0]) <= 128
            or not h.class_ids[0].isascii()):
        _positive_refuse('positive_C16_V6_W3_K1_M2_G4_J320_full_v1')


def _positive_decimal(raw, *, zero=False):
    if (type(raw) is not str or not 1 <= len(raw) <= 20
            or any(c < '0' or c > '9' for c in raw)
            or (len(raw) > 1 and raw[0] == '0')):
        _positive_refuse('positive_proc_decimal')
    value = int(raw)
    return _positive_integer(value, 'proc_decimal', 0 if zero else 1)


def _positive_parse_stat(raw):
    if type(raw) is not str or not 1 <= len(raw) <= 4096 or not raw.isascii():
        _positive_refuse('positive_proc_stat')
    opening, closing = raw.find(' ('), raw.rfind(') ')
    if opening <= 0 or closing <= opening:
        _positive_refuse('positive_proc_stat')
    pid = _positive_decimal(raw[:opening])
    tail = raw[closing + 2:].split()
    # Tail index0 is stat field3; index19 is field22 (starttime).
    if len(tail) != 50 or len(tail[0]) != 1:
        _positive_refuse('positive_proc_stat_fields')
    return pid, _positive_decimal(tail[19])


def _positive_parse_statm(raw, page_size):
    if type(raw) is not str or not 1 <= len(raw) <= 256 or not raw.isascii():
        _positive_refuse('positive_proc_statm')
    fields_ = raw.split()
    if len(fields_) != 7:
        _positive_refuse('positive_proc_statm_fields')
    values = tuple(_positive_decimal(x, zero=True) for x in fields_)
    if values[0] == 0 or values[1] > values[0]:
        _positive_refuse('positive_proc_mapping')
    if (type(page_size) is not int or page_size <= 0 or page_size > 1 << 20
            or page_size & (page_size - 1)):
        _positive_refuse('positive_proc_page_size')
    return _positive_integer(values[0] * page_size, 'mapped', 1)


def _positive_read_proc(path, capacity):
    import os
    import stat

    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            _positive_refuse('positive_proc_file_type')
        raw = os.read(fd, capacity + 1)
        if len(raw) > capacity or os.read(fd, 1):
            _positive_refuse('positive_proc_file_extent')
        return raw.decode('ascii', errors='strict')
    finally:
        os.close(fd)


def _positive_current_mapped(expected_birth=None):
    import os
    import resource
    import sys

    if sys.byteorder != 'little' or sys.maxsize != (1 << 63) - 1:
        _positive_refuse('positive_proc_ABI')
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if type(soft) is not int or type(hard) is not int or soft != 4 << 30 or hard != 4 << 30:
        _positive_refuse('positive_original_soft_hard_AS')
    pid = os.getpid()
    before = _positive_parse_stat(_positive_read_proc('/proc/self/stat', 4096))
    mapped = _positive_parse_statm(_positive_read_proc('/proc/self/statm', 256),
                                   os.sysconf('SC_PAGE_SIZE'))
    after = _positive_parse_stat(_positive_read_proc('/proc/self/stat', 4096))
    if (before != after or before[0] != pid
            or (expected_birth is not None and before != expected_birth)):
        _positive_refuse('positive_process_birth')
    return before, mapped, min(soft, hard)


def _positive_control_projection(context, control):
    _control(context, control)
    if (control.transport_scheme != 'upwind' or control.phase_max_nnz is not None
            or control.storm.max_steps != 10000 or control.storm.max_retries != 10
            or control.max_transport_substeps != 64 or control.max_commit_log != 256
            or control.max_rejections_recorded != 100):
        _positive_refuse('positive_original_control_domain')
    rows = []
    for field in fields(type(control)):
        value = getattr(control, field.name)
        if field.name == 'storm':
            continue
        if type(value) not in (type(None), bool, int, float, str):
            _positive_refuse('positive_control_scalar_projection')
        rows.append((field.name, value))
    storm_rows = []
    for field in fields(type(control.storm)):
        value = getattr(control.storm, field.name)
        if type(value) not in (bool, int, float, str):
            _positive_refuse('positive_storm_scalar_projection')
        storm_rows.append((field.name, value))
    return tuple(rows), tuple(storm_rows)


def _positive_owner(context, control=None):
    import threading

    owner = _POSITIVE_OWNER
    if (type(owner) is not _PositiveOwner or owner.poisoned or owner.closed
            or owner.context is not context or owner.thread != threading.get_ident()
            or owner.storm is not owner.control.storm):
        _positive_refuse('positive_owned_entry_required')
    if _positive_control_projection(context, owner.control) != owner.projection:
        _positive_refuse('positive_original_control_changed')
    if control is not None:
        projection = _positive_control_projection(context, control)
        if control.storm is not owner.storm:
            _positive_refuse('positive_storm_identity')
        if control is owner.control:
            if projection != owner.projection:
                _positive_refuse('positive_original_control_changed')
        else:
            original_rows, original_storm = owner.projection
            expected = tuple((name, False if name == 'force_final_commit' else value)
                             for name, value in original_rows), original_storm
            if projection != expected:
                _positive_refuse('positive_only_original_derived_control')
            if owner.inner_control is None:
                owner.inner_control = control
            elif owner.inner_control is not control:
                _positive_refuse('positive_inner_control_identity')
    return owner


def _positive_require_action(state, context):
    _positive_pair_guard(state, context)
    owner = _positive_owner(context)
    if owner.ordinal < 1:
        _positive_refuse('positive_no_admitted_gate')
    _positive_current_mapped(owner.birth)
    return owner


def _positive_interval_units():
    if (len(_VISIBLE_REQUEST_CENSUS) != len(_POSITIVE_CENSUS_NAMES)
            or set(_VISIBLE_REQUEST_CENSUS) != set(_POSITIVE_CENSUS_NAMES)):
        _positive_refuse('positive_census_names')
    totals = [0, 0]
    for name in _POSITIVE_CENSUS_NAMES:
        vector = (16, 200, 64, 7, 1, 1,
                  _POSITIVE_PLANNER_EXTENT if name == 'complete_entry' else 1)
        both = _VISIBLE_REQUEST_CENSUS[name]
        for side in (0, 1):
            value = 0
            for coefficient, powers in both[side]:
                term = coefficient
                for operand, power in zip(vector, powers):
                    term *= operand ** power
                value += term
            totals[side] += value * (2 if name == 'prepared_commit' else 1)
    if tuple(totals) != (271839, 297579):
        _positive_refuse('positive_census_totals')
    return tuple(totals)


def _positive_source_frontier():
    payload_units, work_units = _positive_interval_units()
    h, e, word, header = 4096, 8194, _POSITIVE_WORD, _POSITIVE_HEADER
    # Two full original report pairs plus two complete pre-refusal planners.
    reports = 2 * h * (16 + 14) * 8
    planner_arrays = 2 * 12 * (8 * e + _POSITIVE_INTERFACE)
    # Public list/set/sort/merge output at two planners plus remaining-plan,
    # boundaries tuple/list/tolist/continuation/pack copies. Six distinct
    # pointer+boxed-float sequences per materialization; never alias credit.
    planner_containers = 2 * 6 * (header + e * (word + 32))
    retained_boundaries = 8 * (header + e * (word + 32))
    # Complete commit record26 keys/489 chars; K1 list;13 stage strings433
    # chars; four terminal integer rows. Dict entries retain key+value refs.
    commit_row = (header + 26 * (2 * word + 32 + header) + 4 * 489
                  + header + word + 32 + header + 13 * (word + header)
                  + 4 * 433 + header + 4 * (2 * word + 32 + header))
    rejection_row = header + 8 * (2 * word + 32 + header) + 4 * 454 + header
    logs = 256 * commit_row + 100 * rejection_row
    # Every original expression unit receives one scalar/header and an
    # eight-slot container/reference bank in addition to payload/interface.
    # Field-rich result objects, closures and terminal dictionaries are
    # separately bounded below; this never prices Native B/Q as a grid.
    scalar_expression = payload_units * (32 + header + 8 * word)
    # Existing core/evolve/complete result/control/closure envelopes: each
    # has <=128 dataclass fields, eight full graph occurrences, and all
    # newly created scalar/text/ref headers. Loops append only logs/plans
    # explicitly priced above, not full Native result histories.
    result_metadata = 8 * (header + 128 * (2 * word + header + 32))
    reductions = 42 * ((2 + 5 * 7) * _POSITIVE_INTERFACE + 2 * 7 * _POSITIVE_GRID)
    # Two transport sites at each of seven calls, plus complete closure/
    # terminal sites; twenty full bound calls include a positive import and
    # phase diagnostic allowance even though the closed upwind path has no phase.
    bounds = 20 * 4 * (_POSITIVE_GRID + _POSITIVE_INTERFACE)
    rows = (('visible_grid_and_interfaces', payload_units * (_POSITIVE_GRID + _POSITIVE_INTERFACE)),
        ('typed_backend_flags_lists_text', payload_units * _POSITIVE_FLAG),
        ('visible_scalar_container_references', scalar_expression),
        ('full_reports', reports), ('public_planner_arrays', planner_arrays),
        ('planner_lists_sets_sort_merge', planner_containers),
        ('boundary_continuation_pack_lists', retained_boundaries),
        ('complete_logs', logs), ('result_control_closure_metadata', result_metadata),
        ('pairwise_reductions', reductions), ('summation_bound_outputs', bounds),
        ('regime_counter_metadata', _REGIME_COUNTER_LOGICAL_HOST_BYTES))
    work = work_units * (_POSITIVE_GRID + _POSITIVE_FLAG + 32 + header + 8 * word)
    return rows, work + sum(value for _, value in rows)


def _positive_bootstrap(state, context):
    _positive_pair_guard(state, context)
    context.bridge_limits.__post_init__()
    owner = _POSITIVE_OWNER
    birth = None if owner is None else _positive_owner(context).birth
    _birth, mapped, cap = _positive_current_mapped(birth)
    p = context.native_publication_limits
    closed = _native()[0]
    closed._k_holder(p, closed._KWindowedAvalanchePublicationLimits)
    p.__post_init__()
    # Full Native metadata prefix, not sparse max_proof_bytes alone.
    amount = (p.max_retained_payload_bytes + p.max_mapped_bytes + p.max_proof_bytes
              + p.reserve_host_bytes + _POSITIVE_HELPER_HOST + context.bridge_limits.reserve_host_bytes)
    if mapped + amount > min(cap, context.bridge_limits.max_host_bytes):
        _positive_refuse('positive_before_metadata_AS')
    if owner is not None:
        _positive_debit(owner, context, p.max_work + _POSITIVE_HELPER_WORK,
                        p.max_logical_deep_read_bytes)


def _positive_debit(owner, context, work, reads, versions=0):
    for name, value in (('work', work), ('reads', reads), ('versions', versions)):
        _positive_integer(value, name)
    limits = context.bridge_limits
    new = owner.work + work, owner.reads + reads, owner.versions + versions
    ceilings = limits.max_work, limits.max_logical_read_bytes, limits.max_new_versions
    if any(x > min(y, (1 << 63) - 1) for x, y in zip(new, ceilings)):
        _positive_refuse('positive_monotone_cumulative_ceiling')
    owner.work, owner.reads, owner.versions = new


def _positive_inventory(context, store):
    import os
    import stat
    from pathlib import Path

    root = _run_root(context)
    if (not store.is_absolute() or '..' in store.parts
            or not store.is_relative_to(root)):
        _positive_refuse('positive_store_outside_root')
    for ancestor in (store, *store.parents):
        if not stat.S_ISDIR(ancestor.lstat().st_mode):
            _positive_refuse('positive_store_ancestor')
    total = entries = store_bytes = store_entries = 0
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        if depth > 16 or not stat.S_ISDIR(directory.lstat().st_mode):
            _positive_refuse('positive_inventory_directory')
        with os.scandir(directory) as scan:
            for entry in scan:
                if entries >= context.bridge_limits.max_entries:
                    _positive_refuse('positive_inventory_entries')
                entries += 1
                if len(entry.path) > 4096 or len(entry.name) > 255:
                    _positive_refuse('positive_inventory_path')
                path = Path(entry.path)
                st = entry.stat(follow_symlinks=False)
                inside = path.is_relative_to(store)
                if inside:
                    store_entries += 1
                if stat.S_ISREG(st.st_mode):
                    _positive_integer(st.st_size, 'file_size')
                    total += st.st_size
                    if inside:
                        store_bytes += st.st_size
                    if total > context.bridge_limits.max_run_bytes:
                        _positive_refuse('positive_inventory_run_bytes')
                elif stat.S_ISDIR(st.st_mode):
                    if len(stack) >= context.bridge_limits.max_entries:
                        _positive_refuse('positive_inventory_entries')
                    stack.append((path, depth + 1))
                else:
                    _positive_refuse('positive_inventory_file_type')
    return total, entries, store_bytes, store_entries


def _positive_writer_forecast(state, context, native_versions=0, native_bytes=0):
    for name, value in (('native_versions', native_versions), ('native_bytes', native_bytes)):
        _positive_integer(value, name)
    version = state.voxel_column.deep_store
    h = version.header
    deep = _native()[4]
    v = _positive_integer(version.version_id, 'version')
    publications = 1 + native_versions
    if v + publications + 1 > (1 << 63) - 1:
        _positive_refuse('positive_version_overflow')
    # Canonical width with largest next integer widths; actual immutable
    # chunk fields/digest widths are complete. Include one complete partial.
    manifest = len(deep._canonical_json_bytes(deep._v1_manifest_object(h,
        chunks=version.chunks, version_id=v + publications + 1,
        parent_version_id=v + publications, parent_manifest_digest='0' * 64,
        journal_digest='0' * 64)))
    manifest = max(256 + 192 * h.n_chunks, manifest)
    if manifest > 8192:
        _positive_refuse('positive_canonical_manifest_domain')
    one = h.logical_bytes + manifest
    # Caller publication plus a complete partial attempt; Native stage's
    # version/disk rows remain fully additive, never reuse/dedup credit.
    future_bytes = 2 * one + native_bytes
    future_entries = 2 + 2 * (2 * h.n_chunks + 2) + native_versions * (2 * h.n_chunks + 2)
    total, entries, store_bytes, store_entries = _positive_inventory(context, version.store_dir)
    limits = context.bridge_limits
    if (store_bytes + future_bytes > limits.max_store_bytes
            or total + future_bytes > limits.max_run_bytes
            or entries + future_entries > limits.max_entries
            or store_entries + future_entries > limits.max_entries):
        _positive_refuse('positive_before_write_disk_or_entries')
    return (('current_run', total), ('current_store', store_bytes),
            ('future_and_partial', future_bytes), ('entries', entries),
            ('future_entries', future_entries)), publications


def _positive_request_prefix(context, metadata):
    """Constructor DATA only, separate from ONE unchanged physical-stage fee.

    This reuses the Native logical-reference projections and complete typed
    schemas. No Q is issued here, no pending marker is forged, and no array
    payload/physical call is selected. Existing actual images are already
    held; their complete scans/return/error reference roles are additional.
    """
    closed, _ids, provider, _original, _storage, _water = metadata
    closed._q_check_image()
    closed._w_check_image()
    owners = _original.owners
    for name in _NATIVE_POLICIES[:5]:
        if getattr(owners, name.removeprefix('native_')) is not getattr(context, name):
            _positive_refuse('positive_original_policy_identity')
    if _storage.limits is not context.native_iteration_limits:
        _positive_refuse('positive_B_plan_limit_identity')
    q = context.native_qualification_limits
    if q.max_frame_roles != 12 or q.max_error_roles != 6:
        _positive_refuse('positive_native_constructor_frame_domain')
    # Original metadata350: complete current/fresh/immediate-check graph,
    # interfaces, dtype/partial/control/error rows at all fixed maxima.
    r, t, _interface, text = closed._k_word_projection(
        27, 177, 34, 272, 1088, 98, 9, 160, 2,
        len(closed._K_METADATA_SYMBOLS), 177*128, 4096+4*64)
    original_metadata = 8*(r+t)+text
    # EXACT retained-plan part of Native _p_plan_projection; use every
    # schema family and all optional/error additions, even K6-only extras.
    # Its cumulative phases/iterations are WORK, not new live generations.
    one = 0
    schemas = closed._P_PLAN_SCHEMAS + closed._P_K6_PLAN_SCHEMAS + closed._P_SMALL_PLAN_SCHEMAS
    if len(schemas) > 128:
        _positive_refuse('positive_native_plan_schema_count')
    for _module, name, names in schemas:
        if len(names) > 128:
            _positive_refuse('positive_native_plan_field_count')
        multiplicity = 3 if name in ('_BedExchangePlan', '_EvidencePlan') else 2 if name == '_DepthClassPlan' else 1
        slots = 3*len(names)+8
        for field in names:
            if field in closed._P_TERM_FIELDS:
                slots += closed._P_PLAN_TERMS*(1+closed._P_PLAN_TERM_FIELDS)
            elif field in closed._P_ROW_FIELDS:
                slots += closed._P_PLAN_ROWS*(1+closed._P_PLAN_ROW_FIELDS+closed._P_PLAN_ROW_RANK)
            elif field in closed._P_KEY_FIELDS:
                slots += closed._P_PLAN_SPARSE_KEYS*closed._P_PLAN_SPARSE_WORDS
            elif field in closed._P_VECTOR_FIELDS:
                slots += 128
        one += multiplicity*slots
    one += 2*177
    journal = 3*(context.native_iteration_limits.max_journal_records
                 + context.native_evidence_limits.max_keys)*(2+19)
    plan_slots = (12*(one+journal)
        +12*closed._P_K6_FRAME_REFERENCE_SLOTS
        +12*closed._P_SMALL_FRAME_REFERENCE_SLOTS
        +12*closed._P_SMALL_OWNER_FRAME_REFERENCE_SLOTS+12*6704+12*60)
    plan_text = (12*(2*4*closed._P_PLAN_PATH_CHARS+4*3*closed._P_PLAN_ROWS
        +closed._P_HEADER_JSON_BYTES+closed._P_PLAN_ERROR_TEXT)+12*5474
        +12*4*closed._K_SMALL_ERROR_TEXT_CHARACTERS
        +12*(2*closed._P_SMALL_OWNER_SOURCE_CHARACTERS+4*closed._P_SMALL_OWNER_ERROR_CHARACTERS))
    # Actual held image/provider field dimensions are Native scalar DATA,
    # not observed array/heap sizes. _p_operator_counts includes recursive
    # default/global/builtin/closure records and all paths; image objects
    # themselves are never re-minted by request construction.
    images = (closed._P_OPERATORS, closed._ClosedNativeImage.operators,
              closed._BoundedCPUQualificationImage.operators, closed._ClosedWaterImage.operators)
    image_slots = sum(closed._p_operator_counts(image) for image in images)
    provider_slots = closed._p_provider_slots(provider)
    staging = 2*closed._P_MAX_CODE_NODES*closed._P_MAX_CODE_NAMES+4*closed._P_MAX_GLOBALS
    href = 9*34+8*34+2*34*34+2*177+2*27+closed._P_FINAL_BINDING_REFERENCE_SLOTS
    # Each separate q/water-domain traversal can retain its whole typed
    # pending stack (value/depth/list edge); both are paid without alias credit.
    scanning = 2*3*image_slots
    # Both fresh/held originalPhysical, Q and wet-Q rows: 16 limit fields,
    # 20 domain rows, 14 required rows, <=32 payload/work rows per table.
    # Four complete row-table occurrences coexist (returned/recheck/error).
    records = 4*(32+16+3*(20+14+32+32))
    wr, wt, _wi, wtext = closed._k_word_projection(
        4, 14, 4, 272, 1088, 98, 0, 0, 0,
        len(closed._K_METADATA_SYMBOLS), len('constant_depth'), 0)
    water_metadata = 4*(8*(wr+wt)+wtext)
    # Owning water projection includes every factory/cold-cache/checker's
    # local frame, request/return record, constant label and error role.
    control = closed._w_control_projection()
    # Full closure-depth bound from _p_python_binding/_p_check_python.
    # Every fixed request constructor/check/provider local capacity is paid
    # in all12frame+6error roles. References only; no allocator ABI claim.
    if sum(len(image) for image in images) > 4096:
        _positive_refuse('positive_native_image_frame_count')
    # The existing complete Native image contains the constructor roots AND
    # their protected direct/transitive Native callees/typed constructors.
    # Count every Python/wrapped code capacity, not a handpicked23-root list.
    frames = 0
    for image in images:
        for op in image:
            for python in (op.python, op.wrapped):
                if python is None:
                    continue
                if len(python.nested_codes) > 128:
                    _positive_refuse('positive_native_code_frame_count')
                for code in python.nested_codes:
                    frames += 1+code.co_nlocals+code.co_stacksize+len(code.co_cellvars)+len(code.co_freevars)
    # Retain this complete Native constructor term even where it is only a
    # forward F charge in the original budget. Packet terms are the genuine
    # Native maintained-packet scalar, already checked by bed._inventory;
    # packet length is not the voxel count. No sampled array/heap size.
    packet_terms = _positive_integer(owners.column.height_packet.binding.packet_terms,
                                      'native_packet_terms', 1)
    constructor_arrays = (closed._P_COLD_E1_BYTES+3*16+16*packet_terms+3
                          +25*320+320+2*closed._P_HEADER_JSON_BYTES)
    host = (original_metadata+water_metadata+plan_text+control[3]+constructor_arrays
        +8*(href+plan_slots+2*image_slots+2*provider_slots+staging+scanning+records
            +control[1]+control[2]+18*5*frames)
        +24*(2*16+16*8+32))
    return _positive_integer(host, 'native_constructor_prefix', 1)


def _positive_gate(state, context, metadata, arrays, role, *, control=None,
                   native_request=None, writer=False, native_prefixes=1, prior_metadata=None):
    closed, _ids, provider, original, storage, water_prepasses = metadata
    limits, p = context.bridge_limits, context.native_publication_limits
    if (type(arrays) is not tuple or len(arrays) > 256
            or type(original.prepass.chains) is not tuple or len(original.prepass.chains) > 34
            or type(water_prepasses) is not tuple or len(water_prepasses) > 34
            or type(state.voxel_column.deep_store.chunks) is not tuple
            or not 0 < len(state.voxel_column.deep_store.chunks) <= 16):
        _positive_refuse('positive_gate_metadata_loop_domain')
    for prepass in water_prepasses:
        if type(prepass.chains) is not tuple or len(prepass.chains) > 34:
            _positive_refuse('positive_gate_water_loop_domain')
    owner = _POSITIVE_OWNER
    if owner is not None:
        owner = _positive_owner(context, control)
    elif role not in ('initial', 'validate', 'water_metadata'):
        _positive_refuse('positive_owned_entry_required')
    limits.__post_init__()
    ext_prepasses, external = _array_rows(closed, arrays, limits.max_external_backing_bytes)
    if any(a.nbytes > 4096 * 30 * 8 for a in arrays):
        _positive_refuse('positive_external_logical_extent')
    native_roots = sum(chain[-1].nbytes for chain in original.prepass.chains)
    water_roots = sum(chain[-1].nbytes for pre in water_prepasses for chain in pre.chains)
    prior_roots = 0
    if prior_metadata is not None:
        if (type(prior_metadata) is not tuple or len(prior_metadata) != 6
                or len(prior_metadata[3].prepass.chains) > 34
                or len(prior_metadata[5]) > 34):
            _positive_refuse('positive_prior_metadata_loop_domain')
        for pre in prior_metadata[5]:
            if len(pre.chains) > 34:
                _positive_refuse('positive_prior_water_loop_domain')
        prior_roots = sum(chain[-1].nbytes for chain in prior_metadata[3].prepass.chains)
        prior_roots += sum(chain[-1].nbytes for pre in prior_metadata[5] for chain in pre.chains)
    source, source_work = _positive_source_frontier()
    native_host = p.max_retained_payload_bytes + p.max_proof_bytes + p.reserve_host_bytes
    native_mapped, native_work, native_read = p.max_mapped_bytes, p.max_work, p.max_logical_deep_read_bytes
    native_versions = p.max_new_versions if writer else 0
    native_disk = p.max_disk_write_bytes if writer else 0
    if native_request is not None:
        if type(native_request) is not closed._ClosedWaterRequest:
            _positive_refuse('positive_exact_native_water_request')
        closed._w_check_request(native_request)
        native_plan = native_request.native.original.publication_plan
        if (native_request.native.limits is not context.native_qualification_limits
                or native_plan.iteration_limits is not context.native_iteration_limits
                or native_plan.evidence_limits is not context.native_evidence_limits
                or native_plan.shape != (4, 4, 6, 3, 1)
                or native_plan.prefix != original.owners.column.height_packet.binding.packet_terms):
            _positive_refuse('positive_actual_native_plan_identity')
        required = dict(native_request.required)
        native_host = (required['max_retained_payload_bytes'] + required['max_proof_bytes']
                       + p.reserve_host_bytes)
        native_mapped = required['max_mapped_bytes']
        native_work = required['max_work']
        native_read = required['max_logical_deep_read_bytes']
        native_versions = required['max_new_versions']
        native_disk = required['max_disk_write_bytes']
    for x in (native_host, native_mapped, native_work, native_read, native_versions, native_disk):
        _positive_integer(x, 'native_requirement')
    if type(native_prefixes) is not int or not 1 <= native_prefixes <= 2:
        _positive_refuse('positive_fixed_prefixes')
    chunk = max(ref.nbytes for ref in state.voxel_column.deep_store.chunks)
    directory = limits.max_entries * (4 * 4096 + 256)
    # Complete caller/Original/current/prior/water/external occurrences stay
    # additive even though actual current mapped extent also includes them.
    constructor_prefix = (_positive_request_prefix(context, metadata)
                          if role in ('prepublication', 'wet_stage') else 0)
    payload = source + (('native_future_one_physical_stage', native_host),
        ('outer_request_construction_metadata', constructor_prefix),
        ('original_and_water_roots', native_roots + water_roots),
        ('coexisting_prior_metadata_roots', prior_roots),
        ('external_current_prior_roots', 2 * external),
        ('held_caller_ceiling', limits.max_external_backing_bytes),
        ('bed_storage_caller', storage.caller_live_host_bytes),
        ('Native_B_positive_reserve', context.native_iteration_limits.reserve_host_bytes),
        ('two_storage_and_validation', 2 * storage.new_host_bytes_peak),
        ('four_publication_chunks', 4 * chunk),
        ('directory_scan_error', directory),
        ('rainfall_text', 5 * 4096 * 4 + 128 * 32),
        ('new_helpers_frames_metadata_errors', _POSITIVE_HELPER_HOST))
    mapped = native_mapped + 2 * storage.mapped_bytes_max + 2 * chunk
    reads = native_prefixes * native_read + 4 * storage.logical_deep_read_bytes
    work = (('native', native_prefixes * native_work), ('existing_Source', source_work),
        ('Original_metadata', original.request.logical_work + 256 * len(arrays)),
        ('storage_validation', 8 * chunk + 2 * storage.new_host_bytes_peak),
        ('directory_scan', 2 * directory), ('new_helpers', _POSITIVE_HELPER_WORK))
    total = sum(value for _, value in payload) + mapped + limits.reserve_host_bytes
    birth, current, cap = _positive_current_mapped(None if owner is None else owner.birth)
    if (total > limits.max_host_bytes or mapped > limits.max_mapped_bytes
            or current + total > min(cap, limits.max_host_bytes)
            or reads > limits.max_logical_read_bytes or sum(v for _, v in work) > limits.max_work):
        _positive_refuse('positive_prospective_AS_or_resource_ceiling')
    available = closed._p_modules['maple.aeolian.cohorts.capacity'].default_available_memory_bytes()
    if type(available) is not int or available < total:
        _positive_refuse('genuine_host_headroom')
    disk, versions = (), 0
    if writer:
        disk, versions = _positive_writer_forecast(state, context, native_versions, native_disk)
    if owner is not None:
        _positive_debit(owner, context, sum(v for _, v in work), reads, versions)
        owner.ordinal += 1
        owner.maximum = max(owner.maximum, current + total)
    closed._p_check_provider(provider)
    return _Admission(original, storage, provider, water_prepasses + ext_prepasses,
                       payload, mapped, reads, work, disk)


def _positive_reserve(state, context, metadata, arrays, control, max_report_rows):
    import sys

    caller = sys._getframe(2).f_code
    roles = ((admit_event, 'event'), (admit_initial, 'initial'),
             (admit_water_operands, 'water_metadata'), (validate_shared_bed, 'validate'))
    role = next((name for function, name in roles if function.__code__ is caller), None)
    if role is None:
        _positive_refuse('positive_fixed_metadata_caller')
    if max_report_rows is not None and max_report_rows != 4096:
        _positive_refuse('positive_original_report_declaration')
    if role == 'event':
        _positive_owner(context, control)
    return _positive_gate(state, context, metadata, arrays, role, control=control)


def _positive_publish(state, context, current_time_s):
    _positive_require_action(state, context)
    _time(current_time_s)
    metadata = _metadata(state, context)
    _positive_gate(state, context, metadata, (), 'publish', writer=True)
    _value_validate(state, context, current_time_s, published=False)
    column = _native()[6].publish_pending_column(state.voxel_column, context.geometry,
        context.mass_resolution_kg, max_records=context.window_max_journal_records,
        encoding='full', overlay_max_bytes=None)
    candidate = replace(state, voxel_column=column)
    if candidate.voxel_column.pending_deep is not None:
        _positive_refuse('positive_publication_pending_not_cleared')
    validate_shared_bed(candidate, context, current_time_s=current_time_s, published=True)
    if candidate.reconciliation_baseline is not state.reconciliation_baseline:
        _positive_refuse('publication_baseline_reference')
    return candidate


def _positive_commit(state, context, graph, reference, current_time_s, force, routing_source_m):
    from maple_syrup.sediment_bed import check_routing_source, reroute_threshold_m
    from maple.surface.topographic_commit.triggers import evaluate_commit_triggers

    _positive_require_action(state, context)
    _time(current_time_s)
    if type(force) is not bool or routing_source_m is None:
        _positive_refuse('force_and_held_routing_source')
    _geometry(context, graph, reference)
    arrays = _external_arrays(graph=graph, terrain=reference, routing_source_m=routing_source_m,
                              max_rows=context.bridge_limits.max_report_rows)
    metadata = _metadata(state, context)
    _positive_gate(state, context, metadata, arrays, 'commit_trigger')
    _value_validate(state, context, current_time_s, published=state.voxel_column.pending_deep is None)
    check_routing_source(routing_source_m, graph.shape, graph.xp)
    threshold = reroute_threshold_m(context)
    trigger = evaluate_commit_triggers(state.ledger, state.committed_topography, context.geometry,
                                      current_time_s, context.topographic_commit_spec)
    if not force and not trigger.should_commit:
        return None, trigger, None, None, threshold
    # Before first write: request-CONSTRUCTION metadata PLUS one full
    # unchanged future wet stage. Two physical-stage copies never coexist.
    _positive_gate(state, context, metadata, arrays, 'prepublication', writer=True, native_prefixes=2)
    published = publish_shared_bed(state, context, current_time_s=current_time_s)
    if published.voxel_column.pending_deep is not None:
        _positive_refuse('positive_outer_request_requires_published')
    closed, _bed, ids, _provider = _profile(published, context)
    args, kwargs = _native_args(published, context, ids), _native_kwargs(published, context, current_time_s)
    request = closed._preflight_bounded_cpu_native_water_frame(*args, water=published.water,
        qualification_limits=context.native_qualification_limits, **kwargs)
    if type(request) is not closed._ClosedWaterRequest:
        _positive_refuse('positive_exact_native_water_request')
    closed._w_check_request(request)
    _positive_gate(published, context, _metadata(published, context), arrays,
                   'wet_stage', native_request=request, writer=True, prior_metadata=metadata)
    _positive_require_action(published, context)
    wet = closed._stage_bounded_cpu_native_windowed_water_topography(*args, water=published.water,
        qualification_limits=context.native_qualification_limits, **kwargs)
    if type(wet) is not closed._ClosedWaterReturn:
        _positive_refuse('actual_native_wet_return')
    result = wet.result
    if (type(result) is not _native()[3].WindowedCommitResult or wet.depth_update.rule != 'constant_depth'
            or wet.water.depth_m is not wet.depth_update.new_depth_m
            or wet.depth_update.displaced_volume_m3 != 0.0 or wet.depth_update.clamp_added_volume_m3 != 0.0
            or wet.depth_update.clamp_cell_count != 0
            or not closed._p_modules['maple.surface.voxels.windowed_evidence']._same(
                wet.water.depth_m, published.water.depth_m)):
        _positive_refuse('genuine_constant_depth_return')
    candidate = replace(published, voxel_column=result.new_voxel_column,
        active_layer=result.new_active_layer, ledger=result.new_ledger,
        committed_topography=result.new_committed_state, sediment_availability=wet.availability,
        water=wet.water, reconciliation_baseline=result.new_reconciliation_baseline)
    validate_shared_bed(candidate, context, current_time_s=current_time_s, published=False)
    return candidate, trigger, result, wet, threshold


def complete_positive_windowed_event(state, context, column, field, schedule, vegetation, sediment,
        *, max_end_s, control, policy, report_every_s):
    """One original complete_event; genuine positive water, no checkpoint lane.

    The selected Linux/proc/ABI/provider/Source runtime is Root-qualified.
    Current mappings plus forward logical fees do not guarantee completion.
    """
    import threading
    from maple_syrup.complete_event import CompletionPolicy, DryHandoff, complete_event

    global _POSITIVE_OWNER
    if _POSITIVE_OWNER is not None or not _positive_context(context):
        _positive_refuse('positive_nonreentrant_exact_lane')
    _positive_pair_guard(state.bed, context)
    if control is None or type(policy) is not CompletionPolicy:
        _positive_refuse('positive_explicit_original_control_policy')
    _holder(policy, CompletionPolicy)
    policy.validated()
    projection = _positive_control_projection(context, control)
    birth, _mapped, _cap = _positive_current_mapped()
    owner = _PositiveOwner(context, control, birth, threading.get_ident(), projection)
    _POSITIVE_OWNER = owner
    try:
        result = complete_event(state, context, column, field, schedule, vegetation, sediment,
            max_end_s=max_end_s, control=control, policy=policy,
            report_every_s=report_every_s, max_report_rows=4096,
            continuation=None, checkpoint_callback=None, hydraulics=None)
        if type(result) is not DryHandoff or result.status != 'complete':
            _positive_refuse('positive_actual_complete_handoff_required')
        _positive_owner(context)
        summary = PositiveWaterAdmissionSummary(owner.ordinal, owner.work, owner.reads,
                                                 owner.versions, owner.maximum)
        return result, summary
    except BaseException:
        owner.poisoned = True
        raise
    finally:
        owner.closed = True
        _POSITIVE_OWNER = None


def _positive_profile(state, context):
    if not is_shared_pair(state, context):
        _positive_refuse('shared_family_required')
    _holder(state, WindowedSharedBedState, _BED_FIELDS)
    _holder(context, WindowedAvalancheContext)
    closed, bed, *_rest = _native()
    closed._k_holder(context.geometry, closed._KGeometrySpec)
    closed._k_holder(context.avalanching_spec, closed._KAvalanchingSpec)
    closed._k_holder(state.voxel_column, closed._KWindowedVoxelColumnState)
    if context.geometry.dx_m != context.geometry.dy_m:
        _positive_refuse('square_cpu_geometry')
    limits = context.bridge_limits
    _holder(limits, PositiveWindowedWaterBridgeLimits)
    limits.__post_init__()
    _positive_fixed_geometry(state, context)
    WindowedWaterBridgeLimits.__post_init__(limits)
    if (context.depth_update_rule != 'constant_depth' or context.record_exchange_history is not False
            or context.deep_chunk_encoding != 'full' or context.deep_overlay_max_bytes is not None
            or type(context.window_max_journal_records) is not int
            or not 0 < context.window_max_journal_records <= 4096
            or context.avalanching_spec.enabled is not True):
        _positive_refuse('constant_depth_enabled_avalanche_full_step_profile')
    if (type(context.source_identity) is not str or len(context.source_identity) != 64
            or any(ch not in '0123456789abcdef' for ch in context.source_identity)):
        _positive_refuse('source_identity_metadata')
    from maple.core.parameters.grain_classes import GrainClass, GrainClassSet

    _holder(context.grain_classes, GrainClassSet, ('classes',))
    classes = context.grain_classes.classes
    if type(classes) is not tuple or len(classes) != 1:
        _positive_refuse('K1_profile')
    _holder(classes[0], GrainClass)
    ids = (classes[0].class_id,)
    closed._q_check_limits(context.native_qualification_limits)
    closed._q_check_image()
    closed._w_check_image()
    closed._p_check_static_contracts()
    provider = closed._p_provider()  # Genuine held state, never a probe/qualifier.
    prefix = closed._p_modules['maple.core.backend.prefix_cell_reductions']
    identity = closed._p_modules['maple.core.backend.depth_class_reductions']
    kernels = identity._kernel_objects(1)
    if (type(prefix._NUMBA) is not dict or len(kernels) != 6
            or any(prefix._NUMBA.get(name) is not kernels[i + 3]
                   for i, name in enumerate(('encode_cells', 'advance_cells', 'decode_cells')))):
        _positive_refuse('held_E1_dispatchers_required')
    order = closed._w_order_module
    import numpy as np

    layout = (state.voxel_column.n_voxels, 1)
    if type(order._VERIFIED) is not set:
        _positive_refuse('held_numpy_order_required')
    if (type(order.PROBE_EXTENTS) not in (tuple, list) or len(order.PROBE_EXTENTS) > 126
            or any(type(n) is not int or not 0 <= n <= (1 << 63) - 1
                   for n in order.PROBE_EXTENTS)):
        _positive_refuse('positive_held_order_loop_domain')
    for shape in tuple((n,) for n in sorted(set(order.PROBE_EXTENTS) | {layout[0]})) + (layout,):
        if (np.__version__, shape) not in order._VERIFIED:
            _positive_refuse('held_numpy_order_required')
    if context.native_iteration_limits.max_journal_records != context.window_max_journal_records:
        _positive_refuse('single_journal_authority')
    return closed, bed, ids, provider

def _positive_metadata(state, context):
    _positive_bootstrap(state, context)
    closed, bed, ids, provider = _positive_profile(state, context)
    original = closed._preflight_original_metadata(*_native_args(state, context, ids),
        reconciliation_baseline=state.reconciliation_baseline,
        availability=state.sediment_availability, initial_available_fraction=context.initial_available_fraction)
    for channel in ('water_erosion_deposition', 'avalanching'):
        if closed._k_ledger.process_index(channel) not in state.ledger.present_processes:
            _positive_refuse('both_native_process_channels_required')
    if state.water is None or state.sediment_availability is None or context.initial_available_fraction is None:
        _positive_refuse('tracked_water_and_availability_required')
    # No water value validator or pointer/interface capture is run yet.
    if type(closed._W_WATER_FIELDS) is not tuple or len(closed._W_WATER_FIELDS) > 34:
        _positive_refuse('positive_water_field_loop_domain')
    closed._w_holder(state.water, closed._WWaterState, closed._W_WATER_FIELDS)
    water_arrays = tuple(getattr(state.water, name) for name in closed._W_WATER_FIELDS)
    prepass = closed._prepass_rank_chains(water_arrays)
    closed._k_dtype_anchors(prepass)
    closed._w_native_rows(closed._w_graph(state.water)[1],
                         (context.geometry.ny, context.geometry.nx, 1))
    plan = bed._plan_windowed_bed_exchange(state.voxel_column, state.active_layer, context.geometry,
        class_ids=ids, limits=context.native_iteration_limits)
    # This exact new family forwards G4 at the actual public demand entry.
    # The real wet depth callback is separately covered by unchanged NativeQ.
    if (context.native_iteration_limits.max_gather_columns != 4
            or context.native_iteration_limits.max_operations != 2):
        _positive_refuse('positive_native_water_batch')
    return closed, ids, provider, original, plan, (prepass,)
