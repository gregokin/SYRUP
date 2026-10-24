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
            _append(arrays, value, shape=())
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
               ('visible_syrup_requests_including_terminal_error', visible))
    mapped = policy.max_mapped_bytes + 2 * storage.mapped_bytes_max + 2 * chunk
    reads = (commits * policy.max_logical_deep_read_bytes
             + (2 * attempts + 2 * commits) * storage.logical_deep_read_bytes
             + (metadata_calls + 8) * h.logical_bytes)
    work = (('future_native_wet_policies', commits * policy.max_work),
            ('additional_metadata_and_native_value_preflights', metadata_calls * policy.max_work),
            ('visible_syrup_loop_expanded_byte_visits', visible_work),
            ('original_and_external_metadata', original.request.logical_work + 256 * len(external_arrays)),
            ('two_owned_directory_scans', 2 * limits.max_entries * (4 * 4096 + 256)),
            ('publication_and_validation_byte_visits', 8 * chunk + 2 * storage.new_host_bytes_peak))
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
