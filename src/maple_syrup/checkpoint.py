"""Versioned SYRUP continuation bundle around actual MAPLE snapshots.

The snapshot owns bed storage/active layer/availability. The companion holds
pending ledger, committed terrain, water/transport state and all accumulators.
No commit is performed here. No pickle or dynamically imported types are used.

Backends (CW-C1). The bundle is a HOST disk format: every array of a CuPy
continuation is brought over through MAPLE's counted `to_host` at encode time
(explicit per array, never a blind tree conversion), the graph digest hashes
the graph's array BYTES whatever namespace they live in (the host digest of
every existing NumPy bundle is unchanged), and the manifest records the
`backend` the bundle was written by (`numpy` / `cupy`; a manifest without the
key is a pre-CW-C1 NumPy bundle). Loading decodes to host objects, checks the
MAPLE snapshot against the companion on the host, then reconstructs the
continuation in the namespace of the caller's column / sediment parameters
(`event_backend.item_to_namespace`: physical arrays placed, graph / network /
grid / phase REBUILT with their `xp`), and validates the reconstructed item in
that namespace. A bundle written by one backend is refused by the other
(cross-backend migration is not qualified and is not needed). Same-backend
resume reproduces the physical arrays, counters and final handoff bitwise;
setup / timing counters are not persisted. Schema `maple-syrup-checkpoint/2`
is unchanged: the backend tag is manifest metadata, no new physical state is
persisted.

CW-B104: every encoded event state carries its HELD hydraulic routing source
(`routing_source`, the committed elevation its graph was built from; it may
lag the latest committed bed). Loading rebuilds the graph from THAT source and
requires the saved graph digest; the saved discharge, phase and hydraulics are
kept, nothing is refreshed on load. A state node without the source (a bundle
written before net-elevation routing) is refused: no reference is guessed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from maple_syrup.characteristic_sparse import SparsePhaseState
from maple_syrup.characteristic_transport import MAX_N_BINS, PhaseState
from maple_syrup.complete_event import (
    CompletionPolicy,
    DryHandoff,
    EventProgress,
    QuietProgress,
    quiet_metrics,
    water_budget,
)
from maple_syrup.event_backend import (
    BACKEND_NAMES,
    arrays_equal,
    backend_name,
    item_to_namespace,
)
from maple_syrup.routing_newton import DEFAULT_NEWTON_MAX_ITERATIONS
from maple_syrup.sediment_bed import (
    BedIntegrationError,
    BedState,
    TerrainReference,
    _maple_voxel_class,
    check_deep_storage_options,
    check_graph_matches_routing_source,
    deep_storage_record,
    routing_graph_from_elevation,
    validate_deep_storage,
)
from maple_syrup.sediment_event import (
    SedimentEventControl,
    SedimentEventError,
    SedimentEventResult,
    SedimentEventState,
    _check_event_inputs,
    sediment_hydrograph_columns,
    validate_exchange_history,
)
from maple_syrup.sediment_physics import REGIME_CODES, physics_grid_from_graph
from maple_syrup.sediment_transport import transport_network
from maple_syrup.storm import HYDROGRAPH_COLUMNS, StormControl, StormState

SCHEMA = 'maple-syrup-checkpoint/2'  # /2: sub-cell phase state (Phase 7b) in the continuation
SUPERSEDED_SCHEMAS = ('maple-syrup-checkpoint/1',)
PAYLOADS = ('maple_state.npz', 'continuation.npz')
STATE_FIELDS = ('storm', 'bed', 'sediment_velocity_m_s', 'terrain', 'phase')
MANIFEST_KEYS = frozenset({'schema', 'identity', 'status', 'files', 'payload'})
# 'backend' absent in pre-CW-C1 bundles: NumPy. 'phase_storage' (CW-B5) is written ONLY by CSR continuations
# ('csr'); a dense bundle is byte-for-byte what it was before CW-B5 and a manifest without the key is dense.
OPTIONAL_MANIFEST_KEYS = frozenset({'backend', 'phase_storage', 'bed_storage', 'deep_storage'})
#: CW-B108: the manifest record of a NON-default deep-chunk encoding (absent = the default 'full' storage, so every
#: default bundle is unchanged): the run option and the saved bed's physical / logical deep-version identities.
#: CW-B108C1: the MAPLE deep-store manifest formats a record may name (v1 full, v2 overlay), plain ints only.
DEEP_STORAGE_FORMAT_VERSIONS = (1, 2)
DEEP_STORAGE_MANIFEST_KEYS = frozenset({'deep_chunk_encoding', 'deep_overlay_max_bytes', 'store_format_version',
                                        'physical_manifest_digest', 'logical_manifest_digest'})
# The typed CSR phase node: every array is a whitelisted dtype / shape, no dense (ny, nx, nc, B) array is ever formed
# when writing, loading or validating it.
CSR_PHASE_ARRAYS = ('row_ptr', 'bin', 'fraction', 'position_m', 'cell_of_slot', 'slot_of_cell')
CSR_PHASE_SCALARS = ('dx_m', 'n_bins', 'n_classes', 'shape', 'graph_digest')
#: CW-F2: default bound on the summed uncompressed bytes of both payload files of a bundle (unchanged since before
#: CW-F2; dense and other callers keep it). A larger bound is a caller-explicit resource control, never inferred from
#: the manifest.
DEFAULT_MAX_CHECKPOINT_BYTES = 1024**3


def _is_hex_digest(value):
    """CW-B108C1: a lowercase 64-hex SHA-256 string (the MAPLE manifest digest form)."""
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def checkpoint_byte_bound(value, name='max_bytes'):
    """CW-F2: a checkpoint uncompressed-byte bound must be a positive integer (Python or NumPy integer, never a
    bool, float, NaN, infinity or None). Returned as a plain `int` (MAPLE's windowed loader requires `int`)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise SedimentEventError(f'{name} must be a positive integer byte bound, got {value!r}')
    return int(value)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _graph_digest(graph):
    """Canonical digest of a routing graph from its array BYTES (host or device, read through MAPLE's counted
    `to_host`) and the `repr` of its scalar / tuple fields. A NumPy graph hashes exactly as before CW-C1."""
    from maple.core.backend import is_array, to_host

    h = hashlib.sha256()
    for f in dataclasses.fields(graph):
        if f.name in ('xp', 'input_sha256'):
            continue
        value = getattr(graph, f.name)
        # opt-in policy fields (RFID): omitted for the strict default so the digest of every existing graph is unchanged
        if (f.name == 'policy' and value == 'strict') or (f.name == 'pit_storage' and (value is None or not np.any(value))):
            continue
        h.update(f.name.encode())
        if is_array(value):
            host = np.asarray(to_host(value))
            h.update(str((host.shape, host.dtype.str)).encode())
            h.update(host.tobytes())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


_NEWTON_CONTROL_FIELDS = frozenset({'root_solver', 'newton_max_iterations'})
# CW-B5 control fields, encoded ONLY for the CSR storage (a dense control encodes exactly as before CW-B5, so every
# existing bundle is unchanged and a bundle without them decodes to the dense default).
_CSR_CONTROL_FIELDS = frozenset({'phase_storage', 'phase_max_nnz'})
# CW-C2 opt-in fields, encoded ONLY when set (the strict default encodes exactly as before, so every existing
# bundle is unchanged and a bundle without them decodes to the strict default; one of the two sets must be present).
_TERMINAL_POLICY_FIELDS = frozenset({'terminal_storage_water'})
_TERRAIN_POLICY_FIELDS = frozenset({'allow_masked_nodata', 'nodata_value', 'allow_pit_storage', 'allow_flat_storage'})
# CW-D2 opt-in fields: OMITTED from the bundle when they hold their default (so every pre-CW-D2 bundle and every
# default-run bundle keeps its exact payload) and defaulted on load when absent. A present value is encoded and
# validated like any other field; an absent value never means "recorded" (`_validate_accumulators` checks the
# history against the control and the commit count).
_OPTIONAL_DEFAULT_FIELDS = {'BedContext': {'window_max_journal_records': None,
                                           # CW-B108 deep storage option (absent = the default full storage)
                                           'deep_chunk_encoding': 'full', 'deep_overlay_max_bytes': None},
                            'SedimentEventControl': {'record_exchange_history': False,
                                                     # CW-B102 opt-in deep publication (absent = the default)
                                                     'deep_publication': 'step',
                                                     'deep_publication_high_water_records': None},
                            'SedimentEventResult': {'exchange_history': None,
                                                    # CW-B102 per-execution operational record (absent = None)
                                                    'deep_publication': None},
                            # CW-B51: omitted when False, so every existing / default bundle payload is unchanged
                            'CompletionPolicy': {'deposit_mobile_at_flow_end': False}}


MAPLE_LEDGER_MODULE = 'maple.core.types.sediment_ledger'


def _native_ledger_types():
    """The native ledger classes the public MAPLE ledger module binds, shared by this module's `_registry` and
    `pond_checkpoint._native_registry`. Same binding rule as the bed classifier (`sediment_bed._maple_voxel_class`):
    `SedimentLedgerState` is required; `CompactSedimentLedgerState` is included only when literally present in
    `vars(module)` (no module `__getattr__` synthesis) and then must be a class. Returns `(dense,)` or
    `(dense, compact)`: the class objects themselves (public re-exports pass; definition names are not checked).

    Nothing is caught here: an import failure of the module (ImportError or any error raised while executing it)
    propagates unchanged, a missing Dense class raises AttributeError and a present non-class binding of either name
    raises BedIntegrationError. Both registries are built before any checkpoint file or directory is created."""
    import importlib

    ledger = importlib.import_module(MAPLE_LEDGER_MODULE)
    dense = _maple_voxel_class(ledger, 'SedimentLedgerState', required=True)
    compact = _maple_voxel_class(ledger, 'CompactSedimentLedgerState', required=False)
    return (dense,) if compact is None else (dense, compact)


def _registry():
    from maple.core.types.topographic_commit import CommittedTopographyState
    from maple.core.types.water import WaterState

    types = (CompletionPolicy, DryHandoff, EventProgress, QuietProgress,
             SedimentEventControl, SedimentEventResult, StormControl, StormState,
             TerrainReference, *_native_ledger_types(), CommittedTopographyState, WaterState)
    return {t.__name__: t for t in types}


def _primary(item):
    return item.dry_state if isinstance(item, DryHandoff) else item.result.state


def _validate_accumulators(progress, context):
    """Reject broadcastable/castable corruption before continuation arithmetic. Backend-aware: every physical
    accumulator must be an array of the state's namespace (0-d reductions may also be host scalars, as the
    NumPy loop produces), finite and of the declared sign, read through MAPLE's counted predicates."""
    from maple.core.backend import array_namespace, is_array, to_bool

    r = progress.result
    xp = r.state.graph.xp
    shape = (context.geometry.ny, context.geometry.nx)
    nc = len(context.grain_classes.classes)

    def array(value, expected, name, *, dtype=np.float64, signed=False, host_scalar_ok=False):
        """Physical arrays must be arrays of the state's namespace with the declared shape and dtype. A 0-D
        reduction (`host_scalar_ok`) may also be a host scalar: a NumPy scalar, or the plain Python `int` /
        `float` the bundle encodes for a NumPy scalar (`np.generic.item()`); it is interpreted through
        `np.asarray` with NO dtype argument so the declared dtype still has to match (an int for a float64
        field or a float for an int64 field is refused, as before CW-C1). Bools, sequences and other
        broadcastable objects are refused."""
        if not is_array(value):
            if (not host_scalar_ok or isinstance(value, bool)
                    or not isinstance(value, (np.generic, int, float))):
                raise SedimentEventError(f'invalid checkpoint accumulator {name}')
            value = np.asarray(value)
        namespace = array_namespace(value)
        if (namespace is not xp and not (host_scalar_ok and namespace is np)) or tuple(value.shape) != expected \
                or value.dtype != dtype:
            raise SedimentEventError(f'invalid checkpoint accumulator {name}')
        ns = namespace
        if not to_bool(ns.all(ns.isfinite(value))) or (not signed and to_bool(ns.any(value < 0))):
            raise SedimentEventError(f'invalid checkpoint accumulator {name}')

    for name in ('cumulative_rain_m', 'cumulative_intake_m', 'cumulative_saturation_return_m',
                 'cumulative_drainage_m', 'peak_depth_m', 'peak_velocity_m_s', 'last_velocity_m_s'):
        array(getattr(r, name), shape, name)
    for name in ('cumulative_pickup_kg', 'cumulative_deposition_kg', 'cumulative_export_request_kg',
                 'initial_bed_by_cell_class_kg'):
        array(getattr(r, name), (*shape, nc), name)
    for name in ('initial_bed_inventory_kg', 'initial_mobile_kg', 'committed_net_bed_change_kg'):
        array(getattr(r, name), (nc,), name, signed=name == 'committed_net_bed_change_kg')
    array(r.ledger_process_totals_reset_kg, tuple(r.state.bed.ledger.process_totals_kg.shape),
          'ledger_process_totals_reset_kg', signed=True)
    expected_classes = {'requested_pickup', 'raindrop_pickup_requested', 'flow_pickup_requested',
                        'actual_pickup', 'availability_shortfall', 'holdings_shortfall',
                        'pickup_numerical_residual', 'deposition_requested', 'deposition_actual',
                        'deposition_unmet', 'export_requested', 'export_actual', 'export_unmet',
                        'transport_budget_residual', 'transport_budget_tolerance', 'deposit_numerical_residual'}
    if set(r.by_class) != expected_classes or set(r.regime_cell_steps) != set(REGIME_CODES):
        raise SedimentEventError('checkpoint accounting keys mismatch')
    for name, value in r.by_class.items():
        array(value, (nc,), name, signed=True)
    for name, value in r.regime_cell_steps.items():
        array(value, (), name, dtype=np.int64, host_scalar_ok=True)
    for name in ('cell_steps_no_runon', 'cell_steps_partial_runon', 'cell_steps_complete_runon'):
        array(getattr(r, name), (), name, dtype=np.int64, host_scalar_ok=True)
    for name in ('cumulative_export_m3', 'peak_outlet_discharge_m3_s', 'time_of_peak_outlet_s',
                 'max_courant_old', 'max_courant_new', 'max_routing_cell_balance_residual_m',
                 'max_constitutive_residual_m', 'numerical_residual_scalar_abs_kg', 'peak_mobile_kg',
                 'time_of_peak_mobile_s', 'peak_export_rate_kg_s', 'time_of_peak_export_s',
                 'max_sediment_courant', 'max_decay_exponent', 'max_transport_cell_residual_kg',
                 'max_phase_reconciliation_residual_kg'):
        # Reduction results may be NumPy scalars or zero-dimensional arrays.
        array(getattr(r, name), (), name, host_scalar_ok=True)
    for name in ('max_transport_substeps_used', 'n_transport_rejections', 'n_commits', 'n_forced_commits',
                 'n_graph_changes', 'rerouted_cells_total', 'n_accepted_steps', 'n_rejected_attempts',
                 'n_maple_water_calls', 'n_inventory_terms', 'n_phase_canonicalized', 'n_phase_rounding_remnants',
                 'phase_steered_cells_total', 'n_reroutes', 'n_net_commits', 'n_routing_gate_evaluations'):
        v = getattr(r, name)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise SedimentEventError(f'invalid checkpoint counter {name}')
    # CW-D2: the history is present exactly when the control records it, and it must cover every commit so far.
    if progress.control.record_exchange_history != (r.exchange_history is not None):
        raise SedimentEventError('checkpoint exchange history disagrees with the control (recorded without the '
                                 'option, or the option without a history)')
    if r.exchange_history is not None:
        validate_exchange_history(r.exchange_history, xp, (*shape, nc), r.n_commits)
    # CW-B104 routing counters: reroutes / net commits are commits; graph changes and rerouted cells need a reroute.
    unrouted = r.max_unrouted_net_displacement_m
    if (isinstance(unrouted, bool) or not isinstance(unrouted, float) or not math.isfinite(unrouted) or unrouted < 0
            or r.routing_source_origin not in ('entry', 'net_reroute')
            or (r.routing_source_origin == 'entry') != (r.routing_source_t_s is None)
            or (r.routing_source_origin == 'entry') != (r.n_reroutes == 0)
            or (r.routing_source_t_s is not None and (isinstance(r.routing_source_t_s, bool)
                                                      or not isinstance(r.routing_source_t_s, float)
                                                      or not math.isfinite(r.routing_source_t_s)))
            or r.n_reroutes > r.n_commits or r.n_forced_commits + r.n_net_commits > r.n_commits
            or r.n_graph_changes > r.n_reroutes or (r.rerouted_cells_total and not r.n_reroutes)):
        raise SedimentEventError('inconsistent checkpoint routing counters')
    if (r.n_forced_commits > r.n_commits or r.n_graph_changes > r.n_commits
            or r.phase_steered_cells_total > r.rerouted_cells_total
            or (not progress.control.characteristic and (r.n_phase_canonicalized or r.n_phase_rounding_remnants
                                                         or r.phase_steered_cells_total))
            or r.n_maple_water_calls not in (2 * r.n_accepted_steps, 2 * r.n_accepted_steps + 1)
            or len(r.commit_log) > progress.control.max_commit_log
            or len(r.rejections) > progress.control.max_rejections_recorded):
        raise SedimentEventError('inconsistent checkpoint counters/log limits')
    if (not is_array(r.hydrograph) or not is_array(r.sediment_hydrograph)
            or array_namespace(r.hydrograph) is not xp or array_namespace(r.sediment_hydrograph) is not xp
            or tuple(r.hydrograph.shape) != (len(r.boundaries), len(HYDROGRAPH_COLUMNS))
            or r.sediment_columns != sediment_hydrograph_columns(nc)
            or tuple(r.sediment_hydrograph.shape) != (len(r.boundaries), len(r.sediment_columns))):
        raise SedimentEventError('checkpoint hydrograph columns mismatch')
    for name in ('origin_t_s', 'initial_surface_m3', 'initial_soil_m3', 'rainfall_end_s', 'report_every_s'):
        v = getattr(progress, name)
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0:
            raise SedimentEventError(f'invalid checkpoint progress {name}')
    if (isinstance(progress.max_report_rows, bool) or not isinstance(progress.max_report_rows, int)
            or progress.max_report_rows < 1 or progress.report_every_s <= 0):
        raise SedimentEventError('invalid checkpoint reporting limits')


def validate_checkpoint_state(item, context, column, sediment):
    from maple.core.backend import to_bool, to_host, to_int
    from maple.core.types.topographic_commit import validate_committed_topography_state
    from maple.coupling.sediment_ledger.validation import validate_ledger_state
    from maple.surface.active_layer.validation import check_active_layer_voxel_partition
    from maple.surface.availability.validation import check_sediment_availability_state
    from maple.surface.topographic_commit.commit import (
        _reconcile_ledger_with_physical_state,
        ledger_is_empty,
        resolve_reconciliation_reference_inventory_kg,
        validate_reconciliation_context,
    )
    from maple.water import validate_water_state

    if not isinstance(item, (EventProgress, DryHandoff)):
        raise SedimentEventError('unknown checkpoint root')
    progress = item.progress if isinstance(item, DryHandoff) else item
    if progress.status != 'wet':
        raise SedimentEventError('unknown continuation status')
    progress.policy.validated()
    progress.control.validated()
    if progress.control.force_final_commit or not progress.control.commit:
        raise SedimentEventError('checkpoint requires normal commits without intermediate forced commits')
    r = progress.result
    xp = r.state.graph.xp
    backend_name(xp)
    if getattr(column, 'xp', None) is not xp or getattr(sediment, 'xp', None) is not xp:
        raise SedimentEventError('checkpoint column / sediment parameters are not in the continuation namespace')
    _validate_accumulators(progress, context)
    plan = progress.planned_boundaries
    if (not isinstance(plan, np.ndarray) or plan.dtype != np.float64 or plan.ndim != 1
            or len(plan) == 0 or not np.isfinite(plan).all() or np.any(np.diff(plan) <= 0)
            or plan[0] <= progress.origin_t_s or r.state.t_s > plan[-1]):
        raise SedimentEventError('invalid checkpoint step/report plan')
    if not isinstance(item, DryHandoff) and (r.state.t_s not in plan or progress.quiet.ready):
        raise SedimentEventError('wet checkpoint must be an unfinished original boundary')
    boundaries = np.asarray(r.boundaries)
    if (r.hydrograph.dtype != np.float64 or r.sediment_hydrograph.dtype != np.float64
            or r.hydrograph.ndim != 2 or r.sediment_hydrograph.ndim != 2
            or len(r.hydrograph) != len(boundaries) or len(r.sediment_hydrograph) != len(boundaries)
            or len(boundaries) == 0 or len(boundaries) > progress.max_report_rows
            or not np.array_equal(np.asarray(to_host(r.hydrograph[:, 0])), boundaries)
            or not np.array_equal(np.asarray(to_host(r.sediment_hydrograph[:, 0])), boundaries)
            or boundaries[-1] != r.state.t_s or np.any(np.diff(boundaries) <= 0)):
        raise SedimentEventError('invalid checkpoint cumulative reporting position')
    if progress.quiet.last_t_s != r.state.t_s:
        raise SedimentEventError('quiet monitor time differs from accepted state')
    since = progress.quiet.since_s
    if since is not None and (not math.isfinite(since) or since < progress.rainfall_end_s or since > r.state.t_s):
        raise SedimentEventError('invalid checkpoint quiet interval')
    qualifies, _ = quiet_metrics(r.state, context, progress.policy, progress.rainfall_end_s)
    expected_ready = since is not None and r.state.t_s - since >= progress.policy.hold_s
    if (not isinstance(progress.quiet.ready, bool) or progress.quiet.ready != expected_ready
            or (since is not None and not qualifies)):
        raise SedimentEventError('checkpoint quiet interval inconsistent with physical state')
    if (isinstance(r.n_accepted_steps, bool) or not isinstance(r.n_accepted_steps, int)
            or r.n_accepted_steps < 1 or r.n_accepted_steps > progress.control.storm.max_steps
            or to_int(r.hydrograph[-1, 11]) != r.n_accepted_steps):
        raise SedimentEventError('invalid checkpoint global step count')
    states = (r.state, item.dry_state) if isinstance(item, DryHandoff) else (r.state,)
    shape = (context.geometry.ny, context.geometry.nx, len(context.grain_classes.classes))
    for state in states:
        if state.graph.xp is not xp:
            raise SedimentEventError('checkpoint states are in mixed namespaces')
        b = state.bed
        from maple_syrup.sediment_bed import is_windowed_bed

        if is_windowed_bed(b):
            from maple.surface.voxels.windowed_snapshot import (
                validate_windowed_partition,
            )

            validate_windowed_partition(b.voxel_column, b.active_layer, context.geometry, context.mass_resolution_kg)
        else:
            check_active_layer_voxel_partition(b.active_layer, b.voxel_column, context.geometry, context.mass_resolution_kg)
        check_sediment_availability_state(b.sediment_availability, b.active_layer, context.mass_resolution_kg)
        if validate_ledger_state(b.ledger) != shape or validate_committed_topography_state(b.committed_topography) != shape:
            raise SedimentEventError('checkpoint MAPLE state geometry mismatch')
        validate_water_state(b.water, *shape)
        validate_reconciliation_context(
            b.ledger, b.voxel_column, b.active_layer, context.geometry,
            committed_state=b.committed_topography, reconciliation_baseline=None,
            current_time_s=state.t_s, operation='SYRUP checkpoint', ledger_shape=shape)
        reference, label = resolve_reconciliation_reference_inventory_kg(b.committed_topography, None)
        _reconcile_ledger_with_physical_state(b.ledger, reference, b.voxel_column, b.active_layer,
                                            context.mass_resolution_kg, reference_label=label, ledger_shape=shape)
        # Includes the phase partition against the ACTUAL mobile mass and the
        # scheme / bin count of the control (a mismatch is refused, not reinterpreted),
        # and the namespace binding of the control (a CUDA control needs a CuPy state).
        _check_event_inputs(state, context, column, sediment, progress.control.storm.root_tolerance_m,
                            progress.control)
    # Same conservation policy as MAPLE; never trust a saved boolean alone.
    if not r.closure()['closed'] or not r.closure()['request_reconciled'] or not water_budget(progress)['closed']:
        raise SedimentEventError('checkpoint cumulative budgets do not close')
    if isinstance(item, DryHandoff):
        if item.status != 'complete' or not progress.quiet.ready or not ledger_is_empty(item.dry_state.bed.ledger):
            raise SedimentEventError('invalid completed handoff status/ledger')
        d = item.dry_state
        if any(to_bool(xp.any(a != 0)) for a in (d.storm.depth_m, d.storm.soil_water_m, d.storm.discharge_m2_s,
                                                 d.bed.water.mobile_mass_by_cell_class_kg, d.sediment_velocity_m_s)):
            raise SedimentEventError('completed handoff is not dry and free of mobile sediment')
        if isinstance(d.phase, SparsePhaseState):
            # CSR canonical = EMPTY: no stored bins, every row pointer 0 (no dense array is formed to check this).
            if d.phase.nnz != 0 or to_bool(xp.any(d.phase.row_ptr != 0)):
                raise SedimentEventError('completed handoff CSR phase state is not the canonical empty state')
        elif d.phase is not None and (to_bool(xp.any(d.phase.position_m != 0))
                                      or to_bool(xp.any(d.phase.fraction[..., 1:] != 0))
                                      or to_bool(xp.any(d.phase.fraction[..., 0] != 1))):
            raise SedimentEventError('completed handoff phase state is not canonical')
        budget = water_budget(progress)
        terminal = item.terminal_deposition_kg
        if (not arrays_equal(item.pre_reset_surface_m, r.state.storm.depth_m)
                or not arrays_equal(item.pre_reset_soil_m, r.state.storm.soil_water_m)
                or tuple(terminal.shape) != shape
                or terminal.dtype != np.float64
                or to_bool(xp.any(terminal < 0))
                or item.accounting['storm_water_budget'] != budget
                or item.accounting['surface_removed_m3'] != budget['surface_final_m3']
                or item.accounting['soil_removed_m3'] != budget['soil_final_m3']
                or item.accounting['sediment_closure'] != r.closure()
                or item.accounting['terminal_deposition_by_class_kg']
                != np.asarray(to_host(terminal.sum(axis=(0, 1))), dtype=np.float64).tolist()):
            raise SedimentEventError('completed reset accounting disagrees with physical reservoirs')


# Private modes are selected by the two owning entry points, never supplied by a public caller.
_WRITE_WIRE = object()
_LAYOUT_WIRE = object()
_SIZE_ONLY_DIGEST = '0' * 64  # byte-count placeholder, never a state digest or authority


def _encode_wire(value, primary, arrays, registry, is_array, to_host, *, _mode=_WRITE_WIRE,
                 _layout=None, _depth=0):
    """The entire native structural encoder, shared by writing and metadata layout."""
    if _mode is _LAYOUT_WIRE:
        if type(_layout) is not _WriteLayout:
            raise SedimentEventError('invalid private checkpoint layout mode')
        _layout.enter(value, _depth, registry)
    elif _mode is not _WRITE_WIRE or _layout is not None:
        raise SedimentEventError('invalid private checkpoint write mode')

    def encode(child):
        return _encode_wire(child, primary, arrays, registry, is_array, to_host,
                            _mode=_mode, _layout=_layout, _depth=_depth + 1)

    if _mode is _WRITE_WIRE and is_array(value) and not isinstance(value, np.ndarray):
        value = np.asarray(to_host(value))  # explicit, counted device-to-host transfer of ONE array
    if isinstance(value, np.ndarray):
        if _mode is _WRITE_WIRE:
            if value.dtype.kind not in 'fiub' or not np.isfinite(value).all():
                raise SedimentEventError('checkpoint arrays must be finite numeric/bool data')
        name = f'a{len(arrays):04d}'
        if _mode is _LAYOUT_WIRE:
            _layout.array(name, value)
        arrays[name] = value
        return {'array': name, 'dtype': value.dtype.str, 'shape': list(value.shape)}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, SedimentEventState):
        if value.graph.xp is not primary.graph.xp:
            raise SedimentEventError('bundle states are in mixed namespaces')
        return {'state': {k: encode(getattr(value, k)) for k in STATE_FIELDS},
                'graph_digest': (_graph_digest(value.graph) if _mode is _WRITE_WIRE else _SIZE_ONLY_DIGEST), 'graph_binding': value.graph.input_sha256,
                'routing_source': encode(value.routing_source_elevation_m)}
    if isinstance(value, SparsePhaseState):
        if value.xp is not primary.graph.xp:
            raise SedimentEventError('checkpoint CSR phase state is not in the continuation namespace')
        node = {name: encode(getattr(value, name)) for name in CSR_PHASE_ARRAYS}  # counted to_host per array
        node.update({'dx_m': float(value.dx_m), 'n_bins': int(value.n_bins), 'n_classes': int(value.n_classes),
                     'shape': [int(value.shape[0]), int(value.shape[1])], 'graph_digest': str(value.network_sha256)})
        return {'csr_phase': node}
    if isinstance(value, PhaseState):
        if value.xp is not primary.graph.xp:
            raise SedimentEventError('checkpoint phase state is not in the continuation namespace')
        return {'phase': {'fraction': encode(value.fraction), 'position_m': encode(value.position_m),
                          'dx_m': float(value.dx_m), 'n_bins': int(value.n_bins)}}
    if isinstance(value, BedState):
        if _mode is _LAYOUT_WIRE:
            if value is not primary.bed:
                raise SedimentEventError('write budget requires the primary wet bed object')
        else:
            # No duplicate bed authority in the sidecar: all bed storage is
            # loaded through MAPLE's actual snapshot. Wet/dry water may differ.
            from maple_syrup.sediment_bed import is_windowed_bed

            if is_windowed_bed(value) != is_windowed_bed(primary.bed):
                raise SedimentEventError('bundle contains incompatible bed storage types')
            if is_windowed_bed(value):
                from maple.surface.voxels.windowed_pending import (
                    require_published_column,
                )
                from maple.surface.voxels.windowed_snapshot import COLUMN_ARRAYS

                require_published_column(value.voxel_column, operation='save_checkpoint')
                if value.voxel_column.manifest_digest != primary.bed.voxel_column.manifest_digest:
                    raise SedimentEventError('bundle contains divergent immutable bed versions')
                if any(not arrays_equal(getattr(value.voxel_column,k),getattr(primary.bed.voxel_column,k))
                       for k in COLUMN_ARRAYS):
                    raise SedimentEventError('bundle contains divergent resident sediment beds')
                column_pairs = ()
            else:
                column_pairs = ((value.voxel_column.mass_kg, primary.bed.voxel_column.mass_kg),)
            for a, b in (*column_pairs,
                         (value.active_layer.mass_kg, primary.bed.active_layer.mass_kg),
                         (value.sediment_availability.available_mass_kg, primary.bed.sediment_availability.available_mass_kg),
                         (value.sediment_availability.bound_mass_kg, primary.bed.sediment_availability.bound_mass_kg)):
                if not arrays_equal(a, b):
                    raise SedimentEventError('bundle contains divergent sediment beds')
        return {'bed': {k: encode(getattr(value, k)) for k in ('water', 'ledger', 'committed_topography')}}
    if dataclasses.is_dataclass(value):
        name = type(value).__name__
        if registry.get(name) is not type(value):
            raise SedimentEventError(f'unregistered checkpoint type {name}')
        names = [f.name for f in dataclasses.fields(value)]
        if isinstance(value, StormControl) and value.root_solver == 'bisection' \
                and value.newton_max_iterations == DEFAULT_NEWTON_MAX_ITERATIONS:
            # conditional metadata: the default bisection control encodes exactly as before the Newton option
            names = [n for n in names if n not in _NEWTON_CONTROL_FIELDS]
        if isinstance(value, SedimentEventControl) and value.phase_storage == 'dense' and value.phase_max_nnz is None:
            names = [n for n in names if n not in _CSR_CONTROL_FIELDS]  # the dense control encodes as before CW-B5
        if isinstance(value, CompletionPolicy) and not value.terminal_storage_water:
            names = [n for n in names if n not in _TERMINAL_POLICY_FIELDS]
        if isinstance(value, TerrainReference) and value.strict:
            names = [n for n in names if n not in _TERRAIN_POLICY_FIELDS]
        optional = _OPTIONAL_DEFAULT_FIELDS.get(name, {})
        # CW-D2: default-valued opt-in fields are omitted so the payload of a default run is unchanged
        names = [n for n in names if not (n in optional and getattr(value, n) == optional[n])]
        return {'type': name, 'fields': {n: encode(getattr(value, n)) for n in names}}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise SedimentEventError('checkpoint mapping keys must be strings')
        return {'dict': {k: encode(v) for k, v in value.items()}}
    if isinstance(value, (tuple, list)):
        return {'tuple' if isinstance(value, tuple) else 'list': [encode(v) for v in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise SedimentEventError('nonfinite checkpoint scalar')
        return value
    raise SedimentEventError(f'unsupported checkpoint field {type(value)}')


def save_checkpoint(path, item, context, column, sediment, identity):
    """Write a NEW bundle atomically. Caller supplies hash-bound run identity. Every device array is encoded
    through an explicit counted `to_host`; the manifest records the backend the item lives in."""
    from maple.core.backend import is_array, to_host
    from maple.io.outputs.snapshot import save_state_snapshot

    path = Path(path).resolve()
    if path.exists():
        raise SedimentEventError(f'checkpoint path already exists: {path}')
    validate_checkpoint_state(item, context, column, sediment)
    # CW-B108: the run's deep-chunk storage option must be valid for the saved bed (read-only; nothing written yet).
    try:
        validate_deep_storage(context, _primary(item).bed)
    except BedIntegrationError as exc:
        raise SedimentEventError(f'save_checkpoint: {exc}') from None
    # CW-B104C1: every saved state's graph must be the graph its terrain reference builds from its HELD routing
    # source (one host rebuild per distinct graph / source / terrain triple of objects, compared on host evidence)
    # BEFORE anything is written: an inconsistent state is refused here rather than published and refused only on
    # load. The source may lag the committed bed (valid). CW-B104C2: the terrain is part of the dedup key (load
    # rebuilds from terrain + source). Load needs no extra rebuild: it reconstructs and checks the saved digest.
    checked = []
    for state in ((item.progress.result.state, item.dry_state) if isinstance(item, DryHandoff)
                  else (item.result.state,)):
        if any(state.graph is g and state.routing_source_elevation_m is s and state.terrain is t
               for g, s, t in checked):
            continue
        try:
            check_graph_matches_routing_source(state.graph, state.terrain, state.routing_source_elevation_m)
        except BedIntegrationError as exc:
            raise SedimentEventError(f'save_checkpoint: {exc}') from None
        checked.append((state.graph, state.routing_source_elevation_m, state.terrain))
    primary = _primary(item)
    backend = backend_name(primary.graph.xp)
    arrays = {}
    registry = _registry()


    encoded = _encode_wire(item, primary, arrays, registry, is_array, to_host)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{path.name}.partial-', dir=path.parent))
    try:
        b = primary.bed
        from maple_syrup.sediment_bed import is_windowed_bed

        if is_windowed_bed(b):
            from maple.surface.voxels.windowed_snapshot import (
                save_windowed_state_snapshot,
            )

            save_windowed_state_snapshot(b.voxel_column,b.active_layer,primary.t_s,staging/PAYLOADS[0],
                                          step=(item.progress if isinstance(item,DryHandoff) else item).result.n_accepted_steps,
                                          geometry=context.geometry,mass_resolution_kg=context.mass_resolution_kg,
                                          sediment_availability=b.sediment_availability,water=b.water,
                                          provenance={'syrup_checkpoint_schema':SCHEMA,'identity':identity,'backend':backend})
        else:
            save_state_snapshot(b.voxel_column, b.active_layer, primary.t_s, staging / PAYLOADS[0],
                                step=(item.progress if isinstance(item, DryHandoff) else item).result.n_accepted_steps,
                                reason='SYRUP bundle component: companion continuation file required',
                                geometry=context.geometry, mass_resolution_kg=context.mass_resolution_kg,
                                sediment_availability=b.sediment_availability, water=b.water,
                                provenance={'syrup_checkpoint_schema': SCHEMA, 'requires_companion': True,
                                            'identity': identity, 'backend': backend})

        np.savez(staging / PAYLOADS[1], **arrays)
        manifest = _checkpoint_manifest(SCHEMA, identity,
                                          'complete' if isinstance(item, DryHandoff) else 'wet',
                                          backend, {name: sha256_file(staging / name) for name in PAYLOADS},
                                          encoded)
        if is_windowed_bed(b):
            manifest['bed_storage'] = 'windowed'
            option = deep_storage_record(context)
            if option is not None:
                # CW-B108: written ONLY for a non-default encoding. Physical (file) and logical (stored-value)
                # identities of the saved deep version are distinct and both recorded.
                store = b.voxel_column.deep_store
                manifest['deep_storage'] = {**option, 'store_format_version': int(store.format_version),
                                            'physical_manifest_digest': store.manifest_digest,
                                            'logical_manifest_digest': store.logical_manifest_digest}
        if isinstance(primary.phase, SparsePhaseState):
            manifest['phase_storage'] = 'csr'  # written only for CSR continuations; dense manifests are unchanged
        (staging / 'checkpoint.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        if path.exists():
            raise SedimentEventError('checkpoint destination appeared during write')
        staging.rename(path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return path


def load_checkpoint(path, context, column, sediment, expected_identity, *, allow_complete=False,
                    max_bytes=DEFAULT_MAX_CHECKPOINT_BYTES):
    """Validate everything in private objects before returning a continuation.

    The continuation is reconstructed in the namespace of `column` / `sediment`
    (which must agree); the bundle's recorded backend must be that namespace
    (cross-backend migration refused). `max_bytes` (CW-F2: validated positive
    integer, refused before any file is read) bounds the summed uncompressed
    size of both payloads, read from their zip central directories before any
    array is materialized."""
    from maple.io.outputs.snapshot import load_state_snapshot

    max_bytes = checkpoint_byte_bound(max_bytes)
    target_xp = getattr(column, 'xp', None)
    if target_xp is None or getattr(sediment, 'xp', None) is not target_xp:
        raise SedimentEventError('column and sediment parameters must share one array namespace')
    target = backend_name(target_xp)
    path = Path(path).resolve()
    metadata = path / 'checkpoint.json'
    if metadata.stat().st_size > 32 * 1024**2:
        raise SedimentEventError('checkpoint metadata exceeds bounded size')
    data = json.loads(metadata.read_text())
    if isinstance(data, dict) and data.get('schema') in SUPERSEDED_SCHEMAS:
        raise SedimentEventError(f"checkpoint schema {data['schema']} predates the sub-cell phase state and cannot "
                                 f"be resumed by {SCHEMA}; no migration is performed")
    if (not isinstance(data, dict) or not MANIFEST_KEYS <= set(data) or set(data) - MANIFEST_KEYS - OPTIONAL_MANIFEST_KEYS
            or data['schema'] != SCHEMA or data['identity'] != expected_identity or set(data['files']) != set(PAYLOADS)):
        raise SedimentEventError('checkpoint schema, case/source/config identity or payload set mismatch')
    saved_backend = data.get('backend', 'numpy')  # a manifest without the key is a pre-CW-C1 NumPy bundle
    if saved_backend not in BACKEND_NAMES:
        raise SedimentEventError(f'checkpoint records an unsupported backend {saved_backend!r}')
    if saved_backend != target:
        raise SedimentEventError(f'checkpoint was written by the {saved_backend!r} backend; loading it into '
                                 f'{target!r} parameters is a cross-backend migration, which is refused (not '
                                 f'qualified); resume with the backend that wrote it')
    if data['status'] not in ('wet', 'complete') or (data['status'] == 'complete' and not allow_complete):
        raise SedimentEventError('completed event cannot resume or repeat its dry reset')
    saved_storage = data.get('phase_storage', 'dense')
    if saved_storage not in ('dense', 'csr'):
        raise SedimentEventError(f'checkpoint records an unsupported phase storage {saved_storage!r}')
    # CW-B108: the deep-chunk storage option of the bundle (absent = default full) must be EXACTLY the resuming
    # context's: a cross-encoding or overlay-bound change is refused here, before any payload or snapshot is read.
    try:
        check_deep_storage_options(context.deep_chunk_encoding, context.deep_overlay_max_bytes)
    except BedIntegrationError as exc:
        raise SedimentEventError(f'load_checkpoint: {exc}') from None
    # CW-B108C1: presence of the KEY decides (an explicit null is malformed, never "absent"); a present record must be
    # an exact, typed, non-default record: plain-int supported store format (1 or 2; never bool / float / string) and
    # 64-hex string digests. The later binding to the actually loaded MAPLE version is retained below.
    has_deep = 'deep_storage' in data
    saved_deep = data.get('deep_storage')
    expected_deep = deep_storage_record(context)
    if has_deep:
        if (not isinstance(saved_deep, dict) or set(saved_deep) != DEEP_STORAGE_MANIFEST_KEYS
                or data.get('bed_storage') != 'windowed'
                or type(saved_deep['store_format_version']) is not int
                or saved_deep['store_format_version'] not in DEEP_STORAGE_FORMAT_VERSIONS
                or not all(_is_hex_digest(saved_deep[k]) for k in ('physical_manifest_digest', 'logical_manifest_digest'))):
            raise SedimentEventError('checkpoint deep_storage record is malformed')
        saved_option = {k: saved_deep[k] for k in ('deep_chunk_encoding', 'deep_overlay_max_bytes')}
        try:
            check_deep_storage_options(*saved_option.values())
        except BedIntegrationError as exc:
            raise SedimentEventError(f'checkpoint deep_storage record is malformed: {exc}') from None
        if saved_option == {'deep_chunk_encoding': 'full', 'deep_overlay_max_bytes': None}:
            raise SedimentEventError('checkpoint deep_storage record is malformed (the default is never recorded)')
    else:
        saved_option = None
    if saved_option != expected_deep:
        raise SedimentEventError(f'checkpoint deep storage {saved_option or "default full"} differs from the resuming '
                                 f'context {expected_deep or "default full"}; a cross-encoding / bound change is '
                                 'refused (no migration)')
    total_bytes = 0
    for name in PAYLOADS:
        file = path / name
        if file.is_symlink() or sha256_file(file) != data['files'][name]:
            raise SedimentEventError(f'checkpoint payload hash mismatch: {name}')
        with zipfile.ZipFile(file) as archive:
            total_bytes += sum(info.file_size for info in archive.infolist())
        if total_bytes > max_bytes:
            raise SedimentEventError('checkpoint exceeds bounded uncompressed size')
    storage = data.get('bed_storage','dense')
    if storage not in ('dense','windowed'):
        raise SedimentEventError('unsupported checkpoint bed storage')
    if storage == 'windowed':
        from maple.surface.voxels.windowed_snapshot import load_windowed_state_snapshot

        snapshot = load_windowed_state_snapshot(path/PAYLOADS[0],geometry=context.geometry,
                                                grain_classes=context.grain_classes,
                                                mass_resolution_kg=context.mass_resolution_kg,max_bytes=max_bytes)
    else:
        snapshot = load_state_snapshot(path / PAYLOADS[0])
    if not snapshot['validated'] or not snapshot['restartable'] or snapshot['is_diagnostic']:
        raise SedimentEventError('invalid MAPLE component snapshot')
    if storage == 'windowed':
        # CW-B108: the loaded deep version (MAPLE's version abstraction, v1 or v2) must be the recorded one: a default
        # bundle names a v1 version; an overlay bundle its recorded physical manifest, logical identity and format.
        store = snapshot['voxel_column'].deep_store
        fmt = getattr(store, 'format_version', 1)
        if not has_deep:
            if fmt != 1:
                raise SedimentEventError('default (full) checkpoint references an overlay (v2) deep version')
        elif (fmt != saved_deep['store_format_version'] or store.manifest_digest != saved_deep['physical_manifest_digest']
              or store.logical_manifest_digest != saved_deep['logical_manifest_digest']):
            raise SedimentEventError('checkpoint deep_storage record disagrees with the loaded deep version')
    registry = _registry()
    used = set()
    with np.load(path / PAYLOADS[1], allow_pickle=False) as archive:
        def decode(value):
            if not isinstance(value, dict):
                if value is None or isinstance(value, (str, bool, int, float)):
                    if isinstance(value, float) and not math.isfinite(value):
                        raise SedimentEventError('nonfinite checkpoint scalar')
                    return value
                raise SedimentEventError('malformed checkpoint scalar')
            keys = set(value)
            if keys == {'array', 'dtype', 'shape'}:
                name = value['array']
                if name in used:
                    raise SedimentEventError('aliased checkpoint array')
                used.add(name)
                a = np.array(archive[name], copy=True, order='C')
                if a.dtype.str != value['dtype'] or list(a.shape) != value['shape'] or a.dtype.kind not in 'fiub' or not np.isfinite(a).all():
                    raise SedimentEventError('checkpoint array dtype, shape or values invalid')
                a.flags.writeable = False
                return a
            if keys == {'state', 'graph_digest', 'graph_binding'}:
                raise SedimentEventError('checkpoint state has no saved hydraulic routing source (written before '
                                         'net-elevation routing); refused: no routing reference is guessed from the '
                                         'latest committed bed')
            if keys == {'state', 'graph_digest', 'graph_binding', 'routing_source'}:
                values = {k: decode(v) for k, v in value['state'].items()}
                source = decode(value['routing_source'])
                if (not isinstance(source, np.ndarray) or source.dtype != np.float64 or source.ndim != 2
                        or source.shape != values['terrain'].initial_committed_elevation_m.shape):
                    raise SedimentEventError('checkpoint routing source dtype or shape invalid')
                if set(values) != set(STATE_FIELDS):
                    raise SedimentEventError('checkpoint state fields mismatch')
                if values['phase'] is not None and not isinstance(values['phase'], (PhaseState, SparsePhaseState)):
                    raise SedimentEventError('checkpoint phase field is not a phase state')
                if isinstance(values['phase'], SparsePhaseState) != (saved_storage == 'csr'):
                    raise SedimentEventError('checkpoint phase storage tag disagrees with the stored phase state')
                # Host reconstruction FROM THE SAVED ROUTING SOURCE and digest check (the digest is of array bytes,
                # namespace-independent). Never rebuilt from the latest committed bed; nothing refreshed on load.
                graph = routing_graph_from_elevation(values['terrain'], source)
                if _graph_digest(graph) != value['graph_digest']:
                    raise SedimentEventError('restored geometry does not reproduce checkpoint routing')
                graph = dataclasses.replace(graph, input_sha256=value['graph_binding'])
                network = transport_network(graph)
                phase = values['phase']
                if isinstance(phase, SparsePhaseState):
                    # The CSR node is bound by its graph digest and slot maps; both must be the restored network's
                    # (the full partition / mobile / inactive / terminal checks follow in the state validation).
                    if phase.network_sha256 != value['graph_binding']:
                        raise SedimentEventError('checkpoint CSR phase state is bound to another routing graph')
                    if tuple(phase.shape) != tuple(graph.shape) or phase.dx_m != graph.dx_m:
                        raise SedimentEventError('checkpoint CSR phase state shape or spacing invalid')
                return SedimentEventState(**values, graph=graph, network=network, grid=physics_grid_from_graph(graph),
                                          routing_source_elevation_m=source)
            if keys == {'csr_phase'}:
                fields = value['csr_phase']
                if not isinstance(fields, dict) or set(fields) != set(CSR_PHASE_ARRAYS) | set(CSR_PHASE_SCALARS):
                    raise SedimentEventError('checkpoint CSR phase fields mismatch')
                arrays_ = {name: decode(fields[name]) for name in CSR_PHASE_ARRAYS}
                n_bins, dx_m, n_classes = fields['n_bins'], fields['dx_m'], fields['n_classes']
                shape, digest = fields['shape'], fields['graph_digest']
                if (isinstance(n_bins, bool) or not isinstance(n_bins, int) or not (1 <= n_bins <= MAX_N_BINS)
                        or isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 1
                        or isinstance(dx_m, bool) or not isinstance(dx_m, (int, float)) or not math.isfinite(dx_m)
                        or dx_m <= 0 or not isinstance(shape, list) or len(shape) != 2
                        or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape)
                        or not isinstance(digest, str)):
                    raise SedimentEventError('checkpoint CSR phase metadata invalid')
                n = shape[0] * shape[1]
                spec = {'row_ptr': (np.int64, None), 'bin': (np.uint8, None), 'fraction': (np.float64, None),
                        'position_m': (np.float64, None), 'cell_of_slot': (np.int64, None), 'slot_of_cell': (np.int64, (n,))}
                for name, (dtype, shp) in spec.items():
                    a = arrays_[name]
                    if not isinstance(a, np.ndarray) or a.dtype != dtype or a.ndim != 1 or (shp is not None and a.shape != shp):
                        raise SedimentEventError(f'checkpoint CSR phase array {name} dtype or shape invalid')
                n_active = int(arrays_['cell_of_slot'].shape[0])
                nnz = int(arrays_['fraction'].shape[0])
                if (arrays_['row_ptr'].shape != (n_active * n_classes + 1,) or arrays_['bin'].shape != (nnz,)
                        or arrays_['position_m'].shape != (nnz,) or n_active < 1 or n_active > n):
                    raise SedimentEventError('checkpoint CSR phase array lengths inconsistent')
                return SparsePhaseState(row_ptr=arrays_['row_ptr'], bin=arrays_['bin'], fraction=arrays_['fraction'],
                                        position_m=arrays_['position_m'], dx_m=float(dx_m), n_bins=int(n_bins),
                                        n_classes=int(n_classes), shape=(int(shape[0]), int(shape[1])),
                                        cell_of_slot=arrays_['cell_of_slot'], slot_of_cell=arrays_['slot_of_cell'],
                                        network_sha256=digest, xp=np)
            if keys == {'phase'}:
                fields = value['phase']
                if not isinstance(fields, dict) or set(fields) != {'fraction', 'position_m', 'dx_m', 'n_bins'}:
                    raise SedimentEventError('checkpoint phase fields mismatch')
                fraction, position = decode(fields['fraction']), decode(fields['position_m'])
                n_bins, dx_m = fields['n_bins'], fields['dx_m']
                if (isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1
                        or isinstance(dx_m, bool) or not isinstance(dx_m, (int, float)) or not math.isfinite(dx_m)
                        or dx_m <= 0 or not isinstance(fraction, np.ndarray) or not isinstance(position, np.ndarray)
                        or fraction.ndim != 4 or fraction.shape != position.shape or fraction.shape[-1] != n_bins
                        or fraction.dtype != np.float64 or position.dtype != np.float64):
                    raise SedimentEventError('checkpoint phase state shape, dtype or bin count invalid')
                return PhaseState(fraction=fraction, position_m=position, dx_m=float(dx_m), n_bins=int(n_bins), xp=np)
            if keys == {'bed'}:
                values = {k: decode(v) for k, v in value['bed'].items()}
                if set(values) != {'water', 'ledger', 'committed_topography'}:
                    raise SedimentEventError('checkpoint MAPLE companion fields mismatch')
                return BedState(snapshot['voxel_column'], snapshot['active_layer'],
                                sediment_availability=snapshot['sediment_availability'], **values)
            if keys == {'type', 'fields'}:
                cls = registry.get(value['type'])
                if cls is None or not isinstance(value['fields'], dict):
                    raise SedimentEventError('unknown checkpoint type or fields')
                expected = {f.name for f in dataclasses.fields(cls)}
                # historical StormControl data has NEITHER Newton field (the bisection control); new data has BOTH.
                # One of the two is malformed metadata and is refused, never defaulted.
                present = set(value['fields'])
                optional = set(_OPTIONAL_DEFAULT_FIELDS.get(value['type'], {}))
                if cls is StormControl:
                    ok = present in (expected - _NEWTON_CONTROL_FIELDS, expected)
                elif cls is SedimentEventControl:
                    # Independent optional history flag; CSR fields remain an inseparable pair.
                    required = expected - optional
                    core_present = present - optional
                    ok = present <= expected and core_present in (required - _CSR_CONTROL_FIELDS, required)
                elif cls is CompletionPolicy:
                    # C2 terminal field: all-or-nothing as before; CW-B51 flow-end flag: optional (absent = False).
                    required = expected - optional
                    ok = present <= expected and (present - optional) in (required - _TERMINAL_POLICY_FIELDS, required)
                elif cls is TerrainReference:
                    ok = present in (expected - _TERRAIN_POLICY_FIELDS, expected)
                else:
                    # CW-D2 opt-in fields may be absent (their dataclass default applies); nothing else may be.
                    ok = (expected - optional) <= present <= expected
                if not ok:
                    raise SedimentEventError('unknown checkpoint type or fields')
                return cls(**{k: decode(v) for k, v in value['fields'].items()})
            if keys == {'dict'}:
                return {k: decode(v) for k, v in value['dict'].items()}
            if keys in ({'tuple'}, {'list'}):
                key = next(iter(keys))
                values = [decode(v) for v in value[key]]
                return tuple(values) if key == 'tuple' else values
            raise SedimentEventError('unknown checkpoint value encoding')

        item = decode(data['payload'])
        if used != set(archive.files):
            raise SedimentEventError('unused or missing checkpoint arrays')
    if isinstance(item, DryHandoff) != (data['status'] == 'complete'):
        raise SedimentEventError('checkpoint root does not match status')
    primary = _primary(item)
    if (primary.t_s != snapshot['time_s'] or snapshot['water'] is None
            or not np.array_equal(primary.bed.water.depth_m, snapshot['water'].depth_m)
            or not np.array_equal(primary.bed.water.mobile_mass_by_cell_class_kg, snapshot['water'].mobile_mass_by_cell_class_kg)):
        raise SedimentEventError('MAPLE snapshot and companion disagree')
    # Explicit reconstruction in the caller's namespace (identity on NumPy), then validation THERE.
    item = item_to_namespace(item, target_xp)
    validate_checkpoint_state(item, context, column, sediment)
    try:
        validate_deep_storage(context, _primary(item).bed)  # CW-B108: source / bed support of the option
    except BedIntegrationError as exc:
        raise SedimentEventError(f'load_checkpoint: {exc}') from None
    return item


# Optional P1 layout admission.  Nothing below is selected by save_checkpoint/load_checkpoint.
# Source contract: CPython 3.12 / NumPy 2.5.3 and the source-qualified MAPLE
# schema-2 snapshot projection.  The caller must bind the immutable dependency
# sources; version text alone is not a substitute for that binding.
@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointWriteLimits:
    max_uncompressed_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES
    max_json_bytes: int = 32 * 1024**2
    max_members_per_archive: int = 256
    max_nodes: int = 65536
    max_depth: int = 32
    max_string: int = 1024
    max_int_bits: int = 256


@dataclasses.dataclass(frozen=True, slots=True)
class NpyMemberLayout:
    key: str
    dtype: str
    shape: tuple[int, ...]
    fortran_order: bool
    data_bytes: int
    npy_header_bytes: int
    npy_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class ArchiveLayout:
    filename: str
    members: tuple[NpyMemberLayout, ...]
    data_bytes: int
    npy_bytes: int
    zip_metadata_bytes: int
    file_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointWriteBudget:
    native_schema: str
    profile: str
    archives: tuple[ArchiveLayout, ArchiveLayout]
    checkpoint_json_bytes: int
    snapshot_provenance_bytes: int
    encode_finite_check_bytes: int
    whole_array_serialization_fallback_bytes: int
    nominal_iterator_buffer_pair_bytes: int
    header_scalar_metadata_bytes: int
    encoded_metadata_upper_bytes: int
    json_overlap_upper_bytes: int
    graph_digest_scratch_bytes: int
    host_payload_copy_bytes: int
    native_call_multiplicities: tuple[tuple[str, int], ...]
    additional_serialization_upper_bytes: int
    uncompressed_payload_bytes: int
    serialized_payload_bytes: int
    native_child_bytes: int


_WRITE_LIMIT_MAXIMA = (DEFAULT_MAX_CHECKPOINT_BYTES, 32 * 1024**2, 256, 65536, 32, 1024, 256)
_WRITE_SNAPSHOT_REASON = 'SYRUP bundle component: companion continuation file required'
_WRITE_CALLS = (
    ('validate_checkpoint_state', 1), ('_validate_accumulators', 1),
    ('quiet_metrics', 1),
    ('validate_checkpoint_state.check_active_layer_voxel_partition', 1),
    ('validate_checkpoint_state.check_sediment_availability_state', 1),
    ('validate_checkpoint_state.validate_ledger_state', 1),
    ('validate_checkpoint_state.validate_committed_topography_state', 1),
    ('validate_checkpoint_state.validate_water_state', 1),
    ('validate_checkpoint_state.validate_reconciliation_context', 1),
    ('validate_checkpoint_state.resolve_reconciliation_reference_inventory_kg', 1),
    ('validate_checkpoint_state._reconcile_ledger_with_physical_state', 1),
    ('validate_checkpoint_state._check_event_inputs', 1),
    ('SedimentEventResult.closure', 2), ('water_budget', 1),
    ('validate_deep_storage', 1), ('check_graph_matches_routing_source', 1),
    ('routing_graph_host_rebuild', 1), ('routing_graph_evidence_compare', 1),
    ('_encode_wire', 1), ('_graph_digest', 1), ('save_state_snapshot', 1),
    ('snapshot.direct_check_voxel_column_state', 1),
    ('snapshot.check_active_layer_voxel_partition', 1),
    ('snapshot.check_sediment_availability_state', 1),
    ('numpy.savez', 2), ('sha256_file', 2), ('checkpoint_json_write', 1),
)


def _write_limits(limits):
    if type(limits) is not CheckpointWriteLimits:
        raise SedimentEventError('write budget requires exact CheckpointWriteLimits')
    for field, maximum in zip(dataclasses.fields(limits), _WRITE_LIMIT_MAXIMA):
        value = getattr(limits, field.name)
        if type(value) is not int or not 1 <= value <= maximum:
            raise SedimentEventError('write budget limit outside the closed positive range')


def _write_plain_metadata(value, limits, *, depth=0, counter=None):
    """Preflight raw JSON metadata without allocating its encoded representation."""
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > limits.max_nodes or depth > limits.max_depth:
        raise SedimentEventError('write budget metadata node/depth cap')
    kind = type(value)
    if kind is str:
        if len(value) > limits.max_string:
            raise SedimentEventError('write budget metadata string cap')
    elif kind is int:
        if value.bit_length() > limits.max_int_bits:
            raise SedimentEventError('write budget metadata integer cap')
    elif kind is float:
        if not math.isfinite(value):
            raise SedimentEventError('write budget nonfinite metadata scalar')
    elif value is None or kind is bool:
        pass
    elif kind in (dict, tuple, list):
        # A mapping has one key node and one value node per entry.
        minimum = len(value) * (2 if kind is dict else 1)
        if minimum > limits.max_nodes - counter[0]:
            raise SedimentEventError('write budget metadata child cap')
        if kind is dict:
            for key, child in value.items():
                if type(key) is not str:
                    raise SedimentEventError('write budget metadata mapping keys must be strings')
                _write_plain_metadata(key, limits, depth=depth + 1, counter=counter)
                _write_plain_metadata(child, limits, depth=depth + 1, counter=counter)
        else:
            for child in value:
                _write_plain_metadata(child, limits, depth=depth + 1, counter=counter)
    else:
        raise SedimentEventError('write budget unsupported metadata type')
    return counter[0]


def _write_json_measure(value, limits, *, indent=None, sort_keys=False, newline=False):
    """Actual JSONEncoder; no dump is materialized before the byte cap."""
    import struct
    import sys

    _write_plain_metadata(value, limits)
    total, chunks, chunk_objects = int(newline), 0, 0
    encoder = json.JSONEncoder(indent=indent, sort_keys=sort_keys, allow_nan=False, ensure_ascii=True)
    for chunk in encoder.iterencode(value):
        total += len(chunk)  # ensure_ascii makes every emitted code point one byte
        if total > limits.max_json_bytes:
            raise SedimentEventError('write budget JSON byte cap')
        chunks += 1
        chunk_objects += sys.getsizeof(chunk)
    # CPython list growth is bounded above by 2*n+8 pointer slots. Native
    # dumps(indent=2) retains the iterencode chunks before joining them.
    chunk_table = sys.getsizeof([]) + (2 * chunks + 8) * struct.calcsize('P')
    return total, chunk_objects + chunk_table


def _write_metadata_size(value):
    """Source-qualified CPython shallow objects plus every bounded child occurrence."""
    import sys

    total = sys.getsizeof(value)
    if type(value) is dict:
        for key, child in value.items():
            total += _write_metadata_size(key) + _write_metadata_size(child)
    elif type(value) in (tuple, list):
        for child in value:
            total += _write_metadata_size(child)
    return total


class _WriteHeaderSink:
    __slots__ = ('size',)

    def __init__(self):
        self.size = 0

    def write(self, data):
        self.size += len(data)
        if self.size > 10000:
            raise SedimentEventError('write budget NPY header cap')


def _write_npy_member(key, value, limits, *, scalar=False):
    """Use the same actual NumPy normalization/header implementation as savez."""
    from numpy.lib import _format_impl

    if type(key) is not str or not key.isascii() or not key or len(key) + 4 > 200 \
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in key):
        raise SedimentEventError('write budget unsafe NPY member name')
    # scalar=True is only used for small, already capped projection metadata.
    if scalar:
        value = np.asanyarray(value)
    if type(value) is not np.ndarray or not value.flags.c_contiguous:
        raise SedimentEventError('write budget requires exact C-contiguous NumPy arrays')
    dtype = value.dtype
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None \
            or dtype.metadata is not None or type(dtype)._legacy is not True \
            or dtype.kind not in 'fiubSU' or not 1 <= dtype.itemsize <= limits.max_uncompressed_bytes:
        raise SedimentEventError('write budget unsupported normalized NPY dtype')
    if not scalar and dtype.kind not in 'fiub':
        raise SedimentEventError('write budget native arrays must be numeric/bool')
    if len(value.shape) > 4:
        raise SedimentEventError('write budget NPY rank cap')
    elements = 1
    for axis in value.shape:
        if type(axis) is not int or not 0 <= axis <= limits.max_uncompressed_bytes:
            raise SedimentEventError('write budget NPY axis cap')
        elements *= axis
        if elements > limits.max_uncompressed_bytes:
            raise SedimentEventError('write budget NPY element cap')
    data_bytes = elements * dtype.itemsize
    if data_bytes > limits.max_uncompressed_bytes:
        raise SedimentEventError('write budget NPY data byte cap')
    # Metadata has been admitted before the native header routine can build its
    # repr/framing. Rank<=4, bounded plain axes and scalar nonstructured dtypes
    # bound that metadata; this is not a copied NPY codec.
    sink = _WriteHeaderSink()
    _format_impl._write_array_header(sink, _format_impl.header_data_from_array_1_0(value))
    if sink.size + data_bytes > limits.max_uncompressed_bytes:
        raise SedimentEventError('write budget NPY byte cap')
    return NpyMemberLayout(key, dtype.str, tuple(value.shape), False,
                           data_bytes, sink.size, sink.size + data_bytes)


def _write_archive(filename, members, limits):
    if len(members) > limits.max_members_per_archive:
        raise SedimentEventError('write budget archive member cap')
    data_bytes = sum(m.data_bytes for m in members)
    npy_bytes = sum(m.npy_bytes for m in members)
    overhead = 22 + sum(96 + 2 * (len(m.key) + 4) for m in members)
    file_bytes = npy_bytes + overhead
    if npy_bytes > limits.max_uncompressed_bytes or file_bytes >= zipfile.ZIP64_LIMIT:
        raise SedimentEventError('write budget archive byte cap')
    return ArchiveLayout(filename, tuple(members), data_bytes, npy_bytes, overhead, file_bytes)


class _WriteLayout:
    __slots__ = ('limits', 'nodes', 'members', 'data_bytes', 'npy_bytes', 'finite_bytes')

    def __init__(self, limits, identity_nodes):
        self.limits = limits
        self.nodes = 32 + 2 * identity_nodes  # two actual identity JSON occurrences plus manifest wrappers
        self.members = []
        self.data_bytes = self.npy_bytes = self.finite_bytes = 0
        self.charge(0)

    def charge(self, nodes):
        self.nodes += nodes
        if self.nodes > self.limits.max_nodes:
            raise SedimentEventError('write budget encoded metadata node cap')

    def enter(self, value, depth, registry):
        # The wire adds wrappers: two levels per source container, at most
        # three at a leaf. This pre-child bound is intentionally conservative.
        if 2 * depth + 3 > self.limits.max_depth:
            raise SedimentEventError('write budget encoded metadata depth cap')
        kind = type(value)
        if kind is np.ndarray:
            if len(value.shape) > 4:
                raise SedimentEventError('write budget NPY rank cap')
            self.charge(10 + len(value.shape))
        elif isinstance(value, np.generic):
            if kind is not value.dtype.type or value.dtype.kind not in 'fiub' or value.dtype.itemsize > 8:
                raise SedimentEventError('write budget unsupported NumPy scalar')
            self.charge(1)
        elif kind in (dict, tuple, list):
            self.charge(3 + (len(value) if kind is dict else 0))
            minimum = len(value)
            if minimum > self.limits.max_nodes - self.nodes:
                raise SedimentEventError('write budget encoded child cap')
            if kind is dict:
                for key in value:
                    if type(key) is not str or len(key) > self.limits.max_string:
                        raise SedimentEventError('write budget mapping key cap')
        elif kind in (str, bool, int, float) or value is None:
            _write_plain_metadata(value, self.limits)
            self.charge(1)
        elif kind in (SedimentEventState, BedState) or kind in tuple(registry.values()):
            names = _WRITE_PROFILE_FIELDS.get(kind.__name__)
            if names is None or tuple(f.name for f in dataclasses.fields(kind)) != names:
                raise SedimentEventError('write budget native field inventory changed')
            if kind is SedimentEventState:
                if value.phase is not None or not _is_hex_digest(value.graph.input_sha256):
                    raise SedimentEventError('write budget requires phase-free held graph identity')
                self.charge(15 + len(STATE_FIELDS))
            elif kind is BedState:
                self.charge(6)
            else:
                # These fields are inspected by native omission predicates
                # BEFORE descending into children. Admit their exact scalar
                # kinds here rather than executing array/custom comparisons.
                if kind is StormControl and (type(value.root_solver) is not str
                        or type(value.newton_max_iterations) is not int):
                    raise SedimentEventError('write budget malformed root-solver metadata')
                if kind is CompletionPolicy and (type(value.terminal_storage_water) is not bool
                        or type(value.deposit_mobile_at_flow_end) is not bool):
                    raise SedimentEventError('write budget malformed completion metadata')
                if kind is TerrainReference and (any(type(getattr(value, name)) is not bool
                        for name in ('allow_masked_nodata', 'allow_pit_storage', 'allow_flat_storage'))
                        or (value.nodata_value is not None and type(value.nodata_value) not in (int, float))):
                    raise SedimentEventError('write budget malformed terrain metadata')
                self.charge(7 + len(names))
        else:
            raise SedimentEventError('write budget unsupported exact native type')

    def array(self, name, value):
        if len(self.members) >= self.limits.max_members_per_archive:
            raise SedimentEventError('write budget archive member cap')
        if self.npy_bytes + value.nbytes > self.limits.max_uncompressed_bytes:
            raise SedimentEventError('write budget companion aggregate byte cap')
        member = _write_npy_member(name, value, self.limits)
        if self.npy_bytes + member.npy_bytes > self.limits.max_uncompressed_bytes:
            raise SedimentEventError('write budget companion aggregate byte cap')
        self.members.append(member)
        self.data_bytes += member.data_bytes
        self.npy_bytes += member.npy_bytes
        self.finite_bytes = max(self.finite_bytes, member.data_bytes // value.dtype.itemsize + 1)


def _write_profile(item, context, column, sediment):
    """Closed metadata admission only; never calls a physical validator."""
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.sediment_availability import SedimentAvailabilityState
    from maple.core.types.voxel import VoxelColumnState
    from maple.core.types.water import WaterState
    from maple_syrup.infiltration import ColumnParameters
    from maple_syrup.routing import RoutingGraph
    from maple_syrup.sediment_bed import BedContext
    from maple_syrup.sediment_physics import SedimentPhysicsParameters

    if type(item) is not EventProgress or type(item.status) is not str or item.status != 'wet' \
            or type(item.result) is not SedimentEventResult \
            or type(context) is not BedContext or type(column) is not ColumnParameters \
            or type(sediment) is not SedimentPhysicsParameters:
        raise SedimentEventError('write budget requires the exact default Dense wet profile')
    state = item.result.state
    if type(state) is not SedimentEventState or type(state.bed) is not BedState \
            or type(state.graph) is not RoutingGraph or state.graph.xp is not np \
            or column.xp is not np or sediment.xp is not np:
        raise SedimentEventError('write budget requires one actual NumPy native state')
    b, control = state.bed, item.control
    if type(b.voxel_column) is not VoxelColumnState or type(b.active_layer) is not ActiveLayerState \
            or type(b.sediment_availability) is not SedimentAvailabilityState or type(b.water) is not WaterState \
            or type(control) is not SedimentEventControl or type(control.storm) is not StormControl:
        raise SedimentEventError('write budget unsupported bed/control type')
    for record, fields in ((control, ('transport_scheme', 'transport_implementation', 'phase_storage',
                                      'deep_publication')),
                           (control.storm, ('implementation', 'root_solver')),
                           (context, ('deep_chunk_encoding',))):
        for name in fields:
            value = getattr(record, name)
            if type(value) is not str or len(value) > 1024:
                raise SedimentEventError('write budget malformed native profile selector')
    high_water = control.deep_publication_high_water_records
    if high_water is not None and type(high_water) is not int:
        raise SedimentEventError('write budget malformed native omission metadata')
    if type(column.ksat_m_per_s) is not np.ndarray:
        raise SedimentEventError('write budget unsupported native column metadata')
    if control.transport_scheme != 'upwind' or control.transport_implementation not in ('auto', 'array') \
            or control.phase_storage != 'dense' or control.phase_max_nnz is not None \
            or control.commit is not True or control.force_final_commit is not False \
            or control.record_exchange_history is not False or control.deep_publication != 'step' \
            or control.storm.implementation != 'array' or control.storm.root_solver != 'bisection' \
            or context.record_exchange_history is not False or context.deep_chunk_encoding != 'full' \
            or context.deep_overlay_max_bytes is not None or item.result.exchange_history is not None \
            or item.result.deep_publication is not None or state.phase is not None:
        raise SedimentEventError('write budget unsupported optional native profile')
    dims = (context.geometry.ny, context.geometry.nx, len(context.grain_classes.classes))
    if any(type(n) is not int or not 1 <= n <= 64 for n in dims[:2]) \
            or type(dims[2]) is not int or not 1 <= dims[2] <= 6 or dims[0] * dims[1] > 64:
        raise SedimentEventError('write budget cell/class cap')
    if type(state.graph.shape) is not tuple or len(state.graph.shape) != 2 \
            or any(type(n) is not int for n in state.graph.shape) \
            or type(sediment.n_classes) is not int or state.graph.shape != dims[:2] \
            or column.shape != dims[:2] or sediment.n_classes != dims[2]:
        raise SedimentEventError('write budget native geometry/class metadata mismatch')
    mass = b.voxel_column.mass_kg
    if type(mass) is not np.ndarray or len(mass.shape) != 4 or mass.shape[:2] != dims[:2] \
            or not 1 <= mass.shape[2] <= 16 or mass.shape[3] != dims[2]:
        raise SedimentEventError('write budget voxel layout cap')
    for n, maximum in ((item.max_report_rows, 4096), (control.storm.max_steps, 10000),
                       (control.max_commit_log, 65536), (control.max_rejections_recorded, 65536)):
        if type(n) is not int or not 1 <= n <= maximum:
            raise SedimentEventError('write budget native counter policy cap')
    if len(item.result.commit_log) > control.max_commit_log \
            or len(item.result.rejections) > control.max_rejections_recorded:
        raise SedimentEventError('write budget native log length cap')
    for reports in (item.result.hydrograph, item.result.sediment_hydrograph):
        if type(reports) is not np.ndarray or len(reports.shape) != 2 or reports.shape[0] > item.max_report_rows:
            raise SedimentEventError('write budget native report row cap')
    return state


def _write_snapshot_members(state, context, item, identity, provenance_json, limits):
    """Closed source-qualified metadata projection of the ONE dense native subcall."""
    from maple.io.outputs.snapshot import SNAPSHOT_SCHEMA_VERSION, _EXACT_HISTORY_SCHEMA_VERSION

    if SNAPSHOT_SCHEMA_VERSION != 2 or _EXACT_HISTORY_SCHEMA_VERSION != 'maple-phase4r4-v4':
        raise SedimentEventError('write budget native snapshot schema changed')
    if limits.max_members_per_archive < 22:
        raise SedimentEventError('write budget snapshot member cap')
    step = item.result.n_accepted_steps
    if type(step) is not int or not 0 <= step <= 10000 or type(state.t_s) not in (int, float) \
            or not math.isfinite(state.t_s):
        raise SedimentEventError('write budget snapshot scalar cap')
    # No model array is constructed here: only the native writer's scalar
    # metadata conversions. Empty bytes use actual asanyarray normalization.
    b = state.bed
    values = (
        ('voxel_mass_kg', b.voxel_column.mass_kg, False),
        ('active_mass_kg', b.active_layer.mass_kg, False),
        ('time_s', np.float64(state.t_s), True), ('run_name', np.bytes_(b''), True),
        ('seed', np.int64(0), True), ('step', np.int64(step), True),
        ('reason', np.bytes_(_WRITE_SNAPSHOT_REASON.encode('utf-8')), True),
        ('validated', np.bool_(True), True), ('is_diagnostic', np.bool_(False), True),
        ('restartable', np.bool_(True), True), ('error_type', np.bytes_(b''), True),
        ('error_message', np.bytes_(b''), True), ('snapshot_schema_version', np.int64(2), True),
        ('history_schema_version', np.bytes_(_EXACT_HISTORY_SCHEMA_VERSION.encode('utf-8')), True),
        ('provenance_json', np.bytes_(provenance_json.encode('utf-8')), True),
        ('history_store_path', np.bytes_(b''), True),
        ('has_sediment_availability', np.bool_(True), True),
        ('sediment_availability_available_mass_kg', b.sediment_availability.available_mass_kg, False),
        ('sediment_availability_bound_mass_kg', b.sediment_availability.bound_mass_kg, False),
        ('has_water', np.bool_(True), True), ('water_depth_m', b.water.depth_m, False),
        ('water_mobile_mass_by_cell_class_kg', b.water.mobile_mass_by_cell_class_kg, False),
    )
    return tuple(_write_npy_member(key, value, limits, scalar=scalar) for key, value, scalar in values)


def checkpoint_write_budget(item, context, column, sediment, identity, *, limits):
    """Describe native wire/serialization increments, without validating/saving state.

    This optional SOURCE-qualified CPU profile is not a checkpoint authority,
    whole-heap bound, or native-work reservation. Caller admits the separate
    native passes/owners before calling the unchanged save_checkpoint.
    """
    import os
    import struct
    import sys

    _write_limits(limits)
    if os.linesep != '\n' or os.name != 'posix' or sys.implementation.name != 'cpython' \
            or sys.version_info[:2] != (3, 12) or np.__version__ != '2.5.3' \
            or zipfile.ZIP64_LIMIT != (2**31) - 1:
        raise SedimentEventError('write budget unsupported source-qualified writer stack')
    identity_nodes = _write_plain_metadata(identity, limits)
    primary = _write_profile(item, context, column, sediment)
    registry = _registry()
    layout = _WriteLayout(limits, identity_nodes)
    arrays = {}
    encoded = _encode_wire(item, primary, arrays, registry, None, None,
                           _mode=_LAYOUT_WIRE, _layout=layout)
    provenance = {'syrup_checkpoint_schema': SCHEMA, 'requires_companion': True,
                  'identity': identity, 'backend': 'numpy'}
    provenance_bytes, provenance_chunks = _write_json_measure(provenance, limits, sort_keys=True)
    # A bounded metadata-only scalar is required for exact np.bytes_/asanyarray
    # normalization. The preflight above runs before this JSON allocation.
    provenance_json = json.dumps(provenance, sort_keys=True)
    snapshot = _write_archive(PAYLOADS[0],
                              _write_snapshot_members(primary, context, item, identity,
                                                      provenance_json, limits), limits)
    companion = _write_archive(PAYLOADS[1], layout.members, limits)
    uncompressed = snapshot.npy_bytes + companion.npy_bytes
    if uncompressed > limits.max_uncompressed_bytes:
        raise SedimentEventError('write budget summed uncompressed byte cap')
    manifest = _checkpoint_manifest(SCHEMA, identity, 'wet', 'numpy',
                                      {name: _SIZE_ONLY_DIGEST for name in PAYLOADS}, encoded)
    json_bytes, json_chunks = _write_json_measure(manifest, limits, indent=2, newline=True)
    members = snapshot.members + companion.members
    fallback = 2 * max(m.data_bytes for m in members)
    buffer_pair = 2 * max(max((16 * 1024**2) // np.dtype(m.dtype).itemsize, 1)
                          * np.dtype(m.dtype).itemsize for m in members)
    # Exactly one saved graph digest; inspect only actual graph array metadata.
    graph_scratch = 0
    for field in dataclasses.fields(primary.graph):
        value = getattr(primary.graph, field.name)
        if type(value) is np.ndarray:
            # Graph serialization is not performed by this helper. It is still
            # a caller-reserved rebuild/hash pass, with this separate byte scratch.
            _write_npy_member('graph_metadata', value, limits)
            graph_scratch = max(graph_scratch, value.nbytes)
    pointer = struct.calcsize('P')
    # Count concrete bounded metadata objects, with occurrence counting (no
    # alias deduction). ZipInfo's slot values are scalar/header metadata only.
    zip_objects = 0
    for archive in (snapshot, companion):
        current = (sys.getsizeof([]) + (2 * len(archive.members) + 8) * pointer
                   + sys.getsizeof({m.key + '.npy': None for m in archive.members}))
        for member in archive.members:
            info = zipfile.ZipInfo(member.key + '.npy')
            current += sys.getsizeof(info)
            for slot in zipfile.ZipInfo.__slots__:
                if hasattr(info, slot):
                    slot_value = getattr(info, slot)
                    current += (_write_metadata_size(slot_value) if type(slot_value) is not int
                                else max(sys.getsizeof(slot_value), sys.getsizeof(0xffffffff)))
        zip_objects = max(zip_objects, current)
    member_objects = sum(sys.getsizeof(m)
                         + sum(_write_metadata_size(getattr(m, f.name)) for f in dataclasses.fields(m))
                         for m in members)
    budget_header = (sys.getsizeof(snapshot)
                     + pointer * (len(dataclasses.fields(CheckpointWriteBudget))
                                  - len(dataclasses.fields(ArchiveLayout))))
    metadata = (_write_metadata_size(manifest) + _write_metadata_size(provenance)
                + sys.getsizeof(arrays) + sys.getsizeof(layout.members) + member_objects + zip_objects
                + sys.getsizeof(snapshot) + sys.getsizeof(companion)
                + sys.getsizeof(snapshot.members) + sys.getsizeof(companion.members)
                + sys.getsizeof(members) + budget_header + _write_metadata_size(_WRITE_CALLS))
    # 10000 is the enforced NPY header cap. The small scalar descriptors and
    # their normalized arrays/bytes coexist with header construction; provenance
    # text is charged separately at its actual counted length.
    header_scalar = (2 * 10000 + sum(m.npy_header_bytes for m in members)
                     + max(snapshot.zip_metadata_bytes, companion.zip_metadata_bytes)
                     + 3 * provenance_bytes + 22 * (sys.getsizeof(np.float64(0))
                                                    + sys.getsizeof(np.asanyarray(np.float64(0)))
                                                    + sys.getsizeof(()) + 4 * pointer))
    json_overlap = (3 * json_bytes + 3 * provenance_bytes + json_chunks + provenance_chunks
                    + 12 * limits.max_string + 2)
    additional = (layout.finite_bytes + fallback + buffer_pair + header_scalar + metadata
                  + json_overlap + graph_scratch)
    serialized = snapshot.file_bytes + companion.file_bytes
    return CheckpointWriteBudget(SCHEMA, 'default_dense_numpy_wet_v1', (snapshot, companion),
                                  json_bytes, provenance_bytes, layout.finite_bytes,
                                  fallback, buffer_pair, header_scalar, metadata, json_overlap,
                                  graph_scratch, 0, _WRITE_CALLS, additional,
                                  uncompressed, serialized, serialized + json_bytes)


_WRITE_PROFILE_FIELDS = {'CompletionPolicy': ('max_depth_m', 'surface_volume_m3', 'max_unit_discharge_m2_s', 'outlet_discharge_m3_s', 'hold_s', 'mobile_per_cell_class_kg', 'terminal_storage_water', 'deposit_mobile_at_flow_end'), 'EventProgress': ('result', 'policy', 'quiet', 'control', 'planned_boundaries', 'origin_t_s', 'initial_surface_m3', 'initial_soil_m3', 'rainfall_end_s', 'report_every_s', 'max_report_rows', 'status'), 'QuietProgress': ('since_s', 'last_t_s', 'ready'), 'SedimentEventControl': ('storm', 'sediment_courant_max', 'max_transport_substeps', 'commit', 'force_final_commit', 'max_commit_log', 'max_rejections_recorded', 'transport_scheme', 'phase_bins', 'transport_implementation', 'phase_storage', 'phase_max_nnz', 'record_exchange_history', 'deep_publication', 'deep_publication_high_water_records'), 'SedimentEventResult': ('state', 'hydrograph', 'sediment_hydrograph', 'sediment_columns', 'boundaries', 'cumulative_rain_m', 'cumulative_intake_m', 'cumulative_saturation_return_m', 'cumulative_drainage_m', 'cumulative_export_m3', 'peak_depth_m', 'peak_velocity_m_s', 'last_velocity_m_s', 'peak_outlet_discharge_m3_s', 'time_of_peak_outlet_s', 'max_courant_old', 'max_courant_new', 'max_routing_cell_balance_residual_m', 'max_constitutive_residual_m', 'cell_steps_no_runon', 'cell_steps_partial_runon', 'cell_steps_complete_runon', 'cumulative_pickup_kg', 'cumulative_deposition_kg', 'cumulative_export_request_kg', 'by_class', 'initial_bed_by_cell_class_kg', 'initial_bed_inventory_kg', 'initial_mobile_kg', 'numerical_residual_scalar_abs_kg', 'peak_mobile_kg', 'time_of_peak_mobile_s', 'peak_export_rate_kg_s', 'time_of_peak_export_s', 'max_sediment_courant', 'max_decay_exponent', 'max_transport_cell_residual_kg', 'max_transport_substeps_used', 'n_transport_rejections', 'regime_cell_steps', 'ledger_process_totals_reset_kg', 'committed_net_bed_change_kg', 'n_commits', 'n_forced_commits', 'n_graph_changes', 'rerouted_cells_total', 'commit_log', 'max_phase_reconciliation_residual_kg', 'n_phase_canonicalized', 'n_phase_rounding_remnants', 'phase_steered_cells_total', 'n_accepted_steps', 'n_rejected_attempts', 'rejections', 'min_accepted_dt_s', 'max_accepted_dt_s', 'n_maple_water_calls', 'first_step_wall_s', 'first_step_cpu_s', 'remaining_wall_s', 'remaining_cpu_s', 'commit_wall_s', 'n_inventory_terms', 'exchange_history', 'deep_publication', 'n_reroutes', 'n_net_commits', 'n_routing_gate_evaluations', 'max_unrouted_net_displacement_m', 'routing_source_origin', 'routing_source_t_s'), 'SedimentEventState': ('storm', 'bed', 'sediment_velocity_m_s', 'graph', 'network', 'grid', 'terrain', 'phase', 'routing_source_elevation_m'), 'StormControl': ('max_dt_s', 'min_dt_s', 'max_retries', 'max_steps', 'courant_max', 'bisection_iterations', 'root_tolerance_m', 'implementation', 'root_solver', 'newton_max_iterations'), 'StormState': ('t_s', 'depth_m', 'soil_water_m', 'discharge_m2_s'), 'TerrainReference': ('initial_full_elevation_m', 'initial_committed_elevation_m', 'export_receiver_full', 'active_mask', 'friction_factor', 'dx_m', 'allow_masked_nodata', 'nodata_value', 'allow_pit_storage', 'allow_flat_storage'), 'BedState': ('voxel_column', 'active_layer', 'water', 'ledger', 'committed_topography', 'sediment_availability'), 'SedimentLedgerState': ('pending_bed_mass_change_kg', 'pending_compensation_kg', 'process_totals_kg', 'process_totals_compensation_kg', 'numerical_residual_positive_kg', 'numerical_residual_positive_compensation_kg', 'numerical_residual_negative_kg', 'numerical_residual_negative_compensation_kg', 'actual_mass_positive_kg', 'actual_mass_positive_compensation_kg', 'actual_mass_negative_kg', 'actual_mass_negative_compensation_kg', 'largest_residual_kg', 'n_physical_touches'), 'CommittedTopographyState': ('reference_elevation_m', 'elevation_offset_m', 'committed_active_layer_mass_by_class_kg', 'committed_voxel_mass_by_class_kg', 'last_commit_time_s', 'commit_count'), 'WaterState': ('depth_m', 'mobile_mass_by_cell_class_kg')}


# Optional P2: resource/header audit of one genuine native-written private child.
# The unchanged native loader remains the independent state/physical authority.

def _checkpoint_manifest(schema, identity, status, backend, files, payload):
    """The existing native six-key literal, shared without moving evaluation."""
    return {'schema': schema, 'identity': identity, 'status': status, 'backend': backend,
            'files': files, 'payload': payload}


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointRawFile:
    """Actual caller custody of one fixed file, sealed AFTER genuine native save.

    Records do not establish the genuine-writer premise or confer load authority.
    All three records bind the same still-owned private child directory.
    """
    name: str
    size_bytes: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int
    mode: int
    directory_device: int
    directory_inode: int
    directory_mtime_ns: int
    directory_ctime_ns: int


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointAuditedMember:
    key: str
    dtype: str
    shape: tuple[int, ...]
    npy_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointAuditedArchive:
    filename: str
    members: tuple[CheckpointAuditedMember, ...]
    uncompressed_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointChildAudit:
    """Bounded metadata only; no arrays, progress, streams or finish witness."""
    native_schema: str
    profile: str
    numpy_version: str
    raw_files: tuple[CheckpointRawFile, ...]
    archives: tuple[CheckpointAuditedArchive, CheckpointAuditedArchive]
    raw_child_bytes: int
    uncompressed_payload_bytes: int
    checkpoint_json_bytes: int
    recapture_metadata_bytes: int
    audited_header_metadata_bytes: int
    hash_buffer_pair_bytes: int
    native_call_multiplicities: tuple[tuple[str, int], ...]


_CHILD_NAMES = ('checkpoint.json', *PAYLOADS)
_CHILD_AUDIT_CALLS = (('checkpoint_write_budget', 1),
                      ('_encode_wire.LAYOUT_traversals', 1), ('_encode_wire.WRITE_traversals', 1),
                      ('_graph_digest', 1), ('sha256_file', 3),
                      ('physical_checkpoint._audit_archive', 2),
                      ('json.loads', 0), ('load_checkpoint', 0))


def _child_uint(value):
    # The existing closed256-bit metadata maximum also bounds supplied stat
    # integers. It is not the JSON limit selected for the physical input.
    return type(value) is int and value >= 0 and value.bit_length() <= 256


def _child_layout(layout, limits):
    """Admit only bounded HELD descriptor metadata, without rereading the item."""
    _write_limits(limits)
    if type(layout) is not CheckpointWriteBudget or type(layout.native_schema) is not str \
            or layout.native_schema != SCHEMA or type(layout.profile) is not str \
            or layout.profile != 'default_dense_numpy_wet_v1' \
            or type(layout.archives) is not tuple or len(layout.archives) != 2:
        raise SedimentEventError('child audit requires the exact P1 descriptor')
    for field in dataclasses.fields(layout):
        if field.name not in ('native_schema', 'profile', 'archives', 'native_call_multiplicities') \
                and not _child_uint(getattr(layout, field.name)):
            raise SedimentEventError('child audit malformed P1 scalar metadata')
    calls = layout.native_call_multiplicities
    if type(calls) is not tuple or len(calls) != len(_WRITE_CALLS):
        raise SedimentEventError('child audit malformed P1 call descriptor')
    for pair, expected in zip(calls, _WRITE_CALLS):
        if type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str \
                or len(pair[0]) > 1024 or not _child_uint(pair[1]) or pair != expected:
            raise SedimentEventError('child audit malformed P1 call descriptor')
    for index, archive in enumerate(layout.archives):
        if type(archive) is not ArchiveLayout or type(archive.filename) is not str \
                or archive.filename != PAYLOADS[index] or type(archive.members) is not tuple \
                or not 1 <= len(archive.members) <= limits.max_members_per_archive \
                or (index == 0 and len(archive.members) != 22):
            raise SedimentEventError('child audit malformed archive descriptor')
        for field in ('data_bytes', 'npy_bytes', 'zip_metadata_bytes', 'file_bytes'):
            if not _child_uint(getattr(archive, field)):
                raise SedimentEventError('child audit malformed archive scalar')
        seen = set()
        for member in archive.members:
            if type(member) is not NpyMemberLayout or type(member.key) is not str \
                    or not member.key or len(member.key) + 4 > 200 or not member.key.isascii() \
                    or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
                           for c in member.key) or member.key in seen \
                    or type(member.dtype) is not str or not 1 <= len(member.dtype) <= 1024 \
                    or type(member.shape) is not tuple or len(member.shape) > 4 \
                    or any(not _child_uint(n) for n in member.shape) \
                    or member.fortran_order is not False \
                    or not _child_uint(member.data_bytes) or member.data_bytes > limits.max_uncompressed_bytes \
                    or not _child_uint(member.npy_header_bytes) or not 1 <= member.npy_header_bytes <= 10000 \
                    or not _child_uint(member.npy_bytes) \
                    or member.npy_bytes != member.data_bytes + member.npy_header_bytes:
                raise SedimentEventError('child audit malformed member descriptor')
            seen.add(member.key)
        # Reuse the original metadata-only resource formula; no copied ZIP codec.
        if _write_archive(archive.filename, archive.members, limits) != archive:
            raise SedimentEventError('child audit inconsistent archive descriptor')
    if not 1 <= layout.checkpoint_json_bytes <= limits.max_json_bytes \
            or not 1 <= layout.snapshot_provenance_bytes <= limits.max_json_bytes \
            or layout.host_payload_copy_bytes != 0 \
            or layout.uncompressed_payload_bytes != sum(a.npy_bytes for a in layout.archives) \
            or not 1 <= layout.uncompressed_payload_bytes <= limits.max_uncompressed_bytes \
            or layout.serialized_payload_bytes != sum(a.file_bytes for a in layout.archives) \
            or layout.native_child_bytes != layout.serialized_payload_bytes + layout.checkpoint_json_bytes:
        raise SedimentEventError('child audit inconsistent P1 byte descriptor')


def _child_records(raw_files, layout):
    if type(raw_files) is not tuple or len(raw_files) != 3:
        raise SedimentEventError('child audit requires exactly three raw records')
    expected_sizes = (layout.checkpoint_json_bytes, *(a.file_bytes for a in layout.archives))
    directory = None
    for row, name, size in zip(raw_files, _CHILD_NAMES, expected_sizes):
        if type(row) is not CheckpointRawFile or type(row.name) is not str or row.name != name \
                or type(row.sha256) is not str or not _is_hex_digest(row.sha256) \
                or row.sha256 == _SIZE_ONLY_DIGEST:
            raise SedimentEventError('child audit malformed raw custody record')
        for field in dataclasses.fields(row):
            if field.name not in ('name', 'sha256') and not _child_uint(getattr(row, field.name)):
                raise SedimentEventError('child audit malformed raw custody scalar')
        if row.size_bytes != size:
            raise SedimentEventError('child audit raw record exceeds exact P1 file cap')
        current = (row.directory_device, row.directory_inode, row.directory_mtime_ns, row.directory_ctime_ns)
        if directory is not None and current != directory:
            raise SedimentEventError('child audit raw records name different directories')
        directory = current
    if sum(row.size_bytes for row in raw_files) != layout.native_child_bytes:
        raise SedimentEventError('child audit raw aggregate cap mismatch')


def _child_file_stat(path, row):
    import stat

    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or \
            (observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns,
             observed.st_ctime_ns, observed.st_mode) != \
            (row.device, row.inode, row.size_bytes, row.mtime_ns, row.ctime_ns, row.mode):
        raise SedimentEventError('child audit raw file identity or cap changed')


def _child_inventory(path, raw_files):
    """Fixed three-name nofollow inventory; a fourth entry refuses immediately."""
    import os
    import stat

    directory = path.lstat()
    first = raw_files[0]
    if not stat.S_ISDIR(directory.st_mode) or \
            (directory.st_dev, directory.st_ino, directory.st_mtime_ns, directory.st_ctime_ns) != \
            (first.directory_device, first.directory_inode,
             first.directory_mtime_ns, first.directory_ctime_ns):
        raise SedimentEventError('child audit private directory identity changed')
    names = set()
    with os.scandir(path) as entries:
        for entry in entries:
            if len(names) == 3 or entry.name not in _CHILD_NAMES or entry.name in names:
                raise SedimentEventError('child audit unexpected private child inventory')
            names.add(entry.name)
    if len(names) != 3:
        raise SedimentEventError('child audit missing private child file')
    for row in raw_files:
        _child_file_stat(path / row.name, row)


def _child_match_archive(observed, used, expected):
    if type(observed) is not dict or len(observed) != len(expected.members) \
            or tuple(observed) != tuple(m.key for m in expected.members) or used != expected.npy_bytes:
        raise SedimentEventError('child audit native archive inventory mismatch')
    rows = []
    for member in expected.members:
        actual = observed[member.key]
        if actual != {'dtype': member.dtype, 'shape': list(member.shape), 'npy_bytes': member.npy_bytes}:
            raise SedimentEventError('child audit native member layout mismatch')
        rows.append(CheckpointAuditedMember(member.key, actual['dtype'],
                                            tuple(actual['shape']), actual['npy_bytes']))
    return CheckpointAuditedArchive(expected.filename, tuple(rows), used)


def audit_checkpoint_child(path, item, context, column, sediment, identity, *,
                           limits, expected_layout, raw_files):
    """Audit a genuine private native child BEFORE a separately admitted load.

    The caller retains unchanged actual writer input and selected child/custody,
    and admits ALL native/metadata/copy owners before writing. This function is
    not a hostile-upload parser, atomic-descriptor API, physical validator or
    authority to load/finish/publish. It does not write or return model arrays.
    Native encoder/audit exceptions propagate unchanged.
    """
    import sys

    _child_layout(expected_layout, limits)
    _write_plain_metadata(identity, limits)
    _child_records(raw_files, expected_layout)
    if type(path) not in (str, type(Path())):
        raise SedimentEventError('child audit requires a plain private path')
    if type(path) is str and len(path) > 4096:
        raise SedimentEventError('child audit private path metadata cap')
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or len(str(path)) > 4096:
        raise SedimentEventError('child audit requires a bounded absolute private path')

    # ALL fixed count, identities, exact raw file and aggregate caps precede
    # hashes, held-item rerecapture, JSON reads and any ZipFile construction.
    _child_inventory(path, raw_files)
    for row in raw_files:
        _child_file_stat(path / row.name, row)
        actual_digest = sha256_file(path / row.name)
        _child_file_stat(path / row.name, row)
        if actual_digest != row.sha256:
            raise SedimentEventError('child audit actual raw digest mismatch')
    _child_inventory(path, raw_files)

    actual_layout = checkpoint_write_budget(item, context, column, sediment, identity, limits=limits)
    if actual_layout != expected_layout:
        raise SedimentEventError('child audit held input or P1 layout changed')
    from maple.core.backend import is_array, to_host

    primary = _primary(item)
    arrays = {}
    encoded = _encode_wire(item, primary, arrays, _registry(), is_array, to_host)
    payload_hashes = {name: raw_files[index + 1].sha256 for index, name in enumerate(PAYLOADS)}
    manifest = _checkpoint_manifest(SCHEMA, identity, 'wet', 'numpy', payload_hashes, encoded)
    json_bytes, _json_chunks = _write_json_measure(manifest, limits, indent=2, newline=True)
    if json_bytes != expected_layout.checkpoint_json_bytes:
        raise SedimentEventError('child audit native manifest size mismatch')
    expected_json = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('ascii')
    with (path / 'checkpoint.json').open('rb') as stream:
        actual_json = stream.read(json_bytes + 1)
    if len(actual_json) != json_bytes or actual_json != expected_json:
        raise SedimentEventError('child audit native manifest bytes mismatch')

    from maple.aeolian.scheduler.physical_checkpoint import _audit_archive

    first, used0 = _audit_archive(path / PAYLOADS[0], limits.max_uncompressed_bytes)
    audited0 = _child_match_archive(first, used0, expected_layout.archives[0])
    second, used1 = _audit_archive(path / PAYLOADS[1], limits.max_uncompressed_bytes - used0)
    audited1 = _child_match_archive(second, used1, expected_layout.archives[1])
    if used0 + used1 != expected_layout.uncompressed_payload_bytes \
            or used0 + used1 > limits.max_uncompressed_bytes:
        raise SedimentEventError('child audit native summed uncompressed mismatch')
    _child_inventory(path, raw_files)
    # These are observed bounded metadata increments, not a whole-heap proof.
    recapture_metadata = (_write_metadata_size(manifest) + sys.getsizeof(arrays)
                          + sum(sys.getsizeof(name) for name in arrays))
    header_metadata = _write_metadata_size(first) + _write_metadata_size(second)
    return CheckpointChildAudit(SCHEMA, expected_layout.profile, np.__version__, raw_files,
                                (audited0, audited1), expected_layout.native_child_bytes,
                                used0 + used1, json_bytes, recapture_metadata, header_metadata,
                                2 * 1024**2, _CHILD_AUDIT_CALLS)
