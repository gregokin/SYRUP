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


def _registry():
    from maple.core.types.sediment_ledger import (
        CompactSedimentLedgerState,
        SedimentLedgerState,
    )
    from maple.core.types.topographic_commit import CommittedTopographyState
    from maple.core.types.water import WaterState

    types = (CompletionPolicy, DryHandoff, EventProgress, QuietProgress,
             SedimentEventControl, SedimentEventResult, StormControl, StormState,
             TerrainReference, SedimentLedgerState, CompactSedimentLedgerState, CommittedTopographyState, WaterState)
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

    def encode(value):
        if is_array(value) and not isinstance(value, np.ndarray):
            value = np.asarray(to_host(value))  # explicit, counted device-to-host transfer of ONE array
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in 'fiub' or not np.isfinite(value).all():
                raise SedimentEventError('checkpoint arrays must be finite numeric/bool data')
            name = f'a{len(arrays):04d}'
            arrays[name] = value
            return {'array': name, 'dtype': value.dtype.str, 'shape': list(value.shape)}
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, SedimentEventState):
            if value.graph.xp is not primary.graph.xp:
                raise SedimentEventError('bundle states are in mixed namespaces')
            return {'state': {k: encode(getattr(value, k)) for k in STATE_FIELDS},
                    'graph_digest': _graph_digest(value.graph), 'graph_binding': value.graph.input_sha256,
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

    encoded = encode(item)
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
        manifest = {'schema': SCHEMA, 'identity': identity,
                    'status': 'complete' if isinstance(item, DryHandoff) else 'wet',
                    'backend': backend,
                    'files': {name: sha256_file(staging / name) for name in PAYLOADS}, 'payload': encoded}
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
