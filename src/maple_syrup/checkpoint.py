"""Versioned SYRUP continuation bundle around actual MAPLE snapshots.

The snapshot owns bed storage/active layer/availability. The companion holds
pending ledger, committed terrain, water/transport state and all accumulators.
No commit is performed here. No pickle or dynamically imported types are used.
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

from maple_syrup.characteristic_transport import PhaseState
from maple_syrup.complete_event import (
    CompletionPolicy,
    DryHandoff,
    EventProgress,
    QuietProgress,
    quiet_metrics,
    water_budget,
)
from maple_syrup.routing_newton import DEFAULT_NEWTON_MAX_ITERATIONS
from maple_syrup.sediment_bed import BedState, TerrainReference, refresh_routing
from maple_syrup.sediment_event import (
    SedimentEventControl,
    SedimentEventError,
    SedimentEventResult,
    SedimentEventState,
    _check_event_inputs,
    sediment_hydrograph_columns,
)
from maple_syrup.sediment_physics import REGIME_CODES, physics_grid_from_graph
from maple_syrup.sediment_transport import transport_network
from maple_syrup.storm import HYDROGRAPH_COLUMNS, StormControl, StormState

SCHEMA = 'maple-syrup-checkpoint/2'  # /2: sub-cell phase state (Phase 7b) in the continuation
SUPERSEDED_SCHEMAS = ('maple-syrup-checkpoint/1',)
PAYLOADS = ('maple_state.npz', 'continuation.npz')
STATE_FIELDS = ('storm', 'bed', 'sediment_velocity_m_s', 'terrain', 'phase')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _graph_digest(graph):
    h = hashlib.sha256()
    for f in dataclasses.fields(graph):
        if f.name in ('xp', 'input_sha256'):
            continue
        value = getattr(graph, f.name)
        # opt-in policy fields (RFID): omitted for the strict default so the digest of every existing graph is unchanged
        if (f.name == 'policy' and value == 'strict') or (f.name == 'pit_storage' and (value is None or not np.any(value))):
            continue
        h.update(f.name.encode())
        if isinstance(value, np.ndarray):
            h.update(str((value.shape, value.dtype.str)).encode())
            h.update(value.tobytes())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


_NEWTON_CONTROL_FIELDS = frozenset({'root_solver', 'newton_max_iterations'})


def _registry():
    from maple.core.types.sediment_ledger import SedimentLedgerState
    from maple.core.types.topographic_commit import CommittedTopographyState
    from maple.core.types.water import WaterState

    types = (CompletionPolicy, DryHandoff, EventProgress, QuietProgress,
             SedimentEventControl, SedimentEventResult, StormControl, StormState,
             TerrainReference, SedimentLedgerState, CommittedTopographyState, WaterState)
    return {t.__name__: t for t in types}


def _primary(item):
    return item.dry_state if isinstance(item, DryHandoff) else item.result.state


def _validate_accumulators(progress, context):
    """Reject broadcastable/castable corruption before continuation arithmetic."""
    r = progress.result
    shape = (context.geometry.ny, context.geometry.nx)
    nc = len(context.grain_classes.classes)

    def array(value, expected, name, *, dtype=np.float64, signed=False):
        if (not isinstance(value, np.ndarray) or value.shape != expected or value.dtype != dtype
                or not np.isfinite(value).all() or (not signed and np.any(value < 0))):
            raise SedimentEventError(f'invalid checkpoint accumulator {name}')

    for name in ('cumulative_rain_m', 'cumulative_intake_m', 'cumulative_saturation_return_m',
                 'cumulative_drainage_m', 'peak_depth_m', 'peak_velocity_m_s', 'last_velocity_m_s'):
        array(getattr(r, name), shape, name)
    for name in ('cumulative_pickup_kg', 'cumulative_deposition_kg', 'cumulative_export_request_kg',
                 'initial_bed_by_cell_class_kg'):
        array(getattr(r, name), (*shape, nc), name)
    for name in ('initial_bed_inventory_kg', 'initial_mobile_kg', 'committed_net_bed_change_kg'):
        array(getattr(r, name), (nc,), name, signed=name == 'committed_net_bed_change_kg')
    array(r.ledger_process_totals_reset_kg, r.state.bed.ledger.process_totals_kg.shape,
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
        array(np.asarray(value), (), name, dtype=np.int64)
    for name in ('cell_steps_no_runon', 'cell_steps_partial_runon', 'cell_steps_complete_runon'):
        array(np.asarray(getattr(r, name)), (), name, dtype=np.int64)
    for name in ('cumulative_export_m3', 'peak_outlet_discharge_m3_s', 'time_of_peak_outlet_s',
                 'max_courant_old', 'max_courant_new', 'max_routing_cell_balance_residual_m',
                 'max_constitutive_residual_m', 'numerical_residual_scalar_abs_kg', 'peak_mobile_kg',
                 'time_of_peak_mobile_s', 'peak_export_rate_kg_s', 'time_of_peak_export_s',
                 'max_sediment_courant', 'max_decay_exponent', 'max_transport_cell_residual_kg',
                 'max_phase_reconciliation_residual_kg'):
        # Reduction results may be NumPy scalars or zero-dimensional arrays.
        array(np.asarray(getattr(r, name)), (), name)
    for name in ('max_transport_substeps_used', 'n_transport_rejections', 'n_commits', 'n_forced_commits',
                 'n_graph_changes', 'rerouted_cells_total', 'n_accepted_steps', 'n_rejected_attempts',
                 'n_maple_water_calls', 'n_inventory_terms', 'n_phase_canonicalized', 'n_phase_rounding_remnants',
                 'phase_steered_cells_total'):
        v = getattr(r, name)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise SedimentEventError(f'invalid checkpoint counter {name}')
    if (r.n_forced_commits > r.n_commits or r.n_graph_changes > r.n_commits
            or r.phase_steered_cells_total > r.rerouted_cells_total
            or (not progress.control.characteristic and (r.n_phase_canonicalized or r.n_phase_rounding_remnants
                                                         or r.phase_steered_cells_total))
            or r.n_maple_water_calls not in (2 * r.n_accepted_steps, 2 * r.n_accepted_steps + 1)
            or len(r.commit_log) > progress.control.max_commit_log
            or len(r.rejections) > progress.control.max_rejections_recorded):
        raise SedimentEventError('inconsistent checkpoint counters/log limits')
    if (r.hydrograph.shape != (len(r.boundaries), len(HYDROGRAPH_COLUMNS))
            or r.sediment_columns != sediment_hydrograph_columns(nc)
            or r.sediment_hydrograph.shape != (len(r.boundaries), len(r.sediment_columns))):
        raise SedimentEventError('checkpoint hydrograph columns mismatch')
    for name in ('origin_t_s', 'initial_surface_m3', 'initial_soil_m3', 'rainfall_end_s', 'report_every_s'):
        v = getattr(progress, name)
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0:
            raise SedimentEventError(f'invalid checkpoint progress {name}')
    if (isinstance(progress.max_report_rows, bool) or not isinstance(progress.max_report_rows, int)
            or progress.max_report_rows < 1 or progress.report_every_s <= 0):
        raise SedimentEventError('invalid checkpoint reporting limits')


def validate_checkpoint_state(item, context, column, sediment):
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
    _validate_accumulators(progress, context)
    plan = progress.planned_boundaries
    if (not isinstance(plan, np.ndarray) or plan.dtype != np.float64 or plan.ndim != 1
            or len(plan) == 0 or not np.isfinite(plan).all() or np.any(np.diff(plan) <= 0)
            or plan[0] <= progress.origin_t_s or r.state.t_s > plan[-1]):
        raise SedimentEventError('invalid checkpoint step/report plan')
    if not isinstance(item, DryHandoff) and (r.state.t_s not in plan or progress.quiet.ready):
        raise SedimentEventError('wet checkpoint must be an unfinished original boundary')
    if (r.hydrograph.dtype != np.float64 or r.sediment_hydrograph.dtype != np.float64
            or r.hydrograph.ndim != 2 or r.sediment_hydrograph.ndim != 2
            or len(r.hydrograph) != len(r.boundaries) or len(r.sediment_hydrograph) != len(r.boundaries)
            or len(r.boundaries) == 0 or len(r.boundaries) > progress.max_report_rows
            or not np.array_equal(r.hydrograph[:, 0], r.boundaries)
            or not np.array_equal(r.sediment_hydrograph[:, 0], r.boundaries)
            or r.boundaries[-1] != r.state.t_s or np.any(np.diff(r.boundaries) <= 0)):
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
            or int(r.hydrograph[-1, 11]) != r.n_accepted_steps):
        raise SedimentEventError('invalid checkpoint global step count')
    states = (r.state, item.dry_state) if isinstance(item, DryHandoff) else (r.state,)
    shape = (context.geometry.ny, context.geometry.nx, len(context.grain_classes.classes))
    for state in states:
        b = state.bed
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
        # scheme / bin count of the control (a mismatch is refused, not reinterpreted).
        _check_event_inputs(state, context, column, sediment, progress.control.storm.root_tolerance_m,
                            progress.control)
    # Same conservation policy as MAPLE; never trust a saved boolean alone.
    if not r.closure()['closed'] or not r.closure()['request_reconciled'] or not water_budget(progress)['closed']:
        raise SedimentEventError('checkpoint cumulative budgets do not close')
    if isinstance(item, DryHandoff):
        if item.status != 'complete' or not progress.quiet.ready or not ledger_is_empty(item.dry_state.bed.ledger):
            raise SedimentEventError('invalid completed handoff status/ledger')
        d = item.dry_state
        if any(np.any(a != 0) for a in (d.storm.depth_m, d.storm.soil_water_m, d.storm.discharge_m2_s,
                                       d.bed.water.mobile_mass_by_cell_class_kg, d.sediment_velocity_m_s)):
            raise SedimentEventError('completed handoff is not dry and free of mobile sediment')
        if d.phase is not None and (np.any(d.phase.position_m != 0) or np.any(d.phase.fraction[..., 1:] != 0)
                                    or np.any(d.phase.fraction[..., 0] != 1)):
            raise SedimentEventError('completed handoff phase state is not canonical')
        budget = water_budget(progress)
        if (not np.array_equal(item.pre_reset_surface_m, r.state.storm.depth_m)
                or not np.array_equal(item.pre_reset_soil_m, r.state.storm.soil_water_m)
                or item.terminal_deposition_kg.shape != shape
                or item.terminal_deposition_kg.dtype != np.float64
                or np.any(item.terminal_deposition_kg < 0)
                or item.accounting['storm_water_budget'] != budget
                or item.accounting['surface_removed_m3'] != budget['surface_final_m3']
                or item.accounting['soil_removed_m3'] != budget['soil_final_m3']
                or item.accounting['sediment_closure'] != r.closure()
                or item.accounting['terminal_deposition_by_class_kg'] != item.terminal_deposition_kg.sum(axis=(0, 1)).tolist()):
            raise SedimentEventError('completed reset accounting disagrees with physical reservoirs')


def save_checkpoint(path, item, context, column, sediment, identity):
    """Write a NEW bundle atomically. Caller supplies hash-bound run identity."""
    from maple.io.outputs.snapshot import save_state_snapshot

    path = Path(path).resolve()
    if path.exists():
        raise SedimentEventError(f'checkpoint path already exists: {path}')
    validate_checkpoint_state(item, context, column, sediment)
    primary = _primary(item)
    arrays = {}
    registry = _registry()

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in 'fiub' or not np.isfinite(value).all():
                raise SedimentEventError('checkpoint arrays must be finite numeric/bool data')
            name = f'a{len(arrays):04d}'
            arrays[name] = value
            return {'array': name, 'dtype': value.dtype.str, 'shape': list(value.shape)}
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, SedimentEventState):
            return {'state': {k: encode(getattr(value, k)) for k in STATE_FIELDS},
                    'graph_digest': _graph_digest(value.graph), 'graph_binding': value.graph.input_sha256}
        if isinstance(value, PhaseState):
            if value.xp is not np:
                raise SedimentEventError('checkpoint phase state must be host NumPy')
            return {'phase': {'fraction': encode(value.fraction), 'position_m': encode(value.position_m),
                              'dx_m': float(value.dx_m), 'n_bins': int(value.n_bins)}}
        if isinstance(value, BedState):
            # No duplicate bed authority in the sidecar: all bed storage is
            # loaded through MAPLE's actual snapshot. Wet/dry water may differ.
            for a, b in ((value.voxel_column.mass_kg, primary.bed.voxel_column.mass_kg),
                         (value.active_layer.mass_kg, primary.bed.active_layer.mass_kg),
                         (value.sediment_availability.available_mass_kg, primary.bed.sediment_availability.available_mass_kg),
                         (value.sediment_availability.bound_mass_kg, primary.bed.sediment_availability.bound_mass_kg)):
                if not np.array_equal(a, b):
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
        save_state_snapshot(b.voxel_column, b.active_layer, primary.t_s, staging / PAYLOADS[0],
                            step=(item.progress if isinstance(item, DryHandoff) else item).result.n_accepted_steps,
                            reason='SYRUP bundle component: companion continuation file required',
                            geometry=context.geometry, mass_resolution_kg=context.mass_resolution_kg,
                            sediment_availability=b.sediment_availability, water=b.water,
                            provenance={'syrup_checkpoint_schema': SCHEMA, 'requires_companion': True,
                                        'identity': identity})
        np.savez(staging / PAYLOADS[1], **arrays)
        manifest = {'schema': SCHEMA, 'identity': identity,
                    'status': 'complete' if isinstance(item, DryHandoff) else 'wet',
                    'files': {name: sha256_file(staging / name) for name in PAYLOADS}, 'payload': encoded}
        (staging / 'checkpoint.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        if path.exists():
            raise SedimentEventError('checkpoint destination appeared during write')
        staging.rename(path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return path


def load_checkpoint(path, context, column, sediment, expected_identity, *, allow_complete=False,
                    max_bytes=1024**3):
    """Validate everything in private objects before returning a continuation."""
    from maple.io.outputs.snapshot import load_state_snapshot

    path = Path(path).resolve()
    metadata = path / 'checkpoint.json'
    if metadata.stat().st_size > 32 * 1024**2:
        raise SedimentEventError('checkpoint metadata exceeds bounded size')
    data = json.loads(metadata.read_text())
    if isinstance(data, dict) and data.get('schema') in SUPERSEDED_SCHEMAS:
        raise SedimentEventError(f"checkpoint schema {data['schema']} predates the sub-cell phase state and cannot "
                                 f"be resumed by {SCHEMA}; no migration is performed")
    if (set(data) != {'schema', 'identity', 'status', 'files', 'payload'} or data['schema'] != SCHEMA
            or data['identity'] != expected_identity or set(data['files']) != set(PAYLOADS)):
        raise SedimentEventError('checkpoint schema, case/source/config identity or payload set mismatch')
    if data['status'] not in ('wet', 'complete') or (data['status'] == 'complete' and not allow_complete):
        raise SedimentEventError('completed event cannot resume or repeat its dry reset')
    total_bytes = 0
    for name in PAYLOADS:
        file = path / name
        if file.is_symlink() or sha256_file(file) != data['files'][name]:
            raise SedimentEventError(f'checkpoint payload hash mismatch: {name}')
        with zipfile.ZipFile(file) as archive:
            total_bytes += sum(info.file_size for info in archive.infolist())
        if total_bytes > max_bytes:
            raise SedimentEventError('checkpoint exceeds bounded uncompressed size')
    snapshot = load_state_snapshot(path / PAYLOADS[0])
    if not snapshot['validated'] or not snapshot['restartable'] or snapshot['is_diagnostic']:
        raise SedimentEventError('invalid MAPLE component snapshot')
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
                values = {k: decode(v) for k, v in value['state'].items()}
                if set(values) != set(STATE_FIELDS):
                    raise SedimentEventError('checkpoint state fields mismatch')
                if values['phase'] is not None and not isinstance(values['phase'], PhaseState):
                    raise SedimentEventError('checkpoint phase field is not a phase state')
                graph = refresh_routing(values['terrain'], values['bed'])
                if _graph_digest(graph) != value['graph_digest']:
                    raise SedimentEventError('restored geometry does not reproduce checkpoint routing')
                graph = dataclasses.replace(graph, input_sha256=value['graph_binding'])
                return SedimentEventState(**values, graph=graph, network=transport_network(graph), grid=physics_grid_from_graph(graph))
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
                if cls is StormControl:
                    ok = present in (expected - _NEWTON_CONTROL_FIELDS, expected)
                else:
                    ok = present == expected
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
    validate_checkpoint_state(item, context, column, sediment)
    return item
