"""Explicit Small K1 positive water on the genuine Native device window.

This adapter owns admission and a single synchronous operation.  Every mass,
water, routing, avalanche, publication and completion operation remains in
Native/SYRUP.  The resource law is a finite constructor projection, not a heap
or allocator theorem.  Its external runtime contracts have no default values.
"""
from __future__ import annotations

import math
import os
import sys
import threading
from dataclasses import dataclass, fields, replace
from pathlib import Path

from maple_syrup.sediment_bed import BedContext, BedIntegrationError
from maple_syrup.windowed_shared_bed import WindowedAvalancheContext, WindowedSharedBedState

_BED_FIELDS = ('voxel_column', 'active_layer', 'water', 'ledger',
               'committed_topography', 'sediment_availability', 'reconciliation_baseline')
_UNIT_NAMES = ('reference', 'scalar', 'text_character', 'sequence_header',
               'mapping_header', 'array_header', 'frame_slot', 'error_shell')
_CONSTRUCTOR_V2_SCHEMA = 'maple_syrup.windowed_device_water.constructor_accounting.v2'
_CONSTRUCTOR_V2_DOMAIN = (16, 1, 6, 3, 4, 320, 10000, 11, 64, 4096)
_CONSTRUCTOR_V2_BANK_NAMES = ('reference_bootstrap', 'oracle_pregenesis', 'phase', 'metadata_gate')
_CONSTRUCTOR_V2_COUNTS = ('call_rows_max', 'call_text_max', 'unrefined_rows', 'unicode_objects', 'unicode_characters',
    'byte_objects', 'byte_payload', 'list_objects', 'list_slots', 'tuple_objects', 'tuple_slots',
    'dict_objects', 'dict_entries', 'set_objects', 'set_entries', 'remaining_v1_text_characters')
_CONSTRUCTOR_V2_UNITS = ('unicode_shell_bytes', 'unicode_character_bytes',
    'bytes_shell_bytes', 'bytes_payload_bytes', 'list_header_bytes', 'list_slot_bytes',
    'tuple_header_bytes', 'tuple_slot_bytes', 'dict_header_bytes', 'dict_entry_bytes',
    'set_header_bytes', 'set_entry_bytes')
# These are the components EXCLUDED by the native payload planners.  Each
# supplied bound must cite the exact current Source/selected runtime contract.
_EXTERNAL_ROLES = ('native_metadata', 'cuda_prepare', 'cuda_attempt', 'transport',
                   'report', 'writer', 'query', 'validation', 'fixture', 'oracle', 'observer')
_PHASES = ('initial', 'entry', 'event', 'attempt', 'water', 'trigger', 'commit', 'publish',
           'terminal', 'validate')
_OWNER = None


def _refuse(code):
    owner = _OWNER
    if type(owner) is _Owner:
        owner.poisoned = True
    raise BedIntegrationError('windowed device water: ' + code)


def _integer(value, name, low=0, high=(1 << 63) - 1):
    if type(value) is not int or not low <= value <= high:
        _refuse('integer_' + name)
    return value


def _holder(value, kind):
    if type(value) is not kind or len(fields(kind)) > 128:
        _refuse('exact_holder')
    names = tuple(f.name for f in fields(kind))
    try:
        actual = vars(value)
    except TypeError:
        _refuse('ordinary_holder_dictionary_required')
    if (type(actual) is not dict or len(actual) > 128
            or any(type(name) is not str for name in actual) or set(actual) != set(names)):
        _refuse('closed_holder')
    return value


def _time(value):
    if type(value) is not float or not math.isfinite(value) or value < 0:
        _refuse('finite_plain_time')
    return value


@dataclass(frozen=True)
class PositiveWindowedDeviceWaterLimits:
    """Logical ceilings within the unchanged protected GPU worker caps.

    Host/device figures are additional logical charges, never measured RSS,
    pool usage or permission to exceed the original monitor's hard limits.
    No CPU PositiveWindowedWaterBridgeLimits is embedded or inherited.
    """
    max_host_bytes: int
    max_device_bytes: int
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
    reserve_device_bytes: int
    safety_fraction: float
    run_dir: object

    def __post_init__(self):
        fixed = {'max_host_bytes': 5 << 30, 'max_device_bytes': 2 << 30,
                 'max_mapped_bytes': 128 << 20, 'max_store_bytes': 64 << 20,
                 'max_run_bytes': 512 << 20, 'max_entries': 4096,
                 'max_external_backing_bytes': 64 << 20, 'max_event_steps': 10000,
                 'max_retries': 10, 'max_transport_substeps': 64,
                 'max_report_rows': 4096, 'reserve_host_bytes': 64 << 20,
                 'reserve_device_bytes': 256 << 20}
        for f in fields(type(self)):
            if f.name in ('run_dir', 'safety_fraction'):
                continue
            value = _integer(getattr(self, f.name), f.name, 1)
            if f.name in fixed and value != fixed[f.name]:
                _refuse('fixed_limit_' + f.name)
        if type(self.safety_fraction) is not float or self.safety_fraction != 0.5:
            _refuse('original_half_safety')


@dataclass(frozen=True)
class SourceReference:
    """Root-issued immutable metadata/Source reference; no success Boolean."""
    path: str
    size: int
    sha256: str

    def validated(self):
        if (type(self.path) is not str or not 1 <= len(self.path) <= 4096
                or '\x00' in self.path or not Path(self.path).is_absolute() or '..' in Path(self.path).parts
                or type(self.sha256) is not str or len(self.sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.sha256)):
            _refuse('immutable_reference')
        _integer(self.size, 'reference_size', 1, 2 << 20)
        return self


@dataclass(frozen=True)
class ExternalRuntimeBound:
    """Positive selected-runtime bound for one *excluded* operation role.

    A Source projection never invents these allocator/compiler/library costs.
    Root must authenticate the complete finite operation/domain contract named
    by evidence; a constructor observation alone is not that contract.
    """
    role: str
    host_bytes: int
    device_bytes: int
    work: int
    logical_read_bytes: int
    h2d_bytes: int
    d2h_bytes: int
    scalar_reads: int
    evidence: SourceReference

    def validated(self):
        if type(self.role) is not str or self.role not in _EXTERNAL_ROLES:
            _refuse('external_role')
        for f in fields(type(self)):
            if f.name not in ('role', 'evidence'):
                # Every excluded role has a positive own/error frontier;
                # zero/unknown is not a release or allocator contract.
                _integer(getattr(self, f.name), 'external_' + f.name, 1)
        _holder(self.evidence, SourceReference).validated()
        return self


@dataclass(frozen=True, repr=False)
class DeviceWaterConstructorAccountingV2:
    """Opt-in finite typed constructor policy, not an observed heap/fit proof.

    source_census_sha256 is Root's attestation carried in the complete issued
    contract and primitive_evidence, not a separately read SourceReference.
    Root must join the full current/next/error population and selected runtime
    capacity laws. Unicode shell includes the fresh shell PLUS >=1 byte for a
    possible UTF8-cache terminator; character slope includes Unicode<=4 plus
    cachedUTF8<=4. Encoded bytes/writers and provider errors remain additional.
    Counts are occurrences/capacities, never measured peaks or alias credit.
    """
    schema: str
    source_census_sha256: str
    domain: tuple
    unicode_shell_bytes: int
    unicode_character_bytes: int
    bytes_shell_bytes: int
    bytes_payload_bytes: int
    list_header_bytes: int
    list_slot_bytes: int
    tuple_header_bytes: int
    tuple_slot_bytes: int
    dict_header_bytes: int
    dict_entry_bytes: int
    set_header_bytes: int
    set_entry_bytes: int
    banks: tuple

    def validated(self):
        _holder(self, DeviceWaterConstructorAccountingV2)
        if type(self.schema) is not str or self.schema != _CONSTRUCTOR_V2_SCHEMA:
            _refuse('constructor_v2_schema')
        if (type(self.source_census_sha256) is not str or len(self.source_census_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.source_census_sha256)):
            _refuse('constructor_v2_Root_census_attestation')
        if (type(self.domain) is not tuple or any(type(x) is not int for x in self.domain)
                or self.domain != _CONSTRUCTOR_V2_DOMAIN):
            _refuse('constructor_v2_reserved_domain')
        for name in _CONSTRUCTOR_V2_UNITS:
            # At least one positive fresh-shell byte plus the cache terminator;
            # the genuine selected fresh-shell value still requires Root proof.
            low = 8 if name == 'unicode_character_bytes' else 2 if name == 'unicode_shell_bytes' else 1
            _integer(getattr(self, name), 'constructor_v2_' + name, low, 1 << 30)
        if type(self.banks) is not tuple or len(self.banks) != len(_CONSTRUCTOR_V2_BANK_NAMES):
            _refuse('constructor_v2_complete_banks')
        for expected, row in zip(_CONSTRUCTOR_V2_BANK_NAMES, self.banks):
            if (type(row) is not tuple or len(row) != 1 + len(_CONSTRUCTOR_V2_COUNTS)
                    or type(row[0]) is not str or row[0] != expected):
                _refuse('constructor_v2_ordered_bank')
            for name, value in zip(_CONSTRUCTOR_V2_COUNTS, row[1:]):
                _integer(value, 'constructor_v2_' + name)
            if row[3] > row[1] or row[16] > row[2]:
                _refuse('constructor_v2_residual_exceeds_call_max')
            # Zero means a Source-proven absent family, never an unknown. A
            # payload/slot count cannot exist without its typed object owner.
            for objects, payload in ((4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)):
                if row[objects] == 0 and row[payload] != 0:
                    _refuse('constructor_v2_missing_typed_owner')
        return self


@dataclass(frozen=True)
class DeviceWaterRuntimeBounds:
    """Finite Root operands, absent until genuinely issued for this Source.

    primitive_units are typed constructor upper units, not sampled peaks.
    source_bindings are (canonical module name, immutable Source ref).  The
    external operation contracts cover provider/JIT/allocator/query/error
    banks which the real Native planners explicitly exclude.
    """
    native_source_identity: str
    runtime_evidence: SourceReference
    current_runtime_ref: SourceReference
    primitive_evidence: SourceReference
    primitive_units: tuple
    external_bounds: tuple
    source_bindings: tuple
    constructor_accounting_v2: DeviceWaterConstructorAccountingV2 | None = None

    def validated(self):
        _holder(self, DeviceWaterRuntimeBounds)
        if (type(self.native_source_identity) is not str or len(self.native_source_identity) != 64
                or any(c not in '0123456789abcdef' for c in self.native_source_identity)):
            _refuse('genuine_Root_native_source_identity')
        _holder(self.runtime_evidence, SourceReference).validated()
        _holder(self.current_runtime_ref, SourceReference).validated()
        _holder(self.primitive_evidence, SourceReference).validated()
        if (type(self.primitive_units) is not tuple or len(self.primitive_units) != len(_UNIT_NAMES)
                or tuple(row[0] for row in self.primitive_units if type(row) is tuple and len(row) == 2) != _UNIT_NAMES):
            _refuse('complete_primitive_units')
        for name, amount in self.primitive_units:
            _integer(amount, name, 1, 1 << 30)
        if type(self.external_bounds) is not tuple or len(self.external_bounds) != len(_EXTERNAL_ROLES):
            _refuse('complete_external_contracts')
        for expected, bound in zip(_EXTERNAL_ROLES, self.external_bounds):
            _holder(bound, ExternalRuntimeBound).validated()
            if bound.role != expected:
                _refuse('ordered_external_contracts')
        if type(self.source_bindings) is not tuple or not 1 <= len(self.source_bindings) <= 64:
            _refuse('source_binding_count')
        names = []
        for row in self.source_bindings:
            if type(row) is not tuple or len(row) != 2 or type(row[0]) is not str or len(row[0]) > 256:
                _refuse('source_binding_row')
            names.append(row[0])
            _holder(row[1], SourceReference).validated()
        if (sum(row[1].size for row in self.source_bindings) > 16 << 20
                or len(set(names)) != len(names) or set(names) != set(_SOURCE_MODULES)):
            _refuse('complete_source_roster')
        if self.constructor_accounting_v2 is not None:
            _holder(self.constructor_accounting_v2, DeviceWaterConstructorAccountingV2).validated()
        return self


@dataclass(frozen=True, kw_only=True)
class WindowedDeviceWaterContext(BedContext):
    native_solve_limits: object
    native_validation_limits: object
    native_commit_limits: object
    source_identity: str
    bridge_limits: PositiveWindowedDeviceWaterLimits
    runtime_bounds: DeviceWaterRuntimeBounds | None = None


@dataclass(frozen=True)
class DeviceWaterPhase:
    phase: str
    host_new: int
    device_new: int
    host_retained: int
    device_retained: int
    mapped: int
    logical_reads: int
    work: int
    h2d: int
    d2h: int
    scalar_reads: int
    writer_bytes: int
    writer_entries: int
    writer_versions: int
    terms: tuple


@dataclass(frozen=True)
class PositiveDeviceWaterAdmissionSummary:
    gate_count: int
    logical_work_debited: int
    logical_read_bytes_debited: int
    versions_reserved: int
    h2d_bytes_reserved: int
    d2h_bytes_reserved: int
    scalar_reads_reserved: int
    maximum_host_predicate: int
    maximum_device_predicate: int
    hydraulics: object


class _Owner:
    __slots__ = ('context', 'control', 'projection', 'inner_control', 'pid', 'thread',
                 'ordinal', 'work', 'reads', 'versions', 'h2d', 'd2h', 'scalars',
                 'max_host', 'max_device', 'poisoned', 'closed', 'hydraulics', 'source_refs', 'terminal_result', 'operators')

    def __init__(self, context, control, projection, source_refs):
        self.context, self.control, self.projection = context, control, projection
        self.inner_control = self.hydraulics = self.terminal_result = None
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.ordinal = self.work = self.reads = self.versions = 0
        self.h2d = self.d2h = self.scalars = self.max_host = self.max_device = 0
        self.poisoned = self.closed = False
        self.source_refs = source_refs
        self.operators = _operator_roster()



def _device_tag(context):
    # Canonical type-owned MRO access does not evaluate foreign instance
    # properties, descriptors or caller-defined __class__/__getattribute__.
    mro = type.__getattribute__(type(context), '__mro__')
    if any(kind is WindowedDeviceWaterContext or kind is PositiveWindowedDeviceWaterLimits for kind in mro):
        return True
    if type(context) is WindowedAvalancheContext:
        limits = object.__getattribute__(context, 'bridge_limits')
        return any(kind is PositiveWindowedDeviceWaterLimits
                   for kind in type.__getattribute__(type(limits), '__mro__'))
    return False


# Pure/idempotent: no imports, queries, Source scans, Native calls or counters.
def _positive_pair_guard(state, context):
    active = _OWNER
    selected = _device_tag(context)
    if selected or active is not None:
        if (type(state) is not WindowedSharedBedState or type(context) is not WindowedDeviceWaterContext
                or type(context.bridge_limits) is not PositiveWindowedDeviceWaterLimits
                or (active is not None and (active.context is not context
                    or active.pid != os.getpid() or active.thread != threading.get_ident()
                    or active.poisoned or active.closed))):
            _refuse('exact_active_process_thread_pair')


def is_shared_pair(state, context):
    selected = _device_tag(context)
    if selected:
        _positive_pair_guard(state, context)
    return selected


def _positive_context(context):
    if (type(context) is not WindowedDeviceWaterContext
            or type(context.bridge_limits) is not PositiveWindowedDeviceWaterLimits):
        _refuse('exact_device_context')
    return True


def _owner(context, control=None):
    active = _OWNER
    if (type(active) is not _Owner or active.context is not context or active.closed or active.poisoned
            or active.pid != os.getpid() or active.thread != threading.get_ident()):
        _refuse('owned_synchronous_entry_required')
    if _control_projection(context, active.control) != active.projection:
        _refuse('original_control_changed')
    if control is not None and control is not active.control:
        expected = tuple((n, False if n == 'force_final_commit' else v) for n, v in active.projection[0])
        if (control.storm is not active.control.storm
                or _control_projection(context, control) != (expected, active.projection[1])):
            _refuse('only_original_derived_force_control')
        if active.inner_control is None:
            active.inner_control = control
        elif active.inner_control is not control:
            _refuse('derived_control_identity')
    _check_code()
    _check_operators(active.operators)
    return active


def _positive_require_action(state, context):
    _positive_pair_guard(state, context)
    active = _owner(context)
    if active.ordinal < 1:
        _refuse('no_prior_admission')
    return active


def _native():
    from maple.surface.active_layer import bed_exchange_device as bed
    from maple.surface.avalanche import windowed_device as solve
    from maple.surface.avalanche import windowed_device_publication as avalanche
    from maple.surface.topographic_commit import windowed_device as commit
    from maple.surface.topographic_commit import windowed_commit
    from maple.surface.voxels import windowed_admission, windowed_inventory, windowed_pending
    return bed, solve, avalanche, commit, windowed_commit, windowed_admission, windowed_inventory, windowed_pending


def _control_projection(context, control):
    from maple_syrup.sediment_event import SedimentEventControl
    from maple_syrup.storm import StormControl
    _holder(control, SedimentEventControl)
    _holder(control.storm, StormControl)
    control.validated()
    if (control.storm.implementation != 'cuda' or control.storm.root_solver != 'bisection'
            or not 1 <= _integer(control.storm.bisection_iterations, 'bisection', 1) <= 200
            or type(control.storm.max_dt_s) is not float or control.storm.max_dt_s != 1.0
            or control.storm.max_steps != 10000 or control.storm.max_retries != 10
            or control.transport_scheme != 'upwind' or control.phase_storage != 'dense'
            or control.resolved_transport_implementation() != 'array'
            or control.phase_max_nnz is not None or control.max_transport_substeps != 64
            or control.max_commit_log != 256 or control.max_rejections_recorded != 100
            or control.cuda is not True or control.commit is not True
            or control.deep_publication != 'step' or control.record_exchange_history is not False):
        _refuse('original_cuda_bisection_upwind_step_profile')
    def rows(value, omit=()):
        result = []
        for f in fields(type(value)):
            if f.name in omit:
                continue
            x = getattr(value, f.name)
            if type(x) not in (type(None), bool, int, float, str):
                _refuse('scalar_control_projection')
            if type(x) is str and len(x) > 256:
                _refuse('control_text')
            result.append((f.name, x))
        return tuple(result)
    return rows(control, ('storm',)), rows(control.storm)



def _fixed_native_policies(context):
    """Exact existing Small native engineering declarations, no relaxation."""
    bed_owner, solve, *_ = _native()
    _holder(context.native_solve_limits.bed_limits, bed_owner.DeviceBedExchangeLimits)
    _holder(context.native_validation_limits.bed_limits, bed_owner.DeviceBedExchangeLimits)
    _holder(context.native_validation_limits.delta_budgets, solve.WindowedDeltaBudgets)
    def same(value, expected, nested=()):
        if {f.name for f in fields(type(value))} != {name for name, _ in expected} | set(nested):
            _refuse('complete_unchanged_Native_policy_fields')
        for name, literal in expected:
            actual = getattr(value, name)
            if type(actual) is not type(literal) or actual != literal:
                _refuse('unchanged_Native_policy_' + name)
    bed = (('max_operations', 2), ('max_gather_columns', 4), ('max_journal_records', 320),
        ('max_cells', 16), ('max_classes', 1), ('max_voxels', 6), ('max_impact_rows', 256),
        ('max_new_device_bytes', 64 << 20), ('max_host_staging_bytes', 64 << 20),
        ('max_logical_deep_read_bytes', 4 << 20), ('max_mapped_bytes', 4 << 20),
        ('device_reserve_bytes', 256 << 20), ('host_reserve_bytes', 64 << 20), ('device_safety_fraction', 0.5))
    same(context.native_solve_limits.bed_limits, bed)
    same(context.native_validation_limits.bed_limits, bed)
    same(context.native_solve_limits, (('max_iterations', 10000), ('max_snapshot_device_bytes', 64 << 20),
        ('max_total_logical_deep_read_bytes', 1 << 32), ('max_total_miss_batches', 1 << 24),
        ('max_total_h2d_bytes', 8 << 30), ('max_total_d2h_bytes', 8 << 30),
        ('boundary_host_bytes', 16 << 20), ('reserve_host_bytes', 64 << 20),
        ('device_reserve_bytes', 256 << 20), ('device_safety_fraction', 0.5)), ('bed_limits',))
    same(context.native_validation_limits, (('max_snapshot_device_bytes', 64 << 20),
        ('full_partition_host_bytes', 16 << 20), ('max_h2d_bytes', 64 << 20), ('max_d2h_bytes', 64 << 20),
        ('reserve_host_bytes', 64 << 20), ('device_reserve_bytes', 256 << 20), ('device_safety_fraction', 0.5)),
        ('bed_limits', 'delta_budgets'))
    same(context.native_validation_limits.delta_budgets, (('max_gathered_cells', 16),
        ('max_gather_columns', 4), ('max_journal_records', 320), ('max_deep_read_bytes', 2*16*6*8),
        ('max_gather_temp_bytes', 1 << 20), ('max_resident_temp_bytes', 4 << 20)))
    same(context.native_commit_limits, (('max_guard_device_bytes', 64 << 20), ('max_guard_host_bytes', 16 << 20),
        ('publication_max_records', 320), ('publication_encoding', 'full'), ('publication_overlay_max_bytes', None),
        ('reserve_host_bytes', 64 << 20), ('device_reserve_bytes', 256 << 20), ('device_safety_fraction', 0.5)))


def _profile(state, context):
    _positive_pair_guard(state, context)
    _holder(state, WindowedSharedBedState)
    _holder(context, WindowedDeviceWaterContext)
    limits = _holder(context.bridge_limits, PositiveWindowedDeviceWaterLimits)
    limits.__post_init__()
    from maple.core.parameters.geometry import GeometrySpec
    from maple.core.parameters.grain_classes import GrainClass, GrainClassSet
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.types.voxel import WindowedVoxelColumnState
    _holder(context.geometry, GeometrySpec)
    _holder(context.grain_classes, GrainClassSet)
    _holder(context.avalanching_spec, AvalanchingSpec)
    _holder(state.voxel_column, WindowedVoxelColumnState)
    g, col = context.geometry, state.voxel_column
    if ((g.ny, g.nx, col.n_voxels, col.deep_store.header.window_voxels) != (4, 4, 6, 3)
            or g.dx_m != g.dy_m or context.depth_update_rule != 'constant_depth'
            or context.record_exchange_history is not False or context.avalanching_spec.enabled is not True
            or context.deep_chunk_encoding != 'full' or context.deep_overlay_max_bytes is not None
            or type(context.window_max_journal_records) is not int or context.window_max_journal_records != 320
            or type(context.mass_resolution_kg) is not float or context.mass_resolution_kg <= 0):
        _refuse('Small_K1_constant_depth_enabled_avalanche_full_profile')
    if (type(context.source_identity) is not str or len(context.source_identity) != 64
            or any(c not in '0123456789abcdef' for c in context.source_identity)):
        _refuse('source_identity')
    classes = context.grain_classes.classes
    if type(classes) is not tuple or len(classes) != 1 or type(classes[0].class_id) is not str:
        _refuse('exact_K1_order')
    _holder(classes[0], GrainClass)
    ids = (classes[0].class_id,)
    bed, solve, avalanche, commit, *_ = _native()
    for value, kind in ((context.native_solve_limits, solve.DeviceWindowedAvalancheLimits),
                        (context.native_validation_limits, solve.DeviceWindowedAvalancheValidationLimits),
                        (context.native_commit_limits, commit.DeviceWindowedCommitLimits)):
        _holder(value, kind).__post_init__()
    if (type(context.avalanching_spec.max_iterations) is not int
            or context.native_solve_limits.max_iterations != context.avalanching_spec.max_iterations
            or context.native_solve_limits.bed_limits.max_gather_columns != 4
            or context.native_validation_limits.bed_limits.max_gather_columns != 4
            or context.native_solve_limits.bed_limits.max_journal_records != 320
            or context.native_validation_limits.bed_limits.max_journal_records != 320
            or context.native_commit_limits.publication_max_records != 320
            or context.native_commit_limits.publication_encoding != 'full'
            or context.native_commit_limits.publication_overlay_max_bytes is not None):
        _refuse('single_native_G4_J320_authority')
    _fixed_native_policies(context)
    fraction = context.initial_available_fraction
    if type(fraction) is not bed._cupy_ndarray():
        _refuse('exact_initial_device_fraction')
    import numpy as np
    admitted = bed._plan_device_windowed_bed_exchange(state.voxel_column, state.active_layer,
        context.geometry, class_ids=ids, limits=context.native_solve_limits.bed_limits)
    bed._device_array(fraction, 'initial_available_fraction', (4, 4, 1), np.float64,
                      admitted.device_id)
    if commit._api() != commit._ORIGINAL_API or avalanche._api() != avalanche._ORIGINAL_API:
        _refuse('native_current_operator_image')
    return ids, bed, solve, avalanche, commit


# The exact reached modules, not every installed library. Runtime/provider
# Source/compiled/origin qualification remains the original Root prerequisite.
_SOURCE_MODULES = (
    'maple_syrup.windowed_shared_bed_device', 'maple_syrup.windowed_shared_bed',
    'maple_syrup.sediment_bed', 'maple_syrup.sediment_event', 'maple_syrup.complete_event',
    'maple_syrup.hydrology_cuda', 'maple_syrup.event_backend', 'maple_syrup.routing_cuda',
    'maple_syrup.storm', 'maple_syrup.sediment_transport',
    'maple.surface.active_layer.bed_exchange_device', 'maple.surface.avalanche.windowed_device',
    'maple.surface.avalanche.windowed_device_publication', 'maple.surface.topographic_commit.windowed_device',
    'maple.surface.topographic_commit.windowed_commit', 'maple.surface.voxels.windowed_admission',
    'maple.surface.voxels.windowed_inventory', 'maple.surface.voxels.windowed_pending',
    'maple.water.windowed_step', 'maple.water.windowed_transaction', 'maple.water.windowed_commit_callback',
    'maple.surface.voxels.windowed_device_partition')


def _read_reference(ref):
    import hashlib
    import stat
    _holder(ref, SourceReference).validated()
    p = Path(ref.path)
    before = p.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_mode & 0o222
            or before.st_size != ref.size or p.is_symlink()):
        _refuse('immutable_reference_file')
    descriptor = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns):
            _refuse('opened_reference_identity')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            raw = stream.read(ref.size + 1)
        finished = os.fstat(descriptor)
        if (finished.st_dev, finished.st_ino, finished.st_size, finished.st_mtime_ns, finished.st_ctime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns):
            _refuse('descriptor_reference_currentness')
    finally:
        os.close(descriptor)
    after = p.lstat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or len(raw) != ref.size or hashlib.sha256(raw).hexdigest() != ref.sha256):
        _refuse('reference_current_size_sha')
    return raw



def _bootstrap_reference_frontier(context, bounds):
    """Before reference reads/origin resolution/Native plan constructors.

    Root's genuine metadata bootstrap must already have loaded this finite
    module roster; an admission call never cold-imports a provider as a proof.
    The original independent Root cold/image/origin guards remain mandatory.
    """
    if any(name not in sys.modules for name in _SOURCE_MODULES):
        _refuse('genuine_loaded_Source_bootstrap_required')
    limits = context.bridge_limits
    if type(limits) is not PositiveWindowedDeviceWaterLimits:
        _refuse('bootstrap_limits')
    limits.__post_init__()
    refs = (bounds.runtime_evidence, bounds.current_runtime_ref, bounds.primitive_evidence,
            *(b.evidence for b in bounds.external_bounds), *(r for _n, r in bounds.source_bindings))
    raw = sum(r.size for r in refs)
    # Raw bytes + decoded/path/hash/currentness/error lifetimes; two full
    # reference traversals remain additive. No source-only RSS/heap assertion.
    metadata = _constructor_projection(bounds, 'reference_bootstrap', rows=2*len(refs)*128+16384, arrays=512,
        text=4*raw+len(refs)*4*4096+16384*2048+2305, frames=18*5*_code_frontier()[1])
    banks = tuple(b for b in bounds.external_bounds if b.role in ('native_metadata', 'query'))
    host = 3*raw+metadata+3*sum(b.host_bytes for b in banks)+limits.reserve_host_bytes
    device = 3*sum(b.device_bytes for b in banks)+limits.reserve_device_bytes
    if host > limits.max_host_bytes or device > limits.max_device_bytes:
        _refuse('before_reference_constructor_ceiling')
    from maple.aeolian.cohorts.capacity import default_available_memory_bytes
    available = default_available_memory_bytes()
    if type(available) is not int or 2*host > available:
        _refuse('before_reference_HOST_headroom')
    bed = _native()[0]
    fraction = context.initial_available_fraction
    if type(fraction) is not bed._cupy_ndarray():
        _refuse('bootstrap_actual_device_fraction')
    bed._check_device_admission(device, device_id=int(fraction.device.id), safety_fraction=0.5)
    return raw+metadata+sum(b.work for b in banks)



def runtime_contract_bytes(bounds):
    """Pure before-write Root DTO encoder; no runtime observation or authority.

    The contract excludes its own containing reference (acyclic). Root first
    supplies genuine current-runtime/ABI/external/Source operands, computes
    these bytes, writes one immutable file and fills its exact size/SHA ref.
    Entry then compares the COMPLETE file to this exact scalar/ref projection;
    a genuine reference with changed incoming numeric values cannot pass.
    """
    import json
    _holder(bounds, DeviceWaterRuntimeBounds).validated()
    def ref(r):
        return {'path': r.path, 'size': r.size, 'sha256': r.sha256}
    payload = dict(schema='maple_syrup.windowed_device_water.runtime_contract.v1',
        native_source_identity=bounds.native_source_identity,
        current_runtime_ref=ref(bounds.current_runtime_ref), primitive_evidence=ref(bounds.primitive_evidence),
        primitive_units=bounds.primitive_units,
        external_bounds=tuple(dict(role=b.role, host_bytes=b.host_bytes, device_bytes=b.device_bytes,
            work=b.work, logical_read_bytes=b.logical_read_bytes, h2d_bytes=b.h2d_bytes,
            d2h_bytes=b.d2h_bytes, scalar_reads=b.scalar_reads, evidence=ref(b.evidence))
            for b in bounds.external_bounds),
        source_bindings=tuple(dict(module=name, **ref(r)) for name, r in bounds.source_bindings))
    if bounds.constructor_accounting_v2 is not None:
        profile = bounds.constructor_accounting_v2
        payload['schema'] = 'maple_syrup.windowed_device_water.runtime_contract.v2'
        payload['constructor_accounting_v2'] = {f.name: getattr(profile, f.name) for f in fields(type(profile))}
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
    if len(raw) > 2 << 20:
        _refuse('finite_runtime_contract_encoding')
    return raw


def _runtime_bounds(context, *, verify_files=False):
    bounds = context.runtime_bounds
    if type(bounds) is not DeviceWaterRuntimeBounds:
        _refuse('unissued_runtime_ABI_provider_contract')
    bounds.validated()
    if context.source_identity != bounds.native_source_identity:
        _refuse('context_Root_native_source_identity_join')
    if verify_files:
        _bootstrap_reference_frontier(context, bounds)
        for ref in (bounds.current_runtime_ref, bounds.primitive_evidence,
                    *(b.evidence for b in bounds.external_bounds)):
            _read_reference(ref)
        if _read_reference(bounds.runtime_evidence) != runtime_contract_bytes(bounds):
            _refuse('complete_issued_runtime_contract_value_join')
        for name, ref in bounds.source_bindings:
            module = sys.modules.get(name)
            origin = getattr(module, '__file__', None)
            if (type(origin) is not str or not origin or '\x00' in origin
                    or not Path(origin).is_absolute() or '..' in Path(origin).parts
                    or Path(origin).resolve() != Path(ref.path).resolve()):
                _refuse('actual_loaded_source_origin')
            _read_reference(ref)
    return bounds



def _operator_roster():
    """Immediate Python Source owners; Native retains its transitive images."""
    import types
    rows = []
    for name in _SOURCE_MODULES:
        module = sys.modules.get(name)
        if module is None or len(vars(module)) > 4096:
            _refuse('loaded_Source_module_domain')
        for key, value in vars(module).items():
            if type(value) is types.FunctionType and value.__module__ == name:
                rows.append((module, key, value, value.__code__))
            elif type(value) is type and value.__module__ == name:
                for member, fn in vars(value).items():
                    if type(fn) is types.FunctionType:
                        rows.append((value, member, fn, fn.__code__))
            if len(rows) > 8192:
                _refuse('Source_operator_capacity')
    return tuple(rows)


def _check_operators(rows):
    for parent, name, fn, code in rows:
        if vars(parent).get(name) is not fn or getattr(fn, '__code__', None) is not code:
            _refuse('current_Source_operator_identity')



def reserve_device_water_oracle_frontier(bounds, *, device_id):
    """Test-only before-genesis/copy envelope; no fixture/model operation.

    Root-issued Source projections cover the authentic native genesis, the
    independent Dense CuPy oracle and the bounded full output/Original/file
    observer. They are separate from production event fees, never free outer
    owners or an inferred amount from available memory. This returns only a
    reservation record and does not assert physical success.
    """
    _holder(bounds, DeviceWaterRuntimeBounds).validated()
    _integer(device_id, 'test_device_id', 0, 1023)
    banks = tuple(b for b in bounds.external_bounds if b.role in ('fixture', 'oracle', 'observer'))
    if len(banks) != 3:
        _refuse('complete_before_fixture_oracle_observer_banks')
    source_bytes = sum(ref.size for _name, ref in bounds.source_bindings)
    own = _constructor_projection(bounds, 'oracle_pregenesis', rows=16384, arrays=1024,
        text=source_bytes+4*4096*4096, frames=18*5*_code_frontier()[1])
    # The Original/file observer retains a complete bounded old store plus
    # current+next raw/readback generations, in addition to its typed bank.
    host = sum(b.host_bytes for b in banks)+3*(64 << 20)+own+(64 << 20)
    device = sum(b.device_bytes for b in banks)+(256 << 20)
    if host > 5 << 30 or device > 2 << 30:
        _refuse('oracle_original_hard_profile')
    from maple.aeolian.cohorts.capacity import default_available_memory_bytes
    available = default_available_memory_bytes()
    if type(available) is not int or 2*host > available:
        _refuse('oracle_HOST_headroom')
    _native()[0]._check_device_admission(device, device_id=device_id, safety_fraction=0.5)
    return DeviceWaterPhase('entry', host, device, 0, 0, 0,
        sum(b.logical_read_bytes for b in banks), sum(b.work for b in banks)+own,
        sum(b.h2d_bytes for b in banks), sum(b.d2h_bytes for b in banks),
        sum(b.scalar_reads for b in banks), 0, 0, 0,
        tuple((b.role, b.host_bytes+b.device_bytes) for b in banks))


def _constructor_frontier(bounds, *, rows, arrays, text, frames, errors=6):
    """Positive known constructors using genuine typed Root ABI upper units.

    Both returned and failed/current/new rows coexist. No AST-to-heap claim:
    frames counts complete reachable frame SLOTS supplied by the frozen code
    bank below; excluded provider frames are in their own runtime contracts.
    """
    u = dict(bounds.primitive_units)
    return (2 * rows * (u['mapping_header'] + 32 * (2*u['reference'] + u['scalar']))
            + 4 * arrays * (u['array_header'] + 8*u['reference'])
            + text * u['text_character'] + frames * u['frame_slot']
            + 4 * rows * u['sequence_header'] + errors * u['error_shell'])


def _constructor_frontier_v2(bounds, bank, *, rows, arrays, text, frames, errors=6):
    """Typed affine population arithmetic, with actual call-time caps.

    The reserved domain is Root-attested here; pregenesis has no physical state
    or context. Original state-bearing callers still perform genuine _profile.
    No census is generated or vendor/backend/fixture operation executed here.
    """
    _holder(bounds, DeviceWaterRuntimeBounds).validated()
    profile = bounds.constructor_accounting_v2
    if profile is None:
        _refuse('constructor_v2_required')
    if type(bank) is not str or bank not in _CONSTRUCTOR_V2_BANK_NAMES:
        _refuse('constructor_v2_bank')
    for name, value in (('rows', rows), ('arrays', arrays), ('text', text), ('frames', frames), ('errors', errors)):
        _integer(value, 'constructor_v2_call_' + name)
    record = profile.banks[_CONSTRUCTOR_V2_BANK_NAMES.index(bank)]
    (_name, call_rows_max, call_text_max, remaining_rows, unicode_objects, unicode_characters, byte_objects, byte_payload,
     list_objects, list_slots, tuple_objects, tuple_slots, dict_objects, dict_entries,
     set_objects, set_entries, remaining_text) = record
    if not (remaining_rows <= rows <= call_rows_max and remaining_text <= text <= call_text_max):
        _refuse('constructor_v2_current_call_caps')
    u = dict(bounds.primitive_units)
    total = (unicode_objects*profile.unicode_shell_bytes + unicode_characters*profile.unicode_character_bytes
        + byte_objects*profile.bytes_shell_bytes + byte_payload*profile.bytes_payload_bytes
        + list_objects*profile.list_header_bytes + list_slots*profile.list_slot_bytes
        + tuple_objects*profile.tuple_header_bytes + tuple_slots*profile.tuple_slot_bytes
        + dict_objects*profile.dict_header_bytes + dict_entries*profile.dict_entry_bytes
        + set_objects*profile.set_header_bytes + set_entries*profile.set_entry_bytes
        + remaining_text*u['text_character']
        + 2*remaining_rows*(u['mapping_header'] + 64*u['reference'])
        + 4*remaining_rows*u['sequence_header']
        + 4*arrays*(u['array_header'] + 8*u['reference'])
        + 2*rows*32*u['scalar'] + frames*u['frame_slot'] + errors*u['error_shell'])
    # Bounded inputs imply intermediate arithmetic <128bits. The resulting
    # logical frontier must still fit the unchanged plain-int63 counter domain.
    return _integer(total, 'constructor_v2_projection')


def _constructor_projection(bounds, bank, *, rows, arrays, text, frames, errors=6):
    """Literal four-site opt-in dispatch; unchanged original F for None."""
    if bounds.constructor_accounting_v2 is None:
        return _constructor_frontier(bounds, rows=rows, arrays=arrays, text=text, frames=frames, errors=errors)
    return _constructor_frontier_v2(bounds, bank, rows=rows, arrays=arrays, text=text, frames=frames, errors=errors)


def _runtime_bounds_repr_witness():
    """Only the selected standard dataclass RuntimeBounds repr wrapper.

    Its generated user_function is held in a closure, not co_consts. Preserve
    the real repr and bind that exact function/code/cells; no graph traversal,
    descriptor invocation or generated substitute supplies authority.
    """
    import types
    wrapper = vars(DeviceWaterRuntimeBounds).get('__repr__')
    if type(wrapper) is not types.FunctionType:
        _refuse('runtime_bounds_repr_wrapper')
    closure = wrapper.__closure__
    wrapped = getattr(wrapper, '__wrapped__', None)
    if (type(wrapped) is not types.FunctionType or type(closure) is not tuple or len(closure) != 2
            or wrapper.__code__.co_freevars != ('repr_running', 'user_function')
            or any(type(cell) is not types.CellType for cell in closure)):
        _refuse('runtime_bounds_repr_selected_closure')
    try:
        running, user_function = closure[0].cell_contents, closure[1].cell_contents
    except ValueError:
        _refuse('runtime_bounds_repr_empty_cell')
    if type(running) is not set or user_function is not wrapped:
        _refuse('runtime_bounds_repr_closure_binding')
    return wrapper, wrapper.__code__, closure, closure[0], closure[1], wrapped, wrapped.__code__, running


def _check_runtime_bounds_repr_witness():
    current = _runtime_bounds_repr_witness()
    if any(before is not after for before, after in zip(_RUNTIME_BOUNDS_REPR_WITNESS, current)):
        _refuse('runtime_bounds_repr_owner_changed')


def _code_frontier():
    import types
    # Complete own classes/functions including generated dataclass methods.
    slots = codes = text = 0
    seen = set()
    pending = list(_CODE_BANK)
    while pending:
        fn, code = pending.pop()
        if type(code) is not types.CodeType or id(code) in seen:
            continue
        seen.add(id(code)); codes += 1
        if codes > 4096:
            _refuse('code_frontier_nodes')
        slots += 1 + code.co_nlocals + code.co_stacksize + len(code.co_cellvars) + len(code.co_freevars)
        text += len(code.co_code) + len(code.co_exceptiontable) + len(code.co_linetable)
        for value in code.co_consts:
            if type(value) is types.CodeType:
                pending.append((None, value))
            elif type(value) in (str, bytes):
                text += len(value)
    return codes, slots, text


def _check_code():
    for namespace, name, fn, code in _OWN_BINDINGS:
        if namespace.get(name) is not fn or getattr(fn, '__code__', None) is not code:
            _refuse('adapter_function_or_method_rebound')
    for fn, code in _CODE_BANK:
        if getattr(fn, '__code__', None) is not code:
            _refuse('adapter_code_identity_changed')
    _check_runtime_bounds_repr_witness()


def _transaction_plan(state, context, demand=None):
    """The real Native PUBLIC plan, exact options, never paired savings."""
    ids, bed, *_ = _profile(state, context)
    column = state.voxel_column
    native = bed._plan_device_windowed_bed_exchange(column, state.active_layer, context.geometry,
        class_ids=ids, limits=context.native_solve_limits.bed_limits)
    if native.shape != (4, 4, 6, 3, 1) or len(ids) != 1:
        _refuse('exact_Native_Small_shape')
    from maple.surface.voxels.windowed_admission import windowed_transaction_bytes
    from maple.core.types.sediment_ledger import N_LEDGER_PROCESSES, SedimentLedgerState, CompactSedimentLedgerState
    ledger = state.ledger
    if type(ledger) is SedimentLedgerState:
        n_processes = N_LEDGER_PROCESSES
    elif type(ledger) is CompactSedimentLedgerState:
        n_processes = len(ledger.present_processes)
    else:
        _refuse('native_ledger_type')
    version = column.deep_store
    if version.header.n_chunks != 1 or len(version.chunks) != 1:
        _refuse('C16_native_min64_one_chunk_layout')
    plan = windowed_transaction_bytes(ny=4, nx=4, n_classes=1, n_voxels=6, window_voxels=3,
        packet_terms=native.packet_terms, base_itemsize=column.base.dtype.itemsize,
        ledger_present_processes=n_processes, has_availability=state.sediment_availability is not None,
        has_boundary_input=demand is None or demand.boundary_input_by_cell_class_kg is not None,
        has_boundary_export=demand is None or demand.boundary_export_by_cell_class_kg is not None,
        has_face_flux=demand is None or demand.face_flux is not None, max_gather_columns=4,
        max_journal_records=320, pending_records=native.carried_records,
        chunk_bytes_max=version.header.chunk_bytes_max,
        overlay_bytes_max=max((getattr(ref, 'overlay_nbytes', 0) for ref in version.chunks), default=0),
        mapping_cache_chunks=version.mapping_cache_chunks, namespace='cupy', path='public')
    if plan.unbounded_terms or plan.admissible is not True:
        _refuse('native_unbounded_water_plan')
    return native, plan


def produce_device_water_phase(state, context, phase, *, demand=None, external_host=0,
                               external_device=0, writer=False):
    """Concrete closed producer, before any physical/query/writer action.

    Native plans price Native arrays. Named CUDA constructors, full report/
    planner widths, current/prior/error trees, own Source/frame/constructor
    rows and separately authenticated excluded runtime banks add positively.
    No maximum is inferred from free memory or from an observed failure.
    """
    if type(phase) is not str or phase not in _PHASES or type(writer) is not bool:
        _refuse('phase_domain')
    _integer(external_host, 'external_host'); _integer(external_device, 'external_device')
    bounds = _runtime_bounds(context)
    ids, bed, solve, *_ = _profile(state, context)
    native, water = _transaction_plan(state, context, demand)
    solved = solve._plan_device_windowed_avalanche_solve(state.voxel_column, state.active_layer,
        context.geometry, context.avalanching_spec, context.mass_resolution_kg, class_ids=ids,
        limits=context.native_solve_limits, sediment_availability=state.sediment_availability,
        initial_available_fraction=context.initial_available_fraction)
    if (solved.shape != native.shape or solved.device_id != native.device_id
            or solved.bed_plan.shape != native.shape or solved.bed_plan.device_id != native.device_id):
        _refuse('solve_and_water_shape_device_join')
    # Complete actual allocating public water+full-I Native solve, never a
    # caller-owned adoption/donation discount. Validation/commit are retained
    # as full native ceilings until their genuine future result exists.
    validation = context.native_validation_limits
    commit = context.native_commit_limits
    c, k, v, w, p, j = 16, 1, 6, 3, native.packet_terms, 320
    endpoint = c * (8*w*k + 8 + 8*k + 2*8 + 8*k + 8*p + 8 + 8*w)
    # Same closed shapes as the genuine validation owner; include its whole
    # delta-resident/gather, two full partitions/mappings and byte snapshots.
    from maple.surface.avalanche.windowed_validation import _NATIVE_EXPRESSION_CREDITS
    credits = sum(n for _name, n in _NATIVE_EXPRESSION_CREDITS)
    validation_device = (4*endpoint + credits*8*c*k + 2*(3*k*8+c*k*8)
        + validation.delta_budgets.max_resident_temp_bytes
        + 2*bed._scratch('cheap_column_validator.cupy', c, v, w, k, p))
    validation_host = (2*validation.full_partition_host_bytes
        + validation.delta_budgets.max_resident_temp_bytes
        + validation.delta_budgets.max_gather_temp_bytes + 2*native.mapped_bytes_max)
    # Complete actual B2b Source families, beyond its snapshot-only ceilings:
    # current/output/compare/callback/later endpoint owners, all ledger/water/
    # committed/baseline/availability auxiliary leaves and native report rows.
    # The larger auxiliary logical ceiling remains separately charged below.
    auxiliary = 512*c*k*8
    report_bytes = (128+3*c)*k*8
    commit_device = (6*endpoint + 2*auxiliary + 2*report_bytes + 6*c*k*8 + 4*k*8
        + 4890*c*k+952*k+8 + 200*c*k+680*k+112
        + 8*(10*c+(2*w+25)*c*k)+4*c+17*c*k)
    commit_host = (native.carried_records+2*j)*(16+8*k)+2*3*3*8+2*c*k*8
    # Two contexts (current/prospective), two attempt/result owner generations,
    # and one error frontier. No claim that a retired allocator block is free.
    cuda_static = 58*c + 40*c + 8*(c+1)
    cuda_attempt = 161*c + 144
    host_prepare = 2*(58*c + 40*c + 8*(c+1)) + 8*c*6 + 64*c
    # Existing shared generic event/transport constructor census remains
    # unchanged. It overcovers the skipped array hydraulic branch; the new
    # CUDA static/attempt/provider frontiers are additional, not substitutions.
    # These are logical Source projections, not CPU AS or ABI certificates.
    from maple_syrup import windowed_shared_bed as generic
    generic_rows, generic_work = generic._positive_source_frontier()
    generic_payload, _generic_visits = generic._positive_interval_units()
    generic_host = sum(value for _name, value in generic_rows)
    generic_device = generic_payload*(generic._POSITIVE_GRID + generic._POSITIVE_FLAG)
    reports = 2*4096*(16+14)*8
    planner = 2*12*(8194*8) + 8*8194*(8+32)
    codes, slots, code_text = _code_frontier()
    operators = _OWNER.operators if type(_OWNER) is _Owner else _operator_roster()
    for _parent, _name, _fn, code in operators:
        slots += 1+code.co_nlocals+code.co_stacksize+len(code.co_cellvars)+len(code.co_freevars)
        code_text += len(code.co_code)+len(code.co_exceptiontable)+len(code.co_linetable)
        codes += 1
    source_bytes = sum(ref.size for _name, ref in bounds.source_bindings)
    constructor = _constructor_projection(bounds, 'phase', rows=4096+256+100+128,
        arrays=512, text=source_bytes+code_text+5*4096*4, frames=18*5*slots)
    # The existing event driver can take Q accepted steps, A attempts and
    # S transport substeps. Admission is for the WHOLE requested interval;
    # calls reserve independently before each Native mutation too.
    q, a, s = context.bridge_limits.max_event_steps, 11, 64
    event_calls = q*a if phase == 'event' else 1
    terms = [
        ('native_bed_exchange_host', native.new_host_bytes_peak),
        ('native_bed_exchange_device', native.new_device_bytes_peak),
        ('native_public_water_host', water.new_host_bytes_peak),
        ('native_public_water_device', water.new_device_bytes_peak),
        ('native_full_solve_host', solved.peak_host_bytes),
        ('native_full_solve_device', solved.peak_device_bytes),
        ('native_validation_host', validation_host),
        ('native_validation_device', validation_device),
        ('native_commit_full_host', commit_host + commit.max_guard_host_bytes),
        ('native_commit_full_device', commit_device + commit.max_guard_device_bytes),
        ('cuda_static_current_prospective_error', 3*cuda_static),
        ('cuda_attempt_current_next_error', 3*cuda_attempt),
        ('cuda_HOST_prepare_downloads_and_upload_sources', host_prepare),
        ('unchanged_generic_Source_HOST_projection', generic_host),
        ('unchanged_generic_Source_device_grid_projection', generic_device),
        ('complete_reports_and_public_planners', reports+planner),
        ('new_Source_constructor_frames_errors', constructor),
        ('canonical_HOST_journal_current_next_error', 3*j*(16+8*k)),
    ]
    host_new = (native.new_host_bytes_peak + water.new_host_bytes_peak + solved.peak_host_bytes + validation_host
        + commit_host + commit.max_guard_host_bytes + host_prepare + generic_host + reports + planner + constructor + 3*j*(16+8*k))
    device_new = (native.new_device_bytes_peak + water.new_device_bytes_peak + solved.peak_device_bytes + validation_device
        + commit_device + commit.max_guard_device_bytes + generic_device + 3*cuda_static + 3*cuda_attempt)
    required = dict(solved.required)
    water_calls = 2*q*a+1 if phase == 'event' else 1
    commit_calls = q+4096+1 if phase == 'event' else 1
    reads = (water_calls*(water.mapped_bytes_max+native.logical_deep_read_bytes)
             + commit_calls*required['max_total_logical_deep_read_bytes'])
    native_work = (water_calls*(native.new_host_bytes_peak+native.new_device_bytes_peak
                                +water.new_host_bytes_peak+water.new_device_bytes_peak)
        + commit_calls*solved.iterations*(solved.peak_host_bytes+solved.peak_device_bytes))
    work = source_bytes + code_text + slots + codes + reports + planner + constructor + generic_work + native_work
    h2d = water_calls*native.h2d_bytes_max + commit_calls*required['max_total_h2d_bytes'] + 3*cuda_static
    d2h = water_calls*native.d2h_bytes_max + commit_calls*required['max_total_d2h_bytes'] + host_prepare + 3*144
    scalars = 3*18
    for bound in bounds.external_bounds:
        if bound.role in ('fixture', 'oracle', 'observer'):
            continue  # Explicit whole-test frontier, retained separately by Root/test caller.
        # Transport has S launches per attempt; reporting/logging includes the
        # whole finite report domain; every other role occurs at least once.
        count = (event_calls*s if bound.role == 'transport' else event_calls
                 if bound.role in ('cuda_attempt', 'validation') else
                 q+4096+1 if phase == 'event' and bound.role in ('cuda_prepare', 'writer') else 1)
        host_new += 3*bound.host_bytes
        device_new += 3*bound.device_bytes
        work += count*bound.work
        reads += count*bound.logical_read_bytes
        h2d += count*bound.h2d_bytes
        d2h += count*bound.d2h_bytes
        scalars += count*bound.scalar_reads
        terms.append(('external_current_next_error_' + bound.role, 3*(bound.host_bytes + bound.device_bytes)))
    # Actual full backing occurrences; no deduplication/alias/free credit.
    retained_device = sum(row.backing_nbytes for row in native.arrays) + external_device
    outer = tuple(b for b in bounds.external_bounds if b.role in ('fixture', 'oracle', 'observer'))
    retained_host = (external_host + source_bytes + constructor
                     + 3*sum(b.host_bytes for b in outer) + 3*(64 << 20))
    retained_device += 3*sum(b.device_bytes for b in outer)
    mapped = water.mapped_bytes_max + native.mapped_bytes_max
    disk_bytes = disk_entries = versions = 0
    if writer:
        version = state.voxel_column.deep_store
        chunks = len(version.chunks)
        if chunks != 1 or version.header.n_chunks != 1 or version.version_id > (1 << 63)-3:
            _refuse('fixed_chunk_count_or_future_revision')
        # Full v1 chunk writes, next manifest and one complete partial at
        # widest int64 revision widths; all old/orphan files stay counted.
        manifest = 8192
        disk_bytes = 2*(version.header.logical_bytes + manifest)
        disk_entries = 2*(2*chunks+2)+2
        versions = 2
    result = DeviceWaterPhase(phase, host_new, device_new, retained_host, retained_device,
        mapped, reads, work, h2d, d2h, scalars, disk_bytes, disk_entries, versions, tuple(terms))
    for f in fields(DeviceWaterPhase):
        if f.name not in ('phase', 'terms'):
            _integer(getattr(result, f.name), f.name)
    return result

# Native graph HOST evidence is intentional even when graph.xp is CuPy.
_GRAPH_HOST = ('aspect', 'slope', 'slope_before_edge_rule', 'edge_rule_applied',
               'friction_factor', 'active', 'outlet', 'receiver', 'level',
               'level_order_host', 'pit_storage', 'flat_storage')
_NETWORK_HOST = ('receiver_host', 'outlet_host', 'active_host', 'terminal_host')
_SEDIMENT_HOST = ('diameter_m', 'raindrop_a', 'raindrop_b', 'raindrop_c',
    'raindrop_depth_attenuation_per_cm', 'raindrop_max_depth_per_reference_s_m',
    'settling_velocity_m_s', 'particle_mass_g')
_TERRAIN_HOST = ('initial_full_elevation_m', 'initial_committed_elevation_m',
                 'export_receiver_full', 'active_mask', 'friction_factor')


def _backing_census(roots, context):
    """Bounded hybrid metadata walk; opaque storage has a separate Native census.

    All occurrence/backing charges stay additive. HOST leaves are permitted
    only by exact owning field paths; physical payload cannot escape through a
    generic "host leaf" exception. Nothing is copied or converted here.
    """
    import dataclasses
    import types
    import numpy as np
    bed = _native()[0]
    cp_type = bed._cupy_ndarray()
    device_id = int(context.initial_available_fraction.device.id)
    host = device = nodes = leaves = 0
    if type(roots) is not tuple or len(roots) > 64:
        _refuse('closed_hybrid_root_count')
    pending = []
    expanded = 0
    def push(path, value, depth):
        nonlocal expanded
        if (type(path) is not str or len(path) > 2048 or depth > 12
                or expanded >= 16384 or len(pending) >= 16384):
            _refuse('before_append_hybrid_metadata_frontier')
        expanded += 1
        pending.append((path, value, depth))
    for name, value in roots:
        push(name, value, 0)
    while pending:
        path, value, depth = pending.pop()
        nodes += 1
        if nodes > 16384 or depth > 12 or len(path) > 2048:
            _refuse('hybrid_metadata_forest_domain')
        if value is None or type(value) in (bool, int, float, str, np.int64, np.float64, np.bool_):
            if type(value) is str and len(value) > 4096:
                _refuse('metadata_text_width')
            if type(value) is int and value.bit_length() > 128:
                _refuse('metadata_integer_width')
            continue
        if type(value) is types.ModuleType:
            if (path not in ('graph.xp', 'state.graph.xp', 'prior.state.graph.xp',
                    'state.network.xp', 'prior.state.network.xp', 'state.grid.xp', 'prior.state.grid.xp',
                    'column.xp', 'sediment.xp', 'field.xp')
                    or value is not sys.modules.get('cupy')):
                _refuse('only_actual_graph_namespace_module')
            continue
        if type(value) is np.ndarray or type(value) is cp_type:
            leaves += 1
            if leaves > 512 or value.ndim > 4 or value.nbytes > context.bridge_limits.max_external_backing_bytes:
                _refuse('external_array_domain')
            if type(value) is np.ndarray:
                prefix, _, member = path.rpartition('.')
                # Original plan_boundaries and SedimentEventResult.boundaries
                # are HOST report-clock metadata, including at finalize_event.
                # This does not permit a HOST physical payload or conversion.
                report_clock = path in ('planned_boundaries', 'prior.boundaries')
                if report_clock and (value.ndim != 1 or value.dtype != np.float64
                        or value.size > context.bridge_limits.max_report_rows):
                    _refuse('original_HOST_report_clock_domain')
                allowed = ((prefix in ('terrain', 'state.terrain', 'prior.state.terrain') and member in _TERRAIN_HOST)
                    or (prefix == 'pending_journal' and member in ('cell_flat', 'voxel', 'values_kg'))
                    or (prefix in ('graph', 'state.graph', 'prior.state.graph') and member in _GRAPH_HOST)
                    or (prefix in ('state.network', 'prior.state.network') and member in _NETWORK_HOST)
                    or (prefix == 'sediment' and member in _SEDIMENT_HOST) or report_clock)
                if not allowed or value.dtype.hasobject or not value.flags.c_contiguous:
                    _refuse('unexpected_HOST_payload_' + path)
                root = value
                for _ in range(8):
                    if root.base is None:
                        break
                    if type(root.base) is not np.ndarray:
                        _refuse('HOST_backing_owner')
                    root = root.base
                else:
                    _refuse('HOST_backing_chain')
                host += root.nbytes
            else:
                rows = []
                bed._device_array(value, path, tuple(value.shape), value.dtype, device_id, rows)
                device += rows[0].backing_nbytes
            if host + device > 4*context.bridge_limits.max_external_backing_bytes:
                _refuse('whole_external_backing_occurrences')
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            kind = type(value)
            schema = vars(kind).get('__dataclass_fields__') if type(kind) is type else None
            if (type(schema) is not dict or len(schema) > 128
                    or any(type(f) is not dataclasses.Field for f in schema.values())):
                _refuse('before_fields_external_holder_domain')
            names = tuple(f.name for f in fields(kind))
            if any(type(name) is not str or len(name) > 256 for name in names):
                _refuse('external_holder_field_name')
            try:
                actual = vars(value)
            except TypeError:
                _refuse('ordinary_external_holder_dictionary_required')
            if (type(actual) is not dict or len(actual) > 128
                    or any(type(name) is not str for name in actual) or set(actual) != set(names)):
                _refuse('external_holder_schema')
            for name in names:
                if name == 'bed' and path in ('state', 'prior.state'):
                    # A complete actual bed census is supplied separately.
                    continue
                push(path + '.' + name, actual[name], depth + 1)
        elif type(value) in (tuple, list):
            if len(value) > 8194:
                _refuse('metadata_sequence_width')
            for i, item in enumerate(value):
                push(path + '[' + str(i) + ']', item, depth + 1)
        elif type(value) is dict:
            if len(value) > 4096:
                _refuse('metadata_mapping_width')
            for name, item in value.items():
                if type(name) is not str or len(name) > 256:
                    _refuse('metadata_mapping_key')
                push(path + '.' + name, item, depth + 1)
        else:
            _refuse('unsupported_metadata_holder_' + path)
    return host, device, nodes, leaves


def _bed_roots(state, context):
    # The opaque column/pending journal and full backing allocations belong to
    # Native inventory. Every other physical field is traversed explicitly.
    roots = tuple(('physical.' + name, getattr(state, name)) for name in _BED_FIELDS
                  if name not in ('voxel_column', 'active_layer')) + (
                      ('initial_available_fraction', context.initial_available_fraction),)
    pending = state.voxel_column.pending_deep
    if pending is not None:
        from maple.surface.voxels.windowed_pending import PendingDeepBinding
        from maple.surface.voxels.deep_store import DeepJournal
        if type(pending) is not PendingDeepBinding or type(pending.journal) is not DeepJournal:
            _refuse('genuine_HOST_pending_journal')
        roots += tuple(('pending_journal.' + name, getattr(pending.journal, name))
                       for name in ('cell_flat', 'voxel', 'values_kg'))
    return roots


def _geometry(context, graph, terrain):
    from maple_syrup.routing import RoutingGraph
    from maple_syrup.sediment_bed import TerrainReference
    _holder(graph, RoutingGraph)
    _holder(terrain, TerrainReference)
    if (graph.shape != (4, 4) or graph.xp is not sys.modules.get('cupy')
            or graph.dx_m != context.geometry.dx_m or terrain.dx_m != graph.dx_m
            or terrain.initial_committed_elevation_m.shape != graph.shape
            or terrain.allow_masked_nodata or graph.max_level_width > 16):
        _refuse('actual_C16_CuPy_unmasked_geometry')


def _inventory_disk(context, store):
    import stat
    limits = context.bridge_limits
    root = Path(limits.run_dir)
    if (not root.is_absolute() or '..' in root.parts or not store.is_absolute()
            or '..' in store.parts or not store.is_relative_to(root)):
        _refuse('owned_run_store_paths')
    for path in (root, store, *store.parents):
        if not stat.S_ISDIR(path.lstat().st_mode):
            _refuse('real_directory_ancestors')
    total = entries = store_bytes = 0
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        if depth > 16:
            _refuse('disk_depth')
        with os.scandir(directory) as iterator:
            for entry in iterator:
                if entries >= limits.max_entries or len(entry.name) > 255 or len(entry.path) > 4096:
                    _refuse('disk_metadata_domain')
                entries += 1
                st = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(st.st_mode):
                    _integer(st.st_size, 'actual_file_bytes')
                    total += st.st_size
                    if Path(entry.path).is_relative_to(store):
                        store_bytes += st.st_size
                    if total > limits.max_run_bytes or store_bytes > limits.max_store_bytes:
                        _refuse('retained_disk_ceiling')
                elif stat.S_ISDIR(st.st_mode):
                    if len(stack) >= limits.max_entries:
                        _refuse('disk_stack')
                    stack.append((Path(entry.path), depth + 1))
                else:
                    _refuse('disk_symlink_or_special')
    return total, store_bytes, entries


def _gate(state, context, phase, *, roots=(), demand=None, writer=False):
    """Reserve before the action; original Native guards repeat independently."""
    _positive_pair_guard(state, context)
    active = None if _OWNER is None else _owner(context)
    if active is None and phase not in ('initial', 'validate'):
        _refuse('entry_required_before_action')
    bounds = _runtime_bounds(context, verify_files=active is None)
    # This bounded complete metadata frontier is admitted BEFORE its walk;
    # the later action gate adds all Native/physical/current-next terms.
    _bootstrap_reference_frontier(context, bounds)
    _profile(state, context)
    external_host, external_device, _nodes, _leaves = _backing_census(
        _bed_roots(state, context) + roots, context)
    plan = produce_device_water_phase(state, context, phase, demand=demand,
        external_host=external_host, external_device=external_device, writer=writer)
    bounds, limits = _runtime_bounds(context), context.bridge_limits
    # Construction of the census/phase/currentness/error records is additive;
    # it does not disappear merely because the logical result is small.
    own = _constructor_projection(bounds, 'metadata_gate', rows=16384+128, arrays=512+64,
        text=16384*2048+2305+4*4096+256*128, frames=18*5*_code_frontier()[1])
    host = plan.host_new + plan.host_retained + plan.mapped + own + limits.reserve_host_bytes
    device = plan.device_new + plan.device_retained + limits.reserve_device_bytes
    if (host > limits.max_host_bytes or device > limits.max_device_bytes
            or plan.mapped > limits.max_mapped_bytes or plan.work + own > limits.max_work
            or plan.logical_reads > limits.max_logical_read_bytes):
        _refuse('prospective_logical_resource_ceiling')
    from maple.aeolian.cohorts.capacity import default_available_memory_bytes
    available = default_available_memory_bytes()
    if type(available) is not int or host*2 > available:
        _refuse('genuine_HOST_half_headroom')
    ids, bed, *_ = _profile(state, context)
    native = bed._plan_device_windowed_bed_exchange(state.voxel_column, state.active_layer,
        context.geometry, class_ids=ids, limits=context.native_solve_limits.bed_limits)
    bed._check_device_admission(device, device_id=native.device_id,
                               safety_fraction=limits.safety_fraction)
    if writer:
        total, store, entries = _inventory_disk(context, state.voxel_column.deep_store.store_dir)
        if (total + plan.writer_bytes > limits.max_run_bytes
                or store + plan.writer_bytes > limits.max_store_bytes
                or entries + plan.writer_entries > limits.max_entries):
            _refuse('before_writer_current_next_partial_disk')
    if active is not None:
        work = active.work + plan.work + own
        reads = active.reads + plan.logical_reads
        versions = active.versions + plan.writer_versions
        if (work > limits.max_work or reads > limits.max_logical_read_bytes
                or versions > limits.max_new_versions):
            _refuse('monotone_cumulative_resource_ceiling')
        # Planned transfer/query counters have the closed int63 DTO domain;
        # real Native per-solve caps and the original worker monitor remain
        # independent. No new measured-traffic authority or relaxed cap.
        h2d = _integer(active.h2d + plan.h2d, 'cumulative_h2d')
        d2h = _integer(active.d2h + plan.d2h, 'cumulative_d2h')
        scalars = _integer(active.scalars + plan.scalar_reads, 'cumulative_scalar_reads')
        active.work, active.reads, active.versions = work, reads, versions
        active.h2d, active.d2h, active.scalars = h2d, d2h, scalars
        active.max_host, active.max_device = max(active.max_host, host), max(active.max_device, device)
        active.ordinal += 1
    _check_code()
    _profile(state, context)  # Real Source/owner recheck after both queries.
    return plan


def _validate(state, context):
    ids, *_ = _profile(state, context)
    from maple.surface.voxels import windowed_device_partition as partition
    limits = context.native_validation_limits
    plan = partition._plan_device_pending_partition(state.voxel_column, state.active_layer,
        context.geometry, context.mass_resolution_kg, class_ids=ids, bed_limits=limits.bed_limits,
        max_host_bytes=limits.full_partition_host_bytes,
        max_snapshot_device_bytes=limits.max_snapshot_device_bytes)
    return partition._validate_device_pending_partition(state.voxel_column, state.active_layer,
        context.geometry, context.mass_resolution_kg, class_ids=ids, plan=plan)


def validate_shared_bed(state, context, *, current_time_s, published):
    _time(current_time_s)
    if type(published) is not bool:
        _refuse('publication_marker_type')
    _gate(state, context, 'validate')
    record = _validate(state, context)
    if published and state.voxel_column.pending_deep is not None:
        _refuse('published_endpoint_still_pending')
    return record


def admit_initial(bed, context, graph, terrain, soil_water_m, current_time_s, control):
    _time(current_time_s)
    _control_projection(context, control)
    _geometry(context, graph, terrain)
    _runtime_bounds(context, verify_files=True)
    result = _gate(bed, context, 'initial', roots=(('graph', graph), ('terrain', terrain),
                                                   ('soil_water_m', soil_water_m)))
    _validate(bed, context)
    return result


def admit_event(state, context, control, **operands):
    from maple_syrup.sediment_event import SedimentEventState
    _holder(state, SedimentEventState)
    active = _owner(context, control)
    _time(state.t_s)
    _geometry(context, state.graph, state.terrain)
    end = operands.get('end_s')
    if end is not None and (type(end) is not float or not math.isfinite(end) or end <= state.t_s):
        _refuse('event_end')
    if operands.get('max_report_rows') not in (None, 4096):
        _refuse('report_domain')
    from maple_syrup import complete_event as completion
    from maple_syrup import sediment_event as event
    caller = sys._getframe(2).f_code
    allowed = ((completion.complete_event, 'event'), (completion.finalize_event, 'terminal'),
               (event._check_event_inputs, 'attempt'), (event._sediment_coupled_core, 'attempt'),
               (event.evolve_sediment_event, 'event'))
    phase = next((role for fn, role in allowed if fn.__code__ is caller), None)
    if phase is None:
        _refuse('exact_original_event_caller')
    previous = operands.get('continuation')
    if previous is not None:
        if (caller is not completion.finalize_event.__code__
                or type(previous) is not event.SedimentEventResult or previous.state is not state):
            _refuse('only_original_terminal_result_continuation')
        if active.terminal_result is None:
            active.terminal_result = previous
        elif active.terminal_result is not previous:
            _refuse('terminal_result_identity')
    roots = [('state', state)]
    for name in ('column', 'sediment', 'field', 'schedule', 'vegetation', 'planned_boundaries', 'rain_rate', 'zero_demand'):
        value = operands.get(name)
        if value is not None:
            roots.append((name, value))
    if previous is not None:
        roots.append(('prior', previous))
        roots.extend(_bed_roots(previous.state.bed, context))
    # Native CudaHydraulics retains and checks its real graph/column binding;
    # the genuine event driver alone decides when to prepare/adopt a reroute.
    return _gate(state.bed, context, phase, roots=tuple(roots))


def admit_water_operands(state, context, *, depth_m=None, mobile_kg=None, demand=None):
    _positive_require_action(state, context)
    from maple.water.interfaces import WaterProcessDemand
    if demand is not None:
        _holder(demand, WaterProcessDemand)
        if demand.hydraulic_diagnostics is not None:
            _refuse('array_only_native_demand')
    import numpy as np
    bed = _native()[0]
    device_id = int(state.voxel_column.window_kg.device.id)
    if demand is None:
        if depth_m is not None:
            bed._device_array(depth_m, 'water.depth_m', (4, 4), np.float64, device_id)
        mobile = state.water.mobile_mass_by_cell_class_kg if mobile_kg is None else mobile_kg
        bed._device_array(mobile, 'water.mobile_mass_by_cell_class_kg', (4, 4, 1), np.float64, device_id)
    else:
        for name in ('requested_removal_by_cell_class_kg', 'requested_deposition_by_cell_class_kg',
                     'boundary_input_by_cell_class_kg', 'boundary_export_by_cell_class_kg'):
            array = getattr(demand, name)
            if array is None and name.startswith('boundary_'):
                continue
            bed._device_array(array, name, (4, 4, 1), np.float64, device_id)
        if demand.face_flux is not None:
            from maple.aeolian.flux.face_flux import FaceFluxResult
            _holder(demand.face_flux, FaceFluxResult)
            for name in ('x_face_crossing_mass_kg', 'y_face_crossing_mass_kg',
                         'x_face_net_crossing_mass_kg', 'y_face_net_crossing_mass_kg'):
                shape = (4, 5, 1) if name.startswith('x_') else (5, 4, 1)
                bed._device_array(getattr(demand.face_flux, name), name, shape, np.float64, device_id)
    roots = (('depth_m', depth_m), ('mobile_kg', mobile_kg), ('demand', demand))
    return _gate(state, context, 'water', roots=roots, demand=demand)


def commit_shared_bed(state, context, graph, reference, *, current_time_s, force, routing_source_m):
    _positive_require_action(state, context)
    _time(current_time_s)
    if type(force) is not bool or routing_source_m is None:
        _refuse('explicit_force_and_retained_routing_source')
    _geometry(context, graph, reference)
    roots = (('graph', graph), ('terrain', reference), ('routing_source_m', routing_source_m))
    _gate(state, context, 'trigger', roots=roots)
    from maple_syrup.sediment_bed import check_routing_source, reroute_threshold_m
    from maple.surface.topographic_commit.triggers import evaluate_commit_triggers
    check_routing_source(routing_source_m, graph.shape, graph.xp)
    threshold = reroute_threshold_m(context)
    trigger = evaluate_commit_triggers(state.ledger, state.committed_topography, context.geometry,
                                      current_time_s, context.topographic_commit_spec)
    if not force and not trigger.should_commit:
        return None, trigger, None, None, threshold
    # B2b accepts the genuine current HOST journal and publishes once itself.
    # There is no CPU-lane prepublication or invented empty-journal marker.
    _gate(state, context, 'commit', roots=roots, writer=True)
    ids, _bed, _solve, avalanche, native_commit = _profile(state, context)
    from maple.core.backend import to_bool
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.water.windowed_commit_callback import make_windowed_water_depth_commit_callback
    callback, availability = avalanche._make_device_windowed_avalanche_callback(
        context.avalanching_spec, class_ids=ids, solve_limits=context.native_solve_limits,
        validation_limits=context.native_validation_limits,
        sediment_availability=state.sediment_availability,
        initial_available_fraction=context.initial_available_fraction)
    elevation = committed_elevation_m(state.committed_topography)
    water_callback, water_box = make_windowed_water_depth_commit_callback(
        state.water, context.geometry, elevation, depth_update_rule='constant_depth', trusted_packet=True)
    result = native_commit._stage_device_windowed_topography_with_native_avalanche(
        state.ledger, state.committed_topography, state.voxel_column, state.active_layer,
        context.geometry, context.mass_resolution_kg, current_time_s,
        avalanche_callback=callback, water_callback=water_callback, water_box=water_box, water=state.water,
        committed_elevation_before_m=elevation, limits=context.native_commit_limits,
        reconciliation_baseline=state.reconciliation_baseline, record_exchange_history=False)
    xp = sys.modules.get('cupy')
    if xp is None:
        _refuse('actual_CuPy_namespace_required')
    update, water = water_box.depth_update, water_box.water
    if (not water_box.ran or update.rule != 'constant_depth'
            or water.depth_m is not update.new_depth_m or water.depth_m is state.water.depth_m
            or water.mobile_mass_by_cell_class_kg is not state.water.mobile_mass_by_cell_class_kg
            or update.clamp_cell_count != 0 or update.clamp_added_volume_m3 != 0.0
            or update.displaced_volume_m3 != 0.0
            or not to_bool(xp.all(water.depth_m == state.water.depth_m))):
        _refuse('genuine_constant_depth_copy_mobile_identity_zero_clamp')
    candidate = replace(state, voxel_column=result.new_voxel_column, active_layer=result.new_active_layer,
        ledger=result.new_ledger, committed_topography=result.new_committed_state,
        sediment_availability=availability.new_sediment_availability, water=water,
        reconciliation_baseline=result.new_reconciliation_baseline)
    # Native B2b completed full physical/delta/partition validation and one
    # publication before adopting its boxes. Additional boundary verification
    # retains current and candidate and independently reserves its readback.
    _gate(candidate, context, 'validate', roots=_bed_roots(state, context))
    _validate(candidate, context)
    return candidate, trigger, result, water_box, threshold


def publish_shared_bed(state, context, *, current_time_s):
    _positive_require_action(state, context)
    _time(current_time_s)
    _gate(state, context, 'publish', writer=True)
    _validate(state, context)
    column = _native()[7].publish_pending_column(state.voxel_column, context.geometry,
        context.mass_resolution_kg, max_records=320, encoding='full', overlay_max_bytes=None)
    candidate = replace(state, voxel_column=column)
    if column.pending_deep is not None or candidate.reconciliation_baseline is not state.reconciliation_baseline:
        _refuse('publication_binding')
    _gate(candidate, context, 'validate', roots=_bed_roots(state, context))
    _validate(candidate, context)
    return candidate


def storage_reference_time(state):
    time = state.committed_topography.last_commit_time_s
    if state.reconciliation_baseline is not None:
        time = max(time, state.reconciliation_baseline.last_reconciliation_time_s)
    return _time(time)


def shared_bed_already_committed(state, context, current_time_s):
    _positive_require_action(state, context)
    _time(current_time_s)
    if state.voxel_column.pending_deep is not None:
        return False
    _gate(state, context, 'validate')
    _validate(state, context)
    return _native()[4].is_windowed_already_committed(state.ledger, state.committed_topography,
        state.voxel_column, state.active_layer, context.geometry, context.mass_resolution_kg,
        current_time_s, reconciliation_baseline=state.reconciliation_baseline)


def require_terminal_pair(wet, dry, context, *, current_time_s):
    _positive_require_action(wet, context)
    _positive_pair_guard(dry, context)
    for name in _BED_FIELDS:
        if name != 'water' and getattr(wet, name) is not getattr(dry, name):
            _refuse('terminal_exact_physical_references')
    for state in (wet, dry):
        _gate(state, context, 'terminal')
        _validate(state, context)
    _time(current_time_s)


def splice_terminal_shared_beds(accepted, dry, context, *, current_time_s):
    _positive_require_action(accepted, context)
    _positive_pair_guard(dry, context)
    candidate = replace(dry, voxel_column=accepted.voxel_column,
                        reconciliation_baseline=accepted.reconciliation_baseline)
    require_terminal_pair(accepted, candidate, context, current_time_s=current_time_s)
    if accepted.voxel_column.pending_deep is not None:
        _refuse('terminal_published_endpoint_required')
    return candidate


def complete_positive_windowed_device_event(state, context, column, field, schedule, vegetation, sediment,
        *, max_end_s, control, policy, report_every_s):
    """One original complete_event with ONE real caller-owned CudaHydraulics.

    Complete Source/runtime/provider/cold/origin/UUID/grant prerequisites are
    Root's original execution controls, not replaced by this entry.  Runtime
    unit/evidence absence refuses.  Native errors stay errors; no retry or
    disposal API is invented. Authority closes in finally, allocations do not.
    """
    global _OWNER
    from maple_syrup import windowed_shared_bed as cpu
    from maple_syrup.complete_event import CompletionPolicy, DryHandoff, complete_event
    from maple_syrup.sediment_event import CudaHydraulics
    if _OWNER is not None:
        _refuse('nonreentrant_device_operation')
    _positive_pair_guard(state.bed, context)
    cpu._positive_pair_guard(state.bed, context)  # Genuine CPU exclusion before owner construction.
    from maple_syrup.sediment_event import SedimentEventState
    _holder(state, SedimentEventState)
    _profile(state.bed, context)
    projection = _control_projection(context, control)
    _holder(policy, CompletionPolicy).validated()
    if policy.deposit_mobile_at_flow_end is not True:
        _refuse('original_deposit_mobile_completion_policy')
    if (type(max_end_s) is not float or not math.isfinite(max_end_s) or max_end_s <= state.t_s
            or type(report_every_s) is not float or report_every_s <= 0):
        _refuse('explicit_finite_event_domain')
    _runtime_bounds(context, verify_files=True)
    owner = _Owner(context, control, projection, context.runtime_bounds.source_bindings)
    _OWNER = owner
    try:
        # Full prospective context/preparation/attempt/terminal/frontier BEFORE
        # the real lifecycle constructor; complete_event repeats real checks.
        _gate(state.bed, context, 'entry', roots=(('state', state), ('column', column),
            ('field', field), ('schedule', schedule), ('vegetation', vegetation), ('sediment', sediment)))
        owner.hydraulics = CudaHydraulics(mode='auto', max_log=256)
        result = complete_event(state, context, column, field, schedule, vegetation, sediment,
            max_end_s=max_end_s, control=control, policy=policy, report_every_s=report_every_s,
            max_report_rows=4096, continuation=None, checkpoint_callback=None, hydraulics=owner.hydraulics)
        if type(result) is not DryHandoff or result.status != 'complete':
            _refuse('actual_complete_dry_handoff_required')
        _owner(context)
        summary = PositiveDeviceWaterAdmissionSummary(owner.ordinal, owner.work, owner.reads,
            owner.versions, owner.h2d, owner.d2h, owner.scalars, owner.max_host, owner.max_device,
            owner.hydraulics.summary())
        return result, summary
    except BaseException:
        owner.poisoned = True
        raise
    finally:
        owner.closed = True
        if _OWNER is owner:
            _OWNER = None
        # No Native close exists. Keep current/prospective/error/traceback
        # lifetimes charged and let original worker cleanup own the process.


# Immutable function/code witness; includes dataclass-generated methods and
# every own helper. No selected runtime/ABI fact is manufactured here.
_RUNTIME_BOUNDS_REPR_WITNESS = _runtime_bounds_repr_witness()
_CODE_BANK = tuple((fn, fn.__code__)
    for value in tuple(globals().values())
    for fn in ((value,) if callable(value) and hasattr(value, '__code__') else
               tuple(vars(value).values()) if isinstance(value, type) and value.__module__ == __name__ else ())
    if callable(fn) and hasattr(fn, '__code__'))
_CODE_BANK += ((_RUNTIME_BOUNDS_REPR_WITNESS[5], _RUNTIME_BOUNDS_REPR_WITNESS[6]),)

_OWN_BINDINGS = tuple(
    (namespace, name, fn, fn.__code__)
    for namespace in (globals(), *(vars(value) for value in tuple(globals().values())
        if type(value) is type and value.__module__ == __name__))
    for name, fn in tuple(namespace.items())
    if callable(fn) and getattr(fn, '__module__', None) == __name__ and hasattr(fn, '__code__'))
