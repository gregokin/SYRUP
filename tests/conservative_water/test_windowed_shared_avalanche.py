"""Synthetic K1 D2 interfaces; Root executes under the locked CPU controls.

Both physical controls are independently initialized through Native operators.
Test-only dense materialization is an oracle, never the storage handoff.  No
production calibration, K6 wet, restart, GPU or large-domain claim is made.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from maple.core.backend.depth_class_reductions import qualify_depth_class_order
from maple.core.boundaries import AxisBoundary, BoundaryKind
from maple.core.parameters.geometry import GeometrySpec
from maple.core.parameters.grain_classes import GrainClass, GrainClassSet
from maple.core.parameters.topographic_commit import TopographicCommitSpec
from maple.core.types.sediment_availability import SedimentAvailabilityState
from maple.core.types.sediment_ledger import process_index, zeros_sediment_ledger_state
from maple.core.types.water import WaterState
from maple.surface.avalanche import callback as native_avalanche
from maple.surface.avalanche import windowed_publication as publication_core
from maple.surface.topographic_commit import closed_windowed_commit as closed
from maple.surface.topographic_commit import commit as native_commit
from maple.surface.topographic_commit import initialization, windowed_commit
from maple.surface.voxels import windowed_inventory
from maple.water import WaterProcessDemand
from maple.water.commit_callback import make_water_depth_commit_callback
from tests.unit.test_avalanche_transfer import _build_state
from tests.windowed_wind.test_w2_exchange_seam import same as _same_leaf
from tests.windowed_wind.test_w3_native_partition_maintenance import evidence_limits, limits
from tests.windowed_wind.test_w4_avalanche_publication import (
    publication_policy, solve_policy, spec, validation_policy,
)
from tests.windowed_wind.test_w4_closed_native_commit import _bounded_cpu_limits

from maple_syrup import checkpoint
from maple_syrup.routing import LEGACY_FRICTION_FLOOR, build_routing_graph
from maple_syrup.sediment_bed import (
    BedContext, BedIntegrationError, BedState, _shared_profile, apply_bed_demand,
    bed_already_committed, commit_bed, publish_bed_storage, terrain_reference, with_water,
)
from maple_syrup.sediment_event import SedimentEventControl, initial_event_state, sediment_coupled_step
from maple_syrup.storm import StormControl
from maple_syrup.windowed_shared_bed import (
    WindowedAvalancheContext, WindowedSharedBedState, WindowedWaterBridgeLimits,
    admit_event, require_terminal_pair, splice_terminal_shared_beds, validate_shared_bed,
)


def same(a, b, path="result"):
    """Compare diagnostic mappings recursively; keep exact native leaf checks."""
    assert type(a) is type(b), path
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            same(a[key], b[key], path + "." + str(key))
    elif dataclasses.is_dataclass(a) and not isinstance(a, type):
        for field in dataclasses.fields(a):
            same(getattr(a, field.name), getattr(b, field.name), path + "." + field.name)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "[" + str(index) + "]")
    else:
        _same_leaf(a, b, path)


def _limits(run_dir):
    # Finite aggregate logical counters: A2, R+1=2, M=A+1=3,
    # metadata calls=17+5*A*(R+1)+2*A+4*M=53.  Native policy domains
    # and original hardware caps are unchanged.  One TiB here is a
    # Source-checked upper for this tiny fixture's visible/operator visits,
    # not retained payload or permission to exceed CPU/wall/AS.
    native = publication_policy()
    meta, commits, attempts, deep_bytes = 53, 3, 4, 2 * 6 * 1 * 8
    return WindowedWaterBridgeLimits(max_host_bytes=4 << 30, max_mapped_bytes=128 << 20,
        max_logical_read_bytes=(commits * native.max_logical_deep_read_bytes
            + (2 * attempts + 2 * commits) * (4 << 20) + (meta + 8) * deep_bytes),
        max_work=(meta + commits) * native.max_work + (1 << 40),
        max_store_bytes=64 << 20, max_run_bytes=128 << 20, max_new_versions=16,
        max_entries=4096, max_external_backing_bytes=64 << 20,
        max_event_steps=2, max_retries=1, max_transport_substeps=64,
        max_report_rows=2, reserve_host_bytes=64 << 20, run_dir=run_dir)


def _files(root):
    result = {}
    for path in root.rglob('*'):
        if path.is_file():
            assert len(result) < 4096
            result[str(path.relative_to(root))] = path.read_bytes()
    return result


def _bits(state, context):
    inputs = closed._preflight_original_metadata(state.voxel_column, state.active_layer,
        state.ledger, state.committed_topography, context.geometry,
        tuple(c.class_id for c in context.grain_classes.classes), context.avalanching_spec,
        context.native_iteration_limits, context.native_evidence_limits,
        context.native_solve_limits, context.native_validation_limits,
        context.native_publication_limits, reconciliation_baseline=state.reconciliation_baseline,
        availability=state.sediment_availability, initial_available_fraction=context.initial_available_fraction)
    return tuple((leaf.array, leaf.array.tobytes()) for leaf in inputs.leaves) + tuple(
        (getattr(state.water, name), getattr(state.water, name).tobytes())
        for name in ('depth_m', 'mobile_mass_by_cell_class_kg'))


def _unchanged(bits):
    for array, raw in bits:
        assert array.tobytes() == raw



def _validation_work_context(state, context):
    """The actual owning metadata law; only the fixture work declaration changes."""
    column, active, geometry = state.voxel_column, state.active_layer, context.geometry
    ids = tuple(grain.class_id for grain in context.grain_classes.classes)
    iteration, evidence = context.native_iteration_limits, context.native_evidence_limits
    shape, _prefix, _pending, rows = publication_core.bed._inventory(
        column, active, geometry, ids, iteration)
    availability_rows = publication_core._check_availability(shape,
        state.sediment_availability, context.initial_available_fraction)
    _backings, caller_bytes = publication_core._caller_backings(tuple(rows), availability_rows)
    entry = publication_core.a0._plan_native_partition_domain_entry(
        column, active, geometry, context.mass_resolution_kg, class_ids=ids,
        limits=iteration, evidence_limits=evidence)
    required = publication_core._validation_requirements_upper(shape, caller_bytes, entry,
        availability_present=state.sediment_availability is not None)
    work = max(context.native_validation_limits.max_work, required['max_work'])
    assert type(work) is int and 0 <= work <= context.native_publication_limits.max_work
    return replace(context, native_validation_limits=replace(
        context.native_validation_limits, max_work=work))


def _fixture(tmp_path, *, mobile=0.000125, flat=False):
    """Square cells, fixed V6/W3 and genuine occupied nonresident sediment.

    These are explicitly synthetic interface declarations.  Native geometry,
    initializer, avalanche law/tolerances/I10000 and water law are unmodified.
    The Dense oracle is a second independent call, never the windowed result.
    """
    geo = GeometrySpec(nx=2, ny=1, dx_m=0.01, dy_m=0.01, voxel_dz_m=0.1,
        bulk_density_kg_m3=1000.0, active_layer_thickness_m=0.28,
        boundary_x=AxisBoundary(kind=BoundaryKind.PRESCRIBED, inflow_flux_kg_m_s={}),
        boundary_y=AxisBoundary(kind=BoundaryKind.PRESCRIBED, inflow_flux_kg_m_s={}))
    z = np.full((1, 2), 0.6) if flat else np.array([[0.6, 0.0]], np.float64)
    dense, active = _build_state(geo, z, n_voxels=6)
    oracle, oracle_active = _build_state(geo, z.copy(), n_voxels=6)
    assert not np.shares_memory(dense.mass_kg, oracle.mass_kg)
    assert not np.shares_memory(active.mass_kg, oracle_active.mass_kg)
    ids = ('synthetic_c0',)
    column = windowed_inventory.windowed_column_from_dense(dense, geo, 1e-10,
        store_dir=tmp_path / 'store', class_ids=ids, max_store_bytes=64 << 20,
        window_voxels=3, chunk_columns=64, with_prefix_packets=True)
    assert np.any(column.deep_total_kg > 0.0)  # Occupied deep material after actual active extraction.
    initial = initialization.initial_committed_topography_state_from_physical_state(dense, active, geo, 1e-10)
    committed, _regrouping = windowed_commit.windowed_committed_state_from_dense_baseline(
        initial, column, active, geo)
    # Ordinary Native baseline factory over genuine initialized physical refs.
    baseline = native_commit.ledger_reconciliation_baseline(
        committed.committed_active_layer_mass_by_class_kg,
        committed.committed_voxel_mass_by_class_kg,
        last_reconciliation_time_s=committed.last_commit_time_s, reconciliation_count=0)
    fraction = np.full_like(active.mass_kg, 0.5)
    availability = SedimentAvailabilityState(active.mass_kg * fraction, active.mass_kg * (1.0 - fraction))
    ledger = zeros_sediment_ledger_state(geo.ny, geo.nx, 1)
    assert process_index('water_erosion_deposition') in ledger.present_processes
    assert process_index('avalanching') in ledger.present_processes
    water = WaterState(np.full((1, 2), 0.0125), np.full((1, 2, 1), mobile))
    bed = WindowedSharedBedState(column, active, water, ledger, committed, availability, baseline)
    native = SimpleNamespace(geo=geo, dense=dense, layer=active, windowed=column, ids=ids, mres=1e-10,
        availability=availability, fraction=fraction, ledger=ledger)
    iteration = replace(limits(native), max_gather_columns=2)
    context = WindowedAvalancheContext(geo, GrainClassSet((GrainClass(ids[0], 0.00025, 2650.0, False),)),
        1e-10, TopographicCommitSpec(), spec(), fraction,
        window_max_journal_records=iteration.max_journal_records,
        native_iteration_limits=iteration, native_evidence_limits=evidence_limits(),
        native_solve_limits=solve_policy(), native_validation_limits=validation_policy(),
        native_publication_limits=publication_policy(),
        native_qualification_limits=_bounded_cpu_limits(max_leaf_occurrences=36,
            max_backing_nodes=36 * 8, max_sum_rank=36 * 8 * 4, max_frame_roles=12, max_error_roles=6),
        source_identity='a' * 64, bridge_limits=_limits(tmp_path))
    # Genuine existing cold/order procedure: no forged held kernel or bool.
    qualify_depth_class_order(6, 1)
    context = _validation_work_context(bed, context)
    full = np.full((3, 4), -1.0)
    # Labelled synthetic ring: high N/S, low E/W open export cells.
    full[0, :] = full[-1, :] = 1.0
    full[1:-1, 1:-1] = windowed_commit.committed_elevation_m(committed)
    exports = np.ones((3, 4), bool)
    exports[1:-1, 1:-1] = False
    graph = build_routing_graph(full, exports, np.full((1, 2), LEGACY_FRICTION_FLOOR), 0.01, xp=np)
    terrain = terrain_reference(full, exports, bed, graph)
    source = windowed_commit.committed_elevation_m(committed)
    return SimpleNamespace(bed=bed, context=context, graph=graph, terrain=terrain, source=source,
        dense=oracle, active=oracle_active, original_committed=initial, native=native)


def _run(f, *, force=True, time=7.25):
    return commit_bed(f.bed, f.context, f.graph, f.terrain, time,
                      force=force, routing_source_m=f.source)


def test_first_shared_native_avalanche_commit_matches_independent_dense(tmp_path):
    f = _fixture(tmp_path)
    original = _bits(f.bed, f.context)
    answer = _run(f)
    assert answer.did_commit and answer.maple_result.avalanche_applied
    assert type(answer.water_result) is closed._ClosedWaterReturn
    assert answer.water_result.result is answer.maple_result
    assert answer.state.reconciliation_baseline is answer.maple_result.new_reconciliation_baseline
    assert answer.state.reconciliation_baseline is not None
    assert answer.state.reconciliation_baseline.reconciliation_count == f.bed.reconciliation_baseline.reconciliation_count + 1
    assert answer.state.water is answer.water_result.water
    assert answer.water_result.depth_update.displaced_volume_m3 == 0.0
    assert answer.water_result.depth_update.clamp_added_volume_m3 == 0.0
    assert answer.water_result.depth_update.clamp_cell_count == 0
    assert answer.state.water.depth_m.tobytes() == f.bed.water.depth_m.tobytes()
    assert answer.maple_result.committed_height_provenance['avalanche_phases']['status'] == 'validated'
    # Actual positive Native activity, not merely a declared enabled flag.
    assert np.any(answer.maple_result.pre_commit_process_totals_kg[process_index('avalanching')] != 0.0)
    oracle_committed = initialization.initial_committed_topography_state_from_physical_state(
        f.dense, f.active, f.context.geometry, f.context.mass_resolution_kg)
    oracle_baseline = native_commit.ledger_reconciliation_baseline(
        oracle_committed.committed_active_layer_mass_by_class_kg,
        oracle_committed.committed_voxel_mass_by_class_kg,
        last_reconciliation_time_s=oracle_committed.last_commit_time_s, reconciliation_count=0)
    oracle_water = WaterState(f.bed.water.depth_m.copy(), f.bed.water.mobile_mass_by_cell_class_kg.copy())
    av, avbox = native_avalanche.make_availability_aware_avalanche_callback(
        f.context.avalanching_spec, f.bed.sediment_availability, f.context.initial_available_fraction)
    wc, box = make_water_depth_commit_callback(oracle_water, f.context.geometry,
        windowed_commit.committed_elevation_m(oracle_committed), depth_update_rule='constant_depth')
    expected = native_commit.commit_topography(f.bed.ledger, oracle_committed, f.dense, f.active,
        f.context.geometry, f.context.mass_resolution_kg, 7.25, avalanche_callback=av,
        water_callback=wc, reconciliation_baseline=oracle_baseline)
    box.publish()
    # Native endpoint helper carries the existing Native comparison contract.
    from tests.windowed_wind.test_w2_exchange_seam import endpoint

    with pytest.MonkeyPatch.context() as patch:
        endpoint(f.native, SimpleNamespace(new_column=answer.state.voxel_column,
            new_active_layer=answer.state.active_layer), expected.new_voxel_column,
            expected.new_active_layer, patch)
    for name in ('new_ledger', 'new_active_layer', 'pre_commit_process_totals_kg',
                 'avalanche_diagnostics', 'new_reconciliation_baseline'):
        same(getattr(answer.maple_result, name), getattr(expected, name), name)
    same(answer.state.water, box.water, 'water')
    same(answer.state.sediment_availability, avbox.new_sediment_availability, 'availability')
    _unchanged(original)


def test_no_trigger_keeps_exact_graph_bed_and_published_files(tmp_path):
    f = _fixture(tmp_path, flat=True)
    files, bits = _files(tmp_path), _bits(f.bed, f.context)
    answer = _run(f, force=False, time=0.0)
    assert answer.did_commit is False and answer.state is f.bed and answer.graph is f.graph
    assert answer.maple_result is None and answer.water_result is None
    assert _files(tmp_path) == files
    _unchanged(bits)


def test_same_actual_seventh_field_survives_water_and_real_demand(tmp_path):
    f = _fixture(tmp_path)
    changed_water = with_water(f.bed, f.context, f.bed.water.depth_m.copy())
    assert type(changed_water) is WindowedSharedBedState
    assert changed_water.reconciliation_baseline is f.bed.reconciliation_baseline
    request = f.bed.active_layer.mass_kg * 0.001
    changed, actual = apply_bed_demand(changed_water, f.context,
        WaterProcessDemand(request, np.zeros_like(request)))
    assert np.any(actual.actual_removal_by_cell_class_kg > 0.0)
    assert type(changed) is WindowedSharedBedState
    assert changed.reconciliation_baseline is f.bed.reconciliation_baseline
    assert changed.ledger is not f.bed.ledger



@pytest.mark.parametrize('entry', ['water', 'demand'])
def test_direct_water_entries_refuse_nonprimitive_inputs_without_publication(tmp_path, entry):
    f = _fixture(tmp_path)
    original, files = _bits(f.bed, f.context), _files(tmp_path)
    backing = np.zeros(2, dtype=[('value', '<f8')])
    if entry == 'water':
        with pytest.raises((ValueError, closed._ClosedCommitMetadataRefusal)):
            with_water(f.bed, f.context, backing.view(np.float64).reshape(1, 2))
    else:
        request = backing.view(np.float64).reshape(1, 2, 1)
        with pytest.raises((ValueError, closed._ClosedCommitMetadataRefusal)):
            apply_bed_demand(f.bed, f.context, WaterProcessDemand(request, np.zeros_like(request)))
    _unchanged(original)
    assert _files(tmp_path) == files


def test_water_batch_must_be_covered_by_actual_native_storage_plan(tmp_path):
    f = _fixture(tmp_path)
    context = replace(f.context, native_iteration_limits=replace(
        f.context.native_iteration_limits, max_gather_columns=1))
    original, files = _bits(f.bed, context), _files(tmp_path)
    with pytest.raises(BedIntegrationError, match='native_water_batch_not_covered'):
        with_water(f.bed, context, f.bed.water.depth_m)
    _unchanged(original)
    assert _files(tmp_path) == files


def test_actual_publication_and_joint_terminal_splice_keep_own_water(tmp_path):
    f = _fixture(tmp_path, mobile=0.0)
    result = _run(f)
    dry = with_water(result.state, f.context, np.zeros_like(result.state.water.depth_m))
    require_terminal_pair(result.state, dry, f.context, current_time_s=7.25)
    accepted = publish_bed_storage(result.state, f.context)
    assert accepted.voxel_column.pending_deep is None
    spliced = splice_terminal_shared_beds(accepted, dry, f.context, current_time_s=7.25)
    assert spliced.voxel_column is accepted.voxel_column
    assert spliced.reconciliation_baseline is accepted.reconciliation_baseline is not None
    assert spliced.water is dry.water and spliced.water is not accepted.water
    assert not np.any(spliced.water.depth_m) and not np.any(spliced.water.mobile_mass_by_cell_class_kg)
    assert bed_already_committed(accepted, f.context, 7.25)


@pytest.mark.parametrize('bad', ['water', 'history', 'deferred', 'baseline_time', 'host', 'dtype'])
def test_unsupported_inputs_refuse_before_publication_and_preserve_original(tmp_path, bad):
    f = _fixture(tmp_path)
    original, files = _bits(f.bed, f.context), _files(tmp_path)
    bed, context = f.bed, f.context
    if bad == 'water':
        bed = replace(bed, water=None)
    elif bad == 'history':
        context = replace(context, record_exchange_history=True)
    elif bad == 'deferred':
        context = replace(context, deep_chunk_encoding='overlay')
    elif bad == 'baseline_time':
        bed = replace(bed, reconciliation_baseline=replace(bed.reconciliation_baseline,
            last_reconciliation_time_s=8.0))
    elif bad == 'host':
        context = replace(context, bridge_limits=replace(context.bridge_limits, max_host_bytes=1))
    else:
        backing = np.zeros(2, dtype=[('value', '<f8')])
        water = replace(bed.water, depth_m=backing.view(np.float64).reshape(1, 2))
        bed = replace(bed, water=water)
    with pytest.raises((ValueError, closed._ClosedCommitMetadataRefusal)):
        commit_bed(bed, context, f.graph, f.terrain, 7.25, force=True, routing_source_m=f.source)
    _unchanged(original)
    assert _files(tmp_path) == files


def test_missing_water_channel_is_not_silently_expanded(tmp_path):
    f = _fixture(tmp_path)
    ledger = zeros_sediment_ledger_state(1, 2, 1, present_processes=(process_index('avalanching'),))
    bad = replace(f.bed, ledger=ledger)
    with pytest.raises(BedIntegrationError, match='both_native_process_channels_required'):
        validate_shared_bed(bad, f.context, current_time_s=7.25, published=True)
    assert bad.ledger is ledger and ledger.present_processes == (process_index('avalanching'),)


def test_real_current_clock_is_not_replaced_by_storage_anchor(tmp_path):
    f = _fixture(tmp_path)
    bad = replace(f.bed, reconciliation_baseline=replace(f.bed.reconciliation_baseline,
        last_reconciliation_time_s=8.0))
    files = _files(tmp_path)
    with pytest.raises(ValueError, match='FUTURE'):
        commit_bed(bad, f.context, f.graph, f.terrain, 7.25, force=True, routing_source_m=f.source)
    assert _files(tmp_path) == files


def test_pair_and_default_subclass_contracts_are_distinct():
    @dataclass(frozen=True)
    class HistoricalBed(BedState):
        old_extension: int = 1

    old = BedState(None, None, None, None, None, None)
    context = BedContext(None, None, 1.0, None, None, None)
    assert _shared_profile(old, context) is None
    assert _shared_profile(HistoricalBed(None, None, None, None, None, None), context) is None
    new = WindowedSharedBedState(None, None, None, None, None, None, None)
    with pytest.raises(BedIntegrationError, match='exact_state_context_pair'):
        _shared_profile(new, context)
    assert tuple(field.name for field in fields(WindowedSharedBedState))[-1] == 'reconciliation_baseline'
    for name in ('native_iteration_limits', 'source_identity', 'bridge_limits'):
        assert WindowedAvalancheContext.__dataclass_fields__[name].kw_only


def test_early_control_refusal_precedes_water_event_work(tmp_path):
    f = _fixture(tmp_path)
    control = SedimentEventControl(storm=StormControl(max_steps=2, max_retries=1),
                                   record_exchange_history=True)
    files = _files(tmp_path)
    with pytest.raises(BedIntegrationError, match='step_publication_cpu_no_history'):
        initial_event_state(f.graph, f.terrain, f.bed, f.context, np.zeros((1, 2)), control=control)
    assert _files(tmp_path) == files


def test_new_family_save_load_and_both_wire_modes_refuse_before_io(tmp_path):
    f = _fixture(tmp_path)
    target = tmp_path / 'must_not_exist'
    with pytest.raises(BedIntegrationError, match='checkpoint_codec_unqualified'):
        checkpoint.save_checkpoint(target, object(), f.context, object(), object(), {})
    with pytest.raises(BedIntegrationError, match='checkpoint_codec_unqualified'):
        checkpoint.load_checkpoint(target, f.context, object(), object(), {})
    for mode in (checkpoint._WRITE_WIRE, checkpoint._LAYOUT_WIRE):
        arrays = {}
        with pytest.raises(BedIntegrationError, match='checkpoint_codec_unqualified'):
            checkpoint._encode_wire(f.bed, None, arrays, {}, None, None, _mode=mode)
        assert arrays == {}
    assert not target.exists()


def test_actual_accepted_water_step_keeps_native_baseline_and_soil(tmp_path):
    from maple_syrup.infiltration import column_parameters
    from maple_syrup.sediment_physics import sediment_physics_parameters

    f = _fixture(tmp_path, mobile=0.0, flat=True)
    control = SedimentEventControl(storm=StormControl(max_steps=2, max_retries=1), phase_bins=8)
    state = initial_event_state(f.graph, f.terrain, f.bed, f.context, np.full((1, 2), 1e-5),
                                t_s=0.0, control=control)
    ones, zeros = np.ones((1, 2)), np.zeros((1, 2))
    column = column_parameters(model='fixed_ksat', ksat_m_per_s=zeros,
        suction_m=zeros, drainage_parameter=zeros, theta_sat=ones * 0.5, soil_thickness_m=ones)
    sediment = sediment_physics_parameters(diameter_m=np.array([0.00025]),
        raindrop_a=np.array([0.0]), raindrop_b=np.array([1.0]), raindrop_c=np.array([0.0]),
        raindrop_max_depth_mm=np.array([0.0]), particle_density_g_cm3=2.65,
        active_layer_sensitivity_mm=1.0, ke_model='wainwright_log', xp=np)
    # These are synthetic input declarations, not a changed Native response.
    actual = sediment_coupled_step(state, f.context, column, zeros, zeros, sediment,
                                   1e-6, control)
    assert actual.storm.state.t_s > state.t_s
    assert actual.bed.reconciliation_baseline is f.bed.reconciliation_baseline
    assert type(actual.bed) is WindowedSharedBedState
    assert np.array_equal(state.storm.soil_water_m, np.full((1, 2), 1e-5))


@pytest.mark.parametrize('kind', ['hydraulics', 'root', 'transport', 'storage'])
def test_unqualified_solver_implementations_refuse_before_event_work(tmp_path, kind):
    f = _fixture(tmp_path)
    control = SedimentEventControl(storm=StormControl(max_steps=2, max_retries=1))
    if kind == 'hydraulics':
        control = replace(control, storm=replace(control.storm, implementation='numba'))
    elif kind == 'root':
        control = replace(control, storm=replace(control.storm, root_solver='newton'))
    elif kind == 'transport':
        control = replace(control, transport_implementation='numba')
    else:
        control = replace(control, phase_storage='csr')
    before, bits = _files(tmp_path), _bits(f.bed, f.context)
    with pytest.raises(BedIntegrationError, match='array_bisection_dense_transport_profile'):
        initial_event_state(f.graph, f.terrain, f.bed, f.context,
                            np.zeros((1, 2)), control=control)
    assert _files(tmp_path) == before
    _unchanged(bits)


def test_real_complete_event_joint_splice_preserves_native_default_dry_reset(tmp_path):
    from maple_syrup.complete_event import CompletionPolicy, DryHandoff, complete_event
    from maple_syrup.infiltration import column_parameters
    from maple_syrup.rainfall import constant_rainfall, rainfall_field
    from maple_syrup.sediment_physics import sediment_physics_parameters

    f = _fixture(tmp_path, mobile=0.0, flat=True)
    # Zero-domain terminal/reset interface.  The Native default 60s quiet
    # policy is unchanged; two existing 60s steps cover it after rain-off.
    # This is separate from the positive moving-avalanche/Dense oracle above.
    control = SedimentEventControl(storm=StormControl(max_dt_s=60.0, max_steps=2, max_retries=1),
                                   phase_bins=8)
    bed = with_water(f.bed, f.context, np.zeros((1, 2)))
    soil = np.full((1, 2), 1e-5)
    state = initial_event_state(f.graph, f.terrain, bed, f.context, soil, t_s=0.0, control=control)
    ones, zeros = np.ones((1, 2)), np.zeros((1, 2))
    column = column_parameters(model='fixed_ksat', ksat_m_per_s=zeros,
        suction_m=zeros, drainage_parameter=zeros, theta_sat=ones * 0.5, soil_thickness_m=ones)
    sediment = sediment_physics_parameters(diameter_m=np.array([0.00025]),
        raindrop_a=np.array([0.0]), raindrop_b=np.array([1.0]), raindrop_c=np.array([0.0]),
        raindrop_max_depth_mm=np.array([0.0]), particle_density_g_cm3=2.65,
        active_layer_sensitivity_mm=1.0, ke_model='wainwright_log', xp=np)
    field = rainfall_field(1, 2, xp=np)
    schedule = constant_rainfall(0.0, 60.0, 0.0)
    bits = _bits(bed, f.context)
    actual = complete_event(state, f.context, column, field, schedule, zeros, sediment,
        max_end_s=120.0, control=control, policy=CompletionPolicy(),
        report_every_s=60.0, max_report_rows=2)
    assert type(actual) is DryHandoff and actual.status == 'complete'
    assert actual.progress.policy == CompletionPolicy()
    assert actual.progress.result.state.t_s == actual.dry_state.t_s == 120.0
    assert actual.dry_state.bed.voxel_column is actual.progress.result.state.bed.voxel_column
    assert actual.dry_state.bed.reconciliation_baseline is actual.progress.result.state.bed.reconciliation_baseline
    assert actual.dry_state.bed.reconciliation_baseline is not None
    assert actual.dry_state.bed.water is not actual.progress.result.state.bed.water
    assert not np.any(actual.dry_state.bed.water.depth_m)
    assert not np.any(actual.dry_state.bed.water.mobile_mass_by_cell_class_kg)
    assert not np.any(actual.dry_state.storm.soil_water_m)
    assert np.array_equal(actual.pre_reset_soil_m, soil)
    assert actual.accounting['policy'] == 'external dry-again removal; no evapotranspiration simulated'
    assert actual.accounting['storm_water_budget']['closed']
    _unchanged(bits)
