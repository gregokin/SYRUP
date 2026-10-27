"""ONE genuine Small K1 complete device event vs independent Dense CuPy.

Author did not import/run this file. Root must first accept StageA wet Native
commit and StageB direct+paired exchanges, then issue the separate qualification
budget/grant and retain original Source/provider/device controls. All11 external
operation bounds and primitive/constructor allocation terms remain opaque/None.
External MASY fixture imports are real Source dependencies, never fake modules.
No forced journal, fixture mass injection, clipping or new comparison epsilon.
"""
from __future__ import annotations

import dataclasses

import fixture_dense_alternation as legacy
import fixture_dense_production as production
import numpy as np
import pytest

from tests.windowed_wind_gpu.test_device_reduced_step import _device_fixture, cupy  # noqa: F401
from tests.windowed_wind_gpu.test_device_avalanche import _solve_limits, _validation_limits
from tests.windowed_wind_gpu.test_device_commit import _commit_limits
from maple.core.backend import arrays_share_memory, to_bool, to_device, to_device_tree, to_host
from maple.core.types.topographic_commit import committed_elevation_m
from maple.surface.topographic_commit.windowed_commit import windowed_committed_state_from_dense_baseline
from maple.surface.voxels.windowed_inventory import (
    column_mass_by_class_kg, gather_full_columns, resident_reduction_bound_by_cell_class_kg,
)
from maple_syrup import windowed_shared_bed_device as device
from maple_syrup.complete_event import CompletionPolicy, DryHandoff, complete_event
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule
from maple_syrup.sediment_bed import BedState, bed_by_cell_class, bed_inventory, check_graph_evidence_matches, routing_graph_from_elevation, terrain_reference
from maple_syrup.sediment_event import CudaHydraulics, SedimentEventResult, initial_event_state
from maple_syrup.windowed_shared_bed import WindowedSharedBedState

pytestmark = pytest.mark.gpu
_TIMING = ('first_step_wall_s', 'first_step_cpu_s', 'remaining_wall_s', 'remaining_cpu_s', 'commit_wall_s')
_RESULT_SEPARATE = ('state', 'initial_bed_by_cell_class_kg', 'initial_bed_inventory_kg', 'commit_log')
_ACCOUNTING_OWN = ('terminal_sediment_residual_kg', 'terminal_sediment_tolerance_kg', 'sediment_closure')
_CLOSURE_OWN = ('initial_bed_kg', 'final_bed_kg', 'net_bed_change_kg', 'residual_kg', 'tolerance_kg')


def _same(a, b, path):
    assert type(a) is type(b), path
    if isinstance(a, (np.ndarray,)) or type(a) is getattr(__import__('sys').modules.get('cupy'), 'ndarray', None):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert np.ascontiguousarray(to_host(a)).tobytes() == np.ascontiguousarray(to_host(b)).tobytes(), path
    elif dataclasses.is_dataclass(a) and not isinstance(a, type):
        assert tuple(vars(a)) == tuple(vars(b)), path
        for f in dataclasses.fields(a):
            _same(getattr(a, f.name), getattr(b, f.name), path+'.'+f.name)
    elif type(a) is dict:
        assert a.keys() == b.keys(), path
        for key in a:
            _same(a[key], b[key], path+'.'+str(key))
    elif type(a) in (tuple, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, path+'['+str(i)+']')
    elif type(a) is float:
        assert a.hex() == b.hex(), path
    else:
        assert a == b, path


def _arrays(value, out):
    import sys
    if type(value) in (np.ndarray, getattr(sys.modules.get('cupy'), 'ndarray', None)):
        assert len(out) < 4096
        out.append(value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            _arrays(getattr(value, f.name), out)
    elif type(value) in (tuple, list):
        for item in value:
            _arrays(item, out)
    elif type(value) is dict:
        for item in value.values():
            _arrays(item, out)
    return out


def _snapshot(*roots):
    arrays = _arrays(roots, [])
    assert sum(a.nbytes for a in arrays) <= 64 << 20
    return tuple((a, a.dtype.str, a.shape, np.ascontiguousarray(to_host(a)).tobytes()) for a in arrays)


def _unchanged(rows):
    for a, dtype, shape, raw in rows:
        assert (a.dtype.str, a.shape, np.ascontiguousarray(to_host(a)).tobytes()) == (dtype, shape, raw)


def _files(root):
    import stat
    rows, entries, total = {}, 0, 0
    for path in root.rglob('*'):
        entries += 1
        assert entries <= 4096
        st = path.lstat()
        if stat.S_ISREG(st.st_mode):
            assert st.st_size <= (64 << 20)-total
            raw = path.read_bytes()
            assert len(raw) == st.st_size
            total += st.st_size
            rows[str(path.relative_to(root))] = raw
        else:
            assert stat.S_ISDIR(st.st_mode)
    return rows


def _limits(root):
    return device.PositiveWindowedDeviceWaterLimits(
        max_host_bytes=5 << 30, max_device_bytes=2 << 30, max_mapped_bytes=128 << 20,
        max_logical_read_bytes=(1 << 63)-1, max_work=(1 << 63)-1,
        max_store_bytes=64 << 20, max_run_bytes=512 << 20, max_new_versions=1000000,
        max_entries=4096, max_external_backing_bytes=64 << 20, max_event_steps=10000,
        max_retries=10, max_transport_substeps=64, max_report_rows=4096,
        reserve_host_bytes=64 << 20, reserve_device_bytes=256 << 20, safety_fraction=0.5, run_dir=root)


def _water_inputs(host, cp):
    """Actual832 all-edge-open scientific inputs; lossless initial placement."""
    water = production.water_inputs(host, all_edges_open=True)
    column = dataclasses.replace(water.column, xp=cp,
        **{f.name: to_device(getattr(water.column, f.name), cp)
           for f in dataclasses.fields(water.column) if type(getattr(water.column, f.name)) is np.ndarray})
    field = dataclasses.replace(water.field, xp=cp, multiplier=to_device(water.field.multiplier, cp))
    sediment = dataclasses.replace(water.sediment, xp=cp,
        **{f.name: to_device(getattr(water.sediment, f.name), cp)
           for f in dataclasses.fields(water.sediment) if f.name.startswith('d_')})
    return water, column, field, sediment, to_device(water.vegetation_cover_fraction, cp), to_device(water.initial_soil_water_m, cp)


def _rainfall(t):
    return RainfallSchedule(edges_s=[t, t+legacy.RAIN_S], intensity_mm_per_h=[legacy.RAIN_MM_H],
                           provenance=RainfallProvenance(kind='constant'))


def _check_dry(handoff):
    result, dry = handoff.progress.result, handoff.dry_state
    assert dry.t_s == result.state.t_s
    assert dry.bed.voxel_column is result.state.bed.voxel_column
    assert dry.bed.water is not result.state.bed.water
    for a in (dry.bed.water.depth_m, dry.bed.water.mobile_mass_by_cell_class_kg, dry.storm.depth_m,
              dry.storm.soil_water_m, dry.storm.discharge_m2_s, dry.sediment_velocity_m_s):
        assert not to_bool(a.any())
    _same(handoff.pre_reset_surface_m, result.state.storm.depth_m, 'pre-reset surface')
    _same(handoff.pre_reset_soil_m, result.state.storm.soil_water_m, 'pre-reset soil')
    assert not arrays_share_memory(handoff.pre_reset_surface_m, result.state.storm.depth_m)
    assert not arrays_share_memory(handoff.pre_reset_soil_m, result.state.storm.soil_water_m)
    accounting = handoff.accounting
    budget = accounting['storm_water_budget']
    assert budget['closed'] and abs(accounting['combined_water_residual_m3']) <= budget['tolerance_m3']
    assert accounting['surface_removed_m3'] == budget['surface_final_m3']
    assert accounting['soil_removed_m3'] == budget['soil_final_m3']
    assert accounting['water_added_m3'] == 0.0 and accounting['target_soil_water_m'] == 0.0
    assert accounting['policy'] == 'external dry-again removal; no evapotranspiration simulated'
    assert not to_bool(result.state.bed.water.mobile_mass_by_cell_class_kg.any())
    assert result.closure()['closed'] and result.closure()['request_reconciled']
    assert accounting['sediment_closure']['closed']
    assert np.all(np.abs(accounting['terminal_sediment_residual_kg']) <= accounting['terminal_sediment_tolerance_kg'])


def test_small_complete_device_water_qualification_matches_dense(tmp_path, cupy, device_water_qualification_budget):
    """Root fixture supplies a separate authenticated qualification budget.

    device_water_qualification_budget is an explicit Root-issued metadata fixture.
    Its excluded allocation terms stay None; it supplies no Original proof DTO.
    """
    cp = cupy
    device.reserve_qualification_device_water_oracle_frontier(device_water_qualification_budget, device_id=int(cp.cuda.Device().id))
    fx = _device_fixture(tmp_path, cp)  # genuine Native Small genesis, no wind block
    host = production.build_bed(production.model())
    original_host_identity = host.identity_sha256()
    h = host.state
    assert (fx.geo.ny, fx.geo.nx, fx.physical.voxel_column.n_voxels, len(fx.ids)) == (4, 4, 6, 1)
    assert fx.geo == host.context.geometry and fx.model.grain_classes == host.context.grain_classes
    cells = np.arange(16, dtype=np.int64)
    full = gather_full_columns(fx.state.physical.voxel_column, cells)
    _same(full, h.voxel_column.mass_kg.reshape(full.shape), 'genesis complete columns')
    for name in ('active_layer', 'ledger', 'water', 'sediment_availability', 'committed_topography'):
        _same(getattr(fx.dense_state.physical, name), getattr(h, name), 'genesis.'+name)
    committed, regroup = windowed_committed_state_from_dense_baseline(
        fx.physical.committed_topography, fx.physical.voxel_column, fx.physical.active_layer,
        fx.geo, trusted_packet=True)
    assert regroup['exact_height_checked'] is True
    p = fx.physical
    bed0 = WindowedSharedBedState(p.voxel_column, p.active_layer, p.water, p.ledger, committed,
                                 p.sediment_availability, p.ledger_reconciliation_baseline)
    dense0 = to_device_tree(h, cp)
    dense_context = dataclasses.replace(host.context, initial_available_fraction=to_device(host.context.initial_available_fraction, cp))
    assert type(dense0) is BedState
    context = device.WindowedDeviceWaterContext(
        geometry=fx.geo, grain_classes=fx.model.grain_classes, mass_resolution_kg=fx.mres,
        topographic_commit_spec=fx.config.topographic_commit_spec, avalanching_spec=fx.config.avalanching_spec,
        initial_available_fraction=fx.config_device.initial_available_fraction, window_max_journal_records=320,
        native_solve_limits=_solve_limits(fx.config.avalanching_spec), native_validation_limits=_validation_limits(),
        native_commit_limits=dataclasses.replace(_commit_limits(), publication_max_records=320),
        source_identity=device_water_qualification_budget.native_source_identity,
        bridge_limits=_limits(tmp_path), runtime_bounds=device_water_qualification_budget)
    w, column, field, sediment, vegetation, soil = _water_inputs(host, cp)
    d, dcolumn, dfield, dsediment, dvegetation, dsoil = _water_inputs(host, cp)
    for a, b, label in ((column, dcolumn, 'column'), (field, dfield, 'rainfield'),
                        (sediment, dsediment, 'sediment'), (vegetation, dvegetation, 'vegetation'), (soil, dsoil, 'soil')):
        _same(a, b, 'complete initial inputs.'+label)
    graph = routing_graph_from_elevation(w.terrain, committed_elevation_m(bed0.committed_topography), xp=cp)
    terrain = terrain_reference(w.terrain.initial_full_elevation_m, w.terrain.export_receiver_full, bed0, graph)
    dgraph = routing_graph_from_elevation(d.terrain, committed_elevation_m(dense0.committed_topography), xp=cp)
    check_graph_evidence_matches(graph, dgraph)
    control = legacy.sediment_control(implementation='cuda', root_solver='bisection')
    dense_control = legacy.sediment_control(implementation='cuda', root_solver='bisection')
    assert (legacy.RAIN_S, legacy.RAIN_MM_H, legacy.GUARD_S, legacy.REPORT_EVERY_S) == (60.0, 72.0, 180.0, 10.0)
    assert control.cuda and control.transport_scheme == 'upwind' and control.deep_publication == 'step'
    assert context.avalanching_spec.enabled and context.depth_update_rule == 'constant_depth'
    policy, dpolicy = legacy.policy(), legacy.policy()
    assert policy == CompletionPolicy(deposit_mobile_at_flow_end=True)
    schedule, dschedule = _rainfall(fx.state.physical.time_s), _rainfall(host.time_s)
    _same(schedule, dschedule, 'rainfall')
    old = bed0.voxel_column.deep_store
    old_files = _files(old.store_dir)
    roots = tuple(getattr(bed0, n) for n in device._BED_FIELDS if n != 'voxel_column')
    from maple.surface.topographic_commit.windowed_device import _bed_leaves
    inputs = _snapshot(roots, tuple(a for _name, a in _bed_leaves(bed0.voxel_column, bed0.active_layer)),
        w, d, column, dcolumn, sediment, dsediment, field, dfield, vegetation, dvegetation, soil, dsoil,
        dense0, graph, dgraph, schedule, dschedule)
    state = initial_event_state(graph, terrain, bed0, context, soil, t_s=fx.state.physical.time_s, control=control)
    dense_state = initial_event_state(dgraph, d.terrain, dense0, dense_context, dsoil, t_s=host.time_s, control=dense_control)
    inputs += _snapshot(state, dense_state, context, dense_context, terrain, w.terrain, d.terrain,
                        context.initial_available_fraction, dense_context.initial_available_fraction,
                        bed0.committed_topography, fx.state, fx.config_device)
    handoff, admission = device.complete_qualification_device_event(

        state, context, column, field, schedule, vegetation, sediment,
        max_end_s=state.t_s+legacy.RAIN_S+legacy.GUARD_S, control=control, policy=policy,
        report_every_s=legacy.REPORT_EVERY_S)
    assert device._OWNER is None
    dense = complete_event(dense_state, dense_context, dcolumn, dfield, dschedule, dvegetation, dsediment,
        max_end_s=dense_state.t_s+legacy.RAIN_S+legacy.GUARD_S, control=dense_control, policy=dpolicy,
        report_every_s=legacy.REPORT_EVERY_S, max_report_rows=4096,
        hydraulics=CudaHydraulics(mode='auto', max_log=256))
    assert type(handoff) is DryHandoff and type(dense) is DryHandoff
    assert handoff.status == dense.status == 'complete'
    a, b = handoff.progress.result, dense.progress.result
    assert a.n_accepted_steps > 0 and a.n_maple_water_calls > 0 and a.n_commits > 0
    assert a.state.t_s > schedule.end_s and handoff.progress.quiet.ready and dense.progress.quiet.ready
    assert float(to_host(a.cumulative_rain_m.sum())) > 0.0 and float(to_host(a.peak_depth_m.max())) > 0.0
    assert float(to_host(a.by_class['actual_pickup'].sum())) > 0.0
    assert float(to_host(a.by_class['deposition_actual'].sum()+a.by_class['export_actual'].sum())) > 0.0
    for f in dataclasses.fields(SedimentEventResult):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if f.name in _TIMING:
            assert all(type(t) is float and np.isfinite(t) and t >= 0 for t in (x, y))
        elif f.name not in _RESULT_SEPARATE:
            _same(x, y, 'complete result.'+f.name)
    assert len(a.commit_log) == len(b.commit_log)
    for i, (x, y) in enumerate(zip(a.commit_log, b.commit_log)):
        assert x.keys() == y.keys()
        for key in x:
            if key == 'commit_stage_log':
                # Exact owning Native logs: Dense labels the bed-preserving
                # real water callback '(identity)'; windowed labels the real
                # non-default callback by name. No other stage is excluded.
                from tests.windowed_wind_gpu.test_device_commit import _NATIVE_STAGE_LOG
                assert tuple(x[key]) == _NATIVE_STAGE_LOG
                dense_stages = tuple('water_callback(identity)' if n == 'water_callback' else n
                                     for n in _NATIVE_STAGE_LOG)
                assert tuple(y[key]) == dense_stages
            else:
                _same(x[key], y[key], f'commit[{i}].{key}')
    for result, initial in ((a, bed0), (b, dense0)):
        _same(result.initial_bed_inventory_kg, bed_inventory(initial), 'own native initial inventory')
        _same(result.initial_bed_by_cell_class_kg, bed_by_cell_class(initial), 'own native initial cells')
    wb, db = a.state.bed, b.state.bed
    assert wb.voxel_column.pending_deep is None
    full = gather_full_columns(wb.voxel_column, cells)
    expected = to_host(db.voxel_column.mass_kg).reshape(full.shape)
    assert np.ascontiguousarray(to_host(full)).tobytes() == np.ascontiguousarray(expected).tobytes()
    for name in ('active_layer', 'ledger', 'water', 'sediment_availability'):
        _same(getattr(wb, name), getattr(db, name), 'final bed.'+name)
    for name in ('reference_elevation_m', 'elevation_offset_m', 'committed_active_layer_mass_by_class_kg', 'last_commit_time_s', 'commit_count'):
        _same(getattr(wb.committed_topography, name), getattr(db.committed_topography, name), 'committed.'+name)
    # Native Small genesis supplies no separate accounting baseline. Both
    # commit bodies preserve None exactly; the legacy Dense six-field family
    # uses its committed inventory as the Native reconciliation reference.
    assert bed0.reconciliation_baseline is None and wb.reconciliation_baseline is None
    assert not hasattr(db, 'reconciliation_baseline')
    assert a.n_commits > 0
    assert wb.committed_topography.last_commit_time_s == a.state.t_s
    assert db.committed_topography.last_commit_time_s == b.state.t_s
    _same(wb.committed_topography.committed_voxel_mass_by_class_kg,
          column_mass_by_class_kg(wb.voxel_column), 'actual committed windowed inventory')
    _same(db.committed_topography.committed_voxel_mass_by_class_kg,
          db.voxel_column.mass_kg.sum(axis=-2), 'actual committed Dense inventory')
    from maple.surface.voxels._numerics import summation_error_bound_kg
    from maple.surface.voxels.capacity import max_voxel_mass_kg
    regroup_bound = summation_error_bound_kg(6+1, max_voxel_mass_kg(fx.geo)*6)
    gap = cp.abs(wb.committed_topography.committed_voxel_mass_by_class_kg
                 - db.committed_topography.committed_voxel_mass_by_class_kg)
    assert to_bool(cp.all(gap <= regroup_bound))  # Existing Native commit oracle policy.
    for x, y in ((bed0, dense0), (wb, db)):

        bound = resident_reduction_bound_by_cell_class_kg(x.voxel_column, fx.geo)
        assert to_bool(cp.all(cp.abs(column_mass_by_class_kg(x.voxel_column)-y.voxel_column.mass_kg.sum(axis=-2)) <= bound))
    for name in ('storm', 'sediment_velocity_m_s', 'routing_source_elevation_m', 'network', 'grid', 'graph'):
        _same(getattr(a.state, name), getattr(b.state, name), 'state.'+name)
    check_graph_evidence_matches(a.state.graph, b.state.graph)
    assert a.state.phase is b.state.phase is None
    for name in ('planned_boundaries', 'origin_t_s', 'initial_surface_m3', 'initial_soil_m3', 'rainfall_end_s',
                 'report_every_s', 'max_report_rows', 'quiet', 'policy', 'control', 'status'):
        _same(getattr(handoff.progress, name), getattr(dense.progress, name), 'progress.'+name)
    for name in ('pre_reset_surface_m', 'pre_reset_soil_m', 'terminal_deposition_kg', 'status'):
        _same(getattr(handoff, name), getattr(dense, name), 'handoff.'+name)
    assert handoff.accounting.keys() == dense.accounting.keys()
    for name in handoff.accounting:
        if name not in _ACCOUNTING_OWN:
            _same(handoff.accounting[name], dense.accounting[name], 'accounting.'+name)
    wa, da = handoff.accounting['sediment_closure'], dense.accounting['sediment_closure']
    assert wa.keys() == da.keys()
    for name in wa:
        if name not in _CLOSURE_OWN:
            _same(wa[name], da[name], 'sediment closure.'+name)
    _check_dry(handoff); _check_dry(dense)
    _same(handoff.dry_state.storm, dense.dry_state.storm, 'dry storm')
    _same(handoff.dry_state.bed.water, dense.dry_state.bed.water, 'dry water')
    _unchanged(inputs)
    host.verify_case_inputs()
    assert host.identity_sha256() == original_host_identity
    assert bed0.voxel_column.deep_store is old
    new = wb.voxel_column.deep_store
    assert new.store_dir == old.store_dir and (new is old or new.version_id > old.version_id)
    after = _files(old.store_dir)
    for name, raw in old_files.items():
        assert after[name] == raw
    assert type(admission) is device.QualificationAdmissionSummary
    assert admission.gate_count > 0 and admission.logical_work_debited <= context.bridge_limits.max_work
    assert admission.logical_read_bytes_debited <= context.bridge_limits.max_logical_read_bytes
    assert admission.versions_reserved <= context.bridge_limits.max_new_versions
    assert admission.maximum_host_predicate <= context.bridge_limits.max_host_bytes
    assert admission.maximum_device_predicate <= context.bridge_limits.max_device_bytes
    assert admission.hydraulics['context']['mode'] == 'fused'

    assert admission.known_subset_only is True
    assert all(value is None for _name, value in admission.opaque_terms)
    assert admission.claim == 'bounded physical qualification; known subset only; no allocation proof'


def _entry_component_inputs(root, cp, budget):
    """A separate genuine Small genesis; no input bed is reconstructed."""
    fx = _device_fixture(root, cp)
    host = production.build_bed(production.model())
    assert fx.geo == host.context.geometry
    assert fx.model.grain_classes == host.context.grain_classes
    committed, regroup = windowed_committed_state_from_dense_baseline(
        fx.physical.committed_topography, fx.physical.voxel_column,
        fx.physical.active_layer, fx.geo, trusted_packet=True)
    assert regroup['exact_height_checked'] is True
    p = fx.physical
    bed = WindowedSharedBedState(p.voxel_column, p.active_layer, p.water, p.ledger,
        committed, p.sediment_availability, p.ledger_reconciliation_baseline)
    context = device.WindowedDeviceWaterContext(
        geometry=fx.geo, grain_classes=fx.model.grain_classes, mass_resolution_kg=fx.mres,
        topographic_commit_spec=fx.config.topographic_commit_spec,
        avalanching_spec=fx.config.avalanching_spec,
        initial_available_fraction=fx.config_device.initial_available_fraction,
        window_max_journal_records=320, native_solve_limits=_solve_limits(fx.config.avalanching_spec),
        native_validation_limits=_validation_limits(),
        native_commit_limits=dataclasses.replace(_commit_limits(), publication_max_records=320),
        source_identity=budget.native_source_identity, bridge_limits=_limits(root), runtime_bounds=budget)
    water = production.water_inputs(host, all_edges_open=True)
    source = committed_elevation_m(bed.committed_topography)
    graph = routing_graph_from_elevation(water.terrain, source, xp=cp)
    reference = terrain_reference(water.terrain.initial_full_elevation_m,
        water.terrain.export_receiver_full, bed, graph)
    control = legacy.sediment_control(implementation='cuda', root_solver='bisection')
    return bed, context, graph, reference, source, control, fx.state.physical.time_s


def _direct_native_entry_oracle(state, context, graph, reference, source, time_s):
    """Independent real Native calls; no SYRUP entry owner or fake BedCommit."""
    from maple.water.windowed_commit_callback import make_windowed_water_depth_commit_callback
    from maple_syrup.sediment_bed import net_routing_displacement_m, reroute_threshold_m
    _bed, _solve, avalanche, commit, *_ = device._native()
    ids = tuple(gc.class_id for gc in context.grain_classes.classes)
    callback, availability = avalanche._make_device_windowed_avalanche_callback(
        context.avalanching_spec, class_ids=ids, solve_limits=context.native_solve_limits,
        validation_limits=context.native_validation_limits,
        sediment_availability=state.sediment_availability,
        initial_available_fraction=context.initial_available_fraction)
    before = committed_elevation_m(state.committed_topography)
    water_callback, water_box = make_windowed_water_depth_commit_callback(
        state.water, context.geometry, before, depth_update_rule='constant_depth', trusted_packet=True)
    result = commit._stage_device_windowed_topography_with_native_avalanche(
        state.ledger, state.committed_topography, state.voxel_column, state.active_layer,
        context.geometry, context.mass_resolution_kg, time_s,
        avalanche_callback=callback, water_callback=water_callback, water_box=water_box,
        water=state.water, committed_elevation_before_m=before, limits=context.native_commit_limits,
        reconciliation_baseline=state.reconciliation_baseline, record_exchange_history=False)
    assert water_box.ran and water_box.depth_update.rule == 'constant_depth'
    assert water_box.water.depth_m is not state.water.depth_m
    assert water_box.water.mobile_mass_by_cell_class_kg is state.water.mobile_mass_by_cell_class_kg
    assert water_box.depth_update.clamp_cell_count == 0
    assert water_box.depth_update.clamp_added_volume_m3 == 0.0
    assert water_box.depth_update.displaced_volume_m3 == 0.0
    _same(water_box.water.depth_m, state.water.depth_m, 'direct constant depth')
    candidate = dataclasses.replace(state, voxel_column=result.new_voxel_column,
        active_layer=result.new_active_layer, committed_topography=result.new_committed_state,
        ledger=result.new_ledger, water=water_box.water,
        sediment_availability=availability.new_sediment_availability,
        reconciliation_baseline=result.new_reconciliation_baseline)
    committed = committed_elevation_m(candidate.committed_topography)
    net = net_routing_displacement_m(committed, source, graph.active_flat.reshape(graph.shape))
    rerouted = net >= reroute_threshold_m(context)
    if rerouted:
        return candidate, routing_graph_from_elevation(reference, committed, xp=graph.xp), committed, net, True
    return candidate, graph, source, net, False


def test_qualification_device_entry_matches_direct_native_commit_and_routing(
        tmp_path, cupy, device_water_qualification_budget):
    """Two real forced entry commits; all work stays in the original budget."""
    from maple_syrup.sediment_bed import BedCommit
    from maple.surface.topographic_commit.windowed_device import _bed_leaves
    cp = cupy
    budget = device_water_qualification_budget
    device.reserve_qualification_device_water_oracle_frontier(budget, device_id=int(cp.cuda.Device().id))
    actual, context, graph, reference, source, control, time_s = _entry_component_inputs(
        tmp_path/'entry', cp, budget)
    expected, direct_context, direct_graph, direct_reference, direct_source, _direct_control, direct_time = _entry_component_inputs(
        tmp_path/'oracle', cp, budget)
    assert time_s == direct_time
    old_versions = (actual.voxel_column.deep_store, expected.voxel_column.deep_store)
    old_files = tuple(_files(v.store_dir) for v in old_versions)
    inputs = _snapshot(actual, expected, source, direct_source,
        tuple(a for _n, a in _bed_leaves(actual.voxel_column, actual.active_layer)),
        tuple(a for _n, a in _bed_leaves(expected.voxel_column, expected.active_layer)),
        context.initial_available_fraction, direct_context.initial_available_fraction)
    summaries = []
    for ordinal in range(2):
        previous, held_graph, held_source = actual, graph, source
        direct_held_graph, direct_held_source = direct_graph, direct_source
        committed, summary = device.commit_qualification_device_entry(
            actual, context, graph, reference, time_s, source, control=control)
        assert device._OWNER is None
        assert type(committed) is BedCommit and committed.did_commit is True
        assert committed.state is not previous
        assert committed.water_result.water.mobile_mass_by_cell_class_kg is previous.water.mobile_mass_by_cell_class_kg
        assert committed.state.water.depth_m is not previous.water.depth_m
        _same(committed.state.water.depth_m, previous.water.depth_m, 'entry constant depth')
        expected, direct_graph, direct_source, net, rerouted = _direct_native_entry_oracle(
            expected, direct_context, direct_graph, direct_reference, direct_source, direct_time)
        assert committed.rerouted is rerouted and committed.net_displacement_m == net
        actual, graph, source = committed.state, committed.graph, committed.routing_source_m
        if rerouted:
            assert graph is not held_graph and source is not held_source
            assert direct_graph is not direct_held_graph and direct_source is not direct_held_source
        else:
            assert graph is held_graph and source is held_source
            assert direct_graph is direct_held_graph and direct_source is direct_held_source
        check_graph_evidence_matches(graph, direct_graph)
        _same(source, direct_source, 'entry routing source')
        for name in device._BED_FIELDS:
            if name != 'voxel_column':
                _same(getattr(actual, name), getattr(expected, name), 'entry.'+name)
        cells = np.arange(16, dtype=np.int64)
        _same(gather_full_columns(actual.voxel_column, cells),
              gather_full_columns(expected.voxel_column, cells), 'entry complete logical column')
        for name in ('window_kg', 'base', 'deep_by_class_kg', 'deep_total_kg', 'deep_room_kg', 'n_voxels'):
            _same(getattr(actual.voxel_column, name), getattr(expected.voxel_column, name), 'entry.column.'+name)
        assert actual.voxel_column.pending_deep is expected.voxel_column.pending_deep is None
        assert type(summary) is device.QualificationAdmissionSummary
        assert summary.gate_count == 4  # One entry + the unchanged trigger/writer/candidate gates.
        assert summary.hydraulics is None and summary.known_subset_only is True
        assert len(summary.opaque_terms) == 20 and all(v is None for _n, v in summary.opaque_terms)
        assert summary.claim == 'bounded physical qualification; known subset only; no allocation proof'
        assert summary.maximum_host_predicate <= context.bridge_limits.max_host_bytes
        assert summary.maximum_device_predicate <= context.bridge_limits.max_device_bytes
        summaries.append(summary)
        _unchanged(inputs)
    assert sum(s.logical_work_debited for s in summaries) <= context.bridge_limits.max_work
    assert sum(s.logical_read_bytes_debited for s in summaries) <= context.bridge_limits.max_logical_read_bytes
    assert sum(s.versions_reserved for s in summaries) <= context.bridge_limits.max_new_versions
    for old, files, final in zip(old_versions, old_files, (actual, expected), strict=True):
        current = final.voxel_column.deep_store
        assert current.store_dir == old.store_dir and (current is old or current.version_id >= old.version_id)
        after = _files(old.store_dir)
        for name, raw in files.items():
            assert after[name] == raw
