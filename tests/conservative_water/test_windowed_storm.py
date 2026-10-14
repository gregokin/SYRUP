"""Actual resident MAPLE bed, compiled CPU/GPU storms, completion and restart."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_coupled_backends import H
from test_exchange_history import IDENTITY, complete, inputs

from maple_syrup.checkpoint import load_checkpoint, save_checkpoint
from maple_syrup.complete_event import DryHandoff
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_event import initial_event_state


def window_inputs(xp, tmp_path, storage='csr'):
    from maple.core.backend import to_device_tree, to_host_tree
    from maple.core.types.sediment_ledger import (
        process_index,
        zeros_sediment_ledger_state,
    )
    from maple.surface.topographic_commit.windowed_commit import (
        windowed_committed_state_from_dense_baseline,
    )
    from maple.surface.voxels.windowed_inventory import windowed_column_from_dense
    from maple.surface.voxels.windowed_prefix import move_prefix_packets

    implementation='numba' if xp is np else 'cuda'
    s,ctx,control,schedule=inputs(xp,commit_interval_s=2.0,implementation=implementation)
    control=replace(control,phase_storage=storage)
    state=s['state0'];old=state.bed;host=to_host_tree(old)
    # A genuine deeper MAPLE column, with a uniform added datum thickness.
    # Keeps the original layer/availability, adds real packed buried voxels,
    # and rebuilds the terrain reference through actual MAPLE diagnostics.
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.core.types.voxel import VoxelColumnState
    from maple.surface.topographic_commit.initialization import (
        initial_committed_topography_state_from_physical_state,
    )

    from maple_syrup.sediment_bed import refresh_routing
    deep = np.repeat(host.voxel_column.mass_kg[:,:,:1,:], 32, axis=2)
    dense_column = VoxelColumnState(np.concatenate((deep,host.voxel_column.mass_kg),axis=2))
    baseline = initial_committed_topography_state_from_physical_state(
        dense_column,host.active_layer,ctx.geometry,ctx.mass_resolution_kg)
    host = replace(host,voxel_column=dense_column,committed_topography=baseline)
    old_ref = state.terrain
    shift = committed_elevation_m(baseline) - old_ref.initial_committed_elevation_m
    full = old_ref.initial_full_elevation_m.copy() + float(shift[0,0])
    full[1:-1,1:-1] = old_ref.initial_full_elevation_m[1:-1,1:-1] + shift
    ref = replace(old_ref,initial_full_elevation_m=full,
                  initial_committed_elevation_m=committed_elevation_m(baseline).copy())
    column=windowed_column_from_dense(host.voxel_column,ctx.geometry,ctx.mass_resolution_kg,
                                      store_dir=tmp_path/'store',max_store_bytes=10**8,
                                      class_ids=tuple(g.class_id for g in ctx.grain_classes.classes),
                                      window_voxels=min(4,host.voxel_column.mass_kg.shape[2]),
                                      chunk_columns=64,with_prefix_packets=True)
    committed,_=windowed_committed_state_from_dense_baseline(host.committed_topography,column,host.active_layer,ctx.geometry)
    ledger=zeros_sediment_ledger_state(ctx.geometry.ny,ctx.geometry.nx,len(ctx.grain_classes.classes),
                                       present_processes=(process_index('water_erosion_deposition'),))
    bed=replace(host,voxel_column=column,committed_topography=committed,ledger=ledger)
    if xp is not np:
        moved=to_device_tree(bed,xp)
        bed=replace(moved,voxel_column=move_prefix_packets(moved.voxel_column,column,xp))
    graph=refresh_routing(ref,bed,xp=xp)
    s['state0']=initial_event_state(graph,ref,bed,ctx,state.storm.soil_water_m,control=control)
    assert np.any(H(bed.voxel_column.base)>0)
    return s,ctx,control,schedule



@pytest.mark.parametrize('backend', ['numpy','cupy'])
@pytest.mark.parametrize('storage', ['dense','csr'])
def test_complete_window_storm_and_wet_restart(backend,storage,tmp_path):
    from maple.core.backend import (
        BackendUnavailableError,
        array_leaves,
        resolve_backend,
    )
    from maple.surface.voxels.windowed_inventory import windowed_to_dense

    try:
        xp=resolve_backend(backend).xp
    except BackendUnavailableError as exc:
        pytest.skip(str(exc))
    s,ctx,ctl,schedule=window_inputs(xp,tmp_path,storage)
    paused=complete(s,ctx,ctl,schedule,checkpoint_callback=lambda p:True)
    assert H(paused.result.state.bed.water.mobile_mass_by_cell_class_kg).sum()>0
    assert paused.result.state.bed.voxel_column.pending_deep is None
    checkpoint=tmp_path/'wet'
    save_checkpoint(checkpoint,paused,ctx,s['column'],s['sediment'],IDENTITY)
    restored=load_checkpoint(checkpoint,ctx,s['column'],s['sediment'],IDENTITY)
    assert restored.result.state.bed.voxel_column.manifest_digest==paused.result.state.bed.voxel_column.manifest_digest
    reference=complete(s,ctx,ctl,schedule)
    resumed=complete(dict(s,state0=restored.result.state),ctx,ctl,schedule,continuation=restored)
    assert isinstance(reference,DryHandoff) and isinstance(resumed,DryHandoff)
    assert reference.progress.result.closure()['closed'] and resumed.progress.result.closure()['closed']
    assert reference.dry_state.bed.voxel_column.pending_deep is None
    a=windowed_to_dense(reference.dry_state.bed.voxel_column,max_bytes=10**8)
    b=windowed_to_dense(resumed.dry_state.bed.voxel_column,max_bytes=10**8)
    assert a.mass_kg.tobytes()==b.mass_kg.tobytes()
    aa,bb=array_leaves(reference.dry_state),array_leaves(resumed.dry_state)
    assert len(aa)==len(bb)
    for u,v in zip(aa,bb,strict=True):
        assert H(u).dtype==H(v).dtype and H(u).shape==H(v).shape and H(u).tobytes()==H(v).tobytes()
    for field in ('n_accepted_steps','n_commits','n_graph_changes','n_forced_commits'):
        assert getattr(reference.progress.result,field)==getattr(resumed.progress.result,field)
    assert reference.accounting==resumed.accounting
    drypath=tmp_path/'dry'
    save_checkpoint(drypath,reference,ctx,s['column'],s['sediment'],IDENTITY)
    loaded=load_checkpoint(drypath,ctx,s['column'],s['sediment'],IDENTITY,allow_complete=True)
    assert isinstance(loaded,DryHandoff)


def _store_pins(store):
    import hashlib
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in store.rglob('*') if p.is_file()}


def test_late_full_step_refusal_does_not_publish_deep_versions_history(tmp_path,monkeypatch):
    """The ordinary ACCOUNTING commit path (no reroute): the exchange-history fold of the first terrain commit,
    triggered on the first accepted candidate, is refused; nothing is published and the input is untouched."""
    from maple.core.backend import array_leaves
    from test_exchange_history import run

    import maple_syrup.sediment_event as event_module

    s,ctx,ctl,schedule=window_inputs(np,tmp_path)
    # Trigger the first terrain commit on the first accepted candidate.
    ctx=replace(ctx,topographic_commit_spec=replace(ctx.topographic_commit_spec,commit_interval_s=0.25))
    store=tmp_path/'store'
    before=_store_pins(store)
    arrays=[a.tobytes() for a in array_leaves(s['state0'])]
    reached=[]
    def reject(*a,**kw):
        reached.append('fold_commit_exchange_history')
        raise ValueError('injected late candidate refusal')
    monkeypatch.setattr(event_module,'fold_commit_exchange_history',reject)
    with pytest.raises(ValueError,match='injected late candidate refusal'):
        run(s,ctx,ctl,schedule,2.)
    assert reached==['fold_commit_exchange_history']
    assert before==_store_pins(store)
    assert arrays==[a.tobytes() for a in array_leaves(s['state0'])]


# CW-E47C1: the reroute-only late stages (`phase_for_network` of a REROUTED commit, and the routing graph rebuild of a
# genuine net crossing) are reached only when the post-commit net elevation displacement from the held routing
# source reaches MAPLE's actual `pending_elevation_trigger_m` (default 0.00075 m, `>=`). The original 2 s / 100 mm/h
# window never gets near it (accounting commits only, max net ~5e-6 m), so the injected refusals were never reached.
# A stronger PHYSICAL storm on the same actual resident-window fixture (1000 mm/h, ordinary 2 s accounting commits,
# default threshold) produces a genuine `net_elevation` reroute; the run is paused one second BEFORE it and the
# continuation's first candidate is the rerouting one.
REROUTE_RAIN_MM_H, REROUTE_END_S = 1000.0, 90.0


def _reroute_fixture(tmp_path):
    s,ctx,ctl,_=window_inputs(np,tmp_path)
    ctx=replace(ctx,topographic_commit_spec=replace(ctx.topographic_commit_spec,commit_interval_s=2.0))
    return s,ctx,ctl,constant_rainfall(0,REROUTE_END_S,REROUTE_RAIN_MM_H)


def test_reroute_fixture_genuinely_crosses_the_default_threshold(tmp_path):
    """Positive control of the reroute fixture (no injection): the first reroute is a genuine net crossing at the
    default threshold after ordinary below-threshold accounting commits that did NOT reroute; budgets close."""
    from test_exchange_history import run

    s,ctx,ctl,schedule=_reroute_fixture(tmp_path)
    assert ctx.topographic_commit_spec.pending_elevation_trigger_m==0.00075  # MAPLE's default, unchanged
    r=run(s,ctx,ctl,schedule,REROUTE_END_S)
    assert r.closure()['closed'] and r.n_reroutes>=1
    first=next(i for i,e in enumerate(r.commit_log) if e['rerouted'])
    e=r.commit_log[first]
    assert e['reroute_threshold_m']==0.00075 and e['net_displacement_m']>=e['reroute_threshold_m']
    assert e['commit_reason']=='net_elevation' and not e['forced']
    before=r.commit_log[:first]
    assert before and all(not c['rerouted'] and c['net_displacement_m']<c['reroute_threshold_m'] for c in before)
    assert all(c['commit_reason']=='maple_trigger' for c in before)  # time-only accounting commits never rerouted
    assert r.routing_source_origin in ('net_reroute',) and r.max_unrouted_net_displacement_m<0.00075


@pytest.mark.parametrize('late_failure',['phase','routing'])
def test_late_full_step_refusal_does_not_publish_deep_versions(tmp_path,monkeypatch,late_failure):
    from maple.core.backend import array_leaves
    from test_exchange_history import run

    import maple_syrup.sediment_bed as bed_module
    import maple_syrup.sediment_event as event_module

    s,ctx,ctl,schedule=_reroute_fixture(tmp_path)
    reference=run(*_reroute_fixture(tmp_path/'reference'),REROUTE_END_S)
    t_reroute=next(e['t_s'] for e in reference.commit_log if e['rerouted'])
    # pause one second before the genuine crossing (no forced final commit in this control), then refuse it
    paused=run(s,ctx,ctl,schedule,t_reroute-1.0)
    assert paused.state.t_s==t_reroute-1.0 and paused.n_reroutes==0
    store=tmp_path/'store'
    before=_store_pins(store)
    arrays=[a.tobytes() for a in array_leaves(paused.state)]
    reached=[]
    real_phase=event_module.phase_for_network
    real_route=bed_module.routing_graph_from_elevation
    import sys

    def phase_hook(phase,network):
        commit=sys._getframe(1).f_locals.get('commit')  # prepared_commit_values(commit, phase)
        if commit is not None and commit.rerouted:
            reached.append(('phase_for_network',commit.record['t_s'],commit.record['commit_reason'],
                            commit.record['net_displacement_m'],commit.record['reroute_threshold_m']))
            raise ValueError('injected late candidate refusal')
        return real_phase(phase,network)

    def route_hook(reference_,elevation,*,xp=np):
        frame=sys._getframe(1)
        if frame.f_code.co_name=='commit_bed':  # the genuine net crossing's graph rebuild
            reached.append(('routing_graph_from_elevation',frame.f_locals['time_s'],None,float(frame.f_locals['net']),
                            float(frame.f_locals['threshold'])))
            raise ValueError('injected late candidate refusal')
        return real_route(reference_,elevation,xp=xp)
    if late_failure=='phase':
        monkeypatch.setattr(event_module,'phase_for_network',phase_hook)
    else:
        monkeypatch.setattr(bed_module,'routing_graph_from_elevation',route_hook)
    with pytest.raises(ValueError,match='injected late candidate refusal'):
        run(s,ctx,ctl,schedule,REROUTE_END_S,continuation=paused,state=paused.state)
    # the hook was reached exactly once, by the genuine crossing at the reference's reroute time
    assert len(reached)==1
    name,t_s,reason,net,threshold=reached[0]
    assert t_s==t_reroute and threshold==0.00075 and net>=threshold
    assert reason in (None,'net_elevation')
    assert before==_store_pins(store)
    assert arrays==[a.tobytes() for a in array_leaves(paused.state)]
