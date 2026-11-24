"""Native lifecycle contract tests; synthetic handoff fixtures, no wind-physics qualification."""
import numpy as np
import pytest
import wind_builders as wb
from maple_syrup import floating_boundary_outlets as bo
from maple_syrup import floating_event_lifecycle as lc

QUOTA = dict(available_bytes_fn=wb.AVAIL, memory_budget_bytes=1 << 28)


def changed(only_one=False):
    from maple_syrup import floating_wind_handoff as wh
    from maple_syrup.pond_event import PondEvent
    from maple_syrup.pond_bed import native_commit
    ns = wb.build(base=0.20, bottom="closed", saturation=0.3, depth=0.001)
    # Authored synthetic observed step, small enough for one native pickup.
    topo = wb.ph.make_topology(ns.bed, [(1,3,0,1)] if only_one else wb.ph.east_outlets(3,4), drop=1e-5)
    def factory(v, carried=None, schedule=None):
        return PondEvent(ns.ctx,None,ns.ev0.sediment,ns.ev0.schedule if schedule is None else schedule,
            topo if carried is None else carried,control=ns.ev0.control,policy=ns.ev0.policy,floating_vertical=v)
    ns.factory = factory
    ns.lin,ns.b,ns.wctx=wh.start_wind_boundary(ns.owner,ns.bed,factory,bed_context=ns.ctx,limits=wb.LIMITS,available_bytes_fn=wb.AVAIL)
    child=wb.child_move(ns.b.bed,ns.ctx,src=(1,3),dst=(1,2),dz_m=1e-4)
    child=native_commit(child,ns.ctx,120.0,force=True).state
    wb.wind(ns,child)
    return ns


def test_noop_keeps_exact_routing_source_generation_and_sequence():
    ns = wb.started(base=0.20)
    parent = ns.b
    got, receipt = bo.close_uphill_boundary_outlets(ns.lin, parent, ns.ctx, **QUOTA)
    assert got is parent and ns.lin.head is parent and not receipt['changed']
    assert got.routing is parent.routing and got.seq == parent.seq


def test_closure_is_native_compatibility_rebuild_with_zero_exchange():
    ns = changed()
    old = ns.b
    old_qx, old_qy = old.qx_m2_s.copy(), old.qy_m2_s.copy()
    new, receipt = bo.close_uphill_boundary_outlets(ns.lin, old, ns.ctx, **QUOTA)
    assert new is ns.lin.head and new is not old and receipt['changed']
    assert receipt['closed_faces'] and receipt['water_exchange_m3'] == receipt['sediment_exchange_kg'] == receipt['elapsed_s'] == 0
    for name in ('root','bed','surface_depth_m','ledger','cursor','pending_forcing','storm_match'):
        assert getattr(new,name) is getattr(old,name)
    assert new.role == old.role and new.t_s == old.t_s and new.seq == old.seq+1
    assert new.routing[0].generation == old.routing[0].generation+1
    assert new.routing[0].identity_sha256 != old.routing[0].identity_sha256
    np.testing.assert_array_equal(new.routing[0].source_z_full[0],old.routing[0].source_z_full[0])
    np.testing.assert_array_equal(new.routing[0].source_z_full[-1],old.routing[0].source_z_full[-1])
    for r,c,dr,dc in receipt['closed_faces']:
        if dc:old_qx[r,c+(dc>0)] = 0
        else:old_qy[r+(dr>0),c] = 0
    np.testing.assert_array_equal(new.qx_m2_s,old_qx)
    np.testing.assert_array_equal(new.qy_m2_s,old_qy)
    again, r = bo.close_uphill_boundary_outlets(ns.lin,new,ns.ctx,**QUOTA)
    assert again is new and not r['changed']
    with pytest.raises(lc.LifecycleError):bo.close_uphill_boundary_outlets(ns.lin,old,ns.ctx,**QUOTA)


def test_callback_failure_leaves_genuine_parent_and_arrays_unchanged():
    ns = changed();old=ns.b;snap=wb.snapshot(old)
    def fail():raise RuntimeError('before publication')
    with pytest.raises(RuntimeError,match='before publication'):
        bo.close_uphill_boundary_outlets(ns.lin,old,ns.ctx,before_publish=fail,**QUOTA)
    assert ns.lin.head is old
    wb.assert_same(snap,wb.snapshot(old))
    got,_=bo.close_uphill_boundary_outlets(ns.lin,old,ns.ctx,**QUOTA)
    assert got is ns.lin.head


def test_low_budget_refuses_before_native_array_work_and_bad_reader_is_rejected():
    ns = wb.started(base=0.20);old=ns.b
    with pytest.raises(MemoryError):
        bo.close_uphill_boundary_outlets(ns.lin,old,ns.ctx,available_bytes_fn=wb.AVAIL,memory_budget_bytes=1)
    assert ns.lin.head is old
    with pytest.raises(ValueError):
        bo.close_uphill_boundary_outlets(ns.lin,old,ns.ctx,available_bytes_fn=lambda:True,memory_budget_bytes=1<<28)
    assert ns.lin.head is old


def test_all_outlets_closed_retains_water_and_next_native_event_accepts_routing():
    ns=changed(only_one=True);old=ns.b
    new,r=bo.close_uphill_boundary_outlets(ns.lin,old,ns.ctx,**QUOTA)
    assert r['outlets_after']==0 and not new.routing[0].outlets
    assert new.routing[0].n_closed_components==1
    assert new.surface_depth_m is old.surface_depth_m and new.provenance is old.provenance
    occ,state=lc.begin_occurrence(ns.lin,new,ns.factory,available_bytes_fn=wb.AVAIL)
    assert state.topology is new.routing[0] and state.t_s==new.t_s
    np.testing.assert_array_equal(state.hydraulic.depth_m,new.surface_depth_m)
