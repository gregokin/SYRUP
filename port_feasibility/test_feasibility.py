from dataclasses import replace
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import brentq

from kernels import Network, StepRejected, incoming, sediment_step, water_step


def chain(n=4, dx=.5):
    return Network(np.r_[np.arange(1,n),-1], np.arange(n), dx)


@pytest.mark.parametrize('dt', [.1, 1., 5.])
def test_water_equation_against_independent_scalar_root(dt):
    net = chain(1)
    h, slope, f, rain = .02, .015, 1., 1e-5
    k = np.sqrt(8*9.81*slope/f)
    q = k*h**1.5
    if h+dt*rain-dt*q/(2*net.dx_m) < 0:
        with pytest.raises(StepRejected):
            water_step(net, [h], [q], [rain], [slope], [f], dt)
        return
    result = water_step(net, [h], [q], [rain], [slope], [f], dt)
    rhs = h+dt*rain-dt*q/(2*net.dx_m)
    reference = brentq(lambda x: x+dt*k*x**1.5/(2*net.dx_m)-rhs, 0, rhs,
                       xtol=1e-15, rtol=1e-14)
    np.testing.assert_allclose(result.depth_m, reference, atol=1e-14, rtol=1e-12)


def test_water_branching_network_local_and_global_budgets():
    net = Network(np.array([2,2,3,-1]), np.arange(4), .5)
    h = np.array([.01,.005,.003,.001])
    slope = np.full(4,.01)
    q = np.sqrt(8*9.81*slope)*h**1.5
    rain = np.array([1e-5,2e-5,0,1e-5])
    result = water_step(net,h,q,rain,slope,np.ones(4),.1)
    np.testing.assert_allclose(result.balance_by_cell_m3,0,atol=1e-17)
    assert abs(h.sum()*.25+.1*rain.sum()*.25-result.depth_m.sum()*.25-result.export_m3)<1e-17
    assert np.all(result.depth_m>=0)


def test_dry_domain_with_no_rain_is_exactly_inactive():
    result = water_step(chain(),np.zeros(4),np.zeros(4),np.zeros(4),np.ones(4),np.ones(4),1.)
    assert not np.any(result.depth_m)
    assert result.export_m3 == 0


def test_rain_wets_downstream_cells_in_same_ordered_sweep():
    rain = np.array([1e-3,0,0,0])
    result = water_step(chain(),np.zeros(4),np.zeros(4),rain,np.full(4,.02),np.ones(4),1.)
    assert np.all(result.depth_m>0)  # all levels use new upstream discharge
    np.testing.assert_allclose(result.balance_by_cell_m3,0,atol=1e-18)


def test_water_timestep_refinement_against_analytic_drainage():
    net = chain(1)
    h0, slope, duration = .01,.02,1.
    k=np.sqrt(8*9.81*slope)
    exact=(h0**-.5+.5*k/net.dx_m*duration)**-2
    errors=[]
    for dt in [.2,.1,.05]:
        h=np.array([h0]); q=k*h**1.5
        for _ in range(round(duration/dt)):
            result=water_step(net,h,q,[0],[slope],[1],dt)
            h,q=result.depth_m,result.discharge_m2_s
        errors.append(abs(h[0]-exact))
    assert errors[1] < errors[0]/3
    assert errors[2] < errors[1]/3


@pytest.mark.parametrize('kind',['water','sediment'])
def test_negative_rhs_rejects_without_mutation(kind):
    net=chain(1)
    if kind=='water':
        h=np.array([.001]);q=np.array([1.]); before=(h.copy(),q.copy())
        with pytest.raises(StepRejected):
            water_step(net,h,q,[0],[.01],[1],1.)
    else:
        h=np.array([[.001]]);q=np.array([[1.]]);before=(h.copy(),q.copy())
        with pytest.raises(StepRejected):
            sediment_step(net,h,q,[[1.]],1.)
    np.testing.assert_array_equal(h,before[0]);np.testing.assert_array_equal(q,before[1])


def test_invalid_order_and_cycles_rejected():
    with pytest.raises(ValueError,match='upstream'):
        Network(np.array([1,-1]),np.array([1,0]),.5)
    with pytest.raises(ValueError,match='cycles'):
        Network(np.array([1,0]),np.array([0,1]),.5)


def test_multiclass_advection_against_dense_linear_system():
    net=Network(np.array([2,2,3,-1]),np.arange(4),.5)
    m=np.array([[.1,.02],[.2,.03],[0,.01],[.01,0]])
    v=np.array([[.2,.1],[.3,.2],[.1,.3],[.2,.1]])
    q=m*v/net.dx_m;dt=.2
    result=sediment_step(net,m,q,v,dt)
    for c in range(2):
        matrix=np.diag(1+.5*dt*v[:,c]/net.dx_m)
        for source,dest in enumerate(net.receiver):
            if dest>=0: matrix[dest,source]-=.5*dt*v[source,c]/net.dx_m
        reference=np.linalg.solve(matrix,m[:,c]+.5*dt*(incoming(q,net)[:,c]-q[:,c]))
        np.testing.assert_allclose(result.mobile_kg[:,c],reference,atol=1e-16)
    np.testing.assert_allclose(result.balance_by_cell_class_kg,0,atol=1e-16)
    np.testing.assert_allclose(m.sum(0),result.mobile_kg.sum(0)+result.export_by_class_kg,atol=1e-16)


def test_hydraulic_checkpoint_resume_matches_continuous(tmp_path):
    net=chain(); h=np.full(4,.01);slope=np.full(4,.01);q=np.sqrt(8*9.81*slope)*h**1.5
    def advance(h,q,n):
        for _ in range(n):
            r=water_step(net,h,q,np.full(4,1e-5),slope,np.ones(4),.1)
            h,q=r.depth_m,r.discharge_m2_s
        return h,q
    expected=advance(h,q,10)
    mid=advance(h,q,4)
    np.savez(tmp_path/'hydrology.npz',h=mid[0],q=mid[1])
    with np.load(tmp_path/'hydrology.npz') as stored:
        actual=advance(stored['h'],stored['q'],6)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_maple_pickup_transport_deposition_uses_one_bed():
    maple_root=Path(os.environ.get('MAPLE_ROOT','/home/okin/MAPLE'))
    sys.path.insert(0,str(maple_root))
    from tests.unit._phase19_helpers import build_geometry,build_bed,MASS_RESOLUTION_KG
    from maple.aeolian.flux.step import bed_inventory_by_class_kg
    from maple.core.types.water import WaterState
    from maple.water.interfaces import WaterProcessDemand
    from maple.water.step import apply_water_process_demand
    geometry=build_geometry(ny=1,nx=2,dx_m=.5,dy_m=.5)
    classes,column,layer,ledger,water=build_bed(geometry)
    before=bed_inventory_by_class_kg(column,layer)
    zeros=np.zeros((1,2,1));removal=zeros.copy();removal[0,0,0]=1000 # force holdings cap
    pickup=apply_water_process_demand(column,layer,water,ledger,WaterProcessDemand(removal,zeros),
                                     geometry,classes,MASS_RESOLUTION_KG)
    actual=pickup.actual_removal_by_cell_class_kg
    assert 0 < actual.sum() < removal.sum()
    # Terminal cell is a zero-velocity sink; no export. This diagnostic uses
    # closed transfers, so the helper's periodic grid faces are not exercised.
    moved=sediment_step(chain(2),pickup.new_water.mobile_mass_by_cell_class_kg.reshape(2,1),
                        np.zeros((2,1)),np.array([[.2],[0.]]),1.)
    assert moved.mobile_kg[1,0]>0
    assert moved.export_by_class_kg[0]==0
    routed=WaterState(depth_m=water.depth_m,mobile_mass_by_cell_class_kg=moved.mobile_kg.reshape(1,2,1))
    deposited=apply_water_process_demand(pickup.new_voxel_column,pickup.new_active_layer,routed,
        pickup.new_ledger,WaterProcessDemand(zeros,routed.mobile_mass_by_cell_class_kg),
        geometry,classes,MASS_RESOLUTION_KG)
    np.testing.assert_allclose(bed_inventory_by_class_kg(deposited.new_voxel_column,deposited.new_active_layer),before,atol=1e-12)
    assert deposited.deposition_by_cell_class_kg[0,1,0]>0
    assert not np.any(deposited.new_water.mobile_mass_by_cell_class_kg)
    assert not np.any(water.mobile_mass_by_cell_class_kg)
