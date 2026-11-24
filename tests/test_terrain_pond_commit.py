"""Real native avalanche/commit on a wet world-base irregular footprint."""
import numpy as np
from maple.core.boundaries import AxisBoundary, BoundaryKind
from maple.core.parameters.geometry import GeometrySpec
from maple.core.parameters.grain_classes import resolve_grain_classes
from maple.core.parameters.topographic_commit import TopographicCommitSpec
from maple.core.parameters.avalanching import AvalanchingSpec
from maple.core.types.fixed_base import fixed_base_state_from_elevation
from maple.core.types.domain import domain_state_from_valid_mask
from maple.core.types.voxel import zeros_voxel_column_state
from maple.core.types.water import water_state_from_depth
from maple.core.types.sediment_ledger import zeros_sediment_ledger_state
from maple.core.types.topographic_commit import committed_elevation_m
from maple.surface.voxels.transfer import deposit_surface_mixture_batch
from maple.surface.active_layer.initialization import initialize_active_layer_from_voxels
from maple.surface.topographic_commit.initialization import initial_committed_topography_state_from_physical_state
from maple_syrup.sediment_bed import BedState, BedContext, current_routing_surface_m
from maple_syrup.pond_bed import native_commit, physical_full_source, gate_displacement_m, reroute_threshold_m


def test_native_avalanche_preserves_water_and_refreshes_true_world_surface():
    periodic = AxisBoundary(kind=BoundaryKind.PERIODIC)
    g = GeometrySpec(nx=5, ny=3, dx_m=.25, dy_m=.25, boundary_x=periodic,
        boundary_y=periodic, voxel_dz_m=.1, bulk_density_kg_m3=1250.,
        active_layer_thickness_m=.002, surface_edge_policy='closed')
    gc = resolve_grain_classes([dict(class_id='sand',diameter_m=.0002,particle_density_kg_m3=2650.,is_aggregate=False)])
    valid = np.ones((3,5),bool); valid[1,2] = False
    base_z = 1400.+.05*np.indices(valid.shape)[1]
    base = fixed_base_state_from_elevation(base_z, source='imported', source_label='synthetic plane')
    domain = domain_state_from_valid_mask(valid, boundary_x=BoundaryKind.PERIODIC, boundary_y=BoundaryKind.PERIODIC)
    thickness = np.full(valid.shape,.3);thickness[1,1] = .55
    column = zeros_voxel_column_state(ny=3,nx=5,n_voxels=8,n_classes=1)
    request = np.where(valid,thickness*g.bulk_density_kg_m3*g.dx_m*g.dy_m,0.)[...,None]
    column,_ = deposit_surface_mixture_batch(column, request, g, 1e-12)
    column,active,_ = initialize_active_layer_from_voxels(column,g,1e-12)
    committed = initial_committed_topography_state_from_physical_state(column,active,g,1e-12,fixed_base=base)
    water = water_state_from_depth(np.where(valid,.02,0.),1)
    bed = BedState(column,active,water,zeros_sediment_ledger_state(ny=3,nx=5,n_classes=1),committed,None)
    ctx = BedContext(g,gc,1e-12,TopographicCommitSpec(),AvalanchingSpec(),
        np.where(valid,1.,0.)[...,None],fixed_base=base,domain=domain)
    before = committed_elevation_m(committed).copy()
    result = native_commit(bed,ctx,1.,force=True)
    assert result.did_commit
    child = result.state
    assert not np.array_equal(child.voxel_column.mass_kg,column.mass_kg)
    np.testing.assert_allclose(child.voxel_column.mass_kg.sum()+child.active_layer.mass_kg.sum(),
        column.mass_kg.sum()+active.mass_kg.sum(),rtol=0,atol=1e-9)
    np.testing.assert_array_equal(child.water.depth_m,water.depth_m)
    assert np.all(child.voxel_column.mass_kg[~valid] == 0)
    assert np.all(child.active_layer.mass_kg[~valid] == 0)
    actual = current_routing_surface_m(child,ctx)
    assert actual[valid].min() > 1000.
    assert gate_displacement_m(child,ctx,before,valid) >= reroute_threshold_m(ctx)
    full = np.full((5,7),-9999.);full[1:-1,1:-1][valid] = before[valid]
    refreshed = physical_full_source(full,before,committed_elevation_m(child.committed_topography),valid)
    np.testing.assert_array_equal(refreshed[1:-1,1:-1][valid],actual[valid])
    assert np.all(refreshed[1:-1,1:-1][~valid] == -9999.)
