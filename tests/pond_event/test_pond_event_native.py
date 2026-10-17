"""The native transaction the event composes -- ACTUAL MAPLE pickup / deposit / export on an actual bed with the multi-face pond
transport and the real topology faces -- with PRESCRIBED sediment physics, so deposition, split and export are analytic and
deterministic. Also the real-bed D4 slope / aspect and the friction-slope shear adaptation. Nothing here was run by its author."""
import numpy as np
import pytest
from pond_helpers import DX, NC, arrays_of, assert_unchanged, basin, host, make_bed, make_topology, plane

from maple_syrup.characteristic_transport import empty_phase_state
from maple_syrup.conservation import reservoir_bound_kg
from maple_syrup.pond_bed import PondBedError, pickup_transport_deposit, require_supported_bed, steepest_d4_descent
from maple_syrup.pond_transport import face_network_from_pond_topology, pond_transport_step
from maple_syrup.sediment_bed import bed_inventory
from maple_syrup.sediment_transport import TransportStepRejected, water_demand_from_transport

DEPTH = 0.03


def basin_case():
    bed, ctx = make_bed(basin(), depth=DEPTH)
    topo = make_topology(bed, [(1, 4, 0, 1)])
    net = face_network_from_pond_topology(topo)
    vx = np.zeros((3, 6))
    vy = np.zeros((4, 5))
    vx[1, 3], vx[1, 2] = 2.0e-4, -1.0e-4  # (1,2) sends 2/3 east and 1/3 west
    vx[1, 4], vx[1, 5] = 2.0e-4, 2.0e-4   # (1,3) -> (1,4) -> outlet (explicit export face)
    return bed, ctx, topo, net, vx, vy


def run_transaction(bed, ctx, net, vx, vy, *, v, rate=0.5, dt=2.0, n_sub=8, request_scale=1.0):
    holder = {}
    layer = host(bed.active_layer.mass_kg)[1, 2]
    requested = np.zeros((3, 5, NC))
    requested[1, 2] = layer * request_scale
    mobile0 = bed.water.mobile_mass_by_cell_class_kg
    phase = empty_phase_state((3, 5), NC, 8, DX)
    zeros = np.zeros_like(requested)

    def transport(pickup):
        pts = pond_transport_step(net, mobile0, phase, pickup.actual_removal_by_cell_class_kg,
                                  np.full((3, 5, NC), v), np.full((3, 5, NC), rate), np.zeros((3, 5, NC), bool), vx, vy, dt,
                                  n_substeps=n_sub)
        holder["pts"] = pts
        return pts.transport.mobile_after_transfer_kg, water_demand_from_transport(pts.transport, zeros)

    out = pickup_transport_deposit(bed, ctx, bed.water.depth_m, requested, transport)
    return (*out, holder["pts"], requested)


def test_multiface_split_deposition_and_overflow_export_close_on_the_native_bed():
    bed, ctx, topo, net, vx, vy = basin_case()
    after, pickup, deposit, pts, requested = run_transaction(bed, ctx, net, vx, vy, v=1.0)
    actual = host(pickup.actual_removal_by_cell_class_kg)
    np.testing.assert_allclose(actual[1, 2], requested[1, 2], rtol=0.0, atol=1e-12)  # fully available here
    # the first crossing out of (1,2) splits by the actual face volumes 2 : 1 (east : west), whatever the decay
    east = host(pts.face_crossing_by_direction_kg)[1, 2, :, 0].sum()
    west = host(pts.face_crossing_by_direction_kg)[1, 2, :, 2].sum()
    assert east > 0.0 and west > 0.0 and east == pytest.approx(2.0 * west, rel=1e-12)
    # multi-face deposition: the actual native deposition touches several cells of the basin
    dep_cells = host(deposit.deposition_by_cell_class_kg).sum(axis=-1)
    assert np.count_nonzero(dep_cells > 1e-15) >= 3
    # overflow export through the explicit outlet, native and equal to the request up to MAPLE's sub-resolution residual
    export = host(deposit.boundary_export_by_class_kg)
    request_export = host(pts.transport.export_request_by_class_kg)
    assert export.sum() > 0.0
    np.testing.assert_allclose(export, request_export, rtol=0.0, atol=NC * 15 * ctx.mass_resolution_kg + 1e-12)
    assert host(pts.blocked_cell)[1, 1]  # the west receiver has no outflow: its mass waits, nothing is invented
    # global class closure with MAPLE's own bound: bed + mobile before == bed + mobile after + native export
    before = host(bed_inventory(bed)) + host(bed.water.mobile_mass_by_cell_class_kg).sum(axis=(0, 1))
    final = host(bed_inventory(after)) + host(after.water.mobile_mass_by_cell_class_kg).sum(axis=(0, 1))
    bound = reservoir_bound_kg(2, before, final, export)
    np.testing.assert_allclose(before, final + export, rtol=0.0, atol=bound)
    # phase reconciliation against the ACTUAL pool uses the existing rules and stays within their bound
    from maple_syrup.sediment_event import _PhaseReconcileInputs, _reconcile_phase_inputs

    inputs = _PhaseReconcileInputs(False, pts.transport.mobile_after_transfer_kg, pts.transport.deposition_request_kg,
                                   pts.transport.export_request_kg, pts.mobile_remaining_kg, pts.phase_after)
    phase, diag = _reconcile_phase_inputs(inputs, after.water.mobile_mass_by_cell_class_kg, np)
    assert float(diag["residual_max_kg"]) <= 1e-9 and phase.n_bins == 8


def test_requested_pickup_is_not_the_actual_pickup_and_transport_uses_the_actual():
    bed, ctx, topo, net, vx, vy = basin_case()
    after, pickup, deposit, pts, requested = run_transaction(bed, ctx, net, vx, vy, v=1.0, request_scale=1.0e3)
    requested_total, actual_total = requested.sum(), host(pickup.actual_removal_by_cell_class_kg).sum()
    assert actual_total <= 1.0e-3 * requested_total * (1.0 + 1e-9)  # MAPLE caps at the active-layer holdings
    assert requested_total / actual_total > 900.0
    np.testing.assert_allclose(host(pts.transport.mobile_before_by_class_kg),
                               host(pickup.actual_removal_by_class_kg) + host(bed.water.mobile_mass_by_cell_class_kg).sum((0, 1)),
                               rtol=1e-13, atol=0.0)


def test_transport_rejection_inside_the_native_transaction_leaves_the_bed_untouched():
    bed, ctx, topo, net, vx, vy = basin_case()
    before = arrays_of(bed)
    with pytest.raises(TransportStepRejected, match="Courant"):
        run_transaction(bed, ctx, net, vx, vy, v=50.0, n_sub=1)  # v dt / dx = 200 > 0.5: a recoverable rejection
    assert_unchanged(before, bed)
    assert not np.any(host(bed.water.mobile_mass_by_cell_class_kg))


def test_unsupported_beds_are_refused_before_any_mutation():
    from dataclasses import replace

    from maple.core.parameters.avalanching import AvalanchingSpec

    bed, ctx = make_bed(plane(), depth=0.003)
    with pytest.raises(PondBedError, match="avalanching"):
        require_supported_bed(bed, replace(ctx, avalanching_spec=AvalanchingSpec(enabled=True)))
    with pytest.raises(PondBedError):
        require_supported_bed(object(), ctx)
    require_supported_bed(bed, ctx)


def full_bed(z, ring):
    out = np.full((z.shape[0] + 2, z.shape[1] + 2), ring)
    out[1:-1, 1:-1] = z
    return out


def test_steepest_d4_descent_uses_the_real_held_bed():
    from maple_syrup.pond_network import OutletFace

    # flat: no direction, slope 0, aspect 1
    z = np.full((3, 3), 0.2)
    slope, aspect = steepest_d4_descent(full_bed(z, 0.0), np.ones((3, 3), bool), DX, ())
    assert not slope.any() and np.all(aspect == 1)  # the closed ring (value 0.0) is NOT terrain
    # falling east by 0.02 per cell: S = 0.04, aspect 2 (east); the last column drops to the ring across its outlet
    z = np.array([[0.24, 0.22, 0.20]] * 3)
    full = full_bed(z, 0.0)
    full[1:-1, -1] = 0.18
    outlets = tuple(OutletFace(r, 2, 0, 1) for r in range(3))
    slope, aspect = steepest_d4_descent(full, np.ones((3, 3), bool), DX, outlets)
    np.testing.assert_allclose(slope, 0.04, rtol=1e-12)
    assert np.all(aspect == 2)
    # without the outlet the last column has no usable lower neighbour: flat (the ring is closed)
    slope, aspect = steepest_d4_descent(full, np.ones((3, 3), bool), DX, ())
    assert not slope[:, 2].any() and np.all(aspect[:, 2] == 1) and np.allclose(slope[:, :2], 0.04)
    # an inactive hole is not terrain: the cell above it ignores the (lower) hole
    z = np.array([[0.2, 0.2, 0.2], [0.2, 0.0, 0.2], [0.2, 0.2, 0.2]])
    active = np.ones((3, 3), bool)
    active[1, 1] = False
    slope, aspect = steepest_d4_descent(full_bed(z, 0.2), active, DX, ())
    assert not slope.any() and aspect[1, 1] == 0
    # steepest wins and equal drops keep the earlier code (north before east)
    z = np.array([[0.2, 0.2, 0.2], [0.2, 0.3, 0.2], [0.2, 0.2, 0.2]])
    slope, aspect = steepest_d4_descent(full_bed(z, 0.2), np.ones((3, 3), bool), DX, ())
    assert slope[1, 1] == pytest.approx(0.1 / DX) and aspect[1, 1] == 1


def test_friction_slope_gives_shear_sqrt_f_over_8_times_u_and_keeps_the_rain_slope_topographic():
    from maple_syrup.sediment_physics import (
        GRAVITY_M_S2,
        physics_grid,
        plot1_sediment_parameters,
        sediment_physics_step,
    )

    params = plot1_sediment_parameters()
    ny, nx = 2, 3
    h = np.array([[0.004, 0.0, 0.01], [0.002, 0.006, 0.003]])
    u = np.array([[0.05, 0.0, 0.2], [0.01, 0.1, 0.0]])
    f = 5.0
    rain = np.full((ny, nx), 5.0e-6)
    veg = np.zeros((ny, nx))
    mass = np.ones((ny, nx, NC)) * 0.1
    memory = np.zeros((ny, nx, NC))
    flat = physics_grid(np.zeros((ny, nx)), np.ones((ny, nx), bool), DX * DX)
    sloped = physics_grid(np.full((ny, nx), 0.04), np.ones((ny, nx), bool), DX * DX)
    wet = h > 0.0
    sf = np.where(wet, f * u * u / (8.0 * GRAVITY_M_S2 * np.where(wet, h, 1.0)), 0.0)

    default = sediment_physics_step(params, sloped, h, u, rain, veg, mass, memory, 1.0)
    explicit = sediment_physics_step(params, sloped, h, u, rain, veg, mass, memory, 1.0,
                                     shear_slope=sloped.slope, stream_power_slope=sloped.slope)
    for name in ("requested_pickup_kg", "sediment_velocity_m_s", "deposition_rate_per_m", "shear_velocity_m_s",
                 "stream_power_w_m2", "raindrop_pickup_kg"):
        np.testing.assert_array_equal(getattr(default, name), getattr(explicit, name), err_msg=name)  # default path

    friction = sediment_physics_step(params, sloped, h, u, rain, veg, mass, memory, 1.0, shear_slope=sf,
                                     stream_power_slope=sf)
    np.testing.assert_allclose(friction.shear_velocity_m_s[wet], np.sqrt(f / 8.0) * u[wet], rtol=1e-12)
    np.testing.assert_allclose(friction.stream_power_w_m2[wet], 1000.0 * f * u[wet] ** 3 / 8.0, rtol=1e-12)
    # the raindrop law keeps the TOPOGRAPHIC slope: identical raindrop pickup under either shear choice
    np.testing.assert_array_equal(friction.raindrop_pickup_kg, default.raindrop_pickup_kg)
    # on a flat the legacy estimate has no shear at all, the friction slope does; rain detachment is 0 by the slope law
    legacy_flat = sediment_physics_step(params, flat, h, u, rain, veg, mass, memory, 1.0)
    friction_flat = sediment_physics_step(params, flat, h, u, rain, veg, mass, memory, 1.0, shear_slope=sf)
    moving = wet & (u > 0.0)
    assert not legacy_flat.shear_velocity_m_s.any() and friction_flat.shear_velocity_m_s[moving].min() > 0.0
    assert not friction_flat.raindrop_pickup_kg.any()
    from maple_syrup.sediment_physics import SedimentPhysicsError

    with pytest.raises(SedimentPhysicsError):
        sediment_physics_step(params, sloped, h, u, rain, veg, mass, memory, 1.0, shear_slope=-sf - 1.0)
