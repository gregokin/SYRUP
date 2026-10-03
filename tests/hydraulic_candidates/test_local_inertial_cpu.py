"""EXPERIMENTAL uniform-grid local inertia: CPU reference tests (no GPU, no Numba).

References are scalar hand computations of the documented face equation, the 1-D steady Darcy-Weisbach law
v^2 = 8 g h S / f, a lake at rest on a sloping bed, explicit python loops over faces for conservation, and the single-cell /
multiple-outflow limiter identities. Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import numpy as np
import pytest
from cand_cases import DX, build, host, make_params, param_arrays, schedule
from test_routing import make_graph, valley_full

pytest.importorskip("maple")

from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.experimental_hydrology import (
    LIMITER_SAFETY,
    CpuHydraulicSolver,
    ExperimentalGeometryError,
    ExperimentalHydrologyError,
    HydraulicControl,
    HydraulicState,
    HydraulicStepRejected,
    build_local_inertial_geometry,
    local_inertial_face_flux,
    open_faces_from_graph,
)
from maple_syrup.experimental_storm import ExperimentalControl, evolve_experimental
from maple_syrup.rainfall import rainfall_field
from maple_syrup.routing import GRAVITY_M_S2 as G

AREA = DX * DX


def scalar_face(z_a, z_b, h_a, h_b, q_old, fric, dt, dx=DX):
    """Hand computation of one interior face, straight from the documented equation."""
    eta_a, eta_b = z_a + h_a, z_b + h_b
    hf = max(max(eta_a, eta_b) - max(z_a, z_b), 0.0)
    if hf <= 0.0:
        return 0.0, hf
    numerator = q_old - G * hf * dt * (eta_b - eta_a) / dx
    return numerator / (1.0 + dt * (fric / 8.0) * abs(q_old) / hf ** 2), hf


# --- the face equation -------------------------------------------------------------------------------------------------
def test_face_update_matches_a_hand_computation():
    z_a, z_b, h_a, h_b, q_old, fric, dt = 1.0, 0.99, 0.01, 0.004, 0.003, 20.0, 0.2
    expected, hf = scalar_face(z_a, z_b, h_a, h_b, q_old, fric, dt)
    q, face_depth = local_inertial_face_flux(
        np.array([1], dtype=np.int8), np.array([max(z_a, z_b)]), np.array([fric]), np.array([0.0]),
        np.array([0], dtype=np.int8), np.array([q_old]), np.array([h_a]), np.array([h_b]), np.array([z_a + h_a]),
        np.array([z_b + h_b]), dt, DX)
    assert q[0] == pytest.approx(expected, rel=1e-14) and face_depth[0] == pytest.approx(hf, rel=1e-14)


@pytest.mark.parametrize("fric, slope, depth", [(21.45, 0.05, 0.01), (5.0, 0.02, 0.003), (0.1, 0.3, 0.002)])
def test_steady_uniform_flow_obeys_the_darcy_weisbach_law(fric, slope, depth):
    """Iterating the face update at fixed depth and surface slope must converge to v^2 = 8 g h S / f (the legacy law)."""
    z_a, z_b = 1.0, 1.0 - slope * DX
    q = np.zeros(1)
    for _ in range(4000):
        q, _hf = local_inertial_face_flux(
            np.array([1], dtype=np.int8), np.array([z_a]), np.array([fric]), np.array([0.0]), np.array([0], dtype=np.int8),
            q, np.array([depth]), np.array([depth]), np.array([z_a + depth]), np.array([z_b + depth]), 0.2, DX)
    v_expected = np.sqrt(8.0 * G * depth * slope / fric)
    assert q[0] / depth == pytest.approx(v_expected, rel=1e-9)
    k = np.sqrt(8.0 * G * slope / fric)  # the same law in the legacy conveyance form q = k h^(3/2)
    assert q[0] == pytest.approx(k * depth ** 1.5, rel=1e-9)


def test_dry_face_carries_nothing_and_friction_never_divides_by_zero():
    q, hf = local_inertial_face_flux(
        np.array([1, 1], dtype=np.int8), np.array([2.0, 2.0]), np.array([20.0, 20.0]), np.zeros(2),
        np.zeros(2, dtype=np.int8), np.array([0.0, 0.5]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]),
        np.array([1.5, 1.5]), 0.1, DX)  # both sides below the sill: dry face, even with an old momentum
    assert np.all(q == 0.0) and np.all(hf == 0.0)


# --- closed-domain physics -----------------------------------------------------------------------------------------------
def test_lake_at_rest_on_a_sloping_bed_stays_at_rest():
    cs = build("valley:8x7", "local_inertial", closed=True, depth="dry")
    z = np.asarray(cs.geometry.z)
    eta0 = float(np.median(z))
    depth = np.maximum(eta0 - z, 0.0)
    assert 0.2 < np.count_nonzero(depth) / depth.size < 0.8  # a genuine wet/dry shoreline
    state = cs.solver.initial_state(depth, np.zeros(cs.shape))
    volume0 = float(depth.sum()) * AREA
    for _ in range(60):
        step = cs.solver.step(np.zeros(cs.shape), state, 0.05)
        state = step.state
    assert float(np.max(np.abs(state.qx_m2_s))) < 1e-12 and float(np.max(np.abs(state.qy_m2_s))) < 1e-12
    np.testing.assert_allclose(state.depth_m, depth, rtol=0.0, atol=1e-14)
    assert float(state.depth_m.sum()) * AREA == pytest.approx(volume0, rel=1e-14)
    assert float(state.depth_m.min()) >= 0.0 and not np.any(state.depth_m[depth == 0.0] != 0.0)  # dry bank stays dry


def test_backwater_flow_runs_against_the_bed_slope_and_conserves_volume():
    cs = build("chain:3", "local_inertial", closed=True, depth="dry")
    z = np.asarray(cs.geometry.z)[:, 0]
    depth = np.array([[0.02], [0.001], [0.001]])  # deep water at the DOWNSTREAM (south) end
    state = cs.solver.initial_state(depth, np.zeros(cs.shape))
    dt = 0.05
    step = cs.solver.step(np.zeros(cs.shape), state, dt)
    expected, _ = scalar_face(z[0], z[1], depth[0, 0], depth[1, 0], 0.0, float(cs.geometry.fy_fric[1, 0]), dt)
    assert expected > 0.0 and step.state.qy_m2_s[1, 0] == pytest.approx(expected, rel=1e-13)  # positive = north, uphill
    assert float(step.state.depth_m.sum()) * AREA == pytest.approx(float(depth.sum()) * AREA, rel=1e-14)
    assert step.state.depth_m[0, 0] < depth[0, 0] and step.state.depth_m[1, 0] > depth[1, 0]
    assert step.export_m3 == 0.0 and step.outlet_discharge_m3_s == 0.0  # no open face in a closed domain
    assert np.all(step.state.qx_m2_s == 0.0)  # the walls (x faces) are closed


def test_closed_box_keeps_every_drop_of_rain_and_infiltration_budget_closes():
    cs = build("valley:6x5", "local_inertial", closed=True, depth="dry", ksat=0.0)
    ny, nx = cs.shape
    field = rainfall_field(ny, nx, scale=np.where(cs.graph.active, 1.0, 0.0))
    sched = schedule([0, 40], [100.0])
    result = evolve_experimental(cs.solver, field, sched, cs.state, 40.0, ExperimentalControl(), report_every_s=20.0)
    expected = sched.depth_m(0.0, 40.0) * cs.graph.active.sum() * AREA
    assert float(np.sum(result.state.depth_m)) * AREA == pytest.approx(expected, rel=1e-13)  # no infiltration, closed
    assert float(result.cumulative_export_m3) == 0.0
    # saturated start: intake (Ksat dt) > drainage (0.5 Ksat dt), so the surplus is forced back to the surface
    wet = build("valley:6x5", "local_inertial", closed=True, depth="dry", ksat=3e-6, soil_fraction=1.0, drainage=0.5)
    out = evolve_experimental(wet.solver, field, sched, wet.state, 40.0, ExperimentalControl(), report_every_s=20.0)

    def vol(a):
        return float(np.sum(host(a))) * AREA

    initial = vol(wet.state.depth_m) + vol(wet.state.soil_water_m)
    rain, drain = vol(out.cumulative_rain_m), vol(out.cumulative_drainage_m)
    final = vol(out.state.depth_m) + vol(out.state.soil_water_m)
    bound = volume_roundoff_bound_m3(4 * ny * nx * out.n_accepted_steps + 7, max(initial, rain, drain, final))
    assert abs(final + drain - initial - rain) <= bound
    assert vol(out.cumulative_saturation_return_m) > 0.0 and drain > 0.0  # overflow and drainage both exercised


# --- open outlets: shared faces and export ------------------------------------------------------------------------------
def divergence_by_loops(qx, qy, ny, nx):
    """Net outflow volume per cell per unit (dt dx), summed face by face in plain python (independent of the vectorised form)."""
    net = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            net[i, j] = (qx[i, j + 1] - qx[i, j]) + (qy[i + 1, j] - qy[i, j])
    return net


def test_every_face_is_shared_and_the_export_is_counted_once():
    cs = build("valley:8x7", "local_inertial", seed=3)
    ny, nx = cs.shape
    dt, state, total_export = 0.1, cs.state, 0.0
    for _ in range(8):
        step = cs.solver.step(np.zeros(cs.shape), state, dt)
        net = divergence_by_loops(step.state.qx_m2_s, step.state.qy_m2_s, ny, nx)
        np.testing.assert_allclose((state.depth_m - step.state.depth_m) * AREA, dt * DX * net, rtol=1e-12, atol=1e-18)
        outward = 0.0
        for r, c, dr, dc, _drop, _k in cs.geometry.open_faces:
            face = (step.state.qx_m2_s[r, c + (1 if dc > 0 else 0)] if dc != 0
                    else step.state.qy_m2_s[r + (1 if dr > 0 else 0), c])
            outward += face * (dc + dr)  # +1 east/north, -1 west/south: outward flux positive
        assert step.export_m3 == pytest.approx(dt * DX * outward, rel=1e-12, abs=1e-18)
        assert step.outlet_discharge_m3_s == pytest.approx(DX * outward, rel=1e-12, abs=1e-18)
        assert float((step.state.depth_m - state.depth_m).sum()) * AREA + step.export_m3 == pytest.approx(0.0, abs=1e-15)
        total_export += step.export_m3
        state = step.state
    assert total_export > 0.0


def test_only_the_legacy_outlet_faces_are_open_and_every_other_face_stays_closed():
    cs = build("valley:8x7", "local_inertial", seed=3)
    g = cs.geometry
    assert len(g.open_faces) == int(np.asarray(cs.graph.outlet).sum()) == int(np.sum(g.fx_type == 2) + np.sum(g.fy_type == 2))
    assert np.all(g.fy_sign[g.fy_type == 2] == -1)  # the legacy outlets of this terrain drain south
    state = cs.state
    for _ in range(5):
        state = cs.solver.step(np.zeros(cs.shape), state, 0.1).state
    assert not np.any(state.qx_m2_s[g.fx_type == 0]) and not np.any(state.qy_m2_s[g.fy_type == 0])
    assert np.all(state.qy_m2_s[g.fy_type == 2] * g.fy_sign[g.fy_type == 2] >= 0.0)  # outflow only, never inflow
    assert g.summary()["n_open_faces"] == len(g.open_faces) and "normal-flow" in g.summary()["boundary"]


def test_inactive_cells_keep_their_water_and_exchange_nothing():
    cs = build("valley_masked", "local_inertial", seed=4, soil_fraction=0.3)
    inactive = ~cs.graph.active
    depth = np.where(inactive, 0.05, cs.state.depth_m)
    state = cs.solver.initial_state(depth, cs.state.soil_water_m)
    for _ in range(6):
        state = cs.solver.step(np.zeros(cs.shape), state, 0.05).state
    np.testing.assert_array_equal(state.depth_m[inactive], depth[inactive])
    np.testing.assert_array_equal(state.soil_water_m[inactive], cs.state.soil_water_m[inactive])


# --- wet/dry, rejection and the donor limiter ---------------------------------------------------------------------------
def steep_case(limiter):
    z = valley_full(5, 5, sy=0.2, sx=0.4)
    graph = make_graph(z, ff=0.1)
    params = make_params(param_arrays(graph.shape), graph.active)
    geometry = build_local_inertial_geometry(z, np.asarray(graph.active), np.asarray(graph.friction_factor), graph.dx_m, [])
    solver = CpuHydraulicSolver("local_inertial", graph, params, geometry=geometry,
                                control=HydraulicControl(limiter=limiter))
    return solver, geometry


def test_without_the_limiter_a_draining_cell_rejects_the_step_instead_of_clipping():
    solver, _ = steep_case("off")
    depth = np.zeros((5, 5))
    depth[2, 2] = 2e-4
    state = solver.initial_state(depth, np.zeros((5, 5)))
    before = depth.copy()
    with pytest.raises(HydraulicStepRejected, match="negative depth") as info:
        solver.step(np.zeros((5, 5)), state, 1.0)  # wave CFL ~0.13 is fine; the cell would lose 7x what it holds
    assert type(info.value) is HydraulicStepRejected
    np.testing.assert_array_equal(state.depth_m, before)  # nothing was clipped or modified
    assert solver.step(np.zeros((5, 5)), state, 0.05).state.depth_m.min() >= 0.0  # a smaller step is accepted


def test_donor_limiter_single_outflow_is_conservative_and_both_ends_see_one_flux():
    solver, g = steep_case("donor")
    depth = np.zeros((5, 5))
    depth[2, 2] = 2e-4
    state = solver.initial_state(depth, np.zeros((5, 5)))
    dt = 1.0
    step = solver.step(np.zeros((5, 5)), state, dt)
    z = np.asarray(g.z)
    q_unlimited, _ = scalar_face(z[1, 2], z[2, 2], 0.0, 2e-4, 0.0, float(g.fy_fric[2, 2]), dt)  # the only wet face (south)
    assert q_unlimited < 0.0  # southward
    out_volume, available = dt * DX * abs(q_unlimited), 2e-4 * AREA
    assert out_volume > available  # the limiter must bind in this construction
    assert step.limited_cells == 1
    assert step.limited_volume_m3 == pytest.approx(out_volume - available * LIMITER_SAFETY, rel=1e-9)
    assert 0.0 <= step.state.depth_m[2, 2] <= 1e-13  # the donor keeps only the 16 eps safety margin
    assert float(step.state.depth_m.sum()) * AREA == pytest.approx(2e-4 * AREA, rel=1e-13)  # exactly conservative
    face_volume = step.face_volume_m3["y"][2, 2]
    assert face_volume < 0.0  # southward
    assert step.state.depth_m[1, 2] * AREA == pytest.approx(-face_volume, rel=1e-13)  # the receiver gains that SAME volume
    assert step.state.qy_m2_s[2, 2] * dt * DX == pytest.approx(face_volume, rel=1e-15)  # state momentum = flux applied once


def test_donor_limiter_scales_every_outgoing_face_of_a_cell_by_the_same_factor():
    solver, g = steep_case("donor")
    depth = np.zeros((5, 5))
    depth[2, 1] = 2e-4  # two lower neighbours: east (axis cell) and south
    state = solver.initial_state(depth, np.zeros((5, 5)))
    dt = 1.0
    step = solver.step(np.zeros((5, 5)), state, dt)
    z = np.asarray(g.z)
    q_east, _ = scalar_face(z[2, 1], z[2, 2], 2e-4, 0.0, 0.0, float(g.fx_fric[2, 2]), dt)
    q_south, _ = scalar_face(z[1, 1], z[2, 1], 0.0, 2e-4, 0.0, float(g.fy_fric[2, 1]), dt)
    assert q_east > 0.0 and q_south < 0.0  # both leave the cell
    phi_east, phi_south = step.state.qx_m2_s[2, 2] / q_east, step.state.qy_m2_s[2, 1] / q_south
    assert phi_east == pytest.approx(phi_south, rel=1e-13) and 0.0 < phi_east < 1.0  # one factor for both
    assert step.limited_cells == 1
    for receiver, volume in (((2, 2), step.face_volume_m3["x"][2, 2]), ((1, 1), -step.face_volume_m3["y"][2, 1])):
        assert step.state.depth_m[receiver] * AREA == pytest.approx(volume, rel=1e-13)  # each receiver gains its face
    assert float(step.state.depth_m.sum()) * AREA == pytest.approx(2e-4 * AREA, rel=1e-13)
    assert step.state.depth_m.min() >= 0.0


def test_the_limiter_is_inactive_when_no_cell_overdraws_and_reports_zero():
    solver, _ = steep_case("donor")
    depth = np.zeros((5, 5))
    depth[2, 2] = 2e-4
    state = solver.initial_state(depth, np.zeros((5, 5)))
    step = solver.step(np.zeros((5, 5)), state, 0.05)
    assert step.limited_cells == 0 and step.limited_volume_m3 == 0.0


# --- geometry and state refusals ---------------------------------------------------------------------------------------
def test_geometry_refusals():
    cs = build("valley:6x5", "local_inertial", closed=True)
    z, active, f = cs.z_full, np.asarray(cs.graph.active), np.asarray(cs.graph.friction_factor)
    faces = open_faces_from_graph(cs.graph)
    ok = build_local_inertial_geometry(z, active, f, DX, faces)
    assert len(ok.open_faces) == len(faces) and ok.fy_sign.dtype == np.int8
    with pytest.raises(ExperimentalGeometryError, match="interior"):
        build_local_inertial_geometry(z, active, f, DX, [(2, 2, 0, 1)])  # leads into another interior cell
    with pytest.raises(ExperimentalGeometryError, match="positive bed drop"):
        build_local_inertial_geometry(z, active, f, DX, [(2, 0, 0, -1)])  # the west ring is HIGHER than the cell
    with pytest.raises(ExperimentalGeometryError, match="opened twice"):
        build_local_inertial_geometry(z, active, f, DX, [faces[0], faces[0]])
    bad_active = active.copy()
    bad_active[faces[0][0], faces[0][1]] = False
    with pytest.raises(ExperimentalGeometryError, match="not an active"):
        build_local_inertial_geometry(z, bad_active, f, DX, [faces[0]])
    for bad_z in (z.astype(np.float32), z[:2], np.where(np.arange(z.size).reshape(z.shape) == 5, np.nan, z)):
        with pytest.raises(ExperimentalGeometryError):
            build_local_inertial_geometry(bad_z, active, f, DX, [])
    for bad_dx in (0.0, -1.0, float("nan"), True, "0.5"):
        with pytest.raises(ExperimentalGeometryError):
            build_local_inertial_geometry(z, active, f, bad_dx, [])
    with pytest.raises(ExperimentalGeometryError, match="friction"):
        build_local_inertial_geometry(z, active, np.zeros_like(f), DX, [])
    for array in (ok.z, ok.fx_type, ok.fy_kb):
        assert not array.flags.writeable  # read-only owned geometry


def test_solver_refuses_a_geometry_that_does_not_match_the_graph():
    cs = build("valley:6x5", "local_inertial", closed=True)
    other = build("valley:8x7", "local_inertial", closed=True)
    with pytest.raises(ExperimentalHydrologyError, match="geometry"):
        CpuHydraulicSolver("local_inertial", cs.graph, cs.params, geometry=other.geometry)
    with pytest.raises(ExperimentalHydrologyError, match="LocalInertialGeometry"):
        CpuHydraulicSolver("local_inertial", cs.graph, cs.params)


def test_state_momentum_validation_refuses_invalid_closed_and_open_faces():
    cs = build("valley:6x5", "local_inertial", seed=5)
    g, st = cs.geometry, cs.state
    ny, nx = cs.shape

    def with_faces(qx=None, qy=None):
        return HydraulicState(0.0, st.depth_m, st.soil_water_m, st.qx_m2_s if qx is None else qx,
                              st.qy_m2_s if qy is None else qy)

    closed = np.argwhere(g.fx_type == 0)[0]
    qx = st.qx_m2_s.copy()
    qx[tuple(closed)] = 0.1
    r, c, dr, _dc, _drop, _k = g.open_faces[0]
    qy = st.qy_m2_s.copy()
    qy[r + (1 if dr > 0 else 0), c] = -g.fy_sign[r + (1 if dr > 0 else 0), c] * 0.1  # points INTO the domain
    nan_x = st.qx_m2_s.copy()
    nan_x[1, 1] = np.nan
    bad = {"closed_face": with_faces(qx=qx), "open_inflow": with_faces(qy=qy), "nan": with_faces(qx=nan_x),
           "missing": HydraulicState(0.0, st.depth_m, st.soil_water_m), "shape": with_faces(qx=np.zeros((ny, nx))),
           "float32": with_faces(qx=st.qx_m2_s.astype(np.float32))}
    for state in bad.values():
        with pytest.raises(ExperimentalHydrologyError):
            cs.solver.validate_state(state)
        with pytest.raises(ExperimentalHydrologyError):
            cs.solver.step(np.zeros(cs.shape), state, 0.05)
    before = [a.copy() for a in (st.depth_m, st.qx_m2_s, st.qy_m2_s)]
    with pytest.raises(HydraulicStepRejected):
        cs.solver.step(np.zeros(cs.shape), st, 100.0)  # wave CFL rejection from a valid state
    for a, b in zip(before, (st.depth_m, st.qx_m2_s, st.qy_m2_s), strict=True):
        np.testing.assert_array_equal(a, b)


def test_wave_cfl_uses_the_two_dimensional_bound_and_marks_the_admissibility_boundary():
    # Isolate the GRAVITY-WAVE bound from the separate negative-depth rejection: a CLOSED lake at rest (no infiltration, no
    # drainage, no open outflow) cannot overdraw any cell at any dt, so only the wave CFL can reject here.
    cs = build("valley:8x7", "local_inertial", closed=True, depth="dry")
    z = np.asarray(cs.geometry.z)
    state = cs.solver.initial_state(np.maximum(float(np.median(z)) - z, 0.0), np.zeros(cs.shape))
    probe = cs.solver.step(np.zeros(cs.shape), state, 0.01)  # max_cfl = dt/dx sqrt(2 g hf_max): linear in dt for a fixed depth
    assert probe.max_cfl > 0.0
    dt_limit = 0.5 * 0.01 / probe.max_cfl
    inside = cs.solver.step(np.zeros(cs.shape), state, dt_limit * (1.0 - 1e-6))
    assert inside.max_cfl <= 0.5 and float(inside.state.depth_m.min()) >= 0.0
    with pytest.raises(HydraulicStepRejected, match="gravity-wave CFL") as info:
        cs.solver.step(np.zeros(cs.shape), state, dt_limit * (1.0 + 1e-3))
    assert "negative depth" not in str(info.value)  # rejected by the wave bound alone


def test_outputs_are_fresh_and_the_method_contract_is_visible():
    cs = build("random:8x6", "local_inertial", seed=7)
    first = cs.solver.step(np.zeros(cs.shape), cs.state, 0.05)
    assert first.method == "local_inertial" and "gravity-wave" in first.cfl_kind and set(first.face_volume_m3) == {"x", "y"}
    assert not hasattr(first, "root_residual") and "bisection" not in str(first.cfl_kind)
    arrays = [first.state.depth_m, first.state.qx_m2_s, first.state.qy_m2_s, first.velocity_m_s, first.face_volume_m3["x"],
              first.face_volume_m3["y"], cs.state.depth_m, cs.state.qx_m2_s, cs.state.qy_m2_s]
    for i, a in enumerate(arrays):
        for b in arrays[i + 1:]:
            assert not np.shares_memory(a, b)
