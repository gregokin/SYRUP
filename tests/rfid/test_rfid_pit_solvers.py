"""Terminal pit storage through the shared solvers: legacy-style (array / prepared Numba / CUDA), explicit candidate and the
local-inertial candidate (different physics by design). Small synthetic chain with one strict pit. Bounds are the
unchanged ones (rtol 2e-12, atol 1e-14). Nothing here was run by its author (file-only tools)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("maple")

from cand_cases import host, make_params, param_arrays
from rfid_helpers import DX, build, pit_chain

from maple_syrup import hydrology_numba as hn
from maple_syrup.experimental_hydrology import (
    CpuHydraulicSolver,
    build_local_inertial_geometry,
    open_faces_from_graph,
)
from maple_syrup.routing import RoutingError
from maple_syrup.storm import StormControl, coupled_step, initial_state

RTOL, ATOL = 2.0e-12, 1.0e-14
AREA = DX * DX
PIT = (2, 0)
NUMBA = pytest.mark.skipif(not hn.numba_available(), reason="Numba not installed; compiled path not exercised")


def make_case(depth_pit=0.0, depth_up=3.0e-3, xp=np):
    z = pit_chain(6, 3)
    g = build(z, allow_pit_storage=True, xp=xp)
    params = make_params(param_arrays(g.shape, ksat=0.0), np.asarray(g.active), xp=xp)
    depth = np.zeros(g.shape)
    depth[3:, 0] = depth_up  # the cells that drain into the pit
    depth[PIT] = depth_pit
    return z, g, params, depth, np.zeros(g.shape)


def test_legacy_array_steps_store_water_in_the_pit_and_close_the_budget():
    _, g, params, depth, soil = make_case()
    state = initial_state(g, depth, soil)
    control = StormControl(max_dt_s=0.5)
    rate = np.zeros(g.shape)
    export = 0.0
    water0 = depth.sum() * AREA
    last_pit = 0.0
    for _ in range(40):
        step = coupled_step(g, params, rate, state, 0.5, control)
        export += float(step.route.export_m3)
        state = step.state
        assert float(state.discharge_m2_s[PIT]) == 0.0 and float(step.route.velocity_m_s[PIT]) == 0.0
        assert float(step.route.face_volume_m3[PIT]) == 0.0  # nothing leaves the pit
        assert abs(float(step.route.budget_residual_m3)) <= 1e-13
        assert float(state.depth_m[PIT]) >= last_pit - 1e-18  # only grows while donors drain in (no overtopping)
        last_pit = float(state.depth_m[PIT])
    assert last_pit > 0.0
    assert abs(float(np.sum(state.depth_m)) * AREA + export - water0) <= 1e-13


@pytest.mark.parametrize("implementation", ["array", pytest.param("numba", marks=NUMBA)])
def test_deep_pit_needs_enough_bisection_halvings_not_a_wider_tolerance(implementation):
    """Conveyance 0 makes the exact root the right-hand side; 40 halvings leave rhs 2^-40 > the unchanged 1e-11 m root
    tolerance above ~10 m. The tolerance is NOT relaxed: more halvings (an existing control) are used."""
    _, g, params, depth, soil = make_case(depth_pit=20.0, depth_up=0.0)
    state = initial_state(g, depth, soil)
    rate = np.zeros(g.shape)
    with pytest.raises(RoutingError, match="bisection did not reach root_tolerance_m"):
        coupled_step(g, params, rate, state, 0.5, StormControl(max_dt_s=0.5, implementation=implementation))
    ok = coupled_step(g, params, rate, state, 0.5, StormControl(max_dt_s=0.5, implementation=implementation,
                                                                bisection_iterations=64))
    assert float(ok.state.depth_m[PIT]) == 20.0 and float(ok.state.discharge_m2_s[PIT]) == 0.0


@NUMBA
def test_prepared_numba_pit_steps_match_the_array_reference_at_the_unchanged_bounds():
    _, g, params, depth, soil = make_case()
    ctx = hn.prepare_hydrology(g, params)
    control = StormControl(max_dt_s=0.5, bisection_iterations=64)
    ref = initial_state(g, depth, soil)
    new = initial_state(g, depth, soil)
    rate = np.zeros(g.shape)
    for _ in range(30):
        a = coupled_step(g, params, rate, ref, 0.5, control)
        b = hn.prepared_coupled_step(ctx, rate, new, 0.5, StormControl(max_dt_s=0.5, implementation="numba",
                                                                         bisection_iterations=64))
        np.testing.assert_allclose(b.state.depth_m, a.state.depth_m, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(b.state.discharge_m2_s, a.state.discharge_m2_s, rtol=RTOL, atol=ATOL)
        ref, new = a.state, b.state
    assert float(new.depth_m[PIT]) > 0.0 and float(new.discharge_m2_s[PIT]) == 0.0


def test_a_positive_discharge_state_on_a_zero_conveyance_pit_is_refused_not_ignored():
    """q > 0 with k = 0 is inconsistent (q = k h^1.5 cannot hold): the unchanged consistency check sees implied depth infinity."""
    from maple_syrup.storm import StormError

    _, g, params, depth, soil = make_case(depth_pit=0.02)
    state = initial_state(g, depth, soil)
    q = np.array(state.discharge_m2_s, copy=True)
    q[PIT] = 1.0e-3
    forged = type(state)(state.t_s, state.depth_m, state.soil_water_m, q)
    with pytest.raises(RoutingError, match="old_discharge_m2_s does not satisfy"):
        coupled_step(g, params, np.zeros(g.shape), forged, 0.5, StormControl(max_dt_s=0.5))
    from maple_syrup.storm import _validate_state

    with pytest.raises(StormError, match="not consistent"):
        _validate_state(g, forged, 1.0e-11)


def test_explicit_candidate_keeps_pit_water_and_conserves():
    _z, g, params, depth, soil = make_case()
    solver = CpuHydraulicSolver("explicit", g, params)
    state = solver.initial_state(depth, soil)
    step = solver.step(np.zeros(g.shape), state, 0.5)
    out = np.asarray(step.face_volume_m3["out"])
    assert out[PIT] == 0.0 and float(step.velocity_m_s[PIT]) == 0.0
    assert np.asarray(step.state.depth_m)[PIT] == pytest.approx(depth[PIT] + out[3, 0] / AREA, rel=1e-14)
    assert abs(float(step.budget_residual_m3)) <= 1e-13


def test_local_inertia_can_overtop_a_pit_through_the_water_surface_gradient():
    """Intentionally DIFFERENT physics: a pit holding more than the lowest rim height loses water through its pressure gradient."""
    z, g, params, depth, soil = make_case(depth_pit=0.05, depth_up=0.0)
    geometry = build_local_inertial_geometry(z, np.asarray(g.active), np.asarray(g.friction_factor), g.dx_m,
                                             open_faces_from_graph(g))
    solver = CpuHydraulicSolver("local_inertial", g, params, geometry=geometry)
    state = solver.initial_state(depth, soil)
    export = 0.0
    for _ in range(5):
        step = solver.step(np.zeros(g.shape), state, 0.2)
        export += float(step.export_m3)
        state = step.state
    after = np.asarray(state.depth_m)
    assert after[PIT] < 0.05 - 1e-6  # water left the pit although it is a strict D4 sink
    assert after[1, 0] + after[3, 0] > 0.0  # and arrived in its rim neighbours
    assert abs(float(after.sum()) * AREA + export - 0.05 * AREA) <= 1e-13


# ----------------------------------------------------------------------------------------------------------------- CUDA
def test_cuda_legacy_explicit_and_local_inertial_pit_steps_match_the_host(gpu):
    cp = gpu
    from maple_syrup.experimental_cuda import CudaHydraulicSolver

    z, g, params, depth, soil = make_case(depth_pit=0.02, depth_up=2.0e-3)
    _, gd, pd, _, _ = make_case(depth_pit=0.02, depth_up=2.0e-3, xp=cp)
    # legacy prepared CUDA vs the array reference (64 halvings both)
    control_h = StormControl(max_dt_s=0.5, bisection_iterations=64)
    control_d = StormControl(max_dt_s=0.5, implementation="cuda", bisection_iterations=64)
    sh, sd = initial_state(g, depth, soil), initial_state(gd, cp.asarray(depth), cp.asarray(soil))
    for _ in range(10):
        a = coupled_step(g, params, np.zeros(g.shape), sh, 0.5, control_h)
        b = coupled_step(gd, pd, cp.zeros(g.shape), sd, 0.5, control_d)
        np.testing.assert_allclose(host(b.state.depth_m), host(a.state.depth_m), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(host(b.state.discharge_m2_s), host(a.state.discharge_m2_s), rtol=RTOL, atol=ATOL)
        sh, sd = a.state, b.state
    assert float(host(sd.discharge_m2_s)[PIT]) == 0.0
    # candidates: explicit and local inertia, host reference vs device
    geometry = build_local_inertial_geometry(z, np.asarray(g.active), np.asarray(g.friction_factor), g.dx_m,
                                             open_faces_from_graph(g))
    for method, geo in (("explicit", None), ("local_inertial", geometry)):
        ref = CpuHydraulicSolver(method, g, params, geometry=geo)
        dev = CudaHydraulicSolver(method, gd, pd, geometry=geo)
        s_ref = ref.initial_state(depth, soil)
        s_dev = type(s_ref)(0.0, cp.asarray(depth), cp.asarray(soil),
                            None if s_ref.qx_m2_s is None else cp.asarray(s_ref.qx_m2_s),
                            None if s_ref.qy_m2_s is None else cp.asarray(s_ref.qy_m2_s))
        for _ in range(5):
            r = ref.step(np.zeros(g.shape), s_ref, 0.2)
            d = dev.step(cp.zeros(g.shape), s_dev, 0.2)
            np.testing.assert_allclose(host(d.state.depth_m), r.state.depth_m, rtol=RTOL, atol=ATOL)
            s_ref, s_dev = r.state, d.state
