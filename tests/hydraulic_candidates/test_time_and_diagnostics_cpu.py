"""Strict state-time contract of the CPU reference/driver and the stage-consistent face diagnostics (CPU only, no Numba).

Time: `state.t_s` must be a real (not bool/str/None) finite number >= 0; for dt > 0 the end time must be finite and must
advance floating time; all of it is checked BEFORE any column work and nothing is mutated; continuation validates it too.
Diagnostics: the reconstructed local-inertial cell speed (face-averaged flux / END depth) is unbounded near a draining cell
while the stage-consistent face velocity (flux / the depth it was computed from) is not; nothing is clipped. Nothing here was
run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import numpy as np
import pytest
from cand_cases import DX, build, field, make_params, param_arrays, schedule
from test_routing import make_graph, valley_full

pytest.importorskip("maple")

from maple_syrup.experimental_hydrology import (
    CpuHydraulicSolver,
    ExperimentalHydrologyError,
    HydraulicControl,
    HydraulicState,
    advance_time,
    build_local_inertial_geometry,
    check_state_time,
    stage_face_diagnostics,
)
from maple_syrup.experimental_storm import ExperimentalControl, evolve_experimental
from maple_syrup.infiltration import InfiltrationError
from maple_syrup.routing import GRAVITY_M_S2 as G

METHODS = ("explicit", "local_inertial")
BAD_TIMES = [-1.0, -1e-300, float("nan"), float("inf"), -float("inf"), True, False, "1", None, b"1", [0.0]]


@pytest.mark.parametrize("bad", BAD_TIMES, ids=repr)
def test_the_time_checker_refuses_everything_but_a_finite_nonnegative_real(bad):
    with pytest.raises(ExperimentalHydrologyError, match="state.t_s"):
        check_state_time(bad)


@pytest.mark.parametrize("good, expected", [(0.0, 0.0), (0, 0.0), (7, 7.0), (np.float64(2.5), 2.5), (np.int64(3), 3.0)])
def test_the_time_checker_returns_a_float(good, expected):
    value = check_state_time(good)
    assert value == expected and type(value) is float


def test_advance_time_refuses_overflow_and_a_step_that_does_not_advance():
    assert advance_time(1.0, 0.5) == 1.5
    with pytest.raises(ExperimentalHydrologyError, match="overflows"):
        advance_time(1.7e308, 1.7e308)
    with pytest.raises(ExperimentalHydrologyError, match="does not advance"):
        advance_time(1e20, 1.0)
    with pytest.raises(ExperimentalHydrologyError, match="does not advance"):
        advance_time(1.0, 5e-324)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", BAD_TIMES, ids=repr)
def test_initial_state_refuses_a_bad_time(method, bad):
    cs = build("valley:6x5", method, seed=1)
    with pytest.raises(ExperimentalHydrologyError):
        cs.solver.initial_state(cs.state.depth_m, cs.state.soil_water_m, t_s=bad)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", BAD_TIMES, ids=repr)
def test_step_refuses_a_bad_state_time_before_any_column_work_and_mutates_nothing(method, bad, monkeypatch):
    cs = build("valley:6x5", method, seed=2)
    st = cs.state
    broken = HydraulicState(bad, st.depth_m, st.soil_water_m, st.qx_m2_s, st.qy_m2_s)
    before = [a.copy() for a in (st.depth_m, st.soil_water_m)]
    from maple_syrup import experimental_hydrology as eh

    monkeypatch.setattr(eh, "column_step", lambda *a, **k: pytest.fail("column work ran before the time check"))
    with pytest.raises(ExperimentalHydrologyError):
        cs.solver.step(np.zeros(cs.shape), broken, 0.1)
    for a, b in zip(before, (st.depth_m, st.soil_water_m), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("t, dt, match", [(1.7e308, 1.7e308, "overflows"), (1e20, 1.0, "does not advance"),
                                          (1.0, 5e-324, "does not advance")])
def test_overflowing_or_non_advancing_steps_are_refused_before_column_work(method, t, dt, match, monkeypatch):
    cs = build("valley:6x5", method, seed=3)
    st = cs.state
    from maple_syrup import experimental_hydrology as eh

    monkeypatch.setattr(eh, "column_step", lambda *a, **k: pytest.fail("column work ran before the time check"))
    with pytest.raises(ExperimentalHydrologyError, match=match):
        cs.solver.step(np.zeros(cs.shape), HydraulicState(t, st.depth_m, st.soil_water_m, st.qx_m2_s, st.qy_m2_s), dt)


@pytest.mark.parametrize("method", METHODS)
def test_error_precedence_time_then_dt_class_then_column_then_dt_zero(method):
    cs = build("valley:6x5", method, seed=4)
    st = cs.state
    nan_soil = st.soil_water_m.copy()
    nan_soil[2, 2] = np.nan
    both = HydraulicState(-1.0, st.depth_m, nan_soil, st.qx_m2_s, st.qy_m2_s)
    with pytest.raises(ExperimentalHydrologyError, match="state.t_s"):  # a bad time outranks a bad column input
        cs.solver.step(np.zeros(cs.shape), both, 0.1)
    for bad_dt in ("1", True, float("nan"), -1.0):  # dt problems keep the accepted column stage's class and message
        with pytest.raises(InfiltrationError):
            cs.solver.step(np.zeros(cs.shape), st, bad_dt)
    with pytest.raises(InfiltrationError):  # then the column input
        cs.solver.step(np.zeros(cs.shape), HydraulicState(0.0, st.depth_m, nan_soil, st.qx_m2_s, st.qy_m2_s), 0.1)
    with pytest.raises(ExperimentalHydrologyError, match="dt = 0 is rejected"):  # dt = 0 last, after the column stage
        cs.solver.step(np.zeros(cs.shape), st, 0.0)
    with pytest.raises(InfiltrationError):  # ... so a column failure outranks it
        cs.solver.step(np.zeros(cs.shape), HydraulicState(0.0, st.depth_m, nan_soil, st.qx_m2_s, st.qy_m2_s), 0.0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True, "0"], ids=repr)
def test_continuation_from_a_state_with_a_bad_time_is_refused_before_any_step(method, bad):
    cs = build("valley:6x5", method, seed=5)
    st = cs.state
    cs.solver.step = lambda *a, **k: pytest.fail("a step ran before the entry validation")
    with pytest.raises(ExperimentalHydrologyError):
        evolve_experimental(cs.solver, field(cs), schedule([0.0, 20.0], [100.0]),
                            HydraulicState(bad, st.depth_m, st.soil_water_m, st.qx_m2_s, st.qy_m2_s), 20.0,
                            ExperimentalControl(max_dt_s=0.5), report_every_s=10.0)


@pytest.mark.parametrize("method", METHODS)
def test_a_valid_continuation_keeps_the_exact_time(method):
    cs = build("valley:6x5", method, depth="wet", seed=6)
    sched = schedule([0.0, 20.0], [100.0])
    control = ExperimentalControl(max_dt_s=0.5)
    first = evolve_experimental(cs.solver, field(cs), sched, cs.state, 10.0, control, report_every_s=5.0)
    assert first.state.t_s == 10.0
    again = evolve_experimental(cs.solver, field(cs), sched, first.state, 20.0, control, report_every_s=5.0)
    assert again.state.t_s == 20.0 and check_state_time(again.state.t_s) == 20.0


# --- stage-consistent face diagnostics ---------------------------------------------------------------------------------
def test_explicit_stage_diagnostics_are_the_closed_form_k_sqrt_h():
    cs = build("valley:8x7", "explicit", seed=7)
    step = cs.solver.step(np.zeros(cs.shape), cs.state, 0.5)
    diag = stage_face_diagnostics(step)
    k = np.asarray(cs.graph.conveyance).reshape(cs.shape)
    h_c = cs.state.depth_m  # ksat = 0: the column leaves the depth unchanged
    np.testing.assert_allclose(diag["face_flow_depth_m"]["out"], h_c, rtol=1e-14)
    np.testing.assert_allclose(diag["face_velocity_m_s"]["out"], k * np.sqrt(h_c), rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(diag["face_froude"]["out"], k / np.sqrt(G), rtol=1e-12, atol=1e-18)  # Fr = k / sqrt(g)
    assert "post rain/infiltration" in diag["stage"]


def test_local_inertial_stage_diagnostics_pair_each_flux_with_its_own_depth():
    cs = build("valley:8x7", "local_inertial", seed=8)
    state = cs.state
    for _ in range(4):
        step = cs.solver.step(np.zeros(cs.shape), state, 0.1)
        state = step.state
    diag = stage_face_diagnostics(step)
    for key in ("x", "y"):
        h_f, q = step.face_flow_depth_m[key], step.used_flux_m2_s[key]
        wet = h_f > 0.0
        assert diag["face_flow_depth_m"][key] is h_f and wet.any()
        np.testing.assert_allclose(diag["face_velocity_m_s"][key][wet], q[wet] / h_f[wet], rtol=1e-14)
        assert not np.any(diag["face_velocity_m_s"][key][~wet]) and not np.any(diag["face_froude"][key][~wet])
        np.testing.assert_allclose(diag["face_froude"][key][wet],
                                   np.abs(q[wet] / h_f[wet]) / np.sqrt(G * h_f[wet]), rtol=1e-13)
    with pytest.raises(ExperimentalHydrologyError, match="no face flow depths"):
        stage_face_diagnostics(type(step)(**{**{f: getattr(step, f) for f in step.__dataclass_fields__},
                                             "face_flow_depth_m": None}))


def steep_donor_solver():
    z = valley_full(5, 5, sy=0.2, sx=0.4)
    graph = make_graph(z, ff=0.1)
    params = make_params(param_arrays(graph.shape), graph.active)
    geometry = build_local_inertial_geometry(z, np.asarray(graph.active), np.asarray(graph.friction_factor), graph.dx_m, [])
    return CpuHydraulicSolver("local_inertial", graph, params, geometry=geometry, control=HydraulicControl(limiter="donor"))


def test_reconstructed_cell_speed_is_unbounded_near_a_drained_cell_while_the_face_diagnostics_are_not():
    """The documented limitation, made executable. A cell that the (limited) fluxes empty to its 16 eps margin has an END
    depth ~1e-18 m while its fluxes are finite: the reconstructed cell speed is astronomically large, the stage-consistent face
    velocity (flux / the stage depth it came from) is an ordinary few tenths of a m/s. Nothing is clipped in either."""
    solver = steep_donor_solver()
    depth = np.zeros((5, 5))
    depth[2, 2] = 2e-4
    state = solver.initial_state(depth, np.zeros((5, 5)))
    step = solver.step(np.zeros((5, 5)), state, 1.0)
    assert step.limited_cells == 1 and 0.0 <= step.state.depth_m[2, 2] < 1e-13
    reconstructed = float(step.velocity_m_s.max())
    diag = stage_face_diagnostics(step)
    face_speed = max(float(np.abs(diag["face_velocity_m_s"][k]).max()) for k in ("x", "y"))
    assert reconstructed > 1e6  # END-depth reconstruction: unbounded, NOT a physical velocity
    assert 0.0 < face_speed < 5.0  # stage-consistent: q_limited / h_f ~ 0.5 m/s
    assert reconstructed / face_speed > 1e5
    assert float(step.state.depth_m.sum()) * DX * DX == pytest.approx(2e-4 * DX * DX, rel=1e-13)  # no water lost or clipped
