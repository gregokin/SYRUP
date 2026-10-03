"""Corrections of the CUDA candidates, on a real device: the strict state-time contract (parity with the NumPy reference), the
immutable sealed solver (method/shape/dx/control/context), and the stage-consistent face diagnostics (CPU/GPU agreement).

Every forgery test intercepts `experimental_cuda._launch` AND `hydrology_cuda._launch` (the only paths to a raw enqueue) AFTER a
valid warm use, so a refusal that came too late fails the test with no unsafe enqueue; nothing here can execute an
out-of-bounds launch. Skipped without a device. Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from cand_cases import build, device_twin, host, rain_rate

pytest.importorskip("maple")

from maple_syrup import experimental_cuda as ec
from maple_syrup import hydrology_cuda as hc
from maple_syrup.experimental_hydrology import (
    ExperimentalHydrologyError,
    HydraulicControl,
    HydraulicState,
    stage_face_diagnostics,
)

pytestmark = pytest.mark.usefixtures("gpu")
METHODS = ("explicit", "local_inertial")
BAD_TIMES = [-1.0, float("nan"), float("inf"), True, "1", None]


def cupy():
    import cupy as cp

    return cp


_KEEP = object()


def upload(state, cp, t_s=_KEEP):
    return HydraulicState(state.t_s if t_s is _KEEP else t_s, cp.asarray(state.depth_m), cp.asarray(state.soil_water_m),
                          None if state.qx_m2_s is None else cp.asarray(state.qx_m2_s),
                          None if state.qy_m2_s is None else cp.asarray(state.qy_m2_s))


@pytest.fixture
def trap(monkeypatch):
    log: list = []

    def refuse(name, *args, **kwargs):
        log.append(name)
        pytest.fail(f"kernel {name} was enqueued but the call had to be refused first")

    def install():
        monkeypatch.setattr(ec, "_launch", refuse)
        monkeypatch.setattr(hc, "_launch", refuse)

    return log, install


# --- strict state time, parity with the reference ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", BAD_TIMES, ids=repr)
def test_initial_state_and_step_refuse_a_bad_time_like_the_reference_before_any_launch(method, bad, trap):
    cp = cupy()
    log, install = trap
    cs = build("valley:6x5", method, seed=1)
    dev = device_twin(cs, cp)
    st = cs.state
    install()
    with pytest.raises(ExperimentalHydrologyError):
        dev.solver.initial_state(cp.asarray(st.depth_m), cp.asarray(st.soil_water_m), t_s=bad)  # the probe's bool/str hole
    with pytest.raises(ExperimentalHydrologyError):
        cs.solver.initial_state(st.depth_m, st.soil_water_m, t_s=bad)
    forged = HydraulicState(bad, st.depth_m, st.soil_water_m, st.qx_m2_s, st.qy_m2_s)
    with pytest.raises(ExperimentalHydrologyError) as ref_err:
        cs.solver.step(np.zeros(cs.shape), forged, 0.1)
    with pytest.raises(ExperimentalHydrologyError) as new_err:
        dev.solver.step(cp.zeros(cs.shape), upload(st, cp, t_s=bad), 0.1)
    assert type(ref_err.value) is type(new_err.value) and str(ref_err.value) == str(new_err.value)
    assert log == []


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("t, dt, match", [(1.7e308, 1.7e308, "overflows"), (1e20, 1.0, "does not advance"),
                                          (1.0, 5e-324, "does not advance")])
def test_overflowing_and_non_advancing_steps_are_refused_before_any_launch_with_the_reference_message(method, t, dt, match,
                                                                                                    trap):
    cp = cupy()
    log, install = trap
    cs = build("valley:6x5", method, seed=2)
    dev = device_twin(cs, cp)
    st = cs.state
    install()
    with pytest.raises(ExperimentalHydrologyError, match=match) as ref_err:
        cs.solver.step(np.zeros(cs.shape), HydraulicState(t, st.depth_m, st.soil_water_m, st.qx_m2_s, st.qy_m2_s), dt)
    with pytest.raises(ExperimentalHydrologyError, match=match) as new_err:
        dev.solver.step(cp.zeros(cs.shape), upload(st, cp, t_s=t), dt)
    assert str(ref_err.value) == str(new_err.value) and log == []


@pytest.mark.parametrize("method", METHODS)
def test_time_error_precedence_matches_the_reference(method):
    cp = cupy()
    cs = build("valley:6x5", method, seed=3)
    dev = device_twin(cs, cp)
    st = cs.state
    nan_soil = st.soil_water_m.copy()
    nan_soil[2, 2] = np.nan
    cases = [(HydraulicState(-1.0, st.depth_m, nan_soil, st.qx_m2_s, st.qy_m2_s), 0.1),  # time outranks the column input
             (st, "1"), (st, True), (st, float("nan")),  # dt class: the accepted column stage's
             (HydraulicState(0.0, st.depth_m, nan_soil, st.qx_m2_s, st.qy_m2_s), 0.1),  # column input
             (st, 0.0)]  # dt = 0 last
    for state, dt in cases:
        with pytest.raises(Exception) as ref_err:
            cs.solver.step(np.zeros(cs.shape), state, dt)
        with pytest.raises(Exception) as new_err:
            dev.solver.step(cp.zeros(cs.shape), upload(state, cp), dt)
        assert type(ref_err.value) is type(new_err.value) and str(ref_err.value) == str(new_err.value), (state.t_s, dt)


# --- immutable sealed solver ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_solver_attributes_are_read_only_and_a_shape_forgery_has_no_effect(method):
    cp = cupy()
    cs = build("valley:6x5", method, seed=4)
    dev = device_twin(cs, cp)
    solver = dev.solver
    for name, value in (("shape", (1, cs.shape[0] * cs.shape[1])), ("method", "explicit"), ("dx_m", 2.0),
                        ("control", HydraulicControl(cfl_max=0.4)), ("context", solver.context), ("xp", np)):
        with pytest.raises(AttributeError):
            setattr(solver, name, value)
        with pytest.raises(AttributeError):
            object.__setattr__(solver, name, value)  # properties have no setter: not even object.__setattr__
    with pytest.raises(AttributeError, match="immutable"):
        solver.brand_new_attribute = 1
    solver.__dict__["shape"] = (1, cs.shape[0] * cs.shape[1])  # a data descriptor wins over an instance-dict entry
    assert solver.shape == cs.shape and solver.method == method
    ref = cs.solver.step(np.zeros(cs.shape), cs.state, 0.05)
    new = solver.step(cp.zeros(cs.shape), upload(cs.state, cp), 0.05)
    np.testing.assert_allclose(host(new.state.depth_m), ref.state.depth_m, rtol=2e-12, atol=1e-14)


def test_swapped_private_metadata_is_refused_before_any_raw_enqueue(trap):
    cp = cupy()
    log, install = trap
    cs = build("valley:6x5", "local_inertial", seed=5)
    dev = device_twin(cs, cp)
    other_cs = build("valley:8x7", "local_inertial", seed=5)
    other = device_twin(other_cs, cp)
    solver = dev.solver
    ctx, control = solver.context, solver.control
    st = upload(cs.state, cp)
    rain = cp.zeros(cs.shape)
    solver.step(rain, st, 0.05)  # a valid warm use first
    install()
    swaps = {
        "control_other_cfl": ("_control", HydraulicControl(cfl_max=0.4)),
        "control_limiter": ("_control", HydraulicControl(limiter="donor")),
        "control_not_a_control": ("_control", object()),
        "control_invalid": ("_control", HydraulicControl(cfl_max=7.0)),
        "context_of_another_grid": ("_context", other.solver.context),
        "context_forged_counts": ("_context", dataclasses.replace(ctx, n_cells=ctx.n_cells * 4)),
        "context_forged_dx": ("_context", dataclasses.replace(ctx, dx_m=ctx.dx_m * 2.0)),
        "context_forged_shape": ("_context", dataclasses.replace(ctx, shape=(1, ctx.n_cells))),
        "context_forged_method": ("_context", dataclasses.replace(ctx, method="explicit")),
        "context_not_a_context": ("_context", object()),
        "seal_forged": ("_seal", ("local_inertial", (1, ctx.n_cells), ctx.dx_m, 0.5, "off", id(ctx))),
    }
    for label, (attribute, value) in swaps.items():
        original = getattr(solver, attribute)
        object.__setattr__(solver, attribute, value)
        try:
            for call in (lambda: solver.step(rain, st, 0.05), lambda: solver.validate_state(st),
                         lambda: solver.initial_state(st.depth_m, st.soil_water_m)):
                with pytest.raises(ExperimentalHydrologyError):
                    call()
            assert log == [], label
        finally:
            object.__setattr__(solver, attribute, original)
    assert solver.context is ctx and solver.control is control  # restored, still the sealed objects


def test_an_explicit_solver_refuses_a_limiter_swapped_in_behind_its_back(trap):
    cp = cupy()
    log, install = trap
    cs = build("valley:6x5", "explicit", seed=6)
    dev = device_twin(cs, cp)
    solver = dev.solver
    st = upload(cs.state, cp)
    rain = cp.zeros(cs.shape)
    solver.step(rain, st, 0.05)
    install()
    object.__setattr__(solver, "_control", HydraulicControl(limiter="donor"))
    with pytest.raises(ExperimentalHydrologyError):
        solver.step(rain, st, 0.05)
    assert log == []


def test_a_new_solver_over_the_same_context_works_after_all_of_that():
    cp = cupy()
    cs = build("valley:6x5", "local_inertial", seed=7)
    dev = device_twin(cs, cp)
    again = ec.CudaHydraulicSolver("local_inertial", dev.graph, dev.params, geometry=cs.geometry,
                                   context=dev.solver.context)
    out = again.step(cp.zeros(cs.shape), upload(cs.state, cp), 0.05)
    assert out.state.t_s == pytest.approx(cs.state.t_s + 0.05)


# --- stage-consistent face diagnostics --------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_stage_face_diagnostics_agree_between_cpu_and_gpu_and_stay_on_the_device(method):
    cp = cupy()
    cs = build("valley:8x7", method, seed=8, ksat=1e-6, soil_fraction=0.4)
    dev = device_twin(cs, cp)
    state, dstate = cs.state, upload(cs.state, cp)
    rain = rain_rate(cs, 120.0)
    for _ in range(5):
        ref = cs.solver.step(rain, state, 0.1)
        new = dev.solver.step(cp.asarray(rain), dstate, 0.1)
        state, dstate = ref.state, new.state
    d_ref, d_new = stage_face_diagnostics(ref), stage_face_diagnostics(new)
    assert set(d_ref["face_velocity_m_s"]) == set(d_new["face_velocity_m_s"])
    for group in ("face_flow_depth_m", "face_velocity_m_s", "face_froude"):
        for key in d_ref[group]:
            assert type(d_new[group][key]) is cp.ndarray  # computed on the device, nothing downloaded
            np.testing.assert_allclose(host(d_new[group][key]), d_ref[group][key], rtol=2e-12, atol=1e-14,
                                       err_msg=f"{group}.{key}")
    assert d_new["stage"] == d_ref["stage"]
