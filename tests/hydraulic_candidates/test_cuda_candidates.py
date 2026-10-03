"""CUDA forms of the two EXPERIMENTAL hydraulic alternatives against their NumPy references, on a real device.

Step and storm differentials: floats within rtol 2e-12 / atol 1e-14 (state arrays are bitwise equal wherever the accepted column
is the identity, because the lateral kernels use only + - * / sqrt); integers, counts, rejection logs and boundaries exact;
refusals compare exception CLASS and MESSAGE. Also: launch/transfer accounting, sealed-context and binding refusals before any
launch, purity, continuation (face momentum included), no Numba, streams/devices. Skipped without a device. Nothing here was run
by its author (file-only tools); Codex records results. Select the device with CUDA_VISIBLE_DEVICES before starting pytest.
"""
from __future__ import annotations

import dataclasses
import hashlib
import sys

import numpy as np
import pytest
from cand_cases import (
    DX,
    build,
    device_twin,
    field,
    host,
    make_params,
    param_arrays,
    rain_rate,
    schedule,
)
from test_routing import make_graph, valley_full

pytest.importorskip("maple")

from maple.core import backend as mb

from maple_syrup import experimental_cuda as ec
from maple_syrup import hydrology_cuda as hc
from maple_syrup.experimental_hydrology import (
    CpuHydraulicSolver,
    ExperimentalHydrologyError,
    HydraulicControl,
    HydraulicState,
    HydraulicStepRejected,
    build_local_inertial_geometry,
)
from maple_syrup.experimental_storm import ExperimentalControl, evolve_experimental
from maple_syrup.infiltration import InfiltrationError
from maple_syrup.rainfall import rainfall_field

pytestmark = pytest.mark.usefixtures("gpu")
METHODS = ("explicit", "local_inertial")
RTOL, ATOL = 2.0e-12, 1.0e-14
AREA = DX * DX


def cupy():
    import cupy as cp

    return cp


def upload(state, cp):
    return HydraulicState(state.t_s, cp.asarray(state.depth_m), cp.asarray(state.soil_water_m),
                          None if state.qx_m2_s is None else cp.asarray(state.qx_m2_s),
                          None if state.qy_m2_s is None else cp.asarray(state.qy_m2_s))


def close(a, b, *, exact=False, label=""):
    a, b = host(a), host(b)
    assert a.shape == b.shape, label
    if exact:
        np.testing.assert_array_equal(b, a, err_msg=label)
    else:
        np.testing.assert_allclose(b, a, rtol=RTOL, atol=ATOL, err_msg=label)


def compare_step(ref, new, *, exact=False):
    assert new.method == ref.method and new.cfl_kind == ref.cfl_kind and new.implementation == "cuda"
    assert new.dt_s == ref.dt_s and new.state.t_s == ref.state.t_s
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        a, b = getattr(ref.state, name), getattr(new.state, name)
        if a is None:
            assert b is None, name
        else:
            close(a, b, exact=exact and name != "soil_water_m", label=f"state.{name}")
    for name in ("rain_m", "intake_m", "saturation_return_m", "drainage_m", "depth_m", "soil_water_m"):
        close(getattr(ref.column, name), getattr(new.column, name), label=f"column.{name}")
    close(ref.velocity_m_s, new.velocity_m_s, exact=exact, label="velocity")
    for group in ("face_volume_m3", "used_flux_m2_s"):
        assert set(getattr(ref, group)) == set(getattr(new, group))
        for key in getattr(ref, group):
            close(getattr(ref, group)[key], getattr(new, group)[key], exact=exact, label=f"{group}.{key}")
    for name in ("export_m3", "outlet_discharge_m3_s", "storage_change_m3", "budget_residual_m3", "max_cfl",
                 "max_cell_balance_residual_m", "limited_volume_m3"):
        close(getattr(ref, name), getattr(new, name), label=name)
    assert int(host(ref.limited_cells)) == int(host(new.limited_cells))


def pair(kind, method, **kw):
    cs = build(kind, method, **kw)
    return cs, device_twin(cs, cupy())


# --- step differential ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("limiter", ["off", "donor"])
@pytest.mark.parametrize("kind", ["valley:8x7", "random:9x6", "valley_masked", "plane:129"])
@pytest.mark.parametrize("method", METHODS)
def test_step_matches_the_cpu_reference(method, kind, limiter):
    if method == "explicit" and limiter == "donor":
        pytest.skip("the donor limiter belongs to the local-inertial method")
    cp = cupy()
    cs, dev = pair(kind, method, control=HydraulicControl(limiter=limiter), ksat=2e-6, soil_fraction=0.5, drainage=0.2,
                   model="pavement_hawkins", seed=3)
    state = cs.state
    for i, dt in enumerate([0.1, 0.05, 0.1, 0.02, 0.1, 0.1, 0.05, 0.1]):
        rain = rain_rate(cs, 150.0 if i < 5 else 0.0)
        ref = cs.solver.step(rain, state, dt)
        new = dev.solver.step(cp.asarray(rain), upload(state, cp), dt)
        compare_step(ref, new)
        state = ref.state


@pytest.mark.parametrize("method", METHODS)
def test_without_infiltration_the_lateral_arrays_are_bitwise_identical(method):
    """ksat = 0 makes the accepted column the identity; the lateral kernels use only + - * / sqrt, so state, velocity and every
    face array must match the NumPy reference bit for bit (a hypothesis: any 1-ulp difference is a finding, not a reason to widen)."""
    cp = cupy()
    cs, dev = pair("valley:8x7", method, seed=4)
    state = cs.state
    zero = np.zeros(cs.shape)
    for _ in range(12):
        ref = cs.solver.step(zero, state, 0.05)
        new = dev.solver.step(cp.asarray(zero), upload(state, cp), 0.05)
        compare_step(ref, new, exact=True)
        state = ref.state


def steep(limiter, cp=None):
    z = valley_full(5, 5, sy=0.2, sx=0.4)
    out = []
    for xp in ((np,) if cp is None else (np, cp)):
        graph = make_graph(z, ff=0.1, xp=None if xp is np else xp)
        params = make_params(param_arrays(graph.shape), np.asarray(graph.active), xp=xp)
        geometry = build_local_inertial_geometry(z, np.asarray(graph.active), np.asarray(graph.friction_factor), graph.dx_m, [])
        control = HydraulicControl(limiter=limiter)
        if xp is np:
            out.append(CpuHydraulicSolver("local_inertial", graph, params, geometry=geometry, control=control))
        else:
            out.append(ec.CudaHydraulicSolver("local_inertial", graph, params, geometry=geometry, control=control))
    return out


@pytest.mark.parametrize("cell", [(2, 2), (2, 1)])
def test_the_donor_limiter_binds_identically_on_cpu_and_gpu(cell):
    cp = cupy()
    cpu, gpu = steep("donor", cp)
    depth = np.zeros((5, 5))
    depth[cell] = 2e-4
    state = cpu.initial_state(depth, np.zeros((5, 5)))
    ref = cpu.step(np.zeros((5, 5)), state, 1.0)
    new = gpu.step(cp.zeros((5, 5)), upload(state, cp), 1.0)
    assert int(ref.limited_cells) == int(host(new.limited_cells)) == 1
    compare_step(ref, new)
    assert float(host(new.state.depth_m).min()) >= 0.0
    assert float(host(new.state.depth_m).sum()) * AREA == pytest.approx(2e-4 * AREA, rel=1e-13)


def test_the_limiter_off_rejects_with_the_cpu_class_and_message_and_modifies_nothing():
    cp = cupy()
    cpu, gpu = steep("off", cp)
    depth = np.zeros((5, 5))
    depth[2, 2] = 2e-4
    state = cpu.initial_state(depth, np.zeros((5, 5)))
    dstate = upload(state, cp)
    with pytest.raises(HydraulicStepRejected) as ref_err:
        cpu.step(np.zeros((5, 5)), state, 1.0)
    with pytest.raises(HydraulicStepRejected) as new_err:
        gpu.step(cp.zeros((5, 5)), dstate, 1.0)
    assert type(new_err.value) is type(ref_err.value) and str(new_err.value) == str(ref_err.value)
    np.testing.assert_array_equal(host(dstate.depth_m), depth)


# --- refusals: class AND message ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_cfl_rejection_state_errors_and_dt_refusals_match_the_cpu(method):
    cp = cupy()
    cs, dev = pair("valley:6x5", method, seed=5, ksat=1e-6, soil_fraction=0.4)
    st = cs.state
    zero = np.zeros(cs.shape)

    def both(state, dt, rain=zero):
        errors = []
        for solver, r, s in ((cs.solver, rain, state), (dev.solver, cp.asarray(rain), upload(state, cp))):
            with pytest.raises(Exception) as info:
                solver.step(r, s, dt)
            errors.append(info.value)
        return errors

    ref, new = both(st, 1e3)  # CFL
    assert type(ref) is type(new) is HydraulicStepRejected and str(ref) == str(new)
    nan_soil = st.soil_water_m.copy()
    nan_soil[2, 2] = np.nan
    ref, new = both(HydraulicState(0.0, st.depth_m, nan_soil, st.qx_m2_s, st.qy_m2_s), 1e3)  # column outranks CFL
    assert type(ref) is type(new) is InfiltrationError and str(ref) == str(new)
    for dt in (0.0, -1.0, float("nan"), True):
        ref, new = both(st, dt)
        assert type(ref) is type(new) and str(ref) == str(new), dt
    if method == "local_inertial":
        g = cs.geometry
        closed = np.argwhere(g.fx_type == 0)[0]
        qx = st.qx_m2_s.copy()
        qx[tuple(closed)] = 0.1
        r, c, dr, _dc, _drop, _k = g.open_faces[0]
        idx = (r + (1 if dr > 0 else 0), c)
        qy = st.qy_m2_s.copy()
        qy[idx] = -g.fy_sign[idx] * 0.1
        nan_x = st.qx_m2_s.copy()
        nan_x[1, 1] = np.nan
        for bad in (HydraulicState(0.0, st.depth_m, st.soil_water_m, qx, st.qy_m2_s),
                    HydraulicState(0.0, st.depth_m, st.soil_water_m, st.qx_m2_s, qy),
                    HydraulicState(0.0, st.depth_m, st.soil_water_m, nan_x, st.qy_m2_s)):
            ref, new = both(bad, 0.05)
            assert type(ref) is type(new) is ExperimentalHydrologyError and str(ref) == str(new)
            dev.solver.validate_state(upload(st, cp))  # the valid state is accepted
            with pytest.raises(ExperimentalHydrologyError):
                dev.solver.validate_state(upload(bad, cp))
    else:
        with pytest.raises(ExperimentalHydrologyError, match="no momentum"):
            dev.solver.step(cp.asarray(zero), HydraulicState(0.0, cp.asarray(st.depth_m), cp.asarray(st.soil_water_m),
                                                             cp.zeros((6, 6)), cp.zeros((7, 5))), 0.05)


def test_structure_refusals_never_transfer_or_convert():
    cp = cupy()
    cs, dev = pair("valley:6x5", "local_inertial", seed=6)
    st = upload(cs.state, cp)
    rain = cp.zeros(cs.shape)
    for bad in (HydraulicState(0.0, cs.state.depth_m, st.soil_water_m, st.qx_m2_s, st.qy_m2_s),  # NumPy depth
                HydraulicState(0.0, st.depth_m, st.soil_water_m, cs.state.qx_m2_s, st.qy_m2_s),  # NumPy face array
                HydraulicState(0.0, st.depth_m, st.soil_water_m, st.qx_m2_s.astype(cp.float32), st.qy_m2_s),
                HydraulicState(0.0, st.depth_m, st.soil_water_m, st.qx_m2_s[:, :-1], st.qy_m2_s),
                HydraulicState(0.0, st.depth_m, st.soil_water_m, None, None)):
        with pytest.raises((ExperimentalHydrologyError, InfiltrationError)):
            dev.solver.step(rain, bad, 0.05)
    with pytest.raises(InfiltrationError):
        dev.solver.step(np.zeros(cs.shape), st, 0.05)  # NumPy rain


# --- launches, transfers, purity ----------------------------------------------------------------------------------------
class Counting:
    def __init__(self, fn, log, name):
        self.fn, self.log, self.name = fn, log, name

    def __call__(self, grid, block, args):
        self.log.append(self.name)
        return self.fn(grid, block, args)

    def __getattr__(self, name):
        return getattr(self.fn, name)


@pytest.mark.parametrize("method, limiter, lateral", [("explicit", "off", 3), ("local_inertial", "off", 4),
                                                      ("local_inertial", "donor", 7)])
def test_constant_launch_count_and_two_counted_packet_reads_per_attempt(monkeypatch, method, limiter, lateral):
    cp = cupy()
    cs, dev = pair("valley:8x7", method, control=HydraulicControl(limiter=limiter), seed=7)
    state = upload(cs.state, cp)
    rain = cp.zeros(cs.shape)
    dev.solver.step(rain, state, 0.05)  # warm
    log: list = []
    for module in (ec, hc):
        real = module._function
        monkeypatch.setattr(module, "_function", lambda name, real=real: Counting(real(name), log, name))
    before = mb.read_transfer_counters()
    dev.solver.step(rain, state, 0.05)
    delta = mb.read_transfer_counters().delta(before)
    lateral_names = [n for n in log if n.startswith(("maple_syrup_exp_", "maple_syrup_li_"))]
    assert len(lateral_names) == lateral  # constant: no per-cell and no per-level launches
    assert delta.device_to_host == 2 and delta.host_to_device == 0 and delta.scalar_reads == 0  # column + lateral packet
    assert delta.device_to_host_bytes == hc.PACKET_WORDS * 8 + ec.PACKET_WORDS * 8


@pytest.mark.parametrize("method", METHODS)
def test_outputs_are_fresh_and_inputs_and_earlier_results_never_change(method):
    cp = cupy()
    cs, dev = pair("valley:6x5", method, seed=8)
    st = upload(cs.state, cp)
    rain = cp.zeros(cs.shape)
    arrays = [st.depth_m, st.soil_water_m] + ([st.qx_m2_s, st.qy_m2_s] if method == "local_inertial" else [])
    before = [hashlib.sha256(host(a).tobytes()).hexdigest() for a in arrays]
    first = dev.solver.step(rain, st, 0.05)
    snap = {k: host(v).copy() for k, v in (("depth", first.state.depth_m), ("vel", first.velocity_m_s))}
    dev.solver.step(rain, first.state, 0.05)
    with pytest.raises(HydraulicStepRejected):
        dev.solver.step(rain, first.state, 1e3)
    np.testing.assert_array_equal(host(first.state.depth_m), snap["depth"])
    np.testing.assert_array_equal(host(first.velocity_m_s), snap["vel"])
    assert before == [hashlib.sha256(host(a).tobytes()).hexdigest() for a in arrays]
    outs = [first.state.depth_m, first.velocity_m_s, *first.face_volume_m3.values(), *arrays]
    for i, a in enumerate(outs):
        for b in outs[i + 1:]:
            assert not cp.shares_memory(a, b)
    for value in (first.export_m3, first.max_cfl, first.outlet_discharge_m3_s):
        assert type(value) is cp.ndarray and value.shape == ()  # 0-d views of the fresh packet


# --- sealed context, binding, immutability -----------------------------------------------------------------------------
def test_forged_contexts_are_refused_before_any_launch(monkeypatch):
    cp = cupy()
    cs, dev = pair("valley:6x5", "local_inertial", seed=9)
    solver = dev.solver
    ctx = solver.context
    st = upload(cs.state, cp)
    rain = cp.zeros(cs.shape)
    solver.step(rain, st, 0.05)  # valid use first

    def refuse(name, *args, **kwargs):
        pytest.fail(f"kernel {name} was enqueued with a forged context")

    monkeypatch.setattr(ec, "_launch", refuse)
    monkeypatch.setattr(hc, "_launch", refuse)
    forgeries = {
        "n_cells": {"n_cells": ctx.n_cells * 4}, "n_active": {"n_active": ctx.n_active + 5},
        "dx": {"dx_m": ctx.dx_m * 2.0}, "shape": {"shape": (ctx.shape[1], ctx.shape[0])},
        "shape_list": {"shape": list(ctx.shape)}, "names": {"names": ctx.names[:-1]},
        "arrays": {"arrays": {**ctx.arrays, "z": cp.zeros_like(ctx.arrays["z"])}},
        "short_array": {"arrays": {**ctx.arrays, "fx_type": ctx.arrays["fx_type"][:-1]}},
        "method": {"method": "explicit"},
    }
    for label, change in forgeries.items():
        object.__setattr__(solver, "_context", dataclasses.replace(ctx, **change))  # the public attribute is read-only
        with pytest.raises(ExperimentalHydrologyError, match="context|metadata|extents|prepared"):
            solver.step(rain, st, 0.05)
        with pytest.raises(ExperimentalHydrologyError):
            solver.validate_state(st)
    for label, change in {"hydrology_n_cells": {"n_cells": ctx.hydrology.n_cells * 4},
                          "hydrology_mode": {"mode": "split" if ctx.hydrology.mode == "fused" else "fused"},
                          "hydrology_bounds": {"level_bounds": list(ctx.hydrology.level_bounds)}}.items():
        object.__setattr__(solver, "_context",
                           dataclasses.replace(ctx, hydrology=dataclasses.replace(ctx.hydrology, **change)))
        with pytest.raises(Exception, match="metadata|extents|sealed|device") as info:  # baseline seal, before any launch
            solver.step(rain, st, 0.05)
        assert not isinstance(info.value, AssertionError), label
    object.__setattr__(solver, "_context", ctx)
    monkeypatch.undo()
    solver.step(rain, st, 0.05)  # the untouched context still works


def test_binding_to_graph_params_and_geometry_objects():
    cp = cupy()
    cs, dev = pair("valley:6x5", "local_inertial", seed=10)
    ctx = dev.solver.context
    assert ctx.is_bound_to(dev.graph, dev.params, cs.geometry)
    other_params = make_params({**cs.arrays, "ksat_m_per_s": cs.arrays["ksat_m_per_s"] + 1e-6}, np.asarray(cs.graph.active),
                               model=cs.model, xp=cp)  # same shape and model, different conductivity
    assert not ctx.is_bound_to(dev.graph, other_params, cs.geometry)
    faces = [(r, c, dr, dc) for r, c, dr, dc, _drop, _k in cs.geometry.open_faces]
    twin_geometry = build_local_inertial_geometry(cs.z_full, np.asarray(cs.graph.active),
                                                  np.asarray(cs.graph.friction_factor), cs.graph.dx_m, faces)
    assert not ctx.is_bound_to(dev.graph, dev.params, twin_geometry)  # an equal-data geometry object is another object
    with pytest.raises(ExperimentalHydrologyError, match="prepared for exactly"):
        ec.CudaHydraulicSolver("local_inertial", dev.graph, other_params, geometry=cs.geometry, context=ctx)
    with pytest.raises(ExperimentalHydrologyError, match="prepared for exactly"):
        ec.CudaHydraulicSolver("explicit", dev.graph, dev.params, context=ctx)
    same = ec.CudaHydraulicSolver("local_inertial", dev.graph, dev.params, geometry=cs.geometry, context=ctx)
    assert same.context is ctx


def test_preparation_refusals_and_owned_static_copies():
    cp = cupy()
    cs, dev = pair("valley:6x5", "local_inertial", seed=11)
    ctx = dev.solver.context
    assert ctx.static_bytes == sum(a.nbytes for a in ctx.arrays.values()) > 0
    assert not any(v is dev.graph or v is dev.params or v is cs.geometry for v in vars(ctx).values())
    assert all(int(a.data.ptr) != int(b.data.ptr) for a in ctx.arrays.values() for b in (dev.graph.conveyance,))
    with pytest.raises(ExperimentalHydrologyError, match="LocalInertialGeometry"):
        ec.prepare_experimental_cuda("local_inertial", dev.graph, dev.params)
    with pytest.raises(ExperimentalHydrologyError, match="no geometry"):
        ec.prepare_experimental_cuda("explicit", dev.graph, dev.params, geometry=cs.geometry)
    with pytest.raises(ExperimentalHydrologyError, match="method"):
        ec.prepare_experimental_cuda("diffusive", dev.graph, dev.params)
    with pytest.raises(hc.CudaHydrologyPreparationError):  # a host graph is never transferred
        ec.prepare_experimental_cuda("explicit", cs.graph, dev.params)
    other = build("valley:8x7", "local_inertial", closed=True)
    with pytest.raises(ExperimentalHydrologyError, match="does not match"):
        ec.prepare_experimental_cuda("local_inertial", dev.graph, dev.params, geometry=other.geometry)
    assert cp is not None


# --- storm differential (the shared driver) ------------------------------------------------------------------------------
def compare_results(ref, new):
    assert new.method == ref.method and new.implementation == "cuda" and new.columns == ref.columns
    np.testing.assert_array_equal(new.boundaries, ref.boundaries)
    assert new.n_accepted_steps == ref.n_accepted_steps and new.n_rejected_attempts == ref.n_rejected_attempts
    assert new.rejections == ref.rejections
    assert new.min_accepted_dt_s == ref.min_accepted_dt_s and new.max_accepted_dt_s == ref.max_accepted_dt_s
    close(ref.hydrograph, new.hydrograph, label="hydrograph")
    for name in ("cumulative_rain_m", "cumulative_intake_m", "cumulative_saturation_return_m", "cumulative_drainage_m",
                 "peak_depth_m", "peak_velocity_m_s", "last_velocity_m_s"):
        close(getattr(ref, name), getattr(new, name), label=name)
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        a, b = getattr(ref.state, name), getattr(new.state, name)
        assert (a is None) == (b is None), name
        if a is not None:
            close(a, b, label=f"state.{name}")
    for name in ("cumulative_export_m3", "peak_outlet_discharge_m3_s", "max_cfl", "limited_volume_total_m3"):
        close(getattr(ref, name), getattr(new, name), label=name)
    assert float(host(ref.time_of_peak_outlet_s)) == float(host(new.time_of_peak_outlet_s))
    assert int(host(ref.limited_cells_total)) == int(host(new.limited_cells_total))
    assert set(ref.snapshots) == set(new.snapshots)
    for t in ref.snapshots:
        for key in ref.snapshots[t]:
            close(ref.snapshots[t][key], new.snapshots[t][key], label=f"snapshot {t} {key}")


def storm(cs, dev, *, end=60.0, cadence=15.0, control=None, sched=None, state=None, **kw):
    cp = cupy()
    sched = sched or schedule([0.0, 20.0, 40.0, 60.0], [200.0, 0.0, 120.0])
    control = control or ExperimentalControl(max_dt_s=0.5)
    start = state or cs.state
    ref = evolve_experimental(cs.solver, field(cs), sched, start, end, control, report_every_s=cadence, **kw)
    new = evolve_experimental(dev.solver, field(cs, cp), sched, upload(start, cp), end, control, report_every_s=cadence, **kw)
    return ref, new


@pytest.mark.parametrize("limiter", ["off", "donor"])
@pytest.mark.parametrize("method", METHODS)
def test_storm_matches_the_cpu_driver(method, limiter):
    if method == "explicit" and limiter == "donor":
        pytest.skip("the donor limiter belongs to the local-inertial method")
    cs, dev = pair("valley:8x7", method, control=HydraulicControl(limiter=limiter), ksat=2e-6, soil_fraction=0.5,
                   drainage=0.2, depth="dry", seed=12)
    ref, new = storm(cs, dev, snapshot_times_s=(20.0, 33.0))
    compare_results(ref, new)
    assert ref.n_accepted_steps > 100 and float(host(ref.cumulative_export_m3)) > 0.0


def test_forced_retries_follow_the_same_sequence_on_the_gpu():
    from maple_syrup.experimental_hydrology import HydraulicState as S

    cs, dev = pair("valley:6x5", "explicit", depth="dry", seed=13)
    state = S(0.0, np.where(cs.graph.active, 0.05, 0.0), cs.state.soil_water_m)
    ref, new = storm(cs, dev, end=48.0, cadence=16.0, control=ExperimentalControl(max_dt_s=16.0), state=state,
                     sched=schedule([0.0, 48.0], [0.0]))
    assert ref.n_rejected_attempts >= 3
    compare_results(ref, new)


@pytest.mark.parametrize("method", METHODS)
def test_gpu_continuation_equals_one_run_bitwise_including_the_face_momentum(method):
    cp = cupy()
    cs, dev = pair("valley:6x5", method, depth="wet", ksat=1e-6, soil_fraction=0.4, seed=14)
    sched = schedule([0.0, 10.0, 40.0], [120.0, 60.0])
    control = ExperimentalControl(max_dt_s=0.25)
    fld = field(cs, cp)

    def go(end, state):
        return evolve_experimental(dev.solver, fld, sched, state, end, control, report_every_s=10.0)

    once = go(40.0, dev.state)
    first = go(20.0, dev.state)
    second = go(40.0, first.state)
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        a, b = getattr(second.state, name), getattr(once.state, name)
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_array_equal(host(a), host(b), err_msg=name)


def test_the_driver_reads_two_packets_per_attempt_and_nothing_else_in_the_loop(monkeypatch):
    cp = cupy()
    cs, dev = pair("valley:6x5", "local_inertial", depth="wet", seed=15)
    fld = field(cs, cp)
    sched = schedule([0.0, 60.0], [100.0])
    control = ExperimentalControl(max_dt_s=0.25)

    def run(end):
        before = mb.read_transfer_counters()
        res = evolve_experimental(dev.solver, fld, sched, upload(cs.state, cp), end, control, report_every_s=10.0)
        return res, mb.read_transfer_counters().delta(before)

    short, d_short = run(10.0)
    long, d_long = run(20.0)
    extra = (long.n_accepted_steps + long.n_rejected_attempts) - (short.n_accepted_steps + short.n_rejected_attempts)
    assert extra > 0
    assert d_long.device_to_host - d_short.device_to_host == 2 * extra
    assert d_long.host_to_device == d_short.host_to_device and d_long.scalar_reads == d_short.scalar_reads
    assert d_long.device_to_host_bytes - d_short.device_to_host_bytes == extra * (hc.PACKET_WORDS + ec.PACKET_WORDS) * 8


def test_the_cuda_candidates_need_no_numba(monkeypatch):
    from maple_syrup import routing_numba

    cp = cupy()
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setattr(routing_numba, "numba_available", lambda: False)
    for method in METHODS:
        cs, dev = pair("valley:6x5", method, seed=16)
        rain = rainfall_field(*cs.shape, scale=cp.asarray(np.where(cs.graph.active, 1.0, 0.0)))
        res = evolve_experimental(dev.solver, rain, schedule([0.0, 10.0], [100.0]), dev.state, 10.0,
                                  ExperimentalControl(max_dt_s=0.5), report_every_s=5.0)
        assert res.n_accepted_steps >= 20


def test_nondefault_stream_matches_the_default_stream():
    cp = cupy()
    cs, dev = pair("valley:6x5", "local_inertial", seed=17)
    rain = cp.asarray(rain_rate(cs, 100.0))
    ref = dev.solver.step(rain, upload(cs.state, cp), 0.05)
    cs2, dev2 = pair("valley:6x5", "local_inertial", seed=17)
    with cp.cuda.Stream(non_blocking=True) as stream:
        got = dev2.solver.step(cp.asarray(rain_rate(cs2, 100.0)), upload(cs2.state, cp), 0.05)
        stream.synchronize()
    close(ref.state.depth_m, got.state.depth_m, exact=True)
    close(ref.state.qx_m2_s, got.state.qx_m2_s, exact=True)


def test_wrong_device_inputs_are_refused_without_migration():
    cp = cupy()
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("only one CUDA device is visible; the multi-device refusals are not exercised")
    cs, dev = pair("valley:6x5", "explicit", seed=18)
    with cp.cuda.Device(1):
        far = upload(cs.state, cp)
        rain = cp.zeros(cs.shape)
        with pytest.raises(Exception, match="device"):
            dev.solver.step(rain, far, 0.05)
    with pytest.raises(Exception, match="device"):
        dev.solver.step(rain, far, 0.05)
