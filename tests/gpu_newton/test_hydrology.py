"""Prepared/resident CUDA hydrology with the Newton root (`StormControl(implementation="cuda", root_solver="newton")`):
the coupled step and the full storm scheduler against the CPU array/numba Newton on identical inputs (declared bounds
rtol 2e-12 / atol 1e-14, integers exact), both column laws, rain / no rain, fused / split / auto launch structures,
retries and rejected steps, in-memory continuation, the packet-only transfer contract and the guards. Needs a device."""
from __future__ import annotations

import dataclasses
import hashlib

import hydro_cases as hcs
import numpy as np
import pytest

pytest.importorskip("maple")

from maple.core import backend as mb

from maple_syrup import hydrology_cuda as hc
from maple_syrup import hydrology_numba as hn
from maple_syrup.infiltration import InfiltrationError
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule, rainfall_field
from maple_syrup.routing import RoutingError, RoutingStepRejected
from maple_syrup.routing_numba import numba_available
from maple_syrup.storm import (
    EvolveResult,
    StormControl,
    StormError,
    coupled_step,
    evolve,
)

pytestmark = pytest.mark.usefixtures("gpu")
TIMING_FIELDS = {"first_step_wall_s", "first_step_cpu_s", "remaining_wall_s", "remaining_cpu_s"}
EDGES = [0.0, 20.0, 40.0, 80.0, 100.0]
MM_H = [220.0, 0.0, 300.0, 0.0]
NEWTON = {"root_solver": "newton"}
SKIP = ("implementation", "root_stats")  # CPU Newton reports counters; the device production step reports None


def cupy():
    import cupy as cp

    return cp


def schedule(edges=EDGES, intensity=MM_H):
    return RainfallSchedule(edges_s=edges, intensity_mm_per_h=intensity, provenance=RainfallProvenance(kind="constant"))


def fields_for(case, cp):
    ny, nx = case.graph.shape
    scale = np.where(case.graph.active, np.random.default_rng(5).uniform(1.0, 1.5, (ny, nx)), 0.0)
    return rainfall_field(ny, nx, scale=scale), rainfall_field(ny, nx, scale=cp.asarray(scale))


def compare_results(ref: EvolveResult, new: EvolveResult) -> None:
    for f in dataclasses.fields(EvolveResult):
        if f.name in TIMING_FIELDS:
            continue
        a, b = getattr(ref, f.name), getattr(new, f.name)
        if f.name == "rejections":
            assert a == b
        elif f.name == "boundaries":
            np.testing.assert_array_equal(b, a)
        elif f.name in ("n_accepted_steps", "n_rejected_attempts", "min_accepted_dt_s", "max_accepted_dt_s"):
            assert a == b, f.name
        elif f.name == "state":
            assert b.t_s == a.t_s
            hcs.compare(a, b, exact_arrays=False, path="state")
        else:
            hcs.compare(a, b, exact_arrays=False, path=f.name)


def cpu_impl():
    return "numba" if numba_available() else "array"


def run_pair(case, *, end_s=150.0, report_every_s=25.0, mode="auto", sched=None, state=None, impl=None, **control):
    cp = cupy()
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params, mode=mode)
    host_field, dev_field = fields_for(case, cp)
    sched = sched or schedule()
    state = state or case.state
    ref = evolve(case.graph, case.params, host_field, sched, state, end_s,
                 StormControl(implementation=impl or cpu_impl(), **NEWTON, **control), report_every_s=report_every_s)
    new = evolve(dev.graph, dev.params, dev_field, sched, hcs.upload_state(state, cp), end_s,
                 StormControl(implementation="cuda", **NEWTON, **control), report_every_s=report_every_s,
                 cuda_context=ctx)
    return ref, new, dev, ctx


# --- one coupled step ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["fused", "split", "auto"])
@pytest.mark.parametrize("rain", ["on", "off"])
@pytest.mark.parametrize("kind", ["valley", "random", "valley_masked"])
@pytest.mark.parametrize("model", ["pavement_hawkins", "fixed_ksat"])
def test_prepared_newton_step_matches_the_cpu_newton_step(model, kind, rain, mode):
    cp = cupy()
    case = hcs.build_case(3, kind=kind, model=model)
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params, mode=mode)
    rate = case.rate_on if rain == "on" else np.zeros_like(case.rate_on)
    state = case.state
    for dt in (0.5, 2.0):  # two consecutive steps: the second consumes the first's discharge/depth/soil
        ref = hcs.ref_step(case, rate, state, dt, **NEWTON)
        got = hc.prepared_coupled_step(ctx, cp.asarray(rate), hcs.upload_state(state, cp), dt,
                                       StormControl(implementation="cuda", **NEWTON))
        hcs.compare(ref, got, path="step", skip=SKIP)
        route = got.route
        assert (route.root_solver, route.newton_max_iterations, route.bisection_iterations) == ("newton", 50, 0)
        assert route.implementation == "cuda" and route.root_stats is None and route.conservative
        # conservation, positivity and the constitutive residual of the device step itself
        assert abs(float(route.budget_residual_m3)) <= 1e-12 and float(route.max_constitutive_residual_m) <= 1e-12
        assert float(route.max_cell_balance_residual_m) <= 1e-12
        assert (route.depth_m.get() >= 0).all() and (route.discharge_m2_s.get() >= 0).all()
        state = ref.state


def test_default_cuda_control_still_runs_the_unchanged_bisection_kernels():
    cp = cupy()
    case = hcs.build_case(3, kind="random", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    d_rate, d_state = cp.asarray(case.rate_on), hcs.upload_state(case.state, cp)
    default = hc.prepared_coupled_step(ctx, d_rate, d_state, 1.0, StormControl(implementation="cuda"))
    explicit = hc.prepared_coupled_step(ctx, d_rate, d_state, 1.0,
                                        StormControl(implementation="cuda", root_solver="bisection",
                                                     newton_max_iterations=7))
    assert default.route.root_solver == "bisection" and default.route.bisection_iterations == 40
    assert default.route.newton_max_iterations == 0 and default.route.root_stats is None
    hcs.compare(default, explicit, exact_arrays=True, path="step")


def test_newton_and_bisection_cuda_steps_agree_within_the_root_accuracy():
    cp = cupy()
    case = hcs.build_case(4, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    a = hc.prepared_coupled_step(ctx, cp.asarray(case.rate_on), hcs.upload_state(case.state, cp), 1.0,
                                 StormControl(implementation="cuda"))
    b = hc.prepared_coupled_step(ctx, cp.asarray(case.rate_on), hcs.upload_state(case.state, cp), 1.0,
                                 StormControl(implementation="cuda", **NEWTON))
    np.testing.assert_allclose(b.route.flow_depth_m.get(), a.route.flow_depth_m.get(), rtol=0, atol=1e-9)
    np.testing.assert_allclose(b.route.discharge_m2_s.get(), a.route.discharge_m2_s.get(), rtol=1e-6, atol=1e-12)


# --- storms ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["fused", "split"])
@pytest.mark.parametrize("kind", ["valley", "random", "valley_masked"])
@pytest.mark.parametrize("model", ["pavement_hawkins", "fixed_ksat"])
def test_newton_storm_matches_the_cpu_newton_scheduler(model, kind, mode):
    case = hcs.build_case(3, kind=kind, model=model)
    ref, new, _dev, _ctx = run_pair(case, mode=mode)
    compare_results(ref, new)
    assert ref.n_accepted_steps > 100 and float(ref.cumulative_export_m3) > 0.0


def test_a_low_newton_cap_still_agrees_through_the_bracketed_fallback():
    case = hcs.build_case(3, kind="random", model="fixed_ksat")
    ref, new, _dev, _ctx = run_pair(case, end_s=60.0, report_every_s=20.0, newton_max_iterations=2)
    compare_results(ref, new)


def _courant_case():
    from maple_syrup.storm import initial_state

    case = hcs.build_case(41, kind="valley_masked", model="fixed_ksat", saturated=True, adversarial=True)
    depth = np.where(case.graph.active, 0.05, 0.3)
    return case, initial_state(case.graph, depth, case.state.soil_water_m), schedule([0.0, 48.0], [0.0])


def test_retries_follow_the_same_halving_sequence_with_newton():
    case, state, sched = _courant_case()
    ref, new, _dev, _ctx = run_pair(case, end_s=48.0, report_every_s=16.0, sched=sched, state=state, max_dt_s=16.0)
    assert ref.n_rejected_attempts >= 2
    compare_results(ref, new)
    assert [r["dt_tried_s"] for r in new.rejections] == [r["dt_tried_s"] for r in ref.rejections]


def test_a_rejected_prepared_step_is_recoverable_and_mutates_nothing():
    cp = cupy()
    case, state, _sched = _courant_case()
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    d_state = hcs.upload_state(state, cp)
    d_rate = cp.zeros(dev.graph.shape)
    before = [hashlib.sha256(a.get().tobytes()).hexdigest() for a in (d_state.depth_m, d_state.soil_water_m,
                                                                       d_state.discharge_m2_s, d_rate)]
    control = StormControl(implementation="cuda", **NEWTON)
    with pytest.raises(RoutingStepRejected, match="Courant") as new_err:
        hc.prepared_coupled_step(ctx, d_rate, d_state, 16.0, control)
    with pytest.raises(RoutingStepRejected, match="Courant") as ref_err:
        coupled_step(case.graph, case.params, np.zeros(case.graph.shape), state, 16.0,
                     StormControl(implementation="array", **NEWTON))
    assert str(new_err.value) == str(ref_err.value)
    assert before == [hashlib.sha256(a.get().tobytes()).hexdigest() for a in (d_state.depth_m, d_state.soil_water_m,
                                                                              d_state.discharge_m2_s, d_rate)]
    ok = hc.prepared_coupled_step(ctx, d_rate, d_state, 1.0, control)  # the SAME state with a smaller dt succeeds
    assert ok.route.root_solver == "newton"


def test_nonconvergence_is_a_newton_labelled_routing_error_that_modifies_no_input():
    cp = cupy()
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    host_field, dev_field = fields_for(case, cp)
    state = hcs.upload_state(case.state, cp)
    before = [hashlib.sha256(a.get().tobytes()).hexdigest() for a in (state.depth_m, state.soil_water_m,
                                                                       state.discharge_m2_s)]
    kw = {"root_tolerance_m": 1e-300, "newton_max_iterations": 9, "root_solver": "newton"}
    with pytest.raises(RoutingError, match="Newton root solver did not reach") as ref_err:
        evolve(case.graph, case.params, host_field, schedule(), case.state, 60.0,
               StormControl(implementation=cpu_impl(), **kw), report_every_s=20.0)
    with pytest.raises(RoutingError, match="Newton root solver did not reach") as new_err:
        evolve(dev.graph, dev.params, dev_field, schedule(), state, 60.0, StormControl(implementation="cuda", **kw),
               report_every_s=20.0)
    assert type(new_err.value) is type(ref_err.value) and str(new_err.value) == str(ref_err.value)
    assert "bisection" not in str(new_err.value)
    assert before == [hashlib.sha256(a.get().tobytes()).hexdigest() for a in (state.depth_m, state.soil_water_m,
                                                                               state.discharge_m2_s)]


def test_split_water_storm_continues_like_the_uninterrupted_storm():
    cp = cupy()
    case = hcs.build_case(3, kind="random", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    _hf, dev_field = fields_for(case, cp)
    control = StormControl(implementation="cuda", **NEWTON)
    kw = {"report_every_s": 20.0, "cuda_context": ctx}
    once = evolve(dev.graph, dev.params, dev_field, schedule(), dev.state, 100.0, control, **kw)
    first = evolve(dev.graph, dev.params, dev_field, schedule(), dev.state, 40.0, control, **kw)
    second = evolve(dev.graph, dev.params, dev_field, schedule(), first.state, 100.0, control, **kw)
    assert second.state.t_s == once.state.t_s == 100.0
    for name in ("depth_m", "soil_water_m", "discharge_m2_s"):
        np.testing.assert_array_equal(getattr(second.state, name).get(), getattr(once.state, name).get(),
                                      err_msg=name)
    assert first.n_accepted_steps + second.n_accepted_steps == once.n_accepted_steps
    np.testing.assert_allclose(first.cumulative_export_m3.get() + second.cumulative_export_m3.get(),
                               once.cumulative_export_m3.get(), rtol=2e-12, atol=1e-14)
    # the continuation also agrees with the CPU Newton storm run in one piece
    host_field, _ = fields_for(case, cp)
    ref = evolve(case.graph, case.params, host_field, schedule(), case.state, 100.0,
                 StormControl(implementation=cpu_impl(), **NEWTON), report_every_s=20.0)
    hcs.compare(ref.state, second.state, path="state")


class CountingFunction:
    def __init__(self, fn, log, name):
        self.fn, self.log, self.name = fn, log, name

    def __call__(self, *args, **kwargs):
        self.log.append(self.name)
        return self.fn(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.fn, name)


@pytest.mark.parametrize("mode", ["fused", "split"])
def test_newton_adds_no_transfer_one_packet_per_attempt_and_the_planned_kernels(monkeypatch, mode):
    cp = cupy()
    case, state, sched = _courant_case()
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params, mode=mode)
    hc.load_newton_kernels()  # startup outside the measured loops
    _hf, dev_field = fields_for(case, cp)
    dstate = hcs.upload_state(state, cp)
    log: list = []
    real = hc._function
    monkeypatch.setattr(hc, "_function", lambda name, newton=False: CountingFunction(real(name, newton), log,
                                                                                   (name, newton)))

    def run(end_s, solver):
        control = StormControl(implementation="cuda", max_dt_s=16.0, root_solver=solver)
        log.clear()
        before = mb.read_transfer_counters()
        res = evolve(dev.graph, dev.params, dev_field, sched, dstate, end_s, control, report_every_s=16.0,
                     cuda_context=ctx)
        return res, mb.read_transfer_counters().delta(before), list(log)

    attempts = lambda r: r.n_accepted_steps + r.n_rejected_attempts
    short, d_short, _ = run(32.0, "newton")
    long, d_long, log_long = run(48.0, "newton")
    extra = attempts(long) - attempts(short)
    assert extra > 0
    assert d_long.device_to_host - d_short.device_to_host == extra
    assert d_long.device_to_host_bytes - d_short.device_to_host_bytes == extra * hc.PACKET_WORDS * 8
    assert d_long.host_to_device == d_short.host_to_device and d_long.scalar_reads == d_short.scalar_reads
    # the same transfer counts as bisection on the same storm
    _bis, d_bis, log_bis = run(48.0, "bisection")
    assert (d_bis.device_to_host, d_bis.device_to_host_bytes, d_bis.host_to_device) == \
        (d_long.device_to_host, d_long.device_to_host_bytes, d_long.host_to_device)
    per_attempt = hc.launch_count(ctx.level_bounds, mode)
    step_kernels = [n for n in log_long if n[0].startswith("maple_syrup_hydro_")]
    assert len(step_kernels) == attempts(long) * per_attempt and all(n[1] is True for n in step_kernels)
    assert all(n[1] is False for n in log_bis)  # bisection never touches the Newton module
    assert [n for n in log_long if n[0].startswith("maple_syrup_storm_")] == \
        [n for n in log_bis if n[0].startswith("maple_syrup_storm_")]


def test_evolve_loads_the_newton_kernels_before_the_first_step(monkeypatch):
    cp = cupy()
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    _hf, dev_field = fields_for(case, cp)
    order: list = []
    real_load, real_dispatch = hc.load_newton_kernels, hc._dispatch
    monkeypatch.setattr(hc, "load_newton_kernels", lambda: (order.append("load"), real_load())[1])
    monkeypatch.setattr(hc, "_dispatch", lambda *a, **k: (order.append("step"), real_dispatch(*a, **k))[1])
    evolve(dev.graph, dev.params, dev_field, schedule(), dev.state, 40.0,
           StormControl(implementation="cuda", **NEWTON), report_every_s=20.0)
    assert order[0] == "load" and order.count("load") >= 1 and "step" in order


# --- guards ---------------------------------------------------------------------------------------------------------
def _prepared():
    cp = cupy()
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    return cp, case, dev, hc.prepare_cuda_hydrology(dev.graph, dev.params)


@pytest.mark.parametrize("bad", [{"root_solver": "brent"}, {"newton_max_iterations": 0},
                                 {"newton_max_iterations": 1001}, {"newton_max_iterations": True},
                                 {"newton_max_iterations": 2.5}])
def test_invalid_root_options_raise_routing_errors_and_mutate_nothing(bad):
    _cp, _case, dev, ctx = _prepared()
    d_rate, d_state = dev.rate_on, dev.state
    before = [a.get().copy() for a in (d_rate, d_state.depth_m, d_state.soil_water_m, d_state.discharge_m2_s)]
    kw = {"root_solver": "newton"} | bad
    with pytest.raises(RoutingError, match="root_solver|newton_max_iterations"):
        hc.prepared_coupled_step(ctx, d_rate, d_state, 1.0, StormControl(implementation="cuda", **kw))
    for a, b in zip((d_rate, d_state.depth_m, d_state.soil_water_m, d_state.discharge_m2_s), before, strict=True):
        np.testing.assert_array_equal(a.get(), b)


def test_a_column_failure_still_outranks_an_invalid_newton_option():
    _cp, _case, dev, ctx = _prepared()
    bad_rate = dev.rate_on.copy()
    bad_rate[2, 2] = -1.0
    with pytest.raises(InfiltrationError):
        hc.prepared_coupled_step(ctx, bad_rate, dev.state, 1.0,
                                 StormControl(implementation="cuda", root_solver="newton", newton_max_iterations=0))


def test_host_arrays_wrong_implementation_and_wrong_context_are_refused():
    cp, case, dev, ctx = _prepared()
    control = StormControl(implementation="cuda", **NEWTON)
    with pytest.raises(InfiltrationError, match="cupy"):
        hc.prepared_coupled_step(ctx, case.rate_on, dev.state, 1.0, control)
    for impl in ("array", "numba"):
        with pytest.raises(RoutingError, match="CUDA kernels only"):
            hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, StormControl(implementation=impl, **NEWTON))
    other = hcs.build_case(9, kind="random", model="fixed_ksat")
    odev = hcs.device_case(other, cp)
    with pytest.raises(StormError, match="cuda_context"):
        evolve(odev.graph, odev.params, fields_for(other, cp)[1], schedule(), odev.state, 20.0, control,
               report_every_s=10.0, cuda_context=ctx)
    with pytest.raises(hc.CudaHydrologyPreparationError):
        hc.prepared_coupled_step("not a context", dev.rate_on, dev.state, 1.0, control)


def test_context_seal_and_flat_control_guards_are_unchanged_for_newton():
    _cp, _case, dev, ctx = _prepared()
    forged = dataclasses.replace(ctx, n_active=ctx.n_active + 1)
    with pytest.raises(RoutingError, match="scalar metadata|inconsistent"):
        hc.prepared_coupled_step(forged, dev.rate_on, dev.state, 1.0, StormControl(implementation="cuda", **NEWTON))


def test_public_coupled_step_dispatches_newton_to_the_prepared_device_hydrology():
    cp = cupy()
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    got = coupled_step(dev.graph, dev.params, dev.rate_on, dev.state, 1.0,
                       StormControl(implementation="cuda", **NEWTON))  # a plain, not validated(), control
    ref = hcs.ref_step(case, case.rate_on, case.state, 1.0, **NEWTON)
    hcs.compare(ref, got, path="step", skip=SKIP)
    assert got.route.root_solver == "newton"


@pytest.mark.skipif(not numba_available(), reason="Numba not installed")
def test_prepared_cpu_hydrology_is_the_same_oracle_as_the_array_reference():
    case = hcs.build_case(3, kind="random", model="pavement_hawkins")
    ctx = hn.prepare_hydrology(case.graph, case.params)
    ref = hcs.ref_step(case, case.rate_on, case.state, 1.0, **NEWTON)
    cpu = hn.prepared_coupled_step(ctx, case.rate_on, case.state, 1.0,
                                   StormControl(implementation="numba", **NEWTON))
    hcs.compare(ref, cpu, path="step", skip=SKIP)
