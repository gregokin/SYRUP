"""Phase 4S Task B corrections: (a) a provided CUDA context must be bound to the actual graph/parameters objects, and
(b) the raw-pointer helper `CudaStormAccumulator` validates every public entry before any enqueue.

Both groups intercept `hydrology_cuda._launch` (the only path to a kernel enqueue), so a refusal that came too late is
recorded and fails the test; nothing here can run a kernel with a bad pointer or forged metadata. Skipped without a
device. Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import dataclasses
import gc
import weakref

import hydro_cases as hcs
import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup import hydrology_cuda as hc
from maple_syrup.infiltration import column_parameters
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule, rainfall_field
from maple_syrup.routing import RoutingError
from maple_syrup.storm import StormControl, StormError, evolve

pytestmark = pytest.mark.usefixtures("gpu")
CUDA = StormControl(implementation="cuda")


def cupy():
    import cupy as cp

    return cp


def schedule():
    return RainfallSchedule(edges_s=[0.0, 20.0, 40.0], intensity_mm_per_h=[220.0, 0.0],
                            provenance=RainfallProvenance(kind="constant"))


def dev_field(case, cp):
    ny, nx = case.graph.shape
    scale = np.where(case.graph.active, 1.2, 0.0)
    return rainfall_field(ny, nx, scale=cp.asarray(scale))


@pytest.fixture
def trap(monkeypatch):
    """Install AFTER every valid preparation; any enqueue is recorded and fails."""
    log: list = []

    def refuse(name, *args, **kwargs):
        log.append(name)
        raise AssertionError(f"kernel {name} was enqueued but the call must have been refused")

    def install():
        monkeypatch.setattr(hc, "_launch", refuse)

    return log, install


# --- (a) the provided context must belong to these objects ---------------------------------------------------------
def _scaled_params(case, cp, factor):
    spec = case.spec
    return column_parameters(model=spec["model"], ksat_m_per_s=cp.asarray(spec["ksat"]) * factor,
                             active_mask=cp.asarray(spec["mask"]),
                             **{k: cp.asarray(v) for k, v in spec["fields"].items()})


def test_a_context_prepared_for_other_parameters_is_refused_before_any_step(trap):
    cp = cupy()
    log, install = trap
    case = hcs.build_case(3, kind="valley", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    ctx_a = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    params_b = _scaled_params(case, cp, 2.0)  # same shape, same model, DOUBLE the conductivity
    assert params_b.shape == dev.params.shape and params_b.model == dev.params.model
    assert ctx_a.is_bound_to(dev.graph, dev.params) and not ctx_a.is_bound_to(dev.graph, params_b)
    install()
    field = dev_field(case, cp)
    for graph, params in ((dev.graph, params_b), (dev.graph, dataclasses.replace(dev.params))):
        with pytest.raises(StormError, match="prepared for exactly this graph"):
            evolve(graph, params, field, schedule(), dev.state, 40.0, CUDA, report_every_s=20.0, cuda_context=ctx_a)
    assert log == []


def test_a_context_of_an_equal_but_different_graph_object_is_refused(trap):
    cp = cupy()
    log, install = trap
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    dev, twin = hcs.device_case(case, cp), hcs.device_case(case, cp)  # equal data, distinct objects and arrays
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    assert not ctx.is_bound_to(twin.graph, twin.params) and not ctx.is_bound_to(dev.graph, twin.params)
    install()
    with pytest.raises(StormError, match="prepared for exactly this graph"):
        evolve(twin.graph, twin.params, dev_field(case, cp), schedule(), twin.state, 40.0, CUDA,
               report_every_s=20.0, cuda_context=ctx)
    assert log == []


def test_the_same_graph_and_parameters_are_accepted_and_the_binding_holds_no_strong_reference():
    cp = cupy()
    case = hcs.build_case(3, kind="random", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    assert ctx.is_bound_to(dev.graph, dev.params)
    result = evolve(dev.graph, dev.params, dev_field(case, cp), schedule(), dev.state, 40.0, CUDA,
                    report_every_s=20.0, cuda_context=ctx)
    assert result.n_accepted_steps > 0
    refs = (weakref.ref(dev.graph), weakref.ref(dev.params))
    del dev, result
    gc.collect()
    assert refs[0]() is None and refs[1]() is None, "the context kept the caller's graph or parameters alive"
    assert not ctx.is_bound_to(case.graph, case.params)  # host objects are never "bound"; dead references never match


def test_the_binding_detects_replaced_arrays_but_not_in_place_content_changes():
    cp = cupy()
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    dev.params.ksat_m_per_s[...] = 0.0  # documented: in-place CONTENT mutation is not detected (static copies owned)
    assert ctx.is_bound_to(dev.graph, dev.params)
    swapped = dataclasses.replace(dev.params, ksat_m_per_s=dev.params.ksat_m_per_s.copy())
    assert not ctx.is_bound_to(dev.graph, swapped)


# --- (b) CudaStormAccumulator: every public entry validates before the enqueue --------------------------------------
def _accumulator(cp, case, mode="auto"):
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params, mode=mode)
    grids = [cp.zeros(dev.graph.shape) for _ in range(6)]
    acc = hc.CudaStormAccumulator(ctx, cum_rain=grids[0], cum_intake=grids[1], cum_return=grids[2],
                                  cum_drain=grids[3], peak_depth=grids[4], peak_velocity=grids[5],
                                  peak_q=cp.zeros(()), peak_t=cp.zeros(()))
    step, packet = hc.cuda_step_with_packet(ctx, dev.rate_on, dev.state, 1.0, CUDA)
    hydro = cp.zeros((3, 16))
    return dev, ctx, acc, step, packet, hydro


def _bad_packets(cp, packet):
    bad = {"numpy": packet.get(), "float64": packet.astype(cp.float64), "short": packet[:17], "long": cp.zeros(19, cp.uint64),
           "two_d": packet.reshape(1, -1), "strided": cp.zeros(2 * hc.PACKET_WORDS, cp.uint64)[::2],
           "list": packet.get().tolist(), "none": None}
    return bad


def test_accept_validates_the_packet_time_and_context_before_any_launch(trap):
    cp = cupy()
    log, install = trap
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    _dev, _ctx, acc, step, packet, _hydro = _accumulator(cp, case)
    acc.accept(step, packet, 1.0)  # a valid call works (real launch, before the trap)
    install()
    for label, bad in _bad_packets(cp, packet).items():
        with pytest.raises(StormError, match="packet"):
            acc.accept(step, bad, 1.0)
        assert log == [], label
    for label, t in {"bool": True, "nan": float("nan"), "inf": float("inf"), "str": "1", "negative": -1.0,
                     "none": None}.items():
        with pytest.raises(StormError, match="t_s"):
            acc.accept(step, packet, t)
        assert log == [], label
    # a step whose arrays are on the host / not contiguous is refused too
    host_step = dataclasses.replace(step, column=dataclasses.replace(step.column, rain_m=step.column.rain_m.get()))
    with pytest.raises(StormError):
        acc.accept(host_step, packet, 1.0)
    assert log == []


def test_report_validates_packet_row_counts_and_times_before_any_launch(trap):
    cp = cupy()
    log, install = trap
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    _dev, _ctx, acc, step, packet, hydro = _accumulator(cp, case)
    state, vel = step.state, step.route.velocity_m_s
    acc.report(hydro, 1, state, vel, packet, 2.0, 5, 1, 0.5)  # valid
    assert float(hydro[1, 0]) == 2.0
    install()
    for label, bad in _bad_packets(cp, packet).items():
        with pytest.raises(StormError, match="packet"):
            acc.report(hydro, 0, state, vel, bad, 2.0, 5, 1, 0.5)
        assert log == [], label
    cases = {
        "row_bool": {"row": True}, "row_negative": {"row": -1}, "row_too_large": {"row": 3},
        "row_float": {"row": 1.0}, "row_str": {"row": "0"}, "n_steps_bool": {"n_steps": True},
        "n_steps_negative": {"n_steps": -1}, "n_steps_float": {"n_steps": 1.5}, "n_rejected_negative": {"n_rejected": -3},
        "dt_min_nan": {"dt_min": float("nan")}, "dt_min_negative": {"dt_min": -1.0}, "dt_min_bool": {"dt_min": True},
        "t_nan": {"t_s": float("nan")}, "t_bool": {"t_s": False},
    }
    base = {"row": 0, "t_s": 2.0, "n_steps": 5, "n_rejected": 1, "dt_min": 0.5}
    for label, change in cases.items():
        args = {**base, **change}
        with pytest.raises(StormError):
            acc.report(hydro, args["row"], state, vel, packet, args["t_s"], args["n_steps"], args["n_rejected"],
                       args["dt_min"])
        assert log == [], label
    bad_hydro = {"numpy": hydro.get(), "float32": hydro.astype(cp.float32), "columns": cp.zeros((3, 15)),
                 "one_d": cp.zeros(48), "strided": cp.zeros((6, 16))[::2]}
    for label, h in bad_hydro.items():
        with pytest.raises(StormError):
            acc.report(h, 0, state, vel, packet, 2.0, 5, 1, 0.5)
        assert log == [], label
    with pytest.raises(StormError):
        acc.report(hydro, 0, state, vel.get(), packet, 2.0, 5, 1, 0.5)  # a host velocity map is never transferred
    assert log == []


def test_the_constructor_validates_the_initial_peak_scalars_and_grids():
    cp = cupy()
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    grids = [cp.zeros(dev.graph.shape) for _ in range(6)]
    names = ("cum_rain", "cum_intake", "cum_return", "cum_drain", "peak_depth", "peak_velocity")
    good = dict(zip(names, grids, strict=True), peak_q=cp.zeros(()), peak_t=cp.zeros(()))
    hc.CudaStormAccumulator(ctx, **good)
    for key, bad in (("peak_q", np.zeros(())), ("peak_q", cp.zeros(1)), ("peak_q", cp.zeros((), cp.float32)),
                     ("peak_t", 0.0), ("peak_t", cp.zeros((), cp.int64)), ("cum_rain", np.zeros(dev.graph.shape)),
                     ("cum_rain", cp.zeros((2, 2))), ("peak_depth", cp.zeros(dev.graph.shape, cp.float32)),
                     ("peak_velocity", cp.zeros((2 * dev.graph.shape[0], dev.graph.shape[1]))[::2])):
        with pytest.raises(StormError):
            hc.CudaStormAccumulator(ctx, **{**good, key: bad})


def test_modified_context_or_grid_metadata_is_refused_on_every_entry(trap):
    cp = cupy()
    log, install = trap
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    _dev, _ctx, acc, step, packet, hydro = _accumulator(cp, case)
    install()
    original = acc._ctx
    acc._ctx = dataclasses.replace(original, n_cells=original.n_cells * 4)  # forged scalar metadata
    with pytest.raises(RoutingError, match="metadata"):
        acc.accept(step, packet, 1.0)
    with pytest.raises(RoutingError, match="metadata"):
        acc.report(hydro, 0, step.state, step.route.velocity_m_s, packet, 1.0, 1, 0, 1.0)
    acc._ctx = original
    # a replaced/modified array object (new pointer): its sealed fingerprint no longer matches
    acc._grids = (cp.zeros_like(acc._grids[0]), *acc._grids[1:])
    with pytest.raises(StormError, match="no longer match"):
        acc.accept(step, packet, 1.0)
    with pytest.raises(StormError, match="no longer match"):
        acc.report(hydro, 0, step.state, step.route.velocity_m_s, packet, 1.0, 1, 0, 1.0)
    assert log == []


def test_a_packet_from_another_device_is_refused_before_any_launch(trap):
    cp = cupy()
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("only one CUDA device is visible; the cross-device packet refusal is not exercised")
    log, install = trap
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    _dev, _ctx, acc, step, _packet, hydro = _accumulator(cp, case)
    with cp.cuda.Device(1):
        far = cp.zeros(hc.PACKET_WORDS, cp.uint64)
    install()
    with pytest.raises(StormError, match="packet"):
        acc.accept(step, far, 1.0)
    with pytest.raises(StormError, match="packet"):
        acc.report(hydro, 0, step.state, step.route.velocity_m_s, far, 1.0, 1, 0, 1.0)
    assert log == []


def test_the_authoritative_driver_still_works_with_the_stricter_helpers():
    cp = cupy()
    case = hcs.build_case(3, kind="random", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    result = evolve(dev.graph, dev.params, dev_field(case, cp), schedule(), dev.state, 40.0, CUDA, report_every_s=10.0)
    assert result.hydrograph.shape[0] == result.boundaries.size and result.n_accepted_steps >= 40
