"""Phase 4S Task B: `hydrology_cuda.prepared_column_step` (the reusable column physics, column stage only) against the
accepted `infiltration.column_step` on the host, on a real device.

Fields: rtol 2e-12 / atol 1e-14 (arrays bitwise in the saturated regime); refusals: class AND message equal to the
reference; purity: inputs and the context never written, every output fresh; and the API applies NO routing guard (it must
accept states the coupled step refuses). Skipped without a device (explicit skip). Nothing here was run by its author
(file-only tools); Codex records results.
"""
from __future__ import annotations

import hashlib
import inspect

import hydro_cases as hcs
import numpy as np
import pytest

pytest.importorskip("maple")

from maple.core import backend as mb
from test_routing import chain_full, make_graph

from maple_syrup import hydrology_cuda as hc
from maple_syrup import routing_numba
from maple_syrup.infiltration import (
    ColumnStep,
    InfiltrationError,
    column_parameters,
    column_step,
)
from maple_syrup.routing import RoutingStepRejected
from maple_syrup.storm import StormControl, StormState, initial_state

pytestmark = pytest.mark.usefixtures("gpu")
MODES = ("fused", "split")
CUDA = StormControl(implementation="cuda")


def cupy():
    import cupy as cp

    return cp


def prepared(case, mode="auto"):
    dev = hcs.device_case(case, cupy())
    return dev, hc.prepare_cuda_hydrology(dev.graph, dev.params, mode=mode)


def digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array.get()).tobytes()).hexdigest()


def column_arrays(step) -> dict:
    return {"depth": step.depth_m, "soil": step.soil_water_m, "rain": step.rain_m, "intake": step.intake_m,
            "return": step.saturation_return_m, "drainage": step.drainage_m}


# --- differential -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["random", "valley_masked", "plane:129"])
@pytest.mark.parametrize("model", ["pavement_hawkins", "fixed_ksat"])
def test_column_step_matches_the_reference(model, kind, mode):
    cp = cupy()
    case = hcs.build_case(4, kind=kind, model=model)
    dev, ctx = prepared(case, mode)
    zero, zero_d = np.zeros(case.graph.shape), cp.zeros(case.graph.shape)
    h, s = case.state.depth_m, case.state.soil_water_m
    seen = {"intake": 0.0, "drain": 0.0}
    for i, dt in enumerate(([1.0, 0.5, 0.25, 1.0 / 1024.0, 1.0, 2.0, 0.5] * 4)[:24]):
        on = i < 14
        ref = column_step(case.params, h, s, case.rate_on if on else zero, dt)
        new = hc.prepared_column_step(ctx, cp.asarray(h), cp.asarray(s), dev.rate_on if on else zero_d, dt)
        assert isinstance(new, ColumnStep)
        hcs.compare(ref, new, skip=())
        seen["intake"] += float(ref.intake_m.sum())
        seen["drain"] += float(ref.drainage_m.sum())
        h, s = ref.depth_m, ref.soil_water_m
    assert seen["intake"] > 0.0 and seen["drain"] > 0.0


@pytest.mark.parametrize("mode", MODES)
def test_saturated_regime_column_arrays_are_bitwise(mode):
    cp = cupy()
    case = hcs.build_case(11, kind="valley", model="fixed_ksat", saturated=True)
    dev, ctx = prepared(case, mode)
    h, s = case.state.depth_m, case.state.soil_water_m
    returned = 0.0
    for _ in range(25):
        ref = column_step(case.params, h, s, case.rate_on, 1.0)
        new = hc.prepared_column_step(ctx, cp.asarray(h), cp.asarray(s), dev.rate_on, 1.0)
        hcs.compare(ref, new, exact_arrays=True, skip=())
        returned += float(ref.saturation_return_m.sum())
        h, s = ref.depth_m, ref.soil_water_m
    assert returned > 0.0


@pytest.mark.skipif(not routing_numba.numba_available(), reason="Numba prepared CPU oracle not installed")
def test_column_step_matches_the_prepared_cpu_column_step_too():
    from maple_syrup import hydrology_numba as hn

    case = hcs.build_case(9, kind="random", model="pavement_hawkins")
    dev, ctx = prepared(case)
    cpu = hn.prepare_hydrology(case.graph, case.params)
    cp = cupy()
    cpu_step = hn.prepared_column_step(cpu, case.state.depth_m, case.state.soil_water_m, case.rate_on, 0.5)
    new = hc.prepared_column_step(ctx, cp.asarray(case.state.depth_m), cp.asarray(case.state.soil_water_m),
                                  dev.rate_on, 0.5)
    hcs.compare(cpu_step, new, skip=())


def test_masked_cells_keep_their_inventory_and_exchange_nothing():
    cp = cupy()
    case = hcs.build_case(8, kind="valley_masked", model="pavement_hawkins")
    dev, ctx = prepared(case)
    new = hc.prepared_column_step(ctx, cp.asarray(case.state.depth_m), cp.asarray(case.state.soil_water_m),
                                  dev.rate_on, 1.0)
    inactive = ~case.graph.active
    assert inactive.any()
    np.testing.assert_array_equal(new.depth_m.get()[inactive], case.state.depth_m[inactive])
    np.testing.assert_array_equal(new.soil_water_m.get()[inactive], case.state.soil_water_m[inactive])
    for array in (new.intake_m, new.drainage_m, new.saturation_return_m, new.rain_m):
        assert not np.any(array.get()[inactive])


# --- dt = 0 identity, refusals -------------------------------------------------------------------------------------
def test_dt_zero_is_the_identity_with_fresh_arrays_and_still_validates_inputs():
    cp = cupy()
    case = hcs.build_case(9, kind="random", model="pavement_hawkins")
    dev, ctx = prepared(case)
    h, s = cp.asarray(case.state.depth_m), cp.asarray(case.state.soil_water_m)
    ref = column_step(case.params, case.state.depth_m, case.state.soil_water_m, case.rate_on, 0.0)
    new = hc.prepared_column_step(ctx, h, s, dev.rate_on, 0.0)
    hcs.compare(ref, new, exact_arrays=True, skip=())
    assert not cp.shares_memory(new.depth_m, h) and not cp.shares_memory(new.soil_water_m, s)
    arrays = column_arrays(new)
    flux = ("rain", "intake", "return", "drainage")
    for a in flux:
        for b in flux:
            if a != b:
                assert not cp.shares_memory(arrays[a], arrays[b])
    bad = h.copy()
    bad[hcs.ACT] = cp.nan
    with pytest.raises(InfiltrationError) as ref_err:
        column_step(case.params, bad.get(), case.state.soil_water_m, case.rate_on, 0.0)
    with pytest.raises(InfiltrationError) as new_err:
        hc.prepared_column_step(ctx, bad, s, dev.rate_on, 0.0)
    assert str(new_err.value) == str(ref_err.value)


def poke(array, index, value):
    out = array.copy()
    out[index] = value
    return out


def _case():
    case = hcs.build_case(41, kind="valley_masked", model="fixed_ksat", saturated=True, adversarial=True)
    return case, {"rate": case.rate_on.copy(), "depth": case.state.depth_m.copy(),
                  "soil": case.state.soil_water_m.copy(), "dt": 1.0, "smax": case.params.storage_max_m}


MUTATIONS = {
    "depth_nan": lambda d: d.update(depth=poke(d["depth"], hcs.ACT, np.nan)),
    "depth_inf": lambda d: d.update(depth=poke(d["depth"], hcs.ACT, np.inf)),
    "depth_negative": lambda d: d.update(depth=poke(d["depth"], hcs.ACT, -1e-3)),
    "soil_nan": lambda d: d.update(soil=poke(d["soil"], hcs.ACT, np.nan)),
    "soil_negative": lambda d: d.update(soil=poke(d["soil"], hcs.ACT, -1e-3)),
    "soil_exceeds_capacity": lambda d: d.update(soil=poke(d["soil"], hcs.ACT, float(d["smax"][hcs.ACT]) * 1.01)),
    "rain_nan": lambda d: d.update(rate=poke(d["rate"], hcs.ACT, np.nan)),
    "rain_negative": lambda d: d.update(rate=poke(d["rate"], hcs.ACT, -1e-6)),
    "rain_on_inactive_cell": lambda d: d.update(rate=poke(d["rate"], hcs.INACT, 1e-5)),
    "dt_negative": lambda d: d.update(dt=-1.0),
    "dt_nan": lambda d: d.update(dt=float("nan")),
    "dt_inf": lambda d: d.update(dt=float("inf")),
    "dt_bool": lambda d: d.update(dt=True),
    "dt_string": lambda d: d.update(dt="1"),
    "depth_float32": lambda d: d.update(depth=d["depth"].astype(np.float32)),
    "depth_wrong_shape": lambda d: d.update(depth=np.zeros((2, 3))),
    "rain_wrong_dtype": lambda d: d.update(rate=d["rate"].astype(np.float32)),
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_refusals_match_the_reference_class_and_message_and_mutate_nothing(name, mode):
    cp = cupy()
    case, d = _case()
    MUTATIONS[name](d)
    _dev, ctx = prepared(case, mode)
    h, s, r = cp.asarray(d["depth"]), cp.asarray(d["soil"]), cp.asarray(d["rate"])
    before = [digest(a) for a in (h, s, r)]
    with pytest.raises(Exception) as ref_err:
        column_step(case.params, d["depth"], d["soil"], d["rate"], d["dt"])
    with pytest.raises(Exception) as new_err:
        hc.prepared_column_step(ctx, h, s, r, d["dt"])
    assert type(new_err.value) is type(ref_err.value), (name, repr(ref_err.value), repr(new_err.value))
    assert str(new_err.value) == str(ref_err.value), name
    assert before == [digest(a) for a in (h, s, r)]


def test_structure_refusals_name_the_cupy_contract():
    cp = cupy()
    case, d = _case()
    _dev, ctx = prepared(case)
    h, s, r = cp.asarray(d["depth"]), cp.asarray(d["soil"]), cp.asarray(d["rate"])
    with pytest.raises(InfiltrationError, match="cupy.ndarray"):
        hc.prepared_column_step(ctx, d["depth"], s, r, 1.0)  # NumPy: never transferred
    with pytest.raises(InfiltrationError, match="cupy.ndarray"):
        hc.prepared_column_step(ctx, h, s, d["rate"].tolist(), 1.0)
    with pytest.raises(InfiltrationError, match="C-contiguous"):
        hc.prepared_column_step(ctx, h, s, r.T.copy().T, 1.0)
    with pytest.raises(hc.CudaHydrologyPreparationError):
        hc.prepared_column_step(object(), h, s, r, 1.0)


# --- purity -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_the_column_step_is_pure_deterministic_and_fresh(mode):
    cp = cupy()
    case = hcs.build_case(12, kind="random", model="pavement_hawkins")
    dev, ctx = prepared(case, mode)
    h, s = cp.asarray(case.state.depth_m), cp.asarray(case.state.soil_water_m)
    inputs = (h, s, dev.rate_on)
    before_in = [digest(a) for a in inputs]
    before_ctx = [digest(a.reshape(-1)) for a in hc._arrays_of(ctx)]
    runs = [hc.prepared_column_step(ctx, h, s, dev.rate_on, 0.5) for _ in range(4)]
    assert before_in == [digest(a) for a in inputs]
    assert before_ctx == [digest(a.reshape(-1)) for a in hc._arrays_of(ctx)]
    first = {k: v.get().tobytes() for k, v in column_arrays(runs[0]).items()}
    for run in runs[1:]:
        assert first == {k: v.get().tobytes() for k, v in column_arrays(run).items()}
    pointers = {int(v.data.ptr) for run in runs for v in column_arrays(run).values()}
    assert len(pointers) == 6 * len(runs)  # fresh outputs every call
    for run in runs:
        for name, a in column_arrays(run).items():
            assert type(a) is cp.ndarray and a.dtype == cp.float64 and a.shape == case.graph.shape, name
            for inp in inputs:
                assert not cp.shares_memory(a, inp)
    runs[0].depth_m.fill(123.0)  # scribbling on one result affects nothing else
    again = hc.prepared_column_step(ctx, h, s, dev.rate_on, 0.5)
    assert first == {k: v.get().tobytes() for k, v in column_arrays(again).items()}


def test_launches_and_transfers_of_one_column_step(monkeypatch):
    log: list = []
    real = hc._function

    class Counting:
        def __init__(self, fn, name):
            self.fn, self.name = fn, name

        def __call__(self, grid, block, args):
            log.append((self.name, grid, block))
            return self.fn(grid, block, args)

        def __getattr__(self, name):
            return getattr(self.fn, name)

    monkeypatch.setattr(hc, "_function", lambda name: Counting(real(name), name))
    cp = cupy()
    for mode, expected in (("fused", ["maple_syrup_hydro_fused"]),
                           ("split", ["maple_syrup_hydro_pre", "maple_syrup_hydro_reduce"])):
        case = hcs.build_case(13, kind="valley", model="fixed_ksat")
        dev, ctx = prepared(case, mode)
        h, s = cp.asarray(case.state.depth_m), cp.asarray(case.state.soil_water_m)
        hc.prepared_column_step(ctx, h, s, dev.rate_on, 1.0)
        log.clear()
        before = mb.read_transfer_counters()
        hc.prepared_column_step(ctx, h, s, dev.rate_on, 1.0)
        delta = mb.read_transfer_counters().delta(before)
        assert [n for n, _g, _b in log] == expected, mode
        assert delta.device_to_host == 1 and delta.device_to_host_bytes == hc.PACKET_WORDS * 8
        assert delta.host_to_device == 0 and delta.scalar_reads == 0


# --- NO routing guards --------------------------------------------------------------------------------------------
def _chain_case(depth, soil, rate) -> hcs.Case:
    """Three full-capacity-style fixed-Ksat chain columns (Smith-Parlange limits exercised by the caller's state)."""
    graph = make_graph(chain_full(3), ff=5.0)
    shape = graph.shape

    def full(v):
        return np.full(shape, float(v))

    fields = {"suction_m": full(0.01), "drainage_parameter": full(0.0), "theta_sat": full(0.4),
              "soil_thickness_m": full(0.3)}
    params = column_parameters(model="fixed_ksat", ksat_m_per_s=full(1e-6), active_mask=graph.active.copy(), **fields)
    spec = {"z": chain_full(3), "ff": 5.0, "active": None, "ksat": full(1e-6), "mask": graph.active.copy(),
            "model": "fixed_ksat", "fields": fields}
    return hcs.Case(graph, params, initial_state(graph, depth, soil), rate, spec)


def test_the_column_step_accepts_states_whose_coupled_step_the_courant_guard_refuses():
    cp = cupy()
    # (a) the former roundoff trigger (hpre one ulp above h*) now runs the coherent complete branch in the coupled step too, and the
    # column step agrees with the reference; the coupled-step guard itself is unchanged (see test_step: direct route_step test)
    rng = np.random.default_rng(2024)
    rain = np.float64(1e-5)
    h = next(np.float64(v) for v in rng.uniform(1e-6, 1e-3, 4000)
             if max(np.float64(v) - max((np.float64(v) + rain) - rain, 0.0), 0.0) > 0.0)
    shape = make_graph(chain_full(3), ff=5.0).shape
    depth = np.zeros(shape)
    depth[1, 0] = h
    case = _chain_case(depth, np.zeros(shape), np.full(shape, float(rain)))
    dev, ctx = prepared(case)
    drate = cp.asarray(case.rate_on)
    coupled = hc.prepared_coupled_step(ctx, drate, dev.state, 1.0, CUDA)  # formerly refused
    hcs.compare(hcs.ref_step(case, case.rate_on, case.state, 1.0), coupled)
    ref = column_step(case.params, case.state.depth_m, case.state.soil_water_m, case.rate_on, 1.0)
    new = hc.prepared_column_step(ctx, dev.state.depth_m, dev.state.soil_water_m, drate, 1.0)
    hcs.compare(ref, new, exact_arrays=True, skip=())
    # (b) a Courant rejection of the coupled step (16 s on 5 cm of water): a column step has no old flux to reject
    case2, d = _case()
    wet = np.where(case2.graph.active, 0.05, 0.3)
    k = np.asarray(case2.graph.conveyance).reshape(case2.graph.shape)
    q = np.where(case2.graph.active, (np.sqrt(wet) * wet) * k, 0.0)
    _dev2, ctx2 = prepared(case2)
    zero = cp.zeros(case2.graph.shape)
    with pytest.raises(RoutingStepRejected, match="Courant"):
        hc.prepared_coupled_step(ctx2, zero, StormState(0.0, cp.asarray(wet), cp.asarray(d["soil"]), cp.asarray(q)),
                                 16.0, CUDA)
    ref2 = column_step(case2.params, wet, d["soil"], np.zeros(case2.graph.shape), 16.0)
    new2 = hc.prepared_column_step(ctx2, cp.asarray(wet), cp.asarray(d["soil"]), zero, 16.0)
    hcs.compare(ref2, new2, skip=())


def test_the_column_step_has_no_discharge_or_route_input():
    assert list(inspect.signature(hc.prepared_column_step).parameters) == [
        "ctx", "depth_m", "soil_water_m", "rain_rate_m_per_s", "dt_s"]
