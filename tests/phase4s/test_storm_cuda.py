"""Phase 4S Task B: the water-only CUDA storm through the SINGLE shared host scheduler `storm.evolve`, against the CPU
`array` implementation of the very same scheduler, on a real device.

Short analytic schedules (rain on/off/on with knots that do not fall on the reporting cadence, recession, retries,
guards, continuation) are compared field by field: floats within rtol 2e-12 / atol 1e-14, integers, step counts,
rejection logs, boundaries and times exactly. Residency is checked with MAPLE's transfer counters and a launch counter:
one counted packet read per attempted step, one accumulate launch per accepted step, one report launch per row, nothing
else. Heavy full-storm qualification is done by the root scripts, not here. Skipped without a device. Nothing here was run
by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import dataclasses
import hashlib
import sys

import hydro_cases as hcs
import numpy as np
import pytest

pytest.importorskip("maple")

from maple.core import backend as mb

from maple_syrup import hydrology_cuda as hc
from maple_syrup import storm
from maple_syrup.infiltration import InfiltrationError
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule, rainfall_field
from maple_syrup.routing import RoutingError
from maple_syrup.storm import EvolveResult, StormControl, StormError, StormState, evolve

pytestmark = pytest.mark.usefixtures("gpu")
TIMING_FIELDS = {"first_step_wall_s", "first_step_cpu_s", "remaining_wall_s", "remaining_cpu_s"}
# Rain >= 216 mm/h on every raining cell keeps intake below rain (pure no run-on; see the phase 7h notes), so the known
# strict hpre/h* roundoff guard is not what these storms exercise; zero-rain pieces are the recession.
SCHEDULE_EDGES = [0.0, 20.0, 40.0, 80.0, 100.0]
SCHEDULE_MM_H = [220.0, 0.0, 300.0, 0.0]


def cupy():
    import cupy as cp

    return cp


def schedule(edges=SCHEDULE_EDGES, intensity=SCHEDULE_MM_H):
    return RainfallSchedule(edges_s=edges, intensity_mm_per_h=intensity, provenance=RainfallProvenance(kind="constant"))


def fields_for(case, cp):
    """Host and device rainfall fields with the same per-cell scale (zero on inactive cells)."""
    ny, nx = case.graph.shape
    rng = np.random.default_rng(5)
    scale = np.where(case.graph.active, rng.uniform(1.0, 1.5, (ny, nx)), 0.0)
    return rainfall_field(ny, nx, scale=scale), rainfall_field(ny, nx, scale=cp.asarray(scale))


def run_pair(case, *, end_s=150.0, report_every_s=25.0, mode="auto", sched=None, state=None, **control):
    """CPU array evolve and CUDA evolve of the same case; returns (cpu result, cuda result, device case, context)."""
    cp = cupy()
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params, mode=mode)
    host_field, dev_field = fields_for(case, cp)
    sched = sched or schedule()
    state = state or case.state
    ref = evolve(case.graph, case.params, host_field, sched, state, end_s, StormControl(implementation="array",
                                                                                       **control),
                 report_every_s=report_every_s)
    new = evolve(dev.graph, dev.params, dev_field, sched, hcs.upload_state(state, cp), end_s,
                 StormControl(implementation="cuda", **control), report_every_s=report_every_s, cuda_context=ctx)
    return ref, new, dev, ctx


def compare_results(ref: EvolveResult, new: EvolveResult, *, exact_arrays: bool = False) -> None:
    for f in dataclasses.fields(EvolveResult):
        if f.name in TIMING_FIELDS:
            continue
        a, b = getattr(ref, f.name), getattr(new, f.name)
        if f.name == "rejections":
            assert a == b, "rejection log (t, dt tried, reason) differs"
        elif f.name == "boundaries":
            np.testing.assert_array_equal(b, a)
        elif f.name in ("n_accepted_steps", "n_rejected_attempts"):
            assert a == b and type(a) is type(b), f.name
        elif f.name in ("min_accepted_dt_s", "max_accepted_dt_s"):
            assert a == b, f.name
        elif f.name == "state":
            assert b.t_s == a.t_s
            hcs.compare(a, b, exact_arrays=exact_arrays, path="state")
        else:
            hcs.compare(a, b, exact_arrays=False, path=f.name)


# --- differential: the whole EvolveResult -------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["fused", "split"])
@pytest.mark.parametrize("kind", ["valley", "random", "valley_masked"])
@pytest.mark.parametrize("model", ["pavement_hawkins", "fixed_ksat"])
def test_evolve_matches_the_cpu_scheduler(model, kind, mode):
    cp = cupy()
    case = hcs.build_case(3, kind=kind, model=model)
    ref, new, _dev, _ctx = run_pair(case, mode=mode)
    compare_results(ref, new)
    assert ref.n_accepted_steps > 100 and float(ref.cumulative_export_m3) > 0.0
    # result conventions: grids resident CuPy arrays, accumulated scalars 0-d CuPy views, counts plain ints
    for name in ("cumulative_rain_m", "cumulative_intake_m", "cumulative_saturation_return_m",
                 "cumulative_drainage_m", "peak_depth_m", "peak_velocity_m_s", "last_velocity_m_s", "hydrograph"):
        assert type(getattr(new, name)) is cp.ndarray, name
    for name in ("cumulative_export_m3", "peak_outlet_discharge_m3_s", "time_of_peak_outlet_s", "max_courant_old",
                 "max_courant_new", "cell_steps_no_runon", "cell_steps_partial_runon", "cell_steps_complete_runon"):
        value = getattr(new, name)
        assert type(value) is cp.ndarray and value.shape == (), name
    assert type(new.state.depth_m) is cp.ndarray and isinstance(new.n_accepted_steps, int)


def test_forcing_boundaries_cadence_and_end_are_planned_by_the_same_code():
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    ref, new, _dev, _ctx = run_pair(case, end_s=130.0, report_every_s=25.0)  # knots at 20/40/80/100, end 130 off-cadence
    np.testing.assert_array_equal(new.boundaries, ref.boundaries)
    rows = new.hydrograph.get()
    np.testing.assert_array_equal(rows[:, 0], new.boundaries)  # column t_s: written by the report kernel
    assert new.boundaries[-1] == 130.0 and np.all(np.isin([20.0, 40.0, 80.0, 100.0], new.boundaries))
    compare_results(ref, new)


def test_every_hydrograph_column_matches_and_counts_are_exact():
    from maple_syrup.storm import HYDROGRAPH_COLUMNS

    case = hcs.build_case(3, kind="random", model="pavement_hawkins")
    ref, new, _dev, _ctx = run_pair(case)
    got, want = new.hydrograph.get(), ref.hydrograph
    assert got.shape == want.shape == (len(new.boundaries), len(HYDROGRAPH_COLUMNS))
    for j, name in enumerate(HYDROGRAPH_COLUMNS):
        np.testing.assert_allclose(got[:, j], want[:, j], rtol=2e-12, atol=1e-14, err_msg=name)
    for name in ("accepted_steps", "rejected_attempts", "t_s", "min_accepted_dt_s"):
        j = HYDROGRAPH_COLUMNS.index(name)
        np.testing.assert_array_equal(got[:, j], want[:, j], err_msg=name)


def test_context_is_prepared_once_when_not_supplied_and_gives_the_same_result():
    cp = cupy()
    case = hcs.build_case(3, kind="valley", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    _host_field, dev_field = fields_for(case, cp)
    kwargs = {"report_every_s": 25.0}
    own = evolve(dev.graph, dev.params, dev_field, schedule(), dev.state, 100.0, StormControl(implementation="cuda"),
                 **kwargs)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    given = evolve(dev.graph, dev.params, dev_field, schedule(), dev.state, 100.0,
                   StormControl(implementation="cuda"), cuda_context=ctx, **kwargs)
    compare_results(own, given, exact_arrays=True)


# --- retries and guards --------------------------------------------------------------------------------------------
def _courant_case():
    """Consistent old flux but wide steps on 5 cm of water: the first attempts are Courant-rejected (dt 16, 8, 4)."""
    from maple_syrup.storm import initial_state

    case = hcs.build_case(41, kind="valley_masked", model="fixed_ksat", saturated=True, adversarial=True)
    depth = np.where(case.graph.active, 0.05, 0.3)
    state = initial_state(case.graph, depth, case.state.soil_water_m)
    return case, state, schedule([0.0, 48.0], [0.0])


def test_retries_follow_the_same_halving_sequence_and_records():
    case, state, sched = _courant_case()
    ref, new, _dev, _ctx = run_pair(case, end_s=48.0, report_every_s=16.0, sched=sched, state=state, max_dt_s=16.0)
    assert ref.n_rejected_attempts >= 2 and len(ref.rejections) >= 2
    compare_results(ref, new)
    assert [r["dt_tried_s"] for r in new.rejections] == [r["dt_tried_s"] for r in ref.rejections]


@pytest.mark.parametrize("control", [
    {"max_dt_s": 16.0, "max_retries": 1},  # retry budget
    {"max_dt_s": 16.0, "min_dt_s": 10.0},  # retry floor
    {"max_dt_s": 16.0, "max_steps": 1},  # step budget (the first attempt is rejected, the retry then hits it)
])
def test_guard_failures_carry_the_same_class_and_message(control):
    case, state, sched = _courant_case()
    cp = cupy()
    dev = hcs.device_case(case, cp)
    _hf, dev_field = fields_for(case, cp)
    host_field, _df = fields_for(case, cp)
    with pytest.raises(StormError) as ref_err:
        evolve(case.graph, case.params, host_field, sched, state, 48.0, StormControl(implementation="array", **control),
               report_every_s=16.0)
    with pytest.raises(StormError) as new_err:
        evolve(dev.graph, dev.params, dev_field, sched, hcs.upload_state(state, cp), 48.0,
               StormControl(implementation="cuda", **control), report_every_s=16.0)
    assert str(new_err.value) == str(ref_err.value)


def test_non_recoverable_failures_propagate_unchanged_and_modify_no_input():
    cp = cupy()
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    host_field, dev_field = fields_for(case, cp)
    state = hcs.upload_state(case.state, cp)
    before = [hashlib.sha256(a.get().tobytes()).hexdigest() for a in (state.depth_m, state.soil_water_m,
                                                                       state.discharge_m2_s)]
    with pytest.raises(RoutingError, match="bisection did not reach") as ref_err:
        evolve(case.graph, case.params, host_field, schedule(), case.state, 60.0,
               StormControl(implementation="array", bisection_iterations=5), report_every_s=20.0)
    with pytest.raises(RoutingError, match="bisection did not reach") as new_err:
        evolve(dev.graph, dev.params, dev_field, schedule(), state, 60.0,
               StormControl(implementation="cuda", bisection_iterations=5), report_every_s=20.0)
    assert type(new_err.value) is type(ref_err.value) and str(new_err.value) == str(ref_err.value)
    assert before == [hashlib.sha256(a.get().tobytes()).hexdigest() for a in (state.depth_m, state.soil_water_m,
                                                                               state.discharge_m2_s)]


def test_invalid_states_and_fields_are_refused_by_the_shared_validation_before_any_step(monkeypatch):
    cp = cupy()
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    host_field, dev_field = fields_for(case, cp)
    bad_depth = case.state.depth_m.copy()
    bad_depth[1, 1] = np.nan
    bad = StormState(0.0, bad_depth, case.state.soil_water_m, case.state.discharge_m2_s)
    monkeypatch.setattr(hc, "_function", lambda name: pytest.fail("a kernel was requested before validation"))
    with pytest.raises(StormError) as ref_err:
        evolve(case.graph, case.params, host_field, schedule(), bad, 60.0, StormControl(implementation="array"),
               report_every_s=20.0)
    with pytest.raises(StormError) as new_err:
        evolve(dev.graph, dev.params, dev_field, schedule(), hcs.upload_state(bad, cp), 60.0,
               StormControl(implementation="cuda"), report_every_s=20.0)
    assert str(new_err.value) == str(ref_err.value)
    with pytest.raises(StormError, match="rainfall field"):  # a NumPy field with a CuPy graph
        evolve(dev.graph, dev.params, host_field, schedule(), dev.state, 60.0, StormControl(implementation="cuda"),
               report_every_s=20.0)


def test_cuda_context_of_another_graph_is_refused():
    cp = cupy()
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    other = hcs.build_case(3, kind="valley:9x7", model="fixed_ksat")
    dev, dev2 = hcs.device_case(case, cp), hcs.device_case(other, cp)
    foreign = hc.prepare_cuda_hydrology(dev2.graph, dev2.params)
    _hf, dev_field = fields_for(case, cp)
    with pytest.raises(StormError, match="cuda_context"):
        evolve(dev.graph, dev.params, dev_field, schedule(), dev.state, 60.0, StormControl(implementation="cuda"),
               report_every_s=20.0, cuda_context=foreign)


# --- continuation and purity --------------------------------------------------------------------------------------
def test_in_memory_continuation_equals_one_run_bitwise_on_cellwise_arrays():
    cp = cupy()
    case = hcs.build_case(3, kind="random", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    _hf, dev_field = fields_for(case, cp)
    control = StormControl(implementation="cuda")
    once = evolve(dev.graph, dev.params, dev_field, schedule(), dev.state, 100.0, control, report_every_s=20.0,
                  cuda_context=ctx)
    first = evolve(dev.graph, dev.params, dev_field, schedule(), dev.state, 40.0, control, report_every_s=20.0,
                   cuda_context=ctx)
    second = evolve(dev.graph, dev.params, dev_field, schedule(), first.state, 100.0, control, report_every_s=20.0,
                    cuda_context=ctx)
    assert second.state.t_s == once.state.t_s == 100.0
    for name in ("depth_m", "soil_water_m", "discharge_m2_s"):
        np.testing.assert_array_equal(getattr(second.state, name).get(), getattr(once.state, name).get(), err_msg=name)
    assert first.n_accepted_steps + second.n_accepted_steps == once.n_accepted_steps
    # cumulative water over the two legs adds up to the single run (rounding-level)
    np.testing.assert_allclose(first.cumulative_rain_m.get() + second.cumulative_rain_m.get(),
                               once.cumulative_rain_m.get(), rtol=2e-12, atol=1e-14)


def test_results_are_fresh_and_a_later_run_never_changes_an_earlier_one():
    cp = cupy()
    case = hcs.build_case(3, kind="valley", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    _hf, dev_field = fields_for(case, cp)
    control = StormControl(implementation="cuda")
    inputs = (dev.state.depth_m, dev.state.soil_water_m, dev.state.discharge_m2_s)
    before = [hashlib.sha256(a.get().tobytes()).hexdigest() for a in inputs]
    first = evolve(dev.graph, dev.params, dev_field, schedule(), dev.state, 100.0, control, report_every_s=20.0,
                   cuda_context=ctx)
    names = ("cumulative_rain_m", "cumulative_intake_m", "cumulative_saturation_return_m", "cumulative_drainage_m",
             "peak_depth_m", "peak_velocity_m_s", "last_velocity_m_s", "hydrograph")
    snap = {n: getattr(first, n).get().copy() for n in names}
    snap |= {f"state.{n}": getattr(first.state, n).get().copy() for n in ("depth_m", "soil_water_m",
                                                                         "discharge_m2_s")}
    scal = {n: float(getattr(first, n)) for n in ("cumulative_export_m3", "peak_outlet_discharge_m3_s",
                                                  "time_of_peak_outlet_s")}
    evolve(dev.graph, dev.params, dev_field, schedule(), dev.state, 100.0, control, report_every_s=20.0,
           cuda_context=ctx)
    for n in names:
        np.testing.assert_array_equal(getattr(first, n).get(), snap[n], err_msg=n)
    for n in ("depth_m", "soil_water_m", "discharge_m2_s"):
        np.testing.assert_array_equal(getattr(first.state, n).get(), snap[f"state.{n}"], err_msg=n)
    for n, v in scal.items():
        assert float(getattr(first, n)) == v, n
    assert before == [hashlib.sha256(a.get().tobytes()).hexdigest() for a in inputs]
    grids = [getattr(first, n) for n in names[:-1]] + [first.state.depth_m, first.state.soil_water_m]
    for i, a in enumerate(grids):
        for inp in inputs:
            assert not cp.shares_memory(a, inp)
        for b in grids[i + 1:]:
            if a is not b:
                assert not cp.shares_memory(a, b)


def test_the_public_coupled_step_dispatches_to_the_prepared_cuda_hydrology():
    cp = cupy()
    case = hcs.build_case(3, kind="random", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    control = StormControl(implementation="cuda")  # not validated
    via_storm = storm.coupled_step(dev.graph, dev.params, dev.rate_on, dev.state, 1.0, control)
    direct = hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, control)
    for k, v in hcs.all_arrays(via_storm).items():
        np.testing.assert_array_equal(v.get(), hcs.all_arrays(direct)[k].get(), err_msg=k)
    hcs.compare(hcs.ref_step(case, case.rate_on, case.state, 1.0), via_storm)
    with pytest.raises(InfiltrationError):
        storm.coupled_step(dev.graph, dev.params, case.rate_on, dev.state, 1.0, control)  # NumPy rain: no transfer


def test_the_cuda_storm_needs_no_numba(monkeypatch):
    from maple_syrup import routing_numba

    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    cp = cupy()
    dev = hcs.device_case(case, cp)
    _hf, dev_field = fields_for(case, cp)
    monkeypatch.setitem(sys.modules, "numba", None)  # `import numba` -> ImportError
    monkeypatch.setattr(routing_numba, "numba_available", lambda: False)
    res = evolve(dev.graph, dev.params, dev_field, schedule(), dev.state, 60.0, StormControl(implementation="cuda"),
                 report_every_s=20.0)
    assert res.n_accepted_steps > 0


# --- residency ---------------------------------------------------------------------------------------------------
class CountingFunction:
    def __init__(self, fn, log, name):
        self.fn, self.log, self.name = fn, log, name

    def __call__(self, grid, block, args):
        self.log.append(self.name)
        return self.fn(grid, block, args)

    def __getattr__(self, name):
        return getattr(self.fn, name)


@pytest.mark.parametrize("mode", ["fused", "split"])
def test_the_loop_reads_one_packet_per_attempt_and_launches_only_the_planned_kernels(monkeypatch, mode):
    cp = cupy()
    case, state, sched = _courant_case()  # includes rejected attempts
    dev = hcs.device_case(case, cp)
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params, mode=mode)
    _hf, dev_field = fields_for(case, cp)
    dstate = hcs.upload_state(state, cp)
    control = StormControl(implementation="cuda", max_dt_s=16.0)
    log: list = []
    real = hc._function
    monkeypatch.setattr(hc, "_function", lambda name: CountingFunction(real(name), log, name))

    def run(end_s):
        log.clear()
        before = mb.read_transfer_counters()
        res = evolve(dev.graph, dev.params, dev_field, sched, dstate, end_s, control, report_every_s=16.0,
                     cuda_context=ctx)
        return res, mb.read_transfer_counters().delta(before), list(log)

    short, d_short, log_short = run(32.0)
    long, d_long, log_long = run(48.0)
    def attempts(r):
        return r.n_accepted_steps + r.n_rejected_attempts

    extra = attempts(long) - attempts(short)
    assert extra > 0
    # one counted 144-byte packet per attempted step; no other host read and no upload inside the loop
    assert d_long.device_to_host - d_short.device_to_host == extra
    assert d_long.device_to_host_bytes - d_short.device_to_host_bytes == extra * hc.PACKET_WORDS * 8
    assert d_long.host_to_device == d_short.host_to_device and d_long.scalar_reads == d_short.scalar_reads
    # launches: per attempt the planned hydrology kernels, per accepted step one accumulate, per row one report
    per_attempt = hc.launch_count(ctx.level_bounds, mode)
    for res, names in ((short, log_short), (long, log_long)):
        step_kernels = [n for n in names if n.startswith("maple_syrup_hydro_")]
        assert len(step_kernels) == attempts(res) * per_attempt
        assert names.count("maple_syrup_storm_accumulate") == res.n_accepted_steps
        assert names.count("maple_syrup_storm_report") == res.boundaries.size
        assert len(names) == len(step_kernels) + res.n_accepted_steps + res.boundaries.size


def test_evolve_prepares_once_and_downloads_the_static_data_once(monkeypatch):
    cp = cupy()
    case = hcs.build_case(3, kind="valley", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    _hf, dev_field = fields_for(case, cp)
    prepares: list = []
    real = hc.prepare_cuda_hydrology
    monkeypatch.setattr(hc, "prepare_cuda_hydrology", lambda *a, **k: (prepares.append(1), real(*a, **k))[1])
    evolve(dev.graph, dev.params, dev_field, schedule(), dev.state, 100.0, StormControl(implementation="cuda"),
           report_every_s=20.0)
    assert prepares == [1]
