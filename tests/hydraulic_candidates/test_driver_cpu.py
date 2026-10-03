"""The ONE experimental driver (`experimental_storm.evolve_experimental`) with the CPU reference solvers: rainfall forcing and
boundaries, water budget, forced retries and guards, in-memory continuation (including the face momentum), snapshots and entry
validation. Short analytic storms only. Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import numpy as np
import pytest
from cand_cases import DX, budget, build, field, schedule

pytest.importorskip("maple")

from maple_syrup.experimental_hydrology import (
    ExperimentalHydrologyError,
    HydraulicStepRejected,
)
from maple_syrup.experimental_storm import (
    EXPERIMENT_HYDROGRAPH_COLUMNS,
    ExperimentalControl,
    evolve_experimental,
)
from maple_syrup.rainfall import rainfall_field
from maple_syrup.storm import StormError, plan_boundaries

METHODS = ("explicit", "local_inertial")
AREA = DX * DX
EDGES, MM_H = [0.0, 10.0, 25.0, 40.0], [120.0, 0.0, 60.0]


def run(cs, end=40.0, *, state=None, sched=None, control=None, cadence=15.0, **kw):
    return evolve_experimental(cs.solver, field(cs), sched or schedule(EDGES, MM_H), state or cs.state, end,
                               control or ExperimentalControl(max_dt_s=0.5), report_every_s=cadence, **kw)


@pytest.mark.parametrize("method", METHODS)
def test_forcing_edges_report_times_and_the_end_are_the_boundaries_and_rain_is_exact(method):
    cs = build("valley:6x5", method, depth="dry", seed=1)
    sched = schedule(EDGES, MM_H)
    result = run(cs)
    np.testing.assert_array_equal(result.boundaries, plan_boundaries(sched, 0.0, 40.0, 15.0))
    assert set(result.boundaries) >= {10.0, 25.0, 40.0} and result.boundaries[-1] == 40.0
    assert result.hydrograph.shape == (result.boundaries.size, len(EXPERIMENT_HYDROGRAPH_COLUMNS))
    np.testing.assert_array_equal(result.hydrograph[:, 0], result.boundaries)
    expected = sched.depth_m(0.0, 40.0) * cs.graph.active.sum() * AREA  # every raining cell, exact schedule integral
    assert result.cumulative_rain_m.sum() * AREA == pytest.approx(expected, rel=1e-12)
    assert result.hydrograph[-1, EXPERIMENT_HYDROGRAPH_COLUMNS.index("cumulative_rain_m3")] == pytest.approx(expected, rel=1e-12)
    assert result.state.t_s == 40.0 and result.method == method and result.implementation == "numpy"


@pytest.mark.parametrize("method", METHODS)
def test_the_event_water_budget_closes_with_infiltration_and_open_outlets(method):
    cs = build("valley:8x7", method, depth="dry", ksat=2e-6, soil_fraction=0.5, drainage=0.3, seed=2)
    result = run(cs, end=60.0, sched=schedule([0, 30, 60], [180.0, 40.0]), cadence=20.0)
    residual, bound, v = budget(result, cs.state)
    assert abs(residual) <= bound, (residual, bound)
    assert v["rain"] > 0.0 and v["drain"] > 0.0 and v["export"] > 0.0
    rows = result.hydrograph
    col = {name: rows[:, i] for i, name in enumerate(EXPERIMENT_HYDROGRAPH_COLUMNS)}
    row_residual = (col["surface_storage_m3"] + col["soil_storage_m3"] + col["cumulative_drainage_m3"]
                    + col["cumulative_export_m3"]) - (v["initial"] + col["cumulative_rain_m3"])
    assert np.all(np.abs(row_residual) <= bound)
    assert np.all(np.diff(col["accepted_steps"]) >= 0) and col["accepted_steps"][-1] == result.n_accepted_steps
    assert float(result.max_cfl) <= 0.5 and float(result.time_of_peak_outlet_s) <= 60.0
    assert np.all(result.peak_depth_m >= result.state.depth_m - 1e-18)


# --- retries and guards --------------------------------------------------------------------------------------------------
def deep_explicit():
    cs = build("valley:6x5", "explicit", depth="dry", seed=3)
    depth = np.where(cs.graph.active, 0.05, 0.0)
    state = cs.solver.initial_state(depth, cs.state.soil_water_m)
    return cs, state, schedule([0.0, 48.0], [0.0])


def test_a_cfl_rejection_halves_dt_from_the_unchanged_state_and_the_run_still_hits_every_boundary():
    cs, state, sched = deep_explicit()
    result = run(cs, end=48.0, state=state, sched=sched, control=ExperimentalControl(max_dt_s=16.0), cadence=16.0)
    tried = [r["dt_tried_s"] for r in result.rejections]
    assert result.n_rejected_attempts >= 3 and tried[:3] == [16.0, 8.0, 4.0]  # halving, from the same state
    assert all("CFL" in r["reason"] for r in result.rejections)
    assert result.state.t_s == 48.0 and result.max_accepted_dt_s <= 16.0
    np.testing.assert_array_equal(result.hydrograph[:, 0], result.boundaries)  # every boundary reached exactly
    residual, bound, _ = budget(result, state)
    assert abs(residual) <= bound  # rejected attempts accumulated nothing
    assert float(result.max_cfl) <= 0.5


@pytest.mark.parametrize("control, match", [
    (ExperimentalControl(max_dt_s=16.0, max_retries=1), r"rejected 2 times \(max_retries = 1\)"),
    (ExperimentalControl(max_dt_s=16.0, min_dt_s=10.0), "retry floor"),
    (ExperimentalControl(max_dt_s=0.5, max_steps=3), r"max_steps = 3 reached"),
])
def test_guards_stop_the_run_with_a_bounded_error(control, match):
    cs, state, sched = deep_explicit()
    with pytest.raises(StormError, match=match):
        run(cs, end=48.0, state=state, sched=sched, control=control, cadence=16.0)


def test_control_validation_is_strict():
    for bad in (ExperimentalControl(max_dt_s=0.0), ExperimentalControl(max_dt_s=True), ExperimentalControl(min_dt_s=2.0),
                ExperimentalControl(max_retries=0), ExperimentalControl(max_retries=1.5), ExperimentalControl(max_steps=True),
                ExperimentalControl(max_dt_s="1")):
        with pytest.raises(StormError):
            bad.validated()


def test_a_non_recoverable_failure_propagates_unchanged_and_is_not_retried():
    cs = build("valley:6x5", "explicit", depth="wet", seed=4)
    original, calls = cs.solver.step, []

    def failing(rain, state, dt):
        calls.append(dt)
        if len(calls) == 3:
            raise ExperimentalHydrologyError("boom")
        return original(rain, state, dt)

    cs.solver.step = failing
    with pytest.raises(ExperimentalHydrologyError, match="boom") as info:
        run(cs, 10.0, cadence=5.0)
    assert not isinstance(info.value, HydraulicStepRejected) and len(calls) == 3  # raised once, never retried


# --- continuation, snapshots, entry validation --------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_in_memory_continuation_equals_one_run_bitwise_including_face_momentum(method):
    cs = build("valley:6x5", method, depth="wet", ksat=1e-6, soil_fraction=0.4, seed=5)
    sched = schedule([0.0, 10.0, 40.0], [120.0, 60.0])
    kw = {"sched": sched, "cadence": 10.0, "control": ExperimentalControl(max_dt_s=0.25)}
    once = run(cs, 40.0, **kw)
    first = run(cs, 20.0, **kw)
    second = run(cs, 40.0, state=first.state, **kw)
    assert first.state.t_s == 20.0 and second.state.t_s == 40.0
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        a, b = getattr(second.state, name), getattr(once.state, name)
        if a is None:
            assert b is None
        else:
            np.testing.assert_array_equal(a, b, err_msg=name)
    assert first.n_accepted_steps + second.n_accepted_steps == once.n_accepted_steps
    if method == "local_inertial":
        assert float(np.max(np.abs(first.state.qx_m2_s))) > 0.0  # the retained momentum was genuinely non-trivial


def test_snapshots_merge_onto_boundaries_or_insert_one_without_altering_forcing_or_steps():
    cs = build("valley:6x5", "local_inertial", depth="dry", seed=6)
    kw = {"cadence": 10.0, "control": ExperimentalControl(max_dt_s=0.5)}
    plain = run(cs, 40.0, **kw)
    snap = run(cs, 40.0, snapshot_times_s=(10.0, 12.5, 40.0), **kw)
    assert 12.5 in snap.boundaries and snap.boundaries.size == plain.boundaries.size + 1  # 10 and 40 merged, 12.5 added
    assert set(snap.snapshots) == {10.0, 12.5, 40.0}
    for t, s in snap.snapshots.items():
        assert s["t_s"] == t and s["depth_m"].shape == cs.shape and s["qx_m2_s"].shape == (cs.shape[0], cs.shape[1] + 1)
    np.testing.assert_array_equal(snap.snapshots[40.0]["depth_m"], snap.state.depth_m)
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):  # 12.5 is on the 0.5 s step grid: same steps, same bits
        np.testing.assert_array_equal(getattr(snap.state, name), getattr(plain.state, name), err_msg=name)
    np.testing.assert_array_equal(snap.cumulative_rain_m, plain.cumulative_rain_m)
    assert plain.snapshots == {}


@pytest.mark.parametrize("bad", [0.0, 41.0, float("nan"), float("inf"), True, "10", -1.0])
def test_invalid_snapshot_times_are_refused_before_stepping(bad):
    cs = build("valley:6x5", "explicit", depth="dry", seed=7)
    cs.solver.step = lambda *a, **k: pytest.fail("a step ran before the snapshot times were validated")
    with pytest.raises(StormError):
        run(cs, 40.0, snapshot_times_s=(bad,), cadence=10.0)


@pytest.mark.parametrize("method", METHODS)
def test_entry_validation_refuses_before_any_step(method):
    cs = build("valley:6x5", method, depth="dry", seed=8)
    cs.solver.step = lambda *a, **k: pytest.fail("a step ran before validation")
    st = cs.state
    bad_depth = st.depth_m.copy()
    bad_depth[1, 1] = np.nan
    bad = type(st)(0.0, bad_depth, st.soil_water_m, st.qx_m2_s, st.qy_m2_s)
    with pytest.raises(ExperimentalHydrologyError):
        run(cs, 10.0, state=bad)
    wrong_field = rainfall_field(3, 3)
    with pytest.raises(StormError, match="rainfall field"):
        evolve_experimental(cs.solver, wrong_field, schedule(EDGES, MM_H), st, 10.0, ExperimentalControl(),
                            report_every_s=5.0)
    with pytest.raises(StormError):
        evolve_experimental(cs.solver, field(cs), schedule(EDGES, MM_H), st, 10.0, ExperimentalControl(max_dt_s=-1.0),
                            report_every_s=5.0)


def test_results_are_fresh_and_do_not_alias_the_input_state():
    cs = build("valley:6x5", "local_inertial", depth="wet", seed=9)
    before = [a.copy() for a in (cs.state.depth_m, cs.state.soil_water_m, cs.state.qx_m2_s, cs.state.qy_m2_s)]
    first = run(cs, 20.0, cadence=10.0)
    run(cs, 20.0, cadence=10.0)
    for a, b in zip(before, (cs.state.depth_m, cs.state.soil_water_m, cs.state.qx_m2_s, cs.state.qy_m2_s), strict=True):
        np.testing.assert_array_equal(a, b)
    grids = [first.state.depth_m, first.state.qx_m2_s, first.peak_depth_m, first.cumulative_rain_m, first.last_velocity_m_s,
             cs.state.depth_m]
    for i, a in enumerate(grids):
        for b in grids[i + 1:]:
            assert not np.shares_memory(a, b)
