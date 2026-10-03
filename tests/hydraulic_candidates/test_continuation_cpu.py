"""Adaptive-step CONTINUATION of the experimental driver (CPU reference solvers, no Numba).

The driver's adaptive step cap shapes the step sequence and therefore the physical state, so a run to a boundary followed by a
resumed run must equal one continuous run on the same forcing/report/grid/control: bitwise state and face momentum, accepted and
rejected counts that ADD UP. The scenarios are the root's continuation probes: forcing edges at 0, 0.451875 max_dt and
3 max_dt (both intensities zero: a pure routing problem), report cadence = max_dt, checkpoint at max_dt, a deep initial lake so
the CFL / positivity bounds really reject steps (the cap really adapts). The cap must not be reset at boundaries, fresh-event
dynamics must not change, and a malformed cap is refused before any step. Nothing here was run by its author (file-only tools);
Codex records results.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from cand_cases import build, field, schedule

pytest.importorskip("maple")

from maple_syrup.experimental_hydrology import (
    ExperimentalHydrologyError,
    check_dt_cap,
)
from maple_syrup.experimental_storm import ExperimentalControl, evolve_experimental

# (method, max_dt, uniform initial depth over the active cells). The explicit rows are the root's measured defect cases.
CASES = [
    ("explicit", 16.0, 0.05), ("explicit", 4.0, 0.2), ("explicit", 1.0, 1.0),
    ("local_inertial", 4.0, 0.02), ("local_inertial", 1.0, 0.1),
]


def scenario(method, max_dt, depth):
    cs = build("valley:6x5", method, depth="dry", seed=3)
    state0 = cs.solver.initial_state(np.where(cs.graph.active, depth, 0.0), cs.state.soil_water_m)
    end = 3.0 * max_dt
    sched = schedule([0.0, 0.451875 * max_dt, end], [0.0, 0.0])  # a non-binary forcing edge, no rain
    control = ExperimentalControl(max_dt_s=max_dt)

    def run(state, until):
        return evolve_experimental(cs.solver, field(cs), sched, state, until, control, report_every_s=max_dt)

    return cs, state0, end, run


def assert_same_trajectory(once, first, second):
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        a, b = getattr(once.state, name), getattr(second.state, name)
        assert (a is None) == (b is None), name
        if a is not None:
            np.testing.assert_array_equal(a, b, err_msg=name)  # bitwise: the same step sequence
    assert first.n_accepted_steps + second.n_accepted_steps == once.n_accepted_steps
    assert first.n_rejected_attempts + second.n_rejected_attempts == once.n_rejected_attempts
    assert min(first.min_accepted_dt_s, second.min_accepted_dt_s) == once.min_accepted_dt_s
    assert max(first.max_accepted_dt_s, second.max_accepted_dt_s) == once.max_accepted_dt_s
    assert second.state.t_s == once.state.t_s and second.state.next_dt_cap_s == once.state.next_dt_cap_s
    export = float(first.cumulative_export_m3) + float(second.cumulative_export_m3)
    assert export == pytest.approx(float(once.cumulative_export_m3), rel=1e-12, abs=1e-300)


@pytest.mark.parametrize(("method", "max_dt", "depth"), CASES)
def test_a_resumed_run_equals_one_continuous_run_with_real_adaptive_rejections(method, max_dt, depth):
    _cs, state0, end, run = scenario(method, max_dt, depth)
    once, first = run(state0, end), run(state0, max_dt)
    assert once.n_rejected_attempts > 0, "the scenario must really exercise the adaptive cap"
    assert first.state.next_dt_cap_s is not None and first.next_dt_cap_s == first.state.next_dt_cap_s
    second = run(first.state, end)
    assert_same_trajectory(once, first, second)


@pytest.mark.parametrize(("method", "max_dt", "depth"), [c for c in CASES if c[0] == "explicit"])
def test_the_old_behaviour_of_resetting_the_cap_does_differ_so_the_test_has_teeth(method, max_dt, depth):
    """The defect the root measured: resuming WITHOUT the cap (what a reset at the checkpoint does) follows another path."""
    _cs, state0, end, run = scenario(method, max_dt, depth)
    once, first = run(state0, end), run(state0, max_dt)
    naive = run(dataclasses.replace(first.state, next_dt_cap_s=None), end)
    different = (not np.array_equal(naive.state.depth_m, once.state.depth_m)
                 or first.n_accepted_steps + naive.n_accepted_steps != once.n_accepted_steps
                 or first.n_rejected_attempts + naive.n_rejected_attempts != once.n_rejected_attempts)
    assert different


def test_the_cap_is_continuation_state_but_a_solver_step_never_carries_or_reads_it():
    cs, state0, _end, _run = scenario("explicit", 1.0, 0.1)
    assert state0.next_dt_cap_s is None  # a fresh state
    step = cs.solver.step(np.zeros(cs.shape), state0, 0.01)
    assert step.state.next_dt_cap_s is None
    positional = type(state0)(0.0, state0.depth_m, state0.soil_water_m)  # old positional constructors still work
    assert positional.qx_m2_s is None and positional.next_dt_cap_s is None


# --- the cap is bounded by the NEW control, and fresh-event dynamics are unchanged ---------------------------------------------
def dry_run(cap, max_dt=1.0, min_dt=1.0 / 1024.0):
    cs = build("valley:6x5", "explicit", depth="dry", seed=1)  # nothing to route: no rejection can interfere
    state = dataclasses.replace(cs.state, next_dt_cap_s=cap)
    return evolve_experimental(cs.solver, field(cs), schedule([0.0, 4.0], [0.0]), state, 4.0,
                               ExperimentalControl(max_dt_s=max_dt, min_dt_s=min_dt), report_every_s=4.0)


def test_a_fresh_event_still_starts_at_max_dt():
    result = dry_run(None)
    assert result.n_accepted_steps == 4 and result.min_accepted_dt_s == 1.0 and result.max_accepted_dt_s == 1.0
    assert result.next_dt_cap_s == 1.0 == result.state.next_dt_cap_s


def test_a_small_resumed_cap_limits_the_first_steps_and_then_grows_back():
    result = dry_run(0.25)  # 0.25 -> 0.5 -> forced 0.25 slice to t = 1 -> then 1, 1, 1
    assert result.n_accepted_steps == 6 and result.min_accepted_dt_s == 0.25 and result.max_accepted_dt_s == 1.0
    assert result.next_dt_cap_s == 1.0


def test_a_cap_above_max_dt_is_bounded_by_the_control():
    result = dry_run(100.0)
    assert result.n_accepted_steps == 4 and result.max_accepted_dt_s == 1.0


def test_a_cap_below_the_retry_floor_is_raised_to_the_floor():
    result = dry_run(1e-9)
    assert result.min_accepted_dt_s == 1.0 / 1024.0


# --- validation before any step -------------------------------------------------------------------------------------------
BAD_CAPS = [0.0, -1.0, -1e-300, float("nan"), float("inf"), -float("inf"), True, False, "1", b"1", [1.0], np.array(1.0)]


@pytest.mark.parametrize("bad", BAD_CAPS, ids=repr)
def test_a_malformed_cap_is_refused_before_any_step(bad):
    cs, state0, end, _run = scenario("explicit", 1.0, 0.1)
    broken = dataclasses.replace(state0, next_dt_cap_s=bad)
    with pytest.raises(ExperimentalHydrologyError, match="next_dt_cap_s"):
        check_dt_cap(bad)
    with pytest.raises(ExperimentalHydrologyError, match="next_dt_cap_s"):
        cs.solver.validate_state(broken)
    with pytest.raises(ExperimentalHydrologyError, match="next_dt_cap_s"):
        cs.solver.step(np.zeros(cs.shape), broken, 0.1)
    cs.solver.step = lambda *a, **k: pytest.fail("a step ran before the continuation metadata was validated")
    with pytest.raises(ExperimentalHydrologyError, match="next_dt_cap_s"):
        evolve_experimental(cs.solver, field(cs), schedule([0.0, end], [0.0]), broken, end,
                            ExperimentalControl(max_dt_s=1.0), report_every_s=1.0)


@pytest.mark.parametrize(("good", "expected"), [(None, None), (1, 1.0), (0.5, 0.5), (np.float64(2.0), 2.0), (np.int64(3), 3.0)])
def test_valid_caps_are_returned_as_floats(good, expected):
    value = check_dt_cap(good)
    assert value == expected and (value is None or type(value) is float)
