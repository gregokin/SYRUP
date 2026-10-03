"""Adaptive-step continuation of the experimental driver on a real device: the same root scenarios as the CPU test (non-binary
forcing edge, zero rain, report cadence = max_dt, deep lake so the CFL / positivity bounds really reject steps). A CUDA run to a
boundary plus a resumed CUDA run must equal one continuous CUDA run (bitwise state and face momentum, counts add up), the
continued CUDA trajectory must match the continued NumPy reference (field bounds rtol 2e-12 / atol 1e-14, accepted and rejected
counts exact), and the cap travels with the state without any host-to-device array. Skipped without a device. Nothing here was run
by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from cand_cases import build, device_twin, field, host, schedule

pytest.importorskip("maple")

from maple_syrup.experimental_hydrology import ExperimentalHydrologyError
from maple_syrup.experimental_storm import ExperimentalControl, evolve_experimental

pytestmark = pytest.mark.usefixtures("gpu")
RTOL, ATOL = 2.0e-12, 1.0e-14
CASES = [("explicit", 4.0, 0.2), ("explicit", 1.0, 1.0), ("local_inertial", 4.0, 0.02), ("local_inertial", 1.0, 0.1)]


def cupy():
    import cupy as cp

    return cp


@pytest.mark.parametrize(("method", "max_dt", "depth"), CASES)
def test_a_resumed_cuda_run_equals_one_continuous_run_and_matches_the_reference(method, max_dt, depth):
    cp = cupy()
    cs = build("valley:6x5", method, depth="dry", seed=3)
    dev = device_twin(cs, cp)
    initial = np.where(cs.graph.active, depth, 0.0)
    cpu0 = cs.solver.initial_state(initial, cs.state.soil_water_m)
    gpu0 = dev.solver.initial_state(cp.asarray(initial), cp.asarray(cs.state.soil_water_m))
    end = 3.0 * max_dt
    sched = schedule([0.0, 0.451875 * max_dt, end], [0.0, 0.0])
    control = ExperimentalControl(max_dt_s=max_dt)

    def run_cpu(state, until):
        return evolve_experimental(cs.solver, field(cs), sched, state, until, control, report_every_s=max_dt)

    def run_gpu(state, until):
        return evolve_experimental(dev.solver, field(cs, cp), sched, state, until, control, report_every_s=max_dt)

    once, first = run_gpu(gpu0, end), run_gpu(gpu0, max_dt)
    assert once.n_rejected_attempts > 0, "the scenario must really exercise the adaptive cap"
    assert isinstance(first.state.next_dt_cap_s, float)  # a host number: it travels with the state with no device array
    second = run_gpu(first.state, end)
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        a, b = getattr(once.state, name), getattr(second.state, name)
        assert (a is None) == (b is None), name
        if a is not None:
            np.testing.assert_array_equal(host(a), host(b), err_msg=name)
    assert first.n_accepted_steps + second.n_accepted_steps == once.n_accepted_steps
    assert first.n_rejected_attempts + second.n_rejected_attempts == once.n_rejected_attempts
    assert second.state.next_dt_cap_s == once.state.next_dt_cap_s

    _ref_once, ref_first = run_cpu(cpu0, end), run_cpu(cpu0, max_dt)
    ref_second = run_cpu(ref_first.state, end)
    assert ref_first.n_accepted_steps + ref_second.n_accepted_steps == once.n_accepted_steps  # GPU counts = CPU counts
    assert ref_first.n_rejected_attempts + ref_second.n_rejected_attempts == once.n_rejected_attempts
    assert ref_second.state.next_dt_cap_s == second.state.next_dt_cap_s
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        a, b = getattr(second.state, name), getattr(ref_second.state, name)
        if a is not None:
            np.testing.assert_allclose(host(a), b, rtol=RTOL, atol=ATOL, err_msg=name)


@pytest.mark.parametrize("method", ("explicit", "local_inertial"))
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True, "1", [1.0]], ids=repr)
def test_a_malformed_cap_is_refused_on_the_device_before_any_launch(method, bad, monkeypatch):
    from maple_syrup import experimental_cuda as ec
    from maple_syrup import hydrology_cuda as hc

    cp = cupy()
    cs = build("valley:6x5", method, depth="wet", seed=4)
    dev = device_twin(cs, cp)
    state = dev.solver.initial_state(cp.asarray(cs.state.depth_m), cp.asarray(cs.state.soil_water_m))
    broken = dataclasses.replace(state, next_dt_cap_s=bad)

    def refuse(name, *args, **kwargs):
        pytest.fail(f"kernel {name} was enqueued but the malformed cap had to be refused first")

    monkeypatch.setattr(ec, "_launch", refuse)
    monkeypatch.setattr(hc, "_launch", refuse)
    with pytest.raises(ExperimentalHydrologyError, match="next_dt_cap_s"):
        dev.solver.validate_state(broken)
    with pytest.raises(ExperimentalHydrologyError, match="next_dt_cap_s"):
        dev.solver.step(cp.zeros(cs.shape), broken, 0.1)
    with pytest.raises(ExperimentalHydrologyError, match="next_dt_cap_s"):
        evolve_experimental(dev.solver, field(cs, cp), schedule([0.0, 5.0], [0.0]), broken, 5.0,
                            ExperimentalControl(max_dt_s=1.0), report_every_s=1.0)
