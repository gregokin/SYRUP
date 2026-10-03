"""Phase 7j: the prepared hydrology (now on the level-batched sweep) against
(a) the SAME prepared hydrology built on the original compiled sweep (bitwise, every public field), and
(b) the unchanged reference `storm.coupled_step` (declared water bound rtol 2e-12 / atol 1e-14).

(a) swaps the sweep dispatcher that `hydrology_numba._build_kernels` binds, then rebuilds the kernels: the old
variant is the Phase 7h prepared path exactly. Case builders, comparison and failure mutations are the committed
Phase 7h ones. Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import contextlib
import dataclasses

import numpy as np
import pytest

pytest.importorskip("maple")

from phase7h.test_hydrology_prepared import (
    CONTROL,
    MUTATIONS,
    _adv,
    _chain_case,
    _find_inconsistent_wet_depth,
    _find_partial_positive_rain_depth,
    build_case,
    compare,
    run_pair,
)

from maple_syrup import hydrology_numba as hn
from maple_syrup import routing_numba as rn
from maple_syrup.routing import RoutingError, RoutingStepRejected
from maple_syrup.storm import StormControl, StormState, coupled_step, initial_state

pytestmark = pytest.mark.skipif(not rn.numba_available(), reason="Numba not installed; no claim made")


_ORIGINAL_KERNELS = None  # test-only cache: the prepared kernels built on the ORIGINAL serial sweep


def kernels_sweep():
    """The sweep function the currently selected prepared kernels call (builds them lazily, as production does)."""
    return hn._kernels().sweep.py_func


@contextlib.contextmanager
def original_sweep():
    """Test-only: select prepared kernels built on the ORIGINAL serial compiled sweep (built once per process by
    temporarily binding the old factory), then restore exactly the previously selected kernels object and factory,
    so neither side is ever recompiled and no state leaks between tests."""
    global _ORIGINAL_KERNELS
    saved_factory = hn.compiled_sweep_batched
    saved_kernels = hn._KERNELS
    try:
        if _ORIGINAL_KERNELS is None:
            hn._KERNELS = None
            hn.compiled_sweep_batched = rn.compiled_sweep
            _ORIGINAL_KERNELS = hn._kernels()
        hn._KERNELS = _ORIGINAL_KERNELS
        assert kernels_sweep() is rn._sweep, "the original-sweep kernels are not the ones in use"
        yield
    finally:
        hn.compiled_sweep_batched = saved_factory
        hn._KERNELS = saved_kernels


def assert_bitwise(a, b, path="step"):
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        assert type(a) is type(b), path
        for f in dataclasses.fields(a):
            assert_bitwise(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif a is None:
        assert b is None, path
    elif isinstance(a, np.ndarray):
        assert type(b) is np.ndarray and a.dtype == b.dtype and a.shape == b.shape, path
        if a.dtype.kind == "f":
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=path)
            keep = ~np.isnan(a)
            np.testing.assert_array_equal(a[keep].view(np.uint64), b[keep].view(np.uint64), err_msg=path)
        else:
            np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert type(a) is type(b), path
        if isinstance(a, (float, np.floating)):
            both_nan = bool(np.isnan(a) and np.isnan(b))
            assert both_nan or np.float64(a).view(np.uint64) == np.float64(b).view(np.uint64), path
        else:
            assert a == b, path


def trajectory(case, rates, dts, control=CONTROL, *, sweep):
    """Prepared steps with `sweep` (`rn._sweep_batched` or `rn._sweep`) asserted to be the one actually in use."""
    assert kernels_sweep() is sweep
    steps, state = [], case.state
    for rate, dt in zip(rates, dts, strict=True):
        step = hn.prepared_coupled_step(case.ctx, rate, state, dt, control)
        steps.append(step)
        state = step.state
    return steps


def old_and_new(seed, kind, model, rates_fn, dts, **kw):
    case = build_case(seed, kind=kind, model=model, **kw)
    rates = rates_fn(case)
    new = trajectory(case, rates, dts, sweep=rn._sweep_batched)
    with original_sweep():
        case_old = build_case(seed, kind=kind, model=model, **kw)
        old = trajectory(case_old, rates, dts, sweep=rn._sweep)
    return old, new


def wet_then_recession(case):
    zero = np.zeros(case.graph.shape)
    return [case.rate_on] * 14 + [zero] * 12


# --- (a) bitwise against the Phase 7h prepared path --------------------------------------------------------------
@pytest.mark.parametrize("model", ["pavement_hawkins", "fixed_ksat"])
@pytest.mark.parametrize("kind", ["random", "valley", "valley_masked"])
@pytest.mark.parametrize("seed", [1, 2])
def test_all_public_fields_are_bitwise_equal_to_the_prepared_path_on_the_original_sweep(seed, kind, model):
    dts = ([1.0, 0.5, 0.25, 1.0 / 1024.0, 1.0, 1.0, 0.5] * 4)[:26]
    old, new = old_and_new(seed, kind, model, wet_then_recession, dts)
    totals_wet = sum(int(s.n_no_runon) + int(s.n_partial_runon) for s in new)
    assert totals_wet > 0
    for i, (a, b) in enumerate(zip(old, new, strict=True)):
        assert_bitwise(a, b, f"step{i}")
    # dry recession tail: the batched sweep skipped non-positive-RHS cells there, results stay equal (asserted above)
    assert float(new[-1].route.outlet_discharge_m3_s) >= 0.0


@pytest.mark.parametrize("model", ["pavement_hawkins", "fixed_ksat"])
def test_dry_start_fully_dry_domain_and_slow_dt_are_bitwise_equal(model):
    def dry(case):
        return [np.zeros(case.graph.shape)] * 3 + [case.rate_on] * 6 + [np.zeros(case.graph.shape)] * 3

    case = build_case(21, kind="valley", model=model)
    state = initial_state(case.graph, np.zeros(case.graph.shape), case.state.soil_water_m)
    case = dataclasses.replace(case, state=state)
    rates = dry(case)
    dts = [1.0] * 12
    new = trajectory(case, rates, dts, sweep=rn._sweep_batched)
    with original_sweep():
        old_case = dataclasses.replace(build_case(21, kind="valley", model=model), state=state)
        old = trajectory(old_case, rates, dts, sweep=rn._sweep)
    for i, (a, b) in enumerate(zip(old, new, strict=True)):
        assert_bitwise(a, b, f"step{i}")
    # first dry steps are exact zeros everywhere
    for name in ("depth_m", "flow_depth_m", "discharge_m2_s", "velocity_m_s", "inflow_m2_s", "face_volume_m3"):
        assert not np.any(getattr(new[0].route, name)), name


def test_dt_courant_iteration_and_tolerance_controls_are_bitwise_equal():
    for dt, courant, iterations, root_tol in ((1.0, 1.0, 40, 1e-11), (0.25, 0.5, 60, 1e-12),
                                              (1.0 / 1024.0, 2.0, 30, 1e-9), (0.5, 1, np.int64(45), 1e-10)):
        control = StormControl(courant_max=courant, bisection_iterations=iterations, root_tolerance_m=root_tol,
                               implementation="numba")
        case = build_case(6, kind="valley", model="pavement_hawkins")
        new = trajectory(case, [case.rate_on] * 5, [dt] * 5, control, sweep=rn._sweep_batched)
        with original_sweep():
            old = trajectory(build_case(6, kind="valley", model="pavement_hawkins"), [case.rate_on] * 5, [dt] * 5,
                             control, sweep=rn._sweep)
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            assert_bitwise(a, b, f"dt{dt}/step{i}")


# --- (b) declared bound against the unchanged reference ----------------------------------------------------------
@pytest.mark.parametrize("model", ["pavement_hawkins", "fixed_ksat"])
@pytest.mark.parametrize("kind", ["random", "valley_masked"])
def test_matches_the_unchanged_reference_within_the_declared_water_bound(kind, model):
    case = build_case(3, kind=kind, model=model)
    totals = run_pair(case, wet_then_recession(case), [1.0] * 26)
    assert totals["intake_m"] > 0.0


# --- failures: identical class, message and non-mutation on both sweeps and the reference -------------------------
def _failure(case, d, sweep):
    assert kernels_sweep() is sweep
    state = StormState(0.0, d["depth"], d["soil"], d["q"])
    with pytest.raises(Exception) as err:
        hn.prepared_coupled_step(case.ctx, d["rate"], state, d["dt"], d["control"])
    return err.value


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_every_refusal_matches_the_original_sweep_and_the_reference_and_mutates_nothing(name):
    case, d = _adv()
    MUTATIONS[name](d)
    before = [np.array(a, copy=True) for a in (d["rate"], d["depth"], d["soil"], d["q"])]
    new_err = _failure(case, d, rn._sweep_batched)
    with original_sweep():
        case_old, d_old = _adv()
        MUTATIONS[name](d_old)
        old_err = _failure(case_old, d_old, rn._sweep)
    state = StormState(0.0, d["depth"], d["soil"], d["q"])
    with pytest.raises(Exception) as ref_err:
        coupled_step(case.graph, case.params, d["rate"], state, d["dt"], d["control"])
    assert type(new_err) is type(old_err) is type(ref_err.value), name
    assert str(new_err) == str(old_err) == str(ref_err.value), name
    for a, b in zip(before, (d["rate"], d["depth"], d["soil"], d["q"]), strict=True):
        np.testing.assert_array_equal(a, b)


def test_bisection_five_iterations_is_refused_with_the_non_recoverable_class_and_precedence():
    case, d = _adv()
    MUTATIONS["bisection_not_converged"](d)
    err = _failure(case, d, rn._sweep_batched)
    assert isinstance(err, RoutingError) and not isinstance(err, RoutingStepRejected)
    assert "bisection did not reach" in str(err)
    # a column failure still outranks it, a Courant rejection still stays recoverable
    case, d = _adv()
    MUTATIONS["bisection_not_converged"](d)
    MUTATIONS["soil_nan"](d)
    assert type(_failure(case, d, rn._sweep_batched)).__name__ == "InfiltrationError"
    case, d = _adv()
    MUTATIONS["courant_rejection_is_recoverable"](d)
    assert isinstance(_failure(case, d, rn._sweep_batched), RoutingStepRejected)


@pytest.mark.parametrize("finder, case_kwargs", [
    (_find_inconsistent_wet_depth, {"rate_value": 1e-5}),
    (_find_partial_positive_rain_depth, {"rate_value": 1e-5, "ksat": 3e-5}),
])
def test_former_positive_rain_roundoff_triggers_now_run_identically_in_every_path(finder, case_kwargs):
    """The seeded inputs that used to trip `old_flow_depth_m exceeds depth_start_m` (hpre one ulp above h*) now run: the column
    depth and the old-flow depth share one arithmetic. The reference, the prepared batched sweep and the prepared ORIGINAL serial
    sweep all accept the input; the two prepared paths are bitwise equal, both agree with the reference at the unchanged water
    bound, the per-step budget closes, and no input is mutated. The route guard itself is unchanged (genuine bad-old-depth
    refusals are tested in phase7h/phase4s/rfid)."""
    h = finder()
    case = _chain_case(0.0, **case_kwargs)
    depth = np.zeros(case.graph.shape)
    depth[1, 0] = h
    soil = np.zeros(case.graph.shape) if finder is _find_inconsistent_wet_depth else case.params.storage_max_m - 1e-4
    state = initial_state(case.graph, depth, soil)
    rate = np.full(case.graph.shape, 1e-5)
    before = [a.copy() for a in (depth, soil, rate, state.discharge_m2_s)]
    ref = coupled_step(case.graph, case.params, rate, state, 1.0, CONTROL)
    new = hn.prepared_coupled_step(case.ctx, rate, state, 1.0, CONTROL)
    with original_sweep():
        old_case = _chain_case(0.0, **case_kwargs)
        old = hn.prepared_coupled_step(old_case.ctx, rate, state, 1.0, CONTROL)
    assert_bitwise(old, new)
    compare(ref, new, exact=False)
    assert abs(float(new.route.budget_residual_m3)) <= 1e-13 and abs(float(ref.route.budget_residual_m3)) <= 1e-13
    branch = new.n_complete_runon if finder is _find_inconsistent_wet_depth else new.n_partial_runon
    assert int(branch) >= 1
    for a, b in zip(before, (depth, soil, rate, state.discharge_m2_s), strict=True):
        np.testing.assert_array_equal(a, b)


def test_missing_numba_after_preparation_is_an_explicit_error_without_fallback(monkeypatch):
    case = build_case(2, kind="valley", model="fixed_ksat")
    monkeypatch.setattr(hn, "_KERNELS", None)
    monkeypatch.setattr(hn, "numba_available", lambda: False)
    before = case.state.depth_m.copy()
    with pytest.raises(hn.HydrologyNumbaUnavailableError):
        hn.prepared_coupled_step(case.ctx, case.rate_on, case.state, 1.0, CONTROL)
    np.testing.assert_array_equal(case.state.depth_m, before)


# --- ownership, continuation, provenance -------------------------------------------------------------------------
def test_static_context_and_inputs_are_unchanged_and_outputs_are_fresh_and_continuation_is_exact():
    case = build_case(13, kind="random", model="pavement_hawkins")
    static_before = {n: np.array(getattr(case.ctx, n), copy=True) for n in hn.HydrologyContext._ARRAYS}
    rates = wet_then_recession(case)
    full = trajectory(case, rates, [1.0] * len(rates), sweep=rn._sweep_batched)
    for n, a in static_before.items():
        np.testing.assert_array_equal(getattr(case.ctx, n), a, err_msg=n)
        assert not getattr(case.ctx, n).flags.writeable
    # restart in the middle with a REBUILT context and copied state arrays
    mid = full[9].state
    copied = StormState(mid.t_s, mid.depth_m.copy(), mid.soil_water_m.copy(), mid.discharge_m2_s.copy())
    ctx2 = hn.prepare_hydrology(case.graph, case.params)
    state, resumed = copied, []
    for rate in rates[10:]:
        step = hn.prepared_coupled_step(ctx2, rate, state, 1.0, CONTROL)
        resumed.append(step)
        state = step.state
    for i, (a, b) in enumerate(zip(full[10:], resumed, strict=True)):
        assert_bitwise(a, b, f"resumed{i}")
    first, second = full[0], full[1]
    assert not np.shares_memory(first.route.depth_m, second.route.depth_m)


def test_provenance_names_the_batched_sweep_and_the_original_stays_reachable():
    info = hn.kernel_provenance()
    assert "_sweep_batched" in info["ordered_sweep"] and info["numba_options"]["fastmath"] is False
    assert rn.compiled_sweep().py_func is rn._sweep
