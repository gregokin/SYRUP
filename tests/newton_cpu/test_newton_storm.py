"""Newton in the coupled step and storm driver (`StormControl.root_solver`), the prepared compiled hydrology, and the
explicit refusal of GPU/Newton combinations. Defaults stay the bisection."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from rfid_helpers import build, pit_chain
from test_hydrology_prepared import WATER_ATOL, WATER_RTOL, build_case
from test_routing import chain_full, make_graph, valley_full
from test_storm import budget, schedule_from, soil

pytest.importorskip("maple")

from maple_syrup import hydrology_numba as hn
from maple_syrup import routing_newton as rn
from maple_syrup.rainfall import rainfall_field
from maple_syrup.routing import DEFAULT_NEWTON_MAX_ITERATIONS, RoutingError
from maple_syrup.routing_numba import numba_available
from maple_syrup.storm import (
    HYDROGRAPH_COLUMNS,
    StormControl,
    StormError,
    coupled_step,
    evolve,
    initial_state,
)

NUMBA = pytest.mark.skipif(not numba_available(), reason="Numba not installed; compiled form not exercised, no claim made")
IMPLS = ["array", pytest.param("numba", marks=NUMBA)]
COL = {name: i for i, name in enumerate(HYDROGRAPH_COLUMNS)}


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


# --- control ------------------------------------------------------------------------------------------------------
def test_control_defaults_and_validation():
    assert StormControl().root_solver == "bisection" and StormControl().newton_max_iterations == DEFAULT_NEWTON_MAX_ITERATIONS
    assert StormControl() == StormControl(root_solver="bisection")
    assert StormControl(root_solver="newton").validated().root_solver == "newton"
    for bad in ({"root_solver": "brent"}, {"root_solver": None}, {"root_solver": "newton", "newton_max_iterations": 0},
                {"root_solver": "newton", "newton_max_iterations": True}, {"newton_max_iterations": 2.5},
                {"root_solver": "newton", "newton_max_iterations": rn.MAX_NEWTON_ITERATIONS + 1},
                {"root_solver": "newton", "newton_max_iterations": -1}, {"root_solver": "newton", "newton_max_iterations": "9"},
                {"root_solver": "newton", "implementation": "cuda"}):
        with pytest.raises(StormError):
            StormControl(**bad).validated()
    top = StormControl(root_solver="newton", newton_max_iterations=rn.MAX_NEWTON_ITERATIONS).validated()
    assert top.newton_max_iterations == rn.MAX_NEWTON_ITERATIONS


def test_cuda_newton_is_refused_before_anything_is_prepared_or_mutated():
    g = make_graph(valley_full(6, 5), ff=5.0)
    params, s0 = soil(g, 1e-6)
    state = initial_state(g, np.zeros(g.shape), s0)
    before = [a.copy() for a in (state.depth_m, state.soil_water_m, state.discharge_m2_s)]
    rate = np.full(g.shape, 1e-5)
    control = StormControl(implementation="cuda", root_solver="newton")  # deliberately not validated()
    with pytest.raises(StormError, match="CPU-only"):
        coupled_step(g, params, rate, state, 1.0, control)
    with pytest.raises(StormError, match="CPU-only"):
        evolve(g, params, rainfall_field(*g.shape), schedule_from([0.0, 10.0], [36.0]), state, 10.0, control,
               report_every_s=5.0)
    for a, b in zip((state.depth_m, state.soil_water_m, state.discharge_m2_s), before, strict=True):
        np.testing.assert_array_equal(a, b)


# --- coupled step and storm ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("impl", IMPLS)
def test_default_control_is_bitwise_the_explicit_bisection_storm(impl):
    g = make_graph(valley_full(6, 5), ff=5.0)
    s = schedule_from([0.0, 60.0, 90.0], [36.0, 0.0])
    params, s0 = soil(g, 1e-6, drain=0.05)
    out = []
    for control in (StormControl(implementation=impl),
                    StormControl(implementation=impl, root_solver="bisection", newton_max_iterations=3)):
        out.append(evolve(g, params, rainfall_field(*g.shape), s, initial_state(g, np.zeros(g.shape), s0), 120.0,
                          control, report_every_s=20.0))
    for name in ("depth_m", "soil_water_m", "discharge_m2_s"):
        same_bits(getattr(out[0].state, name), getattr(out[1].state, name))
    same_bits(out[0].hydrograph, out[1].hydrograph)


def run_storm(g, params, s, s0, impl, solver, end=200.0, **kw):
    return evolve(g, params, rainfall_field(*g.shape), s, initial_state(g, np.zeros(g.shape), s0), end,
                  StormControl(implementation=impl, root_solver=solver, **kw), report_every_s=25.0)


def test_numpy_and_numba_newton_storms_are_bitwise_identical():
    pytest.importorskip("numba")
    g = make_graph(valley_full(6, 5), ff=5.0)
    s = schedule_from([0.0, 60.0, 90.0, 150.0], [36.0, 0.0, 18.0])
    params, s0 = soil(g, 1e-6, drain=0.05)
    a, b = (run_storm(g, params, s, s0, impl, "newton") for impl in ("array", "numba"))
    for name in ("depth_m", "soil_water_m", "discharge_m2_s"):
        same_bits(getattr(a.state, name), getattr(b.state, name))
    same_bits(a.hydrograph, b.hydrograph)
    assert float(a.cumulative_export_m3) == float(b.cumulative_export_m3) > 0.0
    assert (a.n_accepted_steps, a.n_rejected_attempts) == (b.n_accepted_steps, b.n_rejected_attempts)


@pytest.mark.parametrize("impl", IMPLS)
def test_newton_storm_closes_the_water_budget_and_tracks_the_bisection_storm(impl):
    g = make_graph(chain_full(6), ff=5.0)
    area = g.dx_m ** 2
    s = schedule_from([0.0, 120.0, 180.0, 300.0], [36.0, 0.0, 36.0])
    params, s0 = soil(g, 1e-6, drain=0.05)
    state0 = initial_state(g, np.zeros(g.shape), s0)
    n = run_storm(g, params, s, s0, impl, "newton", end=480.0)
    b = run_storm(g, params, s, s0, impl, "bisection", end=480.0)
    residual, tol, rain = budget(g, state0, n, area)
    assert abs(residual) <= tol < 1e-9
    assert rain == pytest.approx(s.depth_m(0.0, 480.0) * g.n_active * area, rel=1e-13)
    assert n.n_accepted_steps == b.n_accepted_steps == 480 and n.n_rejected_attempts == 0
    hn_, hb = np.asarray(n.hydrograph), np.asarray(b.hydrograph)
    for name in ("surface_storage_m3", "cumulative_export_m3", "outlet_discharge_m3_s"):
        np.testing.assert_allclose(hn_[:, COL[name]], hb[:, COL[name]], rtol=1e-9, atol=1e-13, err_msg=name)
    assert float(np.abs(n.state.depth_m - b.state.depth_m).max()) <= 1e-9 * float(b.state.depth_m.max())
    assert hn_[-1, COL["max_constitutive_residual_m"]] <= max(hb[-1, COL["max_constitutive_residual_m"]], 1e-15)


@pytest.mark.parametrize("impl", IMPLS)
def test_pit_storm_with_newton_retains_the_water_in_the_pit(impl):
    g = build(pit_chain(), allow_pit_storage=True)
    area = g.dx_m ** 2
    s = schedule_from([0.0, 90.0], [36.0])
    params, s0 = soil(g, 1e-7)
    state0 = initial_state(g, np.zeros(g.shape), s0)
    n = run_storm(g, params, s, s0, impl, "newton", end=150.0)
    b = run_storm(g, params, s, s0, impl, "bisection", end=150.0)
    residual, tol, _rain = budget(g, state0, n, area)
    assert abs(residual) <= tol
    assert float(n.state.depth_m[g.pit_storage].sum()) > 0.0
    np.testing.assert_allclose(n.state.depth_m, b.state.depth_m, rtol=1e-9, atol=1e-15)


def test_failed_newton_storm_leaves_the_caller_state_intact():
    g = make_graph(valley_full(6, 5), ff=5.0)
    s = schedule_from([0.0, 30.0], [36.0])
    params, s0 = soil(g, 1e-6)
    state0 = initial_state(g, np.zeros(g.shape), s0)
    before = [a.copy() for a in (state0.depth_m, state0.soil_water_m, state0.discharge_m2_s)]
    with pytest.raises(RoutingError, match="Newton root solver did not reach"):
        evolve(g, params, rainfall_field(*g.shape), s, state0, 30.0,
               StormControl(root_solver="newton", root_tolerance_m=1e-300), report_every_s=10.0)
    for a, b in zip((state0.depth_m, state0.soil_water_m, state0.discharge_m2_s), before, strict=True):
        np.testing.assert_array_equal(a, b)


# --- prepared compiled hydrology ----------------------------------------------------------------------------------
@NUMBA
@pytest.mark.parametrize("kind,model", [("random", "pavement_hawkins"), ("valley", "fixed_ksat"),
                                        ("valley_masked", "pavement_hawkins")])
def test_prepared_newton_matches_the_reference_newton_step(kind, model):
    case = build_case(21, kind=kind, model=model)
    for dt in (0.5, 1.0):
        ref = coupled_step(case.graph, case.params, case.rate_on, case.state, dt,
                           StormControl(implementation="numba", root_solver="newton"))
        got = hn.prepared_coupled_step(case.ctx, case.rate_on, case.state, dt,
                                       StormControl(implementation="numba", root_solver="newton"))
        assert got.route.root_solver == "newton" and got.route.implementation == "numba"
        assert got.route.bisection_iterations == 0 and got.route.newton_max_iterations == DEFAULT_NEWTON_MAX_ITERATIONS
        assert set(got.route.root_stats) == set(rn.STAT_NAMES) and got.route.root_stats["fallback_cells"] == 0
        for name in ("depth_m", "flow_depth_m", "discharge_m2_s", "velocity_m_s", "inflow_m2_s", "face_volume_m3"):
            np.testing.assert_allclose(getattr(got.route, name), getattr(ref.route, name), rtol=WATER_RTOL,
                                       atol=WATER_ATOL, err_msg=name)
        np.testing.assert_allclose(got.state.depth_m, ref.state.depth_m, rtol=WATER_RTOL, atol=WATER_ATOL)
        assert float(got.route.max_constitutive_residual_m) <= 1e-14
        assert abs(float(got.route.budget_residual_m3)) <= 1e-12


@NUMBA
def test_prepared_newton_agrees_with_prepared_bisection_within_the_bisection_accuracy():
    case = build_case(22, kind="random", model="pavement_hawkins")
    b = hn.prepared_coupled_step(case.ctx, case.rate_on, case.state, 1.0, StormControl(implementation="numba"))
    n = hn.prepared_coupled_step(case.ctx, case.rate_on, case.state, 1.0,
                                 StormControl(implementation="numba", root_solver="newton"))
    assert b.route.root_solver == "bisection" and b.route.root_stats is None and b.route.bisection_iterations == 40
    assert float(np.abs(n.route.flow_depth_m - b.route.flow_depth_m).max()) <= 1e-12 * float(b.route.flow_depth_m.max())
    assert float(np.abs(n.route.depth_m - b.route.depth_m).max()) <= 1e-12 * float(b.route.depth_m.max())
    assert float(n.route.max_constitutive_residual_m) <= float(b.route.max_constitutive_residual_m)


@NUMBA
def test_prepared_default_kernels_are_unchanged_and_separate_from_the_newton_kernels():
    from maple_syrup import routing_numba as rnb

    default, newton = hn._kernels(), hn._kernels("newton")
    assert default is not newton and default is hn._kernels() and newton is hn._kernels("newton")
    assert default.sweep.py_func is rnb._sweep_batched  # the default build still calls the batched bisection sweep
    assert newton.sweep is rn.compiled_sweep_newton()
    for kernel in (newton.column, newton.route):
        assert not kernel.targetoptions.get("fastmath", False) and not kernel.targetoptions.get("parallel", False)
    prov = hn.kernel_provenance()
    assert prov["newton_compiled_in_process"] is True and "newton" in prov["root_solvers"]
    assert len(prov["routing_newton_sha256"]) == 64


@NUMBA
def test_prepared_option_errors_and_nonconvergence_name_the_newton_solver():
    case = build_case(23, kind="valley", model="fixed_ksat")
    before = [a.copy() for a in (case.state.depth_m, case.state.soil_water_m, case.state.discharge_m2_s)]
    for control, match in ((StormControl(implementation="numba", root_solver="nope"), "root_solver"),
                           (StormControl(implementation="numba", root_solver="newton", newton_max_iterations=0),
                            "newton_max_iterations"),
                           (StormControl(implementation="numba", root_solver="newton", root_tolerance_m=1e-300),
                            "Newton root solver did not reach")):
        with pytest.raises(RoutingError, match=match):
            hn.prepared_coupled_step(case.ctx, case.rate_on, case.state, 1.0, control)
    with pytest.raises(RoutingError, match="bisection did not reach"):
        hn.prepared_coupled_step(case.ctx, case.rate_on, case.state, 1.0,
                                 StormControl(implementation="numba", root_tolerance_m=1e-300))
    for a, b in zip((case.state.depth_m, case.state.soil_water_m, case.state.discharge_m2_s), before, strict=True):
        np.testing.assert_array_equal(a, b)


@NUMBA
def test_prepared_newton_on_a_pit_graph_matches_the_reference():
    g = build(pit_chain(), allow_pit_storage=True)
    params, s0 = soil(g, 1e-7)
    ctx = hn.prepare_hydrology(g, params)
    state = initial_state(g, np.where(g.active, 2e-3, 0.0), s0)
    rate = np.where(g.active, 1e-5, 0.0)
    control = StormControl(implementation="numba", root_solver="newton")
    ref = coupled_step(g, params, rate, state, 1.0, control)
    got = hn.prepared_coupled_step(ctx, rate, state, 1.0, control)
    for name in ("depth_m", "flow_depth_m", "discharge_m2_s"):
        np.testing.assert_allclose(getattr(got.route, name), getattr(ref.route, name), rtol=WATER_RTOL, atol=WATER_ATOL)
    assert not got.route.discharge_m2_s[g.pit_storage].any()
    assert dataclasses.is_dataclass(got.route) and abs(float(got.route.budget_residual_m3)) <= 1e-13
