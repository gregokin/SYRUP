"""Phase 4c coupled storm helpers (`maple_syrup.storm`) on small controlled
graphs: branch selection of the old flux, saturation return, transactional
adaptive retries, guards, budgets, and array/Numba equality. No MAPLE case
is compiled here (see test_storm_experiment.py for Plot 1). Nothing here
compares with executed MAHLERAN Fortran; that is a later task."""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_routing import (
    NUMBA_SKIP,
    chain_full,
    implementations,
    make_graph,
    valley_full,
)

from maple_syrup import storm as storm_module
from maple_syrup.infiltration import (
    LOCAL_BALANCE_RTOL,
    column_parameters,
    initial_soil_water_m,
)
from maple_syrup.rainfall import (
    RainfallProvenance,
    RainfallSchedule,
    constant_rainfall,
    rainfall_field,
)
from maple_syrup.routing import (
    BALANCE_RTOL,
    RoutingError,
    RoutingStepRejected,
    route_step,
)
from maple_syrup.storm import (
    HYDROGRAPH_COLUMNS,
    StormControl,
    StormError,
    StormState,
    coupled_step,
    evolve,
    initial_state,
    plan_boundaries,
)

pytest.importorskip("maple")
COL = {name: i for i, name in enumerate(HYDROGRAPH_COLUMNS)}
RTOL = LOCAL_BALANCE_RTOL + BALANCE_RTOL


def soil(graph, ksat, *, theta0=0.25, theta_sat=0.4, thickness=0.3, suction=0.0, drain=0.0, active_mask=None):
    """fixed-Ksat columns: with zero suction and S well below saturation the
    capacity is exactly Ksat, so intake = min(h + rain, Ksat dt)."""

    def full(value):
        return np.full(graph.shape, float(value))

    params = column_parameters(model="fixed_ksat", ksat_m_per_s=full(ksat), suction_m=full(suction),
                               drainage_parameter=full(drain), theta_sat=full(theta_sat),
                               soil_thickness_m=full(thickness), active_mask=active_mask)
    return params, initial_soil_water_m(params, full(theta0))


def schedule_from(edges, mm_per_h):
    return RainfallSchedule(edges_s=edges, intensity_mm_per_h=mm_per_h, provenance=RainfallProvenance(kind="constant"))


def budget(graph, state0, result, area):
    """Global identity residual (m3) and its declared tolerance."""
    r = result
    rain, intake = float(r.cumulative_rain_m.sum()), float(r.cumulative_intake_m.sum())
    ret, drain = float(r.cumulative_saturation_return_m.sum()), float(r.cumulative_drainage_m.sum())
    export_depth = float(r.cumulative_export_m3) / area
    h0, s0 = float(state0.depth_m.sum()), float(state0.soil_water_m.sum())
    h1, s1 = float(r.state.depth_m.sum()), float(r.state.soil_water_m.sum())
    residual = (h1 + s1 + drain + export_depth) - (h0 + s0 + rain)
    magnitude = h0 + s0 + rain + intake + drain + ret + export_depth
    tol = RTOL * (r.n_accepted_steps + graph.n_active) * magnitude
    return residual * area, tol * area, rain * area


# --- rejection classification ----------------------------------------------------------------
def test_recoverable_rejections_are_a_narrow_subclass():
    g = make_graph(chain_full(3), ff=1.0)
    h = np.full(g.shape, 0.05)
    with pytest.raises(RoutingStepRejected, match="Courant"):
        route_step(g, h, h, 16.0)
    with pytest.raises(RoutingError) as info:
        route_step(g, h, h, 1.0, bisection_iterations=5)  # non-convergence is NOT recoverable by a smaller dt
    assert not isinstance(info.value, RoutingStepRejected)
    with pytest.raises(RoutingError) as info:
        route_step(g, h, h, 0.0)
    assert not isinstance(info.value, RoutingStepRejected)
    assert issubclass(RoutingStepRejected, RoutingError)


# --- one coupled step ----------------------------------------------------------------------------
def test_initial_state_discharge_is_consistent_with_depth():
    g = make_graph(chain_full(4), ff=5.0)
    h = np.full(g.shape, 2e-3)
    _params, s0 = soil(g, 1e-7)
    state = initial_state(g, h, s0)
    k = g.conveyance.reshape(g.shape)
    np.testing.assert_array_equal(state.discharge_m2_s, (np.sqrt(h) * h) * k)
    route_step(g, h, h, 1.0, old_discharge_m2_s=state.discharge_m2_s)  # accepted by the relation check
    assert not np.shares_memory(state.depth_m, h)


def test_initial_state_and_evolve_reject_invalid_states_at_the_boundary():
    """Bad input is refused up front, never masked by recomputing q in a
    branch of the coupled step."""
    g = make_graph(chain_full(4), ff=5.0)
    params, s0 = soil(g, 1e-6)
    field = rainfall_field(*g.shape)
    s = constant_rainfall(0.0, 10.0, 36.0)
    for bad, match in ((np.full(g.shape, -1e-3), ">= 0"), (np.full(g.shape, np.nan), "finite"),
                       (np.zeros(g.shape, dtype=np.float32), "float64"), (np.zeros((5, 1)), "shape"),
                       ([0.0] * 4, "array")):
        with pytest.raises(StormError, match=match):
            initial_state(g, bad, s0)
    with pytest.raises(StormError, match="t_s"):
        initial_state(g, np.zeros(g.shape), s0, t_s=-1.0)
    good = initial_state(g, np.full(g.shape, 1e-3), s0)
    inconsistent = StormState(0.0, good.depth_m, good.soil_water_m, good.discharge_m2_s * 2.0)
    with pytest.raises(StormError, match="not consistent"):
        evolve(g, params, field, s, inconsistent, 10.0, StormControl(), report_every_s=10.0)
    np.testing.assert_array_equal(inconsistent.depth_m, good.depth_m)  # nothing was touched
    with pytest.raises(StormError, match="rainfall field"):
        evolve(g, params, rainfall_field(5, 1), s, good, 10.0, StormControl(), report_every_s=10.0)
    # column active mask must equal the graph's active cells
    z = valley_full(4, 5)
    inactive = np.ones((4, 5), dtype=bool)
    inactive[3, 0] = False
    g2 = make_graph(z, active=inactive)
    params_all, s0_all = soil(g2, 1e-6)  # column mask all True != graph mask
    with pytest.raises(StormError, match="active_mask differs"):
        evolve(g2, params_all, rainfall_field(*g2.shape), s, initial_state(g2, np.zeros(g2.shape), s0_all), 10.0,
               StormControl(), report_every_s=10.0)
    params_ok, s0_ok = soil(g2, 1e-6, active_mask=inactive)
    r = evolve(g2, params_ok, rainfall_field(*g2.shape, active_mask=inactive), s,
               initial_state(g2, np.zeros(g2.shape), s0_ok), 10.0, StormControl(), report_every_s=10.0)
    assert r.n_accepted_steps == 10 and r.state.depth_m[3, 0] == 0.0
    # discharge on an inactive cell is refused
    q = np.array(r.state.discharge_m2_s, copy=True)
    q[3, 0] = 1e-6
    with pytest.raises(StormError, match="inactive"):
        evolve(g2, params_ok, rainfall_field(*g2.shape, active_mask=inactive), s,
               StormState(0.0, r.state.depth_m, r.state.soil_water_m, q), 10.0, StormControl(), report_every_s=10.0)


@pytest.mark.parametrize("impl", implementations())
def test_coupled_step_selects_the_legacy_old_flux_branch(impl):
    g = make_graph(chain_full(4), ff=5.0)
    control = StormControl(implementation=impl)
    rain_rate = 1e-5  # 36 mm/h
    field = rainfall_field(*g.shape)
    rate = field.apply(rain_rate)
    h = np.full(g.shape, 2e-3)
    k = g.conveyance.reshape(g.shape)

    # (a) no run-on: intake = Ksat dt < rain -> q_old is the carried q(2), hpre = h.
    params, s0 = soil(g, 1e-6)
    state = initial_state(g, h, s0)
    prev = route_step(g, h, h, 1.0, implementation=impl)  # a previous routed step to carry
    state = StormState(0.0, prev.depth_m, s0, prev.discharge_m2_s)
    step = coupled_step(g, params, rate, state, 1.0, control)
    assert np.all(step.column.intake_m <= step.column.rain_m)
    np.testing.assert_array_equal(step.route.old_discharge_m2_s, prev.discharge_m2_s)
    assert int(step.n_no_runon) == g.n_active and int(step.n_partial_runon) == int(step.n_complete_runon) == 0

    # (b) complete run-on: Ksat dt >= h + rain -> hpre = 0, q_old = 0.
    params, s0 = soil(g, 1.0)
    state = initial_state(g, h, s0)
    step = coupled_step(g, params, rate, state, 1.0, control)
    np.testing.assert_array_equal(step.column.intake_m, h + step.column.rain_m)
    assert not np.any(step.route.old_discharge_m2_s) and not np.any(step.state.depth_m)
    assert int(step.n_complete_runon) == g.n_active and int(step.n_no_runon) == int(step.n_partial_runon) == 0
    # (b') dry cell with intake == rain: complete AND no-run-on both hold; legacy precedence counts it
    # complete (infilt.for 106 before 125) and both branches give q_old = 0.
    dry = initial_state(g, np.zeros(g.shape), s0)
    step = coupled_step(g, params, rate, dry, 1.0, control)
    assert int(step.n_complete_runon) == g.n_active and int(step.n_no_runon) == 0
    assert not np.any(step.route.old_discharge_m2_s)

    # (c) partial run-on: rain < Ksat dt < h + rain -> hpre = h - (J - P), q_old = k hpre^1.5.
    params, s0 = soil(g, 1e-3)
    state = initial_state(g, h, s0)
    step = coupled_step(g, params, rate, state, 1.0, control)
    hpre = h - (step.column.intake_m - step.column.rain_m)
    assert np.all(hpre > 0.0) and np.all(hpre < h)
    np.testing.assert_array_equal(step.route.old_discharge_m2_s, (np.sqrt(hpre) * hpre) * k)
    assert int(step.n_partial_runon) == g.n_active
    # the step conserves water: area (h' + S') + export = area (h + S + rain - drainage)
    area = g.dx_m ** 2
    lhs = area * float((step.state.depth_m + step.state.soil_water_m).sum()) + float(step.route.export_m3)
    rhs = area * float((h + s0 + step.column.rain_m - step.column.drainage_m).sum())
    assert abs(lhs - rhs) <= RTOL * (1 + g.n_active) * rhs
    assert step.state.t_s == 1.0 and step.route.implementation == impl


def test_saturation_return_reaches_the_surface_and_is_routed():
    g = make_graph(chain_full(3), ff=5.0)
    smax = 0.4 * 0.3
    params, s0 = soil(g, 1.0, theta0=(smax - 1e-6) / 0.3)  # 1 um below saturation, huge Ksat
    state = initial_state(g, np.zeros(g.shape), s0)
    field = rainfall_field(*g.shape)
    step = coupled_step(g, params, field.apply(1e-5), state, 1.0, StormControl())
    assert np.all(step.column.saturation_return_m > 0.0) and np.all(step.column.intake_m == step.column.rain_m)
    np.testing.assert_allclose(step.state.soil_water_m, smax, rtol=0, atol=1e-17)
    assert int(step.n_complete_runon) == g.n_active and not np.any(step.route.old_discharge_m2_s)
    assert float(step.route.export_m3) > 0.0 and np.all(step.state.depth_m[1:] > 0.0)


# --- boundaries and guards -----------------------------------------------------------------------
def test_plan_boundaries_knots_cadence_end_and_refusals():
    s = schedule_from([0.0, 120.0, 180.0, 300.0], [36.0, 0.0, 36.0])
    b = plan_boundaries(s, 0.0, 480.0, 50.0)
    assert b.tolist() == sorted({50.0, 100.0, 120.0, 150.0, 180.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 480.0})
    assert plan_boundaries(s, 0.0, 60.0, 1000.0).tolist() == [60.0]  # partial window: no knot inside, only end
    assert plan_boundaries(s, 130.0, 180.0, 20.0).tolist() == [150.0, 170.0, 180.0]
    with pytest.raises(StormError, match="max_report_rows"):
        plan_boundaries(s, 0.0, 480.0, 1.0, max_report_rows=10)
    for start, end, every in ((0.0, 0.0, 1.0), (10.0, 5.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 1.0, 1.0)):
        with pytest.raises(StormError):
            plan_boundaries(s, start, end, every)
    with pytest.raises(StormError, match="finite"):
        plan_boundaries(s, 0.0, math.inf, 1.0)


def test_control_validation_is_strict_and_never_coerces():
    with pytest.raises(StormError, match="min_dt_s"):
        StormControl(max_dt_s=1.0, min_dt_s=2.0).validated()
    with pytest.raises(StormError, match="implementation"):
        StormControl(implementation="fortran").validated()
    for kwargs, match in (({"max_retries": 0}, "max_retries"), ({"max_retries": 1.5}, "max_retries"),
                          ({"max_retries": True}, "max_retries"), ({"max_steps": True}, "max_steps"),
                          ({"max_steps": 2.0}, "max_steps"), ({"min_dt_s": True}, "min_dt_s"),
                          ({"max_dt_s": "1"}, "max_dt_s"), ({"bisection_iterations": 2.5}, "bisection_iterations")):
        with pytest.raises(StormError, match=match):
            StormControl(**kwargs).validated()
    assert StormControl().validated().implementation == "array"
    s = constant_rainfall(0.0, 10.0, 36.0)
    for rows in (0, True, 5.0):
        with pytest.raises(StormError, match="max_report_rows"):
            plan_boundaries(s, 0.0, 10.0, 1.0, max_report_rows=rows)


# --- retry floor versus forced short slices ---------------------------------------------------------
def _small(n=3, ksat=1e-6):
    g = make_graph(chain_full(n), ff=5.0)
    params, s0 = soil(g, ksat)
    return g, params, rainfall_field(*g.shape), initial_state(g, np.zeros(g.shape), s0)


def test_short_end_below_the_retry_floor_runs_without_prior_rejection():
    """end_s = 1e-4 s with the default floor 1/1024 s: a single forced slice,
    no rejection, no IndexError (Codex early_repro.log)."""
    g, params, field, state0 = _small()
    r = evolve(g, params, field, constant_rainfall(0.0, 10.0, 36.0), state0, 1e-4, StormControl(), report_every_s=1.0)
    assert r.n_accepted_steps == 1 and r.n_rejected_attempts == 0 and r.state.t_s == 1e-4
    assert float(r.cumulative_rain_m.sum()) == pytest.approx(1e-5 * 1e-4 * g.n_active, rel=1e-13)
    assert r.min_accepted_dt_s == 1e-4 < StormControl().min_dt_s


def test_short_forcing_interval_is_a_forced_slice_not_a_retry_failure():
    g, params, field, state0 = _small()
    s = schedule_from([0.0, 10.0, 10.0005, 20.0], [36.0, 72.0, 36.0])  # a 0.5 ms forcing interval
    r = evolve(g, params, field, s, state0, 20.0, StormControl(), report_every_s=5.0)
    assert 10.0005 in r.boundaries.tolist() and r.n_rejected_attempts == 0
    assert r.min_accepted_dt_s == pytest.approx(0.0005, rel=1e-9) and r.min_accepted_dt_s < StormControl().min_dt_s
    assert float(r.cumulative_rain_m.sum()) == pytest.approx(s.depth_m(0.0, 20.0) * g.n_active, rel=1e-12)


def test_nearly_coincident_decimal_report_and_forcing_edges_are_merged():
    """0.1 * 3 = 0.30000000000000004 is the same instant as the 0.3 forcing
    edge: one boundary, the exact edge kept, no rain lost or skipped."""
    g, params, field, state0 = _small()
    s = schedule_from([0.0, 0.3, 0.6, 0.9], [36.0, 0.0, 72.0])
    b = plan_boundaries(s, 0.0, 1.0, 0.1)
    assert 0.3 in b.tolist() and 0.6 in b.tolist() and 0.9 in b.tolist() and 1.0 == b[-1]
    assert np.all(np.diff(b) > 0.05) and b.size == 10  # no sub-eps slice, nothing skipped
    r = evolve(g, params, field, s, state0, 1.0, StormControl(max_dt_s=0.05), report_every_s=0.1)
    assert r.n_rejected_attempts == 0 and r.state.t_s == 1.0
    assert float(r.cumulative_rain_m.sum()) == pytest.approx(s.depth_m(0.0, 1.0) * g.n_active, rel=1e-12)
    assert r.min_accepted_dt_s > 0.01  # only genuine slices were stepped


def test_halving_below_the_retry_floor_is_refused_only_after_a_rejection():
    g = make_graph(chain_full(4), ff=1.0)
    params, s0 = soil(g, 0.0)
    state0 = initial_state(g, np.full(g.shape, 0.05), s0)  # Courant > 1 at dt 8, < 1 at dt 1
    s = constant_rainfall(0.0, 64.0, 0.0)
    with pytest.raises(StormError, match="retry floor") as info:
        evolve(g, params, rainfall_field(*g.shape), s, state0, 64.0,
               StormControl(max_dt_s=8.0, min_dt_s=6.0, max_retries=10), report_every_s=16.0)
    assert "rejection" in str(info.value) and isinstance(info.value.__cause__, RoutingStepRejected)
    # the same floor is fine when the state never needs a retry
    calm = initial_state(g, np.full(g.shape, 1e-3), s0)
    r = evolve(g, params, rainfall_field(*g.shape), s, calm, 64.0,
               StormControl(max_dt_s=8.0, min_dt_s=6.0, max_retries=10), report_every_s=16.0)
    assert r.n_rejected_attempts == 0 and r.n_accepted_steps == 8


def test_evolve_never_mutates_caller_arrays_even_when_it_fails(monkeypatch):
    """The scratch rain-rate array is owned by `evolve` (no caller-supplied
    buffer), so caller state cannot be overwritten before a later error."""
    g, params, field, state0 = _small()
    depth = np.full(g.shape, 1e-3)
    state = initial_state(g, depth, state0.soil_water_m)
    copies = [np.array(a, copy=True) for a in (state.depth_m, state.soil_water_m, state.discharge_m2_s, depth)]
    s = constant_rainfall(0.0, 10.0, 36.0)

    def failing(*args, **kwargs):
        raise RoutingError("bisection did not reach root_tolerance_m (test)")

    monkeypatch.setattr(storm_module, "route_step", failing)
    with pytest.raises(RoutingError):
        evolve(g, params, field, s, state, 10.0, StormControl(), report_every_s=10.0)
    for before, after in zip(copies, (state.depth_m, state.soil_water_m, state.discharge_m2_s, depth), strict=True):
        np.testing.assert_array_equal(before, after)
    with pytest.raises(TypeError):
        evolve(g, params, field, s, state, 10.0, StormControl(), report_every_s=10.0, rate_buffer=depth)


# --- true peak versus sampled hydrograph ------------------------------------------------------------
@pytest.mark.parametrize("impl", implementations())
def test_true_peak_outlet_discharge_is_tracked_between_report_rows(impl):
    """A 60 s pulse on a chain peaks shortly after the rain stops; with a
    200 s reporting cadence the sampled rows miss it. Two cadences with the
    same accepted partition (all dt = 1 s on integer boundaries) give bitwise
    identical states and true peaks."""
    g = make_graph(chain_full(3), ff=5.0)
    params, s0 = soil(g, 1e-6)
    s = constant_rainfall(0.0, 60.0, 72.0)
    scale = np.zeros(g.shape)
    scale[-1, 0] = 1.0  # upstream pulse: outlet peak follows the forcing edge
    field = rainfall_field(*g.shape, scale=scale)
    results = {}
    for cadence in (200.0, 10.0):
        state0 = initial_state(g, np.zeros(g.shape), s0)
        results[cadence] = evolve(g, params, field, s, state0, 400.0,
                                  StormControl(implementation=impl), report_every_s=cadence)
    coarse, fine = results[200.0], results[10.0]
    assert coarse.n_accepted_steps == fine.n_accepted_steps == 400
    for name in ("depth_m", "soil_water_m", "discharge_m2_s"):
        np.testing.assert_array_equal(getattr(coarse.state, name), getattr(fine.state, name))
    assert float(coarse.peak_outlet_discharge_m3_s) == float(fine.peak_outlet_discharge_m3_s)
    assert float(coarse.time_of_peak_outlet_s) == float(fine.time_of_peak_outlet_s)
    hyd_c, hyd_f = np.asarray(coarse.hydrograph), np.asarray(fine.hydrograph)
    true_peak, true_time = float(coarse.peak_outlet_discharge_m3_s), float(coarse.time_of_peak_outlet_s)
    assert hyd_c[:, COL["t_s"]].tolist() == [60.0, 200.0, 400.0]  # forcing edges are reported too
    assert hyd_c[:, COL["outlet_discharge_m3_s"]].max() < true_peak  # the coarse samples miss the peak
    assert hyd_f[:, COL["outlet_discharge_m3_s"]].max() <= true_peak
    assert true_time not in hyd_c[:, COL["t_s"]].tolist() and 60.0 <= true_time < 200.0
    # the fine cadence's sampled maximum matches the true peak only if it sampled the peak step
    assert (hyd_f[:, COL["outlet_discharge_m3_s"]].max() == true_peak) == (true_time in hyd_f[:, COL["t_s"]].tolist())
    # every accepted step is <= the true peak (by construction) and the rows' values are a subset of steps
    assert np.all(hyd_f[:, COL["outlet_discharge_m3_s"]] <= true_peak)


def test_peak_velocity_includes_the_initial_state_on_pure_recession():
    g = make_graph(chain_full(4), ff=5.0)
    params, s0 = soil(g, 0.0)
    h0 = np.full(g.shape, 5e-3)
    state0 = initial_state(g, h0, s0)
    r = evolve(g, params, rainfall_field(*g.shape), constant_rainfall(0.0, 30.0, 0.0), state0, 30.0, StormControl(),
               report_every_s=30.0)
    k = g.conveyance.reshape(g.shape)
    np.testing.assert_array_equal(r.peak_velocity_m_s, np.sqrt(h0) * k)  # recession: the initial velocity is the max
    np.testing.assert_array_equal(r.peak_depth_m, h0)
    q0 = g.dx_m * float(state0.discharge_m2_s[g.outlet].sum())
    assert float(r.peak_outlet_discharge_m3_s) == q0 and float(r.time_of_peak_outlet_s) == 0.0


# --- evolution -----------------------------------------------------------------------------------
@pytest.mark.parametrize("impl", implementations())
def test_chain_storm_gap_and_recession_closes_and_retains_water(impl):
    g = make_graph(chain_full(6), ff=5.0)
    area = g.dx_m ** 2
    s = schedule_from([0.0, 120.0, 180.0, 300.0], [36.0, 0.0, 36.0])
    params, s0 = soil(g, 1e-6, drain=0.05)
    state0 = initial_state(g, np.zeros(g.shape), s0)
    field = rainfall_field(*g.shape)
    r = evolve(g, params, field, s, state0, 480.0, StormControl(implementation=impl), report_every_s=30.0)
    residual, tol, rain = budget(g, state0, r, area)
    assert abs(residual) <= tol < 1e-9
    assert rain == pytest.approx(s.depth_m(0.0, 480.0) * g.n_active * area, rel=1e-13)
    assert float(r.cumulative_export_m3) > 0.0 and float(r.state.depth_m.sum()) > 0.0  # residual water retained
    assert r.state.t_s == 480.0 and r.n_accepted_steps == 480 and r.n_rejected_attempts == 0
    assert {120.0, 180.0, 300.0, 480.0}.issubset(set(r.boundaries.tolist()))
    hyd = np.asarray(r.hydrograph)
    assert hyd.shape == (r.boundaries.size, len(HYDROGRAPH_COLUMNS))
    np.testing.assert_array_equal(hyd[:, COL["t_s"]], r.boundaries)
    for name in ("cumulative_rain_m3", "cumulative_export_m3", "cumulative_drainage_m3", "accepted_steps"):
        assert np.all(np.diff(hyd[:, COL[name]]) >= 0.0)
    row_residual = (hyd[:, COL["surface_storage_m3"]] + hyd[:, COL["soil_storage_m3"]]
                    + hyd[:, COL["cumulative_drainage_m3"]] + hyd[:, COL["cumulative_export_m3"]]) - (
        float(s0.sum()) * area + hyd[:, COL["cumulative_rain_m3"]])
    assert np.all(np.abs(row_residual) <= tol)
    assert hyd[-1, COL["outlet_discharge_m3_s"]] > 0.0  # still draining at the configured end
    assert hyd[-1, COL["outlet_discharge_m3_s"]] < hyd[:, COL["outlet_discharge_m3_s"]].max()  # recession
    during_gap = (hyd[:, COL["t_s"]] > 120.0) & (hyd[:, COL["t_s"]] <= 180.0)
    assert np.all(np.diff(hyd[:, COL["cumulative_rain_m3"]])[during_gap[1:]] == 0.0)
    assert float(r.max_courant_old) < 1.0 and r.min_accepted_dt_s == r.max_accepted_dt_s == 1.0


def test_partial_window_integrates_only_the_simulated_rain():
    g = make_graph(chain_full(3), ff=5.0)
    s = constant_rainfall(0.0, 120.0, 36.0)
    params, s0 = soil(g, 1e-6)
    state0 = initial_state(g, np.zeros(g.shape), s0)
    r = evolve(g, params, rainfall_field(*g.shape), s, state0, 45.0, StormControl(), report_every_s=20.0)
    assert r.boundaries.tolist() == [20.0, 40.0, 45.0] and r.state.t_s == 45.0
    assert float(r.cumulative_rain_m.sum()) == pytest.approx(1e-5 * 45.0 * g.n_active, rel=1e-13)


def test_initial_dry_with_high_infiltration_produces_no_runoff():
    g = make_graph(valley_full(4, 5), ff=5.0)
    params, s0 = soil(g, 1e-4)  # capacity far above the rain rate
    state0 = initial_state(g, np.zeros(g.shape), s0)
    r = evolve(g, params, rainfall_field(*g.shape), constant_rainfall(0.0, 30.0, 36.0), state0, 60.0,
               StormControl(), report_every_s=30.0)
    assert not np.any(r.state.depth_m) and float(r.cumulative_export_m3) == 0.0
    np.testing.assert_allclose(r.state.soil_water_m - s0, r.cumulative_rain_m, rtol=0, atol=1e-15)
    assert not np.any(r.state.discharge_m2_s)


@pytest.mark.parametrize("impl", implementations())
def test_adaptive_retry_recomputes_from_the_same_state_without_double_counting(impl):
    """Deep water on a steep chain: dt = 8 s violates the Courant guard, the
    step is halved to 1 s and re-run from the unchanged state. Time and the
    budget are exact; rejected attempts leave no trace but their count."""
    g = make_graph(chain_full(4), ff=1.0)
    area = g.dx_m ** 2
    h0 = np.full(g.shape, 0.05)
    params, s0 = soil(g, 0.0)
    state0 = initial_state(g, h0, s0)
    before = (state0.depth_m.copy(), state0.soil_water_m.copy(), state0.discharge_m2_s.copy())
    s = constant_rainfall(0.0, 64.0, 0.0)
    control = StormControl(max_dt_s=8.0, min_dt_s=1.0 / 64.0, max_retries=10, implementation=impl)
    r = evolve(g, params, rainfall_field(*g.shape), s, state0, 64.0, control, report_every_s=16.0)
    assert r.n_rejected_attempts > 0 and r.rejections[0]["dt_tried_s"] == 8.0 and "Courant" in r.rejections[0]["reason"]
    # The first 8 s attempt is rejected and the step is reduced; as the water
    # recedes the requested 8 s becomes admissible again, which is legitimate.
    # Every accepted dt stays within [floor, requested max].
    assert r.state.t_s == 64.0 and r.min_accepted_dt_s <= 1.0 and r.max_accepted_dt_s <= 8.0
    assert r.min_accepted_dt_s >= 1.0 / 64.0
    hyd0 = np.asarray(r.hydrograph)
    assert hyd0[0, COL["min_accepted_dt_s"]] <= 1.0 and hyd0[0, COL["rejected_attempts"]] >= 3
    residual, tol, rain = budget(g, state0, r, area)
    assert rain == 0.0 and abs(residual) <= tol
    for a, b in zip(before, (state0.depth_m, state0.soil_water_m, state0.discharge_m2_s), strict=True):
        np.testing.assert_array_equal(a, b)  # the original state object was never modified
    hyd = np.asarray(r.hydrograph)
    assert hyd[-1, COL["accepted_steps"]] == r.n_accepted_steps and hyd[-1, COL["rejected_attempts"]] == r.n_rejected_attempts
    # A run that never needs retries (dt 1 s from the start) is a valid reference for the budget only.
    ref = evolve(g, params, rainfall_field(*g.shape), s, initial_state(g, h0, s0), 64.0,
                 StormControl(max_dt_s=1.0, implementation=impl), report_every_s=16.0)
    assert ref.n_rejected_attempts == 0
    assert float(ref.cumulative_export_m3) == pytest.approx(float(r.cumulative_export_m3), rel=5e-2)


def test_retry_budget_and_minimum_step_refuse_without_mutation():
    g = make_graph(chain_full(4), ff=1.0)
    h0 = np.full(g.shape, 0.05)
    params, s0 = soil(g, 0.0)
    state0 = initial_state(g, h0, s0)
    copies = (state0.depth_m.copy(), state0.discharge_m2_s.copy())
    s = constant_rainfall(0.0, 64.0, 0.0)
    field = rainfall_field(*g.shape)
    with pytest.raises(StormError, match="max_retries"):
        evolve(g, params, field, s, state0, 64.0, StormControl(max_dt_s=8.0, min_dt_s=1e-3, max_retries=1),
               report_every_s=16.0)
    with pytest.raises(StormError, match="min_dt_s"):
        evolve(g, params, field, s, state0, 64.0, StormControl(max_dt_s=8.0, min_dt_s=8.0, max_retries=10),
               report_every_s=16.0)
    np.testing.assert_array_equal(state0.depth_m, copies[0])
    np.testing.assert_array_equal(state0.discharge_m2_s, copies[1])


def test_max_steps_and_non_advancing_time_are_refused():
    g = make_graph(chain_full(3), ff=5.0)
    params, s0 = soil(g, 1e-6)
    field = rainfall_field(*g.shape)
    state0 = initial_state(g, np.zeros(g.shape), s0)
    with pytest.raises(StormError, match="max_steps"):
        evolve(g, params, field, constant_rainfall(0.0, 10.0, 36.0), state0, 10.0, StormControl(max_steps=2),
               report_every_s=10.0)
    late = initial_state(g, np.zeros(g.shape), s0, t_s=float(2 ** 53))
    with pytest.raises(StormError, match="does not advance"):
        evolve(g, params, field, constant_rainfall(0.0, 1.0, 0.0), late, float(2 ** 53) + 2.0,
               StormControl(max_dt_s=1.0), report_every_s=1.0)


def test_fatal_errors_propagate_unchanged_and_unretried(monkeypatch):
    g = make_graph(chain_full(3), ff=5.0)
    params, s0 = soil(g, 1e-6)
    field = rainfall_field(*g.shape)
    state0 = initial_state(g, np.zeros(g.shape), s0)
    s = constant_rainfall(0.0, 10.0, 36.0)
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        raise RoutingError("bisection did not reach root_tolerance_m (test)")

    monkeypatch.setattr(storm_module, "route_step", failing)
    with pytest.raises(RoutingError, match="bisection") as info:
        evolve(g, params, field, s, state0, 10.0, StormControl(), report_every_s=10.0)
    assert not isinstance(info.value, (StormError, RoutingStepRejected)) and len(calls) == 1  # no retry

    def broken(*args, **kwargs):
        raise TypeError("programming error (test)")

    monkeypatch.setattr(storm_module, "route_step", broken)
    with pytest.raises(TypeError, match="programming error"):
        evolve(g, params, field, s, state0, 10.0, StormControl(), report_every_s=10.0)


def test_evolve_rejects_mismatched_inputs():
    g = make_graph(chain_full(3), ff=5.0)
    params, _ = soil(make_graph(chain_full(4), ff=5.0), 1e-6)
    with pytest.raises(StormError, match="column parameters"):
        evolve(g, params, rainfall_field(*g.shape), constant_rainfall(0.0, 1.0, 0.0),
               initial_state(g, np.zeros(g.shape), np.zeros(g.shape)), 1.0, StormControl(), report_every_s=1.0)
    with pytest.raises(StormError, match="depth_m"):
        initial_state(g, np.zeros((4, 1)), np.zeros(g.shape))


@NUMBA_SKIP
def test_array_and_numba_evolutions_are_identical():
    g = make_graph(valley_full(6, 5), ff=5.0)
    s = schedule_from([0.0, 60.0, 90.0, 150.0], [36.0, 0.0, 18.0])
    params, s0 = soil(g, 1e-6, drain=0.05)
    results = {}
    for impl in ("array", "numba"):
        state0 = initial_state(g, np.zeros(g.shape), s0)
        results[impl] = evolve(g, params, rainfall_field(*g.shape), s, state0, 200.0,
                               StormControl(implementation=impl), report_every_s=25.0)
    a, b = results["array"], results["numba"]
    for name in ("depth_m", "soil_water_m", "discharge_m2_s"):
        np.testing.assert_array_equal(getattr(a.state, name), getattr(b.state, name), err_msg=name)
    np.testing.assert_array_equal(np.asarray(a.hydrograph), np.asarray(b.hydrograph))
    assert float(a.cumulative_export_m3) == float(b.cumulative_export_m3) > 0.0
    assert (a.n_accepted_steps, a.n_rejected_attempts) == (b.n_accepted_steps, b.n_rejected_attempts)


def test_distinct_representable_forcing_edges_are_never_merged():
    """A narrow, finite pulse must not disappear when report times are merged."""
    g = make_graph(chain_full(3), ff=5.0)
    params, s0 = soil(g, 0.0)
    next_time = np.nextafter(1.0, 2.0)
    schedule = schedule_from([0.0, 1.0, next_time, 2.0], [0.0, 3.6e15, 0.0])
    np.testing.assert_array_equal(plan_boundaries(schedule, 0.0, 2.0, 2.0), [1.0, next_time, 2.0])
    result = evolve(g, params, rainfall_field(*g.shape), schedule,
                    initial_state(g, np.zeros(g.shape), s0), 2.0,
                    StormControl(max_dt_s=2.0), report_every_s=2.0)
    assert float(result.cumulative_rain_m.sum()) == pytest.approx(schedule.total_depth_m() * g.n_active, rel=1e-14)
    assert result.n_rejected_attempts == 0


def test_initial_state_refuses_cross_backend_inputs_before_conversion():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    g = make_graph(chain_full(3), ff=5.0, xp=cp)
    h = np.zeros(g.shape)
    s = np.full(g.shape, 0.075)
    with pytest.raises(StormError, match="namespace"):
        initial_state(g, h, s)
    np.testing.assert_array_equal(h, 0.0)
    np.testing.assert_array_equal(s, 0.075)
