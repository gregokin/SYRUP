"""EXPERIMENTAL explicit conservative kinematic wave: CPU reference tests (no GPU, no Numba).

References are independent of the implementation: a closed-form per-cell loop over the receivers (not the donor-slot
gather), the CFL/positivity bound derived in `experimental_hydrology`, the water-budget identity with the MAPLE-derived bound,
and the accepted `infiltration.column_step` for the column. Nothing here was run by its author (file-only tools); Codex
records results.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from cand_cases import DX, build, field, host, rain_rate, schedule

pytest.importorskip("maple")

from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.experimental_hydrology import (
    CpuHydraulicSolver,
    ExperimentalHydrologyError,
    HydraulicControl,
    HydraulicState,
    HydraulicStepRejected,
)
from maple_syrup.experimental_storm import ExperimentalControl, evolve_experimental
from maple_syrup.infiltration import InfiltrationError, column_step

AREA = DX * DX


def budget_residual(result, state0):
    def vol(a):
        return float(np.sum(host(a))) * AREA

    initial = vol(state0.depth_m) + vol(state0.soil_water_m)
    rain, drain = vol(result.cumulative_rain_m), vol(result.cumulative_drainage_m)
    export = float(host(result.cumulative_export_m3))
    final = vol(result.state.depth_m) + vol(result.state.soil_water_m)
    n_cells = host(result.state.depth_m).size
    bound = volume_roundoff_bound_m3(4 * n_cells * max(result.n_accepted_steps, 1) + 7,
                                     max(initial, rain, drain, export, final))
    return final + drain + export - initial - rain, bound, (initial, rain, drain, export, final)


# --- analytic: one explicit step -------------------------------------------------------------------------------------
def test_chain_step_equals_the_closed_form_computed_from_the_receivers():
    cs = build("chain:3", "explicit", seed=1)
    graph, dt = cs.graph, 0.5
    h = cs.state.depth_m[:, 0].copy()
    k = np.asarray(graph.conveyance).reshape(cs.shape)[:, 0]
    step = cs.solver.step(np.zeros(cs.shape), cs.state, dt)
    q = k * h ** 1.5  # q = k h^(3/2)
    out = dt * DX * q  # F = dt dx q, the volume leaving each cell, used once
    expected = h.copy()
    for r in range(3):  # cell r drains into cell r-1 (south); cell 0 exports; donor of r is r+1
        inflow = out[r + 1] if r + 1 < 3 else 0.0
        expected[r] = h[r] + (inflow - out[r]) / AREA
    np.testing.assert_allclose(step.state.depth_m[:, 0], expected, rtol=1e-14, atol=1e-18)
    np.testing.assert_allclose(step.face_volume_m3["out"][:, 0], out, rtol=1e-14, atol=1e-18)
    np.testing.assert_allclose(step.used_flux_m2_s["out"][:, 0], q, rtol=1e-14, atol=1e-18)
    assert step.export_m3 == pytest.approx(out[0], rel=1e-14)  # only the outlet cell exports
    np.testing.assert_allclose(step.velocity_m_s[:, 0], k * np.sqrt(expected), rtol=1e-12, atol=1e-18)  # q(h')/h' = k sqrt(h')
    assert step.method == "explicit" and step.implementation == "numpy" and step.state.qx_m2_s is None
    assert step.state.t_s == dt and step.max_cfl == pytest.approx(float(np.max(1.5 * k * np.sqrt(h) * dt / DX)), rel=1e-14)


def test_the_cfl_bound_is_the_characteristic_speed_and_marks_the_admissibility_boundary():
    cs = build("random:9x6", "explicit", seed=2)
    k = np.asarray(cs.graph.conveyance).reshape(cs.shape)
    h = cs.state.depth_m
    dt_limit = 0.5 * DX / float(np.max(1.5 * k * np.sqrt(h)))  # 1.5 k sqrt(h) dt/dx = cfl_max
    cs.solver.step(np.zeros(cs.shape), cs.state, dt_limit * (1.0 - 1e-9))  # admissible: accepted
    with pytest.raises(HydraulicStepRejected, match=r"characteristic CFL .* exceeds cfl_max 0\.5") as info:
        cs.solver.step(np.zeros(cs.shape), cs.state, dt_limit * (1.0 + 1e-6))
    assert type(info.value) is HydraulicStepRejected  # the recoverable class, deliberately not the base class


def test_positivity_proof_holds_at_the_limit_for_many_random_states():
    """h_new >= h_c (1 - cfl_max/1.5) whatever the inflow: checked on random wet/dry mixtures at the largest admissible dt."""
    for seed in range(6):
        cs = build("random:10x7", "explicit", seed=seed)
        rng = np.random.default_rng(100 + seed)
        depth = np.where(cs.graph.active & (rng.random(cs.shape) > 0.2), rng.uniform(1e-6, 2e-2, cs.shape), 0.0)
        state = cs.solver.initial_state(depth, cs.state.soil_water_m)
        k = np.asarray(cs.graph.conveyance).reshape(cs.shape)
        dt = 0.999999 * 0.5 * DX / float(np.max(1.5 * k * np.sqrt(depth)))
        step = cs.solver.step(np.zeros(cs.shape), state, dt)
        assert float(np.min(step.state.depth_m)) >= 0.0
        assert np.all(step.state.depth_m >= depth * (1.0 - 0.5 / 1.5) - 1e-18)


def test_dry_cells_move_nothing_and_have_zero_velocity():
    cs = build("valley:6x5", "explicit", depth="dry")
    step = cs.solver.step(np.zeros(cs.shape), cs.state, 1.0)
    assert not np.any(step.state.depth_m) and not np.any(step.velocity_m_s) and not np.any(step.face_volume_m3["out"])
    assert step.export_m3 == 0.0 and step.max_cfl == 0.0 and step.outlet_discharge_m3_s == 0.0


def test_interior_face_volumes_cancel_and_the_export_is_the_outlet_volume():
    cs = build("valley:8x7", "explicit", seed=3)
    step = cs.solver.step(np.zeros(cs.shape), cs.state, 0.5)
    out = step.face_volume_m3["out"]
    outlet = np.asarray(cs.graph.outlet)
    assert step.export_m3 == pytest.approx(float(out[outlet].sum()), rel=1e-14)
    gained = (step.state.depth_m - cs.state.depth_m) * AREA
    assert float(gained.sum()) + step.export_m3 == pytest.approx(0.0, abs=1e-15)  # every volume moved was used once
    receiver = np.asarray(cs.graph.receiver)
    expected_gain = -out.copy()
    for cell in np.flatnonzero(np.asarray(cs.graph.active).reshape(-1) & ~outlet.reshape(-1)):
        expected_gain.reshape(-1)[int(receiver.reshape(-1)[cell])] += out.reshape(-1)[cell]  # received exactly once
    np.testing.assert_allclose(gained, expected_gain, rtol=1e-12, atol=1e-18)


def test_inactive_cells_retain_their_inventories_and_exchange_nothing():
    cs = build("valley_masked", "explicit", seed=4, soil_fraction=0.4)
    inactive = ~cs.graph.active
    depth = np.where(inactive, 0.123, cs.state.depth_m)
    state = cs.solver.initial_state(depth, cs.state.soil_water_m)
    step = cs.solver.step(np.zeros(cs.shape), state, 0.5)
    np.testing.assert_array_equal(step.state.depth_m[inactive], depth[inactive])
    np.testing.assert_array_equal(step.state.soil_water_m[inactive], state.soil_water_m[inactive])
    assert not np.any(step.face_volume_m3["out"][inactive]) and not np.any(step.velocity_m_s[inactive])


# --- the column is the accepted column physics ------------------------------------------------------------------------
def test_the_column_stage_is_the_accepted_column_step_and_no_infiltration_formula_is_repeated():
    cs = build("random:8x6", "explicit", seed=5, ksat=2e-6, soil_fraction=0.4, model="pavement_hawkins")
    rain = rain_rate(cs, 80.0)
    step = cs.solver.step(rain, cs.state, 0.5)
    ref = column_step(cs.params, cs.state.depth_m, cs.state.soil_water_m, rain, 0.5)
    for name in ("depth_m", "soil_water_m", "rain_m", "intake_m", "saturation_return_m", "drainage_m"):
        np.testing.assert_array_equal(getattr(step.column, name), getattr(ref, name), err_msg=name)
    np.testing.assert_array_equal(step.state.soil_water_m, ref.soil_water_m)
    import inspect

    from maple_syrup import experimental_hydrology

    source = inspect.getsource(experimental_hydrology)
    for banned in ("expm1", "Smith", "capacity", "overflow ="):
        assert banned not in source.split('"""', 2)[2], banned  # no copy of the infiltration formulas in the code


def test_saturation_return_and_drainage_conserve_water_through_a_storm():
    # start AT saturation: with no deficit the capacity is Ksat, intake (Ksat dt) exceeds drainage (0.5 Ksat dt), so the
    # surplus MUST return to the surface; 0.98 left a deficit the short storm could not fill
    cs = build("valley:8x7", "explicit", seed=6, ksat=3e-6, soil_fraction=1.0, drainage=0.5, model="fixed_ksat")
    result = evolve_experimental(cs.solver, field(cs, scale=1.0), schedule([0, 30, 60, 90], [150.0, 0.0, 150.0]), cs.state,
                                 120.0, ExperimentalControl(), report_every_s=30.0)
    residual, bound, (_initial, rain, drain, export, _final) = budget_residual(result, cs.state)
    assert abs(residual) <= bound, (residual, bound)
    assert float(np.sum(result.cumulative_saturation_return_m)) > 0.0 and drain > 0.0 and export > 0.0
    assert rain > 0.0  # the exact rainfall volume is asserted by the driver tests


# --- refusals, purity, precedence -------------------------------------------------------------------------------------
def test_state_and_option_refusals_and_a_failed_step_modifies_nothing():
    cs = build("valley:6x5", "explicit", seed=7)
    st = cs.state
    before = [a.copy() for a in (st.depth_m, st.soil_water_m)]
    bad_states = {
        "momentum_given": HydraulicState(0.0, st.depth_m, st.soil_water_m, np.zeros((6, 6)), np.zeros((7, 5))),
        "float32": HydraulicState(0.0, st.depth_m.astype(np.float32), st.soil_water_m),
        "wrong_shape": HydraulicState(0.0, np.zeros((2, 3)), st.soil_water_m),
        "negative": HydraulicState(0.0, np.where(cs.graph.active, -1e-3, 0.0), st.soil_water_m),
        "list": HydraulicState(0.0, st.depth_m.tolist(), st.soil_water_m),
    }
    for bad in bad_states.values():
        with pytest.raises((ExperimentalHydrologyError, InfiltrationError)):
            cs.solver.step(np.zeros(cs.shape), bad, 0.5)
        with pytest.raises(ExperimentalHydrologyError):
            cs.solver.validate_state(bad)
    for dt in (0.0, -1.0, float("nan"), float("inf"), True, "1"):
        with pytest.raises((ExperimentalHydrologyError, InfiltrationError)):
            cs.solver.step(np.zeros(cs.shape), st, dt)
    with pytest.raises(HydraulicStepRejected):
        cs.solver.step(np.zeros(cs.shape), st, 1e6)
    for a, b in zip(before, (st.depth_m, st.soil_water_m), strict=True):
        np.testing.assert_array_equal(a, b)


def test_masked_and_subclassed_arrays_cannot_launder_values():
    cs = build("valley:6x5", "explicit", seed=7)
    st = cs.state
    masked = np.ma.masked_array(st.depth_m, mask=np.ones(cs.shape, dtype=bool))

    class Subclass(np.ndarray):
        pass

    for odd in (masked, st.depth_m.view(Subclass)):
        with pytest.raises(ExperimentalHydrologyError, match="exact"):
            cs.solver.validate_state(HydraulicState(0.0, odd, st.soil_water_m))
        with pytest.raises(ExperimentalHydrologyError):
            cs.solver.initial_state(odd, st.soil_water_m)


def test_a_column_failure_outranks_a_cfl_rejection():
    cs = build("valley:6x5", "explicit", seed=8)
    soil = cs.state.soil_water_m.copy()
    soil[2, 2] = np.nan
    with pytest.raises(InfiltrationError):
        cs.solver.step(np.zeros(cs.shape), HydraulicState(0.0, cs.state.depth_m, soil), 1e6)  # huge dt AND bad soil


def test_outputs_are_fresh_and_a_later_step_never_changes_an_earlier_result():
    cs = build("random:8x6", "explicit", seed=9)
    first = cs.solver.step(np.zeros(cs.shape), cs.state, 0.5)
    snapshot = {k: v.copy() for k, v in (("depth", first.state.depth_m), ("soil", first.state.soil_water_m),
                                         ("vel", first.velocity_m_s), ("out", first.face_volume_m3["out"]))}
    cs.solver.step(np.zeros(cs.shape), first.state, 0.5)
    np.testing.assert_array_equal(first.state.depth_m, snapshot["depth"])
    np.testing.assert_array_equal(first.velocity_m_s, snapshot["vel"])
    np.testing.assert_array_equal(first.face_volume_m3["out"], snapshot["out"])
    arrays = [first.state.depth_m, first.state.soil_water_m, first.velocity_m_s, first.face_volume_m3["out"],
              first.used_flux_m2_s["out"], cs.state.depth_m, cs.state.soil_water_m]
    for i, a in enumerate(arrays):
        for b in arrays[i + 1:]:
            assert not np.shares_memory(a, b)


def test_solver_construction_refusals():
    cs = build("valley:6x5", "explicit", seed=10)
    with pytest.raises(ExperimentalHydrologyError, match="limiter"):
        CpuHydraulicSolver("explicit", cs.graph, cs.params, control=HydraulicControl(limiter="donor"))
    for bad in (0.0, -0.1, 0.51, float("nan"), True, "0.5"):
        with pytest.raises(ExperimentalHydrologyError):
            HydraulicControl(cfl_max=bad).validated()
    with pytest.raises(ExperimentalHydrologyError):
        HydraulicControl(limiter="clip").validated()
    with pytest.raises(ExperimentalHydrologyError, match="method"):
        CpuHydraulicSolver("diffusive", cs.graph, cs.params)
    assert dataclasses.is_dataclass(HydraulicState)
    assert cs.solver.step(rain_rate(cs, 0.0), cs.state, 0.5).state.t_s == 0.5
