"""Pond hydraulics tests (NumPy reference; CUDA/Numba tests skip only when the backend is unavailable). Nothing here was run by
its author (file-only tools); Codex records results. Run explicitly (`pytest tests/pond`).

Conventions: every time step and forcing in a comparison is stated in the test (dyadic dt where bitwise continuation is
asserted); beds and depths are dyadic where exactness is asserted; every volume tolerance is
`conservation.volume_roundoff_bound_m3` via the ledger, never an absolute mass floor.
"""
from __future__ import annotations

import numpy as np
import pytest
from test_network import DX, topo_from

from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.experimental_hydrology import HydraulicControl, HydraulicStepRejected, stage_face_diagnostics
from maple_syrup.experimental_storm import ExperimentalControl
from maple_syrup.infiltration import column_parameters
from maple_syrup.pond_hydraulics import (
    SLOPE_NOTE,
    PondHydraulicsError,
    PondLedger,
    build_pond_hydraulics,
    evolve_pond,
    face_speed_diagnostics,
    metadata_routing_graph,
    rebind_pond,
    step_once,
    uniform_rain_rate,
)
from maple_syrup.pond_hydraulics import _to_host as H
from maple_syrup.pond_network import update_pond_topology
from maple_syrup.routing import GRAVITY_M_S2

AREA = DX * DX
FINE = ExperimentalControl(max_dt_s=1.0 / 128.0, min_dt_s=1.0 / 4096.0)

BED = np.array([[0.5, 0.25, 1.5, 0.75, 1.0],
                [0.0, 0.125, 1.25, 0.5, 1.125],
                [0.75, 0.375, 0.625, 1.375, 0.25]])


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip (a skipped test makes no GPU claim)."""
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA pond solver is not exercised")
    return backend.cupy_module()


def col_params(active, xp=np, *, ksat=0.0, drainage=0.0):
    shape = active.shape

    def full(v):
        return xp.asarray(np.full(shape, float(v)))

    return column_parameters(model="fixed_ksat", ksat_m_per_s=full(ksat), suction_m=full(0.01), drainage_parameter=full(drainage),
                             theta_sat=full(0.4), soil_thickness_m=full(0.3), active_mask=xp.asarray(active))


def make_pond(topology, backend="numpy", *, ksat=0.0, drainage=0.0, control=None, graph=None):
    if backend == "cuda":
        from maple.core.backend import cupy_module

        xp = cupy_module()
    else:
        xp = np
    return build_pond_hydraulics(topology, col_params(np.asarray(topology.active), xp, ksat=ksat, drainage=drainage),
                                 backend=backend, control=control, graph=graph)


def state_of(pond, depth, soil_fraction=0.0):
    xp = pond.xp
    return pond.initial_state(xp.asarray(np.array(depth, dtype=np.float64)), pond.params.storage_max_m * soil_fraction)


def flat_drain_case():
    topology = topo_from(np.full((3, 4), 1.0), [(1, 3, 0, 1)], drop=0.125)
    depth = np.full((3, 4), 0.0625)
    depth[0, 0] += 0.0625
    return topology, depth


# --- rest states ----------------------------------------------------------------------------------------------------------
def test_lake_at_rest_on_uneven_bed_is_exact():
    level = 1.0
    depth = np.maximum(level - BED, 0.0)
    pond = make_pond(topo_from(BED))
    state = state_of(pond, depth)
    result = evolve_pond(pond, uniform_rain_rate(pond, 0.0), state, 0.125, FINE)
    assert result.reached_target and result.n_accepted_steps == 16
    assert np.array_equal(result.state.depth_m, depth)
    assert not np.any(result.state.qx_m2_s) and not np.any(result.state.qy_m2_s)
    closure = result.closure()
    assert closure.closes and closure.export_m3 == 0.0
    assert all(not np.any(v) for v in (*closure.face_pos_m3.values(), *closure.face_neg_m3.values()))


def test_flat_closed_lake_rests_and_a_bump_spreads_conservatively():
    pond = make_pond(topo_from(np.zeros((3, 4))))
    depth = np.full((3, 4), 0.125)
    rest = evolve_pond(pond, uniform_rain_rate(pond, 0.0), state_of(pond, depth), 0.125, FINE)
    assert np.array_equal(rest.state.depth_m, depth) and not np.any(rest.state.qx_m2_s)
    bumped = depth.copy()
    bumped[1, 1] += 0.0625
    result = evolve_pond(pond, uniform_rain_rate(pond, 0.0), state_of(pond, bumped), 0.25, FINE)
    out = H(result.state.depth_m)
    closure = result.closure()
    assert result.reached_target and closure.closes and closure.export_m3 == 0.0
    assert out.min() >= 0.0 and out.max() < bumped.max()
    assert np.any(closure.face_pos_m3["x"] > 0.0) or np.any(closure.face_pos_m3["y"] > 0.0)


def test_flat_interior_drains_through_an_explicit_downstep_outlet():
    topology, depth = flat_drain_case()
    pond = make_pond(topology)
    assert not np.asarray(pond.graph.outlet).any()  # the legacy graph has NO outlet: the explicit face is the authority
    result = evolve_pond(pond, uniform_rain_rate(pond, 0.0), state_of(pond, depth), 1.0,
                         ExperimentalControl(max_dt_s=1.0 / 32.0, min_dt_s=1.0 / 4096.0))
    closure = result.closure()
    out = H(result.state.depth_m)
    assert result.reached_target and closure.closes
    assert closure.export_m3 > 0.0 and out.min() >= 0.0 and out.sum() * AREA < depth.sum() * AREA
    assert abs(closure.export_m3 + closure.surface_change_m3) <= closure.global_bound_m3
    closed = make_pond(topo_from(np.full((3, 4), 1.0)))
    shut = evolve_pond(closed, uniform_rain_rate(closed, 0.0), state_of(closed, depth), 0.25, FINE)
    assert shut.closure().export_m3 == 0.0


# --- sills ------------------------------------------------------------------------------------------------------------------
LIP = np.array([[0.0, 1.0, 0.0]])


def test_below_lip_nothing_crosses():
    pond = make_pond(topo_from(LIP))
    depth = np.array([[0.5, 0.0, 0.0]])
    result = evolve_pond(pond, uniform_rain_rate(pond, 0.0), state_of(pond, depth), 0.5,
                         ExperimentalControl(max_dt_s=1.0 / 64.0, min_dt_s=1.0 / 4096.0))
    closure = result.closure()
    assert np.array_equal(result.state.depth_m, depth) and closure.closes
    assert all(not np.any(v) for v in (*closure.face_pos_m3.values(), *closure.face_neg_m3.values(), *closure.face_net_m3.values()))


def test_above_lip_water_crosses_and_is_conserved():
    pond = make_pond(topo_from(LIP))
    depth = np.array([[1.5, 0.0, 0.0]])
    result = evolve_pond(pond, uniform_rain_rate(pond, 0.0), state_of(pond, depth), 0.5, FINE)
    closure = result.closure()
    out = H(result.state.depth_m)
    assert result.reached_target and closure.closes
    assert closure.face_net_m3["x"][0, 1] > 0.0 and closure.face_net_m3["x"][0, 2] > 0.0 and out[0, 2] > 0.0
    assert closure.face_pos_m3["x"][0, 1] > 0.0
    total = out.sum() * AREA
    assert abs(total - depth.sum() * AREA) <= closure.global_bound_m3


def test_connected_multicell_basins_exchange_water_conservatively():
    bed = np.array([[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0]])
    pond = make_pond(topo_from(bed))
    depth = np.array([[0.0, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0]])
    result = evolve_pond(pond, uniform_rain_rate(pond, 0.0), state_of(pond, depth), 1.0, FINE)
    out = H(result.state.depth_m)
    closure = result.closure()
    assert result.reached_target and closure.closes
    assert out[0, 4] > 0.0 and out[0, 0] == 0.0 and out[0, 6] == 0.0  # water crossed the sill; the 2 m walls stayed dry
    assert closure.face_net_m3["x"][0, 3] > 0.0 and closure.face_net_m3["x"][0, 4] > 0.0
    assert abs(out.sum() * AREA - depth.sum() * AREA) <= closure.global_bound_m3


def test_empty_pit_creates_no_water():
    pit = np.array([[1.0, 0.5, 0.25], [0.75, 0.0, 0.5], [1.0, 0.75, 1.0]])
    for outlets in ([], [(0, 0, 0, -1)]):
        pond = make_pond(topo_from(pit, outlets))
        result = evolve_pond(pond, uniform_rain_rate(pond, 0.0), state_of(pond, np.zeros((3, 3))), 0.125, FINE)
        closure = result.closure()
        assert not np.any(result.state.depth_m) and not np.any(result.state.qx_m2_s) and not np.any(result.state.qy_m2_s)
        assert closure.export_m3 == 0.0 and closure.rain_m3 == 0.0 and closure.closes
        assert all(not np.any(v) for v in closure.face_net_m3.values())


# --- ledger -----------------------------------------------------------------------------------------------------------------
def rain_case():
    topology, _ = flat_drain_case()
    pond = make_pond(topology, ksat=1e-6, drainage=1.0)
    return pond, np.full((3, 4), 0.0625)


def test_rain_infiltration_drainage_and_export_ledger_is_exact():
    pond, depth = rain_case()
    rate = 2e-5
    result = evolve_pond(pond, uniform_rain_rate(pond, rate), state_of(pond, depth, 0.1), 1.0,
                         ExperimentalControl(max_dt_s=1.0 / 32.0, min_dt_s=1.0 / 4096.0))
    c = result.closure()
    expected_rain = rate * c.elapsed_s * AREA * 12
    assert result.reached_target and c.elapsed_s == 1.0 and c.closes
    assert c.intake_m3 > 0.0 and c.drainage_m3 > 0.0 and c.export_m3 > 0.0
    assert abs(c.rain_m3 - expected_rain) <= volume_roundoff_bound_m3(4 * 12 * c.n_steps + 7, expected_rain)
    assert c.max_cell_residual_m3 <= c.cell_bound_m3 and abs(c.global_residual_m3) <= c.global_bound_m3
    assert c.inactive_unchanged and np.all(H(result.state.depth_m) >= 0.0)


def test_timestep_sensitivity_is_stated_not_hidden():
    pond, depth = rain_case()
    rate_field = uniform_rain_rate(pond, 2e-5)  # the SAME forcing for both runs
    runs = [evolve_pond(pond, rate_field, state_of(pond, depth, 0.1), 1.0,
                        ExperimentalControl(max_dt_s=dt, min_dt_s=1.0 / 4096.0)) for dt in (1.0 / 16.0, 1.0 / 64.0)]
    coarse, fine = runs
    for run in runs:
        assert run.reached_target and run.closure().closes and run.closure().elapsed_s == 1.0
    assert fine.n_accepted_steps > coarse.n_accepted_steps and fine.min_accepted_dt_s < coarse.min_accepted_dt_s
    bound = volume_roundoff_bound_m3(4 * 12 * fine.n_accepted_steps + 7, coarse.closure().rain_m3)
    assert abs(coarse.closure().rain_m3 - fine.closure().rain_m3) <= bound  # rainfall integration is dt-independent


def test_face_volumes_accumulate_consistently():
    topology, depth = flat_drain_case()
    pond = make_pond(topology)
    state = state_of(pond, depth)
    ledger = PondLedger(pond, state)
    manual = {a: 0.0 for a in "xy"}
    gross = {a: 0.0 for a in "xy"}
    rate = uniform_rain_rate(pond, 0.0)
    for _ in range(6):
        accepted = step_once(pond, rate, state, 1.0 / 128.0)
        ledger.add(accepted)
        state = accepted.state
        for a in "xy":
            v = H(accepted.step.face_volume_m3[a])
            manual[a] = manual[a] + v
            gross[a] = gross[a] + np.abs(v)
    closure = ledger.closure(state)
    for a in "xy":
        assert np.array_equal(closure.face_net_m3[a], manual[a])
        np.testing.assert_allclose(closure.face_pos_m3[a] + closure.face_neg_m3[a], gross[a], rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(closure.face_pos_m3[a] - closure.face_neg_m3[a], manual[a], rtol=1e-12, atol=1e-18)
    assert closure.closes and closure.n_steps == 6


# --- purity, rejection, bounds ----------------------------------------------------------------------------------------------
def snapshot(state):
    return tuple(np.array(H(a)) for a in (state.depth_m, state.soil_water_m, state.qx_m2_s, state.qy_m2_s)) + (state.t_s, state.next_dt_cap_s)


def same(before, state):
    after = snapshot(state)
    return all(np.array_equal(a, b) for a, b in zip(before[:4], after[:4], strict=True)) and before[4:] == after[4:]


def test_rejected_step_does_not_mutate_anything():
    pond = make_pond(topo_from(np.zeros((3, 3))))
    state = state_of(pond, np.ones((3, 3)))
    rate = uniform_rain_rate(pond, 0.0)
    before, rate_before = snapshot(state), np.array(H(rate))
    with pytest.raises(HydraulicStepRejected):
        step_once(pond, rate, state, 10.0)
    assert same(before, state) and np.array_equal(H(rate), rate_before)
    accepted = step_once(pond, rate, state, 1.0 / 128.0)  # the same state is still usable
    assert same(before, state) and accepted.state is not state


def test_step_keeps_qx_qy_and_carries_dt_cap_unchanged():
    topology, depth = flat_drain_case()
    pond = make_pond(topology)
    state = state_of(pond, depth)
    state = type(state)(state.t_s, state.depth_m, state.soil_water_m, state.qx_m2_s, state.qy_m2_s, 0.015625)
    accepted = step_once(pond, uniform_rain_rate(pond, 0.0), state, 1.0 / 128.0)
    assert accepted.state.next_dt_cap_s == 0.015625 and accepted.state.t_s == 1.0 / 128.0
    assert accepted.state.qx_m2_s.shape == (3, 5) and accepted.state.qy_m2_s.shape == (4, 4)
    assert accepted.step.state.next_dt_cap_s is None  # the solver itself never sets the cap


def test_evolve_stops_at_bounds_and_reports_the_actual_elapsed_state():
    pond = make_pond(topo_from(np.zeros((3, 3))))
    start = state_of(pond, np.ones((3, 3)))
    rate = uniform_rain_rate(pond, 0.0)
    floor = evolve_pond(pond, rate, start, 10.0, ExperimentalControl(max_dt_s=10.0, min_dt_s=9.0, max_retries=3))
    assert floor.stop_reason == "min_dt" and not floor.reached_target and floor.elapsed_s == 0.0
    assert floor.state is start and floor.n_accepted_steps == 0 and floor.n_rejected_attempts == 1 and floor.closure().closes
    retries = evolve_pond(pond, rate, start, 10.0, ExperimentalControl(max_dt_s=10.0, min_dt_s=1.0 / 1024.0, max_retries=1))
    assert retries.stop_reason == "max_retries" and retries.n_rejected_attempts == 2 and retries.state is start
    assert len(retries.rejections) == 2 and retries.rejections[0]["dt_tried_s"] == 10.0
    bumped = np.full((3, 3), 0.125)
    bumped[1, 1] += 0.0625
    limited = evolve_pond(pond, rate, state_of(pond, bumped), 1.0,
                          ExperimentalControl(max_dt_s=1.0 / 64.0, min_dt_s=1.0 / 4096.0, max_steps=3))
    assert limited.stop_reason == "max_steps" and limited.n_accepted_steps == 3
    assert limited.elapsed_s == 3.0 / 64.0 and limited.state.t_s == 3.0 / 64.0 and limited.closure().closes
    done = evolve_pond(pond, rate, state_of(pond, bumped), 0.1, FINE)
    assert done.reached_target and done.stop_reason is None and done.state.t_s == 0.1 and done.elapsed_s == 0.1


def test_continuation_is_the_same_step_sequence_with_carried_q_and_cap():
    pond = make_pond(topo_from(np.zeros((3, 3))))
    depth = np.full((3, 3), 0.125)
    depth[1, 1] += 0.0625
    rate = uniform_rain_rate(pond, 0.0)
    whole = evolve_pond(pond, rate, state_of(pond, depth), 0.125, FINE)
    first = evolve_pond(pond, rate, state_of(pond, depth), 0.0625, FINE)
    assert first.state.next_dt_cap_s is not None and np.any(first.state.qx_m2_s != 0.0)
    second = evolve_pond(pond, rate, first.state, 0.0625, FINE)
    assert whole.n_rejected_attempts == first.n_rejected_attempts == second.n_rejected_attempts == 0
    assert first.n_accepted_steps + second.n_accepted_steps == whole.n_accepted_steps == 16
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        assert np.array_equal(H(getattr(second.state, name)), H(getattr(whole.state, name)))
    assert second.state.t_s == whole.state.t_s and second.state.next_dt_cap_s == whole.state.next_dt_cap_s


def test_cfl_validation_and_limiter_refusal_are_kept():
    topology = topo_from(np.zeros((2, 2)))
    for bad in (HydraulicControl(cfl_max=0.6), HydraulicControl(cfl_max=0.0)):
        with pytest.raises(PondHydraulicsError):
            make_pond(topology, control=bad)
    with pytest.raises(PondHydraulicsError, match="limiter"):
        make_pond(topology, control=HydraulicControl(limiter="donor"))
    assert make_pond(topology, control=HydraulicControl(cfl_max=0.25)).control.cfl_max == 0.25


def test_backend_parameter_and_graph_refusals():
    topology = topo_from(np.zeros((2, 2)))
    params = col_params(np.ones((2, 2), bool))
    with pytest.raises(PondHydraulicsError, match="backend"):
        build_pond_hydraulics(topology, params, backend="fortran")
    with pytest.raises(PondHydraulicsError, match="active mask"):
        build_pond_hydraulics(topology, col_params(np.array([[True, False], [True, True]])))
    with pytest.raises(PondHydraulicsError):
        build_pond_hydraulics("topology", params)
    with pytest.raises(PondHydraulicsError, match="friction"):
        make_pond(topo_from(np.zeros((2, 2)), friction=0.05))
    other = topo_from(np.zeros((2, 2)), friction=2.0)
    with pytest.raises(PondHydraulicsError, match="graph"):
        make_pond(topology, graph=metadata_routing_graph(other))
    native = make_pond(topology, graph=metadata_routing_graph(topology))
    assert native.graph_kind == "caller" and make_pond(topology).graph_kind == "metadata"
    assert "no sediment" in native.describe()["scope"]


# --- face-speed diagnostics ---------------------------------------------------------------------------------------------------
def test_face_speed_is_stage_consistent_unclipped_and_volume_weighted():
    topology, depth = flat_drain_case()
    pond = make_pond(topology)
    state = state_of(pond, depth)
    rate = uniform_rain_rate(pond, 0.0)
    for _ in range(4):
        accepted = step_once(pond, rate, state, 1.0 / 64.0)
        state = accepted.state
    diag = face_speed_diagnostics(pond, accepted)
    st = accepted.step
    stage = stage_face_diagnostics(st)
    h = {a: H(st.face_flow_depth_m[a]) for a in "xy"}
    q = {a: H(st.used_flux_m2_s[a]) for a in "xy"}
    v = {a: H(st.face_volume_m3[a]) for a in "xy"}
    u = {a: np.where(h[a] > 0.0, q[a] / np.where(h[a] > 0.0, h[a], 1.0), 0.0) for a in "xy"}
    for a in "xy":
        assert np.array_equal(H(diag.face_velocity_m_s[a]), u[a]) and np.array_equal(H(stage["face_velocity_m_s"][a]), u[a])
        fr = np.where(h[a] > 0.0, np.abs(u[a]) / np.sqrt(GRAVITY_M_S2 * np.where(h[a] > 0.0, h[a], 1.0)), 0.0)
        np.testing.assert_allclose(H(diag.face_froude[a]), fr, rtol=1e-14, atol=0.0)
    ny, nx = topology.shape
    reference = np.zeros((ny, nx))
    ceiling = np.zeros((ny, nx))
    for r in range(ny):
        for c in range(nx):
            faces = [(-v["x"][r, c], u["x"][r, c]), (v["x"][r, c + 1], u["x"][r, c + 1]),
                     (-v["y"][r, c], u["y"][r, c]), (v["y"][r + 1, c], u["y"][r + 1, c])]
            out = [max(vol, 0.0) for vol, _ in faces]
            if sum(out) > 0.0:
                reference[r, c] = sum(o * abs(s) for o, (_, s) in zip(out, faces, strict=True)) / sum(out)
                ceiling[r, c] = max(abs(s) for o, (_, s) in zip(out, faces, strict=True) if o > 0.0)
    speed = H(diag.cell_speed_m_s)
    np.testing.assert_allclose(speed, reference, rtol=1e-12, atol=0.0)
    assert np.all(speed <= ceiling * (1.0 + 1e-12))
    assert diag.stats["n_cells_with_outflow"] == int(np.count_nonzero(ceiling))
    assert diag.stats["max_face_froude"] == pytest.approx(max(np.max(H(diag.face_froude["x"])), np.max(H(diag.face_froude["y"]))), rel=1e-14)
    assert diag.stats["n_open_faces"] == int(np.count_nonzero(topology.geometry.fx_type)) + int(np.count_nonzero(topology.geometry.fy_type))
    assert diag.stats["n_wet_faces"] + diag.stats["n_dry_open_faces"] == diag.stats["n_open_faces"]
    assert diag.slope_note == SLOPE_NOTE and "friction" in SLOPE_NOTE


def test_no_face_flow_gives_exactly_zero_speed():
    pond = make_pond(topo_from(np.zeros((3, 3))))
    state = state_of(pond, np.full((3, 3), 0.125))
    accepted = step_once(pond, uniform_rain_rate(pond, 0.0), state, 1.0 / 128.0)
    diag = face_speed_diagnostics(pond, accepted)
    assert np.array_equal(H(diag.cell_speed_m_s), np.zeros((3, 3))) and np.array_equal(H(diag.cell_outflow_volume_m3), np.zeros((3, 3)))
    assert diag.stats["n_cells_with_outflow"] == 0 and diag.stats["max_face_froude"] == 0.0
    with pytest.raises(PondHydraulicsError):
        face_speed_diagnostics(pond, accepted, froude_reference=0.0)


def test_dry_faces_are_counted_not_clipped():
    topology, _ = flat_drain_case()
    pond = make_pond(topology)
    depth = np.zeros((3, 4))
    depth[0, 0] = 0.125
    accepted = step_once(pond, uniform_rain_rate(pond, 0.0), state_of(pond, depth), 1.0 / 128.0)
    stats = face_speed_diagnostics(pond, accepted).stats
    assert stats["n_wet_faces"] > 0 and stats["n_dry_open_faces"] > 0


# --- net-elevation refresh with carried state -----------------------------------------------------------------------------------
def test_refresh_rebinds_a_new_geometry_and_carries_the_state():
    bed = np.array([[1.0, 1.0625, 1.125, 1.1875], [0.9375, 1.0, 1.0625, 1.125], [0.875, 0.9375, 1.0, 1.0625]])
    topology = topo_from(bed, [(1, 3, 0, 1)], drop=0.25)
    pond = make_pond(topology)
    depth = np.full((3, 4), 0.0625)
    depth[2, 0] += 0.0625
    first = evolve_pond(pond, uniform_rain_rate(pond, 0.0), state_of(pond, depth), 0.125,
                        ExperimentalControl(max_dt_s=1.0 / 64.0, min_dt_s=1.0 / 4096.0))
    assert np.any(first.state.qx_m2_s != 0.0) or np.any(first.state.qy_m2_s != 0.0)
    new_bed = np.array(topology.source_z_full)
    new_bed[1, 1] += 0.5  # the native bed (committed by the caller) rose by 0.5 m at cell (0, 0)
    update = update_pond_topology(topology, new_bed, 0.25)
    assert update.refreshed
    before = snapshot(first.state)
    pond2, carried = rebind_pond(pond, update.topology, first.state)
    assert carried is first.state and same(before, carried) and carried.next_dt_cap_s is not None
    assert pond2.topology.geometry.z[0, 0] == bed[0, 0] + 0.5 and pond.topology.geometry.z[0, 0] == bed[0, 0]
    assert pond2.solver is not pond.solver and pond2.backend == pond.backend
    second = evolve_pond(pond2, uniform_rain_rate(pond2, 0.0), carried, 1.0 / 16.0,
                         ExperimentalControl(max_dt_s=1.0 / 64.0, min_dt_s=1.0 / 4096.0))
    assert second.start_state.next_dt_cap_s == first.state.next_dt_cap_s and second.closure().closes
    other = topo_from(bed, [(0, 3, 0, 1)], drop=0.25)
    with pytest.raises(PondHydraulicsError, match="identity"):
        rebind_pond(pond, other, first.state)


# --- Numba (optional) ---------------------------------------------------------------------------------------------------------------
def test_numba_matches_numpy_on_the_same_input():
    from maple_syrup.experimental_numba import ExperimentalNumbaUnavailableError

    topology, depth = flat_drain_case()
    cpu = make_pond(topology, ksat=1e-6, drainage=1.0)
    try:
        compiled = make_pond(topology, "numba", ksat=1e-6, drainage=1.0)
    except ExperimentalNumbaUnavailableError:
        pytest.skip("Numba is unavailable; the compiled pond solver is not exercised")
    a = step_once(cpu, uniform_rain_rate(cpu, 2e-5), state_of(cpu, depth, 0.1), 1.0 / 64.0)
    b = step_once(compiled, uniform_rain_rate(compiled, 2e-5), state_of(compiled, depth, 0.1), 1.0 / 64.0)
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        np.testing.assert_allclose(H(getattr(b.state, name)), H(getattr(a.state, name)), rtol=2e-12, atol=1e-14)


# --- CUDA (skips only when unavailable; the bounds are the unchanged ones of the existing CPU/GPU agreement) --------------------------------
def test_cuda_lake_at_rest_is_exact(gpu):
    level = 1.0
    depth = np.maximum(level - BED, 0.0)
    pond = make_pond(topo_from(BED), "cuda")
    result = evolve_pond(pond, uniform_rain_rate(pond, 0.0), state_of(pond, depth), 0.125, FINE)
    assert result.reached_target and np.array_equal(H(result.state.depth_m), depth)
    assert not np.any(H(result.state.qx_m2_s)) and not np.any(H(result.state.qy_m2_s))
    assert result.closure().closes


def test_cuda_and_numpy_agree_on_identical_input_and_step_sequence(gpu):
    topology, depth = flat_drain_case()
    cpu = make_pond(topology, ksat=1e-6, drainage=1.0)
    dev = make_pond(topology, "cuda", ksat=1e-6, drainage=1.0)
    assert cpu.topology is dev.topology and dev.xp is gpu
    control = ExperimentalControl(max_dt_s=1.0 / 32.0, min_dt_s=1.0 / 4096.0)
    a = evolve_pond(cpu, uniform_rain_rate(cpu, 2e-5), state_of(cpu, depth, 0.1), 0.5, control)
    b = evolve_pond(dev, uniform_rain_rate(dev, 2e-5), state_of(dev, depth, 0.1), 0.5, control)
    assert (a.n_accepted_steps, a.n_rejected_attempts, a.elapsed_s) == (b.n_accepted_steps, b.n_rejected_attempts, b.elapsed_s)
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        np.testing.assert_allclose(H(getattr(b.state, name)), H(getattr(a.state, name)), rtol=2e-12, atol=1e-14)
    ca, cb = a.closure(), b.closure()
    assert ca.closes and cb.closes
    np.testing.assert_allclose(cb.export_m3, ca.export_m3, rtol=2e-12, atol=1e-14)
    for axis in "xy":
        np.testing.assert_allclose(cb.face_net_m3[axis], ca.face_net_m3[axis], rtol=2e-12, atol=1e-14)


def test_cuda_single_step_diagnostics_match_numpy(gpu):
    topology, depth = flat_drain_case()
    cpu = make_pond(topology)
    dev = make_pond(topology, "cuda")
    a = step_once(cpu, uniform_rain_rate(cpu, 0.0), state_of(cpu, depth), 1.0 / 64.0)
    b = step_once(dev, uniform_rain_rate(dev, 0.0), state_of(dev, depth), 1.0 / 64.0)
    da, db = face_speed_diagnostics(cpu, a), face_speed_diagnostics(dev, b)
    np.testing.assert_allclose(H(db.cell_speed_m_s), H(da.cell_speed_m_s), rtol=2e-12, atol=1e-14)
    for axis in "xy":
        np.testing.assert_allclose(H(db.face_froude[axis]), H(da.face_froude[axis]), rtol=2e-12, atol=1e-14)
        np.testing.assert_allclose(H(b.step.face_volume_m3[axis]), H(a.step.face_volume_m3[axis]), rtol=2e-12, atol=1e-14)
    assert db.stats["n_wet_faces"] == da.stats["n_wet_faces"]


def test_cuda_rejection_leaves_the_device_state_untouched(gpu):
    pond = make_pond(topo_from(np.zeros((3, 3))), "cuda")
    state = state_of(pond, np.ones((3, 3)))
    before = snapshot(state)
    with pytest.raises(HydraulicStepRejected):
        step_once(pond, uniform_rain_rate(pond, 0.0), state, 10.0)
    assert same(before, state)


def test_cuda_refresh_rebind_carries_the_device_state(gpu):
    topology, depth = flat_drain_case()
    pond = make_pond(topology, "cuda")
    first = evolve_pond(pond, uniform_rain_rate(pond, 0.0), state_of(pond, depth), 0.0625,
                        ExperimentalControl(max_dt_s=1.0 / 64.0, min_dt_s=1.0 / 4096.0))
    new_bed = np.array(topology.source_z_full)
    new_bed[1, 1] += 0.5
    update = update_pond_topology(topology, new_bed, 0.25)
    pond2, carried = rebind_pond(pond, update.topology, first.state)
    assert update.refreshed and carried is first.state and pond2.topology.generation == 1
    after = evolve_pond(pond2, uniform_rain_rate(pond2, 0.0), carried, 0.0625,
                        ExperimentalControl(max_dt_s=1.0 / 64.0, min_dt_s=1.0 / 4096.0))
    assert after.closure().closes
