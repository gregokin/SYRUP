"""Coupled physics of the pond event on an ACTUAL MAPLE bed: dry splash (nonzero path), wet pickup, class and water closure,
ordering, backends. Nothing here was run by its author (file-only tools)."""
import math

import numpy as np
import pytest
from pond_helpers import DX, NC, dry_plane, host, wet_plane

from maple_syrup.sediment_physics import plot1_sediment_parameters


def expected_dry_mass(params, rain, slope, dt, area):
    """The wet model's raindrop law on a dry cell (KE model 2, literal precedence, no vegetation, depth 0), as in
    tests/splash_spill: mass over dt for a class present in the cell and not subject to the phi-2 cap."""
    r2 = rain * 1000.0
    ke = 29.0 - 20.88 * math.exp(-180.0 * r2)
    x = (params.raindrop_a / 1200.0) * (ke * r2 * 1200.0) ** params.raindrop_b * (slope * 100.0) ** params.raindrop_c
    rate = 2.0 * x / params.particle_density_kg_m3 / params.reference_interval_s
    return rate * area * params.particle_density_kg_m3 * dt


def test_dry_splash_runs_the_actual_nonzero_native_path(be):
    event, state = dry_plane(be)
    rain = 60.0 / 3.6e6
    out = event.step(state, rain, 0.5)
    assert out.record.splash_exchange and out.state.counters.n_splash_exchanges == 1
    assert not np.any(host(out.state.hydraulic.depth_m))  # post-hydrology depth is exactly 0 on every cell
    params = plot1_sediment_parameters()
    # slope = drop / dx = 0.02 / 0.5 on every cell; one scalar per present, uncapped class
    per_cell = np.full(NC, expected_dry_mass(params, rain, 0.04, 0.5, DX * DX))
    requested = host(out.record.by_class["splash_requested"])
    for k in (0, 3):
        assert requested[k] == pytest.approx(12 * per_cell[k], rel=1e-10)
    actual = host(out.record.by_class["splash_actual"])
    assert np.all(actual <= requested + 1e-15) and actual.sum() > 0.0  # MAPLE caps the ACTUAL pickup
    assert not np.any(host(out.record.by_class["wet_requested"]))  # the wet law owns no cell: no double pickup
    deposited, exported = host(out.record.by_class["splash_deposited"]), host(out.record.by_class["splash_export"])
    slack = NC * 12 * event.context.mass_resolution_kg + 1e-12
    np.testing.assert_allclose(deposited + exported, actual, rtol=0.0, atol=slack)
    audit = event.audit(out.state)
    assert audit["sediment"]["closes"], audit["sediment"]
    assert audit["water"]["closes"]


def test_dry_splash_exports_only_through_explicit_outlet_faces(be):
    event, state = dry_plane(be)
    result = event.evolve(state, 2.0)
    audit = event.audit(result.state)
    assert audit["sediment"]["closes"] and result.state.counters.n_splash_exchanges == result.n_accepted_steps
    exported = np.asarray(audit["sediment"]["export_splash_kg"])
    assert np.all(exported >= 0.0)
    # the splash exchange never moves the mobile pool: preexisting mobile mass stays zero here
    assert np.allclose(audit["sediment"]["mobile_kg"], 0.0, atol=NC * 12 * event.context.mass_resolution_kg)


def test_wet_pickup_is_actual_drives_transport_and_everything_closes(be):
    event, state = wet_plane(be)
    result = event.evolve(state, 1.0)
    assert result.reached_target
    audit = event.audit(result.state)
    s = audit["sediment"]
    assert sum(s["wet_requested_kg"]) > 0.0 and sum(s["wet_actual_kg"]) > 0.0
    assert all(a <= r + 1e-15 for a, r in zip(s["wet_actual_kg"], s["wet_requested_kg"]))  # actual, not requested
    assert s["closes"], s
    assert audit["water"]["closes"], audit["water"]
    rt = event._runtime(result.state.topology, result.state.hydraulic)
    np.testing.assert_allclose(host(rt.grid.slope), 0.04, rtol=1e-12)  # TOPOGRAPHIC slope of the held real bed
    froude = result.last_record.froude
    assert math.isfinite(froude["max_face_froude"]) and froude["max_face_speed_m_s"] >= 0.0
    assert result.state.counters.max_face_froude >= 0.0  # reported, never clipped
    # the sediment faces carry the transported mass in MAPLE's layout, gross >= |net|
    assert np.all(np.asarray(s["face_gross_kg"]["x"]) >= np.abs(np.asarray(s["face_net_kg"]["x"])) - 1e-15)
    assert result.state.sediment.n_native_calls == 2 * result.n_accepted_steps


def test_per_step_class_ledger_matches_the_accumulated_ledger(be):
    event, state = wet_plane(be)
    totals = np.zeros(NC)
    for _ in range(4):
        out = event.step(state, 100.0 / 3.6e6, 0.05)
        totals += host(out.record.by_class["wet_actual"])
        state = out.state
    np.testing.assert_allclose(host(state.sediment.wet_actual), totals, rtol=1e-13, atol=0.0)
    # per cell pickup / deposition totals are consistent with the class totals
    np.testing.assert_allclose(host(state.sediment.pickup_cell).sum(axis=(0, 1)), totals, rtol=1e-12, atol=1e-18)


def test_friction_slope_is_a_separate_documented_choice_from_the_legacy_estimate(cpu):
    from dataclasses import replace

    friction_event, friction_state = wet_plane(cpu)
    # identical time controls (hence the identical hydraulic dt sequence): only the slope choices differ
    legacy_event, legacy_state = wet_plane(
        cpu, control=replace(friction_event.control, shear_slope="topographic", stream_power_slope="topographic"))
    assert friction_event.control.shear_slope == "friction" and legacy_event.control.shear_slope == "topographic"
    a = friction_event.evolve(friction_state, 0.5)
    b = legacy_event.evolve(legacy_state, 0.5)
    # identical hydraulics (the slope choice acts only on the wet laws)...
    np.testing.assert_array_equal(host(a.state.hydraulic.depth_m), host(b.state.hydraulic.depth_m))
    # ...but the stream-power-driven transport velocity differs: S_f = f u^2 / (8 g h) is not the bed slope
    assert not np.allclose(host(a.state.sediment_velocity_m_s), host(b.state.sediment_velocity_m_s))
    for event, result in ((friction_event, a), (legacy_event, b)):
        assert event.audit(result.state)["sediment"]["closes"]
