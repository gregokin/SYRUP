"""Phase 1.5 S2: Root's independent known-answer controls driven through the ACTUAL native daily lifecycle (genuine native Dense bed, real
preferential/matrix/router/evaporation components, one 86400 s day) rather than the bare column."""
from __future__ import annotations

import numpy as np
import pytest
from dcases import DAY, MM, act, build, dw, host, root_layers, start, uniform_soil, wl

GRID = np.full((4, 6), 0.2)
# (initial mm per layer, rain mm, bottom, expected final mm, expected drainage mm per cell) -- Root `root_independent_controls.md` table
TABLE = [
    ([0], 10, "closed", [10], 0),
    ([20], 10, "closed", [30], 0),  # FC (not SAT) triggers movement: 30 mm is above FC and below SAT and stays
    ([20], 10, "open", [20], 10),
    ([0, 0], 30, "closed", [20, 10], 0),
    ([20, 20], 30, "closed", [30, 40], 0),
    ([20, 20], 30, "open", [20, 20], 30),
    ([20, 20, 20], 50, "closed", [30, 40, 40], 0),
    ([20, 20, 20], 50, "open", [20, 20, 20], 50),
    ([40, 40], 1, "open", [20, 20], 41),
    ([40, 40, 0], 10, "closed", [20, 30, 40], 0),
]


@pytest.mark.parametrize("initial,rain,bottom,final,drain", TABLE)
def test_root_table_through_the_native_daily_lifecycle(initial, rain, bottom, final, drain):
    c = build(GRID, lay=root_layers(len(initial), bottom=bottom), theta=[x / 100.0 for x in initial])
    s0 = start(c)
    plan, r = act(c, s0, rain, t=rain)  # T = P: equality attempts recharge
    assert plan.selection == "recharge" and r.status == "complete"
    soil = host(r.state.payload.soil.water_m)
    np.testing.assert_allclose(soil / MM, np.array(final)[:, None] * np.ones((1, c.n)), rtol=0, atol=1e-9)  # EVERY cell
    area = 0.25
    assert r.receipt.drainage_m3 == pytest.approx(drain * MM * c.n * area, abs=1e-12)
    led = r.state.payload.ledger.hydrology
    assert led.surface_return_m3 == 0.0  # no premature top-layer saturation return
    assert led.rain_m3 == pytest.approx(rain * MM * c.n * area, rel=1e-12)  # P credited ONCE, as external rain
    assert led.matrix_intake_m3 == 0.0 and led.preferential_intake_m3 == 0.0  # dry surface: recharge is NOT modeled matrix/pref intake
    assert r.state.t_s == DAY and r.receipt.t_end_s - r.receipt.t_start_s == DAY
    audit = wl.audit_lifecycle(c.lop, r.state)
    assert abs(audit["residual_m3"]) <= audit["bound_m3"]  # actual whole-lifetime closure


def test_gravity_disabled_control_keeps_the_pulse_in_the_top_layer_when_it_fits():
    c = build(GRID, lay=root_layers(2, bottom="closed", gravity=False), theta=[0.0, 0.0])
    _, r = act(c, start(c), 30, t=30)  # 30 mm into a 40 mm top layer; no gravity movement
    np.testing.assert_allclose(host(r.state.payload.soil.water_m)[:, 0] / MM, [30, 0], atol=1e-9)


@pytest.mark.parametrize("p,selection", [(0.0, "no_rain"), (4.999, "recharge"), (5.0, "recharge"), (5.001, "storm")])
def test_selection_boundary_and_branch_api(p, selection):
    c = build(GRID, lay=root_layers(2), theta=[0.1, 0.1])
    s0 = start(c)
    plan = dw.next_plan(s0, p, 10.0, 5.0)
    assert plan.selection == selection
    if selection == "storm":
        with pytest.raises(dw.DailyRefusal, match="storm occurrence"):
            dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, pet_terrain_m_s=np.zeros(c.n))
    else:
        r = dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, pet_terrain_m_s=np.zeros(c.n))
        assert r.status == "complete" and r.state.daily.ordinal == 1 and r.state.daily.last_branch == selection
        assert r.state.t_s == DAY  # ONE actual fixed day, for the P = 0 day too
