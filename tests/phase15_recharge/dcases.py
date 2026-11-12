"""Helpers of the Phase 1.5 S2 daily lifecycle controls (real native Dense bed via `ncases`; CPU)."""
from __future__ import annotations

import numpy as np
from ncases import BIG, DAY, Box, build, field_of, fingerprint, host, pet, raw  # noqa: F401
from storm_cases import layers

from maple_syrup import layered_event as ev
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se  # noqa: F401
from maple_syrup import subsurface as ss  # noqa: F401
from maple_syrup import water_lifecycle as wl
from maple_syrup import daily_water as dw

RUN = "ab" * 32
MM = 1e-3
STORM_CONTROL = ev.LayeredEventControl(max_dt_s=20.0, max_recession_s=7200.0)
ROOT = dict(edges=None)


def root_layers(n, *, bottom="open", gravity=True, diffusivity=0.0):
    """Root's known-answer profile: n layers of 100 mm, theta_FC = 0.2 (20 mm), theta_SAT = 0.4 (40 mm), residual 0."""
    return layers(edges=tuple(0.1 * k for k in range(n + 1)), res=0.0, fc=0.2, sat=0.4, stress=0.1, diffusivity=diffusivity, gravity=gravity,
                  bottom=bottom)


def start(c, run=RUN):
    return dw.start_daily_lineage(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0), run_digest=run)


def act(c, state, p, t=5.0, i=10.0, pet_rate=0.0, **kw):
    plan = dw.next_plan(state, p, i, t)
    return plan, dw.daily_water_action(c.lop, state, expected_parent=wl.expect_parent(state), plan=plan, pet_terrain_m_s=pet(c, pet_rate), **kw)


def with_soil(c, water_mm_by_cell):
    """A NEW closed boundary on the same lifecycle/bed whose soil is `water_mm_by_cell[cell][layer]` (mm)."""
    w = np.ascontiguousarray(np.array(water_mm_by_cell, dtype=np.float64).T * MM)
    soil = ss.column_state(c.col, w, 0.0)
    return wl.start_lifecycle(c.lop, bed=c.bed, soil=soil, history=c.s0.payload.history, t_s=0.0)


def uniform_soil(c, per_layer_mm):
    return with_soil(c, [list(per_layer_mm)] * c.n)


def run_storm(c, state, *, max_steps=None, rounds=12):
    """Drive the EXISTING bounded native storm loop (its own step/target limits) until it reports genuine completion."""
    res = wl.evolve_storm(c.lop, state, expected_parent=wl.expect_parent(state), until_s=state.t_s + 40_000.0, max_accepted_steps=max_steps)
    for _ in range(rounds):
        if res.complete or max_steps is not None:
            break
        res = wl.evolve_storm(c.lop, res.state, expected_parent=wl.expect_parent(res.state), until_s=res.state.t_s + 40_000.0)
    return res


def lifetime(c, state):
    return wl.audit_lifecycle(c.lop, state)
