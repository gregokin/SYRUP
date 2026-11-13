"""Helpers of the G2a suite: the genuine native Dense daily lifecycle fixtures of tests/phase15_recharge (read-only reuse)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase15_recharge"))

from dcases import (  # noqa: F401
    DAY,
    RUN,
    STORM_CONTROL,
    act,
    build,
    dw,
    host,
    pet,
    run_storm,
    start,
    wl,
)
from storm_cases import layers

from maple_syrup import daily_moisture_output as dmo
from maple_syrup import soil_moisture_output as mo

GRID = np.full((4, 6), 0.2)
LIMITS = {"max_snapshots": 50, "max_total_bytes": 1 << 30, "max_manifest_bytes": 8192, "chunk_bytes": 1 << 16,
          "host_memory_budget_bytes": 1 << 30, "host_reserve_bytes": 0, "disk_reserve_bytes": 0}


def limits(**kw):
    return mo.MoistureOutputLimits(**{**LIMITS, **kw})


def case(backend="numpy"):
    return build(GRID, lay=layers(bottom="open"), theta=(0.3, 0.3), backend=backend)


def fp(c, s):
    a = wl._arrays_of(s)
    h = lambda x: hashlib.sha256(np.ascontiguousarray(host(x)).tobytes()).hexdigest()
    return (s.digest, float(s.t_s), int(s.revision), s.role, int(s.daily.ordinal), s.lifetime.as_dict(),
            {"h": h(a.surface_m), "soil": h(a.soil.water_m), "j": h(a.history.wetting_m)})


def interevent(c, state, p, *, withdraw=None, pet_rate=2.0 ** -24):
    """One daily occurrence (actual daily_water_action) plus, optionally, an ACTUAL request_withdrawal (what LEAF's accepted water would do)."""
    _, r = act(c, state, p, pet_rate=pet_rate)
    ie = r.state
    if withdraw is not None:
        q = np.zeros((c.L, c.n))
        q[0] = withdraw
        ie, _ = wl.request_withdrawal(c.lop, ie, expected_parent=wl.expect_parent(ie), depth_m=q, identity="leaf")
    return ie


def storm_interevent(c, state, p=3.0):
    plan = dw.next_plan(state, p, 360.0, 1.0)
    st = dw.begin_daily_storm(c.lop, state, expected_parent=wl.expect_parent(state), plan=plan, control=STORM_CONTROL)
    res = run_storm(c, st)
    assert res.complete
    return dw.finish_daily_storm(c.lop, res.state, expected_parent=wl.expect_parent(res.state), plan=plan, pet_terrain_m_s=pet(c, 0.0)).state


def close(c, ie, output=None, **kw):
    """The facade with the caller's own capability for the accepted interevent parent."""
    return dmo.close_final_day(c.lop, ie, expected_parent=wl.expect_parent(ie), output=output, **kw)
