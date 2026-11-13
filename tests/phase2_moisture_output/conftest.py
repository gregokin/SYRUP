"""Fixtures of the G0 moisture-output suite: the genuine native Dense daily lifecycle fixtures of tests/phase15_recharge (read-only reuse) driven through REAL daily actions, withdrawals and closes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase15_recharge"))

from dcases import RUN, STORM_CONTROL, act, build, dw, host, pet, run_storm, start, wl  # noqa: E402
from storm_cases import layers  # noqa: E402

from maple_syrup import soil_moisture_output as mo  # noqa: E402

GRID = np.full((4, 6), 0.2)
BIGQ = dict(max_snapshots=50, max_total_bytes=1 << 30, max_manifest_bytes=8192, chunk_bytes=1 << 20, host_memory_budget_bytes=1 << 30, host_reserve_bytes=0, disk_reserve_bytes=0)


def limits(**kw):
    return mo.MoistureOutputLimits(**{**BIGQ, **kw})


def numba_ok() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.fixture(params=["numpy"] + (["numba"] if numba_ok() else []))
def case(request):
    return build(GRID, lay=layers(bottom="open"), theta=(0.3, 0.3), backend=request.param)


class Day:
    """Drive one closed day: daily action (+ optional withdrawal) + close; returns the closed boundary."""

    def __init__(self, c):
        self.c = c

    def close(self, state, p=2.0, *, withdraw=None, pet_rate=2.0 ** -24):
        c = self.c
        _, r = act(c, state, p, pet_rate=pet_rate)
        ie = r.state
        if withdraw is not None:
            q = np.zeros((c.L, c.n))
            q[0] = withdraw
            ie, _ = wl.request_withdrawal(c.lop, ie, expected_parent=wl.expect_parent(ie), depth_m=q, identity="leaf")
        return ie, wl.close_interevent(c.lop, ie, expected_parent=wl.expect_parent(ie))

    def storm_day(self, state, p=3.0):
        c = self.c
        plan = dw.next_plan(state, p, 360.0, 1.0)
        st = dw.begin_daily_storm(c.lop, state, expected_parent=wl.expect_parent(state), plan=plan, control=STORM_CONTROL)
        res = run_storm(c, st)
        assert res.complete
        out = dw.finish_daily_storm(c.lop, res.state, expected_parent=wl.expect_parent(res.state), plan=plan, pet_terrain_m_s=pet(c, 0.0))
        return out.state, wl.close_interevent(c.lop, out.state, expected_parent=wl.expect_parent(out.state))


def make(c, tmp_path, n=2, **kw):
    s0 = start(c)
    out = mo.MoistureOutput.create(tmp_path / "out", c.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=n, limits=kw.pop("limits", limits()), **kw)
    return out, s0


def notify(out, c, state, **kw):
    return out.notify_day_closed(c.lop, state, expected_parent=wl.expect_parent(state), **kw)


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip. Hardware tests are opt-in (SYRUP_RELEASE_GPU=1) and need an operator-selected device; the default lane never imports or probes the GPU backend."""
    import os

    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA moisture output is not exercised (no claim)")
    return backend.cupy_module()
