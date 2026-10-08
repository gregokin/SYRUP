"""StepEngine contracts on a tiny synthetic network: caller arrays untouched, refusals before any state changes, pit law inputs."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from maple_syrup import legacy_native as N

from .helpers import DX, channel, pit_channel

pytestmark = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="Numba not installed")


def build(graph):
    from maple_syrup.legacy_native_numba import StepEngine, WetLawRunner
    from maple_syrup.legacy_physics_numba import prepare_legacy_physics
    from maple_syrup.sediment_physics import physics_grid, plot1_sediment_parameters

    ny, nx = graph.shape
    grid = physics_grid(np.asarray(graph.slope), np.asarray(graph.active), DX * DX)
    holdings = np.broadcast_to(np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2]) * 2.5, (ny, nx, 6)).copy()
    ctx = prepare_legacy_physics(plot1_sediment_parameters(), grid, np.zeros((ny, nx)), holdings)
    net = N.native_network(graph)
    return StepEngine(net, WetLawRunner(ctx), dt=1.0), net


def wet_inputs(shape, depth=0.004, velocity=0.05, rain=2.0e-5):
    return np.full(shape, depth), np.full(shape, velocity), np.full(shape, rain)


def test_step_does_not_mutate_caller_arrays_and_returns_a_nonvacuous_row():
    engine, _ = build(channel(5))
    depth, vel, rain = wet_inputs((1, 5))
    before = [a.copy() for a in (depth, vel, rain)]
    res = engine.step(depth, vel, rain)
    for a, b in zip((depth, vel, rain), before, strict=True):
        assert np.array_equal(a, b)
    assert res.ledger_row.shape[0] == 13 and res.ledger_row[0].sum() > 0.0  # pickup happened
    assert engine.cum_det.sum() > 0.0 and engine.total_counts.sum() >= 0


def test_invalid_wet_law_input_raises_before_any_engine_state_changes():
    from maple_syrup.sediment_physics import SedimentPhysicsError

    engine, _ = build(channel(5))
    depth, vel, rain = wet_inputs((1, 5))
    engine.step(depth, vel, rain)
    snapshot = {name: getattr(engine, name).copy() for name in ("M1", "Q1", "Qin1", "cum_det", "cum_dep", "cum_clip", "v_prev")}
    counts, regimes = engine.total_counts.copy(), engine.total_regime_counts.copy()
    bad = depth.copy()
    bad[0, 2] = -1.0
    with pytest.raises(SedimentPhysicsError, match="depth_m"):
        engine.step(bad, vel, rain)
    bad_v = vel.copy()
    bad_v[0, 1] = np.nan
    with pytest.raises(SedimentPhysicsError, match="velocity_m_s"):
        engine.step(depth, bad_v, rain)
    for name, value in snapshot.items():
        assert np.array_equal(getattr(engine, name), value), name
    assert np.array_equal(engine.total_counts, counts) and np.array_equal(engine.total_regime_counts, regimes)


def test_wrong_shape_or_dtype_is_refused():
    from maple_syrup.legacy_native_numba import LegacyNativeError

    engine, _ = build(channel(5))
    depth, vel, rain = wet_inputs((1, 5))
    with pytest.raises(LegacyNativeError):
        engine.step(depth[:, :4], vel, rain)
    with pytest.raises((LegacyNativeError, ValueError, TypeError)):
        engine.runner.run(depth.reshape(-1).astype(np.float32), vel.reshape(-1), rain.reshape(-1), engine.v_prev.reshape(-1), 1.0)


def test_pit_with_wet_rain_inputs_has_zero_detachment_and_never_moves():
    engine, _ = build(pit_channel())
    depth, vel, rain = wet_inputs((1, 5))
    vel[0, 2] = 0.0  # zero slope gives zero velocity
    for _ in range(3):
        res = engine.step(depth, vel, rain)
    assert engine.det[2].sum() == 0.0  # raindrop law: (100 S)^c = 0 at zero slope
    assert engine.v_prev[2].sum() == 0.0 and engine.Q1[2].sum() == 0.0
    assert engine.cum_dep[2].sum() > 0.0  # the walks of the two draining neighbours credit the pit, then stop there
    assert engine.total_counts[N.C_TERMINAL] > 0 and engine.total_counts[N.C_RING] == 0
    assert res.ledger_row[N_COL["mobile_terminal_kg"]].sum() == pytest.approx(engine.M1[2].sum(), rel=1e-14)


def test_legacy_order_erase_needs_flag_combination():
    from maple_syrup.legacy_native_numba import (
        LegacyNativeError,
        StepEngine,
    )

    engine, net = build(channel(5))
    with pytest.raises(LegacyNativeError, match="legacy"):
        StepEngine(net, engine.runner, dt=1.0, erase_on=True)  # default source order is 'index'
    StepEngine(net, engine.runner, dt=1.0, source_order="legacy", erase_on=True)


def _persistent(engine):
    return {name: getattr(engine, name).copy() for name in ("M1", "Q1", "Qin1", "v_prev", "cum_det", "cum_dep", "cum_clip",
                                                           "total_counts", "total_regime_counts")}


@pytest.mark.parametrize("bad", ["float32", "int64", "list", "wrong_shape", "scalar", "masked", "cupy_like"])
def test_inputs_are_strictly_host_float64_of_the_grid_shape_and_never_converted(bad):
    from maple_syrup.legacy_native_numba import LegacyNativeError

    engine, _ = build(channel(5))
    engine.step(*wet_inputs((1, 5)))
    before = _persistent(engine)
    depth, vel, rain = wet_inputs((1, 5))
    depth = {"float32": depth.astype(np.float32), "int64": np.ones((1, 5), dtype=np.int64), "list": depth.tolist(),
             "wrong_shape": np.full((5,), 0.004), "scalar": 0.004, "masked": np.ma.masked_array(depth),
             "cupy_like": type("FakeCuPy", (), {"shape": (1, 5), "dtype": np.float64})()}[bad]
    with pytest.raises(LegacyNativeError):
        engine.step(depth, vel, rain)
    assert engine.poisoned is None
    for name, value in before.items():
        assert np.array_equal(getattr(engine, name), value), name
    engine.step(*wet_inputs((1, 5)))  # still usable


def test_failure_after_the_walk_poisons_the_engine_until_reset(monkeypatch):
    from maple_syrup.legacy_native_numba import LegacyNativeError

    engine, _ = build(channel(5))
    engine.step(*wet_inputs((1, 5)))
    before = _persistent(engine)
    real_cn = engine.k.cn

    def broken_cn(*args):
        real_cn(*args)
        args[12][0, 0] = np.nan  # M2: a non-finite new pool, which the legacy `m < 0` clip cannot catch

    monkeypatch.setattr(engine.k, "cn", broken_cn)
    from maple_syrup.sediment_physics import SedimentPhysicsError

    with pytest.raises(SedimentPhysicsError, match="not finite"):
        engine.step(*wet_inputs((1, 5)))
    assert engine.poisoned
    for name, value in before.items():  # the persistent time levels and maps were never swapped or accumulated
        assert np.array_equal(getattr(engine, name), value), name
    with pytest.raises(LegacyNativeError, match="poisoned"):
        engine.step(*wet_inputs((1, 5)))
    monkeypatch.undo()
    engine.reset()
    assert engine.poisoned is None and not engine.M1.any() and not engine.cum_det.any()
    engine.step(*wet_inputs((1, 5)))


def _col_map():
    from maple_syrup.legacy_native_numba import LEDGER_COLUMNS

    return {name: i for i, name in enumerate(LEDGER_COLUMNS)}


N_COL = _col_map()
