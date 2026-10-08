"""CN parity with the accepted legacy replay, independent clip reconstruction, glue validation, engine contracts."""
from __future__ import annotations

import numpy as np
import pytest

from maple_syrup import legacy_driver as D
from maple_syrup import legacy_native as N
from maple_syrup import legacy_transport as L
from maple_syrup.legacy_native_numba import get_kernels
from maple_syrup.sediment_physics import REGIME_CODES

from .helpers import DX, arrays, channel, converging, pit_channel, run_walk


def _cn(net, det, dep, v, M1, Q1, Qin1, dt=1.0):
    k = get_kernels(False)
    n, nc = det.shape
    M2, Q2, Qin2, clip = (np.zeros((n, nc)) for _ in range(4))
    cn_export, endpoint = np.zeros(nc), np.zeros(nc)
    k.cn(net.order, net.donors, net.cn_receiver, net.outlet, M1, Q1, Qin1, det, dep, v, dt, net.dx_m, M2, Q2, Qin2, clip,
         cn_export, endpoint)
    return M2, Q2, Qin2, clip, cn_export, endpoint


def test_donor_view_and_pool_match_the_accepted_replay_without_pits():
    graph = converging()
    new = N.native_network(graph)
    old = L.legacy_network(graph)
    assert np.array_equal(new.donors, old.donors)
    assert np.array_equal(np.sort(new.order), np.sort(old.order))
    rng = np.random.default_rng(7)
    n, nc = new.active.size, 3
    a = arrays(new, nc=nc, par=0.4)
    a["det"] = np.where(new.active[:, None], rng.uniform(0.0, 1.0, (n, nc)), 0.0)
    v = np.where(new.active[:, None], rng.uniform(0.05, 0.5, (n, nc)), 0.0)
    M1, Q1, Qin1 = (np.where(new.active[:, None], rng.uniform(0.0, 0.5, (n, nc)), 0.0) for _ in range(3))
    step = L.legacy_transport_step(old, a["det"], a["inv_l"], a["law"], a["regime"], v, M1, Q1, Qin1, 1.0)
    depos, ring, _, _, _ = run_walk(new, a)
    assert np.array_equal(depos, step.deposition_rate_kg_s)  # same walk arithmetic and order
    assert np.array_equal(ring, step.ring_deposition_rate_kg_s)
    M2, Q2, _, clip, cn_export, _ = _cn(new, a["det"], depos, v, M1, Q1, Qin1)
    assert np.array_equal(M2, step.mobile_after_kg) and np.array_equal(Q2, step.flux_after_kg_s)
    assert np.array_equal(clip, step.clipping_source_kg) and np.array_equal(cn_export, step.cn_export_kg)


def test_clipping_source_is_reconstructed_from_the_unclipped_trial_and_closes_the_identity():
    net = N.native_network(channel(3))
    n, nc = net.active.size, 1
    det = np.zeros((n, nc))
    dep = np.zeros((n, nc))
    dep[1, 0] = 0.8  # deposition credited to an empty cell: the trial pool is negative
    v = np.full((n, nc), 0.3)
    M1, Q1, Qin1 = np.zeros((n, nc)), np.zeros((n, nc)), np.zeros((n, nc))
    dt = 1.0
    M2, _, _, clip, cn_export, _ = _cn(net, det, dep, v, M1, Q1, Qin1, dt)
    trial = (0.0 - dep[1, 0]) / (1.0 / dt + 0.5 * v[1, 0] / DX)  # independent evaluation of the unclipped solve (cell 1, no inflow)
    assert trial < 0.0 and M2[1, 0] == 0.0
    assert clip[1, 0] == pytest.approx(-trial * (1.0 + 0.5 * dt * v[1, 0] / DX), rel=1e-14)
    resid = M2.sum() - M1.sum() - (det.sum() - dep.sum()) * dt + cn_export.sum() - clip.sum()
    assert abs(resid) < 1e-15


def test_pool_identity_with_terminal_pit_and_ring_diagnostics_not_debited():
    net = N.native_network(pit_channel())
    rng = np.random.default_rng(3)
    n, nc = net.active.size, 2
    a = arrays(net, nc=nc, par=0.5)
    a["det"] = np.where(net.active[:, None], rng.uniform(0.0, 1.0, (n, nc)), 0.0)
    a["det"][2] = 0.0  # a real pit has no own detachment
    v = np.where(net.active[:, None], 0.2, 0.0) * np.ones((n, nc))  # explicit (n, nc)
    v[2] = 0.0  # a pit never moves
    z = np.zeros((n, nc))
    depos, ring, _, _, counts = run_walk(net, a)
    M2, Q2, _, clip, cn_export, _ = _cn(net, a["det"], depos, v, z, z, z)
    resid = M2.sum(0) - (a["det"].sum(0) - depos.sum(0)) + cn_export - clip.sum(0)
    assert np.all(np.abs(resid) < 1e-14)
    assert np.all(Q2[2] == 0.0) and counts[N.C_TERMINAL] > 0 and cn_export.sum() == 0.0  # storage, not export
    assert ring.sum() == 0.0


def test_pack_flags_each_invalid_condition_before_anything_is_walked():
    net = N.native_network(pit_channel())
    k = get_kernels(False)
    n, nc = net.active.size, 1

    def pack(requested=None, law=None, svel=None, rate=None, prev=None, regime=None):
        req = np.zeros((n, nc)) if requested is None else requested
        det, v_used = np.zeros((n, nc)), np.zeros((n, nc))
        erase = np.zeros(n, dtype=bool)
        flags = k.pack(net.active, net.terminal, np.zeros(n), req, np.zeros((n, nc), bool) if law is None else law,
                       np.zeros((n, nc)) if svel is None else svel, np.ones((n, nc)) if rate is None else rate,
                       np.full((n, nc), 2, np.int8) if regime is None else regime,
                       np.zeros((n, nc)) if prev is None else prev, 0.9, 1.0, True, det, v_used, erase)
        return flags, det, v_used, erase

    assert pack()[0] == 0
    bad = np.zeros((n, nc))
    bad[0, 0] = -1.0
    assert pack(requested=bad)[0] & 1
    bad = np.zeros((n, nc))
    bad[0, 0] = np.nan
    assert pack(requested=bad)[0] & 1
    bad = np.zeros((n, nc))
    bad[2, 0] = 0.1
    assert pack(prev=bad)[0] & 16  # a pit with a remembered velocity
    law = np.ones((n, nc), bool)
    req = np.zeros((n, nc))
    req[0, 0] = 1.0
    assert pack(requested=req, law=law, rate=np.zeros((n, nc)))[0] & 8
    flags, _, v_used, _ = pack(prev=np.full((n, nc), 0.5) * np.array([[1], [1], [0], [1], [1]]))
    assert flags == 0 and v_used[0, 0] == pytest.approx(0.45)  # recession memory 0.9


def test_pack_marks_erase_cells_wet_no_rain_and_dry_raining():
    net = N.native_network(channel(3))
    k = get_kernels(False)
    n, nc = net.active.size, 1
    regime = np.array([[1], [0], [2]], dtype=np.int8)
    rain = np.array([0.0, 1e-6, 1e-6])
    det, v_used, erase = np.zeros((n, nc)), np.zeros((n, nc)), np.zeros(n, dtype=bool)
    z = np.zeros((n, nc))
    k.pack(net.active, net.terminal, rain, z, z.astype(bool), z, z, regime, z, 0.9, 1.0, True, det, v_used, erase)
    assert erase.tolist() == [True, True, False]
    k.pack(net.active, net.terminal, rain, z, z.astype(bool), z, z, regime, z, 0.9, 1.0, False, det, v_used, erase)
    assert erase.tolist() == [True, False, False]  # without the no-splash patch a dry raining cell keeps its credits


def test_identity_guard_accepts_closure_and_refuses_a_violation():
    nc = 2
    ledger = np.zeros((3, len(D.LEDGER_COLUMNS), nc))
    i = {name: j for j, name in enumerate(D.LEDGER_COLUMNS)}
    ledger[:, i["pickup_kg"]] = 1.0
    ledger[:, i["deposition_active_kg"]] = 0.75
    ledger[:, i["old_mobile_kg"]] = [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5]]
    ledger[:, i["new_mobile_kg"]] = [[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]]
    residual, worst = D._check_identity(ledger)
    assert worst == 0.0 and not residual.any()
    ledger[1, i["new_mobile_kg"], 0] += 1e-3
    with pytest.raises(D.DriverError, match="identity"):
        D._check_identity(ledger)


def test_snapshot_time_parsing_is_strict():
    assert D._parse_times("", 10.0) == []
    assert D._parse_times("5, 2", 10.0) == [2.0, 5.0]
    for text in ("0", "11", "nan", "a", ",".join(str(k + 1) for k in range(17))):
        with pytest.raises(D.DriverError):
            D._parse_times(text, 100.0 if text.count(",") > 10 else 10.0)


def _check(net, det, dep, clip, m1, m2, q2, qin2, cum=None):
    n, nc = det.shape
    cum = [np.zeros((n, nc)) for _ in range(3)] if cum is None else cum
    return int(get_kernels(False).check(net.active_idx, net.outlet_idx, 1.0, det, dep, clip, m1, m2, q2, qin2, *cum,
                                        np.zeros((5, nc))))


def test_validation_pass_accepts_a_clean_step_and_flags_each_defect():
    net = N.native_network(channel(3))
    n, nc = net.active.size, 1
    z = np.zeros((n, nc))
    ok = np.full((n, nc), 0.5)
    assert _check(net, ok, ok, z, ok, ok, ok, ok) == 0
    for position, bit in ((0, 1), (1, 2), (2, 4)):  # detachment, deposition, clipping
        args = [ok.copy(), ok.copy(), z.copy()]
        args[position][1, 0] = np.nan
        assert _check(net, *args, ok, ok, ok, ok) & bit
    nan_pool = ok.copy()
    nan_pool[1, 0] = np.nan  # NaN passes the legacy `m < 0` clip test, so only this pass can refuse it
    assert _check(net, ok, ok, z, ok, nan_pool, ok, ok) & 8
    neg_flux = ok.copy()
    neg_flux[2, 0] = -1.0
    assert _check(net, ok, ok, z, ok, ok, neg_flux, ok) & 8
    assert _check(net, ok, ok, z, nan_pool, ok, ok, ok) & 16
    huge = [np.full((n, nc), 1.7e308) for _ in range(3)]
    assert _check(net, huge[0], ok, z, ok, ok, ok, ok, cum=huge) & 32  # 1.7e308 + 1.7e308 overflows the cumulative map
    big = np.full((n, nc), 1.0e308)
    assert _check(net, z, z, z, z, big, z, z) & 64  # finite cells whose class sum overflows


def test_extreme_finite_states_are_refused_not_laundered_by_the_zero_clip():
    net = N.native_network(channel(3))
    n, nc = net.active.size, 1
    z = np.zeros((n, nc))
    v = np.full((n, nc), 0.3)
    # a finite state whose Crank-Nicolson right-hand side overflows to +Infinity
    big = z.copy()
    big[1, 0] = 1.7e308
    with np.errstate(all="ignore"):
        M2, Q2, Qin2, clip, *_ = _cn(net, z, z, v, big, z, big)
    assert not np.isfinite(M2).all()
    assert _check(net, z, z, clip, big, M2, Q2, Qin2) & 8
    # a finite state whose right-hand side overflows to -Infinity: the clip would turn it into an Infinity "source"
    dep = z.copy()
    dep[1, 0] = 1.7e308
    flux = z.copy()
    flux[1, 0] = 1.7e308
    with np.errstate(all="ignore"):
        M2, Q2, Qin2, clip, *_ = _cn(net, z, dep, v, z, flux, z)
    assert M2[1, 0] == 0.0 and not np.isfinite(clip[1, 0])
    assert _check(net, z, dep, clip, z, M2, Q2, Qin2) & 4


def test_regime_table_covers_walk_limits():
    assert N.walk_limits(1.0).size == max(REGIME_CODES.values()) + 1
