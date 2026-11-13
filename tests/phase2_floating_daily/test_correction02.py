"""G3 S1/S2 correction02: final-close admission before any query, and the genuine daily-window + lifetime audit (CPU/host; NumPy and Numba)."""
from __future__ import annotations

import tracemalloc

import builders as b
import numpy as np
import pytest
import texture_fixtures as fx
from dbuild import AVAIL, close, day, lineage, make, start

from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_output as fo
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import soil_moisture_output as g0
from maple_syrup import subsurface as ss

BACKENDS = ["numpy"] + (["numba"] if __import__("importlib").util.find_spec("numba") is not None else [])


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def spies(monkeypatch):
    calls = {"storage": 0, "values": 0}
    real_s, real_v = fsl.floating_storage_m3, fsl._check_water_values
    monkeypatch.setattr(fsl, "floating_storage_m3", lambda *a, **k: calls.__setitem__("storage", calls["storage"] + 1) or real_s(*a, **k))
    monkeypatch.setattr(fsl, "_check_water_values", lambda *a, **k: calls.__setitem__("values", calls["values"] + 1) or real_v(*a, **k))
    return calls


# ---- admission precedes every private value check, array temporary and native query --------------------------------------------------------------
def test_zero_reader_and_huge_reserve_refuse_before_any_query_mint_or_io(backend, monkeypatch, tmp_path):
    st, *_ = lineage(backend=backend)
    inter = day(st).state
    calls = spies(monkeypatch)
    reads = []
    for fn, reserve in ((lambda: reads.append(1) or 0, 0), (lambda: reads.append(1) or 10**9, 10**9), (lambda: reads.append(1) or 10**12, 10**13)):
        with pytest.raises(fd.DailyRefusal) as ex:
            fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), available_bytes_fn=fn, reserve_bytes=reserve)
        assert ex.value.code == "memory"
    assert len(reads) == 3 and calls == {"storage": 0, "values": 0}  # the REAL reader was consulted; no value check or native query ran
    assert inter.role == "interevent" and inter.cursor.ordinal == 0
    # with an output: refused before any I/O; the output is unchanged and the retry works
    lim = g0.MoistureOutputLimits(10, 10**7, 8192, 4096, 10**7, 0, 0)
    (tmp_path / "o").mkdir()
    out = fo.FloatingMoistureOutput.create(tmp_path / "o", st, expected_parent=fd.expect_parent(st), every_n_days=1, limits=lim, host_available_bytes_fn=AVAIL, disk_available_bytes_fn=AVAIL)
    with pytest.raises(fd.DailyRefusal):
        fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), output=out, available_bytes_fn=lambda: 0, host_available_bytes_fn=AVAIL, disk_available_bytes_fn=AVAIL)
    assert out.last_day == 0 and not list((tmp_path / "o").glob("frame_*"))
    ok = fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), output=out, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL, disk_available_bytes_fn=AVAIL)
    assert ok.notice.action == "saved" and ok.state.cursor.ordinal == 1


@pytest.mark.parametrize("layers", [(0.16,), (0.01, 0.05, 0.10), tuple([0.16 / 65] * 65)])
def test_exact_need_need_minus_one_and_strict_metadata(backend, layers, monkeypatch):
    st, *_ = lineage(backend=backend, layers=layers, saturation=0.3)  # the default stratified bed has a shallow (partial/rock) cell
    inter = day(st).state
    need = fd._audit_need(inter.ctx, 0)
    assert need >= 8 * inter.ctx.n_cells * fd.AUDIT_DOUBLES_PER_CELL // 1
    calls = spies(monkeypatch)
    with pytest.raises(fd.DailyRefusal):
        fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), available_bytes_fn=lambda: need - 1)
    assert calls == {"storage": 0, "values": 0}
    res = fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), available_bytes_fn=lambda: need)  # the exact declared need
    assert res.state.cursor.ordinal == 1 and calls["storage"] == 1 and calls["values"] == 1
    need9 = fd._audit_need(inter.ctx, 9)
    assert need9 == need + 9
    with pytest.raises(fd.DailyRefusal):
        fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), available_bytes_fn=lambda: need9 - 1, reserve_bytes=9)
    for bad_reader in (lambda: True, lambda: -1, lambda: 1.5, lambda: "9", lambda: None, 5, "x"):
        with pytest.raises(fd.DailyRefusal):
            fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), available_bytes_fn=bad_reader)
    for bad_reserve in (-1, True, 1.5, "0"):
        with pytest.raises(fd.DailyRefusal):
            fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), available_bytes_fn=AVAIL, reserve_bytes=bad_reserve)


def test_default_real_host_reader_is_used_when_none_is_given(backend, monkeypatch):
    st, *_ = lineage(backend=backend)
    inter = day(st).state
    seen = []
    monkeypatch.setattr(ss, "default_host_available_bytes", lambda: seen.append(1) or 10**12)
    assert fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter)).state.cursor.ordinal == 1 and seen == [1]
    monkeypatch.setattr(ss, "default_host_available_bytes", lambda: 0)
    with pytest.raises(fd.DailyRefusal):
        fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter))


def test_modelled_audit_peak_bounds_the_traced_allocations_at_n4096(backend):
    bed = b.bed_from(fx.stratified_dense(shallow_cells=((0, 2), (3, 7)), ny=64, nx=64))
    owner, _ = make(backend=backend, bed=bed)
    st = start(owner, bed)
    inter = day(st).state
    need = fd._audit_need(inter.ctx, 0)
    fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), available_bytes_fn=lambda: need)  # warm
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = tracemalloc.get_traced_memory()[0]
    fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), available_bytes_fn=lambda: need)
    peak = tracemalloc.get_traced_memory()[1] - t0
    tracemalloc.stop()
    assert peak <= need, (peak, need)
    assert peak >= 8 * inter.ctx.n_cells  # non-vacuous: at least one float row


# ---- the genuine daily window + lifetime audit ------------------------------------------------------------------------------------------------------
def test_window_fields_come_from_the_native_budget_and_the_withdrawal_extends_only_the_window(backend):
    st, *_ = lineage(backend=backend, saturation=0.6)
    inter = day(st, p=1.0, i=2.0, t=5.0).state
    led = inter.ledger
    assert led.day_storage_before_m3 > 0.0 and led.day_bound_m3 > 0.0 and led.day_n_terms > 0 and led.last_day_rain_m3 > 0.0
    req = np.zeros((inter.ctx.n_layers, inter.ctx.n_cells))
    req[0] = 1e-5
    after = fd.request_floating_withdrawal(inter, expected_parent=fd.expect_parent(inter), request_m=req, expected_sequence=0, available_bytes_fn=AVAIL).state
    l2 = after.ledger
    assert l2.day_bound_m3 > led.day_bound_m3 and l2.day_n_terms > led.day_n_terms and l2.last_day_transpiration_m3 > 0.0 and l2.day_storage_before_m3 == led.day_storage_before_m3
    assert close(after).state.cursor.ordinal == 1


def test_a_long_history_never_loosens_the_current_days_bound(backend):
    st, *_ = lineage(backend=backend, saturation=0.6)
    for _ in range(30):
        st = close(day(st).state).state
    inter = day(st).state
    led = inter.ledger
    assert led.bound_m3 > 12 * led.day_bound_m3  # the lifetime bound is far larger than one day's
    area = inter.ctx.area
    delta_m3 = 8.0 * led.day_bound_m3  # above the day's own bound, far below the accumulated lifetime bound
    assert delta_m3 < 0.5 * led.bound_m3
    w = inter.water.water_m
    saved = np.array(w, copy=True)
    w.flags.writeable = True
    w[0, 0] += delta_m3 / area
    try:
        with pytest.raises(fd.DailyRefusal, match="daily-window"):  # the window audit catches what the lifetime bound alone would accept
            close(inter)
        assert inter.cursor.ordinal == 30
    finally:
        w[...] = saved
    assert close(inter).state.cursor.ordinal == 31


def test_lifetime_audit_still_catches_an_error_below_the_window_bound_but_accumulated_across_days(backend):
    """Corrupt the LEDGER (an earlier day's accounting), not the water: the window agrees but the lifetime closure does not."""
    import dataclasses

    st, *_ = lineage(backend=backend, saturation=0.6)
    for _ in range(3):
        st = close(day(st).state).state
    inter = day(st).state
    bad = dataclasses.replace(inter.ledger, evaporation_m3=inter.ledger.evaporation_m3 + 1e-6)
    object.__setattr__(inter, "ledger", bad)
    with pytest.raises(fd.DailyRefusal, match="lifetime"):
        close(inter)
    for field, value in (("day_bound_m3", float("nan")), ("bound_m3", -1.0), ("rain_m3", float("inf"))):
        object.__setattr__(inter, "ledger", dataclasses.replace(inter.ledger if field != "x" else bad, **{field: value}))
        with pytest.raises(fd.DailyRefusal):
            close(inter)


def test_long_sequence_with_rain_withdrawals_and_output_closes_every_day_and_adds_no_extra_audit_work(backend, tmp_path, monkeypatch):
    st, *_ = lineage(backend=backend, saturation=0.5)
    lim = g0.MoistureOutputLimits(100, 10**8, 8192, 4096, 10**8, 0, 0)
    (tmp_path / "o").mkdir()
    out = fo.FloatingMoistureOutput.create(tmp_path / "o", st, expected_parent=fd.expect_parent(st), every_n_days=7, limits=lim, host_available_bytes_fn=AVAIL, disk_available_bytes_fn=AVAIL)
    calls = spies(monkeypatch)
    n = st.ctx.n_cells
    for d in range(21):
        r = day(st, p=1.5 if d % 5 == 1 else 0.0, i=3.0, t=5.0)
        inter = r.state
        assert r.outcome == "interevent"
        if d % 4 == 0:
            req = np.zeros((st.ctx.n_layers, n))
            req[1] = 5e-6
            inter = fd.request_floating_withdrawal(inter, expected_parent=fd.expect_parent(inter), request_m=req, expected_sequence=0, available_bytes_fn=AVAIL).state
        before = dict(calls)
        res = fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), output=out, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL, disk_available_bytes_fn=AVAIL)
        assert calls["storage"] - before["storage"] == 1 and calls["values"] - before["values"] == 1  # ONE audited query per close, with or without a due output
        st = res.state
    assert st.cursor.ordinal == 21 and st.ledger.days_applied == 21 and len(list((tmp_path / "o").glob("frame_*"))) == 3
