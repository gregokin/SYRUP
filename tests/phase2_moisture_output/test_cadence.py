"""Cadence semantics on the ACTUAL native daily lifecycle (NumPy and Numba columns): what is a completed day, sequential notification, post-withdrawal capture, no mutation. Author-written."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import BIGQ, GRID, RUN, Day, act, build, host, layers, limits, make, mo, notify, start, wl
from ncases import fingerprint


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.0, 2.5, float("nan"), float("inf"), "2", None, [2], np.float64(2.0)])
def test_invalid_cadence_refused_and_nothing_created(case, tmp_path, bad):
    s0 = start(case)
    (tmp_path / "out").mkdir()
    with pytest.raises(mo.MoistureOutputError):
        mo.MoistureOutput.create(tmp_path / "out", case.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=bad, limits=limits())
    assert list((tmp_path / "out").iterdir()) == []


def test_there_is_no_default_cadence_and_no_default_limits(case, tmp_path):
    s0 = start(case)
    (tmp_path / "out").mkdir()
    with pytest.raises(TypeError):
        mo.MoistureOutput.create(tmp_path / "out", case.lop, s0, expected_parent=wl.expect_parent(s0), limits=limits())
    with pytest.raises(TypeError):
        mo.MoistureOutputLimits(max_snapshots=1)
    with pytest.raises(mo.MoistureOutputError):
        mo.MoistureOutput(object())


@pytest.mark.parametrize("n,expected", [(1, [1, 2, 3, 4, 5, 6]), (2, [2, 4, 6]), (3, [3, 6])])
def test_saves_at_n_2n_3n_completed_days(case, tmp_path, n, expected):
    (tmp_path / "out").mkdir()
    out, s = make(case, tmp_path, n)
    saved = []
    for day in range(1, 7):
        _, s = Day(case).close(s, p=2.0 if day % 2 else 0.0)
        note = notify(out, case, s)
        assert note.day == day and note.due == (day in expected) and note.device_chunks == 0
        if note.action == "saved":
            saved.append(day)
        else:
            assert note.action == "not_due" and note.path is None and note.bytes_written == 0
    assert saved == expected
    assert sorted(p.name for p in (tmp_path / "out").iterdir() if p.name.startswith("day_")) == [mo.snapshot_name(d) for d in expected]
    assert out.verify_all() == len(expected)


def test_origin_is_day_zero_and_nothing_is_saved_for_it(case, tmp_path):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    with pytest.raises(mo.MoistureOutputError):  # the origin boundary is not the close of an interevent: not a completed day
        notify(out, case, s0)
    assert not [p for p in (tmp_path / "out").iterdir() if p.name.startswith("day_")]


def test_the_state_before_t_and_close_is_not_final(case, tmp_path):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    _, r = act(case, s0, 2.0)
    for bad in (r.state,):  # the interevent state right after daily_water_action
        with pytest.raises(mo.MoistureOutputError, match="boundary"):
            notify(out, case, bad)
    q = np.zeros((case.L, case.n))
    q[0] = 2e-4
    s3, _ = wl.request_withdrawal(case.lop, r.state, expected_parent=wl.expect_parent(r.state), depth_m=q, identity="leaf")
    with pytest.raises(mo.MoistureOutputError, match="boundary"):
        notify(out, case, s3)  # still an interevent after T: only the close is final
    assert not [p for p in (tmp_path / "out").iterdir() if p.name.startswith("day_")] and out.last_day == 0


def test_post_withdrawal_water_is_what_is_saved(case, tmp_path):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    ie, closed = Day(case).close(s0, p=2.0, withdraw=3e-4)
    pre = np.asarray(host(ie.payload.soil.water_m)).copy()
    note = notify(out, case, closed)
    snap = mo.read_snapshot(note.path, max_bytes=1 << 20)
    assert snap.water_m.tobytes() == host(closed.payload.soil.water_m).tobytes()
    ie2, _ = Day(case).close(s0, p=2.0)  # the same day WITHOUT the withdrawal holds more water in layer 0
    assert host(ie2.payload.soil.water_m)[0].sum() > snap.water_m[0].sum() + 1e-9
    assert pre.tobytes() == snap.water_m.tobytes()  # the closed boundary shares the post-withdrawal arrays of its interevent


def test_physical_duration_adds_no_days_a_storm_day_is_one_day(case, tmp_path):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    _, s1 = Day(case).close(s0, p=0.0)
    assert notify(out, case, s1).day == 1
    ie, s2 = Day(case).storm_day(s1)
    assert s2.t_s - s1.t_s > 86400.0 + 1.0  # the physical storm made this "day" longer than 86400 s
    note = notify(out, case, s2)
    assert note.day == 2 and note.ordinal == 2
    man = mo.read_snapshot(note.path, max_bytes=1 << 20).manifest
    assert man["nominal_elapsed_s"] == 2 * 86400.0 and man["physical_t_s"] == s2.t_s and man["physical_t_s"] > man["nominal_elapsed_s"]
    assert man["identity"]["last_branch"] == "storm"


def test_dates_follow_nominal_days_only(case, tmp_path):
    (tmp_path / "out").mkdir()
    out, s = make(case, tmp_path, 2, origin_date="2026-02-27")
    for day in (1, 2):
        _, s = Day(case).close(s, p=0.0)
        note = notify(out, case, s)
    assert mo.read_snapshot(note.path, max_bytes=1 << 20).manifest["date"] == "2026-03-01"
    (tmp_path / "x").mkdir()
    with pytest.raises(mo.MoistureOutputError):
        make(case, tmp_path / "x", 2, origin_date="2026-13-01")


def test_sequence_skips_rewinds_and_foreign_branches_are_refused(case, tmp_path):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 2)
    D = Day(case)
    _, s1 = D.close(s0, p=0.0)
    _, s2 = D.close(s1, p=0.0)
    with pytest.raises(mo.MoistureOutputError, match="sequentially"):
        notify(out, case, s2)  # day 1 skipped
    assert out.last_day == 0
    notify(out, case, s1)
    _, s2b = D.close(s1, p=5.0)  # an alternate branch of day 2
    note = notify(out, case, s2)
    assert note.action == "saved"
    with pytest.raises(mo.MoistureOutputConflict):
        notify(out, case, s2b)  # same day, different instance
    assert notify(out, case, s2).action == "repeat_same_state"  # same instance: idempotent
    with pytest.raises(mo.MoistureOutputError, match="sequentially"):
        notify(out, case, s1)  # rewound


def test_idempotent_repeat_does_not_transfer_or_write(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    _, s1 = Day(case).close(s0, p=0.0)
    notify(out, case, s1)
    calls = []
    monkeypatch.setattr(mo, "_to_host", lambda a: calls.append(1))
    before = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert notify(out, case, s1).action == "repeat_same_state" and not calls and sorted(p.name for p in (tmp_path / "out").iterdir()) == before


def test_foreign_lineages_operators_and_states_are_refused(case, tmp_path):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    other = build(GRID, lay=layers(bottom="closed"), theta=(0.3, 0.3), backend=case.lop.backend)  # different sealed content => different operator sha
    o0 = start(other)
    _, o1 = Day(other).close(o0, p=0.0)
    with pytest.raises(mo.MoistureOutputError):
        out.notify_day_closed(case.lop, o1, expected_parent=wl.expect_parent(o1))  # state of another lifecycle
    with pytest.raises(mo.MoistureOutputError):
        out.notify_day_closed(other.lop, o1, expected_parent=wl.expect_parent(o1))  # operator of another lifecycle
    other2 = start(case, run="cd" * 32)
    _, f1 = Day(case).close(other2, p=0.0)
    with pytest.raises(mo.MoistureOutputError, match="lineage"):
        notify(out, case, f1)  # same operator, different daily run
    _, c1 = Day(case).close(s0, p=0.0)
    with pytest.raises(mo.MoistureOutputError):
        out.notify_day_closed(case.lop, c1, expected_parent=wl.expect_parent(s0))  # wrong expected parent
    with pytest.raises(mo.MoistureOutputError):
        out.notify_day_closed(object(), c1, expected_parent=wl.expect_parent(c1))
    with pytest.raises(mo.MoistureOutputError):
        notify(out, case, case.s0)  # a Phase 1 (non-daily) boundary
    assert out.last_day == 0


def test_no_physical_mutation(case, tmp_path):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    _, s1 = Day(case).close(s0, p=2.0, withdraw=2e-4)
    before = fingerprint(s1)
    cursor, soil_id = s1.daily, id(s1.payload.soil.water_m)
    notify(out, case, s1)
    assert fingerprint(s1) == before and s1.daily is cursor and id(s1.payload.soil.water_m) == soil_id
    _, s2 = Day(case).close(s1, p=1.0)  # the lineage continues from the very same state after a save
    assert s2.daily.ordinal == 2


def test_inactive_cells_are_preserved_exactly(tmp_path):
    active = np.ones((4, 6), dtype=bool)
    active[1, 2] = active[3, 5] = False
    c = build(GRID, lay=layers(bottom="open"), theta=(0.3, 0.3), active=active)
    (tmp_path / "out").mkdir()
    out, s0 = make(c, tmp_path, 1)
    _, s1 = Day(c).close(s0, p=2.0)
    snap = mo.read_snapshot(notify(out, c, s1).path, max_bytes=1 << 20)
    assert snap.water_m.tobytes() == host(s1.payload.soil.water_m).tobytes()
    assert snap.active.tolist() == host(c.col.active).tolist() and snap.static["n_active"] == 22 and not snap.active.all()
