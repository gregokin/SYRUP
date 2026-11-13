"""G2a: the daily final-close/output facade on the ACTUAL native daily lifecycle (genuine Dense bed through the Phase 1.5 helpers)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from daily_helpers import (
    DAY,
    case,
    close,
    dmo,
    dw,
    fp,
    host,
    interevent,
    limits,
    mo,
    start,
    storm_interevent,
    wl,
)

HERE = Path(__file__).resolve().parent
PS = (2.0, 0.0, 1.0, 2.0)


@pytest.fixture
def c():
    return case()


def make(c, tmp_path, n=2, **kw):
    (tmp_path / "out").mkdir()
    s0 = start(c)
    return dmo.create_daily_moisture_output(tmp_path / "out", c.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=n, limits=kw.pop("limits", limits()), **kw), s0


def frames(tmp_path):
    return sorted(p.name for p in (tmp_path / "out").iterdir() if p.name.startswith("day_"))


def test_off_by_absence_closes_exactly_like_the_native_call_and_touches_nothing(c, monkeypatch):
    s0 = start(c)
    ie = interevent(c, s0, 2.0)
    monkeypatch.setattr(mo, "_to_host", lambda *a: (_ for _ in ()).throw(AssertionError("transfer without an output")))
    r = close(c, ie)
    ref = wl.close_interevent(c.lop, ie, expected_parent=wl.expect_parent(ie))
    assert r.notice is None and fp(c, r.state) == fp(c, ref) and r.state.role == "boundary"


def test_n2_saves_days_2_and_4_after_actual_withdrawals_and_matches_the_direct_control(c, tmp_path):
    out, s0 = make(c, tmp_path, 2)
    ctrl = s0
    s = s0
    notices = []
    for day, p in enumerate(PS, 1):
        w = 1e-5 if day in (1, 3) else None  # actual request_withdrawal on days 1 and 3; day 2/4 have no plants at all
        ie = interevent(c, s, p, withdraw=w)
        ie_ctrl = interevent(c, ctrl, p, withdraw=w)
        r = close(c, ie, output=out)
        ctrl = wl.close_interevent(c.lop, ie_ctrl, expected_parent=wl.expect_parent(ie_ctrl))
        s = r.state
        notices.append((r.notice.day, r.notice.action, r.notice.due))
        # saving changes nothing physical: bitwise the same accepted state as the direct native control
        assert fp(c, s)[1:] == fp(c, ctrl)[1:] and s.digest == ctrl.digest
    assert notices == [(1, "not_due", False), (2, "saved", True), (3, "not_due", False), (4, "saved", True)]
    assert frames(tmp_path) == [mo.snapshot_name(2), mo.snapshot_name(4)]
    snap = mo.read_snapshot(tmp_path / "out" / mo.snapshot_name(4), max_bytes=1 << 20)
    assert snap.water_m.tobytes() == host(wl._arrays_of(s).soil.water_m).tobytes()  # the FINAL post-T water of day 4
    assert snap.manifest["day"] == 4 and snap.manifest["nominal_elapsed_s"] == 4 * DAY and snap.manifest["not_a_restart"] is True


def test_saved_water_is_the_post_withdrawal_inventory(c, tmp_path):
    out, s0 = make(c, tmp_path, 1)
    ie = interevent(c, s0, 2.0, withdraw=5e-4)
    pre = host(wl._arrays_of(ie).soil.water_m).copy()
    r = close(c, ie, output=out)
    snap = mo.read_snapshot(tmp_path / "out" / mo.snapshot_name(1), max_bytes=1 << 20)
    assert snap.water_m.tobytes() == host(wl._arrays_of(r.state).soil.water_m).tobytes()
    pre_withdrawal = interevent(c, s0, 2.0)
    assert snap.water_m.tobytes() != host(wl._arrays_of(pre_withdrawal).soil.water_m).tobytes() and pre.size


def test_storm_duration_is_additional_and_does_not_count_as_nominal_days(c, tmp_path):
    out, s0 = make(c, tmp_path, 2)
    ie = storm_interevent(c, s0, 3.0)  # day 1 is a real storm day
    r1 = close(c, ie, output=out)
    assert r1.notice.day == 1 and r1.notice.action == "not_due" and int(r1.state.daily.ordinal) == 1
    assert r1.state.t_s > DAY  # the storm/recession physical duration is ADDITIONAL to the nominal day
    r2 = close(c, interevent(c, r1.state, 0.0), output=out)
    assert r2.notice.day == 2 and r2.notice.action == "saved"
    man = json.loads((tmp_path / "out" / mo.snapshot_name(2) / "manifest.json").read_text())
    assert man["nominal_elapsed_s"] == 2 * DAY and man["physical_t_s"] > 2 * DAY


def test_wrong_inputs_refused_before_any_model_call(c, tmp_path, monkeypatch):
    out, s0 = make(c, tmp_path, 1)
    monkeypatch.setattr(wl, "close_interevent", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model call before refusal")))
    with pytest.raises(mo.MoistureOutputError, match="interevent"):
        close(c, s0, output=out)  # a closed boundary is not an interevent
    plan = dw.next_plan(s0, 3.0, 360.0, 1.0)
    st = dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, control=__import__("daily_helpers").STORM_CONTROL)
    with pytest.raises(mo.MoistureOutputError, match="interevent"):
        close(c, st, output=out)  # a storm (pending occurrence) is not final
    monkeypatch.undo()
    plain = wl.enter_interevent_from_boundary(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0))
    with pytest.raises(mo.MoistureOutputError, match="DailyCursor"):
        close(c, plain, output=out)  # not a daily lineage: no cursor, no nominal day
    assert frames(tmp_path) == []


def test_disk_refusal_leaves_the_parent_unchanged_then_retry_saves_and_matches(c, tmp_path):
    out, s0 = make(c, tmp_path, 1)
    ie = interevent(c, s0, 2.0)
    before = fp(c, ie)
    with pytest.raises(mo.MoistureOutputQuotaError):
        close(c, ie, output=out, disk_available_bytes_fn=lambda: 5)
    assert fp(c, ie) == before and frames(tmp_path) == []  # the accepted parent is untouched; the I/O failure is not model completion
    r = close(c, ie, output=out)  # retry with the SAME parent
    assert r.notice.action == "saved" and frames(tmp_path) == [mo.snapshot_name(1)]
    ctrl = wl.close_interevent(c.lop, ie, expected_parent=wl.expect_parent(ie))
    assert r.state.digest == ctrl.digest


def test_snapshot_quota_failure_does_not_advance_cadence_or_clock(c, tmp_path):
    out, s0 = make(c, tmp_path, 1, limits=limits(max_snapshots=1))
    r1 = close(c, interevent(c, s0, 2.0), output=out)
    ie2 = interevent(c, r1.state, 0.0)
    before = fp(c, ie2)
    with pytest.raises(mo.MoistureOutputQuotaError):
        close(c, ie2, output=out)
    assert fp(c, ie2) == before and out.last_day == 1 and frames(tmp_path) == [mo.snapshot_name(1)]


def test_duplicate_calls_follow_g0_semantics(c, tmp_path):
    out, s0 = make(c, tmp_path, 1)
    ie = interevent(c, s0, 2.0)
    first = close(c, ie, output=out)
    again = close(c, ie, output=out)  # a due day: re-proved against the sealed bytes, never rewritten
    assert first.notice.action == "saved" and again.notice.action == "verified_existing" and frames(tmp_path) == [mo.snapshot_name(1)]
    (tmp_path / "second").mkdir()
    out2, s0b = make(c, tmp_path / "second", 2)
    ie1 = interevent(c, s0b, 2.0)
    close(c, ie1, output=out2)  # day 1 of N = 2: not due
    with pytest.raises(mo.MoistureOutputConflict):
        close(c, ie1, output=out2)  # a NON-due duplicate with a new closed instance is refused: reuse the returned state


def test_resume_binds_a_lagging_boundary_and_continues(c, tmp_path):
    out, s0 = make(c, tmp_path, 2)
    s = s0
    for p in PS[:3]:
        s = close(c, interevent(c, s, p), output=out).state
    out2 = dmo.resume_daily_moisture_output(tmp_path / "out", c.lop, s, expected_parent=wl.expect_parent(s), limits=limits())
    assert out2.last_day == 3
    r = close(c, interevent(c, s, PS[3]), output=out2)
    assert r.notice.action == "saved" and frames(tmp_path) == [mo.snapshot_name(2), mo.snapshot_name(4)]


def test_expected_parent_is_required_caller_authority_and_wrong_ones_are_refused_before_any_work(c, tmp_path, monkeypatch):
    import maple_syrup.daily_water as dwm
    import maple_syrup.interevent_transpiration as it

    out, s0 = make(c, tmp_path, 1)
    ie = interevent(c, s0, 2.0)
    other_branch = interevent(c, s0, 2.0)  # an alternate branch: equal scalars, a DIFFERENT instance
    stale = wl.expect_parent(s0)
    with pytest.raises(TypeError):
        dmo.close_final_day(c.lop, ie, output=out)  # the capability is REQUIRED (never re-minted for the state argument)
    boom = AssertionError("model audit or I/O before the capability check")
    monkeypatch.setattr(it, "audit_general_interevent", lambda *a, **k: (_ for _ in ()).throw(boom))
    monkeypatch.setattr(mo, "_to_host", lambda *a: (_ for _ in ()).throw(boom))
    for bad in (wl.expect_parent(other_branch), stale, object(), None):
        with pytest.raises((wl.LifecycleError, mo.MoistureOutputError)):
            dmo.close_final_day(c.lop, ie, expected_parent=bad, output=out)
    assert frames(tmp_path) == [] and dwm is not None
    monkeypatch.undo()
    assert close(c, ie, output=out).notice.action == "saved"  # the right capability works


def test_output_must_be_none_or_an_exact_moisture_output_before_closing(c, tmp_path, monkeypatch):
    out, s0 = make(c, tmp_path, 1)
    ie = interevent(c, s0, 2.0)

    class Sub(mo.MoistureOutput):
        pass

    class Duck:
        def notify_day_closed(self, *a, **k):
            raise AssertionError("duck-typed callback used")

    monkeypatch.setattr(wl, "close_interevent", lambda *a, **k: (_ for _ in ()).throw(AssertionError("closed before the output check")))
    for bad in (Duck(), object(), lambda *a, **k: None, "out"):
        with pytest.raises(mo.MoistureOutputError, match="MoistureOutput"):
            dmo.close_final_day(c.lop, ie, expected_parent=wl.expect_parent(ie), output=bad)
    assert Sub is not mo.MoistureOutput and out is not None


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, None])
def test_cadence_is_explicit_and_validated(c, tmp_path, bad):
    (tmp_path / "out").mkdir()
    s0 = start(c)
    ep = wl.expect_parent(s0)
    with pytest.raises(mo.MoistureOutputError):
        dmo.create_daily_moisture_output(tmp_path / "out", c.lop, s0, expected_parent=ep, every_n_days=bad, limits=limits())
    with pytest.raises(TypeError):
        dmo.create_daily_moisture_output(tmp_path / "out", c.lop, s0, expected_parent=ep, limits=limits())
    with pytest.raises(TypeError):
        dmo.create_daily_moisture_output(tmp_path / "out", c.lop, s0, every_n_days=2, limits=limits())  # the capability is required


@pytest.mark.skipif(sys.platform != "linux", reason="fresh-process fixtures are Linux")
def test_fresh_process_v3_checkpoint_continuation_matches_the_continuous_run_and_verifies_sealed_future_frames(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    run = lambda mode: subprocess.run([sys.executable, "-B", str(HERE / "fresh_worker.py"), mode, str(tmp_path)], capture_output=True,
                                      text=True, env=env, cwd=str(HERE), timeout=900, check=False)
    a = run("A")
    assert a.returncode == 0, a.stderr[-3000:]
    ja = json.loads(a.stdout.strip().splitlines()[-1])
    assert ja["last_day"] == 4
    before = {p.name: (p / "manifest.json").read_bytes() for p in (tmp_path / "out").iterdir() if p.name.startswith("day_")}
    b = run("B")
    assert b.returncode == 0, b.stderr[-3000:]
    jb = json.loads(b.stdout.strip().splitlines()[-1])
    assert jb["actions"] == ["not_due", "verified_existing"]  # day 4 was already sealed AHEAD: verified, never rewritten
    assert jb["prints"] == ja["prints"]  # physical continuation from the V3 day-2 checkpoint equals the continuous run
    assert before == {p.name: (p / "manifest.json").read_bytes() for p in (tmp_path / "out").iterdir() if p.name.startswith("day_")}
