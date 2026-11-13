"""G3 S2: every-N soil-moisture save of the floating daily lineage through the qualified G1 writer (CPU/host). Real lineages, real files; no fake parent."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from dbuild import day, lineage

from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_output as fo
from maple_syrup import floating_moisture_frame as g1
from maple_syrup import soil_moisture_output as g0

LIM = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**7, max_manifest_bytes=8192, chunk_bytes=4096, host_memory_budget_bytes=10**7, host_reserve_bytes=0, disk_reserve_bytes=0)
DISK = lambda: 10**12
HOST = lambda: 10**12




@pytest.fixture
def out_dir(tmp_path):
    (tmp_path / "out").mkdir()
    return tmp_path / "out"


def new_output(out_dir, st, n=2, **kw):
    return fo.FloatingMoistureOutput.create(out_dir, st, expected_parent=fd.expect_parent(st), every_n_days=n, limits=LIM, host_available_bytes_fn=HOST, disk_available_bytes_fn=DISK, **kw)


def close(st, output=None):
    inter = day(st).state
    return fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), output=output, host_available_bytes_fn=HOST, disk_available_bytes_fn=DISK)


def frames(out_dir):
    return sorted(p.name for p in Path(out_dir).iterdir() if p.name.startswith("frame_"))


@pytest.mark.parametrize("bad", [0, -1, True, False, 2.0, "2", None, float("nan"), 1 << 40])
def test_n_must_be_a_strict_positive_integer_and_a_bad_create_does_no_io(out_dir, bad):
    st, *_ = lineage()
    with pytest.raises(g0.MoistureOutputError):
        fo.FloatingMoistureOutput.create(out_dir, st, expected_parent=fd.expect_parent(st), every_n_days=bad, limits=LIM)
    assert list(out_dir.iterdir()) == []


def test_off_by_absence_adds_no_array_traffic(monkeypatch):
    st, *_ = lineage()
    calls = []
    monkeypatch.setattr(g1, "write_floating_frame", lambda *a, **k: calls.append(1))
    res = close(st)
    assert res.notice is None and res.state.cursor.ordinal == 1 and calls == []


def test_every_two_days_over_four_days_saves_exactly_days_2_and_4_from_the_accepted_state(out_dir):
    st, *_ = lineage(saturation=0.6)
    out = new_output(out_dir, st, 2, origin_date="2026-01-01", caller_identity={"scenario": "unit"})
    notices, states = [], [st]
    for d in range(4):
        r = close(states[-1], out)
        notices.append(r.notice)
        states.append(r.state)
    assert [n.action for n in notices] == ["not_due", "saved", "not_due", "saved"] and [n.day for n in notices] == [1, 2, 3, 4]
    assert frames(out_dir) == ["frame_000000000002", "frame_000000000004"]  # the storm/physical time never creates days; only completed nominal days count
    for day_, state in ((2, states[2]), (4, states[4])):
        fr = g1.read_floating_frame(out_dir / g1.frame_name(day_), max_return_bytes=10**7)
        assert np.array_equal(fr.water_m, state.water.water_m) and np.array_equal(fr.depth_m, state.ctx.depth_m) and np.array_equal(fr.active, state.ctx.active)  # post-T W, canonical D, mask
        m = fr.manifest
        assert m["day"] == day_ and m["cadence_n"] == 2 and m["origin_ordinal"] == 0 and m["physical_t_s"] == state.t_s and m["date"] == f"2026-01-{day_ + 1:02d}"
        ident = state.frame_identity()
        assert m["identity"] == {k: ident[k] for k in g1._ID_FIELDS} and m["caller_identity"] == {"scenario": "unit"}  # identities DERIVED from the accepted state
        assert np.array_equal(fr.depth_edges_m, state.ctx.edges_m) and fr.cell_area_m2 == state.ctx.area
    assert states[2].frame_identity()["lineage_sha256"] != states[4].frame_identity()["lineage_sha256"]


def test_not_due_days_touch_no_array(out_dir, monkeypatch):
    st, *_ = lineage()
    out = new_output(out_dir, st, 3)
    calls = []
    real = g1.write_floating_frame
    monkeypatch.setattr(g1, "write_floating_frame", lambda *a, **k: calls.append(1) or real(*a, **k))
    s1 = close(st, out).state
    s2 = close(s1, out).state
    assert calls == [] and frames(out_dir) == []
    close(s2, out)
    assert calls == [1] and len(frames(out_dir)) == 1


def test_output_failure_keeps_the_accepted_parent_retryable_and_a_sealed_day_is_verified_not_overwritten(out_dir):
    st, *_ = lineage()
    out = new_output(out_dir, st, 1)
    inter = day(st).state
    with pytest.raises(g0.MoistureOutputQuotaError):
        fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), output=out, host_available_bytes_fn=HOST, disk_available_bytes_fn=lambda: 0)
    assert frames(out_dir) == [] and out.last_day == 0 and inter.role == "interevent" and inter.cursor.ordinal == 0  # the accepted interevent parent is retained
    ok = fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), output=out, host_available_bytes_fn=HOST, disk_available_bytes_fn=DISK)
    assert ok.notice.action == "saved"
    sealed = {p.name: p.read_bytes() for p in (out_dir / g1.frame_name(1)).iterdir()}
    again = fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), output=out, host_available_bytes_fn=HOST, disk_available_bytes_fn=DISK)
    assert again.notice.action == "verified_existing"  # the same due day re-proved against the live bytes
    assert {p.name: p.read_bytes() for p in (out_dir / g1.frame_name(1)).iterdir()} == sealed  # never overwritten


def test_binding_unknown_files_quota_and_wrong_lineage(out_dir, tmp_path):
    st, *_ = lineage()
    (out_dir / "foreign.txt").write_text("keep")
    with pytest.raises(g0.MoistureOutputError, match="empty"):
        new_output(out_dir, st)
    (out_dir / "foreign.txt").unlink()
    other, *_ = lineage(run="cd" * 32)
    with pytest.raises(fd.DailyRefusal):
        fo.FloatingMoistureOutput.create(out_dir, st, expected_parent=fd.expect_parent(other), every_n_days=1, limits=LIM)  # wrong parent: refused before any I/O
    assert list(out_dir.iterdir()) == []
    out = new_output(out_dir, st, 1)
    (out_dir / "foreign.txt").write_text("keep")
    s1 = close(st, out).state
    assert (out_dir / "foreign.txt").read_text() == "keep"
    inter_other = day(other).state
    other_closed = fd.close_floating_day(inter_other, expected_parent=fd.expect_parent(inter_other)).state
    with pytest.raises(g0.MoistureOutputError, match="run/source"):
        out.notify_day_closed(other_closed, expected_parent=fd.expect_parent(other_closed))  # a different lineage (other run digest/source) is not this output's
    inter = day(s1).state
    tiny = g0.MoistureOutputLimits(max_snapshots=1, max_total_bytes=10**7, max_manifest_bytes=8192, chunk_bytes=4096, host_memory_budget_bytes=10**7, host_reserve_bytes=0, disk_reserve_bytes=0)
    out.__dict__["limits"] = tiny
    with pytest.raises(g0.MoistureOutputQuotaError):
        fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), output=out, host_available_bytes_fn=HOST, disk_available_bytes_fn=DISK)  # the second snapshot exceeds max_snapshots=1
    assert frames(out_dir) == ["frame_000000000001"]


def test_non_closed_state_and_non_output_are_refused_before_io(out_dir):
    st, *_ = lineage()
    out = new_output(out_dir, st, 1)
    inter = day(st).state
    with pytest.raises(fd.DailyRefusal):
        out.notify_day_closed(inter, expected_parent=fd.expect_parent(inter))  # an interevent is not a closed day
    with pytest.raises(g0.MoistureOutputError):
        out.notify_day_closed(st, expected_parent=fd.expect_parent(st))  # the origin is not a completed day
    with pytest.raises(fd.DailyRefusal, match="FloatingMoistureOutput"):
        fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), output=object())
    assert frames(out_dir) == []


def test_resume_at_a_lagging_boundary_replays_sealed_frames_and_cannot_change_n_source_or_origin(out_dir, tmp_path):
    st, *_ = lineage()
    out = new_output(out_dir, st, 2, caller_identity={"k": "v"})
    states = [st]
    for _ in range(4):
        states.append(close(states[-1], out).state)
    before = {p.name: (p / "manifest.json").read_bytes() for p in out_dir.iterdir() if p.name.startswith("frame_")}
    # the physical restart point lags the output (day 2 state, frames of days 2 and 4 are sealed)
    res = fo.FloatingMoistureOutput.resume(out_dir, states[2], expected_parent=fd.expect_parent(states[2]), limits=LIM, caller_identity={"k": "v"}, host_available_bytes_fn=HOST)
    assert res.every_n_days == 2 and res.origin_ordinal == 0 and res.last_day == 2
    i3 = day(states[2]).state
    n3 = fd.close_floating_day(i3, expected_parent=fd.expect_parent(i3), output=res, host_available_bytes_fn=HOST, disk_available_bytes_fn=DISK)
    assert n3.notice.action == "not_due"
    s3 = n3.state
    inter = day(s3).state
    n4 = fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), output=res, host_available_bytes_fn=HOST, disk_available_bytes_fn=DISK)
    assert n4.notice.action == "verified_existing"  # the replay matches the sealed bytes exactly
    assert {p.name: (p / "manifest.json").read_bytes() for p in out_dir.iterdir() if p.name.startswith("frame_")} == before
    with pytest.raises(g0.MoistureOutputError, match="caller identity"):
        fo.FloatingMoistureOutput.resume(out_dir, states[2], expected_parent=fd.expect_parent(states[2]), limits=LIM, caller_identity={"k": "other"}, host_available_bytes_fn=HOST)
    other, *_ = lineage(run="cd" * 32)
    with pytest.raises(g0.MoistureOutputError, match="binding"):
        fo.FloatingMoistureOutput.resume(out_dir, other, expected_parent=fd.expect_parent(other), limits=LIM, caller_identity={"k": "v"}, host_available_bytes_fn=HOST)
    # a missing sealed due frame cannot be recreated
    import shutil

    shutil.rmtree(out_dir / g1.frame_name(2))
    with pytest.raises(g0.MoistureOutputError, match="missing"):
        fo.FloatingMoistureOutput.resume(out_dir, states[4], expected_parent=fd.expect_parent(states[4]), limits=LIM, caller_identity={"k": "v"}, host_available_bytes_fn=HOST)


def test_source_wrong_writer_module_is_refused_on_resume(out_dir):
    st, *_ = lineage()
    new_output(out_dir, st, 1)
    rec = out_dir / fo.OUTPUT_NAME
    data = g0._strict_loads(rec.read_bytes())
    data["module_sha256"] = "0" * 64
    data["body_sha256"] = g0._body_sha(data)
    os.chmod(rec, 0o600)
    rec.write_bytes(g0._canon(data))
    with pytest.raises(g0.MoistureOutputError, match="different writer module"):
        fo.FloatingMoistureOutput.resume(out_dir, st, expected_parent=fd.expect_parent(st), limits=LIM, host_available_bytes_fn=HOST)
