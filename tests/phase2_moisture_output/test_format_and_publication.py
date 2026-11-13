"""Snapshot format, exact reconstruction, untrusted-input validation, quotas before transfer/write, no-clobber, failure injection, restart/lag/duplicate/conflict. Author-written, native daily fixtures."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import GRID, Day, build, host, layers, limits, make, mo, notify, start, wl


def run_days(c, tmp_path, n, days, **kw):
    (tmp_path / "out").mkdir(exist_ok=True)
    out, s0 = make(c, tmp_path, n, **kw)
    states = [s0]
    for k in range(days):
        _, s = Day(c).close(states[-1], p=(2.0, 0.0, 1.0)[k % 3])
        states.append(s)
        notify(out, c, s)
    return out, states


def day_dir(tmp_path, d):
    return tmp_path / "out" / mo.snapshot_name(d)


# ---- exact reconstruction and format ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_exact_reconstruction_units_and_npy_compat(case, tmp_path):
    out, st = run_days(case, tmp_path, 2, 2)
    snap = mo.read_snapshot(day_dir(tmp_path, 2), max_bytes=1 << 20)
    live = host(st[2].payload.soil.water_m)
    assert snap.water_m.tobytes() == live.tobytes() and snap.water_m.shape == (case.L, case.n) and not snap.water_m.flags.writeable
    assert np.array_equal(np.load(day_dir(tmp_path, 2) / "water_m.npy"), live)  # a genuine .npy
    assert np.array_equal(np.load(tmp_path / "out" / "active.npy"), host(case.col.active))
    dz = np.asarray(case.col.layers.thickness_m)
    assert np.array_equal(snap.dz_m, dz) and np.array_equal(snap.theta(), live / dz[:, None])
    assert np.array_equal(snap.volume_m3(), live * case.col.cell_area_m2)
    m = snap.manifest
    assert m["profile"] == "fixed_full_porous_common_column" and "floating" in snap.static["profile_note"] and m["not_a_restart"] is True
    assert m["water"]["units"] == "m water-equivalent depth" and m["water"]["order"] == "C" and m["water"]["dtype"] == "<f8"
    ident = m["identity"]
    s = st[2]
    assert (ident["state_digest"], ident["state_revision"], ident["lop_sha256"], ident["request_sequence"], ident["episode"]) == (s.digest, s.revision, s.lop_sha256, s.request_sequence, s.episode)
    assert m["run_digest"] == s.daily.run_digest and ident["cursor_digest"] == s.daily.digest() and m["day"] == 2 and m["ordinal"] == 2
    assert snap.static["every_n_days"] == 2 and snap.static["n_cells"] == case.n and snap.static["n_layers"] == case.L


def test_manifest_is_strict_canonical_json_with_body_checksum(case, tmp_path):
    run_days(case, tmp_path, 1, 1)
    raw = (day_dir(tmp_path, 1) / "manifest.json").read_bytes()
    obj = json.loads(raw)
    assert raw == json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    body = {k: v for k, v in obj.items() if k != "body_sha256"}
    assert obj["body_sha256"] == hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("chunk", [8, 24, 64, 1000, 1 << 20])
def test_chunked_staging_gives_identical_bytes(case, tmp_path, chunk):
    outs = []
    for tag, cb in (("a", chunk), ("b", 1 << 20)):
        d = tmp_path / tag
        d.mkdir()
        (d / "out").mkdir()
        out, s0 = make(case, d, 1, limits=limits(chunk_bytes=cb))
        _, s1 = Day(case).close(s0, p=2.0)
        notify(out, case, s1)
        outs.append((d / "out" / mo.snapshot_name(1) / "water_m.npy").read_bytes())
    assert outs[0] == outs[1]


def test_untrusted_snapshot_validation(case, tmp_path):
    run_days(case, tmp_path, 1, 1)
    d = day_dir(tmp_path, 1)
    good = mo.read_snapshot(d, max_bytes=1 << 20)
    with pytest.raises(mo.MoistureOutputError, match="max_bytes"):
        mo.read_snapshot(d, max_bytes=8)
    w = d / "water_m.npy"
    raw = bytearray(w.read_bytes())
    raw[-1] ^= 1
    w.write_bytes(bytes(raw))
    with pytest.raises(mo.MoistureOutputError, match="checksum"):
        mo.read_snapshot(d, max_bytes=1 << 20)
    raw[-1] ^= 1
    w.write_bytes(bytes(raw) + b"\0")
    with pytest.raises(mo.MoistureOutputError):
        mo.read_snapshot(d, max_bytes=1 << 20)
    w.write_bytes(bytes(raw)[:-8])
    with pytest.raises(mo.MoistureOutputError):
        mo.read_snapshot(d, max_bytes=1 << 20)
    w.write_bytes(bytes(raw))
    assert mo.read_snapshot(d, max_bytes=1 << 20).water_m.tobytes() == good.water_m.tobytes()
    m = d / "manifest.json"
    orig = m.read_bytes()
    for bad in (orig.replace(b'"day":1', b'"day":3'), orig[:-5], orig.replace(b'"physical_t_s"', b'"physical_t_s":NaN,"x"'), b"{}", b"[]", b"\xff"):
        m.write_bytes(bad)
        with pytest.raises(mo.MoistureOutputError):
            mo.read_snapshot(d, max_bytes=1 << 20)
    m.write_bytes(orig)
    (d / "extra").write_bytes(b"x")
    with pytest.raises(mo.MoistureOutputError, match="unexpected entries"):
        mo.read_snapshot(d, max_bytes=1 << 20)
    os.unlink(d / "extra")
    rec = tmp_path / "out" / "output.json"
    ro = rec.read_bytes()
    rec.write_bytes(ro.replace(b'"every_n_days":1', b'"every_n_days":2'))
    with pytest.raises(mo.MoistureOutputError, match="checksum"):
        mo.read_snapshot(d, max_bytes=1 << 20)
    rec.write_bytes(ro)


def test_symlinks_and_non_regular_files_are_refused(case, tmp_path):
    run_days(case, tmp_path, 1, 1)
    d = day_dir(tmp_path, 1)
    link = tmp_path / "out" / "day_000000000009"
    os.symlink(d, link)
    with pytest.raises(mo.MoistureOutputError):
        mo.read_snapshot(link, max_bytes=1 << 20)
    link.unlink()
    w = d / "water_m.npy"
    keep = w.read_bytes()
    w.unlink()
    os.symlink(d / "manifest.json", w)
    with pytest.raises(mo.MoistureOutputError):
        mo.read_snapshot(d, max_bytes=1 << 20)
    w.unlink()
    w.write_bytes(keep)
    outlink = tmp_path / "linkroot"
    os.symlink(tmp_path / "out", outlink)
    with pytest.raises(mo.MoistureOutputError):
        mo.read_output_record(outlink)  # the root descriptor never follows a symlink


# ---- quotas: before any transfer or write ---------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw,match", [(dict(max_snapshots=1), "max_snapshots"), (dict(max_total_bytes=3000), "max_total_bytes|static record"), (dict(max_manifest_bytes=256), "max_manifest_bytes")])
def test_quota_refusals_precede_transfers_and_writes(case, tmp_path, monkeypatch, kw, match):
    (tmp_path / "out").mkdir()
    lim = limits(**kw)
    try:
        out, s0 = make(case, tmp_path, 1, limits=lim)
    except mo.MoistureOutputQuotaError:
        assert "max_total_bytes" in kw  # the static record alone already exceeded it
        return
    D = Day(case)
    _, s1 = D.close(s0, p=2.0)
    if "max_snapshots" in kw:
        notify(out, case, s1)  # the first fits (max 1)
        _, s2 = D.close(s1, p=0.0)
        s1 = s2
    calls = []
    monkeypatch.setattr(mo, "_to_host", lambda a: calls.append(1) or a)
    before = sorted(p.name for p in (tmp_path / "out").iterdir())
    with pytest.raises(mo.MoistureOutputQuotaError, match=match):
        notify(out, case, s1)
    assert not calls and sorted(p.name for p in (tmp_path / "out").iterdir()) == before


def test_total_bytes_quota_exact_boundary(case, tmp_path):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    static = out.used
    _, s1 = Day(case).close(s0, p=2.0)
    note = notify(out, case, s1)
    one = out.used - static
    assert one == note.bytes_written - 0 or abs(one - note.bytes_written) < 600  # header + 8LN + manifest
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "out").mkdir()
    o2, t0 = make(case, tmp_path / "t", 1, limits=limits(max_total_bytes=static + one))
    _, t1 = Day(case).close(t0, p=2.0)
    notify(o2, case, t1)  # exactly enough
    _, t2 = Day(case).close(t1, p=2.0)
    with pytest.raises(mo.MoistureOutputQuotaError, match="max_total_bytes"):
        notify(o2, case, t2)


def test_host_and_disk_readers_and_reserves_before_transfer(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1, limits=limits(host_reserve_bytes=100, disk_reserve_bytes=100))
    _, s1 = Day(case).close(s0, p=2.0)
    calls = []
    monkeypatch.setattr(mo, "_to_host", lambda a: calls.append(1) or a)
    staging = out.limits.staging_bytes()  # chunk + SCRATCH_FACTOR x manifest bound + 64 KiB + reserve (100)
    with pytest.raises(mo.MoistureOutputQuotaError, match="host bytes"):
        notify(out, case, s1, host_available_bytes_fn=lambda: staging - 1)
    for bad in (lambda: True, lambda: -1, lambda: 1.5):
        with pytest.raises(mo.MoistureOutputError):
            notify(out, case, s1, host_available_bytes_fn=bad)
    with pytest.raises(mo.MoistureOutputQuotaError, match="disk bytes"):
        notify(out, case, s1, host_available_bytes_fn=lambda: 1 << 40, disk_available_bytes_fn=lambda: 100)
    assert not calls and out.last_day == 0 and not [p for p in (tmp_path / "out").iterdir() if p.name.startswith("day_")]
    monkeypatch.undo()
    assert notify(out, case, s1, host_available_bytes_fn=lambda: staging, disk_available_bytes_fn=lambda: 1 << 40).action == "saved"


# ---- failure injection: nothing published, nothing advanced ------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("where", ["stream", "seal", "manifest"])
def test_failed_save_leaves_no_snapshot_and_no_advance_and_retry_works(case, tmp_path, monkeypatch, where):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1, limits=limits(chunk_bytes=8))
    _, s1 = Day(case).close(s0, p=2.0)
    before_state = (s1.digest, host(s1.payload.soil.water_m).tobytes())
    if where == "stream":
        real, count = mo._to_host, []

        def boom(a):
            count.append(1)
            if len(count) == 3:
                raise OSError("disk went away")
            return real(a)

        monkeypatch.setattr(mo, "_to_host", boom)
    elif where == "seal":
        monkeypatch.setattr(mo, "_seal_link", lambda *a, **k: (_ for _ in ()).throw(OSError("link failed")))
    else:
        monkeypatch.setattr(mo, "_write_new", lambda *a, **k: (_ for _ in ()).throw(OSError("no space")))
    with pytest.raises(OSError):
        notify(out, case, s1)
    monkeypatch.undo()
    assert not [p for p in (tmp_path / "out").iterdir() if p.name.startswith("day_")]  # the reservation was cleaned: nothing published
    assert out.last_day == 0 and out.saved == 0 and (s1.digest, host(s1.payload.soil.water_m).tobytes()) == before_state
    assert notify(out, case, s1).action == "saved" and out.last_day == 1  # the same day retries cleanly


def test_no_clobber_existing_snapshot_is_verified_never_overwritten(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 2)
    d = day_dir(tmp_path, 2)
    before = (d / "water_m.npy").read_bytes(), (d / "manifest.json").read_bytes()
    # a fresh writer (restart) replays day 2 from the lagging state of day 1
    out2 = mo.MoistureOutput.resume(tmp_path / "out", case.lop, st[1], expected_parent=wl.expect_parent(st[1]), limits=limits())
    note = notify(out2, case, st[2])
    assert note.action == "verified_existing" and note.bytes_written == 0
    assert (d / "water_m.npy").read_bytes() == before[0] and (d / "manifest.json").read_bytes() == before[1]


def test_unsealed_reservation_is_never_adopted_and_resume_removes_it(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 1)
    crash = day_dir(tmp_path, 2)
    crash.mkdir()
    (crash / "water_m.npy").write_bytes(b"partial")
    (crash / "manifest.json.tmp").write_bytes(b"{")
    _, s2 = Day(case).close(st[1], p=1.0)
    with pytest.raises(mo.MoistureOutputError, match="unsealed"):
        notify(out, case, s2)  # a live writer never adopts or overwrites a reservation it did not make
    out2 = mo.MoistureOutput.resume(tmp_path / "out", case.lop, st[1], expected_parent=wl.expect_parent(st[1]), limits=limits())
    assert not crash.exists()  # the crashed reservation was removed by resume
    assert notify(out2, case, s2).action == "saved"


def test_conflicting_state_for_a_saved_day_is_refused(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 2)
    out2 = mo.MoistureOutput.resume(tmp_path / "out", case.lop, st[1], expected_parent=wl.expect_parent(st[1]), limits=limits())
    _, alt = Day(case).close(st[1], p=5.0)  # a different branch of day 2
    before = (day_dir(tmp_path, 2) / "water_m.npy").read_bytes()
    with pytest.raises(mo.MoistureOutputConflict):
        notify(out2, case, alt)
    assert (day_dir(tmp_path, 2) / "water_m.npy").read_bytes() == before and out2.last_day == 1


# ---- restart continuity ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_resume_lagging_state_replays_and_continues(case, tmp_path):
    out, st = run_days(case, tmp_path, 2, 5)  # saved days 2 and 4; the physical checkpoint of day 3 is older than the output
    out2 = mo.MoistureOutput.resume(tmp_path / "out", case.lop, st[3], expected_parent=wl.expect_parent(st[3]), limits=limits())
    assert out2.last_day == 3 and out2.saved == 2
    assert notify(out2, case, st[4]).action == "verified_existing"  # day 4 was already saved before the crash: verified, not rewritten
    assert notify(out2, case, st[5]).action == "not_due"
    _, s6 = Day(case).close(st[5], p=0.0)
    assert notify(out2, case, s6).action == "saved" and out2.verify_all() == 3


def test_resume_refuses_a_missed_due_day_and_foreign_bindings(case, tmp_path):
    out, st = run_days(case, tmp_path, 2, 4)
    import shutil

    shutil.rmtree(day_dir(tmp_path, 2))
    with pytest.raises(mo.MoistureOutputError, match="missing a due snapshot"):
        mo.MoistureOutput.resume(tmp_path / "out", case.lop, st[4], expected_parent=wl.expect_parent(st[4]), limits=limits())
    other = build(GRID, lay=layers(bottom="closed"), theta=(0.3, 0.3), backend=case.lop.backend)
    o0 = start(other)
    _, o1 = Day(other).close(o0, p=0.0)
    with pytest.raises(mo.MoistureOutputError):
        mo.MoistureOutput.resume(tmp_path / "out", other.lop, o1, expected_parent=wl.expect_parent(o1), limits=limits())


def test_resume_scan_refuses_corrupt_and_unknown_entries(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 2)
    (tmp_path / "out" / "stray.txt").write_text("x")
    with pytest.raises(mo.MoistureOutputError, match="unexpected entry"):
        mo.MoistureOutput.resume(tmp_path / "out", case.lop, st[2], expected_parent=wl.expect_parent(st[2]), limits=limits())
    os.unlink(tmp_path / "out" / "stray.txt")
    (day_dir(tmp_path, 1) / "water_m.npy").write_bytes(b"short")
    with pytest.raises(mo.MoistureOutputError, match="regular file of the expected size"):
        mo.MoistureOutput.resume(tmp_path / "out", case.lop, st[2], expected_parent=wl.expect_parent(st[2]), limits=limits())


def test_create_needs_an_empty_directory_and_never_replaces_a_record(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 1)
    with pytest.raises(mo.MoistureOutputError, match="empty"):
        mo.MoistureOutput.create(tmp_path / "out", case.lop, st[0], expected_parent=wl.expect_parent(st[0]), every_n_days=1, limits=limits())
    with pytest.raises(mo.MoistureOutputError):
        mo.MoistureOutput.create(tmp_path / "missing", case.lop, st[0], expected_parent=wl.expect_parent(st[0]), every_n_days=1, limits=limits())


def test_directory_swap_is_detected(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 1)
    (tmp_path / "out").rename(tmp_path / "moved")
    (tmp_path / "out").mkdir()
    _, s2 = Day(case).close(st[1], p=1.0)
    with pytest.raises(mo.MoistureOutputError):
        notify(out, case, s2)  # a different directory than the bound one (and an empty one): refused


# ---- device-transfer accounting with a CuPy-free double ---------------------------------------------------------------------------------------------------------------------------------------
def test_transfers_are_bounded_pieces_only_when_due(monkeypatch):
    """`_stream` downloads one bounded contiguous piece per call and never the whole array (the cadence layer makes zero calls on a not-due day: see the next test)."""
    from types import SimpleNamespace

    class Dev:
        def __init__(self, a):
            self.a, self.shape = a, a.shape

        def __getitem__(self, idx):
            return Dev(self.a[idx])

    seen = []
    monkeypatch.setattr(mo, "_to_host", lambda x: seen.append(int(x.a.nbytes)) or x.a)
    arr = np.arange(3 * 100, dtype=np.float64).reshape(3, 100)
    got = bytearray()
    sink = SimpleNamespace(write=lambda b: got.extend(bytes(b)))
    mo.MoistureOutput._stream(SimpleNamespace(limits=limits(chunk_bytes=8 * 32)), Dev(arr), b"", 3, 100, sink)
    assert bytes(got) == arr.tobytes() and max(seen) <= 8 * 32 and len(seen) == 3 * 4 and sum(seen) == arr.nbytes


def test_not_due_days_make_zero_transfers(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 3)
    calls = []
    monkeypatch.setattr(mo, "_to_host", lambda a: calls.append(1) or a)
    s = s0
    for _ in range(2):
        _, s = Day(case).close(s, p=0.0)
        assert notify(out, case, s).action == "not_due"
    assert not calls
    _, s = Day(case).close(s, p=0.0)
    assert notify(out, case, s).action == "saved" and len(calls) == case.L


def test_current_device_gate_runs_before_any_reader_or_copy(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    _, s1 = Day(case).close(s0, p=2.0)
    reads = []
    monkeypatch.setattr(wl, "_check_lop", lambda lop, **k: (_ for _ in ()).throw(wl.LifecycleError("the current device is not the owner's")))
    with pytest.raises(mo.MoistureOutputError, match="current device"):
        notify(out, case, s1, host_available_bytes_fn=lambda: reads.append(1) or (1 << 40), disk_available_bytes_fn=lambda: reads.append(1) or (1 << 40))
    assert not reads
