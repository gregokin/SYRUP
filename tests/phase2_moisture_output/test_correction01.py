"""G0 correction01: the independently confirmed defects (saved-file corruption, post-seal failure, create/reader memory and quotas, strict JSON/validation, restart idempotence incl. a FRESH-PROCESS physical restart,
no directory enumeration). Author-written; native daily fixtures (NumPy and Numba columns where cheap)."""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import GRID, Day, build, host, layers, limits, make, mo, notify, start, wl

HERE = Path(__file__).resolve().parent


def run_days(c, tmp_path, n, days, **kw):
    (tmp_path / "out").mkdir(exist_ok=True)
    out, s0 = make(c, tmp_path, n, **kw)
    states = [s0]
    for k in range(days):
        _, s = Day(c).close(states[-1], p=(2.0, 0.0, 1.0)[k % 3])
        states.append(s)
        notify(out, c, s)
    return out, states


def dd(tmp_path, d):
    return tmp_path / "out" / mo.snapshot_name(d)


def resume(c, tmp_path, state, **kw):
    return mo.MoistureOutput.resume(tmp_path / "out", c.lop, state, expected_parent=wl.expect_parent(state), limits=kw.pop("limits", limits()), **kw)


# ---- 1. equal-length corruption of a SAVED array ---------------------------------------------------------------------------------------------------------------------------------------------------
def flip_last_byte(path):
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))


def test_resume_refuses_an_equal_length_corrupt_saved_array(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 2)
    flip_last_byte(dd(tmp_path, 2) / "water_m.npy")  # same size, one bit wrong
    with pytest.raises(mo.MoistureOutputError, match="checksum"):
        resume(case, tmp_path, st[1])
    with pytest.raises(mo.MoistureOutputError, match="checksum"):
        out.verify_all()


def test_notify_never_reports_verified_existing_over_a_corrupt_saved_array(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 2)
    out2 = resume(case, tmp_path, st[1])  # a clean resume first (day 2 is sealed and intact)
    flip_last_byte(dd(tmp_path, 2) / "water_m.npy")  # then the stored bytes rot
    before = (dd(tmp_path, 2) / "manifest.json").read_bytes()
    with pytest.raises(mo.MoistureOutputError, match="checksum"):
        notify(out2, case, st[2])
    assert out2.last_day == 1 and (dd(tmp_path, 2) / "manifest.json").read_bytes() == before  # no adoption, no advance, nothing rewritten


def test_header_size_and_filetype_of_saved_files_are_proved(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 1)
    w = dd(tmp_path, 1) / "water_m.npy"
    keep = w.read_bytes()
    bad = bytearray(keep)
    bad[20] ^= 0x20  # inside the .npy header, same length
    w.write_bytes(bytes(bad))
    with pytest.raises(mo.MoistureOutputError, match="header|checksum"):
        out.verify_all()
    w.write_bytes(keep + b"x")
    with pytest.raises(mo.MoistureOutputError):
        out.verify_all()
    w.unlink()
    os.symlink(dd(tmp_path, 1) / "manifest.json", w)  # a symlink is refused by the scan (not only by read_snapshot)
    with pytest.raises(mo.MoistureOutputError):
        out.verify_all()
    w.unlink()
    w.write_bytes(keep)
    assert out.verify_all() == 1


def test_streamed_proofs_use_one_bounded_buffer(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 1, limits=limits(chunk_bytes=16))
    assert out.verify_all() == 1  # a 16-byte buffer: the header region spans several pieces and the proof still holds
    flip_last_byte(dd(tmp_path, 1) / "water_m.npy")
    with pytest.raises(mo.MoistureOutputError, match="checksum"):
        out.verify_all()


# ---- 2. post-seal failures never destroy sealed data -----------------------------------------------------------------------------------------------------------------------------------------
def _dir_fsync_fails(monkeypatch, after_calls=0):
    real, state = os.fsync, {"n": 0}

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            state["n"] += 1
            if state["n"] > after_calls:
                raise OSError("injected parent-directory fsync failure")
        return real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    return state


def _tree(tmp_path, d):
    return {p.name: p.read_bytes() for p in dd(tmp_path, d).iterdir()}


def test_parent_fsync_failure_after_the_seal_keeps_payload_and_seal_then_retry_advances_once(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    _, s1 = Day(case).close(s0, p=2.0)
    live = host(s1.payload.soil.water_m).tobytes()
    with monkeypatch.context() as mp:
        _dir_fsync_fails(mp)
        with pytest.raises(mo.MoistureOutputIndeterminate) as e:
            notify(out, case, s1)
    assert e.value.path.endswith(mo.snapshot_name(1))
    tree = _tree(tmp_path, 1)
    assert "water_m.npy" in tree and "manifest.json" in tree and out.last_day == 0 and out.saved == 1  # published, nothing deleted, cadence NOT advanced; the sealed snapshot IS counted (correction02)
    assert np.frombuffer(tree["water_m.npy"][len(mo._npy_header((case.L, case.n), "<f8")):], dtype=np.float64).tobytes() == live
    note = notify(out, case, s1)  # the retry verifies the sealed identity and advances exactly once, rewriting nothing
    assert note.action == "verified_existing" and note.bytes_written == 0 and out.last_day == 1
    assert {k: v for k, v in _tree(tmp_path, 1).items() if k != "manifest.json.tmp"} == {k: v for k, v in tree.items() if k != "manifest.json.tmp"}
    assert notify(out, case, s1).action == "repeat_same_state"


def test_seal_temp_unlink_failure_leaves_a_tidyable_hard_link_never_a_deletion(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    _, s1 = Day(case).close(s0, p=2.0)
    real = os.unlink

    def unlink(path, *a, **k):
        if path == "manifest.json.tmp":
            raise OSError("injected unlink failure")
        return real(path, *a, **k)

    with monkeypatch.context() as mp:
        mp.setattr(os, "unlink", unlink)
        with pytest.raises(mo.MoistureOutputIndeterminate):
            notify(out, case, s1)
    names = set(_tree(tmp_path, 1))
    assert names == {"water_m.npy", "manifest.json", "manifest.json.tmp"} and out.last_day == 0
    assert mo.read_snapshot(dd(tmp_path, 1), max_bytes=1 << 20).water_m.tobytes() == host(s1.payload.soil.water_m).tobytes()  # the reader tolerates the sealed manifest's own link
    assert notify(out, case, s1).action == "verified_existing"
    assert set(_tree(tmp_path, 1)) == {"water_m.npy", "manifest.json"}  # only the hard link of the sealed manifest was tidied


def test_failure_after_the_link_itself_is_still_published_not_discarded(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    _, s1 = Day(case).close(s0, p=2.0)
    real = os.link

    def link_then_die(*a, **k):
        real(*a, **k)
        raise OSError("died right after the link")

    with monkeypatch.context() as mp:
        mp.setattr(os, "link", link_then_die)
        with pytest.raises(mo.MoistureOutputIndeterminate):
            notify(out, case, s1)
    assert {"water_m.npy", "manifest.json"} <= set(_tree(tmp_path, 1)) and out.last_day == 0
    assert notify(out, case, s1).action == "verified_existing"


def test_resume_after_a_post_seal_crash_verifies_and_never_loses_the_snapshot(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    _, s1 = Day(case).close(s0, p=2.0)
    with monkeypatch.context() as mp:
        _dir_fsync_fails(mp)
        with pytest.raises(mo.MoistureOutputIndeterminate):
            notify(out, case, s1)
    out2 = resume(case, tmp_path, s0)  # the process "restarted" from the day-0 checkpoint
    assert out2.saved == 1 and notify(out2, case, s1).action == "verified_existing"


def test_unsealed_cleanup_removes_only_own_files(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 1)
    crash = dd(tmp_path, 2)
    crash.mkdir()
    (crash / "water_m.npy").write_bytes(b"partial")
    (crash / "stranger.txt").write_text("not ours")  # an unknown member
    _, s2 = Day(case).close(st[1], p=1.0)
    with pytest.raises(mo.MoistureOutputError, match="unrecognised"):  # correction03: an unknown member proves it is not our reservation: refuse, delete NOTHING
        resume(case, tmp_path, st[1])
    assert (crash / "stranger.txt").exists() and (crash / "water_m.npy").read_bytes() == b"partial"


# ---- 3. create/reader memory and quotas -------------------------------------------------------------------------------------------------------------------------------------------------------------
def _fresh(case, tmp_path):
    (tmp_path / "out").mkdir()
    return start(case)


@pytest.mark.parametrize("kw,match", [(dict(max_total_bytes=1), "max_total_bytes"), (dict(host_memory_budget_bytes=1000), "host_memory_budget_bytes"), (dict(host_reserve_bytes=1 << 40), "host_memory_budget_bytes")])
def test_create_admits_every_quota_before_any_download_or_write(case, tmp_path, monkeypatch, kw, match):
    s0 = _fresh(case, tmp_path)
    calls = []
    monkeypatch.setattr(mo, "_to_host", lambda a: calls.append(1) or a)
    with pytest.raises(mo.MoistureOutputQuotaError, match=match):
        mo.MoistureOutput.create(tmp_path / "out", case.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=1, limits=limits(**kw))
    assert not calls and list((tmp_path / "out").iterdir()) == []


def test_create_reader_and_disk_refusals_precede_the_mask_download(case, tmp_path, monkeypatch):
    s0 = _fresh(case, tmp_path)
    calls = []
    monkeypatch.setattr(mo, "_to_host", lambda a: calls.append(1) or a)
    lim = limits(disk_reserve_bytes=10)
    for kw, match in ((dict(host_available_bytes_fn=lambda: 10), "host bytes"), (dict(host_available_bytes_fn=lambda: 1 << 40, disk_available_bytes_fn=lambda: 10), "disk bytes")):
        with pytest.raises(mo.MoistureOutputQuotaError, match=match):
            mo.MoistureOutput.create(tmp_path / "out", case.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=1, limits=lim, **kw)
    for bad in (lambda: -1, lambda: float("nan"), lambda: True, lambda: None):
        with pytest.raises(mo.MoistureOutputError):
            mo.MoistureOutput.create(tmp_path / "out", case.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=1, limits=lim, host_available_bytes_fn=bad)
    assert not calls and list((tmp_path / "out").iterdir()) == []


@pytest.mark.parametrize("field", ["max_snapshots", "max_total_bytes", "max_manifest_bytes", "chunk_bytes", "host_memory_budget_bytes", "host_reserve_bytes", "disk_reserve_bytes"])
@pytest.mark.parametrize("bad", [-1, True, 1.5, float("nan"), float("inf"), None, "9"])
def test_limits_refuse_negative_boolean_nonfinite_and_unknown_counts(field, bad):
    with pytest.raises(mo.MoistureOutputError):
        limits(**{field: bad})


def test_mask_is_streamed_in_bounded_pieces_and_failure_cleans_only_own_files(case, tmp_path, monkeypatch):
    s0 = _fresh(case, tmp_path)
    seen = []

    def fake(x):
        seen.append(int(np.asarray(x).nbytes))
        return np.asarray(x)

    monkeypatch.setattr(mo, "_to_host", fake)
    out = mo.MoistureOutput.create(tmp_path / "out", case.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=1, limits=limits(chunk_bytes=8))
    assert max(seen) <= 8 and len(seen) == -(-case.n // 8) and sum(seen) == case.n  # N/8 pieces of at most 8 bytes: never the whole mask
    assert np.array_equal(np.load(tmp_path / "out" / "active.npy"), host(case.col.active)) and out.rec["n_active"] == int(host(case.col.active).sum())
    shutil.rmtree(tmp_path / "out")
    (tmp_path / "out").mkdir()
    n = []

    def boom(x):
        n.append(1)
        if len(n) == 2:
            raise OSError("download failed")
        return np.asarray(x)

    monkeypatch.setattr(mo, "_to_host", boom)
    with pytest.raises(OSError):
        mo.MoistureOutput.create(tmp_path / "out", case.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=1, limits=limits(chunk_bytes=8))
    assert list((tmp_path / "out").iterdir()) == []  # only the files this call made were removed; the directory is reusable


class Dev:
    """CuPy-free device-array double: slicing is a view (no copy); `to_host` of it is a COPY (a real download)."""

    def __init__(self, a):
        self.a, self.shape = a, a.shape

    def __getitem__(self, idx):
        return Dev(self.a[idx])


def test_streaming_holds_exactly_one_host_piece(monkeypatch):
    monkeypatch.setattr(mo, "_to_host", lambda x: x.a.copy())
    L, N, chunk = 2, 200_000, 1 << 18
    water = Dev(np.ones((L, N)))
    stub = SimpleNamespace(limits=limits(chunk_bytes=chunk))
    head = mo._npy_header((L, N), "<f8")
    tracemalloc.start()
    mo.MoistureOutput._stream(stub, water, head, L, N, None)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert chunk <= peak < 1.5 * chunk  # one piece alive (two would be >= 2 * chunk); the whole 3.2 MB array is never mirrored
    sink = SimpleNamespace(write=lambda b: None)
    tracemalloc.start()
    sha, pieces = mo.MoistureOutput._stream(stub, water, head, L, N, sink)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert pieces == L * -(-N // (chunk // 8)) and peak < 1.5 * chunk


def test_reader_caps_all_returned_arrays_before_reading_any(case, tmp_path, monkeypatch):
    out, st = run_days(case, tmp_path, 1, 1)
    need = 8 * case.L * case.n + case.n + 8 * case.L
    called = []
    monkeypatch.setattr(mo, "_load_active", lambda *a: called.append("a"))
    monkeypatch.setattr(mo, "_load_water", lambda *a: called.append("w"))
    with pytest.raises(mo.MoistureOutputError, match="nothing was read"):
        mo.read_snapshot(dd(tmp_path, 1), max_bytes=need - 1)
    assert not called
    monkeypatch.undo()
    snap = mo.read_snapshot(dd(tmp_path, 1), max_bytes=need)  # exactly the declared arrays are admitted
    assert snap.water_m.shape == (case.L, case.n)


def _reseal(rec):
    rec = dict(rec)
    base = {k: v for k, v in rec.items() if k not in ("static_sha256", "body_sha256")}
    rec["static_sha256"] = __import__("hashlib").sha256(mo._canon(base)).hexdigest()
    rec["body_sha256"] = mo._body_sha(rec)
    return rec


def test_a_forged_huge_geometry_is_refused_before_any_allocation(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 1)
    rec = dict(json.loads((tmp_path / "out" / "output.json").read_bytes()))
    rec.update(n_cells=1 << 39, shape_yx=[1 << 20, 1 << 19])
    (tmp_path / "out" / "output.json").write_bytes(mo._canon(_reseal(rec)))
    tracemalloc.start()
    with pytest.raises(mo.MoistureOutputError, match="nothing was read"):
        mo.read_snapshot(dd(tmp_path, 1), max_bytes=1 << 20)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 1 << 20


# ---- 4. strict JSON and exact validation ---------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":1e999}', b'{"a":-1e999}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":' + b"9" * 41 + b"}", b"[" * 100000 + b"]" * 100000, b'{"a":"\xc3\xa9"}', b"\xff", b"{"])
def test_strict_json_refuses(raw):
    with pytest.raises(mo.MoistureOutputError):
        mo._strict_loads(raw)


def test_strict_json_accepts_plain_values():
    assert mo._strict_loads(b'{"a":[1,2.5,null,true,"x"]}') == {"a": [1, 2.5, None, True, "x"]}


def _good_record(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 1)
    return dict(json.loads((tmp_path / "out" / "output.json").read_bytes()))


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("dz_m"), lambda r: r.update(extra=1), lambda r: r.update(every_n_days=True), lambda r: r.update(every_n_days=0), lambda r: r.update(n_layers="2"), lambda r: r.update(dz_m=[0.25, -0.75]),
    lambda r: r.update(dz_m=[0.25]), lambda r: r.update(cell_area_m2=0.0), lambda r: r.update(cell_area_m2=True), lambda r: r.update(shape_yx=[3, 6]), lambda r: r.update(units={"water_m": "mm"}),
    lambda r: r.update(run_digest="xyz"), lambda r: r.update(origin_date="2026-13-40"), lambda r: r.update(n_active=10**9), lambda r: r.update(not_a_restart=False), lambda r: r.update(caller_identity={"a": 1}),
    lambda r: r.update(max_manifest_bytes=10), lambda r: r.update(profile="floating"), lambda r: r.update(schema="old")])
def test_output_record_validation_is_exact_and_typed(case, tmp_path, mutate):
    rec = _good_record(case, tmp_path)
    mutate(rec)
    (tmp_path / "out" / "output.json").write_bytes(mo._canon(_reseal(rec)))  # re-sealed: only the FIELD validation can refuse it
    with pytest.raises(mo.MoistureOutputError):
        mo.read_output_record(tmp_path / "out")


@pytest.mark.parametrize("mutate", [
    lambda m: m.pop("water"), lambda m: m.update(day=True), lambda m: m.update(day=3), lambda m: m.update(ordinal=99), lambda m: m.update(physical_t_s=-1.0), lambda m: m.update(nominal_elapsed_s=1.0),
    lambda m: m["water"].update(shape=[1, 1]), lambda m: m["water"].update(bytes=3), lambda m: m["water"].update(file_sha256="q"), lambda m: m["water"].pop("units"), lambda m: m["identity"].pop("state_digest"),
    lambda m: m["identity"].update(state_revision=-1), lambda m: m["identity"].update(lop_sha256="0" * 64), lambda m: m.update(date="2020-01-01"), lambda m: m.update(identity=[]), lambda m: m.update(static_sha256="0" * 64)])
def test_manifest_validation_is_exact_and_typed(case, tmp_path, mutate):
    run_days(case, tmp_path, 1, 1)
    p = dd(tmp_path, 1) / "manifest.json"
    man = json.loads(p.read_bytes())
    mutate(man)
    man["body_sha256"] = mo._body_sha(man)
    p.write_bytes(mo._canon(man))
    with pytest.raises(mo.MoistureOutputError):
        mo.read_snapshot(dd(tmp_path, 1), max_bytes=1 << 20)


# ---- 5. restart idempotence -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_same_state_right_after_resume_is_verified_or_repeated_never_a_conflict(case, tmp_path):
    out, st = run_days(case, tmp_path, 2, 3)  # day 2 saved; day 3 not due
    for k, action in ((2, "verified_existing"), (3, "repeat_same_state")):
        out2 = resume(case, tmp_path, st[k])
        note = notify(out2, case, st[k])
        assert note.action == action and note.bytes_written == 0 and out2.last_day == k
        assert notify(out2, case, st[k]).action == "repeat_same_state"
    _, alt = Day(case).close(st[1], p=5.0)  # an unknown/changed branch of day 2
    with pytest.raises(mo.MoistureOutputConflict):
        notify(resume(case, tmp_path, st[2]), case, alt)
    _, alt3 = Day(case).close(st[2], p=5.0)
    with pytest.raises(mo.MoistureOutputConflict):
        notify(resume(case, tmp_path, st[3]), case, alt3)  # different instance for a non-due, already-accepted day


def test_resume_refuses_a_changed_writer_module_or_caller_identity(case, tmp_path, monkeypatch):
    out, st = run_days(case, tmp_path, 1, 1, caller_identity={"runtime": "pair-A"})
    with pytest.raises(mo.MoistureOutputError, match="caller identity"):
        resume(case, tmp_path, st[1])
    with pytest.raises(mo.MoistureOutputError, match="caller identity"):
        resume(case, tmp_path, st[1], caller_identity={"runtime": "pair-B"})
    assert resume(case, tmp_path, st[1], caller_identity={"runtime": "pair-A"}).last_day == 1
    monkeypatch.setattr(mo, "_MODULE_SHA", "0" * 64)
    with pytest.raises(mo.MoistureOutputError, match="different writer module"):
        resume(case, tmp_path, st[1], caller_identity={"runtime": "pair-A"})
    for bad in ({"a": 1}, {1: "a"}, ["x"], {"k": "v" * 5000}):
        with pytest.raises(mo.MoistureOutputError):
            mo.MoistureOutput.create(tmp_path / "none", case.lop, st[0], expected_parent=wl.expect_parent(st[0]), every_n_days=1, limits=limits(), caller_identity=bad)


def test_phase_source_digest_is_recorded_but_not_a_lineage_binding(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 2)
    digs = {mo.read_snapshot(dd(tmp_path, d), max_bytes=1 << 20).manifest["identity"]["phase_source_digest"] for d in (1, 2)}
    assert len(digs) == 2  # the closing receipt's per-phase digest differs every day: informational only (the runtime source pair belongs to caller_identity)


def test_existence_check_and_due_path_enumerate_no_directory(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    _, s1 = Day(case).close(s0, p=2.0)
    monkeypatch.setattr(os, "listdir", lambda *a, **k: (_ for _ in ()).throw(AssertionError("os.listdir on the hot due path")))
    assert notify(out, case, s1).action == "saved"
    _, s2 = Day(case).close(s1, p=0.0)
    assert notify(out, case, s2).action == "saved"


def test_scan_is_streaming_with_scalar_counters(case, tmp_path):
    out, st = run_days(case, tmp_path, 1, 3)
    count, used, covered = mo._scan(os.open(tmp_path / "out", os.O_RDONLY | os.O_DIRECTORY), out.rec, out.limits, 2)
    assert (count, covered) == (3, 2) and used == out.used  # due coverage is a COUNT of validated unique multiples


@pytest.mark.skipif(sys.platform != "linux", reason="fresh-process checkpoint test uses the Linux fixtures")
def test_fresh_process_physical_restart_with_the_genuine_v3_checkpoint(tmp_path):
    env = dict(os.environ)
    run = lambda mode: subprocess.run([sys.executable, "-B", str(HERE / "fresh_worker.py"), mode, str(tmp_path)], capture_output=True, text=True, env=env, cwd=str(HERE), timeout=900)  # noqa: E731
    a = run("A")
    assert a.returncode == 0, a.stderr[-3000:]
    assert json.loads(a.stdout.strip().splitlines()[-1]) == {"day": 3}
    b = run("B")  # a NEW process: rebuilds components, loads the V3 checkpoint, resumes the output, replays and continues
    assert b.returncode == 0, b.stderr[-3000:]
    res = json.loads(b.stdout.strip().splitlines()[-1])
    assert res["repeat"] == "repeat_same_state" and res["day4"] == "saved" and res["last_day"] == 4
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names == ["active.npy", "day_000000000002", "day_000000000004", "output.json"]
    # the uninterrupted run reaches the same day-4 bytes (the restart is physically exact, not merely writer == reader)
    sys.path.insert(0, str(HERE))
    import fresh_worker as fw

    c = fw.case()
    s = start(c)
    for p in (2.0, 0.0, 1.0, 2.0):
        s = fw.day(c, s, p)
    snap = mo.read_snapshot(tmp_path / "out" / "day_000000000004", max_bytes=1 << 20)
    assert snap.water_m.tobytes() == host(s.payload.soil.water_m).tobytes()
