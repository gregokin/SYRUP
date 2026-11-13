"""G0 correction02: exactly-once quota accounting of a published-but-uncertain snapshot, O(L) metadata admission measured against warm tracemalloc, bounded finiteness check, date/day representability. Author-written."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from conftest import GRID, Day, build, host, layers, limits, make, mo, notify, start, wl
from ncases import fingerprint


def dd(tmp_path, d):
    return tmp_path / "out" / mo.snapshot_name(d)


def dir_fsync_fails(mp, root):
    real, ino = os.fsync, os.stat(root).st_ino

    def fsync(fd):
        if os.fstat(fd).st_ino == ino:  # Root's probe: the output ROOT directory fsync after the seal
            raise OSError("injected post-publication root fsync failure")
        return real(fd)

    mp.setattr(os, "fsync", fsync)


class Counting:
    def __init__(self, mp):
        self.calls = 0
        mp.setattr(mo, "_to_host", lambda a: self._tick() or a)

    def _tick(self):
        self.calls += 1


def day_dirs(tmp_path):
    return sorted(p.name for p in (tmp_path / "out").iterdir() if p.name.startswith("day_"))


# ---- 1. quota accounting of the sealed-but-uncredited publication ----------------------------------------------------------------------------------------------------------------------
def test_root_probe_max_snapshots_one_post_publication_failure_then_next_day_is_refused(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1, limits=limits(max_snapshots=1))
    static = out.used
    D = Day(case)
    _, s1 = D.close(s0, p=0.0)
    fp1 = fingerprint(s1)
    with monkeypatch.context() as mp:
        dir_fsync_fails(mp, tmp_path / "out")
        with pytest.raises(mo.MoistureOutputIndeterminate):
            notify(out, case, s1)
    credited = out.used - static
    assert (out.saved, out.last_day) == (1, 0) and credited > 8 * case.L * case.n  # counted once, the cadence day stays pending
    n = notify(out, case, s1)
    assert n.action == "verified_existing" and (out.saved, out.last_day, out.used) == (1, 1, static + credited)  # the retry does NOT credit it again
    assert notify(out, case, s1).action == "repeat_same_state" and (out.saved, out.used) == (1, static + credited)
    _, s2 = D.close(s1, p=0.0)
    calls = Counting(monkeypatch)
    with pytest.raises(mo.MoistureOutputQuotaError, match="max_snapshots"):
        notify(out, case, s2)
    assert calls.calls == 0 and day_dirs(tmp_path) == [mo.snapshot_name(1)] and fingerprint(s1) == fp1  # refused BEFORE any download or write; physical state unchanged
    assert out.last_day == 1


def test_exact_total_byte_cap_counts_the_published_snapshot(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    ref, r0 = make(case, tmp_path, 1)
    static = ref.used
    _, r1 = Day(case).close(r0, p=0.0)
    notify(ref, case, r1)
    one = ref.used - static
    import shutil

    shutil.rmtree(tmp_path / "out")
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1, limits=limits(max_total_bytes=static + one))
    D = Day(case)
    _, s1 = D.close(s0, p=0.0)
    with monkeypatch.context() as mp:
        dir_fsync_fails(mp, tmp_path / "out")
        with pytest.raises(mo.MoistureOutputIndeterminate):
            notify(out, case, s1)
    assert out.used == static + one  # exactly at the cap, nothing more is available
    notify(out, case, s1)
    _, s2 = D.close(s1, p=0.0)
    calls = Counting(monkeypatch)
    with pytest.raises(mo.MoistureOutputQuotaError, match="max_total_bytes"):
        notify(out, case, s2)
    assert calls.calls == 0 and day_dirs(tmp_path) == [mo.snapshot_name(1)]


def _fail_variants(mp, kind, root):
    if kind == "root_fsync":
        dir_fsync_fails(mp, root)
    elif kind == "tmp_unlink":
        real = os.unlink
        mp.setattr(os, "unlink", lambda path, *a, **k: (_ for _ in ()).throw(OSError("injected")) if path == "manifest.json.tmp" else real(path, *a, **k))
    elif kind == "any_dir_fsync":
        real = os.fsync
        mp.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("injected")) if stat.S_ISDIR(os.fstat(fd).st_mode) else real(fd))
    else:  # crash right after the link
        real = os.link

        def link_then_die(*a, **k):
            real(*a, **k)
            raise OSError("died after the link")

        mp.setattr(os, "link", link_then_die)


@pytest.mark.parametrize("kind", ["root_fsync", "tmp_unlink", "any_dir_fsync", "after_link"])
def test_every_post_publication_variant_is_credited_once_and_the_next_day_refuses_before_io(case, tmp_path, monkeypatch, kind):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1, limits=limits(max_snapshots=1))
    static = out.used
    D = Day(case)
    _, s1 = D.close(s0, p=0.0)
    with monkeypatch.context() as mp:
        _fail_variants(mp, kind, tmp_path / "out")
        with pytest.raises(mo.MoistureOutputIndeterminate):
            notify(out, case, s1)
    first = (out.saved, out.used, out.last_day)
    assert first[0] == 1 and first[1] > static and first[2] == 0
    for _ in range(3):  # repeated retries (the first advances, the rest repeat): never a second credit
        notify(out, case, s1)
    assert (out.saved, out.used, out.last_day) == (1, first[1], 1)
    _, s2 = D.close(s1, p=0.0)
    calls = Counting(monkeypatch)
    with pytest.raises(mo.MoistureOutputQuotaError):
        notify(out, case, s2)
    assert calls.calls == 0 and day_dirs(tmp_path) == [mo.snapshot_name(1)]


def test_two_failures_in_a_row_credit_a_single_snapshot(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    _, s1 = Day(case).close(s0, p=0.0)
    with monkeypatch.context() as mp:
        _fail_variants(mp, "after_link", tmp_path / "out")
        with pytest.raises(mo.MoistureOutputIndeterminate):
            notify(out, case, s1)
        credited = (out.saved, out.used)
        _fail_variants(mp, "after_link", tmp_path / "out")  # a second attempt reaches the verify path (the day exists), not `_write`
        assert notify(out, case, s1).action == "verified_existing"
    assert (out.saved, out.used) == credited


def test_fresh_writer_after_a_post_publication_crash_counts_the_sealed_snapshot_once(case, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1, limits=limits(max_snapshots=1))
    D = Day(case)
    _, s1 = D.close(s0, p=0.0)
    with monkeypatch.context() as mp:
        dir_fsync_fails(mp, tmp_path / "out")
        with pytest.raises(mo.MoistureOutputIndeterminate):
            notify(out, case, s1)
    for start_state, day in ((s0, 0), (s1, 1)):  # restart from the lagging day-0 checkpoint, or from day 1
        fresh = mo.MoistureOutput.resume(tmp_path / "out", case.lop, start_state, expected_parent=wl.expect_parent(start_state), limits=limits(max_snapshots=1))
        assert (fresh.saved, fresh.last_day) == (1, day) and fresh.used == out.used
        if day == 0:
            assert notify(fresh, case, s1).action == "verified_existing" and (fresh.saved, fresh.used) == (1, out.used)  # the replay does not credit it again
        _, s2 = D.close(s1, p=0.0)
        with pytest.raises(mo.MoistureOutputQuotaError, match="max_snapshots"):
            notify(fresh, case, s2) if day == 1 or fresh.last_day == 1 else None
    assert day_dirs(tmp_path) == [mo.snapshot_name(1)]


def test_published_ahead_snapshots_seen_by_resume_are_counted_once_not_twice(case, tmp_path):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 2)
    states = [s0]
    for k in range(4):
        _, s = Day(case).close(states[-1], p=0.0)
        states.append(s)
        notify(out, case, s)
    fresh = mo.MoistureOutput.resume(tmp_path / "out", case.lop, states[2], expected_parent=wl.expect_parent(states[2]), limits=limits())
    assert (fresh.saved, fresh.used) == (out.saved, out.used) == (2, out.used)  # day 4 is ahead of the state but counted exactly once
    for k in (3, 4):
        notify(fresh, case, states[k])
    assert (fresh.saved, fresh.used) == (2, out.used)


# ---- 2. O(L) metadata: admitted before it is built; warm tracemalloc against the plan ------------------------------------------------------------------------------------------------
def wide(L):
    return build(np.array([[0.2]]), dense=False, lay=layers(edges=tuple(k * 0.01 for k in range(L + 1)), bottom="open"), theta=(0.2,) * L, backend="numpy")


def traced_create(c, root, lim, monkeypatch):
    plans = []
    real = mo._admit_host
    monkeypatch.setattr(mo, "_admit_host", lambda lm, staging, fn: plans.append(staging) or real(lm, staging, fn))
    s0 = start(c)
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        out = mo.MoistureOutput.create(root, c.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=1, limits=lim)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        monkeypatch.setattr(mo, "_admit_host", real)
    return out, s0, plans[0], peak


@pytest.mark.parametrize("L", [2, 128, 1024])
def test_create_metadata_peak_is_inside_the_admitted_plan_warm(L, tmp_path, monkeypatch):
    c = wide(L)
    lim = limits(chunk_bytes=64, host_memory_budget_bytes=64 << 20)
    warm = tmp_path / "warm"
    warm.mkdir()
    traced_create(c, warm, lim, monkeypatch)  # warm: imports, lazy modules, caches (excluded: they are not O(L) metadata)
    root = tmp_path / "run"
    root.mkdir()
    out, s0, plan, peak = traced_create(c, root, lim, monkeypatch)
    print(f"L={L}: planned {plan} B, traced {peak} B")
    assert peak <= plan  # the model is an upper bound of the measured warm peak
    assert plan <= max(8 * peak, 160_000)  # and not absurdly loose; the ordinary few-layer overhead stays small
    if L == 2:
        assert plan < 160_000


def test_large_L_is_refused_before_any_large_list_is_built(tmp_path, monkeypatch):
    c = wide(1024)
    s0 = start(c)
    (tmp_path / "out").mkdir()
    tight = limits(chunk_bytes=64, host_memory_budget_bytes=limits(chunk_bytes=64).staging_bytes(0) + 1000)  # fits a few-layer record only
    built = []
    real = np.asarray
    monkeypatch.setattr(mo.np, "asarray", lambda *a, **k: built.append(1) or real(*a, **k))
    tracemalloc.start()
    with pytest.raises(mo.MoistureOutputQuotaError, match="host_memory_budget_bytes"):
        mo.MoistureOutput.create(tmp_path / "out", c.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=1, limits=tight)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert not built and peak < 40_000 and list((tmp_path / "out").iterdir()) == []


@pytest.mark.parametrize("L", [2, 128, 1024])
def test_resume_and_reader_metadata_peaks_are_inside_their_plans(L, tmp_path, monkeypatch):
    c = wide(L)
    lim = limits(chunk_bytes=64, host_memory_budget_bytes=64 << 20)
    root = tmp_path / "out"
    root.mkdir()
    s0 = start(c)
    out = mo.MoistureOutput.create(root, c.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=1, limits=lim)
    size = (root / "output.json").stat().st_size
    plan = lim.staging_bytes(size)
    mo.MoistureOutput.resume(root, c.lop, s0, expected_parent=wl.expect_parent(s0), limits=lim)  # warm
    tracemalloc.start()
    tracemalloc.reset_peak()
    mo.MoistureOutput.resume(root, c.lop, s0, expected_parent=wl.expect_parent(s0), limits=lim)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"L={L}: record {size} B, resume plan {plan} B, traced {peak} B")
    assert peak <= plan
    mo.read_output_record(root)  # warm
    tracemalloc.start()
    tracemalloc.reset_peak()
    mo.read_output_record(root)
    _, rpeak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert rpeak <= mo.SCRATCH_FACTOR * size + 65536  # the reader's documented metadata scratch


def test_model_terms_are_derived_not_fixture_constants():
    lim = limits(chunk_bytes=1000, max_manifest_bytes=2000, host_reserve_bytes=7)
    assert lim.staging_bytes(500) == 1000 + mo.SCRATCH_FACTOR * (2000 + 500) + mo.PY_ALLOWANCE_BYTES + 7
    assert lim.staging_bytes() == 1000 + mo.SCRATCH_FACTOR * 2000 + mo.PY_ALLOWANCE_BYTES + 7
    assert mo.JSON_FIXED_BYTES + mo.JSON_PER_LAYER_BYTES * 1024 > mo.JSON_FIXED_BYTES + mo.JSON_PER_LAYER_BYTES * 2  # the record bound grows with L


# ---- 3. finiteness without a full-domain bitmap -------------------------------------------------------------------------------------------------------------------------------------------------
def _write_day(tmp_path, data):
    L, N = data.shape
    head = mo._npy_header((L, N), "<f8")
    raw = head + data.tobytes()
    d = tmp_path / "day"
    d.mkdir()
    (d / "water_m.npy").write_bytes(raw)
    man = {"water": {"shape": [L, N], "bytes": len(raw), "file_sha256": hashlib.sha256(raw).hexdigest()}}
    return d, man


def test_non_finite_water_is_refused_in_bounded_blocks(tmp_path):
    data = np.ones((2, 100_000))
    data[1, 99_999] = np.nan
    d, man = _write_day(tmp_path, data)
    fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY)
    with pytest.raises(mo.MoistureOutputError, match="non-finite"):
        mo._load_water(fd, man, 2, 100_000)
    os.close(fd)


def test_load_water_makes_no_full_domain_temporary(tmp_path):
    data = np.ones((2, 1_000_000))  # 16 MB; a full-domain boolean would add 2 MB
    d, man = _write_day(tmp_path, data)
    fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY)
    mo._load_water(fd, man, 2, 1_000_000)  # warm
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = mo._load_water(fd, man, 2, 1_000_000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    os.close(fd)
    assert out.nbytes == 16_000_000 and peak < out.nbytes + 300_000  # the array itself + bounded temporaries (the 64k-element block), not a (L, N) bitmap


# ---- 4. dates and day naming --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,origin,ok_days", [(1, "9999-12-31", 0), (2, "9999-12-31", 0), (1, "9999-12-30", 1)])
def test_calendar_overflow_is_a_typed_refusal_before_any_io(case, tmp_path, monkeypatch, n, origin, ok_days):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, n, origin_date=origin)
    D = Day(case)
    s = s0
    for _ in range(ok_days):
        _, s = D.close(s, p=0.0)
        notify(out, case, s)
    _, bad = D.close(s, p=0.0)
    calls = Counting(monkeypatch)
    before = day_dirs(tmp_path)
    with pytest.raises(mo.MoistureOutputError, match="calendar"):
        notify(out, case, bad)
    assert calls.calls == 0 and day_dirs(tmp_path) == before and out.last_day == ok_days


def test_day_cap_matches_the_directory_name_contract(case, tmp_path, monkeypatch):
    assert len(mo.snapshot_name(mo.MAX_DAY)) == len("day_") + 12 and mo.MAX_DAY == 10**12 - 1
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1)
    _, s1 = Day(case).close(s0, p=0.0)
    monkeypatch.setattr(mo, "MAX_DAY", 0)
    calls = Counting(monkeypatch)
    with pytest.raises(mo.MoistureOutputError, match="representable day"):
        notify(out, case, s1)
    assert calls.calls == 0 and day_dirs(tmp_path) == []


def test_resealed_invalid_day_and_date_fields_raise_only_typed_errors(case, tmp_path):
    (tmp_path / "out").mkdir()
    out, s0 = make(case, tmp_path, 1, origin_date="9999-12-30")
    _, s1 = Day(case).close(s0, p=0.0)
    notify(out, case, s1)
    p = dd(tmp_path, 1) / "manifest.json"
    good = json.loads(p.read_bytes())
    for mutate in (lambda m: m.update(day=10**12), lambda m: m.update(day=2, ordinal=2), lambda m: m.update(date="not-a-date"), lambda m: m.update(date=None), lambda m: m.update(day=10**30, ordinal=10**30),
                   lambda m: m.update(day=-5)):
        man = json.loads(json.dumps(good))
        mutate(man)
        man["body_sha256"] = mo._body_sha(man)
        p.write_bytes(mo._canon(man))
        with pytest.raises(mo.MoistureOutputError):
            mo.read_snapshot(dd(tmp_path, 1), max_bytes=1 << 20)
    p.write_bytes(mo._canon(good))
    assert mo.read_snapshot(dd(tmp_path, 1), max_bytes=1 << 20).manifest["date"] == "9999-12-31"
