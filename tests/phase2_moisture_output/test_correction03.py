"""G0 correction03: cold recovery refuses unrecognised unsealed day directories (preserving every byte) and tidies ONLY the hard-link `output.json.tmp` of a validated sealed record. Author-written."""
from __future__ import annotations

import os

import pytest
from conftest import Day, limits, make, mo, notify, start, wl


def snapshot_tree(root):
    """Byte-for-byte picture of a directory tree (names, types, symlink targets, file bytes)."""
    out = {}
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        for nm in dirs + files:
            p = os.path.join(dirpath, nm)
            rel = os.path.relpath(p, root)
            if os.path.islink(p):
                out[rel] = ("link", os.readlink(p))
            elif os.path.isdir(p):
                out[rel] = ("dir",)
            elif os.path.isfile(p):
                out[rel] = ("file", open(p, "rb").read())
            else:
                out[rel] = ("special",)
    return out


def resume(c, tmp_path, state, **kw):
    return mo.MoistureOutput.resume(tmp_path / "out", c.lop, state, expected_parent=wl.expect_parent(state), limits=kw.pop("limits", limits()), **kw)


# ---- 1. unsealed day directories ------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _fresh(case, tmp_path, **kw):
    (tmp_path / "out").mkdir()
    return make(case, tmp_path, 1, **kw)


def test_unknown_member_makes_the_unsealed_directory_not_ours_and_everything_is_preserved(case, tmp_path):
    out, s0 = _fresh(case, tmp_path)
    day = tmp_path / "out" / mo.snapshot_name(1)
    day.mkdir()
    (day / "water_m.npy").write_bytes(b"placeholder that is not ours")
    (day / "foreign_note.txt").write_text("do not delete")
    before = snapshot_tree(tmp_path / "out")
    with pytest.raises(mo.MoistureOutputError, match="unrecognised or non-regular member 'foreign_note.txt'"):
        resume(case, tmp_path, s0)
    assert snapshot_tree(tmp_path / "out") == before  # byte-for-byte: the known file name was NOT deleted either
    with pytest.raises(mo.MoistureOutputError):
        out.verify_all()
    assert snapshot_tree(tmp_path / "out") == before


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo", "symlink_tmp"])
def test_symlink_and_non_regular_members_are_refused_and_preserved(case, tmp_path, kind):
    out, s0 = _fresh(case, tmp_path)
    day = tmp_path / "out" / mo.snapshot_name(1)
    day.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("outside content")
    if kind == "symlink":
        os.symlink(victim, day / "water_m.npy")
    elif kind == "directory":
        (day / "water_m.npy").mkdir()
    elif kind == "fifo":
        os.mkfifo(day / "water_m.npy")
    else:
        (day / "water_m.npy").write_bytes(b"x")
        os.symlink(victim, day / "manifest.json.tmp")
    before = snapshot_tree(tmp_path / "out")
    with pytest.raises(mo.MoistureOutputError, match="non-regular|unrecognised"):
        resume(case, tmp_path, s0)
    assert snapshot_tree(tmp_path / "out") == before and victim.read_text() == "outside content"


def test_a_genuine_unsealed_reservation_is_still_cleaned_and_the_day_retries(case, tmp_path):
    out, s0 = _fresh(case, tmp_path)
    day = tmp_path / "out" / mo.snapshot_name(1)
    day.mkdir()
    (day / "water_m.npy").write_bytes(b"partial write")
    (day / "manifest.json.tmp").write_bytes(b"{")
    fresh = resume(case, tmp_path, s0)
    assert not day.exists() and fresh.saved == 0
    _, s1 = Day(case).close(s0, p=2.0)
    assert notify(fresh, case, s1).action == "saved"
    # a lone water file of the writer's own name is also a recognised reservation
    day2 = tmp_path / "out" / mo.snapshot_name(2)
    day2.mkdir()
    (day2 / "water_m.npy").write_bytes(b"x")
    assert resume(case, tmp_path, s1).saved == 1 and not day2.exists()


def test_an_empty_unsealed_directory_is_a_recognised_reservation(case, tmp_path):
    out, s0 = _fresh(case, tmp_path)
    day = tmp_path / "out" / mo.snapshot_name(1)
    day.mkdir()
    resume(case, tmp_path, s0)
    assert not day.exists()


# ---- 2. the static record's own hard-link temporary --------------------------------------------------------------------------------------------------------------------------------------
def _create_failing(case, tmp_path, monkeypatch, kind):
    (tmp_path / "out").mkdir()
    s0 = start(case)
    real_unlink, real_fsync = os.unlink, os.fsync
    if kind == "unlink":
        monkeypatch.setattr(os, "unlink", lambda path, *a, **k: (_ for _ in ()).throw(OSError("injected")) if path == "output.json.tmp" else real_unlink(path, *a, **k))
    else:  # the root-directory fsync after the link and unlink
        ino = os.stat(tmp_path / "out").st_ino
        monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("injected")) if os.fstat(fd).st_ino == ino else real_fsync(fd))
    with pytest.raises(mo.MoistureOutputIndeterminate):
        mo.MoistureOutput.create(tmp_path / "out", case.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=1, limits=limits())
    monkeypatch.undo()
    return s0


def _on_disk_bytes(root):
    return sum(os.stat(root / n).st_size for n in ("output.json", "active.npy"))


@pytest.mark.parametrize("kind", ["unlink", "root_fsync"])
def test_creation_interrupted_after_the_link_is_recoverable_by_resume(case, tmp_path, monkeypatch, kind):
    s0 = _create_failing(case, tmp_path, monkeypatch, kind)
    root = tmp_path / "out"
    assert (root / "output.json").exists() and (root / "active.npy").exists() and (root / "output.json.tmp").exists() == (kind == "unlink")
    sealed_before = ((root / "output.json").read_bytes(), (root / "active.npy").read_bytes())
    fresh = resume(case, tmp_path, s0)  # from the origin
    assert not (root / "output.json.tmp").exists()
    assert ((root / "output.json").read_bytes(), (root / "active.npy").read_bytes()) == sealed_before  # source/output data unchanged
    assert (fresh.saved, fresh.last_day, fresh.used) == (0, 0, _on_disk_bytes(root))  # counted exactly once, equal to the real file sizes
    _, s1 = Day(case).close(s0, p=2.0)
    n = notify(fresh, case, s1)
    assert n.action == "saved" and fresh.used == _on_disk_bytes(root) + sum(os.stat(f).st_size for f in (root / mo.snapshot_name(1)).iterdir())
    again = resume(case, tmp_path, s1)  # and from the next day
    assert (again.saved, again.last_day, again.used) == (1, 1, fresh.used)


def test_resume_from_the_next_day_after_an_interrupted_creation(case, tmp_path, monkeypatch):
    s0 = _create_failing(case, tmp_path, monkeypatch, "unlink")
    _, s1 = Day(case).close(s0, p=2.0)
    fresh = resume(case, tmp_path, s0)
    notify(fresh, case, s1)
    fresh2 = resume(case, tmp_path, s1)
    assert not (tmp_path / "out" / "output.json.tmp").exists() and (fresh2.saved, fresh2.last_day) == (1, 1)


@pytest.mark.parametrize("kind", ["other_inode", "symlink", "directory", "garbage"])
def test_an_unrelated_output_tmp_is_refused_and_preserved(case, tmp_path, kind):
    out, s0 = _fresh(case, tmp_path)
    root = tmp_path / "out"
    tmp = root / "output.json.tmp"
    victim = tmp_path / "victim.txt"
    victim.write_text("outside")
    if kind == "other_inode":
        tmp.write_bytes((root / "output.json").read_bytes())  # identical bytes, but a DIFFERENT file: not the hard link
    elif kind == "symlink":
        os.symlink(root / "output.json", tmp)
    elif kind == "directory":
        tmp.mkdir()
    else:
        tmp.write_bytes(b"not json")
    before = snapshot_tree(root)
    with pytest.raises(mo.MoistureOutputError, match="not a hard link"):
        resume(case, tmp_path, s0)
    assert snapshot_tree(root) == before and victim.read_text() == "outside"
    with pytest.raises(mo.MoistureOutputError):
        out.verify_all()
    assert snapshot_tree(root) == before


def test_guards_are_checked_before_any_tidy(case, tmp_path, monkeypatch):
    """The module/caller/owner binding of the validated record is verified BEFORE the cold scan, so a mismatching resume tidies nothing."""
    (tmp_path / "out").mkdir()
    s0 = start(case)
    real = os.unlink
    monkeypatch.setattr(os, "unlink", lambda path, *a, **k: (_ for _ in ()).throw(OSError("injected")) if path == "output.json.tmp" else real(path, *a, **k))
    with pytest.raises(mo.MoistureOutputIndeterminate):
        mo.MoistureOutput.create(tmp_path / "out", case.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=1, limits=limits(), caller_identity={"runtime": "A"})
    monkeypatch.undo()
    before = snapshot_tree(tmp_path / "out")
    with pytest.raises(mo.MoistureOutputError, match="caller identity"):
        resume(case, tmp_path, s0, caller_identity={"runtime": "B"})
    monkeypatch.setattr(mo, "_MODULE_SHA", "0" * 64)
    with pytest.raises(mo.MoistureOutputError, match="different writer module"):
        resume(case, tmp_path, s0, caller_identity={"runtime": "A"})
    monkeypatch.undo()
    assert snapshot_tree(tmp_path / "out") == before  # the temporary is still there: nothing was tidied by a refused resume
    assert resume(case, tmp_path, s0, caller_identity={"runtime": "A"}).last_day == 0 and not (tmp_path / "out" / "output.json.tmp").exists()
