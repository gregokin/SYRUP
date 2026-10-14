"""CW-E2: content-addressed actual-MAPLE dependency delivery (export / verify / materialize / runtime)."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tracemalloc
from pathlib import Path

import pytest
from conftest import FROZEN_M24, SRC, write_origin

from maple_syrup import dependency_bundle as db


def snapshot(root: Path) -> dict[str, tuple[str, int, int]]:
    """Every file under root: (sha256, size, mtime_ns) -- detects any write to the source."""
    out = {}
    for dirpath, _d, files in os.walk(root):
        for name in files:
            p = Path(dirpath, name)
            st = p.lstat()
            out[str(p.relative_to(root))] = (hashlib.sha256(p.read_bytes()).hexdigest(), st.st_size, st.st_mtime_ns)
    return out


def export(project, name="bundle", **kw):
    out = project["tmp"] / name
    return db.export_bundle(project["root"], project["dist"], project["origin"], out, **kw), out


def leftovers(parent: Path) -> list[str]:
    return [p.name for p in parent.iterdir() if ".staging-" in p.name]


# --- export, determinism, round trip ----------------------------------------------------------------------
def test_export_is_deterministic_and_never_writes_the_source(project):
    before = snapshot(project["root"])
    rec1, out1 = export(project, "b1")
    rec2, out2 = export(project, "b2")
    assert rec1["bundle_id"] == rec2["bundle_id"]
    assert (out1 / db.ARCHIVE_NAME).read_bytes() == (out2 / db.ARCHIVE_NAME).read_bytes()
    assert snapshot(project["root"]) == before
    paths = [f["path"] for f in rec1["files"]]
    assert "source/src/maple/__pycache__/x.cpython-312.pyc" not in paths  # documented cache exclusion
    assert "source/tests/test_x.py" in paths and "source/pyproject.toml" in paths  # full project, not only pkg
    assert "distribution/METADATA" in paths and "distribution/RECORD" not in paths
    assert rec1["git_supplementary"] == {"status": "not_a_git_repository"}  # recorded, not invented
    assert rec1["origin"]["original_source"]["revision"] is None
    assert sorted(os.listdir(out1)) == sorted([db.ARCHIVE_NAME, db.MANIFEST_NAME])
    with tarfile.open(out1 / db.ARCHIVE_NAME) as tar:  # normalized entries
        for m in tar.getmembers():
            assert m.isreg() and m.mode == 0o644 and m.mtime == 0 and m.uid == m.gid == 0 and not m.uname


def test_package_digest_reuses_the_existing_provenance_digest(project):
    from maple_syrup.provenance import source_tree_digest

    rec, _ = export(project)
    assert rec["project"]["package_source_digest"]["digest_sha256"] == \
        source_tree_digest(project["root"] / "src" / "maple").digest_sha256


def test_materialize_round_trip_bytes_overlay_and_reuse(project):
    rec, out = export(project)
    cache = project["tmp"] / "cache"
    cache.mkdir()
    target = db.materialize_bundle(out, cache)
    assert target.name == rec["bundle_id"]
    for entry in rec["files"]:
        if entry["path"].startswith("source/"):
            rel = entry["path"][len("source/"):]
            assert (target / "source" / rel).read_bytes() == (project["root"] / rel).read_bytes()
    info = target / "editable" / "maple-0.0.1.dist-info"
    direct = json.loads((info / "direct_url.json").read_text())
    assert direct == {"dir_info": {"editable": True}, "url": "file://" + str(target / "source")}
    assert (info / "METADATA").read_bytes() == (project["dist"] / "METADATA").read_bytes()
    assert (info / "INSTALLER").read_text().strip() == db.INSTALLER_NAME
    assert (target / "editable" / "__editable__.maple-0.0.1.pth").read_text().strip() == str(target / "source" / "src")
    assert not os.access(target / "source" / "pyproject.toml", os.W_OK)  # frozen cache
    again = db.materialize_bundle(out, cache)  # existing target is re-verified, not rewritten
    assert again == target and leftovers(cache) == []
    db.verify_materialized(target)


def test_existing_resolver_identifies_the_materialized_cache(project):
    _rec, out = export(project)
    cache = project["tmp"] / "cache"
    cache.mkdir()
    target = db.materialize_bundle(out, cache)
    env = dict(os.environ, **db.runtime_environment(target, SRC)["env"])
    script = ("import json; from maple_syrup.dependency import resolve_maple_dependency as r; "
              "print(json.dumps(r(requirements=()).as_record()))")
    clean = project["tmp"] / "clean_cwd"  # `-c` puts the cwd first on sys.path
    clean.mkdir()
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True,
                          cwd=clean)
    record = json.loads(proc.stdout)
    assert record["source_root"] == str(target / "source")
    assert record["source_kind"] == "editable_working_tree" and record["distribution_version"] == "0.0.1"
    assert record["package_dir"] == str(target / "source" / "src" / "maple")
    # A stray `maple` dist-info earlier on sys.path (here: the cwd holding the fixture's dist-info) shadows the
    # overlay; the existing resolver refuses it rather than accepting the wrong distribution.
    shadowed = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True,
                              cwd=project["tmp"], check=False)
    assert shadowed.returncode != 0 and "differs from the installed `maple` distribution" in shadowed.stderr


# --- export refusals --------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "git_dir", "git_file", "fifo", "dir_symlink"])
def test_export_refuses_unsafe_source_entries(project, kind):
    root = project["root"] / "src" / "maple"
    if kind == "symlink":
        os.symlink(root / "__init__.py", root / "alias.py")
    elif kind == "hardlink":
        os.link(root / "__init__.py", root / "hard.py")
    elif kind == "git_dir":
        (project["root"] / ".git").mkdir()
    elif kind == "git_file":
        (project["root"] / ".git").write_text("gitdir: elsewhere\n")
    elif kind == "fifo":
        os.mkfifo(root / "pipe")
    else:
        os.symlink(root / "core", root / "linkdir")
    with pytest.raises(db.BundleError):
        export(project)
    assert not (project["tmp"] / "bundle").exists() and leftovers(project["tmp"]) == []


@pytest.mark.parametrize("limit", ["max_files", "max_file_bytes", "max_total_bytes"])
def test_export_bounds_are_enforced_before_reading(project, limit):
    small = {"max_files": db.Limits(max_files=3), "max_file_bytes": db.Limits(max_file_bytes=1000),
             "max_total_bytes": db.Limits(max_total_bytes=5000)}[limit]
    with pytest.raises(db.BundleError, match=limit):
        export(project, limits=small)
    assert not (project["tmp"] / "bundle").exists() and leftovers(project["tmp"]) == []


def test_output_is_no_clobber(project):
    out = project["tmp"] / "bundle"
    out.mkdir()
    (out / "keep").write_text("mine")
    with pytest.raises(db.BundleError, match="already exists"):
        export(project)
    assert sorted(os.listdir(out)) == ["keep"] and (out / "keep").read_text() == "mine"
    with pytest.raises(db.BundleError, match="outside"):
        db.export_bundle(project["root"], project["dist"], project["origin"], project["root"] / "inside")


def test_rename_noreplace_refuses_an_empty_directory_created_concurrently(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()  # plain rename(2) would silently replace an empty directory
    with pytest.raises(db.BundleError, match="without replacing"):
        db._rename_noreplace(src, dst)
    assert src.is_dir() and dst.is_dir()


@pytest.mark.parametrize("change", ["version", "requires", "extra", "toplevel", "missing"])
def test_distribution_metadata_must_match_the_project(project, change):
    meta = project["dist"] / "METADATA"
    text = meta.read_text()
    if change == "version":
        meta.write_text(text.replace("Version: 0.0.1", "Version: 9.9"))
    elif change == "requires":
        meta.write_text(text.replace("Requires-Dist: numpy\n", "Requires-Dist: numpy>=2\n"))
    elif change == "extra":
        meta.write_text(text.replace("Provides-Extra: dev\n", "Provides-Extra: dev\nProvides-Extra: gpu\n"))
    elif change == "toplevel":
        (project["dist"] / "top_level.txt").write_text("maple\nother\n")
    else:
        (project["dist"] / "WHEEL").unlink()
    with pytest.raises(db.BundleError):
        export(project)


@pytest.mark.parametrize("bad", ["no_reason", "empty_chain_no_reason", "extra_key", "wrong_schema", "dup_key"])
def test_origin_record_is_explicit_and_strict(project, bad):
    path = project["tmp"] / "origin.json"
    if bad == "no_reason":
        write_origin(path, original_source={"kind": "x", "revision": None, "revision_absence_reason": None,
                                            "dirty_snapshot_note": None})
    elif bad == "empty_chain_no_reason":
        write_origin(path, patch_chain_absence_reason=None)
    elif bad == "extra_key":
        write_origin(path, git_head="invented")
    elif bad == "wrong_schema":
        write_origin(path, schema="other")
    else:
        text = path.read_text()
        path.write_text(text[:-1] + ', "description": "again"}')
    with pytest.raises(db.BundleError):
        export(project)


def test_concurrent_source_change_refuses_publication(project):
    target = project["root"] / "src" / "maple" / "core" / "physics.py"

    def mutate():
        target.write_text("def f(x):\n    return 3 * x\n")

    with pytest.raises(db.SourceChangedError):
        export(project, _after_archive=mutate)
    assert not (project["tmp"] / "bundle").exists() and leftovers(project["tmp"]) == []


# --- verification of tampered bundles ----------------------------------------------------------------------
def rebuild(out: Path, dest: Path, members: list[tuple[tarfile.TarInfo, bytes | None]], record: dict,
            files: list[dict] | None = None) -> Path:
    """A bundle whose manifest is internally CONSISTENT (archive sha/size, bundle_id) around a crafted tar,
    so only the archive-content checks can refuse it."""
    dest.mkdir()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    archive = buf.getvalue()
    (dest / db.ARCHIVE_NAME).write_bytes(archive)
    body = {k: v for k, v in record.items() if k != "bundle_id"}
    if files is not None:
        body.update(files=files, file_count=len(files), total_bytes=sum(f["size"] for f in files))
    body["archive"] = dict(body["archive"], size=len(archive), sha256=hashlib.sha256(archive).hexdigest())
    full = {"bundle_id": hashlib.sha256(db._canonical(body)).hexdigest(), **body}
    (dest / db.MANIFEST_NAME).write_bytes(db._canonical(full) + b"\n")
    return dest


def original_members(out: Path) -> list[tuple[tarfile.TarInfo, bytes]]:
    with tarfile.open(out / db.ARCHIVE_NAME) as tar:
        return [(m, tar.extractfile(m).read()) for m in tar.getmembers()]


def info(name, size=0, kind=tarfile.REGTYPE, **kw):
    t = db._tarinfo(name, size)
    t.type = kind
    for k, v in kw.items():
        setattr(t, k, v)
    return t


@pytest.mark.parametrize("attack", ["traversal", "absolute", "symlink", "hardlink", "directory", "duplicate",
                                    "extra", "missing", "size", "content", "mode", "mtime"])
def test_materialize_refuses_crafted_archives_and_leaves_nothing(project, attack):
    rec, out = export(project)
    members = original_members(out)
    first, data = members[0]
    if attack == "traversal":
        members.append((info("source/../../evil.py", 4), b"evil"))
    elif attack == "absolute":
        members.append((info("/tmp/evil.py", 4), b"evil"))
    elif attack == "symlink":
        members[0] = (info(first.name, 0, tarfile.SYMTYPE, linkname="/etc/passwd"), None)
    elif attack == "hardlink":
        members[0] = (info(first.name, 0, tarfile.LNKTYPE, linkname=members[1][0].name), None)
    elif attack == "directory":
        members.insert(0, (info("source/src", 0, tarfile.DIRTYPE), None))
    elif attack == "duplicate":
        members.append((first, data))
    elif attack == "extra":
        members.append((info("source/unlisted.py", 1), b"x"))
    elif attack == "missing":
        members.pop()
    elif attack == "size":
        members[0] = (info(first.name, len(data) + 1), data + b"!")
    elif attack == "content":
        members[0] = (info(first.name, len(data)), bytes(len(data)) if data else data)
        if not data:
            pytest.skip("first member empty")
    elif attack == "mode":
        members[0] = (info(first.name, len(data), mode=0o755), data)
    else:
        members[0] = (info(first.name, len(data), mtime=12345), data)
    bad = rebuild(out, project["tmp"] / "bad", members, rec)
    with pytest.raises(db.BundleError):
        db.verify_bundle(bad)
    cache = project["tmp"] / "cache"
    cache.mkdir()
    with pytest.raises(db.BundleError):
        db.materialize_bundle(bad, cache)
    assert os.listdir(cache) == []  # no target, no staging
    assert not Path("/tmp/evil.py").exists()


@pytest.mark.parametrize("tamper", ["archive_byte", "bundle_id", "manifest_field", "schema", "dup_key",
                                    "extra_file", "manifest_traversal", "oversized_manifest"])
def test_verify_refuses_tampered_bundles(project, tamper):
    _rec, out = export(project)
    bad = project["tmp"] / "bad"
    shutil.copytree(out, bad)
    for p in bad.iterdir():
        os.chmod(p, 0o644)
    manifest = bad / db.MANIFEST_NAME
    if tamper == "archive_byte":
        raw = bytearray((bad / db.ARCHIVE_NAME).read_bytes())
        raw[600] ^= 1
        (bad / db.ARCHIVE_NAME).write_bytes(bytes(raw))
    elif tamper in ("bundle_id", "manifest_field", "schema", "manifest_traversal"):
        doc = json.loads(manifest.read_text())
        if tamper == "bundle_id":
            doc["bundle_id"] = "0" * 64
        elif tamper == "manifest_field":
            doc["project"]["version"] = "9.9"
        elif tamper == "schema":
            doc["schema"] = "maple_syrup.dependency_bundle.v2"
            body = {k: v for k, v in doc.items() if k != "bundle_id"}
            doc["bundle_id"] = hashlib.sha256(db._canonical(body)).hexdigest()
        else:
            doc["files"][0]["path"] = "source/../escape.py"
            body = {k: v for k, v in doc.items() if k != "bundle_id"}
            doc["bundle_id"] = hashlib.sha256(db._canonical(body)).hexdigest()
        manifest.write_text(json.dumps(doc))
    elif tamper == "dup_key":
        manifest.write_text(manifest.read_text().rstrip()[:-1] + ',"schema":"x"}')
    elif tamper == "extra_file":
        (bad / "README").write_text("extra")
    else:
        manifest.write_bytes(b" " * 100 + manifest.read_bytes())
        with pytest.raises(db.BundleError, match="bound"):
            db.verify_bundle(bad, limits=db.Limits(max_manifest_bytes=64))
        return
    with pytest.raises(db.BundleError):
        db.verify_bundle(bad)


def test_verify_materialized_detects_post_materialization_tampering(project):
    _rec, out = export(project)
    cache = project["tmp"] / "cache"
    cache.mkdir()
    target = db.materialize_bundle(out, cache)
    victim = target / "source" / "src" / "maple" / "__init__.py"
    os.chmod(victim.parent, 0o755)
    os.chmod(victim, 0o644)
    victim.write_text("VALUE = 2\n")
    with pytest.raises(db.BundleError):
        db.verify_materialized(target)
    with pytest.raises(db.BundleError):
        db.materialize_bundle(out, cache)  # an existing but corrupt target is refused, never rewritten


def test_streaming_memory_is_bounded_on_a_large_file(project):
    big = project["root"] / "src" / "maple" / "big.bin"
    big.write_bytes(os.urandom(1 << 20) * 12)  # 12 MiB
    export(project, "warm")  # warm imports/caches outside the measurement
    tracemalloc.start()
    try:
        _rec, out = export(project, "measured")
        cache = project["tmp"] / "cache"
        cache.mkdir()
        db.materialize_bundle(out, cache)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 << 20, peak  # bounded chunks, never the whole 12 MiB file or archive


# --- the actual frozen M24 MAPLE copy (read-only) ----------------------------------------------------------
@pytest.mark.skipif(not (FROZEN_M24 / "source" / "pyproject.toml").is_file(), reason="frozen M24 copy absent")
def test_frozen_m24_round_trip_through_the_genuine_resolver(tmp_path):
    source = FROZEN_M24 / "source"
    dist = FROZEN_M24 / "editable" / "maple-0.0.1.dist-info"
    before = snapshot(FROZEN_M24)
    origin = write_origin(tmp_path / "origin.json", description="CW-B24 frozen M24 actual-MAPLE copy",
                          original_source={"kind": "immutable copied MAPLE project with reviewed water patches",
                                           "revision": None,
                                           "revision_absence_reason": "copied tree carries no .git; baseline "
                                           "git objects are not asserted here",
                                           "dirty_snapshot_note": "content identity is the project digest"},
                          patch_chain=[], patch_chain_absence_reason="test export: patch chain recorded by root")
    rec = db.export_bundle(source, dist, origin, tmp_path / "bundle")
    cache = tmp_path / "cache"
    cache.mkdir()
    target = db.materialize_bundle(tmp_path / "bundle", cache)
    env = dict(os.environ, **db.runtime_environment(target, SRC)["env"])
    proc = subprocess.run([sys.executable, "-m", "maple_syrup.dependency_bundle", "check-runtime", str(target)],
                          env=env, capture_output=True, text=True, cwd=tmp_path, check=False)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["bundle_id"] == rec["bundle_id"]
    assert result["dependency"]["source_root"] == str(target / "source")
    assert result["package_source_digest"] == rec["project"]["package_source_digest"]["digest_sha256"]
    assert snapshot(FROZEN_M24) == before  # the frozen copy was only read
