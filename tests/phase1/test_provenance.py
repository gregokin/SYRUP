"""Source provenance: stable digests, change detection, honest git reporting."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from maple_syrup import provenance
from maple_syrup.dependency import MapleDependency, resolve_maple_dependency
from maple_syrup.provenance import (
    ProvenanceScopeError,
    capture_maple_provenance,
    read_git_head,
    source_tree_digest,
)

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "pkg"
    (root / "sub").mkdir(parents=True)
    (root / "__init__.py").write_text("VALUE = 1\n")
    (root / "sub" / "mod.py").write_text("def f():\n    return 2\n")
    return root


def test_digest_is_stable_and_ignores_bytecode(package):
    first = source_tree_digest(package)
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\x00bytecode")
    (package / "sub" / "stale.pyc").write_bytes(b"\x00bytecode")
    second = source_tree_digest(package)

    assert second.digest_sha256 == first.digest_sha256
    assert second.file_count == 2
    assert [path for path, _ in second.files] == ["__init__.py", "sub/mod.py"]


def test_digest_detects_edit_untracked_addition_and_rename(package):
    base = source_tree_digest(package).digest_sha256

    (package / "sub" / "mod.py").write_text("def f():\n    return 3\n")
    edited = source_tree_digest(package).digest_sha256
    assert edited != base

    (package / "sub" / "new_module.py").write_text("X = 0\n")
    added = source_tree_digest(package).digest_sha256
    assert added not in (base, edited)

    (package / "sub" / "new_module.py").rename(package / "sub" / "renamed.py")
    renamed = source_tree_digest(package).digest_sha256
    assert renamed not in (base, edited, added)


def test_digest_scope_is_bounded(package):
    with pytest.raises(ProvenanceScopeError, match="max_files"):
        source_tree_digest(package, max_files=1)
    with pytest.raises(ProvenanceScopeError, match="max_bytes"):
        source_tree_digest(package, max_bytes=4)


@pytest.fixture
def opened(monkeypatch):
    """Record every path handed to the file-hashing routine."""
    paths = []
    real = provenance._hash_regular_file

    def spy(path, *args):
        paths.append(Path(path).name)
        return real(path, *args)

    monkeypatch.setattr(provenance, "_hash_regular_file", spy)
    return paths


def test_bounds_are_enforced_before_the_next_file_is_opened(package, opened):
    (package / "sub" / "zbig.py").write_bytes(b"x" * 10_000)
    small = (package / "__init__.py").stat().st_size + (package / "sub" / "mod.py").stat().st_size

    with pytest.raises(ProvenanceScopeError, match="max_bytes"):
        source_tree_digest(package, max_bytes=small + 100)
    assert opened == ["__init__.py", "mod.py"]

    opened.clear()
    with pytest.raises(ProvenanceScopeError, match="max_files"):
        source_tree_digest(package, max_files=2)
    assert opened == ["__init__.py", "mod.py"]


@pytest.mark.parametrize(
    "limits", [{"max_files": 0}, {"max_files": True}, {"max_bytes": -1}, {"max_bytes": 1.5}]
)
def test_invalid_limits_are_refused(package, limits):
    with pytest.raises(ValueError, match="positive integer"):
        source_tree_digest(package, **limits)


def test_symlinks_and_special_files_are_refused_not_followed(package, tmp_path, opened):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("SECRET = 1\n")

    link = package / "sub" / "linked.py"
    link.symlink_to(outside / "secret.py")
    with pytest.raises(ProvenanceScopeError, match="symlink"):
        source_tree_digest(package)
    link.unlink()

    dir_link = package / "linked_dir"
    dir_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProvenanceScopeError, match="symlinked directory"):
        source_tree_digest(package)
    dir_link.unlink()
    assert "secret.py" not in opened

    os.mkfifo(package / "sub" / "pipe.py")
    with pytest.raises(ProvenanceScopeError, match="not a regular file"):
        source_tree_digest(package)


def test_growth_during_read_is_detected_within_the_byte_budget(tmp_path):
    grown = tmp_path / "grown.py"
    grown.write_bytes(b"y" * 100)
    # As if lstat had seen 10 bytes and the file then grew.
    with pytest.raises(ProvenanceScopeError, match="grew beyond"):
        provenance._hash_regular_file(grown, 10, 10)
    with pytest.raises(ProvenanceScopeError, match="changed size"):
        provenance._hash_regular_file(grown, 10, 1_000)


def test_git_head_is_read_from_files_without_running_git(tmp_path):
    assert read_git_head(tmp_path)["status"] == "not_a_git_repository"

    git_dir = tmp_path / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "refs" / "heads" / "main").write_text(COMMIT + "\n")
    assert read_git_head(tmp_path) == {"status": "ok", "head_commit": COMMIT, "head_ref": "refs/heads/main"}

    (git_dir / "refs" / "heads" / "main").unlink()
    (git_dir / "packed-refs").write_text(f"# pack-refs with: peeled\n{COMMIT} refs/heads/main\n")
    assert read_git_head(tmp_path)["head_commit"] == COMMIT

    (git_dir / "HEAD").write_text("ref: refs/heads/missing\n")
    assert read_git_head(tmp_path)["status"] == "unreadable"


def test_non_git_source_is_reported_honestly(package):
    dependency = MapleDependency(
        package_dir=package,
        source_root=package.parent,
        source_kind="path_import",
        distribution_name=None,
        distribution_version=None,
        distribution_summary=None,
        direct_url=None,
        project_name="maple",
        project_version=None,
    )
    record = capture_maple_provenance(dependency)
    assert record["git"] == {"status": "not_a_git_repository"}
    assert record["package_source_digest"]["file_count"] == 2
    assert record["metadata_file_sha256"] == {"pyproject.toml": None}


def test_real_maple_provenance_does_not_let_head_stand_for_dirty_source():
    dependency = resolve_maple_dependency()
    record = capture_maple_provenance(dependency, include_files=True)

    digest = record["package_source_digest"]
    assert re.fullmatch(r"[0-9a-f]{64}", digest["digest_sha256"])
    assert digest["file_count"] == len(digest["files"]) > 0
    assert not any("__pycache__" in path for path, _ in digest["files"])
    if dependency.source_kind == "editable_working_tree":
        assert record["working_source_label"].startswith("editable/development working source")

    git = record["git"]
    if git["status"] == "ok":
        assert re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", git["head_commit"])
        scoped = git["scoped_status"]
        assert scoped["status"] in {"clean", "dirty", "unavailable"}
        assert git["head_describes_source"] is (scoped["status"] == "clean")
        if scoped["status"] == "dirty":
            assert scoped["entries"]
