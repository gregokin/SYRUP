#!/usr/bin/env python3
"""Reconstruct the exact actual MAPLE water-runtime project (``src/maple`` + ``pyproject.toml``) from the pinned
upstream MAPLE Git revision and the tracked net patch in ``dependencies/maple_water`` (CW-E9 tool; the pinned
recipe is the CW-E46 24c6 runtime: the CW-E44 M94 runtime plus the CW-B106C1 exact-overlay store and the CW-B107
publisher / snapshot changes -- six existing water files modified, no file added or removed; runtime only, distinct
from the full-project 24c6 delivery).

Read-only on the MAPLE repository: only ``git rev-parse``, ``git ls-tree`` and ``git archive`` of the pinned
commit are used. Never a checkout, HEAD, the working tree or its dirty files. Nothing is installed.

Writes ONLY inside one unique staging directory created beside the destination, then publishes it with one
rename onto the destination, which must not exist. On any failure the staging directory, and nothing else, is
removed. Checks, in order:

1. The manifest bytes match the pinned SHA-256. The manifest structure is valid and every path is canonical and
   in scope (``src/maple/...`` or ``pyproject.toml``) with no duplicates. The patch bytes match the manifest pin.
   The patch headers name exactly the manifest's changed and new files, with no rename/copy/delete/mode/binary/
   symlink sections and no unsafe paths.
2. The revision is the pinned commit. ``git ls-tree`` lists exactly the manifest's pre-patch files (mode 100644,
   blob id, size), within the size bounds.
3. ``git archive`` stays within the byte bound. Every member is checked: regular files and directories only, no
   links or devices, no absolute or ``..`` paths, no duplicates, nothing unexpected, the expected size, and the
   SHA-256 of every extracted file.
4. ``git apply --check`` then ``git apply -p1``, run with repository discovery fenced at the staging directory.
5. The complete post-patch file list and every SHA-256 match. The ``src/maple`` package source digest equals the
   pin. There are no symlinks, special files or leftovers.
6. A receipt is written and the result is published atomically with Linux ``renameat2(RENAME_NOREPLACE)``: the
   kernel refuses the rename if the destination exists in ANY form (empty directory, file, symlink, dangling
   symlink), including one created concurrently after the early check. Linux is the supported platform; elsewhere,
   or where ``renameat2`` / the flag is unavailable, the tool refuses (before any work, and again at publication).
   There is no fallback rename that could replace anything.

Usage::

    python tools/reconstruct_water_maple.py --maple-repo /path/to/MAPLE --dest /new/path/maple_water_runtime
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

RECIPE_DIR = Path(__file__).resolve().parents[1] / "dependencies" / "maple_water"
MANIFEST_NAME = "manifest.json"
# The ONE trust anchor of this tool: the manifest's exact bytes (which pin the revision, every file hash and the
# patch). Changing the recipe requires changing this pin.
EXPECTED_MANIFEST_SHA256 = "038dcedabd704613dee2e41ddcfb1d1cac60c664f11171edab1382d70351748d"  # CW-E46 24c6 recipe
SCHEMA = "maple_syrup.maple_water_runtime_recipe.v1"
DIGEST_ALGORITHM = "sha256(sorted relpath NUL file_sha256 LF), excl __pycache__/*.pyc/*.pyo; v1"
RECEIPT_NAME = "maple_water_reconstruction.json"
MAX_MANIFEST_BYTES = 4 << 20
MAX_PATCH_BYTES = 16 << 20
MAX_TREE_BYTES = 64 << 20  # upstream pre-patch total (actual ~7.7 MB)
MAX_ARCHIVE_BYTES = 96 << 20
MAX_FILES = 5000
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_ENV_DROP = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_NAMESPACE",
                 "GIT_COMMON_DIR", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG_PARAMETERS")


class RecipeError(RuntimeError):
    """A refused reconstruction. Nothing was published; only this run's staging directory was removed."""


# --- validation helpers -----------------------------------------------------------------------------------------
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bounded(path: Path, limit: int, what: str) -> bytes:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RecipeError(f"{what} {path} cannot be read: {exc}") from None
    if not stat.S_ISREG(info.st_mode):
        raise RecipeError(f"{what} {path} is not a regular file")
    if info.st_size > limit:
        raise RecipeError(f"{what} {path} exceeds {limit} bytes")
    with open(path, "rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise RecipeError(f"{what} {path} exceeds {limit} bytes")
    return data


def safe_relpath(name: str) -> str:
    """The canonical in-scope relative POSIX path, or RecipeError."""
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name or name != name.strip():
        raise RecipeError(f"unsafe path {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")) or p.as_posix() != name:
        raise RecipeError(f"unsafe path {name!r}")
    if not (name == "pyproject.toml" or (name.startswith("src/maple/") and len(p.parts) > 2)):
        raise RecipeError(f"path {name!r} is outside the runtime scope (src/maple/..., pyproject.toml)")
    return name


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise RecipeError(f"malformed manifest: {message}")


def load_manifest(recipe_dir: Path, expected_sha256: str) -> tuple[dict, bytes]:
    data = _read_bounded(recipe_dir / MANIFEST_NAME, MAX_MANIFEST_BYTES, "manifest")
    if _sha256(data) != expected_sha256:
        raise RecipeError(f"manifest SHA-256 {_sha256(data)} differs from the pinned {expected_sha256}; refused")
    try:
        m = json.loads(data, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, RecipeError) as exc:
        raise RecipeError(f"malformed manifest JSON: {exc}") from None
    _require(isinstance(m, dict) and m.get("schema") == SCHEMA, "schema")
    up = m.get("upstream")
    _require(isinstance(up, dict) and isinstance(up.get("revision"), str) and _HEX40.match(up["revision"])
             and up.get("pathspecs") == ["src/maple", "pyproject.toml"], "upstream")
    pre, post, patch = m.get("pre_patch"), m.get("post_patch"), m.get("patch")
    _require(isinstance(pre, dict) and isinstance(pre.get("files"), dict), "pre_patch")
    _require(isinstance(post, dict) and isinstance(post.get("files"), dict), "post_patch")
    _require(isinstance(patch, dict), "patch")
    _require(0 < len(pre["files"]) <= MAX_FILES and pre.get("file_count") == len(pre["files"]), "pre_patch count")
    _require(0 < len(post["files"]) <= MAX_FILES and post.get("file_count") == len(post["files"]), "post_patch count")
    total = 0
    for name, meta in pre["files"].items():
        safe_relpath(name)
        _require(isinstance(meta, dict) and meta.get("git_mode") == "100644" and isinstance(meta.get("git_blob"), str)
                 and _HEX40.match(meta["git_blob"]) and isinstance(meta.get("sha256"), str)
                 and _HEX64.match(meta["sha256"]) and isinstance(meta.get("size"), int)
                 and not isinstance(meta.get("size"), bool) and meta["size"] >= 0, f"pre_patch entry {name}")
        total += meta["size"]
    _require(total == pre.get("total_bytes") and total <= MAX_TREE_BYTES, "pre_patch bytes")
    for name, digest in post["files"].items():
        safe_relpath(name)
        _require(isinstance(digest, str) and _HEX64.match(digest), f"post_patch entry {name}")
    pkg = post.get("package_source_digest")
    _require(isinstance(pkg, dict) and pkg.get("root") == "src/maple" and pkg.get("algorithm") == DIGEST_ALGORITHM
             and isinstance(pkg.get("digest_sha256"), str) and _HEX64.match(pkg["digest_sha256"])
             and pkg.get("file_count") == sum(1 for k in post["files"] if k.startswith("src/maple/")),
             "package_source_digest")
    _require(isinstance(patch.get("file"), str) and re.fullmatch(r"[A-Za-z0-9_.-]+\.patch", patch["file"])
             and isinstance(patch.get("sha256"), str) and _HEX64.match(patch["sha256"]) and patch.get("strip") == 1,
             "patch pin")
    for key in ("changed_files", "new_files", "modified_files", "deleted_files"):
        values = patch.get(key)
        _require(isinstance(values, list) and len(values) == len(set(values)), f"patch.{key}")
        for name in values:
            safe_relpath(name)
    _require(patch["deleted_files"] == [], "deleted files are not supported")
    _require(sorted(patch["new_files"] + patch["modified_files"]) == sorted(patch["changed_files"]), "patch sets")
    _require(set(patch["new_files"]) == set(post["files"]) - set(pre["files"]), "new files vs file lists")
    _require(set(pre["files"]) <= set(post["files"]), "a pre-patch file disappears")
    _require(set(patch["modified_files"]) <= set(pre["files"]), "modified file not in the pre-patch list")
    return m, data


def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise RecipeError("duplicate JSON key")
    return dict(pairs)


_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$")
_INDEX = re.compile(r"^index [0-9a-f]{40}\.\.[0-9a-f]{40}( 100644)?$")


def parse_patch(data: bytes, manifest: dict) -> list[str]:
    """Strict structural validation of the net patch against the manifest; returns the changed paths.

    Per file section: `diff --git a/P b/P`, optionally `new file mode 100644`, `index <40hex>..<40hex>[ 100644]`,
    `--- a/P` (or `--- /dev/null` for a new file), `+++ b/P`, then hunks whose lines start with `@@`, ' ', '+', '-'
    or the no-newline marker. Anything else (renames, copies, deletions, mode changes, binary or symlink sections,
    unsafe or mismatched paths, text before the first section) is refused."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise RecipeError("patch is not UTF-8 text") from None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    changed, new = [], []
    i, n = 0, len(lines)
    if n == 0 or not lines[0].startswith("diff --git "):
        raise RecipeError("patch does not start with a file section")
    while i < n:
        m = _HEADER.match(lines[i])
        if not m or m.group(1) != m.group(2):
            raise RecipeError(f"bad patch section header {lines[i][:80]!r}")
        path = safe_relpath(m.group(1))
        i += 1
        is_new = i < n and lines[i] == "new file mode 100644"
        if is_new:
            i += 1
        if i >= n or not _INDEX.match(lines[i]):
            raise RecipeError(f"bad or missing index line for {path!r}")
        i += 1
        if i + 1 >= n or lines[i] != ("--- /dev/null" if is_new else f"--- a/{path}") or lines[i + 1] != f"+++ b/{path}":
            raise RecipeError(f"bad file header lines for {path!r}")
        i += 2
        hunks = 0
        while i < n and not lines[i].startswith("diff --git "):
            line = lines[i]
            if line.startswith("@@ "):
                hunks += 1
            elif not line or line[0] not in " +-\\" or (line[0] == "\\" and line != "\\ No newline at end of file"):
                raise RecipeError(f"unexpected line in hunk for {path!r}: {line[:60]!r}")
            elif hunks == 0:
                raise RecipeError(f"hunk line before any @@ for {path!r}")
            i += 1
        if hunks == 0:
            raise RecipeError(f"section for {path!r} has no hunk")
        changed.append(path)
        if is_new:
            new.append(path)
    p = manifest["patch"]
    if len(changed) != len(set(changed)):
        raise RecipeError("patch names a file twice")
    if sorted(changed) != sorted(p["changed_files"]) or sorted(new) != sorted(p["new_files"]):
        raise RecipeError("patch headers differ from the manifest's changed/new file lists")
    return changed


# --- git (read-only, argument lists, sanitized environment) -------------------------------------------------------
def _git_env(ceiling: Path | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _GIT_ENV_DROP}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    if ceiling is not None:
        env["GIT_CEILING_DIRECTORIES"] = str(ceiling)
    return env


def _git(git: str, repo: Path, *args: str, limit: int = 1 << 24) -> bytes:
    proc = subprocess.run([git, "-C", str(repo), *args], capture_output=True, env=_git_env(), check=False)
    if proc.returncode != 0:
        raise RecipeError(f"git {' '.join(args[:2])} failed: {proc.stderr.decode(errors='replace').strip()[:300]}")
    if len(proc.stdout) > limit:
        raise RecipeError(f"git {args[0]} output exceeds {limit} bytes")
    return proc.stdout


def verify_upstream(git: str, repo: Path, manifest: dict) -> None:
    rev = manifest["upstream"]["revision"]
    got = _git(git, repo, "rev-parse", "--verify", "--end-of-options", f"{rev}^{{commit}}").decode().strip()
    if got != rev:
        raise RecipeError(f"revision {rev} resolves to {got!r}, not the pinned commit")
    listing = _git(git, repo, "ls-tree", "-r", "-l", "--full-tree", "-z", rev, "--", *manifest["upstream"]["pathspecs"])
    seen = {}
    for record in filter(None, listing.decode("utf-8").split("\0")):
        meta, path = record.split("\t", 1)
        mode, kind, oid, size = meta.split()
        if path in seen:
            raise RecipeError(f"git lists {path!r} twice")
        seen[path] = (mode, kind, oid, size)
    expected = manifest["pre_patch"]["files"]
    if set(seen) != set(expected):
        raise RecipeError(f"upstream tree differs from the manifest: {len(set(seen) ^ set(expected))} paths differ")
    for path, (mode, kind, oid, size) in seen.items():
        e = expected[path]
        if (mode, kind, oid, size) != (e["git_mode"], "blob", e["git_blob"], str(e["size"])):
            raise RecipeError(f"upstream {path!r} is {mode} {kind} {oid} {size}, not the pinned blob")


def _archive(git: str, repo: Path, manifest: dict, target: Path) -> None:
    rev = manifest["upstream"]["revision"]
    with open(target, "xb") as out:
        proc = subprocess.Popen([git, "-C", str(repo), "archive", "--format=tar", rev, "--",
                                 *manifest["upstream"]["pathspecs"]], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=_git_env())
        written = 0
        try:
            while True:
                chunk = proc.stdout.read(1 << 20)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_ARCHIVE_BYTES:
                    proc.kill()
                    raise RecipeError(f"git archive exceeds {MAX_ARCHIVE_BYTES} bytes")
                out.write(chunk)
        finally:
            proc.stdout.close()
            err = proc.stderr.read()
            proc.stderr.close()
            code = proc.wait()
        if code != 0:
            raise RecipeError(f"git archive failed: {err.decode(errors='replace').strip()[:300]}")


def extract_archive(archive: Path, tree: Path, manifest: dict) -> None:
    """Write exactly the manifest's pre-patch files from the tar under `tree` (validated member by member)."""
    expected = manifest["pre_patch"]["files"]
    allowed_dirs = {"src", "src/maple"} | {str(PurePosixPath(p).parent) for p in expected}
    allowed_dirs |= {str(q) for p in expected for q in PurePosixPath(p).parents if str(q) != "."}
    seen = set()
    with tarfile.open(archive, mode="r:") as tar:
        rev = tar.pax_headers.get("comment")
        if rev is not None and rev != manifest["upstream"]["revision"]:
            raise RecipeError(f"archive comment {rev!r} is not the pinned revision")
        count = 0
        for member in tar:
            count += 1
            if count > 2 * MAX_FILES:
                raise RecipeError("archive has too many members")
            name = member.name.rstrip("/") if member.isdir() else member.name
            if member.isdir():
                if name not in allowed_dirs:
                    raise RecipeError(f"unexpected archive directory {member.name!r}")
                continue
            if not member.isreg():
                raise RecipeError(f"archive member {member.name!r} is not a regular file (type {member.type!r})")
            safe_relpath(name)
            if name in seen:
                raise RecipeError(f"duplicate archive member {name!r}")
            if name not in expected:
                raise RecipeError(f"unexpected archive member {name!r}")
            if member.size != expected[name]["size"]:
                raise RecipeError(f"archive member {name!r} has {member.size} bytes, expected {expected[name]['size']}")
            handle = tar.extractfile(member)
            data = handle.read(member.size + 1)
            if len(data) != member.size or _sha256(data) != expected[name]["sha256"]:
                raise RecipeError(f"archive member {name!r} does not match its pinned SHA-256")
            seen.add(name)
            dest = tree.joinpath(*name.split("/"))
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "xb") as out:
                out.write(data)
    missing = set(expected) - seen
    if missing:
        raise RecipeError(f"archive lacks {len(missing)} pinned files (e.g. {sorted(missing)[0]!r})")


def apply_patch(git: str, patch: Path, tree: Path, ceiling: Path) -> None:
    for extra in (("--check",), ()):
        proc = subprocess.run([git, "apply", *extra, "-p1", "--whitespace=nowarn", str(patch)], cwd=tree,
                              capture_output=True, env=_git_env(ceiling), check=False)
        if proc.returncode != 0:
            raise RecipeError(f"git apply {' '.join(extra) or ''} failed: "
                              f"{proc.stderr.decode(errors='replace').strip()[:300]}")


def verify_tree(tree: Path, manifest: dict) -> dict:
    """Exact post-patch file set and bytes; the package source digest; no links or special files."""
    expected = manifest["post_patch"]["files"]
    found = {}
    for dirpath, dirnames, filenames in os.walk(tree):
        for d in dirnames:
            if os.path.islink(os.path.join(dirpath, d)):
                raise RecipeError(f"symlinked directory {d!r} in the reconstructed tree")
        for f in filenames:
            path = os.path.join(dirpath, f)
            rel = Path(path).relative_to(tree).as_posix()
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode):
                raise RecipeError(f"{rel!r} is not a regular file")
            if rel not in expected:
                raise RecipeError(f"unexpected file {rel!r} after the patch")
            with open(path, "rb") as handle:
                found[rel] = _sha256(handle.read())
    if set(found) != set(expected):
        raise RecipeError(f"{len(set(expected) - set(found))} pinned post-patch files are missing")
    bad = sorted(k for k in expected if found[k] != expected[k])
    if bad:
        raise RecipeError(f"{len(bad)} post-patch files differ from their pinned SHA-256 (e.g. {bad[0]!r})")
    h = hashlib.sha256()
    for rel in sorted(k[len("src/maple/"):] for k in found if k.startswith("src/maple/")):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(found["src/maple/" + rel].encode("ascii"))
        h.update(b"\n")
    digest = h.hexdigest()
    pinned = manifest["post_patch"]["package_source_digest"]["digest_sha256"]
    if digest != pinned:
        raise RecipeError(f"package source digest {digest} differs from the pinned {pinned}")
    return {"file_count": len(found), "package_source_digest": digest}


# --- no-replace publication ---------------------------------------------------------------------------------------
# Same mechanism as the accepted CW-E2 helper `maple_syrup.dependency_bundle._rename_noreplace` (Linux
# `renameat2(AT_FDCWD, src, AT_FDCWD, dst, RENAME_NOREPLACE)`), duplicated here (a few lines) because importing that
# private helper would make this standard-library recipe depend on the SYRUP package and its modules.
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _renameat2():
    """The libc `renameat2` function, or RecipeError: fail closed on unsupported platforms."""
    if not sys.platform.startswith("linux"):
        raise RecipeError(f"no-replace publication requires Linux renameat2 (platform {sys.platform!r}); refusing")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:
        raise RecipeError(f"cannot load libc for renameat2: {exc}; refusing") from None
    fn = getattr(libc, "renameat2", None)
    if fn is None:
        raise RecipeError("libc has no renameat2; atomic no-replace publication impossible; refusing")
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    return fn


def _rename_noreplace(src: Path, dst: Path) -> None:
    """Rename `src` to `dst` atomically, failing (never replacing) if `dst` exists in any form."""
    fn = _renameat2()
    if fn(_AT_FDCWD, os.fsencode(str(src)), _AT_FDCWD, os.fsencode(str(dst)), _RENAME_NOREPLACE) != 0:
        err = ctypes.get_errno()
        raise RecipeError(f"cannot publish {dst} without replacing anything: {os.strerror(err)} (errno {err})")


# --- read-only source guard (CW-E9 correction 2) ------------------------------------------------------------------
def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def protected_source_roots(git: str, repo: Path) -> tuple[Path, list[str]]:
    """The ACTUAL canonical Git working-tree root of `repo` (a working tree or any directory inside one, symlinks
    resolved) and every canonical path a reconstruction must never write into: the working-tree root, the git dir and
    the common git dir. Read-only `git rev-parse` only. A bare repository or an argument that does not resolve to a
    working tree (e.g. the `.git` metadata directory itself) is REFUSED rather than left unprotected."""
    canon = os.path.realpath(repo)
    if not os.path.isdir(canon):
        raise RecipeError(f"MAPLE repository {repo} is not a directory")

    def rev_parse(*args: str) -> str:
        return _git(git, Path(canon), "rev-parse", *args).decode("utf-8").strip()

    if rev_parse("--is-bare-repository") != "false":
        raise RecipeError(f"{repo} is a bare repository: no working tree root to protect is resolvable; refusing")
    if rev_parse("--is-inside-work-tree") != "true":
        raise RecipeError(f"{repo} is not inside a Git working tree (e.g. git metadata); refusing")
    top = rev_parse("--show-toplevel")
    git_dir = rev_parse("--absolute-git-dir")
    common = rev_parse("--git-common-dir")
    if not top or not git_dir or not common:
        raise RecipeError(f"cannot resolve the working tree / git directories of {repo}; refusing")
    roots = [os.path.realpath(top), os.path.realpath(git_dir), os.path.realpath(os.path.join(canon, common))]
    if not _inside(canon, roots[0]):
        raise RecipeError(f"{repo} resolves outside its reported working tree {roots[0]}; refusing")
    return Path(roots[0]), roots


# --- driver -------------------------------------------------------------------------------------------------------
def reconstruct(maple_repo, dest, *, recipe_dir=RECIPE_DIR, git="git",
                expected_manifest_sha256=EXPECTED_MANIFEST_SHA256) -> dict:
    recipe_dir = Path(recipe_dir)
    dest = Path(os.path.abspath(dest))
    _renameat2()  # fail closed before any work where atomic no-replace publication is unavailable
    if os.path.lexists(dest):  # no-follow: a symlink (even dangling) at dest is an existing destination
        raise RecipeError(f"destination {dest} already exists; it is never reused, replaced or removed")
    if not dest.parent.is_dir():
        raise RecipeError(f"destination parent {dest.parent} must be an existing directory")
    # CW-E9 correction 2: every guard decision uses CANONICAL paths (all symlinks, including ancestor aliases,
    # resolved), and staging / publication then use exactly the canonical parent that was checked. The parent is
    # assumed stable for the duration of one run (trusted local parent; no claim against hostile concurrent ancestor
    # swaps). The no-replace publication below still refuses any target that appears.
    parent = Path(os.path.realpath(dest.parent))
    if not parent.is_dir():
        raise RecipeError(f"destination parent {dest.parent} does not resolve to a directory")
    dest = parent / dest.name
    if os.path.lexists(dest):
        raise RecipeError(f"destination {dest} already exists; it is never reused, replaced or removed")
    canon_repo = os.path.realpath(maple_repo)
    if _inside(str(dest), canon_repo):
        raise RecipeError("the destination may not lie inside the MAPLE repository (read-only source)")
    manifest, manifest_bytes = load_manifest(recipe_dir, expected_manifest_sha256)
    patch_path = recipe_dir / manifest["patch"]["file"]
    patch_bytes = _read_bounded(patch_path, MAX_PATCH_BYTES, "patch")
    if _sha256(patch_bytes) != manifest["patch"]["sha256"]:
        raise RecipeError("patch SHA-256 differs from the manifest pin")
    parse_patch(patch_bytes, manifest)
    repo, protected = protected_source_roots(git, Path(maple_repo))
    for root in protected:
        if _inside(str(dest), root) or _inside(str(parent), root):
            raise RecipeError(f"the destination {dest} lies inside the actual MAPLE working tree / git directory "
                              f"{root} (read-only source); refusing before any staging")
    verify_upstream(git, repo, manifest)

    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.e9-staging-", dir=parent))
    try:
        tree = staging / "tree"
        tree.mkdir()
        archive = staging / "upstream.tar"
        _archive(git, repo, manifest, archive)
        extract_archive(archive, tree, manifest)
        frozen_patch = staging / "net.patch"  # the verified bytes, never re-read from the recipe directory
        frozen_patch.write_bytes(patch_bytes)
        apply_patch(git, frozen_patch, tree, staging)
        result = verify_tree(tree, manifest)
        receipt = {
            "schema": "maple_syrup.maple_water_runtime_reconstruction.v1",
            "revision": manifest["upstream"]["revision"],
            "manifest_sha256": _sha256(manifest_bytes),
            "patch_sha256": manifest["patch"]["sha256"],
            "pre_patch_file_count": manifest["pre_patch"]["file_count"],
            "post_patch_file_count": result["file_count"],
            "package_source_digest": result["package_source_digest"],
            "scope": "runtime project only: src/maple + pyproject.toml (not a full MAPLE project, not the CW-E6 "
                     "artifact)",
        }
        with open(tree / RECEIPT_NAME, "x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=1, sort_keys=True)
            handle.write("\n")
        _rename_noreplace(tree, dest)  # the kernel refuses if dest appeared in any form since the early check
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging, ignore_errors=True)
    return dict(receipt, destination=str(dest))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--maple-repo", required=True, help="MAPLE Git repository (read-only object access)")
    p.add_argument("--dest", required=True, help="NEW destination directory (must not exist)")
    p.add_argument("--recipe-dir", default=str(RECIPE_DIR))
    p.add_argument("--git", default="git", help="git executable")
    a = p.parse_args(argv)
    try:
        receipt = reconstruct(a.maple_repo, a.dest, recipe_dir=a.recipe_dir, git=a.git)
    except RecipeError as exc:
        print(f"RecipeError: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
