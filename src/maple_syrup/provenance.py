"""Bounded source provenance for the MAPLE dependency and for MAPLE-SYRUP.

What identifies the code that actually ran
-------------------------------------------
The primary identity is `package_source_digest`: a SHA-256 over the
content of every file in the imported package directory (sorted relative
path plus per-file SHA-256), read directly from the filesystem. It
therefore covers tracked edits, untracked new modules and non-git
installs equally, and it describes the working source even when a git
HEAD does not. `__pycache__` directories and `*.pyc`/`*.pyo` files are
excluded because the interpreter rewrites them. The walk is scoped to the
package directory, so unrelated large trees in the checkout
(agent_handoffs, outputs, .venv) are never read. It is bounded: the file
count and byte limits are checked before each file is opened and each read
is capped at the remaining budget, so exceeding a bound raises rather than
truncating or reading an oversized file. Symlinks and special files inside
the scope are refused, not followed or skipped.

Git information is supplementary. `read_git_head` reads `.git/HEAD` and the
ref files directly (no subprocess). `scoped_git_status` runs one read-only
`git status` restricted to the package directory and `pyproject.toml`, with
`--no-optional-locks` and `GIT_OPTIONAL_LOCKS=0` so git does not refresh or
lock the index. It reports `clean`, `dirty` (with entries) or `unavailable`
with a reason; a failure is never reported as clean. Nothing here writes to
any repository.

MAPLE's own `maple.case_tools.provenance.capture_code_state` is not reused
for this purpose: it hashes every untracked file in the whole repository
(unbounded for the MAPLE checkout), runs `git status` without
`--no-optional-locks`, and its dirty-content hash is `None` both for a
clean tree and for a failed query. MAPLE's own run provenance continues to
use it when MAPLE's orchestrator runs.

This detects source change; it does not archive the source. Reproducing a
dirty working tree still needs a separately archived immutable snapshot.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maple_syrup.dependency import MapleDependency

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_FILES",
    "SOURCE_DIGEST_ALGORITHM",
    "ProvenanceScopeError",
    "SourceTreeDigest",
    "capture_maple_provenance",
    "capture_syrup_provenance",
    "environment_record",
    "read_git_head",
    "scoped_git_status",
    "source_tree_digest",
]

SOURCE_DIGEST_ALGORITHM = "sha256(sorted relpath NUL file_sha256 LF), excl __pycache__/*.pyc/*.pyo; v1"
EXCLUDED_DIR_NAMES = frozenset({"__pycache__"})
EXCLUDED_FILE_SUFFIXES = frozenset({".pyc", ".pyo"})
DEFAULT_MAX_FILES = 5_000
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 1 << 20
_HEX_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class ProvenanceScopeError(ValueError):
    """The requested digest scope is missing, unreadable, or exceeds its
    declared bound."""


@dataclass(frozen=True)
class SourceTreeDigest:
    root: str
    algorithm: str
    digest_sha256: str
    file_count: int
    total_bytes: int
    # (relative posix path, file sha256), sorted by path.
    files: tuple[tuple[str, str], ...]

    def as_record(self, *, include_files: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "root": self.root,
            "algorithm": self.algorithm,
            "digest_sha256": self.digest_sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }
        if include_files:
            record["files"] = [list(entry) for entry in self.files]
        return record


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_regular_file(path: Path, expected_size: int, byte_budget: int) -> tuple[str, int]:
    """SHA-256 of one regular file, reading at most `byte_budget + 1` bytes.

    Opened without following a symlink and without blocking, and re-checked
    as a regular file on the open descriptor, so a path swapped for a link or
    special file after the caller's `lstat` is refused. Growth beyond the
    budget, or any size change from `expected_size`, raises.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ProvenanceScopeError(f"{path} is not a regular file")
        hasher = hashlib.sha256()
        read = 0
        while True:
            chunk = os.read(fd, min(_READ_CHUNK_BYTES, byte_budget - read + 1))
            if not chunk:
                break
            read += len(chunk)
            if read > byte_budget:
                raise ProvenanceScopeError(
                    f"{path} grew beyond the remaining max_bytes budget while being read"
                )
            hasher.update(chunk)
    finally:
        os.close(fd)
    if read != expected_size:
        raise ProvenanceScopeError(
            f"{path} changed size during read ({expected_size} bytes before, {read} read)"
        )
    return hasher.hexdigest(), read


def _raise_walk_error(exc: OSError) -> None:
    raise ProvenanceScopeError(f"cannot list {exc.filename}: {exc}") from exc


def source_tree_digest(
    root: str | Path,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> SourceTreeDigest:
    """Content digest of every non-bytecode regular file under `root`.

    Scope policy: only regular files and real directories. Any symlink
    (file or directory) or special file inside the scope raises
    `ProvenanceScopeError`; nothing is followed or silently skipped apart
    from the declared `__pycache__`/bytecode exclusions. The file-count and
    byte limits are checked before each file is opened, and each read is
    capped at the remaining byte budget. Unreadable entries raise.
    """
    for name, value in (("max_files", max_files), ("max_bytes", max_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    root = Path(root).resolve()
    if not root.is_dir():
        raise ProvenanceScopeError(f"digest root {root} is not a directory")

    entries: list[tuple[str, str]] = []
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(root, onerror=_raise_walk_error):
        current = Path(dirpath)
        kept = []
        for name in sorted(dirnames):
            if name in EXCLUDED_DIR_NAMES:
                continue
            if (current / name).is_symlink():
                raise ProvenanceScopeError(
                    f"{current / name} is a symlinked directory; symlinks are not allowed in "
                    "the source digest scope"
                )
            kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            path = current / name
            if path.suffix in EXCLUDED_FILE_SUFFIXES:
                continue
            try:
                info = os.lstat(path)
            except OSError as exc:
                raise ProvenanceScopeError(f"cannot stat {path}: {exc}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise ProvenanceScopeError(
                    f"{path} is a symlink; symlinks are not allowed in the source digest scope"
                )
            if not stat.S_ISREG(info.st_mode):
                raise ProvenanceScopeError(f"{path} is not a regular file")
            if len(entries) >= max_files:
                raise ProvenanceScopeError(
                    f"{root} holds more than max_files={max_files} files; refusing to digest an "
                    "unexpectedly large scope"
                )
            remaining = max_bytes - total_bytes
            if info.st_size > remaining:
                raise ProvenanceScopeError(
                    f"{root} holds more than max_bytes={max_bytes} bytes (next file {path} is "
                    f"{info.st_size} bytes); refusing to digest an unexpectedly large scope"
                )
            try:
                file_hash, size = _hash_regular_file(path, info.st_size, remaining)
            except OSError as exc:
                raise ProvenanceScopeError(f"cannot read {path}: {exc}") from exc
            total_bytes += size
            entries.append((path.relative_to(root).as_posix(), file_hash))
    if not entries:
        raise ProvenanceScopeError(f"digest root {root} contains no files")

    entries.sort()
    hasher = hashlib.sha256()
    for relative, file_hash in entries:
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(file_hash.encode("ascii"))
        hasher.update(b"\n")
    return SourceTreeDigest(
        root=str(root),
        algorithm=SOURCE_DIGEST_ALGORITHM,
        digest_sha256=hasher.hexdigest(),
        file_count=len(entries),
        total_bytes=total_bytes,
        files=tuple(entries),
    )


def _git_dir(repo_root: Path) -> Path | None:
    dot_git = repo_root / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            return (repo_root / text[len("gitdir:"):].strip()).resolve()
    return None


def _lookup_ref(git_dir: Path, ref: str) -> str | None:
    common = git_dir
    commondir_file = git_dir / "commondir"
    if commondir_file.is_file():
        common = (git_dir / commondir_file.read_text(encoding="utf-8").strip()).resolve()
    for base in dict.fromkeys((git_dir, common)):
        loose = base / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip()
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            object_id, _, name = line.partition(" ")
            if name.strip() == ref:
                return object_id.strip()
    return None


def read_git_head(repo_root: str | Path) -> dict[str, Any]:
    """Read HEAD from the repository files without running git.

    Returns `status` `ok` (with `head_commit`, `head_ref`),
    `not_a_git_repository`, or `unreadable` (with `reason`).
    """
    repo_root = Path(repo_root)
    try:
        git_dir = _git_dir(repo_root)
        if git_dir is None:
            return {"status": "not_a_git_repository"}
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head[len("ref:"):].strip()
            commit = _lookup_ref(git_dir, ref)
            if commit is None:
                return {"status": "unreadable", "reason": f"ref {ref!r} not found"}
        else:
            ref, commit = None, head
    except OSError as exc:
        return {"status": "unreadable", "reason": f"{type(exc).__name__}: {exc}"}
    if not _HEX_OBJECT_ID.match(commit):
        return {"status": "unreadable", "reason": f"HEAD does not resolve to an object id: {commit!r}"}
    return {"status": "ok", "head_commit": commit, "head_ref": ref}


def _parse_porcelain_z(output: bytes) -> list[dict[str, str]]:
    tokens = output.decode("utf-8", errors="surrogateescape").split("\0")
    entries: list[dict[str, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        code, path = token[:2], token[3:]
        entry = {"code": code, "path": path}
        if code[0] in "RC" or code[1] in "RC":
            # Rename/copy records carry the original path as the next token.
            entry["original_path"] = tokens[index] if index < len(tokens) else ""
            index += 1
        entries.append(entry)
    return entries


def scoped_git_status(
    repo_root: str | Path,
    pathspecs: tuple[str, ...],
    *,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """One read-only `git status` limited to `pathspecs`.

    Untracked files inside the scope are listed individually. Never
    reports `clean` unless git actually ran and printed nothing.
    """
    command = [
        "git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(repo_root),
        "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *pathspecs,
    ]
    record: dict[str, Any] = {"pathspecs": list(pathspecs), "command": command}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        completed = subprocess.run(
            command, capture_output=True, timeout=timeout_s, check=False, env=env,
        )
    except FileNotFoundError:
        return {**record, "status": "unavailable", "reason": "git executable not found"}
    except subprocess.TimeoutExpired:
        return {**record, "status": "unavailable", "reason": f"git status timed out after {timeout_s} s"}
    except OSError as exc:
        return {**record, "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        reason = completed.stderr.decode("utf-8", errors="replace").strip()[:1000]
        return {**record, "status": "unavailable", "reason": reason or f"exit {completed.returncode}"}
    entries = _parse_porcelain_z(completed.stdout)
    return {**record, "status": "dirty" if entries else "clean", "entries": entries}


def environment_record() -> dict[str, Any]:
    import importlib.metadata

    versions: dict[str, str | None] = {}
    for name in ("numpy", "jax", "cupy-cuda12x"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "platform": platform.platform(),
        "package_versions": versions,
    }


def capture_maple_provenance(
    dependency: MapleDependency,
    *,
    include_files: bool = False,
    git_timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Provenance of the imported MAPLE: identity, working-source digest,
    and supplementary git state. Read-only."""
    tree = source_tree_digest(dependency.package_dir)
    root = dependency.source_root
    metadata_files: dict[str, str | None] = {}
    if root is not None:
        pyproject = root / "pyproject.toml"
        metadata_files["pyproject.toml"] = _sha256_file(pyproject) if pyproject.is_file() else None

    git: dict[str, Any]
    if root is None:
        git = {"status": "not_applicable", "reason": "no source root identified"}
    else:
        git = read_git_head(root)
        if git["status"] == "ok":
            try:
                scope = dependency.package_dir.relative_to(root).as_posix()
            except ValueError:
                scope = None
            if scope is None:
                git["scoped_status"] = {
                    "status": "unavailable",
                    "reason": "package directory lies outside the repository root",
                }
            else:
                git["scoped_status"] = scoped_git_status(
                    root, (scope, "pyproject.toml"), timeout_s=git_timeout_s
                )
            # HEAD describes the imported source only when git positively
            # reported the scope clean.
            git["head_describes_source"] = git["scoped_status"]["status"] == "clean"

    if dependency.source_kind == "editable_working_tree":
        label = (
            "editable/development working source: the live, mutable file content under "
            "package_dir at capture time; package_source_digest identifies it, a git HEAD "
            "alone does not"
        )
    elif dependency.source_kind == "path_import":
        label = (
            "path import without distribution metadata: identified only by pyproject.toml "
            "and package_source_digest"
        )
    else:
        label = "installed distribution files, identified by package_source_digest"

    return {
        "schema": "maple_syrup.maple_provenance.v1",
        "working_source_label": label,
        "dependency": dependency.as_record(),
        "package_source_digest": tree.as_record(include_files=include_files),
        "metadata_file_sha256": metadata_files,
        "git": git,
        "limitations": [
            (
                "The digest covers the imported package directory and pyproject.toml only; "
                "third-party packages are identified by version alone and MAPLE tests, docs "
                "and cases are not covered."
            ),
            (
                "Detection, not archival: a dirty working tree is detected and its content "
                "fingerprinted, but reproducing it requires a separately archived snapshot."
            ),
            (
                "Files can change between capture and import; compare a second digest taken "
                "after the run to detect concurrent edits."
            ),
        ],
    }


def capture_syrup_provenance(*, include_files: bool = False) -> dict[str, Any]:
    import maple_syrup

    package_dir = Path(maple_syrup.__file__).resolve().parent
    tree = source_tree_digest(package_dir)
    return {
        "schema": "maple_syrup.syrup_provenance.v1",
        "version": maple_syrup.__version__,
        "package_dir": str(package_dir),
        "package_source_digest": tree.as_record(include_files=include_files),
        "note": "SYRUP was not a git repository when Phase 1 was written; the digest is its "
        "only source identity.",
    }
