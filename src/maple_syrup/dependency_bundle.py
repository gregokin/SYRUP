"""Content-addressed local delivery of an exact actual-MAPLE project tree (CW-E2).

Purpose
-------
MAPLE-SYRUP runs against genuine, immutable copies of the MAPLE project (with
reviewed water patches) instead of a mutable live checkout. A git HEAD cannot
describe a dirty tree and the original git objects may be unavailable, so this
module archives the exact project SOURCE itself, records what is known about
its origin WITHOUT inventing history, and later materializes a digest-pinned,
read-only local project cache plus a truthful editable-distribution overlay so
that the existing resolver (`maple_syrup.dependency.resolve_maple_dependency`)
identifies the materialized copy as THE `maple` distribution. Nothing is
installed, no environment is modified, nothing is published or fetched, no
command is executed. MAPLE's wind code is never copied into SYRUP's source: the
archive is an external dependency artifact.

Commands (``python -m maple_syrup.dependency_bundle ...``)
---------------------------------------------------------
``export``              project tree + actual MAPLE dist-info + origin record -> NEW bundle dir
``verify``              verify a bundle (manifest, digests, every archive member)
``materialize``         verify, extract to ``<cache>/<bundle_id>/`` with overlay (no-clobber)
``verify-materialized`` re-verify a materialized cache byte for byte
``env``                 print (do not run) the ``PYTHONPATH``/``MAPLE_SYRUP_EXPECTED_MAPLE_ROOT``
``check-runtime``       in the CURRENT process: resolve MAPLE and require it to be the cache

Bundle layout: ``bundle-manifest.json`` (canonical JSON; ``bundle_id`` is the
SHA-256 of the canonical manifest without ``bundle_id``) and
``maple-project.tar`` (uncompressed POSIX/pax tar of regular files only, sorted
paths, ``mtime=0``, ``uid=gid=0``, empty owner names, mode ``0644``). Members are
``source/<project-relative path>`` and ``distribution/<dist-info file>``.

Selection rule (export): every file below the project root EXCEPT directories
named ``__pycache__``, ``.pytest_cache``, ``.ruff_cache``, ``.mypy_cache`` and
files ending ``.pyc``/``.pyo`` (interpreter/tool caches only). Refused anywhere:
anything named ``.git``, symlinks, hard links (``st_nlink > 1``), devices, FIFOs,
sockets. Executable bits are NOT preserved (all members 0644).

Safety: every count/size bound is checked before reading or allocating; files
are streamed in bounded chunks (never whole-archive in memory); the source is
re-walked after the archive is written and any change refuses the export;
outputs are staged in a private sibling directory and published with an atomic
no-replace rename (Linux ``renameat2(RENAME_NOREPLACE)``; refused where
unavailable). Any failure removes only this tool's own staging directory and
leaves no output.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import email.parser
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from maple_syrup.dependency import (
    EXPECTED_DISTRIBUTION_NAME,
    EXPECTED_MAPLE_ROOT_ENV,
    resolve_maple_dependency,
)
from maple_syrup.provenance import (
    SOURCE_DIGEST_ALGORITHM,
    _hash_regular_file,
    read_git_head,
    source_tree_digest,
)

__all__ = [
    "ARCHIVE_NAME",
    "MANIFEST_NAME",
    "ORIGIN_SCHEMA",
    "SCHEMA",
    "BundleError",
    "Limits",
    "SourceChangedError",
    "check_runtime",
    "export_bundle",
    "materialize_bundle",
    "runtime_environment",
    "verify_bundle",
    "verify_materialized",
]

SCHEMA = "maple_syrup.dependency_bundle.v1"
ORIGIN_SCHEMA = "maple_syrup.dependency_origin.v1"
MATERIALIZATION_SCHEMA = "maple_syrup.dependency_materialization.v1"
ARCHIVE_NAME = "maple-project.tar"
MANIFEST_NAME = "bundle-manifest.json"
MATERIALIZATION_NAME = "materialization.json"
SOURCE_PREFIX = "source/"
DIST_PREFIX = "distribution/"
EXCLUDED_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"})
EXCLUDED_FILE_SUFFIXES = frozenset({".pyc", ".pyo"})
REFUSED_NAMES = frozenset({".git"})
DIST_REQUIRED = ("METADATA", "WHEEL", "top_level.txt")
DIST_OPTIONAL = ("entry_points.txt",)
PROJECT_DIGEST_ALGORITHM = "sha256(sorted relpath NUL file_sha256 LF) over the selected project files; v1"
INSTALLER_NAME = "maple_syrup.dependency_bundle"
_CHUNK = 1 << 20
_MAX_PATH_BYTES = 1024
_MAX_DIST_FILE_BYTES = 1 << 20
_TAR_BLOCK = 512
_MAX_PYPROJECT_BYTES = 1 << 20
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._+!-]{0,127}$")
SELECTION_RULE = {
    "excluded_dir_names": sorted(EXCLUDED_DIR_NAMES),
    "excluded_file_suffixes": sorted(EXCLUDED_FILE_SUFFIXES),
    "refused": ["names " + ",".join(sorted(REFUSED_NAMES)), "symlinks", "hard links", "devices/FIFOs/sockets"],
    "modes": "all members 0644; executable bits not preserved",
}
ARCHIVE_FORMAT = ("uncompressed pax tar; regular files only; sorted; mtime 0; uid/gid 0; empty owner names; "
                  "mode 0644")


class BundleError(RuntimeError):
    """A bundle or materialization request is invalid, inconsistent, unsafe or over a bound."""


class SourceChangedError(BundleError):
    """The source project changed while it was being exported."""


@dataclass(frozen=True)
class Limits:
    max_files: int = 20_000
    max_file_bytes: int = 64 << 20
    max_total_bytes: int = 512 << 20
    max_manifest_bytes: int = 16 << 20
    max_origin_bytes: int = 64 << 10

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise BundleError(f"limit {name} must be a positive integer, got {value!r}")

    @property
    def max_archive_bytes(self) -> int:
        # Per member: one header block, at most one pax header (2 blocks for a
        # path <= _MAX_PATH_BYTES) and one block of padding; plus the
        # end-of-archive blocks rounded to tarfile's 20-block record.
        per_member = _TAR_BLOCK * (1 + 1 + -(-(_MAX_PATH_BYTES + 64) // _TAR_BLOCK) + 1)
        return self.max_total_bytes + self.max_files * per_member + 20 * _TAR_BLOCK


# --- small utilities ---------------------------------------------------------------------------------------
def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise BundleError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _read_bounded_regular(path: Path, max_bytes: int, what: str) -> bytes:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise BundleError(f"cannot stat {what} {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise BundleError(f"{what} {path} is not a regular file")
    if info.st_size > max_bytes:
        raise BundleError(f"{what} {path} is {info.st_size} bytes, above its bound {max_bytes}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        data = b""
        while len(data) <= max_bytes:
            chunk = os.read(fd, min(_CHUNK, max_bytes + 1 - len(data)))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    if len(data) > max_bytes:
        raise BundleError(f"{what} {path} grew above its bound {max_bytes} while being read")
    return data


def _load_json(path: Path, max_bytes: int, what: str) -> Any:
    raw = _read_bounded_regular(path, max_bytes, what)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"{what} {path} is not valid UTF-8 JSON: {exc}") from exc


def _check_relpath(rel: Any) -> str:
    """A safe, normalized, relative POSIX path (raises otherwise)."""
    if not isinstance(rel, str) or not rel:
        raise BundleError(f"invalid path {rel!r}")
    if "\x00" in rel or "\\" in rel or len(rel.encode("utf-8")) > _MAX_PATH_BYTES:
        raise BundleError(f"invalid path {rel!r}")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or rel.startswith("/") or str(pure) != rel:
        raise BundleError(f"path {rel!r} is absolute or not normalized")
    if any(part in ("", ".", "..") for part in rel.split("/")):
        raise BundleError(f"path {rel!r} has an empty, '.' or '..' component")
    if any(part in REFUSED_NAMES for part in pure.parts):
        raise BundleError(f"path {rel!r} contains a refused name {sorted(REFUSED_NAMES)}")
    return rel


def _sha256_hex(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise BundleError(f"invalid sha256 {value!r}")
    return value


def _safe_component(value: Any, what: str) -> str:
    """One plain path component: no separators, no traversal, no control or odd characters."""
    if not isinstance(value, str) or not _SAFE_COMPONENT.match(value) or value in (".", ".."):
        raise BundleError(f"{what} {value!r} is not a safe single path component")
    return value


def _check_version(value: Any) -> str:
    """A canonical PEP 440 version that is also a safe single path component (used in overlay names)."""
    _safe_component(value, "project version")
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError as exc:  # pragma: no cover - present in the MAPLE venvs
        raise BundleError("the `packaging` module is required to validate versions") from exc
    try:
        canonical = str(Version(value))
    except InvalidVersion as exc:
        raise BundleError(f"project version {value!r} is not a PEP 440 version") from exc
    if canonical != value:
        raise BundleError(f"project version {value!r} is not in canonical PEP 440 form ({canonical!r})")
    return value


def _write_all(fd: int, data: bytes) -> None:
    """os.write until every byte is written (short writes are completed, never ignored)."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise BundleError("write made no progress")
        view = view[written:]


def _confined(base: Path, rel: str) -> Path:
    """`base/rel` for a validated relative path, after creating parents and proving they stay beneath base."""
    target = Path(base, *_check_relpath(rel).split("/"))
    os.makedirs(target.parent, mode=0o755, exist_ok=True)
    parent, root = os.path.realpath(target.parent), os.path.realpath(base)
    if parent != root and not parent.startswith(root + os.sep):
        raise BundleError(f"{rel!r} escapes {base}")
    return target


def _tree_digest(entries: list[tuple[str, int, str]]) -> str:
    hasher = hashlib.sha256()
    for rel, _size, digest in sorted(entries):
        hasher.update(rel.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return hasher.hexdigest()


def _rename_noreplace(src: Path, dst: Path) -> None:
    """Atomically rename `src` to `dst`, failing if `dst` exists in any form."""
    libc = ctypes.CDLL(None, use_errno=True)
    fn = getattr(libc, "renameat2", None)
    if fn is None:
        raise BundleError("atomic no-replace publication needs renameat2 (Linux); refusing to publish")
    at_fdcwd, rename_noreplace = -100, 1
    if fn(at_fdcwd, os.fsencode(src), at_fdcwd, os.fsencode(dst), rename_noreplace) != 0:
        err = ctypes.get_errno()
        raise BundleError(f"cannot publish {dst} without replacing anything: {os.strerror(err)}")


def _staging_for(target: Path) -> Path:
    parent = target.parent
    if not parent.is_dir():
        raise BundleError(f"parent directory {parent} does not exist")
    if os.path.lexists(target):
        raise BundleError(f"{target} already exists; refusing to clobber it")
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))


def _remove_staging(staging: Path) -> None:
    if staging.exists():
        for dirpath, dirnames, filenames in os.walk(staging):
            os.chmod(dirpath, 0o700)
            for name in filenames:
                p = os.path.join(dirpath, name)
                if not os.path.islink(p):
                    os.chmod(p, 0o600)
        shutil.rmtree(staging)


# --- export ------------------------------------------------------------------------------------------------
def _walk_project(root: Path, limits: Limits) -> list[tuple[str, int, str]]:
    """(relative path, size, sha256) of every selected file; refuses unsafe entries; bounded."""
    entries: list[tuple[str, int, str]] = []
    total = 0

    def onerror(exc: OSError) -> None:
        raise BundleError(f"cannot list {exc.filename}: {exc}") from exc

    for dirpath, dirnames, filenames in os.walk(root, onerror=onerror):
        current = Path(dirpath)
        kept = []
        for name in sorted(dirnames):
            path = current / name
            if name in REFUSED_NAMES:
                raise BundleError(f"{path}: version-control metadata ({name}) is refused in a source bundle")
            if os.path.islink(path):
                raise BundleError(f"{path} is a symlink; symlinks are refused")
            if name in EXCLUDED_DIR_NAMES:
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            path = current / name
            if name in REFUSED_NAMES:
                raise BundleError(f"{path}: version-control metadata ({name}) is refused in a source bundle")
            if path.suffix in EXCLUDED_FILE_SUFFIXES:
                continue
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                raise BundleError(f"{path} is a symlink; symlinks are refused")
            if not stat.S_ISREG(info.st_mode):
                raise BundleError(f"{path} is not a regular file (device, FIFO or socket refused)")
            if info.st_nlink > 1:
                raise BundleError(f"{path} has {info.st_nlink} hard links; hard links are refused")
            rel = _check_relpath(path.relative_to(root).as_posix())
            if len(entries) >= limits.max_files:
                raise BundleError(f"project holds more than max_files={limits.max_files} files")
            if info.st_size > limits.max_file_bytes:
                raise BundleError(f"{path} is {info.st_size} bytes, above max_file_bytes={limits.max_file_bytes}")
            if total + info.st_size > limits.max_total_bytes:
                raise BundleError(f"project exceeds max_total_bytes={limits.max_total_bytes}")
            digest, size = _hash_regular_file(path, info.st_size, info.st_size)
            total += size
            entries.append((rel, size, digest))
    if not entries:
        raise BundleError(f"project {root} contains no files")
    entries.sort()
    return entries


def _requirement_set(values: list[str]) -> set[str]:
    try:
        from packaging.requirements import Requirement
    except ImportError as exc:  # pragma: no cover - present in the MAPLE venvs
        raise BundleError("the `packaging` module is required to compare requirement metadata") from exc
    return {str(Requirement(v)) for v in values}


def _check_distribution(dist_dir: Path, project: dict[str, Any]) -> list[tuple[str, int, str, bytes]]:
    """Read and cross-check the actual MAPLE dist-info against the project's pyproject."""
    if not dist_dir.is_dir() or os.path.islink(dist_dir):
        raise BundleError(f"distribution metadata {dist_dir} is not a real directory")
    out = []
    for name in DIST_REQUIRED + DIST_OPTIONAL:
        path = dist_dir / name
        if not os.path.lexists(path):
            if name in DIST_REQUIRED:
                raise BundleError(f"distribution metadata {dist_dir} lacks required {name}")
            continue
        data = _read_bounded_regular(path, _MAX_DIST_FILE_BYTES, "distribution file")
        out.append((name, len(data), hashlib.sha256(data).hexdigest(), data))
    _check_distribution_bytes({name: data for name, _s, _h, data in out}, project)
    return out


def _check_distribution_bytes(files: dict[str, bytes], project: dict[str, Any]) -> None:
    """Pure cross-check of dist-info bytes against a pyproject [project] table (export and verification)."""
    if not set(DIST_REQUIRED) <= set(files) or not set(files) <= set(DIST_REQUIRED + DIST_OPTIONAL):
        raise BundleError(f"distribution files {sorted(files)} are not the required/optional dist-info set")
    try:  # decoded text, as importlib.metadata reads it (non-ASCII values stay str)
        meta = email.parser.Parser().parsestr(files["METADATA"].decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise BundleError(f"distribution METADATA is not UTF-8: {exc}") from exc
    checks = {
        "Name": (meta.get("Name"), project.get("name")),
        "Version": (meta.get("Version"), project.get("version")),
        "Summary": (meta.get("Summary"), project.get("description")),
        "Requires-Python": (meta.get("Requires-Python"), project.get("requires-python")),
    }
    for field, (got, want) in checks.items():
        if got != want:
            raise BundleError(f"distribution METADATA {field} {got!r} != pyproject value {want!r}")
    optional = project.get("optional-dependencies", {}) or {}
    expected = set(_requirement_set(list(project.get("dependencies", []) or [])))
    for extra, reqs in optional.items():
        expected |= _requirement_set([f'{r}; extra == "{extra}"' for r in reqs])
    if _requirement_set(meta.get_all("Requires-Dist") or []) != expected:
        raise BundleError("distribution METADATA Requires-Dist does not match pyproject dependencies")
    if set(meta.get_all("Provides-Extra") or []) != set(optional):
        raise BundleError("distribution METADATA Provides-Extra does not match pyproject optional groups")
    try:
        top_level = files["top_level.txt"].decode("utf-8").split()
    except UnicodeDecodeError as exc:
        raise BundleError("distribution top_level.txt is not UTF-8") from exc
    if top_level != [EXPECTED_DISTRIBUTION_NAME]:
        raise BundleError("distribution top_level.txt must name exactly 'maple'")


def _check_origin(path: Path, limits: Limits) -> dict[str, Any]:
    return _validate_origin_record(_load_json(path, limits.max_origin_bytes, "origin record"))


def _validate_origin_record(origin: Any) -> dict[str, Any]:
    """Pure schema check of an origin record (export input and every manifest read)."""
    required = {"schema", "description", "original_source", "patch_chain", "patch_chain_absence_reason"}
    if not isinstance(origin, dict) or set(origin) != required:
        raise BundleError(f"origin record must have exactly the keys {sorted(required)}")
    if origin["schema"] != ORIGIN_SCHEMA:
        raise BundleError(f"origin record schema must be {ORIGIN_SCHEMA!r}")
    if not isinstance(origin["description"], str) or not origin["description"].strip():
        raise BundleError("origin record description must be a non-empty string")
    src = origin["original_source"]
    src_keys = {"kind", "revision", "revision_absence_reason", "dirty_snapshot_note"}
    if not isinstance(src, dict) or set(src) != src_keys:
        raise BundleError(f"origin original_source must have exactly the keys {sorted(src_keys)}")
    if not isinstance(src["kind"], str) or not src["kind"].strip():
        raise BundleError("origin original_source.kind must be a non-empty string")
    for key in ("revision", "revision_absence_reason", "dirty_snapshot_note"):
        if src[key] is not None and not isinstance(src[key], str):
            raise BundleError(f"origin original_source.{key} must be a string or null")
    if src["revision"] is None and not (src["revision_absence_reason"] or "").strip():
        raise BundleError("an unknown original revision requires an explicit revision_absence_reason")
    chain = origin["patch_chain"]
    if not isinstance(chain, list):
        raise BundleError("origin patch_chain must be a list")
    for item in chain:
        if not isinstance(item, dict) or set(item) - {"id", "description", "evidence"} or \
                not isinstance(item.get("id"), str) or not isinstance(item.get("description"), str):
            raise BundleError("each patch_chain item needs string id and description (optional evidence list)")
        if "evidence" in item and not (isinstance(item["evidence"], list)
                                       and all(isinstance(e, str) for e in item["evidence"])):
            raise BundleError("patch_chain evidence must be a list of strings")
    reason = origin["patch_chain_absence_reason"]
    if reason is not None and not isinstance(reason, str):
        raise BundleError("origin patch_chain_absence_reason must be a string or null")
    if not chain and not (reason or "").strip():
        raise BundleError("an empty patch_chain requires an explicit patch_chain_absence_reason")
    return origin


class _VerifyingReader:
    """Reads one source file exactly once (no symlink follow), hashing what tarfile copies."""

    def __init__(self, path: Path, size: int):
        self._fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        if not stat.S_ISREG(os.fstat(self._fd).st_mode):
            os.close(self._fd)
            raise SourceChangedError(f"{path} is no longer a regular file")
        self.hasher = hashlib.sha256()
        self.remaining = size

    def read(self, n: int = -1) -> bytes:
        n = self.remaining if n is None or n < 0 else min(n, self.remaining)
        data = os.read(self._fd, n) if n else b""
        self.remaining -= len(data)
        self.hasher.update(data)
        return data

    def finish(self) -> bool:
        extra = os.read(self._fd, 1)
        os.close(self._fd)
        return self.remaining == 0 and extra == b""


def _tarinfo(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = size, 0, 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.type = tarfile.REGTYPE
    return info


def _stream_hash(fd: int, size_bound: int) -> tuple[str, int]:
    os.lseek(fd, 0, os.SEEK_SET)
    hasher, count = hashlib.sha256(), 0
    while True:
        chunk = os.read(fd, _CHUNK)
        if not chunk:
            break
        count += len(chunk)
        if count > size_bound:
            raise BundleError(f"archive exceeds its bound of {size_bound} bytes")
        hasher.update(chunk)
    return hasher.hexdigest(), count


def export_bundle(project_root: str | Path, distribution_dir: str | Path, origin_path: str | Path,
                  output_dir: str | Path, *, limits: Limits | None = None, _after_archive=None) -> dict[str, Any]:
    """Archive the exact project tree into a NEW bundle directory; returns the manifest."""
    limits = limits or Limits()
    root = Path(project_root).resolve()
    output = Path(output_dir).absolute()
    if not root.is_dir() or os.path.islink(project_root):
        raise BundleError(f"project root {project_root} is not a real directory")
    if output == root or output.is_relative_to(root):
        raise BundleError("the bundle output must lie outside the exported project tree")
    pyproject_bytes = _read_bounded_regular(root / "pyproject.toml", 1 << 20, "pyproject.toml")
    try:
        project = tomllib.loads(pyproject_bytes.decode("utf-8")).get("project", {})
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise BundleError(f"pyproject.toml is not valid TOML: {exc}") from exc
    if project.get("name") != EXPECTED_DISTRIBUTION_NAME or not isinstance(project.get("version"), str):
        raise BundleError("pyproject.toml must name project 'maple' with a version")
    _check_version(project["version"])
    origin = _check_origin(Path(origin_path), limits)
    dist = _check_distribution(Path(distribution_dir).resolve(), project)

    before = _walk_project(root, limits)
    paths = {rel for rel, _s, _h in before}
    if "pyproject.toml" not in paths or "src/maple/__init__.py" not in paths:
        raise BundleError("project must contain pyproject.toml and src/maple/__init__.py (src layout)")
    package = source_tree_digest(root / "src" / "maple", max_files=limits.max_files,
                                 max_bytes=limits.max_total_bytes)
    selected = {rel: digest for rel, _s, digest in before}
    for rel, digest in package.files:
        if selected.get(f"src/maple/{rel}") != digest:
            raise SourceChangedError(f"package digest entry src/maple/{rel} disagrees with the project selection")

    staging = _staging_for(output)
    try:
        archive_path = staging / ARCHIVE_NAME
        members: list[dict[str, Any]] = []
        fd = os.open(archive_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "w+b") as handle:
            with tarfile.open(fileobj=handle, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for rel, size, digest in before:
                    reader = _VerifyingReader(root / rel, size)
                    tar.addfile(_tarinfo(SOURCE_PREFIX + rel, size), reader)
                    if not reader.finish() or reader.hasher.hexdigest() != digest:
                        raise SourceChangedError(f"{root / rel} changed while it was being archived")
                    members.append({"path": SOURCE_PREFIX + rel, "size": size, "sha256": digest})
                for name, size, digest, data in dist:
                    tar.addfile(_tarinfo(DIST_PREFIX + name, size), io.BytesIO(data))
                    members.append({"path": DIST_PREFIX + name, "size": size, "sha256": digest})
            handle.flush()
            os.fsync(handle.fileno())
            archive_sha, archive_size = _stream_hash(handle.fileno(), limits.max_archive_bytes)
        if _after_archive is not None:
            _after_archive()
        if _walk_project(root, limits) != before:
            raise SourceChangedError(f"project {root} changed during export; nothing was published")
        if [(n, s, h) for n, s, h, _d in _check_distribution(Path(distribution_dir).resolve(), project)] != \
                [(n, s, h) for n, s, h, _d in dist]:
            raise SourceChangedError("distribution metadata changed during export")
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "tool": {"module": __name__, "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "python": platform.python_version(), "implementation": platform.python_implementation()},
            "limits": asdict(limits),
            "selection_rule": SELECTION_RULE,
            "project": {"name": project["name"], "version": project["version"],
                        "pyproject_sha256": hashlib.sha256(pyproject_bytes).hexdigest(),
                        "exported_from": str(root),
                        "project_source_digest": {"algorithm": PROJECT_DIGEST_ALGORITHM,
                                                  "digest_sha256": _tree_digest(before),
                                                  "file_count": len(before),
                                                  "total_bytes": sum(s for _r, s, _h in before)},
                        "package_relpath": "src/maple",
                        "package_source_digest": package.as_record()},
            "git_supplementary": read_git_head(root),
            "origin": origin,
            "distribution": {"source_dir": str(Path(distribution_dir).resolve()),
                             "files": [n for n, _s, _h, _d in dist]},
            "files": members,
            "file_count": len(members),
            "total_bytes": sum(m["size"] for m in members),
            "archive": {"name": ARCHIVE_NAME, "size": archive_size, "sha256": archive_sha,
                        "format": ARCHIVE_FORMAT},
        }
        manifest["project"]["package_source_digest"]["root"] = "src/maple"
        record = {"bundle_id": hashlib.sha256(_canonical(manifest)).hexdigest(), **manifest}
        fd = os.open(staging / MANIFEST_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        verify_bundle(staging, limits=limits)
        for p in (staging / ARCHIVE_NAME, staging / MANIFEST_NAME):
            os.chmod(p, 0o444)
        os.chmod(staging, 0o755)
        _rename_noreplace(staging, output)
    except BaseException:
        _remove_staging(staging)
        raise
    return record


# --- verification and extraction -------------------------------------------------------------------------
_TOP_KEYS = {"bundle_id", "schema", "tool", "limits", "selection_rule", "project", "git_supplementary", "origin",
             "distribution", "files", "file_count", "total_bytes", "archive"}
_PROJECT_KEYS = {"name", "version", "pyproject_sha256", "exported_from", "project_source_digest",
                 "package_relpath", "package_source_digest"}
_DIGEST_KEYS = {"algorithm", "digest_sha256", "file_count", "total_bytes"}
_GIT_KEYS = {"not_a_git_repository": {"status"}, "unreadable": {"status", "reason"},
             "ok": {"status", "head_commit", "head_ref"}}


def _exact_dict(value: Any, keys: set[str], what: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise BundleError(f"bundle {what} must have exactly the keys {sorted(keys)}")
    return value


def _count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BundleError(f"bundle {what} must be a non-negative integer")
    return value


def _validate_manifest(record: Any, limits: Limits) -> dict[str, dict[str, Any]]:
    """Strict, pure schema AND internal-consistency validation of a parsed manifest; returns {path: entry}.

    Consistency, not authenticity: every declared field must agree with the member table and this tool's
    documented schema, but anyone able to rewrite the bundle can still produce a consistent forgery."""
    _exact_dict(record, _TOP_KEYS, "manifest")
    if record["schema"] != SCHEMA:
        raise BundleError(f"unsupported bundle schema {record['schema']!r} (this tool reads {SCHEMA!r})")
    body = {k: v for k, v in record.items() if k != "bundle_id"}
    if _sha256_hex(record["bundle_id"]) != hashlib.sha256(_canonical(body)).hexdigest():
        raise BundleError("bundle_id does not match the canonical manifest content")
    tool = _exact_dict(record["tool"], {"module", "module_sha256", "python", "implementation"}, "tool record")
    if not all(isinstance(tool[k], str) and tool[k] for k in ("module", "python", "implementation")):
        raise BundleError("bundle tool record fields must be non-empty strings")
    _sha256_hex(tool["module_sha256"])
    declared = _exact_dict(record["limits"], set(asdict(limits)), "limits")
    Limits(**declared)  # positive integers
    if record["selection_rule"] != SELECTION_RULE:
        raise BundleError("bundle selection_rule is not this tool's documented rule")
    git = record["git_supplementary"]
    if not isinstance(git, dict) or git.get("status") not in _GIT_KEYS or set(git) != _GIT_KEYS[git["status"]] \
            or not all(isinstance(v, str) or v is None for v in git.values()):
        raise BundleError("bundle git_supplementary record is malformed")
    _validate_origin_record(record["origin"])

    files = record["files"]
    if not isinstance(files, list) or not 0 < len(files) <= limits.max_files or \
            len(files) != _count(record["file_count"], "file_count"):
        raise BundleError("bundle file list is empty, malformed or above max_files")
    table: dict[str, dict[str, Any]] = {}
    total = 0
    previous = ""
    for entry in files:
        _exact_dict(entry, {"path", "size", "sha256"}, "file entry")
        path = _check_relpath(entry["path"])
        if not path.startswith((SOURCE_PREFIX, DIST_PREFIX)):
            raise BundleError(f"bundle member {path!r} is outside source/ and distribution/")
        if path in table:
            raise BundleError(f"duplicate bundle member {path!r}")
        if path.startswith(SOURCE_PREFIX):
            if path <= previous:
                raise BundleError("source members are not strictly sorted")
            previous = path
        size = _count(entry["size"], f"size of {path!r}")
        if size > limits.max_file_bytes:
            raise BundleError(f"bundle member {path!r} is above max_file_bytes")
        _sha256_hex(entry["sha256"])
        total += size
        table[path] = entry
    if total != _count(record["total_bytes"], "total_bytes") or total > limits.max_total_bytes:
        raise BundleError("bundle total_bytes is inconsistent or above max_total_bytes")

    dist = _exact_dict(record["distribution"], {"source_dir", "files"}, "distribution record")
    names = dist["files"]
    if not isinstance(dist["source_dir"], str) or not isinstance(names, list) or \
            not all(isinstance(n, str) for n in names) or len(set(names)) != len(names) or \
            {DIST_PREFIX + n for n in names} != {p for p in table if p.startswith(DIST_PREFIX)} or \
            not set(DIST_REQUIRED) <= set(names) or not set(names) <= set(DIST_REQUIRED + DIST_OPTIONAL):
        raise BundleError("bundle distribution file list is inconsistent")
    for n in names:
        if table[DIST_PREFIX + n]["size"] > _MAX_DIST_FILE_BYTES:
            raise BundleError(f"distribution member {n} is above its {_MAX_DIST_FILE_BYTES}-byte bound")

    archive = _exact_dict(record["archive"], {"name", "size", "sha256", "format"}, "archive record")
    if archive["name"] != ARCHIVE_NAME or archive["format"] != ARCHIVE_FORMAT:
        raise BundleError("bundle archive record is malformed")
    if not 0 < _count(archive["size"], "archive size") <= limits.max_archive_bytes:
        raise BundleError("bundle archive size is invalid or above its bound")
    _sha256_hex(archive["sha256"])

    project = _exact_dict(record["project"], _PROJECT_KEYS, "project record")
    if project["name"] != EXPECTED_DISTRIBUTION_NAME or not isinstance(project["exported_from"], str):
        raise BundleError("bundle project record names a project other than 'maple' or is malformed")
    _check_version(project["version"])
    for key in ("source/pyproject.toml", "source/src/maple/__init__.py"):
        if key not in table:
            raise BundleError(f"bundle lacks {key}")
    if table["source/pyproject.toml"]["size"] > _MAX_PYPROJECT_BYTES:
        raise BundleError("bundle pyproject.toml is above its bound")
    if table["source/pyproject.toml"]["sha256"] != _sha256_hex(project["pyproject_sha256"]):
        raise BundleError("bundle pyproject_sha256 disagrees with its file entry")
    source = [(p[len(SOURCE_PREFIX):], e["size"], e["sha256"]) for p, e in table.items() if p.startswith(SOURCE_PREFIX)]
    expected = {"algorithm": PROJECT_DIGEST_ALGORITHM, "digest_sha256": _tree_digest(source),
                "file_count": len(source), "total_bytes": sum(s for _r, s, _h in source)}
    if _exact_dict(project["project_source_digest"], _DIGEST_KEYS, "project_source_digest") != expected:
        raise BundleError("bundle project_source_digest disagrees with its source members")
    if project["package_relpath"] != "src/maple":
        raise BundleError("bundle package_relpath must be 'src/maple'")
    prefix = SOURCE_PREFIX + "src/maple/"
    package = [(p[len(prefix):], e["size"], e["sha256"]) for p, e in table.items() if p.startswith(prefix)]
    expected_pkg = {"root": "src/maple", "algorithm": SOURCE_DIGEST_ALGORITHM,
                    "digest_sha256": _tree_digest(package), "file_count": len(package),
                    "total_bytes": sum(s for _r, s, _h in package)}
    if _exact_dict(project["package_source_digest"], _DIGEST_KEYS | {"root"}, "package_source_digest") != expected_pkg:
        raise BundleError("bundle package_source_digest disagrees with its src/maple members")
    return table


def _capture_names(record: dict[str, Any]) -> set[str]:
    """The small records whose actual bytes are cross-checked semantically (bounded by the manifest checks)."""
    return {"source/pyproject.toml"} | {DIST_PREFIX + n for n in record["distribution"]["files"]}


def _semantic_check(record: dict[str, Any], captured: dict[str, bytes]) -> None:
    """The manifest's project identity must equal the ACTUAL archived pyproject.toml and dist-info, using the
    same cross-checks as export."""
    try:
        project = tomllib.loads(captured["source/pyproject.toml"].decode("utf-8")).get("project", {})
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise BundleError(f"archived pyproject.toml is missing or invalid: {exc}") from exc
    if project.get("name") != record["project"]["name"] or project.get("version") != record["project"]["version"]:
        raise BundleError(f"manifest project {record['project']['name']!r} {record['project']['version']!r} "
                          f"contradicts the archived pyproject.toml ({project.get('name')!r} "
                          f"{project.get('version')!r})")
    _check_distribution_bytes({n: captured[DIST_PREFIX + n] for n in record["distribution"]["files"]}, project)


def _iterate_archive(fd: int, record: dict[str, Any], table: dict[str, dict[str, Any]],
                     sink: Path | None) -> dict[str, bytes]:
    """Stream every member, enforcing exact names/types/metadata/sizes/hashes; optionally write into `sink`.
    Returns the verified bytes of the small semantic records (`_capture_names`)."""
    os.lseek(fd, 0, os.SEEK_SET)
    seen: set[str] = set()
    capture = _capture_names(record)
    captured: dict[str, bytes] = {}
    handle = os.fdopen(os.dup(fd), "rb")
    try:
        with tarfile.open(fileobj=handle, mode="r:") as tar:
            while True:
                try:
                    member = tar.next()
                except tarfile.TarError as exc:
                    raise BundleError(f"malformed archive: {exc}") from exc
                if member is None:
                    break
                name = member.name
                if len(seen) >= len(table):
                    raise BundleError(f"archive has more members than the manifest ({name!r})")
                _check_relpath(name)
                if name in seen:
                    raise BundleError(f"duplicate archive member {name!r}")
                if name not in table:
                    raise BundleError(f"archive member {name!r} is not in the manifest")
                if not member.isreg() or member.type != tarfile.REGTYPE:
                    raise BundleError(f"archive member {name!r} is not a regular file")
                if member.linkname or member.mode != 0o644 or member.mtime != 0 or member.uid != 0 or \
                        member.gid != 0 or member.uname or member.gname:
                    raise BundleError(f"archive member {name!r} is not normalized")
                if set(member.pax_headers) - {"path"}:
                    raise BundleError(f"archive member {name!r} carries unexpected pax headers")
                entry = table[name]
                if member.size != entry["size"]:
                    raise BundleError(f"archive member {name!r} size {member.size} != manifest {entry['size']}")
                seen.add(name)
                src = tar.extractfile(member)
                if src is None:
                    raise BundleError(f"archive member {name!r} has no data")
                out_fd = None
                if sink is not None:
                    out_fd = os.open(_confined(sink, name), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     0o644)
                keep = [] if name in capture else None
                hasher, remaining = hashlib.sha256(), member.size
                try:
                    while remaining:
                        chunk = src.read(min(_CHUNK, remaining))
                        if not chunk:
                            raise BundleError(f"archive member {name!r} is truncated")
                        remaining -= len(chunk)
                        hasher.update(chunk)
                        if out_fd is not None:
                            _write_all(out_fd, chunk)
                        if keep is not None:
                            keep.append(chunk)
                finally:
                    if out_fd is not None:
                        os.close(out_fd)
                if hasher.hexdigest() != entry["sha256"]:
                    raise BundleError(f"archive member {name!r} does not match its manifest sha256")
                if keep is not None:
                    captured[name] = b"".join(keep)
    finally:
        handle.close()
    if seen != set(table):
        missing = sorted(set(table) - seen)[:5]
        raise BundleError(f"archive lacks manifest members, e.g. {missing}")
    return captured


def _open_verified_archive(bundle: Path, record: dict[str, Any], limits: Limits) -> int:
    path = bundle / ARCHIVE_NAME
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise BundleError(f"{path} is not a regular file")
    if info.st_size != record["archive"]["size"]:
        raise BundleError(f"{path} is {info.st_size} bytes, manifest says {record['archive']['size']}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        digest, _ = _stream_hash(fd, min(limits.max_archive_bytes, record["archive"]["size"]))
        if digest != record["archive"]["sha256"]:
            raise BundleError(f"{path} does not match the manifest archive sha256")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _load_bundle(bundle_dir: str | Path, limits: Limits) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    bundle = Path(bundle_dir)
    if not bundle.is_dir() or os.path.islink(bundle):
        raise BundleError(f"bundle {bundle} is not a real directory")
    names = sorted(os.listdir(bundle))
    if names != sorted([ARCHIVE_NAME, MANIFEST_NAME]):
        raise BundleError(f"bundle {bundle} must contain exactly {ARCHIVE_NAME} and {MANIFEST_NAME}, found {names}")
    record = _load_json(bundle / MANIFEST_NAME, limits.max_manifest_bytes, "bundle manifest")
    return bundle, record, _validate_manifest(record, limits)


def verify_bundle(bundle_dir: str | Path, *, limits: Limits | None = None) -> dict[str, Any]:
    """Verify manifest schema/consistency, archive digest, every member and the archived project identity."""
    limits = limits or Limits()
    bundle, record, table = _load_bundle(bundle_dir, limits)
    fd = _open_verified_archive(bundle, record, limits)
    try:
        captured = _iterate_archive(fd, record, table, None)
    finally:
        os.close(fd)
    _semantic_check(record, captured)
    return record


def _overlay_files(record: dict[str, Any], target: Path, dist_bytes: dict[str, bytes]) -> dict[str, bytes]:
    """The editable overlay (relative to `<target>/editable`) as exact bytes; every name is validated."""
    name = _safe_component(record["project"]["name"], "project name")
    version = _check_version(record["project"]["version"])
    info_dir = _safe_component(f"{name}-{version}.dist-info", "dist-info directory")
    pth = _safe_component(f"__editable__.{name}-{version}.pth", "editable .pth name")
    source_root = target / "source"
    files = {f"{info_dir}/{_safe_component(n, 'distribution file')}": data for n, data in dist_bytes.items()}
    files[f"{info_dir}/INSTALLER"] = (INSTALLER_NAME + "\n").encode()
    files[f"{info_dir}/direct_url.json"] = json.dumps(
        {"dir_info": {"editable": True}, "url": "file://" + quote(str(source_root))}, sort_keys=True).encode()
    files[pth] = (str(source_root / "src") + "\n").encode()
    lines = []
    for rel in sorted(files):
        digest = base64.urlsafe_b64encode(hashlib.sha256(files[rel]).digest()).rstrip(b"=").decode()
        lines.append(f"{rel},sha256={digest},{len(files[rel])}")
    lines.append(f"{info_dir}/RECORD,,")
    files[f"{info_dir}/RECORD"] = ("\n".join(lines) + "\n").encode()
    for rel in files:
        _check_relpath(rel)
    return files


def runtime_environment(target: str | Path, syrup_src: str | Path) -> dict[str, Any]:
    """Structured invocation data (never executed here) for the existing MAPLE venvs."""
    target = Path(target).resolve()
    return {
        "python": ["/home/okin/MAPLE/.venv/bin/python", "/home/okin/MAPLE/.venv-cupy/bin/python"],
        "env": {"PYTHONPATH": os.pathsep.join([str(target / "editable"), str(target / "source" / "src"),
                                               str(Path(syrup_src).resolve())]),
                EXPECTED_MAPLE_ROOT_ENV: str(target / "source")},
        "note": "Use ONLY an existing MAPLE venv; nothing is installed. The overlay's dist-info precedes the "
                "venv's own `maple` distribution on sys.path, so the resolver identifies the cache.",
    }


def _materialization_record(record: dict[str, Any], target: Path) -> bytes:
    return _canonical({"schema": MATERIALIZATION_SCHEMA, "bundle_id": record["bundle_id"],
                       "target": str(target), "layout": {"source": "source/", "distribution": "distribution/",
                                                         "overlay": "editable/", "manifest": MANIFEST_NAME}}) + b"\n"


def _freeze_tree(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            os.chmod(os.path.join(dirpath, name), 0o444)
        os.chmod(dirpath, 0o555)


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        _write_all(fd, data)
    finally:
        os.close(fd)


def materialize_bundle(bundle_dir: str | Path, cache_root: str | Path, *,
                       limits: Limits | None = None) -> Path:
    """Verify and extract into `<cache_root>/<bundle_id>` with an editable overlay; no-clobber.

    Every write goes beneath a private staging directory; the COMPLETE staged tree is verified against the
    intended final target identity before the atomic no-replace rename. An existing target is re-verified
    and returned unchanged (a corrupt one is refused, never rewritten or removed)."""
    limits = limits or Limits()
    bundle, record, table = _load_bundle(bundle_dir, limits)
    cache = Path(cache_root).resolve()
    target = cache / _sha256_hex(record["bundle_id"])
    if os.path.lexists(target):
        verify_materialized(target, limits=limits)
        return target
    fd = _open_verified_archive(bundle, record, limits)
    try:
        staging = _staging_for(target)
        try:
            extracted = staging / "extract"
            os.mkdir(extracted, 0o755)
            captured = _iterate_archive(fd, record, table, extracted)
            _semantic_check(record, captured)  # before ANY name derived from the manifest is used
            os.rename(extracted / "source", staging / "source")
            os.rename(extracted / "distribution", staging / "distribution")
            os.rmdir(extracted)
            dist_bytes = {n: captured[DIST_PREFIX + n] for n in record["distribution"]["files"]}
            for rel, data in _overlay_files(record, target, dist_bytes).items():
                _write_new(_confined(staging / "editable", rel), data)
            _write_new(staging / MANIFEST_NAME, _canonical(record) + b"\n")
            _write_new(staging / MATERIALIZATION_NAME, _materialization_record(record, target))
            _freeze_tree(staging)
            _verify_tree(staging, target, record, table, limits)  # full staged verification, final identity
            _rename_noreplace(staging, target)
        except BaseException:
            _remove_staging(staging)
            raise
    finally:
        os.close(fd)
    verify_materialized(target, limits=limits)  # observation only; the staged tree was already verified
    return target


def _check_materialized_content(root: Path, record: dict[str, Any], limits: Limits) -> None:
    source = root / "source"
    walked = _walk_project(source, limits)
    expected = sorted((e["path"][len(SOURCE_PREFIX):], e["size"], e["sha256"])
                      for e in record["files"] if e["path"].startswith(SOURCE_PREFIX))
    if walked != expected:
        raise BundleError(f"materialized source under {source} does not match the manifest file set")
    if _tree_digest(walked) != record["project"]["project_source_digest"]["digest_sha256"]:
        raise BundleError("materialized project source digest differs from the manifest")
    package = source_tree_digest(source / "src" / "maple", max_files=limits.max_files,
                                 max_bytes=limits.max_total_bytes)
    if package.digest_sha256 != record["project"]["package_source_digest"]["digest_sha256"]:
        raise BundleError("materialized package source digest differs from the manifest")


def _verify_tree(location: Path, identity: Path, record: dict[str, Any], table: dict[str, dict[str, Any]],
                 limits: Limits) -> None:
    """Verify a complete materialized tree found at `location` whose final identity is `identity`
    (the staging directory before publication, or the published target itself)."""
    top = sorted(os.listdir(location))
    if top != sorted(["distribution", "editable", "source", MANIFEST_NAME, MATERIALIZATION_NAME]):
        raise BundleError(f"materialized tree has unexpected entries {top}")
    _check_materialized_content(location, record, limits)
    if sorted(os.listdir(location / "distribution")) != sorted(record["distribution"]["files"]):
        raise BundleError("materialized distribution directory has unexpected entries")
    dist_bytes = {}
    for n in record["distribution"]["files"]:
        data = _read_bounded_regular(location / "distribution" / n, _MAX_DIST_FILE_BYTES, "distribution file")
        if hashlib.sha256(data).hexdigest() != table[DIST_PREFIX + n]["sha256"] or \
                len(data) != table[DIST_PREFIX + n]["size"]:
            raise BundleError(f"materialized distribution/{n} differs from the manifest")
        dist_bytes[n] = data
    pyproject = _read_bounded_regular(location / "source" / "pyproject.toml", _MAX_PYPROJECT_BYTES, "pyproject.toml")
    _semantic_check(record, {"source/pyproject.toml": pyproject,
                             **{DIST_PREFIX + n: d for n, d in dist_bytes.items()}})
    overlay = _overlay_files(record, identity, dist_bytes)
    present = set()
    for dirpath, _dirnames, filenames in os.walk(location / "editable"):
        for name in filenames:
            present.add(Path(dirpath, name).relative_to(location / "editable").as_posix())
    if present != set(overlay):
        raise BundleError("materialized overlay file set differs from the expected overlay")
    for rel, data in overlay.items():
        if _read_bounded_regular(location / "editable" / rel, _MAX_DIST_FILE_BYTES, "overlay file") != data:
            raise BundleError(f"materialized overlay file {rel} differs from the expected bytes")
    if _read_bounded_regular(location / MANIFEST_NAME, limits.max_manifest_bytes, "materialized manifest") != \
            _canonical(record) + b"\n":
        raise BundleError("materialized manifest copy differs from the bundle manifest")
    if _read_bounded_regular(location / MATERIALIZATION_NAME, 1 << 16, "materialization record") != \
            _materialization_record(record, identity):
        raise BundleError("materialization record differs")


def verify_materialized(target: str | Path, *, limits: Limits | None = None) -> dict[str, Any]:
    """Re-verify a published materialized cache: exact file set, every byte, overlay, records, identity."""
    limits = limits or Limits()
    target = Path(target).resolve()
    record = _load_json(target / MANIFEST_NAME, limits.max_manifest_bytes, "materialized manifest")
    table = _validate_manifest(record, limits)
    if target.name != record["bundle_id"]:
        raise BundleError(f"materialized directory name {target.name} is not its bundle_id")
    _verify_tree(target, target, record, table, limits)
    return record


def check_runtime(target: str | Path, *, limits: Limits | None = None) -> dict[str, Any]:
    """In THIS process: resolve MAPLE with the existing resolver and require the materialized cache."""
    target = Path(target).resolve()
    record = verify_materialized(target, limits=limits)
    dependency = resolve_maple_dependency(target / "source")
    if dependency.source_root is None or dependency.source_root.resolve() != (target / "source"):
        raise BundleError(f"resolved MAPLE source root {dependency.source_root} is not {target / 'source'}")
    if dependency.source_kind != "editable_working_tree" or dependency.distribution_version != \
            record["project"]["version"]:
        raise BundleError("resolved MAPLE distribution is not the materialized overlay")
    package = source_tree_digest(dependency.package_dir)
    if package.digest_sha256 != record["project"]["package_source_digest"]["digest_sha256"]:
        raise BundleError("imported MAPLE package digest differs from the bundle")
    return {"bundle_id": record["bundle_id"], "dependency": dependency.as_record(),
            "package_source_digest": package.digest_sha256}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m maple_syrup.dependency_bundle", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--project", required=True)
    p.add_argument("--distribution-metadata", required=True)
    p.add_argument("--origin", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("verify")
    p.add_argument("bundle")
    p = sub.add_parser("materialize")
    p.add_argument("bundle")
    p.add_argument("--cache", required=True)
    p = sub.add_parser("verify-materialized")
    p.add_argument("target")
    p = sub.add_parser("env")
    p.add_argument("target")
    p.add_argument("--syrup-src", required=True)
    p = sub.add_parser("check-runtime")
    p.add_argument("target")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            record = export_bundle(args.project, args.distribution_metadata, args.origin, args.output)
            out = {"bundle_id": record["bundle_id"], "output": str(Path(args.output).absolute())}
        elif args.command == "verify":
            out = {"verified": verify_bundle(args.bundle)["bundle_id"]}
        elif args.command == "materialize":
            target = materialize_bundle(args.bundle, args.cache)
            out = {"target": str(target), **runtime_environment(target, Path(__file__).resolve().parents[1])}
        elif args.command == "verify-materialized":
            out = {"verified": verify_materialized(args.target)["bundle_id"]}
        elif args.command == "env":
            verify_materialized(args.target)
            out = runtime_environment(args.target, args.syrup_src)
        else:
            out = check_runtime(args.target)
    except (BundleError, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
