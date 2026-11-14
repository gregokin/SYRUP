"""Strict portable whole-Phase-2.5 SOURCE-IDENTITY profile (`syrup.phase25.source_profile/1`). TEST-ONLY helper, uniquely named so it imports under importlib mode without a global conftest.

What it proves: the actual `src/maple_syrup` bytes (EVERY non-cache file, read cold on each call, no id/mtime cache) are EXACTLY the Root-supplied source lock. The lock is a SOURCE identity for qualification
(`SOURCE_LOCKED_FOR_QUALIFICATION`, `whole_phase_accepted: false`); it is NOT a physics acceptance. The proof is chained to literal pins in `p25_source_pins.py`:
  base169 manifest (immutable 02388ad listing, file sha pinned) + the 953 WH1 input proof (a289...) + the final lock (file sha, tree digest, counts, changed/added names pinned) + every actual file.
A forged but self-consistent lock, an edited / extra / missing file, a modified proof, an unknown schema/status/version or a missing proof all REFUSE. Nothing is skipped; nothing reads Git, the environment,
an absolute task path or an ambient HEAD. Historical packages are admitted only by an exact literal historical digest and only to their own historical assertions."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

PROFILE = "syrup.phase25.source_profile/1"
LOCK_SCHEMA = "syrup.phase25.source_lock/1"
LOCK_STATUS = "SOURCE_LOCKED_FOR_QUALIFICATION"
WH1_PROOF_SCHEMA = "syrup.phase25.wh1_input_proof/1"
SENTINEL = "REQUIRES_ROOT_FINAL_LOCK"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "phase25"
BASE_KEYS = frozenset({"git_base", "files", "digest", "file_count"})
WH1_KEYS = frozenset({"schema", "git_commit", "path", "sha256", "origin"})
LOCK_KEYS = frozenset({"schema", "status", "whole_phase_accepted", "git_base", "wh1", "files", "changed", "added", "file_count", "total_bytes", "tree_digest_sha256"})
LOCK_WH1_KEYS = frozenset({"path", "input_sha256", "sha256"})
LOCK_CHANGED_KEYS = frozenset({"path", "base_sha256", "sha256"})
LOCK_ADDED_KEYS = frozenset({"path", "sha256"})
_HEX = re.compile(r"[0-9a-f]{64}")
_GIT = re.compile(r"[0-9a-f]{40}")
_MAX_PATH = 200
_MAX_FILES = 4096


class SourceProfileError(Exception):
    """Any source-identity refusal (unknown/edited/extra/missing source, malformed or forged proof)."""


class MissingProof(SourceProfileError):
    """A required shipped proof (fixture / lock) is absent: a hard failure, never a skip."""


class RequiresRootFinalLock(SourceProfileError):
    """The literal final-lock pins are still the sentinel: Root has not locked the final source."""


class UnknownSource(SourceProfileError):
    """The actual package is neither a pinned historical package nor the pinned final package."""


@dataclass(frozen=True)
class VerifiedSource:
    kind: str                      # "final"
    tree_digest: str
    files: Any                     # read-only path -> sha256 of every actual file
    base: Any                      # read-only path -> sha256 of the 169-file immutable base
    changed: frozenset             # base paths whose bytes changed (vs base169)
    added: frozenset               # paths not in base169, other than the WH1 module
    wh1_path: str
    wh1_input_sha256: str
    wh1_sha256: str
    unchanged: frozenset
    lock_sha256: str


@dataclass(frozen=True)
class Branch:
    kind: str                      # "historical" | "final"
    name: str                      # historical pin name, or "final"
    verified: Any = None           # VerifiedSource for "final"


# ------------------------------------------------------------------ strict primitives
def _no_dups(pairs: list) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise SourceProfileError(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def _refuse_constant(name: str) -> Any:
    raise SourceProfileError(f"non-finite JSON constant {name!r}")


def strict_json(data: bytes, what: str) -> Any:
    if not isinstance(data, (bytes, bytearray)):
        raise SourceProfileError(f"{what}: bytes expected")
    try:
        return json.loads(bytes(data).decode("utf-8"), object_pairs_hook=_no_dups, parse_constant=_refuse_constant)
    except SourceProfileError:
        raise
    except (ValueError, UnicodeDecodeError) as exc:
        raise SourceProfileError(f"{what}: not strict JSON ({exc})") from exc


def _int(value: Any, what: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:        # bool / float / str / negative refuse
        raise SourceProfileError(f"{what} must be an exact int >= {minimum}, got {value!r}")
    return value


def _hex(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise SourceProfileError(f"{what} must be 64 lowercase hex characters, got {value!r}")
    return value


def _closed(obj: Any, keys: frozenset, what: str) -> dict:
    if not isinstance(obj, dict) or set(obj) != set(keys):
        extra = sorted(set(obj) - set(keys)) if isinstance(obj, dict) else "not an object"
        missing = sorted(set(keys) - set(obj)) if isinstance(obj, dict) else ""
        raise SourceProfileError(f"{what}: keys must be exactly {sorted(keys)} (extra {extra}, missing {missing})")
    return obj


def _safe_path(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_PATH or "\x00" in value or "\\" in value:
        raise SourceProfileError(f"{what}: unsafe path {value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or p.as_posix() != value or any(part in ("", ".", "..") for part in p.parts) or "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
        raise SourceProfileError(f"{what}: unsafe or non-canonical path {value!r}")
    return value


def _file_table(rows: Any, what: str) -> dict:
    if not isinstance(rows, list) or not rows or len(rows) > _MAX_FILES:
        raise SourceProfileError(f"{what}: a bounded non-empty list is required")
    out: dict = {}
    folded: set = set()
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            raise SourceProfileError(f"{what}: each entry must be [path, sha256]")
        path = _safe_path(row[0], what)
        if path in out or path.casefold() in folded:
            raise SourceProfileError(f"{what}: duplicate path {path!r}")
        folded.add(path.casefold())
        out[path] = _hex(row[1], f"{what} {path}")
    if [r[0] for r in rows] != sorted(out):
        raise SourceProfileError(f"{what}: entries must be sorted by path")
    return out


def combine_digest(table: dict) -> str:
    """The existing `provenance.source_tree_digest` combination: sha256 over sorted `relpath NUL file-sha256 LF`."""
    h = hashlib.sha256()
    for rel in sorted(table):
        h.update(rel.encode("utf-8") + b"\0" + table[rel].encode("ascii") + b"\n")
    return h.hexdigest()


def read_fixture(name: str, fixtures: Path | None = None) -> bytes:
    path = (FIXTURES if fixtures is None else Path(fixtures)) / name
    if not path.is_file():
        raise MissingProof(f"the shipped proof {name!r} is missing from {path.parent}")
    return path.read_bytes()


def actual_listing(src_dir: Any) -> tuple:
    """(digest, {path: sha256}, total_bytes) of EVERY non-cache regular file under `src_dir`, read cold from the bytes (the repository's own `provenance.source_tree_digest`: symlinks / special files refused)."""
    from maple_syrup import provenance
    try:
        d = provenance.source_tree_digest(src_dir, max_files=_MAX_FILES)
    except provenance.ProvenanceScopeError as exc:
        raise SourceProfileError(f"the actual source scope is not admissible: {exc}") from exc
    return d.digest_sha256, dict(d.files), d.total_bytes


# ------------------------------------------------------------------ the pinned proofs
def load_base(base_bytes: bytes, pins: Any) -> dict:
    if hashlib.sha256(base_bytes).hexdigest() != pins.BASE_MANIFEST_SHA256:
        raise SourceProfileError("the base169 manifest bytes differ from the literal pin")
    d = _closed(strict_json(base_bytes, "base169 manifest"), BASE_KEYS, "base169 manifest")
    if d["git_base"] != pins.BASE_GIT or not _GIT.fullmatch(str(d["git_base"])):
        raise SourceProfileError("the base169 manifest is not the pinned base commit")
    table = _file_table(d["files"], "base169 files")
    if _int(d["file_count"], "base169 file_count", minimum=1) != len(table) or len(table) != pins.BASE_FILE_COUNT:
        raise SourceProfileError("the base169 file count is not the pinned count")
    if _hex(d["digest"], "base169 digest") != combine_digest(table) or d["digest"] != pins.BASE_TREE_DIGEST_SHA256:
        raise SourceProfileError("the base169 tree digest is not the recomputed / pinned digest")
    return table


def load_wh1_proof(proof_bytes: bytes, pins: Any, base: dict) -> str:
    if hashlib.sha256(proof_bytes).hexdigest() != pins.WH1_PROOF_SHA256:
        raise SourceProfileError("the WH1 input-proof bytes differ from the literal pin")
    d = _closed(strict_json(proof_bytes, "WH1 proof"), WH1_KEYS, "WH1 proof")
    if d["schema"] != WH1_PROOF_SCHEMA or d["git_commit"] != pins.WH1_GIT or d["path"] != pins.WH1_PATH or not isinstance(d["origin"], str) or not d["origin"]:
        raise SourceProfileError("the WH1 proof is not the pinned 953 input proof")
    if _hex(d["sha256"], "WH1 input sha256") != pins.WH1_INPUT_SHA256:
        raise SourceProfileError("the WH1 input sha256 is not the literal pin")
    if pins.WH1_PATH in base:
        raise SourceProfileError("the WH1 module must not be part of the base169 listing")
    return d["sha256"]


def final_pins_ready(pins: Any) -> None:
    """The literal final-lock pins must be real: the sentinel (not yet locked by Root) is a HARD failure."""
    for name in ("LOCK_SHA256", "TREE_DIGEST_SHA256", "FILE_COUNT", "TOTAL_BYTES"):
        if getattr(pins, name, SENTINEL) == SENTINEL:
            raise RequiresRootFinalLock(f"{name} is {SENTINEL}: Root has not supplied the final source lock; the guard refuses (no skip, no fallback)")
    _hex(pins.LOCK_SHA256, "pinned lock sha256")
    _hex(pins.TREE_DIGEST_SHA256, "pinned tree digest")
    _int(pins.FILE_COUNT, "pinned file count", minimum=1)
    _int(pins.TOTAL_BYTES, "pinned total bytes", minimum=1)
    for name in ("CHANGED", "ADDED"):
        names = getattr(pins, name)
        if not isinstance(names, frozenset) or not names or not all(isinstance(n, str) for n in names):
            raise SourceProfileError(f"pinned {name} must be a non-empty frozenset of names")


def verify_final(src_dir: Any, pins: Any, *, fixtures: Path | None = None) -> VerifiedSource:
    final_pins_ready(pins)
    base = load_base(read_fixture("base169_manifest.json", fixtures), pins)
    wh1_input = load_wh1_proof(read_fixture("wh1_953_input_proof.json", fixtures), pins, base)
    lock_bytes = read_fixture("final_lock.json", fixtures)
    if hashlib.sha256(lock_bytes).hexdigest() != pins.LOCK_SHA256:
        raise SourceProfileError("the final lock bytes differ from the literal pin (a self-consistent forged lock is refused)")
    lock = _closed(strict_json(lock_bytes, "final lock"), LOCK_KEYS, "final lock")
    if lock["schema"] != LOCK_SCHEMA or lock["status"] != LOCK_STATUS:
        raise SourceProfileError(f"the lock schema/status is not {LOCK_SCHEMA} / {LOCK_STATUS}")
    if lock["whole_phase_accepted"] is not False:
        raise SourceProfileError("a source lock never accepts the whole phase: whole_phase_accepted must be exactly false")
    if lock["git_base"] != pins.BASE_GIT:
        raise SourceProfileError("the lock is not on the pinned base commit")
    locked = _file_table(lock["files"], "lock files")
    if _int(lock["file_count"], "lock file_count", minimum=1) != len(locked) or len(locked) != pins.FILE_COUNT:
        raise SourceProfileError("the lock file count is not the pinned count")
    if _int(lock["total_bytes"], "lock total_bytes", minimum=1) != pins.TOTAL_BYTES:
        raise SourceProfileError("the lock total bytes are not the pinned total")
    if _hex(lock["tree_digest_sha256"], "lock tree digest") != combine_digest(locked) or lock["tree_digest_sha256"] != pins.TREE_DIGEST_SHA256:
        raise SourceProfileError("the lock tree digest is not the recomputed / pinned digest")
    # the lock's own classification against base169 and the WH1 proof
    wh1 = _closed(lock["wh1"], LOCK_WH1_KEYS, "lock wh1")
    if wh1["path"] != pins.WH1_PATH or _hex(wh1["input_sha256"], "lock wh1 input") != wh1_input or locked.get(pins.WH1_PATH) != _hex(wh1["sha256"], "lock wh1 sha256"):
        raise SourceProfileError("the lock's WH1 record is not the pinned 953 input proof / its locked file")
    changed = {}
    for row in _rows(lock["changed"], "lock changed"):
        r = _closed(row, LOCK_CHANGED_KEYS, "lock changed entry")
        changed[_safe_path(r["path"], "lock changed")] = (_hex(r["base_sha256"], "changed base"), _hex(r["sha256"], "changed sha256"))
    added = {}
    for row in _rows(lock["added"], "lock added"):
        r = _closed(row, LOCK_ADDED_KEYS, "lock added entry")
        added[_safe_path(r["path"], "lock added")] = _hex(r["sha256"], "added sha256")
    derived_changed = {p: (base[p], locked[p]) for p in base if p in locked and locked[p] != base[p]}
    derived_added = {p: h for p, h in locked.items() if p not in base and p != pins.WH1_PATH}
    if set(base) - set(locked):
        raise SourceProfileError(f"the lock drops base modules: {sorted(set(base) - set(locked))}")
    if changed != derived_changed or added != derived_added:
        raise SourceProfileError("the lock's changed/added classification does not match base169 + the locked files")
    if frozenset(changed) != pins.CHANGED or frozenset(added) != pins.ADDED:
        raise SourceProfileError("the lock's changed/added names are not the pinned sets")
    # the ACTUAL bytes (cold): exact listing, digest and byte total
    digest, actual, total = actual_listing(src_dir)
    if actual != locked:
        missing, extra = sorted(set(locked) - set(actual)), sorted(set(actual) - set(locked))
        edited = sorted(p for p in set(actual) & set(locked) if actual[p] != locked[p])
        raise SourceProfileError(f"the actual source is not the locked source: missing {missing}, extra {extra}, edited {edited}")
    if digest != pins.TREE_DIGEST_SHA256 or total != pins.TOTAL_BYTES or len(actual) != pins.FILE_COUNT:
        raise SourceProfileError("the actual tree digest / byte total / count is not the pinned one")
    return VerifiedSource(kind="final", tree_digest=digest, files=MappingProxyType(dict(actual)), base=MappingProxyType(dict(base)), changed=frozenset(changed), added=frozenset(added), wh1_path=pins.WH1_PATH,
                          wh1_input_sha256=wh1_input, wh1_sha256=locked[pins.WH1_PATH], unchanged=frozenset(p for p in base if p not in changed), lock_sha256=pins.LOCK_SHA256)


def _rows(value: Any, what: str) -> list:
    if not isinstance(value, list) or len(value) > _MAX_FILES:
        raise SourceProfileError(f"{what}: a bounded list is required")
    return value


def select_branch(src_dir: Any, pins: Any, *, historical: tuple = (), fixtures: Path | None = None) -> Branch:
    """The ONLY branch selector: an actual package whose digest equals a literal HISTORICAL pin named by the caller runs that guard's historical assertions; otherwise the package must be exactly the
    literal-pinned FINAL source (whole proof above). Anything else refuses. A sentinel final pin refuses (hard) unless the package is one of the caller's historical packages."""
    digest, _files, _total = actual_listing(src_dir)
    hist = getattr(pins, "HISTORICAL", {})
    for name in historical:
        if name not in hist:
            raise SourceProfileError(f"unknown historical pin {name!r}")
        if hist[name] == digest:
            return Branch("historical", name)
    final_pins_ready(pins)
    if digest != pins.TREE_DIGEST_SHA256:
        raise UnknownSource(f"the actual package digest {digest} is neither a pinned historical package {list(historical)} nor the pinned final locked source")
    return Branch("final", "final", verify_final(src_dir, pins, fixtures=fixtures))


def require_branch(src_dir: Any, *, historical: tuple = ()) -> Branch:
    """Guard entry point with the shipped pin module."""
    import p25_source_pins as pins
    return select_branch(src_dir, pins, historical=historical)
