"""H2: versioned OUTER context container around the UNCHANGED genuine floating child checkpoint (CPU: NumPy/Numba; Dense and published Windowed child).

Layout (new profile `maple-syrup-floating-context-checkpoint/1`; the child is the existing `floating_checkpoint` /5 or /6 directory, saved and loaded AS IS -- its schema, provenance, identity and validators are never rewritten or
duplicated):
  floating_context_checkpoint.json   strict outer manifest (sealed LAST): profile, source pair, backend, expected identity, H0 record meta (strict JSON) + array table, file table (size + SHA-256), child manifest digest
  context_arrays.npz                 the H0 record's named arrays (stored `.npy` members, closed names)
  child/                             `floating_checkpoint.save_floating_pond_checkpoint` output (manifest + the two payloads; Windowed: deep-store REFERENCES only)

SAVE validates and stages in a sibling temporary directory (no accepted state changes), exports the H0 context from the event's vertical (a raw `PreferentialSpec` is required when the vertical has a preferential operator: never
inferred), writes the arrays, RELEASES the record, saves the child through `fc.save`, hashes everything, seals the outer manifest last and publishes by an atomic no-overwrite rename; a failure removes only its own temporary
directory. LOAD verifies the bounded outer manifest/source/files/hashes/headers, builds and verifies the H0 context (`record_from_parts` / `reconstruct_held_context`), calls the CALLER'S static factory `factory(held_context) -> PondEvent`
OUTSIDE every private mint window (the event must be a floating event whose vertical was built by `FloatingPondVertical.from_held_context` from exactly that context), then delegates to the unchanged `fc.load_floating_pond_checkpoint`
(its own strict binding refuses a wrong config/source/topology/forcing/Ks/preferential/held context, validates physically, and adopts LAST). The factory is not trusted: it is only the way to obtain the static event; the context is a pure
law context, never a bed proof.
Memory (declared host model): H0 export/record bytes (its own model) is admitted and released before the child; the child admits its own model; the outer adds manifest/hash/header buffers. Held maps O(L N) exist twice only while the
record is alive (H0 record + the prepared column); no bed copy beyond the genuine native checkpoint. CUDA is refused before any IO. Not a daily/wind/scheduler container.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_held_context as fh
from maple_syrup import floating_pond as fp
from maple_syrup import floating_resume as fr
from maple_syrup import pond_checkpoint as pc
from maple_syrup.checkpoint import DEFAULT_MAX_CHECKPOINT_BYTES

__all__ = ["MANIFEST_NAME", "PROFILE", "ContextCheckpointError", "load_floating_context_checkpoint", "save_floating_context_checkpoint"]

PROFILE = "maple-syrup-floating-context-checkpoint/1"
MANIFEST_NAME = "floating_context_checkpoint.json"
ARRAYS_NAME = "context_arrays.npz"
CHILD_DIR = "child"
MANIFEST_KEYS = frozenset({"profile", "identity", "source", "array_backend", "hydraulic_backend", "bed_storage", "context_schema", "context_meta", "context_arrays", "files", "child", "manifest_sha256"})
CHILD_NAMES = frozenset({fc.MANIFEST_NAME, *pc.PAYLOADS})
CHILD_SCHEMAS = {"dense": fc.SCHEMA_V3, "windowed": fc.WINDOWED_SCHEMA_V3}
COLD_ALLOWANCE_BYTES = 3 * 1024**2  # the first-use source-pair hash (R1 measured 3 MiB), never assumed to fit the 1 MiB hash buffer
MAX_MANIFEST_BYTES = 64 * 1024**2
BUFFER_BYTES = (1 << 20) + (1 << 16)

# max_bytes semantics (explicit): (a) every file is read/hashed under it, (b) the SUM of the outer arrays file and all child files must not exceed it, (c) it is also passed to the child as the child's own total cap.


class ContextCheckpointError(fc.FloatingCheckpointError):
    """Typed narrow refusal of the context container."""


def _fail(message: str) -> None:
    raise ContextCheckpointError(message)


def nfs_default_batch() -> int:
    return fp.nfs.DEFAULT_BATCH_CELLS


def _batch(value: Any) -> int:
    """Explicit bounded canonical-reader batch POLICY shared by the published need and the factory's vertical (the model is monotone in it; a larger factory batch is refused after the factory)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= int(value) <= 1 << 20:
        _fail(f"batch_cells must be an integer in [1, {1 << 20}], got {value!r}")
    return int(value)


def _reserve(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        _fail(f"reserve_bytes must be an integer >= 0, got {value!r}")
    return int(value)


def _seal(m: dict) -> dict:
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    return {**body, "manifest_sha256": pc._sha256_text(pc._json_dumps(body))}


def _check_event(event: Any) -> Any:
    try:
        return fc._check_event(event)
    except fc.FloatingCheckpointError as exc:
        raise ContextCheckpointError(str(exc)) from None


def _file_table(root: Path, names: tuple, limit: int) -> dict:
    out = {}
    for name in names:
        h, _ = pc._hash_file(root / name, limit)
        out[name] = {"size": int((root / name).stat().st_size), "sha256": h}
    return out


def save_floating_context_checkpoint(path: Any, state: Any, event: Any, identity: dict, *, pref_spec: Any = None, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0,
                                     max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES) -> Path:
    """Write a NEW context-container directory (module docstring)."""
    v = _check_event(event)
    reserve = _reserve(reserve_bytes)
    limit = pc._max_bytes(max_bytes)
    ident = pc._canonical_identity(identity)
    target = Path(path)
    if os.path.lexists(target):
        _fail(f"checkpoint path already exists (no overwrite): {target}")
    parent = target.resolve().parent
    if not parent.is_dir():
        _fail(f"the parent directory of the checkpoint does not exist: {parent}")
    target = parent / target.name
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=parent))
    try:
        try:
            rec = fh.export_held_context(v, pref_spec, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
        except fh.HeldContextError as exc:
            if "reported available" in str(exc) or "declared" in str(exc):
                raise fp.nfs.NativeFloatingMemoryError(str(exc)) from None  # the memory refusal keeps its typed class
            _fail(f"the held context cannot be exported: {exc}")
        meta = rec.meta()
        arrays = {k: np.ascontiguousarray(a) for k, a in rec.arrays().items()}
        table = {k: {"dtype": a.dtype.str, "shape": [int(n) for n in a.shape]} for k, a in arrays.items()}
        if sum(a.nbytes for a in arrays.values()) > limit:
            _fail("the held-context arrays exceed max_bytes; nothing was written")
        np.savez(staging / ARRAYS_NAME, **arrays)
        del rec, arrays  # the record and its sealed copies are released BEFORE the child operation
        fc.save_floating_pond_checkpoint(staging / CHILD_DIR, state, event, ident, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve, max_bytes=limit)
        child = json.loads((staging / CHILD_DIR / fc.MANIFEST_NAME).read_text())
        child_names = tuple(sorted(p.name for p in (staging / CHILD_DIR).iterdir()))
        if set(child_names) != CHILD_NAMES or child.get("schema") != CHILD_SCHEMAS[event.control.bed_storage]:
            _fail("the child checkpoint is not the exact closed /5-/6 layout")
        manifest = {"profile": PROFILE, "identity": ident, "source": pc.pond_checkpoint_source_pair(), "array_backend": "numpy", "hydraulic_backend": event.control.backend,
                    "bed_storage": event.control.bed_storage, "context_schema": fh.SCHEMA, "context_meta": meta, "context_arrays": table,
                    "files": _file_table(staging, (ARRAYS_NAME,), limit)}
        cfiles = _file_table(staging / CHILD_DIR, child_names, limit)
        manifest["child"] = {"schema": child["schema"], "manifest_sha256": cfiles[fc.MANIFEST_NAME]["sha256"], "files": cfiles}
        total = manifest["files"][ARRAYS_NAME]["size"] + sum(r["size"] for r in cfiles.values())
        if total > limit:
            _fail(f"the container needs {total} bytes (outer arrays + child files), above max_bytes = {limit}; nothing was published")
        text = pc._json_dumps(_seal(manifest))
        if len(text) > MAX_MANIFEST_BYTES:
            _fail("the outer manifest exceeds the bounded size")
        for rel in (ARRAYS_NAME, *(f"{CHILD_DIR}/{n}" for n in child_names)):  # durability: payload files first, then the directories, the manifest sealed LAST
            _fsync_file(staging / rel)
        with (staging / MANIFEST_NAME).open("x", encoding="utf-8") as h:
            h.write(text)
            h.flush()
            os.fsync(h.fileno())
        _fsync_dir(staging / CHILD_DIR)
        _fsync_dir(staging)
        _publish_noreplace(staging, target)  # atomic renameat2(RENAME_NOREPLACE): never replaces anything (an empty directory included)
    except BaseException:
        if staging.exists():  # only OUR temporary directory; a published target never exists under this name
            shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        _fsync_dir(parent)
    except OSError as exc:
        raise ContextCheckpointError(f"INDETERMINATE durability: the checkpoint was published at {target} but the parent directory fsync failed ({exc}); the published directory is preserved") from None
    return target


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_noreplace(staging: Path, target: Path) -> None:
    from maple_syrup.dependency_bundle import BundleError, _rename_noreplace

    try:
        _rename_noreplace(staging, target)
    except BundleError as exc:
        raise ContextCheckpointError(f"cannot publish without replacing anything (nothing was overwritten): {exc}") from None


def _read_manifest(root: Path, name: str = MANIFEST_NAME) -> dict:
    info = os.lstat(root / name)
    if not stat.S_ISREG(info.st_mode):
        _fail("outer manifest is not a regular file")
    if info.st_size > MAX_MANIFEST_BYTES:
        _fail("outer manifest exceeds the bounded size")

    def hook(pairs: list) -> dict:
        out: dict = {}
        for k, val in pairs:
            if k in out:
                raise ValueError(f"duplicate key {k!r}")
            out[k] = val
        return out

    def const(token: str) -> Any:
        raise ValueError(f"non-finite constant {token}")

    try:
        data = json.loads((root / name).read_bytes().decode("utf-8"), parse_constant=const, object_pairs_hook=hook)
    except (ValueError, UnicodeDecodeError, RecursionError, OSError) as exc:
        _fail(f"outer manifest is not strict JSON: {exc}")
    if not isinstance(data, dict):
        _fail("outer manifest must be a JSON object")
    return data


def _check_regular(path: Path, what: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        _fail(f"{what} is missing: {exc}")
    if not stat.S_ISREG(info.st_mode):
        _fail(f"{what} is not a regular file (symlinks and special files are refused)")


def _preflight(path: Any, expected_identity: dict, limit: int, available_bytes_fn: Callable[[], int], reserve: int, batch_cells: int) -> tuple[Path, dict, dict, int]:
    """Everything that can be verified WITHOUT reading a payload array: stage-0 admission (manifest x3 + buffers + the cold source-pair allowance + the single reserve), strict manifest, bindings, the exact closed file
    layout, the child manifest cross-binding (schema/identity/source/backend/H0 root ids/generation/witness/shape) and the analytic closed H0 array table. Returns `(root, manifest, child_manifest, composed_need)` where the
    need is the declared COMPOSED peak (see `_compose`) used to admit stage 1 BEFORE any hash of a large file or any array."""
    expected = pc._canonical_identity(expected_identity)
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _fail(f"checkpoint path is not a plain directory: {root}")
    root = root.resolve()
    if {p.name for p in root.iterdir()} != {MANIFEST_NAME, ARRAYS_NAME, CHILD_DIR}:
        _fail("context checkpoint directory must contain exactly the manifest, the arrays and the child directory")
    _check_regular(root / MANIFEST_NAME, "outer manifest")
    cold = max(int(fr._cold_source_bytes()), COLD_ALLOWANCE_BYTES)
    fc._read(available_bytes_fn, 3 * int(os.lstat(root / MANIFEST_NAME).st_size) + BUFFER_BYTES + cold + reserve, "context manifest read")
    data = _read_manifest(root)
    if set(data) != MANIFEST_KEYS:
        _fail(f"outer manifest keys mismatch: missing {sorted(MANIFEST_KEYS - set(data))}, unknown {sorted(set(data) - MANIFEST_KEYS)}")
    if pc._hex(data["manifest_sha256"], "manifest_sha256") != _seal(data)["manifest_sha256"]:
        _fail("outer manifest hash mismatch (the manifest was modified)")
    if data["profile"] != PROFILE:
        _fail("unknown context profile")
    if data["identity"] != expected:
        _fail("context checkpoint identity differs from the expected identity")
    if data["source"] != pc.pond_checkpoint_source_pair():
        _fail("source pair mismatch: the imported packages differ from the ones that wrote the checkpoint (refused before any payload is read)")
    if data["array_backend"] != "numpy" or data["hydraulic_backend"] not in fc.CPU_BACKENDS or data["context_schema"] != fh.SCHEMA or data["bed_storage"] not in pc.STORAGES:
        _fail("unsupported backend, context schema or bed storage (CPU only; CUDA refused before any IO)")
    files, child, table = data["files"], data["child"], data["context_arrays"]
    if not isinstance(files, dict) or set(files) != {ARRAYS_NAME} or not isinstance(table, dict):
        _fail("file table is malformed")
    if not isinstance(child, dict) or set(child) != {"schema", "manifest_sha256", "files"} or not isinstance(child["files"], dict) or set(child["files"]) != CHILD_NAMES:
        _fail("child block is malformed (closed native child names required)")
    if child["schema"] != CHILD_SCHEMAS[data["bed_storage"]]:
        _fail("child schema does not match the bed storage")
    if child["manifest_sha256"] != child["files"][fc.MANIFEST_NAME].get("sha256"):
        _fail("child manifest digest differs from its file-table entry")
    child_dir = root / CHILD_DIR
    if child_dir.is_symlink() or not child_dir.is_dir() or {p.name for p in child_dir.iterdir()} != CHILD_NAMES:
        _fail("child directory entries differ from the closed native child layout")
    for n in CHILD_NAMES:
        _check_regular(child_dir / n, n)
    _check_regular(root / ARRAYS_NAME, ARRAYS_NAME)
    for tab, names in ((files, (ARRAYS_NAME,)), (child["files"], tuple(sorted(CHILD_NAMES)))):
        for n in names:
            rec = tab[n]
            if not isinstance(rec, dict) or set(rec) != {"size", "sha256"} or type(rec["size"]) is not int or rec["size"] < 0:
                _fail(f"file record for {n} is malformed")
            pc._hex(rec["sha256"], n)
            if int((root / ARRAYS_NAME if n == ARRAYS_NAME else child_dir / n).stat().st_size) != rec["size"]:
                _fail(f"file size mismatch: {n}")
    total = files[ARRAYS_NAME]["size"] + sum(r["size"] for r in child["files"].values())
    if total > limit:
        _fail(f"the container holds {total} bytes (outer arrays + child files), above max_bytes = {limit}")
    cman = _read_manifest(child_dir, fc.MANIFEST_NAME)  # bounded strict read of the CHILD manifest only (no payload)
    ids = None
    try:
        meta = fh._check_meta(data["context_meta"], len(table))  # closed analytic metadata check BEFORE any array
        ids, facts = meta["identities"], meta["facts"]
        expect_arrays = fh._expected_arrays(meta)
    except fh.HeldContextError as exc:
        _fail(f"the held-context metadata is invalid: {exc}")
    fl = cman.get("floating")
    if (cman.get("schema") != child["schema"] or cman.get("identity") != expected or cman.get("source") != data["source"] or cman.get("hydraulic_backend") != data["hydraulic_backend"]
            or not isinstance(fl, dict) or fl.get("root_column_sha256") != ids["root_column_sha256"] or fl.get("root_matrix_op_sha256") != ids["matrix_op_sha256"]
            or fl.get("root_pref_sha256") != ids["pref_sha256"] or fl.get("owner_config_sha256") != ids["config_sha256"] or fl.get("generation") != facts["generation"]
            or fl.get("source_witness") != facts["source_witness"] or fl.get("n_layers") != meta["n_layers"] or not isinstance(fl.get("shape"), list) or len(fl["shape"]) != 2
            or int(fl["shape"][0]) * int(fl["shape"][1]) != meta["n_cells"]):
        _fail("the child manifest is not bound to this context (schema/identity/source/backend/H0 root ids/generation/witness/shape)")
    if set(table) != set(expect_arrays):
        _fail("the array table names differ from the analytic closed H0 layout")
    table_bytes = 0
    for name, rec in table.items():
        if not isinstance(rec, dict) or set(rec) != {"dtype", "shape"} or not isinstance(rec["shape"], list) or any(type(n) is not int or n < 0 for n in rec["shape"]):
            _fail("array table entry is malformed")
        shape, dtype = expect_arrays[name]
        if tuple(rec["shape"]) != tuple(shape) or rec["dtype"] != np.dtype(dtype).str:
            _fail(f"array {name}: declared dtype/shape differ from the analytic H0 layout")
        table_bytes += int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    m = fh._model(meta)
    child_need = _child_need(cman, meta, child["files"], batch_cells)
    need = _compose(int(os.lstat(root / MANIFEST_NAME).st_size), table_bytes, m, child_need, cold, reserve)
    return root, data, cman, need


def _child_need(cman: dict, meta: dict, child_files: dict, batch_cells: int) -> int:
    """The child's OWN declared load model (`fc.load_model_bytes`) from its manifest's actual dimensions: shape, class count, phase bins, the declared companion table bytes and the native snapshot's recorded uncompressed
    member bytes (dense) / stored file bytes (windowed; the native archive is written with stored members). The manifest values are unverified here (the child re-verifies everything and refuses a lie); a malformed value is a typed refusal."""
    try:
        b, pl = cman["binding"], cman["payload"]["arrays"]
        nc, nb = b["n_classes"], b["phase_bins"]
        nc_classes = meta["n_classes"]
        shape = cman["floating"]["shape"]
        tb = 0
        for rec in pl.values():
            if set(rec) != {"dtype", "shape"} or any(type(n) is not int or n < 0 for n in rec["shape"]):
                raise ValueError("array table entry")
            tb += int(np.prod(rec["shape"], dtype=np.int64)) * np.dtype(rec["dtype"]).itemsize
        nat = cman.get("native_dense")
        nbytes = int(nat["native_member_bytes"]) if isinstance(nat, dict) else int(child_files[pc.PAYLOADS[0]]["size"])
        if not all(type(x) is int and x >= 0 for x in (nc, nb, nbytes)):
            raise ValueError("dimensions")
        N = int(shape[0]) * int(shape[1])
        windowed = nat is None
        if windowed:
            dp = cman["deep_storage"]
            nv, win, chunk = dp["n_voxels"], dp["window_voxels"], dp["chunk_columns"]
        else:
            nv, win, chunk = int(nat["voxel_shape"][2]), 0, 0
        if not all(type(x) is int and x >= 0 for x in (nv, win, chunk)):
            raise ValueError("native dimensions")
        # C = N (every cell active) is the monotone upper bound: the active count is only inside the sealed H0 arrays
        canon = fc.canonical_depth_bytes(meta["n_layers"], N, N, nc_classes, nv, windowed=windowed, batch_cells=batch_cells, window_voxels=win, chunk_cells=chunk, cache_bytes=fc.deep_cache_bound_bytes(dp, nc_classes) if windowed else 0)
        return fc.load_model_bytes(meta["n_layers"], N, nc, nb, table_bytes=tb, native_bytes=nbytes, reserve=0, canonical_bytes=canon)
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"the child manifest dimensions are malformed: {exc}")


def _compose(manifest_bytes: int, table_bytes: int, m: Any, child_need: int, cold: int, reserve: int) -> int:
    """Declared COMPOSED host peak of one load (modelled bytes; not a whole-process bound). Phases (released between phases): A = raw npz read + its copy (2 x arrays) + the H0 record copies and preparation (the H0
    reconstruct model); B = the retained HeldContext + the child's OWN declared model (`fc.load_model_bytes` from its manifest's real shape, classes, phase bins, companion table and native snapshot bytes; reserve excluded, counted once outside); the outer buffers, the cold source-pair allowance, the manifest parse (3 x) and the single
    reserve are always live. The caller's static factory and the child's own composed model are admitted separately by their own readers (the child re-admits its full model)."""
    return 3 * manifest_bytes + BUFFER_BYTES + cold + reserve + max(2 * table_bytes + int(m.reconstruct_bytes), int(m.retained_bytes) + child_need)


def context_load_need_bytes(path: Any, expected_identity: dict, *, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
                            batch_cells: int = nfs_default_batch()) -> int:
    """Public verified preflight: the declared composed need of `load_floating_context_checkpoint` for this container (after the strict manifest/binding checks, before any hash of a payload or any array)."""
    return _preflight(path, expected_identity, pc._max_bytes(max_bytes), available_bytes_fn, _reserve(reserve_bytes), _batch(batch_cells))[3]


def load_floating_context_checkpoint(path: Any, factory: Callable[[Any], Any], expected_identity: dict, *, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0,
                                     max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, batch_cells: int = nfs_default_batch()) -> tuple[Any, Any]:
    """Restore a context container: returns `(event, state)` where `event` is the factory's event (built from the verified held context) and `state` the registered wet state (module docstring). The factory must admit its
    own static setup; the container-owned bound is `context_load_need_bytes`."""
    reserve = _reserve(reserve_bytes)
    limit = pc._max_bytes(max_bytes)
    if not callable(factory):
        _fail("factory must be callable: factory(held_context) -> PondEvent")
    expected = pc._canonical_identity(expected_identity)
    bc = _batch(batch_cells)
    root, data, _cman, need = _preflight(path, expected, limit, available_bytes_fn, reserve, bc)
    fc._read(available_bytes_fn, need, "context checkpoint load (composed)")  # BEFORE any payload hash of a large file, np.load or copy
    table, files, child_dir = data["context_arrays"], data["files"], root / CHILD_DIR
    for base, tab in ((root, files), (child_dir, data["child"]["files"])):
        for rel, rec in tab.items():
            h, size = pc._hash_file(base / rel, limit)
            if size != rec["size"] or h != rec["sha256"]:
                _fail(f"file hash/size mismatch: {rel}")
    sizes = pc._zip_audit(root / ARRAYS_NAME, limit)
    pc._npy_preflight(root / ARRAYS_NAME, sizes, table)  # header/dtype/shape cross-check BEFORE any array is loaded
    if set(sizes) != {n + ".npy" for n in table}:
        _fail("context archive members differ from the array table")
    try:
        with np.load(root / ARRAYS_NAME, allow_pickle=False) as z:
            arrays = {k: np.array(z[k], copy=True) for k in table}
    except (ValueError, OSError) as exc:
        _fail(f"context arrays cannot be read: {exc}")
    for k, a in arrays.items():
        if a.dtype.str != table[k]["dtype"] or list(a.shape) != table[k]["shape"]:
            _fail(f"array {k}: dtype/shape differ from the table")
    try:
        record = fh.record_from_parts(data["context_meta"], arrays, max_bytes=limit, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
        del arrays
        ctx = fh.reconstruct_held_context(record, _expected_from(record.meta()), available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
    except fh.HeldContextError as exc:
        if "reported available" in str(exc):
            raise fp.nfs.NativeFloatingMemoryError(str(exc)) from None
        _fail(f"the held context cannot be restored: {exc}")
    del record
    event = factory(ctx)  # the CALLER's static factory, outside every private mint window
    v = _check_event(event)
    if v._owner is not None or getattr(v, "_held", None) is not ctx:
        _fail("the factory's event must use FloatingPondVertical.from_held_context(<this context>) and no native owner")
    if v.batch_cells > bc:
        _fail(f"the factory's vertical uses compact batch_cells={v.batch_cells} above the container's declared canonical-reader policy batch_cells={bc} (raise the declared policy so the published need covers it)")
    state = fc.load_floating_pond_checkpoint(child_dir, event, expected, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve, max_bytes=limit)
    return event, state


def _expected_from(meta: dict) -> Any:
    ids = meta["identities"]  # validated by record_from_parts (closed key set)
    return fh.ExpectedHeldIdentities(ids["config_sha256"], ids["root_column_sha256"], ids["matrix_op_sha256"], ids["pref_sha256"])
