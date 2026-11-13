"""S1: CPU physical FILE of an accepted lifecycle head -- a complete `Boundary` or a CLOSED daily state of the head `DailyHold` (NumPy/Numba x Dense / published Windowed).

PROFILE `maple-syrup-floating-lifecycle-checkpoint/1` (an OUTER container; every old profile is unchanged and refuses this file, and this loader refuses theirs):
  floating_lifecycle_checkpoint.json   strict outer manifest, sealed LAST: profile, identity, source pair, backends, bed storage, H0 record meta + array table, file table, child digest, and the closed scalar LIFECYCLE record
  context_arrays.npz                   the H0 record arrays of the CURRENT property root (stored `.npy` members, closed names)
  child/                               the genuine CONTINUED-ORIGIN child (`floating_continued_checkpoint`, `maple-syrup-floating-continued-pond-checkpoint/1` Dense, `/2` published Windowed): the native bed, current W/J/I/D, surface, routing
                                       origin/generation, GLOBAL original datum -- exactly as written by the unchanged codec

WHAT IS SAVED. The head is one of
  * a complete `Boundary` (role boundary, nothing pending): the carrier is the actual `PondEvent.continued_state` of a PRIVATE vertical built from the boundary root (`FloatingPondVertical.from_boundary`) at the boundary's
    current bed/W/J/I/D/surface/routing/clock. The carrier is a COLD physical carrier only (fresh quiet monitor, fresh event-local counters): it is NOT a completion proof and the file never claims one;
  * a CLOSED daily state of the head `DailyHold` (genuine `lineage.head is ctx.hold`, the hold's weak `_HOLD_BINDING` naming this lineage -- the old parent Boundary may be dead and is neither required nor retained). The carrier is built
    from the hold's static pieces and the closed day's CURRENT W/J/I and surface; the daily producer record (`floating_daily.closed_state_record`: previous execution digest, receipt, ledger, cursor, revision, request sequence,
    parent digest) and the hold record carry the daily chain. The lifetime ledger of the hold is NOT summed with the day segment: the hold ledger is stored as is and the segment is added ONCE by a future `hand_back`.
A SAVE never moves the head, never hands back, integrates, reroutes, publishes a journal, forces a quiet or closes anything; an Occurrence (a running storm / a mid-day trial) and a pending storm are refused. A Windowed bed must
already be PUBLISHED (refused before the carrier and before any deep read).

LOAD (`expected_manifest_sha256` is REQUIRED: the caller's own record of the sealed digest `save` returned -- the file is never its own external identity). After the strict manifest/source/layout/seal checks and the composed
admission, the H0 context is rebuilt, the caller's STATIC factory builds the event from `factory(vertical, carried_topology)` (the same recipe as the lifecycle's `event_factory`; the vertical is built by this loader from the verified
context: no original bed, owner, probe or property provider), the child is loaded by the unchanged continued codec (adoption LAST), and a one-shot verified token stages the head on a FRESH lineage: a `Boundary` for a boundary file;
for a closed day a `DailyHold` plus the closed `FloatingDailyState` whose execution/state digests are RECOMPUTED from the producer receipt (revision, request sequence, parent digest, cursor, ledger, `frame_identity` preserved, so the
every-N output writer's due-day identity is unchanged). The head is installed LAST. The expected seal binds the restored role/cursor/lifecycle ledger to the written record; no quiet/producer canonical proof is claimed from the fresh
carrier, and process-local forcing-history receipts of a storm are NOT restored (a pending storm is refused by this file).

Memory (declared host model, admitted BEFORE arrays): the outer manifest/buffers/cold source-pair allowance/reserve (counted once) + the H0 reconstruction OR the retained context plus the continued child's own declared load model (its
real dimensions; the Windowed deep cache bound is the actual native mapping bound, never a compressed-file proxy). The cold carrier at save is admitted before it is built and is a rare checkpoint cost, not a moisture-writer one.
Limits: CPU only (CUDA refused before any IO); pending storm / pending recession are not saved; the restored lineage is fresh (no old Boundary), the original parent Boundary is not reconstructed; a caller-supplied static factory is not trusted
beyond the verified context/identity checks; no scheduler, biology or wind state. N-day every-N output cadence is NOT in this file (the immutable writer record/source/N stay separate and bind through the unchanged frame identity).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_context_checkpoint as fcc
from maple_syrup import floating_continued_checkpoint as cc
from maple_syrup import floating_daily as fd
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_held_context as fh
from maple_syrup import floating_pond as fp
from maple_syrup import floating_resume as fr
from maple_syrup import pond_checkpoint as pc
from maple_syrup.checkpoint import DEFAULT_MAX_CHECKPOINT_BYTES

__all__ = ["MANIFEST_NAME", "PROFILE", "LifecycleCheckpointError", "RestoredLifecycle", "SavedLifecycleCheckpoint", "lifecycle_load_need_bytes", "load_floating_lifecycle_checkpoint", "save_floating_lifecycle_checkpoint"]

PROFILE = "maple-syrup-floating-lifecycle-checkpoint/1"
MANIFEST_NAME = "floating_lifecycle_checkpoint.json"
ARRAYS_NAME = fcc.ARRAYS_NAME
CHILD_DIR = fcc.CHILD_DIR
CHILD_SCHEMAS = {"dense": fc.CONTINUED_SCHEMA, "windowed": fc.CONTINUED_WINDOWED_SCHEMA}
MANIFEST_KEYS = frozenset({"profile", "identity", "source", "array_backend", "hydraulic_backend", "bed_storage", "context_schema", "context_meta", "context_arrays", "files", "child", "lifecycle", "manifest_sha256"})
LIFECYCLE_KEYS = frozenset({"kind", "head", "hold", "daily", "bridge", "routing"})
BRIDGE_KEYS = frozenset({"stable_sha256", "physics_config_sha256", "config_sha256", "root_column_sha256", "source_witness", "property_generation", "bed_generation", "depth_generation", "run_digest", "shape",
                         "frame_source_sha256"})
ROUTING_KEYS = frozenset({"generation", "topology_sha256", "source_sha256"})
KINDS = ("boundary", "closed_day")
CHILD_NAMES = fcc.CHILD_NAMES


class LifecycleCheckpointError(fc.FloatingCheckpointError):
    """Typed narrow refusal of the lifecycle checkpoint."""


def _fail(message: str) -> None:
    raise LifecycleCheckpointError(message)


@dataclass(frozen=True)
class SavedLifecycleCheckpoint:
    """`save`'s result: the published directory and the SEALED outer manifest digest the caller records and presents as `expected_manifest_sha256` at load."""

    path: Path
    manifest_sha256: str
    kind: str


@dataclass(frozen=True, eq=False)
class RestoredLifecycle:
    """`load`'s result: the FRESH lineage (head installed LAST), the restored head, the kind, the sealed digest and for a closed day the genuine closed `FloatingDailyState`. It deliberately holds NO event, state, phase arrays or
    verified token: the transient continued carrier (fresh quiet monitor, unused phase arrays) is released when `load` returns; the physical water/history/bed/routing live in the head and the closed state."""

    lineage: Any
    head: Any
    closed: Any
    kind: str
    manifest_sha256: str


# ---------------------------------------------------------------------------------------------------------------------------------------------- save
def _carrier(head_kind: str, head: Any, closed: Any) -> tuple:
    """`(root, bed, depth, qx, qy)` of the carrier at the head's CURRENT physical state (references; no copy, no integration)."""
    if head_kind == "boundary":
        return head.root, head.bed, head.surface_depth_m, head.qx_m2_s, head.qy_m2_s
    surf = closed.surface
    depth, qx, qy = (head.surface_depth_m, head.qx_m2_s, head.qy_m2_s) if surf is None else surf
    bed = head.bed if surf is None else replace(head.bed, water=replace(head.bed.water, depth_m=depth))
    with lc._minting():
        root = lc.BoundaryRoot(head.column, head.matrix_op, head.pref_op, head.ks_m_s, head.depth_m, head.depth_m, closed.history, closed.water, head.config, head.generation, head.source_witness, bed)
    return root, bed, depth, qx, qy


def _lifecycle_record(kind: str, head: Any, closed: Any, state: Any) -> dict:
    topo = head.routing[0]
    rec = {"kind": kind, "head": lc.head_record(head), "hold": None, "daily": None, "bridge": None,
           "routing": {"generation": int(topo.generation), "topology_sha256": topo.topology_sha256, "source_sha256": topo.source_sha256}}
    if kind == "closed_day":
        c = closed.ctx
        rec["hold"] = {"closed_t_s": float(closed.water.t_s)}
        rec["daily"] = fd.closed_state_record(closed)
        rec["bridge"] = {"stable_sha256": c.stable_sha256, "physics_config_sha256": c.physics_config_sha256, "config_sha256": c.config_sha256, "root_column_sha256": c.root_column_sha256,
                         "source_witness": c.source_witness, "property_generation": int(c.property_generation), "bed_generation": int(c.bed_generation), "depth_generation": int(c.depth_generation),
                         "run_digest": c.run_digest, "shape": [int(n) for n in c.shape], "frame_source_sha256": closed.frame_identity()["source_sha256"]}
    return rec


def save_floating_lifecycle_checkpoint(path: Any, lineage: Any, factory: Callable[..., Any], identity: dict, *, closed: Any = None, original_topology: Any, pref_spec: Any = None,
                                       available_bytes_fn: Callable[[], int], reserve_bytes: int = 0, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, batch_cells: int = fp.nfs.DEFAULT_BATCH_CELLS) -> SavedLifecycleCheckpoint:
    """Write a NEW lifecycle checkpoint directory of the lineage's accepted head (module docstring). `closed` is REQUIRED exactly when the head is a `DailyHold` (the genuine closed state of that hold) and refused otherwise.
    `factory(vertical, carried_topology) -> PondEvent` is the caller's static event recipe (the lifecycle's `event_factory`); `pref_spec` the raw preferential spec when the root carries one (never inferred). Nothing of the
    lineage, the closed state, the bed or any journal is changed."""
    if not callable(factory) or not callable(available_bytes_fn):
        _fail("factory and available_bytes_fn must be callable")
    if original_topology is None:
        _fail("original_topology (the genuine generation-0 PondTopology of the original case) is required")
    reserve = fcc._reserve(reserve_bytes)
    bc = fcc._batch(batch_cells)
    limit = pc._max_bytes(max_bytes)
    ident = pc._canonical_identity(identity)
    target = Path(path)
    if os.path.lexists(target):
        _fail(f"checkpoint path already exists (no overwrite): {target}")
    parent = target.resolve().parent
    if not parent.is_dir():
        _fail(f"the parent directory of the checkpoint does not exist: {parent}")
    target = parent / target.name
    try:
        head_kind, head = lc.saveable_head(lineage)
    except lc.LifecycleError as exc:
        _fail(str(exc))
    kind = "boundary" if head_kind == "boundary" else "closed_day"
    if kind == "boundary":
        if closed is not None:
            _fail("a boundary head is saved without a closed daily state")
    else:
        if closed is None:
            _fail("a daily-hold head is saved with its genuine closed daily state (`closed=`)")
        try:
            lc.check_closed_day(head, closed)
        except lc.LifecycleError as exc:
            _fail(str(exc))
    root, bed, depth, qx, qy = _carrier(head_kind, head, closed)
    if root.column.backend not in fc.CPU_BACKENDS:
        _fail("the lifecycle checkpoint serves the CPU backends; CUDA is refused before any IO")
    windowed = fc.is_windowed_bed(bed)
    if windowed:
        pc._require_published(bed)  # a pending journal is refused BEFORE the carrier or any deep read; a save never publishes
    record = _lifecycle_record(kind, head, closed, None)
    cold = max(int(fr._cold_source_bytes()), fcc.COLD_ALLOWANCE_BYTES)
    rec_bytes = len(json.dumps(record).encode())
    fc._read(available_bytes_fn, 3 * rec_bytes + fcc.BUFFER_BYTES + cold + reserve, "lifecycle checkpoint save (record, hash buffers, cold source pair)")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=parent))
    try:
        vert = fp.FloatingPondVertical.from_boundary(root, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve, batch_cells=bc)  # admits its own composed model
        try:
            rec = fh.export_held_context(vert, pref_spec, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
        except fh.HeldContextError as exc:
            if "reported available" in str(exc) or "declared" in str(exc):
                raise fp.nfs.NativeFloatingMemoryError(str(exc)) from None
            _fail(f"the held context cannot be exported: {exc}")
        meta = rec.meta()
        arrays = {k: np.ascontiguousarray(a) for k, a in rec.arrays().items()}
        table = {k: {"dtype": a.dtype.str, "shape": [int(n) for n in a.shape]} for k, a in arrays.items()}
        if sum(a.nbytes for a in arrays.values()) > limit:
            _fail("the held-context arrays exceed max_bytes; nothing was written")
        np.savez(staging / ARRAYS_NAME, **arrays)
        del rec, arrays
        event = factory(vert, head.routing[0])  # the caller's static recipe on the CARRIED routing topology, outside every private mint window
        lc._admit_native(event, vert)
        if event.control.bed_storage != ("windowed" if windowed else "dense"):
            _fail("the event's configured bed storage differs from the bed's storage")
        _admit_carrier(event, vert, bed, windowed, ident, available_bytes_fn, reserve)  # the COLD carrier is admitted BEFORE it is built
        state = event.continued_state(bed, vert.initial_soil_water(), routing=head.routing, depth_m=depth, qx_m2_s=qx, qy_m2_s=qy, t_s=float(closed.water.t_s) if closed is not None else float(head.t_s))
        lc._admit_state(event, vert, state, bed, float(state.t_s))
        cc.save_floating_continued_checkpoint(staging / CHILD_DIR, state, event, ident, original_topology=original_topology, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve, max_bytes=limit)
        del state
        child = json.loads((staging / CHILD_DIR / fc.MANIFEST_NAME).read_text())
        child_names = tuple(sorted(p.name for p in (staging / CHILD_DIR).iterdir()))
        if set(child_names) != CHILD_NAMES or child.get("schema") != CHILD_SCHEMAS[event.control.bed_storage]:
            _fail("the child checkpoint is not the exact closed continued layout")
        manifest = {"profile": PROFILE, "identity": ident, "source": pc.pond_checkpoint_source_pair(), "array_backend": "numpy", "hydraulic_backend": event.control.backend, "bed_storage": event.control.bed_storage,
                    "context_schema": fh.SCHEMA, "context_meta": meta, "context_arrays": table, "files": fcc._file_table(staging, (ARRAYS_NAME,), limit)}
        cfiles = fcc._file_table(staging / CHILD_DIR, child_names, limit)
        manifest["child"] = {"schema": child["schema"], "manifest_sha256": cfiles[fc.MANIFEST_NAME]["sha256"], "files": cfiles}
        manifest["lifecycle"] = record
        total = manifest["files"][ARRAYS_NAME]["size"] + sum(r["size"] for r in cfiles.values())
        if total > limit:
            _fail(f"the container needs {total} bytes (outer arrays + child files), above max_bytes = {limit}; nothing was published")
        sealed = fcc._seal(manifest)
        text = pc._json_dumps(sealed)
        if len(text) > fcc.MAX_MANIFEST_BYTES:
            _fail("the outer manifest exceeds the bounded size")
        for rel in (ARRAYS_NAME, *(f"{CHILD_DIR}/{n}" for n in child_names)):
            fcc._fsync_file(staging / rel)
        with (staging / MANIFEST_NAME).open("x", encoding="utf-8") as h:
            h.write(text)
            h.flush()
            os.fsync(h.fileno())
        fcc._fsync_dir(staging / CHILD_DIR)
        fcc._fsync_dir(staging)
        fcc._publish_noreplace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        fcc._fsync_dir(parent)
    except OSError as exc:
        raise LifecycleCheckpointError(f"INDETERMINATE durability: the checkpoint was published at {target} but the parent directory fsync failed ({exc}); the published directory is preserved") from None
    return SavedLifecycleCheckpoint(target, sealed["manifest_sha256"], kind)


def _admit_carrier(event: Any, vert: Any, bed: Any, windowed: bool, ident: dict, available_bytes_fn: Callable[[], int], reserve: int) -> None:
    """Composed admission of the COLD carrier BEFORE `continued_state` allocates it: the carrier's own state arrays (the child's analytic table) plus the continued codec's declared save peak for the actual native bed (the
    windowed native bound includes the bed's real mapping bound). A rare checkpoint cost; no extra original bed or deep copy exists."""
    spec = 0
    for shape, dtype, _s, _l in {**fc._specs(event, dry=False), **fc._floating_specs(vert.n_layers, vert.n_cells, vert.held_template().ks_m_s is not None), **fc._origin_specs(event)}.values():
        if None not in shape:
            spec += int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if windowed:
        deep = pc._deep_record(bed.voxel_column, event)
        native = pc._windowed_resident_nbytes(bed, deep, ident)
        n_vox = int(bed.voxel_column.n_voxels)
    else:
        deep = None
        native = pc._tree_nbytes(bed.voxel_column) + pc._tree_nbytes(bed.active_layer) + pc._tree_nbytes(bed.sediment_availability) + pc._tree_nbytes(bed.water)
        n_vox = int(bed.voxel_column.mass_kg.shape[2])
    need = fc._model_bytes(event, vert, table_bytes=spec, native_bytes=native, reserve=reserve, n_voxels=n_vox, deep=deep)
    if windowed:
        need = max(need, fc._model_bytes(event, vert, table_bytes=spec, native_bytes=native, reserve=reserve, n_voxels=n_vox, deep=deep, cache_floor=int(bed.voxel_column.deep_store.mapped_bytes_bound)))
    fc._read(available_bytes_fn, need + spec, "lifecycle checkpoint cold carrier + child save (composed)")


# ---------------------------------------------------------------------------------------------------------------------------------------------- load
def _check_lifecycle(rec: Any, bed_storage: str) -> dict:
    if type(rec) is not dict or set(rec) != LIFECYCLE_KEYS or rec["kind"] not in KINDS:
        _fail("lifecycle record keys/kind mismatch (closed key set)")
    if type(rec["routing"]) is not dict or set(rec["routing"]) != ROUTING_KEYS or type(rec["routing"]["generation"]) is not int or rec["routing"]["generation"] < 0:
        _fail("lifecycle routing record is malformed")
    for key in ("topology_sha256", "source_sha256"):
        pc._hex(rec["routing"][key], f"lifecycle.routing.{key}")
    if rec["kind"] == "boundary":
        if rec["hold"] is not None or rec["daily"] is not None or rec["bridge"] is not None:
            _fail("a boundary record carries no hold/daily/bridge parts")
    else:
        b = rec["bridge"]
        if type(b) is not dict or set(b) != BRIDGE_KEYS or rec["hold"] is None or type(rec["daily"]) is not dict:
            _fail("a closed-day record needs the hold, daily and bridge parts (closed key sets)")
        for key in ("stable_sha256", "physics_config_sha256", "config_sha256", "root_column_sha256", "run_digest", "frame_source_sha256"):
            pc._hex(b[key], f"lifecycle.bridge.{key}")
    return rec


def _preflight(path: Any, expected_identity: dict, expected_seal: Any, limit: int, available_bytes_fn: Callable[[], int], reserve: int, batch_cells: int) -> tuple:
    """Everything verifiable WITHOUT a payload array (mirrors `floating_context_checkpoint._preflight` for the CONTINUED child): stage-0 admission, strict manifest, the caller's EXPECTED seal, bindings, closed layout, child
    cross-binding, the analytic closed H0 table and the composed need (the actual `fcc._compose` / `fcc._child_need` models)."""
    expected = pc._canonical_identity(expected_identity)
    seal = pc._hex(expected_seal, "expected_manifest_sha256") if isinstance(expected_seal, str) else _fail("expected_manifest_sha256 is REQUIRED (the sealed digest the writer returned)")
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _fail(f"checkpoint path is not a plain directory: {root}")
    root = root.resolve()
    if {p.name for p in root.iterdir()} != {MANIFEST_NAME, ARRAYS_NAME, CHILD_DIR}:
        _fail("lifecycle checkpoint directory must contain exactly the manifest, the arrays and the child directory")
    fcc._check_regular(root / MANIFEST_NAME, "outer manifest")
    cold = max(int(fr._cold_source_bytes()), fcc.COLD_ALLOWANCE_BYTES)
    msize = int(os.lstat(root / MANIFEST_NAME).st_size)
    fc._read(available_bytes_fn, 3 * msize + fcc.BUFFER_BYTES + cold + reserve, "lifecycle manifest read")
    data = fcc._read_manifest(root, MANIFEST_NAME)
    if set(data) != MANIFEST_KEYS:
        _fail(f"outer manifest keys mismatch: missing {sorted(MANIFEST_KEYS - set(data))}, unknown {sorted(set(data) - MANIFEST_KEYS)}")
    if pc._hex(data["manifest_sha256"], "manifest_sha256") != fcc._seal(data)["manifest_sha256"]:
        _fail("outer manifest hash mismatch (the manifest was modified)")
    if data["manifest_sha256"] != seal:
        _fail("the sealed manifest digest differs from the caller's EXPECTED digest (a different or re-sealed file)")
    if data["profile"] != PROFILE:
        _fail("unknown lifecycle profile")
    if data["identity"] != expected:
        _fail("lifecycle checkpoint identity differs from the expected identity")
    if data["source"] != pc.pond_checkpoint_source_pair():
        _fail("source pair mismatch: the imported packages differ from the ones that wrote the checkpoint (refused before any payload is read)")
    if data["array_backend"] != "numpy" or data["hydraulic_backend"] not in fc.CPU_BACKENDS or data["context_schema"] != fh.SCHEMA or data["bed_storage"] not in pc.STORAGES:
        _fail("unsupported backend, context schema or bed storage (CPU only; CUDA refused before any IO)")
    life = _check_lifecycle(data["lifecycle"], data["bed_storage"])
    files, child, table = data["files"], data["child"], data["context_arrays"]
    if not isinstance(files, dict) or set(files) != {ARRAYS_NAME} or not isinstance(table, dict):
        _fail("file table is malformed")
    if not isinstance(child, dict) or set(child) != {"schema", "manifest_sha256", "files"} or not isinstance(child["files"], dict) or set(child["files"]) != CHILD_NAMES:
        _fail("child block is malformed (closed native child names required)")
    if child["schema"] != CHILD_SCHEMAS[data["bed_storage"]]:
        _fail("child schema does not match the bed storage (the continued-origin profile is required)")
    if child["manifest_sha256"] != child["files"][fc.MANIFEST_NAME].get("sha256"):
        _fail("child manifest digest differs from its file-table entry")
    child_dir = root / CHILD_DIR
    if child_dir.is_symlink() or not child_dir.is_dir() or {p.name for p in child_dir.iterdir()} != CHILD_NAMES:
        _fail("child directory entries differ from the closed native child layout")
    for n in CHILD_NAMES:
        fcc._check_regular(child_dir / n, n)
    fcc._check_regular(root / ARRAYS_NAME, ARRAYS_NAME)
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
    cman = fcc._read_manifest(child_dir, fc.MANIFEST_NAME)
    try:
        meta = fh._check_meta(data["context_meta"], len(table))
        ids, facts = meta["identities"], meta["facts"]
        expect_arrays = fh._expected_arrays(meta)
    except fh.HeldContextError as exc:
        _fail(f"the held-context metadata is invalid: {exc}")
    fl = cman.get("floating")
    if (cman.get("schema") != child["schema"] or cman.get("identity") != expected or cman.get("source") != data["source"] or cman.get("hydraulic_backend") != data["hydraulic_backend"]
            or cman.get("profile") != fc.CONTINUED_PROFILE or not isinstance(fl, dict) or fl.get("root_column_sha256") != ids["root_column_sha256"] or fl.get("root_matrix_op_sha256") != ids["matrix_op_sha256"]
            or fl.get("root_pref_sha256") != ids["pref_sha256"] or fl.get("owner_config_sha256") != ids["config_sha256"] or fl.get("generation") != facts["generation"]
            or fl.get("source_witness") != facts["source_witness"] or fl.get("n_layers") != meta["n_layers"] or not isinstance(fl.get("shape"), list) or len(fl["shape"]) != 2
            or int(fl["shape"][0]) * int(fl["shape"][1]) != meta["n_cells"]):
        _fail("the child manifest is not bound to this context (schema/profile/identity/source/backend/H0 root ids/generation/witness/shape)")
    init = (cman.get("binding") or {}).get("initial_topology")
    if (not isinstance(init, dict) or init.get("generation") != life["routing"]["generation"] or init.get("topology_sha256") != life["routing"]["topology_sha256"]
            or init.get("source_sha256") != life["routing"]["source_sha256"]):
        _fail("the lifecycle routing record differs from the child's carried routing origin")
    if life["kind"] == "closed_day" and (life["bridge"]["config_sha256"] != ids["config_sha256"] or life["bridge"]["root_column_sha256"] != ids["root_column_sha256"]
                                         or life["bridge"]["source_witness"] != facts["source_witness"] or life["bridge"]["property_generation"] != facts["generation"]):
        _fail("the daily bridge identities differ from the held-context root (config / column / witness / generation)")
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
    child_need = fcc._child_need(cman, meta, child["files"], batch_cells)
    need = fcc._compose(msize, table_bytes, m, child_need, cold, reserve)
    return root, data, life, need


def lifecycle_load_need_bytes(path: Any, expected_identity: dict, expected_manifest_sha256: str, *, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
                              batch_cells: int = fp.nfs.DEFAULT_BATCH_CELLS) -> int:
    """Public verified preflight: the declared composed need of `load_floating_lifecycle_checkpoint` after the strict manifest/seal/binding checks and before any payload hash or array."""
    return _preflight(path, expected_identity, expected_manifest_sha256, pc._max_bytes(max_bytes), available_bytes_fn, fcc._reserve(reserve_bytes), fcc._batch(batch_cells))[3]


def load_floating_lifecycle_checkpoint(path: Any, factory: Callable[..., Any], expected_identity: dict, expected_manifest_sha256: str, *, original_topology: Any, partition: Any = None,
                                       available_bytes_fn: Callable[[], int], reserve_bytes: int = 0, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, batch_cells: int = fp.nfs.DEFAULT_BATCH_CELLS) -> RestoredLifecycle:
    """Restore a lifecycle checkpoint (module docstring). `partition` (the caller's `EvaporationPartition`) is REQUIRED for a closed-day file and refused for a boundary file; its identity is verified against the file."""
    reserve = fcc._reserve(reserve_bytes)
    limit = pc._max_bytes(max_bytes)
    bc = fcc._batch(batch_cells)
    if not callable(factory) or not callable(available_bytes_fn):
        _fail("factory and available_bytes_fn must be callable")
    if original_topology is None:
        _fail("original_topology (the genuine generation-0 PondTopology of the original case) is required")
    expected = pc._canonical_identity(expected_identity)
    root, data, life, need = _preflight(path, expected, expected_manifest_sha256, limit, available_bytes_fn, reserve, bc)
    kind = life["kind"]
    if (partition is None) != (kind == "boundary"):
        _fail("a closed-day file needs the caller's evaporation partition; a boundary file takes none")
    fc._read(available_bytes_fn, need, "lifecycle checkpoint load (composed)")  # BEFORE any payload hash of a large file, np.load or copy
    table, files, child_dir = data["context_arrays"], data["files"], root / CHILD_DIR
    for base, tab in ((root, files), (child_dir, data["child"]["files"])):
        for rel, rec in tab.items():
            h, size = pc._hash_file(base / rel, limit)
            if size != rec["size"] or h != rec["sha256"]:
                _fail(f"file hash/size mismatch: {rel}")
    sizes = pc._zip_audit(root / ARRAYS_NAME, limit)
    pc._npy_preflight(root / ARRAYS_NAME, sizes, table)
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
        ctx = fh.reconstruct_held_context(record, fcc._expected_from(record.meta()), available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
    except fh.HeldContextError as exc:
        if "reported available" in str(exc):
            raise fp.nfs.NativeFloatingMemoryError(str(exc)) from None
        _fail(f"the held context cannot be restored: {exc}")
    del record
    origin = cc.read_continued_origin(child_dir, expected, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
    topology = cc.origin_topology(origin, original_topology)
    vert = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve, batch_cells=bc)
    event = factory(vert, topology)  # the CALLER's static factory (the lifecycle's own event recipe), outside every private mint window
    v = fcc._check_event(event)
    if v is not vert or v._owner is not None or getattr(v, "_held", None) is not ctx:
        _fail("the factory's event must run the vertical this loader built from the verified context")
    if event.control.bed_storage != data["bed_storage"] or event.control.backend != data["hydraulic_backend"]:
        _fail("the factory's event storage/backend differs from the file's")
    state = cc.load_floating_continued_checkpoint(child_dir, event, expected, original_topology=original_topology, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve, max_bytes=limit,
                                                  expected_bed_storage=data["bed_storage"])
    try:
        token = lc.verified_restore(event, state, data["manifest_sha256"])
        lineage, head, broot = lc.stage_restored_head(token, "boundary" if kind == "boundary" else "daily_hold", life["head"], life["hold"])
        closed = None
        if kind == "closed_day":
            closed = br.restore_closed_day(token, head, broot, state, life["daily"], partition=partition, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
            _check_bridge(closed, life["bridge"])
        lc.install_restored_head(token, lineage, head)  # the head is installed LAST
    except (lc.LifecycleError, br.BridgeError) as exc:
        _fail(f"the lifecycle head cannot be restored: {exc}")
    return RestoredLifecycle(lineage, head, closed, kind, data["manifest_sha256"])


def _check_bridge(closed: Any, bridge: dict) -> None:
    c = closed.ctx
    mine = {"stable_sha256": c.stable_sha256, "physics_config_sha256": c.physics_config_sha256, "config_sha256": c.config_sha256, "root_column_sha256": c.root_column_sha256, "source_witness": c.source_witness,
            "property_generation": int(c.property_generation), "bed_generation": int(c.bed_generation), "depth_generation": int(c.depth_generation), "run_digest": c.run_digest, "shape": [int(n) for n in c.shape],
            "frame_source_sha256": closed.frame_identity()["source_sha256"]}
    if mine != bridge:
        bad = sorted(k for k in mine if mine[k] != bridge.get(k))
        _fail(f"the restored daily context differs from the written one ({bad}: partition, case, source or generations)")
