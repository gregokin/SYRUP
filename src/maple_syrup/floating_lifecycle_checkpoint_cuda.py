"""S2-CUDA: GPU physical FILE of an accepted lifecycle head -- a complete `Boundary` or a CLOSED daily state of the head `DailyHold` (Dense and published native Windowed device beds). AUTHOR-WRITTEN, NOT RUN ON A DEVICE (Root
qualifies it on hardware). Importing this module never imports CuPy.

PROFILE `maple-syrup-floating-lifecycle-checkpoint/2-cuda`: the CUDA twin of the CPU outer `floating_lifecycle_checkpoint` (`/1`, byte-unchanged and kept strict: it refuses this profile and this loader refuses `/1`; nothing is
migrated). Same closed layout and the SAME scalar lifecycle record (`floating_lifecycle_checkpoint.LIFECYCLE_KEYS` / `_lifecycle_record` / `_check_lifecycle` / `_check_bridge` are REUSED, not copied):
  floating_lifecycle_checkpoint.json   strict outer manifest sealed LAST (profile, identity, source pair, cupy/cuda backends, bed storage, H0 `/2` record meta + array table, file table, child digest, lifecycle record)
  context_arrays.npz                   the H0 `/2` CUDA record's closed HOST arrays of the CURRENT property root
  child/                               the genuine CONTINUED-ORIGIN CUDA child (`floating_continued_checkpoint_cuda`: `maple-syrup-floating-continued-cuda-pond-checkpoint/1` Dense, `/2` Windowed), written and read by the
                                       UNCHANGED codec: the historical `/7 /8` children, the CPU children and the context profiles are never read or rewritten by this file.

WHAT IS SAVED / what is not: exactly the CPU contract. Only a genuinely accepted `Boundary` (nothing pending) or the CLOSED day of the head `DailyHold`; an Occurrence, a pending storm, a pending recession or a mid-trial day is
refused; a save never moves the head, hands back, integrates, reroutes, publishes a journal, advances a clock or forces a quiet monitor. The carrier (`PondEvent.continued_state` of a PRIVATE vertical built from the root) is a COLD
physical carrier with a fresh quiet monitor and fresh event-local counters: no completion proof is claimed, forcing-history receipts of a storm are not restored. The hold ledger is stored as is; the day segment is counted ONCE by a
future `hand_back`. The every-N output writer's record/source/N are separate and bind through the unchanged frame identity.

LOAD: `expected_manifest_sha256` is REQUIRED (the digest `save` returned), the caller supplies the static `factory(vertical, carried_topology) -> PondEvent`, the generation-0 `original_topology`, `expected_device_id` and, for a
closed-day file, the `EvaporationPartition`. Order: strict metadata-only arguments -> the CURRENT-device gate (before any reader, path check, hash, array or callback) -> stage-0 two-pool admission of the manifest read -> strict manifest,
seal, source pair, profile, backends, H0 `/2` analytic layout, child cross-binding incl. the kernel/ABI descriptor, lifecycle record, closed child layout -> the composed two-pool need (`fcxc.compose_need` for the H0 context and the
continued child's `fcc.load_models`, plus `open_plan_cuda` for a closed day) -> hashes and archive/header audits -> bounded `np.load` -> H0 record (host) -> `reconstruct_held_context` (device) -> the host origin reader
(`read_continued_origin_cuda`, header/table audit before decode) -> `origin_topology` -> the vertical from the verified H0 context -> the caller's factory -> the continued CUDA child load (provisional scope, adoption LAST) -> the one-shot
`VerifiedRestore` token -> `stage_restored_head(..., cuda_quota=q)` (resident comparisons/storage, counted scalar reads, sealed host mask/edges/D0) -> closed day: `restore_closed_day_cuda` (producer-receipt restore, digests
RECOMPUTED) -> the head is installed on a FRESH lineage LAST. No original bed, owner, probe or property provider exists on this path.

RESULT: the CPU `RestoredLifecycle` (lineage, head, closed, kind, seal): it retains NO event, state, phase arrays, ledger copy or token; the cold carrier is released when `load` returns (the physical water/history/bed/routing live in
the head and the closed state). MEMORY: declared payload arithmetic from verified headers only (both pools, reserves and the 64 KiB allowance counted once per admission by `fcu.admit`; the Windowed deep-store mapping-cache bound,
never an archive proxy; the cold carrier at save is a rare checkpoint cost). Not claimed: any device run, a pending storm/recession file, scheduler/biology/wind state, large-catchment fit.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_checkpoint_cuda as fcc
from maple_syrup import floating_context_checkpoint as cc
from maple_syrup import floating_context_checkpoint_cuda as fcxc
from maple_syrup import floating_continued_checkpoint_cuda as ccc
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_held_context as fh
from maple_syrup import floating_held_context_cuda as fhc
from maple_syrup import floating_lifecycle_checkpoint as cpul
from maple_syrup import floating_pond as fp
from maple_syrup import floating_resume as fr
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import pond_checkpoint as pc
from maple_syrup import spatial_soil_layers as sp
from maple_syrup.checkpoint import DEFAULT_MAX_CHECKPOINT_BYTES

__all__ = ["MANIFEST_NAME", "PROFILE", "LifecycleCheckpointCudaError", "lifecycle_load_need_bytes_cuda", "load_floating_lifecycle_checkpoint_cuda", "save_floating_lifecycle_checkpoint_cuda"]

PROFILE = "maple-syrup-floating-lifecycle-checkpoint/2-cuda"
MANIFEST_NAME = cpul.MANIFEST_NAME
ARRAYS_NAME = cpul.ARRAYS_NAME
CHILD_DIR = cpul.CHILD_DIR
MANIFEST_KEYS = cpul.MANIFEST_KEYS  # the CPU closed key set: the PROFILE and the backends distinguish the two containers (no placeholder field)
CHILD_NAMES = cpul.CHILD_NAMES
CHILD_SCHEMAS = {"dense": fcc.CONT_GPU_SCHEMA, "windowed": fcc.CONT_GPU_WINDOWED_SCHEMA}
STORAGES = ("dense", "windowed")
RestoredLifecycle = cpul.RestoredLifecycle
SavedLifecycleCheckpoint = cpul.SavedLifecycleCheckpoint


class LifecycleCheckpointCudaError(cpul.LifecycleCheckpointError):
    """Typed narrow refusal of the CUDA lifecycle checkpoint."""


def _fail(message: str) -> None:
    raise LifecycleCheckpointCudaError(message)


def _typed(fn: Callable[..., Any], *a: Any, **k: Any) -> Any:
    """Run an accepted context-container helper and re-type ITS narrow refusals as this container's (memory errors keep their own class)."""
    try:
        return fn(*a, **k)
    except LifecycleCheckpointCudaError:
        raise
    except fc.FloatingCheckpointError as exc:
        raise LifecycleCheckpointCudaError(str(exc)) from None


def _quota(*a: Any) -> dict:
    return _typed(fcxc._quota, *a)


def _resolve_readers(q: dict, cp: Any) -> dict:
    """AFTER the current-device gate: resolve each OMITTED real default reader ONCE from its owning accepted helper (`fcu.device_reader(cp)` for the device pool, `sp.default_host_available_bytes` for the host pool); a caller's
    reader is retained unchanged. Budgets and reserves are untouched (each is applied once by the nested admission/constructor); no budget is invented. The nested constructors (`from_boundary`, `from_held_context`,
    `reconstruct_held_context`, the closed-day restore) require CALLABLES, so no `None` reaches them."""
    out = dict(q)
    if out["available_bytes_fn"] is None:
        out["available_bytes_fn"] = fcu.device_reader(cp)
    if out["host_available_bytes_fn"] is None:
        out["host_available_bytes_fn"] = sp.default_host_available_bytes
    return out


def _capped(reader: Any, budget: Any, default: Callable[[], int]) -> Callable[[], int]:
    return fcxc._capped(reader, budget, default)


# ---------------------------------------------------------------------------------------------------------------------------------------------- models
def _child_models(cman: dict, meta: dict, child_files: dict, batch_cells: int) -> tuple[int, int, dict]:
    """`(device, host, detail)` of the CONTINUED child's load (`fcc.load_models`, the accepted model) from the child manifest's own dimensions. The accepted `fcxc._child_models` keys the Windowed branch on the historical
    `/8` schema, so the continued family needs its own header reading; the arithmetic is the SAME `fcc.load_models` and the host table includes the `co_routing_source` origin array (it is in the declared table)."""
    try:
        b, arrays = cman["binding"], cman["payload"]["arrays"]
        nc_phase, nb, nc, L = b["n_classes"], b["phase_bins"], meta["n_classes"], meta["n_layers"]
        shape = cman["floating"]["shape"]
        table_bytes = 0
        for rec in arrays.values():
            if set(rec) != {"dtype", "shape"} or any(type(n) is not int or n < 0 for n in rec["shape"]):
                raise ValueError("array table entry")
            table_bytes += int(np.prod(rec["shape"], dtype=np.int64)) * np.dtype(rec["dtype"]).itemsize
        N = int(shape[0]) * int(shape[1])
        windowed = cman["schema"] == fcc.CONT_GPU_WINDOWED_SCHEMA
        if windowed:
            native_bytes = int(child_files[pc.PAYLOADS[0]]["size"])
            V, K, cache = 0, int(nc), fc.deep_cache_bound_bytes(cman["deep_storage"], nc)
        else:
            nat = cman["native_dense"]
            native_bytes = int(nat["native_member_bytes"])
            V, K, cache = int(nat["voxel_shape"][2]), int(nat["voxel_shape"][3]), 0
        if not all(type(x) is int and x >= 0 for x in (nc_phase, nb, native_bytes, V, K, cache)):
            raise ValueError("dimensions")
        dev, host = fcc.load_models(L, N, nc_phase, nb, table_bytes=table_bytes, dev_table_bytes=table_bytes, native_bytes=native_bytes, V=V, K=K, n_active=N, batch_cells=batch_cells, windowed=windowed, cache_bytes=cache)
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"the child manifest dimensions are malformed: {exc}")
    return int(dev), int(host), {"table_bytes": table_bytes, "native_bytes": native_bytes, "V": V, "K": K, "cache_bytes": cache, "windowed": windowed}


def _daily_extra(kind: str, meta: dict) -> tuple[int, int]:
    """RAW `(device, host)` of the closed-day context mint (`open_plan_cuda`: zero blocks, storage query, surface copies, evaporation preparation, first-use source hash); 0 for a boundary file."""
    if kind != "closed_day":
        return 0, 0
    from maple_syrup import floating_event_daily_bridge_cuda as bc

    return bc.open_plan_cuda(int(meta["n_layers"]), int(meta["n_cells"]))


def _carrier_need(event: Any, vert: Any, bed: Any, windowed: bool, ident: dict) -> tuple[int, int]:
    """RAW `(device, host)` of the COLD carrier + the continued child's own SAVE (`fcc.save_models` with the table incl. the origin array and, Windowed, the actual deep-store mapping bound), admitted BEFORE the carrier is built."""
    t = vert.held_template()
    L, N = vert.n_layers, vert.n_cells
    layouts = fcc._layouts_of(t.column)
    specs = {**fcc._specs(event, dry=False), **fcc._gpu_floating_specs(L, N, layouts, t.ks_m_s is not None), **fc._origin_specs(event)}
    if windowed:
        deep = pc._deep_record(bed.voxel_column, event)
        native = pc._windowed_resident_nbytes(bed, deep, ident)
        V, K = 0, int(bed.active_layer.mass_kg.shape[-1])
        cache = max(fc.deep_cache_bound_bytes(deep, event.nc), int(bed.voxel_column.deep_store.mapped_bytes_bound))
    else:
        vox = bed.voxel_column.mass_kg
        V, K = int(vox.shape[2]), int(vox.shape[3])
        native = pc._tree_nbytes(bed.voxel_column) + pc._tree_nbytes(bed.active_layer) + pc._tree_nbytes(bed.sediment_availability) + pc._tree_nbytes(bed.water)
        cache = 0
    n_active = int(np.count_nonzero(np.asarray(vert._active_host())))
    return fcc.save_models(L, N, event.nc, event.control.phase_bins, table_bytes=fcc._spec_bytes(specs), native_bytes=native, V=V, K=K, n_active=n_active, batch_cells=vert.batch_cells, windowed=windowed, cache_bytes=cache)


# ---------------------------------------------------------------------------------------------------------------------------------------------- save
def _carrier_cuda(head_kind: str, head: Any, closed: Any) -> tuple:
    """`(root, bed, depth, qx, qy)` of the carrier at the head's CURRENT physical state (references; no copy, no integration). A closed day's root carries the hold's SEALED host mask/edges/D0."""
    if head_kind == "boundary":
        return head.root, head.bed, head.surface_depth_m, head.qx_m2_s, head.qy_m2_s
    surf = closed.surface
    depth, qx, qy = (head.surface_depth_m, head.qx_m2_s, head.qy_m2_s) if surf is None else surf
    bed = head.bed if surf is None else replace(head.bed, water=replace(head.bed.water, depth_m=depth))
    with lc._minting():
        root = lc.BoundaryRoot(head.column, head.matrix_op, head.pref_op, head.ks_m_s, head.depth_m, head.depth_m, closed.history, closed.water, head.config, head.generation, head.source_witness, bed, head.host_active,
                               head.edges_host, head.host_depth)
    return root, bed, depth, qx, qy


def save_floating_lifecycle_checkpoint_cuda(path: Any, lineage: Any, factory: Callable[..., Any], identity: dict, *, closed: Any = None, original_topology: Any, pref_spec: Any = None,
                                            available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0,
                                            memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
                                            batch_cells: int = fp.nfs.DEFAULT_BATCH_CELLS) -> SavedLifecycleCheckpoint:
    """Write a NEW CUDA lifecycle checkpoint directory of the lineage's accepted head (module docstring); same contract as the CPU `save_floating_lifecycle_checkpoint` on a device bed."""
    if not callable(factory):
        _fail("factory must be callable")
    if original_topology is None:
        _fail("original_topology (the genuine generation-0 PondTopology of the original case) is required")
    q = _quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    limit = pc._max_bytes(max_bytes)
    bc = _typed(cc._batch, batch_cells)
    ident = pc._canonical_identity(identity)
    try:
        head_kind, head = lc.saveable_head(lineage)
    except lc.LifecycleError as exc:
        _fail(str(exc))
    kind = "boundary" if head_kind == "boundary" else "closed_day"
    column = head.root.column if kind == "boundary" else head.column
    try:
        cp, dev = fcc.gate(column)  # FIRST: the current device before any reader, path check, IO, export or array
    except fc.FloatingCheckpointError as exc:
        _fail(str(exc))
    q = _resolve_readers(q, cp)
    target = Path(path)
    if os.path.lexists(target):
        _fail(f"checkpoint path already exists (no overwrite): {target}")
    parent = target.resolve().parent
    if not parent.is_dir():
        _fail(f"the parent directory of the checkpoint does not exist: {parent}")
    target = parent / target.name
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
    root, bed, depth, qx, qy = _carrier_cuda(head_kind, head, closed)
    if root.column.backend != "cuda" or root.host_active is None or root.edges_host is None:
        _fail("the CUDA lifecycle checkpoint needs a CUDA root carrying its sealed host mask and edges (refused before any IO)")
    windowed = fc.is_windowed_bed(bed)
    if windowed:
        pc._require_published(bed)  # a pending journal is refused BEFORE the carrier or any deep read; a save never publishes
    record = cpul._lifecycle_record(kind, head, closed, None)
    cold = max(int(fr._cold_source_bytes()), cc.COLD_ALLOWANCE_BYTES)
    rec_bytes = len(json.dumps(record).encode())
    fcxc._admit(cp, dev, 0, 3 * rec_bytes + cc.BUFFER_BYTES + cold, q, "the CUDA lifecycle checkpoint save (record, hash buffers, cold source pair)")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=parent))
    try:
        vert = fp.FloatingPondVertical.from_boundary(root, available_bytes_fn=q["available_bytes_fn"], reserve_bytes=q["reserve_bytes"], batch_cells=bc, memory_budget_bytes=q["memory_budget_bytes"],
                                                     host_memory_budget_bytes=q["host_memory_budget_bytes"], host_reserve_bytes=q["host_reserve_bytes"], host_available_bytes_fn=q["host_available_bytes_fn"])
        try:
            rec = fhc.export_held_context_cuda(vert, pref_spec, available_bytes_fn=_capped(q["available_bytes_fn"], q["memory_budget_bytes"], lambda: fcu.device_reader(cp)()), reserve_bytes=q["reserve_bytes"],
                                               host_available_bytes_fn=_capped(q["host_available_bytes_fn"], q["host_memory_budget_bytes"], sp.default_host_available_bytes), host_reserve_bytes=q["host_reserve_bytes"])
        except fh.HeldContextError as exc:
            if "reported available" in str(exc) or "declared" in str(exc):
                raise fsl.FloatingMemoryError(str(exc)) from None
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
        d_c, h_c = _carrier_need(event, vert, bed, windowed, ident)
        fcxc._admit(cp, dev, d_c, h_c + cold, q, "the CUDA lifecycle checkpoint cold carrier + child save (composed)")  # BEFORE the carrier is built
        state = event.continued_state(bed, vert.initial_soil_water(), routing=head.routing, depth_m=depth, qx_m2_s=qx, qy_m2_s=qy, t_s=float(closed.water.t_s) if closed is not None else float(head.t_s))
        lc._admit_state(event, vert, state, bed, float(state.t_s))
        ccc.save_floating_continued_checkpoint_cuda(staging / CHILD_DIR, state, event, ident, original_topology=original_topology, available_bytes_fn=q["available_bytes_fn"], reserve_bytes=q["reserve_bytes"],
                                                    host_available_bytes_fn=q["host_available_bytes_fn"], host_reserve_bytes=q["host_reserve_bytes"], memory_budget_bytes=q["memory_budget_bytes"],
                                                    host_memory_budget_bytes=q["host_memory_budget_bytes"], max_bytes=limit)
        del state
        child = json.loads((staging / CHILD_DIR / fc.MANIFEST_NAME).read_text())
        child_names = tuple(sorted(p.name for p in (staging / CHILD_DIR).iterdir()))
        storage = "windowed" if windowed else "dense"
        if set(child_names) != CHILD_NAMES or child.get("schema") != CHILD_SCHEMAS[storage]:
            _fail("the child checkpoint is not the exact closed continued CUDA layout")
        manifest = {"profile": PROFILE, "identity": ident, "source": pc.pond_checkpoint_source_pair(), "array_backend": "cupy", "hydraulic_backend": "cuda", "bed_storage": storage, "context_schema": fh.SCHEMA2,
                    "context_meta": meta, "context_arrays": table, "files": cc._file_table(staging, (ARRAYS_NAME,), limit)}
        cfiles = cc._file_table(staging / CHILD_DIR, child_names, limit)
        manifest["child"] = {"schema": child["schema"], "manifest_sha256": cfiles[fc.MANIFEST_NAME]["sha256"], "files": cfiles}
        manifest["lifecycle"] = record
        total = manifest["files"][ARRAYS_NAME]["size"] + sum(r["size"] for r in cfiles.values())
        if total > limit:
            _fail(f"the container needs {total} bytes (outer arrays + child files), above max_bytes = {limit}; nothing was published")
        sealed = cc._seal(manifest)
        text = pc._json_dumps(sealed)
        if len(text) > cc.MAX_MANIFEST_BYTES:
            _fail("the outer manifest exceeds the bounded size")
        for rel in (ARRAYS_NAME, *(f"{CHILD_DIR}/{n}" for n in child_names)):
            cc._fsync_file(staging / rel)
        with (staging / MANIFEST_NAME).open("x", encoding="utf-8") as h:
            h.write(text)
            h.flush()
            os.fsync(h.fileno())
        cc._fsync_dir(staging / CHILD_DIR)
        cc._fsync_dir(staging)
        cc._publish_noreplace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        cc._fsync_dir(parent)
    except OSError as exc:
        _fail(f"INDETERMINATE durability: the checkpoint was published at {target} but the parent directory fsync failed ({exc}); the published directory is preserved")
    return SavedLifecycleCheckpoint(target, sealed["manifest_sha256"], kind)


# ---------------------------------------------------------------------------------------------------------------------------------------------- load
def _preflight(path: Any, expected_identity: dict, expected_seal: Any, limit: int, cp: Any, dev: int, q: dict, batch_cells: int) -> tuple:
    """Everything verifiable WITHOUT a payload array (the CUDA mirror of the CPU preflight with the continued CUDA child, the H0 `/2` layout, the kernel descriptor and two pools)."""
    expected = pc._canonical_identity(expected_identity)
    seal = pc._hex(expected_seal, "expected_manifest_sha256") if isinstance(expected_seal, str) else _fail("expected_manifest_sha256 is REQUIRED (the sealed digest the writer returned)")
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _fail(f"checkpoint path is not a plain directory: {root}")
    root = root.resolve()
    if {p.name for p in root.iterdir()} != {MANIFEST_NAME, ARRAYS_NAME, CHILD_DIR}:
        _fail("lifecycle checkpoint directory must contain exactly the manifest, the arrays and the child directory")
    _typed(fcxc._regular, root / MANIFEST_NAME, "outer manifest")
    msize = int(os.lstat(root / MANIFEST_NAME).st_size)
    cold = max(int(fr._cold_source_bytes()), cc.COLD_ALLOWANCE_BYTES)
    fcxc._admit(cp, dev, 0, 3 * msize + cc.BUFFER_BYTES + cold, q, "the CUDA lifecycle manifest read and first-use source hash")
    try:
        data = cc._read_manifest(root, MANIFEST_NAME)
    except cc.ContextCheckpointError as exc:
        _fail(str(exc))
    if set(data) != MANIFEST_KEYS:
        _fail(f"outer manifest keys mismatch: missing {sorted(MANIFEST_KEYS - set(data))}, unknown {sorted(set(data) - MANIFEST_KEYS)}")
    if pc._hex(data["manifest_sha256"], "manifest_sha256") != cc._seal(data)["manifest_sha256"]:
        _fail("outer manifest hash mismatch (the manifest was modified)")
    if data["manifest_sha256"] != seal:
        _fail("the sealed manifest digest differs from the caller's EXPECTED digest (a different or re-sealed file)")
    if data["profile"] != PROFILE:
        _fail(f"unknown lifecycle profile {data['profile']!r} (this loader reads only {PROFILE}; the CPU `/1` file, the context profiles and the child profiles have their own loaders; nothing is migrated)")
    if data["identity"] != expected:
        _fail("lifecycle checkpoint identity differs from the expected identity")
    if data["source"] != pc.pond_checkpoint_source_pair():
        _fail("source pair mismatch: the imported packages differ from the ones that wrote the checkpoint (refused before any payload is read)")
    if data["array_backend"] != "cupy" or data["hydraulic_backend"] != "cuda" or data["context_schema"] != fh.SCHEMA2 or data["bed_storage"] not in STORAGES:
        _fail("unsupported backend, H0 context schema or bed storage (the CUDA lifecycle file needs cupy/cuda, the H0 /2 record and Dense or Windowed storage)")
    try:
        life = cpul._check_lifecycle(data["lifecycle"], data["bed_storage"])
    except cpul.LifecycleCheckpointError as exc:
        _fail(str(exc))
    files, child, table = data["files"], data["child"], data["context_arrays"]
    if not isinstance(files, dict) or set(files) != {ARRAYS_NAME} or not isinstance(table, dict):
        _fail("file table is malformed")
    if not isinstance(child, dict) or set(child) != {"schema", "manifest_sha256", "files"} or not isinstance(child["files"], dict) or set(child["files"]) != CHILD_NAMES:
        _fail("child block is malformed (closed native child names required)")
    if child["schema"] != CHILD_SCHEMAS[data["bed_storage"]]:
        _fail("child schema does not match the bed storage (the continued CUDA profile is required)")
    if child["manifest_sha256"] != child["files"][fc.MANIFEST_NAME].get("sha256"):
        _fail("child manifest digest differs from its file-table entry")
    child_dir = root / CHILD_DIR
    if child_dir.is_symlink() or not child_dir.is_dir() or {p.name for p in child_dir.iterdir()} != CHILD_NAMES:
        _fail("child directory entries differ from the closed native child layout")
    for n in CHILD_NAMES:
        _typed(fcxc._regular, child_dir / n, n)
    _typed(fcxc._regular, root / ARRAYS_NAME, ARRAYS_NAME)
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
    try:
        cman = cc._read_manifest(child_dir, fc.MANIFEST_NAME)
        meta = fh._check_meta(data["context_meta"], len(table))
        ids, facts = meta["identities"], meta["facts"]
        expect_arrays = fh._expected_arrays(meta)
    except (cc.ContextCheckpointError, fh.HeldContextError) as exc:
        _fail(f"the held-context metadata or the child manifest is invalid: {exc}")
    if meta["schema"] != fh.SCHEMA2 or meta["backend"] != "cuda":
        _fail("the held-context record must be the CUDA /2 record")
    live_kernel = fhc.kernel_descriptor()
    fl, lay = cman.get("floating"), meta["layouts"]
    if (cman.get("schema") != child["schema"] or cman.get("identity") != expected or cman.get("source") != data["source"] or cman.get("hydraulic_backend") != data["hydraulic_backend"]
            or cman.get("array_backend") != data["array_backend"] or cman.get("profile") != fcc.CONT_GPU_PROFILE or cman.get("kernel") != live_kernel or meta["kernel"] != live_kernel or not isinstance(fl, dict)
            or fl.get("root_column_sha256") != ids["root_column_sha256"] or fl.get("root_matrix_op_sha256") != ids["matrix_op_sha256"] or fl.get("root_pref_sha256") != ids["pref_sha256"]
            or fl.get("owner_config_sha256") != ids["config_sha256"] or fl.get("generation") != facts["generation"] or fl.get("source_witness") != facts["source_witness"]
            or fl.get("n_layers") != meta["n_layers"] or not isinstance(fl.get("shape"), list) or len(fl["shape"]) != 2 or int(fl["shape"][0]) * int(fl["shape"][1]) != meta["n_cells"]
            or fl.get("has_ks") != (meta["mode"] == "texture_ks") or fl.get("held_layouts") != {"theta_field_capacity": lay["theta_field_capacity"], "theta_saturation": lay["theta_saturation"]}
            or (cman.get("bed_storage") == "windowed") != (data["bed_storage"] == "windowed")):
        _fail("the child manifest is not bound to this context (schema/profile/identity/source/backends/kernel/H0 root ids/generation/witness/shape/layouts/Ks/storage)")
    init = (cman.get("binding") or {}).get("initial_topology")
    if (not isinstance(init, dict) or init.get("generation") != life["routing"]["generation"] or init.get("topology_sha256") != life["routing"]["topology_sha256"]
            or init.get("source_sha256") != life["routing"]["source_sha256"]):
        _fail("the lifecycle routing record differs from the child's carried routing origin")
    if life["kind"] == "closed_day" and (life["bridge"]["config_sha256"] != ids["config_sha256"] or life["bridge"]["root_column_sha256"] != ids["root_column_sha256"]
                                         or life["bridge"]["source_witness"] != facts["source_witness"] or life["bridge"]["property_generation"] != facts["generation"]):
        _fail("the daily bridge identities differ from the held-context root (config / column / witness / generation)")
    if set(table) != set(expect_arrays):
        _fail("the array table names differ from the analytic closed H0 layout")
    h0_bytes = 0
    for name, rec in table.items():
        if not isinstance(rec, dict) or set(rec) != {"dtype", "shape"} or not isinstance(rec["shape"], list) or any(type(n) is not int or n < 0 for n in rec["shape"]):
            _fail("array table entry is malformed")
        shape, dtype = expect_arrays[name]
        if tuple(rec["shape"]) != tuple(shape) or rec["dtype"] != np.dtype(dtype).str:
            _fail(f"array {name}: declared dtype/shape differ from the analytic H0 layout")
        h0_bytes += int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    c_dev, c_host, detail = _child_models(cman, meta, child["files"], batch_cells)
    x_dev, x_host = _daily_extra(life["kind"], meta)
    dev_raw, host_raw, stages = fcxc.compose_need(meta, msize, h0_bytes, c_dev + x_dev, c_host + x_host, cold)
    stages["child"] = detail
    stages["daily_extra"] = {"device": x_dev, "host": x_host}
    need = fcxc.ContextLoadNeed(dev_raw, host_raw, dev_raw + q["reserve_bytes"], host_raw + fcu.PY_ALLOWANCE_BYTES + q["host_reserve_bytes"], stages)
    return root, data, life, need


def lifecycle_load_need_bytes_cuda(path: Any, expected_identity: dict, expected_manifest_sha256: str, *, expected_device_id: int, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0,
                                   host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0, memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None,
                                   max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, batch_cells: int = cc.nfs_default_batch()) -> Any:
    """Verified scalar preflight (gate first, then the strict manifest/seal/binding checks, before any payload hash or array): the declared composed need of `load_floating_lifecycle_checkpoint_cuda`."""
    dev_id = _typed(fcxc._device_id, expected_device_id)
    q = _quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    limit = pc._max_bytes(max_bytes)
    bc = _typed(fcxc._batch, batch_cells)
    cp, dev = _typed(fcxc.gate_expected, dev_id)
    q = _resolve_readers(q, cp)
    return _preflight(path, expected_identity, expected_manifest_sha256, limit, cp, dev, q, bc)[3]


def load_floating_lifecycle_checkpoint_cuda(path: Any, factory: Callable[..., Any], expected_identity: dict, expected_manifest_sha256: str, *, original_topology: Any, expected_device_id: int, partition: Any = None,
                                            available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0,
                                            memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
                                            batch_cells: int = cc.nfs_default_batch()) -> RestoredLifecycle:
    """Restore a CUDA lifecycle checkpoint (module docstring). `partition` is REQUIRED for a closed-day file and refused for a boundary file."""
    dev_id = _typed(fcxc._device_id, expected_device_id)
    q = _quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    limit = pc._max_bytes(max_bytes)
    bc = _typed(fcxc._batch, batch_cells)
    if not callable(factory):
        _fail("factory must be callable: factory(vertical, carried_topology) -> PondEvent")
    if original_topology is None:
        _fail("original_topology (the genuine generation-0 PondTopology of the original case) is required")
    expected = pc._canonical_identity(expected_identity)
    cp, dev = _typed(fcxc.gate_expected, dev_id)  # FIRST: before any reader, path check, hash, array or callback
    q = _resolve_readers(q, cp)
    root, data, life, need = _preflight(path, expected, expected_manifest_sha256, limit, cp, dev, q, bc)
    kind = life["kind"]
    if (partition is None) != (kind == "boundary"):
        _fail("a closed-day file needs the caller's evaporation partition; a boundary file takes none")
    fcxc._admit(cp, dev, need.device_bytes, need.host_bytes, q, "the CUDA lifecycle checkpoint load (composed: H0 parse + reconstruct, retained context, continued child, closed-day context)")
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
        record = fh.record_from_parts(data["context_meta"], arrays, max_bytes=limit, available_bytes_fn=q["host_available_bytes_fn"], reserve_bytes=q["host_reserve_bytes"])
        del arrays
        ctx = fh.reconstruct_held_context(record, cc._expected_from(record.meta()), available_bytes_fn=q["available_bytes_fn"], reserve_bytes=q["reserve_bytes"],
                                          host_available_bytes_fn=q["host_available_bytes_fn"], host_reserve_bytes=q["host_reserve_bytes"])
    except fh.HeldContextError as exc:
        if "reported available" in str(exc) or "declared" in str(exc):
            raise fsl.FloatingMemoryError(str(exc)) from None
        _fail(f"the held context cannot be restored: {exc}")
    del record
    host_reader = q["host_available_bytes_fn"] if q["host_available_bytes_fn"] is not None else sp.default_host_available_bytes
    origin = ccc.read_continued_origin_cuda(child_dir, expected, available_bytes_fn=host_reader, reserve_bytes=q["host_reserve_bytes"], max_bytes=limit)  # header/table audit BEFORE the one decode
    topology = ccc.origin_topology(origin, original_topology)
    vert = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=q["available_bytes_fn"], reserve_bytes=q["reserve_bytes"], batch_cells=bc, memory_budget_bytes=q["memory_budget_bytes"],
                                                     host_memory_budget_bytes=q["host_memory_budget_bytes"], host_reserve_bytes=q["host_reserve_bytes"], host_available_bytes_fn=q["host_available_bytes_fn"])
    event = factory(vert, topology)  # the CALLER's static factory (the lifecycle's event recipe), outside every private mint window
    try:
        v = fcc.check_event(event)
    except fc.FloatingCheckpointError as exc:
        _fail(str(exc))
    if v is not vert or v._owner is not None or getattr(v, "_held", None) is not ctx:
        _fail("the factory's event must run the vertical this loader built from the verified context")
    if fcc.storage_of(event) != data["bed_storage"]:
        _fail("the factory's event storage differs from the file's bed storage")
    state = ccc.load_floating_continued_checkpoint_cuda(child_dir, event, expected, original_topology=original_topology, available_bytes_fn=q["available_bytes_fn"], reserve_bytes=q["reserve_bytes"],
                                                        host_available_bytes_fn=q["host_available_bytes_fn"], host_reserve_bytes=q["host_reserve_bytes"], memory_budget_bytes=q["memory_budget_bytes"],
                                                        host_memory_budget_bytes=q["host_memory_budget_bytes"], max_bytes=limit)
    try:
        token = lc.verified_restore(event, state, data["manifest_sha256"])
        lineage, head, broot = lc.stage_restored_head(token, "boundary" if kind == "boundary" else "daily_hold", life["head"], life["hold"], cuda_quota=q)
        closed = None
        if kind == "closed_day":
            from maple_syrup import floating_event_daily_bridge_cuda as bridge_cuda

            closed = bridge_cuda.restore_closed_day_cuda(token, head, broot, state, life["daily"], partition=partition, q=q)
            cpul._check_bridge(closed, life["bridge"])
        lc.install_restored_head(token, lineage, head)  # the head is installed LAST
    except (lc.LifecycleError, br.BridgeError) as exc:
        _fail(f"the lifecycle head cannot be restored: {exc}")
    return RestoredLifecycle(lineage, head, closed, kind, data["manifest_sha256"])

