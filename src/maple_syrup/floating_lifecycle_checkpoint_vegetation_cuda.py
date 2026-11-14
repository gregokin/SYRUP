"""Phase 2.5 CUDA physical FILE of an accepted VEGETATION-BOUND lifecycle head (Dense and published native Windowed device beds). AUTHOR-WRITTEN, NOT RUN ON A DEVICE (Root qualifies it on hardware). Importing this module never imports CuPy.

PROFILE `maple-syrup-floating-lifecycle-checkpoint-vegetation/2-cuda`: the explicit CUDA twin of `floating_lifecycle_checkpoint_vegetation` (`/1`, CPU), carrying the dynamic-vegetation held context `maple_syrup.floating_held_context/4`
and the continued CUDA child. Same closed layout, the SAME scalar lifecycle record and the SAME closed vegetation block (`floating_lifecycle_checkpoint_vegetation._veg_block` / `_check_block` / `_cross_check` are REUSED, not
copied). The old profiles (`/1` CPU, `/2-cuda`, the vegetation CPU `/1`) refuse this one and it refuses them; the old held-context `/1 /2` codecs stay LEAF-free; nothing is migrated or rebound.

SAVE / LOAD: exactly the CPU vegetation contract on a device bed. SAVE never moves the head or the record and never uploads, steps, refreshes or hands back; eligibility is the owning `saveable_head(lineage, vegetation=True)`. The cached
recipe is HOST data (a live record is captured on the host under the HOST pool; a restored record saves its own recipe); the `/4` H0 record is exported by `export_held_context_vegetation` (current device FIRST, two pools, cold counted
chunked D2H of the device maps, the CURRENT device operator identities equal to the canonical law of the recipe). LOAD (external expected seal + the caller's ACTUAL LEAF package/helper pins, `expected_device_id`, two mandatory pools):
the current-device gate FIRST (before any reader, path check, hash, array or callback) -> strict manifest/seal/source-pair/profile/backends/`/4` layout/child cross-binding incl. the kernel/ABI descriptor -> the composed TWO-POOL need ->
hashes and archive/header audits -> bounded `np.load` -> `/4` record (host) -> `reconstruct_vegetation_held_context` (device) -> the host origin reader -> the vertical from the verified context -> the caller's static
`factory(vertical, topology, vegetation_cover_fraction)` built with the cached HOST cover -> the continued CUDA child load (provisional scope, adoption LAST) -> the one-shot `VerifiedRestore` token -> the owning
`restore_vegetation_record(..., cuda_quota=q)` (cached-derived record; the two device covers uploaded cold ONCE; the event's ACTUAL device cover compared resident) -> `stage_restored_head(..., cuda_quota=q, vegetation=record)` -> closed
day: `restore_closed_day_cuda` -> install LAST. A restored record is not a new live biology generation. No original bed, owner, probe, provider, plants or PlantSession exists on this path; native saved data stays backend-resident after
the cold staging; there is no per-step transfer, no deep-bed clone, no full host water mirror.

MEMORY: declared payload arithmetic from verified headers only (both pools, reserves and the 64 KiB allowance counted once per admission by `fcu.admit`); the `/4` model is `floating_held_context_vegetation_cuda.plan_cuda` and the
two resident device covers (2 N doubles) are added to the retained device bytes. Not claimed: any device run, a pending storm/recession file, scheduler/biology/wind state, large-catchment fit. Executable GPU definitions:
`tests/phase25_vegetation_checkpoint/test_gpu_definitions.py` (Root runs them; a missing GPU under the release flag is a setup failure).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
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
from maple_syrup import floating_held_context_vegetation as fhv
from maple_syrup import floating_held_context_vegetation_cuda as fhvc
from maple_syrup import floating_lifecycle_checkpoint as cpul
from maple_syrup import floating_lifecycle_checkpoint_cuda as s2
from maple_syrup import floating_lifecycle_checkpoint_vegetation as cpuv
from maple_syrup import floating_pond as fp
from maple_syrup import floating_resume as fr
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import pond_checkpoint as pc
from maple_syrup import spatial_soil_layers as sp
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup import water_vegetation_binding as wvb
from maple_syrup import water_vegetation_held as wvh
from maple_syrup.checkpoint import DEFAULT_MAX_CHECKPOINT_BYTES

__all__ = ["MANIFEST_NAME", "PROFILE", "VegetationLifecycleCheckpointCudaError", "load_floating_lifecycle_vegetation_checkpoint_cuda", "save_floating_lifecycle_vegetation_checkpoint_cuda",
           "vegetation_load_need_bytes_cuda"]

PROFILE = "maple-syrup-floating-lifecycle-checkpoint-vegetation/2-cuda"
MANIFEST_NAME = cpuv.MANIFEST_NAME
ARRAYS_NAME, CHILD_DIR, CHILD_NAMES = cpuv.ARRAYS_NAME, cpuv.CHILD_DIR, cpuv.CHILD_NAMES
MANIFEST_KEYS = cpuv.MANIFEST_KEYS
CHILD_SCHEMAS = s2.CHILD_SCHEMAS
STORAGES = s2.STORAGES
CONTEXT_SCHEMA = fhv.SCHEMA4
RestoredVegetationLifecycle = cpuv.RestoredVegetationLifecycle
SavedVegetationLifecycleCheckpoint = cpuv.SavedVegetationLifecycleCheckpoint


class VegetationLifecycleCheckpointCudaError(cpuv.VegetationLifecycleCheckpointError):
    """Typed narrow refusal of the CUDA vegetation lifecycle checkpoint."""


def _fail(message: str) -> None:
    raise VegetationLifecycleCheckpointCudaError(message)


def _own(fn: Callable[..., Any], *a: Any, **k: Any) -> Any:
    """Run an implementation and re-type EVERY narrow refusal of the named OLD / sibling checkpoint families (`pond_checkpoint.PondCheckpointError` and its Floating / Context / Lifecycle subclasses: the old S2 CUDA profile, the
    context and child containers and the CPU vegetation profile) as THIS profile's `VegetationLifecycleCheckpointCudaError`. Native memory refusals (`FloatingMemoryError`, `NativeFloatingMemoryError`: `MemoryError` subclasses outside
    that family) and every other exception pass through by class untouched; nothing is classified by message text."""
    try:
        return fn(*a, **k)
    except VegetationLifecycleCheckpointCudaError:
        raise
    except pc.PondCheckpointError as exc:
        raise VegetationLifecycleCheckpointCudaError(str(exc)) from None


def _typed(fn: Callable[..., Any], *a: Any, **k: Any) -> Any:
    """Call an accepted old helper (validators, gates, models) and re-type its narrow refusals as this profile's own."""
    return _own(fn, *a, **k)


def _quota(*a: Any) -> dict:
    return _typed(fcxc._quota, *a)


def _need_pools(q: dict) -> None:
    if q["host_memory_budget_bytes"] is None or q["memory_budget_bytes"] is None:
        _fail("the CUDA vegetation checkpoint requires BOTH explicit pools: memory_budget_bytes (device) and host_memory_budget_bytes (host); there is no default")


# ---------------------------------------------------------------------------------------------------------------------------------------------- save
def _save(path: Any, lineage: Any, factory: Callable[..., Any], identity: dict, *, closed: Any = None, original_topology: Any,
                                                       available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None,
                                                       host_reserve_bytes: Any = 0, memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
                                                       batch_cells: int = fp.nfs.DEFAULT_BATCH_CELLS) -> SavedVegetationLifecycleCheckpoint:
    """Write a NEW CUDA vegetation lifecycle checkpoint directory of the lineage's accepted vegetation-bound head (module docstring); same contract as the CPU `save_floating_lifecycle_vegetation_checkpoint` on a device bed."""
    if not callable(factory):
        _fail("factory must be callable")
    if original_topology is None:
        _fail("original_topology (the genuine generation-0 PondTopology of the original case) is required")
    q = _quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    _need_pools(q)
    limit = pc._max_bytes(max_bytes)
    bc = _typed(cc._batch, batch_cells)
    ident = pc._canonical_identity(identity)
    try:
        head_kind, head = lc.saveable_head(lineage, vegetation=True)
    except lc.LifecycleError as exc:
        _fail(str(exc))
    kind = "boundary" if head_kind == "boundary" else "closed_day"
    column = head.root.column if kind == "boundary" else head.column
    try:
        cp, dev = fcc.gate(column)  # FIRST: the current device before any reader, path check, IO, export or array
    except fc.FloatingCheckpointError as exc:
        _fail(str(exc))
    q = s2._resolve_readers(q, cp)
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
    root, bed, depth, qx, qy = s2._carrier_cuda(head_kind, head, closed)
    if root.column.backend != "cuda" or root.host_active is None or root.edges_host is None:
        _fail("the CUDA vegetation lifecycle checkpoint needs a CUDA root carrying its sealed host mask and edges (refused before any IO)")
    rec = cpuv._record_of(head, head_kind)
    if rec.splash_cover_device is None or rec.shade_cover_device is None:
        _fail("the CUDA vegetation record carries no resident device covers (refused)")
    windowed = fc.is_windowed_bed(bed)
    if windowed:
        pc._require_published(bed)
    record = cpul._lifecycle_record(kind, head, closed, None)
    block = cpuv._veg_block(rec)
    cold = max(int(fr._cold_source_bytes()), cc.COLD_ALLOWANCE_BYTES)
    rec_bytes = len(json.dumps(record).encode()) + len(json.dumps(block).encode())
    live_binding = rec.binding.origin == wvb.ORIGIN_LIVE
    capture_need = int(wvh.estimate_capture_bytes_from_binding(rec.binding)["peak_bytes"]) if live_binding else 0
    fcxc._admit(cp, dev, 0, 3 * rec_bytes + cc.BUFFER_BYTES + cold + capture_need, q, "the CUDA vegetation lifecycle checkpoint save (record, hash buffers, cold source pair, cache capture)")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=parent))
    try:
        try:
            cached = (wvh.capture_cached_inputs(rec.binding, max_bytes=q["host_memory_budget_bytes"], reserve_bytes=q["host_reserve_bytes"], available_bytes_fn=q["host_available_bytes_fn"])
                      if live_binding else rec.binding.projection)
            wvh.verify_cached_inputs(cached)
        except wvb.VegetationBindingError as exc:
            if isinstance(exc, wvb.VegetationBindingMemoryError):
                raise fsl.FloatingMemoryError(str(exc)) from None
            _fail(f"the vegetation recipe cannot be captured: {exc}")
        block["cached_record_sha256"] = cached.record_sha256
        # a NEWLY captured recipe is HOST data that stays live until the child is saved: its retained resident bytes join the HOST reserve of every later admission, ONCE (a restored record's recipe is caller-owned: outside the baseline)
        retained = int(cached.resident_bytes) if live_binding else 0
        q = {**q, "host_reserve_bytes": q["host_reserve_bytes"] + retained}
        vert = fp.FloatingPondVertical.from_boundary(root, available_bytes_fn=q["available_bytes_fn"], reserve_bytes=q["reserve_bytes"], batch_cells=bc, memory_budget_bytes=q["memory_budget_bytes"],
                                                     host_memory_budget_bytes=q["host_memory_budget_bytes"], host_reserve_bytes=q["host_reserve_bytes"], host_available_bytes_fn=q["host_available_bytes_fn"])
        try:
            hrec = fhv.export_held_context_vegetation(vert, cached, cpuv._pins_of(cached), memory_budget_bytes=q["memory_budget_bytes"], available_bytes_fn=q["available_bytes_fn"],
                                                      reserve_bytes=q["reserve_bytes"], host_memory_budget_bytes=q["host_memory_budget_bytes"], host_available_bytes_fn=q["host_available_bytes_fn"],
                                                      host_reserve_bytes=q["host_reserve_bytes"])
        except fh.HeldContextError as exc:  # H0 has no typed memory subclass (out of scope): a generic typed staging refusal of THIS profile; native memory classes propagate by type
            _fail(f"the vegetation held context cannot be exported: {exc}")
        meta = hrec.meta()
        arrays = {k: np.ascontiguousarray(a) for k, a in hrec.arrays().items()}
        table = {k: {"dtype": a.dtype.str, "shape": [int(n) for n in a.shape]} for k, a in arrays.items()}
        if sum(a.nbytes for a in arrays.values()) > limit:
            _fail("the held-context arrays exceed max_bytes; nothing was written")
        np.savez(staging / ARRAYS_NAME, **arrays)
        del hrec, arrays
        event = factory(vert, head.routing[0], rec.splash_cover_2d)  # the caller's static recipe: the carried routing topology and the BOUND HOST splash cover
        lc._admit_native(event, vert)
        if event.control.bed_storage != ("windowed" if windowed else "dense"):
            _fail("the event's configured bed storage differs from the bed's storage")
        from maple_syrup import floating_event_lifecycle_cuda as flc

        if not flc.arrays_equal_cuda(column, event.veg, rec.splash_cover_device):
            _fail("the factory's event does not run the bound splash cover it was handed (resident comparison)")
        d_c, h_c = _typed(s2._carrier_need, event, vert, bed, windowed, ident)
        fcxc._admit(cp, dev, d_c, h_c + cold, q, "the CUDA vegetation lifecycle checkpoint cold carrier + child save (composed)")
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
        manifest = {"profile": PROFILE, "identity": ident, "source": pc.pond_checkpoint_source_pair(), "array_backend": "cupy", "hydraulic_backend": "cuda", "bed_storage": storage,
                    "context_schema": CONTEXT_SCHEMA, "context_meta": meta, "context_arrays": table, "files": cc._file_table(staging, (ARRAYS_NAME,), limit)}
        cfiles = cc._file_table(staging / CHILD_DIR, child_names, limit)
        manifest["child"] = {"schema": child["schema"], "manifest_sha256": cfiles[fc.MANIFEST_NAME]["sha256"], "files": cfiles}
        manifest["lifecycle"] = record
        manifest["vegetation"] = block
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
    return SavedVegetationLifecycleCheckpoint(target, sealed["manifest_sha256"], kind, int(rec.generation), rec.recipe_id)


# ---------------------------------------------------------------------------------------------------------------------------------------------- load
def _compose_need(meta: dict, manifest_bytes: int, h0_array_bytes: int, child_dev: int, child_host: int, cold: int) -> tuple[int, int, dict]:
    """The composed RAW `(device, host)` need and its stages: the `fcxc.compose_need` arithmetic with the `/4` plan (`plan_cuda`) and the two resident device covers (2 N doubles) in the retained device bytes. Pure arithmetic."""
    L, N = meta["n_layers"], meta["n_cells"]
    plan_dev, plan_host, _export = fhvc.plan_cuda(meta)
    ks = meta["mode"] == "texture_ks"
    retained_dev = fcu.retained_terms(L, N, dict(meta["layouts"]["all"]))["total"] + N * 8 + N + (N * 8 if ks else 0) + (L * N + N) * 8 + 2 * N * 8
    # the HOST cached recipe the restored context KEEPS while the child loads: the public exact `cache_resident_bytes` term of the `/4` estimator (the transient record is released before the child)
    cache_resident = int(fhv.estimate_vegetation_held_bytes(meta)["cache_resident_bytes"])
    retained_host = N * 8 + N + (N * 8 if ks else 0) + 2 * (L + 1) * 8 + cache_resident
    always = 3 * int(manifest_bytes) + cc.BUFFER_BYTES + int(cold)
    a_host = 2 * int(h0_array_bytes) + fh.ALLOWANCE_BYTES + int(plan_host)
    b_host = retained_host + int(child_host)
    device = max(int(plan_dev), retained_dev + int(child_dev))
    host = always + max(a_host, b_host)
    stages = {"always_host": always, "A_host": a_host, "A_device": int(plan_dev), "B_host": b_host, "B_device": retained_dev + int(child_dev), "child_host": int(child_host), "child_device": int(child_dev),
              "retained_host": retained_host, "retained_device": retained_dev, "cache_resident_bytes": cache_resident, "cold_source_bytes": int(cold)}
    return int(device), int(host), stages


def _preflight(path: Any, expected_identity: dict, expected_seal: Any, limit: int, cp: Any, dev: int, q: dict, batch_cells: int) -> tuple:
    """Everything verifiable WITHOUT a payload array (the CUDA mirror of the CPU vegetation preflight with the continued CUDA child, the `/4` layout, the kernel descriptor and two pools)."""
    expected = pc._canonical_identity(expected_identity)
    seal = pc._hex(expected_seal, "expected_manifest_sha256") if isinstance(expected_seal, str) else _fail("expected_manifest_sha256 is REQUIRED (the sealed digest the writer returned)")
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _fail(f"checkpoint path is not a plain directory: {root}")
    root = root.resolve()
    if {p.name for p in root.iterdir()} != {MANIFEST_NAME, ARRAYS_NAME, CHILD_DIR}:
        _fail("vegetation lifecycle checkpoint directory must contain exactly the vegetation manifest, the arrays and the child directory")
    _typed(fcxc._regular, root / MANIFEST_NAME, "outer manifest")
    msize = int(os.lstat(root / MANIFEST_NAME).st_size)
    cold = max(int(fr._cold_source_bytes()), cc.COLD_ALLOWANCE_BYTES)
    fcxc._admit(cp, dev, 0, 3 * msize + cc.BUFFER_BYTES + cold, q, "the CUDA vegetation lifecycle manifest read and first-use source hash")
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
        _fail(f"unknown vegetation lifecycle profile {data['profile']!r} (this loader reads only {PROFILE}; the CPU vegetation file, the old lifecycle files and the child profiles have their own loaders; nothing is migrated)")
    if data["identity"] != expected:
        _fail("vegetation lifecycle checkpoint identity differs from the expected identity")
    if data["source"] != pc.pond_checkpoint_source_pair():
        _fail("source pair mismatch: the imported packages differ from the ones that wrote the checkpoint (refused before any payload is read)")
    if data["array_backend"] != "cupy" or data["hydraulic_backend"] != "cuda" or data["context_schema"] != CONTEXT_SCHEMA or data["bed_storage"] not in STORAGES:
        _fail("unsupported backend, held context schema or bed storage (the CUDA vegetation file needs cupy/cuda, the `/4` record and Dense or Windowed storage)")
    try:
        life = cpul._check_lifecycle(data["lifecycle"], data["bed_storage"])
        block = cpuv._check_block(data["vegetation"])
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
        meta = data["context_meta"]
        view, sz = fhv._check_meta(meta, len(table))
        ids, facts = meta["identities"], meta["facts"]
        expect_arrays = fhv._expected_all(view, sz)
    except (cc.ContextCheckpointError, fh.HeldContextError) as exc:
        _fail(f"the vegetation held-context metadata or the child manifest is invalid: {exc}")
    if meta["schema"] != CONTEXT_SCHEMA or meta["backend"] != "cuda":
        _fail("the held-context record must be the CUDA `/4` vegetation record")
    cpuv._cross_check(block, meta)
    live_kernel = fhc.kernel_descriptor()
    fl, lay = cman.get("floating"), meta["layouts"]
    if (cman.get("schema") != child["schema"] or cman.get("identity") != expected or cman.get("source") != data["source"] or cman.get("hydraulic_backend") != data["hydraulic_backend"]
            or cman.get("array_backend") != data["array_backend"] or cman.get("profile") != fcc.CONT_GPU_PROFILE or cman.get("kernel") != live_kernel or meta.get("kernel") != live_kernel or not isinstance(fl, dict)
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
        _fail("the array table names differ from the analytic closed H0-VEG layout")
    h0_bytes = 0
    for name, rec in table.items():
        if not isinstance(rec, dict) or set(rec) != {"dtype", "shape"} or not isinstance(rec["shape"], list) or any(type(n) is not int or n < 0 for n in rec["shape"]):
            _fail("array table entry is malformed")
        shape, dtype = expect_arrays[name]
        if tuple(rec["shape"]) != tuple(shape) or rec["dtype"] != np.dtype(dtype).str:
            _fail(f"array {name}: declared dtype/shape differ from the analytic H0-VEG layout")
        h0_bytes += int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    c_dev, c_host, detail = _typed(s2._child_models, cman, meta, child["files"], batch_cells)
    x_dev, x_host = _typed(s2._daily_extra, life["kind"], meta)
    dev_raw, host_raw, stages = _compose_need(meta, msize, h0_bytes, c_dev + x_dev, c_host + x_host, cold)
    stages["child"] = detail
    stages["daily_extra"] = {"device": x_dev, "host": x_host}
    need = fcxc.ContextLoadNeed(dev_raw, host_raw, dev_raw + q["reserve_bytes"], host_raw + fcu.PY_ALLOWANCE_BYTES + q["host_reserve_bytes"], stages)
    return root, data, life, block, need


def _need(path: Any, expected_identity: dict, expected_manifest_sha256: str, *, expected_device_id: int, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0,
                                    host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0, memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None,
                                    max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, batch_cells: int = cc.nfs_default_batch()) -> Any:
    """Verified scalar preflight (gate first, then the strict manifest/seal/binding checks, before any payload hash or array): the declared composed need of `load_floating_lifecycle_vegetation_checkpoint_cuda`."""
    dev_id = _typed(fcxc._device_id, expected_device_id)
    q = _quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    _need_pools(q)
    limit = pc._max_bytes(max_bytes)
    bc = _typed(fcxc._batch, batch_cells)
    cp, dev = _typed(fcxc.gate_expected, dev_id)
    q = s2._resolve_readers(q, cp)
    return _preflight(path, expected_identity, expected_manifest_sha256, limit, cp, dev, q, bc)[4]


def _load(path: Any, factory: Callable[..., Any], expected_identity: dict, expected_manifest_sha256: str, *, leaf_package_sha256: Any, external_helpers: Any,
                                                       original_topology: Any, expected_device_id: int, partition: Any = None, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0,
                                                       host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0, memory_budget_bytes: Any = None,
                                                       host_memory_budget_bytes: Any = None, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
                                                       batch_cells: int = cc.nfs_default_batch()) -> RestoredVegetationLifecycle:
    """Restore a CUDA vegetation lifecycle checkpoint (module docstring). `partition` is REQUIRED for a closed-day file and refused for a boundary file; `leaf_package_sha256` / `external_helpers` are the caller's EXTERNAL pins."""
    dev_id = _typed(fcxc._device_id, expected_device_id)
    q = _quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    _need_pools(q)
    limit = pc._max_bytes(max_bytes)
    bc = _typed(fcxc._batch, batch_cells)
    if not callable(factory):
        _fail("factory must be callable: factory(vertical, carried_topology, vegetation_cover_fraction) -> PondEvent")
    if original_topology is None:
        _fail("original_topology (the genuine generation-0 PondTopology of the original case) is required")
    try:
        pkg = wvb._hex(leaf_package_sha256, "leaf_package_sha256")
        helpers = wvb._normalise_helpers(external_helpers, "external_helpers")
    except wvb.VegetationBindingError as exc:
        _fail(f"the caller's LEAF pins are malformed: {exc}")
    expected = pc._canonical_identity(expected_identity)
    cp, dev = _typed(fcxc.gate_expected, dev_id)  # FIRST: before any reader, path check, hash, array or callback
    q = s2._resolve_readers(q, cp)
    root, data, life, block, need = _preflight(path, expected, expected_manifest_sha256, limit, cp, dev, q, bc)
    try:
        file_helpers = wvb._normalise_helpers(block["external_helpers"], "vegetation.external_helpers")
    except wvb.VegetationBindingError as exc:
        _fail(f"the file's helper pins are malformed: {exc}")
    if pkg != block["leaf_package_sha256"] or helpers != file_helpers:
        _fail("the caller's LEAF package / external helper pins differ from the ones the file was written under (refused before any payload is read)")
    kind = life["kind"]
    if (partition is None) != (kind == "boundary"):
        _fail("a closed-day file needs the caller's evaporation partition; a boundary file takes none")
    fcxc._admit(cp, dev, need.device_bytes, need.host_bytes, q, "the CUDA vegetation lifecycle checkpoint load (composed: `/4` parse + reconstruct, retained context + covers, continued child, closed-day context)")
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
    meta = data["context_meta"]
    pins = fhv.ExpectedVegetationPins(block["cached_record_sha256"], block["binding_sha256"], block["projection_sha256"], pkg, external_helpers, cpuv._expected_source(block["source"]))
    try:
        record = fhv.record_from_parts_vegetation(meta, arrays, max_bytes=limit, available_bytes_fn=q["host_available_bytes_fn"], reserve_bytes=q["host_reserve_bytes"])
        del arrays
        ids = meta["identities"]
        ctx = fh.reconstruct_vegetation_held_context(record, fh.ExpectedHeldIdentities(ids["config_sha256"], ids["root_column_sha256"], ids["matrix_op_sha256"], ids["pref_sha256"]), pins,
                                                     memory_budget_bytes=q["memory_budget_bytes"], available_bytes_fn=q["available_bytes_fn"], reserve_bytes=q["reserve_bytes"],
                                                     host_memory_budget_bytes=q["host_memory_budget_bytes"], host_available_bytes_fn=q["host_available_bytes_fn"], host_reserve_bytes=q["host_reserve_bytes"])
    except fh.HeldContextError as exc:  # generic typed staging refusal (H0 has no memory subclass); native memory classes propagate by type
        _fail(f"the vegetation held context cannot be restored: {exc}")
    del record
    hv = ctx.vegetation
    if hv is None or hv.binding_sha256 != block["binding_sha256"] or hv.record_sha256 != block["cached_record_sha256"]:
        _fail("the restored held context does not carry the recipe the vegetation block names")
    host_reader = q["host_available_bytes_fn"] if q["host_available_bytes_fn"] is not None else sp.default_host_available_bytes
    origin = ccc.read_continued_origin_cuda(child_dir, expected, available_bytes_fn=host_reader, reserve_bytes=q["host_reserve_bytes"], max_bytes=limit)  # header/table audit BEFORE the one decode
    topology = ccc.origin_topology(origin, original_topology)
    vert = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=q["available_bytes_fn"], reserve_bytes=q["reserve_bytes"], batch_cells=bc, memory_budget_bytes=q["memory_budget_bytes"],
                                                     host_memory_budget_bytes=q["host_memory_budget_bytes"], host_reserve_bytes=q["host_reserve_bytes"], host_available_bytes_fn=q["host_available_bytes_fn"])
    cover = wvh.cached_compact(hv.cached)["arrays"]["splash_cover"].reshape(tuple(hv.cached.shape))  # the cached HOST splash cover (immutable view): the static event is CONSTRUCTED with it
    event = factory(vert, topology, cover)
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
        rec = vl.restore_vegetation_record(token, generation=block["generation"], recipe_id=block["recipe_id"], parent_recipe_id=block["parent_recipe_id"], applied=dict(block["applied"]),
                                           max_bytes=q["host_memory_budget_bytes"], reserve_bytes=q["host_reserve_bytes"], available_bytes_fn=q["host_available_bytes_fn"], cuda_quota=q)
        lineage, head, broot = lc.stage_restored_head(token, "boundary" if kind == "boundary" else "daily_hold", life["head"], life["hold"], cuda_quota=q, vegetation=rec)
        closed = None
        if kind == "closed_day":
            from maple_syrup import floating_event_daily_bridge_cuda as bridge_cuda

            closed = bridge_cuda.restore_closed_day_cuda(token, head, broot, state, life["daily"], partition=partition, q=q)
            cpul._check_bridge(closed, life["bridge"])
        lc.install_restored_head(token, lineage, head)  # the head is installed LAST
    except (lc.LifecycleError, br.BridgeError) as exc:
        _fail(f"the vegetation lifecycle head cannot be restored: {exc}")
    return RestoredVegetationLifecycle(lineage, head, closed, kind, data["manifest_sha256"], int(rec.generation), rec.recipe_id)


def save_floating_lifecycle_vegetation_checkpoint_cuda(path: Any, lineage: Any, factory: Callable[..., Any], identity: dict, *, closed: Any = None, original_topology: Any,
                                                       available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None,
                                                       host_reserve_bytes: Any = 0, memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
                                                       batch_cells: int = fp.nfs.DEFAULT_BATCH_CELLS) -> SavedVegetationLifecycleCheckpoint:
    """Public CUDA save (see `_save` and the module docstring): every refusal of the named old / sibling checkpoint families is re-typed as `VegetationLifecycleCheckpointCudaError`; native memory classes pass through."""
    return _own(_save, path, lineage, factory, identity, closed=closed, original_topology=original_topology, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes,
                host_available_bytes_fn=host_available_bytes_fn, host_reserve_bytes=host_reserve_bytes, memory_budget_bytes=memory_budget_bytes, host_memory_budget_bytes=host_memory_budget_bytes,
                max_bytes=max_bytes, batch_cells=batch_cells)


def load_floating_lifecycle_vegetation_checkpoint_cuda(path: Any, factory: Callable[..., Any], expected_identity: dict, expected_manifest_sha256: str, *, leaf_package_sha256: Any, external_helpers: Any,
                                                       original_topology: Any, expected_device_id: int, partition: Any = None, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0,
                                                       host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0, memory_budget_bytes: Any = None,
                                                       host_memory_budget_bytes: Any = None, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
                                                       batch_cells: int = cc.nfs_default_batch()) -> RestoredVegetationLifecycle:
    """Public CUDA load (see `_load` and the module docstring); same error contract as the save."""
    return _own(_load, path, factory, expected_identity, expected_manifest_sha256, leaf_package_sha256=leaf_package_sha256, external_helpers=external_helpers, original_topology=original_topology,
                expected_device_id=expected_device_id, partition=partition, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, host_available_bytes_fn=host_available_bytes_fn,
                host_reserve_bytes=host_reserve_bytes, memory_budget_bytes=memory_budget_bytes, host_memory_budget_bytes=host_memory_budget_bytes, max_bytes=max_bytes, batch_cells=batch_cells)


def vegetation_load_need_bytes_cuda(path: Any, expected_identity: dict, expected_manifest_sha256: str, *, expected_device_id: int, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0,
                                    host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0, memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None,
                                    max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, batch_cells: int = cc.nfs_default_batch()) -> Any:
    """Public CUDA verified scalar preflight (see `_need`); same error contract as the save."""
    return _own(_need, path, expected_identity, expected_manifest_sha256, expected_device_id=expected_device_id, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes,
                host_available_bytes_fn=host_available_bytes_fn, host_reserve_bytes=host_reserve_bytes, memory_budget_bytes=memory_budget_bytes, host_memory_budget_bytes=host_memory_budget_bytes,
                max_bytes=max_bytes, batch_cells=batch_cells)
