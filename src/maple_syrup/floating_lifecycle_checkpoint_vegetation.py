"""Phase 2.5 CPU physical FILE of an accepted VEGETATION-BOUND lifecycle head -- a complete `Boundary` or a CLOSED daily state of the head `DailyHold`, each carrying ONE vegetation generation (NumPy/Numba x Dense / published Windowed).

PROFILE `maple-syrup-floating-lifecycle-checkpoint-vegetation/1`: a NEW explicit outer container beside the S1 profile `maple-syrup-floating-lifecycle-checkpoint/1`, which stays strict and LEAF-free. The two profiles refuse each other by
construction: a different manifest name (`floating_lifecycle_checkpoint_vegetation.json`), a different profile string, the held-context schema `maple_syrup.floating_held_context/3` (CPU; CUDA is `/4`, authored in the explicit twin
`floating_lifecycle_checkpoint_vegetation_cuda`) instead of `/1`, and the owning `lifecycle.saveable_head(vegetation=...)` hook which accepts exactly one kind of head per profile. There is no automatic migration or rebind.

CONTAINER (same closed layout as S1)
  floating_lifecycle_checkpoint_vegetation.json  strict outer manifest, sealed LAST: profile, identity, source pair, backends, bed storage, H0-VEG record meta + array table, file table, child digest, the closed scalar
                                                 LIFECYCLE record (S1 vocabulary) and the closed VEGETATION block
  context_arrays.npz                             the `/3` record arrays: the base H0 arrays AND the compact cached recipe (`veg.*`); stored `.npy` members, closed names
  child/                                         the genuine CONTINUED-ORIGIN child (unchanged codec, Dense `/1` or published Windowed `/2`): native bed, current W/J/I/D, surface, routing origin/generation, global datum

VEGETATION BLOCK (closed keys): the STABLE generation counter, `recipe_id` and `parent_recipe_id`, the binding / controls / projection / cached-record / package / helper / source / mask / surface identities and the HISTORICAL apply-time
provenance (`applied`: column / law / g / matrix / preferential ids of the moment the generation was applied -- never the current law). The CURRENT operator / column ids are those of the restored held context, recomputed canonically
from the compact cached recipe on the held column (`reconstruct_vegetation_held_context`) and compared with the child's root ids. A record that was itself restored from a file (cached-derived) saves its own cached recipe; a live
record is captured into a cached recipe by the public cache `capture_cached_inputs` (nothing of the plants, a PlantSession or a projection is stored). `origin_at_save` documents which of the two it was.

SAVE (`lineage` head + the genuine closed state for a hold): never steps, drains, evaporates, withdraws, grows, refreshes properties/routing, hands back, publishes a journal or mutates the head, the record or any counter. The eligibility
is the OWNING hook `saveable_head(lineage, vegetation=True)` (a head WITHOUT a vegetation generation, a pending storm / recession, an Occurrence, a mid-day trial and an unpublished Windowed bed are refused); the old profile's
guard is untouched. The caller's STATIC event recipe is `factory(vertical, carried_topology, vegetation_cover_fraction) -> PondEvent` and must build the native `PondEvent` with the BOUND splash cover it is handed.

LOAD (fresh process: static case metadata/topography + immutable files + pins ONLY; no original landscape/bed/owner/probe/provider/PlantSession). The caller supplies EXTERNALLY: the expected sealed manifest digest `save` returned, and the
ACTUAL runtime LEAF package digest and external helper pins (`leaf_package_sha256`, `external_helpers`; never read from the file only). Order: external seal + closed headers -> composed quota -> file / header / hash / source-pair checks ->
H0-VEG canonical context (the cache stage verifies the ACTUALLY imported LEAF package / helper bytes against the caller's pins; the other biological expectations -- record / binding / projection sha, source -- are taken from the header
ONLY AFTER its external seal and source-pair checks) -> static event built with the cached HOST splash cover (`veg.splash_cover`) -> continued child load (adoption last) -> one-shot `verified_restore` token -> the owning
`vegetation_lifecycle.restore_vegetation_record` (a cached-derived record bound to that token, the held recipe, the stable counter / recipe id / parent / historical provenance) -> `stage_restored_head(..., vegetation=record)` ->
(closed day) `restore_closed_day` -> `install_restored_head` LAST. A restored record is NOT a new live biology generation: its origin stays `cached_derived`, the counter / recipe id are preserved, and a later NEW live
`apply_vegetation_generation` still needs the actual LEAF producer proof and ordering. No head is minted from matching JSON hashes.

MEMORY (declared host model, admitted BEFORE arrays; reserves counted once): save = the S1 composed cold-carrier/child model + the vegetation H0 export model (`estimate_vegetation_held_bytes`: sealed copies + the verification
reconstruction) + the cache capture of a live recipe (`estimate_capture_bytes_from_binding`); load = outer manifest/buffers/cold allowance + max(raw npz read/copy + the H0-VEG reconstruction peak, retained context + the child's own
declared load model). No deep-bed clone, S x N history or full host water mirror exists; the vegetation covers are views of the sealed cached arrays.
Limits: CPU only (CUDA refused before any IO; the CUDA twin is authored, not run here); no pending storm / recession; the restored lineage is fresh; the caller's static factory is trusted only through the verified context / cover
checks; no scheduler, biology or wind state; every-N output cadence is NOT in this file.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_context_checkpoint as fcc
from maple_syrup import floating_continued_checkpoint as cc
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_held_context as fh
from maple_syrup import floating_held_context_vegetation as fhv
from maple_syrup import floating_lifecycle_checkpoint as lcp
from maple_syrup import floating_pond as fp
from maple_syrup import floating_resume as fr
from maple_syrup import pond_checkpoint as pc
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup import water_vegetation_binding as wvb
from maple_syrup import water_vegetation_held as wvh
from maple_syrup.checkpoint import DEFAULT_MAX_CHECKPOINT_BYTES

__all__ = ["MANIFEST_NAME", "PROFILE", "RestoredVegetationLifecycle", "SavedVegetationLifecycleCheckpoint", "VegetationLifecycleCheckpointError", "load_floating_lifecycle_vegetation_checkpoint",
           "save_floating_lifecycle_vegetation_checkpoint", "vegetation_load_need_bytes"]

PROFILE = "maple-syrup-floating-lifecycle-checkpoint-vegetation/1"
MANIFEST_NAME = "floating_lifecycle_checkpoint_vegetation.json"
ARRAYS_NAME = fcc.ARRAYS_NAME
CHILD_DIR = fcc.CHILD_DIR
CHILD_NAMES = fcc.CHILD_NAMES
CHILD_SCHEMAS = lcp.CHILD_SCHEMAS
MANIFEST_KEYS = lcp.MANIFEST_KEYS | {"vegetation"}
VEG_KEYS = frozenset({"generation", "recipe_id", "parent_recipe_id", "applied", "origin_at_save", "binding_sha256", "controls_sha256", "projection_sha256", "cached_record_sha256", "leaf_package_sha256",
                      "external_helpers", "source", "mask_sha256", "surface_sha256"})
APPLIED_KEYS = frozenset({"column_sha256", "law_sha256", "g_sha256", "matrix_op_sha256", "preferential_sha256"})
CONTEXT_SCHEMA = fhv.SCHEMA3
ORIGINS = ("live", "cached_derived")


class VegetationLifecycleCheckpointError(lcp.LifecycleCheckpointError):
    """Typed narrow refusal of the vegetation lifecycle checkpoint (a subclass of the S1 error; native memory refusals keep their own classes)."""


def _fail(message: str) -> None:
    raise VegetationLifecycleCheckpointError(message)


def _own(fn: Callable[..., Any], *a: Any, **k: Any) -> Any:
    """Run an implementation and re-type EVERY narrow refusal of the named OLD checkpoint families (`pond_checkpoint.PondCheckpointError` and its Floating / Context / Lifecycle subclasses, i.e. the old container, context, child
    and lifecycle-profile errors) as THIS profile's `VegetationLifecycleCheckpointError`. Native memory refusals (`NativeFloatingMemoryError`, `FloatingMemoryError`: `MemoryError` subclasses outside that family) and every other
    exception (a caller's factory, a real allocator `MemoryError`) pass through by class untouched; nothing is classified by message text."""
    try:
        return fn(*a, **k)
    except VegetationLifecycleCheckpointError:
        raise
    except pc.PondCheckpointError as exc:
        raise VegetationLifecycleCheckpointError(str(exc)) from None


@dataclass(frozen=True)
class SavedVegetationLifecycleCheckpoint:
    """`save`'s result: the published directory, the SEALED outer manifest digest the caller records and presents at load, and the kind."""

    path: Path
    manifest_sha256: str
    kind: str
    generation: int
    recipe_id: str


@dataclass(frozen=True, eq=False)
class RestoredVegetationLifecycle:
    """`load`'s result: the FRESH lineage (head installed LAST), the restored head, the kind, the sealed digest and for a closed day the genuine closed daily state. No event, phase array or token is retained."""

    lineage: Any
    head: Any
    closed: Any
    kind: str
    manifest_sha256: str
    generation: int
    recipe_id: str


def _budget(value: Any) -> int:
    if type(value) is not int or value < 1:
        _fail("memory_budget_bytes must be an explicit strict positive int (the composed model of the vegetation recipe is admitted against it)")
    return value


def _cpu_only(root: Any) -> None:
    if root.column.backend not in fc.CPU_BACKENDS:
        _fail("the CPU vegetation lifecycle checkpoint serves the CPU backends; CUDA is refused before any IO (use the explicit CUDA twin)")


def _record_of(head: Any, kind: str) -> Any:
    try:
        return vl.verify_record(head.root.vegetation if kind == "boundary" else head.vegetation)
    except vl.VegetationLifecycleError as exc:
        _fail(f"the head's vegetation record is not genuine: {exc}")


def _veg_block(rec: Any) -> dict:
    return {"generation": int(rec.generation), "recipe_id": rec.recipe_id, "parent_recipe_id": rec.parent_recipe_id,
            "applied": {"column_sha256": rec.applied_column_sha256, "law_sha256": rec.applied_law_sha256, "g_sha256": rec.applied_g_sha256, "matrix_op_sha256": rec.applied_matrix_op_sha256,
                        "preferential_sha256": rec.applied_preferential_sha256},
            "origin_at_save": "cached_derived" if rec.binding.origin == wvb.ORIGIN_CACHED else "live", "binding_sha256": rec.binding_sha256, "controls_sha256": rec.controls_sha256,
            "projection_sha256": rec.projection_sha256, "cached_record_sha256": None, "leaf_package_sha256": rec.leaf_package_sha256, "external_helpers": rec.external_helpers, "source": rec.source,
            "mask_sha256": rec.mask_sha256, "surface_sha256": rec.surface_sha256}


def _expected_source(src: Any) -> dict:
    """The caller-style EXPECTED source record: exactly the mandatory identity keys of its kind (the free-text `note` is not an identity)."""
    if type(src) is not dict or src.get("kind") not in wvb.SOURCE_KEYS:
        _fail("the recorded biological source is not one of the two known kinds")
    try:
        return {k: src[k] for k in wvb.SOURCE_KEYS[src["kind"]]}
    except KeyError:
        _fail("the recorded biological source lacks a mandatory identity key")


def _pins_of(cached: Any) -> Any:
    return fhv.ExpectedVegetationPins(cached.record_sha256, cached.binding_sha256, cached.content_sha256, json.loads(cached.package_json)["digest"], json.loads(cached.helpers_json), _expected_source(cached.source))


# ---------------------------------------------------------------------------------------------------------------------------------------------- save
def _save(path: Any, lineage: Any, factory: Callable[..., Any], identity: dict, *, closed: Any = None, original_topology: Any, available_bytes_fn: Callable[[], int],
                                                  memory_budget_bytes: int, reserve_bytes: int = 0, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
                                                  batch_cells: int = fp.nfs.DEFAULT_BATCH_CELLS) -> SavedVegetationLifecycleCheckpoint:
    """Write a NEW vegetation lifecycle checkpoint directory of the lineage's accepted vegetation-bound head (module docstring). `closed` is REQUIRED exactly when the head is a `DailyHold` and refused otherwise.
    `factory(vertical, carried_topology, vegetation_cover_fraction) -> PondEvent` is the caller's static event recipe. Nothing of the lineage, the closed state, the record, the bed or any journal is changed."""
    if not callable(factory) or not callable(available_bytes_fn):
        _fail("factory and available_bytes_fn must be callable")
    if original_topology is None:
        _fail("original_topology (the genuine generation-0 PondTopology of the original case) is required")
    mb = _budget(memory_budget_bytes)
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
        head_kind, head = lc.saveable_head(lineage, vegetation=True)
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
    root, bed, depth, qx, qy = lcp._carrier(head_kind, head, closed)
    _cpu_only(root)
    rec = _record_of(head, head_kind)
    windowed = fc.is_windowed_bed(bed)
    if windowed:
        pc._require_published(bed)
    record = lcp._lifecycle_record(kind, head, closed, None)
    block = _veg_block(rec)
    cold = max(int(fr._cold_source_bytes()), fcc.COLD_ALLOWANCE_BYTES)
    rec_bytes = len(json.dumps(record).encode()) + len(json.dumps(block).encode())
    live_binding = rec.binding.origin == wvb.ORIGIN_LIVE
    capture_need = int(wvh.estimate_capture_bytes_from_binding(rec.binding)["peak_bytes"]) if live_binding else 0
    fc._read(available_bytes_fn, 3 * rec_bytes + fcc.BUFFER_BYTES + cold + reserve + capture_need, "vegetation lifecycle checkpoint save (record, hash buffers, cold source pair, cache capture)")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=parent))
    try:
        try:
            cached = wvh.capture_cached_inputs(rec.binding, max_bytes=mb, reserve_bytes=reserve, available_bytes_fn=available_bytes_fn) if live_binding else rec.binding.projection
            wvh.verify_cached_inputs(cached)
        except wvb.VegetationBindingError as exc:
            if isinstance(exc, wvb.VegetationBindingMemoryError):
                raise fp.nfs.NativeFloatingMemoryError(str(exc)) from None
            _fail(f"the vegetation recipe cannot be captured: {exc}")
        block["cached_record_sha256"] = cached.record_sha256
        # a NEWLY captured recipe stays live until the child is saved: its retained resident bytes are part of EVERY later admission's reserve, ONCE (a restored record's recipe is caller-owned: outside the incremental baseline)
        retained = int(cached.resident_bytes) if live_binding else 0
        reserve_live = reserve + retained
        vert = fp.FloatingPondVertical.from_boundary(root, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_live, batch_cells=bc)
        try:
            hrec = fhv.export_held_context_vegetation(vert, cached, _pins_of(cached), memory_budget_bytes=mb, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_live)
        except fh.HeldContextError as exc:  # H0 has no typed memory subclass (out of scope): a generic typed staging refusal of THIS profile; the native memory classes (a MemoryError) propagate by type, never by message text
            _fail(f"the vegetation held context cannot be exported: {exc}")
        meta = hrec.meta()
        arrays = {k: np.ascontiguousarray(a) for k, a in hrec.arrays().items()}
        table = {k: {"dtype": a.dtype.str, "shape": [int(n) for n in a.shape]} for k, a in arrays.items()}
        if sum(a.nbytes for a in arrays.values()) > limit:
            _fail("the held-context arrays exceed max_bytes; nothing was written")
        np.savez(staging / ARRAYS_NAME, **arrays)
        del hrec, arrays
        event = factory(vert, head.routing[0], rec.splash_cover_2d)  # the caller's static recipe on the CARRIED routing topology and the BOUND splash cover
        lc._admit_native(event, vert)
        if event.control.bed_storage != ("windowed" if windowed else "dense"):
            _fail("the event's configured bed storage differs from the bed's storage")
        if not np.array_equal(np.asarray(event.veg), np.asarray(rec.splash_cover_2d)):
            _fail("the factory's event does not run the bound splash cover it was handed")
        lcp._admit_carrier(event, vert, bed, windowed, ident, available_bytes_fn, reserve_live)
        state = event.continued_state(bed, vert.initial_soil_water(), routing=head.routing, depth_m=depth, qx_m2_s=qx, qy_m2_s=qy, t_s=float(closed.water.t_s) if closed is not None else float(head.t_s))
        lc._admit_state(event, vert, state, bed, float(state.t_s))
        cc.save_floating_continued_checkpoint(staging / CHILD_DIR, state, event, ident, original_topology=original_topology, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_live, max_bytes=limit)
        del state
        child = json.loads((staging / CHILD_DIR / fc.MANIFEST_NAME).read_text())
        child_names = tuple(sorted(p.name for p in (staging / CHILD_DIR).iterdir()))
        if set(child_names) != CHILD_NAMES or child.get("schema") != CHILD_SCHEMAS[event.control.bed_storage]:
            _fail("the child checkpoint is not the exact closed continued layout")
        manifest = {"profile": PROFILE, "identity": ident, "source": pc.pond_checkpoint_source_pair(), "array_backend": "numpy", "hydraulic_backend": event.control.backend, "bed_storage": event.control.bed_storage,
                    "context_schema": CONTEXT_SCHEMA, "context_meta": meta, "context_arrays": table, "files": fcc._file_table(staging, (ARRAYS_NAME,), limit)}
        cfiles = fcc._file_table(staging / CHILD_DIR, child_names, limit)
        manifest["child"] = {"schema": child["schema"], "manifest_sha256": cfiles[fc.MANIFEST_NAME]["sha256"], "files": cfiles}
        manifest["lifecycle"] = record
        manifest["vegetation"] = block
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
        raise VegetationLifecycleCheckpointError(f"INDETERMINATE durability: the checkpoint was published at {target} but the parent directory fsync failed ({exc}); the published directory is preserved") from None
    return SavedVegetationLifecycleCheckpoint(target, sealed["manifest_sha256"], kind, int(rec.generation), rec.recipe_id)


# ---------------------------------------------------------------------------------------------------------------------------------------------- load
def _check_block(b: Any) -> dict:
    """Closed strict types of the vegetation block (before any array)."""
    if type(b) is not dict or set(b) != VEG_KEYS:
        _fail("vegetation block keys mismatch (closed key set)")
    g = b["generation"]
    if type(g) is not int or g < 1:
        _fail("vegetation generation must be a strict positive int")
    for k in ("recipe_id", "binding_sha256", "controls_sha256", "projection_sha256", "cached_record_sha256", "leaf_package_sha256", "mask_sha256", "surface_sha256"):
        pc._hex(b[k], f"vegetation.{k}")
    if (b["parent_recipe_id"] is None) != (g == 1):
        _fail("only generation 1 has no parent recipe")
    if b["parent_recipe_id"] is not None:
        pc._hex(b["parent_recipe_id"], "vegetation.parent_recipe_id")
    if type(b["applied"]) is not dict or set(b["applied"]) != APPLIED_KEYS:
        _fail("vegetation applied-provenance keys mismatch")
    for k in APPLIED_KEYS:
        pc._hex(b["applied"][k], f"vegetation.applied.{k}")
    if b["origin_at_save"] not in ORIGINS or type(b["external_helpers"]) is not list or type(b["source"]) is not dict:
        _fail("vegetation origin / helpers / source are malformed")
    return b


def _cross_check(block: dict, meta: dict) -> None:
    """The vegetation block must name the SAME recipe the sealed H0-VEG metadata carries (the cached part is the sole biological record)."""
    cm = meta["vegetation"]["cached"]
    pkg = str(cm["package"]["digest"]).removeprefix("sha256:")
    if (cm["record_sha256"], cm["binding_sha256"], str(cm["content_sha256"]).removeprefix("sha256:"), cm["mask_sha256"], cm["surface_sha256"], pkg, cm["controls"]["sha256"], cm["source"], cm["external_helpers"]) != (
            block["cached_record_sha256"], block["binding_sha256"], block["projection_sha256"], block["mask_sha256"], block["surface_sha256"], block["leaf_package_sha256"], block["controls_sha256"],
            block["source"], block["external_helpers"]):
        _fail("the vegetation block differs from the recipe recorded in the held-context metadata")
    if block["recipe_id"] != vl.recipe_id_of(block["binding_sha256"], block["generation"], block["parent_recipe_id"]):
        _fail("the recorded recipe id is not reproduced by the binding, the generation counter and the parent recipe")


def _preflight(path: Any, expected_identity: dict, expected_seal: Any, limit: int, available_bytes_fn: Callable[[], int], reserve: int, batch_cells: int) -> tuple:
    """Everything verifiable WITHOUT a payload array: stage-0 admission, the strict manifest, the caller's EXPECTED seal, bindings, closed layout, child cross-binding, the analytic closed H0-VEG table and the composed need."""
    expected = pc._canonical_identity(expected_identity)
    seal = pc._hex(expected_seal, "expected_manifest_sha256") if isinstance(expected_seal, str) else _fail("expected_manifest_sha256 is REQUIRED (the sealed digest the writer returned)")
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _fail(f"checkpoint path is not a plain directory: {root}")
    root = root.resolve()
    if {p.name for p in root.iterdir()} != {MANIFEST_NAME, ARRAYS_NAME, CHILD_DIR}:
        _fail("vegetation lifecycle checkpoint directory must contain exactly the vegetation manifest, the arrays and the child directory (an S1 lifecycle checkpoint is a different profile)")
    fcc._check_regular(root / MANIFEST_NAME, "outer manifest")
    cold = max(int(fr._cold_source_bytes()), fcc.COLD_ALLOWANCE_BYTES)
    msize = int(os.lstat(root / MANIFEST_NAME).st_size)
    fc._read(available_bytes_fn, 3 * msize + fcc.BUFFER_BYTES + cold + reserve, "vegetation lifecycle manifest read")
    data = fcc._read_manifest(root, MANIFEST_NAME)
    if set(data) != MANIFEST_KEYS:
        _fail(f"outer manifest keys mismatch: missing {sorted(MANIFEST_KEYS - set(data))}, unknown {sorted(set(data) - MANIFEST_KEYS)}")
    if pc._hex(data["manifest_sha256"], "manifest_sha256") != fcc._seal(data)["manifest_sha256"]:
        _fail("outer manifest hash mismatch (the manifest was modified)")
    if data["manifest_sha256"] != seal:
        _fail("the sealed manifest digest differs from the caller's EXPECTED digest (a different or re-sealed file)")
    if data["profile"] != PROFILE:
        _fail("unknown vegetation lifecycle profile")
    if data["identity"] != expected:
        _fail("vegetation lifecycle checkpoint identity differs from the expected identity")
    if data["source"] != pc.pond_checkpoint_source_pair():
        _fail("source pair mismatch: the imported packages differ from the ones that wrote the checkpoint (refused before any payload is read)")
    if data["array_backend"] != "numpy" or data["hydraulic_backend"] not in fc.CPU_BACKENDS or data["context_schema"] != CONTEXT_SCHEMA or data["bed_storage"] not in pc.STORAGES:
        _fail("unsupported backend, context schema or bed storage (CPU /3 only; CUDA /4 is refused before any IO)")
    life = lcp._check_lifecycle(data["lifecycle"], data["bed_storage"])
    block = _check_block(data["vegetation"])
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
        view, sz = fhv._check_meta(data["context_meta"], len(table))
        meta = data["context_meta"]
        ids, facts = meta["identities"], meta["facts"]
        expect_arrays = fhv._expected_all(view, sz)
    except fh.HeldContextError as exc:
        _fail(f"the vegetation held-context metadata is invalid: {exc}")
    _cross_check(block, meta)
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
        _fail("the array table names differ from the analytic closed H0-VEG layout")
    table_bytes = 0
    for name, rec in table.items():
        if not isinstance(rec, dict) or set(rec) != {"dtype", "shape"} or not isinstance(rec["shape"], list) or any(type(n) is not int or n < 0 for n in rec["shape"]):
            _fail("array table entry is malformed")
        shape, dtype = expect_arrays[name]
        if tuple(rec["shape"]) != tuple(shape) or rec["dtype"] != np.dtype(dtype).str:
            _fail(f"array {name}: declared dtype/shape differ from the analytic H0-VEG layout")
        table_bytes += int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    try:
        est = fhv.estimate_vegetation_held_bytes(meta)
    except (fh.HeldContextError, wvb.VegetationBindingError) as exc:
        _fail(f"the vegetation held-context model cannot be evaluated: {exc}")
    # retained while the child loads: the base context + the cached recipe the context KEEPS (the public exact `cache_resident_bytes` term; the transient `/3` record is released before the child)
    m = SimpleNamespace(reconstruct_bytes=int(est["reconstruct_bytes"]), retained_bytes=int(est["live_outside_bytes"]) - int(est["record_bytes"]) + int(est["cache_resident_bytes"]))
    child_need = fcc._child_need(cman, meta, child["files"], batch_cells)
    need = fcc._compose(msize, table_bytes, m, child_need, cold, reserve)
    return root, data, life, block, need


def _need(path: Any, expected_identity: dict, expected_manifest_sha256: str, *, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0,
                               max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, batch_cells: int = fp.nfs.DEFAULT_BATCH_CELLS) -> int:
    """Public verified preflight: the declared composed need of the load after the strict manifest / seal / binding checks and before any payload hash or array."""
    return _preflight(path, expected_identity, expected_manifest_sha256, pc._max_bytes(max_bytes), available_bytes_fn, fcc._reserve(reserve_bytes), fcc._batch(batch_cells))[4]


def _load(path: Any, factory: Callable[..., Any], expected_identity: dict, expected_manifest_sha256: str, *, leaf_package_sha256: Any, external_helpers: Any,
                                                  original_topology: Any, partition: Any = None, available_bytes_fn: Callable[[], int], memory_budget_bytes: int, reserve_bytes: int = 0,
                                                  max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, batch_cells: int = fp.nfs.DEFAULT_BATCH_CELLS) -> RestoredVegetationLifecycle:
    """Restore a vegetation lifecycle checkpoint (module docstring). `leaf_package_sha256` / `external_helpers` are the caller's EXTERNAL pins of the ACTUAL runtime LEAF package and helper files; `partition` is REQUIRED for a
    closed-day file and refused for a boundary file; `factory(vertical, topology, vegetation_cover_fraction)` is the caller's static event recipe and must build the event with the cover it is handed."""
    reserve = fcc._reserve(reserve_bytes)
    limit = pc._max_bytes(max_bytes)
    bc = fcc._batch(batch_cells)
    mb = _budget(memory_budget_bytes)
    if not callable(factory) or not callable(available_bytes_fn):
        _fail("factory and available_bytes_fn must be callable")
    if original_topology is None:
        _fail("original_topology (the genuine generation-0 PondTopology of the original case) is required")
    try:
        pkg = wvb._hex(leaf_package_sha256, "leaf_package_sha256")
        helpers = wvb._normalise_helpers(external_helpers, "external_helpers")
    except wvb.VegetationBindingError as exc:
        _fail(f"the caller's LEAF pins are malformed: {exc}")
    expected = pc._canonical_identity(expected_identity)
    root, data, life, block, need = _preflight(path, expected, expected_manifest_sha256, limit, available_bytes_fn, reserve, bc)
    try:
        file_helpers = wvb._normalise_helpers(block["external_helpers"], "vegetation.external_helpers")
    except wvb.VegetationBindingError as exc:
        _fail(f"the file's helper pins are malformed: {exc}")
    if pkg != block["leaf_package_sha256"] or helpers != file_helpers:  # the caller's EXTERNAL pins must equal the recorded biology package / helpers (the cache stage then checks the ACTUAL imported bytes against them)
        _fail("the caller's LEAF package / external helper pins differ from the ones the file was written under (refused before any payload is read)")
    kind = life["kind"]
    if (partition is None) != (kind == "boundary"):
        _fail("a closed-day file needs the caller's evaporation partition; a boundary file takes none")
    fc._read(available_bytes_fn, need, "vegetation lifecycle checkpoint load (composed)")
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
    pins = fhv.ExpectedVegetationPins(block["cached_record_sha256"], block["binding_sha256"], block["projection_sha256"], pkg, external_helpers, _expected_source(block["source"]))
    try:
        record = fhv.record_from_parts_vegetation(meta, arrays, max_bytes=limit, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
        del arrays
        ids = meta["identities"]
        ctx = fh.reconstruct_vegetation_held_context(record, fh.ExpectedHeldIdentities(ids["config_sha256"], ids["root_column_sha256"], ids["matrix_op_sha256"], ids["pref_sha256"]), pins,
                                                     memory_budget_bytes=mb, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
    except fh.HeldContextError as exc:  # generic typed staging refusal (H0 has no memory subclass); native memory classes propagate by type
        _fail(f"the vegetation held context cannot be restored: {exc}")
    del record
    hv = ctx.vegetation
    if hv is None or hv.binding_sha256 != block["binding_sha256"] or hv.record_sha256 != block["cached_record_sha256"]:
        _fail("the restored held context does not carry the recipe the vegetation block names")
    origin = cc.read_continued_origin(child_dir, expected, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
    topology = cc.origin_topology(origin, original_topology)
    vert = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve, batch_cells=bc)
    arrays_c = wvh.cached_compact(hv.cached)["arrays"]
    cover = arrays_c["splash_cover"].reshape(tuple(hv.cached.shape))  # the cached HOST splash cover (immutable view): the static event is CONSTRUCTED with it
    event = factory(vert, topology, cover)
    v = fcc._check_event(event)
    if v is not vert or v._owner is not None or getattr(v, "_held", None) is not ctx:
        _fail("the factory's event must run the vertical this loader built from the verified context")
    if event.control.bed_storage != data["bed_storage"] or event.control.backend != data["hydraulic_backend"]:
        _fail("the factory's event storage/backend differs from the file's")
    state = cc.load_floating_continued_checkpoint(child_dir, event, expected, original_topology=original_topology, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve, max_bytes=limit,
                                                  expected_bed_storage=data["bed_storage"])
    try:
        token = lc.verified_restore(event, state, data["manifest_sha256"])
        rec = vl.restore_vegetation_record(token, generation=block["generation"], recipe_id=block["recipe_id"], parent_recipe_id=block["parent_recipe_id"], applied=dict(block["applied"]), max_bytes=mb,
                                           reserve_bytes=reserve, available_bytes_fn=available_bytes_fn)
        lineage, head, broot = lc.stage_restored_head(token, "boundary" if kind == "boundary" else "daily_hold", life["head"], life["hold"], vegetation=rec)
        closed = None
        if kind == "closed_day":
            closed = br.restore_closed_day(token, head, broot, state, life["daily"], partition=partition, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
            lcp._check_bridge(closed, life["bridge"])
        lc.install_restored_head(token, lineage, head)  # the head is installed LAST
    except (lc.LifecycleError, br.BridgeError) as exc:
        _fail(f"the vegetation lifecycle head cannot be restored: {exc}")
    return RestoredVegetationLifecycle(lineage, head, closed, kind, data["manifest_sha256"], int(rec.generation), rec.recipe_id)


def save_floating_lifecycle_vegetation_checkpoint(path: Any, lineage: Any, factory: Callable[..., Any], identity: dict, *, closed: Any = None, original_topology: Any, available_bytes_fn: Callable[[], int],
                                                  memory_budget_bytes: int, reserve_bytes: int = 0, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
                                                  batch_cells: int = fp.nfs.DEFAULT_BATCH_CELLS) -> SavedVegetationLifecycleCheckpoint:
    """Public save (see `_save` and the module docstring): every refusal of the named old checkpoint families is re-typed as `VegetationLifecycleCheckpointError`; native memory classes pass through."""
    return _own(_save, path, lineage, factory, identity, closed=closed, original_topology=original_topology, available_bytes_fn=available_bytes_fn, memory_budget_bytes=memory_budget_bytes,
                reserve_bytes=reserve_bytes, max_bytes=max_bytes, batch_cells=batch_cells)


def load_floating_lifecycle_vegetation_checkpoint(path: Any, factory: Callable[..., Any], expected_identity: dict, expected_manifest_sha256: str, *, leaf_package_sha256: Any, external_helpers: Any,
                                                  original_topology: Any, partition: Any = None, available_bytes_fn: Callable[[], int], memory_budget_bytes: int, reserve_bytes: int = 0,
                                                  max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, batch_cells: int = fp.nfs.DEFAULT_BATCH_CELLS) -> RestoredVegetationLifecycle:
    """Public load (see `_load` and the module docstring); same error contract as the save."""
    return _own(_load, path, factory, expected_identity, expected_manifest_sha256, leaf_package_sha256=leaf_package_sha256, external_helpers=external_helpers, original_topology=original_topology,
                partition=partition, available_bytes_fn=available_bytes_fn, memory_budget_bytes=memory_budget_bytes, reserve_bytes=reserve_bytes, max_bytes=max_bytes, batch_cells=batch_cells)


def vegetation_load_need_bytes(path: Any, expected_identity: dict, expected_manifest_sha256: str, *, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0,
                               max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, batch_cells: int = fp.nfs.DEFAULT_BATCH_CELLS) -> int:
    """Public verified preflight: the declared composed need of the load after the strict manifest / seal / binding checks and before any payload hash or array; same error contract as the save."""
    return _own(_need, path, expected_identity, expected_manifest_sha256, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, max_bytes=max_bytes, batch_cells=batch_cells)
