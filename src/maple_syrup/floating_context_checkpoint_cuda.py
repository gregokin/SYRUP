"""G-F4: general CUDA held-context OUTER container around the UNCHANGED genuine CUDA floating child (`floating_checkpoint_cuda` `/7` Dense or `/8` native Windowed). AUTHOR-WRITTEN, NOT RUN ON A DEVICE (Root qualifies it on
hardware). Importing this module never imports CuPy.

New explicit profile `maple-syrup-floating-context-checkpoint/2-cuda`, a separate closed profile: the CPU container `/1` (`floating_context_checkpoint`) and the CPU/GPU child profiles are untouched, the CPU loader refuses this profile
by its exact `profile` check, this loader refuses `/1`, and nothing is migrated or rebound. Layout (same shape as `/1`): `floating_context_checkpoint.json` (strict closed manifest, sealed LAST) + `context_arrays.npz` (the genuine
H0 `/2` CUDA record's closed HOST arrays) + `child/` (the unchanged `floating_checkpoint_cuda` directory: its own schema, source pair, kernel, binding, deep-store references and validators are never rewritten or duplicated).
Scope: the CURRENT genuine H0 `/2` law context + the CURRENT W/J/I/D child only. Not a daily cursor, refresh/lifecycle epoch, wind, scheduler or LEAF container; an unsupported H0 version/profile is refused now and a later
lifecycle codec is a new explicit version.

SAVE (gate -> normalised arguments -> path/IO): export the H0 `/2` record from the event's vertical (`fhc.export_held_context_cuda`; a raw `PreferentialSpec` is required when a preferential operator exists, never inferred),
write the arrays, RELEASE the record, save the child through `fcc.save_floating_pond_checkpoint_cuda` (its own gate, admission, validation), hash everything, seal the manifest LAST, fsync payload files / child dir / staging, publish by
atomic `renameat2(RENAME_NOREPLACE)` (never replaces anything), then fsync the parent (a failure after publication is reported INDETERMINATE and the published directory is preserved). Windowed deep files are REFERENCED by the
child manifest: never copied, fsynced or mutated here.

LOAD ORDER: strict metadata-only argument validation (the explicit `expected_device_id` is a strict int) -> the CURRENT-device gate (`require_cuda` + `gate_device(cp, expected_device_id)`; no CUDA column exists yet, nothing is
faked) BEFORE any reader, path check, hash, array or callback -> stage-0 two-pool admission for the bounded manifest read -> strict manifest, source pair, profile, backends, H0 `/2` analytic metadata/array-table checks, the child
manifest cross-binding (source / kernel / config / root column-matrix-preferential ids / generation / source witness / shape / layouts / Ks / storage / schemas) and every file's size, ALL before any `np.load`, H0 copy, device
object or factory call -> the composed two-pool stage-1 admission (`context_load_need_bytes_cuda` numbers) -> hashes, ZIP/`.npy` header audits -> bounded `np.load` -> `fh.record_from_parts` (HOST) -> `fh.reconstruct_held_context`
(`/2`: its own gate and two pools; a tampered raw H0 property array with recomputed outer hashes fails here on the root column/matrix/preferential identity, BEFORE the factory) -> the caller's STATIC factory
`factory(held_context) -> PondEvent` (outside every private mint window; its own setup is admitted by its own quota) -> `fcc.load_floating_pond_checkpoint_cuda` (native validation, provisional scope, adoption LAST). Any
failure leaves the accepted source state untouched and the target retryable.

MEMORY (declared, header dimensions only; reserves and the 64 KiB `fcu.PY_ALLOWANCE_BYTES` counted once per admission by `fcu.admit`): PHASE A (parse + reconstruct) host = manifest x3 + buffers + first-use source hash + the raw
npz read and its copy (2 x H0 array bytes) + the H0 record allowance + `fhc.plan` host; device = `fhc.plan` device. PHASE B (retained context + child) host = retained host record parts + `fcc.load_models` host (which carries the
Windowed deep-store MAPPING-CACHE bound, never the resident archive size); device = retained device H0 maps + the vertical's two zero rows + `fcc.load_models` device (placed bed, Pond arrays, working, prepare, canonical rows).
The phases are released between each other: need = max(A, B). The child's table bytes are all treated as device-placed (an upper bound). EXCLUDED and stated: the caller's static factory setup (its own quota), the vertical's
retained models beyond the two zero rows, interpreter/JIT/driver/allocator pool, whole-process memory. The first-use source hash is charged once (`fr._cold_source_bytes()` is 0 when warm). No compression multiplier, no
cells x voxels x classes clone, no history term, no hot transfer.

Limits: same device class/CuPy build as the child; Windowed needs the referenced deep store retained and unmodified; every device behaviour is unrun here."""

from __future__ import annotations

import dataclasses
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
from maple_syrup import floating_held_context as fh
from maple_syrup import floating_held_context_cuda as fhc
from maple_syrup import floating_resume as fr
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import pond_checkpoint as pc
from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss
from maple_syrup.checkpoint import DEFAULT_MAX_CHECKPOINT_BYTES

__all__ = ["PROFILE", "ContextCheckpointCudaError", "ContextLoadNeed", "context_load_need_bytes_cuda", "load_floating_context_checkpoint_cuda", "save_floating_context_checkpoint_cuda", "save_need_bytes_cuda"]

PROFILE = "maple-syrup-floating-context-checkpoint/2-cuda"
MANIFEST_NAME = cc.MANIFEST_NAME
ARRAYS_NAME = cc.ARRAYS_NAME
CHILD_DIR = cc.CHILD_DIR
MANIFEST_KEYS = cc.MANIFEST_KEYS  # the CPU `/1` closed key set: the PROFILE (and backends/context schema) distinguish the two containers; no placeholder field is added
CHILD_NAMES = cc.CHILD_NAMES
CHILD_SCHEMAS = {"dense": fcc.GPU_SCHEMA, "windowed": fcc.GPU_WINDOWED_SCHEMA}
STORAGES = ("dense", "windowed")


class ContextCheckpointCudaError(cc.ContextCheckpointError):
    """Typed narrow refusal of the CUDA context container."""


def _fail(message: str) -> None:
    raise ContextCheckpointCudaError(message)


@dataclasses.dataclass(frozen=True)
class ContextLoadNeed:
    """Declared composed need of one load: RAW pool totals (the reserves and the 64 KiB allowance are added by `fcu.admit`), the INCLUSIVE totals (allowance + reserve once) and the stage breakdown."""

    device_bytes: int
    host_bytes: int
    device_total_bytes: int
    host_total_bytes: int
    stages: dict


# ---------------------------------------------------------------------------------------------------------------------------------------------
# arguments, gate, quota
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _device_id(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        _fail(f"expected_device_id must be a strict integer >= 0, got {value!r}")
    return int(value)


def _quota(available_bytes_fn: Any, reserve_bytes: Any, host_available_bytes_fn: Any, host_reserve_bytes: Any, memory_budget_bytes: Any, host_memory_budget_bytes: Any) -> dict:
    try:
        return fcc.quota_of(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    except fc.FloatingCheckpointError as exc:
        _fail(str(exc))
    return {}


def gate_expected(expected_device_id: int) -> tuple[Any, int]:
    """The CURRENT device must be the caller's explicit expected device BEFORE any reader, path check, hash, array or callback. No CUDA column exists yet (the H0 context is not rebuilt), so the qualified runtime gate is used
    directly with the explicit device contract."""
    try:
        cp = fcu._rc().require_cuda()
        return cp, fcu.gate_device(cp, expected_device_id)
    except (ss.SubsurfaceError, RuntimeError) as exc:
        _fail(f"the CUDA device gate refused: {exc}")
    raise AssertionError("unreachable")


def _regular(path: Path, what: str) -> None:
    try:
        cc._check_regular(path, what)
    except cc.ContextCheckpointError as exc:
        _fail(str(exc))


def _admit(cp: Any, dev: int, device_bytes: int, host_bytes: int, q: dict, what: str) -> None:
    fcu.admit(cp, dev, int(device_bytes), int(host_bytes), q, what)  # typed FloatingMemoryError propagates unchanged


# ---------------------------------------------------------------------------------------------------------------------------------------------
# models (pure arithmetic from verified header dimensions)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _child_models(cman: dict, meta: dict, child_files: dict, batch_cells: int) -> tuple[int, int, dict]:
    """`(device, host, detail)` of the CHILD load (`fcc.load_models`) from the child manifest's actual dimensions: shape, class count, phase bins, the declared companion table bytes and the native snapshot's recorded
    uncompressed member bytes (Dense) / stored file bytes plus the deep-store mapping-cache bound (Windowed). The manifest values are unverified here (the child re-verifies everything and refuses a lie)."""
    try:
        b, arrays = cman["binding"], cman["payload"]["arrays"]
        nc_phase, nb = b["n_classes"], b["phase_bins"]
        nc = meta["n_classes"]
        shape = cman["floating"]["shape"]
        table_bytes = 0
        for rec in arrays.values():
            if set(rec) != {"dtype", "shape"} or any(type(n) is not int or n < 0 for n in rec["shape"]):
                raise ValueError("array table entry")
            table_bytes += int(np.prod(rec["shape"], dtype=np.int64)) * np.dtype(rec["dtype"]).itemsize
        N = int(shape[0]) * int(shape[1])
        L = meta["n_layers"]
        windowed = cman["schema"] == fcc.GPU_WINDOWED_SCHEMA
        if windowed:
            native_bytes = int(child_files[pc.PAYLOADS[0]]["size"])
            V, K, cache = 0, int(nc), fc.deep_cache_bound_bytes(cman["deep_storage"], nc)
        else:
            nat = cman["native_dense"]
            native_bytes = int(nat["native_member_bytes"])
            V, K, cache = int(nat["voxel_shape"][2]), int(nat["voxel_shape"][3]), 0
        if not all(type(x) is int and x >= 0 for x in (nc_phase, nb, native_bytes, V, K, cache)):
            raise ValueError("dimensions")
        dev, host = fcc.load_models(L, N, nc_phase, nb, table_bytes=table_bytes, dev_table_bytes=table_bytes, native_bytes=native_bytes, V=V, K=K, n_active=N, batch_cells=batch_cells, windowed=windowed,
                                    cache_bytes=cache)
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"the child manifest dimensions are malformed: {exc}")
    return int(dev), int(host), {"table_bytes": table_bytes, "native_bytes": native_bytes, "V": V, "K": K, "cache_bytes": cache, "windowed": windowed}


def compose_need(meta: dict, manifest_bytes: int, h0_array_bytes: int, child_dev: int, child_host: int, cold: int) -> tuple[int, int, dict]:
    """The composed RAW `(device, host)` need and its stages (module docstring). Pure arithmetic over verified metadata."""
    L, N = meta["n_layers"], meta["n_cells"]
    plan_dev, plan_host, _export = fhc.plan(meta)
    ks = meta["mode"] == "texture_ks"
    retained_dev = fcu.retained_terms(L, N, dict(meta["layouts"]["all"]))["total"] + N * 8 + N + (N * 8 if ks else 0) + (L * N + N) * 8  # H0 device maps + D0/mask/Ks + the vertical's two zero rows
    retained_host = N * 8 + N + (N * 8 if ks else 0) + 2 * (L + 1) * 8
    always = 3 * int(manifest_bytes) + cc.BUFFER_BYTES + int(cold)
    a_host = 2 * int(h0_array_bytes) + fh.ALLOWANCE_BYTES + int(plan_host)
    b_host = retained_host + int(child_host)
    device = max(int(plan_dev), retained_dev + int(child_dev))
    host = always + max(a_host, b_host)
    stages = {"always_host": always, "A_host": a_host, "A_device": int(plan_dev), "B_host": b_host, "B_device": retained_dev + int(child_dev), "child_host": int(child_host), "child_device": int(child_dev),
              "retained_host": retained_host, "retained_device": retained_dev, "cold_source_bytes": int(cold)}
    return int(device), int(host), stages


# ---------------------------------------------------------------------------------------------------------------------------------------------
# preflight: everything verifiable WITHOUT reading a payload array
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _preflight(path: Any, expected_identity: dict, limit: int, cp: Any, dev: int, q: dict, batch_cells: int) -> tuple[Path, dict, dict, ContextLoadNeed]:
    expected = pc._canonical_identity(expected_identity)
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _fail(f"checkpoint path is not a plain directory: {root}")
    root = root.resolve()
    if {p.name for p in root.iterdir()} != {MANIFEST_NAME, ARRAYS_NAME, CHILD_DIR}:
        _fail("context checkpoint directory must contain exactly the manifest, the arrays and the child directory")
    _regular(root / MANIFEST_NAME, "outer manifest")
    msize = int(os.lstat(root / MANIFEST_NAME).st_size)
    cold = max(int(fr._cold_source_bytes()), cc.COLD_ALLOWANCE_BYTES)
    _admit(cp, dev, 0, 3 * msize + cc.BUFFER_BYTES + cold, q, "the CUDA context manifest read and first-use source hash")
    try:
        data = cc._read_manifest(root)
    except cc.ContextCheckpointError as exc:
        _fail(str(exc))
    if set(data) != MANIFEST_KEYS:
        _fail(f"outer manifest keys mismatch: missing {sorted(MANIFEST_KEYS - set(data))}, unknown {sorted(set(data) - MANIFEST_KEYS)}")
    if pc._hex(data["manifest_sha256"], "manifest_sha256") != cc._seal(data)["manifest_sha256"]:
        _fail("outer manifest hash mismatch (the manifest was modified)")
    if data["profile"] != PROFILE:
        _fail(f"unknown context profile {data['profile']!r} (this loader reads only {PROFILE}; the CPU container /1 has its own loader; nothing is migrated)")
    if data["identity"] != expected:
        _fail("context checkpoint identity differs from the expected identity")
    if data["source"] != pc.pond_checkpoint_source_pair():
        _fail("source pair mismatch: the imported packages differ from the ones that wrote the checkpoint (refused before any payload is read)")
    if data["array_backend"] != "cupy" or data["hydraulic_backend"] != "cuda" or data["context_schema"] != fh.SCHEMA2 or data["bed_storage"] not in STORAGES:
        _fail("unsupported backend, H0 context schema or bed storage (the CUDA container needs cupy/cuda, the H0 /2 record and Dense or Windowed storage)")
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
        _regular(child_dir / n, n)
    _regular(root / ARRAYS_NAME, ARRAYS_NAME)
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
        cman = cc._read_manifest(child_dir, fc.MANIFEST_NAME)  # bounded strict read of the CHILD manifest only (no payload)
        meta = fh._check_meta(data["context_meta"], len(table))  # closed analytic metadata check BEFORE any array; a /2 record only (a /1 or unknown version is refused)
        ids, facts = meta["identities"], meta["facts"]
        expect_arrays = fh._expected_arrays(meta)
    except (cc.ContextCheckpointError, fh.HeldContextError) as exc:
        _fail(f"the held-context metadata or the child manifest is invalid: {exc}")
    if meta["schema"] != fh.SCHEMA2 or meta["backend"] != "cuda":
        _fail("the held-context record must be the CUDA /2 record")
    live_kernel = fhc.kernel_descriptor()
    fl = cman.get("floating")
    lay = meta["layouts"]
    if (cman.get("schema") != child["schema"] or cman.get("identity") != expected or cman.get("source") != data["source"] or cman.get("hydraulic_backend") != data["hydraulic_backend"]
            or cman.get("array_backend") != data["array_backend"] or cman.get("kernel") != live_kernel or meta["kernel"] != live_kernel or not isinstance(fl, dict)
            or fl.get("root_column_sha256") != ids["root_column_sha256"] or fl.get("root_matrix_op_sha256") != ids["matrix_op_sha256"] or fl.get("root_pref_sha256") != ids["pref_sha256"]
            or fl.get("owner_config_sha256") != ids["config_sha256"] or fl.get("generation") != facts["generation"] or fl.get("source_witness") != facts["source_witness"]
            or fl.get("n_layers") != meta["n_layers"] or not isinstance(fl.get("shape"), list) or len(fl["shape"]) != 2 or int(fl["shape"][0]) * int(fl["shape"][1]) != meta["n_cells"]
            or fl.get("has_ks") != (meta["mode"] == "texture_ks")
            or fl.get("held_layouts") != {"theta_field_capacity": lay["theta_field_capacity"], "theta_saturation": lay["theta_saturation"]}
            or (cman.get("bed_storage") == "windowed") != (data["bed_storage"] == "windowed")):
        _fail("the child manifest is not bound to this context (schema/identity/source/backends/kernel/H0 root ids/generation/witness/shape/layouts/Ks/storage)")
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
    dev_raw, host_raw, stages = compose_need(meta, msize, h0_bytes, c_dev, c_host, cold)
    stages["child"] = detail
    need = ContextLoadNeed(dev_raw, host_raw, dev_raw + q["reserve_bytes"], host_raw + fcu.PY_ALLOWANCE_BYTES + q["host_reserve_bytes"], stages)
    return root, data, cman, need


def context_load_need_bytes_cuda(path: Any, expected_identity: dict, *, expected_device_id: int, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0,
                                 host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0, memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None,
                                 max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, batch_cells: int = cc.nfs_default_batch()) -> ContextLoadNeed:
    """Verified scalar preflight: the declared composed need of `load_floating_context_checkpoint_cuda` for this container (after the gate and the strict manifest/binding checks, before any payload hash or array). The
    raw totals and the allowance/reserve-inclusive totals are returned; the stage-0 manifest read is admitted from the caller's own readers."""
    dev_id = _device_id(expected_device_id)
    q = _quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    limit = pc._max_bytes(max_bytes)
    bc = _batch(batch_cells)
    cp, dev = gate_expected(dev_id)
    return _preflight(path, expected_identity, limit, cp, dev, q, bc)[3]


def _batch(value: Any) -> int:
    try:
        return cc._batch(value)
    except cc.ContextCheckpointError as exc:
        _fail(str(exc))
    return 0


# ---------------------------------------------------------------------------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _save_stages(v: Any, state: Any, event: Any, ident: dict, pref_spec: Any, limit: int) -> tuple[int, int, dict]:
    """RAW two-pool `(device_bytes, host_bytes, stages)` of one SAVE from header/metadata dimensions only (no array, no reader, no IO): PHASE A = the H0 `/2` export (`fhc.plan` export host) followed by its verification reconstruction
    (`fhc.plan` device + host with the exported record alive); PHASE A2 = the owned `np.ascontiguousarray` copies of the record's arrays alive together with the record plus the zip write (2 x record bytes + the 64 KiB record allowance);
    PHASE B = the unchanged child's OWN save model (`fcc.save_models`, same inputs as `fcc.save_floating_pond_checkpoint_cuda`, incl. the Windowed deep-store MAPPING-CACHE bound); PHASE C = the outer hash/seal buffers. The record and the
    arrays are released (`del rec, arrays`) before B, so the stages are sequential: need = max over phases per pool, plus the first-use source hash once. Reserves, the 64 KiB `fcu.PY_ALLOWANCE_BYTES` and the budgets are applied once by
    `fcu.admit`. Excluded: interpreter/JIT/driver/pool and whole-process memory."""
    t = v.held_template()
    meta, refs = fh._snapshot(v, pref_spec)  # metadata and REFERENCES only
    meta["schema"] = fh.SCHEMA2
    meta["kernel"] = fhc.kernel_descriptor()
    fh._check_meta(meta, len(refs) + (1 if pref_spec is not None and pref_spec.species_receiver_weights else 0))  # the exporter's own strict self-check, before any allocation
    plan_dev, plan_host, export_host = fhc.plan(meta)
    record_bytes = int(fh._analytic(meta)[1])
    bed = state.bed
    windowed = bool(fcc.is_windowed_bed(bed))
    L, N = v.n_layers, v.n_cells
    layouts = fcc._layouts_of(t.column)
    specs = {**fcc._specs(event, dry=False), **fcc._gpu_floating_specs(L, N, layouts, t.ks_m_s is not None)}
    if windowed:
        pc._require_published(bed)  # the child's own guard, BEFORE any deep metadata (no publication, no mutation)
        deep = pc._deep_record(bed.voxel_column, event)
        native_bytes = pc._windowed_resident_nbytes(bed, deep, ident)
        V, K = 0, int(bed.active_layer.mass_kg.shape[-1])
        cache = max(fc.deep_cache_bound_bytes(deep, event.nc), int(bed.voxel_column.deep_store.mapped_bytes_bound))
    else:
        vox = bed.voxel_column.mass_kg
        V, K = int(vox.shape[2]), int(vox.shape[3])
        native_bytes = pc._tree_nbytes(bed.voxel_column) + pc._tree_nbytes(bed.active_layer) + pc._tree_nbytes(bed.sediment_availability) + pc._tree_nbytes(bed.water)
        cache = 0
    n_active = int(np.count_nonzero(np.asarray(v._active_host())))
    b_dev, b_host = fcc.save_models(L, N, event.nc, event.control.phase_bins, table_bytes=fcc._spec_bytes(specs), native_bytes=native_bytes, V=V, K=K, n_active=n_active, batch_cells=v.batch_cells, windowed=windowed,
                                    cache_bytes=cache)
    cold = max(int(fr._cold_source_bytes()), cc.COLD_ALLOWANCE_BYTES)
    a_host = plan_host + export_host  # reconstruction host side with the exported record alive (the export's own host stage is smaller)
    a2_host = 2 * record_bytes + fh.ALLOWANCE_BYTES
    c_host = cc.BUFFER_BYTES
    device = max(plan_dev, b_dev)
    host = max(a_host, a2_host, b_host, c_host) + cold
    stages = {"A_export_verify": {"device": plan_dev, "host": a_host}, "A2_arrays": {"device": 0, "host": a2_host}, "B_child_save": {"device": b_dev, "host": b_host, "windowed": windowed, "V": V, "cache_bytes": cache},
              "C_outer": {"device": 0, "host": c_host}, "cold_source_bytes": cold, "record_bytes": record_bytes}
    return int(device), int(host), stages


def _stages_typed(v: Any, state: Any, event: Any, ident: dict, pref_spec: Any, limit: int) -> tuple[int, int, dict]:
    """`_save_stages` with the held-context metadata refusals (bad/missing `pref_spec`, missing Ks record, ...) converted to the container's typed error; memory refusals keep their class."""
    try:
        return _save_stages(v, state, event, ident, pref_spec, limit)
    except fh.HeldContextError as exc:
        if "reported available" in str(exc) or "declared" in str(exc):
            raise fsl.FloatingMemoryError(str(exc)) from None
        _fail(f"the held context cannot be exported: {exc}")
    raise AssertionError("unreachable")


def save_need_bytes_cuda(state: Any, event: Any, identity: dict, *, pref_spec: Any = None, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES) -> dict:
    """Public RAW composed SAVE need of this state/event (`_save_stages`): `{"device_bytes", "host_bytes", "stages"}` EXCLUDING the reserves and the 64 KiB allowance (`fcu.admit` adds each once: device = raw + reserve_bytes,
    host = raw + 64 KiB + host_reserve_bytes). The CURRENT-device gate runs first; metadata only, nothing allocated."""
    try:
        v = fcc.check_event(event)
    except fc.FloatingCheckpointError as exc:
        _fail(str(exc))
    ident = pc._canonical_identity(identity)
    try:
        fcc.gate(v.held_template().column)
    except fc.FloatingCheckpointError as exc:
        _fail(str(exc))
    d, h, stages = _stages_typed(v, state, event, ident, pref_spec, pc._max_bytes(max_bytes))
    return {"device_bytes": d, "host_bytes": h, "stages": stages}


def _capped(reader: Any, budget: Any, default: Callable[[], int]) -> Callable[[], int]:
    """A reader capped at the explicit budget for a nested helper that has no budget argument (`need > min(available, budget)` is exactly `need > available or need > budget`; reserves are NOT subtracted here)."""
    base = reader if reader is not None else default
    if budget is None:
        return base
    cap = int(budget)
    return lambda: min(int(base()), cap)


def save_floating_context_checkpoint_cuda(path: Any, state: Any, event: Any, identity: dict, *, pref_spec: Any = None, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0,
                                          host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0, memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None,
                                          max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES) -> Path:
    """Write a NEW CUDA context-container directory (module docstring)."""
    try:
        v = fcc.check_event(event)
    except fc.FloatingCheckpointError as exc:
        _fail(str(exc))
    q = _quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    limit = pc._max_bytes(max_bytes)
    ident = pc._canonical_identity(identity)
    try:
        cp, dev = fcc.gate(v.held_template().column)  # FIRST: the current device before any reader, IO, export or array
    except fc.FloatingCheckpointError as exc:
        _fail(str(exc))
    target = Path(path)
    if os.path.lexists(target):
        _fail(f"checkpoint path already exists (no overwrite): {target}")
    parent = target.resolve().parent
    if not parent.is_dir():
        _fail(f"the parent directory of the checkpoint does not exist: {parent}")
    target = parent / target.name
    storage = fcc.storage_of(event)
    if type(state) is not fcc.PondEventState or state.status != "wet" or not isinstance(state.bed, fcc.BedState) or storage != ("windowed" if fcc.is_windowed_bed(state.bed) else "dense"):
        _fail("save needs a genuine wet PondEventState on the event's configured bed storage")
    # the COMPOSED two-pool SAVE admission BEFORE the H0 export, any array/hash/copy or the staging directory (explicit budgets and readers honoured; reserves once)
    d_need, h_need, _stages = _stages_typed(v, state, event, ident, pref_spec, limit)
    _admit(cp, dev, d_need, h_need, q, "the CUDA context checkpoint save (H0 export + verification, array copies and the child save)")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=parent))
    try:
        try:
            rec = fhc.export_held_context_cuda(v, pref_spec, available_bytes_fn=_capped(q["available_bytes_fn"], q["memory_budget_bytes"], lambda: fcu.device_reader(cp)()), reserve_bytes=q["reserve_bytes"],
                                               host_available_bytes_fn=_capped(q["host_available_bytes_fn"], q["host_memory_budget_bytes"], sp.default_host_available_bytes), host_reserve_bytes=q["host_reserve_bytes"])
        except fh.HeldContextError as exc:
            if "reported available" in str(exc) or "declared" in str(exc):
                raise fsl.FloatingMemoryError(str(exc)) from None  # the memory refusal keeps its typed class
            _fail(f"the held context cannot be exported: {exc}")
        meta = rec.meta()
        arrays = {k: np.ascontiguousarray(a) for k, a in rec.arrays().items()}
        table = {k: {"dtype": a.dtype.str, "shape": [int(n) for n in a.shape]} for k, a in arrays.items()}
        if sum(a.nbytes for a in arrays.values()) > limit:
            _fail("the held-context arrays exceed max_bytes; nothing was written")
        np.savez(staging / ARRAYS_NAME, **arrays)
        del rec, arrays  # the record and its sealed copies are released BEFORE the child operation
        fcc.save_floating_pond_checkpoint_cuda(staging / CHILD_DIR, state, event, ident, available_bytes_fn=q["available_bytes_fn"], reserve_bytes=q["reserve_bytes"],
                                               host_available_bytes_fn=q["host_available_bytes_fn"], host_reserve_bytes=q["host_reserve_bytes"], memory_budget_bytes=q["memory_budget_bytes"],
                                               host_memory_budget_bytes=q["host_memory_budget_bytes"], max_bytes=limit)
        child = json.loads((staging / CHILD_DIR / fc.MANIFEST_NAME).read_text())
        child_names = tuple(sorted(p.name for p in (staging / CHILD_DIR).iterdir()))
        if set(child_names) != CHILD_NAMES or child.get("schema") != CHILD_SCHEMAS[storage]:
            _fail("the child checkpoint is not the exact closed /7-/8 layout")
        manifest = {"profile": PROFILE, "identity": ident, "source": pc.pond_checkpoint_source_pair(), "array_backend": "cupy", "hydraulic_backend": "cuda", "bed_storage": storage,
                    "context_schema": fh.SCHEMA2, "context_meta": meta, "context_arrays": table, "files": cc._file_table(staging, (ARRAYS_NAME,), limit)}
        cfiles = cc._file_table(staging / CHILD_DIR, child_names, limit)
        manifest["child"] = {"schema": child["schema"], "manifest_sha256": cfiles[fc.MANIFEST_NAME]["sha256"], "files": cfiles}
        total = manifest["files"][ARRAYS_NAME]["size"] + sum(r["size"] for r in cfiles.values())
        if total > limit:
            _fail(f"the container needs {total} bytes (outer arrays + child files), above max_bytes = {limit}; nothing was published")
        text = pc._json_dumps(cc._seal(manifest))
        if len(text) > cc.MAX_MANIFEST_BYTES:
            _fail("the outer manifest exceeds the bounded size")
        for rel in (ARRAYS_NAME, *(f"{CHILD_DIR}/{n}" for n in child_names)):  # durability: payload files first, then the directories, the manifest sealed LAST
            cc._fsync_file(staging / rel)
        with (staging / MANIFEST_NAME).open("x", encoding="utf-8") as h:
            h.write(text)
            h.flush()
            os.fsync(h.fileno())
        cc._fsync_dir(staging / CHILD_DIR)
        cc._fsync_dir(staging)
        cc._publish_noreplace(staging, target)  # atomic renameat2(RENAME_NOREPLACE): never replaces anything
    except BaseException:
        if staging.exists():  # only OUR temporary directory; a published target never exists under this name
            shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        cc._fsync_dir(parent)
    except OSError as exc:
        _fail(f"INDETERMINATE durability: the checkpoint was published at {target} but the parent directory fsync failed ({exc}); the published directory is preserved")
    return target


# ---------------------------------------------------------------------------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------------------------------------------------------------------------
def load_floating_context_checkpoint_cuda(path: Any, factory: Callable[[Any], Any], expected_identity: dict, *, expected_device_id: int, available_bytes_fn: Callable[[], int] | None = None,
                                          reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0, memory_budget_bytes: Any = None,
                                          host_memory_budget_bytes: Any = None, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, batch_cells: int = cc.nfs_default_batch()) -> tuple[Any, Any]:
    """Restore a CUDA context container: `(event, state)` where `event` is the static factory's event (built from the verified H0 `/2` context with `FloatingPondVertical.from_held_context`) and `state` the registered wet
    state (module docstring for the order)."""
    dev_id = _device_id(expected_device_id)
    q = _quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    limit = pc._max_bytes(max_bytes)
    if not callable(factory):
        _fail("factory must be callable: factory(held_context) -> PondEvent")
    expected = pc._canonical_identity(expected_identity)
    bc = _batch(batch_cells)
    cp, dev = gate_expected(dev_id)  # FIRST: before any reader, path check, hash, array or callback
    root, data, _cman, need = _preflight(path, expected, limit, cp, dev, q, bc)
    _admit(cp, dev, need.device_bytes, need.host_bytes, q, "the CUDA context checkpoint load (composed: H0 parse + reconstruct, retained context, child placed bed/preparation/canonical, mapping cache)")
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
        record = fh.record_from_parts(data["context_meta"], arrays, max_bytes=limit, available_bytes_fn=q["host_available_bytes_fn"], reserve_bytes=q["host_reserve_bytes"])
        del arrays
        ctx = fh.reconstruct_held_context(record, cc._expected_from(record.meta()), available_bytes_fn=q["available_bytes_fn"], reserve_bytes=q["reserve_bytes"],
                                          host_available_bytes_fn=q["host_available_bytes_fn"], host_reserve_bytes=q["host_reserve_bytes"])
    except fh.HeldContextError as exc:
        if "reported available" in str(exc) or "declared" in str(exc):
            raise fsl.FloatingMemoryError(str(exc)) from None
        _fail(f"the held context cannot be restored: {exc}")
    del record
    event = factory(ctx)  # the CALLER's static factory, outside every private mint window
    try:
        v = fcc.check_event(event)
    except fc.FloatingCheckpointError as exc:
        _fail(str(exc))
    if v._owner is not None or getattr(v, "_held", None) is not ctx:
        _fail("the factory's event must use FloatingPondVertical.from_held_context(<this context>) and no native owner")
    if fcc.storage_of(event) != data["bed_storage"]:
        _fail("the factory's event storage differs from the container's bed storage")
    if v.batch_cells > bc:
        _fail(f"the factory's vertical uses compact batch_cells={v.batch_cells} above the container's declared canonical-reader policy batch_cells={bc} (raise the declared policy so the published need covers it)")
    state = fcc.load_floating_pond_checkpoint_cuda(child_dir, event, expected, available_bytes_fn=q["available_bytes_fn"], reserve_bytes=q["reserve_bytes"], host_available_bytes_fn=q["host_available_bytes_fn"],
                                                   host_reserve_bytes=q["host_reserve_bytes"], memory_budget_bytes=q["memory_budget_bytes"], host_memory_budget_bytes=q["host_memory_budget_bytes"], max_bytes=limit)
    return event, state
