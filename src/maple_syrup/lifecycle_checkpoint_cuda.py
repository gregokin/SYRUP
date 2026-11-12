"""Native9f CUDA physical lifecycle checkpoint — SAVE half (A6 G3; UNQUALIFIED, written without any GPU/device/CuPy execution).

Saves ONE accepted `LifecycleState` of the explicit native9f CUDA lifecycle (`wl.CUDA9F_BOUNDARY_PROFILE` / `wl.CUDA9F_PROFILE`) on the genuine ordinary Dense actual-9f
device bed: a closed `boundary` or an accepted `interevent` (source kind `storm_end` or, boundary profile only, `boundary_origin`). An incomplete storm/recession, a caller-side
pending request, a CPU/native39/unknown/subclass/Windowed lifecycle or bed is refused with a typed error before any I/O. It adds no physics, replays no entry/step/reset/close
and mutates no accepted state, bed or operator. It is a PHYSICAL checkpoint, not the MASY calendar restart, and there is NO load: `load_lifecycle_checkpoint_cuda` raises
`NotImplementedError` (a saved directory is never a restart claim until the separate G4 load/restore is qualified).

Layout (new schema, never accepted by the CPU codecs and refusing theirs): `lifecycle_checkpoint_cuda.json` (manifest, written last, no-clobber) + `maple_state.npz` (the
UNCHANGED genuine MAPLE 9f `save_state_snapshot`, fed the device leaves; it stages every host leaf at once) + `lifecycle_state.npz` (companion: soil, J, intake, faces, topology
source, ledger, committed). Native-first write order inside one atomic publication.

Source binding: the ACTUAL imported `maple_syrup` and `maple` packages are freshly hashed on every call (the changing Syrup package digest is captured, never compared with a fixed
self-hash); the MAPLE package must equal the literal 9f pin; the unchanged CPU helper modules reused here are pinned by their file digests taken from that SAME streamed
inventory. See docs/subsurface_hydrology/lifecycle_checkpoint_cuda_save.md. Trust boundary as the CPU codec: digests show consistency, not signatures.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from maple_syrup import checkpoint_capability as cc
from maple_syrup import interevent_transpiration as it
from maple_syrup import interevent_water as iw
from maple_syrup import layered_event as ev
from maple_syrup import layered_checkpoint as lc
from maple_syrup import layered_storm as ls
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup import lifecycle_checkpoint_cuda_plan as plan
from maple_syrup import lifecycle_checkpoint_cuda_policy as pol_mod
from maple_syrup import lifecycle_checkpoint_cuda_v2_plan as v2plan
from maple_syrup import pond_checkpoint as pc
from maple_syrup import provenance as pv
from maple_syrup import subsurface as ss
from maple_syrup import water_lifecycle as wl
from maple_syrup.sediment_bed import BedState, is_windowed_bed

__all__ = [
    "CudaSaveReceipt", "FILES", "LifecycleCheckpointCudaError", "LifecycleCheckpointCudaIndeterminate", "LifecycleCheckpointCudaQuotaError", "MANIFEST_NAME", "PROFILE",
    "SCHEMA", "estimate_lifecycle_budget_cuda", "load_lifecycle_checkpoint_cuda", "physical_digest_cuda", "save_lifecycle_checkpoint_cuda",
]

SCHEMA = pol_mod.V1.schema  # "maple-syrup-water-lifecycle-checkpoint-cuda/1" (the V1 literal policy is the single source of the V1 constants)
PROFILE = pol_mod.V1.profile  # "native9f_dense_lifecycle_cuda_v1"
MANIFEST_NAME = pol_mod.V1.manifest_name  # "lifecycle_checkpoint_cuda.json"
PAYLOAD_NATIVE = "maple_state.npz"
PAYLOAD_COMPANION = "lifecycle_state.npz"
FILES = (PAYLOAD_NATIVE, PAYLOAD_COMPANION)
NATIVE9F_PACKAGE_SHA256 = "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"  # literal MAPLE 9f package pin (equality-pinned to lk by a test)
#: Unchanged CPU helper MODULES this codec reuses, pinned by file sha256 against the freshly STREAMED package inventory (no inspect/linecache source copy).
HELPER_MODULE_PINS = {
    "layered_checkpoint.py": "4f4336d98f40febb56e15bcdb5a4c06e2e1dc1187c330d08b79849c4c0d72d51",
    "pond_checkpoint.py": "94cd2339566646ba44f2015dad13a67194c26a8dd3cc85b56916f62d2b4a704f",
    "lifecycle_checkpoint.py": "ff4d228e9464a29333a544e3ca098577df492d02661b9faaa7c7278c98e6d085",
    "checkpoint_capability.py": "36dc5d6a07245fca4badbadd0d754f2a250e688c2cc8b0b30d3a8c37c55ccbd6",
    "layered_storm.py": "bd125b743e58c588619a86ee952934deccea360ef3af215ae2b230af60d73c78",
}
_HELPER_MODULES = {"layered_checkpoint.py": lc, "pond_checkpoint.py": pc, "lifecycle_checkpoint.py": lk, "checkpoint_capability.py": cc, "layered_storm.py": ls}
MANIFEST_KEYS = pol_mod.V1.manifest_keys
SOURCE_KIND_PAIRS = pol_mod.V1.pairs
_RUNTIME_KEYS = frozenset({"python_implementation", "python_version", "byteorder", "machine", "numpy", "float64_eps", "cupy", "cuda_runtime", "cuda_driver",
                           "compute_capability", "device_name"})


class LifecycleCheckpointCudaError(RuntimeError):
    """Typed refusal; nothing sealed, accepted state/bed/operator unchanged."""


class LifecycleCheckpointCudaQuotaError(LifecycleCheckpointCudaError):
    """A caller quota, a fresh availability reading or the disk figure cannot cover a stage (nothing allocated/written beyond the named stage)."""


class LifecycleCheckpointCudaIndeterminate(LifecycleCheckpointCudaError):
    """Failure AFTER the manifest seal link: the directory may be durable; nothing was deleted and it must not be retried over."""


def _refuse(message: str) -> None:
    raise LifecycleCheckpointCudaError(message)


def _convert(exc: BaseException) -> LifecycleCheckpointCudaError:
    if isinstance(exc, LifecycleCheckpointCudaError):
        return exc
    if isinstance(exc, lc.LayeredCheckpointQuotaError):
        return LifecycleCheckpointCudaQuotaError(str(exc))
    return LifecycleCheckpointCudaError(f"{type(exc).__name__}: {exc}")


def _select_policy(policy: Any) -> Any:
    """Registered literal policy by identity (None -> V1); an unknown/forged policy or a policy whose writer is not implemented is refused BEFORE any side effect."""
    try:
        pol = pol_mod.policy_of(policy)
    except pol_mod.CodecPolicyError as exc:
        raise LifecycleCheckpointCudaError(str(exc)) from None
    if not pol.implemented:
        _refuse(f"the {pol.name} codec policy is declared but its writer is not implemented (no public V2 save exists yet)")
    return pol


@dataclass(frozen=True)
class CudaSaveReceipt:
    path: str
    profile: str
    role: str
    source_kind: str | None
    manifest_sha256: str
    files: dict
    sizes: dict
    t_s: float
    revision: int
    state_digest: str
    physical_digest: str
    material_sha256: str
    budget: dict
    source: dict
    runtime: dict
    transfers: dict
    adopted: bool = False


def load_lifecycle_checkpoint_cuda(*args: Any, **kwargs: Any) -> Any:
    """NOT IMPLEMENTED (G4). A saved directory is not a restart claim."""
    raise NotImplementedError("the native9f CUDA lifecycle checkpoint has no load/restore yet (G4); a saved directory is NOT a restart-ready claim")


# --- argument fences (pure, before any device/reader/hash/file work) --------------------------------------------------------------------
def _reserve(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        _refuse(f"{name} must be a non-Boolean integer >= 0, got {value!r}")
    return int(value)


def _reader(fn: Any, name: str) -> Callable[[], int] | None:
    """`None` is the ONLY value that selects the default; every other value must be callable and is honoured whatever its truthiness (never `or`/`not`/`if fn`)."""
    if fn is not None and not callable(fn):
        _refuse(f"{name} must be callable or None")
    return fn


def _read(fn: Callable[[], int], what: str) -> int:
    value = fn()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        _refuse(f"the availability reader for {what} must return a non-Boolean integer >= 0, got {value!r}")
    return int(value)


def _fence_types(lop: Any, state: Any, _policy: Any = None) -> tuple[str, str | None]:
    pol = pol_mod.policy_of(_policy)
    if type(lop) is not wl.LifecycleOperator or getattr(lop, "_seal", None) is not wl._SEAL:
        _refuse(f"lop must be a genuine LifecycleOperator, got {type(lop).__name__}")
    if lop._cuda is None or lop.backend != "cuda" or lop.xp is np:
        _refuse("the CUDA lifecycle checkpoint needs a genuine CUDA lifecycle operator (a CPU/NumPy/Numba lifecycle uses the unchanged CPU codecs)")
    if lop.context is None:
        _refuse("a bedless lifecycle has no native9f CUDA checkpoint profile (Dense-only)")
    if type(state) is not wl.LifecycleState:
        _refuse("state must be a genuine accepted LifecycleState")
    if state.role not in pol.roles:
        _refuse(f"only a closed 'boundary' or an accepted 'interevent' is checkpointed, got {state.role!r} (an incomplete storm/recession is not a public accepted boundary)")
    kind = None
    if state.role == "interevent":
        try:
            kind = iw.source_kind(state.payload.source)
        except iw.InterEventWaterError as exc:
            raise _convert(exc) from None
        if kind not in plan.SOURCE_KINDS:
            _refuse(f"unknown interevent source kind {kind!r}")
    return state.role, kind


def _is_device_leaf(a: Any, device_id: Any) -> bool:
    return (not isinstance(a, np.ndarray) and type(a).__module__.split(".")[0] == "cupy" and not isinstance(device_id, bool)
            and isinstance(device_id, int) and int(a.device.id) == device_id)


def _fence_bed(lop: Any, state: Any, owner: Any) -> Any:
    a = wl._arrays_of(state)
    bed = a.bed
    if type(bed) is not BedState:
        _refuse("the state's bed must be the genuine sediment_bed.BedState")
    try:
        windowed = is_windowed_bed(bed)
    except Exception as exc:
        raise _convert(exc) from None
    from maple.core.types.voxel import VoxelColumnState

    if windowed or type(bed.voxel_column) is not VoxelColumnState:
        _refuse("an actual native Windowed (or foreign) bed has no CUDA lifecycle checkpoint profile")
    if a.surface_m is not bed.water.depth_m:
        _refuse("the state's surface must BE the device bed's water depth (the single H authority)")
    for path, leaf in plan.leaf_occurrences(bed):
        if not _is_device_leaf(leaf, owner.device_id):
            _refuse(f"{path} is not a CuPy array on the lifecycle's bound device {owner.device_id!r}; refused (no host/CPU bed is saved by the CUDA codec)")
    for name in ("surface_m", "qx", "qy"):
        if not _is_device_leaf(getattr(a, name), owner.device_id):
            _refuse(f"state {name} is not a device array on the bound device")
    for name, arr in (("soil.water_m", a.soil.water_m), ("history.wetting_m", a.history.wetting_m), ("history.cumulative_intake_m", a.history.cumulative_intake_m)):
        if not _is_device_leaf(arr, owner.device_id):
            _refuse(f"state {name} is not a device array on the bound device")
    return bed


# --- source / runtime -------------------------------------------------------------------------------------------------------------------
def _fresh_source_pair() -> dict:
    """The ACTUAL imported packages freshly hashed (no cache). MAPLE must equal the literal 9f pin; the reused CPU helper modules must be the files that were hashed and equal
    their pins (taken from the streamed inventory). The Syrup tree digest is recorded dynamically (this module is part of it)."""
    import maple
    import maple_syrup

    try:
        root = Path(maple_syrup.__file__).resolve().parent
        tree = pv.source_tree_digest(root)
        mtree = pv.source_tree_digest(Path(maple.__file__).resolve().parent)
    except (pv.ProvenanceScopeError, OSError) as exc:
        raise LifecycleCheckpointCudaError(f"the source pair of the imported packages cannot be computed: {exc}") from None
    if mtree.digest_sha256 != NATIVE9F_PACKAGE_SHA256:
        _refuse(f"the imported maple package {mtree.digest_sha256} is not the accepted native9f source {NATIVE9F_PACKAGE_SHA256}; refused before any native writer")
    files = dict(tree.files)
    for name, pin in HELPER_MODULE_PINS.items():
        mod = _HELPER_MODULES[name]
        if Path(mod.__file__).resolve() != root / name:
            _refuse(f"the imported helper module {name} is not the file of the hashed package")
        if files.get(name) != pin:
            _refuse(f"helper module {name} differs from its pinned digest ({files.get(name)} != {pin}); refused")
    return {"maple_syrup_tree_sha256": tree.digest_sha256, "maple_syrup_n_files": int(tree.file_count), "maple_package_sha256": mtree.digest_sha256,
            "algorithm": str(tree.algorithm)}


def _runtime_record(lop: Any, device_id: int) -> dict:
    """Freshly inspected CPU+CUDA runtime of the lifecycle's BOUND device (called only after the current-device gate): CuPy, CUDA runtime/driver, compute capability, name.
    No device ordinal/UUID is recorded and no other device is enumerated."""
    base = lc.runtime_profile("numpy")
    base.pop("numba", None)
    base.pop("llvmlite", None)
    xp = lop.xp
    rt = xp.cuda.runtime
    props = rt.getDeviceProperties(int(device_id))
    name = props["name"]
    name = name.decode("ascii", "replace") if isinstance(name, bytes) else str(name)
    base.update(cupy=str(xp.__version__), cuda_runtime=str(rt.runtimeGetVersion()), cuda_driver=str(rt.driverGetVersion()),
                compute_capability=f"{int(props['major'])}.{int(props['minor'])}", device_name=name[:64])
    if set(base) != _RUNTIME_KEYS:
        _refuse("internal: runtime record key set")
    return base


# --- device-aware digests (never the NumPy-only lc._digest_value) -------------------------------------------------------------------------
def _digest_xp(h: Any, value: Any, depth: int = 0) -> None:
    if depth > 16:
        _refuse("digest nesting too deep")
    if value is None:
        h.update(b"N\0")
    elif isinstance(value, (bool, np.bool_)):
        h.update(b"B1\0" if value else b"B0\0")
    elif isinstance(value, (int, np.integer)):
        h.update(b"I" + str(int(value)).encode() + b"\0")
    elif isinstance(value, (float, np.floating)):
        h.update(b"F" + float(value).hex().encode() + b"\0")
    elif isinstance(value, str):
        h.update(b"S" + value.encode() + b"\0")
    elif isinstance(value, np.ndarray) or plan._is_leaf(value):  # host buffer or device array: `_stream_hash` stages a device array in <= HASH_CHUNK_BYTES chunks
        h.update(b"A")
        ls._stream_hash(h, value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        h.update(b"D" + type(value).__name__.encode() + b"\0")
        for f in dataclasses.fields(value):
            h.update(f.name.encode() + b"\0")
            _digest_xp(h, getattr(value, f.name), depth + 1)
    elif isinstance(value, (tuple, list)):
        h.update(b"T" + str(len(value)).encode() + b"\0")
        for item in value:
            _digest_xp(h, item, depth + 1)
    else:
        _refuse(f"digest: unsupported value {type(value).__name__}")


def _material_sha(bed: Any, ctx: Any) -> str:
    """Cold FULL fixed native material fingerprint of the DEVICE bed (voxels, active layer, availability, ledger, committed, mobile + the context identity); not water depth."""
    h = hashlib.sha256(b"maple_syrup.lifecycle_checkpoint_cuda.full_native_material/1")
    try:
        for name, part in (("voxel_column", bed.voxel_column), ("active_layer", bed.active_layer), ("sediment_availability", bed.sediment_availability),
                           ("ledger", bed.ledger), ("committed_topography", bed.committed_topography), ("mobile_mass", bed.water.mobile_mass_by_cell_class_kg)):
            h.update(name.encode() + b"\0")
            _digest_xp(h, part)
        h.update(lc._context_digest(ctx).encode())
    except lc.LayeredCheckpointError as exc:
        raise _convert(exc) from None
    return h.hexdigest()


def physical_digest_cuda(lop: Any, state: Any) -> str:
    """SHA-256 over every physical value and scalar of the current role (device arrays streamed in bounded chunks; counted D2H). For receipts/tests."""
    a = wl._arrays_of(state)
    h = hashlib.sha256(b"maple_syrup.lifecycle_checkpoint_cuda.physical/1")
    for value in (state.role, float(state.t_s), int(state.revision), int(state.episode), int(state.reset_count), int(state.request_sequence), state.lop_sha256,
                  str(state.storm_operator_sha256), state.digest):
        _digest_xp(h, value)
    _digest_xp(h, state.lifetime)
    _digest_xp(h, state.mark)
    for array in (a.soil.water_m, a.history.wetting_m, a.history.cumulative_intake_m, a.surface_m, a.qx, a.qy):
        _digest_xp(h, array)
    for part in (a.bed.voxel_column, a.bed.active_layer, a.bed.sediment_availability, a.bed.water, a.bed.ledger, a.bed.committed_topography):
        _digest_xp(h, part)
    kind = None if state.role != "interevent" else iw.source_kind(state.payload.source)
    h.update(repr((kind, None if kind is None else state.payload.source.digest)).encode())
    return h.hexdigest()


# --- metadata gather (AFTER the device gate) and the pure budget ----------------------------------------------------------------------------
def _gather(lop: Any, state: Any, owner: Any, role: str, kind: str | None, bed: Any) -> plan.CudaSaveMetadata:
    return plan.extract_save_metadata(
        bed=bed, context_fraction=lop.context.initial_available_fraction, role=role, source_kind=kind, n_cells=owner.n_cells, n_layers=owner.n_layers,
        n_classes=owner.n_classes, shape=owner.shape, wrapper_retained_host_bytes=owner.wrapper_retained_host_bytes,
        owner_audit_device_bytes=int(lop.memory.audit_scratch_bytes))


def estimate_lifecycle_budget_cuda(meta: plan.CudaSaveMetadata, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int,
                                   extra_reserve_bytes: int = 0) -> plan.CudaSaveBudget:
    """The PURE budget over sanitized metadata (see `lifecycle_checkpoint_cuda_plan`); never calls the device-gated owner accessor."""
    return plan.plan_cuda_save(meta, caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes,
                               extra_reserve_bytes=extra_reserve_bytes, profile=PROFILE)


def _admit_limits(b: plan.CudaSaveBudget, limits: Any) -> None:
    pairs = (("serialized checkpoint", b.serialized_bytes, limits.max_serialized_bytes), ("checkpoint host peak (incremental + reserves)", b.host_required_with_reserves,
             limits.max_host_bytes), ("checkpoint device peak (incremental + reserve)", b.device_required_with_reserves, limits.max_device_bytes),
             ("checkpoint transient scratch", b.scratch_bytes, limits.max_scratch_bytes))
    for what, need, lim in pairs:
        if need > lim:
            raise LifecycleCheckpointCudaQuotaError(f"{what} needs {need} B but the caller's limit is {lim} B; nothing was allocated/written")


def _admit_stage(b: plan.CudaSaveBudget, stage: str, host_fn: Callable[[], int], device_fn: Callable[[], int] | None) -> None:
    """Fresh FREE reading per stage vs the stage's incremental requirement + outside reserves (resident bytes are never charged)."""
    req = b.stage_requirements[stage]
    avail = _read(host_fn, f"host stage {stage}")
    if req["host_with_reserves"] > avail:
        raise LifecycleCheckpointCudaQuotaError(f"host stage {stage} needs {req['host_with_reserves']} B but only {avail} B are reported free; nothing further was done")
    if device_fn is not None:
        avail_d = _read(device_fn, f"device stage {stage}")
        if req["device_with_reserves"] > avail_d:
            raise LifecycleCheckpointCudaQuotaError(f"device stage {stage} needs {req['device_with_reserves']} B but only {avail_d} B are reported free; nothing further was done")


def _binding(lop: Any, owner: Any, role: str, kind: str | None, material: str, runtime: dict, n_voxels: int, _policy: Any = None) -> dict:
    pol = pol_mod.policy_of(_policy)
    top = lop.topology
    ny, nx = lop.shape
    return {"schema": pol.schema, "profile": pol.profile, "role": role, "source_kind": kind, "array_backend": "cupy", "hydraulic_backend": "cuda", "lifecycle_profile": owner.profile,
            "shape": [int(ny), int(nx)], "dx_m": float(top.dx_m).hex(), "n_layers": int(lop.n_layers), "n_cells": int(lop.n_cells), "n_faces": int(lop.memory.n_faces),
            "n_classes": int(lop.n_classes), "n_voxels": int(n_voxels), "lop_sha256": lop.sha256,
            "topology": {"identity_sha256": top.identity_sha256, "source_sha256": top.source_sha256, "topology_sha256": top.topology_sha256, "generation": int(top.generation),
                         "n_closed_components": int(top.n_closed_components)},
            "extra_reserve_bytes": int(lop.extra_reserve_bytes), "native_reserve_bytes": int(lop.native_reserve_bytes), "native_material_sha256": material,
            "context_sha256": lc._context_digest(lop.context),
            "runtime_sha256": lc._sha_text(lc._json_dumps(runtime))}


def _counters() -> Any:
    from maple.core.backend import read_transfer_counters

    return read_transfer_counters()


def _host_leaf(x: Any) -> np.ndarray:
    from maple.core.backend import to_host

    return np.ascontiguousarray(to_host(x))


# --- save ---------------------------------------------------------------------------------------------------------------------------------
def _manifest_skeleton(pol: Any, role: str, ident: dict, source: dict, runtime: dict, binding: dict, kind: str | None) -> dict:
    """The manifest dict before payload tables/hashes/seal (pure; pinned by a V1 golden test)."""
    return {"schema": pol.schema, "profile": pol.profile, "role": role, "identity": ident, "array_backend": "cupy", "hydraulic_backend": "cuda", "source": source, "runtime": runtime,
            "binding": binding, "binding_sha256": lc._sha_text(lc._json_dumps(binding)), "files": {}, "sizes": {}, "source_kind": kind, "payload": {"scalars": {}, "arrays": {}, "native": {}}}


def _provenance_record(pol: Any, role: str, ident: dict, binding_sha256: str, runtime: dict, kind: str | None) -> dict:
    """The provenance dict handed to the genuine native writer (pure; pinned by a V1 golden test)."""
    return {"syrup_lifecycle_checkpoint_schema": pol.schema, "profile": pol.profile, "role": role, "requires_companion": True, "identity": ident, "binding_sha256": binding_sha256,
            "array_backend": "cupy", "hydraulic_backend": "cuda", "runtime": runtime, "source_kind": kind}


def save_lifecycle_checkpoint_cuda(path: Any, lop: Any, state: Any, identity: dict, *, limits: Any, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int,
                                   host_available_bytes_fn: Any = None, device_available_bytes_fn: Any = None, disk_available_bytes_fn: Any = None,
                                   extra_reserve_bytes: int = 0) -> CudaSaveReceipt:
    """Write a NEW sealed checkpoint directory for the accepted `state`. Order: pure typed fences -> CURRENT-DEVICE gate (owner accessor) -> state record/bed fences -> metadata ->
    pure budget -> limits -> path/disk -> fresh whole-call host+device readings -> fresh source pair + runtime -> validations -> material fingerprint + before-digest -> reservation ->
    native writer (stage admission, D2H counters) -> companion (stage admission, D2H counters) -> after-digest == before -> manifest-last no-clobber seal. Nothing is committed,
    rerouted, advanced, reset, closed or requested; a failure before the seal removes only this call's own unpublished files; after the seal it is `Indeterminate`."""
    return _save_engine(path, lop, state, identity, limits=limits, caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes,
                        host_available_bytes_fn=host_available_bytes_fn, device_available_bytes_fn=device_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn,
                        extra_reserve_bytes=extra_reserve_bytes, _policy=pol_mod.V1)


def _save_engine(path: Any, lop: Any, state: Any, identity: dict, *, limits: Any, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, host_available_bytes_fn: Any = None,
                 device_available_bytes_fn: Any = None, disk_available_bytes_fn: Any = None, extra_reserve_bytes: int = 0, _policy: Any = None) -> CudaSaveReceipt:
    """The save body under a registered private policy (V1 for every public entry point). The policy is validated first, before any side effect."""
    pol = _select_policy(_policy)
    td = _tdict if pol is pol_mod.V1 else _tdict_full  # V2 receipts serialize ALL six transfer-counter fields; the V1 record is unchanged
    hr = _reserve(caller_host_reserve_bytes, "caller_host_reserve_bytes")
    dr = _reserve(caller_device_reserve_bytes, "caller_device_reserve_bytes")
    ex = _reserve(extra_reserve_bytes, "extra_reserve_bytes")
    host_fn = _reader(host_available_bytes_fn, "host_available_bytes_fn")
    device_fn_user = _reader(device_available_bytes_fn, "device_available_bytes_fn")
    disk_fn_user = _reader(disk_available_bytes_fn, "disk_available_bytes_fn")
    if not isinstance(limits, lc.CheckpointLimits):
        _refuse("limits must be a CheckpointLimits record")
    try:
        limits.validated()
    except lc.LayeredCheckpointError as exc:
        raise _convert(exc) from None
    role, kind = _fence_types(lop, state) if pol is pol_mod.V1 else _fence_types(lop, state, pol)  # V1 call shape unchanged
    try:
        ident = lc._canonical_identity(identity)
    except lc.LayeredCheckpointError as exc:
        raise _convert(exc) from None
    if not isinstance(path, (str, os.PathLike)):
        _refuse("path must be a str or PathLike")
    target = Path(os.path.abspath(path))  # absolute, NOT symlink-resolved: a linked final parent stays visible
    if os.path.lexists(target):
        _refuse(f"checkpoint path already exists (no overwrite): {target}")
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink():
        _refuse(f"the parent directory does not exist or is a link: {parent}")
    # ---- current-device gate: the owner accessor queries the CURRENT device through the column preflight BEFORE any reader/hash/copy
    try:
        owner = wl._cuda_checkpoint_owner(lop)
        wl._check_state(lop, state)
        record = wl._checkpoint_state_record_cuda(lop, state) if pol is pol_mod.V1 else wl._checkpoint_state_record_cuda_v2(lop, state)
    except (wl.LifecycleError, ss.SubsurfaceError) as exc:
        raise _convert(exc) from None
    if isinstance(owner.device_id, bool) or not isinstance(owner.device_id, int):
        _refuse("the lifecycle's bound device id is not an integer")
    bed = _fence_bed(lop, state, owner)
    if (role, kind) not in pol.pairs or (kind == "boundary_origin" and not owner.boundary_capable):
        _refuse(f"role/source kind {(role, kind)!r} is not covered by this lifecycle's profile")
    if pol is pol_mod.V1:
        meta = _gather(lop, state, owner, role, kind, bed)
        budget = estimate_lifecycle_budget_cuda(meta, caller_host_reserve_bytes=hr, caller_device_reserve_bytes=dr, extra_reserve_bytes=ex)
    else:  # V2: the V1 plan arithmetic wrapped with the V2 envelope (+ storm terms); the V1 plan modules are untouched
        from maple_syrup import lifecycle_checkpoint_cuda_v2 as v2

        meta = v2plan.extract_save_metadata_v2(
            bed=bed, context_fraction=lop.context.initial_available_fraction, role=role, source_kind=kind, n_cells=owner.n_cells, n_layers=owner.n_layers, n_classes=owner.n_classes,
            shape=owner.shape, wrapper_retained_host_bytes=owner.wrapper_retained_host_bytes, owner_audit_device_bytes=int(lop.memory.audit_scratch_bytes))
        budget = v2.estimate_save_budget_cuda_v2(meta, caller_host_reserve_bytes=hr, caller_device_reserve_bytes=dr, extra_reserve_bytes=ex)
    _admit_limits(budget, limits)
    host_reader = ss.default_host_available_bytes if host_fn is None else host_fn
    device_reader = lop.column.available_bytes_fn if device_fn_user is None else device_fn_user
    disk_reader = lc._disk_reader(parent, disk_fn_user)
    disk_avail = _read(disk_reader, "disk")
    if budget.disk_bytes > disk_avail:
        raise LifecycleCheckpointCudaQuotaError(f"checkpoint disk needs {budget.disk_bytes} B but only {disk_avail} B are reported free")
    _admit_stage(budget, "source_and_validation", host_reader, device_reader)  # whole-call first stage, fresh readings, BEFORE the source hash
    source = _fresh_source_pair()
    runtime = _runtime_record(lop, owner.device_id)
    # ---- the same physical validation the loader will run, BEFORE anything is written (read-only)
    try:
        wl.audit_lifecycle(lop, state)
        lc._native_checks(bed, lop.context, float(state.t_s), type(bed.ledger))
        if role == "interevent":
            it.audit_general_interevent(state.binding.gop, state.payload)
        elif role == "storm":  # V2 only (the V1 role fence refuses a storm before this point): the genuine anchored event audit incl. the completed instantaneous quiet gates
            ev.audit_event(state.binding.event, state.payload)
    except (wl.LifecycleError, it.GeneralIntereventError, ls.LayeredStormError, ev.LayeredEventError) as exc:
        raise _convert(exc) from None
    except lc.LayeredCheckpointError as exc:
        raise _convert(exc) from None
    _admit_stage(budget, "fingerprint", host_reader, None)
    c0 = _counters()
    material = _material_sha(bed, lop.context)
    digest_before = physical_digest_cuda(lop, state)
    fp_transfers = _counters().delta(c0)
    binding = _binding(lop, owner, role, kind, material, runtime, meta.n_voxels, pol)
    spec_transfers = None
    if pol is not pol_mod.V1:  # V2: closed `binding.storm` (None for boundary/interevent); the storm spec digest is admitted BEFORE the field stream
        if role == "storm":
            _admit_stage(budget, "spec_digest", host_reader, None)
            c_spec = _counters()
            binding["storm"] = v2.storm_binding_of_event(state.binding.event)
            spec_transfers = _counters().delta(c_spec)
            v2.check_spec_digest_traffic(spec_transfers, budget.stage_requirements["spec_digest"]["expected_d2h"], state.binding.event._field.multiplier)
        else:
            binding["storm"] = None
    manifest = _manifest_skeleton(pol, role, ident, source, runtime, binding, kind)
    records = {"lifecycle": record}
    if role == "interevent":
        records["t_state"] = it.checkpoint_state_record(state.payload)
        records["source"] = iw.checkpoint_source_record_v2(state.payload.source)
    if role == "storm":  # V2 only: the storm scalar record (backend-neutral CPU helper) whose DECODED `completed` must be exactly True (the helper that rebuilds the state does not refuse a flipped flag)
        records["storm"] = lk._storm_record(lop, state)
        if cc.decode_tree(records["storm"]).get("completed") is not True:
            _refuse("the storm scalar record is not that of a COMPLETED storm; refused")
    manifest["payload"]["scalars"] = records
    provenance = _provenance_record(pol, role, ident, manifest["binding_sha256"], runtime, kind)
    shim = SimpleNamespace(shape=lop.shape, n_layers=lop.n_layers, n_cells=lop.n_cells, n_classes=lop.n_classes, bed=bed)
    native_specs = lc._native_snapshot_specs(shim)
    flags_dir = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        pfd = os.open(parent, flags_dir)
    except OSError as exc:
        raise LifecycleCheckpointCudaError(f"cannot open the parent directory: {exc}") from None
    res = None
    made_id = None
    dirfd = None
    try:
        try:
            lk._mkdir(target, 0o700)
        except FileExistsError:
            raise LifecycleCheckpointCudaError(f"checkpoint path already exists (no overwrite): {target}") from None
        except OSError as exc:
            raise LifecycleCheckpointCudaError(f"cannot reserve the checkpoint directory: {exc}") from None
        made_id = lk._lstat_identity(target)
        try:
            dirfd = os.open(target, flags_dir)
        except OSError as exc:
            raise LifecycleCheckpointCudaError(f"cannot open the reserved directory: {exc}") from None
        res = lc._Reservation(target, dirfd, lc._identity_of(dirfd))
    except BaseException:
        if res is None:
            if dirfd is not None:
                os.close(dirfd)
            lk._rmdir_owned(target, made_id)
        os.close(pfd)
        raise
    sealed = False
    tmp_name = f".manifest.tmp-{os.getpid()}-{os.urandom(6).hex()}"
    transfers: dict = {"fingerprint_and_digest": td(fp_transfers)}
    if spec_transfers is not None:
        transfers["spec_digest"] = td(spec_transfers)
    try:
        lk._fsync(pfd)
        if os.listdir(res.dirfd):
            _refuse("the reserved checkpoint directory is not empty (a foreign writer raced the reservation)")
        dirfd = res.dirfd
        from maple.io.outputs import snapshot as native_snapshot

        # ---- native payload FIRST (approved ordering; one atomic seal): the genuine writer stages every host leaf at once
        _admit_stage(budget, "native_write", host_reader, None)
        res.stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=parent))
        res.stage_id = lk._lstat_identity(res.stage)
        res.stage_fd = os.open(res.stage, flags_dir)
        if lc._identity_of(res.stage_fd) != res.stage_id:
            _refuse("the private native stage directory changed identity")
        sfd = os.open(PAYLOAD_NATIVE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=res.stage_fd)
        try:
            staged_id = lc._identity_of(sfd)
        finally:
            os.close(sfd)
        res.stage_created[PAYLOAD_NATIVE] = staged_id
        c1 = _counters()
        native_snapshot.save_state_snapshot(
            bed.voxel_column, bed.active_layer, float(state.t_s), res.stage / PAYLOAD_NATIVE, step=int(state.revision), reason=lc._SNAPSHOT_REASON,
            geometry=lop.context.geometry, mass_resolution_kg=lop.context.mass_resolution_kg, sediment_availability=bed.sediment_availability, water=bed.water,
            provenance=provenance)
        d1 = _counters().delta(c1)
        transfers["native_writer"] = td(d1)
        exp = budget.expected_d2h["native_writer"]
        if d1.device_to_host != exp["transfers"] or d1.device_to_host_bytes != exp["bytes"]:
            _refuse(f"native writer D2H counters ({d1.device_to_host} transfers, {d1.device_to_host_bytes} B) differ from the planned ({exp['transfers']}, {exp['bytes']} B); "
                    "a concurrent transfer occurred or the leaf inventory is wrong; nothing sealed")
        if lc._identity_of(PAYLOAD_NATIVE, dir_fd=res.stage_fd) != staged_id:
            _refuse("the staged native payload changed identity during the native write")
        lc._fsync_name(PAYLOAD_NATIVE, res.stage_fd)
        res.check_bundle_identity()
        try:
            lk._link(PAYLOAD_NATIVE, PAYLOAD_NATIVE, src_dir_fd=res.stage_fd, dst_dir_fd=dirfd, follow_symlinks=False)
        except FileExistsError:
            raise LifecycleCheckpointCudaError("the bundle already holds a file at the native payload name (a foreign writer raced); nothing was overwritten") from None
        res.created[PAYLOAD_NATIVE] = staged_id
        os.unlink(PAYLOAD_NATIVE, dir_fd=res.stage_fd)
        res.stage_created.pop(PAYLOAD_NATIVE, None)
        os.rmdir(res.stage)
        # ---- companion
        _admit_stage(budget, "companion", host_reader, None)
        c2 = _counters()
        host: dict = {}
        table: dict = {}

        def add(name: str, array: np.ndarray) -> str:
            if not pc._dtype_ok(array.dtype.str) or array.ndim > 5:
                _refuse(f"{name}: dtype {array.dtype.str} / ndim {array.ndim} not allowed")
            host[name] = array
            table[name] = {"dtype": array.dtype.str, "shape": [int(n) for n in array.shape]}
            return name

        a = wl._arrays_of(state)
        add("soil_water_m", _host_leaf(a.soil.water_m))
        add("wetting_m", _host_leaf(a.history.wetting_m))
        add("cumulative_intake_m", _host_leaf(a.history.cumulative_intake_m))
        add("qx_m2_s", _host_leaf(a.qx))
        add("qy_m2_s", _host_leaf(a.qy))
        add("topo_source_full", np.array(np.asarray(lop.topology.source_z_full), copy=True, order="C"))
        native = {"ledger": pc._native_encode(bed.ledger, lambda arr: add(f"native_{len(host):04d}", arr)),
                  "committed_topography": pc._native_encode(bed.committed_topography, lambda arr: add(f"native_{len(host):04d}", arr))}
        d2 = _counters().delta(c2)
        transfers["companion_capture"] = td(d2)
        expc = budget.expected_d2h["companion"]
        if d2.device_to_host != expc["transfers"] or d2.device_to_host_bytes != expc["bytes"]:
            _refuse(f"companion D2H counters ({d2.device_to_host} transfers, {d2.device_to_host_bytes} B) differ from the planned ({expc['transfers']}, {expc['bytes']} B); nothing sealed")
        manifest["payload"]["arrays"] = table
        manifest["payload"]["native"] = native
        try:
            lc._json_dumps(manifest)
        except (TypeError, ValueError) as exc:
            raise LifecycleCheckpointCudaError(f"manifest is not strict JSON: {exc}") from None
        specs = {PAYLOAD_NATIVE: native_specs, PAYLOAD_COMPANION: lc._companion_expected(shim, table)}
        res.check_bundle_identity()
        lk._write_companion(dirfd, res, host)
        del host
        sizes, hashes = {}, {}
        for name in pol.files:
            fd = lc._open_regular(name, dirfd, budget.serialized_bytes)
            try:
                if lc._identity_of(fd) != res.created[name]:
                    _refuse(f"{name} changed identity after it was written")
                hashes[name], sizes[name] = lc._fd_hash(fd, budget.serialized_bytes)
                with os.fdopen(os.dup(fd), "rb") as handle:
                    lc._audit_archive(handle, specs[name], label=name, byte_limit=limits.max_serialized_bytes)
            finally:
                os.close(fd)
        if sum(sizes.values()) > min(budget.serialized_bytes, limits.max_serialized_bytes):
            raise LifecycleCheckpointCudaQuotaError(f"written payload {sum(sizes.values())} B exceeds the serialized bound")
        # ---- retained-state audit: the accepted state is byte-identical after the save (single-writer contract)
        digest_after = physical_digest_cuda(lop, state)
        if digest_after != digest_before:
            _refuse("the accepted state changed during the save (single-writer contract violated); nothing sealed")
        sealed_manifest = pc._seal_manifest({**manifest, "files": hashes, "sizes": sizes})
        text = json.dumps(sealed_manifest, indent=1, sort_keys=True, allow_nan=False) + "\n"
        if len(text) > budget.manifest_cap_bytes or not text.isascii():
            _refuse("manifest exceeds its derived byte bound")
        lc._scan_json(text.encode("ascii"), lc.MANIFEST_MAX_DEPTH, budget.manifest_max_nodes, lc.MANIFEST_MAX_STRING)
        res.check_bundle_identity()
        tfd = os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=dirfd)
        res.created[tmp_name] = lc._identity_of(tfd)
        with os.fdopen(tfd, "wb") as handle:
            handle.write(text.encode("ascii"))
            handle.flush()
            lk._fsync(handle.fileno())
        lk._fsync(dirfd)
        res.check_bundle_identity()
        try:
            lk._link(tmp_name, pol.manifest_name, src_dir_fd=dirfd, dst_dir_fd=dirfd, follow_symlinks=False)
        except FileExistsError:
            raise LifecycleCheckpointCudaError("the seal name already exists (a foreign writer raced); nothing was overwritten") from None
        sealed = True
        os.unlink(tmp_name, dir_fd=dirfd)
        res.created.pop(tmp_name, None)
        lk._fsync(dirfd)
        lk._fsync(pfd)
    except BaseException as exc:
        if sealed:
            raise LifecycleCheckpointCudaIndeterminate(
                f"failure after the seal link ({type(exc).__name__}: {exc}); the bundle at {target} may be durable: nothing was deleted and it must not be retried over") from exc
        res.remove_own()
        if res.stage is not None and res.stage_fd is None:
            lk._rmdir_owned(res.stage, res.stage_id)
        if isinstance(exc, LifecycleCheckpointCudaError):
            raise
        if isinstance(exc, Exception):
            raise _convert(exc) from exc
        raise
    finally:
        res.close()
        os.close(pfd)
    return CudaSaveReceipt(str(target), pol.profile, role, kind, sealed_manifest["manifest_sha256"], dict(hashes), dict(sizes), float(state.t_s), int(state.revision), state.digest,
                           digest_before, material, budget.as_dict(), dict(source), dict(runtime), transfers)


def _tdict(d: Any) -> dict:
    return {"device_to_host": int(d.device_to_host), "device_to_host_bytes": int(d.device_to_host_bytes), "host_to_device": int(d.host_to_device),
            "scalar_reads": int(d.scalar_reads), "synchronizations": int(d.synchronizations)}


def _tdict_full(d: Any) -> dict:
    """V2 receipt record of a `TransferCounters` delta: every named field (the V1 save record is D2H-oriented and omits `host_to_device_bytes`)."""
    return {name: int(getattr(d, name)) for name in ("host_to_device", "host_to_device_bytes", "device_to_host", "device_to_host_bytes", "scalar_reads", "synchronizations")}
