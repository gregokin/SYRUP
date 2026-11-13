"""G-F1+2 / G-F3: genuine standalone CUDA FILE checkpoint of a continuing FLOATING Pond state, Dense (`/7`) and native Windowed (`/8`). AUTHOR-WRITTEN, NOT RUN ON A DEVICE (Root qualifies it on hardware). Importing this module never imports CuPy.

Closed profiles, never the CPU child: schema `maple-syrup-floating-pond-checkpoint/7` (Dense device bed) and `/8` (native Windowed device bed: published resident window + EXTERNAL deep-store references), profile
`floating-vertical/2-cuda`, `array_backend == "cupy"`, `hydraulic_backend == "cuda"`. The same public save/load functions dispatch by the event's configured bed storage and the file's schema; the storage found must equal the event's
(no cross-storage conversion). The CPU schemas `/1-/6` (`floating_checkpoint`) and the historical `pond_checkpoint` and their source guards are untouched: the CPU loader refuses `/7` and `/8` by its schema check and this loader
refuses `/1-/6`. Dense `/7` keeps its exact manifest keys, bytes and physics; `/8` adds only `bed_storage` and `deep_storage` (the verified deep-store record of `pond_checkpoint._check_deep_record`). No auto-upgrade, no
cross-backend reading, no rebinding of any old lock; a file written under another source pair is refused.

Files (same two payloads + sealed manifest as the CPU child): `maple_state.npz` is the native MAPLE `save_state_snapshot` (Dense) / `save_windowed_state_snapshot` (Windowed) of the device bed (MAPLE's own writer; it converts the
device tree itself; Windowed REFERENCES the deep store, never copies it, never rewrites its path, never mutates it) and `pond_continuation.npz` holds every Pond array (reused `pond_checkpoint._extract/_specs`), the CURRENT floating
inventory/history `W (L, N)`, `J`, `I`, the CURRENT canonical depth `D` and the HELD law facts as sealed HOST arrays (held FC / SAT in the column's own RAW layout shape (1,), (L,) or (L, N); `D0`; the layer-0 conductivity record).
The manifest adds the stable root column / matrix / preferential content identities, the owner config identity, generation, the native source witness, the `held_layouts`, the kernel/ABI descriptor of
`floating_held_context_cuda.kernel_descriptor()` (compared with the live one BEFORE any payload is read), the producer-maintained terminal native-call count and the retained quiet-reset receipt. The state is validated against the
CURRENT depth `D`, never against `D0`; nothing is derived from the loaded (eroded/deposited) bed; no owner, no initial W, no property provider.

Load order (the CPU contract on device): event metadata -> the CURRENT device gate FIRST (before any reader, IO or owned array) -> stage-0 two-pool admission (manifest read) -> bounded manifest, schema/keys/own sha, identity,
SOURCE PAIR and kernel descriptor, backends, binding, floating facts, held IDs (strings), Windowed: the strict deep-store record from the manifest alone -> payload hashes, archive/header audits (reused `pond_checkpoint`
helpers; Windowed: the grid-derived native member layout) -> stage-1 two-pool admission of the FULL composed model from the VERIFIED header dimensions -> MAPLE `load_state_snapshot` / `load_windowed_state_snapshot` (always a HOST
load; Windowed: the exact deep-store version check, the loaded handle's mapping bound vs the modelled one, then the UNTRUSTED native packet validation) -> whitelist companion read -> streamed held-value comparison against the REAL
held template -> `pc._build_state` (MAPLE counted `to_device_tree`, and for Windowed MAPLE's `move_prefix_packets`) -> mint `VerifiedFloatingInputsCuda` -> `floating_resume_cuda.prepare_from_verified_inputs` (public D3/E5 factories)
-> `vertical.resume_scope(prepared)`: device canonical-depth check of EVERY active column (Windowed: the native prefix decoder with `trusted=True`, justified ONLY by the cold untrusted validation above), the floating
validator, scalars, `event._check_state` and the audits -> `vertical.adopt_resumed(prepared)` LAST. A failure anywhere before the adoption leaves the vertical's registry, stats and payloads untouched.

Memory: TWO pools (device `available_bytes_fn` / `reserve_bytes` / `memory_budget_bytes`; host `host_*`), real default readers when None, reserves (and the 64 KiB Python allowance) counted once per admission by `fcu.admit`.
The model is arithmetic over declared header dimensions (`load_models`, `save_models`); Windowed models use the RESIDENT member bytes and the deep-store MAPPING-CACHE bound (`floating_checkpoint.deep_cache_bound_bytes`; real
caches are MiB-scale while the resident archive is KiB-scale), never a compressed/resident proxy of the logical cells x voxels x classes bed. It is a DECLARED conservative allowance (hardware peak unverified).

Transfers (counted through MAPLE's `to_host`/`to_device`): companion floating arrays move in `HASH_CHUNK_BYTES` chunks; the Pond arrays and the native snapshot/ledger use MAPLE's own whole-array helpers (`to_device_tree`,
`move_prefix_packets`, `to_device`, `to_host`, the snapshot writer's conversion): NOT chunked, reported honestly; D2H/H2D of the held template compare are chunked; the depth reductions use one 8-byte count per check.

Limits: same device class and CuPy build (the derived soil sum is compared bitwise against the stored one); no outer container, no daily cursor; the MAPLE snapshot save/load of a CuPy Dense/Windowed tree and every device number
need Root's hardware qualification."""
# ruff: noqa: BLE001  (the loader converts any native failure into one typed refusal, as the CPU child does)
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
import threading
import weakref
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_pond as fp
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import pond_checkpoint as pc
from maple_syrup import subsurface as ss
from maple_syrup.checkpoint import DEFAULT_MAX_CHECKPOINT_BYTES
from maple_syrup.pond_checkpoint import (
    _MAX_NDIM,
    PondCheckpointError,
    _audit_array_table,
    _dtype_ok,
    _hash_file,
    _hex,
    _json_dumps,
    _native_decode,
    _native_encode,
    _npy_preflight,
    _sha256_text,
    _specs,
    _zip_audit,
)
from maple_syrup.pond_event import PondEvent, PondEventState
from maple_syrup.sediment_bed import BedState, is_windowed_bed

__all__ = ["GPU_PROFILE", "GPU_SCHEMA", "GPU_WINDOWED_SCHEMA", "READ_SCHEMAS", "WINDOWED_MANIFEST_KEYS", "VerifiedFloatingInputsCuda", "load_floating_pond_checkpoint_cuda", "load_models", "save_floating_pond_checkpoint_cuda", "save_models"]

GPU_SCHEMA = "maple-syrup-floating-pond-checkpoint/7"  # Dense device bed; the only schema read or written here
GPU_WINDOWED_SCHEMA = "maple-syrup-floating-pond-checkpoint/8"  # native Windowed device bed (published resident window + EXTERNAL deep-store references)
GPU_PROFILE = "floating-vertical/2-cuda"
MANIFEST_NAME = fc.MANIFEST_NAME
PAYLOADS = pc.PAYLOADS
MANIFEST_KEYS = fc.MANIFEST_KEYS_V3 | {"kernel"}
WINDOWED_MANIFEST_KEYS = MANIFEST_KEYS | {"bed_storage", "deep_storage"}  # /8: the Dense /7 closed set + the bed storage and the verified deep-store record
READ_SCHEMAS = (GPU_SCHEMA, GPU_WINDOWED_SCHEMA)
CONT_GPU_SCHEMA = "maple-syrup-floating-continued-cuda-pond-checkpoint/1"  # CC2: CUDA CONTINUED-ORIGIN profile (Dense device bed); written/read ONLY with `original_topology=` (see floating_continued_checkpoint_cuda)
CONT_GPU_WINDOWED_SCHEMA = "maple-syrup-floating-continued-cuda-pond-checkpoint/2"  # native Windowed device bed
CONT_READ_SCHEMAS = (CONT_GPU_SCHEMA, CONT_GPU_WINDOWED_SCHEMA)
CONT_GPU_PROFILE = "floating-vertical-continued-origin/2-cuda"
CONT_MANIFEST_KEYS = MANIFEST_KEYS | {"continued_origin"}
CONT_WINDOWED_MANIFEST_KEYS = WINDOWED_MANIFEST_KEYS | {"continued_origin"}
FLOATING_KEYS = fc.FLOATING_KEYS | {"held_layouts"}
LAYOUT_NAMES = ("theta_field_capacity", "theta_saturation")
HASH_BUFFER_BYTES = fc.HASH_BUFFER_BYTES
HEADER_AUDIT_BYTES = fc.HEADER_AUDIT_BYTES
CHUNK_BYTES = fcu.HASH_CHUNK_BYTES


class FloatingCheckpointCudaError(fc.FloatingCheckpointError):
    """Typed narrow refusal of the CUDA floating file checkpoint."""


_MINT = threading.local()
_INPUTS: weakref.WeakSet = weakref.WeakSet()
_DIGESTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_CONSUMED: weakref.WeakSet = weakref.WeakSet()


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


def _fail(message: str) -> None:
    raise FloatingCheckpointCudaError(message)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# verified inputs (sealed HOST parts; the device payload is built from them later by `floating_resume_cuda`)
# ---------------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True, eq=False)
class VerifiedFloatingInputsCuda:
    """Capability produced ONLY by actually reading and verifying a CUDA floating checkpoint FILE (manifest, source pair, kernel descriptor, identity, binding, hashes, archive/header audits, native snapshot provenance, held IDs and
    held VALUES against the real held template). It claims NO physical acceptance; its array parts are sealed read-only HOST arrays (the device payload is built later). One-shot; not copyable."""

    schema: str
    storage: str
    event_config_sha256: str
    forcing_sha256: str
    vertical_identity: str
    backend: str
    soil_t_s: float
    root_column_sha256: str
    root_matrix_op_sha256: str
    root_pref_sha256: str | None
    owner_config_sha256: str
    generation: int
    source_witness: str
    source_pair: Any
    kernel: Any
    held_layouts: Any
    held: Any
    current: Any
    parent: Any  # raw candidate PondEventState on the DEVICE (soil water NOT yet registered)
    scalars_json: str

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            _fail("VerifiedFloatingInputsCuda is minted only by load_floating_pond_checkpoint_cuda after reading and verifying a file")
        _INPUTS.add(self)

    def __copy__(self):
        _fail("VerifiedFloatingInputsCuda is an owned capability and cannot be copied")

    def __deepcopy__(self, memo):
        _fail("VerifiedFloatingInputsCuda is an owned capability and cannot be copied")

    def __reduce_ex__(self, protocol):
        _fail("VerifiedFloatingInputsCuda is not serialisable")


_INPUT_HELD = ("theta_field_capacity", "theta_saturation", "depth0_m")
_INPUT_CURRENT = ("water_m", "wetting_m", "cumulative_intake_m", "depth_m")
_INPUT_SCALARS = (("schema", str), ("storage", str), ("event_config_sha256", str), ("forcing_sha256", str), ("vertical_identity", str), ("backend", str), ("soil_t_s", float), ("root_column_sha256", str),
                  ("root_matrix_op_sha256", str), ("owner_config_sha256", str), ("generation", int), ("source_witness", str), ("scalars_json", str))


def held_shape(layout: str, L: int, N: int) -> tuple:
    """Shape of a held theta map in the column's own RAW layout."""
    if layout == "scalar":
        return (1,)
    if layout == "shared":
        return (L,)
    if layout == "cell":
        return (L, N)
    raise FloatingCheckpointCudaError(f"unknown held layout {layout!r}")


def _layouts_of(column: Any) -> dict:
    return {name: str(column.layouts[name]) for name in LAYOUT_NAMES}


def _check_input_layout(inputs: Any) -> None:
    """EXACT metadata of every verified-input field BEFORE any byte is hashed or copied (scalar types, containers, key sets, exact C-contiguous read-only float64 ndarrays of the declared shapes)."""
    for name, kind in _INPUT_SCALARS:
        if type(getattr(inputs, name)) is not kind:
            _fail(f"verified-input field {name} is malformed (integrity)")
    if inputs.root_pref_sha256 is not None and type(inputs.root_pref_sha256) is not str:
        _fail("verified-input field root_pref_sha256 is malformed (integrity)")
    for field in ("source_pair", "kernel", "held_layouts", "held", "current"):
        if not isinstance(getattr(inputs, field), MappingProxyType):
            _fail(f"the verified inputs' {field} container was replaced (integrity)")
    if any(type(k) is not str or type(v) not in (str, int) for k, v in inputs.source_pair.items()) or any(type(k) is not str or type(v) is not str for k, v in inputs.kernel.items()):
        _fail("the verified inputs' source pair / kernel descriptor is malformed (integrity)")
    if set(inputs.held_layouts) != set(LAYOUT_NAMES) or any(v not in ("scalar", "shared", "cell") for v in inputs.held_layouts.values()):
        _fail("the verified inputs' held layouts are malformed (integrity)")
    if set(inputs.held) not in (set(_INPUT_HELD), set(_INPUT_HELD) | {"ks_m_s"}) or set(inputs.current) != set(_INPUT_CURRENT):
        _fail("the verified inputs' array sets are malformed (integrity)")
    depth = inputs.current["depth_m"]
    water = inputs.current["water_m"]
    N = int(depth.shape[0]) if type(depth) is np.ndarray and depth.ndim == 1 else -1
    L = int(water.shape[0]) if type(water) is np.ndarray and water.ndim == 2 else -1
    expected = {("held", "theta_field_capacity"): held_shape(inputs.held_layouts["theta_field_capacity"], L, N), ("held", "theta_saturation"): held_shape(inputs.held_layouts["theta_saturation"], L, N),
                ("held", "depth0_m"): (N,), ("held", "ks_m_s"): (N,), ("current", "water_m"): (L, N), ("current", "wetting_m"): (N,), ("current", "cumulative_intake_m"): (N,), ("current", "depth_m"): (N,)}
    for group in ("held", "current"):
        for name, a in getattr(inputs, group).items():
            if type(a) is not np.ndarray or a.dtype != np.float64 or a.shape != expected[(group, name)] or not a.flags.c_contiguous or a.flags.writeable:
                _fail(f"verified-input array {group}.{name} is not an exact read-only C-contiguous float64 ndarray of the declared shape (integrity)")


def input_digest(inputs: Any) -> str:
    """Integrity digest of EVERY verified-input field (scalars, identities, source pair, kernel descriptor, layouts, every array's dtype/shape/C bytes by direct `memoryview`, the identity of the referenced parent state)."""
    h = hashlib.sha256()
    meta = {name: getattr(inputs, name) for name, _k in _INPUT_SCALARS if name != "soil_t_s"}
    meta["soil_t_s"] = float(inputs.soil_t_s).hex()
    meta["root_pref_sha256"] = inputs.root_pref_sha256
    meta["source_pair"] = {k: inputs.source_pair[k] for k in sorted(inputs.source_pair)}
    meta["kernel"] = {k: inputs.kernel[k] for k in sorted(inputs.kernel)}
    meta["held_layouts"] = {k: inputs.held_layouts[k] for k in sorted(inputs.held_layouts)}
    meta["parent_id"] = id(inputs.parent)
    meta["held_keys"], meta["current_keys"] = sorted(inputs.held), sorted(inputs.current)
    h.update(json.dumps(meta, sort_keys=True, default=repr).encode())
    for group, names in (("held", sorted(inputs.held)), ("current", list(_INPUT_CURRENT))):
        for name in names:
            a = getattr(inputs, group)[name]
            h.update(f"{group}.{name}|{a.dtype}|{a.shape}".encode())
            h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def require_inputs(inputs: Any) -> None:
    """Genuinely minted by this loader, unconsumed (private one-shot state) AND intact: exact metadata first, then the recomputed digest must equal the one bound at minting."""
    if type(inputs) is not VerifiedFloatingInputsCuda or inputs not in _INPUTS or inputs not in _DIGESTS:
        _fail("not genuinely minted VerifiedFloatingInputsCuda (a plain record, copy or caller dict carries no authority)")
    if inputs in _CONSUMED:
        _fail("the verified inputs were already consumed")
    _check_input_layout(inputs)
    try:
        digest = input_digest(inputs)
    except (TypeError, ValueError, BufferError) as exc:
        _fail(f"the verified inputs could not be digested ({type(exc).__name__}); integrity refused")
    if digest != _DIGESTS[inputs]:
        _fail("the verified inputs no longer match the integrity digest bound at verification time (altered or rebound)")


def verify_intact(inputs: Any) -> None:
    """Integrity of an ALREADY-CONSUMED capability (used by the prepared record): the same metadata/digest checks without the one-shot test."""
    if type(inputs) is not VerifiedFloatingInputsCuda or inputs not in _INPUTS or inputs not in _DIGESTS:
        _fail("not genuinely minted VerifiedFloatingInputsCuda")
    _check_input_layout(inputs)
    if input_digest(inputs) != _DIGESTS[inputs]:
        _fail("the verified inputs no longer match the integrity digest bound at verification time (altered or rebound)")


def consume_inputs(inputs: Any) -> None:
    """The ONE-SHOT consumption, called by the preparation AFTER every refusable preparation step succeeded."""
    _CONSUMED.add(inputs)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# event / device / quota / models
# ---------------------------------------------------------------------------------------------------------------------------------------------
def check_event(event: Any) -> Any:
    """Metadata-only typed checks (no device, reader or IO): a genuine PondEvent with a CUDA FloatingPondVertical on a CuPy bed whose configured storage is Dense (`/7`) or native Windowed (`/8`). A CPU event and a bucket
    event are refused; the storage actually found in a state or a file must equal the event's (no cross-storage conversion)."""
    if type(event) is not PondEvent or event.vertical is None or type(event.vertical) is not fp.FloatingPondVertical:
        _fail("event must be a PondEvent with a FloatingPondVertical")
    v = event.vertical
    if v.backend != "cuda" or getattr(event.xp, "__name__", None) != "cupy" or event.control.backend != "cuda":
        _fail(f"the CUDA floating file needs a CUDA vertical on a CuPy event (vertical {v.backend!r}, array namespace {getattr(event.xp, '__name__', None)!r}, hydraulic {event.control.backend!r}); "
              "CPU events use floating_checkpoint")
    if getattr(event.control, "bed_storage", "dense") not in ("dense", "windowed"):
        _fail("the CUDA floating file serves the Dense (/7) and native Windowed (/8) bed storages only")
    return v


def storage_of(event: Any) -> str:
    return str(getattr(event.control, "bed_storage", "dense"))


def schema_of(storage: str, continued: bool = False) -> str:
    if continued:
        return CONT_GPU_WINDOWED_SCHEMA if storage == "windowed" else CONT_GPU_SCHEMA
    return GPU_WINDOWED_SCHEMA if storage == "windowed" else GPU_SCHEMA


def quota_of(available_bytes_fn: Any, reserve_bytes: Any, host_available_bytes_fn: Any, host_reserve_bytes: Any, memory_budget_bytes: Any, host_memory_budget_bytes: Any) -> dict:
    for name, fn in (("available_bytes_fn", available_bytes_fn), ("host_available_bytes_fn", host_available_bytes_fn)):
        if fn is not None and not callable(fn):
            _fail(f"{name} must be callable or None (the real default reader)")
    try:
        return {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": fsl._count(reserve_bytes, "reserve_bytes"), "available_bytes_fn": available_bytes_fn, "host_memory_budget_bytes": host_memory_budget_bytes,
                "host_reserve_bytes": fsl._count(host_reserve_bytes, "host_reserve_bytes"), "host_available_bytes_fn": host_available_bytes_fn}
    except fsl.FloatingSoilError as exc:
        _fail(str(exc))
    return {}


def component_quota(q: dict) -> dict:
    """The SAME readers/budgets with ZERO reserves: the composed admission already counted each pool's reserve once."""
    return {**q, "reserve_bytes": 0, "host_reserve_bytes": 0}


def gate(column: Any) -> tuple[Any, int]:
    """The CURRENT device must be the vertical's device BEFORE any reader, IO or owned array."""
    try:
        cp = fcu._rc().require_cuda()
        return cp, fcu.gate_device(cp, column.device_id)
    except (ss.SubsurfaceError, RuntimeError) as exc:
        _fail(f"the CUDA device gate refused: {exc}")
    raise AssertionError("unreachable")


def admit(cp: Any, dev: int, device_bytes: int, host_bytes: int, q: dict, what: str) -> None:
    """Two pools, reserves once (`fcu.admit`); the typed `FloatingMemoryError` of an unfit pool propagates unchanged."""
    fcu.admit(cp, dev, int(device_bytes), int(host_bytes), q, what)


def _spec_bytes(specs: dict, *, device_only: bool = False) -> int:
    total = 0
    for shape, dtype, _sign, loc in specs.values():
        if None in shape or (device_only and loc == "host"):
            continue
        total += int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    return total


def working_bytes(N: int, nc: int, nb: int) -> int:
    """Declared device/host working set of the rebuilt topology / phase / audit validation (the CPU model's arithmetic from N, nc, phase_bins)."""
    return (fc.TOPOLOGY_ARRAYS_PER_CELL * N + 4 * N * nc * nb + 8 * N * nc) * 8


def canonical_depth_bytes(N: int, V: int, K: int, batch_cells: int, n_active: int, *, windowed: bool = False) -> int:
    """Device working set of the canonical-depth check: the active index list, the gathered batch rows, the result and the gather of the stored depth. Windowed (`V` = 0): the bounded active rows plus the whole-grid
    prefix-total decode and its sliced copy (`2 N 8`, as `floating_pond_cuda.after_exchange_cuda` declares); no deep read."""
    return n_active * 8 * 3 + batch_cells * (V * K + K + 4) * 8 + N * 8 + (2 * N * 8 if windowed else 0)


def prepare_models(L: int, N: int) -> tuple[int, int]:
    """`(device, host)` bytes of the current-payload preparation (`floating_resume_cuda.prepare_models`)."""
    from maple_syrup import floating_resume_cuda as frc

    return frc.prepare_models(L, N)


def load_models(L: int, N: int, nc: int, nb: int, *, table_bytes: int, dev_table_bytes: int, native_bytes: int, V: int, K: int, n_active: int, batch_cells: int, windowed: bool = False,
                cache_bytes: int = 0) -> tuple[int, int]:
    """`(device_bytes, host_bytes)` of one load from header dimensions (reserves excluded; `fcu.admit` adds each once). HOST: the companion table read + the sealed copies + the validator views (3x, the CPU factor), the native
    snapshot loaded and its rebuilt copies (2x the verified member bytes), the rebuilt topology/phase/audit working set, hash/header buffers, one transfer chunk and the preparation host side. DEVICE: the placed bed tree
    (`native_bytes`), the placed device-side Pond arrays, the validation working set, the uploads/retained payload of the preparation and the canonical-depth check, one chunk. Windowed (`/8`): `native_bytes` are the RESIDENT
    member bytes (window, base, prefix packets, active, availability, water: NOT cells x voxels x classes) and `cache_bytes` is the host deep-store MAPPING-CACHE bound (`floating_checkpoint.deep_cache_bound_bytes`, plus the
    mapped bound of the opened handle where known), which is independent of the (tiny) resident archive."""
    p_dev, p_host = prepare_models(L, N)
    work = working_bytes(N, nc, nb)
    host = 3 * int(table_bytes) + 2 * int(native_bytes) + work + 8 * N + HASH_BUFFER_BYTES + HEADER_AUDIT_BYTES + CHUNK_BYTES + p_host + n_active * 8 + int(cache_bytes)
    device = int(native_bytes) + int(dev_table_bytes) + work + p_dev + canonical_depth_bytes(N, V, K, batch_cells, n_active, windowed=windowed) + CHUNK_BYTES
    return int(device), int(host)


def save_models(L: int, N: int, nc: int, nb: int, *, table_bytes: int, native_bytes: int, V: int, K: int, n_active: int, batch_cells: int, windowed: bool = False,
                cache_bytes: int = 0) -> tuple[int, int]:
    """`(device_bytes, host_bytes)` of one save from header dimensions (reserves excluded). HOST: the sealed host copies of every table array (3x: chunk staging, the copy, the zip write), the native snapshot writer's own
    conversion of the device tree (2x the native bytes), working sets and one chunk. DEVICE: the validation working set and the canonical-depth check (no array is uploaded)."""
    work = working_bytes(N, nc, nb)
    host = 3 * int(table_bytes) + 2 * int(native_bytes) + work + 8 * N + HASH_BUFFER_BYTES + HEADER_AUDIT_BYTES + CHUNK_BYTES + n_active * 8 + int(cache_bytes)
    device = work + canonical_depth_bytes(N, V, K, batch_cells, n_active, windowed=windowed) + CHUNK_BYTES
    return int(device), int(host)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# chunked cold counted transfers and the streamed comparison
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _per(itemsize: int) -> int:
    return max(1, CHUNK_BYTES // int(itemsize))


def d2h(dev_arr: Any) -> np.ndarray:
    """COLD counted device-to-host copy of a C-contiguous device array into a fresh owned host array in `HASH_CHUNK_BYTES` chunks."""
    from maple.core.backend import to_host

    if not dev_arr.flags.c_contiguous:
        _fail("a device array to be saved must be C-contiguous (no hidden whole-array copy)")
    out = np.empty(tuple(int(x) for x in dev_arr.shape), dtype=dev_arr.dtype)
    flat, dst = dev_arr.reshape(-1), out.reshape(-1)
    per = _per(out.dtype.itemsize)
    for i in range(0, int(flat.size), per):
        dst[i:i + per] = np.asarray(to_host(flat[i:i + per]))
    return out


def h2d(cp: Any, host: np.ndarray) -> Any:
    """COLD counted host-to-device copy into a preallocated device array in `HASH_CHUNK_BYTES` chunks."""
    from maple.core.backend import to_device

    out = cp.empty(tuple(int(x) for x in host.shape), dtype=host.dtype)
    flat, src = out.reshape(-1), host.reshape(-1)
    per = _per(host.dtype.itemsize)
    for i in range(0, int(src.size), per):
        flat[i:i + per] = to_device(np.ascontiguousarray(src[i:i + per]), cp)
    return out


def stream_equal(dev_arr: Any, host: np.ndarray) -> bool:
    """Bit-for-bit equality of a device array and a host array through bounded counted D2H chunks (early exit); shape/dtype/contiguity mismatches are False."""
    from maple.core.backend import to_host

    if tuple(dev_arr.shape) != tuple(host.shape) or dev_arr.dtype != host.dtype or not dev_arr.flags.c_contiguous:
        return False
    flat, src = dev_arr.reshape(-1), np.ascontiguousarray(host).reshape(-1)
    per = _per(src.dtype.itemsize)
    for i in range(0, int(flat.size), per):
        if np.asarray(to_host(flat[i:i + per])).tobytes() != src[i:i + per].tobytes():
            return False
    return True


# ---------------------------------------------------------------------------------------------------------------------------------------------
# manifest / specs
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _gpu_floating_specs(L: int, N: int, layouts: dict, has_ks: bool) -> dict:
    s = {"fl_water": ((L, N), "<f8", "nn", "host"), "fl_wetting": ((N,), "<f8", "nn", "host"), "fl_cum_intake": ((N,), "<f8", "nn", "host"), "fl_depth": ((N,), "<f8", "nn", "host"),
         "fl_depth0": ((N,), "<f8", "nn", "host"), "fl_held_fc": (held_shape(layouts["theta_field_capacity"], L, N), "<f8", "nn", "host"),
         "fl_held_sat": (held_shape(layouts["theta_saturation"], L, N), "<f8", "nn", "host")}
    if has_ks:
        s["fl_ks"] = ((N,), "<f8", "nn", "host")
    return s


def _binding(event: PondEvent, windowed: bool = False, original_topology: Any = None) -> dict:
    """The CPU child's binding of the event (Dense: `fc._binding`, unchanged bytes) with the GPU profile; Windowed: the same fields on the historical WINDOWED event binding (`pc._binding(..., WINDOWED_SCHEMA)`: bed storage, deep
    publication control and deep-store options)."""
    if not windowed:
        b = dict(fc._binding(event, GPU_SCHEMA))
    else:
        b = dict(pc._binding(event, pc.WINDOWED_SCHEMA))
        b["schema"] = GPU_WINDOWED_SCHEMA
        b["profile"] = fc.PROFILE
        b["column_sha256"] = "floating-vertical-no-bucket-column"
        b["vertical_identity"] = event.vertical.identity
    b["profile"] = GPU_PROFILE
    if original_topology is not None:  # the CONTINUED-origin profile: the caller-attested GLOBAL original, bound beside the event's carried routing origin (`initial_topology`, already in the binding)
        o = original_topology
        b["schema"] = schema_of("windowed" if windowed else "dense", True)
        b["profile"] = CONT_GPU_PROFILE
        b["continued_origin"] = {"original_identity_sha256": o.identity_sha256, "original_source_sha256": o.source_sha256, "original_topology_sha256": o.topology_sha256, "original_generation": int(o.generation)}
    return b


def _facts(v: Any, payload: Any) -> dict:
    t = v.held_template()
    return {"vertical_identity": v.identity, "root_column_sha256": t.column.sha256, "root_matrix_op_sha256": t.matrix_op.sha256, "root_pref_sha256": None if t.preferential is None else t.preferential.sha256,
            "owner_config_sha256": t.config_sha256, "generation": int(t.generation), "source_witness": t.source_witness, "soil_t_s_hex": float(payload.state.t_s).hex(), "n_layers": int(v.n_layers),
            "shape": [int(v.shape[0]), int(v.shape[1])], "has_ks": t.ks_m_s is not None, "held_layouts": _layouts_of(t.column)}


def check_held_ids(v: Any, facts: dict) -> None:
    """Metadata/ID half of the held check (pure strings; no array is read): the target's HELD template must carry the file's identities, source facts and layouts."""
    t = v.held_template()
    ok = (facts["root_column_sha256"] == t.column.sha256 and facts["root_matrix_op_sha256"] == t.matrix_op.sha256 and facts["root_pref_sha256"] == (None if t.preferential is None else t.preferential.sha256)
          and facts["owner_config_sha256"] == t.config_sha256 and facts["generation"] == int(t.generation) and facts["source_witness"] == t.source_witness and facts["has_ks"] == (t.ks_m_s is not None)
          and facts["held_layouts"] == _layouts_of(t.column))
    if not ok:
        _fail("the target's HELD template differs from the file's stored held identities / source facts / layouts (the restart target must carry the genuine ORIGINAL held law context)")


def check_held_values(v: Any, held: dict) -> None:
    """Value half of the held check: the stored held bytes against the REAL device held maps through bounded counted chunks (nothing is re-derived from the checkpoint bed)."""
    t = v.held_template()
    maps = t.column._maps
    pairs = [("theta_field_capacity", maps["theta_field_capacity"], held["theta_field_capacity"]), ("theta_saturation", maps["theta_saturation"], held["theta_saturation"]), ("depth0_m", t.depth0_m, held["depth0_m"])]
    if t.ks_m_s is not None:
        pairs.append(("ks_m_s", t.ks_m_s, held["ks_m_s"]))
    for name, dev_arr, stored in pairs:
        if not stream_equal(dev_arr, stored):
            _fail(f"the stored held {name} differs from the target's HELD template value (the restart target must carry the genuine ORIGINAL held law context)")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# device canonical depth
# ---------------------------------------------------------------------------------------------------------------------------------------------
def check_canonical_depth(event: PondEvent, v: Any, bed: Any, depth_dev: Any) -> None:
    """COLD full check that the stored/current canonical depth `D` (device) of EVERY ACTIVE column equals the native canonical depth of the loaded device bed (Dense: `floating_pond_cuda.dense_canonical_depth`, bounded batches of
    `batch_cells` gathered rows, native NumPy-order reductions, the qualified extent `V*K <= 32768` preflighted; Windowed: `windowed_canonical_depth` from the native prefix packets, `K` preflighted; an unsupported shape is a typed
    refusal). Result: ONE counted 8-byte count."""
    from maple.core.backend import to_device

    from maple_syrup import floating_pond_cuda as fpc

    cp = v.cp
    fpc.preflight_depth(v, bed)
    idx_host = np.flatnonzero(np.asarray(v._active_host()).reshape(-1)).astype(np.int64)
    idx = to_device(idx_host, cp)
    if is_windowed_bed(bed):  # native prefix packets (trusted ONLY after the genuine cold validation of the file's packets, see `verify_prefix_untrusted`) + the ordered active thickness; no deep read
        got = fpc.windowed_canonical_depth(v, bed, event.context, idx)
    else:
        got = fpc.dense_canonical_depth(v, bed, event.context, idx)
    same = fpc.scalar(cp.array_equal(got, depth_dev[idx]).astype(np.int64))
    del got, idx
    if not same:
        _fail("the current canonical depth differs from the native canonical depth of the checkpoint bed (stale or tampered depth record)")


def verify_prefix_untrusted(column: Any) -> None:
    """COLD check of a HOST-loaded Windowed column's height packets BEFORE any device placement. The content validation itself is MAPLE's: `load_windowed_state_snapshot` runs `validate_windowed_partition`, which re-encodes
    every chunk's packets from the deep store (bounded `gather_full_columns`) and refuses any byte difference. This helper adds the independent native `trusted=False` decode (NumPy namespace: it verifies the packet binding,
    base, window and deep-store version) and requires a finite non-negative column total. MAPLE requires `trusted=True` on CuPy, so the device decoder is used only afterwards and its trust rests on these two host checks."""
    from maple.surface.voxels.windowed_prefix import windowed_committed_height_total_kg

    try:
        total = np.asarray(windowed_committed_height_total_kg(column, trusted=False))
    except Exception as exc:
        _fail(f"the file's Windowed height packets failed the untrusted native validation (before any device placement): {exc}")
    if not bool(np.isfinite(total).all()) or bool((total < 0.0).any()):
        _fail("the file's Windowed height packets decode to a non-finite or negative column total (refused before any device placement)")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------------------------------------------------------------------------
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
        _fail(f"cannot publish without replacing anything (nothing was overwritten): {exc}")


def save_floating_pond_checkpoint_cuda(path: Any, state: Any, event: PondEvent, identity: dict, *, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0,
                                       host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0, memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None,
                                       max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, original_topology: Any = None) -> Path:
    """Write a NEW CUDA Dense floating checkpoint directory (module docstring). Order: event metadata -> pure argument normalisation -> the CURRENT device gate -> path/IO checks -> two-pool admission of the header-based model ->
    full validation of the live state (the floating validator, the device canonical depth) -> chunked counted D2H of the payload -> MAPLE's own snapshot writer -> staged, fsynced, no-replace publication with the manifest sealed
    LAST (a parent-directory fsync failure after publication is reported as INDETERMINATE and the published directory is preserved). Nothing is committed, rerouted, reset or finalised."""
    from maple.io.outputs.snapshot import save_state_snapshot

    v = check_event(event)
    limit = pc._max_bytes(max_bytes)
    q = quota_of(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    ident = pc._canonical_identity(identity)
    t = v.held_template()
    cp, dev = gate(t.column)  # FIRST, before any reader, IO or owned array
    continued = original_topology is not None  # the explicit CONTINUED-ORIGIN profile; `None` is the historical /7 /8 path, unchanged
    if continued:
        fc._check_original(event, original_topology)  # metadata only
    target = Path(path)
    if os.path.lexists(target):
        _fail(f"checkpoint path already exists (no overwrite): {target}")
    parent = target.resolve().parent
    if not parent.is_dir():
        _fail(f"the parent directory of the checkpoint does not exist: {parent}")
    target = parent / target.name
    if type(state) is not PondEventState or state.status != "wet":
        _fail("save needs a genuine wet PondEventState (the floating profile has no dry handoff)")
    bed = state.bed
    if not isinstance(bed, BedState):
        _fail("the state's bed is not an actual MAPLE BedState")
    windowed = bool(is_windowed_bed(bed))
    storage = "windowed" if windowed else "dense"
    if storage != storage_of(event):
        _fail(f"the bed storage ({storage}) differs from the event's configured storage ({storage_of(event)}); no cross-storage conversion")
    schema = schema_of(storage, continued)
    deep = None
    if windowed:
        pc._require_published(bed)  # a pending deep journal is refused before any other work (publish through the event's own storage policy first); never published here
        deep = pc._deep_record(bed.voxel_column, event)  # the strict source-bound record of the published deep version (metadata only; the store is REFERENCED, never copied)
    L, N = v.n_layers, v.n_cells
    layouts = _layouts_of(t.column)
    specs = {**_specs(event, dry=False), **_gpu_floating_specs(L, N, layouts, t.ks_m_s is not None), **(fc._origin_specs(event) if continued else {})}
    if windowed:  # RESIDENT bytes only (window, base, prefix packets, active, availability, water): never cells x voxels x classes
        native_bytes = pc._windowed_resident_nbytes(bed, deep, ident)
        V, K = 0, int(bed.active_layer.mass_kg.shape[-1])
        cache = max(fc.deep_cache_bound_bytes(deep, event.nc), int(bed.voxel_column.deep_store.mapped_bytes_bound))  # the ACTUAL mapping-cache bound, not the (tiny) resident archive
    else:
        vox = bed.voxel_column.mass_kg
        V, K = int(vox.shape[2]), int(vox.shape[3])
        native_bytes = pc._tree_nbytes(bed.voxel_column) + pc._tree_nbytes(bed.active_layer) + pc._tree_nbytes(bed.sediment_availability) + pc._tree_nbytes(bed.water)
        cache = 0
    n_active = int(np.count_nonzero(np.asarray(v._active_host())))
    d_need, h_need = save_models(L, N, event.nc, event.control.phase_bins, table_bytes=_spec_bytes(specs), native_bytes=native_bytes, V=V, K=K, n_active=n_active, batch_cells=v.batch_cells, windowed=windowed,
                                 cache_bytes=cache)
    admit(cp, dev, d_need, h_need, q, "the CUDA floating checkpoint save (host copies, native snapshot conversion, validation and the canonical-depth check)")
    try:
        event._check_state(state)
    except Exception as exc:
        _fail(f"the state is not valid for this event: {exc}")
    try:
        payload = v.current_payload(state.hydraulic.soil_water_m)
    except fp.FloatingPondError as exc:
        _fail(f"the state's soil water was not issued by this vertical: {exc}")
    if float(state.hydraulic.t_s) != float(payload.state.t_s):
        _fail("the hydraulic clock differs from the soil clock")
    if continued:
        from maple_syrup import floating_continued_checkpoint as cc

        arrays, scalars, _fresh = cc.validate_continued_state(event, state, None, storage, original_topology)  # the xp-neutral continued copy (two declared statements differ)
    else:
        arrays, scalars, _fresh = fc._validate_floating_state(event, state, None, storage)  # xp-neutral; it also proves the derived soil sum through the vertical's solver
    check_canonical_depth(event, v, bed, payload.depth_m)
    table: dict[str, dict] = {}
    host: dict[str, np.ndarray] = {}

    def add_host(name: str, array: Any) -> None:
        from maple.core.backend import is_device_array

        h = d2h(array) if is_device_array(array) else np.array(array, copy=True, order="C")
        if not _dtype_ok(h.dtype.str) or h.ndim > _MAX_NDIM:
            _fail(f"{name}: dtype {h.dtype.str} / ndim {h.ndim} not allowed")
        host[name] = h
        table[name] = {"dtype": h.dtype.str, "shape": [int(n) for n in h.shape]}

    for name, array in arrays.items():
        add_host(name, array)
    maps = t.column._maps
    for name, array in (("fl_water", payload.state.water_m), ("fl_wetting", payload.history.wetting_m), ("fl_cum_intake", payload.history.cumulative_intake_m), ("fl_depth", payload.depth_m),
                        ("fl_depth0", t.depth0_m), ("fl_held_fc", maps["theta_field_capacity"]), ("fl_held_sat", maps["theta_saturation"])):
        add_host(name, array)
    if t.ks_m_s is not None:
        add_host("fl_ks", t.ks_m_s)
    if continued:
        add_host(fc.ORIGIN_ARRAY, np.ascontiguousarray(fc._initial_topology(event).source_z_full, dtype=np.float64))  # host O(Npadded) carried routing origin

    def add_native(array: np.ndarray) -> str:
        name = f"native_{len(host):04d}"
        host[name] = array
        table[name] = {"dtype": array.dtype.str, "shape": [int(n) for n in array.shape]}
        return name

    native = {"ledger": _native_encode(bed.ledger, add_native), "committed_topography": _native_encode(bed.committed_topography, add_native)}
    binding = _binding(event, windowed, original_topology if continued else None)
    manifest = {"schema": schema, "profile": CONT_GPU_PROFILE if continued else GPU_PROFILE, "identity": ident, "status": "wet", "array_backend": event.xp.__name__, "hydraulic_backend": event.control.backend,
                "source": pc.pond_checkpoint_source_pair(), "binding": binding, "binding_sha256": _sha256_text(_json_dumps(binding)), "floating": _facts(v, payload), "files": {},
                "payload": {"scalars": scalars, "arrays": table, "native": native, "accounting": None},
                "native_dense": None if windowed else {"voxel_shape": [int(n) for n in vox.shape], "native_bytes": int(native_bytes)},
                "retained_quiet_reset": {"step": None if state.sediment.retained_quiet_reset_step is None else int(state.sediment.retained_quiet_reset_step)},
                "terminal_exchange": {"n_terminal_native_calls": int(state.sediment.n_terminal_native_calls)}, "kernel": _kernel()}
    if continued:
        manifest["continued_origin"] = cc.origin_record(event, arrays)
    if windowed:
        manifest["bed_storage"] = "windowed"
        manifest["deep_storage"] = deep
    try:
        _json_dumps(manifest)
    except (TypeError, ValueError) as exc:
        _fail(f"manifest is not strict JSON: {exc}")
    estimate = sum(a.nbytes for a in host.values()) + native_bytes
    if estimate > limit:
        _fail(f"checkpoint needs about {estimate} uncompressed bytes, above max_bytes = {limit}; nothing was written")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=parent))
    try:
        provenance = {"syrup_floating_pond_checkpoint_schema": schema, "requires_companion": True, "identity": ident, "binding_sha256": manifest["binding_sha256"], "array_backend": event.xp.__name__,
                      "hydraulic_backend": event.control.backend, "status": "wet"}
        if windowed:
            from maple.surface.voxels.windowed_snapshot import (
                save_windowed_state_snapshot,
            )

            provenance.update(pc._windowed_provenance(deep))
            # MAPLE's own native writer (it re-validates, refuses an existing target and only REFERENCES the deep store); a device-tree save is unqualified until Root's hardware proof
            save_windowed_state_snapshot(bed.voxel_column, bed.active_layer, scalars["t_s"], staging / PAYLOADS[0], step=scalars["counters"]["n_accepted_steps"], geometry=event.context.geometry,
                                         mass_resolution_kg=event.context.mass_resolution_kg, sediment_availability=bed.sediment_availability, water=bed.water, provenance=provenance)
        else:
            save_state_snapshot(bed.voxel_column, bed.active_layer, scalars["t_s"], staging / PAYLOADS[0], step=scalars["counters"]["n_accepted_steps"],
                                reason="SYRUP CUDA floating pond checkpoint component: companion continuation file required", geometry=event.context.geometry,
                                mass_resolution_kg=event.context.mass_resolution_kg, sediment_availability=bed.sediment_availability, water=bed.water, provenance=provenance)
        np.savez(staging / PAYLOADS[1], **host)
        host.clear()
        total, audited = 0, {}
        for name in PAYLOADS:
            audited[name] = _zip_audit(staging / name, limit, budget_used=total)
            total += sum(audited[name].values())
        _npy_preflight(staging / PAYLOADS[1], audited[PAYLOADS[1]], table)
        if windowed:
            pc._windowed_snapshot_preflight(staging / PAYLOADS[0], audited[PAYLOADS[0]], pc._native_expected(event, deep))
        else:
            _npy_preflight(staging / PAYLOADS[0], audited[PAYLOADS[0]], None)
            manifest["native_dense"]["native_member_bytes"] = int(sum(audited[PAYLOADS[0]].values()))
        files = {name: _hash_file(staging / name, limit)[0] for name in PAYLOADS}
        sealed = fc._seal({**manifest, "files": files})
        for name in PAYLOADS:  # durability: payload files first, the manifest sealed LAST, then the directory
            _fsync_file(staging / name)
        fc._write_manifest(staging / MANIFEST_NAME, sealed)
        _fsync_dir(staging)
        _publish_noreplace(staging, target)  # atomic renameat2(RENAME_NOREPLACE): never replaces anything
    except BaseException:
        if staging.exists():  # only OUR temporary directory; a published target never exists under this name
            shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        _fsync_dir(parent)
    except OSError as exc:
        _fail(f"INDETERMINATE durability: the checkpoint was published at {target} but the parent directory fsync failed ({exc}); the published directory is preserved")
    return target


def _kernel() -> dict:
    from maple_syrup import floating_held_context_cuda as fhc

    return fhc.kernel_descriptor()


# ---------------------------------------------------------------------------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _read_manifest_checked(root: Path) -> dict:
    try:
        return fc._read_manifest(root)
    except fc.FloatingCheckpointError as exc:
        _fail(str(exc))
    raise AssertionError("unreachable")


def load_floating_pond_checkpoint_cuda(path: Any, event: PondEvent, expected_identity: dict, *, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0,
                                       host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0, memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None,
                                       max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, original_topology: Any = None) -> PondEventState:
    """Restore a `/7` file into `event` (a fresh CUDA PondEvent whose vertical carries the ORIGINAL held law context: the genuine original owner vertical or the vertical of `FloatingPondVertical.from_held_context`). Returns the registered
    wet `PondEventState` (module docstring for the order)."""
    from maple.core.backend import to_device
    from maple.io.outputs.snapshot import load_state_snapshot

    from maple_syrup import floating_resume_cuda as frc
    from maple_syrup.pond_event import PondEventError

    v = check_event(event)
    limit = pc._max_bytes(max_bytes)
    q = quota_of(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    expected = pc._canonical_identity(expected_identity)
    t = v.held_template()
    cp, dev = gate(t.column)  # FIRST, before any reader, IO or owned array
    continued = original_topology is not None  # explicit CONTINUED-ORIGIN profile (`None` = the historical /7 /8 loader, which refuses a continued file)
    if continued:
        fc._check_original(event, original_topology)  # metadata only
        from maple_syrup import floating_continued_checkpoint as cc
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _fail(f"checkpoint path is not a plain directory: {root}")
    root = root.resolve()
    if {p.name for p in root.iterdir()} != {MANIFEST_NAME, *PAYLOADS}:
        _fail("floating checkpoint directory must contain exactly the manifest and the two payload files")
    try:
        msize = os.lstat(root / MANIFEST_NAME).st_size
    except OSError as exc:
        _fail(f"floating checkpoint manifest is missing: {exc}")
    from maple_syrup import floating_resume as fr

    admit(cp, dev, 0, 3 * int(msize) + HASH_BUFFER_BYTES + HEADER_AUDIT_BYTES + fr._cold_source_bytes(), q, "the CUDA floating checkpoint manifest read and first-use source hash")
    data = _read_manifest_checked(root)
    schema = data.get("schema")
    if schema not in (CONT_READ_SCHEMAS if continued else READ_SCHEMAS):
        _fail(f"checkpoint schema {schema!r} is not a {'CONTINUED-origin ' if continued else ''}CUDA floating schema {CONT_READ_SCHEMAS if continued else READ_SCHEMAS} (the CPU child /1-/6 and the historical pond checkpoints are read by their own loaders; nothing is upgraded)")
    windowed = schema in (GPU_WINDOWED_SCHEMA, CONT_GPU_WINDOWED_SCHEMA)
    if (storage_of(event) == "windowed") != windowed:
        _fail(f"the checkpoint's bed storage ({'windowed' if windowed else 'dense'}) is not the event's ({storage_of(event)}); no cross-storage conversion")
    keys = (CONT_WINDOWED_MANIFEST_KEYS if windowed else CONT_MANIFEST_KEYS) if continued else (WINDOWED_MANIFEST_KEYS if windowed else MANIFEST_KEYS)
    if set(data) != keys:
        _fail(f"manifest keys mismatch: missing {sorted(keys - set(data))}, unknown {sorted(set(data) - keys)}")
    if _hex(data["manifest_sha256"], "manifest_sha256") != fc._seal(data)["manifest_sha256"]:
        _fail("manifest hash mismatch (the manifest was modified)")
    if data["profile"] != (CONT_GPU_PROFILE if continued else GPU_PROFILE):
        _fail("unknown profile")
    if data["identity"] != expected:
        _fail("checkpoint case / source / config identity differs from the expected identity")
    if data["source"] != pc.pond_checkpoint_source_pair():
        _fail("source pair mismatch: the imported maple_syrup / maple packages differ from the ones that wrote the checkpoint (refused before any payload is read)")
    if data["kernel"] != _kernel():
        _fail("kernel/ABI descriptor mismatch: the CUDA kernels differ from the ones that wrote the checkpoint (refused before any payload is read)")
    if data["array_backend"] != "cupy" or data["hydraulic_backend"] != "cuda" or data["array_backend"] != event.xp.__name__ or data["hydraulic_backend"] != event.control.backend:
        _fail("backend mismatch: the CUDA floating file profile is cupy/cuda only and must equal the event's")
    if data["status"] != "wet":
        _fail("a floating checkpoint is a wet state")
    binding = _binding(event, windowed, original_topology if continued else None)
    if data["binding"] != binding or data["binding_sha256"] != _sha256_text(_json_dumps(binding)):
        diff = sorted(k for k in binding if not isinstance(data["binding"], dict) or data["binding"].get(k) != binding[k])
        _fail(f"event binding mismatch (static inputs differ from the ones that wrote the checkpoint): {diff}")
    facts = data["floating"]
    if not isinstance(facts, dict) or set(facts) != FLOATING_KEYS:
        _fail("floating block keys mismatch")
    if facts["vertical_identity"] != v.identity or facts["n_layers"] != v.n_layers or facts["shape"] != [int(v.shape[0]), int(v.shape[1])]:
        _fail("the floating block does not describe the target vertical")
    if not isinstance(facts["held_layouts"], dict) or set(facts["held_layouts"]) != set(LAYOUT_NAMES) or any(x not in ("scalar", "shared", "cell") for x in facts["held_layouts"].values()):
        _fail("held_layouts record is malformed")
    check_held_ids(v, facts)  # strings only, before any payload byte is read
    payload = data["payload"]
    if not isinstance(payload, dict) or set(payload) != pc.PAYLOAD_KEYS or payload["accounting"] is not None:
        _fail("payload keys mismatch")
    files = data["files"]
    if not isinstance(files, dict) or set(files) != set(PAYLOADS):
        _fail("payload file set mismatch")
    rec = data["terminal_exchange"]
    if not isinstance(rec, dict) or set(rec) != {"n_terminal_native_calls"} or type(rec["n_terminal_native_calls"]) is not int or rec["n_terminal_native_calls"] < 0:
        _fail("terminal_exchange record is malformed")
    terminal_calls = rec["n_terminal_native_calls"]
    rrec = data["retained_quiet_reset"]
    if not isinstance(rrec, dict) or set(rrec) != {"step"} or not (rrec["step"] is None or (type(rrec["step"]) is int and rrec["step"] >= 0)):
        _fail("retained_quiet_reset record is malformed")
    reset_step = rrec["step"]
    deep = None
    if windowed:
        if data["bed_storage"] != "windowed" or data["native_dense"] is not None:
            _fail("bed_storage / native_dense mismatch (a Windowed file carries no dense native record)")
        deep = pc._check_deep_record(data["deep_storage"], event, limit)  # the strict source-bound deep-store record, from the manifest alone, BEFORE any payload is opened
    for name in PAYLOADS:
        if _hash_file(root / name, limit)[0] != _hex(files[name], f"files[{name}]"):
            _fail(f"checkpoint payload hash mismatch: {name}")
    total, member_sizes = 0, {}
    for name in PAYLOADS:
        sizes = _zip_audit(root / name, limit, budget_used=total)
        total += sum(sizes.values())
        member_sizes[name] = sizes
    L, N = v.n_layers, v.n_cells
    specs = {**_specs(event, dry=False), **_gpu_floating_specs(L, N, facts["held_layouts"], bool(facts["has_ks"])), **(fc._origin_specs(event) if continued else {})}
    table = payload["arrays"]
    _audit_array_table(table, specs, member_sizes[PAYLOADS[1]], limit)
    _npy_preflight(root / PAYLOADS[1], member_sizes[PAYLOADS[1]], table)
    if windowed:
        pc._windowed_snapshot_preflight(root / PAYLOADS[0], member_sizes[PAYLOADS[0]], pc._native_expected(event, deep))
        native_bytes = int(sum(member_sizes[PAYLOADS[0]].values()))  # RESIDENT member bytes only
        V, K, cache = 0, int(event.nc), fc.deep_cache_bound_bytes(deep, event.nc)  # the deep-store MAPPING-CACHE bound from the verified record (not the resident archive size)
    else:
        _npy_preflight(root / PAYLOADS[0], member_sizes[PAYLOADS[0]], None)
        native_bytes = fc._dense_native_dims_check(root / PAYLOADS[0], member_sizes[PAYLOADS[0]], data["native_dense"])
        V, K = int(data["native_dense"]["voxel_shape"][2]), int(data["native_dense"]["voxel_shape"][3])
        cache = 0
    table_bytes = sum(int(np.prod(r["shape"], dtype=np.int64)) * np.dtype(r["dtype"]).itemsize for r in table.values())
    n_active = int(np.count_nonzero(np.asarray(v._active_host())))
    d_need, h_need = load_models(L, N, event.nc, event.control.phase_bins, table_bytes=table_bytes, dev_table_bytes=_spec_bytes(_specs(event, dry=False), device_only=True), native_bytes=native_bytes, V=V, K=K,
                                 n_active=n_active, batch_cells=v.batch_cells, windowed=windowed, cache_bytes=cache)
    admit(cp, dev, d_need, h_need, q, "the CUDA floating checkpoint load (host companion + native snapshot, placed bed, preparation and the canonical-depth check)")
    if windowed:
        from maple.surface.voxels.windowed_snapshot import load_windowed_state_snapshot

        remaining = limit - sum(member_sizes[PAYLOADS[1]].values())
        if remaining < 1:
            _fail("max_bytes leaves no room for the native snapshot after the companion")
        ctx = event.context
        try:  # the NATIVE protocol: a HOST load (the deep store is opened read-only and its partition validation streamed); the device placement happens only in `pc._build_state`
            snapshot = load_windowed_state_snapshot(root / PAYLOADS[0], geometry=ctx.geometry, grain_classes=ctx.grain_classes, mass_resolution_kg=ctx.mass_resolution_kg, max_bytes=int(remaining))
        except Exception as exc:
            _fail(f"MAPLE windowed snapshot cannot be loaded (is the referenced deep-store version retained and unmodified?): {exc}")
        fc._check_windowed_snapshot_floating(snapshot, data, deep)  # provenance, published column, EXACT deep-store identity/geometry, and the loaded handle's mapping bound vs the modelled one
        verify_prefix_untrusted(snapshot["voxel_column"])  # untrusted native packet validation BEFORE any device placement
    else:
        try:
            snapshot = load_state_snapshot(root / PAYLOADS[0])
        except Exception as exc:
            _fail(f"MAPLE snapshot cannot be loaded: {exc}")
        if not snapshot["validated"] or not snapshot["restartable"] or snapshot["is_diagnostic"]:
            _fail("invalid MAPLE component snapshot (not validated / restartable, or diagnostic)")
        try:
            prov = json.loads(snapshot["provenance_json"])
        except ValueError:
            prov = None
        if (not isinstance(prov, dict) or prov.get("syrup_floating_pond_checkpoint_schema") != schema or prov.get("binding_sha256") != data["binding_sha256"] or prov.get("identity") != data["identity"]
                or prov.get("status") != "wet" or prov.get("array_backend") != data["array_backend"] or prov.get("hydraulic_backend") != data["hydraulic_backend"] or prov.get("requires_companion") is not True):
            _fail("MAPLE snapshot provenance is not bound to this manifest")
        if snapshot["water"] is None or snapshot["sediment_availability"] is None or snapshot["history_store_path"] or any(snapshot.get(k) is not None for k in pc._FOREIGN_SNAPSHOT_FIELDS):
            _fail("MAPLE snapshot lacks water / availability or carries foreign optional state")
    used: set[str] = set()
    try:
        archive = np.load(root / PAYLOADS[1], allow_pickle=False)
    except Exception as exc:
        _fail(f"companion cannot be opened: {exc}")
    with archive:
        if set(archive.files) != set(table):
            _fail("companion arrays differ from the array table")

        def get_array(name: Any) -> np.ndarray:
            if not isinstance(name, str) or name not in table:
                _fail(f"unknown array reference {name!r}")
            if name in used:
                _fail(f"aliased array {name}")
            used.add(name)
            rec2 = table[name]
            try:
                a = np.array(archive[name], copy=True, order="C")
            except (ValueError, OSError, zipfile.BadZipFile, EOFError) as exc:
                _fail(f"array {name} cannot be read: {exc}")
            if a.dtype.str != rec2["dtype"] or list(a.shape) != rec2["shape"]:
                _fail(f"array {name}: stored dtype / shape differ from the table")
            if a.dtype.kind == "f" and not np.isfinite(a).all():
                _fail(f"array {name} holds non-finite values")
            a.flags.writeable = False
            return a

        if not isinstance(payload["native"], dict) or set(payload["native"]) != {"ledger", "committed_topography"}:
            _fail("native companion keys mismatch")
        host_arrays = {name: get_array(name) for name in specs}
        ledger = _native_decode(payload["native"]["ledger"], get_array)
        committed = _native_decode(payload["native"]["committed_topography"], get_array)
    if used != set(table):
        _fail("unused or missing companion arrays")
    held = {"theta_field_capacity": host_arrays["fl_held_fc"], "theta_saturation": host_arrays["fl_held_sat"], "depth0_m": host_arrays["fl_depth0"]}
    if facts["has_ks"]:
        held["ks_m_s"] = host_arrays["fl_ks"]
    check_held_values(v, held)  # the stored held bytes vs the REAL device held template, before any state is built
    if continued:
        cc.check_origin_arrays(event, data["continued_origin"], host_arrays, original_topology)  # stored carried routing / global datum vs the caller's event and original, before any state is built
    current = {"water_m": host_arrays["fl_water"], "wetting_m": host_arrays["fl_wetting"], "cumulative_intake_m": host_arrays["fl_cum_intake"], "depth_m": host_arrays["fl_depth"]}
    scalars = payload["scalars"]
    xp = event.xp
    state = pc._build_state(event, scalars, host_arrays, lambda n: to_device(host_arrays[n], xp), lambda n: np.array(host_arrays[n], copy=True), snapshot, ledger, committed, "wet")
    del snapshot, ledger, committed
    state = dataclasses.replace(state, sediment=dataclasses.replace(state.sediment, n_terminal_native_calls=terminal_calls, retained_quiet_reset_step=reset_step))
    for name in [n for n in host_arrays if not n.startswith("fl_")]:  # the placed device copies are the authority; release the host pond arrays before the device validation
        del host_arrays[name]
    with _minting():
        inputs = VerifiedFloatingInputsCuda(schema, storage_of(event), event.config_sha256, event.forcing_sha256, v.identity, v.backend, float.fromhex(facts["soil_t_s_hex"]), facts["root_column_sha256"],
                                            facts["root_matrix_op_sha256"], facts["root_pref_sha256"], facts["owner_config_sha256"], int(facts["generation"]), facts["source_witness"],
                                            MappingProxyType(dict(data["source"])), MappingProxyType(dict(data["kernel"])), MappingProxyType(dict(facts["held_layouts"])), MappingProxyType(held),
                                            MappingProxyType(current), state, _json_dumps(scalars))
    _DIGESTS[inputs] = input_digest(inputs)  # computed OUTSIDE the tight mint window
    prepared, new_state = frc.prepare_from_verified_inputs(event, inputs, quota=q)
    with v.resume_scope(prepared):  # provisional resolution: the registry is NOT touched; the INDEPENDENT validation runs here
        try:
            check_canonical_depth(event, v, new_state.bed, v.current_payload(new_state.hydraulic.soil_water_m).depth_m)  # the provisional payload's device depth
            if continued:
                _a, check_scalars, _fresh = cc.validate_continued_state(event, new_state, None, storage_of(event), original_topology)
            else:
                _a, check_scalars, _fresh = fc._validate_floating_state(event, new_state, None, storage_of(event))
            if _json_dumps(check_scalars) != inputs.scalars_json:
                _fail("scalars of the reconstructed state differ from the saved payload")
            event._check_state(new_state)
        except (PondCheckpointError, PondEventError, fp.FloatingPondError) as exc:
            if isinstance(exc, fc.FloatingCheckpointError):
                raise
            _fail(f"the restored floating state failed the independent validation: {exc}")
    v.adopt_resumed(prepared)  # the ONE registration, LAST
    return new_state

