"""Source-bound physical continuation checkpoint of the persistent layered-water event (CPU NumPy/Numba, Dense/nonnative).

A NEW closed schema (`SCHEMA`) and two profiles. It saves the whole physical continuation of a `layered_event.LayeredEventState`
(soil water, matrix wetting history, lifetime intake, surface depth, signed face momentum, cumulative and anchor ledgers,
adaptive cap, forcing cursor, quiet window, rejection counters, completion) WITHOUT committing, rerouting, settling, advancing,
resetting, or touching plants, sediment or wind. It is not an adaptation of `checkpoint.py` / `pond_checkpoint.py` (which stay
unchanged and cannot decode it) and it is not scheduler acceptance: documentation `docs/subsurface_hydrology/layered_checkpoint.md`,
design `agent_handoffs/tasks/phase1_layered_checkpoint_20261110/design.md`.

    save_layered_checkpoint(path, event, state, identity, *, limits, host_available_bytes_fn) -> SaveReceipt
    load_layered_checkpoint(path, expected_event, expected_identity, *, limits, allow_complete=False,
                            host_available_bytes_fn) -> LayeredRestoration(event, state, receipt)

Profiles (never cross-loadable):
  `native39_dense_fixed_bed_water_v1`  actual native MAPLE Dense `BedState` (genuine `save_state_snapshot` / `load_state_snapshot`,
                                       exactly its 22 members) + a finite companion carrying the REAL native ledger (Dense or
                                       genuine Compact) and committed-topography record, pending accounting preserved.
  `nonnative_fixed_bed_water_v1`       the genuine `bed=None` owner: manifest + companion; the surface is saved in the companion;
                                       no native snapshot and no fake Dense wrapper exist.
  `native39_windowed_cpu_full_v1`       EXPLICIT opt-in (`bed_storage="windowed"`): the actual native resident-window `WindowedVoxelColumnState`
                                       bed over a PUBLISHED full/v1 external deep store (genuine `save/load_windowed_state_snapshot`, exactly
                                       its 12 members, no pending deep journal) + the same companion. The deep files stay external, immutable and
                                       NOT archived. Overlay/v2 stores are refused before any deep work.
By default (`bed_storage="dense"`) actual native Windowed beds are refused BEFORE any native or deep operation; CUDA is refused
(no hardware qualification here).

Publication = exclusive `mkdir` reservation + manifest-LAST `os.link` seal (atomic no-clobber); readers reject an unsealed directory.
The loader copies each payload through its open descriptor into a private 0700 directory while hashing, so the hash, the archive
audit and the native reopen see the same bytes. Limits are an explicit caller record; nothing is raised automatically.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import shutil
import stat
import tempfile
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from maple_syrup import layered_event as ev
from maple_syrup import layered_storm as ls
from maple_syrup import matrix_infiltration as mi
from maple_syrup import pond_checkpoint as pc
from maple_syrup import subsurface as ss
from maple_syrup.pond_network import build_pond_topology
from maple_syrup.sediment_bed import BedState, is_windowed_bed
from maple_syrup.surface_routing import SurfaceState

__all__ = [
    "MANIFEST_NAME", "NATIVE_PROFILE", "NONNATIVE_PROFILE", "WINDOWED_PROFILE", "PAYLOAD_NATIVE", "PAYLOAD_COMPANION", "SCHEMA",
    "CheckpointBudget", "CheckpointLimits", "LayeredCheckpointDeepStoreError", "LayeredCheckpointError", "LayeredCheckpointIndeterminate",
    "LayeredCheckpointQuotaError", "LayeredRestoration", "SaveReceipt", "estimate_checkpoint_budget",
    "estimate_windowed_checkpoint_budget", "external_metadata_bounds", "layered_checkpoint_source_pair", "load_layered_checkpoint", "physical_digest", "save_layered_checkpoint",
]

SCHEMA = "maple-syrup-layered-event-checkpoint/1"
NATIVE_PROFILE = "native39_dense_fixed_bed_water_v1"
NONNATIVE_PROFILE = "nonnative_fixed_bed_water_v1"
WINDOWED_PROFILE = "native39_windowed_cpu_full_v1"
PROFILES = (NATIVE_PROFILE, NONNATIVE_PROFILE, WINDOWED_PROFILE)
BED_STORAGES = ("dense", "windowed")
MANIFEST_NAME = "layered_checkpoint.json"
PAYLOAD_NATIVE = "maple_state.npz"
PAYLOAD_COMPANION = "layered_state.npz"
FILES = {NATIVE_PROFILE: (PAYLOAD_NATIVE, PAYLOAD_COMPANION), NONNATIVE_PROFILE: (PAYLOAD_COMPANION,),
         WINDOWED_PROFILE: (PAYLOAD_NATIVE, PAYLOAD_COMPANION)}
STATUSES = ("wet", "complete")
MANIFEST_KEYS = frozenset({"schema", "profile", "identity", "status", "array_backend", "hydraulic_backend", "source", "runtime",
                           "binding", "binding_sha256", "files", "sizes", "payload", "manifest_sha256"})
WINDOWED_MANIFEST_KEYS = MANIFEST_KEYS | {"deep_storage"}
PAYLOAD_KEYS = frozenset({"scalars", "arrays", "native"})
# --- declared metadata policy (bytes, node, string, depth, header and member caps; all closed by the vocabulary) ----------------------
IDENTITY_MAX_NODES = 256           # caller identity, checked by an iterative walk BEFORE any canonical serialisation
IDENTITY_MAX_DEPTH = 6
IDENTITY_MAX_STRING = 2048
IDENTITY_MAX_CHARS = 16384         # summed key + string characters
MAX_IDENTITY_BYTES = 64 * 1024     # canonical (ASCII-escaped) JSON of the identity
MANIFEST_FIXED_BYTES = 32 * 1024 + MAX_IDENTITY_BYTES   # scalars, binding, source, runtime, files + the one identity copy
MANIFEST_PER_ENTRY_BYTES = 1024    # per companion array entry or native leaf (pretty-printed table record / node tree)
MANIFEST_FIXED_NODES = 2048        # includes the identity's <= IDENTITY_MAX_NODES
MANIFEST_NODES_PER_ENTRY = 24
MANIFEST_MAX_DEPTH = 12
MANIFEST_MAX_STRING = MAX_IDENTITY_BYTES
NODE_BYTES = 256                   # upper bound per decoded JSON node (CPython dict entry <= ~100 B, str 49 B + len, int/float 32 B)
ZIP_NAME_MAX = 64
ZIP_EXTRA_MAX = 64                 # numpy writes a 20-28 byte ZIP64 extra per member
ZIP_CD_ENTRY_MAX = 46 + ZIP_NAME_MAX + ZIP_EXTRA_MAX
ZIP_INFO_OBJECT_BYTES = 2048       # one ZipInfo object with its strings
MAX_MEMBERS = 128
MAX_NPY_HEADER = 1024              # closed dtype/shape vocabulary: <= 5 dimensions, fixed dtype strings
NATIVE_STRING_MAX = 1024           # every native string member except provenance
NATIVE_PROVENANCE_MAX = MAX_IDENTITY_BYTES + 4096
NPY_HEADER_ALLOWANCE = 256     # numpy writes 128-byte-aligned headers; the exact member size is audited
ZIP_MEMBER_ALLOWANCE = 512     # local header + central entry + data descriptor, bounded names
PROVENANCE_ALLOWANCE = NATIVE_PROVENANCE_MAX
NPY_WRITE_CHUNK_BYTES = 16 * 1024**2  # numpy's own buffered writer chunk (`format.write_array` buffersize)
NPY_READ_BUFFER_BYTES = 1 << 18       # numpy `format.BUFFER_SIZE` for `read_array` on a zip member
IO_CHUNK_BYTES = 1 << 20
HASH_CHUNK_BYTES = ls.HASH_CHUNK_BYTES
#: Source-derived temporary counts (see `estimate_checkpoint_budget` and the term table `BUDGET_TERM_SOURCES`):
#: `surface/voxels/diagnostics.py check_voxel_column_state` holds one (N,V,K) bool mask (finite / negative, sequential; x2 allowance) and
#: ~16 (N,V)-wide arrays (voxel_totals, local_scale + nested maximum temporaries, tolerance masks, full/partial/empty, int code, where temporaries).
VOXEL_SCAN_MASKS = 2
VOXEL_SCAN_NV_TEMPORARIES = 16
CLASS_SCAN_NK_TEMPORARIES = 16      # availability / active-layer / water class scans: (N,K)-wide temporaries
LEDGER_LEAF_TEMPORARIES = 4         # ledger / committed validators and reconciliation: <= 4 leaf-sized temporaries (mask + float)
OWNER_AUDIT_LN_TEMPORARIES = 2      # `layered_event._volumes` / `_validate_inventories`: (L,N) where/compare temporaries
# --- Windowed profile: deep-record vocabulary and native metadata policy (see design.md section 6) ---------------------------------
DEEP_RECORD_KEYS = frozenset({"deep_chunk_encoding", "deep_overlay_max_bytes", "store_format_version", "physical_manifest_digest",
                              "logical_manifest_digest", "header_digest", "store_dir", "generation", "n_voxels", "window_voxels",
                              "chunk_columns", "n_chunks", "class_ids", "packet_terms"})
WINDOWED_META_KEYS = frozenset({"schema", "store_dir", "manifest_digest", "generation", "header_digest", "n_voxels", "time_s", "step",
                                "class_ids", "provenance"})
WINDOWED_NATIVE_ARRAYS = ("window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg", "prefix_packet", "active_mass_kg",
                          "available_mass_kg", "bound_mass_kg", "water_depth_m", "mobile_mass_kg")
WINDOWED_SNAPSHOT_SCHEMA = "maple-windowed-snapshot/1"
STORE_DIR_MAX = 4096               # characters of the absolute external deep-store path
CLASS_ID_MAX = 64
METADATA_JSON_NODES = 1024         # structural node cap of the native metadata JSON (provenance + class ids dominate)
METADATA_JSON_DEPTH = 10
_SNAPSHOT_REASON = "SYRUP layered event checkpoint component: companion continuation file required"
_NATIVE_STRING_MEMBERS = ("run_name", "reason", "error_type", "error_message", "history_schema_version", "provenance_json",
                          "history_store_path")
_NATIVE_BOOL_MEMBERS = ("validated", "is_diagnostic", "restartable", "has_sediment_availability", "has_water")
_FOREIGN_SNAPSHOT_FIELDS = pc._FOREIGN_SNAPSHOT_FIELDS + ("coupled_trajectory_semantics",)
_LEDGER_FIELDS = tuple(f.name for f in dataclasses.fields(ls.LayeredWaterLedger))
_SOURCE_CACHE: dict = {}
struct_error = struct.error

# Indirections so tests can inject genuine OS failures at exactly one boundary.
_fsync = os.fsync
_link = os.link
_mkdir = os.mkdir


class LayeredCheckpointError(RuntimeError):
    """Definite refusal: the checkpoint was not sealed (save) or nothing was adopted (load). Inputs and files unchanged."""


class LayeredCheckpointQuotaError(LayeredCheckpointError):
    """A caller limit or a reported availability is below the derived requirement; raised before the guarded allocation."""


class LayeredCheckpointDeepStoreError(LayeredCheckpointError):
    """The native Windowed snapshot or its EXTERNAL deep-store dependency refused (missing, corrupted, changed or inconsistent data). The
    deep files are an immutable dependency, never copied, repaired, pruned or substituted by a sibling/latest version; inputs unchanged."""


class LayeredCheckpointIndeterminate(LayeredCheckpointError):
    """A failure AFTER the atomic seal link: the bundle may be durable. Nothing was deleted; do not retry over it."""


def _refuse(message: str) -> None:
    raise LayeredCheckpointError(message)


def _convert(exc: BaseException) -> LayeredCheckpointError:
    return LayeredCheckpointError(f"{type(exc).__name__}: {exc}")


# --- records ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckpointLimits:
    """Four INDEPENDENT caller quotas, all required: summed uncompressed serialized bytes on disk, host peak bytes, device peak
    bytes and host transient scratch bytes. The device quota is meaningful only for a CUDA profile (refused here); NumPy/Numba
    need 0 device bytes. A quota is never raised or inferred; a reported availability is a reading, not a reservation."""

    max_serialized_bytes: int
    max_host_bytes: int
    max_device_bytes: int
    max_scratch_bytes: int

    def validated(self) -> CheckpointLimits:
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
                _refuse(f"limits.{f.name} must be a non-Boolean integer >= 0, got {value!r}")
        return self


@dataclass(frozen=True)
class CheckpointBudget:
    """Derived from the ACTUAL reference arrays and the declared metadata policy (never a fitted factor or an arbitrary floor).

    SEMANTICS (whole-owner quota). `host_save_bytes` / `host_load_bytes` = the live resident set of the caller's reference owner
    (`resident_live_bytes`: the owner's accepted aggregate + prepared maps, the actual native non-water leaves, the context's
    availability fractions, the event's owned arrays; native water aliases the owner's surface/mobile and is counted once) PLUS the codec's own
    transient bytes. `limits.max_host_bytes` is compared with that whole figure, and so is the reported `host_available_bytes_fn()`:
    like the existing column convention, resident inputs are counted again under a system-free reading (conservative), and a fixed
    caller budget is never treated as extra free memory. `scratch_bytes` is the non-payload transient (metadata, validators, I/O chunks)
    under its own independent quota. Exclusions: allocator pools/fragmentation, JIT/import workspaces, the owner's declared reserves
    (admitted by the genuine owner's own admissions, never raised or ignored here), the interpreter and everything outside the owner."""

    profile: str
    n_cells: int
    n_layers: int
    n_classes: int
    n_voxels: int
    native_snapshot_bytes: int
    ledger_bytes: int
    committed_bytes: int
    companion_bytes: int
    n_members_native: int
    n_members_companion: int
    manifest_cap_bytes: int
    manifest_max_nodes: int
    serialized_bytes: int
    metadata_bytes: int
    validation_scratch_bytes: int
    scratch_bytes: int
    resident_live_bytes: int
    host_save_transient_bytes: int
    host_load_transient_bytes: int
    host_save_bytes: int
    host_load_bytes: int
    device_bytes: int
    disk_scratch_bytes: int
    terms: dict

    def as_dict(self) -> dict:
        return dict(self.__dict__)


#: term -> (source operation, lifetime). The coefficients in `estimate_checkpoint_budget` are COUNTS of the arrays those operations hold
#: (upper bounds read from the native source), not fitted multipliers.
BUDGET_TERM_SOURCES = {
    "resident_owner": "layered_storm.estimate_layered_memory.resident_bytes (accepted aggregate incl. native mobile + surface, prepared maps); live for the whole call",
    "resident_native": "actual nbytes of voxel, active layer, availability (2), ledger leaves, committed leaves and the context's initial_available_fraction; water aliases are in resident_owner; live for the whole call",
    "resident_event": "layered_event.required_bytes - operator required_bytes: owned forcing/quiet arrays; live for the whole call",
    "voxel_scan": "surface/voxels/diagnostics.check_voxel_column_state: <= VOXEL_SCAN_MASKS (N,V,K) bool masks + VOXEL_SCAN_NV_TEMPORARIES (N,V) float-wide arrays; transient inside one validator call",
    "class_scan": "availability / active-layer / water class validators: CLASS_SCAN_NK_TEMPORARIES (N,K) float arrays; transient",
    "ledger_scan": "validate_ledger_state / committed validators / _reconcile_ledger_with_physical_state: LEDGER_LEAF_TEMPORARIES x the largest leaf; transient",
    "owner_audit": "layered_event.audit_event + _validate_inventories + router.validate_state: OWNER_AUDIT_LN_TEMPORARIES (L,N) + a few (N) arrays and bool masks; transient",
    "capture": "save: host copies of every companion array and native-node leaf (np.array copy) until np.savez returns",
    "writer_chunks": "np.savez/format.write_array buffered write: 2 x min(16 MiB, largest member); transient inside the write",
    "native_load": "snapshot.load_state_snapshot: every member read exactly once and kept (voxel, active, availability x2, depth, mobile) + active_mass_kg re-read for its shape + 2 x numpy read buffer + validate_water_state masks (N*K + N bytes); alive until the restored bed is released",
    "decode": "restore: companion arrays decoded (np.load read + np.array copy: one extra largest member transient) while the native snapshot is alive",
    "restore": "new state copies (soil L*N, history 2N), new operator statics, new event arrays, and prepare/validation scratch while native + companion arrays are alive",
    "retained_metadata": "the parsed manifest tree kept by _load while _restore runs: manifest_nodes x NODE_BYTES + string payload (<= byte cap); lifetime = restore phase",
    "metadata": "manifest raw + decoded + tree (declared caps), identity work, zip central directory, native strings; see _metadata_bytes",
}


@dataclass(frozen=True)
class SaveReceipt:
    path: str
    profile: str
    status: str
    manifest_sha256: str
    files: dict
    sizes: dict
    t_s: float
    generation: int
    physical_digest: str
    budget: dict
    source: dict


@dataclass(frozen=True, eq=False)
class LayeredRestoration:
    """The NEWLY prepared event and the restored state bound to it. Nothing is adopted: the caller decides."""

    event: Any
    state: Any
    receipt: dict


def layered_checkpoint_source_pair() -> dict:
    """The digests of the ACTUAL imported `maple_syrup` and `maple` packages (the pond codec's own primitive)."""
    try:
        return pc.pond_checkpoint_source_pair()
    except pc.PondCheckpointError as exc:
        raise _convert(exc) from None


# --- generic strict helpers ------------------------------------------------------------------------------------------------------
def _fx(value: float) -> str:
    return float(value).hex()


def _unhex(text: Any, name: str, *, lo: float | None = None) -> float:
    if not isinstance(text, str):
        _refuse(f"{name} must be a float hex string")
    try:
        value = float.fromhex(text)
    except (ValueError, OverflowError):
        raise LayeredCheckpointError(f"{name} is not a valid float hex string") from None
    if not math.isfinite(value) or (lo is not None and value < lo) or value.hex() != text:
        _refuse(f"{name} is not a canonical finite float{'' if lo is None else f' >= {lo}'}")
    return value


def _int(value: Any, name: str, lo: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < lo or int(value) >= 2**62:
        _refuse(f"{name} must be a non-Boolean integer in [{lo}, 2**62)")
    return int(value)


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _refuse(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _bound_identity(identity: Any) -> None:
    """Structural bound of the caller identity BEFORE any copy or canonical serialisation. Vocabulary: EXACT builtin `dict` (str keys),
    `list`, `tuple`, `str`, `int` (64-bit), finite `float`, `bool`, `None` (subclasses are refused: their iteration is no resource guarantee;
    this is the JSON contract the canonical serialiser already enforced). Every node is CHARGED against the cap with `len()` BEFORE a
    container's children are iterated or queued (dict: keys + values; list/tuple: items), so an enormous container, a repeated
    sub-container or a cycle is refused by constant work with a pending stack never larger than the cap; nothing proportional to hostile
    input is allocated. Depth is checked on every popped node, strings and characters on keys/values."""
    if type(identity) is not dict:
        _refuse(f"identity must be an exact dict, got {type(identity).__name__}")
    nodes = 1
    chars = 0
    stack: list = [(identity, 1)]
    while stack:
        value, depth = stack.pop()
        if depth > IDENTITY_MAX_DEPTH:
            _refuse(f"identity exceeds the structural depth bound {IDENTITY_MAX_DEPTH}")
        kind = type(value)
        if kind is dict:
            nodes += 2 * len(value)                      # charged before iterating
            if nodes > IDENTITY_MAX_NODES:
                _refuse(f"identity exceeds the structural node bound ({IDENTITY_MAX_NODES})")
            for key, item in value.items():
                if type(key) is not str:
                    _refuse("identity keys must be exact strings at every depth")
                chars += len(key)
                if len(key) > IDENTITY_MAX_STRING or chars > IDENTITY_MAX_CHARS:
                    _refuse("identity exceeds the structural string bound")
                stack.append((item, depth + 1))
        elif kind is list or kind is tuple:
            nodes += len(value)                          # charged before iterating
            if nodes > IDENTITY_MAX_NODES:
                _refuse(f"identity exceeds the structural node bound ({IDENTITY_MAX_NODES})")
            for item in value:
                stack.append((item, depth + 1))
        elif kind is str:
            chars += len(value)
            if len(value) > IDENTITY_MAX_STRING or chars > IDENTITY_MAX_CHARS:
                _refuse("identity exceeds the structural string bound")
        elif kind is bool or value is None:
            pass
        elif kind is int:
            if not -(2**63) <= value < 2**63:
                _refuse("identity integers must fit in 64 bits")
        elif kind is float:
            if not math.isfinite(value):
                _refuse("identity floats must be finite")
        else:
            _refuse(f"identity must be JSON-serialisable with exact builtin types: unsupported {kind.__name__}")


def _canonical_identity(identity: Any) -> dict:
    _bound_identity(identity)  # structural bound first; only then the (now bounded) canonicalisation
    try:
        ident = pc._canonical_identity(identity)
        if len(pc._json_dumps(ident)) > MAX_IDENTITY_BYTES:
            _refuse("identity exceeds the bounded canonical size")
    except pc.PondCheckpointError as exc:
        raise _convert(exc) from None
    return ident


_RUNTIME_KEYS = frozenset({"python_implementation", "python_version", "byteorder", "machine", "numpy", "float64_eps",
                           "numba", "llvmlite"})


def runtime_profile(hydraulic_backend: str) -> dict:
    """The CPU runtime profile, FRESHLY inspected from the running interpreter (never caller-supplied, never a repo HEAD): Python
    implementation/version, byte order, machine, NumPy, the float64 epsilon, and Numba/llvmlite only for the compiled path
    (None for NumPy, which then never imports Numba). The closed key set is compared exactly at load before any payload work."""
    import platform
    import sys

    numba = llvmlite = None
    if hydraulic_backend == "numba":
        import llvmlite as _llvmlite
        import numba as _numba

        numba, llvmlite = str(_numba.__version__), str(_llvmlite.__version__)
    return {"python_implementation": platform.python_implementation(), "python_version": platform.python_version(),
            "byteorder": sys.byteorder, "machine": platform.machine(), "numpy": str(np.__version__),
            "float64_eps": float(np.finfo(np.float64).eps).hex(), "numba": numba, "llvmlite": llvmlite}


def _check_runtime_record(record: Any, hydraulic_backend: str) -> None:
    if (not isinstance(record, dict) or set(record) != _RUNTIME_KEYS
            or any(v is not None and (not isinstance(v, str) or len(v) > 64) for v in record.values())):
        _refuse("the manifest's runtime record is missing, unknown-keyed or invalid")
    fresh = runtime_profile(hydraulic_backend)
    if record != fresh:
        diff = sorted(k for k in fresh if record.get(k) != fresh[k])
        _refuse(f"runtime mismatch: the freshly inspected runtime differs from the one that wrote the checkpoint in {diff} "
                "(refused before any payload work; no migration)")


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_value(h: Any, value: Any, depth: int = 0) -> None:
    """Bounded-staging digest of arrays (streamed), dataclasses, tuples, floats (hex), ints, bools, strings and None."""
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
    elif isinstance(value, np.ndarray):
        h.update(b"A")
        ls._stream_hash(h, value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        h.update(b"D" + type(value).__name__.encode() + b"\0")
        for f in dataclasses.fields(value):
            h.update(f.name.encode() + b"\0")
            _digest_value(h, getattr(value, f.name), depth + 1)
    elif isinstance(value, (tuple, list)):
        h.update(b"T" + str(len(value)).encode() + b"\0")
        for item in value:
            _digest_value(h, item, depth + 1)
    else:
        _refuse(f"digest: unsupported value {type(value).__name__}")


# --- reference binding -------------------------------------------------------------------------------------------------------------
def _require_cpu_dense(event: Any, state: Any | None, bed_storage: str = "dense") -> None:
    """Refusals that precede every native or deep operation: bed classification (identity of the genuine class objects) against the
    requested storage, unsupported backend. Only types are inspected: no snapshot, scan, copy or deep-store read. With the default
    `bed_storage="dense"` an actual native Windowed bed is refused (unchanged); with `"windowed"` ONLY an actual Windowed bed passes."""
    if bed_storage not in BED_STORAGES:
        _refuse(f"bed_storage must be one of {BED_STORAGES}, got {bed_storage!r}")
    op = getattr(event, "op", None)
    beds = []
    if op is not None and getattr(op, "bed", None) is not None:
        beds.append(op.bed)
    if state is not None and getattr(state, "water", None) is not None and getattr(state.water, "bed", None) is not None:
        beds.append(state.water.bed)
    for bed in beds:
        if type(bed) is not BedState:
            _refuse("the bed must be the genuine sediment_bed.BedState")
        try:
            windowed = is_windowed_bed(bed)
        except Exception as exc:  # unknown / subclass / duck-typed representation or a binding error
            raise LayeredCheckpointError(f"unsupported bed representation: {exc}") from None
        if windowed and bed_storage == "dense":
            _refuse("an actual native Windowed bed is not supported by the default (dense) layered checkpoint profile (refused before "
                    "any native or deep operation); select bed_storage='windowed' for the explicit Windowed profile. Existing "
                    "Windowed codecs are unchanged")
        if not windowed and bed_storage == "windowed":
            _refuse("bed_storage='windowed' needs an actual native Windowed bed (no dense/nonnative conversion; refused before any "
                    "native or deep operation)")
    if bed_storage == "windowed" and not beds:
        _refuse("bed_storage='windowed' needs an actual native Windowed bed; this owner has none")


def _profile(event: Any) -> str:
    if event.op.bed is None:
        return NONNATIVE_PROFILE
    return WINDOWED_PROFILE if is_windowed_bed(event.op.bed) else NATIVE_PROFILE


def _check_backend(event: Any) -> None:
    op = event.op
    if op.backend not in ("numpy", "numba") or op.xp is not np:
        _refuse(f"backend {op.backend!r} is not supported by the layered checkpoint codec (CUDA qualification is pending; "
                "no cross-backend migration)")


def _context_digest(ctx: Any) -> str:
    try:
        return pc._digest_of(ctx, schema=SCHEMA)
    except pc.PondCheckpointError as exc:
        raise _convert(exc) from None


def _bed_dims(op: Any) -> tuple[int, int]:
    """(V, K) of the reference native bed; (0, 0) without a bed."""
    if op.bed is None:
        return 0, 0
    if is_windowed_bed(op.bed):
        return int(op.bed.voxel_column.n_voxels), int(op.n_classes)
    shape = op.bed.voxel_column.mass_kg.shape
    return int(shape[2]), int(op.n_classes)


def _binding(event: Any) -> dict:
    op, top = event.op, event.op.topology
    V, K = _bed_dims(op)
    ny, nx = op.shape
    out = {
        "schema": SCHEMA, "profile": _profile(event), "array_backend": op.xp.__name__, "hydraulic_backend": op.backend,
        "shape": [int(ny), int(nx)], "dx_m": _fx(op.dx_m), "n_layers": int(op.n_layers), "n_cells": int(op.n_cells),
        "n_faces": int(op.n_faces), "n_classes": K, "n_voxels": V, "operator_sha256": op.sha256, "event_sha256": event.sha256,
        "layers_sha256": op._component_shas[2], "preferential_sha256": op.preferential.sha256,
        "matrix_sha256": op.matrix.sha256, "graph_input_sha256": str(op.graph.input_sha256),
        "router_control": [_fx(op._control_values[0]), str(op._control_values[1]), str(op._control_values[2])],
        "topology": {"identity_sha256": top.identity_sha256, "source_sha256": top.source_sha256,
                     "topology_sha256": top.topology_sha256, "generation": int(top.generation),
                     "n_closed_components": int(top.n_closed_components)},
        "extra_reserve_bytes": int(op.extra_reserve_bytes), "native_reserve_bytes": int(op.native_reserve_bytes),
        "native_material_sha256": op.native_material_sha256,
        "context_sha256": None if op.context is None else _context_digest(op.context),
        "event_origin_s": _fx(event.event_origin_s), "rainfall_end_s": _fx(event.rainfall_end_s),
        "deadline_s": _fx(event.deadline_s),
    }
    if out["profile"] == WINDOWED_PROFILE:  # Windowed only: the Dense/nonnative binding (and its hash) is unchanged
        out["bed_storage"] = "windowed"
        out["deep_storage"] = _deep_record(op.bed.voxel_column, op.context, op.shape)
    return out


# --- Windowed profile: deep record, native layout, cold-boundary validation ---------------------------------------------------------
def _windowed_modules() -> tuple[Any, Any, Any]:
    """Lazy native Windowed imports, reached only AFTER the profile dispatch and the type-level source guards."""
    from maple.surface.voxels import deep_store, windowed_pending, windowed_prefix

    return deep_store, windowed_pending, windowed_prefix


def _meta_max_chars(n_classes: int) -> int:
    """Characters of the native `metadata` JSON: the provenance cap + the absolute store path + class ids + the fixed keys. The native
    Unicode scalar costs 4 bytes per code point (not the JSON's UTF-8 length)."""
    return NATIVE_PROVENANCE_MAX + STORE_DIR_MAX + (CLASS_ID_MAX + 4) * int(n_classes) + 1024


def _check_deep_record(rec: Any, ctx: Any, shape: tuple, limit: int | None = None) -> dict:
    """Exact typed check of a `deep_storage` record against the native context (no fake event). Pure value arithmetic: no file is
    opened and no deep read happens. Only the FULL/v1 store is supported: overlay/v2 refuse here, before any deep work."""
    from maple_syrup.sediment_bed import BedIntegrationError, check_deep_storage_options

    if not isinstance(rec, dict) or set(rec) != DEEP_RECORD_KEYS:
        _refuse(f"deep_storage record keys mismatch: {sorted(rec) if isinstance(rec, dict) else type(rec).__name__}")
    try:
        check_deep_storage_options(ctx.deep_chunk_encoding, ctx.deep_overlay_max_bytes)
        check_deep_storage_options(rec["deep_chunk_encoding"], rec["deep_overlay_max_bytes"])
    except BedIntegrationError as exc:
        raise LayeredCheckpointError(f"deep-chunk storage option is invalid: {exc}") from None
    if (rec["deep_chunk_encoding"], rec["deep_overlay_max_bytes"]) != (ctx.deep_chunk_encoding, ctx.deep_overlay_max_bytes):
        _refuse("checkpoint deep storage option differs from the native context (no cross-encoding migration)")
    if (rec["deep_chunk_encoding"], rec["deep_overlay_max_bytes"]) != ("full", None):
        _refuse("only the full deep-chunk encoding (store v1) is supported by this profile; overlay is refused before any deep work")
    if type(rec["store_format_version"]) is not int or rec["store_format_version"] != 1:
        _refuse("deep_storage store_format_version must be the int 1 (overlay / v2 is refused before any deep work)")
    for key in ("physical_manifest_digest", "logical_manifest_digest", "header_digest"):
        _sha(rec[key], f"deep_storage.{key}")
    if rec["physical_manifest_digest"] != rec["logical_manifest_digest"]:
        _refuse("a v1 deep version has equal physical and logical manifest digests")
    store_dir = rec["store_dir"]
    if not isinstance(store_dir, str) or not store_dir.startswith("/") or len(store_dir) > STORE_DIR_MAX or "\x00" in store_dir:
        _refuse("deep_storage store_dir must be a bounded absolute path string")
    generation = _int(rec["generation"], "deep_storage.generation")
    n_voxels = _int(rec["n_voxels"], "deep_storage.n_voxels", 1)
    window = _int(rec["window_voxels"], "deep_storage.window_voxels", 1)
    chunk = _int(rec["chunk_columns"], "deep_storage.chunk_columns", 1)
    n_chunks = _int(rec["n_chunks"], "deep_storage.n_chunks", 1)
    _int(rec["packet_terms"], "deep_storage.packet_terms", 1)
    if window > n_voxels or generation < 0:
        _refuse("deep_storage window / voxel counts are out of range")
    n_cells = int(shape[0]) * int(shape[1])
    if n_chunks != -(-n_cells // chunk):
        _refuse("deep_storage chunk count disagrees with the grid and chunk width")
    ids = [g.class_id for g in ctx.grain_classes.classes]
    if rec["class_ids"] != ids or any(not isinstance(c, str) or len(c) > CLASS_ID_MAX for c in ids):
        _refuse("deep_storage class ordering differs from the context's grain classes")
    resident = n_cells * window * len(ids) * 8  # window_kg alone: a strict lower bound
    if limit is not None and resident > limit:
        _refuse(f"deep_storage window geometry needs at least {resident} resident bytes, above the serialized limit {limit} "
                "(refused from the record alone; nothing was read)")
    from maple.core.backend.numpy_pairwise_order import QUALIFIED_MAX_TERMS

    if n_voxels * len(ids) > QUALIFIED_MAX_TERMS:
        _refuse("deep_storage n_voxels x n_classes exceeds the native height-packet bound (refused from the record alone)")
    try:
        terms = _windowed_modules()[2].packet_terms_for(n_voxels, len(ids))
    except Exception as exc:
        raise LayeredCheckpointError(f"the native prefix height-packet support is unavailable: {exc}") from None
    if rec["packet_terms"] != terms:
        _refuse("deep_storage packet_terms disagrees with the native packet size for this column")
    return rec


def _require_published_full(column: Any, ctx: Any, operation: str) -> None:
    """Native-first refusals that precede every deep read: fixed capacity, NO pending deep journal (the codec never publishes), the
    full/v1 store only. Metadata reads only."""
    from maple.surface.voxels.windowed_inventory import require_fixed_windowed_capacity

    _, pending, _ = _windowed_modules()
    try:
        require_fixed_windowed_capacity(ctx.geometry, operation=operation)
        pending.require_published_column(column, operation=operation)
    except (ValueError, TypeError) as exc:
        raise LayeredCheckpointError(f"{exc}; the codec never publishes, canonicalises or discards a pending deep journal: publish the "
                                     "accepted storage state explicitly first. Nothing was written/loaded.") from None
    if int(getattr(column.deep_store, "format_version", 0)) != 1:
        _refuse("the bed's deep store is an overlay (v2) version; only full/v1 is supported (refused before any deep work)")
    if (ctx.deep_chunk_encoding, ctx.deep_overlay_max_bytes) != ("full", None):
        _refuse("the native context selects overlay deep-chunk storage; only full is supported (refused before any deep work)")


def _deep_record(column: Any, ctx: Any, shape: tuple) -> dict:
    """The record of the published deep version the bed is bound to (identities and geometry only; the version itself stays in its
    external immutable store and must be retained)."""
    _require_published_full(column, ctx, "SYRUP layered checkpoint")
    _, _, prefix = _windowed_modules()
    store, header = column.deep_store, column.deep_store.header
    try:
        packets = prefix.require_prefix_packets(column, operation="SYRUP layered checkpoint")
    except (ValueError, TypeError) as exc:
        raise LayeredCheckpointError(f"the windowed bed has no valid height packets: {exc}") from None
    record = {"deep_chunk_encoding": ctx.deep_chunk_encoding, "deep_overlay_max_bytes": ctx.deep_overlay_max_bytes,
              "store_format_version": int(store.format_version), "physical_manifest_digest": str(store.manifest_digest),
              "logical_manifest_digest": str(store.logical_manifest_digest), "header_digest": str(header.digest()),
              "store_dir": str(Path(store.store_dir).resolve()), "generation": int(store.version_id),
              "n_voxels": int(column.n_voxels), "window_voxels": int(header.window_voxels),
              "chunk_columns": int(header.chunk_columns), "n_chunks": int(store.n_chunks),
              "class_ids": [str(c) for c in header.class_ids], "packet_terms": int(packets.packet_terms)}
    return _check_deep_record(record, ctx, shape)


def _windowed_provenance(deep: dict) -> dict:
    return {"bed_storage": "windowed", "physical_manifest_digest": deep["physical_manifest_digest"],
            "logical_manifest_digest": deep["logical_manifest_digest"]}


def _windowed_native_specs(shape: tuple, K: int, deep: dict) -> dict[str, tuple]:
    """Name -> (shape, dtype) of the native snapshot's 11 array members from trusted dimensions, plus its one bounded Unicode scalar."""
    ny, nx = int(shape[0]), int(shape[1])
    w, p = deep["window_voxels"], deep["packet_terms"]
    cell, cc = (ny, nx), (ny, nx, K)
    out: dict[str, tuple] = {"window_kg": ((ny, nx, w, K), "<f8"), "base": (cell, "<i4"), "deep_by_class_kg": (cc, "<f8"),
                             "deep_total_kg": (cell, "<f8"), "deep_room_kg": (cell, "<f8"), "prefix_packet": ((ny, nx, p), "<f8"),
                             "active_mass_kg": (cc, "<f8"), "available_mass_kg": (cc, "<f8"), "bound_mass_kg": (cc, "<f8"),
                             "water_depth_m": (cell, "<f8"), "mobile_mass_kg": (cc, "<f8")}
    out["metadata"] = (None, "string")
    return out


def _native_checks_windowed(bed: Any, ctx: Any, t_s: float, ref_ledger_type: type | None = None) -> None:
    """The genuine native validators of the Windowed bed WITHOUT the cold deep scan: `validate_windowed_partition` (full streamed
    column validation + independent prefix re-encode) is run EXACTLY ONCE per operation by the native writer (save, before any native
    byte is written) or the native loader (load); this function runs the remaining non-scan checks (bed/context support, availability,
    ledger/committed shapes and class, water, zero mobile, pending-ledger reconciliation) and never reads the deep store."""
    from maple.core.backend import to_int
    from maple.core.types.topographic_commit import validate_committed_topography_state
    from maple.coupling.sediment_ledger.validation import validate_ledger_state
    from maple.surface.availability.validation import check_sediment_availability_state
    from maple.surface.topographic_commit.commit import (
        _reconcile_ledger_with_physical_state,
        resolve_reconciliation_reference_inventory_kg,
        validate_reconciliation_context,
    )
    from maple.water import validate_water_state

    from maple_syrup.pond_bed import require_supported_bed

    shape = (ctx.geometry.ny, ctx.geometry.nx, len(ctx.grain_classes.classes))
    try:
        require_supported_bed(bed, ctx, allow_windowed=True)
        _require_published_full(bed.voxel_column, ctx, "SYRUP layered checkpoint")
        check_sediment_availability_state(bed.sediment_availability, bed.active_layer, ctx.mass_resolution_kg)
        if validate_ledger_state(bed.ledger) != shape or validate_committed_topography_state(bed.committed_topography) != shape:
            _refuse("native ledger / committed topography geometry mismatch")
        if ref_ledger_type is not None and type(bed.ledger) is not ref_ledger_type:
            _refuse("native ledger class differs from the reference bed's (Dense vs Compact is restart-relevant)")
        validate_water_state(bed.water, *shape)
        if to_int(np.count_nonzero(bed.water.mobile_mass_by_cell_class_kg)) != 0:
            _refuse("the native bed carries mobile sediment; this water-only profile needs none")
        validate_reconciliation_context(
            bed.ledger, bed.voxel_column, bed.active_layer, ctx.geometry, committed_state=bed.committed_topography,
            reconciliation_baseline=None, current_time_s=t_s, operation="SYRUP layered checkpoint", ledger_shape=shape)
        reference, label = resolve_reconciliation_reference_inventory_kg(bed.committed_topography, None)
        _reconcile_ledger_with_physical_state(bed.ledger, reference, bed.voxel_column, bed.active_layer,
                                              ctx.mass_resolution_kg, reference_label=label, ledger_shape=shape)
    except LayeredCheckpointError:
        raise
    except Exception as exc:
        raise LayeredCheckpointError(f"native MAPLE Windowed bed validation failed: {type(exc).__name__}: {exc}") from None


def _digest_windowed_column(h: Any, column: Any) -> None:
    """Resident arrays (window, base, deep metadata, packets) and the deep identity; the external deep files are bound by their digests."""
    for name in ("window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg"):
        _digest_value(h, getattr(column, name))
    _digest_value(h, column.height_packet.packet)
    store = column.deep_store
    for text in (str(store.manifest_digest), str(store.logical_manifest_digest), str(store.header.digest()),
                 str(Path(store.store_dir).resolve())):
        _digest_value(h, text)
    _digest_value(h, int(column.generation))
    _digest_value(h, int(column.n_voxels))


# --- external deep-store metadata preflight (Windowed profile; correction01) ---------------------------------------------------------
EXT_CHUNK_REF_BYTES = 6 * NODE_BYTES   # one chunk-reference / fingerprint object: the instance + 5 fields, each <= NODE_BYTES (same convention)
EXT_STORE_MAX_BOUND = 2**63 - 1        # schema bound of store.json `max_store_bytes` (a positive JSON integer)
EXT_HASH_BYTES = IO_CHUNK_BYTES


def _canon(obj: Any) -> bytes:
    """The native canonical JSON dump (`deep_store._canonical_json_bytes`): sorted keys, compact separators, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _expected_version_manifest(store: Any) -> bytes:
    """The EXACT bytes of the reference handle's v1 version manifest, rebuilt from its public attributes (native `_v1_manifest_object`)
    and required to hash to the handle's own digest: if they do not, the format is not the supported one and the preflight refuses."""
    from maple.surface.voxels.deep_store import DEEP_STORE_FORMAT

    header = store.header
    obj = {"format": DEEP_STORE_FORMAT, "kind": "version", "header": header.to_json_dict(), "header_digest": header.digest(),
           "version_id": int(store.version_id), "parent_version_id": store.parent_version_id,
           "parent_manifest_digest": store.parent_manifest_digest, "journal_digest": store.journal_digest,
           "chunks": [ref.to_json_dict() for ref in store.chunks]}
    data = _canon(obj)
    if hashlib.sha256(data).hexdigest() != store.manifest_digest:
        raise LayeredCheckpointDeepStoreError("the reference handle's version manifest cannot be reproduced from its public attributes "
                                              "(unsupported deep-store format); refused before any external read")
    return data


def _store_object(header: Any, max_store_bytes: int) -> dict:
    from maple.surface.voxels.deep_store import DEEP_STORE_FORMAT

    return {"format": DEEP_STORE_FORMAT, "kind": "store", "header": header.to_json_dict(), "header_digest": header.digest(),
            "max_store_bytes": int(max_store_bytes)}


def external_metadata_bounds(header: Any, n_voxels: int, n_classes: int, store: Any) -> dict:
    """Analytic caps from the exact reference header / class ids / chunk count / v1 schema (no allocation, no file access): the manifest byte
    cap `manifest_bytes` (empty-table manifest + per-entry template with the digit widths of cells, chunks, chunk columns and chunk bytes),
    the store.json cap, the chunk count, and the JSON node count of the manifest (`{ [ , :` outside strings, as the codec's scanner counts)."""
    n = int(header.n_chunks)
    chunk_nominal = int(header.chunk_bytes_max)
    widths = (len(str(int(header.n_cells))), len(str(max(n - 1, 0))), len(str(int(header.chunk_columns))), len(str(chunk_nominal)))
    entry = len(_canon({"index": 0, "cell_start": 0, "n_cells": 0, "nbytes": 0, "sha256": "0" * 64})) - 4 + sum(widths)
    empty = {"format": "x" * 0, "kind": "version", "header": header.to_json_dict(), "header_digest": header.digest(),
             "version_id": int(store.version_id), "parent_version_id": store.parent_version_id,
             "parent_manifest_digest": store.parent_manifest_digest, "journal_digest": store.journal_digest, "chunks": []}
    from maple.surface.voxels.deep_store import DEEP_STORE_FORMAT

    empty["format"] = DEEP_STORE_FORMAT
    manifest = len(_canon(empty)) + n * entry + max(n - 1, 0)
    store_bytes = len(_canon(_store_object(header, EXT_STORE_MAX_BOUND)))
    from maple.surface.voxels import deep_store as native_store

    block = getattr(native_store, "_HASH_BLOCK_BYTES", None)   # native `_sha256_file` reads this many bytes per call (a real allocation)
    if type(block) is not int or block < 1:
        raise LayeredCheckpointDeepStoreError("the native chunk hash block size cannot be read from the bound deep_store module")
    return {"n_chunks": n, "manifest_bytes": manifest, "store_bytes": store_bytes, "entry_bytes": entry, "hash_block": block,
            "manifest_nodes": 11 * n + int(n_classes) + 200, "store_nodes": 2 * (int(n_classes) + 40)}


def external_metadata_transient(b: dict) -> dict:
    """Concurrent lifetimes (see correction01_design.md): `live` (whole call), `preflight` and `native` stages."""
    n, m, st = b["n_chunks"], b["manifest_bytes"], b["store_bytes"]
    store_tree = b["store_nodes"] * NODE_BYTES
    preflight = max(2 * m + n * EXT_CHUNK_REF_BYTES, 2 * st + store_tree) + EXT_HASH_BYTES
    # native `load_version`: manifest bytes + decoded str + json tree + new chunk refs + store.json bytes/str/tree, all still alive while the
    # chunk loop runs `_sha256_file` (one `hash_block` read buffer, allocated at full size even for a small file; found by a warm trace)
    native = 2 * m + b["manifest_nodes"] * NODE_BYTES + n * EXT_CHUNK_REF_BYTES + 2 * st + store_tree + b["hash_block"]
    live_save = 2 * n * EXT_CHUNK_REF_BYTES            # the reference's chunk references + the fingerprint table
    live_load = 3 * n * EXT_CHUNK_REF_BYTES            # + the new handle's chunk references
    return {"preflight": preflight, "native": native, "live_save": live_save, "live_load": live_load,
            "save": live_save + preflight, "load": live_load + max(preflight, native)}


def _open_dir(parent_fd: int | None, name: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(name, flags, dir_fd=parent_fd) if parent_fd is not None else os.open(name, flags)
    except OSError as exc:
        raise LayeredCheckpointDeepStoreError(f"deep-store directory {name!r} is missing or not a plain directory: {exc}") from None


def _fingerprint(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_store_file(dirfd: int, name: str, what: str, *, max_bytes: int | None = None, exact_bytes: int | None = None) -> tuple[int, tuple]:
    """Open a plain regular store file under a held directory descriptor (no symlink, no FIFO) and check its SIZE by `fstat` BEFORE any read."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(name, flags, dir_fd=dirfd)
    except OSError as exc:
        raise LayeredCheckpointDeepStoreError(f"external deep-store {what} is missing or unreadable: {exc}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise LayeredCheckpointDeepStoreError(f"external deep-store {what} is not a plain regular file")
        if exact_bytes is not None and info.st_size != exact_bytes:
            raise LayeredCheckpointDeepStoreError(
                f"external deep-store {what} has {info.st_size} bytes but the reference schema requires exactly {exact_bytes} "
                "(refused before any read)")
        if max_bytes is not None and info.st_size > max_bytes:
            raise LayeredCheckpointDeepStoreError(
                f"external deep-store {what} has {info.st_size} bytes, above the derived bound {max_bytes} (refused before any read)")
    except BaseException:
        os.close(fd)
        raise
    return fd, _fingerprint(info)


def _read_exact(fd: int, nbytes: int) -> bytes | None:
    """Read at most `nbytes` + 1 bytes (the +1 detects growth); returns None when the content is not exactly `nbytes`."""
    chunks, total = [], 0
    while total <= nbytes:
        piece = os.read(fd, min(IO_CHUNK_BYTES, nbytes + 1 - total))
        if not piece:
            break
        chunks.append(piece)
        total += len(piece)
    data = b"".join(chunks)
    return data if len(data) == nbytes else None


def preflight_external_deep(column: Any, ctx: Any, n_classes: int) -> dict[str, tuple]:
    """Bounded, fresh verification of the ACTUAL external files the native loader will reopen, whatever the reference handle has cached:
    the version manifest (exact bytes), `store.json` (strict parse under its derived cap) and every referenced chunk file (plain file, exact
    size, streamed digest). Returns the fingerprints for `recheck_external_deep`. Nothing is written; native files are never modified."""
    store = column.deep_store
    header = store.header
    bounds = external_metadata_bounds(header, int(column.n_voxels), n_classes, store)
    root = Path(store.store_dir)
    expected = _expected_version_manifest(store)
    if len(expected) > bounds["manifest_bytes"]:
        raise LayeredCheckpointDeepStoreError("the reference manifest exceeds its derived analytic cap (schema mismatch)")
    prints: dict[str, tuple] = {}
    sfd = _open_dir(None, str(root))
    try:
        vfd = _open_dir(sfd, "versions")
        try:
            fd, fp = _open_store_file(vfd, f"{store.manifest_digest}.json", "version manifest", exact_bytes=len(expected))
            try:
                data = _read_exact(fd, len(expected))
            finally:
                os.close(fd)
            if data != expected:
                raise LayeredCheckpointDeepStoreError("the external version manifest bytes differ from the reference version's exact manifest")
            del data
            prints["versions/" + f"{store.manifest_digest}.json"] = fp
        finally:
            os.close(vfd)
        del expected
        fd, fp = _open_store_file(sfd, "store.json", "store descriptor", max_bytes=bounds["store_bytes"])
        try:
            raw = _read_exact(fd, os.fstat(fd).st_size)
        finally:
            os.close(fd)
        if raw is None:
            raise LayeredCheckpointDeepStoreError("the external store descriptor changed while it was read")
        _check_store_descriptor(raw, header, bounds)
        prints["store.json"] = fp
        cfd = _open_dir(sfd, "chunks")
        try:
            for ref in store.chunks:
                name = Path(store.chunk_path(ref)).name
                fd, fp = _open_store_file(cfd, name, f"chunk {ref.index} file", exact_bytes=int(ref.nbytes))
                try:
                    h, left = hashlib.sha256(), int(ref.nbytes)
                    while left > 0:
                        piece = os.read(fd, min(EXT_HASH_BYTES, left))
                        if not piece:
                            break
                        h.update(piece)
                        left -= len(piece)
                    if left != 0 or os.read(fd, 1) or h.hexdigest() != ref.sha256:
                        raise LayeredCheckpointDeepStoreError(f"external chunk {ref.index} file does not hash to its recorded digest")
                finally:
                    os.close(fd)
                prints["chunks/" + name] = fp
        finally:
            os.close(cfd)
    finally:
        os.close(sfd)
    return prints


def _check_store_descriptor(raw: bytes, header: Any, bounds: dict) -> None:
    """Structural scan BEFORE `json.loads`, then strict closed-schema comparison with the reference header."""
    def no_duplicates(pairs: list) -> dict:  # no last-key-wins: a duplicate member at ANY depth is a refusal (bounded message)
        out: dict = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"duplicate member {str(k)[:64]!r}")
            out[k] = v
        return out

    try:
        _scan_json(raw, 8, bounds["store_nodes"], len(raw) + 1)
        obj = json.loads(raw.decode("ascii"), object_pairs_hook=no_duplicates,
                         parse_constant=lambda t: (_ for _ in ()).throw(ValueError(f"non-finite constant {t[:16]}")))
    except (LayeredCheckpointError, ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise LayeredCheckpointDeepStoreError(f"the external store descriptor is malformed: {str(exc)[:200]}") from None
    cap = obj.get("max_store_bytes") if isinstance(obj, dict) else None
    if (not isinstance(obj, dict) or set(obj) != {"format", "kind", "header", "header_digest", "max_store_bytes"}
            or type(cap) is not int or not 0 < cap <= EXT_STORE_MAX_BOUND
            or _canon(obj) != _canon(_store_object(header, cap))):
        raise LayeredCheckpointDeepStoreError("the external store descriptor does not match the reference header (identity, class table, geometry)")


def recheck_external_deep(column: Any, prints: dict[str, tuple]) -> None:
    """Refuse a DETECTABLE change of any preflighted external file (identity, size, mtimes) after the native operation. Not race-proof."""
    root = Path(column.deep_store.store_dir)
    sfd = _open_dir(None, str(root))
    try:
        for rel, want in prints.items():
            parts = rel.split("/")
            fd = sfd if len(parts) == 1 else _open_dir(sfd, parts[0])
            try:
                f, got = _open_store_file(fd, parts[-1], rel)
                os.close(f)
            finally:
                if fd != sfd:
                    os.close(fd)
            if got != want:
                raise LayeredCheckpointDeepStoreError(f"the external deep-store file {rel} changed identity/size/time during the operation")
    finally:
        os.close(sfd)



# --- budget -------------------------------------------------------------------------------------------------------------------------
def _leaf_specs(value: Any) -> list[tuple[str, tuple[int, ...]]]:
    """(dtype.str, shape) of every array leaf of a native dataclass record, in field order."""
    out: list = []
    if isinstance(value, np.ndarray):
        out.append((value.dtype.str, tuple(int(n) for n in value.shape)))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            out.extend(_leaf_specs(getattr(value, f.name)))
    return out


def _spec_bytes(specs: list) -> int:
    return sum(int(np.dtype(d).itemsize) * math.prod(s) for d, s in specs)


def _companion_specs(op: Any) -> dict[str, tuple[tuple[int, ...], str]]:
    ny, nx = op.shape
    L, N = op.n_layers, op.n_cells
    specs = {"soil_water_m": ((L, N), "<f8"), "wetting_m": ((N,), "<f8"), "cumulative_intake_m": ((N,), "<f8"),
             "qx_m2_s": ((ny, nx + 1), "<f8"), "qy_m2_s": ((ny + 1, nx), "<f8"), "topo_source_full": ((ny + 2, nx + 2), "<f8")}
    if op.bed is None:
        specs["surface_m"] = ((ny, nx), "<f8")
    return specs


def _native_snapshot_specs(op: Any) -> dict[str, tuple]:
    ny, nx = op.shape
    V, K = _bed_dims(op)
    s: dict[str, tuple] = {"voxel_mass_kg": ((ny, nx, V, K), "<f8"), "active_mass_kg": ((ny, nx, K), "<f8"),
                            "time_s": ((), "<f8"), "seed": ((), "<i8"), "step": ((), "<i8"),
                            "snapshot_schema_version": ((), "<i8"),
                            "sediment_availability_available_mass_kg": ((ny, nx, K), "<f8"),
                            "sediment_availability_bound_mass_kg": ((ny, nx, K), "<f8"),
                            "water_depth_m": ((ny, nx), "<f8"), "water_mobile_mass_by_cell_class_kg": ((ny, nx, K), "<f8")}
    for name in _NATIVE_BOOL_MEMBERS:
        s[name] = ((), "|b1")
    for name in _NATIVE_STRING_MEMBERS:
        s[name] = ((), "string")
    return s


def _manifest_dims(n_entries: int) -> tuple[int, int]:
    """(byte cap, node cap) of a manifest with `n_entries` companion array entries (fixed arrays + native leaves)."""
    return MANIFEST_FIXED_BYTES + MANIFEST_PER_ENTRY_BYTES * n_entries, MANIFEST_FIXED_NODES + MANIFEST_NODES_PER_ENTRY * n_entries


def _metadata_bytes(n_entries: int, n_members: int) -> int:
    """Concurrent lifetimes of the declared metadata policy: manifest raw bytes + its ASCII decode + the string payload held by the tree +
    the tree nodes (`NODE_BYTES` each) = 3 cap + nodes x NODE_BYTES (the same arithmetic bounds the save-side dict + text + sealed text);
    identity work = 4 x the escaped identity cap + its nodes; zip central directory (EOCD-bounded) = members x (cd entry + ZipInfo
    object); native strings = 3 provenance copies + 3 x 6 small strings."""
    cap, nodes = _manifest_dims(n_entries)
    manifest = 3 * cap + nodes * NODE_BYTES
    identity = 4 * MAX_IDENTITY_BYTES + IDENTITY_MAX_NODES * NODE_BYTES
    zip_meta = n_members * (ZIP_CD_ENTRY_MAX + ZIP_INFO_OBJECT_BYTES) + 4096
    strings = 3 * NATIVE_PROVENANCE_MAX + 18 * NATIVE_STRING_MAX
    return manifest + identity + zip_meta + strings


def estimate_checkpoint_budget(event: Any) -> CheckpointBudget:
    """Pure arithmetic from the actual reference arrays and the declared metadata policy; no allocation. Every term, its native source
    and its lifetime are in `BUDGET_TERM_SOURCES`; see `CheckpointBudget` for the whole-owner quota semantics."""
    ev._check_event(event)
    op = event.op
    if op.bed is not None and is_windowed_bed(op.bed):
        return estimate_windowed_checkpoint_budget(event)
    N, L = op.n_cells, op.n_layers
    V, K = _bed_dims(op)
    comp_specs = _companion_specs(op)
    companion = sum(int(np.dtype(d).itemsize) * math.prod(s) for s, d in comp_specs.values())
    n_fixed = len(comp_specs)
    members_comp = n_fixed
    ledger = committed = native = 0
    members_native = 0
    leaf_bytes = [int(np.dtype(d).itemsize) * math.prod(s) for s, d in comp_specs.values()]
    native_arrays: dict = {}
    voxel_scan = class_scan = ledger_scan = resident_native = 0
    ledger_leaves: list = []
    if op.bed is not None:
        ledger_specs, committed_specs = _leaf_specs(op.bed.ledger), _leaf_specs(op.bed.committed_topography)
        ledger, committed = _spec_bytes(ledger_specs), _spec_bytes(committed_specs)
        companion += ledger + committed
        members_comp += len(ledger_specs) + len(committed_specs)
        ledger_leaves = [int(np.dtype(d).itemsize) * math.prod(s) for d, s in ledger_specs + committed_specs]
        leaf_bytes += ledger_leaves
        native_arrays = {k: int(np.dtype(d).itemsize) * math.prod(s) for k, (s, d) in _native_snapshot_specs(op).items()
                         if d != "string"}
        native = sum(native_arrays.values())
        members_native = len(_native_snapshot_specs(op))
        leaf_bytes += list(native_arrays.values())
        voxel_scan = VOXEL_SCAN_MASKS * N * V * K + VOXEL_SCAN_NV_TEMPORARIES * 8 * N * V
        class_scan = CLASS_SCAN_NK_TEMPORARIES * 8 * N * K
        ledger_scan = LEDGER_LEAF_TEMPORARIES * max(ledger_leaves + [0])
        # actual resident native leaves; the water record (depth + mobile) aliases the owner's counted surface/mobile
        bed, ctx = op.bed, op.context
        resident_native = (pc._tree_nbytes(bed.voxel_column) + pc._tree_nbytes(bed.active_layer)
                           + pc._tree_nbytes(bed.sediment_availability) + ledger + committed
                           + int(np.asarray(ctx.initial_available_fraction).nbytes))
    owner_audit = OWNER_AUDIT_LN_TEMPORARIES * 8 * L * N + 8 * 8 * N + 4 * (N + op.n_faces)
    validation = voxel_scan + class_scan + ledger_scan + owner_audit
    largest = max(leaf_bytes + [0])
    n_entries = n_fixed + (len(ledger_leaves) if op.bed is not None else 0)
    manifest_cap, manifest_nodes = _manifest_dims(n_entries)
    metadata = _metadata_bytes(n_entries, max(members_comp, members_native))
    serialized = (companion + native + (members_comp + members_native) * (NPY_HEADER_ALLOWANCE + ZIP_MEMBER_ALLOWANCE)
                  + PROVENANCE_ALLOWANCE + manifest_cap)
    write_chunks = 2 * min(NPY_WRITE_CHUNK_BYTES, largest)
    io = IO_CHUNK_BYTES + HASH_CHUNK_BYTES
    # --- resident (whole-owner) ---------------------------------------------------------------------------------------------------------
    resident_event = event.required_bytes - op.memory.required_bytes
    resident = op.memory.resident_bytes + resident_native + resident_event
    # --- save transient: validation first, then capture + (native writer validation | write chunks) + manifest build + hashing --------------
    save_transient = max(validation, companion + max(voxel_scan + class_scan + ledger_scan, write_chunks) + metadata + io)
    save_scratch = max(validation, max(voxel_scan + class_scan + ledger_scan, write_chunks) + metadata + io)
    # --- load transient ----------------------------------------------------------------------------------------------------------------------
    # `load_state_snapshot`: NpzFile + ZipFile objects (one ZipInfo per member), decoded string members (each read as bytes then str,
    # provenance is the only large one) in addition to the arrays it keeps
    native_objects = members_native * (ZIP_INFO_OBJECT_BYTES + 3 * NATIVE_STRING_MAX) + 3 * NATIVE_PROVENANCE_MAX
    native_load = native + (native_arrays.get("active_mass_kg", 0) if op.bed is not None else 0) + 2 * NPY_READ_BUFFER_BYTES \
        + ((N * K + N + native_objects) if op.bed is not None else 0)
    decode = native + companion + largest + 2 * NPY_READ_BUFFER_BYTES
    new_state = (L * N + 2 * N) * 8 + (0 if op.bed is not None else N * 8)
    new_static = op.memory.static_bytes
    # The parsed manifest tree (scalars, array table, native encoding) is RETAINED by `_load` while `_restore` runs: its nodes x NODE_BYTES plus
    # the string payload (<= the byte cap); raw bytes and the ASCII decode are already released.
    retained_metadata = manifest_nodes * NODE_BYTES + manifest_cap
    restore = (native + companion + new_state + new_static + resident_event + retained_metadata
               + max(validation, op.memory.build_bytes + HASH_CHUNK_BYTES))
    load_transient = max(metadata + io, native_load + metadata, decode + metadata, restore)
    load_scratch = max(metadata + io, validation + HASH_CHUNK_BYTES + 2 * NPY_READ_BUFFER_BYTES + metadata,
                       op.memory.build_bytes + HASH_CHUNK_BYTES + retained_metadata)
    scratch = max(save_scratch, load_scratch)
    terms = {"resident_owner": op.memory.resident_bytes, "resident_native": resident_native, "resident_event": resident_event,
             "voxel_scan": voxel_scan, "class_scan": class_scan, "ledger_scan": ledger_scan, "owner_audit": owner_audit,
             "capture": companion, "writer_chunks": write_chunks, "native_load": native_load, "decode": decode,
             "restore": restore, "metadata": metadata, "retained_metadata": retained_metadata}
    return CheckpointBudget(_profile(event), N, L, K, V, native, ledger, committed, companion, members_native, members_comp,
                            manifest_cap, manifest_nodes, serialized, metadata, validation, scratch, resident, save_transient,
                            load_transient, resident + save_transient, resident + load_transient, 0, serialized, terms)


#: Windowed cold-scan scratch is counted from native39 source (see design.md section 6); these are COUNTS of arrays, not fit multipliers.
WINDOWED_FOLD_CHUNK_COPIES = 2   # `_metadata_from_columns`: prefix-by-class fold + room fold over the (C,V,K) block (assumed worst case)
WINDOWED_PREFIX_BUFFERS = 3      # prefix re-encode: expected + actual packet slice + encoder output, each (C,P)
BUDGET_TERM_SOURCES.update({
    "windowed_validate_scan": "windowed_inventory.validate_windowed_column_full: one full (C,V,K) chunk + window rows + the dense voxel validator's masks/temporaries + the "
                              "window copy and fold buffers of _metadata_from_columns; one chunk at a time",
    "windowed_prefix_scan": "windowed_snapshot.validate_windowed_partition: gather_columns (C,V,K) with the PREVIOUS block still bound while the next is "
                            "gathered (2 blocks) + window rows + packet buffers; sequential with the validate scan",
    "windowed_mapping": "DeepStoreVersion.mapped_bytes_bound of the reference handle (+ DEFAULT_MAPPING_CACHE_CHUNKS x chunk_bytes_max for the handle the native loader "
                        "opens; the loader accepts no cap argument): file-backed mappings counted conservatively as host, NOT Python heap/RSS/page cache",
})


def estimate_windowed_checkpoint_budget(event: Any) -> CheckpointBudget:
    """Pure arithmetic from the ACTUAL reference arrays, the actual deep-store header and the declared metadata policy (no allocation, no
    deep read). The native resident snapshot is `8NWK + 40NK + 28N + 8NP` bytes; the external deep files are NOT part of it (they exist
    already) but validating them costs bounded chunk scratch and mappings, counted below. See `CheckpointBudget` for the quota semantics."""
    ev._check_event(event)
    op = event.op
    if op.bed is None or not is_windowed_bed(op.bed):
        _refuse("the Windowed budget needs an actual native Windowed bed")
    deep_store_mod, _, _ = _windowed_modules()
    N, L = op.n_cells, op.n_layers
    V, K = _bed_dims(op)
    bed, ctx = op.bed, op.context
    column = bed.voxel_column
    header = column.deep_store.header
    W, P = int(column.window_kg.shape[2]), int(column.height_packet.packet_terms)
    C_nominal, chunk_nominal = int(header.chunk_columns), int(header.chunk_bytes_max)
    C = min(C_nominal, N)                    # the LARGEST ACTUAL chunk holds min(chunk_columns, cells) columns (scan scratch)
    chunk_bytes = C * V * K * 8              # == header.chunk_bytes_max when the grid fills a chunk
    deep = {"window_voxels": W, "packet_terms": P}
    native_specs = _windowed_native_specs(op.shape, K, deep)
    native_arrays = {k: int(np.dtype(d).itemsize) * math.prod(sh) for k, (sh, d) in native_specs.items() if d != "string"}
    meta_chars = _meta_max_chars(K)
    native = sum(native_arrays.values())
    comp_specs = _companion_specs(op)
    ledger_specs, committed_specs = _leaf_specs(bed.ledger), _leaf_specs(bed.committed_topography)
    ledger, committed = _spec_bytes(ledger_specs), _spec_bytes(committed_specs)
    companion = sum(int(np.dtype(d).itemsize) * math.prod(sh) for sh, d in comp_specs.values()) + ledger + committed
    n_fixed = len(comp_specs)
    members_comp = n_fixed + len(ledger_specs) + len(committed_specs)
    members_native = len(native_specs)
    ledger_leaves = [int(np.dtype(d).itemsize) * math.prod(sh) for d, sh in ledger_specs + committed_specs]
    leaf_bytes = [int(np.dtype(d).itemsize) * math.prod(sh) for sh, d in comp_specs.values()] + ledger_leaves + list(native_arrays.values())
    # actual resident native leaves; the water record (depth + mobile) aliases the owner's counted surface/mobile
    resident_native = (sum(int(getattr(column, n).nbytes) for n in ("window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg"))
                       + int(column.height_packet.nbytes) + pc._tree_nbytes(bed.active_layer) + pc._tree_nbytes(bed.sediment_availability)
                       + ledger + committed + int(np.asarray(ctx.initial_available_fraction).nbytes))
    # --- cold boundary scan scratch (sequential: validate pass then prefix pass) ----------------------------------------------------------
    window_rows = 8 * C * W * K
    validate_scan = (chunk_bytes + window_rows + VOXEL_SCAN_MASKS * C * V * K + VOXEL_SCAN_NV_TEMPORARIES * 8 * C * V + window_rows
                     + WINDOWED_FOLD_CHUNK_COPIES * chunk_bytes)
    prefix_scan = 2 * chunk_bytes + window_rows + WINDOWED_PREFIX_BUFFERS * 8 * C * P
    cold_scan = max(validate_scan, prefix_scan) + 8 * 8 * N       # + the partition's per-cell reductions
    class_scan = CLASS_SCAN_NK_TEMPORARIES * 8 * N * K
    ledger_scan = LEDGER_LEAF_TEMPORARIES * max(ledger_leaves + [0])
    owner_audit = OWNER_AUDIT_LN_TEMPORARIES * 8 * L * N + 8 * 8 * N + 4 * (N + op.n_faces)
    validation = cold_scan + class_scan + ledger_scan + owner_audit
    old_map = int(column.deep_store.mapped_bytes_bound)
    new_map = int(deep_store_mod.DEFAULT_MAPPING_CACHE_CHUNKS) * chunk_nominal    # native's own nominal bound (cap x header chunk bytes)
    largest = max(leaf_bytes + [0])
    n_entries = n_fixed + len(ledger_leaves)
    manifest_cap, manifest_nodes = _manifest_dims(n_entries)
    manifest_cap += STORE_DIR_MAX + (CLASS_ID_MAX + 4) * K + 2048    # the deep_storage record and its binding copy
    manifest_nodes += 4 * K + 64
    metadata = _metadata_bytes(n_entries, max(members_comp, members_native)) + 3 * 4 * meta_chars + 2 * (STORE_DIR_MAX + (CLASS_ID_MAX + 4) * K)
    serialized = (companion + native + 4 * meta_chars + (members_comp + members_native) * (NPY_HEADER_ALLOWANCE + ZIP_MEMBER_ALLOWANCE)
                  + PROVENANCE_ALLOWANCE + manifest_cap)
    write_chunks = 2 * min(NPY_WRITE_CHUNK_BYTES, largest)
    io = IO_CHUNK_BYTES + HASH_CHUNK_BYTES
    resident_event = event.required_bytes - op.memory.required_bytes
    resident = op.memory.resident_bytes + resident_native + resident_event
    ext_bounds = external_metadata_bounds(header, V, K, column.deep_store)
    ext = external_metadata_transient(ext_bounds)
    save_transient = max(validation, companion + max(cold_scan + class_scan + ledger_scan, write_chunks) + metadata + io) + old_map + ext["save"]
    save_scratch = max(validation, max(cold_scan + class_scan + ledger_scan, write_chunks) + metadata + io) + ext["save"]
    # load: the native loader copies every array it reads (all kept), reads `metadata` as U (4 B/char) then str + JSON, opens the NEW handle
    # (default cap), then validates (one cold scan) while the arrays and both handles' mappings are alive
    native_objects = members_native * (ZIP_INFO_OBJECT_BYTES + 3 * NATIVE_STRING_MAX) + 3 * NATIVE_PROVENANCE_MAX
    native_load = native + 2 * NPY_READ_BUFFER_BYTES + 3 * 4 * meta_chars + native_objects + cold_scan + class_scan + (N * K + N)
    decode = native + companion + largest + 2 * NPY_READ_BUFFER_BYTES
    new_state = (L * N + 2 * N) * 8
    retained_metadata = manifest_nodes * NODE_BYTES + manifest_cap
    restore = (native + companion + new_state + op.memory.static_bytes + resident_event + retained_metadata
               + max(validation - cold_scan, op.memory.build_bytes + HASH_CHUNK_BYTES))
    load_transient = max(metadata + io, native_load + metadata, decode + metadata, restore) + old_map + new_map + ext["load"]
    load_scratch = max(metadata + io, validation + HASH_CHUNK_BYTES + 2 * NPY_READ_BUFFER_BYTES + metadata,
                       op.memory.build_bytes + HASH_CHUNK_BYTES + retained_metadata) + ext["load"]
    scratch = max(save_scratch, load_scratch)
    terms = {"resident_owner": op.memory.resident_bytes, "resident_native": resident_native, "resident_event": resident_event,
             "windowed_validate_scan": validate_scan, "windowed_prefix_scan": prefix_scan, "class_scan": class_scan,
             "ledger_scan": ledger_scan, "owner_audit": owner_audit, "capture": companion, "writer_chunks": write_chunks,
             "native_load": native_load, "decode": decode, "restore": restore, "metadata": metadata,
             "retained_metadata": retained_metadata, "windowed_mapping": old_map + new_map, "windowed_mapping_old": old_map,
             "windowed_mapping_new": new_map, "chunk_columns": C_nominal, "chunk_columns_actual": C, "chunk_bytes_max": chunk_nominal, "chunk_bytes_actual": chunk_bytes,
             "window_voxels": W,
             "packet_terms": P, "external_n_chunks": ext_bounds["n_chunks"], "external_manifest_bytes": ext_bounds["manifest_bytes"],
             "external_store_bytes": ext_bounds["store_bytes"], "external_preflight": ext["preflight"], "external_native": ext["native"], "external_hash_block": ext_bounds["hash_block"],
             "external_save": ext["save"], "external_load": ext["load"]}
    return CheckpointBudget(WINDOWED_PROFILE, N, L, K, V, native, ledger, committed, companion, members_native, members_comp,
                            manifest_cap, manifest_nodes, serialized, metadata, validation, scratch, resident, save_transient,
                            load_transient, resident + save_transient, resident + load_transient, 0, serialized, terms)


def _admit(need: int, limit: int, available_fn: Callable[[], int] | None, what: str) -> None:
    if need > limit:
        raise LayeredCheckpointQuotaError(f"{what} needs {need} B but the caller's limit is {limit} B; nothing was allocated/written")
    if available_fn is not None:
        available = available_fn()
        if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
            _refuse(f"the availability reader for {what} must return a non-Boolean integer >= 0")
        if need > int(available):
            raise LayeredCheckpointQuotaError(f"{what} needs {need} B but only {int(available)} B are reported available; "
                                              "nothing was allocated/written")


def _admit_all(budget: CheckpointBudget, limits: CheckpointLimits, *, host_need: int, host_fn: Any, device_fn: Any,
               disk_fn: Callable[[], int] | None, disk_need: int, op_backend: str) -> None:
    if not callable(host_fn):
        _refuse("host_available_bytes_fn is required and must be callable")
    if device_fn is not None:
        _refuse(f"device_available_bytes_fn is for a CUDA profile; backend {op_backend!r} has no device quota (not silently ignored)")
    _admit(budget.serialized_bytes, limits.max_serialized_bytes, None, "serialized checkpoint")
    _admit(host_need, limits.max_host_bytes, host_fn, "checkpoint host peak")
    _admit(budget.device_bytes, limits.max_device_bytes, None, "checkpoint device peak")
    _admit(budget.scratch_bytes, limits.max_scratch_bytes, None, "checkpoint transient scratch")
    if disk_fn is not None:
        _admit(disk_need, 2**62, disk_fn, "checkpoint disk")


# --- native + physical validation (the same function save and load run) -------------------------------------------------------------
def _native_checks(bed: Any, ctx: Any, t_s: float, ref_ledger_type: type | None = None) -> None:
    """Genuine native validators, as `pond_checkpoint._native_checks` calls them, on a bed with an explicit context (no fake event).
    Reconciliation here VALIDATES the pending ledger against the physical bed; it never commits."""
    from maple.core.backend import to_int
    from maple.core.types.topographic_commit import validate_committed_topography_state
    from maple.coupling.sediment_ledger.validation import validate_ledger_state
    from maple.surface.active_layer.validation import check_active_layer_voxel_partition
    from maple.surface.availability.validation import check_sediment_availability_state
    from maple.surface.topographic_commit.commit import (
        _reconcile_ledger_with_physical_state,
        resolve_reconciliation_reference_inventory_kg,
        validate_reconciliation_context,
    )
    from maple.water import validate_water_state

    from maple_syrup.pond_bed import require_supported_bed

    shape = (ctx.geometry.ny, ctx.geometry.nx, len(ctx.grain_classes.classes))
    try:
        require_supported_bed(bed, ctx, allow_windowed=False)
        check_active_layer_voxel_partition(bed.active_layer, bed.voxel_column, ctx.geometry, ctx.mass_resolution_kg)
        check_sediment_availability_state(bed.sediment_availability, bed.active_layer, ctx.mass_resolution_kg)
        if validate_ledger_state(bed.ledger) != shape or validate_committed_topography_state(bed.committed_topography) != shape:
            _refuse("native ledger / committed topography geometry mismatch")
        if ref_ledger_type is not None and type(bed.ledger) is not ref_ledger_type:
            _refuse("native ledger class differs from the reference bed's (Dense vs Compact is restart-relevant)")
        validate_water_state(bed.water, *shape)
        if to_int(np.count_nonzero(bed.water.mobile_mass_by_cell_class_kg)) != 0:
            _refuse("the native bed carries mobile sediment; this water-only profile needs none")
        validate_reconciliation_context(
            bed.ledger, bed.voxel_column, bed.active_layer, ctx.geometry, committed_state=bed.committed_topography,
            reconciliation_baseline=None, current_time_s=t_s, operation="SYRUP layered checkpoint", ledger_shape=shape)
        reference, label = resolve_reconciliation_reference_inventory_kg(bed.committed_topography, None)
        _reconcile_ledger_with_physical_state(bed.ledger, reference, bed.voxel_column, bed.active_layer,
                                              ctx.mass_resolution_kg, reference_label=label, ledger_shape=shape)
    except LayeredCheckpointError:
        raise
    except Exception as exc:
        raise LayeredCheckpointError(f"native MAPLE bed validation failed: {type(exc).__name__}: {exc}") from None


def _validate_physical(event: Any, state: Any) -> None:
    """Genuine owner/event validation of an accepted record: types, aliasing, bindings, anchored closure (derived bounds),
    inventories on EVERY cell, router state, completed instantaneous gates, and the native bed. Never trusts a saved flag."""
    try:
        ev.audit_event(event, state)
        w = state.water
        ls._validate_inventories(event.op, w.soil, w.history)
        event.op.router.validate_state(SurfaceState(float(w.t_s), w.surface_m, w.qx_m2_s, w.qy_m2_s))
    except LayeredCheckpointError:
        raise
    except Exception as exc:
        raise LayeredCheckpointError(f"layered state validation failed: {type(exc).__name__}: {exc}") from None
    if event.op.bed is not None:
        if w.bed is None or w.bed.water.depth_m is not w.surface_m:
            _refuse("alias contract violated: native water depth must be the layered surface")
        if is_windowed_bed(w.bed):  # the cold deep scan belongs to the native writer/loader (once); see `_native_checks_windowed`
            _native_checks_windowed(w.bed, event.op.context, float(w.t_s), type(event.op.bed.ledger))
        else:
            _native_checks(w.bed, event.op.context, float(w.t_s), type(event.op.bed.ledger))


# --- scalars ---------------------------------------------------------------------------------------------------------------------------
def _ledger_scalars(ledger: Any) -> dict:
    out = {}
    for name in _LEDGER_FIELDS:
        value = getattr(ledger, name)
        out[name] = int(value) if name == "n_steps" else _fx(value)
    return out


def _ledger_from(scalars: Any, name: str) -> ls.LayeredWaterLedger:
    if not isinstance(scalars, dict) or set(scalars) != set(_LEDGER_FIELDS):
        _refuse(f"{name} ledger keys mismatch")
    kwargs = {f: (_int(scalars[f], f"{name}.{f}") if f == "n_steps" else _unhex(scalars[f], f"{name}.{f}", lo=0.0))
              for f in _LEDGER_FIELDS}
    return ls.LayeredWaterLedger(**kwargs)


def _state_scalars(state: Any) -> dict:
    w, q = state.water, state.quiet
    return {
        "t_s": _fx(w.t_s), "soil_t_s": _fx(w.soil.t_s), "generation": int(w.generation), "operator_sha256": w.operator_sha256,
        "history_operator_sha256": w.history.operator_sha256, "event_sha256": state.event_sha256,
        "ledger": _ledger_scalars(w.ledger), "anchor_ledger": _ledger_scalars(state.anchor_ledger),
        "anchor_surface_m3": _fx(state.anchor_surface_m3), "anchor_soil_m3": _fx(state.anchor_soil_m3),
        "start_generation": int(state.start_generation), "next_dt_cap_s": _fx(state.next_dt_cap_s),
        "forcing_index": int(state.forcing_index), "accepted_steps": int(state.accepted_steps),
        "rejected_attempts": int(state.rejected_attempts),
        "quiet": {"since_s": None if q.since_s is None else _fx(q.since_s), "window_depth_change_m": _fx(q.window_depth_change_m)},
        "completed": bool(state.completed),
    }


_SCALAR_KEYS = frozenset({"t_s", "soil_t_s", "generation", "operator_sha256", "history_operator_sha256", "event_sha256", "ledger",
                          "anchor_ledger", "anchor_surface_m3", "anchor_soil_m3", "start_generation", "next_dt_cap_s",
                          "forcing_index", "accepted_steps", "rejected_attempts", "quiet", "completed"})


# --- native node codec through the existing whitelist -----------------------------------------------------------------------------------
def _encode_native(bed: Any, add_array: Callable[[np.ndarray], str]) -> dict:
    try:
        return {"ledger": pc._native_encode(bed.ledger, add_array),
                "committed_topography": pc._native_encode(bed.committed_topography, add_array)}
    except Exception as exc:  # incl. an invalid present/absent native class binding raised by the genuine classifier
        raise _convert(exc) from None


# --- file / archive audit --------------------------------------------------------------------------------------------------------------
def _read_header(member: Any, label: str, member_size: int) -> tuple[tuple, bool, np.dtype]:
    """Parse an `.npy` header with BOUNDED reads only. NumPy reads and decodes the declared header length before testing its own
    `max_header_size`, so the magic, version and length prefix are read here with small fixed reads, the declared length is checked against
    `MAX_NPY_HEADER` and the trusted remaining member bytes BEFORE the body is read, and only the exact prefix + body (<= 10 + MAX_NPY_HEADER
    bytes) is handed to NumPy's own literal-only parser in a bounded BytesIO, with the real version semantics (v1/v2 latin-1, v3 UTF-8).
    Returns (shape, fortran_order, dtype); `member` is left positioned after the header."""
    from numpy.lib import format as npf

    magic = member.read(8)
    if len(magic) != 8 or magic[:6] != b"\x93NUMPY":
        _refuse(f"{label}: not an .npy member (bad or truncated magic)")
    version = (magic[6], magic[7])
    if version == (1, 0):
        width = 2
    elif version in ((2, 0), (3, 0)):
        width = 4
    else:
        _refuse(f"{label}: unsupported .npy version {version}")
        raise AssertionError
    prefix = member.read(width)
    if len(prefix) != width:
        _refuse(f"{label}: truncated .npy length prefix")
    length = int.from_bytes(prefix, "little")
    if length > MAX_NPY_HEADER:
        _refuse(f"{label}: declared .npy header length {length} exceeds the policy {MAX_NPY_HEADER} (refused before the body is read)")
    if 8 + width + length > member_size:
        _refuse(f"{label}: declared .npy header length {length} exceeds the member's {member_size} bytes")
    body = member.read(length)
    if len(body) != length:
        _refuse(f"{label}: truncated .npy header body")
    buffer = io.BytesIO(magic + prefix + body)
    npf.read_magic(buffer)
    try:  # NumPy's own version-aware parser (v3 = UTF-8); no public v3 entry point exists, so the private one is the delegate
        from numpy.lib._format_impl import _read_array_header
    except ImportError:
        _read_array_header = npf._read_array_header
    shape, fortran, dtype = _read_array_header(buffer, version, max_header_size=MAX_NPY_HEADER)
    if buffer.tell() != 8 + width + length:
        _refuse(f"{label}: .npy header length is inconsistent")
    return shape, fortran, dtype


def _zip_preflight(fileobj: Any, size: int, expected_entries: int, label: str) -> None:
    """Bound the ZIP central directory BEFORE `zipfile` materialises it: the end-of-central-directory record is parsed here at the end
    of the file (no comment allowed), the entry count must be exactly the profile's, the directory size must fit the declared per-entry
    caps and tile the gap to the EOCD exactly. ZIP64 end records (needed only above 4 GiB / 65535 entries) are refused (declared limit)."""
    import struct

    if size < 22:
        _refuse(f"{label}: too short to be an archive")
    fileobj.seek(size - 22)
    eocd = fileobj.read(22)
    sig, disk, cd_disk, here, total, cd_size, cd_offset, comment = struct.unpack("<IHHHHIIH", eocd)
    if sig != 0x06054B50 or disk or cd_disk or comment:
        _refuse(f"{label}: no plain end-of-central-directory record at the end of the file (comments / multi-disk refused)")
    if here != total or total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        _refuse(f"{label}: ZIP64 / inconsistent end record is outside the declared metadata policy")
    if total != expected_entries or total > MAX_MEMBERS:
        _refuse(f"{label}: {total} entries differ from the profile's {expected_entries}")
    if cd_size > total * ZIP_CD_ENTRY_MAX or cd_offset + cd_size + 22 != size:
        _refuse(f"{label}: central directory size/offset exceed the declared per-entry caps or do not tile the file")


def _audit_archive(source: Any, expected: dict[str, tuple], *, label: str, byte_limit: int,
                   string_caps: dict[str, int] | None = None) -> dict[str, tuple[tuple, np.dtype]]:
    """Bounded structural audit BEFORE any array expands (source = a binary file object positioned anywhere; its size is read with
    seek). Order: EOCD/central-directory bound (`_zip_preflight`), then exact member set, stored (uncompressed, unencrypted) regular
    `.npy` members without paths, links, comments or oversized extras, every header parsed by numpy's literal-only parser capped at
    `MAX_NPY_HEADER`, size arithmetic equal to the declared member size, no object/structured/Fortran data and exact dtype/shape against
    `expected` ((shape, dtype) or (None, "string") with a per-member string cap; `string_caps` names an explicit per-member character cap
    where the fixed Dense caps do not apply, e.g. the Windowed `metadata` scalar; default None = the unchanged Dense policy). Returns the parsed headers."""
    out: dict[str, tuple] = {}
    total = 0
    try:
        source.seek(0, os.SEEK_END)
        size = source.tell()
        _zip_preflight(source, size, len(expected), label)
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBERS or len(infos) != len(expected):
                _refuse(f"{label}: member count {len(infos)} differs from the {len(expected)} members of the profile")
            for info in infos:
                name = info.filename
                if (info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1 or info.is_dir()
                        or not name.endswith(".npy") or "/" in name or "\\" in name or name.startswith(".") or "\x00" in name
                        or len(name.encode("utf-8")) > ZIP_NAME_MAX or len(info.extra) > ZIP_EXTRA_MAX or info.comment
                        or info.compress_size != info.file_size or name in out
                        or ((info.external_attr >> 16) & 0o170000) == 0o120000):
                    _refuse(f"{label}: member {name!r} is not an allowed stored regular .npy array")
                key = name[:-4]
                if key not in expected:
                    _refuse(f"{label}: unexpected member {name!r}")
                total += int(info.file_size)
                if total > byte_limit:
                    _refuse(f"{label}: declared members exceed the bound {byte_limit}")
                with archive.open(info) as member:
                    shape, fortran, dtype = _read_header(member, f"{label}:{name}", int(info.file_size))
                    offset = member.tell()
                if (fortran or dtype.hasobject or dtype.fields is not None or dtype.kind not in "fiubSU"
                        or not isinstance(shape, tuple) or len(shape) > 5 or any(type(n) is not int or n < 0 for n in shape)):
                    _refuse(f"{label}:{name}: dtype/shape/order not allowed (no object, structured, Fortran data)")
                if offset + math.prod(shape) * dtype.itemsize != int(info.file_size):
                    _refuse(f"{label}:{name}: header (shape {shape}, dtype {dtype}) disagrees with the member size {info.file_size}")
                want_shape, want_dtype = expected[key]
                if want_dtype == "string":
                    cap = (string_caps or {}).get(key) or (NATIVE_PROVENANCE_MAX if key == "provenance_json" else NATIVE_STRING_MAX)
                    if dtype.kind not in "SU" or shape != () or dtype.itemsize > cap * (4 if dtype.kind == "U" else 1):
                        _refuse(f"{label}:{name}: not a bounded scalar string")
                elif dtype.str != want_dtype or tuple(shape) != tuple(want_shape):
                    _refuse(f"{label}:{name}: header {dtype.str} {shape} differs from the profile's {want_dtype} {tuple(want_shape)}")
                out[key] = (shape, dtype)
    except LayeredCheckpointError:
        raise
    except (zipfile.BadZipFile, OSError, ValueError, EOFError, SyntaxError, TypeError, KeyError, RecursionError,
            struct_error) as exc:
        raise LayeredCheckpointError(f"{label}: malformed archive or .npy header: {type(exc).__name__}: {exc}") from None
    if set(out) != set(expected):
        _refuse(f"{label}: member set differs from the profile")
    return out


def _audit_path(path: Path, expected: dict, *, label: str, byte_limit: int, string_caps: dict[str, int] | None = None) -> dict:
    """Audit a file the caller OWNS exclusively (the private stable copy); never a shared path."""
    with open(path, "rb") as handle:
        return _audit_archive(handle, expected, label=label, byte_limit=byte_limit, string_caps=string_caps)


def _fd_hash(fd: int, limit: int) -> tuple[str, int]:
    h, total = hashlib.sha256(), 0
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, IO_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            _refuse("a payload grew beyond its bound while being read")
        h.update(chunk)
    return h.hexdigest(), total


def _open_regular(name: str, dirfd: int, limit: int) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(name, flags, dir_fd=dirfd)
    except OSError as exc:
        raise LayeredCheckpointError(f"cannot open {name} without following links: {exc}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _refuse(f"{name} is not a plain single-link regular file")
        if info.st_size > limit:
            _refuse(f"{name} ({info.st_size} B) exceeds the serialized bound {limit}")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _scan_json(raw: bytes, max_depth: int, max_nodes: int, max_string: int) -> None:
    """Linear structural scan of ASCII JSON text BEFORE `json.loads`: depth, node (`{ [ , :` outside strings), string-length and
    ASCII-only caps. It allocates nothing proportional to the input (one pass over the bytes), so no hostile manifest can make the
    parser build more than the declared policy before it is refused."""
    if not raw.isascii():
        _refuse("manifest must be ASCII (non-ASCII text is escaped by the writer)")
    depth = nodes = slen = 0
    in_str = esc = False
    for ch in raw:
        if in_str:
            if esc:
                esc = False
            elif ch == 0x5C:
                esc = True
            elif ch == 0x22:
                in_str = False
            else:
                slen += 1
                if slen > max_string:
                    _refuse("manifest string exceeds the declared per-string bound")
            continue
        if ch == 0x22:
            in_str, slen = True, 0
        elif ch in (0x7B, 0x5B):
            depth += 1
            nodes += 1
            if depth > max_depth:
                _refuse("manifest nesting exceeds the declared depth bound")
        elif ch in (0x7D, 0x5D):
            depth -= 1
        elif ch in (0x2C, 0x3A):
            nodes += 1
        if nodes > max_nodes:
            _refuse("manifest node count exceeds the declared bound")
    if depth != 0 or in_str:
        _refuse("manifest is not balanced JSON")


def _read_manifest(dirfd: int, cap_bytes: int, max_nodes: int) -> dict:
    """The manifest under its DERIVED policy: file size is checked by `fstat` against `cap_bytes` BEFORE any read, the raw bytes are
    scanned structurally BEFORE `json.loads`, and only then decoded (ASCII) and parsed (strict, no duplicate keys, no non-finite)."""
    fd = _open_regular(MANIFEST_NAME, dirfd, cap_bytes)
    try:
        raw = b""
        while len(raw) <= cap_bytes:
            chunk = os.read(fd, IO_CHUNK_BYTES)
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(fd)
    if len(raw) > cap_bytes:
        _refuse("manifest exceeds its derived byte bound")
    _scan_json(raw, MANIFEST_MAX_DEPTH, max_nodes, MANIFEST_MAX_STRING)

    def no_constants(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant {token}")

    def no_duplicates(pairs: list) -> dict:
        out: dict = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"duplicate manifest key {k!r}")
            out[k] = v
        return out

    try:
        data = json.loads(raw.decode("ascii"), parse_constant=no_constants, object_pairs_hook=no_duplicates)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise LayeredCheckpointError(f"manifest is not strict JSON: {exc}") from None
    if not isinstance(data, dict):
        _refuse("manifest must be a JSON object")
    return data


# --- save ---------------------------------------------------------------------------------------------------------------------------------
def _disk_reader(path: Path, override: Callable[[], int] | None) -> Callable[[], int]:
    if override is not None:
        if not callable(override):
            _refuse("disk_available_bytes_fn must be callable")
        return override

    def reader() -> int:
        info = os.statvfs(path)
        return int(info.f_bavail) * int(info.f_frsize)

    return reader


def _identity_of(fd_or_path: Any, *, dir_fd: int | None = None) -> tuple[int, int]:
    info = os.fstat(fd_or_path) if isinstance(fd_or_path, int) else os.stat(fd_or_path, dir_fd=dir_fd, follow_symlinks=False)
    return (info.st_dev, info.st_ino)


def _unlink_owned(dirfd: int, owned: dict) -> None:
    """Remove ONLY regular files whose current (dev, inode) equals the identity this call recorded when it created them. There is no
    name-only ownership: an entry without a recorded identity is never removed."""
    for name, ident in list(owned.items()):
        try:
            info = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
        except OSError:
            continue
        if ident is not None and stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == ident:
            try:
                os.unlink(name, dir_fd=dirfd)
            except OSError:
                pass


class _Reservation:
    """The writer's own artifacts, identified by (dev, inode): the reserved bundle directory, its files, and a private staging
    directory (same filesystem) for the path-only native writer. Cleanup touches nothing else."""

    def __init__(self, target: Path, dirfd: int, dir_id: tuple[int, int]):
        self.target, self.dirfd, self.dir_id = target, dirfd, dir_id
        self.created: dict[str, tuple[int, int]] = {}
        self.stage: Path | None = None
        self.stage_fd: int | None = None
        self.stage_id: tuple[int, int] | None = None
        self.stage_created: dict[str, tuple[int, int]] = {}

    def check_bundle_identity(self) -> None:
        """The held directory must still be what the bundle PATH names (a rename/replace race refuses before any publication)."""
        try:
            now = os.lstat(self.target)
        except OSError as exc:
            raise LayeredCheckpointError(f"the bundle directory path vanished: {exc}") from None
        if (now.st_dev, now.st_ino) != self.dir_id or not stat.S_ISDIR(now.st_mode):
            _refuse("the bundle directory path no longer names the reserved directory (replaced/renamed); refused before sealing")

    def remove_own(self) -> None:
        _unlink_owned(self.dirfd, self.created)
        if self.stage is not None and self.stage_fd is not None:
            _unlink_owned(self.stage_fd, self.stage_created)
            try:
                now = os.lstat(self.stage)
                if (now.st_dev, now.st_ino) == self.stage_id and stat.S_ISDIR(now.st_mode):
                    os.rmdir(self.stage)
            except OSError:
                pass
        try:
            now = os.lstat(self.target)
            if (now.st_dev, now.st_ino) == self.dir_id and stat.S_ISDIR(now.st_mode):
                os.rmdir(self.target)  # fails harmlessly if a foreign file appeared
        except OSError:
            pass

    def close(self) -> None:
        for fd in (self.stage_fd,):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        os.close(self.dirfd)


def _write_companion(dirfd: int, res: _Reservation, host: dict[str, np.ndarray]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(PAYLOAD_COMPANION, flags, 0o600, dir_fd=dirfd)
    res.created[PAYLOAD_COMPANION] = _identity_of(fd)
    with os.fdopen(fd, "wb") as handle:
        np.savez(handle, **host)
        handle.flush()
        _fsync(handle.fileno())


def _fsync_name(name: str, dirfd: int) -> None:
    fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=dirfd)
    try:
        _fsync(fd)
    finally:
        os.close(fd)


def save_layered_checkpoint(path: Any, event: Any, state: Any, identity: dict, *, limits: CheckpointLimits,
                            host_available_bytes_fn: Callable[[], int], device_available_bytes_fn: Any = None,
                            disk_available_bytes_fn: Callable[[], int] | None = None, bed_storage: str = "dense") -> SaveReceipt:
    """Write a NEW sealed checkpoint directory for the accepted `state` of `event`. By default (`bed_storage="dense"`) Windowed beds,
    CUDA and every unsupported representation are refused first; `bed_storage="windowed"` selects the explicit Windowed profile
    (published full/v1 external deep store, no pending journal; overlay refused before any deep work). Admission of the four quotas precedes capture, validation and every write; the same physical
    validation the loader runs precedes the first byte. No commit, reroute, settle, advance, reset or wind/plant/sediment operation."""
    from maple.io.outputs import snapshot as native_snapshot

    _require_cpu_dense(event, state, bed_storage)
    if not isinstance(limits, CheckpointLimits):
        _refuse("limits must be a CheckpointLimits record")
    limits.validated()
    ident = _canonical_identity(identity)
    try:
        ev._check_event(event)
        _check_backend(event)
        ev._check_state(event, state)  # metadata-only type/binding checks before the budget is derived
    except LayeredCheckpointError:
        raise
    except Exception as exc:
        raise _convert(exc) from None
    if bed_storage == "windowed":
        # pending-journal / overlay / prefix-binding refusals come from metadata alone: before the budget, the cold scan and every write
        saved_deep = _deep_record(state.water.bed.voxel_column, event.op.context, event.op.shape)
        if saved_deep != _deep_record(event.op.bed.voxel_column, event.op.context, event.op.shape):
            _refuse("the state's deep-store version differs from the reference event's (different physical bed)")
    target = Path(path)
    if os.path.lexists(target):
        _refuse(f"checkpoint path already exists (no overwrite): {target}")
    parent = target.resolve().parent
    if not parent.is_dir() or parent.is_symlink():
        _refuse(f"the parent directory does not exist or is a link: {parent}")
    target = parent / target.name
    budget = estimate_checkpoint_budget(event)
    _admit_all(budget, limits, host_need=budget.host_save_bytes, host_fn=host_available_bytes_fn, device_fn=device_available_bytes_fn,
               disk_fn=_disk_reader(parent, disk_available_bytes_fn), disk_need=budget.serialized_bytes, op_backend=event.op.backend)
    ext_prints = None
    if bed_storage == "windowed":  # fresh external-dependency verification (a warm handle's cached mappings never stand in for the files)
        ext_prints = preflight_external_deep(state.water.bed.voxel_column, event.op.context, int(event.op.n_classes))
    _validate_physical(event, state)  # same checks as the loader, BEFORE anything is written

    op = event.op
    profile = _profile(event)
    status = "complete" if state.completed else "wet"
    scalars = _state_scalars(state)
    host: dict[str, np.ndarray] = {}
    table: dict[str, dict] = {}

    def add(name: str, array: Any) -> str:
        a = np.array(np.asarray(array), copy=True, order="C")
        if not pc._dtype_ok(a.dtype.str) or a.ndim > 5:
            _refuse(f"{name}: dtype {a.dtype.str} / ndim {a.ndim} not allowed")
        host[name] = a
        table[name] = {"dtype": a.dtype.str, "shape": [int(n) for n in a.shape]}
        return name

    w = state.water
    add("soil_water_m", w.soil.water_m)
    add("wetting_m", w.history.wetting_m)
    add("cumulative_intake_m", w.history.cumulative_intake_m)
    add("qx_m2_s", w.qx_m2_s)
    add("qy_m2_s", w.qy_m2_s)
    add("topo_source_full", op.topology.source_z_full)
    native: dict = {}
    if op.bed is None:
        add("surface_m", w.surface_m)
    else:
        def add_native(array: np.ndarray) -> str:
            return add(f"native_{len(host):04d}", array)

        native = _encode_native(w.bed, add_native)
    binding = _binding(event)
    runtime = runtime_profile(op.backend)
    manifest = {"schema": SCHEMA, "profile": profile, "identity": ident, "status": status, "array_backend": op.xp.__name__,
                "hydraulic_backend": op.backend, "source": layered_checkpoint_source_pair(), "runtime": runtime,
                "binding": binding, "binding_sha256": _sha_text(_json_dumps(binding)), "files": {}, "sizes": {},
                "payload": {"scalars": scalars, "arrays": table, "native": native}}
    if profile == WINDOWED_PROFILE:
        manifest["deep_storage"] = binding["deep_storage"]
    try:
        _json_dumps(manifest)
    except (TypeError, ValueError) as exc:
        raise LayeredCheckpointError(f"manifest is not strict JSON: {exc}") from None
    provenance = {"syrup_layered_checkpoint_schema": SCHEMA, "profile": profile, "requires_companion": True, "identity": ident,
                  "binding_sha256": manifest["binding_sha256"], "array_backend": op.xp.__name__,
                  "hydraulic_backend": op.backend, "status": status, "runtime": runtime}
    string_caps: dict[str, int] | None = None
    if profile == WINDOWED_PROFILE:
        provenance.update(_windowed_provenance(binding["deep_storage"]))
        string_caps = {"metadata": _meta_max_chars(_bed_dims(op)[1])}
    digest = physical_digest(event, state)
    specs = {PAYLOAD_COMPANION: _companion_expected(op, table)}
    if profile == NATIVE_PROFILE:
        specs[PAYLOAD_NATIVE] = _native_snapshot_specs(op)
    elif profile == WINDOWED_PROFILE:
        specs[PAYLOAD_NATIVE] = _windowed_native_specs(op.shape, _bed_dims(op)[1], binding["deep_storage"])

    flags_dir = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        pfd = os.open(parent, flags_dir)
    except OSError as exc:
        raise LayeredCheckpointError(f"cannot open the parent directory: {exc}") from None
    res = None
    try:
        try:
            _mkdir(target, 0o700)
        except FileExistsError:
            raise LayeredCheckpointError(f"checkpoint path already exists (no overwrite): {target}") from None
        except OSError as exc:
            raise LayeredCheckpointError(f"cannot reserve the checkpoint directory: {exc}") from None
        try:
            dirfd = os.open(target, flags_dir)
        except OSError as exc:
            raise LayeredCheckpointError(f"cannot open the reserved directory: {exc}") from None
        res = _Reservation(target, dirfd, _identity_of(dirfd))
    except BaseException:
        os.close(pfd)
        raise
    sealed = False
    tmp_name = f".manifest.tmp-{os.getpid()}-{os.urandom(6).hex()}"
    try:
        _fsync(pfd)  # the reserved directory NAME is durable in its parent before anything is built inside it
        if os.listdir(res.dirfd):
            _refuse("the reserved checkpoint directory is not empty (a foreign writer raced the reservation)")
        dirfd = res.dirfd
        files = FILES[profile]
        if profile in (NATIVE_PROFILE, WINDOWED_PROFILE):
            # The native writers are path-only. The Dense one truncates a file this call created exclusively (identity recorded); the
            # Windowed one REFUSES an existing path and creates its own temp file + no-clobber final link, so it gets a nonexistent
            # name in a private same-filesystem stage directory this call owns (inode held). The finished file is then published into
            # the held bundle directory by a no-clobber hard link: a foreign file already at the bundle name makes the link fail and
            # nothing is overwritten.
            res.stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=parent))
            res.stage_fd = os.open(res.stage, flags_dir)
            res.stage_id = _identity_of(res.stage_fd)
            if profile == NATIVE_PROFILE:
                sfd = os.open(PAYLOAD_NATIVE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600,
                              dir_fd=res.stage_fd)
                try:
                    staged_id = _identity_of(sfd)
                finally:
                    os.close(sfd)
                res.stage_created[PAYLOAD_NATIVE] = staged_id
                native_snapshot.save_state_snapshot(
                    w.bed.voxel_column, w.bed.active_layer, float(w.t_s), res.stage / PAYLOAD_NATIVE, step=int(w.generation),
                    reason=_SNAPSHOT_REASON, geometry=op.context.geometry, mass_resolution_kg=op.context.mass_resolution_kg,
                    sediment_availability=w.bed.sediment_availability, water=w.bed.water, provenance=provenance)
            else:
                from maple.surface.voxels.windowed_snapshot import save_windowed_state_snapshot

                try:  # validates (the ONE cold deep scan) before any native byte is written
                    save_windowed_state_snapshot(
                        w.bed.voxel_column, w.bed.active_layer, float(w.t_s), res.stage / PAYLOAD_NATIVE, step=int(w.generation),
                        geometry=op.context.geometry, mass_resolution_kg=op.context.mass_resolution_kg,
                        sediment_availability=w.bed.sediment_availability, water=w.bed.water, provenance=provenance)
                except LayeredCheckpointError:
                    raise
                except Exception as exc:
                    raise LayeredCheckpointError(f"the native Windowed snapshot writer refused: {type(exc).__name__}: {exc}") from None
                staged_id = _identity_of(PAYLOAD_NATIVE, dir_fd=res.stage_fd)
                res.stage_created[PAYLOAD_NATIVE] = staged_id
                if not stat.S_ISREG(os.stat(PAYLOAD_NATIVE, dir_fd=res.stage_fd, follow_symlinks=False).st_mode):
                    _refuse("the native Windowed writer produced a non-regular staged payload")
            if _identity_of(PAYLOAD_NATIVE, dir_fd=res.stage_fd) != staged_id:
                _refuse("the staged native payload changed identity during the native write")
            _fsync_name(PAYLOAD_NATIVE, res.stage_fd)
            res.check_bundle_identity()
            try:
                _link(PAYLOAD_NATIVE, PAYLOAD_NATIVE, src_dir_fd=res.stage_fd, dst_dir_fd=dirfd, follow_symlinks=False)
            except FileExistsError:
                raise LayeredCheckpointError(
                    "the bundle already holds a file at the native payload name (a foreign writer raced); nothing was overwritten") \
                    from None
            res.created[PAYLOAD_NATIVE] = staged_id
            os.unlink(PAYLOAD_NATIVE, dir_fd=res.stage_fd)
            res.stage_created.pop(PAYLOAD_NATIVE, None)
            os.rmdir(res.stage)
        if ext_prints is not None:  # the dependency must still be what the writer validated: refuse BEFORE the seal
            recheck_external_deep(state.water.bed.voxel_column, ext_prints)
        res.check_bundle_identity()
        _write_companion(dirfd, res, host)
        del host
        sizes, hashes = {}, {}
        for name in files:  # post-write audit through bound descriptors (never a path): identity, plain file, hash, headers
            fd = _open_regular(name, dirfd, budget.serialized_bytes)
            try:
                if _identity_of(fd) != res.created[name]:
                    _refuse(f"{name} changed identity after it was written")
                hashes[name], sizes[name] = _fd_hash(fd, budget.serialized_bytes)
                with os.fdopen(os.dup(fd), "rb") as handle:
                    _audit_archive(handle, specs[name], label=name, byte_limit=limits.max_serialized_bytes,
                                   string_caps=string_caps if name == PAYLOAD_NATIVE else None)
            finally:
                os.close(fd)
        if sum(sizes.values()) > min(budget.serialized_bytes, limits.max_serialized_bytes):
            raise LayeredCheckpointQuotaError(f"written payload {sum(sizes.values())} B exceeds the serialized bound")
        sealed_manifest = pc._seal_manifest({**manifest, "files": hashes, "sizes": sizes})
        text = json.dumps(sealed_manifest, indent=1, sort_keys=True, allow_nan=False) + "\n"
        if len(text) > budget.manifest_cap_bytes or not text.isascii():
            _refuse("manifest exceeds its derived byte bound")
        res.check_bundle_identity()
        tfd = os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=dirfd)
        res.created[tmp_name] = _identity_of(tfd)
        with os.fdopen(tfd, "wb") as handle:
            handle.write(text.encode("ascii"))
            handle.flush()
            _fsync(handle.fileno())
        _fsync(dirfd)
        res.check_bundle_identity()
        try:
            _link(tmp_name, MANIFEST_NAME, src_dir_fd=dirfd, dst_dir_fd=dirfd, follow_symlinks=False)  # atomic no-clobber seal
        except FileExistsError:
            raise LayeredCheckpointError("the seal name already exists (a foreign writer raced); nothing was overwritten") from None
        sealed = True
        os.unlink(tmp_name, dir_fd=dirfd)
        res.created.pop(tmp_name, None)
        _fsync(dirfd)
        _fsync(pfd)
    except BaseException as exc:
        if sealed:
            raise LayeredCheckpointIndeterminate(
                f"failure after the seal link ({type(exc).__name__}: {exc}); the bundle at {target} may be durable: nothing "
                "was deleted and it must not be retried over") from exc
        res.remove_own()
        if isinstance(exc, LayeredCheckpointError):
            raise
        if isinstance(exc, Exception):
            raise LayeredCheckpointError(f"save failed before the seal: {type(exc).__name__}: {exc}") from exc
        raise
    finally:
        res.close()
        os.close(pfd)
    return SaveReceipt(str(target), profile, status, sealed_manifest["manifest_sha256"], dict(hashes), dict(sizes), float(w.t_s),
                       int(w.generation), digest, budget.as_dict(), manifest["source"])


def _companion_expected(op: Any, table: dict) -> dict[str, tuple]:
    """Exact companion member table: the fixed arrays from trusted dimensions, native leaves from the declared table (which the
    loader independently equates to the reference bed's leaf multiset before any expansion)."""
    fixed = _companion_specs(op)
    expected: dict[str, tuple] = dict(fixed)
    for name, rec in table.items():
        if name in fixed:
            continue
        expected[name] = (tuple(rec["shape"]), rec["dtype"])
    return expected


# --- digest (receipts / tests; whole-array streaming, bounded staging) -------------------------------------------------------------------
def physical_digest(event: Any, state: Any) -> str:
    """SHA-256 over every physical value of the continuation: arrays (streamed), all scalars as float hex, both ledgers, anchors,
    cap, cursor, counters, quiet, completion and, with a native bed, its voxel/active/availability/water/ledger/committed state."""
    w = state.water
    h = hashlib.sha256(b"maple_syrup.layered_checkpoint.physical/1")
    for value in (w.t_s, w.soil.t_s, int(w.generation), w.operator_sha256, state.event_sha256, w.ledger, state.anchor_ledger,
                  float(state.anchor_surface_m3), float(state.anchor_soil_m3), int(state.start_generation),
                  float(state.next_dt_cap_s), int(state.forcing_index), int(state.accepted_steps), int(state.rejected_attempts),
                  state.quiet.since_s, float(state.quiet.window_depth_change_m), bool(state.completed)):
        _digest_value(h, value)
    for array in (w.soil.water_m, w.history.wetting_m, w.history.cumulative_intake_m, w.surface_m, w.qx_m2_s, w.qy_m2_s):
        _digest_value(h, array)
    if w.bed is not None:
        if is_windowed_bed(w.bed):
            _digest_windowed_column(h, w.bed.voxel_column)
        else:
            _digest_value(h, w.bed.voxel_column)
        _digest_value(h, w.bed.active_layer)
        _digest_value(h, w.bed.sediment_availability)
        _digest_value(h, w.bed.water)
        _digest_value(h, w.bed.ledger)
        _digest_value(h, w.bed.committed_topography)
    return h.hexdigest()


# --- load ----------------------------------------------------------------------------------------------------------------------------------
def _stable_copy(name: str, dirfd: int, private: Path, expected_sha: str, expected_size: int, limit: int) -> Path:
    """Copy `name` through its open descriptor into the private directory while hashing; the copy is what every later step reads."""
    fd = _open_regular(name, dirfd, limit)
    try:
        if os.fstat(fd).st_size != expected_size:
            _refuse(f"{name} size differs from the manifest")
        dest = private / name
        out = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        h, total = hashlib.sha256(), 0
        try:
            while True:
                chunk = os.read(fd, IO_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit or total > expected_size:
                    _refuse(f"{name} grew beyond its declared size while being read")
                h.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(out, view)
                    if written <= 0:
                        _refuse(f"no write progress copying {name} to the private stable copy (I/O failure; refused, not retried)")
                    view = view[written:]
        finally:
            os.close(out)
        if total != expected_size or h.hexdigest() != expected_sha:
            _refuse(f"payload hash/size mismatch: {name}")
    finally:
        os.close(fd)
    return dest


def _check_topology(saved_source: np.ndarray, ref_top: Any, binding_top: dict) -> None:
    """Independent rebuild of the held topology through the actual builder; compared field by field with the reference and the
    saved identities. The reference object itself stays the binding (never a new routing source)."""
    try:
        fresh = build_pond_topology(np.array(saved_source, dtype=np.float64, copy=True), ref_top.active, ref_top.friction,
                                    ref_top.dx_m, ref_top.outlets, nodata_value=ref_top.nodata_value,
                                    allow_closed_components=ref_top.n_closed_components > 0)
    except Exception as exc:
        raise LayeredCheckpointError(f"the saved held terrain does not rebuild a valid pond topology: {exc}") from None
    fresh = dataclasses.replace(fresh, generation=ref_top.generation)
    bad = pc._fields_mismatch(pc._topology_fields(fresh), pc._topology_fields(ref_top))
    if bad:
        _refuse(f"saved held terrain rebuilds a topology that differs from the reference in {bad}")
    if (binding_top.get("identity_sha256"), binding_top.get("source_sha256"), binding_top.get("topology_sha256")) != (
            ref_top.identity_sha256, ref_top.source_sha256, ref_top.topology_sha256):
        _refuse("saved topology identities differ from the reference")


def load_layered_checkpoint(path: Any, expected_event: Any, expected_identity: dict, *, limits: CheckpointLimits,
                            allow_complete: bool = False, host_available_bytes_fn: Callable[[], int],
                            device_available_bytes_fn: Any = None,
                            disk_available_bytes_fn: Callable[[], int] | None = None,
                            bed_storage: str = "dense") -> LayeredRestoration:
    """Restore a sealed checkpoint against a genuine prepared reference event. Returns the NEW event and its restored state; nothing
    is adopted. See the module docstring and design.md for the strict order; every refusal leaves all inputs and files unchanged."""
    from maple.io.outputs import snapshot as native_snapshot

    _require_cpu_dense(expected_event, None, bed_storage)
    if not isinstance(limits, CheckpointLimits):
        _refuse("limits must be a CheckpointLimits record")
    limits.validated()
    if type(allow_complete) is not bool:
        _refuse("allow_complete must be a bool")
    expected = _canonical_identity(expected_identity)
    try:
        ev._check_event(expected_event)
        _check_backend(expected_event)
    except LayeredCheckpointError:
        raise
    except Exception as exc:
        raise _convert(exc) from None
    if bed_storage == "windowed":  # pending journal / overlay / prefix refusals from the REFERENCE's metadata, before budget and any read
        _deep_record(expected_event.op.bed.voxel_column, expected_event.op.context, expected_event.op.shape)
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _refuse(f"checkpoint path is not a plain directory: {root}")
    op = expected_event.op
    profile = _profile(expected_event)
    budget = estimate_checkpoint_budget(expected_event)
    _admit_all(budget, limits, host_need=budget.host_load_bytes, host_fn=host_available_bytes_fn, device_fn=device_available_bytes_fn,
               disk_fn=_disk_reader(Path(tempfile.gettempdir()), disk_available_bytes_fn), disk_need=budget.serialized_bytes,
               op_backend=op.backend)  # the private stable copy lives in the temp directory's filesystem
    dirfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    private = Path(tempfile.mkdtemp(prefix="layered-checkpoint-stable-"))
    try:
        names = set(os.listdir(dirfd))
        if MANIFEST_NAME not in names:
            _refuse("unsealed or partial checkpoint directory (no sealed manifest)")
        if names != {MANIFEST_NAME, *FILES[profile]}:
            _refuse(f"checkpoint directory must contain exactly {sorted({MANIFEST_NAME, *FILES[profile]})}, found {sorted(names)}")
        data = _read_manifest(dirfd, budget.manifest_cap_bytes, budget.manifest_max_nodes)
        return _load(data, dirfd, private, expected_event, expected, profile, budget, limits, allow_complete, native_snapshot)
    except LayeredCheckpointError:
        raise
    except Exception as exc:
        raise LayeredCheckpointError(f"load refused: {type(exc).__name__}: {exc}") from exc
    finally:
        os.close(dirfd)
        shutil.rmtree(private, ignore_errors=True)


def _load(data: dict, dirfd: int, private: Path, ref: Any, expected: dict, profile: str, budget: CheckpointBudget,
          limits: CheckpointLimits, allow_complete: bool, native_snapshot: Any) -> LayeredRestoration:
    op = ref.op
    ny, nx = op.shape
    V, K = _bed_dims(op)
    # -- manifest and every binding BEFORE any payload is opened --------------------------------------------------------
    want_keys = WINDOWED_MANIFEST_KEYS if profile == WINDOWED_PROFILE else MANIFEST_KEYS
    if set(data) != want_keys:
        _refuse(f"manifest keys mismatch: missing {sorted(want_keys - set(data))}, unknown {sorted(set(data) - want_keys)}")
    if data["schema"] != SCHEMA:
        _refuse(f"checkpoint schema {data['schema']!r} is not {SCHEMA!r}")
    if _sha(data["manifest_sha256"], "manifest_sha256") != pc._seal_manifest(data)["manifest_sha256"]:
        _refuse("manifest hash mismatch (the manifest was modified)")
    if data["profile"] != profile:
        _refuse(f"profile {data['profile']!r} does not match the reference's {profile!r}; native and nonnative never cross-load")
    if data["identity"] != expected:
        _refuse("checkpoint identity differs from the expected identity")
    if data["source"] != layered_checkpoint_source_pair():
        _refuse("source pair mismatch: the imported maple_syrup / maple packages differ from the writer's (refused before any payload)")
    if data["array_backend"] != op.xp.__name__ or data["hydraulic_backend"] != op.backend:
        _refuse("backend mismatch: no cross-backend migration (numpy and numba are not interchangeable)")
    _check_runtime_record(data["runtime"], op.backend)  # freshly inspected runtime, BEFORE any payload work
    if data["status"] not in STATUSES:
        _refuse(f"unknown status {data['status']!r}")
    if data["status"] == "complete" and not allow_complete:
        _refuse("a completed event loads only with allow_complete=True")
    binding = _binding(ref)
    deep = None
    if profile == WINDOWED_PROFILE:  # typed record check + equality with the REFERENCE's published version, before any payload/deep access
        deep = _check_deep_record(data["deep_storage"], ref.op.context, op.shape, limits.max_serialized_bytes)
        if deep != binding["deep_storage"]:
            _refuse("the checkpoint's deep-store version differs from the reference bed's (identity, geometry or store location)")
    if data["binding"] != binding or data["binding_sha256"] != _sha_text(_json_dumps(binding)):
        diff = sorted(k for k in binding if not isinstance(data["binding"], dict) or data["binding"].get(k) != binding[k])
        _refuse(f"binding mismatch (source/runtime/context/properties/forcing differ from the reference): {diff}")
    payload = data["payload"]
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        _refuse("payload keys mismatch")
    files, sizes = data["files"], data["sizes"]
    names = FILES[profile]
    if not isinstance(files, dict) or set(files) != set(names) or not isinstance(sizes, dict) or set(sizes) != set(names):
        _refuse("payload file set mismatch")
    declared = sum(_int(sizes[n], f"sizes[{n}]") for n in names)
    if declared > limits.max_serialized_bytes or declared > budget.serialized_bytes:
        raise LayeredCheckpointQuotaError(f"declared payload {declared} B exceeds the serialized limit/derived bound")
    # -- companion table against the reference BEFORE any payload expansion -----------------------------------------------
    table = payload["arrays"]
    fixed = _companion_specs(op)
    if not isinstance(table, dict):
        _refuse("array table must be an object")
    native_names = [n for n in table if n not in fixed]
    for n, (shape, dtype) in fixed.items():
        rec = table.get(n)
        if not isinstance(rec, dict) or set(rec) != {"dtype", "shape"} or rec["dtype"] != dtype or tuple(rec.get("shape", ())) != shape:
            _refuse(f"array {n}: declared dtype/shape differ from the profile")
    for n in native_names:
        rec = table[n]
        if (not isinstance(n, str) or not n.startswith("native_") or not isinstance(rec, dict) or set(rec) != {"dtype", "shape"}
                or not pc._dtype_ok(rec["dtype"]) or not isinstance(rec["shape"], list) or len(rec["shape"]) > 5
                or any(type(v) is not int or v < 0 for v in rec["shape"])):
            _refuse(f"array {n!r}: not an allowed declared native leaf")
    if profile in (NATIVE_PROFILE, WINDOWED_PROFILE):
        ref_leaves = sorted(_leaf_specs(op.bed.ledger) + _leaf_specs(op.bed.committed_topography))
        got = sorted((table[n]["dtype"], tuple(table[n]["shape"])) for n in native_names)
        if got != ref_leaves:
            _refuse("declared native ledger/committed leaves differ from the reference bed's (class, shapes, present processes)")
    elif native_names or payload["native"]:
        _refuse("the nonnative profile carries no native components")
    scalars = payload["scalars"]
    if not isinstance(scalars, dict) or set(scalars) != _SCALAR_KEYS:
        _refuse("scalar keys mismatch")
    # -- stable private copies, hashed through the open descriptors -------------------------------------------------------
    stable = {n: _stable_copy(n, dirfd, private, _sha(files[n], f"files[{n}]"), int(sizes[n]), limits.max_serialized_bytes)
              for n in names}
    # -- archive audit of every member and every header -------------------------------------------------------------------
    if profile == NATIVE_PROFILE:
        _audit_path(stable[PAYLOAD_NATIVE], _native_snapshot_specs(op), label=PAYLOAD_NATIVE, byte_limit=limits.max_serialized_bytes)
    elif profile == WINDOWED_PROFILE:
        _audit_path(stable[PAYLOAD_NATIVE], _windowed_native_specs(op.shape, K, deep), label=PAYLOAD_NATIVE,
                    byte_limit=limits.max_serialized_bytes, string_caps={"metadata": _meta_max_chars(K)})
    _audit_path(stable[PAYLOAD_COMPANION], _companion_expected(op, table), label=PAYLOAD_COMPANION,
                byte_limit=limits.max_serialized_bytes)
    # -- native snapshot -------------------------------------------------------------------------------------------------------
    snapshot = None
    t_s = _unhex(scalars["t_s"], "t_s", lo=0.0)
    generation = _int(scalars["generation"], "generation")
    if profile == NATIVE_PROFILE:
        try:
            snapshot = native_snapshot.load_state_snapshot(stable[PAYLOAD_NATIVE])
        except Exception as exc:
            raise LayeredCheckpointError(f"MAPLE snapshot cannot be loaded: {type(exc).__name__}: {exc}") from None
        _check_snapshot(snapshot, data, t_s, generation)
    elif profile == WINDOWED_PROFILE:
        snapshot = _load_windowed_snapshot(stable[PAYLOAD_NATIVE], int(sizes[PAYLOAD_NATIVE]), data, deep, op, K, t_s, generation)
    # -- companion decode through the fixed whitelist --------------------------------------------------------------------------
    arrays, ledger_node, committed_node = _decode_companion(stable[PAYLOAD_COMPANION], table, payload["native"], profile)
    # -- rebuild + validation + re-prepare ------------------------------------------------------------------------------------
    _check_topology(arrays["topo_source_full"], op.topology, data["binding"]["topology"])
    return _restore(ref, scalars, arrays, snapshot, ledger_node, committed_node, data, budget, t_s, generation)


def _expected_windowed_provenance(data: dict, deep: dict) -> dict:
    return {"syrup_layered_checkpoint_schema": SCHEMA, "profile": WINDOWED_PROFILE, "requires_companion": True,
            "identity": data["identity"], "binding_sha256": data["binding_sha256"], "array_backend": data["array_backend"],
            "hydraulic_backend": data["hydraulic_backend"], "status": data["status"], "runtime": data["runtime"],
            **_windowed_provenance(deep)}


def _read_native_metadata(path: Path, cap_chars: int, n_classes: int) -> dict:
    """The native `metadata` member of the AUDITED private copy, read with bounded reads (header already checked: 0-d Unicode scalar of
    at most 4*cap_chars bytes), decoded here (UTF-32), structurally scanned (ASCII JSON depth/nodes/strings) and strictly parsed
    (no duplicate keys, no non-finite constants) BEFORE the native loader is allowed to open any path the metadata names."""
    try:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo("metadata.npy")
            with archive.open(info) as member:
                shape, fortran, dtype = _read_header(member, "metadata", int(info.file_size))
                if shape != () or fortran or dtype.kind != "U" or dtype.byteorder not in ("<", "=") or dtype.itemsize > 4 * cap_chars:
                    _refuse("metadata must be one bounded little-endian 0-d Unicode scalar")
                raw = member.read(dtype.itemsize + 1)
    except LayeredCheckpointError:
        raise
    except (zipfile.BadZipFile, OSError, KeyError, ValueError, EOFError) as exc:
        raise LayeredCheckpointError(f"native metadata member cannot be read: {type(exc).__name__}: {exc}") from None
    if len(raw) != dtype.itemsize:
        _refuse("native metadata member length disagrees with its header")
    try:
        text = raw.decode("utf-32-le").rstrip("\x00")
    except UnicodeDecodeError:
        _refuse("native metadata is not valid UTF-32 text")
    if not text.isascii():
        _refuse("native metadata must be ASCII JSON (the native writer escapes non-ASCII)")
    _scan_json(text.encode("ascii"), METADATA_JSON_DEPTH, MANIFEST_FIXED_NODES + 4 * n_classes + 64, MAX_IDENTITY_BYTES)

    def no_constants(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant {token}")

    def no_duplicates(pairs: list) -> dict:
        out: dict = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"duplicate metadata key {k!r}")
            out[k] = v
        return out

    try:
        meta = json.loads(text, parse_constant=no_constants, object_pairs_hook=no_duplicates)
    except (ValueError, RecursionError) as exc:
        raise LayeredCheckpointError(f"native metadata is not strict JSON: {exc}") from None
    if not isinstance(meta, dict) or set(meta) != WINDOWED_META_KEYS:
        _refuse("native metadata keys differ from the closed native set")
    return meta


def _check_native_metadata(meta: dict, data: dict, deep: dict, t_s: float, generation: int) -> None:
    """Typed comparison of the preflighted metadata with the manifest's deep record and expectations, before any deep access."""
    def exact_int(v: Any) -> bool:
        return type(v) is int

    if (meta["schema"] != WINDOWED_SNAPSHOT_SCHEMA or meta["store_dir"] != deep["store_dir"]
            or meta["manifest_digest"] != deep["physical_manifest_digest"] or meta["header_digest"] != deep["header_digest"]
            or not exact_int(meta["generation"]) or meta["generation"] != deep["generation"]
            or not exact_int(meta["n_voxels"]) or meta["n_voxels"] != deep["n_voxels"]
            or not exact_int(meta["step"]) or meta["step"] != generation
            or type(meta["time_s"]) is not float or meta["time_s"] != t_s or meta["class_ids"] != deep["class_ids"]):
        _refuse("native snapshot metadata is not the manifest's deep-store version / clock (refused before any deep access)")
    if meta["provenance"] != _expected_windowed_provenance(data, deep):
        _refuse("native Windowed snapshot provenance is not bound to this manifest (refused before any deep access)")


def _load_windowed_snapshot(native_path: Path, native_size: int, data: dict, deep: dict, op: Any, K: int, t_s: float,
                            generation: int) -> dict:
    """Preflight, then the genuine native Windowed loader (default mapping cache; its ONE cold validation), then the returned-handle
    comparison. A missing/corrupt/changed external deep dependency is a typed refusal; nothing is substituted or repaired."""
    from maple.core.types.voxel import WindowedVoxelColumnState
    from maple.surface.voxels.windowed_snapshot import load_windowed_state_snapshot

    meta = _read_native_metadata(native_path, _meta_max_chars(K), K)
    _check_native_metadata(meta, data, deep, t_s, generation)
    try:
        resolved = str(Path(deep["store_dir"]).resolve())
    except (OSError, RuntimeError) as exc:
        raise LayeredCheckpointDeepStoreError(f"the deep-store directory cannot be resolved: {exc}") from None
    if resolved != deep["store_dir"]:
        raise LayeredCheckpointDeepStoreError("the recorded deep-store path no longer resolves to itself (moved/relinked); no sibling or "
                                              "latest-version fallback exists")
    ctx = op.context
    ref_column = op.bed.voxel_column
    ext_prints = preflight_external_deep(ref_column, ctx, K)   # bounded, fresh; BEFORE the native loader may reopen any store file
    try:
        snapshot = load_windowed_state_snapshot(native_path, geometry=ctx.geometry, grain_classes=ctx.grain_classes,
                                                mass_resolution_kg=ctx.mass_resolution_kg, max_bytes=max(int(native_size), 1))
    except Exception as exc:
        raise LayeredCheckpointDeepStoreError(
            f"the native Windowed snapshot or its external deep-store version refused (missing, corrupted, changed or inconsistent "
            f"deep data; the deep files are an immutable dependency): {type(exc).__name__}: {exc}") from None
    recheck_external_deep(ref_column, ext_prints)
    keys = {"voxel_column", "active_layer", "sediment_availability", "water", "time_s", "step", "validated", "restartable",
            "is_diagnostic", "provenance"}
    if not isinstance(snapshot, dict) or set(snapshot) != keys:
        _refuse("native Windowed snapshot fields differ from the native layout")
    if snapshot["validated"] is not True or snapshot["restartable"] is not True or snapshot["is_diagnostic"] is not False:
        _refuse("invalid native Windowed snapshot (not validated / restartable, or diagnostic); flags are checks, never proof")
    if snapshot["provenance"] != _expected_windowed_provenance(data, deep):
        _refuse("native Windowed snapshot provenance is not bound to this manifest")
    column = snapshot["voxel_column"]
    if type(column) is not WindowedVoxelColumnState or column.pending_deep is not None:
        _refuse("the snapshot column is not a published WindowedVoxelColumnState")
    store, header = column.deep_store, column.deep_store.header
    prefix = _windowed_modules()[2]
    try:
        packets = prefix.require_prefix_packets(column, operation="SYRUP layered checkpoint")
    except (ValueError, TypeError) as exc:
        raise LayeredCheckpointError(f"the loaded column's prefix packets are not bound to it: {exc}") from None
    if (int(store.format_version) != deep["store_format_version"] or store.manifest_digest != deep["physical_manifest_digest"]
            or store.logical_manifest_digest != deep["logical_manifest_digest"] or header.digest() != deep["header_digest"]
            or str(Path(store.store_dir).resolve()) != deep["store_dir"] or int(store.version_id) != deep["generation"]
            or int(column.generation) != deep["generation"] or column.manifest_digest != deep["physical_manifest_digest"]
            or int(column.n_voxels) != deep["n_voxels"] or int(column.window_kg.shape[2]) != deep["window_voxels"]
            or int(header.chunk_columns) != deep["chunk_columns"] or int(store.n_chunks) != deep["n_chunks"]
            or [str(c) for c in header.class_ids] != deep["class_ids"] or int(packets.packet_terms) != deep["packet_terms"]):
        _refuse("the loaded deep-store version is not the one recorded in the manifest (identity, format, geometry or location differ)")
    if snapshot["time_s"] != t_s or snapshot["step"] != generation:
        _refuse("native snapshot clock/step differ from the companion scalars")
    # the call-local validation receipt: these exact objects were validated by the native loader in THIS call (read-only arrays)
    for a in (column.window_kg, column.base, column.deep_by_class_kg, column.deep_total_kg, column.deep_room_kg, packets.packet,
              snapshot["active_layer"].mass_kg):
        if a.flags.writeable:
            _refuse("a natively validated array is writable: the validation receipt cannot be bound to immutable values")
    return snapshot


def _check_snapshot(snapshot: dict, data: dict, t_s: float, generation: int) -> None:
    from maple.io.outputs.snapshot import SNAPSHOT_SCHEMA_VERSION
    from maple.io.outputs.visualization_schema import SCHEMA_VERSION as HISTORY_SCHEMA

    if not snapshot["validated"] or not snapshot["restartable"] or snapshot["is_diagnostic"]:
        _refuse("MAPLE snapshot is not validated/restartable or is diagnostic (flags are checks, never proof)")
    if snapshot["snapshot_schema_version"] != SNAPSHOT_SCHEMA_VERSION or snapshot["history_schema_version"] != HISTORY_SCHEMA:
        _refuse("MAPLE snapshot schema/history semantic identity is not the exact supported one")
    try:
        prov = json.loads(snapshot["provenance_json"])
    except ValueError:
        prov = None
    if (not isinstance(prov, dict) or prov.get("syrup_layered_checkpoint_schema") != SCHEMA or prov.get("profile") != NATIVE_PROFILE
            or prov.get("binding_sha256") != data["binding_sha256"] or prov.get("identity") != data["identity"]
            or prov.get("status") != data["status"] or prov.get("array_backend") != data["array_backend"]
            or prov.get("hydraulic_backend") != data["hydraulic_backend"] or prov.get("requires_companion") is not True
            or prov.get("runtime") != data["runtime"]):
        _refuse("MAPLE snapshot provenance is not bound to this manifest")
    if (snapshot["water"] is None or snapshot["sediment_availability"] is None or snapshot["history_store_path"]
            or snapshot["reason"] != _SNAPSHOT_REASON
            or any(snapshot.get(k) is not None for k in _FOREIGN_SNAPSHOT_FIELDS)
            or snapshot.get("sediment_availability_fallback_sha256") or snapshot.get("scheduler_coupled_status")
            or snapshot.get("scheduler_wind_record_source_sha256") or snapshot["run_name"] or snapshot["error_type"]
            or snapshot["error_message"] or snapshot["seed"] != 0):
        _refuse("MAPLE snapshot lacks water/availability or carries foreign optional state")
    if snapshot["time_s"] != t_s or snapshot["step"] != generation:
        _refuse("MAPLE snapshot clock/step differ from the companion scalars")


def _decode_companion(path: Path, table: dict, native: Any, profile: str) -> tuple[dict, Any, Any]:
    used: set[str] = set()
    arrays: dict[str, np.ndarray] = {}
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise LayeredCheckpointError(f"companion cannot be opened: {type(exc).__name__}: {exc}") from None
    with archive:
        if set(archive.files) != set(table):
            _refuse("companion arrays differ from the array table")

        def get(name: Any) -> np.ndarray:
            if not isinstance(name, str) or name not in table:
                _refuse(f"unknown array reference {name!r}")
            if name in used:
                _refuse(f"aliased array reference {name}")
            used.add(name)
            rec = table[name]
            try:
                a = np.array(archive[name], copy=True, order="C")
            except Exception as exc:
                raise LayeredCheckpointError(f"array {name} cannot be read: {type(exc).__name__}: {exc}") from None
            if a.dtype.str != rec["dtype"] or list(a.shape) != rec["shape"]:
                _refuse(f"array {name}: stored dtype/shape differ from the table")
            if a.dtype.kind == "f" and not np.isfinite(a).all():
                _refuse(f"array {name} holds non-finite values")
            return a

        for name in _companion_fixed_names(profile):
            arrays[name] = get(name)
        ledger_node = committed_node = None
        if profile in (NATIVE_PROFILE, WINDOWED_PROFILE):
            if not isinstance(native, dict) or set(native) != {"ledger", "committed_topography"}:
                _refuse("native companion keys mismatch")
            try:
                ledger_node = pc._native_decode(native["ledger"], get)
                committed_node = pc._native_decode(native["committed_topography"], get)
            except LayeredCheckpointError:
                raise
            except Exception as exc:
                raise _convert(exc) from None
    if used != set(table):
        _refuse("unused, missing or duplicate companion array references")
    return arrays, ledger_node, committed_node


def _companion_fixed_names(profile: str) -> tuple[str, ...]:
    base = ("soil_water_m", "wetting_m", "cumulative_intake_m", "qx_m2_s", "qy_m2_s", "topo_source_full")
    return base + (("surface_m",) if profile == NONNATIVE_PROFILE else ())


def _restore(ref: Any, scalars: dict, arrays: dict, snapshot: dict | None, ledger: Any, committed: Any, data: dict,
             budget: CheckpointBudget, t_s: float, generation: int) -> LayeredRestoration:
    op = ref.op
    # typed scalars (exact keys and types before any object is built)
    if scalars["operator_sha256"] != op.sha256 or scalars["history_operator_sha256"] != op.matrix.sha256 \
            or scalars["event_sha256"] != ref.sha256:
        _refuse("saved operator/matrix/event identities differ from the reference")
    if _unhex(scalars["soil_t_s"], "soil_t_s", lo=0.0) != t_s:
        _refuse("saved soil clock differs from the physical clock")
    saved_ledger = _ledger_from(scalars["ledger"], "ledger")
    anchor_ledger = _ledger_from(scalars["anchor_ledger"], "anchor_ledger")
    q = scalars["quiet"]
    if not isinstance(q, dict) or set(q) != {"since_s", "window_depth_change_m"}:
        _refuse("quiet record keys mismatch")
    quiet = ev.QuietProgress(None if q["since_s"] is None else _unhex(q["since_s"], "quiet.since_s", lo=0.0),
                             _unhex(q["window_depth_change_m"], "quiet.window_depth_change_m", lo=0.0))
    if type(scalars["completed"]) is not bool:
        _refuse("completed must be a bool")
    new_bed = None
    new_op_kwargs: dict = {}
    if op.bed is not None:
        new_bed = BedState(voxel_column=snapshot["voxel_column"], active_layer=snapshot["active_layer"], water=snapshot["water"],
                           ledger=ledger, committed_topography=committed, sediment_availability=snapshot["sediment_availability"])
        if data["profile"] == WINDOWED_PROFILE:  # the cold scan was the native loader's, on exactly these (immutable) objects
            if (new_bed.voxel_column is not snapshot["voxel_column"] or new_bed.active_layer is not snapshot["active_layer"]):
                _refuse("the validated Windowed objects were replaced before adoption")
            _native_checks_windowed(new_bed, op.context, t_s, type(op.bed.ledger))
        else:
            _native_checks(new_bed, op.context, t_s, type(op.bed.ledger))
        if getattr(ledger, "present_processes", None) != getattr(op.bed.ledger, "present_processes", None):
            _refuse("native ledger present_processes differ from the reference (process selection is restart-relevant)")
        new_op_kwargs = {"native_reserve_bytes": op.native_reserve_bytes}
    # genuine re-preparation: reference column/pref/matrix/router/graph/topology/context + the NEW loaded bed
    new_op = ls.prepare_layered_storm(op.column, op.preferential, op.matrix, op.router, op.graph, op.topology, bed=new_bed,
                                      context=op.context, extra_reserve_bytes=op.extra_reserve_bytes, **new_op_kwargs)
    if new_op.sha256 != op.sha256 or new_op.native_material_sha256 != op.native_material_sha256:
        _refuse("the re-prepared operator identity differs from the saved one (native material or bindings differ)")
    new_event = ev.prepare_layered_event(new_op, ref.schedule, ref.field, event_origin_s=ref.event_origin_s,
                                         control=ref.control, policy=ref.policy)
    if new_event.sha256 != ref.sha256:
        _refuse("the re-prepared event identity differs from the saved one")
    soil = ss.column_state(op.column, arrays["soil_water_m"], t_s)
    history = mi.wetting_history(op.matrix, wetting_m=arrays["wetting_m"], cumulative_intake_m=arrays["cumulative_intake_m"])
    surface = new_bed.water.depth_m if new_bed is not None else np.ascontiguousarray(arrays["surface_m"])
    water = ls.LayeredWaterState(t_s, surface, np.ascontiguousarray(arrays["qx_m2_s"]), np.ascontiguousarray(arrays["qy_m2_s"]), soil,
                                 history, new_bed, saved_ledger, generation, new_op.sha256)
    state = ev.LayeredEventState(
        water, new_event.sha256, _unhex(scalars["anchor_surface_m3"], "anchor_surface_m3", lo=0.0),
        _unhex(scalars["anchor_soil_m3"], "anchor_soil_m3", lo=0.0), anchor_ledger, _int(scalars["start_generation"], "start_generation"),
        _unhex(scalars["next_dt_cap_s"], "next_dt_cap_s", lo=0.0), _int(scalars["forcing_index"], "forcing_index", lo=-1),
        _int(scalars["accepted_steps"], "accepted_steps"), _int(scalars["rejected_attempts"], "rejected_attempts"), quiet,
        scalars["completed"])
    if (data["status"] == "complete") != state.completed:
        _refuse("manifest status disagrees with the saved completion flag")
    _validate_physical(new_event, state)
    receipt = {"path_profile": data["profile"], "status": data["status"], "t_s": t_s, "generation": generation,
               "manifest_sha256": data["manifest_sha256"], "files": dict(data["files"]), "source": data["source"],
               "physical_digest": physical_digest(new_event, state), "budget": budget.as_dict(),
               "operator_sha256": new_op.sha256, "event_sha256": new_event.sha256, "adopted": False}
    return LayeredRestoration(new_event, state, receipt)
