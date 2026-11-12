"""Source-bound CPU checkpoint of the accepted layered-water LIFECYCLE (task phase1_lifecycle_checkpoint_20261110).

A NEW closed schema (`SCHEMA`) and two profiles save/load/continue ONE current role of `water_lifecycle`: an audited closed
`boundary`, a genuine `storm` (rain/recession), or a genuine general requested-T `interevent`. NumPy/Numba CPU only; actual MAPLE39
Dense bed (`native39_dense_lifecycle_cpu_v1`) or the genuine bedless lifecycle (`bedless_lifecycle_cpu_v1`). It adds no physical law.

    save_lifecycle_checkpoint(path, lop, state, identity, *, limits, host_available_bytes_fn, ...) -> LifecycleSaveReceipt
    load_lifecycle_checkpoint(path, lop, expected_identity, *, expected_role, limits, host_available_bytes_fn,
                              reference_bed=None, storm=None, ...) -> LifecycleRestoration(lop, state, receipt)

Reuses the qualified generic archive / native / reservation helpers of `layered_checkpoint` (unchanged) and the owner restore seams
of `interevent_water`, `interevent_transpiration` and `water_lifecycle`, reached ONLY through a call-local verified capability
(`checkpoint_capability`). Persisted: the one current physical payload (H once through the native water, faces, layered soil, J,
SEPARATE lifetime intake, genuine native bed) and bounded scalars (local ledgers, source receipt, lifetime anchors/fluxes, mark,
exact clocks/counters, last phase/control receipts). Nothing about past events, past arrays or an adaptive interval cap / PET
cursor / calendar (the current lifecycle owns none) is stored, and nothing here is a combined MASY scheduler restart.

Source: every save/load freshly hashes the ACTUAL imported `maple_syrup`/`maple` packages (after quota admission, before any native
reader/writer; no cache) and the native39 profile requires the literal MAPLE39 package digest; the metadata-only estimator is
source-neutral. Material: the fixed native material is bound by a cold FULL streamed fingerprint (voxels, active layer, availability,
ledger, committed, mobile, context; not water depth), compared for the expected reference and the loaded bed.

Trust boundary: SHA-256 digests and the recomputed owner digests demonstrate source consistency and integrity of THIS payload.
They are not signatures and cannot re-prove that a past storm's historical quiet audit physically ran. A verified file identity
is not adversarial-filesystem race proof. Load returns a NEW owner and state (a new parent instance); nothing is published or
adopted; pending external requests are not restorable. See docs/subsurface_hydrology/lifecycle_checkpoint.md.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable

import numpy as np
from maple.core.types.voxel import VoxelColumnState

from maple_syrup import checkpoint_capability as cc
from maple_syrup import interevent_entry as ie
from maple_syrup import interevent_transpiration as it
from maple_syrup import interevent_water as iw
from maple_syrup import layered_checkpoint as lc
from maple_syrup import layered_event as ev
from maple_syrup import layered_storm as ls
from maple_syrup import matrix_infiltration as mi
from maple_syrup import pond_checkpoint as pc
from maple_syrup import subsurface as ss
from maple_syrup import water_lifecycle as wl
from maple_syrup import provenance as pv
from maple_syrup.conservation import volume_roundoff_bound_m3 as ie_volume_roundoff
from maple_syrup.provenance import DEFAULT_MAX_FILES
from maple_syrup.sediment_bed import BedState, is_windowed_bed

__all__ = [
    "BEDLESS_PROFILE", "LifecycleBudget", "LifecycleCheckpointError", "LifecycleCheckpointIndeterminate",
    "LifecycleCheckpointQuotaError", "LifecycleRestoration", "LifecycleSaveReceipt", "NATIVE_PROFILE", "SCHEMA", "StormSpec",
    "estimate_lifecycle_budget", "load_lifecycle_checkpoint", "physical_digest", "save_lifecycle_checkpoint",
]

SCHEMA = "maple-syrup-water-lifecycle-checkpoint/1"
NATIVE_PROFILE = "native39_dense_lifecycle_cpu_v1"
BEDLESS_PROFILE = "bedless_lifecycle_cpu_v1"
ROLES = ("boundary", "storm", "interevent")
MANIFEST_NAME = "lifecycle_checkpoint.json"
PAYLOAD_NATIVE = "maple_state.npz"
PAYLOAD_COMPANION = "lifecycle_state.npz"
FILES = {NATIVE_PROFILE: (PAYLOAD_NATIVE, PAYLOAD_COMPANION), BEDLESS_PROFILE: (PAYLOAD_COMPANION,)}
MANIFEST_KEYS = frozenset({"schema", "profile", "role", "identity", "array_backend", "hydraulic_backend", "source", "runtime",
                           "binding", "binding_sha256", "files", "sizes", "payload", "manifest_sha256"})
PAYLOAD_KEYS = frozenset({"scalars", "arrays", "native"})
SCALAR_PARTS = {"boundary": frozenset({"lifecycle"}), "storm": frozenset({"lifecycle", "storm"}),
                "interevent": frozenset({"lifecycle", "t_state", "receipt"})}
STORM_RECORD_KEYS = frozenset({"generation", "event_sha256", "ledger", "anchor_ledger", "anchor_surface_m3", "anchor_soil_m3",
                               "start_generation", "next_dt_cap_s", "forcing_index", "accepted_steps", "rejected_attempts",
                               "quiet", "completed", "event_origin_s", "rainfall_end_s", "deadline_s"})
#: Mandatory FRESH source-pair audit (`_fresh_source_pair`, two `source_tree_digest` calls, no cache): one 1 MiB read chunk + the entry list of up to
#: `DEFAULT_MAX_FILES` files (`relative path` + 64-hex digest + tuple overhead, 512 B each); charged on EVERY call (cold hash).
SOURCE_AUDIT_BYTES = (1 << 20) + DEFAULT_MAX_FILES * 512
#: The ONLY `maple` package accepted by the `native39_dense_lifecycle_cpu_v1` profile (merged MAPLE 39bcf79e...). The frozen 9f
#: package and every other source are refused for this profile before any native reader/writer; no 9f archive is ever written.
NATIVE39_PACKAGE_SHA256 = "af54626ab432670539ed71480fd884f13581dfdfcfec63d7c85053ca215adfec"
OBJECT_BYTES = 4096
SOURCE_KINDS_V2 = (iw.STORM_END, iw.BOUNDARY_ORIGIN)
_fsync = os.fsync
_link = os.link
_mkdir = os.mkdir


class LifecycleCheckpointError(RuntimeError):
    """Definite refusal: nothing was sealed (save) or adopted (load); inputs, the accepted state and foreign files unchanged."""


class LifecycleCheckpointQuotaError(LifecycleCheckpointError):
    """A caller limit or a reported availability is below the derived requirement; raised before the guarded allocation."""


class LifecycleCheckpointIndeterminate(LifecycleCheckpointError):
    """A failure AFTER the atomic seal link: the bundle may be durable. Nothing was deleted; do not retry over it."""


def _refuse(message: str) -> None:
    raise LifecycleCheckpointError(message)


def _convert(exc: BaseException) -> LifecycleCheckpointError:
    return LifecycleCheckpointError(f"{type(exc).__name__}: {exc}")


# --- sealed private policy (engine): v1 default and the explicit v2 -----------------------------------------------------------------
_POLICY_SEAL = object()


class _Policy:
    """Immutable, registered, module-private description of ONE codec version (schema, profile names, file sets, manifest key set, closed
    scalar-part table, literal package pins, source-kind vocabulary, metadata allowances). Built only at import by this module
    (constructor refuses without `_POLICY_SEAL`); every map is an owned read-only copy. There is no public way to construct, copy, replace,
    subclass or override one: the engine accepts only the two registered literal objects (`_policy_of`)."""

    __slots__ = ("name", "schema", "native_profile", "bedless_profile", "files", "package_pins", "manifest_keys", "scalar_parts",
                 "source_kinds", "v2", "envelope_bytes", "envelope_nodes", "native_label", "snapshot_census", "census_bytes")

    def __init__(self, seal: object, **kw: Any) -> None:
        if seal is not _POLICY_SEAL:
            raise LifecycleCheckpointError("a checkpoint policy is created only by the codec module")
        for key in self.__slots__:
            value = kw.pop(key)
            if key in ("files", "package_pins"):
                value = MappingProxyType({k: tuple(v) if key == "files" else str(v) for k, v in value.items()})
            elif key == "scalar_parts":
                value = MappingProxyType({k: frozenset(v) for k, v in value.items()})
            elif key in ("manifest_keys",):
                value = frozenset(value)
            elif key == "source_kinds":
                value = tuple(value)
            elif key == "snapshot_census":
                value = None if value is None else frozenset(value)
            object.__setattr__(self, key, value)
        if kw:
            raise LifecycleCheckpointError(f"unexpected policy fields {sorted(kw)}")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a checkpoint policy is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a checkpoint policy is immutable")

    def parts(self, role: str, kind: str | None) -> frozenset:
        return self.scalar_parts[(role, kind)]


def _count_json_nodes(text: str) -> int:
    """Node count by the manifest scanner's own rule (`{ [ , :` outside strings)."""
    n, in_str, esc = 0, False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[,:":
            n += 1
    return n


def _derive_envelope_allowance() -> tuple[int, int]:
    """Closed worst-case bytes and nodes the v2 source envelope (and the two `source_kind` fields) can add to the manifest. Derived from the
    closed key sets and types (never from one observed manifest): every float at its widest hex form, every integer at 2**63-1, a
    128-character token at its widest ASCII-escaped form, every closed key present; indentation at the manifest nesting depth (3 extra
    spaces per line). The larger of the two kinds is taken."""
    wide = float.fromhex("-0x1.fffffffffffffp-1022").hex()
    sha, big = "f" * 64, 2 ** 63 - 1
    f = {"f": wide}
    led = {k: (big if k == "n_steps" else f) for k in iw._LEDGER_RECORD_FIELDS}
    storm = {k: (sha if k in iw._RECEIPT_SHAS else big if k in iw._RECEIPT_COUNTS else f) for k in iw._RECEIPT_SCALARS}
    storm.update(anchor_ledger=led, end_ledger=dict(led), digest=sha)
    boundary = {k: f for k in iw._BOUNDARY_FLOATS}
    boundary.update({k: sha for k in iw._BOUNDARY_SHAS}, parent_token="\U00010000" * iw.MAX_TOKEN_CHARS, parent_revision=big,
                    eligibility=iw.ELIGIBILITY, policy={k: f for k in ie.POLICY_NAMES}, evidence={k: f for k in ie.EVIDENCE_NAMES},
                    digest=sha)
    worst_b = worst_n = 0
    for kind, body in ((iw.STORM_END, storm), (iw.BOUNDARY_ORIGIN, boundary)):
        text = json.dumps({"source_kind": kind, "body": body}, indent=1, sort_keys=True)
        size = len(text) + 3 * text.count("\n") + len('"source": ') + 2
        worst_b, worst_n = max(worst_b, size), max(worst_n, _count_json_nodes(text) + 2)
    kind_field = 2 * (4 + len('"source_kind": ') + len('"boundary_origin"') + 2)  # manifest + binding
    return worst_b + kind_field, worst_n + 4


_PARTS_V1 = {(r, None): p for r, p in (("boundary", {"lifecycle"}), ("storm", {"lifecycle", "storm"}),
                                       ("interevent", {"lifecycle", "t_state", "receipt"}))}
_PARTS_V2 = {("boundary", None): {"lifecycle"}, ("storm", None): {"lifecycle", "storm"},
             ("interevent", iw.STORM_END): {"lifecycle", "t_state", "source"},
             ("interevent", iw.BOUNDARY_ORIGIN): {"lifecycle", "t_state", "source"}}
SCHEMA_V2 = "maple-syrup-water-lifecycle-checkpoint/2"
NATIVE_PROFILE_V2 = "native39_dense_lifecycle_cpu_v2"
BEDLESS_PROFILE_V2 = "bedless_lifecycle_cpu_v2"
_ENV_BYTES, _ENV_NODES = _derive_envelope_allowance()

_V1 = _Policy(_POLICY_SEAL, name="v1", schema=SCHEMA, native_profile=NATIVE_PROFILE, bedless_profile=BEDLESS_PROFILE, files=FILES,
              package_pins={NATIVE_PROFILE: NATIVE39_PACKAGE_SHA256}, manifest_keys=MANIFEST_KEYS, scalar_parts=_PARTS_V1,
              source_kinds=(), v2=False, envelope_bytes=0, envelope_nodes=0, native_label="native39", snapshot_census=None, census_bytes=0)
_V2 = _Policy(_POLICY_SEAL, name="v2", schema=SCHEMA_V2, native_profile=NATIVE_PROFILE_V2, bedless_profile=BEDLESS_PROFILE_V2,
              files={NATIVE_PROFILE_V2: (PAYLOAD_NATIVE, PAYLOAD_COMPANION), BEDLESS_PROFILE_V2: (PAYLOAD_COMPANION,)},
              package_pins={NATIVE_PROFILE_V2: NATIVE39_PACKAGE_SHA256}, manifest_keys=MANIFEST_KEYS | {"source_kind"},
              scalar_parts=_PARTS_V2, source_kinds=SOURCE_KINDS_V2, v2=True, envelope_bytes=_ENV_BYTES, envelope_nodes=_ENV_NODES,
              native_label="native39", snapshot_census=None, census_bytes=0)
# Stage B: the actual MAPLE 9f466d2c ordinary (absent fixed base, absent regional source) snapshot reader returns EXACTLY these 34 keys (32 of the 39 reader
# plus the two unconditional `fixed_base` / `regional_slope_wind_source`); the conditional admission/contract keys are never part of this census.
NATIVE9F_PACKAGE_SHA256 = "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"
NATIVE9F_PROFILE_V2 = "native9f_dense_lifecycle_cpu_v2"
NATIVE9F_SNAPSHOT_CENSUS = frozenset({
    "active_layer", "commit_numerical_residual_mass_kg", "coupled_trajectory_semantics", "error_message", "error_type", "fixed_base",
    "grain_borne_stress", "history_schema_version", "history_store_path", "is_diagnostic", "mobile_state", "provenance_json", "reason",
    "regional_slope_wind_source", "restartable", "run_name", "scheduler_block_index", "scheduler_block_offset_s", "scheduler_coupled_status",
    "scheduler_last_wind", "scheduler_transport_accuracy", "scheduler_wind_record_source_sha256", "sediment_availability",
    "sediment_availability_fallback_sha256", "seed", "snapshot_schema_version", "step", "threshold_modifier_state", "time_s",
    "topographic_wind_source", "validated", "vegetation", "voxel_column", "water"})
_V2_9F = _Policy(_POLICY_SEAL, name="v2-native9f", schema=SCHEMA_V2, native_profile=NATIVE9F_PROFILE_V2, bedless_profile=None,
                 files={NATIVE9F_PROFILE_V2: (PAYLOAD_NATIVE, PAYLOAD_COMPANION)}, package_pins={NATIVE9F_PROFILE_V2: NATIVE9F_PACKAGE_SHA256},
                 manifest_keys=MANIFEST_KEYS | {"source_kind"}, scalar_parts=_PARTS_V2, source_kinds=SOURCE_KINDS_V2, v2=True,
                 envelope_bytes=_ENV_BYTES, envelope_nodes=_ENV_NODES, native_label="native9f", snapshot_census=NATIVE9F_SNAPSHOT_CENSUS,
                 census_bytes=len(NATIVE9F_SNAPSHOT_CENSUS) * lc.NODE_BYTES)
_POLICIES = (_V1, _V2, _V2_9F)


def _policy_of(policy: Any) -> "_Policy":
    """`None` is the v1 default; anything but one of the two registered literal objects is refused before any work."""
    if policy is None:
        return _V1
    if not any(policy is p for p in _POLICIES):
        _refuse("unknown checkpoint policy (only the registered literal policies are accepted)")
    return policy


@dataclass(frozen=True)
class StormSpec:
    """The storm forcing a `storm` checkpoint is bound to (the caller's EXPECTED forcing; its re-prepared event identity must
    equal the saved one): schedule, rainfall field, origin, control and quiet policy."""

    schedule: Any
    field: Any
    event_origin_s: float
    control: Any = ev.LayeredEventControl()
    policy: Any = ev.LayeredQuietPolicy()


@dataclass(frozen=True, eq=False)
class LifecycleRestoration:
    """The NEW lifecycle operator (prepared on the LOADED bed) and the restored state bound to it. Nothing is adopted."""

    lop: Any
    state: Any
    receipt: dict


@dataclass(frozen=True)
class LifecycleSaveReceipt:
    path: str
    profile: str
    role: str
    manifest_sha256: str
    files: dict
    sizes: dict
    t_s: float
    revision: int
    state_digest: str
    physical_digest: str
    budget: dict
    source: dict


@dataclass(frozen=True)
class LifecycleBudget:
    """Whole-call, metadata-only host/disk model (see `estimate_lifecycle_budget`). `terms` names every lifetime."""

    profile: str
    role: str
    n_cells: int
    n_layers: int
    n_classes: int
    n_voxels: int
    native_snapshot_bytes: int
    companion_bytes: int
    n_members_native: int
    n_members_companion: int
    manifest_cap_bytes: int
    manifest_max_nodes: int
    serialized_bytes: int
    metadata_bytes: int
    scratch_bytes: int
    resident_old_bytes: int
    resident_new_bytes: int
    host_save_bytes: int
    host_load_bytes: int
    device_bytes: int
    terms: dict

    def as_dict(self) -> dict:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.__dict__.items()}


# --- fences ------------------------------------------------------------------------------------------------------------------------
def _bed_fence(bed: Any, what: str) -> None:
    """Type-level refusals that precede every native, deep or device operation."""
    if type(bed) is not BedState:
        _refuse(f"{what} must be the genuine sediment_bed.BedState")
    try:
        windowed = is_windowed_bed(bed)
    except Exception as exc:
        raise LifecycleCheckpointError(f"unsupported bed representation: {exc}") from None
    if windowed or type(bed.voxel_column) is not VoxelColumnState:
        _refuse("an actual native Windowed (or foreign) bed has no lifecycle checkpoint profile (refused before any native or "
                "deep operation)")


def _fence(lop: Any, bed: Any, pol: "_Policy | None" = None) -> str:
    """Operator/backend/bed fences. `bed` is the state's current bed (save) or the caller's reference bed (load). Returns the
    policy's profile name for the Dense / bedless lifecycle."""
    pol = _policy_of(pol)
    if type(lop) is not wl.LifecycleOperator:
        _refuse(f"lop must be a genuine LifecycleOperator, got {type(lop).__name__}")
    if lop.backend not in wl.BACKENDS or lop.xp is not np:
        _refuse(f"backend {lop.backend!r} is not supported by the lifecycle checkpoint (CPU NumPy/Numba only; no CUDA/9f)")
    try:
        wl._check_lop(lop)
    except wl.LifecycleError as exc:
        raise _convert(exc) from None
    if lop.context is None:
        if pol.bedless_profile is None:
            _refuse(f"the {pol.native_profile} policy is Dense-only: a bedless lifecycle has no profile under it (use the bedless v2 profile)")
        if bed is not None:
            _refuse("the lifecycle holds no native bed but a bed was supplied")
        return pol.bedless_profile
    if bed is None:
        _refuse("a Dense lifecycle needs its genuine native bed")
    _bed_fence(bed, "the bed")
    return pol.native_profile


def _shim(lop: Any, bed: Any) -> Any:
    """Metadata-only stand-in with the attributes the generic spec helpers read (no array is touched)."""
    return SimpleNamespace(shape=lop.shape, n_layers=lop.n_layers, n_cells=lop.n_cells, n_classes=lop.n_classes, bed=bed)


def _fresh_source_pair(profile: str, pol: "_Policy | None" = None) -> dict:
    """The ACTUAL imported `maple_syrup` and `maple` package digests, freshly hashed on EVERY call through the provenance digest
    API (no process cache is read, cleared or modified: `layered_checkpoint`'s cached pair is deliberately not used). The caller has
    already admitted `SOURCE_AUDIT_BYTES` (one read chunk + the retained file-entry lists). The native39 profile additionally requires
    the literal accepted MAPLE39 package digest; equality of two pairs alone is never sufficient."""
    import maple
    import maple_syrup

    try:
        tree = pv.source_tree_digest(Path(maple_syrup.__file__).resolve().parent)
        maple_tree = pv.source_tree_digest(Path(maple.__file__).resolve().parent)
    except (pv.ProvenanceScopeError, OSError) as exc:
        raise LifecycleCheckpointError(f"the source pair of the imported packages cannot be computed: {exc}") from None
    pair = {"maple_syrup_tree_sha256": tree.digest_sha256, "maple_syrup_n_files": int(tree.file_count),
            "maple_package_sha256": maple_tree.digest_sha256, "algorithm": str(tree.algorithm)}
    pol = _policy_of(pol)
    # v1 keeps reading its module constant at call time (existing behaviour, incl. the existing test that substitutes it); the v2 pin is
    # the policy's frozen literal and cannot be substituted through a module attribute.
    pin = NATIVE39_PACKAGE_SHA256 if (pol is _V1 and profile == NATIVE_PROFILE) else pol.package_pins.get(profile)
    if pin is not None and pair["maple_package_sha256"] != pin:
        _refuse(f"the imported maple package {pair['maple_package_sha256']} is not the accepted {pol.native_label} source "
                f"{pin}: the {profile} profile refuses it before any native reader/writer")
    return pair


def _material_sha(bed: Any, ctx: Any) -> str:
    """COLD, portable, FULL fixed native material fingerprint (never an object id): the actual voxel column, active layer, availability,
    ledger, committed topography, the (zero) mobile array and the context identity, each streamed with its type name, dtype and shape
    through the generic digest walker. Only the evolving water depth is excluded. Used once per bed per call (no per-step walk); the
    cold staging is one `HASH_CHUNK_BYTES` read chunk, charged by `estimate_lifecycle_budget` (`material_stage`)."""
    h = hashlib.sha256(b"maple_syrup.lifecycle_checkpoint.full_native_material/1")
    try:
        for name, part in (("voxel_column", bed.voxel_column), ("active_layer", bed.active_layer),
                           ("sediment_availability", bed.sediment_availability), ("ledger", bed.ledger),
                           ("committed_topography", bed.committed_topography),
                           ("mobile_mass", bed.water.mobile_mass_by_cell_class_kg)):
            h.update(name.encode() + b"\0")
            lc._digest_value(h, part)
        h.update(lc._context_digest(ctx).encode())
    except lc.LayeredCheckpointError as exc:
        raise _convert(exc) from None
    except Exception as exc:
        raise _convert(exc) from None
    return h.hexdigest()


def _binding(lop: Any, profile: str, role: str, bed: Any, material_sha: str | None, pol: "_Policy | None" = None,
             source_kind: str | None = None) -> dict:
    pol = _policy_of(pol)
    top = lop.topology
    V, K = lc._bed_dims(_shim(lop, bed))
    ny, nx = lop.shape
    out = {
        "schema": pol.schema, "profile": profile, "role": role, "array_backend": "numpy", "hydraulic_backend": lop.backend,
        "shape": [int(ny), int(nx)], "dx_m": float(top.dx_m).hex(), "n_layers": int(lop.n_layers), "n_cells": int(lop.n_cells),
        "n_faces": int(lop.memory.n_faces), "n_classes": int(K), "n_voxels": int(V), "lop_sha256": lop.sha256,
        "topology": {"identity_sha256": top.identity_sha256, "source_sha256": top.source_sha256,
                     "topology_sha256": top.topology_sha256, "generation": int(top.generation),
                     "n_closed_components": int(top.n_closed_components)},
        "extra_reserve_bytes": int(lop.extra_reserve_bytes), "native_reserve_bytes": int(lop.native_reserve_bytes),
        "native_material_sha256": material_sha,
        "context_sha256": None if lop.context is None else lc._context_digest(lop.context),
    }
    if pol.v2:
        out["source_kind"] = source_kind
    return out


# --- memory ------------------------------------------------------------------------------------------------------------------------
def _native_resident(bed: Any, ctx: Any) -> int:
    """Actual nbytes of the native leaves that are NOT the owner's counted surface/mobile (voxel, active, availability, ledger,
    committed, initial availability fraction)."""
    return (pc._tree_nbytes(bed.voxel_column) + pc._tree_nbytes(bed.active_layer) + pc._tree_nbytes(bed.sediment_availability)
            + pc._tree_nbytes(bed.ledger) + pc._tree_nbytes(bed.committed_topography)
            + int(np.asarray(ctx.initial_available_fraction).nbytes))


def _interevent_shim(lop: Any) -> Any:
    m = lop.memory
    return SimpleNamespace(
        n_layers=lop.n_layers, n_cells=lop.n_cells, n_classes=lop.n_classes, backend=lop.backend, shape=lop.shape,
        column=lop.column, preferential=lop.preferential, bed=None if lop.context is None else True,
        _control_values=lop._control_values, extra_reserve_bytes=lop.extra_reserve_bytes,
        native_reserve_bytes=lop.native_reserve_bytes, memory=SimpleNamespace(accepted_bytes=m.accepted_bytes,
                                                                              static_bytes=m.static_bytes))


def _role_resident(lop: Any, role: str, event_extra: int) -> tuple[int, dict]:
    """Resident bytes of ONE current role (metadata-only; the retained original-depth H0 is counted once in the storm resident)."""
    m = lop.memory
    if role == "boundary":
        return m.storm["resident_bytes"] + m.reserves_bytes + m.owned_fixed_bytes, {}
    if role == "storm":
        return m.storm["resident_bytes"] + int(event_extra) + m.reserves_bytes + m.owned_fixed_bytes, {}
    iwm, gm = wl._interevent_estimates(lop, _interevent_shim(lop))
    return gm.resident_bytes + gm.reserves_bytes + gm.column_reserve_bytes + m.owned_fixed_bytes, {"iw": iwm, "gm": gm}


def estimate_lifecycle_budget(lop: Any, role: str, *, bed: Any = None, event_extra_bytes: int = 0,
                              extra_reserve_bytes: int = 0) -> LifecycleBudget:
    """Pure arithmetic from the actual component estimates and the declared metadata policy; no allocation, no file or array
    read. `bed` is the current bed (save) or the caller's reference bed (load); `event_extra_bytes` is the live event-map
    resident of a storm (`required_bytes - op.required_bytes`, save; `_event_extra_bytes`, load).

    Whole-call semantics (every term in `terms`): `resident_old` = the caller's still-live owner (statics, the original depth H0
    counted once, reserves) + the reference/current native leaves; `resident_new` = the candidate role resident + the loaded native
    leaves (the loaded H aliases the native water); transient phases are the old declared codec policy (manifest, ZIP/NPY, native
    read, decode, constructor copies, validators) PLUS the mandatory source-pair audit and the native material-hash staging
    (counted, never treated as metadata). Disjoint phases take the maximum, overlapping lifetimes add. Excluded: import/JIT
    workspaces, allocator pools, OS file cache, the process, caller reserves other than `extra_reserve_bytes`."""
    return _estimate_budget(lop, role, bed=bed, event_extra_bytes=event_extra_bytes, extra_reserve_bytes=extra_reserve_bytes)


def _estimate_budget(lop: Any, role: str, *, bed: Any = None, event_extra_bytes: int = 0, extra_reserve_bytes: int = 0,
                     _policy: "_Policy | None" = None, source_kind: str | None = None) -> LifecycleBudget:
    """The whole former body of `estimate_lifecycle_budget` under a private registered policy (`None` = the v1 default: identical
    numbers). A v2 policy adds the named `source_envelope` / `quiet_scratch` terms and the derived manifest allowance."""
    pol = _policy_of(_policy)
    if role not in ROLES:
        _refuse(f"role must be one of {ROLES}")
    if pol.v2 and ((role == "interevent") != (source_kind is not None) or (source_kind is not None and source_kind not in pol.source_kinds)):
        _refuse("a v2 interevent budget needs a closed source_kind and every other role must give none")
    profile = _fence(lop, bed, pol)
    try:
        extra = lc._int(extra_reserve_bytes, "extra_reserve_bytes")
    except lc.LayeredCheckpointError as exc:
        raise _convert(exc) from None
    N, L, K = lop.n_cells, lop.n_layers, lop.n_classes
    shim = _shim(lop, bed)
    V, _ = lc._bed_dims(shim)
    comp_specs = lc._companion_specs(shim)
    companion = sum(int(np.dtype(d).itemsize) * math.prod(s) for s, d in comp_specs.values())
    members_comp = len(comp_specs)
    leaf_bytes = [int(np.dtype(d).itemsize) * math.prod(s) for s, d in comp_specs.values()]
    ledger = committed = native = members_native = 0
    ledger_leaves: list = []
    native_arrays: dict = {}
    voxel_scan = class_scan = ledger_scan = resident_native = 0
    if bed is not None:
        ledger_specs, committed_specs = lc._leaf_specs(bed.ledger), lc._leaf_specs(bed.committed_topography)
        ledger, committed = lc._spec_bytes(ledger_specs), lc._spec_bytes(committed_specs)
        companion += ledger + committed
        members_comp += len(ledger_specs) + len(committed_specs)
        ledger_leaves = [int(np.dtype(d).itemsize) * math.prod(s) for d, s in ledger_specs + committed_specs]
        leaf_bytes += ledger_leaves
        native_arrays = {k: int(np.dtype(d).itemsize) * math.prod(s) for k, (s, d) in lc._native_snapshot_specs(shim).items()
                         if d != "string"}
        native = sum(native_arrays.values())
        members_native = len(lc._native_snapshot_specs(shim))
        leaf_bytes += list(native_arrays.values())
        voxel_scan = lc.VOXEL_SCAN_MASKS * N * V * K + lc.VOXEL_SCAN_NV_TEMPORARIES * 8 * N * V
        class_scan = lc.CLASS_SCAN_NK_TEMPORARIES * 8 * N * K
        ledger_scan = lc.LEDGER_LEAF_TEMPORARIES * max(ledger_leaves + [0])
        resident_native = _native_resident(bed, lop.context)
    owner_audit = lc.OWNER_AUDIT_LN_TEMPORARIES * 8 * L * N + 8 * 8 * N + 4 * (N + lop.memory.n_faces)
    validation = voxel_scan + class_scan + ledger_scan + owner_audit + lop.memory.audit_scratch_bytes
    largest = max(leaf_bytes + [0])
    n_entries = len(comp_specs) + len(ledger_leaves)
    manifest_cap, manifest_nodes = lc._manifest_dims(n_entries)
    metadata = lc._metadata_bytes(n_entries, max(members_comp, members_native))
    if pol.v2:  # derived closed envelope allowance: three concurrent text lifetimes + the decoded nodes (never free slack)
        manifest_cap += pol.envelope_bytes
        manifest_nodes += pol.envelope_nodes
        metadata += 3 * pol.envelope_bytes + pol.envelope_nodes * lc.NODE_BYTES
    if pol.snapshot_census is not None:  # native9f: the returned reader dict (34 records incl. the two None keys) + the census comparison
        metadata += pol.census_bytes
    serialized = (companion + native + (members_comp + members_native) * (lc.NPY_HEADER_ALLOWANCE + lc.ZIP_MEMBER_ALLOWANCE)
                  + lc.PROVENANCE_ALLOWANCE + manifest_cap)
    write_chunks = 2 * min(lc.NPY_WRITE_CHUNK_BYTES, largest)
    io = lc.IO_CHUNK_BYTES + lc.HASH_CHUNK_BYTES
    material_stage = (lc.HASH_CHUNK_BYTES + OBJECT_BYTES) if bed is not None else 0
    role_res, _extras = _role_resident(lop, role, event_extra_bytes)
    resident_old = role_res + resident_native
    resident_new = role_res + resident_native
    # --- save: the live current role (+ native leaves) beside capture, validators/writers, metadata, I/O, source audit, material hash
    save_transient = max(validation, companion + max(voxel_scan + class_scan + ledger_scan, write_chunks, SOURCE_AUDIT_BYTES,
                                                     material_stage) + metadata + io)
    # --- load: old caller resident beside each phase's transient
    native_objects = members_native * (lc.ZIP_INFO_OBJECT_BYTES + 3 * lc.NATIVE_STRING_MAX) + 3 * lc.NATIVE_PROVENANCE_MAX
    native_load = native + (native_arrays.get("active_mass_kg", 0) if bed is not None else 0) + 2 * lc.NPY_READ_BUFFER_BYTES \
        + ((N * K + N + native_objects) if bed is not None else 0)
    decode = native + companion + largest + 2 * lc.NPY_READ_BUFFER_BYTES
    new_state = (L * N + 2 * N) * 8 + (0 if bed is not None else N * 8)  # soil + history copies (+ bedless surface)
    retained_metadata = manifest_nodes * lc.NODE_BYTES + manifest_cap
    prep_build = lop.memory.storm["build_bytes"] + lop.memory.storm["initialization_bytes"] + lop.memory.owned_fixed_bytes
    if role == "interevent":
        iwm, gm = _extras["iw"], _extras["gm"]
        prep_build = max(prep_build, iwm.build_bytes + iwm.required_bytes, gm.supplied_prepare_required_bytes,
                         gm.audit_required_bytes)
    quiet_scratch = 0
    if pol.v2 and role == "interevent":  # immediate-entry quiet-evidence recheck (read-only) beside the restored arrays
        quiet_scratch = ie.QUIET_SCRATCH_DOUBLES_PER_CELL * 8 * N + 2 * 8 * lop.memory.n_faces
        prep_build = max(prep_build, quiet_scratch + lop.memory.audit_scratch_bytes)
    if role == "boundary":  # the immediate-reset revalidation (read-only quiet evidence) is live beside the restored arrays
        prep_build = max(prep_build, lop.memory.reset_scratch_bytes)
    restore = (native + companion + new_state + role_res + retained_metadata + material_stage
               + max(validation, prep_build + lc.HASH_CHUNK_BYTES))
    pre_payload = metadata + SOURCE_AUDIT_BYTES + io
    load_transient = max(pre_payload, native_load + metadata + material_stage, decode + metadata, restore)
    scratch = max(max(voxel_scan + class_scan + ledger_scan, write_chunks, SOURCE_AUDIT_BYTES, material_stage) + metadata + io,
                  validation + lc.HASH_CHUNK_BYTES + 2 * lc.NPY_READ_BUFFER_BYTES + metadata + SOURCE_AUDIT_BYTES,
                  prep_build + lc.HASH_CHUNK_BYTES + retained_metadata)
    terms = {"resident_role": role_res, "resident_native": resident_native, "voxel_scan": voxel_scan, "class_scan": class_scan,
             "ledger_scan": ledger_scan, "owner_audit": owner_audit, "audit_scratch": lop.memory.audit_scratch_bytes,
             "capture": companion, "writer_chunks": write_chunks, "native_load": native_load, "decode": decode,
             "restore": restore, "metadata": metadata, "retained_metadata": retained_metadata,
             "source_audit": SOURCE_AUDIT_BYTES, "material_stage": material_stage, "prep_build": prep_build,
             "constructor_copies": new_state, "extra_reserve": extra}
    if pol.snapshot_census is not None:
        terms["native_census"] = pol.census_bytes
    if pol.v2:
        terms.update(source_envelope=3 * pol.envelope_bytes + pol.envelope_nodes * lc.NODE_BYTES, quiet_scratch=quiet_scratch,
                     manifest_envelope_bytes=pol.envelope_bytes, manifest_envelope_nodes=pol.envelope_nodes)
    return LifecycleBudget(profile, role, N, L, K, V, native, companion, members_native, members_comp, manifest_cap,
                           manifest_nodes, serialized, metadata, scratch, resident_old, resident_new,
                           resident_old + save_transient + extra, resident_old + load_transient + extra, 0, terms)


# --- tagged records ----------------------------------------------------------------------------------------------------------------
def _storm_record(lop: Any, state: Any) -> dict:
    """The bounded scalar record of the genuine storm event state (EXACT floats), plus the event identities."""
    e = state.payload
    event = state.binding.event
    w, q = e.water, e.quiet
    body = {"generation": int(w.generation), "event_sha256": e.event_sha256, "ledger": dataclasses.asdict(w.ledger),
            "anchor_ledger": dataclasses.asdict(e.anchor_ledger), "anchor_surface_m3": float(e.anchor_surface_m3),
            "anchor_soil_m3": float(e.anchor_soil_m3), "start_generation": int(e.start_generation),
            "next_dt_cap_s": float(e.next_dt_cap_s), "forcing_index": int(e.forcing_index),
            "accepted_steps": int(e.accepted_steps), "rejected_attempts": int(e.rejected_attempts),
            "quiet": {"since_s": None if q.since_s is None else float(q.since_s),
                      "window_depth_change_m": float(q.window_depth_change_m)},
            "completed": bool(e.completed), "event_origin_s": float(event.event_origin_s),
            "rainfall_end_s": float(event.rainfall_end_s), "deadline_s": float(event.deadline_s)}
    return cc.encode_tree(body)


def _scalar_records(lop: Any, state: Any, pol: "_Policy | None" = None) -> dict:
    pol = _policy_of(pol)
    out = {"lifecycle": wl.checkpoint_state_record(lop, state)}
    if state.role == "storm":
        out["storm"] = _storm_record(lop, state)
    elif state.role == "interevent":
        out["t_state"] = it.checkpoint_state_record(state.payload)
        if pol.v2:
            out["source"] = iw.checkpoint_source_record_v2(state.payload.source)
        else:
            out["receipt"] = iw.checkpoint_receipt_record(state.payload.source)
    return out


def _state_source_kind(pol: "_Policy", state: Any) -> str | None:
    """v2: the authenticated kind of the CURRENT interevent source (registry-checked by the IW owner); None for every other role / for v1."""
    if not pol.v2 or state.role != "interevent":
        return None
    try:
        return iw.source_kind(state.payload.source)
    except iw.InterEventWaterError as exc:
        raise _convert(exc) from None


def _current_arrays(state: Any) -> SimpleNamespace:
    return wl._arrays_of(state)


def physical_digest(lop: Any, state: Any) -> str:
    """SHA-256 over every physical value and scalar of the current role (streamed, bounded staging): raw arrays, native leaves,
    exact clocks/counters, both ledgers' scalars and the owner digests. For receipts/tests."""
    a = _current_arrays(state)
    h = hashlib.sha256(b"maple_syrup.lifecycle_checkpoint.physical/1")
    for value in (state.role, float(state.t_s), int(state.revision), int(state.episode), int(state.reset_count),
                  int(state.request_sequence), state.lop_sha256, str(state.storm_operator_sha256), state.digest):
        lc._digest_value(h, value)
    lc._digest_value(h, state.lifetime)
    lc._digest_value(h, state.mark)
    for array in (a.soil.water_m, a.history.wetting_m, a.history.cumulative_intake_m, a.surface_m, a.qx, a.qy):
        lc._digest_value(h, array)
    if a.bed is not None:
        for part in (a.bed.voxel_column, a.bed.active_layer, a.bed.sediment_availability, a.bed.water, a.bed.ledger,
                     a.bed.committed_topography):
            lc._digest_value(h, part)
    return h.hexdigest()


# --- files -------------------------------------------------------------------------------------------------------------------------
def _read_manifest(dirfd: int, cap_bytes: int, max_nodes: int) -> dict:
    """The manifest under its derived policy (bounded read, structural scan BEFORE `json.loads`, strict parse)."""
    fd = lc._open_regular(MANIFEST_NAME, dirfd, cap_bytes)
    try:
        raw = b""
        while len(raw) <= cap_bytes:
            chunk = os.read(fd, lc.IO_CHUNK_BYTES)
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(fd)
    if len(raw) > cap_bytes:
        _refuse("manifest exceeds its derived byte bound")
    try:
        lc._scan_json(raw, lc.MANIFEST_MAX_DEPTH, max_nodes, lc.MANIFEST_MAX_STRING)
    except lc.LayeredCheckpointError as exc:
        raise LifecycleCheckpointError(str(exc)) from None

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
        raise LifecycleCheckpointError(f"manifest is not strict JSON: {exc}") from None
    if not isinstance(data, dict):
        _refuse("manifest must be a JSON object")
    return data


def _lstat_identity(path: Any) -> tuple[int, int] | None:
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return (info.st_dev, info.st_ino) if stat.S_ISDIR(info.st_mode) else None


def _rmdir_owned(path: Any, ident: tuple[int, int] | None) -> None:
    """Remove ONLY an EMPTY directory whose current (dev, inode) equals the identity this call recorded for it (`rmdir` itself
    refuses a non-empty directory, so a foreign file inside is never touched). Unknown identity: nothing is removed."""
    if ident is None:
        return
    try:
        if _lstat_identity(path) == ident:
            os.rmdir(path)
    except OSError:
        pass


def _write_companion(dirfd: int, res: Any, host: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(PAYLOAD_COMPANION, flags, 0o600, dir_fd=dirfd)
    res.created[PAYLOAD_COMPANION] = lc._identity_of(fd)
    with os.fdopen(fd, "wb") as handle:
        np.savez(handle, **host)
        handle.flush()
        _fsync(handle.fileno())


def _admit_whole(budget: LifecycleBudget, limits: Any, host_need: int, host_fn: Any, device_fn: Any, disk_fn: Any,
                 disk_need: int, backend: str) -> None:
    lc._admit_all(budget, limits, host_need=host_need, host_fn=host_fn, device_fn=device_fn, disk_fn=disk_fn,
                  disk_need=disk_need, op_backend=backend)


def _wrap_quota(fn: Callable, *args: Any, **kw: Any) -> Any:
    try:
        return fn(*args, **kw)
    except lc.LayeredCheckpointQuotaError as exc:
        raise LifecycleCheckpointQuotaError(str(exc)) from None
    except lc.LayeredCheckpointError as exc:
        raise LifecycleCheckpointError(str(exc)) from None


# --- save --------------------------------------------------------------------------------------------------------------------------
def save_lifecycle_checkpoint(path: Any, lop: Any, state: Any, identity: dict, *, limits: Any,
                              host_available_bytes_fn: Callable[[], int], device_available_bytes_fn: Any = None,
                              disk_available_bytes_fn: Callable[[], int] | None = None,
                              extra_reserve_bytes: int = 0) -> LifecycleSaveReceipt:
    """Write a NEW sealed checkpoint directory for the accepted `state` (one current role); see `_save`. Errors of the shared
    generic helpers are re-raised as this module's typed errors."""
    try:
        return _save(path, lop, state, identity, limits=limits, host_available_bytes_fn=host_available_bytes_fn,
                     device_available_bytes_fn=device_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn,
                     extra_reserve_bytes=extra_reserve_bytes)
    except LifecycleCheckpointError:
        raise
    except lc.LayeredCheckpointQuotaError as exc:
        raise LifecycleCheckpointQuotaError(str(exc)) from None
    except lc.LayeredCheckpointError as exc:
        raise LifecycleCheckpointError(str(exc)) from None


def _save(path: Any, lop: Any, state: Any, identity: dict, *, limits: Any, host_available_bytes_fn: Callable[[], int],
          device_available_bytes_fn: Any = None, disk_available_bytes_fn: Callable[[], int] | None = None,
          extra_reserve_bytes: int = 0, _policy: "_Policy | None" = None, _profile: str | None = None) -> LifecycleSaveReceipt:
    """Write a NEW sealed checkpoint directory for the accepted `state` (one current role). Order: type/profile fences (before any
    native/deep/device work) -> metadata-only whole-call admission -> the full lifecycle audit and genuine native validators ->
    bounded capture -> exclusive reservation, native writer, companion, descriptor-bound header/hash audits -> manifest-last
    no-clobber seal. No commit, reroute, advance, reset or request happens; the accepted state is never touched."""
    from maple.io.outputs import snapshot as native_snapshot

    pol = _policy_of(_policy)
    if type(lop) is not wl.LifecycleOperator:
        _refuse(f"lop must be a genuine LifecycleOperator, got {type(lop).__name__}")
    if type(state) is not wl.LifecycleState:
        _refuse("state must be a genuine accepted LifecycleState")
    try:
        wl._check_state(lop, state)  # owner-side genuineness (minted, scalar digest, bindings); metadata only, before the budget
    except wl.LifecycleError as exc:
        raise _convert(exc) from None
    bed = _current_arrays(state).bed if lop.context is not None else None
    profile = _fence(lop, bed, pol)
    if _profile is not None and _profile != profile:
        _refuse(f"the requested profile {_profile!r} does not match this lifecycle ({profile!r})")
    if not isinstance(limits, lc.CheckpointLimits):
        _refuse("limits must be a CheckpointLimits record")
    limits.validated()
    try:
        ident = lc._canonical_identity(identity)
    except lc.LayeredCheckpointError as exc:
        raise _convert(exc) from None
    role = state.role
    event_extra = 0
    if role == "storm":
        event_extra = max(int(state.binding.event.required_bytes) - int(state.binding.op.memory.required_bytes), 0)
    kind = _state_source_kind(pol, state)
    budget = _estimate_budget(lop, role, bed=bed, event_extra_bytes=event_extra, extra_reserve_bytes=extra_reserve_bytes, _policy=pol,
                              source_kind=kind)
    target = Path(os.path.abspath(path))  # absolute but NOT symlink-resolved: a linked final parent must stay visible
    if os.path.lexists(target):
        _refuse(f"checkpoint path already exists (no overwrite): {target}")
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink():
        _refuse(f"the parent directory does not exist or is a link: {parent}")
    _wrap_quota(_admit_whole, budget, limits, budget.host_save_bytes, host_available_bytes_fn, device_available_bytes_fn,
                lc._disk_reader(parent, disk_available_bytes_fn), budget.serialized_bytes, lop.backend)
    # fresh actual source pair (and the literal native39 fence) AFTER admission and BEFORE any native validator/writer
    source = _fresh_source_pair(profile, pol)
    # --- the same physical validation the loader runs, BEFORE anything is written
    try:
        wl.audit_lifecycle(lop, state)
        if bed is not None:
            lc._native_checks(bed, lop.context, float(state.t_s), type(bed.ledger))
        if role == "storm":
            ev.audit_event(state.binding.event, state.payload)
        elif role == "interevent":
            it.audit_general_interevent(state.binding.gop, state.payload)
    except (wl.LifecycleError, ev.LayeredEventError, ls.LayeredStormError, it.GeneralIntereventError) as exc:
        raise _convert(exc) from None
    except lc.LayeredCheckpointError as exc:
        raise LifecycleCheckpointError(str(exc)) from None
    a = _current_arrays(state)
    records = _scalar_records(lop, state, pol)
    host: dict[str, np.ndarray] = {}
    table: dict[str, dict] = {}

    def add(name: str, array: Any) -> str:
        arr = np.array(np.asarray(array), copy=True, order="C")
        if not pc._dtype_ok(arr.dtype.str) or arr.ndim > 5:
            _refuse(f"{name}: dtype {arr.dtype.str} / ndim {arr.ndim} not allowed")
        host[name] = arr
        table[name] = {"dtype": arr.dtype.str, "shape": [int(n) for n in arr.shape]}
        return name

    add("soil_water_m", a.soil.water_m)
    add("wetting_m", a.history.wetting_m)
    add("cumulative_intake_m", a.history.cumulative_intake_m)
    add("qx_m2_s", a.qx)
    add("qy_m2_s", a.qy)
    add("topo_source_full", lop.topology.source_z_full)
    native: dict = {}
    if bed is None:
        add("surface_m", a.surface_m)
    else:
        native = lc._encode_native(bed, lambda arr: add(f"native_{len(host):04d}", arr))
    material = None if bed is None else _material_sha(bed, lop.context)
    binding = _binding(lop, profile, role, bed, material, pol, kind)
    runtime = lc.runtime_profile(lop.backend)
    manifest = {"schema": pol.schema, "profile": profile, "role": role, "identity": ident, "array_backend": "numpy",
                "hydraulic_backend": lop.backend, "source": source, "runtime": runtime, "binding": binding,
                "binding_sha256": lc._sha_text(lc._json_dumps(binding)), "files": {}, "sizes": {},
                "payload": {"scalars": records, "arrays": table, "native": native}}
    if pol.v2:
        manifest["source_kind"] = kind
    try:
        lc._json_dumps(manifest)
    except (TypeError, ValueError) as exc:
        raise LifecycleCheckpointError(f"manifest is not strict JSON: {exc}") from None
    provenance = {"syrup_lifecycle_checkpoint_schema": pol.schema, "profile": profile, "role": role, "requires_companion": True,
                  "identity": ident, "binding_sha256": manifest["binding_sha256"], "array_backend": "numpy",
                  "hydraulic_backend": lop.backend, "runtime": runtime}
    if pol.v2:
        provenance["source_kind"] = kind
    digest = physical_digest(lop, state)
    shim = _shim(lop, bed)
    specs = {PAYLOAD_COMPANION: lc._companion_expected(shim, table)}
    if profile == pol.native_profile:
        specs[PAYLOAD_NATIVE] = lc._native_snapshot_specs(shim)

    flags_dir = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        pfd = os.open(parent, flags_dir)
    except OSError as exc:
        raise LifecycleCheckpointError(f"cannot open the parent directory: {exc}") from None
    res = None
    made_id = None
    dirfd = None
    try:
        try:
            _mkdir(target, 0o700)
        except FileExistsError:
            raise LifecycleCheckpointError(f"checkpoint path already exists (no overwrite): {target}") from None
        except OSError as exc:
            raise LifecycleCheckpointError(f"cannot reserve the checkpoint directory: {exc}") from None
        made_id = _lstat_identity(target)  # the directory THIS call just created, by (dev, inode)
        try:
            dirfd = os.open(target, flags_dir)
        except OSError as exc:
            raise LifecycleCheckpointError(f"cannot open the reserved directory: {exc}") from None
        res = lc._Reservation(target, dirfd, lc._identity_of(dirfd))
    except BaseException:
        if res is None:  # ownership never reached the reservation: close the descriptor and remove ONLY the own empty directory
            if dirfd is not None:
                os.close(dirfd)
            _rmdir_owned(target, made_id)
        os.close(pfd)
        raise
    sealed = False
    tmp_name = f".manifest.tmp-{os.getpid()}-{os.urandom(6).hex()}"
    try:
        _fsync(pfd)
        if os.listdir(res.dirfd):
            _refuse("the reserved checkpoint directory is not empty (a foreign writer raced the reservation)")
        dirfd = res.dirfd
        files = pol.files[profile]
        if profile == pol.native_profile:
            res.stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=parent))
            res.stage_id = _lstat_identity(res.stage)  # recorded BEFORE the descriptor open so a failed open stays cleanable
            res.stage_fd = os.open(res.stage, flags_dir)
            if lc._identity_of(res.stage_fd) != res.stage_id:
                _refuse("the private native stage directory changed identity")
            sfd = os.open(PAYLOAD_NATIVE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600,
                          dir_fd=res.stage_fd)
            try:
                staged_id = lc._identity_of(sfd)
            finally:
                os.close(sfd)
            res.stage_created[PAYLOAD_NATIVE] = staged_id
            native_snapshot.save_state_snapshot(
                bed.voxel_column, bed.active_layer, float(state.t_s), res.stage / PAYLOAD_NATIVE, step=int(state.revision),
                reason=lc._SNAPSHOT_REASON, geometry=lop.context.geometry, mass_resolution_kg=lop.context.mass_resolution_kg,
                sediment_availability=bed.sediment_availability, water=bed.water, provenance=provenance)
            if lc._identity_of(PAYLOAD_NATIVE, dir_fd=res.stage_fd) != staged_id:
                _refuse("the staged native payload changed identity during the native write")
            lc._fsync_name(PAYLOAD_NATIVE, res.stage_fd)
            res.check_bundle_identity()
            try:
                _link(PAYLOAD_NATIVE, PAYLOAD_NATIVE, src_dir_fd=res.stage_fd, dst_dir_fd=dirfd, follow_symlinks=False)
            except FileExistsError:
                raise LifecycleCheckpointError(
                    "the bundle already holds a file at the native payload name (a foreign writer raced); nothing was "
                    "overwritten") from None
            res.created[PAYLOAD_NATIVE] = staged_id
            os.unlink(PAYLOAD_NATIVE, dir_fd=res.stage_fd)
            res.stage_created.pop(PAYLOAD_NATIVE, None)
            os.rmdir(res.stage)
        res.check_bundle_identity()
        _write_companion(dirfd, res, host)
        del host
        sizes, hashes = {}, {}
        for name in files:  # post-write audit through bound descriptors (never a path)
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
            raise LifecycleCheckpointQuotaError(f"written payload {sum(sizes.values())} B exceeds the serialized bound")
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
            _fsync(handle.fileno())
        _fsync(dirfd)
        res.check_bundle_identity()
        try:
            _link(tmp_name, MANIFEST_NAME, src_dir_fd=dirfd, dst_dir_fd=dirfd, follow_symlinks=False)  # atomic no-clobber seal
        except FileExistsError:
            raise LifecycleCheckpointError("the seal name already exists (a foreign writer raced); nothing was overwritten") \
                from None
        sealed = True
        os.unlink(tmp_name, dir_fd=dirfd)
        res.created.pop(tmp_name, None)
        _fsync(dirfd)
        _fsync(pfd)
    except BaseException as exc:
        if sealed:
            raise LifecycleCheckpointIndeterminate(
                f"failure after the seal link ({type(exc).__name__}: {exc}); the bundle at {target} may be durable: nothing "
                "was deleted and it must not be retried over") from exc
        res.remove_own()
        if res.stage is not None and res.stage_fd is None:  # the stage was created but its descriptor never opened
            _rmdir_owned(res.stage, res.stage_id)
        if isinstance(exc, LifecycleCheckpointError):
            raise
        if isinstance(exc, lc.LayeredCheckpointQuotaError):
            raise LifecycleCheckpointQuotaError(str(exc)) from exc
        if isinstance(exc, Exception):
            raise LifecycleCheckpointError(f"save failed before the seal: {type(exc).__name__}: {exc}") from exc
        raise
    finally:
        res.close()
        os.close(pfd)
    return LifecycleSaveReceipt(str(target), profile, role, sealed_manifest["manifest_sha256"], dict(hashes), dict(sizes),
                                float(state.t_s), int(state.revision), state.digest, digest, budget.as_dict(), dict(source))


# --- load --------------------------------------------------------------------------------------------------------------------------
def load_lifecycle_checkpoint(path: Any, lop: Any, expected_identity: dict, *, expected_role: str, limits: Any,
                              host_available_bytes_fn: Callable[[], int], reference_bed: Any = None, storm: Any = None,
                              device_available_bytes_fn: Any = None, disk_available_bytes_fn: Callable[[], int] | None = None,
                              extra_reserve_bytes: int = 0) -> LifecycleRestoration:
    """Restore a sealed checkpoint against the caller's EXPECTED fixed components (`lop`, and for Dense the caller's live genuine
    `reference_bed`, used only for dimensions/ledger class/native content identity) and EXPECTED role/identity (and, for a storm,
    the expected `StormSpec`). Returns a NEW lifecycle operator prepared on the LOADED bed and a state bound to it; nothing is
    adopted. Every refusal leaves all inputs, the caller's operator and the files unchanged."""
    return _load_path(path, lop, expected_identity, expected_role=expected_role, limits=limits,
                      host_available_bytes_fn=host_available_bytes_fn, reference_bed=reference_bed, storm=storm,
                      device_available_bytes_fn=device_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn,
                      extra_reserve_bytes=extra_reserve_bytes)


def _load_path(path: Any, lop: Any, expected_identity: dict, *, expected_role: str, limits: Any,
               host_available_bytes_fn: Callable[[], int], reference_bed: Any = None, storm: Any = None,
               device_available_bytes_fn: Any = None, disk_available_bytes_fn: Callable[[], int] | None = None,
               extra_reserve_bytes: int = 0, _policy: "_Policy | None" = None, _profile: str | None = None,
               expected_source_kind: str | None = None) -> LifecycleRestoration:
    """The whole former body of `load_lifecycle_checkpoint` (fences, limits, identity, path, whole-call admission, fresh source audit,
    directory descriptor, private stable directory, manifest, payload, typed conversion, cleanup) under a private registered policy
    (`None` = the v1 default). v2 reaches the payload-level `_load` ONLY through here, so its admitted stable-FD lifetime is identical."""
    pol = _policy_of(_policy)
    from maple.io.outputs import snapshot as native_snapshot

    profile = _fence(lop, reference_bed, pol)
    if _profile is not None and _profile != profile:
        _refuse(f"the requested profile {_profile!r} does not match this lifecycle ({profile!r})")
    if not isinstance(limits, lc.CheckpointLimits):
        _refuse("limits must be a CheckpointLimits record")
    limits.validated()
    if expected_role not in ROLES:
        _refuse(f"expected_role must be one of {ROLES}")
    if (storm is None) != (expected_role != "storm"):
        _refuse("a StormSpec is required for (and only for) the 'storm' role")
    if pol.v2:
        if (expected_role == "interevent") != (expected_source_kind is not None) or (
                expected_source_kind is not None and expected_source_kind not in pol.source_kinds):
            _refuse("a v2 interevent load needs a closed expected_source_kind and every other role must give none")
    elif expected_source_kind is not None:
        _refuse("the v1 codec has no source kind")
    event_extra = 0
    if storm is not None:
        if type(storm) is not StormSpec:
            _refuse("storm must be a StormSpec")
        try:
            event_extra = wl._event_extra_bytes(lop, storm.schedule)
        except (AttributeError, TypeError) as exc:
            raise LifecycleCheckpointError(f"invalid storm schedule: {exc}") from None
    try:
        expected = lc._canonical_identity(expected_identity)
    except lc.LayeredCheckpointError as exc:
        raise _convert(exc) from None
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _refuse(f"checkpoint path is not a plain directory: {root}")
    budget = _estimate_budget(lop, expected_role, bed=reference_bed, event_extra_bytes=event_extra,
                              extra_reserve_bytes=extra_reserve_bytes, _policy=pol, source_kind=expected_source_kind)
    _wrap_quota(_admit_whole, budget, limits, budget.host_load_bytes, host_available_bytes_fn, device_available_bytes_fn,
                lc._disk_reader(Path(tempfile.gettempdir()), disk_available_bytes_fn), budget.serialized_bytes, lop.backend)
    source = _fresh_source_pair(profile, pol)  # fresh actual pair + native39 fence AFTER admission, BEFORE any native reader
    try:
        dirfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise LifecycleCheckpointError(f"cannot open the checkpoint directory: {exc}") from None
    private = None
    try:
        try:
            private = Path(tempfile.mkdtemp(prefix="lifecycle-checkpoint-stable-"))
        except OSError as exc:
            raise LifecycleCheckpointError(f"cannot create the private stable directory: {exc}") from None
        names = set(os.listdir(dirfd))
        if MANIFEST_NAME not in names:
            _refuse("unsealed or partial checkpoint directory (no sealed manifest)")
        if names != {MANIFEST_NAME, *pol.files[profile]}:
            _refuse(f"checkpoint directory must contain exactly {sorted({MANIFEST_NAME, *pol.files[profile]})}, found {sorted(names)}")
        data = _read_manifest(dirfd, budget.manifest_cap_bytes, budget.manifest_max_nodes)
        return _load(data, dirfd, private, lop, reference_bed, storm, expected, expected_role, profile, budget, limits,
                     native_snapshot, source, pol, expected_source_kind)
    except LifecycleCheckpointError:
        raise
    except lc.LayeredCheckpointQuotaError as exc:
        raise LifecycleCheckpointQuotaError(str(exc)) from None
    except lc.LayeredCheckpointError as exc:
        raise LifecycleCheckpointError(str(exc)) from None
    except Exception as exc:
        raise LifecycleCheckpointError(f"load refused: {type(exc).__name__}: {exc}") from exc
    finally:
        os.close(dirfd)
        if private is not None:
            shutil.rmtree(private, ignore_errors=True)


def _load(data: dict, dirfd: int, private: Path, lop: Any, ref_bed: Any, storm: Any, expected: dict, role: str, profile: str,
          budget: LifecycleBudget, limits: Any, native_snapshot: Any, source: dict, pol: "_Policy | None" = None,
          expected_kind: str | None = None) -> LifecycleRestoration:
    pol = _policy_of(pol)
    # -- manifest and every binding BEFORE any payload is opened ---------------------------------------------------------------
    keys = pol.manifest_keys
    if set(data) != keys:
        _refuse(f"manifest keys mismatch: missing {sorted(keys - set(data))}, unknown {sorted(set(data) - keys)}")
    if data["schema"] != pol.schema:
        _refuse(f"checkpoint schema {data['schema']!r} is not {pol.schema!r}")
    if lc._sha(data["manifest_sha256"], "manifest_sha256") != pc._seal_manifest(data)["manifest_sha256"]:
        _refuse("manifest hash mismatch (the manifest was modified)")
    if data["profile"] != profile:
        _refuse(f"profile {data['profile']!r} does not match the expected {profile!r}; profiles never cross-load")
    if data["role"] != role:
        _refuse(f"saved role {data['role']!r} is not the expected role {role!r}")
    if pol.v2:  # the closed (role, source_kind) pair BEFORE identity/source/binding/payload expansion
        if (role, data["source_kind"]) not in pol.scalar_parts or data["source_kind"] != expected_kind:
            _refuse(f"saved source kind {data['source_kind']!r} is not the expected {expected_kind!r} for role {role!r}")
    if data["identity"] != expected:
        _refuse("checkpoint identity differs from the expected identity")
    if data["source"] != source:
        _refuse("source pair mismatch: the imported maple_syrup / maple packages differ from the writer's (refused before any payload)")
    if data["array_backend"] != "numpy" or data["hydraulic_backend"] != lop.backend:
        _refuse("backend mismatch: no cross-backend migration (numpy and numba are not interchangeable)")
    lc._check_runtime_record(data["runtime"], lop.backend)
    material = None if ref_bed is None else _material_sha(ref_bed, lop.context)
    binding = _binding(lop, profile, role, ref_bed, material, pol, expected_kind)
    if data["binding"] != binding or data["binding_sha256"] != lc._sha_text(lc._json_dumps(binding)):
        diff = sorted(k for k in binding if not isinstance(data["binding"], dict) or data["binding"].get(k) != binding[k])
        _refuse(f"binding mismatch (source/runtime/context/properties/material differ from the expected): {diff}")
    payload = data["payload"]
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        _refuse("payload keys mismatch")
    scalars = payload["scalars"]
    if not isinstance(scalars, dict) or set(scalars) != pol.parts(role, expected_kind if pol.v2 else None):
        _refuse("scalar record parts differ from the role's closed set")
    files, sizes = data["files"], data["sizes"]
    names = pol.files[profile]
    if not isinstance(files, dict) or set(files) != set(names) or not isinstance(sizes, dict) or set(sizes) != set(names):
        _refuse("payload file set mismatch")
    declared = sum(lc._int(sizes[n], f"sizes[{n}]") for n in names)
    if declared > limits.max_serialized_bytes or declared > budget.serialized_bytes:
        raise LifecycleCheckpointQuotaError(f"declared payload {declared} B exceeds the serialized limit/derived bound")
    shim = _shim(lop, ref_bed)
    table = payload["arrays"]
    fixed = lc._companion_specs(shim)
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
    if profile == pol.native_profile:
        ref_leaves = sorted(lc._leaf_specs(ref_bed.ledger) + lc._leaf_specs(ref_bed.committed_topography))
        got = sorted((table[n]["dtype"], tuple(table[n]["shape"])) for n in native_names)
        if got != ref_leaves:
            _refuse("declared native ledger/committed leaves differ from the reference bed's (class, shapes, present processes)")
    elif native_names or payload["native"]:
        _refuse("the bedless profile carries no native components")
    # -- exact scalar identity of the clock BEFORE any payload expansion ----------------------------------------------------------
    try:
        life = cc.decode_tree(scalars["lifecycle"])
        t_s, revision = life["t_s"], life["revision"]
    except (cc.CheckpointCapabilityError, KeyError, TypeError) as exc:
        raise LifecycleCheckpointError(f"lifecycle scalar record refused: {exc}") from None
    if type(t_s) is not float or t_s < 0.0 or type(revision) is not int or revision < 0:
        _refuse("lifecycle clock/revision must be an exact non-negative float / integer")
    # -- stable private copies, hashed through the open descriptors -----------------------------------------------------------------
    stable = {n: lc._stable_copy(n, dirfd, private, lc._sha(files[n], f"files[{n}]"), int(sizes[n]), limits.max_serialized_bytes)
              for n in names}
    if profile == pol.native_profile:
        lc._audit_path(stable[PAYLOAD_NATIVE], lc._native_snapshot_specs(shim), label=PAYLOAD_NATIVE,
                       byte_limit=limits.max_serialized_bytes)
    lc._audit_path(stable[PAYLOAD_COMPANION], lc._companion_expected(shim, table), label=PAYLOAD_COMPANION,
                   byte_limit=limits.max_serialized_bytes)
    snapshot = None
    if profile == pol.native_profile:
        try:
            snapshot = native_snapshot.load_state_snapshot(stable[PAYLOAD_NATIVE])
        except Exception as exc:
            raise LifecycleCheckpointError(f"MAPLE snapshot cannot be loaded: {type(exc).__name__}: {exc}") from None
        _check_snapshot(snapshot, data, t_s, revision, pol)
    arrays, ledger_node, committed_node = _decode_companion(stable[PAYLOAD_COMPANION], table, payload["native"], profile, pol)
    lc._check_topology(arrays["topo_source_full"], lop.topology, data["binding"]["topology"])
    return _restore(lop, ref_bed, storm, data, scalars, arrays, snapshot, ledger_node, committed_node, budget, t_s, pol)


def _check_snapshot(snapshot: dict, data: dict, t_s: float, revision: int, pol: "_Policy | None" = None) -> None:
    pol = _policy_of(pol)
    from maple.io.outputs.snapshot import SNAPSHOT_SCHEMA_VERSION
    from maple.io.outputs.visualization_schema import SCHEMA_VERSION as HISTORY_SCHEMA

    if pol.snapshot_census is not None:  # native9f: the EXACT ordinary reader return census, with both unsupported-state keys explicitly None
        got = set(snapshot)
        if got != pol.snapshot_census:
            _refuse(f"the {pol.native_label} snapshot reader returned an unexpected key set (missing {sorted(pol.snapshot_census - got)}, "
                    f"unexpected {sorted(got - pol.snapshot_census)}): conditional fixed-base / regional-source admission keys or a changed "
                    "reader interface are not part of the ordinary profile; refused")
        for key in ("fixed_base", "regional_slope_wind_source"):
            if snapshot[key] is not None:
                _refuse(f"the {pol.native_label} snapshot carries unsupported native state {key!r} (it must be explicitly None); refused")
    if not snapshot["validated"] or not snapshot["restartable"] or snapshot["is_diagnostic"]:
        _refuse("MAPLE snapshot is not validated/restartable or is diagnostic (flags are checks, never proof)")
    if snapshot["snapshot_schema_version"] != SNAPSHOT_SCHEMA_VERSION or snapshot["history_schema_version"] != HISTORY_SCHEMA:
        _refuse("MAPLE snapshot schema/history semantic identity is not the exact supported one")
    try:
        prov = json.loads(snapshot["provenance_json"])
    except ValueError:
        prov = None
    if (not isinstance(prov, dict) or prov.get("syrup_lifecycle_checkpoint_schema") != pol.schema or prov.get("profile") != pol.native_profile
            or (pol.v2 and prov.get("source_kind") != data["source_kind"])
            or prov.get("role") != data["role"] or prov.get("binding_sha256") != data["binding_sha256"]
            or prov.get("identity") != data["identity"] or prov.get("array_backend") != data["array_backend"]
            or prov.get("hydraulic_backend") != data["hydraulic_backend"] or prov.get("requires_companion") is not True
            or prov.get("runtime") != data["runtime"]):
        _refuse("MAPLE snapshot provenance is not bound to this manifest")
    if (snapshot["water"] is None or snapshot["sediment_availability"] is None or snapshot["history_store_path"]
            or snapshot["reason"] != lc._SNAPSHOT_REASON or any(snapshot.get(k) is not None for k in lc._FOREIGN_SNAPSHOT_FIELDS)
            or snapshot.get("sediment_availability_fallback_sha256") or snapshot.get("scheduler_coupled_status")
            or snapshot.get("scheduler_wind_record_source_sha256") or snapshot["run_name"] or snapshot["error_type"]
            or snapshot["error_message"] or snapshot["seed"] != 0):
        _refuse("MAPLE snapshot lacks water/availability or carries foreign optional state")
    if snapshot["time_s"] != t_s or snapshot["step"] != revision:
        _refuse("MAPLE snapshot clock/step differ from the lifecycle record")


def _decode_companion(path: Path, table: dict, native: Any, profile: str, pol: "_Policy | None" = None) -> tuple[dict, Any, Any]:
    pol = _policy_of(pol)
    used: set[str] = set()
    arrays: dict[str, np.ndarray] = {}
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise LifecycleCheckpointError(f"companion cannot be opened: {type(exc).__name__}: {exc}") from None
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
                arr = np.array(archive[name], copy=True, order="C")
            except Exception as exc:
                raise LifecycleCheckpointError(f"array {name} cannot be read: {type(exc).__name__}: {exc}") from None
            if arr.dtype.str != rec["dtype"] or list(arr.shape) != rec["shape"]:
                _refuse(f"array {name}: stored dtype/shape differ from the table")
            if arr.dtype.kind == "f" and not np.isfinite(arr).all():
                _refuse(f"array {name} holds non-finite values")
            return arr

        names = ("soil_water_m", "wetting_m", "cumulative_intake_m", "qx_m2_s", "qy_m2_s", "topo_source_full") \
            + (("surface_m",) if profile == pol.bedless_profile else ())
        for name in names:
            arrays[name] = get(name)
        ledger_node = committed_node = None
        if profile == pol.native_profile:
            if not isinstance(native, dict) or set(native) != {"ledger", "committed_topography"}:
                _refuse("native companion keys mismatch")
            try:
                ledger_node = pc._native_decode(native["ledger"], get)
                committed_node = pc._native_decode(native["committed_topography"], get)
            except lc.LayeredCheckpointError:
                raise
            except Exception as exc:
                raise LifecycleCheckpointError(f"{type(exc).__name__}: {exc}") from None
    if used != set(table):
        _refuse("unused, missing or duplicate companion array references")
    return arrays, ledger_node, committed_node


def _storm_state_from(record: Any, event: Any, water: Any) -> Any:
    tree = cc.decode_tree(record)
    if type(tree) is not dict or set(tree) != STORM_RECORD_KEYS:
        _refuse("storm record keys mismatch (closed key set)")

    def ledger(rec: Any, name: str) -> Any:
        try:
            return it._typed_ledger(ls.LayeredWaterLedger, rec, name)
        except it.GeneralIntereventError as exc:
            raise _convert(exc) from None

    q = tree["quiet"]
    if type(q) is not dict or set(q) != {"since_s", "window_depth_change_m"}:
        _refuse("quiet record keys mismatch")
    if (q["since_s"] is not None and type(q["since_s"]) is not float) or type(q["window_depth_change_m"]) is not float:
        _refuse("quiet record types are not exact")
    for key in ("anchor_surface_m3", "anchor_soil_m3", "next_dt_cap_s"):
        if type(tree[key]) is not float or tree[key] < 0.0:
            _refuse(f"storm {key} must be an exact non-negative float")
    for key in ("generation", "start_generation", "accepted_steps", "rejected_attempts", "forcing_index"):
        if type(tree[key]) is not int:
            _refuse(f"storm {key} must be an exact integer")
    if type(tree["completed"]) is not bool:
        _refuse("completed must be a bool")
    if (tree["event_sha256"] != event.sha256 or tree["event_origin_s"] != float(event.event_origin_s)
            or tree["rainfall_end_s"] != float(event.rainfall_end_s) or tree["deadline_s"] != float(event.deadline_s)):
        _refuse("the re-prepared storm event differs from the saved identity/origin/rainfall end/deadline; refused")
    lc._int(tree["generation"], "generation")
    lc._int(tree["start_generation"], "start_generation")
    lc._int(tree["accepted_steps"], "accepted_steps")
    lc._int(tree["rejected_attempts"], "rejected_attempts")
    lc._int(tree["forcing_index"], "forcing_index", lo=-1)
    led = ledger(tree["ledger"], "ledger")
    if tree["generation"] != water.generation or led != water.ledger:
        _refuse("storm record generation/ledger differ from the restored water state")
    return ev.LayeredEventState(
        water, event.sha256, tree["anchor_surface_m3"], tree["anchor_soil_m3"], ledger(tree["anchor_ledger"], "anchor_ledger"),
        tree["start_generation"], tree["next_dt_cap_s"], tree["forcing_index"], tree["accepted_steps"],
        tree["rejected_attempts"], ev.QuietProgress(q["since_s"], q["window_depth_change_m"]), tree["completed"])


def _v2_gen0_inventory_check(state: Any) -> dict:
    """Separate SURFACE and SOIL closure of a genuine BOUNDARY_ORIGIN window at hydro generation 0 (the IW source core already requires the
    zero hydrology ledger and clock == origin, so no hydrology ever ran; only requested T can have acted, and T changes soil only). The owner's
    combined anchored residual lets an internal surface<->soil transfer cancel; here each inventory must close on its own:

        surface:  |H_now - anchor_surface| <= B_H            (T cannot change H: no T / lifetime term)
        soil:     |(W_now - anchor_soil) + (T.actual - Tanchor.actual)| <= B_W

    H_now / W_now come from the owner's own audited `_fresh_inventories` (every-cell inventory validation, router state, the same masked
    reductions); anchors are never rebased and a changed T counter is not a hydro step. Bounds use only MAPLE `volume_roundoff_bound_m3` with counts
    read from the code: the masked reductions sum N (surface) / L*N (soil) terms then scale by the cell area (n+1 terms), and two independent
    evaluations (entry anchor, now) are compared (x2); algebra operations as the owner counts them. The soil bound adds the native per-transaction
    soil bounds actually accumulated and the rounding of the n_t new additions into the large cumulative actual / soil_bound. Arrays/ledgers are only
    read; no tolerance, floor, cap, ledger adjustment or clipping. Returns the residuals and bounds (for receipts/tests)."""
    p, gop = state.payload, state.binding.gop
    base = gop.base
    N, L = int(base.n_cells), int(base.n_layers)
    try:
        h, w = iw._fresh_inventories(base, p.t_s, p.surface_m, p.qx_m2_s, p.qy_m2_s, p.soil, p.history)
    except iw.InterEventWaterError as exc:
        raise _convert(exc) from None
    aH, aW = float(p.anchor_surface_m3), float(p.anchor_soil_m3)
    t, ta = p.ledger.transpiration, p.t_anchor
    tr = float(t.actual_m3) - float(ta.actual_m3)
    n_t = int(t.n_requests) - int(ta.n_requests)
    vrb = ie_volume_roundoff
    sH = max(h, aH)
    res_h = h - aH
    bound_h = 2.0 * vrb(N + 1, sH) + vrb(2, sH)
    res_w = (w - aW) + tr
    bound_w = 2.0 * vrb(L * N + 1, max(w, aW)) + (float(t.soil_bound_m3) - float(ta.soil_bound_m3))
    for name in ("actual_m3", "soil_bound_m3"):
        bound_w += vrb(n_t + 1, max(abs(float(getattr(t, name))), abs(float(getattr(ta, name)))))
    bound_w += vrb(3, max(w, aW, abs(tr)))
    out = dict(surface_m3=h, soil_m3=w, surface_residual_m3=res_h, surface_bound_m3=bound_h, soil_residual_m3=res_w,
               soil_bound_m3=bound_w)
    if not all(math.isfinite(v) for v in out.values()) or bound_h < 0.0 or bound_w < 0.0:
        _refuse("non-finite generation-0 inventory closure")
    if abs(res_h) > bound_h:
        _refuse(f"the boundary-origin generation-0 surface inventory differs from its entry anchor by {res_h} m3 (bound {bound_h} m3): no "
                "hydrology has run and requested T cannot change the surface; refused")
    if abs(res_w) > bound_w:
        _refuse(f"the boundary-origin generation-0 soil inventory does not close with the accepted T alone: residual {res_w} m3 (bound "
                f"{bound_w} m3); refused")
    return out


def _v2_cross_checks(state: Any, role: str, kind: str | None) -> None:
    """v2-only closed phase/source consistency and the immediate-entry evidence recheck, on the ACTUAL restored owner objects. The last
    phase receipt vocabulary per current role is closed; for an interevent the receipt must describe exactly the current source. The
    boundary-origin entry receipt stays at the entry instant while the window evolves; after a close the phase is `interevent` and the
    current kind is None. Nothing is folded, reset or replayed; a hash match is consistency, never proof the past ran."""
    phase = state.last_phase_receipt
    if role != "interevent":
        if phase is not None and phase.kind not in ("storm", "interevent"):
            _refuse(f"a {role!r} state cannot carry a {phase.kind!r} last-phase receipt")
        return
    p = state.payload
    if iw.source_kind(p.source) != kind:
        _refuse("the restored interevent source kind differs from the manifest")
    if phase is None or phase.source_digest != p.source.digest or phase.episode != state.episode:
        _refuse("the interevent's last-phase receipt does not describe its current source/episode")
    if kind == iw.STORM_END:
        if (phase.kind != "storm" or phase.t_start_s != p.origin_s or phase.t_end_s != p.source.end_t_s
                or phase.hydro_steps != p.source.accepted_steps):
            _refuse("the storm-origin interevent's phase receipt does not match its StormEndReceipt")
        if phase.as_dict()["local_ledger"] != dataclasses.asdict(p.source.end_ledger):  # the SAME completed storm's end ledger, exactly
            _refuse("the storm-origin interevent's phase local ledger differs from its StormEndReceipt end ledger")
        return
    zero = it.GeneralWaterLedger(iw.WaterWindowLedger(), it.TranspirationLedger())
    if (phase.kind != "boundary_origin" or phase.hydro_steps != 0 or phase.t_start_s != p.origin_s or phase.t_end_s != p.origin_s
            or phase.as_dict()["local_ledger"] != zero.as_dict()):
        _refuse("the boundary-origin interevent's phase receipt is not its zero-duration entry receipt")
    if phase.parent_digest != p.source.parent_token:  # the entry the receipt describes is the one the genuine source names
        _refuse("the boundary-origin entry receipt's parent digest differs from its source's parent token")
    if p.generation == 0:
        _v2_gen0_inventory_check(state)
    untouched = (p.generation == 0 and p.ledger == zero and p.last_request_identity is None
                 and p.next_request_sequence == p.initial_sequence and p.t_s == p.origin_s)
    if untouched:  # no physical step and no accepted (even no-op) request: the entry instant is still current
        if state.mark != wl.Watermark():
            _refuse("an untouched boundary-origin entry must carry the zero local watermark")
        try:
            now = ie.quiet_evidence(state.binding.op, p.surface_m, p.qx_m2_s, p.qy_m2_s)
        except ls.LayeredStormError as exc:
            raise _convert(exc) from None
        if dict(now) != dict(p.source.evidence):
            _refuse("the untouched boundary-origin entry's current hydraulic evidence differs from the saved entry evidence")


def _restore(lop: Any, ref_bed: Any, storm: Any, data: dict, scalars: dict, arrays: dict, snapshot: Any, ledger: Any,
             committed: Any, budget: LifecycleBudget, t_s: float, pol: "_Policy | None" = None) -> LifecycleRestoration:
    pol = _policy_of(pol)
    role = data["role"]
    kind = data["source_kind"] if pol.v2 else None
    dense = lop.context is not None
    new_bed = None
    if dense:
        new_bed = BedState(voxel_column=snapshot["voxel_column"], active_layer=snapshot["active_layer"], water=snapshot["water"],
                           ledger=ledger, committed_topography=committed, sediment_availability=snapshot["sediment_availability"])
        try:
            lc._native_checks(new_bed, lop.context, t_s, type(ref_bed.ledger))
        except lc.LayeredCheckpointError as exc:
            raise LifecycleCheckpointError(str(exc)) from None
        if getattr(ledger, "present_processes", None) != getattr(ref_bed.ledger, "present_processes", None):
            _refuse("native ledger present_processes differ from the reference (process selection is restart-relevant)")
        if _material_sha(new_bed, lop.context) != data["binding"]["native_material_sha256"]:
            _refuse("the loaded native material content differs from the expected material (portable content SHA)")
    # genuine rebinding on the LOADED bed with the same fixed components; the expected operator is never edited
    try:
        new_lop = wl.prepare_lifecycle(
            lop.column, lop.preferential, lop.matrix, lop.router, lop.graph, lop.topology, lop.evaporation, lop.cover,
            bed=new_bed, context=lop.context, native_reserve_bytes=lop.native_reserve_bytes if dense else None,
            extra_reserve_bytes=lop.extra_reserve_bytes, t_op=lop.t_op)
    except (wl.LifecycleError, ss.SubsurfaceError) as exc:
        raise _convert(exc) from None
    if new_lop.sha256 != lop.sha256:
        _refuse("the re-prepared lifecycle operator identity differs from the expected one")
    try:
        soil = ss.column_state(lop.column, arrays["soil_water_m"], t_s)
        history = mi.wetting_history(lop.matrix, wetting_m=arrays["wetting_m"], cumulative_intake_m=arrays["cumulative_intake_m"])
    except (ss.SubsurfaceError, mi.MatrixError) as exc:
        raise _convert(exc) from None
    surface = new_bed.water.depth_m if dense else np.ascontiguousarray(arrays["surface_m"])
    qx, qy = np.ascontiguousarray(arrays["qx_m2_s"]), np.ascontiguousarray(arrays["qy_m2_s"])
    base = (surface, qx, qy, soil, history, new_bed)
    life = cc.decode_tree(scalars["lifecycle"])
    sop = life.get("storm_operator_sha256") if isinstance(life, dict) else None
    records: dict = {"lifecycle": scalars["lifecycle"]}
    objects: dict = {}
    extra: dict = {}
    new_op = None
    try:
        if role in ("storm", "interevent"):
            kw = dict(bed=new_bed, context=lop.context, native_reserve_bytes=lop.native_reserve_bytes) if dense else {}
            new_op = ls.prepare_layered_storm(lop.column, lop.preferential, lop.matrix, lop.router, lop.graph, lop.topology,
                                              extra_reserve_bytes=lop.extra_reserve_bytes, **kw)
            if new_op.sha256 != sop:
                _refuse("the re-prepared storm operator identity differs from the saved one (native material or bindings differ)")
            extra["storm_op"] = new_op
        if role == "storm":
            event = ev.prepare_layered_event(new_op, storm.schedule, storm.field, event_origin_s=storm.event_origin_s,
                                             control=storm.control, policy=storm.policy)
            tree = cc.decode_tree(scalars["storm"])
            gen = tree["generation"] if isinstance(tree, dict) and type(tree.get("generation")) is int else -1
            led_tree = tree["ledger"] if isinstance(tree, dict) else None
            ledger_state = it._typed_ledger(ls.LayeredWaterLedger, led_tree, "ledger")
            water = ls.LayeredWaterState(t_s, surface, qx, qy, soil, history, new_bed, ledger_state, gen, new_op.sha256)
            estate = _storm_state_from(scalars["storm"], event, water)
            ev.audit_event(event, estate)
            extra.update(event=event, estate=estate)
            records["storm"] = scalars["storm"]
        elif role == "interevent":
            prep = iw.prepare_interevent_water(new_op, lop.evaporation, lop.cover)
            gop = it.prepare_general_interevent(prep, lop.t_op)
            extra.update(prep=prep, gop=gop)
            records["t_state"] = scalars["t_state"]
            objects["t_state"] = base
            if pol.v2:
                env = scalars["source"]  # RAW encoded envelope (digest-bound as saved; decoded exactly once by its owner)
                if type(env) is not dict or set(env) != {"source_kind", "body"} or env["source_kind"] != kind:
                    _refuse("the saved source envelope is malformed or names a different kind than the manifest")
                records["source_envelope"] = env
                objects["source_envelope"] = ()
                if kind == iw.STORM_END:
                    records["receipt"] = env["body"]  # the unchanged v1 restorer's capability part (raw, encoded body)
                    objects["receipt"] = ()
            else:
                records["receipt"] = scalars["receipt"]
                objects["receipt"] = ()
        caps: list = []

        def mint(recs: dict, objs: dict) -> Any:
            cap = cc._issue(role=role, lop_sha256=new_lop.sha256, manifest_sha256=data["manifest_sha256"], records=recs, objects=objs)
            caps.append(cap)
            return cap

        try:
            tstate = None
            if role == "interevent":
                # the T owner restores the general state first (capability bound to ITS record and the exact decoded arrays),
                # then the lifecycle owner restores the outer account around that genuine state with a second capability
                if pol.v2:
                    t_parts = {k: records[k] for k in ("t_state", "source_envelope", "receipt") if k in records}
                    cap_t = mint(t_parts, {k: objects[k] for k in t_parts})
                    tstate = it._restore_checkpoint_general_state(
                        extra["gop"], cap_t, records["t_state"], records["source_envelope"], arrays=base,
                        _source_restorer=iw._restore_checkpoint_source_v2)
                else:
                    cap_t = mint({"t_state": records["t_state"], "receipt": records["receipt"]}, objects)
                    tstate = it._restore_checkpoint_general_state(extra["gop"], cap_t, records["t_state"], records["receipt"],
                                                                  arrays=base)
            full = (*base, extra.get("storm_op"), extra.get("event"), extra.get("estate"), extra.get("prep"), extra.get("gop"),
                    tstate)
            cap = mint({"lifecycle": records["lifecycle"]}, {"lifecycle": full})
            state = wl._restore_checkpoint_lifecycle(
                new_lop, cap, records["lifecycle"], role=role, arrays=base, storm_op=extra.get("storm_op"),
                event=extra.get("event"), estate=extra.get("estate"), prep=extra.get("prep"), gop=extra.get("gop"),
                tstate=tstate, _phase_kinds=wl._V2_PHASE_KINDS if pol.v2 else None)
            if pol.v2:
                _v2_cross_checks(state, role, kind)
        finally:
            for c in caps:
                cc.close(c)
    except (wl.LifecycleError, ls.LayeredStormError, ev.LayeredEventError, iw.InterEventWaterError, cc.CheckpointCapabilityError,
            ss.SubsurfaceError) as exc:
        raise _convert(exc) from None
    receipt = {"profile": data["profile"], "role": role, "t_s": t_s, "revision": int(state.revision),
               "manifest_sha256": data["manifest_sha256"], "files": dict(data["files"]), "source": data["source"],
               "state_digest": state.digest, "physical_digest": physical_digest(new_lop, state), "budget": budget.as_dict(),
               "lop_sha256": new_lop.sha256, "adopted": False}
    return LifecycleRestoration(new_lop, state, receipt)
