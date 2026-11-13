"""Narrow, genuine DENSE-CUDA source association (candidate A1; nothing existing is edited or routed through this module).

It answers ONE question for a later CUDA owner: is the imported `maple` package exactly one of two named, immutable native
packages (the merged MAPLE39 package or the native9f package), and which genuine module/class objects does this process bind to
that package? The answer is a sealed, factory-minted association record. It is EVIDENCE OF ASSOCIATION, not CPU/GPU physics
acceptance and not a device-capability claim; it does not touch a device, an array or CuPy.

* Registry: two module-private literal package digests under two distinct profile names. There is no caller-supplied digest, no
  generic whitelist, no extension API and no fallback; an unknown profile or a digest of the other family is a typed refusal.
* Cold verification (`verify_dense_cuda_source`) runs only after its streamed-hash stage was ADMITTED against the caller's
  `available_bytes_fn` + reserve; class/binding checks that need no hash precede the hash. The hash is the process-cached
  `sediment_bed.bound_maple_package_digest()` (the first call IS the fresh streamed hash).
* Hot guard (`check_dense_cuda_source_hot`) compares recorded object identities only: no hash, no read, no device query.
* Pure planning (`plan_source_verification`) does no I/O and never warms the cache. Every dependency it needs is imported when this
  module loads (normal import/setup cost, excluded from plans); the first plan call performs no open, hash, import or query.
* Canonical bindings: at module load the genuine `maple`/voxel modules, the native dense class, the owner's own binding of it and SYRUP's
  `sediment_bed` module/classes/function are frozen; cold verification refuses any process binding that differs (before the hash).
  A rebinding made BEFORE this module loads is outside the trusted-load premise.

Admission scope: a reading of available host bytes is not a reservation, and no whole-process memory bound is claimed (interpreter,
allocator, import and file-system costs are excluded). Trusted in-process tampering with this module's private registry is outside
the contract; ordinary class/module rebinding is detected by the hot guard.
"""
from __future__ import annotations

import sys
import types
import weakref
from dataclasses import dataclass
from typing import Any

import maple as _maple  # canonical load-time references (normal import/setup cost, excluded from every plan)
import numpy as np
from maple.core.types import voxel as _voxel

from maple_syrup import interevent_transpiration as _it  # loaded here so planning/admission never import (A2 keeps its edge lazy)
from maple_syrup import sediment_bed as sb
from maple_syrup import subsurface as _ss

__all__ = [
    "DenseCudaSource",
    "DenseCudaSourceError",
    "DenseCudaSourceMemoryError",
    "DenseCudaSourceMismatchError",
    "DenseCudaSourcePlan",
    "DenseCudaSourceProfileError",
    "PROFILE_NATIVE39",
    "PROFILE_NATIVE9F",
    "MAX_RECORD_PATH_CHARS",
    "RECORD_RETAINED_BYTES",
    "check_dense_cuda_source_hot",
    "plan_source_verification",
    "registered_profiles",
    "source_cold",
    "source_stage_bytes",
    "verify_dense_cuda_source",
]

PROFILE_NATIVE39 = "native39_dense_cuda"
PROFILE_NATIVE9F = "native9f_dense_cuda"
#: module-private, immutable; deliberately not derived from or added to any other source list
_REGISTRY = types.MappingProxyType({
    PROFILE_NATIVE39: types.MappingProxyType(
        {"family": "native39", "package_sha256": "af54626ab432670539ed71480fd884f13581dfdfcfec63d7c85053ca215adfec"}),
    PROFILE_NATIVE9F: types.MappingProxyType(
        {"family": "native9f", "package_sha256": "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"}),
})
#: Retained-record charge. The record BORROWS (does not copy) the module/class/function objects and the registry literal strings. Owned
#: allocations: the object, the read-only payload mapping (~16 entries) and a weakref slot (<= 2048 B declared). The two file strings are
#: kept as references to the modules' own attributes and are bounded by refusal (<= MAX_RECORD_PATH_CHARS chars): 2 * (1024 + 64) B.
#: 2048 + 2 * 1088 = 4224 <= RECORD_RETAINED_BYTES (6144 declared, margin included). Independent of N, K and L.
MAX_RECORD_PATH_CHARS = 1024
RECORD_RETAINED_BYTES = 6144
_RECORD_FIELDS = ("profile", "family", "package_sha256", "maple_module", "maple_module_file", "voxel_module", "voxel_module_name",
                  "voxel_module_file", "dense_class", "bed_module", "bed_state_class", "bed_context_class", "bed_from_case",
                  "cache_was_cold", "stage_bytes", "retained_bytes")
_VOXEL_MODULE = sb.MAPLE_VOXEL_MODULE
_BED_MODULE = "maple_syrup.sediment_bed"


class DenseCudaSourceError(RuntimeError):
    """Base of every typed refusal of this module."""


class DenseCudaSourceProfileError(DenseCudaSourceError):
    """Unknown/non-string profile name or invalid argument type."""


class DenseCudaSourceMemoryError(DenseCudaSourceError):
    """The cold verification stage was not admitted; nothing was read or hashed."""


class DenseCudaSourceMismatchError(DenseCudaSourceError):
    """The imported package/classes are not the named profile's source (or an association changed)."""


def registered_profiles() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def _profile_entry(profile: Any) -> types.MappingProxyType:
    if type(profile) is not str or profile not in _REGISTRY:
        raise DenseCudaSourceProfileError(f"profile must be exactly one of {tuple(_REGISTRY)}, got {profile!r}")
    return _REGISTRY[profile]


def _checked_nonneg(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise DenseCudaSourceProfileError(f"{name} must be a non-Boolean integer >= 0, got {value!r}")
    return int(value)


# --- pure planning (no I/O, no hash, no arrays, no device, no cache warming) ----------------------------------------------------
def source_stage_bytes() -> int:
    """The existing declared peak of the streamed source-tree hash (`interevent_transpiration._source_stage_bytes`), a pure
    arithmetic call. That module is imported when THIS module loads (normal setup cost), so no import or I/O happens here; a later
    interevent_transpiration -> this module edge must itself be lazy to avoid an import cycle."""
    return int(_it._source_stage_bytes())


def source_cold() -> bool:
    """True while this process has not yet computed the bound package digest (a membership test only)."""
    return "digest" not in sb._MAPLE_DIGEST_CACHE


@dataclass(frozen=True)
class DenseCudaSourcePlan:
    """Whole-call host figure of one cold verification. `stage_bytes` is 0 for a warm process cache; stage and record are sequential
    (the hash stage is freed before the record exists), so `required_bytes = max(stage, retained)`, not their sum."""

    profile: str
    cold: bool
    stage_bytes: int
    retained_bytes: int
    required_bytes: int


def plan_source_verification(profile: str) -> DenseCudaSourcePlan:
    _profile_entry(profile)
    cold = source_cold()
    stage = source_stage_bytes() if cold else 0
    retained = RECORD_RETAINED_BYTES
    return DenseCudaSourcePlan(profile, cold, int(stage), retained, int(max(stage, retained)))


# --- sealed record ----------------------------------------------------------------------------------------------------------------
_SEALED: "weakref.WeakKeyDictionary[Any, types.MappingProxyType]" = weakref.WeakKeyDictionary()


class DenseCudaSource:
    """Sealed association record minted only by `verify_dense_cuda_source` (payload in a private read-only registry)."""

    __slots__ = ("__weakref__",)

    def __init__(self, *a, **k):
        raise DenseCudaSourceError("DenseCudaSource is created only by verify_dense_cuda_source")

    def _p(self):
        p = _SEALED.get(self) if type(self) is DenseCudaSource else None
        if p is None:
            raise DenseCudaSourceError("unauthenticated DenseCudaSource (not factory-produced)")
        return p

    def __setattr__(self, name, value):
        raise DenseCudaSourceError("DenseCudaSource is immutable")

    def __delattr__(self, name):
        raise DenseCudaSourceError("DenseCudaSource is immutable")

    def __copy__(self):
        raise DenseCudaSourceError("DenseCudaSource cannot be copied; re-run verify_dense_cuda_source")

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __reduce_ex__(self, protocol):
        return self.__copy__()

    def __repr__(self):
        return f"<DenseCudaSource {'sealed' if _SEALED.get(self) is not None else 'UNAUTHENTICATED'}>"

    def __dir__(self):
        return list(_RECORD_FIELDS)

    def __getattr__(self, name):  # only reached for names that are not slots
        if name in _RECORD_FIELDS:
            return self._p()[name]
        raise AttributeError(name)


# --- canonical load-time bindings and genuine association gathering (no hash, no import) ----------------------------------------
def _freeze_canonical() -> tuple[dict[str, Any], str | None]:
    """The genuine objects bound when this module loads. A structurally invalid canonical set is remembered as an error string that
    every cold verification re-raises (typed), never an import failure."""
    try:
        dense = sb._maple_voxel_class(_voxel, "VoxelColumnState", required=True)
    except (AttributeError, sb.BedIntegrationError) as exc:
        return {}, f"{_VOXEL_MODULE}.VoxelColumnState is not a genuine class at load: {exc}"
    canon = dict(maple_module=_maple, maple_module_file=getattr(_maple, "__file__", None), voxel_module=_voxel,
                 voxel_module_file=getattr(_voxel, "__file__", None), dense_class=dense, owner_dense_class=getattr(_it, "VoxelColumnState", None),
                 bed_module=sb, bed_state_class=sb.BedState, bed_context_class=sb.BedContext, bed_from_case=sb.bed_from_case)
    err = None
    if type(_maple) is not types.ModuleType or type(_voxel) is not types.ModuleType:
        err = "the canonical maple/voxel modules are not plain modules"
    elif canon["owner_dense_class"] is not dense:
        err = "the interevent owner's VoxelColumnState is not the voxel module's class at load"
    else:
        for k in ("maple_module_file", "voxel_module_file"):
            if type(canon[k]) is not str or not 0 < len(canon[k]) <= MAX_RECORD_PATH_CHARS:
                err = f"{k} is not an exact str of 1..{MAX_RECORD_PATH_CHARS} chars"
    return canon, err


_CANON, _CANON_ERROR = _freeze_canonical()


def _gather_association() -> dict[str, Any]:
    """Compare the CURRENT process bindings with the canonical load-time references by identity; any difference is a typed refusal
    raised BEFORE any hash. No import happens here."""
    if _CANON_ERROR is not None:
        raise DenseCudaSourceMismatchError(f"the canonical load-time bindings are invalid: {_CANON_ERROR}")
    c = _CANON
    if sys.modules.get("maple") is not c["maple_module"] or getattr(c["maple_module"], "__file__", None) != c["maple_module_file"]:
        raise DenseCudaSourceMismatchError("the imported `maple` module/file is not the canonical load-time one")
    voxel = sys.modules.get(_VOXEL_MODULE)
    if (voxel is not c["voxel_module"] or type(voxel) is not types.ModuleType
            or getattr(voxel, "__file__", None) != c["voxel_module_file"]):
        raise DenseCudaSourceMismatchError(f"{_VOXEL_MODULE} is not the canonical load-time module object/file")
    try:
        dense = sb._maple_voxel_class(voxel, "VoxelColumnState", required=True)
    except (AttributeError, sb.BedIntegrationError) as exc:
        raise DenseCudaSourceMismatchError(f"{_VOXEL_MODULE}.VoxelColumnState is not a genuine class: {exc}") from None
    if dense is not c["dense_class"] or getattr(_it, "VoxelColumnState", None) is not c["dense_class"]:
        raise DenseCudaSourceMismatchError("VoxelColumnState is not the canonical native class object (a subclass or foreign class is refused)")
    if (sys.modules.get(_BED_MODULE) is not c["bed_module"] or sb.BedState is not c["bed_state_class"]
            or sb.BedContext is not c["bed_context_class"] or sb.bed_from_case is not c["bed_from_case"]):
        raise DenseCudaSourceMismatchError("the SYRUP sediment_bed module/classes/function are not the canonical load-time ones")
    return dict(maple_module=c["maple_module"], maple_module_file=c["maple_module_file"], voxel_module=voxel,
                voxel_module_name=_VOXEL_MODULE, voxel_module_file=c["voxel_module_file"], dense_class=dense,
                bed_module=sb, bed_state_class=c["bed_state_class"], bed_context_class=c["bed_context_class"],
                bed_from_case=c["bed_from_case"])


def _admit(required: int, reserve: int, available_bytes_fn: Any) -> None:
    available = available_bytes_fn()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise DenseCudaSourceProfileError("available_bytes_fn must return a non-Boolean integer >= 0")
    if required + reserve > int(available):
        raise DenseCudaSourceMemoryError(
            f"the cold source verification needs {required} B plus a {reserve} B caller reserve, but only {int(available)} B are "
            "reported (a reading, not a reservation); nothing was read or hashed")


def verify_dense_cuda_source(profile: str, *, available_bytes_fn: Any = None, reserve_bytes: int = 0) -> DenseCudaSource:
    """COLD qualification of the imported package against the NAMED profile. Order: profile/arguments, pure plan, ADMISSION
    (before any read or hash), genuine class/binding checks (before the hash), the process-cached streamed package digest, mint."""
    entry = _profile_entry(profile)
    if available_bytes_fn is not None and not callable(available_bytes_fn):
        raise DenseCudaSourceProfileError("available_bytes_fn must be callable or None")
    reserve = _checked_nonneg(reserve_bytes, "reserve_bytes")
    plan = plan_source_verification(profile)
    if available_bytes_fn is None:
        available_bytes_fn = _ss.default_host_available_bytes
    _admit(plan.required_bytes, reserve, available_bytes_fn)
    assoc = _gather_association()
    digest = sb.bound_maple_package_digest()
    if digest != entry["package_sha256"]:
        known = [n for n, e in _REGISTRY.items() if e["package_sha256"] == digest]
        raise DenseCudaSourceMismatchError(
            f"the imported MAPLE package {digest[:12]}... is not the source named by profile {profile!r} "
            f"({entry['package_sha256'][:12]}...)" + (f"; it is the source of profile {known[0]!r}" if known else "; it is no registered source"))
    obj = object.__new__(DenseCudaSource)
    _SEALED[obj] = types.MappingProxyType(dict(profile=profile, family=entry["family"], package_sha256=digest, **assoc,
                                               cache_was_cold=bool(plan.cold), stage_bytes=plan.stage_bytes,
                                               retained_bytes=plan.retained_bytes))
    return obj


def check_dense_cuda_source_hot(src: Any, profile: str | None = None) -> None:
    """Cheap continuing identity guard; NEVER hashes, reads, queries a device, builds an array or warms the cache. A changed
    association is a typed refusal; re-run the cold verification explicitly."""
    if type(src) is not DenseCudaSource:
        raise DenseCudaSourceMismatchError("the dense-CUDA source association is missing or not a DenseCudaSource")
    p = _SEALED.get(src)
    if p is None:
        raise DenseCudaSourceMismatchError("unauthenticated DenseCudaSource (not factory-produced)")
    if profile is not None and p["profile"] != profile:
        raise DenseCudaSourceMismatchError(f"the association is for profile {p['profile']!r}, not {profile!r}")
    voxel = sys.modules.get(_VOXEL_MODULE)
    maple = sys.modules.get("maple")
    if maple is not p["maple_module"] or str(getattr(maple, "__file__", "")) != p["maple_module_file"]:
        raise DenseCudaSourceMismatchError("the imported `maple` package is no longer the one verified; re-verify (no hash in a hot call)")
    if voxel is not p["voxel_module"] or str(getattr(voxel, "__file__", "")) != p["voxel_module_file"]:
        raise DenseCudaSourceMismatchError("the imported voxel module is no longer the one verified; re-verify")
    if vars(voxel).get("VoxelColumnState") is not p["dense_class"]:
        raise DenseCudaSourceMismatchError("VoxelColumnState is no longer the verified native class object; re-verify")
    if getattr(_it, "VoxelColumnState", None) is not p["dense_class"]:
        raise DenseCudaSourceMismatchError("the interevent owner's VoxelColumnState binding is no longer the verified native class; re-verify")
    if (sys.modules.get(_BED_MODULE) is not p["bed_module"] or sb.BedState is not p["bed_state_class"]
            or sb.BedContext is not p["bed_context_class"] or sb.bed_from_case is not p["bed_from_case"]):
        raise DenseCudaSourceMismatchError("the SYRUP sediment_bed bindings changed since verification; re-verify")
    cached = sb._MAPLE_DIGEST_CACHE.get("digest")
    if cached is not None and cached != p["package_sha256"]:
        raise DenseCudaSourceMismatchError("the process source digest no longer equals the verified package; re-verify")
