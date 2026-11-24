"""Opt-in exact CPU native MAPLE ORDINARY DENSE bed preparation.

SCOPE. Separate narrow cold setup profiles ``native_maple_9f_dense_cpu_v1`` and ``native_maple_1dca_dense_cpu_v1``: from a genuine `NativeCaseAdmission`
(`native_case_profile`, same process, same exact selected CPU bundle) derive the actual `(BedState, BedContext)` through the UNCHANGED
`sediment_bed.bed_from_case`, after refusing every unsupported source/case/storage/feature/backend request from cheap
metadata, and after the native PUBLIC full validators accept the result. It then seals scalar provenance with REFERENCES to the
genuine returned objects. It does NOT:

* copy or reimplement native pickup/deposit/export/commit arithmetic (callers keep using `sediment_bed.apply_bed_demand`,
  `commit_bed`, `with_water`, ... on the returned objects; each returns a new genuine `BedState` the caller owns),
* clone material or freeze a second authority (the sealed record references the INITIAL bed; a caller that evolves state
  should drop the preparation, otherwise the initial arrays stay alive -- see ``memory``),
* accept any codec/checkpoint/storage-admission (`storage_admission`, journal recognizers, layered/pond/Windowed codecs are OUT;
  nothing here is a native39 profile and no 9f digest is added to any whitelist),
* claim a universal fence: generic low-level APIs called OUTSIDE this helper keep their existing behaviour,
* provide Windowed, CUDA, wind, MASY, LEAF, availability builders, fixed base/regional slope or avalanching/creep (the 9f/1dca
  profiles; unchanged).

TERRAIN (separately named ``native_maple_terrain_20261123_dense_cpu_v1``, selected ONLY by a `native_case_profile.TERRAIN_PROFILE_NAME`
admission): the same cold preparation for the committed c4b9343 terrain case. It additionally carries the genuine native `FixedBaseState`
/ `DomainState` the case compiled with on the returned `BedContext` (BY REFERENCE on NumPy: no copy, charged as resident/aliased,
not as new allocation), admits creep/avalanche, bounded voxel growth and the regional context, and runs the native terrain validators
in the suite: structural base/footprint validation, MAPLE's own `check_committed_reference_consistent_with_fixed_base` (the committed
reference must be the world reference), the native initial-committed derivation WITH the base for the fresh-baseline comparison,
the native footprint inventory invariant (`check_domain_inventory`: every sediment owner exactly zero on NoData) and the SYRUP water
rule (all water on NoData exactly zero). It still refuses imported sediment, an airborne seed, non-square cells, an enabled native
water_coupling and a non-bare initial vegetation. No surface formula is copied: the native terrain APIs are the authority.

Memory (whole call, derived terms; see `required_bytes`). The compiled native case is ALREADY allocated by native initialization
(by reference; compile/load is inherited and NOT bounded here). BEFORE the mandatory cold `admission.recheck()` and before any array
work the helper computes, from metadata only (`os.stat`, shapes, nbytes), one required total
``max(source_audit, dense_validation) + fixed`` and compares ``required + caller_reserve`` with the reported available bytes. The two
stages are sequential (the audit's allocations are released before bed conversion; only the profile record survives), hence ``max``.

* source_audit terms (named in `source_audit_terms`): manifest text lifetimes, parsed-manifest object graph, largest-file lifetimes,
  1 MiB read buffers, per-file path/entry tables, authored-file reads. The audit is NEVER cached, skipped or digest-only.
* dense_validation terms (named in `dense_terms`): the new available-fraction array, the native voxel/partition/ledger/availability/
  committed validators' enumerated temporaries and the native initial-committed derivation used for the baseline comparison.

CPython/metadata exclusions: first-use import/byte-compilation (SYRUP modules are imported at this module's import time; the native
modules were imported by the genuine admission), allocator pools/fragmentation, interpreter/import state, the OS page cache, the native compile/load
footprint, the retained initial bed itself, and any growth of the authored/source trees after the preflight (inputs are treated as
immutable during the call; this is not adversarial-race proof). A reading is not a reservation. Terms are an auditable allocation-count
MODEL for the closed real bundle/schema (not a hostile-input bound), checked against measured whole-helper tracemalloc peaks (no stubs) on several N/V/K in the tests; they are not a proof.
"""
from __future__ import annotations

import dataclasses
import types
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from maple_syrup import native_case_profile as ncp
# Imported at MODULE level (import time, not call time) so the preparation call itself never byte-compiles/imports SYRUP code; the
# native maple modules used below were already imported by the genuine admission that this call requires.
from maple_syrup import conservation as _conservation  # noqa: F401
from maple_syrup import sediment_bed as _sediment_bed  # noqa: F401
from maple_syrup import subsurface as _subsurface  # noqa: F401

__all__ = ["PROFILE_NAME", "CURRENT_PROFILE_NAME", "TERRAIN_PROFILE_NAME", "DenseLimits", "NativeDensePreparation", "NativeDenseProfileError",
           "dense_terms", "required_bytes", "source_audit_terms", "plan_preparation", "prepare_native_dense_case"]

PROFILE_NAME = "native_maple_9f_dense_cpu_v1"
CURRENT_PROFILE_NAME = "native_maple_1dca_dense_cpu_v1"
#: Separately named TERRAIN Dense CPU profile (committed c4b9343; see the module notes). Never a default.
TERRAIN_PROFILE_NAME = "native_maple_terrain_20261123_dense_cpu_v1"
_CPU_ADMISSIONS = (ncp.PROFILE_NAME, ncp.CURRENT_PROFILE_NAME, ncp.TERRAIN_PROFILE_NAME)
_DENSE_PROFILE_BY_ADMISSION = {ncp.PROFILE_NAME: PROFILE_NAME, ncp.CURRENT_PROFILE_NAME: CURRENT_PROFILE_NAME,
                               ncp.TERRAIN_PROFILE_NAME: TERRAIN_PROFILE_NAME}
#: Allocation-count constants of the MODEL (each justified where it is used, in `dense_terms` / `source_audit_terms`).
_READ_CHUNK = 1 << 20  # dependency_bundle._CHUNK and provenance._READ_CHUNK_BYTES (os.read buffer, both 1 MiB)
_JSON_OBJECT_BYTES_PER_TEXT_BYTE = 16  # source-specific MODEL coefficient for the CLOSED real manifests (see source_audit_terms); not a universal JSON bound
_PER_FILE_TABLE_BYTES = 1024  # path str + tuple + dict/set slots per file in the digest/manifest/overlay tables
_FIXED_AUDIT_BYTES = 1 << 18  # exception/Path objects, dependency record, scandir state (explicitly NOT scaled)
_FIXED_DENSE_BYTES = 1 << 16  # dataclass shells and validator bookkeeping (explicitly NOT scaled)
_MAX_PREFLIGHT_ENTRIES = 20_000  # same ceiling as dependency_bundle.Limits.max_files; a larger tree is refused before any hash
_MAX_PREFLIGHT_MANIFEST_BYTES = 16 << 20  # dependency_bundle.Limits.max_manifest_bytes


def _r(value, limit: int = 60) -> str:
    """Bounded diagnostic: type name plus a truncated repr (never an unbounded repr of a caller object)."""
    try:
        text = repr(value)
    except Exception:  # a hostile __repr__ must not escape the typed refusal
        text = "<unrepresentable>"
    return f"{type(value).__name__}:{text[:limit]}" + ("..." if len(text) > limit else "")


class NativeDenseProfileError(RuntimeError):
    """Unsupported source/case/storage/feature/backend request, failed validation or insufficient memory (refused)."""


@dataclass(frozen=True)
class DenseLimits:
    """Hard scope bounds applied to metadata before any scan or allocation (not a whole-model memory bound)."""

    max_cells: int = 1 << 20
    max_voxel_elements: int = 1 << 26
    max_classes: int = 32
    # Terrain profile only (default LAST): the explicit cap on the resident terrain owners (fixed-base raster + the footprint's
    # six tables) a preparation may carry/place; older profiles carry none and never read it.
    max_terrain_bytes: int = 1 << 28

    def __post_init__(self):
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if type(v) is not int or v <= 0:
                raise NativeDenseProfileError(f"{f.name} must be a positive int, got {_r(v)}")


# --- sealed record ----------------------------------------------------------------------------------------
_SEALED: "weakref.WeakKeyDictionary[Any, types.MappingProxyType]" = weakref.WeakKeyDictionary()
_FIELDS = ("profile_name", "bundle_id", "package_digest", "syrup_package_digest", "case_identity_sha256", "latitude_deg",
           "route", "shape", "n_voxels", "n_classes", "class_ids", "ledger_kind", "backend", "memory", "caller_reserve_bytes",
           "bed", "context", "admission", "terrain")


class NativeDensePreparation:
    """Sealed result of `prepare_native_dense_case` (factory-only; payload in a private read-only registry)."""

    __slots__ = ("__weakref__",)

    def __init__(self, *a, **k):
        raise NativeDenseProfileError("NativeDensePreparation is created only by prepare_native_dense_case")

    def _p(self):
        p = _SEALED.get(self) if type(self) is NativeDensePreparation else None
        if p is None:
            raise NativeDenseProfileError("unauthenticated NativeDensePreparation (not factory-produced)")
        return p

    def __setattr__(self, name, value):
        raise NativeDenseProfileError("NativeDensePreparation is immutable")

    def __delattr__(self, name):
        raise NativeDenseProfileError("NativeDensePreparation is immutable")

    def __copy__(self):
        raise NativeDenseProfileError("NativeDensePreparation cannot be copied; re-run the factory")

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __reduce_ex__(self, protocol):
        return self.__copy__()

    def __repr__(self):
        return f"<NativeDensePreparation {'sealed' if _SEALED.get(self) is not None else 'UNAUTHENTICATED'}>"

    def __dir__(self):
        return list(_FIELDS)

    def latitude_binding(self) -> dict[str, Any]:
        return self._p()["admission"].latitude_binding()

    def recheck(self, *, caller_reserve_bytes: int, available_bytes_fn: Callable[[], int] | None = None) -> None:
        """Fresh cold recheck of the CURRENT initial preparation: the same whole-call memory admission (before any work), the fresh
        source/runtime/authored-file audit of the admission, Dense class, bed-leaf identity with the compiled case, and the SAME full
        native validation suite as preparation (voxel/partition, ledger, availability, water, committed topography and its
        initial-baseline reconciliation). It validates the initial preparation, not a later evolved bed (callers own those)."""
        p = self._p()
        compiled, meta = _scope(p["admission"], "numpy", None, caller_reserve_bytes, available_bytes_fn)
        _recheck_admission(p["admission"])
        _require_dense(p["bed"].voxel_column)
        _check_identity(p["bed"], compiled, p["context"])
        _validate_bed(p["bed"], p["context"], compiled, meta)


def _getter(name):
    return property(lambda self: self._p()[name])


for _n in _FIELDS:
    setattr(NativeDensePreparation, _n, _getter(_n))
del _n


# --- scope checks -----------------------------------------------------------------------------------------
def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _types():
    import importlib

    voxel = importlib.import_module("maple.core.types.voxel")
    ns = types.SimpleNamespace(
        dense=voxel.VoxelColumnState, windowed=vars(voxel).get("WindowedVoxelColumnState"),
        active=importlib.import_module("maple.core.types.active_layer").ActiveLayerState,
        water=importlib.import_module("maple.core.types.water").WaterState,
        availability=importlib.import_module("maple.core.types.sediment_availability").SedimentAvailabilityState,
        committed=importlib.import_module("maple.core.types.topographic_commit").CommittedTopographyState,
        ledger=importlib.import_module("maple.core.types.sediment_ledger"),
        compiled=importlib.import_module("maple.case_tools.compilers.case_compiler").CompiledCase)
    return ns


def _terrain_types():
    """Native terrain classes/validators, imported ONLY for the terrain profile (older pinned sources lack the domain module)."""
    import importlib

    try:
        fb = importlib.import_module("maple.core.types.fixed_base")
        dom = importlib.import_module("maple.core.types.domain")
        height = importlib.import_module("maple.surface.topographic_commit.height")
        footprint = importlib.import_module("maple.case_tools.validators.domain_footprint")
    except ImportError as exc:
        raise NativeDenseProfileError(f"the imported maple source lacks the terrain API: {exc}") from exc
    return types.SimpleNamespace(fixed_base=fb.FixedBaseState, validate_fixed_base=fb.validate_fixed_base_state,
                                 domain=dom.DomainState, validate_domain=dom.validate_domain_state,
                                 check_world_reference=height.check_committed_reference_consistent_with_fixed_base,
                                 check_domain_inventory=footprint.check_domain_inventory)


def _require_dense(column) -> None:
    t = _types()
    import importlib

    if "WindowedVoxelColumnState" in vars(importlib.import_module("maple.core.types.voxel")) and not isinstance(t.windowed, type):
        raise NativeDenseProfileError("maple.core.types.voxel.WindowedVoxelColumnState is present but not a class (malformed binding)")
    if t.windowed is not None and type(column) is t.windowed:
        raise NativeDenseProfileError("a Windowed voxel column is refused: this capability is ordinary Dense only")
    if type(column) is not t.dense:
        raise NativeDenseProfileError(f"voxel column must be exactly the native VoxelColumnState, got {type(column).__module__}."
                                      f"{type(column).__qualname__}")


def _leaf_bytes(obj) -> int:
    total = 0
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, np.ndarray):
            total += int(v.nbytes)
    return total


def _walk_metadata(root: Path) -> tuple[int, int]:
    """(regular-file count, largest regular-file bytes) of a tree by `os.scandir` + lstat only (streaming; nothing is read or hashed).
    Symlinks/special files are not followed or counted here (the fresh audit refuses them); over `_MAX_PREFLIGHT_ENTRIES` refuses."""
    import os

    count = largest = 0
    stack = [str(root)]
    while stack:
        with os.scandir(stack.pop()) as it:
            for e in it:
                if e.is_symlink():
                    continue
                if e.is_dir(follow_symlinks=False):
                    if e.name != "__pycache__":
                        stack.append(e.path)
                elif e.is_file(follow_symlinks=False):
                    count += 1
                    largest = max(largest, e.stat(follow_symlinks=False).st_size)
                    if count > _MAX_PREFLIGHT_ENTRIES:
                        raise NativeDenseProfileError(f"{root} holds more than {_MAX_PREFLIGHT_ENTRIES} files; refused before any hash")
    return count, largest


def source_audit_terms(profile_target, syrup_root, authored_sizes) -> dict[str, int]:
    """Named bytes of ONE fresh `admission.recheck()` (bundle verify x2 incl. package digest, source-census reads, SYRUP tree digest,
    authored-file rehash), from METADATA only. Terms and the lifetimes they count (see `dependency_bundle`, `provenance`,
    `native_case_profile`):

    * manifest_text = 7 M: raw read with `data += chunk` (2 M peak), UTF-8 decode (M), `_canonical(record)` (2 M) and the second
      bounded read compared against it (2 M); M = manifest file bytes.
    * manifest_objects = 16 M: the parsed manifest, its validated path table and the overlay map. SOURCE-SPECIFIC model, not a general
      JSON bound: the manifests are the closed, schema-validated files of an immutable bundle (a dict of fixed keys plus one
      {path, sha256, size}-style entry per file, no deep nesting), whose parsed object graph is several allocated bytes per text byte
      (strings, dict slots; small ints are cached). 16 is a stated modeling coefficient for that shape. CPython can allocate more per
      text byte for hostile JSON (e.g. deep or dense lists/dicts/one-digit values), so no hostile-JSON bound is claimed. The declared
      exclusions are the immutable bundle schema (`dependency_bundle._validate_manifest`) and the preflight caps (manifest <= 16 MiB,
      <= 20,000 files); a manifest outside that closed shape is outside this model's claim. The coefficient was not tuned to a peak;
      measured whole-helper peaks are reported only as a check.
    * largest_file = 4 F: `_read_bounded_regular` accumulation (2 F), the census/overlay whole-file read (F) and one comparison copy
      (F); F = largest regular file under the bundle or the SYRUP package.
    * read_buffers = 2 x 1 MiB: `os.read(fd, 1 MiB)` chunk plus its accumulated copy (hash loops hold one chunk).
    * file_tables = 1 KiB x files in the bundle + SYRUP package (digest entry lists, manifest table, overlay/present sets).
    * authored_reads = 2 x largest authored file (`_read_regular` accumulation).
    * fixed = 256 KiB of non-scaling interpreter objects."""
    target = Path(profile_target)
    mf = target / "bundle-manifest.json"
    m = mf.stat().st_size
    if m > _MAX_PREFLIGHT_MANIFEST_BYTES:
        raise NativeDenseProfileError(f"bundle manifest is {m} bytes, over the {_MAX_PREFLIGHT_MANIFEST_BYTES}-byte preflight cap")
    n1, f1 = _walk_metadata(target)
    n2, f2 = _walk_metadata(Path(syrup_root))
    terms = {"manifest_text": 7 * m, "manifest_objects": _JSON_OBJECT_BYTES_PER_TEXT_BYTE * m, "largest_file": 4 * max(f1, f2),
             "read_buffers": 2 * _READ_CHUNK, "file_tables": _PER_FILE_TABLE_BYTES * (n1 + n2),
             "authored_reads": 2 * max(authored_sizes, default=0), "fixed": _FIXED_AUDIT_BYTES}
    terms["total"] = sum(terms.values())
    return terms


def dense_terms(n_cells: int, n_voxels: int, n_classes: int, ledger_bytes: int, *, has_fixed_base: bool = False,
                has_domain: bool = False) -> dict[str, int]:
    """Named bytes of the Dense stage (N cells, V voxels, K classes; f8 = 8 bytes). Enumerated from the native validators' source:

    Terrain profile only (both flags False for every older profile, which then yields exactly the previous terms):

    * fixed_base_check = 8 x 8 N: `check_committed_reference_consistent_with_fixed_base` (committed inventory summed over classes,
      its conversion, the world sum, `reference + offset`, their difference and four magnitude reductions, each `(ny, nx)`), plus the
      base added once more inside the native initial-committed derivation used for the baseline comparison.
    * domain_check = 7 x 2 x 8 N + 2 x 8 N + N K + 2 N: `check_domain_inventory` reduces each of the seven owners per cell with a max
      and a min (two `(ny, nx)` f8 temporaries each, no N V K boolean), the SYRUP water rule's two masked copies (`(ny, nx)` f8 and
      `(ny, nx, K)` f8 counted as N K bytes of f8 -> 8 N K) and two `(ny, nx)` boolean masks.

    * fractions = 8 N K: the ONE retained new array (`resolve_initial_available_fraction_array`).
    * voxel_totals = 12 x 8 N V: `check_voxel_column_state` holds `mass.sum(-1)`, `local_scale` (maximum + its operand), the
      tolerance helper's float temporaries (<= 3), `capacity - totals`, and the int64 packing codes (where x2, diff) -- counted at 12
      float64-sized N V arrays (bool masks are smaller than the f8 they replace).
    * voxel_masks = 3 x N V K bytes: finite / negative / partial boolean scans over the whole mass array.
    * partition_active = 8 x 8 N K: `check_active_layer_voxel_partition` and the availability/ledger-baseline checks hold <= 8 N K
      float64 temporaries (sums, compares, per-class fold).
    * committed = 12 x 8 N K + 6 x 8 N: the native initial-committed derivation (a second partition check is counted in the two terms
      above; it then creates owned active/voxel inventories, their copies and elevation reference/offset/surface views) plus the
      reconciliation baseline copies and the equality temporaries.
    * ledger = 2 x ledger_bytes: `validate_ledger_state` holds at most one finite/sign temporary per ledger leaf at a time, bounded by
      the leaves' total twice over.
    * fixed = 64 KiB of non-scaling shells."""
    N, V, K = n_cells, n_voxels, n_classes
    terms = {"fractions": 8 * N * K, "voxel_totals": 12 * 8 * N * V, "voxel_masks": 3 * N * V * K,
             "partition_active": 8 * 8 * N * K, "committed": 12 * 8 * N * K + 6 * 8 * N, "ledger": 2 * int(ledger_bytes),
             "fixed": _FIXED_DENSE_BYTES}
    if has_fixed_base:
        terms["fixed_base_check"] = 8 * 8 * N
    if has_domain:
        terms["domain_check"] = 7 * 2 * 8 * N + 2 * 8 * N + 8 * N * K + 2 * N
    terms["total"] = sum(terms.values())
    return terms


def required_bytes(audit: dict[str, int], dense: dict[str, int]) -> int:
    """Stages are sequential (the audit's transients are gone before bed conversion): the call needs the LARGER stage."""
    return max(audit["total"], dense["total"])


# --- scope checks -----------------------------------------------------------------------------------------
def _recheck_admission(admission) -> None:
    try:
        admission.recheck()
    except ncp.NativeCaseProfileError as exc:
        raise NativeDenseProfileError(f"cold source/case recheck refused: {exc}") from exc


def _scope(admission, backend, limits, reserve, available_bytes_fn):
    """Everything refusable from metadata, then the whole-call memory admission. Returns (compiled, meta). No cold source audit, no
    material scan and no allocation of size O(N V K) happens here."""
    if type(backend) is not str or backend != "numpy":
        raise NativeDenseProfileError(f"backend must be exactly 'numpy' (CPU Dense only; no device is touched), got {_r(backend)}")
    if limits is None:
        limits = DenseLimits()
    if type(limits) is not DenseLimits:
        raise NativeDenseProfileError("limits must be a DenseLimits")
    if not _is_int(reserve) or int(reserve) < 0:
        raise NativeDenseProfileError("caller_reserve_bytes must be a nonnegative integer (explicit; there is no default)")
    if available_bytes_fn is not None and not callable(available_bytes_fn):
        raise NativeDenseProfileError("available_bytes_fn must be callable")
    if type(admission) is not ncp.NativeCaseAdmission:
        raise NativeDenseProfileError("a genuine NativeCaseAdmission from native_case_profile is required")
    try:
        compiled = admission.compiled  # authenticates the sealed record
        payload = admission._payload("admission")  # read-only mapping view of the sealed scalars
    except ncp.NativeCaseProfileError as exc:
        raise NativeDenseProfileError(f"unauthenticated admission: {exc}") from exc
    if payload["profile_name"] not in _CPU_ADMISSIONS:  # no device-runtime admission
        raise NativeDenseProfileError(f"this CPU Dense preparation requires an explicit qualified CPU admission, got {payload['profile_name']!r}")
    t = _types()
    if type(compiled) is not t.compiled:
        raise NativeDenseProfileError("admission does not carry the genuine native CompiledCase")
    try:
        meta = _scope_metadata(admission, compiled, t, limits, payload)
    except (AttributeError, TypeError, KeyError, IndexError) as exc:  # a malformed native binding, not a Dense decision
        raise NativeDenseProfileError(f"malformed native case binding: {type(exc).__name__}: {str(exc)[:120]}") from exc
    if available_bytes_fn is None:
        from maple_syrup.subsurface import default_host_available_bytes as available_bytes_fn
    avail = available_bytes_fn()
    if not _is_int(avail) or int(avail) < 0:
        raise NativeDenseProfileError(f"available_bytes_fn must return a nonnegative integer, got {_r(avail)}")
    need = meta["required_bytes"] + int(reserve)
    if need > int(avail):
        raise NativeDenseProfileError(
            f"preparation needs an estimated {need} B (whole call {meta['required_bytes']}: source audit "
            f"{meta['audit']['total']}, dense stage {meta['dense']['total']}; caller reserve {int(reserve)}) but only {int(avail)} B "
            "are reported available; nothing was audited or allocated")
    meta["memory"] = (("available_reading_bytes", int(avail)), ("required_bytes", meta["required_bytes"]),
                      *(("audit_" + k, v) for k, v in meta["audit"].items()), *(("dense_" + k, v) for k, v in meta["dense"].items()),
                      ("resident_bytes_inherited_aliased", meta["resident"]),
                      ("exchange_original_plus_candidate_bytes", 2 * meta["resident"]),
                      ("terrain_resident_bytes_inherited_aliased", meta.get("terrain_bytes", 0)))
    return compiled, meta


def _is_terrain_admission(payload) -> bool:
    return payload["profile_name"] in (ncp.TERRAIN_PROFILE_NAME, ncp.TERRAIN_CUDA_PROFILE_NAME)


def _terrain_metadata(compiled, limits) -> tuple[int, bool, bool]:
    """Terrain profile: the compiled case's terrain owners from METADATA (exact native classes, structural validators, byte
    cap); returns `(terrain_bytes, has_fixed_base, has_domain)`. Admits creep/avalanche/regional context/bounded growth; refuses an
    enabled native water_coupling, an external shear, imported sediment, an airborne seed and LEAF vegetation."""
    tt = _terrain_types()
    cfg = compiled.config
    g = cfg.geometry
    if compiled.leaf_vegetation is not None or compiled.initial_mobile_state is not None:
        raise NativeDenseProfileError("initial case must be bare and carry no airborne seed (the LEAF runtime owns plants)")
    if cfg.water_coupling.enabled is not False or cfg.prescribed_external_shear.enabled is not False:
        raise NativeDenseProfileError("native water_coupling must be disabled and no external shear prescribed (SYRUP owns water)")
    if compiled.case_import.sediment is not None or compiled.case_import.elevation is None:
        raise NativeDenseProfileError("the terrain profile needs an imported elevation and admits no imported sediment")
    if type(cfg.avalanching.enabled) is not bool or type(cfg.surface_creep.enabled) is not bool:
        raise NativeDenseProfileError("avalanching / surface creep must resolve to Booleans")
    growth = g.voxel_growth
    if type(growth.enabled) is not bool or (growth.enabled and growth.max_levels is None):
        raise NativeDenseProfileError("voxel growth must be a bounded native policy (max_levels when enabled)")
    terrain_bytes = 0
    fixed_base, domain = compiled.fixed_base, compiled.domain
    if (compiled.fixed_base_spec.present != (fixed_base is not None)) or (compiled.domain_spec.present is False and domain is not None):
        raise NativeDenseProfileError("compiled terrain owners disagree with the authored sections")
    try:
        if fixed_base is not None:
            if type(fixed_base) is not tt.fixed_base:
                raise ValueError("fixed_base is not exactly the native FixedBaseState")
            tt.validate_fixed_base(fixed_base, int(g.ny), int(g.nx))
            terrain_bytes += _leaf_bytes(fixed_base)
        if domain is not None:
            if type(domain) is not tt.domain:
                raise ValueError("domain is not exactly the native DomainState")
            tt.validate_domain(domain, int(g.ny), int(g.nx))
            terrain_bytes += _leaf_bytes(domain)
    except ValueError as exc:
        raise NativeDenseProfileError(f"native terrain validation refused: {exc}") from exc
    if terrain_bytes > limits.max_terrain_bytes:
        raise NativeDenseProfileError(f"terrain owners ({terrain_bytes} B) exceed the declared max_terrain_bytes={limits.max_terrain_bytes}")
    return terrain_bytes, fixed_base is not None, domain is not None


def _scope_metadata(admission, compiled, t, limits, payload):
    _require_dense(compiled.voxel_column)
    terrain = _is_terrain_admission(payload)
    terrain_bytes, has_fixed_base, has_domain = 0, False, False
    if terrain:
        terrain_bytes, has_fixed_base, has_domain = _terrain_metadata(compiled, limits)
    else:
        if payload["profile_name"] in (ncp.CURRENT_PROFILE_NAME, ncp.CURRENT_CUDA_PROFILE_NAME) and compiled.leaf_vegetation is not None:
            raise NativeDenseProfileError("initial case must be bare; LEAF vegetation is supplied by the owning runtime adapter")
        if compiled.fixed_base_spec.present or compiled.fixed_base is not None or compiled.regional_slope.enabled \
                or compiled.regional_slope_result is not None or compiled.initial_mobile_state is not None:
            raise NativeDenseProfileError("fixed base / regional slope / initial mobile seed are outside this capability")
    cfg = compiled.config
    if not terrain and (cfg.avalanching.enabled is not False or cfg.surface_creep.enabled is not False):
        raise NativeDenseProfileError("avalanching and surface creep must be disabled")
    spec = compiled.sediment_availability_spec
    if spec.builder is not None or spec.builder_by_class:
        raise NativeDenseProfileError("availability builders are unsupported (explicit maps/fractions only)")
    g = cfg.geometry
    if g.dx_m != g.dy_m:
        raise NativeDenseProfileError("non-square cells are unsupported")
    if not terrain and getattr(getattr(g, "voxel_growth", None), "enabled", False):
        raise NativeDenseProfileError("voxel growth must be disabled (no phantom padding)")
    if compiled.water is None:
        raise NativeDenseProfileError("the case must carry a native WaterState")
    for name, cls in (("active_layer", t.active), ("water", t.water), ("sediment_availability", t.availability),
                      ("committed_topography", t.committed)):
        if type(getattr(compiled, name)) is not cls:
            raise NativeDenseProfileError(f"{name} must be exactly the native {cls.__name__}")
    ledger = compiled.sediment_ledger
    ledger_kind = {t.ledger.SedimentLedgerState: "dense", t.ledger.CompactSedimentLedgerState: "compact"}.get(type(ledger))
    if ledger_kind is None:
        raise NativeDenseProfileError("the ledger must be exactly the native Dense or Compact SedimentLedgerState")
    col, act = compiled.voxel_column.mass_kg, compiled.active_layer.mass_kg
    for name, a in (("voxel mass", col), ("active mass", act), ("water depth", compiled.water.depth_m),
                    ("mobile mass", compiled.water.mobile_mass_by_cell_class_kg)):
        if type(a) is not np.ndarray or a.dtype != np.float64:
            raise NativeDenseProfileError(f"{name} must be a float64 NumPy array (host Dense only)")
    if col.ndim != 4:
        raise NativeDenseProfileError("voxel mass must be a rank-4 (ny, nx, n_voxels, n_classes) array")
    ny, nx, V, K = col.shape
    if (ny, nx) != (g.ny, g.nx) or act.shape != (ny, nx, K) or K != len(cfg.grain_classes.classes):
        raise NativeDenseProfileError("voxel/active/geometry/class shapes disagree")
    if ny * nx > limits.max_cells or col.size > limits.max_voxel_elements or K > limits.max_classes:
        raise NativeDenseProfileError(f"case ({ny}x{nx} cells, {V} voxels, {K} classes) exceeds the declared DenseLimits")
    want = compiled.expected_total_mass_by_class_kg
    if type(want) is not np.ndarray or want.dtype != np.float64 or want.shape != (K,):
        raise NativeDenseProfileError(f"expected_total_mass_by_class_kg must be a float64 NumPy array of shape ({K},), got {_r(want)}")
    if not np.isfinite(want).all() or (want < 0.0).any():  # O(K) scan, K <= limits.max_classes; no repair, no clipping
        raise NativeDenseProfileError("expected_total_mass_by_class_kg must be finite and nonnegative")
    trig = float(cfg.topographic_commit.pending_elevation_trigger_m)
    if not (np.isfinite(trig) and trig > 0):
        raise NativeDenseProfileError("pending_elevation_trigger_m must be finite and positive")
    import maple_syrup

    audit = source_audit_terms(payload["profile_target"], Path(maple_syrup.__file__).resolve().parent,
                               [size for _n, _p, _h, size in payload["files"]])
    ledger_bytes = _leaf_bytes(ledger)
    dense = dense_terms(ny * nx, V, K, ledger_bytes, has_fixed_base=has_fixed_base, has_domain=has_domain)
    # The terrain owners are RESIDENT compiled arrays, aliased by reference on NumPy (listed, never charged as new).
    resident = _leaf_bytes(compiled.voxel_column) + _leaf_bytes(compiled.active_layer) + _leaf_bytes(compiled.water) \
        + _leaf_bytes(compiled.sediment_availability) + ledger_bytes + _leaf_bytes(compiled.committed_topography) + terrain_bytes
    return dict(bundle_id=admission.bundle_id, shape=(ny, nx), n_voxels=V, n_classes=K, ledger_kind=ledger_kind, audit=audit,
                dense=dense, required_bytes=required_bytes(audit, dense), resident=resident, terrain_bytes=terrain_bytes,
                class_ids=tuple(c.class_id for c in cfg.grain_classes.classes))


def _validate_terrain(state, context, meta) -> None:
    """Terrain part of the validation suite (the NATIVE validators are the authority): the committed reference must be MAPLE's own
    world reference of the inventories with the base; every sediment owner must be exactly zero outside the footprint (native
    footprint inventory check on host arrays); every water leaf on NoData must be exactly zero (fused predicate, no mask download)."""
    from maple_syrup import sediment_bed as sb

    fixed_base, domain = getattr(context, "fixed_base", None), getattr(context, "domain", None)
    if fixed_base is None and domain is None:
        return
    tt = _terrain_types()
    ny, nx = meta["shape"]
    try:
        if fixed_base is not None:
            tt.validate_fixed_base(fixed_base, ny, nx)
            tt.check_world_reference(state.committed_topography, context.geometry, fixed_base, n_voxels=meta["n_voxels"],
                                     operation="SYRUP terrain Dense preparation")
        if domain is not None:
            tt.validate_domain(domain, ny, nx)
            if isinstance(domain.valid_mask, np.ndarray):  # host bed: the native footprint inventory validator on the actual arrays
                committed = state.committed_topography
                tt.check_domain_inventory(
                    domain.valid_mask, context="SYRUP terrain Dense preparation",
                    voxel_mass_kg=state.voxel_column.mass_kg, active_layer_mass_kg=state.active_layer.mass_kg,
                    available_mass_kg=state.sediment_availability.available_mass_kg, bound_mass_kg=state.sediment_availability.bound_mass_kg,
                    committed_active_layer_mass_kg=committed.committed_active_layer_mass_by_class_kg,
                    committed_voxel_mass_kg=committed.committed_voxel_mass_by_class_kg)
    except ValueError as exc:
        raise NativeDenseProfileError(f"native terrain validator refused the case bed: {str(exc)[:200]}") from exc
    if sb.water_outside_domain_nonzero(state, context):
        raise NativeDenseProfileError("water is nonzero on a NoData cell of the domain footprint; all water on NoData must stay zero")


def _check_identity(state, compiled, context=None) -> None:
    """One authority, no clone: the bed leaves ARE the compiled case's arrays (and, on a terrain context, so are the terrain owners)."""
    if state.voxel_column.mass_kg is not compiled.voxel_column.mass_kg or state.active_layer.mass_kg is not compiled.active_layer.mass_kg:
        raise NativeDenseProfileError("the bed no longer shares the compiled case's native material arrays (memory/authority contract)")
    if context is not None:
        for name in ("fixed_base", "domain"):
            mine, theirs = getattr(context, name, None), getattr(compiled, name, None)
            if (mine is None) != (theirs is None):
                raise NativeDenseProfileError(f"the bed context's {name} does not match the compiled case")
            if mine is not None and any(getattr(mine, f.name) is not getattr(theirs, f.name) for f in dataclasses.fields(mine)
                                        if isinstance(getattr(mine, f.name), np.ndarray)):
                raise NativeDenseProfileError(f"the bed context's {name} is not the compiled case's own array (memory/authority contract)")


def _validate_bed(state, context, compiled, meta) -> None:
    """The ONE full native validation suite (preparation and recheck). Native public validators; a failure is a typed refusal and
    nothing accepted is mutated (every call is read-only)."""
    from maple.coupling.sediment_ledger.validation import validate_ledger_state
    from maple.core.types.ledger_reconciliation import (
        ledger_reconciliation_baseline_from_committed_state,
        reconciled_inventory_by_class_kg,
        validate_ledger_reconciliation_baseline,
    )
    from maple.core.types.topographic_commit import validate_committed_topography_state
    from maple.surface.active_layer.validation import check_active_layer_voxel_partition
    from maple.surface.availability.validation import check_sediment_availability_state
    from maple.surface.topographic_commit import initial_committed_topography_state_from_physical_state as native_initial
    from maple.water import validate_water_state

    from maple_syrup import sediment_bed as sb
    from maple_syrup.conservation import reservoir_bound_kg

    ny, nx = meta["shape"]
    K, V = meta["n_classes"], meta["n_voxels"]
    g, mres = context.geometry, context.mass_resolution_kg
    axes = (ny, nx, K)
    try:
        check_active_layer_voxel_partition(state.active_layer, state.voxel_column, g, mres)  # includes check_voxel_column_state
        if tuple(validate_ledger_state(state.ledger)) != axes:
            raise ValueError("ledger axes differ from the bed (ny, nx, n_classes)")
        check_sediment_availability_state(state.sediment_availability, state.active_layer, mres)
        validate_water_state(state.water, ny, nx, K)
        committed = state.committed_topography
        if tuple(validate_committed_topography_state(committed)) != axes:
            raise ValueError("committed topography axes differ from the bed (ny, nx, n_classes)")
        # The advertised profile is a FRESH case: the committed baseline must be exactly what the native initial derivation yields
        # from this very physical state (no pending history, clock 0, zero offset) -- finite-but-inconsistent baselines fail here.
        # With a fixed base the native derivation yields the WORLD reference (keyword present only on a terrain context).
        fresh = native_initial(state.voxel_column, state.active_layer, g, mres, **sb.terrain_kwargs(context, domain=False))
        for f in dataclasses.fields(committed):
            if not np.array_equal(getattr(committed, f.name), getattr(fresh, f.name)):
                raise ValueError(f"committed topography {f.name} differs from the native initial state of this physical bed")
        baseline = ledger_reconciliation_baseline_from_committed_state(committed)
        if tuple(validate_ledger_reconciliation_baseline(baseline)) != axes:
            raise ValueError("ledger reconciliation baseline axes differ from the bed")
        physical = state.active_layer.mass_kg + state.voxel_column.mass_kg.sum(axis=-2)
        reconciled = reconciled_inventory_by_class_kg(baseline)
        if np.max(np.abs(reconciled - physical), initial=0.0) > reservoir_bound_kg(1 + V, reconciled, physical):
            raise ValueError("reconciled baseline inventory differs from the physical bed beyond the MAPLE roundoff bound")
    except NativeDenseProfileError:
        raise
    except Exception as exc:
        raise NativeDenseProfileError(f"native validator refused the case bed: {type(exc).__name__}: {str(exc)[:200]}") from exc
    if np.any(state.water.mobile_mass_by_cell_class_kg != 0.0) or np.any(state.ledger.pending_bed_mass_change_kg != 0.0):
        raise NativeDenseProfileError("an initial airborne/mobile load or pending ledger change is outside this capability")
    if np.any(~np.isfinite(context.initial_available_fraction)) or context.initial_available_fraction.shape != axes:
        raise NativeDenseProfileError("initial available fraction is not a finite (ny, nx, K) array")
    inv = np.asarray(sb.bed_inventory(state), dtype=np.float64)
    want = compiled.expected_total_mass_by_class_kg  # type/dtype/shape/finite/nonnegative already refused in `_scope_metadata`
    if inv.shape != want.shape or np.max(np.abs(inv - want), initial=0.0) > reservoir_bound_kg(ny * nx * V, inv, want):
        raise NativeDenseProfileError("bed class totals differ from the native expected class totals")
    _validate_terrain(state, context, meta)


def plan_preparation(admission, *, caller_reserve_bytes: int = 0, available_bytes_fn: Callable[[], int] | None = None) -> dict[str, Any]:
    """The whole-call requirement of `prepare_native_dense_case` for this admission, from metadata only (no cold audit, no material
    scan, no allocation of size O(N V K)): named audit/dense terms, ``required_bytes`` and the inherited resident bytes. Raises the
    same typed refusals as preparation, including the memory refusal when `available_bytes_fn` reports less than required + reserve."""
    _compiled, meta = _scope(admission, "numpy", None, caller_reserve_bytes, available_bytes_fn if available_bytes_fn is not None
                             else (lambda: 1 << 62))
    return {"required_bytes": meta["required_bytes"], "audit": dict(meta["audit"]), "dense": dict(meta["dense"]),
            "resident_bytes_inherited_aliased": meta["resident"], "shape": meta["shape"], "n_voxels": meta["n_voxels"],
            "n_classes": meta["n_classes"]}


def prepare_native_dense_case(admission, *, caller_reserve_bytes: int, backend: str = "numpy", limits: DenseLimits | None = None,
                              available_bytes_fn: Callable[[], int] | None = None) -> NativeDensePreparation:
    """Genuine admitted CPU (9f or 1dca) case -> sealed Dense `(BedState, BedContext)` preparation (see module notes).

    Order: argument/scope/geometry/class/backend refusals and the WHOLE-CALL memory admission (source audit + dense stage) from
    METADATA -> fresh cold `admission.recheck()` -> unchanged `bed_from_case` (NumPy) -> the full native validation suite -> seal.
    Any failure leaves the admission and the compiled case untouched (nothing here writes to them)."""
    compiled, meta = _scope(admission, backend, limits, caller_reserve_bytes, available_bytes_fn)
    _recheck_admission(admission)
    from maple_syrup import sediment_bed as sb

    try:
        state, context = sb.bed_from_case(compiled, xp=np)
    except sb.BedIntegrationError as exc:
        raise NativeDenseProfileError(f"bed_from_case refused: {exc}") from exc
    try:
        windowed = sb.is_windowed_bed(state)
    except sb.BedIntegrationError as exc:
        raise NativeDenseProfileError(f"unsupported bed representation: {exc}") from exc
    if type(state) is not sb.BedState or type(context) is not sb.BedContext or windowed:
        raise NativeDenseProfileError("bed_from_case did not return a genuine Dense BedState/BedContext")
    if context.depth_update_rule != "constant_depth":
        raise NativeDenseProfileError("constant_depth is required for SYRUP hydraulic ownership")
    _check_identity(state, compiled, context)
    _validate_bed(state, context, compiled, meta)
    obj = object.__new__(NativeDensePreparation)
    _SEALED[obj] = types.MappingProxyType(dict(
        profile_name=_DENSE_PROFILE_BY_ADMISSION[admission.profile_name],
        bundle_id=meta["bundle_id"], package_digest=admission.package_digest,
        syrup_package_digest=admission.syrup_package_digest, case_identity_sha256=admission.case_identity_sha256,
        latitude_deg=admission.latitude_deg, route=admission.route, shape=meta["shape"], n_voxels=meta["n_voxels"],
        n_classes=meta["n_classes"], class_ids=meta["class_ids"], ledger_kind=meta["ledger_kind"], backend="numpy",
        memory=meta["memory"], caller_reserve_bytes=int(caller_reserve_bytes), bed=state, context=context, admission=admission,
        terrain=admission.terrain))
    return obj
