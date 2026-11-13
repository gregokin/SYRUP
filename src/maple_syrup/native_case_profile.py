"""Opt-in native MAPLE 9f466d2c source profiles + ordinary-case/latitude admission (CANDIDATE).

SCOPE. Two explicit, literal source profiles that share every pinned 9f check: the DEFAULT CPU profile
`native_maple_9f_case_latitude_cpu_v1` (unchanged record/behavior; refuses a process that imported cupy) and the separate device-runtime
SOURCE profile `native_maple_9f_case_latitude_cuda_v1`. Each (1) freshly verifies the exact committed MAPLE
``9f466d2cfcd52991b24496cf67f4562d6268afa9`` source/runtime that THIS process imported (the SAME real MAPLE package for either profile) and
(2) admits one ordinary native case (three-file package or single file) through MAPLE's own resolve -> compile -> load, binding the
authored ``case.yaml`` (path, SHA-256, ``location.latitude_deg``) to the same compiled landscape.

The CUDA name only PERMITS a CuPy import in the process. It performs no GPU execution, creates no device arrays, and asserts neither that
CuPy is installed/authentic nor that a device is available; the compiled landscape stays host data. Neither profile is native water
storage/exchange, a codec, wind, a global digest whitelist, a production default, a MASY bridge, a CLI or new physics preparation; those keep
refusing under the existing 39bcf79 gates, which this module neither imports nor edits. No GPU, Git, install or upstream change.

One process, one source profile: MAPLE is imported once per process; this module never swaps ``sys.modules``. Build/export/
materialization are orchestrated OUTSIDE (``dependency_bundle``); this module consumes a strictly verified materialized
target and re-verifies it at every admission (cold, no cached trust).

Ownership: MAPLE initializes the landscape (the genuine ``CompiledCase`` is returned by reference, no copy); SYRUP owns
water; LEAF/MASY are not imported. Latitude is native descriptive metadata; this module exposes the validated
source association (`NativeCaseAdmission.latitude_binding`) for a later MASY join and never defaults an omitted latitude.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import stat
import sys
import types
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from maple_syrup import dependency_bundle
from maple_syrup.provenance import source_tree_digest

__all__ = [
    "CUDA_SOURCE_PROFILE_NAME", "REFERENCE_BUNDLE_ID", "COMPILED_CASE_FIELDS", "NATIVE_COMMIT", "PACKAGE_DIGEST", "PROFILE_NAME", "RESOLVED_PACKAGE_FIELDS",
    "CaseLimits", "NativeCaseAdmission", "NativeCaseProfileError", "NativeSourceProfile", "admit_package_case",
    "admit_single_file_case", "inspect_headers", "load_admitted_case", "verify_source_profile",
]

PROFILE_NAME = "native_maple_9f_case_latitude_cpu_v1"
#: Device-runtime SOURCE capability only: same pinned 9f source checks, but a process that imported CuPy is not refused. It does not
#: assert CuPy is installed/authentic or that any device exists, and it prepares no device arrays.
CUDA_SOURCE_PROFILE_NAME = "native_maple_9f_case_latitude_cuda_v1"
_SOURCE_PROFILES = {PROFILE_NAME: False, CUDA_SOURCE_PROFILE_NAME: True}  # literal name -> device-runtime capability
NATIVE_COMMIT = "9f466d2cfcd52991b24496cf67f4562d6268afa9"
PACKAGE_DIGEST = "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"
PROJECT_DIGEST = "90fa69cf32afbec3bf5fb5a3cf682de44a80ea286322547e3c1cebb791241d25"
ARCHIVE_SHA256 = "5f7eaaf5a964c42fede434b737272a4ff9913c99a1dce65fec9b9e95ed31ecf8"
DISTRIBUTION_SHA256 = {
    "METADATA": "dc1b5653aab75868578b7d478b3d015b146ba1fe4ea4907cc26d806d68120cca",
    "WHEEL": "b59a1e1a3b56c5645f769944ec4ddde3954f94b3509ef6ca8989f1ee0c00cbc0",
    "top_level.txt": "08ab2a56c8503c742c762013475ad2955377d2001a9fc03e4e8bc22e09af0844",
    "entry_points.txt": "9430d41e28e96a53fa455b84c6a4b6122484d21e6ce8151c6c40854db190c870"}
PYPROJECT_SHA256 = "8779c55d823bf610da9c9f087c7d5eaaf8d276bfbdc6fb428726e4e7c06350be"
#: Informational: the exported reference bundle. bundle_id also hashes the exporting tool/path/python, so it is RECORDED, never
#: a gate; the environment-independent content pins (archive/project/package/pyproject/distribution) are the gates.
REFERENCE_BUNDLE_ID = "55d1485a03c385240bab57edf7fba6cad5230628729b30b8a19d96d094ddaa5c"
PACKAGE_FILE_COUNT = 412
#: Real native dataclass fields censused from the frozen 9f source. A different tuple means a different native API.
RESOLVED_PACKAGE_FIELDS = (
    "config", "topography", "grain_mixture", "stratigraphy", "threshold_modifiers", "vegetation", "sediment_availability",
    "initial_mobile_seed", "water_initial_depth", "prescribed_external_shear", "requested_diagnostics", "movie_request",
    "wind_record", "case_import", "import_sources", "case_yaml_path", "model_yaml_path", "wind_yaml_path", "location",
    "regional_slope", "roughness_ownership", "slope_yaml_path", "slope_yaml_sha256", "fixed_base")
COMPILED_CASE_FIELDS = (
    "case_path", "processed_path", "diagnostics_path", "provenance_path", "case_name", "code_version", "config",
    "topography", "grain_mixture", "coordinates", "voxel_column", "active_layer", "committed_topography",
    "sediment_ledger", "vegetation", "sediment_availability_spec", "sediment_availability", "water_initial_depth_spec",
    "water", "threshold_modifier_spec", "threshold_modifier_state", "stratigraphy", "initial_mobile_seed_spec",
    "initial_mobile_state", "case_import", "import_sources", "masks", "transform_logs", "masks_path", "topography_result",
    "expected_total_mass_by_class_kg", "expected_mass_by_cell_class_kg", "expected_mass_by_level_class_kg",
    "provenance_record", "compile_seconds", "location", "regional_slope", "regional_slope_result", "roughness_ownership",
    "fixed_base_spec", "fixed_base", "fixed_base_source", "fixed_base_admission", "fixed_base_load_admission")

_ROUTES = ("package", "single_file")
_PACKAGE_FILES = ("case.yaml", "model.yaml", "wind.yaml")
_OUTPUT_NAMES = ("processed", "diagnostics", "masks", "provenance.yaml")
#: Authored case-level sections the first profile refuses BEFORE native compilation (presence alone refuses).
_REFUSED_CASE_KEYS = {
    "fixed_base": "an authored fixed base (nonerodible world datum) is not in the first profile",
    "import": "empirical import is not in the first profile",
    "initial_mobile_seed": "an initial airborne/mobile seed is not in the first profile",
    "regional_slope": "regional_slope is not in the first profile",
}


class NativeCaseProfileError(RuntimeError):
    """Wrong/mixed/mutated source or runtime, unsupported case, or an inconsistent native result (refused)."""


@dataclass(frozen=True)
class CaseLimits:
    """Bounds applied to authored files BEFORE any parse or native call (not a whole-model memory bound)."""

    max_case_bytes: int = 1 << 20
    max_model_bytes: int = 1 << 20
    max_wind_bytes: int = 4 << 20
    max_yaml_nodes: int = 200_000
    max_yaml_depth: int = 64

    def __post_init__(self):
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if type(v) is not int or v <= 0:
                raise NativeCaseProfileError(f"{f.name} must be a positive int, got {v!r}")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular(path: Path, cap: int, what: str) -> bytes:
    """Bounded read of a regular, non-link file; the SAME bytes are hashed and parsed."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise NativeCaseProfileError(f"{what} {path} cannot be opened as a regular file: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise NativeCaseProfileError(f"{what} {path} is not a regular file")
        if st.st_size > cap:
            raise NativeCaseProfileError(f"{what} {path} is {st.st_size} bytes, over its {cap}-byte cap")
        # size was already checked against the cap; read st_size+1 so growth is detected without allocating the cap
        data = b""
        while len(data) <= st.st_size:
            more = os.read(fd, st.st_size + 1 - len(data))
            if not more:
                break
            data += more
    finally:
        os.close(fd)
    if len(data) != st.st_size:
        raise NativeCaseProfileError(f"{what} {path} changed size or exceeds its cap while being read")
    return data


def _bounded_yaml(data: bytes, limits: CaseLimits, what: str) -> Any:
    """Event-stream audit (nodes, depth, NO anchors/aliases/tags-as-objects) then safe_load of the same bytes."""
    nodes = depth = 0
    try:
        for ev in yaml.parse(data):
            if isinstance(ev, yaml.AliasEvent) or getattr(ev, "anchor", None):
                raise NativeCaseProfileError(f"{what}: YAML anchors/aliases are refused")
            if isinstance(ev, (yaml.ScalarEvent, yaml.SequenceStartEvent, yaml.MappingStartEvent)):
                nodes += 1
                if nodes > limits.max_yaml_nodes:
                    raise NativeCaseProfileError(f"{what}: more than {limits.max_yaml_nodes} YAML nodes")
            if isinstance(ev, (yaml.SequenceStartEvent, yaml.MappingStartEvent)):
                depth += 1
                if depth > limits.max_yaml_depth:
                    raise NativeCaseProfileError(f"{what}: YAML deeper than {limits.max_yaml_depth}")
            elif isinstance(ev, (yaml.SequenceEndEvent, yaml.MappingEndEvent)):
                depth -= 1
        doc = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        raise NativeCaseProfileError(f"{what}: invalid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise NativeCaseProfileError(f"{what} must parse to a mapping, got {type(doc).__name__}")
    return doc


# --- sealed records -------------------------------------------------------------------------------------
# Public records are NOT dataclasses and hold NO data slots: their payload lives in this private registry, written only by
# the factories. A copied/replaced/pickled/`object.__new__` instance is not registered and every property/method refuses it.
_SEALED: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


class _Sealed:
    __slots__ = ("__weakref__",)
    _KIND = "record"
    _FIELDS: tuple[str, ...] = ()

    def __init__(self, *args, **kwargs):
        raise NativeCaseProfileError(f"{type(self).__name__} is created only by its factory")

    def _payload(self, kind: str) -> "types.MappingProxyType":
        """Authenticate that `self` is a registered record of EXACTLY `kind`; returns a READ-ONLY view of its sealed payload."""
        p = _SEALED.get(self) if type(self) in (NativeSourceProfile, NativeCaseAdmission) else None
        if p is None or p["_kind"] != kind or self._KIND != kind:
            raise NativeCaseProfileError(f"unauthenticated {type(self).__name__} (not a factory-produced {kind} record)")
        return p

    def __setattr__(self, name, value):
        raise NativeCaseProfileError(f"{type(self).__name__} is immutable")

    __delattr__ = __setattr__

    def __copy__(self):
        raise NativeCaseProfileError(f"{type(self).__name__} cannot be copied; re-run the factory")

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __reduce_ex__(self, protocol):
        return self.__copy__()

    def __repr__(self):
        p = _SEALED.get(self)
        return f"<{type(self).__name__} {'sealed' if p else 'UNAUTHENTICATED'}>"

    def __dir__(self):
        return list(self._FIELDS)


def _make(cls, payload: dict):
    obj = object.__new__(cls)
    # Read-only view over a dict no one else holds: scalars/tuples/strings only, plus the genuine `compiled` BY REFERENCE.
    _SEALED[obj] = types.MappingProxyType({**payload, "_kind": cls._KIND})
    return obj


def _prop(name, kind):
    def get(self):
        return self._payload(kind)[name]
    return property(get)


class NativeSourceProfile(_Sealed):
    """Result of `verify_source_profile`: a sealed, immutable cold-verification record (factory-only, see module notes)."""

    __slots__ = ()
    _KIND = "profile"
    _FIELDS = ("profile", "maple_commit", "bundle_id", "target", "package_digest", "project_digest", "syrup_package_digest",
               "syrup_package_root", "maple_modules", "dependency", "python")
    for _n in _FIELDS:
        locals()[_n] = _prop(_n, "profile")
    del _n

    @property
    def device_runtime_capability(self) -> bool:
        """True only for the explicit CUDA source profile (not in `dir()`/payload: the CPU record shape is unchanged)."""
        return _SOURCE_PROFILES[self._payload("profile")["profile"]]

    def recheck(self) -> None:
        """Cold re-verification of the same target and SYRUP tree under this record's OWN profile; refuses any drift."""
        p = self._payload("profile")
        fresh = verify_source_profile(p["target"], expected_syrup_digest=p["syrup_package_digest"], profile=p["profile"])
        fp = fresh._payload("profile")
        # The imported-module census may legitimately GROW (native compile imports more modules); every earlier entry
        # must be unchanged, and each new one was verified against the bundle manifest by the fresh verification.
        keys = ("profile", "maple_commit", "bundle_id", "target", "package_digest", "project_digest", "syrup_package_digest",
                "syrup_package_root", "dependency", "python")
        if any(fp[k] != p[k] for k in keys) or not set(p["maple_modules"]) <= set(fp["maple_modules"]):
            raise NativeCaseProfileError("source profile no longer matches a fresh verification")


def _census(sys_modules, root: Path, table: dict[str, str]) -> list[tuple[str, str]]:
    """Fail-closed census of every imported `maple`/`maple.*` entry (see module notes)."""
    out = []
    for name in sorted(sys_modules):
        if name != "maple" and not name.startswith("maple."):
            continue
        mod = sys_modules[name]
        if type(mod) is not types.ModuleType:
            raise NativeCaseProfileError(f"imported {name} is not a plain module object")
        rel = name.split(".")[1:]
        spec = getattr(mod, "__spec__", None)
        f = getattr(mod, "__file__", None)
        if f is None:
            # The pinned 9f tree has NO namespace packages (every package has a hashed __init__.py), so EVERY originless
            # maple.* entry is refused: a regular package cannot masquerade as an unhashed namespace. Namespace support
            # would need a separate upstream-source qualification.
            raise NativeCaseProfileError(f"imported {name} has no __file__; originless/namespace modules are refused in this profile")
        fp = Path(f).resolve()
        if spec is None or spec.name != name or spec.origin is None or Path(spec.origin).resolve() != fp:
            raise NativeCaseProfileError(f"imported {name} has an inconsistent spec/origin")
        if not fp.is_relative_to(root):
            raise NativeCaseProfileError(f"imported module {name} comes from {fp}, outside the verified bundle")
        expected = {root.joinpath(*rel).with_suffix(".py") if rel else None, root.joinpath(*rel, "__init__.py")}
        if fp not in expected:
            raise NativeCaseProfileError(f"imported module {name} is bound to the wrong file {fp}")
        want = table.get("source/src/maple/" + fp.relative_to(root).as_posix())
        have = _sha(fp.read_bytes())
        if want is None or have != want:
            raise NativeCaseProfileError(f"imported module {name} does not match the bundle manifest")
        out.append((name, have))
    return out


def verify_source_profile(target: str | os.PathLike, *, expected_syrup_digest: str | None = None,
                          profile: str = PROFILE_NAME) -> NativeSourceProfile:
    """Freshly verify the materialized 9f bundle AND that this process imported exactly it.

    Refuses (before any case work): not materialized/mutated bundle, wrong bundle/origin/revision/distribution/recipe,
    a different package or pyproject digest, any imported ``maple.*`` module that is not the bundle's own byte-identical
    file (or a verified namespace package), a ``cupy`` import (CPU-only profile), or a SYRUP tree that differs from
    `expected_syrup_digest` when given. `profile` is one of the two literal names (default CPU, unchanged); only the CPU profile
    refuses a process that imported cupy. The CUDA name is a source capability, not a claim that CuPy or a device is usable."""
    if type(profile) is not str or profile not in _SOURCE_PROFILES:
        raise NativeCaseProfileError(f"profile must be exactly one of {tuple(_SOURCE_PROFILES)}, got {profile!r}")
    if not _SOURCE_PROFILES[profile] and "cupy" in sys.modules:
        raise NativeCaseProfileError("the CPU-only 9f case profile refuses a process that imported cupy")
    target_path = Path(target).resolve()
    try:
        runtime = dependency_bundle.check_runtime(target_path)
        manifest = dependency_bundle.verify_materialized(target_path)
    except Exception as exc:  # BundleError, MapleDependencyError, MapleApiIncompatibleError, OSError
        raise NativeCaseProfileError(f"materialized bundle/runtime check failed: {type(exc).__name__}: {exc}") from exc
    origin = manifest["origin"]
    if manifest["bundle_id"] != runtime["bundle_id"] or target_path.name != manifest["bundle_id"]:
        raise NativeCaseProfileError("materialized bundle identity is inconsistent")
    if origin["original_source"]["revision"] != NATIVE_COMMIT or origin["patch_chain"]:
        raise NativeCaseProfileError("bundle origin is not the unpatched committed 9f466d2c archive")
    proj = manifest["project"]
    if (proj["package_source_digest"]["digest_sha256"] != PACKAGE_DIGEST or runtime["package_source_digest"] != PACKAGE_DIGEST
            or proj["pyproject_sha256"] != PYPROJECT_SHA256 or proj["package_source_digest"]["file_count"] != PACKAGE_FILE_COUNT
            or proj["project_source_digest"]["digest_sha256"] != PROJECT_DIGEST or manifest["archive"]["sha256"] != ARCHIVE_SHA256):
        raise NativeCaseProfileError("bundle project/package digest is not the pinned 9f source")
    dist = {e["path"][len("distribution/"):]: e["sha256"] for e in manifest["files"] if e["path"].startswith("distribution/")}
    if dist != DISTRIBUTION_SHA256:
        raise NativeCaseProfileError("bundle distribution metadata is not the pinned recorded metadata")
    root = target_path / "source" / "src" / "maple"
    census = _census(sys.modules, root, {e["path"]: e["sha256"] for e in manifest["files"]})
    if not census:
        raise NativeCaseProfileError("no maple module was imported from the bundle")
    import maple_syrup
    syrup_root = Path(maple_syrup.__file__).resolve().parent
    syrup = source_tree_digest(syrup_root).digest_sha256
    if expected_syrup_digest is not None and syrup != expected_syrup_digest:
        raise NativeCaseProfileError("SYRUP package digest differs from the expected source")
    return _make(NativeSourceProfile, dict(
        profile=profile, maple_commit=NATIVE_COMMIT, bundle_id=manifest["bundle_id"], target=str(target_path),
        package_digest=PACKAGE_DIGEST, project_digest=proj["project_source_digest"]["digest_sha256"],
        syrup_package_digest=syrup, syrup_package_root=str(syrup_root), maple_modules=tuple(census),
        dependency=json.dumps(runtime["dependency"], sort_keys=True), python=sys.version.split()[0]))


def _require_profile(profile: Any) -> None:
    if type(profile) is not NativeSourceProfile:
        raise NativeCaseProfileError("a NativeSourceProfile from verify_source_profile is required")
    profile._payload("profile")
    profile.recheck()


# --- header gate (ONE gate for compile and load) ----------------------------------------------------------
@dataclass(frozen=True)
class _Headers:
    case_dir: str
    files: tuple[tuple[str, str, str, int], ...]  # (name, canonical path, sha256, size)
    case_raw: Any
    model_raw: Any


def _disabled(doc: dict, key: str) -> bool:
    section = doc.get(key)
    return isinstance(section, dict) and section.get("enabled") is False


def _check_route(route: Any) -> str:
    if type(route) is not str or route not in _ROUTES:
        raise NativeCaseProfileError(f"route must be exactly one of {_ROUTES}, got {route!r}")
    return route


def _gate(case_dir, route, limits: CaseLimits | None, *, existing_output: bool) -> _Headers:
    """Pin and inspect the authored files with bounded reads BEFORE any native resolve/compile/load or output write.

    `existing_output=False` (compile): no compiled output may exist. `True` (load): processed/diagnostics/provenance.yaml must
    exist and are never rewritten. Everything else is IDENTICAL, so load cannot use a weaker parser."""
    route = _check_route(route)
    limits = limits or CaseLimits()
    base = Path(case_dir)
    if os.path.islink(base) or not base.is_dir():
        raise NativeCaseProfileError(f"{base} is not a real case directory")
    base = base.resolve()
    caps = {"case.yaml": limits.max_case_bytes, "model.yaml": limits.max_model_bytes, "wind.yaml": limits.max_wind_bytes}
    names = _PACKAGE_FILES if route == "package" else ("case.yaml",)
    if os.path.lexists(base / "slope.yaml"):
        raise NativeCaseProfileError("an authored slope.yaml (or a link of that name) is not in the first profile")
    if route == "single_file" and (os.path.lexists(base / "model.yaml") or os.path.lexists(base / "wind.yaml")):
        raise NativeCaseProfileError("single-file route refuses sibling model.yaml/wind.yaml (ambiguous route)")
    if existing_output:
        for out in ("processed", "diagnostics", "provenance.yaml"):
            if not os.path.lexists(base / out):
                raise NativeCaseProfileError(f"{base / out} is missing; compile through admit_*_case first")
    else:
        for out in _OUTPUT_NAMES:
            if os.path.lexists(base / out):
                raise NativeCaseProfileError(f"{base / out} already exists; compiled output is never overwritten")
    docs, files = {}, []
    for name in names:
        data = _read_regular(base / name, caps[name], name)
        files.append((name, str(base / name), _sha(data), len(data)))
        docs[name] = _bounded_yaml(data, limits, name)
    case_raw, model_raw = docs["case.yaml"], docs.get("model.yaml", {})
    for key, why in _REFUSED_CASE_KEYS.items():
        if key in case_raw or key in model_raw:
            raise NativeCaseProfileError(f"authored `{key}` refused: {why}")
    # Defaults of both are ENABLED natively, so the first profile requires them authored disabled (in either file).
    for key in ("avalanching", "surface_creep"):
        if not (_disabled(case_raw, key) or _disabled(model_raw, key)):
            raise NativeCaseProfileError(f"`{key}: {{enabled: false}}` must be authored (native default is enabled)")
    veg = case_raw.get("vegetation")
    if veg not in (None, {}) and not (isinstance(veg, dict) and veg.get("source") in (None, "none")):
        raise NativeCaseProfileError("only bare/neutral vegetation is in the first profile")
    return _Headers(case_dir=str(base), files=tuple(files), case_raw=case_raw, model_raw=model_raw)


def inspect_headers(case_dir, *, route: str, limits: CaseLimits | None = None) -> _Headers:
    """Compile-side gate (no compiled output may exist). Kept as the public name of the shared gate."""
    return _gate(case_dir, route, limits, existing_output=False)


# --- admission -------------------------------------------------------------------------------------------
class NativeCaseAdmission(_Sealed):
    """Sealed source-bound association of one authored case with the genuine native compiled landscape.

    `compiled` is the actual `CompiledCase` (by reference; no copied bed or physics). The seal binds the COLD association at
    admission; `recheck()` is the caller's continuing fresh check (no global CAS or durable replay is claimed)."""

    __slots__ = ()
    _KIND = "admission"
    _FIELDS = ("profile_name", "bundle_id", "package_digest", "syrup_package_digest", "route", "case_dir", "files",
               "case_identity_sha256", "latitude_deg", "compiled", "resolved_roughness_owner", "provenance_code_version")
    for _n in _FIELDS:
        locals()[_n] = _prop(_n, "admission")
    del _n

    @property
    def device_runtime_capability(self) -> bool:
        """True only when admitted under the explicit CUDA source profile (derived from the sealed profile name)."""
        return _SOURCE_PROFILES[self._payload("admission")["profile_name"]]

    @property
    def case_yaml(self) -> tuple[str, str]:
        _name, path, sha, _ = self._payload("admission")["files"][0]
        return path, sha

    def latitude_binding(self) -> dict[str, Any]:
        """Exact authored source association for a future MASY join (no MASY import). Raises if latitude was omitted."""
        p = self._payload("admission")
        if p["latitude_deg"] is None:
            raise NativeCaseProfileError("the case authored no location.latitude_deg; PET must refuse (no default)")
        if p["compiled"].location.latitude_deg != p["latitude_deg"]:
            raise NativeCaseProfileError("the native compiled location no longer matches the sealed latitude")
        path, sha = self.case_yaml
        return {"case_yaml_path": path, "case_yaml_sha256": sha, "field": "location.latitude_deg",
                "value": p["latitude_deg"], "case_identity_sha256": p["case_identity_sha256"], "bundle_id": p["bundle_id"]}

    def require_latitude_for_pet(self) -> float:
        self._payload("admission")  # authenticate the record kind BEFORE any attribute lookup
        return NativeCaseAdmission.latitude_binding(self)["value"]

    def recheck(self) -> None:
        """Re-hash every authored file and re-verify the source profile; refuses any change since admission."""
        p = self._payload("admission")
        verify_source_profile(p["profile_target"], expected_syrup_digest=p["syrup_package_digest"], profile=p["profile_name"])
        for name, path, sha, size in p["files"]:
            try:
                same = _sha(_read_regular(Path(path), size + 1, name)) == sha
            except NativeCaseProfileError:
                same = False  # grown, replaced by a link/special file, or unreadable
            if not same:
                raise NativeCaseProfileError(f"{path} changed since admission")
        if p["compiled"].location.latitude_deg != p["latitude_deg"]:
            raise NativeCaseProfileError("the native compiled location differs from the sealed latitude")


def _native():
    from maple.case_tools.compilers import case_compiler
    from maple.config import resolve_config
    from maple.core.parameters.case_import import resolve_case_import
    from maple.core.parameters.case_initialization import resolve_threshold_modifiers
    from maple.core.parameters.location import resolve_location_section
    from maple.experiment import orchestrator, schema
    return types.SimpleNamespace(case_compiler=case_compiler, schema=schema, orchestrator=orchestrator,
                                 resolve_location=resolve_location_section, neutral_modifiers=resolve_threshold_modifiers,
                                 resolve_case_import=resolve_case_import, resolve_config=resolve_config)


def _latitude(nat, case_raw) -> float | None:
    """The genuine native validator on the PINNED document; its refusal becomes this profile's typed refusal."""
    try:
        return nat.resolve_location(case_raw).latitude_deg
    except ValueError as exc:
        raise NativeCaseProfileError(f"native location refused: {exc}") from exc


def _check_field_census(nat, route: str) -> None:
    if tuple(f.name for f in dataclasses.fields(nat.case_compiler.CompiledCase)) != COMPILED_CASE_FIELDS:
        raise NativeCaseProfileError("native CompiledCase fields differ from the pinned census; refused")
    if route == "package" and tuple(f.name for f in dataclasses.fields(nat.schema.ResolvedCasePackage)) != RESOLVED_PACKAGE_FIELDS:
        raise NativeCaseProfileError("native ResolvedCasePackage fields differ from the pinned census; refused")


def _check_config(config, label: str) -> None:
    if config.avalanching.enabled is not False or config.surface_creep.enabled is not False:
        raise NativeCaseProfileError(f"{label}: avalanching and surface creep must resolve disabled")
    g = config.geometry
    for v in (g.dx_m, g.dy_m):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise NativeCaseProfileError(f"{label}: cell size must be finite and positive, got {v!r}")
    if g.dx_m != g.dy_m:
        raise NativeCaseProfileError(f"{label}: non-square cells ({g.dx_m} x {g.dy_m}) are not supported")


def _check_owner(own, config, label: str) -> None:
    if own is None or own.owner != "topographic_wind" or own.roughness_m != config.topographic_wind.aerodynamic_roughness_m:
        raise NativeCaseProfileError(f"{label}: roughness ownership is not the ordinary topographic_wind owner")


def _resolve_package(nat, headers: _Headers, lat) -> str:
    """Genuine package resolve + full scope/location/ordinary-roughness check; returns the verified owner."""
    try:
        res = nat.schema.resolve_case_package(headers.case_dir)
    except Exception as exc:
        raise NativeCaseProfileError(f"native resolve refused: {type(exc).__name__}: {exc}") from exc
    if res.fixed_base.present or res.regional_slope.enabled or res.slope_yaml_path is not None or res.slope_yaml_sha256 is not None:
        raise NativeCaseProfileError("resolved case carries a fixed base / regional slope / slope.yaml")
    _check_owner(res.roughness_ownership, res.config, "resolved package")
    if res.vegetation.source != "none" or res.case_import.sediment is not None:
        raise NativeCaseProfileError("resolved case is not bare/neutral")
    _check_config(res.config, "resolved package")
    if res.location.latitude_deg != lat:
        raise NativeCaseProfileError("resolved location disagrees with the pinned authored case.yaml")
    return res.roughness_ownership.owner


def _resolve_single(nat, headers: _Headers, lat) -> str:
    """Single-file owner/geometry seam through the SAME public native functions compile_case/load use."""
    raw = copy.deepcopy(headers.case_raw)
    try:
        config = nat.resolve_config(raw)
    except Exception as exc:
        raise NativeCaseProfileError(f"native config resolve refused: {type(exc).__name__}: {exc}") from exc
    try:
        spec, config2, own = nat.case_compiler.resolve_regional_slope_for_case(
            raw, config, nat.resolve_case_import(raw.get("import")))
    except Exception as exc:
        raise NativeCaseProfileError(f"native regional/roughness resolve refused: {type(exc).__name__}: {exc}") from exc
    if spec.enabled:
        raise NativeCaseProfileError("regional slope is enabled in the single-file case")
    _check_owner(own, config2, "single-file")
    _check_config(config2, "single-file")
    return own.owner


def _check_compiled(nat, c, lat, label: str) -> None:
    if c.fixed_base_spec.present or c.fixed_base is not None or c.fixed_base_source is not None \
            or c.fixed_base_admission is not None or c.fixed_base_load_admission is not None:
        raise NativeCaseProfileError(f"{label}: a fixed base is present in the compiled case")
    if c.regional_slope.enabled or c.regional_slope_result is not None:
        raise NativeCaseProfileError(f"{label}: regional slope is present in the compiled case")
    if c.roughness_ownership is not None:  # None on a native LOAD (upstream limit); verified earlier from the resolver
        _check_owner(c.roughness_ownership, c.config, label)
    if c.initial_mobile_state is not None:
        raise NativeCaseProfileError(f"{label}: an initial mobile/airborne seed is present")
    if c.vegetation is None or c.threshold_modifier_spec != nat.neutral_modifiers({}):
        raise NativeCaseProfileError(f"{label}: non-neutral threshold modifiers")
    _check_config(c.config, label)
    if c.location.latitude_deg != lat:
        raise NativeCaseProfileError(f"{label}: compiled location disagrees with the pinned authored case.yaml")
    rec = c.provenance_record.get("location")
    if (rec is None) != (lat is None) or (rec is not None and rec.get("latitude_deg") != lat):
        raise NativeCaseProfileError(f"{label}: provenance location disagrees with the pinned authored case.yaml")


def _check_source_pins(c, headers: _Headers, route: str) -> None:
    src = c.provenance_record["source"]
    pinned = {name: sha for name, _p, sha, _s in headers.files}
    if src.get("case_yaml_sha256") != pinned["case.yaml"] or Path(src["case_yaml_path"]).resolve() != Path(headers.files[0][1]):
        raise NativeCaseProfileError("compiled provenance case.yaml path/hash differs from the pinned authored file")
    if route == "package":
        got = {e["authored_path"]: e["sha256"] for e in src.get("package_files", [])}
        if got != pinned:
            raise NativeCaseProfileError("compiled provenance package files differ from the pinned authored files")


def _early(nat, profile, headers, route) -> tuple[float | None, str]:
    """Everything that must hold BEFORE any native compile/load entry."""
    _check_field_census(nat, route)
    lat = _latitude(nat, headers.case_raw)
    owner = _resolve_package(nat, headers, lat) if route == "package" else _resolve_single(nat, headers, lat)
    return lat, owner


def _seal_admission(profile, headers, route, compiled, lat, owner) -> NativeCaseAdmission:
    pp = profile._payload("profile")
    return _make(NativeCaseAdmission, dict(
        profile_name=pp["profile"], bundle_id=pp["bundle_id"], package_digest=pp["package_digest"],
        syrup_package_digest=pp["syrup_package_digest"], route=route, case_dir=headers.case_dir, files=headers.files,
        case_identity_sha256=compiled.provenance_record["case_identity_sha256"], latitude_deg=lat, compiled=compiled,
        resolved_roughness_owner=owner, provenance_code_version=str(compiled.code_version), profile_target=pp["target"]))


def _finish(nat, profile, headers, route, compiled, loaded, lat, owner) -> NativeCaseAdmission:
    for label, c in (("compiled", compiled), ("loaded", loaded)):
        _check_compiled(nat, c, lat, label)
        _check_source_pins(c, headers, route)
    import numpy as np
    for name in ("voxel_column", "active_layer"):
        if not np.array_equal(getattr(compiled, name).mass_kg, getattr(loaded, name).mass_kg):
            raise NativeCaseProfileError(f"compiled and loaded {name} masses differ")
    if compiled.provenance_record["case_identity_sha256"] != loaded.provenance_record["case_identity_sha256"]:
        raise NativeCaseProfileError("compiled and loaded case identities differ")
    return _seal_admission(profile, headers, route, loaded, lat, owner)


def admit_package_case(profile: NativeSourceProfile, case_dir, *, limits: CaseLimits | None = None) -> NativeCaseAdmission:
    """Ordinary three-file package: bounded pin -> early gate -> native resolve -> native compile -> native load."""
    _require_profile(profile)
    headers = _gate(case_dir, "package", limits, existing_output=False)
    nat = _native()
    lat, owner = _early(nat, profile, headers, "package")
    try:
        _resolved, compiled = nat.orchestrator.compile_case_package(headers.case_dir)
        loaded = nat.case_compiler.load_compiled_case(headers.case_dir)
    except Exception as exc:
        raise NativeCaseProfileError(f"native compile/load refused: {type(exc).__name__}: {exc}") from exc
    return _finish(nat, profile, headers, "package", compiled, loaded, lat, owner)


def admit_single_file_case(profile: NativeSourceProfile, case_dir, *, limits: CaseLimits | None = None) -> NativeCaseAdmission:
    """Full single-file case: bounded pin -> early gate (owner/geometry via native public resolvers) -> compile -> load."""
    _require_profile(profile)
    headers = _gate(case_dir, "single_file", limits, existing_output=False)
    nat = _native()
    lat, owner = _early(nat, profile, headers, "single_file")
    try:
        compiled = nat.case_compiler.compile_case(headers.case_dir)
        loaded = nat.case_compiler.load_compiled_case(headers.case_dir)
    except Exception as exc:
        raise NativeCaseProfileError(f"native compile/load refused: {type(exc).__name__}: {exc}") from exc
    return _finish(nat, profile, headers, "single_file", compiled, loaded, lat, owner)


def load_admitted_case(profile: NativeSourceProfile, case_dir, *, route: str, limits: CaseLimits | None = None) -> NativeCaseAdmission:
    """Re-admit an ALREADY compiled case (native load only). Same gate and early checks as compile; never rewrites output."""
    route = _check_route(route)  # before any I/O, including the profile's cold re-verification
    _require_profile(profile)
    headers = _gate(case_dir, route, limits, existing_output=True)
    nat = _native()
    lat, owner = _early(nat, profile, headers, route)
    try:
        loaded = nat.case_compiler.load_compiled_case(headers.case_dir)
    except Exception as exc:
        raise NativeCaseProfileError(f"native load refused: {type(exc).__name__}: {exc}") from exc
    _check_compiled(nat, loaded, lat, "loaded")
    _check_source_pins(loaded, headers, route)
    return _seal_admission(profile, headers, route, loaded, lat, owner)
