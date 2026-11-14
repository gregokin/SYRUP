"""H0-VEG: the WATER-FREE held LAW context WITH a dynamic vegetation law (record `/3` CPU, `/4` CUDA; export, strict parse and reconstruction without the original bed, owner, plants or LEAF session).

WHAT IT IS. The same pure law context as `floating_held_context` (`/1` | `/2`: true native config inputs, ROOT floating column with its raw maps / layouts / `D0` / mask / area, the layer-0 conductivity record, facts) whose
matrix operator carries the per-cell vegetation factor `g` and whose preferential operator is the canonical LEAF derivation, BOTH re-prepared from ONE compact sealed `CachedVegetationInputs` recipe (the cache component:
the 13 LEAF compact arrays, the explicit site controls, the enabled receiver table and the recorded biological identity / package / helper pins). It holds NO water, NO J/I, NO plants / session / projection, NO
per-species `N` contribution, NO bed and NO original owner. The config keeps the BASE scalar `matrix_vegetation_factor` truthfully; the dynamic `g` identity is the matrix OPERATOR identity plus the recorded law
identity (`g_sha256`, `law_sha256`), distinct from the config identity. Old `/1` | `/2` records and functions are untouched and refuse the new schemas (and refuse a per-cell `g` vertical: see `floating_held_context`).

HELD vs CURRENT. The biological recipe may have been derived against a surface that PREDATES the held column's bed (a completed storm may have changed the bed first): its recorded surface hash is carried, never
compared with any current terrain; only the exact active mask and the cell / layer counts must agree with the held column (the cache stage and `prepare_vegetation_law` enforce them). An ordinary property / Ks / column
change keeps the SAME compact recipe and gives NEW operator identities: the identities in a record are the CURRENT ones of ITS column, recomputed (never applied) from the recipe.

EXPORT: `export_held_context_vegetation(vertical, cached, pins, *, memory_budget_bytes, ...)` needs a genuine `FloatingPondVertical` that ACTUALLY HOLDS the dynamic law (e.g. `from_boundary` of the root the
lifecycle's own `apply_vegetation_generation` produced, or `from_held_context` of a vegetation context) and a genuine sealed cached record; it re-prepares the law canonically from the cache on the vertical's own
column and ALWAYS requires the vertical's CURRENT matrix and preferential operator identities to equal that canonical law (there is no prospective / "apply" override: an export never records a law the vertical does
not hold); with `verify` a full dry reconstruction proves the record rebuilds the exact identities. CPU verticals give `/3`, CUDA verticals `/4`
(module `floating_held_context_vegetation_cuda`: current device FIRST, two pools, cold counted copies, authored without hardware).
RECORD: `VegetationHeldRecord.meta()` / `.arrays()` are the serialization parts (strict JSON-compatible metadata; sealed C-contiguous arrays; the cached recipe lives under `veg.*`); `record_from_parts_vegetation` is the strict
inverse: closed key sets, exact dtypes / shapes / counts and an analytic byte count are checked BEFORE any copy, the admission is composed (the arrays, the metadata copy measured by a streaming size pass), then the copies and
the mint. An integrity digest (streamed over the metadata and every array) is re-verified before every use. NOT file / head trust and not a source proof.
RECONSTRUCT: `floating_held_context.reconstruct_vegetation_held_context(record, expected, pins, ...)` (the owning hook) takes the caller's EXTERNAL expected identities and `ExpectedVegetationPins`; this module's `prepare_parts`
re-prepares the config and column through the public factories, stages the cached recipe cold through `stage_cached_inputs` (actual LEAF package / helper pins, H1 / H3 / H2 re-derivation, controls, record / binding SHAs),
builds the cached binding and calls `prepare_vegetation_law` with the CURRENT expected matrix / preferential operator identities; the hook re-checks the parts and mints the `HeldContext` (field `vegetation`: the one
cached record reference). Ordinary SYRUP profiles never import this module (and LEAF is imported lazily by the cache).

MEMORY (declared modelled host bytes, not a whole-process bound): `estimate_vegetation_held_bytes` composes the existing public estimates stage by stage (the record, the retained column / operators, the cache stage, the
law preparation; the max live set, caller reserve counted once). The streamed metadata size is part of the admission; the per-species strings / identity JSON of the CACHE are caller-owned metadata (dependency: the cache
component's own identity-JSON model).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from maple_syrup import floating_held_context as fh
from maple_syrup import floating_pond as fp
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs
from maple_syrup import subsurface as ss

SCHEMA3 = "maple_syrup.floating_held_context/3"
SCHEMA4 = "maple_syrup.floating_held_context/4"
VEG = "veg."
HeldContextError = fh.HeldContextError
DEFAULT_MAX_RECORD_BYTES = 1 << 30
# The typed native refusals this component converts into `HeldContextError`. `fsl.FloatingMemoryError` subclasses `ss.SubsurfaceMemoryError` (which is a `MemoryError`) and is NEITHER a `FloatingSoilError` nor a
# `SubsurfaceError`, so the typed memory refusal is listed explicitly; a BARE allocator `MemoryError` is deliberately not caught and propagates.
_ERRORS = (fsl.FloatingSoilError, ss.SubsurfaceError, ss.SubsurfaceMemoryError, nfs.NativeFloatingError, ValueError)

__all__ = [
    "SCHEMA3",
    "SCHEMA4",
    "ExpectedVegetationPins",
    "VegetationHeldRecord",
    "estimate_vegetation_held_bytes",
    "export_held_context_vegetation",
    "prepare_parts",
    "record_from_parts_vegetation",
    "verify_record",
]

_MINT = threading.local()
_RECORDS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_BASE_KEYS = frozenset(fh._META_KEYS - {"pref"}) | {"vegetation"}
_BASE_KEYS4 = _BASE_KEYS | {"kernel"}
_LAW_KEYS = frozenset({"g_sha256", "law_sha256"})


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


def _wh() -> Any:
    from maple_syrup import water_vegetation_held as wh

    return wh


def _wvb() -> Any:
    from maple_syrup import water_vegetation_binding as wvb

    return wvb


# ---------------------------------------------------------------------------------------------------------------------------------------------
# the external pins (the caller's independent expectations)
# ---------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ExpectedVegetationPins:
    """The caller's EXTERNAL expectations of the cached recipe (the arguments of `stage_cached_inputs`): the record SHA, the semantic binding SHA, the LEAF projection content SHA, the FULL actual LEAF package digest,
    the external helper pins and the complete expected source record. They are compared with the ACTUALLY IMPORTED LEAF package / helper bytes by the cache stage; nothing here is read from the record only."""

    record_sha256: str
    binding_sha256: str
    projection_sha256: str
    leaf_package_sha256: str
    external_helpers: Any
    source: dict

    @property
    def record_sha256_hex(self) -> str:
        return str(self.record_sha256).removeprefix("sha256:")

    @property
    def binding_sha256_hex(self) -> str:
        return str(self.binding_sha256).removeprefix("sha256:")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# metadata / array layout (closed)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _veg_sizes(vm: Any) -> dict:
    """Strict, bounded validation of the `vegetation` block and the counts the array layout needs. The cached metadata goes through the CACHE's own bounded pre-quota gate (`_gate_meta`: closed keys, strict types,
    per-identifier length, the walked JSON size of the O(1) documents with a fan-out / depth bound): nothing is serialised, copied or sorted here, and a hostile document is refused in time linear in what it carries."""
    wh = _wh()
    if type(vm) is not dict or set(vm) != {"cached", "law"}:
        raise HeldContextError("vegetation metadata keys differ from the schema")
    law = vm["law"]
    if type(law) is not dict or set(law) != _LAW_KEYS or not all(fh._is_hex64(law[k]) for k in _LAW_KEYS):
        raise HeldContextError("vegetation.law must carry the 64-hex g_sha256 / law_sha256")
    cm = vm["cached"]
    try:
        g = wh._gate_meta(cm)
    except _wvb().VegetationBindingError as exc:
        raise HeldContextError(f"vegetation.cached metadata refused: {exc}") from None
    if not fh._is_hex64(cm["record_sha256"]) or not fh._is_hex64(cm["binding_sha256"]):
        raise HeldContextError("cached.record_sha256 / binding_sha256 must be 64-hex digests")
    return {"N": g["N"], "M": g["M"], "S": g["S"], "L": g["L"], "K": g["K"], "n_en": g["n_receivers"], "ctl": g["ctl"], "shape": g["shape"], "max_id": g["max_id"], "meta_bytes": g["meta_bytes"]}


def _veg_arrays(sz: dict) -> dict:
    """The exact `veg.*` array names, shapes and dtypes (the cache's own closed layout: its dtype table and analytic shapes)."""
    wh = _wh()
    N, M, S, L, K, ctl = sz["N"], sz["M"], sz["S"], sz["L"], sz["K"], sz["ctl"]
    out = {VEG + k: (tuple(shp), np.dtype(wh._DTYPES[k])) for k, shp in wh._shapes(N, M, S, L, K).items()}
    if ctl["a3_layout"] == "cell":
        out[VEG + "controls.a3"] = ((N,), np.dtype(np.float64))
    if ctl["max_rate_layout"] == "cell":
        out[VEG + "controls.max_intake_rate_m_s"] = ((N,), np.dtype(np.float64))
    if ctl["has_background"]:
        out[VEG + "controls.background_receiver_weights"] = ((L,), np.dtype(np.float64))
    out[VEG + "receiver_weights_enabled"] = ((sz["n_en"], L), np.dtype(np.float64))
    return out


def _base_view(meta: Any) -> dict:
    """The `/1`-shaped view of a vegetation record's BASE metadata (no preferential block): validated by the OLD strict checker so the base fields are not re-implemented."""
    if type(meta) is not dict:
        raise HeldContextError("held-context metadata must be a dict")
    v4 = meta.get("schema") == SCHEMA4
    if meta.get("schema") not in (SCHEMA3, SCHEMA4) or set(meta) != (_BASE_KEYS4 if v4 else _BASE_KEYS):
        raise HeldContextError("vegetation held-context metadata keys / schema differ from the closed /3 | /4 schema")
    view = {k: v for k, v in meta.items() if k != "vegetation"}
    view["schema"] = fh.SCHEMA2 if v4 else fh.SCHEMA
    view["pref"] = None
    ident = meta["identities"]
    if type(ident) is not dict or set(ident) != fh._IDENT_KEYS or not fh._is_hex64(ident.get("pref_sha256")):
        raise HeldContextError("a dynamic-vegetation record requires the closed identities with a 64-hex preferential identity")
    view["identities"] = {**ident, "pref_sha256": None}
    return view


def _check_meta(meta: Any, n_arrays: int) -> tuple[dict, dict]:
    view = _base_view(meta)
    sz = _veg_sizes(meta["vegetation"])
    fh._check_meta(view, n_arrays)
    return view, sz


def _expected_all(view: dict, sz: dict) -> dict:
    return {**fh._expected_arrays(view), **_veg_arrays(sz)}


def _analytic(view: dict, sz: dict) -> tuple[int, int]:
    count, nbytes = fh._analytic(view)
    va = _veg_arrays(sz)
    return count + len(va), nbytes + sum(int(np.prod(shape, dtype=np.int64)) * dt.itemsize for shape, dt in va.values())


def _check_layout(view: dict, sz: dict, arrays: Any) -> None:
    want = _expected_all(view, sz)
    if not hasattr(arrays, "keys") or set(arrays.keys()) != set(want):
        raise HeldContextError("array names differ from the schema (extra or missing arrays)")
    for name, (shape, dtype) in want.items():
        a = arrays[name]
        if type(a) is not np.ndarray or a.dtype != dtype or tuple(a.shape) != tuple(shape) or not a.flags.c_contiguous:
            raise HeldContextError(f"array {name} is not an exact C-contiguous {dtype} ndarray of shape {tuple(shape)}")


def _iter_meta(meta: dict):
    for chunk in json.JSONEncoder(sort_keys=True).iterencode(meta):
        yield chunk.encode()


def _meta_bytes(meta: dict, sz: dict) -> int:
    """UPPER BOUND of the canonical JSON size of the whole metadata, WITHOUT serialising the cached part: the streamed size of the small closed BASE metadata (everything but `vegetation.cached`), plus the cache's
    own walked bound of its O(1) documents (`meta_bytes`), plus the identifiers (species and receiver ids: each at most `6 * max_id + 4` bytes when escaped) and the small fixed cached fields."""
    base = {k: v for k, v in meta.items() if k != "vegetation"}
    n = sum(len(c) for c in _iter_meta(base)) + sz["meta_bytes"] + (sz["S"] + sz["n_en"]) * (6 * sz["max_id"] + 4)
    return int(n + 4096)


def _digest(meta: dict, arrays: Any) -> str:
    h = hashlib.sha256()
    for c in _iter_meta(meta):
        h.update(c)
    for name in sorted(arrays.keys()):
        a = arrays[name]
        h.update(f"|{name}|{a.dtype}|{a.shape}".encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------------------------------
# memory (composed, literal)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def estimate_vegetation_held_bytes(meta: dict, *, backend: str | None = None) -> dict:
    """Declared host bytes of the vegetation held context, stage by stage, from a VALIDATED metadata dict by arithmetic over the PUBLIC estimators (the cache stage, the law preparation, LEAF's stream model, the old
    column / operator estimates). `record_bytes` = the record arrays; `reconstruct_bytes` = the MAX live set of the stages (base column, cache stage, law preparation) with the record and the retained column
    live throughout; `export_bytes` = sealed copies + the verification reconstruction. The caller's reserve is counted ONCE by the quota (not inside these numbers)."""
    wh, wvb = _wh(), _wvb()
    view, sz = _check_meta(meta, 1 << 30)
    be = backend or ("cuda" if meta["schema"] == SCHEMA4 else meta["backend"])
    _count, record = _analytic(view, sz)
    mb = _meta_bytes(meta, sz)
    L, N, K = meta["n_layers"], meta["n_cells"], meta["n_classes"]
    base = fh.estimate_held_context_memory(L, N, K, 0, len(meta["overrides"] or ()), backend="numpy" if be == "cuda" else be, layouts=meta["layouts"]["all"], has_pref=False, texture_ks=meta["mode"] == "texture_ks")
    cm = meta["vegetation"]["cached"]
    stage = wh.estimate_cached_stage_bytes_from_meta(cm, backend="numpy", source_cold=True, with_mask=True)               # the cache's FINAL public model (max id / metadata bytes from its bounded gate)
    lw = wvb._load_leaf()
    held = lw.estimate_stream_bytes(sz["N"], sz["M"], sz["n_en"], sz["L"], previous_held=True)["per_next_bytes"]
    free = lw.estimate_stream_bytes(sz["N"], sz["M"], sz["n_en"], sz["L"], previous_held=False)["per_next_bytes"]
    wstage = lw.estimate_stream_bytes(sz["N"], sz["M"], sz["n_en"], sz["L"])["weights_stage_bytes"]
    law = wvb.estimate_law_bytes(L, N, sz["n_en"], binding_resident_bytes=stage["resident_bytes"] + wvb.PY_RECORD_ALLOWANCE_BYTES, stream_per_next_held_bytes=held, stream_per_next_free_bytes=free,
                                 stream_weights_stage_bytes=wstage, backend="numpy", k_suction_cell_arrays=int(meta["mode"] == "texture_ks"), a3_cell_map=sz["ctl"]["a3_layout"] == "cell",
                                 max_rate_cell_map=sz["ctl"]["max_rate_layout"] == "cell")
    live = record + int(base.retained_bytes)
    stages = {"base": int(base.reconstruct_bytes) + record, "cache": live + int(stage["peak_bytes"]), "law": live + int(law["peak_bytes"])}
    reconstruct = max(stages.values()) + 2 * mb
    return {"record_bytes": int(record), "meta_bytes": int(mb), "stages": {k: int(v) for k, v in stages.items()}, "reconstruct_bytes": int(reconstruct), "live_outside_bytes": int(live),
            "export_bytes": int(record + 2 * mb + reconstruct), "cache_resident_bytes": int(stage["resident_bytes"]), "scope": "declared host arithmetic over public estimators; caller reserve counted once by the quota"}


def _admit(max_bytes: Any, reader: Callable[[], int] | None, reserve: int, need: int, what: str) -> None:
    mb = fsl._count(max_bytes, "memory_budget_bytes")
    if reserve + need > mb:
        raise HeldContextError(f"{what} needs {reserve + need} modelled bytes (reserve {reserve} + {need}) above memory_budget_bytes={mb}; nothing was allocated")
    fh._admit(reader, need + reserve, what)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class VegetationHeldRecord:
    """Opaque owned record of the dynamic-vegetation held law context: minted ONLY by `export_held_context_vegetation` / `record_from_parts_vegetation`. Old `/1` | `/2` records are a different class."""

    _meta: Any
    _arrays: Any

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise HeldContextError("a VegetationHeldRecord is minted only by export_held_context_vegetation / record_from_parts_vegetation (no direct construction or replace())")

    def __copy__(self):
        raise HeldContextError("a VegetationHeldRecord is an owned record and cannot be copied")

    def __deepcopy__(self, memo):
        raise HeldContextError("a VegetationHeldRecord is an owned record and cannot be copied")

    def __reduce_ex__(self, protocol):
        raise HeldContextError("a VegetationHeldRecord is not pickled; serialise `meta()` and `arrays()` explicitly")

    def meta(self) -> dict:
        verify_record(self)
        return json.loads(json.dumps(self._meta))

    def arrays(self) -> Any:
        verify_record(self)
        return self._arrays


def record_schema(record: Any) -> str:
    """The MINIMAL peek the public entry needs to select a backend BEFORE anything is read, hashed or admitted: the exact record class and its schema string (no digest, no layout check, no array touch)."""
    if type(record) is not VegetationHeldRecord or record not in _RECORDS or type(record._meta) is not dict:
        raise HeldContextError("not a genuinely minted VegetationHeldRecord (an old /1 | /2 HeldContextRecord is a different record)")
    schema = record._meta.get("schema")
    if schema not in (SCHEMA3, SCHEMA4):
        raise HeldContextError("the record's schema is not a dynamic-vegetation schema")
    return schema


def verify_record(record: Any) -> dict:
    """Public genuine-record proof: exact class, private registry, closed metadata and array layout, and the streamed integrity digest (a replaced field, mapping entry or array byte is refused). Returns the internal metadata."""
    if type(record) is not VegetationHeldRecord or record not in _RECORDS:
        raise HeldContextError("not a genuinely minted VegetationHeldRecord (an old /1 | /2 HeldContextRecord is a different record)")
    meta, arrays = record._meta, record._arrays
    if not isinstance(arrays, MappingProxyType) or type(meta) is not dict:
        raise HeldContextError("the record's containers were replaced (integrity)")
    view, sz = _check_meta(meta, len(arrays))
    _check_layout(view, sz, arrays)
    if _digest(meta, arrays) != _RECORDS[record]:
        raise HeldContextError("the record's contents no longer match the integrity digest bound at minting (altered)")
    return meta


def _mint(meta: dict, arrays: dict) -> VegetationHeldRecord:
    ro = MappingProxyType(dict(arrays))
    digest = _digest(meta, ro)
    with _minting():
        rec = VegetationHeldRecord(meta, ro)
    _RECORDS[rec] = digest
    return rec


def record_from_parts_vegetation(meta: Any, arrays: Any, *, max_bytes: int = DEFAULT_MAX_RECORD_BYTES, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0) -> VegetationHeldRecord:
    """STRICT parse of externally stored parts into a minted record. ORDER: reserve / `max_bytes` -> the closed metadata checks (counts read as constants; nothing iterated) -> the analytic array count and byte cap
    BEFORE any name, shape or copy -> exact array metadata -> the streamed metadata size -> the real admission (arrays + the metadata copy) -> the sealed copies -> the metadata copy -> the mint."""
    try:
        reserve = fsl._count(reserve_bytes, "reserve_bytes")
    except fsl.FloatingSoilError as exc:
        raise HeldContextError(str(exc)) from None
    if isinstance(max_bytes, (bool, np.bool_)) or not isinstance(max_bytes, (int, np.integer)) or int(max_bytes) < 1:
        raise HeldContextError("max_bytes must be a positive integer")
    if not hasattr(arrays, "keys") or not hasattr(arrays, "__getitem__") or not hasattr(arrays, "__len__"):
        raise HeldContextError("arrays must be a mapping of names to arrays")
    view, sz = _check_meta(meta, len(arrays))
    count, nbytes = _analytic(view, sz)
    if count != len(arrays):
        raise HeldContextError(f"the declared inputs require {count} arrays but {len(arrays)} were supplied")
    if nbytes > int(max_bytes):
        raise HeldContextError(f"the declared inputs require {nbytes} array bytes, above max_bytes ({int(max_bytes)} B)")
    _check_layout(view, sz, arrays)
    total = sum(int(arrays[n].nbytes) for n in arrays)
    if total != nbytes:
        raise HeldContextError(f"the arrays hold {total} B but the declared inputs imply {nbytes} B")
    mb = _meta_bytes(meta, sz)
    fh._admit(available_bytes_fn, total + 2 * mb + fh.ALLOWANCE_BYTES + reserve, "parsing the vegetation held-context record (sealed copies and the metadata copy)")
    sealed = {n: fh._seal(arrays[n]) for n in arrays}
    return _mint(json.loads(json.dumps(meta)), sealed)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _check_pins(pins: Any) -> ExpectedVegetationPins:
    if type(pins) is not ExpectedVegetationPins:
        raise HeldContextError("pins must be ExpectedVegetationPins")
    return pins


def _cached_check(cached: Any, pins: ExpectedVegetationPins) -> Any:
    """The genuine sealed cached record (cache `verify_cached_inputs`: metadata only) whose record / binding SHA equal the caller's pins. No copy is made here."""
    wh, wvb = _wh(), _wvb()
    try:
        rec = wh.verify_cached_inputs(cached)
    except wvb.VegetationBindingError as exc:
        raise HeldContextError(f"cached must be a genuine sealed CachedVegetationInputs: {exc}") from None
    if rec.record_sha256 != pins.record_sha256_hex or rec.binding_sha256 != pins.binding_sha256_hex:
        raise HeldContextError("the cached record / binding SHA differ from the caller's pins")
    return rec


def _cached_export_need(rec: Any) -> int:
    """Literal bytes of the cache's `export_cached_record` detached copies (the identity / source / package / helper documents parsed once more, the species and receiver id lists): charged BEFORE the export is made."""
    docs = len(rec.source_json) + len(rec.package_json) + len(rec.helpers_json) + len(rec.identity_json)
    return int(3 * docs + 8 * (len(rec.species_ids) + len(rec.receiver_ids)) + _wvb().PY_RECORD_ALLOWANCE_BYTES)


def _cached_export(rec: Any) -> tuple[dict, dict]:
    ex = _wh().export_cached_record(rec)
    return ex["meta"], dict(ex["arrays"])


def _law_conductivity(t: Any, cfg: Any) -> Any:
    return t.ks_m_s if cfg.matrix_conductivity_mode == "texture_ks" else cfg.matrix_conductivity_m_s


def export_held_context_vegetation(vertical: Any, cached: Any, pins: Any, *, memory_budget_bytes: Any, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0, verify: bool = True,
                                   host_memory_budget_bytes: Any = None, host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0) -> VegetationHeldRecord:
    """Capture the held law context + the dynamic vegetation law of a genuine `FloatingPondVertical` that ACTUALLY HOLDS the law (module docstring). The vertical's CURRENT matrix and preferential operator identities are
    ALWAYS compared with the canonical law re-prepared from `cached` on its own column (no prospective override). Order: the CPU / CUDA backend is read from the vertical; a CUDA vertical is dispatched FIRST to `/4`
    (`floating_held_context_vegetation_cuda`: the current device gate precedes everything); CPU: strict argument checks -> metadata-only cached verification -> the literal admission of the cache export copies -> the
    export -> the composed admission of the whole export -> the canonical law -> the sealed copies -> the mint -> (verify) a dry reconstruction."""
    if type(vertical) is not fp.FloatingPondVertical:
        raise HeldContextError("vertical must be a genuine FloatingPondVertical")
    pins = _check_pins(pins)
    if vertical.backend == "cuda":
        from maple_syrup import floating_held_context_vegetation_cuda as fvc

        return fvc.export_held_context_vegetation_cuda(vertical, cached, pins, memory_budget_bytes=memory_budget_bytes, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes,
                                                       verify=verify, host_memory_budget_bytes=host_memory_budget_bytes, host_available_bytes_fn=host_available_bytes_fn, host_reserve_bytes=host_reserve_bytes)
    if vertical.backend not in ("numpy", "numba"):
        raise HeldContextError(f"backend {vertical.backend!r} is not supported by the CPU held context")
    if host_memory_budget_bytes is not None or host_available_bytes_fn is not None or host_reserve_bytes:
        raise HeldContextError("host_* arguments apply only to a CUDA vertical")
    try:
        reserve = fsl._count(reserve_bytes, "reserve_bytes")
    except fsl.FloatingSoilError as exc:
        raise HeldContextError(str(exc)) from None
    wvb = _wvb()
    t = vertical.held_template()
    cfg = vertical.config
    if type(cfg) is not nfs.NativeFloatingConfig or cfg.sha256 != t.config_sha256:
        raise HeldContextError("the vertical's native config does not match its held template identity")
    if t.preferential is None:
        raise HeldContextError("the vertical holds no preferential operator: it cannot hold the dynamic vegetation law")
    rec = _cached_check(cached, pins)
    _admit(memory_budget_bytes, available_bytes_fn, reserve, _cached_export_need(rec), "exporting the cached recipe metadata (detached copies)")
    cmeta, carrays = _cached_export(rec)
    meta, refs = _snapshot_base(vertical)
    meta = {k: v for k, v in meta.items() if k != "pref"}
    meta["schema"] = SCHEMA3
    meta["vegetation"] = {"cached": cmeta, "law": {"g_sha256": "0" * 64, "law_sha256": "0" * 64}}
    meta["identities"] = {**meta["identities"], "pref_sha256": "0" * 64}
    _check_meta(meta, 1 << 30)  # strict bounded self-check of the layout before any allocation (identities are placeholders)
    est = estimate_vegetation_held_bytes(meta)
    _admit(memory_budget_bytes, available_bytes_fn, reserve, est["export_bytes"] if verify else est["record_bytes"] + 2 * est["meta_bytes"] + fh.ALLOWANCE_BYTES, "exporting the vegetation held law context")
    try:
        binding = _wh().cached_binding(cached, expected_binding_sha256=pins.binding_sha256, max_bytes=memory_budget_bytes, reserve_bytes=reserve + est["live_outside_bytes"], available_bytes_fn=available_bytes_fn)
        law = wvb.prepare_vegetation_law(t.column, binding, conductivity_m_s=_law_conductivity(t, cfg), suction_m=cfg.matrix_suction_m, memory_budget_bytes=memory_budget_bytes,
                                         reserve_bytes=reserve + est["live_outside_bytes"], available_bytes_fn=available_bytes_fn,
                                         expected_matrix_op_sha256=t.matrix_op.sha256, expected_preferential_op_sha256=t.preferential.sha256)
    except (wvb.VegetationBindingError, *_ERRORS) as exc:
        raise HeldContextError(f"the vertical's CURRENT operators are not the canonical law of the cached recipe on its column (identity mismatch or refusal): {exc}") from None
    meta["identities"] = {**meta["identities"], "matrix_op_sha256": law.matrix_sha256, "pref_sha256": law.preferential_sha256}
    meta["vegetation"]["law"] = {"g_sha256": law.g_sha256, "law_sha256": law.law_sha256}
    arrays = {k: fh._seal(v) for k, v in refs.items()}
    arrays.update({VEG + k: fh._seal(v) for k, v in carrays.items()})
    record = _mint(json.loads(json.dumps(meta)), arrays)
    if verify:
        ident = meta["identities"]
        from maple_syrup import floating_held_context as fhm

        fhm.reconstruct_vegetation_held_context(record, fhm.ExpectedHeldIdentities(ident["config_sha256"], ident["root_column_sha256"], ident["matrix_op_sha256"], ident["pref_sha256"]), pins,
                                                memory_budget_bytes=memory_budget_bytes, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
    return record


def _snapshot_base(vertical: Any) -> tuple[dict, dict]:
    """Metadata and REFERENCES of the BASE held law (config / column / D0 / mask / Ks / overrides / facts) of a genuine vertical: the `/1` snapshot fields without the preferential block (a copy of the
    published `_snapshot` arithmetic, kept here so the old module stays untouched). The identities carry the template's matrix identity as a placeholder until the law is re-prepared."""
    t = vertical.held_template()
    cfg = vertical.config
    col = t.column
    L, N, K = int(col.n_layers), int(col.n_cells), int(cfg.grain_texture.shape[0])
    lay = {k: str(v) for k, v in dict(col.layouts).items()}
    hexf = fh._hex
    meta = {"schema": SCHEMA3, "backend": cfg.backend, "n_layers": L, "n_cells": N, "n_classes": K, "mode": cfg.matrix_conductivity_mode, "bottom_drainage": cfg.bottom_drainage,
            "scalars": {"bulk_density_kg_m3": hexf(cfg.bulk_density_kg_m3), "theta_residual": hexf(cfg.theta_residual), "theta_evaporation_stress": hexf(cfg.theta_evaporation_stress),
                        "evaporation_factor": hexf(cfg.evaporation_factor), "matrix_conductivity_m_s": None if cfg.matrix_conductivity_m_s is None else hexf(cfg.matrix_conductivity_m_s),
                        "matrix_suction_m": hexf(cfg.matrix_suction_m), "matrix_vegetation_factor": hexf(cfg.matrix_vegetation_factor), "initial_wetting_m": hexf(cfg.initial_wetting_m),
                        "initial_cumulative_intake_m": hexf(cfg.initial_cumulative_intake_m),
                        "latent_saturated_conductivity_m_s": None if cfg.latent_saturated_conductivity_m_s is None else hexf(cfg.latent_saturated_conductivity_m_s), "cell_area_m2": hexf(col.cell_area_m2)},
            "overrides": None if cfg.overrides is None else sorted(cfg.overrides), "layouts": {"theta_field_capacity": lay["theta_field_capacity"], "theta_saturation": lay["theta_saturation"], "all": lay},
            "identities": {"config_sha256": cfg.sha256, "root_column_sha256": col.sha256, "matrix_op_sha256": t.matrix_op.sha256, "pref_sha256": None},
            "facts": {"generation": int(t.generation), "source_witness": t.source_witness}, "pref": None}
    arrays = {"thickness_m": cfg.layers.thickness_m, "grain_texture": cfg.grain_texture, "organic_matter_percent": cfg.organic_matter_percent, "latent_theta_field_capacity": cfg.latent_theta_field_capacity,
              "latent_theta_saturation": cfg.latent_theta_saturation, "diffusivity_m2_s": cfg.diffusivity_m2_s, "theta_field_capacity": col._maps["theta_field_capacity"],
              "theta_saturation": col._maps["theta_saturation"], "depth0_m": t.depth0_m, "active": col._maps["active"]}
    if meta["mode"] == "texture_ks":
        if t.ks_m_s is None:
            raise HeldContextError("a texture_ks owner must carry its layer-0 conductivity record")
        arrays["ks_m_s"] = t.ks_m_s
    for k in meta["overrides"] or ():
        arrays[fh._OVERRIDE_PREFIX + k] = cfg.overrides[k]
    return meta, arrays


# ---------------------------------------------------------------------------------------------------------------------------------------------
# reconstruction (parts for the owning hook)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _split(arrays: Any) -> tuple[dict, dict]:
    base = {k: arrays[k] for k in arrays if not k.startswith(VEG)}
    veg = {k[len(VEG):]: arrays[k] for k in arrays if k.startswith(VEG)}
    return base, veg


def prepare_parts(record: Any, pins: Any, *, memory_budget_bytes: Any, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0, host_memory_budget_bytes: Any = None,
                  host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0) -> dict:
    """Re-prepare the objects of an intact record (called ONLY by `floating_held_context.reconstruct_vegetation_held_context`, which re-checks the plain parts and mints). The CPU path here; `/4` dispatches to
    `floating_held_context_vegetation_cuda.prepare_parts_cuda` (current device first, two pools). Mutates nothing on failure. The hook has ALREADY verified the record (class, registry, closed layout, digest): this
    function only peeks the schema and reads the record, it does not hash the arrays a second time; its plain parts are re-checked by the hook before anything is minted."""
    meta = record._meta if record_schema(record) else None
    pins = _check_pins(pins)
    if meta["schema"] == SCHEMA4:
        from maple_syrup import floating_held_context_vegetation_cuda as fvc

        return fvc.prepare_parts_cuda(record, pins, memory_budget_bytes=memory_budget_bytes, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes,
                                      host_memory_budget_bytes=host_memory_budget_bytes, host_available_bytes_fn=host_available_bytes_fn, host_reserve_bytes=host_reserve_bytes)
    wh, wvb = _wh(), _wvb()
    try:
        reserve = fsl._count(reserve_bytes, "reserve_bytes")
    except fsl.FloatingSoilError as exc:
        raise HeldContextError(str(exc)) from None
    est = estimate_vegetation_held_bytes(meta)
    _admit(memory_budget_bytes, available_bytes_fn, reserve, est["reconstruct_bytes"], "reconstructing the vegetation held law context")
    arr = record._arrays
    base_arr, veg_arr = _split(arr)
    ident = meta["identities"]
    sc = {k: (None if v is None else fh._unhex(v, k)) for k, v in meta["scalars"].items()}
    overrides = None if meta["overrides"] is None else {k: base_arr[fh._OVERRIDE_PREFIX + k] for k in meta["overrides"]}
    L = meta["n_layers"]
    cm = meta["vegetation"]["cached"]
    shape = tuple(cm["compact"]["shape"])
    live = est["live_outside_bytes"]
    try:
        cfg = nfs.native_floating_config(layer_thickness_m=base_arr["thickness_m"], grain_texture=base_arr["grain_texture"], organic_matter_percent=base_arr["organic_matter_percent"],
                                         bulk_density_kg_m3=sc["bulk_density_kg_m3"], latent_theta_field_capacity=base_arr["latent_theta_field_capacity"],
                                         latent_theta_saturation=base_arr["latent_theta_saturation"], theta_residual=sc["theta_residual"], theta_evaporation_stress=sc["theta_evaporation_stress"],
                                         evaporation_factor=sc["evaporation_factor"], diffusivity_m2_s=base_arr["diffusivity_m2_s"], bottom_drainage=meta["bottom_drainage"],
                                         matrix_conductivity_m_s=sc["matrix_conductivity_m_s"], matrix_suction_m=sc["matrix_suction_m"], matrix_vegetation_factor=sc["matrix_vegetation_factor"],
                                         initial_wetting_m=sc["initial_wetting_m"], initial_cumulative_intake_m=sc["initial_cumulative_intake_m"], overrides=overrides, backend=meta["backend"],
                                         matrix_conductivity_mode=meta["mode"], latent_saturated_conductivity_m_s=sc["latent_saturated_conductivity_m_s"])
        if cfg.sha256 != ident["config_sha256"]:
            raise HeldContextError("the reconstructed native config does not reproduce the recorded config sha256")

        def raw(name: str, kind: str) -> Any:
            a = base_arr[name]
            return float(a[0]) if kind == "scalar" else a

        fc, sat = raw("theta_field_capacity", meta["layouts"]["theta_field_capacity"]), raw("theta_saturation", meta["layouts"]["theta_saturation"])
        col = fsl.prepare_floating_column(cfg.nominal_edges_m, theta_residual=cfg.theta_residual, theta_field_capacity=fc, theta_saturation=sat, diffusivity_m2_s=cfg.diffusivity_m2_s if L > 1 else 0.0,
                                          theta_evaporation_stress=cfg.theta_evaporation_stress, evaporation_factor=cfg.evaporation_factor, diffusion_mode="constant", bottom_drainage=cfg.bottom_drainage,
                                          soil_depth_m=base_arr["depth0_m"], active_mask=base_arr["active"], cell_area_m2=sc["cell_area_m2"], backend=cfg.backend, available_bytes_fn=available_bytes_fn)
        if col.sha256 != ident["root_column_sha256"]:
            raise HeldContextError("the re-prepared root column does not reproduce the recorded root column sha256")
        ks = base_arr["ks_m_s"] if meta["mode"] == "texture_ks" else None
        mask2d = base_arr["active"].reshape(shape)
        cmeta = json.loads(json.dumps(cm))
        cached = wh.stage_cached_inputs(cmeta, dict(veg_arr), expected_record_sha256=pins.record_sha256, expected_binding_sha256=pins.binding_sha256, expected_projection_sha256=pins.projection_sha256,
                                        expected_leaf_package_sha256=pins.leaf_package_sha256, expected_external_helpers=pins.external_helpers, expected_source=pins.source, active_mask=mask2d,
                                        backend="numpy", max_bytes=memory_budget_bytes, reserve_bytes=reserve + live, available_bytes_fn=available_bytes_fn)
        binding = wh.cached_binding(cached, expected_binding_sha256=pins.binding_sha256, max_bytes=memory_budget_bytes, reserve_bytes=reserve + live, available_bytes_fn=available_bytes_fn)
        law = wvb.prepare_vegetation_law(col, binding, conductivity_m_s=ks if ks is not None else cfg.matrix_conductivity_m_s, suction_m=cfg.matrix_suction_m, memory_budget_bytes=memory_budget_bytes,
                                         reserve_bytes=reserve + live, available_bytes_fn=available_bytes_fn, expected_matrix_op_sha256=ident["matrix_op_sha256"],
                                         expected_preferential_op_sha256=ident["pref_sha256"])
    except HeldContextError:
        raise
    except (wvb.VegetationBindingError, *_ERRORS) as exc:
        raise HeldContextError(f"a public factory or the cached recipe refused the record: {exc}") from None
    vm = meta["vegetation"]
    if law.law_sha256 != vm["law"]["law_sha256"] or law.g_sha256 != vm["law"]["g_sha256"]:
        raise HeldContextError("the re-prepared vegetation law does not reproduce the recorded law identities")
    return {"config": cfg, "column": col, "matrix_op": law.matrix_op, "preferential": law.preferential_op, "depth0": fh._seal(base_arr["depth0_m"]), "ks": None if ks is None else fh._seal(ks),
            "active": fh._seal(base_arr["active"]), "area": sc["cell_area_m2"], "edges": None, "host_depth0": None, "host_active": None, "host_ks": None, "cached": cached, "law": law}
