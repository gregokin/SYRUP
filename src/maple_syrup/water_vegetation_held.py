"""Compact CACHED vegetation inputs (Phase 2.5, slice cached-inputs): the water-free, DATA-ONLY record a later strict held / restart profile will carry.

ONE immutable SYRUP-owned record of the LEAF projection's compact export (O(N + M + S L): no S x N, no P x N, no plants, no session, no history), the explicit S0 site controls and the complete
recorded biological identity / full LEAF package / external helper pins / semantic binding SHA. Three ways in, one shape out:

    rec = capture_cached_inputs(binding, max_bytes=)                      # from a genuine S0 live binding (borrows its immutable bytes, or copies under admission)
    rec = stage_cached_inputs(meta, arrays, expected_*..., active_mask=)  # COLD: from a closed metadata / array record, every check before the record exists
    bnd = cached_binding(rec, expected_binding_sha256=)                   # a labelled `cached_derived` binding that feeds the unchanged `prepare_vegetation_law`

What it is NOT. It is a recipe and integrity-checked DATA, not a trusted file, not an accepted parent capability and not proof of a live `PlantSession` or of any current vegetation generation:
`origin` is `cached_derived`, `live_session_proof` is False, and the record SHA only protects integrity against whoever supplies the matching hash. A future outer file codec verifies the caller's
external seal, headers and sources and then owns the head LAST. The cache is the INPUT recipe only: an ordinary property / Ks / column refresh changes the native operator identities while the
recipe stays the same, so the CURRENT operator ids come from the caller's held root and are checked by `prepare_vegetation_law(expected_*_op_sha256=)`; stored operator arrays are never injected.

Reused, not reimplemented: LEAF's public H1 `compact_content_sha256`, H2 `preferential_inputs_from_compact` / `estimate_stream_bytes` and H3 `rederive_fields_from_compact`; S0's strict source
validation (`_verify_source_pins`), semantic `_binding_sha`, `site_controls` and the native preparers. LEAF is imported lazily inside the functions only.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
import threading
import types
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import water_vegetation_binding as wvb
from maple_syrup.water_vegetation_binding import (
    VegetationBindingError,
)

__all__ = [
    "CACHED_VERSION",
    "CachedVegetationInputs",
    "cached_binding",
    "capture_cached_inputs",
    "estimate_cached_stage_bytes",
    "estimate_cached_stage_bytes_from_meta",
    "estimate_capture_bytes",
    "estimate_capture_bytes_from_binding",
    "export_cached_record",
    "stage_cached_inputs",
    "verify_cached_inputs",
]

CACHED_VERSION = "2"  # a STRING (strictly compared); v2: identity stored without the duplicated species ids, streamed hashing, exact cell-area binding v2. v1 records are historical, never rebound.
CACHED_TAG = "maple_syrup.water_vegetation.cached_inputs.v2"
MAX_ID_CHARS = 64  # the genuine LEAF catalog contract: [A-Za-z0-9_.-]{1,64}; a longer identifier in a record is hostile and refused BEFORE any owned work
META_JSON_MAX_BYTES = 1 << 20  # hostile-input bound on the O(1) metadata documents (source / package / helpers / identity); a genuine record needs a few KB
MAX_HELPERS = 64
HELD_ID_FIELDS = ("species_ids", "receiver_ids")
_F8 = 8
_MINT = threading.local()
COMPARE_CHUNK = 1 << 16

# The documented dtype contract of the LEAF compact export (H1 re-validates it strictly; any drift fails closed there). Shapes are analytic in (N, M, S, L, K).
_DTYPES = {"splash_cover": np.float64, "shading_cover": np.float64, "matrix_root_index": np.float64, "cell_offsets": np.int64, "entry_cell": np.int64, "entry_species": np.int32,
           "entry_area_all_m2": np.float64, "entry_area_live_m2": np.float64, "root": np.float64, "lower": np.float64, "weights": np.float64, "has_w": np.uint8, "omitted_ids": np.int64}
_FIELD_ORDER = tuple(_DTYPES)
META_KEYS = frozenset(("tag", "version", "compact", "content_sha256", "controls", "receiver_ids", "source", "package", "external_helpers", "identity", "mask_sha256", "surface_sha256",
                       "binding_sha256", "record_sha256"))
COMPACT_KEYS = frozenset(("schema", "version", "field_order", "species_ids", "n_cells", "shape", "cell_area_hex", "n_entries", "n_plants", "root_layers", "n_omitted"))
CONTROL_KEYS = frozenset(("a1", "a2", "a3_layout", "a3", "max_fraction_per_hour", "max_rate_layout", "max_intake_rate_m_s", "has_background", "n_layers", "n_cells", "sha256"))


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


def _shapes(N: int, M: int, S: int, L: int, K: int) -> dict:
    return {"splash_cover": (N,), "shading_cover": (N,), "matrix_root_index": (N,), "cell_offsets": (N + 1,), "entry_cell": (M,), "entry_species": (M,), "entry_area_all_m2": (M,),
            "entry_area_live_m2": (M,), "root": (S, L), "lower": (S,), "weights": (S, L), "has_w": (S,), "omitted_ids": (K,)}


def _seal_any(a: np.ndarray) -> np.ndarray:
    """One declared copy into an immutable `bytes` object; the result is a read-only view whose base chain ends in `bytes` (`setflags(write=True)` is impossible anywhere in it)."""
    a = np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _ends_in_bytes(a: np.ndarray) -> bool:
    node: Any = a
    while isinstance(node, np.ndarray):
        if node.flags.writeable:
            return False
        node = node.base
    return type(node) is bytes


MAX_META_DEPTH = 8  # the genuine metadata documents are at most 4 deep (helpers list -> dict; identity -> domain / cap / transfers); anything deeper (or cyclic) is hostile
_SCALAR_FROZEN = (np.integer, np.floating)


def _str_cost(v: str) -> int:
    """UPPER bound of the JSON size of one string without allocating: exact length when it needs no escaping, 6 per character otherwise (`ensure_ascii`)."""
    plain = v.isascii() and v.isprintable() and '"' not in v and "\\" not in v
    return (len(v) if plain else 6 * len(v)) + 2


def _json_bound(obj: Any, limit: int, *, frozen: bool = False, max_depth: int = MAX_META_DEPTH) -> int:
    """UPPER bound of the canonical JSON size of a metadata document, walked as a STREAM: an explicit stack of container ITERATORS (never a list of children, no set, no dict copy, no dump), so the
    owned memory is O(depth) however wide the document is, and the total is checked after EVERY element (a hostile million-element list stops at the limit after ~limit / 4 elements). Depth is
    bounded (`MAX_META_DEPTH`), which also ends cycles without recursion or an unbounded walk. Types are checked EXACTLY (`type(v) is ...`; with `frozen=True` also a read-only mapping, tuple and numpy
    scalars, the layout of LEAF's frozen identity). A string costs its escaped upper bound, an int its decimal digits from `bit_length` (no truncation to 24 bytes), a finite float 26. Refused:
    any other type, a non-string key, a non-finite float, a total above `limit`, a depth above `max_depth` (all before any copy of the document)."""
    from collections.abc import Mapping
    total = 0
    stack: list = []

    def over() -> None:
        raise VegetationBindingError(f"the metadata document exceeds {limit} bytes (bounded rejection before any owned work)")

    def visit(v: Any) -> int:
        """Cost of a SCALAR, or -1 after pushing a container's iterator."""
        t = type(v)
        if t is str:
            if len(v) + 2 > limit:
                over()
            return _str_cost(v)
        if v is None or t is bool:
            return 5
        if t is int or (frozen and isinstance(v, np.integer)):
            return (int(v).bit_length() * 30103) // 100000 + 3
        if t is float or (frozen and isinstance(v, np.floating)):
            if not math.isfinite(float(v)):
                raise VegetationBindingError("a non-finite float is not allowed in the record metadata")
            return 26
        if len(stack) >= max_depth:
            raise VegetationBindingError(f"the metadata document is nested deeper than {max_depth} levels (hostile or cyclic)")
        if t is list or (frozen and t is tuple):
            stack.append((iter(v), False))
            return -1
        if t is dict or (frozen and isinstance(v, Mapping)):
            stack.append((iter(v.items()), True))
            return -1
        raise VegetationBindingError(f"unsupported metadata value type {t.__name__}")

    c = visit(obj)
    total += max(c, 0) + (2 if c < 0 else 0)
    while stack:
        it, is_dict = stack[-1]
        try:
            item = next(it)
        except StopIteration:
            stack.pop()
            continue
        total += 1
        if is_dict:
            k, v = item
            if type(k) is not str:
                raise VegetationBindingError("metadata keys must be strings")
            total += _str_cost(k) + 1
        else:
            v = item
        c = visit(v)
        total += max(c, 0)
        if total > limit:
            over()
    return total


# ------------------------------------------------------------------------------------------------------------------------------ memory models
def _resident_bytes(N: int, M: int, S: int, L: int, K: int, n_enabled: int, control_cell_maps: int, meta_json_bytes: int = 0) -> int:
    arrays = 3 * _F8 * N + _F8 * (N + 1) + M * (_F8 + 4 + _F8 + _F8) + _F8 * S * L * 2 + _F8 * S + S + _F8 * K
    return int(arrays + _F8 * S + _F8 * n_enabled + _F8 * n_enabled * L + _F8 * N * control_cell_maps + meta_json_bytes + wvb.PY_RECORD_ALLOWANCE_BYTES)


def estimate_cached_stage_bytes(n_cells: int, n_entries: int, n_species: int, n_layers: int, n_omitted: int = 0, *, n_enabled: int = 0, control_cell_maps: int = 0, backend: str = "numpy",
                                source_cold: bool = True, with_mask: bool = False, max_id_chars: int = 0, meta_json_bytes: int = 0) -> dict:
    """Declared host bytes of ONE `stage_cached_inputs`, stage by stage (the max stage is the peak; the caller reserve is counted ONCE by the quota, never inside these numbers).

    Live simultaneously, in order: IDS (after the cheap pre-quota gate): the sorted species-id pointer list 8 S + the sorted receiver list 8 n (subset check by bisection; no quadratic
    membership). SEAL: the immutable copies being built (resident, grows to the full `resident_bytes`; the CALLER's load arrays and identifier strings are excluded; the record keeps pointers
    to them: 8 S + 8 n). SOURCE: resident + the cold LEAF package digest / helper hash allowance. H1: resident + LEAF's `estimate_compact_validation_bytes(max_id_chars=)`. H3: resident +
    max(LEAF's `estimate_rederive_bytes` peak, its validation) + the chunked bitwise comparison temporaries. H2: resident + LEAF's PUBLIC `estimate_stream_bytes(...)['weights_stage_bytes']`.
    CONTROLS: resident + the S0 controls build. RECORD: resident + 2 x the metadata JSON documents (the dump and its hash encoding; the resident copy is inside `resident`); identifiers are
    STREAMED into the hashes (a few bytes at a time). `meta_json_bytes` is the measured upper bound of the metadata documents and `max_id_chars` the longest identifier (<= 64)."""
    from maple_syrup.water_vegetation_binding import _load_leaf

    lw = _load_leaf()
    N, M, S, L, K = (wvb._count(v, k, lo) for v, k, lo in ((n_cells, "n_cells", 1), (n_entries, "n_entries", 0), (n_species, "n_species", 1), (n_layers, "n_layers", 1), (n_omitted, "n_omitted", 0)))
    n_en, jb, cid = wvb._count(n_enabled, "n_enabled"), wvb._count(meta_json_bytes, "meta_json_bytes"), wvb._count(max_id_chars, "max_id_chars")
    resident = _resident_bytes(N, M, S, L, K, n_en, wvb._count(control_cell_maps, "control_cell_maps"), jb)
    h1 = int(lw.estimate_compact_validation_bytes(N, M, S, L, K, with_mask=with_mask, max_id_chars=cid))
    h3v = int(lw.estimate_compact_validation_bytes(N, M, S, L, K, with_mask=with_mask))  # LEAF admits H3's validation without the header term
    re = int(lw.estimate_rederive_bytes(N, S, backend=backend)["peak_bytes"])
    wss = int(lw.estimate_stream_bytes(N, M, n_en, L)["weights_stage_bytes"])
    compare = 2 * _F8 * min(COMPARE_CHUNK, N)
    stages = {"ids": _F8 * S + _F8 * n_en,
              "seal": resident,
              "source": resident + (wvb.SOURCE_DIGEST_ALLOWANCE_BYTES if source_cold else 0),
              "h1": resident + h1,
              "h3": resident + max(re, h3v) + compare,
              "h2": resident + wss,
              "controls": resident + wvb.estimate_controls_bytes(L, a3_cells=N * int(control_cell_maps > 0), max_rate_cells=N * int(control_cell_maps > 1))["peak_bytes"],
              "record": resident + 2 * jb}
    return {"stages": {k: int(v) for k, v in stages.items()}, "peak_bytes": int(max(stages.values())), "resident_bytes": int(resident), "h1_bytes": h1, "h3_bytes": re,
            "h2_weights_stage_bytes": wss,
            "scope": "declared host arithmetic; the caller's load arrays and identifier strings, the interpreter and JIT caches are excluded; one cold derivation, no history"}


def estimate_cached_stage_bytes_from_meta(meta: Any, *, backend: str = "numpy", source_cold: bool = True, with_mask: bool = False) -> dict:
    """The stage model for a given closed `meta` (the sizes, the longest identifier and the metadata document bytes come from the cheap pre-quota gate; nothing is allocated). A codec that
    admits the whole load BEFORE reading arrays uses this."""
    g = _gate_meta(meta)
    maps = int(g["ctl"]["a3_layout"] == "cell") + int(g["ctl"]["max_rate_layout"] == "cell")
    return estimate_cached_stage_bytes(g["N"], g["M"], g["S"], g["L"], g["K"], n_enabled=g["n_receivers"], control_cell_maps=maps, backend=backend, source_cold=source_cold,
                                       with_mask=with_mask, max_id_chars=g["max_id"], meta_json_bytes=g["meta_bytes"])


def estimate_capture_bytes(n_cells: int, n_entries: int, n_species: int, n_layers: int, n_omitted: int = 0, *, copy_arrays: bool = False, n_enabled: int = 0, meta_json_bytes: int = 0) -> dict:
    """Declared host bytes of `capture_cached_inputs`. The sealed (immutable bytes) arrays of a LIVE binding are BORROWED by reference (zero copy); only an array whose base chain does not end in
    immutable bytes is copied (`copy_arrays=True` models the worst case: every array). The binding's controls and receiver table are borrowed too. The metadata documents are the only O(bytes)
    terms: the plain copy of the identity / source / helpers (1x), their canonical dumps (1x) and the resident strings (1x) = `3 x meta_json_bytes`; the species ids are NOT copied (the record shares
    the producer's immutable tuple and the identity is copied without them). Hashing is chunked / streamed. `peak = copies + 3 meta + the Python allowance` (no padding)."""
    N, M, S, L, K = (wvb._count(v, k, lo) for v, k, lo in ((n_cells, "n_cells", 1), (n_entries, "n_entries", 0), (n_species, "n_species", 1), (n_layers, "n_layers", 1), (n_omitted, "n_omitted", 0)))
    jb = wvb._count(meta_json_bytes, "meta_json_bytes")
    full = _resident_bytes(N, M, S, L, K, n_enabled, 0)
    return {"peak_bytes": int((full if copy_arrays else 0) + 3 * jb + wvb.PY_RECORD_ALLOWANCE_BYTES), "borrowed_bytes": int(0 if copy_arrays else full), "meta_json_bytes": int(jb),
            "scope": "declared host arithmetic; arrays borrowed unless not immutable bytes; the identity accessor copies of the caller are excluded"}


def _capture_sizes(b: Any) -> dict:
    """The sizes `capture_cached_inputs` needs, from a verified binding WITHOUT copying anything (the frozen identity is walked as a stream; the species ids are skipped)."""
    comp = wvb._compact_of(b)
    arrays_in = comp["arrays"]
    ident = comp["identity"]
    idb = _json_bound({k: ident[k] for k in ident if k != "species_ids"}, META_JSON_MAX_BYTES, frozen=True)
    jb = idb + len(b.source_json) + len(b.external_helpers_json)
    need_copy = any(not _ends_in_bytes(arrays_in[k]) for k in _FIELD_ORDER) or not _ends_in_bytes(b.receiver_weights)
    return {"comp": comp, "meta_bytes": jb, "need_copy": need_copy, "N": int(comp["n_cells"]), "M": int(comp["n_entries"]), "S": len(comp["species_ids"]), "L": int(comp["root_layers"]),
            "K": int(arrays_in["omitted_ids"].shape[0]), "n_enabled": len(b.enabled_species)}


def estimate_capture_bytes_from_binding(binding: Any) -> dict:
    """The capture model for a given genuine S0 binding (metadata walk only; nothing is copied)."""
    z = _capture_sizes(wvb.verify_binding(binding))
    return estimate_capture_bytes(z["N"], z["M"], z["S"], z["L"], z["K"], copy_arrays=z["need_copy"], n_enabled=z["n_enabled"], meta_json_bytes=z["meta_bytes"])


# ------------------------------------------------------------------------------------------------------------------------------ the record
@dataclass(frozen=True, eq=False)
class CachedVegetationInputs:
    """The sealed cached recipe. Minted ONLY by `capture_cached_inputs` / `stage_cached_inputs`. Arrays are read-only views whose base chain ends in immutable `bytes`; metadata is plain frozen
    JSON / tuples / scalars. Every field is in the registry fingerprint. `origin` is `cached_derived`; the record is NOT a live `VegetationProjection`, NOT a `PlantSession` receipt and NOT a
    trusted file: the recorded source identity is the HELD identity of an earlier generation."""

    arrays: tuple
    species_ids: tuple
    n_cells: int
    shape: tuple
    cell_area_m2: float
    n_entries: int
    n_plants: int
    root_layers: int
    n_omitted: int
    schema: str
    compact_version: int
    content_sha256: str
    controls: Any
    receiver_ids: tuple
    receiver_weights: Any
    source_json: str
    package_json: str
    helpers_json: str
    identity_json: str
    mask_sha256: str
    surface_sha256: str
    mask_entry_guard_checked: bool
    binding_sha256: str
    record_sha256: str
    resident_bytes: int
    origin: str = wvb.ORIGIN_CACHED
    tag: str = CACHED_TAG
    live_session_proof: bool = False

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise VegetationBindingError("CachedVegetationInputs is produced only by capture_cached_inputs / stage_cached_inputs (no direct construction, replace() or copy)")

    def __init_subclass__(cls, **kw):
        raise TypeError("CachedVegetationInputs cannot be subclassed (sealed class)")

    def __copy__(self):
        raise VegetationBindingError("CachedVegetationInputs is not copyable; capture or stage another")

    def __deepcopy__(self, memo):
        raise VegetationBindingError("CachedVegetationInputs is not copyable; capture or stage another")

    def __reduce_ex__(self, protocol):
        raise VegetationBindingError("CachedVegetationInputs is not serialisable here; use export_cached_record (a closed metadata / array record)")

    @property
    def source(self) -> dict:
        """A fresh detached copy of the RECORDED (held) biological source identity."""
        return json.loads(self.source_json)

    @property
    def identity(self) -> dict:
        """A fresh detached, caller-owned copy of the recorded LEAF projection identity (held; not a live proof). The species ids are stored once (the record field) and re-added here."""
        d = json.loads(self.identity_json)
        d["species_ids"] = list(self.species_ids)
        return d


_REGISTRY: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def verify_cached_inputs(record: Any) -> CachedVegetationInputs:
    """Public genuine-factory proof: exact class (no subclass), registry, every field by value / pointer, every array's immutable-bytes base chain, the controls' own seal. Metadata only: no content
    rehash and no copy."""
    if type(record) is not CachedVegetationInputs:
        raise VegetationBindingError(f"record must be a CachedVegetationInputs, got {type(record).__name__}")
    fp = _REGISTRY.get(record)
    if fp is None:
        raise VegetationBindingError("the CachedVegetationInputs record is forged, uninitialised or was not produced by its factories")
    try:
        now = wvb._fields_fp(record, HELD_ID_FIELDS)
    except (AttributeError, TypeError) as exc:
        raise VegetationBindingError("the CachedVegetationInputs record is incomplete") from exc
    if now != fp:
        raise VegetationBindingError("the CachedVegetationInputs record was altered after sealing (a field or array was replaced)")
    for name, a in record.arrays:
        if not _ends_in_bytes(a):
            raise VegetationBindingError(f"array {name!r} is not backed by immutable bytes")
    if not _ends_in_bytes(record.receiver_weights):
        raise VegetationBindingError("the receiver table is not backed by immutable bytes")
    wvb.verify_site_controls(record.controls)
    return record


def _arrays_dict(record: CachedVegetationInputs) -> dict:
    return {name: a for name, a in record.arrays}


def cached_compact(record: CachedVegetationInputs, *, with_identity: bool = False) -> Any:
    """The compact mapping of the record (same layout as LEAF's `export_compact()`): read-only no-copy views plus the recorded scalars and content hash. `with_identity=True` adds the detached
    identity (a caller-owned copy: parse + the species-id list); the internal H1 / H2 calls never pay for it."""
    verify_cached_inputs(record)
    arrays = types.MappingProxyType({k: v.view() for k, v in _arrays_dict(record).items()})
    out = {"schema": record.schema, "version": record.compact_version, "field_order": _FIELD_ORDER, "arrays": arrays, "species_ids": tuple(record.species_ids), "n_cells": record.n_cells,
           "shape": tuple(record.shape), "cell_area_m2": record.cell_area_m2, "n_entries": record.n_entries, "n_plants": record.n_plants, "root_layers": record.root_layers,
           "content_sha256": record.content_sha256}
    if with_identity:
        out["identity"] = types.MappingProxyType(record.identity)
    return types.MappingProxyType(out)


def cached_r1(record: CachedVegetationInputs) -> np.ndarray:
    verify_cached_inputs(record)
    return _arrays_dict(record)["matrix_root_index"]


def cached_stream_estimate(record: CachedVegetationInputs, previous_held: bool) -> dict:
    """The PUBLIC LEAF stream model for this record (same numbers the H2 stream admits)."""
    verify_cached_inputs(record)
    lw = wvb._load_leaf()
    return lw.estimate_stream_bytes(record.n_cells, record.n_entries, len(record.receiver_ids), record.root_layers, previous_held=bool(previous_held))


def cached_preferential_inputs(record: CachedVegetationInputs, max_bytes: Any, reserve_bytes: Any, reader: Any) -> tuple:
    """`(ids, raw weights (n, L), contributions)` through the canonical H2 on the record's compact arrays (validated once when the record was staged / captured)."""
    lw = wvb._load_leaf()
    return lw.preferential_inputs_from_compact(cached_compact(record), max_bytes=max_bytes, reserve_bytes=reserve_bytes, available_bytes_fn=reader)


# ------------------------------------------------------------------------------------------------------------------------------ record identity
def _record_sha(content: str, controls_sha: str, binding_sha: str, source_json: str, package_json: str, helpers_json: str, identity_json: str, mask: str, surface: str,
                receiver_ids: tuple, receiver_weights: np.ndarray, species: tuple, shape: tuple, area: float, schema: str, compact_version: int) -> str:
    """The record SHA, version 2: one canonical JSON document of the scalars and the (already canonical) metadata documents by reference, then the identifiers STREAMED one at a time and the
    receiver table chunked. No parse, no identifier-sized JSON copy."""
    h = hashlib.sha256()
    h.update(wvb._frozen_json({"tag": CACHED_TAG, "version": CACHED_VERSION, "content": content, "controls": controls_sha, "binding": binding_sha, "mask": mask, "surface": surface,
                               "shape": list(shape), "cell_area_hex": float(area).hex(), "schema": schema, "compact_version": compact_version, "n_species": len(species),
                               "n_receivers": len(receiver_ids)}).encode())
    for doc in (source_json, package_json, helpers_json, identity_json):
        h.update(doc.encode() + b"\n")
    for group in (species, receiver_ids):
        h.update(b"[")
        for name in group:
            h.update(name.encode() + b"\0")
        h.update(b"]")
    wvb._hash_chunked(h, receiver_weights)
    return h.hexdigest()


def _record_from(arrays: dict, *, species: tuple, N: int, shape: tuple, area: float, M: int, P: int, L: int, K: int, schema: str, compact_version: int, content: str, controls: Any,
                 ids: tuple, table: np.ndarray, source: dict, package: dict, helpers: list, identity: dict, mask: str, surface: str, guard: bool, binding_sha: str) -> CachedVegetationInputs:
    """Mint the record. `identity` is the recorded identity WITHOUT `species_ids` (stored once, as the record field)."""
    if "species_ids" in identity:
        raise VegetationBindingError("the stored identity must not duplicate the species ids")
    species, ids = wvb._check_id_tuple(tuple(species), "species_ids"), wvb._check_id_tuple(tuple(ids), "receiver_ids")  # validated ONCE here; the proofs fingerprint them in O(1)
    source_json, package_json, helpers_json, identity_json = (wvb._frozen_json(x) for x in (source, package, helpers, identity))
    rsha = _record_sha(content, controls.sha256, binding_sha, source_json, package_json, helpers_json, identity_json, mask, surface, ids, table, species, shape, area, schema, compact_version)
    maps = int(controls.a3_layout == "cell") + int(controls.max_rate_layout == "cell")
    jb = sum(len(x) for x in (source_json, package_json, helpers_json, identity_json))
    resident = _resident_bytes(N, M, len(species), L, K, len(ids), maps, jb)
    items = tuple((k, arrays[k]) for k in _FIELD_ORDER)
    with _minting():
        rec = CachedVegetationInputs(items, tuple(species), N, tuple(shape), float(area), M, P, L, K, schema, int(compact_version), content, controls, tuple(ids), table, source_json,
                                     package_json, helpers_json, identity_json, mask, surface, bool(guard), binding_sha, rsha, int(resident))
    _REGISTRY[rec] = wvb._fields_fp(rec, HELD_ID_FIELDS)
    return rec


# ------------------------------------------------------------------------------------------------------------------------------ capture (from a genuine S0 binding)
def capture_cached_inputs(binding: Any, *, max_bytes: Any, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None) -> CachedVegetationInputs:
    """Capture the cached recipe of a genuine S0 `VegetationBinding` (`verify_binding` first). The binding's arrays are sealed immutable bytes: they are BORROWED by reference (their base chain keeps
    the bytes alive after the projection / binding / plant objects are freed); an array whose chain does not end in immutable bytes is copied under admission. Nothing of the plants, a session
    or the water is read. The record keeps the binding's SEMANTIC SHA (origin-independent) and the recorded held surface / mask hashes."""
    b = wvb.verify_binding(binding)
    z = _capture_sizes(b)  # metadata walk only: the identity is measured as a stream, nothing is copied before the admission
    comp, arrays_in, N, M, L, K = z["comp"], z["comp"]["arrays"], z["N"], z["M"], z["L"], z["K"]
    est = estimate_capture_bytes(N, M, z["S"], L, K, copy_arrays=z["need_copy"], n_enabled=z["n_enabled"], meta_json_bytes=z["meta_bytes"])
    wvb._admit("capture_cached_inputs", est["peak_bytes"], max_bytes, reserve_bytes, available_bytes_fn)
    arrays = {k: (arrays_in[k] if _ends_in_bytes(arrays_in[k]) else _seal_any(arrays_in[k])) for k in _FIELD_ORDER}
    table = b.receiver_weights if _ends_in_bytes(b.receiver_weights) else _seal_any(b.receiver_weights)
    ident = comp["identity"]
    ident_plain = _plain(ident, skip=("species_ids",))  # the species ids (S pointers) are never copied: the record shares the producer's immutable tuple
    helpers = json.loads(b.external_helpers_json)
    package = {"digest": "sha256:" + b.leaf_package_sha256, "file_count": b.leaf_package_file_count}
    full_pkg = ident_plain.get("package")
    if isinstance(full_pkg, dict):
        package = dict(full_pkg)
    return _record_from(arrays, species=tuple(comp["species_ids"]), N=N, shape=tuple(comp["shape"]), area=float(comp["cell_area_m2"]), M=M, P=int(comp["n_plants"]), L=L, K=K,
                        schema=str(comp["schema"]), compact_version=int(comp["version"]), content=comp["content_sha256"], controls=b.controls, ids=tuple(b.enabled_species), table=table, source=b.source, package=package, helpers=helpers,
                        identity=ident_plain, mask=b.mask_sha256, surface=b.surface_sha256, guard=False, binding_sha=b.sha256)


def _plain(obj: Any, skip: tuple = ()) -> Any:
    """A mutable-free plain copy (dict / list / scalars) of a frozen mapping tree; `skip` names TOP-LEVEL keys that are not copied."""
    from collections.abc import Mapping
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items() if k not in skip}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ------------------------------------------------------------------------------------------------------------------------------ export (closed metadata / array record)
def _f(v: float) -> str:
    return float(v).hex()


def export_cached_record(record: CachedVegetationInputs) -> dict:
    """The CLOSED metadata / array record of the cached recipe for a later strict codec: `{'meta': plain JSON-able dict, 'arrays': read-only views (no copy)}`. Floats are exact hex strings. The
    arrays are the 13 compact arrays, `receiver_weights_enabled` and the controls' maps / background; nothing S x N, nothing derived from plants. Metadata-only validation."""
    r = verify_cached_inputs(record)
    c = r.controls
    ctl = {"a1": _f(c.a1), "a2": _f(c.a2), "a3_layout": c.a3_layout, "a3": _f(c.a3) if c.a3_layout == "scalar" else None, "max_fraction_per_hour": _f(c.max_fraction_per_hour),
           "max_rate_layout": c.max_rate_layout, "max_intake_rate_m_s": _f(c.max_intake_rate_m_s) if c.max_rate_layout == "scalar" else None,
           "has_background": c.background_receiver_weights is not None, "n_layers": c.n_layers, "n_cells": c.n_cells, "sha256": c.sha256}
    meta = {"tag": CACHED_TAG, "version": CACHED_VERSION,
            "compact": {"schema": r.schema, "version": r.compact_version, "field_order": list(_FIELD_ORDER), "species_ids": list(r.species_ids), "n_cells": r.n_cells,
                        "shape": list(r.shape), "cell_area_hex": _f(r.cell_area_m2), "n_entries": r.n_entries, "n_plants": r.n_plants, "root_layers": r.root_layers, "n_omitted": r.n_omitted},
            "content_sha256": r.content_sha256, "controls": ctl, "receiver_ids": list(r.receiver_ids), "source": r.source, "package": json.loads(r.package_json),
            "external_helpers": json.loads(r.helpers_json), "identity": json.loads(r.identity_json), "mask_sha256": r.mask_sha256, "surface_sha256": r.surface_sha256, "binding_sha256": r.binding_sha256,
            "record_sha256": r.record_sha256}
    arrays = {k: v.view() for k, v in r.arrays}
    arrays["receiver_weights_enabled"] = r.receiver_weights.view()
    for name, arr in (("a3", r.controls.a3), ("max_intake_rate_m_s", r.controls.max_intake_rate_m_s), ("background_receiver_weights", r.controls.background_receiver_weights)):
        if isinstance(arr, np.ndarray):
            arrays[f"controls.{name}"] = arr.view()
    return {"meta": meta, "arrays": types.MappingProxyType(arrays)}


# ------------------------------------------------------------------------------------------------------------------------------ cold stage (closed record -> sealed record)
def _gate_meta(meta: Any) -> dict:
    """CHEAP, PRE-QUOTA metadata gate: closed key sets, STRICT tag / version / compact version types, exact counts, per-identifier type and length (a genuine catalog id is 1..64 characters), list
    lengths and the walked JSON size of the O(1) documents. It allocates no container, copy, set or JSON string (only small ints): a hostile file is rejected here in time linear in what it
    carries and with O(1) owned memory. Everything that needs a sort / copy / set (uniqueness, subset, pattern) is done AFTER the admission, modelled."""
    if not isinstance(meta, dict) or len(meta) != len(META_KEYS) or any(k not in META_KEYS for k in meta):
        raise VegetationBindingError("meta must be a dict with exactly the closed cached-inputs keys")
    if type(meta["tag"]) is not str or meta["tag"] != CACHED_TAG or type(meta["version"]) is not str or meta["version"] != CACHED_VERSION:
        raise VegetationBindingError("unknown cached-inputs tag / version (the version is the STRING " + repr(CACHED_VERSION) + ")")
    cp = meta["compact"]
    if not isinstance(cp, dict) or len(cp) != len(COMPACT_KEYS) or any(k not in COMPACT_KEYS for k in cp):
        raise VegetationBindingError("meta['compact'] must have exactly the closed compact keys")
    if type(cp["schema"]) is not str or type(cp["version"]) is not int:
        raise VegetationBindingError("compact schema must be a string and compact version a strict integer")
    fo = cp["field_order"]
    if not isinstance(fo, (list, tuple)) or len(fo) != len(_FIELD_ORDER) or any(a != b for a, b in zip(fo, _FIELD_ORDER, strict=True)):
        raise VegetationBindingError("meta['compact']['field_order'] is not the canonical order")
    ids = cp["species_ids"]
    if type(ids) is not list or not ids:
        raise VegetationBindingError("species_ids must be a non-empty list")
    max_id = 0
    for v in ids:
        if type(v) is not str or not 0 < len(v) <= MAX_ID_CHARS:
            raise VegetationBindingError(f"every species id must be a string of 1..{MAX_ID_CHARS} characters (the catalog contract); a hostile identifier is refused before any owned work")
        max_id = max(max_id, len(v))
    N, M, P, L, K = (wvb._count(cp[k], k, lo) for k, lo in (("n_cells", 1), ("n_entries", 0), ("n_plants", 0), ("root_layers", 1), ("n_omitted", 0)))
    sh = cp["shape"]
    if not isinstance(sh, list) or len(sh) != 2 or any(type(v) is not int or v < 1 for v in sh) or sh[0] * sh[1] != N:
        raise VegetationBindingError("shape must be [ny, nx] with ny * nx == n_cells")
    if M > N * len(ids):
        raise VegetationBindingError("n_entries exceeds n_cells * n_species")
    try:
        area = float.fromhex(cp["cell_area_hex"])
    except (TypeError, ValueError) as exc:
        raise VegetationBindingError("cell_area_hex is not an exact float hex string") from exc
    if not (area > 0.0 and np.isfinite(area)):
        raise VegetationBindingError("cell area must be finite and > 0")
    for k in ("content_sha256", "binding_sha256", "record_sha256", "mask_sha256", "surface_sha256"):
        wvb._hex(meta[k], k)
    for k in ("source", "package", "identity"):
        if not isinstance(meta[k], dict):
            raise VegetationBindingError(f"meta[{k!r}] must be a dict")
    helpers, recv = meta["external_helpers"], meta["receiver_ids"]
    if type(helpers) is not list or len(helpers) > MAX_HELPERS or type(recv) is not list or len(recv) > len(ids):
        raise VegetationBindingError("external_helpers must be a short list and receiver_ids a list no longer than the species list")
    for v in recv:
        if type(v) is not str or not 0 < len(v) <= MAX_ID_CHARS:
            raise VegetationBindingError("every receiver id must be a string of 1..64 characters")
    jb = _json_bound([meta["source"], meta["package"], helpers, meta["identity"]], META_JSON_MAX_BYTES)
    ctl = meta["controls"]
    if not isinstance(ctl, dict) or len(ctl) != len(CONTROL_KEYS) or any(k not in CONTROL_KEYS for k in ctl):
        raise VegetationBindingError("meta['controls'] must have exactly the closed controls keys")
    if ctl["a3_layout"] not in ("scalar", "cell") or ctl["max_rate_layout"] not in ("scalar", "cell") or type(ctl["has_background"]) is not bool:
        raise VegetationBindingError("controls layouts must be 'scalar' or 'cell' and has_background a bool")
    return {"N": N, "M": M, "P": P, "L": L, "K": K, "S": len(ids), "shape": tuple(sh), "area": area, "ids": ids, "ctl": ctl, "max_id": max_id, "meta_bytes": jb, "n_receivers": len(recv)}


def _post_meta(g: dict, meta: dict, lw: Any) -> None:
    """Checks that need a sort / pattern / cross-reference, run AFTER the admission (modelled: the `ids` stage 8 S + 8 n): species id pattern and uniqueness, receivers a subset of the species
    (bisection, no quadratic membership), and the recorded identity cross-checked against the compact record like the live `_check_domain_fields`."""
    import bisect

    from leaf import traits as _traits
    pat = _traits.SPECIES_ID_PATTERN
    ids, recv = g["ids"], meta["receiver_ids"]
    for v in ids:
        if pat.fullmatch(v) is None:
            raise VegetationBindingError(f"species id {v!r} does not match the catalog pattern")
    order = sorted(ids)
    for a, b in itertools.pairwise(order):
        if a == b:
            raise VegetationBindingError("species_ids must be distinct")
    for v in recv:
        k = bisect.bisect_left(order, v)
        if k >= len(order) or order[k] != v:
            raise VegetationBindingError("receiver_ids names an unknown species")
    del order
    cp, ident = meta["compact"], meta["identity"]
    if "species_ids" in ident:
        raise VegetationBindingError("the stored identity must not duplicate the species ids")
    if cp["schema"] != lw.SCHEMA or cp["version"] != lw.VERSION or type(ident.get("schema")) is not str or ident["schema"] != cp["schema"] or type(ident.get("version")) is not int or ident["version"] != cp["version"]:
        raise VegetationBindingError(f"unknown projection schema/version; this consumer knows {lw.SCHEMA!r} version {lw.VERSION!r} only, and the identity must agree with the compact record")
    if type(ident.get("root_layers")) is not int or ident["root_layers"] != g["L"]:
        raise VegetationBindingError("the identity root_layers differs from the compact record")
    if ident.get("catalog_hash") != meta["source"].get("catalog_hash"):
        raise VegetationBindingError("the identity catalog_hash differs from the recorded source catalog_hash")
    wvb._check_domain_area(ident.get("domain"), g["shape"], g["area"])
    for key, want in (("source", meta["source"]), ("package", meta["package"]), ("active_mask_sha256", meta["mask_sha256"]), ("surface_sha256", meta["surface_sha256"]),
                      ("content_sha256", meta["content_sha256"])):
        if ident.get(key) != want:
            raise VegetationBindingError(f"the recorded identity disagrees with the record metadata ({key})")


def _gate_arrays(g: dict, arrays: Any) -> None:
    from collections.abc import Mapping
    N, M, S, L, K = g["N"], g["M"], g["S"], g["L"], g["K"]
    ctl = g["ctl"]
    want = {k: (np.dtype(_DTYPES[k]), shp) for k, shp in _shapes(N, M, S, L, K).items()}
    if ctl["a3_layout"] == "cell":
        want["controls.a3"] = (np.dtype(np.float64), (N,))
    if ctl["max_rate_layout"] == "cell":
        want["controls.max_intake_rate_m_s"] = (np.dtype(np.float64), (N,))
    if ctl["has_background"]:
        want["controls.background_receiver_weights"] = (np.dtype(np.float64), (L,))
    want["receiver_weights_enabled"] = None
    if not isinstance(arrays, Mapping) or set(arrays) != set(want):
        raise VegetationBindingError(f"arrays must hold exactly {sorted(want)}")
    for k, spec in want.items():
        a = arrays[k]
        if type(a) is not np.ndarray or not a.flags.c_contiguous:
            raise VegetationBindingError(f"array {k!r} must be a C-contiguous numpy array")
        if spec is not None and (a.dtype != spec[0] or tuple(a.shape) != spec[1]):
            raise VegetationBindingError(f"array {k!r} must be {spec[0]} of shape {spec[1]}, got {a.dtype} {tuple(a.shape)}")
    r = arrays["receiver_weights_enabled"]
    if r.dtype != np.float64 or r.ndim != 2 or r.shape[1] != L:
        raise VegetationBindingError(f"receiver_weights_enabled must be float64 (n, {L})")


def stage_cached_inputs(meta: dict, arrays: Any, *, expected_record_sha256: Any, expected_binding_sha256: Any, expected_projection_sha256: Any, expected_leaf_package_sha256: Any,
                        expected_external_helpers: Any, expected_source: dict, active_mask: np.ndarray | None, backend: str = "numpy", max_bytes: Any, reserve_bytes: Any = 0,
                        available_bytes_fn: Callable[[], int] | None = None) -> CachedVegetationInputs:
    """COLD, explicit staging of a cached record from a closed `{'meta', 'arrays'}` pair (`export_cached_record` layout) plus EXTERNAL expectations. Nothing exists until every check passed; the
    caller's arrays and metadata are never modified (a failure or retry leaves them as they were).

    Order: metadata gate (closed keys, tag / version, types, counts; arrays' dtype / shape / contiguity) -> composed ADMISSION of the whole pipeline (`estimate_cached_stage_bytes`) -> one immutable
    sealed COPY of each array (the loaded arrays are untrusted and may be writable; everything below reads the sealed copies) -> ACTUAL LEAF package / external helper pins through the same strict
    S0 validation as `bind_projection` (recorded-only is refused) and the expected source with every mandatory key -> H1 `compact_content_sha256` == recorded == expected -> positive fixed species
    root fractions sum to 1 within the trait contract -> H3 `rederive_fields_from_compact` bitwise equal to the stored splash / shading / R1 (explicit `active_mask` with the recorded sha256 =>
    entry-active guard; `None` records that the guard was NOT checked, never inferred from empty groups) -> H2 canonical receiver table == stored -> controls re-minted by `site_controls` (native
    validation) == recorded -> semantic binding SHA == recorded == expected -> record SHA == recorded == expected -> mint. The record SHA protects integrity only: it is not a trusted-file or
    accepted-parent proof. The HELD surface hash is preserved (it may predate the current bed)."""
    g = _gate_meta(meta)  # cheap, pre-quota, O(1) owned bytes
    _gate_arrays(g, arrays)
    N, M, S, L, K = g["N"], g["M"], g["S"], g["L"], g["K"]
    ctl = g["ctl"]
    pkg_expected = wvb._hex(expected_leaf_package_sha256, "expected_leaf_package_sha256")
    proj_expected, bind_expected, rec_expected = (wvb._hex(v, n) for v, n in ((expected_projection_sha256, "expected_projection_sha256"), (expected_binding_sha256, "expected_binding_sha256"),
                                                                              (expected_record_sha256, "expected_record_sha256")))
    helpers_expected = wvb._normalise_helpers(expected_external_helpers, "expected_external_helpers")
    if not isinstance(expected_source, dict) or not expected_source:
        raise VegetationBindingError("expected_source must be a non-empty dict carrying every mandatory key of its source kind")
    if backend not in ("numpy", "numba"):
        raise VegetationBindingError(f"backend must be numpy or numba (the CPU re-derivation), got {backend!r}")
    mask_flat = None
    if active_mask is not None:
        if type(active_mask) is not np.ndarray or active_mask.dtype != np.bool_ or tuple(active_mask.shape) != g["shape"] or not active_mask.flags.c_contiguous:
            raise VegetationBindingError(f"active_mask must be a C-contiguous bool numpy array of shape {g['shape']} (or an explicit None)")
        mask_flat = active_mask.reshape(-1)
    n_en = g["n_receivers"]
    maps = int(ctl["a3_layout"] == "cell") + int(ctl["max_rate_layout"] == "cell")
    if arrays["receiver_weights_enabled"].shape != (n_en, L):
        raise VegetationBindingError(f"receiver_weights_enabled must have shape ({n_en}, {L})")
    lw = wvb._load_leaf()
    for name in ("compact_content_sha256", "rederive_fields_from_compact", "preferential_inputs_from_compact", "estimate_stream_bytes", "estimate_compact_validation_bytes", "estimate_rederive_bytes"):
        if not hasattr(lw, name):
            raise wvb.VegetationConsumerUnavailableError(f"the imported LEAF lacks the public {name}; the cached-inputs consumer needs the compact helpers (H1 / H2 / H3)")
    import os
    cold = (os.path.dirname(os.path.realpath(lw.__file__)) not in wvb._LEAF_CACHE or any(wvb._helper_path(m) not in wvb._HELPER_CACHE for m, _, _ in helpers_expected))
    est = estimate_cached_stage_bytes(N, M, S, L, K, n_enabled=n_en, control_cell_maps=maps, backend=backend, source_cold=cold, with_mask=mask_flat is not None,
                                      max_id_chars=g["max_id"], meta_json_bytes=g["meta_bytes"])
    rb = wvb._count(reserve_bytes, "reserve_bytes")
    wvb._admit("stage_cached_inputs", est["peak_bytes"], max_bytes, rb, available_bytes_fn)
    live = est["resident_bytes"]
    _post_meta(g, meta, lw)  # sort / pattern / identity cross-checks (admitted `ids` stage); still before the first array copy
    ids = tuple(g["ids"])
    recv = tuple(meta["receiver_ids"])
    # ---- sealed immutable copies of EVERY loaded array: all later reads are of these
    sealed = {k: _seal_any(arrays[k]) for k in _FIELD_ORDER}
    table = _seal_any(arrays["receiver_weights_enabled"])
    cmaps = {k: _seal_any(arrays[f"controls.{k}"]) for k in ("a3", "max_intake_rate_m_s", "background_receiver_weights") if f"controls.{k}" in arrays}
    # ---- actual source pins (shared strict S0 validation) and the expected source
    wvb._verify_source_pins(lw, meta["package"], {"helpers": meta["external_helpers"]}, pkg_expected, helpers_expected)
    wvb._check_source(meta["source"], expected_source)
    identity = meta["identity"]
    # ---- H1 (the exact V3 hash)
    compact = types.MappingProxyType({"schema": meta["compact"]["schema"], "version": meta["compact"]["version"], "field_order": _FIELD_ORDER,
                                      "arrays": types.MappingProxyType(dict(sealed)), "species_ids": ids, "n_cells": N, "shape": g["shape"], "cell_area_m2": g["area"], "n_entries": M,
                                      "n_plants": g["P"], "root_layers": L})
    try:
        got = lw.compact_content_sha256(compact, max_bytes=max_bytes, reserve_bytes=rb + live, available_bytes_fn=available_bytes_fn)
    except lw.VegetationProjectionError as exc:
        raise VegetationBindingError(f"compact content refused: {exc}") from exc
    if got != meta["content_sha256"] or wvb._hex(got, "content") != proj_expected:
        raise VegetationBindingError("the recomputed compact content hash differs from the recorded and / or the expected projection hash")
    # ---- fixed species root fractions: positive, sum 1 within the trait contract (the producer's own validity)
    from leaf import traits as _traits
    tol = float(_traits.LAYER_FRACTION_TOL)
    root = sealed["root"]
    for i in range(S):
        if abs(float(sum(float(v) for v in root[i])) - 1.0) > tol:
            raise VegetationBindingError(f"species {ids[i]!r}: root layer fractions must sum to 1 within {tol}")
    # ---- H3: recompute the O(N) fields and compare bitwise with the stored ones
    try:
        fields = lw.rederive_fields_from_compact(compact, backend=backend, active_mask=active_mask,
                                                 expected_active_mask_sha256=meta["mask_sha256"] if active_mask is not None else None,
                                                 max_bytes=max_bytes, reserve_bytes=rb + live, available_bytes_fn=available_bytes_fn)
    except lw.VegetationProjectionError as exc:
        raise VegetationBindingError(f"compact re-derivation refused: {exc}") from exc
    for name, fresh in (("splash_cover", fields.splash_cover), ("shading_cover", fields.shading_cover), ("matrix_root_index", fields.matrix_root_index)):
        stored = sealed[name]
        for lo in range(0, N, COMPARE_CHUNK):
            if fresh[lo:lo + COMPARE_CHUNK].tobytes() != stored[lo:lo + COMPARE_CHUNK].tobytes():
                raise VegetationBindingError(f"the stored {name} differs bitwise from its re-derivation from the compact entries")
    del fields
    # ---- H2: the canonical receiver table equals the stored one
    r_ids, r_w, gen = lw.preferential_inputs_from_compact(compact, max_bytes=max_bytes, reserve_bytes=rb + live, available_bytes_fn=available_bytes_fn)
    gen.close()
    if tuple(r_ids) != recv or r_w.shape != table.shape or r_w.tobytes() != table.tobytes():
        raise VegetationBindingError("the stored receiver table / enabled species differ from the canonical re-derivation (H2)")
    del r_w
    # ---- controls: re-minted through the native-validating factory and compared with the record
    def _ctl_value(layout: str, scalar_hex: Any, name: str):
        if layout == "scalar":
            if not isinstance(scalar_hex, str):
                raise VegetationBindingError(f"controls[{name!r}] scalar must be a float hex string")
            return float.fromhex(scalar_hex)
        if scalar_hex is not None:
            raise VegetationBindingError(f"controls[{name!r}] must be None for a cell layout")
        return cmaps[name]
    if (ctl["has_background"]) != ("background_receiver_weights" in cmaps):
        raise VegetationBindingError("controls.has_background disagrees with the supplied background weights")
    try:
        controls = wvb.site_controls(a1=float.fromhex(ctl["a1"]), a2=float.fromhex(ctl["a2"]), a3=_ctl_value(ctl["a3_layout"], ctl["a3"], "a3"),
                                     max_fraction_per_hour=float.fromhex(ctl["max_fraction_per_hour"]),
                                     max_intake_rate_m_s=_ctl_value(ctl["max_rate_layout"], ctl["max_intake_rate_m_s"], "max_intake_rate_m_s"),
                                     background_receiver_weights=cmaps.get("background_receiver_weights"), n_layers=ctl["n_layers"], n_cells=ctl["n_cells"], max_bytes=max_bytes,
                                     reserve_bytes=rb + live, available_bytes_fn=available_bytes_fn)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, VegetationBindingError):
            raise
        raise VegetationBindingError(f"controls refused: {exc}") from exc
    if controls.sha256 != ctl["sha256"] or controls.n_layers != L or (controls.n_cells is not None and controls.n_cells != N):
        raise VegetationBindingError("the re-minted controls differ from the recorded controls (sha / layers / cells)")
    # ---- semantic binding SHA and record SHA
    bsha = wvb._binding_sha(controls.sha256, pkg_expected, proj_expected, meta["source"], meta["mask_sha256"], meta["surface_sha256"], g["shape"], g["area"], ids, recv, L, helpers_expected, table)
    if bsha != meta["binding_sha256"] or bsha != bind_expected:
        raise VegetationBindingError("the recomputed semantic binding SHA differs from the recorded and / or the expected binding SHA")
    rec = _record_from(sealed, species=ids, N=N, shape=g["shape"], area=g["area"], M=M, P=g["P"], L=L, K=K, schema=meta["compact"]["schema"], compact_version=meta["compact"]["version"],
                       content=got, controls=controls, ids=recv, table=table,
                       source=meta["source"], package=meta["package"], helpers=meta["external_helpers"], identity=identity, mask=meta["mask_sha256"], surface=meta["surface_sha256"],
                       guard=active_mask is not None, binding_sha=bsha)
    if rec.record_sha256 != meta["record_sha256"] or rec.record_sha256 != rec_expected:
        raise VegetationBindingError("the recomputed record SHA differs from the recorded and / or the expected record SHA (integrity only; no trust is implied by a matching hash)")
    return rec


# ------------------------------------------------------------------------------------------------------------------------------ cached binding
def cached_binding(record: CachedVegetationInputs, *, expected_binding_sha256: Any, max_bytes: Any, reserve_bytes: Any = 0,
                   available_bytes_fn: Callable[[], int] | None = None) -> Any:
    """A labelled `cached_derived` `VegetationBinding` over the record (its `projection` is the typed cached record, never a LEAF object), feeding the UNCHANGED `prepare_vegetation_law` on the
    CURRENT column. It reproduces the live binding's semantic SHA. It is a restored recipe, not a new live vegetation generation: nothing here can apply an arbitrary new plant generation, and
    `bind_projection` (the strict LEAF-class path) refuses a cached record."""
    r = verify_cached_inputs(record)
    want = wvb._hex(expected_binding_sha256, "expected_binding_sha256")
    if r.binding_sha256 != want:
        raise VegetationBindingError("the record's semantic binding SHA differs from the expected binding SHA")
    # the two parsed documents (source, package) are the only O(bytes) terms: admitted literally from the recorded string lengths
    wvb._admit("cached_binding", wvb.PY_RECORD_ALLOWANCE_BYTES + 2 * (len(r.source_json) + len(r.package_json)), max_bytes, reserve_bytes, available_bytes_fn)
    src = r.source
    pkg = r.package_json
    pk = json.loads(pkg)
    digest = wvb._hex(pk["digest"], "recorded package digest")
    fc = pk.get("file_count")
    if type(fc) is not int or fc < 1:
        raise VegetationBindingError("the recorded package file_count must be a strict positive integer (no coercion)")
    return wvb._mint_cached_binding(projection=r, controls=r.controls, n_cells=r.n_cells, n_layers=r.root_layers, shape=tuple(r.shape), cell_area_m2=r.cell_area_m2, species_ids=tuple(r.species_ids),
                                    enabled_species=tuple(r.receiver_ids), receiver_weights=r.receiver_weights, n_entries=r.n_entries, leaf_package_sha256=digest, leaf_package_file_count=fc,
                                    external_helpers_json=r.helpers_json, projection_sha256=wvb._hex(r.content_sha256, "content"), source_kind=str(src["kind"]), source_json=r.source_json,
                                    mask_sha256=r.mask_sha256, surface_sha256=r.surface_sha256, sha256=r.binding_sha256, borrowed_bytes=int(r.resident_bytes),
                                    resident_bytes=int(r.receiver_weights.nbytes + wvb.PY_RECORD_ALLOWANCE_BYTES), origin=wvb.ORIGIN_CACHED)

