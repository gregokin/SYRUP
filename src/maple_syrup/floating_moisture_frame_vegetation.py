"""Phase 2.5 vegetation-aware floating soil-moisture DIAGNOSTIC frame (profile v2; CPU/host and device arrays; pure codec, NO LEAF / model import).

A NEW explicit profile beside `floating_moisture_frame` (schema `.../1`, which is unchanged and keeps refusing what it does not know). Same stored arrays (`water_m (L, N)`, `depth_m (N,)`, `active (N,)`, nominal edges, area, geometry, nominal
day, cadence, origin, PHYSICAL time) and the SAME `FloatingFrameIdentity` with its OLD meaning (`property_sha256` = the CURRENT root column identity); the manifest adds ONE closed `vegetation` block (or an explicit `null` for a lineage with
no vegetation generation: the generation-0 contract is explicit, never a fake LEAF value):

    generation, recipe_id, binding_sha256, controls_sha256, projection_sha256, leaf_package_sha256, mask_sha256, surface_sha256, source (plain closed identity record), external_helpers,
    current_column_sha256, current_matrix_op_sha256, current_preferential_sha256 (or null), current_property_generation, current_law_id

The STABLE part (generation counter, recipe, binding, source) is the vegetation generation; the CURRENT part is the law identity of the accepted holder at THIS frame (it changes with a real property refresh while the generation does not).
The apply-time ids are never stored. The values are caller-supplied attribution (`caller_supplied_not_proof`): the codec proves nothing; the genuine writer (`floating_daily_output_vegetation`) derives them from the accepted state.
No plant population, CSR, S x N array or history is stored.

Why a sibling module and not a parameter of the v1 codec. The v1 module binds its schema, closed key sets, identity fields, writer-module hash and verification to module constants that sealed v1 frames and the v1 output depend on; making them
parameters would rewrite the v1 codec and risk its seals. This module therefore reuses the v1 module's PARAMETER-FREE bounded primitives (array/device validation, the chunked device-to-host traversal, positional writes/compare, root scan,
discard, npy layout) and the G0 no-follow / no-clobber / fsync / limits primitives, and owns only the v2 manifest, identity, write/replay flow and reader. Mutual refusal: the v1 reader rejects a v2 manifest (schema, closed keys) and this reader rejects
v1; a v2 frame is sealed with THIS module's hash, so a replay by other source is refused.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup import floating_moisture_frame as g1
from maple_syrup import soil_moisture_output as g0
from maple_syrup.soil_moisture_output import (
    MoistureOutputConflict,
    MoistureOutputError,
    MoistureOutputIndeterminate,
    MoistureOutputLimits,
    MoistureOutputQuotaError,
)

__all__ = ["PROFILE_V2", "SCHEMA_FRAME_V2", "VegetationAttribution", "attribution_from_dict", "read_vegetation_frame", "write_vegetation_frame"]

SCHEMA_FRAME_V2 = "maple_syrup.floating_moisture_frame_vegetation/1"
PROFILE_V2 = "surface_following_floating_profile_vegetation"
ATTRIBUTION = g1.ATTRIBUTION
SOURCE_VALUE_MAX_CHARS = 128  # closed bound of every source-record string (session id, token, lineage, hashes ...): printable ASCII from a no-escape charset, so one character is one manifest byte
HELPER_MODULE_MAX_CHARS = 128
MAX_HELPERS = 4
MAX_SOURCE_ITEMS = 8  # the largest closed source record (plant_session)
MAX_RAW_SOURCE_KEYS = 16  # raw producer mappings (closed keys + free-text extras such as `note`) are counted, never copied, before any key is read
_SAFE_CHARS = frozenset(chr(c) for c in range(32, 127)) - {'"', "\\"}  # printable ASCII without the two JSON-escaped characters: one character is exactly one manifest byte
V_MANIFEST_EXTRA_BYTES = 4096  # analytic upper bound of the added `vegetation` block UNDER THE CLOSED BOUNDS ABOVE (the worst-case attribution is built and written in the tests: its real manifest <= the model)
_MODULE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_MAN_KEYS = g1._MAN_KEYS | {"vegetation"}
_VEG_HEX = ("recipe_id", "binding_sha256", "controls_sha256", "projection_sha256", "leaf_package_sha256", "mask_sha256", "surface_sha256", "current_column_sha256", "current_matrix_op_sha256", "current_law_id")
_VEG_KEYS = set(_VEG_HEX) | {"generation", "source", "external_helpers", "current_preferential_sha256", "current_property_generation"}
_SOURCE_KEYS = {"plant_session": ("kind", "token", "sequence", "pop_sha256", "geometry_sha256", "session_id", "lineage", "catalog_hash"), "initial_state": ("kind", "state_sha256", "catalog_hash")}


def _is_hex64(v: Any) -> bool:
    return g1._hex64(v)


@dataclass(frozen=True)
class VegetationAttribution:
    """Closed, immutable attribution record of ONE frame (see the module docstring). Validated structurally on construction; it is NOT a proof of acceptance."""

    generation: int
    recipe_id: str
    binding_sha256: str
    controls_sha256: str
    projection_sha256: str
    leaf_package_sha256: str
    mask_sha256: str
    surface_sha256: str
    source: tuple  # sorted ((key, value), ...) of the closed source identity
    external_helpers: tuple  # sorted ((module, sha256, bytes), ...)
    current_column_sha256: str
    current_matrix_op_sha256: str
    current_preferential_sha256: str | None
    current_property_generation: int
    current_law_id: str

    def __post_init__(self) -> None:
        for name in _VEG_HEX:
            if not _is_hex64(getattr(self, name)):
                raise MoistureOutputError(f"vegetation.{name} must be a 64-character lowercase hex digest")
        if self.current_preferential_sha256 is not None and not _is_hex64(self.current_preferential_sha256):
            raise MoistureOutputError("vegetation.current_preferential_sha256 must be a hex digest or None")
        for name in ("generation", "current_property_generation"):
            v = getattr(self, name)
            if type(v) is not int or not (name == "current_property_generation" or v >= 1) or not 0 <= v <= g0.MAX_COUNT:
                raise MoistureOutputError(f"vegetation.{name} must be a non-negative integer (generation >= 1), got {v!r}")
        # CHEAP structural gates FIRST (O(1) length / exact type, no copy, sort or set of caller data): the source is a tuple of at most 8 (key, value) pairs, the helpers a tuple of at most MAX_HELPERS rows
        if type(self.source) is not tuple or len(self.source) > MAX_SOURCE_ITEMS or any(type(kv) is not tuple or len(kv) != 2 or type(kv[0]) is not str for kv in self.source):
            raise MoistureOutputError(f"vegetation.source must be a tuple of at most {MAX_SOURCE_ITEMS} (key, value) pairs")
        if type(self.external_helpers) is not tuple or len(self.external_helpers) > MAX_HELPERS:
            raise MoistureOutputError(f"vegetation.external_helpers must be a tuple of at most {MAX_HELPERS} records")
        if any(type(h) is not tuple or len(h) != 3 for h in self.external_helpers):
            raise MoistureOutputError("vegetation.external_helpers must be (module, 'sha256:<hex>', bytes) records within the closed bounds")
        src = dict(self.source)
        kind = src.get("kind")
        if kind not in _SOURCE_KEYS or set(src) != set(_SOURCE_KEYS[kind]) or any(type(v) not in (str, int) or type(v) is bool for v in src.values()):
            raise MoistureOutputError("vegetation.source must be a closed plant_session / initial_state identity record")
        for k, v in src.items():
            if type(v) is str and (len(v) > SOURCE_VALUE_MAX_CHARS or not set(v) <= _SAFE_CHARS):
                raise MoistureOutputError(f"vegetation.source[{k!r}] must be at most {SOURCE_VALUE_MAX_CHARS} characters of the closed no-escape charset")
            if type(v) is int and not 0 <= v <= g0.MAX_COUNT:
                raise MoistureOutputError(f"vegetation.source[{k!r}] is out of range")
        for h in self.external_helpers:
            if (type(h[0]) is not str or not 0 < len(h[0]) <= HELPER_MODULE_MAX_CHARS or not set(h[0]) <= _SAFE_CHARS or type(h[1]) is not str
                    or not h[1].startswith("sha256:") or not _is_hex64(h[1][7:]) or type(h[2]) is not int or isinstance(h[2], bool) or not 0 <= h[2] <= 1 << 40):
                raise MoistureOutputError("vegetation.external_helpers must be (module, 'sha256:<hex>', bytes) records within the closed bounds")

    @property
    def order_key(self) -> tuple:
        """`(generation, recipe_id)`: along the sealed frames of ONE lineage the generation never decreases and a repeated generation keeps its recipe."""
        return (self.generation, self.recipe_id)

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in _VEG_KEYS if k not in ("source", "external_helpers")}
        d["source"] = dict(self.source)
        d["external_helpers"] = [{"module": m, "sha256": s, "bytes": b} for m, s, b in self.external_helpers]
        return d


def _raw_strict_int(v: Any, what: str) -> int:
    if type(v) is not int:
        raise MoistureOutputError(f"the vegetation attribution {what} must be a strict integer, got {type(v).__name__}")
    return v


def attribution_from_dict(d: dict) -> VegetationAttribution:
    """Build the record from the dict `vegetation_lifecycle.vegetation_attribution(...)` returns (only the closed stable + CURRENT keys are read; the `applied` provenance is ignored by design). Pure data adapter: no import of LEAF or the
    lifecycle. Raw gates (exact container types, counts, element types, strict integers) run BEFORE any dict / list / sort / tuple copy of caller data; value ranges and string bounds are then the constructor's (one validation)."""
    try:
        if type(d) is not dict:
            raise MoistureOutputError("the vegetation attribution must be a plain dict")
        raw_src, raw_h = d["source"], d["external_helpers"]
        if type(raw_src) is not dict or len(raw_src) > MAX_RAW_SOURCE_KEYS or type(raw_src.get("kind")) is not str or raw_src["kind"] not in _SOURCE_KEYS:
            raise MoistureOutputError("the vegetation attribution source must be a plain closed plant_session / initial_state record")
        if type(raw_h) not in (list, tuple) or len(raw_h) > MAX_HELPERS or any(type(h) is not dict or len(h) > 8 for h in raw_h):
            raise MoistureOutputError(f"the vegetation attribution external_helpers must be at most {MAX_HELPERS} plain records")
        generation = _raw_strict_int(d["generation"], "generation")
        pgen = _raw_strict_int(d["current_property_generation"], "current_property_generation")
        helpers = [(h["module"], h["sha256"], _raw_strict_int(h["bytes"], "helper bytes")) for h in raw_h]
        src = {k: raw_src[k] for k in _SOURCE_KEYS[raw_src["kind"]]}  # ONLY the closed identifying keys of the source record (a free-text `note` of the producer is not part of the identity)
        return VegetationAttribution(
            generation, d["recipe_id"], d["binding_sha256"], d["controls_sha256"], d["projection_sha256"], d["leaf_package_sha256"], d["mask_sha256"], d["surface_sha256"],
            tuple(sorted(src.items())), tuple(sorted(helpers)), d["current_column_sha256"], d["current_matrix_op_sha256"], d["current_preferential_sha256"], pgen, d["current_law_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MoistureOutputError(f"the vegetation attribution is incomplete or malformed: {exc!r}") from None


def check_order(previous: tuple | None, current: VegetationAttribution | None) -> None:
    """`previous` = `(generation, recipe_id)` of the latest sealed frame of the lineage (None: no frame yet, or an explicit null block = generation 0). A frame may not REGRESS the vegetation generation, may not drop to the null block after a
    generation, and a repeated generation must keep its recipe. Raises `MoistureOutputError`."""
    gen = 0 if current is None else current.generation
    pg = 0 if previous is None else previous[0]
    if gen < pg:
        raise MoistureOutputError(f"the vegetation generation {gen} regresses below the already sealed generation {pg} of this output (a stale or foreign biological lineage)")
    if previous is not None and gen == pg and (current is None or current.recipe_id != previous[1]):
        raise MoistureOutputError("the vegetation recipe differs from the one sealed for the same generation of this output (a foreign biological lineage)")


def _veg_block(v: VegetationAttribution | None) -> dict | None:
    return None if v is None else v.as_dict()


def _manifest_model_bytes(L: int, caller_bound: int) -> int:
    return g1._manifest_model_bytes(L, caller_bound) + V_MANIFEST_EXTRA_BYTES


def _build_manifest(meta: Any, ident: Any, veg: VegetationAttribution | None, caller: dict | None, edges: np.ndarray, L: int, N: int, shape_yx: tuple, area: float, n_active: int, hashes: dict) -> dict:
    lay = g1._layout(L, N)
    arrays = {}
    for key, name, shape, descr in (("water_m", g1.WATER, [L, N], "<f8"), ("depth_m", g1.DEPTH, [N], "<f8"), ("active", g1.ACTIVE, [N], "|b1")):
        arrays[key] = {"file": name, "dtype": descr, "shape": shape, "order": "C", "bytes": lay[name][2], "file_sha256": hashes[name]}
    man = {"schema": SCHEMA_FRAME_V2, "kind": "floating_soil_moisture_frame", "profile": PROFILE_V2, "not_a_restart": True, "attribution": ATTRIBUTION, "units": dict(g1.UNITS), "day": meta.day,
           "cadence_n": meta.cadence_n, "origin_ordinal": meta.origin_ordinal, "origin_date": meta.origin_date, "date": meta.date, "physical_t_s": float(meta.physical_t_s), "n_layers": L, "n_cells": N,
           "shape_yx": list(shape_yx), "cell_area_m2": float(area), "depth_edges_m": [float(e) for e in edges], "n_active": int(n_active), "identity": ident.as_dict(), "vegetation": _veg_block(veg),
           "caller_identity": caller, "arrays": arrays, "module_sha256": _MODULE_SHA}
    man["body_sha256"] = g0._body_sha(man)
    return man


def _check_veg_block(b: Any) -> None:
    if b is None:
        return
    b = g0._need(b, _VEG_KEYS, "vegetation")
    g0._chk(g0._is_int(b["generation"], 1) and g0._is_int(b["current_property_generation"], 0) and all(_is_hex64(b[k]) for k in _VEG_HEX)
            and (b["current_preferential_sha256"] is None or _is_hex64(b["current_preferential_sha256"])), "vegetation counters / digests")
    try:
        attribution_from_dict(b)
    except MoistureOutputError:
        raise MoistureOutputError("invalid vegetation attribution") from None


def _check_manifest(man: Any) -> dict:
    man = g0._need(man, _MAN_KEYS, "frame manifest")
    g0._chk(man["schema"] == SCHEMA_FRAME_V2 and man["kind"] == "floating_soil_moisture_frame" and man["profile"] == PROFILE_V2 and man["not_a_restart"] is True and man["attribution"] == ATTRIBUTION
            and man["units"] == g1.UNITS, "vegetation frame manifest header")
    g0._chk(g0._is_int(man["day"], 1, g0.MAX_DAY) and g0._is_int(man["cadence_n"], 1, 1 << 31) and man["day"] % man["cadence_n"] == 0 and g0._is_int(man["origin_ordinal"], 0)
            and g0._is_int(man["n_layers"], 1, g0.MAX_LAYERS) and g0._is_int(man["n_cells"], 1, g0.MAX_CELLS) and g0._is_int(man["n_active"], 0, man["n_cells"]), "frame manifest counts")
    g0._chk(g0._is_num(man["physical_t_s"]) and man["physical_t_s"] >= 0.0 and g0._is_num(man["cell_area_m2"]) and man["cell_area_m2"] > 0.0, "frame manifest numbers")
    sy = man["shape_yx"]
    g0._chk(type(sy) is list and len(sy) == 2 and all(g0._is_int(v, 1) for v in sy) and sy[0] * sy[1] == man["n_cells"], "frame shape")
    e = man["depth_edges_m"]
    g0._chk(type(e) is list and len(e) == man["n_layers"] + 1 and all(g0._is_num(x) for x in e) and e[0] == 0.0 and all(b > a for a, b in itertools.pairwise(e)), "frame depth edges")
    try:
        g1.FloatingFrameMeta(man["day"], man["cadence_n"], man["origin_ordinal"], man["physical_t_s"], man["origin_date"], man["date"])
        g1.FloatingFrameIdentity(**g0._need(man["identity"], set(g1._ID_FIELDS), "identity"))
    except TypeError:
        raise MoistureOutputError("invalid frame identity") from None
    _check_veg_block(man["vegetation"])
    if man["vegetation"] is not None:  # the writer's cross-field invariants: the CURRENT column / property generation are the identity's
        g0._chk(man["vegetation"]["current_column_sha256"] == man["identity"]["property_sha256"] and man["vegetation"]["current_property_generation"] == man["identity"]["property_generation"],
                "vegetation current ids differ from the identity")
    g0._chk(man["caller_identity"] is None or type(man["caller_identity"]) is dict, "caller_identity")
    g0._chk(g0._is_hex(man["module_sha256"]) and g0._is_hex(man["body_sha256"]) and man["body_sha256"] == g0._body_sha(man), "frame checksums")
    lay = g1._layout(man["n_layers"], man["n_cells"])
    arrs = g0._need(man["arrays"], {"water_m", "depth_m", "active"}, "arrays")
    for key, name, shape, descr in (("water_m", g1.WATER, [man["n_layers"], man["n_cells"]], "<f8"), ("depth_m", g1.DEPTH, [man["n_cells"]], "<f8"), ("active", g1.ACTIVE, [man["n_cells"]], "|b1")):
        a = g0._need(arrs[key], g1._ARRAY_KEYS, f"arrays.{key}")
        g0._chk(a["file"] == name and a["dtype"] == descr and a["shape"] == shape and a["order"] == "C" and a["bytes"] == lay[name][2] and g0._is_hex(a["file_sha256"]), f"arrays.{key}")
    return man


def write_vegetation_frame(
    root: str | Path, *, water_m: Any, depth_m: Any, active: Any, depth_edges_m: Any, cell_area_m2: float, shape_yx: tuple[int, int], meta: g1.FloatingFrameMeta, identity: g1.FloatingFrameIdentity,
    vegetation: VegetationAttribution | None, limits: MoistureOutputLimits, caller_identity: dict | None = None, host_available_bytes_fn: Callable[[], int] | None = None,
    disk_available_bytes_fn: Callable[[], int] | None = None,
) -> g1.FrameNotice:
    """Store ONE v2 frame (or verify an identical sealed one): the v1 order and guarantees (metadata checks, device guard, analytic O(L) host admission, exact manifest size check, quotas and disk BEFORE any transfer, exclusive
    reservation, positional chunk writes, fsync, streamed re-hash, manifest-LAST link seal, no-clobber replay with a positional compare) with the v2 manifest. `vegetation=None` is the explicit generation-0 contract."""
    if (type(meta) is not g1.FloatingFrameMeta or type(identity) is not g1.FloatingFrameIdentity or type(limits) is not MoistureOutputLimits
            or not (vegetation is None or type(vegetation) is VegetationAttribution)):
        raise MoistureOutputError("meta, identity, vegetation and limits must be FloatingFrameMeta, FloatingFrameIdentity, VegetationAttribution | None and MoistureOutputLimits")
    caller_bound = g1._caller_identity_bound(caller_identity)
    n_edges_layers = g1._edges_metadata(depth_edges_m)
    if type(cell_area_m2) not in (int, float) or not math.isfinite(cell_area_m2) or cell_area_m2 <= 0.0:
        raise MoistureOutputError("cell_area_m2 must be finite and positive")
    L, N, namespace = g1._check_arrays(water_m, depth_m, active, shape_yx)
    if n_edges_layers != L:
        raise MoistureOutputError("depth_edges_m must have L+1 entries")
    g1._guard_device(namespace, [water_m, depth_m, active])  # BEFORE any admission read, scan, scratch or transfer
    on_device = namespace == "cupy"
    model_len = _manifest_model_bytes(L, caller_bound)
    if model_len > limits.max_manifest_bytes:
        raise MoistureOutputQuotaError(f"the frame manifest model needs up to {model_len} bytes but max_manifest_bytes is {limits.max_manifest_bytes}")
    g0._admit_host(limits, limits.staging_bytes(model_len), host_available_bytes_fn)
    edges = g1._edges_values(depth_edges_m)
    caller = g0._check_caller_identity(caller_identity)
    lay = g1._layout(L, N)
    placeholder = {name: "0" * 64 for name in (g1.WATER, g1.DEPTH, g1.ACTIVE)}
    manifest_len = len(g0._canon(_build_manifest(meta, identity, vegetation, caller, edges, L, N, shape_yx, cell_area_m2, N, placeholder)))
    if manifest_len > limits.max_manifest_bytes:
        raise MoistureOutputQuotaError(f"the frame manifest needs {manifest_len} bytes but max_manifest_bytes is {limits.max_manifest_bytes}")
    frame_bytes = sum(v[2] for v in lay.values()) + manifest_len
    rootfd = g0._open_dir(root)
    name = g1.frame_name(meta.day)
    try:
        count, used, exists = g1._scan_root(rootfd, limits, name)
        if exists:
            return _verify_existing(rootfd, name, water_m, depth_m, active, edges, L, N, shape_yx, cell_area_m2, meta, identity, vegetation, caller, limits, lay, on_device)
        if count + 1 > limits.max_snapshots or used + frame_bytes > limits.max_total_bytes:
            raise MoistureOutputQuotaError("the frame would exceed max_snapshots or max_total_bytes; nothing was transferred or written")
        g0._check_disk(rootfd, frame_bytes, limits, disk_available_bytes_fn)
        try:
            os.mkdir(name, 0o700, dir_fd=rootfd)
        except FileExistsError:
            raise MoistureOutputConflict(f"{name} appeared concurrently; nothing was overwritten") from None
        made: list[str] = []
        dayfd = g0._open_dir(name, dir_fd=rootfd)
        path = str(Path(root) / name)
        try:
            try:  # before the seal: any failure removes ONLY this call's own unsealed files
                fds: dict[str, int] = {}
                try:
                    for fname in (g1.WATER, g1.DEPTH, g1.ACTIVE):
                        made.append(fname)
                        fd = os.open(fname, g0._FLAGS_NEW, 0o600, dir_fd=dayfd)
                        fds[fname] = fd
                        g1._pwrite_all(fd, 0, np.frombuffer(lay[fname][0], dtype=np.uint8))
                        os.ftruncate(fd, lay[fname][2])
                    n_active, pieces = g1._traverse(water_m, depth_m, active, edges, L, N, limits.chunk_bytes, lay, lambda nm, off, arr: g1._pwrite_all(fds[nm], off, arr), on_device)
                    for fd in fds.values():
                        os.fsync(fd)
                finally:
                    for fd in fds.values():
                        os.close(fd)
                hashes = {fname: g0._hash_stored(dayfd, fname, lay[fname][0], lay[fname][2], limits.chunk_bytes) for fname in (g1.WATER, g1.DEPTH, g1.ACTIVE)}
                raw = g0._canon(_build_manifest(meta, identity, vegetation, caller, edges, L, N, shape_yx, cell_area_m2, n_active, hashes))
                if len(raw) > limits.max_manifest_bytes:
                    raise MoistureOutputQuotaError("the frame manifest exceeds max_manifest_bytes")
                made.append(g1.MANIFEST + g0.TMP_SUFFIX)
                g0._write_new(dayfd, g1.MANIFEST + g0.TMP_SUFFIX, [raw])
                os.link(g1.MANIFEST + g0.TMP_SUFFIX, g1.MANIFEST, src_dir_fd=dayfd, dst_dir_fd=dayfd, follow_symlinks=False)
            except BaseException:
                g1._discard(rootfd, name, dayfd, made)
                raise
            try:  # the manifest is SEALED: nothing sealed is ever removed; any failure is an indeterminate publication
                os.unlink(g1.MANIFEST + g0.TMP_SUFFIX, dir_fd=dayfd)
                os.fsync(dayfd)
                os.fsync(rootfd)
            except BaseException as exc:
                raise MoistureOutputIndeterminate(f"the frame was PUBLISHED (manifest sealed) but a later durability step failed ({type(exc).__name__}); payload and seal are intact", path) from exc
        finally:
            os.close(dayfd)
        return g1.FrameNotice("saved", path, sum(v[2] for v in lay.values()) + len(raw), pieces)
    finally:
        os.close(rootfd)


def _verify_existing(rootfd, name, water, depth, active, edges, L, N, shape_yx, area, meta, identity, vegetation, caller, limits, lay, on_device) -> g1.FrameNotice:
    """Replay of an existing v2 frame: never writes or removes anything; the FIRST difference is a typed conflict. The sealed manifest must equal what THIS module would write for the supplied identity AND attribution."""
    dayfd = g0._open_dir(name, dir_fd=rootfd)
    try:
        if g0._lstat(dayfd, g1.MANIFEST) is None:
            raise MoistureOutputConflict(f"{name} exists UNSEALED (own earlier crash or foreign); it is not touched or completed")
        man = _check_manifest(g0._strict_loads(g0._read_small(dayfd, g1.MANIFEST, limits.max_manifest_bytes)))
        hashes = {a["file"]: a["file_sha256"] for a in man["arrays"].values()}
        if man != _build_manifest(meta, identity, vegetation, caller, edges, L, N, shape_yx, area, man["n_active"], hashes):
            raise MoistureOutputConflict(f"{name} is sealed with a different identity / vegetation attribution / metadata or by a different codec source; nothing was overwritten")
        for fname in (g1.WATER, g1.DEPTH, g1.ACTIVE):
            if g0._hash_stored(dayfd, fname, lay[fname][0], lay[fname][2], limits.chunk_bytes) != hashes[fname]:
                raise MoistureOutputConflict(f"{name}/{fname} does not match its sealed checksum (corrupt)")
        fds: dict[str, int] = {}
        buf = bytearray(8 * max(1, limits.chunk_bytes // g1.BYTES_PER_CELL))

        def compare(nm: str, off: int, arr: np.ndarray) -> None:
            if not g1._pread_equal(fds[nm], off, arr, buf):
                raise MoistureOutputConflict(f"{name}/{nm} is sealed with different array bytes; nothing was overwritten")

        try:
            for fname in (g1.WATER, g1.DEPTH, g1.ACTIVE):
                fds[fname] = g0._open_regular(dayfd, fname, size=lay[fname][2])
            n_active, pieces = g1._traverse(water, depth, active, edges, L, N, limits.chunk_bytes, lay, compare, on_device)
        finally:
            for fd in fds.values():
                os.close(fd)
        if n_active != man["n_active"]:
            raise MoistureOutputConflict(f"{name} is sealed with a different active count; nothing was overwritten")
        return g1.FrameNotice("verified_existing", name, 0, pieces)
    finally:
        os.close(dayfd)


def read_vegetation_frame(path: str | Path, *, max_return_bytes: int) -> g1.FloatingFrame:
    """Reconstruct ONE sealed v2 frame as the v1 `FloatingFrame` (arrays, `theta`, thickness) whose `manifest['vegetation']` is the attribution block (or None). No model, LEAF or lifecycle import; a v1 frame is refused here
    and a v2 frame is refused by the v1 reader."""
    cap = g0._count(max_return_bytes, "max_return_bytes", minimum=1)
    fd = g0._open_dir(path)
    try:
        man = _check_manifest(g0._strict_loads(g0._read_small(fd, g1.MANIFEST, g0.MAX_SMALL_JSON_BYTES)))
        L, N = man["n_layers"], man["n_cells"]
        need = 8 * L * N + 8 * N + N + 8 * (L + 1)
        if need > cap:
            raise MoistureOutputQuotaError(f"the frame returns {need} bytes but max_return_bytes is {cap}; nothing was read")
        lay = g1._layout(L, N)
        out = {}
        for key, fname, dtype, shape in (("water_m", g1.WATER, np.float64, (L, N)), ("depth_m", g1.DEPTH, np.float64, (N,)), ("active", g1.ACTIVE, np.bool_, (N,))):
            a = man["arrays"][key]
            head, size = lay[fname][0], lay[fname][2]
            f = g0._open_regular(fd, fname, size=size)
            try:
                if os.pread(f, len(head), 0) != head:
                    raise MoistureOutputError(f"{fname} header mismatch")
                os.lseek(f, len(head), os.SEEK_SET)
                arr = np.empty(shape, dtype=dtype)
                h = hashlib.sha256(head)
                g0._read_into(f, memoryview(arr).cast("B"), h)
            finally:
                os.close(f)
            if h.hexdigest() != a["file_sha256"]:
                raise MoistureOutputError(f"{fname} does not match its manifest checksum (corrupt frame)")
            out[key] = arr
        edges = np.array(man["depth_edges_m"], dtype=np.float64)
        if int(np.count_nonzero(out["active"])) != man["n_active"]:
            raise MoistureOutputError("the stored mask disagrees with n_active")
        step = max(1, g0.FINITE_BLOCK)
        for c0 in range(0, N, step):
            c1 = min(N, c0 + step)
            m = out["active"][c0:c1]
            Da = g1._validate_depth(out["depth_m"][c0:c1], m)
            for l in range(L):
                g1._validate_row(out["water_m"][l, c0:c1], m, Da, edges, l)
        for arr in out.values():
            arr.setflags(write=False)
        edges.setflags(write=False)
        return g1.FloatingFrame(man, out["water_m"], out["depth_m"], out["active"], edges, float(man["cell_area_m2"]))
    finally:
        os.close(fd)
