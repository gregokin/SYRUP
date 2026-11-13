"""Phase 2 G1: bounded stream storage/reconstruction of ONE floating-profile soil-moisture DIAGNOSTIC frame.

Scope
-----
A pure codec: `write_floating_frame` stores, `read_floating_frame` reconstructs. It is NOT a cadence, scheduler, accepted-state
authority or restart: it never counts days, never builds a `DailyCursor` and decides nothing about WHEN to save. The later genuine
floating daily owner (E3/G2) supplies the arrays and the identity record at its accepted final post-T day close. The supplied
identity/generation fields are ATTRIBUTION metadata recorded verbatim (`attribution: "caller_supplied_not_proof"`); they do not
prove that the state was accepted or that any parent is genuine. Full property maps, beds and histories are not stored: a frame
holds exactly what reconstructs moisture without any model or source.

Frame content (schema `maple_syrup.floating_moisture_frame/1`, profile `surface_following_floating_profile`)
------------------------------------------------------------------------------------------------------------
`water_m (L, N)` float64 (m water-equivalent depth of each fixed nominal layer slab), `depth_m (N,)` float64 (the actual canonical
soil depth D supplied by the owner), `active (N,)` bool, the nominal depth edges `(L+1,)` (m, from 0, strictly increasing), shape
`(ny, nx)` / cell area, the completed nominal day, cadence N, origin ordinal/date and optional ISO date, the PHYSICAL elapsed time
`physical_t_s` (separate from the nominal day), the stable run/source/lineage identity and the CURRENT property/bed/depth
generations plus parent/state digests (the property hash may change from frame to frame). Each frame is self-contained:
`<root>/frame_<day:012d>/{water_m.npy, depth_m.npy, active.npy, manifest.json}`; the manifest carries SHA-256 of all three arrays.

Reconstruction (no physics beyond geometry): `h_l = max(0, min(D, edge_{l+1}) - edge_l)` is the porous thickness of layer l and
`theta = W / h` where `h > 0`; where `h == 0` (rock slot) theta is DEFINED as 0 in this diagnostic and on ACTIVE cells the stored
water must be exactly 0 (otherwise the write/read is refused). Active cells need finite `D >= 0` and finite `W >= 0`. INACTIVE
cells' raw W/D/mask values are preserved bit for bit and no physical bound is imposed on them.

Storage and bounds
------------------
Arrays may be NumPy or CuPy. Data are traversed in cell chunks of `c = max(1, chunk_bytes // 48)` cells (declared 48 B/cell: W piece
8, D 8, mask 1, thickness chain 24, flags 3-ish); per chunk one D piece, one mask piece and L water pieces `W[l, c0:c1]` are
copied with MAPLE's counted `to_host` (host arrays are used directly) and written with positional writes into preallocated `.npy`
files, so staging is O(chunk_bytes + L metadata): no full host mirror, no `L x N` boolean, no stacked frames. Transfers happen
only inside a call (only due caller writes). Admission BEFORE any scratch, transfer or file write: argument/metadata checks, the
device guard (a device array must live on the CURRENT device, no switching), the host staging model vs the explicit budget and a
truthful reading, the frame/quota scan (count, total bytes) and the disk reading with reserve. Truthful byte counts are exact
(`.npy` header + data + manifest), the manifest size is bounded by `max_manifest_bytes`.

Publication / replay / failure (reuses the G0 descriptor-relative, no-follow, no-clobber primitives and `MoistureOutputLimits`)
------------------------------------------------------------------------------------------------------------------------------
Exclusive `mkdir` reservation of `frame_<day>`, exclusive-create files, fsync, streamed re-hash of the stored files, manifest
temporary written, then the manifest-LAST `os.link` seal. A failure before the seal removes only this call's own unsealed
files and reservation. A reservation or sealed frame that already exists is never overwritten or removed: if it is SEALED it is
verified against the exact current identity, metadata, stored checksums and the live bytes (positional compare, same chunk bound);
an identical frame returns `verified_existing`, anything else raises `MoistureOutputConflict`; an existing UNSEALED reservation
(own earlier crash or foreign) is refused untouched. A failure at/after the seal raises `MoistureOutputIndeterminate`: the sealed
frame and its bytes are intact.

Reader
------
`read_floating_frame(path, max_return_bytes=...)`: strict JSON, exact keys/types, header/size/checksum verification, the total of
the returned arrays (`8LN + 8N + N + 8(L+1)`) checked against `max_return_bytes` BEFORE any array is read, arrays read straight into
their final read-only buffers, value validation in bounded blocks. `FloatingFrame.theta(max_bytes=...)` allocates its own
`16 L N` bytes (admitted against `max_bytes`).

Not claimed: cadence, accepted-state proof, restart, GPU behaviour (author tests use host doubles; Root runs hardware).
"""

from __future__ import annotations

import datetime
import hashlib
import itertools
import math
import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup import soil_moisture_output as g0
from maple_syrup.soil_moisture_output import (
    MoistureOutputConflict,
    MoistureOutputError,
    MoistureOutputIndeterminate,
    MoistureOutputLimits,
    MoistureOutputQuotaError,
)

__all__ = [
    "SCHEMA_FRAME",
    "FloatingFrame",
    "FloatingFrameIdentity",
    "FloatingFrameMeta",
    "FrameNotice",
    "frame_name",
    "read_floating_frame",
    "write_floating_frame",
]

SCHEMA_FRAME = "maple_syrup.floating_moisture_frame/1"
PROFILE = "surface_following_floating_profile"
ATTRIBUTION = "caller_supplied_not_proof"
WATER, DEPTH, ACTIVE, MANIFEST = "water_m.npy", "depth_m.npy", "active.npy", "manifest.json"
BYTES_PER_CELL = 64  # declared live host bytes per cell of one chunk (D 8, mask 1, active-D 8, one water row 8, its active copy 8, thickness 8,
# ~4 bool/flag bytes, compare buffer 8, slack); calibrated by the tracemalloc controls in the tests
MANIFEST_FIXED_BYTES = 2560  # analytic upper bound of the fixed manifest keys/digests (checked against the real manifest in the tests)
MANIFEST_PER_EDGE_BYTES = 26  # one edge float (<= 24 characters) plus its comma
UNITS = {"water_m": "m water-equivalent depth", "depth_m": "m", "depth_edges_m": "m", "cell_area_m2": "m2", "physical_t_s": "s"}
_FRAME_RE = re.compile(r"^frame_(\d{12})$")
_HEX = set("0123456789abcdef")
_MAN_KEYS = {"schema", "kind", "profile", "not_a_restart", "attribution", "units", "day", "cadence_n", "origin_ordinal", "origin_date",
             "date", "physical_t_s", "n_layers", "n_cells", "shape_yx", "cell_area_m2", "depth_edges_m", "n_active", "identity",
             "caller_identity", "arrays", "module_sha256", "body_sha256"}
_ARRAY_KEYS = {"file", "dtype", "shape", "order", "bytes", "file_sha256"}
_ID_FIELDS = ("run_digest", "lineage_sha256", "source_sha256", "property_generation", "property_sha256", "bed_generation",
              "depth_generation", "parent_digest", "state_digest", "state_revision")
_MODULE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def frame_name(day: int) -> str:
    return f"frame_{int(day):012d}"


def _to_host(array: Any) -> np.ndarray:  # indirection so tests can count the (MAPLE-counted) device transfers
    from maple.core.backend import to_host

    return to_host(array)


def _hex64(v: Any) -> bool:
    return type(v) is str and len(v) == 64 and set(v) <= _HEX


# --- records ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FloatingFrameIdentity:
    """Attribution metadata of the later genuine owner (recorded verbatim; NOT a proof of acceptance). The property hash and the
    generations describe the CURRENT frame and may change between frames."""

    run_digest: str
    lineage_sha256: str
    source_sha256: str
    property_generation: int
    property_sha256: str
    bed_generation: int
    depth_generation: int
    parent_digest: str
    state_digest: str
    state_revision: int

    def __post_init__(self) -> None:
        for name in ("run_digest", "lineage_sha256", "source_sha256", "property_sha256", "parent_digest", "state_digest"):
            if not _hex64(getattr(self, name)):
                raise MoistureOutputError(f"identity.{name} must be a 64-character lowercase hex digest")
        for name in ("property_generation", "bed_generation", "depth_generation", "state_revision"):
            v = getattr(self, name)
            if type(v) is not int or not 0 <= v <= g0.MAX_COUNT:
                raise MoistureOutputError(f"identity.{name} must be a non-negative integer, got {v!r}")

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in _ID_FIELDS}


@dataclass(frozen=True)
class FloatingFrameMeta:
    """Nominal-day metadata: completed nominal `day` (a multiple of `cadence_n`), cadence, origin ordinal/date, optional ISO `date`
    (both dates or neither; `date == origin_date + day days`), and the PHYSICAL elapsed `physical_t_s` (separate from the day)."""

    day: int
    cadence_n: int
    origin_ordinal: int
    physical_t_s: float
    origin_date: str | None = None
    date: str | None = None

    def __post_init__(self) -> None:
        if type(self.day) is not int or not 1 <= self.day <= g0.MAX_DAY:
            raise MoistureOutputError(f"day must be an integer in [1, {g0.MAX_DAY}], got {self.day!r}")
        if type(self.cadence_n) is not int or not 1 <= self.cadence_n <= 1 << 31:
            raise MoistureOutputError(f"cadence_n must be a positive integer, got {self.cadence_n!r}")
        if self.day % self.cadence_n:
            raise MoistureOutputError(f"day {self.day} is not a multiple of cadence_n {self.cadence_n}")
        if type(self.origin_ordinal) is not int or not 0 <= self.origin_ordinal <= g0.MAX_COUNT:
            raise MoistureOutputError("origin_ordinal must be a non-negative integer")
        if type(self.physical_t_s) not in (int, float) or not math.isfinite(self.physical_t_s) or self.physical_t_s < 0.0:
            raise MoistureOutputError("physical_t_s must be a finite number >= 0")
        if (self.origin_date is None) != (self.date is None):
            raise MoistureOutputError("origin_date and date must both be given or both be None")
        if self.date is not None:
            try:
                o, d = datetime.date.fromisoformat(self.origin_date), datetime.date.fromisoformat(self.date)
                expected = o + datetime.timedelta(days=self.day)
            except (TypeError, ValueError, OverflowError):
                raise MoistureOutputError("origin_date/date must be ISO calendar dates and date = origin_date + day days must be "
                                          "representable") from None
            if d != expected:
                raise MoistureOutputError("date must equal origin_date + day days")


@dataclass(frozen=True)
class FrameNotice:
    action: str  # saved | verified_existing
    path: str
    bytes_written: int
    device_chunks: int  # counted `to_host` pieces of THIS call that came from a device array


@dataclass(frozen=True)
class FloatingFrame:
    manifest: dict
    water_m: np.ndarray  # (L, N) float64, read-only
    depth_m: np.ndarray  # (N,) float64, read-only
    active: np.ndarray  # (N,) bool, read-only
    depth_edges_m: np.ndarray  # (L+1,) float64, read-only
    cell_area_m2: float

    def thickness_m(self, *, max_bytes: int) -> np.ndarray:
        """`(L, N)` porous thickness `max(0, min(D, hi) - lo)`: ONE `8 L N` output computed row by row in place (no temporaries);
        `max_bytes` is checked (typed) before the allocation."""
        L, N = self.water_m.shape
        _admit_bytes(max_bytes, 8 * L * N, "thickness")
        return self._thickness_into(np.empty((L, N), dtype=np.float64))

    def _thickness_into(self, out: np.ndarray) -> np.ndarray:
        for l in range(out.shape[0]):
            row = out[l]
            np.minimum(self.depth_m, float(self.depth_edges_m[l + 1]), out=row)
            row -= float(self.depth_edges_m[l])
            np.maximum(row, 0.0, out=row)
        return out

    def theta(self, *, max_bytes: int) -> np.ndarray:
        """`W / h` where `h > 0`, 0 in rock slots (documented definition). Peak = ONE `8 L N` output (thickness first, then divided in
        place) plus one `N`-byte boolean row; `max_bytes` must admit `8 L N + N` and is checked before any allocation."""
        L, N = self.water_m.shape
        _admit_bytes(max_bytes, 8 * L * N + N, "theta")
        out = self._thickness_into(np.empty((L, N), dtype=np.float64))
        for l in range(L):
            np.divide(self.water_m[l], out[l], out=out[l], where=out[l] > 0.0)  # h == 0 slots keep their 0
        return out


def _admit_bytes(max_bytes: Any, need: int, what: str) -> None:
    if type(max_bytes) is not int and not isinstance(max_bytes, np.integer) or isinstance(max_bytes, bool | np.bool_) or int(max_bytes) < 0:
        raise MoistureOutputError(f"max_bytes must be a non-negative integer, got {max_bytes!r}")
    if need > int(max_bytes):
        raise MoistureOutputQuotaError(f"{what} needs {need} bytes but max_bytes is {int(max_bytes)}; nothing was allocated")


# --- validation helpers ------------------------------------------------------------------------------------------------------
def _edges_metadata(edges: Any) -> int:
    """Cheap metadata only (type, dtype, rank, size): no value scan, no allocation. Returns L."""
    if type(edges) is not np.ndarray or edges.dtype != np.float64 or edges.ndim != 1 or edges.size < 2 or edges.size - 1 > g0.MAX_LAYERS:
        raise MoistureOutputError("depth_edges_m must be a host float64 array of shape (L+1,) with L >= 1")
    return int(edges.size) - 1


def _edges_values(edges: np.ndarray) -> np.ndarray:
    """Value checks and the owned read-only copy (O(L) bytes, admitted analytically BEFORE this runs)."""
    if not np.all(np.isfinite(edges)) or edges[0] != 0.0 or not np.all(edges[1:] > edges[:-1]):
        raise MoistureOutputError("depth_edges_m must start at exactly 0 and be finite and strictly increasing")
    out = edges.copy()
    out.setflags(write=False)
    return out


def _caller_identity_bound(ci: Any) -> int:
    """Upper bound of the canonical JSON bytes of `caller_identity` computed from lengths only (no JSON is built); hostile large or
    non-str content is refused here. ASCII escaping can expand a character at most 6x."""
    if ci is None:
        return 4
    if type(ci) is not dict or len(ci) > 256:
        raise MoistureOutputError("caller_identity must be a dict of at most 256 str to str entries")
    chars = 0
    for k, v in ci.items():
        if type(k) is not str or type(v) is not str:
            raise MoistureOutputError("caller_identity must be a dict of str to str")
        chars += len(k) + len(v)
        if 6 * chars + 8 * len(ci) + 2 > g0.MAX_CALLER_IDENTITY_BYTES * 6:
            raise MoistureOutputError(f"caller_identity exceeds {g0.MAX_CALLER_IDENTITY_BYTES} bytes")
    return 6 * chars + 8 * len(ci) + 2


def _manifest_model_bytes(L: int, caller_bound: int) -> int:
    """Analytic O(L) upper bound of the manifest JSON size, used for admission before any JSON, list or copy is built."""
    return MANIFEST_FIXED_BYTES + MANIFEST_PER_EDGE_BYTES * (L + 1) + caller_bound


def _check_arrays(water: Any, depth: Any, active: Any, shape_yx: Any) -> tuple[int, int, str]:
    from maple.core.backend import (
        MixedArrayNamespaceError,
        array_namespace,
        is_array,
        namespace_name,
    )

    for name, a in (("water_m", water), ("depth_m", depth), ("active", active)):
        if not is_array(a):
            raise MoistureOutputError(f"{name} must be an array")
    if water.ndim != 2 or depth.ndim != 1 or active.ndim != 1:
        raise MoistureOutputError("water_m must be (L, N), depth_m and active (N,)")
    if water.dtype != np.float64 or depth.dtype != np.float64 or active.dtype != np.bool_:
        raise MoistureOutputError("water_m/depth_m must be float64 and active bool")
    L, N = int(water.shape[0]), int(water.shape[1])
    if L < 1 or N < 1 or L > g0.MAX_LAYERS or N > g0.MAX_CELLS or depth.shape[0] != N or active.shape[0] != N:
        raise MoistureOutputError("array shapes disagree or are outside the supported range")
    if not (water.flags.c_contiguous and depth.flags.c_contiguous and active.flags.c_contiguous):
        raise MoistureOutputError("arrays must be C-contiguous")
    if (type(shape_yx) is not tuple or len(shape_yx) != 2 or any(type(v) is not int or v < 1 for v in shape_yx)
            or shape_yx[0] * shape_yx[1] != N):
        raise MoistureOutputError("shape_yx must be a (ny, nx) tuple of positive ints with ny*nx == N")
    try:
        ns = array_namespace(water, depth, active)
    except MixedArrayNamespaceError as exc:  # mixed host/device namespaces are one refusal
        raise MoistureOutputError(f"arrays must share one namespace: {exc}") from None
    return L, N, namespace_name(ns)


def _device_ids(arrays: list) -> set[int]:
    return {int(a.device.id) for a in arrays}


def _current_device(arrays: list) -> int:
    from maple.core.backend import array_namespace

    return int(array_namespace(arrays[0]).cuda.runtime.getDevice())


def _guard_device(namespace: str, arrays: list) -> int | None:
    """CuPy arrays must share ONE device that is the CURRENT device; checked before admission reads/transfers; never switches."""
    if namespace != "cupy":
        return None
    ids = _device_ids(arrays)
    if len(ids) != 1:
        raise MoistureOutputError(f"arrays live on different devices {sorted(ids)}; no implicit cross-device work")
    (dev,) = ids
    cur = _current_device(arrays)
    if cur != dev:
        raise MoistureOutputError(f"arrays live on device {dev} but the current device is {cur}; the device is never switched")
    return dev


def _npy_head(shape: tuple[int, ...], descr: str) -> bytes:
    return g0._npy_header(shape, descr)


def _layout(L: int, N: int) -> dict[str, tuple[bytes, int, int]]:
    """name -> (npy header, data offset, total file bytes)."""
    out = {}
    for name, shape, descr, item in ((WATER, (L, N), "<f8", 8), (DEPTH, (N,), "<f8", 8), (ACTIVE, (N,), "|b1", 1)):
        head = _npy_head(shape, descr)
        out[name] = (head, len(head), len(head) + item * int(np.prod(shape, dtype=np.int64)))
    return out


def _build_manifest(meta: FloatingFrameMeta, ident: FloatingFrameIdentity, caller: dict | None, edges: np.ndarray, L: int, N: int,
                    shape_yx: tuple, area: float, n_active: int, hashes: dict[str, str]) -> dict:
    lay = _layout(L, N)
    arrays = {}
    for key, name, shape, descr in (("water_m", WATER, [L, N], "<f8"), ("depth_m", DEPTH, [N], "<f8"), ("active", ACTIVE, [N], "|b1")):
        arrays[key] = {"file": name, "dtype": descr, "shape": shape, "order": "C", "bytes": lay[name][2], "file_sha256": hashes[name]}
    man = {"schema": SCHEMA_FRAME, "kind": "floating_soil_moisture_frame", "profile": PROFILE, "not_a_restart": True,
           "attribution": ATTRIBUTION, "units": dict(UNITS), "day": meta.day, "cadence_n": meta.cadence_n,
           "origin_ordinal": meta.origin_ordinal, "origin_date": meta.origin_date, "date": meta.date,
           "physical_t_s": float(meta.physical_t_s), "n_layers": L, "n_cells": N, "shape_yx": list(shape_yx),
           "cell_area_m2": float(area), "depth_edges_m": [float(e) for e in edges], "n_active": int(n_active),
           "identity": ident.as_dict(), "caller_identity": caller, "arrays": arrays, "module_sha256": _MODULE_SHA}
    man["body_sha256"] = g0._body_sha(man)
    return man


def _check_manifest(man: Any) -> dict:
    man = g0._need(man, _MAN_KEYS, "frame manifest")
    g0._chk(man["schema"] == SCHEMA_FRAME and man["kind"] == "floating_soil_moisture_frame" and man["profile"] == PROFILE
            and man["not_a_restart"] is True and man["attribution"] == ATTRIBUTION and man["units"] == UNITS, "frame manifest header")
    g0._chk(g0._is_int(man["day"], 1, g0.MAX_DAY) and g0._is_int(man["cadence_n"], 1, 1 << 31) and man["day"] % man["cadence_n"] == 0
            and g0._is_int(man["origin_ordinal"], 0) and g0._is_int(man["n_layers"], 1, g0.MAX_LAYERS)
            and g0._is_int(man["n_cells"], 1, g0.MAX_CELLS) and g0._is_int(man["n_active"], 0, man["n_cells"]), "frame manifest counts")
    g0._chk(g0._is_num(man["physical_t_s"]) and man["physical_t_s"] >= 0.0 and g0._is_num(man["cell_area_m2"]) and man["cell_area_m2"] > 0.0,
            "frame manifest numbers")
    sy = man["shape_yx"]
    g0._chk(type(sy) is list and len(sy) == 2 and all(g0._is_int(v, 1) for v in sy) and sy[0] * sy[1] == man["n_cells"], "frame shape")
    e = man["depth_edges_m"]
    g0._chk(type(e) is list and len(e) == man["n_layers"] + 1 and all(g0._is_num(x) for x in e) and e[0] == 0.0
            and all(b > a for a, b in itertools.pairwise(e)), "frame depth edges")
    try:
        FloatingFrameMeta(man["day"], man["cadence_n"], man["origin_ordinal"], man["physical_t_s"], man["origin_date"], man["date"])
        FloatingFrameIdentity(**g0._need(man["identity"], set(_ID_FIELDS), "identity"))
    except TypeError:
        raise MoistureOutputError("invalid frame identity") from None
    g0._chk(man["caller_identity"] is None or type(man["caller_identity"]) is dict, "caller_identity")
    g0._chk(g0._is_hex(man["module_sha256"]) and g0._is_hex(man["body_sha256"]) and man["body_sha256"] == g0._body_sha(man), "frame checksums")
    lay = _layout(man["n_layers"], man["n_cells"])
    arrs = g0._need(man["arrays"], {"water_m", "depth_m", "active"}, "arrays")
    for key, name, shape, descr in (("water_m", WATER, [man["n_layers"], man["n_cells"]], "<f8"), ("depth_m", DEPTH, [man["n_cells"]], "<f8"),
                                    ("active", ACTIVE, [man["n_cells"]], "|b1")):
        a = g0._need(arrs[key], _ARRAY_KEYS, f"arrays.{key}")
        g0._chk(a["file"] == name and a["dtype"] == descr and a["shape"] == shape and a["order"] == "C" and a["bytes"] == lay[name][2]
                and g0._is_hex(a["file_sha256"]), f"arrays.{key}")
    return man


# --- traversal (validation + sink) --------------------------------------------------------------------------------------------
def _validate_depth(D: np.ndarray, m: np.ndarray) -> np.ndarray | None:
    """Active-cell depth check of one chunk; returns the active-depth copy (<= chunk, retained for the row checks) or None."""
    if not bool(m.any()):
        return None
    Da = D[m]
    if not (np.all(np.isfinite(Da)) and np.all(Da >= 0.0)):
        raise MoistureOutputError("active cells need finite depth_m >= 0")
    return Da


def _validate_row(w: np.ndarray, m: np.ndarray, Da: np.ndarray | None, edges: np.ndarray, l: int) -> None:
    """ONE water row of one chunk: active W finite >= 0 and exactly 0 where the porous thickness is 0 (computed in place)."""
    if Da is None:
        return
    wa = w[m]
    if not (np.all(np.isfinite(wa)) and np.all(wa >= 0.0)):
        raise MoistureOutputError("active cells need finite water_m >= 0")
    h = np.minimum(Da, float(edges[l + 1]))
    h -= float(edges[l])
    np.maximum(h, 0.0, out=h)
    if np.any((wa != 0.0) & (h == 0.0)):
        raise MoistureOutputError(f"layer {l}: an active cell holds water in a rock slot (porous thickness 0)")


def _host_piece(a: Any, expect: type) -> np.ndarray:
    h = np.ascontiguousarray(_to_host(a))
    if h.dtype != expect:
        raise MoistureOutputError("a transferred piece has an unexpected dtype")
    return h


def _traverse(water: Any, depth: Any, active: Any, edges: np.ndarray, L: int, N: int, chunk_bytes: int, lay: dict,
              sink: Callable[[str, int, np.ndarray], None], on_device: bool) -> tuple[int, int]:
    """Cell-chunk-outer traversal. Per chunk: one D piece and one mask piece are kept; the L water rows are then downloaded, validated
    and handed to `sink(name, byte_offset, array)` ONE AT A TIME (each row is released before the next is requested), so at most
    one water row, the active-depth copy and the row-validation temporaries are live. Returns (n_active, counted device pieces)."""
    c = max(1, chunk_bytes // BYTES_PER_CELL)
    n_active = pieces = 0
    for c0 in range(0, N, c):
        c1 = min(N, c0 + c)
        D = _host_piece(depth[c0:c1], np.float64)
        m = _host_piece(active[c0:c1], np.bool_)
        pieces += 2 if on_device else 0
        Da = _validate_depth(D, m)
        n_active += int(np.count_nonzero(m))
        sink(DEPTH, lay[DEPTH][1] + 8 * c0, D)
        sink(ACTIVE, lay[ACTIVE][1] + c0, m)
        for l in range(L):
            w = _host_piece(water[l, c0:c1], np.float64)
            pieces += 1 if on_device else 0
            _validate_row(w, m, Da, edges, l)
            sink(WATER, lay[WATER][1] + 8 * (l * N + c0), w)
            del w
        del D, m, Da
    return n_active, pieces


def _pwrite_all(fd: int, offset: int, arr: np.ndarray) -> None:
    view = memoryview(arr).cast("B")
    pos = 0
    while pos < len(view):
        n = os.pwrite(fd, view[pos:], offset + pos)
        if n <= 0:
            raise MoistureOutputError("a positional write made no progress")
        pos += n


def _pread_equal(fd: int, offset: int, arr: np.ndarray, buf: bytearray) -> bool:
    """Compare `arr` with the stored bytes at `offset` through ONE reusable buffer (no per-piece copies)."""
    view = memoryview(arr).cast("B")
    pos = 0
    while pos < len(view):
        n = os.preadv(fd, [memoryview(buf)[:len(view) - pos]], offset + pos)
        if n <= 0 or memoryview(buf)[:n] != view[pos:pos + n]:
            return False
        pos += n
    return True


# --- admission --------------------------------------------------------------------------------------------------------------
def _scan_root(rootfd: int, limits: MoistureOutputLimits, day_name: str) -> tuple[int, int, bool]:
    """(frame count, bytes used by `frame_*` directories, whether `day_name` exists); only metadata is read, no file content."""
    count = used = 0
    exists = False
    with os.scandir(rootfd) as it:
        for entry in it:
            if not _FRAME_RE.match(entry.name):
                continue
            count += 1
            if count > limits.max_snapshots + 1:
                raise MoistureOutputQuotaError("the root holds more frame directories than max_snapshots allows")
            exists = exists or entry.name == day_name
            info = g0._lstat(rootfd, entry.name)
            if info is None or not stat.S_ISDIR(info.st_mode):
                raise MoistureOutputError(f"{entry.name} is not a directory")
            dfd = g0._open_dir(entry.name, dir_fd=rootfd)
            try:
                with os.scandir(dfd) as inner:
                    for n, member in enumerate(inner):
                        if n > 8:
                            raise MoistureOutputError(f"{entry.name} has too many entries")
                        mi = g0._lstat(dfd, member.name)
                        if mi is None or not stat.S_ISREG(mi.st_mode):
                            raise MoistureOutputError(f"{entry.name}/{member.name} is not a regular file")
                        used += mi.st_size
            finally:
                os.close(dfd)
    return count, used, exists


def _discard(rootfd: int, name: str, dayfd: int, made: list[str]) -> None:
    g0._discard(rootfd, name, dayfd, made)


# --- write -----------------------------------------------------------------------------------------------------------------------
def write_floating_frame(
    root: str | Path, *, water_m: Any, depth_m: Any, active: Any, depth_edges_m: Any, cell_area_m2: float, shape_yx: tuple[int, int],
    meta: FloatingFrameMeta, identity: FloatingFrameIdentity, limits: MoistureOutputLimits, caller_identity: dict | None = None,
    host_available_bytes_fn: Callable[[], int] | None = None, disk_available_bytes_fn: Callable[[], int] | None = None,
) -> FrameNotice:
    """Store ONE frame (or verify an identical sealed one). See the module docstring; the caller owns the arrays, which are never
    mutated. Order: type/metadata checks (no value scans) -> device guard -> analytic O(L) host admission -> edge values, owned
    copy, caller canon, exact manifest size check -> root scan/quotas/disk -> reservation, data, seal. `FrameNotice.bytes_written` is
    EXACT (`.npy` files + the real manifest); the pre-I/O quota estimate uses `n_active = N` and is therefore never smaller."""
    if type(meta) is not FloatingFrameMeta or type(identity) is not FloatingFrameIdentity or type(limits) is not MoistureOutputLimits:
        raise MoistureOutputError("meta, identity and limits must be FloatingFrameMeta, FloatingFrameIdentity and MoistureOutputLimits")
    caller_bound = _caller_identity_bound(caller_identity)
    n_edges_layers = _edges_metadata(depth_edges_m)
    if type(cell_area_m2) not in (int, float) or not math.isfinite(cell_area_m2) or cell_area_m2 <= 0.0:
        raise MoistureOutputError("cell_area_m2 must be finite and positive")
    L, N, namespace = _check_arrays(water_m, depth_m, active, shape_yx)
    if n_edges_layers != L:
        raise MoistureOutputError("depth_edges_m must have L+1 entries")
    _guard_device(namespace, [water_m, depth_m, active])  # BEFORE any admission read, scan, scratch or transfer
    on_device = namespace == "cupy"
    model_len = _manifest_model_bytes(L, caller_bound)
    if model_len > limits.max_manifest_bytes:
        raise MoistureOutputQuotaError(f"the frame manifest model needs up to {model_len} bytes but max_manifest_bytes is {limits.max_manifest_bytes}")
    g0._admit_host(limits, limits.staging_bytes(model_len), host_available_bytes_fn)
    edges = _edges_values(depth_edges_m)
    caller = g0._check_caller_identity(caller_identity)
    lay = _layout(L, N)
    placeholder = {name: "0" * 64 for name in (WATER, DEPTH, ACTIVE)}
    manifest_len = len(g0._canon(_build_manifest(meta, identity, caller, edges, L, N, shape_yx, cell_area_m2, N, placeholder)))
    if manifest_len > limits.max_manifest_bytes:
        raise MoistureOutputQuotaError(f"the frame manifest needs {manifest_len} bytes but max_manifest_bytes is {limits.max_manifest_bytes}")
    frame_bytes = sum(v[2] for v in lay.values()) + manifest_len  # conservative estimate (n_active = N)
    rootfd = g0._open_dir(root)
    name = frame_name(meta.day)
    try:
        count, used, exists = _scan_root(rootfd, limits, name)
        if exists:
            return _verify_existing(rootfd, name, water_m, depth_m, active, edges, L, N, shape_yx, cell_area_m2, meta, identity, caller,
                                    limits, lay, on_device)
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
            try:  # --- before the seal: any failure removes ONLY this call's own unsealed files ---
                fds: dict[str, int] = {}
                try:
                    for fname in (WATER, DEPTH, ACTIVE):
                        made.append(fname)
                        fd = os.open(fname, g0._FLAGS_NEW, 0o600, dir_fd=dayfd)
                        fds[fname] = fd
                        _pwrite_all(fd, 0, np.frombuffer(lay[fname][0], dtype=np.uint8))
                        os.ftruncate(fd, lay[fname][2])
                    n_active, pieces = _traverse(water_m, depth_m, active, edges, L, N, limits.chunk_bytes, lay,
                                                 lambda nm, off, arr: _pwrite_all(fds[nm], off, arr), on_device)
                    for fd in fds.values():
                        os.fsync(fd)
                finally:
                    for fd in fds.values():
                        os.close(fd)
                hashes = {fname: g0._hash_stored(dayfd, fname, lay[fname][0], lay[fname][2], limits.chunk_bytes)
                          for fname in (WATER, DEPTH, ACTIVE)}
                raw = g0._canon(_build_manifest(meta, identity, caller, edges, L, N, shape_yx, cell_area_m2, n_active, hashes))
                if len(raw) > limits.max_manifest_bytes:
                    raise MoistureOutputQuotaError("the frame manifest exceeds max_manifest_bytes")
                made.append(MANIFEST + g0.TMP_SUFFIX)
                g0._write_new(dayfd, MANIFEST + g0.TMP_SUFFIX, [raw])
                os.link(MANIFEST + g0.TMP_SUFFIX, MANIFEST, src_dir_fd=dayfd, dst_dir_fd=dayfd, follow_symlinks=False)
            except BaseException:
                _discard(rootfd, name, dayfd, made)
                raise
            # --- the manifest is SEALED (published): from here on nothing sealed is ever removed, and ANY failure (including a
            # MoistureOutputError) is surfaced as an indeterminate publication ---
            try:
                os.unlink(MANIFEST + g0.TMP_SUFFIX, dir_fd=dayfd)
                os.fsync(dayfd)
                os.fsync(rootfd)
            except BaseException as exc:
                raise MoistureOutputIndeterminate(
                    f"the frame was PUBLISHED (manifest sealed) but a later durability step failed ({type(exc).__name__}); payload and "
                    "seal are intact", path) from exc
        finally:
            os.close(dayfd)
        written = sum(v[2] for v in lay.values()) + len(raw)
        return FrameNotice("saved", path, written, pieces)
    finally:
        os.close(rootfd)


def _verify_existing(rootfd, name, water, depth, active, edges, L, N, shape_yx, area, meta, identity, caller, limits, lay, on_device) -> FrameNotice:
    """Replay of an existing frame: never writes or removes anything. The FIRST difference raises a typed conflict (no accumulated
    list). The sealed manifest must equal the manifest THIS codec source would write for the supplied identity (strict writer
    identity: a replay by a changed `floating_moisture_frame` source is refused; any wider read-compatibility policy belongs to the
    runtime/source guard of the final G2 capsule and is not decided here)."""
    dayfd = g0._open_dir(name, dir_fd=rootfd)
    try:
        if g0._lstat(dayfd, MANIFEST) is None:
            raise MoistureOutputConflict(f"{name} exists UNSEALED (own earlier crash or foreign); it is not touched or completed")
        man = _check_manifest(g0._strict_loads(g0._read_small(dayfd, MANIFEST, limits.max_manifest_bytes)))
        hashes = {a["file"]: a["file_sha256"] for a in man["arrays"].values()}
        expected = _build_manifest(meta, identity, caller, edges, L, N, shape_yx, area, man["n_active"], hashes)
        if man != expected:
            raise MoistureOutputConflict(f"{name} is sealed with a different identity/metadata or by a different codec source "
                                         "(module_sha256); nothing was overwritten")
        for fname in (WATER, DEPTH, ACTIVE):
            if g0._hash_stored(dayfd, fname, lay[fname][0], lay[fname][2], limits.chunk_bytes) != hashes[fname]:
                raise MoistureOutputConflict(f"{name}/{fname} does not match its sealed checksum (corrupt)")
        fds: dict[str, int] = {}
        buf = bytearray(8 * max(1, limits.chunk_bytes // BYTES_PER_CELL))

        def compare(nm: str, off: int, arr: np.ndarray) -> None:
            if not _pread_equal(fds[nm], off, arr, buf):
                raise MoistureOutputConflict(f"{name}/{nm} is sealed with different array bytes; nothing was overwritten")

        try:
            for fname in (WATER, DEPTH, ACTIVE):
                fds[fname] = g0._open_regular(dayfd, fname, size=lay[fname][2])
            n_active, pieces = _traverse(water, depth, active, edges, L, N, limits.chunk_bytes, lay, compare, on_device)
        finally:
            for fd in fds.values():
                os.close(fd)
        if n_active != man["n_active"]:
            raise MoistureOutputConflict(f"{name} is sealed with a different active count; nothing was overwritten")
        return FrameNotice("verified_existing", name, 0, pieces)
    finally:
        os.close(dayfd)


# --- read ------------------------------------------------------------------------------------------------------------------------
def read_floating_frame(path: str | Path, *, max_return_bytes: int) -> FloatingFrame:
    """Reconstruct ONE sealed frame (see the module docstring). `path` is a `frame_*` directory."""
    cap = g0._count(max_return_bytes, "max_return_bytes", minimum=1)
    fd = g0._open_dir(path)
    try:
        man = _check_manifest(g0._strict_loads(g0._read_small(fd, MANIFEST, g0.MAX_SMALL_JSON_BYTES)))
        L, N = man["n_layers"], man["n_cells"]
        need = 8 * L * N + 8 * N + N + 8 * (L + 1)
        if need > cap:
            raise MoistureOutputQuotaError(f"the frame returns {need} bytes but max_return_bytes is {cap}; nothing was read")
        lay = _layout(L, N)
        out = {}
        for key, fname, dtype, shape in (("water_m", WATER, np.float64, (L, N)), ("depth_m", DEPTH, np.float64, (N,)),
                                          ("active", ACTIVE, np.bool_, (N,))):
            a = man["arrays"][key]
            head, size = lay[fname][0], lay[fname][2]
            f = g0._open_regular(fd, fname, size=size)
            try:
                got = os.pread(f, len(head), 0)
                if got != head:
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
        n_active = int(np.count_nonzero(out["active"]))
        if n_active != man["n_active"]:
            raise MoistureOutputError("the stored mask disagrees with n_active")
        step = max(1, g0.FINITE_BLOCK)
        for c0 in range(0, N, step):
            c1 = min(N, c0 + step)
            m = out["active"][c0:c1]
            Da = _validate_depth(out["depth_m"][c0:c1], m)
            for l in range(L):  # one row view at a time (no list of L views)
                _validate_row(out["water_m"][l, c0:c1], m, Da, edges, l)
        for arr in out.values():
            arr.setflags(write=False)
        edges.setflags(write=False)
        return FloatingFrame(man, out["water_m"], out["depth_m"], out["active"], edges, float(man["cell_area_m2"]))
    finally:
        os.close(fd)
