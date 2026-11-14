"""Water-free preparation of a LEAF vegetation projection into native floating-soil laws (Phase 2.5, slice S0).

One NEW module; it mutates no water, history, bed, lineage or checkpoint and is NOT an accepted water head. It only (1) validates explicit site controls, (2) binds ONE authentic LEAF
`leaf.water_vegetation.VegetationProjection` to those controls and to the expected source / package / mask / surface, and (3) prepares the real native operators from that binding.

    controls = site_controls(a1=, a2=, a3=, max_fraction_per_hour=, max_intake_rate_m_s=, background_receiver_weights=, n_layers=, n_cells=, max_bytes=, reserve_bytes=)
    binding  = bind_projection(projection, controls, expected_leaf_package_sha256=, expected_projection_sha256=, expected_source=, active_mask=, current_surface=, max_bytes=)
    law      = prepare_vegetation_law(column, binding, conductivity_m_s=, suction_m=, memory_budget_bytes=)      # real FloatingMatrixOperator + FloatingPreferentialOperator

Physics is NOT written here. g = `matrix_infiltration.reference_vegetation_function(R1, a1, a2)` is computed ONCE on the host and handed to the existing `prepare_floating_matrix`; the per-species
contributions (c_s, one fresh (N,) array at a time from LEAF) go through the existing `prepare_floating_preferential` and so through the one canonical `preferential_infiltration._derive_pathway`
(shared background, one aggregate rate, one cap). No extra plant K / soil factor and no stored S x N or P x N array.

Ownership. LEAF owns plants and the projection (borrowed, immutable, never copied here); SYRUP owns water; MAPLE owns the bed. A `VegetationBinding` is law PREPARATION: it carries no authority over
any water head. Every type is sealed: a private mint, a registry fingerprint over EVERY field (scalars by value, arrays by pointer / shape / dtype / flags / base type), no direct construction, copy or
serialisation. LEAF is imported lazily inside `bind_projection` only. CUDA: the path is AUTHORED against the actual native owners and the existing `floating_soil_layers_cuda` helpers; it has not run on
hardware (Root only).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import threading
import types
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import matrix_infiltration as mi
from maple_syrup import preferential_infiltration as pi
from maple_syrup import provenance

__all__ = [
    "BINDING_VERSION",
    "PreparedVegetationLaw",
    "SiteControls",
    "VegetationBinding",
    "VegetationBindingError",
    "VegetationBindingMemoryError",
    "VegetationConsumerUnavailableError",
    "bind_projection",
    "estimate_binding_bytes",
    "estimate_controls_bytes",
    "estimate_law_bytes",
    "prepare_vegetation_law",
    "site_controls",
    "verify_binding",
    "verify_prepared_law",
    "verify_site_controls",
]

BINDING_VERSION = "2"  # v2: the semantic SHA covers the exact cell area and streams the identifiers; v1 SHAs are historical and are never rebound
ORIGIN_LIVE = "live_projection"
ORIGIN_CACHED = "cached_derived"
ORIGINS = (ORIGIN_LIVE, ORIGIN_CACHED)
BINDING_ID_FIELDS = ("species_ids", "enabled_species")
CONTROLS_TAG = "maple_syrup.water_vegetation.site_controls.v1"
BINDING_TAG = "maple_syrup.water_vegetation.binding.v2"
LAW_TAG = "maple_syrup.water_vegetation.law.v1"
_F8 = 8
HASH_CHUNK_BYTES = 1 << 20
SOURCE_DIGEST_ALLOWANCE_BYTES = 4 << 20  # declared host allowance of the cold LEAF package digest (one 1 MiB read buffer + the bounded file list)
PY_RECORD_ALLOWANCE_BYTES = 8192
SOURCE_KEYS = {
    "plant_session": ("kind", "token", "sequence", "pop_sha256", "geometry_sha256", "session_id", "lineage", "catalog_hash"),
    "initial_state": ("kind", "state_sha256", "catalog_hash"),
}
_MINT = threading.local()


class VegetationBindingError(ValueError):
    """Invalid control, projection, expectation or column; nothing was minted and nothing was mutated."""


class VegetationBindingMemoryError(VegetationBindingError):
    """The declared budget / reserve / reader cannot hold the modelled bytes; raised BEFORE the stage allocates."""


class VegetationConsumerUnavailableError(VegetationBindingError):
    """The actual LEAF package (or a public API of its projection that this consumer needs) is unavailable; there is no fake projection and no fallback."""


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


def _no_direct(what: str) -> None:
    if not getattr(_MINT, "ok", False):
        raise VegetationBindingError(f"{what} is produced only by its factory (no direct construction, replace() or copy)")


def _count(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise VegetationBindingError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _admit(stage: str, need: int, max_bytes: Any, reserve_bytes: Any, reader: Callable[[], int] | None) -> None:
    """Host admission: `reserve + need <= max_bytes` and (when given) the reader."""
    mb, rb = _count(max_bytes, "max_bytes"), _count(reserve_bytes, "reserve_bytes")
    if reader is not None and not callable(reader):
        raise VegetationBindingError("available_bytes_fn must be callable or None")
    if rb + need > mb:
        raise VegetationBindingMemoryError(f"{stage}: reserve {rb} + {need} modelled bytes exceed max_bytes={mb}; nothing allocated")
    if reader is not None:
        avail = reader()
        if isinstance(avail, (bool, np.bool_)) or not isinstance(avail, (int, np.integer)) or int(avail) < 0:
            raise VegetationBindingError(f"the available-bytes reader must return an integer >= 0, got {avail!r}")
        if int(avail) < rb + need:
            raise VegetationBindingMemoryError(f"{stage}: the reader reports {int(avail)} B available, {rb + need} B needed; nothing allocated")


def _seal(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(array, dtype=np.float64).tobytes(), dtype=np.float64)  # bytes-backed, read-only (one counted copy)


def _hash_chunked(h: Any, a: np.ndarray) -> None:
    """Hash an array's bytes in bounded chunks through memoryview slices (no whole-array copy for a C-contiguous array)."""
    mv = memoryview(a).cast("B") if a.size and a.flags.c_contiguous else memoryview(np.ascontiguousarray(a)).cast("B") if a.size else memoryview(b"")
    for lo in range(0, len(mv), HASH_CHUNK_BYTES):
        h.update(mv[lo:lo + HASH_CHUNK_BYTES])


def _hex(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise VegetationBindingError(f"{name} must be a 'sha256:<64 hex>' or 64-hex string, got {type(value).__name__}")
    v = value.removeprefix("sha256:")
    if len(v) != 64 or any(ch not in "0123456789abcdef" for ch in v):
        raise VegetationBindingError(f"{name} must be a lowercase 64-hex SHA-256 (optionally prefixed 'sha256:'), got {value!r}")
    return v


def _all_finite(a: np.ndarray, chunk: int = 1 << 16) -> bool:
    flat = a.reshape(-1)
    return all(bool(np.isfinite(flat[lo:lo + chunk]).all()) for lo in range(0, flat.shape[0], chunk))


def _val_fp(v: Any) -> Any:
    """Fingerprint of one field: arrays by identity of the storage (pointer, shape, dtype, writeable flag, base type); scalars / strings / tuples by value; any other object by its id."""
    if isinstance(v, np.ndarray):
        return ("nd", int(v.__array_interface__["data"][0]), tuple(v.shape), v.dtype.str, bool(v.flags.writeable), type(v.base).__name__)
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return ("v", type(v).__name__, v if not isinstance(v, float) or math.isfinite(v) else repr(v))
    if isinstance(v, tuple):
        if all(x is None or type(x) in (str, int, float, bool) for x in v):  # an immutable tuple of immutable scalars: its identity + length fix its content (no per-element tuple is built: O(1) owned memory)
            return ("T", id(v), len(v))
        return ("t", tuple(_val_fp(x) for x in v))
    return ("o", type(v).__name__, id(v))


def _check_id_tuple(t: Any, name: str) -> tuple:
    """Factory / data-gate validation, ONCE: an identifier tuple must be exactly a tuple of strings. Afterwards the proof fingerprints it by (id, len) in O(1): an immutable tuple of immutable strings
    cannot change, and any replaced tuple has another identity."""
    if type(t) is not tuple or any(type(x) is not str for x in t):
        raise VegetationBindingError(f"{name} must be a tuple of strings")
    return t


def _fields_fp(obj: Any, id_fields: tuple = ()) -> tuple:
    """Fingerprint of EVERY dataclass field. `id_fields` names the identifier tuples that were validated once at their factory: they are fingerprinted by (id, len) in O(1) instead of an O(S) element scan on
    every proof; every other field keeps its full value / pointer / shape / dtype / flags fingerprint."""
    out = []
    for f in dataclasses.fields(obj):
        v = object.__getattribute__(obj, f.name)
        if f.name in id_fields:
            if type(v) is not tuple:
                raise VegetationBindingError(f"field {f.name} must be a tuple of validated identifiers")
            out.append((f.name, ("I", id(v), len(v))))
        else:
            out.append((f.name, _val_fp(v)))
    return tuple(out)


# ------------------------------------------------------------------------------------------------------------------------------ memory models
def estimate_controls_bytes(n_layers: int, *, a3_cells: int = 0, max_rate_cells: int = 0, has_background: bool = True) -> dict:
    """Declared host bytes of ONE controls build: the sealed maps (`a3`, `max_intake_rate_m_s` when (N,) maps; the length-L background) plus the native weight record built once for validation
    (`weight_record_bytes`) and the Python allowance. The map copy is the only copy (sealed from the caller's array); validation reads the caller's array in place; the hash is chunked."""
    L = _count(n_layers, "n_layers", 1)
    maps = _F8 * (_count(a3_cells, "a3_cells") + _count(max_rate_cells, "max_rate_cells"))
    bg = _F8 * L if has_background else 0
    weights = fpi.weight_record_bytes(L, 0)
    resident = maps + bg
    return {"resident_bytes": int(resident), "peak_bytes": int(resident + weights + PY_RECORD_ALLOWANCE_BYTES), "scope": "declared host arithmetic; the caller's input arrays are excluded"}


def estimate_binding_bytes(n_layers: int, n_species: int, *, weights_stage_bytes: int, retained_metadata_bytes: int, source_cold: bool = True, borrowed_bytes: int = 0) -> dict:
    """Declared host bytes of ONE `bind_projection`, stage by stage (peak = max; the BORROWED LEAF projection bytes are live throughout):
    IDENTITY: LEAF's `identity_record()` detached copy (the species-id list 8 S pointers; the strings are borrowed) + the Python allowance. SOURCE: the cold LEAF package digest and
    external-helper hash allowance (when `source_cold`; the helper hash reuses the same sequential read buffer). WEIGHTS: LEAF's PUBLIC `stream_estimate()['weights_stage_bytes']` (admitted by
    LEAF BEFORE anything is built, with the same reserve). SEAL: the returned raw weights (8 n L), their sealed copy and LEAF's `retained_metadata_bytes`. The surface and mask hashes are
    zero-copy chunked views (no temporary)."""
    L, S = _count(n_layers, "n_layers", 1), _count(n_species, "n_species")
    wss, retained = _count(weights_stage_bytes, "weights_stage_bytes"), _count(retained_metadata_bytes, "retained_metadata_bytes")
    borrowed = _count(borrowed_bytes, "borrowed_bytes")
    table = _F8 * S * L  # upper bound: every species enabled
    stages = {"identity": _F8 * S + PY_RECORD_ALLOWANCE_BYTES, "source": SOURCE_DIGEST_ALLOWANCE_BYTES if source_cold else 0, "weights": PY_RECORD_ALLOWANCE_BYTES + wss,
              "seal": PY_RECORD_ALLOWANCE_BYTES + 2 * table + retained}
    peak = borrowed + max(stages.values())
    return {"stages": {k: int(borrowed + v) for k, v in stages.items()}, "resident_bytes": int(table + PY_RECORD_ALLOWANCE_BYTES), "peak_bytes": int(peak), "borrowed_bytes": int(borrowed),
            "scope": "declared host arithmetic; the caller's surface and mask are excluded"}


def estimate_law_bytes(n_layers: int, n_cells: int, n_species_enabled: int, *, binding_resident_bytes: int, stream_per_next_held_bytes: int = 0, stream_per_next_free_bytes: int = 0,
                       stream_weights_stage_bytes: int = 0, backend: str = "numpy", k_suction_cell_arrays: int = 0, a3_cell_map: bool = False, max_rate_cell_map: bool = False) -> dict:
    """Declared bytes of ONE `prepare_vegetation_law`, stage by stage; `peak_*` are the MAX stage (never a padded sum), `*_reserve*` the 'everything else live' reserves handed to each native
    estimator and to LEAF's stream quota, so the caller reserve is counted ONCE and no array is counted twice.

    `stream_per_next_held_bytes` / `stream_per_next_free_bytes` are LEAF's PUBLIC `projection.stream_estimate(previous_held=True/False)['per_next_bytes']`: the arrays ONE `next()` keeps live
    inside LEAF = the new (N,) result + its species scratch + the raw weights copy it returned + (held variant only) the PREVIOUS yielded array. The previous array is counted exactly there; the
    native derivation's own estimate does NOT contain the caller's contribution arrays, so nothing is counted twice.

    CPU (numpy / numba), host only; simultaneously owned arrays per stage:
      matrix:        binding (resident + borrowed LEAF) + max(g build = 4 N doubles [the existing function copies R1, forms the two sums and the quotient], g (N) + native matrix prepare)
      preferential:  binding + the RETAINED matrix operator + native preferential prepare (`estimate_floating_preferential_memory(... 'prepare', n_species)`) + LEAF per-next (held variant)
    CUDA: two pools. DEVICE: g / per-cell a3 / max-rate / contribution device arrays (held AND next = 2 N doubles, they are adapter-owned device arrays), the retained matrix device arrays (g + K/psi
    cell arrays), the native device terms. HOST: the g build, binding, the native host staging + weight records, LEAF per-next (free variant: the host contribution is dropped after its H2D copy).

    Transfers (cold, once per preparation; no per-hydraulic-step transfer): H2D = mask N B + g 8N + per-cell maps + one 8N contribution per species (adapter-owned, exact). D2H is NOT zero:
    `d2h_adapter_bytes` = the N-byte mask read-back of `stream_hash`; `d2h_native_identity_bytes` = the native operators' cold identity hashes stream their law arrays through the host
    (matrix g + K/psi, preferential rate + max-rate + receiver); both are modelled exactly. The native 8-byte range diagnostics and 32-byte cold-reducer packets are NOT modelled (their count is
    measured by the host double and reported)."""
    L, N, S = _count(n_layers, "n_layers", 1), _count(n_cells, "n_cells", 1), _count(n_species_enabled, "n_species_enabled")
    br = _count(binding_resident_bytes, "binding_resident_bytes")
    sh, sf = _count(stream_per_next_held_bytes, "stream_per_next_held_bytes"), _count(stream_per_next_free_bytes, "stream_per_next_free_bytes")
    wss = _count(stream_weights_stage_bytes, "stream_weights_stage_bytes")
    kc = _count(k_suction_cell_arrays, "k_suction_cell_arrays")
    if kc > 2:
        raise VegetationBindingError("k_suction_cell_arrays must be 0, 1 or 2")
    if backend not in ("numpy", "numba", "cuda"):
        raise VegetationBindingError(f"backend must be numpy, numba or cuda, got {backend!r}")
    g_build = 4 * _F8 * N
    if backend != "cuda":
        mat = fm.estimate_floating_matrix_memory(L, N, backend=backend, operation="prepare")
        pref = fpi.estimate_floating_preferential_memory(L, N, backend=backend, operation="prepare", n_species=S)
        stream = sh if S else 0
        stage_a = br + max(g_build, _F8 * N + int(mat.required_bytes))
        stage_b = br + int(mat.retained_bytes) + int(pref.required_bytes) + stream
        stage_w = br + int(mat.retained_bytes) + int(pref.required_bytes) + (wss if S else 0)  # LEAF admits its public weights stage BEFORE the stream exists
        return {"stages": {"matrix": int(stage_a), "weights": int(stage_w), "preferential": int(stage_b)}, "peak_bytes": int(max(stage_a, stage_b, stage_w)), "native_reserve_matrix": int(br + _F8 * N),
                "native_reserve_preferential": int(br + int(mat.retained_bytes) + stream), "p1_stream_reserve": int(br + int(mat.retained_bytes) + int(pref.required_bytes)),
                "stream_per_next_held_bytes": int(stream),
                "scope": "declared host arithmetic of this call; the caller's column/state/K/psi arrays, the interpreter and JIT caches are excluded"}
    from maple_syrup import floating_matrix_infiltration_cuda as fmc
    from maple_syrup import floating_preferential_infiltration_cuda as fpc

    tm, tp = fmc.operation_terms("prepare", L, N), fpc.operation_terms("prepare", L, N)
    wr = fpi.weight_record_bytes(L, S)
    maps = int(a3_cell_map) + int(max_rate_cell_map)
    ret_matrix = _F8 * N * (1 + kc)  # the device g plus the K / psi cell arrays the matrix operator keeps (scalars are 8 B each: below the allowance)
    dev_a = _F8 * N + int(tm["outputs"]) + int(tm["scratch"]) + 1 * N  # g device + native + the transient bool mask (hash check precedes; bounded by N bytes)
    dev_b = ret_matrix + _F8 * N * maps + 2 * _F8 * N + int(tp["outputs"]) + int(tp["scratch"])  # retained matrix + per-cell maps + held and next contribution + native derivation
    stream = sf if S else 0
    host_a = br + max(g_build, _F8 * N + int(tm["host"]))
    host_b = br + int(tp["host"]) + wr + stream
    host_w = br + int(tp["host"]) + wr + (wss if S else 0)
    h2d = N + _F8 * N + _F8 * N * maps + S * _F8 * N
    identity_matrix = _F8 * N + _F8 * N * kc + _F8 * (2 - kc)
    identity_pref = _F8 * N + _F8 * (N if max_rate_cell_map else 1) + _F8 * L * N
    return {"stages": {"matrix": int(dev_a), "preferential": int(dev_b)}, "peak_bytes": int(max(dev_a, dev_b)), "peak_host_bytes": int(max(host_a, host_b, host_w)),
            "host_stages": {"matrix": int(host_a), "weights": int(host_w), "preferential": int(host_b)},
            "h2d_adapter_bytes": int(h2d), "h2d_bytes": int(h2d), "d2h_adapter_bytes": int(N), "d2h_native_identity_bytes": int(identity_matrix + identity_pref),
            "d2h_modelled_bytes": int(N + identity_matrix + identity_pref),
            "d2h_not_modelled": "native 8-byte range diagnostics and 32-byte cold-reducer packets (counted by the host double; hardware definitions assert the same bound)",
            "native_reserve_device_matrix": int(N + _F8 * N), "native_reserve_device_preferential": int(ret_matrix + _F8 * N * maps + 2 * _F8 * N),
            "native_reserve_host_matrix": int(br + _F8 * N), "native_reserve_host_preferential": int(br + stream),
            "p1_stream_reserve": int(br + int(tp["host"]) + wr),
            "scope": "declared; device terms = native operation_terms + this module's live device arrays; excludes the CUDA context and allocator rounding; cold once per preparation, "
                     "no per-hydraulic-step transfer exists"}


# ------------------------------------------------------------------------------------------------------------------------------ controls
def _host_or_map(value: Any, name: str, n_cells: int | None, low: float, high: float, *, allow_inf: bool) -> tuple[str, Any]:
    """('scalar', float) or ('cell', sealed (N,) float64): a Boolean is refused; a map must be exactly a 1-D numpy float64 array of length N, finite (inf allowed only when `allow_inf`) and within [low, high]."""
    if isinstance(value, (bool, np.bool_)):
        raise VegetationBindingError(f"{name} must be numeric, got a Boolean")
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        if math.isnan(v) or (not allow_inf and not math.isfinite(v)) or (math.isinf(v) and v < 0):
            raise VegetationBindingError(f"{name} must be a finite real{' (or +inf)' if allow_inf else ''}, got {value!r}")
        if v < low or v > high:
            raise VegetationBindingError(f"{name} must lie in [{low!r}, {high!r}], got {v!r}")
        return "scalar", v
    if type(value) is not np.ndarray or value.dtype != np.float64 or value.ndim != 1:
        raise VegetationBindingError(f"{name} must be a host scalar or an exact 1-D numpy float64 array of N cells, got {type(value).__name__}")
    if n_cells is None or value.shape[0] != n_cells:
        raise VegetationBindingError(f"{name} map has length {value.shape[0]} but n_cells={n_cells!r}")
    if np.isnan(value).any() or (not allow_inf and not _all_finite(value)):
        raise VegetationBindingError(f"{name} must be finite on every cell (inactive included)")
    lo, hi = float(value.min()) if value.size else 0.0, float(value.max()) if value.size else 0.0
    if lo < low or hi > high:
        raise VegetationBindingError(f"{name} must lie in [{low!r}, {high!r}] on every cell, got [{lo!r}, {hi!r}]")
    return "cell", _seal(value)


@dataclass(frozen=True, eq=False)
class SiteControls:
    """Explicit site controls (no defaults), minted ONLY by `site_controls`. `a1`, `a2` drive the matrix response g = (a1 + R1)/(a2 + R1); `a3`, `max_fraction_per_hour`, `max_intake_rate_m_s`
    and the raw `background_receiver_weights` drive the preferential pathway. `a3` / `max_intake_rate_m_s` are a host scalar or an exact sealed (N,) map; weights are raw (length L) or None."""

    a1: float
    a2: float
    a3: Any
    a3_layout: str
    max_fraction_per_hour: float
    max_intake_rate_m_s: Any
    max_rate_layout: str
    background_receiver_weights: Any
    n_layers: int
    n_cells: int | None
    sha256: str
    resident_bytes: int
    tag: str = CONTROLS_TAG

    def __post_init__(self) -> None:
        _no_direct("SiteControls")

    def __copy__(self):
        raise VegetationBindingError("SiteControls is not copyable; build another with site_controls")

    def __deepcopy__(self, memo):
        raise VegetationBindingError("SiteControls is not copyable; build another with site_controls")

    def __reduce_ex__(self, protocol):
        raise VegetationBindingError("SiteControls is not serialisable (no checkpoint support in S0)")


_CTL_REGISTRY: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _check_controls(c: Any) -> SiteControls:
    if type(c) is not SiteControls:
        raise VegetationBindingError(f"controls must be a SiteControls, got {type(c).__name__}")
    fp = _CTL_REGISTRY.get(c)
    if fp is None or _fields_fp(c) != fp:
        raise VegetationBindingError("the SiteControls record is forged, replaced or altered after sealing")
    return c


def _controls_sha(a1: float, a2: float, a3: Any, a3_layout: str, mf: float, mr: Any, mr_layout: str, bg: Any, L: int, N: int | None) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"tag": CONTROLS_TAG, "a1": repr(a1), "a2": repr(a2), "max_fraction_per_hour": repr(mf), "a3_layout": a3_layout, "max_rate_layout": mr_layout,
                         "n_layers": L, "n_cells": N, "background": bg is not None}, sort_keys=True).encode())
    for name, v in (("a3", a3), ("max_intake_rate_m_s", mr)):
        h.update(name.encode() + b"\0")
        if isinstance(v, np.ndarray):
            _hash_chunked(h, v)
        else:
            h.update(repr(float(v)).encode())
    h.update(b"background\0")
    if bg is not None:
        _hash_chunked(h, bg)
    return h.hexdigest()


def site_controls(*, a1: Any, a2: Any, a3: Any, max_fraction_per_hour: Any, max_intake_rate_m_s: Any, background_receiver_weights: Any, n_layers: int, n_cells: int | None = None,
                  max_bytes: Any, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None) -> SiteControls:
    """Validate and seal the explicit site controls. Every value is required (no defaults). Order: metadata and sizes -> ADMISSION -> value validation -> sealing -> mint.

    a1 >= 0, a2 > 0 and a1 <= a2 (so g = (a1 + R1)/(a2 + R1) lies in [a1/a2, 1], the native matrix range); validated with the actual `reference_vegetation_function`. `a3` in [0, 1] and
    `max_fraction_per_hour` in [0, 1]; `max_intake_rate_m_s` >= 0 (`inf` is legal); `a3` / `max_intake_rate_m_s` are host scalars or exact (N,) float64 arrays (`n_cells` required for a map).
    `background_receiver_weights` is a raw nonnegative length-`n_layers` vector or an explicit None (no background pathway). The caller's arrays are never modified or aliased."""
    L = _count(n_layers, "n_layers", 1)
    N = None if n_cells is None else _count(n_cells, "n_cells", 1)
    a3_cells = a3.shape[0] if isinstance(a3, np.ndarray) and a3.ndim == 1 else 0
    mr_cells = max_intake_rate_m_s.shape[0] if isinstance(max_intake_rate_m_s, np.ndarray) and max_intake_rate_m_s.ndim == 1 else 0
    est = estimate_controls_bytes(L, a3_cells=a3_cells, max_rate_cells=mr_cells, has_background=background_receiver_weights is not None)
    _admit("site_controls", est["peak_bytes"], max_bytes, reserve_bytes, available_bytes_fn)
    try:
        a1v, a2v = fsl._real(a1, "a1"), fsl._real(a2, "a2")
        mi.reference_vegetation_function(0.0, a1v, a2v)  # the native validation of a1 >= 0 and a2 > 0
    except (fsl.FloatingSoilError, mi.MatrixError, ValueError, TypeError) as exc:
        raise VegetationBindingError(f"a1/a2 refused by the native vegetation function: {exc}") from exc
    if a1v > a2v:
        raise VegetationBindingError(f"a1 ({a1v!r}) must not exceed a2 ({a2v!r}): the native matrix factor g = (a1 + R1)/(a2 + R1) must stay <= 1")
    mfv = _host_or_map(max_fraction_per_hour, "max_fraction_per_hour", None, 0.0, 1.0, allow_inf=False)
    if mfv[0] != "scalar":
        raise VegetationBindingError("max_fraction_per_hour must be a host scalar")
    a3_layout, a3_v = _host_or_map(a3, "a3", N, 0.0, 1.0, allow_inf=False)
    mr_layout, mr_v = _host_or_map(max_intake_rate_m_s, "max_intake_rate_m_s", N, 0.0, math.inf, allow_inf=True)
    bg = None
    if background_receiver_weights is not None:
        try:
            rec = pi.receiver_weights(background_receiver_weights, L, "background_receiver_weights")  # the native validation (real, finite, >= 0, length L)
        except (pi.PreferentialError, ValueError, TypeError) as exc:
            raise VegetationBindingError(f"background_receiver_weights refused by the native weight record: {exc}") from exc
        bg = _seal(np.asarray(rec.raw))
    sha = _controls_sha(a1v, a2v, a3_v, a3_layout, mfv[1], mr_v, mr_layout, bg, L, N)
    resident = sum(a.nbytes for a in (a3_v, mr_v, bg) if isinstance(a, np.ndarray))
    with _minting():
        c = SiteControls(a1v, a2v, a3_v, a3_layout, mfv[1], mr_v, mr_layout, bg, L, N, sha, int(resident))
    _CTL_REGISTRY[c] = _fields_fp(c)
    return c


# ------------------------------------------------------------------------------------------------------------------------------ binding
_LEAF_CACHE: dict = {}


def _load_leaf() -> Any:
    """Lazy import of the ACTUAL LEAF projection module; any failure is the typed unavailable error (no fake class, no fallback)."""
    try:
        import leaf.water_vegetation as lw
    except Exception as exc:
        raise VegetationConsumerUnavailableError(f"the actual LEAF package (leaf.water_vegetation) cannot be imported: {type(exc).__name__}: {exc}") from exc
    for name in ("VegetationProjection", "package_source_record", "SCHEMA", "VERSION"):
        if not hasattr(lw, name):
            raise VegetationConsumerUnavailableError(f"leaf.water_vegetation lacks the public {name}; the projection API is not usable")
    return lw


def _runtime_leaf_digest(lw: Any) -> dict:
    """SYRUP `provenance.source_tree_digest` over the directory of the ACTUALLY IMPORTED `leaf` package (every non-bytecode regular file), computed independently of LEAF's own record; cold once per
    process and package path."""
    root = os.path.dirname(os.path.realpath(lw.__file__))
    if root not in _LEAF_CACHE:
        d = provenance.source_tree_digest(root)
        _LEAF_CACHE[root] = {"digest_sha256": d.digest_sha256, "file_count": d.file_count, "total_bytes": d.total_bytes, "algorithm": d.algorithm, "root": d.root}
    return dict(_LEAF_CACHE[root])


def _frozen_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, eq=False)
class VegetationBinding:
    """ONE current compact binding record (no history), minted ONLY by `bind_projection`. It BORROWS the sealed LEAF projection (never copies its arrays) and owns the sealed raw receiver table of the
    enabled species, the controls, and the identities. It is law preparation, not a water head and not an accepted generation."""

    projection: Any
    controls: SiteControls
    n_cells: int
    n_layers: int
    shape: tuple
    cell_area_m2: float
    species_ids: tuple
    enabled_species: tuple
    receiver_weights: Any  # (n_enabled, L) raw float64, sealed
    n_entries: int
    leaf_package_sha256: str
    leaf_package_file_count: int
    external_helpers_json: str
    projection_sha256: str
    source_kind: str
    source_json: str
    mask_sha256: str
    surface_sha256: str
    sha256: str
    borrowed_bytes: int
    resident_bytes: int
    origin: str
    tag: str = BINDING_TAG

    def __post_init__(self) -> None:
        _no_direct("VegetationBinding")

    def __copy__(self):
        raise VegetationBindingError("VegetationBinding is not copyable; bind again")

    def __deepcopy__(self, memo):
        raise VegetationBindingError("VegetationBinding is not copyable; bind again")

    def __reduce_ex__(self, protocol):
        raise VegetationBindingError("VegetationBinding is not serialisable (no checkpoint support in S0)")

    @property
    def source(self) -> dict:
        """A fresh detached copy of the recorded LEAF source identity (session token/sequence/pop/geometry hashes, or the labelled initial-state hash)."""
        return json.loads(self.source_json)

    def export_metadata(self) -> dict:
        """Fresh plain record of every scalar identity of the binding (for a later strict held record). Arrays are NOT included (see `export_arrays`)."""
        return {"tag": self.tag, "version": BINDING_VERSION, "sha256": self.sha256, "controls_sha256": self.controls.sha256, "leaf_package_sha256": "sha256:" + self.leaf_package_sha256,
                "leaf_package_file_count": self.leaf_package_file_count, "external_helpers": json.loads(self.external_helpers_json), "projection_sha256": "sha256:" + self.projection_sha256, "source_kind": self.source_kind, "source": self.source,
                "mask_sha256": self.mask_sha256, "surface_sha256": self.surface_sha256, "n_cells": self.n_cells, "n_layers": self.n_layers, "shape": list(self.shape), "cell_area_m2_hex": float(self.cell_area_m2).hex(),
                "species_ids": list(self.species_ids), "enabled_species": list(self.enabled_species), "n_entries": self.n_entries, "borrowed_bytes": self.borrowed_bytes,
                "resident_bytes": self.resident_bytes, "origin": self.origin, "water_head_authority": False}

    def export_compact(self) -> Any:
        """Read-only record of EVERYTHING a strict held root needs, without any copy or S x N / P x N array: the binding metadata (identities, source, helper pins, mask / surface hashes, sizes), the
        controls (scalars by value, maps / background as read-only views), LEAF's own `export_compact()` of the projection (immutable views of R1, splash / shading cover, the cell-major
        (cell, species) area groups, the species root table and RAW receiver table, omitted ids, cell area, content hash, frozen identity) and the enabled-species receiver table (owned,
        sealed). Metadata-only validation (registry + producer fingerprint); it creates nothing and is not a rehydration factory (no plant is re-derived)."""
        b = _check_binding(self)
        c = b.controls
        ctl = {"a1": c.a1, "a2": c.a2, "a3": c.a3, "a3_layout": c.a3_layout, "max_fraction_per_hour": c.max_fraction_per_hour, "max_intake_rate_m_s": c.max_intake_rate_m_s,
               "max_rate_layout": c.max_rate_layout, "background_receiver_weights": c.background_receiver_weights, "n_layers": c.n_layers, "n_cells": c.n_cells, "sha256": c.sha256}
        return types.MappingProxyType({"binding": types.MappingProxyType(self.export_metadata()), "controls": types.MappingProxyType(ctl), "projection": _compact_of(b),
                                       "receiver_weights_enabled": b.receiver_weights})

    def export_arrays(self) -> Any:
        """Flat read-only mapping of every array of `export_compact` (views, no copies): the projection's fields and tables, the controls' maps and background weights, and the enabled-species table."""
        e = self.export_compact()
        out = {f"projection.{k}": v for k, v in e["projection"]["arrays"].items()}
        out["receiver_weights_enabled"] = e["receiver_weights_enabled"]
        for k in ("a3", "max_intake_rate_m_s", "background_receiver_weights"):
            if isinstance(e["controls"][k], np.ndarray):
                out[f"controls.{k}"] = e["controls"][k]
        return types.MappingProxyType(out)


_BND_REGISTRY: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def verify_site_controls(controls: Any) -> SiteControls:
    """Public genuine-factory proof of a `SiteControls` (exact type, registry, every field; metadata only, no rehash)."""
    return _check_controls(controls)


def verify_binding(binding: Any) -> VegetationBinding:
    """Public genuine-factory proof of a `VegetationBinding` for an owning lifecycle: exact type, registry, every field, the nested controls and the borrowed projection's own sealed-record check.
    Metadata only: no content rehash and no copy."""
    return _check_binding(binding)


def _check_binding(b: Any) -> VegetationBinding:
    if type(b) is not VegetationBinding:
        raise VegetationBindingError(f"binding must be a VegetationBinding, got {type(b).__name__}")
    fp = _BND_REGISTRY.get(b)
    if fp is None or _fields_fp(b, BINDING_ID_FIELDS) != fp:
        raise VegetationBindingError("the VegetationBinding record is forged, replaced or altered after sealing")
    _check_controls(b.controls)
    if b.origin == ORIGIN_CACHED:
        _held().verify_cached_inputs(b.projection)  # the cached record's own registry / every-field check (a labelled cached recipe, never a live projection)
        return b
    if b.origin != ORIGIN_LIVE:
        raise VegetationBindingError(f"unknown binding origin {b.origin!r}")
    if not callable(getattr(type(b.projection), "stream_estimate", None)):
        raise VegetationConsumerUnavailableError("this LEAF projection has no public stream_estimate(); the stream model is unavailable (no private fallback)")
    try:
        view = b.projection.matrix_root_index  # a PUBLIC sealed view (header only, no copy): LEAF's own sealed-record check runs, with no per-species work (`stream_estimate` loops over the species)
        del view
    except Exception as exc:
        raise VegetationBindingError(f"the borrowed LEAF projection is no longer a genuine sealed record: {exc}") from exc
    return b


def _held() -> Any:
    """The cached-inputs module (lazy: it imports this one). Exact-type typed dispatch only; no duck typing."""
    from maple_syrup import water_vegetation_held as wh

    return wh


def _compact_of(b: VegetationBinding) -> Any:
    return _held().cached_compact(b.projection, with_identity=True) if b.origin == ORIGIN_CACHED else b.projection.export_compact()


def _r1_of(b: VegetationBinding) -> Any:
    return _held().cached_r1(b.projection) if b.origin == ORIGIN_CACHED else b.projection.matrix_root_index


def _inputs_of(b: VegetationBinding, max_bytes: Any, reserve_bytes: Any, reader: Any) -> tuple:
    """`(ids, raw weights (n, L), contribution iterator)` of the binding's origin: LEAF's `preferential_inputs` (live) or the canonical H2 `preferential_inputs_from_compact` (cached)."""
    if b.origin == ORIGIN_CACHED:
        return _held().cached_preferential_inputs(b.projection, max_bytes, reserve_bytes, reader)
    return b.projection.preferential_inputs(max_bytes=max_bytes, reserve_bytes=reserve_bytes, available_bytes_fn=reader)


def _check_domain_area(domain: Any, shape: tuple, area: float) -> None:
    """The identity's domain must be the given (ny, nx) grid with SQUARE cells and EXACTLY the projection's cell area (dx * dy)."""
    if not isinstance(domain, dict) or any(k not in domain for k in ("nx", "ny", "dx_m", "dy_m")):
        raise VegetationBindingError("the identity domain lacks nx / ny / dx_m / dy_m")
    nx, ny, dx, dy = domain["nx"], domain["ny"], domain["dx_m"], domain["dy_m"]
    if (ny, nx) != tuple(shape):
        raise VegetationBindingError(f"the identity domain grid ({ny}, {nx}) differs from the (ny, nx) shape {tuple(shape)}")
    if isinstance(dx, bool) or isinstance(dy, bool) or not isinstance(dx, (int, float)) or not isinstance(dy, (int, float)) or float(dx) != float(dy) or not float(dx) > 0.0:
        raise VegetationBindingError("cells must be square with a positive size (dx_m == dy_m > 0)")
    if float(dx) * float(dy) != float(area) or not math.isfinite(area):
        raise VegetationBindingError(f"the cell area {area!r} is not dx_m * dy_m = {float(dx) * float(dy)!r}")


def _check_domain_fields(lw: Any, proj: Any, ident: dict, N: int, L: int, shape: tuple) -> None:
    for key in ("schema", "version", "catalog_hash", "domain", "surface_sha256", "active_mask_sha256", "package", "external_helpers", "species_ids", "root_layers", "content_sha256", "source"):
        if key not in ident:
            raise VegetationBindingError(f"the LEAF projection identity lacks {key!r}; the consumer refuses an unknown projection layout")
    if ident["schema"] != lw.SCHEMA or ident["version"] != lw.VERSION:
        raise VegetationBindingError(f"unknown projection schema/version {ident['schema']!r}/{ident['version']!r}; this consumer knows {lw.SCHEMA!r}/{lw.VERSION!r} only")
    if int(ident["root_layers"]) != L:
        raise VegetationBindingError(f"projection root_layers {ident['root_layers']!r} != controls n_layers {L}")
    if int(proj.n_cells) != N or tuple(int(v) for v in proj.shape) != shape:
        raise VegetationBindingError(f"projection domain {tuple(proj.shape)} / {proj.n_cells} cells != the given mask {shape} / {N}")
    ids_a, ids_b = ident["species_ids"], proj.species_ids
    if len(ids_a) != len(ids_b) or any(x != y for x, y in zip(ids_a, ids_b, strict=True)):
        raise VegetationBindingError("projection species ids disagree with its identity")
    _check_domain_area(ident["domain"], shape, float(proj.cell_area_m2))
    for name, lo, hi in (("splash_cover", 0.0, 1.0), ("shading_cover", 0.0, 1.0), ("matrix_root_index", 0.0, math.inf)):
        a = getattr(proj, name)
        if a.dtype != np.float64 or a.shape != (N,) or a.flags.writeable or not _all_finite(a) or (a.size and (float(a.min()) < lo or float(a.max()) > hi)):
            raise VegetationBindingError(f"projection.{name} is not a read-only finite float64 ({N},) array within [{lo}, {hi}]")


def _check_source(src: Any, expected: dict) -> None:
    """The recorded source must be one of the two known kinds with EXACTLY its identity keys (plus the free-text `note`), and the caller's expected record must carry every mandatory key of that kind
    with equal values (a bare {'kind': ...} cannot skip the generation / state hash)."""
    if not isinstance(src, dict) or src.get("kind") not in SOURCE_KEYS:
        raise VegetationBindingError(f"unknown projection source record (kind {src.get('kind') if isinstance(src, dict) else src!r})")
    kind = src["kind"]
    req = SOURCE_KEYS[kind]
    if set(src) - set(req) - {"note"} or set(req) - set(src):
        raise VegetationBindingError(f"the projection source record has keys {sorted(src)} but a {kind!r} source has {sorted(req)} (+ note)")
    missing = [k for k in req if k not in expected]
    if missing:
        raise VegetationBindingError(f"expected_source must carry every mandatory {kind!r} key; missing {missing}")
    extra = [k for k in expected if k not in req]
    if extra:
        raise VegetationBindingError(f"expected_source has unknown keys {extra} for a {kind!r} source")
    for k in req:
        if src[k] != expected[k]:
            raise VegetationBindingError(f"expected_source[{k!r}]={expected[k]!r} does not match the projection source ({src[k]!r}); stale or mismatched generation")


def _stream_estimate(b: VegetationBinding, previous_held: bool) -> dict:
    """LEAF's PUBLIC stream model (`stream_estimate(previous_held=)`; cached origin: the public `estimate_stream_bytes`): weights stage, retained metadata, per-next bytes. A projection without
    it is unusable for this consumer (typed unavailable): no private mirror, no caller-supplied arithmetic, no fallback."""
    if b.origin == ORIGIN_CACHED:
        return _held().cached_stream_estimate(b.projection, previous_held)
    if not callable(getattr(type(b.projection), "stream_estimate", None)):
        raise VegetationConsumerUnavailableError("this LEAF projection has no public stream_estimate(); the consumer cannot model the contribution stream (no private mirror, no declaration)")
    return b.projection.stream_estimate(previous_held=previous_held)


def _stream_per_next(b: VegetationBinding, previous_held: bool) -> int:
    return _count(_stream_estimate(b, previous_held)["per_next_bytes"], "stream_estimate['per_next_bytes']")


HELPER_KEYS = ("module", "sha256", "bytes")
HELPER_MAX_BYTES = 64 << 20


def _normalise_helpers(helpers: Any, name: str) -> tuple:
    """Exact list of {'module', 'sha256', 'bytes'} (no other key, no duplicate module, non-empty), as a sorted tuple of (module, hex, bytes)."""
    if isinstance(helpers, (str, bytes, dict)) or not isinstance(helpers, (list, tuple)) or not helpers:
        raise VegetationBindingError(f"{name} must be a non-empty list of {{'module', 'sha256', 'bytes'}} records")
    out = []
    for h in helpers:
        if not isinstance(h, dict) or set(h) != set(HELPER_KEYS):
            raise VegetationBindingError(f"{name} records must have exactly the keys {HELPER_KEYS}, got {sorted(h) if isinstance(h, dict) else type(h).__name__}")
        if not isinstance(h["module"], str) or not h["module"]:
            raise VegetationBindingError(f"{name}: module must be a non-empty string")
        out.append((h["module"], _hex(h["sha256"], f"{name}.sha256"), _count(h["bytes"], f"{name}.bytes")))
    if len({m for m, _, _ in out}) != len(out):
        raise VegetationBindingError(f"{name} lists a module twice")
    return tuple(sorted(out))


_HELPER_CACHE: dict = {}


def _helper_path(module: str) -> str:
    import importlib.util
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError) as exc:
        raise VegetationBindingError(f"the external helper module {module!r} cannot be located: {exc}") from exc
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        raise VegetationBindingError(f"the external helper module {module!r} cannot be located")
    return os.path.realpath(spec.origin)


def _runtime_helper(module: str) -> tuple:
    """(module, sha256 hex, bytes) of the EXTERNAL helper module the LEAF builders import, located WITHOUT importing it and hashed through SYRUP's bounded provenance file reader (regular file
    only, no symlink, size-capped); cold once per source path. A missing helper refuses; nothing is cleared or swapped to accept a foreign file."""
    from pathlib import Path
    path = _helper_path(module)
    if path not in _HELPER_CACHE:
        info = os.lstat(path)
        try:
            digest, size = provenance._hash_regular_file(Path(path), info.st_size, HELPER_MAX_BYTES)
        except (provenance.ProvenanceScopeError, OSError) as exc:
            raise VegetationBindingError(f"the external helper {module!r} cannot be hashed as a bounded regular file: {exc}") from exc
        _HELPER_CACHE[path] = (digest, int(size))
    digest, size = _HELPER_CACHE[path]
    return (module, digest, size)


def _verify_source_pins(lw: Any, recorded_package: Any, recorded_helpers: Any, pkg_expected: str, helpers_expected: tuple) -> dict:
    """The ONE strict source validation shared by `bind_projection` and the cached-inputs stage: the actual imported LEAF package digest (SYRUP's own `source_tree_digest`), the recorded package
    digest and the expected pin must be equal; the recorded external helpers, the expected pins and the helper bytes SYRUP hashes now must be equal. Returns the runtime digest record."""
    runtime = _runtime_leaf_digest(lw)
    if not isinstance(recorded_package, dict) or "digest" not in recorded_package or "file_count" not in recorded_package:
        raise VegetationBindingError("the recorded package must be a record with 'digest' and 'file_count'")
    rec_digest, rec_count = recorded_package["digest"], recorded_package["file_count"]
    if type(rec_count) is not int or type(runtime["file_count"]) is not int or rec_count != runtime["file_count"]:
        raise VegetationBindingError(f"the recorded package file_count {rec_count!r} must be a strict integer equal to the actually imported package's {runtime['file_count']!r} files "
                                     "(no coercion; a forged or stale count is refused)")
    if _hex(rec_digest, "recorded package") != pkg_expected or runtime["digest_sha256"] != pkg_expected:
        raise VegetationBindingError("the actual imported LEAF package digest, the recorded package digest and the expected pin are not all equal (stale or foreign source); "
                                     f"expected {pkg_expected}, runtime {runtime['digest_sha256']}, recorded {rec_digest}")
    rec_list = recorded_helpers.get("helpers") if isinstance(recorded_helpers, dict) else None
    if not isinstance(rec_list, list):
        raise VegetationBindingError("the recorded identity has no external_helpers record; the consumer refuses an unknown layout")
    recorded = _normalise_helpers([{k: h[k] for k in HELPER_KEYS} for h in rec_list], "recorded external_helpers") if rec_list else ()
    if recorded != helpers_expected:
        raise VegetationBindingError(f"the recorded external helpers {recorded} differ from the expected pins {helpers_expected} (missing / extra / wrong helper)")
    runtime_helpers = tuple(sorted(_runtime_helper(m) for m, _, _ in helpers_expected))
    if runtime_helpers != helpers_expected:
        raise VegetationBindingError(f"the actual external helper bytes {runtime_helpers} differ from the expected pins {helpers_expected} (changed or foreign helper)")
    return runtime


def _binding_sha(controls_sha: str, pkg: str, proj: str, src: dict, mask: str, surf: str, shape: tuple, area: float, species: tuple, enabled: tuple, layers: int, helpers: tuple,
                 table: np.ndarray) -> str:
    """The semantic binding SHA, version 2 (origin-independent; a cached-derived binding of the same recipe reproduces the live binding's SHA exactly). The scalar / O(1) parts are one canonical
    JSON document; the species ids and the enabled ids are STREAMED one at a time (no identifier-sized JSON copy); the exact cell area is part of the recipe."""
    top = hashlib.sha256()
    top.update(_frozen_json({"tag": BINDING_TAG, "version": BINDING_VERSION, "controls": controls_sha, "package": pkg, "projection": proj, "source": src, "mask": mask, "surface": surf,
                             "shape": list(shape), "cell_area_hex": float(area).hex(), "n_species": len(species), "n_enabled": len(enabled), "layers": layers,
                             "helpers": [list(h) for h in helpers]}).encode())
    for group in (species, enabled):
        top.update(b"[")
        for name in group:
            top.update(name.encode() + b"\0")
        top.update(b"]")
    _hash_chunked(top, table)
    return top.hexdigest()


def bind_projection(projection: Any, controls: SiteControls, *, expected_leaf_package_sha256: Any, expected_projection_sha256: Any, expected_source: dict,
                    expected_external_helpers: Any, active_mask: np.ndarray, current_surface: np.ndarray, session: Any = None, max_bytes: Any, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None) -> VegetationBinding:
    """Bind ONE authentic LEAF projection to `controls` (law preparation only). Order: cheap metadata -> genuine-projection check -> ADMISSION -> actual-source and projection identity checks -> mint.

    Required (all keyword-only, none defaulted): the expected FULL LEAF package digest (compared with the digest SYRUP computes over the ACTUALLY IMPORTED package files AND with the projection's own
    recorded one), the expected projection content hash, the expected source record WITH EVERY MANDATORY KEY of its kind (session: kind, token, sequence, pop_sha256, geometry_sha256, session_id,
    lineage, catalog_hash; initial state: kind, state_sha256, catalog_hash), the REQUIRED expected external helper pins (`[{'module', 'sha256', 'bytes'}, ...]`: the helper
    modules outside the LEAF package digest, e.g. `benchmarks.spatial_mapping`; compared with the projection's recorded `external_helpers` AND with the helper file bytes SYRUP hashes at run time;
    a missing, extra, wrong or changed helper refuses before anything is minted), the (ny, nx) bool `active_mask` and the (ny, nx) float64 `current_surface` that the projection was built from (cold hashes;
    a mismatch is stale). `session`, when given, is additionally checked with LEAF's own `check_session`. An `initial_state` projection is labelled in the binding and is never a session generation.
    The sequence is NOT required to advance: the same biology on a new legitimate surface is a new binding with the same source record."""
    ctl = _check_controls(controls)
    if not isinstance(expected_source, dict) or not expected_source:
        raise VegetationBindingError("expected_source must be a non-empty dict carrying every mandatory key of its source kind")
    pkg_expected, proj_expected = _hex(expected_leaf_package_sha256, "expected_leaf_package_sha256"), _hex(expected_projection_sha256, "expected_projection_sha256")
    for name, a, dt in (("active_mask", active_mask, np.bool_), ("current_surface", current_surface, np.float64)):
        if type(a) is not np.ndarray or a.dtype != dt or a.ndim != 2 or not a.flags.c_contiguous:
            raise VegetationBindingError(f"{name} must be a C-contiguous 2-D numpy {np.dtype(dt).name} array")
    if active_mask.shape != current_surface.shape:
        raise VegetationBindingError("active_mask and current_surface must have the same (ny, nx) shape")
    shape = tuple(int(v) for v in active_mask.shape)
    N = shape[0] * shape[1]
    if ctl.n_cells is not None and ctl.n_cells != N:
        raise VegetationBindingError(f"the controls' (N,) maps are for n_cells={ctl.n_cells} but the domain has {N}")
    lw = _load_leaf()
    if type(projection) is not lw.VegetationProjection:
        raise VegetationBindingError(f"projection must be a genuine leaf.water_vegetation.VegetationProjection, got {type(projection).__module__}.{type(projection).__name__}")
    if not callable(getattr(type(projection), "stream_estimate", None)):
        raise VegetationConsumerUnavailableError("this LEAF projection has no public stream_estimate(); the binding cannot model LEAF's weights stage")
    S, L = len(projection.species_ids), ctl.n_layers
    helpers_expected = _normalise_helpers(expected_external_helpers, "expected_external_helpers")
    cold = os.path.dirname(os.path.realpath(lw.__file__)) not in _LEAF_CACHE or any(_helper_path(m) not in _HELPER_CACHE for m, _, _ in helpers_expected)
    pub = projection.stream_estimate(previous_held=False)  # PUBLIC numbers (also the genuine-record check): admitted BEFORE any owned copy below
    est = estimate_binding_bytes(L, S, weights_stage_bytes=pub["weights_stage_bytes"], retained_metadata_bytes=pub["retained_metadata_bytes"], source_cold=cold,
                                 borrowed_bytes=int(projection.resident_bytes))
    _admit("bind_projection", est["peak_bytes"], max_bytes, reserve_bytes, available_bytes_fn)
    ident = projection.identity_record()  # the detached identity copy (8 S pointers + O(1) metadata) is inside the admitted `identity` stage
    _check_domain_fields(lw, projection, ident, N, L, shape)
    runtime = _verify_source_pins(lw, ident["package"], ident["external_helpers"], pkg_expected, helpers_expected)
    if _hex(ident["content_sha256"], "projection content") != proj_expected:
        raise VegetationBindingError("the projection content hash differs from the expected projection hash")
    src = ident["source"]
    _check_source(src, expected_source)
    mh = hashlib.sha256()
    _hash_chunked(mh, active_mask.reshape(-1).view(np.uint8))  # zero-copy: a bool array's bytes ARE its uint8 values
    mask_sha = mh.hexdigest()
    sh = hashlib.sha256()
    _hash_chunked(sh, current_surface)
    surf_sha = sh.hexdigest()
    if mask_sha != ident["active_mask_sha256"]:
        raise VegetationBindingError("the active mask differs from the projection's (stale or mismatched mask)")
    if surf_sha != ident["surface_sha256"]:
        raise VegetationBindingError("the current surface differs from the projection's (stale or mismatched surface)")
    if session is not None:
        try:
            projection.check_session(session, current_surface=current_surface, active_mask=active_mask)
        except Exception as exc:
            raise VegetationBindingError(f"LEAF check_session refused: {exc}") from exc
    ids, weights, gen = projection.preferential_inputs(max_bytes=max_bytes, reserve_bytes=_count(reserve_bytes, "reserve_bytes") + PY_RECORD_ALLOWANCE_BYTES + int(projection.resident_bytes),
                                                       available_bytes_fn=available_bytes_fn)
    gen.close()  # the contributions are streamed only at law preparation; none is consumed or retained here
    if weights.shape != (len(ids), L) or weights.dtype != np.float64:
        raise VegetationBindingError(f"the LEAF receiver table has shape {weights.shape}, expected ({len(ids)}, {L}) float64")
    table = _seal(weights).reshape(weights.shape)
    del weights
    source_json = _frozen_json(src)
    area = float(projection.cell_area_m2)
    _check_id_tuple(projection.species_ids, "species_ids")
    _check_id_tuple(ids, "enabled_species")
    bsha = _binding_sha(ctl.sha256, pkg_expected, proj_expected, src, mask_sha, surf_sha, shape, area, projection.species_ids, ids, L, helpers_expected, table)
    with _minting():
        b = VegetationBinding(projection, ctl, N, L, shape, area, tuple(projection.species_ids), tuple(ids), table, int(projection.n_entries), pkg_expected, int(runtime["file_count"]), _frozen_json([{"module": m, "sha256": "sha256:" + h, "bytes": n} for m, h, n in helpers_expected]), proj_expected,
                              str(src["kind"]), source_json, mask_sha, surf_sha, bsha, int(projection.resident_bytes), int(table.nbytes + ctl.resident_bytes + PY_RECORD_ALLOWANCE_BYTES), ORIGIN_LIVE)
    _BND_REGISTRY[b] = _fields_fp(b, BINDING_ID_FIELDS)
    return b


# ------------------------------------------------------------------------------------------------------------------------------ prepared law
@dataclass(frozen=True, eq=False)
class PreparedVegetationLaw:
    """The REAL native operators prepared from one binding (matrix g computed once; preferential through the canonical derivation) plus the law identity. A pure trial result: no step was taken, no
    history created, no lineage adopted, no water head authority. Minted ONLY by `prepare_vegetation_law`; `verify_prepared_law` checks the registry and every field (O(1), no rehash)."""

    matrix_op: Any
    preferential_op: Any
    binding_sha256: str
    column_sha256: str
    backend: str
    binding_origin: str
    device_id: int | None
    g_sha256: str
    g_min: float
    g_max: float
    n_contributions: int
    law_sha256: str
    matrix_sha256: str
    preferential_sha256: str
    memory_json: str
    tag: str = LAW_TAG
    water_head_authority: bool = False
    step_taken: bool = False

    def __post_init__(self) -> None:
        _no_direct("PreparedVegetationLaw")

    def __copy__(self):
        raise VegetationBindingError("PreparedVegetationLaw is not copyable; prepare again")

    def __deepcopy__(self, memo):
        raise VegetationBindingError("PreparedVegetationLaw is not copyable; prepare again")

    def __reduce_ex__(self, protocol):
        raise VegetationBindingError("PreparedVegetationLaw is not serialisable (no checkpoint support in S0)")

    @property
    def memory(self) -> dict:
        """A fresh detached copy of the declared memory / transfer record of the preparation."""
        return json.loads(self.memory_json)


_LAW_REGISTRY: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def verify_prepared_law(law: Any) -> PreparedVegetationLaw:
    """Genuine-factory proof of a prepared law (registry + every field + the two operators' identities); metadata only, no hashing of content."""
    if type(law) is not PreparedVegetationLaw:
        raise VegetationBindingError(f"law must be a PreparedVegetationLaw, got {type(law).__name__}")
    fp = _LAW_REGISTRY.get(law)
    if fp is None or _fields_fp(law) != fp:
        raise VegetationBindingError("the PreparedVegetationLaw record is forged, replaced or altered after sealing")
    if law.matrix_op.sha256 != law.matrix_sha256 or law.preferential_op.sha256 != law.preferential_sha256:
        raise VegetationBindingError("a prepared operator no longer carries the recorded identity")
    if law.matrix_op.column_sha256 != law.column_sha256 or law.preferential_op.column_sha256 != law.column_sha256:
        raise VegetationBindingError("a prepared operator is not bound to the recorded column")
    return law


def _counted_stream(gen: Any, n_cells: int, state: dict, to_native: Callable[[np.ndarray], Any] | None = None) -> Any:
    """Yield LEAF's contributions one at a time, never retaining one; every array must be a fresh finite >= 0 float64 (N,) array (checked before the native derivation sees it). `to_native` (CUDA:
    the admitted cold H2D) converts each host array to the column's namespace."""
    for c in gen:
        if type(c) is not np.ndarray or c.dtype != np.float64 or c.shape != (n_cells,) or not c.flags.writeable or c.base is not None or not _all_finite(c) or (c.size and float(c.min()) < 0.0):
            raise VegetationBindingError("a LEAF contribution is not a fresh writeable finite non-negative float64 (N,) array")
        state["count"] += 1
        out = c if to_native is None else to_native(c)
        c = None
        yield out
        out = None


def _is_cell_array(value: Any) -> bool:
    return not isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating))


def prepare_vegetation_law(column: fsl.FloatingSoilColumn, binding: VegetationBinding, *, conductivity_m_s: Any, suction_m: Any, memory_budget_bytes: Any, reserve_bytes: Any = 0,
                           available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                           host_available_bytes_fn: Callable[[], int] | None = None, host_active_mask: np.ndarray | None = None,
                           expected_matrix_op_sha256: Any = None, expected_preferential_op_sha256: Any = None) -> PreparedVegetationLaw:
    """Prepare the real native operators for `column` from `binding`. `conductivity_m_s` and `suction_m` are the EXPLICIT matrix K and suction (scalar or exact (N,) arrays in the COLUMN's namespace: host
    arrays for numpy / numba, device arrays for cuda; no texture inference).

    Order: current-device gate (CUDA: before any reader, copy or allocation) -> binding/controls/column checks (layers, cells, active-mask identity) -> composed whole-call admission of the modelled peak
    (`estimate_law_bytes`, caller reserve counted once; CUDA: device and host pools separately) -> g = `reference_vegetation_function(R1, a1, a2)` ONCE on the host -> `prepare_floating_matrix(g)`
    -> the streamed `prepare_floating_preferential` (one LEAF contribution at a time under the same quota) -> seal. Nothing steps, creates wetting history, adopts a lineage or mutates inputs.

    `expected_matrix_op_sha256` / `expected_preferential_op_sha256` (optional, EXTERNAL): the CURRENT native operator identities the caller holds for this column; the operators are recomputed canonically
    here and, when an expectation is given, a mismatch refuses BEFORE the law is minted (the stored/old operator ids of an earlier generation are never the expectation after a property refresh).
    The contribution stream is modelled with LEAF's PUBLIC `projection.stream_estimate(previous_held=)` only (typed unavailable when absent; no declaration parameter exists).
    CUDA (authored, hardware-unrun): `host_active_mask` (the exact (ny, nx) bool mask of the current boundary, equal to the binding's) is required; it is uploaded once (admitted, cold), hashed with the
    column's own `stream_hash(..., b"mask")` recipe and must equal `column.mask_sha256`. g, the per-cell `a3` / max-rate maps and each contribution are admitted cold H2D copies (bounded, counted);
    host `host_*` quotas are required. There is no per-hydraulic-step transfer."""
    fsl._gate(column)  # CUDA: the CURRENT device is checked first, before any reader / copy / allocation
    fsl._check_column(column)
    b = _check_binding(binding)
    ctl = b.controls
    if (column.n_layers, column.n_cells) != (b.n_layers, b.n_cells):
        raise VegetationBindingError(f"column ({column.n_layers} layers, {column.n_cells} cells) != binding ({b.n_layers}, {b.n_cells})")
    if float(column.cell_area_m2) != float(b.cell_area_m2):
        raise VegetationBindingError(f"the native column's cell_area_m2 {column.cell_area_m2!r} is not the binding's {b.cell_area_m2!r}: the LEAF cover fractions are fractions of THAT cell area")
    is_cuda = column.backend == "cuda"
    if is_cuda:
        if host_memory_budget_bytes is None:
            raise VegetationBindingError("a cuda column requires explicit host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn (separate device and host pools; no default)")
        if type(host_active_mask) is not np.ndarray or host_active_mask.dtype != np.bool_ or host_active_mask.shape != b.shape or not host_active_mask.flags.c_contiguous:
            raise VegetationBindingError(f"a cuda column requires the exact host active_mask: a C-contiguous bool array of shape {b.shape}")
        mh = hashlib.sha256()
        _hash_chunked(mh, host_active_mask.reshape(-1).view(np.uint8))
        if mh.hexdigest() != b.mask_sha256:
            raise VegetationBindingError("the host active mask differs from the binding's (LEAF projection) mask")
    else:
        if host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None:
            raise VegetationBindingError("host_* arguments apply only to a cuda column")
        if host_active_mask is not None:
            raise VegetationBindingError("host_active_mask applies only to a cuda column")
        if column.mask_sha256 != b.mask_sha256:
            raise VegetationBindingError("the column's active mask differs from the binding's (LEAF projection) mask")
    for name, v in (("conductivity_m_s", conductivity_m_s), ("suction_m", suction_m)):
        if isinstance(v, (bool, np.bool_)):
            raise VegetationBindingError(f"{name} must be numeric, got a Boolean")
    n_en = len(b.enabled_species)
    exp_m = None if expected_matrix_op_sha256 is None else _hex(expected_matrix_op_sha256, "expected_matrix_op_sha256")
    exp_p = None if expected_preferential_op_sha256 is None else _hex(expected_preferential_op_sha256, "expected_preferential_op_sha256")
    held = _stream_per_next(b, True) if n_en else 0
    free = _stream_per_next(b, False) if n_en else 0
    wstage = _stream_estimate(b, False)["weights_stage_bytes"] if n_en else 0
    kc = sum(int(_is_cell_array(v)) for v in (conductivity_m_s, suction_m))
    kin = _F8 * column.n_cells * kc
    est = estimate_law_bytes(b.n_layers, b.n_cells, n_en, binding_resident_bytes=b.resident_bytes + b.borrowed_bytes, stream_per_next_held_bytes=held, stream_per_next_free_bytes=free,
                             stream_weights_stage_bytes=wstage, backend=column.backend, k_suction_cell_arrays=kc, a3_cell_map=ctl.a3_layout == "cell", max_rate_cell_map=ctl.max_rate_layout == "cell")
    rb = _count(reserve_bytes, "reserve_bytes")
    N = b.n_cells
    if is_cuda:
        from maple_syrup import floating_soil_layers_cuda as fcu

        cp = fcu._rc().require_cuda()
        quota = {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": rb, "available_bytes_fn": available_bytes_fn, "host_memory_budget_bytes": host_memory_budget_bytes,
                 "host_reserve_bytes": host_reserve_bytes, "host_available_bytes_fn": host_available_bytes_fn}
        fcu.admit(cp, column.device_id, est["peak_bytes"], est["peak_host_bytes"], quota, "vegetation law preparation")
        dev_mask = cp.array(host_active_mask.reshape(-1), dtype=np.bool_)  # admitted cold H2D (N bytes)
        if fcu.stream_hash(cp, [("active", dev_mask)], b"mask") != column.mask_sha256:
            raise VegetationBindingError("the host active mask does not reproduce the cuda column's device mask identity")
        del dev_mask
        mat_kw = {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": rb + est["native_reserve_device_matrix"], "available_bytes_fn": available_bytes_fn,
                  "host_memory_budget_bytes": host_memory_budget_bytes, "host_reserve_bytes": _count(host_reserve_bytes, "host_reserve_bytes") + est["native_reserve_host_matrix"],
                  "host_available_bytes_fn": host_available_bytes_fn}
        pref_kw = {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": rb + est["native_reserve_device_preferential"], "available_bytes_fn": available_bytes_fn,
                   "host_memory_budget_bytes": host_memory_budget_bytes, "host_reserve_bytes": _count(host_reserve_bytes, "host_reserve_bytes") + est["native_reserve_host_preferential"],
                   "host_available_bytes_fn": host_available_bytes_fn}
        p1_budget, p1_reserve, p1_reader = host_memory_budget_bytes, _count(host_reserve_bytes, "host_reserve_bytes") + est["p1_stream_reserve"], host_available_bytes_fn
        up = cp.array
    else:
        _admit("prepare_vegetation_law", est["peak_bytes"], memory_budget_bytes, rb, available_bytes_fn)
        mat_kw = {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": rb + est["native_reserve_matrix"], "available_bytes_fn": available_bytes_fn}
        pref_kw = {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": rb + est["native_reserve_preferential"], "available_bytes_fn": available_bytes_fn}
        p1_budget, p1_reserve, p1_reader = memory_budget_bytes, rb + est["p1_stream_reserve"], available_bytes_fn
        up = None
    # ---- matrix: g ONCE on the host from the LEAF R1 through the existing function; handed (CUDA: after ONE admitted cold H2D) to the existing preparer
    g = mi.reference_vegetation_function(_r1_of(b), ctl.a1, ctl.a2)
    h = hashlib.sha256()
    _hash_chunked(h, np.asarray(g))
    g_sha, g_min, g_max = h.hexdigest(), float(np.min(g)) if N else 0.0, float(np.max(g)) if N else 0.0
    g_in = up(np.asarray(g), dtype=np.float64) if is_cuda else g
    if is_cuda:
        del g
    matrix_op = fm.prepare_floating_matrix(column, conductivity_m_s=conductivity_m_s, suction_m=suction_m, vegetation_factor=g_in, **mat_kw)
    del g_in
    # ---- preferential: LEAF contributions streamed one at a time under the same quota
    state = {"count": 0}
    a3_in, mr_in = ctl.a3, ctl.max_intake_rate_m_s
    if is_cuda:
        a3_in = up(ctl.a3, dtype=np.float64) if ctl.a3_layout == "cell" else ctl.a3
        mr_in = up(ctl.max_intake_rate_m_s, dtype=np.float64) if ctl.max_rate_layout == "cell" else ctl.max_intake_rate_m_s
    if n_en:
        ids, weights, gen = _inputs_of(b, p1_budget, p1_reserve, p1_reader)
        if tuple(ids) != b.enabled_species or weights.tobytes() != b.receiver_weights.tobytes():
            gen.close()
            raise VegetationBindingError("the LEAF receiver table changed since binding (stale binding)")
        species_weights = list(weights)
        contributions = _counted_stream(gen, N, state, (lambda c: up(c, dtype=np.float64)) if is_cuda else None)
    else:
        species_weights, contributions = (), None
    pref_op = fpi.prepare_floating_preferential(column, a3=a3_in, max_fraction_per_hour=ctl.max_fraction_per_hour, max_intake_rate_m_s=mr_in,
                                                background_receiver_weights=ctl.background_receiver_weights, species_receiver_weights=species_weights, species_contributions=contributions,
                                                **pref_kw)
    del a3_in, mr_in
    for name, got, want in (("matrix", matrix_op.sha256, exp_m), ("preferential", pref_op.sha256, exp_p)):
        if want is not None and got != want:
            raise VegetationBindingError(f"the recomputed {name} operator identity {got} differs from the expected CURRENT operator identity {want}; the law is not minted")
    if state["count"] != n_en:
        raise VegetationBindingError(f"the native derivation consumed {state['count']} of {n_en} LEAF contributions")
    memory = {"declared": est, "k_suction_cell_input_bytes_excluded": kin, "stream_per_next_held_bytes": held, "stream_per_next_free_bytes": free, "scope": est["scope"]}
    top = hashlib.sha256(_frozen_json({"tag": LAW_TAG, "binding": b.sha256, "column": column.sha256, "backend": column.backend, "g": g_sha, "matrix": matrix_op.sha256,
                                       "preferential": pref_op.sha256, "contributions": state["count"]}).encode())
    with _minting():
        law = PreparedVegetationLaw(matrix_op, pref_op, b.sha256, column.sha256, column.backend, b.origin, column.device_id, g_sha, g_min, g_max, int(state["count"]), top.hexdigest(), matrix_op.sha256,
                                    pref_op.sha256, _frozen_json(memory))
    _LAW_REGISTRY[law] = _fields_fp(law)
    return law


def _mint_cached_binding(**fields: Any) -> VegetationBinding:
    """Mint the cached-derived binding (the owner of the registry mints; the cached-inputs module supplies verified fields). Origin is always the labelled cached recipe."""
    if fields.get("origin") != ORIGIN_CACHED:
        raise VegetationBindingError("only a cached-derived binding is minted here")
    _check_id_tuple(fields.get("species_ids"), "species_ids")
    _check_id_tuple(fields.get("enabled_species"), "enabled_species")
    with _minting():
        b = VegetationBinding(**fields)
    _BND_REGISTRY[b] = _fields_fp(b, BINDING_ID_FIELDS)
    return b
