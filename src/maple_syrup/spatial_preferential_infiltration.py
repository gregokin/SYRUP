"""Fixed explicit SPATIAL preferential (bypass) infiltration TRIAL operator (CPU: NumPy and compiled Numba; CUDA: S2-B2.d, `spatial_preferential_infiltration_cuda`). Phase 2 S2-B2.b / S2-B2.d.

A pure, standalone component on a REAL `spatial_soil_layers.SpatialSoilColumn` / `SpatialSoilState` (sealed owner with retained RAW per-cell maps; not a `SoilColumn`, not a `SoilLayers`, no legacy
operator). The law is EXACTLY the Phase 1 law of `preferential_infiltration` (read its module note): the same ONE aggregate pathway (one soil background vector plus one LEAF receiving-weight vector per
species; normalisation of a positive vector, all-zero disables it; the layer map `g_l = a3 b_l + (1 - a3)/(1 + C) sum_s c_s u_sl` with the closed-form layer sum), the ONE hourly fraction `beta = maxFL2 sum_l g_l`
converted to the continuous rate `k = -log1p(-beta)/3600 s^-1` (`+inf` for `beta = 1`), the isolated attempt `H (-expm1(-k dt))`, ONE aggregate intake cap `max_rate dt` and the allocation TOP LAYER FIRST,
each receiver bounded by its SATURATION room `sat_m[l] - W[l]` (field capacity is a different quantity and is not used) and by the surface water still standing. A rejected receiver share stays on the
surface: it is never redistributed to another layer, there is no per-species debit, no root plasticity and no double count. The derivation of the pathway maps is the common `_derive_pathway` (not a copy);
this module adds no physics and changes no number. The matrix wetting history `J` and the ledger are not touched, and daily simplified recharge is not preferential intake.

    op  = prepare_spatial_preferential(column, a3=..., max_fraction_per_hour=..., max_intake_rate_m_s=..., background_receiver_weights=[0, 1], species_receiver_weights=[[0, 1], ...],
                                        species_contributions=iter_of_(N,)_arrays, memory_budget_bytes=...)
    res = spatial_preferential_step(op, surface_water_m, state, dt_s, memory_budget_bytes=...)                  # SpatialPreferentialStep

Units / inputs: `a3` in [0, 1] and `max_intake_rate_m_s` >= 0 (`+inf` = no cap) are each a host scalar or an exact `(N,)` float64 C-contiguous NumPy array (every cell validated, inactive included, never
retained as a caller array: a per-cell `max_intake_rate_m_s` is copied once into sealed storage, a scalar keeps one value, `a3` is consumed during the build only); `max_fraction_per_hour` a host scalar in [0, 1];
receiving weights are explicit length-L vectors (the existing `ReceiverWeights` records are reused); species contributions are an iterable yielding one exact `(N,)` array `c_s >= 0` per species in
order, consumed lazily and never retained (no `S x N` stack). `surface_water_m` `(N,)` m, `dt_s >= 0` s. The stage preserves `state.t_s` (the owner of the hydraulic step advances time once);
`dt = 0` is the identity for otherwise valid inputs; inactive water and surface are preserved and nothing is allocated to inactive cells. Every cell is validated (typed `PreferentialStepError` with the
common reason text and bit); a refused trial writes nothing and publishes nothing; non-finite global volumes are refused (typed) BEFORE the strict MAPLE bound helper.

Retained per operator (sealed, bytes-backed, read-only; counted): the derived per-cell rate `k` `(N,)`, the receiver map `(L, N)` (a legitimately derived quantity), the max-rate input (one value or `(N,)`)
and the small weight records. No `(L, N)` saturation/tolerance bound tensor, no copy of the mask or geometry, no host mirror.

One physical body per backend: NumPy runs the SAME private body as the common operator (`preferential_infiltration._reference_step_body`) through a private honest context
`(n_layers, n_cells, mask, cell tolerance)` and lazy per-layer readers of the owner's RAW maps (`sat = theta_sat dz`, `cap_tol = coef sat`); Numba runs the SPATIAL compile-time variant of the ONE kernel
body of `preferential_infiltration_numba` (explicit read-only signature, strided raw-map reads, no NumPy fallback).

Binding semantics: the operator identity `sha256` is the digest of (tag, the column's INPUT SHA-256, backend, dimensions, the scalar parameters, the raw weight records and the sealed derived maps),
computed once, cold and streamed in bounded chunks. A state is bound to the column SHA only. NOTHING here is a water-state generation, an accepted-transition receipt, an exactly-once record, a LEAF
adoption, a lifecycle, restart or storm claim; species cover and weights remain prescribed inputs (Phase 2.5 is not implemented). Legacy and spatial operators/states refuse each other (typed). A CUDA
owner (S2-B2.d) needs CuPy and its device current, otherwise it is refused (typed) before any copy, reader call or launch; its maps are private device arrays (logically immutable) and its identity is the same
byte recipe streamed from the device through a counted bounded host staging buffer.

Memory: `estimate_spatial_preferential_memory` (payload model, host; the caller's state, surface and contributions and the owner's retained maps/mask are excluded; interpreter, JIT, allocator and
whole-process memory are excluded; a reading is not a reservation). Admission (explicit quota + available-bytes reading + reserve) precedes every allocation, conversion, compile and NumPy lazy temporary.
"""

from __future__ import annotations

import binascii
import contextlib
import hashlib
import math
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from maple_syrup import preferential_infiltration as pi
from maple_syrup import spatial_matrix_infiltration as sm
from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = ["OPERATOR_TAG", "SPATIAL_PREFERENTIAL_VERSION", "SpatialPreferentialMemory", "SpatialPreferentialOperator", "SpatialPreferentialStep", "estimate_spatial_preferential_memory",
           "prepare_spatial_preferential", "spatial_preferential_step", "verify_operator_identity"]

SPATIAL_PREFERENTIAL_VERSION = "1"
OPERATOR_TAG = "maple_syrup.spatial_preferential.operator.v1"
BACKENDS = ("numpy", "numba", "cuda")
OPERATIONS = ("prepare", "step")
ROW_TEMP_DOUBLES = 6  # (N,) float64 rows of the lazily derived per-layer values a NumPy step may hold live at once (checked by tracemalloc in the tests)
_F8 = 8
WEIGHT_HASH_CHUNK_BYTES = 4096  # raw float64 bytes hexlified per update (8192 hex bytes live at once): O(1) scratch independent of L, counted in the Python allowance

# ---------------------------------------------------------------------------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialPreferentialMemory:
    """Modelled HOST bytes of ONE spatial preferential operation (see the module note for what is excluded)."""

    operation: str
    backend: str
    n_layers: int
    n_cells: int
    n_species: int
    output_bytes: int  # step: surface, water, intake, total; prepare: the retained sealed maps (upper bound) that survive the call
    row_bytes: int  # the 8 dense `(N,)` rows of the common packet reduction (step only)
    scratch_bytes: int  # backend temporaries, masks and the packet scratch (step) | the build's transient arrays and sealing copy (prepare)
    row_temp_bytes: int  # NumPy lazily derived per-layer rows
    local_bytes: int  # Numba per-call local `(L,)` rows (NRT heap; 2 per layer; independent of N)
    weight_bytes: int  # the owned receiver-weight vectors (host): raw + normalised for S species and the background
    packet_bytes: int
    python_allowance_bytes: int
    reserve_bytes: int
    required_bytes: int
    input_bytes: dict
    terms: dict
    scope: str = ("modelled bytes of this call's own arrays (host): the caller's state, surface and contributions and the owner's retained maps/mask are excluded; interpreter, JIT/compiler, allocator "
                  "pool and whole-process memory are excluded; the Numba local rows are counted from the kernel source; an availability reading is not a reservation")
    device_required_bytes: int = 0  # CUDA: the DEVICE requirement (== required_bytes); CPU backends: 0
    host_required_bytes: int = -1  # CUDA: the separate HOST requirement; CPU backends: == required_bytes
    host_reserve_bytes: int = 0

    def __post_init__(self) -> None:
        if self.host_required_bytes < 0:
            object.__setattr__(self, "host_required_bytes", self.required_bytes)
        object.__setattr__(self, "input_bytes", sp._freeze(dict(self.input_bytes)))
        object.__setattr__(self, "terms", sp._freeze(dict(self.terms)))

    def as_dict(self) -> dict:
        return {k: (dict(v) if hasattr(v, "keys") else v) for k, v in self.__dict__.items()}


def _estimate_cuda(operation: str, L: int, N: int, S: int, reserve: int, host_reserve: int) -> SpatialPreferentialMemory:
    from maple_syrup import spatial_preferential_infiltration_cuda as tc

    tc.check_launch_ranges(L, N)
    t = tc.operation_terms(operation, L, N)
    f8 = _F8
    weights = 2 * (S + 1) * L * f8 if operation == "prepare" else 0
    seal = 2 * L * f8 if operation == "prepare" else 0  # the record under construction holds raw + normalised copies (O(L)) beyond the retained weight records
    inputs = {"state.water_m": L * N * f8, "surface_water_m": N * f8} if operation == "step" else {"species_contributions": "streamed, one (N,) caller-owned device array at a time, not retained"}
    device = t["outputs"] + t["scratch"] + reserve
    host = t["host"] + weights + seal + sp.PY_ALLOWANCE_BYTES + host_reserve
    terms = {"backend": "cuda", "operation": operation, "n_blocks": t["n_blocks"], "static_shared_bytes_per_block": t["shared"], "static_shared_bytes_finalize": t["shared_finalize"], "no_dense_rows": True,
             "no_cub_or_cupy_reduction": True, "no_LN_derived_bound_tensor": True, "no_mask_copy": True, "no_host_mirror": True, "no_S_by_N_stack": True,
             "host_packet_buffers": 2 if operation == "step" else 0, "cold_reducer_buffers": 2 if operation == "prepare" else 0,
             "excluded": "NVRTC/module load, CUDA context, allocator pool policy and 512-byte block rounding, interpreter; estimates are payload models (conservative where noted), not reservations"}
    scope = ("modelled bytes of this call's own arrays: DEVICE main requirement in required_bytes/device_required_bytes (retained/outputs + the derivation's or the step's released scratch + device reserve) and the "
             "separate HOST requirement in host_required_bytes (prepare: weight records + the record-under-construction scratch + cold reducer seed/result buffers + hash staging; step: raw 72 B + rebuilt 72 B packets; "
             "+ Python allowance + host reserve); the caller's state/surface/contributions and the owner's retained maps/mask are excluded; allocator rounding, compiler/context and whole-process memory are excluded; "
             "an availability reading is not a reservation")
    return SpatialPreferentialMemory(operation, "cuda", L, N, S, t["outputs"], 0, t["scratch"], 0, 0, weights, t["host"] if operation == "step" else 0, sp.PY_ALLOWANCE_BYTES, reserve, device, inputs, terms, scope,
                                     device, host, host_reserve)


def estimate_spatial_preferential_memory(n_layers: int, n_cells: int, *, backend: str, operation: str = "step", n_species: int = 0, reserve_bytes: int = 0, host_reserve_bytes: int = 0) -> SpatialPreferentialMemory:
    """Pure arithmetic requirement (no allocation).

    step     outputs `(2 N + 2 L N) 8` (surface, total, water, intake) + rows `8 N 8` + scratch (numpy `NUMPY_SCRATCH_DOUBLES_PER_CELL N 8`, numba `KERNEL_SCRATCH_DOUBLES_PER_CELL N 8`; plus
             `SCRATCH_MASK_BYTES_PER_CELL N` and the packet scratch of the common estimate) + NumPy lazy reader rows `6 N 8` | Numba local rows `2 L 8` + the 64 KiB Python allowance + reserve
    prepare  retained maps `(2 N + L N) 8` (rate, max-rate map upper bound, receiver) + the build transients `(4 N + L N) 8` (beta, cover, denominator, layer sum and the sealing copy of the receiver) +
             masks `8 N` + weights `2 (S + 1) L 8` + allowance + reserve (a Numba compile is not counted)
    Caller state/surface/contributions and the owner's retained maps are EXCLUDED (`input_bytes` records them)."""
    if backend not in BACKENDS:
        raise sp.SpatialLayerError(f"backend must be one of {BACKENDS} for a spatial preferential operation, got {backend!r}")
    if operation not in OPERATIONS:
        raise sp.SpatialLayerError(f"operation must be one of {OPERATIONS}, got {operation!r}")
    L, N = sp._count(n_layers, "n_layers", minimum=1), sp._count(n_cells, "n_cells", minimum=1)
    S = sp._count(n_species, "n_species")
    reserve = sp._count(reserve_bytes, "reserve_bytes")
    host_reserve = sp._count(host_reserve_bytes, "host_reserve_bytes")
    if backend != "cuda" and host_reserve:
        raise sp.SpatialLayerError("host_reserve_bytes applies only to the cuda backend")
    if backend == "cuda":
        return _estimate_cuda(operation, L, N, S, reserve, host_reserve)
    allowance = sp.PY_ALLOWANCE_BYTES
    weights = 2 * (S + 1) * L * _F8
    if operation == "prepare":
        retained = (2 * N + L * N) * _F8
        transient = (4 * N + L * N) * _F8 + pi.BUILD_SCRATCH_MASK_BYTES_PER_CELL * N
        required = retained + transient + weights + allowance + reserve
        return SpatialPreferentialMemory("prepare", backend, L, N, S, retained, 0, transient, 0, 0, weights, 0, allowance, reserve, required,
                                         {"species_contributions": "streamed, one (N,) array at a time, not retained", "a3": "read during the build only"},
                                         {"backend": backend, "retained_upper_bound": "(2 N + L N) doubles", "no_S_by_N_stack": True})
    out = (2 * N + 2 * L * N) * _F8
    rows = pi.ROW_COUNT * N * _F8
    per_cell = pi.NUMPY_SCRATCH_DOUBLES_PER_CELL if backend == "numpy" else pi.KERNEL_SCRATCH_DOUBLES_PER_CELL
    scratch = per_cell * N * _F8 + pi.SCRATCH_MASK_BYTES_PER_CELL * N + pi._PACKET_SCRATCH_BYTES
    row_temp = ROW_TEMP_DOUBLES * N * _F8 if backend == "numpy" else 0
    local = 0
    if backend == "numba":
        from maple_syrup import preferential_infiltration_numba as pn

        local = pn.local_scratch_doubles(L, spatial=True) * _F8
    required = out + rows + scratch + row_temp + local + allowance + reserve
    inputs = {"state.water_m": L * N * _F8, "surface_water_m": N * _F8}
    terms = {"backend": backend, "no_LN_derived_bound_tensor": True, "no_mask_copy": True, "no_map_copy": True, "row_temp_doubles": ROW_TEMP_DOUBLES if backend == "numpy" else 0,
             "numba_local_doubles": local // _F8, "packet_length": pi.PACKET_LENGTH, "dense_rows": pi.ROW_COUNT}
    return SpatialPreferentialMemory("step", backend, L, N, S, out, rows, scratch, row_temp, local, 0, _F8 * pi.PACKET_LENGTH, allowance, reserve, required, inputs, terms)


def _admit(column: Any, operation: str, budget: Any, reserve: Any, reader: Callable[[], int] | None, n_species: int = 0, host_budget: Any = None, host_reserve: Any = 0, host_reader: Any = None,
           cp: Any = None) -> SpatialPreferentialMemory:
    """Estimate + admission BEFORE any allocation, conversion, weight record, species consumption, compile or launch. The separate host quota/reserve/reading exist only for a CUDA owner (refused for CPU owners)."""
    hb, hr, hrd = sp._quotas(column, host_budget, host_reserve, host_reader)
    if column.backend == "cuda":
        est = estimate_spatial_preferential_memory(column.n_layers, column.n_cells, backend="cuda", operation=operation, n_species=n_species, reserve_bytes=reserve, host_reserve_bytes=hr)
        sp._admit_cuda_operation(cp, column, est, budget, reader, hb, hrd)
        return est
    est = estimate_spatial_preferential_memory(column.n_layers, column.n_cells, backend=column.backend, operation=operation, n_species=n_species, reserve_bytes=reserve)
    sp._admit_operation(est, budget, reader)
    return est


# ---------------------------------------------------------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------------------------------------------------------
_CONSTRUCT = threading.local()


@contextlib.contextmanager
def _minting():
    _CONSTRUCT.allowed = True
    try:
        yield
    finally:
        _CONSTRUCT.allowed = False


@dataclass(frozen=True, eq=False)
class SpatialPreferentialOperator:
    """Sealed operator minted ONLY by `prepare_spatial_preferential`. It holds the real column object (a reference, not a copy), the sealed derived maps `(rate (N,), max-rate (1,) or (N,), receiver (L, N))`,
    the weight records and metadata; never a host mirror, a bound tensor or a copy of the mask or geometry. `sha256` is the identity described in the module note."""

    column: Any
    backend: str
    n_layers: int
    n_cells: int
    column_sha256: str
    sha256: str
    layouts: Any
    max_fraction_per_hour: float
    n_species: int
    n_species_enabled: int
    background: Any
    species: tuple
    retained_bytes: int
    prepare_memory: SpatialPreferentialMemory
    _law: tuple  # (rate, max rate, receiver): sealed read-only host arrays; CUDA: private device arrays, logically immutable
    tag: str = OPERATOR_TAG
    device_id: int | None = None  # the CUDA owner's device (None for a CPU owner)

    def __post_init__(self) -> None:
        if not getattr(_CONSTRUCT, "allowed", False):
            raise sp.SpatialLayerError("a SpatialPreferentialOperator is created only by prepare_spatial_preferential (no direct construction, replace() or copy)")
        object.__setattr__(self, "layouts", sp._freeze(dict(self.layouts)))

    def __copy__(self):
        raise sp.SpatialLayerError("a SpatialPreferentialOperator is not copyable; prepare another")

    def __deepcopy__(self, memo):
        raise sp.SpatialLayerError("a SpatialPreferentialOperator is not copyable; prepare another")

    def __reduce_ex__(self, protocol):
        raise sp.SpatialLayerError("a SpatialPreferentialOperator is not serialisable (no checkpoint support)")

    def summary(self) -> dict:
        return {"tag": self.tag, "version": SPATIAL_PREFERENTIAL_VERSION, "backend": self.backend, "n_layers": self.n_layers, "n_cells": self.n_cells, "column_sha256": self.column_sha256,
                "sha256": self.sha256, "layouts": dict(self.layouts), "max_fraction_per_hour": self.max_fraction_per_hour, "n_species": self.n_species, "n_species_enabled": self.n_species_enabled,
                "background": None if self.background is None else self.background.summary(), "species": [s.summary() for s in self.species], "retained_bytes": self.retained_bytes,
                "pure_trial": True, "generation_bound": False, "exactly_once": False,
                "law": "continuous preferential pathway of preferential_infiltration (Phase 1), explicit spatial raw readers; no new physics; weights and cover prescribed"}


@dataclass(frozen=True, eq=False)
class SpatialPreferentialStep:
    """`surface_water_m` `(N,)` remaining surface water; `state` a fresh `SpatialSoilState` with the SAME clock; `intake_m` `(L, N)` actual intake per receiver; `intake_total_m` `(N,)`; `budget` the fields and
    meanings of `preferential_infiltration.PreferentialBudget` (volumes over ACTIVE cells, area applied once, MAPLE FP64 bound); `memory` the model of the call."""

    surface_water_m: Any
    state: Any
    intake_m: Any
    intake_total_m: Any
    dt_s: float
    budget: pi.PreferentialBudget
    memory: SpatialPreferentialMemory


# ---------------------------------------------------------------------------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _check_column_object(column: Any) -> None:
    if isinstance(column, ss.SoilColumn):
        raise sp.SpatialLayerError("prepare_spatial_preferential takes a SpatialSoilColumn, not a legacy SoilColumn (use preferential_infiltration.prepare_preferential for the common column)")
    if type(column) is not sp.SpatialSoilColumn:
        raise sp.SpatialLayerError(f"column must be a SpatialSoilColumn, got {type(column).__name__}")
    if column.backend not in BACKENDS:
        raise sp.SpatialLayerBackendUnavailableError(f"unsupported column backend {column.backend!r}")


def _values(value: Any, kind: str, name: str, low: float, high: float, *, allow_inf: bool, n: int) -> tuple[np.ndarray, float | None]:
    """Value validation on EVERY cell. Returns `(read-only (N,) view for the derivation, scalar or None)`; a scalar is a zero-stride broadcast (no copy), a cell array the caller's own (read, never written)."""
    if kind == "scalar":
        v = float(value)
        if math.isnan(v) or (not allow_inf and not math.isfinite(v)):
            raise sp.SpatialLayerError(f"{name} must be finite, got {value!r}")
        if v < low or v > high:
            raise sp.SpatialLayerError(f"{name} must lie in [{low!r}, {high!r}], got {v!r}")
        return np.broadcast_to(np.float64(v), (n,)), v
    if bool(np.isnan(value).any()) or (not allow_inf and not bool(np.isfinite(value).all())):
        raise sp.SpatialLayerError(f"{name} must {'not contain NaN' if allow_inf else 'be finite'} everywhere (inactive cells included)")
    lo, hi = float(value.min()), float(value.max())
    if lo < low or hi > high:
        raise sp.SpatialLayerError(f"{name} must lie in [{low!r}, {high!r}] everywhere (inactive cells included), got [{lo!r}, {hi!r}]")
    return value, None


def _seal_map(array: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(array, dtype=np.float64)
    flat = np.frombuffer(a.tobytes(), dtype=np.float64)  # bytes-backed read-only storage (one counted copy)
    return flat if a.ndim == 1 else flat.reshape(a.shape)


def _hash_weights(h: Any, label: str, rec: Any) -> None:
    """Feed the canonical bytes `label | hex(raw float64 bytes) | NUL` (or `label | none | NUL` for an absent background) into `h` WITHOUT materialising the whole serialisation: the prefix and then the
    hexlified raw bytes in sequential slices of at most `WEIGHT_HASH_CHUNK_BYTES` raw bytes (read-only memoryview slices; one small hex buffer at a time), so the working set is independent of L. The byte
    stream is identical to the former `f"{label}|{raw.tobytes().hex()}".encode() + b"\\0"` for every chunk size."""
    h.update(f"{label}|".encode())
    if rec is None:
        h.update(b"none")
    else:
        view = memoryview(rec.raw).cast("B")
        step = max(int(WEIGHT_HASH_CHUNK_BYTES), 1)
        for start in range(0, len(view), step):
            h.update(binascii.hexlify(view[start:start + step]))
    h.update(b"\0")


def _identity(column: Any, law: tuple, kinds: dict, max_fraction: float, background: Any, species: Sequence[Any], host_buf: np.ndarray | None = None) -> str:
    """The cold identity digest. CPU: streamed from the sealed host arrays. CUDA: the SAME byte recipe (original-shape header + flat bytes) streamed from the device arrays through the counted bounded host
    staging buffer `host_buf`; the weight records are host records on every backend and use the unchanged bounded `_hash_weights`."""
    h = hashlib.sha256()
    for part in (OPERATOR_TAG, column.sha256, column.backend, str((column.n_layers, column.n_cells)), repr(float(max_fraction)), repr(sorted(kinds.items()))):
        h.update(part.encode())
        h.update(b"\0")
    for label, rec in (("background", background), *((f"species{s}", w) for s, w in enumerate(species))):
        _hash_weights(h, label, rec)
    if column.backend == "cuda":
        from maple_syrup import spatial_preferential_infiltration_cuda as tc

        for name, a in zip(("rate_s", "max_rate_m_s", "receiver"), law, strict=True):
            tc.hash_device_array(h, name, "map", a, host_buf)
        return h.hexdigest()
    for name, a in zip(("rate_s", "max_rate_m_s", "receiver"), law, strict=True):
        sm._hash_array(h, name, "map", a)
    return h.hexdigest()


def _prepare_cuda(cp: Any, column: Any, a3: Any, kinds: dict, max_fraction: float, max_intake_rate_m_s: Any, background: Any, species: tuple, species_contributions: Any) -> tuple:
    """CUDA preparation (admission, metadata checks and the weight records were done by the caller): typed module load -> every-cell device validation of the scalar/cell inputs -> ONE device copy of a per-cell
    max rate (a scalar keeps a one-element device array) -> the COMMON derivation (`pi._derive_pathway`, species streamed lazily) with the bounded counted cold reducer -> private logically immutable device arrays
    (no sealing copy, no host mirror) -> bounded streamed identity hash. Returns `(law, n_enabled, sha256)`."""
    from maple_syrup import spatial_preferential_infiltration_cuda as tc

    L, N = column.n_layers, column.n_cells
    with sp._device_errors("preferential preparation"):
        try:
            tc.load_kernels(cp, column.device_id)
        except ss.SubsurfaceBackendUnavailableError as exc:
            raise sp.SpatialLayerBackendUnavailableError(str(exc)) from exc
        views = []
        for value, kind, name, lo, hi in ((a3, kinds["a3"], "a3", 0.0, 1.0), (max_intake_rate_m_s, kinds["max_intake_rate_m_s"], "max_intake_rate_m_s", 0.0, math.inf)):
            if kind == "scalar":
                v = float(value)
                if math.isnan(v) or v < lo or v > hi:
                    raise sp.SpatialLayerError(f"{name} must lie in [{lo!r}, {hi!r}], got {v!r}")
                views.append(cp.array(np.array([v], dtype=np.float64)))  # one-element device array (8 B)
            else:
                if not tc.validate_range(cp, value, lo, hi):
                    raise sp.SpatialLayerError(f"{name} must lie in [{lo!r}, {hi!r}] on every cell (inactive cells included; NaN refused)")
                views.append(value)
        a3_dev, mr_dev = views
        a3_view = cp.broadcast_to(a3_dev, (N,)) if kinds["a3"] == "scalar" else a3_dev  # a zero-copy read-only broadcast of a scalar; a cell array is the caller's own (read, never written)
        mr = mr_dev if kinds["max_intake_rate_m_s"] == "scalar" else cp.array(mr_dev, dtype=np.float64, copy=True)  # ONE device copy of a per-cell max rate
        reducer = tc.ColdReducer(cp, column.device_id)
        ctx = pi._DeriveContext(cp, L, N, column._mask, lambda s_i, c: sp._check_array(f"species_contributions[{s_i}]", c, (N,), column), reducer)
        rate, receiver, beta, n_enabled = pi._derive_pathway(ctx, a3_view, max_fraction, background, species, species_contributions)
        del beta, a3_view, a3_dev
        law = (rate, mr, receiver)
        host_buf = np.empty(min(L * N, tc.STAGE_ELEMS), dtype=np.float64)
        sha = _identity(column, law, kinds, max_fraction, background, species, host_buf)
        del host_buf
    return law, n_enabled, sha


def prepare_spatial_preferential(column: Any, *, a3: Any, max_fraction_per_hour: Any, max_intake_rate_m_s: Any, background_receiver_weights: Any, species_receiver_weights: Sequence[Any] = (),
                                species_contributions: Iterable[Any] | None = None, memory_budget_bytes: Any, reserve_bytes: Any = 0,
                                available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                                host_available_bytes_fn: Callable[[], int] | None = None) -> SpatialPreferentialOperator:
    """Mint the sealed operator for a `SpatialSoilColumn` (backend numpy | numba | cuda). Inputs and meanings as `preferential_infiltration.prepare_preferential` (module note). Order: object/backend/device
    refusals (no copy) -> allocation-free metadata checks -> ADMISSION (derived maps, build transients, weights, allowance, reserve; CUDA: device AND host) -> weight records -> value validation on EVERY cell ->
    the common derivation (species streamed lazily; CUDA: through the bounded counted cold reducer) -> sealing (CPU) | private device arrays (CUDA) -> the one-off compile (startup, outside every hot call) ->
    bounded streamed identity hash -> mint. CUDA: `memory_budget_bytes`/`reserve_bytes`/`available_bytes_fn` are the DEVICE quota/reserve/reading and `host_*` the separate host ones (refused for a CPU owner);
    `a3`/`max_intake_rate_m_s`/contributions are exact cupy arrays of the owner's device (or host scalars)."""
    _check_column_object(column)
    cp = sp._device_cp(column)
    L, N = column.n_layers, column.n_cells
    reserve = sp._count(reserve_bytes, "reserve_bytes")
    max_fraction = sp._real(max_fraction_per_hour, "max_fraction_per_hour")
    if not (0.0 <= max_fraction <= 1.0):
        raise sp.SpatialLayerError(f"max_fraction_per_hour must lie in [0, 1], got {max_fraction!r}")
    if isinstance(species_receiver_weights, (str, bytes)) or not isinstance(species_receiver_weights, Sequence):
        raise sp.SpatialLayerError("species_receiver_weights must be a sequence of per-species length-L vectors")
    S = len(species_receiver_weights)
    if S and species_contributions is None:
        raise sp.SpatialLayerError(f"{S} species were given but species_contributions is None")
    if not S and species_contributions is not None:
        raise sp.SpatialLayerError("species_contributions were given without species_receiver_weights")
    kinds = {"a3": sm._input_kind(column, a3, "a3"), "max_intake_rate_m_s": sm._input_kind(column, max_intake_rate_m_s, "max_intake_rate_m_s")}
    est = _admit(column, "prepare", memory_budget_bytes, reserve, available_bytes_fn, S, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, cp)
    background = None if background_receiver_weights is None else pi.receiver_weights(background_receiver_weights, L, "background_receiver_weights")
    species = tuple(pi.receiver_weights(w, L, f"species_receiver_weights[{s}]") for s, w in enumerate(species_receiver_weights))
    if column.backend == "cuda":
        law, n_enabled, sha = _prepare_cuda(cp, column, a3, kinds, max_fraction, max_intake_rate_m_s, background, species, species_contributions)
        with _minting():
            return SpatialPreferentialOperator(column, "cuda", L, N, column.sha256, sha, kinds, max_fraction, S, n_enabled, background, species, int(sum(a.nbytes for a in law)), est, law, device_id=column.device_id)
    a3_view, _ = _values(a3, kinds["a3"], "a3", 0.0, 1.0, allow_inf=False, n=N)
    mr_view, mr_scalar = _values(max_intake_rate_m_s, kinds["max_intake_rate_m_s"], "max_intake_rate_m_s", 0.0, math.inf, allow_inf=True, n=N)
    ctx = pi._DeriveContext(np, L, N, column._mask, lambda s_i, c: sp._check_array(f"species_contributions[{s_i}]", c, (N,), column))
    rate, receiver, beta, n_enabled = pi._derive_pathway(ctx, a3_view, max_fraction, background, species, species_contributions)
    del beta, a3_view
    sealed_receiver = _seal_map(receiver)
    del receiver
    sealed_rate = _seal_map(rate)
    del rate
    sealed_mr = _seal_map(np.array([mr_scalar], dtype=np.float64) if mr_scalar is not None else mr_view)
    law = (sealed_rate, sealed_mr, sealed_receiver)
    if column.backend == "numba":
        from maple_syrup import preferential_infiltration_numba as pn

        try:
            pn.compiled_spatial_kernel()
        except ss.SubsurfaceBackendUnavailableError as exc:
            raise sp.SpatialLayerBackendUnavailableError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - a compiler defect is NOT an availability fallback
            raise sp.SpatialLayerBackendUnavailableError(f"the compiled spatial preferential kernel failed to build ({type(exc).__name__}: {exc}); there is no NumPy fallback") from exc
    sha = _identity(column, law, kinds, max_fraction, background, species)
    with _minting():
        return SpatialPreferentialOperator(column, column.backend, L, N, column.sha256, sha, kinds, max_fraction, S, n_enabled, background, species, int(sum(a.nbytes for a in law)), est, law)


def verify_operator_identity(op: Any, *, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> bool:
    """Explicit COLD re-hash of the sealed maps against the recorded identity (streamed, bounded; never run per step). A CUDA owner streams its device arrays through a bounded host staging buffer admitted
    against the host quota/reserve/reading (refused for a CPU owner)."""
    _check_operator(op)
    column = op.column
    hb, hr, hrd = sp._quotas(column, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn)
    if column.backend != "cuda":
        return _identity(column, op._law, dict(op.layouts), op.max_fraction_per_hour, op.background, op.species) == op.sha256
    from maple_syrup import spatial_preferential_infiltration_cuda as tc

    stage = min(column.n_layers * column.n_cells, tc.STAGE_ELEMS)
    need = stage * 8 + sp.PY_ALLOWANCE_BYTES + hr
    if hb is not None and need > sp._count(hb, "host_memory_budget_bytes"):
        raise sp.SpatialLayerMemoryError(f"the cold identity verification needs an estimated {need} B on the host but the declared host_memory_budget_bytes is {hb}; nothing was copied")
    available = (hrd if hrd is not None else sp.default_host_available_bytes)()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise sp.SpatialLayerError(f"the host available-bytes reader must return a non-Boolean integer >= 0, got {available!r}")
    if need > int(available):
        raise sp.SpatialLayerMemoryError(f"the cold identity verification needs an estimated {need} B on the host but only {int(available)} B are reported available; nothing was copied")
    with sp._device_errors("identity verification"):
        return _identity(column, op._law, dict(op.layouts), op.max_fraction_per_hour, op.background, op.species, np.empty(stage, dtype=np.float64)) == op.sha256


def _check_operator(op: Any) -> Any:
    """Metadata-only revalidation of the sealed operator (O(1), no hashing, no copy). Returns CuPy for a CUDA owner AFTER the current-device check (nothing is allocated, copied or launched here), else None."""
    if isinstance(op, pi.PreferentialOperator):
        raise sp.SpatialLayerError("a legacy PreferentialOperator cannot be used with spatial states; use prepare_spatial_preferential")
    if type(op) is not SpatialPreferentialOperator:
        raise sp.SpatialLayerError(f"operator must come from prepare_spatial_preferential, got {type(op).__name__}")
    column = op.column
    if type(column) is not sp.SpatialSoilColumn or column.sha256 != op.column_sha256 or column.backend != op.backend or (column.n_layers, column.n_cells) != (op.n_layers, op.n_cells):
        raise sp.SpatialLayerError("the operator no longer matches its column (identity, backend or dimensions); refused")
    if column.backend not in BACKENDS:
        raise sp.SpatialLayerBackendUnavailableError(f"unsupported column backend {column.backend!r}")
    if not isinstance(op._law, tuple) or len(op._law) != 3 or not isinstance(op.sha256, str) or len(op.sha256) != 64:
        raise sp.SpatialLayerError("the operator record carries a replaced map set or identity; refused")
    shapes = ((op.n_cells,), (1,) if op.layouts["max_intake_rate_m_s"] == "scalar" else (op.n_cells,), (op.n_layers, op.n_cells))
    cp = sp._device_cp(column)
    for name, a, shape in zip(("rate_s", "max_rate_m_s", "receiver"), op._law, shapes, strict=True):
        if cp is not None:  # CuPy has no read-only flag: the maps are logically immutable (module-created, const kernel arguments, owner-bound identity)
            if type(a) is not cp.ndarray or a.dtype != np.float64 or tuple(a.shape) != shape or not a.flags.c_contiguous or int(a.device.id) != column.device_id or op.device_id != column.device_id:
                raise sp.SpatialLayerError(f"operator map {name} lost its exact type/dtype/shape/device; refused")
        elif type(a) is not np.ndarray or a.dtype != np.float64 or a.shape != shape or a.flags.writeable or not a.flags.c_contiguous:
            raise sp.SpatialLayerError(f"operator map {name} lost its exact type/dtype/shape/read-only storage; refused")
    return cp


# ---------------------------------------------------------------------------------------------------------------------------------------------
# readers and the step
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _kinds(column: Any) -> dict:
    return {n: v["kind"] for n, v in column.estimate.retained.items()}


def _numpy_readers(column: Any):
    """Lazy per-layer readers of the two values the shared body reads, in the accepted product order (a scalar for a scalar/shared layout, an `(N,)` row for a per-cell layout)."""
    maps, kinds = column._maps, _kinds(column)
    ts, dz = maps["theta_saturation"], maps["dz"]
    kts, kdz = kinds["theta_saturation"], kinds["dz"]
    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    row = sp._retained_row
    return sp._Rows(lambda k: row(ts, kts, k) * row(dz, kdz, k)), sp._Rows(lambda k: coef * (row(ts, kts, k) * row(dz, kdz, k)))


def _run(op: SpatialPreferentialOperator, surface: Any, water: Any, dt: float) -> pi.StepArrays:
    column = op.column
    L, N = op.n_layers, op.n_cells
    cell_tol = pi.cell_closure_coefficient(L)
    rate, max_rate, recv = op._law
    scalar = op.layouts["max_intake_rate_m_s"] == "scalar"
    if op.backend == "numba":
        from maple_syrup import preferential_infiltration_numba as pn

        maps, kinds = column._maps, _kinds(column)
        return pn.spatial_launch(surface, water, column._mask.view(np.uint8), dt, rate=rate, max_rate=max_rate, k_mr=0 if scalar else 1, recv=recv, dz=maps["dz"].reshape(-1),
                                 k_dz=sp._KIND_CODE[kinds["dz"]], theta_sat=maps["theta_saturation"].reshape(-1), k_sat=sp._KIND_CODE[kinds["theta_saturation"]],
                                 coef=volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0), cell_tol=cell_tol)
    sat, cap = _numpy_readers(column)
    ctx = pi._StepContext(L, N, column._mask, cell_tol)
    return pi._reference_step_body(ctx, rate, np.broadcast_to(max_rate, (N,)) if scalar else max_rate, recv, sat, cap, surface, water, dt)


def spatial_preferential_step(op: Any, surface_water_m: Any, state: Any, dt_s: Any, *, memory_budget_bytes: Any, reserve_bytes: Any = 0,
                              available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                              host_available_bytes_fn: Callable[[], int] | None = None) -> SpatialPreferentialStep:
    """One pure preferential stage over `dt_s` (module note). `surface_water_m` `(N,)` is the standing surface water of every cell, `state` a `SpatialSoilState` of the operator's column. Order: operator ->
    state/surface metadata -> dt -> admission -> allocation -> body -> ONE packet reduction -> reject or publish. Inputs are read only; the clock is preserved; `dt = 0` is the identity for valid inputs;
    a refusal publishes nothing. CUDA owner: exact `cupy.ndarray`s of the owner's device (current device checked first), nothing transferred, results stay on the device, ONE 72-byte raw packet is read back
    (and rebuilt into the unchanged 9-float packet), `memory_budget_bytes`/`reserve_bytes`/`available_bytes_fn` are the DEVICE quota/reserve/reading with the separate `host_*` ones (refused for a CPU owner)."""
    cp = _check_operator(op)
    column = op.column
    t_s = sp._check_spatial_state(column, state)
    dt = ss._check_dt(dt_s)
    sp._check_array("surface_water_m", surface_water_m, (column.n_cells,), column)
    est = _admit(column, "step", memory_budget_bytes, sp._count(reserve_bytes, "reserve_bytes"), available_bytes_fn, op.n_species, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, cp)
    if cp is not None:
        from maple_syrup import spatial_preferential_infiltration_cuda as tc

        try:
            with sp._device_errors("preferential step"):
                surface_o, water_o, intake_o, total_o, packet = tc.step(cp, op, surface_water_m, state.water_m, dt)
        except ss.SubsurfaceBackendUnavailableError as exc:
            raise sp.SpatialLayerBackendUnavailableError(str(exc)) from exc
        budget = pi._resolve_core(column.cell_area_m2, op.n_layers, op.n_cells, packet)
        return SpatialPreferentialStep(surface_o, sp._new_state(column, water_o, t_s), intake_o, total_o, dt, budget, est)
    arrays = _run(op, surface_water_m, state.water_m, dt)
    packet = pi._read_packet(np, arrays)
    budget = pi._resolve_core(column.cell_area_m2, op.n_layers, op.n_cells, packet)
    return SpatialPreferentialStep(arrays.surface_m, sp._new_state(column, arrays.water_m, t_s), arrays.intake_m, arrays.intake_total_m, dt, budget, est)
