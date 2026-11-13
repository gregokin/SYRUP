"""Fixed explicit SPATIAL persistent Green-Ampt MATRIX infiltration TRIAL operator and history (CPU: NumPy and compiled Numba; CUDA: S2-B2.c, `spatial_matrix_infiltration_cuda`). Phase 2 S2-B2.a / S2-B2.c.

A pure, standalone component on a REAL `spatial_soil_layers.SpatialSoilColumn` / `SpatialSoilState` (sealed owner with retained RAW per-cell maps; not a `SoilColumn`, not a `SoilLayers`, no legacy
operator). The law is EXACTLY the Phase 1 law of `matrix_infiltration` (read its module note): the same equations, branch order, thresholds, Newton helpers/tolerance/iteration cap, flags, cell and
global conservation and area-once volumes; this module adds no physics and changes no number. Per active cell the top layer supplies `b = psi * (max(fc_m0 - W0, 0) / dz0)` (`fc_m0 = theta_fc0 * dz0`
rounded first; the deficit is evaluated in exactly that order), the matrix room is `max(sat_m0 - W0, 0)` (SATURATION; the FIELD-CAPACITY gravity threshold is a different quantity and is not executed
here), unponded/pre-onset admitted intake advances the persistent hydraulic history `J` UNSCALED, ponded admitted post-onset intake advances `J` by `post / g` for `g > 0` (nothing for `g = 0`), and the
separate lifetime ledger adds the ACTUAL `pre + post`; rejected capacity never advances `J`. Daily simplified recharge is NOT matrix intake and does not touch `J`.

    op   = prepare_spatial_matrix(column, conductivity_m_s=K, suction_m=psi, vegetation_factor=g, memory_budget_bytes=...)     # K/psi/g: host scalar or exact (N,) float64 array
    hist = spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=...)                        # EXPLICIT values: nothing is defaulted
    res  = spatial_matrix_step(op, surface_pool_m, standing_before_m, incoming_rate_m_s, state, hist, dt_s, memory_budget_bytes=...)   # SpatialMatrixStep

Units/clock: pool, standing depth `[m]`, incoming-rate descriptor `[m/s]`, `dt_s >= 0` seconds, `K [m/s]`, `psi [m]`, `g` dimensionless in `[0, 1]`, history `[m]`. The stage preserves `state.t_s` (the owner
of the hydraulic step advances time once); `dt = 0` is the identity for otherwise valid inputs (history preserved); supply convention B exactly as the common stage. Every cell is validated, inactive
ones included (typed `MatrixStepError` with the common reason text and bit, never a clipped or partial result); a refused trial never writes the state, pool, standing, rate, history, maps or mask and
publishes nothing. Non-finite global quantities are refused (typed) BEFORE the strict MAPLE bound helper.

One physical body per backend: NumPy runs the SAME private body as the common operator (`matrix_infiltration._reference_step_body`) through a private honest context `(n_layers, n_cells, mask, cell tolerance,
Newton tolerance)` and lazy per-layer readers of the owner's RAW maps (accepted product order `fc = theta_fc dz`, `sat = theta_sat dz`, `cap_tol = coef sat`, `coef = volume_roundoff_bound_m3(LAYER_OPERATION_TERMS,
1.0)`); Numba runs the SPATIAL compile-time variant of the ONE kernel body of `matrix_infiltration_numba` (explicit read-only signature, strided raw-map reads, no NumPy fallback). Neither backend builds an
`(L, N)` bound tensor or copies the owner's maps or mask. The Newton helpers (`_ga_increment_vec` / `ga_increment`, tolerance for 8 terms, 64 iterations, same start) are the common ones, unchanged.

Binding semantics (read before relying on them): the operator identity `sha256` is the digest of (tag, the column's INPUT SHA-256, backend, dimensions, and each law input's name, layout and bytes), computed
once, cold and streamed in bounded chunks. A history and a state are bound to inputs only: two operators prepared from identical inputs on identical columns have the same identity and accept each other's
histories (that is what the binding means); a state additionally binds the column SHA. NOTHING here is a water-state generation, an accepted-transition receipt, an exactly-once record, a LEAF adoption,
a lifecycle, restart or storm claim, and the public history constructor cannot verify that its values come from a genuine run boundary: it only validates and copies explicit arrays. No reset, recovery,
rescaling or automatic zero history exists; any future policy and its accepted-boundary binding belong to the later native integration (B3). Legacy operators/states/histories and these spatial ones refuse each
other (typed). A CUDA owner (S2-B2.c) needs CuPy and its device current; otherwise it is refused (typed) before any copy, validation, reader call or launch; its law/history arrays are private device arrays (logically immutable), its identity recipe is the same byte stream streamed from the device through a counted host staging buffer.

Memory: `estimate_spatial_matrix_memory` (payload model, host; the caller's state, forcing and history and the owner's retained maps/mask are excluded; interpreter, JIT, allocator and whole-process memory are
excluded; a reading is not a reservation). Admission (explicit quota + available-bytes reading + reserve) precedes every allocation, conversion, compile and NumPy lazy temporary, on every path.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from maple_syrup import matrix_infiltration as mi
from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = ["HISTORY_TAG", "OPERATOR_TAG", "SPATIAL_MATRIX_VERSION", "SpatialMatrixHistory", "SpatialMatrixMemory", "SpatialMatrixOperator", "SpatialMatrixStep",
           "estimate_spatial_matrix_memory", "prepare_spatial_matrix", "spatial_matrix_history", "spatial_matrix_step", "verify_operator_identity"]

SPATIAL_MATRIX_VERSION = "1"
OPERATOR_TAG = "maple_syrup.spatial_matrix.operator.v1"
HISTORY_TAG = "maple_syrup.spatial_matrix.history.v1"
BACKENDS = ("numpy", "numba", "cuda")
OPERATIONS = ("prepare", "history", "step")
LAW_NAMES = ("conductivity_m_s", "suction_m", "vegetation_factor")
ROW_TEMP_DOUBLES = 6  # (N,) float64 rows of the lazily derived per-layer values a NumPy step may hold live at once (checked by tracemalloc in the tests)
_F8 = 8
_HASH_CHUNK = sp.DEFAULT_HASH_CHUNK_BYTES

# ---------------------------------------------------------------------------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialMatrixMemory:
    """Modelled HOST bytes of ONE spatial matrix operation (see the module note for what is excluded)."""

    operation: str
    backend: str
    n_layers: int
    n_cells: int
    output_bytes: int  # arrays that survive the call (step: pool, water, J, ledger, intake; history: the two copies; prepare: the sealed law maps, upper bound with all three per cell)
    row_bytes: int  # the 16 dense `(N,)` rows of the common packet reduction (step only)
    scratch_bytes: int  # backend temporaries, validation masks and the packet scratch of the common estimate
    row_temp_bytes: int  # NumPy lazily derived per-layer rows
    local_bytes: int  # Numba per-call local `(L,)` rows (NRT heap; 2 per layer; independent of N)
    packet_bytes: int
    python_allowance_bytes: int
    reserve_bytes: int
    required_bytes: int
    input_bytes: dict
    terms: dict
    scope: str = ("modelled bytes of this call's own arrays (host): the caller's state, forcing and history and the owner's retained maps/mask are excluded; interpreter, JIT/compiler, allocator pool and "
                  "whole-process memory are excluded; the Numba local rows are counted from the kernel source; an availability reading is not a reservation")
    device_required_bytes: int = 0  # CUDA: the DEVICE requirement (== required_bytes); CPU backends: 0
    host_required_bytes: int = -1  # CUDA: the separate HOST requirement (staging + packets + Python allowance + host reserve); CPU backends: == required_bytes
    host_reserve_bytes: int = 0

    def __post_init__(self) -> None:
        if self.host_required_bytes < 0:
            object.__setattr__(self, "host_required_bytes", self.required_bytes)
        object.__setattr__(self, "input_bytes", sp._freeze(dict(self.input_bytes)))
        object.__setattr__(self, "terms", sp._freeze(dict(self.terms)))

    def as_dict(self) -> dict:
        return {k: (dict(v) if hasattr(v, "keys") else v) for k, v in self.__dict__.items()}


def _estimate_cuda(operation: str, L: int, N: int, reserve: int, host_reserve: int) -> SpatialMatrixMemory:
    from maple_syrup import spatial_matrix_infiltration_cuda as tc

    tc.check_launch_ranges(L, N)
    t = tc.operation_terms(operation, L, N)
    f8 = 8
    inputs = ({"state.water_m": L * N * f8, "surface_pool_m": N * f8, "standing_before_m": N * f8, "incoming_rate_m_s": N * f8, "history": 2 * N * f8} if operation == "step" else {})
    device = t["outputs"] + t["scratch"] + reserve
    host = t["host"] + sp.PY_ALLOWANCE_BYTES + host_reserve
    terms = {"backend": "cuda", "operation": operation, "n_blocks": t["n_blocks"], "static_shared_bytes_per_block": t["shared"], "static_shared_bytes_finalize": t["shared_finalize"], "no_dense_rows": True,
             "no_cub_or_cupy_reduction": True, "no_LN_derived_bound_tensor": True, "no_mask_copy": True, "no_host_mirror": True, "host_packet_buffers": 2 if operation == "step" else 0,
             "excluded": "NVRTC/module load, CUDA context, allocator pool policy and 512-byte block rounding, interpreter; estimates are payload models, not reservations"}
    scope = ("modelled bytes of this call's own arrays: DEVICE main requirement in required_bytes/device_required_bytes (outputs + released scratch + device reserve) and the separate HOST requirement in "
             "host_required_bytes (raw 136 B + reconstructed 136 B packets or the staging buffer + Python allowance + host reserve); the caller's state/forcing/history and the owner's retained maps/mask are excluded; "
             "allocator rounding, compiler/context and whole-process memory are excluded; an availability reading is not a reservation")
    return SpatialMatrixMemory(operation, "cuda", L, N, t["outputs"], 0, t["scratch"], 0, 0, t["host"], sp.PY_ALLOWANCE_BYTES, reserve, device, inputs, terms, scope, device, host, host_reserve)


def estimate_spatial_matrix_memory(n_layers: int, n_cells: int, *, backend: str, operation: str = "step", reserve_bytes: int = 0, host_reserve_bytes: int = 0) -> SpatialMatrixMemory:
    """Pure arithmetic requirement (no allocation).

    step     outputs `(4 N + L N) 8` (pool, J, ledger, intake and water) + rows `16 N 8` + scratch (numpy: `48 N 8` + `16 N` + 4 packets; numba: `2 N 8` + `16 N` + 4 packets, the common estimate)
             + NumPy lazy reader rows `6 N 8` | Numba local rows `2 L 8` + the 64 KiB Python allowance + reserve
    history  the two fresh `(N,)` arrays `2 N 8` + two validation masks `2 N` + allowance + reserve
    prepare  the sealed law maps `3 N 8` (UPPER BOUND: all three inputs per cell; scalar inputs retain 8 B each) + validation masks `2 N` + allowance + reserve (a Numba compile is not counted)
    Caller state/forcing/history and the owner's retained maps are EXCLUDED (`input_bytes` records them)."""
    if backend not in BACKENDS:
        raise sp.SpatialLayerError(f"backend must be one of {BACKENDS} for a spatial matrix operation, got {backend!r}")
    if operation not in OPERATIONS:
        raise sp.SpatialLayerError(f"operation must be one of {OPERATIONS}, got {operation!r}")
    L, N = sp._count(n_layers, "n_layers", minimum=1), sp._count(n_cells, "n_cells", minimum=1)
    reserve = sp._count(reserve_bytes, "reserve_bytes")
    host_reserve = sp._count(host_reserve_bytes, "host_reserve_bytes")
    if backend != "cuda" and host_reserve:
        raise sp.SpatialLayerError("host_reserve_bytes applies only to the cuda backend")
    if backend == "cuda":
        return _estimate_cuda(operation, L, N, reserve, host_reserve)
    allowance = sp.PY_ALLOWANCE_BYTES
    packet = _F8 * mi.PACKET_LENGTH
    if operation == "prepare":
        out, scratch = 3 * N * _F8, 2 * N
        inputs: dict = {}
        required = out + scratch + allowance + reserve
        return SpatialMatrixMemory("prepare", backend, L, N, out, 0, scratch, 0, 0, 0, allowance, reserve, required, inputs, {"backend": backend, "retained_upper_bound": "3 N doubles"})
    if operation == "history":
        out, scratch = 2 * N * _F8, 2 * N
        required = out + scratch + allowance + reserve
        return SpatialMatrixMemory("history", backend, L, N, out, 0, scratch, 0, 0, 0, allowance, reserve, required, {"wetting_m": N * _F8, "cumulative_intake_m": N * _F8}, {"backend": backend})
    out = (4 * N + L * N) * _F8
    rows = mi.ROW_COUNT * N * _F8
    per_cell = mi.NUMPY_SCRATCH_DOUBLES_PER_CELL if backend == "numpy" else mi.KERNEL_SCRATCH_DOUBLES_PER_CELL
    scratch = per_cell * N * _F8 + mi.SCRATCH_MASK_BYTES_PER_CELL * N + mi._PACKET_SCRATCH_BYTES
    row_temp = ROW_TEMP_DOUBLES * N * _F8 if backend == "numpy" else 0
    local = 0
    if backend == "numba":
        from maple_syrup import matrix_infiltration_numba as mn

        local = mn.local_scratch_doubles(L, spatial=True) * _F8
    required = out + rows + scratch + row_temp + local + allowance + reserve
    inputs = {"state.water_m": L * N * _F8, "surface_pool_m": N * _F8, "standing_before_m": N * _F8, "incoming_rate_m_s": N * _F8, "history": 2 * N * _F8}
    terms = {"backend": backend, "no_LN_derived_bound_tensor": True, "no_mask_copy": True, "no_map_copy": True, "row_temp_doubles": ROW_TEMP_DOUBLES if backend == "numpy" else 0,
             "numba_local_doubles": local // _F8, "packet_length": mi.PACKET_LENGTH, "dense_rows": mi.ROW_COUNT}
    return SpatialMatrixMemory("step", backend, L, N, out, rows, scratch, row_temp, local, packet, allowance, reserve, required, inputs, terms)


def _admit(column: Any, operation: str, budget: Any, reserve: Any, reader: Callable[[], int] | None, host_budget: Any = None, host_reserve: Any = 0, host_reader: Any = None,
           cp: Any = None) -> SpatialMatrixMemory:
    """Estimate + admission BEFORE any allocation, conversion, compile or launch. The separate host quota/reserve/reading exist only for a CUDA owner (refused for CPU owners)."""
    hb, hr, hrd = sp._quotas(column, host_budget, host_reserve, host_reader)
    if column.backend == "cuda":
        est = estimate_spatial_matrix_memory(column.n_layers, column.n_cells, backend="cuda", operation=operation, reserve_bytes=reserve, host_reserve_bytes=hr)
        sp._admit_cuda_operation(cp, column, est, budget, reader, hb, hrd)
        return est
    est = estimate_spatial_matrix_memory(column.n_layers, column.n_cells, backend=column.backend, operation=operation, reserve_bytes=reserve)
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


def _no_direct(name: str) -> None:
    if not getattr(_CONSTRUCT, "allowed", False):
        raise sp.SpatialLayerError(f"a {name} is created only by its factory function (no direct construction, replace() or copy)")


@dataclass(frozen=True, eq=False)
class SpatialMatrixOperator:
    """Sealed operator minted ONLY by `prepare_spatial_matrix`. It holds the real column object (a reference, not a copy), the law inputs in minimal sealed form (a one-element array for a scalar
    input, the `(N,)` bytes-backed read-only array for a per-cell input; never an `(L, N)` tensor, never a host mirror, never a copy of the mask or geometry) and metadata. `sha256` is the identity of
    (tag, column input SHA-256, backend, dimensions, law input names/layouts/bytes); see the module note for what the binding does and does not mean."""

    column: Any
    backend: str
    n_layers: int
    n_cells: int
    column_sha256: str
    sha256: str
    layouts: Any
    retained_bytes: int
    prepare_memory: SpatialMatrixMemory
    _law: tuple  # (conductivity, suction, vegetation) sealed read-only float64 arrays, shape (1,) scalar | (N,) cell (CUDA: private device arrays, logically immutable)
    tag: str = OPERATOR_TAG
    device_id: int | None = None  # the CUDA owner's device (None for a CPU owner)

    def __post_init__(self) -> None:
        _no_direct("SpatialMatrixOperator")
        object.__setattr__(self, "layouts", sp._freeze(dict(self.layouts)))

    def __copy__(self):
        raise sp.SpatialLayerError("a SpatialMatrixOperator is not copyable; prepare another")

    def __deepcopy__(self, memo):
        raise sp.SpatialLayerError("a SpatialMatrixOperator is not copyable; prepare another")

    def __reduce_ex__(self, protocol):
        raise sp.SpatialLayerError("a SpatialMatrixOperator is not serialisable (no checkpoint support)")

    def summary(self) -> dict:
        return {"tag": self.tag, "version": SPATIAL_MATRIX_VERSION, "backend": self.backend, "n_layers": self.n_layers, "n_cells": self.n_cells, "column_sha256": self.column_sha256,
                "sha256": self.sha256, "law_layouts": dict(self.layouts), "retained_bytes": self.retained_bytes, "pure_trial": True, "generation_bound": False, "exactly_once": False,
                "law": "persistent Green-Ampt matrix infiltration of matrix_infiltration (Phase 1), explicit spatial raw readers; no new physics"}


@dataclass(frozen=True, eq=False)
class SpatialMatrixHistory:
    """Persistent per-cell hydraulic history `wetting_m` (J, m; NOT a water store) and the separate lifetime ACTUAL matrix-intake ledger `cumulative_intake_m` (m): `(N,)` float64 arrays owned by the
    record, bound to the operator INPUT identity they were created for (see the module note). Created only by `spatial_matrix_history` or by a step; never written."""

    wetting_m: Any
    cumulative_intake_m: Any
    operator_sha256: str
    n_cells: int
    tag: str = HISTORY_TAG
    namespace: str = "numpy"  # "numpy" | "cupy" (CPU histories are unchanged)
    device_id: int | None = None

    def __post_init__(self) -> None:
        _no_direct("SpatialMatrixHistory")

    def __copy__(self):
        raise sp.SpatialLayerError("a SpatialMatrixHistory is not copyable; create another with spatial_matrix_history")

    def __deepcopy__(self, memo):
        raise sp.SpatialLayerError("a SpatialMatrixHistory is not copyable; create another with spatial_matrix_history")

    def __reduce_ex__(self, protocol):
        raise sp.SpatialLayerError("a SpatialMatrixHistory is not serialisable (no checkpoint support)")

    def summary(self) -> dict:
        return {"tag": self.tag, "operator_sha256": self.operator_sha256, "n_cells": self.n_cells, "bytes": int(self.wetting_m.nbytes + self.cumulative_intake_m.nbytes),
                "pure_trial": True, "generation_bound": False, "boundary_enforced": False}


@dataclass(frozen=True, eq=False)
class SpatialMatrixStep:
    """`surface_pool_m` `(N,)` remaining pool; `state` a fresh `SpatialSoilState` with the SAME clock; `history` the advanced record; `intake_m` `(N,)` actual intake (0 on inactive cells); `budget` the
    fields and meanings of `matrix_infiltration.MatrixBudget` (volumes over ACTIVE cells, area applied once, MAPLE FP64 bound); `memory` the model of the call."""

    surface_pool_m: Any
    state: Any
    history: SpatialMatrixHistory
    intake_m: Any
    dt_s: float
    budget: mi.MatrixBudget
    memory: SpatialMatrixMemory


# ---------------------------------------------------------------------------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _check_column_object(column: Any) -> None:
    if isinstance(column, ss.SoilColumn):
        raise sp.SpatialLayerError("prepare_spatial_matrix takes a SpatialSoilColumn, not a legacy SoilColumn (use matrix_infiltration.prepare_matrix for the common column)")
    if type(column) is not sp.SpatialSoilColumn:
        raise sp.SpatialLayerError(f"column must be a SpatialSoilColumn, got {type(column).__name__}")
    if column.backend not in BACKENDS:
        raise sp.SpatialLayerBackendUnavailableError(f"unsupported column backend {column.backend!r}")


def _input_kind(column: Any, value: Any, name: str) -> str:
    """Allocation-free metadata pre-check of one law input: a real host scalar ('scalar') or an exact `(N,)` float64 C-contiguous ndarray ('cell')."""
    if isinstance(value, (bool, np.bool_)):
        raise sp.SpatialLayerError(f"{name} must be numeric, got a Boolean")
    if sp._is_real_scalar(value):
        return "scalar"
    sp._check_array(name, value, (column.n_cells,), column)
    return "cell"


def _seal(array: np.ndarray) -> np.ndarray:
    a = np.frombuffer(np.ascontiguousarray(array, dtype=np.float64).tobytes(), dtype=np.float64)  # bytes-backed read-only storage (one counted copy)
    return a


def _law_input(value: Any, kind: str, name: str, low: float, high: float) -> np.ndarray:
    if kind == "scalar":
        v = float(value)
        if not math.isfinite(v):
            raise sp.SpatialLayerError(f"{name} must be finite, got {value!r}")
        if v < low or v > high:
            raise sp.SpatialLayerError(f"{name} must lie in [{low!r}, {high!r}], got {v!r}")
        return _seal(np.array([v], dtype=np.float64))
    if not bool(np.isfinite(value).all()):
        raise sp.SpatialLayerError(f"{name} must be finite everywhere (inactive cells included)")
    lo, hi = float(value.min()), float(value.max())
    if lo < low or hi > high:
        raise sp.SpatialLayerError(f"{name} must lie in [{low!r}, {high!r}] everywhere (inactive cells included), got [{lo!r}, {hi!r}]")
    return _seal(value)


def _hash_array(h: Any, name: str, kind: str, a: np.ndarray) -> None:
    h.update(f"{name}|{kind}|{a.shape}|".encode())
    view = memoryview(a).cast("B")
    for start in range(0, len(view), _HASH_CHUNK):  # streamed: no copy, bounded working set
        h.update(view[start:start + _HASH_CHUNK])
    h.update(b"\0")


def _identity(column: Any, law: tuple, kinds: tuple, host_buf: np.ndarray | None = None) -> str:
    """The cold identity digest. CPU: streamed from the sealed host arrays. CUDA: the SAME byte recipe streamed from the device arrays through the counted bounded host staging buffer `host_buf`."""
    h = hashlib.sha256()
    for part in (OPERATOR_TAG, column.sha256, column.backend, str((column.n_layers, column.n_cells))):
        h.update(part.encode())
        h.update(b"\0")
    if column.backend == "cuda":
        from maple_syrup import spatial_matrix_infiltration_cuda as tc

        for name, kind, a in zip(LAW_NAMES, kinds, law, strict=True):
            tc.hash_device_array(h, name, kind, a, host_buf)
        return h.hexdigest()
    for name, kind, a in zip(LAW_NAMES, kinds, law, strict=True):
        _hash_array(h, name, kind, a)
    return h.hexdigest()


def _law_input_cuda(cp: Any, tc: Any, value: Any, kind: str, name: str, low: float, high: float) -> Any:
    """CUDA: validate EVERY element on the device (a scalar on the host), then make ONE device copy (a scalar keeps a one-element device array). A refusal returns nothing."""
    if kind == "scalar":
        v = float(value)
        if not math.isfinite(v):
            raise sp.SpatialLayerError(f"{name} must be finite, got {value!r}")
        if v < low or v > high:
            raise sp.SpatialLayerError(f"{name} must lie in [{low!r}, {high!r}], got {v!r}")
        return cp.array(np.array([v], dtype=np.float64))
    if not tc.validate_range(cp, value, low, high):
        raise sp.SpatialLayerError(f"{name} must be finite and lie in [{low!r}, {high!r}] on every cell (inactive cells included)")
    return cp.array(value, dtype=np.float64, copy=True)


def prepare_spatial_matrix(column: Any, *, conductivity_m_s: Any, suction_m: Any, vegetation_factor: Any, memory_budget_bytes: Any, reserve_bytes: Any = 0,
                          available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                          host_available_bytes_fn: Callable[[], int] | None = None) -> SpatialMatrixOperator:
    """Mint the sealed operator for a `SpatialSoilColumn` (backend numpy | numba | cuda). `conductivity_m_s` (K >= 0), `suction_m` (psi >= 0) and `vegetation_factor` (g in [0, 1]) are each an explicit host
    scalar or an exact `(N,)` float64 array of the owner's namespace (NumPy for a CPU owner, CuPy on the owner's device for a CUDA owner; no texture inference, no default). Order: object/backend/device
    refusals (no copy) -> allocation-free metadata checks -> ADMISSION -> value validation on EVERY cell and sealing (one copy of each per-cell input; a scalar keeps one value) -> the one-off compile
    (Numba function or CUDA module of the owner's device; startup, outside every hot call) -> bounded streamed identity hash -> mint. The geometry stays in the owner: nothing of it is copied.
    CUDA: `memory_budget_bytes`/`reserve_bytes`/`available_bytes_fn` are the DEVICE quota/reserve/reading and `host_*` the separate host ones (refused for a CPU owner); the cold identity hash streams the
    device arrays through a counted bounded host staging buffer."""
    _check_column_object(column)
    cp = sp._device_cp(column)
    reserve = sp._count(reserve_bytes, "reserve_bytes")
    kinds = tuple(_input_kind(column, v, n) for v, n in zip((conductivity_m_s, suction_m, vegetation_factor), LAW_NAMES, strict=True))
    est = _admit(column, "prepare", memory_budget_bytes, reserve, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, cp)
    bounds = ((0.0, math.inf), (0.0, math.inf), (0.0, 1.0))
    values = (conductivity_m_s, suction_m, vegetation_factor)
    host_buf = None
    if column.backend == "cuda":
        from maple_syrup import spatial_matrix_infiltration_cuda as tc

        with sp._device_errors("matrix preparation"):
            try:
                tc.load_kernels(cp, column.device_id)
            except ss.SubsurfaceBackendUnavailableError as exc:
                raise sp.SpatialLayerBackendUnavailableError(str(exc)) from exc
            law = tuple(_law_input_cuda(cp, tc, v, k, n, lo, hi) for v, k, n, (lo, hi) in zip(values, kinds, LAW_NAMES, bounds, strict=True))
            host_buf = np.empty(min(column.n_cells, tc.STAGE_ELEMS), dtype=np.float64)
            sha = _identity(column, law, kinds, host_buf)
        del host_buf
    else:
        law = tuple(_law_input(v, k, n, lo, hi) for v, k, n, (lo, hi) in zip(values, kinds, LAW_NAMES, bounds, strict=True))
        if column.backend == "numba":
            from maple_syrup import matrix_infiltration_numba as mn

            try:
                mn.compiled_spatial_kernel()
            except ss.SubsurfaceBackendUnavailableError as exc:
                raise sp.SpatialLayerBackendUnavailableError(str(exc)) from exc
            except Exception as exc:  # noqa: BLE001 - a compiler defect is NOT an availability fallback
                raise sp.SpatialLayerBackendUnavailableError(f"the compiled spatial matrix kernel failed to build ({type(exc).__name__}: {exc}); there is no NumPy fallback") from exc
        sha = _identity(column, law, kinds)
    with _minting():
        return SpatialMatrixOperator(column, column.backend, column.n_layers, column.n_cells, column.sha256, sha, dict(zip(LAW_NAMES, kinds, strict=True)),
                                     int(sum(a.nbytes for a in law)), est, law, device_id=column.device_id)


def verify_operator_identity(op: Any, *, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> bool:
    """Explicit COLD re-hash of the sealed law inputs against the recorded identity (streamed, bounded; never run per step). A CUDA owner streams its device arrays through a bounded host staging buffer
    admitted against the host quota/reserve/reading (refused for a CPU owner)."""
    _check_operator(op)
    column = op.column
    hb, hr, hrd = sp._quotas(column, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn)
    kinds = tuple(op.layouts[n] for n in LAW_NAMES)
    if column.backend != "cuda":
        return _identity(column, op._law, kinds) == op.sha256
    from maple_syrup import spatial_matrix_infiltration_cuda as tc

    need = min(column.n_cells, tc.STAGE_ELEMS) * 8 + sp.PY_ALLOWANCE_BYTES + hr
    if hb is not None and need > sp._count(hb, "host_memory_budget_bytes"):
        raise sp.SpatialLayerMemoryError(f"the cold identity verification needs an estimated {need} B on the host but the declared host_memory_budget_bytes is {hb}; nothing was copied")
    available = (hrd if hrd is not None else sp.default_host_available_bytes)()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise sp.SpatialLayerError(f"the host available-bytes reader must return a non-Boolean integer >= 0, got {available!r}")
    if need > int(available):
        raise sp.SpatialLayerMemoryError(f"the cold identity verification needs an estimated {need} B on the host but only {int(available)} B are reported available; nothing was copied")
    with sp._device_errors("identity verification"):
        return _identity(column, op._law, kinds, np.empty(min(column.n_cells, tc.STAGE_ELEMS), dtype=np.float64)) == op.sha256


def _check_operator(op: Any) -> Any:
    """Metadata-only revalidation of the sealed operator (O(1), no hashing, no copy). Returns CuPy for a CUDA owner AFTER the current-device check (nothing is allocated, copied or launched here), else None."""
    if isinstance(op, mi.MatrixOperator):
        raise sp.SpatialLayerError("a legacy MatrixOperator cannot be used with spatial states; use prepare_spatial_matrix")
    if type(op) is not SpatialMatrixOperator:
        raise sp.SpatialLayerError(f"operator must come from prepare_spatial_matrix, got {type(op).__name__}")
    column = op.column
    if type(column) is not sp.SpatialSoilColumn or column.sha256 != op.column_sha256 or column.backend != op.backend or (column.n_layers, column.n_cells) != (op.n_layers, op.n_cells):
        raise sp.SpatialLayerError("the operator no longer matches its column (identity, backend or dimensions); refused")
    if column.backend not in BACKENDS:
        raise sp.SpatialLayerBackendUnavailableError(f"unsupported column backend {column.backend!r}")
    if not isinstance(op._law, tuple) or len(op._law) != 3 or not isinstance(op.sha256, str) or len(op.sha256) != 64:
        raise sp.SpatialLayerError("the operator record carries a replaced law set or identity; refused")
    cp = sp._device_cp(column)
    for name, a in zip(LAW_NAMES, op._law, strict=True):
        shape = (1,) if op.layouts[name] == "scalar" else (op.n_cells,)
        if cp is not None:  # CuPy has no read-only flag: the law arrays are logically immutable (module-created, const kernel arguments, owner-bound identity)
            if type(a) is not cp.ndarray or a.dtype != np.float64 or tuple(a.shape) != shape or not a.flags.c_contiguous or int(a.device.id) != column.device_id or op.device_id != column.device_id:
                raise sp.SpatialLayerError(f"operator law input {name} lost its exact type/dtype/shape/device; refused")
        elif type(a) is not np.ndarray or a.dtype != np.float64 or a.shape != shape or a.flags.writeable or not a.flags.c_contiguous:
            raise sp.SpatialLayerError(f"operator law input {name} lost its exact type/dtype/shape/read-only storage; refused")
    return cp


# ---------------------------------------------------------------------------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _history_input(op: SpatialMatrixOperator, value: Any, name: str) -> tuple[Any, float | None]:
    if isinstance(value, (bool, np.bool_)):
        raise sp.SpatialLayerError(f"{name} must be numeric, got a Boolean")
    if sp._is_real_scalar(value):
        v = float(value)
        if not math.isfinite(v) or v < 0.0:
            raise sp.SpatialLayerError(f"{name} must be finite and >= 0, got {value!r}")
        return None, v
    sp._check_array(name, value, (op.n_cells,), op.column)
    return value, None


def _fresh_history_array(arr: Any, scalar: float | None, n: int, name: str) -> np.ndarray:
    if arr is None:
        out = np.full(n, scalar, dtype=np.float64)
    else:
        if not bool(np.isfinite(arr).all()) or bool((arr < 0.0).any()):
            raise sp.SpatialLayerError(f"{name} must be finite and >= 0 on every cell (inactive included)")
        out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


def _fresh_history_array_cuda(cp: Any, tc: Any, arr: Any, scalar: float | None, n: int, name: str) -> Any:
    """CUDA: a scalar fills a fresh device array; an array is validated on EVERY cell on the device (finite, >= 0) and then copied once. A refusal returns nothing."""
    if arr is None:
        out = cp.empty(n, dtype=np.float64)
        out.fill(scalar)
        return out
    if not tc.validate_range(cp, arr, 0.0, math.inf):
        raise sp.SpatialLayerError(f"{name} must be finite and >= 0 on every cell (inactive included)")
    return cp.array(arr, dtype=np.float64, copy=True)


def spatial_matrix_history(op: Any, *, wetting_m: Any, cumulative_intake_m: Any, memory_budget_bytes: Any, reserve_bytes: Any = 0,
                          available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                          host_available_bytes_fn: Callable[[], int] | None = None) -> SpatialMatrixHistory:
    """EXPLICIT validated history bound to `op.sha256`: `wetting_m` and `cumulative_intake_m` are each a host scalar `>= 0` or an exact `(N,)` float64 array of the owner's namespace (copied) with every cell
    finite and `>= 0` (inactive cells included). Both arguments are REQUIRED: nothing is defaulted to zero. The constructor cannot verify a genuine run boundary and claims no accepted generation; it implements
    no reset, recovery or rescaling. Admitted before any allocation. CUDA: device quota/reserve/reading plus the separate host ones (refused for a CPU owner); the arrays are validated on the device and the
    record is a fresh device pair."""
    cp = _check_operator(op)
    w_arr, w_val = _history_input(op, wetting_m, "wetting_m")
    c_arr, c_val = _history_input(op, cumulative_intake_m, "cumulative_intake_m")
    _admit(op.column, "history", memory_budget_bytes, sp._count(reserve_bytes, "reserve_bytes"), available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, cp)
    if cp is not None:
        from maple_syrup import spatial_matrix_infiltration_cuda as tc

        with sp._device_errors("matrix history construction"):
            try:
                tc.load_kernels(cp, op.column.device_id)
            except ss.SubsurfaceBackendUnavailableError as exc:
                raise sp.SpatialLayerBackendUnavailableError(str(exc)) from exc
            w = _fresh_history_array_cuda(cp, tc, w_arr, w_val, op.n_cells, "wetting_m")
            c = _fresh_history_array_cuda(cp, tc, c_arr, c_val, op.n_cells, "cumulative_intake_m")
        with _minting():
            return SpatialMatrixHistory(w, c, op.sha256, op.n_cells, namespace="cupy", device_id=op.column.device_id)
    w = _fresh_history_array(w_arr, w_val, op.n_cells, "wetting_m")
    c = _fresh_history_array(c_arr, c_val, op.n_cells, "cumulative_intake_m")
    with _minting():
        return SpatialMatrixHistory(w, c, op.sha256, op.n_cells)


def _check_history(op: SpatialMatrixOperator, history: Any) -> None:
    if isinstance(history, mi.WettingHistory):
        raise sp.SpatialLayerError("a legacy WettingHistory cannot be used with a spatial matrix operator; use spatial_matrix_history")
    if type(history) is not SpatialMatrixHistory:
        raise sp.SpatialLayerError(f"history must be a SpatialMatrixHistory, got {type(history).__name__}")
    if history.operator_sha256 != op.sha256 or history.n_cells != op.n_cells:
        raise sp.SpatialLayerError("the history is bound to a different operator identity (other column, K/psi/g inputs or backend); a hydraulic history is never reinterpreted under other inputs; refused")
    want_ns, want_dev = ("cupy", op.column.device_id) if op.backend == "cuda" else ("numpy", None)
    if history.namespace != want_ns or history.device_id != want_dev:
        raise sp.SpatialLayerError(f"the history lives in namespace {history.namespace!r} (device {history.device_id}) but this {op.backend!r} operator needs {want_ns!r} (device {want_dev}); histories are never converted between backends")
    sp._check_array("history.wetting_m", history.wetting_m, (op.n_cells,), op.column)
    sp._check_array("history.cumulative_intake_m", history.cumulative_intake_m, (op.n_cells,), op.column)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# readers and the step
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _kinds(column: Any) -> dict:
    return {n: v["kind"] for n, v in column.estimate.retained.items()}


def _numpy_readers(column: Any):
    """Lazy per-layer readers of the four values the shared body reads, in the accepted product order (a scalar for a scalar/shared layout, an `(N,)` row for a per-cell layout)."""
    maps, kinds = column._maps, _kinds(column)
    tf, ts, dz = maps["theta_field_capacity"], maps["theta_saturation"], maps["dz"]
    ktf, kts, kdz = kinds["theta_field_capacity"], kinds["theta_saturation"], kinds["dz"]
    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    row = sp._retained_row
    return (sp._Rows(lambda k: row(dz, kdz, k)), sp._Rows(lambda k: row(tf, ktf, k) * row(dz, kdz, k)), sp._Rows(lambda k: row(ts, kts, k) * row(dz, kdz, k)),
            sp._Rows(lambda k: coef * (row(ts, kts, k) * row(dz, kdz, k))))


def _law_views(op: SpatialMatrixOperator):
    """NumPy: each law input as an `(N,)` read-only view (a scalar as a zero-stride broadcast of its one value: no copy)."""
    return tuple(np.broadcast_to(a, (op.n_cells,)) if op.layouts[n] == "scalar" else a for n, a in zip(LAW_NAMES, op._law, strict=True))


def _strided(column: Any):
    maps, kinds = column._maps, _kinds(column)
    out = {}
    for name in ("dz", "theta_field_capacity", "theta_saturation"):
        out[name] = (maps[name].reshape(-1), sp._KIND_CODE[kinds[name]])
    return out


def _run(op: SpatialMatrixOperator, pool: Any, standing: Any, rate: Any, water: Any, wet: Any, cum: Any, dt: float) -> mi.StepArrays:
    column = op.column
    L, N = op.n_layers, op.n_cells
    cell_tol = mi.cell_closure_coefficient(L)
    tol = mi.newton_tolerance()
    if op.backend == "numba":
        from maple_syrup import matrix_infiltration_numba as mn

        s = _strided(column)
        k, psi, g = op._law
        strides = tuple(0 if op.layouts[n] == "scalar" else 1 for n in LAW_NAMES)
        return mn.spatial_launch(pool, standing, rate, water, wet, cum, column._mask.view(np.uint8), dt, k=k, k_i=strides[0], psi=psi, p_i=strides[1], g=g, g_i=strides[2],
                                 dz=s["dz"][0], k_dz=s["dz"][1], theta_fc=s["theta_field_capacity"][0], k_fc=s["theta_field_capacity"][1], theta_sat=s["theta_saturation"][0],
                                 k_sat=s["theta_saturation"][1], coef=volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0), cell_tol=cell_tol, tol=tol)
    dz, fc, sat, cap = _numpy_readers(column)
    k, psi, g = _law_views(op)
    ctx = mi._StepContext(L, N, column._mask, cell_tol, tol)
    return mi._reference_step_body(ctx, k, psi, g, dz, fc, sat, cap, pool, standing, rate, water, wet, cum, dt)


def spatial_matrix_step(op: Any, surface_pool_m: Any, standing_before_m: Any, incoming_rate_m_s: Any, state: Any, history: Any, dt_s: Any, *, memory_budget_bytes: Any,
                        reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                        host_available_bytes_fn: Callable[[], int] | None = None) -> SpatialMatrixStep:
    """One pure matrix stage over `dt_s` (supply convention B of `matrix_infiltration`). `surface_pool_m` `(N,)` is the actual pool offered over the interval (supplied once by the owner),
    `standing_before_m` `(N,)` the preexisting standing depth at the interval start (contact descriptor), `incoming_rate_m_s` `(N,)` the onset descriptor only, `state` a `SpatialSoilState` of the
    operator's column, `history` a `SpatialMatrixHistory` bound to `op`. Order: operator -> state/forcing/history metadata -> dt -> admission -> allocation -> body -> ONE packet reduction -> reject
    or publish. Inputs are read only; the clock is preserved; `dt = 0` is the identity for valid inputs; a refusal publishes nothing.
    CUDA owner: every array must be an exact `cupy.ndarray` of the owner's device (current device checked first), nothing is transferred, the result arrays stay on the device, ONE 136-byte raw packet is
    read back (and rebuilt into the unchanged 17-float packet), and `memory_budget_bytes`/`reserve_bytes`/`available_bytes_fn` are the DEVICE quota/reserve/reading with the separate `host_*` ones
    (refused for a CPU owner)."""
    cp = _check_operator(op)
    column = op.column
    t_s = sp._check_spatial_state(column, state)
    dt = ss._check_dt(dt_s)
    n = (column.n_cells,)
    sp._check_array("surface_pool_m", surface_pool_m, n, column)
    sp._check_array("standing_before_m", standing_before_m, n, column)
    sp._check_array("incoming_rate_m_s", incoming_rate_m_s, n, column)
    _check_history(op, history)
    est = _admit(column, "step", memory_budget_bytes, sp._count(reserve_bytes, "reserve_bytes"), available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, cp)
    if cp is not None:
        from maple_syrup import spatial_matrix_infiltration_cuda as tc

        try:
            with sp._device_errors("matrix step"):
                pool_o, water_o, wet_o, cum_o, intake_o, packet = tc.step(cp, op, surface_pool_m, standing_before_m, incoming_rate_m_s, state.water_m, history.wetting_m, history.cumulative_intake_m, dt)
        except ss.SubsurfaceBackendUnavailableError as exc:
            raise sp.SpatialLayerBackendUnavailableError(str(exc)) from exc
        budget = mi._resolve_core(column.cell_area_m2, op.n_layers, op.n_cells, packet)
        new_state = sp._new_state(column, water_o, t_s)
        with _minting():
            new_history = SpatialMatrixHistory(wet_o, cum_o, op.sha256, op.n_cells, namespace="cupy", device_id=column.device_id)
        return SpatialMatrixStep(pool_o, new_state, new_history, intake_o, dt, budget, est)
    arrays = _run(op, surface_pool_m, standing_before_m, incoming_rate_m_s, state.water_m, history.wetting_m, history.cumulative_intake_m, dt)
    packet = mi._read_packet(np, arrays)
    budget = mi._resolve_core(column.cell_area_m2, op.n_layers, op.n_cells, packet)
    new_state = sp._new_state(column, arrays.water_m, t_s)
    arrays.wetting_m.flags.writeable = False
    arrays.cumulative_intake_m.flags.writeable = False
    with _minting():
        new_history = SpatialMatrixHistory(arrays.wetting_m, arrays.cumulative_intake_m, op.sha256, op.n_cells)
    return SpatialMatrixStep(arrays.pool_m, new_state, new_history, arrays.intake_m, dt, budget, est)
