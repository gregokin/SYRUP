"""Persistent Green-Ampt (Tietjen-factor) MATRIX intake on a FLOATING porous column (D2; NumPy and compiled Numba).

A pure trial stage on a real `floating_soil_layers.FloatingSoilColumn` / `FloatingSoilState` (never a `SoilColumn` or `SpatialSoilColumn`; the old classes keep refusing each other). The law is the existing one
(`matrix_infiltration`, supply convention B: supplied pool, standing contact descriptor, onset descriptor; pre-/post-ponding vegetation placement; persistent wetting coordinate `J`) with the ACTUAL porous thickness
of the receiving layer: the top-layer capacity is `theta_sat * h0`, the field-capacity deficit is `max(theta_fc * h0 - W0, 0)` and the suction head `b = psi * (deficit / h0)`, `h = max(0, min(D, e_1) - e_0)`.
The physical body is the SAME function the common and spatial operators call (`matrix_infiltration._reference_step_body`; Numba: the FLOATING compile-time variant of the shared kernel); the readers supply the
floating quantities. Where there is no porous top layer (`h0 = 0`: a wholly-rock column) the receiving capacity is exactly 0, so the actual intake is 0, the offered pool is unchanged, and `J` and the lifetime
actual intake are returned unchanged: the safe stand-in 1.0 replaces the denominator `h0` ONLY in that zero-capacity arithmetic (the deficit there is `0 - 0`), the returned actual thickness stays 0 and a positive `h0`,
however small, is never replaced (no epsilon-to-zero). `J` is a hydraulic COORDINATE, not a water store.

    op   = prepare_floating_matrix(column, conductivity_m_s=..., suction_m=..., vegetation_factor=...)
    hist = floating_matrix_history(op, wetting_m=..., cumulative_intake_m=...)        # explicit; nothing is defaulted
    step = floating_matrix_step(op, pool, standing, rate, state, hist, dt_s)          # FloatingMatrixStep (fresh state, same clock, advanced history)
    rb   = rebind_floating_matrix_history(old_op, old_hist, new_op)                   # explicit pure handoff to a changed operator

INACTIVE cells carry no capacity law (bound `+inf` in the reader; finiteness and `>= 0` are validated on every cell); their water, history and pool are passed through bit for bit and excluded from the active
budgets. History rebinding (`rebind_floating_matrix_history`) is explicit and PURE: it validates the old record against the old operator (identity, shape, namespace, finite and `>= 0` values), requires the new
operator to describe the same physical columns (same dimensions, active mask, nominal edges, cell area, backend), copies the EXACT `J` and cumulative actual-intake arrays under the new operator identity
(changed K, psi, g or soil properties/depth are legitimate; nothing is reset, rescaled or recovered) and returns a receipt; native generation adoption belongs to the later integration slice (E). CUDA owners are
served by the same entry points through `floating_matrix_infiltration_cuda` (current device gate FIRST, metadata, then device/host admission, then the shared body; arrays stay on the device; the host receives
ONE 136-byte raw packet and rebuilds the unchanged 17-float packet; histories and law inputs are validated by a tiny kernel with one 8-byte read each; nothing of the state or the maps is downloaded; the cold identity
hash streams the law arrays through a bounded counted staging buffer).
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import floating_soil_layers as fsl
from maple_syrup import matrix_infiltration as mi
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = ["FLOATING_MATRIX_VERSION", "FloatingMatrixHistory", "FloatingMatrixMemory", "FloatingMatrixOperator", "FloatingMatrixRebinding", "FloatingMatrixStep", "estimate_floating_matrix_memory",
           "floating_matrix_history", "floating_matrix_step", "held_cell_vegetation_factor", "prepare_floating_matrix", "rebind_floating_matrix_history", "verify_operator_identity"]

FLOATING_MATRIX_VERSION = "1"
OPERATOR_TAG = "maple_syrup.floating_matrix.operator.v1"
HISTORY_TAG = "maple_syrup.floating_matrix.history.v1"
LAW_NAMES = ("conductivity_m_s", "suction_m", "vegetation_factor")
ROW_TEMP_DOUBLES = 8
_F8 = 8
_CONSTRUCT = threading.local()


@contextlib.contextmanager
def _minting():
    _CONSTRUCT.allowed = True
    try:
        yield
    finally:
        _CONSTRUCT.allowed = False


def _no_direct(what: str) -> None:
    if not getattr(_CONSTRUCT, "allowed", False):
        raise fsl.FloatingSoilError(f"a {what} is created only by its prepare/history/step/rebinding function (no direct construction, replace() or copy)")


@dataclass(frozen=True)
class FloatingMatrixMemory:
    backend: str
    operation: str
    n_layers: int
    n_cells: int
    outputs_bytes: int
    scratch_bytes: int
    retained_bytes: int
    required_bytes: int
    scope: str = "modelled host payload bytes of this call; caller state/forcing and the owner's retained maps are excluded"


def estimate_floating_matrix_memory(n_layers: int, n_cells: int, *, backend: str = "numpy", operation: str = "step", reserve_bytes: int = 0) -> FloatingMatrixMemory:
    L, N = fsl._count(n_layers, "n_layers", minimum=1), fsl._count(n_cells, "n_cells", minimum=1)
    if backend not in ("numpy", "numba"):
        raise fsl.FloatingBackendUnavailableError(f"this module serves the numpy and numba owners, got {backend!r}")
    if operation not in ("prepare", "history", "rebind", "step"):
        raise fsl.FloatingSoilError(f"operation must be one of prepare/history/rebind/step, got {operation!r}")
    reserve = fsl._count(reserve_bytes, "reserve_bytes")
    law = 3 * N * _F8
    if operation == "prepare":
        return FloatingMatrixMemory(backend, operation, L, N, 0, 0, law, law + reserve)
    if operation in ("history", "rebind"):
        out = 2 * N * _F8
        return FloatingMatrixMemory(backend, operation, L, N, out, N * _F8, 0, out + N * _F8 + reserve)
    outputs = (L * N + 4 * N + mi.ROW_COUNT * N) * _F8
    scratch = (3 * L * N * _F8 if backend == "numpy" else 2 * L * _F8) + ROW_TEMP_DOUBLES * N * _F8 + 4 * N * _F8
    return FloatingMatrixMemory(backend, operation, L, N, outputs, scratch, 0, outputs + scratch + reserve)


@dataclass(frozen=True, eq=False)
class FloatingMatrixOperator:
    """Sealed operator minted ONLY by `prepare_floating_matrix`: the real column (a reference), the law inputs in minimal sealed form (a one-element array for a scalar, an `(N,)` bytes-backed array for a cell
    input) and metadata. `sha256` is the identity of (tag, column identity, backend, dimensions, law layouts and bytes)."""

    column: Any
    backend: str
    n_layers: int
    n_cells: int
    column_sha256: str
    sha256: str
    layouts: Any
    retained_bytes: int
    _law: tuple
    tag: str = OPERATOR_TAG
    device_id: int | None = None  # the CUDA owner's device (None for a CPU owner)

    def __post_init__(self) -> None:
        _no_direct("FloatingMatrixOperator")

    def __copy__(self):
        raise fsl.FloatingSoilError("a FloatingMatrixOperator is not copyable; prepare another")

    def __deepcopy__(self, memo):
        raise fsl.FloatingSoilError("a FloatingMatrixOperator is not copyable; prepare another")

    def __reduce_ex__(self, protocol):
        raise fsl.FloatingSoilError("a FloatingMatrixOperator is not serialisable (no checkpoint support)")

    def summary(self) -> dict:
        return {"tag": self.tag, "version": FLOATING_MATRIX_VERSION, "backend": self.backend, "n_layers": self.n_layers, "n_cells": self.n_cells, "column_sha256": self.column_sha256,
                "sha256": self.sha256, "law_layouts": dict(self.layouts), "retained_bytes": self.retained_bytes, "pure_trial": True, "generation_bound": False, "exactly_once": False,
                "law": "persistent Green-Ampt matrix infiltration of matrix_infiltration on the actual porous thickness; no new physics"}


@dataclass(frozen=True, eq=False)
class FloatingMatrixHistory:
    """Persistent per-cell hydraulic history `wetting_m` (J, m; NOT a water store) and the separate lifetime ACTUAL matrix-intake ledger `cumulative_intake_m` (m): `(N,)` float64 arrays owned by the record,
    bound to the operator identity they were created for."""

    wetting_m: Any
    cumulative_intake_m: Any
    operator_sha256: str
    n_cells: int
    tag: str = HISTORY_TAG
    namespace: str = "numpy"  # "numpy" | "cupy"
    device_id: int | None = None

    def __post_init__(self) -> None:
        _no_direct("FloatingMatrixHistory")

    def __copy__(self):
        raise fsl.FloatingSoilError("a FloatingMatrixHistory is not copyable; create another with floating_matrix_history")

    def __deepcopy__(self, memo):
        raise fsl.FloatingSoilError("a FloatingMatrixHistory is not copyable; create another with floating_matrix_history")

    def __reduce_ex__(self, protocol):
        raise fsl.FloatingSoilError("a FloatingMatrixHistory is not serialisable (no checkpoint support)")

    def summary(self) -> dict:
        return {"tag": self.tag, "operator_sha256": self.operator_sha256, "n_cells": self.n_cells, "bytes": int(self.wetting_m.nbytes + self.cumulative_intake_m.nbytes), "pure_trial": True,
                "generation_bound": False}


@dataclass(frozen=True, eq=False)
class FloatingMatrixStep:
    surface_pool_m: Any
    state: Any
    history: FloatingMatrixHistory
    intake_m: Any
    dt_s: float
    budget: mi.MatrixBudget
    memory: FloatingMatrixMemory


@dataclass(frozen=True)
class FloatingMatrixRebinding:
    """Receipt of one explicit pure history rebinding: the proposed history under the NEW operator identity (exact copies of `J` and the cumulative intake) and the bound identities it relates. Not an accepted
    generation (native adoption is slice E); `reset` is always False."""

    history: FloatingMatrixHistory
    old_operator_sha256: str
    new_operator_sha256: str
    old_column_sha256: str
    new_column_sha256: str
    n_cells: int
    wetting_sum_m: float | None  # None for a CUDA rebinding (nothing is downloaded)
    cumulative_sum_m: float | None
    reset: bool = False
    memory: FloatingMatrixMemory | None = None


# ---------------------------------------------------------------------------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _check_column_object(column: Any) -> None:
    fsl._gate(column)  # a CUDA owner: the CURRENT device is checked first
    fsl._check_column(column)
    if column.backend not in ("numpy", "numba", "cuda"):
        raise fsl.FloatingBackendUnavailableError(f"unsupported backend {column.backend!r}")


def _fcu():
    from maple_syrup import floating_soil_layers_cuda as fcu

    return fcu


def _fmc():
    from maple_syrup import floating_matrix_infiltration_cuda as fmc

    return fmc


def _quota(column: Any, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn) -> dict:
    if column.backend != "cuda" and (host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None):
        raise fsl.FloatingSoilError("host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to the cuda backend")
    return {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": reserve_bytes, "available_bytes_fn": available_bytes_fn, "host_memory_budget_bytes": host_memory_budget_bytes,
                "host_reserve_bytes": host_reserve_bytes, "host_available_bytes_fn": host_available_bytes_fn}


def _admit(column: Any, operation: str, quota: dict, what: str) -> FloatingMatrixMemory:
    """CPU: the modelled host payload against the budget/reading. CUDA: device and host separately (`floating_soil_layers_cuda.admit`)."""
    reserve = fsl._count(quota["reserve_bytes"], "reserve_bytes")
    if column.backend == "cuda":
        t = _fmc().operation_terms(operation, column.n_layers, column.n_cells)
        _fcu().admit(_fcu()._rc().require_cuda(), column.device_id, t["outputs"] + t["scratch"], t["host"], quota, what)
        return FloatingMatrixMemory("cuda", operation, column.n_layers, column.n_cells, t["outputs"], t["scratch"], 0, t["outputs"] + t["scratch"] + t["host"] + reserve)
    est = estimate_floating_matrix_memory(column.n_layers, column.n_cells, backend=column.backend, operation=operation, reserve_bytes=reserve)
    budget = quota["memory_budget_bytes"]
    fsl._admit(est.required_bytes, None if budget is None else fsl._count(budget, "memory_budget_bytes"), quota["available_bytes_fn"], what)
    return est


def _input_kind(column: Any, value: Any, name: str) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise fsl.FloatingSoilError(f"{name} must be numeric, got a Boolean")
    if fsl.sp._is_real_scalar(value):
        return "scalar"
    fsl._check_vec(column, name, value, (column.n_cells,))
    return "cell"


def _seal(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(array, dtype=np.float64).tobytes(), dtype=np.float64)  # bytes-backed read-only storage (one counted copy)


def _law_input(value: Any, kind: str, name: str, low: float, high: float) -> np.ndarray:
    if kind == "scalar":
        v = float(value)
        if not math.isfinite(v):
            raise fsl.FloatingSoilError(f"{name} must be finite, got {value!r}")
        if v < low or v > high:
            raise fsl.FloatingSoilError(f"{name} must lie in [{low!r}, {high!r}], got {v!r}")
        return _seal(np.array([v], dtype=np.float64))
    if not bool(np.isfinite(value).all()):
        raise fsl.FloatingSoilError(f"{name} must be finite on every cell (inactive included)")
    lo, hi = float(value.min()), float(value.max())
    if lo < low or hi > high:
        raise fsl.FloatingSoilError(f"{name} must lie in [{low!r}, {high!r}] on every cell (inactive included), got [{lo!r}, {hi!r}]")
    return _seal(value)


def _hash_array(h: Any, name: str, kind: str, a: np.ndarray) -> None:
    h.update(f"{name}|{kind}|{a.shape}|".encode())
    view = memoryview(a).cast("B")
    for start in range(0, len(view), 1 << 20):  # streamed: no copy, bounded working set
        h.update(view[start:start + (1 << 20)])
    h.update(b"\0")


def _identity(column: Any, law: tuple, kinds: tuple) -> str:
    h = hashlib.sha256()
    for part in (OPERATOR_TAG, column.sha256, column.backend, str((column.n_layers, column.n_cells))):
        h.update(part.encode())
        h.update(b"\0")
    for name, kind, a in zip(LAW_NAMES, kinds, law, strict=True):
        _hash_array(h, name, kind, a)
    return h.hexdigest()


def _law_input_cuda(cp: Any, fmc: Any, column: Any, value: Any, kind: str, name: str, lo: float, hi: float) -> Any:
    """CUDA law input: a scalar is validated on the host and kept as a one-element device array (8 B); a cell array is validated on the DEVICE on every cell (finite, in range) and copied once."""
    if kind == "scalar":
        v = float(value)
        if not math.isfinite(v) or v < lo or v > hi:
            raise fsl.FloatingSoilError(f"{name} must be finite and lie in [{lo!r}, {hi!r}], got {value!r}")
        return cp.array(np.array([v], dtype=np.float64))
    if not fmc.validate_range(cp, column.device_id, value, lo, hi):
        raise fsl.FloatingSoilError(f"{name} must be finite and lie in [{lo!r}, {hi!r}] on every cell (inactive included)")
    return cp.array(value, dtype=np.float64, copy=True)


def _identity_cuda(cp: Any, fmc: Any, column: Any, law: tuple, kinds: tuple) -> str:
    h = hashlib.sha256()
    for part in (OPERATOR_TAG, column.sha256, column.backend, str((column.n_layers, column.n_cells))):
        h.update(part.encode())
        h.update(b"\0")
    host_buf = np.empty(min(column.n_cells, fmc.STAGE_ELEMS), dtype=np.float64)  # the counted bounded cold staging buffer
    for name, kind, a in zip(LAW_NAMES, kinds, law, strict=True):
        fmc.hash_device_array(h, name, kind, a, host_buf)
    return h.hexdigest()


def prepare_floating_matrix(column: fsl.FloatingSoilColumn, *, conductivity_m_s: Any, suction_m: Any, vegetation_factor: Any, memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                            available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                            host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingMatrixOperator:
    """Mint the sealed operator for a FLOATING column (numpy | numba | cuda). `conductivity_m_s` (K >= 0), `suction_m` (psi >= 0) and `vegetation_factor` (g in [0, 1]) are each an explicit host scalar or an exact
    `(N,)` float64 array of the owner's namespace (no texture inference, no default, no interface conductivity). Order: current-device gate (CUDA) -> owner/metadata refusals -> ADMISSION -> value validation on
    EVERY cell and sealing (CUDA: private device arrays) -> the one-off compile -> streamed identity hash (CUDA: bounded counted host staging) -> mint."""
    _check_column_object(column)
    quota = _quota(column, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn)
    kinds = tuple(_input_kind(column, v, n) for v, n in zip((conductivity_m_s, suction_m, vegetation_factor), LAW_NAMES, strict=True))
    est = _admit(column, "prepare", quota, "floating matrix preparation")
    del est
    bounds = ((0.0, math.inf), (0.0, math.inf), (0.0, 1.0))
    values = (conductivity_m_s, suction_m, vegetation_factor)
    if column.backend == "cuda":
        fmc, cp = _fmc(), _fcu()._rc().require_cuda()
        try:
            fmc.load_kernels(cp, column.device_id)
        except ss.SubsurfaceBackendUnavailableError as exc:
            raise fsl.FloatingBackendUnavailableError(str(exc)) from exc
        law = tuple(_law_input_cuda(cp, fmc, column, v, k, n, lo, hi) for v, k, n, (lo, hi) in zip(values, kinds, LAW_NAMES, bounds, strict=True))
        sha = _identity_cuda(cp, fmc, column, law, kinds)
    else:
        law = tuple(_law_input(v, k, n, lo, hi) for v, k, n, (lo, hi) in zip(values, kinds, LAW_NAMES, bounds, strict=True))
        if column.backend == "numba":
            from maple_syrup import matrix_infiltration_numba as mn

            try:
                mn.compiled_floating_kernel()
            except ss.SubsurfaceBackendUnavailableError as exc:
                raise fsl.FloatingBackendUnavailableError(str(exc)) from exc
            except Exception as exc:
                raise fsl.FloatingBackendUnavailableError(f"the compiled floating matrix kernel failed to build ({type(exc).__name__}: {exc}); there is no NumPy fallback") from exc
        sha = _identity(column, law, kinds)
    with _minting():
        return FloatingMatrixOperator(column, column.backend, column.n_layers, column.n_cells, column.sha256, sha, dict(zip(LAW_NAMES, kinds, strict=True)), int(sum(a.nbytes for a in law)), law,
                                      device_id=column.device_id)


def held_cell_vegetation_factor(op: Any) -> Any:
    """READ-ONLY accessor of the operator's HELD per-cell vegetation factor g (Phase 2.5): the sealed `(N,)` float64 law array itself (a reference, no copy, no download; a CUDA operator returns its device array) when the
    operator was prepared with a cell `vegetation_factor`, else None (a scalar law: the caller's configuration scalar stays authoritative and nothing changes). Metadata revalidation only (`_check_operator`)."""
    _check_operator(op)
    return op._law[2] if op.layouts["vegetation_factor"] == "cell" else None


def verify_operator_identity(op: Any) -> bool:
    """Explicit COLD re-hash of the sealed law inputs against the recorded identity (streamed, bounded; never run per step)."""
    column = _check_operator(op)
    kinds = tuple(op.layouts[n] for n in LAW_NAMES)
    if column.backend == "cuda":
        return _identity_cuda(_fcu()._rc().require_cuda(), _fmc(), column, op._law, kinds) == op.sha256
    return _identity(column, op._law, kinds) == op.sha256


def _check_operator(op: Any) -> fsl.FloatingSoilColumn:
    """Metadata-only revalidation (O(1), no hashing, no copy). A CUDA owner: the current device is checked FIRST."""
    if type(op) is not FloatingMatrixOperator:
        raise fsl.FloatingSoilError(f"operator must come from prepare_floating_matrix, got {type(op).__name__}")
    column = op.column
    fsl._gate(column)
    fsl._check_column(column)
    if column.sha256 != op.column_sha256 or column.backend != op.backend or (column.n_layers, column.n_cells) != (op.n_layers, op.n_cells):
        raise fsl.FloatingSoilError("the operator no longer matches its column (identity, backend or dimensions); refused")
    if not isinstance(op._law, tuple) or len(op._law) != 3 or not isinstance(op.sha256, str) or len(op.sha256) != 64:
        raise fsl.FloatingSoilError("the operator record carries a replaced law set or identity; refused")
    for name, a in zip(LAW_NAMES, op._law, strict=True):
        shape = (1,) if op.layouts[name] == "scalar" else (op.n_cells,)
        if column.backend == "cuda":  # CuPy has no read-only flag: the law arrays are logically immutable (module-created, owner-bound identity)
            if type(a) is not fsl._ns_type(column) or a.dtype != np.float64 or tuple(a.shape) != shape or not a.flags.c_contiguous or int(a.device.id) != column.device_id or op.device_id != column.device_id:
                raise fsl.FloatingSoilError(f"operator law input {name} lost its exact type/dtype/shape/device; refused")
        elif type(a) is not np.ndarray or a.dtype != np.float64 or a.shape != shape or a.flags.writeable or not a.flags.c_contiguous:
            raise fsl.FloatingSoilError(f"operator law input {name} lost its exact type/dtype/shape/read-only storage; refused")
    return column


# ---------------------------------------------------------------------------------------------------------------------------------------------
# history and rebinding
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _history_input(op: FloatingMatrixOperator, value: Any, name: str) -> tuple[Any, float | None]:
    if isinstance(value, (bool, np.bool_)):
        raise fsl.FloatingSoilError(f"{name} must be numeric, got a Boolean")
    if fsl.sp._is_real_scalar(value):
        v = float(value)
        if not math.isfinite(v) or v < 0.0:
            raise fsl.FloatingSoilError(f"{name} must be finite and >= 0, got {value!r}")
        return None, v
    fsl._check_vec(op.column, name, value, (op.n_cells,))
    return value, None


def _fresh(arr: Any, scalar: float | None, n: int, name: str) -> np.ndarray:
    if arr is None:
        out = np.full(n, scalar, dtype=np.float64)
    else:
        if not bool(np.isfinite(arr).all()) or bool((arr < 0.0).any()):
            raise fsl.FloatingSoilError(f"{name} must be finite and >= 0 on every cell (inactive included)")
        out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


def _fresh_cuda(cp: Any, column: Any, arr: Any, scalar: float | None, n: int, name: str) -> Any:
    """CUDA: a scalar fills a fresh device array; an array is validated on the device on EVERY cell (finite, >= 0) and then copied once."""
    if arr is None:
        out = cp.empty(n, dtype=np.float64)
        out.fill(scalar)
        return out
    if not _fmc().validate_range(cp, column.device_id, arr, 0.0, math.inf):
        raise fsl.FloatingSoilError(f"{name} must be finite and >= 0 on every cell (inactive included)")
    return cp.array(arr, dtype=np.float64, copy=True)


def floating_matrix_history(op: Any, *, wetting_m: Any, cumulative_intake_m: Any, memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                            available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                            host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingMatrixHistory:
    """EXPLICIT validated history bound to `op.sha256`; both arguments are REQUIRED (nothing is defaulted to zero), each a host scalar `>= 0` or an exact `(N,)` float64 array of the owner's namespace with every
    cell finite and `>= 0`. It cannot verify a genuine run boundary and claims no accepted generation; it implements no reset, recovery or rescaling. Admitted before any allocation."""
    column = _check_operator(op)
    quota = _quota(column, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn)
    w_arr, w_val = _history_input(op, wetting_m, "wetting_m")
    c_arr, c_val = _history_input(op, cumulative_intake_m, "cumulative_intake_m")
    _admit(column, "history", quota, "floating matrix history")
    if column.backend == "cuda":
        cp = _fcu()._rc().require_cuda()
        w = _fresh_cuda(cp, column, w_arr, w_val, op.n_cells, "wetting_m")
        c = _fresh_cuda(cp, column, c_arr, c_val, op.n_cells, "cumulative_intake_m")
        with _minting():
            return FloatingMatrixHistory(w, c, op.sha256, op.n_cells, namespace="cupy", device_id=column.device_id)
    w = _fresh(w_arr, w_val, op.n_cells, "wetting_m")
    c = _fresh(c_arr, c_val, op.n_cells, "cumulative_intake_m")
    with _minting():
        return FloatingMatrixHistory(w, c, op.sha256, op.n_cells)


def _check_history(op: FloatingMatrixOperator, history: Any, who: str = "") -> None:
    if type(history) is not FloatingMatrixHistory:
        raise fsl.FloatingSoilError(f"{who}history must be a FloatingMatrixHistory, got {type(history).__name__}")
    if history.operator_sha256 != op.sha256 or history.n_cells != op.n_cells:
        raise fsl.FloatingSoilError(f"{who}the history is bound to a different operator identity (other column, K/psi/g inputs or backend); a hydraulic history is never reinterpreted under other inputs; refused")
    want_ns, want_dev = ("cupy", op.column.device_id) if op.backend == "cuda" else ("numpy", None)
    if history.namespace != want_ns or history.device_id != want_dev:
        raise fsl.FloatingSoilError(f"{who}the history lives in namespace {history.namespace!r} (device {history.device_id}) but this {op.backend!r} operator needs {want_ns!r} (device {want_dev}); histories are never converted between backends")
    for name, a in (("wetting_m", history.wetting_m), ("cumulative_intake_m", history.cumulative_intake_m)):
        fsl._check_vec(op.column, f"{who}history.{name}", a, (op.n_cells,))


def rebind_floating_matrix_history(old_op: Any, old_history: Any, new_op: Any, *, memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                                   available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                                   host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingMatrixRebinding:
    """EXPLICIT PURE handoff of a persistent history to a changed operator. Order: both operators (CUDA: current device first; metadata) -> the old history against the OLD operator (identity, type, shape, device;
    then admission; then finite and `>= 0` on every cell) -> same physical columns (dimensions, backend, device, active mask, nominal edges, cell area) -> exact copies of `J` and the cumulative actual intake
    bound to `new_op.sha256`. The values are carried bit for bit: nothing is reset, rescaled, clipped or recovered, whatever changed in K, psi, g, properties or depth. Nothing is mutated; a refusal returns
    nothing. This is a PROPOSED binding, not an accepted generation (native adoption is slice E). A CUDA rebinding downloads nothing: the receipt sums are None."""
    oc, nc = _check_operator(old_op), _check_operator(new_op)
    _check_history(old_op, old_history, "old ")
    if (oc.n_layers, oc.n_cells, oc.backend, oc.device_id) != (nc.n_layers, nc.n_cells, nc.backend, nc.device_id):
        raise fsl.FloatingSoilError("the new operator has other dimensions, backend or device than the old one; refused")
    if oc.mask_sha256 != nc.mask_sha256 or oc.edges_sha256 != nc.edges_sha256:
        raise fsl.FloatingSoilError("the new operator's column has another active mask or nominal depth edges than the old one; refused")
    if float(oc.cell_area_m2) != float(nc.cell_area_m2):
        raise fsl.FloatingSoilError(f"the new operator's cell area {nc.cell_area_m2!r} differs from the old {oc.cell_area_m2!r}; a history is never rescaled; refused")
    quota = _quota(nc, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn)
    est = _admit(nc, "rebind", quota, "floating matrix history rebinding")
    if nc.backend == "cuda":
        cp, fmc = _fcu()._rc().require_cuda(), _fmc()
        for name, a in (("wetting_m", old_history.wetting_m), ("cumulative_intake_m", old_history.cumulative_intake_m)):
            if not fmc.validate_range(cp, nc.device_id, a, 0.0, math.inf):
                raise fsl.FloatingSoilError(f"old history.{name} must be finite and >= 0 on every cell; refused")
        w, c = cp.array(old_history.wetting_m, dtype=np.float64, copy=True), cp.array(old_history.cumulative_intake_m, dtype=np.float64, copy=True)
        with _minting():
            h = FloatingMatrixHistory(w, c, new_op.sha256, new_op.n_cells, namespace="cupy", device_id=nc.device_id)
        return FloatingMatrixRebinding(h, old_op.sha256, new_op.sha256, oc.sha256, nc.sha256, new_op.n_cells, None, None, False, est)
    for name, a in (("wetting_m", old_history.wetting_m), ("cumulative_intake_m", old_history.cumulative_intake_m)):
        if not bool(np.isfinite(a).all()) or bool((a < 0.0).any()):
            raise fsl.FloatingSoilError(f"old history.{name} must be finite and >= 0 on every cell; refused")
    w = np.array(old_history.wetting_m, dtype=np.float64, copy=True)
    c = np.array(old_history.cumulative_intake_m, dtype=np.float64, copy=True)
    w.flags.writeable = False
    c.flags.writeable = False
    with _minting():
        h = FloatingMatrixHistory(w, c, new_op.sha256, new_op.n_cells)
    return FloatingMatrixRebinding(h, old_op.sha256, new_op.sha256, oc.sha256, nc.sha256, new_op.n_cells, float(w.sum()), float(c.sum()), False, est)


def rebind_matrix_geometry_view(op: Any, view: fsl.FloatingSoilColumn, history: Any) -> tuple[FloatingMatrixOperator, FloatingMatrixHistory]:
    """HOT rebinding of a CPU operator and its persistent history to an official geometry view of the SAME prepared column (`fsl.floating_geometry_view`). The law inputs are the SAME sealed arrays (the law does not
    depend on the depth: only the porous slab does) and the history arrays are SHARED, read-only and bit-identical (nothing copied, hashed, reset, rescaled or recovered). O(1) apart from the metadata checks. The
    new operator identity digests the old operator identity and the view identity (a generation identity, never a content hash)."""
    column = _check_operator(op)
    _check_history(op, history)
    fsl._check_column(view)
    fsl._gate(view)  # CUDA: the current device again for the view, before any identity work
    root = column.sha256 if column.parent_sha256 is None else column.parent_sha256
    if (view.parent_sha256 != root or view.backend != column.backend or view.device_id != column.device_id or (view.n_layers, view.n_cells) != (column.n_layers, column.n_cells)
            or view.cell_area_m2 != column.cell_area_m2 or view.mask_sha256 != column.mask_sha256 or view.edges_sha256 != column.edges_sha256):
        raise fsl.FloatingSoilError("the view is not a geometry view of the operator's prepared column (parent identity, backend, device, dimensions, area, active mask or edges); refused")
    sha = hashlib.sha256(f"{OPERATOR_TAG}\0view\0{op.sha256}\0{view.sha256}".encode()).hexdigest()
    with _minting():
        new_op = FloatingMatrixOperator(view, op.backend, op.n_layers, op.n_cells, view.sha256, sha, op.layouts, op.retained_bytes, op._law, device_id=op.device_id)
        # the sealed law arrays and the persistent history arrays are SHARED (namespace and device preserved): no reset, rescale, copy, download or hash
        new_history = FloatingMatrixHistory(history.wetting_m, history.cumulative_intake_m, sha, op.n_cells, namespace=history.namespace, device_id=history.device_id)
    return new_op, new_history


# ---------------------------------------------------------------------------------------------------------------------------------------------
# readers and the step
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _numpy_readers(column: fsl.FloatingSoilColumn):
    """Lazy floating rows of the four values the shared body reads: `dz[0]` (h0, or the safe stand-in 1.0 only where h0 == 0), `fc_m = theta_fc * h`, `sat_m = theta_sat * h` (`+inf` on inactive cells: no
    capacity law) and `cap_tol = coef * sat`."""
    m, kinds = column._maps, column.layouts
    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    e, depth, active = m["edges"], m["soil_depth"], m["active"]
    tf, ts = m["theta_field_capacity"], m["theta_saturation"]
    ktf, kts = kinds["theta_field_capacity"], kinds["theta_saturation"]

    def h(k: int) -> np.ndarray:
        return np.maximum(np.minimum(depth, e[k + 1]) - e[k], 0.0)

    def dz(k: int) -> np.ndarray:
        if k != 0:
            raise IndexError("the matrix body reads only layer 0 of dz")
        x = h(0)
        return np.where(x > 0.0, x, 1.0)

    return (fsl._Rows(dz), fsl._Rows(lambda k: fsl._row(tf, ktf, k) * h(k)), fsl._Rows(lambda k: np.where(active, fsl._row(ts, kts, k) * h(k), np.inf)),
            fsl._Rows(lambda k: np.where(active, coef * (fsl._row(ts, kts, k) * h(k)), 0.0)))


def _law_views(op: FloatingMatrixOperator):
    return tuple(np.broadcast_to(a, (op.n_cells,)) if op.layouts[n] == "scalar" else a for n, a in zip(LAW_NAMES, op._law, strict=True))


def _run(op: FloatingMatrixOperator, pool: Any, standing: Any, rate: Any, water: Any, wet: Any, cum: Any, dt: float) -> mi.StepArrays:
    column = op.column
    L, N = op.n_layers, op.n_cells
    cell_tol = mi.cell_closure_coefficient(L)
    tol = mi.newton_tolerance()
    m = column._maps
    if op.backend == "numba":
        from maple_syrup import matrix_infiltration_numba as mn

        k, psi, g = op._law
        strides = tuple(0 if op.layouts[n] == "scalar" else 1 for n in LAW_NAMES)
        code = {"scalar": 0, "shared": 1, "cell": 2}
        return mn.floating_launch(pool, standing, rate, water, wet, cum, m["active"].view(np.uint8), dt, k=k, k_i=strides[0], psi=psi, p_i=strides[1], g=g, g_i=strides[2], edges=m["edges"],
                                  depth=m["soil_depth"], theta_fc=m["theta_field_capacity"].reshape(-1), k_fc=code[column.layouts["theta_field_capacity"]], theta_sat=m["theta_saturation"].reshape(-1),
                                  k_sat=code[column.layouts["theta_saturation"]], coef=volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0), cell_tol=cell_tol, tol=tol)
    dz, fc, sat, cap = _numpy_readers(column)
    k, psi, g = _law_views(op)
    ctx = mi._StepContext(L, N, m["active"], cell_tol, tol)
    return mi._reference_step_body(ctx, k, psi, g, dz, fc, sat, cap, pool, standing, rate, water, wet, cum, dt)


def floating_matrix_step(op: Any, surface_pool_m: Any, standing_before_m: Any, incoming_rate_m_s: Any, state: fsl.FloatingSoilState, history: Any, dt_s: Any, *, memory_budget_bytes: Any = None,
                         reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                         host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingMatrixStep:
    """One pure matrix stage over `dt_s` (supply convention B of `matrix_infiltration`): `surface_pool_m` `(N,)` the actual pool offered over the interval (supplied once by the owner),
    `standing_before_m` `(N,)` the preexisting standing depth (contact descriptor), `incoming_rate_m_s` `(N,)` the onset descriptor only. Order: operator (CUDA: current device first) -> state/forcing/history
    metadata -> dt -> admission -> allocation -> body -> ONE packet reduction -> reject or publish. Inputs are read only; the clock is preserved; `dt = 0` is the identity for valid inputs; a refusal publishes
    nothing. CUDA owner: exact `cupy.ndarray`s of the owner's device, nothing transferred, results on the device, ONE 136-byte raw packet read back (and rebuilt into the unchanged 17-float packet)."""
    column = _check_operator(op)
    t_s = fsl._check_state(column, state)
    dt = ss._check_dt(dt_s)
    n = (column.n_cells,)
    fsl._check_vec(column, "surface_pool_m", surface_pool_m, n)
    fsl._check_vec(column, "standing_before_m", standing_before_m, n)
    fsl._check_vec(column, "incoming_rate_m_s", incoming_rate_m_s, n)
    _check_history(op, history)
    quota = _quota(column, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn)
    est = _admit(column, "step", quota, "floating matrix step")
    if column.backend == "cuda":
        pool_o, water_o, wet_o, cum_o, intake_o, packet = _fmc().step(_fcu()._rc().require_cuda(), op, surface_pool_m, standing_before_m, incoming_rate_m_s, state.water_m, history.wetting_m,
                                                                     history.cumulative_intake_m, dt)
        budget = mi._resolve_core(column.cell_area_m2, op.n_layers, op.n_cells, packet)
        new_state = fsl._new_state(column, water_o, t_s)
        with _minting():
            new_history = FloatingMatrixHistory(wet_o, cum_o, op.sha256, op.n_cells, namespace="cupy", device_id=column.device_id)
        return FloatingMatrixStep(pool_o, new_state, new_history, intake_o, dt, budget, est)
    arrays = _run(op, surface_pool_m, standing_before_m, incoming_rate_m_s, state.water_m, history.wetting_m, history.cumulative_intake_m, dt)
    packet = mi._read_packet(np, arrays)
    budget = mi._resolve_core(column.cell_area_m2, op.n_layers, op.n_cells, packet)
    new_state = fsl._new_state(column, arrays.water_m, t_s)
    arrays.wetting_m.flags.writeable = False
    arrays.cumulative_intake_m.flags.writeable = False
    with _minting():
        new_history = FloatingMatrixHistory(arrays.wetting_m, arrays.cumulative_intake_m, op.sha256, op.n_cells)
    return FloatingMatrixStep(arrays.pool_m, new_state, new_history, arrays.intake_m, dt, budget, est)
