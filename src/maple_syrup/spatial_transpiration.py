"""Fixed explicit SPATIAL withdrawal / availability TRIAL operators (CPU: NumPy and compiled Numba). Phase 2 S2-B1.b.

A pure, standalone component on a REAL `spatial_soil_layers.SpatialSoilColumn` / `SpatialSoilState` (sealed owner with retained RAW per-cell maps; not a `SoilColumn`, not a `SoilLayers`). The law is the existing one
(`subsurface_transpiration`): per active cell and layer `available = max(W - residual, 0)`, `actual = min(requested, available)`, `W' = W - actual`, `unmet = requested - actual`,
`scale = actual / requested` (dimensionless fulfilment fraction; 0 where the request is 0). The residual `theta_res * dz` is a withdrawal FLOOR, not a minimum inventory (zero or below-residual water is legal and
nothing is ever added). No clock advance, evaporation, drainage, vertical exchange, plant-demand calculation, competition or root plasticity: LEAF owns the requests, SYRUP validates and debits the aggregate in one
inventory and returns the actuals. Residual and saturation bounds are derived inline from the owner's raw maps in the accepted product order (`res = theta_res * dz`; `sat = theta_sat * dz`;
`satb = sat + coef * sat`, `coef = volume_roundoff_bound_m3(LAYER_OPERATION_TERMS, 1.0)`): no `(L, N)` bound tensor, no copy of a map or of the mask.

    op  = prepare_spatial_withdrawal(column, memory_budget_bytes=...)          # sealed; backend numpy | numba; a cuda owner is refused (B1.c is later)
    av  = spatial_layer_availability_m(op, state, memory_budget_bytes=...)     # fresh (L, N) depth in m, inactive cells 0
    res = spatial_withdraw(op, state, requested_water_m, memory_budget_bytes=...)   # SpatialWithdrawalResult

One physical body per backend: NumPy reuses `subsurface_transpiration._fused_numpy` through lazy per-layer row readers; Numba compiles a thin loop kernel (explicit read-only signature, strided raw-map readers) around the SAME
scalar-only inline decision helpers `subsurface_transpiration._withdraw_cell/_withdraw_available` (plus `_withdraw_init_rows`) that the common compiled kernel calls; only the stores and the per-cell reduction-row bookkeeping are repeated structurally in the two loop kernels (checked bit for bit against the common operator). There is no copied solver, no fake legacy class and no fallback: a Numba
owner that cannot compile is a typed `SpatialLayerBackendUnavailableError`; a numba owner never runs the NumPy body.

BINDING AND LIMITS (explicit). The operator binds the real column OBJECT and its canonical input SHA-256 (parameters + mask + area), backend, namespace and `(L, N)`. A `SpatialSoilState` carries `column_sha256` (the same
input SHA), dimensions, time and namespace but NO water-state generation and NO owner-object identity: a state made for another owner object prepared from IDENTICAL inputs is accepted, a state of other inputs is refused, and
two different water states can have identical metadata and clock (a withdrawal changes `W` but neither the SHA nor the time). The operation is therefore a PURE TRIAL: it makes no exactly-once, stale-proposal, LEAF-adoption,
lifecycle or restart claim, and a returned availability array is NOT a generation-bound accepted-availability receipt; a later integration must bind the accepted parent state/generation/occurrence. Legacy native
operators/states/columns and these spatial ones refuse each other (typed). Old global flag behaviour is preserved: the reported reasons come from `max` over the per-cell flag rows and may omit further bits of other invalid
cells, while the call still refuses; the spatial text names the same reasons as the common operator.

Memory: `estimate_spatial_withdrawal_memory` (payload model, host; caller state/request and the owner's retained maps are excluded; interpreter/JIT/allocator state and whole-process memory are excluded; a reading is not
a reservation). Admission precedes every allocation, including the NumPy lazy row temporaries and failure paths. A failed call publishes nothing and never writes state, requests, mask or maps.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = ["OPERATOR_TAG", "SPATIAL_WITHDRAWAL_VERSION", "SpatialWithdrawalMemory", "SpatialWithdrawalOperator", "SpatialWithdrawalResult", "compiled_kernels",
           "estimate_spatial_withdrawal_memory", "prepare_spatial_withdrawal", "spatial_layer_availability_m", "spatial_withdraw"]

SPATIAL_WITHDRAWAL_VERSION = "1"
OPERATOR_TAG = "maple_syrup.spatial_withdrawal.operator.v1"
BACKENDS = ("numpy", "numba", "cuda")
OPERATIONS = ("prepare", "availability", "withdraw")
NUMBA_STAGE_ROWS = 2  # (N,) float64 local rows of the Numba withdrawal kernel (residual and saturation bound of one layer), Numba NRT heap
ROW_TEMP_DOUBLES = 6  # (N,) float64 rows of the lazily derived bounds a NumPy call may hold (res, sat, coef*sat, satb and two products), counted for every NumPy operation
NUMPY_LN_TEMPS = 6  # (L, N)-sized NumPy temporaries reserved by the common NumPy body (see `estimate_transpiration_memory`)
_F8 = 8

# ---------------------------------------------------------------------------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialWithdrawalMemory:
    """Modelled HOST bytes of ONE spatial withdrawal operation (see the module note for what is excluded)."""

    operation: str
    backend: str
    n_layers: int
    n_cells: int
    output_bytes: int  # arrays that survive the call (withdraw: water', actual, unmet, scale; availability: the (L, N) result)
    row_bytes: int  # the fused reduction rows (withdraw only)
    scratch_bytes: int  # backend temporaries, reduction workspace and the stacked/host packets
    row_temp_bytes: int  # NumPy lazily derived bound rows
    packet_bytes: int
    python_allowance_bytes: int
    reserve_bytes: int
    required_bytes: int
    input_bytes: dict
    terms: dict
    scope: str = ("modelled bytes of this call's own arrays (host): the caller's state and request and the owner's retained maps/mask are excluded; interpreter, JIT/compiler, allocator pool and whole-process memory "
                  "are excluded; an availability reading is not a reservation")
    device_required_bytes: int = 0  # CUDA: the DEVICE requirement (== required_bytes); CPU backends: 0
    host_required_bytes: int = -1  # CUDA: the separate HOST requirement (packet + Python allowance + host reserve); CPU backends: == required_bytes
    host_reserve_bytes: int = 0

    def __post_init__(self) -> None:
        if self.host_required_bytes < 0:
            object.__setattr__(self, "host_required_bytes", self.required_bytes)
        object.__setattr__(self, "input_bytes", sp._freeze(dict(self.input_bytes)))
        object.__setattr__(self, "terms", sp._freeze(dict(self.terms)))

    def as_dict(self) -> dict:
        return {k: (dict(v) if hasattr(v, "keys") else v) for k, v in self.__dict__.items()}


def _estimate_cuda(operation: str, L: int, N: int, reserve: int, host_reserve: int) -> SpatialWithdrawalMemory:
    from maple_syrup import spatial_transpiration_cuda as tc

    tc.check_launch_ranges(L, N)
    t = tc.operation_terms(operation, L, N)
    f8 = 8
    inputs = {"state.water_m": L * N * f8} if operation == "availability" else ({"state.water_m": L * N * f8, "requested_water_m": L * N * f8} if operation == "withdraw" else {})
    device = t["outputs"] + t["scratch"] + reserve
    host = t["host"] + sp.PY_ALLOWANCE_BYTES + host_reserve
    terms = {"backend": "cuda", "operation": operation, "n_blocks": t["n_blocks"], "static_shared_bytes_per_block": t["shared"], "no_rows_array": True, "no_cub_or_cupy_reduction": True,
             "no_retained_bound_maps": True, "no_LN_derived_bound_tensor": True, "no_mask_copy": True, "host_packet_arrays": 1 if operation == "withdraw" else 0,
             "excluded": "NVRTC/module load, CUDA context, allocator pool policy and 512-byte block rounding, interpreter; estimates are payload models, not reservations"}
    scope = ("modelled bytes of this call's own arrays: DEVICE main requirement in required_bytes/device_required_bytes (outputs + released scratch + device reserve) and the separate HOST requirement in "
             "host_required_bytes (the host packet + Python allowance + host reserve); caller state/request and the owner's retained maps/mask are excluded; allocator rounding, compiler/context and whole-process "
             "memory are excluded; an availability reading is not a reservation")
    return SpatialWithdrawalMemory(operation, "cuda", L, N, t["outputs"], 0, t["scratch"], 0, t["host"] if operation == "withdraw" else 0, sp.PY_ALLOWANCE_BYTES, reserve, device, inputs, terms, scope, device, host, host_reserve)


def estimate_spatial_withdrawal_memory(n_layers: int, n_cells: int, *, backend: str, operation: str = "withdraw", reserve_bytes: int = 0, host_reserve_bytes: int = 0) -> SpatialWithdrawalMemory:
    """Pure arithmetic requirement (no allocation). `operation`: "prepare" (the compile/preparation scratch: the one-cell reference call of the common estimate), "availability" or "withdraw".

    withdraw     outputs `4 L N 8` (W', actual, unmet, scale) + rows `8 N 8` + scratch (numba: `N 8` reduction workspace + 3 packets of 64 B; numpy: the same plus `6 L N 8` temporaries as the common NumPy body)
                 + bound rows (numpy: `6 N 8` lazy rows; numba: `2 N 8` staged local rows) + the 64 B host packet + the 64 KiB Python allowance
    availability outputs `L N 8` + scratch (numba: `N` flag bytes; numpy: `N` + `4 N 8` per-layer temporaries) + NumPy lazy bound rows `6 N 8` + the 64 KiB Python allowance
    prepare      the withdrawal call on a 1 x 1 domain + the allowance (compile excluded, JIT is not counted)
    Caller state/request and the owner's retained maps are EXCLUDED (`input_bytes` records them).
    `backend="cuda"` (S2-B1.c) returns the DEVICE requirement in `required_bytes`/`device_required_bytes` (outputs + released scratch + device reserve: withdraw `4LN8 + 8*8*nb + 64`, availability `LN8 + 8`, prepare 0)
    and the separate HOST requirement in `host_required_bytes` (the 64-byte packet that `to_host` returns, or the 8-byte diagnostic, + the Python allowance + `host_reserve_bytes`); no rows array, no raw/reconstructed packet pair."""
    if backend not in BACKENDS:
        raise sp.SpatialLayerError(f"backend must be one of {BACKENDS} for a spatial withdrawal, got {backend!r}")
    if operation not in OPERATIONS:
        raise sp.SpatialLayerError(f"operation must be one of {OPERATIONS}, got {operation!r}")
    L, N = sp._count(n_layers, "n_layers", minimum=1), sp._count(n_cells, "n_cells", minimum=1)
    reserve = sp._count(reserve_bytes, "reserve_bytes")
    host_reserve = sp._count(host_reserve_bytes, "host_reserve_bytes")
    if backend != "cuda" and host_reserve:
        raise sp.SpatialLayerError("host_reserve_bytes applies only to the cuda backend")
    if backend == "cuda":
        return _estimate_cuda(operation, L, N, reserve, host_reserve)
    if operation == "prepare":
        inner = estimate_spatial_withdrawal_memory(1, 1, backend=backend, operation="withdraw")
        return SpatialWithdrawalMemory("prepare", backend, L, N, 0, 0, inner.scratch_bytes + inner.output_bytes + inner.row_bytes, inner.row_temp_bytes, inner.packet_bytes, sp.PY_ALLOWANCE_BYTES, reserve,
                                       inner.output_bytes + inner.row_bytes + inner.scratch_bytes + inner.row_temp_bytes + sp.PY_ALLOWANCE_BYTES + reserve, {}, {"backend": backend, "one_cell_reference_call": True})
    packet = _F8 * st._PACKET
    row_temp = (ROW_TEMP_DOUBLES if backend == "numpy" else NUMBA_STAGE_ROWS) * N * _F8 if operation == "withdraw" or backend == "numpy" else 0
    if operation == "withdraw":
        out, rows = 4 * L * N * _F8, st._N_ROWS * N * _F8
        scratch = (NUMPY_LN_TEMPS * L * N * _F8 if backend == "numpy" else 0) + N * _F8 + 3 * packet
        inputs = {"state.water_m": L * N * _F8, "requested_water_m": L * N * _F8}
    else:
        out, rows = L * N * _F8, 0
        scratch = N + (4 * N * _F8 if backend == "numpy" else 0)
        packet = 0
        inputs = {"state.water_m": L * N * _F8}
    allowance = sp.PY_ALLOWANCE_BYTES
    required = out + rows + scratch + row_temp + packet + allowance + reserve
    terms = {"backend": backend, "operation": operation, "no_retained_bound_maps": True, "no_LN_derived_bound_tensor": True, "no_mask_copy": True, "numpy_LN_temporaries": NUMPY_LN_TEMPS if backend == "numpy" else 0,
             "row_temp_doubles": ROW_TEMP_DOUBLES if backend == "numpy" else (NUMBA_STAGE_ROWS if operation == "withdraw" else 0)}
    return SpatialWithdrawalMemory(operation, backend, L, N, out, rows, scratch, row_temp, packet, allowance, reserve, required, inputs, terms)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# compiled kernels (functions only; never an owner, state or array)
# ---------------------------------------------------------------------------------------------------------------------------------------------
_LOCK = threading.Lock()
_KERNELS: Any = None  # (withdraw kernel, availability kernel) once compiled


def compiled_kernels():
    """The process-wide compiled spatial kernels `(withdraw, availability)`, built lazily at the first preparation of a numba owner (outside hot calls). Each is compiled for ONE explicit signature with READ-ONLY
    inputs and calls the shared inline physical helpers of `subsurface_transpiration`. A missing Numba or a failed compilation is a typed `SpatialLayerBackendUnavailableError`; nothing falls back."""
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    with _LOCK:
        if _KERNELS is not None:
            return _KERNELS
        if st._fused_numba is None or st._withdraw_cell is None:
            raise sp.SpatialLayerBackendUnavailableError("backend 'numba' requires Numba, which is not importable; there is no fallback to the NumPy body")
        try:
            import numba

            f8, i8 = numba.types.float64, numba.types.int64
            ro1, ro2 = numba.types.Array(f8, 1, "C", readonly=True), numba.types.Array(f8, 2, "C", readonly=True)
            ro_u8 = numba.types.Array(numba.types.uint8, 1, "C", readonly=True)
            rw1, rw2 = numba.types.Array(f8, 1, "C"), numba.types.Array(f8, 2, "C")
            rw_u8 = numba.types.Array(numba.types.uint8, 1, "C")
            cell, init_rows, available = st._withdraw_cell, st._withdraw_init_rows, st._withdraw_available
            maps = (ro1, i8, i8, ro1, i8, i8, ro1, i8, i8, f8)  # theta_res, theta_sat, dz (map, stride_layer, stride_cell) and coef

            @numba.njit(numba.types.void(ro2, ro2, *maps, ro_u8, rw2, rw2, rw2, rw2, rw2), cache=False, fastmath=False)
            def withdraw(w, q, tr, tr_l, tr_i, ts, ts_l, ts_i, dz, dz_l, dz_i, coef, active_u8, out_w, actual, unmet, scale, rows):
                L, N = w.shape
                init_rows(rows, N)
                # the two bounds of one layer are staged into two `(N,)` local rows (counted in the memory model) so that the decision loop reads them contiguously like the common body
                # (measured: ~2x faster than re-deriving them from the strided raw maps inside that loop); never an `(L, N)` tensor
                r_row = np.empty(N, dtype=np.float64)
                sb_row = np.empty(N, dtype=np.float64)
                for l in range(L):
                    for j in range(N):
                        d = dz[l * dz_l + j * dz_i]
                        r_row[j] = tr[l * tr_l + j * tr_i] * d  # res = theta_res * dz
                        sat = ts[l * ts_l + j * ts_i] * d  # sat = theta_sat * dz
                        sb_row[j] = sat + coef * sat  # saturation + capacity tolerance
                    for j in range(N):
                        wv = w[l, j]
                        qv = q[l, j]
                        act = active_u8[j] != 0
                        f, t, o, um, sc = cell(wv, qv, r_row[j], sb_row[j], act, rows[6, j])
                        rows[6, j] = f
                        out_w[l, j] = o
                        actual[l, j] = t
                        unmet[l, j] = um
                        scale[l, j] = sc
                        if act:
                            rows[0, j] += wv
                            rows[1, j] += o
                            rows[2, j] += t
                            rows[3, j] += qv - t
                            rows[4, j] += qv
                        if o < rows[5, j]:
                            rows[5, j] = o
                        if qv > rows[7, j]:
                            rows[7, j] = qv

            @numba.njit(numba.types.void(ro2, *maps, ro_u8, rw2, rw_u8), cache=False, fastmath=False)
            def availability(w, tr, tr_l, tr_i, ts, ts_l, ts_i, dz, dz_l, dz_i, coef, active_u8, out, bad):
                L, N = w.shape
                for j in range(N):
                    bad[j] = 0
                for l in range(L):
                    for j in range(N):
                        d = dz[l * dz_l + j * dz_i]
                        r = tr[l * tr_l + j * tr_i] * d
                        sat = ts[l * ts_l + j * ts_i] * d
                        sb = sat + coef * sat
                        wv = w[l, j]
                        if not (wv >= 0.0 and wv <= sb):
                            bad[j] = 1
                        a = available(wv, r)
                        out[l, j] = a if active_u8[j] != 0 else 0.0

            _KERNELS = (withdraw, availability)
        except sp.SpatialLayerBackendUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - a compiler defect is NOT an availability fallback
            raise sp.SpatialLayerBackendUnavailableError(f"the compiled spatial withdrawal kernels failed to build ({type(exc).__name__}: {exc}); there is no NumPy fallback") from exc
        return _KERNELS


# ---------------------------------------------------------------------------------------------------------------------------------------------
# operator
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
class SpatialWithdrawalOperator:
    """Sealed operator minted ONLY by `prepare_spatial_withdrawal`. It holds the real column object (a reference, not a copy) and metadata; it retains NO copied bound map, mask or state. `sha256` is the identity of
    (tag, column input SHA-256, backend, dimensions); see the module note for what the binding does and does not mean."""

    column: Any
    backend: str
    n_layers: int
    n_cells: int
    column_sha256: str
    sha256: str
    prepare_memory: SpatialWithdrawalMemory
    tag: str = OPERATOR_TAG

    def __post_init__(self) -> None:
        if not getattr(_CONSTRUCT, "allowed", False):
            raise sp.SpatialLayerError("a SpatialWithdrawalOperator is created only by prepare_spatial_withdrawal (no direct construction, replace() or copy)")

    def __copy__(self):
        raise sp.SpatialLayerError("a SpatialWithdrawalOperator is not copyable; prepare another")

    def __deepcopy__(self, memo):
        raise sp.SpatialLayerError("a SpatialWithdrawalOperator is not copyable; prepare another")

    def __reduce_ex__(self, protocol):
        raise sp.SpatialLayerError("a SpatialWithdrawalOperator is not serialisable (no checkpoint support)")

    def summary(self) -> dict:
        return {"tag": self.tag, "version": SPATIAL_WITHDRAWAL_VERSION, "backend": self.backend, "n_layers": self.n_layers, "n_cells": self.n_cells, "column_sha256": self.column_sha256, "sha256": self.sha256,
                "pure_trial": True, "generation_bound": False, "exactly_once": False}


def _identity(column: Any) -> str:
    h = hashlib.sha256()
    for part in (OPERATOR_TAG, column.sha256, column.backend, str((column.n_layers, column.n_cells))):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def _check_column_object(column: Any) -> None:
    if isinstance(column, ss.SoilColumn):
        raise sp.SpatialLayerError("prepare_spatial_withdrawal takes a SpatialSoilColumn, not a legacy SoilColumn (use subsurface_transpiration.prepare_transpiration for the common column)")
    if type(column) is not sp.SpatialSoilColumn:
        raise sp.SpatialLayerError(f"column must be a SpatialSoilColumn, got {type(column).__name__}")
    if column.backend not in BACKENDS:
        raise sp.SpatialLayerBackendUnavailableError(f"unsupported column backend {column.backend!r}")


def _check_operator(op: Any) -> Any:
    """Typed checks of the sealed operator against its real column; returns CuPy for a CUDA owner AFTER the current-device check (nothing is allocated, copied or launched here), else None."""
    if isinstance(op, st.TranspirationOperator):
        raise sp.SpatialLayerError("a legacy TranspirationOperator cannot be used with spatial states; use prepare_spatial_withdrawal")
    if type(op) is not SpatialWithdrawalOperator:
        raise sp.SpatialLayerError(f"operator must come from prepare_spatial_withdrawal, got {type(op).__name__}")
    column = op.column
    if type(column) is not sp.SpatialSoilColumn or column.sha256 != op.column_sha256 or column.backend != op.backend or (column.n_layers, column.n_cells) != (op.n_layers, op.n_cells):
        raise sp.SpatialLayerError("the operator no longer matches its column (identity, backend or dimensions); refused")
    if op.sha256 != _identity(column):
        raise sp.SpatialLayerError("the operator identity does not match its column; refused")
    return sp._device_cp(column)


def _admit_call(column: Any, operation: str, budget: Any, reserve: Any, reader: Any, host_budget: Any, host_reserve: Any, host_reader: Any, cp: Any) -> SpatialWithdrawalMemory:
    """Estimate + admission of ONE operation for either backend BEFORE any allocation, conversion, copy or launch. The separate host quota/reserve/reading exist only for a CUDA owner (refused for CPU owners)."""
    hb, hr, hrd = sp._quotas(column, host_budget, host_reserve, host_reader)
    if column.backend == "cuda":
        est = estimate_spatial_withdrawal_memory(column.n_layers, column.n_cells, backend="cuda", operation=operation, reserve_bytes=reserve, host_reserve_bytes=hr)
        sp._admit_cuda_operation(cp, column, est, budget, reader, hb, hrd)
        return est
    est = estimate_spatial_withdrawal_memory(column.n_layers, column.n_cells, backend=column.backend, operation=operation, reserve_bytes=reserve)
    sp._admit_operation(est, budget, reader)
    return est


def prepare_spatial_withdrawal(column: Any, *, memory_budget_bytes: Any, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None,
                              host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> SpatialWithdrawalOperator:
    """Mint the sealed operator for a `SpatialSoilColumn` (backend numpy | numba | cuda). Order: object/backend/device refusals (no copy) -> admission of the bounded preparation scratch -> the one-off compilation (numba
    functions, or the CUDA module of the owner's device; startup outside every hot call, no physical dummy launch or copy) -> mint. Nothing is retained but a reference to the column and metadata. CUDA: `memory_budget_bytes`/
    `reserve_bytes`/`available_bytes_fn` are the DEVICE quota/reserve/reading and `host_*` the separate host ones (refused for a CPU owner)."""
    _check_column_object(column)
    cp = sp._device_cp(column)
    est = _admit_call(column, "prepare", memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, cp)
    if column.backend == "numba":
        compiled_kernels()
    elif column.backend == "cuda":
        from maple_syrup import spatial_transpiration_cuda as tc

        try:
            with sp._device_errors("withdrawal preparation"):
                tc.load_kernels(cp, column.device_id)
        except ss.SubsurfaceBackendUnavailableError as exc:
            raise sp.SpatialLayerBackendUnavailableError(str(exc)) from exc
    with _minting():
        return SpatialWithdrawalOperator(column, column.backend, column.n_layers, column.n_cells, column.sha256, _identity(column), est)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# readers (NumPy) and launches
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _kinds(column: Any) -> dict:
    return {n: v["kind"] for n, v in column.estimate.retained.items()}


def _numpy_readers(column: Any):
    """Lazy per-layer rows of the two bounds, evaluated only when the body asks for layer `k` (a scalar for a scalar/shared layout, an `(N,)` row for a per-cell layout): the accepted product order."""
    maps, kinds = column._maps, _kinds(column)
    tr, ts, dz = maps["theta_residual"], maps["theta_saturation"], maps["dz"]
    ktr, kts, kdz = kinds["theta_residual"], kinds["theta_saturation"], kinds["dz"]
    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)

    def residual(k):
        return sp._retained_row(tr, ktr, k) * sp._retained_row(dz, kdz, k)

    def satb(k):
        sat = sp._retained_row(ts, kts, k) * sp._retained_row(dz, kdz, k)
        return sat + coef * sat

    return sp._Rows(residual), sp._Rows(satb)


def _strided(column: Any):
    """Flat read-only views of the owner's retained maps with their strides (no copy): `(tr, tr_l, tr_i, ts, ts_l, ts_i, dz, dz_l, dz_i)`."""
    n = column.n_cells
    maps, kinds = column._maps, _kinds(column)
    out = []
    for name in ("theta_residual", "theta_saturation", "dz"):
        sl, si = {"scalar": (0, 0), "shared": (1, 0), "cell": (n, 1)}[kinds[name]]
        out += [maps[name].reshape(-1), np.int64(sl), np.int64(si)]
    return tuple(out)


def _packet(rows: np.ndarray) -> np.ndarray:
    return np.stack([np.float64(rows[st._ROW_W_BEFORE].sum()), np.float64(rows[st._ROW_W_AFTER].sum()), np.float64(rows[st._ROW_ACTUAL].sum()), np.float64(rows[st._ROW_UNMET].sum()),
                     np.float64(rows[st._ROW_REQ].sum()), np.float64(rows[st._ROW_MIN_AFTER].min()), np.float64(rows[st._ROW_FLAGS].max()), np.float64(rows[st._ROW_QMAX].max())])


@dataclass(frozen=True, eq=False)
class SpatialWithdrawalResult:
    """`state` is a fresh `SpatialSoilState` with the SAME clock; `actual_m`, `unmet_m` `(L, N)` depths [m]; `scale` `(L, N)` DIMENSIONLESS fulfilment fraction (`actual/requested`, 0 where the request is 0);
    `budget` carries the fields and meanings of `subsurface_transpiration.TranspirationBudget` (volumes over ACTIVE cells, area applied once, MAPLE FP64 bound); `memory` the model of the call."""

    state: Any
    actual_m: Any
    unmet_m: Any
    scale: Any
    budget: st.TranspirationBudget
    memory: SpatialWithdrawalMemory

    @property
    def fulfillment_fraction(self):
        return self.scale


def _mask_u8(column: Any) -> np.ndarray:
    return column._mask.view(np.uint8)  # a zero-copy read-only view of the owner's bool mask


def spatial_withdraw(operator: Any, state: Any, requested_water_m: Any, *, memory_budget_bytes: Any, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None,
                    host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> SpatialWithdrawalResult:
    """Pure fixed-spatial withdrawal (module note). Units: `requested_water_m` `(L, N)` float64 depth [m] over the ALREADY DEFINED interval (nothing is scaled by a rate), zero on inactive cells. Order: operator ->
    state/request metadata -> admission -> allocation -> compiled/NumPy body -> ONE host packet -> reject or publish. The state, request, mask and maps are never written; a refusal publishes nothing; the clock is
    preserved. Invalid values (non-finite/negative/over-saturated water, non-finite/negative request, positive request on an inactive cell) are validated on EVERY cell and refused with a typed `SubsurfaceError`.
    CUDA owner (S2-B1.c): `state` and `requested_water_m` must be exact `cupy.ndarray`s of the owner's device (current device checked first), nothing is transferred, the result arrays stay on the device, one 64-byte
    packet is read back, and `memory_budget_bytes`/`reserve_bytes`/`available_bytes_fn` are the DEVICE quota/reserve/reading with the separate `host_*` ones (refused for a CPU owner)."""
    cp = _check_operator(operator)
    column = operator.column
    t_s = sp._check_spatial_state(column, state)
    L, N = operator.n_layers, operator.n_cells
    sp._check_array("requested_water_m", requested_water_m, (L, N), column)
    est = _admit_call(column, "withdraw", memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, cp)
    w = state.water_m
    if operator.backend == "cuda":
        from maple_syrup import spatial_transpiration_cuda as tc

        try:
            with sp._device_errors("withdrawal"):
                out_w, actual, unmet, scale, p = tc.withdraw(cp, column, w, requested_water_m, float(volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)))
        except ss.SubsurfaceBackendUnavailableError as exc:
            raise sp.SpatialLayerBackendUnavailableError(str(exc)) from exc
    else:
        out_w, actual, unmet, scale = (np.empty((L, N), dtype=np.float64) for _ in range(4))
        rows = np.empty((st._N_ROWS, N), dtype=np.float64)
        p = _run_cpu(operator, column, w, requested_water_m, out_w, actual, unmet, scale, rows)
        del rows
    flags = p[st._ROW_FLAGS]
    stage = "spatial_withdraw"
    if not (flags == 0.0):
        bits = int(flags) if math.isfinite(flags) else 7
        why = "; ".join(t for b, t in st._FLAG_TEXT.items() if bits & b) or "invalid values"
        raise ss.SubsurfaceError(f"{stage}: {why} (checked on every cell, inactive included)")
    return _publish(column, p, out_w, actual, unmet, scale, t_s, est, stage)


def _run_cpu(operator: Any, column: Any, w: Any, requested: Any, out_w: Any, actual: Any, unmet: Any, scale: Any, rows: Any) -> np.ndarray:
    if operator.backend == "numba":
        withdraw_kernel, _ = compiled_kernels()
        coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
        withdraw_kernel(w, requested, *_strided(column), float(coef), _mask_u8(column), out_w, actual, unmet, scale, rows)
    else:
        residual, satb = _numpy_readers(column)
        st._fused_numpy(w, requested, residual, satb, column._mask, out_w, actual, unmet, scale, rows)
    return _packet(rows)


def _publish(column: Any, p: np.ndarray, out_w: Any, actual: Any, unmet: Any, scale: Any, t_s: float, est: SpatialWithdrawalMemory, stage: str) -> SpatialWithdrawalResult:
    """Conservation bookkeeping shared by every backend: volumes over ACTIVE cells (area applied once), the typed refusal of unrepresentable global quantities BEFORE the strict conservation helper, the MAPLE FP64
    bound with the common `n_terms`, closure checks, then the fresh same-clock state."""
    L = column.n_layers
    area = float(column.cell_area_m2)
    before, after, act, unm, req = (float(v) * area for v in p[:5])
    wmin, qmax = float(p[st._ROW_MIN_AFTER]), float(p[st._ROW_QMAX])
    n_terms = 3 * L * int(column.n_active) + st._PACKET
    # Typed refusal BEFORE the strict conservation helper: finite accepted parameters can still give a global volume (depth sum x cell area) or the closure scale `before + req` that FP64 cannot represent.
    # Unrepresentable quantities are refused, never clipped or approximated; valid arithmetic, coefficient and n_terms are untouched.
    scale_m3 = before + req
    if not all(math.isfinite(v) for v in (before, after, act, unm, req, scale_m3)):
        raise ss.SubsurfaceError(f"{stage}: a global volume or the closure scale is not representable in FP64 (before {before!r}, after {after!r}, actual {act!r}, unmet {unm!r}, requested {req!r} m3; "
                                 f"before + requested {scale_m3!r} m3; cell area {area!r} m2); refused, nothing was published")
    bound = float(volume_roundoff_bound_m3(n_terms, max(scale_m3, 0.0)))
    budget = st.TranspirationBudget(before, after, req, act, unm, after - before + act, req - act - unm, bound, n_terms, wmin, qmax)
    if not (math.isfinite(after) and math.isfinite(act) and math.isfinite(unm) and math.isfinite(wmin)) or wmin < 0.0:
        raise ss.SubsurfaceError(f"{stage}: withdrawal produced non-finite or negative water: {budget}")
    if abs(budget.residual_m3) > bound or abs(budget.request_residual_m3) > bound:
        raise ss.SubsurfaceError(f"{stage}: transpiration closure violated beyond the FP64 bound: {budget}")
    return SpatialWithdrawalResult(sp._new_state(column, out_w, t_s), actual, unmet, scale, budget, est)


def spatial_layer_availability_m(operator: Any, state: Any, *, memory_budget_bytes: Any, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None,
                                host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> Any:
    """PURE READ: the fresh `(L, N)` float64 depth [m] `max(W - theta_res * dz, 0)` of ACTIVE cells (inactive cells 0). The input water is validated on EVERY cell (finite, `>= 0`, `<= saturation + tolerance`) with the
    same reason text as the withdrawal; a refusal returns nothing. Nothing is advanced or stored; the result is NOT a generation-bound accepted-availability receipt (a state has no generation), only a trial value.
    CUDA owner: a fresh device array, a one-word diagnostic read, device/host quotas as for the withdrawal."""
    cp = _check_operator(operator)
    column = operator.column
    sp._check_spatial_state(column, state)
    L, N = operator.n_layers, operator.n_cells
    _admit_call(column, "availability", memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, cp)
    w = state.water_m
    if operator.backend == "cuda":
        from maple_syrup import spatial_transpiration_cuda as tc

        try:
            with sp._device_errors("availability"):
                out, invalid = tc.availability(cp, column, w, float(volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)))
        except ss.SubsurfaceBackendUnavailableError as exc:
            raise sp.SpatialLayerBackendUnavailableError(str(exc)) from exc
        if invalid:
            raise ss.SubsurfaceError(f"spatial_layer_availability_m: {st._FLAG_TEXT[1]} (checked on every cell, inactive included)")
        return out
    out = np.empty((L, N), dtype=np.float64)
    if operator.backend == "numba":
        bad = np.empty(N, dtype=np.uint8)
        _, availability_kernel = compiled_kernels()
        availability_kernel(w, *_strided(column), float(volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)), _mask_u8(column), out, bad)
        invalid = bool(bad.any())
        del bad
    else:
        residual, satb = _numpy_readers(column)
        act = column._mask
        invalid_rows = np.zeros(N, dtype=np.bool_)
        for l in range(L):
            wl = w[l]
            a = wl - residual[l]
            np.maximum(a, 0.0, out=a)
            np.copyto(out[l], a)
            np.copyto(out[l], 0.0, where=~act)
            with np.errstate(invalid="ignore"):
                invalid_rows |= ~((wl >= 0.0) & (wl <= satb[l]))
        invalid = bool(invalid_rows.any())
    if invalid:
        raise ss.SubsurfaceError(f"spatial_layer_availability_m: {st._FLAG_TEXT[1]} (checked on every cell, inactive included)")
    return out
