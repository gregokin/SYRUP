"""Requested root withdrawals and layer availability on a FLOATING porous column (D2; NumPy and compiled Numba).

A pure component on a real `floating_soil_layers.FloatingSoilColumn` / `FloatingSoilState` (never a `SoilColumn` or `SpatialSoilColumn`; those classes keep refusing each other). The law is the existing one
(`subsurface_transpiration`): per ACTIVE cell and layer `available = max(W - theta_res * h, 0)`, `actual = min(requested, available)`, `W' = W - actual`, `unmet = requested - actual`,
`scale = actual / requested` (0 where the request is 0), where `h` is the ACTUAL porous thickness of the layer (`floating_soil_layers.porous_thickness`). A rock slot (`h = 0`) therefore has `available = 0`:
its request is entirely unmet, nothing is reallocated to another depth, and no denominator is ever zero. Demands stay at LEAF's fixed cell/depth fractions (no root plasticity, no renormalisation to accessible
depths). LEAF owns the requests; SYRUP validates and debits the aggregate once and reports the accepted amounts back.

One physical body per backend: NumPy reuses `subsurface_transpiration._fused_numpy` through lazy per-layer rows (`theta_res * h` and the saturation bound `sat + coef * sat` with `sat = theta_sat * h`); Numba
compiles a thin strided loop around the SAME scalar helpers (`_withdraw_cell`, `_withdraw_available`, `_withdraw_init_rows`) the common and fixed-spatial kernels call. INACTIVE cells have no capacity law: their saturation
bound is `+inf` in the reader (finiteness and `>= 0` are still validated on every cell), their water is passed through bit for bit and a positive request on one is refused.

    av  = floating_layer_availability_m(column, state)                    # fresh (L, N) depth in m, inactive cells 0
    res = floating_withdraw(column, state, requested_water_m)             # FloatingWithdrawalResult

Several plants: LEAF aggregates its requests per layer and cell (its sparse root connections) and passes ONE aggregate `(L, N)` request; each plant then receives `request_p * scale` from the returned
fulfilment fraction `scale` (demand-proportional, no priority). This module deliberately has NO dense per-plant `P * L * N` helper: the native coupling must use the sparse connections and the aggregate grid
withdrawal/scale. The operation is a PURE TRIAL on a state: no clock advance, evaporation, drainage or generation/exactly-once claim (native
adoption is slice E). CUDA owners dispatch from the same entry points to `floating_transpiration_cuda` (current device gate FIRST, metadata, then device/host admission, then the shared body; the host receives
ONE 64-byte packet and, for availability, one 8-byte diagnostic; nothing of the state or the maps is downloaded or hashed).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import floating_soil_layers as fsl
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = ["FLOATING_WITHDRAWAL_VERSION", "FloatingWithdrawalMemory", "FloatingWithdrawalResult", "compiled_kernels", "estimate_floating_withdrawal_memory", "floating_layer_availability_m",
           "floating_withdraw"]

FLOATING_WITHDRAWAL_VERSION = "1"
ROW_TEMP_DOUBLES = 8  # (N,) float64 rows a NumPy call may hold while deriving h, residual and the saturation bound of one layer
NUMPY_LN_TEMPS = 6  # (L, N)-sized temporaries of the shared NumPy body
NUMBA_STAGE_ROWS = 2
_F8 = 8


@dataclass(frozen=True)
class FloatingWithdrawalMemory:
    backend: str
    operation: str
    n_layers: int
    n_cells: int
    outputs_bytes: int
    scratch_bytes: int
    packet_bytes: int
    required_bytes: int
    scope: str = "modelled host payload bytes of this call (fresh outputs, reduction rows, derived row/temporary scratch, packet); caller state/request and the owner's retained maps are excluded"


def estimate_floating_withdrawal_memory(n_layers: int, n_cells: int, *, backend: str = "numpy", operation: str = "withdraw", reserve_bytes: int = 0) -> FloatingWithdrawalMemory:
    L, N = fsl._count(n_layers, "n_layers", minimum=1), fsl._count(n_cells, "n_cells", minimum=1)
    if backend not in ("numpy", "numba"):
        raise fsl.FloatingBackendUnavailableError(f"this module serves the numpy and numba owners (cuda: floating_transpiration_cuda), got {backend!r}")
    if operation not in ("availability", "withdraw"):
        raise fsl.FloatingSoilError(f"operation must be 'availability' or 'withdraw', got {operation!r}")
    ln = L * N * _F8
    if operation == "availability":
        out, scr, pk = ln, (ROW_TEMP_DOUBLES * N * _F8 + N), 0
    else:
        out = 4 * ln
        rows = st._N_ROWS * N * _F8
        scr = rows + (ROW_TEMP_DOUBLES * N * _F8 if backend == "numpy" else NUMBA_STAGE_ROWS * N * _F8) + (NUMPY_LN_TEMPS * ln if backend == "numpy" else 0)
        pk = st._PACKET * _F8
    return FloatingWithdrawalMemory(backend, operation, L, N, out, scr, pk, out + scr + pk + fsl._count(reserve_bytes, "reserve_bytes"))


# ---------------------------------------------------------------------------------------------------------------------------------------------
# readers (lazy per-layer rows; no (L, N) bound tensor)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _readers(column: fsl.FloatingSoilColumn):
    m, kinds = column._maps, column.layouts
    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    e, depth, active = m["edges"], m["soil_depth"], m["active"]
    tr, ts = m["theta_residual"], m["theta_saturation"]
    ktr, kts = kinds["theta_residual"], kinds["theta_saturation"]

    def h(k: int) -> np.ndarray:
        return np.maximum(np.minimum(depth, e[k + 1]) - e[k], 0.0)

    def residual(k: int) -> np.ndarray:
        return fsl._row(tr, ktr, k) * h(k)

    def satb(k: int) -> np.ndarray:
        sat = fsl._row(ts, kts, k) * h(k)
        b = sat + coef * sat
        return np.where(active, b, np.inf)  # inactive placeholders carry no capacity law

    return fsl._Rows(residual), fsl._Rows(satb)


def _mask_u8(column: fsl.FloatingSoilColumn) -> np.ndarray:
    return column._maps["active"].view(np.uint8)


def _strided(column: fsl.FloatingSoilColumn):
    n = column.n_cells
    m, kinds = column._maps, column.layouts
    out = []
    for name in ("theta_residual", "theta_saturation"):
        sl, si = {"scalar": (0, 0), "shared": (1, 0), "cell": (n, 1)}[kinds[name]]
        out += [m[name].reshape(-1), np.int64(sl), np.int64(si)]
    return tuple(out) + (m["edges"], m["soil_depth"])


def _packet(rows: np.ndarray) -> np.ndarray:
    return np.stack([np.float64(rows[st._ROW_W_BEFORE].sum()), np.float64(rows[st._ROW_W_AFTER].sum()), np.float64(rows[st._ROW_ACTUAL].sum()), np.float64(rows[st._ROW_UNMET].sum()),
                     np.float64(rows[st._ROW_REQ].sum()), np.float64(rows[st._ROW_MIN_AFTER].min()), np.float64(rows[st._ROW_FLAGS].max()), np.float64(rows[st._ROW_QMAX].max())])


# ---------------------------------------------------------------------------------------------------------------------------------------------
# compiled kernels (Numba): one thin loop around the shared scalar helpers
# ---------------------------------------------------------------------------------------------------------------------------------------------
_KERNELS = None
_LOCK = threading.Lock()


def compiled_kernels():
    """`(withdraw, availability)` compiled lazily once per process for ONE explicit signature each; a missing Numba or a failed build is a typed `FloatingBackendUnavailableError` (no NumPy fallback)."""
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    with _LOCK:
        if _KERNELS is not None:
            return _KERNELS
        if st._fused_numba is None or st._withdraw_cell is None:
            raise fsl.FloatingBackendUnavailableError("backend 'numba' requires Numba, which is not importable; there is no fallback to the NumPy body")
        try:
            import numba

            f8, i8 = numba.types.float64, numba.types.int64
            ro1, ro2 = numba.types.Array(f8, 1, "C", readonly=True), numba.types.Array(f8, 2, "C", readonly=True)
            ro_u8 = numba.types.Array(numba.types.uint8, 1, "C", readonly=True)
            rw1, rw2 = numba.types.Array(f8, 1, "C"), numba.types.Array(f8, 2, "C")
            cell, init_rows, available = st._withdraw_cell, st._withdraw_init_rows, st._withdraw_available
            maps = (ro1, i8, i8, ro1, i8, i8, ro1, ro1, f8)  # theta_res, theta_sat (map, stride_layer, stride_cell), edges, soil depth, coef

            @numba.njit(numba.types.void(ro2, ro2, *maps, ro_u8, rw2, rw2, rw2, rw2, rw2), cache=False, fastmath=False)
            def withdraw(w, q, tr, tr_l, tr_i, ts, ts_l, ts_i, edges, depth, coef, active_u8, out_w, actual, unmet, scale, rows):
                L, N = w.shape
                init_rows(rows, N)
                for l in range(L):
                    for j in range(N):
                        m = depth[j] if depth[j] < edges[l + 1] else edges[l + 1]  # noqa: FURB136
                        h = m - edges[l]
                        if h < 0.0:  # noqa: PLR1730
                            h = 0.0
                        r = tr[l * tr_l + j * tr_i] * h
                        sat = ts[l * ts_l + j * ts_i] * h
                        sb = sat + coef * sat
                        act = active_u8[j] != 0
                        if not act:
                            sb = np.inf
                        wv = w[l, j]
                        qv = q[l, j]
                        f, t, o, um, sc = cell(wv, qv, r, sb, act, rows[6, j])
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
                        if o < rows[5, j]:  # noqa: PLR1730
                            rows[5, j] = o
                        if qv > rows[7, j]:  # noqa: PLR1730
                            rows[7, j] = qv

            @numba.njit(numba.types.void(ro2, ro1, i8, i8, ro1, i8, i8, ro1, ro1, ro_u8, rw2, rw1, f8), cache=False, fastmath=False)
            def availability(w, tr, tr_l, tr_i, ts, ts_l, ts_i, edges, depth, active_u8, out, bad, coef):
                L, N = w.shape
                for j in range(N):
                    bad[j] = 0
                for l in range(L):
                    for j in range(N):
                        m = depth[j] if depth[j] < edges[l + 1] else edges[l + 1]  # noqa: FURB136
                        h = m - edges[l]
                        if h < 0.0:  # noqa: PLR1730
                            h = 0.0
                        r = tr[l * tr_l + j * tr_i] * h
                        sat = ts[l * ts_l + j * ts_i] * h
                        sb = sat + coef * sat
                        act = active_u8[j] != 0
                        if not act:
                            sb = np.inf
                        wv = w[l, j]
                        if not (wv >= 0.0 and wv <= sb):
                            bad[j] = 1.0
                        out[l, j] = available(wv, r) if act else 0.0
        except fsl.FloatingSoilError:
            raise
        except Exception as exc:
            raise fsl.FloatingBackendUnavailableError(f"the compiled floating withdrawal kernel failed to build ({type(exc).__name__}: {exc}); there is no NumPy fallback") from exc
        _KERNELS = (withdraw, availability)
        return _KERNELS


# ---------------------------------------------------------------------------------------------------------------------------------------------
# public operations
# ---------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class FloatingWithdrawalResult:
    """`state` a fresh `FloatingSoilState` with the SAME clock; `actual_m`, `unmet_m` `(L, N)` depths [m]; `scale` `(L, N)` dimensionless fulfilment fraction; `budget` the fields of
    `subsurface_transpiration.TranspirationBudget`; `plant_actual_m` the per-plant actuals (tuple, empty for a single aggregate request)."""

    state: Any
    actual_m: Any
    unmet_m: Any
    scale: Any
    budget: st.TranspirationBudget
    memory: FloatingWithdrawalMemory
    plant_actual_m: tuple = ()

    @property
    def fulfillment_fraction(self):
        return self.scale


def _quota(column: Any, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn) -> dict:
    if type(column) is fsl.FloatingSoilColumn and column.backend != "cuda" and (host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None):
        raise fsl.FloatingSoilError("host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to the cuda backend")
    return {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": reserve_bytes, "available_bytes_fn": available_bytes_fn, "host_memory_budget_bytes": host_memory_budget_bytes,
                "host_reserve_bytes": host_reserve_bytes, "host_available_bytes_fn": host_available_bytes_fn}


def _check_call(column: Any, state: Any, request: Any) -> float:
    """Metadata only. A CUDA owner: the CURRENT device is checked first (before the state/request metadata reads, readers, copies or accounting)."""
    fsl._gate(column)
    t_s = fsl._check_state(column, state)
    if request is not None:
        fsl._check_vec(column, "requested_water_m", request, (column.n_layers, column.n_cells))
    if column.backend not in ("numpy", "numba", "cuda"):
        raise fsl.FloatingBackendUnavailableError(f"unsupported backend {column.backend!r}")
    return t_s


def _admit_call(column: fsl.FloatingSoilColumn, operation: str, quota: dict) -> FloatingWithdrawalMemory:
    reserve = fsl._count(quota["reserve_bytes"], "reserve_bytes")
    budget = quota["memory_budget_bytes"]
    if column.backend == "cuda":
        from maple_syrup import floating_soil_layers_cuda as fcu
        from maple_syrup import floating_transpiration_cuda as ftc

        t = ftc.operation_terms(operation, column.n_layers, column.n_cells)
        fcu.admit(fcu._rc().require_cuda(), column.device_id, t["outputs"] + t["scratch"], t["host"], quota, f"floating {operation}")
        return FloatingWithdrawalMemory("cuda", operation, column.n_layers, column.n_cells, t["outputs"], t["scratch"], t["host"], t["outputs"] + t["scratch"] + t["host"] + reserve)
    est = estimate_floating_withdrawal_memory(column.n_layers, column.n_cells, backend=column.backend, operation=operation, reserve_bytes=reserve)
    fsl._admit(est.required_bytes, None if budget is None else fsl._count(budget, "memory_budget_bytes"), quota["available_bytes_fn"], f"floating {operation}")
    return est


def floating_layer_availability_m(column: fsl.FloatingSoilColumn, state: fsl.FloatingSoilState, *, memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                                  available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                                  host_available_bytes_fn: Callable[[], int] | None = None) -> Any:
    """PURE READ: fresh `(L, N)` depth `max(W - theta_res * h, 0)` of ACTIVE cells (inactive cells and rock slots 0). The water is validated on EVERY cell (finite, `>= 0`, `<= saturation bound` on active cells;
    finite and `>= 0` on inactive placeholders) and a violation is refused with nothing returned. CUDA owner: a fresh device array and one 8-byte diagnostic read."""
    quota = _quota(column, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn)
    _check_call(column, state, None)
    L, N = column.n_layers, column.n_cells
    _admit_call(column, "availability", quota)
    w = state.water_m
    coef = float(volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0))
    if column.backend == "cuda":
        from maple_syrup import floating_soil_layers_cuda as fcu
        from maple_syrup import floating_transpiration_cuda as ftc

        out, invalid = ftc.availability(fcu._rc().require_cuda(), column, w, coef)
        if invalid:
            raise ss.SubsurfaceError(f"floating_layer_availability_m: {st._FLAG_TEXT[1]} (checked on every cell, inactive included)")
        return out
    out = np.empty((L, N), dtype=np.float64)
    if column.backend == "numba":
        bad = np.empty(N, dtype=np.float64)
        _, kernel = compiled_kernels()
        kernel(w, *_strided(column), _mask_u8(column), out, bad, coef)
        invalid = bool(bad.any())
        del bad
    else:
        residual, satb = _readers(column)
        act = column._maps["active"]
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
        raise ss.SubsurfaceError(f"floating_layer_availability_m: {st._FLAG_TEXT[1]} (checked on every cell, inactive included)")
    return out


def _run(column: fsl.FloatingSoilColumn, w: np.ndarray, q: np.ndarray, out_w, actual, unmet, scale, rows) -> np.ndarray:
    if column.backend == "numba":
        kernel, _ = compiled_kernels()
        kernel(w, q, *_strided(column), float(volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)), _mask_u8(column), out_w, actual, unmet, scale, rows)
    else:
        residual, satb = _readers(column)
        st._fused_numpy(w, q, residual, satb, column._maps["active"], out_w, actual, unmet, scale, rows)
    return _packet(rows)


def publish_budget(column: Any, p: np.ndarray, stage: str) -> st.TranspirationBudget:
    """The shared conservation bookkeeping of one withdrawal packet (volumes over ACTIVE cells, area applied once; typed refusal of unrepresentable globals; the MAPLE FP64 bound)."""
    L = column.n_layers
    area = float(column.cell_area_m2)
    before, after, act, unm, req = (float(v) * area for v in p[:5])
    wmin, qmax = float(p[st._ROW_MIN_AFTER]), float(p[st._ROW_QMAX])
    n_active = column.n_cells if column.n_active is None else column.n_active  # a CUDA owner never downloads its mask: the bound uses the (larger) cell count
    n_terms = 3 * L * int(n_active) + st._PACKET
    scale_m3 = before + req
    if not all(math.isfinite(v) for v in (before, after, act, unm, req, scale_m3)):
        raise ss.SubsurfaceError(f"{stage}: a global volume or the closure scale is not representable in FP64 (before {before!r}, after {after!r}, actual {act!r}, unmet {unm!r}, requested {req!r} m3); refused, nothing was published")
    bound = float(volume_roundoff_bound_m3(n_terms, max(scale_m3, 0.0)))
    budget = st.TranspirationBudget(before, after, req, act, unm, after - before + act, req - act - unm, bound, n_terms, wmin, qmax)
    if not (math.isfinite(after) and math.isfinite(act) and math.isfinite(unm) and math.isfinite(wmin)) or wmin < 0.0:
        raise ss.SubsurfaceError(f"{stage}: withdrawal produced non-finite or negative water: {budget}")
    if abs(budget.residual_m3) > bound or abs(budget.request_residual_m3) > bound:
        raise ss.SubsurfaceError(f"{stage}: transpiration closure violated beyond the FP64 bound: {budget}")
    return budget


def refuse_flags(p: np.ndarray, stage: str) -> None:
    flags = p[st._ROW_FLAGS]
    if flags != 0.0:
        bits = int(flags) if math.isfinite(flags) else 7
        why = "; ".join(t for b, t in st._FLAG_TEXT.items() if bits & b) or "invalid values"
        raise ss.SubsurfaceError(f"{stage}: {why} (checked on every cell, inactive included)")


def floating_withdraw(column: fsl.FloatingSoilColumn, state: fsl.FloatingSoilState, requested_water_m: Any, *, memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                      available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                      host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingWithdrawalResult:
    """Pure withdrawal of ONE aggregate request `(L, N)` [m] over an already defined interval (nothing is scaled by a rate). Order: owner (CUDA: current device first) -> state/request metadata -> admission ->
    allocation -> shared body -> ONE host packet -> reject or publish. The state, request, mask and maps are never written; a refusal publishes nothing; the clock is preserved. Inactive cells keep their water
    bit for bit. CUDA owner: exact `cupy.ndarray`s of the owner's device, nothing transferred, results stay on the device, ONE 64-byte packet read back."""
    quota = _quota(column, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn)
    t_s = _check_call(column, state, requested_water_m)
    L, N = column.n_layers, column.n_cells
    est = _admit_call(column, "withdraw", quota)
    stage = "floating_withdraw"
    if column.backend == "cuda":
        from maple_syrup import floating_soil_layers_cuda as fcu
        from maple_syrup import floating_transpiration_cuda as ftc

        out_w, actual, unmet, scale, p = ftc.withdraw(fcu._rc().require_cuda(), column, state.water_m, requested_water_m, float(volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)))
    else:
        out_w, actual, unmet, scale = (np.empty((L, N), dtype=np.float64) for _ in range(4))
        rows = np.empty((st._N_ROWS, N), dtype=np.float64)
        p = _run(column, state.water_m, requested_water_m, out_w, actual, unmet, scale, rows)
        del rows
    refuse_flags(p, stage)
    budget = publish_budget(column, p, stage)
    return FloatingWithdrawalResult(fsl._new_state(column, out_w, t_s), actual, unmet, scale, budget, est)
