"""Compiled Numba CPU runner of `maple_syrup.soil_water_remap` (backend "numba"). Component/trial only.

One serial loop over the columns; per column the kernel evaluates EXACTLY the arithmetic of `soil_water_remap.reference_remap`, term for term in the same expression
order (ascending sums, `W_i * (overlap / w_i)` products, `min`/`max` of finite values): no fastmath, no prange, no reordering, no clipping. IEEE operations are identical, so the
per-column outputs are expected bitwise equal to the NumPy reference (this is tested, not argued; the module claims only what the tests record). Inputs are only read; every output
is freshly allocated by `run`. Columns whose edges are numerically unchanged are an identity copy (all water bits preserved, fluxes +0). A column with a failed input check is skipped (its outputs stay 0) and later checks are not evaluated for it, as in the reference.

Numba is imported lazily; the kernel is compiled once per process on the first non-empty call (the compile cost is outside any steady-state claim). Missing Numba is a
`SoilRemapBackendUnavailableError`; there is no fallback to the reference. Nothing here was run by its author's tools beyond the recorded test commands.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from maple_syrup.soil_water_remap import (
    FLAG_CLOSURE,
    FLAG_NEW_EDGES,
    FLAG_OLD_EDGES,
    FLAG_OUTPUT,
    FLAG_THETA,
    FLAG_WATER,
    SoilRemapBackendUnavailableError,
)

__all__ = ["numba_available", "numba_versions", "require_numba", "reset_compiled", "run"]

_KERNEL: Any = None
_LOCK = threading.Lock()


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True


def numba_versions() -> dict:
    import numba

    return {"numba": numba.__version__, "numpy": np.__version__}


def require_numba():
    try:
        import numba
    except Exception as exc:  # noqa: BLE001
        raise SoilRemapBackendUnavailableError(f"backend 'numba' requires Numba, which is not importable ({type(exc).__name__}: {exc}); there is no fallback to the NumPy reference") from exc
    return numba


def reset_compiled() -> None:
    global _KERNEL
    with _LOCK:
        _KERNEL = None


def _build_kernel():
    numba = require_numba()
    import math

    f_old, f_new, f_wat, f_th, f_out, f_cl = FLAG_OLD_EDGES, FLAG_NEW_EDGES, FLAG_WATER, FLAG_THETA, FLAG_OUTPUT, FLAG_CLOSURE

    def kernel(eo, en, W, up, lo, out, rem_a, rem_b, add_a, add_b, flags, stats, coef_col, coef_out, L, C):
        for c in range(C):
            flag = 0
            for i in range(L):
                w = eo[i + 1, c] - eo[i, c]
                if not (math.isfinite(w) and w > 0.0):
                    flag |= f_old
                wn_ = en[i + 1, c] - en[i, c]
                if not (math.isfinite(wn_) and wn_ > 0.0):
                    flag |= f_new
                w = eo[i + 1, c] - eo[i, c]
                x = W[i, c]
                if not (math.isfinite(x) and x >= 0.0 and x <= w):
                    flag |= f_wat
            u = up[c]
            v = lo[c]
            if not (math.isfinite(u) and u >= 0.0 and u <= 1.0 and math.isfinite(v) and v >= 0.0 and v <= 1.0):
                flag |= f_th
            if flag != 0:
                flags[c] = flag
                continue
            old_total = 0.0
            scale = 0.0
            for i in range(L):
                w = eo[i + 1, c] - eo[i, c]
                scale = scale + w
                old_total = old_total + W[i, c]
            for j in range(L):
                wn_ = en[j + 1, c] - en[j, c]
                scale = scale + wn_
            ra = 0.0
            rb = 0.0
            for i in range(L):
                w = eo[i + 1, c] - eo[i, c]
                t = min(eo[i + 1, c], en[0, c])
                t = t - eo[i, c]
                t = max(t, 0.0)
                t = t / w
                t = W[i, c] * t
                ra = ra + t
                s = max(eo[i, c], en[L, c])
                s = eo[i + 1, c] - s
                s = max(s, 0.0)
                s = s / w
                s = W[i, c] * s
                rb = rb + s
            a1 = min(eo[0, c], en[L, c])
            a1 = a1 - en[0, c]
            a1 = max(a1, 0.0)
            aa = u * a1
            b1 = max(eo[L, c], en[0, c])
            b1 = en[L, c] - b1
            b1 = max(b1, 0.0)
            ab = v * b1
            new_total = 0.0
            bad_out = False
            for j in range(L):
                acc = 0.0
                for i in range(L):
                    w = eo[i + 1, c] - eo[i, c]
                    p = min(eo[i + 1, c], en[j + 1, c])
                    q = max(eo[i, c], en[j, c])
                    p = p - q
                    p = max(p, 0.0)
                    p = p / w
                    p = W[i, c] * p
                    acc = acc + p
                p = min(en[j + 1, c], eo[0, c])
                p = p - en[j, c]
                p = max(p, 0.0)
                p = u * p
                acc = acc + p
                q = max(en[j, c], eo[L, c])
                q = en[j + 1, c] - q
                q = max(q, 0.0)
                q = v * q
                acc = acc + q
                out[j, c] = acc
                wn_ = en[j + 1, c] - en[j, c]
                tol = wn_ * coef_out
                tol = wn_ + tol
                if not (math.isfinite(acc) and acc <= tol):
                    bad_out = True
                new_total = new_total + acc
            same = True
            for i in range(L + 1):
                if eo[i, c] != en[i, c]:
                    same = False
            if same:  # unchanged geometry: an identity copy keeps every water bit (signed zeros, subnormals); every flux is exactly +0
                new_total = 0.0
                for j in range(L):
                    out[j, c] = W[j, c]
                    new_total = new_total + W[j, c]
                ra = 0.0
                rb = 0.0
                aa = 0.0
                ab = 0.0
            rem_a[c] = ra
            rem_b[c] = rb
            add_a[c] = aa
            add_b[c] = ab
            if not (math.isfinite(ra) and math.isfinite(rb) and math.isfinite(aa) and math.isfinite(ab)):
                bad_out = True
            stats[2, c] = old_total
            stats[3, c] = new_total
            stats[4, c] = scale
            if bad_out:
                flags[c] = f_out
                continue
            d1 = new_total - old_total
            d2 = aa + ab
            d3 = ra + rb
            d2 = d2 - d3
            res = d1 - d2
            bound = scale * coef_col
            stats[0, c] = res
            stats[1, c] = bound
            if not (math.isfinite(res) and math.isfinite(bound) and abs(res) <= bound):  # NaN/inf never pass
                flags[c] = f_cl

    return numba.njit(kernel, nogil=True, fastmath=False, cache=False, error_model="numpy")


def compiled_kernel():
    global _KERNEL
    with _LOCK:
        if _KERNEL is None:
            _KERNEL = _build_kernel()
        return _KERNEL


def run(old_edges_m, new_edges_m, water_m, upper_theta, lower_theta, coef_col: float, coef_out: float) -> tuple:
    """Fresh `(new_water, rem_a, rem_b, add_a, add_b, flags, stats)` host arrays. `C == 0` returns empty arrays without compiling. Inputs are only read."""
    L, C = int(water_m.shape[0]), int(water_m.shape[1])
    out = np.zeros((L, C))
    rem_a, rem_b, add_a, add_b = np.zeros(C), np.zeros(C), np.zeros(C), np.zeros(C)
    flags = np.zeros(C, dtype=np.uint8)
    stats = np.zeros((5, C))
    if C == 0:
        return out, rem_a, rem_b, add_a, add_b, flags, stats
    kernel = compiled_kernel()
    kernel(old_edges_m, new_edges_m, water_m, upper_theta, lower_theta, out, rem_a, rem_b, add_a, add_b, flags, stats, float(coef_col), float(coef_out), L, C)
    return out, rem_a, rem_b, add_a, add_b, flags, stats
