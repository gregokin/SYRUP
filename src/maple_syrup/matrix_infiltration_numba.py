"""Compiled Numba CPU runner of the SYRUP matrix infiltration stage (`maple_syrup.matrix_infiltration`, backend
"numba").

The kernel evaluates EXACTLY the stages of `matrix_infiltration.reference_step`, term for term in the same
expression order, as a serial cell loop (layer loops inside): no fastmath, no prange, no changed order, no clipping.
The implicit Green-Ampt solve is the scalar form of the reference's masked vectorised Newton iteration (same start
bound, same stop rules, same iteration count); only `math.log1p` / `math.log` (LLVM/libm) may differ from NumPy's by
an ulp. Declared agreement with the reference: rtol 2e-12 / atol 1e-14 on state, history and fluxes (bitwise where no
logarithm is evaluated: unponded intervals, b = 0 or K = 0).

Numba is imported lazily; `prepare_runner` compiles the kernel once (a one-cell warm-up with sealed read-only maps,
so the cost sits at `prepare_matrix`, outside steps). Missing Numba is a `MatrixBackendUnavailableError`; there is
no fallback. Every output array is freshly allocated per call; inputs and the operator's maps are never written.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from maple_syrup.matrix_infiltration import (
    LOG1PMX_TERMS,
    NEWTON_MAX_ITERATIONS,
    NO_FLAG,
    ROW_COUNT,
    KernelMaps,
    MatrixBackendUnavailableError,
    StepArrays,
)

__all__ = ["compiled_kernel", "numba_available", "numba_versions", "prepare_runner", "require_numba", "reset_compiled"]

_KERNEL: Any = None
_LOCK = threading.Lock()


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def numba_versions() -> dict[str, str] | None:
    try:
        import llvmlite
        import numba
    except ImportError:
        return None
    return {"numba": numba.__version__, "llvmlite": llvmlite.__version__}


def require_numba() -> None:
    if not numba_available():
        raise MatrixBackendUnavailableError(
            "backend 'numba' requires Numba (optional extra maple-syrup[numba]), which is not importable; there is no "
            "fallback to the NumPy reference (select backend='numpy' explicitly)")


def _build(numba: Any):
    jit = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")

    @jit
    def log1pmx(u):
        """log1p(u) - u for u >= 0 without cancellation (atanh series for u <= 1, LOG1PMX_TERMS fixed terms)."""
        if u > 1.0:
            return math.log1p(u) - u
        z = u / (2.0 + u)
        z2 = z * z
        p = z * z2
        s = p / 3.0
        for k in range(2, LOG1PMX_TERMS + 1):
            p = p * z2
            s = s + p / float(2 * k + 1)
        return (-2.0 * z2) / (1.0 - z) + 2.0 * s

    @jit
    def ga_increment(j_old, b, r_dt, tol, max_iter):
        """x of x - b log1p(x/(j_old + b)) = r_dt; returns (x, iterations, converged). Residual in the
        well-conditioned form psi = (y J - b R)/D - b log1pmx((R + y)/D)."""
        if r_dt == 0.0 or b == 0.0:
            return r_dt, 0.0, True
        d = j_old + b
        y = math.sqrt(r_dt * r_dt + (2.0 * r_dt) * b)
        if j_old > 0.0:
            y2 = (r_dt * b) / j_old
            if y2 < y:  # noqa: PLR1730 - explicit ordered FP comparison, NaN-propagating; no min()
                y = y2
        dr = d + r_dt
        jr = j_old + r_dt
        br = b * r_dt
        it = 0.0
        converged = False
        for _ in range(max_iter):
            s = r_dt + y
            u = s / d
            if math.isinf(u):
                psi = y - b * (math.log(s) - math.log(d))
            else:
                psi = ((y * j_old) - br) / d - b * log1pmx(u)
            if psi <= 0.0:
                converged = True
                break
            dpsi = (jr + y) / (dr + y)
            step = psi / dpsi
            y_new = y - step
            it += 1.0
            if y_new >= y:
                converged = True
                break
            if y_new < 0.0:  # noqa: PLR1730
                y_new = 0.0
            y = y_new
            if step <= tol * y:
                converged = True
                break
        return r_dt + y, it, converged

    @jit
    def step_kernel(pool_in, standing_in, rate_in, w_in, wet_in, cum_in, active, dt, kmap, psimap, gmap, dz, fc_m,
                    sat_m, cap_tol, cell_tol, tol, max_iter, no_flag, pool_out, w_out, wet_out, cum_out, intake_out, rows):
        n_layers = w_in.shape[0]
        n_cells = w_in.shape[1]
        for i in range(n_cells):
            a = active[i] != 0
            f = 0
            before = 0.0
            for l in range(n_layers):
                wl = w_in[l, i]
                if not math.isfinite(wl):
                    f |= 1 << 0
                if wl < 0.0:
                    f |= 1 << 1
                if wl > sat_m[l] + cap_tol[l]:
                    f |= 1 << 2
                before = before + wl
                w_out[l, i] = wl
            pool = pool_in[i]
            if not math.isfinite(pool):
                f |= 1 << 3
            if pool < 0.0:
                f |= 1 << 4
            h0 = standing_in[i]
            if not math.isfinite(h0):
                f |= 1 << 5
            if h0 < 0.0:
                f |= 1 << 6
            r = rate_in[i]
            if not math.isfinite(r):
                f |= 1 << 7
            if r < 0.0:
                f |= 1 << 8
            if (not a) and r != 0.0:
                f |= 1 << 9
            jin = wet_in[i]
            if not math.isfinite(jin):
                f |= 1 << 10
            if jin < 0.0:
                f |= 1 << 11
            cin = cum_in[i]
            if not math.isfinite(cin):
                f |= 1 << 12
            if cin < 0.0:
                f |= 1 << 13
            pool_o = pool
            j_o = jin
            c_o = cin
            after = before
            candidate = 0.0
            intake = 0.0
            pre = 0.0
            post = 0.0
            rejected = 0.0
            unused = 0.0
            onset = 0.0
            ponded = 0.0
            iters = 0.0
            inc = r * dt
            contact = h0 > 0.0
            if a and (not contact):
                scale_c = (pool + inc) * cell_tol
                if pool > inc + scale_c:
                    f |= 1 << 14
            if a and dt > 0.0:
                k = kmap[i]
                psi = psimap[i]
                g = gmap[i]
                w0 = w_in[0, i]
                deficit = fc_m[0] - w0
                if deficit < 0.0:  # noqa: PLR1730
                    deficit = 0.0
                b = psi * (deficit / dz[0])
                room = sat_m[0] - w0
                if room < 0.0:  # noqa: PLR1730
                    room = 0.0
                t_u = 0.0
                tau = 0.0
                run_post = False
                inside = False
                if contact:
                    run_post = True
                    tau = dt
                elif r <= k:
                    t_u = dt
                else:
                    fs = ((g * b) * k) / (r - k)
                    if jin >= fs:
                        run_post = True
                        tau = dt
                    else:
                        t_p = (fs - jin) / r
                        if t_p < dt:
                            t_u = t_p
                            tau = dt - t_p
                            run_post = True
                            inside = True
                        else:
                            t_u = dt
                c1 = r * t_u
                pre = c1
                if pool < pre:  # noqa: PLR1730
                    pre = pool
                if room < pre:  # noqa: PLR1730
                    pre = room
                candidate = c1
                if (t_u > 0.0) and (pre < c1):
                    run_post = False
                    inside = False
                if inside:
                    onset = 1.0
                j1 = jin + pre
                adv = 0.0
                if run_post:
                    ponded = 1.0
                    x, it, conv = ga_increment(j1, b, k * tau, tol, max_iter)
                    iters = it
                    if not conv:
                        f |= 1 << 15
                    c2 = g * x
                    candidate = candidate + c2
                    pool_left = pool - pre
                    room_left = room - pre
                    post = c2
                    if pool_left < post:  # noqa: PLR1730
                        post = pool_left
                    if room_left < post:  # noqa: PLR1730
                        post = room_left
                    if g > 0.0:
                        adv = post / g
                    j_o = j1 + adv
                else:
                    j_o = j1
                intake = pre + post
                bounded = candidate
                if pool < bounded:  # noqa: PLR1730
                    bounded = pool
                rejected = bounded - intake
                unused = candidate - bounded
                w_out[0, i] = w0 + intake
                pool_o = pool - intake
                c_o = cin + intake
                after = 0.0
                for l in range(n_layers):
                    w = w_out[l, i]
                    if not math.isfinite(w):
                        f |= 1 << 16
                    if w < 0.0:
                        f |= 1 << 17
                    if w > sat_m[l] + cap_tol[l]:
                        f |= 1 << 18
                    after = after + w
                if (not math.isfinite(pool_o)) or (not math.isfinite(j_o)) or (not math.isfinite(c_o)):
                    f |= 1 << 16
                if pool_o < 0.0 or j_o < 0.0 or c_o < 0.0:
                    f |= 1 << 17
                scale = ((((before + after) + pool) + pool_o) + intake) * cell_tol
                closure = (after - before) - intake
                if abs(closure) > scale:
                    f |= 1 << 19
                closure_p = (pool - pool_o) - intake
                if abs(closure_p) > scale:
                    f |= 1 << 20
                adv_total = pre + adv
                h_scale = ((jin + j_o) + adv_total) * cell_tol
                if abs((j_o - jin) - adv_total) > h_scale:
                    f |= 1 << 21
                c_scale = ((cin + c_o) + intake) * cell_tol
                if abs((c_o - cin) - intake) > c_scale:
                    f |= 1 << 21
                if (not math.isfinite(candidate)) or (not math.isfinite(intake)):
                    f |= 1 << 22
            pool_out[i] = pool_o
            wet_out[i] = j_o
            cum_out[i] = c_o
            if a:
                intake_out[i] = intake
                rows[0, i] = before
                rows[1, i] = after
                rows[2, i] = pool
                rows[3, i] = pool_o
                rows[4, i] = candidate
                rows[5, i] = intake
                rows[6, i] = pre
                rows[7, i] = post
                rows[8, i] = rejected
                rows[9, i] = unused
                rows[10, i] = jin
                rows[11, i] = j_o
                rows[12, i] = onset
                rows[13, i] = ponded
                rows[14, i] = iters
            else:
                intake_out[i] = 0.0
                for row in range(ROW_COUNT - 1):
                    rows[row, i] = 0.0
            if f == 0:
                rows[15, i] = no_flag
            else:
                rows[15, i] = float(f & -f)

    return step_kernel


def compiled_kernel():
    """The process-wide compiled kernel (built lazily; Numba imported here only)."""
    global _KERNEL
    if _KERNEL is None:
        require_numba()
        import numba

        with _LOCK:
            if _KERNEL is None:
                _KERNEL = _build(numba)
    return _KERNEL


def reset_compiled() -> None:
    """Drop the compiled kernel (tests of the compile path only)."""
    global _KERNEL
    with _LOCK:
        _KERNEL = None


def _launch(kernel, maps: KernelMaps, cell_tol: float, tol: float, pool_in, standing_in, rate_in, water_in, wetting_in,
            cumulative_in, active_u8: np.ndarray, dt: float) -> StepArrays:
    n_layers, n_cells = water_in.shape
    pool_out = np.empty(n_cells, dtype=np.float64)
    water_out = np.empty((n_layers, n_cells), dtype=np.float64)
    wet_out = np.empty(n_cells, dtype=np.float64)
    cum_out = np.empty(n_cells, dtype=np.float64)
    intake = np.empty(n_cells, dtype=np.float64)
    rows = np.empty((ROW_COUNT, n_cells), dtype=np.float64)
    kernel(pool_in, standing_in, rate_in, water_in, wetting_in, cumulative_in, active_u8, float(dt),
           maps.conductivity_m_s, maps.suction_m, maps.vegetation, maps.thickness_m, maps.field_capacity_m,
           maps.saturation_m, maps.capacity_tolerance_m, float(cell_tol), float(tol), int(NEWTON_MAX_ITERATIONS),
           NO_FLAG, pool_out, water_out, wet_out, cum_out, intake, rows)
    return StepArrays(pool_out, water_out, wet_out, cum_out, intake, rows)


def _readonly(array: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    flat = np.frombuffer(a.tobytes(), dtype=np.float64)
    return flat if a.ndim == 1 else flat.reshape(a.shape)


def prepare_runner(maps: KernelMaps, cell_tol: float, tol: float):
    """Compile (if needed) and warm up the kernel for the SEALED (read-only) map types; return `(runner, info)`."""
    kernel = compiled_kernel()
    n_layers = int(maps.thickness_m.shape[0])
    warm = KernelMaps(_readonly(np.zeros(1)), _readonly(np.zeros(1)), _readonly(np.zeros(1)), _readonly(maps.thickness_m),
                      _readonly(maps.field_capacity_m), _readonly(maps.saturation_m), _readonly(maps.capacity_tolerance_m))
    z = np.zeros(1)
    _launch(kernel, warm, cell_tol, tol, z, z, z, np.zeros((n_layers, 1)), z, z, np.zeros(1, np.uint8), 1.0)

    def run(op, maps_internal, pool_in, standing_in, rate_in, water_in, wetting_in, cumulative_in, dt):
        return _launch(kernel, maps_internal, cell_tol, tol, pool_in, standing_in, rate_in, water_in, wetting_in,
                       cumulative_in, op.column.active_u8, dt)

    info = {"implementation": "numba serial cell loop, fastmath=False; sealed read-only maps; scalar Newton",
            "versions": numba_versions(),
            "libm": "LLVM/libm log1p/log (may differ from NumPy by an ulp); declared bound rtol 2e-12 / atol 1e-14",
            "parameter_host_bytes": int(sum(int(a.nbytes) for a in maps))}
    return run, info
