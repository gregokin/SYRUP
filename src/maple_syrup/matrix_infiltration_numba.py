"""Compiled Numba CPU runner of the SYRUP matrix infiltration stage (`maple_syrup.matrix_infiltration`, backend
"numba").

The kernel evaluates EXACTLY the stages of `matrix_infiltration.reference_step`, term for term in the same
expression order, as a serial cell loop (layer loops inside): no fastmath, no prange, no changed order, no clipping.
The implicit Green-Ampt solve is the scalar form of the reference's masked vectorised Newton iteration (same start
bound, same stop rules, same iteration count); only `math.log1p` / `math.log` (LLVM/libm) may differ from NumPy's by
an ulp. Declared agreement with the reference: rtol 2e-12 / atol 1e-14 on state, history and fluxes (bitwise where no
logarithm is evaluated: unponded intervals, b = 0 or K = 0).

Phase 2 S2-B2.a (ONE physical body, two compile-time variants): the cell loop is written ONCE in `_build(numba, common)` and compiled twice by Numba's
dead-branch pruning of the closure constant `common` (the S2-A.3 technique of `subsurface_numba`). The COMMON variant reads the operator's cached `(L,)` vectors AS THEY ARE
(`compiled_kernel`, a thin wrapper with the historical 25-argument signature calls the explicit-signature body once per step; its arrays, options, outputs and arithmetic are the
previous ones); the SPATIAL variant (`compiled_spatial_kernel`, used by `maple_syrup.spatial_matrix_infiltration`) reads the explicit owner's RAW `theta_field_capacity`, `theta_saturation`
and `dz` maps through strides (0 scalar, 1 shared, N per cell) and derives `fc = theta_fc dz`, `sat = theta_sat dz`, `cap_tol = coef sat` with the very IEEE products of
`subsurface.soil_layers`; only two `(L,)` local rows (`sat`, `cap_tol`) exist per call, never an `(L, N)` tensor. No per-element helper receives an array argument.

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

__all__ = [
    "compiled_floating_kernel",
    "compiled_kernel",
    "compiled_spatial_kernel",
    "floating_launch",
    "local_scratch_doubles",
    "numba_available",
    "numba_versions",
    "prepare_runner",
    "require_numba",
    "reset_compiled",
    "spatial_launch",
]

_KERNEL: Any = None
_SPATIAL_KERNEL: Any = None
LOCAL_DOUBLES_PER_LAYER_SPATIAL = 2  # SPATIAL variant per-call local rows: sat and cap_tol, one value per layer (Numba NRT heap, independent of the cell count)
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


def _build(numba: Any, common: bool = True, floating: bool = False):
    """Build the physical body variant. `common=True` returns the historical 25-argument wrapper around the common variant of the body; `common=False` returns the spatial variant;
    `common=False, floating=True` the D2 FLOATING variant (private, used by `floating_matrix_infiltration`): the same body, where `dz` carries the nominal edges `(L+1,)` and `cap_tol` the per-cell
    soil depth `(N,)`; the layer bounds are `theta_sat * h`, the layer-0 field capacity `theta_fc * h0` with the ACTUAL porous thickness `h = max(0, min(D, e_{l+1}) - e_l)`, the layer-0 denominator is
    `h0` where `h0 > 0` and the safe stand-in 1.0 otherwise (zero-capacity arithmetic only), and an INACTIVE cell carries no capacity law (bound `+inf`)."""
    jit = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
    # helpers that receive arrays are inlined at IR level (a non-inlined array argument costs NRT reference counting at every call; measured in S2-A.3 / S2-B1.b)
    jit_inline = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy", inline="always")
    f8, i8, u8 = numba.types.float64, numba.types.int64, numba.types.uint8
    ro1, ro2 = numba.types.Array(f8, 1, "C", readonly=True), numba.types.Array(f8, 2, "C", readonly=True)
    ro_u8 = numba.types.Array(u8, 1, "C", readonly=True)
    rw1, rw2 = numba.types.Array(f8, 1, "C"), numba.types.Array(f8, 2, "C")

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

    @jit_inline
    def layer_stride(kind, n):
        # stride of the layer index: 0 scalar, 1 shared (one value per layer), N per cell (flat C-contiguous (rows, N))
        if kind == 0:
            return 0
        if kind == 1:
            return 1
        return n

    @jit_inline
    def cell_stride(kind):
        return 1 if kind == 2 else 0

    @jit_inline
    def bind_layers(i, n_layers, dz, dz_l, dz_i, b_sat, s_l, s_i, coef, sat_c, tol_c):
        # SPATIAL variant only: sat = theta_sat dz and cap_tol = coef sat of this cell, the very IEEE products of `soil_layers`
        for l in range(n_layers):
            s = b_sat[l * s_l + i * s_i] * dz[l * dz_l + i * dz_i]
            sat_c[l] = s
            tol_c[l] = coef * s

    @jit_inline
    def bind_floating(i, n_layers, edges, depth, b_sat, s_l, s_i, coef, active_cell, sat_c, tol_c):
        # FLOATING variant only: sat = theta_sat h of this cell (h from the shared nominal edges and the cell's soil depth); an inactive cell has no capacity law
        d = depth[i]
        for l in range(n_layers):
            top = edges[l + 1]
            m = d if d < top else top  # noqa: FURB136
            h = m - edges[l]
            if h < 0.0:  # noqa: PLR1730
                h = 0.0
            s = b_sat[l * s_l + i * s_i] * h
            if not active_cell:
                s = np.inf
            sat_c[l] = s
            tol_c[l] = coef * s

    signature = numba.types.void(
        ro1, ro1, ro1, ro2, ro1, ro1, ro_u8, f8,  # pool_in, standing_in, rate_in, w_in, wet_in, cum_in, active, dt
        ro1, i8, ro1, i8, ro1, i8,  # kmap,k_i | psimap,p_i | gmap,g_i (strides: 1 per cell, 0 one scalar)
        ro1, i8, ro1, i8, ro1, i8, ro1, f8,  # dz,k_dz | fc (theta_fc),k_fc | sat (theta_sat),k_sat | cap_tol | coef
        f8, f8, i8, f8,  # cell_tol, tol, max_iter, no_flag
        rw1, rw2, rw1, rw1, rw1, rw2)  # pool_out, w_out, wet_out, cum_out, intake_out, rows

    @numba.njit(signature, cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
    def step_core(pool_in, standing_in, rate_in, w_in, wet_in, cum_in, active, dt, kmap, k_i, psimap, p_i, gmap, g_i, dz, k_dz,
                  fc_m, k_fc, sat_m, k_sat, cap_tol, coef, cell_tol, tol, max_iter, no_flag, pool_out, w_out, wet_out, cum_out,
                  intake_out, rows):
        n_layers = w_in.shape[0]
        n_cells = w_in.shape[1]
        # `common` is a closure constant: exactly one branch of each `if common` survives compilation.
        if common:
            sat_c = sat_m  # the cached (L,) metre arrays ARE the bound values: aliased, nothing allocated
            tol_c = cap_tol
        else:
            sat_c = np.empty(n_layers, dtype=np.float64)
            tol_c = np.empty(n_layers, dtype=np.float64)
            dz_l, dz_i = layer_stride(k_dz, n_cells), cell_stride(k_dz)
            f_i = cell_stride(k_fc)
            s_l, s_i = layer_stride(k_sat, n_cells), cell_stride(k_sat)
            layers_cell = k_dz == 2 or k_sat == 2
            if floating:
                layers_cell = True
            if not layers_cell:
                bind_layers(0, n_layers, dz, dz_l, dz_i, sat_m, s_l, s_i, coef, sat_c, tol_c)
        for i in range(n_cells):
            a = active[i] != 0
            if common:
                dz0 = dz[0]
                fc0 = fc_m[0]
            elif floating:
                bind_floating(i, n_layers, dz, cap_tol, sat_m, s_l, s_i, coef, a, sat_c, tol_c)
                top0 = dz[1]
                d0 = cap_tol[i]
                m0 = d0 if d0 < top0 else top0  # noqa: FURB136
                h0c = m0 - dz[0]
                if h0c < 0.0:  # noqa: PLR1730
                    h0c = 0.0
                fc0 = fc_m[i * f_i] * h0c  # fc = theta_fc h0 (0 on a rock slot)
                dz0 = h0c if h0c > 0.0 else 1.0  # safe denominator ONLY where there is no porous layer 0
            else:
                if layers_cell:
                    bind_layers(i, n_layers, dz, dz_l, dz_i, sat_m, s_l, s_i, coef, sat_c, tol_c)
                dz0 = dz[i * dz_i]  # layer 0 (the layer term vanishes)
                fc0 = fc_m[i * f_i] * dz0  # fc = theta_fc dz
            f = 0
            before = 0.0
            for l in range(n_layers):
                wl = w_in[l, i]
                if not math.isfinite(wl):
                    f |= 1 << 0
                if wl < 0.0:
                    f |= 1 << 1
                if wl > sat_c[l] + tol_c[l]:
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
                if common:
                    k = kmap[i]
                    psi = psimap[i]
                    g = gmap[i]
                else:
                    k = kmap[i * k_i]
                    psi = psimap[i * p_i]
                    g = gmap[i * g_i]
                w0 = w_in[0, i]
                deficit = fc0 - w0
                if deficit < 0.0:  # noqa: PLR1730
                    deficit = 0.0
                b = psi * (deficit / dz0)
                room = sat_c[0] - w0
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
                # ONE combined actual intake (S2-B2.c review02), the same ordering as the NumPy body: bound `pre + post` by the pool, then by the room, BEFORE anything is published. Inactive bound:
                # byte-identical to the previous arithmetic. Active bound: the ponded actual is re-derived, `post = intake - pre` (>= 0 exactly: pre <= pool and pre <= room), so pool, soil, history and ledger
                # share the SAME actuals. No epsilon, no clipping afterwards.
                intake = pre + post
                reduced = False
                if intake > pool:
                    intake = pool
                    reduced = True
                if intake > room:
                    intake = room
                    reduced = True
                if run_post:
                    if reduced:
                        post = intake - pre
                    if g > 0.0:
                        adv = post / g
                    j_o = j1 + adv
                else:
                    j_o = j1
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
                    if w > sat_c[l] + tol_c[l]:
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

    if not common:
        return step_core

    @jit
    def step_kernel(pool_in, standing_in, rate_in, w_in, wet_in, cum_in, active, dt, kmap, psimap, gmap, dz, fc_m,
                    sat_m, cap_tol, cell_tol, tol, max_iter, no_flag, pool_out, w_out, wet_out, cum_out, intake_out, rows):
        # historical 25-argument signature: one call of the explicit-signature body per step (per-cell strides 1, cached (L,) vectors)
        step_core(pool_in, standing_in, rate_in, w_in, wet_in, cum_in, active, dt, kmap, 1, psimap, 1, gmap, 1, dz, 1, fc_m, 1,
                  sat_m, 1, cap_tol, 0.0, cell_tol, tol, max_iter, no_flag, pool_out, w_out, wet_out, cum_out, intake_out, rows)

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


def compiled_spatial_kernel():
    """The process-wide compiled SPATIAL variant of the same body (explicit read-only signature; built lazily at an explicit spatial preparation, never per owner or per call)."""
    global _SPATIAL_KERNEL
    if _SPATIAL_KERNEL is None:
        require_numba()
        import numba

        with _LOCK:
            if _SPATIAL_KERNEL is None:
                _SPATIAL_KERNEL = _build(numba, False)
    return _SPATIAL_KERNEL


_FLOATING_KERNEL: Any = None


def compiled_floating_kernel():
    """The process-wide compiled FLOATING variant of the same body (D2; built lazily at a floating preparation, never per owner or per call)."""
    global _FLOATING_KERNEL
    if _FLOATING_KERNEL is None:
        require_numba()
        import numba

        with _LOCK:
            if _FLOATING_KERNEL is None:
                _FLOATING_KERNEL = _build(numba, False, True)
    return _FLOATING_KERNEL


def floating_launch(pool_in, standing_in, rate_in, water_in, wetting_in, cumulative_in, mask_u8: np.ndarray, dt: float, *, k, k_i, psi, p_i, g, g_i, edges, depth, theta_fc, k_fc, theta_sat, k_sat,
                    coef: float, cell_tol: float, tol: float) -> StepArrays:
    """One compiled step of a FLOATING matrix operator (`floating_matrix_infiltration` calls this; nothing else should). `edges` `(L+1,)` and `depth` `(N,)` are the owner's read-only maps;
    `theta_fc`/`theta_sat` its retained raw maps with layout codes 0 scalar / 1 shared / 2 per cell. Fresh output arrays only."""
    kernel = compiled_floating_kernel()
    n_layers, n_cells = water_in.shape
    pool_out = np.empty(n_cells, dtype=np.float64)
    water_out = np.empty((n_layers, n_cells), dtype=np.float64)
    wet_out = np.empty(n_cells, dtype=np.float64)
    cum_out = np.empty(n_cells, dtype=np.float64)
    intake = np.empty(n_cells, dtype=np.float64)
    rows = np.empty((ROW_COUNT, n_cells), dtype=np.float64)
    kernel(pool_in, standing_in, rate_in, water_in, wetting_in, cumulative_in, mask_u8, float(dt), k, int(k_i), psi, int(p_i), g, int(g_i), edges, 1, theta_fc, int(k_fc), theta_sat, int(k_sat),
           depth, float(coef), float(cell_tol), float(tol), int(NEWTON_MAX_ITERATIONS), NO_FLAG, pool_out, water_out, wet_out, cum_out, intake, rows)
    return StepArrays(pool_out, water_out, wet_out, cum_out, intake, rows)


def local_scratch_doubles(n_layers: int, *, spatial: bool = True) -> int:
    """Doubles of the kernel's per-call LOCAL rows (Numba NRT heap, independent of N): the SPATIAL variant 2 per layer, the COMMON variant none (its bound values alias the cached arrays)."""
    return LOCAL_DOUBLES_PER_LAYER_SPATIAL * int(n_layers) if spatial else 0


def reset_compiled() -> None:
    """Drop the compiled kernels (tests of the compile path only)."""
    global _KERNEL, _SPATIAL_KERNEL, _FLOATING_KERNEL
    with _LOCK:
        _KERNEL = None
        _SPATIAL_KERNEL = None
        _FLOATING_KERNEL = None


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


def spatial_launch(pool_in, standing_in, rate_in, water_in, wetting_in, cumulative_in, mask_u8: np.ndarray, dt: float, *, k, k_i, psi, p_i, g, g_i, dz, k_dz, theta_fc, k_fc, theta_sat, k_sat,
                   coef: float, cell_tol: float, tol: float) -> StepArrays:
    """One compiled step of an EXPLICIT SPATIAL matrix operator (`spatial_matrix_infiltration` calls this; nothing else should). `k`/`psi`/`g` are flat read-only `(N,)` maps (stride 1) or
    one-element arrays (stride 0); `dz`, `theta_fc`, `theta_sat` are the owner's retained RAW maps as flat read-only views with layout codes 0 scalar / 1 shared / 2 per cell. Fresh output arrays
    only; nothing of the owner (map, mask) is copied or written."""
    kernel = compiled_spatial_kernel()
    n_layers, n_cells = water_in.shape
    pool_out = np.empty(n_cells, dtype=np.float64)
    water_out = np.empty((n_layers, n_cells), dtype=np.float64)
    wet_out = np.empty(n_cells, dtype=np.float64)
    cum_out = np.empty(n_cells, dtype=np.float64)
    intake = np.empty(n_cells, dtype=np.float64)
    rows = np.empty((ROW_COUNT, n_cells), dtype=np.float64)
    kernel(pool_in, standing_in, rate_in, water_in, wetting_in, cumulative_in, mask_u8, float(dt), k, int(k_i), psi, int(p_i), g, int(g_i), dz, int(k_dz), theta_fc, int(k_fc), theta_sat, int(k_sat),
           dz, float(coef), float(cell_tol), float(tol), int(NEWTON_MAX_ITERATIONS), NO_FLAG, pool_out, water_out, wet_out, cum_out, intake, rows)
    return StepArrays(pool_out, water_out, wet_out, cum_out, intake, rows)
