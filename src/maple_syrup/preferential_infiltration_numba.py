"""Compiled Numba CPU runner of the SYRUP preferential infiltration stage (`maple_syrup.preferential_infiltration`,
backend "numba").

The kernel evaluates EXACTLY the stages of `preferential_infiltration.reference_step`, term for term in the same
expression order, as serial cell / layer loops: no fastmath, no prange, no changed order, no clipping. Identical IEEE
operations give identical bits; `math.expm1` (LLVM/libm) may differ from NumPy's `expm1` by an ulp, which enters
only the isolated attempted amount H x (-expm1(-(k dt))). The declared agreement bound with the reference is
therefore rtol 2e-12 / atol 1e-14 on the state and fluxes (bitwise where every beta is 0 or 1, since
expm1(0) = 0 and expm1(-inf) = -1 are exact in both libraries).

Phase 2 S2-B2.b (ONE physical body, two compile-time variants): the cell loop is written ONCE in `_build(numba, common)` and compiled twice by dead-branch pruning of the closure
constant `common` (the S2-A.3 / S2-B2.a technique). The COMMON variant reads the operator's cached `(L,)` vectors as they are (`compiled_kernel`, a thin wrapper with the historical 16-argument
signature calls the explicit-signature body once per step; arrays, options, outputs and arithmetic are the previous ones). The SPATIAL variant (`compiled_spatial_kernel`, used by
`maple_syrup.spatial_preferential_infiltration`) reads the explicit owner's RAW `theta_saturation` and `dz` maps through strides (0 scalar, 1 shared, N per cell) and derives
`sat = theta_sat dz`, `cap_tol = coef sat` with the very IEEE products of `subsurface.soil_layers`; only two `(L,)` local rows exist per call, never an `(L, N)` tensor.

Numba is imported lazily; `prepare_runner` compiles the kernel once (a one-cell warm-up call, so the cost sits at
`prepare_preferential`, outside steps). Missing Numba is a `PreferentialBackendUnavailableError`; there is no
fallback to the reference. Every output array is freshly allocated per call; the kernel reads the caller's arrays
and the operator's owned maps and never writes them.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from maple_syrup.preferential_infiltration import (
    NO_FLAG,
    ROW_COUNT,
    KernelMaps,
    PreferentialBackendUnavailableError,
    StepArrays,
)

__all__ = ["compiled_floating_kernel", "compiled_kernel", "compiled_spatial_kernel", "floating_launch", "local_scratch_doubles", "numba_available", "numba_versions", "prepare_runner", "require_numba", "reset_compiled",
           "spatial_launch"]

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
        raise PreferentialBackendUnavailableError(
            "backend 'numba' requires Numba (optional extra maple-syrup[numba]), which is not importable; there is no "
            "fallback to the NumPy reference (select backend='numpy' explicitly)")


def _build(numba: Any, common: bool = True, floating: bool = False):
    """Build the physical body variant. `common=True` returns the historical 16-argument wrapper around the common variant of the body; `common=False` returns the spatial variant;
    `common=False, floating=True` the D2 FLOATING variant (private, used by `floating_preferential_infiltration`): the same body, where `dz` carries the nominal edges `(L+1,)` and `sat_m` the
    per-cell soil depth `(N,)`, the layer bounds are `theta_sat * h` with the ACTUAL porous thickness `h = max(0, min(D, e_{l+1}) - e_l)`, and an INACTIVE cell carries no capacity law (bound `+inf`)."""
    jit = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
    # helpers that receive arrays are inlined at IR level (a non-inlined array argument costs NRT reference counting at every call; measured in S2-A.3 / S2-B1.b)
    jit_inline = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy", inline="always")
    f8, i8, u8 = numba.types.float64, numba.types.int64, numba.types.uint8
    ro1, ro2 = numba.types.Array(f8, 1, "C", readonly=True), numba.types.Array(f8, 2, "C", readonly=True)
    ro_u8 = numba.types.Array(u8, 1, "C", readonly=True)
    rw1, rw2 = numba.types.Array(f8, 1, "C"), numba.types.Array(f8, 2, "C")

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
        # FLOATING variant only: h of this cell/layer from the shared nominal edges and the cell's soil depth; an inactive cell has no capacity law
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
        ro1, ro2, ro_u8, f8,  # surface_in, w_in, active, dt
        ro1, ro1, i8, ro2,  # rate, max_rate, k_mr (stride: 1 per cell, 0 one scalar), recv
        ro1, i8, ro1, i8,  # dz,k_dz | theta_sat,k_sat (layout codes)
        ro1, ro1, f8, f8, f8,  # sat_m, cap_tol (common vectors), coef, cell_tol, no_flag
        rw1, rw2, rw2, rw1, rw2)  # surface_out, w_out, intake, total, rows

    @numba.njit(signature, cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
    def step_core(surface_in, w_in, active, dt, rate, max_rate, k_mr, recv, dz, k_dz, b_sat, k_sat, sat_m, cap_tol, coef, cell_tol, no_flag,
                  surface_out, w_out, intake, total, rows):
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
            s_l, s_i = layer_stride(k_sat, n_cells), cell_stride(k_sat)
            layers_cell = k_dz == 2 or k_sat == 2
            if floating:
                layers_cell = True
            if not layers_cell:
                bind_layers(0, n_layers, dz, dz_l, dz_i, b_sat, s_l, s_i, coef, sat_c, tol_c)
        for i in range(n_cells):
            a = active[i] != 0
            if not common:
                if floating:
                    bind_floating(i, n_layers, dz, sat_m, b_sat, s_l, s_i, coef, a, sat_c, tol_c)
                elif layers_cell:
                    bind_layers(i, n_layers, dz, dz_l, dz_i, b_sat, s_l, s_i, coef, sat_c, tol_c)
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
                intake[l, i] = 0.0
            h_in = surface_in[i]
            if not math.isfinite(h_in):
                f |= 1 << 3
            if h_in < 0.0:
                f |= 1 << 4
            left = h_in
            total_v = 0.0
            isolated_v = 0.0
            attempted_v = 0.0
            after = before
            if a and dt > 0.0:
                # isolated attempt: H x (-expm1(-(k dt)))
                x = rate[i] * dt
                fr = -math.expm1(-x)
                attempted = h_in * fr
                isolated_v = attempted
                # ONE aggregate rate cap
                if common:
                    cap = max_rate[i] * dt
                else:
                    cap = max_rate[i * k_mr] * dt
                if cap < attempted:  # noqa: PLR1730 - explicit ordered FP comparison, NaN-propagating; no min()
                    attempted = cap
                attempted_v = attempted
                # receiver allocation, top layer first; bounded by the room and the remaining surface water
                for l in range(n_layers):
                    alloc = attempted * recv[l, i]
                    room = sat_c[l] - w_out[l, i]
                    if room < 0.0:  # noqa: PLR1730
                        room = 0.0
                    if room < alloc:  # noqa: PLR1730
                        alloc = room
                    if left < alloc:  # noqa: PLR1730
                        alloc = left
                    w_out[l, i] = w_out[l, i] + alloc
                    left = left - alloc
                    intake[l, i] = alloc
                    total_v = total_v + alloc
                # output checks
                after = 0.0
                for l in range(n_layers):
                    w = w_out[l, i]
                    if not math.isfinite(w):
                        f |= 1 << 5
                    if w < 0.0:
                        f |= 1 << 6
                    if w > sat_c[l] + tol_c[l]:
                        f |= 1 << 7
                    after = after + w
                if not math.isfinite(left):
                    f |= 1 << 5
                if left < 0.0:
                    f |= 1 << 6
                closure = (after - before) - total_v
                scale = ((((before + after) + h_in) + left) + total_v) * cell_tol
                if abs(closure) > scale:
                    f |= 1 << 8
                closure_h = (h_in - left) - total_v
                if abs(closure_h) > scale:
                    f |= 1 << 9
                nonfinite = not math.isfinite(total_v)
                for l in range(n_layers):
                    if not math.isfinite(intake[l, i]):
                        nonfinite = True
                if nonfinite:
                    f |= 1 << 10
            surface_out[i] = left
            total[i] = total_v
            if a:
                rows[0, i] = before
                rows[1, i] = after
                rows[2, i] = h_in
                rows[3, i] = left
            else:
                rows[0, i] = 0.0
                rows[1, i] = 0.0
                rows[2, i] = 0.0
                rows[3, i] = 0.0
            rows[4, i] = isolated_v
            rows[5, i] = attempted_v
            rows[6, i] = total_v
            if f == 0:
                rows[7, i] = no_flag
            else:
                rows[7, i] = float(f & -f)

    if not common:
        return step_core

    @jit
    def step_kernel(surface_in, w_in, active, dt, rate, max_rate, recv, sat_m, cap_tol, cell_tol, no_flag,
                    surface_out, w_out, intake, total, rows):
        # historical 16-argument signature: one call of the explicit-signature body per step (stride 1, cached (L,) vectors; dz/theta_sat/coef are never read by the common variant)
        step_core(surface_in, w_in, active, dt, rate, max_rate, 1, recv, sat_m, 1, sat_m, 1, sat_m, cap_tol, 0.0, cell_tol, no_flag,
                  surface_out, w_out, intake, total, rows)

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


def floating_launch(surface_in, water_in, mask_u8: np.ndarray, dt: float, *, rate, max_rate, k_mr, recv, edges, depth, theta_sat, k_sat, coef: float, cell_tol: float) -> StepArrays:
    """One compiled step of a FLOATING preferential operator (`floating_preferential_infiltration` calls this; nothing else should). `edges` `(L+1,)` and `depth` `(N,)` are the owner's read-only
    maps; `theta_sat` its retained raw map with layout code 0 scalar / 1 shared / 2 per cell. Fresh outputs only."""
    kernel = compiled_floating_kernel()
    n_layers, n_cells = water_in.shape
    surface_out = np.empty(n_cells, dtype=np.float64)
    water_out = np.empty((n_layers, n_cells), dtype=np.float64)
    intake = np.empty((n_layers, n_cells), dtype=np.float64)
    total = np.empty(n_cells, dtype=np.float64)
    rows = np.empty((ROW_COUNT, n_cells), dtype=np.float64)
    kernel(surface_in, water_in, mask_u8, float(dt), rate, max_rate, int(k_mr), recv, edges, 1, theta_sat, int(k_sat), depth, depth, float(coef), float(cell_tol), NO_FLAG,
           surface_out, water_out, intake, total, rows)
    return StepArrays(surface_out, water_out, intake, total, rows)


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


def _launch(kernel, maps: KernelMaps, cell_tol: float, surface_in: np.ndarray, water_in: np.ndarray,
            active_u8: np.ndarray, dt: float) -> StepArrays:
    n_layers, n_cells = water_in.shape
    surface_out = np.empty(n_cells, dtype=np.float64)
    water_out = np.empty((n_layers, n_cells), dtype=np.float64)
    intake = np.empty((n_layers, n_cells), dtype=np.float64)
    total = np.empty(n_cells, dtype=np.float64)
    rows = np.empty((ROW_COUNT, n_cells), dtype=np.float64)
    kernel(surface_in, water_in, active_u8, float(dt), maps.rate_s, maps.max_rate_m_s, maps.receiver,
           maps.saturation_m, maps.capacity_tolerance_m, float(cell_tol), NO_FLAG,
           surface_out, water_out, intake, total, rows)
    return StepArrays(surface_out, water_out, intake, total, rows)


def _readonly(array: np.ndarray) -> np.ndarray:
    """A bytes-backed read-only copy with the same shape: the Numba array type of the operator's SEALED maps."""
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    flat = np.frombuffer(a.tobytes(), dtype=np.float64)
    return flat if a.ndim == 1 else flat.reshape(a.shape)


def prepare_runner(maps: KernelMaps, cell_tol: float):
    """Compile (if needed) and warm up the kernel for the SEALED (read-only) map types; return `(runner, info)`. The
    runner receives the operator's private map handles per call, allocates fresh outputs and never writes an input."""
    kernel = compiled_kernel()
    n_layers = int(maps.receiver.shape[0])
    # Warm-up with read-only maps of the sealed kind (Numba types read-only arrays separately), writable inputs.
    warm = KernelMaps(_readonly(np.zeros(1)), _readonly(np.zeros(1)), _readonly(np.zeros((n_layers, 1))),
                      _readonly(maps.saturation_m), _readonly(maps.capacity_tolerance_m))
    _launch(kernel, warm, cell_tol, np.zeros(1), np.zeros((n_layers, 1)), np.zeros(1, np.uint8), 1.0)

    def run(op, maps_internal, surface_in, water_in, dt):
        return _launch(kernel, maps_internal, cell_tol, surface_in, water_in, op.column.active_u8, dt)

    info = {"implementation": "numba serial cell/layer loops, fastmath=False; sealed read-only maps", "versions": numba_versions(),
            "expm1": "LLVM/libm expm1 (may differ from NumPy by an ulp); declared bound rtol 2e-12 / atol 1e-14",
            "parameter_host_bytes": int(sum(int(a.nbytes) for a in maps))}
    return run, info


def spatial_launch(surface_in, water_in, mask_u8: np.ndarray, dt: float, *, rate, max_rate, k_mr, recv, dz, k_dz, theta_sat, k_sat, coef: float, cell_tol: float) -> StepArrays:
    """One compiled step of an EXPLICIT SPATIAL preferential operator (`spatial_preferential_infiltration` calls this; nothing else should). `rate`/`recv` are the operator's sealed read-only
    maps, `max_rate` a flat read-only `(N,)` map (stride 1) or a one-element array (stride 0); `dz` and `theta_sat` are the owner's retained RAW maps as flat read-only views with layout codes
    0 scalar / 1 shared / 2 per cell. Fresh output arrays only; nothing of the owner (map, mask) is copied or written."""
    kernel = compiled_spatial_kernel()
    n_layers, n_cells = water_in.shape
    surface_out = np.empty(n_cells, dtype=np.float64)
    water_out = np.empty((n_layers, n_cells), dtype=np.float64)
    intake = np.empty((n_layers, n_cells), dtype=np.float64)
    total = np.empty(n_cells, dtype=np.float64)
    rows = np.empty((ROW_COUNT, n_cells), dtype=np.float64)
    kernel(surface_in, water_in, mask_u8, float(dt), rate, max_rate, int(k_mr), recv, dz, int(k_dz), theta_sat, int(k_sat), dz, dz, float(coef), float(cell_tol), NO_FLAG,
           surface_out, water_out, intake, total, rows)
    return StepArrays(surface_out, water_out, intake, total, rows)
