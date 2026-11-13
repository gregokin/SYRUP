"""Compiled Numba CPU runner of the SYRUP preferential infiltration stage (`maple_syrup.preferential_infiltration`,
backend "numba").

The kernel evaluates EXACTLY the stages of `preferential_infiltration.reference_step`, term for term in the same
expression order, as serial cell / layer loops: no fastmath, no prange, no changed order, no clipping. Identical IEEE
operations give identical bits; `math.expm1` (LLVM/libm) may differ from NumPy's `expm1` by an ulp, which enters
only the isolated attempted amount H x (-expm1(-(k dt))). The declared agreement bound with the reference is
therefore rtol 2e-12 / atol 1e-14 on the state and fluxes (bitwise where every beta is 0 or 1, since
expm1(0) = 0 and expm1(-inf) = -1 are exact in both libraries).

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
        raise PreferentialBackendUnavailableError(
            "backend 'numba' requires Numba (optional extra maple-syrup[numba]), which is not importable; there is no "
            "fallback to the NumPy reference (select backend='numpy' explicitly)")


def _build(numba: Any):
    jit = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")

    @jit
    def step_kernel(surface_in, w_in, active, dt, rate, max_rate, recv, sat_m, cap_tol, cell_tol, no_flag,
                    surface_out, w_out, intake, total, rows):
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
                cap = max_rate[i] * dt
                if cap < attempted:  # noqa: PLR1730 - explicit ordered FP comparison, NaN-propagating; no min()
                    attempted = cap
                attempted_v = attempted
                # receiver allocation, top layer first; bounded by the room and the remaining surface water
                for l in range(n_layers):
                    alloc = attempted * recv[l, i]
                    room = sat_m[l] - w_out[l, i]
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
                    if w > sat_m[l] + cap_tol[l]:
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
