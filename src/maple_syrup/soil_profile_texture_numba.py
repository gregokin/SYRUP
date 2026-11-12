"""Compiled CPU kernel of `maple_syrup.soil_profile_texture` (backend "numba").

The SAME declared arithmetic as `soil_profile_texture.reference_texture_batch`, written as a serial cell loop with
the segment loop (active cap, then voxels from the highest index down), the layer loop and the sequential class
folds inside: identical IEEE operations in the identical order, `fastmath=False`, no `prange` (one build per run;
threading is a later, separately qualified optimisation), no clipping. Bitwise agreement with the NumPy reference is
TESTED by the suite (`tests/soil_profile_texture/test_backends.py`), not assumed.

Numba is optional (`maple-syrup[numba]`); absence is a `SoilProfileBackendUnavailableError`, never a NumPy fallback.
Inputs must be host C-contiguous float64 / bool arrays (the caller guarantees it); every output is freshly
allocated per call and no input is written.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from maple_syrup.soil_profile_texture import (
    FLAG_INSUFFICIENT_DEPTH,
    FLAG_TOP_LAYER_THINNER_THAN_CAP,
    SoilProfileBackendUnavailableError,
)

__all__ = ["compiled_kernel", "numba_available", "prepare_runner", "require_numba", "reset_compiled"]

_KERNEL: Any = None
_LOCK = threading.Lock()


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def require_numba() -> None:
    if not numba_available():
        raise SoilProfileBackendUnavailableError(
            "backend 'numba' requires Numba (optional extra maple-syrup[numba]), which is not importable; there is no "
            "fallback to the NumPy reference (select backend='numpy' explicitly)")


def _build(numba: Any):
    jit = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")

    @jit
    def kernel(active, voxels, mask, edges, denominator, bound_factor, mass, fraction, total, coverage, depth, cap,
               flags):
        n_cells, n_classes = active.shape
        n_voxels = voxels.shape[1]
        n_layers = edges.shape[0] - 1
        for c in range(n_cells):
            for l in range(n_layers):
                for k in range(n_classes):
                    mass[l, c, k] = 0.0
                    fraction[l, c, k] = 0.0
                total[l, c] = 0.0
                coverage[l, c] = 0.0
            depth[c] = 0.0
            cap[c] = 0.0
            flags[c] = 0
            if not mask[c]:
                continue
            t_active = active[c, 0]
            for k in range(1, n_classes):
                t_active = t_active + active[c, k]
            h_active = t_active / denominator
            cap[c] = h_active
            z = 0.0
            for s in range(n_voxels + 1):
                v = n_voxels - s
                if s == 0:
                    h = h_active
                else:
                    t = voxels[c, v, 0]
                    for k in range(1, n_classes):
                        t = t + voxels[c, v, k]
                    h = t / denominator
                if h > 0.0:
                    top = z
                    bottom = z + h
                    for l in range(n_layers):
                        lo = max(top, edges[l])
                        hi = min(bottom, edges[l + 1])
                        o = hi - lo
                        if o > 0.0:
                            w = o / h
                            if s == 0:
                                for k in range(n_classes):
                                    mass[l, c, k] = mass[l, c, k] + w * active[c, k]
                            else:
                                for k in range(n_classes):
                                    mass[l, c, k] = mass[l, c, k] + w * voxels[c, v, k]
                            coverage[l, c] = coverage[l, c] + o
                    z = bottom
            depth[c] = z
            # Declared bound: factor * max(represented depth, deepest edge).
            tol = bound_factor * max(z, edges[n_layers])
            f = 0
            for l in range(n_layers):
                tt = mass[l, c, 0]
                for k in range(1, n_classes):
                    tt = tt + mass[l, c, k]
                total[l, c] = tt
                if tt > 0.0:
                    for k in range(n_classes):
                        fraction[l, c, k] = mass[l, c, k] / tt
                thickness = edges[l + 1] - edges[l]
                if (thickness - coverage[l, c]) > tol:
                    f = f | FLAG_INSUFFICIENT_DEPTH
            if (h_active - (edges[1] - edges[0])) > tol:
                f = f | FLAG_TOP_LAYER_THINNER_THAN_CAP
            flags[c] = f

    return kernel


def compiled_kernel():
    """The compiled kernel (built once per process, under a lock)."""
    global _KERNEL
    require_numba()
    with _LOCK:
        if _KERNEL is None:
            import numba

            _KERNEL = _build(numba)
        return _KERNEL


def reset_compiled() -> None:
    global _KERNEL
    with _LOCK:
        _KERNEL = None


def prepare_runner():
    """`runner(xp, active_rows, voxel_rows, mask_rows, edges, denominator, bound_factor)` with the reference's
    return convention. `xp` must be NumPy (host arrays)."""
    kernel = compiled_kernel()

    def runner(xp, active, voxels, mask, edges, denominator, bound_factor):
        if xp is not np:
            raise SoilProfileBackendUnavailableError("backend 'numba' computes on host NumPy arrays only")
        active = np.ascontiguousarray(active, dtype=np.float64)
        voxels = np.ascontiguousarray(voxels, dtype=np.float64)
        mask = np.ascontiguousarray(mask, dtype=np.bool_)
        edges = np.ascontiguousarray(edges, dtype=np.float64)
        B, K = active.shape
        L = edges.shape[0] - 1
        mass = np.empty((L, B, K), dtype=np.float64)
        fraction = np.empty((L, B, K), dtype=np.float64)
        total = np.empty((L, B), dtype=np.float64)
        coverage = np.empty((L, B), dtype=np.float64)
        depth = np.empty((B,), dtype=np.float64)
        cap = np.empty((B,), dtype=np.float64)
        flags = np.empty((B,), dtype=np.uint8)
        kernel(active, voxels, mask, edges, float(denominator), float(bound_factor), mass, fraction, total, coverage,
               depth, cap, flags)
        return mass, fraction, total, coverage, depth, cap, flags

    return runner
