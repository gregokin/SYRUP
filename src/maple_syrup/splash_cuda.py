"""Optional CUDA ray sweep for `splash.splash_redistribute(..., implementation="cuda")` (C1 acceleration).

The SAME per-cell recurrence as the array reference `splash._array_rays`, one GPU thread per (line, class), the thread
walking its ray sequentially (causal in the ray direction) and keeping only the running flux in a register:

    src  = pickup * weight[cell]
    out  = g (flux + src)                  bin  = flux (1 - g)  -> bins_total[cell]   (+=)
    passed = out if keep[cell] else 0      tail = out - passed  -> tail_total[cell]   (+=)
    edge cell: export_edge[cell] += passed       face[...] = passed       flux = passed

Four launches (east, south, west, north slots) on the CURRENT CuPy stream, in that order, so the accumulation order
into `bins_total` / `tail_total` / `export_edge` equals the array path's, and stream order is the only barrier. Within
one launch every cell belongs to exactly one line, so no two threads touch the same element: no atomics, no fences, no
shared memory, no host synchronization, no host read, no per-source ray history (only the register flux). The kernel
works on CELL-FRAME contiguous arrays through (line, start, step) strides (negative steps for the south/west slots),
so no transposed or reversed view is ever materialized or passed. Outputs are fresh zero-initialised device arrays;
the caller's arrays (`pickup`, the grid) are only read.

Floating point: compiled with `--fmad=false` and written only with the round-to-nearest intrinsics `__dadd_rn`,
`__dsub_rn`, `__dmul_rn`, no fast math, `--ftz=false` (the options of `routing_cuda`), so no operation can be
contracted into an FMA. The operations and their order match the array path term for term, so bitwise equality with
the array path is the intent and a hypothesis to be tested, not a property asserted here; the declared comparison
bounds are in the tests. `g = exp(-a)` and `1 - g = -expm1(-a)` are computed by the caller with the namespace's
own functions and passed in, so both implementations share them.

Types: `long long` for every size, stride and offset (negative steps allowed), `double` data, `unsigned char` for
the CuPy bool masks (one byte per element). Indices: cell = start + line*line_stride + s*step; element = cell*nc + k;
face element = (face_start + line*face_line_stride + s*face_step)*nc + k.

Scope limits (nothing here was compiled or run by its author): the kernel is a straightforward desk-checked
one-thread-per-line design; for very few lines (a 1 x N strip or a single column) it is launch-latency bound and gives
no speed-up, and the rows/columns access pattern is uncoalesced across lines for east/west (adjacent threads are
adjacent classes, then rows `nx*nc` elements apart). No performance claim is made; root profiles.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

import numpy as np

from maple_syrup.splash import SplashError

__all__ = [
    "BLOCK_THREADS",
    "COMPILE_OPTIONS",
    "CUDA_QUALIFIED",
    "KERNEL_NAME",
    "SplashCudaUnavailableError",
    "cuda_rays",
    "kernel_provenance",
    "kernel_source",
]

BLOCK_THREADS = 128
KERNEL_NAME = "maple_syrup_splash_sweep"
COMPILE_OPTIONS = ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")
# `implementation="auto"` selects CUDA only for a CuPy grid and only while this is True. It is False until a run that
# qualified the kernel (parity + conservation tests on a device) sets it explicitly; this module never sets it.
CUDA_QUALIFIED = False

_SOURCE = r"""
extern "C" __global__ void maple_syrup_splash_sweep(
    const long long n_lines, const long long n_steps, const long long nc,
    const long long cell_line_stride, const long long cell_start, const long long cell_step,
    const long long face_line_stride, const long long face_start, const long long face_step,
    const double* __restrict__ pickup, const double* __restrict__ weight,
    const unsigned char* __restrict__ keep, const unsigned char* __restrict__ edge,
    const double* __restrict__ g, const double* __restrict__ omg,
    double* bins_total, double* tail_total, double* export_edge, double* face)
{
    const long long t = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (t >= n_lines * nc) return;
    const long long line = t / nc;
    const long long k = t - line * nc;
    const double gk = g[k];
    const double ok = omg[k];
    double flux = 0.0;
    long long cell = line * cell_line_stride + cell_start;
    long long fcell = line * face_line_stride + face_start;
    for (long long s = 0; s < n_steps; ++s) {
        const long long e = cell * nc + k;
        const double src = __dmul_rn(pickup[e], weight[cell]);
        const double out = __dmul_rn(gk, __dadd_rn(flux, src));
        bins_total[e] = __dadd_rn(bins_total[e], __dmul_rn(flux, ok));
        const double passed = (keep[cell] != 0) ? out : 0.0;
        tail_total[e] = __dadd_rn(tail_total[e], __dsub_rn(out, passed));
        if (edge[cell] != 0) {
            export_edge[e] = __dadd_rn(export_edge[e], passed);
        }
        face[fcell * nc + k] = passed;
        flux = passed;
        cell += cell_step;
        fcell += face_step;
    }
}
"""


class SplashCudaUnavailableError(SplashError):
    """`implementation="cuda"` was requested but CuPy, a CUDA device or the compiled kernel is unavailable. There is
    no fallback to the array path."""


_KERNEL: Any = None
_LOCK = threading.Lock()


def kernel_source() -> str:
    return _SOURCE


def _cupy():
    from maple.core.backend import cupy_module

    cp = cupy_module()
    if cp is None:
        raise SplashCudaUnavailableError("implementation 'cuda' requested but CuPy/CUDA is unavailable; no fallback")
    return cp


def _kernel(cp):
    """Lazily created process-wide RawKernel; first access of `.kernel` compiles/loads for the current device, so
    a compile failure surfaces here, before any output exists."""
    global _KERNEL
    if _KERNEL is None:
        with _LOCK:
            if _KERNEL is None:
                try:
                    _KERNEL = cp.RawKernel(_SOURCE, KERNEL_NAME, options=COMPILE_OPTIONS, backend="nvrtc")
                except Exception as exc:
                    raise SplashCudaUnavailableError(
                        f"could not create the CUDA splash kernel ({type(exc).__name__}: {exc}); no fallback") from exc
    try:
        _KERNEL.kernel  # noqa: B018 - CuPy compiles/loads on first access
    except Exception as exc:
        raise SplashCudaUnavailableError(
            f"CUDA splash kernel failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
    return _KERNEL


def kernel_provenance() -> dict[str, Any]:
    return {"kernel": KERNEL_NAME, "source_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
            "compile_options": list(COMPILE_OPTIONS), "block_threads": BLOCK_THREADS,
            "intrinsics": ["__dadd_rn", "__dsub_rn", "__dmul_rn"], "fastmath": False,
            "launch_structure": "4 launches (east, south, west, north), current stream, one thread per (line, class)",
            "cuda_qualified": CUDA_QUALIFIED}


def cuda_rays(grid: Any, pickup_kg: Any, g: Any, omg: Any, nc: int):
    """The four ray sweeps on the device. Returns `(bins_total, tail_total, export_edge, x_east, x_west, y_north,
    y_south)`, the same fresh arrays `splash._array_rays` returns. `pickup_kg` `(ny, nx, nc)` float64, `g` / `omg`
    `(nc,)` float64 (device); all validated by the caller. Asynchronous on the current stream."""
    cp = _cupy()
    if grid.xp is not cp:
        raise SplashCudaUnavailableError("the grid is not in the CuPy namespace of this process")
    kernel = _kernel(cp)
    ny, nx = grid.shape
    pickup = cp.ascontiguousarray(pickup_kg)
    gk = cp.ascontiguousarray(g)
    ok = cp.ascontiguousarray(omg)
    if pickup.dtype != np.float64 or gk.dtype != np.float64 or ok.dtype != np.float64 or gk.shape != (nc,) \
            or ok.shape != (nc,) or pickup.shape != (ny, nx, nc):
        raise SplashError("cuda sweep inputs must be float64 with shapes (ny, nx, nc), (nc,), (nc,)")
    bins_total = cp.zeros((ny, nx, nc), dtype=np.float64)
    tail_total = cp.zeros((ny, nx, nc), dtype=np.float64)
    export_edge = cp.zeros((ny, nx, nc), dtype=np.float64)
    x_east = cp.zeros((ny, nx + 1, nc), dtype=np.float64)
    x_west = cp.zeros((ny, nx + 1, nc), dtype=np.float64)
    y_north = cp.zeros((ny + 1, nx, nc), dtype=np.float64)
    y_south = cp.zeros((ny + 1, nx, nc), dtype=np.float64)
    last_row = (ny - 1) * nx
    # slot: (n_lines, n_steps, cell line stride, cell start, cell step, face line stride, face start, face step, face array)
    # cell index = row * nx + col; x face index = row * (nx + 1) + face_col; y face index = face_row * nx + col.
    table = (
        (ny, nx, nx, 0, 1, nx + 1, 1, 1, x_east),                 # east: cells (r, s); east face of col s = col face s + 1
        (nx, ny, 1, last_row, -nx, 1, last_row, -nx, y_south),    # south: cells (ny-1-s, c); south face of row r = face r
        (ny, nx, nx, nx - 1, -1, nx + 1, nx - 1, -1, x_west),     # west: cells (r, nx-1-s); west face of col c = face c
        (nx, ny, 1, 0, nx, 1, nx, nx, y_north),                   # north: cells (s, c); north face of row r = face r + 1
    )
    i64 = np.int64
    try:
        for slot, (n_lines, n_steps, cls, cst, cstep, fls, fst, fstep, face) in enumerate(table):
            threads = n_lines * nc
            blocks = (threads + BLOCK_THREADS - 1) // BLOCK_THREADS
            kernel((blocks,), (BLOCK_THREADS,), (
                i64(n_lines), i64(n_steps), i64(nc), i64(cls), i64(cst), i64(cstep), i64(fls), i64(fst), i64(fstep),
                pickup, grid.weights[slot], grid.ray_continues[slot], grid.edge_cell[slot], gk, ok,
                bins_total, tail_total, export_edge, face))
    except SplashError:
        raise
    except Exception as exc:
        raise SplashError(f"CUDA splash sweep launch failed ({type(exc).__name__}: {exc}); no fallback") from exc
    return bins_total, tail_total, export_edge, x_east, x_west, y_north, y_south
