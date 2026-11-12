"""CuPy RawKernel CUDA kernel of `maple_syrup.soil_profile_texture` (backend "cuda", DENSE device-resident beds).

One thread per cell; inside the thread the segment loop (active cap, then voxels from the highest index down), the
layer loop and the sequential class folds, EXACTLY the operations of `soil_profile_texture.reference_texture_batch`
in the same order. Compiled with `routing_cuda.COMPILE_OPTIONS` (`--fmad=false --prec-div=true --prec-sqrt=true
--ftz=false`, no fast math) so no multiply-add is contracted; `fmax`/`fmin` on finite doubles are IEEE max/min.
Bitwise agreement with the NumPy reference is a hypothesis the CUDA tests check on real hardware; nothing here was
run by its author (the task forbids GPU launches by Claude), so Codex records the device results.

Host traffic per batch: the `(L + 1,)` edges are uploaded once per build (`prepare_runner`), the batch mask is
sliced from a device copy made once by the caller, the kernel writes freshly allocated device outputs, and the
caller reads the `(B,)` uint8 flags (one counted D2H per batch). No per-cell host loop, no scratch.

CuPy, a visible device and loadable kernels are required (`require_cuda`); any failure is a
`SoilProfileBackendUnavailableError`, never a NumPy fallback.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

import numpy as np

from maple_syrup.routing_cuda import COMPILE_OPTIONS
from maple_syrup.soil_profile_texture import (
    FLAG_INSUFFICIENT_DEPTH,
    FLAG_TOP_LAYER_THINNER_THAN_CAP,
    SoilProfileBackendUnavailableError,
)

__all__ = ["COMPILE_OPTIONS", "KERNEL_NAME", "THREADS", "kernel_provenance", "kernel_source", "load_kernel",
           "prepare_runner", "require_cuda"]

KERNEL_NAME = "maple_syrup_soil_profile_texture"
THREADS = 128

_SOURCE = r"""
extern "C" __global__ void maple_syrup_soil_profile_texture(
    const long long n_cells, const int n_voxels, const int n_classes, const int n_layers,
    const double* __restrict__ active, const double* __restrict__ voxels, const unsigned char* __restrict__ mask,
    const double* __restrict__ edges, const double denominator, const double bound_factor,
    double* __restrict__ mass, double* __restrict__ fraction, double* __restrict__ total,
    double* __restrict__ coverage, double* __restrict__ depth, double* __restrict__ cap,
    unsigned char* __restrict__ flags)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    for (int l = 0; l < n_layers; ++l) {
        for (int k = 0; k < n_classes; ++k) {
            mass[((long long)l * n_cells + c) * n_classes + k] = 0.0;
            fraction[((long long)l * n_cells + c) * n_classes + k] = 0.0;
        }
        total[(long long)l * n_cells + c] = 0.0;
        coverage[(long long)l * n_cells + c] = 0.0;
    }
    depth[c] = 0.0;
    cap[c] = 0.0;
    flags[c] = 0;
    if (!mask[c]) return;
    const double* a = active + c * n_classes;
    double t_active = a[0];
    for (int k = 1; k < n_classes; ++k) t_active = t_active + a[k];
    const double h_active = t_active / denominator;
    cap[c] = h_active;
    double z = 0.0;
    for (int s = 0; s <= n_voxels; ++s) {
        const int v = n_voxels - s;
        const double* m;
        double h;
        if (s == 0) {
            h = h_active;
            m = a;
        } else {
            m = voxels + (c * n_voxels + v) * n_classes;
            double t = m[0];
            for (int k = 1; k < n_classes; ++k) t = t + m[k];
            h = t / denominator;
        }
        if (h > 0.0) {
            const double top = z;
            const double bottom = z + h;
            for (int l = 0; l < n_layers; ++l) {
                const double lo = fmax(top, edges[l]);
                const double hi = fmin(bottom, edges[l + 1]);
                const double o = hi - lo;
                if (o > 0.0) {
                    const double w = o / h;
                    double* ml = mass + ((long long)l * n_cells + c) * n_classes;
                    for (int k = 0; k < n_classes; ++k) ml[k] = ml[k] + w * m[k];
                    coverage[(long long)l * n_cells + c] = coverage[(long long)l * n_cells + c] + o;
                }
            }
            z = bottom;
        }
    }
    depth[c] = z;
    // Declared bound: factor * max(represented depth, deepest edge).
    const double tol = bound_factor * fmax(z, edges[n_layers]);
    unsigned char f = 0;
    for (int l = 0; l < n_layers; ++l) {
        double* ml = mass + ((long long)l * n_cells + c) * n_classes;
        double tt = ml[0];
        for (int k = 1; k < n_classes; ++k) tt = tt + ml[k];
        total[(long long)l * n_cells + c] = tt;
        if (tt > 0.0) {
            double* fl = fraction + ((long long)l * n_cells + c) * n_classes;
            for (int k = 0; k < n_classes; ++k) fl[k] = ml[k] / tt;
        }
        const double thickness = edges[l + 1] - edges[l];
        if ((thickness - coverage[(long long)l * n_cells + c]) > tol) f = f | FLAG_SHORT;
    }
    if ((h_active - (edges[1] - edges[0])) > tol) f = f | FLAG_CAP;
    flags[c] = f;
}
"""

_FUNCTIONS: dict[int, Any] = {}
_LOCK = threading.Lock()


def kernel_source() -> str:
    return _SOURCE.replace("FLAG_SHORT", str(int(FLAG_INSUFFICIENT_DEPTH))).replace(
        "FLAG_CAP", str(int(FLAG_TOP_LAYER_THINNER_THAN_CAP)))


def kernel_provenance() -> dict[str, Any]:
    return {"kernel": KERNEL_NAME, "source_sha256": hashlib.sha256(kernel_source().encode()).hexdigest(),
            "compile_options": list(COMPILE_OPTIONS), "threads": THREADS}


def require_cuda():
    """The CuPy module with at least one visible device, through MAPLE's own probes; otherwise a named setup error."""
    from maple.core.backend import cuda_device_count, cupy_import_error, cupy_module

    cp = cupy_module()
    if cp is None:
        raise SoilProfileBackendUnavailableError(
            f"backend 'cuda' requires CuPy, which is not usable ({cupy_import_error() or 'not installed'}); there is "
            "no fallback to the NumPy reference")
    if cuda_device_count() < 1:
        raise SoilProfileBackendUnavailableError(
            "backend 'cuda' requires a visible CUDA device (device count 0); there is no fallback to the NumPy reference")
    return cp


def _current_device_id(cp) -> int:
    try:
        return int(cp.cuda.Device().id)
    except Exception as exc:  # reported as a setup failure
        raise SoilProfileBackendUnavailableError(f"CUDA device query failed ({type(exc).__name__}: {exc})") from exc


def load_kernel(cp, device_id: int):
    """Compile/load the kernel for `device_id` (once per process and device). Failure is a setup error."""
    with _LOCK:
        fn = _FUNCTIONS.get(device_id)
        if fn is not None:
            return fn
        try:
            module = cp.RawModule(code=kernel_source(), options=tuple(COMPILE_OPTIONS))
            fn = module.get_function(KERNEL_NAME)
        except Exception as exc:
            raise SoilProfileBackendUnavailableError(
                f"CUDA kernel compile/load failed on device {device_id} ({type(exc).__name__}: {exc})") from exc
        _FUNCTIONS[device_id] = fn
        return fn


def prepare_runner():
    """`runner(xp, active_rows, voxel_rows, mask_rows, edges, denominator, bound_factor)`; `xp` must be CuPy and
    the arrays device-resident C-contiguous. The edges are uploaded once (cached by value inside this runner)."""
    cp = require_cuda()
    device_id = _current_device_id(cp)
    kernel = load_kernel(cp, device_id)
    cache: dict[str, Any] = {}

    def runner(xp, active, voxels, mask, edges, denominator, bound_factor):
        if xp is not cp:
            raise SoilProfileBackendUnavailableError("backend 'cuda' computes on CuPy device arrays only")
        edges_host = np.ascontiguousarray(edges, dtype=np.float64)
        key = edges_host.tobytes()
        if cache.get("key") != key:
            cache["key"], cache["edges"] = key, cp.asarray(edges_host)  # one small H2D per distinct edge vector
        d_edges = cache["edges"]
        if not (active.flags.c_contiguous and voxels.flags.c_contiguous):
            raise SoilProfileBackendUnavailableError("cuda runner requires C-contiguous device batch views")
        B, K = (int(s) for s in active.shape)
        V = int(voxels.shape[1])
        L = int(edges_host.shape[0]) - 1
        mask_u8 = mask.astype(np.uint8)
        mass = cp.empty((L, B, K), dtype=np.float64)
        fraction = cp.empty((L, B, K), dtype=np.float64)
        total = cp.empty((L, B), dtype=np.float64)
        coverage = cp.empty((L, B), dtype=np.float64)
        depth = cp.empty((B,), dtype=np.float64)
        cap = cp.empty((B,), dtype=np.float64)
        flags = cp.empty((B,), dtype=np.uint8)
        blocks = (B + THREADS - 1) // THREADS
        try:
            kernel((max(blocks, 1),), (THREADS,), (
                np.int64(B), np.int32(V), np.int32(K), np.int32(L), active, voxels, mask_u8, d_edges,
                np.float64(denominator), np.float64(bound_factor), mass, fraction, total, coverage, depth, cap, flags))
        except Exception as exc:
            raise SoilProfileBackendUnavailableError(f"CUDA kernel launch failed ({type(exc).__name__}: {exc})") from exc
        return mass, fraction, total, coverage, depth, cap, flags

    return runner
