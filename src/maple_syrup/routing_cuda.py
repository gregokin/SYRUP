"""Optional CUDA ordered sweep for `route_step(..., implementation="cuda")` (Phase 4R, step 1).

The SAME discrete method-5 sweep as `routing_numba._sweep_batched`, per cell: donor sum
`((((0 + d0) + d1) + d2) + d3)` in DONOR_SLOTS order (absent donors add exactly 0.0), raw `rhs = base + qin * c`,
the proven `[0, rhs]` bracket, `iterations` (1..200) bisections with the unchanged multiply sequence and the strict
`t < rhs` test, `q = (sqrt(lo) * lo) * k`. Cells with `not (rhs > 0.0)` skip the iterations (exact, see
docs/phase7j/design.md) and keep `lo = +0.0`; the raw rhs is stored so the shared refusals see the same values.

Execution (`run_sweep(..., mode)`, see SWEEP_MODES): "level" = one thread per cell, ONE kernel launch per dependency
level, in level order, on the CURRENT CuPy stream; in-stream ordering is the only barrier between levels (donors always
lie in earlier levels, so cells of one level are independent). "block" = ONE launch of ONE block of BLOCK_THREADS
threads that walks every level with an unconditional `__syncthreads()` between levels (strided cell loop; every thread
reaches every barrier; no inter-block synchronization). "auto" (default) = "block" when the widest level has at most
NARROW_MAX_WIDTH (= BLOCK_THREADS) cells, else "level". Per-cell arithmetic is identical in both kernels. No atomics,
fences, host synchronization, host reads, per-level allocation or scratch. Each sweep allocates four fresh outputs
(`qin_new`, `q_new`, `flow`, `rhs`, level order, length n_active).

Floating point: compiled with `--fmad=false` and written with the round-to-nearest intrinsics `__dadd_rn`,
`__dmul_rn`, `__dsqrt_rn`, so no operation can be contracted into an FMA. No fast math. `--prec-div` and
`--prec-sqrt` are passed only to document the intent; they apply to single precision and are NOT what makes the
double arithmetic IEEE. The sweep contains no division. Bitwise equality with the CPU sweeps is a hypothesis to be
tested, not a property this module asserts.

Ownership / freshness contract (`prepare_cuda_routing`): the graph's CuPy arrays cannot be frozen. Preparation
downloads the static graph data ONCE through `maple.core.backend.to_host`, validates memory and dependency safety
(the kernel does no bounds checking), uploads fresh contiguous device copies, and synchronizes the current stream
once. The context is cached by graph IDENTITY in a `WeakKeyDictionary`; the value holds no reference to the graph.
The caller must NOT mutate the graph's host or runtime arrays after preparation: the sweep uses the owned copies, so
a mutation would make the graph and the wrapper's other (non-owned) reads inconsistent. Only metadata (device,
pointers, shapes, dtypes) is re-checked on a cache hit, which performs no transfer and no synchronization.
`release_cuda_routing` drops a context; the cache is bounded by graph lifetime.

CuPy is optional and imported lazily; importing this module does not import CuPy. No fallback: a missing CuPy,
device, compile or load failure raises `CudaUnavailableError`; a launch failure raises `RoutingError`. Nothing here
was run by its author (file-only tools); Codex records results.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup.routing import (
    DONOR_SLOTS,
    EXPORT,
    PIT_STORAGE,
    RoutingError,
    RoutingGraph,
)

__all__ = [
    "BLOCK_THREADS",
    "COMPILE_OPTIONS",
    "DEFAULT_SWEEP_MODE",
    "NARROW_MAX_WIDTH",
    "SWEEP_MODES",
    "CudaRoutingContext",
    "CudaUnavailableError",
    "bisect_device_source",
    "block_kernel_source",
    "kernel_provenance",
    "kernel_source",
    "prepare_cuda_routing",
    "release_cuda_routing",
    "require_inputs",
    "resolve_sweep_mode",
    "run_sweep",
]

BLOCK_THREADS = 128
KERNEL_NAME = "maple_syrup_route_level"
BLOCK_KERNEL_NAME = "maple_syrup_route_block"
COMPILE_OPTIONS = ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")
_MAX_ITERATIONS = 200

# Sweep launch structure (Phase 4R task phase4r_gpu_storm). The per-cell arithmetic is identical in every mode.
#   "level": one launch per dependency level (the accepted Phase 4R kernel, any width).
#   "block": ONE launch, one block of BLOCK_THREADS threads walks every level with an unconditional __syncthreads()
#            between levels (strided cell loop inside a level). Dependency-safe without inter-block communication.
#   "auto" : "block" when the widest level has at most NARROW_MAX_WIDTH cells, else "level". The threshold is the
#            block size, not a tuned value; Codex measured the one-block form faster only on the narrow network.
SWEEP_MODES = ("auto", "level", "block")
NARROW_MAX_WIDTH = BLOCK_THREADS
# Module default used when `run_sweep(..., mode=None)`; tests may set it to "level" to exercise the old comparator.
DEFAULT_SWEEP_MODE = "auto"

_KERNEL_SOURCE = r"""
extern "C" __global__ void maple_syrup_route_level(
    const long long b0, const long long m, const long long n_active,
    const double* __restrict__ k_lo, const long long* __restrict__ donor_position,
    const unsigned char* __restrict__ donor_mask, const double* __restrict__ base_lo,
    const double c, const int iterations,
    double* qin_new_lo, double* q_new_lo, double* flow_lo, double* rhs_lo)
{
    const long long j = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (j >= m) return;
    const long long p = b0 + j;
    double qin = 0.0;
    for (int s = 0; s < 4; ++s) {
        const long long i = (long long)s * n_active + p;
        if (donor_mask[i] != 0) {
            qin = __dadd_rn(qin, q_new_lo[donor_position[i]]);
        } else {
            qin = __dadd_rn(qin, 0.0);
        }
    }
    const double rhs = __dadd_rn(base_lo[p], __dmul_rn(qin, c));
    const double k = k_lo[p];
    double lo = 0.0;
    if (rhs > 0.0) {
        double w = rhs;
        for (int it = 0; it < iterations; ++it) {
            w = __dmul_rn(w, 0.5);
            const double mid = __dadd_rn(lo, w);
            double t = __dsqrt_rn(mid);
            t = __dmul_rn(t, mid);
            t = __dmul_rn(t, k);
            t = __dmul_rn(t, c);
            t = __dadd_rn(t, mid);
            if (t < rhs) lo = mid;
        }
    }
    double q = __dsqrt_rn(lo);
    q = __dmul_rn(q, lo);
    q = __dmul_rn(q, k);
    qin_new_lo[p] = qin;
    q_new_lo[p] = q;
    flow_lo[p] = lo;
    rhs_lo[p] = rhs;
}
"""


# The root search shared (text for text) by the one-block sweep here and by the fused hydrology kernels.
_BISECT_DEVICE_SOURCE = r"""
__device__ __forceinline__ double maple_syrup_bisect(const double rhs, const double k, const double c,
                                                     const int iterations)
{
    double lo = 0.0;
    if (rhs > 0.0) {
        double w = rhs;
        for (int it = 0; it < iterations; ++it) {
            w = __dmul_rn(w, 0.5);
            const double mid = __dadd_rn(lo, w);
            double t = __dsqrt_rn(mid);
            t = __dmul_rn(t, mid);
            t = __dmul_rn(t, k);
            t = __dmul_rn(t, c);
            t = __dadd_rn(t, mid);
            if (t < rhs) lo = mid;
        }
    }
    return lo;
}
"""

# Whole-sweep single-block kernel. Every thread of the block reaches every barrier (the level loop bounds are read
# from device memory by all threads and the barrier is outside the strided cell loop); threads without a cell at a
# level only skip work. Plain (non-restrict) pointers for the arrays written inside the loop.
_BLOCK_SOURCE = _BISECT_DEVICE_SOURCE + r"""
extern "C" __global__ void maple_syrup_route_block(
    const long long* level_bounds, const int n_levels, const long long n_active,
    const double* k_lo, const long long* donor_position, const unsigned char* donor_mask, const double* base_lo,
    const double c, const int iterations,
    double* qin_new_lo, double* q_new_lo, double* flow_lo, double* rhs_lo)
{
    for (int lev = 0; lev < n_levels; ++lev) {
        const long long b0 = level_bounds[lev];
        const long long m = level_bounds[lev + 1] - b0;
        for (long long j = (long long)threadIdx.x; j < m; j += (long long)blockDim.x) {
            const long long p = b0 + j;
            double qin = 0.0;
            for (int s = 0; s < 4; ++s) {
                const long long i = (long long)s * n_active + p;
                if (donor_mask[i] != 0) {
                    qin = __dadd_rn(qin, q_new_lo[donor_position[i]]);
                } else {
                    qin = __dadd_rn(qin, 0.0);
                }
            }
            const double rhs = __dadd_rn(base_lo[p], __dmul_rn(qin, c));
            const double k = k_lo[p];
            const double lo = maple_syrup_bisect(rhs, k, c, iterations);
            double q = __dsqrt_rn(lo);
            q = __dmul_rn(q, lo);
            q = __dmul_rn(q, k);
            qin_new_lo[p] = qin;
            q_new_lo[p] = q;
            flow_lo[p] = lo;
            rhs_lo[p] = rhs;
        }
        __syncthreads();
    }
}
"""


class CudaUnavailableError(RoutingError):
    """`implementation="cuda"` was requested but CuPy, a CUDA device, or the compiled kernel is unavailable. There
    is no fallback to the array or numba paths."""


def kernel_source() -> str:
    """The per-level kernel source (unchanged Phase 4R text; contains no barrier)."""
    return _KERNEL_SOURCE


def block_kernel_source() -> str:
    """The whole-sweep single-block kernel source (contains `__syncthreads`)."""
    return _BLOCK_SOURCE


def bisect_device_source() -> str:
    """The shared device root search `maple_syrup_bisect(rhs, k, c, iterations)` (RN FP64 intrinsics only)."""
    return _BISECT_DEVICE_SOURCE


def resolve_sweep_mode(max_level_width: int, mode: str | None = None) -> str:
    """Pure host selection: "level" or "block". `mode=None` uses `DEFAULT_SWEEP_MODE`; "auto" picks "block" only when
    the widest level has at most NARROW_MAX_WIDTH cells. An explicit "level"/"block" is always honoured."""
    chosen = DEFAULT_SWEEP_MODE if mode is None else mode
    if chosen not in SWEEP_MODES:
        raise RoutingError(f"sweep mode must be one of {SWEEP_MODES}, got {chosen!r}")
    if chosen == "auto":
        return "block" if int(max_level_width) <= NARROW_MAX_WIDTH else "level"
    return chosen


# --- lazy optional CuPy / kernel ------------------------------------------------------------------------------------
_KERNEL: Any = None
_BLOCK_KERNEL: Any = None
_KERNEL_LOCK = threading.Lock()


def _cupy():
    try:
        import cupy
    except Exception as exc:  # any import-time failure means "unavailable"
        raise CudaUnavailableError(
            f"implementation 'cuda' requested but CuPy is not importable ({type(exc).__name__}: {exc}); "
            "there is no fallback to the array or numba implementation") from exc
    return cupy


def _current_device_id(cp) -> int:
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise CudaUnavailableError("implementation 'cuda' requested but no CUDA device is visible; no fallback")
        return int(cp.cuda.Device().id)
    except CudaUnavailableError:
        raise
    except Exception as exc:
        raise CudaUnavailableError(f"CUDA runtime/device query failed ({type(exc).__name__}: {exc}); no fallback") from exc


def _get_kernel():
    """The (lazily created, process-wide) RawKernel object. Creating it does not compile; `_load_kernel` does."""
    global _KERNEL
    if _KERNEL is None:
        cp = _cupy()
        with _KERNEL_LOCK:
            if _KERNEL is None:
                try:
                    _KERNEL = cp.RawKernel(_KERNEL_SOURCE, KERNEL_NAME, options=COMPILE_OPTIONS, backend="nvrtc")
                except Exception as exc:
                    raise CudaUnavailableError(
                        f"could not create the CUDA routing kernel ({type(exc).__name__}: {exc}); no fallback") from exc
    return _KERNEL


def _get_block_kernel():
    """The (lazily created, process-wide) single-block RawKernel; creating it does not compile."""
    global _BLOCK_KERNEL
    if _BLOCK_KERNEL is None:
        cp = _cupy()
        with _KERNEL_LOCK:
            if _BLOCK_KERNEL is None:
                try:
                    _BLOCK_KERNEL = cp.RawKernel(_BLOCK_SOURCE, BLOCK_KERNEL_NAME, options=COMPILE_OPTIONS,
                                                 backend="nvrtc")
                except Exception as exc:
                    raise CudaUnavailableError(
                        f"could not create the CUDA one-block routing kernel ({type(exc).__name__}: {exc}); "
                        "no fallback") from exc
    return _BLOCK_KERNEL


def _load_kernel(kernel) -> None:
    """Force NVRTC compilation and module load for the CURRENT device now (so failures surface at preparation, not
    mid-sweep). Any failure is `CudaUnavailableError`."""
    try:
        kernel.kernel  # noqa: B018 - CuPy compiles/loads on first access of this property
    except Exception as exc:
        raise CudaUnavailableError(
            f"CUDA routing kernel failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc


def kernel_provenance() -> dict[str, Any]:
    """Source hash, compile options and actual toolchain/device. Optional import/runtime query failures are
    reported as unavailable pieces with diagnostic details; unexpected programming errors are not hidden."""
    info: dict[str, Any] = {
        "kernel": KERNEL_NAME,
        "source_sha256": hashlib.sha256(_KERNEL_SOURCE.encode()).hexdigest(),
        "block_kernel": BLOCK_KERNEL_NAME,
        "block_source_sha256": hashlib.sha256(_BLOCK_SOURCE.encode()).hexdigest(),
        "compile_options": list(COMPILE_OPTIONS),
        "block_threads": BLOCK_THREADS,
        "intrinsics": ["__dadd_rn", "__dmul_rn", "__dsqrt_rn"],
        "fastmath": False,
        "sweep_modes": list(SWEEP_MODES),
        "default_sweep_mode": DEFAULT_SWEEP_MODE,
        "narrow_max_width": NARROW_MAX_WIDTH,
        "launch_structure": "level: one launch per dependency level, current stream, one thread per cell; "
                            "block: one launch, one block of block_threads threads, unconditional __syncthreads "
                            "between levels (auto selects block when the widest level <= narrow_max_width)",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy": None, "cuda_runtime": None, "cuda_driver": None, "nvrtc": None, "device": None,
    }
    try:
        import cupy as cp

        info["cupy"] = cp.__version__
        info["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
        info["cuda_driver"] = int(cp.cuda.runtime.driverGetVersion())
        try:
            info["nvrtc"] = list(cp.cuda.nvrtc.getVersion())
        except (ImportError, OSError, RuntimeError, AttributeError, ValueError) as exc:
            info["nvrtc"] = None
            info["nvrtc_error"] = f"{type(exc).__name__}: {exc}"
        if cp.cuda.runtime.getDeviceCount() > 0:
            dev = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(dev.id)
            name = props["name"]
            info["device"] = {"visible_id": int(dev.id),
                              "name": name.decode() if isinstance(name, bytes) else str(name),
                              "compute_capability": str(dev.compute_capability),
                              "total_memory_bytes": int(props["totalGlobalMem"])}
    except (ImportError, OSError, RuntimeError, AttributeError, ValueError) as exc:  # provenance reports optional-runtime failures without raising
        info["provenance_incomplete"] = True
        info["provenance_error"] = f"{type(exc).__name__}: {exc}"
    return info


# --- static context -------------------------------------------------------------------------------------------------
def _fingerprint(array: Any) -> tuple:
    return (int(array.data.ptr), tuple(array.shape), str(array.dtype), bool(array.flags.c_contiguous),
            int(array.device.id))


_GRAPH_RUNTIME_FIELDS = ("conveyance", "active_flat", "outlet_flat", "level_order", "conveyance_lo",
                         "donor_position", "donor_mask")


@dataclass(frozen=True, eq=False)
class CudaRoutingContext:
    """Owned, validated static device data of one graph. Holds NO reference to the graph. Do not mutate the arrays."""

    device_id: int
    n_cells: int
    n_active: int
    level_bounds: tuple[int, ...]
    max_level_width: int
    conveyance_lo: Any  # (n_active,) float64
    donor_position: Any  # (4, n_active) int64, slot-major
    donor_mask: Any  # (4, n_active) bool
    level_bounds_device: Any  # (n_levels + 1,) int64, read by the one-block sweep
    graph_input_sha256: str
    static_bytes: int
    preparation_s: float
    host_to_device_bytes: int
    device_to_host_bytes: int
    graph_fingerprints: tuple
    owned_fingerprints: tuple

    def summary(self) -> dict[str, Any]:
        return {"device_id": self.device_id, "n_cells": self.n_cells, "n_active": self.n_active,
                "n_levels": len(self.level_bounds) - 1, "max_level_width": self.max_level_width,
                "static_bytes": self.static_bytes, "preparation_s": self.preparation_s,
                "host_to_device_bytes": self.host_to_device_bytes,
                "device_to_host_bytes": self.device_to_host_bytes,
                "graph_input_sha256": self.graph_input_sha256}


_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_CACHE_LOCK = threading.Lock()


def _owned_fingerprints(ctx_arrays) -> tuple:
    return tuple(_fingerprint(a) for a in ctx_arrays)


def _check_device_array(cp, array: Any, name: str, dtype, shape: tuple[int, ...], device_id: int) -> None:
    if type(array) is not cp.ndarray:
        raise RoutingError(f"{name} must be an exact cupy.ndarray (no transfer or conversion is made), "
                           f"got {type(array).__name__}")
    if array.dtype != dtype:
        raise RoutingError(f"{name} must have dtype {np.dtype(dtype)}, got {array.dtype}")
    if tuple(array.shape) != shape:
        raise RoutingError(f"{name} shape {tuple(array.shape)} != {shape}")
    if not array.flags.c_contiguous:
        raise RoutingError(f"{name} must be C-contiguous")
    if int(array.device.id) != device_id:
        raise RoutingError(f"{name} lives on CUDA device {int(array.device.id)} but the current device is "
                           f"{device_id}; no peer access or migration is performed")


def _host(array: Any, dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
    if type(array) is not np.ndarray or array.dtype != dtype or tuple(array.shape) != shape:
        raise RoutingError(f"{name} must be a host numpy.ndarray {np.dtype(dtype)}{shape}")
    return array


def _validate_host_graph(graph, active, outlet, k, order, k_lo, donor_position, donor_mask) -> tuple[int, ...]:
    """Dependency/memory safety of the downloaded static data, mirroring `hydrology_numba.prepare_hydrology`.
    Returns the level bounds."""
    shape = tuple(graph.shape)
    n = active.size
    n_active = int(np.count_nonzero(active))
    if n_active < 1:
        raise RoutingError("the graph has no active cell; refusing to launch")
    host_active = _host(graph.active, np.bool_, shape, "graph.active")
    level = _host(graph.level, np.int32, shape, "graph.level").reshape(-1)
    receiver = _host(graph.receiver, np.int64, shape, "graph.receiver").reshape(-1)
    host_order = _host(graph.level_order_host, np.int64, (n_active,), "graph.level_order_host")
    if not np.array_equal(host_active.reshape(-1), active):
        raise RoutingError("graph.active and graph.active_flat disagree")
    if not (np.all(np.isfinite(k)) and np.all(k >= 0.0)):
        raise RoutingError("graph.conveyance must be finite and >= 0 everywhere")
    if order.min() < 0 or order.max() >= n or np.unique(order).size != n_active or not np.all(active[order]):
        raise RoutingError("graph.level_order must be a permutation of the active flat cell indices")
    if not np.array_equal(order, host_order):
        raise RoutingError("graph.level_order differs from graph.level_order_host")
    if not np.array_equal(k_lo, k[order]):
        raise RoutingError("graph.conveyance_lo differs from graph.conveyance[level_order]")
    raw = graph.level_bounds
    if not isinstance(raw, tuple) or len(raw) < 2 or not all(
            isinstance(b, (int, np.integer)) and not isinstance(b, bool) for b in raw):
        raise RoutingError("graph.level_bounds must be a tuple of at least two ints")
    bounds = np.array(raw, dtype=np.int64)
    if bounds[0] != 0 or bounds[-1] != n_active or np.any(np.diff(bounds) < 0):
        raise RoutingError("graph.level_bounds must be non-decreasing from 0 to n_active")
    level_of_p = np.repeat(np.arange(bounds.size - 1, dtype=np.int64), np.diff(bounds))
    if not np.array_equal(level[order].astype(np.int64), level_of_p):
        raise RoutingError("graph.level_order is not consistent with graph.level and level_bounds")
    if donor_position.min() < 0 or donor_position.max() >= n_active:
        raise RoutingError("graph.donor_position holds an index outside [0, n_active)")
    level_start = bounds[level_of_p]
    if np.any(donor_mask & (donor_position >= level_start[None, :])):
        raise RoutingError("a donor is not in an earlier dependency level than its receiver")
    receiving_cell = np.broadcast_to(order[None, :], donor_position.shape)[donor_mask]
    donor_cell = order[donor_position[donor_mask]]
    if np.any(receiver[donor_cell] != receiving_cell):
        raise RoutingError("graph.donor_position names a cell that does not drain into the receiver")
    # Slot geometry (stricter than `prepare_hydrology`): slot s of a receiver must hold the neighbour at
    # receiver + DONOR_SLOTS[s], because the kernel sums donors in slot order and that order is the legacy sum order.
    ny, nx = shape
    slot_of_entry = np.broadcast_to(np.arange(4)[:, None], donor_position.shape)[donor_mask]
    d_row = np.array([off[0] for off in DONOR_SLOTS], dtype=np.int64)[slot_of_entry]
    d_col = np.array([off[1] for off in DONOR_SLOTS], dtype=np.int64)[slot_of_entry]
    want_row = receiving_cell // nx + d_row
    want_col = receiving_cell % nx + d_col
    if np.any((want_row < 0) | (want_row >= ny) | (want_col < 0) | (want_col >= nx)) \
            or np.any(donor_cell != want_row * nx + want_col):
        raise RoutingError("graph.donor_position slot does not hold the neighbour at receiver + DONOR_SLOTS[slot]; "
                           "the donor summation order would differ from the legacy order")
    n_internal = int(np.count_nonzero(receiver[active] >= 0))
    if donor_cell.size != n_internal or np.unique(donor_cell).size != n_internal:
        raise RoutingError("graph donors are not exactly the active cells with an internal receiver")
    active_receiver = receiver[active]
    if np.any((active_receiver < EXPORT) & (active_receiver != PIT_STORAGE)):
        raise RoutingError("an active cell has no receiver (neither a cell, EXPORT nor PIT_STORAGE)")
    pit_cells = active & (receiver == PIT_STORAGE)
    declared_pits = graph.pit_storage
    if np.any(pit_cells) or (declared_pits is not None and np.any(declared_pits)):
        if np.any(pit_cells) and (np.any(k[pit_cells] != 0.0) or np.any(outlet[pit_cells])):
            raise RoutingError("a PIT_STORAGE cell must have zero conveyance and cannot be an outlet")
        mask = _host(declared_pits, np.bool_, shape, "graph.pit_storage").reshape(-1) if declared_pits is not None else None
        aspect = _host(graph.aspect, np.int8, shape, "graph.aspect").reshape(-1)
        slope = _host(graph.slope, np.float64, shape, "graph.slope").reshape(-1)
        if (mask is None or not np.array_equal(mask, pit_cells) or graph.policy == "strict"
                or np.any(aspect[pit_cells] != 0) or np.any(slope[pit_cells] != 0.0)):
            raise RoutingError("PIT_STORAGE receivers are inconsistent with graph.pit_storage / policy / aspect 0 / slope 0")
    if not np.array_equal(outlet, active & (receiver == EXPORT)):
        raise RoutingError("graph.outlet is not exactly the exporting active cells")
    return tuple(int(b) for b in raw)


def _graph_arrays(cp, graph, device_id: int) -> dict[str, Any]:
    if not isinstance(graph, RoutingGraph):
        raise RoutingError("graph must be a RoutingGraph (use build_routing_graph)")
    if graph.xp is not cp:
        raise RoutingError(f"implementation 'cuda' needs a graph built with xp=cupy; the graph lives in "
                           f"{getattr(graph.xp, '__name__', graph.xp)!r} and no transfer is performed")
    raw_shape = graph.shape
    if not isinstance(raw_shape, (tuple, list)) or len(raw_shape) != 2 or not all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v >= 1 for v in raw_shape):
        raise RoutingError(f"graph shape must be a tuple of two positive ints (ny, nx), got {raw_shape!r}")
    shape = (int(raw_shape[0]), int(raw_shape[1]))
    n = shape[0] * shape[1]
    order_host = graph.level_order_host
    if type(order_host) is not np.ndarray or order_host.ndim != 1:
        raise RoutingError("graph.level_order_host must be a 1-D host numpy.ndarray")
    n_active = int(order_host.size)
    if n_active < 1:
        raise RoutingError("the graph has no active cell; refusing to launch")
    spec = {"conveyance": (np.float64, (n,)), "active_flat": (np.bool_, (n,)), "outlet_flat": (np.bool_, (n,)),
            "level_order": (np.int64, (n_active,)), "conveyance_lo": (np.float64, (n_active,)),
            "donor_position": (np.int64, (4, n_active)), "donor_mask": (np.bool_, (4, n_active))}
    arrays = {}
    for name, (dtype, shp) in spec.items():
        array = getattr(graph, name)
        _check_device_array(cp, array, f"graph.{name}", dtype, shp, device_id)
        arrays[name] = array
    return arrays


def prepare_cuda_routing(graph: RoutingGraph) -> CudaRoutingContext:
    """Validate the graph and build (or return the cached) owned static device context for the CURRENT device. The
    First call downloads each static array once through counted MAPLE transfers, validates, uploads three owned
    arrays and synchronizes the current stream once; a cache hit does none of these (metadata checks only)."""
    cp = _cupy()
    device_id = _current_device_id(cp)
    arrays = _graph_arrays(cp, graph, device_id)  # structure + current device of every runtime array, before work
    with _CACHE_LOCK:
        ctx = _CACHE.get(graph)
    if ctx is not None:
        if ctx.device_id != device_id:
            raise RoutingError(f"the graph was prepared on CUDA device {ctx.device_id} but the current device is "
                               f"{device_id}; release_cuda_routing(graph) and prepare again on purpose")
        if ctx.graph_fingerprints != tuple(_fingerprint(arrays[nm]) for nm in _GRAPH_RUNTIME_FIELDS):
            raise RoutingError("the graph's runtime arrays changed after preparation (graphs must not be mutated "
                               "or have fields replaced); release_cuda_routing(graph) and prepare again")
        if ctx.owned_fingerprints != _owned_fingerprints((ctx.conveyance_lo, ctx.donor_position, ctx.donor_mask,
                                                          ctx.level_bounds_device)):
            raise RoutingError("the owned static device arrays no longer match the prepared metadata")
        return ctx

    from maple.core.backend import read_transfer_counters, to_device, to_host

    t0 = time.perf_counter()
    before = read_transfer_counters()
    kernel = _get_kernel()
    _load_kernel(kernel)  # compile/load for this device now
    _load_kernel(_get_block_kernel())
    host = {nm: to_host(arrays[nm]) for nm in _GRAPH_RUNTIME_FIELDS}
    bounds = _validate_host_graph(graph, host["active_flat"], host["outlet_flat"], host["conveyance"],
                                  host["level_order"], host["conveyance_lo"], host["donor_position"],
                                  host["donor_mask"])
    owned_k = to_device(np.ascontiguousarray(host["conveyance_lo"]), cp)
    owned_pos = to_device(np.ascontiguousarray(host["donor_position"]), cp)
    owned_mask = to_device(np.ascontiguousarray(host["donor_mask"]), cp)
    owned_bounds = to_device(np.array(bounds, dtype=np.int64), cp)
    cp.cuda.get_current_stream().synchronize()  # uploads complete before any other stream may use them
    delta = read_transfer_counters().delta(before)
    widths = [b - a for a, b in itertools.pairwise(bounds)]
    ctx = CudaRoutingContext(
        device_id=device_id, n_cells=int(host["active_flat"].size), n_active=int(graph.n_active),
        level_bounds=bounds, max_level_width=int(max(widths)), conveyance_lo=owned_k, donor_position=owned_pos,
        donor_mask=owned_mask, level_bounds_device=owned_bounds, graph_input_sha256=str(graph.input_sha256),
        static_bytes=int(owned_k.nbytes + owned_pos.nbytes + owned_mask.nbytes),
        preparation_s=time.perf_counter() - t0, host_to_device_bytes=int(delta.host_to_device_bytes),
        device_to_host_bytes=int(delta.device_to_host_bytes),
        graph_fingerprints=tuple(_fingerprint(arrays[nm]) for nm in _GRAPH_RUNTIME_FIELDS),
        owned_fingerprints=_owned_fingerprints((owned_k, owned_pos, owned_mask, owned_bounds)),
    )
    with _CACHE_LOCK:
        _CACHE[graph] = ctx
    return ctx


def release_cuda_routing(graph: RoutingGraph) -> bool:
    """Drop the cached context of `graph` (frees its owned device arrays once unreferenced). True if one existed."""
    with _CACHE_LOCK:
        return _CACHE.pop(graph, None) is not None


def require_inputs(graph: RoutingGraph, dynamic: dict[str, Any]) -> CudaRoutingContext:
    """Called by `route_step` before any shared arithmetic: CuPy/device availability, every runtime graph array and
    every provided dynamic array (exact cupy.ndarray, on the current device), and the owned static context."""
    cp = _cupy()
    device_id = _current_device_id(cp)
    for name, array in dynamic.items():
        if type(array) is not cp.ndarray:
            raise RoutingError(f"{name} must be an exact cupy.ndarray for implementation 'cuda', "
                               f"got {type(array).__name__}")
        if int(array.device.id) != device_id:
            raise RoutingError(f"{name} lives on CUDA device {int(array.device.id)} but the current device is "
                               f"{device_id}; no peer access or migration is performed")
    return prepare_cuda_routing(graph)


# --- sweep ----------------------------------------------------------------------------------------------------------
def _check_sweep_options(c: Any, iterations: Any) -> tuple[float, int]:
    if isinstance(c, bool) or not isinstance(c, (int, float, np.integer, np.floating)):
        raise RoutingError(f"c must be a real number, got {type(c).__name__}")
    c = float(c)
    if not (math.isfinite(c) and c > 0.0):
        raise RoutingError(f"c must be finite and > 0, got {c!r}")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) \
            or not (1 <= int(iterations) <= _MAX_ITERATIONS):
        raise RoutingError(f"bisection_iterations must be an int in [1, {_MAX_ITERATIONS}], got {iterations!r}")
    return c, int(iterations)


def run_sweep(graph: RoutingGraph, base_lo: Any, c: float, iterations: int, mode: str | None = None):
    """Level-ordered sweep on the device. `base_lo` is the level-ordered `h_start + c (Qin_old - q_old)` as an exact
    contiguous float64 `(n_active,)` cupy array on the current device (its VALUES are not inspected: negative, NaN
    and Inf pass through for the shared refusals). Returns FOUR FRESH arrays `(qin_new_lo, q_new_lo, flow_lo,
    rhs_lo)` in level order, like `routing_numba.run_sweep`. Inputs and the static context are never written.
    After preparation, asynchronous on the current stream without host reads or synchronization. The first call may
    prepare static graph data with counted transfers and one synchronization; prepare explicitly before a steady loop.
    Callers must order graph/input producers on other streams with synchronization or events before preparation/use.

    `mode` (see SWEEP_MODES): "level" = one launch per level (accepted Phase 4R comparator), "block" = one launch of
    one block walking all levels with a barrier between them, "auto"/None = `resolve_sweep_mode` (block for graphs
    whose widest level has at most NARROW_MAX_WIDTH cells). All modes give bitwise identical outputs (same per-cell
    operations); an invalid mode is refused before any launch."""
    c, iterations = _check_sweep_options(c, iterations)
    resolve_sweep_mode(0, mode)  # an invalid mode name is refused before any preparation or launch
    cp = _cupy()
    ctx = prepare_cuda_routing(graph)
    _check_device_array(cp, base_lo, "base_lo", np.float64, (ctx.n_active,), ctx.device_id)
    n = ctx.n_active
    c64, it32, n64 = np.float64(c), np.int32(iterations), np.int64(n)
    if resolve_sweep_mode(ctx.max_level_width, mode) == "block":
        kernel = _get_block_kernel()
        outputs = tuple(cp.empty(n, dtype=np.float64) for _ in range(4))
        try:
            kernel((1,), (BLOCK_THREADS,),
                   (ctx.level_bounds_device, np.int32(len(ctx.level_bounds) - 1), n64, ctx.conveyance_lo,
                    ctx.donor_position, ctx.donor_mask, base_lo, c64, it32, *outputs))
        except Exception as exc:  # re-raised with context, never swallowed
            raise RoutingError(f"CUDA one-block routing kernel launch failed ({type(exc).__name__}: {exc}); "
                               "no fallback") from exc
        return outputs
    kernel = _get_kernel()
    outputs = tuple(cp.empty(n, dtype=np.float64) for _ in range(4))
    for lev, (b0, b1) in enumerate(itertools.pairwise(ctx.level_bounds)):
        m = b1 - b0
        if m == 0:
            continue
        grid = (m + BLOCK_THREADS - 1) // BLOCK_THREADS
        try:
            kernel((grid,), (BLOCK_THREADS,),
                   (np.int64(b0), np.int64(m), n64, ctx.conveyance_lo, ctx.donor_position, ctx.donor_mask,
                    base_lo, c64, it32, *outputs))
        except Exception as exc:  # re-raised with context, never swallowed
            raise RoutingError(f"CUDA routing kernel launch failed at level {lev} of {len(ctx.level_bounds) - 1} "
                               f"({type(exc).__name__}: {exc}); no fallback") from exc
    return outputs
