"""CUDA safeguarded Newton root solver for the method-5 cell equation (explicit; bisection stays the default).

The device function `maple_syrup_newton` is the FP64 port, operation for operation, of the per-cell root of
`routing_newton.newton_root_scalar` (the single specification; read that docstring for the equation, the upper-bound
start `x0`, the bracket, the safeguard step, the polishing tries and the bounded bisection fallback). Every
arithmetic operation is a round-to-nearest intrinsic (`__dadd_rn`, `__dsub_rn`, `__dmul_rn`, `__ddiv_rn`,
`__dsqrt_rn`), the module is compiled with `--fmad=false --ftz=false` and no fast math, and the numeric constants
(`eps`, `16 eps`, `4 eps`, the halving limit, the polishing count) are taken from `routing_newton` at import, so they
cannot drift from the CPU form. Given identical inputs the root is therefore expected to be bitwise equal to the
Numba/pure-Python root (tests check this on scalar batches; a full storm still differs by device libm and reduction
order, with unchanged bounds).

The device text is ONE helper (`newton_device_source`), shared by

* the stand-alone sweep kernels here (`run_sweep`: "level" = one launch per dependency level, "block" = one launch of
  one block with `__syncthreads()` between levels, selected exactly like `routing_cuda.run_sweep`),
* the stand-alone per-root probe kernel (`solve_roots`, for scalar batches and untimed diagnostics) and
* the fused/split coupled hydrology Newton variant of `hydrology_cuda` (through the stats-free wrapper
  `maple_syrup_newton_root`).

The bisection sources of `routing_cuda`/`hydrology_cuda` are untouched; Newton lives in separate kernels, so the
default carries no extra branch. Inputs, donor order, dependency barriers and the storage identity are those of the
bisection sweep; the CPU safeguards (`rhs > 0`, analytic pit/tiny branch, bracket completion, <= 1200 halvings) are
inside the root, intentionally, and are not a bisection solver substitution.

Statistics (untimed diagnostics): `run_sweep(..., stats=True)` additionally returns a device `uint64[5]`
(`routing_newton.STAT_NAMES`: max passes, total passes, safeguard+fallback bisection steps, fallback cells, iterated
cells) accumulated with device atomics (integer, so order independent). Production calls (`stats=False`, what
`route_step` and the hydrology use) pass a null pointer: no atomics, no extra allocation, no transfer, and
`RouteStep.root_stats` is None (not zeros). The coupled hydrology reports no Newton counters; reproduce them with the
routing probe on the same inputs.

CuPy is optional and imported lazily (importing this module does not import CuPy). No fallback: missing CuPy, device,
compile or load failure raises `CudaUnavailableError`; a launch failure raises `RoutingError`.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import threading
import time
from typing import Any

import numpy as np

from maple_syrup import routing_cuda
from maple_syrup.routing import RoutingError, RoutingGraph
from maple_syrup.routing_cuda import CudaUnavailableError
from maple_syrup.routing_newton import (
    _CONVERGED_STEP,
    _EPS,
    _FALLBACK_LIMIT,
    _FALLBACK_STEP,
    _POLISH_TRIES,
    DEFAULT_NEWTON_MAX_ITERATIONS,
    MAX_NEWTON_ITERATIONS,
    STAT_NAMES,
)

__all__ = [
    "BLOCK_KERNEL_NAME",
    "LEVEL_KERNEL_NAME",
    "ROOTS_KERNEL_NAME",
    "check_newton_options",
    "ensure_loaded",
    "kernel_provenance",
    "newton_device_source",
    "prepare_cuda_newton",
    "require_inputs",
    "run_sweep",
    "solve_roots",
    "stats_to_dict",
]

LEVEL_KERNEL_NAME = "maple_syrup_newton_level"
BLOCK_KERNEL_NAME = "maple_syrup_newton_block"
ROOTS_KERNEL_NAME = "maple_syrup_newton_roots"
_KERNEL_NAMES = (LEVEL_KERNEL_NAME, BLOCK_KERNEL_NAME, ROOTS_KERNEL_NAME)


def _lit(value: float) -> str:
    return repr(float(value))  # shortest round-trip literal: the compiler reads back exactly the same double


# One copy of the physics. Constants are substituted from the CPU module. `stats` pointers are per-call outputs.
_DEVICE_SOURCE = r"""
#define MS_EPS __EPS__
#define MS_CONVERGED_STEP __CONVERGED__
#define MS_FALLBACK_STEP __FALLBACK__
#define MS_FALLBACK_LIMIT __LIMIT__
#define MS_POLISH_TRIES __TRIES__

// exactly the bisection operation sequence: mid + (((sqrt(mid) mid) k) c)
__device__ __forceinline__ double maple_syrup_trial(const double x, const double k, const double c)
{
    double t = __dsqrt_rn(x);
    t = __dmul_rn(t, x);
    t = __dmul_rn(t, k);
    t = __dmul_rn(t, c);
    return __dadd_rn(t, x);
}

// routing_newton.newton_root_scalar. Returns h_flow; o_it = Newton passes, o_sf = safeguard + fallback bisection
// steps, o_fb = 1 if the bounded bisection fallback ran.
__device__ __forceinline__ double maple_syrup_newton(const double rhs, const double k, const double c,
                                                     const int max_iter, int& o_it, int& o_sf, int& o_fb)
{
    o_it = 0; o_sf = 0; o_fb = 0;
    if (!(rhs > 0.0)) return 0.0;
    if (!(maple_syrup_trial(rhs, k, c) > rhs)) return rhs;
    const double a = __dmul_rn(c, k);
    const double h3 = __dmul_rn(1.5, a);
    double lo = 0.0;
    double hi = rhs;
    double x = rhs;
    if (a > 0.0) {
        const double inner = __ddiv_rn(rhs, __dadd_rn(1.0, __dmul_rn(a, __dsqrt_rn(rhs))));
        x = __ddiv_rn(rhs, __dadd_rn(1.0, __dmul_rn(a, __dsqrt_rn(inner))));
        if (!(x < rhs)) x = rhs;
    }
    int iters = 0;
    int safe = 0;
    bool converged = false;
    while (iters < max_iter) {
        ++iters;
        const double t = maple_syrup_trial(x, k, c);
        if (t < rhs) {
            if (x > lo) lo = x;
        } else if (x < hi) {
            hi = x;
        }
        const double f = __dsub_rn(t, rhs);
        if (f == 0.0) { converged = true; break; }
        double xn = __dsub_rn(x, __ddiv_rn(f, __dadd_rn(1.0, __dmul_rn(h3, __dsqrt_rn(x)))));
        bool tiny = fabs(__dsub_rn(xn, x)) <= __dmul_rn(MS_CONVERGED_STEP, xn);
        if (!tiny && !(xn > lo && xn < hi)) {
            xn = __dadd_rn(lo, __dmul_rn(0.5, __dsub_rn(hi, lo)));
            ++safe;
            tiny = fabs(__dsub_rn(xn, x)) <= __dmul_rn(MS_CONVERGED_STEP, xn);
        }
        x = xn;
        if (tiny) { converged = true; break; }
    }
    bool accepted = false;
    if (converged) {
        for (int j = 0; j < MS_POLISH_TRIES; ++j) {
            const double m = (j == 0) ? 0.0 : (double)(1 << (j - 1));
            const double y = __dsub_rn(x, __dmul_rn(__dmul_rn(m, MS_EPS), x));
            if (!(y > lo)) { accepted = lo > 0.0; break; }
            if (maple_syrup_trial(y, k, c) < rhs) { lo = y; accepted = true; break; }
            if (y < hi) hi = y;
        }
    }
    int fb = 0;
    if (!accepted) {
        fb = 1;
        for (int i = 0; i < MS_FALLBACK_LIMIT; ++i) {
            const double mid = __dadd_rn(lo, __dmul_rn(0.5, __dsub_rn(hi, lo)));
            if (!(mid > lo && mid < hi)) break;
            if (maple_syrup_trial(mid, k, c) < rhs) lo = mid; else hi = mid;
            ++safe;
            if (__dsub_rn(hi, lo) <= __dmul_rn(MS_FALLBACK_STEP, hi)) break;
        }
    }
    o_it = iters; o_sf = safe; o_fb = fb;
    return lo;
}

// Stats-free form used by the coupled hydrology kernels (same root, no counters).
__device__ __forceinline__ double maple_syrup_newton_root(const double rhs, const double k, const double c,
                                                          const int max_iter)
{
    int it, sf, fb;
    return maple_syrup_newton(rhs, k, c, max_iter, it, sf, fb);
}
"""

# Stand-alone sweep and probe kernels around the shared helper. `stats` may be null (production).
_SWEEP_SOURCE = r"""
__device__ __forceinline__ void maple_syrup_newton_cell(
    const long long p, const long long n_active, const double* k_lo, const long long* donor_position,
    const unsigned char* donor_mask, const double* base_lo, const double c, const int max_iter,
    double* qin_new_lo, double* q_new_lo, double* flow_lo, double* rhs_lo, unsigned long long* stats)
{
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
    int it, sf, fb;
    const double lo = maple_syrup_newton(rhs, k, c, max_iter, it, sf, fb);
    double q = __dsqrt_rn(lo);
    q = __dmul_rn(q, lo);
    q = __dmul_rn(q, k);
    qin_new_lo[p] = qin;
    q_new_lo[p] = q;
    flow_lo[p] = lo;
    rhs_lo[p] = rhs;
    if (stats != nullptr) {
        atomicMax(&stats[0], (unsigned long long)it);
        atomicAdd(&stats[1], (unsigned long long)it);
        atomicAdd(&stats[2], (unsigned long long)sf);
        atomicAdd(&stats[3], (unsigned long long)fb);
        if (it > 0) atomicAdd(&stats[4], 1ULL);
    }
}

extern "C" __global__ void maple_syrup_newton_level(
    const long long b0, const long long m, const long long n_active,
    const double* k_lo, const long long* donor_position, const unsigned char* donor_mask, const double* base_lo,
    const double c, const int max_iter,
    double* qin_new_lo, double* q_new_lo, double* flow_lo, double* rhs_lo, unsigned long long* stats)
{
    const long long j = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (j >= m) return;
    maple_syrup_newton_cell(b0 + j, n_active, k_lo, donor_position, donor_mask, base_lo, c, max_iter,
                            qin_new_lo, q_new_lo, flow_lo, rhs_lo, stats);
}

// One block walks every level; every thread reaches every barrier (bounds are read by all threads, the barrier is
// outside the strided cell loop).
extern "C" __global__ void maple_syrup_newton_block(
    const long long* level_bounds, const int n_levels, const long long n_active,
    const double* k_lo, const long long* donor_position, const unsigned char* donor_mask, const double* base_lo,
    const double c, const int max_iter,
    double* qin_new_lo, double* q_new_lo, double* flow_lo, double* rhs_lo, unsigned long long* stats)
{
    for (int lev = 0; lev < n_levels; ++lev) {
        const long long b0 = level_bounds[lev];
        const long long m = level_bounds[lev + 1] - b0;
        for (long long j = (long long)threadIdx.x; j < m; j += (long long)blockDim.x) {
            maple_syrup_newton_cell(b0 + j, n_active, k_lo, donor_position, donor_mask, base_lo, c, max_iter,
                                    qin_new_lo, q_new_lo, flow_lo, rhs_lo, stats);
        }
        __syncthreads();
    }
}

// Independent roots (scalar batches, untimed diagnostics): one thread per (rhs, k) pair.
extern "C" __global__ void maple_syrup_newton_roots(
    const long long n, const double* rhs, const double* k, const double c, const int max_iter,
    double* flow, long long* passes, long long* steps, unsigned char* fallback)
{
    const long long j = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (j >= n) return;
    int it, sf, fb;
    flow[j] = maple_syrup_newton(rhs[j], k[j], c, max_iter, it, sf, fb);
    passes[j] = it;
    steps[j] = sf;
    fallback[j] = (unsigned char)fb;
}
"""

_DEVICE_TEXT = (_DEVICE_SOURCE.replace("__EPS__", _lit(_EPS)).replace("__CONVERGED__", _lit(_CONVERGED_STEP))
                .replace("__FALLBACK__", _lit(_FALLBACK_STEP)).replace("__LIMIT__", str(int(_FALLBACK_LIMIT)))
                .replace("__TRIES__", str(int(_POLISH_TRIES))))
_SOURCE = _DEVICE_TEXT + _SWEEP_SOURCE
_BLOCK_THREADS = routing_cuda.BLOCK_THREADS

_MODULE: Any = None
_FUNCTIONS: dict[tuple[int, str], Any] = {}
_LOCK = threading.Lock()


def newton_device_source() -> str:
    """The shared device helpers `maple_syrup_trial`, `maple_syrup_newton` and `maple_syrup_newton_root` (RN FP64
    intrinsics only; constants substituted from `routing_newton`). Included verbatim by the coupled hydrology."""
    return _DEVICE_TEXT


def kernel_source() -> str:
    """The full source of this module's kernels (device helper + sweep/probe kernels)."""
    return _SOURCE


def _function(name: str):
    global _MODULE
    cp = routing_cuda._cupy()
    key = (routing_cuda._current_device_id(cp), name)
    fn = _FUNCTIONS.get(key)
    if fn is not None:
        return fn
    with _LOCK:
        try:
            if _MODULE is None:
                _MODULE = cp.RawModule(code=_SOURCE, options=routing_cuda.COMPILE_OPTIONS, backend="nvrtc")
            fn = _MODULE.get_function(name)
            attrs = dict(fn.attributes)  # forces compile + load for this device now
            limit = int(attrs.get("max_threads_per_block", 0))
            if limit < _BLOCK_THREADS:
                raise RuntimeError(f"only {limit} threads per block supported, {_BLOCK_THREADS} needed")
        except Exception as exc:
            raise CudaUnavailableError(
                f"CUDA Newton kernel {name!r} failed to compile/load ({type(exc).__name__}: {exc}); "
                "no fallback") from exc
        _FUNCTIONS[key] = fn
    return fn


def ensure_loaded() -> float:
    """Compile/load the Newton sweep and probe kernels for the CURRENT device (idempotent). Returns the seconds this
    call spent (about 0 once loaded), so a caller can record startup apart from the timed evolution."""
    t0 = time.perf_counter()
    for name in _KERNEL_NAMES:
        _function(name)
    return time.perf_counter() - t0


def prepare_cuda_newton(graph: RoutingGraph):
    """`routing_cuda.prepare_cuda_routing(graph)` plus the Newton kernels (the static context is the shared one)."""
    ctx = routing_cuda.prepare_cuda_routing(graph)
    ensure_loaded()
    return ctx


def require_inputs(graph: RoutingGraph, dynamic: dict[str, Any]):
    """`routing_cuda.require_inputs` plus the Newton kernels, called by `route_step` before any shared arithmetic."""
    ctx = routing_cuda.require_inputs(graph, dynamic)
    ensure_loaded()
    return ctx


def check_newton_options(c: Any, max_iterations: Any) -> tuple[float, int]:
    if isinstance(c, bool) or not isinstance(c, (int, float, np.integer, np.floating)):
        raise RoutingError(f"c must be a real number, got {type(c).__name__}")
    c = float(c)
    if not (math.isfinite(c) and c > 0.0):
        raise RoutingError(f"c must be finite and > 0, got {c!r}")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)) \
            or not (1 <= int(max_iterations) <= MAX_NEWTON_ITERATIONS):
        raise RoutingError(f"newton_max_iterations must be an int in [1, {MAX_NEWTON_ITERATIONS}], "
                           f"got {max_iterations!r}")
    return c, int(max_iterations)


def stats_to_dict(stats: Any) -> dict[str, int]:
    """Host dict of a downloaded `run_sweep(..., stats=True)` counter block (a counted transfer; untimed use)."""
    from maple.core.backend import to_host

    return {name: int(v) for name, v in zip(STAT_NAMES, to_host(stats), strict=True)}


def run_sweep(graph: RoutingGraph, base_lo: Any, c: float, max_iterations: int = DEFAULT_NEWTON_MAX_ITERATIONS,
              mode: str | None = None, *, stats: bool = False):
    """The Newton form of `routing_cuda.run_sweep`: same inputs, static context, launch structure (`mode`), donor
    order and four fresh level-ordered outputs `(qin_new_lo, q_new_lo, flow_lo, rhs_lo)`; asynchronous on the current
    stream, no host read. With `stats=True` a fifth fresh `uint64[5]` device block of counters is returned (see the
    module docstring); the default allocates and writes none. Options are validated before any launch."""
    c, max_iterations = check_newton_options(c, max_iterations)
    routing_cuda.resolve_sweep_mode(0, mode)
    cp = routing_cuda._cupy()
    ctx = prepare_cuda_newton(graph)
    routing_cuda._check_device_array(cp, base_lo, "base_lo", np.float64, (ctx.n_active,), ctx.device_id)
    n = ctx.n_active
    c64, it32, n64 = np.float64(c), np.int32(max_iterations), np.int64(n)
    outputs = tuple(cp.empty(n, dtype=np.float64) for _ in range(4))
    counters = cp.zeros(len(STAT_NAMES), dtype=np.uint64) if stats else None
    stat_arg = counters if stats else np.uint64(0)  # a null device pointer for production (no atomics)
    if routing_cuda.resolve_sweep_mode(ctx.max_level_width, mode) == "block":
        try:
            _function(BLOCK_KERNEL_NAME)(
                (1,), (_BLOCK_THREADS,),
                (ctx.level_bounds_device, np.int32(len(ctx.level_bounds) - 1), n64, ctx.conveyance_lo,
                 ctx.donor_position, ctx.donor_mask, base_lo, c64, it32, *outputs, stat_arg))
        except Exception as exc:
            raise RoutingError(f"CUDA one-block Newton kernel launch failed ({type(exc).__name__}: {exc}); "
                               "no fallback") from exc
    else:
        kernel = _function(LEVEL_KERNEL_NAME)
        for lev, (b0, b1) in enumerate(itertools.pairwise(ctx.level_bounds)):
            m = b1 - b0
            if m == 0:
                continue
            grid = (m + _BLOCK_THREADS - 1) // _BLOCK_THREADS
            try:
                kernel((grid,), (_BLOCK_THREADS,),
                       (np.int64(b0), np.int64(m), n64, ctx.conveyance_lo, ctx.donor_position, ctx.donor_mask,
                        base_lo, c64, it32, *outputs, stat_arg))
            except Exception as exc:
                raise RoutingError(f"CUDA Newton kernel launch failed at level {lev} of "
                                   f"{len(ctx.level_bounds) - 1} ({type(exc).__name__}: {exc}); no fallback") from exc
    return (*outputs, counters) if stats else outputs


def solve_roots(rhs: Any, k: Any, c: float, max_iterations: int = DEFAULT_NEWTON_MAX_ITERATIONS):
    """Independent roots of `h + c k h^{3/2} = rhs` on the device: exact contiguous float64 cupy 1-D arrays `rhs` and
    `k` of one length (VALUES are not inspected). Returns fresh device arrays `(flow float64, passes int64,
    bisection_steps int64, fallback uint8)`, the per-root results of `routing_newton.newton_root_scalar`. A
    stand-alone probe for scalar batches and untimed diagnostics: one thread per root, no donors, no transfer."""
    c, max_iterations = check_newton_options(c, max_iterations)
    cp = routing_cuda._cupy()
    device_id = routing_cuda._current_device_id(cp)
    if type(rhs) is not cp.ndarray or rhs.ndim != 1:
        raise RoutingError("rhs must be a 1-D exact cupy.ndarray")
    n = int(rhs.shape[0])
    routing_cuda._check_device_array(cp, rhs, "rhs", np.float64, (n,), device_id)
    routing_cuda._check_device_array(cp, k, "k", np.float64, (n,), device_id)
    kernel = _function(ROOTS_KERNEL_NAME)
    flow = cp.empty(n, dtype=np.float64)
    passes = cp.empty(n, dtype=np.int64)
    steps = cp.empty(n, dtype=np.int64)
    fallback = cp.empty(n, dtype=np.uint8)
    if n:
        try:
            kernel(((n + _BLOCK_THREADS - 1) // _BLOCK_THREADS,), (_BLOCK_THREADS,),
                   (np.int64(n), rhs, k, np.float64(c), np.int32(max_iterations), flow, passes, steps, fallback))
        except Exception as exc:
            raise RoutingError(f"CUDA Newton root kernel launch failed ({type(exc).__name__}: {exc}); "
                               "no fallback") from exc
    return flow, passes, steps, fallback


def kernel_provenance() -> dict[str, Any]:
    """Newton source hash, kernels, options and the toolchain/device of `routing_cuda.kernel_provenance`."""
    info = routing_cuda.kernel_provenance()
    info.update({
        "root_solver": "newton",
        "newton_module": "maple_syrup.routing_newton_cuda",
        "newton_source_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
        "newton_device_source_sha256": hashlib.sha256(_DEVICE_TEXT.encode()).hexdigest(),
        "newton_kernels": list(_KERNEL_NAMES),
        "newton_intrinsics": ["__dadd_rn", "__dsub_rn", "__dmul_rn", "__ddiv_rn", "__dsqrt_rn"],
        "newton_constants": {"eps": _EPS, "converged_step": _CONVERGED_STEP, "fallback_step": _FALLBACK_STEP,
                             "fallback_limit": int(_FALLBACK_LIMIT), "polish_tries": int(_POLISH_TRIES),
                             "default_max_iterations": DEFAULT_NEWTON_MAX_ITERATIONS,
                             "max_iterations": MAX_NEWTON_ITERATIONS},
        "newton_stats": "production: null pointer, no counters (RouteStep.root_stats None); probes: uint64[5] "
                        + ",".join(STAT_NAMES),
    })
    return info
