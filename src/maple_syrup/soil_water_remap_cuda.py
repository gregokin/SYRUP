"""CuPy RawKernel CUDA runner of `maple_syrup.soil_water_remap` (backend "cuda"). Component/trial only.

Two explicit kernels, both on the current stream, in order; there is NO library reduction (no CuPy `sum`/`stack`/`argmax`, hence no hidden library workspace or per-reduction scalar allocation):

  1. `maple_syrup_soil_water_remap` (grid `ceil(C/CUDA_BLOCK)`, `CUDA_BLOCK` threads): one thread per column evaluates EXACTLY the arithmetic of `soil_water_remap.reference_remap` in the same expression order
     with the same IEEE operations (compiled with `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`, no fast math), writes the outputs, and keeps its five contributions (old total, new total,
     added, removed, scale; 0 for a flagged column or a thread beyond C) in registers. The block then reduces the contributions in STATIC SHARED MEMORY (`5 * 8 * CUDA_BLOCK` bytes, a fixed binary tree: the
     result is deterministic) and writes `5 * n_blocks` doubles of block partials. Failures are reported by device atomics into the diagnostic words (below); a thread never returns before the barriers.
  2. `maple_syrup_soil_water_remap_finalize` (ONE block of `CUDA_FINAL_BLOCK` threads, static shared memory `5 * 8 * CUDA_FINAL_BLOCK` bytes): each thread accumulates a strided share of the partials, a fixed
     tree reduces them, and thread 0 stores the five totals as raw double bit patterns into diagnostic words 5..9.

Device memory of this call (all counted by `estimate_remap_memory` BEFORE allocation): the retained outputs `8LC + 32C`; the five-word-pair diagnostic/packet `DIAG_WORDS * 8` = 80 B; the block partials
`5 * 8 * n_blocks`; nothing else (no per-column stats/flag/index/mask array, no |residual|/ratio array). Static shared memory per block is reported in the estimate (it is not a global allocation).
Excluded: NVRTC/module load, the CUDA context and the allocator pool policy (not this call's algorithm memory).

Diagnostic/packet = `DIAG_WORDS = 10` `uint64` words, read back with ONE device-to-host transfer of exactly `PACKET_BYTES` = 80 B and kept as RAW INTEGERS: [0] OR of all flags, [1] number of flagged columns,
[2] `min(column << 8 | flags)` = the first flagged column with its flags (no float conversion, so exact), [3] bit pattern of max |residual|, [4] bit pattern of max |residual|/bound (monotone for nonnegative doubles, so
`atomicMax` on the bit patterns is the double maximum), [5..9] bit patterns of the five totals. EXPLICIT RANGE GATE (checked before any allocation, `estimate_remap_memory`): `C <= MAX_CUDA_COLUMNS =
min(2**56 - 1, (2**31 - 1) * CUDA_BLOCK)`, so `column << 8` fits 64 bits and the grid fits the launch limit.

Adjacent threads read adjacent columns of the C-contiguous `(L+1, C)` / `(L, C)` arrays (coalesced); no per-thread local arrays. A flagged column is skipped (its outputs stay 0) and the caller raises before any
result is built, so a failed call publishes nothing. Columns whose edges are numerically unchanged are an identity copy (every water bit, fluxes +0). CuPy, a visible device and a loadable kernel are required
(`require_cuda`); any failure is a `SoilRemapBackendUnavailableError`, never a NumPy fallback. The device available-bytes reading is the driver's free memory plus the default pool's free bytes limited by the pool
limit (the same four-number arithmetic as the other SYRUP CUDA modules, kept local). Written by its author without launching a GPU; Root qualifies it on hardware.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from maple_syrup.soil_water_remap import (
    CUDA_BLOCK,
    CUDA_FINAL_BLOCK,
    CUDA_REDUCED_ROWS,
    DIAG_WORDS,
    FLAG_CLOSURE,
    FLAG_NEW_EDGES,
    FLAG_OLD_EDGES,
    FLAG_OUTPUT,
    FLAG_THETA,
    FLAG_WATER,
    SoilRemapBackendUnavailableError,
    SoilRemapError,
    cuda_block_count,
)

__all__ = ["COMPILE_OPTIONS", "current_device_id", "default_device_available_bytes_fn", "effective_free_device_bytes", "require_cuda", "run"]

COMPILE_OPTIONS = ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")
_LOCK = threading.Lock()
_FUNCTIONS: dict[int, Any] = {}

_SOURCE = r"""
__device__ __forceinline__ void report_flag(unsigned long long* diag, long long c, int f)
{
    /* diag[0] OR of all flags, diag[1] number of flagged columns, diag[2] min over (column << 8 | flags): the first flagged column with its flags */
    atomicOr(&diag[0], (unsigned long long)f);
    atomicAdd(&diag[1], 1ULL);
    atomicMin(&diag[2], (((unsigned long long)c) << 8) | (unsigned long long)f);
}

/* One column. Returns 1 and fills contrib[0..4] = old total, new total, added (aa+ab), removed (ra+rb), scale on success; returns 0 (after reporting a flag) otherwise. No barrier inside. */
__device__ int column_work(
    const double* __restrict__ eo, const double* __restrict__ en, const double* __restrict__ W,
    const double* __restrict__ up, const double* __restrict__ lo,
    double* __restrict__ out, double* __restrict__ rem_a, double* __restrict__ rem_b,
    double* __restrict__ add_a, double* __restrict__ add_b,
    unsigned long long* diag, double* contrib,
    const double coef_col, const double coef_out, const long long L, const long long C, const long long c)
{
    int flag = 0;
    for (long long i = 0; i < L; ++i) {
        double w = eo[(i + 1) * C + c] - eo[i * C + c];
        if (!(isfinite(w) && w > 0.0)) flag |= FLAG_OLD;
        double wn_ = en[(i + 1) * C + c] - en[i * C + c];
        if (!(isfinite(wn_) && wn_ > 0.0)) flag |= FLAG_NEW;
        w = eo[(i + 1) * C + c] - eo[i * C + c];
        double x = W[i * C + c];
        if (!(isfinite(x) && x >= 0.0 && x <= w)) flag |= FLAG_WAT;
    }
    double u = up[c];
    double v = lo[c];
    if (!(isfinite(u) && u >= 0.0 && u <= 1.0 && isfinite(v) && v >= 0.0 && v <= 1.0)) flag |= FLAG_TH;
    if (flag != 0) { report_flag(diag, c, flag); return 0; }
    double old_total = 0.0, scale = 0.0;
    for (long long i = 0; i < L; ++i) {
        double w = eo[(i + 1) * C + c] - eo[i * C + c];
        scale = scale + w;
        old_total = old_total + W[i * C + c];
    }
    for (long long j = 0; j < L; ++j) {
        double wn_ = en[(j + 1) * C + c] - en[j * C + c];
        scale = scale + wn_;
    }
    double ra = 0.0, rb = 0.0;
    for (long long i = 0; i < L; ++i) {
        double w = eo[(i + 1) * C + c] - eo[i * C + c];
        double t = fmin(eo[(i + 1) * C + c], en[c]);
        t = t - eo[i * C + c];
        t = fmax(t, 0.0);
        t = t / w;
        t = W[i * C + c] * t;
        ra = ra + t;
        double s = fmax(eo[i * C + c], en[L * C + c]);
        s = eo[(i + 1) * C + c] - s;
        s = fmax(s, 0.0);
        s = s / w;
        s = W[i * C + c] * s;
        rb = rb + s;
    }
    double a1 = fmin(eo[c], en[L * C + c]);
    a1 = a1 - en[c];
    a1 = fmax(a1, 0.0);
    double aa = u * a1;
    double b1 = fmax(eo[L * C + c], en[c]);
    b1 = en[L * C + c] - b1;
    b1 = fmax(b1, 0.0);
    double ab = v * b1;
    double new_total = 0.0;
    bool bad_out = false;
    for (long long j = 0; j < L; ++j) {
        double acc = 0.0;
        for (long long i = 0; i < L; ++i) {
            double w = eo[(i + 1) * C + c] - eo[i * C + c];
            double p = fmin(eo[(i + 1) * C + c], en[(j + 1) * C + c]);
            double q = fmax(eo[i * C + c], en[j * C + c]);
            p = p - q;
            p = fmax(p, 0.0);
            p = p / w;
            p = W[i * C + c] * p;
            acc = acc + p;
        }
        double p2 = fmin(en[(j + 1) * C + c], eo[c]);
        p2 = p2 - en[j * C + c];
        p2 = fmax(p2, 0.0);
        p2 = u * p2;
        acc = acc + p2;
        double q2 = fmax(en[j * C + c], eo[L * C + c]);
        q2 = en[(j + 1) * C + c] - q2;
        q2 = fmax(q2, 0.0);
        q2 = v * q2;
        acc = acc + q2;
        out[j * C + c] = acc;
        double wn_ = en[(j + 1) * C + c] - en[j * C + c];
        double tol = wn_ * coef_out;
        tol = wn_ + tol;
        if (!(isfinite(acc) && acc <= tol)) bad_out = true;
        new_total = new_total + acc;
    }
    bool same = true;
    for (long long i = 0; i <= L; ++i) { if (eo[i * C + c] != en[i * C + c]) same = false; }
    if (same) {  /* unchanged geometry: identity copy keeps every water bit; every flux is exactly +0 */
        new_total = 0.0;
        for (long long j = 0; j < L; ++j) { out[j * C + c] = W[j * C + c]; new_total = new_total + W[j * C + c]; }
        ra = 0.0; rb = 0.0; aa = 0.0; ab = 0.0;
    }
    rem_a[c] = ra; rem_b[c] = rb; add_a[c] = aa; add_b[c] = ab;
    if (!(isfinite(ra) && isfinite(rb) && isfinite(aa) && isfinite(ab))) bad_out = true;
    if (bad_out) { report_flag(diag, c, FLAG_OUT); return 0; }
    double d1 = new_total - old_total;
    double d2 = aa + ab;
    double d3 = ra + rb;
    d2 = d2 - d3;
    double res = d1 - d2;
    double bound = scale * coef_col;
    if (!(isfinite(res) && isfinite(bound) && fabs(res) <= bound)) { report_flag(diag, c, FLAG_CL); return 0; }
    /* maxima of the nonnegative finite doubles |res| and |res|/bound through their (monotone) bit patterns */
    atomicMax(&diag[3], (unsigned long long)__double_as_longlong(fabs(res)));
    atomicMax(&diag[4], (unsigned long long)__double_as_longlong(fabs(res) / bound));
    contrib[0] = old_total; contrib[1] = new_total; contrib[2] = aa + ab; contrib[3] = ra + rb; contrib[4] = scale;
    return 1;
}

extern "C" __global__ void maple_syrup_soil_water_remap(
    const double* __restrict__ eo, const double* __restrict__ en, const double* __restrict__ W,
    const double* __restrict__ up, const double* __restrict__ lo,
    double* __restrict__ out, double* __restrict__ rem_a, double* __restrict__ rem_b,
    double* __restrict__ add_a, double* __restrict__ add_b,
    unsigned long long* diag, double* __restrict__ partials,
    const double coef_col, const double coef_out, const long long L, const long long C)
{
    __shared__ double sh[NROWS][BLOCK];
    const int tid = threadIdx.x;
    const long long c = (long long)blockIdx.x * (long long)BLOCK + (long long)tid;
    double contrib[NROWS];
    for (int k = 0; k < NROWS; ++k) contrib[k] = 0.0;
    if (c < C) {
        if (!column_work(eo, en, W, up, lo, out, rem_a, rem_b, add_a, add_b, diag, contrib, coef_col, coef_out, L, C, c)) {
            for (int k = 0; k < NROWS; ++k) contrib[k] = 0.0;  /* a flagged column contributes nothing */
        }
    }
    for (int k = 0; k < NROWS; ++k) sh[k][tid] = contrib[k];
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) { for (int k = 0; k < NROWS; ++k) sh[k][tid] = sh[k][tid] + sh[k][tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { for (int k = 0; k < NROWS; ++k) partials[(long long)k * (long long)gridDim.x + (long long)blockIdx.x] = sh[k][0]; }
}

extern "C" __global__ void maple_syrup_soil_water_remap_finalize(
    const double* __restrict__ partials, const long long n_blocks, unsigned long long* diag)
{
    __shared__ double sh[NROWS][FINAL_BLOCK];
    const int tid = threadIdx.x;
    for (int k = 0; k < NROWS; ++k) {
        double acc = 0.0;
        for (long long b = tid; b < n_blocks; b += FINAL_BLOCK) acc = acc + partials[(long long)k * n_blocks + b];
        sh[k][tid] = acc;
    }
    __syncthreads();
    for (int s = FINAL_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) { for (int k = 0; k < NROWS; ++k) sh[k][tid] = sh[k][tid] + sh[k][tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { for (int k = 0; k < NROWS; ++k) diag[5 + k] = (unsigned long long)__double_as_longlong(sh[k][0]); }
}
"""


def _source() -> str:
    defs = {"FLAG_OLD": FLAG_OLD_EDGES, "FLAG_NEW": FLAG_NEW_EDGES, "FLAG_WAT": FLAG_WATER, "FLAG_TH": FLAG_THETA, "FLAG_OUT": FLAG_OUTPUT, "FLAG_CL": FLAG_CLOSURE,
            "NROWS": CUDA_REDUCED_ROWS, "BLOCK": CUDA_BLOCK, "FINAL_BLOCK": CUDA_FINAL_BLOCK}
    return "".join(f"#define {k} {v}\n" for k, v in defs.items()) + _SOURCE


def require_cuda():
    """The CuPy module with at least one visible device, through MAPLE's own probes; otherwise a named setup error (no fallback)."""
    from maple.core.backend import cuda_device_count, cupy_import_error, cupy_module

    cp = cupy_module()
    if cp is None:
        raise SoilRemapBackendUnavailableError(f"backend 'cuda' requires CuPy, which is not usable ({cupy_import_error() or 'not installed'}); there is no fallback to the NumPy reference")
    if cuda_device_count() < 1:
        raise SoilRemapBackendUnavailableError("backend 'cuda' requires a visible CUDA device (device count 0); there is no fallback to the NumPy reference")
    return cp


def current_device_id(cp) -> int:
    try:
        return int(cp.cuda.Device().id)
    except Exception as exc:  # noqa: BLE001 - reported as a setup failure
        raise SoilRemapBackendUnavailableError(f"CUDA device query failed ({type(exc).__name__}: {exc})") from exc


def effective_free_device_bytes(driver_free_bytes: int, pool_free_bytes: int, pool_limit_bytes: int, pool_used_bytes: int) -> int:
    """`driver_free + pool_free`, limited to `pool_limit - pool_used` when the pool has a positive limit; never negative."""
    for name, value in (("driver_free_bytes", driver_free_bytes), ("pool_free_bytes", pool_free_bytes), ("pool_limit_bytes", pool_limit_bytes), ("pool_used_bytes", pool_used_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SoilRemapError(f"{name} must be a nonnegative int, got {value!r}")
    reachable = driver_free_bytes + pool_free_bytes
    if pool_limit_bytes > 0:
        reachable = min(reachable, pool_limit_bytes - pool_used_bytes)
    return max(0, reachable)


def default_device_available_bytes_fn(cp):
    """Reading for the CURRENT device; refuses at call time if the device changed. Never changes the device or a limit."""
    device_id = current_device_id(cp)

    def query() -> int:
        current = current_device_id(cp)
        if current != device_id:
            raise SoilRemapError(f"the call is bound to device {device_id} but the current device is {current}")
        pool = cp.get_default_memory_pool()
        return effective_free_device_bytes(int(cp.cuda.Device().mem_info[0]), int(pool.free_bytes()), int(pool.get_limit()), int(pool.used_bytes()))

    return query


def _load_kernel(cp, device_id: int):
    with _LOCK:
        fn = _FUNCTIONS.get(device_id)
        if fn is not None:
            return fn
        try:
            module = cp.RawModule(code=_source(), options=COMPILE_OPTIONS, backend="nvrtc")
            fn = (module.get_function("maple_syrup_soil_water_remap"), module.get_function("maple_syrup_soil_water_remap_finalize"))
        except Exception as exc:  # noqa: BLE001 - a setup failure, never a fallback
            raise SoilRemapBackendUnavailableError(f"the CUDA remap kernel could not be compiled/loaded on device {device_id} ({type(exc).__name__}: {exc})") from exc
        _FUNCTIONS[device_id] = fn
        return fn


def run(cp, old_edges_m, new_edges_m, water_m, upper_theta, lower_theta, coef_col: float, coef_out: float) -> tuple:
    """Fresh device outputs plus the host `info = (flag OR, n_flagged, first flagged column, that column's flags)` and the host `sums` dict. Allocates exactly the outputs, the 10-word diagnostic and the
    `5 x n_blocks` partials (the memory model's terms); two kernel launches; ONE 80-byte device-to-host read of the raw diagnostic words."""
    L, C = int(water_m.shape[0]), int(water_m.shape[1])
    out = cp.zeros((L, C), dtype=cp.float64)
    rem_a, rem_b, add_a, add_b = (cp.zeros(C, dtype=cp.float64) for _ in range(4))
    if C == 0:
        return out, rem_a, rem_b, add_a, add_b, (0, 0, None, 0), {"old": 0.0, "new": 0.0, "added": 0.0, "removed": 0.0, "scale": 0.0, "max_abs_residual": 0.0, "max_ratio": 0.0}
    n_blocks = cuda_block_count(C)
    diag = cp.zeros(DIAG_WORDS, dtype=cp.uint64)
    diag[2] = np.uint64(0xFFFFFFFFFFFFFFFF)
    partials = cp.zeros((CUDA_REDUCED_ROWS, n_blocks), dtype=cp.float64)
    column_kernel, finalize_kernel = _load_kernel(cp, current_device_id(cp))
    column_kernel((n_blocks,), (CUDA_BLOCK,), (old_edges_m, new_edges_m, water_m, upper_theta, lower_theta, out, rem_a, rem_b, add_a, add_b, diag, partials,
                                                np.float64(coef_col), np.float64(coef_out), np.int64(L), np.int64(C)))
    finalize_kernel((1,), (CUDA_FINAL_BLOCK,), (partials, np.int64(n_blocks), diag))
    words = diag.get()  # the one device-to-host read: DIAG_WORDS raw uint64 words
    n_flagged = int(words[1])
    if n_flagged:
        packed = int(words[2])
        info = (int(words[0]), n_flagged, packed >> 8, packed & 0xFF)
    else:
        info = (0, 0, None, 0)
    doubles = words[3:10].view(np.float64)
    sums = {"old": float(doubles[2]), "new": float(doubles[3]), "added": float(doubles[4]), "removed": float(doubles[5]), "scale": float(doubles[6]),
            "max_abs_residual": float(doubles[0]), "max_ratio": float(doubles[1])}
    return out, rem_a, rem_b, add_a, add_b, info, sums
