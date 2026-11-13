"""CuPy RawKernel CUDA step of the fixed explicit SPATIAL preferential (bypass) infiltration (`spatial_preferential_infiltration`, backend "cuda"). Trial component only (S2-B2.d).

ONE physical body: the preferential kernel body of `preferential_infiltration_cuda` rendered by the pure, fail-closed `preferential_infiltration_cuda.render_source` with SPATIAL readers, a `__device__` head and a
reduction tail. The body becomes the `__device__` function `spatial_preferential_cell` (its `return` for `i >= n` is the padded-lane guard); the wrapper kernel then lets EVERY lane (padded or inactive) reach every
barrier. Readers are the A1 strided raw maps `p[l * stride_layer + i * stride_cell]` (scalar 0/0, shared 1/0, cell N/1):

    SAT_L  = sat_at(theta_sat, dz)          sat = theta_sat * dz                         (a rounded product)
    SATB_L = satb_at(theta_sat, dz, coef)   sat + coef * sat                             (the validity bound `sat_m[l] + cap_tol[l]`, cap_tol = coef * sat)
    RATE   = rate[i]                        the sealed derived continuous rate k (N,)
    MAXRATE= max_rate[i * stride]           the sealed max-rate input (one value, stride 0, or (N,), stride 1)

NO FUSION: the products are rounded before the next operation exactly as on the CPU; the module is compiled with the existing `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`. The law (ONE aggregate
continuous hourly-fraction path `H (-expm1(-k dt))`, ONE aggregate cap `max_rate dt`, TOP-LAYER-FIRST allocation bounded by each receiver's saturation room and the remaining surface, no remapping of a rejected
share) is the shared body, untouched. No dense `8 N` rows, no `(L, N)` bound tensor, no L-sized or per-thread contribution array, no host mirror, no CuPy/CUB reduction.

HOT STEP, in stream order: `maple_syrup_spatial_pref_step` (per lane the seven per-cell quantities of the common packet are stored by the shared body straight into the block's static shared rows with literal
offsets -- rows 0-6 are the sums: soil before/after, surface in/out, isolated attempt, attempted, intake; padded and inactive lanes hold 0; the flag word is OR-ed and the invalid cells counted in two more shared
words; one fixed block tree `7 * 8 * 128 + 8 * 128 + 4 * 128` B of static shared memory, `7 * nb` block partials, one `atomicOr` of the flag word and one `atomicAdd` of the flagged count per block into the raw packet)
-> `maple_syrup_spatial_pref_finalize` (one block of 256, a fixed tree, `7 * 8 * 256` B static shared; sums written as bit patterns) -> ONE `maple.core.backend.to_host` of the 9-word `uint64` raw packet (72 B: word 0 the
OR of the flag words, word 1 the flagged-cell count, words 2-8 the seven float64 bit patterns). The host rebuilds the unchanged 9-float common packet for `preferential_infiltration._resolve_core` (seven sums, the LOWEST
set bit of the OR = the minimum per-cell lowest bit, or `NO_FLAG`, and the count; each invalid cell counted once, inactive water included). Zero host-to-device copies per step; raw (72 B) and rebuilt (72 B) host
packets are both counted.

COLD derivation reducer (`ColdReducer`): the common derivation (`preferential_infiltration._derive_pathway`, reused, never copied) reaches its eight cold scalar reductions through the optional `_DeriveContext.reduce`
hook. This module supplies a bounded, counted implementation instead of the library `sum/min/max` (no CuPy/CUB workspace, no temporary mask): one pass kernel `maple_syrup_spatial_pref_reduce` keeps a 32-byte
`uint64` diagnostic `[#non-finite, #NaN, min-key, max-key]` (order-preserving integer keys of the doubles, `atomicMin/atomicMax` on `unsigned long long`). The 32-byte seed is a COLD host-to-device copy and the 32-byte
result a COLD synchronising device-to-host read (both host buffers counted: 64 B); the host decodes min/max (NaN propagates exactly as `xp.min/xp.max`; +-inf and the sign of zero follow the key order). Unknown kinds,
non-device/ non-float64/ non-contiguous/ wrong-device arrays are refused. Cold range validation of the scalar/cell inputs uses `maple_syrup_spatial_pref_validate` (one 8-byte diagnostic).

Memory/lifetimes (`operation_terms`): hot scratch (partials, raw packet) is held until after the synchronising read; outputs are fresh pool arrays; static shared memory is reported (not a global allocation). Excluded
and declared: NVRTC/module load, the CUDA context, allocator pool policy and rounding, interpreter. Written by its author without a GPU; Root qualifies it on hardware.
"""

from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import preferential_infiltration as pi
from maple_syrup import preferential_infiltration_cuda as pc
from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = ["COMPILE_OPTIONS", "FINAL_THREADS", "PACKET_WORDS", "RAW_PACKET_BYTES", "ROWS", "STAGE_ELEMS", "THREADS", "ColdReducer", "build_packet", "check_launch_ranges", "hash_device_array",
           "kernel_provenance", "kernel_source", "load_kernels", "operation_terms", "render_source", "step", "validate_range"]

COMPILE_OPTIONS = pc.COMPILE_OPTIONS  # --fmad=false --prec-div=true --prec-sqrt=true --ftz=false
THREADS = 128
FINAL_THREADS = 256
ROWS = 7  # the seven sums of the common packet
PACKET_WORDS = 9  # word 0: OR of the flag words, word 1: flagged cells, words 2..8: the 7 float64 bit patterns
RAW_PACKET_BYTES = 8 * PACKET_WORDS  # 72
DIAG_BYTES = 32  # the cold reducer's uint64 diagnostic [#non-finite, #NaN, min-key, max-key]
MAX_GRID = 2**31 - 1
STAGE_ELEMS = 16384  # float64 elements of the host staging buffer of the cold identity hash (128 KiB)
STEP_KERNEL = "maple_syrup_spatial_pref_step"
FINAL_KERNEL = "maple_syrup_spatial_pref_finalize"
REDUCE_KERNEL = "maple_syrup_spatial_pref_reduce"
VALIDATE_KERNEL = "maple_syrup_spatial_pref_validate"
_KERNELS = (STEP_KERNEL, FINAL_KERNEL, REDUCE_KERNEL, VALIDATE_KERNEL)
_SIGN = 1 << 63
_MAX_KEY = (1 << 64) - 1

SPATIAL_READERS = r"""
#define BLOCK 128
#define SPM(p, sl, sc, l) (p)[(long long)(l) * (sl) + i * (sc)]
__device__ __forceinline__ double sat_at(const double theta_sat, const double dz)
{
    const double sat = theta_sat * dz;  /* sat = theta_sat * dz (rounded): --fmad=false keeps it unfused */
    return sat;
}
__device__ __forceinline__ double satb_at(const double theta_sat, const double dz, const double coef)
{
    const double sat = theta_sat * dz;  /* sat = theta_sat * dz (rounded) */
    return sat + coef * sat;            /* the validity bound sat + cap_tol, cap_tol = coef * sat */
}
__device__ __forceinline__ unsigned long long order_key(const double v)
{
    const unsigned long long b = (unsigned long long)__double_as_longlong(v);
    return (b >> 63) ? ~b : (b | 0x8000000000000000ULL);  /* order-preserving: -inf < ... < -0.0 < +0.0 < ... < +inf */
}
"""

_INPUTS = r"""
    const long long n, const int nl,
    const double* surface_in, const double* w_in, const unsigned char* active, const double dt,
    const double* rate, const double* max_rate, const long long mr_sc, const double* recv,
    const double* dz_p, const long long dz_sl, const long long dz_sc,
    const double* ts_p, const long long ts_sl, const long long ts_sc, const double coef,
    const double cell_tol,
    double* surface_out, double* w_out, double* intake, double* total"""

SPATIAL_HEAD = ('__device__ __forceinline__ void spatial_preferential_cell(\n    const long long i, const int tid,' + _INPUTS + ',\n    double (*sh)[BLOCK], unsigned int* flag_out)\n{\n')

_ROW_NAMES = ("a ? before : 0.0", "a ? after : 0.0", "a ? h_in : 0.0", "a ? left : 0.0", "isolated_v", "attempted_v", "total_v")
_STORE_ROWS = "\n".join(f"    sh[{r}][tid] = {expr};" for r, expr in enumerate(_ROW_NAMES))

SPATIAL_SUBSTITUTIONS = {
    "@@LANE@@": "    if (i >= n) return;\n",
    "@@SATB_L@@": "satb_at(SPM(ts_p, ts_sl, ts_sc, l), SPM(dz_p, dz_sl, dz_sc, l), coef)",
    "@@SAT_L@@": "sat_at(SPM(ts_p, ts_sl, ts_sc, l), SPM(dz_p, dz_sl, dz_sc, l))",
    "@@RATE@@": "rate[i]",
    "@@MAXRATE@@": "max_rate[i * mr_sc]",
    "@@STORE@@": "    surface_out[i] = left;\n    total[i] = total_v;\n" + _STORE_ROWS + "\n    *flag_out = f;",
}

_STEP_ARGS = ", ".join(("n", "nl", "surface_in", "w_in", "active", "dt", "rate", "max_rate", "mr_sc", "recv", "dz_p", "dz_sl", "dz_sc", "ts_p", "ts_sl", "ts_sc", "coef", "cell_tol", "surface_out", "w_out",
                        "intake", "total"))

SPATIAL_TAIL = ('''
extern "C" __global__ void maple_syrup_spatial_pref_step(''' + _INPUTS + r''',
    double* partials, unsigned long long* packet)
{
    __shared__ double sh[7][BLOCK];
    __shared__ unsigned long long shf[BLOCK];
    __shared__ unsigned int shc[BLOCK];
    const int tid = threadIdx.x;
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)tid;
    #pragma unroll
    for (int r = 0; r < 7; ++r) sh[r][tid] = 0.0;  /* neutral values for a padded lane */
    unsigned int f = 0u;
    spatial_preferential_cell(i, tid, ''' + _STEP_ARGS + r''', sh, &f);  /* leaves at once for i >= n */
    shf[tid] = (unsigned long long)f;
    shc[tid] = (f != 0u) ? 1u : 0u;
    __syncthreads();  /* every lane, padded or not, reaches every barrier */
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int r = 0; r < 7; ++r) sh[r][tid] = sh[r][tid] + sh[r][tid + s];
            shf[tid] = shf[tid] | shf[tid + s];
            shc[tid] = shc[tid] + shc[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int r = 0; r < 7; ++r) partials[(long long)r * (long long)gridDim.x + (long long)blockIdx.x] = sh[r][0];
        atomicOr(&packet[0], shf[0]);
        atomicAdd(&packet[1], (unsigned long long)shc[0]);
    }
}

extern "C" __global__ void maple_syrup_spatial_pref_finalize(const double* partials, const long long n_blocks, unsigned long long* packet)
{
    __shared__ double sh[7][@FINAL@];
    const int tid = threadIdx.x;
    for (int r = 0; r < 7; ++r) {
        double acc = 0.0;
        for (long long b = tid; b < n_blocks; b += @FINAL@) acc = acc + partials[(long long)r * n_blocks + b];
        sh[r][tid] = acc;
    }
    __syncthreads();
    for (int s = @FINAL@ / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int r = 0; r < 7; ++r) sh[r][tid] = sh[r][tid] + sh[r][tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) { for (int r = 0; r < 7; ++r) packet[2 + r] = (unsigned long long)__double_as_longlong(sh[r][0]); }
}

extern "C" __global__ void maple_syrup_spatial_pref_reduce(const long long n, const double* a, unsigned long long* diag)
{
    __shared__ unsigned long long s_nf[BLOCK];
    __shared__ unsigned long long s_nan[BLOCK];
    __shared__ unsigned long long s_min[BLOCK];
    __shared__ unsigned long long s_max[BLOCK];
    const int tid = threadIdx.x;
    const long long j = (long long)blockIdx.x * (long long)blockDim.x + (long long)tid;
    unsigned long long nf = 0ULL, nn = 0ULL, kmin = 0xFFFFFFFFFFFFFFFFULL, kmax = 0ULL;  /* neutral values for a padded lane */
    if (j < n) {
        const double v = a[j];
        if (!isfinite(v)) nf = 1ULL;
        if (v != v) nn = 1ULL;
        else { const unsigned long long k = order_key(v); kmin = k; kmax = k; }
    }
    s_nf[tid] = nf; s_nan[tid] = nn; s_min[tid] = kmin; s_max[tid] = kmax;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_nf[tid] = s_nf[tid] + s_nf[tid + s];
            s_nan[tid] = s_nan[tid] + s_nan[tid + s];
            s_min[tid] = (s_min[tid + s] < s_min[tid]) ? s_min[tid + s] : s_min[tid];
            s_max[tid] = (s_max[tid + s] > s_max[tid]) ? s_max[tid + s] : s_max[tid];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&diag[0], s_nf[0]);
        atomicAdd(&diag[1], s_nan[0]);
        atomicMin(&diag[2], s_min[0]);
        atomicMax(&diag[3], s_max[0]);
    }
}

extern "C" __global__ void maple_syrup_spatial_pref_validate(const long long n, const double* a, const double lo, const double hi, unsigned long long* diag)
{
    const long long j = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (j >= n) return;
    const double v = a[j];
    if (!(v >= lo && v <= hi)) atomicMax(&diag[0], 1ULL);  /* NaN fails; +inf passes only when hi is +inf; every cell, inactive included */
}
''').replace("@FINAL@", str(FINAL_THREADS))


def render_source() -> str:
    """The spatial CUDA preferential source: the shared common body (`preferential_infiltration_cuda`) rendered with the spatial readers/head/substitutions and this module's wrapper kernels (pure, stateless)."""
    return pc.render_source(readers=SPATIAL_READERS, head=SPATIAL_HEAD, substitutions=SPATIAL_SUBSTITUTIONS, tail=SPATIAL_TAIL)


def kernel_source() -> str:
    return render_source()


def kernel_provenance() -> dict[str, Any]:
    src = render_source()
    return {"module": "maple_syrup.spatial_preferential_infiltration_cuda", "kernels": list(_KERNELS), "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "common_source_sha256": pc.COMMON_SOURCE_SHA256, "spatial_readers_sha256": hashlib.sha256(SPATIAL_READERS.encode()).hexdigest(), "compile_options": list(COMPILE_OPTIONS),
            "threads_per_block": THREADS, "final_threads": FINAL_THREADS, "raw_packet_bytes": RAW_PACKET_BYTES, "cold_diagnostic_bytes": DIAG_BYTES,
            "reductions": "explicit fused block tree + one-block finalize; integer atomics for the flag OR and flagged count; cold 32-byte key/count diagnostic for the derivation; no CUB/CuPy reduction; no 8N rows array",
            "fastmath": False}


_LOCK = threading.Lock()
_FUNCTIONS: dict[int, SimpleNamespace] = {}  # device id -> compiled functions only (never an owner, state, map or array)


def load_kernels(cp, device_id: int) -> SimpleNamespace:
    """Compile/load the spatial preferential module once per process and device (preparation, outside every hot call; no physical dummy launch or copy). Failure is a typed setup error, never a fallback."""
    with _LOCK:
        fns = _FUNCTIONS.get(device_id)
        if fns is not None:
            return fns
        try:
            module = cp.RawModule(code=render_source(), options=COMPILE_OPTIONS, backend="nvrtc")
            got = {name: module.get_function(name) for name in _KERNELS}
            for fn in got.values():
                if int(dict(fn.attributes).get("max_threads_per_block", 0)) < THREADS:
                    raise ss.SubsurfaceBackendUnavailableError(f"a spatial preferential kernel supports fewer than {THREADS} threads per block")
        except ss.SubsurfaceBackendUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ss.SubsurfaceBackendUnavailableError(f"the CUDA spatial preferential kernels failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        ns = SimpleNamespace(step=got[STEP_KERNEL], finalize=got[FINAL_KERNEL], reduce=got[REDUCE_KERNEL], validate=got[VALIDATE_KERNEL])
        _FUNCTIONS[device_id] = ns
        return ns


def n_blocks(n_cells: int) -> int:
    return (int(n_cells) + THREADS - 1) // THREADS


def check_launch_ranges(n_layers: int, n_cells: int) -> None:
    """Integer launch widths are validated BEFORE any cast or allocation (`nl` is a 32-bit kernel argument; the grid is limited to 2^31 - 1 blocks)."""
    if not (1 <= n_layers <= 2**31 - 1):
        raise ss.SubsurfaceError(f"n_layers = {n_layers} does not fit the 32-bit kernel argument")
    if not (1 <= n_cells and n_blocks(n_cells) <= MAX_GRID):
        raise ss.SubsurfaceError(f"n_cells = {n_cells} exceeds the CUDA launch grid limit of {MAX_GRID} blocks of {THREADS} threads")


def operation_terms(operation: str, n_layers: int, n_cells: int) -> dict[str, int]:
    """Exact byte terms of one CUDA operation (payload model; allocator rounding, NVRTC, the context and the interpreter are excluded). `outputs` survive the call (for `prepare`: the retained sealed device arrays,
    an upper bound with a per-cell max rate), `scratch` is released after the call, `host` is the call's own host allocation, `shared` the static shared memory per block (not global).
      step     outputs `(2 N + 2 L N) 8` (surface, total, water, intake); scratch partials `7 * 8 * nb` + raw device packet 72; host raw 72 + rebuilt 72 = 144;
               shared (step kernel) `7 * 8 * 128 + 8 * 128 + 4 * 128`, (finalize) `7 * 8 * 256`
      prepare  outputs `(L N + 2 N) 8` (rate, receiver, max rate); the DERIVATION live set adds scratch `4 N 8` (cover, beta, denominator, layer sum; temp/numerator become rate/receiver) + masks `3 N` (inactive,
               positive, ~positive) + the cold diagnostic/seed `32` + the scalar a3/max-rate device arrays `8 + 8` + `8` slack = `(L N + 6 N) 8 + 3 N + 64` in total; host the cold buffers `64`
               (seed + result of the reducer) + the hash staging buffer `min(L N, STAGE_ELEMS) 8` (the weight records are added by the caller: they depend on the species count)"""
    L, N = int(n_layers), int(n_cells)
    nb, f8 = n_blocks(N), 8
    if operation == "step":
        t = dict(outputs=(2 * N + 2 * L * N) * f8, scratch=ROWS * f8 * nb + RAW_PACKET_BYTES, host=2 * RAW_PACKET_BYTES, shared=ROWS * f8 * THREADS + f8 * THREADS + 4 * THREADS, shared_finalize=ROWS * f8 * FINAL_THREADS)
    elif operation == "prepare":
        t = dict(outputs=(L * N + 2 * N) * f8, scratch=4 * N * f8 + 3 * N + 64, host=2 * DIAG_BYTES + min(L * N, STAGE_ELEMS) * f8, shared=0, shared_finalize=0)
    else:
        raise ss.SubsurfaceError(f"unknown CUDA operation {operation!r}")
    t["n_blocks"] = nb
    return t


_STRIDES = {"scalar": lambda n: (0, 0), "shared": lambda n: (1, 0), "cell": lambda n: (n, 1)}


def _map_args(column) -> tuple:
    n = column.n_cells
    kinds = {name: v["kind"] for name, v in column.estimate.retained.items()}
    out = []
    for name in ("dz", "theta_saturation"):
        sl, si = _STRIDES[kinds[name]](int(n))
        out += [column._maps[name], np.int64(sl), np.int64(si)]
    return tuple(out)


def _check_device_array(cp: Any, array: Any, what: str, *, device_id: int | None = None) -> None:
    if type(array) is not cp.ndarray or array.dtype != np.float64 or array.ndim != 1 or not array.flags.c_contiguous:
        raise sp.SpatialLayerError(f"{what} must be an exact one-dimensional C-contiguous float64 cupy.ndarray")
    if device_id is not None and int(array.device.id) != device_id:
        raise sp.SpatialLayerError(f"{what} lives on device {int(array.device.id)} but the owner is bound to device {device_id}; no implicit transfer")


def validate_range(cp, array: Any, lo: float, hi: float) -> bool:
    """True when EVERY element of the device array lies in `[lo, hi]` (NaN fails; `+inf` passes only for `hi = +inf`): one tiny kernel, one 8-byte diagnostic read, no temporary array."""
    from maple.core.backend import to_host

    device_id = int(array.device.id)
    _check_device_array(cp, array, "the validated array", device_id=device_id)
    fns = load_kernels(cp, device_id)
    diag = cp.zeros(1, dtype=np.uint64)
    fns.validate((n_blocks(int(array.shape[0])),), (THREADS,), (np.int64(array.shape[0]), array, np.float64(lo), np.float64(hi), diag))
    words = np.asarray(to_host(diag), dtype=np.uint64)
    return not bool(int(words[0]))


def _decode_key(key: int) -> float:
    bits = (key & ~_SIGN) if key & _SIGN else (~key & _MAX_KEY)
    return float(np.array([bits], dtype=np.uint64).view(np.float64)[0])


class ColdReducer:
    """The bounded cold reducer handed to `_DeriveContext.reduce`. One call = one kernel launch over one device array, one 32-byte seed copy (COLD H2D) and one 32-byte synchronising read (COLD D2H); the seed and
    result host buffers (64 B) are the only host allocations. It never calls the library `sum/min/max`. Counters (`calls`, `h2d_bytes`, `d2h_bytes`) make the cold traffic auditable."""

    def __init__(self, cp: Any, device_id: int) -> None:
        self.cp, self.device_id = cp, int(device_id)
        self.calls, self.h2d_bytes, self.d2h_bytes = 0, 0, 0

    def __call__(self, kind: str, array: Any) -> float:
        from maple.core.backend import to_host

        if kind not in pi.REDUCE_KINDS:
            raise pi.PreferentialError(f"unknown cold reduction kind {kind!r}; expected one of {pi.REDUCE_KINDS}")
        cp = self.cp
        _check_device_array(cp, array, f"the array of the cold {kind} reduction", device_id=self.device_id)
        n = int(array.shape[0])
        fns = load_kernels(cp, self.device_id)
        seed = np.array([0, 0, _MAX_KEY, 0], dtype=np.uint64)  # seed host buffer (32 B)
        diag = cp.array(seed)  # COLD H2D, 32 B
        fns.reduce((n_blocks(n),), (THREADS,), (np.int64(n), array, diag))
        words = np.asarray(to_host(diag), dtype=np.uint64)  # COLD synchronising D2H, 32 B (result host buffer)
        self.calls += 1
        self.h2d_bytes += int(seed.nbytes)
        self.d2h_bytes += int(words.nbytes)
        nonfinite, nan, kmin, kmax = (int(w) for w in words)
        if kind == "count_nonfinite":
            return float(nonfinite)
        if kind == "count_nan":
            return float(nan)
        if nan:  # xp.min / xp.max propagate NaN
            return float("nan")
        return _decode_key(kmin if kind == "min" else kmax)


def hash_device_array(h: Any, name: str, kind: str, array: Any, host_buf: np.ndarray) -> None:
    """The cold identity recipe of the CPU operator for ANY rank: the header `name|kind|shape|` carries the ORIGINAL shape (`(N,)` or `(L, N)`), followed by the flat C-contiguous float64 bytes and a NUL, streamed
    from the device through the counted bounded host staging buffer `host_buf` (the FLAT view is sliced, so a 2-D array is handled; no host mirror)."""
    h.update(f"{name}|{kind}|{tuple(int(s) for s in array.shape)}|".encode())
    flat = array.reshape(-1)
    n = int(flat.shape[0])
    step_ = max(1, int(host_buf.size))
    for a in range(0, n, step_):
        piece = flat[a:a + step_]
        m = int(piece.shape[0])
        piece.get(out=host_buf[:m])
        h.update(memoryview(host_buf[:m]).cast("B"))
    h.update(b"\0")


def build_packet(raw: np.ndarray) -> np.ndarray:
    """The unchanged 9-float common packet (`preferential_infiltration.PACKET_LENGTH`) from the 9-word raw packet: seven sums, the LOWEST set bit of the OR (the minimum per-cell lowest bit, or `NO_FLAG`), the count."""
    raw = np.asarray(raw)
    if raw.dtype != np.uint64 or raw.shape != (PACKET_WORDS,):
        raise pi.PreferentialError(f"internal: raw packet {raw.dtype} {raw.shape}")
    flags, count = int(raw[0]), int(raw[1])
    out = np.empty(pi.PACKET_LENGTH, dtype=np.float64)
    out[:7] = np.ascontiguousarray(raw[2:]).view(np.float64)
    out[7] = float(flags & -flags) if flags else pi.NO_FLAG
    out[8] = float(count)
    return out


def step(cp, op, surface, water, dt: float):
    """One spatial CUDA preferential step (checks and admission were done by the caller). Returns `(surface_out, water_out, intake, total, packet)` with fresh device outputs and the host 9-float packet."""
    from maple.core.backend import to_host

    column = op.column
    L, N = int(op.n_layers), int(op.n_cells)
    check_launch_ranges(L, N)
    fns = load_kernels(cp, column.device_id)
    surface_out, total = cp.empty(N, dtype=np.float64), cp.empty(N, dtype=np.float64)
    water_out, intake = cp.empty((L, N), dtype=np.float64), cp.empty((L, N), dtype=np.float64)
    nb = n_blocks(N)
    partials = cp.empty(ROWS * nb, dtype=np.float64)
    packet_dev = cp.zeros(PACKET_WORDS, dtype=np.uint64)  # words 0-1 are the atomic targets: zero-initialised on the device
    rate, max_rate, recv = op._law
    mr_sc = 0 if op.layouts["max_intake_rate_m_s"] == "scalar" else 1
    coef = float(volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0))
    fns.step((nb,), (THREADS,), (np.int64(N), np.int32(L), surface, water, column._mask, np.float64(dt), rate, max_rate, np.int64(mr_sc), recv, *_map_args(column), np.float64(coef),
                                  np.float64(pi.cell_closure_coefficient(L)), surface_out, water_out, intake, total, partials, packet_dev))
    fns.finalize((1,), (FINAL_THREADS,), (partials, np.int64(nb), packet_dev))
    raw = np.asarray(to_host(packet_dev), dtype=np.uint64)  # the ONE counted constant-size read; synchronises the stream: partials are released only afterwards
    return surface_out, water_out, intake, total, build_packet(raw)
