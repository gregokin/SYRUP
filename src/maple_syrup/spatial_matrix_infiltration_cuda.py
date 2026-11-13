"""CuPy RawKernel CUDA step of the fixed explicit SPATIAL persistent Green-Ampt matrix infiltration (`spatial_matrix_infiltration`, backend "cuda"). Trial component only (S2-B2.c).

ONE physical body: the matrix kernel body of `matrix_infiltration_cuda` (and its Newton helpers `log1pmx` / `ga_increment`, written once and never edited here) rendered by the pure, fail-closed
`matrix_infiltration_cuda.render_source` with SPATIAL readers, a `__device__` head and a reduction tail. The body becomes the `__device__` function `spatial_matrix_cell` (its `return` for `i >= n` is the padded-lane
guard); the wrapper kernel then lets EVERY lane (padded or inactive) reach every barrier. Readers are the A1 strided raw maps `p[l * stride_layer + i * stride_cell]` (scalar 0/0, shared 1/0, cell N/1):

    FC0   = fc_at(theta_fc, dz)          fc  = theta_fc * dz      (a rounded product, formed BEFORE the deficit: b = psi * (max(fc - W0, 0) / dz0))
    SAT0  = sat_at(theta_sat, dz)        sat = theta_sat * dz
    SATB  = satb_at(theta_sat, dz, coef) sat + coef * sat         (the validity bound `sat_m[l] + cap_tol[l]`, cap_tol = coef * sat)
    K / PSI / G                          `p[i * stride]` of the sealed law maps (a one-element array with stride 0 for a scalar input)

NO FUSION: the products must be rounded before the next operation exactly as on the CPU; the module is compiled with the existing `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`. No `(L, N)` bound
tensor, no dense `16 N` rows array, no L-sized per-thread array, no per-thread contribution array, no host mirror, no CuPy/CUB reduction or workspace.

Launches on the current stream, in order: `maple_syrup_spatial_matrix_step` (per lane the fifteen per-cell contributions are stored by the shared body straight into the block's static shared rows with literal
offsets: rows 0-13 are the sums of the common packet (soil before/after, pool in/out, candidate, intake, pre, post, rejected, unused, J before/after, onsets, ponded), row 14 the maximum Newton iteration count; padded
and inactive lanes hold 0; the flag word is OR-ed and the invalid cells counted in two more shared words; then ONE fixed block tree in `15 * 8 * THREADS + 8 * THREADS + 4 * THREADS` B of static shared memory, `15 * nb`
block partials and, per block, one `atomicOr` of the flag word and one `atomicAdd` of the flagged count into the raw packet) -> `maple_syrup_spatial_matrix_finalize` (one block of `FINAL_THREADS`, a fixed
tree over the partials, `15 * 8 * FINAL_THREADS` B of static shared memory; the sums are written as 64-bit bit patterns) -> ONE `maple.core.backend.to_host` of the 17-word `uint64` raw packet (136 B: word 0 the
OR of the flag words, word 1 the number of flagged cells, words 2-16 the fifteen float64 bit patterns). The host rebuilds the unchanged 17-float common packet for `matrix_infiltration._resolve_core`: sums, the
maximum iteration count, the LOWEST set bit of the OR (equal to the minimum over cells of each cell's lowest bit, or `NO_FLAG` when no bit is set) and the flagged-cell count (each invalid cell counted once, inactive
water included). Totals are summed in a fixed tree order (roundoff-level different from NumPy/Numba); the existing closure bound still applies. Both host buffers (raw 136 B and reconstructed 136 B) are counted.

Memory/lifetimes (`operation_terms`): scratch (partials, raw packet) are Python locals held until after the synchronising read, then released to the CuPy pool; outputs are fresh pool arrays; static shared memory is
reported (not a global allocation). Excluded and declared: NVRTC/module load, the CUDA context, allocator pool policy and rounding, interpreter. Cold validation of the law/history arrays uses one tiny validation
kernel and one 8-byte diagnostic word. Written by its author without a GPU; Root qualifies it on hardware.
"""

from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import matrix_infiltration as mi
from maple_syrup import matrix_infiltration_cuda as mc
from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss

__all__ = ["COMPILE_OPTIONS", "FINAL_THREADS", "PACKET_WORDS", "RAW_PACKET_BYTES", "ROWS", "THREADS", "build_packet", "check_launch_ranges", "hash_device_array", "kernel_provenance", "kernel_source",
           "load_kernels", "operation_terms", "render_source", "step", "validate_range"]

COMPILE_OPTIONS = mc.COMPILE_OPTIONS  # --fmad=false --prec-div=true --prec-sqrt=true --ftz=false (the common matrix kernel's own options)
THREADS = 128
FINAL_THREADS = 256
ROWS = 15  # 14 sums + the maximum Newton iteration count
PACKET_WORDS = 17  # word 0: OR of the flag words, word 1: flagged cells, words 2..16: the 15 float64 bit patterns
RAW_PACKET_BYTES = 8 * PACKET_WORDS  # 136
DIAG_WORDS = 1
MAX_GRID = 2**31 - 1
STAGE_ELEMS = 16384  # float64 elements of the host staging buffer of the cold identity hash (128 KiB)
STEP_KERNEL = "maple_syrup_spatial_matrix_step"
FINAL_KERNEL = "maple_syrup_spatial_matrix_finalize"
VALIDATE_KERNEL = "maple_syrup_spatial_matrix_validate"
_KERNELS = (STEP_KERNEL, FINAL_KERNEL, VALIDATE_KERNEL)

SPATIAL_READERS = r"""
#define BLOCK 128
#define SPM(p, sl, sc, l) (p)[(long long)(l) * (sl) + i * (sc)]
__device__ __forceinline__ double fc_at(const double theta_fc, const double dz)
{
    const double fc = theta_fc * dz;  /* a rounded product: --fmad=false keeps it unfused */
    return fc;
}
__device__ __forceinline__ double sat_at(const double theta_sat, const double dz)
{
    const double sat = theta_sat * dz;  /* sat = theta_sat * dz (rounded) */
    return sat;
}
__device__ __forceinline__ double satb_at(const double theta_sat, const double dz, const double coef)
{
    const double sat = theta_sat * dz;  /* sat = theta_sat * dz (rounded) */
    return sat + coef * sat;            /* the validity bound sat + cap_tol, cap_tol = coef * sat */
}
"""

_MAPS = r"""
    const double* dz_p, const long long dz_sl, const long long dz_sc,
    const double* tf_p, const long long tf_sl, const long long tf_sc,
    const double* ts_p, const long long ts_sl, const long long ts_sc, const double coef,
    const double* k_p, const long long k_sc, const double* psi_p, const long long psi_sc, const double* g_p, const long long g_sc"""

_INPUTS = r"""
    const long long n, const int nl,
    const double* pool_in, const double* standing_in, const double* rate_in, const double* w_in,
    const double* wet_in, const double* cum_in, const unsigned char* active, const double dt,""" + _MAPS + r""",
    const double cell_tol, const double tol, const int max_iter,
    double* pool_out, double* w_out, double* wet_out, double* cum_out, double* intake_out"""

SPATIAL_HEAD = ('__device__ __forceinline__ void spatial_matrix_cell(\n    const long long i, const int tid,' + _INPUTS + ',\n    double (*sh)[BLOCK], unsigned int* flag_out)\n{\n')

_ROW_NAMES = ("before", "after", "pool", "pool_o", "candidate", "intake", "pre", "post", "rejected", "unused", "jin", "j_o", "onset", "ponded", "iters")
_STORE_ACTIVE = "\n".join(f"        sh[{r}][tid] = {name};" for r, name in enumerate(_ROW_NAMES))
_STORE_INACTIVE = "\n".join(f"        sh[{r}][tid] = 0.0;" for r in range(len(_ROW_NAMES)))

SPATIAL_SUBSTITUTIONS = {
    "@@LANE@@": "    if (i >= n) return;\n",
    "@@SATB_L@@": "satb_at(SPM(ts_p, ts_sl, ts_sc, l), SPM(dz_p, dz_sl, dz_sc, l), coef)",
    "@@FC0@@": "fc_at(SPM(tf_p, tf_sl, tf_sc, 0), SPM(dz_p, dz_sl, dz_sc, 0))",
    "@@DZ0@@": "SPM(dz_p, dz_sl, dz_sc, 0)",
    "@@SAT0@@": "sat_at(SPM(ts_p, ts_sl, ts_sc, 0), SPM(dz_p, dz_sl, dz_sc, 0))",
    "@@K@@": "k_p[i * k_sc]",
    "@@PSI@@": "psi_p[i * psi_sc]",
    "@@G@@": "g_p[i * g_sc]",
    "@@STORE@@": ("    if (a) {\n        intake_out[i] = intake;\n" + _STORE_ACTIVE + "\n    } else {\n        intake_out[i] = 0.0;\n" + _STORE_INACTIVE + "\n    }\n    *flag_out = f;"),
}

_STEP_ARGS = ", ".join(("n", "nl", "pool_in", "standing_in", "rate_in", "w_in", "wet_in", "cum_in", "active", "dt", "dz_p", "dz_sl", "dz_sc", "tf_p", "tf_sl", "tf_sc", "ts_p", "ts_sl", "ts_sc", "coef",
                        "k_p", "k_sc", "psi_p", "psi_sc", "g_p", "g_sc", "cell_tol", "tol", "max_iter", "pool_out", "w_out", "wet_out", "cum_out", "intake_out"))

SPATIAL_TAIL = ('''
extern "C" __global__ void maple_syrup_spatial_matrix_step(''' + _INPUTS + r''',
    double* partials, unsigned long long* packet)
{
    __shared__ double sh[15][BLOCK];
    __shared__ unsigned long long shf[BLOCK];
    __shared__ unsigned int shc[BLOCK];
    const int tid = threadIdx.x;
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)tid;
    #pragma unroll
    for (int r = 0; r < 15; ++r) sh[r][tid] = 0.0;  /* neutral values for a padded lane: sums 0, max iterations 0 */
    unsigned int f = 0u;
    spatial_matrix_cell(i, tid, ''' + _STEP_ARGS + r''', sh, &f);  /* leaves at once for i >= n */
    shf[tid] = (unsigned long long)f;
    shc[tid] = (f != 0u) ? 1u : 0u;
    __syncthreads();  /* every lane, padded or not, reaches every barrier */
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int r = 0; r < 14; ++r) sh[r][tid] = sh[r][tid] + sh[r][tid + s];
            sh[14][tid] = (sh[14][tid + s] > sh[14][tid]) ? sh[14][tid + s] : sh[14][tid];
            shf[tid] = shf[tid] | shf[tid + s];
            shc[tid] = shc[tid] + shc[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int r = 0; r < 15; ++r) partials[(long long)r * (long long)gridDim.x + (long long)blockIdx.x] = sh[r][0];
        atomicOr(&packet[0], shf[0]);
        atomicAdd(&packet[1], (unsigned long long)shc[0]);
    }
}

extern "C" __global__ void maple_syrup_spatial_matrix_finalize(const double* partials, const long long n_blocks, unsigned long long* packet)
{
    __shared__ double sh[15][@FINAL@];
    const int tid = threadIdx.x;
    for (int r = 0; r < 15; ++r) {
        double acc = 0.0;
        for (long long b = tid; b < n_blocks; b += @FINAL@) {
            const double v = partials[(long long)r * n_blocks + b];
            if (r < 14) acc = acc + v;
            else if (v > acc) acc = v;
        }
        sh[r][tid] = acc;
    }
    __syncthreads();
    for (int s = @FINAL@ / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int r = 0; r < 14; ++r) sh[r][tid] = sh[r][tid] + sh[r][tid + s];
            sh[14][tid] = (sh[14][tid + s] > sh[14][tid]) ? sh[14][tid + s] : sh[14][tid];
        }
        __syncthreads();
    }
    if (tid == 0) { for (int r = 0; r < 15; ++r) packet[2 + r] = (unsigned long long)__double_as_longlong(sh[r][0]); }
}

extern "C" __global__ void maple_syrup_spatial_matrix_validate(const long long n, const double* a, const double lo, const double hi, unsigned long long* diag)
{
    const long long j = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (j >= n) return;
    const double v = a[j];
    if (!(isfinite(v) && v >= lo && v <= hi)) atomicMax(&diag[0], 1ULL);  /* every cell, inactive included */
}
''').replace("@FINAL@", str(FINAL_THREADS))


def render_source() -> str:
    """The spatial CUDA matrix source: the shared common Newton helpers and body (`matrix_infiltration_cuda`) rendered with the spatial readers/head/substitutions and this module's wrapper kernels (pure, stateless)."""
    return mc.render_source(readers=SPATIAL_READERS, head=SPATIAL_HEAD, substitutions=SPATIAL_SUBSTITUTIONS, tail=SPATIAL_TAIL)


def kernel_source() -> str:
    return render_source()


def kernel_provenance() -> dict[str, Any]:
    src = render_source()
    return {"module": "maple_syrup.spatial_matrix_infiltration_cuda", "kernels": list(_KERNELS), "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "common_source_sha256": mc.COMMON_SOURCE_SHA256, "spatial_readers_sha256": hashlib.sha256(SPATIAL_READERS.encode()).hexdigest(), "compile_options": list(COMPILE_OPTIONS),
            "threads_per_block": THREADS, "final_threads": FINAL_THREADS, "raw_packet_bytes": RAW_PACKET_BYTES,
            "reductions": "explicit fused block tree + one-block finalize; integer atomics for the flag OR and flagged count; no CUB/CuPy reduction; no 16N rows array; no per-thread contribution array", "fastmath": False}


_LOCK = threading.Lock()
_FUNCTIONS: dict[int, SimpleNamespace] = {}  # device id -> compiled functions only (never an owner, state, history or array)


def load_kernels(cp, device_id: int) -> SimpleNamespace:
    """Compile/load the spatial matrix module once per process and device (preparation, outside every hot call; no physical dummy launch or copy). Failure is a typed setup error, never a fallback."""
    with _LOCK:
        fns = _FUNCTIONS.get(device_id)
        if fns is not None:
            return fns
        try:
            module = cp.RawModule(code=render_source(), options=COMPILE_OPTIONS, backend="nvrtc")
            got = {name: module.get_function(name) for name in _KERNELS}
            for fn in got.values():
                if int(dict(fn.attributes).get("max_threads_per_block", 0)) < THREADS:
                    raise ss.SubsurfaceBackendUnavailableError(f"a spatial matrix kernel supports fewer than {THREADS} threads per block")
        except ss.SubsurfaceBackendUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ss.SubsurfaceBackendUnavailableError(f"the CUDA spatial matrix kernels failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        ns = SimpleNamespace(step=got[STEP_KERNEL], finalize=got[FINAL_KERNEL], validate=got[VALIDATE_KERNEL])
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
    """Exact byte terms of one CUDA operation (payload model; allocator rounding, NVRTC, the context and the interpreter are excluded). `outputs` survive the call, `scratch` is released after the synchronising
    read, `host` is the call's own host allocation, `shared` the static shared memory per block (not global).
      step      outputs `(4 N + L N) 8` (pool, J, ledger, intake, water); scratch partials `15 * 8 * nb` + raw device packet 136; host raw packet 136 + reconstructed 17-float packet 136 = 272;
                shared (step kernel) `15 * 8 * 128 + 8 * 128 + 4 * 128` and (finalize kernel) `15 * 8 * 256`
      history   outputs `2 N 8` (the two arrays) + the 8-byte validation diagnostic (device); host: the 8-byte diagnostic read
      prepare   outputs `3 N 8` (UPPER BOUND on the sealed law arrays; a scalar retains 8 B) + the 8-byte diagnostic; host: the staging buffer of the cold identity hash `min(N, STAGE_ELEMS) 8`"""
    L, N = int(n_layers), int(n_cells)
    nb, f8 = n_blocks(N), 8
    if operation == "step":
        t = dict(outputs=(4 * N + L * N) * f8, scratch=ROWS * f8 * nb + RAW_PACKET_BYTES, host=2 * RAW_PACKET_BYTES, shared=ROWS * f8 * THREADS + f8 * THREADS + 4 * THREADS, shared_finalize=ROWS * f8 * FINAL_THREADS)
    elif operation == "history":
        t = dict(outputs=2 * N * f8, scratch=f8 * DIAG_WORDS, host=f8 * DIAG_WORDS, shared=0, shared_finalize=0)
    elif operation == "prepare":
        t = dict(outputs=3 * N * f8, scratch=f8 * DIAG_WORDS, host=min(N, STAGE_ELEMS) * f8 + f8 * DIAG_WORDS, shared=0, shared_finalize=0)
    else:
        raise ss.SubsurfaceError(f"unknown CUDA operation {operation!r}")
    t["n_blocks"] = nb
    return t


_STRIDES = {"scalar": lambda n: (0, 0), "shared": lambda n: (1, 0), "cell": lambda n: (n, 1)}


def _map_args(column) -> tuple:
    n = column.n_cells
    kinds = {name: v["kind"] for name, v in column.estimate.retained.items()}
    out = []
    for name in ("dz", "theta_field_capacity", "theta_saturation"):
        sl, si = _STRIDES[kinds[name]](int(n))
        out += [column._maps[name], np.int64(sl), np.int64(si)]
    return tuple(out)


def validate_range(cp, array, lo: float, hi: float) -> bool:
    """True when EVERY element of the device array is finite and within `[lo, hi]` (one tiny kernel, one 8-byte diagnostic read; no temporary array)."""
    from maple.core.backend import to_host

    n = int(array.shape[0])
    fns = load_kernels(cp, int(array.device.id))
    diag = cp.zeros(DIAG_WORDS, dtype=np.uint64)
    fns.validate((n_blocks(n),), (THREADS,), (np.int64(n), array, np.float64(lo), np.float64(hi), diag))
    words = np.asarray(to_host(diag), dtype=np.uint64)
    return not bool(int(words[0]))


def hash_device_array(h: Any, name: str, kind: str, array: Any, host_buf: np.ndarray) -> None:
    """The cold identity recipe of the CPU operator (`name|kind|shape|` + raw float64 bytes + NUL) streamed from a device array through the counted host staging buffer `host_buf` (bounded; no full mirror)."""
    h.update(f"{name}|{kind}|{tuple(int(s) for s in array.shape)}|".encode())
    n = int(array.shape[0])
    step_ = max(1, int(host_buf.size))
    for a in range(0, n, step_):
        piece = array[a:a + step_]
        m = int(piece.shape[0])
        piece.get(out=host_buf[:m])
        h.update(memoryview(host_buf[:m]).cast("B"))
    h.update(b"\0")


def build_packet(raw: np.ndarray) -> np.ndarray:
    """The unchanged 17-float common packet (`matrix_infiltration.PACKET_LENGTH`) from the 17-word raw packet: 14 sums, max iterations, the LOWEST set bit of the OR (as the minimum per-cell lowest bit, or
    `NO_FLAG`), the flagged-cell count."""
    raw = np.asarray(raw)
    if raw.dtype != np.uint64 or raw.shape != (PACKET_WORDS,):
        raise mi.MatrixError(f"internal: raw packet {raw.dtype} {raw.shape}")
    flags, count = int(raw[0]), int(raw[1])
    out = np.empty(mi.PACKET_LENGTH, dtype=np.float64)
    out[:15] = np.ascontiguousarray(raw[2:]).view(np.float64)
    out[15] = float(flags & -flags) if flags else mi.NO_FLAG
    out[16] = float(count)
    return out


def step(cp, op, pool, standing, rate, water, wet, cum, dt: float):
    """One spatial CUDA matrix step (checks and admission were done by the caller). Returns `(pool_out, water_out, wet_out, cum_out, intake, packet)` with fresh device outputs and the host 17-float packet."""
    from maple.core.backend import to_host

    column = op.column
    L, N = int(op.n_layers), int(op.n_cells)
    check_launch_ranges(L, N)
    fns = load_kernels(cp, column.device_id)
    pool_out, wet_out, cum_out, intake = (cp.empty(N, dtype=np.float64) for _ in range(4))
    water_out = cp.empty((L, N), dtype=np.float64)
    nb = n_blocks(N)
    partials = cp.empty(ROWS * nb, dtype=np.float64)
    packet_dev = cp.zeros(PACKET_WORDS, dtype=np.uint64)  # words 0-1 are the atomic targets: zero-initialised on the device
    k, psi, g = op._law
    strides = tuple(0 if op.layouts[name] == "scalar" else 1 for name in ("conductivity_m_s", "suction_m", "vegetation_factor"))
    fns.step((nb,), (THREADS,), (np.int64(N), np.int32(L), pool, standing, rate, water, wet, cum, column._mask, np.float64(dt), *_map_args(column), np.float64(ss_coef()),
                                  k, np.int64(strides[0]), psi, np.int64(strides[1]), g, np.int64(strides[2]), np.float64(mi.cell_closure_coefficient(L)), np.float64(mi.newton_tolerance()),
                                  np.int32(mi.NEWTON_MAX_ITERATIONS), pool_out, water_out, wet_out, cum_out, intake, partials, packet_dev))
    fns.finalize((1,), (FINAL_THREADS,), (partials, np.int64(nb), packet_dev))
    raw = np.asarray(to_host(packet_dev), dtype=np.uint64)  # the ONE counted constant-size read; synchronises the stream: partials are released only afterwards
    return pool_out, water_out, wet_out, cum_out, intake, build_packet(raw)


def ss_coef() -> float:
    from maple_syrup.conservation import volume_roundoff_bound_m3

    return float(volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0))
