"""CuPy RawKernel CUDA withdrawal and availability of a fixed explicit spatial column (`spatial_transpiration`, backend "cuda"). Trial component only (S2-B1.c).

ONE physical withdrawal body: `subsurface_transpiration._CUDA_BODY` (the common kernel's body, byte for byte) rendered with the pure `render_withdrawal_source` using SPATIAL readers and a reduction tail. The body becomes the
`__device__` function `spatial_withdraw_cell` (its early `return` is the padded-lane guard); the wrapper kernel then lets EVERY lane (padded or inactive) reach the block reduction. Readers are the A1 strided raw maps
`p[l * stride_layer + j * stride_cell]` (scalar 0/0, shared 1/0, cell N/1):

    WITHDRAW_RESIDUAL(l) = res_at(theta_res, dz)            res = theta_res * dz                      (a rounded product)
    WITHDRAW_SATB(l)     = satb_at(theta_sat, dz, coef)     sat = theta_sat * dz;  satb = sat + coef * sat

NO FUSION. These products must be rounded before the next operation exactly as on the CPU (`res = tr*dz` then `wv - res`; `sat = ts*dz` then `sat + coef*sat`). A contracted multiply-add (FMA) would change the bits, so this
module is compiled with `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false` (the A4 spatial options); the COMMON withdrawal kernel keeps its own `-std=c++11` options and precomputed bounds and is not
touched. No `(L, N)` bound tensor, no 8N rows array, no host mirror, no CuPy/CUB reduction.

Launches on the current stream, in order: `maple_syrup_spatial_withdraw` (per lane: the eight per-cell quantities of the common body in registers -- five sums gated by the active mask, the minimum water after, the flag
word as a double, the request maximum; neutral values for padded lanes: 0, 0, 0, 0, 0, +inf, 0, 0 -- then a fixed block tree in static shared memory `8 * 8 * THREADS` B and `8 * nb` block partials) ->
`maple_syrup_spatial_withdraw_finalize` (one block, fixed tree; sum for rows 0-4, min for row 5, max for rows 6-7) -> ONE `maple.core.backend.to_host` of the 8-double packet (64 B) in the SAME order as the common packet
(`before, after, actual, unmet, requested, min water after, max flag word, max request`); no raw/reconstructed pair exists. Totals are summed in a fixed tree order (roundoff-level different from NumPy and CuPy);
the existing closure bound still applies. `maple_syrup_spatial_availability`: one thread per cell, `max(W - res, 0)` on active cells (0 on inactive), validity flagged for EVERY cell, a one-word diagnostic.

Memory/lifetimes (`operation_terms`): scratch (partials, packet, diagnostic) are Python locals held until after the synchronising read, then released to the CuPy pool; outputs are fresh pool arrays; static shared memory is reported
(not a global allocation). Excluded and declared: NVRTC/module load, the CUDA context, allocator pool policy and rounding, interpreter. Written by its author without a GPU; Root qualifies it on hardware.
"""

from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_cuda as sc
from maple_syrup import subsurface_transpiration as st

__all__ = ["COMPILE_OPTIONS", "FINAL_THREADS", "PACKET_BYTES", "ROWS", "THREADS", "availability", "kernel_provenance", "kernel_source", "load_kernels", "operation_terms", "render_source", "withdraw"]

COMPILE_OPTIONS = sc.COMPILE_OPTIONS  # --fmad=false --prec-div=true --prec-sqrt=true --ftz=false (the common withdrawal kernel keeps its own `-std=c++11`)
THREADS = 128
FINAL_THREADS = 256
ROWS = st._N_ROWS  # the eight per-cell quantities of the common body
PACKET_BYTES = 8 * ROWS  # ONE host packet: 8 doubles
AVAIL_DIAG_WORDS = 1
MAX_GRID = 2**31 - 1
WITHDRAW_KERNEL = "maple_syrup_spatial_withdraw"
FINAL_KERNEL = "maple_syrup_spatial_withdraw_finalize"
AVAIL_KERNEL = "maple_syrup_spatial_availability"
_KERNELS = (WITHDRAW_KERNEL, FINAL_KERNEL, AVAIL_KERNEL)

SPATIAL_READERS = r"""
#define BLOCK 128
#define SP(p, sl, sc, l) (p)[(long long)(l) * (sl) + j * (sc)]
__device__ __forceinline__ double res_at(const double theta_res, const double dz)
{
    const double res = theta_res * dz;  /* a rounded product: --fmad=false keeps it unfused */
    return res;
}
__device__ __forceinline__ double satb_at(const double theta_sat, const double dz, const double coef)
{
    const double sat = theta_sat * dz;  /* sat = theta_sat * dz (rounded) */
    return sat + coef * sat;            /* satb = sat + coef * sat */
}
#define WITHDRAW_GUARD if (j >= N) return;
#define WITHDRAW_RESIDUAL(l) res_at(SP(tr_p, tr_sl, tr_sc, l), SP(dz_p, dz_sl, dz_sc, l))
#define WITHDRAW_SATB(l) satb_at(SP(ts_p, ts_sl, ts_sc, l), SP(dz_p, dz_sl, dz_sc, l), coef)
#define WITHDRAW_STORE(sw_, sa_, st_, su_, sq_, mn_, flags_, qm_) \
    contrib[0] = sw_; contrib[1] = sa_; contrib[2] = st_; contrib[3] = su_; contrib[4] = sq_; \
    contrib[5] = mn_; contrib[6] = (double)flags_; contrib[7] = qm_;
"""

_MAP_PARAMS = r"""
    const double* dz_p, const long long dz_sl, const long long dz_sc,
    const double* tr_p, const long long tr_sl, const long long tr_sc,
    const double* ts_p, const long long ts_sl, const long long ts_sc, const double coef"""

SPATIAL_HEAD = ('__device__ __forceinline__ void spatial_withdraw_cell(\n    const long long j, const long long N, const int L,\n    const double* w, const double* q, const unsigned char* active_u8,' + _MAP_PARAMS
                + ',\n    double* out_w, double* actual, double* unmet, double* scale, double* contrib)\n{\n')

SPATIAL_TAIL = ('''
extern "C" __global__ void maple_syrup_spatial_withdraw(
    const long long N, const int L, const double* w, const double* q, const unsigned char* active_u8,''' + _MAP_PARAMS + r''',
    double* out_w, double* actual, double* unmet, double* scale, double* partials)
{
    __shared__ double sh[8][BLOCK];
    const int tid = threadIdx.x;
    const long long j = (long long)blockIdx.x * (long long)blockDim.x + (long long)tid;
    const double positive_inf = __longlong_as_double(0x7ff0000000000000LL);
    double contrib[8] = {0.0, 0.0, 0.0, 0.0, 0.0, positive_inf, 0.0, 0.0};  /* neutral values for a padded lane */
    spatial_withdraw_cell(j, N, L, w, q, active_u8, dz_p, dz_sl, dz_sc, tr_p, tr_sl, tr_sc, ts_p, ts_sl, ts_sc, coef, out_w, actual, unmet, scale, contrib);  /* returns at once for j >= N */
    for (int r = 0; r < 8; ++r) sh[r][tid] = contrib[r];
    __syncthreads();  /* every lane, padded or not, reaches every barrier */
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int r = 0; r < 5; ++r) sh[r][tid] = sh[r][tid] + sh[r][tid + s];
            sh[5][tid] = (sh[5][tid + s] < sh[5][tid]) ? sh[5][tid + s] : sh[5][tid];
            sh[6][tid] = (sh[6][tid + s] > sh[6][tid]) ? sh[6][tid + s] : sh[6][tid];
            sh[7][tid] = (sh[7][tid + s] > sh[7][tid]) ? sh[7][tid + s] : sh[7][tid];
        }
        __syncthreads();
    }
    if (tid == 0) { for (int r = 0; r < 8; ++r) partials[(long long)r * (long long)gridDim.x + (long long)blockIdx.x] = sh[r][0]; }
}

extern "C" __global__ void maple_syrup_spatial_withdraw_finalize(const double* partials, const long long n_blocks, double* packet)
{
    __shared__ double sh[8][@FINAL@];
    const int tid = threadIdx.x;
    const double positive_inf = __longlong_as_double(0x7ff0000000000000LL);
    for (int r = 0; r < 8; ++r) {
        double acc = (r == 5) ? positive_inf : 0.0;
        for (long long b = tid; b < n_blocks; b += @FINAL@) {
            const double v = partials[(long long)r * n_blocks + b];
            if (r < 5) acc = acc + v;
            else if (r == 5) { if (v < acc) acc = v; }
            else { if (v > acc) acc = v; }
        }
        sh[r][tid] = acc;
    }
    __syncthreads();
    for (int s = @FINAL@ / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int r = 0; r < 5; ++r) sh[r][tid] = sh[r][tid] + sh[r][tid + s];
            sh[5][tid] = (sh[5][tid + s] < sh[5][tid]) ? sh[5][tid + s] : sh[5][tid];
            sh[6][tid] = (sh[6][tid + s] > sh[6][tid]) ? sh[6][tid + s] : sh[6][tid];
            sh[7][tid] = (sh[7][tid + s] > sh[7][tid]) ? sh[7][tid + s] : sh[7][tid];
        }
        __syncthreads();
    }
    if (tid == 0) { for (int r = 0; r < 8; ++r) packet[r] = sh[r][0]; }
}

extern "C" __global__ void maple_syrup_spatial_availability(
    const long long N, const int L, const double* w, const unsigned char* active_u8,''' + _MAP_PARAMS + r''',
    double* out, unsigned long long* diag)
{
    const long long j = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (j >= N) return;
    const bool act = active_u8[j] != 0;
    bool bad = false;
    for (int l = 0; l < L; ++l) {
        const double wv = w[(long long)l * N + j];
        if (!(wv >= 0.0 && wv <= WITHDRAW_SATB(l))) bad = true;  /* validated on EVERY cell, inactive included */
        double a = wv - WITHDRAW_RESIDUAL(l);
        if (a < 0.0) a = 0.0;
        out[(long long)l * N + j] = act ? a : 0.0;
    }
    if (bad) atomicMax(&diag[0], 1ULL);
}
''').replace("@FINAL@", str(FINAL_THREADS))


def render_source() -> str:
    """The spatial CUDA withdrawal source: the shared common body (`subsurface_transpiration._CUDA_BODY`) rendered with the spatial readers/head and this module's wrapper kernels (pure, stateless)."""
    return st.render_withdrawal_source(SPATIAL_READERS, SPATIAL_HEAD, SPATIAL_TAIL)


def kernel_source() -> str:
    return render_source()


def kernel_provenance() -> dict[str, Any]:
    src = render_source()
    return {"module": "maple_syrup.spatial_transpiration_cuda", "kernels": list(_KERNELS), "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "shared_body_sha256": hashlib.sha256(st._CUDA_BODY.encode()).hexdigest(), "spatial_readers_sha256": hashlib.sha256(SPATIAL_READERS.encode()).hexdigest(),
            "compile_options": list(COMPILE_OPTIONS), "threads_per_block": THREADS, "final_threads": FINAL_THREADS, "packet_bytes": PACKET_BYTES,
            "common_compile_options": "unchanged (-std=c++11); this module adds --fmad=false so that res/sat products are never fused",
            "reductions": "explicit fused block tree + one-block finalize; no CUB/CuPy reduction; no 8N rows array", "fastmath": False}


_LOCK = threading.Lock()
_FUNCTIONS: dict[int, SimpleNamespace] = {}  # device id -> compiled functions only (never an owner, state or array)


def load_kernels(cp, device_id: int) -> SimpleNamespace:
    """Compile/load the spatial withdrawal module once per process and device (preparation, outside every hot call; no physical dummy launch or copy). Failure is a typed setup error, never a fallback."""
    with _LOCK:
        fns = _FUNCTIONS.get(device_id)
        if fns is not None:
            return fns
        try:
            module = cp.RawModule(code=render_source(), options=COMPILE_OPTIONS, backend="nvrtc")
            got = {name: module.get_function(name) for name in _KERNELS}
            for fn in got.values():
                if int(dict(fn.attributes).get("max_threads_per_block", 0)) < THREADS:
                    raise ss.SubsurfaceBackendUnavailableError(f"a spatial withdrawal kernel supports fewer than {THREADS} threads per block")
        except ss.SubsurfaceBackendUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ss.SubsurfaceBackendUnavailableError(f"the CUDA spatial withdrawal kernels failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        ns = SimpleNamespace(withdraw=got[WITHDRAW_KERNEL], finalize=got[FINAL_KERNEL], availability=got[AVAIL_KERNEL])
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
    """Exact byte terms of one CUDA operation (payload model; allocator rounding, NVRTC, the context and the interpreter are excluded). `outputs` survive the call, `scratch` is released after the synchronising read,
    `host` is the call's own host allocation (the packet that `to_host` returns), `shared` the static shared memory per block (not global).
      withdraw      outputs 4 L N 8; scratch partials 8*8*nb + device packet 64; host 64 (ONE array: the float64 packet is read back directly, no raw/reconstructed pair)
      availability  outputs L N 8; scratch diagnostic 8; host 8
      prepare       0 (module load only; no physical dummy launch or copy)"""
    L, N = int(n_layers), int(n_cells)
    nb, f8 = n_blocks(N), 8
    if operation == "withdraw":
        t = dict(outputs=4 * L * N * f8, scratch=ROWS * f8 * nb + PACKET_BYTES, host=PACKET_BYTES, shared=ROWS * f8 * THREADS)
    elif operation == "availability":
        t = dict(outputs=L * N * f8, scratch=f8 * AVAIL_DIAG_WORDS, host=f8 * AVAIL_DIAG_WORDS, shared=0)
    elif operation == "prepare":
        t = dict(outputs=0, scratch=0, host=0, shared=0)
    else:
        raise ss.SubsurfaceError(f"unknown CUDA operation {operation!r}")
    t["n_blocks"] = nb
    return t


_STRIDES = {"scalar": lambda n: (0, 0), "shared": lambda n: (1, 0), "cell": lambda n: (n, 1)}


def _map_args(column) -> tuple:
    n = column.n_cells
    kinds = {name: v["kind"] for name, v in column.estimate.retained.items()}
    out = []
    for name in ("dz", "theta_residual", "theta_saturation"):
        sl, si = _STRIDES[kinds[name]](int(n))
        out += [column._maps[name], np.int64(sl), np.int64(si)]
    return tuple(out)


def withdraw(cp, column, water, requested, coef: float):
    """One spatial CUDA withdrawal (checks and admission were done by the caller). Returns `(out_w, actual, unmet, scale, packet)` with fresh device outputs and the host float64 packet of 8 doubles in the common order."""
    from maple.core.backend import to_host

    L, N = int(column.n_layers), int(column.n_cells)
    check_launch_ranges(L, N)
    fns = load_kernels(cp, column.device_id)
    out_w, actual, unmet, scale = (cp.empty((L, N), dtype=np.float64) for _ in range(4))
    nb = n_blocks(N)
    partials = cp.empty(ROWS * nb, dtype=np.float64)
    packet_dev = cp.empty(ROWS, dtype=np.float64)
    fns.withdraw((nb,), (THREADS,), (np.int64(N), np.int32(L), water, requested, column._mask, *_map_args(column), np.float64(coef), out_w, actual, unmet, scale, partials))
    fns.finalize((1,), (FINAL_THREADS,), (partials, np.int64(nb), packet_dev))
    packet = np.asarray(to_host(packet_dev), dtype=np.float64)  # the ONE counted constant-size read; synchronises the stream: partials are released only afterwards
    return out_w, actual, unmet, scale, packet


def availability(cp, column, water, coef: float):
    """Fresh `(L, N)` availability on the device and whether any cell's water was invalid (one 8-byte diagnostic read)."""
    from maple.core.backend import to_host

    L, N = int(column.n_layers), int(column.n_cells)
    check_launch_ranges(L, N)
    fns = load_kernels(cp, column.device_id)
    out = cp.empty((L, N), dtype=np.float64)
    diag = cp.zeros(AVAIL_DIAG_WORDS, dtype=np.uint64)
    fns.availability((n_blocks(N),), (THREADS,), (np.int64(N), np.int32(L), water, column._mask, *_map_args(column), np.float64(coef), out, diag))
    words = np.asarray(to_host(diag), dtype=np.uint64)
    return out, bool(int(words[0]))
