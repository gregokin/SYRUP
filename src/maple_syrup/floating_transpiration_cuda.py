"""CuPy RawKernel CUDA withdrawal / availability on a FLOATING porous column (`floating_transpiration`, backend "cuda"). D2; written by its author WITHOUT a GPU (Root qualifies it on hardware).

ONE physical body: `subsurface_transpiration._CUDA_BODY` (the common withdrawal body, byte for byte) rendered by the pure `render_withdrawal_source` with FLOATING readers. The kernel head, the block-tree wrapper, the
finalize kernel and the availability kernel are the ones of `spatial_transpiration_cuda`, re-targeted by FAIL-CLOSED textual substitution (every replaced fragment must occur exactly the expected number of times,
otherwise the render raises); no physical text is copied or edited here. The only differences from the spatial rendering are the reader macros and the map parameters:

    h(l)                 = max(0, min(D[j], e[l+1]) - e[l])          the actual porous thickness (e: the O(L) nominal edges, one small shared array; D: the per-cell soil depth)
    WITHDRAW_RESIDUAL(l) = res_at(theta_res, h)                      res = theta_res * h   (a rounded product)
    WITHDRAW_SATB(l)     = act ? satb_at(theta_sat, h, coef) : +inf  sat = theta_sat * h;  satb = sat + coef * sat;  an INACTIVE placeholder carries no capacity law

NO FUSION (`--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`, the spatial options). No `(L, N)` bound tensor, no layer-sized per-thread array, no 64-layer cap, no 8N rows array, no host mirror, no CuPy/CUB
reduction. Launches on the current stream: `maple_syrup_floating_withdraw` (fixed block tree, `8 * 8 * THREADS` B static shared memory) -> `maple_syrup_floating_withdraw_finalize` (one block) -> ONE `to_host` of the
8-double packet (64 B). `maple_syrup_floating_availability`: one thread per cell, a one-word diagnostic. Every lane, padded or inactive, reaches every barrier. The current device is checked by the caller BEFORE any reader, copy or accounting; admission (device and host separately) precedes every allocation.
"""

from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import spatial_transpiration_cuda as stc
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st

__all__ = ["COMPILE_OPTIONS", "FINAL_THREADS", "PACKET_BYTES", "THREADS", "availability", "kernel_provenance", "kernel_source", "load_kernels", "operation_terms", "render_source", "withdraw"]

COMPILE_OPTIONS = stc.COMPILE_OPTIONS
THREADS = stc.THREADS
FINAL_THREADS = stc.FINAL_THREADS
ROWS = stc.ROWS
PACKET_BYTES = stc.PACKET_BYTES
MAX_GRID = stc.MAX_GRID
WITHDRAW_KERNEL = "maple_syrup_floating_withdraw"
FINAL_KERNEL = "maple_syrup_floating_withdraw_finalize"
AVAIL_KERNEL = "maple_syrup_floating_availability"
_KERNELS = (WITHDRAW_KERNEL, FINAL_KERNEL, AVAIL_KERNEL)

FLOATING_READERS = r"""
#define BLOCK 128
#define FINF __longlong_as_double(0x7ff0000000000000LL)
#define SP(p, sl, sc, l) (p)[(long long)(l) * (sl) + j * (sc)]
__device__ __forceinline__ double fh_at(const double* e, const double d, const int l)
{
    const double top = e[l + 1];
    const double m = (d < top) ? d : top;
    const double h = m - e[l];
    return (h < 0.0) ? 0.0 : h;  /* the actual porous thickness of layer l */
}
__device__ __forceinline__ double res_at(const double theta_res, const double h)
{
    const double res = theta_res * h;  /* a rounded product: --fmad=false keeps it unfused */
    return res;
}
__device__ __forceinline__ double satb_at(const double theta_sat, const double h, const double coef)
{
    const double sat = theta_sat * h;  /* sat = theta_sat * h (rounded) */
    return sat + coef * sat;           /* satb = sat + coef * sat */
}
#define WITHDRAW_GUARD if (j >= N) return;
#define WITHDRAW_RESIDUAL(l) res_at(SP(tr_p, tr_sl, tr_sc, l), fh_at(edges_p, depth_p[j], l))
#define WITHDRAW_SATB(l) (act ? satb_at(SP(ts_p, ts_sl, ts_sc, l), fh_at(edges_p, depth_p[j], l), coef) : FINF)
#define WITHDRAW_STORE(sw_, sa_, st_, su_, sq_, mn_, flags_, qm_) \
    contrib[0] = sw_; contrib[1] = sa_; contrib[2] = st_; contrib[3] = su_; contrib[4] = sq_; \
    contrib[5] = mn_; contrib[6] = (double)flags_; contrib[7] = qm_;
"""

_SPATIAL_PARAMS = stc._MAP_PARAMS
_FLOATING_PARAMS = r"""
    const double* edges_p, const double* depth_p,
    const double* tr_p, const long long tr_sl, const long long tr_sc,
    const double* ts_p, const long long ts_sl, const long long ts_sc, const double coef"""
_SPATIAL_CALL = "dz_p, dz_sl, dz_sc, tr_p, tr_sl, tr_sc, ts_p, ts_sl, ts_sc, coef"
_FLOATING_CALL = "edges_p, depth_p, tr_p, tr_sl, tr_sc, ts_p, ts_sl, ts_sc, coef"


def _retarget(text: str, expected: dict[str, int]) -> str:
    """Fail-closed re-targeting of the spatial head/tail: every fragment must occur exactly as often as expected."""
    pairs = ((_SPATIAL_PARAMS, _FLOATING_PARAMS), (_SPATIAL_CALL, _FLOATING_CALL), ("maple_syrup_spatial_", "maple_syrup_floating_"), ("spatial_withdraw_cell", "floating_withdraw_cell"))
    for old, new in pairs:
        if text.count(old) != expected.get(old, 0):
            raise ss.SubsurfaceError(f"render contract drift: {old[:40]!r} occurs {text.count(old)} times, expected {expected.get(old, 0)}")
        text = text.replace(old, new)
    return text


FLOATING_HEAD = _retarget(stc.SPATIAL_HEAD, {_SPATIAL_PARAMS: 1, "spatial_withdraw_cell": 1})
FLOATING_TAIL = _retarget(stc.SPATIAL_TAIL, {_SPATIAL_PARAMS: 2, _SPATIAL_CALL: 1, "maple_syrup_spatial_": 3, "spatial_withdraw_cell": 1})


def render_source() -> str:
    """The floating CUDA withdrawal source: the shared common body with the floating readers/head and the re-targeted wrapper kernels (pure, stateless)."""
    return st.render_withdrawal_source(FLOATING_READERS, FLOATING_HEAD, FLOATING_TAIL)


def kernel_source() -> str:
    return render_source()


def kernel_provenance() -> dict[str, Any]:
    src = render_source()
    return {"module": "maple_syrup.floating_transpiration_cuda", "kernels": list(_KERNELS), "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "shared_body_sha256": hashlib.sha256(st._CUDA_BODY.encode()).hexdigest(), "floating_readers_sha256": hashlib.sha256(FLOATING_READERS.encode()).hexdigest(),
            "compile_options": list(COMPILE_OPTIONS), "threads_per_block": THREADS, "final_threads": FINAL_THREADS, "packet_bytes": PACKET_BYTES,
            "reductions": "explicit fused block tree + one-block finalize; no CUB/CuPy reduction; no 8N rows array; no layer-sized per-thread array", "fastmath": False}


_LOCK = threading.Lock()
_FUNCTIONS: dict[int, SimpleNamespace] = {}  # device id -> compiled functions only (never an owner, state or array)


def load_kernels(cp, device_id: int) -> SimpleNamespace:
    """Compile/load the floating withdrawal module once per process and device (outside every hot call). Failure is a typed setup error, never a fallback."""
    with _LOCK:
        fns = _FUNCTIONS.get(device_id)
        if fns is not None:
            return fns
        try:
            module = cp.RawModule(code=render_source(), options=COMPILE_OPTIONS, backend="nvrtc")
            got = {name: module.get_function(name) for name in _KERNELS}
            for fn in got.values():
                if int(dict(fn.attributes).get("max_threads_per_block", 0)) < THREADS:
                    raise ss.SubsurfaceBackendUnavailableError(f"a floating withdrawal kernel supports fewer than {THREADS} threads per block")
        except ss.SubsurfaceBackendUnavailableError:
            raise
        except Exception as exc:
            raise ss.SubsurfaceBackendUnavailableError(f"the CUDA floating withdrawal kernels failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        ns = SimpleNamespace(withdraw=got[WITHDRAW_KERNEL], finalize=got[FINAL_KERNEL], availability=got[AVAIL_KERNEL])
        _FUNCTIONS[device_id] = ns
        return ns


def n_blocks(n_cells: int) -> int:
    return stc.n_blocks(n_cells)


def check_launch_ranges(n_layers: int, n_cells: int) -> None:
    stc.check_launch_ranges(n_layers, n_cells)


def operation_terms(operation: str, n_layers: int, n_cells: int) -> dict[str, int]:
    """Exact byte terms of one CUDA operation (payload model; allocator rounding, NVRTC, the context and the interpreter are excluded): as `spatial_transpiration_cuda.operation_terms` (`withdraw`: outputs
    `4 L N 8`, scratch partials `8 * 8 * nb` + packet 64, host 64; `availability`: outputs `L N 8`, scratch/host 8)."""
    return stc.operation_terms(operation, n_layers, n_cells)


def _map_args(column) -> tuple:
    n = column.n_cells
    out = [column._maps["edges"], column._maps["soil_depth"]]
    for name in ("theta_residual", "theta_saturation"):
        sl, si = stc._STRIDES[column.layouts[name]](int(n))
        out += [column._maps[name], np.int64(sl), np.int64(si)]
    return tuple(out)


def withdraw(cp, column, water, requested, coef: float):
    """One floating CUDA withdrawal (checks and admission were done by the caller). Returns `(out_w, actual, unmet, scale, packet)`: fresh device outputs and the host float64 packet of 8 doubles."""
    from maple.core.backend import to_host

    L, N = int(column.n_layers), int(column.n_cells)
    check_launch_ranges(L, N)
    fns = load_kernels(cp, column.device_id)
    out_w, actual, unmet, scale = (cp.empty((L, N), dtype=np.float64) for _ in range(4))
    nb = n_blocks(N)
    partials = cp.empty(ROWS * nb, dtype=np.float64)
    packet_dev = cp.empty(ROWS, dtype=np.float64)
    fns.withdraw((nb,), (THREADS,), (np.int64(N), np.int32(L), water, requested, column._maps["active"], *_map_args(column), np.float64(coef), out_w, actual, unmet, scale, partials))
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
    diag = cp.zeros(1, dtype=np.uint64)
    fns.availability((n_blocks(N),), (THREADS,), (np.int64(N), np.int32(L), water, column._maps["active"], *_map_args(column), np.float64(coef), out, diag))
    words = np.asarray(to_host(diag), dtype=np.uint64)
    return out, bool(int(words[0]))
