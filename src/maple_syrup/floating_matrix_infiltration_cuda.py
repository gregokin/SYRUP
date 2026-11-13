"""CuPy RawKernel CUDA step of the FLOATING persistent Green-Ampt matrix intake (`floating_matrix_infiltration`, backend "cuda"). D2; written by its author WITHOUT a GPU (Root qualifies it on hardware).

ONE physical body: the matrix kernel body of `matrix_infiltration_cuda` (and its Newton helpers, never edited here) rendered by the pure fail-closed `matrix_infiltration_cuda.render_source` with FLOATING readers. The head,
the block-tree wrapper, the finalize kernel and the validation kernel are those of `spatial_matrix_infiltration_cuda`, re-targeted by FAIL-CLOSED textual substitution (every replaced fragment must occur exactly the
expected number of times, otherwise the render raises). Only the designated read sites differ from the spatial rendering:

    h(l)    = max(0, min(D[i], e[l+1]) - e[l])                    the actual porous thickness (e: the O(L) nominal edges, one small shared array; D: the per-cell soil depth)
    FC0     = fc_at(theta_fc[0], h0)                              fc = theta_fc * h0   (a rounded product, 0 on a rock slot)
    DZ0     = h0 > 0 ? h0 : 1.0                                   the safe stand-in ONLY where there is no porous layer 0 (the deficit there is 0 - 0); a positive h0, however small, is never replaced
    SAT0    = sat_at(theta_sat[0], h0)                            sat = theta_sat * h0
    SATB_L  = a ? satb_at(theta_sat[l], h(l), coef) : +inf        satb = sat + coef * sat;  an INACTIVE placeholder carries no capacity law (finiteness and >= 0 are still flagged on every cell)

NO FUSION (`--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`). No `(L, N)` bound tensor, no layer-sized per-thread array, no 64-layer cap, no dense `16 N` rows array, no host mirror, no CuPy/CUB reduction.
Launch order, packets and static shared memory are those of the spatial rendering: `maple_syrup_floating_matrix_step` -> `maple_syrup_floating_matrix_finalize` -> ONE `to_host` of the 17-word raw packet (136 B)
rebuilt into the unchanged 17-float common packet (counted on the host: raw + rebuilt). The current device is checked by the caller BEFORE any reader, copy or accounting; admission (device and host) precedes
every allocation. The cold identity hash streams the three device law arrays through a bounded counted host staging buffer.
"""

from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import matrix_infiltration as mi
from maple_syrup import matrix_infiltration_cuda as mc
from maple_syrup import spatial_matrix_infiltration_cuda as smc
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = ["COMPILE_OPTIONS", "FINAL_THREADS", "PACKET_WORDS", "RAW_PACKET_BYTES", "ROWS", "STAGE_ELEMS", "THREADS", "build_packet", "check_launch_ranges", "hash_device_array", "kernel_provenance",
           "kernel_source", "load_kernels", "operation_terms", "render_source", "step", "validate_range"]

COMPILE_OPTIONS = smc.COMPILE_OPTIONS
THREADS = smc.THREADS
FINAL_THREADS = smc.FINAL_THREADS
ROWS = smc.ROWS
PACKET_WORDS = smc.PACKET_WORDS
RAW_PACKET_BYTES = smc.RAW_PACKET_BYTES
DIAG_WORDS = smc.DIAG_WORDS
STAGE_ELEMS = smc.STAGE_ELEMS
MAX_GRID = smc.MAX_GRID
STEP_KERNEL = "maple_syrup_floating_matrix_step"
FINAL_KERNEL = "maple_syrup_floating_matrix_finalize"
VALIDATE_KERNEL = "maple_syrup_floating_matrix_validate"
_KERNELS = (STEP_KERNEL, FINAL_KERNEL, VALIDATE_KERNEL)

FLOATING_READERS = r"""
#define BLOCK 128
#define FINF __longlong_as_double(0x7ff0000000000000LL)
#define SPM(p, sl, sc, l) (p)[(long long)(l) * (sl) + i * (sc)]
__device__ __forceinline__ double fh_at(const double* e, const double d, const int l)
{
    const double top = e[l + 1];
    const double m = (d < top) ? d : top;
    const double h = m - e[l];
    return (h < 0.0) ? 0.0 : h;  /* the actual porous thickness of layer l */
}
__device__ __forceinline__ double fdz0_at(const double* e, const double d)
{
    const double h = fh_at(e, d, 0);
    return (h > 0.0) ? h : 1.0;  /* safe denominator ONLY where there is no porous layer 0 */
}
__device__ __forceinline__ double fc_at(const double theta_fc, const double h)
{
    const double fc = theta_fc * h;  /* a rounded product: --fmad=false keeps it unfused */
    return fc;
}
__device__ __forceinline__ double sat_at(const double theta_sat, const double h)
{
    const double sat = theta_sat * h;  /* sat = theta_sat * h (rounded) */
    return sat;
}
__device__ __forceinline__ double satb_at(const double theta_sat, const double h, const double coef)
{
    const double sat = theta_sat * h;  /* sat = theta_sat * h (rounded) */
    return sat + coef * sat;           /* the validity bound sat + cap_tol, cap_tol = coef * sat */
}
"""

FLOATING_SUBSTITUTIONS = dict(smc.SPATIAL_SUBSTITUTIONS)
FLOATING_SUBSTITUTIONS.update({
    "@@SATB_L@@": "(a ? satb_at(SPM(ts_p, ts_sl, ts_sc, l), fh_at(edges_p, depth_p[i], l), coef) : FINF)",
    "@@FC0@@": "fc_at(SPM(tf_p, tf_sl, tf_sc, 0), fh_at(edges_p, depth_p[i], 0))",
    "@@DZ0@@": "fdz0_at(edges_p, depth_p[i])",
    "@@SAT0@@": "sat_at(SPM(ts_p, ts_sl, ts_sc, 0), fh_at(edges_p, depth_p[i], 0))",
})

_SP_MAP = "    const double* dz_p, const long long dz_sl, const long long dz_sc,\n"
_FL_MAP = "    const double* edges_p, const double* depth_p,\n"
_SP_CALL = '"dz_p, dz_sl, dz_sc, tf_p'
_SP_CALL_TEXT = "dz_p, dz_sl, dz_sc, tf_p"
_FL_CALL_TEXT = "edges_p, depth_p, tf_p"


def _retarget(text: str, expected: dict[str, int]) -> str:
    pairs = ((_SP_MAP, _FL_MAP), (_SP_CALL_TEXT, _FL_CALL_TEXT), ("maple_syrup_spatial_matrix_", "maple_syrup_floating_matrix_"), ("spatial_matrix_cell", "floating_matrix_cell"))
    for old, new in pairs:
        if text.count(old) != expected.get(old, 0):
            raise ss.SubsurfaceError(f"render contract drift: {old[:40]!r} occurs {text.count(old)} times, expected {expected.get(old, 0)}")
        text = text.replace(old, new)
    return text


FLOATING_HEAD = _retarget(smc.SPATIAL_HEAD, {_SP_MAP: 1, "spatial_matrix_cell": 1})
FLOATING_TAIL = _retarget(smc.SPATIAL_TAIL, {_SP_MAP: 1, _SP_CALL_TEXT: 1, "maple_syrup_spatial_matrix_": 3, "spatial_matrix_cell": 1})


def render_source() -> str:
    """The floating CUDA matrix source: the shared Newton helpers and body (`matrix_infiltration_cuda`) rendered with the floating readers/head/substitutions and the re-targeted wrapper kernels (pure, stateless)."""
    return mc.render_source(readers=FLOATING_READERS, head=FLOATING_HEAD, substitutions=FLOATING_SUBSTITUTIONS, tail=FLOATING_TAIL)


def kernel_source() -> str:
    return render_source()


def kernel_provenance() -> dict[str, Any]:
    src = render_source()
    return {"module": "maple_syrup.floating_matrix_infiltration_cuda", "kernels": list(_KERNELS), "source_sha256": hashlib.sha256(src.encode()).hexdigest(), "common_source_sha256": mc.COMMON_SOURCE_SHA256,
            "floating_readers_sha256": hashlib.sha256(FLOATING_READERS.encode()).hexdigest(), "compile_options": list(COMPILE_OPTIONS), "threads_per_block": THREADS, "final_threads": FINAL_THREADS,
            "raw_packet_bytes": RAW_PACKET_BYTES, "reductions": "explicit fused block tree + one-block finalize; integer atomics for the flag OR and flagged count; no CUB/CuPy reduction; no 16N rows array",
            "fastmath": False}


_LOCK = threading.Lock()
_FUNCTIONS: dict[int, SimpleNamespace] = {}  # device id -> compiled functions only (never an owner, state, history or array)


def load_kernels(cp, device_id: int) -> SimpleNamespace:
    """Compile/load the floating matrix module once per process and device (outside every hot call). Failure is a typed setup error, never a fallback."""
    with _LOCK:
        fns = _FUNCTIONS.get(device_id)
        if fns is not None:
            return fns
        try:
            module = cp.RawModule(code=render_source(), options=COMPILE_OPTIONS, backend="nvrtc")
            got = {name: module.get_function(name) for name in _KERNELS}
            for fn in got.values():
                if int(dict(fn.attributes).get("max_threads_per_block", 0)) < THREADS:
                    raise ss.SubsurfaceBackendUnavailableError(f"a floating matrix kernel supports fewer than {THREADS} threads per block")
        except ss.SubsurfaceBackendUnavailableError:
            raise
        except Exception as exc:
            raise ss.SubsurfaceBackendUnavailableError(f"the CUDA floating matrix kernels failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        ns = SimpleNamespace(step=got[STEP_KERNEL], finalize=got[FINAL_KERNEL], validate=got[VALIDATE_KERNEL])
        _FUNCTIONS[device_id] = ns
        return ns


def n_blocks(n_cells: int) -> int:
    return smc.n_blocks(n_cells)


def check_launch_ranges(n_layers: int, n_cells: int) -> None:
    smc.check_launch_ranges(n_layers, n_cells)


def operation_terms(operation: str, n_layers: int, n_cells: int) -> dict[str, int]:
    """Byte terms as `spatial_matrix_infiltration_cuda.operation_terms` (`step`: outputs `(4 N + L N) 8`, scratch partials `15 * 8 * nb` + raw device packet 136, host raw + rebuilt 272; `history`: outputs `2 N 8` + an
    8-byte diagnostic; `prepare`: outputs `3 N 8` (upper bound), 8-byte diagnostic, host staging buffer `min(N, STAGE_ELEMS) 8`) plus `rebind`: outputs `2 N 8`, scratch two 8-byte diagnostics (old values)."""
    if operation == "rebind":
        return {"outputs": 2 * int(n_cells) * 8, "scratch": 2 * 8 * DIAG_WORDS, "host": 2 * 8 * DIAG_WORDS, "shared": 0, "shared_finalize": 0, "n_blocks": n_blocks(int(n_cells))}
    return smc.operation_terms(operation, n_layers, n_cells)


def _map_args(column) -> tuple:
    n = int(column.n_cells)
    out = [column._maps["edges"], column._maps["soil_depth"]]
    for name in ("theta_field_capacity", "theta_saturation"):
        sl, si = smc._STRIDES[column.layouts[name]](n)
        out += [column._maps[name], np.int64(sl), np.int64(si)]
    return tuple(out)


def validate_range(cp, device_id: int, array: Any, lo: float, hi: float) -> bool:
    """True when EVERY element of the device array is finite and within `[lo, hi]` (one tiny kernel, one 8-byte diagnostic read, no temporary array)."""
    from maple.core.backend import to_host

    fns = load_kernels(cp, device_id)
    n = int(array.shape[0])
    diag = cp.zeros(DIAG_WORDS, dtype=np.uint64)
    fns.validate((n_blocks(n),), (THREADS,), (np.int64(n), array, np.float64(lo), np.float64(hi), diag))
    words = np.asarray(to_host(diag), dtype=np.uint64)
    return not bool(int(words[0]))


hash_device_array = smc.hash_device_array
build_packet = smc.build_packet


def step(cp, op, pool, standing, rate, water, wet, cum, dt: float):
    """One floating CUDA matrix step (checks and admission were done by the caller). Returns `(pool_out, water_out, wet_out, cum_out, intake, packet)`: fresh device outputs and the host 17-float packet."""
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
    coef = float(volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0))
    fns.step((nb,), (THREADS,), (np.int64(N), np.int32(L), pool, standing, rate, water, wet, cum, column._maps["active"], np.float64(dt), *_map_args(column), np.float64(coef),
                                  k, np.int64(strides[0]), psi, np.int64(strides[1]), g, np.int64(strides[2]), np.float64(mi.cell_closure_coefficient(L)), np.float64(mi.newton_tolerance()),
                                  np.int32(mi.NEWTON_MAX_ITERATIONS), pool_out, water_out, wet_out, cum_out, intake, partials, packet_dev))
    fns.finalize((1,), (FINAL_THREADS,), (partials, np.int64(nb), packet_dev))
    raw = np.asarray(to_host(packet_dev), dtype=np.uint64)  # the ONE counted constant-size read; synchronises the stream: partials are released only afterwards
    return pool_out, water_out, wet_out, cum_out, intake, build_packet(raw)
