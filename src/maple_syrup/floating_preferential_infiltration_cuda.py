"""CuPy RawKernel CUDA step of the FLOATING preferential intake (`floating_preferential_infiltration`, backend "cuda"). D2; written by its author WITHOUT a GPU (Root qualifies it on hardware).

ONE physical body: the preferential kernel body of `preferential_infiltration_cuda` rendered by its pure fail-closed `render_source` with FLOATING read sites. The head, the block-tree wrapper, the finalize kernel, the
cold reduction and the validation kernels are those of `spatial_preferential_infiltration_cuda`, re-targeted by FAIL-CLOSED textual substitution (every replaced fragment must occur exactly the expected number of times,
otherwise the render raises). The spatial reader helpers are reused with the actual porous thickness in the place of `dz`:

    h(l)    = max(0, min(D[i], e[l+1]) - e[l])                    the actual porous thickness (e: the O(L) nominal edges, one small shared array; D: the per-cell soil depth)
    SAT_L   = sat_at(theta_sat[l], h(l))                          sat = theta_sat * h (0 on a rock slot: that receiver admits nothing and its share stays on the surface)
    SATB_L  = a ? satb_at(theta_sat[l], h(l), coef) : +inf        an INACTIVE placeholder carries no capacity law (finiteness and >= 0 are still flagged on every cell)

The receiver weights are FIXED and are never renormalised over porous layers. NO FUSION (`--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`). No `(L, N)` bound tensor, no layer-sized per-thread array, no
64-layer cap, no dense rows array, no host mirror, no CuPy/CUB reduction. ONE 72-byte raw packet is read back and rebuilt into the unchanged 9-float packet (both counted). The current device is checked by the caller
BEFORE any reader, copy or accounting; admission (device and host) precedes every allocation; the cold derivation uses the bounded counted 32-byte reducer of this module.
"""

from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import preferential_infiltration as pi
from maple_syrup import preferential_infiltration_cuda as pc
from maple_syrup import spatial_preferential_infiltration_cuda as spc
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = [
    "COMPILE_OPTIONS",
    "FINAL_THREADS",
    "PACKET_WORDS",
    "RAW_PACKET_BYTES",
    "ROWS",
    "STAGE_ELEMS",
    "THREADS",
    "ColdReducer",
    "build_packet",
    "check_launch_ranges",
    "hash_device_array",
    "kernel_provenance",
    "kernel_source",
    "load_kernels",
    "operation_terms",
    "render_source",
    "step",
    "validate_range",
]

COMPILE_OPTIONS = spc.COMPILE_OPTIONS
THREADS = spc.THREADS
FINAL_THREADS = spc.FINAL_THREADS
ROWS = spc.ROWS
PACKET_WORDS = spc.PACKET_WORDS
RAW_PACKET_BYTES = spc.RAW_PACKET_BYTES
DIAG_BYTES = spc.DIAG_BYTES
STAGE_ELEMS = spc.STAGE_ELEMS
STEP_KERNEL = "maple_syrup_floating_pref_step"
FINAL_KERNEL = "maple_syrup_floating_pref_finalize"
REDUCE_KERNEL = "maple_syrup_floating_pref_reduce"
VALIDATE_KERNEL = "maple_syrup_floating_pref_validate"
_KERNELS = (STEP_KERNEL, FINAL_KERNEL, REDUCE_KERNEL, VALIDATE_KERNEL)

FLOATING_READERS = spc.SPATIAL_READERS + r"""
#define FINF __longlong_as_double(0x7ff0000000000000LL)
__device__ __forceinline__ double fh_at(const double* e, const double d, const int l)
{
    const double top = e[l + 1];
    const double m = (d < top) ? d : top;
    const double h = m - e[l];
    return (h < 0.0) ? 0.0 : h;  /* the actual porous thickness of layer l (passed where the spatial helpers take dz) */
}
"""

FLOATING_SUBSTITUTIONS = dict(spc.SPATIAL_SUBSTITUTIONS)
FLOATING_SUBSTITUTIONS.update({
    "@@SATB_L@@": "(a ? satb_at(SPM(ts_p, ts_sl, ts_sc, l), fh_at(edges_p, depth_p[i], l), coef) : FINF)",
    "@@SAT_L@@": "sat_at(SPM(ts_p, ts_sl, ts_sc, l), fh_at(edges_p, depth_p[i], l))",
})

_SP_MAP = "    const double* dz_p, const long long dz_sl, const long long dz_sc,\n"
_FL_MAP = "    const double* edges_p, const double* depth_p,\n"
_SP_CALL_TEXT = "dz_p, dz_sl, dz_sc, ts_p"
_FL_CALL_TEXT = "edges_p, depth_p, ts_p"


def _retarget(text: str, expected: dict[str, int]) -> str:
    pairs = ((_SP_MAP, _FL_MAP), (_SP_CALL_TEXT, _FL_CALL_TEXT), ("maple_syrup_spatial_pref_", "maple_syrup_floating_pref_"), ("spatial_preferential_cell", "floating_preferential_cell"))
    for old, new in pairs:
        if text.count(old) != expected.get(old, 0):
            raise ss.SubsurfaceError(f"render contract drift: {old[:40]!r} occurs {text.count(old)} times, expected {expected.get(old, 0)}")
        text = text.replace(old, new)
    return text


FLOATING_HEAD = _retarget(spc.SPATIAL_HEAD, {_SP_MAP: 1, "spatial_preferential_cell": 1})
FLOATING_TAIL = _retarget(spc.SPATIAL_TAIL, {_SP_MAP: 1, _SP_CALL_TEXT: 1, "maple_syrup_spatial_pref_": 4, "spatial_preferential_cell": 1})


def render_source() -> str:
    """The floating CUDA preferential source: the shared body (`preferential_infiltration_cuda`) rendered with the floating readers/head/substitutions and the re-targeted wrapper kernels (pure, stateless)."""
    return pc.render_source(readers=FLOATING_READERS, head=FLOATING_HEAD, substitutions=FLOATING_SUBSTITUTIONS, tail=FLOATING_TAIL)


def kernel_source() -> str:
    return render_source()


def kernel_provenance() -> dict[str, Any]:
    src = render_source()
    return {"module": "maple_syrup.floating_preferential_infiltration_cuda", "kernels": list(_KERNELS), "source_sha256": hashlib.sha256(src.encode()).hexdigest(), "common_source_sha256": pc.COMMON_SOURCE_SHA256,
            "floating_readers_sha256": hashlib.sha256(FLOATING_READERS.encode()).hexdigest(), "compile_options": list(COMPILE_OPTIONS), "threads_per_block": THREADS, "final_threads": FINAL_THREADS,
            "raw_packet_bytes": RAW_PACKET_BYTES, "cold_diagnostic_bytes": DIAG_BYTES,
            "reductions": "explicit fused block tree + one-block finalize; integer atomics for the flag OR and flagged count; cold 32-byte key/count diagnostic for the derivation; no CUB/CuPy reduction", "fastmath": False}


_LOCK = threading.Lock()
_FUNCTIONS: dict[int, SimpleNamespace] = {}  # device id -> compiled functions only (never an owner, state, map or array)


def load_kernels(cp, device_id: int) -> SimpleNamespace:
    """Compile/load the floating preferential module once per process and device (outside every hot call). Failure is a typed setup error, never a fallback."""
    with _LOCK:
        fns = _FUNCTIONS.get(device_id)
        if fns is not None:
            return fns
        try:
            module = cp.RawModule(code=render_source(), options=COMPILE_OPTIONS, backend="nvrtc")
            got = {name: module.get_function(name) for name in _KERNELS}
            for fn in got.values():
                if int(dict(fn.attributes).get("max_threads_per_block", 0)) < THREADS:
                    raise ss.SubsurfaceBackendUnavailableError(f"a floating preferential kernel supports fewer than {THREADS} threads per block")
        except ss.SubsurfaceBackendUnavailableError:
            raise
        except Exception as exc:
            raise ss.SubsurfaceBackendUnavailableError(f"the CUDA floating preferential kernels failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        ns = SimpleNamespace(step=got[STEP_KERNEL], finalize=got[FINAL_KERNEL], reduce=got[REDUCE_KERNEL], validate=got[VALIDATE_KERNEL])
        _FUNCTIONS[device_id] = ns
        return ns


def n_blocks(n_cells: int) -> int:
    return spc.n_blocks(n_cells)


def check_launch_ranges(n_layers: int, n_cells: int) -> None:
    spc.check_launch_ranges(n_layers, n_cells)


def operation_terms(operation: str, n_layers: int, n_cells: int) -> dict[str, int]:
    """Byte terms of `spatial_preferential_infiltration_cuda.operation_terms` (`step`: outputs `(2 N + 2 L N) 8`, scratch partials `7 * 8 * nb` + raw device packet 72, host raw + rebuilt 144; `prepare`: the retained
    maps + the derivation live set + the cold buffers; the weight records are added by the caller)."""
    return spc.operation_terms(operation, n_layers, n_cells)


def validate_range(cp, device_id: int, array: Any, lo: float, hi: float) -> bool:
    """True when EVERY element of the device array lies in `[lo, hi]` (NaN fails; `+inf` passes only for `hi = +inf`): one tiny kernel, one 8-byte diagnostic read, no temporary array."""
    from maple.core.backend import to_host

    fns = load_kernels(cp, device_id)
    n = int(array.shape[0])
    diag = cp.zeros(1, dtype=np.uint64)
    fns.validate((n_blocks(n),), (THREADS,), (np.int64(n), array, np.float64(lo), np.float64(hi), diag))
    words = np.asarray(to_host(diag), dtype=np.uint64)
    return not bool(int(words[0]))


class ColdReducer:
    """The bounded cold reducer handed to `_DeriveContext.reduce` (as the spatial one, on THIS module's kernels): one call = one launch over one device array, one 32-byte seed copy (COLD H2D) and one 32-byte
    synchronising read (COLD D2H); counters make the cold traffic auditable."""

    def __init__(self, cp: Any, device_id: int) -> None:
        self.cp, self.device_id = cp, int(device_id)
        self.calls, self.h2d_bytes, self.d2h_bytes = 0, 0, 0

    def __call__(self, kind: str, array: Any) -> float:
        from maple.core.backend import to_host

        if kind not in pi.REDUCE_KINDS:
            raise pi.PreferentialError(f"unknown cold reduction kind {kind!r}; expected one of {pi.REDUCE_KINDS}")
        cp = self.cp
        n = int(array.shape[0])
        fns = load_kernels(cp, self.device_id)
        seed = np.array([0, 0, spc._MAX_KEY, 0], dtype=np.uint64)
        diag = cp.array(seed)  # COLD H2D, 32 B
        fns.reduce((n_blocks(n),), (THREADS,), (np.int64(n), array, diag))
        words = np.asarray(to_host(diag), dtype=np.uint64)  # COLD synchronising D2H, 32 B
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
        return spc._decode_key(kmin if kind == "min" else kmax)


hash_device_array = spc.hash_device_array
build_packet = spc.build_packet


def step(cp, op, surface, water, dt: float):
    """One floating CUDA preferential step (checks and admission were done by the caller). Returns `(surface_out, water_out, intake, total, packet)`: fresh device outputs and the host 9-float packet."""
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
    sl, si = spc._STRIDES[column.layouts["theta_saturation"]](int(N))
    fns.step((nb,), (THREADS,), (np.int64(N), np.int32(L), surface, water, column._maps["active"], np.float64(dt), rate, max_rate, np.int64(mr_sc), recv, column._maps["edges"], column._maps["soil_depth"],
                                  column._maps["theta_saturation"], np.int64(sl), np.int64(si), np.float64(coef), np.float64(pi.cell_closure_coefficient(L)), surface_out, water_out, intake, total,
                                  partials, packet_dev))
    fns.finalize((1,), (FINAL_THREADS,), (partials, np.int64(nb), packet_dev))
    raw = np.asarray(to_host(packet_dev), dtype=np.uint64)  # the ONE counted constant-size read; synchronises the stream: partials are released only afterwards
    return surface_out, water_out, intake, total, build_packet(raw)
