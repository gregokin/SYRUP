"""CuPy RawKernel CUDA runner of the SYRUP matrix infiltration stage (`maple_syrup.matrix_infiltration`, backend
"cuda").

One thread per cell, layer loops inside the thread (the layer count is a kernel argument: no fixed maximum). The
kernel evaluates EXACTLY the stages of `matrix_infiltration.reference_step` in the same expression order with the same
IEEE operations (compiled with the routing options `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`, no
fast math); the implicit Green-Ampt solve is the scalar Newton loop of the Numba kernel; the device `log1p`/`log`
may differ from NumPy's by an ulp. Declared agreement with the reference: rtol 2e-12 / atol 1e-14 (bitwise where no
logarithm is evaluated).

ONE launch per step on the current stream. Host traffic per step: ZERO uploads (scalars travel as kernel arguments;
the operator owns device copies of its maps from `prepare_matrix`), ONE counted device-to-host read of the stacked
packet (`matrix_infiltration._read_packet`). Every output array is freshly allocated per call on the operator's
device; inputs are never written. The runner's `preflight` refuses a step when the current device is not the
column's device (checked BEFORE any allocation or launch).

CuPy, a visible device and a loadable kernel are required (`require_cuda`); any failure is a
`MatrixBackendUnavailableError`, never a NumPy fallback. Nothing here was run by its author; Codex records results.

Phase 2 S2-B2.c: the kernel text is now rendered from shared pieces by the pure, fail-closed `render_source` (see the comment above `_NEWTON`). The common rendering is byte-identical to the previous text
(`COMMON_SOURCE_SHA256`); the explicit spatial matrix operator renders the SAME Newton helpers and body with its own readers in `maple_syrup.spatial_matrix_infiltration_cuda`.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

import numpy as np

from maple_syrup.matrix_infiltration import (
    LOG1PMX_TERMS,
    NEWTON_MAX_ITERATIONS,
    NO_FLAG,
    ROW_COUNT,
    THREADS,
    KernelMaps,
    MatrixBackendUnavailableError,
    MatrixError,
    StepArrays,
)
from maple_syrup.routing_cuda import COMPILE_OPTIONS

__all__ = ["BODY_TOKENS", "COMMON_SOURCE_SHA256", "COMMON_SUBSTITUTIONS", "HISTORICAL_COMMON_SOURCE_SHA256", "NUMERICAL_REVISION", "COMPILE_OPTIONS", "KERNEL_NAME", "LAUNCHES_PER_STEP", "THREADS", "current_device_id",
           "kernel_provenance", "kernel_source", "load_kernel", "prepare_runner", "render_source", "require_cuda"]

KERNEL_NAME = "maple_syrup_matrix_step"
LAUNCHES_PER_STEP = 1

# Flag bits, stage order and the Newton loop mirror matrix_infiltration.reference_step / _numba exactly.
#
# Phase 2 S2-B2.c (ONE physical body, two renderings): the kernel text is written ONCE as the pieces below. The COMMON rendering (`_SOURCE`) reproduces the recorded current kernel text byte for byte (its sha256 is
# `COMMON_SOURCE_SHA256`, checked at import and by the tests): the common launch ABI and options are unchanged; the arithmetic differs from the pre-review02 text ONLY by the bounded combined intake (`NUMERICAL_REVISION`). The explicit SPATIAL rendering (`maple_syrup.spatial_matrix_infiltration_cuda`)
# supplies its own reader macros, its own `__device__` head and a reduction tail, and substitutes the designated `@@TOKEN@@` read sites of the shared body. `render_source` is PURE and fails closed:
# every token must be supplied, each must occur exactly the declared number of times in the body, no token may remain, and the Newton text (`_NEWTON`) is never edited.
_NEWTON = r"""
#define LOG1PMX_TERMS __LOG1PMX_TERMS__

__device__ __forceinline__ double log1pmx(const double u)
{
    if (u > 1.0) return log1p(u) - u;
    const double z = u / (2.0 + u);
    const double z2 = z * z;
    double p = z * z2;
    double s = p / 3.0;
    for (int k = 2; k <= LOG1PMX_TERMS; ++k) {
        p = p * z2;
        s = s + p / (double)(2 * k + 1);
    }
    return (-2.0 * z2) / (1.0 - z) + 2.0 * s;
}

__device__ __forceinline__ double ga_increment(const double j_old, const double b, const double r_dt,
                                               const double tol, const int max_iter, double* iters, bool* converged)
{
    *iters = 0.0;
    if (r_dt == 0.0 || b == 0.0) { *converged = true; return r_dt; }
    const double d = j_old + b;
    double y = sqrt(r_dt * r_dt + (2.0 * r_dt) * b);
    if (j_old > 0.0) {
        const double y2 = (r_dt * b) / j_old;
        if (y2 < y) y = y2;
    }
    const double dr = d + r_dt;
    const double jr = j_old + r_dt;
    const double br = b * r_dt;
    double it = 0.0;
    bool conv = false;
    for (int n = 0; n < max_iter; ++n) {
        const double s = r_dt + y;
        const double u = s / d;
        double psi;
        if (isinf(u)) psi = y - b * (log(s) - log(d));
        else psi = ((y * j_old) - br) / d - b * log1pmx(u);
        if (psi <= 0.0) { conv = true; break; }
        const double dpsi = (jr + y) / (dr + y);
        const double step = psi / dpsi;
        double y_new = y - step;
        it += 1.0;
        if (y_new >= y) { conv = true; break; }
        if (y_new < 0.0) y_new = 0.0;
        y = y_new;
        if (step <= tol * y) { conv = true; break; }
    }
    *iters = it;
    *converged = conv;
    return r_dt + y;
}

"""

_COMMON_HEAD = r"""extern "C" __global__ void maple_syrup_matrix_step(
    const long long n, const int nl,
    const double* pool_in, const double* standing_in, const double* rate_in, const double* w_in,
    const double* wet_in, const double* cum_in, const unsigned char* active, const double dt,
    const double* kmap, const double* psimap, const double* gmap,
    const double* dz, const double* fc_m, const double* sat_m, const double* cap_tol,
    const double cell_tol, const double tol, const int max_iter, const double no_flag,
    double* pool_out, double* w_out, double* wet_out, double* cum_out, double* intake_out, double* rows)
{
"""

_BODY = r"""@@LANE@@    const bool a = active[i] != 0;
    unsigned int f = 0u;
    double before = 0.0;
    for (int l = 0; l < nl; ++l) {
        const long long p = (long long)l * n + i;
        const double wl = w_in[p];
        if (!isfinite(wl)) f |= 1u << 0;
        if (wl < 0.0) f |= 1u << 1;
        if (wl > @@SATB_L@@) f |= 1u << 2;
        before = before + wl;
        w_out[p] = wl;
    }
    const double pool = pool_in[i];
    if (!isfinite(pool)) f |= 1u << 3;
    if (pool < 0.0) f |= 1u << 4;
    const double h0 = standing_in[i];
    if (!isfinite(h0)) f |= 1u << 5;
    if (h0 < 0.0) f |= 1u << 6;
    const double r = rate_in[i];
    if (!isfinite(r)) f |= 1u << 7;
    if (r < 0.0) f |= 1u << 8;
    if ((!a) && r != 0.0) f |= 1u << 9;
    const double jin = wet_in[i];
    if (!isfinite(jin)) f |= 1u << 10;
    if (jin < 0.0) f |= 1u << 11;
    const double cin = cum_in[i];
    if (!isfinite(cin)) f |= 1u << 12;
    if (cin < 0.0) f |= 1u << 13;
    double pool_o = pool, j_o = jin, c_o = cin, after = before;
    double candidate = 0.0, intake = 0.0, pre = 0.0, post = 0.0, rejected = 0.0, unused = 0.0;
    double onset = 0.0, ponded = 0.0, iters = 0.0;
    const double inc = r * dt;
    const bool contact = h0 > 0.0;
    if (a && (!contact)) {
        const double scale_c = (pool + inc) * cell_tol;
        if (pool > inc + scale_c) f |= 1u << 14;
    }
    if (a && dt > 0.0) {
        const double k = @@K@@;
        const double psi = @@PSI@@;
        const double g = @@G@@;
        const double w0 = w_in[i];
        double deficit = @@FC0@@ - w0;
        if (deficit < 0.0) deficit = 0.0;
        const double b = psi * (deficit / @@DZ0@@);
        double room = @@SAT0@@ - w0;
        if (room < 0.0) room = 0.0;
        double t_u = 0.0, tau = 0.0;
        bool run_post = false, inside = false;
        if (contact) { run_post = true; tau = dt; }
        else if (r <= k) { t_u = dt; }
        else {
            const double fs = ((g * b) * k) / (r - k);
            if (jin >= fs) { run_post = true; tau = dt; }
            else {
                const double t_p = (fs - jin) / r;
                if (t_p < dt) { t_u = t_p; tau = dt - t_p; run_post = true; inside = true; }
                else { t_u = dt; }
            }
        }
        const double c1 = r * t_u;
        pre = c1;
        if (pool < pre) pre = pool;
        if (room < pre) pre = room;
        candidate = c1;
        if ((t_u > 0.0) && (pre < c1)) { run_post = false; inside = false; }
        if (inside) onset = 1.0;
        const double j1 = jin + pre;
        double adv = 0.0;
        if (run_post) {
            ponded = 1.0;
            bool conv = false;
            const double x = ga_increment(j1, b, k * tau, tol, max_iter, &iters, &conv);
            if (!conv) f |= 1u << 15;
            const double c2 = g * x;
            candidate = candidate + c2;
            const double pool_left = pool - pre;
            const double room_left = room - pre;
            post = c2;
            if (pool_left < post) post = pool_left;
            if (room_left < post) post = room_left;
        }
        /* ONE combined actual intake (S2-B2.c review02), the same ordering as the NumPy/Numba bodies: bound pre + post by the pool, then by the room, BEFORE anything is published. Inactive bound:
           byte-identical arithmetic. Active bound: the ponded actual is re-derived, post = intake - pre (>= 0 exactly: pre <= pool and pre <= room), so pool, soil, history and ledger share the SAME actuals. */
        intake = pre + post;
        bool reduced = false;
        if (intake > pool) { intake = pool; reduced = true; }
        if (intake > room) { intake = room; reduced = true; }
        if (run_post) {
            if (reduced) post = intake - pre;
            if (g > 0.0) adv = post / g;
            j_o = j1 + adv;
        } else {
            j_o = j1;
        }
        double bounded = candidate;
        if (pool < bounded) bounded = pool;
        rejected = bounded - intake;
        unused = candidate - bounded;
        w_out[i] = w0 + intake;
        pool_o = pool - intake;
        c_o = cin + intake;
        after = 0.0;
        for (int l = 0; l < nl; ++l) {
            const double w = w_out[(long long)l * n + i];
            if (!isfinite(w)) f |= 1u << 16;
            if (w < 0.0) f |= 1u << 17;
            if (w > @@SATB_L@@) f |= 1u << 18;
            after = after + w;
        }
        if ((!isfinite(pool_o)) || (!isfinite(j_o)) || (!isfinite(c_o))) f |= 1u << 16;
        if (pool_o < 0.0 || j_o < 0.0 || c_o < 0.0) f |= 1u << 17;
        const double scale = ((((before + after) + pool) + pool_o) + intake) * cell_tol;
        const double closure = (after - before) - intake;
        if (fabs(closure) > scale) f |= 1u << 19;
        const double closure_p = (pool - pool_o) - intake;
        if (fabs(closure_p) > scale) f |= 1u << 20;
        const double adv_total = pre + adv;
        const double h_scale = ((jin + j_o) + adv_total) * cell_tol;
        if (fabs((j_o - jin) - adv_total) > h_scale) f |= 1u << 21;
        const double c_scale = ((cin + c_o) + intake) * cell_tol;
        if (fabs((c_o - cin) - intake) > c_scale) f |= 1u << 21;
        if ((!isfinite(candidate)) || (!isfinite(intake))) f |= 1u << 22;
    }
    pool_out[i] = pool_o;
    wet_out[i] = j_o;
    cum_out[i] = c_o;
@@STORE@@
"""

# Designated read/store sites of the body and how often each occurs (fail-closed render contract).
BODY_TOKENS = {"@@LANE@@": 1, "@@SATB_L@@": 2, "@@FC0@@": 1, "@@DZ0@@": 1, "@@SAT0@@": 1, "@@K@@": 1, "@@PSI@@": 1, "@@G@@": 1, "@@STORE@@": 1}

_COMMON_STORE = r"""    if (a) {
        intake_out[i] = intake;
        rows[0 * n + i] = before;
        rows[1 * n + i] = after;
        rows[2 * n + i] = pool;
        rows[3 * n + i] = pool_o;
        rows[4 * n + i] = candidate;
        rows[5 * n + i] = intake;
        rows[6 * n + i] = pre;
        rows[7 * n + i] = post;
        rows[8 * n + i] = rejected;
        rows[9 * n + i] = unused;
        rows[10 * n + i] = jin;
        rows[11 * n + i] = j_o;
        rows[12 * n + i] = onset;
        rows[13 * n + i] = ponded;
        rows[14 * n + i] = iters;
    } else {
        intake_out[i] = 0.0;
        for (int row = 0; row < 15; ++row) rows[(long long)row * n + i] = 0.0;
    }
    rows[15 * n + i] = (f == 0u) ? no_flag : (double)(f & (0u - f));"""

#: The common rendering of every token: exactly the text the previous kernel contained at that site.
COMMON_SUBSTITUTIONS = {
    "@@LANE@@": r"""    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
""",
    "@@SATB_L@@": "sat_m[l] + cap_tol[l]", "@@FC0@@": "fc_m[0]", "@@DZ0@@": "dz[0]", "@@SAT0@@": "sat_m[0]", "@@K@@": "kmap[i]", "@@PSI@@": "psimap[i]", "@@G@@": "gmap[i]",
    "@@STORE@@": _COMMON_STORE,
}


def render_source(*, readers: str = "", head: str | None = None, substitutions: dict[str, str] | None = None, tail: str = "") -> str:
    """PURE rendering (no registry, no cache, no state) of a matrix CUDA source: the shared Newton text, the caller's reader macros, the caller's kernel/device head (default: the common kernel head), the ONE
    shared body with every `@@TOKEN@@` replaced by the caller's text (default: the common text) and the closing brace, then an optional tail. Fails closed (`MatrixError`) on any contract drift."""
    if not all(isinstance(x, str) for x in (readers, tail)) or not (head is None or isinstance(head, str)):
        raise MatrixError("readers, head and tail must be strings")
    subs = dict(COMMON_SUBSTITUTIONS if substitutions is None else substitutions)
    if set(subs) != set(BODY_TOKENS) or not all(isinstance(v, str) for v in subs.values()):
        raise MatrixError(f"the substitutions must define exactly {sorted(BODY_TOKENS)}")
    if any("@@" in v for v in subs.values()):
        raise MatrixError("render contract drift: a substitution may not contain a token marker")
    body = _BODY
    for token, count in BODY_TOKENS.items():
        if body.count(token) != count:
            raise MatrixError(f"render contract drift: token {token} occurs {body.count(token)} times in the shared body, expected {count}")
    for token in BODY_TOKENS:
        body = body.replace(token, subs[token])
    if "@@" in body:
        raise MatrixError("render contract drift: an unreplaced token remains in the shared body")
    return _NEWTON.replace("__LOG1PMX_TERMS__", str(int(LOG1PMX_TERMS))) + readers + (_COMMON_HEAD if head is None else head) + body + "}\n" + tail


_SOURCE = render_source()
#: sha256 of the common kernel text BEFORE the shared-body factoring (S2-B2.c) and BEFORE the review02 arithmetic correction: HISTORICAL evidence only (old runs/snapshots keep their own recorded identities and
#: are never rebound to the current source).
HISTORICAL_COMMON_SOURCE_SHA256 = "b58178dfc303f48364f7bfdcd744a08bf821da481ae61d8f347981c3bee5c05f"
#: sha256 of the CURRENT common kernel text: the previous text with the S2-B2.c review02 bounded-combined-intake arithmetic (the only intended difference; see `NUMERICAL_REVISION`). The common rendering must
#: reproduce it exactly; the digest changes explicitly with that approved arithmetic fix and nothing else.
COMMON_SOURCE_SHA256 = "b9e131308266af692f9a40b3edb09a21df349c67406b6b6927820e1607b4f3d2"
NUMERICAL_REVISION = ("S2-B2.c review02: the combined actual intake `pre + post` is bounded by the offered pool and by the top-layer room before anything is published; the ponded actual is re-derived as "
                      "`intake - pre` ONLY when that bound is active (valid inputs whose separately bounded parts recombine one ulp above a bound no longer produce a negative pool); no epsilon, no clipping of negatives afterwards")
if hashlib.sha256(_SOURCE.encode()).hexdigest() != COMMON_SOURCE_SHA256:  # fail closed at import: the common text/ABI must not drift
    raise MatrixError("the common matrix CUDA source no longer renders to its recorded current text (COMMON_SOURCE_SHA256)")

_MODULES: dict[int, Any] = {}
_FUNCTIONS: dict[int, Any] = {}
_LOCK = threading.Lock()


def kernel_source() -> str:
    return _SOURCE


def require_cuda():
    """The CuPy module with at least one visible device, through MAPLE's own probes; otherwise a named setup error."""
    from maple.core.backend import cuda_device_count, cupy_import_error, cupy_module

    cp = cupy_module()
    if cp is None:
        raise MatrixBackendUnavailableError(
            f"backend 'cuda' requires CuPy, which is not usable ({cupy_import_error() or 'not installed'}); there is "
            "no fallback to the NumPy reference")
    if cuda_device_count() < 1:
        raise MatrixBackendUnavailableError(
            "backend 'cuda' requires a visible CUDA device (device count 0); there is no fallback to the NumPy reference")
    return cp


def current_device_id(cp) -> int:
    try:
        return int(cp.cuda.Device().id)
    except Exception as exc:  # reported as a setup failure
        raise MatrixBackendUnavailableError(f"CUDA device query failed ({type(exc).__name__}: {exc})") from exc


def load_kernel(cp, device_id: int) -> Any:
    """Compile/load the kernel for `device_id` (once per process and device). Failure is a setup error."""
    with _LOCK:
        fn = _FUNCTIONS.get(device_id)
        if fn is not None:
            return fn
        try:
            module = cp.RawModule(code=_SOURCE, options=COMPILE_OPTIONS, backend="nvrtc")
            fn = module.get_function(KERNEL_NAME)
            attrs = dict(fn.attributes)  # forces the module load on this device
            if int(attrs.get("max_threads_per_block", 0)) < THREADS:
                raise MatrixBackendUnavailableError(
                    f"kernel supports only {attrs.get('max_threads_per_block')} threads per block, {THREADS} needed")
        except MatrixBackendUnavailableError:
            raise
        except Exception as exc:
            raise MatrixBackendUnavailableError(
                f"the CUDA matrix kernel failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        _MODULES[device_id] = module
        _FUNCTIONS[device_id] = fn
        return fn


def kernel_provenance() -> dict[str, Any]:
    return {
        "module": "maple_syrup.matrix_infiltration_cuda",
        "kernels": [KERNEL_NAME],
        "source_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
        "historical_source_sha256": HISTORICAL_COMMON_SOURCE_SHA256,
        "numerical_revision": NUMERICAL_REVISION,
        "compile_options": list(COMPILE_OPTIONS),
        "threads_per_block": THREADS,
        "launches_per_step": LAUNCHES_PER_STEP,
        "launch_structure": "one thread per cell, grid ceil(n_cells/threads), layer loops and the scalar Newton loop "
                            "inside the thread reading the operator's device maps; no shared memory, no atomics, no "
                            "fixed maximum layer count",
        "newton": {"max_iterations": NEWTON_MAX_ITERATIONS, "device_functions": ["log1p", "log", "sqrt"]},
        "reductions": "CuPy reductions of kernel rows stacked into one packet; one counted host read per step",
        "fastmath": False,
        "libm": "device log1p/log (may differ from NumPy by an ulp); declared bound rtol 2e-12 / atol 1e-14",
    }


def _launch(cp, fn, maps: KernelMaps, cell_tol: float, tol: float, pool_in, standing_in, rate_in, water_in, wetting_in,
            cumulative_in, active_u8, dt: float) -> StepArrays:
    n_layers, n_cells = int(water_in.shape[0]), int(water_in.shape[1])
    pool_out = cp.empty(n_cells, dtype=np.float64)
    water_out = cp.empty((n_layers, n_cells), dtype=np.float64)
    wet_out = cp.empty(n_cells, dtype=np.float64)
    cum_out = cp.empty(n_cells, dtype=np.float64)
    intake = cp.empty(n_cells, dtype=np.float64)
    rows = cp.empty((ROW_COUNT, n_cells), dtype=np.float64)
    grid = ((n_cells + THREADS - 1) // THREADS,)
    fn(grid, (THREADS,), (
        np.int64(n_cells), np.int32(n_layers), pool_in, standing_in, rate_in, water_in, wetting_in, cumulative_in,
        active_u8, np.float64(dt), maps.conductivity_m_s, maps.suction_m, maps.vegetation, maps.thickness_m,
        maps.field_capacity_m, maps.saturation_m, maps.capacity_tolerance_m, np.float64(cell_tol), np.float64(tol),
        np.int32(NEWTON_MAX_ITERATIONS), np.float64(NO_FLAG), pool_out, water_out, wet_out, cum_out, intake, rows))
    return StepArrays(pool_out, water_out, wet_out, cum_out, intake, rows)


def prepare_runner(maps: KernelMaps, cell_tol: float, tol: float, cp, device_id: int):
    """Load the kernel for the current device and warm it up on one inactive cell (`maps` are the operator's private
    resident device arrays). Returns `(runner, info, preflight)`."""
    if current_device_id(cp) != device_id:
        raise MatrixBackendUnavailableError("prepare_runner must run on the column's device")
    for name, a in zip(KernelMaps._fields, maps, strict=True):
        if type(a) is not cp.ndarray or int(a.device.id) != device_id:
            raise MatrixError(f"internal: map {name} is not a device array of device {device_id}")
    fn = load_kernel(cp, device_id)
    n_layers = int(maps.thickness_m.shape[0])
    warm = KernelMaps(cp.zeros(1), cp.zeros(1), cp.zeros(1), maps.thickness_m, maps.field_capacity_m, maps.saturation_m,
                      maps.capacity_tolerance_m)
    z = cp.zeros(1)
    _launch(cp, fn, warm, cell_tol, tol, z, z, z, cp.zeros((n_layers, 1)), z, z, cp.zeros(1, np.uint8), 1.0)
    cp.cuda.Device().synchronize()

    def preflight() -> None:
        current = current_device_id(cp)
        if current != device_id:
            raise MatrixError(f"the operator is bound to device {device_id} but the current device is {current}")

    def run(op, maps_internal, pool_in, standing_in, rate_in, water_in, wetting_in, cumulative_in, dt):
        return _launch(cp, fn, maps_internal, cell_tol, tol, pool_in, standing_in, rate_in, water_in, wetting_in,
                       cumulative_in, op.column.active_u8, dt)

    info = {"implementation": "cupy RawKernel: one thread per cell; private resident maps; scalar Newton",
            "device_id": device_id, "launches_per_step": LAUNCHES_PER_STEP, "provenance": kernel_provenance(),
            "parameter_device_bytes": int(sum(int(a.nbytes) for a in maps))}
    return run, info, preflight
