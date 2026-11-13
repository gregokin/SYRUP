"""CuPy RawKernel CUDA runner of the SYRUP preferential infiltration stage (`maple_syrup.preferential_infiltration`,
backend "cuda").

One thread per cell, a short loop over the layers of that cell inside the thread; the layer count is a kernel
argument, so there is no hard-coded maximum. The kernel evaluates EXACTLY the stages of
`preferential_infiltration.reference_step` in the same expression order with the same IEEE operations (compiled with
the routing options `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`, no fast math); the device `expm1`
may differ from NumPy's by an ulp and enters only the isolated attempted amount. Declared agreement with the
reference: rtol 2e-12 / atol 1e-14 on state and fluxes (bitwise where every beta is 0 or 1).

ONE launch per step on the current stream. Host traffic per step: ZERO uploads (scalars travel as kernel arguments;
the operator owns device copies of its maps from `prepare_preferential`), ONE counted device-to-host read of the
stacked validation/budget packet (`preferential_infiltration._read_packet`). Every output array is freshly allocated
per call on the operator's device; inputs are never written.

CuPy, a visible device and a loadable kernel are required (`require_cuda`); any failure is a
`PreferentialBackendUnavailableError`, never a NumPy fallback. Dynamic arrays must already live on the column's
device and be ordered with respect to the current stream. Nothing here was run by its author; Codex records results.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

import numpy as np

from maple_syrup.preferential_infiltration import (
    NO_FLAG,
    ROW_COUNT,
    THREADS,
    KernelMaps,
    PreferentialBackendUnavailableError,
    PreferentialError,
    StepArrays,
)
from maple_syrup.routing_cuda import COMPILE_OPTIONS

__all__ = ["COMPILE_OPTIONS", "KERNEL_NAME", "LAUNCHES_PER_STEP", "THREADS", "current_device_id", "kernel_provenance",
           "kernel_source", "load_kernel", "prepare_runner", "require_cuda"]

KERNEL_NAME = "maple_syrup_preferential_step"
LAUNCHES_PER_STEP = 1

# Flag bits and stage order mirror preferential_infiltration.reference_step / _numba exactly.
_SOURCE = r"""
extern "C" __global__ void maple_syrup_preferential_step(
    const long long n, const int nl,
    const double* surface_in, const double* w_in, const unsigned char* active,
    const double dt, const double* rate, const double* max_rate, const double* recv,
    const double* sat_m, const double* cap_tol, const double cell_tol, const double no_flag,
    double* surface_out, double* w_out, double* intake, double* total, double* rows)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    const bool a = active[i] != 0;
    unsigned int f = 0u;
    double before = 0.0;
    for (int l = 0; l < nl; ++l) {
        const long long p = (long long)l * n + i;
        const double wl = w_in[p];
        if (!isfinite(wl)) f |= 1u << 0;
        if (wl < 0.0) f |= 1u << 1;
        if (wl > sat_m[l] + cap_tol[l]) f |= 1u << 2;
        before = before + wl;
        w_out[p] = wl;
        intake[p] = 0.0;
    }
    const double h_in = surface_in[i];
    if (!isfinite(h_in)) f |= 1u << 3;
    if (h_in < 0.0) f |= 1u << 4;
    double left = h_in;
    double total_v = 0.0, isolated_v = 0.0, attempted_v = 0.0;
    double after = before;
    if (a && dt > 0.0) {
        // isolated attempt: H x (-expm1(-(k dt)))
        const double x = rate[i] * dt;
        const double fr = -expm1(-x);
        double attempted = h_in * fr;
        isolated_v = attempted;
        // ONE aggregate rate cap
        const double cap = max_rate[i] * dt;
        if (cap < attempted) attempted = cap;
        attempted_v = attempted;
        // receiver allocation, top layer first; bounded by the room and the remaining surface water
        for (int l = 0; l < nl; ++l) {
            const long long p = (long long)l * n + i;
            double alloc = attempted * recv[p];
            double room = sat_m[l] - w_out[p];
            if (room < 0.0) room = 0.0;
            if (room < alloc) alloc = room;
            if (left < alloc) alloc = left;
            w_out[p] = w_out[p] + alloc;
            left = left - alloc;
            intake[p] = alloc;
            total_v = total_v + alloc;
        }
        // output checks
        after = 0.0;
        for (int l = 0; l < nl; ++l) {
            const double w = w_out[(long long)l * n + i];
            if (!isfinite(w)) f |= 1u << 5;
            if (w < 0.0) f |= 1u << 6;
            if (w > sat_m[l] + cap_tol[l]) f |= 1u << 7;
            after = after + w;
        }
        if (!isfinite(left)) f |= 1u << 5;
        if (left < 0.0) f |= 1u << 6;
        const double closure = (after - before) - total_v;
        const double scale = ((((before + after) + h_in) + left) + total_v) * cell_tol;
        if (fabs(closure) > scale) f |= 1u << 8;
        const double closure_h = (h_in - left) - total_v;
        if (fabs(closure_h) > scale) f |= 1u << 9;
        bool nonfinite = !isfinite(total_v);
        for (int l = 0; l < nl; ++l) {
            if (!isfinite(intake[(long long)l * n + i])) nonfinite = true;
        }
        if (nonfinite) f |= 1u << 10;
    }
    surface_out[i] = left;
    total[i] = total_v;
    rows[0 * n + i] = a ? before : 0.0;
    rows[1 * n + i] = a ? after : 0.0;
    rows[2 * n + i] = a ? h_in : 0.0;
    rows[3 * n + i] = a ? left : 0.0;
    rows[4 * n + i] = isolated_v;
    rows[5 * n + i] = attempted_v;
    rows[6 * n + i] = total_v;
    rows[7 * n + i] = (f == 0u) ? no_flag : (double)(f & (0u - f));
}
"""

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
        raise PreferentialBackendUnavailableError(
            f"backend 'cuda' requires CuPy, which is not usable ({cupy_import_error() or 'not installed'}); there is "
            "no fallback to the NumPy reference")
    if cuda_device_count() < 1:
        raise PreferentialBackendUnavailableError(
            "backend 'cuda' requires a visible CUDA device (device count 0); there is no fallback to the NumPy reference")
    return cp


def current_device_id(cp) -> int:
    try:
        return int(cp.cuda.Device().id)
    except Exception as exc:  # reported as a setup failure
        raise PreferentialBackendUnavailableError(f"CUDA device query failed ({type(exc).__name__}: {exc})") from exc


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
                raise PreferentialBackendUnavailableError(
                    f"kernel supports only {attrs.get('max_threads_per_block')} threads per block, {THREADS} needed")
        except PreferentialBackendUnavailableError:
            raise
        except Exception as exc:
            raise PreferentialBackendUnavailableError(
                f"the CUDA preferential kernel failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        _MODULES[device_id] = module
        _FUNCTIONS[device_id] = fn
        return fn


def kernel_provenance() -> dict[str, Any]:
    return {
        "module": "maple_syrup.preferential_infiltration_cuda",
        "kernels": [KERNEL_NAME],
        "source_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
        "compile_options": list(COMPILE_OPTIONS),
        "threads_per_block": THREADS,
        "launches_per_step": LAUNCHES_PER_STEP,
        "launch_structure": "one thread per cell, grid ceil(n_cells/threads), layer loops inside the thread reading "
                            "the operator's device maps; no shared memory, no atomics, no fixed maximum layer count",
        "reductions": "CuPy reductions of kernel rows stacked into one packet; one counted host read per step",
        "fastmath": False,
        "libm": "device expm1 (may differ from NumPy by an ulp); declared bound rtol 2e-12 / atol 1e-14",
    }


def _launch(cp, fn, maps: KernelMaps, cell_tol: float, surface_in, water_in, active_u8, dt: float) -> StepArrays:
    n_layers, n_cells = int(water_in.shape[0]), int(water_in.shape[1])
    surface_out = cp.empty(n_cells, dtype=np.float64)
    water_out = cp.empty((n_layers, n_cells), dtype=np.float64)
    intake = cp.empty((n_layers, n_cells), dtype=np.float64)
    total = cp.empty(n_cells, dtype=np.float64)
    rows = cp.empty((ROW_COUNT, n_cells), dtype=np.float64)
    grid = ((n_cells + THREADS - 1) // THREADS,)
    fn(grid, (THREADS,), (
        np.int64(n_cells), np.int32(n_layers), surface_in, water_in, active_u8, np.float64(dt),
        maps.rate_s, maps.max_rate_m_s, maps.receiver, maps.saturation_m, maps.capacity_tolerance_m,
        np.float64(cell_tol), np.float64(NO_FLAG), surface_out, water_out, intake, total, rows))
    return StepArrays(surface_out, water_out, intake, total, rows)


def prepare_runner(maps: KernelMaps, cell_tol: float, cp, device_id: int):
    """Load the kernel for the current device and warm it up on one inactive cell (`maps` are the operator's private
    resident device arrays, computed by the build). Returns `(runner, info, preflight)`; the runner receives the
    private map handles per call; `preflight` refuses a step when the current device is not `device_id`."""
    if current_device_id(cp) != device_id:
        raise PreferentialBackendUnavailableError("prepare_runner must run on the column's device")
    for name, a in zip(KernelMaps._fields, maps, strict=True):
        if type(a) is not cp.ndarray or int(a.device.id) != device_id:
            raise PreferentialError(f"internal: map {name} is not a device array of device {device_id}")
    fn = load_kernel(cp, device_id)
    n_layers = int(maps.receiver.shape[0])
    warm = KernelMaps(cp.zeros(1), cp.zeros(1), cp.zeros((n_layers, 1)), maps.saturation_m, maps.capacity_tolerance_m)
    _launch(cp, fn, warm, cell_tol, cp.zeros(1), cp.zeros((n_layers, 1)), cp.zeros(1, np.uint8), 1.0)
    cp.cuda.Device().synchronize()

    def preflight() -> None:
        current = current_device_id(cp)
        if current != device_id:
            raise PreferentialError(f"the operator is bound to device {device_id} but the current device is {current}")

    def run(op, maps_internal, surface_in, water_in, dt):
        return _launch(cp, fn, maps_internal, cell_tol, surface_in, water_in, op.column.active_u8, dt)

    info = {"implementation": "cupy RawKernel: one thread per cell; private resident maps", "device_id": device_id,
            "launches_per_step": LAUNCHES_PER_STEP, "provenance": kernel_provenance(),
            "parameter_device_bytes": int(sum(int(a.nbytes) for a in maps))}
    return run, info, preflight
