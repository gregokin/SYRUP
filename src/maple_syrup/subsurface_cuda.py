"""CuPy RawKernel CUDA runner of the SYRUP subsurface column (`maple_syrup.subsurface`, backend "cuda").

One thread per column (cell), a short loop over the layers of that column inside the thread; the layer count is a
kernel argument, so there is no hard-coded maximum. The kernels evaluate EXACTLY the stages of
`subsurface.reference_step` in the same expression order with the same IEEE operations (compiled with the routing
options `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`, no fast math); the device `expm1` may differ
from NumPy's by an ulp and enters only the diffusive exchange amount. Declared agreement with the reference: rtol
2e-12 / atol 1e-14 on state and fluxes (bitwise where the diffusivity is 0).

Two launches per step, both on the current stream, in order:

  1. `maple_syrup_subsurface_factors`: one thread per interface computes the exchange factor
     `-expm1(-((G_k s_k) dt))` ON DEVICE into an `(max(L-1, 1),)` scratch array (the properties are COMMON to
     every cell in this first implementation, so the factor is evaluated once per interface per step, with the very
     expression order of the former per-cell form; skipped for a one-layer stack). In the donor-depth diffusion
     mode (`subsurface.DIFFUSION_MODES`, `directional=1`) the SAME single launch also writes the upward-donor factor
     from the distinct upward conductance into a second scratch array; in constant mode the second array and the
     upward conductance alias the first (`directional=0`, nothing extra is computed or allocated). No host field is
     uploaded, no per-layer launch, no shared memory, no layer cap.
  2. `maple_syrup_subsurface_step`: the column kernel reads the factor of the donor selected by the CURRENT pair
     gradient for every cell (in constant mode both reads hit the same array: the historical arithmetic bit for
     bit). With `bottom=0` (closed bottom, `subsurface.BOTTOM_DRAINAGE_MODES`) the cascade accumulates the deeper
     pore capacity bottom-up into the interface rows of `perc` (no per-thread scratch, no layer cap) before forwarding
     `min(excess, capacity)`; the bottom layer forwards nothing and `drain` is exactly 0. With `has_share != 0` the
     column kernel reads the caller's device-resident (N,) evaporation share and debits `min(demand, available) x
     share` after the moisture cap (unmet = demand x share - actual); `has_share == 0` runs the historical statements
     and receives the already resident `thickness_m` device buffer as the never-read share pointer (nothing extra is
     allocated; the parameter ledger `kernel_parameter_doubles` x 8 equals the resident device parameter bytes).

The factor scratch is returned inside `StepArrays.scratch` as an internal keep-alive only: `subsurface.step` holds
the arrays through the synchronising packet download, so the queued column kernel has finished reading the scratch
before Python can release it, and the public `SubsurfaceStep` never carries it (no history). Its O(L) bytes are part
of the CUDA memory estimate. The factor reuse is conditional on the current common properties; a per-cell
heterogeneous conductance map would need the per-cell evaluation inside the column kernel again.

Host traffic per step: ZERO uploads (scalars travel as kernel arguments; the column owns device copies of the layer
parameters from `prepare_column`), ONE counted device-to-host read of the stacked validation/budget packet
(`subsurface._read_packet`, CuPy reductions of the kernel-produced rows). No full-field download, no per-layer host
synchronisation. Every output array is freshly allocated per call on the column's device; inputs are never written.

CuPy, a visible device and loadable kernels are required (`require_cuda`); any failure is a
`SubsurfaceBackendUnavailableError`, never a NumPy fallback. Dynamic arrays must already live on the column's device
and be ordered with respect to the current stream. Nothing here was run by its author (file-only tools); Codex records
results.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

import numpy as np

from maple_syrup.routing_cuda import COMPILE_OPTIONS
from maple_syrup.subsurface import (
    NO_FLAG,
    ROW_COUNT,
    KernelParameters,
    SoilLayers,
    StepArrays,
    SubsurfaceBackendUnavailableError,
    SubsurfaceError,
    kernel_parameters,
)

__all__ = ["COMPILE_OPTIONS", "FACTOR_KERNEL_NAME", "KERNEL_NAME", "LAUNCHES_PER_STEP", "THREADS",
           "current_device_id", "default_device_available_bytes_fn", "effective_free_device_bytes",
           "kernel_provenance", "kernel_source", "load_kernels", "prepare_runner", "require_cuda"]

KERNEL_NAME = "maple_syrup_subsurface_step"
FACTOR_KERNEL_NAME = "maple_syrup_subsurface_factors"
THREADS = 128
LAUNCHES_PER_STEP = 2  # factor preparation (skipped for one layer) + column kernel

# Flag bits and stage order mirror subsurface.reference_step / subsurface_numba exactly.
_SOURCE = r"""
// Per-interface exchange factor(s), once per step: s = 1/dz_k + 1/dz_{k+1}; rate = G_k s; x = rate dt; -expm1(-x).
// directional == 0: one factor (cond_up / factor_up alias cond / factor and are not touched).
// directional != 0: the upward-donor factor from the distinct upward conductance into the second scratch array.
extern "C" __global__ void maple_syrup_subsurface_factors(
    const int n_if, const double* cond, const double* cond_up, const double* pair_inv, const double dt,
    const int directional, double* factor, double* factor_up)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_if) return;
    const double s = pair_inv[k];
    const double rate = cond[k] * s;
    const double x = rate * dt;
    factor[k] = -expm1(-x);
    if (directional != 0) {
        const double rate_up = cond_up[k] * s;
        const double x_up = rate_up * dt;
        factor_up[k] = -expm1(-x_up);
    }
}

extern "C" __global__ void maple_syrup_subsurface_step(
    const long long n, const int nl,
    const double* w_in, const double* add, const double* pet, const unsigned char* active,
    const double* dz, const double* res_m, const double* fc_m, const double* sat_m, const double* cap_tol,
    const double* factor, const double* pair_inv,
    const double dt, const double res_theta0, const double stress, const double fct, const int gravity,
    const double cell_tol, const double no_flag,
    const double* factor_up, const int bottom, const double* share, const int has_share,
    double* w_out, double* perc, double* diff, double* ret, double* drain, double* evap, double* unmet, double* rows)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    const int n_if = nl - 1;
    const bool a = active[i] != 0;
    unsigned int f = 0u;
    double before = 0.0;
    double added = 0.0;
    for (int l = 0; l < nl; ++l) {
        const long long p = (long long)l * n + i;
        const double wl = w_in[p];
        const double al = add[p];
        if (!isfinite(wl)) f |= 1u << 0;
        if (wl < 0.0) f |= 1u << 1;
        if (wl > sat_m[l] + cap_tol[l]) f |= 1u << 2;
        if (!isfinite(al)) f |= 1u << 3;
        if (al < 0.0) f |= 1u << 4;
        if ((!a) && al != 0.0) f |= 1u << 5;
        if (dt == 0.0 && al != 0.0) f |= 1u << 9;
        before = before + wl;
        added = added + al;
        w_out[p] = wl;
    }
    const double e_pot = pet[i];
    if (!isfinite(e_pot)) f |= 1u << 6;
    if (e_pot < 0.0) f |= 1u << 7;
    if ((!a) && e_pot != 0.0) f |= 1u << 8;
    bool share_ok = true;  // derived-output bits are withheld for a cell whose share is invalid (bits 15/16 name it)
    if (has_share != 0) {  // every cell, inactive included
        const double sv = share[i];
        if (!isfinite(sv)) { f |= 1u << 15; share_ok = false; }
        if (sv < 0.0 || sv > 1.0) { f |= 1u << 16; share_ok = false; }
    }
    for (int k = 0; k < n_if; ++k) {
        perc[(long long)k * n + i] = 0.0;
        diff[(long long)k * n + i] = 0.0;
    }
    double ret_v = 0.0, drain_v = 0.0, evap_v = 0.0, unmet_v = 0.0;
    double after = before;
    if (a && dt > 0.0) {
        // addition with saturation return
        for (int l = 0; l < nl; ++l) {
            const long long p = (long long)l * n + i;
            double w = w_out[p] + add[p];
            const double ex = w - sat_m[l];
            if (ex > 0.0) {
                w = w - ex;
                ret_v = ret_v + ex;
            }
            w_out[p] = w;
        }
        // gravity cascade, top to bottom within the step
        if (gravity != 0) {
            if (bottom == 0) {
                // closed bottom: deeper pore capacity (post-addition storage) accumulated bottom-up into the
                // interface rows of perc; exactly 0 beneath the bottom layer; overwritten by the forward amount
                double capacity = 0.0;
                for (int k = n_if - 1; k >= 0; --k) {
                    double term = sat_m[k + 1] - w_out[(long long)(k + 1) * n + i];
                    if (term < 0.0) term = 0.0;
                    capacity = capacity + term;
                    perc[(long long)k * n + i] = capacity;
                }
            }
            double carry = 0.0;
            for (int l = 0; l < nl; ++l) {
                const long long p = (long long)l * n + i;
                double w = w_out[p] + carry;
                const double ex = w - fc_m[l];
                if (ex > 0.0) {
                    double fwd;
                    if (bottom != 0) {
                        fwd = ex;
                    } else if (l < n_if) {
                        const double cap = perc[p];
                        fwd = ex;
                        if (!(ex < cap)) fwd = cap;
                    } else {
                        fwd = 0.0;
                    }
                    w = w - fwd;
                    carry = fwd;
                } else {
                    carry = 0.0;
                }
                w_out[p] = w;
                if (l < n_if) perc[p] = carry;
            }
            drain_v = carry;  // exactly 0 with a closed bottom
        }
        // evaporation from layer 0
        {
            const double w0 = w_out[i];
            const double theta0 = w0 / dz[0];
            double m;
            if (theta0 <= res_theta0) {
                m = 0.0;
            } else if (theta0 < stress) {
                const double r = theta0 / stress;
                m = r * r;
            } else {
                m = 1.0;
            }
            const double demand = ((fct * e_pot) * dt) * m;
            double avail0 = w0 - res_m[0];
            if (avail0 < 0.0) avail0 = 0.0;
            double limited;
            if (demand <= avail0) limited = demand; else limited = avail0;
            if (has_share != 0) {
                // post-cap share: actual E = limited x s, unmet = demand x s - actual E; the (1 - s) remainder of
                // `limited` stays in the layer for LEAF's separate transpiration request
                const double sv = share[i];
                evap_v = limited * sv;
                w_out[i] = w0 - evap_v;
                unmet_v = demand * sv - evap_v;
            } else {
                evap_v = limited;
                w_out[i] = w0 - evap_v;
                unmet_v = demand - evap_v;
            }
        }
        // sequential pair diffusion from the top; the factor(s) were prepared once per interface for this step.
        // The donor is selected by the CURRENT pair gradient g (after the earlier pairs of this step); factor_up
        // aliases factor in constant mode, so the arithmetic is the historical ((ti - tj) / s) e bit for bit.
        for (int k = 0; k < n_if; ++k) {
            const long long pi = (long long)k * n + i;
            const long long pj = (long long)(k + 1) * n + i;
            const double wi = w_out[pi];
            const double wj = w_out[pj];
            const double ti = wi / dz[k];
            const double tj = wj / dz[k + 1];
            const double s = pair_inv[k];
            const double g = (ti - tj) / s;
            const double e = (g > 0.0) ? factor[k] : factor_up[k];
            const double amount = g * e;
            bool down;
            double amt, wd, wr, resd, satr;
            if (amount > 0.0) {
                down = true; amt = amount; wd = wi; wr = wj; resd = res_m[k]; satr = sat_m[k + 1];
            } else {
                down = false; amt = -amount; wd = wj; wr = wi; resd = res_m[k + 1]; satr = sat_m[k];
            }
            double avail = wd - resd;
            if (avail < 0.0) avail = 0.0;
            double cap = satr - wr;
            if (cap < 0.0) cap = 0.0;
            double act = amt;
            if (avail < act) act = avail;
            if (cap < act) act = cap;
            if (down) {
                w_out[pi] = wi - act;
                w_out[pj] = wj + act;
                diff[pi] = act;
            } else {
                w_out[pj] = wj - act;
                w_out[pi] = wi + act;
                diff[pi] = -act;
            }
        }
        // output checks (collected in fo, applied only when the share is valid)
        unsigned int fo = 0u;
        after = 0.0;
        for (int l = 0; l < nl; ++l) {
            const double w = w_out[(long long)l * n + i];
            if (!isfinite(w)) fo |= 1u << 10;
            if (w < 0.0) fo |= 1u << 11;
            if (w > sat_m[l] + cap_tol[l]) fo |= 1u << 12;
            after = after + w;
        }
        const double closure = (after - before) - (((added - ret_v) - drain_v) - evap_v);
        const double scale = ((((before + after) + added) + ret_v) + drain_v) + evap_v;
        if (fabs(closure) > cell_tol * scale) fo |= 1u << 13;
        bool nonfinite = (!isfinite(ret_v)) || (!isfinite(drain_v)) || (!isfinite(evap_v)) || (!isfinite(unmet_v))
                         || (!isfinite(closure));
        for (int k = 0; k < n_if; ++k) {
            const long long p = (long long)k * n + i;
            if ((!isfinite(perc[p])) || (!isfinite(diff[p]))) nonfinite = true;
        }
        if (nonfinite) fo |= 1u << 14;
        if (share_ok) f |= fo;
    }
    ret[i] = ret_v;
    drain[i] = drain_v;
    evap[i] = evap_v;
    unmet[i] = unmet_v;
    rows[i] = a ? before : 0.0;
    rows[n + i] = a ? after : 0.0;
    rows[2 * n + i] = a ? added : 0.0;
    rows[3 * n + i] = (f == 0u) ? no_flag : (double)(f & (0u - f));
}
"""

_MODULES: dict[int, Any] = {}
_FUNCTIONS: dict[int, tuple[Any, Any]] = {}
_LOCK = threading.Lock()


def kernel_source() -> str:
    return _SOURCE


def require_cuda():
    """The CuPy module with at least one visible device, through MAPLE's own probes; otherwise a named setup error."""
    from maple.core.backend import cuda_device_count, cupy_import_error, cupy_module

    cp = cupy_module()
    if cp is None:
        raise SubsurfaceBackendUnavailableError(
            f"backend 'cuda' requires CuPy, which is not usable ({cupy_import_error() or 'not installed'}); there is "
            "no fallback to the NumPy reference")
    if cuda_device_count() < 1:
        raise SubsurfaceBackendUnavailableError(
            "backend 'cuda' requires a visible CUDA device (device count 0); there is no fallback to the NumPy reference")
    return cp


def current_device_id(cp) -> int:
    try:
        return int(cp.cuda.Device().id)
    except Exception as exc:  # reported as a setup failure
        raise SubsurfaceBackendUnavailableError(f"CUDA device query failed ({type(exc).__name__}: {exc})") from exc


def load_kernels(cp, device_id: int) -> tuple[Any, Any]:
    """Compile/load both kernels for `device_id` (once per process and device): `(factor_kernel, step_kernel)`.
    Failure is a setup error."""
    with _LOCK:
        fns = _FUNCTIONS.get(device_id)
        if fns is not None:
            return fns
        try:
            module = cp.RawModule(code=_SOURCE, options=COMPILE_OPTIONS, backend="nvrtc")
            factor_fn = module.get_function(FACTOR_KERNEL_NAME)
            step_fn = module.get_function(KERNEL_NAME)
            for fn in (factor_fn, step_fn):
                attrs = dict(fn.attributes)  # forces the module load on this device
                if int(attrs.get("max_threads_per_block", 0)) < THREADS:
                    raise SubsurfaceBackendUnavailableError(
                        f"kernel supports only {attrs.get('max_threads_per_block')} threads per block, {THREADS} needed")
        except SubsurfaceBackendUnavailableError:
            raise
        except Exception as exc:
            raise SubsurfaceBackendUnavailableError(
                f"the CUDA subsurface kernels failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        _MODULES[device_id] = module
        _FUNCTIONS[device_id] = (factor_fn, step_fn)
        return factor_fn, step_fn


def kernel_provenance() -> dict[str, Any]:
    return {
        "module": "maple_syrup.subsurface_cuda",
        "kernels": [FACTOR_KERNEL_NAME, KERNEL_NAME],
        "source_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
        "compile_options": list(COMPILE_OPTIONS),
        "threads_per_block": THREADS,
        "launches_per_step": LAUNCHES_PER_STEP,
        "launch_structure": "1) factor kernel: one thread per interface, grid ceil((L-1)/threads), writes the "
                            "per-interface -expm1(-((G s) dt)) scratch on device (skipped for one layer; in the "
                            "donor-depth mode the same launch also writes the upward-donor factor); "
                            "2) column kernel: one thread per column, grid ceil(n_cells/threads), layer loops inside "
                            "the thread reading that scratch; no shared memory, no atomics, no fixed maximum layer count",
        "factor_scratch": "(max(L-1,1),) float64 on device per step per factor array (one in constant mode, two in "
                          "donor-depth mode), kept alive through the packet download (StepArrays.scratch), never "
                          "published; reuse is conditional on common layer properties",
        "diffusion_modes": "constant (common D, historical arithmetic bit for bit) | tietjen_donor_depth (D = d K "
                           "h_donor, donor selected by the current pair gradient)",
        "bottom_drainage_modes": "open (historical) | closed (capacity-limited internal cascade, drain exactly 0)",
        "evaporation_share": "optional caller (N,) device field (has_share=1): actual E = min(demand, available) x share "
                             "after the moisture cap, unmet = demand x share - actual; has_share=0 runs the historical "
                             "statements and passes the already resident thickness_m parameter buffer as the never-read "
                             "share pointer (no extra allocation, parameter ledger unchanged)",
        "reductions": "CuPy reductions of kernel rows stacked into one packet; one counted host read per step",
        "fastmath": False,
        "libm": "device expm1 (may differ from NumPy by an ulp); declared bound rtol 2e-12 / atol 1e-14",
    }


def effective_free_device_bytes(driver_free_bytes: int, pool_free_bytes: int, pool_limit_bytes: int,
                                pool_used_bytes: int) -> int:
    """`driver_free + pool_free`, limited to `pool_limit - pool_used` when the pool has a positive limit; never negative.
    The same four-number arithmetic as `storage_admission.effective_free_device_bytes` (kept local so this module does
    not import the bed-storage layer)."""
    for name, value in (("driver_free_bytes", driver_free_bytes), ("pool_free_bytes", pool_free_bytes),
                        ("pool_limit_bytes", pool_limit_bytes), ("pool_used_bytes", pool_used_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SubsurfaceError(f"{name} must be a nonnegative int, got {value!r}")
    reachable = driver_free_bytes + pool_free_bytes
    if pool_limit_bytes > 0:
        reachable = min(reachable, pool_limit_bytes - pool_used_bytes)
    return max(0, reachable)


def default_device_available_bytes_fn(cp, device_id: int):
    """Default admission reading for the column's device: driver free memory plus the default pool's free bytes under
    the pool limit if one is set. Refuses (before reading) when the current device is not the column's device; never
    changes the device or any limit."""

    def query() -> int:
        current = current_device_id(cp)
        if current != device_id:
            raise SubsurfaceError(f"the column is bound to device {device_id} but the current device is {current}; "
                                  "select the column's device or inject available_bytes_fn")
        pool = cp.get_default_memory_pool()
        return effective_free_device_bytes(int(cp.cuda.Device().mem_info[0]), int(pool.free_bytes()),
                                           int(pool.get_limit()), int(pool.used_bytes()))

    return query


def _launch(cp, factor_fn, step_fn, dev: dict, params: KernelParameters, water_in, added, pet, active_u8,
            dt: float, share=None) -> StepArrays:
    n_layers, n_cells = int(water_in.shape[0]), int(water_in.shape[1])
    n_if = n_layers - 1
    directional = int(params.directional)
    # has_share == 0: the kernel never reads the share pointer, so the ALREADY resident thickness_m device buffer
    # (length L >= 1, FP64) is passed; nothing extra is allocated and the parameter ledger stays exact
    share_arr, has_share = (dev["thickness_m"], 0) if share is None else (share, 1)
    factor = cp.empty(max(n_if, 1), dtype=np.float64)  # O(L) device scratch, kept alive through the packet read
    factor_up = cp.empty(max(n_if, 1), dtype=np.float64) if directional else factor  # alias in constant mode
    if n_if > 0:
        factor_fn(((n_if + THREADS - 1) // THREADS,), (THREADS,), (
            np.int32(n_if), dev["interface_conductance_m_s"], dev["interface_conductance_up_m_s"],
            dev["pair_inverse_thickness"], np.float64(dt), np.int32(directional), factor, factor_up))
    water_out = cp.empty((n_layers, n_cells), dtype=np.float64)
    percolation = cp.empty((n_if, n_cells), dtype=np.float64)
    diffusion = cp.empty((n_if, n_cells), dtype=np.float64)
    ret = cp.empty(n_cells, dtype=np.float64)
    drain = cp.empty(n_cells, dtype=np.float64)
    evap = cp.empty(n_cells, dtype=np.float64)
    unmet = cp.empty(n_cells, dtype=np.float64)
    rows = cp.empty((ROW_COUNT, n_cells), dtype=np.float64)
    grid = ((n_cells + THREADS - 1) // THREADS,)
    step_fn(grid, (THREADS,), (
        np.int64(n_cells), np.int32(n_layers), water_in, added, pet, active_u8,
        dev["thickness_m"], dev["residual_m"], dev["field_capacity_m"], dev["saturation_m"], dev["capacity_tolerance_m"],
        factor, dev["pair_inverse_thickness"],
        np.float64(dt), np.float64(params.residual_theta0), np.float64(params.theta_evaporation_stress),
        np.float64(params.evaporation_factor), np.int32(params.gravity_code),
        np.float64(params.cell_closure_coefficient), np.float64(NO_FLAG),
        factor_up, np.int32(params.bottom_code), share_arr, np.int32(has_share),
        water_out, percolation, diffusion, ret, drain, evap, unmet, rows))
    return StepArrays(water_out, percolation, diffusion, ret, drain, evap, unmet, rows, scratch=(factor, factor_up))


def prepare_runner(layers: SoilLayers, cp, device_id: int):
    """Load the kernels for the current device and upload owned copies of the layer parameters (counted transfers).
    Returns `(runner, info, preflight)`; `preflight` refuses a step when the current device is not `device_id`."""
    if not isinstance(layers, SoilLayers):
        raise SubsurfaceError("layers must be a SoilLayers")
    if current_device_id(cp) != device_id:
        raise SubsurfaceBackendUnavailableError("prepare_runner must run on the column's device")
    from maple.core.backend import to_device

    factor_fn, step_fn = load_kernels(cp, device_id)
    params = kernel_parameters(layers)
    dev = {name: cp.ascontiguousarray(to_device(np.ascontiguousarray(getattr(params, name)), cp))
           for name in ("thickness_m", "residual_m", "field_capacity_m", "saturation_m", "capacity_tolerance_m",
                        "interface_conductance_m_s", "pair_inverse_thickness")}
    if params.directional:
        dev["interface_conductance_up_m_s"] = cp.ascontiguousarray(
            to_device(np.ascontiguousarray(params.interface_conductance_up_m_s), cp))
    else:
        dev["interface_conductance_up_m_s"] = dev["interface_conductance_m_s"]  # alias: no second upload, no bytes

    def preflight() -> None:
        current = current_device_id(cp)
        if current != device_id:
            raise SubsurfaceError(f"the column is bound to device {device_id} but the current device is {current}")

    def run(column, water_in, added, pet, dt, share=None):
        return _launch(cp, factor_fn, step_fn, dev, params, water_in, added, pet, column.active_u8, dt, share)

    factor_arrays = 2 if params.directional else 1
    info = {"implementation": "cupy RawKernel: per-interface factor kernel then one thread per column",
            "device_id": device_id, "launches_per_step": LAUNCHES_PER_STEP if layers.n_layers > 1 else 1,
            "provenance": kernel_provenance(),
            "diffusion_mode": layers.diffusion_mode,
            "bottom_drainage": layers.bottom_drainage,
            "factor_arrays_per_step": factor_arrays,
            # aliased buffers (constant mode) are counted once
            "parameter_device_bytes": int(sum(int(a.nbytes) for a in {id(a): a for a in dev.values()}.values())),
            "factor_scratch_device_bytes": factor_arrays * max(layers.n_layers - 1, 1) * 8,
            "parameter_host_bytes": int(sum(int(a.nbytes) for a in {id(getattr(params, name)): getattr(params, name)
                                                                    for name in params._fields
                                                                    if hasattr(getattr(params, name), "nbytes")}.values()))}
    return run, info, preflight
