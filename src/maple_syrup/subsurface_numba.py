"""Compiled Numba CPU runner of the SYRUP subsurface column (`maple_syrup.subsurface`, backend "numba").

The kernel evaluates EXACTLY the stages of `subsurface.reference_step`, term for term in the same expression order,
as serial cell / layer loops: no fastmath, no prange, no changed order, no clipping. Identical IEEE operations give
identical bits; `math.expm1` (LLVM/libm) may differ from NumPy's `expm1` by an ulp, which only enters the diffusive
exchange amount. The declared agreement bound with the reference is therefore rtol 2e-12 / atol 1e-14 on the state
and fluxes (bitwise where the diffusivity is 0).

Constant-factor optimisation: the per-interface factor `-expm1(-((G_k s_k) dt))` depends only on the common layer
properties and `dt`, so it is evaluated once per interface per call into a local `(L - 1,)` array before the cell
loop (same expression order, same libm `expm1`) and reused by every cell. There is no cache across calls, so a
changed `dt` can never see a stale factor. This reuse is conditional on the CURRENT common properties; per-cell
heterogeneous conductance maps would need the per-cell evaluation again.

Numba is imported lazily; `prepare_runner` compiles the kernel once (a one-cell warm-up call, so the cost sits at
`prepare_column`, outside steps). Missing Numba is a `SubsurfaceBackendUnavailableError`; there is no fallback to the
reference. Every output array is freshly allocated per call; the kernel reads the caller's arrays and never writes
them. Nothing here was run by its author (file-only tools); Codex records results.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

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

__all__ = ["compiled_kernel", "numba_available", "numba_versions", "prepare_runner", "require_numba", "reset_compiled"]

_KERNEL: Any = None
_LOCK = threading.Lock()


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def numba_versions() -> dict[str, str] | None:
    try:
        import llvmlite
        import numba
    except ImportError:
        return None
    return {"numba": numba.__version__, "llvmlite": llvmlite.__version__}


def require_numba() -> None:
    if not numba_available():
        raise SubsurfaceBackendUnavailableError(
            "backend 'numba' requires Numba (optional extra maple-syrup[numba]), which is not importable; there is no "
            "fallback to the NumPy reference (select backend='numpy' explicitly)")


def _build(numba: Any):
    jit = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")

    @jit
    def step_kernel(w_in, add, pet, active, dt, dz, res_m, fc_m, sat_m, cap_tol, cond, pair_inv, res_theta0, stress,
                    factor, gravity, cell_tol, no_flag, w_out, perc, diff, ret, drain, evap, unmet, rows):
        n_layers = w_in.shape[0]
        n_cells = w_in.shape[1]
        n_if = n_layers - 1
        # Per-interface exchange factor -expm1(-((G s) dt)): the properties are COMMON to every cell in this first
        # implementation, so the factor is evaluated once per interface per call (a local array, never cached across
        # calls) with the very expression order of the per-cell form, then reused for all cells. A later per-cell
        # heterogeneous conductance map would have to move this back inside the cell loop.
        relaxation_factors = np.empty(n_if, dtype=np.float64)  # distinct from the scalar evaporation `factor`
        for k in range(n_if):
            s = pair_inv[k]
            rate = cond[k] * s
            x = rate * dt
            relaxation_factors[k] = -math.expm1(-x)
        for i in range(n_cells):
            a = active[i] != 0
            f = 0
            before = 0.0
            added = 0.0
            for l in range(n_layers):
                wl = w_in[l, i]
                al = add[l, i]
                if not math.isfinite(wl):
                    f |= 1 << 0
                if wl < 0.0:
                    f |= 1 << 1
                if wl > sat_m[l] + cap_tol[l]:
                    f |= 1 << 2
                if not math.isfinite(al):
                    f |= 1 << 3
                if al < 0.0:
                    f |= 1 << 4
                if (not a) and al != 0.0:
                    f |= 1 << 5
                if dt == 0.0 and al != 0.0:
                    f |= 1 << 9
                before = before + wl
                added = added + al
                w_out[l, i] = wl
            e_pot = pet[i]
            if not math.isfinite(e_pot):
                f |= 1 << 6
            if e_pot < 0.0:
                f |= 1 << 7
            if (not a) and e_pot != 0.0:
                f |= 1 << 8
            for k in range(n_if):
                perc[k, i] = 0.0
                diff[k, i] = 0.0
            ret_v = 0.0
            drain_v = 0.0
            evap_v = 0.0
            unmet_v = 0.0
            after = before
            if a and dt > 0.0:
                # addition with saturation return
                for l in range(n_layers):
                    w = w_out[l, i] + add[l, i]
                    ex = w - sat_m[l]
                    if ex > 0.0:
                        w = w - ex
                        ret_v = ret_v + ex
                    w_out[l, i] = w
                # gravity cascade, top to bottom within the step
                if gravity != 0:
                    carry = 0.0
                    for l in range(n_layers):
                        w = w_out[l, i] + carry
                        ex = w - fc_m[l]
                        if ex > 0.0:
                            w = w - ex
                            carry = ex
                        else:
                            carry = 0.0
                        w_out[l, i] = w
                        if l < n_if:
                            perc[l, i] = carry
                    drain_v = carry
                # evaporation from layer 0
                w0 = w_out[0, i]
                theta0 = w0 / dz[0]
                if theta0 <= res_theta0:
                    m = 0.0
                elif theta0 < stress:
                    r = theta0 / stress
                    m = r * r
                else:
                    m = 1.0
                demand = ((factor * e_pot) * dt) * m
                avail0 = w0 - res_m[0]
                if avail0 < 0.0:  # noqa: PLR1730 - explicit ordered FP comparison, NaN-propagating; no max()
                    avail0 = 0.0
                if demand <= avail0:
                    evap_v = demand
                else:
                    evap_v = avail0
                w_out[0, i] = w0 - evap_v
                unmet_v = demand - evap_v
                # sequential pair diffusion from the top
                for k in range(n_if):
                    wi = w_out[k, i]
                    wj = w_out[k + 1, i]
                    ti = wi / dz[k]
                    tj = wj / dz[k + 1]
                    s = pair_inv[k]
                    e = relaxation_factors[k]
                    amount = ((ti - tj) / s) * e
                    if amount > 0.0:
                        down = True
                        amt = amount
                        wd = wi
                        wr = wj
                        resd = res_m[k]
                        satr = sat_m[k + 1]
                    else:
                        down = False
                        amt = -amount
                        wd = wj
                        wr = wi
                        resd = res_m[k + 1]
                        satr = sat_m[k]
                    avail = wd - resd
                    if avail < 0.0:  # noqa: PLR1730 - explicit ordered FP comparison, NaN-propagating; no max()
                        avail = 0.0
                    cap = satr - wr
                    if cap < 0.0:  # noqa: PLR1730 - explicit ordered FP comparison, NaN-propagating; no max()
                        cap = 0.0
                    act = amt
                    if avail < act:  # noqa: PLR1730 - explicit ordered FP comparison, NaN-propagating; no min()
                        act = avail
                    if cap < act:  # noqa: PLR1730 - explicit ordered FP comparison, NaN-propagating; no min()
                        act = cap
                    if down:
                        w_out[k, i] = wi - act
                        w_out[k + 1, i] = wj + act
                        diff[k, i] = act
                    else:
                        w_out[k + 1, i] = wj - act
                        w_out[k, i] = wi + act
                        diff[k, i] = -act
                # output checks
                after = 0.0
                for l in range(n_layers):
                    w = w_out[l, i]
                    if not math.isfinite(w):
                        f |= 1 << 10
                    if w < 0.0:
                        f |= 1 << 11
                    if w > sat_m[l] + cap_tol[l]:
                        f |= 1 << 12
                    after = after + w
                closure = (after - before) - (((added - ret_v) - drain_v) - evap_v)
                scale = ((((before + after) + added) + ret_v) + drain_v) + evap_v
                if abs(closure) > cell_tol * scale:
                    f |= 1 << 13
                nonfinite = ((not math.isfinite(ret_v)) or (not math.isfinite(drain_v)) or (not math.isfinite(evap_v))
                             or (not math.isfinite(unmet_v)) or (not math.isfinite(closure)))
                for k in range(n_if):
                    if (not math.isfinite(perc[k, i])) or (not math.isfinite(diff[k, i])):
                        nonfinite = True
                if nonfinite:
                    f |= 1 << 14
            ret[i] = ret_v
            drain[i] = drain_v
            evap[i] = evap_v
            unmet[i] = unmet_v
            if a:
                rows[0, i] = before
                rows[1, i] = after
                rows[2, i] = added
            else:
                rows[0, i] = 0.0
                rows[1, i] = 0.0
                rows[2, i] = 0.0
            if f == 0:
                rows[3, i] = no_flag
            else:
                rows[3, i] = float(f & -f)

    return step_kernel


def compiled_kernel():
    """The process-wide compiled kernel (built lazily; Numba imported here only)."""
    global _KERNEL
    if _KERNEL is None:
        require_numba()
        import numba

        with _LOCK:
            if _KERNEL is None:
                _KERNEL = _build(numba)
    return _KERNEL


def reset_compiled() -> None:
    """Drop the compiled kernel (tests of the compile path only)."""
    global _KERNEL
    with _LOCK:
        _KERNEL = None


def _launch(kernel, params: KernelParameters, water_in: np.ndarray, added: np.ndarray, pet: np.ndarray,
            active_u8: np.ndarray, dt: float) -> StepArrays:
    n_layers, n_cells = water_in.shape
    n_if = n_layers - 1
    water_out = np.empty((n_layers, n_cells), dtype=np.float64)
    percolation = np.empty((n_if, n_cells), dtype=np.float64)
    diffusion = np.empty((n_if, n_cells), dtype=np.float64)
    ret = np.empty(n_cells, dtype=np.float64)
    drain = np.empty(n_cells, dtype=np.float64)
    evap = np.empty(n_cells, dtype=np.float64)
    unmet = np.empty(n_cells, dtype=np.float64)
    rows = np.empty((ROW_COUNT, n_cells), dtype=np.float64)
    kernel(water_in, added, pet, active_u8, float(dt), params.thickness_m, params.residual_m, params.field_capacity_m,
           params.saturation_m, params.capacity_tolerance_m, params.interface_conductance_m_s,
           params.pair_inverse_thickness, params.residual_theta0, params.theta_evaporation_stress,
           params.evaporation_factor, params.gravity_code, params.cell_closure_coefficient, NO_FLAG,
           water_out, percolation, diffusion, ret, drain, evap, unmet, rows)
    return StepArrays(water_out, percolation, diffusion, ret, drain, evap, unmet, rows)


def prepare_runner(layers: SoilLayers):
    """Compile (if needed) and warm up the kernel for `layers`; return `(runner, info)`. The runner allocates fresh
    outputs per call and never writes an input."""
    if not isinstance(layers, SoilLayers):
        raise SubsurfaceError("layers must be a SoilLayers")
    kernel = compiled_kernel()
    params = kernel_parameters(layers)
    n_layers = layers.n_layers
    # Warm-up on one inactive cell with zero state: compiles the single (float64 C arrays) specialisation here.
    _launch(kernel, params, np.zeros((n_layers, 1)), np.zeros((n_layers, 1)), np.zeros(1), np.zeros(1, np.uint8), 1.0)

    def run(column, water_in, added, pet, dt):
        return _launch(kernel, params, water_in, added, pet, column.active_u8, dt)

    info = {"implementation": "numba serial cell/layer loops, fastmath=False; per-interface expm1 factor evaluated "
                              "once per call before the cell loop (common properties), local array of L-1 doubles",
            "versions": numba_versions(),
            "expm1": "LLVM/libm expm1 (may differ from NumPy by an ulp); declared bound rtol 2e-12 / atol 1e-14",
            "parameter_host_bytes": int(sum(int(getattr(params, name).nbytes) for name in params._fields
                                            if hasattr(getattr(params, name), "nbytes")))}
    return run, info
