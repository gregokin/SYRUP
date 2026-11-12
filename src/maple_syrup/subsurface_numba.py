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

Phase 1 extensions (`subsurface.DIFFUSION_MODES`, `subsurface.BOTTOM_DRAINAGE_MODES`): in the donor-depth
diffusion mode a SECOND local `(L - 1,)` factor array holds the upward-donor factor (`directional=1`, distinct
conductance array); the per-cell exchange reads the current pair gradient `g = (ti - tj) / s` and multiplies it by
the factor of the donor that gradient selects. In constant mode the second array IS the first (no allocation) and
the arithmetic is the historical `((ti - tj) / s) e`, bit for bit. With `bottom_code=0` (closed bottom) the cascade
first accumulates the deeper pore capacity bottom-up into the interface rows of `perc` (no per-thread scratch, no
layer cap) and then forwards `min(excess, capacity)`; the bottom layer forwards nothing and `drain` is exactly 0.
Optional post-cap evaporation share (`has_share != 0`, a caller (N,) field): the stage keeps the historical
`limited = min(demand, available)` and debits `limited x share`, reporting `demand x share - actual` as unmet; with
`has_share == 0` the historical statements run in their historical order; the share pointer then aliases the
already resident `thickness_m` parameter buffer (never read, nothing allocated, the parameter ledger is unchanged).

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
    cell_closure_coefficient,
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
                    factor, gravity, cell_tol, no_flag, cond_up, directional, bottom, share, has_share, pulse, pulse_on, w_out, perc,
                    diff, ret, drain, evap, unmet, rows):
        n_layers = w_in.shape[0]
        n_cells = w_in.shape[1]
        n_if = n_layers - 1
        # Per-interface exchange factor -expm1(-((G s) dt)): the properties are COMMON to every cell in this first
        # implementation, so the factor is evaluated once per interface per call (a local array, never cached across
        # calls) with the very expression order of the per-cell form, then reused for all cells. A later per-cell
        # heterogeneous conductance map would have to move this back inside the cell loop. In the donor-depth mode
        # (directional != 0) a second local array holds the upward-donor factor; otherwise it aliases the first.
        relaxation_factors = np.empty(n_if, dtype=np.float64)  # distinct from the scalar evaporation `factor`
        if directional != 0:
            relaxation_factors_up = np.empty(n_if, dtype=np.float64)
        else:
            relaxation_factors_up = relaxation_factors
        for k in range(n_if):
            s = pair_inv[k]
            rate = cond[k] * s
            x = rate * dt
            relaxation_factors[k] = -math.expm1(-x)
            if directional != 0:
                rate_up = cond_up[k] * s
                x_up = rate_up * dt
                relaxation_factors_up[k] = -math.expm1(-x_up)
        for i in range(n_cells):
            a = active[i] != 0
            f = 0
            before = 0.0
            added = 0.0
            for l in range(n_layers):
                wl = w_in[l, i]
                if not math.isfinite(wl):
                    f |= 1 << 0
                if wl < 0.0:
                    f |= 1 << 1
                if wl > sat_m[l] + cap_tol[l]:
                    f |= 1 << 2
                if pulse_on == 0:  # the pulse-aware step has no added-water tensor (the pointer is never read then)
                    al = add[l, i]
                    if not math.isfinite(al):
                        f |= 1 << 3
                    if al < 0.0:
                        f |= 1 << 4
                    if (not a) and al != 0.0:
                        f |= 1 << 5
                    if dt == 0.0 and al != 0.0:
                        f |= 1 << 9
                    added = added + al
                before = before + wl
                w_out[l, i] = wl
            e_pot = pet[i]
            if not math.isfinite(e_pot):
                f |= 1 << 6
            if e_pot < 0.0:
                f |= 1 << 7
            if (not a) and e_pot != 0.0:
                f |= 1 << 8
            share_ok = True  # derived-output bits are withheld for a cell whose share is invalid (bits 15/16 name it)
            if has_share != 0:  # every cell, inactive included
                sv = share[i]
                if not math.isfinite(sv):
                    f |= 1 << 15
                    share_ok = False
                if sv < 0.0 or sv > 1.0:
                    f |= 1 << 16
                    share_ok = False
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
                    if pulse_on == 0:
                        w = w_out[l, i] + add[l, i]
                    else:
                        w = w_out[l, i] + 0.0  # zero added water: the historical return statements still run
                    ex = w - sat_m[l]
                    if ex > 0.0:
                        w = w - ex
                        ret_v = ret_v + ex
                    w_out[l, i] = w
                if pulse_on != 0:  # Phase 1.5: the uniform pulse enters the top layer with NO saturation return
                    w_out[0, i] = w_out[0, i] + pulse
                    added = added + pulse
                # gravity cascade, top to bottom within the step
                if gravity != 0:
                    if bottom == 0:
                        # closed bottom: deeper pore capacity (post-addition storage) accumulated bottom-up into the
                        # interface rows of perc; exactly 0 beneath the bottom layer; overwritten by the forward amount
                        capacity = 0.0
                        for k in range(n_if - 1, -1, -1):
                            term = sat_m[k + 1] - w_out[k + 1, i]
                            if term < 0.0:  # noqa: PLR1730 - explicit ordered FP comparison, NaN-propagating; no max()
                                term = 0.0
                            capacity = capacity + term
                            perc[k, i] = capacity
                    carry = 0.0
                    for l in range(n_layers):
                        w = w_out[l, i] + carry
                        ex = w - fc_m[l]
                        if ex > 0.0:
                            if bottom != 0:
                                fwd = ex
                            elif l < n_if:
                                cap = perc[l, i]
                                fwd = ex
                                if not (ex < cap):
                                    fwd = cap
                            else:
                                fwd = 0.0
                            w = w - fwd
                            carry = fwd
                        else:
                            carry = 0.0
                        w_out[l, i] = w
                        if l < n_if:
                            perc[l, i] = carry
                    drain_v = carry  # exactly 0 with a closed bottom
                fit_ok = True
                if pulse_on != 0:  # fit test right after the ONE movement, BEFORE evaporation/diffusion (also gravity off)
                    for l in range(n_layers):
                        if w_out[l, i] > sat_m[l] + cap_tol[l]:
                            fit_ok = False
                    if not fit_ok:
                        f |= 1 << 17
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
                    limited = demand
                else:
                    limited = avail0
                if has_share != 0:
                    # post-cap share: actual E = limited x s, unmet = demand x s - actual E (partitioned potential
                    # minus actual); the (1 - s) remainder of `limited` stays for LEAF's separate T request
                    sv = share[i]
                    evap_v = limited * sv
                    w_out[0, i] = w0 - evap_v
                    unmet_v = demand * sv - evap_v
                else:
                    evap_v = limited
                    w_out[0, i] = w0 - evap_v
                    unmet_v = demand - evap_v
                # sequential pair diffusion from the top
                for k in range(n_if):
                    wi = w_out[k, i]
                    wj = w_out[k + 1, i]
                    ti = wi / dz[k]
                    tj = wj / dz[k + 1]
                    s = pair_inv[k]
                    g = (ti - tj) / s  # the CURRENT pair gradient after the earlier pairs of this step
                    if g > 0.0:
                        e = relaxation_factors[k]  # upper layer donates (the common factor in constant mode)
                    else:
                        e = relaxation_factors_up[k]  # lower layer donates (aliases the common factor in constant mode)
                    amount = g * e
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
                # output checks (collected in fo, applied only when the share is valid)
                fo = 0
                after = 0.0
                for l in range(n_layers):
                    w = w_out[l, i]
                    if not math.isfinite(w):
                        fo |= 1 << 10
                    if w < 0.0:
                        fo |= 1 << 11
                    if w > sat_m[l] + cap_tol[l]:
                        fo |= 1 << 12
                    after = after + w
                closure = (after - before) - (((added - ret_v) - drain_v) - evap_v)
                scale = ((((before + after) + added) + ret_v) + drain_v) + evap_v
                if abs(closure) > cell_tol * scale:
                    fo |= 1 << 13
                nonfinite = ((not math.isfinite(ret_v)) or (not math.isfinite(drain_v)) or (not math.isfinite(evap_v))
                             or (not math.isfinite(unmet_v)) or (not math.isfinite(closure)))
                for k in range(n_if):
                    if (not math.isfinite(perc[k, i])) or (not math.isfinite(diff[k, i])):
                        nonfinite = True
                if nonfinite:
                    fo |= 1 << 14
                if share_ok and fit_ok:  # a fit-failed cell keeps bit 17 as the named cause
                    f |= fo
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


def _launch(kernel, params: KernelParameters, water_in: np.ndarray, added: np.ndarray | None, pet: np.ndarray,
            active_u8: np.ndarray, dt: float, share: np.ndarray | None = None, recharge: float | None = None) -> StepArrays:
    n_layers, n_cells = water_in.shape
    # Phase 1.5 pulse-aware step: no added-water tensor (an ALREADY resident parameter buffer is passed as the never-read
    # pointer, exactly like the share), the uniform pulse is a scalar and the closure coefficient counts the one extra term
    pulse_on = 0 if recharge is None else 1
    added_arr = added if recharge is None else params.thickness_m.reshape(-1, 1)  # a (L, 1) VIEW (same compiled specialisation; never read)
    cell_tol = params.cell_closure_coefficient if recharge is None else cell_closure_coefficient(n_layers, True)
    n_if = n_layers - 1
    # has_share == 0: the kernel never reads the share pointer, so an ALREADY resident nonempty FP64 parameter buffer
    # (thickness_m, length L >= 1) is passed; no extra allocation exists and the parameter ledger stays exact
    share_arr, has_share = (params.thickness_m, 0) if share is None else (share, 1)
    water_out = np.empty((n_layers, n_cells), dtype=np.float64)
    percolation = np.empty((n_if, n_cells), dtype=np.float64)
    diffusion = np.empty((n_if, n_cells), dtype=np.float64)
    ret = np.empty(n_cells, dtype=np.float64)
    drain = np.empty(n_cells, dtype=np.float64)
    evap = np.empty(n_cells, dtype=np.float64)
    unmet = np.empty(n_cells, dtype=np.float64)
    rows = np.empty((ROW_COUNT, n_cells), dtype=np.float64)
    kernel(water_in, added_arr, pet, active_u8, float(dt), params.thickness_m, params.residual_m, params.field_capacity_m,
           params.saturation_m, params.capacity_tolerance_m, params.interface_conductance_m_s,
           params.pair_inverse_thickness, params.residual_theta0, params.theta_evaporation_stress,
           params.evaporation_factor, params.gravity_code, cell_tol, NO_FLAG,
           params.interface_conductance_up_m_s, params.directional, params.bottom_code, share_arr, has_share,
           0.0 if recharge is None else float(recharge), pulse_on, water_out, percolation, diffusion, ret, drain, evap, unmet, rows)
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

    def run(column, water_in, added, pet, dt, share=None, recharge=None):
        return _launch(kernel, params, water_in, added, pet, column.active_u8, dt, share, recharge)

    directional = bool(params.directional)
    info = {"implementation": "numba serial cell/layer loops, fastmath=False; per-interface expm1 factor(s) evaluated "
                              "once per call before the cell loop (common properties), local array(s) of L-1 doubles",
            "versions": numba_versions(),
            "expm1": "LLVM/libm expm1 (may differ from NumPy by an ulp); declared bound rtol 2e-12 / atol 1e-14",
            "diffusion_mode": layers.diffusion_mode,
            "bottom_drainage": layers.bottom_drainage,
            "factor_arrays_per_call": 2 if directional else 1,
            "factor_scratch_host_bytes": (2 if directional else 1) * max(n_layers - 1, 1) * 8,
            # the upward conductance aliases the downward array in constant mode: count each distinct buffer once
            "parameter_host_bytes": int(sum(int(a.nbytes) for a in {id(getattr(params, name)): getattr(params, name)
                                                                    for name in params._fields
                                                                    if hasattr(getattr(params, name), "nbytes")}.values()))}
    return run, info
