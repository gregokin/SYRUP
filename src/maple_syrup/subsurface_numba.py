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

Phase 2 S2-A.3 (ONE physical body, two compile-time variants): the statements below are written ONCE in `_build(numba, common)` and compiled twice by Numba's
dead-branch pruning of the closure constant `common`: the COMMON variant (`compiled_kernel`, used by `prepare_runner` for a common-layer column) and the SPATIAL
variant (`compiled_spatial_kernel`, used by `spatial_launch` for the explicit spatial owner of `maple_syrup.spatial_soil_layers`). Every per-layer / per-interface
parameter map is read through the strides `a[k * stride_layer + i * stride_cell]`, resolved ONCE per call from the layout kind (0 scalar: 0, 0; 1 shared: 1, 0;
2 per-cell: N, 1 for a C-contiguous `(rows, N)` map viewed flat); a per-element branch on the kind measurably prevented the compiler from optimising the cell loop.

* COMMON variant: the cached `(L,)` metre arrays of `KernelParameters` (`res`, `fc`, `sat`, `cap_tol`, thickness, pair inverse, conductances) are read AS THEY ARE: the
  bound "locals" are ALIASES of those inputs, so the only per-call Numba allocations are the one or two per-interface factor arrays of the pre-A3 kernel (the frozen
  `estimate_step_memory` counts exactly these). The parameter ledger is the historical one.
* SPATIAL variant: the owner's retained RAW maps (`theta_res`, `theta_fc`, `theta_sat`, `dz`, `1/pair`, `G_down`, `G_up`) are passed as flat read-only views; everything that
  does not vary with the cell is bound ONCE per call into local `(L,)` / `(L - 1,)` arrays before the cell loop (`res = theta_res dz`, `fc`, `sat`, `cap_tol = coef sat`,
  `1 / pair` and the per-interface `-expm1(-((G s) dt))` factors, with the same IEEE products as `subsurface.soil_layers` and the NumPy spatial reader), and only a map
  whose layout is per-cell is re-bound inside the cell loop with the same expressions. No `(L, N)` depth/tolerance tensor exists.

Phase 2 D1 (third compile-time variant, `floating=True`; the default `floating=False` leaves both historical variants' statements, arguments and arithmetic unchanged): the FLOATING porous column
(`maple_syrup.floating_soil_layers`) runs the SAME body. Its extra inputs travel through the SAME 40-argument signature (a documented private binding, see `floating_launch`): `dz` carries the shared nominal edges, `tol` the per-cell soil
depth `D`, `pair_inv` the nominal pair separations, `g_down` the raw interface input, `g_up` the per-cell bottom-open flags (0/1 doubles) and `res_theta0` the donor coefficient. Per cell, the actual porous thickness
`h = max(0, min(D, e_{l+1}) - e_l)` is bound into the SAME local arrays the spatial variant uses (`dz_c` holds the finite stand-in 1.0 where `h = 0`, used only for arithmetic, never as a thickness/capacity), the lower boundary is read per cell, an
input above capacity is flagged only on ACTIVE cells (published inactive policy), and an interface touching a zero-thickness slot gets the exchange factors 0.0 (no exchange; the donor/receiver limits would give exactly 0 anyway). The gravity cascade is
the UNCHANGED gravity block of the step kernel (the default variants keep their historical code generation: a jitted inline helper added reference-count traffic and a 12-22 % warm slowdown, see the D1b correction report); the floating reconciliation kernel is GENERATED from a fixed trusted template and that very block (`floating_reconcile_source`, fail-closed anchors, hashed provenance): one copy of the cascade equations.

Both variants are compiled for ONE explicit signature whose inputs are READ-ONLY array types: Numba therefore refuses, at compile time, any statement that writes an
input, and a writable caller array converts to that signature without a second specialisation.

Numba is imported lazily; `prepare_runner` / `prepare_spatial_column(backend="numba")` compile their variant once (cost at preparation, outside steps).
Missing Numba is a `SubsurfaceBackendUnavailableError`; there is no fallback to the reference. Every output array is freshly allocated per call;
the kernel reads the caller's arrays and never writes them.
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

__all__ = ["compiled_floating_kernel", "compiled_floating_reconcile", "floating_reconcile_provenance", "floating_reconcile_source", "compiled_kernel", "compiled_spatial_kernel", "floating_launch", "local_scratch_doubles", "numba_available", "numba_versions", "prepare_runner", "require_numba", "reset_compiled", "spatial_launch"]

_FLOATING_KERNEL: Any = None  # the compiled FLOATING variant (Phase 2 D1)
_FLOATING_RECONCILE: Any = None  # the compiled floating staged-reconciliation kernel (generated from the step kernel's gravity block)
_KERNEL: Any = None  # the compiled COMMON variant
_SPATIAL_KERNEL: Any = None  # the compiled SPATIAL variant
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




KIND_SCALAR, KIND_SHARED, KIND_CELL = 0, 1, 2  # layout codes of the indexed readers (scalar | one value per layer/interface | per cell)
LOCAL_DOUBLES_PER_LAYER = 5  # SPATIAL variant per-call local arrays: dz, res, fc, sat, cap_tol (one value per layer)
LOCAL_DOUBLES_PER_INTERFACE = 3  # SPATIAL variant per-call local arrays: 1/pair, downward factor, upward factor (one value per interface)


def local_scratch_doubles(n_layers: int, *, spatial: bool = True, diffusion_mode: str = "tietjen_donor_depth") -> int:
    """Doubles of the kernel's per-call LOCAL arrays (Numba NRT heap, independent of the cell count). SPATIAL variant: five per layer and three per interface (`max(L - 1, 1)`,
    an upper bound: the upward factor aliases the downward one in constant mode and a one-layer stack needs none). COMMON variant: exactly the per-interface factor arrays the
    frozen estimate counts (`subsurface.factor_scratch_doubles`: one array in constant mode, two in donor-depth mode); its bound locals alias the cached parameter arrays."""
    n = int(n_layers)
    if not spatial:
        return 2 * max(n - 1, 1) if diffusion_mode != "constant" else max(n - 1, 1)
    return LOCAL_DOUBLES_PER_LAYER * n + LOCAL_DOUBLES_PER_INTERFACE * max(n - 1, 1)


def _build(numba: Any, common: bool, floating: bool = False):
    jit = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
    # the helpers are inlined into the kernel at IR level: a non-inlined call passes every array by its unpacked struct and increments/decrements the
    # NRT reference counts atomically on each call, which dominated the per-cell binding cost (measured, see the report)
    jit_inline = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy", inline="always")
    f8, i8, u8 = numba.types.float64, numba.types.int64, numba.types.uint8
    ro1, ro2 = numba.types.Array(f8, 1, "C", readonly=True), numba.types.Array(f8, 2, "C", readonly=True)
    ro_u8 = numba.types.Array(u8, 1, "C", readonly=True)
    rw1, rw2 = numba.types.Array(f8, 1, "C"), numba.types.Array(f8, 2, "C")

    @jit_inline
    def layer_stride(kind, n):
        # stride of the layer/interface index: 0 scalar, 1 shared (one value per layer), N per-cell (flat C-contiguous (rows, N))
        if kind == 0:
            return 0
        if kind == 1:
            return 1
        return n

    @jit_inline
    def cell_stride(kind):
        return 1 if kind == 2 else 0

    @jit_inline
    def bind_layer(l, i, dz, dz_l, dz_i, b_res, r_l, r_i, b_fc, f_l, f_i, b_sat, s_l, s_i, coef, dz_c, res_c, fc_c, sat_c, tol_c):
        # SPATIAL variant only: the RAW theta/dz maps give res = theta_res dz, fc = theta_fc dz, sat = theta_sat dz and cap_tol = coef sat with the very IEEE
        # products of `soil_layers`
        d = dz[l * dz_l + i * dz_i]
        dz_c[l] = d
        res_c[l] = b_res[l * r_l + i * r_i] * d
        fc_c[l] = b_fc[l * f_l + i * f_i] * d
        s = b_sat[l * s_l + i * s_i] * d
        sat_c[l] = s
        tol_c[l] = coef * s

    @jit_inline
    def bind_interface(k, i, dt, directional, pair_inv, p_l, p_i, g_down, g_up, g_l, g_i, s_c, e_c, eu_c):
        # 1/pair (SPATIAL variant: stored in the local s_c; COMMON variant: s_c aliases the cached read-only array) and the per-interface exchange factor(s)
        # -expm1(-((G s) dt)) with the expression order of the per-cell NumPy form; the upward factor exists only in the donor-depth mode (in constant mode
        # eu_c IS e_c: nothing is written here)
        s = pair_inv[k * p_l + i * p_i]
        if not common:
            s_c[k] = s
        rate = g_down[k * g_l + i * g_i] * s
        x = rate * dt
        e_c[k] = -math.expm1(-x)
        if directional != 0:
            rate_up = g_up[k * g_l + i * g_i] * s
            x_up = rate_up * dt
            eu_c[k] = -math.expm1(-x_up)

    @jit_inline
    def bind_floating(i, edges, depth, sep_nom, b_res, r_l, r_i, b_fc, f_l, f_i, b_sat, s_l, s_i, kif, k_l, k_i, coef, dt, directional, dcoef, n_layers, n_if, dz_c, res_c, fc_c, sat_c, tol_c, s_c, e_c, eu_c):
        # FLOATING variant only: the actual porous thickness of every layer of cell i from the O(L) edges and the cell's soil depth, bound into the same locals as the spatial variant. h = max(0, min(D, e1) - e0) is exactly the nominal
        # `e1 - e0` for a full slab. dz_c is the finite stand-in 1.0 where h == 0 (arithmetic only: a zero slot has capacity theta * 0 = 0 and water exactly 0); capacities are theta * h, tol = coef * (theta_sat h).
        d = depth[i]
        for l in range(n_layers):
            x = min(d, edges[l + 1]) - edges[l]
            if x < 0.0:
                x = 0.0
            dz_c[l] = x if x > 0.0 else 1.0
            res_c[l] = b_res[l * r_l + i * r_i] * x
            fc_c[l] = b_fc[l * f_l + i * f_i] * x
            sv = b_sat[l * s_l + i * s_i] * x
            sat_c[l] = sv
            tol_c[l] = coef * sv
        for k in range(n_if):
            sk = 1.0 / dz_c[k] + 1.0 / dz_c[k + 1]
            s_c[k] = sk
            hk = min(d, edges[k + 1]) - edges[k]
            hk1 = min(d, edges[k + 2]) - edges[k + 1]
            if hk < 0.0:
                hk = 0.0
            if hk1 < 0.0:
                hk1 = 0.0
            both = hk > 0.0 and hk1 > 0.0
            if d >= edges[k + 2]:
                sp = sep_nom[k]
            else:
                sp = (hk + hk1) / 2.0
            if not (sp > 0.0):
                sp = 1.0
            v = kif[k * k_l + i * k_i]
            if directional != 0:
                g_dn = ((dcoef * v) * dz_c[k]) / sp
                g_up = ((dcoef * v) * dz_c[k + 1]) / sp
            else:
                g_dn = v / sp
                g_up = g_dn
            if both:
                e_c[k] = -math.expm1(-((g_dn * sk) * dt))
                if directional != 0:
                    eu_c[k] = -math.expm1(-((g_up * sk) * dt))
            else:  # an interface touching an all-rock slot cannot exchange: factor 0.0 (amount = g * 0.0 = 0)
                e_c[k] = 0.0
                if directional != 0:
                    eu_c[k] = 0.0

    signature = numba.types.void(
        ro2, ro2, ro1, ro_u8, f8,  # water_in, add, pet, active, dt
        ro1, i8, ro1, i8, ro1, i8, ro1, i8, ro1, f8,  # dz,k_dz | res,k | fc,k | sat,k | cap_tol | coef
        ro1, i8, ro1, ro1, i8,  # pair_inv,k_pi | g_down | g_up,k_g
        f8, f8, f8, i8, f8, f8, i8, i8,  # res_theta0, stress, factor, gravity, cell_tol, no_flag, directional, bottom
        ro1, i8, f8, i8,  # share, has_share, pulse, pulse_on
        rw2, rw2, rw2, rw1, rw1, rw1, rw1, rw2)  # w_out, perc, diff, ret, drain, evap, unmet, rows

    @numba.njit(signature, cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
    def step_kernel(w_in, add, pet, active, dt, dz, k_dz, b_res, k_res, b_fc, k_fc, b_sat, k_sat, tol, coef,
                    pair_inv, k_pi, g_down, g_up, k_g, res_theta0, stress, factor, gravity, cell_tol, no_flag, directional,
                    bottom, share, has_share, pulse, pulse_on, w_out, perc, diff, ret, drain, evap, unmet, rows):
        n_layers = w_in.shape[0]
        n_cells = w_in.shape[1]
        n_if = n_layers - 1
        # Per-call locals (see the module docstring). `common` is a closure constant: exactly one branch below survives compilation.
        n_pair = n_if if n_if > 0 else 1
        if common:
            # the cached (L,) metre arrays and the padded pair inverse ARE the bound values: aliased, nothing is allocated for them
            dz_c = dz
            res_c = b_res
            fc_c = b_fc
            sat_c = b_sat
            tol_c = tol
            s_c = pair_inv
            relaxation_factors = np.empty(n_if, dtype=np.float64)  # distinct from the scalar evaporation `factor`
        else:
            dz_c = np.empty(n_layers, dtype=np.float64)
            res_c = np.empty(n_layers, dtype=np.float64)
            fc_c = np.empty(n_layers, dtype=np.float64)
            sat_c = np.empty(n_layers, dtype=np.float64)
            tol_c = np.empty(n_layers, dtype=np.float64)
            s_c = np.empty(n_pair, dtype=np.float64)
            relaxation_factors = np.empty(n_pair, dtype=np.float64)
        if directional != 0:
            relaxation_factors_up = np.empty(n_if if common else n_pair, dtype=np.float64)
        else:
            relaxation_factors_up = relaxation_factors
        layers_cell = k_dz == 2 or k_res == 2 or k_fc == 2 or k_sat == 2  # (spatial variant) some layer map is per-cell
        ifaces_cell = k_pi == 2 or k_g == 2
        # the two strides of each map, resolved once per call (index = layer * stride_layer + cell * stride_cell)
        dz_l, dz_i = layer_stride(k_dz, n_cells), cell_stride(k_dz)
        r_l, r_i = layer_stride(k_res, n_cells), cell_stride(k_res)
        f_l, f_i = layer_stride(k_fc, n_cells), cell_stride(k_fc)
        s_l, s_i = layer_stride(k_sat, n_cells), cell_stride(k_sat)
        p_l, p_i = layer_stride(k_pi, n_cells), cell_stride(k_pi)
        g_l, g_i = layer_stride(k_g, n_cells), cell_stride(k_g)
        if (not common) and (not floating):
            for l in range(n_layers):
                bind_layer(l, 0, dz, dz_l, dz_i, b_res, r_l, r_i, b_fc, f_l, f_i, b_sat, s_l, s_i, coef, dz_c, res_c, fc_c, sat_c, tol_c)
        if not floating:
            for k in range(n_if):
                bind_interface(k, 0, dt, directional, pair_inv, p_l, p_i, g_down, g_up, g_l, g_i, s_c, relaxation_factors, relaxation_factors_up)
        for i in range(n_cells):
            if floating:  # FLOATING variant: every geometry-dependent local is bound per cell from the edges and the cell's soil depth (module docstring); g_down carries the raw interface input, dz the edges, tol the depth
                bind_floating(i, dz, tol, pair_inv, b_res, r_l, r_i, b_fc, f_l, f_i, b_sat, s_l, s_i, g_down, g_l, g_i, coef, dt, directional, res_theta0, n_layers, n_if, dz_c, res_c, fc_c, sat_c, tol_c, s_c,
                           relaxation_factors, relaxation_factors_up)
            if (not common) and (not floating):  # a common column has only shared maps: nothing is re-bound per cell (this whole block is pruned from the common variant)
                if layers_cell:
                    for l in range(n_layers):
                        bind_layer(l, i, dz, dz_l, dz_i, b_res, r_l, r_i, b_fc, f_l, f_i, b_sat, s_l, s_i, coef, dz_c, res_c, fc_c, sat_c, tol_c)
                if ifaces_cell:
                    for k in range(n_if):
                        bind_interface(k, i, dt, directional, pair_inv, p_l, p_i, g_down, g_up, g_l, g_i, s_c, relaxation_factors, relaxation_factors_up)
            if common:
                rt0 = res_theta0  # layer-0 residual theta of the common stack
            else:
                rt0 = b_res[i * r_i]  # layer 0 of the explicit map (layer term vanishes)
            a = active[i] != 0
            if floating:  # per-cell lower boundary (configured open AND D > H) for the UNCHANGED gravity block below, which reads `bottom`; g_up carries the 0/1 flags in the floating binding
                bottom = 1 if g_up[i] != 0.0 else 0
            f = 0
            before = 0.0
            added = 0.0
            for l in range(n_layers):
                wl = w_in[l, i]
                if not math.isfinite(wl):
                    f |= 1 << 0
                if wl < 0.0:
                    f |= 1 << 1
                if floating:  # published inactive policy: no capacity law on an inactive placeholder
                    if a and wl > sat_c[l] + tol_c[l]:
                        f |= 1 << 2
                else:
                    if wl > sat_c[l] + tol_c[l]:
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
                    ex = w - sat_c[l]
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
                            term = sat_c[k + 1] - w_out[k + 1, i]
                            if term < 0.0:  # noqa: PLR1730 - explicit ordered FP comparison, NaN-propagating; no max()
                                term = 0.0
                            capacity = capacity + term
                            perc[k, i] = capacity
                    carry = 0.0
                    for l in range(n_layers):
                        w = w_out[l, i] + carry
                        ex = w - fc_c[l]
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
                        if w_out[l, i] > sat_c[l] + tol_c[l]:
                            fit_ok = False
                    if not fit_ok:
                        f |= 1 << 17
                # evaporation from layer 0
                w0 = w_out[0, i]
                theta0 = w0 / dz_c[0]
                if theta0 <= rt0:
                    m = 0.0
                elif theta0 < stress:
                    r = theta0 / stress
                    m = r * r
                else:
                    m = 1.0
                demand = ((factor * e_pot) * dt) * m
                avail0 = w0 - res_c[0]
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
                    ti = wi / dz_c[k]
                    tj = wj / dz_c[k + 1]
                    s = s_c[k]
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
                        resd = res_c[k]
                        satr = sat_c[k + 1]
                    else:
                        down = False
                        amt = -amount
                        wd = wj
                        wr = wi
                        resd = res_c[k + 1]
                        satr = sat_c[k]
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
                    if w > sat_c[l] + tol_c[l]:
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
    """The process-wide compiled COMMON variant (built lazily; Numba imported here only; compiled for its ONE explicit read-only-input signature at build, so the cost is paid
    by the first caller, which is a preparation, never a step)."""
    global _KERNEL
    if _KERNEL is None:
        require_numba()
        import numba

        with _LOCK:
            if _KERNEL is None:
                _KERNEL = _build(numba, True)
    return _KERNEL


def compiled_spatial_kernel():
    """The process-wide compiled SPATIAL variant of the same body (same signature; built lazily at an explicit spatial preparation)."""
    global _SPATIAL_KERNEL
    if _SPATIAL_KERNEL is None:
        require_numba()
        import numba

        with _LOCK:
            if _SPATIAL_KERNEL is None:
                _SPATIAL_KERNEL = _build(numba, False)
    return _SPATIAL_KERNEL


_GRAV_BEGIN = "                # gravity cascade, top to bottom within the step\n                if gravity != 0:\n"
_GRAV_END = "                    drain_v = carry  # exactly 0 with a closed bottom\n"

_RECONCILE_TEMPLATE = """
@jit
def reconcile_kernel(w_in, active, bopen, depth, edges, b_fc, k_fc, b_sat, k_sat, w_out, perc, surface, export):
    n_layers = w_in.shape[0]
    n_cells = w_in.shape[1]
    n_if = n_layers - 1
    fc_c = np.empty(n_layers, dtype=np.float64)
    sat_c = np.empty(n_layers, dtype=np.float64)
    f_l = 0 if k_fc == 0 else (1 if k_fc == 1 else n_cells)
    f_i = 1 if k_fc == 2 else 0
    s_l = 0 if k_sat == 0 else (1 if k_sat == 1 else n_cells)
    s_i = 1 if k_sat == 2 else 0
    gravity = 1
    for i in range(n_cells):
        for l in range(n_layers):
            w_out[l, i] = w_in[l, i]
        for k in range(n_if):
            perc[k, i] = 0.0
        surface[i] = 0.0
        export[i] = 0.0
        if active[i] == 0:
            continue
        d = depth[i]
        for l in range(n_layers):
            x = min(d, edges[l + 1]) - edges[l]
            if x < 0.0:
                x = 0.0
            fc_c[l] = b_fc[l * f_l + i * f_i] * x
            sat_c[l] = b_sat[l * s_l + i * s_i] * x
        bottom = 1 if bopen[i] != 0 else 0
        drain_v = 0.0
@@GRAVITY@@
        if bottom != 0:
            export[i] = drain_v
        for l in range(n_layers):
            ex = w_out[l, i] - sat_c[l]
            if ex > 0.0:
                w_out[l, i] = w_out[l, i] - ex
                surface[i] = surface[i] + ex
"""


def _gravity_block_source() -> str:
    """The ONE gravity cascade, taken VERBATIM from this module's own text of the step kernel (fail closed: each anchor must occur exactly once as real source lines, otherwise nothing is built). Nothing user-supplied is ever executed."""
    from pathlib import Path

    text = Path(__file__).read_text()
    for marker in (_GRAV_BEGIN, _GRAV_END):
        if text.count(marker) != 1:
            raise SubsurfaceError(f"the step kernel text does not contain the gravity anchor exactly once ({marker[:60]!r}: {text.count(marker)}); the floating reconciliation is not built")
    a = text.index(_GRAV_BEGIN)
    b = text.index(_GRAV_END, a) + len(_GRAV_END)
    return text[a:b]


def floating_reconcile_source() -> tuple[str, str]:
    """`(gravity block, generated reconcile kernel source)`: the generated text splices the step kernel's own gravity statements (dedented, re-indented) into a fixed trusted template."""
    import textwrap

    block = _gravity_block_source()
    body = textwrap.indent(textwrap.dedent(block), " " * 8)
    return block, _RECONCILE_TEMPLATE.replace("@@GRAVITY@@\n", body)


def floating_reconcile_provenance() -> dict:
    import hashlib

    block, code = floating_reconcile_source()
    return {"gravity_block_sha256": hashlib.sha256(block.encode()).hexdigest(), "generated_source_sha256": hashlib.sha256(code.encode()).hexdigest(), "template_sha256": hashlib.sha256(_RECONCILE_TEMPLATE.encode()).hexdigest()}


def _build_floating_reconcile(numba: Any):
    """The floating staged reconciliation, GENERATED from a fixed trusted template and the step kernel's own gravity block (`floating_reconcile_source`): the cascade statements exist once in the source, exactly as the historical kernel
    has them (so the default step kernels keep their historical code generation), and the reconciliation runs them on the NEW field capacity, then returns the saturation excess to the surface. No addition, evaporation, diffusion or time."""
    jit = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
    _, code = floating_reconcile_source()
    namespace = {"np": np, "math": math, "jit": jit}
    exec(compile(code, "<generated floating reconcile kernel>", "exec"), namespace)  # noqa: S102 - trusted template + this module's own text
    return namespace["reconcile_kernel"]


def compiled_floating_kernel():
    """The process-wide compiled FLOATING variant of the same body (same signature; built lazily at a floating-column preparation)."""
    global _FLOATING_KERNEL
    if _FLOATING_KERNEL is None:
        require_numba()
        import numba

        with _LOCK:
            if _FLOATING_KERNEL is None:
                _FLOATING_KERNEL = _build(numba, False, True)
    return _FLOATING_KERNEL


def compiled_floating_reconcile():
    global _FLOATING_RECONCILE
    if _FLOATING_RECONCILE is None:
        require_numba()
        import numba

        with _LOCK:
            if _FLOATING_RECONCILE is None:
                _FLOATING_RECONCILE = _build_floating_reconcile(numba)
    return _FLOATING_RECONCILE


def floating_launch(water_in: np.ndarray, added: np.ndarray | None, pet: np.ndarray, mask_u8: np.ndarray, dt: float, *, edges: np.ndarray, depth: np.ndarray, sep_nominal: np.ndarray, maps: dict, kinds: dict,
                    interface: np.ndarray, interface_kind: int, bottom_open_f64: np.ndarray, tolerance_coefficient: float, donor_coefficient: float, stress: float, factor: float, directional: int,
                    share: np.ndarray | None, recharge: float | None) -> StepArrays:
    """One compiled step of a FLOATING porous column (`floating_soil_layers` calls this). The private argument binding of the floating variant (module docstring): `dz`=edges, `tol`=soil depth, `pair_inv`=nominal separations,
    `g_down`=raw interface input, `g_up`=per-cell bottom-open flags, `res_theta0`=donor coefficient; `maps` are the owner's RAW theta maps as flat read-only views with their layout kinds."""
    n_layers, n_cells = water_in.shape
    kernel = compiled_floating_kernel()
    pulse_on = 0 if recharge is None else 1
    cell_tol = cell_closure_coefficient(n_layers, pulse_on == 1)
    share_arr, has_share = (pet, 0) if share is None else (share, 1)
    water_out, percolation, diffusion, ret, drain, evap, unmet, rows = _outputs(n_layers, n_cells)
    kernel(water_in, water_in if recharge is not None else added, pet, mask_u8, float(dt),
           edges, KIND_SHARED, maps["theta_residual"], kinds["theta_residual"], maps["theta_field_capacity"], kinds["theta_field_capacity"], maps["theta_saturation"], kinds["theta_saturation"], depth,
           float(tolerance_coefficient), sep_nominal, KIND_SHARED, interface, bottom_open_f64, int(interface_kind), float(donor_coefficient), float(stress), float(factor), 1, cell_tol, NO_FLAG, int(directional), 0,
           share_arr, has_share, 0.0 if recharge is None else float(recharge), pulse_on, water_out, percolation, diffusion, ret, drain, evap, unmet, rows)
    return StepArrays(water_out, percolation, diffusion, ret, drain, evap, unmet, rows)


def reset_compiled() -> None:
    """Drop every compiled variant (tests of the compile path only)."""
    global _KERNEL, _SPATIAL_KERNEL, _FLOATING_KERNEL, _FLOATING_RECONCILE
    with _LOCK:
        _KERNEL = None
        _SPATIAL_KERNEL = None
        _FLOATING_KERNEL = None
        _FLOATING_RECONCILE = None


def _outputs(n_layers: int, n_cells: int):
    n_if = n_layers - 1
    return (np.empty((n_layers, n_cells), dtype=np.float64), np.empty((n_if, n_cells), dtype=np.float64), np.empty((n_if, n_cells), dtype=np.float64),
            np.empty(n_cells, dtype=np.float64), np.empty(n_cells, dtype=np.float64), np.empty(n_cells, dtype=np.float64),
            np.empty(n_cells, dtype=np.float64), np.empty((ROW_COUNT, n_cells), dtype=np.float64))


def _launch(kernel, params: KernelParameters, water_in: np.ndarray, added: np.ndarray | None, pet: np.ndarray,
            active_u8: np.ndarray, dt: float, share: np.ndarray | None = None, recharge: float | None = None) -> StepArrays:
    n_layers, n_cells = water_in.shape
    # Phase 1.5 pulse-aware step: no added-water tensor (an ALREADY resident parameter buffer is passed as the never-read
    # pointer, exactly like the share), the uniform pulse is a scalar and the closure coefficient counts the one extra term
    pulse_on = 0 if recharge is None else 1
    added_arr = added if recharge is None else params.thickness_m.reshape(-1, 1)  # a (L, 1) VIEW (never read)
    cell_tol = params.cell_closure_coefficient if recharge is None else cell_closure_coefficient(n_layers, True)
    # has_share == 0: the kernel never reads the share pointer, so an ALREADY resident nonempty FP64 parameter buffer
    # (thickness_m, length L >= 1) is passed; no extra allocation exists and the parameter ledger stays exact
    share_arr, has_share = (params.thickness_m, 0) if share is None else (share, 1)
    water_out, percolation, diffusion, ret, drain, evap, unmet, rows = _outputs(n_layers, n_cells)
    # COMMON column: the cached (L,) metre arrays are shared by every cell (kind 1) and read as they are (the common variant aliases them; no coefficient)
    kernel(water_in, added_arr, pet, active_u8, float(dt),
           params.thickness_m, KIND_SHARED, params.residual_m, KIND_SHARED, params.field_capacity_m, KIND_SHARED, params.saturation_m, KIND_SHARED,
           params.capacity_tolerance_m, 0.0,
           params.pair_inverse_thickness, KIND_SHARED, params.interface_conductance_m_s, params.interface_conductance_up_m_s, KIND_SHARED,
           params.residual_theta0, params.theta_evaporation_stress, params.evaporation_factor, params.gravity_code, cell_tol, NO_FLAG,
           params.directional, params.bottom_code, share_arr, has_share,
           0.0 if recharge is None else float(recharge), pulse_on, water_out, percolation, diffusion, ret, drain, evap, unmet, rows)
    return StepArrays(water_out, percolation, diffusion, ret, drain, evap, unmet, rows)


def spatial_launch(water_in: np.ndarray, added: np.ndarray | None, pet: np.ndarray, mask_u8: np.ndarray, dt: float, *, maps: dict, kinds: dict,
                   tolerance_coefficient: float, residual_theta_stress_factor: tuple, gravity_code: int, directional: int, bottom_code: int,
                   share: np.ndarray | None, recharge: float | None) -> StepArrays:
    """One compiled step of an EXPLICIT SPATIAL parameter set (`spatial_soil_layers` calls this; nothing else should). `maps` are the owner's
    retained RAW maps as FLAT C-contiguous read-only views (`theta_residual`, `theta_field_capacity`, `theta_saturation`, `dz` and, for L > 1,
    `pair_inverse_thickness`, `interface_conductance_down` and `interface_conductance_up` (donor mode)); `kinds` the layout code of each; the
    metre bounds are derived inline by the SPATIAL variant. No `(L, N)` parameter tensor, no copy of the owner's maps or mask, and no added-water
    placeholder is allocated: the never-read pointers alias arrays that already exist (`water_in`, `pet`)."""
    n_layers, n_cells = water_in.shape
    kernel = compiled_spatial_kernel()
    pulse_on = 0 if recharge is None else 1
    cell_tol = cell_closure_coefficient(n_layers, pulse_on == 1)
    stress, factor = residual_theta_stress_factor
    dz = maps["dz"]
    pair = maps.get("pair_inverse_thickness", dz)  # L = 1: no interface map exists; the kernel never reads these (n_if = 0)
    g_down = maps.get("interface_conductance_down", dz)
    g_up = maps.get("interface_conductance_up", g_down)
    k_pi = kinds.get("pair_inverse_thickness", KIND_SHARED)
    k_g = kinds.get("interface_conductance_down", KIND_SHARED)
    share_arr, has_share = (pet, 0) if share is None else (share, 1)  # a never-read alias when there is no share
    water_out, percolation, diffusion, ret, drain, evap, unmet, rows = _outputs(n_layers, n_cells)
    kernel(water_in, water_in if recharge is not None else added, pet, mask_u8, float(dt),
           dz, kinds["dz"], maps["theta_residual"], kinds["theta_residual"], maps["theta_field_capacity"], kinds["theta_field_capacity"],
           maps["theta_saturation"], kinds["theta_saturation"], dz, float(tolerance_coefficient),
           pair, k_pi, g_down, g_up, k_g, 0.0, float(stress), float(factor), int(gravity_code), cell_tol, NO_FLAG, int(directional), int(bottom_code),
           share_arr, has_share, 0.0 if recharge is None else float(recharge), pulse_on, water_out, percolation, diffusion, ret, drain, evap, unmet, rows)
    return StepArrays(water_out, percolation, diffusion, ret, drain, evap, unmet, rows)


def prepare_runner(layers: SoilLayers):
    """Compile (if needed) and warm up the kernel for `layers`; return `(runner, info)`. The runner allocates fresh
    outputs per call and never writes an input."""
    if not isinstance(layers, SoilLayers):
        raise SubsurfaceError("layers must be a SoilLayers")
    kernel = compiled_kernel()
    params = kernel_parameters(layers)
    n_layers = layers.n_layers
    # Warm-up on one inactive cell with zero state (the kernel is already compiled for its one explicit signature).
    _launch(kernel, params, np.zeros((n_layers, 1)), np.zeros((n_layers, 1)), np.zeros(1), np.zeros(1, np.uint8), 1.0)

    def run(column, water_in, added, pet, dt, share=None, recharge=None):
        return _launch(kernel, params, water_in, added, pet, column.active_u8, dt, share, recharge)

    directional = bool(params.directional)
    info = {"implementation": "numba serial cell/layer loops, fastmath=False; ONE kernel body shared with the explicit spatial owner "
                              "(compile-time common variant; its bound values alias the cached parameter arrays); the per-interface expm1 factor(s) are "
                              "evaluated once per call before the cell loop (common properties), local array(s) of L-1 doubles",
            "versions": numba_versions(),
            "expm1": "LLVM/libm expm1 (may differ from NumPy by an ulp); declared bound rtol 2e-12 / atol 1e-14",
            "diffusion_mode": layers.diffusion_mode,
            "bottom_drainage": layers.bottom_drainage,
            "factor_arrays_per_call": 2 if directional else 1,
            "factor_scratch_host_bytes": (2 if directional else 1) * max(n_layers - 1, 1) * 8,
            "kernel_local_scratch_host_bytes": local_scratch_doubles(n_layers, spatial=False, diffusion_mode=layers.diffusion_mode) * 8,
            # the upward conductance aliases the downward array in constant mode: count each distinct buffer once
            "parameter_host_bytes": int(sum(int(a.nbytes) for a in {id(getattr(params, name)): getattr(params, name)
                                                                    for name in params._fields
                                                                    if hasattr(getattr(params, name), "nbytes")}.values()))}
    return run, info
