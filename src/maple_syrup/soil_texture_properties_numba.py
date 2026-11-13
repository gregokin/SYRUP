"""Compiled Numba CPU runner of `maple_syrup.soil_texture_properties` (backend "numba"). Component only.

One serial loop over (layer, cell). The element body evaluates EXACTLY the arithmetic of `soil_texture_properties.reference_block`, term for term in the same expression order, reading the SAME coefficient
tuples (module constants of `soil_texture_properties`); no fastmath, no prange, no reordering, no clipping, and `error_model="numpy"` so that a non-finite quotient is a value (then refused by the
checks), not a Python exception. Operations without a library call (`+ - * /`, comparisons) are the same IEEE operations as NumPy's, so `theta_saturation`, `theta_field_capacity_6kpa`,
`theta_1500` and `air_entry_kpa` are expected bitwise equal to the reference (tested, not argued); `saturated_conductivity_m_s` involves `log`/`pow` and is compared at the declared bound
(rtol 2e-12, atol 1e-14). An absent override is a NaN stand-in. Inputs are only read; outputs are the caller's freshly allocated arrays; a flagged element is counted and reported through the same four
diagnostic words as the reference. A read-only versus writeable input array is a distinct Numba type, so the first call of each combination compiles once (startup, not steady state).

Numba is imported lazily; missing Numba is a `TexturePropertyBackendUnavailableError`; there is no fallback to the reference.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from maple_syrup import soil_texture_properties as tp

__all__ = ["numba_available", "numba_versions", "require_numba", "reset_compiled", "run"]

_KERNEL: Any = None
_LOCK = threading.Lock()


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def numba_versions() -> dict:
    import numba

    return {"numba": numba.__version__, "numpy": np.__version__}


def require_numba():
    try:
        import numba
    except Exception as exc:  # noqa: BLE001
        raise tp.TexturePropertyBackendUnavailableError(f"backend 'numba' requires Numba, which is not importable ({type(exc).__name__}: {exc}); there is no fallback to the NumPy reference") from exc
    return numba


def reset_compiled() -> None:
    global _KERNEL
    with _LOCK:
        _KERNEL = None


def _build():
    numba = require_numba()
    import math

    E1R, E1A = tp.EQ1_REGRESSION, tp.EQ1_ADJUST
    E2R, E2A = tp.EQ2_REGRESSION, tp.EQ2_ADJUST
    E3R, E3A, E3S = tp.EQ3_REGRESSION, tp.EQ3_ADJUST, tp.EQ3_SAT_TERMS
    E4R, E4A = tp.EQ4_REGRESSION, tp.EQ4_ADJUST
    PD, DFA = tp.PARTICLE_DENSITY_G_CM3, tp.DF_AIR_COEFF
    DF_LO, DF_HI = tp.DENSITY_FACTOR_RANGE
    RHO_LO, RHO_HI = tp.DENSITY_RANGE_G_CM3
    CLAY_MAX, OM_MAX, GAP_MIN, AIR_MAX, P_FC, SPAN = tp.CLAY_MAX, tp.OM_MAX_PERCENT, tp.MIN_DENSITY_GAP, tp.AIR_ENTRY_MAX_KPA, tp.FC_PRESSURE_KPA, tp.FC_SPAN_KPA
    LNR, KS_C, KS_DIV = tp.LN_RATIO, tp.KS_COEFF_MM_H, tp.MM_H_PER_M_S
    F_NF, F_FR, F_CL, F_OM, F_DE, F_KO = tp.FLAG_NONFINITE, tp.FLAG_FRACTION, tp.FLAG_CLAY, tp.FLAG_ORGANIC_MATTER, tp.FLAG_DENSITY, tp.FLAG_KS_OVERRIDE
    F_DF, F_CU, F_AI, F_GA, F_KS, F_IN = tp.FLAG_DENSITY_FACTOR, tp.FLAG_CURVE, tp.FLAG_AIR_ENTRY, tp.FLAG_DENSITY_GAP, tp.FLAG_KS, tp.INPUT_FLAGS
    P_WP, P_33, P_SAT, P_AIR, P_KS, P_EQ = tp.PROV_THETA_1500, tp.PROV_THETA_33, tp.PROV_THETA_SAT, tp.PROV_AIR, tp.PROV_KS, tp.PROV_FC_EQ_SAT

    @numba.njit(error_model="numpy", cache=False)
    def element(S, Si, C, OM, rho_kg, o_wp, o_33, o_sat, o_air, o_ks, closure):
        flags = 0
        finite_in = math.isfinite(S) and math.isfinite(Si) and math.isfinite(C) and math.isfinite(OM) and math.isfinite(rho_kg)
        if (not finite_in) or math.isinf(o_wp) or math.isinf(o_33) or math.isinf(o_sat) or math.isinf(o_air) or math.isinf(o_ks):
            flags |= F_NF
        if math.isfinite(S) and math.isfinite(Si) and math.isfinite(C):
            in_range = S >= 0.0 and S <= 1.0 and Si >= 0.0 and Si <= 1.0 and C >= 0.0 and C <= 1.0
            if (not in_range) or abs(((S + Si) + C) - 1.0) > closure:
                flags |= F_FR
        if math.isfinite(C) and C > CLAY_MAX:
            flags |= F_CL
        if math.isfinite(OM) and (OM < 0.0 or OM > OM_MAX):
            flags |= F_OM
        rho = rho_kg / 1000.0
        if math.isfinite(rho_kg) and (rho < RHO_LO or rho > RHO_HI):
            flags |= F_DE
        has_wp, has_33, has_sat, has_air, has_ks = not math.isnan(o_wp), not math.isnan(o_33), not math.isnan(o_sat), not math.isnan(o_air), not math.isnan(o_ks)
        if has_ks and o_ks <= 0.0:
            flags |= F_KO
        clean = (flags & F_IN) == 0
        t = E1R[0] * S + E1R[1] * C + E1R[2] * OM + E1R[3] * S * OM + E1R[4] * C * OM + E1R[5] * S * C + E1R[6]
        wp_n = t + E1A[0] * t + E1A[1]
        t = E2R[0] * S + E2R[1] * C + E2R[2] * OM + E2R[3] * S * OM + E2R[4] * C * OM + E2R[5] * S * C + E2R[6]
        fc_n = t + E2A[0] * t * t + E2A[1] * t + E2A[2]
        t = E3R[0] * S + E3R[1] * C + E3R[2] * OM + E3R[3] * S * OM + E3R[4] * C * OM + E3R[5] * S * C + E3R[6]
        g = t + E3A[0] * t + E3A[1]
        sat_n = fc_n + g + E3S[0] * S + E3S[1]
        t = E4R[0] * S + E4R[1] * C + E4R[2] * g + E4R[3] * S * g + E4R[4] * C * g + E4R[5] * S * C + E4R[6]
        air_n = t + E4A[0] * t * t + E4A[1] * t + E4A[2]
        rho_n = (1.0 - sat_n) * PD
        df = rho / rho_n
        sat_d = 1.0 - rho / PD
        fc_d = fc_n - DFA * (sat_n - sat_d)
        wp = o_wp if has_wp else wp_n
        fc33 = o_33 if has_33 else fc_d
        sat = o_sat if has_sat else sat_d
        air = o_air if has_air else air_n
        gap = sat - fc33
        d_bad = not (df >= DF_LO and df <= DF_HI)
        c_bad = not (wp > 0.0 and wp < fc33 and fc33 < sat and sat < 1.0)
        a_bad = not (air >= 0.0 and air < AIR_MAX)
        g_bad = not (gap >= GAP_MIN)
        if clean:
            if d_bad:
                flags |= F_DF
            if c_bad:
                flags |= F_CU
            if a_bad:
                flags |= F_AI
            if g_bad:
                flags |= F_GA
        curve_ok = clean and not (d_bad or c_bad or a_bad or g_bad)
        B = LNR / (math.log(fc33) - math.log(wp))
        lam = 1.0 / B
        ks_est = (KS_C * math.pow(gap, 3.0 - lam)) / KS_DIV
        ks = o_ks if has_ks else ks_est
        if curve_ok and not (math.isfinite(ks) and ks > 0.0):
            flags |= F_KS
        eq = P_FC <= air
        fc6 = sat if eq else fc33 + SPAN * gap / (AIR_MAX - air)
        prov = 0
        if has_wp:
            prov |= P_WP
        if has_33:
            prov |= P_33
        if has_sat:
            prov |= P_SAT
        if has_air:
            prov |= P_AIR
        if has_ks:
            prov |= P_KS
        if eq:
            prov |= P_EQ
        return flags, fc6, sat, ks, wp, air, prov, eq

    @numba.njit(error_model="numpy", cache=False)
    def kernel(L, cells, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, st, out_fc, out_sat, out_ks, out_wp, out_air, out_prov, osl, osc, want_diag, closure):
        ov = 0
        n_flagged = 0
        first = -1
        n_eq = 0
        for l in range(L):
            for c in range(cells):
                flags, fc6, sat, ks, wp, air, prov, eq = element(
                    a0[l * st[0, 0] + c * st[0, 1]], a1[l * st[1, 0] + c * st[1, 1]], a2[l * st[2, 0] + c * st[2, 1]], a3[l * st[3, 0] + c * st[3, 1]], a4[l * st[4, 0] + c * st[4, 1]],
                    a5[l * st[5, 0] + c * st[5, 1]], a6[l * st[6, 0] + c * st[6, 1]], a7[l * st[7, 0] + c * st[7, 1]], a8[l * st[8, 0] + c * st[8, 1]], a9[l * st[9, 0] + c * st[9, 1]], closure)
                if flags != 0:
                    n_flagged += 1
                    ov |= flags
                    packed = ((l * cells + c) << 16) | flags
                    if first < 0 or packed < first:
                        first = packed
                    continue
                if eq:
                    n_eq += 1
                k = l * osl + c * osc
                out_fc[k] = fc6
                out_sat[k] = sat
                out_ks[k] = ks
                if want_diag != 0:
                    out_wp[k] = wp
                    out_air[k] = air
                    out_prov[k] = prov
        return ov, n_flagged, first, n_eq

    return kernel


def _kernel():
    global _KERNEL
    with _LOCK:
        if _KERNEL is None:
            _KERNEL = _build()
        return _KERNEL


def run(inputs: dict, out: dict, L: int, N: int, cells: int, layout: str, closure: float) -> tuple[int, int, int, int]:
    """Fill `out` (the caller's fresh arrays) from `inputs` (`_Input` records); returns the four diagnostic words of the reference (`first` is all ones when nothing is flagged)."""
    kernel = _kernel()
    nan = np.full(1, np.nan)
    arrays, strides = [], np.zeros((len(tp.ALL_INPUTS), 2), dtype=np.int64)
    for j, key in enumerate(tp.ALL_INPUTS):
        i = inputs.get(key)
        if i is None:
            arrays.append(nan)
        elif i.kind == tp.LAYOUT_SCALAR:
            arrays.append(np.full(1, i.scalar))
        elif i.kind == tp.LAYOUT_SHARED:
            arrays.append(i.array)
            strides[j] = (1, 0)
        else:
            arrays.append(i.array.reshape(-1))
            strides[j] = (N, 1)
    diag = "theta_1500" in out
    dummy_f, dummy_u = np.zeros(1), np.zeros(1, dtype=np.uint8)
    osl, osc = (N, 1) if layout == tp.LAYOUT_CELL else (1, 0)
    flat = lambda a: a.reshape(-1)  # noqa: E731 - views of the caller's C-contiguous outputs
    ov, n_flagged, first, n_eq = kernel(L, cells, *arrays, strides, flat(out["theta_field_capacity_6kpa"]), flat(out["theta_saturation"]), flat(out["saturated_conductivity_m_s"]),
                                        flat(out["theta_1500"]) if diag else dummy_f, flat(out["air_entry_kpa"]) if diag else dummy_f, flat(out["provenance"]) if diag else dummy_u,
                                        osl, osc, 1 if diag else 0, closure)
    return int(ov), int(n_flagged), (0xFFFFFFFFFFFFFFFF if first < 0 else int(first)), int(n_eq)
