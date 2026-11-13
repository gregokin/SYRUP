"""CUDA RawKernel runner of `maple_syrup.soil_texture_properties` (backend "cuda"). Component only; AUTHOR-WRITTEN, NOT RUN ON A DEVICE (no GPU, CuPy initialisation or NVRTC compile was performed by the author).

ONE kernel, one thread per cell, looping the layers (coalesced for CELL `(L, N)` inputs). The element body `maple_syrup_texture_element` is a plain `__device__` function evaluating the arithmetic of
`soil_texture_properties.reference_block` in the same expression order from coefficient macros GENERATED from the same Python tuples (`repr` of a float round-trips), compiled with
`--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`. Library calls (`log`, `pow`) may differ from NumPy by an ulp: only `saturated_conductivity_m_s` involves them and it is compared at the
declared bound (rtol 2e-12, atol 1e-14) by the gated device tests. Failures are reported through device atomics into `DIAG_WORDS = 4` `uint64` words (flag OR, flagged-element count,
`min(element << 16 | flags)`, count of `fc == saturation` outcomes), read back with ONE 32-byte transfer (thread-local accumulation, a few atomics per thread). Scalar and absent inputs are views of ONE small
device array (`aux`, NaN = absent). There is no CuPy reduction, mask/index/flag array, library workspace or full-size download; the host receives the packet and the identity chunks only.

`render_source()` returns the complete source text (what the host-compiler parity test compiles with stand-ins for the CUDA keywords; that checks the arithmetic, flags and diagnostics of the kernel text
against the NumPy reference but NOT a launch, atomics or the device libm: Root's hardware run does). Importing this module never imports CuPy or touches a device.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from maple_syrup import soil_texture_properties as tp

__all__ = ["CUDA_BLOCK", "KERNEL_NAME", "render_source", "run"]

CUDA_BLOCK = tp.CUDA_BLOCK
KERNEL_NAME = "maple_syrup_texture_properties"
_KERNELS: dict[int, Any] = {}
_LOCK = threading.Lock()


def _defs() -> str:
    lines = []

    def add(prefix: str, values: tuple) -> None:
        for k, v in enumerate(values):
            lines.append(f"#define {prefix}{k} ({float(v)!r})")

    add("E1R", tp.EQ1_REGRESSION), add("E1A", tp.EQ1_ADJUST), add("E2R", tp.EQ2_REGRESSION), add("E2A", tp.EQ2_ADJUST), add("E3R", tp.EQ3_REGRESSION), add("E3A", tp.EQ3_ADJUST)
    add("E3S", tp.EQ3_SAT_TERMS), add("E4R", tp.EQ4_REGRESSION), add("E4A", tp.EQ4_ADJUST)
    consts = {"PD": tp.PARTICLE_DENSITY_G_CM3, "DFA": tp.DF_AIR_COEFF, "DF_LO": tp.DENSITY_FACTOR_RANGE[0], "DF_HI": tp.DENSITY_FACTOR_RANGE[1], "RHO_LO": tp.DENSITY_RANGE_G_CM3[0],
              "RHO_HI": tp.DENSITY_RANGE_G_CM3[1], "CLAY_MAX": tp.CLAY_MAX, "OM_MAX": tp.OM_MAX_PERCENT, "GAP_MIN": tp.MIN_DENSITY_GAP, "AIR_MAX": tp.AIR_ENTRY_MAX_KPA, "P_FC": tp.FC_PRESSURE_KPA,
              "SPAN": tp.FC_SPAN_KPA, "LNR": tp.LN_RATIO, "KS_C": tp.KS_COEFF_MM_H, "KS_DIV": tp.MM_H_PER_M_S}
    for k, v in consts.items():
        lines.append(f"#define {k} ({float(v)!r})")
    ints = {"F_NF": tp.FLAG_NONFINITE, "F_FR": tp.FLAG_FRACTION, "F_CL": tp.FLAG_CLAY, "F_OM": tp.FLAG_ORGANIC_MATTER, "F_DE": tp.FLAG_DENSITY, "F_KO": tp.FLAG_KS_OVERRIDE, "F_DF": tp.FLAG_DENSITY_FACTOR,
            "F_CU": tp.FLAG_CURVE, "F_AI": tp.FLAG_AIR_ENTRY, "F_GA": tp.FLAG_DENSITY_GAP, "F_KS": tp.FLAG_KS, "F_IN": tp.INPUT_FLAGS, "P_WP": tp.PROV_THETA_1500, "P_33": tp.PROV_THETA_33,
            "P_SAT": tp.PROV_THETA_SAT, "P_AIR": tp.PROV_AIR, "P_KS": tp.PROV_KS, "P_EQ": tp.PROV_FC_EQ_SAT}
    for k, v in ints.items():
        lines.append(f"#define {k} {int(v)}u")
    return "\n".join(lines) + "\n"


_BODY = r"""
__device__ __forceinline__ unsigned int maple_syrup_texture_element(double S, double Si, double C, double OM, double rho_kg, double o_wp, double o_33, double o_sat, double o_air, double o_ks,
                                                                    double closure, double* fc6p, double* satp, double* ksp, double* wpp, double* airp, unsigned int* provp, int* eqp)
{
    unsigned int flags = 0u;
    const bool finite_in = isfinite(S) && isfinite(Si) && isfinite(C) && isfinite(OM) && isfinite(rho_kg);
    if ((!finite_in) || isinf(o_wp) || isinf(o_33) || isinf(o_sat) || isinf(o_air) || isinf(o_ks)) flags |= F_NF;
    if (isfinite(S) && isfinite(Si) && isfinite(C)) {
        const bool in_range = S >= 0.0 && S <= 1.0 && Si >= 0.0 && Si <= 1.0 && C >= 0.0 && C <= 1.0;
        if ((!in_range) || fabs(((S + Si) + C) - 1.0) > closure) flags |= F_FR;
    }
    if (isfinite(C) && C > CLAY_MAX) flags |= F_CL;
    if (isfinite(OM) && (OM < 0.0 || OM > OM_MAX)) flags |= F_OM;
    const double rho = rho_kg / 1000.0;
    if (isfinite(rho_kg) && (rho < RHO_LO || rho > RHO_HI)) flags |= F_DE;
    const bool has_wp = !isnan(o_wp), has_33 = !isnan(o_33), has_sat = !isnan(o_sat), has_air = !isnan(o_air), has_ks = !isnan(o_ks);
    if (has_ks && o_ks <= 0.0) flags |= F_KO;
    const bool clean = (flags & F_IN) == 0u;
    double t = E1R0 * S + E1R1 * C + E1R2 * OM + E1R3 * S * OM + E1R4 * C * OM + E1R5 * S * C + E1R6;
    const double wp_n = t + E1A0 * t + E1A1;
    t = E2R0 * S + E2R1 * C + E2R2 * OM + E2R3 * S * OM + E2R4 * C * OM + E2R5 * S * C + E2R6;
    const double fc_n = t + E2A0 * t * t + E2A1 * t + E2A2;
    t = E3R0 * S + E3R1 * C + E3R2 * OM + E3R3 * S * OM + E3R4 * C * OM + E3R5 * S * C + E3R6;
    const double g = t + E3A0 * t + E3A1;
    const double sat_n = fc_n + g + E3S0 * S + E3S1;
    t = E4R0 * S + E4R1 * C + E4R2 * g + E4R3 * S * g + E4R4 * C * g + E4R5 * S * C + E4R6;
    const double air_n = t + E4A0 * t * t + E4A1 * t + E4A2;
    const double rho_n = (1.0 - sat_n) * PD;
    const double df = rho / rho_n;
    const double sat_d = 1.0 - rho / PD;
    const double fc_d = fc_n - DFA * (sat_n - sat_d);
    const double wp = has_wp ? o_wp : wp_n;
    const double fc33 = has_33 ? o_33 : fc_d;
    const double sat = has_sat ? o_sat : sat_d;
    const double air = has_air ? o_air : air_n;
    const double gap = sat - fc33;
    const bool d_bad = !(df >= DF_LO && df <= DF_HI);
    const bool c_bad = !(wp > 0.0 && wp < fc33 && fc33 < sat && sat < 1.0);
    const bool a_bad = !(air >= 0.0 && air < AIR_MAX);
    const bool g_bad = !(gap >= GAP_MIN);
    if (clean) {
        if (d_bad) flags |= F_DF;
        if (c_bad) flags |= F_CU;
        if (a_bad) flags |= F_AI;
        if (g_bad) flags |= F_GA;
    }
    const bool curve_ok = clean && !(d_bad || c_bad || a_bad || g_bad);
    const double B = LNR / (log(fc33) - log(wp));
    const double lam = 1.0 / B;
    const double ks_est = (KS_C * pow(gap, 3.0 - lam)) / KS_DIV;
    const double ks = has_ks ? o_ks : ks_est;
    if (curve_ok && !(isfinite(ks) && ks > 0.0)) flags |= F_KS;
    const bool eq = P_FC <= air;
    *fc6p = eq ? sat : fc33 + SPAN * gap / (AIR_MAX - air);
    *satp = sat; *ksp = ks; *wpp = wp; *airp = air; *eqp = eq ? 1 : 0;
    unsigned int prov = 0u;
    if (has_wp) prov |= P_WP;
    if (has_33) prov |= P_33;
    if (has_sat) prov |= P_SAT;
    if (has_air) prov |= P_AIR;
    if (has_ks) prov |= P_KS;
    if (eq) prov |= P_EQ;
    *provp = prov;
    return flags;
}

extern "C" __global__ void maple_syrup_texture_properties(
    const double* a0, long long s0l, long long s0c, const double* a1, long long s1l, long long s1c, const double* a2, long long s2l, long long s2c,
    const double* a3, long long s3l, long long s3c, const double* a4, long long s4l, long long s4c, const double* a5, long long s5l, long long s5c,
    const double* a6, long long s6l, long long s6c, const double* a7, long long s7l, long long s7c, const double* a8, long long s8l, long long s8c,
    const double* a9, long long s9l, long long s9c,
    double* out_fc, double* out_sat, double* out_ks, double* out_wp, double* out_air, unsigned char* out_prov, long long osl, long long osc,
    unsigned long long* diag, double closure, long long L, long long cells, int want_diag)
{
    const long long c = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (c >= cells) return;
    unsigned long long ov = 0ull, n_flagged = 0ull, first = 0xFFFFFFFFFFFFFFFFull, n_eq = 0ull;
    for (long long l = 0; l < L; ++l) {
        double fc6, sat, ks, wp, air;
        unsigned int prov;
        int eq;
        const unsigned int flags = maple_syrup_texture_element(
            a0[l * s0l + c * s0c], a1[l * s1l + c * s1c], a2[l * s2l + c * s2c], a3[l * s3l + c * s3c], a4[l * s4l + c * s4c], a5[l * s5l + c * s5c], a6[l * s6l + c * s6c],
            a7[l * s7l + c * s7c], a8[l * s8l + c * s8c], a9[l * s9l + c * s9c], closure, &fc6, &sat, &ks, &wp, &air, &prov, &eq);
        if (flags != 0u) {
            n_flagged += 1ull;
            ov |= (unsigned long long)flags;
            const unsigned long long packed = (((unsigned long long)(l * cells + c)) << 16) | (unsigned long long)flags;
            if (packed < first) first = packed;
            continue;
        }
        n_eq += (unsigned long long)eq;
        const long long k = l * osl + c * osc;
        out_fc[k] = fc6; out_sat[k] = sat; out_ks[k] = ks;
        if (want_diag) { out_wp[k] = wp; out_air[k] = air; out_prov[k] = (unsigned char)prov; }
    }
    if (ov) { atomicOr(&diag[0], ov); atomicAdd(&diag[1], n_flagged); atomicMin(&diag[2], first); }
    if (n_eq) atomicAdd(&diag[3], n_eq);
}
"""


def render_source() -> str:
    """The complete kernel source text (macros generated from the recipe tuples + the body)."""
    return _defs() + _BODY


def _load_kernel(cp, device_id: int):
    with _LOCK:
        fn = _KERNELS.get(device_id)
        if fn is None:
            try:
                module = cp.RawModule(code=render_source(), options=tp.COMPILE_OPTIONS, backend="nvrtc")
                fn = module.get_function(KERNEL_NAME)
            except Exception as exc:  # noqa: BLE001 - a setup failure, never a fallback
                raise tp.TexturePropertyBackendUnavailableError(f"the CUDA texture-property kernel could not be compiled/loaded on device {device_id} ({type(exc).__name__}: {exc})") from exc
            _KERNELS[device_id] = fn
        return fn


def run(cp, inputs: dict, out: dict, L: int, N: int, cells: int, layout: str, closure: float) -> tuple[int, int, int, int]:
    """One launch. Returns the four diagnostic words (`first` is all ones when nothing is flagged). Allocates only the 4-word diagnostic and ONE small auxiliary array of scalar inputs."""
    from maple_syrup import soil_water_remap_cuda as rc

    n_in = len(tp.ALL_INPUTS)
    aux = cp.full(n_in, np.nan, dtype=cp.float64)  # aux[j] = the scalar value of input j; NaN stands for an absent override
    args = []
    for j, key in enumerate(tp.ALL_INPUTS):
        i = inputs.get(key)
        if i is None:
            args += [aux[j:j + 1], np.int64(0), np.int64(0)]
        elif i.kind == tp.LAYOUT_SCALAR:
            aux[j] = i.scalar
            args += [aux[j:j + 1], np.int64(0), np.int64(0)]
        elif i.kind == tp.LAYOUT_SHARED:
            args += [i.array, np.int64(1), np.int64(0)]
        else:
            args += [i.array, np.int64(N), np.int64(1)]
    diag_on = "theta_1500" in out
    osl, osc = (N, 1) if layout == tp.LAYOUT_CELL else (1, 0)
    wp = out["theta_1500"] if diag_on else out["theta_saturation"]  # never written when the diagnostics are off
    air = out["air_entry_kpa"] if diag_on else out["theta_saturation"]
    prov = out["provenance"] if diag_on else out["theta_saturation"]
    diag = cp.zeros(tp.DIAG_WORDS, dtype=cp.uint64)
    diag[2] = np.uint64(0xFFFFFFFFFFFFFFFF)
    fn = _load_kernel(cp, rc.current_device_id(cp))
    grid = (cells + CUDA_BLOCK - 1) // CUDA_BLOCK
    fn((grid,), (CUDA_BLOCK,), tuple(args) + (out["theta_field_capacity_6kpa"], out["theta_saturation"], out["saturated_conductivity_m_s"], wp, air, prov, np.int64(osl), np.int64(osc), diag,
                                              np.float64(closure), np.int64(L), np.int64(cells), np.int32(1 if diag_on else 0)))
    from maple.core.backend import to_host

    words = np.asarray(to_host(diag))  # the one validation-packet read (DIAG_WORDS uint64 = 32 B) through MAPLE's counted transfer path; the cold identity chunks are separate bounded reads
    return int(words[0]), int(words[1]), int(words[2]), int(words[3])
