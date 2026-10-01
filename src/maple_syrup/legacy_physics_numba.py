"""Compiled CPU evaluation of the wet MAHLERAN laws for the frozen-composition legacy replay (Phase 7f).

`sediment_physics_step` (NumPy/CuPy, unchanged) stays the reference. This module evaluates the SAME
equations, units, regime boundaries (Re 500/2500, D* 10, suspension `>=`), caps and conventions in ONE
serial Numba loop over cells, after a one-time preparation of everything that is constant in a legacy
replay: the class fractions and median diameter of the (frozen) initial holdings, the cap rates, the
vegetation energy factor, the slope power terms and every grain-only constant. Nothing here changes
physics: expression and multiplication order follow `sediment_physics_step` term by term so that the
only expected float differences are libm-versus-NumPy-SIMD `exp/log/pow` differences (a few ulp).
Counts, regimes and masks come from comparisons of exactly-rounded quantities and are identical except at
a knife edge of a transcendental threshold (`xs > 0`, `L > 0`), which is documented in
docs/phase7f/compiled_physics.md.

Contract:

- The context OWNS read-only arrays derived from the parameters, grid, vegetation and the ORIGINAL holdings
  (fractions, d50, cap rates, ...; the holdings themselves are not stored); mutating the caller's arrays after
  preparation cannot change it. It is valid only for a frozen composition
  (evolving-bed holdings must NOT be prepared; build a new context or use the reference function).
- Host NumPy only. A CuPy parameter set, grid or array is refused; no transfer is made.
- Every call allocates fresh output arrays (no buffer is reused, so an earlier result is never changed by
  a later call). Failure raises `SedimentPhysicsError` before any result exists.
- Validation matches the reference: shapes, dtypes, namespace, finite and nonnegative dynamic inputs,
  zero velocity on inactive cells, positive finite dt, and refusal of any non-finite law intermediate or
  output, including masked/unused ones (overflow is refused, not clipped). The kernel evaluates the same
  intermediates as the reference for every cell and class and reports a bit set; the wrapper raises.
- Missing Numba raises `LegacyPhysicsNumbaUnavailableError`; there is no fallback to the array path.
  `fastmath` is off; `error_model="numpy"` makes division by zero produce inf/nan (then refused) instead
  of a Python exception.
- No GPU kernel. The prepared arrays are flat C-contiguous FP64 so a later device implementation can
  reuse the same layout without changing the physics.

Nothing here was run by its author (file-only tools); see docs/phase7f/compiled_physics.md.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup.sediment_physics import (
    BAGNOLD_DEPTH_UNITS,
    CONCENTRATED_DISTANCE_CAP_M,
    DISTANCE_CONVENTIONS,
    DSTAR_CONVENTIONS,
    GRAVITY_M_S2,
    KE_MODELS,
    KE_VEGETATION_FORMS,
    LEGACY_MEDIAN_FACTOR,
    RAINDROP_COMPOSITION_SCALINGS,
    REGIME_CODES,
    REYNOLDS_CONCENTRATED,
    REYNOLDS_TRANSITIONAL,
    SUSPENDED_EXPONENT_CAP,
    VEGETATION_ENERGY_COEFFICIENT_PER_PERCENT,
    PhysicsGrid,
    SedimentPhysicsError,
    SedimentPhysicsParameters,
    SedimentPhysicsStep,
    _positive,
    _require_grid_arrays,
    median_diameter_m,
    recession_velocity,
)

__all__ = [
    "LegacyPhysicsContext",
    "LegacyPhysicsNumbaUnavailableError",
    "compiled_kernel",
    "legacy_physics_step",
    "numba_available",
    "prepare_legacy_physics",
    "reset_compiled",
]

_KERNEL: Any = None

# Bit positions of the kernel's failure flags; `_FLAG_MESSAGES[i]` describes bit i.
_FLAG_MESSAGES = (
    "depth_m must be finite everywhere",  # 0
    "depth_m must be >= 0 everywhere",  # 1
    "velocity_m_s must be finite everywhere",  # 2
    "velocity_m_s must be >= 0 everywhere",  # 3
    "rain_rate_m_per_s must be finite everywhere",  # 4
    "rain_rate_m_per_s must be >= 0 everywhere",  # 5
    "previous_sediment_velocity_m_s must be finite everywhere",  # 6
    "previous_sediment_velocity_m_s must be >= 0 everywhere",  # 7
    "velocity_m_s must be 0 on inactive cells",  # 8
    "physics intermediate rain intensity is non-finite",  # 9
    "physics intermediate rain energy term is non-finite",  # 10
    "physics intermediate raindrop law mass is non-finite",  # 11
    "physics intermediate rain energy flux is non-finite",  # 12
    "physics intermediate diffuse velocity is non-finite",  # 13
    "physics intermediate diffuse travel distance is non-finite",  # 14
    "physics intermediate Bagnold threshold is non-finite",  # 15
    "physics intermediate concentrated travel distance is non-finite",  # 16
    "physics intermediate concentrated velocity is non-finite",  # 17
    "physics intermediate suspended travel distance is non-finite",  # 18
    "physics intermediate dimensionless shear is non-finite",  # 19
    "physics intermediate decayed memory velocity is non-finite",  # 20
    "physics produced non-finite requested_pickup_kg",  # 21
    "physics produced non-finite raindrop_pickup_kg",  # 22
    "physics produced non-finite flow_pickup_kg",  # 23
    "physics produced non-finite sediment_velocity_m_s",  # 24
    "physics produced non-finite deposition_rate_per_m",  # 25
    "physics produced non-finite shear_velocity_m_s",  # 26
    "physics produced non-finite reynolds_number",  # 27
    "physics produced non-finite stream_power_w_m2",  # 28
    "physics produced non-finite rain_energy_j_m2_mm",  # 29
    "physics produced non-finite rain_energy_flux_j_m2_s",  # 30
    "physics produced non-finite pickup_probability",  # 31
    "physics produced negative requested_pickup_kg",  # 32
    "physics produced negative raindrop_pickup_kg",  # 33
    "physics produced negative flow_pickup_kg",  # 34
    "physics produced negative sediment_velocity_m_s",  # 35
    "physics produced negative deposition_rate_per_m",  # 36
    "physics produced negative shear_velocity_m_s",  # 37
    "physics produced negative reynolds_number",  # 38
    "physics produced negative stream_power_w_m2",  # 39
    "physics produced negative rain_energy_j_m2_mm",  # 40
    "physics produced negative rain_energy_flux_j_m2_s",  # 41
    "physics produced negative pickup_probability",  # 42
    "pickup probability exceeded 1",  # 43
)

# Row indices of the per-class constant table and slots of the scalar / config vectors.
_C_A, _C_B, _C_SPQ, _C_THETA_DEN, _C_DPOW, _C_EXP_D, _C_PM, _C_PM_POW, _C_SUSP = range(9)
_N_CLS = 9
_S_RHO, _S_REF, _S_MASS, _S_NU, _S_HZ, _S_RESERVED, _S_BAG_SCALE, _S_DECAY, _S_P_PAR = range(9)
_N_SC = 9
_K_KE_MODEL, _K_VEG_LITERAL, _K_FRACTION_SCALING, _K_DIST_LITERAL, _K_DEPTH_MM = range(5)
_N_CFG = 5

_QUANSAH = 1.2e3  # sediment_physics._LEGACY_QUANSAH_SCALE


class LegacyPhysicsNumbaUnavailableError(SedimentPhysicsError):
    """The compiled legacy physics was requested but Numba cannot be imported. No fallback."""


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def reset_compiled() -> None:
    """Drop the compiled dispatcher (tests: cold-start and missing-Numba paths)."""
    global _KERNEL
    _KERNEL = None


# --- kernel -------------------------------------------------------------------------------
def _build_kernel(numba: Any) -> Any:
    """Define the helpers and the kernel as nopython functions (Numba is imported lazily by the caller)."""
    # Shared with the reference so a constant cannot drift between the two implementations.
    re_conc = REYNOLDS_CONCENTRATED
    re_trans = REYNOLDS_TRANSITIONAL
    conc_cap = CONCENTRATED_DISTANCE_CAP_M
    susp_cap = SUSPENDED_EXPONENT_CAP
    grav = GRAVITY_M_S2
    median = LEGACY_MEDIAN_FACTOR
    jit = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")

    @jit
    def fmax(a, b):  # np.maximum: NaN propagates
        # `x != x` is the explicit NaN test; builtin max() would not propagate NaN or define signed-zero ties.
        if a != a or b != b:  # noqa: PLR0124
            return np.nan
        return a if a > b else b  # noqa: FURB136

    @jit
    def fmin(a, b):  # np.minimum: NaN propagates
        if a != a or b != b:  # noqa: PLR0124
            return np.nan
        return a if a < b else b  # noqa: FURB136

    @jit
    def pw(x, y):
        # every runtime exponent is a positive non-integer or a positive class constant; a negative base is
        # NaN exactly as in NumPy (flagged by the caller), and 0 ** y = 0 for y > 0.
        if x < 0.0:
            return np.nan
        if x == 0.0:
            return 0.0
        return x ** y

    def kernel(depth, vel, rain, prev, active, slope, veg_factor, d50, bag_c1, slope_pow, fractions, cap_rate,
               cls, sc, cfg, requested, rdp, fdp, svel, rate, law, settle, regime, ustar_o, re_o, sp_o, ke_o,
               kef_o, p_o, capapp, d50_o, counts):
        n = depth.shape[0]
        nc = cls.shape[1]
        rho = sc[_S_RHO]
        ref = sc[_S_REF]
        mass = sc[_S_MASS]
        nu = sc[_S_NU]
        hz = sc[_S_HZ]
        bag_scale = sc[_S_BAG_SCALE]
        decay = sc[_S_DECAY]
        p_par = sc[_S_P_PAR]
        ke_model = cfg[_K_KE_MODEL]
        veg_literal = cfg[_K_VEG_LITERAL]
        fraction_scaling = cfg[_K_FRACTION_SCALING]
        dist_literal = cfg[_K_DIST_LITERAL]
        depth_mm = cfg[_K_DEPTH_MM]
        flags = np.int64(0)
        for i in range(n):
            d = depth[i]
            v = vel[i]
            r = rain[i]
            a = active[i]
            s = slope[i]
            if not np.isfinite(d):
                flags |= np.int64(1) << 0
            if d < 0.0:
                flags |= np.int64(1) << 1
            if not np.isfinite(v):
                flags |= np.int64(1) << 2
            if v < 0.0:
                flags |= np.int64(1) << 3
            if not np.isfinite(r):
                flags |= np.int64(1) << 4
            if r < 0.0:
                flags |= np.int64(1) << 5
            if (not a) and v != 0.0:
                flags |= np.int64(1) << 8

            ustar = np.sqrt(grav * d * s)
            re = v * d / nu
            spw = 1000.0 * grav * d * v * s
            rm = r * 1000.0
            inten = r * 3.6e6
            wet = a and d > 0.0
            raining = a and r > 0.0
            conc = wet and re >= re_conc
            trans = wet and (not conc) and re > re_trans
            rain_b = wet and (not conc) and raining
            dry_b = wet and (not conc) and (not raining)
            flow_cell = conc or trans
            if conc:
                rcell = 5
            elif rain_b and trans:
                rcell = 3
            elif rain_b:
                rcell = 2
            elif dry_b and trans:
                rcell = 4
            elif wet:
                rcell = 1
            else:
                rcell = 0

            # rain energy (raindrop_detachment 24-54) and its flux (diffuse_flow_transport 26-27)
            if raining:
                lg = np.log10(inten)
                if ke_model == 0:
                    ke = fmax(11.9 + 8.73 * lg, 0.0) * veg_factor[i]
                elif veg_literal == 1:
                    ke = 29.0 - 20.88 * np.exp(-180.0 * rm) * veg_factor[i]
                else:
                    ke = (29.0 - 20.88 * np.exp(-180.0 * rm)) * veg_factor[i]
                kf = fmax((11.9 + 8.73 * lg) * rm, 0.0)
            else:
                ke = 0.0
                kf = 0.0
            energy = fmax(ke * rm * 1.2e3, 0.0)
            if not np.isfinite(inten):
                flags |= np.int64(1) << 9
            if not np.isfinite(energy):
                flags |= np.int64(1) << 10
            if not np.isfinite(kf):
                flags |= np.int64(1) << 12
                flags |= np.int64(1) << 30
            if kf < 0.0:
                flags |= np.int64(1) << 41
            if not np.isfinite(ke):
                flags |= np.int64(1) << 29
            if ke < 0.0:
                flags |= np.int64(1) << 40
            if not np.isfinite(ustar):
                flags |= np.int64(1) << 26
            if ustar < 0.0:
                flags |= np.int64(1) << 37
            if not np.isfinite(re):
                flags |= np.int64(1) << 27
            if re < 0.0:
                flags |= np.int64(1) << 38
            if not np.isfinite(spw):
                flags |= np.int64(1) << 28
            if spw < 0.0:
                flags |= np.int64(1) << 39

            # per-cell transport terms shared by every class (same multiplication order as the reference)
            vd_b = (0.525 * pw(kf, 2.35)) * pw(spw, 0.981)
            ld_b = (5.0e-2 * pw(kf, 1.85)) * pw(spw, 0.481)
            d50i = d50[i]
            dfl = d * 1000.0 if depth_mm == 1 else d * 1.0
            log_arg = 12.0 * dfl / d50i
            la = log_arg if log_arg > 0.0 else 1.0
            bag = fmax(bag_c1[i] * np.log10(la), 0.0)
            xs = spw - bag
            capacity = xs > 0.0
            xsp = xs if capacity else 0.0
            lc_a = 2.85e-3 * pw(xsp, 1.31)
            vc = fmin(((1.92e-2 * pw(xsp, 1.01)) * (1000.0 / 3600.0)) * 1.0e-3, v)
            spf = fmin(7.331976e-3 * spw, susp_cap)
            es = 727.51805244 * np.exp(spf)
            if not np.isfinite(bag):
                flags |= np.int64(1) << 15
            if not np.isfinite(vc):
                flags |= np.int64(1) << 17

            if a:
                ustar_o[i] = ustar
                re_o[i] = re
                sp_o[i] = spw
            else:
                ustar_o[i] = 0.0
                re_o[i] = 0.0
                sp_o[i] = 0.0
            ke_o[i] = ke
            kef_o[i] = kf
            d50_o[i] = d50i

            for k in range(nc):
                idx = i * nc + k
                f = fractions[idx]
                cap = cap_rate[idx]

                # --- raindrop law (all cells for the law-mass check; applied on rain_b cells) ---
                pwe = 0.0 if energy == 0.0 else pw(energy, cls[_C_B, k])
                x = (cls[_C_A, k] * pwe) * slope_pow[idx]
                if not np.isfinite(x):
                    flags |= np.int64(1) << 11
                rd = 0.0
                rcap = False
                if rain_b:
                    rd = 2.0 * x / rho / ref
                    rd = rd * np.exp(-cls[_C_SPQ, k] * (d * 100.0))
                    rd = fmax(rd, 0.0)
                    if fraction_scaling == 1:
                        rd = rd * f
                    if k == 1 and rd > cap:
                        rd = cap
                        rcap = True
                    if not (f > 0.0):
                        rd = 0.0

                # --- flow detachment (flow_detachment 18-47) ---
                theta = (ustar * ustar) / cls[_C_THETA_DEN, k]
                if not np.isfinite(theta):
                    flags |= np.int64(1) << 19
                pos = theta > 0.0
                arg = theta if pos else 1.0
                pc = np.log(0.049 / (arg * 0.25))
                t = pc / 0.702
                inner = 1.0 - np.exp(p_par * (t * t))
                if pc > 0.0:
                    sg = 1.0
                elif pc < 0.0:
                    sg = -1.0
                elif pc == 0.0:
                    sg = 0.0
                else:
                    sg = np.nan
                p = 0.5 - (0.5 * sg) * np.sqrt(fmax(inner, 0.0))
                if not pos:
                    p = 0.0
                if not np.isfinite(p):
                    flags |= np.int64(1) << 31
                if p < 0.0:
                    flags |= np.int64(1) << 42
                if p > 1.0:
                    flags |= np.int64(1) << 43
                fd = ((p * hz) * f) / ref
                fcap = False
                if flow_cell and fd > cap:
                    fd = cap
                    fcap = True
                if not (f > 0.0):
                    fd = 0.0
                if not flow_cell:
                    fd = 0.0
                rp = rd * mass
                fp = fd * mass
                req = rp + fp
                if not np.isfinite(rp):
                    flags |= np.int64(1) << 22
                if rp < 0.0:
                    flags |= np.int64(1) << 33
                if not np.isfinite(fp):
                    flags |= np.int64(1) << 23
                if fp < 0.0:
                    flags |= np.int64(1) << 34
                if a:
                    requested[idx] = req
                    rdp[idx] = rp
                    fdp[idx] = fp
                    p_o[idx] = p
                    if not np.isfinite(req):
                        flags |= np.int64(1) << 21
                    if req < 0.0:
                        flags |= np.int64(1) << 32
                else:
                    requested[idx] = 0.0
                    rdp[idx] = 0.0
                    fdp[idx] = 0.0
                    p_o[idx] = 0.0
                capapp[idx] = a and (rcap or fcap)

                # --- transport laws ---
                vd = fmin((vd_b / cls[_C_PM, k]) * (1.0e-2 / 60.0), v)
                ld = ld_b * cls[_C_PM_POW, k]
                lc = lc_a * cls[_C_DPOW, k]
                ls = es * cls[_C_EXP_D, k]
                if dist_literal == 1:
                    lc = lc * median
                    ls = ls * median
                lc = fmin(lc, conc_cap)
                ls = ls * bag_scale
                if not np.isfinite(vd):
                    flags |= np.int64(1) << 13
                if not np.isfinite(ld):
                    flags |= np.int64(1) << 14
                if not np.isfinite(lc):
                    flags |= np.int64(1) << 16
                if not np.isfinite(ls):
                    flags |= np.int64(1) << 18
                dec = prev[idx] * decay
                if not np.isfinite(dec):
                    flags |= np.int64(1) << 20
                if not np.isfinite(prev[idx]):
                    flags |= np.int64(1) << 6
                if prev[idx] < 0.0:
                    flags |= np.int64(1) << 7

                susp = conc and (ustar >= cls[_C_SUSP, k])
                conc_class = (conc and (not susp)) or (dry_b and trans)
                diff_class = rain_b
                lawk = susp or conc_class or diff_class
                if susp:
                    vl = v
                    ll = ls
                elif conc_class:
                    vl = vc
                    ll = lc
                else:
                    vl = vd
                    ll = ld
                no_cap = (conc_class and (not capacity)) or (diff_class and (not (kf > 0.0)))
                st = (not wet) or no_cap
                ldef = lawk and (not no_cap) and (ll > 0.0)
                if ldef:
                    rt = 1.0 / ll
                    vel_k = vl
                else:
                    rt = 0.0
                    vel_k = dec
                if not a:
                    vel_k = 0.0
                if st:
                    vel_k = 0.0
                if not np.isfinite(rt):
                    flags |= np.int64(1) << 25
                if rt < 0.0:
                    flags |= np.int64(1) << 36
                if not np.isfinite(vel_k):
                    flags |= np.int64(1) << 24
                if vel_k < 0.0:
                    flags |= np.int64(1) << 35
                svel[idx] = vel_k
                rate[idx] = rt
                law[idx] = ldef
                settle[idx] = st
                rk = 6 if susp else rcell
                regime[idx] = rk
                if a:
                    counts[rk] += 1
        return flags

    # Not cached on disk: the kernel closes over other dispatchers, which Numba's cache cannot serialize.
    return jit(kernel)


def compiled_kernel() -> Any:
    """The nopython-compiled kernel (compiled lazily on first call, once per process)."""
    global _KERNEL
    if _KERNEL is None:
        try:
            import numba
        except ImportError as exc:
            raise LegacyPhysicsNumbaUnavailableError(
                "the compiled legacy physics was requested but Numba is not installed (optional extra "
                "maple-syrup[numba]); there is no fallback to the array implementation"
            ) from exc
        _KERNEL = _build_kernel(numba)
    return _KERNEL


# --- prepared context -----------------------------------------------------------------------
def _owned(array: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
    out = np.array(array, dtype=dtype, order="C", copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class LegacyPhysicsContext:
    """Frozen-composition static data for `legacy_physics_step`. Built by `prepare_legacy_physics`; every
    array is an owned, read-only, flat C-contiguous copy (cell index `i = iy * nx + ix`, class-minor).

    `preparation_s` is the wall time of the preparation itself, excluding Numba compilation (which happens
    on the first step)."""

    shape: tuple[int, int]
    n_classes: int
    cell_area_m2: float
    ke_model: str
    particle_density_kg_m3: float
    reference_interval_s: float
    recession_factor_per_reference_s: float
    kinematic_viscosity_m2_s: float
    flow_detachment_depth_scale_m: float
    bagnold_density_scale: float
    config: Any
    class_constants: Any  # (9, nc)
    p_par: float
    active: Any  # (n,) bool
    slope: Any  # (n,)
    veg_factor: Any  # (n,)
    d50: Any  # (n,)
    bagnold_prefactor: Any  # (n,)
    slope_power: Any  # (n * nc,)
    fractions: Any  # (n * nc,)
    cap_rate: Any  # (n * nc,)
    preparation_s: float

    def step(self, depth_m: Any, velocity_m_s: Any, rain_rate_m_per_s: Any,
             previous_sediment_velocity_m_s: Any, dt_s: float) -> SedimentPhysicsStep:
        return legacy_physics_step(self, depth_m, velocity_m_s, rain_rate_m_per_s,
                                   previous_sediment_velocity_m_s, dt_s)

    def nbytes(self) -> int:
        return int(sum(getattr(self, name).nbytes for name in (
            "config", "class_constants", "active", "slope", "veg_factor", "d50", "bagnold_prefactor", "slope_power",
            "fractions", "cap_rate")))

    def summary(self) -> dict[str, Any]:
        return {"implementation": "numba", "frozen_composition": True, "shape": list(self.shape),
                "n_classes": self.n_classes, "static_bytes": self.nbytes(), "preparation_s": self.preparation_s,
                "fastmath": False, "error_model": "numpy"}


def _host_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    from maple.core.backend import is_array

    if not is_array(value):
        raise SedimentPhysicsError(f"{name} must be a NumPy array, got {type(value).__name__}")
    if not isinstance(value, np.ndarray):
        raise SedimentPhysicsError(f"{name} must be a host NumPy array (compiled legacy physics is CPU only; "
                                   f"no host/device transfer is made), got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise SedimentPhysicsError(f"{name} shape {tuple(value.shape)} != {shape}")
    if value.dtype != np.float64:
        raise SedimentPhysicsError(f"{name} must be float64, got {value.dtype}")
    return value


def _validate_parameters(params: Any) -> None:
    """Static parameter checks; the dataclass can be built by hand, so do not trust it blindly."""
    if not isinstance(params, SedimentPhysicsParameters):
        raise SedimentPhysicsError("params must be SedimentPhysicsParameters")
    if params.xp is not np:
        raise SedimentPhysicsError(f"compiled legacy physics is host NumPy only; parameters are in "
                                   f"{params.xp.__name__!r} and no transfer is made")
    nc = params.n_classes
    if not isinstance(nc, (int, np.integer)) or isinstance(nc, bool) or nc < 1:
        raise SedimentPhysicsError(f"n_classes must be a positive integer, got {nc!r}")
    for name, options, value in (("ke_model", KE_MODELS, params.ke_model),
                                 ("ke_vegetation_form", KE_VEGETATION_FORMS, params.ke_vegetation_form),
                                 ("raindrop_composition_scaling", RAINDROP_COMPOSITION_SCALINGS,
                                  params.raindrop_composition_scaling),
                                 ("distance_convention", DISTANCE_CONVENTIONS, params.distance_convention),
                                 ("dstar_convention", DSTAR_CONVENTIONS, params.dstar_convention),
                                 ("bagnold_depth_units", BAGNOLD_DEPTH_UNITS, params.bagnold_depth_units)):
        if value not in options:
            raise SedimentPhysicsError(f"{name} must be one of {options}, got {value!r}")
    for name in ("particle_density_kg_m3", "flow_detachment_depth_scale_m", "kinematic_viscosity_m2_s",
                 "reference_interval_s", "recession_factor_per_reference_s", "excess_density_kg_m3",
                 "bagnold_density_scale", "dstar_const_per_m", "sigma"):
        value = getattr(params, name)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
                or not math.isfinite(float(value)):
            raise SedimentPhysicsError(f"parameter {name} must be a finite real number, got {value!r}")
    for name in ("particle_density_kg_m3", "kinematic_viscosity_m2_s", "reference_interval_s", "sigma",
                 "bagnold_density_scale", "dstar_const_per_m"):
        if not getattr(params, name) > 0.0:
            raise SedimentPhysicsError(f"parameter {name} must be > 0")
    if params.flow_detachment_depth_scale_m < 0.0:
        raise SedimentPhysicsError("parameter flow_detachment_depth_scale_m must be >= 0")
    if not (0.0 < params.recession_factor_per_reference_s < 1.0):
        raise SedimentPhysicsError("parameter recession_factor_per_reference_s must lie in (0, 1)")
    for name in ("d_diameter", "d_raindrop_a", "d_raindrop_b", "d_raindrop_c", "d_attenuation", "d_raindrop_max",
                 "d_settling", "d_particle_mass_g"):
        array = getattr(params, name)
        if not isinstance(array, np.ndarray) or array.dtype != np.float64 or array.shape != (1, 1, nc):
            raise SedimentPhysicsError(f"parameter {name} must be a host float64 array of shape (1, 1, {nc})")
        if not np.all(np.isfinite(array)):
            raise SedimentPhysicsError(f"parameter {name} must be finite")
    if np.any(params.d_diameter <= 0.0) or np.any(np.diff(params.d_diameter[0, 0]) <= 0.0):
        raise SedimentPhysicsError("diameters must be positive and strictly increasing")
    if np.any(params.d_raindrop_b <= 0.0):
        raise SedimentPhysicsError("raindrop exponents b must be > 0")
    if np.any(params.d_raindrop_a < 0.0) or np.any(params.d_raindrop_c < 0.0) \
            or np.any(params.d_attenuation < 0.0) or np.any(params.d_raindrop_max < 0.0):
        raise SedimentPhysicsError("raindrop a, c, attenuation and maximum depth must be >= 0")
    if np.any(params.d_particle_mass_g <= 0.0) or np.any(params.d_settling < 0.0):
        raise SedimentPhysicsError("particle mass must be > 0 and settling velocity >= 0")


def prepare_legacy_physics(params: SedimentPhysicsParameters, grid: PhysicsGrid,
                           vegetation_cover_fraction: Any, active_layer_mass_kg: Any) -> LegacyPhysicsContext:
    """Validate once and precompute everything invariant in a FROZEN-composition legacy replay.

    `active_layer_mass_kg` is the ORIGINAL holdings (the legacy `sed_propn` is fixed); only quantities derived from it are
    stored, and it is never read again after preparation. Do not call this with holdings that evolve between steps."""
    from maple.core.backend import errstate

    t0 = time.perf_counter()
    _validate_parameters(params)
    if not isinstance(grid, PhysicsGrid) or grid.xp is not np:
        raise SedimentPhysicsError("grid must be a host NumPy PhysicsGrid (compiled legacy physics is CPU only)")
    ny, nx = (int(v) for v in grid.shape)
    nc = int(params.n_classes)
    n = ny * nx
    slope2 = _host_array(grid.slope, "grid.slope", (ny, nx))
    active_in = grid.active
    if not isinstance(active_in, np.ndarray) or active_in.dtype != np.bool_ or active_in.shape != (ny, nx):
        raise SedimentPhysicsError("grid.active must be a host bool array of the grid shape")
    if not np.all(np.isfinite(slope2)):
        raise SedimentPhysicsError("grid.slope must be finite")
    if not np.all(slope2[active_in] >= 0.0):
        raise SedimentPhysicsError("grid.slope must be >= 0 on active cells")
    area = _positive(grid.cell_area_m2, "cell_area_m2")
    veg = _host_array(vegetation_cover_fraction, "vegetation_cover_fraction", (ny, nx))
    hold = _host_array(active_layer_mass_kg, "active_layer_mass_kg", (ny, nx, nc))
    if not np.all(np.isfinite(veg)) or np.any(veg < 0.0) or np.any(veg > 1.0):
        raise SedimentPhysicsError("vegetation_cover_fraction must be finite and within [0, 1]")
    if not np.all(np.isfinite(hold)) or np.any(hold < 0.0):
        raise SedimentPhysicsError("active_layer_mass_kg must be finite and >= 0 everywhere")

    ref = float(params.reference_interval_s)
    diam = params.d_diameter[0, 0]
    with errstate(xp=np, all="ignore"):
        total = np.sum(hold, axis=-1)
        if not np.all(np.isfinite(total)):
            raise SedimentPhysicsError(
                "active_layer_mass_kg per-cell total overflowed FP64 (unsupported holdings)")
        fractions = hold / np.where(total > 0.0, total, 1.0)[..., None]
        fractions = np.where((total > 0.0)[..., None], fractions, 0.0)
        d50 = median_diameter_m(fractions, diam)
        d50 = np.where(active_in, d50, float(params.diameter_m[-1]))
        veg_factor = 1.0 - VEGETATION_ENERGY_COEFFICIENT_PER_PERCENT * (veg * 100.0)
        cap_rate = fractions * params.d_raindrop_max / ref
        slope_power = (slope2 * 100.0)[..., None] ** params.d_raindrop_c
        bag_c1 = 4.554e-3 * (params.excess_density_kg_m3 * d50) ** 1.5
        dstar = params.d_diameter * params.dstar_const_per_m
        susp_crit = np.where(dstar <= 10.0, 4.0 * params.d_settling / dstar, 0.4 * params.d_settling)
        cls = np.empty((_N_CLS, nc), dtype=np.float64)
        cls[_C_A] = (params.d_raindrop_a / _QUANSAH)[0, 0]
        cls[_C_B] = params.d_raindrop_b[0, 0]
        cls[_C_SPQ] = params.d_attenuation[0, 0]
        cls[_C_THETA_DEN] = (params.sigma * GRAVITY_M_S2 * params.d_diameter)[0, 0]
        cls[_C_DPOW] = (params.d_diameter ** (-0.94))[0, 0]
        cls[_C_EXP_D] = np.exp(-6.12683698 * params.d_diameter * 1000.0)[0, 0]
        cls[_C_PM] = params.d_particle_mass_g[0, 0]
        cls[_C_PM_POW] = (params.d_particle_mass_g ** (-0.425))[0, 0]
        cls[_C_SUSP] = susp_crit[0, 0]
    # Static quantities the reference also refuses when non-finite (it checks them every step).
    for name, array in (("cap rate", cap_rate), ("raindrop slope power", slope_power),
                        ("Bagnold prefactor", bag_c1), ("vegetation factor", veg_factor),
                        ("grain constants", cls), ("median diameter", d50)):
        if not np.all(np.isfinite(array)):
            raise SedimentPhysicsError(f"static physics term {name} is non-finite (unsupported inputs; refused)")
    config = np.array([
        0 if params.ke_model == "wainwright_log" else 1,
        1 if params.ke_vegetation_form == "legacy_literal" else 0,
        1 if params.raindrop_composition_scaling == "fraction" else 0,
        1 if params.distance_convention == "legacy_literal" else 0,
        1 if params.bagnold_depth_units == "legacy_mm" else 0,
    ], dtype=np.int64)
    config.setflags(write=False)
    return LegacyPhysicsContext(
        shape=(ny, nx), n_classes=nc, cell_area_m2=area, ke_model=params.ke_model,
        particle_density_kg_m3=float(params.particle_density_kg_m3), reference_interval_s=ref,
        recession_factor_per_reference_s=float(params.recession_factor_per_reference_s),
        kinematic_viscosity_m2_s=float(params.kinematic_viscosity_m2_s),
        flow_detachment_depth_scale_m=float(params.flow_detachment_depth_scale_m),
        bagnold_density_scale=float(params.bagnold_density_scale),
        config=config, class_constants=_owned(cls), p_par=-2.0 / math.pi,
        active=_owned(active_in.reshape(n), np.bool_), slope=_owned(slope2.reshape(n)),
        veg_factor=_owned(veg_factor.reshape(n)), d50=_owned(d50.reshape(n)),
        bagnold_prefactor=_owned(bag_c1.reshape(n)), slope_power=_owned(slope_power.reshape(n * nc)),
        fractions=_owned(fractions.reshape(n * nc)), cap_rate=_owned(cap_rate.reshape(n * nc)),
        preparation_s=time.perf_counter() - t0,
    )


# --- step ---------------------------------------------------------------------------------------
def legacy_physics_step(context: LegacyPhysicsContext, depth_m: Any, velocity_m_s: Any, rain_rate_m_per_s: Any,
                        previous_sediment_velocity_m_s: Any, dt_s: float) -> SedimentPhysicsStep:
    """Compiled equivalent of `sediment_physics_step` for the frozen composition held by `context`.

    Same dynamic inputs and `SedimentPhysicsStep` fields as the reference (all newly allocated, owned by the
    caller). Raises `SedimentPhysicsError` on invalid input or any non-finite law value before returning."""
    if not isinstance(context, LegacyPhysicsContext):
        raise SedimentPhysicsError("context must be a LegacyPhysicsContext (see prepare_legacy_physics)")
    ny, nx = context.shape
    nc = context.n_classes
    n = ny * nx
    dt = _positive(dt_s, "dt_s")
    cell = {"depth_m": depth_m, "velocity_m_s": velocity_m_s, "rain_rate_m_per_s": rain_rate_m_per_s}
    cls_arrays = {"previous_sediment_velocity_m_s": previous_sediment_velocity_m_s}
    _require_grid_arrays(cell, (ny, nx), np)
    _require_grid_arrays(cls_arrays, (ny, nx, nc), np)

    scalars = np.empty(_N_SC, dtype=np.float64)
    scalars[:] = 0.0
    scalars[_S_RHO] = context.particle_density_kg_m3
    scalars[_S_REF] = context.reference_interval_s
    scalars[_S_MASS] = context.cell_area_m2 * context.particle_density_kg_m3 * dt
    scalars[_S_NU] = context.kinematic_viscosity_m2_s
    scalars[_S_HZ] = context.flow_detachment_depth_scale_m
    scalars[_S_BAG_SCALE] = context.bagnold_density_scale
    # exactly the reference's `recession_velocity(previous, dt, ...)` factor, evaluated in Python math
    scalars[_S_DECAY] = recession_velocity(1.0, dt, factor_per_reference_s=context.recession_factor_per_reference_s,
                                           reference_interval_s=context.reference_interval_s)
    scalars[_S_P_PAR] = context.p_par

    kernel = compiled_kernel()
    flat = np.ascontiguousarray
    shape3 = (ny, nx, nc)
    requested = np.empty(shape3)
    raindrop = np.empty(shape3)
    flow = np.empty(shape3)
    velocity = np.empty(shape3)
    rate = np.empty(shape3)
    law = np.empty(shape3, dtype=np.bool_)
    settle = np.empty(shape3, dtype=np.bool_)
    regime = np.empty(shape3, dtype=np.int8)
    cap_applied = np.empty(shape3, dtype=np.bool_)
    prob = np.empty(shape3)
    ustar = np.empty((ny, nx))
    reynolds = np.empty((ny, nx))
    power = np.empty((ny, nx))
    ke = np.empty((ny, nx))
    ke_flux = np.empty((ny, nx))
    d50 = np.empty((ny, nx))
    counts = np.zeros(len(REGIME_CODES), dtype=np.int64)
    flags = int(kernel(
        flat(depth_m).reshape(n), flat(velocity_m_s).reshape(n), flat(rain_rate_m_per_s).reshape(n),
        flat(previous_sediment_velocity_m_s).reshape(n * nc), context.active, context.slope, context.veg_factor,
        context.d50, context.bagnold_prefactor, context.slope_power, context.fractions, context.cap_rate,
        context.class_constants, scalars, context.config,
        requested.reshape(-1), raindrop.reshape(-1), flow.reshape(-1), velocity.reshape(-1), rate.reshape(-1),
        law.reshape(-1), settle.reshape(-1), regime.reshape(-1), ustar.reshape(-1), reynolds.reshape(-1),
        power.reshape(-1), ke.reshape(-1), ke_flux.reshape(-1), prob.reshape(-1), cap_applied.reshape(-1),
        d50.reshape(-1), counts))
    if flags:
        problems = [msg for bit, msg in enumerate(_FLAG_MESSAGES) if flags >> bit & 1]
        raise SedimentPhysicsError("; ".join(problems))
    return SedimentPhysicsStep(
        dt_s=dt, requested_pickup_kg=requested, raindrop_pickup_kg=raindrop, flow_pickup_kg=flow,
        sediment_velocity_m_s=velocity, deposition_rate_per_m=rate, law_applies=law, settle_mask=settle,
        regime=regime, d50_m=d50, shear_velocity_m_s=ustar, reynolds_number=reynolds, stream_power_w_m2=power,
        rain_energy_j_m2_mm=ke, rain_energy_flux_j_m2_s=ke_flux, pickup_probability=prob,
        legacy_cap_applied=cap_applied,
        regime_counts={name: np.int64(counts[code]) for name, code in REGIME_CODES.items()},
    )
