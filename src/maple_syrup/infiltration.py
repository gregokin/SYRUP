"""Minimal conservative infiltration / soil-water columns (Phase 3b).

Reference: MAHLERAN 1.2.3 `src/Subroutines_Water/infilt.for`, the
`inf_type < 5` branch (Smith & Parlange 1978 capacity, linear drainage,
saturation excess), with `inf_model = 2` (Hawkins-type final infiltration
from pavement and LOCAL rain) selected and `inf_model = 1` (fixed Ksat)
available for controlled tests. Initial and maximum soil water follow
`initialize_values_xml.f90` 228-229 and 360-362. See
docs/phase3/infiltration_spec.md and docs/phase3/infiltration.md.

Units: depths m, rates m/s, time s. Per cell, one step of length `dt`
wholly inside one constant-rain piece:

    A    = h + P                                  available surface water
    J    = min(A, capacity * dt)                  intake
    D    = min((S / Smax) * Ksat * c_drain * dt, S + J)   drainage (leaves)
    O    = max(S + J - D - Smax, 0)               saturation return
    S'   = S + J - D - O,   h' = A - J + O
    I    = J - O                                  net infiltration

so that h' + S' + D = h + S + P exactly up to FP64 roundoff. `S` is the
RETAINED soil water (legacy `cum_inf`, which includes antecedent water and
has drainage subtracted); it is not cumulative infiltration.

`column_step` is pure: it reads its inputs, returns new arrays and never
writes to a caller-owned array, so a rejected step leaves nothing changed.
All arrays stay in the parameters' own NumPy/CuPy namespace; validation
costs one batched flag read per validated step. No routing, sediment,
evapotranspiration, dry reset or splash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

__all__ = [
    "INFILTRATION_MODELS",
    "LOCAL_BALANCE_RTOL",
    "ColumnParameters",
    "ColumnStep",
    "InfiltrationError",
    "column_parameters",
    "column_step",
    "initial_soil_water_m",
    "pavement_lambda_m_per_s",
]

# Legacy `inf_model` 2 and 1.
INFILTRATION_MODELS = ("pavement_hawkins", "fixed_ksat")

# Legacy Hawkins-type constants (mm/s; infilt.for 62-67). The Fortran
# literals are default REAL (single precision); the decimal values are used
# here in FP64 (relative difference ~1e-8).
_LAMBDA_SLOPE_MM_S = 0.022891667
_LAMBDA_OFFSET_MM_S = 0.098575
_LAMBDA_BARE_MM_S = 0.16
# Legacy pave = percent * 1e-4 (storm_setting 375-379) = cover fraction * 1e-2.
_PAVE_PER_FRACTION = 1.0e-2

_EPS = float(np.finfo(np.float64).eps)
# Per-cell |h' + S' + D - (h + S + P)| <= LOCAL_BALANCE_RTOL * (h + P + S + J + D + O):
# a handful of roundings of quantities no larger than that scale.
LOCAL_BALANCE_RTOL = 16.0 * _EPS


class InfiltrationError(ValueError):
    """Invalid parameters, state or step request. Raised before any result
    is returned; no caller-owned array is modified."""


# --- parameters -------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ColumnParameters:
    """Validated per-cell `(ny, nx)` FP64 parameters in one namespace. Build
    with `column_parameters`; arrays are private copies (read-only on NumPy)."""

    model: str
    ksat_m_per_s: Any
    suction_m: Any
    drainage_parameter: Any  # dimensionless legacy drain_par
    theta_sat: Any
    soil_thickness_m: Any
    storage_max_m: Any  # theta_sat * soil_thickness (legacy stmax)
    lambda_m_per_s: Any  # model "pavement_hawkins" only, else None
    active_mask: Any  # bool
    xp: ModuleType

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.ksat_m_per_s.shape)


def pavement_lambda_m_per_s(pavement_cover_fraction: Any) -> Any:
    """Legacy lambda (infilt.for 62-67) for a pavement cover FRACTION in
    [0, 1], converted from mm/s to m/s: -0.022891667 ln(p) - 0.098575 for
    p = 0.01 * fraction > 0, else 0.16 mm/s."""
    from maple.core.backend import array_namespace, errstate

    xp = array_namespace(pavement_cover_fraction)
    p = pavement_cover_fraction * _PAVE_PER_FRACTION
    positive = p > 0.0
    with errstate(xp=xp, divide="ignore", invalid="ignore"):
        log_p = xp.log(xp.where(positive, p, 1.0))
    lambda_mm = xp.where(positive, -_LAMBDA_SLOPE_MM_S * log_p - _LAMBDA_OFFSET_MM_S, _LAMBDA_BARE_MM_S)
    return lambda_mm * 1.0e-3


def _require_arrays(named: dict[str, Any], shape: tuple[int, int] | None, xp: ModuleType | None):
    from maple.core.backend import MixedArrayNamespaceError, array_namespace, is_array

    for name, array in named.items():
        if not is_array(array):
            raise InfiltrationError(f"{name} must be a NumPy/CuPy array, got {type(array).__name__}")
    arrays = list(named.values())
    try:
        namespace = array_namespace(*arrays)
    except MixedArrayNamespaceError as exc:
        raise InfiltrationError(str(exc)) from None
    if xp is not None and namespace is not xp:
        raise InfiltrationError(f"arrays must be in namespace {xp.__name__!r}, got {namespace.__name__!r}")
    if shape is None:
        shape = tuple(arrays[0].shape)
        if len(shape) != 2 or min(shape) < 1:
            raise InfiltrationError(f"arrays must be non-empty (ny, nx), got shape {shape}")
    for name, array in named.items():
        if tuple(array.shape) != shape:
            raise InfiltrationError(f"{name} shape {tuple(array.shape)} != {shape}")
    return namespace, shape


def _resolve(checks) -> None:
    try:
        checks.resolve()
    except ValueError as exc:
        raise InfiltrationError(str(exc)) from None


def column_parameters(
    *,
    model: str,
    ksat_m_per_s: Any,
    suction_m: Any,
    drainage_parameter: Any,
    theta_sat: Any,
    soil_thickness_m: Any,
    pavement_cover_fraction: Any = None,
    active_mask: Any = None,
) -> ColumnParameters:
    """Validate and copy per-cell parameters (all float64 `(ny, nx)` of one
    namespace; `active_mask` bool, default all True). Ksat, suction and the
    drainage parameter must be finite and >= 0 -- a negative conductivity is
    rejected, never floored. 0 < theta_sat <= 1, soil thickness > 0.
    `pavement_cover_fraction` in [0, 1] is required for model
    "pavement_hawkins" and refused otherwise. One batched flag read."""
    from maple.core.backend import (
        DeferredChecks,
        finite_flag,
        freeze,
        negative_flag,
        true_flag,
    )

    if model not in INFILTRATION_MODELS:
        raise InfiltrationError(f"model must be one of {INFILTRATION_MODELS}, got {model!r}")
    floats = {
        "ksat_m_per_s": ksat_m_per_s,
        "suction_m": suction_m,
        "drainage_parameter": drainage_parameter,
        "theta_sat": theta_sat,
        "soil_thickness_m": soil_thickness_m,
    }
    if model == "pavement_hawkins":
        if pavement_cover_fraction is None:
            raise InfiltrationError("model 'pavement_hawkins' needs pavement_cover_fraction")
        floats["pavement_cover_fraction"] = pavement_cover_fraction
    elif pavement_cover_fraction is not None:
        raise InfiltrationError("pavement_cover_fraction is only used by model 'pavement_hawkins'")
    named = dict(floats) if active_mask is None else {**floats, "active_mask": active_mask}
    xp, shape = _require_arrays(named, None, None)
    for name, array in floats.items():
        if array.dtype != np.float64:
            raise InfiltrationError(f"{name} must be float64, got {array.dtype}")
    if active_mask is not None and active_mask.dtype != np.bool_:
        raise InfiltrationError(f"active_mask must be bool, got {active_mask.dtype}")

    checks = DeferredChecks()
    for name, array in floats.items():
        checks.require(finite_flag(array), f"{name} must be finite everywhere")
    for name in ("ksat_m_per_s", "suction_m", "drainage_parameter"):
        checks.forbid(negative_flag(floats[name]), f"{name} must be >= 0 everywhere (never floored)")
    checks.forbid(true_flag(theta_sat <= 0.0), "theta_sat must be > 0 everywhere")
    checks.forbid(true_flag(theta_sat > 1.0), "theta_sat must be <= 1 everywhere")
    checks.forbid(true_flag(soil_thickness_m <= 0.0), "soil_thickness_m must be > 0 everywhere")
    lam = None
    if model == "pavement_hawkins":
        cover = pavement_cover_fraction
        checks.forbid(true_flag((cover < 0.0) | (cover > 1.0)), "pavement_cover_fraction must lie in [0, 1]")
        lam = pavement_lambda_m_per_s(xp.where((cover >= 0.0) & (cover <= 1.0), cover, 0.0))
        checks.forbid(true_flag(~(lam > 0.0)), "legacy lambda must be > 0 for the supplied pavement cover")
    _resolve(checks)

    def own(array):
        return freeze(xp.array(array, copy=True))

    mask = xp.ones(shape, dtype=np.bool_) if active_mask is None else active_mask
    return ColumnParameters(
        model=model,
        ksat_m_per_s=own(ksat_m_per_s),
        suction_m=own(suction_m),
        drainage_parameter=own(drainage_parameter),
        theta_sat=own(theta_sat),
        soil_thickness_m=own(soil_thickness_m),
        storage_max_m=freeze(theta_sat * soil_thickness_m),
        lambda_m_per_s=None if lam is None else freeze(lam),
        active_mask=own(mask),
        xp=xp,
    )


def initial_soil_water_m(parameters: ColumnParameters, initial_theta: Any) -> Any:
    """Legacy `ciinit = theta_0 * soil_thick`: retained soil water (m) for a
    per-cell initial volumetric moisture in [0, theta_sat]. New array."""
    from maple.core.backend import DeferredChecks, finite_flag, negative_flag, true_flag

    _require_arrays({"initial_theta": initial_theta}, parameters.shape, parameters.xp)
    if initial_theta.dtype != np.float64:
        raise InfiltrationError(f"initial_theta must be float64, got {initial_theta.dtype}")
    checks = DeferredChecks()
    checks.require(finite_flag(initial_theta), "initial_theta must be finite everywhere")
    checks.forbid(negative_flag(initial_theta), "initial_theta must be >= 0 everywhere")
    checks.forbid(true_flag(initial_theta > parameters.theta_sat), "initial_theta must be <= theta_sat everywhere")
    _resolve(checks)
    # theta0 <= theta_sat and rounding is monotone, so theta0 * L <= Smax.
    return initial_theta * parameters.soil_thickness_m


# --- step ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ColumnStep:
    """Result of one step: new stores and the step's per-cell fluxes (m).

    `intake_m` (J) is water entering the soil from the surface;
    `saturation_return_m` (O) is water pushed back to the surface when the
    column is full; `drainage_m` (D) leaves the column (not tracked further);
    `net_infiltration_m` = J - O. `rain_m` = rate * dt on active cells.
    """

    dt_s: float
    depth_m: Any
    soil_water_m: Any
    rain_m: Any
    intake_m: Any
    saturation_return_m: Any
    drainage_m: Any

    @property
    def net_infiltration_m(self) -> Any:
        return self.intake_m - self.saturation_return_m


def _check_dt(dt_s: Any) -> float:
    if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float, np.integer, np.floating)):
        raise InfiltrationError(f"dt_s must be a real number, got {type(dt_s).__name__}")
    dt = float(dt_s)
    if not math.isfinite(dt) or dt < 0.0:
        raise InfiltrationError(f"dt_s must be finite and >= 0, got {dt!r}")
    return dt


def _final_infiltration(parameters: ColumnParameters, rain_rate: Any) -> Any:
    """K (m/s). Model 2: lambda * (1 - exp(-r / lambda)) for LOCAL r > 0,
    Ksat for r == 0 (legacy tests r2(i, 2); corrected to the cell's own r).
    Model 1: Ksat (legacy ksat_mod calibration is not applied here)."""
    if parameters.model == "fixed_ksat":
        return parameters.ksat_m_per_s
    xp, lam = parameters.xp, parameters.lambda_m_per_s
    return xp.where(rain_rate > 0.0, -lam * xp.expm1(-rain_rate / lam), parameters.ksat_m_per_s)


def _intake(parameters: ColumnParameters, depth: Any, soil: Any, rain_rate: Any, available: Any, dt: float):
    """J = min(A, capacity * dt), capacity = K / (1 - exp(-x)),
    x = S / ((psi + h)(theta_sat - theta)), with the limits: K = 0 -> 0;
    vanishing (psi + h) * deficit -> K (no capillary term); 1 - exp(-x) == 0
    (S = 0 or underflow) with positive K -> unbounded, i.e. J = A."""
    from maple.core.backend import errstate

    xp = parameters.xp
    k = _final_infiltration(parameters, rain_rate)
    deficit = (parameters.storage_max_m - soil) / parameters.soil_thickness_m  # theta_sat - theta, >= 0
    scale = (parameters.suction_m + depth) * deficit
    capillary = scale > 0.0
    with errstate(xp=xp, divide="ignore", over="ignore", invalid="ignore"):
        x = soil / xp.where(capillary, scale, 1.0)
        denominator = -xp.expm1(-x)  # in [0, 1]
        finite = denominator > 0.0
        capacity = xp.where(capillary, k / xp.where(finite, denominator, 1.0), k)
        unbounded = capillary & ~finite & (k > 0.0)
        # capacity * dt may overflow to +inf for a near-empty column; min() then selects A.
        intake = xp.where(unbounded, available, xp.minimum(available, capacity * dt))
    return intake


def column_step(
    parameters: ColumnParameters,
    depth_m: Any,
    soil_water_m: Any,
    rain_rate_m_per_s: Any,
    dt_s: float,
    *,
    validate: bool = True,
) -> ColumnStep:
    """Advance every column by `dt_s` under a per-cell rain RATE that is
    constant over the step (split steps at every rainfall knot). Returns new
    arrays; inputs are never modified.

    Structural checks (namespace, shape, dtype, dt) always run. With
    `validate` (default) one batched flag read also checks the inputs
    (finite, >= 0, S <= Smax, no rain on inactive cells) and the outputs
    (finite, >= 0, S' <= Smax, local balance within LOCAL_BALANCE_RTOL);
    any failure raises `InfiltrationError` and returns nothing. Inactive
    cells keep h and S exactly and exchange nothing. `dt_s = 0` is the
    identity."""
    from maple.core.backend import (
        DeferredChecks,
        errstate,
        finite_flag,
        negative_flag,
        true_flag,
    )

    if not isinstance(parameters, ColumnParameters):
        raise InfiltrationError("parameters must be a ColumnParameters (use column_parameters)")
    dt = _check_dt(dt_s)
    named = {"depth_m": depth_m, "soil_water_m": soil_water_m, "rain_rate_m_per_s": rain_rate_m_per_s}
    xp, _ = _require_arrays(named, parameters.shape, parameters.xp)
    for name, array in named.items():
        if array.dtype != np.float64:
            raise InfiltrationError(f"{name} must be float64, got {array.dtype}")
    active, smax = parameters.active_mask, parameters.storage_max_m

    checks = DeferredChecks()
    if validate:
        for name, array in named.items():
            checks.require(finite_flag(array), f"{name} must be finite everywhere")
            checks.forbid(negative_flag(array), f"{name} must be >= 0 everywhere")
        checks.forbid(true_flag(soil_water_m > smax), "soil_water_m exceeds theta_sat * soil_thickness")
        checks.forbid(true_flag((rain_rate_m_per_s > 0.0) & ~active), "rain on an inactive (masked) cell")

    if dt == 0.0:
        _resolve(checks)
        zero = xp.zeros(parameters.shape, dtype=np.float64)
        return ColumnStep(0.0, xp.array(depth_m, copy=True), xp.array(soil_water_m, copy=True),
                          zero, zero.copy(), zero.copy(), zero.copy())

    h, s = depth_m, soil_water_m
    # Valid inputs raise no floating-point exception except the overflow
    # handled in _intake; invalid ones are reported by the checks instead.
    with errstate(xp=xp, all="ignore"):
        rain = rain_rate_m_per_s * dt
        available = h + rain
        intake = xp.where(active, _intake(parameters, h, s, rain_rate_m_per_s, available, dt), 0.0)
        wetted = s + intake
        demand = (s / smax) * parameters.ksat_m_per_s * parameters.drainage_parameter * dt
        drainage = xp.where(active, xp.minimum(demand, wetted), 0.0)
        retained = wetted - drainage  # >= 0: drainage <= wetted
        overflow = xp.maximum(retained - smax, 0.0)
        soil_new = xp.minimum(retained, smax)
        depth_new = (available - intake) + overflow  # available - intake >= 0: intake <= available

    if validate:
        for name, array in (("depth", depth_new), ("soil water", soil_new), ("intake", intake),
                            ("drainage", drainage), ("saturation return", overflow)):
            checks.require(finite_flag(array), f"step produced non-finite {name}")
            checks.forbid(negative_flag(array), f"step produced negative {name}")
        with errstate(xp=xp, all="ignore"):
            residual = (depth_new + soil_new + drainage) - (h + s + rain)
            scale = h + rain + s + intake + drainage + overflow
        checks.forbid(true_flag(xp.abs(residual) > LOCAL_BALANCE_RTOL * scale),
                      "local water balance h' + S' + D = h + S + P violated beyond FP64 tolerance")
    _resolve(checks)
    return ColumnStep(dt, depth_new, soil_new, rain, intake, overflow, drainage)
