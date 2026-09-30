"""MAHLERAN wet detachment and transport laws, vectorized (Phase 5a).

Reference: MAHLERAN 1.2.3 (read-only), line numbers as read 2026-09-30:

- `Subroutines_Sediment/route_sediment_xml.f90` 44-187: per-cell regime
  selection (d50 66-85, u* 89, Re 95, Re >= 2500 concentrated/suspended
  98-120, rain and Re < 2500 diffuse with transitional flow detachment
  123-144, no rain and Re > 500 concentrated 148-160, otherwise nothing
  161-170, dry + rain splash 180-182 -- DEFERRED here).
- `raindrop_detachment.for` 24-54 (rain kinetic energy, two models,
  vegetation), 69-98 (Quansah law, /density, /dt, depth attenuation,
  clip < 0), 103-112 (phi 2 cap, zero if absent), 117-120 (gravel
  feedback, `grav_propn = 0` since JWAug2005: route_sediment 59).
- `flow_detachment.for` 18-47 (Shields-type pickup probability, `hz`
  scale, cap, zero if absent).
- `diffuse_flow_transport.for` 26-70 (rain energy flux, stream power,
  particle mass, virtual velocity, mean travel distance, speed cap).
- `conc_flow_transport.for` 26-81 (Bagnold threshold, excess stream power,
  Hassan distance x 0.693, 30 m cap, virtual velocity, speed cap).
- `suspended_transport.for` 22-56 (distance, exp cap 100, x 0.693,
  density scale, sediment velocity = water velocity).
- `initialize_values_xml.f90` 133-137 (dt, density, excess density,
  Bagnold scale, hz), 211-219 (sigma, p_par, dstar_const), 268 (spa /
  1200), 294-302 (diameter = 2 radius, settling velocity).
- `shared_data.f90` 176-180 (spq, radius, viscosity).
- `update_sediment_flow.for` 69 (`v_soil = 0.9 v_soil` every step).
- Root `mahleran_input.xml` (Plot 1): a/b/c/hs per class 106-145, particle
  density 2.65 (105, 149), `active_layer_sensitivity` 1.52e-6 (150),
  `KE_model_type` 2 (208), `time_step` 1.0 (6).

Units here are SI (kg, m, s) throughout; every legacy mm / g cm^-3 / cm
conversion is applied once, at the parameter or input boundary, and named.
Rates from the legacy per-step form are converted to physical rates with
the declared legacy reference interval (`reference_interval_s`, 1 s): the
legacy `detach_soil = X / dt` followed by `(detach - depos) * dt` in the
routing update picks up the same depth per STEP whatever dt is; here the
same depth is picked up per REFERENCE SECOND, so a refined timestep does
not change the imposed rate. Pickup DEMAND is returned in kg per cell and
class over `dt`; capping by holdings and availability is MAPLE's
(`apply_water_process_demand`), never done here.

Departures from the literal routines, each with an explicit switch or a
documented boundary (docs/phase5/physics.md):

- Rain kinetic energy at low intensity: `11.9 + 8.73 log10(I)` is negative
  below 0.0433 mm/h and the legacy raises a negative base to a real power
  (NaN). Here the energy is floored at zero: no rain energy, no rain
  detachment or diffuse transport. `I = 0` never reaches the law (dry or
  rain-free branches).
- KE model 2 with vegetation: the literal Fortran precedence (33-34)
  multiplies only the subtracted exponential term by `(1 - 8.1e-3 veg%)`,
  so cover INCREASES the energy (counter-intuitive, but it is the source
  relationship). `ke_vegetation_form="legacy_literal"` (default)
  reproduces that expression exactly; `"intended"` applies the factor to
  the whole energy as model 1 does and is an explicit scientific
  VARIATION, not a verified correction of the empirical relationship.
- Median/mean: `conc_flow_transport` and `suspended_transport` multiply a
  MEAN travel distance by 0.693 ("median") and `flow_distrib` then uses
  the value as the exponential MEAN. `distance_convention="legacy_literal"`
  (default) keeps that; `"formula_mean"` uses the formula value as the
  mean. `exponential_mean_from_median`/`median_from_exponential_mean` give
  the exact ln 2 conversion.
- Suspension criterion: `dstar_const` uses `(sigma - 1)` with sigma already
  the relative EXCESS density (1.65), i.e. 0.65 where van Rijn's D* has
  1.65. `dstar_convention="legacy_sigma_minus_one"` (default) keeps the
  literal; `"van_rijn"` uses sigma.
- Bagnold threshold: the literal `log10(12 d / d50)` mixes d in mm with d50
  in m. `bagnold_depth_units="legacy_mm"` (default) keeps that;
  `"si_m"` uses metres for both.
- Pickup probability at theta = 0.196 (p_const = 0): the literal
  `p_const / abs(p_const)` is 0/0; the continuous limit 0.5 is used.
  theta = 0 gives probability 0 without evaluating log(inf).
- Composition: d50 and the class fractions come from the CURRENT
  active-layer holdings (`active_layer_mass_kg`), not a static map. The
  raindrop law is not scaled by the class fraction in the legacy
  (`raindrop_composition_scaling="legacy_none"`, default; `"fraction"`
  multiplies by the fraction, an explicit option, not a calibration).
- Recession: `v_soil *= 0.9` per legacy 1 s step becomes
  `v *= exp(-dt / tau)` with `tau = -1 s / ln 0.9`, so the memory decays
  identically over any partition of the same interval.
- Depth / velocity time levels: the legacy evaluates `d(1)` (old) with the
  new `v`; the caller passes one consistent depth and velocity (documented
  one-step difference, vanishing with dt).
- `flow_distrib` walk limits (10 / 100 / 500 m) and the ring deposition are
  not reproduced: the transport operator (sediment_transport.py) is
  conservative and applies the same exponential travel-distance law as a
  deposition hazard `v / L` along the actual path, with the travel
  distance taken from LOCAL hydraulics at every cell (no source-assigned
  distance memory).
- Recession memory is NOT capped at the current water velocity (legacy:
  `v_soil` decays from its last law value whatever the water does), so
  the sediment velocity can exceed a slowing water's velocity; the
  transport operator validates its own Courant number.

Overflow: the holdings total, every intermediate law value and every
returned diagnostic must be finite; an input that overflows FP64 anywhere
(for example a rain rate or holdings of order 1e308) is refused rather
than masked into zero fractions or an infinite diagnostic.

Nothing here writes to a caller-owned array; a failed validation raises
`SedimentPhysicsError` before any result exists. All arrays stay in the
parameters' NumPy/CuPy namespace; there is no Python loop over cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

__all__ = [
    "BAGNOLD_DEPTH_UNITS",
    "DISTANCE_CONVENTIONS",
    "DSTAR_CONVENTIONS",
    "GRAVITY_M_S2",
    "KE_MODELS",
    "KE_VEGETATION_FORMS",
    "LEGACY_CLASS_RADII_M",
    "LEGACY_KINEMATIC_VISCOSITY_M2_S",
    "LEGACY_MEDIAN_FACTOR",
    "LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM",
    "LEGACY_RECESSION_FACTOR_PER_REFERENCE_S",
    "LEGACY_REFERENCE_INTERVAL_S",
    "PLOT1_ACTIVE_LAYER_SENSITIVITY_MM",
    "PLOT1_KE_MODEL",
    "PLOT1_PARTICLE_DENSITY_G_CM3",
    "PLOT1_RAINDROP_A",
    "PLOT1_RAINDROP_B",
    "PLOT1_RAINDROP_C",
    "PLOT1_RAINDROP_MAX_MM",
    "RAINDROP_COMPOSITION_SCALINGS",
    "REGIME_CODES",
    "REYNOLDS_CONCENTRATED",
    "REYNOLDS_TRANSITIONAL",
    "PhysicsGrid",
    "SedimentPhysicsError",
    "SedimentPhysicsParameters",
    "SedimentPhysicsStep",
    "exponential_mean_from_median",
    "median_diameter_m",
    "median_from_exponential_mean",
    "physics_grid",
    "physics_grid_from_graph",
    "plot1_sediment_parameters",
    "recession_velocity",
    "sediment_physics_parameters",
    "sediment_physics_step",
]

# route_sediment_xml 89, 95: 9.81e-3 converts g and mm -> m.
GRAVITY_M_S2 = 9.81
# shared_data.f90 179-180.
LEGACY_CLASS_RADII_M = (3.125e-5, 7.1825e-5, 1.875e-4, 6.25e-4, 3.5e-3, 1.2e-2)
LEGACY_KINEMATIC_VISCOSITY_M2_S = 1.003e-6
# shared_data.f90 178: exp(-spq d_cm) attenuation of raindrop detachment
# under a water layer (raindrop_detachment 87-95, Parsons et al. 2004).
LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM = (2.72, 1.61, 0.92, 0.85, 0.75, 0.30)
# conc_flow_transport 63, suspended_transport 44: literal 0.693, not ln 2.
LEGACY_MEDIAN_FACTOR = 0.693
# update_sediment_flow 69, applied once per legacy step (dt = 1 s in Plot 1).
LEGACY_RECESSION_FACTOR_PER_REFERENCE_S = 0.9
LEGACY_REFERENCE_INTERVAL_S = 1.0
# route_sediment_xml 98, 132, 156.
REYNOLDS_CONCENTRATED = 2500.0
REYNOLDS_TRANSITIONAL = 500.0
# conc_flow_transport 64-68; suspended_transport 32-34.
CONCENTRATED_DISTANCE_CAP_M = 30.0
SUSPENDED_EXPONENT_CAP = 100.0
# raindrop_detachment 24-34: KE (J m^-2 mm^-1) times (1 - 8.1e-3 veg%).
VEGETATION_ENERGY_COEFFICIENT_PER_PERCENT = 8.1e-3
# initialize_values_xml 268 and raindrop_detachment 69-72: the Quansah
# energy scaling `spa / 1200` and `(ke r2 1200)^spb` with r2 in mm/s, i.e.
# `ke * I_mm_h / 3`.
_LEGACY_QUANSAH_SCALE = 1.2e3

# Root mahleran_input.xml (Plot 1), per class phi_1..phi_6.
PLOT1_RAINDROP_A = (4.25e-5, 8.07e-4, 5.1e-4, 8.07e-4, 8.07e-5, 8.49e-6)
PLOT1_RAINDROP_B = (1.2, 1.08, 0.79, 0.75, 0.75, 0.5)
PLOT1_RAINDROP_C = (0.23, 0.21, 0.11, 1.06, 0.1, 0.1)
PLOT1_RAINDROP_MAX_MM = (1000.0,) * 6
PLOT1_PARTICLE_DENSITY_G_CM3 = 2.65
PLOT1_ACTIVE_LAYER_SENSITIVITY_MM = 1.52e-6
PLOT1_KE_MODEL = "verstraeten_exp"

KE_MODELS = ("wainwright_log", "verstraeten_exp")  # legacy KE_model_type 1, 2
KE_VEGETATION_FORMS = ("legacy_literal", "intended")
RAINDROP_COMPOSITION_SCALINGS = ("legacy_none", "fraction")
DISTANCE_CONVENTIONS = ("legacy_literal", "formula_mean")
DSTAR_CONVENTIONS = ("legacy_sigma_minus_one", "van_rijn")
BAGNOLD_DEPTH_UNITS = ("legacy_mm", "si_m")

# Per cell and class, int8. Regimes 0-4 are per cell; 5 and 6 per class.
REGIME_CODES = {
    "dry": 0,  # no surface water; rain splash deferred
    "wet_no_law": 1,  # wet, no rain, Re <= 500: nothing (route_sediment 161-170)
    "diffuse": 2,  # rain, Re <= 500: raindrop detachment, diffuse transport
    "transitional_rain": 3,  # rain, 500 < Re < 2500: raindrop + flow detachment, diffuse
    "transitional_dry": 4,  # no rain, 500 < Re < 2500: flow detachment, concentrated
    "concentrated": 5,  # Re >= 2500, class not suspended
    "suspended": 6,  # Re >= 2500, class suspended
}

_EPS = float(np.finfo(np.float64).eps)
_LN2 = math.log(2.0)


class SedimentPhysicsError(ValueError):
    """Invalid parameters, grid or step inputs. Raised before any result is
    returned; no caller-owned array is modified."""


# --- helpers ------------------------------------------------------------------------
def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SedimentPhysicsError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _positive(value: Any, name: str) -> float:
    v = _real(value, name)
    if not (math.isfinite(v) and v > 0.0):
        raise SedimentPhysicsError(f"{name} must be finite and > 0, got {value!r}")
    return v


def _choice(value: Any, options: tuple[str, ...], name: str) -> str:
    if value not in options:
        raise SedimentPhysicsError(f"{name} must be one of {options}, got {value!r}")
    return value


def _class_vector(values: Any, n: int, name: str, *, positive: bool = False, nonnegative: bool = False) -> np.ndarray:
    array = np.array(values, dtype=np.float64)
    if array.shape != (n,):
        raise SedimentPhysicsError(f"{name} must have shape ({n},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise SedimentPhysicsError(f"{name} must be finite")
    if positive and np.any(array <= 0.0):
        raise SedimentPhysicsError(f"{name} must be > 0 for every class")
    if nonnegative and np.any(array < 0.0):
        raise SedimentPhysicsError(f"{name} must be >= 0 for every class")
    return array


def exponential_mean_from_median(median: Any) -> Any:
    """Mean of an exponential step-length distribution whose median is
    `median`: `median / ln 2`. Exact, unlike the legacy 0.693 factor."""
    return median / _LN2


def median_from_exponential_mean(mean: Any) -> Any:
    """Median of an exponential distribution with mean `mean`: `mean ln 2`."""
    return mean * _LN2


def recession_velocity(previous_velocity_m_s: Any, dt_s: float, *,
                       factor_per_reference_s: float = LEGACY_RECESSION_FACTOR_PER_REFERENCE_S,
                       reference_interval_s: float = LEGACY_REFERENCE_INTERVAL_S) -> Any:
    """`update_sediment_flow.for` 69 as a continuous decay: the legacy
    multiplies `v_soil` by 0.9 once per step; here `v exp(-dt / tau)` with
    `tau = -reference_interval / ln(factor)`, so `dt = reference_interval`
    gives exactly the legacy factor and any partition of an interval gives
    the same decay (timestep-invariant)."""
    dt = _positive(dt_s, "dt_s")
    f = _real(factor_per_reference_s, "factor_per_reference_s")
    ref = _positive(reference_interval_s, "reference_interval_s")
    if not (0.0 < f < 1.0):
        raise SedimentPhysicsError(f"factor_per_reference_s must lie in (0, 1), got {f!r}")
    return previous_velocity_m_s * math.exp(math.log(f) * dt / ref)


# --- parameters ---------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SedimentPhysicsParameters:
    """Validated per-class and scalar parameters (host copies) plus
    `(1, 1, n_classes)` device copies in namespace `xp` for broadcasting.
    Build with `sediment_physics_parameters`."""

    n_classes: int
    diameter_m: np.ndarray
    raindrop_a: np.ndarray  # legacy spa as authored (before / 1200)
    raindrop_b: np.ndarray
    raindrop_c: np.ndarray
    raindrop_depth_attenuation_per_cm: np.ndarray  # legacy spq
    raindrop_max_depth_per_reference_s_m: np.ndarray  # legacy hs (mm -> m)
    particle_density_kg_m3: float
    flow_detachment_depth_scale_m: float  # legacy hz (mm -> m)
    kinematic_viscosity_m2_s: float
    reference_interval_s: float
    recession_factor_per_reference_s: float
    ke_model: str
    ke_vegetation_form: str
    raindrop_composition_scaling: str
    distance_convention: str
    dstar_convention: str
    bagnold_depth_units: str
    # Derived (initialize_values_xml 134-136, 211-219, 294-302).
    sigma: float  # relative excess density (density_g_cm3 - 1)
    excess_density_kg_m3: float
    bagnold_density_scale: float
    dstar_const_per_m: float
    settling_velocity_m_s: np.ndarray
    particle_mass_g: np.ndarray
    xp: ModuleType
    # Device (1, 1, n_classes) copies.
    d_diameter: Any
    d_raindrop_a: Any
    d_raindrop_b: Any
    d_raindrop_c: Any
    d_attenuation: Any
    d_raindrop_max: Any
    d_settling: Any
    d_particle_mass_g: Any

    def summary(self) -> dict[str, Any]:
        return {
            "n_classes": self.n_classes,
            "diameter_m": self.diameter_m.tolist(),
            "raindrop_a": self.raindrop_a.tolist(),
            "raindrop_b": self.raindrop_b.tolist(),
            "raindrop_c": self.raindrop_c.tolist(),
            "raindrop_depth_attenuation_per_cm": self.raindrop_depth_attenuation_per_cm.tolist(),
            "raindrop_max_depth_per_reference_s_m": self.raindrop_max_depth_per_reference_s_m.tolist(),
            "particle_density_kg_m3": self.particle_density_kg_m3,
            "flow_detachment_depth_scale_m": self.flow_detachment_depth_scale_m,
            "kinematic_viscosity_m2_s": self.kinematic_viscosity_m2_s,
            "reference_interval_s": self.reference_interval_s,
            "recession_factor_per_reference_s": self.recession_factor_per_reference_s,
            "recession_timescale_s": -self.reference_interval_s / math.log(self.recession_factor_per_reference_s),
            "ke_model": self.ke_model,
            "ke_vegetation_form": self.ke_vegetation_form,
            "raindrop_composition_scaling": self.raindrop_composition_scaling,
            "distance_convention": self.distance_convention,
            "dstar_convention": self.dstar_convention,
            "bagnold_depth_units": self.bagnold_depth_units,
            "sigma": self.sigma,
            "excess_density_kg_m3": self.excess_density_kg_m3,
            "bagnold_density_scale": self.bagnold_density_scale,
            "dstar_const_per_m": self.dstar_const_per_m,
            "settling_velocity_m_s": self.settling_velocity_m_s.tolist(),
            "particle_mass_g": self.particle_mass_g.tolist(),
            "namespace": self.xp.__name__,
        }


def sediment_physics_parameters(
    *,
    diameter_m: Any,
    raindrop_a: Any,
    raindrop_b: Any,
    raindrop_c: Any,
    raindrop_max_depth_mm: Any,
    particle_density_g_cm3: float,
    active_layer_sensitivity_mm: float,
    ke_model: str,
    raindrop_depth_attenuation_per_cm: Any = LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM,
    kinematic_viscosity_m2_s: float = LEGACY_KINEMATIC_VISCOSITY_M2_S,
    reference_interval_s: float = LEGACY_REFERENCE_INTERVAL_S,
    recession_factor_per_reference_s: float = LEGACY_RECESSION_FACTOR_PER_REFERENCE_S,
    ke_vegetation_form: str = "legacy_literal",
    raindrop_composition_scaling: str = "legacy_none",
    distance_convention: str = "legacy_literal",
    dstar_convention: str = "legacy_sigma_minus_one",
    bagnold_depth_units: str = "legacy_mm",
    xp: ModuleType | None = None,
) -> SedimentPhysicsParameters:
    """Validate and derive the legacy parameter set. Inputs are in the
    legacy authoring units named in the argument (mm, g cm^-3); everything
    stored is SI except `raindrop_a` (kept as authored: it is applied with
    the legacy 1/1200 scaling inside the law) and `particle_mass_g` (the
    Parsons et al. relationships take grams)."""
    from maple.core.backend import freeze, to_device

    d = np.array(diameter_m, dtype=np.float64)
    if d.ndim != 1 or d.size < 1:
        raise SedimentPhysicsError(f"diameter_m must be a 1-D sequence with at least one class, got shape {d.shape}")
    n = int(d.size)
    d = _class_vector(d, n, "diameter_m", positive=True)
    if np.any(np.diff(d) <= 0.0):
        raise SedimentPhysicsError("diameter_m must increase strictly with the class index "
                                   "(the legacy d50 interpolation and MAPLE's MAHLERAN order assume it)")
    a = _class_vector(raindrop_a, n, "raindrop_a", nonnegative=True)
    b = _class_vector(raindrop_b, n, "raindrop_b", positive=True)
    c = _class_vector(raindrop_c, n, "raindrop_c", nonnegative=True)
    hs_mm = _class_vector(raindrop_max_depth_mm, n, "raindrop_max_depth_mm", nonnegative=True)
    spq = _class_vector(raindrop_depth_attenuation_per_cm, n, "raindrop_depth_attenuation_per_cm",
                        nonnegative=True)
    rho_g = _positive(particle_density_g_cm3, "particle_density_g_cm3")
    if rho_g <= 1.0:
        raise SedimentPhysicsError("particle_density_g_cm3 must exceed 1 (water) for the excess-density terms")
    hz_mm = _real(active_layer_sensitivity_mm, "active_layer_sensitivity_mm")
    if not (math.isfinite(hz_mm) and hz_mm >= 0.0):
        raise SedimentPhysicsError(f"active_layer_sensitivity_mm must be finite and >= 0, got {hz_mm!r}")
    nu = _positive(kinematic_viscosity_m2_s, "kinematic_viscosity_m2_s")
    ref = _positive(reference_interval_s, "reference_interval_s")
    rec = _real(recession_factor_per_reference_s, "recession_factor_per_reference_s")
    if not (0.0 < rec < 1.0):
        raise SedimentPhysicsError(f"recession_factor_per_reference_s must lie in (0, 1), got {rec!r}")
    ke_model = _choice(ke_model, KE_MODELS, "ke_model")
    veg_form = _choice(ke_vegetation_form, KE_VEGETATION_FORMS, "ke_vegetation_form")
    comp = _choice(raindrop_composition_scaling, RAINDROP_COMPOSITION_SCALINGS, "raindrop_composition_scaling")
    dist = _choice(distance_convention, DISTANCE_CONVENTIONS, "distance_convention")
    dstar = _choice(dstar_convention, DSTAR_CONVENTIONS, "dstar_convention")
    bag = _choice(bagnold_depth_units, BAGNOLD_DEPTH_UNITS, "bagnold_depth_units")

    # initialize_values_xml 134-136, 211, 219.
    sigma = (1.0e3 * rho_g - 1.0e3) / 1.0e3
    excess_density = 1000.0 * (rho_g - 1.0)
    bagnold_scale = (excess_density / 1650.0) ** (-0.5)
    sigma_for_dstar = (sigma - 1.0) if dstar == "legacy_sigma_minus_one" else sigma
    if sigma_for_dstar <= 0.0:
        raise SedimentPhysicsError(
            f"the suspension criterion needs a positive (sigma - 1) = {sigma_for_dstar} under "
            f"{dstar!r}; choose dstar_convention='van_rijn' or a denser particle"
        )
    dstar_const = ((sigma_for_dstar * GRAVITY_M_S2) / (nu ** 2)) ** (1.0 / 3.0)
    # 298-302: Stokes below 0.1 mm, else 1.1 sqrt(sigma g D).
    settling = np.where(d < 1.0e-4, sigma * GRAVITY_M_S2 * d ** 2 / (18.0 * nu),
                        1.1 * np.sqrt(sigma * GRAVITY_M_S2 * d))
    # diffuse_flow_transport 40-41: density [g cm^-3] * 1e6 * 4/3 pi r^3 [m^3] = g.
    particle_mass_g = rho_g * 1.0e6 * (4.0 / 3.0) * math.pi * (0.5 * d) ** 3

    namespace = np if xp is None else xp

    def device(v):
        return freeze(to_device(np.ascontiguousarray(v.reshape(1, 1, n)), namespace))

    def host(v):
        return freeze(np.ascontiguousarray(v))

    return SedimentPhysicsParameters(
        n_classes=n,
        diameter_m=host(d), raindrop_a=host(a), raindrop_b=host(b), raindrop_c=host(c),
        raindrop_depth_attenuation_per_cm=host(spq),
        raindrop_max_depth_per_reference_s_m=host(hs_mm * 1.0e-3),
        particle_density_kg_m3=rho_g * 1000.0,
        flow_detachment_depth_scale_m=hz_mm * 1.0e-3,
        kinematic_viscosity_m2_s=nu, reference_interval_s=ref, recession_factor_per_reference_s=rec,
        ke_model=ke_model, ke_vegetation_form=veg_form, raindrop_composition_scaling=comp,
        distance_convention=dist, dstar_convention=dstar, bagnold_depth_units=bag,
        sigma=sigma, excess_density_kg_m3=excess_density, bagnold_density_scale=bagnold_scale,
        dstar_const_per_m=dstar_const, settling_velocity_m_s=host(settling), particle_mass_g=host(particle_mass_g),
        xp=namespace,
        d_diameter=device(d), d_raindrop_a=device(a), d_raindrop_b=device(b), d_raindrop_c=device(c),
        d_attenuation=device(spq), d_raindrop_max=device(hs_mm * 1.0e-3), d_settling=device(settling),
        d_particle_mass_g=device(particle_mass_g),
    )


def plot1_sediment_parameters(*, xp: ModuleType | None = None, **overrides: Any) -> SedimentPhysicsParameters:
    """The root `mahleran_input.xml` Plot 1 sediment parameters with the
    `shared_data.f90` class radii (diameter = 2 radius). Keyword overrides
    select the documented conventions. The next task should cross-check
    these constants against the import report's XML record."""
    kwargs: dict[str, Any] = {
        "diameter_m": tuple(2.0 * r for r in LEGACY_CLASS_RADII_M),
        "raindrop_a": PLOT1_RAINDROP_A, "raindrop_b": PLOT1_RAINDROP_B, "raindrop_c": PLOT1_RAINDROP_C,
        "raindrop_max_depth_mm": PLOT1_RAINDROP_MAX_MM,
        "particle_density_g_cm3": PLOT1_PARTICLE_DENSITY_G_CM3,
        "active_layer_sensitivity_mm": PLOT1_ACTIVE_LAYER_SENSITIVITY_MM,
        "ke_model": PLOT1_KE_MODEL,
    }
    kwargs.update(overrides)
    return sediment_physics_parameters(xp=xp, **kwargs)


# --- grid -------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PhysicsGrid:
    """Static per-cell terrain the laws need, placed once in namespace
    `xp`: `slope` (dimensionless, legacy post-edge-rule) and `active`
    (bool), both `(ny, nx)`; `cell_area_m2`."""

    shape: tuple[int, int]
    cell_area_m2: float
    slope: Any
    active: Any
    xp: ModuleType


def physics_grid(slope: np.ndarray, active: np.ndarray, cell_area_m2: float, *,
                 xp: ModuleType | None = None) -> PhysicsGrid:
    from maple.core.backend import freeze, to_device

    s = np.array(slope, dtype=np.float64)
    a = np.array(active, dtype=np.bool_)
    if s.ndim != 2 or a.shape != s.shape:
        raise SedimentPhysicsError(f"slope and active must be equal 2-D shapes, got {s.shape} and {a.shape}")
    if not np.all(np.isfinite(s)) or np.any(s[a] < 0.0):
        raise SedimentPhysicsError("slope must be finite and >= 0 on active cells")
    area = _positive(cell_area_m2, "cell_area_m2")
    namespace = np if xp is None else xp
    return PhysicsGrid(shape=tuple(s.shape), cell_area_m2=area,
                       slope=freeze(to_device(np.ascontiguousarray(s), namespace)),
                       active=freeze(to_device(np.ascontiguousarray(a), namespace)), xp=namespace)


def physics_grid_from_graph(graph: Any) -> PhysicsGrid:
    """`PhysicsGrid` from a `maple_syrup.routing.RoutingGraph` (its final
    slope, active mask, dx^2 and namespace)."""
    return physics_grid(graph.slope, graph.active, graph.dx_m * graph.dx_m, xp=graph.xp)


# --- d50 ----------------------------------------------------------------------------------
def median_diameter_m(fractions: Any, diameter_m: Any) -> Any:
    """`route_sediment_xml.f90` 66-85 vectorized: cumulative fraction over
    classes (fine to coarse); at the first class where the cumulative sum
    reaches 0.5, `d50 = D1 / (2 dsum)` for class 1 (a proportional
    fraction of the first diameter), otherwise linear interpolation
    between `D(phi-1)` and `D(phi)`; `D(last)` when 0.5 is never reached
    (fractions summing below 0.5, including an empty cell).

    `fractions` `(..., n_classes)`, `diameter_m` `(n_classes,)` or
    broadcastable `(..., n_classes)`; result `(...)`."""
    from maple.core.backend import array_namespace, errstate

    xp = array_namespace(fractions)
    f = fractions
    n = f.shape[-1]
    diam = xp.asarray(diameter_m, dtype=np.float64)
    diam = xp.broadcast_to(diam, f.shape)
    cum = xp.cumsum(f, axis=-1)
    # The legacy `dsumlast` is the PREVIOUS cumulative sum exactly, not `cum - f`.
    prev = xp.concatenate([xp.zeros(f.shape[:-1] + (1,), dtype=np.float64), cum[..., :-1]], axis=-1)
    hit = (cum >= 0.5) & (prev < 0.5)
    reached = xp.any(hit, axis=-1)
    # Exactly one hit where reached (cum is nondecreasing); argmax gives it.
    index = xp.argmax(hit, axis=-1)
    idx = index[..., None]
    dsum = xp.take_along_axis(cum, idx, axis=-1)[..., 0]
    dsumlast = xp.take_along_axis(prev, idx, axis=-1)[..., 0]
    d_here = xp.take_along_axis(diam, idx, axis=-1)[..., 0]
    d_prev = xp.take_along_axis(diam, xp.maximum(idx - 1, 0), axis=-1)[..., 0]
    with errstate(xp=xp, divide="ignore", invalid="ignore"):
        first = (d_here / xp.where(dsum > 0.0, dsum, 1.0)) * 0.5
        width = dsum - dsumlast
        interp = d_prev + (d_here - d_prev) / xp.where(width > 0.0, width, 1.0) * (0.5 - dsumlast)
    d50 = xp.where(index == 0, first, interp)
    return xp.where(reached, d50, diam[..., n - 1])


# --- step result -------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SedimentPhysicsStep:
    """Result of `sediment_physics_step`. Class-indexed arrays are
    `(ny, nx, n_classes)` FP64 in the parameters' namespace; per-cell
    arrays `(ny, nx)`.

    requested_pickup_kg     detachment DEMAND over dt (raindrop + flow),
                            uncapped by holdings or availability
    raindrop_pickup_kg / flow_pickup_kg   the two contributions
    sediment_velocity_m_s   virtual velocity used this step: the law's
                            value where a transport law applies, else the
                            decayed memory (NOT capped at the current water
                            velocity; legacy); this is the new memory state
    deposition_rate_per_m   1 / (exponential mean travel distance) from the
                            LOCAL law at this cell where one applies, else 0
                            (no deposition law: the pool advects at the memory
                            velocity without depositing, as the legacy pool)
    law_applies             a transport law defined velocity and distance
    settle_mask             the whole mobile pool at this cell/class must
                            be requested for deposition this step (dry cell,
                            no excess stream power, zero rain energy)
    regime                  int8 REGIME_CODES
    d50_m, shear_velocity_m_s, reynolds_number, stream_power_w_m2,
    rain_energy_j_m2_mm, rain_energy_flux_j_m2_s, flow_energy_w_m2   per cell
    pickup_probability      flow-detachment probability per cell and class
    regime_counts           dict name -> count of active cell/class entries
    """

    dt_s: float
    requested_pickup_kg: Any
    raindrop_pickup_kg: Any
    flow_pickup_kg: Any
    sediment_velocity_m_s: Any
    deposition_rate_per_m: Any
    law_applies: Any
    settle_mask: Any
    regime: Any
    d50_m: Any
    shear_velocity_m_s: Any
    reynolds_number: Any
    stream_power_w_m2: Any
    rain_energy_j_m2_mm: Any
    rain_energy_flux_j_m2_s: Any
    pickup_probability: Any
    legacy_cap_applied: Any
    regime_counts: dict[str, Any]

    def travel_distance_m(self) -> Any:
        """Mean travel distance where a law applies, `inf` elsewhere
        (diagnostic only; the transport operator takes the rate)."""
        xp = self.rate_namespace()
        rate = self.deposition_rate_per_m
        return xp.where(rate > 0.0, 1.0 / xp.where(rate > 0.0, rate, 1.0), xp.inf)

    def rate_namespace(self) -> ModuleType:
        from maple.core.backend import array_namespace

        return array_namespace(self.deposition_rate_per_m)


def _require_grid_arrays(named: dict[str, Any], shape: tuple[int, ...], xp: ModuleType) -> None:
    from maple.core.backend import MixedArrayNamespaceError, array_namespace, is_array

    for name, array in named.items():
        if not is_array(array):
            raise SedimentPhysicsError(f"{name} must be a NumPy/CuPy array, got {type(array).__name__}")
    try:
        namespace = array_namespace(*named.values())
    except MixedArrayNamespaceError as exc:
        raise SedimentPhysicsError(str(exc)) from None
    if namespace is not xp:
        raise SedimentPhysicsError(f"arrays must be in the parameters' namespace {xp.__name__!r}, got {namespace.__name__!r}")
    for name, array in named.items():
        if tuple(array.shape) != shape:
            raise SedimentPhysicsError(f"{name} shape {tuple(array.shape)} != {shape}")
        if array.dtype != np.float64:
            raise SedimentPhysicsError(f"{name} must be float64, got {array.dtype}")


def _rain_energy(params: SedimentPhysicsParameters, intensity_mm_h: Any, rain_rate_mm_s: Any,
                 veg_percent: Any, raining: Any, xp: ModuleType) -> Any:
    """`raindrop_detachment.for` 24-54: KE in J m^-2 mm^-1, floored at 0 on
    the log model (documented boundary), vegetation factor per the
    selected form. Zero where not raining."""
    from maple.core.backend import errstate

    factor = 1.0 - VEGETATION_ENERGY_COEFFICIENT_PER_PERCENT * veg_percent
    with errstate(xp=xp, divide="ignore", invalid="ignore"):
        if params.ke_model == "wainwright_log":
            base = 11.9 + 8.73 * xp.log10(xp.where(raining, intensity_mm_h, 1.0))
            base = xp.maximum(base, 0.0)
            ke = base * factor
        elif params.ke_vegetation_form == "legacy_literal":
            # Literal Fortran precedence (lines 33-34): only the exponential
            # term is scaled, so cover raises the energy. Source relationship
            # kept as written.
            ke = 29.0 - 20.88 * xp.exp(-180.0 * rain_rate_mm_s) * factor
        else:  # "intended": explicit variation, whole energy scaled as in model 1
            ke = (29.0 - 20.88 * xp.exp(-180.0 * rain_rate_mm_s)) * factor
    return xp.where(raining, ke, 0.0)


def sediment_physics_step(
    params: SedimentPhysicsParameters,
    grid: PhysicsGrid,
    depth_m: Any,
    velocity_m_s: Any,
    rain_rate_m_per_s: Any,
    vegetation_cover_fraction: Any,
    active_layer_mass_kg: Any,
    previous_sediment_velocity_m_s: Any,
    dt_s: float,
) -> SedimentPhysicsStep:
    """One evaluation of the wet MAHLERAN laws on the current state.

    `depth_m`, `velocity_m_s` (water, from the routing step), `rain_rate_m_per_s`
    and `vegetation_cover_fraction` (0..1) are `(ny, nx)`;
    `active_layer_mass_kg` (MAPLE `active_layer.mass_kg`, current holdings)
    and `previous_sediment_velocity_m_s` (transport memory, zeros at event
    start) are `(ny, nx, n_classes)`; all FP64 in the parameters' namespace.
    Returns demand over `dt_s` and the transport inputs; see
    `SedimentPhysicsStep`. Pure; raises `SedimentPhysicsError` before
    returning anything on invalid input (one batched flag read)."""
    from maple.core.backend import (
        DeferredChecks,
        errstate,
        finite_flag,
        negative_flag,
        true_flag,
    )

    xp = params.xp
    if not isinstance(grid, PhysicsGrid) or grid.xp is not xp:
        raise SedimentPhysicsError("grid must be a PhysicsGrid in the parameters' namespace")
    ny, nx = grid.shape
    nc = params.n_classes
    dt = _positive(dt_s, "dt_s")
    cell = {"depth_m": depth_m, "velocity_m_s": velocity_m_s, "rain_rate_m_per_s": rain_rate_m_per_s,
            "vegetation_cover_fraction": vegetation_cover_fraction}
    cls = {"active_layer_mass_kg": active_layer_mass_kg,
           "previous_sediment_velocity_m_s": previous_sediment_velocity_m_s}
    _require_grid_arrays(cell, (ny, nx), xp)
    _require_grid_arrays(cls, (ny, nx, nc), xp)

    checks = DeferredChecks()
    for name, array in {**cell, **cls}.items():
        checks.require(finite_flag(array), f"{name} must be finite everywhere")
        checks.forbid(negative_flag(array), f"{name} must be >= 0 everywhere")
    checks.forbid(true_flag(vegetation_cover_fraction > 1.0), "vegetation_cover_fraction must be <= 1")
    active = grid.active
    checks.forbid(true_flag(~active & (velocity_m_s != 0.0)), "velocity_m_s must be 0 on inactive cells")

    area = grid.cell_area_m2
    ref = params.reference_interval_s
    rho = params.particle_density_kg_m3
    slope = grid.slope
    d3 = depth_m[..., None]
    a3 = active[..., None]

    with errstate(xp=xp, all="ignore"):
        # --- composition from current holdings --------------------------------
        total = xp.sum(active_layer_mass_kg, axis=-1)
        fractions = active_layer_mass_kg / xp.where(total > 0.0, total, 1.0)[..., None]
        fractions = xp.where((total > 0.0)[..., None], fractions, 0.0)
        d50 = median_diameter_m(fractions, params.d_diameter[0, 0])  # (ny, nx), m
        d50 = xp.where(active, d50, float(params.diameter_m[-1]))

        # --- hydraulic variables (route_sediment 89-95, SI) --------------------
        wet = active & (depth_m > 0.0)
        raining = active & (rain_rate_m_per_s > 0.0)
        ustar = xp.sqrt(GRAVITY_M_S2 * depth_m * slope)  # m/s
        reynolds = velocity_m_s * depth_m / params.kinematic_viscosity_m2_s
        # diffuse 31-32, conc 42-43, susp 22-23: rho_w g d v S in W m^-2.
        stream_power = 1000.0 * GRAVITY_M_S2 * depth_m * velocity_m_s * slope
        rain_mm_s = rain_rate_m_per_s * 1000.0
        intensity_mm_h = rain_rate_m_per_s * 3.6e6
        veg_percent = vegetation_cover_fraction * 100.0

        # --- regime (route_sediment 98-170, 180-183) ---------------------------
        concentrated_flow = wet & (reynolds >= REYNOLDS_CONCENTRATED)
        transitional = wet & ~concentrated_flow & (reynolds > REYNOLDS_TRANSITIONAL)
        rain_branch = wet & ~concentrated_flow & raining
        dry_branch = wet & ~concentrated_flow & ~raining
        raindrop_cells = rain_branch
        # 100 (Re >= 2500), 136 (rain, Re > 500), 157 (no rain, Re > 500).
        flow_detach_cells = concentrated_flow | transitional
        diffuse_cells = rain_branch
        conc_cells = dry_branch & transitional  # 156-160; classes under Re >= 2500 decided below
        regime_cell = xp.where(
            concentrated_flow, REGIME_CODES["concentrated"],
            xp.where(rain_branch & transitional, REGIME_CODES["transitional_rain"],
                     xp.where(rain_branch, REGIME_CODES["diffuse"],
                              xp.where(dry_branch & transitional, REGIME_CODES["transitional_dry"],
                                       xp.where(wet, REGIME_CODES["wet_no_law"], REGIME_CODES["dry"])))))

        # --- raindrop detachment (raindrop_detachment.for) ---------------------
        ke = _rain_energy(params, intensity_mm_h, rain_mm_s, veg_percent, raining, xp)  # J m^-2 mm^-1
        # 69-72 with the 268 pre-scaling: spa/1200 * (1200 ke r2)^spb * (100 S)^spc,
        # kg m^-2 per reference interval.
        energy_term = xp.maximum(ke * rain_mm_s * _LEGACY_QUANSAH_SCALE, 0.0)[..., None]
        slope_term = (slope * 100.0)[..., None]
        x_kg_m2 = (params.d_raindrop_a / _LEGACY_QUANSAH_SCALE) * energy_term ** params.d_raindrop_b \
            * slope_term ** params.d_raindrop_c
        # 82-85: x2 (up/downslope data), / density -> depth per reference interval; here
        # a depth RATE (m/s) = 2 X / rho / ref.
        raindrop_depth_rate = 2.0 * x_kg_m2 / rho / ref
        # 87-95: exp(-spq d_cm) under standing water (all raindrop cells are wet here).
        raindrop_depth_rate = raindrop_depth_rate * xp.exp(-params.d_attenuation * (d3 * 100.0))
        raindrop_depth_rate = xp.maximum(raindrop_depth_rate, 0.0)  # 96-98
        if params.raindrop_composition_scaling == "fraction":
            raindrop_depth_rate = raindrop_depth_rate * fractions
        # 103-109: cap for phi 2 only (legacy quirk), tptable = f hs / dt -> f hs / ref.
        cap_rate = fractions * params.d_raindrop_max / ref
        phi2 = xp.zeros((1, 1, nc), dtype=np.bool_)
        if nc >= 2:
            phi2[0, 0, 1] = True
        raindrop_capped = raindrop_cells[..., None] & phi2 & (raindrop_depth_rate > cap_rate)
        raindrop_depth_rate = xp.where(raindrop_capped, cap_rate, raindrop_depth_rate)
        # 110-112: zero if the class is absent; 117-120: exp(grav_propn) = 1.
        raindrop_depth_rate = xp.where(fractions > 0.0, raindrop_depth_rate, 0.0)
        raindrop_depth_rate = xp.where(raindrop_cells[..., None], raindrop_depth_rate, 0.0)

        # --- flow detachment (flow_detachment.for 18-47) ------------------------
        theta = (ustar ** 2)[..., None] / (params.sigma * GRAVITY_M_S2 * params.d_diameter)
        positive_theta = theta > 0.0
        p_const = xp.log(0.049 / (xp.where(positive_theta, theta, 1.0) * 0.25))
        p_par = -2.0 / math.pi
        inner = 1.0 - xp.exp(p_par * (p_const / 0.702) ** 2)
        p_pickup = 0.5 - 0.5 * xp.sign(p_const) * xp.sqrt(xp.maximum(inner, 0.0))
        p_pickup = xp.where(positive_theta, p_pickup, 0.0)
        flow_depth_rate = p_pickup * params.flow_detachment_depth_scale_m * fractions / ref
        flow_capped = flow_detach_cells[..., None] & (flow_depth_rate > cap_rate)
        flow_depth_rate = xp.where(flow_capped, cap_rate, flow_depth_rate)
        flow_depth_rate = xp.where(fractions > 0.0, flow_depth_rate, 0.0)
        flow_depth_rate = xp.where(flow_detach_cells[..., None], flow_depth_rate, 0.0)

        # Depth rates (solid-volume depth at particle density, legacy mm/s) to
        # mass demand over dt: rate * area * rho * dt.
        mass_factor = area * rho * dt
        raindrop_pickup = raindrop_depth_rate * mass_factor
        flow_pickup = flow_depth_rate * mass_factor
        requested = raindrop_pickup + flow_pickup

        # --- transport laws ---------------------------------------------------
        v3 = velocity_m_s[..., None]
        # Diffuse (diffuse_flow_transport.for 26-70): rain energy FLUX uses the
        # log model regardless of KE_model_type and no vegetation (26-27).
        ke_flux = (11.9 + 8.73 * xp.log10(xp.where(raining, intensity_mm_h, 1.0))) * rain_mm_s
        ke_flux = xp.where(raining, xp.maximum(ke_flux, 0.0), 0.0)  # J m^-2 s^-1, floored (boundary)
        kf3 = ke_flux[..., None]
        sp3 = stream_power[..., None]
        pm = params.d_particle_mass_g
        v_diffuse = 0.525 * kf3 ** 2.35 * sp3 ** 0.981 / pm  # cm/min
        v_diffuse = v_diffuse * (1.0e-2 / 60.0)  # m/s (legacy 1/6 -> mm/s, then 1e-3)
        L_diffuse = 5.0e-2 * kf3 ** 1.85 * sp3 ** 0.481 * pm ** (-0.425)  # m, used as the mean
        v_diffuse = xp.minimum(v_diffuse, v3)  # 68-70

        # Concentrated (conc_flow_transport.for 26-81).
        d50_3 = d50[..., None]
        depth_for_log = depth_m * (1000.0 if params.bagnold_depth_units == "legacy_mm" else 1.0)
        log_arg = 12.0 * depth_for_log[..., None] / d50_3
        bagnold = 4.554e-3 * (params.excess_density_kg_m3 * d50_3) ** 1.5 \
            * xp.log10(xp.where(log_arg > 0.0, log_arg, 1.0))
        bagnold = xp.maximum(bagnold, 0.0)  # 30-35
        xs = sp3 - bagnold
        capacity = xs > 0.0
        xs_pos = xp.where(capacity, xs, 0.0)
        L_conc = 2.85e-3 * xs_pos ** 1.31 * params.d_diameter ** (-0.94)
        if params.distance_convention == "legacy_literal":
            L_conc = L_conc * LEGACY_MEDIAN_FACTOR
        L_conc = xp.minimum(L_conc, CONCENTRATED_DISTANCE_CAP_M)
        v_conc = 1.92e-2 * xs_pos ** 1.01 * (1000.0 / 3600.0) * 1.0e-3  # m/h -> m/s
        v_conc = xp.minimum(v_conc, v3)

        # Suspended (suspended_transport.for 22-56).
        spf = xp.minimum(7.331976e-3 * sp3, SUSPENDED_EXPONENT_CAP)
        L_susp = 727.51805244 * xp.exp(spf) * xp.exp(-6.12683698 * params.d_diameter * 1000.0)
        if params.distance_convention == "legacy_literal":
            L_susp = L_susp * LEGACY_MEDIAN_FACTOR
        L_susp = L_susp * params.bagnold_density_scale
        v_susp = xp.broadcast_to(v3, (ny, nx, nc))

        # Suspension criterion (route_sediment 102-118).
        dstar = params.d_diameter * params.dstar_const_per_m
        susp_crit = xp.where(dstar <= 10.0, 4.0 * params.d_settling / dstar, 0.4 * params.d_settling)
        suspended = concentrated_flow[..., None] & (ustar[..., None] >= susp_crit)
        conc_class = (concentrated_flow[..., None] & ~suspended) | conc_cells[..., None]
        diffuse_class = xp.broadcast_to(diffuse_cells[..., None], (ny, nx, nc))

        regime = xp.where(suspended, REGIME_CODES["suspended"],
                          xp.broadcast_to(regime_cell[..., None], (ny, nx, nc))).astype(np.int8)

        # Assemble velocity, rate, settle and law masks.
        law = suspended | conc_class | diffuse_class
        v_law = xp.where(suspended, v_susp, xp.where(conc_class, v_conc, v_diffuse))
        L_law = xp.where(suspended, L_susp, xp.where(conc_class, L_conc, L_diffuse))
        # No capacity: concentrated without excess stream power (conc 49-54),
        # diffuse with zero rain energy (L -> 0, boundary), dry cells.
        no_capacity = (conc_class & ~capacity) | (diffuse_class & ~(kf3 > 0.0))
        settle = xp.broadcast_to(~wet[..., None], (ny, nx, nc)) | no_capacity
        law_defined = law & ~no_capacity & (L_law > 0.0)
        rate = xp.where(law_defined, 1.0 / xp.where(law_defined, L_law, 1.0), 0.0)
        decayed = recession_velocity(previous_sediment_velocity_m_s, dt,
                                     factor_per_reference_s=params.recession_factor_per_reference_s,
                                     reference_interval_s=ref)
        velocity = xp.where(law_defined, v_law, decayed)
        velocity = xp.where(a3, velocity, 0.0)
        velocity = xp.where(settle, 0.0, velocity)  # settled pools do not travel
        requested = xp.where(a3, requested, 0.0)
        cap_applied = raindrop_capped | flow_capped

        counts = {name: xp.sum(a3 & (regime == code)) for name, code in REGIME_CODES.items()}

    # Overflow is refused, never masked: the holdings total (an overflowed
    # total would silently become empty fractions), every intermediate law
    # value (masked or not) and every returned diagnostic must be finite.
    checks.require(finite_flag(total), "active_layer_mass_kg per-cell total overflowed FP64 (unsupported holdings)")
    for name, array in (("rain intensity", intensity_mm_h), ("rain energy term", energy_term),
                        ("raindrop law mass", x_kg_m2), ("rain energy flux", ke_flux),
                        ("diffuse velocity", v_diffuse), ("diffuse travel distance", L_diffuse),
                        ("Bagnold threshold", bagnold), ("concentrated travel distance", L_conc),
                        ("concentrated velocity", v_conc), ("suspended travel distance", L_susp),
                        ("dimensionless shear", theta), ("decayed memory velocity", decayed),
                        ("cap rate", cap_rate)):
        checks.require(finite_flag(array), f"physics intermediate {name} is non-finite (input outside the "
                                           "supported FP64 domain; refused, not clipped)")
    for name, array in (("requested_pickup_kg", requested), ("raindrop_pickup_kg", raindrop_pickup),
                        ("flow_pickup_kg", flow_pickup), ("sediment_velocity_m_s", velocity),
                        ("deposition_rate_per_m", rate), ("d50_m", d50), ("shear_velocity_m_s", ustar),
                        ("reynolds_number", reynolds), ("stream_power_w_m2", stream_power),
                        ("rain_energy_j_m2_mm", ke), ("rain_energy_flux_j_m2_s", ke_flux),
                        ("pickup_probability", p_pickup)):
        checks.require(finite_flag(array), f"physics produced non-finite {name}")
        checks.forbid(negative_flag(array), f"physics produced negative {name}")
    checks.forbid(true_flag(p_pickup > 1.0), "pickup probability exceeded 1")
    try:
        checks.resolve()
    except ValueError as exc:
        raise SedimentPhysicsError(str(exc)) from None

    return SedimentPhysicsStep(
        dt_s=dt,
        requested_pickup_kg=requested,
        raindrop_pickup_kg=xp.where(a3, raindrop_pickup, 0.0),
        flow_pickup_kg=xp.where(a3, flow_pickup, 0.0),
        sediment_velocity_m_s=velocity,
        deposition_rate_per_m=rate,
        law_applies=law_defined,
        settle_mask=settle,
        regime=regime,
        d50_m=d50,
        shear_velocity_m_s=xp.where(active, ustar, 0.0),
        reynolds_number=xp.where(active, reynolds, 0.0),
        stream_power_w_m2=xp.where(active, stream_power, 0.0),
        rain_energy_j_m2_mm=ke,
        rain_energy_flux_j_m2_s=ke_flux,
        pickup_probability=xp.where(a3, p_pickup, 0.0),
        legacy_cap_applied=cap_applied & a3,
        regime_counts=counts,
    )
