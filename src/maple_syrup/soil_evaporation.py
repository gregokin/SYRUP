"""Evaporation-ONLY vegetation/surface forcing helper for the SYRUP soil column (Phase 1, interevent intervals).

This module turns a shared, TERRAIN-APPLIED reference potential evapotranspiration rate (PET, m/s per cell, from
`maple_syrup.weather_pet`: the aspect factor and cos(slope) are ALREADY inside it and are NOT applied again here),
an explicit aggregate vegetation cover fraction C in [0, 1] (LEAF's spatial vegetation contribution) and the standing
surface water H (m) into two SEPARATE physical branches, exactly as the EcoHyD descendant computes them
(`WaterCell::actLayerEvapotranspiration`, GitHub snapshot 80b10e6b `WaterCell.cpp:27-120`; both Dryad copies are
byte-identical in these lines):

  surface branch (WaterCell.cpp:48-75):  potential_surface_E_rate = PET x (1 - C / 2)            [m/s]
                                          actual surface E = min(rate x dt, H), the rest is unmet; H' = H - actual
  soil branch (two staged fields):        canopy = Vegcoef1 - Vegcoef2 x C                         (WaterCell.cpp:33-34)
                                          potential_combined_soil_rate = PET x canopy x EPfactor    [m/s] (UNPARTITIONED)
                                          soil_evaporation_share = 1 - clip(Veg_ETcoef1 x C - Veg_ETcoef2, 0, 1)  [-]

The native column (`subsurface.step(..., potential_evaporation_m_s=potential_combined_soil_rate,
evaporation_share=soil_evaporation_share)`) then applies the USER-SELECTED post-cap law (2026-11-10), in the order of
the reference `WaterCell.cpp:84-92` which caps the COMBINED upper-layer loss by the available water BEFORE any E/T
labelling:

    potential_combined_soil_loss = potential_combined_soil_rate x stress(theta_0) x dt
    moisture_limited_combined    = min(potential_combined_soil_loss, max(top_water - residual_water, 0))
    actual_soil_evaporation      = moisture_limited_combined x soil_evaporation_share       (the only water debit)
    evaporation_unmet            = potential_combined_soil_loss x share - actual            (partitioned potential E - actual E)

The (1 - share) part of the moisture-limited combined loss is NOT subtracted: it stays in the layer for LEAF's later,
separately requested and accounted transpiration (`subsurface_transpiration`), checked against the water that
remains. Nothing restores a T fraction after diffusion and no global E + T <= PET cap exists. Example (stress-free):
withdrawable 1 mm, potential combined >= 1 mm, share 0.5 -> actual E 0.5 mm and 0.5 mm withdrawable left for LEAF;
potential combined 0.2 mm -> actual E 0.1 mm. Share 0 removes no water; share 1 reproduces the historical column
arithmetic exactly. `StagedEvaporation.native_step_kwargs()` hands both fields to the column together so a caller
cannot feed a before-cap-partitioned potential as if it were the combined one.

Attribution: the source computes `TL1 = max(0, Veg_ETcoef1 C - Veg_ETcoef2) * EPL1` WITHOUT an upper clip at 1
(`WaterCell.cpp:126-133`) and `EL1 = max(EPL1 - TL1, 0)` (`152-158`). Since EL1 = EPL1 x max(1 - max(0, a C - b), 0)
= EPL1 x (1 - clip(a C - b, 0, 1)), the staged share gives EXACT correspondence of the evaporation diagnostic; the
clip at 1 is a bounded ADAPTATION of the source T fraction (whose raw value may exceed 1, e.g. 1.01 for the Tabernas
coefficients at C = 1), not literal T-fraction code identity. No transpiration budget is moved to SYRUP by it.

`PARTITION_LEVEL = "post_cap_share"` records the accepted choice. The earlier potential-level candidate (partition
applied to the potential BEFORE the cap) is superseded and no longer staged; an informational E-only potential
(`informational_e_only_fraction`, ledger `soil_e_only_informational_m3`) is reported as a SCALAR total only and is
NOT a column forcing.

The soil fields are FORCING, not fluxes: the column's existing layer-0 law applies the quadratic moisture stress
(theta / theta_evaporation_stress)^2 below the stress threshold (the user may set that threshold to the wilting
point WP, the source's `WP`) and the residual cap (`resWater`), and reports the actual E and the unmet part. Neither
branch is transpiration: LEAF computes plant withdrawals and SYRUP debits them separately and exactly once. No deep
direct soil evaporation (lower-layer evaporation is 0 in the source, `transpirationlower`), no temperature term
(temperature already entered the shared PET).

What is reference, what is a departure (read before parameterising):

* The source withdraws the COMBINED upper-layer evapotranspiration EPL1 = min(PET x veg x EPfactor x stress,
  available) FIRST and only then labels TL1 and EL1 (`WaterCell.cpp:122-158`, Tong's "separation of
  evapotranspiration"). Copying EPL1 wholesale as soil evaporation and then adding LEAF transpiration would
  double-debit the upper layer; this helper and the native post-cap law debit only EL1's counterpart and leave the
  rest for LEAF's own request. The original 2009 paper does not identify E and T separately.
* The EcoHyD evaporation factor `EPfactor` (`ep_factor_cell`, a site/crust parameter, dimensionless) is applied
  HERE (`soil_factor`) and NOWHERE ELSE: `prepare_evaporation` refuses a column whose `evaporation_factor` is not
  exactly 1.0, so the factor cannot be applied twice (`SOIL_FACTOR_PLACEMENT`). The column's historical
  `evaporation_factor` (MAHLERAN 1.2 bare-soil form) stays available for columns that do not use this helper.
* Coefficients are EXPLICIT inputs with declared units (all dimensionless, cover as a fraction): there are no
  defaults and no universal calibration. The downloaded Tabernas parameter file (`Parameters/
  soilparameters_tabernas.txt`: VegregETL1 1.2, VegregETL2 0.2, Veg_ETcoef1 1.05, Veg_ETcoef2 0.04, EP 1) is a
  SITE example, not a default. A parameterisation that would make the canopy factor negative anywhere on C in
  [0, 1] is refused at construction (the source would produce a negative flux; refusing is the only safe choice).
* The source applies these on DAILY steps including rainy days; the user's rule is that E, T and growth run ONLY in
  interevent intervals. This helper has no calendar and no phase: it computes forcing for whatever interval the
  caller declares. It does NOT make evaporation during wind/rain/recession safe or authorised; the native
  coordinator (MASY) enforces the event rule, and `weather_pet` refuses to integrate PET for non-interevent phases.

Scope: a PURE staged helper. It allocates fresh outputs, validates every cell in ONE batched packet (one host read),
rejects BEFORE publishing (caller arrays untouched) and returns the surface branch (new surface water, actual and
unmet surface evaporation, a surface-water ledger over active cells) plus the soil potential forcing array. It
holds no authoritative water (surface water belongs to the storm/pond owner, soil water to the column), performs no
growth, calls no storm routine and does not step the column. Inactive cells: the surface water is returned unchanged
and every flux/forcing output is 0 there, so the array can be passed to `subsurface.step` directly. Arrays live in
the column's namespace (NumPy for "numpy"/"numba" columns, CuPy on the column's device for "cuda"): a few N-sized
vectorised maps, no compiled kernel.

Units: PET and the returned soil forcing in m/s; cover a fraction in [0, 1]; surface water and evaporation in m;
dt in s; ledger volumes in m3 (depth x column.cell_area_m2, multiplied exactly once, active cells only). Nothing
here was run by its author against a GPU; Codex records executions. A paper/source equation check is not a field
validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = [
    "PACKET_LENGTH",
    "PARTITION_LEVEL",
    "SCRATCH_DOUBLES_PER_CELL",
    "SOIL_FACTOR_PLACEMENT",
    "SOURCE_IDENTITY",
    "EvaporationMemory",
    "EvaporationOperator",
    "EvaporationPartition",
    "StagedEvaporation",
    "SurfaceEvaporationLedger",
    "estimate_evaporation_memory",
    "pet_bridge_bytes",
    "pet_terrain_rate_from_demand",
    "prepare_evaporation",
    "stage_evaporation",
]

SubsurfaceError = ss.SubsurfaceError
SubsurfaceMemoryError = ss.SubsurfaceMemoryError

SOURCE_IDENTITY = {
    "model": "EcoHyD (Tietjen-lineage), Baldauf, Cantón & Tietjen 2023 GitHub snapshot",
    "commit": "80b10e6b6b82ad16e9fd41a31901cc3157d880e9",
    "local_root": "reference_models/ecohyd/github_baldauf_2023/source/model/src/",
    "WaterCell.cpp_sha256": "af476e1aa727ed515d75ed06a4a25b5285470caddebb141c35b8c0cf6201d9e9",
    "dryad_WaterCell.cpp_sha256": "65c3978d9a474a582598c633bbcff243acd39d57c04220c2d6136f1f5bdc1389",
    "canopy_function": "WaterCell.cpp:33-34 vegfunctionL1 = Vegcoef1 - Vegcoef2 * (cGtotal + cAtotal + cStotal)",
    "surface_potential": "WaterCell.cpp:48-50 potL0EP = potEP * (1 - cover / 2) * cos_inclination (terrain already in PET here)",
    "surface_actual": "WaterCell.cpp:55-67 min(potL0EP, waterL0_abs)",
    "upper_combined_et": "WaterCell.cpp:84-92 EPL1 = min(potEP * cos * vegfunctionL1 * ep_factor, (theta - resWater) * depth)",
    "transpiration_fraction": "WaterCell.cpp:126-133 TL1 = max(0, Veg_ETcoef1 * cover - Veg_ETcoef2) * EPL1 (no upper "
                              "clip at 1 in the source)",
    "evaporation_partition": "WaterCell.cpp:152-158 EL1 = max(EPL1 - TL1, 0) (diagnostic, after the combined withdrawal)",
    "helper_adaptation": "fT = clip(Veg_ETcoef1 C - Veg_ETcoef2, 0, 1) and E-only factor (1 - fT): bounded-complement "
                         "adaptation of the two source bounds, not literal T-fraction code identity",
    "lower_layer_evaporation": "WaterCell.cpp:139-147 transpirationlower: lower-layer E is 0",
    "site_example_parameters": "Parameters/soilparameters_tabernas.txt (VegregETL1 1.2, VegregETL2 0.2, "
                               "Veg_ETcoef1 1.05, Veg_ETcoef2 0.04, EP 1): a site example, NOT a default",
    "paper": "Tietjen, Zehe & Jeltsch 2009, Water Resour. Res. 45, W01418, https://doi.org/10.1029/2007WR006589 "
             "(combined upper-layer ET only; E/T not separately identified)",
}
#: Where the EcoHyD evaporation factor (EPfactor) is applied: in this helper's `soil_factor`; the column's
#: `evaporation_factor` must be exactly 1.0 when its forcing comes from here.
SOIL_FACTOR_PLACEMENT = "helper"
#: User-accepted (2026-11-10) placement of the soil evaporation share: AFTER the column's moisture/residual cap on the
#: combined potential loss (reference order WaterCell.cpp:84-92). The earlier potential-level candidate is superseded.
PARTITION_LEVEL = "post_cap_share"
PACKET_LENGTH = 11
# Transient N-sized temporaries of one staging (canopy, transpiration fraction / its complement, masked copies for
# the active-only sums, validity masks) on top of the five outputs; declared conservatively.
SCRATCH_DOUBLES_PER_CELL = 6
SCRATCH_MASK_BYTES_PER_CELL = 8
_PACKET_SCRATCH_BYTES = 4 * PACKET_LENGTH * 8
_F8 = 8


# --- parameterisation ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaporationPartition:
    """Explicit, dimensionless EcoHyD-descendant coefficients of the E-only partition (no defaults):

    canopy_intercept  = Vegcoef1   (source `VegregETL1`): canopy factor at zero cover
    canopy_slope      = Vegcoef2   (source `VegregETL2`): canopy factor reduction per unit cover
    transpiration_slope  = Veg_ETcoef1: transpiration fraction regression slope on cover
    transpiration_offset = Veg_ETcoef2: transpiration fraction regression offset
    soil_factor       = EPfactor   (source `EP`): additional site/crust factor on the soil potential (applied here only)

    Validation: every value finite and real; canopy factor >= 0 at C = 0 AND C = 1 (it is linear, so then >= 0 on the
    whole domain); soil_factor >= 0. The source bounds the regression only below (max(0, .)) and bounds its E
    diagnostic separately (max(EPL1 - TL1, 0)); the helper's clip to [0, 1] plus the complement (1 - fT) is the
    bounded-complement adaptation of those two bounds (same nonnegative E-only response), not literal source code."""

    canopy_intercept: float
    canopy_slope: float
    transpiration_slope: float
    transpiration_offset: float
    soil_factor: float

    def __post_init__(self) -> None:
        for name in ("canopy_intercept", "canopy_slope", "transpiration_slope", "transpiration_offset", "soil_factor"):
            object.__setattr__(self, name, ss._real(getattr(self, name), name))
        if self.canopy_intercept < 0.0:
            raise SubsurfaceError(f"canopy_intercept (Vegcoef1) must be >= 0 (canopy factor at zero cover), got "
                                  f"{self.canopy_intercept!r}; refusing a negative potential evaporation")
        if self.canopy_intercept - self.canopy_slope < 0.0:
            raise SubsurfaceError(f"canopy_intercept - canopy_slope must be >= 0 (canopy factor at full cover), got "
                                  f"{self.canopy_intercept!r} - {self.canopy_slope!r}; refusing a negative potential "
                                  "evaporation instead of producing a negative physical flux")
        if self.soil_factor < 0.0:
            raise SubsurfaceError(f"soil_factor (EPfactor) must be >= 0, got {self.soil_factor!r}")
        # Derived endpoint values must be representable: finite coefficients can still overflow in the differences
        # and products the staging evaluates (canopy at C = 0 and 1, the raw transpiration regression at C = 0 and 1,
        # the soil fraction at both ends). Refused here, never clipped later.
        derived = (self.canopy_intercept - self.canopy_slope, self.transpiration_slope - self.transpiration_offset,
                   -self.transpiration_offset, self.canopy_intercept * self.soil_factor,
                   (self.canopy_intercept - self.canopy_slope) * self.soil_factor)
        if not all(math.isfinite(v) for v in derived):
            raise SubsurfaceError("the partition's derived endpoint values (canopy factor, transpiration regression, "
                                  "soil fraction at zero and full cover) are not finite in FP64; refused, not clipped")

    def canopy_factor(self, cover: float) -> float:
        """Vegcoef1 - Vegcoef2 C for one scalar cover in [0, 1] (host arithmetic, documentation/tests)."""
        c = _cover_scalar(cover)
        return self.canopy_intercept - self.canopy_slope * c

    def transpiration_fraction(self, cover: float) -> float:
        """clip(Veg_ETcoef1 C - Veg_ETcoef2, 0, 1) for one scalar cover (host arithmetic, documentation/tests). The
        upper clip is the helper's adaptation (the source has max(0, .) only and bounds E separately)."""
        c = _cover_scalar(cover)
        raw = self.transpiration_slope * c - self.transpiration_offset
        return 0.0 if raw < 0.0 else (min(raw, 1.0))

    def combined_potential_fraction(self, cover: float) -> float:
        """canopy x EPfactor: the UNPARTITIONED combined potential soil loss as a fraction of the terrain-applied PET
        (what `potential_combined_soil_rate_m_s` carries to the native column)."""
        return self.canopy_factor(cover) * self.soil_factor

    def evaporation_share(self, cover: float) -> float:
        """1 - clip(Veg_ETcoef1 C - Veg_ETcoef2, 0, 1): the post-cap soil evaporation share (what
        `soil_evaporation_share` carries to the native column)."""
        return 1.0 - self.transpiration_fraction(cover)

    def informational_e_only_fraction(self, cover: float) -> float:
        """canopy x share x EPfactor: the before-cap E-only potential as a fraction of PET. INFORMATIONAL ONLY (the
        superseded potential-level candidate); never feed it to the native column as `potential_evaporation_m_s`
        together with a share, that would partition twice."""
        return self.combined_potential_fraction(cover) * self.evaporation_share(cover)

    def surface_potential_fraction(self, cover: float) -> float:
        """1 - C / 2: the surface-water potential as a fraction of the terrain-applied PET."""
        return 1.0 - _cover_scalar(cover) / 2.0

    def summary(self) -> dict[str, Any]:
        return {
            "canopy_intercept_Vegcoef1": self.canopy_intercept,
            "canopy_slope_Vegcoef2": self.canopy_slope,
            "transpiration_slope_Veg_ETcoef1": self.transpiration_slope,
            "transpiration_offset_Veg_ETcoef2": self.transpiration_offset,
            "soil_factor_EPfactor": self.soil_factor,
            "soil_factor_placement": SOIL_FACTOR_PLACEMENT,
            "partition_level": PARTITION_LEVEL,
            "partition_status": "user-accepted 2026-11-10: soil evaporation share applied AFTER the column's moisture cap "
                                "on the combined potential (reference order); potential-level candidate superseded",
            "units": "all dimensionless; cover is a fraction in [0, 1]; PET in m/s; surface water in m",
            "soil_law": "potential_combined_soil_rate = PET_terrain x (Vegcoef1 - Vegcoef2 C) x EPfactor; share = "
                        "1 - clip(Veg_ETcoef1 C - Veg_ETcoef2, 0, 1); native column: actual E = min(potential x stress x dt, "
                        "available) x share",
            "soil_law_attribution": "canopy, EPfactor and the combined cap from the source; E correspondence exact "
                                    "(EL1 = max(EPL1 - TL1, 0) = EPL1 (1 - clip(a C - b, 0, 1))); the clip at 1 is a bounded "
                                    "adaptation of the source T fraction, not literal T-fraction code identity",
            "surface_law": "potential_surface_E = PET_terrain x (1 - C / 2); actual <= standing surface water",
            "not_included": "transpiration (LEAF), deep direct evaporation (0), temperature (in PET), terrain (in PET), "
                            "global E + T <= PET cap (none)",
            "source": dict(SOURCE_IDENTITY),
        }


def _cover_scalar(cover: Any) -> float:
    c = ss._real(cover, "cover")
    if c < 0.0 or c > 1.0:
        raise SubsurfaceError(f"cover must be a fraction in [0, 1], got {c!r}")
    return c


# --- memory ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaporationMemory:
    """Bytes of one staging in the column namespace: the caller's three input vectors, the FIVE fresh outputs
    (surface after, surface E, surface unmet, combined soil potential rate, soil evaporation share), the declared
    transient temporaries and the packet; `required_bytes` is the admitted side (host for numpy/numba, device for
    cuda) plus the column reserve and the caller's extra reserve. Scope as `subsurface.MEMORY_SCOPE`."""

    backend: str
    n_cells: int
    input_bytes: int
    output_bytes: int
    scratch_bytes: int
    packet_bytes: int
    reserve_bytes: int
    extra_reserve_bytes: int
    host_bytes: int
    device_bytes: int
    required_bytes: int
    total_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def estimate_evaporation_memory(n_cells: int, *, backend: str, reserve_bytes: int = 0,
                                extra_reserve_bytes: int = 0) -> EvaporationMemory:
    """Pure arithmetic (no allocation, no query)."""
    n = ss._checked_int(n_cells, "n_cells")
    if n < 1:
        raise SubsurfaceError("n_cells must be >= 1")
    if backend not in ss.BACKENDS:
        raise SubsurfaceError(f"backend must be one of {ss.BACKENDS}, got {backend!r}")
    reserve = ss._checked_int(reserve_bytes, "reserve_bytes")
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    inputs = 3 * n * _F8
    outputs = 5 * n * _F8
    scratch = SCRATCH_DOUBLES_PER_CELL * n * _F8 + SCRATCH_MASK_BYTES_PER_CELL * n + _PACKET_SCRATCH_BYTES
    packet = PACKET_LENGTH * _F8
    compute = inputs + outputs + scratch
    host = compute if backend != "cuda" else packet
    device = compute if backend == "cuda" else 0
    side = device if backend == "cuda" else host
    required = side + reserve + extra
    return EvaporationMemory(backend, n, inputs, outputs, scratch, packet, reserve, extra, host, device, required,
                             host + device + reserve + extra)


# --- operator ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class EvaporationOperator:
    """A partition bound to one prepared column (namespace, mask, area, memory query). Holds no water."""

    column: Any
    partition: EvaporationPartition
    backend: str
    xp: Any
    n_cells: int
    extra_reserve_bytes: int
    memory: EvaporationMemory

    def summary(self) -> dict[str, Any]:
        return {"partition": self.partition.summary(), "backend": self.backend, "n_cells": self.n_cells,
                "n_active": int(self.column.n_active), "cell_area_m2": float(self.column.cell_area_m2),
                "column_evaporation_factor": float(self.column.layers.evaporation_factor),
                "memory": self.memory.as_dict(), "host_reads_per_staging": 1, "packet_length": PACKET_LENGTH,
                "partition_level": PARTITION_LEVEL,
                "scope": "pure staged forcing: surface branch applied to the returned copy only; soil branch is the "
                         "UNPARTITIONED combined potential rate plus the post-cap evaporation share for subsurface.step "
                         "(native_step_kwargs); no transpiration, growth, storm or calendar; the coordinator restricts "
                         "use to interevent intervals"}


def prepare_evaporation(column: Any, partition: EvaporationPartition, *, extra_reserve_bytes: int = 0) -> EvaporationOperator:
    """Bind `partition` to `column`. Refuses a column whose `evaporation_factor` is not exactly 1.0: the EcoHyD
    evaporation factor is `partition.soil_factor` and is applied here only (`SOIL_FACTOR_PLACEMENT`); any other
    column factor would multiply the soil potential twice. Admits the per-staging requirement against the column's
    memory query before returning (nothing is allocated here)."""
    ss._check_column(column)
    if not isinstance(partition, EvaporationPartition):
        raise SubsurfaceError(f"partition must be an EvaporationPartition, got {type(partition).__name__}")
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    factor = float(column.layers.evaporation_factor)
    if factor != 1.0:
        raise SubsurfaceError(
            f"the column's evaporation_factor is {factor!r}; the E-only helper applies the EcoHyD evaporation factor "
            f"itself (soil_factor={partition.soil_factor!r}, placement {SOIL_FACTOR_PLACEMENT!r}), so the column factor "
            "must be exactly 1.0 to avoid applying a factor twice. Build the layers with evaporation_factor=1.0 or do "
            "not use this helper for that column")
    memory = estimate_evaporation_memory(column.n_cells, backend=column.backend, reserve_bytes=column.reserve_bytes,
                                         extra_reserve_bytes=extra)
    ss._admit(_as_estimate(memory), column.available_bytes_fn, "evaporation staging preparation")
    return EvaporationOperator(column, partition, column.backend, column.xp, int(column.n_cells), extra, memory)


class _Admitted:
    """Minimal duck-typed estimate for `subsurface._admit` (which reads required/reserve/n_layers/n_cells/backend)."""

    def __init__(self, memory: EvaporationMemory, n_layers: int):
        self.required_bytes = memory.required_bytes
        self.reserve_bytes = memory.reserve_bytes + memory.extra_reserve_bytes
        self.n_layers = n_layers
        self.n_cells = memory.n_cells
        self.backend = memory.backend


def _as_estimate(memory: EvaporationMemory, n_layers: int = 0) -> _Admitted:
    return _Admitted(memory, n_layers)


# --- staging -------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SurfaceEvaporationLedger:
    """Surface-water branch over ACTIVE cells (m3, area multiplied once): after - before = -evaporation. The soil
    branch is reported as POTENTIAL totals, forcing figures, not fluxes: `soil_combined_potential_m3` (sum of the
    UNPARTITIONED combined rate x dt x area, what the column caps) and `soil_e_only_informational_m3` (the same times
    the share; the superseded before-cap E-only potential, informational only). The actual soil E is reported by the
    column step, not here."""

    storage_before_m3: float
    storage_after_m3: float
    evaporation_m3: float
    unmet_m3: float
    residual_m3: float
    bound_m3: float
    n_terms: int
    soil_combined_potential_m3: float
    soil_e_only_informational_m3: float
    dt_s: float
    n_active: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True, eq=False)
class StagedEvaporation:
    """Fresh arrays of the column namespace (N,): the surface water AFTER the surface branch, the actual and unmet
    surface evaporation (m), the UNPARTITIONED combined potential soil loss rate (m/s;
    `potential_combined_soil_rate_m_s`, NOT an E-only potential) and the post-cap soil evaporation share
    (`soil_evaporation_share`, [0, 1] on every cell). Pass BOTH to the column with `native_step_kwargs()`.
    Inactive cells: surface water unchanged, surface fluxes and the combined rate 0, the share still a valid value."""

    surface_water_after_m: Any
    surface_evaporation_m: Any
    surface_evaporation_unmet_m: Any
    potential_combined_soil_rate_m_s: Any
    soil_evaporation_share: Any
    dt_s: float
    ledger: SurfaceEvaporationLedger
    memory: EvaporationMemory
    partition: EvaporationPartition

    def native_step_kwargs(self) -> dict[str, Any]:
        """The two soil forcing fields paired for `subsurface.step` / `subsurface_transpiration.step_with_transpiration`:
        `{"potential_evaporation_m_s": combined rate, "evaporation_share": share}`. Using this pairing is how a caller
        avoids feeding a before-cap-partitioned potential (which no longer exists here) as the combined one."""
        return {"potential_evaporation_m_s": self.potential_combined_soil_rate_m_s,
                "evaporation_share": self.soil_evaporation_share}


_FLAG_TEXT = (
    "pet_terrain_m_s must be finite and >= 0 on every cell",
    "cover must be finite and within [0, 1] on every cell",
    "surface_water_m must be finite and >= 0 on every cell",
    "staging produced a non-finite output (overflow of rate x dt, or a non-finite share)",
)


def stage_evaporation(operator: EvaporationOperator, *, pet_terrain_m_s: Any, cover: Any, surface_water_m: Any,
                      dt_s: Any) -> StagedEvaporation:
    """One pure staging (see the module docstring). Inputs are (N,) float64 C-contiguous arrays of the column
    namespace and are never written; `dt_s` >= 0 is the declared interval (0 gives zero demand). Metadata checks,
    then admission, then allocation, then the vectorised laws, then ONE host packet; any invalid cell (checked on
    every cell, inactive included) or non-finite output rejects before anything is published."""
    if not isinstance(operator, EvaporationOperator):
        raise SubsurfaceError("operator must come from prepare_evaporation")
    column, part = operator.column, operator.partition
    ss._check_column(column)
    if not isinstance(part, EvaporationPartition):
        raise SubsurfaceError("operator.partition must be an EvaporationPartition")
    # The requirement is re-derived from the ACTUAL column every time; a hand-built operator carrying a fabricated
    # memory record, cell count or namespace cannot lower the admitted requirement.
    memory = estimate_evaporation_memory(column.n_cells, backend=column.backend, reserve_bytes=column.reserve_bytes,
                                         extra_reserve_bytes=operator.extra_reserve_bytes)
    if (operator.memory != memory or operator.n_cells != int(column.n_cells) or operator.xp is not column.xp
            or operator.backend != column.backend):
        raise SubsurfaceError("operator record does not match its column (memory estimate, cell count, namespace or "
                              "backend); rebuild it with prepare_evaporation")
    if float(column.layers.evaporation_factor) != 1.0:
        raise SubsurfaceError("the column's evaporation_factor must be exactly 1.0 for the E-only helper (factor once)")
    xp = column.xp
    n = int(column.n_cells)
    dt = ss._check_dt(dt_s)
    for name, array in (("pet_terrain_m_s", pet_terrain_m_s), ("cover", cover), ("surface_water_m", surface_water_m)):
        ss._check_dynamic(column, name, array, (n,))
    if column.preflight is not None:
        column.preflight()
    ss._admit(_as_estimate(memory), column.available_bytes_fn, "evaporation staging")
    from maple.core.backend import errstate, to_host

    active = column.active
    inactive = ~active
    with errstate(xp=xp, all="ignore"):
        # validity (every cell)
        bad_pet = ~xp.isfinite(pet_terrain_m_s) | (pet_terrain_m_s < 0.0)
        bad_cover = ~xp.isfinite(cover) | (cover < 0.0) | (cover > 1.0)
        bad_h = ~xp.isfinite(surface_water_m) | (surface_water_m < 0.0)
        # soil branch, two fields: the UNPARTITIONED combined potential rate PET x canopy x EPfactor and the post-cap
        # share 1 - clip(Veg_ETcoef1 C - Veg_ETcoef2, 0, 1); the column partitions AFTER its moisture cap
        canopy = part.canopy_slope * cover
        xp.subtract(part.canopy_intercept, canopy, out=canopy)  # Vegcoef1 - Vegcoef2 C
        share = part.transpiration_slope * cover  # fresh output array
        xp.subtract(share, part.transpiration_offset, out=share)  # Veg_ETcoef1 C - Veg_ETcoef2
        xp.clip(share, 0.0, 1.0, out=share)
        xp.subtract(1.0, share, out=share)  # 1 - fT
        soil = pet_terrain_m_s * canopy  # fresh output array
        xp.multiply(soil, part.soil_factor, out=soil)
        # surface branch: PET x (1 - C / 2) x dt, bounded by the standing water
        xp.multiply(cover, 0.5, out=canopy)
        xp.subtract(1.0, canopy, out=canopy)  # 1 - C / 2
        demand = pet_terrain_m_s * canopy
        xp.multiply(demand, dt, out=demand)
        del canopy
        evap = xp.minimum(demand, surface_water_m)
        xp.subtract(demand, evap, out=demand)  # unmet
        unmet = demand
        after = surface_water_m - evap
        # inactive cells: surface water unchanged, no flux, no combined forcing (the share keeps its valid value,
        # the column requires a finite [0, 1] share on every cell)
        for vector in (soil, evap, unmet):
            xp.copyto(vector, 0.0, where=inactive)
        xp.copyto(after, surface_water_m, where=inactive)
        nonfinite = (~xp.isfinite(soil) | ~xp.isfinite(evap) | ~xp.isfinite(unmet) | ~xp.isfinite(after)
                     | ~xp.isfinite(share))
        before_active = xp.where(active, surface_water_m, 0.0)
        after_active = xp.where(active, after, 0.0)
        soil_depth = soil * dt  # combined potential depth (active cells carry the forcing, inactive are 0)
        e_only_depth = soil_depth * share  # informational before-cap E-only potential depth
        values = [xp.count_nonzero(bad_pet), xp.count_nonzero(bad_cover), xp.count_nonzero(bad_h),
                  xp.count_nonzero(active & nonfinite), xp.sum(before_active), xp.sum(after_active), xp.sum(evap),
                  xp.sum(unmet), xp.sum(soil_depth), xp.sum(e_only_depth), xp.max(soil)]
        stacked = xp.stack([xp.asarray(v, dtype=np.float64) for v in values])
        del bad_pet, bad_cover, bad_h, nonfinite, before_active, after_active, soil_depth, e_only_depth
    packet = np.asarray(to_host(stacked), dtype=np.float64)  # the ONE host read
    if packet.shape != (PACKET_LENGTH,):
        raise SubsurfaceError(f"internal: packet shape {packet.shape}")
    for bit in range(4):
        if packet[bit] != 0.0:
            raise SubsurfaceError(f"evaporation staging refused: {_FLAG_TEXT[bit]} ({int(packet[bit])} cell(s)); "
                                  "nothing was published, inputs untouched")
    area = float(column.cell_area_m2)
    before, after_v, evap_v, unmet_v, soil_v, e_only_v = (float(v) * area for v in packet[4:10])
    n_active = int(column.n_active)
    n_terms = 3 * n_active + PACKET_LENGTH
    scale = max(abs(before), abs(after_v), abs(evap_v))
    if not all(math.isfinite(v) for v in (before, after_v, evap_v, unmet_v, soil_v, e_only_v)):
        raise SubsurfaceError("evaporation staging refused: non-finite surface ledger (overflow in the volume sums); "
                              "nothing was published")
    residual = (after_v - before) + evap_v
    bound = volume_roundoff_bound_m3(n_terms, scale)
    if abs(residual) > bound:
        raise SubsurfaceError(f"surface evaporation closure violated: residual {residual!r} m3 exceeds the FP64 bound "
                              f"{bound!r} m3 ({n_terms} terms); nothing was published")
    ledger = SurfaceEvaporationLedger(before, after_v, evap_v, unmet_v, residual, bound, n_terms, soil_v, e_only_v, dt,
                                      n_active)
    return StagedEvaporation(after, evap, unmet, soil, share, dt, ledger, memory, part)


# --- bridge from the shared PET library -----------------------------------------------------------------------------------
def pet_bridge_bytes(n_cells: int) -> int:
    """Bytes the cold PET bridge allocates: ONE fresh (N,) float64 array (the detached per-cell copy, converted to
    m/s in place). Admitted against the column's memory query together with the column reserve before the copy."""
    return ss._checked_int(n_cells, "n_cells") * _F8


def pet_terrain_rate_from_demand(demand: Any, column: Any, *, extra_reserve_bytes: int = 0) -> Any:
    """The (N,) m/s per-cell rate of a TERRAIN-APPLIED `weather_pet.PotentialDemand` for `column` (fresh array of the
    column namespace; the demand's owned payload is not exposed). Refuses a demand without terrain (the helper must
    not be the place where terrain is applied, and a bare scalar PET would silently drop aspect/slope), a shape,
    namespace or DEVICE mismatch (column preflight plus the payload's owning device metadata, checked before any
    copy), and a demand of the wrong type. The caller applies no aspect/slope factor afterwards.

    Cold bridge (once per date, not per step): it allocates exactly one (N,) float64 array (`pet_bridge_bytes`) and
    converts mm/day to m/s in place with `weather_pet`'s own factor; that allocation plus the column reserve and the
    caller's `extra_reserve_bytes` is admitted against the column's memory query BEFORE the copy is made."""
    from maple_syrup import weather_pet

    ss._check_column(column)
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    if not isinstance(demand, weather_pet.PotentialDemand):
        raise SubsurfaceError(f"demand must be a weather_pet.PotentialDemand, got {type(demand).__name__}")
    if not demand.terrain_applied:
        raise SubsurfaceError("the demand has no terrain applied; apply weather_pet.apply_terrain(demand, terrain) once "
                              "upstream (aspect x cos(slope)); this helper never applies terrain itself")
    terrain = demand.terrain
    if terrain.shape != (int(column.n_cells),):
        raise SubsurfaceError(f"the terrain-applied demand has shape {terrain.shape}, the column has {column.n_cells} "
                              "cells (flat cell order required)")
    if weather_pet.array_namespace(terrain.backend) is not column.xp:
        raise SubsurfaceError(f"the terrain-applied demand lives in backend {terrain.backend!r}, the column in "
                              f"{column.xp.__name__}; no implicit transfer")
    # Device binding BEFORE any copy/allocation (correction 01): the column's own preflight (current device must be
    # the column's device; a metadata check, no device switch) and the payload's OWNING device from the demand's
    # read-only metadata accessor (no probe, no launch, no host copy to discover it). Namespace equality alone only
    # establishes CuPy versus NumPy, not GPU identity.
    if column.preflight is not None:
        column.preflight()
    source_device = demand.payload_device_id
    if column.backend == "cuda":
        if source_device is None or int(source_device) != int(column.device_id):
            raise SubsurfaceError(f"the terrain-applied demand payload lives on CUDA device {source_device!r}, the column "
                                  f"is bound to device {column.device_id}; refused before any copy (no implicit "
                                  "cross-device transfer, no device switch)")
    elif source_device is not None:
        raise SubsurfaceError(f"the terrain-applied demand payload reports CUDA device {source_device!r} but the column "
                              f"is a host ({column.backend}) column; refused before any copy")
    bridge = estimate_evaporation_memory(column.n_cells, backend=column.backend, reserve_bytes=column.reserve_bytes,
                                         extra_reserve_bytes=extra)
    admitted = _as_estimate(bridge)
    admitted.required_bytes = pet_bridge_bytes(column.n_cells) + column.reserve_bytes + extra  # the bridge's own cost
    ss._admit(admitted, column.available_bytes_fn, "PET bridge copy")
    cells = demand.detached_cells()  # the ONE fresh copy, mm/day, float64, column namespace (checked above)
    if type(cells) is not column.xp.ndarray or cells.dtype != np.float64 or tuple(cells.shape) != (int(column.n_cells),):
        raise SubsurfaceError("internal: the detached PET payload is not the expected float64 (N,) array")
    column.xp.multiply(cells, 1e-3 / weather_pet.SECONDS_PER_DAY, out=cells)  # weather_pet.pet_m_s, in place
    return column.xp.ascontiguousarray(cells)
