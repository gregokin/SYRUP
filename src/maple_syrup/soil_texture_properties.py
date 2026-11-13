"""Phase 2 S2-C: PURE bounded texture -> hydraulic-property provider (Saxton & Rawls 2006), fixed profile. Parameter component only.

What this is
------------
A pure function of explicit inputs: for `L` fixed layers and `N` cells it turns mandatory sand/silt/clay fractions, organic matter (percent by weight) and an explicit bulk density into the three
hydraulic maps a later explicit soil column consumes: the field capacity at 6 kPa (`theta_field_capacity_6kpa`), the saturated water content (`theta_saturation`) and the saturated conductivity
(`saturated_conductivity_m_s`). Optional diagnostics (`theta_1500`, `air_entry_kpa`, a provenance code) are off by default. It reads and mutates NO soil water, bed, geometry, rock/base datum, clock,
checkpoint or history, performs no native adoption, and chooses no remapping, water-fate, wetting-history, averaging or Green-Ampt rule. Green-Ampt wetting-front suction, the residual content and the
interface conductivity/diffusivity stay EXPLICIT inputs of the consumer (nothing here derives `Ks/2`, `K_geom` or an interface average). The native bed class -> (sand, silt, clay, OM, density)
mapping and the packing/porosity consistency are LATER required qualification: this pure provider does not accept them. Fine-earth domain only: no gravel/coarse fraction exists (refused by construction).

Recipe (Table 1 of Saxton & Rawls, SSSAJ 70:1569-1578, doi 10.2136/sssaj2005.0117; equation numbers as in that table; parsed primary-text evidence of Eq 7-10, no visual-render or full-SPAW identity claim)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
`S, Si, C` fractions (0..1), `OM` percent by weight, `rho` = `bulk_density_kg_m3 / 1000` (g/cm3; units are explicit in the input name; the paper is in g/cm3).
  Eq 1  `theta_1500 = t + 0.14 t - 0.02`,                 `t = -0.024 S + 0.487 C + 0.006 OM + 0.005 S OM - 0.013 C OM + 0.068 S C + 0.031`
  Eq 2  `theta_33N  = t + 1.283 t^2 - 0.374 t - 0.015`,   `t = -0.251 S + 0.195 C + 0.011 OM + 0.006 S OM - 0.027 C OM + 0.452 S C + 0.299`
  Eq 3  `g = t + 0.636 t - 0.107` (theta_(S-33)),         `t = 0.278 S + 0.034 C + 0.022 OM - 0.018 S OM - 0.027 C OM - 0.584 S C + 0.078`;  `theta_SN = theta_33N + g - 0.097 S + 0.043`
  Eq 4  `psi_e = t + 0.02 t^2 - 0.113 t - 0.70` (kPa),    `t = -21.67 S - 27.93 C - 81.97 g + 71.12 S g + 8.29 C g + 14.05 S C + 27.16`
  Eq 6  `rho_N = (1 - theta_SN) 2.65`;   density factor `DF = rho / rho_N` (computed, never a second mandatory caller input)
  Eq 7-10 `theta_S = 1 - rho/2.65`,  `theta_33 = theta_33N - 0.2 (theta_SN - theta_S)`  (no adjustment of theta_1500)
  Eq 11-15 retention: `B = (ln 1500 - ln 33) / (ln theta_33 - ln theta_1500)`, `lambda = 1/B`; Eq 12 above the air entry:
           `theta(psi) = theta_33 + (33 - psi)(theta_S - theta_33)/(33 - psi_e)`.
  Eq 16 `Ks[mm/h] = 1930 (theta_S - theta_33)^(3 - lambda)`;   `Ks[m/s] = Ks[mm/h] / 3 600 000`.
  Field capacity (Tietjen/agreed): `theta_6 = theta_S` when `psi_e >= 6 kPa` (provenance bit `fc_equals_saturation`, a VALID outcome, never a Ks = 0), else
           `theta_6 = theta_33 + 27 (theta_S - theta_33)/(33 - psi_e)`. `theta_33` (NOT `theta_6`) anchors Eq 16: for a fine soil `theta_6 == theta_S` would give Ks = 0.
Declared recipe assumption (not a full-SPAW identity claim): the air entry `psi_e` is evaluated from the NORMAL-density Eq 3 input (literal Eq 4) and held fixed under density adjustment.

Domains: REFUSED, never clamped
-------------------------------
Every refusal is a typed `TexturePropertyError` BEFORE any output exists (nothing is published). Per element (layer, cell), input checks first (bits 1..32), derived checks only for elements that passed them:
  1 non-finite mandatory input / infinite override | 2 a fraction outside [0,1] or `|(S+Si)+C-1| > summation_error_bound_kg(3, 1)` (MAPLE's own FP64 sum contract, 4*3*eps: the existing derived bound,
  nothing user-adjustable) | 4 clay > 0.6 | 8 OM outside [0, 8] % | 16 bulk density outside [1000, 1800] kg/m3 | 32 measured Ks not > 0 |
  64 computed DF outside [0.9, 1.3] | 128 final anchors not 0 < theta_1500 < theta_33 < theta_S < 1 | 256 final air entry not in [0, 33) kPa (exact 0 is a valid continuous curve; negative,
  non-finite or >= 33 is an unsupported prediction, e.g. the published sand example gives about -0.964 kPa) | 512 final `theta_S - theta_33 < 0.005` | 1024 non-finite or non-positive Ks.
The DF/density/texture domain checks always apply to the explicit inputs (the density is mandatory); measured anchors replace the FINAL values only (below).

Optional MEASURED overrides (NaN in an override array = "not measured here"; no other sentinel, +-inf is refused)
-----------------------------------------------------------------------------------------------------------------
`theta_1500_override`, `theta_33_override`, `theta_saturation_override`, `air_entry_kpa_override` replace the corresponding FINAL anchor (not density-adjusted afterwards); B, lambda, Ks and `theta_6` are
recomputed from the final set, which must pass the same checks 128-512 (so a valid measured `psi_e >= 0` can rescue a sand cell but a bare field capacity cannot: a curve is claimed). A measured
`saturated_conductivity_override_m_s` (> 0) replaces Eq 16 only. A measured/prescribed field capacity or conductivity WITHOUT a claimed curve already has the existing explicit path
(`spatial_layer_spec` / `soil_layers`); no second prescribed-mode engine exists here. No invalid curve is ever returned as valid.

Layouts, outputs, backends (A1 conventions)
-------------------------------------------
Every input is a real host scalar (SCALAR), an exactly `(L,)` float64 C-contiguous array (SHARED) or an exactly `(L, N)` array (CELL) in the backend's namespace (NumPy for "numpy"/"numba", CuPy for
"cuda"); there is no implicit conversion or transfer. All outputs share ONE layout, the join of the layouts of the supplied inputs: CELL `(L, N)` if any input is CELL, otherwise SHARED `(L,)` (all-scalar
inputs give `(L,)`, the smallest layout the A1 array interface accepts; shared-only inputs cost O(L) and never expand to N). "numpy" = the reference and the DEFINITION (vectorised over cell chunks);
"numba" = a compiled serial loop (`soil_texture_properties_numba`); "cuda" = one RawKernel, one thread per cell looping the layers (`soil_texture_properties_cuda`; compiled with
`--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`). The compiled bodies evaluate the same operations in the same order from the SAME coefficient tuples; bitwise agreement is claimed
only for the outputs without a transcendental (`theta_saturation`, `theta_field_capacity_6kpa`, `theta_1500`, `air_entry_kpa`) where the tests record it; `saturated_conductivity_m_s` uses `log`/`pow` library calls
and is compared at the declared relative bound 2e-12 (absolute 1e-14). No claim about the CUDA device is made by the author (no GPU was run).

Identity, memory, ownership
---------------------------
`sha256` hashes the schema, the recipe constants, the options and every supplied input (name, layout, shape, then bytes row by row in C order, streamed in `hash_chunk_bytes` pieces; CUDA copies each piece
through ONE host buffer): it identifies the explicit INPUTS and the recipe, is chunk-size invariant, and does not depend on CPU/GPU roundoff in the outputs. `source_digests()` records the SHA-256 of the three
module files. Memory (`estimate_property_memory`) is a MODEL counted before any allocation, host and device separately, against caller quotas and injectable truthful availability readings; the caller's input arrays
are excluded (reported as `input_bytes`); NVRTC/Numba compilation, the CUDA context, the allocator pool policy and whole-process memory are excluded; no library (CUB) workspace exists. Outputs are owned
by the result: read-only NumPy arrays (`array(name)`), or private CuPy arrays (`detached(name, ...)` hands out an admitted copy). The caller's inputs are never modified or retained.

This is an equation/numerical verification target, NOT field validation of the regressions (published R^2: theta_33 0.63, theta_(S-33) 0.36, psi_e 0.78).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss

__all__ = [
    "BACKENDS",
    "COMPONENT_TAG",
    "FLAG_TEXT",
    "SCHEMA",
    "SoilTextureProperties",
    "TexturePropertyBackendUnavailableError",
    "TexturePropertyError",
    "TexturePropertyMemoryError",
    "TextureMemoryEstimate",
    "estimate_property_memory",
    "estimate_soil_texture_properties",
    "recipe_record",
    "reference_block",
    "source_digests",
]

SCHEMA = "maple_syrup.soil_texture_properties/1"
COMPONENT_TAG = "texture_property_provider_pure"
RECIPE_TAG = "saxton_rawls_2006_table1_eq1_18_explicit_density_v1"
BACKENDS = ss.BACKENDS
LAYOUT_SCALAR, LAYOUT_SHARED, LAYOUT_CELL = sp.LAYOUT_SCALAR, sp.LAYOUT_SHARED, sp.LAYOUT_CELL

# ---- recipe constants: the ONE source of every coefficient (the Numba and CUDA bodies read these same tuples) ------------------------------------------------------------------------------------
EQ1_REGRESSION = (-0.024, 0.487, 0.006, 0.005, -0.013, 0.068, 0.031)  # S, C, OM, S*OM, C*OM, S*C, constant
EQ1_ADJUST = (0.14, -0.02)
EQ2_REGRESSION = (-0.251, 0.195, 0.011, 0.006, -0.027, 0.452, 0.299)
EQ2_ADJUST = (1.283, -0.374, -0.015)
EQ3_REGRESSION = (0.278, 0.034, 0.022, -0.018, -0.027, -0.584, 0.078)
EQ3_ADJUST = (0.636, -0.107)
EQ3_SAT_TERMS = (-0.097, 0.043)
EQ4_REGRESSION = (-21.67, -27.93, -81.97, 71.12, 8.29, 14.05, 27.16)  # S, C, g, S*g, C*g, S*C, constant
EQ4_ADJUST = (0.02, -0.113, -0.70)
PARTICLE_DENSITY_G_CM3 = 2.65
DF_AIR_COEFF = 0.2
DENSITY_FACTOR_RANGE = (0.9, 1.3)
DENSITY_RANGE_G_CM3 = (1.0, 1.8)
CLAY_MAX, OM_MAX_PERCENT = 0.6, 8.0
MIN_DENSITY_GAP = 0.005
AIR_ENTRY_MAX_KPA = 33.0
FC_PRESSURE_KPA = 6.0
FC_SPAN_KPA = 27.0  # 33 - 6 kPa
LN_RATIO = math.log(1500.0) - math.log(33.0)
KS_COEFF_MM_H, MM_H_PER_M_S = 1930.0, 3600000.0
FRACTION_CLOSURE_TERMS = 3

INPUTS = ("sand_fraction", "silt_fraction", "clay_fraction", "organic_matter_percent", "bulk_density_kg_m3")
OVERRIDES = ("theta_1500_override", "theta_33_override", "theta_saturation_override", "air_entry_kpa_override", "saturated_conductivity_override_m_s")
ALL_INPUTS = INPUTS + OVERRIDES
OUTPUTS = ("theta_field_capacity_6kpa", "theta_saturation", "saturated_conductivity_m_s")
DIAGNOSTICS = ("theta_1500", "air_entry_kpa", "provenance")

FLAG_NONFINITE, FLAG_FRACTION, FLAG_CLAY, FLAG_ORGANIC_MATTER, FLAG_DENSITY, FLAG_KS_OVERRIDE = 1, 2, 4, 8, 16, 32
FLAG_DENSITY_FACTOR, FLAG_CURVE, FLAG_AIR_ENTRY, FLAG_DENSITY_GAP, FLAG_KS = 64, 128, 256, 512, 1024
INPUT_FLAGS = FLAG_NONFINITE | FLAG_FRACTION | FLAG_CLAY | FLAG_ORGANIC_MATTER | FLAG_DENSITY | FLAG_KS_OVERRIDE
FLAG_TEXT = {
    FLAG_NONFINITE: "a mandatory input is not finite or an override is infinite (NaN in an override means 'not measured')",
    FLAG_FRACTION: "sand/silt/clay fractions are outside [0, 1] or do not close to 1 within the declared FP64 bound",
    FLAG_CLAY: f"clay fraction exceeds {CLAY_MAX} (outside the supported domain)",
    FLAG_ORGANIC_MATTER: f"organic matter is outside [0, {OM_MAX_PERCENT}] percent by weight",
    FLAG_DENSITY: f"bulk density is outside [{DENSITY_RANGE_G_CM3[0] * 1000:g}, {DENSITY_RANGE_G_CM3[1] * 1000:g}] kg/m3",
    FLAG_KS_OVERRIDE: "a measured saturated conductivity is not > 0",
    FLAG_DENSITY_FACTOR: f"the computed density factor rho/rho_normal is outside [{DENSITY_FACTOR_RANGE[0]}, {DENSITY_FACTOR_RANGE[1]}]",
    FLAG_CURVE: "the final retention anchors do not satisfy 0 < theta_1500 < theta_33 < theta_saturation < 1",
    FLAG_AIR_ENTRY: f"the final air entry is not in [0, {AIR_ENTRY_MAX_KPA:g}) kPa (unsupported prediction; supply a valid measured air entry)",
    FLAG_DENSITY_GAP: f"theta_saturation - theta_33 is below {MIN_DENSITY_GAP}",
    FLAG_KS: "the saturated conductivity is not finite and > 0",
}
PROV_THETA_1500, PROV_THETA_33, PROV_THETA_SAT, PROV_AIR, PROV_KS, PROV_FC_EQ_SAT = 1, 2, 4, 8, 16, 32
DIAG_WORDS = 4  # flag OR, flagged-element count, min(element << 16 | flags), count of fc == saturation outcomes
PACKET_BYTES = DIAG_WORDS * 8
MAX_CUDA_ELEMENTS = 2**47 - 1  # (element << 16 | flags) fits 64 bits
CUDA_BLOCK = 128
MAX_CUDA_GRID_CELLS = (2**31 - 1) * CUDA_BLOCK  # the launch grid limit of the one-thread-per-cell kernel
COMPILE_OPTIONS = sp.COMPILE_OPTIONS
DEFAULT_CHUNK_COLUMNS = 65536
DEFAULT_HASH_CHUNK_BYTES = 1 << 20
REF_VEC_F64, REF_VEC_BOOL = 40, 24  # live (chunk,) work vectors of `reference_block` (upper bounds, checked against tracemalloc by the tests)
PY_ALLOWANCE_BYTES = 65536
_SOURCE_FILES = ("soil_texture_properties.py", "soil_texture_properties_numba.py", "soil_texture_properties_cuda.py")


class TexturePropertyError(ss.SubsurfaceError):
    """Invalid explicit input (metadata or value) or an unsupported prediction. Nothing was published."""

    def __init__(self, message: str, *, bit: int | None = None, n_flagged: int = 0, first_layer: int | None = None, first_cell: int | None = None, flags: int = 0):
        super().__init__(message)
        self.bit, self.n_flagged, self.first_layer, self.first_cell, self.flags = bit, int(n_flagged), first_layer, first_cell, int(flags)


class TexturePropertyMemoryError(ss.SubsurfaceMemoryError):
    """The modelled requirement plus reserve exceeds the declared quota or the reported availability of one side. Nothing was allocated."""


class TexturePropertyBackendUnavailableError(ss.SubsurfaceBackendUnavailableError):
    """The requested backend (Numba, CuPy/CUDA) or a memory reading is unavailable. There is no fallback."""


def _count(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise TexturePropertyError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def recipe_record() -> dict[str, Any]:
    """The immutable constants/units/assumptions of the recipe (what the identity hashes)."""
    return {"recipe": RECIPE_TAG, "source": "Saxton & Rawls 2006 SSSAJ 70:1569, Table 1 Eq 1-16, 18 (doi 10.2136/sssaj2005.0117)", "units": {"fractions": "0..1", "organic_matter": "percent by weight",
            "bulk_density_input": "kg/m3 (divided by 1000 to the paper's g/cm3 inside)", "ks_out": "m/s (mm/h / 3600000)", "air_entry": "kPa"},
            "eq1": [EQ1_REGRESSION, EQ1_ADJUST], "eq2": [EQ2_REGRESSION, EQ2_ADJUST], "eq3": [EQ3_REGRESSION, EQ3_ADJUST, EQ3_SAT_TERMS], "eq4": [EQ4_REGRESSION, EQ4_ADJUST],
            "particle_density_g_cm3": PARTICLE_DENSITY_G_CM3, "df_air_coefficient": DF_AIR_COEFF, "density_factor_range": DENSITY_FACTOR_RANGE, "density_range_g_cm3": DENSITY_RANGE_G_CM3,
            "clay_max": CLAY_MAX, "om_max_percent": OM_MAX_PERCENT, "min_density_gap": MIN_DENSITY_GAP, "air_entry_max_kpa": AIR_ENTRY_MAX_KPA, "fc_pressure_kpa": FC_PRESSURE_KPA, "fc_span_kpa": FC_SPAN_KPA,
            "ln_ratio_repr": repr(LN_RATIO), "ks_coefficient_mm_h": KS_COEFF_MM_H, "ks_anchor": "theta_33 (not theta_6)",
            "assumptions": ["air entry from the normal-density Eq3 input, held fixed under density adjustment", "fine earth only; no gravel/coarse", "overrides replace FINAL anchors", "no clamping: unsupported predictions are refused"]}


def closure_bound() -> float:
    from maple.surface.voxels._numerics import summation_error_bound_kg

    return float(summation_error_bound_kg(FRACTION_CLOSURE_TERMS, 1.0))


@dataclass(frozen=True, eq=False)
class _Input:
    name: str
    kind: str  # scalar | shared | cell
    array: Any  # the CALLER's array (no copy) or None for a host scalar
    scalar: float | None
    nbytes: int


def _namespace_type(backend: str):
    if backend != "cuda":
        return np.ndarray
    try:
        from maple.core.backend import cupy_import_error, cupy_module

        cp = cupy_module()
        reason = None if cp is not None else (cupy_import_error() or "not installed")
    except Exception as exc:  # noqa: BLE001
        raise TexturePropertyBackendUnavailableError(f"backend 'cuda' needs CuPy and it could not be probed ({type(exc).__name__}: {exc}); there is no fallback") from exc
    if cp is None:
        raise TexturePropertyBackendUnavailableError(f"backend 'cuda' needs CuPy and it is not usable ({reason}); there is no fallback")
    return cp.ndarray


def _classify(name: str, value: Any, L: int, N: int, ns_type: Any, backend: str) -> _Input:
    if sp._is_real_scalar(value):
        return _Input(name, LAYOUT_SCALAR, None, float(value), 0)
    if isinstance(value, (bool, np.bool_)):
        raise TexturePropertyError(f"{name} must be numeric, got a Boolean")
    if type(value) is not ns_type:
        other = type(value).__module__.split(".")[0] in ("numpy", "cupy")
        raise TexturePropertyError(f"{name} must be exactly a {'cupy' if backend == 'cuda' else 'numpy'}.ndarray (backend {backend!r}; no subclass, list or masked array, no implicit transfer)"
                                   f"{' -- it lives in another namespace' if other else ''}, got {type(value).__module__}.{type(value).__name__}")
    if value.dtype != np.float64:
        raise TexturePropertyError(f"{name} must be float64, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise TexturePropertyError(f"{name} must be C-contiguous")
    shape = tuple(int(n) for n in value.shape)
    if shape == (L,):
        return _Input(name, LAYOUT_SHARED, value, None, int(value.nbytes))
    if shape == (L, N):
        return _Input(name, LAYOUT_CELL, value, None, int(value.nbytes))
    raise TexturePropertyError(f"{name} must be a real scalar, shape {(L,)} (common) or {(L, N)} (per cell), got {shape}")


def _join(inputs: dict) -> str:
    kinds = {i.kind for i in inputs.values()}
    return LAYOUT_CELL if LAYOUT_CELL in kinds else LAYOUT_SHARED


def _rows_cells(L: int, N: int, layout: str) -> tuple[int, int]:
    """Output rows and the cells actually computed (a shared join is independent of N)."""
    return L, (N if layout == LAYOUT_CELL else 1)


# ---- memory model --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TextureMemoryEstimate:
    backend: str
    n_layers: int
    n_cells: int
    layouts: Any
    output_layout: str
    outputs: Any  # name -> bytes
    output_bytes: int
    input_bytes: Any  # the caller's arrays (actual bytes), EXCLUDED from the requirement
    chunk_columns: int
    hash_chunk_bytes: int
    main_required_bytes: int  # host for numpy/numba, DEVICE for cuda (includes the main reserve)
    host_required_bytes: int
    device_required_bytes: int
    reserve_bytes: int
    host_reserve_bytes: int
    terms: Any
    scope: str = ("modelled bytes of this call's own arrays only; the caller's inputs, interpreter/JIT/NVRTC/driver state, the allocator pool policy, the deep identity files and whole-process memory are "
                  "excluded; an availability reading is not a reservation")

    def __post_init__(self) -> None:
        for name in ("layouts", "outputs", "input_bytes", "terms"):
            object.__setattr__(self, name, sp._freeze(getattr(self, name)))


def estimate_property_memory(n_layers: int, n_cells: int, *, layouts: dict, backend: str = "numpy", diagnostics: bool = False, chunk_columns: int = DEFAULT_CHUNK_COLUMNS,
                             hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES, reserve_bytes: int = 0, host_reserve_bytes: int = 0, input_bytes: dict | None = None) -> TextureMemoryEstimate:
    """Metadata-only requirement of one `estimate_soil_texture_properties` call. `layouts` maps every SUPPLIED input name to "scalar" | "shared" | "cell" (an absent override is simply not a key)."""
    L, N = _count(n_layers, "n_layers", minimum=1), _count(n_cells, "n_cells", minimum=1)
    if backend not in BACKENDS:
        raise TexturePropertyError(f"backend must be one of {BACKENDS}, got {backend!r}")
    chunk, hc = _count(chunk_columns, "chunk_columns", minimum=1), _count(hash_chunk_bytes, "hash_chunk_bytes", minimum=8)
    reserve, host_reserve = _count(reserve_bytes, "reserve_bytes"), _count(host_reserve_bytes, "host_reserve_bytes")
    if host_reserve and backend != "cuda":
        raise TexturePropertyError("host_reserve_bytes applies only to the cuda backend (the CPU backends use reserve_bytes on the host)")
    for key in INPUTS:
        if key not in layouts:
            raise TexturePropertyError(f"layouts[{key!r}] is mandatory")
    unknown = set(layouts) - set(ALL_INPUTS)
    if unknown:
        raise TexturePropertyError(f"unknown input layout name(s) {sorted(unknown)}")
    for key, kind in layouts.items():  # every SUPPLIED layout (mandatory and optional overrides) must be a known kind
        if kind not in (LAYOUT_SCALAR, LAYOUT_SHARED, LAYOUT_CELL):
            raise TexturePropertyError(f"layouts[{key!r}] must be 'scalar', 'shared' or 'cell', got {kind!r}")
    kinds = set(layouts.values())
    out_layout = LAYOUT_CELL if LAYOUT_CELL in kinds else LAYOUT_SHARED
    if backend == "cuda":
        launched = N if out_layout == LAYOUT_CELL else 1  # a shared-only call launches one thread: its logical N is never expanded
        if launched > MAX_CUDA_GRID_CELLS:
            raise TexturePropertyError(f"the cuda backend launches at most {MAX_CUDA_GRID_CELLS} cells per call ((2**31 - 1) blocks of {CUDA_BLOCK} threads), got {launched}")
        if L * launched > MAX_CUDA_ELEMENTS:
            raise TexturePropertyError(f"the cuda backend supports at most {MAX_CUDA_ELEMENTS} elements per call (64-bit packed diagnostic), got {L * launched}")
    rows, cells = _rows_cells(L, N, out_layout)
    per = rows * cells
    outputs = {k: 8 * per for k in OUTPUTS}
    if diagnostics:
        outputs.update({"theta_1500": 8 * per, "air_entry_kpa": 8 * per, "provenance": per})
    output_bytes = sum(outputs.values())
    ch = min(chunk, cells)
    ref_work = REF_VEC_F64 * 8 * ch + REF_VEC_BOOL * ch if backend == "numpy" else 0
    hash_buf = hc if backend == "cuda" else 0
    packet = PACKET_BYTES if backend == "cuda" else 0
    aux = 8 * len(ALL_INPUTS) if backend == "cuda" else 0  # one device array holding the scalar inputs (kernel args are views of it)
    terms = {"outputs": output_bytes, "numpy_chunk_work_vectors": ref_work, "cuda_host_identity_buffer": hash_buf, "cuda_packet": packet, "cuda_scalar_aux": aux, "python_allowance": PY_ALLOWANCE_BYTES,
             "no_workspace": "no library reduction/CUB workspace exists", "excluded": "NVRTC/Numba compilation, CUDA context, allocator pool policy, whole-process memory"}
    if backend == "cuda":
        device, host = output_bytes + PACKET_BYTES + aux + reserve, hash_buf + packet + PY_ALLOWANCE_BYTES + host_reserve
        main = device
    else:
        host = output_bytes + ref_work + PY_ALLOWANCE_BYTES + reserve
        device, main = 0, host
    return TextureMemoryEstimate(backend, L, N, dict(layouts), out_layout, outputs, output_bytes, dict(input_bytes or {}), ch, hc, main, host, device, reserve, host_reserve, terms)


def _admit(required: int, budget: int | None, reader: Callable[[], int] | None, side: str, est: TextureMemoryEstimate, quota_name: str) -> None:
    if budget is not None and required > budget:
        raise TexturePropertyMemoryError(f"texture property provider ({est.n_layers} layers x {est.n_cells} cells, {est.backend}) needs an estimated {required} B on the {side} side but the "
                                         f"caller's declared {quota_name} is {budget}; nothing was allocated. Scope: {est.scope}")
    if reader is None:
        if side == "device":
            from maple_syrup import soil_water_remap_cuda as rc

            reader = rc.default_device_available_bytes_fn(rc.require_cuda())
        else:
            reader = sp.default_host_available_bytes
    available = reader()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise TexturePropertyError(f"the {side} available-bytes reader must return a non-Boolean integer >= 0, got {available!r}")
    if required > int(available):
        raise TexturePropertyMemoryError(f"texture property provider ({est.n_layers} layers x {est.n_cells} cells, {est.backend}) needs an estimated {required} B on the {side} side but only "
                                         f"{int(available)} B are reported available; nothing was allocated. Scope: {est.scope}")


# ---- the NumPy reference (the definition of the arithmetic) -------------------------------------------------------------------------------------------------------------------------------------------
def _poly(c: tuple, S: Any, C: Any, OM: Any) -> Any:
    """`c0 S + c1 C + c2 OM + c3 S OM + c4 C OM + c5 S C + c6`, summed left to right in this exact order (Eq 1-3)."""
    return c[0] * S + c[1] * C + c[2] * OM + c[3] * S * OM + c[4] * C * OM + c[5] * S * C + c[6]


def reference_block(S: Any, Si: Any, C: Any, OM: Any, rho_kg: Any, o_wp: Any, o_33: Any, o_sat: Any, o_air: Any, o_ks: Any, closure: float):
    """The recipe on `(m,)` float64 arrays (every argument already broadcast). Returns `(flags uint16, fc6, sat, ks, wp, air, provenance uint8, fc_equals_saturation bool)`; values of flagged elements are
    meaningless. Pure: allocates only work vectors, writes nothing it was given."""
    with np.errstate(all="ignore"):
        fS, fSi, fC, fOM, fR = np.isfinite(S), np.isfinite(Si), np.isfinite(C), np.isfinite(OM), np.isfinite(rho_kg)
        inf_ov = np.isinf(o_wp) | np.isinf(o_33) | np.isinf(o_sat) | np.isinf(o_air) | np.isinf(o_ks)
        flags = ((~(fS & fSi & fC & fOM & fR)) | inf_ov).astype(np.uint16) * np.uint16(FLAG_NONFINITE)
        ffr = fS & fSi & fC
        in_range = (S >= 0.0) & (S <= 1.0) & (Si >= 0.0) & (Si <= 1.0) & (C >= 0.0) & (C <= 1.0)
        flags |= (ffr & (~in_range | (np.abs(((S + Si) + C) - 1.0) > closure))).astype(np.uint16) * np.uint16(FLAG_FRACTION)
        flags |= (fC & (C > CLAY_MAX)).astype(np.uint16) * np.uint16(FLAG_CLAY)
        flags |= (fOM & ((OM < 0.0) | (OM > OM_MAX_PERCENT))).astype(np.uint16) * np.uint16(FLAG_ORGANIC_MATTER)
        rho = rho_kg / 1000.0
        flags |= (fR & ((rho < DENSITY_RANGE_G_CM3[0]) | (rho > DENSITY_RANGE_G_CM3[1]))).astype(np.uint16) * np.uint16(FLAG_DENSITY)
        has_wp, has_33, has_sat, has_air, has_ks = ~np.isnan(o_wp), ~np.isnan(o_33), ~np.isnan(o_sat), ~np.isnan(o_air), ~np.isnan(o_ks)
        flags |= (has_ks & (o_ks <= 0.0)).astype(np.uint16) * np.uint16(FLAG_KS_OVERRIDE)
        clean = (flags & np.uint16(INPUT_FLAGS)) == 0
        # normal-density estimates (Eq 1-4, 6)
        t = _poly(EQ1_REGRESSION, S, C, OM)
        wp_n = t + EQ1_ADJUST[0] * t + EQ1_ADJUST[1]
        t = _poly(EQ2_REGRESSION, S, C, OM)
        fc_n = t + EQ2_ADJUST[0] * t * t + EQ2_ADJUST[1] * t + EQ2_ADJUST[2]
        t = _poly(EQ3_REGRESSION, S, C, OM)
        g = t + EQ3_ADJUST[0] * t + EQ3_ADJUST[1]
        sat_n = fc_n + g + EQ3_SAT_TERMS[0] * S + EQ3_SAT_TERMS[1]
        t = EQ4_REGRESSION[0] * S + EQ4_REGRESSION[1] * C + EQ4_REGRESSION[2] * g + EQ4_REGRESSION[3] * S * g + EQ4_REGRESSION[4] * C * g + EQ4_REGRESSION[5] * S * C + EQ4_REGRESSION[6]
        air_n = t + EQ4_ADJUST[0] * t * t + EQ4_ADJUST[1] * t + EQ4_ADJUST[2]
        rho_n = (1.0 - sat_n) * PARTICLE_DENSITY_G_CM3
        df = rho / rho_n
        # density adjustment (Eq 7-10) and the FINAL anchors (measured values replace, never re-adjusted)
        sat_d = 1.0 - rho / PARTICLE_DENSITY_G_CM3
        fc_d = fc_n - DF_AIR_COEFF * (sat_n - sat_d)
        wp = np.where(has_wp, o_wp, wp_n)
        fc33 = np.where(has_33, o_33, fc_d)
        sat = np.where(has_sat, o_sat, sat_d)
        air = np.where(has_air, o_air, air_n)
        gap = sat - fc33
        d_bad = ~((df >= DENSITY_FACTOR_RANGE[0]) & (df <= DENSITY_FACTOR_RANGE[1]))
        c_bad = ~((wp > 0.0) & (wp < fc33) & (fc33 < sat) & (sat < 1.0))
        a_bad = ~((air >= 0.0) & (air < AIR_ENTRY_MAX_KPA))
        g_bad = ~(gap >= MIN_DENSITY_GAP)
        dflags = (d_bad.astype(np.uint16) * np.uint16(FLAG_DENSITY_FACTOR) | c_bad.astype(np.uint16) * np.uint16(FLAG_CURVE) | a_bad.astype(np.uint16) * np.uint16(FLAG_AIR_ENTRY)
                  | g_bad.astype(np.uint16) * np.uint16(FLAG_DENSITY_GAP))
        flags |= np.where(clean, dflags, np.uint16(0))
        curve_ok = clean & ~(d_bad | c_bad | a_bad | g_bad)
        B = LN_RATIO / (np.log(fc33) - np.log(wp))
        lam = 1.0 / B
        ks_est = (KS_COEFF_MM_H * np.power(gap, 3.0 - lam)) / MM_H_PER_M_S
        ks = np.where(has_ks, o_ks, ks_est)
        flags |= (curve_ok & ~(np.isfinite(ks) & (ks > 0.0))).astype(np.uint16) * np.uint16(FLAG_KS)
        fc_eq_sat = FC_PRESSURE_KPA <= air
        fc6 = np.where(fc_eq_sat, sat, fc33 + FC_SPAN_KPA * gap / (AIR_ENTRY_MAX_KPA - air))
        prov = (has_wp.astype(np.uint8) * np.uint8(PROV_THETA_1500) | has_33.astype(np.uint8) * np.uint8(PROV_THETA_33) | has_sat.astype(np.uint8) * np.uint8(PROV_THETA_SAT)
                | has_air.astype(np.uint8) * np.uint8(PROV_AIR) | has_ks.astype(np.uint8) * np.uint8(PROV_KS) | fc_eq_sat.astype(np.uint8) * np.uint8(PROV_FC_EQ_SAT))
        return flags, fc6, sat, ks, wp, air, prov, fc_eq_sat


def _row(inp: _Input | None, l: int, s: int, e: int, absent: float) -> Any:
    m = e - s
    if inp is None:
        return np.broadcast_to(np.float64(absent), (m,))
    if inp.kind == LAYOUT_SCALAR:
        return np.broadcast_to(np.float64(inp.scalar), (m,))
    if inp.kind == LAYOUT_SHARED:
        return np.broadcast_to(inp.array[l], (m,))
    return inp.array[l, s:e]


def _derive_numpy(inputs: dict, out: dict, L: int, cells: int, chunk: int, closure: float) -> tuple[int, int, int, int]:
    """Vectorised reference over (layer, cell-chunk). Returns the host diagnostic words (flag OR, n flagged, min packed element|flags (all ones if none), fc==saturation count)."""
    ov, n_flagged, first, n_eq = 0, 0, 0xFFFFFFFFFFFFFFFF, 0
    cell_out = out["theta_saturation"].ndim == 2
    for l in range(L):
        for s in range(0, cells, chunk):
            e = min(cells, s + chunk)
            vals = [_row(inputs.get(k), l, s, e, 0.0 if k in INPUTS else float("nan")) for k in ALL_INPUTS]
            flags, fc6, sat, ks, wp, air, prov, eq = reference_block(*vals, closure)
            bad = np.flatnonzero(flags)
            if bad.size:
                n_flagged += int(bad.size)
                ov |= int(np.bitwise_or.reduce(flags[bad]))
                idx = bad[0]
                first = min(first, ((l * cells + s + int(idx)) << 16) | int(flags[idx]))
                continue
            n_eq += int(np.count_nonzero(eq))
            sl = (l, slice(s, e)) if cell_out else (slice(l, l + 1),)
            out["theta_field_capacity_6kpa"][sl], out["theta_saturation"][sl], out["saturated_conductivity_m_s"][sl] = fc6, sat, ks
            if "theta_1500" in out:
                out["theta_1500"][sl], out["air_entry_kpa"][sl], out["provenance"][sl] = wp, air, prov
    return ov, n_flagged, first, n_eq


# ---- identity --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _hash_array(h: Any, name: str, kind: str, array: Any, host_buf: np.ndarray | None) -> None:
    shape = tuple(int(n) for n in array.shape)
    h.update(f"{name}|{kind}|{shape}|float64\0".encode())
    rows = shape[0] if len(shape) > 1 else 1
    flat = array.reshape(rows, -1)
    width = flat.shape[1]
    step = max(1, width if host_buf is None else host_buf.size)
    for row in range(rows):
        for a in range(0, width, step):
            piece = flat[row, a:a + step]
            if host_buf is None:
                h.update(memoryview(piece))
            else:
                n = int(piece.shape[0])
                piece.get(out=host_buf[:n])
                h.update(memoryview(host_buf[:n]))


def _identity(inputs: dict, L: int, N: int, diagnostics: bool, host_buf: np.ndarray | None) -> str:
    h = hashlib.sha256()
    h.update(f"{SCHEMA}\0{json.dumps(recipe_record(), sort_keys=True)}\0diagnostics={diagnostics};L={L};N={N}\0".encode())
    for key in ALL_INPUTS:
        i = inputs.get(key)
        if i is None:
            h.update(f"{key}|absent\0".encode())
        elif i.kind == LAYOUT_SCALAR:
            h.update(f"{key}|scalar|{i.scalar!r}\0".encode())
        else:
            _hash_array(h, key, i.kind, i.array, host_buf)
    return h.hexdigest()


_SOURCE_DIGESTS: dict[str, str] | None = None


def source_digests() -> dict[str, str]:
    """SHA-256 of the three module files of this component (a descriptive record; cached, streamed)."""
    global _SOURCE_DIGESTS
    if _SOURCE_DIGESTS is None:
        here = Path(__file__).resolve().parent
        out = {}
        for name in _SOURCE_FILES:
            h = hashlib.sha256()
            with open(here / name, "rb") as fh:
                for piece in iter(lambda: fh.read(8192), b""):
                    h.update(piece)
            out[name] = h.hexdigest()
        _SOURCE_DIGESTS = out
    return dict(_SOURCE_DIGESTS)


# ---- the result owner --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
_CONSTRUCT = threading.local()


@contextlib.contextmanager
def _constructing():
    _CONSTRUCT.allowed = True
    try:
        yield
    finally:
        _CONSTRUCT.allowed = False


@dataclass(frozen=True, eq=False)
class SoilTextureProperties:
    """Sealed owner of the output maps of ONE provider call. `component_tag = 'texture_property_provider_pure'`. Created only by `estimate_soil_texture_properties`."""

    backend: str
    n_layers: int
    n_cells: int
    layout: str
    layouts: Any  # supplied input name -> layout kind
    sha256: str
    source_digests: Any
    estimate: TextureMemoryEstimate
    n_fc_equals_saturation: int
    device_id: int | None
    _maps: Any
    component_tag: str = COMPONENT_TAG

    def __post_init__(self) -> None:
        if not getattr(_CONSTRUCT, "allowed", False):
            raise TexturePropertyError("a SoilTextureProperties is created only by estimate_soil_texture_properties (no direct construction, replace() or copy)")
        object.__setattr__(self, "layouts", sp._freeze(dict(self.layouts)))
        object.__setattr__(self, "source_digests", sp._freeze(dict(self.source_digests)))
        object.__setattr__(self, "_maps", MappingProxyType(dict(self._maps)))

    def __copy__(self):
        raise TexturePropertyError("a SoilTextureProperties owns private maps and is not copyable; call the provider again")

    def __deepcopy__(self, memo):
        raise TexturePropertyError("a SoilTextureProperties owns private maps and is not copyable; call the provider again")

    def __reduce_ex__(self, protocol):
        raise TexturePropertyError("a SoilTextureProperties is not serialisable; call the provider again")

    def names(self) -> tuple[str, ...]:
        return tuple(self._maps)

    @property
    def output_bytes(self) -> int:
        return self.estimate.output_bytes

    def array(self, name: str) -> np.ndarray:
        """The OWNED read-only NumPy output (no copy). A CUDA result owns private device arrays: use `detached`."""
        if name not in self._maps:
            raise TexturePropertyError(f"unknown output {name!r}; available: {tuple(self._maps)}")
        if self.backend == "cuda":
            raise TexturePropertyError("a cuda result owns private device arrays; use detached(name, ...) for an admitted copy")
        return self._maps[name]

    def detached(self, name: str, *, memory_budget_bytes: Any, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None) -> Any:
        """A fresh writeable copy of one output in the result's own namespace (host for numpy/numba, device for cuda), admitted against the quota/reading BEFORE it is made."""
        if name not in self._maps:
            raise TexturePropertyError(f"unknown output {name!r}; available: {tuple(self._maps)}")
        arr = self._maps[name]
        if self.backend == "cuda":  # a copy targets the CURRENT device: bind it to the owner's device before any reader or copy
            from maple_syrup import soil_water_remap_cuda as rc

            current = rc.current_device_id(rc.require_cuda())
            if current != self.device_id:
                raise TexturePropertyError(f"this result lives on device {self.device_id} but the current device is {current}; detached() never copies across devices (nothing was copied)")
        need = int(arr.nbytes) + _count(reserve_bytes, "reserve_bytes")
        budget = _count(memory_budget_bytes, "memory_budget_bytes")
        side = "device" if self.backend == "cuda" else "host"
        if need > budget:
            raise TexturePropertyMemoryError(f"detaching {name!r} needs {need} B on the {side} side but the declared memory_budget_bytes is {budget}; nothing was copied")
        reader = available_bytes_fn
        if reader is None:
            if side == "device":
                from maple_syrup import soil_water_remap_cuda as rc

                reader = rc.default_device_available_bytes_fn(rc.require_cuda())
            else:
                reader = sp.default_host_available_bytes
        available = reader()
        if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
            raise TexturePropertyError(f"the {side} available-bytes reader must return a non-Boolean integer >= 0, got {available!r}")
        if need > int(available):
            raise TexturePropertyMemoryError(f"detaching {name!r} needs {need} B on the {side} side but only {int(available)} B are reported available; nothing was copied")
        out = arr.copy()
        return out

    def summary(self) -> dict:
        """Bounded metadata only (no array value, no reduction, no transfer)."""
        return {"component_tag": self.component_tag, "schema": SCHEMA, "recipe": RECIPE_TAG, "backend": self.backend, "n_layers": self.n_layers, "n_cells": self.n_cells, "output_layout": self.layout,
                "input_layouts": dict(self.layouts), "outputs": {k: v for k, v in self.estimate.outputs.items()}, "output_bytes": self.output_bytes, "sha256": self.sha256,
                "source_digests": dict(self.source_digests), "n_fc_equals_saturation": self.n_fc_equals_saturation,
                "scope": "pure texture->property component; no native adoption, no bed/water/clock mutation, equation verification only (not field validation)"}


def _raise(info: tuple, cells: int) -> None:
    ov, n_flagged, first, _ = info
    if not n_flagged:
        return
    element, flags = first >> 16, first & 0xFFFF
    low = flags & -flags
    layer, cell = element // cells, element % cells
    where = f"layer {layer}, cell {cell}" if cells > 1 else f"layer {layer}"
    raise TexturePropertyError(f"{FLAG_TEXT[low]} ({n_flagged} element(s) flagged, first at {where}, its flags {flags}; OR of all flags {ov}); nothing was published",
                               bit=low, n_flagged=n_flagged, first_layer=layer, first_cell=(cell if cells > 1 else None), flags=flags)


def estimate_soil_texture_properties(*, sand_fraction: Any, silt_fraction: Any, clay_fraction: Any, organic_matter_percent: Any, bulk_density_kg_m3: Any, n_layers: Any, n_cells: Any,
                                     theta_1500_override: Any = None, theta_33_override: Any = None, theta_saturation_override: Any = None, air_entry_kpa_override: Any = None,
                                     saturated_conductivity_override_m_s: Any = None, backend: str = "numpy", diagnostics: bool = False, memory_budget_bytes: Any,
                                     reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                                     host_available_bytes_fn: Callable[[], int] | None = None, chunk_columns: Any = None,
                                     hash_chunk_bytes: Any = DEFAULT_HASH_CHUNK_BYTES) -> SoilTextureProperties:
    """Validate the explicit inputs, compute the property maps (module docstring) and return an owned `SoilTextureProperties`. Order: metadata (no value read), quotas and availability, then
    output allocation and the value-checking kernel; any refusal is raised before anything is returned and the inputs are never modified or retained.

    Quotas follow the S1/A1 convention: `memory_budget_bytes`, `reserve_bytes`, `available_bytes_fn` apply to the main side (host for numpy/numba, DEVICE for cuda); the host side of a CUDA call (identity
    buffer + packet) is admitted separately against `host_memory_budget_bytes` (None = no separate quota), `host_reserve_bytes`, `host_available_bytes_fn`; these three are refused for the CPU backends."""
    L, N = _count(n_layers, "n_layers", minimum=1), _count(n_cells, "n_cells", minimum=1)
    if backend not in BACKENDS:
        raise TexturePropertyError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if type(diagnostics) is not bool:
        raise TexturePropertyError(f"diagnostics must be a bool, got {diagnostics!r}")
    budget, reserve = _count(memory_budget_bytes, "memory_budget_bytes"), _count(reserve_bytes, "reserve_bytes")
    host_reserve = _count(host_reserve_bytes, "host_reserve_bytes")
    host_budget = None if host_memory_budget_bytes is None else _count(host_memory_budget_bytes, "host_memory_budget_bytes")
    for fn_name, fn in (("available_bytes_fn", available_bytes_fn), ("host_available_bytes_fn", host_available_bytes_fn)):
        if fn is not None and not callable(fn):
            raise TexturePropertyError(f"{fn_name} must be callable or None")
    if backend != "cuda" and (host_reserve or host_budget is not None or host_available_bytes_fn is not None):
        raise TexturePropertyError("host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to the cuda backend")
    chunk = DEFAULT_CHUNK_COLUMNS if chunk_columns is None else _count(chunk_columns, "chunk_columns", minimum=1)
    hc = _count(hash_chunk_bytes, "hash_chunk_bytes", minimum=8)
    supplied = dict(zip(ALL_INPUTS, (sand_fraction, silt_fraction, clay_fraction, organic_matter_percent, bulk_density_kg_m3, theta_1500_override, theta_33_override, theta_saturation_override,
                                     air_entry_kpa_override, saturated_conductivity_override_m_s)))
    for key in INPUTS:
        if supplied[key] is None:
            raise TexturePropertyError(f"{key} is mandatory (explicit; no default is inferred)")
    cp = None
    if backend == "cuda":
        ns_type = _namespace_type(backend)
        from maple_syrup import soil_water_remap_cuda as rc

        cp = rc.require_cuda()
    else:
        ns_type = np.ndarray
        if backend == "numba":
            try:
                import numba  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                raise TexturePropertyBackendUnavailableError(f"backend 'numba' requires Numba ({type(exc).__name__}: {exc}); there is no fallback") from exc
    inputs = {k: _classify(k, v, L, N, ns_type, backend) for k, v in supplied.items() if v is not None}
    if cp is not None:
        for key, i in inputs.items():
            if i.array is not None and int(i.array.device.id) != rc.current_device_id(cp):
                raise TexturePropertyError(f"{key} lives on device {int(i.array.device.id)} but the current device is {rc.current_device_id(cp)}; no implicit transfer")
    layouts = {k: i.kind for k, i in inputs.items()}
    est = estimate_property_memory(L, N, layouts=layouts, backend=backend, diagnostics=diagnostics, chunk_columns=chunk, hash_chunk_bytes=hc, reserve_bytes=reserve,
                                   host_reserve_bytes=host_reserve, input_bytes={k: i.nbytes for k, i in inputs.items()})
    if backend == "cuda":
        _admit(est.device_required_bytes, budget, available_bytes_fn, "device", est, "memory_budget_bytes")
        _admit(est.host_required_bytes, host_budget, host_available_bytes_fn, "host", est, "host_memory_budget_bytes")
    else:
        _admit(est.main_required_bytes, budget, available_bytes_fn, "host", est, "memory_budget_bytes")
    layout = est.output_layout
    rows, cells = _rows_cells(L, N, layout)
    closure = closure_bound()
    xp = cp if cp is not None else np
    shape = (L, N) if layout == LAYOUT_CELL else (L,)
    out = {k: xp.zeros(shape, dtype=xp.float64) for k in OUTPUTS}
    if diagnostics:
        out["theta_1500"], out["air_entry_kpa"] = xp.zeros(shape, dtype=xp.float64), xp.zeros(shape, dtype=xp.float64)
        out["provenance"] = xp.zeros(shape, dtype=xp.uint8)
    if backend == "numpy":
        info = _derive_numpy(inputs, out, L, cells, min(chunk, cells), closure)
    elif backend == "numba":
        from maple_syrup import soil_texture_properties_numba as nb

        info = nb.run(inputs, out, L, N, cells, layout, closure)
    else:
        from maple_syrup import soil_texture_properties_cuda as cu

        info = cu.run(cp, inputs, out, L, N, cells, layout, closure)
    _raise(info, cells)
    host_buf = np.empty(max(1, hc // 8), dtype=np.float64) if cp is not None else None
    sha = _identity(inputs, L, N, diagnostics, host_buf)
    if cp is None:
        for arr in out.values():
            arr.flags.writeable = False
    device_id = None if cp is None else rc.current_device_id(cp)
    with _constructing():
        return SoilTextureProperties(backend, L, N, layout, layouts, sha, source_digests(), est, int(info[3]), device_id, out)
