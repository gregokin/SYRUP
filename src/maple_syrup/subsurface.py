"""Opt-in layered soil-water column for SYRUP: first bounded subsurface milestone (branch feature/subsurface-hydrology).

SYRUP owns all physical water in MAPLE-SYRUP-LEAF. This module is the isolated, explicitly FORCED column of the
design in docs/subsurface_hydrology/design.md: antecedent moisture plus time-resolved water ACTUALLY DIRECTED TO THE
SOIL (not gross rainfall), plus a potential soil-evaporation rate. It is not connected to the storm bucket, computes no
rainfall or infiltration capacity, invents no profile from an end-of-storm scalar, and changes no existing default.

    layers = soil_layers([0.0, 0.1, 0.3, 1.0], theta_residual=0.05, theta_field_capacity=0.25,
                         theta_saturation=0.45, diffusivity_m2_s=2e-7, theta_evaporation_stress=0.2,
                         evaporation_factor=1.2)                         # host, frozen, common per-layer values
    column = prepare_column(layers, active_mask, cell_area_m2, backend="numpy", reserve_bytes=0)
    state = uniform_state(column, [0.10, 0.12, 0.15])                   # antecedent theta per layer
    result = step(column, state, dt_s, added_water_m=intake, potential_evaporation_m_s=pet)
    profile = initialize_profile(column, state, records)                # streaming, no history

State: FP64 water-equivalent depth `water_m[layer, cell]` (m) and a finite physical time `t_s`; volumetric moisture
is depth / thickness; volume is depth x `cell_area_m2`. Cells are a flat 1-D index with an explicit boolean active
mask; nothing here guesses a grid.

Stage order of one step (dt > 0, active cells only; dt == 0 is the identity and rejects any added water):

  1. validation (inputs are never written; a refused step allocates no result; the candidate time t + dt must be
     finite),
  2. addition: `water += added_water_m` per layer; the part of an addition that would exceed the layer's saturation
     depth is returned to the surface (`surface_return_m`), never discarded or clipped,
  3. gravity cascade (`gravity_drainage=True`): top-to-bottom within the step, water above field capacity drains to
     the layer below (`percolation_m`), the bottom layer's excess leaves as `drainage_m`. With `gravity_drainage=False`
     nothing moves (isolated equation tests),
  4. evaporation from layer 0 (the physically declared upper evaporation layer of this scope): multiplier 0 at/below
     residual, (theta / theta_stress)^2 between residual and theta_stress, 1 above; demand = factor x E x dt x
     multiplier, withdrawn up to the water above residual; `evaporation_unmet_m` reports the rest,
  5. diffusive redistribution between adjacent layers, sequential pair exchange from the top: conductance
     G = D / (centre separation) [m/s], exact two-compartment amount (theta_i - theta_j) / (1/dz_i + 1/dz_j) x
     (-expm1(-G (1/dz_i + 1/dz_j) dt)), bounded BEFORE transfer by the donor's water above residual and the
     receiver's saturation capacity, both sides updated by the same number,
  6. output checks (finite, >= 0, <= saturation, per-cell closure) and ONE host read of the batched packet.

Every tolerance is the MAPLE-derived `conservation.volume_roundoff_bound_m3` (no new empirical floor): the per-cell
closure uses its dimensionless coefficient for 4 L + 7 terms, the global closure 4 L N + 7 terms on the largest
operand, the capacity checks LAYER_OPERATION_TERMS terms on the layer's saturation depth.

Backends: "numpy" (this reference: loops over layers, vectorised over cells, in-place on the fresh output arrays with
a counted transient peak), "numba" (compiled cell/layer loops, no fastmath, `subsurface_numba`), "cuda" (a tiny
per-interface factor kernel then one CuPy RawKernel thread per column, bounded layer loops, no fixed maximum layer
count, `subsurface_cuda`). Because the layer properties are COMMON to every cell in this first implementation, the
compiled backends evaluate the per-interface exchange factor `-expm1(-((G s) dt))` once per interface per step (same
expression order, same math implementation) and reuse it for all cells; the reference keeps the scalar evaluation
per interface it always had. That reuse is conditional on common properties, not a claim about per-cell maps. A missing backend
is a `SubsurfaceBackendUnavailableError`, never a silent fallback. Dynamic arrays must be genuine `numpy.ndarray` /
`cupy.ndarray` objects (no subclass, no masked array) of the column's namespace, float64, C-contiguous and exactly
shaped; nothing is converted implicitly.

Memory: `estimate_step_memory` is a backend-specific, documented upper bound on the arrays this module and its
caller hold for one step (the caller's state and forcing arrays, the owned mask/thickness copies, every SoilLayers
vector, the kernel-parameter copies on host and device, the fresh outputs, the transient reference temporaries,
the packet) plus the caller's reserve. The relevant part (`required_bytes`: host bytes for "numpy"/"numba", device
bytes for "cuda", reserve included) is compared with the reported available bytes BEFORE every bulk allocation:
`prepare_column` (before the mask copies, parameter copies, compilation and warm-up), `column_state`,
`uniform_state` and `step`. It is not the allocator pool, the CUDA context, Numba/NVRTC JIT workspaces, Python
objects, or the whole MAPLE/LEAF process memory (see MEMORY_SCOPE); a reading is not a reservation, and the budget is
never raised silently.

References and departures: Tietjen, Zehe & Jeltsch (2009) two-layer hydrology; its MAHLERAN daily adaptation
(`Subroutines_Interstorm/calc_inf.f90`, `calc_et.f90`, `calc_sm.f90`); ECOTONE `hwdrain`; SOILWAT layered
accounting. The cascade drains within the step (ECOTONE-like), whereas the MAHLERAN adaptation moves OLD storage one
interface per day and then clips to [theta_min, theta_sat] (`calc_sm.f90`); this column never clips, so no numerical
identity with that daily model is claimed. Changing the thickness of layer 0 changes the evaporation zone. Nothing
here is a Richards or heterogeneous-potential solver, a plant-uptake model, a storm coupling, a checkpoint format or a
LEAF interface. Nothing here was run by its author (file-only tools); Codex records executions.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np

from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = [
    "BACKENDS",
    "FLAG_MESSAGES",
    "KERNEL_SCRATCH_DOUBLES_PER_CELL",
    "LAYER_OPERATION_TERMS",
    "MEMORY_SCOPE",
    "NO_FLAG",
    "NUMPY_SCRATCH_DOUBLES_PER_CELL",
    "PACKET_LENGTH",
    "SCRATCH_MASK_BYTES_PER_CELL",
    "STAGE_ORDER",
    "CumulativeBudget",
    "ForcingRecord",
    "KernelParameters",
    "MemoryEstimate",
    "ProfileInitialization",
    "SoilColumn",
    "SoilColumnState",
    "SoilLayers",
    "StepArrays",
    "StepBudget",
    "SubsurfaceBackendUnavailableError",
    "SubsurfaceError",
    "SubsurfaceMemoryError",
    "SubsurfaceStep",
    "SubsurfaceStepError",
    "SubsurfaceStreamError",
    "admit_step",
    "cell_closure_coefficient",
    "column_state",
    "default_host_available_bytes",
    "estimate_step_memory",
    "initialize_profile",
    "kernel_parameter_doubles",
    "kernel_parameters",
    "layer_vector_doubles",
    "moisture",
    "prepare_column",
    "reference_step",
    "soil_layers",
    "step",
    "storage_m3",
    "summary",
    "uniform_state",
]

BACKENDS = ("numpy", "numba", "cuda")
STAGE_ORDER = ("validation", "addition", "gravity_cascade", "evaporation", "diffusion", "output_checks")
# Floating-point accumulations one layer value undergoes in one step (addition, saturation return, cascade inflow,
# cascade outflow, evaporation, diffusion with the pair above, diffusion with the pair below) plus one.
LAYER_OPERATION_TERMS = 8
NO_FLAG = float(2 ** 40)  # row sentinel: no failure bit recorded for this cell
PACKET_LENGTH = 9
ROW_COUNT = 4  # storage before, storage after, added, lowest failure bit
# Transient scratch of the NumPy reference, per cell: the counted worst case is 8 doubles live at once (the int64 flag
# word plus at most seven float64 temporaries during the diffusion stage), declared at twice that count, plus boolean
# masks (at most a few N-byte arrays at once, declared as 8 bytes per cell). Kernel backends allocate only the outputs;
# their scratch is the packet stage's boolean mask and reduction workspace, declared as 2 doubles plus 8 bytes per cell.
NUMPY_SCRATCH_DOUBLES_PER_CELL = 16
KERNEL_SCRATCH_DOUBLES_PER_CELL = 2
SCRATCH_MASK_BYTES_PER_CELL = 8
_PACKET_SCRATCH_BYTES = 4 * PACKET_LENGTH * 8
MEMORY_SCOPE = (
    "arrays held for ONE step: the caller's state and forcing arrays (known inputs), the column's owned mask and "
    "thickness copies, every SoilLayers vector, the kernel-parameter copies (host for numba/cuda, device for cuda), "
    "the freshly allocated outputs, the transient reference temporaries and the packet, plus the caller's reserve; "
    "host and device bytes are listed separately and the backend's own side is the admitted requirement. NOT "
    "included: the NumPy/CuPy allocator pool and its retained blocks, the CUDA context, Numba/NVRTC JIT and compile "
    "workspaces, Python object overhead, other model state, and the remaining MAPLE/LEAF process memory. The "
    "estimate is an upper bound on what this module allocates, not a reservation."
)

FLAG_MESSAGES = (
    "soil water state must be finite everywhere",  # 0
    "soil water state must be >= 0 everywhere",  # 1
    "soil water state exceeds the layer saturation depth",  # 2
    "added_water_m must be finite everywhere",  # 3
    "added_water_m must be >= 0 everywhere",  # 4
    "added_water_m must be 0 on inactive cells (no forcing on inactive cells)",  # 5
    "potential_evaporation_m_s must be finite everywhere",  # 6
    "potential_evaporation_m_s must be >= 0 everywhere",  # 7
    "potential_evaporation_m_s must be 0 on inactive cells (no forcing on inactive cells)",  # 8
    "added_water_m must be 0 when dt_s == 0 (water cannot enter without advancing time)",  # 9
    "step produced non-finite soil water",  # 10
    "step produced negative soil water",  # 11
    "step produced soil water above the layer saturation depth",  # 12
    "per-cell water closure violated beyond the FP64 bound",  # 13
    "step produced a non-finite flux",  # 14
)
N_FLAGS = len(FLAG_MESSAGES)


# --- errors ----------------------------------------------------------------------------------------------------------
class SubsurfaceError(ValueError):
    """Invalid layers, column, state, forcing, time or option. Raised before any result exists; inputs are untouched."""


class SubsurfaceStepError(SubsurfaceError):
    """A step's batched validation recorded a failure (lowest bit first) or the global closure failed."""

    def __init__(self, message: str, *, bit: int | None = None, n_flagged: int = 0):
        super().__init__(message)
        self.bit = bit
        self.n_flagged = int(n_flagged)


class SubsurfaceBackendUnavailableError(RuntimeError):
    """The requested backend (Numba, CuPy/CUDA, or a host memory query) is unavailable. No fallback."""


class SubsurfaceMemoryError(MemoryError):
    """The working-array estimate plus the caller's reserve exceeds the reported available bytes (nothing allocated)."""


class SubsurfaceStreamError(SubsurfaceError):
    """`initialize_profile` stopped. Carries the completed step count, the last ACCEPTED state and the cumulative
    budget up to it (never a partially published step); the caller's original state and records are untouched."""

    def __init__(self, message: str, *, completed_steps: int, last_state: Any, cumulative: Any):
        super().__init__(message)
        self.completed_steps = int(completed_steps)
        self.last_state = last_state
        self.cumulative = cumulative


# --- small validators --------------------------------------------------------------------------------------------------
def _real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SubsurfaceError(f"{name} must be a real host number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise SubsurfaceError(f"{name} must be finite, got {value!r}")
    return out


def _checked_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise SubsurfaceError(f"{name} must be a non-Boolean integer >= 0, got {value!r}")
    return int(value)


def _layer_array(value: Any, n: int, name: str) -> np.ndarray:
    """A float64 `(n,)` host array from a real scalar or a length-`n` real sequence; Booleans, non-numeric and
    non-finite values are refused. Always a new array."""
    if isinstance(value, (bool, np.bool_)):
        raise SubsurfaceError(f"{name} must be numeric, got a Boolean")
    if isinstance(value, (int, float, np.integer, np.floating)):
        arr = np.full(n, float(value), dtype=np.float64)
    else:
        try:
            raw = np.asarray(value)
        except Exception as exc:  # noqa: BLE001 - reported by name
            raise SubsurfaceError(f"{name} is not array-like ({type(exc).__name__}: {exc})") from None
        if raw.dtype.kind not in "iuf":
            raise SubsurfaceError(f"{name} must be real numbers, got dtype {raw.dtype}")
        if raw.ndim == 0:
            arr = np.full(n, float(raw), dtype=np.float64)
        elif raw.shape != (n,):
            raise SubsurfaceError(f"{name} must be a scalar or have shape ({n},), got shape {raw.shape}")
        else:
            arr = np.array(raw, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise SubsurfaceError(f"{name} must be finite everywhere")
    return arr


def _require_finite(name: str, *arrays: np.ndarray) -> None:
    for a in arrays:
        if not np.all(np.isfinite(a)):
            raise SubsurfaceError(f"derived {name} is not representable in FP64 (non-finite); refuse this layer setup "
                                  "rather than floor or clip it")


# --- layers ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SoilLayers:
    """Common per-layer soil properties (host, frozen, owned copies). Built by `soil_layers`."""

    depth_edges_m: np.ndarray  # (L + 1,) strictly increasing from 0
    thickness_m: np.ndarray  # (L,)
    centre_depth_m: np.ndarray  # (L,)
    theta_residual: np.ndarray  # (L,)
    theta_field_capacity: np.ndarray  # (L,)
    theta_saturation: np.ndarray  # (L,)
    diffusivity_m2_s: np.ndarray  # (L - 1,) per interface
    theta_evaporation_stress: float  # layer 0 threshold of the quadratic evaporation reduction
    evaporation_factor: float  # nonnegative multiplier of the potential rate (1.2 = bare upper layer reference)
    gravity_drainage: bool
    residual_m: np.ndarray  # (L,) theta_residual x thickness
    field_capacity_m: np.ndarray  # (L,)
    saturation_m: np.ndarray  # (L,)
    interface_conductance_m_s: np.ndarray  # (L - 1,) D / centre separation
    pair_inverse_thickness: np.ndarray  # (L - 1,) 1/dz_k + 1/dz_{k+1}
    capacity_tolerance_m: np.ndarray  # (L,) MAPLE coefficient(LAYER_OPERATION_TERMS) x saturation_m

    @property
    def n_layers(self) -> int:
        return int(self.thickness_m.shape[0])

    def summary(self) -> dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "depth_edges_m": self.depth_edges_m.tolist(),
            "theta_residual": self.theta_residual.tolist(),
            "theta_field_capacity": self.theta_field_capacity.tolist(),
            "theta_saturation": self.theta_saturation.tolist(),
            "diffusivity_m2_s": self.diffusivity_m2_s.tolist(),
            "interface_conductance_m_s": self.interface_conductance_m_s.tolist(),
            "theta_evaporation_stress": self.theta_evaporation_stress,
            "evaporation_factor": self.evaporation_factor,
            "gravity_drainage": self.gravity_drainage,
            "evaporation_layer": 0,
            "stage_order": list(STAGE_ORDER),
            "vector_bytes": layer_vector_doubles(self.n_layers) * 8,
        }


def layer_vector_doubles(n_layers: int) -> int:
    """Doubles held by the thirteen SoilLayers vectors: (L + 1) edges, nine length-L vectors, three length-(L-1)."""
    n = int(n_layers)
    return (n + 1) + 9 * n + 3 * max(n - 1, 0)


def soil_layers(depth_edges_m: Any, *, theta_residual: Any, theta_field_capacity: Any, theta_saturation: Any,
                diffusivity_m2_s: Any, theta_evaporation_stress: Any, evaporation_factor: Any = 1.0,
                gravity_drainage: bool = True) -> SoilLayers:
    """Validate and freeze a common layer stack. `depth_edges_m` are strictly increasing from 0 (L = len - 1 >= 1
    layers). Retention values are scalars or length-L sequences with 0 <= residual <= field capacity <= saturation <= 1
    and residual < saturation in every layer. `diffusivity_m2_s` (D, m2/s, >= 0) is a scalar or a length-(L-1)
    sequence per interface; the conductance is D / (centre separation) and is NOT rescaled by the pair thickness
    (for the two-layer MAHLERAN adaptation D = K_mean x d x (dz0 + dz1) reproduces its exchange). The evaporation
    stress threshold applies to layer 0 and must exceed its residual. Every DERIVED quantity (thickness inverses,
    centre separations, conductances, exchange rates, storage depths, tolerances) must be finite and representable;
    a setup whose derived values overflow or vanish in FP64 is refused, never floored."""
    try:
        edges = np.asarray(depth_edges_m)
    except Exception as exc:  # noqa: BLE001
        raise SubsurfaceError(f"depth_edges_m is not array-like ({type(exc).__name__}: {exc})") from None
    if edges.dtype.kind not in "iuf" or edges.ndim != 1:
        raise SubsurfaceError("depth_edges_m must be a 1-D sequence of real numbers")
    if edges.shape[0] < 2:
        raise SubsurfaceError("depth_edges_m needs at least two edges (one layer)")
    edges = np.array(edges, dtype=np.float64)
    if not np.all(np.isfinite(edges)):
        raise SubsurfaceError("depth_edges_m must be finite")
    if edges[0] != 0.0:
        raise SubsurfaceError(f"depth_edges_m must start at 0, got {edges[0]!r}")
    thickness = np.diff(edges)
    if not np.all(thickness > 0.0):
        raise SubsurfaceError("depth_edges_m must be strictly increasing")
    n = int(thickness.shape[0])
    res = _layer_array(theta_residual, n, "theta_residual")
    fc = _layer_array(theta_field_capacity, n, "theta_field_capacity")
    sat = _layer_array(theta_saturation, n, "theta_saturation")
    if np.any(res < 0.0):
        raise SubsurfaceError("theta_residual must be >= 0")
    if np.any(fc < res):
        raise SubsurfaceError("theta_field_capacity must be >= theta_residual in every layer")
    if np.any(sat < fc):
        raise SubsurfaceError("theta_saturation must be >= theta_field_capacity in every layer")
    if np.any(sat > 1.0):
        raise SubsurfaceError("theta_saturation must be <= 1")
    if np.any(sat <= res):
        raise SubsurfaceError("theta_residual must be strictly below theta_saturation in every layer")
    if n > 1:
        diffusivity = _layer_array(diffusivity_m2_s, n - 1, "diffusivity_m2_s")
    else:
        if isinstance(diffusivity_m2_s, (bool, np.bool_)):
            raise SubsurfaceError("diffusivity_m2_s must be numeric, got a Boolean")
        raw = np.asarray(diffusivity_m2_s)
        if raw.dtype.kind not in "iuf" or raw.ndim > 1 or (raw.ndim == 1 and raw.shape[0] != 0):
            raise SubsurfaceError("a single layer accepts a scalar or an empty diffusivity_m2_s")
        if raw.ndim == 0 and not np.isfinite(float(raw)):
            raise SubsurfaceError("diffusivity_m2_s must be finite")
        if raw.ndim == 0 and float(raw) < 0.0:
            raise SubsurfaceError("diffusivity_m2_s must be >= 0")
        diffusivity = np.zeros(0, dtype=np.float64)
    if np.any(diffusivity < 0.0):
        raise SubsurfaceError("diffusivity_m2_s must be >= 0")
    stress = _real(theta_evaporation_stress, "theta_evaporation_stress")
    if stress <= res[0]:
        raise SubsurfaceError(f"theta_evaporation_stress must exceed the layer-0 residual {res[0]!r}, got {stress!r}")
    factor = _real(evaporation_factor, "evaporation_factor")
    if factor < 0.0:
        raise SubsurfaceError(f"evaporation_factor must be >= 0, got {factor!r}")
    if type(gravity_drainage) is not bool:
        raise SubsurfaceError(f"gravity_drainage must be a bool, got {gravity_drainage!r}")
    # Derived quantities, each required to be finite and representable (finite inputs can still overflow or vanish).
    with np.errstate(all="ignore"):
        inverse_thickness = 1.0 / thickness
        centres = edges[:-1] + thickness / 2.0
        separation = centres[1:] - centres[:-1]
        conductance = diffusivity / separation if n > 1 else np.zeros(0, dtype=np.float64)
        pair_inv = inverse_thickness[:-1] + inverse_thickness[1:] if n > 1 else np.zeros(0, dtype=np.float64)
        exchange_rate = conductance * pair_inv  # 1/s, the argument scale of expm1 per unit dt
        residual_m, field_capacity_m, saturation_m = res * thickness, fc * thickness, sat * thickness
        coefficient = volume_roundoff_bound_m3(LAYER_OPERATION_TERMS, 1.0)  # MAPLE's dimensionless FP64 coefficient
        capacity_tolerance = coefficient * saturation_m
    _require_finite("inverse thickness", inverse_thickness)
    _require_finite("layer centre depth", centres)
    _require_finite("centre separation", separation)
    if n > 1 and not np.all(separation > 0.0):
        raise SubsurfaceError("derived centre separations must be positive (layers too thin to be representable)")
    _require_finite("interface conductance", conductance)
    _require_finite("pair inverse thickness sum", pair_inv)
    _require_finite("interface exchange rate (conductance x inverse thickness sum)", exchange_rate)
    _require_finite("storage depths", residual_m, field_capacity_m, saturation_m)
    if not np.all(saturation_m > 0.0):
        raise SubsurfaceError("derived saturation depth vanished in FP64 (layer too thin); refused, not floored")
    _require_finite("capacity tolerance", capacity_tolerance)
    if not np.all(saturation_m + capacity_tolerance > saturation_m):
        raise SubsurfaceError("derived capacity tolerance is not resolvable at the layer saturation depth; refused")
    from maple.core.backend import freeze

    def own(a: np.ndarray) -> np.ndarray:
        return freeze(np.ascontiguousarray(np.array(a, dtype=np.float64)))

    return SoilLayers(
        depth_edges_m=own(edges), thickness_m=own(thickness), centre_depth_m=own(centres), theta_residual=own(res),
        theta_field_capacity=own(fc), theta_saturation=own(sat), diffusivity_m2_s=own(diffusivity),
        theta_evaporation_stress=stress, evaporation_factor=factor, gravity_drainage=gravity_drainage,
        residual_m=own(residual_m), field_capacity_m=own(field_capacity_m), saturation_m=own(saturation_m),
        interface_conductance_m_s=own(conductance), pair_inverse_thickness=own(pair_inv),
        capacity_tolerance_m=own(capacity_tolerance))


class KernelParameters(NamedTuple):
    """Host float64 copies in kernel order; interface arrays are padded to length max(L - 1, 1) so a one-layer stack
    passes non-empty buffers (the kernels loop to L - 1 and never read the padding)."""

    thickness_m: np.ndarray
    residual_m: np.ndarray
    field_capacity_m: np.ndarray
    saturation_m: np.ndarray
    capacity_tolerance_m: np.ndarray
    interface_conductance_m_s: np.ndarray
    pair_inverse_thickness: np.ndarray
    residual_theta0: float
    theta_evaporation_stress: float
    evaporation_factor: float
    gravity_code: int
    cell_closure_coefficient: float


def kernel_parameter_doubles(n_layers: int) -> int:
    """Doubles of one KernelParameters copy: five length-L vectors and two padded interface vectors."""
    n = int(n_layers)
    return 5 * n + 2 * max(n - 1, 1)


def cell_closure_coefficient(n_layers: int) -> float:
    """MAPLE's dimensionless FP64 bound coefficient for the 4 L + 7 terms of one cell's step balance."""
    return volume_roundoff_bound_m3(4 * int(n_layers) + 7, 1.0)


def kernel_parameters(layers: SoilLayers) -> KernelParameters:
    n = layers.n_layers

    def padded(a: np.ndarray) -> np.ndarray:
        out = np.zeros(max(n - 1, 1), dtype=np.float64)
        out[: n - 1] = a
        return out

    return KernelParameters(
        np.array(layers.thickness_m), np.array(layers.residual_m), np.array(layers.field_capacity_m),
        np.array(layers.saturation_m), np.array(layers.capacity_tolerance_m),
        padded(layers.interface_conductance_m_s), padded(layers.pair_inverse_thickness),
        float(layers.theta_residual[0]), float(layers.theta_evaporation_stress), float(layers.evaporation_factor),
        1 if layers.gravity_drainage else 0, cell_closure_coefficient(n))


# --- memory ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MemoryEstimate:
    """Backend-specific upper bound of the bytes held for one step (see MEMORY_SCOPE). `required_bytes` is the part
    compared with `available_bytes_fn` (host side for numpy/numba, device side for cuda), reserve included."""

    backend: str
    n_layers: int
    n_cells: int
    input_bytes: int  # caller's state and forcing arrays, in the column namespace
    column_bytes: int  # owned mask copies (bool + uint8) and thickness copy, in the column namespace
    layer_bytes: int  # every SoilLayers vector (host)
    parameter_bytes: int  # KernelParameters copies: host (numba, cuda) and device (cuda)
    output_bytes: int  # fresh candidate outputs of one step (state, fluxes, rows), in the column namespace
    scratch_bytes: int  # transient temporaries of the backend's step plus the packet
    reserve_bytes: int  # caller-specified
    host_bytes: int
    device_bytes: int
    required_bytes: int  # admitted side + reserve
    step_allocation_bytes: int  # output_bytes + scratch_bytes: what one step itself allocates at its peak
    total_bytes: int  # host + device + reserve
    scope: str = MEMORY_SCOPE


def estimate_step_memory(n_layers: int, n_cells: int, *, backend: str, reserve_bytes: int = 0) -> MemoryEstimate:
    """Conservative, backend-specific working-array bytes of ONE step (see MEMORY_SCOPE). Pure arithmetic."""
    n_layers = _checked_int(n_layers, "n_layers")
    n_cells = _checked_int(n_cells, "n_cells")
    if n_layers < 1 or n_cells < 1:
        raise SubsurfaceError("n_layers and n_cells must be >= 1")
    if backend not in BACKENDS:
        raise SubsurfaceError(f"backend must be one of {BACKENDS}, got {backend!r}")
    reserve = _checked_int(reserve_bytes, "reserve_bytes")
    f8 = 8
    n_if = n_layers - 1
    inputs = (2 * n_layers * n_cells + n_cells) * f8  # state, added water, evaporation rate
    column = 2 * n_cells + n_layers * f8  # bool + uint8 masks, thickness copy
    layers = layer_vector_doubles(n_layers) * f8
    params_host = kernel_parameter_doubles(n_layers) * f8
    outputs = (n_layers * n_cells + 2 * n_if * n_cells + 4 * n_cells + ROW_COUNT * n_cells) * f8
    if backend == "numpy":
        scratch = NUMPY_SCRATCH_DOUBLES_PER_CELL * n_cells * f8 + SCRATCH_MASK_BYTES_PER_CELL * n_cells + _PACKET_SCRATCH_BYTES
        parameters = 0
        host = inputs + column + layers + outputs + scratch
        device = 0
    elif backend == "numba":
        # + the per-call local (L-1,) exchange-factor array of the compiled kernel (host)
        scratch = (KERNEL_SCRATCH_DOUBLES_PER_CELL * n_cells * f8 + SCRATCH_MASK_BYTES_PER_CELL * n_cells
                   + _PACKET_SCRATCH_BYTES + max(n_if, 1) * f8)
        parameters = params_host
        host = inputs + column + layers + parameters + outputs + scratch
        device = 0
    else:
        # + the per-step (max(L-1,1),) exchange-factor scratch written by the factor kernel (device)
        scratch = (KERNEL_SCRATCH_DOUBLES_PER_CELL * n_cells * f8 + SCRATCH_MASK_BYTES_PER_CELL * n_cells
                   + _PACKET_SCRATCH_BYTES + max(n_if, 1) * f8)
        parameters = 2 * params_host  # host copies and owned device copies
        host = layers + params_host + _PACKET_SCRATCH_BYTES
        device = inputs + column + params_host + outputs + scratch - _PACKET_SCRATCH_BYTES
    required = (device if backend == "cuda" else host) + reserve
    return MemoryEstimate(backend, n_layers, n_cells, inputs, column, layers, parameters, outputs, scratch, reserve,
                          host, device, required, outputs + scratch, host + device + reserve)


def default_host_available_bytes() -> int:
    """Available physical host memory reported by the OS (`sysconf`); a platform without it is a setup error (inject
    `available_bytes_fn`). This is a reading, not a reservation."""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError) as exc:
        raise SubsurfaceBackendUnavailableError(
            f"the host cannot report available memory ({type(exc).__name__}: {exc}); pass available_bytes_fn") from exc
    if pages < 0 or page <= 0:
        raise SubsurfaceBackendUnavailableError("the host reported an invalid available-memory figure; pass available_bytes_fn")
    return int(pages) * int(page)


def _admit(estimate: MemoryEstimate, available_bytes_fn: Callable[[], int], what: str, extra_bytes: int = 0) -> int:
    """Compare `estimate.required_bytes + extra_bytes` with the reported available bytes and raise
    `SubsurfaceMemoryError` BEFORE anything is allocated. Returns the reading. A reading is not a reservation; the
    budget is never raised silently."""
    available = available_bytes_fn()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise SubsurfaceError(f"available_bytes_fn must return a non-Boolean integer >= 0, got {available!r}")
    available = int(available)
    need = estimate.required_bytes + int(extra_bytes)
    if need > available:
        raise SubsurfaceMemoryError(
            f"subsurface {what} on {estimate.n_layers} layers x {estimate.n_cells} cells ({estimate.backend}) needs an "
            f"estimated {need} B on its admitted side (reserve {estimate.reserve_bytes} B included) but only "
            f"{available} B are reported available; nothing was allocated. Scope: {MEMORY_SCOPE}")
    return available


# --- column ------------------------------------------------------------------------------------------------------------
class StepArrays(NamedTuple):
    """Fresh per-step arrays of the column namespace, produced by a backend runner (no host read yet)."""

    water_m: Any  # (L, N)
    percolation_m: Any  # (L - 1, N) cascade flow across each interface, downward
    diffusion_m: Any  # (L - 1, N) diffusive exchange across each interface, positive downward
    surface_return_m: Any  # (N,)
    drainage_m: Any  # (N,)
    evaporation_m: Any  # (N,)
    evaporation_unmet_m: Any  # (N,)
    rows: Any  # (ROW_COUNT, N): storage before, storage after, added, lowest failure bit (NO_FLAG if none)
    scratch: Any = None  # backend-internal keep-alive (CUDA per-interface factor) held only through the packet read


@dataclass(frozen=True, eq=False)
class SoilColumn:
    """A layer stack bound to a flat cell set, an area, a namespace and a backend runner. Built by `prepare_column`.
    Owns copies of the mask and the parameters; never aliases a caller array."""

    layers: SoilLayers
    backend: str
    xp: ModuleType
    n_layers: int
    n_cells: int
    n_active: int
    cell_area_m2: float
    active: Any  # (N,) bool, owned
    active_u8: Any  # (N,) uint8, owned (kernel mask)
    thickness_xp: Any  # (L,) float64 in xp, owned
    device_id: int | None
    reserve_bytes: int
    available_bytes_fn: Callable[[], int]
    memory: MemoryEstimate
    runner: Callable[..., StepArrays]
    preflight: Callable[[], None] | None
    backend_info: dict

    def summary(self) -> dict[str, Any]:
        return summary(self)


@dataclass(frozen=True, eq=False)
class SoilColumnState:
    """`water_m` (L, N) float64 of the column namespace plus the finite physical time. The state owns its array
    (built by `column_state` / `uniform_state` / a step); steps never write it."""

    water_m: Any
    t_s: float


def _check_column(column: Any) -> None:
    if not isinstance(column, SoilColumn):
        raise SubsurfaceError(f"column must be a SoilColumn, got {type(column).__name__}")


def _check_dynamic(column: SoilColumn, name: str, array: Any, shape: tuple[int, ...]) -> None:
    """Exact type of the column namespace (no subclass, no masked array, no other namespace), float64, C-contiguous,
    exact shape, and for CUDA the column's device. Metadata only: no element is read."""
    xp = column.xp
    expected = xp.ndarray
    if type(array) is not expected:
        raise SubsurfaceError(f"{name} must be exactly a {xp.__name__}.ndarray of the column namespace "
                              f"(no subclass, no masked array, no implicit transfer), got {type(array).__module__}."
                              f"{type(array).__name__}")
    if array.dtype != np.float64:
        raise SubsurfaceError(f"{name} must be float64, got {array.dtype}")
    if tuple(array.shape) != tuple(shape):
        raise SubsurfaceError(f"{name} must have shape {tuple(shape)}, got {tuple(array.shape)}")
    if not array.flags.c_contiguous:
        raise SubsurfaceError(f"{name} must be C-contiguous")
    if column.backend == "cuda" and int(array.device.id) != column.device_id:
        raise SubsurfaceError(f"{name} lives on device {int(array.device.id)}, the column is bound to device "
                              f"{column.device_id}; no implicit device transfer")


def _check_state(column: SoilColumn, state: Any) -> float:
    """Array metadata and a FINITE real time; returns the time."""
    if not isinstance(state, SoilColumnState):
        raise SubsurfaceError(f"state must be a SoilColumnState, got {type(state).__name__}")
    _check_dynamic(column, "state.water_m", state.water_m, (column.n_layers, column.n_cells))
    return _real(state.t_s, "state.t_s")


def _check_dt(dt_s: Any) -> float:
    dt = _real(dt_s, "dt_s")
    if dt < 0.0:
        raise SubsurfaceError(f"dt_s must be >= 0, got {dt!r}")
    return dt


def _check_mask(xp: ModuleType, active_mask: Any, device_id: int | None) -> None:
    if type(active_mask) is not xp.ndarray:
        raise SubsurfaceError(f"active_mask must be exactly a {xp.__name__}.ndarray (flat cell order), got "
                              f"{type(active_mask).__module__}.{type(active_mask).__name__}")
    if active_mask.dtype != np.bool_:
        raise SubsurfaceError(f"active_mask must be bool, got {active_mask.dtype}")
    if active_mask.ndim != 1:
        raise SubsurfaceError(f"active_mask must be 1-D (flat cells; no grid is inferred), got ndim {active_mask.ndim}")
    if active_mask.shape[0] < 1:
        raise SubsurfaceError("active_mask must have at least one cell")
    if device_id is not None and int(active_mask.device.id) != device_id:
        raise SubsurfaceError(f"active_mask lives on device {int(active_mask.device.id)}, the current device is {device_id}")


def prepare_column(layers: SoilLayers, active_mask: Any, cell_area_m2: Any, *, backend: str = "numpy",
                   reserve_bytes: int = 0, available_bytes_fn: Callable[[], int] | None = None) -> SoilColumn:
    """Bind `layers` to a flat cell set. `active_mask` is a 1-D bool array of the backend namespace (host NumPy for
    "numpy"/"numba", CuPy on the CURRENT device for "cuda"); `cell_area_m2` > 0 converts depths to volumes.
    `reserve_bytes` (required integer, 0 allowed) is added to the per-step working-array requirement; `available_bytes_fn`
    defaults to the OS/host or device query. The requirement is admitted BEFORE the mask copies, parameter copies,
    kernel compilation and warm-up; "numba" and "cuda" compile here (startup, outside steps). An unavailable backend
    raises `SubsurfaceBackendUnavailableError` and never falls back."""
    if not isinstance(layers, SoilLayers):
        raise SubsurfaceError(f"layers must be a SoilLayers, got {type(layers).__name__}")
    if backend not in BACKENDS:
        raise SubsurfaceError(f"backend must be one of {BACKENDS}, got {backend!r}")
    reserve = _checked_int(reserve_bytes, "reserve_bytes")
    area = _real(cell_area_m2, "cell_area_m2")
    if area <= 0.0:
        raise SubsurfaceError(f"cell_area_m2 must be > 0, got {area!r}")
    if available_bytes_fn is not None and not callable(available_bytes_fn):
        raise SubsurfaceError("available_bytes_fn must be callable or None")
    from maple.core.backend import freeze, to_device, to_int

    preflight = None
    if backend == "cuda":
        from maple_syrup import subsurface_cuda

        cp = subsurface_cuda.require_cuda()  # no allocation
        device_id = subsurface_cuda.current_device_id(cp)
        xp = cp
        _check_mask(cp, active_mask, device_id)
        if available_bytes_fn is None:
            available_bytes_fn = subsurface_cuda.default_device_available_bytes_fn(cp, device_id)
    else:
        if backend == "numba":
            from maple_syrup import subsurface_numba

            subsurface_numba.require_numba()  # import probe only
        xp = np
        device_id = None
        _check_mask(np, active_mask, None)
        if available_bytes_fn is None:
            available_bytes_fn = default_host_available_bytes
    n_cells = int(active_mask.shape[0])
    n_layers = layers.n_layers
    memory = estimate_step_memory(n_layers, n_cells, backend=backend, reserve_bytes=reserve)
    _admit(memory, available_bytes_fn, "column preparation")  # BEFORE any bulk copy, compilation or warm-up
    active = xp.ascontiguousarray(xp.array(active_mask, dtype=np.bool_, copy=True))
    active_u8 = xp.ascontiguousarray(active.astype(np.uint8))
    n_active = int(to_int(xp.count_nonzero(active)))
    thickness_xp = to_device(np.array(layers.thickness_m), xp) if xp is not np else freeze(np.array(layers.thickness_m))
    if backend == "numpy":
        runner, info = reference_runner, {"implementation": "numpy reference (layer loop, vectorised cells, in place)"}
    elif backend == "numba":
        runner, info = subsurface_numba.prepare_runner(layers)
    else:
        runner, info, preflight = subsurface_cuda.prepare_runner(layers, cp, device_id)
    return SoilColumn(layers=layers, backend=backend, xp=xp, n_layers=n_layers, n_cells=n_cells, n_active=n_active,
                      cell_area_m2=area, active=active, active_u8=active_u8, thickness_xp=thickness_xp,
                      device_id=device_id, reserve_bytes=reserve, available_bytes_fn=available_bytes_fn, memory=memory,
                      runner=runner, preflight=preflight, backend_info=dict(info))


def admit_step(column: SoilColumn) -> int:
    """Compare the column's per-step requirement (reserve included) with the reported available bytes and raise
    `SubsurfaceMemoryError` BEFORE anything is allocated. Returns the reading."""
    _check_column(column)
    return _admit(column.memory, column.available_bytes_fn, "step")


# --- states ------------------------------------------------------------------------------------------------------------
def column_state(column: SoilColumn, water_m: Any, t_s: float = 0.0) -> SoilColumnState:
    """A state owning a COPY of `water_m` (same namespace, strict array type/dtype/shape/device). The copy is one extra
    state-sized allocation and is admitted (per-step requirement + that copy) before it is made. Values are checked
    by the next step's batched validation (finite, >= 0, <= saturation)."""
    _check_column(column)
    _check_dynamic(column, "water_m", water_m, (column.n_layers, column.n_cells))
    t = _real(t_s, "t_s")
    _admit(column.memory, column.available_bytes_fn, "state copy", extra_bytes=column.n_layers * column.n_cells * 8)
    return SoilColumnState(column.xp.ascontiguousarray(column.xp.array(water_m, dtype=np.float64, copy=True)), t)


def uniform_state(column: SoilColumn, theta_by_layer: Any, t_s: float = 0.0) -> SoilColumnState:
    """Antecedent moisture from explicit per-layer theta values (length L), the same in every cell. Admitted before
    allocation; on CUDA only the L per-layer depths are uploaded and the (L, N) array is allocated and broadcast on the
    device. This is the only convenience initialiser: there is deliberately no profile from a scalar total."""
    _check_column(column)
    theta = _layer_array(theta_by_layer, column.n_layers, "theta_by_layer")
    if np.any(theta < 0.0) or np.any(theta > column.layers.theta_saturation):
        raise SubsurfaceError("theta_by_layer must lie within [0, theta_saturation] in every layer")
    t = _real(t_s, "t_s")
    depth = np.ascontiguousarray(theta * np.array(column.layers.thickness_m))  # (L,) host, L doubles
    _admit(column.memory, column.available_bytes_fn, "uniform state")
    from maple.core.backend import to_device

    xp = column.xp
    depth_xp = to_device(depth, xp)  # L doubles: a counted transfer on CUDA, the identity on NumPy
    water = xp.empty((column.n_layers, column.n_cells), dtype=np.float64)
    water[...] = depth_xp[:, None]
    return SoilColumnState(water, t)


def moisture(column: SoilColumn, state: SoilColumnState) -> Any:
    """Volumetric moisture theta = water / thickness: a fresh (L, N) array of the column namespace, no host read."""
    _check_column(column)
    _check_state(column, state)
    return state.water_m / column.thickness_xp[:, None]


def storage_m3(column: SoilColumn, state: SoilColumnState) -> float:
    """Total soil water over ACTIVE cells (m3): the same per-cell layer-sequential sum as the step rows, then the
    namespace's cell reduction, ONE counted host read. One transient (N,) array."""
    _check_column(column)
    _check_state(column, state)
    from maple.core.backend import to_float

    xp = column.xp
    total = xp.zeros(column.n_cells, dtype=np.float64)
    for layer in range(column.n_layers):
        xp.add(total, state.water_m[layer], out=total)
    xp.copyto(total, 0.0, where=~column.active)
    return to_float(xp.sum(total)) * column.cell_area_m2


# --- the reference step -------------------------------------------------------------------------------------------------
def reference_step(column: SoilColumn, water_in: Any, added: Any, pet: Any, dt: float) -> StepArrays:
    """NumPy reference of one step: loops over layers, vectorised over cells, the SAME IEEE operations in the SAME order
    as the compiled kernels (only `expm1` may differ by an ulp between libraries), evaluated IN PLACE on the fresh
    output arrays with a counted transient peak (see NUMPY_SCRATCH_DOUBLES_PER_CELL). Validation is recorded in flag
    rows, not raised here; `step` reads the packet once and raises. Inputs are never written."""
    from maple.core.backend import errstate

    xp = np
    lay = column.layers
    n_layers, n_cells = column.n_layers, column.n_cells
    n_if = n_layers - 1
    active = column.active
    inactive = ~active
    dz = lay.thickness_m
    res_m, fc_m, sat_m, cap_tol = lay.residual_m, lay.field_capacity_m, lay.saturation_m, lay.capacity_tolerance_m
    cond, pair_inv = lay.interface_conductance_m_s, lay.pair_inverse_thickness
    res_theta0, stress, factor = float(lay.theta_residual[0]), lay.theta_evaporation_stress, lay.evaporation_factor
    cell_tol = cell_closure_coefficient(n_layers)

    water_out = xp.array(water_in, dtype=np.float64, copy=True)
    percolation = xp.zeros((n_if, n_cells), dtype=np.float64)
    diffusion = xp.zeros((n_if, n_cells), dtype=np.float64)
    ret = xp.zeros(n_cells, dtype=np.float64)
    drain = xp.zeros(n_cells, dtype=np.float64)
    evap = xp.zeros(n_cells, dtype=np.float64)
    unmet = xp.zeros(n_cells, dtype=np.float64)
    rows = xp.zeros((ROW_COUNT, n_cells), dtype=np.float64)
    before, after, added_row = rows[0], rows[1], rows[2]  # accumulated in place: x = x + term, in layer order
    flags = xp.zeros(n_cells, dtype=np.int64)

    def flag(mask: Any, bit: int) -> None:
        xp.bitwise_or(flags, np.int64(1 << bit), out=flags, where=mask)

    with errstate(xp=xp, all="ignore"):
        for layer in range(n_layers):
            wl = water_in[layer]
            al = added[layer]
            flag(~xp.isfinite(wl), 0)
            flag(wl < 0.0, 1)
            flag(wl > sat_m[layer] + cap_tol[layer], 2)
            flag(~xp.isfinite(al), 3)
            flag(al < 0.0, 4)
            flag(inactive & (al != 0.0), 5)
            if dt == 0.0:
                flag(al != 0.0, 9)
            xp.add(before, wl, out=before)
            xp.add(added_row, al, out=added_row)
        flag(~xp.isfinite(pet), 6)
        flag(pet < 0.0, 7)
        flag(inactive & (pet != 0.0), 8)

        if dt > 0.0:
            # 2. addition with saturation return
            for layer in range(n_layers):
                w = water_out[layer]
                xp.add(w, added[layer], out=w)
                ex = w - sat_m[layer]
                pos = ex > 0.0
                xp.subtract(w, ex, out=w, where=pos)
                xp.add(ret, ex, out=ret, where=pos)
                del ex, pos
            # 3. gravity cascade, top to bottom within the step
            if lay.gravity_drainage:
                carry = xp.zeros(n_cells, dtype=np.float64)
                for layer in range(n_layers):
                    w = water_out[layer]
                    xp.add(w, carry, out=w)
                    ex = w - fc_m[layer]
                    pos = ex > 0.0
                    xp.subtract(w, ex, out=w, where=pos)
                    carry[...] = 0.0
                    xp.copyto(carry, ex, where=pos)
                    if layer < n_if:
                        percolation[layer] = carry
                    del ex, pos
                drain[...] = carry
                del carry
            # 4. evaporation from layer 0
            w0 = water_out[0]
            theta0 = w0 / dz[0]
            multiplier = theta0 / stress
            xp.multiply(multiplier, multiplier, out=multiplier)  # ratio * ratio
            xp.copyto(multiplier, 1.0, where=~(theta0 < stress))  # where(theta0 < stress, r r, 1.0)
            xp.copyto(multiplier, 0.0, where=theta0 <= res_theta0)  # where(theta0 <= res, 0.0, ...)
            demand = factor * pet
            xp.multiply(demand, dt, out=demand)
            xp.multiply(demand, multiplier, out=demand)  # ((factor E) dt) m
            del multiplier
            avail0 = w0 - res_m[0]
            xp.copyto(avail0, 0.0, where=avail0 < 0.0)
            xp.copyto(evap, avail0)
            xp.copyto(evap, demand, where=demand <= avail0)
            xp.subtract(w0, evap, out=w0)
            xp.subtract(demand, evap, out=unmet)
            del theta0, demand, avail0
            # 5. sequential pair diffusion from the top
            for k in range(n_if):
                wi, wj = water_out[k], water_out[k + 1]
                s = pair_inv[k]
                rate = cond[k] * s
                x = rate * dt
                e = -xp.expm1(-x)
                amount = wi / dz[k]
                tj = wj / dz[k + 1]
                xp.subtract(amount, tj, out=amount)
                del tj
                xp.divide(amount, s, out=amount)
                xp.multiply(amount, e, out=amount)  # ((ti - tj) / s) e
                down = amount > 0.0
                act = -amount
                xp.copyto(act, amount, where=down)  # where(down, amount, -amount)
                del amount
                avail = xp.where(down, wi, wj)
                xp.subtract(avail, xp.where(down, res_m[k], res_m[k + 1]), out=avail)  # wd - resd
                xp.copyto(avail, 0.0, where=avail < 0.0)
                cap = xp.where(down, sat_m[k + 1], sat_m[k])
                xp.subtract(cap, xp.where(down, wj, wi), out=cap)  # satr - wr
                xp.copyto(cap, 0.0, where=cap < 0.0)
                xp.copyto(act, avail, where=avail < act)
                xp.copyto(act, cap, where=cap < act)
                del avail, cap
                signed = diffusion[k]
                xp.negative(act, out=signed)
                xp.copyto(signed, act, where=down)  # where(down, act, -act)
                del act, down
                xp.subtract(wi, signed, out=wi)  # down: wi - act; up: wi - (-act) == wi + act bit for bit
                xp.add(wj, signed, out=wj)  # down: wj + act; up: wj + (-act) == wj - act bit for bit
            # 6. output checks (active cells only)
            for layer in range(n_layers):
                wl = water_out[layer]
                flag(active & ~xp.isfinite(wl), 10)
                flag(active & (wl < 0.0), 11)
                flag(active & (wl > sat_m[layer] + cap_tol[layer]), 12)
                xp.add(after, wl, out=after)
            flux = added_row - ret
            xp.subtract(flux, drain, out=flux)
            xp.subtract(flux, evap, out=flux)  # ((added - ret) - drain) - evap
            closure = after - before
            xp.subtract(closure, flux, out=closure)
            scale = before + after
            xp.add(scale, added_row, out=scale)
            xp.add(scale, ret, out=scale)
            xp.add(scale, drain, out=scale)
            xp.add(scale, evap, out=scale)
            xp.multiply(scale, cell_tol, out=scale)  # cell_tol * scale
            xp.abs(closure, out=flux)
            flag(active & (flux > scale), 13)
            nonfinite = ~xp.isfinite(ret)
            for vector in (drain, evap, unmet, closure):
                nonfinite |= ~xp.isfinite(vector)
            for k in range(n_if):
                nonfinite |= ~xp.isfinite(percolation[k])
                nonfinite |= ~xp.isfinite(diffusion[k])
            flag(active & nonfinite, 14)
            del flux, closure, scale, nonfinite
            # inactive cells: identity state, zero fluxes
            for layer in range(n_layers):
                xp.copyto(water_out[layer], water_in[layer], where=inactive)
            for k in range(n_if):
                xp.copyto(percolation[k], 0.0, where=inactive)
                xp.copyto(diffusion[k], 0.0, where=inactive)
            for vector in (ret, drain, evap, unmet):
                xp.copyto(vector, 0.0, where=inactive)
        else:
            after[...] = before
        xp.copyto(before, 0.0, where=inactive)
        xp.copyto(after, 0.0, where=inactive)
        xp.copyto(added_row, 0.0, where=inactive)
        low = xp.negative(flags)
        xp.bitwise_and(flags, low, out=low)  # flags & -flags
        rows[3][...] = low
        xp.copyto(rows[3], NO_FLAG, where=flags == 0)
        del low
    return StepArrays(water_out, percolation, diffusion, ret, drain, evap, unmet, rows)


def reference_runner(column: SoilColumn, water_in: Any, added: Any, pet: Any, dt: float) -> StepArrays:
    return reference_step(column, water_in, added, pet, dt)


# --- step ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StepBudget:
    """Host volumes (m3) over active cells from ONE packet read, and the MAPLE-derived closure bound."""

    storage_before_m3: float
    storage_after_m3: float
    added_m3: float
    surface_return_m3: float
    drainage_m3: float
    evaporation_m3: float
    evaporation_unmet_m3: float
    residual_m3: float
    bound_m3: float
    n_terms: int


@dataclass(frozen=True, eq=False)
class SubsurfaceStep:
    state: SoilColumnState
    dt_s: float
    surface_return_m: Any  # (N,)
    drainage_m: Any  # (N,) bottom drainage leaving the profile
    evaporation_m: Any  # (N,) actual withdrawal from layer 0
    evaporation_unmet_m: Any  # (N,) demand not met (water at/below residual)
    percolation_m: Any  # (L - 1, N) cascade flow across each interface
    diffusion_m: Any  # (L - 1, N) diffusive exchange, positive downward
    budget: StepBudget


def _read_packet(xp: ModuleType, arrays: StepArrays) -> np.ndarray:
    """The batched reduction packet: sums of the rows and flux vectors, the minimum failure bit and the flagged cell
    count, stacked in the namespace and brought to the host in ONE counted read."""
    from maple.core.backend import to_host

    rows = arrays.rows
    values = [xp.sum(rows[0]), xp.sum(rows[1]), xp.sum(rows[2]), xp.sum(arrays.surface_return_m),
              xp.sum(arrays.drainage_m), xp.sum(arrays.evaporation_m), xp.sum(arrays.evaporation_unmet_m),
              xp.min(rows[3]), xp.count_nonzero(rows[3] < NO_FLAG)]
    stacked = xp.stack([xp.asarray(v, dtype=np.float64) for v in values])
    return np.asarray(to_host(stacked), dtype=np.float64)


def _resolve(column: SoilColumn, packet: np.ndarray) -> StepBudget:
    if packet.shape != (PACKET_LENGTH,):
        raise SubsurfaceError(f"internal: packet shape {packet.shape}")
    low, n_flagged = float(packet[7]), int(packet[8])
    if not math.isfinite(low) or low < NO_FLAG:
        bit = int(low).bit_length() - 1 if math.isfinite(low) and low >= 1.0 else None
        message = FLAG_MESSAGES[bit] if bit is not None and 0 <= bit < N_FLAGS else "validation packet is corrupt"
        raise SubsurfaceStepError(f"{message} ({n_flagged} cell(s) flagged)", bit=bit, n_flagged=n_flagged)
    area = column.cell_area_m2
    before, after, added, ret, drain, evap, unmet = (float(v) * area for v in packet[:7])
    residual = (after - before) - (((added - ret) - drain) - evap)
    n_terms = 4 * column.n_layers * column.n_cells + 7
    operands = (abs(before), abs(after), abs(added), abs(ret), abs(drain), abs(evap))
    if not all(math.isfinite(v) for v in (*operands, unmet, residual)):
        raise SubsurfaceStepError("non-finite global water budget", bit=None, n_flagged=n_flagged)
    bound = volume_roundoff_bound_m3(n_terms, max(operands))
    if abs(residual) > bound:
        raise SubsurfaceStepError(
            f"global water closure violated: residual {residual!r} m3 exceeds the FP64 bound {bound!r} m3 "
            f"({n_terms} terms)", bit=None, n_flagged=n_flagged)
    return StepBudget(before, after, added, ret, drain, evap, unmet, residual, bound, n_terms)


def step(column: SoilColumn, state: SoilColumnState, dt_s: Any, *, added_water_m: Any,
         potential_evaporation_m_s: Any) -> SubsurfaceStep:
    """One pure step (see the module docstring for the stage order). `added_water_m` (L, N) is water ACTUALLY directed
    into each layer during `dt_s` (matrix intake in layer 0, separately admitted bypass in its target layer), counted
    once; `potential_evaporation_m_s` (N,) is the supplied potential rate (a forcing, not an actual flux). Both are 0
    on inactive cells. The source time and the candidate time `t + dt` must be finite. Inputs are never written; a
    refused step allocates no result."""
    _check_column(column)
    t0 = _check_state(column, state)
    dt = _check_dt(dt_s)
    t1 = t0 + dt
    if not math.isfinite(t1):
        raise SubsurfaceError(f"the candidate time t + dt = {t0!r} + {dt!r} is not finite; the step is refused before "
                              "any numerical work (state untouched)")
    shape = (column.n_layers, column.n_cells)
    _check_dynamic(column, "added_water_m", added_water_m, shape)
    _check_dynamic(column, "potential_evaporation_m_s", potential_evaporation_m_s, (column.n_cells,))
    if column.preflight is not None:
        column.preflight()
    admit_step(column)
    arrays = column.runner(column, state.water_m, added_water_m, potential_evaporation_m_s, dt)
    packet = _read_packet(column.xp, arrays)  # synchronising read: queued kernels have finished with any scratch
    budget = _resolve(column, packet)
    # `arrays.scratch` (CUDA factor keep-alive) dies with this frame; the published step never carries it.
    return SubsurfaceStep(state=SoilColumnState(arrays.water_m, t1), dt_s=dt,
                          surface_return_m=arrays.surface_return_m, drainage_m=arrays.drainage_m,
                          evaporation_m=arrays.evaporation_m, evaporation_unmet_m=arrays.evaporation_unmet_m,
                          percolation_m=arrays.percolation_m, diffusion_m=arrays.diffusion_m, budget=budget)


# --- streaming initialisation -------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ForcingRecord:
    """One explicit interval: `dt_s`, water actually directed to each layer (L, N) and the potential evaporation rate
    (N,). Records are consumed one at a time and never retained."""

    dt_s: float
    added_water_m: Any
    potential_evaporation_m_s: Any


@dataclass(frozen=True)
class CumulativeBudget:
    n_steps: int = 0
    elapsed_s: float = 0.0
    added_m3: float = 0.0
    surface_return_m3: float = 0.0
    drainage_m3: float = 0.0
    evaporation_m3: float = 0.0
    evaporation_unmet_m3: float = 0.0

    def accumulate(self, budget: StepBudget, dt_s: float) -> CumulativeBudget:
        """The sums after one more accepted step; every accumulated scalar (elapsed time included) must stay finite,
        otherwise `SubsurfaceError` is raised and `self` remains the accepted budget."""
        out = CumulativeBudget(self.n_steps + 1, self.elapsed_s + dt_s, self.added_m3 + budget.added_m3,
                               self.surface_return_m3 + budget.surface_return_m3,
                               self.drainage_m3 + budget.drainage_m3, self.evaporation_m3 + budget.evaporation_m3,
                               self.evaporation_unmet_m3 + budget.evaporation_unmet_m3)
        for name in ("elapsed_s", "added_m3", "surface_return_m3", "drainage_m3", "evaporation_m3",
                     "evaporation_unmet_m3"):
            if not math.isfinite(getattr(out, name)):
                raise SubsurfaceError(f"cumulative {name} overflowed FP64 after {out.n_steps} steps")
        return out


@dataclass(frozen=True, eq=False)
class ProfileInitialization:
    state: SoilColumnState
    cumulative: CumulativeBudget
    storage_initial_m3: float
    storage_final_m3: float
    residual_m3: float
    bound_m3: float
    n_terms: int


def initialize_profile(column: SoilColumn, state: SoilColumnState, records: Iterable[ForcingRecord]) -> ProfileInitialization:
    """Advance `state` through an iterable of explicit forcing records (a generator is consumed lazily), carrying only
    the current state and the cumulative scalar budget: no per-step history is stored, and the previous step's result
    (its flux arrays) is released before the next record is advanced, so only the current state array, the caller's
    own inputs and the scalars are live across records. The whole-sequence closure uses the MAPLE bound for
    4 L N n_steps + 7 terms. Any failure, including an overflowing cumulative scalar, raises `SubsurfaceStreamError`
    with the completed step count, the last ACCEPTED state and the prior accepted budget; the caller's inputs are
    untouched. There is no profile from a scalar total."""
    _check_column(column)
    _check_state(column, state)
    storage0 = storage_m3(column, state)
    cumulative = CumulativeBudget()
    current = state
    completed = 0
    storage_final = storage0
    for record in records:
        if not isinstance(record, ForcingRecord):
            raise SubsurfaceStreamError(f"record {completed} is not a ForcingRecord ({type(record).__name__})",
                                        completed_steps=completed, last_state=current, cumulative=cumulative)
        try:
            result = step(column, current, record.dt_s, added_water_m=record.added_water_m,
                          potential_evaporation_m_s=record.potential_evaporation_m_s)
        except (SubsurfaceError, SubsurfaceMemoryError, SubsurfaceBackendUnavailableError) as exc:
            raise SubsurfaceStreamError(f"record {completed} failed: {exc}", completed_steps=completed,
                                        last_state=current, cumulative=cumulative) from exc
        budget, next_state, dt = result.budget, result.state, result.dt_s
        del result  # release the flux arrays now; only the new state array survives
        try:
            next_cumulative = cumulative.accumulate(budget, dt)
        except SubsurfaceError as exc:
            del next_state  # the step is not published: last accepted state and budget are carried
            raise SubsurfaceStreamError(f"record {completed}: {exc}", completed_steps=completed,
                                        last_state=current, cumulative=cumulative) from exc
        cumulative, current, storage_final = next_cumulative, next_state, budget.storage_after_m3
        completed += 1
        del budget, next_state, next_cumulative
    c = cumulative
    residual = (storage_final - storage0) - (((c.added_m3 - c.surface_return_m3) - c.drainage_m3) - c.evaporation_m3)
    n_terms = 4 * column.n_layers * column.n_cells * max(completed, 1) + 7
    scale = max(abs(storage0), abs(storage_final), c.added_m3, c.surface_return_m3, c.drainage_m3, c.evaporation_m3)
    bound = volume_roundoff_bound_m3(n_terms, scale)
    if not math.isfinite(residual) or abs(residual) > bound:
        raise SubsurfaceStreamError(f"cumulative water closure violated over {completed} step(s): residual {residual!r} "
                                    f"m3 exceeds the FP64 bound {bound!r} m3", completed_steps=completed,
                                    last_state=current, cumulative=cumulative)
    return ProfileInitialization(current, cumulative, storage0, storage_final, residual, bound, n_terms)


# --- provenance ------------------------------------------------------------------------------------------------------
def summary(column: SoilColumn) -> dict[str, Any]:
    _check_column(column)
    memory = {k: getattr(column.memory, k) for k in ("input_bytes", "column_bytes", "layer_bytes", "parameter_bytes",
                                                     "output_bytes", "scratch_bytes", "reserve_bytes", "host_bytes",
                                                     "device_bytes", "required_bytes", "step_allocation_bytes",
                                                     "total_bytes", "scope")}
    return {
        "backend": column.backend,
        "namespace": column.xp.__name__,
        "device_id": column.device_id,
        "n_layers": column.n_layers,
        "n_cells": column.n_cells,
        "n_active": column.n_active,
        "cell_area_m2": column.cell_area_m2,
        "layers": column.layers.summary(),
        "memory": memory,
        "admission": "required_bytes compared with available_bytes_fn before prepare_column, column_state, "
                     "uniform_state and every step; a reading, not a reservation",
        "closure": "conservation.volume_roundoff_bound_m3: per cell 4 L + 7 terms, global 4 L N + 7 terms, capacity "
                   f"{LAYER_OPERATION_TERMS} terms on the layer saturation depth",
        "packet_length": PACKET_LENGTH,
        "host_reads_per_step": 1,
        "backend_info": dict(column.backend_info),
        "scope": "isolated forced column: no storm coupling, no plant uptake, no checkpoint, no LEAF interface, no "
                 "moving-bed remapping; added water is accepted soil intake, not rainfall",
    }
