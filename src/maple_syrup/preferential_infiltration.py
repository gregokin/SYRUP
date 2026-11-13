"""Opt-in continuous preferential (bypass) infiltration for the SYRUP layered soil column (Phase 1 component).

SYRUP owns all physical water. This module is ONE staged operation of a future storm hydraulic step: it moves water
from the cell's standing surface water `H` into the configured receiving soil layers of `maple_syrup.subsurface`
(the authoritative layered inventory), bounded by the shared surface supply, ONE aggregate maximum intake rate and
every receiver's actual saturation capacity. It performs no matrix (Green-Ampt) infiltration, no gravity/diffusive
redistribution, no evaporation/transpiration/growth and no surface routing; the later storm operator applies this
preferential stage first and the matrix stage second (reference source order). There is no second owned water
inventory: the surface array and the `SoilColumnState` passed in are the caller's, and the published result holds
fresh arrays only.

    column = subsurface.prepare_column(layers, active_mask, cell_area_m2, backend=...)
    op = prepare_preferential(column, a3=reference_a3(k_geom_m_s), max_fraction_per_hour=maxFL2,
                              max_intake_rate_m_s=mm_per_hour_to_m_s(maxAmountFL2),
                              background_receiver_weights=[0.0, 1.0],             # soil pathway, or None
                              species_receiver_weights=[[0.0, 1.0], [0.0, 0.0]],  # LEAF traits per species
                              species_contributions=iter_of_(N,)_arrays)         # LEAF per-cell c_s, streamed
    result = preferential_step(op, surface_water_m, state, dt_s)

Reference (Tietjen-lineage EcoHyD descendant, `WaterCell::twoLayerInfiltration`, Dryad WaterCell.cpp lines
305-325; GitHub 354-369): a3 = 0.1 / 0.5 / 0.8 for K_geom < 1 mm/h, < 10 mm/h, else; shrubFunction =
(a3 + C) / (1 + C) with C the cover-weighted lower-layer root sum; FL2 = min(H x shrubFunction x maxFL2,
maxAmountFL2) applied ONCE PER HOUR before matrix infiltration. The descendants are reference evidence, not verified
original HydroVeg releases.

Approved generalisation (user decisions 2026-11-10, docs/subsurface_hydrology/development_phases_20261109.md):

  * L user layers. ONE explicit background soil receiving-weight vector b (optional; `None` or all zeros DISABLES the
    background pathway explicitly, nothing is invented) and one LEAF species receiving-weight vector u_s per species.
    Weights are finite, nonnegative, need not sum to one; a positive vector is normalised for allocation, an all-zero
    vector disables that species' contribution (`[0,1,1]` sends equal shares to layers 2 and 3). These are NOT root
    uptake fractions and NOT the canopy/matrix vegetation functions; nothing here guesses species types.
  * Per-cell species contributions c_s >= 0 come from LEAF; C = sum over ENABLED species only (a species with all-zero
    weights contributes nothing even with positive cover).
  * Layer map g_l = a3 b_l + (1 - a3) / (1 + C) sum_s c_s u_sl. With the background enabled sum_l g_l = (a3 + C) /
    (1 + C), which is the reference two-layer response when every vector is [0, 1]. With the background disabled the
    a3 term is absent and sum_l g_l = (1 - a3) C / (1 + C): the species enhancement alone. The module evaluates
    sum_l g_l in that closed form (so beta <= 1 holds exactly in FP64) and uses g_l only for the receiver split
    r_l = g_l / sum_l g_l; it checks at build time that the layer sum agrees with the closed form within the
    MAPLE-derived FP64 bound.
  * beta = maxFL2 x sum_l g_l is the reference HOURLY fraction of the standing surface water (0 <= beta <= 1). The
    user chose the equivalent continuous rate k = -log1p(-beta) / 3600 s^-1 (k = +inf for beta = 1), so the isolated
    attempted amount over dt is H x (-expm1(-k dt)): beta = 0 -> 0, beta = 1 -> all of H for dt > 0, dt = 0 -> the
    identity. One isolated hour reproduces beta H exactly; repeated sub-steps of one hour reproduce it too (the
    survival factors multiply). The same hourly fraction is never re-applied at every shorter hydraulic step.
  * ONE aggregate rate cap: attempted = min(H x (-expm1(-k dt)), max_intake_rate_m_s x dt). The attempted amount is
    allocated as attempted x r_l to the receivers, top layer first, each receiver bounded by its saturation room
    (saturation_m - water) and by the surface water still standing (which only binds against roundoff because the
    allocation never exceeds H). A rejected share stays on the surface; it is NOT redistributed to another layer. The
    sum of the actual layer intakes is subtracted exactly once from H and added exactly once to the soil.
  * No clipping source, no per-species background allowance, no per-species cap: background and vegetation share one
    supply, one cap and one capacity bound.

Closure: per cell (active cells) |(soil_after - soil_before) - intake| and |(H_in - H_out) - intake| are bounded by
MAPLE's `conservation.volume_roundoff_bound_m3` coefficient for 4 L + 4 terms times the magnitude scale; the global
closure uses (4 L + 4) N + 8 terms on the largest operand. No new empirical epsilon.

Backends: "numpy" (reference: layer loop, vectorised cells, in place on the fresh outputs), "numba" (compiled serial
cell/layer loops, `preferential_infiltration_numba`), "cuda" (one CuPy RawKernel thread per cell,
`preferential_infiltration_cuda`). A missing backend raises `PreferentialBackendUnavailableError`, never a fallback.
Dynamic arrays must be exactly the column namespace's `ndarray` (no subclass, no masked array, no other namespace,
no implicit transfer), float64, C-contiguous and exactly shaped; the caller's buffers are read-only and the result is
fresh. An invalid input, dt or refused step leaves the inputs untouched and publishes nothing.

Sealed prepared parameters (correction 01): `ReceiverWeights` and `PreferentialOperator` are factory-only records
(`receiver_weights`, `prepare_preferential`); direct construction, subclass instances, attribute assignment and
forged/inconsistent records are refused, and every `preferential_step` re-validates the operator's record against
its sealed arrays with metadata-only checks (exact types, dtypes, shapes, contiguity, read-only state, device; no
hashing, no copy). The parameter maps are OWNED: on NumPy the storage itself is a read-only `bytes`-backed buffer,
so neither a public view nor any `.base` in its chain can ever be made writable; on CuPy, which has no read-only
mode, the resident maps are private and public access (`maps`, `beta`, `host_maps`) returns detached copies or the
sealed host mirrors. The kernels use the private zero-copy handles. A SHA-256 identity of the host bytes of every
map AND beta is recorded once at build (`sha256`, `identity()`); `verify_identity()` re-hashes explicitly (cold; one
download per map on CUDA) and is never called per step. Public summaries are fresh dictionaries, never the record.

Memory: `estimate_preferential_memory` is the documented upper bound of the arrays this module holds for one step
(the caller's surface/soil arrays, the owned parameter maps, the fresh outputs, the reference temporaries, the
packet), the owned per-species receiver-weight vectors (2 (S + 1) L host doubles), and the build's transient scratch
(including the sealing copy of the largest map), plus the column's reserve and the caller's extra reserve. The
admitted side (host for numpy/numba, device for cuda) is compared with the column's `available_bytes_fn` BEFORE every
allocation in `prepare_preferential` (step requirement + build scratch, before even the small weight vectors) and
`preferential_step`. A reading is not a reservation, and the budget is never raised silently. Per step: no host
hashing, no upload, ONE counted host read of the batched packet. Build-time checks are cold and may synchronise.

Nothing here was run against a GPU by its author; Codex records executions.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Any, NamedTuple

import numpy as np

from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = [
    "A3_CONDUCTIVITY_BOUNDS_MM_H",
    "A3_CONDUCTIVITY_BOUNDS_M_S",
    "A3_VALUES",
    "BACKENDS",
    "BUILD_SCRATCH_DOUBLES_PER_CELL",
    "BUILD_SCRATCH_MASK_BYTES_PER_CELL",
    "CELL_OPERATION_TERMS",
    "FLAG_MESSAGES",
    "GARDNER_EXPONENT",
    "GARDNER_SUCTION_RATIO",
    "KERNEL_SCRATCH_DOUBLES_PER_CELL",
    "MEMORY_SCOPE",
    "NO_FLAG",
    "NUMPY_SCRATCH_DOUBLES_PER_CELL",
    "PACKET_LENGTH",
    "ROW_COUNT",
    "SECONDS_PER_HOUR",
    "SOURCE_IDENTITY",
    "STAGE_ORDER",
    "THREADS",
    "KernelMaps",
    "PreferentialBackendUnavailableError",
    "PreferentialBudget",
    "PreferentialError",
    "PreferentialMemory",
    "PreferentialMemoryError",
    "PreferentialOperator",
    "PreferentialStep",
    "PreferentialStepError",
    "ReceiverWeights",
    "StepArrays",
    "SubsurfaceBackendUnavailableError",
    "SubsurfaceError",
    "SubsurfaceMemoryError",
    "admit_step",
    "cell_closure_coefficient",
    "estimate_preferential_memory",
    "geometric_conductivity_m_s",
    "hourly_fraction_to_rate_s",
    "isolated_intake_fraction",
    "mm_per_hour_to_m_s",
    "preferential_step",
    "prepare_preferential",
    "receiver_weights",
    "reference_a3",
    "reference_a3_from_mm_per_hour",
    "reference_step",
    "reference_two_layer_hourly_fraction",
    "summary",
    "unsaturated_conductivity_m_s",
]

BACKENDS = ss.BACKENDS
# The native column's error family, re-exported so callers catching the subsurface family catch this stage too.
SubsurfaceError = ss.SubsurfaceError
SubsurfaceMemoryError = ss.SubsurfaceMemoryError
SubsurfaceBackendUnavailableError = ss.SubsurfaceBackendUnavailableError
STAGE_ORDER = ("validation", "isolated_attempt", "rate_cap", "receiver_allocation", "output_checks")
SECONDS_PER_HOUR = 3600.0
THREADS = 128
# Reference constants (Dryad WaterCell.cpp 311-314, 490-495; GitHub WaterCell.cpp equivalent lines).
A3_VALUES = (0.1, 0.5, 0.8)
A3_CONDUCTIVITY_BOUNDS_MM_H = (1.0, 10.0)
GARDNER_SUCTION_RATIO = 30.0 / 26.0  # Ku = Ks / (1 + 30/26)^3  (Kemp et al. 1997 p. 81 after Gardner 1958)
GARDNER_EXPONENT = 3.0
SOURCE_IDENTITY = {
    "dryad_watercell": "reference_models/ecohyd/dryad_4j0zpc8gn/extracted/Irob_et_al_drought_resilience/Model/Source/"
                       "WaterCell.cpp: a3 selection 311-314, shrubFunction 319, FL2 = min(H shrubFunction maxFL2, "
                       "maxAmountFL2) 321, surface debit 324, Ku/K_geom 492-495 (hourly call, before matrix intake)",
    "audits": ("agent_handoffs/tasks/shared_scheduler_phase0_20261109/phase1_source_numerical_audit.md",
               "agent_handoffs/tasks/shared_scheduler_phase0_20261109/independent_tietjen_audit.md"),
    "decisions": "docs/subsurface_hydrology/development_phases_20261109.md (Phase 1, 2026-11-10 user decisions)",
    "status": "descendant sources are reference evidence, not verified original HydroVeg releases",
}
# FP accumulations one cell undergoes in one step: per layer the allocation, room, intake add, surface debit (4 L),
# plus the attempted product, the rate cap, the surface difference and the closure (4).
CELL_OPERATION_TERMS_PER_LAYER = 4
CELL_OPERATION_TERMS = 4
NO_FLAG = ss.NO_FLAG
ROW_COUNT = 8  # soil before, soil after, surface before, surface after, isolated attempt, capped attempt, intake, flag
PACKET_LENGTH = 9
NUMPY_SCRATCH_DOUBLES_PER_CELL = 16
KERNEL_SCRATCH_DOUBLES_PER_CELL = 2
SCRATCH_MASK_BYTES_PER_CELL = 8
# Build scratch beyond the persistent maps (see the scratch plan in `prepare_preferential`): cover, denominator,
# layer sum and the a3 map (4 doubles per cell) plus transient boolean masks (declared as 8 bytes per cell). Sealing
# the largest map (receiver, L N doubles) holds the computed array and its bytes copy at once: + L N doubles.
BUILD_SCRATCH_DOUBLES_PER_CELL = 4
BUILD_SCRATCH_MASK_BYTES_PER_CELL = 8
_PACKET_SCRATCH_BYTES = 4 * PACKET_LENGTH * 8
_F8 = 8
MEMORY_SCOPE = (
    "arrays held for ONE preferential step: the caller's surface (N) and soil (L, N) arrays (known inputs), the "
    "operator's owned parameter maps (beta N, rate N, max rate N, receiver L N, saturation L, capacity tolerance L; "
    "sealed host arrays on numpy/numba, private device arrays plus sealed host mirrors on cuda), the owned "
    "per-species receiver-weight vectors (raw and normalised, 2 (S + 1) L host doubles including the background), "
    "the fresh outputs (surface N, soil L N, intake L N, intake total N, rows 8 N), the transient reference "
    "temporaries and the packet, plus the column reserve and the caller's extra reserve; the build additionally "
    "holds its transient contribution sum, denominator, layer sum and a3 map (4 N doubles), boolean masks and the "
    "sealing copy of the largest map (L N doubles), while its layer numerator, beta and rate buffers BECOME the "
    "persistent maps. NOT included: allocator pools, the CUDA context, Numba/NVRTC JIT workspaces, Python object "
    "headers, the surface routing state, LEAF or the remaining process memory. An upper bound on what this module "
    "allocates, not a reservation. On cuda the host side (mirrors, weights, packet) is reported in host_bytes and is "
    "not compared with the device reading."
)

FLAG_MESSAGES = (
    "soil water state must be finite everywhere",  # 0
    "soil water state must be >= 0 everywhere",  # 1
    "soil water state exceeds the layer saturation depth",  # 2
    "surface_water_m must be finite everywhere",  # 3
    "surface_water_m must be >= 0 everywhere",  # 4
    "step produced non-finite soil or surface water",  # 5
    "step produced negative soil or surface water",  # 6
    "step produced soil water above the layer saturation depth",  # 7
    "per-cell soil closure violated beyond the FP64 bound",  # 8
    "per-cell surface closure violated beyond the FP64 bound",  # 9
    "step produced a non-finite intake",  # 10
)
N_FLAGS = len(FLAG_MESSAGES)

#: Module-private construction seal: only this module can mint receiver-weight records and operators.
_SEAL = object()


# --- errors ----------------------------------------------------------------------------------------------------------
class PreferentialError(ss.SubsurfaceError):
    """Invalid weights, contributions, parameters, arrays, dt, option or record. Raised before any result exists."""


class PreferentialStepError(PreferentialError):
    """A step's batched validation recorded a failure (lowest bit first) or the global closure failed."""

    def __init__(self, message: str, *, bit: int | None = None, n_flagged: int = 0):
        super().__init__(message)
        self.bit = bit
        self.n_flagged = int(n_flagged)


class PreferentialBackendUnavailableError(ss.SubsurfaceBackendUnavailableError):
    """The requested backend is unavailable. No fallback."""


class PreferentialMemoryError(ss.SubsurfaceMemoryError):
    """The working-array estimate plus the reserves exceeds the reported available bytes (nothing allocated)."""


# --- unit and reference helpers (host scalars or host NumPy arrays) -----------------------------------------------
def _host_values(value: Any, name: str, *, allow_inf: bool = False) -> tuple[np.ndarray, bool]:
    """Real host scalar or NumPy array as float64; returns (array, was_scalar). Booleans and non-real refused."""
    if isinstance(value, (bool, np.bool_)):
        raise PreferentialError(f"{name} must be numeric, got a Boolean")
    if isinstance(value, (int, float, np.integer, np.floating)):
        arr, scalar = np.array(float(value), dtype=np.float64), True
    else:
        if type(value) is not np.ndarray:
            raise PreferentialError(f"{name} must be a real host number or a numpy.ndarray, got {type(value).__name__}")
        if value.dtype.kind not in "iuf":
            raise PreferentialError(f"{name} must be real numbers, got dtype {value.dtype}")
        arr, scalar = np.array(value, dtype=np.float64), False
    if allow_inf:
        if np.any(np.isnan(arr)):
            raise PreferentialError(f"{name} must not be NaN")
    elif not np.all(np.isfinite(arr)):
        raise PreferentialError(f"{name} must be finite")
    return arr, scalar


def _emit(arr: np.ndarray, scalar: bool) -> Any:
    return float(arr) if scalar else arr


def mm_per_hour_to_m_s(value: Any) -> Any:
    """`value` [mm/h] to [m/s]: (value x 1e-3) / 3600. The a3 boundaries use THIS conversion, so a caller who
    converts reference conductivities with it compares against exactly the same FP64 numbers."""
    arr, scalar = _host_values(value, "value_mm_per_hour", allow_inf=True)
    return _emit((arr * 1e-3) / SECONDS_PER_HOUR, scalar)


A3_CONDUCTIVITY_BOUNDS_M_S = tuple(mm_per_hour_to_m_s(v) for v in A3_CONDUCTIVITY_BOUNDS_MM_H)


def unsaturated_conductivity_m_s(ks_m_s: Any) -> Any:
    """Reference Ku = Ks / (1 + 30/26)^3 (WaterCell.cpp 492), units of the input (SI m/s here)."""
    ks, scalar = _host_values(ks_m_s, "ks_m_s")
    if np.any(ks < 0.0):
        raise PreferentialError("ks_m_s must be >= 0")
    return _emit(ks / (1.0 + GARDNER_SUCTION_RATIO) ** GARDNER_EXPONENT, scalar)


def geometric_conductivity_m_s(ks_m_s: Any) -> Any:
    """Reference K_geom = sqrt(Ks x Ku) (WaterCell.cpp 493-495)."""
    ks, scalar = _host_values(ks_m_s, "ks_m_s")
    if np.any(ks < 0.0):
        raise PreferentialError("ks_m_s must be >= 0")
    ku = ks / (1.0 + GARDNER_SUCTION_RATIO) ** GARDNER_EXPONENT
    return _emit(np.sqrt(ks * ku), scalar)


def reference_a3(k_geom_m_s: Any) -> Any:
    """Reference a3: 0.1 if K_geom < 1 mm/h, 0.5 if K_geom < 10 mm/h, else 0.8 (WaterCell.cpp 311-314), with the
    comparison on SI m/s against the exactly converted boundaries `A3_CONDUCTIVITY_BOUNDS_M_S`. Separate from the
    explicit `a3` input of `prepare_preferential`: a caller may supply any a3 in [0, 1]."""
    k, scalar = _host_values(k_geom_m_s, "k_geom_m_s")
    if np.any(k < 0.0):
        raise PreferentialError("k_geom_m_s must be >= 0")
    low, high = A3_CONDUCTIVITY_BOUNDS_M_S
    out = np.where(k < low, A3_VALUES[0], np.where(k < high, A3_VALUES[1], A3_VALUES[2])).astype(np.float64)
    return _emit(out, scalar)


def reference_a3_from_mm_per_hour(k_geom_mm_h: Any) -> Any:
    """The same selection in the reference's own units (K_geom in mm/h compared with 1 and 10)."""
    k, scalar = _host_values(k_geom_mm_h, "k_geom_mm_h")
    if np.any(k < 0.0):
        raise PreferentialError("k_geom_mm_h must be >= 0")
    low, high = A3_CONDUCTIVITY_BOUNDS_MM_H
    out = np.where(k < low, A3_VALUES[0], np.where(k < high, A3_VALUES[1], A3_VALUES[2])).astype(np.float64)
    return _emit(out, scalar)


def reference_two_layer_hourly_fraction(a3: float, cover_sum: float, max_fraction_per_hour: float) -> float:
    """The reference's hourly fraction of standing surface water: maxFL2 x (a3 + C) / (1 + C) (WaterCell.cpp
    319-321 before the maxAmountFL2 cap). For documentation and tests."""
    a = ss._real(a3, "a3")
    c = ss._real(cover_sum, "cover_sum")
    f = ss._real(max_fraction_per_hour, "max_fraction_per_hour")
    if not (0.0 <= a <= 1.0) or c < 0.0 or not (0.0 <= f <= 1.0):
        raise PreferentialError("a3 and max_fraction_per_hour must lie in [0, 1], cover_sum must be >= 0")
    return f * ((a + c) / (1.0 + c))


def hourly_fraction_to_rate_s(beta: Any) -> Any:
    """k = -log1p(-beta) / 3600 [s^-1]; beta = 1 gives +inf (all standing water over any dt > 0)."""
    b, scalar = _host_values(beta, "beta")
    if np.any(b < 0.0) or np.any(b > 1.0):
        raise PreferentialError("beta must lie in [0, 1]")
    with np.errstate(divide="ignore"):
        k = -np.log1p(-b) / SECONDS_PER_HOUR
    return _emit(k, scalar)


def isolated_intake_fraction(beta: Any, dt_s: Any) -> Any:
    """Fraction of the standing surface water removed in isolation over dt: -expm1(-k dt). beta = 0 -> 0,
    beta = 1 -> 1 for dt > 0, dt = 0 -> 0 (identity). One hour returns beta exactly up to FP64 rounding."""
    dt = ss._check_dt(dt_s)
    k, scalar = _host_values(hourly_fraction_to_rate_s(beta), "rate", allow_inf=True)
    if dt == 0.0:
        return _emit(np.zeros_like(k), scalar)
    with np.errstate(invalid="ignore"):
        f = -np.expm1(-(k * dt))
    return _emit(f, scalar)


# --- sealed owned arrays -------------------------------------------------------------------------------------------------
def _sealed_host(array: Any) -> np.ndarray:
    """A float64 C-contiguous host array whose storage is an immutable `bytes` buffer: NumPy refuses to set the
    WRITEABLE flag on it, on any view of it and on any `.base` in the chain. One copy (cold)."""
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    flat = np.frombuffer(a.tobytes(), dtype=np.float64)
    return flat if a.ndim == 1 else flat.reshape(a.shape)


class _Owned:
    """A float64 array OWNED by this module (sealed construction). NumPy: bytes-backed read-only storage (no view can
    become writable); CuPy: a private resident array with public access by detached copy only. `_internal()` is the
    zero-copy handle for the module's own kernels."""

    __slots__ = ("_array", "_xp")

    def __init__(self, seal: object, xp: ModuleType, array: Any) -> None:
        if seal is not _SEAL:
            raise PreferentialError("owned parameter arrays are created only by maple_syrup.preferential_infiltration")
        if type(array) is not xp.ndarray or array.dtype != np.float64:
            raise PreferentialError("internal: owned payloads are float64 arrays of the operator namespace")
        object.__setattr__(self, "_xp", xp)
        object.__setattr__(self, "_array", _sealed_host(array) if xp is np else xp.ascontiguousarray(array))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("owned parameter arrays are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("owned parameter arrays are immutable")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._array.shape)

    @property
    def nbytes(self) -> int:
        return int(self._array.nbytes)

    def _internal(self) -> Any:
        """Zero-copy handle for module-internal READ use only (kernels)."""
        return self._array

    def readonly(self) -> Any:
        """NumPy: zero-copy read-only view over read-only storage (cannot be made writable). CuPy: detached copy."""
        if self._xp is np:
            view = self._array.view()
            view.flags.writeable = False
            return view
        return self._array.copy()

    def detached(self) -> Any:
        """Always a fresh writable copy (allocates on the payload's side)."""
        return self._array.copy()

    def host_bytes(self) -> bytes:
        """Host bytes (ONE counted device->host read on CuPy). Cold boundary only."""
        if self._xp is np:
            return self._array.tobytes()
        from maple.core.backend import to_host

        return np.ascontiguousarray(to_host(self._array)).tobytes()


# --- receiver weights -------------------------------------------------------------------------------------------------
class ReceiverWeights:
    """One finite nonnegative receiving-weight vector over the L layers, minted ONLY by `receiver_weights`. `raw` and
    `normalized` are read-only views over sealed storage; `enabled` is False for an all-zero vector (that pathway
    contributes nothing) and `normalized` is raw / sum for an enabled vector, zeros otherwise. Direct construction,
    attribute assignment and inconsistent records are refused."""

    __slots__ = ("_normalized", "_raw", "enabled", "n_layers", "total")

    def __init__(self, seal: object = None, raw: Any = None, normalized: Any = None, enabled: Any = None,
                 total: Any = None) -> None:
        if seal is not _SEAL or not isinstance(raw, _Owned) or not isinstance(normalized, _Owned):
            raise PreferentialError("ReceiverWeights is constructed only by receiver_weights(); direct construction "
                                    "(any argument pattern) is refused")
        for name, value in (("_raw", raw), ("_normalized", normalized), ("enabled", bool(enabled)),
                            ("total", float(total)), ("n_layers", raw.shape[0])):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ReceiverWeights is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ReceiverWeights is immutable")

    def __repr__(self) -> str:
        return f"ReceiverWeights(raw={self.raw.tolist()}, enabled={self.enabled})"

    @property
    def raw(self) -> np.ndarray:
        return self._raw.readonly()

    @property
    def normalized(self) -> np.ndarray:
        return self._normalized.readonly()

    def summary(self) -> dict[str, Any]:
        return {"raw": self.raw.tolist(), "normalized": self.normalized.tolist(), "enabled": self.enabled,
                "total": self.total}


def _derive_weights(raw: np.ndarray, n: int, name: str) -> tuple[np.ndarray, np.ndarray, bool, float]:
    if raw.dtype.kind not in "iuf":
        raise PreferentialError(f"{name} must be real numbers, got dtype {raw.dtype}")
    if raw.shape != (n,):
        raise PreferentialError(f"{name} must have shape ({n},), got {raw.shape}")
    raw = np.array(raw, dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise PreferentialError(f"{name} must be finite everywhere")
    if np.any(raw < 0.0):
        raise PreferentialError(f"{name} must be >= 0 everywhere")
    total = float(np.sum(raw))
    if not math.isfinite(total):
        raise PreferentialError(f"{name} sum is not representable in FP64")
    enabled = total > 0.0
    normalized = raw / total if enabled else np.zeros(n, dtype=np.float64)
    if enabled and not np.all(np.isfinite(normalized)):
        raise PreferentialError(f"{name} normalisation is not representable in FP64")
    return raw, normalized, enabled, total


def receiver_weights(weights: Any, n_layers: int, name: str = "receiver_weights") -> ReceiverWeights:
    """Mint a length-L receiving-weight record: real, finite, >= 0; positive sum -> normalised; all zero -> disabled.
    Booleans, scalars (no broadcast: a receiver vector must be explicit per layer), wrong shapes, NaN, infinities and
    negative weights are refused. An existing `ReceiverWeights` is re-derived from its `raw` vector and must agree
    with its declared `normalized`, `enabled` and `total` bit for bit (a forged or stale record is refused)."""
    n = ss._checked_int(n_layers, "n_layers")
    if isinstance(weights, ReceiverWeights):
        if type(weights) is not ReceiverWeights:
            raise PreferentialError(f"{name} must be a genuine ReceiverWeights, got a subclass")
        raw, normalized, enabled, total = _derive_weights(np.asarray(weights.raw), n, name)
        if (weights.normalized.tobytes() != normalized.tobytes() or weights.enabled is not enabled
                or weights.total != total or weights.n_layers != n):
            raise PreferentialError(f"{name} is an inconsistent ReceiverWeights record (normalised/enabled/total do "
                                    "not follow from its raw weights); refused")
        return ReceiverWeights(_SEAL, _Owned(_SEAL, np, raw), _Owned(_SEAL, np, normalized), enabled, total)
    if isinstance(weights, (bool, np.bool_, int, float, np.integer, np.floating, str, bytes)):
        raise PreferentialError(f"{name} must be an explicit length-{n} sequence, got {type(weights).__name__}")
    try:
        raw = np.asarray(weights)
    except Exception as exc:  # noqa: BLE001
        raise PreferentialError(f"{name} is not array-like ({type(exc).__name__}: {exc})") from None
    raw, normalized, enabled, total = _derive_weights(raw, n, name)
    return ReceiverWeights(_SEAL, _Owned(_SEAL, np, raw), _Owned(_SEAL, np, normalized), enabled, total)


# --- memory ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PreferentialMemory:
    """Backend-specific upper bound of the bytes held for one step (see MEMORY_SCOPE). `required_bytes` is the part
    compared with the column's `available_bytes_fn` (host for numpy/numba, device for cuda), reserves included;
    `build_bytes` is the extra transient scratch of `prepare_preferential` on the admitted side (admitted there on top
    of the step requirement); `build_host_bytes` is the cuda build's transient host scratch (download + sealing)."""

    backend: str
    n_layers: int
    n_cells: int
    n_species: int
    input_bytes: int  # caller's surface (N) and soil (L, N) arrays in the column namespace
    parameter_bytes: int  # owned maps on the compute side: beta N + rate N + max rate N + receiver L N + 2 L
    parameter_host_bytes: int  # sealed host mirrors of the same maps (cuda only; the compute maps on numpy/numba)
    weight_bytes: int  # owned receiver-weight vectors: raw + normalised for S species and the background (host)
    output_bytes: int  # surface N + soil L N + intake L N + total N + rows 8 N
    scratch_bytes: int  # transient temporaries of the backend's step plus the packet
    build_bytes: int  # transient build scratch on the admitted side (cover, denominator, layer sum, a3 map, masks, sealing)
    build_host_bytes: int  # cuda: transient host scratch of the build (download + sealing of the largest map)
    reserve_bytes: int  # column reserve
    extra_reserve_bytes: int  # caller's simultaneously live bytes
    host_bytes: int
    device_bytes: int
    required_bytes: int  # admitted side + reserves
    step_allocation_bytes: int  # output_bytes + scratch_bytes
    total_bytes: int  # host + device + reserves
    scope: str = MEMORY_SCOPE

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def estimate_preferential_memory(n_layers: int, n_cells: int, *, backend: str, n_species: int = 0,
                                 reserve_bytes: int = 0, extra_reserve_bytes: int = 0) -> PreferentialMemory:
    """Conservative, backend-specific working-array bytes of ONE step plus the owned weights and the build scratch.
    Pure arithmetic (no allocation, no query)."""
    L = ss._checked_int(n_layers, "n_layers")
    N = ss._checked_int(n_cells, "n_cells")
    S = ss._checked_int(n_species, "n_species")
    if L < 1 or N < 1:
        raise PreferentialError("n_layers and n_cells must be >= 1")
    if backend not in BACKENDS:
        raise PreferentialError(f"backend must be one of {BACKENDS}, got {backend!r}")
    reserve = ss._checked_int(reserve_bytes, "reserve_bytes")
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    inputs = (N + L * N) * _F8
    params = (3 * N + L * N + 2 * L) * _F8
    weights = 2 * (S + 1) * L * _F8
    outputs = (N + L * N + L * N + N + ROW_COUNT * N) * _F8
    scratch_build = BUILD_SCRATCH_DOUBLES_PER_CELL * N * _F8 + BUILD_SCRATCH_MASK_BYTES_PER_CELL * N
    sealing = L * N * _F8  # the largest map and its bytes copy are live at once while sealing
    if backend == "numpy":
        scratch = NUMPY_SCRATCH_DOUBLES_PER_CELL * N * _F8 + SCRATCH_MASK_BYTES_PER_CELL * N + _PACKET_SCRATCH_BYTES
        params_host, build, build_host = 0, scratch_build + sealing, 0
        host = inputs + params + weights + outputs + scratch
        device = 0
    elif backend == "numba":
        scratch = KERNEL_SCRATCH_DOUBLES_PER_CELL * N * _F8 + SCRATCH_MASK_BYTES_PER_CELL * N + _PACKET_SCRATCH_BYTES
        params_host, build, build_host = 0, scratch_build + sealing, 0
        host = inputs + params + weights + outputs + scratch
        device = 0
    else:
        scratch = KERNEL_SCRATCH_DOUBLES_PER_CELL * N * _F8 + SCRATCH_MASK_BYTES_PER_CELL * N
        params_host, build, build_host = params, scratch_build, sealing
        host = params_host + weights + _PACKET_SCRATCH_BYTES
        device = inputs + params + outputs + scratch
    required = (device if backend == "cuda" else host) + reserve + extra
    return PreferentialMemory(backend, L, N, S, inputs, params, params_host, weights, outputs, scratch, build,
                              build_host, reserve, extra, host, device, required, outputs + scratch,
                              host + device + reserve + extra)


def _admit(column: ss.SoilColumn, need: int, what: str) -> int:
    available = column.available_bytes_fn()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise PreferentialError(f"available_bytes_fn must return a non-Boolean integer >= 0, got {available!r}")
    available = int(available)
    if int(need) > available:
        raise PreferentialMemoryError(
            f"preferential {what} on {column.n_layers} layers x {column.n_cells} cells ({column.backend}) needs an "
            f"estimated {int(need)} B on its admitted side (reserves included) but only {available} B are reported "
            f"available; nothing was allocated. Scope: {MEMORY_SCOPE}")
    return available


# --- operator ------------------------------------------------------------------------------------------------------------
class StepArrays(NamedTuple):
    """Fresh per-step arrays of the column namespace, produced by a backend runner (no host read yet)."""

    surface_m: Any  # (N,)
    water_m: Any  # (L, N)
    intake_m: Any  # (L, N)
    intake_total_m: Any  # (N,)
    rows: Any  # (ROW_COUNT, N)


class KernelMaps(NamedTuple):
    """Parameter maps in kernel order (public accessors return read-only views / detached copies of the owned maps)."""

    rate_s: Any  # (N,) k per cell, +inf where beta == 1, 0 on inactive cells
    max_rate_m_s: Any  # (N,)
    receiver: Any  # (L, N) r_l, zero column where beta == 0
    saturation_m: Any  # (L,)
    capacity_tolerance_m: Any  # (L,)


_MAP_SHAPES: tuple[Callable[[int, int], tuple[int, ...]], ...] = (
    lambda L, N: (N,), lambda L, N: (N,), lambda L, N: (L, N), lambda L, N: (L,), lambda L, N: (L,))


def cell_closure_coefficient(n_layers: int) -> float:
    """MAPLE's dimensionless FP64 bound coefficient for the 4 L + 4 terms of one cell's step balance."""
    return volume_roundoff_bound_m3(CELL_OPERATION_TERMS_PER_LAYER * int(n_layers) + CELL_OPERATION_TERMS, 1.0)


def _digest(owned: Sequence[_Owned], names: Sequence[str]) -> str:
    """SHA-256 over the host bytes of the owned maps and beta (ONE download per array on CuPy). Cold only."""
    h = hashlib.sha256()
    for name, arr in zip(names, owned, strict=True):
        h.update(name.encode())
        h.update(str(arr.shape).encode())
        h.update(arr.host_bytes())
    return h.hexdigest()


_IDENTITY_NAMES = (*KernelMaps._fields, "beta")


class PreferentialOperator:
    """Prepared preferential pathway bound to one `SoilColumn`, minted ONLY by `prepare_preferential`. Owns its
    sealed parameter maps (`_Owned`), never aliases a caller array, and is immutable (attribute assignment and
    direct construction refuse). Public `maps`, `beta`, `host_maps`, `host_beta` return read-only views (NumPy) or
    detached copies / sealed host mirrors (CuPy); `detached_maps()` always copies. `identity()` and `summary()` return
    fresh dictionaries; `verify_identity()` is the explicit cold re-hash."""

    __slots__ = (
        "_backend_info",
        "_beta",
        "_host",
        "_host_beta",
        "_maps",
        "_preflight",
        "_runner",
        "_seal",
        "a3_summary",
        "backend",
        "background",
        "cell_closure_coefficient",
        "column",
        "device_id",
        "max_fraction_per_hour",
        "max_rate_summary",
        "memory",
        "n_cells",
        "n_layers",
        "n_species",
        "n_species_enabled",
        "sha256",
        "species",
        "xp",
    )

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise PreferentialError("PreferentialOperator is constructed only by prepare_preferential()")
        for name in self.__slots__:
            if name == "_seal":
                continue
            if name not in fields:
                raise PreferentialError(f"internal: operator field {name} missing")
            object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise PreferentialError(f"internal: unexpected operator fields {sorted(fields)}")
        object.__setattr__(self, "_seal", seal)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("PreferentialOperator is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("PreferentialOperator is immutable")

    def __repr__(self) -> str:
        return (f"PreferentialOperator(backend={self.backend!r}, n_layers={self.n_layers}, n_cells={self.n_cells}, "
                f"n_species={self.n_species}, sha256={self.sha256[:12]}...)")

    # public inspection (never per step; CuPy accessors allocate copies)
    @property
    def maps(self) -> KernelMaps:
        """Read-only views (NumPy) or detached copies (CuPy) of the compute-side maps."""
        return KernelMaps(*(o.readonly() for o in self._maps))

    @property
    def beta(self) -> Any:
        """Per-cell hourly fraction: read-only view (NumPy) or detached copy (CuPy)."""
        return self._beta.readonly()

    @property
    def host_maps(self) -> KernelMaps:
        """Read-only views of the sealed HOST arrays (the compute maps on numpy/numba, the mirrors on cuda)."""
        return KernelMaps(*(o.readonly() for o in self._host))

    @property
    def host_beta(self) -> np.ndarray:
        return self._host_beta.readonly()

    @property
    def backend_info(self) -> dict[str, Any]:
        """Backend provenance as a fresh DEEP copy (nested provenance mappings/lists never alias the record)."""
        return copy.deepcopy(self._backend_info)

    def detached_maps(self) -> KernelMaps:
        """Fresh writable copies of the compute-side maps (allocates), e.g. for a checkpoint writer."""
        return KernelMaps(*(o.detached() for o in self._maps))

    def _kernel_maps(self) -> KernelMaps:
        """Zero-copy private handles for the module's own kernels."""
        return KernelMaps(*(o._internal() for o in self._maps))

    def identity(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "n_layers": self.n_layers, "n_cells": self.n_cells, "n_species": self.n_species,
                "backend": self.backend, "fields": list(_IDENTITY_NAMES),
                "note": "digest of the host bytes of the sealed parameter maps and beta, computed once at build; "
                        "verify_identity() re-hashes explicitly (one download per array on cuda)"}

    def verify_identity(self) -> bool:
        """Re-hash the compute-side maps and beta (downloads on CuPy) AND the host mirrors and compare both with the
        digest recorded at build. Cold and explicit; never called per step."""
        compute = _digest((*self._maps, self._beta), _IDENTITY_NAMES)
        if compute != self.sha256:
            return False
        if self.backend == "cuda":
            return _digest((*self._host, self._host_beta), _IDENTITY_NAMES) == self.sha256
        return True

    def summary(self) -> dict[str, Any]:
        return summary(self)


def _check_operator(op: Any) -> None:
    """Boundary revalidation of a prepared operator (metadata only, O(L), no hashing, no copy): genuine sealed
    instance, record fields consistent with its column and its owned arrays, owned arrays of the exact namespace
    type, dtype, shape, contiguity, read-only state (NumPy) and device (CuPy)."""
    if type(op) is not PreferentialOperator:
        raise PreferentialError(f"operator must be a genuine PreferentialOperator, got {type(op).__name__}")
    if getattr(op, "_seal", None) is not _SEAL:
        raise PreferentialError("operator record is not sealed by prepare_preferential(); refused")
    column = op.column
    ss._check_column(column)
    xp = op.xp
    if op.backend != column.backend or xp is not column.xp or op.n_layers != column.n_layers or op.n_cells != column.n_cells:
        raise PreferentialError("operator record disagrees with its column (backend, namespace, layers or cells); refused")
    if op.device_id != column.device_id:
        raise PreferentialError("operator record disagrees with its column's device; refused")
    L, N = op.n_layers, op.n_cells
    maps, host = op._maps, op._host
    if (not isinstance(maps, tuple) or len(maps) != 5 or not isinstance(host, tuple) or len(host) != 5
            or not isinstance(op._beta, _Owned) or not isinstance(op._host_beta, _Owned)):
        raise PreferentialError("operator record carries a replaced parameter set; refused")
    for name, owned, shape_of in zip(KernelMaps._fields, maps, _MAP_SHAPES, strict=True):
        if not isinstance(owned, _Owned):
            raise PreferentialError(f"operator map {name} is not an owned array; refused")
        _check_owned(xp, owned, shape_of(L, N), name, column.device_id)
    _check_owned(xp, op._beta, (N,), "beta", column.device_id)
    if xp is not np:
        for name, owned, shape_of in zip(KernelMaps._fields, host, _MAP_SHAPES, strict=True):
            if not isinstance(owned, _Owned):
                raise PreferentialError(f"operator host mirror {name} is not an owned array; refused")
            _check_owned(np, owned, shape_of(L, N), f"host {name}", None)
        _check_owned(np, op._host_beta, (N,), "host beta", None)
    elif any(h is not m for h, m in zip(host, maps, strict=True)) or op._host_beta is not op._beta:
        raise PreferentialError("operator record carries replaced host mirrors; refused")
    if not callable(op._runner) or not isinstance(op.sha256, str) or len(op.sha256) != 64:
        raise PreferentialError("operator record carries a replaced runner or identity; refused")


def _check_owned(xp: ModuleType, owned: _Owned, shape: tuple[int, ...], name: str, device_id: int | None) -> None:
    array = owned._internal()
    if type(array) is not xp.ndarray or array.dtype != np.float64 or tuple(array.shape) != shape or not array.flags.c_contiguous:
        raise PreferentialError(f"operator map {name} lost its exact type/dtype/shape/contiguity; refused")
    if xp is np:
        if array.flags.writeable or not isinstance(array.base if array.base is not None else array, (np.ndarray, bytes)):
            raise PreferentialError(f"operator map {name} is not sealed read-only storage; refused")
    elif int(array.device.id) != device_id:
        raise PreferentialError(f"operator map {name} lives on device {int(array.device.id)}, the column on {device_id}")


def _scalar_or_cell_map(column: ss.SoilColumn, value: Any, name: str, *, low: float, high: float,
                        allow_inf: bool = False) -> tuple[Any, dict[str, Any]]:
    """A real host scalar or an exact (N,) array of the column namespace -> a fresh owned (N,) map plus a host
    summary. Bounds are checked everywhere (cold host read on cuda)."""
    xp = column.xp
    n = column.n_cells
    if isinstance(value, (bool, np.bool_)):
        raise PreferentialError(f"{name} must be numeric, got a Boolean")
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        if math.isnan(v) or (not allow_inf and not math.isfinite(v)):
            raise PreferentialError(f"{name} must be finite, got {value!r}")
        if v < low or v > high:
            raise PreferentialError(f"{name} must lie in [{low!r}, {high!r}], got {v!r}")
        return xp.full(n, v, dtype=np.float64), {"kind": "scalar", "value": v}
    ss._check_dynamic(column, name, value, (n,))
    from maple.core.backend import to_float

    arr = xp.ascontiguousarray(xp.array(value, dtype=np.float64, copy=True))
    if to_float(xp.sum(xp.isnan(arr))) != 0.0:
        raise PreferentialError(f"{name} must not contain NaN")
    if not allow_inf and to_float(xp.sum(~xp.isfinite(arr))) != 0.0:
        raise PreferentialError(f"{name} must be finite everywhere")
    lo, hi = to_float(xp.min(arr)), to_float(xp.max(arr))
    if lo < low or hi > high:
        raise PreferentialError(f"{name} must lie in [{low!r}, {high!r}] everywhere, got [{lo!r}, {hi!r}]")
    return arr, {"kind": "map", "min": lo, "max": hi}


def _validate_scalar_map_input(column: ss.SoilColumn, value: Any, name: str) -> None:
    """Allocation-free pre-check of a scalar/map input so admission can precede every allocation."""
    if isinstance(value, (bool, np.bool_)):
        raise PreferentialError(f"{name} must be numeric, got a Boolean")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return
    ss._check_dynamic(column, name, value, (column.n_cells,))


class _DeriveContext(NamedTuple):
    """PRIVATE honest context of the geometry-independent pathway derivation (Phase 2 S2-B2.b): the namespace, the
    dimensions, the active mask and the per-contribution array check. It is NOT a column, layers object or operator:
    the explicit spatial operator supplies the same fields from its own sealed owner."""

    xp: ModuleType
    n_layers: int
    n_cells: int
    active: Any  # (N,) bool in the namespace
    check_contribution: Callable[[int, Any], None]  # exact-array metadata check of one streamed contribution
    reduce: Callable[[str, Any], float] | None = None  # OPTIONAL cold-reduction hook (S2-B2.d); None = the namespace's own reductions, expression for expression (see `_reduce`)


REDUCE_KINDS = ("count_nonfinite", "min", "max", "count_nan")


def _reduce(ctx: _DeriveContext, kind: str, array: Any) -> float:
    """The cold scalar reductions of the derivation. With `ctx.reduce is None` (the common CPU/CUDA and the CPU spatial callers) they are exactly the previous expressions
    (`to_float(xp.sum(~xp.isfinite(a)))`, `to_float(xp.min(a))`, `to_float(xp.max(a))`, `to_float(xp.sum(xp.isnan(a)))`); an explicit spatial-CUDA caller supplies a bounded, counted
    reducer instead of the library reductions. An unknown kind is refused (never treated as another kind), by the default path and by the hook alike."""
    if kind not in REDUCE_KINDS:
        raise PreferentialError(f"unknown cold reduction kind {kind!r}; expected one of {REDUCE_KINDS}")
    if ctx.reduce is not None:
        return float(ctx.reduce(kind, array))
    from maple.core.backend import to_float

    xp = ctx.xp
    if kind == "count_nonfinite":
        return to_float(xp.sum(~xp.isfinite(array)))
    if kind == "min":
        return to_float(xp.min(array))
    if kind == "max":
        return to_float(xp.max(array))
    return to_float(xp.sum(xp.isnan(array)))


def _derive_pathway(ctx: _DeriveContext, a3_map: Any, max_fraction: float, background: Any, species: Sequence[Any],
                    species_contributions: Iterable[Any] | None) -> tuple[Any, Any, Any, int]:
    """The ONE derivation of the per-cell pathway maps from the explicit inputs: returns `(rate, receiver, beta,
    n_enabled)` with `rate` = k = -log1p(-beta)/3600 (+inf where beta == 1, 0 on inactive cells), `receiver` (L, N) =
    r_l (zero where beta == 0 or inactive) and `beta` (N,) the hourly fraction, the layer sum checked against the
    closed form. `a3_map` is read only: an `(N,)` array or a read-only broadcast view of a scalar. Species
    contributions are consumed lazily one `(N,)` array at a time and never retained. Shared by the common and the
    explicit spatial preparations (no copy of the law)."""
    xp, L, N, active = ctx.xp, ctx.n_layers, ctx.n_cells, ctx.active
    from maple.core.backend import errstate

    inactive = ~active
    background_on = background is not None and background.enabled
    # Build scratch plan (BUILD_SCRATCH_DOUBLES_PER_CELL doubles + BUILD_SCRATCH_MASK_BYTES_PER_CELL bytes per cell
    # beyond the persistent maps): cover (N), denominator (N), layer-sum (N) and one (N) temporary, the a3 map (N,
    # counted as scratch because it is not kept), plus transient boolean masks. `numerator` becomes the receiver map,
    # `scratch` becomes beta and `temp` becomes the rate map: all persistent, counted in parameter_bytes.
    cover = xp.zeros(N, dtype=np.float64)
    numerator = xp.zeros((L, N), dtype=np.float64)
    temp = xp.empty(N, dtype=np.float64)
    n_enabled = 0
    consumed = 0
    if species:
        iterator = iter(species_contributions)
        for s, sp in enumerate(species):
            try:
                c = next(iterator)
            except StopIteration:
                raise PreferentialError(f"species_contributions yielded {consumed} arrays for {len(species)} species") from None
            consumed += 1
            ctx.check_contribution(s, c)
            if _reduce(ctx, "count_nonfinite", c) != 0.0:  # transient (N,) bool (default path)
                raise PreferentialError(f"species_contributions[{s}] must be finite everywhere")
            if _reduce(ctx, "min", c) < 0.0:
                raise PreferentialError(f"species_contributions[{s}] must be >= 0 everywhere")
            if not sp.enabled:
                continue  # all-zero receiving weights: this species' pathway is disabled, its cover is ignored
            n_enabled += 1
            xp.add(cover, c, out=cover)
            normalized = sp.normalized
            for layer in range(L):
                u = float(normalized[layer])
                if u != 0.0:
                    xp.multiply(c, u, out=temp)
                    xp.add(numerator[layer], temp, out=numerator[layer])
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise PreferentialError(f"species_contributions yielded more than {len(species)} arrays")
    # numerator_l <= C term by term (u <= 1), so a finite C certifies every numerator entry
    if _reduce(ctx, "count_nonfinite", cover) != 0.0:
        raise PreferentialError("the summed species contributions overflowed FP64; refused, not clipped")
    with errstate(xp=xp, all="ignore"):
        denominator = cover + 1.0  # (N,)
        # g_l = a3 b_l + (1 - a3) / (1 + C) numerator_l  (in place in `numerator`)
        xp.subtract(1.0, a3_map, out=temp)
        xp.divide(temp, denominator, out=temp)  # (1 - a3) / (1 + C)
        scratch = xp.empty(N, dtype=np.float64)
        background_normalized = background.normalized if background_on else None
        for layer in range(L):
            xp.multiply(numerator[layer], temp, out=numerator[layer])
            if background_on:
                b = float(background_normalized[layer])
                if b != 0.0:
                    xp.multiply(a3_map, b, out=scratch)
                    xp.add(numerator[layer], scratch, out=numerator[layer])
        layer_sum = xp.zeros(N, dtype=np.float64)
        for layer in range(L):
            xp.add(layer_sum, numerator[layer], out=layer_sum)
        # closed-form sum_l g_l in `scratch`: <= 1 exactly by monotone rounding (see the module docstring)
        if background_on:
            xp.add(a3_map, cover, out=scratch)
            xp.divide(scratch, denominator, out=scratch)  # (a3 + C) / (1 + C)
        else:
            xp.divide(cover, denominator, out=scratch)
            xp.subtract(1.0, a3_map, out=temp)
            xp.multiply(temp, scratch, out=scratch)  # (1 - a3) C / (1 + C)
        # build sanity: the layer sum agrees with the closed form within the MAPLE bound for L (S + 2) + 3 terms
        coefficient = volume_roundoff_bound_m3(L * (n_enabled + 2) + 3, 1.0)
        xp.subtract(layer_sum, scratch, out=temp)
        xp.abs(temp, out=temp)
        xp.add(scratch, layer_sum, out=denominator)  # both are >= 0: the magnitude scale
        xp.multiply(denominator, coefficient, out=denominator)
        xp.subtract(temp, denominator, out=temp)
        worst = _reduce(ctx, "max", temp)
        if not (worst <= 0.0):
            raise PreferentialError("internal: the receiver layer sum disagrees with the closed-form response beyond "
                                    f"the FP64 bound (worst excess {worst!r}); refused")
        del denominator, a3_map
        beta = scratch
        xp.multiply(beta, max_fraction, out=beta)
        xp.copyto(beta, 0.0, where=inactive)
        lo, hi = _reduce(ctx, "min", beta), _reduce(ctx, "max", beta)
        if not (0.0 <= lo and hi <= 1.0):
            raise PreferentialError(f"beta must lie in [0, 1] everywhere, got [{lo!r}, {hi!r}]")
        rate = temp  # k = -log1p(-beta) / 3600, +inf where beta == 1, 0 on inactive cells (beta is 0 there)
        xp.negative(beta, out=rate)
        xp.log1p(rate, out=rate)
        xp.negative(rate, out=rate)
        xp.divide(rate, SECONDS_PER_HOUR, out=rate)
        # receiver split r_l = g_l / sum_l g_l (zero column where the sum vanishes), zero on inactive cells
        receiver = numerator
        positive = layer_sum > 0.0  # (N,) bool
        xp.copyto(layer_sum, 1.0, where=~positive)  # safe divisor
        for layer in range(L):
            xp.divide(receiver[layer], layer_sum, out=receiver[layer])
            xp.copyto(receiver[layer], 0.0, where=~positive)
            xp.copyto(receiver[layer], 0.0, where=inactive)
            if _reduce(ctx, "count_nonfinite", receiver[layer]) != 0.0:
                raise PreferentialError("internal: derived receiver map is not representable; refused")
        del positive, layer_sum, cover
    if _reduce(ctx, "count_nan", rate) != 0.0:
        raise PreferentialError("internal: derived rate map is not representable; refused")
    return rate, receiver, beta, n_enabled


def prepare_preferential(column: ss.SoilColumn, *, a3: Any, max_fraction_per_hour: Any, max_intake_rate_m_s: Any,
                         background_receiver_weights: Any, species_receiver_weights: Sequence[Any] = (),
                         species_contributions: Iterable[Any] | None = None,
                         extra_reserve_bytes: int = 0) -> PreferentialOperator:
    """Build the per-cell hourly fraction `beta`, the continuous rate map, the normalised receiver map and the rate
    cap map for `column` (O(L N) memory; species are streamed one (N,) contribution at a time).

    `a3` in [0, 1]: host scalar or exact (N,) array of the column namespace (use `reference_a3` for the reference
    selection). `max_fraction_per_hour` (maxFL2, dimensionless, [0, 1]) is a host scalar. `max_intake_rate_m_s`
    (maxAmountFL2 converted to m/s, >= 0; +inf means no rate cap) is a host scalar or (N,) array.
    `background_receiver_weights`: a length-L vector (all zero disables) or `None` (no background pathway declared).
    `species_receiver_weights`: a sequence with one length-L vector per LEAF species (all zero disables that species).
    `species_contributions`: an iterable yielding one exact (N,) column-namespace array c_s >= 0 per species in the
    same order (an (S, N) array iterates by rows); consumed lazily, never retained. Values on inactive cells are
    validated (finite, >= 0) but contribute nothing. Required when species are given; refused otherwise.

    Admission (step requirement + build scratch + owned weights + reserves) precedes EVERY allocation, including the
    small weight records; "numba" and "cuda" compile and warm up here (startup, outside steps). An unavailable
    backend raises `PreferentialBackendUnavailableError`."""
    ss._check_column(column)
    backend = column.backend
    xp = column.xp
    L, N = column.n_layers, column.n_cells
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    max_fraction = ss._real(max_fraction_per_hour, "max_fraction_per_hour")
    if not (0.0 <= max_fraction <= 1.0):
        raise PreferentialError(f"max_fraction_per_hour must lie in [0, 1], got {max_fraction!r}")
    if isinstance(species_receiver_weights, (str, bytes)) or not isinstance(species_receiver_weights, Sequence):
        raise PreferentialError("species_receiver_weights must be a sequence of per-species length-L vectors")
    S = len(species_receiver_weights)
    if S and species_contributions is None:
        raise PreferentialError(f"{S} species were given but species_contributions is None")
    if not S and species_contributions is not None:
        raise PreferentialError("species_contributions were given without species_receiver_weights")
    _validate_scalar_map_input(column, a3, "a3")
    _validate_scalar_map_input(column, max_intake_rate_m_s, "max_intake_rate_m_s")
    # backend probes (no allocation)
    preflight = None
    if backend == "cuda":
        from maple_syrup import preferential_infiltration_cuda as pc

        cp = pc.require_cuda()
        if pc.current_device_id(cp) != column.device_id:
            raise PreferentialBackendUnavailableError("prepare_preferential must run on the column's device")
    elif backend == "numba":
        from maple_syrup import preferential_infiltration_numba as pn

        pn.require_numba()
    memory = estimate_preferential_memory(L, N, backend=backend, n_species=S, reserve_bytes=column.reserve_bytes,
                                          extra_reserve_bytes=extra)
    _admit(column, memory.required_bytes + memory.build_bytes, "build")  # BEFORE any allocation, weights included
    # owned weight records (2 (S + 1) L host doubles, counted in weight_bytes)
    background = None if background_receiver_weights is None else receiver_weights(
        background_receiver_weights, L, "background_receiver_weights")
    species = tuple(receiver_weights(w, L, f"species_receiver_weights[{s}]")
                    for s, w in enumerate(species_receiver_weights))
    from maple.core.backend import to_device, to_host

    a3_map, a3_summary = _scalar_or_cell_map(column, a3, "a3", low=0.0, high=1.0)
    max_rate, rate_summary = _scalar_or_cell_map(column, max_intake_rate_m_s, "max_intake_rate_m_s", low=0.0,
                                                 high=math.inf, allow_inf=True)
    active = column.active
    rate, receiver, beta, n_enabled = _derive_pathway(
        _DeriveContext(xp, L, N, active, lambda s_i, c: ss._check_dynamic(column, f"species_contributions[{s_i}]", c, (N,))),
        a3_map, max_fraction, background, species, species_contributions)
    del a3_map
    xp.copyto(max_rate, 0.0, where=~active)
    sat_xp = to_device(np.array(column.layers.saturation_m), xp) if xp is not np else np.array(column.layers.saturation_m)
    tol_xp = to_device(np.array(column.layers.capacity_tolerance_m), xp) if xp is not np else np.array(
        column.layers.capacity_tolerance_m)
    # sealing: largest map first so the sealing copy and the freed scratch do not overlap more than declared
    owned_receiver = _Owned(_SEAL, xp, xp.ascontiguousarray(receiver))
    del receiver
    owned = KernelMaps(_Owned(_SEAL, xp, xp.ascontiguousarray(rate)), _Owned(_SEAL, xp, xp.ascontiguousarray(max_rate)),
                       owned_receiver, _Owned(_SEAL, xp, xp.ascontiguousarray(sat_xp)),
                       _Owned(_SEAL, xp, xp.ascontiguousarray(tol_xp)))
    owned_beta = _Owned(_SEAL, xp, xp.ascontiguousarray(beta))
    del rate, max_rate, beta, sat_xp, tol_xp
    if xp is np:
        host_owned, host_beta = owned, owned_beta
    else:  # one-time cold downloads into sealed host mirrors (largest first)
        host_owned = KernelMaps(*(_Owned(_SEAL, np, np.ascontiguousarray(to_host(o._internal()))) for o in owned))
        host_beta = _Owned(_SEAL, np, np.ascontiguousarray(to_host(owned_beta._internal())))
    sha256 = _digest((*host_owned, host_beta), _IDENTITY_NAMES)
    coefficient_cell = cell_closure_coefficient(L)
    kernel_maps = KernelMaps(*(o._internal() for o in owned))
    if backend == "numpy":
        runner, info = reference_runner, {"implementation": "numpy reference (layer loop, vectorised cells, in place)"}
    elif backend == "numba":
        runner, info = pn.prepare_runner(kernel_maps, coefficient_cell)
    else:
        runner, info, preflight = pc.prepare_runner(kernel_maps, coefficient_cell, cp, column.device_id)
    return PreferentialOperator(
        _SEAL, column=column, backend=backend, xp=xp, n_layers=L, n_cells=N, n_species=S, n_species_enabled=n_enabled,
        background=background, species=species, max_fraction_per_hour=max_fraction, _maps=tuple(owned),
        _beta=owned_beta, _host=tuple(host_owned), _host_beta=host_beta, cell_closure_coefficient=coefficient_cell,
        memory=memory, sha256=sha256, _runner=runner, _preflight=preflight, device_id=column.device_id,
        _backend_info=copy.deepcopy(dict(info)), a3_summary=MappingProxyType(dict(a3_summary)),
        max_rate_summary=MappingProxyType(dict(rate_summary)))


def admit_step(op: PreferentialOperator) -> int:
    """Compare the operator's per-step requirement (reserves included) with the column's reported available bytes
    and raise `PreferentialMemoryError` BEFORE anything is allocated. Returns the reading."""
    _check_operator(op)
    return _admit(op.column, op.memory.required_bytes, "step")


# --- the reference step -------------------------------------------------------------------------------------------------
class _StepContext(NamedTuple):
    """PRIVATE honest context of the shared NumPy body: exactly what the body needs besides the arrays (Phase 2
    S2-B2.b). It is NOT a column, layers object or operator."""

    n_layers: int
    n_cells: int
    active: Any  # (N,) bool
    cell_tol: float  # cell_closure_coefficient(L)


def reference_step(op: PreferentialOperator, maps: KernelMaps, surface_in: np.ndarray, water_in: np.ndarray,
                   dt: float) -> StepArrays:
    """NumPy reference of one step: loops over layers, vectorised over cells, the SAME IEEE operations in the SAME
    order as the compiled kernels (only `expm1` may differ by an ulp between libraries), in place on the fresh
    outputs with a counted transient peak. `maps` are the operator's private zero-copy handles. Validation is
    recorded in flag rows, not raised here. The body is `_reference_step_body`, shared with the explicit spatial
    operator; this wrapper passes the cached `(L,)` vectors exactly as before."""
    rate, max_rate, recv, sat_m, cap_tol = maps
    ctx = _StepContext(op.n_layers, op.n_cells, op.column.active, op.cell_closure_coefficient)
    return _reference_step_body(ctx, rate, max_rate, recv, sat_m, cap_tol, surface_in, water_in, dt)


def _reference_step_body(ctx: _StepContext, rate: Any, max_rate: Any, recv: Any, sat_m: Any, cap_tol: Any,
                         surface_in: np.ndarray, water_in: np.ndarray, dt: float) -> StepArrays:
    """The ONE NumPy physical body. `rate`/`max_rate` are `(N,)` arrays (a read-only broadcast view allowed),
    `recv` the `(L, N)` receiver map, `sat_m` and `cap_tol` INDEXABLE per-layer readers (`x[layer]` -> a scalar or
    an `(N,)` row): the cached `(L,)` arrays of the common operator or lazy readers of an explicit spatial owner."""
    from maple.core.backend import errstate

    xp = np
    L, N = ctx.n_layers, ctx.n_cells
    active = ctx.active
    inactive = ~active
    cell_tol = ctx.cell_tol

    surface_out = xp.array(surface_in, dtype=np.float64, copy=True)
    water_out = xp.array(water_in, dtype=np.float64, copy=True)
    intake = xp.zeros((L, N), dtype=np.float64)
    total = xp.zeros(N, dtype=np.float64)
    rows = xp.zeros((ROW_COUNT, N), dtype=np.float64)
    before, after, h_in, h_out, isolated, attempted_row = rows[0], rows[1], rows[2], rows[3], rows[4], rows[5]
    flags = xp.zeros(N, dtype=np.int64)

    def flag(mask: Any, bit: int) -> None:
        xp.bitwise_or(flags, np.int64(1 << bit), out=flags, where=mask)

    with errstate(xp=xp, all="ignore"):
        for layer in range(L):
            wl = water_in[layer]
            flag(~xp.isfinite(wl), 0)
            flag(wl < 0.0, 1)
            flag(wl > sat_m[layer] + cap_tol[layer], 2)
            xp.add(before, wl, out=before)
        flag(~xp.isfinite(surface_in), 3)
        flag(surface_in < 0.0, 4)
        xp.copyto(h_in, surface_in)
        if dt > 0.0:
            # 2. isolated attempt: H x (-expm1(-(k dt)))
            x = rate * dt
            xp.negative(x, out=x)
            xp.expm1(x, out=x)
            xp.negative(x, out=x)  # f = -expm1(-(k dt))
            attempted = surface_in * x
            del x
            xp.copyto(isolated, attempted)
            # 3. ONE aggregate rate cap
            cap = max_rate * dt
            xp.copyto(attempted, cap, where=cap < attempted)
            del cap
            xp.copyto(attempted_row, attempted)
            # 4. receiver allocation, top layer first; each receiver bounded by its room and the remaining surface
            left = surface_out
            for layer in range(L):
                alloc = attempted * recv[layer]
                room = sat_m[layer] - water_out[layer]
                xp.copyto(room, 0.0, where=room < 0.0)
                xp.copyto(alloc, room, where=room < alloc)
                xp.copyto(alloc, left, where=left < alloc)
                del room
                xp.add(water_out[layer], alloc, out=water_out[layer])
                xp.subtract(left, alloc, out=left)
                intake[layer] = alloc
                xp.add(total, alloc, out=total)
                del alloc
            del attempted
            # 5. output checks (active cells only)
            for layer in range(L):
                wl = water_out[layer]
                flag(active & ~xp.isfinite(wl), 5)
                flag(active & (wl < 0.0), 6)
                flag(active & (wl > sat_m[layer] + cap_tol[layer]), 7)
                xp.add(after, wl, out=after)
            flag(active & ~xp.isfinite(surface_out), 5)
            flag(active & (surface_out < 0.0), 6)
            closure = after - before
            xp.subtract(closure, total, out=closure)
            scale = before + after
            xp.add(scale, h_in, out=scale)
            xp.add(scale, surface_out, out=scale)
            xp.add(scale, total, out=scale)
            xp.multiply(scale, cell_tol, out=scale)
            xp.abs(closure, out=closure)
            flag(active & (closure > scale), 8)
            xp.subtract(h_in, surface_out, out=closure)
            xp.subtract(closure, total, out=closure)
            xp.abs(closure, out=closure)
            flag(active & (closure > scale), 9)
            del closure, scale
            nonfinite = ~xp.isfinite(total)
            for layer in range(L):
                nonfinite |= ~xp.isfinite(intake[layer])
            flag(active & nonfinite, 10)
            del nonfinite
            # inactive cells: identity state, zero fluxes
            xp.copyto(surface_out, surface_in, where=inactive)
            for layer in range(L):
                xp.copyto(water_out[layer], water_in[layer], where=inactive)
                xp.copyto(intake[layer], 0.0, where=inactive)
            xp.copyto(total, 0.0, where=inactive)
            xp.copyto(isolated, 0.0, where=inactive)
            xp.copyto(attempted_row, 0.0, where=inactive)
        else:
            after[...] = before
        xp.copyto(h_out, surface_out)
        rows[6][...] = total
        for row in (before, after, h_in, h_out):
            xp.copyto(row, 0.0, where=inactive)
        low = xp.negative(flags)
        xp.bitwise_and(flags, low, out=low)
        rows[7][...] = low
        xp.copyto(rows[7], NO_FLAG, where=flags == 0)
        del low
    return StepArrays(surface_out, water_out, intake, total, rows)


def reference_runner(op: PreferentialOperator, maps: KernelMaps, surface_in: Any, water_in: Any, dt: float) -> StepArrays:
    return reference_step(op, maps, surface_in, water_in, dt)


# --- step ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PreferentialBudget:
    """Host volumes (m3) over active cells from ONE packet read, and the MAPLE-derived closure bound."""

    soil_before_m3: float
    soil_after_m3: float
    surface_before_m3: float
    surface_after_m3: float
    isolated_attempt_m3: float  # H x (-expm1(-k dt)) before the rate cap
    attempted_m3: float  # after the ONE aggregate rate cap
    intake_m3: float  # actual, added to the soil once and removed from the surface once
    rate_capped_m3: float  # isolated - attempted
    capacity_rejected_m3: float  # attempted - intake: stayed on the surface
    soil_residual_m3: float  # (after - before) - intake
    surface_residual_m3: float  # (H_in - H_out) - intake
    bound_m3: float
    n_terms: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True, eq=False)
class PreferentialStep:
    """Fresh arrays of the column namespace plus the batched scalar budget. `state.t_s` is PRESERVED: this is one
    stage of the caller's hydraulic step, which advances time exactly once."""

    surface_water_m: Any  # (N,)
    state: ss.SoilColumnState
    intake_m: Any  # (L, N) actual intake per receiver
    intake_total_m: Any  # (N,)
    dt_s: float
    budget: PreferentialBudget


def _read_packet(xp: ModuleType, arrays: StepArrays) -> np.ndarray:
    from maple.core.backend import to_host

    rows = arrays.rows
    values = [xp.sum(rows[r]) for r in range(7)] + [xp.min(rows[7]), xp.count_nonzero(rows[7] < NO_FLAG)]
    stacked = xp.stack([xp.asarray(v, dtype=np.float64) for v in values])
    return np.asarray(to_host(stacked), dtype=np.float64)


def _resolve(op: PreferentialOperator, packet: np.ndarray) -> PreferentialBudget:
    return _resolve_core(op.column.cell_area_m2, op.n_layers, op.n_cells, packet)


def _resolve_core(area: float, n_layers: int, n_cells: int, packet: np.ndarray) -> PreferentialBudget:
    """Backend- and owner-independent budget of ONE packet (private; shared with the explicit spatial operator):
    `area` is applied ONCE, finiteness is checked BEFORE the strict MAPLE bound helper."""
    if packet.shape != (PACKET_LENGTH,):
        raise PreferentialError(f"internal: packet shape {packet.shape}")
    low, n_flagged = float(packet[7]), int(packet[8])
    if not math.isfinite(low) or low < NO_FLAG:
        bit = int(low).bit_length() - 1 if math.isfinite(low) and low >= 1.0 else None
        message = FLAG_MESSAGES[bit] if bit is not None and 0 <= bit < N_FLAGS else "validation packet is corrupt"
        raise PreferentialStepError(f"{message} ({n_flagged} cell(s) flagged)", bit=bit, n_flagged=n_flagged)
    before, after, h_in, h_out, isolated, attempted, intake = (float(v) * area for v in packet[:7])
    soil_residual = (after - before) - intake
    surface_residual = (h_in - h_out) - intake
    n_terms = (CELL_OPERATION_TERMS_PER_LAYER * n_layers + CELL_OPERATION_TERMS) * n_cells + ROW_COUNT
    operands = (abs(before), abs(after), abs(h_in), abs(h_out), abs(intake))
    if not all(math.isfinite(v) for v in (*operands, isolated, attempted, soil_residual, surface_residual)):
        raise PreferentialStepError("non-finite global water budget", bit=None, n_flagged=n_flagged)
    bound = volume_roundoff_bound_m3(n_terms, max(operands))
    if abs(soil_residual) > bound or abs(surface_residual) > bound:
        raise PreferentialStepError(
            f"global water closure violated: soil residual {soil_residual!r} m3, surface residual "
            f"{surface_residual!r} m3 against the FP64 bound {bound!r} m3 ({n_terms} terms)", bit=None,
            n_flagged=n_flagged)
    return PreferentialBudget(before, after, h_in, h_out, isolated, attempted, intake, isolated - attempted,
                              attempted - intake, soil_residual, surface_residual, bound, n_terms)


def preferential_step(op: PreferentialOperator, surface_water_m: Any, state: ss.SoilColumnState, dt_s: Any) -> PreferentialStep:
    """One pure preferential stage over `dt_s` (see the module docstring for the stage order). `surface_water_m`
    (N,) is the standing surface water of every cell; `state` is the authoritative soil column state. Both are read
    only. dt = 0 is the identity. A refused step allocates no result and publishes nothing. The operator record is
    re-validated against its sealed arrays first (metadata only)."""
    _check_operator(op)
    column = op.column
    t0 = ss._check_state(column, state)
    dt = ss._check_dt(dt_s)
    ss._check_dynamic(column, "surface_water_m", surface_water_m, (column.n_cells,))
    if op._preflight is not None:
        op._preflight()
    _admit(column, op.memory.required_bytes, "step")
    arrays = op._runner(op, op._kernel_maps(), surface_water_m, state.water_m, dt)
    packet = _read_packet(op.xp, arrays)  # the ONE counted host read; synchronising on cuda
    budget = _resolve(op, packet)
    return PreferentialStep(surface_water_m=arrays.surface_m, state=ss.SoilColumnState(arrays.water_m, t0),
                            intake_m=arrays.intake_m, intake_total_m=arrays.intake_total_m, dt_s=dt, budget=budget)


# --- provenance ------------------------------------------------------------------------------------------------------
def summary(op: PreferentialOperator) -> dict[str, Any]:
    _check_operator(op)
    return {
        "backend": op.backend,
        "namespace": op.xp.__name__,
        "device_id": op.device_id,
        "n_layers": op.n_layers,
        "n_cells": op.n_cells,
        "n_active": op.column.n_active,
        "n_species": op.n_species,
        "n_species_enabled": op.n_species_enabled,
        "background": None if op.background is None else op.background.summary(),
        "background_enabled": op.background is not None and op.background.enabled,
        "species": [s.summary() for s in op.species],
        "max_fraction_per_hour": op.max_fraction_per_hour,
        "a3": dict(op.a3_summary),
        "max_intake_rate_m_s": dict(op.max_rate_summary),
        "time_basis": "beta is the reference HOURLY fraction; k = -log1p(-beta)/3600 s^-1; isolated attempt "
                      "H (-expm1(-k dt)); never beta H per sub-step",
        "closed_form_response": "(a3 + C)/(1 + C) with the background enabled; (1 - a3) C/(1 + C) without",
        "stage_order": list(STAGE_ORDER),
        "time": "state.t_s preserved; the owning hydraulic step advances time once",
        "memory": op.memory.as_dict(),
        "admission": "required_bytes (+ build_bytes at build) compared with the column's available_bytes_fn before "
                     "prepare_preferential (before the weight records) and every step; a reading, not a reservation",
        "closure": f"conservation.volume_roundoff_bound_m3: per cell {CELL_OPERATION_TERMS_PER_LAYER} L + "
                   f"{CELL_OPERATION_TERMS} terms, global ({CELL_OPERATION_TERMS_PER_LAYER} L + "
                   f"{CELL_OPERATION_TERMS}) N + {ROW_COUNT} terms",
        "identity": op.identity(),
        "sealing": "factory-only records; numpy maps are bytes-backed read-only storage (no view or base can become "
                   "writable); cuda maps are private with detached public copies and sealed host mirrors; every step "
                   "re-validates the record by metadata only",
        "packet_length": PACKET_LENGTH,
        "host_reads_per_step": 1,
        "backend_info": op.backend_info,  # fresh deep copy
        "source": dict(SOURCE_IDENTITY),
        "scope": "isolated staged preferential intake: no matrix infiltration, redistribution, E/T/growth, routing, "
                 "checkpoint or LEAF plant objects; not an EcoHyD identity claim",
    }
