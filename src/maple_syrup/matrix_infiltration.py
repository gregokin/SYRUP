"""Opt-in persistent Tietjen-placement matrix infiltration stage for the SYRUP layered soil column (Phase 1 component).

SYRUP owns all physical water. This module is ONE staged operation of a future storm hydraulic step: it moves water
from the cell's offered surface pool into the TOP layer of `maple_syrup.subsurface` (the authoritative layered
inventory) through a persistent Green-Ampt wetting history that preserves Tietjen's SEPARATE vegetation placements
before and after ponding. It performs no rain addition, no preferential stage, no redistribution, drainage,
evaporation, transpiration, growth, routing, sediment or bed edit; `state.t_s` is preserved (the owning hydraulic
step advances time exactly once). There is no second water inventory: the pool array, the `SoilColumnState` and the
`WettingHistory` are the caller's; results are fresh arrays. A refused step publishes nothing.

    column = subsurface.prepare_column(layers, active_mask, cell_area_m2, backend=...)
    op = prepare_matrix(column, conductivity_m_s=K, suction_m=psi, vegetation_factor=g)   # scalars or (N,) maps
    history = wetting_history(op, wetting_m=0.0, cumulative_intake_m=0.0)               # explicit initial history
    result = matrix_step(op, surface_pool_m, standing_before_m, incoming_rate_m_s, state, history, dt_s)

Source (read-only; descendants are reference evidence, not verified original HydroVeg releases): EcoHyD
`WaterCell::twoLayerInfiltration`, GitHub `WaterCell.cpp` lines 378 (fillable porosity), 393-397 (low-rate branch
admits the surface amount), 404-408 (onset `ts`, prepond `Fs`), 422-431 (Peschke hour, wetting flag); Dryad copies
lines 332, 344-348, 355-357, 371-378. Paper: Tietjen, Zehe & Jeltsch (2009) WRR 45 W01418 eqs (4)-(8). Audits:
`agent_handoffs/tasks/shared_scheduler_phase0_20261109/phase1_original_ponding_audit.md`,
`agent_handoffs/tasks/phase1_matrix_infiltration_20261110/independent_piecewise_law_audit.md` and
`root_design_correction01.md`. Units here are SI: m, s, m/s (the source uses mm, h, mm/h).

Approved law (user decisions 2026-11-10; `docs/subsurface_hydrology/development_phases_20261109.md`):

  * Local current deficit of the TOP layer, paper-first: b = psi x max(theta_fc,0 - theta_0, 0) [m], re-evaluated
    from the accepted native column water at every attempt and held fixed over it (the descendants' saturation
    minus domain mean is NOT used).
  * Persistent hydraulic history J [m] ("wetting_m"), NEVER a water store. Unponded intake (the low-rate branch
    r <= K and the prepond phase) advances J by the admitted amount itself (unscaled). The onset threshold for an
    incoming rate r > K is F_s = g b K / (r - K) (from a fresh history the source's t_s = g b / [(r/K - 1) r]);
    with an existing unponded history the remaining time is max(F_s - J, 0) / r (never restarted per call).
  * Ponded intake: dJ/dt = K (1 + b/J), physical dM = g dJ. Over a ponded duration tau with fixed b the hydraulic
    increment x solves x - b log1p(x / (J_old + b)) = K tau (exact implicit integration, replacing the hourly
    Peschke approximation: a declared numerical extension, not an EcoHyD identity claim). The physical candidate is
    g x; the ADMITTED amount min(g x, pool, room) advances J by admitted / g (g > 0). For g = 0 the postpond intake
    is 0 and J is unchanged (no division). Rejected potential never advances J.
  * Onset discontinuity (labelled, not smoothed): at onset the physical capacity drops from r to
    g K (1 + b/F_s) = r - (1 - g) K. The step counts in-interval onsets (`budget.n_onsets`).
  * Lifetime actual intake ("cumulative_intake_m") is a separate ledger: a future history recovery policy may
    change J but can never remove physical water. NO recovery/reset policy is implemented here (not selected).

Combined actual intake (S2-B2.c review02, a numerical correction, not a new law): the two separately bounded parts `pre` (unponded) and `post` (ponded, bounded by `pool - pre` and `room - pre`) can recombine, in
FP64, one ulp ABOVE the pool or the room (`fl(pre + fl(pool - pre)) != pool`). The recombined `intake = pre + post` is therefore bounded, in this order, by the pool and by the top-layer room BEFORE anything is
published; `pre` is unchanged; ONLY when that bound is active the ponded actual is re-derived as `post = intake - pre` (>= 0 exactly) so the history advance `post / g`, the soil credit, the pool debit, the ledger
and the diagnostic rows share the SAME actuals. When the bound is inactive every result is byte-identical to the previous arithmetic. No epsilon, no clipping of a negative result, no tolerance change; the returned
candidate is unchanged and rejected potential never enters the soil. The Numba and CUDA bodies use the same ordering.

Supply convention B (explicit finite-pool split; see the task design): the stage receives the ACTUAL remaining
surface pool of the interval, already supplied ONCE by the owning timestep (rain/run-on added once, preferential
stage applied), and never adds r dt again. `standing_before_m` is the genuine preexisting standing depth at the
START of the interval (before the interval's supply was added); only its positivity is used, as the contact
descriptor. `incoming_rate_m_s` is the onset descriptor only: never a water source, never a pond depth divided by
dt. Regimes per cell: contact -> ponded for the whole interval (recession with r = 0 included); no contact and
r <= K -> unponded (admits the pool up to the top room); no contact and r > K -> ponded from the start if
J >= F_s, else onset after t_p = (F_s - J)/r inside the interval (prepond r t_p first, bounded; if the pool or room
is exhausted before the threshold the onset is not reached), else unponded. Declared approximation of B: a pond
emptying inside the interval under continued low rain is not switched to the unponded branch within that interval;
refinement of the owner's interval converges to the continuous-input trajectory (tested as convergence).
Consistency flag: a cell without contact must satisfy pool <= r dt within the MAPLE FP64 bound.

Closure (active cells): |(soil_after - soil_before) - intake|, |(pool_before - pool_after) - intake| and the
history/ledger advances are bounded by MAPLE's `conservation.volume_roundoff_bound_m3` coefficient for
2 L + CELL_OPERATION_TERMS terms times the magnitude scale; globally (2 L + CELL_OPERATION_TERMS) N + ROW_COUNT terms
on the largest operand. No new empirical epsilon. The Newton stop uses the same coefficient for
NEWTON_TOLERANCE_TERMS terms as a relative step tolerance; a cell that does not converge within
NEWTON_MAX_ITERATIONS is flagged and the step refused (never clipped).

Backends: "numpy" (reference, vectorised cells, masked Newton iteration), "numba" (compiled serial cell loop,
`matrix_infiltration_numba`), "cuda" (one CuPy RawKernel thread per cell, `matrix_infiltration_cuda`); no fastmath,
no fallback. Dynamic arrays must be exactly the column namespace's `ndarray`, float64, C-contiguous, exactly shaped;
inputs are never written. Sealed factory-only operator with owned parameter maps (bytes-backed read-only storage on
NumPy; private device arrays with detached public copies and sealed host mirrors on CuPy), SHA-256 identity recorded
once, metadata-only revalidation per step; `WettingHistory` records are bound to the operator identity (a history
made for other K/psi/g or another column is refused, never reinterpreted). Memory admission (`estimate_matrix_memory`)
precedes EVERY allocation (build, history creation, step), one counted host read of the batched packet per step.

Nothing here was run against a GPU by its author; Codex records executions.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Any, NamedTuple

import numpy as np

from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.preferential_infiltration import geometric_conductivity_m_s, mm_per_hour_to_m_s

__all__ = [
    "BACKENDS",
    "BUILD_SCRATCH_DOUBLES_PER_CELL",
    "BUILD_SCRATCH_MASK_BYTES_PER_CELL",
    "CELL_OPERATION_TERMS",
    "CELL_OPERATION_TERMS_PER_LAYER",
    "FLAG_MESSAGES",
    "KERNEL_SCRATCH_DOUBLES_PER_CELL",
    "LOG1PMX_TERMS",
    "MEMORY_SCOPE",
    "NEWTON_MAX_ITERATIONS",
    "NEWTON_TOLERANCE_TERMS",
    "NO_FLAG",
    "NUMPY_SCRATCH_DOUBLES_PER_CELL",
    "PACKET_LENGTH",
    "REFERENCE_A1_A2",
    "ROW_COUNT",
    "ROW_NAMES",
    "SCRATCH_MASK_BYTES_PER_CELL",
    "SECONDS_PER_HOUR",
    "SOURCE_IDENTITY",
    "STAGE_ORDER",
    "SUPPLY_CONVENTION",
    "THREADS",
    "KernelMaps",
    "MatrixBackendUnavailableError",
    "MatrixBudget",
    "MatrixError",
    "MatrixMemory",
    "MatrixMemoryError",
    "MatrixOperator",
    "MatrixStep",
    "MatrixStepError",
    "StepArrays",
    "SubsurfaceBackendUnavailableError",
    "SubsurfaceError",
    "SubsurfaceMemoryError",
    "WettingHistory",
    "admit_step",
    "cell_closure_coefficient",
    "estimate_matrix_memory",
    "geometric_conductivity_m_s",
    "green_ampt_increment_m",
    "matrix_step",
    "mm_per_hour_to_m_s",
    "newton_tolerance",
    "prepare_matrix",
    "reference_hourly_intake_m",
    "reference_onset_capacities_m_s",
    "reference_onset_threshold_m",
    "reference_onset_time_s",
    "reference_step",
    "reference_vegetation_function",
    "summary",
    "wetting_history",
]

BACKENDS = ss.BACKENDS
SubsurfaceError = ss.SubsurfaceError
SubsurfaceMemoryError = ss.SubsurfaceMemoryError
SubsurfaceBackendUnavailableError = ss.SubsurfaceBackendUnavailableError
STAGE_ORDER = ("validation", "deficit_and_room", "regime_selection", "unponded_phase", "ponded_phase",
               "bounds_and_updates", "output_checks")
SUPPLY_CONVENTION = "B: explicit finite pool supplied once by the owner; incoming rate is the onset descriptor only"
SECONDS_PER_HOUR = 3600.0
THREADS = 128
#: Source Holling coefficients of vegFunction = (a1 + C)/(a2 + C) (GitHub WaterCell.cpp 384-386, Dryad 338-339).
#: Exposed for the reference helper; NEVER applied as a default (vegetation_factor is an explicit input).
REFERENCE_A1_A2 = (0.013, 0.067)
NEWTON_MAX_ITERATIONS = 64
NEWTON_TOLERANCE_TERMS = 8
#: Fixed term count of the atanh series of log1pmx for u <= 1 (z = u/(2+u) <= 1/3: z^41 / z^3 < 1e-18).
LOG1PMX_TERMS = 20
SOURCE_IDENTITY = {
    "github_watercell": "reference_models/ecohyd/github_baldauf_2023/source/model/src/WaterCell.cpp "
                        "(sha256 af476e1aa727ed515d75ed06a4a25b5285470caddebb141c35b8c0cf6201d9e9): Na 378, "
                        "vegFunction a1/a2 384-386, low-rate branch 393-397, onset ts 404 / Fs 408, Peschke A/B/FL1 "
                        "422-431, wetting flag 431; suction [mm] WaterCell.h:59, Ks [mm/h] WaterCell.h:60",
    "dryad_watercell": "reference_models/ecohyd/dryad_4j0zpc8gn/extracted/Irob_et_al_drought_resilience/Model/Source/"
                       "WaterCell.cpp (sha256 65c3978d9a474a582598c633bbcff243acd39d57c04220c2d6136f1f5bdc1389, both "
                       "Dryad copies byte-identical): Na 332, a1/a2 338-339, low-rate 344-348, ts/Fs 355-357, Peschke "
                       "371-378",
    "paper": "Tietjen, Zehe & Jeltsch (2009) WRR 45 W01418, https://doi.org/10.1029/2007WR006589, eqs (4)-(8)",
    "audits": ("agent_handoffs/tasks/shared_scheduler_phase0_20261109/phase1_original_ponding_audit.md",
               "agent_handoffs/tasks/phase1_matrix_infiltration_20261110/independent_piecewise_law_audit.md",
               "agent_handoffs/tasks/phase1_matrix_infiltration_20261110/root_design_correction01.md"),
    "decisions": "docs/subsurface_hydrology/development_phases_20261109.md (Phase 1, 2026-11-10 user decisions)",
    "status": "paper-first local deficit; persistent implicit Green-Ampt history is a declared numerical extension of "
              "the hourly source, not an EcoHyD result identity",
}
NO_FLAG = ss.NO_FLAG
ROW_NAMES = ("soil_before", "soil_after", "pool_before", "pool_after", "candidate", "intake", "prepond_intake",
             "postpond_intake", "room_rejected", "unused_capacity", "wetting_before", "wetting_after", "onset", "ponded",
             "iterations", "flag")
ROW_COUNT = len(ROW_NAMES)  # 16
PACKET_LENGTH = 17  # 14 row sums, max iterations, lowest flag, flagged count
# FP accumulations one cell undergoes: per layer the before/after sums (2 L) plus the fixed-count scalar chain
# (deficit, b, room, threshold, phase bounds, intake sum, pool debit, soil credit, history/ledger advances, closures).
CELL_OPERATION_TERMS_PER_LAYER = 2
CELL_OPERATION_TERMS = 16
NUMPY_SCRATCH_DOUBLES_PER_CELL = 48
KERNEL_SCRATCH_DOUBLES_PER_CELL = 2
SCRATCH_MASK_BYTES_PER_CELL = 16
BUILD_SCRATCH_DOUBLES_PER_CELL = 3
BUILD_SCRATCH_MASK_BYTES_PER_CELL = 8
_PACKET_SCRATCH_BYTES = 4 * PACKET_LENGTH * 8
_F8 = 8
_N_LAYER_MAPS = 4
MEMORY_SCOPE = (
    "arrays held for ONE matrix step: the caller's pool (N), standing-before (N), incoming-rate (N), soil (L, N) and "
    "history (2 N) arrays (known inputs), the operator's owned parameter maps (conductivity N, suction N, vegetation "
    "N, thickness L, field capacity L, saturation L, capacity tolerance L; sealed host arrays on numpy/numba, private "
    "device arrays plus sealed host mirrors on cuda), the fresh outputs (pool N, soil L N, history 2 N, intake N, "
    "rows ROW_COUNT N), the transient reference temporaries (masked Newton iteration) and the packet, plus the column "
    "reserve and the caller's extra reserve; the build additionally holds its transient map validation temporaries "
    "(3 N doubles), boolean masks and the sealing copy of the largest map (N doubles). NOT included: allocator pools, "
    "the CUDA context, Numba/NVRTC JIT workspaces, Python object headers, the surface routing state, LEAF or the "
    "remaining process memory. An upper bound on what this module allocates, not a reservation. On cuda the host side "
    "(mirrors, packet) is reported in host_bytes and is not compared with the device reading."
)

FLAG_MESSAGES = (
    "soil water state must be finite everywhere",  # 0
    "soil water state must be >= 0 everywhere",  # 1
    "soil water state exceeds the layer saturation depth",  # 2
    "surface_pool_m must be finite everywhere",  # 3
    "surface_pool_m must be >= 0 everywhere",  # 4
    "standing_before_m must be finite everywhere",  # 5
    "standing_before_m must be >= 0 everywhere",  # 6
    "incoming_rate_m_s must be finite everywhere",  # 7
    "incoming_rate_m_s must be >= 0 everywhere",  # 8
    "incoming_rate_m_s must be 0 on inactive cells (no forcing on inactive cells)",  # 9
    "history wetting_m must be finite everywhere",  # 10
    "history wetting_m must be >= 0 everywhere",  # 11
    "history cumulative_intake_m must be finite everywhere",  # 12
    "history cumulative_intake_m must be >= 0 everywhere",  # 13
    "a cell without preexisting contact offers a pool larger than its incoming supply r dt (inconsistent split)",  # 14
    "the implicit Green-Ampt solve did not converge within NEWTON_MAX_ITERATIONS",  # 15
    "step produced non-finite soil, pool or history",  # 16
    "step produced negative soil, pool or history",  # 17
    "step produced soil water above the layer saturation depth",  # 18
    "per-cell soil closure violated beyond the FP64 bound",  # 19
    "per-cell pool closure violated beyond the FP64 bound",  # 20
    "per-cell history/ledger advance violated beyond the FP64 bound",  # 21
    "step produced a non-finite candidate or intake",  # 22
)
N_FLAGS = len(FLAG_MESSAGES)

_SEAL = object()


# --- errors ----------------------------------------------------------------------------------------------------------
class MatrixError(ss.SubsurfaceError):
    """Invalid parameters, arrays, history, dt, option or record. Raised before any result exists."""


class MatrixStepError(MatrixError):
    """A step's batched validation recorded a failure (lowest bit first) or the global closure failed."""

    def __init__(self, message: str, *, bit: int | None = None, n_flagged: int = 0):
        super().__init__(message)
        self.bit = bit
        self.n_flagged = int(n_flagged)


class MatrixBackendUnavailableError(ss.SubsurfaceBackendUnavailableError):
    """The requested backend is unavailable. No fallback."""


class MatrixMemoryError(ss.SubsurfaceMemoryError):
    """The working-array estimate plus the reserves exceeds the reported available bytes (nothing allocated)."""


# --- host reference helpers --------------------------------------------------------------------------------------------
def _host_values(value: Any, name: str, *, allow_inf: bool = False) -> tuple[np.ndarray, bool]:
    if isinstance(value, (bool, np.bool_)):
        raise MatrixError(f"{name} must be numeric, got a Boolean")
    if isinstance(value, (int, float, np.integer, np.floating)):
        arr, scalar = np.array(float(value), dtype=np.float64), True
    else:
        if type(value) is not np.ndarray:
            raise MatrixError(f"{name} must be a real host number or a numpy.ndarray, got {type(value).__name__}")
        if value.dtype.kind not in "iuf":
            raise MatrixError(f"{name} must be real numbers, got dtype {value.dtype}")
        arr, scalar = np.array(value, dtype=np.float64), False
    if np.any(np.isnan(arr)) or (not allow_inf and not np.all(np.isfinite(arr))):
        raise MatrixError(f"{name} must be finite")
    return arr, scalar


def _emit(arr: np.ndarray, scalar: bool) -> Any:
    return float(arr) if scalar else arr


def _nonnegative(value: Any, name: str, *, allow_inf: bool = False) -> tuple[np.ndarray, bool]:
    arr, scalar = _host_values(value, name, allow_inf=allow_inf)
    if np.any(arr < 0.0):
        raise MatrixError(f"{name} must be >= 0")
    return arr, scalar


def reference_vegetation_function(cover_sum: Any, a1: Any, a2: Any) -> Any:
    """Source `vegFunction = (a1 + C) / (a2 + C)` (GitHub WaterCell.cpp 388-390) with EXPLICIT coefficients (the
    source's 0.013 / 0.067 are `REFERENCE_A1_A2`, never a default here). C >= 0 is the cover-weighted upper-layer
    root sum of the source; with a1 < a2 the result lies in [a1/a2, 1)."""
    c, scalar = _nonnegative(cover_sum, "cover_sum")
    a1v = ss._real(a1, "a1")
    a2v = ss._real(a2, "a2")
    if a1v < 0.0 or a2v <= 0.0:
        raise MatrixError("a1 must be >= 0 and a2 > 0")
    return _emit((a1v + c) / (a2v + c), scalar)


def reference_onset_threshold_m(incoming_rate_m_s: Any, conductivity_m_s: Any, suction_deficit_m: Any,
                                vegetation_factor: Any) -> Any:
    """Onset threshold F_s = g b K / (r - K) [m] for r > K (`+inf` for r <= K: the low-rate branch never ponds).
    From a fresh history this is the source prepond amount `Fs = ts x r` (WaterCell.cpp 404-408) in SI."""
    r, s1 = _nonnegative(incoming_rate_m_s, "incoming_rate_m_s")
    k, s2 = _nonnegative(conductivity_m_s, "conductivity_m_s")
    b, s3 = _nonnegative(suction_deficit_m, "suction_deficit_m")
    g, s4 = _nonnegative(vegetation_factor, "vegetation_factor")
    r, k, b, g = np.broadcast_arrays(r, k, b, g)
    with np.errstate(all="ignore"):
        fs = ((g * b) * k) / (r - k)
    out = np.where(r > k, fs, np.inf).astype(np.float64)
    return _emit(out, s1 and s2 and s3 and s4)


def reference_onset_time_s(incoming_rate_m_s: Any, conductivity_m_s: Any, suction_deficit_m: Any,
                           vegetation_factor: Any) -> Any:
    """Source onset time from a FRESH history, t_s = F_s / r = g b / [(r/K - 1) r] (eq. 4; WaterCell.cpp 404), in
    seconds; `+inf` for r <= K."""
    fs = reference_onset_threshold_m(incoming_rate_m_s, conductivity_m_s, suction_deficit_m, vegetation_factor)
    r, scalar = _nonnegative(incoming_rate_m_s, "incoming_rate_m_s")
    fs_arr = np.asarray(fs, dtype=np.float64)
    with np.errstate(all="ignore"):
        ts = np.where(np.isinf(fs_arr), np.inf, fs_arr / r)
    return _emit(np.asarray(ts, dtype=np.float64), scalar and np.ndim(fs) == 0)


def reference_onset_capacities_m_s(incoming_rate_m_s: Any, conductivity_m_s: Any, vegetation_factor: Any) -> tuple[Any, Any]:
    """Physical intake capacity just before and just after the onset from a fresh history: (r, r - (1 - g) K).
    The jump is the labelled consequence of the separate source placements (vanishes for g = 1)."""
    r, s1 = _nonnegative(incoming_rate_m_s, "incoming_rate_m_s")
    k, s2 = _nonnegative(conductivity_m_s, "conductivity_m_s")
    g, s3 = _nonnegative(vegetation_factor, "vegetation_factor")
    r, k, g = np.broadcast_arrays(r, k, g)
    after = r - (1.0 - g) * k
    scalar = s1 and s2 and s3
    return _emit(np.array(r, dtype=np.float64), scalar), _emit(np.array(after, dtype=np.float64), scalar)


def reference_hourly_intake_m(available_m: float, conductivity_m_s: float, suction_m: float, fillable_deficit: float,
                              vegetation_factor: float, previously_ponded: bool) -> tuple[float, float, bool]:
    """The SOURCE's hourly upper-layer intake (GitHub WaterCell.cpp 393-437) transcribed to SI for comparison tests:
    `available_m` is the surface amount offered in the hour (the source's `waterL0_abs`, mm, which the source also
    uses as the rate descriptor), K in m/s (x 3600 inside), suction in m, `fillable_deficit` dimensionless (the
    source's Na), `previously_ponded` the source wetting flag. Returns (intake_m, ts_fraction_of_hour, new_flag).
    Documentation/test helper; the persistent step does NOT use this expression.

    K = 0 with a positive offered amount (correction 02): the raw source expression divides by K; the helper returns
    the analytic K -> 0+ limit of the same expression, which is the same for either flag and any deficit:
    ts = g b / [(W/K_h - 1) W] -> 0+, Fs = ts W -> 0, A = K_h (1 - ts) -> 0, the Peschke rate term
    (A^2 + 2 A B)/(4 sqrt(h5)) ~ sqrt(b K_h / 2) -> 0 (b > 0) or = K_h/2 -> 0 (b = 0), so (0.0, 0.0, True).
    A zero offered amount with K = 0 stays in the low-rate branch (W <= K_h), as before. For representable tiny
    positive K whose Peschke radicand h5 underflows to 0, the same ratio is evaluated in an algebraically identical
    factored form (ordinary inputs use the source expression unchanged). This is a bounded mathematical extension of
    the comparison helper, not a claim that the raw source division is defined at zero."""
    w = ss._real(available_m, "available_m")
    kh = ss._real(conductivity_m_s, "conductivity_m_s") * SECONDS_PER_HOUR
    b = ss._real(suction_m, "suction_m") * ss._real(fillable_deficit, "fillable_deficit")
    g = ss._real(vegetation_factor, "vegetation_factor")
    if w < 0.0 or kh < 0.0 or b < 0.0 or g < 0.0:
        raise MatrixError("reference_hourly_intake_m inputs must be >= 0")
    if type(previously_ponded) is not bool:
        raise MatrixError("previously_ponded must be a bool")
    if w <= kh:  # 393-397
        return w, -1.0, False
    if kh == 0.0:  # analytic K -> 0+ limit of the ponding branches for W > 0 (see the docstring)
        return 0.0, 0.0, True
    if not previously_ponded:  # 403-408
        ts = (b / (w / kh - 1.0) / w) * g
        fs = min(1.0, ts) * w
    else:  # 410-413
        ts, fs = 0.0, 0.0
    if 0.0 <= ts < 1.0:  # 417-431 (Peschke after Dyck & Peschke 1989)
        a = kh * (1.0 - ts)
        bb = fs + 2.0 * b
        h5 = a * a / 4.0 + a * bb + fs * fs
        if h5 > 0.0:
            rate_term = (a * a + 2.0 * a * bb) / (4.0 * math.sqrt(h5))  # the source expression, unchanged
        else:
            # h5 underflowed (a > 0 here, so h5 > 0 in exact arithmetic): the SAME ratio factored by sqrt(a),
            # (a^2 + 2 a B)/(4 sqrt(h5)) = sqrt(a) (a + 2 B) / (4 sqrt(a/4 + B + (fs/a) fs)), stays representable.
            rate_term = math.sqrt(a) * (a + 2.0 * bb) / (4.0 * math.sqrt(a / 4.0 + bb + (fs / a) * fs))
        intake = min((a / 2.0 + rate_term) * g + fs, w)
        return intake, ts, True
    return fs, ts, False  # 434-437


def newton_tolerance() -> float:
    """Relative Newton step tolerance: MAPLE's dimensionless FP64 coefficient for NEWTON_TOLERANCE_TERMS terms."""
    return volume_roundoff_bound_m3(NEWTON_TOLERANCE_TERMS, 1.0)


def _log1pmx_vec(xp: ModuleType, u: Any) -> Any:
    """log1p(u) - u for u >= 0 without cancellation: for u <= 1 the atanh series in z = u/(2 + u)
    (log1p(u) = 2 atanh z, so log1pmx = -2 z^2/(1 - z) + 2 sum_{k>=1} z^(2k+1)/(2k+1), LOG1PMX_TERMS fixed terms,
    z <= 1/3), for u > 1 the direct difference (no catastrophic cancellation there). Same operation order as the
    kernels' scalar `log1pmx`."""
    z = u / (2.0 + u)
    z2 = z * z
    p = z * z2
    s = p / 3.0
    for k in range(2, LOG1PMX_TERMS + 1):
        p = p * z2
        s = s + p / float(2 * k + 1)
    series = (-2.0 * z2) / (1.0 - z) + 2.0 * s
    direct = xp.log1p(u) - u
    return xp.where(u > 1.0, direct, series)


def _ga_increment_vec(xp: ModuleType, j_old: Any, b: Any, r_dt: Any, tol: float, max_iter: int) -> tuple[Any, Any, Any]:
    """Vectorised implicit Green-Ampt increment x of  x - b log1p(x/(J_old + b)) = R  (R = K tau), evaluated with
    EXACTLY the scalar kernels' operation order on every cell (converged cells frozen by masks). The residual in
    y = x - R is written in the equivalent well-conditioned form psi(y) = (y J - b R)/D - b log1pmx((R + y)/D),
    D = J + b (the naive y - b log1p(u) loses up to ~sqrt(b/R) digits when J is small). Returns (x, iterations as
    float64, converged as bool). R == 0 or b == 0 give x = R directly."""
    from maple.core.backend import errstate

    n = j_old.shape[0]
    with errstate(xp=xp, all="ignore"):
        active = (r_dt != 0.0) & (b != 0.0)
        d = j_old + b
        y = xp.sqrt(r_dt * r_dt + (2.0 * r_dt) * b)
        y2 = (r_dt * b) / j_old
        xp.copyto(y, y2, where=(j_old > 0.0) & (y2 < y))
        del y2
        xp.copyto(y, 0.0, where=~active)
        iters = xp.zeros(n, dtype=np.float64)
        remaining = active.copy()
        dr = d + r_dt
        jr = j_old + r_dt
        br = b * r_dt
        for _ in range(max_iter):
            if not bool(xp.any(remaining)):
                break
            s = r_dt + y
            u = s / d
            psi = ((y * j_old) - br) / d - b * _log1pmx_vec(xp, u)
            alt = y - b * (xp.log(s) - xp.log(d))
            xp.copyto(psi, alt, where=xp.isinf(u))
            del alt, u
            stop_root = remaining & (psi <= 0.0)
            progress = remaining & ~stop_root
            dpsi = (jr + y) / (dr + y)
            step = psi / dpsi
            del psi, dpsi
            y_new = y - step
            xp.add(iters, progress, out=iters, casting="unsafe")
            stop_noprog = progress & (y_new >= y)
            move = progress & ~stop_noprog
            xp.copyto(y_new, 0.0, where=y_new < 0.0)
            xp.copyto(y, y_new, where=move)
            del y_new
            stop_small = move & (step <= tol * y)
            del step
            remaining &= ~(stop_root | stop_noprog | stop_small)
            del stop_root, stop_noprog, stop_small, progress, move, s
        converged = ~remaining
        x = r_dt + y
    return x, iters, converged


def green_ampt_increment_m(wetting_m: Any, suction_deficit_m: Any, conductivity_m_s: Any, duration_s: Any) -> tuple[Any, Any, Any]:
    """Host reference of the implicit increment x [m] of x - b log1p(x/(J_old + b)) = K tau (the HYDRAULIC
    increment; the physical candidate is g x). Scalars or broadcastable host arrays; returns (x, iterations,
    converged). Uses the module's own vectorised Newton (the NumPy reference path), so it is a convenience for tests
    and callers, not an independent control."""
    j, s1 = _nonnegative(wetting_m, "wetting_m")
    b, s2 = _nonnegative(suction_deficit_m, "suction_deficit_m")
    k, s3 = _nonnegative(conductivity_m_s, "conductivity_m_s")
    tau, s4 = _nonnegative(duration_s, "duration_s")
    j, b, k, tau = (np.ascontiguousarray(a, dtype=np.float64).reshape(-1) for a in np.broadcast_arrays(j, b, k, tau))
    x, iters, conv = _ga_increment_vec(np, j, b, k * tau, newton_tolerance(), NEWTON_MAX_ITERATIONS)
    scalar = s1 and s2 and s3 and s4
    if scalar:
        return float(x[0]), int(iters[0]), bool(conv[0])
    return x, iters.astype(np.int64), conv


# --- sealed owned arrays -------------------------------------------------------------------------------------------------
def _sealed_host(array: Any) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    flat = np.frombuffer(a.tobytes(), dtype=np.float64)
    return flat if a.ndim == 1 else flat.reshape(a.shape)


class _Owned:
    """A float64 array OWNED by this module. NumPy: bytes-backed read-only storage (no view or base can become
    writable); CuPy: private resident array, public access by detached copy only."""

    __slots__ = ("_array", "_xp")

    def __init__(self, seal: object, xp: ModuleType, array: Any) -> None:
        if seal is not _SEAL:
            raise MatrixError("owned parameter arrays are created only by maple_syrup.matrix_infiltration")
        if type(array) is not xp.ndarray or array.dtype != np.float64:
            raise MatrixError("internal: owned payloads are float64 arrays of the operator namespace")
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
        return self._array

    def readonly(self) -> Any:
        if self._xp is np:
            view = self._array.view()
            view.flags.writeable = False
            return view
        return self._array.copy()

    def detached(self) -> Any:
        return self._array.copy()

    def host_bytes(self) -> bytes:
        if self._xp is np:
            return self._array.tobytes()
        from maple.core.backend import to_host

        return np.ascontiguousarray(to_host(self._array)).tobytes()


# --- memory ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MatrixMemory:
    """Backend-specific upper bound of the bytes held for one step (see MEMORY_SCOPE). `required_bytes` is compared
    with the column's `available_bytes_fn` (host for numpy/numba, device for cuda), reserves included;
    `build_bytes` is the extra transient build scratch on the admitted side; `build_host_bytes` the cuda build's
    transient host scratch (download + sealing)."""

    backend: str
    n_layers: int
    n_cells: int
    input_bytes: int  # pool N + standing N + rate N + soil L N + history 2 N
    parameter_bytes: int  # compute-side maps: 3 N + 4 L
    parameter_host_bytes: int  # sealed host mirrors (cuda only)
    output_bytes: int  # pool N + soil L N + history 2 N + intake N + rows ROW_COUNT N
    scratch_bytes: int
    build_bytes: int
    build_host_bytes: int
    reserve_bytes: int
    extra_reserve_bytes: int
    host_bytes: int
    device_bytes: int
    required_bytes: int
    step_allocation_bytes: int
    total_bytes: int
    scope: str = MEMORY_SCOPE

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def estimate_matrix_memory(n_layers: int, n_cells: int, *, backend: str, reserve_bytes: int = 0,
                           extra_reserve_bytes: int = 0) -> MatrixMemory:
    """Conservative, backend-specific working-array bytes of ONE step plus the build scratch. Pure arithmetic."""
    L = ss._checked_int(n_layers, "n_layers")
    N = ss._checked_int(n_cells, "n_cells")
    if L < 1 or N < 1:
        raise MatrixError("n_layers and n_cells must be >= 1")
    if backend not in BACKENDS:
        raise MatrixError(f"backend must be one of {BACKENDS}, got {backend!r}")
    reserve = ss._checked_int(reserve_bytes, "reserve_bytes")
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    inputs = (3 * N + L * N + 2 * N) * _F8
    params = (3 * N + _N_LAYER_MAPS * L) * _F8
    outputs = (N + L * N + 2 * N + N + ROW_COUNT * N) * _F8
    scratch_build = BUILD_SCRATCH_DOUBLES_PER_CELL * N * _F8 + BUILD_SCRATCH_MASK_BYTES_PER_CELL * N
    sealing = N * _F8
    if backend == "numpy":
        scratch = NUMPY_SCRATCH_DOUBLES_PER_CELL * N * _F8 + SCRATCH_MASK_BYTES_PER_CELL * N + _PACKET_SCRATCH_BYTES
        params_host, build, build_host = 0, scratch_build + sealing, 0
        host, device = inputs + params + outputs + scratch, 0
    elif backend == "numba":
        scratch = KERNEL_SCRATCH_DOUBLES_PER_CELL * N * _F8 + SCRATCH_MASK_BYTES_PER_CELL * N + _PACKET_SCRATCH_BYTES
        params_host, build, build_host = 0, scratch_build + sealing, 0
        host, device = inputs + params + outputs + scratch, 0
    else:
        scratch = KERNEL_SCRATCH_DOUBLES_PER_CELL * N * _F8 + SCRATCH_MASK_BYTES_PER_CELL * N
        params_host, build, build_host = params, scratch_build, sealing
        host = params_host + _PACKET_SCRATCH_BYTES
        device = inputs + params + outputs + scratch
    required = (device if backend == "cuda" else host) + reserve + extra
    return MatrixMemory(backend, L, N, inputs, params, params_host, outputs, scratch, build, build_host, reserve, extra,
                        host, device, required, outputs + scratch, host + device + reserve + extra)


def _admit(column: ss.SoilColumn, need: int, what: str) -> int:
    available = column.available_bytes_fn()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise MatrixError(f"available_bytes_fn must return a non-Boolean integer >= 0, got {available!r}")
    available = int(available)
    if int(need) > available:
        raise MatrixMemoryError(
            f"matrix {what} on {column.n_layers} layers x {column.n_cells} cells ({column.backend}) needs an estimated "
            f"{int(need)} B on its admitted side (reserves included) but only {available} B are reported available; "
            f"nothing was allocated. Scope: {MEMORY_SCOPE}")
    return available


# --- operator ------------------------------------------------------------------------------------------------------------
class StepArrays(NamedTuple):
    """Fresh per-step arrays of the column namespace produced by a backend runner (no host read yet)."""

    pool_m: Any  # (N,)
    water_m: Any  # (L, N)
    wetting_m: Any  # (N,)
    cumulative_intake_m: Any  # (N,)
    intake_m: Any  # (N,)
    rows: Any  # (ROW_COUNT, N)


class KernelMaps(NamedTuple):
    """Parameter maps in kernel order."""

    conductivity_m_s: Any  # (N,)
    suction_m: Any  # (N,)
    vegetation: Any  # (N,)
    thickness_m: Any  # (L,)
    field_capacity_m: Any  # (L,)
    saturation_m: Any  # (L,)
    capacity_tolerance_m: Any  # (L,)


_MAP_SHAPES: tuple[Callable[[int, int], tuple[int, ...]], ...] = (
    lambda L, N: (N,), lambda L, N: (N,), lambda L, N: (N,), lambda L, N: (L,), lambda L, N: (L,), lambda L, N: (L,),
    lambda L, N: (L,))


def cell_closure_coefficient(n_layers: int) -> float:
    """MAPLE's dimensionless FP64 bound coefficient for the 2 L + CELL_OPERATION_TERMS terms of one cell's step."""
    return volume_roundoff_bound_m3(CELL_OPERATION_TERMS_PER_LAYER * int(n_layers) + CELL_OPERATION_TERMS, 1.0)


def _digest(owned: tuple, names: tuple) -> str:
    h = hashlib.sha256()
    for name, arr in zip(names, owned, strict=True):
        h.update(name.encode())
        h.update(str(arr.shape).encode())
        h.update(arr.host_bytes())
    return h.hexdigest()


class MatrixOperator:
    """Prepared matrix stage bound to one `SoilColumn`, minted ONLY by `prepare_matrix`. Owns sealed parameter maps,
    never aliases a caller array, immutable. `maps` / `host_maps` return read-only views (NumPy) or detached copies /
    sealed host mirrors (CuPy); `identity()` / `summary()` are fresh dictionaries; `verify_identity()` re-hashes
    explicitly (cold)."""

    __slots__ = (
        "_backend_info", "_host", "_maps", "_preflight", "_runner", "_seal", "backend", "cell_closure_coefficient",
        "column", "conductivity_summary", "device_id", "memory", "n_cells", "n_layers", "newton_tolerance", "sha256",
        "suction_summary", "vegetation_summary", "xp",
    )

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise MatrixError("MatrixOperator is constructed only by prepare_matrix()")
        for name in self.__slots__:
            if name == "_seal":
                continue
            if name not in fields:
                raise MatrixError(f"internal: operator field {name} missing")
            object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise MatrixError(f"internal: unexpected operator fields {sorted(fields)}")
        object.__setattr__(self, "_seal", seal)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("MatrixOperator is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("MatrixOperator is immutable")

    def __repr__(self) -> str:
        return (f"MatrixOperator(backend={self.backend!r}, n_layers={self.n_layers}, n_cells={self.n_cells}, "
                f"sha256={self.sha256[:12]}...)")

    @property
    def maps(self) -> KernelMaps:
        return KernelMaps(*(o.readonly() for o in self._maps))

    @property
    def host_maps(self) -> KernelMaps:
        return KernelMaps(*(o.readonly() for o in self._host))

    @property
    def backend_info(self) -> dict[str, Any]:
        return copy.deepcopy(self._backend_info)

    def detached_maps(self) -> KernelMaps:
        return KernelMaps(*(o.detached() for o in self._maps))

    def _kernel_maps(self) -> KernelMaps:
        return KernelMaps(*(o._internal() for o in self._maps))

    def identity(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "n_layers": self.n_layers, "n_cells": self.n_cells, "backend": self.backend,
                "fields": list(KernelMaps._fields),
                "note": "digest of the host bytes of the sealed parameter maps, computed once at build; "
                        "verify_identity() re-hashes explicitly (one download per array on cuda)"}

    def verify_identity(self) -> bool:
        if _digest(self._maps, KernelMaps._fields) != self.sha256:
            return False
        if self.backend == "cuda":
            return _digest(self._host, KernelMaps._fields) == self.sha256
        return True

    def summary(self) -> dict[str, Any]:
        return summary(self)


def _check_owned(xp: ModuleType, owned: _Owned, shape: tuple[int, ...], name: str, device_id: int | None) -> None:
    array = owned._internal()
    if type(array) is not xp.ndarray or array.dtype != np.float64 or tuple(array.shape) != shape or not array.flags.c_contiguous:
        raise MatrixError(f"operator map {name} lost its exact type/dtype/shape/contiguity; refused")
    if xp is np:
        if array.flags.writeable or not isinstance(array.base if array.base is not None else array, (np.ndarray, bytes)):
            raise MatrixError(f"operator map {name} is not sealed read-only storage; refused")
    elif int(array.device.id) != device_id:
        raise MatrixError(f"operator map {name} lives on device {int(array.device.id)}, the column on {device_id}")


def _check_operator(op: Any) -> None:
    """Metadata-only revalidation of a prepared operator (O(L), no hashing, no copy)."""
    if type(op) is not MatrixOperator:
        raise MatrixError(f"operator must be a genuine MatrixOperator, got {type(op).__name__}")
    if getattr(op, "_seal", None) is not _SEAL:
        raise MatrixError("operator record is not sealed by prepare_matrix(); refused")
    column = op.column
    ss._check_column(column)
    xp = op.xp
    if op.backend != column.backend or xp is not column.xp or op.n_layers != column.n_layers or op.n_cells != column.n_cells:
        raise MatrixError("operator record disagrees with its column (backend, namespace, layers or cells); refused")
    if op.device_id != column.device_id:
        raise MatrixError("operator record disagrees with its column's device; refused")
    L, N = op.n_layers, op.n_cells
    maps, host = op._maps, op._host
    if not isinstance(maps, tuple) or len(maps) != len(KernelMaps._fields) or not isinstance(host, tuple) or len(host) != len(maps):
        raise MatrixError("operator record carries a replaced parameter set; refused")
    for name, owned, shape_of in zip(KernelMaps._fields, maps, _MAP_SHAPES, strict=True):
        if not isinstance(owned, _Owned):
            raise MatrixError(f"operator map {name} is not an owned array; refused")
        _check_owned(xp, owned, shape_of(L, N), name, column.device_id)
    if xp is not np:
        for name, owned, shape_of in zip(KernelMaps._fields, host, _MAP_SHAPES, strict=True):
            if not isinstance(owned, _Owned):
                raise MatrixError(f"operator host mirror {name} is not an owned array; refused")
            _check_owned(np, owned, shape_of(L, N), f"host {name}", None)
    elif any(h is not m for h, m in zip(host, maps, strict=True)):
        raise MatrixError("operator record carries replaced host mirrors; refused")
    if not callable(op._runner) or not isinstance(op.sha256, str) or len(op.sha256) != 64:
        raise MatrixError("operator record carries a replaced runner or identity; refused")


def _validate_scalar_map_input(column: ss.SoilColumn, value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise MatrixError(f"{name} must be numeric, got a Boolean")
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        if not math.isfinite(v):
            raise MatrixError(f"{name} must be finite, got {value!r}")
        return
    ss._check_dynamic(column, name, value, (column.n_cells,))


def _scalar_or_cell_map(column: ss.SoilColumn, value: Any, name: str, *, low: float, high: float) -> tuple[Any, dict[str, Any]]:
    """A real host scalar or an exact (N,) array of the column namespace -> a fresh (N,) map plus a host summary.
    Bounds are checked everywhere (cold host reads on cuda). Allocation-free pre-checks ran already."""
    xp = column.xp
    n = column.n_cells
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        if v < low or v > high:
            raise MatrixError(f"{name} must lie in [{low!r}, {high!r}], got {v!r}")
        return xp.full(n, v, dtype=np.float64), {"kind": "scalar", "value": v}
    from maple.core.backend import to_float

    arr = xp.ascontiguousarray(xp.array(value, dtype=np.float64, copy=True))
    if to_float(xp.sum(~xp.isfinite(arr))) != 0.0:
        raise MatrixError(f"{name} must be finite everywhere")
    lo, hi = to_float(xp.min(arr)), to_float(xp.max(arr))
    if lo < low or hi > high:
        raise MatrixError(f"{name} must lie in [{low!r}, {high!r}] everywhere, got [{lo!r}, {hi!r}]")
    return arr, {"kind": "map", "min": lo, "max": hi}


def prepare_matrix(column: ss.SoilColumn, *, conductivity_m_s: Any, suction_m: Any, vegetation_factor: Any,
                   extra_reserve_bytes: int = 0) -> MatrixOperator:
    """Bind explicit matrix parameters to `column`: `conductivity_m_s` (K >= 0, m/s; the source's K_geom, derivable
    with `geometric_conductivity_m_s`), `suction_m` (psi >= 0, m), `vegetation_factor` (g in [0, 1]; the source
    Holling form with a1 < a2 lies in [a1/a2, 1) and g > 1 would make the source onset threshold inconsistent with a
    supply-bounded intake, so it is refused). Each is a host scalar or an exact (N,) array of the column namespace;
    no texture inference, no default site parameter. The layer vectors (thickness, field capacity, saturation,
    capacity tolerance) come from the genuine prepared column. Admission (step requirement + build scratch +
    reserves) precedes EVERY allocation; "numba" and "cuda" compile and warm up here (startup, outside steps)."""
    ss._check_column(column)
    backend = column.backend
    xp = column.xp
    L, N = column.n_layers, column.n_cells
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    _validate_scalar_map_input(column, conductivity_m_s, "conductivity_m_s")
    _validate_scalar_map_input(column, suction_m, "suction_m")
    _validate_scalar_map_input(column, vegetation_factor, "vegetation_factor")
    preflight = None
    if backend == "cuda":
        from maple_syrup import matrix_infiltration_cuda as mc

        cp = mc.require_cuda()
        if mc.current_device_id(cp) != column.device_id:
            raise MatrixBackendUnavailableError("prepare_matrix must run on the column's device")
    elif backend == "numba":
        from maple_syrup import matrix_infiltration_numba as mn

        mn.require_numba()
    memory = estimate_matrix_memory(L, N, backend=backend, reserve_bytes=column.reserve_bytes, extra_reserve_bytes=extra)
    _admit(column, memory.required_bytes + memory.build_bytes, "build")  # BEFORE any allocation
    from maple.core.backend import to_device, to_host

    k_map, k_summary = _scalar_or_cell_map(column, conductivity_m_s, "conductivity_m_s", low=0.0, high=math.inf)
    psi_map, psi_summary = _scalar_or_cell_map(column, suction_m, "suction_m", low=0.0, high=math.inf)
    g_map, g_summary = _scalar_or_cell_map(column, vegetation_factor, "vegetation_factor", low=0.0, high=1.0)
    lay = column.layers

    def layer_vector(a: np.ndarray) -> Any:
        host = np.ascontiguousarray(np.array(a, dtype=np.float64))
        return to_device(host, xp) if xp is not np else host

    owned = KernelMaps(_Owned(_SEAL, xp, k_map), _Owned(_SEAL, xp, psi_map), _Owned(_SEAL, xp, g_map),
                       _Owned(_SEAL, xp, layer_vector(lay.thickness_m)), _Owned(_SEAL, xp, layer_vector(lay.field_capacity_m)),
                       _Owned(_SEAL, xp, layer_vector(lay.saturation_m)), _Owned(_SEAL, xp, layer_vector(lay.capacity_tolerance_m)))
    del k_map, psi_map, g_map
    if xp is np:
        host_owned = owned
    else:
        host_owned = KernelMaps(*(_Owned(_SEAL, np, np.ascontiguousarray(to_host(o._internal()))) for o in owned))
    sha256 = _digest(tuple(host_owned), KernelMaps._fields)
    coefficient_cell = cell_closure_coefficient(L)
    tol = newton_tolerance()
    kernel_maps = KernelMaps(*(o._internal() for o in owned))
    if backend == "numpy":
        runner, info = reference_runner, {"implementation": "numpy reference (vectorised cells, masked Newton iteration)"}
    elif backend == "numba":
        runner, info = mn.prepare_runner(kernel_maps, coefficient_cell, tol)
    else:
        runner, info, preflight = mc.prepare_runner(kernel_maps, coefficient_cell, tol, cp, column.device_id)
    return MatrixOperator(
        _SEAL, column=column, backend=backend, xp=xp, n_layers=L, n_cells=N, _maps=tuple(owned), _host=tuple(host_owned),
        cell_closure_coefficient=coefficient_cell, newton_tolerance=tol, memory=memory, sha256=sha256, _runner=runner,
        _preflight=preflight, device_id=column.device_id, _backend_info=copy.deepcopy(dict(info)),
        conductivity_summary=MappingProxyType(dict(k_summary)), suction_summary=MappingProxyType(dict(psi_summary)),
        vegetation_summary=MappingProxyType(dict(g_summary)))


def admit_step(op: MatrixOperator) -> int:
    """Compare the operator's per-step requirement (reserves included) with the column's reported available bytes
    and raise `MatrixMemoryError` BEFORE anything is allocated. Returns the reading."""
    _check_operator(op)
    return _admit(op.column, op.memory.required_bytes, "step")


# --- wetting history -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class WettingHistory:
    """Persistent per-cell hydraulic history `wetting_m` (J, m; NOT a water store) and the separate lifetime actual
    matrix-intake ledger `cumulative_intake_m` (m), both (N,) float64 of the column namespace, owned by the record,
    bound to the operator identity they were created for. Built by `wetting_history` or a step; never written."""

    wetting_m: Any
    cumulative_intake_m: Any
    operator_sha256: str
    n_cells: int


def _history_array(op: MatrixOperator, value: Any, name: str) -> Any:
    if isinstance(value, (bool, np.bool_)):
        raise MatrixError(f"{name} must be numeric, got a Boolean")
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        if not math.isfinite(v) or v < 0.0:
            raise MatrixError(f"{name} must be finite and >= 0, got {value!r}")
        return None, v
    ss._check_dynamic(op.column, name, value, (op.n_cells,))
    return value, None


def wetting_history(op: MatrixOperator, *, wetting_m: Any, cumulative_intake_m: Any) -> WettingHistory:
    """Explicit history at a genuinely initialised run/test boundary (fresh episode: both 0.0) or restored from an
    accepted record (exact (N,) arrays of the column namespace, copied; values are validated by the next step's
    batched checks, inactive cells included). Bound to `op.sha256`. Admitted (step requirement + the two copies)
    before allocation. No reset or recovery policy is provided."""
    _check_operator(op)
    w_arr, w_val = _history_array(op, wetting_m, "wetting_m")
    c_arr, c_val = _history_array(op, cumulative_intake_m, "cumulative_intake_m")
    n = op.n_cells
    _admit(op.column, op.memory.required_bytes + 2 * n * _F8, "history")
    xp = op.xp
    w = xp.full(n, w_val, dtype=np.float64) if w_arr is None else xp.ascontiguousarray(xp.array(w_arr, dtype=np.float64, copy=True))
    c = xp.full(n, c_val, dtype=np.float64) if c_arr is None else xp.ascontiguousarray(xp.array(c_arr, dtype=np.float64, copy=True))
    return WettingHistory(w, c, op.sha256, n)


def _check_history(op: MatrixOperator, history: Any) -> None:
    if not isinstance(history, WettingHistory):
        raise MatrixError(f"history must be a WettingHistory, got {type(history).__name__}")
    if history.operator_sha256 != op.sha256 or history.n_cells != op.n_cells:
        raise MatrixError("history is bound to a different operator identity (other K/psi/g maps, layers or column); "
                          "a hydraulic history is never reinterpreted under other parameters; refused")
    ss._check_dynamic(op.column, "history.wetting_m", history.wetting_m, (op.n_cells,))
    ss._check_dynamic(op.column, "history.cumulative_intake_m", history.cumulative_intake_m, (op.n_cells,))


# --- the reference step -------------------------------------------------------------------------------------------------
class _StepContext(NamedTuple):
    """PRIVATE honest context of the shared NumPy body: exactly what the body needs besides the arrays. It is NOT a
    column, layers object or operator (Phase 2 S2-B2.a: the explicit spatial operator supplies the same fields from its
    own sealed owner, so no fake legacy column facade exists)."""

    n_layers: int
    n_cells: int
    active: Any  # (N,) bool
    cell_tol: float  # cell_closure_coefficient(L)
    newton_tol: float  # newton_tolerance()


def reference_step(op: MatrixOperator, maps: KernelMaps, pool_in: np.ndarray, standing_in: np.ndarray, rate_in: np.ndarray,
                   water_in: np.ndarray, wetting_in: np.ndarray, cumulative_in: np.ndarray, dt: float) -> StepArrays:
    """NumPy reference of one step: vectorised over cells with the SAME IEEE operations in the SAME order as the
    compiled kernels (only `log1p`/`log` may differ by an ulp between libraries). Validation is recorded in flag
    rows, not raised here. The body is `_reference_step_body`, shared with the explicit spatial operator; this wrapper
    passes the operator's cached `(L,)` vectors exactly as before."""
    k_map, psi_map, g_map, dz, fc_m, sat_m, cap_tol = maps
    ctx = _StepContext(op.n_layers, op.n_cells, op.column.active, op.cell_closure_coefficient, op.newton_tolerance)
    return _reference_step_body(ctx, k_map, psi_map, g_map, dz, fc_m, sat_m, cap_tol, pool_in, standing_in, rate_in,
                                water_in, wetting_in, cumulative_in, dt)


def _reference_step_body(ctx: _StepContext, k_map: Any, psi_map: Any, g_map: Any, dz: Any, fc_m: Any, sat_m: Any,
                         cap_tol: Any, pool_in: np.ndarray, standing_in: np.ndarray, rate_in: np.ndarray,
                         water_in: np.ndarray, wetting_in: np.ndarray, cumulative_in: np.ndarray, dt: float) -> StepArrays:
    """The ONE NumPy physical body. `k_map`/`psi_map`/`g_map` are `(N,)` arrays (read-only broadcast views allowed);
    `dz`, `fc_m`, `sat_m`, `cap_tol` are INDEXABLE per-layer readers (`x[layer]` -> a scalar or an `(N,)` row): the
    cached `(L,)` arrays of the common operator, or lazy readers of an explicit spatial owner. Only `dz[0]`,
    `fc_m[0]`, `sat_m[0]` and `sat_m[l] + cap_tol[l]` are read."""
    from maple.core.backend import errstate

    xp = np
    L, N = ctx.n_layers, ctx.n_cells
    active = ctx.active
    inactive = ~active
    cell_tol = ctx.cell_tol

    pool_out = xp.array(pool_in, dtype=np.float64, copy=True)
    water_out = xp.array(water_in, dtype=np.float64, copy=True)
    wet_out = xp.array(wetting_in, dtype=np.float64, copy=True)
    cum_out = xp.array(cumulative_in, dtype=np.float64, copy=True)
    intake = xp.zeros(N, dtype=np.float64)
    rows = xp.zeros((ROW_COUNT, N), dtype=np.float64)
    before, after = rows[0], rows[1]
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
        flag(~xp.isfinite(pool_in), 3)
        flag(pool_in < 0.0, 4)
        flag(~xp.isfinite(standing_in), 5)
        flag(standing_in < 0.0, 6)
        flag(~xp.isfinite(rate_in), 7)
        flag(rate_in < 0.0, 8)
        flag(inactive & (rate_in != 0.0), 9)
        flag(~xp.isfinite(wetting_in), 10)
        flag(wetting_in < 0.0, 11)
        flag(~xp.isfinite(cumulative_in), 12)
        flag(cumulative_in < 0.0, 13)
        xp.copyto(rows[2], pool_in)
        xp.copyto(rows[10], wetting_in)
        after[...] = before
        # no-contact consistency: pool <= r dt within the FP64 bound (active cells)
        inc = rate_in * dt
        contact = standing_in > 0.0
        scale_c = (pool_in + inc) * cell_tol
        flag(active & ~contact & (pool_in > inc + scale_c), 14)
        del scale_c
        if dt > 0.0:
            w0 = water_in[0]
            deficit = fc_m[0] - w0
            xp.copyto(deficit, 0.0, where=deficit < 0.0)
            b = psi_map * (deficit / dz[0])
            del deficit
            room = sat_m[0] - w0
            xp.copyto(room, 0.0, where=room < 0.0)
            # regime selection
            low = ~contact & (rate_in <= k_map)
            high = ~contact & (rate_in > k_map)
            fs = ((g_map * b) * k_map) / (rate_in - k_map)
            already = high & (wetting_in >= fs)
            t_p = (fs - wetting_in) / rate_in
            inside = high & ~already & (t_p < dt)
            whole = low | (high & ~already & ~(t_p < dt))
            del fs
            t_u = xp.where(whole, dt, xp.where(inside, t_p, 0.0))
            tau = xp.where(contact | already, dt, xp.where(inside, dt - t_p, 0.0))
            run_post = contact | already | inside
            del t_p, low, high, already
            # unponded phase
            c1 = rate_in * t_u
            pre = xp.array(c1)
            xp.copyto(pre, pool_in, where=pool_in < pre)
            xp.copyto(pre, room, where=room < pre)
            candidate = xp.array(c1)
            exhausted = (t_u > 0.0) & (pre < c1)
            run_post &= ~exhausted
            onset = inside & ~exhausted
            del c1, exhausted, inside, whole, t_u
            j1 = wetting_in + pre
            # ponded phase
            r_dt = xp.where(run_post, k_map * tau, 0.0)
            del tau
            x, iters, conv = _ga_increment_vec(xp, j1, b, r_dt, ctx.newton_tol, NEWTON_MAX_ITERATIONS)
            del r_dt
            flag(active & run_post & ~conv, 15)
            c2 = g_map * x
            del x, conv
            xp.add(candidate, c2, out=candidate, where=run_post)
            pool_left = pool_in - pre
            room_left = room - pre
            post = xp.where(run_post, c2, 0.0)
            xp.copyto(post, pool_left, where=run_post & (pool_left < post))
            xp.copyto(post, room_left, where=run_post & (room_left < post))
            del c2, pool_left, room_left
            # ONE combined actual intake (S2-B2.c review02): bound the recombination `pre + post` by the offered pool and the top-layer room BEFORE anything is published. When the bound is inactive every
            # value below is byte-identical to the previous arithmetic; only when it is ACTIVE (the recombination of two separately bounded parts exceeded a bound by roundoff) is the ponded actual
            # re-derived from the bounded total, `post = intake - pre` (>= 0 exactly: pre <= pool and pre <= room), so pool, soil, history and ledger share the SAME actuals. No epsilon, no clipping afterwards.
            xp.add(pre, post, out=intake)
            over = intake > pool_in
            xp.copyto(intake, pool_in, where=over)
            over_room = intake > room
            xp.copyto(intake, room, where=over_room)
            over |= over_room
            del over_room, room
            over &= run_post
            xp.subtract(intake, pre, out=post, where=over)
            del over
            adv = xp.where(run_post & (g_map > 0.0), post / g_map, 0.0)
            j_out = xp.where(run_post, j1 + adv, j1)
            xp.copyto(wet_out, j_out, where=active)
            del j_out
            bounded = xp.array(candidate)
            xp.copyto(bounded, pool_in, where=pool_in < bounded)
            rejected = bounded - intake
            unused = candidate - bounded
            del bounded
            xp.copyto(water_out[0], w0 + intake, where=active)
            xp.copyto(pool_out, pool_in - intake, where=active)
            xp.copyto(cum_out, cumulative_in + intake, where=active)
            # output checks (active cells only)
            after[...] = 0.0
            for layer in range(L):
                wl = water_out[layer]
                flag(active & ~xp.isfinite(wl), 16)
                flag(active & (wl < 0.0), 17)
                flag(active & (wl > sat_m[layer] + cap_tol[layer]), 18)
                xp.add(after, wl, out=after)
            for vector in (pool_out, wet_out, cum_out):
                flag(active & ~xp.isfinite(vector), 16)
                flag(active & (vector < 0.0), 17)
            scale = before + after
            xp.add(scale, pool_in, out=scale)
            xp.add(scale, pool_out, out=scale)
            xp.add(scale, intake, out=scale)
            xp.multiply(scale, cell_tol, out=scale)
            closure = after - before
            xp.subtract(closure, intake, out=closure)
            xp.abs(closure, out=closure)
            flag(active & (closure > scale), 19)
            xp.subtract(pool_in, pool_out, out=closure)
            xp.subtract(closure, intake, out=closure)
            xp.abs(closure, out=closure)
            flag(active & (closure > scale), 20)
            adv_total = pre + adv
            h_scale = wetting_in + wet_out
            xp.add(h_scale, adv_total, out=h_scale)
            xp.multiply(h_scale, cell_tol, out=h_scale)
            xp.subtract(wet_out, wetting_in, out=closure)
            xp.subtract(closure, adv_total, out=closure)
            xp.abs(closure, out=closure)
            flag(active & (closure > h_scale), 21)
            c_scale = cumulative_in + cum_out
            xp.add(c_scale, intake, out=c_scale)
            xp.multiply(c_scale, cell_tol, out=c_scale)
            xp.subtract(cum_out, cumulative_in, out=closure)
            xp.subtract(closure, intake, out=closure)
            xp.abs(closure, out=closure)
            flag(active & (closure > c_scale), 21)
            flag(active & (~xp.isfinite(candidate) | ~xp.isfinite(intake)), 22)
            del closure, scale, h_scale, c_scale, adv_total
            # rows
            xp.copyto(rows[4], candidate)
            xp.copyto(rows[6], pre)
            xp.copyto(rows[7], post)
            xp.copyto(rows[8], rejected)
            xp.copyto(rows[9], unused)
            xp.copyto(rows[12], onset.astype(np.float64))
            xp.copyto(rows[13], run_post.astype(np.float64))
            xp.copyto(rows[14], iters)
            del candidate, pre, post, rejected, unused, onset, run_post, iters, adv, j1, b
        xp.copyto(rows[5], intake)
        xp.copyto(rows[3], pool_out)
        xp.copyto(rows[11], wet_out)
        for row in range(ROW_COUNT - 1):
            xp.copyto(rows[row], 0.0, where=inactive)
        xp.copyto(intake, 0.0, where=inactive)
        low_bit = xp.negative(flags)
        xp.bitwise_and(flags, low_bit, out=low_bit)
        rows[15][...] = low_bit
        xp.copyto(rows[15], NO_FLAG, where=flags == 0)
        del low_bit
    return StepArrays(pool_out, water_out, wet_out, cum_out, intake, rows)


def reference_runner(op: MatrixOperator, maps: KernelMaps, pool_in: Any, standing_in: Any, rate_in: Any, water_in: Any,
                     wetting_in: Any, cumulative_in: Any, dt: float) -> StepArrays:
    return reference_step(op, maps, pool_in, standing_in, rate_in, water_in, wetting_in, cumulative_in, dt)


# --- step ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MatrixBudget:
    """Host volumes (m3) over active cells from ONE packet read, counts, and the MAPLE-derived closure bound.
    `wetting_*_m3` are the history coordinate times the cell area (an equivalent volume for reporting, NOT water)."""

    soil_before_m3: float
    soil_after_m3: float
    pool_before_m3: float
    pool_after_m3: float
    candidate_m3: float  # r t_u + g x: the physical capacity candidate before the pool/room bounds
    intake_m3: float  # actual, credited once to the top layer and debited once from the pool
    prepond_intake_m3: float  # admitted through the unponded branch (advances J unscaled)
    postpond_intake_m3: float  # admitted through the ponded law (advances J by intake / g)
    room_rejected_m3: float  # min(candidate, pool) - intake: wanted in, top layer full, stays in the pool
    unused_capacity_m3: float  # per cell candidate - min(candidate, pool): capacity without water to admit
    wetting_before_m3: float
    wetting_after_m3: float
    n_onsets: int  # cells whose onset lies inside the interval (the labelled capacity discontinuity)
    n_ponded: int  # cells that ran the ponded law for some part of the interval
    max_newton_iterations: int
    soil_residual_m3: float
    pool_residual_m3: float
    bound_m3: float
    n_terms: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True, eq=False)
class MatrixStep:
    """Fresh arrays of the column namespace plus the batched scalar budget. `state.t_s` is PRESERVED."""

    surface_pool_m: Any  # (N,) remaining pool
    state: ss.SoilColumnState
    history: WettingHistory
    intake_m: Any  # (N,)
    dt_s: float
    budget: MatrixBudget


def _read_packet(xp: ModuleType, arrays: StepArrays) -> np.ndarray:
    from maple.core.backend import to_host

    rows = arrays.rows
    values = [xp.sum(rows[r]) for r in range(14)] + [xp.max(rows[14]), xp.min(rows[15]), xp.count_nonzero(rows[15] < NO_FLAG)]
    stacked = xp.stack([xp.asarray(v, dtype=np.float64) for v in values])
    return np.asarray(to_host(stacked), dtype=np.float64)


def _resolve(op: MatrixOperator, packet: np.ndarray) -> MatrixBudget:
    return _resolve_core(op.column.cell_area_m2, op.n_layers, op.n_cells, packet)


def _resolve_core(area: float, n_layers: int, n_cells: int, packet: np.ndarray) -> MatrixBudget:
    """Backend- and owner-independent budget of ONE packet (private; shared with the explicit spatial operator):
    `area` is applied ONCE, finiteness is checked BEFORE the strict MAPLE bound helper."""
    if packet.shape != (PACKET_LENGTH,):
        raise MatrixError(f"internal: packet shape {packet.shape}")
    low, n_flagged = float(packet[15]), int(packet[16])
    if not math.isfinite(low) or low < NO_FLAG:
        bit = int(low).bit_length() - 1 if math.isfinite(low) and low >= 1.0 else None
        message = FLAG_MESSAGES[bit] if bit is not None and 0 <= bit < N_FLAGS else "validation packet is corrupt"
        raise MatrixStepError(f"{message} ({n_flagged} cell(s) flagged)", bit=bit, n_flagged=n_flagged)
    (before, after, pool_in, pool_out, candidate, intake, pre, post, rejected, unused, wet_in, wet_out) = (
        float(v) * area for v in packet[:12])
    n_onsets, n_ponded, max_iter = int(packet[12]), int(packet[13]), int(packet[14])
    soil_residual = (after - before) - intake
    pool_residual = (pool_in - pool_out) - intake
    n_terms = (CELL_OPERATION_TERMS_PER_LAYER * n_layers + CELL_OPERATION_TERMS) * n_cells + ROW_COUNT
    operands = (abs(before), abs(after), abs(pool_in), abs(pool_out), abs(intake))
    if not all(math.isfinite(v) for v in (*operands, candidate, pre, post, rejected, unused, wet_in, wet_out, soil_residual,
                                           pool_residual)):
        raise MatrixStepError("non-finite global water budget", bit=None, n_flagged=n_flagged)
    bound = volume_roundoff_bound_m3(n_terms, max(operands))
    if abs(soil_residual) > bound or abs(pool_residual) > bound:
        raise MatrixStepError(
            f"global water closure violated: soil residual {soil_residual!r} m3, pool residual {pool_residual!r} m3 "
            f"against the FP64 bound {bound!r} m3 ({n_terms} terms)", bit=None, n_flagged=n_flagged)
    return MatrixBudget(before, after, pool_in, pool_out, candidate, intake, pre, post, rejected, unused,
                        wet_in, wet_out, n_onsets, n_ponded, max_iter, soil_residual, pool_residual, bound, n_terms)


def matrix_step(op: MatrixOperator, surface_pool_m: Any, standing_before_m: Any, incoming_rate_m_s: Any,
                state: ss.SoilColumnState, history: WettingHistory, dt_s: Any) -> MatrixStep:
    """One pure matrix stage over `dt_s` (convention B, see the module docstring). `surface_pool_m` (N,) is the
    actual pool offered over the interval (supplied once by the owner); `standing_before_m` (N,) the preexisting
    standing depth at the interval start (contact descriptor); `incoming_rate_m_s` (N,) the onset descriptor;
    `state` the authoritative column state; `history` the persistent record bound to `op`. All inputs are read only;
    dt = 0 is the identity (history preserved); a refused step allocates no result and publishes nothing."""
    _check_operator(op)
    column = op.column
    t0 = ss._check_state(column, state)
    dt = ss._check_dt(dt_s)
    n = (column.n_cells,)
    ss._check_dynamic(column, "surface_pool_m", surface_pool_m, n)
    ss._check_dynamic(column, "standing_before_m", standing_before_m, n)
    ss._check_dynamic(column, "incoming_rate_m_s", incoming_rate_m_s, n)
    _check_history(op, history)
    if op._preflight is not None:
        op._preflight()
    _admit(column, op.memory.required_bytes, "step")
    arrays = op._runner(op, op._kernel_maps(), surface_pool_m, standing_before_m, incoming_rate_m_s, state.water_m,
                        history.wetting_m, history.cumulative_intake_m, dt)
    packet = _read_packet(op.xp, arrays)  # the ONE counted host read; synchronising on cuda
    budget = _resolve(op, packet)
    return MatrixStep(surface_pool_m=arrays.pool_m, state=ss.SoilColumnState(arrays.water_m, t0),
                      history=WettingHistory(arrays.wetting_m, arrays.cumulative_intake_m, op.sha256, op.n_cells),
                      intake_m=arrays.intake_m, dt_s=dt, budget=budget)


# --- provenance ------------------------------------------------------------------------------------------------------
def summary(op: MatrixOperator) -> dict[str, Any]:
    _check_operator(op)
    return {
        "backend": op.backend,
        "namespace": op.xp.__name__,
        "device_id": op.device_id,
        "n_layers": op.n_layers,
        "n_cells": op.n_cells,
        "n_active": op.column.n_active,
        "conductivity_m_s": dict(op.conductivity_summary),
        "suction_m": dict(op.suction_summary),
        "vegetation_factor": dict(op.vegetation_summary),
        "law": "paper-first local top-layer deficit b = psi max(theta_fc - theta_0, 0); unponded intake advances J "
               "unscaled; onset threshold F_s = g b K/(r - K); ponded dJ/dt = K (1 + b/J), dM = g dJ, implicit "
               "x - b log1p(x/(J + b)) = K tau; admitted ponded intake advances J by intake/g (g = 0: no postpond "
               "intake, J unchanged); onset capacity jump r -> r - (1 - g) K labelled via n_onsets",
        "supply_convention": SUPPLY_CONVENTION,
        "history": "wetting_m (J) persistent and bound to the operator identity; cumulative_intake_m separate "
                   "lifetime ledger; preserved across calls, splits and checkpoints; no reset/recovery policy",
        "stage_order": list(STAGE_ORDER),
        "time": "state.t_s preserved; the owning hydraulic step advances time once",
        "newton": {"max_iterations": NEWTON_MAX_ITERATIONS, "relative_tolerance": op.newton_tolerance,
                   "tolerance_terms": NEWTON_TOLERANCE_TERMS, "start": "smaller of sqrt(R^2 + 2 R b) and R b / J_old "
                   "(upper bounds), monotone Newton from above; non-convergence is flagged, never clipped"},
        "memory": op.memory.as_dict(),
        "admission": "required_bytes (+ build_bytes at build) compared with the column's available_bytes_fn before "
                     "prepare_matrix, wetting_history and every step; a reading, not a reservation",
        "closure": f"conservation.volume_roundoff_bound_m3: per cell {CELL_OPERATION_TERMS_PER_LAYER} L + "
                   f"{CELL_OPERATION_TERMS} terms, global ({CELL_OPERATION_TERMS_PER_LAYER} L + {CELL_OPERATION_TERMS}) N "
                   f"+ {ROW_COUNT} terms",
        "identity": op.identity(),
        "sealing": "factory-only records; numpy maps are bytes-backed read-only storage; cuda maps are private with "
                   "detached public copies and sealed host mirrors; every step re-validates the record by metadata only",
        "packet_length": PACKET_LENGTH,
        "host_reads_per_step": 1,
        "backend_info": op.backend_info,
        "source": dict(SOURCE_IDENTITY),
        "scope": "isolated staged matrix intake into the top layer: no rain addition, preferential stage, "
                 "redistribution, E/T/growth, routing, checkpoint codec or recovery policy; not an EcoHyD identity claim",
    }
