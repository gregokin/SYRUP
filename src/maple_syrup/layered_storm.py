"""Pure persistent layered-water storm step (Phase 1 composition, task phase1_persistent_water_step_20261110).

SYRUP owns all physical water. This module composes the EXISTING accepted components into ONE pure, fixed-bed,
water-only, single-wetting-episode trial step over `[t0, t0 + dt]` on NumPy, Numba or CUDA, without copying or
re-deriving any physics:

    op = prepare_layered_storm(column, preferential, matrix, router, graph, topology, bed=None, context=None)
    accepted = initial_layered_state(op, soil=soil_state, history=history, t_s=0.0, surface_m=depth)   # or the native bed's
    candidate = layered_step(op, accepted, rain_rate_m_s, dt_s)       # LayeredStepCandidate: fresh aggregate + receipt
    accepted = candidate.state                                         # the CALLER adopts the whole candidate, or drops it

Stage order of one trial (every stage is the existing pure function; the owner only forms the pool, adds the
saturation return, binds times and checks closure):

  1. validate types/bindings/time/scalars (metadata only), admit the COMPOSED peak against the column's reading;
  2. rain ONCE: `pool = H0 + rain dt` on the flat cells (inactive rain must be 0, so inactive pool == H0);
  3. `preferential_infiltration.preferential_step` on the pool and the accepted soil (clock preserved);
  4. `matrix_infiltration.matrix_step` on the remaining pool with `standing_before = ORIGINAL H0` (contact descriptor)
     and `incoming_rate = actual rain` (onset descriptor, never a water source; run-on is accepted lateral water of
     the previous interval and enters as standing water, never as pool/dt);
  5. `subsurface.step` on the already credited soil with ZERO added water and ZERO potential evaporation (gravity
     drainage, redistribution and saturation return act; evaporation is exactly 0); the soil advances t0 -> t1;
  6. `router.step(SurfaceState(t0, remaining pool + surface return, qx0, qy0), dt)` -- the ACCEPTED start time, not
     the advanced soil time; the surface end time must equal the soil end time `t0 + dt`;
  7. optional genuine native wrapper `sediment_bed.with_water(bed0, context, routed depth)` (water only; the accepted
     all-zero mobile array is reused), then ONE batched owner packet: per-cell surface and soil closure with
     MAPLE-derived bounds, exact inactive preservation, zero evaporation, finite bounds and the composed global
     budget; a fresh `LayeredWaterState` (generation + 1, advanced ledger) is returned.

Contracts. Pure: no accepted array (surface, momentum, layers, history, native bed, ledger, counters) is written;
every trial array is local until the candidate exists. The ONLY recoverable failure is the router's
`HydraulicStepRejected` (CFL / negative depth): retry the SAME accepted aggregate at a smaller dt, which reruns ALL
stages from the ORIGINAL accepted state (trial intake, history advance and drainage are discarded with the trial).
Source/type/shape/device/time/memory/non-finite errors are typed refusals, never retried by shortening dt. No hidden
dry reset, no automatic front reset, no sediment exchange, no reroute/commit/publication, no E/T/growth, no
checkpoint, event loop, scheduler, CLI or LEAF. The native surface water is ONE array: when a bed is held,
`state.bed.water.depth_m is state.surface_m` (never two authoritative surfaces); face momentum is SYRUP numerical state.

Memory. `estimate_layered_memory` composes the components' declared step allocations into the trial peak that is
admitted BEFORE the pool is allocated (see `MEMORY_SCOPE`); the stages then repeat their own admissions. Cold binding
hashes once (`prepare_layered_storm`); hot steps are metadata-only and never revisit voxel columns.

Closure. Only `conservation.volume_roundoff_bound_m3` (MAPLE's linear `4 n eps |scale|`): per cell the surface
residual against `coeff(SURFACE_CELL_TERMS(L))` and the soil residual against `coeff(SOIL_CELL_TERMS(L))`, both
term counts the SUM of the stages' own per-cell counts plus the owner's operations (derived in the task design); the
gross absolute face volumes enter the surface scale. Globally the residual is bounded by the router's own residual
(already verified against its `BALANCE_RTOL` policy) plus the preferential, matrix and column stage bounds plus the
owner's `coeff(OWNER_GLOBAL_TERMS_PER_CELL N + OWNER_GLOBAL_TERMS, max operand)`. Internal transfers cancel exactly
once; the external terms are rain, bottom drainage and outlet export.

Nothing here was run on a GPU by its author; Codex records executions.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np

from maple_syrup import matrix_infiltration as mi
from maple_syrup import preferential_infiltration as pi
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.experimental_hydrology import CpuSurfaceRouter, HydraulicControl
from maple_syrup.pond_network import PondTopology
from maple_syrup.routing import BALANCE_RTOL, RoutingGraph
from maple_syrup.sediment_bed import BedContext, BedState, current_routing_surface_m, with_water
from maple_syrup.surface_routing import (
    ExperimentalHydrologyError,
    HydraulicStepRejected,
    SurfaceState,
    SurfaceStep,
    advance_time,
)

__all__ = [
    "BACKENDS",
    "FLAG_MESSAGES",
    "MEMORY_SCOPE",
    "NO_FLAG",
    "OWNER_GLOBAL_TERMS",
    "OWNER_GLOBAL_TERMS_PER_CELL",
    "OWNER_PACKET_LENGTH",
    "OWNER_SCRATCH_DOUBLES_PER_CELL",
    "OWNER_SCRATCH_MASK_BYTES_PER_CELL",
    "OWNER_SOIL_CELL_TERMS",
    "OWNER_SURFACE_CELL_TERMS",
    "ROUTER_CELL_TERMS",
    "STAGE_ORDER",
    "SURFACE_METHOD",
    "HydraulicStepRejected",
    "LayeredMemory",
    "LayeredMemoryError",
    "LayeredStepCandidate",
    "LayeredStepError",
    "LayeredStepReceipt",
    "LayeredStormError",
    "LayeredStormOperator",
    "LayeredWaterLedger",
    "LayeredWaterState",
    "VerticalArrays",
    "admit_step",
    "describe",
    "estimate_layered_memory",
    "initial_layered_state",
    "layered_step",
    "prepare_layered_storm",
    "soil_cell_terms",
    "surface_cell_terms",
]

BACKENDS = ss.BACKENDS
SURFACE_METHOD = "local_inertial"  # the step carries face momentum; the explicit method is refused by name
STAGE_ORDER = ("validation_and_admission", "rain_once", "preferential", "matrix", "column_zero_forcing",
               "surface_return_rejoin", "lateral_routing_at_t0", "native_water_wrapper", "owner_packet")
NO_FLAG = ss.NO_FLAG
_F8 = 8
_SEAL = object()

# --- derived term counts (see design.md section 5) ---------------------------------------------------------------------
#: The router's per-cell balance policy `BALANCE_RTOL x scale` expressed in MAPLE coefficient terms (8 for 32 eps / 4 eps).
ROUTER_CELL_TERMS = int(math.ceil(BALANCE_RTOL / volume_roundoff_bound_m3(1, 1.0)))
#: Owner operations on one cell's surface value: pool (rain dt, + H0) 2, post-pool (+ return) 1, residual evaluation
#: (H1 - H0, x A; rain dt - pref - matrix + return, x A; four face-volume conversions dt dx q vs A dt/dx q (8), lateral
#: sum (3), two subtractions) 21 -> 24.
OWNER_SURFACE_CELL_TERMS = 24
#: Owner operations on one cell's soil value beyond the two layer sums (2 L): flux combination 4, scaling/compare 4.
OWNER_SOIL_CELL_TERMS = 8
#: Owner roundoff NOT covered by any stage residual, globally: pool (2) and post-pool (1) per cell, the three active
#: N-term reductions (rain dt, H0, H1) per cell, plus the host scalar combination.
OWNER_GLOBAL_TERMS_PER_CELL = 6
OWNER_GLOBAL_TERMS = 16
#: Owner packet: 3 reductions, 3 maxima, lowest flag, flagged count, 8 router scalars, gross face volume.
OWNER_PACKET_LENGTH = 17
#: Declared transient owner scratch per cell (doubles): rain dt, lateral, gross (+ face temporaries), vertical flux,
#: surface residual/scale, two layer sums, soil flux/residual/scale, flags, lowest bit, reduction temporaries.
OWNER_SCRATCH_DOUBLES_PER_CELL = 24
OWNER_SCRATCH_MASK_BYTES_PER_CELL = 16
_PACKET_SCRATCH_BYTES = 4 * OWNER_PACKET_LENGTH * _F8
#: Native water-state validation inside `with_water` (`maple.water.validate_water_state`, correction01): on the HOST the
#: genuine validator forms one boolean mask of the mobile array (N K bytes; `all_finite`, then `any_negative`
#: sequentially) and one of the depth (N bytes); on CUDA MAPLE's fused predicates return 0-d flags (no mask). Measured
#: warm host peaks (N = 1024): K = 1 -> 1289 B, 64 -> 65 801 B, 256 -> 262 409 B, 1024 -> 1 048 841 B, i.e. N K + ~265 B.
#: `NATIVE_OBJECT_BYTES` is the fixed Python-object allowance of that call (measured 230-500 B).
NATIVE_OBJECT_BYTES = 4096
#: Cold native identity hashing streams the material arrays in bounded row chunks (host: a memoryview, no copy; device:
#: one bounded host download per chunk), never a whole `tobytes()` copy of an (N, K) array.
HASH_CHUNK_BYTES = 1 << 20


def native_validation_bytes(n_cells: int, n_classes: int, backend: str) -> int:
    """Transient bytes of the genuine native water-state validation performed by `with_water` on the admitted side
    (see the constants above): host backends `N K + N + NATIVE_OBJECT_BYTES`; cuda `NATIVE_OBJECT_BYTES` (fused flags)."""
    N, K = ss._checked_int(n_cells, "n_cells"), ss._checked_int(n_classes, "n_classes")
    if backend == "cuda":
        return NATIVE_OBJECT_BYTES
    return N * K + N + NATIVE_OBJECT_BYTES


def surface_cell_terms(n_layers: int) -> int:
    """Per-cell surface closure terms: preferential (4 L + 4) + matrix (2 L + 16) + router (ROUTER_CELL_TERMS) + owner."""
    L = int(n_layers)
    return (pi.CELL_OPERATION_TERMS_PER_LAYER * L + pi.CELL_OPERATION_TERMS) + \
        (mi.CELL_OPERATION_TERMS_PER_LAYER * L + mi.CELL_OPERATION_TERMS) + ROUTER_CELL_TERMS + OWNER_SURFACE_CELL_TERMS


def soil_cell_terms(n_layers: int) -> int:
    """Per-cell soil closure terms: preferential (4 L + 4) + matrix (2 L + 16) + column (4 L + 7) + owner (2 L + 8)."""
    L = int(n_layers)
    return (pi.CELL_OPERATION_TERMS_PER_LAYER * L + pi.CELL_OPERATION_TERMS) + \
        (mi.CELL_OPERATION_TERMS_PER_LAYER * L + mi.CELL_OPERATION_TERMS) + (4 * L + 7) + (2 * L + OWNER_SOIL_CELL_TERMS)


FLAG_MESSAGES = (
    "rain_rate_m_s must be finite everywhere",  # 0
    "rain_rate_m_s must be >= 0 everywhere",  # 1
    "rain_rate_m_s must be 0 on inactive cells (no forcing on inactive cells)",  # 2
    "an inactive cell's surface water changed (inactive cells must be preserved exactly)",  # 3
    "an inactive cell's soil water changed (inactive cells must be preserved exactly)",  # 4
    "an inactive cell's wetting history changed (inactive cells must be preserved exactly)",  # 5
    "per-cell surface water closure violated beyond the derived FP64 bound",  # 6
    "per-cell soil water closure violated beyond the derived FP64 bound",  # 7
    "non-finite owner closure residual or scale",  # 8
    "evaporation or unmet evaporation is non-zero in a storm step (E/T must be exactly 0)",  # 9
)
N_FLAGS = len(FLAG_MESSAGES)

MEMORY_SCOPE = (
    "Composed water-owner ledger under the EXISTING column convention (`subsurface.estimate_step_memory`: the whole working "
    "set held for one step plus the reserve is the admitted requirement). ADMITTED before every step, against the COLUMN's "
    "available_bytes_fn on the column's side (host for numpy/numba, device for cuda): required_bytes = resident_bytes + "
    "incremental_bytes + column reserve + extra reserve + native reserve. resident_bytes = the accepted aggregate (surface, "
    "faces, layers, history; with a native bed its mobile (N, K) array, while its CURRENT depth IS the surface and is counted "
    "once, never as a second reservoir) + the owned prepared arrays (preferential/matrix parameter maps and weights, column "
    "mask/parameter copies, router statics, the owner's zero forcing, inactive mask and saturation bound) + with a native "
    "bed `native_reference_depth_bytes` = 8 N: the operator's immutable `op.bed` retains the ORIGINAL native depth H0 for "
    "its whole life, while after an advancing step the accepted state's depth H1 is a fresh array, so the live set holds "
    "two (N,) depth arrays. The charge is a conservative constant from preparation (also while the initial surface still "
    "aliases H0); it is a retained REFERENCE, not a water inventory or a second reservoir, and not timestep history (older "
    "accepted surfaces are released when the caller drops them). "
    "incremental_bytes (= trial_bytes, the bytes ONE trial newly allocates, for diagnostics) = the trial pool (N doubles) "
    "+ the declared step allocations (outputs + scratch + packet) of the preferential, matrix, column and surface-router "
    "stages + the post-vertical pool (N doubles) + the owner's closure scratch and packet + the genuine native water "
    "validation masks of `with_water` (K-dependent, `native_validation_bytes`). The stage sum is a bound even if every "
    "stage output were live at once (the true peak is lower because stage scratch is released between stages); it is NOT "
    "the sum of the components' separate required_bytes. SEMANTICS: under a fresh system-free reading the resident bytes "
    "are already allocated and are counted again (conservative, as the column estimator counts its already-resident "
    "inputs); under a fixed caller budget (a constant reading) required_bytes is the whole-owner quota. A fixed budget is "
    "never treated as extra free memory. COLD (prepare): build_bytes on the admitted side (zero forcing, masks, saturation "
    "bound, the mobile zero check which allocates no mask) and build_host_bytes of host staging (mask and routing-surface "
    "downloads, one bounded hash chunk), admitted before any download. NATIVE RESERVE: caller-owned physical "
    "material outside the water owner (the known, internally retained H0 above is charged by the estimator, NOT left to "
    "this reserve) (voxel column, active layer, ledger, availability, MAPLE's own routing-surface diagnostic scratch at cold binding, "
    "measured 1.6-42 MB at N = 1024 for K = 1-1024 with 40 voxels, V- and K-dependent) is represented ONLY by the caller's "
    "explicit native_reserve_bytes; no whole-model fit is claimed. EXCLUDED: allocator pools and fragmentation, CUDA "
    "context/NVRTC/Numba JIT workspaces, Python objects beyond the components' allowances, checkpoint staging, LEAF and the "
    "remaining process memory; the stages' own admissions are repeated inside each stage. A reading is not a reservation."
)


# --- errors ----------------------------------------------------------------------------------------------------------
class LayeredStormError(ValueError):
    """Invalid binding, record, array, time, dt or option. Raised before any result exists; inputs are untouched."""


class LayeredStepError(LayeredStormError):
    """The owner's batched validation recorded a failure (lowest bit first) or the composed closure failed."""

    def __init__(self, message: str, *, bit: int | None = None, n_flagged: int = 0):
        super().__init__(message)
        self.bit = bit
        self.n_flagged = int(n_flagged)


class LayeredMemoryError(MemoryError):
    """The composed trial estimate plus the reserves exceeds the reported available bytes (nothing allocated)."""


# --- memory ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LayeredMemory:
    """Composed water-owner ledger (see MEMORY_SCOPE). `required_bytes` (resident + incremental + every reserve) is
    admitted before every step under the existing column convention; `incremental_bytes` is what one trial newly
    allocates (diagnostic); the cold build and the initialisation have their own admitted figures."""

    backend: str
    n_layers: int
    n_cells: int
    n_faces: int
    n_classes: int  # native grain classes K (0 without a bed)
    accepted_bytes: int  # resident accepted aggregate (surface, faces, layers, history; + native mobile N K doubles)
    static_bytes: int  # resident owned prepared arrays (component maps/statics, owner zero forcing/masks/bound)
    resident_bytes: int  # accepted + static
    stage_allocation_bytes: dict  # per stage: declared incremental allocation of one trial
    owner_scratch_bytes: int
    native_validation_bytes: int  # genuine with_water validation masks (K-dependent, backend-dependent)
    trial_bytes: int  # = incremental_bytes: everything ONE trial newly allocates (bound)
    incremental_bytes: int
    initialization_bytes: int  # one `initial_layered_state`: surface copy + zero faces + inventory validation masks
    build_bytes: int  # cold prepare allocations on the admitted side (zero forcing, masks, saturation bound)
    build_host_bytes: int  # cold host staging at prepare (mask/surface downloads, one bounded hash chunk)
    reserve_bytes: int  # column reserve
    extra_reserve_bytes: int  # caller's reserve for its own simultaneously live bytes
    native_reserve_bytes: int  # caller's explicit reserve for native material outside the water owner (0 without a bed)
    native_reference_depth_bytes: int  # retained original native depth H0 (8 N with a bed, else 0); inside static/resident
    host_bytes: int
    device_bytes: int
    required_bytes: int  # resident + incremental + reserves on the admitted side (admitted before every step)
    scope: str = MEMORY_SCOPE

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def estimate_layered_memory(n_layers: int, shape: Any, *, backend: str, limiter: str = "off", reserve_bytes: int = 0,
                            extra_reserve_bytes: int = 0, n_species: int = 0, diffusion_mode: str = "constant",
                            native_bed: bool = False, n_classes: int = 0, native_reserve_bytes: int = 0) -> LayeredMemory:
    """Pure arithmetic from the components' own estimates (`estimate_preferential_memory`, `estimate_matrix_memory`,
    `estimate_step_memory`, `estimate_surface_memory`) for `n_layers` on a `(ny, nx)` grid; with a native bed the
    genuine class count `n_classes` (K >= 1) sizes the resident mobile array and the K-dependent native validation."""
    from maple_syrup.surface_routing import estimate_surface_memory

    L = ss._checked_int(n_layers, "n_layers")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise LayeredStormError(f"shape must be (ny, nx), got {shape!r}")
    ny, nx = (ss._checked_int(v, "shape entry") for v in shape)
    if L < 1 or ny < 1 or nx < 1:
        raise LayeredStormError("n_layers and the grid extents must be >= 1")
    if backend not in BACKENDS:
        raise LayeredStormError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if type(native_bed) is not bool:
        raise LayeredStormError("native_bed must be a bool")
    reserve = ss._checked_int(reserve_bytes, "reserve_bytes")
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    native_reserve = ss._checked_int(native_reserve_bytes, "native_reserve_bytes")
    K = ss._checked_int(n_classes, "n_classes")
    if native_bed and K < 1:
        raise LayeredStormError("a native bed needs its genuine class count n_classes >= 1 (no K-independent ledger)")
    if not native_bed and (K != 0 or native_reserve != 0):
        raise LayeredStormError("n_classes and native_reserve_bytes belong to a native bed (refused without one)")
    N = ny * nx
    pref = pi.estimate_preferential_memory(L, N, backend=backend, n_species=n_species)
    mat = mi.estimate_matrix_memory(L, N, backend=backend)
    col = ss.estimate_step_memory(L, N, backend=backend, diffusion_mode=diffusion_mode)
    surf = estimate_surface_memory(SURFACE_METHOD, (ny, nx), backend=backend, limiter=limiter)
    n_faces = surf.n_faces
    accepted = (N + n_faces + L * N + 2 * N) * _F8 + (N * K * _F8 if native_bed else 0)  # the native depth IS the surface
    static = (pref.parameter_bytes + pref.weight_bytes + mat.parameter_bytes + col.column_bytes + col.parameter_bytes
              + surf.static_bytes + (L * N + N + L) * _F8 + N)
    # the operator retains its original native depth H0 (a reference, not water) while accepted states carry fresh depths
    reference_depth = N * _F8 if native_bed else 0
    static += reference_depth
    owner = OWNER_SCRATCH_DOUBLES_PER_CELL * N * _F8 + OWNER_SCRATCH_MASK_BYTES_PER_CELL * N + _PACKET_SCRATCH_BYTES
    native_validation = native_validation_bytes(N, K, backend) if native_bed else 0
    stages = {"pool": N * _F8, "preferential": pref.step_allocation_bytes, "matrix": mat.step_allocation_bytes,
              "column": col.step_allocation_bytes, "post_pool": N * _F8, "surface": surf.step_allocation_bytes,
              "owner": owner, "native_wrapper": native_validation}
    trial = sum(stages.values())
    # initialisation: the surface copy and zero faces, the (L, N) saturation-bound mask, the N-byte sign/finite masks of
    # the inventory checks (sequential) and the router's own depth/momentum validation masks (N + faces bytes)
    initialization = (N + n_faces) * _F8 + L * N + 2 * N + n_faces + NATIVE_OBJECT_BYTES
    build = (L * N + N + L) * _F8 + N + NATIVE_OBJECT_BYTES  # zero forcing, inactive mask, saturation bound, zero check
    build_host = N + N * _F8 + (HASH_CHUNK_BYTES if native_bed else 0)  # mask and surface downloads, one hash chunk
    if backend != "cuda":
        build += build_host  # cold staging and ordinary arrays share the CPU quota
    resident = accepted + static
    if backend == "cuda":
        host = (pref.host_bytes + mat.host_bytes + col.host_bytes + surf.host_bytes + _PACKET_SCRATCH_BYTES + build_host)
        device = resident + trial
    else:
        host = resident + trial
        device = 0
    required = resident + trial + reserve + extra + native_reserve
    return LayeredMemory(backend, L, N, n_faces, K, accepted, static, resident, stages, owner, native_validation, trial,
                         trial, initialization, build, build_host, reserve, extra, native_reserve, reference_depth, host, device,
                         required)


def _admit(column: ss.SoilColumn, shape: tuple[int, int], need: int, what: str) -> int:
    """Compare `need` with the COLUMN's reported available bytes; raise `LayeredMemoryError` before any allocation."""
    available = column.available_bytes_fn()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise LayeredStormError(f"available_bytes_fn must return a non-Boolean integer >= 0, got {available!r}")
    available = int(available)
    if int(need) > available:
        raise LayeredMemoryError(
            f"layered storm {what} on {column.n_layers} layers x {shape[0]}x{shape[1]} cells ({column.backend}) needs an "
            f"estimated {int(need)} B on its admitted side (reserves included) but only {available} B are reported "
            f"available; nothing was allocated. Scope: {MEMORY_SCOPE}")
    return available


def _admit_host_build(need: int, available_bytes_fn: Any) -> None:
    """Separate host staging admission for CUDA; device free memory is not RAM."""
    available = available_bytes_fn()
    if (isinstance(available, (bool, np.bool_))
            or not isinstance(available, (int, np.integer)) or int(available) < 0):
        raise LayeredStormError("host_available_bytes_fn must return a non-Boolean integer >= 0")
    if need > int(available):
        raise LayeredMemoryError(f"layered storm cold host staging needs {need} B, but only {int(available)} B are reported")


# --- records -------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LayeredWaterLedger:
    """Cumulative host scalars of the accepted aggregate (m3 over active cells; no per-step history)."""

    n_steps: int = 0
    elapsed_s: float = 0.0
    rain_m3: float = 0.0
    preferential_intake_m3: float = 0.0
    matrix_intake_m3: float = 0.0
    surface_return_m3: float = 0.0
    drainage_m3: float = 0.0
    export_m3: float = 0.0
    evaporation_m3: float = 0.0  # always exactly 0 in a storm step; carried so the invariant is visible
    global_residual_bound_m3: float = 0.0  # running sum of the composed bounds (a bound on the accumulated residual)
    max_abs_global_residual_m3: float = 0.0

    def advanced(self, receipt: LayeredStepReceipt) -> LayeredWaterLedger:
        _check_ledger(self)
        result = LayeredWaterLedger(
            int(self.n_steps) + 1, float(self.elapsed_s) + float(receipt.dt_s), float(self.rain_m3) + float(receipt.rain_m3),
            float(self.preferential_intake_m3) + float(receipt.preferential.intake_m3),
            float(self.matrix_intake_m3) + float(receipt.matrix.intake_m3),
            float(self.surface_return_m3) + float(receipt.column.surface_return_m3),
            float(self.drainage_m3) + float(receipt.column.drainage_m3),
            float(self.export_m3) + float(receipt.export_m3), float(self.evaporation_m3) + float(receipt.column.evaporation_m3),
            float(self.global_residual_bound_m3) + float(receipt.global_bound_m3),
            max(float(self.max_abs_global_residual_m3), abs(float(receipt.global_residual_m3))))
        _check_ledger(result, int(self.n_steps) + 1)
        return result

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True, eq=False)
class LayeredWaterState:
    """One accepted aggregate instant. Arrays of the operator namespace: `surface_m (ny, nx)`, `qx_m2_s (ny, nx+1)`,
    `qy_m2_s (ny+1, nx)`; `soil` at the same `t_s`; the bound `history`; the native `bed` (its `water.depth_m` IS
    `surface_m`) or None; the cumulative `ledger`; `generation` (0 at initialisation, +1 per adopted candidate);
    `operator_sha256` (the binding). Built by `initial_layered_state` or a step; never written."""

    t_s: float
    surface_m: Any
    qx_m2_s: Any
    qy_m2_s: Any
    soil: ss.SoilColumnState
    history: mi.WettingHistory
    bed: Any
    ledger: LayeredWaterLedger
    generation: int
    operator_sha256: str


class VerticalArrays(NamedTuple):
    """The stage arrays of one candidate (referenced, not copied): per-receiver preferential intake `(L, N)`, matrix
    intake `(N,)`, saturation return `(N,)` and bottom drainage `(N,)`."""

    preferential_intake_m: Any
    matrix_intake_m: Any
    surface_return_m: Any
    drainage_m: Any


@dataclass(frozen=True)
class LayeredStepReceipt:
    """Host scalars of one candidate: the stage budgets, the surface scalars, the composed terms and the closure checks."""

    dt_s: float
    t0_s: float
    t1_s: float
    preferential: pi.PreferentialBudget
    matrix: mi.MatrixBudget
    column: ss.StepBudget
    rain_m3: float
    surface_before_m3: float
    surface_after_m3: float
    soil_before_m3: float
    soil_after_m3: float
    export_m3: float
    storage_change_m3: float
    router_residual_m3: float
    max_cfl: float
    limited_cells: int
    limited_volume_m3: float
    max_cell_balance_residual_m: float
    outlet_discharge_m3_s: float
    gross_face_volume_m3: float
    global_residual_m3: float
    global_bound_m3: float
    owner_bound_m3: float
    owner_global_terms: int
    max_surface_cell_residual_m3: float
    max_soil_cell_residual_m3: float
    surface_cell_coefficient: float
    soil_cell_coefficient: float
    surface_cell_terms: int
    soil_cell_terms: int
    handoff_exact: bool  # the stage packets agree bitwise on the handed-over pool/soil totals
    n_flagged: int

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        for key in ("preferential", "matrix", "column"):
            value = out[key]
            out[key] = value.as_dict() if hasattr(value, "as_dict") else dict(value.__dict__)
        return out


@dataclass(frozen=True, eq=False)
class LayeredStepCandidate:
    """One proposed step: the fresh accepted-shaped `state` at `t1`, the `receipt`, the router's `surface_step` and the
    stage `vertical` arrays. The caller adopts `state` whole or drops the candidate; nothing is adopted partially."""

    state: LayeredWaterState
    receipt: LayeredStepReceipt
    dt_s: float
    surface_step: SurfaceStep
    vertical: VerticalArrays


# --- operator ------------------------------------------------------------------------------------------------------------
class LayeredStormOperator:
    """The prepared binding, minted ONLY by `prepare_layered_storm`, immutable. Holds the components by reference (they
    are their own sealed records), the held topology (fixed bed), the optional native bed/context and the identities of
    its material objects, the owner's zero forcing and masks, the composed memory estimate and a cold SHA-256."""

    __slots__ = (
        "_bed_material_ids", "_component_shas", "_control_values", "_inactive", "_layers", "_router_type", "_saturation_bound",
        "_seal", "_zero_added", "_zero_pet", "backend", "bed", "column", "context", "device_id", "dx_m",
        "extra_reserve_bytes", "graph", "matrix", "memory", "n_cells", "n_classes", "n_faces", "n_layers",
        "native_material_sha256", "native_reserve_bytes", "preferential", "router", "sha256", "shape",
        "soil_cell_coefficient", "surface_cell_coefficient", "topology", "xp",
    )

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise LayeredStormError("LayeredStormOperator is constructed only by prepare_layered_storm()")
        for name in self.__slots__:
            if name == "_seal":
                continue
            if name not in fields:
                raise LayeredStormError(f"internal: operator field {name} missing")
            object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise LayeredStormError(f"internal: unexpected operator fields {sorted(fields)}")
        object.__setattr__(self, "_seal", seal)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("LayeredStormOperator is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("LayeredStormOperator is immutable")

    def __repr__(self) -> str:
        return (f"LayeredStormOperator(backend={self.backend!r}, shape={self.shape}, n_layers={self.n_layers}, "
                f"native_bed={self.bed is not None}, sha256={self.sha256[:12]}...)")

    def describe(self) -> dict[str, Any]:
        return describe(self)


def _real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise LayeredStormError(f"{name} must be a real host number, got {type(value).__name__}")
    try:
        out = float(value)
    except OverflowError:
        raise LayeredStormError(f"{name} is too large to represent as a finite float64") from None
    if not math.isfinite(out):
        raise LayeredStormError(f"{name} must be finite, got {value!r}")
    return out


def _check_dt(dt_s: Any) -> float:
    dt = _real(dt_s, "dt_s")
    if not dt > 0.0:
        raise LayeredStormError(f"dt_s must be > 0 (the storm step has no dt = 0 identity), got {dt!r}")
    return dt


def _host(array: Any) -> np.ndarray:
    from maple.core.backend import to_host

    return np.asarray(to_host(array))


def _cuda_router_class():
    from maple_syrup.experimental_cuda import CudaSurfaceRouter

    return CudaSurfaceRouter


def _check_router_binding(router: Any, backend: str, graph: RoutingGraph, topology: PondTopology, column: ss.SoilColumn) -> None:
    """The router is the surface router of the column's backend for the local-inertial method, prepared on exactly
    `graph` and `topology.geometry` (object identity, as the pond lifecycle binds its solvers)."""
    if backend == "cuda":
        cls = _cuda_router_class()
        if type(router) is not cls:
            raise LayeredStormError(f"router must be a CudaSurfaceRouter for a cuda column, got {type(router).__name__}")
        ctx = router.context
        if not ctx.is_bound_to(graph, topology.geometry):
            raise LayeredStormError("the CUDA router's context is not bound to exactly this graph and the topology's "
                                    "geometry; refused")
        if int(ctx.device_id) != column.device_id:
            raise LayeredStormError(f"the router context lives on device {ctx.device_id}, the column on {column.device_id}")
    else:
        if not isinstance(router, CpuSurfaceRouter):
            raise LayeredStormError(f"router must be a CPU surface router (CpuSurfaceRouter / NumbaSurfaceRouter), got "
                                    f"{type(router).__name__}")
    if getattr(router, "implementation", None) != backend:
        raise LayeredStormError(f"router implementation {getattr(router, 'implementation', None)!r} must equal the column "
                                f"backend {backend!r} (one declared backend per step)")
    if router.method != SURFACE_METHOD:
        raise LayeredStormError(f"the layered storm step carries face momentum and binds the {SURFACE_METHOD!r} router "
                                f"only; {router.method!r} is refused by name")
    if backend == "cuda":
        # Availability callbacks can be injected. They must not bypass the
        # genuine context's current-device check before our first allocation.
        from maple_syrup.experimental_cuda import _check_surface_context

        _check_surface_context(column.xp, router.context)
    else:
        if router.graph is not graph:
            raise LayeredStormError("the router was not built on exactly this graph object; refused")
        if router.geometry is not topology.geometry:
            raise LayeredStormError("the router was not built on exactly the topology's geometry object; refused")
    if tuple(router.shape) != tuple(topology.shape) or float(router.dx_m) != float(topology.dx_m):
        raise LayeredStormError("the router's shape/dx disagree with the topology; refused")
    if not isinstance(router.control, HydraulicControl):
        raise LayeredStormError("the router's control is not a HydraulicControl; refused")


def _control_values(control: Any) -> tuple:
    if not isinstance(control, HydraulicControl):
        raise LayeredStormError("the router's control is not a HydraulicControl; refused")
    return (float(control.cfl_max), str(control.limiter), str(control.friction_treatment))


def _check_router_bound(op: Any) -> None:
    """Cheap per-call rebinding of the router (correction01, root_draft_router_binding): type, implementation, method,
    extents, the sealed CONTROL VALUES (a replaced `router.control` with another CFL/limiter/friction mode is refused,
    it would change the physics under the old operator identity), graph/geometry object identity (CPU) or the CUDA
    context binding and device. No array content is hashed; the availability reading stays injectable."""
    router = op.router
    if type(router) is not op._router_type:
        raise LayeredStormError("the operator's router object is not of the type sealed at prepare; refused")
    if getattr(router, "implementation", None) != op.backend or router.method != SURFACE_METHOD:
        raise LayeredStormError("the operator's router changed implementation or method; refused")
    if _control_values(router.control) != op._control_values:
        raise LayeredStormError("the router's control (cfl_max, limiter, friction_treatment) differs from the values sealed in "
                                "the operator identity; a changed hydraulic control needs a new operator; refused")
    if tuple(router.shape) != tuple(op.shape) or float(router.dx_m) != op.dx_m:
        raise LayeredStormError("the router's shape/dx no longer match the operator; refused")
    graph, topology = op.graph, op.topology
    if not isinstance(graph, RoutingGraph) or not isinstance(topology, PondTopology):
        raise LayeredStormError("the operator's graph/topology records were replaced; refused")
    if tuple(graph.shape) != tuple(op.shape) or tuple(topology.shape) != tuple(op.shape) or graph.xp is not op.xp:
        raise LayeredStormError("the operator's graph/topology no longer match its extents or namespace; refused")
    if op.backend == "cuda":
        ctx = router.context
        from maple_syrup.experimental_cuda import _check_surface_context

        _check_surface_context(op.xp, ctx)
        if not ctx.is_bound_to(graph, topology.geometry) or int(ctx.device_id) != op.device_id:
            raise LayeredStormError("the CUDA router's context is no longer bound to the operator's graph, geometry and "
                                    "device; refused")
    elif router.graph is not graph or router.geometry is not topology.geometry:
        raise LayeredStormError("the router's graph/geometry are not the operator's objects; refused")


def _check_operator(op: Any) -> None:
    """Metadata-only revalidation (every call, before any allocation): genuine sealed record, component records still
    consistent and bound to the same column, router/control/graph/geometry/source identities as sealed."""
    if type(op) is not LayeredStormOperator:
        raise LayeredStormError(f"operator must be a genuine LayeredStormOperator, got {type(op).__name__}")
    if getattr(op, "_seal", None) is not _SEAL:
        raise LayeredStormError("operator record is not sealed by prepare_layered_storm(); refused")
    column = op.column
    ss._check_column(column)
    pi._check_operator(op.preferential)
    mi._check_operator(op.matrix)
    if op.preferential.column is not column or op.matrix.column is not column:
        raise LayeredStormError("the preferential/matrix operators are no longer bound to the operator's column; refused")
    if ((op.preferential.sha256, op.matrix.sha256) != op._component_shas[:2]
            or column.layers is not op._layers):
        raise LayeredStormError("a component identity (preferential, matrix or layers) differs from the one sealed in the "
                                "operator; refused")
    if op.backend != column.backend or op.xp is not column.xp or op.n_layers != column.n_layers or op.n_cells != column.n_cells:
        raise LayeredStormError("operator record disagrees with its column (backend, namespace, layers or cells); refused")
    if op.device_id != column.device_id:
        raise LayeredStormError("operator record disagrees with its column's device; refused")
    L, N = op.n_layers, op.n_cells
    ss._check_dynamic(column, "operator zero added water", op._zero_added, (L, N))
    ss._check_dynamic(column, "operator zero evaporation", op._zero_pet, (N,))
    ss._check_dynamic(column, "operator saturation bound", op._saturation_bound, (L,))
    inactive = op._inactive
    if type(inactive) is not op.xp.ndarray or inactive.dtype != np.bool_ or tuple(inactive.shape) != (N,):
        raise LayeredStormError("operator inactive mask lost its exact type/dtype/shape; refused")
    _check_router_bound(op)
    if (op.bed is None) != (op.context is None):
        raise LayeredStormError("operator record carries a bed without a context or a context without a bed; refused")
    if op.bed is not None:
        if type(op.bed) is not BedState or type(op.context) is not BedContext:
            raise LayeredStormError("the operator's native bed/context records were replaced; refused")
        if _bed_material_ids(op.bed) != op._bed_material_ids:
            raise LayeredStormError("the operator's bound native material objects were replaced; refused")
        g = op.context.geometry
        if (int(g.ny), int(g.nx)) != tuple(op.shape) or float(g.dx_m) != op.dx_m or float(g.dy_m) != op.dx_m:
            raise LayeredStormError("the operator's native geometry no longer matches its extents; refused")
    if not isinstance(op.sha256, str) or len(op.sha256) != 64:
        raise LayeredStormError("operator record carries a replaced identity; refused")


def _bed_material_ids(bed: BedState) -> tuple[int, ...]:
    return (id(bed.voxel_column), id(bed.active_layer), id(bed.ledger), id(bed.committed_topography),
            id(bed.sediment_availability), id(bed.water.mobile_mass_by_cell_class_kg))


def _check_ledger(ledger: Any, generation: int | None = None) -> None:
    """Validate host accounting before stages and after accumulation; no device work.

    Elapsed time is relative to the initial aggregate, which may have a nonzero
    physical epoch. Event/restart owners must bind that origin and the inventory
    anchors; finite scalar checks cannot establish a cumulative water balance.
    """
    if type(ledger) is not LayeredWaterLedger:
        raise LayeredStormError("state.ledger must be a genuine LayeredWaterLedger")
    if (isinstance(ledger.n_steps, (bool, np.bool_))
            or not isinstance(ledger.n_steps, (int, np.integer)) or int(ledger.n_steps) < 0):
        raise LayeredStormError("ledger.n_steps must be a non-Boolean integer >= 0")
    if generation is not None and int(ledger.n_steps) != int(generation):
        raise LayeredStormError("ledger.n_steps must equal state.generation")
    for name in ("elapsed_s", "rain_m3", "preferential_intake_m3", "matrix_intake_m3",
                 "surface_return_m3", "drainage_m3", "export_m3", "evaporation_m3",
                 "global_residual_bound_m3", "max_abs_global_residual_m3"):
        try:
            value = _real(getattr(ledger, name), "ledger." + name)
        except (OverflowError, TypeError) as exc:
            raise LayeredStormError(f"ledger.{name} must be a finite real host number") from exc
        if value < 0.0:
            raise LayeredStormError(f"ledger.{name} must be >= 0")
    if ledger.evaporation_m3 != 0.0:
        raise LayeredStormError("ledger.evaporation_m3 must be exactly zero during a storm")
    if ledger.max_abs_global_residual_m3 > ledger.global_residual_bound_m3:
        raise LayeredStormError("ledger.max_abs_global_residual_m3 exceeds its accumulated bound")


def _check_state(op: LayeredStormOperator, state: Any) -> float:
    """Array metadata, bindings, alias contract and times of an accepted aggregate; returns `t0`."""
    if not isinstance(state, LayeredWaterState):
        raise LayeredStormError(f"state must be a LayeredWaterState, got {type(state).__name__}")
    if state.operator_sha256 != op.sha256:
        raise LayeredStormError("the accepted state is bound to a different layered storm operator; refused")
    if isinstance(state.generation, (bool, np.bool_)) or not isinstance(state.generation, (int, np.integer)) or int(state.generation) < 0:
        raise LayeredStormError(f"state.generation must be a non-Boolean integer >= 0, got {state.generation!r}")
    _check_ledger(state.ledger, int(state.generation))
    t0 = _real(state.t_s, "state.t_s")
    if t0 < 0.0:
        raise LayeredStormError(f"state.t_s must be >= 0, got {t0!r}")
    column = op.column
    ny, nx = op.shape
    ss._check_dynamic(column, "state.surface_m", state.surface_m, (ny, nx))
    ss._check_dynamic(column, "state.qx_m2_s", state.qx_m2_s, (ny, nx + 1))
    ss._check_dynamic(column, "state.qy_m2_s", state.qy_m2_s, (ny + 1, nx))
    t_soil = ss._check_state(column, state.soil)
    if t_soil != t0:
        raise LayeredStormError(f"state.soil.t_s = {t_soil!r} disagrees with state.t_s = {t0!r}; one clock, never restamped")
    mi._check_history(op.matrix, state.history)
    if op.bed is None:
        if state.bed is not None:
            raise LayeredStormError("the operator holds no native bed but the state carries one; refused")
    else:
        bed = state.bed
        if type(bed) is not BedState:
            raise LayeredStormError(f"state.bed must be a genuine BedState, got {type(bed).__name__}")
        if bed.water.depth_m is not state.surface_m:
            raise LayeredStormError("alias contract violated: state.bed.water.depth_m must BE state.surface_m (one native "
                                    "surface water array, never two authoritative surfaces)")
        if _bed_material_ids(bed) != op._bed_material_ids:
            raise LayeredStormError("the state's native bed material (voxel column, active layer, ledger, committed "
                                    "topography, availability, mobile array) is not the operator-bound material; refused")
    return t0


# --- preparation -------------------------------------------------------------------------------------------------------
def _stream_hash(h: Any, array: Any) -> None:
    """Feed `array` to the digest in bounded row chunks (at most `HASH_CHUNK_BYTES` of host staging): a C-contiguous
    host array is hashed through its buffer (no copy); a non-contiguous or device array is staged one chunk at a time.
    Never a whole `tobytes()` copy of an (N, K) array."""
    h.update(f"{array.dtype}{tuple(array.shape)}".encode())
    if type(array) is np.ndarray and array.size == 0:
        return  # the dtype/shape header already identifies it; a zero-length view cannot be cast
    if type(array) is np.ndarray and array.flags.c_contiguous:
        h.update(memoryview(array).cast("B"))
        return
    if type(array) is np.ndarray:
        # reshape/ravel would allocate the entire strided leaf. nditer's
        # contiguous external buffers preserve logical C order with bounded
        # staging, including Fortran-order and sliced genuine native leaves.
        iterator = np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                             op_flags=["readonly", "contig"], order="C",
                             buffersize=max(1, HASH_CHUNK_BYTES // int(array.dtype.itemsize)))
        with iterator:
            for block in iterator:
                h.update(memoryview(block).cast("B"))
        return
    flat = array.reshape(-1) if array.flags.c_contiguous else array
    step = max(1, HASH_CHUNK_BYTES // int(flat.dtype.itemsize))
    if flat.ndim != 1:
        # A single strided device row can itself exceed the staging budget.
        # Slice its final axis as well; do not download a whole row or reshape
        # a non-contiguous device array into an implicit full-sized copy.
        for prefix in np.ndindex(flat.shape[:-1]):
            for start in range(0, int(flat.shape[-1]), step):
                block = np.ascontiguousarray(_host(flat[prefix + (slice(start, start + step),)]))
                h.update(memoryview(block).cast("B"))
                del block  # release before downloading the next chunk
        return
    for start in range(0, int(flat.shape[0]), step):
        block = np.ascontiguousarray(_host(flat[start:start + step]))
        h.update(memoryview(block).cast("B"))
        del block


def _native_digest(bed: BedState, surface_host: np.ndarray) -> str:
    """Cold SHA-256 of the native material the step must never touch: the active layer, the availability leaves and the
    current routing surface, streamed in bounded chunks. The voxel column is bound by identity (never scanned per step)."""
    from maple.core.backend import array_leaves

    h = hashlib.sha256(b"maple_syrup.layered_storm.native_material.v2")
    for name, obj in (("active_layer", bed.active_layer), ("sediment_availability", bed.sediment_availability)):
        h.update(name.encode())
        for leaf in array_leaves(obj):
            _stream_hash(h, leaf)
    h.update(b"routing_surface")
    _stream_hash(h, np.ascontiguousarray(surface_host))
    return h.hexdigest()


def prepare_layered_storm(column: ss.SoilColumn, preferential: pi.PreferentialOperator, matrix: mi.MatrixOperator,
                          router: Any, graph: RoutingGraph, topology: PondTopology, *, bed: Any = None,
                          context: Any = None, extra_reserve_bytes: int = 0,
                          native_reserve_bytes: int | None = None,
                          host_available_bytes_fn: Any = None, host_reserve_bytes: int = 0) -> LayeredStormOperator:
    """Bind the existing prepared components into one storm operator (cold: one streamed hash, one native surface read).
    `column` is the prepared layered column on the flat row-major cells of `topology.shape` with area `dx*dx`;
    `preferential`/`matrix` are prepared on exactly `column`; `router` is the local-inertial surface router of the
    column's backend prepared on exactly `graph` (the metadata routing graph in the column namespace) and
    `topology.geometry`; `topology` holds the fixed bed. `bed`/`context` (both or neither) are the genuine native
    MAPLE bed whose water IS the surface: the bed must carry no mobile sediment and its current routing surface must
    equal the held topology source on every active cell. With a bed, `native_reserve_bytes` is REQUIRED (a non-Boolean
    integer >= 0, 0 allowed only as an explicit declaration): the caller's reserve for native material outside the
    water owner (see MEMORY_SCOPE); without a bed it must be None. Admission (resident + one trial + reserves + build)
    precedes every allocation, download and hash."""
    ss._check_column(column)
    pi._check_operator(preferential)
    mi._check_operator(matrix)
    if preferential.column is not column:
        raise LayeredStormError("the preferential operator must be prepared on exactly this column object")
    if matrix.column is not column:
        raise LayeredStormError("the matrix operator must be prepared on exactly this column object")
    if not isinstance(topology, PondTopology):
        raise LayeredStormError(f"topology must be a PondTopology, got {type(topology).__name__}")
    if not isinstance(graph, RoutingGraph):
        raise LayeredStormError(f"graph must be a RoutingGraph, got {type(graph).__name__}")
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    host_reserve = ss._checked_int(host_reserve_bytes, "host_reserve_bytes")
    if host_available_bytes_fn is not None and not callable(host_available_bytes_fn):
        raise LayeredStormError("host_available_bytes_fn must be callable or None")
    backend, xp = column.backend, column.xp
    if backend != "cuda" and (host_available_bytes_fn is not None or host_reserve != 0):
        raise LayeredStormError("separate host staging controls are CUDA-only; CPU uses the column quota")
    ny, nx = (int(v) for v in topology.shape)
    N, L = column.n_cells, column.n_layers
    if N != ny * nx:
        raise LayeredStormError(f"the column binds {N} flat cells but the topology has {ny}x{nx} = {ny * nx}")
    if tuple(graph.shape) != (ny, nx) or float(graph.dx_m) != float(topology.dx_m):
        raise LayeredStormError("the graph's shape/dx disagree with the topology; refused")
    if graph.xp is not xp:
        raise LayeredStormError("the graph must live in the column namespace (no transfer)")
    dx = float(topology.dx_m)
    if column.cell_area_m2 != dx * dx:
        raise LayeredStormError(f"the column cell area {column.cell_area_m2!r} must equal dx*dx = {dx * dx!r} of the topology")
    _check_router_binding(router, backend, graph, topology, column)
    if (bed is None) != (context is None):
        raise LayeredStormError("bed and context must be given together (both) or not at all")
    K = 0
    if bed is not None:
        if type(bed) is not BedState or type(context) is not BedContext:
            raise LayeredStormError("bed must be a genuine BedState and context a genuine BedContext")
        if native_reserve_bytes is None:
            raise LayeredStormError("a native bed needs an explicit native_reserve_bytes (the caller's reserve for the "
                                    "native material outside the water owner; 0 must be declared, never assumed)")
        native_reserve = ss._checked_int(native_reserve_bytes, "native_reserve_bytes")
        g = context.geometry
        if (int(g.ny), int(g.nx)) != (ny, nx) or float(g.dx_m) != dx or float(g.dy_m) != dx:
            raise LayeredStormError("the native geometry (ny, nx, dx, dy) disagrees with the topology; refused")
        ss._check_dynamic(column, "bed.water.depth_m", bed.water.depth_m, (ny, nx))
        mobile = bed.water.mobile_mass_by_cell_class_kg
        K = len(context.grain_classes.classes)
        if (type(mobile) is not xp.ndarray or mobile.dtype != np.float64 or mobile.ndim != 3
                or tuple(mobile.shape) != (ny, nx, K)):
            raise LayeredStormError("bed.water.mobile_mass_by_cell_class_kg must be an exact (ny, nx, K) float64 array of "
                                    "the column namespace with the context's K classes")
    else:
        if native_reserve_bytes is not None:
            raise LayeredStormError("native_reserve_bytes is meaningless without a native bed (refused, not ignored)")
        native_reserve = 0
    control_values = _control_values(router.control)
    memory = estimate_layered_memory(L, (ny, nx), backend=backend, limiter=control_values[1],
                                     reserve_bytes=column.reserve_bytes, extra_reserve_bytes=extra,
                                     n_species=preferential.n_species, diffusion_mode=column.layers.diffusion_mode,
                                     native_bed=bed is not None, n_classes=K, native_reserve_bytes=native_reserve)
    _admit(column, (ny, nx), memory.required_bytes + memory.build_bytes, "build")  # BEFORE any download/mask/forcing/hash
    if backend == "cuda":
        _admit_host_build(memory.host_bytes + host_reserve,
                          host_available_bytes_fn or ss.default_host_available_bytes)
    from maple.core.backend import to_device, to_int

    # cold binding checks (one host download of the column mask on cuda; topology/graph masks are host metadata)
    active_host = _host(column.active).reshape(ny, nx)
    if not np.array_equal(active_host, np.asarray(topology.active)):
        raise LayeredStormError("the column's active mask disagrees with the topology's active mask (flat row-major order)")
    if not np.array_equal(np.asarray(graph.active), np.asarray(topology.active)):
        raise LayeredStormError("the graph's active mask disagrees with the topology's; refused")
    if not np.array_equal(np.asarray(graph.friction_factor), np.asarray(topology.friction)):
        raise LayeredStormError("the graph's friction disagrees with the topology's; refused")
    native_sha, material_ids = None, ()
    if bed is not None:
        # the zero check is a counted reduction without an (N, K) mask (measured: 24 B on the host)
        if to_int(xp.count_nonzero(bed.water.mobile_mass_by_cell_class_kg)) != 0:
            raise LayeredStormError("the native bed carries mobile sediment; this water-only step accepts a tested "
                                    "no-mobile-sediment state only")
        surface_host = _host(current_routing_surface_m(bed, context))  # the one allowed native column visit (cold)
        if surface_host.shape != (ny, nx) or not np.all(np.isfinite(surface_host)):
            raise LayeredStormError("the native routing surface is not a finite (ny, nx) field; refused")
        act = np.asarray(topology.active)
        if not np.array_equal(surface_host[act], np.asarray(topology.source_bed_m)[act]):
            raise LayeredStormError("the native bed's current routing surface differs from the held topology source on an "
                                    "active cell: the fixed-bed binding is refused (no reroute/refresh is performed here)")
        native_sha = _native_digest(bed, surface_host)
        material_ids = _bed_material_ids(bed)
    zero_added = xp.zeros((L, N), dtype=np.float64)
    zero_pet = xp.zeros(N, dtype=np.float64)
    if xp is np:
        zero_added.flags.writeable = False
        zero_pet.flags.writeable = False
    inactive = xp.ascontiguousarray(~column.active)
    lay = column.layers
    bound_host = np.ascontiguousarray(np.asarray(lay.saturation_m, dtype=np.float64) + np.asarray(lay.capacity_tolerance_m))
    saturation_bound = to_device(bound_host, xp) if xp is not np else bound_host  # (L,) the native capacity bound
    if xp is np:
        saturation_bound.flags.writeable = False
    coeff_s = volume_roundoff_bound_m3(surface_cell_terms(L), 1.0)
    coeff_w = volume_roundoff_bound_m3(soil_cell_terms(L), 1.0)
    layer_sha = lay.identity()["sha256"]
    h = hashlib.sha256(b"maple_syrup.layered_storm.operator.v2")
    for part in (backend, str((ny, nx)), repr(dx), layer_sha, preferential.sha256, matrix.sha256,
                 topology.topology_sha256, str(graph.input_sha256), repr(control_values), str(K),
                 native_sha or "no-native-bed"):
        h.update(part.encode())
        h.update(b"\0")
    return LayeredStormOperator(
        _SEAL, column=column, preferential=preferential, matrix=matrix, router=router, graph=graph, topology=topology,
        bed=bed, context=context, backend=backend, xp=xp, device_id=column.device_id, shape=(ny, nx), dx_m=dx, n_cells=N,
        n_layers=L, n_faces=memory.n_faces, n_classes=K, memory=memory, extra_reserve_bytes=extra,
        native_reserve_bytes=native_reserve, _zero_added=zero_added, _zero_pet=zero_pet, _inactive=inactive,
        _saturation_bound=saturation_bound, _bed_material_ids=material_ids, _control_values=control_values,
        _router_type=type(router), _component_shas=(preferential.sha256, matrix.sha256, layer_sha), _layers=column.layers,
        native_material_sha256=native_sha, surface_cell_coefficient=coeff_s, soil_cell_coefficient=coeff_w,
        sha256=h.hexdigest())


def admit_step(op: LayeredStormOperator) -> int:
    """Compare the composed trial requirement (reserves included) with the column's reported available bytes and
    raise `LayeredMemoryError` BEFORE anything is allocated. Returns the reading."""
    _check_operator(op)
    return _admit(op.column, op.shape, op.memory.required_bytes, "step")


def _validate_inventories(op: LayeredStormOperator, soil: ss.SoilColumnState, history: mi.WettingHistory) -> None:
    """Physical bounds of a newly minted ACCEPTED aggregate on EVERY cell (inactive included), one batched read:
    soil finite, >= 0, <= saturation_m + capacity_tolerance_m per layer (the native `subsurface` bounds); wetting and
    lifetime history finite and >= 0. Raises `LayeredStormError`; nothing is clipped, reset or published."""
    from maple.core.backend import DeferredChecks, finite_flag, negative_flag, true_flag

    checks = DeferredChecks()
    water = soil.water_m
    checks.require(finite_flag(water), "initial soil water must be finite on every cell (inactive cells included)")
    checks.forbid(negative_flag(water), "initial soil water must be >= 0 on every cell (inactive cells included)")
    checks.forbid(true_flag(water > op._saturation_bound[:, None]),
                  "initial soil water exceeds the layer saturation depth (+ MAPLE capacity tolerance) on a cell")
    for name, arr in (("wetting_m", history.wetting_m), ("cumulative_intake_m", history.cumulative_intake_m)):
        checks.require(finite_flag(arr), f"initial history {name} must be finite on every cell")
        checks.forbid(negative_flag(arr), f"initial history {name} must be >= 0 on every cell")
    try:
        checks.resolve()
    except ValueError as exc:
        raise LayeredStormError(f"initial inventories refused: {exc}") from None


def initial_layered_state(op: LayeredStormOperator, *, soil: ss.SoilColumnState, history: mi.WettingHistory,
                          t_s: float = 0.0, surface_m: Any = None, qx_m2_s: Any = None,
                          qy_m2_s: Any = None) -> LayeredWaterState:
    """The accepted aggregate at a genuine run/test boundary. `soil.t_s` must equal `t_s` (never restamped). With a
    native bed the surface IS `bed.water.depth_m` (`surface_m` must be None); otherwise `surface_m` is an exact (ny, nx)
    array of the namespace, copied once (admitted). Momentum None = at rest; given arrays are copied. PHYSICAL
    inventories are validated on EVERY cell (inactive included) before anything is published (correction01,
    root_draft_initial_bounds): soil finite, >= 0 and within the native layer saturation depth plus MAPLE's capacity
    tolerance (`subsurface` bounds, no clipping); wetting and lifetime history finite and >= 0; the surface and
    momentum by the router (finite, >= 0; momentum 0 on closed faces, never inward on open faces). `ss.column_state`
    and `mi.wetting_history` remain metadata-only constructors; no physical dt = 0 step is used as a validator."""
    _check_operator(op)
    t = _real(t_s, "t_s")
    if t < 0.0:
        raise LayeredStormError(f"t_s must be >= 0, got {t!r}")
    column, xp = op.column, op.xp
    ny, nx = op.shape
    t_soil = ss._check_state(column, soil)
    if t_soil != t:
        raise LayeredStormError(f"soil.t_s = {t_soil!r} must equal t_s = {t!r}; the initial state is never restamped")
    mi._check_history(op.matrix, history)
    if op.bed is not None:
        if surface_m is not None:
            raise LayeredStormError("the operator holds a native bed: its water IS the surface; surface_m must be None")
        surface = op.bed.water.depth_m
        bed = op.bed
    else:
        if surface_m is None:
            raise LayeredStormError("surface_m is required when no native bed is held")
        ss._check_dynamic(column, "surface_m", surface_m, (ny, nx))
        bed = None
    if qx_m2_s is not None:
        ss._check_dynamic(column, "qx_m2_s", qx_m2_s, (ny, nx + 1))
    if qy_m2_s is not None:
        ss._check_dynamic(column, "qy_m2_s", qy_m2_s, (ny + 1, nx))
    _admit(op.column, op.shape, op.memory.required_bytes + op.memory.initialization_bytes, "initialization")
    _validate_inventories(op, soil, history)  # physical bounds on every cell, BEFORE any copy or publication
    if bed is None:
        surface = xp.ascontiguousarray(xp.array(surface_m, dtype=np.float64, copy=True))
    qx = xp.zeros((ny, nx + 1), dtype=np.float64) if qx_m2_s is None else xp.ascontiguousarray(xp.array(qx_m2_s, copy=True))
    qy = xp.zeros((ny + 1, nx), dtype=np.float64) if qy_m2_s is None else xp.ascontiguousarray(xp.array(qy_m2_s, copy=True))
    try:
        op.router.validate_state(SurfaceState(t, surface, qx, qy))
    except ExperimentalHydrologyError as exc:
        raise LayeredStormError(f"initial surface refused by the router: {exc}") from exc
    return LayeredWaterState(t, surface, qx, qy, soil, history, bed, LayeredWaterLedger(), 0, op.sha256)


# --- the step ------------------------------------------------------------------------------------------------------------
def _packet(xp: ModuleType, values: list) -> np.ndarray:
    from maple.core.backend import to_host

    stacked = xp.stack([xp.asarray(v, dtype=np.float64) for v in values])
    return np.asarray(to_host(stacked), dtype=np.float64)


def layered_step(op: LayeredStormOperator, accepted: LayeredWaterState, rain_rate_m_s: Any, dt_s: Any) -> LayeredStepCandidate:
    """ONE pure trial from `accepted` over `dt_s > 0` with `rain_rate_m_s` (ny, nx) of the namespace (finite, >= 0, 0 on
    inactive cells). Returns a candidate or raises before any result exists (`HydraulicStepRejected` = retry the SAME
    accepted state at a smaller dt; everything else is a refusal). The accepted aggregate is never written."""
    _check_operator(op)
    t0 = _check_state(op, accepted)
    dt = _check_dt(dt_s)
    try:
        t1 = advance_time(t0, dt)
    except ExperimentalHydrologyError as exc:
        raise LayeredStormError(str(exc)) from None
    column, xp = op.column, op.xp
    ny, nx = op.shape
    N, L = op.n_cells, op.n_layers
    ss._check_dynamic(column, "rain_rate_m_s", rain_rate_m_s, (ny, nx))
    _admit(op.column, op.shape, op.memory.required_bytes, "step")  # the composed peak, BEFORE the pool or any mask is allocated
    area = column.cell_area_m2
    active, inactive = column.active, op._inactive

    # 2. rain once (flat views of the C-contiguous (ny, nx) arrays: no copy, no transfer)
    h0 = accepted.surface_m.reshape(N)
    rain = rain_rate_m_s.reshape(N)
    pool = rain * dt
    xp.add(pool, h0, out=pool)
    # 3. preferential on the pool, clock preserved
    p = pi.preferential_step(op.preferential, pool, accepted.soil, dt)
    # 4. matrix on the remaining pool; ORIGINAL H0 is the contact descriptor, the actual rain the onset descriptor
    m = mi.matrix_step(op.matrix, p.surface_water_m, h0, rain, p.state, accepted.history, dt)
    # 5. column with zero forcing on the credited soil: drainage/diffusion/return act, soil clock t0 -> t1
    c = ss.step(column, m.state, dt, added_water_m=op._zero_added, potential_evaporation_m_s=op._zero_pet)
    # 6. the saturation return rejoins the pool once; lateral stage from the ACCEPTED time t0
    post = m.surface_pool_m + c.surface_return_m
    s = op.router.step(SurfaceState(t0, post.reshape(ny, nx), accepted.qx_m2_s, accepted.qy_m2_s), dt)
    if s.state.t_s != t1 or c.state.t_s != t1:
        raise LayeredStormError(f"clock mismatch: surface end {s.state.t_s!r}, soil end {c.state.t_s!r}, required {t1!r}; "
                                "no state is restamped")
    del post
    # 7. genuine native water wrapper (water only; the accepted all-zero mobile array is reused by with_water)
    bed1 = None if op.bed is None else with_water(accepted.bed, op.context, s.state.depth_m)

    # owner packet: per-cell closures, inactive preservation, zero E, finite bounds (vectorised, one host read)
    from maple.core.backend import errstate

    h1 = s.state.depth_m.reshape(N)
    pref_tot, mat_in = p.intake_total_m, m.intake_m
    ret, drain, evap, unmet = c.surface_return_m, c.drainage_m, c.evaporation_m, c.evaporation_unmet_m
    vx, vy = s.face_volume_m3["x"], s.face_volume_m3["y"]
    soil0, soil1 = accepted.soil.water_m, c.state.water_m
    flags = xp.zeros(N, dtype=np.int64)

    def flag(mask: Any, bit: int) -> None:  # namespace-generic (no ufunc `where=`): OR the shifted mask in
        xp.bitwise_or(flags, xp.left_shift(mask.astype(np.int64), bit), out=flags)

    with errstate(xp=xp, all="ignore"):
        rd = rain * dt
        flag(~xp.isfinite(rain), 0)
        flag(rain < 0.0, 1)
        flag(inactive & (rain != 0.0), 2)
        flag(inactive & (h1 != h0), 3)
        w0 = xp.zeros(N, dtype=np.float64)
        w1 = xp.zeros(N, dtype=np.float64)
        for layer in range(L):
            flag(inactive & (soil1[layer] != soil0[layer]), 4)
            xp.add(w0, soil0[layer], out=w0)
            xp.add(w1, soil1[layer], out=w1)
        flag(inactive & ((m.history.wetting_m != accepted.history.wetting_m)
                         | (m.history.cumulative_intake_m != accepted.history.cumulative_intake_m)), 5)
        lat = ((vx[:, :-1] - vx[:, 1:]) + (vy[:-1, :] - vy[1:, :])).reshape(N)
        gross = ((xp.abs(vx[:, :-1]) + xp.abs(vx[:, 1:])) + (xp.abs(vy[:-1, :]) + xp.abs(vy[1:, :]))).reshape(N)
        vert = rd - pref_tot
        xp.subtract(vert, mat_in, out=vert)
        xp.add(vert, ret, out=vert)  # rain dt - pref - matrix + return
        res_s = h1 - h0
        xp.multiply(res_s, area, out=res_s)
        xp.multiply(vert, area, out=vert)
        xp.subtract(res_s, vert, out=res_s)
        xp.subtract(res_s, lat, out=res_s)
        xp.abs(res_s, out=res_s)
        scale_s = h0 + h1
        xp.add(scale_s, rd, out=scale_s)
        xp.add(scale_s, pref_tot, out=scale_s)
        xp.add(scale_s, mat_in, out=scale_s)
        xp.add(scale_s, ret, out=scale_s)
        xp.multiply(scale_s, area, out=scale_s)
        xp.add(scale_s, gross, out=scale_s)
        xp.multiply(scale_s, op.surface_cell_coefficient, out=scale_s)
        flag(active & (res_s > scale_s), 6)
        flux_w = pref_tot + mat_in
        xp.subtract(flux_w, ret, out=flux_w)
        xp.subtract(flux_w, drain, out=flux_w)
        res_w = w1 - w0
        xp.subtract(res_w, flux_w, out=res_w)
        xp.multiply(res_w, area, out=res_w)
        xp.abs(res_w, out=res_w)
        scale_w = w0 + w1
        xp.add(scale_w, pref_tot, out=scale_w)
        xp.add(scale_w, mat_in, out=scale_w)
        xp.add(scale_w, ret, out=scale_w)
        xp.add(scale_w, drain, out=scale_w)
        xp.multiply(scale_w, area * op.soil_cell_coefficient, out=scale_w)
        flag(active & (res_w > scale_w), 7)
        flag(active & (~xp.isfinite(res_s) | ~xp.isfinite(res_w) | ~xp.isfinite(scale_s) | ~xp.isfinite(scale_w)), 8)
        flag(active & ((evap != 0.0) | (unmet != 0.0)), 9)
        del vert, flux_w, lat, w0, w1, scale_s, scale_w
        low = xp.negative(flags)
        xp.bitwise_and(flags, low, out=low)
        low_row = xp.where(flags == 0, NO_FLAG, low.astype(np.float64))
        del low
        values = [xp.sum(xp.where(active, rd, 0.0)), xp.sum(xp.where(active, h0, 0.0)), xp.sum(xp.where(active, h1, 0.0)),
                  xp.max(xp.where(active, res_s, 0.0)), xp.max(xp.where(active, res_w, 0.0)), xp.max(xp.abs(evap)),
                  xp.min(low_row), xp.count_nonzero(flags != 0), s.export_m3, s.storage_change_m3, s.budget_residual_m3,
                  s.max_cfl, s.limited_cells, s.limited_volume_m3, s.max_cell_balance_residual_m, s.outlet_discharge_m3_s,
                  xp.sum(xp.abs(vx)) + xp.sum(xp.abs(vy))]
        packet = _packet(xp, values)  # the ONE owner host read (synchronising on cuda)
        del values, low_row, res_s, res_w, rd, gross, flags
    if packet.shape != (OWNER_PACKET_LENGTH,):
        raise LayeredStormError(f"internal: owner packet shape {packet.shape}")
    low_bit, n_flagged = float(packet[6]), int(packet[7])
    if not math.isfinite(low_bit) or low_bit < NO_FLAG:
        bit = int(low_bit).bit_length() - 1 if math.isfinite(low_bit) and low_bit >= 1.0 else None
        message = FLAG_MESSAGES[bit] if bit is not None and 0 <= bit < N_FLAGS else "owner packet is corrupt"
        raise LayeredStepError(f"{message} ({n_flagged} cell(s) flagged)", bit=bit, n_flagged=n_flagged)
    rain_m3, h0_m3, h1_m3 = (float(packet[i]) * area for i in range(3))
    max_res_s, max_res_w, max_evap = (float(packet[i]) for i in range(3, 6))
    export, storage, router_residual, max_cfl = (float(packet[i]) for i in range(8, 12))
    limited_cells, limited_volume, max_cell_balance, outlet_q, gross_total = (int(packet[12]), float(packet[13]),
                                                                              float(packet[14]), float(packet[15]),
                                                                              float(packet[16]))
    if c.budget.evaporation_m3 != 0.0 or c.budget.evaporation_unmet_m3 != 0.0 or max_evap != 0.0:
        raise LayeredStepError("the column stage reported non-zero evaporation under zero forcing; refused", bit=9,
                               n_flagged=n_flagged)
    soil0_m3, soil1_m3 = p.budget.soil_before_m3, c.budget.storage_after_m3
    drain_m3 = c.budget.drainage_m3
    # stage handoff: the matrix stage's pool/soil totals must be the preferential stage's, the column's the matrix's
    handoff_exact = (m.budget.pool_before_m3 == p.budget.surface_after_m3 and m.budget.soil_before_m3 == p.budget.soil_after_m3
                     and c.budget.storage_before_m3 == m.budget.soil_after_m3)
    if (abs(m.budget.pool_before_m3 - p.budget.surface_after_m3) > p.budget.bound_m3 + m.budget.bound_m3
            or abs(m.budget.soil_before_m3 - p.budget.soil_after_m3) > p.budget.bound_m3 + m.budget.bound_m3
            or abs(c.budget.storage_before_m3 - m.budget.soil_after_m3) > m.budget.bound_m3 + c.budget.bound_m3):
        raise LayeredStepError("stage handoff mismatch: a stage received other water than the previous stage returned",
                               bit=None, n_flagged=n_flagged)
    residual = (((h1_m3 - h0_m3) + (soil1_m3 - soil0_m3)) + export) + drain_m3 - rain_m3
    operands = (abs(h0_m3), abs(h1_m3), abs(soil0_m3), abs(soil1_m3), abs(export), abs(drain_m3), abs(rain_m3))
    if not all(math.isfinite(v) for v in (*operands, storage, router_residual, max_res_s, max_res_w, residual)):
        raise LayeredStepError("non-finite composed water budget", bit=None, n_flagged=n_flagged)
    owner_terms = OWNER_GLOBAL_TERMS_PER_CELL * N + OWNER_GLOBAL_TERMS
    owner_bound = volume_roundoff_bound_m3(owner_terms, max(operands))
    bound = (abs(router_residual) + 2.0 * p.budget.bound_m3 + 2.0 * m.budget.bound_m3 + c.budget.bound_m3) + owner_bound
    if abs(residual) > bound:
        raise LayeredStepError(
            f"composed global water closure violated: residual {residual!r} m3 against the derived bound {bound!r} m3 "
            f"(router residual {router_residual!r}, stage bounds {p.budget.bound_m3!r}/{m.budget.bound_m3!r}/"
            f"{c.budget.bound_m3!r}, owner {owner_bound!r} for {owner_terms} terms)", bit=None, n_flagged=n_flagged)
    receipt = LayeredStepReceipt(
        dt_s=dt, t0_s=t0, t1_s=t1, preferential=p.budget, matrix=m.budget, column=c.budget, rain_m3=rain_m3,
        surface_before_m3=h0_m3, surface_after_m3=h1_m3, soil_before_m3=soil0_m3, soil_after_m3=soil1_m3,
        export_m3=export, storage_change_m3=storage, router_residual_m3=router_residual, max_cfl=max_cfl,
        limited_cells=limited_cells, limited_volume_m3=limited_volume, max_cell_balance_residual_m=max_cell_balance,
        outlet_discharge_m3_s=outlet_q, gross_face_volume_m3=gross_total, global_residual_m3=residual,
        global_bound_m3=bound, owner_bound_m3=owner_bound, owner_global_terms=owner_terms,
        max_surface_cell_residual_m3=max_res_s, max_soil_cell_residual_m3=max_res_w,
        surface_cell_coefficient=op.surface_cell_coefficient, soil_cell_coefficient=op.soil_cell_coefficient,
        surface_cell_terms=surface_cell_terms(L), soil_cell_terms=soil_cell_terms(L), handoff_exact=handoff_exact,
        n_flagged=n_flagged)
    state = LayeredWaterState(t1, s.state.depth_m, s.state.qx_m2_s, s.state.qy_m2_s, c.state, m.history, bed1,
                              accepted.ledger.advanced(receipt), int(accepted.generation) + 1, op.sha256)
    return LayeredStepCandidate(state=state, receipt=receipt, dt_s=dt, surface_step=s,
                                vertical=VerticalArrays(p.intake_m, m.intake_m, c.surface_return_m, c.drainage_m))


# --- provenance ------------------------------------------------------------------------------------------------------
def describe(op: LayeredStormOperator) -> dict[str, Any]:
    _check_operator(op)
    return {
        "operation": "pure fixed-bed water-only single-wetting-episode layered storm step",
        "backend": op.backend, "namespace": op.xp.__name__, "device_id": op.device_id, "shape": list(op.shape),
        "dx_m": op.dx_m, "n_layers": op.n_layers, "n_cells": op.n_cells, "n_active": op.column.n_active,
        "n_faces": op.n_faces, "stage_order": list(STAGE_ORDER), "surface_method": SURFACE_METHOD,
        "components": {"column": op.column.layers.identity(), "preferential": op.preferential.identity(),
                       "matrix": op.matrix.identity(), "router": op.router.describe(),
                       "topology": op.topology.summary(), "graph_input_sha256": str(op.graph.input_sha256)},
        "native_bed": None if op.bed is None else {"material_sha256": op.native_material_sha256,
                                                   "windowed": type(op.bed.voxel_column).__name__,
                                                   "n_classes": op.n_classes, "native_reserve_bytes": op.native_reserve_bytes,
                                                   "alias": "state.bed.water.depth_m IS state.surface_m"},
        "binding_per_call": "router type/implementation/method/extents, sealed control values (cfl_max, limiter, "
                            "friction_treatment), graph/geometry object identity (CPU) or CUDA context binding and device, "
                            "component identities, native material object identities; no array hashing per call",
        "initial_inventory_validation": "soil finite/>= 0/<= saturation + capacity tolerance per layer, history finite/>= 0, "
                                        "every cell incl. inactive; surface/momentum by the router; refused before publication",
        "time": "stages 1-2 preserve t0; the column advances the soil to t0 + dt; the router advances the surface from t0 "
                "to t0 + dt; both ends must agree; nothing is restamped",
        "rain": "added once as pool = H0 + rain dt on the flat cells; inactive rain must be 0",
        "matrix_descriptors": "standing_before = ORIGINAL H0 (contact), incoming_rate = actual rain (onset); run-on is "
                              "the previous interval's accepted lateral water, entering as standing water",
        "column_forcing": "zero added water (both infiltration stages already credited the soil) and zero potential "
                          "evaporation: E/T/growth exactly 0 in rain and recession",
        "closure": {"per_cell_surface_terms": surface_cell_terms(op.n_layers), "per_cell_soil_terms": soil_cell_terms(op.n_layers),
                    "router_cell_terms": ROUTER_CELL_TERMS, "owner_global_terms": f"{OWNER_GLOBAL_TERMS_PER_CELL} N + {OWNER_GLOBAL_TERMS}",
                    "global": "|router residual| + 2 pref.bound + 2 matrix.bound + column.bound + owner bound",
                    "policy": "conservation.volume_roundoff_bound_m3 only; no empirical epsilon"},
        "host_reads_per_attempt": "preferential 1 + matrix 1 + column 1 + router 1 + owner 1 (+ with_water's native "
                                  "validation reads when a bed is held)",
        "recoverable": "HydraulicStepRejected only: retry the SAME accepted aggregate at a smaller dt (all stages rerun)",
        "memory": op.memory.as_dict(),
        "admission": "required_bytes = resident (accepted + owned statics, native mobile counted once) + incremental "
                     "trial + column/extra/native reserves (the column convention), + build_bytes at build, + "
                     "initialization_bytes at initialisation; compared with the column's available_bytes_fn before every "
                     "allocation; incremental_bytes reported for diagnostics; a reading, not a reservation",
        "identity": {"sha256": op.sha256, "note": "backend, shape, dx, layer identity, preferential/matrix/topology/graph "
                                                  "digests, router control and the native material digest"},
        "excluded": "event loop, forcing knots, adaptive cap, quiet/complete policy, checkpoint, CLI, scheduler, LEAF, "
                    "sediment exchange, reroute/commit/publication, wetting reset/recovery, explicit method, "
                    "stage/backwater outlets; no EcoHyD identity, full-storm, restart or phase acceptance claim",
    }
