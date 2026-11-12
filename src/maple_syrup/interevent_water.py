"""Pure surface-aware INTEREVENT water step over a completed persistent layered storm (Phase 1, task
phase1_interevent_step_20261110).

SYRUP owns all physical water. This module composes the EXISTING accepted components into ONE pure, fixed-bed,
fixed-property, fixed-trait, water-only trial step over `[t0, t0 + dt]` with ZERO rain and with surface AND soil
evaporation, on NumPy, Numba or CUDA, without re-deriving any physics and without weakening the storm owner:

    op   = prepare_layered_storm(...)                       # the actual final storm operator (immutable, unchanged)
    evap = soil_evaporation.prepare_evaporation(column, partition)       # SAME column
    prep = prepare_interevent_water(op, evap, cover)                     # factory-only immutable binding
    st0  = enter_interevent(prep, event, completed_event_state)          # genuine completed storm -> general record
    cand = interevent_water_step(prep, st0, pet_terrain_m_s, dt_s, phase="interevent")
    st1  = cand.state                                                    # the CALLER adopts the whole candidate or drops it

Stage order of one trial (every stage is the existing pure function; the owner only forms the pool, adds the surface
return, binds the times and checks closure; see docs/subsurface_hydrology/interevent_water.md):

  1. validate types/bindings/phase/time/scalars (metadata only; native/CUDA context preflight), admit the COMPOSED
     peak against the column's reading;
  2. `preferential_step(H0, W0, dt)`: retained-pond water continues to enter the soil (one aggregate budget). No rain
     is manufactured: the pool is exactly H0;
  3. `matrix_step(remaining pool, ORIGINAL H0, ZERO incoming rate, credited soil, J0, dt)`: the persistent Green-Ampt
     history carries on; the zero onset rate is a descriptor, never replaced by H/dt;
  4. `stage_evaporation(PET_terrain, fixed cover C, remaining ACTUAL pool, dt)`: the surface branch removes actual
     surface E (bounded by the standing pool) and the helper returns BOTH the unpartitioned combined soil potential
     rate and the post-cap evaporation share; the two branches receive the same PET independently (no new combined
     E + T <= PET cap, no subtraction of surface E from the soil potential);
  5. ONE `subsurface.step` on the already credited soil with ZERO added water and the exact helper pair
     (`StagedEvaporation.native_step_kwargs()`): gravity drainage, the post-cap soil E, exact pair relaxation, soil
     advanced t0 -> t1. T is exactly 0 (no implicit root withdrawal, no LEAF catch-up);
  6. `post = surface after E + actual column surface return`; the genuine local-inertial `SurfaceRouter.step` from
     `SurfaceState(t0, post, q0)`; the surface end time must equal the soil end time `t0 + dt`;
  7. optional genuine native `with_water`, then ONE batched owner packet (per-cell surface/soil closure with MAPLE
     bounds, exact inactive preservation, E bounds, composed global closure) and a fresh candidate.

Contracts. Pure: nothing accepted is written. The ONLY recoverable failure is the router's `HydraulicStepRejected`
(late CFL, raised AFTER the intake, surface E, soil E and drainage trials): the caller retries the SAME accepted record at
a smaller dt; every stage reruns from the unchanged accepted state, so actual E/intake are spent once. There are no
retries inside this function. Every other defect is a typed nonrecoverable refusal. No wind/rain/E during recession (the
entry accepts a genuinely completed storm only), no wetting reset or recovery, no changed trait/property binding, no
LEAF growth or transpiration (T = 0), no sediment, reroute, checkpoint, weather driver, calendar or scheduler.

Numerical departure (declared, not a new law): the existing column performs gravity -> E -> exact diffusion in ONE
advancing call and this owner routes after it; the EcoHyD descendant orders runoff -> diffusion -> daily ET. First-order
operator split; no exact source discretization is claimed. Timestep refinement must be tested by the caller.

Nothing here was run on a GPU by its author; Codex records executions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np
from maple.core.types.voxel import VoxelColumnState

from maple_syrup import interevent_entry as ie
from maple_syrup import layered_event as ev
from maple_syrup import layered_storm as ls
from maple_syrup import matrix_infiltration as mi
from maple_syrup import preferential_infiltration as pi
from maple_syrup import soil_evaporation as se
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.sediment_bed import BedState, with_water
from maple_syrup.surface_routing import (
    ExperimentalHydrologyError,
    HydraulicStepRejected,
    SurfaceState,
    SurfaceStep,
    advance_time,
    estimate_surface_memory,
)

__all__ = [
    "BOUNDARY_ORIGIN",
    "BoundaryEntryMemory",
    "BoundaryOriginReceipt",
    "INTEREVENT",
    "MEMORY_SCOPE",
    "HydraulicStepRejected",
    "InterEventCandidate",
    "InterEventMemory",
    "InterEventMemoryError",
    "InterEventState",
    "InterEventStepError",
    "InterEventStepReceipt",
    "InterEventVertical",
    "InterEventWaterError",
    "InterEventWaterOperator",
    "StormEndReceipt",
    "WaterWindowLedger",
    "STORM_END",
    "audit_interevent",
    "describe",
    "enter_interevent",
    "enter_interevent_from_boundary",
    "estimate_boundary_entry_memory",
    "estimate_interevent_memory",
    "interevent_water_step",
    "prepare_interevent_water",
    "require_storm_origin",
    "required_boundary_entry_bytes",
    "soil_cell_terms",
    "source_kind",
    "surface_cell_terms",
]

INTEREVENT = "interevent"
#: Authenticated source kinds of an interevent window (see `source_kind`). The storm-end receipt keeps its own meaning; the
#: boundary origin is a distinct, genuinely initial/closed-boundary entry, never a fabricated storm end.
STORM_END = "storm_end"
BOUNDARY_ORIGIN = "boundary_origin"
_F8 = 8
_SEAL = object()

# --- derived term counts (see interevent_water.md section 6) -------------------------------------------------------------
#: Evaporation helper, per cell on the surface branch: cover/2, 1 - ., PET x ., x dt, min, subtraction (E and after), unmet 7 -> 8.
EVAP_SURFACE_CELL_TERMS = 8
#: Owner operations on one cell's surface value: post-pool (+ return) 1, residual evaluation (H1 - H0, x A; return - pref -
#: matrix - Es (3), x A; four face-volume conversions (8), lateral sum (3), two subtractions) 20, the scale 3 -> 24.
OWNER_SURFACE_CELL_TERMS = 24
#: Owner operations on one cell's soil value beyond the two layer sums (2 L): flux combination (pref + matrix - return -
#: drain - Esoil) 5, scaling/compare 4 -> 9.
OWNER_SOIL_CELL_TERMS = 9
#: The share multiplication `limited x share` and the unmet partition inside the column's evaporation stage.
COLUMN_SHARE_CELL_TERMS = 2
#: Owner roundoff NOT covered by any stage residual, globally: the two N-term surface reductions (H0, H1) and the post-pool
#: addition per cell (3), plus the host scalar combination (16).
OWNER_GLOBAL_TERMS_PER_CELL = 3
OWNER_GLOBAL_TERMS = 16
OWNER_PACKET_LENGTH = 17
#: Declared transient owner scratch per cell (no rain array, so the storm's 24 doubles bound it): lateral, gross (+ face
#: temporaries), vertical flux, surface residual/scale, two layer sums, soil flux/residual/scale, flags, lowest bit,
#: reduction temporaries; flag masks as bytes.
OWNER_SCRATCH_DOUBLES_PER_CELL = 24
OWNER_SCRATCH_MASK_BYTES_PER_CELL = 16
_PACKET_SCRATCH_BYTES = 4 * OWNER_PACKET_LENGTH * _F8
#: Cold preparation: the cover validity read forms three N-byte masks (not finite, < 0, > 1) at once.
COLD_MASK_BYTES_PER_CELL = 3
OBJECT_BYTES = 4096

FLAG_MESSAGES = (
    "an inactive cell's surface water changed (inactive cells must be preserved exactly)",  # 0
    "an inactive cell's soil water changed (inactive cells must be preserved exactly)",  # 1
    "an inactive cell's wetting history changed (inactive cells must be preserved exactly)",  # 2
    "per-cell surface water closure violated beyond the derived FP64 bound",  # 3
    "per-cell soil water closure violated beyond the derived FP64 bound",  # 4
    "non-finite owner closure residual or scale",  # 5
    "an inactive cell carries evaporation or unmet evaporation (forcing must be 0 there)",  # 6
    "actual surface evaporation is negative, non-finite or exceeds the pool it was taken from",  # 7
    "actual soil evaporation is negative or non-finite",  # 8
)
N_FLAGS = len(FLAG_MESSAGES)
NO_FLAG = ss.NO_FLAG

MEMORY_SCOPE = (
    "Composed interevent water-owner ledger under the EXISTING column convention (the whole working set held for one step "
    "plus the reserves is the admitted requirement), ADMITTED before every step against the COLUMN's available_bytes_fn on "
    "the column's side (host for numpy/numba, device for cuda). required_bytes = resident (the accepted aggregate: surface, "
    "faces, layers, history and, with a native bed, its mobile (N, K) array, whose depth IS the surface and is counted "
    "once) + static (the storm operator's owned prepared arrays, the owner's fixed cover map and zero incoming-rate map) + "
    "ONE trial (the caller's PET vector; the declared step allocations of the preferential, matrix, evaporation helper "
    "(outputs + scratch + packet), column WITH the evaporation share (`ss.estimate_step_memory(with_evaporation_share="
    "True)`: the share input is the helper's own output, counted once there) and surface-router stages; the post-"
    "evaporation pool; the owner's closure scratch and packet; the genuine native validation masks of `with_water`) + the "
    "column reserve + the storm's extra and native reserves + this owner's extra reserve. The stage sum is a bound as if "
    "every stage output were live at once. A fixed budget (constant reading) is a whole-owner quota, never extra free "
    "memory. COLD (prepare): the cover/zero maps and masks admitted before any allocation or download, with CUDA host "
    "staging admitted separately. EXCLUDED: allocator pools, CUDA context/NVRTC/Numba JIT workspaces, Python objects beyond "
    "declared allowances, native voxel/ledger material outside the water owner (represented ONLY by the storm operator's "
    "explicit native_reserve_bytes), checkpoint staging, LEAF and the remaining process memory. A reading is not a "
    "reservation; no whole-model fit is claimed."
)


# --- errors ----------------------------------------------------------------------------------------------------------
class InterEventWaterError(ValueError):
    """Invalid binding, record, array, time, dt, phase or option. Raised before any result exists; inputs are untouched."""


class InterEventStepError(InterEventWaterError):
    """The owner's batched validation recorded a failure (lowest bit first) or the composed closure failed."""

    def __init__(self, message: str, *, bit: int | None = None, n_flagged: int = 0):
        super().__init__(message)
        self.bit = bit
        self.n_flagged = int(n_flagged)


class InterEventMemoryError(MemoryError):
    """The composed estimate plus the reserves exceeds the reported available bytes (nothing allocated)."""


def _real(value: Any, name: str) -> float:
    try:
        return ls._real(value, name)
    except ls.LayeredStormError as exc:
        raise InterEventWaterError(str(exc)) from None


def _nonneg(value: Any, name: str) -> float:
    out = _real(value, name)
    if out < 0.0:
        raise InterEventWaterError(f"{name} must be >= 0, got {out!r}")
    return out


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise InterEventWaterError(f"{name} must be a non-Boolean integer >= 0, got {value!r}")
    return int(value)


# --- memory ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class InterEventMemory:
    """Composed ledger (see MEMORY_SCOPE). `required_bytes` is admitted before every step."""

    backend: str
    n_layers: int
    n_cells: int
    n_classes: int
    accepted_bytes: int
    static_bytes: int
    resident_bytes: int
    stage_allocation_bytes: dict
    owner_scratch_bytes: int
    native_validation_bytes: int
    trial_bytes: int
    share_input_bytes: int  # the column's extra input with the share; contained in the helper's declared outputs
    column_reserve_bytes: int
    storm_extra_reserve_bytes: int
    native_reserve_bytes: int
    extra_reserve_bytes: int
    build_bytes: int  # cold prepare allocations on the admitted side
    build_host_bytes: int  # CUDA cold host staging (0 on CPU: shares the quota)
    required_bytes: int
    scope: str = MEMORY_SCOPE

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def estimate_interevent_memory(op: ls.LayeredStormOperator, evaporation: se.EvaporationOperator, *,
                               extra_reserve_bytes: int = 0) -> InterEventMemory:
    """Pure arithmetic from the components' own estimates for the actual operator; no allocation, no query."""
    L, N, K = op.n_layers, op.n_cells, op.n_classes
    backend = op.backend
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    ny, nx = op.shape
    pref = pi.estimate_preferential_memory(L, N, backend=backend, n_species=op.preferential.n_species)
    mat = mi.estimate_matrix_memory(L, N, backend=backend)
    mode = op.column.layers.diffusion_mode
    col = ss.estimate_step_memory(L, N, backend=backend, diffusion_mode=mode, with_evaporation_share=True)
    col_plain = ss.estimate_step_memory(L, N, backend=backend, diffusion_mode=mode)
    share_input = col.input_bytes - col_plain.input_bytes
    evap = se.estimate_evaporation_memory(N, backend=backend, reserve_bytes=op.column.reserve_bytes,
                                          extra_reserve_bytes=evaporation.extra_reserve_bytes)
    surf = estimate_surface_memory(ls.SURFACE_METHOD, (ny, nx), backend=backend, limiter=op._control_values[1])
    owner = OWNER_SCRATCH_DOUBLES_PER_CELL * N * _F8 + OWNER_SCRATCH_MASK_BYTES_PER_CELL * N + _PACKET_SCRATCH_BYTES
    native_validation = ls.native_validation_bytes(N, K, backend) if op.bed is not None else 0
    stages = {"pet_input": N * _F8, "preferential": pref.step_allocation_bytes, "matrix": mat.step_allocation_bytes,
              "evaporation": evap.output_bytes + evap.scratch_bytes + evap.packet_bytes,
              "column_with_share": col.step_allocation_bytes, "post_pool": N * _F8,
              "surface": surf.step_allocation_bytes, "owner": owner, "native_wrapper": native_validation}
    trial = sum(stages.values())
    accepted = op.memory.accepted_bytes
    static = op.memory.static_bytes + 2 * N * _F8  # + the fixed cover and the zero incoming-rate maps
    resident = accepted + static
    build = 2 * N * _F8 + COLD_MASK_BYTES_PER_CELL * N + OBJECT_BYTES
    build_host = min(ls.HASH_CHUNK_BYTES, N * _F8) if backend == "cuda" else 0
    required = (resident + trial + op.column.reserve_bytes + op.extra_reserve_bytes + op.native_reserve_bytes
                + evaporation.extra_reserve_bytes + extra)
    return InterEventMemory(backend, L, N, K, accepted, static, resident, stages, owner, native_validation, trial,
                            share_input, op.column.reserve_bytes, op.extra_reserve_bytes, op.native_reserve_bytes,
                            evaporation.extra_reserve_bytes + extra, build, build_host, required)


def _admit(op: ls.LayeredStormOperator, need: int, what: str) -> int:
    """Compare `need` with the COLUMN's reported bytes; raise `InterEventMemoryError` before any allocation."""
    available = op.column.available_bytes_fn()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise InterEventWaterError(f"available_bytes_fn must return a non-Boolean integer >= 0, got {available!r}")
    if int(need) > int(available):
        raise InterEventMemoryError(
            f"interevent water {what} on {op.n_layers} layers x {op.shape[0]}x{op.shape[1]} cells ({op.backend}) needs an "
            f"estimated {int(need)} B on its admitted side (reserves included) but only {int(available)} B are reported "
            f"available; nothing was allocated. Scope: {MEMORY_SCOPE}")
    return int(available)


# --- ledger / receipts -------------------------------------------------------------------------------------------------
_CUMULATIVE = ("rain_m3", "preferential_intake_m3", "matrix_intake_m3", "surface_return_m3", "drainage_m3", "export_m3",
               "surface_evaporation_m3", "soil_evaporation_m3", "transpiration_m3", "surface_evaporation_unmet_m3",
               "soil_evaporation_unmet_m3", "global_residual_bound_m3", "max_abs_global_residual_m3")


@dataclass(frozen=True)
class WaterWindowLedger:
    """Cumulative host scalars (m3, active cells) of ONE water-only window: the storm that preceded this interevent chain
    plus the interevent steps. It is the GENERAL window ledger (separate surface E, soil E and accepted T), distinct from
    the storm-only `LayeredWaterLedger` (whose evaporation must be exactly 0 and which is left unchanged). It carries NO
    full-run, multi-model or lifetime authority: the window starts at the source storm's original anchor (see
    `InterEventState`). `transpiration_m3` is exactly 0 here (T0: no LEAF request is accepted in this module); the unmet
    columns are diagnostics, never losses."""

    n_steps: int = 0
    elapsed_s: float = 0.0
    rain_m3: float = 0.0
    preferential_intake_m3: float = 0.0
    matrix_intake_m3: float = 0.0
    surface_return_m3: float = 0.0
    drainage_m3: float = 0.0
    export_m3: float = 0.0
    surface_evaporation_m3: float = 0.0
    soil_evaporation_m3: float = 0.0
    transpiration_m3: float = 0.0
    surface_evaporation_unmet_m3: float = 0.0
    soil_evaporation_unmet_m3: float = 0.0
    global_residual_bound_m3: float = 0.0
    max_abs_global_residual_m3: float = 0.0

    @classmethod
    def from_storm(cls, ledger: ls.LayeredWaterLedger) -> WaterWindowLedger:
        """Explicit conversion of a validated storm ledger (E exactly 0): counts, elapsed time and every cumulative
        flux/bound carry over unchanged; the new E/T/unmet columns start at exactly 0."""
        ls._check_ledger(ledger)
        return cls(int(ledger.n_steps), float(ledger.elapsed_s), float(ledger.rain_m3), float(ledger.preferential_intake_m3),
                   float(ledger.matrix_intake_m3), float(ledger.surface_return_m3), float(ledger.drainage_m3),
                   float(ledger.export_m3), 0.0, 0.0, 0.0, 0.0, 0.0,
                   float(ledger.global_residual_bound_m3), float(ledger.max_abs_global_residual_m3))

    def advanced(self, receipt: InterEventStepReceipt) -> WaterWindowLedger:
        _check_ledger(self)
        col = receipt.column
        result = WaterWindowLedger(
            int(self.n_steps) + 1, float(self.elapsed_s) + float(receipt.dt_s), float(self.rain_m3),
            float(self.preferential_intake_m3) + float(receipt.preferential.intake_m3),
            float(self.matrix_intake_m3) + float(receipt.matrix.intake_m3),
            float(self.surface_return_m3) + float(col.surface_return_m3), float(self.drainage_m3) + float(col.drainage_m3),
            float(self.export_m3) + float(receipt.export_m3),
            float(self.surface_evaporation_m3) + float(receipt.surface_evaporation_m3),
            float(self.soil_evaporation_m3) + float(col.evaporation_m3), float(self.transpiration_m3),
            float(self.surface_evaporation_unmet_m3) + float(receipt.evaporation.unmet_m3),
            float(self.soil_evaporation_unmet_m3) + float(col.evaporation_unmet_m3),
            float(self.global_residual_bound_m3) + float(receipt.global_bound_m3),
            max(float(self.max_abs_global_residual_m3), abs(float(receipt.global_residual_m3))))
        _check_ledger(result, int(self.n_steps) + 1)
        return result

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _check_ledger(ledger: Any, steps: int | None = None) -> None:
    if type(ledger) is not WaterWindowLedger:
        raise InterEventWaterError("ledger must be a genuine WaterWindowLedger")
    _integer(ledger.n_steps, "ledger.n_steps")
    if steps is not None and int(ledger.n_steps) != int(steps):
        raise InterEventWaterError("ledger.n_steps must equal the generation")
    for name in ("elapsed_s", *_CUMULATIVE):
        _nonneg(getattr(ledger, name), "ledger." + name)
    if ledger.transpiration_m3 != 0.0:
        raise InterEventWaterError("ledger.transpiration_m3 must be exactly 0 in the water-only interevent owner (T0)")
    if ledger.max_abs_global_residual_m3 > ledger.global_residual_bound_m3:
        raise InterEventWaterError("ledger.max_abs_global_residual_m3 exceeds its accumulated bound")


_RECEIPT_SCALARS = ("event_sha256", "storm_operator_sha256", "prepared_sha256", "event_origin_s", "end_t_s",
                    "start_generation", "end_generation", "accepted_steps", "quiet_since_s", "rainfall_end_s", "hold_s",
                    "anchor_surface_m3",
                    "anchor_soil_m3")
_MINTED = __import__('weakref').WeakSet()  # receipts minted by enter_interevent (in-memory authenticity only)


class StormEndReceipt:
    """Bounded scalar phase-end receipt of the source storm (never a chain of beds, owners or arrays): the prepared event,
    storm-operator and interevent-binding identities, the physical origin/end, the generation range, the quiet start, the
    ORIGINAL volume anchors and the storm ledger at its original anchor and at its end.

    Factory-only and immutable (slots, no setattr): it is minted ONLY by `enter_interevent` after the actual completed-storm
    audit, so it cannot be built or `dataclasses.replace`d with a claimed source (this class is intentionally no longer a
    dataclass at this unpublished milestone; every field stays readable and `as_dict()` gives the scalar record). Every
    step revalidates it against the prepared binding and the current state (`_check_receipt`), including a canonical
    scalar `digest` that detects an in-place edit. This authenticates an IN-MEMORY continuation only: it is not a disk
    authentication, a codec or a lifetime-scheduler record."""

    __slots__ = ("__weakref__", "anchor_ledger", "digest", "end_ledger", *_RECEIPT_SCALARS)

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise InterEventWaterError("StormEndReceipt is minted only by enter_interevent() after the actual storm audit")
        for name in (*_RECEIPT_SCALARS, "anchor_ledger", "end_ledger"):
            if name not in fields:
                raise InterEventWaterError(f"internal: receipt field {name} missing")
            object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise InterEventWaterError(f"internal: unexpected receipt fields {sorted(fields)}")
        object.__setattr__(self, "digest", _receipt_digest(self))
        _MINTED.add(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("StormEndReceipt is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("StormEndReceipt is immutable")

    def __repr__(self) -> str:
        return (f"StormEndReceipt(event={self.event_sha256[:12]}..., end_t_s={self.end_t_s!r}, "
                f"generations={self.start_generation}->{self.end_generation})")

    def as_dict(self) -> dict[str, Any]:
        out = {name: getattr(self, name) for name in _RECEIPT_SCALARS}
        out["anchor_ledger"], out["end_ledger"] = asdict(self.anchor_ledger), asdict(self.end_ledger)
        out["digest"] = self.digest
        return out


def _receipt_digest(r: Any) -> str:
    """Canonical SHA-256 of the receipt's bounded scalars (a non-finite or foreign-typed scalar raises ValueError/TypeError)."""
    body = {name: getattr(r, name) for name in _RECEIPT_SCALARS}
    body["anchor_ledger"], body["end_ledger"] = asdict(r.anchor_ledger), asdict(r.end_ledger)
    return hashlib.sha256(b"maple_syrup.interevent_water.storm_end_receipt/1"
                          + json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _sha(value: Any, name: str) -> str:
    if (type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
        raise InterEventWaterError(f"{name} must be a 64-hex SHA-256 string")
    return value


def _check_receipt(prep: Any, state: Any) -> None:
    """T0 wrapper: authenticates `state.source` through the shared scalar core on the state's own fields."""
    _check_receipt_core(prep, state.source, prepared_sha256=state.prepared_sha256, origin_s=state.origin_s,
                        anchor_surface_m3=state.anchor_surface_m3, anchor_soil_m3=state.anchor_soil_m3,
                        anchor_ledger=state.anchor_ledger, t_s=state.t_s, ledger=state.ledger, generation=state.generation)


def _check_receipt_core(prep: Any, r: Any, *, prepared_sha256: str, origin_s: float, anchor_surface_m3: float,
                        anchor_soil_m3: float, anchor_ledger: Any, t_s: float, ledger: Any, generation: int) -> None:
    """Narrow common-source dispatcher: each genuine source type keeps its OWN real validation branch (the storm-end branch is
    the unchanged original; the boundary-origin branch has its own mint identity, digest and zero-window checks). An object
    of any other type reaches the storm branch, whose original refusal applies."""
    if type(r) is BoundaryOriginReceipt:
        _check_boundary_receipt_core(prep, r, prepared_sha256=prepared_sha256, origin_s=origin_s,
                                     anchor_surface_m3=anchor_surface_m3, anchor_soil_m3=anchor_soil_m3,
                                     anchor_ledger=anchor_ledger, t_s=t_s, ledger=ledger, generation=generation)
        return
    _check_storm_receipt_core(prep, r, prepared_sha256=prepared_sha256, origin_s=origin_s, anchor_surface_m3=anchor_surface_m3,
                              anchor_soil_m3=anchor_soil_m3, anchor_ledger=anchor_ledger, t_s=t_s, ledger=ledger,
                              generation=generation)


def source_kind(source: Any) -> str:
    """The authenticated kind of an interevent source: `STORM_END` or `BOUNDARY_ORIGIN`. A source that is not a genuine,
    owner-minted receipt of either kind is refused (never classified by a field or a dictionary)."""
    if type(source) is StormEndReceipt and source in _MINTED:
        return STORM_END
    if type(source) is BoundaryOriginReceipt and source in _MINTED_BOUNDARY:
        return BOUNDARY_ORIGIN
    raise InterEventWaterError("state.source is not a StormEndReceipt or BoundaryOriginReceipt minted by this owner; refused")


def require_storm_origin(source: Any, what: str) -> None:
    """Interfaces whose contract is storm-origin only (for example the Windowed profile, or a codec profile that declares only
    the storm-end source) call this: an authenticated boundary origin is refused there, explicitly and by name."""
    if source_kind(source) != STORM_END:
        raise InterEventWaterError(f"{what} supports a genuine storm-end source only; the boundary-origin source is not "
                                   "supported by it (extend and requalify that interface explicitly); refused")


def _check_storm_receipt_core(prep: Any, r: Any, *, prepared_sha256: str, origin_s: float, anchor_surface_m3: float,
                              anchor_soil_m3: float, anchor_ledger: Any, t_s: float, ledger: Any, generation: int) -> None:
    """Bounded scalar authentication of the source receipt `r` against the prepared (BASE) binding and the window's own
    hydrology ledger/anchors (explicit scalars, so the T0 state and the general state share one body; the ledger here is
    always the hydrology-only `WaterWindowLedger`). No array is read and no hash of source material is made; every
    comparison is exact Python scalar arithmetic or the declared first-order elapsed-time bound. A forged-format SHA with a
    plausible past end fails the minted-identity, digest and cross-binding checks, not only an obviously future clock."""
    if type(r) is not StormEndReceipt or r not in _MINTED:
        raise InterEventWaterError("state.source is not a StormEndReceipt minted by enter_interevent(); refused")
    try:
        if _receipt_digest(r) != r.digest:
            raise InterEventWaterError("the source receipt was altered after it was minted; refused")
    except (ValueError, TypeError, AttributeError) as exc:
        raise InterEventWaterError(f"the source receipt carries a blank, non-finite or foreign-typed scalar: {exc}") from None
    for name in ("event_sha256", "storm_operator_sha256", "prepared_sha256"):
        _sha(getattr(r, name), "receipt." + name)
    if r.storm_operator_sha256 != prep.op.sha256:
        raise InterEventWaterError("the source storm belongs to a different storm operator than this binding; refused")
    if r.prepared_sha256 != prep.sha256 or prepared_sha256 != prep.sha256:
        raise InterEventWaterError("the source receipt was minted for a different interevent binding; refused")
    origin, end, quiet, rain_end, hold = (_nonneg(getattr(r, n), "receipt." + n) for n in
                                          ("event_origin_s", "end_t_s", "quiet_since_s", "rainfall_end_s", "hold_s"))
    a_h, a_w = _nonneg(r.anchor_surface_m3, "receipt.anchor_surface_m3"), _nonneg(r.anchor_soil_m3, "receipt.anchor_soil_m3")
    g0, g1, steps = (_integer(getattr(r, n), "receipt." + n) for n in ("start_generation", "end_generation", "accepted_steps"))
    if origin != origin_s or a_h != anchor_surface_m3 or a_w != anchor_soil_m3:
        raise InterEventWaterError("the source receipt's origin/volume anchors differ from the window's; refused")
    if not (origin < rain_end <= quiet <= end and hold > 0.0 and end - quiet >= hold):
        raise InterEventWaterError("source quiet chronology violated: need origin < rainfall end <= quiet start and a "
                                   "completed hold before the end; refused")
    if g1 != g0 + steps:
        raise InterEventWaterError("source generation range disagrees with its accepted step count; refused")
    try:
        ls._check_ledger(r.anchor_ledger, g0)
        ls._check_ledger(r.end_ledger, g1)
    except ls.LayeredStormError as exc:
        raise InterEventWaterError(f"source storm ledger refused (storm evaporation must be exactly 0): {exc}") from None
    if WaterWindowLedger.from_storm(r.anchor_ledger) != anchor_ledger:
        raise InterEventWaterError("the window anchor ledger is not the exact conversion of the source anchor ledger; refused")
    end_led = WaterWindowLedger.from_storm(r.end_ledger)
    for name in _CUMULATIVE:
        if getattr(end_led, name) < getattr(anchor_ledger, name):
            raise InterEventWaterError(f"source end ledger {name} predates its anchor; refused")
    eps = np.finfo(np.float64).eps
    if abs((end_led.elapsed_s - anchor_ledger.elapsed_s) - (end - origin)) > 8.0 * (steps + 2) * eps * max(end, origin, end_led.elapsed_s):
        raise InterEventWaterError("source ledger elapsed time disagrees with its origin/end; refused")
    t, led, gen = t_s, ledger, generation
    if end > t or gen < g1:
        raise InterEventWaterError("the source storm ends after the current state (future source clock/generation); refused")
    for name in ("n_steps", "elapsed_s", *_CUMULATIVE):
        if getattr(led, name) < getattr(end_led, name):
            raise InterEventWaterError(f"current ledger {name} predates the source storm end; refused")
    later = gen - g1
    if led.n_steps - end_led.n_steps != later:
        raise InterEventWaterError("current ledger count disagrees with the generations since the source end; refused")
    if abs((led.elapsed_s - end_led.elapsed_s) - (t - end)) > 8.0 * (later + 2) * eps * max(t, end, led.elapsed_s):
        raise InterEventWaterError("current ledger elapsed time since the source end disagrees with the clock; refused")
    if later == 0 and (t != end or led != end_led):
        raise InterEventWaterError("a state at the source end must equal the converted source end ledger; refused")
    for name in ("surface_evaporation_m3", "soil_evaporation_m3", "surface_evaporation_unmet_m3",
                 "soil_evaporation_unmet_m3", "transpiration_m3"):
        if getattr(end_led, name) != 0.0:
            raise InterEventWaterError("the source storm ledger carries evaporation; refused")


@dataclass(frozen=True, eq=False)
class InterEventState:
    """One accepted interevent aggregate instant. Arrays of the operator namespace, SHARED with the completed storm
    aggregate at entry and fresh per step: `surface_m (ny, nx)`, `qx_m2_s (ny, nx+1)`, `qy_m2_s (ny+1, nx)`; `soil` at the
    same `t_s`; the bound `history` (J carried, never reset); the native `bed` (its `water.depth_m` IS `surface_m`) or
    None; the general `ledger` and the window `anchor_*` (the source storm's ORIGINAL volume anchors and anchor ledger
    converted explicitly, so the closure is a window difference, never an invented initial water); `origin_s`; the
    bounded `source` receipt; `generation`; `prepared_sha256` (the binding). Built by `enter_interevent` or a step."""

    t_s: float
    surface_m: Any
    qx_m2_s: Any
    qy_m2_s: Any
    soil: ss.SoilColumnState
    history: mi.WettingHistory
    bed: Any
    ledger: WaterWindowLedger
    generation: int
    prepared_sha256: str
    origin_s: float
    anchor_surface_m3: float
    anchor_soil_m3: float
    anchor_ledger: WaterWindowLedger
    source: StormEndReceipt


class InterEventVertical(NamedTuple):
    """The stage arrays of one candidate (referenced, not copied): preferential intake `(L, N)`, matrix intake `(N,)`,
    actual surface E `(N,)`, actual soil E `(N,)`, saturation return `(N,)`, bottom drainage `(N,)`."""

    preferential_intake_m: Any
    matrix_intake_m: Any
    surface_evaporation_m: Any
    soil_evaporation_m: Any
    surface_return_m: Any
    drainage_m: Any


@dataclass(frozen=True)
class InterEventStepReceipt:
    """Host scalars of one candidate: the stage budgets, the surface scalars, the composed terms and the closure checks."""

    dt_s: float
    t0_s: float
    t1_s: float
    preferential: pi.PreferentialBudget
    matrix: mi.MatrixBudget
    evaporation: se.SurfaceEvaporationLedger
    column: ss.StepBudget
    surface_evaporation_m3: float
    soil_evaporation_m3: float
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
    handoff_exact: bool
    n_flagged: int

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        for key in ("preferential", "matrix", "evaporation", "column"):
            value = out[key]
            out[key] = value.as_dict() if hasattr(value, "as_dict") else dict(value.__dict__)
        return out


@dataclass(frozen=True, eq=False)
class InterEventCandidate:
    """One proposed step: the fresh accepted-shaped `state` at `t1`, the `receipt`, the router's `surface_step` and the
    stage `vertical` arrays. The caller adopts `state` whole or drops the candidate; nothing is adopted partially."""

    state: InterEventState
    receipt: InterEventStepReceipt
    dt_s: float
    surface_step: SurfaceStep
    vertical: InterEventVertical


def surface_cell_terms(n_layers: int) -> int:
    """Per-cell surface closure terms: preferential (4 L + 4) + matrix (2 L + 16) + helper surface branch + router + owner."""
    L = int(n_layers)
    return ((pi.CELL_OPERATION_TERMS_PER_LAYER * L + pi.CELL_OPERATION_TERMS)
            + (mi.CELL_OPERATION_TERMS_PER_LAYER * L + mi.CELL_OPERATION_TERMS)
            + EVAP_SURFACE_CELL_TERMS + ls.ROUTER_CELL_TERMS + OWNER_SURFACE_CELL_TERMS)


def soil_cell_terms(n_layers: int) -> int:
    """Per-cell soil closure terms: preferential (4 L + 4) + matrix (2 L + 16) + column (4 L + 7 + share 2) + owner (2 L + 9)."""
    L = int(n_layers)
    return ((pi.CELL_OPERATION_TERMS_PER_LAYER * L + pi.CELL_OPERATION_TERMS)
            + (mi.CELL_OPERATION_TERMS_PER_LAYER * L + mi.CELL_OPERATION_TERMS)
            + (4 * L + 7 + COLUMN_SHARE_CELL_TERMS) + (2 * L + OWNER_SOIL_CELL_TERMS))


# --- the prepared binding ------------------------------------------------------------------------------------------------
class InterEventWaterOperator:
    """Immutable factory-only binding minted ONLY by `prepare_interevent_water`: the actual final storm operator, the
    actual evaporation operator prepared on the SAME column, the owner's canonical cover map and zero incoming-rate map
    (NumPy: immutable byte buffers; device: private arrays, inspected only through detached copies), the composed memory
    estimate and a cold SHA-256. Not a dataclass: a forcing/quota bypass by `replace` is impossible."""

    __slots__ = ("_cover", "_cover_digest", "_partition_values", "_seal", "_zero_rate", "backend", "evaporation",
                 "extra_reserve_bytes", "memory", "n_cells", "n_layers", "op", "sha256", "soil_cell_coefficient",
                 "surface_cell_coefficient", "xp")

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise InterEventWaterError("InterEventWaterOperator is constructed only by prepare_interevent_water()")
        for name in self.__slots__:
            if name == "_seal":
                continue
            if name not in fields:
                raise InterEventWaterError(f"internal: operator field {name} missing")
            object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise InterEventWaterError(f"internal: unexpected operator fields {sorted(fields)}")
        object.__setattr__(self, "_seal", seal)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("InterEventWaterOperator is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("InterEventWaterOperator is immutable")

    def __repr__(self) -> str:
        return (f"InterEventWaterOperator(backend={self.backend!r}, shape={self.op.shape}, n_layers={self.n_layers}, "
                f"sha256={self.sha256[:12]}...)")

    @property
    def cover(self) -> Any:
        """The fixed cover map: the immutable host array, or on a device a fresh DETACHED copy (admitted)."""
        if self.xp is np:
            return self._cover
        _check_prepared(self)
        _admit(self.op, self.memory.required_bytes + self.n_cells * _F8, "cover inspection")
        return self._cover.copy()

    def describe(self) -> dict[str, Any]:
        return describe(self)


def _partition_values(part: se.EvaporationPartition) -> tuple[float, ...]:
    return (part.canopy_intercept, part.canopy_slope, part.transpiration_slope, part.transpiration_offset, part.soil_factor)


def _check_prepared(prep: Any) -> None:
    """Metadata-only revalidation (every call, before any allocation): genuine sealed record, storm operator and its
    router/context/component bindings (CUDA: the native context preflight), evaporation operator on the same column with
    the sealed partition, the exact owned maps."""
    if type(prep) is not InterEventWaterOperator or getattr(prep, "_seal", None) is not _SEAL:
        raise InterEventWaterError(f"operator must be a genuine InterEventWaterOperator, got {type(prep).__name__}")
    op = prep.op
    try:
        ls._check_operator(op)
    except ls.LayeredStormError as exc:
        raise InterEventWaterError(f"the bound storm operator is no longer valid: {exc}") from None
    column = op.column
    if column.preflight is not None:
        column.preflight()
    e = prep.evaporation
    if (not isinstance(e, se.EvaporationOperator) or e.column is not column or e.xp is not column.xp
            or e.backend != column.backend or int(e.n_cells) != op.n_cells
            or _partition_values(e.partition) != prep._partition_values):
        raise InterEventWaterError("the evaporation operator is no longer the one sealed on this column and partition")
    if float(column.layers.evaporation_factor) != 1.0:
        raise InterEventWaterError("the column's evaporation_factor must be exactly 1.0 (the helper applies it once)")
    if prep.xp is not op.xp or prep.backend != op.backend or prep.n_cells != op.n_cells or prep.n_layers != op.n_layers:
        raise InterEventWaterError("operator record disagrees with its storm operator; refused")
    ss._check_dynamic(column, "operator cover", prep._cover, (op.n_cells,))
    ss._check_dynamic(column, "operator zero incoming rate", prep._zero_rate, (op.n_cells,))
    if prep.xp is np and (prep._cover.flags.writeable or prep._zero_rate.flags.writeable):
        raise InterEventWaterError("operator host maps lost their immutability; refused")
    if not isinstance(prep.sha256, str) or len(prep.sha256) != 64:
        raise InterEventWaterError("operator record carries a replaced identity; refused")


def prepare_interevent_water(op: ls.LayeredStormOperator, evaporation: se.EvaporationOperator, cover: Any, *,
                             extra_reserve_bytes: int = 0, host_available_bytes_fn: Any = None,
                             host_reserve_bytes: int = 0) -> InterEventWaterOperator:
    """Bind the ACTUAL final storm operator, the actual evaporation operator on the SAME column and an explicit fixed cover
    fraction map C in [0, 1] `(N,)` (a given array of the column namespace; copied once; never derived from root
    fractions). All metadata/device/quota checks precede any allocation, download or hash; the cold read of the cover
    is one counted packet. `extra_reserve_bytes` is the caller's reserve for its own simultaneously live bytes; the
    CUDA-only `host_*` controls admit the cold host staging separately."""
    try:
        ls._check_operator(op)
    except ls.LayeredStormError as exc:
        raise InterEventWaterError(f"op must be a genuine, valid LayeredStormOperator: {exc}") from None
    column, xp = op.column, op.xp
    if column.preflight is not None:
        column.preflight()  # CUDA: the current device must be the column's, before any allocation or download
    if not isinstance(evaporation, se.EvaporationOperator):
        raise InterEventWaterError(f"evaporation must be an EvaporationOperator, got {type(evaporation).__name__}")
    if evaporation.column is not column or evaporation.xp is not xp or evaporation.backend != op.backend:
        raise InterEventWaterError("the evaporation operator must be prepared on exactly the storm operator's column")
    if float(column.layers.evaporation_factor) != 1.0:
        raise InterEventWaterError("the column's evaporation_factor must be exactly 1.0 for the E helper (factor once)")
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    host_reserve = ss._checked_int(host_reserve_bytes, "host_reserve_bytes")
    if host_available_bytes_fn is not None and not callable(host_available_bytes_fn):
        raise InterEventWaterError("host_available_bytes_fn must be callable or None")
    if op.backend != "cuda" and (host_available_bytes_fn is not None or host_reserve != 0):
        raise InterEventWaterError("separate host staging controls are CUDA-only; CPU uses the column quota")
    N = op.n_cells
    try:
        ss._check_dynamic(column, "cover", cover, (N,))
    except ss.SubsurfaceError as exc:
        raise InterEventWaterError(str(exc)) from None
    memory = estimate_interevent_memory(op, evaporation, extra_reserve_bytes=extra)
    _admit(op, memory.required_bytes + memory.build_bytes, "preparation")  # BEFORE any allocation/download/hash
    if op.backend == "cuda":
        try:
            ls._admit_host_build(memory.build_host_bytes + host_reserve + OBJECT_BYTES,
                                 host_available_bytes_fn or ss.default_host_available_bytes)
        except ls.LayeredMemoryError as exc:
            raise InterEventMemoryError(str(exc)) from None
    from maple.core.backend import errstate, to_device

    with errstate(xp=xp, all="ignore"):
        bad = xp.count_nonzero(~xp.isfinite(cover) | (cover < 0.0) | (cover > 1.0))
    if ls._packet(xp, [bad])[0] != 0.0:
        raise InterEventWaterError("cover must be finite and within [0, 1] on every cell (inactive included)")
    if xp is np:
        owned_cover = ev._immutable_host(np.asarray(cover, dtype=np.float64))
        zero_rate = ev._immutable_host(np.zeros(N, dtype=np.float64))
    else:
        owned_cover = xp.ascontiguousarray(xp.array(cover, dtype=np.float64, copy=True))
        zero_rate = xp.zeros(N, dtype=np.float64)
    h = hashlib.sha256(b"maple_syrup.interevent_water.cover/1")
    ls._stream_hash(h, owned_cover)
    cover_digest = h.hexdigest()
    part = evaporation.partition
    partition_values = _partition_values(part)
    g = hashlib.sha256(b"maple_syrup.interevent_water.operator/1")
    for piece in (op.sha256, cover_digest, json.dumps(asdict(part), sort_keys=True, allow_nan=False), se.PARTITION_LEVEL,
                  str(extra), "T0"):
        g.update(piece.encode())
        g.update(b"\0")
    return InterEventWaterOperator(
        _SEAL, op=op, evaporation=evaporation, _cover=owned_cover, _cover_digest=cover_digest,
        _partition_values=partition_values, _zero_rate=zero_rate, backend=op.backend, xp=xp, n_cells=N,
        n_layers=op.n_layers, memory=memory, extra_reserve_bytes=extra,
        surface_cell_coefficient=volume_roundoff_bound_m3(surface_cell_terms(op.n_layers), 1.0),
        soil_cell_coefficient=volume_roundoff_bound_m3(soil_cell_terms(op.n_layers), 1.0), sha256=g.hexdigest())


# --- entry -------------------------------------------------------------------------------------------------------------------
def enter_interevent(prep: InterEventWaterOperator, event: ev.LayeredEvent,
                     completed: ev.LayeredEventState) -> InterEventState:
    """Accept ONLY a genuinely completed storm: `completed.completed` is True and the ACTUAL physical checks pass
    (`layered_event.audit_event`: sealed event identity, ledger/anchor/clock consistency, anchored water closure, every-cell
    inventory bounds, router state, and the instantaneous hydraulic quiet gates). A bound-stopped or still-flowing
    storm (any `completed=False` record) is refused and receives NO interevent evaporation. The event must be prepared on
    the SAME storm operator this binding wraps. The aggregate is shared by reference (never duplicated), its ledger
    converted explicitly (`WaterWindowLedger.from_storm`), its ORIGINAL anchor ledger/volume anchors/clock origin carried
    as the window anchor and a bounded scalar `StormEndReceipt` kept; wetting history J is carried, never reset.
    Nothing is allocated beyond `audit_event`'s own admitted reductions."""
    _check_prepared(prep)
    try:
        ev._check_event(event)
    except (ev.LayeredEventError, ls.LayeredStormError) as exc:
        raise InterEventWaterError(str(exc)) from None
    if event.op is not prep.op:
        raise InterEventWaterError("the completed event was prepared on a different storm operator than this binding wraps")
    if type(completed) is not ev.LayeredEventState:
        raise InterEventWaterError(f"completed must be a LayeredEventState, got {type(completed).__name__}")
    if completed.completed is not True:
        raise InterEventWaterError("the storm is not completed (a bound-stopped or flowing storm remains rain/recession "
                                   "and receives no interevent E/T); refused")
    try:
        ev._check_state(event, completed)
        if float(completed.water.t_s) < event.rainfall_end_s:
            raise InterEventWaterError("the completed storm clock precedes its rainfall end; refused")
        ev.audit_event(event, completed)  # actual inventories, anchored closure, router state and the quiet gates
    except (ev.LayeredEventError, ls.LayeredStormError, ExperimentalHydrologyError) as exc:
        raise InterEventWaterError(f"completed storm refused by its actual physical checks: {exc}") from None
    water = completed.water
    if water.operator_sha256 != prep.op.sha256:
        raise InterEventWaterError("the completed aggregate is bound to a different storm operator; refused")
    anchor = WaterWindowLedger.from_storm(completed.anchor_ledger)
    ledger = WaterWindowLedger.from_storm(water.ledger)
    receipt = StormEndReceipt(
        _SEAL, event_sha256=event.sha256, storm_operator_sha256=prep.op.sha256, prepared_sha256=prep.sha256,
        event_origin_s=float(event.event_origin_s), end_t_s=float(water.t_s), start_generation=int(completed.start_generation),
        end_generation=int(water.generation), accepted_steps=int(completed.accepted_steps),
        quiet_since_s=float(completed.quiet.since_s), rainfall_end_s=float(event.rainfall_end_s),
        hold_s=float(event.policy.hold_s), anchor_surface_m3=float(completed.anchor_surface_m3),
        anchor_soil_m3=float(completed.anchor_soil_m3), anchor_ledger=completed.anchor_ledger, end_ledger=water.ledger)
    return InterEventState(float(water.t_s), water.surface_m, water.qx_m2_s, water.qy_m2_s, water.soil, water.history,
                           water.bed, ledger, int(water.generation), prep.sha256, float(event.event_origin_s),
                           float(completed.anchor_surface_m3), float(completed.anchor_soil_m3), anchor, receipt)


# --- genuine initial / closed-boundary entry -----------------------------------------------------------------------------------
_MINTED_BOUNDARY = __import__('weakref').WeakSet()  # receipts minted by enter_interevent_from_boundary (own registry)
_BOUNDARY_SCALARS = ("prepared_sha256", "storm_operator_sha256", "matrix_sha256", "cover_sha256", "parent_token",
                     "parent_revision", "origin_s", "anchor_surface_m3", "anchor_soil_m3", "eligibility",
                     "max_active_surface_depth_m")
ELIGIBILITY = "instantaneous_initial_condition"
MAX_TOKEN_CHARS = 128


class BoundaryOriginReceipt:
    """Bounded scalar origin record of an interevent window that began at an INITIALIZED or previously CLOSED physical boundary
    (not at the end of a storm). Distinct from `StormEndReceipt`: it carries no rain end, quiet hold, generation range, ledger,
    event identity, array, bed, operator or history chain, and claims NO historical storm. It records the prepared/storm-operator
    /matrix/cover identities, the parent token and revision, the physical origin, the ACTUAL volume anchors at entry, the
    explicit quiet policy and the actual instantaneous hydraulic evidence (`eligibility` states that this is the instantaneous
    eligibility of an initial condition, not an executed hold).

    Factory-only (`enter_interevent_from_boundary`), immutable (slots, no setattr/delattr) and registered in its OWN registry,
    so it can neither be built nor `dataclasses.replace`d with a claimed source and never enters the storm registry. Every use
    revalidates it (`_check_boundary_receipt_core`), including a canonical scalar `digest` that detects an in-place edit. In-memory
    authenticity only: not a disk authentication or a codec record (a future codec dispatches on `source_kind`)."""

    __slots__ = ("__weakref__", "digest", "evidence", "policy", *_BOUNDARY_SCALARS)

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise InterEventWaterError("BoundaryOriginReceipt is minted only by enter_interevent_from_boundary()")
        for name in (*_BOUNDARY_SCALARS, "policy", "evidence"):
            if name not in fields:
                raise InterEventWaterError(f"internal: boundary receipt field {name} missing")
            value = fields.pop(name)
            if name in ("policy", "evidence"):
                value = __import__("types").MappingProxyType(dict(value))
            object.__setattr__(self, name, value)
        if fields:
            raise InterEventWaterError(f"internal: unexpected boundary receipt fields {sorted(fields)}")
        object.__setattr__(self, "digest", _boundary_digest(self))
        _MINTED_BOUNDARY.add(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("BoundaryOriginReceipt is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("BoundaryOriginReceipt is immutable")

    def __repr__(self) -> str:
        return f"BoundaryOriginReceipt(origin_s={self.origin_s!r}, parent_revision={self.parent_revision}, ...)"

    @property
    def source_kind(self) -> str:
        return BOUNDARY_ORIGIN

    def as_dict(self) -> dict[str, Any]:
        out = {name: getattr(self, name) for name in _BOUNDARY_SCALARS}
        out["policy"], out["evidence"] = dict(self.policy), dict(self.evidence)
        out["source_kind"] = BOUNDARY_ORIGIN
        out["digest"] = self.digest
        return out


def _boundary_digest(r: Any) -> str:
    """Canonical SHA-256 of the receipt's bounded scalars (a non-finite or foreign-typed scalar raises ValueError/TypeError)."""
    body = {name: getattr(r, name) for name in _BOUNDARY_SCALARS}
    body["policy"], body["evidence"] = dict(r.policy), dict(r.evidence)
    return hashlib.sha256(b"maple_syrup.interevent_water.boundary_origin_receipt/1"
                          + json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _check_boundary_receipt_core(prep: Any, r: Any, *, prepared_sha256: str, origin_s: float, anchor_surface_m3: float,
                                 anchor_soil_m3: float, anchor_ledger: Any, t_s: float, ledger: Any, generation: int) -> None:
    """Bounded scalar authentication of a boundary-origin source against the prepared binding and the window's own
    ledger/anchors. No array is read, nothing is hashed; every comparison is exact scalar arithmetic. The window must have begun
    with an exactly all-zero local ledger at the receipt's origin and anchors; it never requires or invents a storm."""
    if type(r) is not BoundaryOriginReceipt or r not in _MINTED_BOUNDARY:
        raise InterEventWaterError("state.source is not a BoundaryOriginReceipt minted by enter_interevent_from_boundary(); "
                                   "refused")
    try:
        if _boundary_digest(r) != r.digest:
            raise InterEventWaterError("the boundary source receipt was altered after it was minted; refused")
    except (ValueError, TypeError, AttributeError) as exc:
        raise InterEventWaterError(f"the boundary source receipt carries a blank, non-finite or foreign-typed scalar: "
                                   f"{exc}") from None
    for name in ("prepared_sha256", "storm_operator_sha256", "matrix_sha256", "cover_sha256"):
        _sha(getattr(r, name), "receipt." + name)
    if r.storm_operator_sha256 != prep.op.sha256 or r.matrix_sha256 != prep.op.matrix.sha256:
        raise InterEventWaterError("the boundary source was minted for different storm-operator/matrix components; refused")
    if r.cover_sha256 != prep._cover_digest:
        raise InterEventWaterError("the boundary source was minted for a different cover map; refused")
    if r.prepared_sha256 != prep.sha256 or prepared_sha256 != prep.sha256:
        raise InterEventWaterError("the boundary source was minted for a different interevent binding; refused")
    _token(r.parent_token, "receipt.parent_token")
    _integer(r.parent_revision, "receipt.parent_revision")
    origin = _nonneg(r.origin_s, "receipt.origin_s")
    a_h, a_w = _nonneg(r.anchor_surface_m3, "receipt.anchor_surface_m3"), _nonneg(r.anchor_soil_m3, "receipt.anchor_soil_m3")
    _nonneg(r.max_active_surface_depth_m, "receipt.max_active_surface_depth_m")
    if origin != origin_s or a_h != anchor_surface_m3 or a_w != anchor_soil_m3:
        raise InterEventWaterError("the boundary source's origin/volume anchors differ from the window's; refused")
    if r.eligibility != ELIGIBILITY:
        raise InterEventWaterError("the boundary source carries an unknown eligibility statement; refused")
    if set(r.policy) != set(ie.POLICY_NAMES) or set(r.evidence) != set(ie.EVIDENCE_NAMES):
        raise InterEventWaterError("the boundary source's policy/evidence scalars are malformed; refused")
    try:
        policy = ie.policy_scalars(ev.LayeredQuietPolicy(**{k: r.policy[k] for k in ie.POLICY_NAMES}))
    except (ls.LayeredStormError, ev.LayeredEventError, TypeError, ValueError) as exc:
        raise InterEventWaterError(f"the boundary source's quiet policy is invalid: {exc}") from None
    ok, failing = ie.eligibility({k: r.evidence[k] for k in ie.EVIDENCE_NAMES}, policy)
    if not ok or r.max_active_surface_depth_m != r.evidence["max_active_surface_depth_m"]:
        raise InterEventWaterError(f"the boundary source's recorded evidence is not eligible under its own policy "
                                   f"({failing}); refused")
    if anchor_ledger != WaterWindowLedger():
        raise InterEventWaterError("a boundary-origin window must anchor at an exactly all-zero local ledger; refused")
    if t_s < origin:
        raise InterEventWaterError("the state clock precedes the boundary origin; refused")
    if generation == 0 and (t_s != origin or ledger != anchor_ledger):
        raise InterEventWaterError("a state at the boundary origin must carry the zero origin ledger at the origin clock; refused")
    if int(ledger.n_steps) != int(generation):
        raise InterEventWaterError("the window ledger count disagrees with the generation since the boundary origin; refused")


def _token(value: Any, name: str) -> str:
    if type(value) is not str or not value or len(value) > MAX_TOKEN_CHARS or not value.isprintable():
        raise InterEventWaterError(f"{name} must be a printable non-empty string of at most {MAX_TOKEN_CHARS} characters")
    return value


# --- entry memory ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BoundaryEntryMemory:
    """Named, metadata-only requirement of ONE `enter_interevent_from_boundary` call under the column convention. Every term is
    derived from the actual simultaneous parent aggregate (counted once: the native mobile array, its depth H0 and the shared
    material appear once, inside `resident_bytes`), the real prepared statics (cover and zero-rate maps), the inventory
    reductions and validation masks, the quiet-evidence scratch and the reserves. No source tree and no hot J hash is read
    (`source_hash_bytes` is 0 by construction: the receipt carries the already-computed cold cover digest and scalar identities).
    `required_bytes` is admitted BEFORE any reduction, hash, native validation or allocation."""

    backend: str
    n_layers: int
    n_cells: int
    n_faces: int
    resident_bytes: int
    inventory_scratch_bytes: int
    validation_mask_bytes: int
    quiet_scratch_bytes: int
    source_hash_bytes: int
    column_reserve_bytes: int
    storm_extra_reserve_bytes: int
    native_reserve_bytes: int
    extra_reserve_bytes: int
    object_bytes: int
    required_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


#: Explicit UNQUALIFIED opt-in literal that lets `enter_interevent_from_boundary` run on a genuine CUDA operator (native39 Dense or bedless; the lifecycle
#: composes it with the real CUDA general owner). Missing/other literal: the unchanged early refusal. The CPU refuses the option.
CUDA_BOUNDARY_ENTRY_PROFILE = "maple_syrup.interevent_water.boundary_entry/native39_cuda_dense_or_bedless_UNQUALIFIED"


def estimate_boundary_entry_memory(op: Any, evaporation: se.EvaporationOperator, *,
                                   extra_reserve_bytes: int = 0) -> BoundaryEntryMemory:
    """Pure arithmetic from the components' own estimates for the actual (or lifecycle-implied) storm operator; no allocation,
    no query. `op` needs the storm operator's metadata fields only, so the lifecycle can admit BEFORE preparing it."""
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    base = estimate_interevent_memory(op, evaporation, extra_reserve_bytes=extra)
    L, N = op.n_layers, op.n_cells
    F = int(op.memory.n_faces)
    inventory = (L + 2) * N * _F8 + OBJECT_BYTES  # the two masked copies (H: N, soil: L N) and the host packet
    masks = int(op.memory.initialization_bytes)  # the genuine inventory-validation masks of the storm initializer
    quiet = ie.QUIET_SCRATCH_DOUBLES_PER_CELL * N * _F8 + 2 * F * _F8
    if op.backend == "cuda":
        quiet += 2 * N * _F8 + F  # the quiet maps uploaded once per boundary entry: escape level + source bed (N doubles each) and the two outlet-type masks
    reserves = (int(op.column.reserve_bytes) + int(op.extra_reserve_bytes) + int(op.native_reserve_bytes)
                + int(evaporation.extra_reserve_bytes) + extra)
    required = int(base.resident_bytes) + inventory + masks + quiet + reserves + OBJECT_BYTES
    return BoundaryEntryMemory(op.backend, L, N, F, int(base.resident_bytes), inventory, masks, quiet, 0,
                               int(op.column.reserve_bytes), int(op.extra_reserve_bytes), int(op.native_reserve_bytes),
                               int(evaporation.extra_reserve_bytes) + extra, OBJECT_BYTES, required)


def required_boundary_entry_bytes(prep: InterEventWaterOperator) -> int:
    """Metadata-only whole-call requirement for `prep` (the figure `enter_interevent_from_boundary` admits first)."""
    _check_prepared(prep)
    return int(estimate_boundary_entry_memory(prep.op, prep.evaporation, extra_reserve_bytes=prep.extra_reserve_bytes)
               .required_bytes)


# --- the factory ---------------------------------------------------------------------------------------------------------------
def enter_interevent_from_boundary(prep: InterEventWaterOperator, *, t_s: Any, surface_m: Any, qx_m2_s: Any, qy_m2_s: Any,
                                   soil: ss.SoilColumnState, history: mi.WettingHistory, bed: Any,
                                   policy: ev.LayeredQuietPolicy, parent_token: Any, parent_revision: Any,
                                   cuda_boundary_profile: Any = None) -> InterEventState:
    """Begin an interevent window at an ACTUAL initialized or closed physical boundary, without any storm event.

    Accepts the real current aggregate (surface depth H, signed faces, soil, wetting history J with its lifetime intake, the
    native Dense bed or None, the physical time) and an explicit existing `LayeredQuietPolicy`. Order: metadata fences and
    the named whole-call admission (`required_boundary_entry_bytes`) FIRST; then type/array/alias/material metadata; then the
    ACTUAL inventories (every-cell bounds, router state); then the instantaneous hydraulic eligibility from the actual signed
    faces, outlet discharge and depression excess. An unquiet instant (a role name does not matter) is refused: nothing is
    drained, rounded or reclassified. A quiet retained pond is eligible and stays actual H.

    The product is a zero-time interevent state: new phase-local `WaterWindowLedger()` and anchor, generation 0, origin = the
    physical time, volume anchors = the actual inventories, and a `BoundaryOriginReceipt`. All arrays, the soil object (with its own
    cumulative accounts), J, lifetime intake and the bed are SHARED by reference, so they are bitwise unchanged; there is no
    dry reset, recovery or event replay. CPU (NumPy/Numba) Dense or bedless; CUDA only with the separate explicit
    `cuda_boundary_profile=CUDA_BOUNDARY_ENTRY_PROFILE` opt-in (the same body; the lifecycle composes the genuine CUDA general owner and its
    native39 source/kernel guards). Windowed/foreign storage has genuine early typed fences; qualification is the actual native39 source only."""
    _check_prepared(prep)
    op = prep.op
    cpu = prep.backend in ("numpy", "numba") and op.xp is np
    if cpu and cuda_boundary_profile is not None:
        raise InterEventWaterError("cuda_boundary_profile is a CUDA-only option; the CPU entry refuses it (not ignored)")
    if not cpu:
        if cuda_boundary_profile is None:
            raise InterEventWaterError(f"the boundary interevent entry supports the NumPy/Numba backends only (CUDA is not "
                                       f"qualified here), got {prep.backend!r}; the separate explicit cuda_boundary_profile opt-in selects the "
                                       "UNQUALIFIED CUDA capability; refused")
        if type(cuda_boundary_profile) is not str or cuda_boundary_profile != CUDA_BOUNDARY_ENTRY_PROFILE:
            raise InterEventWaterError(f"cuda_boundary_profile must be exactly CUDA_BOUNDARY_ENTRY_PROFILE, got {cuda_boundary_profile!r}; refused")
        if prep.backend != "cuda" or op.xp is np:
            raise InterEventWaterError("the CUDA boundary entry needs a genuine CUDA operator namespace; refused")
    if op.bed is not None and type(op.bed.voxel_column) is not VoxelColumnState:
        raise InterEventWaterError("the boundary interevent entry supports a native Dense VoxelColumnState bed or no bed only "
                                   f"(got {type(op.bed.voxel_column).__name__}); refused")
    _admit(op, required_boundary_entry_bytes(prep), "boundary entry")  # BEFORE any value read, hash, reduction or allocation
    t = _nonneg(t_s, "t_s")
    revision = _integer(parent_revision, "parent_revision")
    token = _token(parent_token, "parent_token")
    try:
        pol = ie.policy_scalars(policy)
    except (ls.LayeredStormError, ev.LayeredEventError) as exc:  # a genuine invalid policy raises LayeredEventError
        raise InterEventWaterError(str(exc)) from None
    column = op.column
    ny, nx = op.shape
    try:
        ss._check_dynamic(column, "surface_m", surface_m, (ny, nx))
        ss._check_dynamic(column, "qx_m2_s", qx_m2_s, (ny, nx + 1))
        ss._check_dynamic(column, "qy_m2_s", qy_m2_s, (ny + 1, nx))
        t_soil = ss._check_state(column, soil)
        if t_soil != t:
            raise InterEventWaterError(f"soil.t_s = {t_soil!r} disagrees with t_s = {t!r}; one clock, never restamped")
        mi._check_history(op.matrix, history)
    except (ss.SubsurfaceError, mi.MatrixError) as exc:
        raise InterEventWaterError(str(exc)) from None
    if op.bed is None:
        if bed is not None:
            raise InterEventWaterError("the operator holds no native bed but a bed was given; refused")
    else:
        if type(bed) is not BedState or type(bed.voxel_column) is not VoxelColumnState:
            raise InterEventWaterError("the bed must be a genuine native Dense BedState; refused")
        if bed.water.depth_m is not surface_m:
            raise InterEventWaterError("alias contract violated: bed.water.depth_m must BE the surface array")
        if ls._bed_material_ids(bed) != op._bed_material_ids:
            raise InterEventWaterError("the bed's native material is not the operator-bound material; refused")
    h_m3, w_m3 = _fresh_inventories(prep, t, surface_m, qx_m2_s, qy_m2_s, soil, history)
    try:
        evidence = ie.quiet_evidence(op, surface_m, qx_m2_s, qy_m2_s)
    except ls.LayeredStormError as exc:
        raise InterEventWaterError(str(exc)) from None
    ok, failing = ie.eligibility(evidence, pol)
    if not ok:
        raise InterEventWaterError(
            f"the current closed boundary is not hydraulically quiet under the supplied policy ({failing} = "
            f"{evidence.get(failing)!r} against {pol.get(failing, 'n/a')!r}): the actual signed faces, outlet discharge and "
            f"depression excess decide, not the role name; refused (nothing is drained or reclassified)")
    zero = WaterWindowLedger()
    receipt = BoundaryOriginReceipt(
        _SEAL, prepared_sha256=prep.sha256, storm_operator_sha256=op.sha256, matrix_sha256=op.matrix.sha256,
        cover_sha256=prep._cover_digest, parent_token=token, parent_revision=revision, origin_s=t,
        anchor_surface_m3=float(h_m3), anchor_soil_m3=float(w_m3), eligibility=ELIGIBILITY,
        max_active_surface_depth_m=float(evidence["max_active_surface_depth_m"]), policy=pol, evidence=evidence)
    state = InterEventState(t, surface_m, qx_m2_s, qy_m2_s, soil, history, bed, zero, 0, prep.sha256, t, float(h_m3),
                            float(w_m3), zero, receipt)
    _check_state(prep, state)  # the shared metadata validation of the product (receipt core included)
    return state


# --- state checks / audit --------------------------------------------------------------------------------------------------
def _check_state(prep: InterEventWaterOperator, state: Any) -> float:
    """Array metadata, bindings, alias contract, ledger and anchors of an accepted T0 aggregate; returns `t0`."""
    if type(state) is not InterEventState:
        raise InterEventWaterError(f"state must be an InterEventState, got {type(state).__name__}")
    if state.prepared_sha256 != prep.sha256:
        raise InterEventWaterError("the accepted state is bound to a different interevent operator; refused")
    return _check_state_core(prep, t_s=state.t_s, surface_m=state.surface_m, qx_m2_s=state.qx_m2_s, qy_m2_s=state.qy_m2_s,
                             soil=state.soil, history=state.history, bed=state.bed, ledger=state.ledger,
                             generation=state.generation, prepared_sha256=state.prepared_sha256, origin_s=state.origin_s,
                             anchor_surface_m3=state.anchor_surface_m3, anchor_soil_m3=state.anchor_soil_m3,
                             anchor_ledger=state.anchor_ledger, source=state.source)


def _check_state_core(prep: InterEventWaterOperator, *, t_s: Any, surface_m: Any, qx_m2_s: Any, qy_m2_s: Any, soil: Any,
                      history: Any, bed: Any, ledger: Any, generation: Any, prepared_sha256: str, origin_s: Any,
                      anchor_surface_m3: Any, anchor_soil_m3: Any, anchor_ledger: Any, source: Any) -> float:
    """The shared validation body on explicit operands (the T0 state and the general state each pass their own fields;
    `ledger` is ALWAYS the hydrology-only `WaterWindowLedger`, whose T is exactly 0 - the general profile keeps its T account
    elsewhere). Array metadata only; returns `t0`."""
    gen = _integer(generation, "state.generation")
    _check_ledger(ledger, gen)
    _check_ledger(anchor_ledger)
    t0 = _nonneg(t_s, "state.t_s")
    origin = _nonneg(origin_s, "state.origin_s")
    _nonneg(anchor_surface_m3, "state.anchor_surface_m3")
    _nonneg(anchor_soil_m3, "state.anchor_soil_m3")
    _check_receipt_core(prep, source, prepared_sha256=prepared_sha256, origin_s=origin_s,
                        anchor_surface_m3=anchor_surface_m3, anchor_soil_m3=anchor_soil_m3, anchor_ledger=anchor_ledger,
                        t_s=t_s, ledger=ledger, generation=generation)  # scalar source authentication BEFORE any array check
    op = prep.op
    column = op.column
    ny, nx = op.shape
    try:
        ss._check_dynamic(column, "state.surface_m", surface_m, (ny, nx))
        ss._check_dynamic(column, "state.qx_m2_s", qx_m2_s, (ny, nx + 1))
        ss._check_dynamic(column, "state.qy_m2_s", qy_m2_s, (ny + 1, nx))
        t_soil = ss._check_state(column, soil)
        if t_soil != t0:
            raise InterEventWaterError(f"state.soil.t_s = {t_soil!r} disagrees with state.t_s = {t0!r}; one clock")
        mi._check_history(op.matrix, history)
    except (ss.SubsurfaceError, mi.MatrixError) as exc:
        raise InterEventWaterError(str(exc)) from None
    if op.bed is None:
        if bed is not None:
            raise InterEventWaterError("the operator holds no native bed but the state carries one; refused")
    else:
        if type(bed) is not BedState:
            raise InterEventWaterError(f"state.bed must be a genuine BedState, got {type(bed).__name__}")
        if bed.water.depth_m is not surface_m:
            raise InterEventWaterError("alias contract violated: state.bed.water.depth_m must BE state.surface_m")
        if ls._bed_material_ids(bed) != op._bed_material_ids:
            raise InterEventWaterError("the state's native bed material is not the operator-bound material; refused")
    if t0 < origin:
        raise InterEventWaterError("state clock precedes the window origin")
    for name in _CUMULATIVE:
        if getattr(ledger, name) < getattr(anchor_ledger, name):
            raise InterEventWaterError(f"cumulative ledger {name} predates its window anchor")
    steps = int(ledger.n_steps) - int(anchor_ledger.n_steps)
    elapsed = float(ledger.elapsed_s) - float(anchor_ledger.elapsed_s)
    bound_s = 8.0 * (steps + 2) * np.finfo(np.float64).eps * max(t0, origin, abs(elapsed))
    if abs(elapsed - (t0 - origin)) > bound_s:
        raise InterEventWaterError("ledger elapsed time disagrees with the window origin")
    return t0


def _window_bound(prep: InterEventWaterOperator, state: InterEventState, h: float, soil: float, rain: float,
                  export: float, drain: float, es: float, eso: float) -> float:
    return _window_bound_core(prep, state.ledger, state.anchor_ledger, state.anchor_surface_m3, state.anchor_soil_m3, h, soil,
                              rain, export, drain, es, eso)


def _window_bound_core(prep: InterEventWaterOperator, ledger: WaterWindowLedger, anchor: WaterWindowLedger,
                       anchor_surface_m3: float, anchor_soil_m3: float, h: float, soil: float, rain: float,
                       export: float, drain: float, es: float, eso: float, extra_scale: float = 0.0) -> float:
    """Anchored closure bound, MAPLE `volume_roundoff_bound_m3` only: (1) the bounds the owner published for the window's
    own storm and interevent steps (cumulative bound minus anchor); (2) rounding of NEW additions into each LARGE prior
    cumulative (rain, export, drainage, both evaporations and the cumulative bound): window steps + 1 operations at the
    scale of that cumulative; (3) the fresh physical reductions and the final algebra: 2 (L + 1) N + 14 operations at the
    largest actual inventory or window flux. `extra_scale` lets a caller with an additional window term (the general
    profile's accepted T) enter that magnitude into the scale of (3) without changing any count here."""
    steps = int(ledger.n_steps) - int(anchor.n_steps) + 1
    total = float(ledger.global_residual_bound_m3) - float(anchor.global_residual_bound_m3)
    for name in ("rain_m3", "export_m3", "drainage_m3", "surface_evaporation_m3", "soil_evaporation_m3",
                 "global_residual_bound_m3"):
        scale = max(abs(float(getattr(ledger, name))), abs(float(getattr(anchor, name))))
        total += volume_roundoff_bound_m3(steps, scale)
    fresh = 2 * (prep.n_layers + 1) * prep.n_cells + 14
    total += volume_roundoff_bound_m3(fresh, max(h, soil, float(anchor_surface_m3), float(anchor_soil_m3),
                                                 rain, export, drain, es, eso, float(extra_scale)))
    if not math.isfinite(total) or total < 0.0:
        raise InterEventWaterError("non-finite anchored closure bound")
    return total


def _fresh_inventories(prep: InterEventWaterOperator, t_s: float, surface_m: Any, qx_m2_s: Any, qy_m2_s: Any, soil: Any,
                       history: Any) -> tuple[float, float]:
    """The actual active-cell surface and soil volumes (m3) from the raw arrays plus the every-cell inventory bounds and
    the router state; shared by the T0 and the general audit. Never a saved pass bit."""
    op = prep.op
    xp, area = op.xp, op.column.cell_area_m2
    active = op.column.active
    packet = ls._packet(xp, [xp.sum(xp.where(active.reshape(op.shape), surface_m, 0.0)) * area,
                             xp.sum(xp.where(active.reshape(1, -1), soil.water_m, 0.0)) * area])
    if not np.all(np.isfinite(packet)) or np.any(packet < 0.0):
        raise InterEventWaterError("non-finite/negative physical inventory volumes")
    try:
        ls._validate_inventories(op, soil, history)
        op.router.validate_state(SurfaceState(float(t_s), surface_m, qx_m2_s, qy_m2_s))
    except (ls.LayeredStormError, ExperimentalHydrologyError) as exc:
        raise InterEventWaterError(f"accepted inventories refused: {exc}") from None
    return float(packet[0]), float(packet[1])


def audit_interevent(prep: InterEventWaterOperator, state: InterEventState) -> dict:
    """Actual inventories plus the anchored window fluxes (never a saved pass bit): the closure
    `(V - V_anchor) + export + drainage + surfaceE + soilE + T - rain = 0` over the window that began at the source storm's
    original anchor, with fresh inventories, every-cell inventory bounds and the router state."""
    _check_prepared(prep)
    _admit(prep.op, prep.memory.required_bytes, "audit")  # quota first, then the metadata-only state/source checks
    _check_state(prep, state)
    h, soil = _fresh_inventories(prep, state.t_s, state.surface_m, state.qx_m2_s, state.qy_m2_s, state.soil, state.history)
    led, anc = state.ledger, state.anchor_ledger
    rain = float(led.rain_m3) - float(anc.rain_m3)
    export = float(led.export_m3) - float(anc.export_m3)
    drain = float(led.drainage_m3) - float(anc.drainage_m3)
    es = float(led.surface_evaporation_m3) - float(anc.surface_evaporation_m3)
    eso = float(led.soil_evaporation_m3) - float(anc.soil_evaporation_m3)
    tr = float(led.transpiration_m3) - float(anc.transpiration_m3)
    residual = ((((h - float(state.anchor_surface_m3)) + (soil - float(state.anchor_soil_m3))) + export) + drain
                + es + eso + tr) - rain
    bound = _window_bound(prep, state, h, soil, rain, export, drain, es, eso)
    if not math.isfinite(residual) or abs(residual) > bound:
        raise InterEventWaterError(f"anchored window water closure failed: {residual} m3 against {bound} m3")
    return dict(surface_m3=h, soil_m3=soil, rain_m3=rain, export_m3=export, drainage_m3=drain,
                surface_evaporation_m3=es, soil_evaporation_m3=eso, transpiration_m3=tr, residual_m3=residual,
                bound_m3=bound)


# --- the step ------------------------------------------------------------------------------------------------------------------
class _PhysicalTrial(NamedTuple):
    """The raw products of one shared physical trial (referenced, not copied): the stage results and the receipt."""

    receipt: InterEventStepReceipt
    pref: Any
    matrix: Any
    evap: Any
    column: Any
    surface_step: Any
    bed: Any


def _physical_trial(prep: InterEventWaterOperator, t0: float, dt: float, t1: float, surface_m: Any, qx_m2_s: Any,
                    qy_m2_s: Any, soil: Any, history: Any, bed: Any, pet_terrain_m_s: Any) -> _PhysicalTrial:
    """The ONE existing physical stage body (preferential -> matrix(H0, zero rate) -> helper(surface + post-cap soil E) ->
    ONE column step -> return -> genuine router -> optional native `with_water` -> per-cell/stage/global checks), extracted
    unchanged from `interevent_water_step` so the T0 wrapper and the general (requested-transpiration) wrapper run the SAME
    code on their already-validated physical operands. It never reads a ledger, generation or profile and does not branch
    on T: a T debit made earlier is already in `soil`. Pure; the operands are never written."""
    op = prep.op
    column, xp = op.column, op.xp
    ny, nx = op.shape
    N, L = op.n_cells, op.n_layers
    try:
        ss._check_dynamic(column, "pet_terrain_m_s", pet_terrain_m_s, (N,))
    except ss.SubsurfaceError as exc:
        raise InterEventWaterError(str(exc)) from None
    area = column.cell_area_m2
    active, inactive = column.active, op._inactive

    h0 = surface_m.reshape(N)
    # 2. retained-pond water continues to enter the soil; no rain: the pool is exactly H0
    p = pi.preferential_step(op.preferential, h0, soil, dt)
    # 3. persistent matrix history continues with the actual remaining pool, ORIGINAL H0 and a ZERO onset rate
    m = mi.matrix_step(op.matrix, p.surface_water_m, h0, prep._zero_rate, p.state, history, dt)
    # 4. both evaporation branches from the same PET: surface on the actual remaining pool; soil pair for the column
    e = se.stage_evaporation(prep.evaporation, pet_terrain_m_s=pet_terrain_m_s, cover=prep._cover,
                             surface_water_m=m.surface_pool_m, dt_s=dt)
    # 5. ONE column step on the credited soil: zero added water, the helper's exact pair; T is exactly 0
    c = ss.step(column, m.state, dt, added_water_m=op._zero_added, **e.native_step_kwargs())
    # 6. the actual saturation return rejoins the post-E pool once; lateral stage from the ACCEPTED time t0
    post = e.surface_water_after_m + c.surface_return_m
    s = op.router.step(SurfaceState(t0, post.reshape(ny, nx), qx_m2_s, qy_m2_s), dt)
    if s.state.t_s != t1 or c.state.t_s != t1:
        raise InterEventWaterError(f"clock mismatch: surface end {s.state.t_s!r}, soil end {c.state.t_s!r}, required "
                                   f"{t1!r}; no state is restamped")
    del post
    # 7. genuine native water wrapper (water only)
    bed1 = None if op.bed is None else with_water(bed, op.context, s.state.depth_m)

    from maple.core.backend import errstate

    h1 = s.state.depth_m.reshape(N)
    pref_tot, mat_in = p.intake_total_m, m.intake_m
    ret, drain = c.surface_return_m, c.drainage_m
    es_arr, eso_arr = e.surface_evaporation_m, c.evaporation_m
    unmet_s, unmet_w = e.surface_evaporation_unmet_m, c.evaporation_unmet_m
    pool = m.surface_pool_m
    vx, vy = s.face_volume_m3["x"], s.face_volume_m3["y"]
    soil0, soil1 = soil.water_m, c.state.water_m
    flags = xp.zeros(N, dtype=np.int64)

    def flag(mask: Any, bit: int) -> None:
        xp.bitwise_or(flags, xp.left_shift(mask.astype(np.int64), bit), out=flags)

    with errstate(xp=xp, all="ignore"):
        flag(inactive & (h1 != h0), 0)
        w0 = xp.zeros(N, dtype=np.float64)
        w1 = xp.zeros(N, dtype=np.float64)
        for layer in range(L):
            flag(inactive & (soil1[layer] != soil0[layer]), 1)
            xp.add(w0, soil0[layer], out=w0)
            xp.add(w1, soil1[layer], out=w1)
        flag(inactive & ((m.history.wetting_m != history.wetting_m)
                         | (m.history.cumulative_intake_m != history.cumulative_intake_m)), 2)
        lat = ((vx[:, :-1] - vx[:, 1:]) + (vy[:-1, :] - vy[1:, :])).reshape(N)
        gross = ((xp.abs(vx[:, :-1]) + xp.abs(vx[:, 1:])) + (xp.abs(vy[:-1, :]) + xp.abs(vy[1:, :]))).reshape(N)
        vert = ret - pref_tot
        xp.subtract(vert, mat_in, out=vert)
        xp.subtract(vert, es_arr, out=vert)  # return - pref - matrix - Esurface
        res_s = h1 - h0
        xp.multiply(res_s, area, out=res_s)
        xp.multiply(vert, area, out=vert)
        xp.subtract(res_s, vert, out=res_s)
        xp.subtract(res_s, lat, out=res_s)
        xp.abs(res_s, out=res_s)
        scale_s = h0 + h1
        xp.add(scale_s, pref_tot, out=scale_s)
        xp.add(scale_s, mat_in, out=scale_s)
        xp.add(scale_s, ret, out=scale_s)
        xp.add(scale_s, es_arr, out=scale_s)
        xp.multiply(scale_s, area, out=scale_s)
        xp.add(scale_s, gross, out=scale_s)
        xp.multiply(scale_s, prep.surface_cell_coefficient, out=scale_s)
        flag(active & (res_s > scale_s), 3)
        flux_w = pref_tot + mat_in
        xp.subtract(flux_w, ret, out=flux_w)
        xp.subtract(flux_w, drain, out=flux_w)
        xp.subtract(flux_w, eso_arr, out=flux_w)
        res_w = w1 - w0
        xp.subtract(res_w, flux_w, out=res_w)
        xp.multiply(res_w, area, out=res_w)
        xp.abs(res_w, out=res_w)
        scale_w = w0 + w1
        xp.add(scale_w, pref_tot, out=scale_w)
        xp.add(scale_w, mat_in, out=scale_w)
        xp.add(scale_w, ret, out=scale_w)
        xp.add(scale_w, drain, out=scale_w)
        xp.add(scale_w, eso_arr, out=scale_w)
        xp.multiply(scale_w, area * prep.soil_cell_coefficient, out=scale_w)
        flag(active & (res_w > scale_w), 4)
        flag(active & (~xp.isfinite(res_s) | ~xp.isfinite(res_w) | ~xp.isfinite(scale_s) | ~xp.isfinite(scale_w)), 5)
        flag(inactive & ((es_arr != 0.0) | (eso_arr != 0.0) | (unmet_s != 0.0) | (unmet_w != 0.0)), 6)
        flag(~xp.isfinite(es_arr) | (es_arr < 0.0) | (es_arr > pool), 7)
        flag(~xp.isfinite(eso_arr) | (eso_arr < 0.0), 8)
        del vert, flux_w, lat, w0, w1, scale_s, scale_w
        low = xp.negative(flags)
        xp.bitwise_and(flags, low, out=low)
        low_row = xp.where(flags == 0, NO_FLAG, low.astype(np.float64))
        del low
        values = [xp.sum(xp.where(active, h0, 0.0)), xp.sum(xp.where(active, h1, 0.0)),
                  xp.max(xp.where(active, res_s, 0.0)), xp.max(xp.where(active, res_w, 0.0)),
                  xp.sum(xp.where(active, es_arr, 0.0)), xp.sum(xp.where(active, eso_arr, 0.0)),
                  xp.min(low_row), xp.count_nonzero(flags != 0), s.export_m3, s.storage_change_m3, s.budget_residual_m3,
                  s.max_cfl, s.limited_cells, s.limited_volume_m3, s.max_cell_balance_residual_m, s.outlet_discharge_m3_s,
                  xp.sum(xp.abs(vx)) + xp.sum(xp.abs(vy))]
        packet = ls._packet(xp, values)  # the ONE owner host read (synchronising on cuda)
        del values, low_row, res_s, res_w, gross, flags
    if packet.shape != (OWNER_PACKET_LENGTH,):
        raise InterEventWaterError(f"internal: owner packet shape {packet.shape}")
    low_bit, n_flagged = float(packet[6]), int(packet[7])
    if not math.isfinite(low_bit) or low_bit < NO_FLAG:
        bit = int(low_bit).bit_length() - 1 if math.isfinite(low_bit) and low_bit >= 1.0 else None
        message = FLAG_MESSAGES[bit] if bit is not None and 0 <= bit < N_FLAGS else "owner packet is corrupt"
        raise InterEventStepError(f"{message} ({n_flagged} cell(s) flagged)", bit=bit, n_flagged=n_flagged)
    h0_m3, h1_m3 = float(packet[0]) * area, float(packet[1]) * area
    max_res_s, max_res_w = float(packet[2]), float(packet[3])
    es_sum, eso_sum = float(packet[4]) * area, float(packet[5]) * area
    export, storage, router_residual, max_cfl = (float(packet[i]) for i in range(8, 12))
    limited_cells, limited_volume, max_cell_balance, outlet_q, gross_total = (int(packet[12]), float(packet[13]),
                                                                              float(packet[14]), float(packet[15]),
                                                                              float(packet[16]))
    es_m3, eso_m3 = e.ledger.evaporation_m3, c.budget.evaporation_m3
    soil0_m3, soil1_m3 = p.budget.soil_before_m3, c.budget.storage_after_m3
    drain_m3 = c.budget.drainage_m3
    # stage handoffs: each stage received the water the previous stage returned (bitwise, else within the stage bounds)
    handoff_exact = (m.budget.pool_before_m3 == p.budget.surface_after_m3 and m.budget.soil_before_m3 == p.budget.soil_after_m3
                     and c.budget.storage_before_m3 == m.budget.soil_after_m3
                     and e.ledger.storage_before_m3 == m.budget.pool_after_m3)
    if (abs(m.budget.pool_before_m3 - p.budget.surface_after_m3) > p.budget.bound_m3 + m.budget.bound_m3
            or abs(m.budget.soil_before_m3 - p.budget.soil_after_m3) > p.budget.bound_m3 + m.budget.bound_m3
            or abs(c.budget.storage_before_m3 - m.budget.soil_after_m3) > m.budget.bound_m3 + c.budget.bound_m3
            or abs(e.ledger.storage_before_m3 - m.budget.pool_after_m3) > m.budget.bound_m3 + e.ledger.bound_m3
            or abs(p.budget.surface_before_m3 - h0_m3) > p.budget.bound_m3 + volume_roundoff_bound_m3(N + 1, h0_m3)):
        raise InterEventStepError("stage handoff mismatch: a stage received other water than the previous stage returned",
                                  bit=None, n_flagged=n_flagged)
    if (abs(es_sum - es_m3) > e.ledger.bound_m3 + volume_roundoff_bound_m3(N + 1, es_m3)
            or abs(eso_sum - eso_m3) > c.budget.bound_m3 + volume_roundoff_bound_m3(N + 1, eso_m3)):
        raise InterEventStepError("the per-cell evaporation arrays disagree with the stage evaporation totals",
                                  bit=None, n_flagged=n_flagged)
    residual = ((((h1_m3 - h0_m3) + (soil1_m3 - soil0_m3)) + export) + drain_m3 + es_m3) + eso_m3
    operands = (abs(h0_m3), abs(h1_m3), abs(soil0_m3), abs(soil1_m3), abs(export), abs(drain_m3), abs(es_m3), abs(eso_m3))
    if not all(math.isfinite(v) for v in (*operands, storage, router_residual, max_res_s, max_res_w, residual)):
        raise InterEventStepError("non-finite composed water budget", bit=None, n_flagged=n_flagged)
    owner_terms = OWNER_GLOBAL_TERMS_PER_CELL * N + OWNER_GLOBAL_TERMS
    owner_bound = volume_roundoff_bound_m3(owner_terms, max(operands))
    bound = (abs(router_residual) + 2.0 * p.budget.bound_m3 + 2.0 * m.budget.bound_m3 + 2.0 * e.ledger.bound_m3
             + c.budget.bound_m3) + owner_bound
    if abs(residual) > bound:
        raise InterEventStepError(
            f"composed global water closure violated: residual {residual!r} m3 against the derived bound {bound!r} m3 "
            f"(router residual {router_residual!r}, stage bounds {p.budget.bound_m3!r}/{m.budget.bound_m3!r}/"
            f"{e.ledger.bound_m3!r}/{c.budget.bound_m3!r}, owner {owner_bound!r} for {owner_terms} terms)",
            bit=None, n_flagged=n_flagged)
    receipt = InterEventStepReceipt(
        dt_s=dt, t0_s=t0, t1_s=t1, preferential=p.budget, matrix=m.budget, evaporation=e.ledger, column=c.budget,
        surface_evaporation_m3=es_m3, soil_evaporation_m3=eso_m3, surface_before_m3=h0_m3, surface_after_m3=h1_m3,
        soil_before_m3=soil0_m3, soil_after_m3=soil1_m3, export_m3=export, storage_change_m3=storage,
        router_residual_m3=router_residual, max_cfl=max_cfl, limited_cells=limited_cells, limited_volume_m3=limited_volume,
        max_cell_balance_residual_m=max_cell_balance, outlet_discharge_m3_s=outlet_q, gross_face_volume_m3=gross_total,
        global_residual_m3=residual, global_bound_m3=bound, owner_bound_m3=owner_bound, owner_global_terms=owner_terms,
        max_surface_cell_residual_m3=max_res_s, max_soil_cell_residual_m3=max_res_w,
        surface_cell_coefficient=prep.surface_cell_coefficient, soil_cell_coefficient=prep.soil_cell_coefficient,
        surface_cell_terms=surface_cell_terms(L), soil_cell_terms=soil_cell_terms(L), handoff_exact=handoff_exact,
        n_flagged=n_flagged)
    return _PhysicalTrial(receipt, p, m, e, c, s, bed1)


def interevent_water_step(prep: InterEventWaterOperator, accepted: InterEventState, pet_terrain_m_s: Any, dt_s: Any, *,
                          phase: str) -> InterEventCandidate:
    """ONE pure trial from `accepted` over `dt_s > 0` with the terrain-applied potential evapotranspiration rate
    `pet_terrain_m_s` `(N,)` of the namespace (resident float64, m/s, finite, >= 0; aspect and cos(slope) are ALREADY in it
    and are not applied again) and ZERO rain. `phase` must be exactly "interevent" (wind, rain, recession or anything else is
    refused: no E/T during those, and nothing is caught up). Returns a candidate or raises before any result exists:
    `HydraulicStepRejected` (late CFL, after the intake/E/drainage trials) means retry the SAME accepted record at a
    smaller dt (everything reruns from the unchanged accepted state; no retry happens here); every other error is a
    nonrecoverable refusal. The accepted record, its arrays, history, native bed, ledger and the PET are never written."""
    _check_prepared(prep)
    if phase != INTEREVENT or type(phase) is not str:
        raise InterEventWaterError(f"phase must be exactly {INTEREVENT!r}; wind/rain/recession spend no E/T demand "
                                   f"(got {phase!r})")
    _admit(prep.op, prep.memory.required_bytes, "step")  # quota/preflight first: BEFORE any state or source check or trial
    t0 = _check_state(prep, accepted)
    dt = _real(dt_s, "dt_s")
    if not dt > 0.0:
        raise InterEventWaterError(f"dt_s must be > 0 (the interevent step has no dt = 0 identity), got {dt!r}")
    try:
        t1 = advance_time(t0, dt)
    except ExperimentalHydrologyError as exc:
        raise InterEventWaterError(str(exc)) from None
    trial = _physical_trial(prep, t0, dt, t1, accepted.surface_m, accepted.qx_m2_s, accepted.qy_m2_s, accepted.soil,
                            accepted.history, accepted.bed, pet_terrain_m_s)
    receipt, p, m, e, c, s, bed1 = trial
    state = InterEventState(t1, s.state.depth_m, s.state.qx_m2_s, s.state.qy_m2_s, c.state, m.history, bed1,
                            accepted.ledger.advanced(receipt), int(accepted.generation) + 1, prep.sha256,
                            accepted.origin_s, accepted.anchor_surface_m3, accepted.anchor_soil_m3, accepted.anchor_ledger,
                            accepted.source)
    return InterEventCandidate(state=state, receipt=receipt, dt_s=dt, surface_step=s,
                               vertical=InterEventVertical(p.intake_m, m.intake_m, e.surface_evaporation_m,
                                                           c.evaporation_m, c.surface_return_m, c.drainage_m))


# --- provenance ------------------------------------------------------------------------------------------------------------
def describe(prep: InterEventWaterOperator) -> dict[str, Any]:
    _check_prepared(prep)
    op = prep.op
    return {
        "operation": "pure fixed-bed, fixed-property, fixed-trait, water-only interevent step (zero rain, T0)",
        "backend": prep.backend, "namespace": prep.xp.__name__, "shape": list(op.shape), "n_layers": prep.n_layers,
        "n_cells": prep.n_cells, "storm_operator_sha256": op.sha256, "cover_sha256": prep._cover_digest,
        "partition": prep.evaporation.partition.summary(),
        "stage_order": ["validation_and_admission", "preferential_on_H0", "matrix_remaining_pool_zero_rate",
                        "stage_evaporation_both_branches", "column_zero_added_helper_pair", "surface_return_rejoin",
                        "lateral_routing_at_t0", "native_water_wrapper", "owner_packet"],
        "phase": INTEREVENT, "transpiration": "T = 0: no implicit root withdrawal, no LEAF catch-up",
        "ledger": "WaterWindowLedger: surface E, soil E and accepted T separate; window difference from the source "
                  "storm's original anchors; not a full-run authority",
        "closure": {"per_cell_surface_terms": surface_cell_terms(prep.n_layers),
                    "per_cell_soil_terms": soil_cell_terms(prep.n_layers),
                    "owner_global_terms": f"{OWNER_GLOBAL_TERMS_PER_CELL} N + {OWNER_GLOBAL_TERMS}",
                    "global": "|router residual| + 2 pref + 2 matrix + 2 evaporation + column + owner bound",
                    "policy": "conservation.volume_roundoff_bound_m3 only; no empirical epsilon"},
        "recoverable": "HydraulicStepRejected only: retry the SAME accepted record at a smaller dt",
        "numerical_departure": "gravity -> E -> exact diffusion in one column call, then routing: first-order split; "
                               "not the EcoHyD runoff -> diffusion -> daily ET order; refinement is the caller's to test",
        "memory": prep.memory.as_dict(),
        "identity": {"sha256": prep.sha256},
        "excluded": "calendar/weather/PET driver, LEAF requests/growth/transpiration, evolving cover or properties, "
                    "wetting reset/recovery, checkpoint/codec/CAS, sediment, reroute, wind, Windowed admission, "
                    "full-catchment or multi-day claims",
    }


# --- checkpoint restoration seam (owner-controlled; used only by the verified lifecycle checkpoint codec) ------------------------
_LEDGER_RECORD_FIELDS = tuple(f for f in ("n_steps", "elapsed_s", "rain_m3", "preferential_intake_m3", "matrix_intake_m3",
                                          "surface_return_m3", "drainage_m3", "export_m3", "evaporation_m3",
                                          "global_residual_bound_m3", "max_abs_global_residual_m3"))
_RECEIPT_RECORD_KEYS = frozenset({*_RECEIPT_SCALARS, "anchor_ledger", "end_ledger", "digest"})
_RECEIPT_COUNTS = frozenset({"start_generation", "end_generation", "accepted_steps"})
_RECEIPT_SHAS = ("event_sha256", "storm_operator_sha256", "prepared_sha256")


def checkpoint_receipt_record(receipt: StormEndReceipt) -> dict[str, Any]:
    """The bounded scalar record of a GENUINE receipt in the checkpoint vocabulary (an encoded tree). Refuses a non-minted
    object: a dictionary is never a receipt."""
    from maple_syrup import checkpoint_capability as cc

    if type(receipt) is BoundaryOriginReceipt and receipt in _MINTED_BOUNDARY:
        # The lifecycle codec (EXACT C1) is qualified for a genuine old StormEndReceipt window only. A genuine boundary-origin
        # window is refused with the SHARED typed checkpoint error (the codec's save wrapper re-raises it as its own typed
        # error) BEFORE any writer, directory or native validator output exists; no schema/profile is extended.
        from maple_syrup import layered_checkpoint as lcp

        raise lcp.LayeredCheckpointError(
            "a boundary-origin interevent window is not covered by the qualified lifecycle checkpoint codec (source/schema/origin "
            "scoped to a genuine StormEndReceipt); a versioned boundary-origin codec extension is a separate qualification; refused")
    if type(receipt) is not StormEndReceipt or receipt not in _MINTED:
        raise InterEventWaterError("only a StormEndReceipt minted by enter_interevent() can be recorded; refused")
    return cc.encode_tree(receipt.as_dict())


def _storm_ledger_from_record(record: Any, name: str) -> ls.LayeredWaterLedger:
    if type(record) is not dict or set(record) != set(_LEDGER_RECORD_FIELDS):
        raise InterEventWaterError(f"{name} ledger record keys mismatch")
    kwargs = {}
    for key in _LEDGER_RECORD_FIELDS:
        value = record[key]
        if key == "n_steps":
            if type(value) is not int:
                raise InterEventWaterError(f"{name}.{key} must be an exact integer")
        elif type(value) is not float:
            raise InterEventWaterError(f"{name}.{key} must be an exact float")
        kwargs[key] = value
    return ls.LayeredWaterLedger(**kwargs)


def _restore_checkpoint_source_receipt(prep: InterEventWaterOperator, cap: Any, record: Any, *, t_s: float, ledger: Any,
                                       generation: int, origin_s: float, anchor_surface_m3: float, anchor_soil_m3: float,
                                       anchor_ledger: Any) -> StormEndReceipt:
    """Owner-controlled second origin of a genuine `StormEndReceipt` (the first is `enter_interevent`): it is reached ONLY with
    a live verified-checkpoint capability bound to exactly this encoded record. The saved bounded scalars are typed, the real
    receipt is minted here with this module's own seal, its recomputed digest must equal the saved one, and the existing
    `_check_receipt_core` then authenticates it against the prepared binding and the CURRENT window ledger/anchors. No storm
    event or array is faked. A matching checksum shows consistency with the checkpoint, not that the past storm was executed."""
    from maple_syrup import checkpoint_capability as cc

    _check_prepared(prep)
    try:
        cc.require(cap, "receipt", record, (), lop_sha256=None)
        tree = cc.decode_tree(record)
    except cc.CheckpointCapabilityError as exc:
        raise InterEventWaterError(f"checkpoint receipt refused: {exc}") from None
    if type(tree) is not dict or set(tree) != _RECEIPT_RECORD_KEYS:
        raise InterEventWaterError("checkpoint receipt record keys mismatch (closed key set)")
    fields: dict[str, Any] = {}
    for key in _RECEIPT_SCALARS:
        value = tree[key]
        if key in _RECEIPT_SHAS:
            fields[key] = _sha(value, "receipt." + key)
        elif key in _RECEIPT_COUNTS:
            if type(value) is not int:
                raise InterEventWaterError(f"receipt.{key} must be an exact integer")
            fields[key] = _integer(value, "receipt." + key)
        else:
            if type(value) is not float:
                raise InterEventWaterError(f"receipt.{key} must be an exact float")
            fields[key] = _nonneg(value, "receipt." + key)
    fields["anchor_ledger"] = _storm_ledger_from_record(tree["anchor_ledger"], "receipt.anchor_ledger")
    fields["end_ledger"] = _storm_ledger_from_record(tree["end_ledger"], "receipt.end_ledger")
    receipt = StormEndReceipt(_SEAL, **fields)
    if receipt.digest != _sha(tree["digest"], "receipt.digest"):
        raise InterEventWaterError("the saved receipt digest differs from its recomputed digest; refused")
    _check_receipt_core(prep, receipt, prepared_sha256=prep.sha256, origin_s=origin_s, anchor_surface_m3=anchor_surface_m3,
                        anchor_soil_m3=anchor_soil_m3, anchor_ledger=anchor_ledger, t_s=t_s, ledger=ledger,
                        generation=generation)
    return receipt


# --- versioned (v2) source-envelope seam: STORM_END or BOUNDARY_ORIGIN (task phase1_boundary_origin_checkpoint_impl_20261110) ------
# The v1 functions above (`checkpoint_receipt_record`, `_restore_checkpoint_source_receipt`) are unchanged; a genuine boundary-origin
# window is still refused by the former. The v2 codec saves one closed ENVELOPE `{"source_kind", "body"}` and restores through the
# functions below, reached only with a live verified-checkpoint capability.
_SOURCE_ENVELOPE_KEYS = frozenset({"source_kind", "body"})
_BOUNDARY_BODY_KEYS = frozenset({*_BOUNDARY_SCALARS, "policy", "evidence", "digest"})
_BOUNDARY_SHAS = ("prepared_sha256", "storm_operator_sha256", "matrix_sha256", "cover_sha256")
_BOUNDARY_FLOATS = ("origin_s", "anchor_surface_m3", "anchor_soil_m3", "max_active_surface_depth_m")


def checkpoint_source_record_v2(receipt: Any) -> dict[str, Any]:
    """The closed encoded envelope of a GENUINE (registry-authenticated) interevent source of either kind. The storm body is exactly
    the v1 record; the boundary body is the receipt's `as_dict()` without its duplicate `source_kind` (the envelope is the only
    discriminator). A dictionary or any other object is refused."""
    from maple_syrup import checkpoint_capability as cc

    kind = source_kind(receipt)  # raises for anything not minted by this owner
    if kind == STORM_END:
        return {"source_kind": kind, "body": checkpoint_receipt_record(receipt)}
    body = receipt.as_dict()
    del body["source_kind"]
    return {"source_kind": kind, "body": cc.encode_tree(body)}


def _restore_checkpoint_boundary_receipt(prep: InterEventWaterOperator, tree: Any, *, t_s: float, ledger: Any, generation: int,
                                         origin_s: float, anchor_surface_m3: float, anchor_soil_m3: float,
                                         anchor_ledger: Any) -> BoundaryOriginReceipt:
    """Owner-controlled mint of a genuine `BoundaryOriginReceipt` from its ONCE-decoded closed body: exact builtin types, this module's
    own seal and registry, recomputed digest equal to the saved one, then the unchanged `_check_boundary_receipt_core` against the
    CURRENT window scalars. No storm, event, ledger history or array is fabricated."""
    if type(tree) is not dict or set(tree) != _BOUNDARY_BODY_KEYS:
        raise InterEventWaterError("checkpoint boundary-origin body keys mismatch (closed key set)")
    fields: dict[str, Any] = {}
    for key in _BOUNDARY_SHAS:
        fields[key] = _sha(tree[key], "receipt." + key)
    fields["parent_token"] = _token(tree["parent_token"], "receipt.parent_token")
    if type(tree["parent_revision"]) is not int:
        raise InterEventWaterError("receipt.parent_revision must be an exact integer")
    fields["parent_revision"] = _integer(tree["parent_revision"], "receipt.parent_revision")
    if fields["parent_revision"] > 2 ** 63 - 1:  # closed width: the manifest allowance is derived from this bound
        raise InterEventWaterError("receipt.parent_revision exceeds its closed 63-bit bound")
    for key in _BOUNDARY_FLOATS:
        if type(tree[key]) is not float:
            raise InterEventWaterError(f"receipt.{key} must be an exact float")
        fields[key] = _nonneg(tree[key], "receipt." + key)
    if type(tree["eligibility"]) is not str or tree["eligibility"] != ELIGIBILITY:
        raise InterEventWaterError("the boundary source carries an unknown eligibility statement; refused")
    fields["eligibility"] = tree["eligibility"]
    for key, names in (("policy", ie.POLICY_NAMES), ("evidence", ie.EVIDENCE_NAMES)):
        rec = tree[key]
        if type(rec) is not dict or set(rec) != set(names):
            raise InterEventWaterError(f"receipt.{key} keys mismatch (closed key set)")
        if any(type(rec[n]) is not float for n in names):
            raise InterEventWaterError(f"receipt.{key} values must be exact floats")
        fields[key] = {n: rec[n] for n in names}
    receipt = BoundaryOriginReceipt(_SEAL, **fields)
    if receipt.digest != _sha(tree["digest"], "receipt.digest"):
        raise InterEventWaterError("the saved boundary receipt digest differs from its recomputed digest; refused")
    _check_boundary_receipt_core(prep, receipt, prepared_sha256=prep.sha256, origin_s=origin_s,
                                 anchor_surface_m3=anchor_surface_m3, anchor_soil_m3=anchor_soil_m3, anchor_ledger=anchor_ledger,
                                 t_s=t_s, ledger=ledger, generation=generation)
    return receipt


def _restore_checkpoint_source_v2(prep: InterEventWaterOperator, cap: Any, record: Any, *, t_s: float, ledger: Any, generation: int,
                                  origin_s: float, anchor_surface_m3: float, anchor_soil_m3: float, anchor_ledger: Any) -> Any:
    """Owner-controlled v2 second origin of the CURRENT interevent source: validates the RAW encoded envelope (closed keys, closed kind,
    capability digest of the saved text), then dispatches through a two-entry literal table. The storm-end body is handed, still
    encoded, to the unchanged v1 restorer (which decodes it exactly once under capability part `receipt`); the boundary body is decoded
    exactly once here. A matching checksum shows consistency, not that a past window was executed."""
    from maple_syrup import checkpoint_capability as cc

    _check_prepared(prep)
    if type(record) is not dict or set(record) != _SOURCE_ENVELOPE_KEYS:
        raise InterEventWaterError("checkpoint source envelope keys mismatch (closed key set)")
    kind, body = record["source_kind"], record["body"]
    if type(kind) is not str or kind not in (STORM_END, BOUNDARY_ORIGIN):
        raise InterEventWaterError(f"unknown checkpoint source kind {kind!r}; refused")
    try:
        cc.require(cap, "source_envelope", record, (), lop_sha256=None)
    except cc.CheckpointCapabilityError as exc:
        raise InterEventWaterError(f"checkpoint source refused: {exc}") from None
    common = dict(t_s=t_s, ledger=ledger, generation=generation, origin_s=origin_s, anchor_surface_m3=anchor_surface_m3,
                  anchor_soil_m3=anchor_soil_m3, anchor_ledger=anchor_ledger)
    if kind == STORM_END:
        return _restore_checkpoint_source_receipt(prep, cap, body, **common)
    try:
        tree = cc.decode_tree(body)
    except cc.CheckpointCapabilityError as exc:
        raise InterEventWaterError(f"checkpoint boundary-origin body refused: {exc}") from None
    return _restore_checkpoint_boundary_receipt(prep, tree, **common)
