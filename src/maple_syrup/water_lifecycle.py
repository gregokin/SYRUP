"""Continuous fixed-property water LIFECYCLE owner (Phase 1, task phase1_water_lifecycle_20261110).

SYRUP owns all physical water. This module adds ONE typed lifetime-accounting owner above the EXISTING, unchanged, qualified
storm / interevent / requested-transpiration components so that a run can go

    storm  ->  interevent E (real hydrology) + zero-time T  ->  next storm  ->  ...

with the soil water, retained surface water, signed face discharges, wetting history J and lifetime matrix intake carried
exactly across every boundary. It is a CPU (NumPy / Numba) component for a genuine native MAPLE39 DENSE bed or no bed, with fixed
bed, layers, properties, maps, material and prescribed cover. Nothing in the existing runtime is changed; see
docs/subsurface_hydrology/water_lifecycle.md and the task design.md for the contracts, the fold rule and the derived bound.

    lop  = prepare_lifecycle(column, pref, matrix, router, graph, topology, evaporation, cover, ...)
    s0   = start_lifecycle(lop, bed=bed, soil=soil, history=history, t_s=0.0)              # closed 'boundary'
    s1   = begin_storm(lop, s0, expected_parent=expect_parent(s0), schedule=..., field=...)  # 'storm'
    s2   = evolve_storm(lop, s1, expected_parent=expect_parent(s1), until_s=...).state       # until complete
    s3   = enter_interevent_phase(lop, s2, expected_parent=expect_parent(s2))              # 'interevent' (E + T allowed)
    s4   = evolve_interevent(lop, s3, ..., pet_terrain_m_s=..., until_s=...).state
    s5,fb= request_withdrawal(lop, s4, ..., depth_m=..., identity="...")                    # zero-time T
    s6   = close_interevent(lop, s5, expected_parent=expect_parent(s5))                    # closed 'boundary'
    s7   = reset_controlled_wetting_history(lop, s6, ...)                                  # OPTIONAL explicit test control
    s8   = begin_storm(lop, s6 or s7, ...)                                                 # next storm: carry by default

Authority. The lifecycle state holds exactly ONE current physical role (a closed boundary payload, a genuine storm event state
or a genuine requested-transpiration interevent state) plus bounded scalars: it is an accounting envelope, never a second
water inventory. Every function is PURE: it returns a NEW state or raises, never writing an accepted array, clock, ledger,
counter, request sequence or source. There is no global registry or compare-and-swap; the caller adopts or drops a returned
state, and `expect_parent` (a non-owning weak reference) distinguishes real physical branches that share every scalar.

Out of scope and refused before any trial: Windowed or foreign storage, CUDA, a disk lifecycle codec, the LEAF model/growth,
the MASY calendar/weather, wind, sediment/erosion, changed cover/material/property maps, automatic Green-Ampt recovery and any
claim of general repeated-storm physical validation. Nothing here was run on a GPU.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import json
import math
import sys
import weakref
from dataclasses import asdict, dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
from maple.core.types.voxel import VoxelColumnState

from maple_syrup import interevent_transpiration as it
from maple_syrup import interevent_water as iw
from maple_syrup import layered_event as ev
from maple_syrup import layered_storm as ls
from maple_syrup import matrix_infiltration as mi
from maple_syrup import preferential_infiltration as pi
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st
from maple_syrup import water_lifecycle_cuda as wlc
from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.pond_network import PondTopology
from maple_syrup.rainfall import RainfallSchedule
from maple_syrup.routing import RoutingGraph
from maple_syrup import sediment_bed as _sb
from maple_syrup.sediment_bed import BedContext, BedState
from maple_syrup.surface_routing import ExperimentalHydrologyError, HydraulicStepRejected, SurfaceState

__all__ = [
    "BoundaryPayload",
    "CUDA9F_BOUNDARY_PROFILE",
    "CUDA9F_PROFILE",
    "CUDA_BOUNDARY_PROFILE",
    "CUDA_PROFILE",
    "ExpectedParent",
    "LifecycleError",
    "LifecycleMemory",
    "LifecycleOperator",
    "LifecycleResetRefused",
    "LifecycleState",
    "LifecycleStormResult",
    "LifecycleIntereventResult",
    "LifetimeLedger",
    "PhaseReceipt",
    "ResetReceipt",
    "Watermark",
    "audit_lifecycle",
    "begin_storm",
    "close_interevent",
    "enter_interevent_from_boundary",
    "enter_interevent_phase",
    "evolve_interevent",
    "evolve_storm",
    "expect_parent",
    "lifecycle_metadata",
    "prepare_lifecycle",
    "request_withdrawal",
    "required_audit_bytes",
    "required_begin_storm_bytes",
    "required_close_bytes",
    "required_evolve_interevent_bytes",
    "required_evolve_storm_bytes",
    "required_request_bytes",
    "required_enter_from_boundary_bytes",
    "required_enter_interevent_bytes",
    "reset_controlled_wetting_history",
    "start_lifecycle",
]

PROFILE = "maple_syrup.water_lifecycle/fixed_property_dense_or_bedless/1"
CUDA_PROFILE = wlc.CUDA_PROFILE  # the separate explicit UNQUALIFIED CUDA opt-in literal (never PROFILE; BACKENDS stays CPU-only)
CUDA_BOUNDARY_PROFILE = wlc.CUDA_BOUNDARY_PROFILE  # additionally enables genuine boundary-origin interevent entry and the controlled J reset on CUDA
BACKENDS = ("numpy", "numba")
CUDA9F_PROFILE = wlc.CUDA9F_PROFILE  # explicit UNQUALIFIED native9f storm-first CUDA opt-in (distinct literal; verified through the dense-CUDA source helper)
CUDA9F_BOUNDARY_PROFILE = wlc.CUDA9F_BOUNDARY_PROFILE  # native9f counterpart that additionally enables boundary-origin interevent entry and the controlled J reset

OBJECT_BYTES = 4096
_F8 = 8
_SEAL = object()
_MINTED = weakref.WeakSet()
_MINTED_EXPECTED = weakref.WeakSet()
_MINTED_PAYLOADS = weakref.WeakSet()
MAX_IDENTITY_CHARS = 128
#: Reset quiet-evidence scratch: the excess/masked temporaries are about a dozen (N,) doubles plus two face-absolute copies.
RESET_SCRATCH_DOUBLES_PER_CELL = 12

MEMORY_SCOPE = (
    "Lifecycle ledger under the EXISTING column convention: every transition is ADMITTED against column.available_bytes_fn() "
    "BEFORE any new operator, canonical array, hash or copy. Live components counted: the accepted physical arrays (surface, "
    "faces, soil, J, lifetime intake, native mobile), the fixed components (column/preferential/matrix/router statics, owned "
    "cover and transpiration bounds), the CURRENT phase binding, the per-transition extras (fresh storm-operator statics and "
    "event maps, interevent owners, the lifetime audit scratch, the reset J copy and quiet scratch) and the caller's explicit "
    "extra/native/feedback reserves. With a native Dense bed the storm operator's retained reference depth is charged 8 N ONCE "
    "and, because every episode's operator is prepared on the CURRENT bed, it IS the current surface depth at handoff (no stale "
    "original). Bedless handoff copies the surface once (admitted). No old phase operator, state, bed or array-bearing receipt "
    "is retained. EXCLUDED: allocator pools, Numba JIT workspaces, native voxel/ledger material beyond the declared native "
    "reserve, checkpoint staging, LEAF, MASY and the process; a reading is not a reservation and no whole-process fit is claimed."
)


class LifecycleError(ValueError):
    """Invalid lifecycle operation, binding, state, counter or unsupported profile. Raised before any result exists."""


class LifecycleResetRefused(LifecycleError):
    """The explicit controlled history reset is not eligible at this boundary (typed, pure)."""


# --- scalars -----------------------------------------------------------------------------------------------------------
def _real(value: Any, name: str, *, nonneg: bool = False, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise LifecycleError(f"{name} must be a real host number, got {type(value).__name__}")
    try:
        out = float(value)
    except OverflowError:
        raise LifecycleError(f"{name} is too large for a finite float64") from None
    if not math.isfinite(out):
        raise LifecycleError(f"{name} must be finite, got {value!r}")
    if positive and not out > 0.0:
        raise LifecycleError(f"{name} must be > 0, got {value!r}")
    if nonneg and out < 0.0:
        raise LifecycleError(f"{name} must be >= 0, got {value!r}")
    return out


def _freeze(obj: Any) -> Any:
    """Deep read-only view of a plain-dict tree (a detached copy: the source dict stays the caller's)."""
    if isinstance(obj, (dict, MappingProxyType)):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    return obj


def _thaw(obj: Any) -> Any:
    """Fresh plain, JSON-ready dict tree from a frozen one (serialization views never alias the immutable metadata)."""
    if isinstance(obj, (dict, MappingProxyType)):
        return {k: _thaw(v) for k, v in obj.items()}
    return obj


MAX_COUNT = 2 ** 53  # host count controls: exact in float64 and far beyond any bounded work/byte figure


def _count(value: Any, name: str) -> int:
    """A non-Boolean host integer in [0, 2**53]. The rejected value is NEVER formatted (a giant int would raise Python's bare
    4300-digit ValueError from its repr); the bound is declared instead. 0 is legitimate (e.g. a zero-step work stop)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise LifecycleError(f"{name} must be a non-Boolean integer in [0, 2**53] (got type {type(value).__name__})")
    if value < 0 or value > MAX_COUNT:
        raise LifecycleError(f"{name} must be a non-Boolean integer in [0, 2**53] (the value is outside that range)")
    return int(value)


def _ident(value: Any, name: str) -> str:
    if type(value) is not str or not value or len(value) > MAX_IDENTITY_CHARS or not value.isprintable():
        raise LifecycleError(f"{name} must be a printable non-empty string of at most {MAX_IDENTITY_CHARS} characters")
    return value


def _digest(tag: str, body: Any) -> str:
    return hashlib.sha256(tag.encode() + b"\0" + json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _hash_array(array: Any) -> str:
    h = hashlib.sha256()
    ls._stream_hash(h, np.ascontiguousarray(array))
    return h.hexdigest()


def _take(obj: Any, fields: dict, *, skip: tuple, kind: str) -> None:
    """Typed field transfer for a sealed record: a missing or unexpected field is a LifecycleError, never a KeyError."""
    names = [n for n in obj.__slots__ if n not in skip]
    missing = [n for n in names if n not in fields]
    if missing or set(fields) - set(names):
        raise LifecycleError(f"internal: {kind} fields missing {missing} / unexpected {sorted(set(fields) - set(names))}")
    for n in names:
        object.__setattr__(obj, n, fields[n])


# --- ledgers -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Watermark:
    """Bounded scalar copy of the last-ACCOUNTED local ledger of the current role (cumulative m3 over active cells). A fold
    is `local_now - watermark`; the watermark then becomes `local_now`. Common to the storm ledger and the general interevent
    ledger (E/T columns are exactly 0 for a storm)."""

    n_steps: int = 0
    n_requests: int = 0
    rain_m3: float = 0.0
    preferential_intake_m3: float = 0.0
    matrix_intake_m3: float = 0.0
    surface_return_m3: float = 0.0
    drainage_m3: float = 0.0
    export_m3: float = 0.0
    surface_evaporation_m3: float = 0.0
    soil_evaporation_m3: float = 0.0
    transpiration_m3: float = 0.0
    requested_transpiration_m3: float = 0.0
    unmet_transpiration_m3: float = 0.0
    surface_evaporation_unmet_m3: float = 0.0
    soil_evaporation_unmet_m3: float = 0.0
    hydro_bound_m3: float = 0.0
    t_soil_bound_m3: float = 0.0
    max_abs_residual_m3: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


_FLUX = ("rain_m3", "preferential_intake_m3", "matrix_intake_m3", "surface_return_m3", "drainage_m3", "export_m3",
         "surface_evaporation_m3", "soil_evaporation_m3", "transpiration_m3", "requested_transpiration_m3",
         "unmet_transpiration_m3", "surface_evaporation_unmet_m3", "soil_evaporation_unmet_m3", "hydro_bound_m3",
         "t_soil_bound_m3")


def _mark_from_storm(led: ls.LayeredWaterLedger) -> Watermark:
    ls._check_ledger(led)
    return Watermark(int(led.n_steps), 0, float(led.rain_m3), float(led.preferential_intake_m3), float(led.matrix_intake_m3),
                     float(led.surface_return_m3), float(led.drainage_m3), float(led.export_m3), 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, float(led.global_residual_bound_m3), 0.0, float(led.max_abs_global_residual_m3))


def _mark_from_general(led: it.GeneralWaterLedger) -> Watermark:
    h, t = led.hydrology, led.transpiration
    return Watermark(int(h.n_steps), int(t.n_requests), float(h.rain_m3), float(h.preferential_intake_m3),
                     float(h.matrix_intake_m3), float(h.surface_return_m3), float(h.drainage_m3), float(h.export_m3),
                     float(h.surface_evaporation_m3), float(h.soil_evaporation_m3), float(t.actual_m3), float(t.requested_m3),
                     float(t.unmet_m3), float(h.surface_evaporation_unmet_m3), float(h.soil_evaporation_unmet_m3),
                     float(h.global_residual_bound_m3), float(t.soil_bound_m3),
                     max(float(h.max_abs_global_residual_m3), float(t.max_abs_soil_residual_m3),
                         float(t.max_abs_request_residual_m3)))


@dataclass(frozen=True)
class LifetimeLedger:
    """Run-lifetime external-flux ledger (m3, active cells). `rain` is the only source; `export`, bottom `drainage`, surface and
    soil evaporation and ACTUAL transpiration are the only losses. Preferential/matrix intake and saturation return are
    internal transfers; requested/unmet T and unmet E are diagnostics; the wetting history J is not water. The run anchors are
    the inventories at `start_lifecycle`. `stage_bound_m3` accumulates the composed per-step bounds and native T-soil bounds of
    every folded window; the operation counts feed the derived cumulative FP64 bound."""

    anchor_surface_m3: float = 0.0
    anchor_soil_m3: float = 0.0
    anchor_t_s: float = 0.0
    rain_m3: float = 0.0
    export_m3: float = 0.0
    drainage_m3: float = 0.0
    surface_evaporation_m3: float = 0.0
    soil_evaporation_m3: float = 0.0
    transpiration_m3: float = 0.0
    preferential_intake_m3: float = 0.0
    matrix_intake_m3: float = 0.0
    surface_return_m3: float = 0.0
    requested_transpiration_m3: float = 0.0
    unmet_transpiration_m3: float = 0.0
    surface_evaporation_unmet_m3: float = 0.0
    soil_evaporation_unmet_m3: float = 0.0
    stage_bound_m3: float = 0.0
    max_abs_local_residual_m3: float = 0.0
    n_hydro_steps: int = 0
    n_t_requests: int = 0
    n_folds: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _check_lifetime(lt: Any) -> None:
    if type(lt) is not LifetimeLedger:
        raise LifecycleError("lifetime must be a genuine LifetimeLedger")
    for name in lt.__dict__:
        if name in ("n_hydro_steps", "n_t_requests", "n_folds"):
            _count(getattr(lt, name), "lifetime." + name)
        else:
            _real(getattr(lt, name), "lifetime." + name, nonneg=True)
    if lt.max_abs_local_residual_m3 > lt.stage_bound_m3:
        raise LifecycleError("lifetime max local residual exceeds its accumulated stage bound")


def _fold(lt: LifetimeLedger, mark: Watermark, cur: Watermark) -> tuple[LifetimeLedger, Watermark]:
    """The ONLY crediting path: add `cur - mark` (never `cur`) and return the new watermark. Pure."""
    if cur.n_steps < mark.n_steps or cur.n_requests < mark.n_requests:
        raise LifecycleError("the local ledger count moved backwards relative to its accounted watermark; refused")
    new = {}
    for name in _FLUX:
        d = float(getattr(cur, name)) - float(getattr(mark, name))
        if d < 0.0:
            raise LifecycleError(f"the local ledger {name} moved backwards relative to its accounted watermark; refused")
        new[name] = d
    out = LifetimeLedger(
        lt.anchor_surface_m3, lt.anchor_soil_m3, lt.anchor_t_s,
        lt.rain_m3 + new["rain_m3"], lt.export_m3 + new["export_m3"], lt.drainage_m3 + new["drainage_m3"],
        lt.surface_evaporation_m3 + new["surface_evaporation_m3"], lt.soil_evaporation_m3 + new["soil_evaporation_m3"],
        lt.transpiration_m3 + new["transpiration_m3"], lt.preferential_intake_m3 + new["preferential_intake_m3"],
        lt.matrix_intake_m3 + new["matrix_intake_m3"], lt.surface_return_m3 + new["surface_return_m3"],
        lt.requested_transpiration_m3 + new["requested_transpiration_m3"],
        lt.unmet_transpiration_m3 + new["unmet_transpiration_m3"],
        lt.surface_evaporation_unmet_m3 + new["surface_evaporation_unmet_m3"],
        lt.soil_evaporation_unmet_m3 + new["soil_evaporation_unmet_m3"],
        lt.stage_bound_m3 + new["hydro_bound_m3"] + new["t_soil_bound_m3"],
        max(lt.max_abs_local_residual_m3, float(cur.max_abs_residual_m3)),
        lt.n_hydro_steps + (cur.n_steps - mark.n_steps), lt.n_t_requests + (cur.n_requests - mark.n_requests),
        lt.n_folds + 1)
    _check_lifetime(out)
    return out, cur


def _lifetime_bound(lop: LifecycleOperator, lt: LifetimeLedger, h: float, soil: float) -> float:
    """MAPLE `volume_roundoff_bound_m3` only (design section 4). Every local value is <= the final monotone lifetime cumulative,
    so the final cumulative is the scale of every addition/subtraction ever performed."""
    ops = int(lt.n_hydro_steps) + 2 * int(lt.n_folds) + int(lt.n_t_requests) + 1
    total = float(lt.stage_bound_m3) + volume_roundoff_bound_m3(ops, float(lt.stage_bound_m3))
    cums = (lt.rain_m3, lt.export_m3, lt.drainage_m3, lt.surface_evaporation_m3, lt.soil_evaporation_m3,
            lt.transpiration_m3)
    for c in cums:
        total += volume_roundoff_bound_m3(ops, float(c))
    fresh = 2 * (lop.n_layers + 1) * lop.n_cells + 14
    scale = max(h, soil, float(lt.anchor_surface_m3), float(lt.anchor_soil_m3), *(float(c) for c in cums))
    total += volume_roundoff_bound_m3(fresh, scale)
    if not math.isfinite(total) or total < 0.0:
        raise LifecycleError("non-finite lifetime closure bound")
    return total


# --- memory ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LifecycleMemory:
    """Pure arithmetic (see MEMORY_SCOPE). `*_bytes` are the figures admitted at the corresponding transition; the
    begin/enter figures that depend on the actual schedule/operators are given by `required_begin_storm_bytes` and
    `required_enter_interevent_bytes`."""

    backend: str
    n_layers: int
    n_cells: int
    n_faces: int
    n_classes: int
    accepted_bytes: int
    static_bytes: int
    reference_depth_bytes: int
    reserves_bytes: int
    storm_required_bytes: int
    begin_storm_base_bytes: int
    audit_scratch_bytes: int
    audit_bytes: int
    reset_scratch_bytes: int
    reset_bytes: int
    start_bytes: int
    owned_fixed_bytes: int  # the lifecycle's OWN canonical cover (8 N) + owned T bounds (2 L doubles): live beside every phase
    call_parent_bytes: int  # distinct accepted arrays a multi-step call keeps through its parent (native mobile is shared)
    begin_copy_bytes: int  # faces (+ bedless surface) copied by the storm initializer beside the boundary arrays
    storm: Any  # deep read-only mapping (the layered-storm estimate); never a caller-mutable dict
    scope: str = MEMORY_SCOPE

    def __post_init__(self) -> None:
        # detach from the component estimate's dicts and freeze deeply: no ordinary alias can change a future requirement
        object.__setattr__(self, "storm", _freeze(copy.deepcopy(_thaw(self.storm))))

    def as_dict(self) -> dict[str, Any]:
        """A fresh plain, JSON-ready, DEEP copy: mutating it never reaches this record or any later requirement."""
        return {k: _thaw(v) for k, v in self.__dict__.items()}


#: Map of the public `_admit` labels to the CUDA plan entries.
_ENTRY_OF = {"enter_interevent_from_boundary": "enter_from_boundary", "reset": "reset", "start": "start", "begin_storm": "begin_storm", "evolve_storm": "evolve_storm", "enter_interevent": "enter_interevent",
             "evolve_interevent": "evolve_interevent", "request_withdrawal": "request", "close": "close", "audit": "audit_boundary",
             "audit_storm": "audit_storm", "audit_interevent": "audit_interevent", "cover inspection": "cover_inspection",
             # Phase 1.5 daily transactions: registered CUDA entries with their own separate host/device whole-call plans
             "daily_start": "daily_start", "daily": "daily", "daily_begin_storm": "daily_begin_storm", "daily_finish": "daily_finish"}


def _cuda_source_record_bytes(profile: Any) -> int:
    """Additional retained host bytes of the capability's source-association record (0 for the native39 pair); pure arithmetic, no import."""
    return it._cuda_source_record_extra_bytes(wlc.general_profile_of(profile))


def _host_kw(lop: LifecycleOperator) -> dict[str, Any]:
    """CUDA-only host quota controls forwarded to the genuine component factories; empty on CPU (call shapes unchanged). The reserve is the caller's
    declared reserve PLUS the wrapper's retained host leaves (lifecycle canonical vector, capability and source records) that are alive beside every
    child stage but are not in that child's own requirement. It is never a whole-call peak: each child's own fresh-query guard still decides."""
    cap = lop._cuda
    if cap is None:
        return {}
    return {"host_available_bytes_fn": cap.host_available_bytes_fn,
            "host_reserve_bytes": cap.host_reserve_bytes + wlc.wrapper_retained_host_bytes(lop.n_layers)
            + _cuda_source_record_bytes(cap.profile)}  # + the native9f association record (0 for the native39 pair)


def _cuda_enter_stage_table(lop: LifecycleOperator, parent: LifecycleState) -> dict[str, int]:
    """Named DEVICE stages of the CUDA `enter_interevent` (the genuine no-supplied-operator branch). The nested wrapper packet (16 L) belongs only to
    stages where a genuine prepared general owner exists (enable/transition and audit); the IW build and the cold primitive preparation precede it.
    Every stage also carries the live event maps + lifetime audit scratch + the wrapper's owned fixed leaves. Pure arithmetic over sealed estimates."""
    m = lop.memory
    op = parent.binding.op
    iwm, gm = _interevent_estimates(lop, op)
    pk = lop._cuda.memory.nested_general_packet_bytes
    fixed = max(int(parent.binding.event.required_bytes) - int(op.memory.required_bytes), 0) + m.audit_scratch_bytes + m.owned_fixed_bytes
    return {"interevent_water_build": int(iwm.required_bytes + iwm.build_bytes + fixed),
            "general_cold_prepare_no_supplied_operator": int(gm.cold_prepare_required_bytes + fixed),
            "enable_transition_with_nested_packet": int(gm.transition_required_bytes + pk + fixed),
            "audit_with_nested_packet": int(gm.audit_required_bytes + pk + fixed)}


def _cuda_boundary_enter_stage_table(lop: LifecycleOperator) -> dict[str, int]:
    """Named DEVICE stages of the CUDA boundary-origin interevent entry. No completed parent operator exists, so a fresh storm operator is built on the
    CURRENT bed (its native diagnostic stage is real work); each stage also carries the live fixed leaves (audit scratch + owned fixed); the live parent
    aggregate is inside every stage's resident. The nested T packet (16 L) appears only where a genuine prepared general owner exists. The CPU
    `supplied_prepare` branch is excluded (no supplied primitive on CUDA). Pure arithmetic over sealed estimates (no operator, no allocation)."""
    m = lop.memory
    iwm, bem, gm = _boundary_estimates(lop)
    pk = lop._cuda.memory.nested_general_packet_bytes
    fixed = m.audit_scratch_bytes + m.owned_fixed_bytes
    return {"storm_operator_build": int(m.storm_required_bytes + m.storm["build_bytes"] + fixed),
            "interevent_water_prepare": int(iwm.required_bytes + iwm.build_bytes + fixed),
            "boundary_entry_with_quiet_maps": int(bem.required_bytes + fixed),
            "general_cold_prepare_no_supplied_operator": int(gm.cold_prepare_required_bytes + fixed),
            "enable_transition_with_nested_packet": int(gm.transition_required_bytes + pk + fixed),
            "audit_with_nested_packet": int(gm.audit_required_bytes + pk + fixed)}


#: Device bytes of the uniform active rain field built by `rainfall.rainfall_field` on the device (`ones`, the `where` result and at most one frozen copy; the
#: active-mask reshape is a device VIEW, never a copy): 3 x 8 N, declared.
_DEVICE_FIELD_DOUBLES = 3


def _daily_step_device_bytes(lop: LifecycleOperator, gm: Any, feedback: int, live_event: int) -> int:
    """Device figure of the ONE 86400 s day step beside the live parent: the general owner's step/audit/post figures (which count its resident operands,
    including the retained zero-added tensor the pulse step does not read), the nested wrapper packet, the live storm event maps (0 for a boundary parent),
    the call-parent arrays, the lifecycle's owned fixed leaves and the caller's retained feedback."""
    m = lop.memory
    after = gm.resident_bytes + gm.reserves_bytes + gm.column_reserve_bytes + m.audit_scratch_bytes
    return int(max(int(gm.step_required_bytes), int(gm.audit_required_bytes), int(after)) + lop._cuda.memory.nested_general_packet_bytes + m.owned_fixed_bytes
               + m.call_parent_bytes + live_event + feedback)


def _cuda_daily_plan(lop: LifecycleOperator, entry: str, parent: Any, schedule: Any, feedback: int) -> dict[str, Any]:
    """METADATA-ONLY whole-call host+device plan of one Phase 1.5 daily CUDA transaction (no array, no read/hash/query): the largest simultaneous DEVICE stage of
    the genuine entry (the existing stage tables) and of the one day step beside the live parent, composed with the separate HOST figure by the registered
    composer. Only the native9f BOUNDARY-capable lifecycle has these entries."""
    cap = lop._cuda
    if cap.profile != wlc.CUDA9F_BOUNDARY_PROFILE:
        raise LifecycleError(f"the daily CUDA entries need the explicit native9f CUDA9F_BOUNDARY_PROFILE lifecycle (this one is {cap.profile!r}); refused")
    stages: dict[str, int] | None = None
    if entry == "daily_start":
        _check_role(parent, "boundary")
        raw = _raw_audit_bytes(lop, parent)
    elif entry == "daily_begin_storm":
        _check_role(parent, "boundary")
        raw = _raw_begin_storm_bytes(lop, schedule) + _DEVICE_FIELD_DOUBLES * _F8 * lop.n_cells
    elif entry == "daily":
        _check_role(parent, "boundary")
        stages = dict(_cuda_boundary_enter_stage_table(lop))
        _, _, gm = _boundary_estimates(lop)
        stages["day_step_with_pulse"] = _daily_step_device_bytes(lop, gm, feedback, 0)
        raw = max(stages.values())
    else:  # daily_finish
        _check_role(parent, "storm")
        stages = dict(_cuda_enter_stage_table(lop, parent))
        op, event = parent.binding.op, parent.binding.event
        _, gm = _interevent_estimates(lop, op)
        stages["day_step_beside_completed_storm"] = _daily_step_device_bytes(lop, gm, feedback, max(int(event.required_bytes) - int(op.memory.required_bytes), 0))
        raw = max(stages.values())
    sched = () if schedule is None else (schedule.edges_s.nbytes, schedule.intensity_mm_per_h.nbytes, schedule.rate_m_per_s.nbytes)
    iw_host = (_interevent_estimates(lop, parent.binding.op)[0].build_host_bytes if entry == "daily_finish"
               else _boundary_estimates(lop)[0].build_host_bytes if entry == "daily" else 0)
    plan = wlc.compose_entry_plan(lop.n_layers, lop.n_cells, entry, device_raw_bytes=int(raw), host_reserve_bytes=cap.host_reserve_bytes,
                                  source_cold=it._windowed_source_cold(), schedule_bytes=sched, n_faces=lop.memory.n_faces,
                                  storm_host_stage_bytes=int(lop.memory.storm.get("host_bytes", 0)), interevent_build_host_bytes=int(iw_host),
                                  source_record_bytes=_cuda_source_record_bytes(cap.profile))
    if stages is not None:
        plan["device_stage_table"] = stages
    return plan


def cuda_entry_plan(lop: LifecycleOperator, entry: str, *, parent: Any = None, schedule: Any = None, retained_feedback_bytes: int = 0) -> dict[str, Any]:
    """The ONE authoritative, METADATA-ONLY complete host+device plan of a public CUDA entry; `_admit` uses exactly this on every CUDA entry. It performs
    the sealed type/component/source/cover-metadata checks (`_check_lop(preflight=False)`: no IO) but NO device or host availability read, NO current-device
    query, no array, no read/hash and no cache warming (it may build small Python records). Execution paths call the real `column.preflight()` first.
    `entry`: `start`, `begin_storm`, `evolve_storm`, `enter_interevent`, `evolve_interevent`, `request`, `close`, `audit_boundary|audit_storm|audit_interevent`,
    `cover_inspection`; `parent` is the state the entry receives (not needed for start/cover_inspection/begin_storm); `schedule` for begin_storm."""
    _check_lop(lop, preflight=False)
    cap = lop._cuda
    if cap is None:
        raise LifecycleError("cuda_entry_plan exists only for a CUDA lifecycle (CPU figures are the unchanged required_* functions)")
    fb = _count(retained_feedback_bytes, "retained_feedback_bytes")
    raw_of = {"start": lambda: lop.memory.start_bytes, "evolve_storm": lambda: _raw_evolve_storm_bytes(lop, parent),
              "evolve_interevent": lambda: _raw_evolve_interevent_bytes(lop, parent, fb), "request": lambda: _raw_request_bytes(lop, parent, fb),
              "close": lambda: _raw_close_bytes(lop, parent), "audit_boundary": lambda: _raw_audit_bytes(lop, parent),
              "audit_storm": lambda: _raw_audit_bytes(lop, parent), "audit_interevent": lambda: _raw_audit_bytes(lop, parent),
              "begin_storm": lambda: _raw_begin_storm_bytes(lop, schedule),
              "cover_inspection": lambda: lop.memory.storm_required_bytes + lop.memory.owned_fixed_bytes}
    stages = None
    if entry in wlc.DAILY_ENTRIES:
        return _cuda_daily_plan(lop, entry, parent, schedule, fb)
    if entry in ("enter_from_boundary", "reset") and not cap.boundary_capable:
        raise LifecycleError(f"the {entry} entry needs the explicit CUDA_BOUNDARY_PROFILE lifecycle (this one is the storm-first profile); refused")
    if entry == "enter_interevent":
        _check_role(parent, "storm")
        stages = _cuda_enter_stage_table(lop, parent)
        raw = max(stages.values())
    elif entry == "enter_from_boundary":
        stages = _cuda_boundary_enter_stage_table(lop)
        raw = max(stages.values())
    elif entry == "reset":
        # CPU-shaped reset scratch + the quiet maps the shared evidence helper uploads once for this control (escape + source bed doubles, two type masks)
        raw = lop.memory.reset_bytes + 2 * _F8 * lop.n_cells + lop.memory.n_faces
    elif entry in raw_of:
        raw = raw_of[entry]()
    else:
        raise LifecycleError(f"unknown CUDA lifecycle entry {entry!r}")
    sched = () if schedule is None else (schedule.edges_s.nbytes, schedule.intensity_mm_per_h.nbytes, schedule.rate_m_per_s.nbytes)
    iw_host = (_interevent_estimates(lop, parent.binding.op)[0].build_host_bytes if entry == "enter_interevent"
               else _boundary_estimates(lop)[0].build_host_bytes if entry == "enter_from_boundary" else 0)
    plan = wlc.compose_entry_plan(lop.n_layers, lop.n_cells, entry, device_raw_bytes=int(raw), host_reserve_bytes=cap.host_reserve_bytes,
                                  source_cold=it._windowed_source_cold(), schedule_bytes=sched, n_faces=lop.memory.n_faces,
                                  storm_host_stage_bytes=int(lop.memory.storm.get("host_bytes", 0)), interevent_build_host_bytes=int(iw_host),
                                  source_record_bytes=_cuda_source_record_bytes(cap.profile))
    if stages is not None:
        plan["device_stage_table"] = stages  # named stages (maximum = raw_entry_figure); the nested packet is inside its two stages, not added again
    return plan


def _admit(lop: LifecycleOperator, need: int, what: str, *, parent: Any = None, schedule: Any = None, feedback: int = 0) -> int:
    """Device-side (column) admission. CPU: exactly the unchanged check of `need`. CUDA: the authoritative `cuda_entry_plan` is computed once and its
    SEPARATE host figure and complete device figure are both admitted (host first; wrong device was refused earlier by `_check_lop`)."""
    if lop._cuda is not None and what not in _ENTRY_OF:
        # a CUDA lifecycle has no CPU-integer fallthrough: an unregistered operation would silently admit a host/CPU figure against the device
        raise LifecycleError(f"no CUDA admission plan is registered for the lifecycle operation {what!r}; refused (never a CPU-integer admission on a CUDA lifecycle)")
    if lop._cuda is not None and what in _ENTRY_OF:
        entry = _ENTRY_OF[what]
        plan = cuda_entry_plan(lop, entry, parent=parent, schedule=schedule, retained_feedback_bytes=feedback)  # pure metadata (no device query)
        try:
            if callable(getattr(lop.column, "preflight", None)):
                lop.column.preflight()  # the actual CURRENT-device gate immediately before the first host/device availability read
        except ss.SubsurfaceError as exc:
            raise LifecycleError(str(exc)) from None
        try:
            ls._admit_host_build(plan["host_bytes"], lop._cuda.host_available_bytes_fn)
        except ls.LayeredMemoryError as exc:
            raise ls.LayeredMemoryError(f"lifecycle CUDA {entry} host stage: {exc}") from None
        except ls.LayeredStormError as exc:
            raise LifecycleError(str(exc)) from None
        need = plan["device_bytes"]
    return ls._admit(lop.column, lop.shape, int(need), "lifecycle " + what)


def _event_extra_bytes(lop: LifecycleOperator, schedule: Any) -> int:
    """The bytes `prepare_layered_event` adds beyond the operator (its own formula, mirrored for a PRE-admission)."""
    N, L, F = lop.n_cells, lop.n_layers, lop.memory.n_faces
    extra = ((16 * 8 + 1 + L * 8) * N + 4 * schedule.edges_s.nbytes + 2 * schedule.intensity_mm_per_h.nbytes
             + 2 * schedule.rate_m_per_s.nbytes + 4 * F + 4096)
    extra += ((L + 3) * N + F) * 8
    if lop.context is not None:
        extra += N * 8
    return int(extra)


def _raw_begin_storm_bytes(lop: LifecycleOperator, schedule: Any) -> int:
    """Arithmetic body of `required_begin_storm_bytes` without the sealed-operator check (shared with the metadata-only CUDA planner)."""
    if type(schedule) is not RainfallSchedule:
        raise LifecycleError("schedule must be a genuine RainfallSchedule")
    m = lop.memory
    return int(m.storm_required_bytes + m.owned_fixed_bytes + m.begin_copy_bytes
               + max(m.storm["build_bytes"], m.storm["initialization_bytes"], _event_extra_bytes(lop, schedule)))


def required_begin_storm_bytes(lop: LifecycleOperator, schedule: Any) -> int:
    """Bytes admitted BEFORE `begin_storm` prepares the fresh operator, the initial state and the event."""
    _check_lop(lop)
    return _raw_begin_storm_bytes(lop, schedule)


def _interevent_estimates(lop: LifecycleOperator, op: ls.LayeredStormOperator) -> tuple[Any, Any]:
    iwm = iw.estimate_interevent_memory(op, lop.evaporation, extra_reserve_bytes=0)
    base = SimpleNamespace(op=op, n_layers=op.n_layers, n_cells=op.n_cells, backend=op.backend, memory=iwm)
    return iwm, it.estimate_general_memory(base, extra_reserve_bytes=0)


def required_enter_interevent_bytes(lop: LifecycleOperator, parent: LifecycleState) -> int:
    """Bytes admitted BEFORE `enter_interevent_phase` prepares the interevent owners (storm role still live)."""
    _check_lop(lop)
    if type(parent) is not LifecycleState or parent.role != "storm":
        raise LifecycleError("a 'storm' lifecycle state is required")
    op, event = parent.binding.op, parent.binding.event
    iwm, gm = _interevent_estimates(lop, op)
    live_event = int(event.required_bytes) - int(op.memory.required_bytes)  # the event maps beside the operator
    need = max(iwm.required_bytes + iwm.build_bytes, gm.transition_required_bytes, gm.cold_prepare_required_bytes,
               gm.supplied_prepare_required_bytes, gm.audit_required_bytes)
    return int(need + max(live_event, 0) + lop.memory.audit_scratch_bytes + lop.memory.owned_fixed_bytes)


def _boundary_estimates(lop: LifecycleOperator) -> tuple[Any, Any, Any]:
    """Metadata-only owner estimates BEFORE the fresh storm operator exists: a duck of exactly the operator metadata the estimators
    read, built from the lifecycle's own sealed figures (no allocation, no preparation)."""
    duck = SimpleNamespace(n_layers=lop.n_layers, n_cells=lop.n_cells, n_classes=lop.n_classes, backend=lop.backend,
                           shape=lop.shape, column=lop.column, preferential=lop.preferential,
                           _control_values=lop._control_values, memory=ls.LayeredMemory(**dict(lop.memory.storm)),
                           bed=None if lop.context is None else True, extra_reserve_bytes=lop.extra_reserve_bytes,
                           native_reserve_bytes=lop.native_reserve_bytes)
    iwm = iw.estimate_interevent_memory(duck, lop.evaporation, extra_reserve_bytes=0)
    bem = iw.estimate_boundary_entry_memory(duck, lop.evaporation, extra_reserve_bytes=0)
    base = SimpleNamespace(op=duck, n_layers=lop.n_layers, n_cells=lop.n_cells, backend=lop.backend, memory=iwm)
    return iwm, bem, it.estimate_general_memory(base, extra_reserve_bytes=0)


def required_enter_from_boundary_bytes(lop: LifecycleOperator, parent: LifecycleState) -> int:
    """METADATA-ONLY whole-call requirement of `enter_interevent_from_boundary`, admitted BEFORE the fresh operator, any hash, any
    reduction or any canonical array. Largest simultaneous stage of: the fresh storm-operator build's own admission beside the one
    current boundary (`prepare_layered_storm` admits its whole-step requirement plus the cold build), the interevent-owner preparation, the named boundary-entry requirement (inventory
    reductions, validation masks, quiet scratch), and the general owner's transition/cold-preparation/audit peaks; plus the
    lifecycle's own cover/T bounds and lifetime-audit scratch. Only the current boundary role is live (no event, no old phase
    owner); there is no empirical floor and no unused-slack claim."""
    _check_lop(lop)
    if type(parent) is not LifecycleState or parent.role != "boundary":
        raise LifecycleError("a 'boundary' lifecycle state is required")
    m = lop.memory
    iwm, bem, gm = _boundary_estimates(lop)
    build = int(m.storm_required_bytes) + int(m.storm["build_bytes"])  # the operator build's OWN admission (`prepare_layered_storm`)
    need = max(build, iwm.required_bytes + iwm.build_bytes, bem.required_bytes, gm.transition_required_bytes,
               gm.cold_prepare_required_bytes, gm.supplied_prepare_required_bytes, gm.audit_required_bytes)
    return int(need + m.audit_scratch_bytes + m.owned_fixed_bytes)


def _check_role(parent: Any, role: str) -> None:
    if type(parent) is not LifecycleState or parent.role != role:
        raise LifecycleError(f"a '{role}' lifecycle state is required")


def _raw_audit_bytes(lop: LifecycleOperator, state: LifecycleState) -> int:
    """Arithmetic body of `required_audit_bytes` without the sealed-operator check (shared with the metadata-only CUDA planner)."""
    if type(state) is not LifecycleState or state not in _MINTED:
        raise LifecycleError("state must be a genuine LifecycleState")
    m = lop.memory
    if state.role == "boundary":
        role = m.storm["resident_bytes"] + m.reserves_bytes
    elif state.role == "storm":
        b = state.binding
        if type(b) is not _Binding or b.event is None:
            raise LifecycleError("a storm state needs its event binding; refused")
        role = (m.storm["resident_bytes"] + max(int(b.event.required_bytes) - int(b.op.memory.required_bytes), 0)
                + m.reserves_bytes)
    elif state.role == "interevent":
        b = state.binding
        if type(b) is not _Binding or b.gop is None:
            raise LifecycleError("an interevent state needs its general owner binding; refused")
        gm = b.gop.memory
        role = gm.resident_bytes + gm.reserves_bytes + gm.column_reserve_bytes
    else:
        raise LifecycleError("unknown lifecycle role; refused")
    return int(role + m.audit_scratch_bytes + m.owned_fixed_bytes)


def required_audit_bytes(lop: LifecycleOperator, state: LifecycleState) -> int:
    """METADATA-ONLY composed bytes `audit_lifecycle` admits BEFORE it validates or reads any inventory: the ACTUAL current-role
    resident + the wrapper's own cover/T bounds + the reserves + the lifetime audit scratch (named object allowance included once).
    Role resident: boundary = the layered accepted arrays + statics (no phase operator is live; conservative superset, H0 8N
    counted once for Dense); storm = the storm operator resident (H0 once) + the live event maps (`event.required_bytes -
    op.required_bytes`); interevent = the general owner's resident (storm resident + cover/zero-rate maps 16 N + owned T bounds
    16 L, H0 once) with its own reserves + column reserve. No empirical floor."""
    _check_lop(lop)
    return _raw_audit_bytes(lop, state)


def _raw_evolve_storm_bytes(lop: LifecycleOperator, parent: LifecycleState) -> int:
    """Arithmetic body of `required_evolve_storm_bytes` without the sealed-operator check (shared with the metadata-only CUDA planner)."""
    _check_role(parent, "storm")
    m = lop.memory
    during = int(parent.binding.event.required_bytes) + m.owned_fixed_bytes
    after = m.storm["resident_bytes"] + m.call_parent_bytes + m.audit_scratch_bytes + m.reserves_bytes + m.owned_fixed_bytes
    return int(max(during, after))


def required_evolve_storm_bytes(lop: LifecycleOperator, parent: LifecycleState) -> int:
    """Composed bytes admitted BEFORE `evolve_storm` calls the event: the event's own requirement (it already counts the
    call-argument parent beside the loop state and one trial) + the lifecycle's OWN cover/T bounds, and the post-loop lifetime
    audit (current role statics incl. the retained H0 once + the retained call parent + audit scratch + reserves)."""
    _check_lop(lop)
    return _raw_evolve_storm_bytes(lop, parent)


def _raw_evolve_interevent_bytes(lop: LifecycleOperator, parent: LifecycleState, retained_feedback_bytes: Any = 0) -> int:
    """Arithmetic body of `required_evolve_interevent_bytes` without the sealed-operator check (shared with the metadata-only CUDA planner)."""
    _check_role(parent, "interevent")
    fb = _count(retained_feedback_bytes, "retained_feedback_bytes")
    gm, m = parent.binding.gop.memory, lop.memory
    own = m.owned_fixed_bytes + m.call_parent_bytes + fb
    after = gm.resident_bytes + gm.reserves_bytes + gm.column_reserve_bytes + m.audit_scratch_bytes
    return int(max(int(gm.step_required_bytes), int(gm.audit_required_bytes), after) + own)


def required_evolve_interevent_bytes(lop: LifecycleOperator, parent: LifecycleState, retained_feedback_bytes: Any = 0) -> int:
    """Composed bytes admitted BEFORE `evolve_interevent` runs a step: the general owner's step/audit requirement + the
    wrapper-owned ORIGINAL call parent arrays (kept through the whole loop, not shared with the current state) + the lifecycle's
    OWN cover/T bounds + the CALLER's declared `retained_feedback_bytes` (distinct from the wrapper's own retention), and the
    post-loop lifetime audit scratch beside the parent and the result."""
    _check_lop(lop)
    return _raw_evolve_interevent_bytes(lop, parent, retained_feedback_bytes)


def _raw_request_bytes(lop: LifecycleOperator, parent: LifecycleState, retained_feedback_bytes: Any = 0) -> int:
    """Arithmetic body of `required_request_bytes` without the sealed-operator check (shared with the metadata-only CUDA planner)."""
    _check_role(parent, "interevent")
    fb = _count(retained_feedback_bytes, "retained_feedback_bytes")
    gm, m = parent.binding.gop.memory, lop.memory
    ln = _F8 * m.n_layers * m.n_cells
    after = (gm.resident_bytes + gm.reserves_bytes + gm.column_reserve_bytes + int(gm.feedback_bytes) + ln
             + m.audit_scratch_bytes)
    return int(max(int(gm.request_prepare_required_bytes), int(gm.withdraw_required_bytes), after) + m.owned_fixed_bytes + fb)


def required_request_bytes(lop: LifecycleOperator, parent: LifecycleState, retained_feedback_bytes: Any = 0) -> int:
    """Composed bytes admitted BEFORE `request_withdrawal` prepares the owned request: the general owner's preparation and
    withdrawal requirements + OWN cover/T bounds + the caller's retained feedback, and the post-transaction lifetime audit while
    the owned request, the returned actual/unmet/scale feedback, the old and the new soil are all live."""
    _check_lop(lop)
    return _raw_request_bytes(lop, parent, retained_feedback_bytes)


def _raw_close_bytes(lop: LifecycleOperator, parent: LifecycleState) -> int:
    """Arithmetic body of `required_close_bytes` without the sealed-operator check (shared with the metadata-only CUDA planner)."""
    if type(parent) is not LifecycleState or parent.role != "interevent":
        raise LifecycleError("an 'interevent' lifecycle state is required")
    gm, m = parent.binding.gop.memory, lop.memory
    post = gm.resident_bytes + gm.reserves_bytes + gm.column_reserve_bytes + m.audit_scratch_bytes
    return int(max(int(gm.audit_required_bytes), post) + m.owned_fixed_bytes)


def required_close_bytes(lop: LifecycleOperator, parent: LifecycleState) -> int:
    """Bytes admitted BEFORE `close_interevent` audits and builds the boundary."""
    _check_lop(lop)
    return _raw_close_bytes(lop, parent)


# --- the operator ------------------------------------------------------------------------------------------------------
class LifecycleOperator:
    """Factory-only immutable binding of the FIXED components (no array-bearing storm operator, no bed). Minted only by
    `prepare_lifecycle`."""

    __slots__ = ("_cover", "_material_ids", "_seal", "backend", "column", "context", "cover_sha256", "evaporation",
                 "extra_reserve_bytes", "graph", "matrix", "memory", "n_cells", "n_classes", "n_layers",
                 "native_reserve_bytes", "preferential", "router", "sha256", "shape", "t_op", "topology", "xp",
                 "_control_values", "_cuda")

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise LifecycleError("LifecycleOperator is constructed only by prepare_lifecycle()")
        _take(self, fields, skip=("_seal",), kind="operator")
        object.__setattr__(self, "_seal", seal)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("LifecycleOperator is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("LifecycleOperator is immutable")

    def __repr__(self) -> str:
        return (f"LifecycleOperator(backend={self.backend!r}, shape={self.shape}, n_layers={self.n_layers}, "
                f"dense_bed={self.context is not None}, sha256={self.sha256[:12]}...)")

    @property
    def cover(self) -> Any:
        """CPU: the immutable host cover (unchanged). CUDA: an ADMITTED fresh DETACHED device copy; internals use the private cover."""
        if self._cuda is None:
            return self._cover
        _check_lop(self)
        _admit(self, self.memory.storm_required_bytes + self.memory.owned_fixed_bytes + self.n_cells * _F8, "cover inspection")  # CUDA: cuda_entry_plan
        return self._cuda.cover.copy()


def _private_cover(lop: LifecycleOperator) -> Any:
    """The owned cover for internal callers (never a copying property): the host array on CPU, the private device array on CUDA."""
    return lop._cover if lop._cuda is None else lop._cuda.cover


def _check_cuda_capability(lop: LifecycleOperator, preflight: bool = True) -> None:
    """CUDA branch of `_check_lop` (metadata/actual-device identity only; no hash, no download): sealed capability, genuine CUDA column,
    the CURRENT device (column.preflight), the private device cover's dtype/shape/contiguity/device, and the cached native39 source
    association plus the explicit Dense class binding (cached evidence, not a fresh source hash)."""
    cap, column = lop._cuda, lop.column
    if type(cap) is not wlc.CudaLifecycleCapability or cap.seal is not _SEAL or cap.profile not in wlc.ALL_CUDA_PROFILES:
        raise LifecycleError("the CUDA lifecycle capability record is missing or foreign; refused")
    if column.backend != "cuda" or column.xp is np or lop.backend != "cuda" or lop.xp is np:
        raise LifecycleError("the CUDA lifecycle needs the genuine CUDA column namespace; refused")
    try:
        if preflight and callable(getattr(column, "preflight", None)):
            column.preflight()  # the CURRENT device must be the column's, before any quota query or copy (skipped ONLY by the pure planner)
        ss._check_dynamic(column, "cover", cap.cover, (lop.n_cells,))  # recorded-device metadata only (no current-device query)
    except ss.SubsurfaceError as exc:
        raise LifecycleError(str(exc)) from None
    if cap.cover.dtype != np.float64 or not cap.cover.flags.c_contiguous:
        raise LifecycleError("the private device cover must be a contiguous float64 array; refused")
    try:
        if cap.profile in wlc.CUDA_PROFILES:
            it._check_windowed_source_hot(cap.source)  # native39 pair: the unchanged historical guard
        else:  # native9f pair: the dense-CUDA guard for the EXPECTED family (never by whichever genuine record is presented)
            it._check_cuda_source_hot(cap.source, wlc.general_profile_of(cap.profile))
    except it.GeneralIntereventError as exc:
        raise LifecycleError(f"the verified native source association is no longer valid: {exc}") from None
    voxel = sys.modules.get(_sb.MAPLE_VOXEL_MODULE)
    if voxel is None or getattr(voxel, "VoxelColumnState", None) is not cap.dense_class or cap.dense_class is not VoxelColumnState:
        raise LifecycleError("the imported native Dense voxel class is no longer the one bound at preparation; refused")


def _check_lop(lop: Any, *, preflight: bool = True) -> None:
    """Metadata-only revalidation (every call, before any allocation). On CUDA `preflight=True` (the default, used by every execution path) also
    queries the CURRENT device through the column's genuine `preflight`; the metadata-only planner passes `preflight=False` and still runs every
    sealed type/component/source/cover-metadata/Dense-class association check (none of which does IO)."""
    if type(lop) is not LifecycleOperator or getattr(lop, "_seal", None) is not _SEAL:
        raise LifecycleError(f"lop must be a genuine LifecycleOperator, got {type(lop).__name__}")
    column = lop.column
    ss._check_column(column)
    pi._check_operator(lop.preferential)
    mi._check_operator(lop.matrix)
    if lop.preferential.column is not column or lop.matrix.column is not column:
        raise LifecycleError("the preferential/matrix operators are no longer bound to the lifecycle column; refused")
    if lop.evaporation.column is not column or float(column.layers.evaporation_factor) != 1.0:
        raise LifecycleError("the evaporation operator is no longer bound to the lifecycle column; refused")
    if lop._cuda is not None:
        _check_cuda_capability(lop, preflight)
    elif column.backend not in BACKENDS or lop.backend != column.backend or column.xp is not np:
        raise LifecycleError("unsupported or changed backend (NumPy/Numba only); refused")
    if lop.router.method != ls.SURFACE_METHOD or ls._control_values(lop.router.control) != lop._control_values:
        raise LifecycleError("the router method/control differs from the sealed values; refused")
    if lop._cuda is None and lop._cover.flags.writeable:
        raise LifecycleError("the owned cover lost its immutability; refused")
    if (lop.context is None) != (len(lop._material_ids) == 0):
        raise LifecycleError("the sealed native binding is inconsistent; refused")


def prepare_lifecycle(column: ss.SoilColumn, preferential: pi.PreferentialOperator, matrix: mi.MatrixOperator, router: Any,
                      graph: RoutingGraph, topology: PondTopology, evaporation: se.EvaporationOperator, cover: Any, *,
                      bed: Any = None, context: Any = None, native_reserve_bytes: int | None = None,
                      extra_reserve_bytes: int = 0, t_op: st.TranspirationOperator | None = None, cuda_profile: Any = None,
                      host_available_bytes_fn: Any = None, host_reserve_bytes: int = 0) -> LifecycleOperator:
    """Bind the fixed components. `cuda_profile=CUDA_PROFILE` is the separate explicit UNQUALIFIED opt-in for a genuine CUDA column
    (native39 Dense or bedless only; `host_available_bytes_fn`/`host_reserve_bytes` are CUDA-only host-RAM controls; a supplied `t_op`
    is unsupported: each interevent role mints the genuine primitive). The CPU refuses every CUDA-only option instead of ignoring it.
    Unsupported profiles (default CUDA, Windowed/foreign storage, a subclass bed) are refused here, before
    any trial. `bed`/`context` (both or neither) fix the native MAPLE39 DENSE material ids, the class count and the K-dependent
    memory; the bed is NOT retained (only its material identities), so no original array is held for the life of the run.
    `extra_reserve_bytes` is the caller's reserve for its own simultaneously live bytes (shared by the storm, interevent and
    general owners; declared once). `t_op` None prepares the native transpiration operator once."""
    ss._check_column(column)
    host_reserve = _count(host_reserve_bytes, "host_reserve_bytes")
    if host_available_bytes_fn is not None and not callable(host_available_bytes_fn):
        raise LifecycleError("host_available_bytes_fn must be callable or None")
    cuda = column.backend not in BACKENDS or column.xp is not np
    if not cuda:
        if cuda_profile is not None or host_available_bytes_fn is not None or host_reserve != 0:
            raise LifecycleError("cuda_profile and the host staging controls are CUDA-only; the CPU lifecycle refuses them (not ignored)")
    else:
        if cuda_profile is None:
            raise LifecycleError(f"the lifecycle supports the NumPy and Numba backends only (CUDA is not qualified here), got "
                                 f"{column.backend!r}; the separate explicit cuda_profile opt-in selects the UNQUALIFIED CUDA capability")
        if type(cuda_profile) is not str or cuda_profile not in wlc.ALL_CUDA_PROFILES:
            raise LifecycleError(f"cuda_profile must be exactly CUDA_PROFILE or CUDA_BOUNDARY_PROFILE (or the native9f pair CUDA9F_PROFILE / "
                                 f"CUDA9F_BOUNDARY_PROFILE), got {cuda_profile!r}")
        if column.backend != "cuda" or column.xp is np:
            raise LifecycleError("the CUDA lifecycle needs the genuine CUDA column namespace (backend 'cuda'); refused")
        if t_op is not None:
            raise LifecycleError("a supplied transpiration operator is unsupported on CUDA (each interevent role mints the genuine "
                                 "primitive through the general owner; an arbitrary RawKernel is never accepted)")
        try:
            if callable(getattr(column, "preflight", None)):
                column.preflight()  # the CURRENT device must be the column's, BEFORE any resource query, copy or hash
        except ss.SubsurfaceError as exc:
            raise LifecycleError(str(exc)) from None
    pi._check_operator(preferential)
    mi._check_operator(matrix)
    if preferential.column is not column or matrix.column is not column:
        raise LifecycleError("preferential and matrix operators must be prepared on exactly this column")
    if not isinstance(evaporation, se.EvaporationOperator) or evaporation.column is not column:
        raise LifecycleError("the evaporation operator must be prepared on exactly this column")
    if float(column.layers.evaporation_factor) != 1.0:
        raise LifecycleError("the column's evaporation_factor must be exactly 1.0 (the helper applies it once)")
    if not isinstance(topology, PondTopology) or not isinstance(graph, RoutingGraph):
        raise LifecycleError("topology and graph must be genuine PondTopology / RoutingGraph records")
    ls._check_router_binding(router, column.backend, graph, topology, column)
    if (bed is None) != (context is None):
        raise LifecycleError("bed and context must be given together (both) or not at all")
    extra = _count(extra_reserve_bytes, "extra_reserve_bytes")
    K, material_ids = 0, ()
    if bed is not None:
        if type(bed) is not BedState or type(context) is not BedContext:
            raise LifecycleError("bed must be a genuine BedState and context a genuine BedContext")
        if type(bed.voxel_column) is not VoxelColumnState:
            raise LifecycleError("only a native DENSE voxel column (exact VoxelColumnState) is supported; Windowed or foreign "
                                 f"storage ({type(bed.voxel_column).__name__}) has no lifecycle support")
        if native_reserve_bytes is None:
            raise LifecycleError("a native bed needs an explicit native_reserve_bytes (0 must be declared, never assumed)")
        native_reserve = _count(native_reserve_bytes, "native_reserve_bytes")
        K = len(context.grain_classes.classes)
        material_ids = ls._bed_material_ids(bed)
    else:
        if native_reserve_bytes is not None:
            raise LifecycleError("native_reserve_bytes is meaningless without a native bed (refused, not ignored)")
        native_reserve = 0
    ny, nx = (int(v) for v in topology.shape)
    N, L = column.n_cells, column.n_layers
    if N != ny * nx or tuple(graph.shape) != (ny, nx):
        raise LifecycleError("the column/graph extents disagree with the topology")
    control_values = ls._control_values(router.control)
    sm = ls.estimate_layered_memory(L, (ny, nx), backend=column.backend, limiter=control_values[1],
                                    reserve_bytes=column.reserve_bytes, extra_reserve_bytes=extra,
                                    n_species=preferential.n_species, diffusion_mode=column.layers.diffusion_mode,
                                    native_bed=bed is not None, n_classes=K, native_reserve_bytes=native_reserve)
    F = sm.n_faces
    scratch_audit = (L + 2) * N * _F8 + OBJECT_BYTES
    reserves = int(column.reserve_bytes) + extra + native_reserve
    reset_scratch = RESET_SCRATCH_DOUBLES_PER_CELL * N * _F8 + 2 * F * _F8 + OBJECT_BYTES
    # CPU: lifecycle cover + owned T residual/saturation bounds (80 B at N = 6, L = 2). CUDA: private device cover + capacity bound only
    # (no unused CUDA t_op leaf; the general owner counts its own device/host T parameters per entered role).
    owned_fixed = N * _F8 + (L * _F8 if cuda else 2 * L * _F8)
    call_parent = sm.accepted_bytes - N * K * _F8  # surface, faces, soil, J, intake; the native mobile array is shared
    begin_copy = F * _F8 + (0 if bed is not None else N * _F8)
    memory = LifecycleMemory(
        column.backend, L, N, F, K, sm.accepted_bytes, sm.static_bytes, sm.native_reference_depth_bytes, reserves,
        sm.required_bytes, 0, scratch_audit, sm.resident_bytes + scratch_audit + reserves + owned_fixed,
        reset_scratch, sm.accepted_bytes + sm.static_bytes + N * _F8 + reset_scratch + reserves + OBJECT_BYTES + owned_fixed,
        sm.required_bytes + sm.initialization_bytes + owned_fixed, owned_fixed, call_parent, begin_copy, sm.as_dict())
    if cuda:
        return _prepare_cuda_lifecycle(column, preferential, matrix, router, graph, topology, evaporation, cover, bed, context, extra,
                                       native_reserve, K, material_ids, control_values, memory, reserves, host_available_bytes_fn,
                                       host_reserve, cuda_profile)
    ls._admit(column, (ny, nx), N * _F8 + 4 * _F8 * L + OBJECT_BYTES + reserves, "lifecycle preparation")  # BEFORE any copy
    try:
        ss._check_dynamic(column, "cover", cover, (N,))
    except ss.SubsurfaceError as exc:
        raise LifecycleError(str(exc)) from None
    if not np.all(np.isfinite(cover)) or np.any(cover < 0.0) or np.any(cover > 1.0):
        raise LifecycleError("cover must be finite and within [0, 1] on every cell")
    owned_cover = ev._immutable_host(np.asarray(cover, dtype=np.float64))
    if t_op is None:
        t_op = st.prepare_transpiration(column, extra_reserve_bytes=extra)
    else:
        it._check_t_parameters(column, t_op, owned=False)
    cover_sha = _hash_array(owned_cover)
    h = hashlib.sha256(PROFILE.encode())
    part = evaporation.partition
    for piece in (column.backend, str((ny, nx)), repr(float(topology.dx_m)), column.layers.identity()["sha256"],
                  preferential.sha256, matrix.sha256, topology.topology_sha256, str(graph.input_sha256),
                  repr(control_values), str(K), cover_sha, json.dumps(asdict(part), sort_keys=True, allow_nan=False),
                  se.PARTITION_LEVEL, str(extra), str(native_reserve), st.VERSION, "dense" if bed is not None else "bedless"):
        h.update(piece.encode())
        h.update(b"\0")
    return LifecycleOperator(
        _SEAL, column=column, preferential=preferential, matrix=matrix, router=router, graph=graph, topology=topology,
        context=context, evaporation=evaporation, t_op=t_op, _cover=owned_cover, cover_sha256=cover_sha,
        _material_ids=material_ids, backend=column.backend, xp=np, shape=(ny, nx), n_cells=N, n_layers=L, n_classes=K,
        extra_reserve_bytes=extra, native_reserve_bytes=native_reserve, memory=memory, sha256=h.hexdigest(),
        _control_values=control_values, _cuda=None)


def _prepare_cuda_lifecycle(column, preferential, matrix, router, graph, topology, evaporation, cover, bed, context, extra,
                            native_reserve, K, material_ids, control_values, memory, reserves, host_fn, host_reserve, profile=CUDA_PROFILE):
    """CUDA cold ownership (UNQUALIFIED; written without any device access). Order: (current-device gate ran in the caller) -> metadata
    arithmetic plan -> SEPARATE device and host admission -> cached native39 source verification + Dense class binding -> private
    device cover (one validation packet, one streamed hash) and the `(L,)` capacity bound (host canonical + device copy)."""
    ny, nx = (int(v) for v in topology.shape)
    N, L = column.n_cells, column.n_layers
    xp = column.xp
    rec = _cuda_source_record_bytes(profile)
    cm = wlc.estimate_cuda_lifecycle_memory(L, N, host_reserve_bytes=host_reserve, source_cold=it._windowed_source_cold(), source_record_bytes=rec)
    plan = wlc.compose_entry_plan(L, N, "prepare", device_raw_bytes=cm.cold_device_bytes + reserves, host_reserve_bytes=host_reserve,
                                  source_cold=it._windowed_source_cold(), source_record_bytes=rec)
    ls._admit(column, (ny, nx), plan["device_bytes"], "lifecycle CUDA preparation")  # DEVICE, before any copy
    if host_fn is None:  # a caller-supplied callable is honoured whatever its truthiness (a zero-budget callable may be False-valued)
        host_fn = ss.default_host_available_bytes
    try:
        ls._admit_host_build(plan["host_bytes"], host_fn)  # HOST RAM (cold stages max incl. retained leaves), separately
    except ls.LayeredMemoryError as exc:
        raise ls.LayeredMemoryError(f"lifecycle CUDA preparation (cold host stage included): {exc}") from None
    except ls.LayeredStormError as exc:
        raise LifecycleError(str(exc)) from None
    try:
        ss._check_dynamic(column, "cover", cover, (N,))
    except ss.SubsurfaceError as exc:
        raise LifecycleError(str(exc)) from None
    try:
        # cached package pin + imported module/file association (NOT a fresh per-step hash), selected by the EXPECTED family of the requested profile
        if profile in wlc.CUDA_PROFILES:
            source = it._verify_windowed_source()  # native39 pair: the unchanged historical verification
        else:
            source = it._verify_cuda_source(wlc.general_profile_of(profile), available_bytes_fn=host_fn, reserve_bytes=host_reserve)
    except (it.GeneralIntereventError, iw.InterEventWaterError) as exc:
        family = "native39" if profile in wlc.CUDA_PROFILES else "native9f"
        raise LifecycleError(f"the bound native source is not the pinned {family} package: {exc}") from None
    voxel = sys.modules.get(_sb.MAPLE_VOXEL_MODULE)
    if voxel is None or getattr(voxel, "VoxelColumnState", None) is not VoxelColumnState:
        raise LifecycleError("the native Dense voxel class binding is missing or rebound; refused")
    try:
        owned_cover, cover_sha = wlc.cold_cover_ownership(xp, cover, N)
    except wlc.CudaLifecycleError as exc:
        raise LifecycleError(str(exc)) from None
    lay = column.layers
    host_sat = ev._immutable_host(np.asarray(lay.saturation_m, dtype=np.float64) + np.asarray(lay.capacity_tolerance_m, dtype=np.float64))
    sat_dev = xp.ascontiguousarray(xp.array(host_sat, dtype=np.float64))
    cap = wlc.CudaLifecycleCapability(
        profile=profile, device_id=getattr(column, "device_id", None), source=source, dense_class=VoxelColumnState,
        voxel_module=_sb.MAPLE_VOXEL_MODULE, cover=owned_cover, saturation_xp=sat_dev, host_saturation=host_sat,
        host_available_bytes_fn=host_fn, host_reserve_bytes=host_reserve, memory=cm, seal=_SEAL)
    h = hashlib.sha256(profile.encode())  # the profile literal seeds the identity: the storm-first and boundary profiles never alias
    part = evaporation.partition
    for piece in (column.backend, str((ny, nx)), repr(float(topology.dx_m)), column.layers.identity()["sha256"],
                  preferential.sha256, matrix.sha256, topology.topology_sha256, str(graph.input_sha256),
                  repr(control_values), str(K), cover_sha, json.dumps(asdict(part), sort_keys=True, allow_nan=False),
                  se.PARTITION_LEVEL, str(extra), str(native_reserve), st.VERSION, "dense" if bed is not None else "bedless",
                  source.package_sha256, str(host_reserve), wlc.general_profile_of(profile)):
        h.update(piece.encode())
        h.update(b"\0")
    return LifecycleOperator(
        _SEAL, column=column, preferential=preferential, matrix=matrix, router=router, graph=graph, topology=topology,
        context=context, evaporation=evaporation, t_op=None, _cover=None, cover_sha256=cover_sha,
        _material_ids=material_ids, backend=column.backend, xp=xp, shape=(ny, nx), n_cells=N, n_layers=L, n_classes=K,
        extra_reserve_bytes=extra, native_reserve_bytes=native_reserve, memory=memory, sha256=h.hexdigest(),
        _control_values=control_values, _cuda=cap)


# --- payloads, receipts, state -----------------------------------------------------------------------------------------
class BoundaryPayload:
    """The closed audited physical arrays of a boundary (surface, signed faces, soil, history J + lifetime intake, native
    bed). Factory-only: minted by this module after a genuine validation; arrays are shared by reference with the previous
    role where unchanged. Immutable."""

    __slots__ = ("__weakref__", "bed", "history", "qx_m2_s", "qy_m2_s", "soil", "surface_m", "t_s")

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise LifecycleError("BoundaryPayload is minted only by the lifecycle (start/close/reset)")
        _take(self, fields, skip=("__weakref__",), kind="payload")
        _MINTED_PAYLOADS.add(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("BoundaryPayload is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("BoundaryPayload is immutable")


@dataclass(frozen=True)
class PhaseReceipt:
    """Bounded scalar record of the last closed/entered phase (never arrays or a chain)."""

    kind: str
    episode: int
    t_start_s: float
    t_end_s: float
    hydro_steps: int
    local_ledger: dict
    source_digest: str | None
    parent_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_ledger", _freeze(copy.deepcopy(_thaw(self.local_ledger))))

    def as_dict(self) -> dict[str, Any]:
        return {f: _thaw(getattr(self, f)) for f in self.__dataclass_fields__}

    def digest(self) -> str:
        return _digest("maple_syrup.water_lifecycle.phase_receipt/1", self.as_dict())


@dataclass(frozen=True)
class ResetReceipt:
    """Bounded scalar source-bound receipt of one explicit controlled wetting-history reset."""

    parent_digest: str
    t_s: float
    episode: int
    control_identity: str
    old_wetting_sha256: str
    new_wetting_sha256: str
    lifetime_intake_sha256: str
    n_active_wet_cells_before: int
    wetting_sum_before_m: float
    quiet_policy: dict
    quiet_evidence: dict
    revision_before: int
    revision_after: int
    reset_count_after: int
    matrix_sha256: str

    def __post_init__(self) -> None:
        for f in ("quiet_policy", "quiet_evidence"):
            object.__setattr__(self, f, _freeze(copy.deepcopy(_thaw(getattr(self, f)))))

    def as_dict(self) -> dict[str, Any]:
        return {f: _thaw(getattr(self, f)) for f in self.__dataclass_fields__}

    def digest(self) -> str:
        return _digest("maple_syrup.water_lifecycle.reset_receipt/1", self.as_dict())


_STATE_SCALARS = ("role", "t_s", "revision", "episode", "reset_count", "request_sequence", "storm_operator_sha256",
                  "lop_sha256")


class LifecycleState:
    """One accepted lifecycle instant. Factory-only and immutable; carries exactly ONE current physical role payload plus the
    phase `binding`, the `lifetime` ledger, the `mark` (last accounted local ledger), counters and bounded receipts. The scalar
    `digest` detects an in-place edit; it is NOT a physical branch identity (see `expect_parent`)."""

    __slots__ = ("__weakref__", "binding", "daily", "digest", "last_control_receipt", "last_phase_receipt", "lifetime", "mark",
                 "payload", *_STATE_SCALARS)

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise LifecycleError("LifecycleState is minted only by the lifecycle functions")
        fields.setdefault("daily", None)  # Phase 1.5: the optional bounded daily cursor; every Phase 1 constructor/restore leaves it None
        _take(self, fields, skip=("__weakref__", "digest"), kind="state")
        object.__setattr__(self, "digest", _state_digest(self))
        _MINTED.add(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("LifecycleState is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("LifecycleState is immutable")

    def __repr__(self) -> str:
        return (f"LifecycleState(role={self.role!r}, t_s={self.t_s!r}, revision={self.revision}, episode={self.episode}, "
                f"resets={self.reset_count}, next_request_sequence={self.request_sequence})")

    @property
    def hydro_count(self) -> int:
        return int(self.lifetime.n_hydro_steps)

    @property
    def n_requests(self) -> int:
        return int(self.lifetime.n_t_requests)


class _Binding:
    """Per-phase fixed-component bundle; dropped at every phase close (never chained)."""

    __slots__ = ("event", "gop", "op", "prep")

    def __init__(self, op=None, event=None, prep=None, gop=None):
        self.op, self.event, self.prep, self.gop = op, event, prep, gop


def _payload_scalars(role: str, payload: Any) -> Any:
    if role == "boundary":
        return None
    if role == "storm":
        w = payload.water
        return {"generation": int(w.generation), "accepted": int(payload.accepted_steps), "completed": bool(payload.completed),
                "ledger": asdict(w.ledger), "event": payload.event_sha256}
    return {"digest": payload.digest}


def _state_digest(s: Any) -> str:
    body = {name: getattr(s, name) for name in _STATE_SCALARS}
    body["lifetime"] = asdict(s.lifetime)
    body["mark"] = asdict(s.mark)
    body["payload"] = _payload_scalars(s.role, s.payload)
    body["phase"] = None if s.last_phase_receipt is None else s.last_phase_receipt.digest()
    body["control"] = None if s.last_control_receipt is None else s.last_control_receipt.digest()
    if s.daily is not None:  # only a daily lineage carries (and digests) a cursor: every Phase 1 state digest is unchanged
        body["daily"] = s.daily.digest()
    return _digest("maple_syrup.water_lifecycle.state/1", body)


def _mint(parent: LifecycleState, **over: Any) -> LifecycleState:
    f = {name: getattr(parent, name) for name in _STATE_SCALARS}
    f.update(binding=parent.binding, daily=parent.daily, last_control_receipt=parent.last_control_receipt,
             last_phase_receipt=parent.last_phase_receipt, lifetime=parent.lifetime, mark=parent.mark, payload=parent.payload)
    f.update(over)
    return LifecycleState(_SEAL, **f)


class ExpectedParent:
    """Non-owning in-memory capability naming the exact physical parent INSTANCE: two equal-scalar branches differ. Minted by
    `expect_parent`; it never keeps the parent alive."""

    __slots__ = ("__weakref__", "_ref", "digest", "revision", "role")

    def __init__(self, seal: object, state: LifecycleState) -> None:
        if seal is not _SEAL:
            raise LifecycleError("ExpectedParent is minted only by expect_parent()")
        object.__setattr__(self, "_ref", weakref.ref(state))
        object.__setattr__(self, "digest", state.digest)
        object.__setattr__(self, "revision", int(state.revision))
        object.__setattr__(self, "role", state.role)
        _MINTED_EXPECTED.add(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ExpectedParent is immutable")


def expect_parent(state: LifecycleState) -> ExpectedParent:
    if type(state) is not LifecycleState or state not in _MINTED:
        raise LifecycleError("expect_parent needs a genuine LifecycleState")
    return ExpectedParent(_SEAL, state)


def _require_parent(state: Any, expected: Any) -> None:
    if type(state) is not LifecycleState or state not in _MINTED:
        raise LifecycleError(f"parent must be a genuine LifecycleState, got {type(state).__name__}")
    if type(expected) is not ExpectedParent or expected not in _MINTED_EXPECTED:
        raise LifecycleError("expected_parent must be an ExpectedParent minted by expect_parent()")
    if expected._ref() is not state:
        raise LifecycleError("the expected parent is a different physical instance (an alternate branch can share every "
                             "scalar), or it is gone; refused before any trial")
    if expected.digest != state.digest or expected.revision != state.revision or expected.role != state.role:
        raise LifecycleError("the expected-parent capability no longer matches the parent's scalars; refused")


# --- physical validation -----------------------------------------------------------------------------------------------
def _arrays_of(s: LifecycleState) -> SimpleNamespace:
    p = s.payload
    if s.role == "boundary":
        return SimpleNamespace(t_s=p.t_s, surface_m=p.surface_m, qx=p.qx_m2_s, qy=p.qy_m2_s, soil=p.soil,
                               history=p.history, bed=p.bed)
    if s.role == "storm":
        w = p.water
        return SimpleNamespace(t_s=w.t_s, surface_m=w.surface_m, qx=w.qx_m2_s, qy=w.qy_m2_s, soil=w.soil, history=w.history,
                               bed=w.bed)
    return SimpleNamespace(t_s=p.t_s, surface_m=p.surface_m, qx=p.qx_m2_s, qy=p.qy_m2_s, soil=p.soil, history=p.history,
                           bed=p.bed)


def _saturation_bound(lop: LifecycleOperator) -> np.ndarray:
    if lop._cuda is not None:
        return lop._cuda.saturation_xp  # the private device (L,) bound (no per-layer host read)
    lay = lop.column.layers
    return np.asarray(lay.saturation_m, dtype=np.float64) + np.asarray(lay.capacity_tolerance_m, dtype=np.float64)


def _check_physical_meta(lop: LifecycleOperator, a: SimpleNamespace) -> float:
    """Array metadata, bindings and aliases of one physical role (no element read)."""
    column = lop.column
    ny, nx = lop.shape
    try:
        ss._check_dynamic(column, "surface_m", a.surface_m, (ny, nx))
        ss._check_dynamic(column, "qx_m2_s", a.qx, (ny, nx + 1))
        ss._check_dynamic(column, "qy_m2_s", a.qy, (ny + 1, nx))
        t = ss._check_state(column, a.soil)
        if t != _real(a.t_s, "t_s", nonneg=True):
            raise LifecycleError(f"soil.t_s = {t!r} disagrees with the role clock {a.t_s!r}; one clock, never restamped")
        mi._check_history(lop.matrix, a.history)
    except (ss.SubsurfaceError, mi.MatrixError) as exc:
        raise LifecycleError(str(exc)) from None
    if lop.context is None:
        if a.bed is not None:
            raise LifecycleError("the lifecycle holds no native bed but the role carries one; refused")
    else:
        if type(a.bed) is not BedState or type(a.bed.voxel_column) is not VoxelColumnState:
            raise LifecycleError("the role's bed must be a genuine native Dense BedState; refused")
        if a.bed.water.depth_m is not a.surface_m:
            raise LifecycleError("alias contract violated: bed.water.depth_m must BE the surface array (one authoritative "
                                 "surface)")
        if ls._bed_material_ids(a.bed) != lop._material_ids:
            raise LifecycleError("the role's native bed material differs from the lifecycle's sealed material; refused")
    return float(a.t_s)


def _validate_inventories(lop: LifecycleOperator, a: SimpleNamespace) -> None:
    """Physical bounds on EVERY cell (inactive included): soil finite, >= 0, <= saturation + capacity tolerance; J and lifetime
    intake finite, >= 0; the surface/faces by the router. Nothing is clipped or reset."""
    if lop._cuda is not None:
        wlc.validate_inventories_xp(lop, a, LifecycleError)  # ONE bounded host packet; every cell incl. inactive; no clipping
    else:
        w = a.soil.water_m
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise LifecycleError("soil water must be finite and >= 0 on every cell (inactive included)")
        if np.any(w > _saturation_bound(lop)[:, None]):
            raise LifecycleError("soil water exceeds the layer saturation depth (+ MAPLE capacity tolerance) on a cell")
        for name, arr in (("wetting_m", a.history.wetting_m), ("cumulative_intake_m", a.history.cumulative_intake_m)):
            if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
                raise LifecycleError(f"history {name} must be finite and >= 0 on every cell")
    try:
        lop.router.validate_state(SurfaceState(float(a.t_s), a.surface_m, a.qx, a.qy))
    except ExperimentalHydrologyError as exc:
        raise LifecycleError(f"surface/faces refused by the router: {exc}") from None


def _inventories(lop: LifecycleOperator, a: SimpleNamespace) -> tuple[float, float]:
    if lop._cuda is not None:
        return wlc.inventories_xp(lop, a, LifecycleError)  # active-only volumes in ONE two-scalar packet
    area = lop.column.cell_area_m2
    active = lop.column.active
    h = float(np.sum(np.where(active.reshape(lop.shape), a.surface_m, 0.0))) * area
    soil = float(np.sum(np.where(active.reshape(1, -1), a.soil.water_m, 0.0))) * area
    if not (math.isfinite(h) and math.isfinite(soil)) or h < 0.0 or soil < 0.0:
        raise LifecycleError("non-finite/negative physical inventory volumes")
    return h, soil


def _local_mark(s: LifecycleState) -> Watermark:
    if s.role == "storm":
        return _mark_from_storm(s.payload.water.ledger)
    if s.role == "interevent":
        return _mark_from_general(s.payload.ledger)
    return s.mark  # a boundary has no local ledger: its mark is carried unchanged


def _check_state(lop: LifecycleOperator, s: Any) -> float:
    """Genuine/minted/scalar-digest, profile and counters, role payload and the watermark invariant. Metadata only."""
    _check_lop(lop)
    if type(s) is not LifecycleState or s not in _MINTED:
        raise LifecycleError(f"state must be a LifecycleState minted by this module, got {type(s).__name__}")
    try:
        if _state_digest(s) != s.digest:
            raise LifecycleError("the lifecycle state's scalars/ledgers were altered after it was minted; refused")
    except (ValueError, TypeError, AttributeError) as exc:
        raise LifecycleError(f"the lifecycle state carries a blank, non-finite or foreign-typed scalar: {exc}") from None
    if s.lop_sha256 != lop.sha256:
        raise LifecycleError("the state is bound to a different lifecycle operator; refused")
    if s.role not in ("boundary", "storm", "interevent"):
        raise LifecycleError("unknown lifecycle role; refused")
    if s.daily is not None:
        if type(s.daily) is not rc.DailyCursor:
            raise LifecycleError("state.daily must be a genuine DailyCursor; refused")
        if (s.role == "storm") != (s.daily.pending is not None):
            raise LifecycleError("a daily lineage's storm state carries exactly its pending storm occurrence (and no other role does); refused")
    for name in ("revision", "episode", "reset_count", "request_sequence"):
        _count(getattr(s, name), "state." + name)
    _check_lifetime(s.lifetime)
    if type(s.mark) is not Watermark:
        raise LifecycleError("state.mark must be a genuine Watermark")
    if s.request_sequence != s.lifetime.n_t_requests:
        raise LifecycleError("the request sequence disagrees with the accepted lifetime request count; refused")
    if s.storm_operator_sha256 is not None and (type(s.storm_operator_sha256) is not str or len(s.storm_operator_sha256) != 64):
        raise LifecycleError("invalid storm-operator identity; refused")
    if s.role == "boundary":
        if type(s.payload) is not BoundaryPayload or s.payload not in _MINTED_PAYLOADS or s.binding is not None:
            raise LifecycleError("a boundary state needs a minted BoundaryPayload and no phase binding; refused")
        t = _check_physical_meta(lop, _arrays_of(s))
    elif s.role == "storm":
        b = s.binding
        if type(b) is not _Binding or b.op is None or b.event is None or b.prep is not None or b.gop is not None:
            raise LifecycleError("a storm state needs exactly its storm operator and event binding; refused")
        try:
            ev._check_state(b.event, s.payload)
        except (ev.LayeredEventError, ls.LayeredStormError) as exc:
            raise LifecycleError(f"the storm payload failed its genuine checks: {exc}") from None
        if b.event.op is not b.op or b.op.sha256 != s.storm_operator_sha256:
            raise LifecycleError("the storm binding disagrees with the lifecycle's storm-operator identity; refused")
        t = float(s.payload.water.t_s)
    else:
        b = s.binding
        if type(b) is not _Binding or b.gop is None or b.prep is None or b.op is None or b.event is not None:
            raise LifecycleError("an interevent state needs exactly its storm operator and interevent owners; refused")
        try:
            t = float(it._check_general_state(b.gop, s.payload))
        except it.GeneralIntereventError as exc:
            raise LifecycleError(f"the interevent payload failed its genuine checks: {exc}") from None
        if b.prep.op is not b.op or b.gop.base is not b.prep or b.op.sha256 != s.storm_operator_sha256:
            raise LifecycleError("the interevent binding disagrees with the lifecycle's identities; refused")
        if lop._cuda is not None and getattr(b.gop.cuda, "profile", None) != wlc.general_profile_of(lop._cuda.profile):
            raise LifecycleError("the bound general owner's CUDA profile is not the one this lifecycle's profile requires (no cross-family binding); refused")
        if s.payload.next_request_sequence != s.request_sequence:
            raise LifecycleError("the interevent request cursor disagrees with the lifecycle sequence; refused")
        _check_physical_meta(lop, _arrays_of(s))
    if t != float(s.t_s):
        raise LifecycleError("the lifecycle clock disagrees with its payload clock; refused")
    if _local_mark(s) != s.mark:
        raise LifecycleError("the local ledger and the accounted watermark disagree: unaccounted or doubly accounted flux; "
                             "refused")
    return t


def audit_lifecycle(lop: LifecycleOperator, state: LifecycleState) -> dict[str, Any]:
    """Actual inventories (fresh reductions over the raw arrays) against the run anchors plus the lifetime fluxes:

        (S - S0) + export + drainage + surfaceE + soilE + actualT - rain = 0   within the derived FP64 bound."""
    _check_lop(lop)
    _admit(lop, required_audit_bytes(lop, state), {"interevent": "audit_interevent", "storm": "audit_storm"}.get(getattr(state, "role", None), "audit"),
           parent=state)  # BEFORE any inventory validation or read
    t = _check_state(lop, state)
    a = _arrays_of(state)
    _validate_inventories(lop, a)
    return _audit_core(lop, state, a, t)


def _audit_core(lop: LifecycleOperator, state: LifecycleState, a: SimpleNamespace, t: float) -> dict[str, Any]:
    h, soil = _inventories(lop, a)
    lt = state.lifetime
    residual = (((((h - lt.anchor_surface_m3) + (soil - lt.anchor_soil_m3)) + lt.export_m3) + lt.drainage_m3
                 + lt.surface_evaporation_m3 + lt.soil_evaporation_m3) + lt.transpiration_m3) - lt.rain_m3
    bound = _lifetime_bound(lop, lt, h, soil)
    if not math.isfinite(residual) or abs(residual) > bound:
        raise LifecycleError(f"lifetime water closure failed: {residual} m3 against {bound} m3")
    return dict(t_s=t, surface_m3=h, soil_m3=soil, anchor_surface_m3=lt.anchor_surface_m3, anchor_soil_m3=lt.anchor_soil_m3,
                rain_m3=lt.rain_m3, export_m3=lt.export_m3, drainage_m3=lt.drainage_m3,
                surface_evaporation_m3=lt.surface_evaporation_m3, soil_evaporation_m3=lt.soil_evaporation_m3,
                transpiration_m3=lt.transpiration_m3, residual_m3=residual, bound_m3=bound)


# --- start -------------------------------------------------------------------------------------------------------------
def start_lifecycle(lop: LifecycleOperator, *, soil: ss.SoilColumnState, history: mi.WettingHistory, t_s: float,
                    bed: Any = None, surface_m: Any = None, qx_m2_s: Any = None, qy_m2_s: Any = None) -> LifecycleState:
    """The run start: a closed boundary whose physical inventories become the lifetime anchors S0. With a native Dense bed the
    surface IS `bed.water.depth_m` (`surface_m` must be None); otherwise `surface_m` is an exact (ny, nx) array, copied once.
    Faces None = at rest. The soil and history objects are carried by reference. Admitted before any copy."""
    _check_lop(lop)
    t = _real(t_s, "t_s", nonneg=True)
    ny, nx = lop.shape
    _admit(lop, lop.memory.start_bytes, "start")
    if lop.context is not None:
        if surface_m is not None or bed is None:
            raise LifecycleError("the lifecycle holds a native bed: pass `bed` (its water IS the surface), not surface_m")
        surface = bed.water.depth_m if type(bed) is BedState else None
        if surface is None:
            raise LifecycleError("bed must be a genuine BedState")
    else:
        if bed is not None or surface_m is None:
            raise LifecycleError("bedless lifecycle: pass `surface_m` and no bed")
        try:
            ss._check_dynamic(lop.column, "surface_m", surface_m, (ny, nx))
        except ss.SubsurfaceError as exc:
            raise LifecycleError(str(exc)) from None
        xp = lop.xp  # numpy on the CPU (the unchanged branch), the column's device namespace on CUDA: one admitted copy, no host round trip
        surface = xp.ascontiguousarray(xp.array(surface_m, dtype=np.float64, copy=True))
    xp = lop.xp
    qx = xp.zeros((ny, nx + 1)) if qx_m2_s is None else xp.ascontiguousarray(xp.array(qx_m2_s, copy=True))
    qy = xp.zeros((ny + 1, nx)) if qy_m2_s is None else xp.ascontiguousarray(xp.array(qy_m2_s, copy=True))
    payload = BoundaryPayload(_SEAL, bed=bed, history=history, qx_m2_s=qx, qy_m2_s=qy, soil=soil, surface_m=surface, t_s=t)
    a = _arrays_of(SimpleNamespace(role="boundary", payload=payload))
    _check_physical_meta(lop, a)
    _validate_inventories(lop, a)
    h, s0 = _inventories(lop, a)
    lt = LifetimeLedger(anchor_surface_m3=h, anchor_soil_m3=s0, anchor_t_s=t)
    return LifecycleState(
        _SEAL, role="boundary", t_s=t, revision=0, episode=0, reset_count=0, request_sequence=0, storm_operator_sha256=None,
        lop_sha256=lop.sha256, binding=None, last_control_receipt=None, last_phase_receipt=None, lifetime=lt,
        mark=Watermark(), payload=payload)


# --- storm -------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LifecycleStormResult:
    state: LifecycleState
    reason: str
    accepted_steps: int
    rejected_attempts: int

    @property
    def complete(self) -> bool:
        return bool(self.state.payload.completed)


def _prepare_current_operator(lop: LifecycleOperator, parent: LifecycleState) -> ls.LayeredStormOperator:
    """The fresh real storm operator of the CURRENT boundary (shared by `begin_storm` and the boundary interevent entry): same
    fixed components, prepared on the CURRENT native bed (so its retained reference depth is the current H, never an original).
    It is a fixed-component binding only: no event, no initial-state initializer and no ledger exists here. Its identity must
    equal the lifecycle's when one is recorded (changed material/topology/properties refused, never retagged)."""
    pay = parent.payload
    if lop.context is None:
        op = ls.prepare_layered_storm(lop.column, lop.preferential, lop.matrix, lop.router, lop.graph, lop.topology,
                                      extra_reserve_bytes=lop.extra_reserve_bytes, **_host_kw(lop))
    else:
        op = ls.prepare_layered_storm(lop.column, lop.preferential, lop.matrix, lop.router, lop.graph, lop.topology,
                                      bed=pay.bed, context=lop.context, extra_reserve_bytes=lop.extra_reserve_bytes,
                                      native_reserve_bytes=lop.native_reserve_bytes, **_host_kw(lop))
    if parent.storm_operator_sha256 is not None and op.sha256 != parent.storm_operator_sha256:
        raise LifecycleError("the fresh storm operator's identity differs from the lifecycle's: the fixed material/topology/"
                             "properties changed (no retag, no silent rebind); refused")
    return op


_DAILY_PERMIT: contextvars.ContextVar = contextvars.ContextVar("maple_syrup_daily_engine_permit", default=None)


class _DailyPermit:
    """Module-private engine permit: while active for ONE exact parent instance, the three Phase 1 transitions run on that TRUE cursor-carrying
    parent (their bodies are unchanged and receive the real parent and ExpectedParent) and stamp the STAGED result with the cursor the daily
    contract prescribes (`daily_for(payload)`), so role/cursor checks stay strict. It is a context variable, not a flag on the state or a public
    argument; ordinary public calls on a daily state still refuse."""

    __slots__ = ("daily_for", "parent")

    def __init__(self, parent: Any, daily_for: Any) -> None:
        self.parent, self.daily_for = parent, daily_for


@contextlib.contextmanager
def _daily_engine(parent: Any, daily_for: Any):
    token = _DAILY_PERMIT.set(_DailyPermit(parent, daily_for))
    try:
        yield
    finally:
        _DAILY_PERMIT.reset(token)


def _daily_over(parent: Any, payload: Any) -> dict[str, Any]:
    """The staged cursor of a permitted daily engine call ({} for every ordinary call: nothing is overridden)."""
    permit = _DAILY_PERMIT.get()
    if permit is None or permit.parent is not parent or getattr(parent, "daily", None) is None:
        return {}
    return {"daily": permit.daily_for(payload)}


def _refuse_daily(parent: Any, what: str) -> None:
    """A Phase 1.5 DAILY lineage (its state carries a cursor) is advanced only by the daily transactions (`daily_water`, which holds the private
    engine permit for the exact parent): the old freely callable transition functions would add a day, a storm or an entry behind the cursor and
    are refused before any work. A state WITHOUT a cursor (every Phase 1 lineage) takes exactly the unchanged path."""
    if getattr(parent, "daily", None) is None:
        return
    permit = _DAILY_PERMIT.get()
    if permit is not None and permit.parent is parent:
        return
    raise LifecycleError(f"{what} is not available on a daily lineage (its state carries a daily cursor); use the daily_water transactions")


def begin_storm(lop: LifecycleOperator, parent: LifecycleState, *, expected_parent: ExpectedParent, schedule: Any, field: Any,
                control: ev.LayeredEventControl = ev.LayeredEventControl(),
                policy: ev.LayeredQuietPolicy = ev.LayeredQuietPolicy()) -> LifecycleState:
    """Begin the next genuine storm at the CURRENT boundary instant. Carry is the default: the actual soil, retained surface
    water H, signed faces, J and lifetime intake continue; the storm-local ledger/generation/forcing/quiet progress are fresh
    (phase-local). A fresh real storm operator is prepared on the CURRENT native bed (so `op.bed.water.depth_m` is the current H,
    never the original) with the same fixed components; its identity must equal the lifecycle's (changed material refused)."""
    _check_lop(lop)
    _require_parent(parent, expected_parent)
    _refuse_daily(parent, "begin_storm")
    if parent.role != "boundary":
        raise LifecycleError(f"begin_storm needs a closed 'boundary' parent, got {parent.role!r}")
    need = required_begin_storm_bytes(lop, schedule)
    _admit(lop, need, "begin_storm", schedule=schedule)  # BEFORE the operator, the initial state or the event are built
    _check_state(lop, parent)
    pay = parent.payload
    a = _arrays_of(parent)
    _validate_inventories(lop, a)
    op = _prepare_current_operator(lop, parent)
    water = ls.initial_layered_state(op, soil=pay.soil, history=pay.history, t_s=pay.t_s, qx_m2_s=pay.qx_m2_s,
                                     qy_m2_s=pay.qy_m2_s, surface_m=None if lop.context is not None else pay.surface_m)
    if lop.context is not None and water.surface_m is not pay.surface_m:
        raise LifecycleError("internal: the native storm state did not alias the current surface depth")
    event = ev.prepare_layered_event(op, schedule, field, event_origin_s=pay.t_s, control=control, policy=policy, **_host_kw(lop))
    estate = ev.initial_event_state(event, water)
    return _mint(parent, role="storm", revision=parent.revision + 1, episode=parent.episode + 1,
                 storm_operator_sha256=op.sha256, binding=_Binding(op=op, event=event), payload=estate,
                 mark=_mark_from_storm(water.ledger), **_daily_over(parent, estate))


def evolve_storm(lop: LifecycleOperator, parent: LifecycleState, *, expected_parent: ExpectedParent, until_s: Any,
                 max_accepted_steps: int | None = None, stop_when_quiet: bool = True) -> LifecycleStormResult:
    """Advance the actual bounded storm (rain and recession; E = T = 0) and fold ONLY the new local ledger delta. A bound stop
    (`max_steps`, `max_retries`, `min_dt`, `recession_limit`, a target short of completion) is an INCOMPLETE storm: it does not
    permit interevent E/T."""
    _check_lop(lop)
    _require_parent(parent, expected_parent)
    if parent.role != "storm":
        raise LifecycleError(f"evolve_storm needs a 'storm' parent, got {parent.role!r}")
    steps = None if max_accepted_steps is None else _count(max_accepted_steps, "max_accepted_steps")
    _real(until_s, "until_s")
    if type(stop_when_quiet) is not bool:
        raise LifecycleError("stop_when_quiet must be bool")
    _admit(lop, required_evolve_storm_bytes(lop, parent), "evolve_storm", parent=parent)  # BEFORE the event reads or allocates anything
    _check_state(lop, parent)
    max_accepted_steps = steps
    try:
        res = ev.evolve_layered_event(parent.binding.event, parent.payload, until_s=until_s,
                                      max_accepted_steps=max_accepted_steps, stop_when_quiet=stop_when_quiet)
    except (ev.LayeredEventError, ls.LayeredStormError) as exc:
        raise LifecycleError(f"storm evolution refused: {exc}") from None
    cur = _mark_from_storm(res.state.water.ledger)
    lt, mark = _fold(parent.lifetime, parent.mark, cur)
    new = _mint(parent, revision=parent.revision + 1, t_s=float(res.state.water.t_s), payload=res.state, lifetime=lt, mark=mark)
    _audit_core(lop, new, _arrays_of(new), new.t_s)
    return LifecycleStormResult(new, res.reason, int(res.accepted_steps), int(res.rejected_attempts))


# --- interevent --------------------------------------------------------------------------------------------------------
def enter_interevent_phase(lop: LifecycleOperator, parent: LifecycleState, *, expected_parent: ExpectedParent) -> LifecycleState:
    """Enter the interevent exactly once from an ACTUALLY completed storm (reusing the genuine `enter_interevent` and
    `enable_requested_transpiration`). The entered ledger already contains the storm totals (prefix), which the lifecycle has
    folded during the storm: the new watermark is the CONVERTED STORM-END ledger, so nothing is recounted. The T role continues
    the lifetime request sequence."""
    _check_lop(lop)
    _require_parent(parent, expected_parent)
    _refuse_daily(parent, "enter_interevent_phase")
    if parent.role != "storm":
        raise LifecycleError(f"enter_interevent_phase needs a 'storm' parent, got {parent.role!r}")
    if parent.payload.completed is not True:
        raise LifecycleError("the storm is not completed (a bound-stopped or flowing storm is still rain/recession and "
                             "receives no interevent E/T); refused before any owner is prepared")
    need = required_enter_interevent_bytes(lop, parent)
    _admit(lop, need, "enter_interevent", parent=parent)  # CUDA: the authoritative plan (actual no-supplied-operator branch, separate host stage)
    _check_state(lop, parent)
    op, event = parent.binding.op, parent.binding.event
    try:
        prep = iw.prepare_interevent_water(op, lop.evaporation, _private_cover(lop), **_host_kw(lop))
        t0 = iw.enter_interevent(prep, event, parent.payload)
        if lop._cuda is None:
            gop = it.prepare_general_interevent(prep, lop.t_op)
        else:  # no supplied operator: the owner mints its genuine primitive, canonical bounds and kernel binding (cuda_profile opt-in)
            gop = it.prepare_general_interevent(prep, cuda_profile=it.CUDA_PROFILE if lop._cuda.profile in wlc.CUDA_PROFILES else it.CUDA9F_PROFILE,
                                           **_host_kw(lop))
        g0 = it.enable_requested_transpiration(gop, t0, expected_t_s=t0.t_s, expected_generation=t0.generation,
                                               initial_sequence=parent.request_sequence)
    except (iw.InterEventWaterError, it.GeneralIntereventError, ls.LayeredStormError) as exc:
        raise LifecycleError(f"interevent entry refused: {exc}") from None
    mark = _mark_from_general(g0.ledger)
    if mark != parent.mark:
        raise LifecycleError("the converted storm-end ledger differs from the accounted storm watermark (the prefix would be "
                             "recounted or lost); refused")
    receipt = PhaseReceipt("storm", int(parent.episode), float(event.event_origin_s), float(parent.t_s),
                           int(parent.payload.accepted_steps), asdict(parent.payload.water.ledger), t0.source.digest,
                           parent.digest)
    return _mint(parent, role="interevent", revision=parent.revision + 1, binding=_Binding(op=op, prep=prep, gop=gop),
                 payload=g0, mark=mark, last_phase_receipt=receipt, **_daily_over(parent, g0))


def enter_interevent_from_boundary(lop: LifecycleOperator, parent: LifecycleState, *, expected_parent: ExpectedParent,
                                   policy: ev.LayeredQuietPolicy = ev.LayeredQuietPolicy()) -> LifecycleState:
    """Enter an interevent from a genuine initialized or previously CLOSED 'boundary' (a schedule may begin with an interevent or
    run consecutive interevents), with no storm, no dummy zero-rain event and no completed-flag fabrication.

    The fresh real storm operator is prepared on the CURRENT native bed (the shared `begin_storm` helper: no event, no initializer,
    no ledger reset); the interevent and general owners are prepared; the genuine boundary factory
    (`interevent_water.enter_interevent_from_boundary`) audits the ACTUAL current inventories, history, native alias/material,
    router state and the instantaneous hydraulic quiet gates under the explicit `policy` (an unquiet boundary is refused whatever
    its role name; a quiet retained pond is carried); requested T is enabled continuing the lifetime request sequence.

    The result advances ONLY `revision` (and the interevent role/binding and the storm-operator identity binding): `t_s`, `episode`,
    `reset_count`, the lifetime anchors/totals/counts/folds and `request_sequence`, and every physical array/J/intake/bed are
    unchanged (shared by reference, bitwise). The local watermark is the actual new all-zero local ledger, so the previous closed
    phase is never folded again. The phase receipt honestly says `boundary_origin`. CPU Dense/bedless, and CUDA Dense/bedless only for a lifecycle prepared with the
    explicit CUDA_BOUNDARY_PROFILE (the storm-first profile refuses it before any work); qualified on the actual native39 source only (the explicit native9f
    CUDA9F_BOUNDARY_PROFILE is the separate UNQUALIFIED counterpart); Windowed
    storage has genuine early typed fences. Other sources, MASY/LEAF/calendar adapters and the old lifecycle
    codec are NOT qualified for this entry (no literal fence or typed refusal object is claimed). Pure: the parent is never written."""
    _check_lop(lop)
    cuda = lop._cuda
    if lop._cuda is not None and not lop._cuda.boundary_capable:
        raise LifecycleError("initialized-boundary interevent entry is CPU-only in the storm-first CUDA profile (explicit fence, refused before any "
                             "preparation, hash or physical work; prepare the lifecycle with the separate explicit CUDA_BOUNDARY_PROFILE; a zero-rain "
                             "storm is never used to emulate it)")
    _require_parent(parent, expected_parent)
    _refuse_daily(parent, "enter_interevent_from_boundary")
    if parent.role != "boundary":
        raise LifecycleError(f"enter_interevent_from_boundary needs a closed 'boundary' parent, got {parent.role!r}")
    if type(policy) is not ev.LayeredQuietPolicy:
        raise LifecycleError("policy must be a genuine LayeredQuietPolicy")
    try:
        policy.validated()
    except (ev.LayeredEventError, ls.LayeredStormError) as exc:
        raise LifecycleError(f"invalid quiet policy: {exc}") from None
    need = 0 if cuda is not None else required_enter_from_boundary_bytes(lop, parent)  # CUDA: the authoritative plan replaces the figure in `_admit`
    _admit(lop, need, "enter_interevent_from_boundary", parent=parent)  # BEFORE the operator, any hash, any reduction
    _check_state(lop, parent)
    a = _arrays_of(parent)
    try:
        op = _prepare_current_operator(lop, parent)
        prep = iw.prepare_interevent_water(op, lop.evaporation, _private_cover(lop), **_host_kw(lop))
        extra = {} if cuda is None else {"cuda_boundary_profile": iw.CUDA_BOUNDARY_ENTRY_PROFILE}
        t0 = iw.enter_interevent_from_boundary(prep, t_s=a.t_s, surface_m=a.surface_m, qx_m2_s=a.qx, qy_m2_s=a.qy, soil=a.soil,
                                               history=a.history, bed=a.bed, policy=policy, parent_token=parent.digest,
                                               parent_revision=parent.revision, **extra)
        if cuda is None:
            gop = it.prepare_general_interevent(prep, lop.t_op)
        else:  # the genuine CUDA general owner: its own operator/canonical bounds/kernel/source guards; no supplied primitive
            gop = it.prepare_general_interevent(prep, cuda_profile=it.CUDA_PROFILE if lop._cuda.profile in wlc.CUDA_PROFILES else it.CUDA9F_PROFILE,
                                           **_host_kw(lop))
        g0 = it.enable_requested_transpiration(gop, t0, expected_t_s=t0.t_s, expected_generation=t0.generation,
                                               initial_sequence=parent.request_sequence)
    except (iw.InterEventWaterError, it.GeneralIntereventError, ls.LayeredStormError) as exc:
        raise LifecycleError(f"boundary interevent entry refused: {exc}") from None
    mark = _mark_from_general(g0.ledger)
    if mark != Watermark():
        raise LifecycleError("internal: the new local interevent ledger is not exactly zero; refused")
    receipt = PhaseReceipt("boundary_origin", int(parent.episode), float(parent.t_s), float(parent.t_s), 0,
                           g0.ledger.as_dict(), t0.source.digest, parent.digest)
    new = _mint(parent, role="interevent", revision=parent.revision + 1, storm_operator_sha256=op.sha256,
                binding=_Binding(op=op, prep=prep, gop=gop), payload=g0, mark=mark, last_phase_receipt=receipt, **_daily_over(parent, g0))
    _check_state(lop, new)
    _audit_core(lop, new, _arrays_of(new), new.t_s)
    return new


@dataclass(frozen=True)
class LifecycleIntereventResult:
    state: LifecycleState
    reason: str
    accepted_steps: int
    rejected_attempts: int


def _fold_interevent(lop: LifecycleOperator, parent: LifecycleState, payload: Any, *, revision_inc: int = 1, **over: Any) -> LifecycleState:
    cur = _mark_from_general(payload.ledger)
    lt, mark = _fold(parent.lifetime, parent.mark, cur)
    new = _mint(parent, revision=parent.revision + revision_inc, t_s=float(payload.t_s),
                request_sequence=int(payload.next_request_sequence), payload=payload, lifetime=lt, mark=mark, **over)
    return new


def evolve_interevent(lop: LifecycleOperator, parent: LifecycleState, *, expected_parent: ExpectedParent, until_s: Any,
                      pet_terrain_m_s: Any, max_dt_s: Any, max_accepted_steps: int = 100_000, max_retries: int = 12,
                      min_dt_s: Any = 2.0 ** -20, retained_feedback_bytes: int = 0) -> LifecycleIntereventResult:
    """Real interevent hydrology + E to `until_s` with a CONSTANT terrain-applied PET `(N,)` (MASY owns the weather edges). Time
    advances once per accepted step; `HydraulicStepRejected` alone halves dt (the SAME accepted state reruns). One window audit
    and one lifetime fold at the end; any failure leaves the parent untouched (pure, no partial adoption). A bound stop
    (`max_steps`) returns the valid partial interevent."""
    _check_lop(lop)
    _require_parent(parent, expected_parent)
    _refuse_daily(parent, "evolve_interevent")  # a daily lineage's day is the ONE fixed 86400 s transaction, never a sub-stepped interval
    if parent.role != "interevent":
        raise LifecycleError(f"evolve_interevent needs an 'interevent' parent, got {parent.role!r}")
    end = _real(until_s, "until_s")
    max_dt = _real(max_dt_s, "max_dt_s", positive=True)
    floor = _real(min_dt_s, "min_dt_s", positive=True)
    budget, retries = _count(max_accepted_steps, "max_accepted_steps"), _count(max_retries, "max_retries")
    feedback = _count(retained_feedback_bytes, "retained_feedback_bytes")
    _admit(lop, required_evolve_interevent_bytes(lop, parent, feedback), "evolve_interevent", parent=parent, feedback=feedback)  # BEFORE state/PET/step
    _check_state(lop, parent)
    gop = parent.binding.gop
    state, n, rejected, reason = parent.payload, 0, 0, "target"
    if end < state.t_s:
        raise LifecycleError("until_s precedes the accepted physical clock")
    while state.t_s < end:
        if n >= budget:
            reason = "max_steps"
            break
        t = float(state.t_s)
        dt = min(end - t, max_dt)
        if t + dt > end:
            dt = math.nextafter(dt, 0.0)
        if not t + dt > t:
            reason = "time_resolution"
            break
        attempts = 0
        while True:
            try:
                cand = it.general_interevent_water_step(gop, state, pet_terrain_m_s, dt, phase="interevent",
                                                        retained_feedback_bytes=feedback)
                break
            except HydraulicStepRejected:
                attempts += 1
                rejected += 1
                if attempts >= retries or dt / 2.0 < floor:
                    raise LifecycleError("the interevent step could not be retried within its bounds (the parent is "
                                         "unchanged)") from None
                dt /= 2.0
            except (it.GeneralIntereventError, iw.InterEventWaterError) as exc:
                raise LifecycleError(f"interevent step refused: {exc}") from None
        state = cand.state
        del cand
        n += 1
    try:
        it.audit_general_interevent(gop, state)
    except (it.GeneralIntereventError, iw.InterEventWaterError) as exc:
        raise LifecycleError(f"interevent window audit refused: {exc}") from None
    new = _fold_interevent(lop, parent, state)
    _audit_core(lop, new, _arrays_of(new), new.t_s)
    return LifecycleIntereventResult(new, reason, n, rejected)


def request_withdrawal(lop: LifecycleOperator, parent: LifecycleState, *, expected_parent: ExpectedParent, depth_m: Any,
                       identity: Any, retained_feedback_bytes: Any = 0) -> tuple[LifecycleState, it.TranspirationCandidate]:
    """ONE zero-time T transaction at the accepted instant through the genuine general owner (the request is bound to the
    exact physical interevent parent instance). Folds only the new T delta; the returned candidate carries the actual/unmet/scale
    feedback arrays the caller may retain (declare their bytes as `retained_feedback_bytes` on later steps). Time, hydro count,
    surface, faces and J are untouched; the lifetime request sequence advances once."""
    _check_lop(lop)
    _require_parent(parent, expected_parent)
    if parent.role != "interevent":
        raise LifecycleError(f"request_withdrawal needs an 'interevent' parent, got {parent.role!r}")
    _admit(lop, required_request_bytes(lop, parent, retained_feedback_bytes), "request_withdrawal", parent=parent,
           feedback=retained_feedback_bytes)  # BEFORE the request copy
    _check_state(lop, parent)
    gop = parent.binding.gop
    try:
        req = it.prepare_withdrawal_request(gop, parent.payload, depth_m, identity=identity)
        cand = it.request_transpiration(gop, parent.payload, req, expected_revision=parent.payload.revision)
    except (it.GeneralIntereventError, iw.InterEventWaterError) as exc:
        raise LifecycleError(f"withdrawal refused: {exc}") from None
    new = _fold_interevent(lop, parent, cand.state)
    _audit_core(lop, new, _arrays_of(new), new.t_s)
    return new, cand


def close_interevent(lop: LifecycleOperator, parent: LifecycleState, *, expected_parent: ExpectedParent) -> LifecycleState:
    """Close the interevent at its actual end: full whole-water window audit and lifetime audit, a final fold (zero by
    construction: never a second credit), then a closed 'boundary' sharing the same physical arrays. Every phase owner and the
    old operator/event/receipt graph is released; only a bounded scalar `PhaseReceipt` remains. A retained pond is carried; no dry
    requirement or reset is implied."""
    _check_lop(lop)
    _require_parent(parent, expected_parent)
    if parent.role != "interevent":
        raise LifecycleError(f"close_interevent needs an 'interevent' parent, got {parent.role!r}")
    gop = parent.binding.gop
    _admit(lop, required_close_bytes(lop, parent), "close", parent=parent)
    _check_state(lop, parent)
    try:
        it.audit_general_interevent(gop, parent.payload)
    except (it.GeneralIntereventError, iw.InterEventWaterError) as exc:
        raise LifecycleError(f"interevent window audit refused at close: {exc}") from None
    cur = _mark_from_general(parent.payload.ledger)
    lt, mark = _fold(parent.lifetime, parent.mark, cur)  # == parent.lifetime + 0: the watermark already equals `cur`
    a = _arrays_of(parent)
    boundary = BoundaryPayload(_SEAL, bed=a.bed, history=a.history, qx_m2_s=a.qx, qy_m2_s=a.qy, soil=a.soil,
                               surface_m=a.surface_m, t_s=float(a.t_s))
    receipt = PhaseReceipt("interevent", int(parent.episode), float(parent.payload.origin_s), float(parent.t_s),
                           int(parent.payload.generation) - int(parent.payload.entry_generation),
                           parent.payload.ledger.as_dict(), parent.payload.source.digest, parent.digest)
    new = _mint(parent, role="boundary", revision=parent.revision + 1, binding=None, payload=boundary, lifetime=lt, mark=mark,
                last_phase_receipt=receipt, request_sequence=int(parent.payload.next_request_sequence))
    _audit_core(lop, new, _arrays_of(new), new.t_s)
    return new


# --- explicit controlled wetting-history reset -------------------------------------------------------------------------
def _quiet_evidence(lop: LifecycleOperator, a: SimpleNamespace) -> dict[str, float]:
    """Actual current faces/excess/outlet evidence with the same formulae as the event quiet gate (read-only)."""
    topo, area = lop.topology, lop.column.cell_area_m2
    active = lop.column.active.reshape(lop.shape)
    h = a.surface_m
    excess = np.where(active, np.minimum(np.maximum(np.asarray(topo.source_bed_m) + h - np.asarray(topo.escape_level_m), 0.0), h),
                      0.0)
    fx, fy = topo.geometry.fx_type == 2, topo.geometry.fy_type == 2
    outlet = lop.topology.dx_m * (float(np.sum(np.where(fx, np.abs(a.qx), 0.0))) + float(np.sum(np.where(fy, np.abs(a.qy), 0.0))))
    out = {"max_unit_discharge_m2_s": float(max(np.max(np.abs(a.qx)), np.max(np.abs(a.qy)))),
           "max_excess_depth_m": float(np.max(excess)), "excess_volume_m3": float(np.sum(excess)) * area,
           "outlet_discharge_m3_s": float(outlet),
           "max_active_surface_depth_m": float(np.max(np.where(active, h, 0.0)))}
    if not all(math.isfinite(v) for v in out.values()):
        raise LifecycleResetRefused("non-finite quiet evidence; refused")
    return out


def reset_controlled_wetting_history(lop: LifecycleOperator, parent: LifecycleState, *, expected_parent: ExpectedParent,
                                     control_identity: Any, policy: ev.LayeredQuietPolicy) -> LifecycleState:
    """EXPLICIT controlled-test reset of the ACTIVE wetting history J to 0 at a CLOSED quiescent boundary (approved bounded
    control; NOT a drying/recovery law and never inferred from rain-off, a file, wind, a checkpoint or a schedule).

    Eligibility (every condition, no rounding/clipping/draining of water): a 'boundary' parent with a genuine current payload;
    EXACTLY dry active surface (`H == 0` bitwise on every active cell); the actual current signed faces, outlet and depression
    excess within the supplied `LayeredQuietPolicy`. A wet pond (even a tiny nonzero depth) CARRIES but refuses this reset and is
    never deleted. Preserved BITWISE (shared objects): soil, surface, faces, native bed, the lifetime-intake array (same array
    object), time, ledgers, hydro and request counts. Only the global revision and the reset count advance; a source-bound
    `ResetReceipt` is kept. J is not water: no physical ledger moves."""
    _check_lop(lop)
    cuda = lop._cuda
    if lop._cuda is not None and not lop._cuda.boundary_capable:
        raise LifecycleError("the controlled wetting-history reset is CPU-only in the storm-first CUDA profile (explicit fence, refused before any "
                             "hash, copy or quiet scan; prepare the lifecycle with the explicit CUDA_BOUNDARY_PROFILE; no inferred recovery)")
    _require_parent(parent, expected_parent)
    ident = _ident(control_identity, "control_identity")
    if type(policy) is not ev.LayeredQuietPolicy:
        raise LifecycleError("policy must be a genuine LayeredQuietPolicy")
    try:
        policy.validated()
    except (ev.LayeredEventError, ls.LayeredStormError) as exc:
        raise LifecycleError(f"invalid quiet policy: {exc}") from None
    if parent.role != "boundary":
        raise LifecycleResetRefused(f"a controlled reset needs a closed 'boundary', got {parent.role!r}: no active event, "
                                    "recession or request may be open")
    _admit(lop, lop.memory.reset_bytes, "reset", parent=parent)  # BEFORE the J copy and the evidence temporaries (CUDA: the authoritative plan)
    _check_state(lop, parent)
    a = _arrays_of(parent)
    _validate_inventories(lop, a)
    if cuda is not None:
        return _reset_cuda(lop, parent, a, ident, policy)
    active = lop.column.active.reshape(lop.shape)
    if np.any(a.surface_m[active] != 0.0):
        raise LifecycleResetRefused("the active surface is not EXACTLY dry (a retained pond may carry but is not eligible; no "
                                    "water is drained, rounded or deleted to gain eligibility); refused")
    evidence = _quiet_evidence(lop, a)
    pol = asdict(policy)
    if (evidence["max_unit_discharge_m2_s"] > policy.max_unit_discharge_m2_s
            or evidence["max_excess_depth_m"] > policy.max_excess_depth_m or evidence["excess_volume_m3"] > policy.excess_volume_m3
            or evidence["outlet_discharge_m3_s"] > policy.outlet_discharge_m3_s):
        raise LifecycleResetRefused(f"the current faces/excess/outlet are not quiet under the bound policy: {evidence}; refused")
    old = a.history
    act = lop.column.active
    new_j = np.where(act, 0.0, old.wetting_m)  # one fresh (N,) vector: active cells 0, inactive cells exact
    new_hist = mi.WettingHistory(new_j, old.cumulative_intake_m, old.operator_sha256, old.n_cells)  # intake ALIASED, not copied
    mi._check_history(lop.matrix, new_hist)
    if new_hist.cumulative_intake_m is not old.cumulative_intake_m or np.any(new_hist.wetting_m[act] != 0.0) \
            or np.any(new_hist.wetting_m[~act] != old.wetting_m[~act]):
        raise LifecycleError("internal: the reset product failed its validation; nothing adopted")
    payload = BoundaryPayload(_SEAL, bed=a.bed, history=new_hist, qx_m2_s=a.qx, qy_m2_s=a.qy, soil=a.soil,
                              surface_m=a.surface_m, t_s=float(a.t_s))
    rev = int(parent.revision) + 1
    receipt = ResetReceipt(parent.digest, float(parent.t_s), int(parent.episode), ident, _hash_array(old.wetting_m),
                           _hash_array(new_j), _hash_array(old.cumulative_intake_m), int(np.count_nonzero(old.wetting_m[act])),
                           float(np.sum(old.wetting_m[act])), pol, evidence, int(parent.revision), rev,
                           int(parent.reset_count) + 1, lop.matrix.sha256)
    return _mint(parent, revision=rev, reset_count=int(parent.reset_count) + 1, payload=payload, last_control_receipt=receipt)


def _hash_any(array: Any) -> str:
    """Streamed SHA-256 with the same bytes/header as the CPU `_hash_array` for equal values; a device array is staged one bounded chunk at a time."""
    h = hashlib.sha256()
    ls._stream_hash(h, array if getattr(array, "__module__", "").split(".")[0] == "cupy" else np.ascontiguousarray(array))
    return h.hexdigest()


def _reset_cuda(lop: LifecycleOperator, parent: LifecycleState, a: SimpleNamespace, ident: str, policy: ev.LayeredQuietPolicy) -> LifecycleState:
    """CUDA body of the controlled reset (same eligibility and product as the CPU body; device reductions/one packet each, bounded streamed hashes).
    Exact dry active H (one packet), the actual quiet evidence through the shared `interevent_entry.quiet_evidence` (same keys and comparisons), a fresh
    `J' = where(active, 0, J)`, the cumulative intake ALIASED, a validation packet, real receipt hashes. Nothing is drained, rounded or adopted on failure."""
    from maple_syrup import interevent_entry as ie

    xp = lop.xp
    active = lop.column.active.reshape(lop.shape)
    dry = ls._packet(xp, [xp.any(xp.where(active, a.surface_m, 0.0) != 0.0)])
    if dry[0] != 0.0:
        raise LifecycleResetRefused("the active surface is not EXACTLY dry (a retained pond may carry but is not eligible; no "
                                    "water is drained, rounded or deleted to gain eligibility); refused")
    duck = SimpleNamespace(xp=xp, topology=lop.topology, column=lop.column, shape=lop.shape, dx_m=lop.topology.dx_m)
    try:
        evidence = ie.quiet_evidence(duck, a.surface_m, a.qx, a.qy)
    except ls.LayeredStormError as exc:
        raise LifecycleResetRefused(f"{exc}; refused") from None
    pol = asdict(policy)
    if (evidence["max_unit_discharge_m2_s"] > policy.max_unit_discharge_m2_s or evidence["max_excess_depth_m"] > policy.max_excess_depth_m
            or evidence["excess_volume_m3"] > policy.excess_volume_m3 or evidence["outlet_discharge_m3_s"] > policy.outlet_discharge_m3_s):
        raise LifecycleResetRefused(f"the current faces/excess/outlet are not quiet under the bound policy: {evidence}; refused")
    old = a.history
    act = lop.column.active
    new_j = xp.where(act, 0.0, old.wetting_m)  # one fresh (N,) device vector: active cells 0, inactive cells exact
    new_hist = mi.WettingHistory(new_j, old.cumulative_intake_m, old.operator_sha256, old.n_cells)  # intake ALIASED, not copied
    mi._check_history(lop.matrix, new_hist)
    pk = ls._packet(xp, [xp.any(xp.where(act, new_j, 0.0) != 0.0), xp.any(xp.where(act, 0.0, new_j != old.wetting_m)),
                         xp.sum(xp.where(act, old.wetting_m != 0.0, False)), xp.sum(xp.where(act, old.wetting_m, 0.0))])
    if new_hist.cumulative_intake_m is not old.cumulative_intake_m or pk[0] != 0.0 or pk[1] != 0.0:
        raise LifecycleError("internal: the reset product failed its validation; nothing adopted")
    payload = BoundaryPayload(_SEAL, bed=a.bed, history=new_hist, qx_m2_s=a.qx, qy_m2_s=a.qy, soil=a.soil,
                              surface_m=a.surface_m, t_s=float(a.t_s))
    rev = int(parent.revision) + 1
    receipt = ResetReceipt(parent.digest, float(parent.t_s), int(parent.episode), ident, _hash_any(old.wetting_m), _hash_any(new_j),
                           _hash_any(old.cumulative_intake_m), int(pk[2]), float(pk[3]), pol, evidence, int(parent.revision), rev,
                           int(parent.reset_count) + 1, lop.matrix.sha256)
    return _mint(parent, revision=rev, reset_count=int(parent.reset_count) + 1, payload=payload, last_control_receipt=receipt)


# --- metadata ----------------------------------------------------------------------------------------------------------
def lifecycle_metadata(lop: LifecycleOperator, state: LifecycleState) -> dict[str, Any]:
    """Bounded scalar metadata (no arrays) a later exact lifecycle codec would need. Not a codec."""
    _check_state(lop, state)
    return {"profile": PROFILE if lop._cuda is None else lop._cuda.profile, "lop_sha256": lop.sha256, "role": state.role, "t_s": state.t_s, "revision": state.revision,
            "episode": state.episode, "reset_count": state.reset_count, "hydro_count": state.hydro_count,
            "request_sequence": state.request_sequence, "storm_operator_sha256": state.storm_operator_sha256,
            "lifetime": state.lifetime.as_dict(), "watermark": state.mark.as_dict(), "digest": state.digest,
            "last_phase_receipt": None if state.last_phase_receipt is None else state.last_phase_receipt.as_dict(),
            "last_control_receipt": None if state.last_control_receipt is None else state.last_control_receipt.as_dict()}



# --- checkpoint restoration seam (owner-controlled; used only by the verified lifecycle checkpoint codec) ------------------------
_LIFECYCLE_RECORD_KEYS = frozenset({"role", "t_s", "revision", "episode", "reset_count", "request_sequence",
                                    "storm_operator_sha256", "lop_sha256", "digest", "lifetime", "mark", "phase", "control"})
_PHASE_KEYS = frozenset({"kind", "episode", "t_start_s", "t_end_s", "hydro_steps", "local_ledger", "source_digest",
                         "parent_digest"})
_RESET_KEYS = frozenset({"parent_digest", "t_s", "episode", "control_identity", "old_wetting_sha256", "new_wetting_sha256",
                         "lifetime_intake_sha256", "n_active_wet_cells_before", "wetting_sum_before_m", "quiet_policy",
                         "quiet_evidence", "revision_before", "revision_after", "reset_count_after", "matrix_sha256"})
_EVIDENCE_KEYS = frozenset({"max_unit_discharge_m2_s", "max_excess_depth_m", "excess_volume_m3", "outlet_discharge_m3_s",
                            "max_active_surface_depth_m"})


def _refuse_daily_checkpoint(state: Any) -> None:
    """The existing (Phase 1 / C1) checkpoint writers have no field for the daily cursor, the pending storm occurrence or the versioned daily
    clock contract; they refuse such a state BEFORE any I/O instead of silently dropping them. A versioned codec that carries them is a later
    stage (S4); there is no migration path and no cursor-stripping."""
    if getattr(state, "daily", None) is not None:
        raise LifecycleError("this lifecycle state belongs to a daily lineage (it carries a daily cursor); the existing checkpoint codecs cannot "
                             "record it and refuse rather than drop the cursor (a daily checkpoint version is a later stage); refused before any I/O")


def checkpoint_state_record(lop: LifecycleOperator, state: LifecycleState) -> dict[str, Any]:
    """Bounded scalar record (an ENCODED tree) of a GENUINE accepted lifecycle state: scalars, lifetime ledger, accounted
    watermark and the last phase/control receipts. No array, no payload, no past state."""
    _refuse_daily_checkpoint(state)
    return _checkpoint_state_record_body(lop, state)


def checkpoint_state_record_daily(lop: LifecycleOperator, state: LifecycleState) -> dict[str, Any]:
    """The SAME closed lifecycle record for a state of a DAILY lineage (the V3 codec; its digest already includes the cursor, which travels as the separate
    closed `daily` scalar part). A state without a cursor is refused (the V3 codec records daily lineages only)."""
    if getattr(state, "daily", None) is None:
        raise LifecycleError("the daily checkpoint record needs a state of a daily lineage (a cursor); refused")
    return _checkpoint_state_record_body(lop, state)


def _checkpoint_state_record_body(lop: LifecycleOperator, state: LifecycleState) -> dict[str, Any]:
    from maple_syrup import checkpoint_capability as cc

    if lop._cuda is not None:
        raise LifecycleError("no lifecycle checkpoint exists for a CUDA lifecycle (explicit refusal; disk CUDA restart is a separate task)")
    _check_state(lop, state)
    body = {name: getattr(state, name) for name in _STATE_SCALARS}
    body["digest"] = state.digest
    body["lifetime"] = asdict(state.lifetime)
    body["mark"] = asdict(state.mark)
    body["phase"] = None if state.last_phase_receipt is None else state.last_phase_receipt.as_dict()
    body["control"] = None if state.last_control_receipt is None else state.last_control_receipt.as_dict()
    return cc.encode_tree(body)


def _typed(cls: Any, record: Any, name: str) -> Any:
    try:
        return it._typed_ledger(cls, record, name)
    except it.GeneralIntereventError as exc:
        raise LifecycleError(str(exc)) from None


def _hex64(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise LifecycleError(f"{name} must be a lowercase 64-hex SHA-256 string")
    return value


_V1_PHASE_KINDS = ("storm", "interevent")  # default (v1) vocabulary of a restored last-phase receipt
_V2_PHASE_KINDS = ("storm", "interevent", "boundary_origin")  # v2 adds the genuine zero-duration boundary-origin entry receipt


def _phase_from(tree: Any, kinds: tuple = _V1_PHASE_KINDS) -> PhaseReceipt | None:
    if tree is None:
        return None
    if type(tree) is not dict or set(tree) != _PHASE_KEYS:
        raise LifecycleError("phase receipt record keys mismatch (closed key set)")
    kind = tree["kind"]
    if kind not in kinds:
        raise LifecycleError(f"phase receipt kind must be one of {kinds}")
    ledger = tree["local_ledger"]
    if kind == "storm":
        _typed(ls.LayeredWaterLedger, ledger, "phase.local_ledger")
    else:
        if type(ledger) is not dict or set(ledger) != {"hydrology", "transpiration"}:
            raise LifecycleError("phase local_ledger keys mismatch (closed key set)")
        _typed(iw.WaterWindowLedger, ledger["hydrology"], "phase.local_ledger.hydrology")
        _typed(it.TranspirationLedger, ledger["transpiration"], "phase.local_ledger.transpiration")
    for key in ("t_start_s", "t_end_s"):
        if type(tree[key]) is not float or tree[key] < 0.0:
            raise LifecycleError(f"phase.{key} must be an exact non-negative float")
    for key in ("episode", "hydro_steps"):
        _count(tree[key], "phase." + key)
        if type(tree[key]) is not int:
            raise LifecycleError(f"phase.{key} must be an exact integer")
    if tree["t_end_s"] < tree["t_start_s"]:
        raise LifecycleError("phase receipt ends before it starts")
    src = tree["source_digest"]
    if src is not None:
        _hex64(src, "phase.source_digest")
    _hex64(tree["parent_digest"], "phase.parent_digest")
    return PhaseReceipt(kind, tree["episode"], tree["t_start_s"], tree["t_end_s"], tree["hydro_steps"], ledger, src,
                        tree["parent_digest"])


def _reset_from(tree: Any, lop: LifecycleOperator) -> ResetReceipt | None:
    if tree is None:
        return None
    if type(tree) is not dict or set(tree) != _RESET_KEYS:
        raise LifecycleError("reset receipt record keys mismatch (closed key set)")
    pol = tree["quiet_policy"]
    if type(pol) is not dict or set(pol) != set(ev.LayeredQuietPolicy.__dataclass_fields__):
        raise LifecycleError("reset quiet_policy keys mismatch (closed key set)")
    evid = tree["quiet_evidence"]
    if type(evid) is not dict or set(evid) != _EVIDENCE_KEYS:
        raise LifecycleError("reset quiet_evidence keys mismatch (closed key set)")
    for name, value in (*pol.items(), *evid.items(), ("t_s", tree["t_s"]), ("wetting_sum_before_m", tree["wetting_sum_before_m"])):
        if type(value) is not float or value < 0.0:
            raise LifecycleError(f"reset {name} must be an exact non-negative float")
    for key in ("episode", "n_active_wet_cells_before", "revision_before", "revision_after", "reset_count_after"):
        if type(tree[key]) is not int:
            raise LifecycleError(f"reset {key} must be an exact integer")
        _count(tree[key], "reset." + key)
    for key in ("parent_digest", "old_wetting_sha256", "new_wetting_sha256", "lifetime_intake_sha256", "matrix_sha256"):
        _hex64(tree[key], "reset." + key)
    _ident(tree["control_identity"], "reset.control_identity")
    if tree["matrix_sha256"] != lop.matrix.sha256:
        raise LifecycleError("the reset receipt belongs to a different matrix operator; refused")
    if tree["revision_after"] != tree["revision_before"] + 1 or tree["reset_count_after"] < 1:
        raise LifecycleError("reset receipt revision/count relation violated; refused")
    return ResetReceipt(tree["parent_digest"], tree["t_s"], tree["episode"], tree["control_identity"],
                        tree["old_wetting_sha256"], tree["new_wetting_sha256"], tree["lifetime_intake_sha256"],
                        tree["n_active_wet_cells_before"], tree["wetting_sum_before_m"], pol, evid, tree["revision_before"],
                        tree["revision_after"], tree["reset_count_after"], tree["matrix_sha256"])


def _revalidate_immediate_reset(lop: LifecycleOperator, control: ResetReceipt, a: SimpleNamespace) -> None:
    """Restore-only, READ-ONLY revalidation of the reset eligibility at the UNCHANGED immediate control boundary (no reset is
    executed, no water or J is touched): the saved policy must be a valid `LayeredQuietPolicy`, the active surface must be EXACTLY dry,
    the current signed faces/outlet/depression excess (the same admitted evidence as the runtime reset) must be within that policy and
    equal the receipt's saved evidence. Called only while the receipt still names the current boundary: a later legitimate storm or
    infiltration is never judged against this historical dry condition."""
    _admit(lop, lop.memory.reset_bytes, "restore reset revalidation")  # the reset's own scratch + accepted + statics, BEFORE any read
    try:
        policy = ev.LayeredQuietPolicy(**dict(control.quiet_policy))
        policy.validated()
    except (TypeError, ev.LayeredEventError, ls.LayeredStormError) as exc:
        raise LifecycleError(f"the reset receipt's quiet policy is invalid: {exc}") from None
    _validate_inventories(lop, a)  # finite/bounded faces and surface BEFORE the evidence reductions
    active = lop.column.active.reshape(lop.shape)
    if np.any(a.surface_m[active] != 0.0):
        raise LifecycleError("the immediate reset boundary's active surface is not exactly dry (the reset requires H == 0 on every "
                             "active cell); refused")
    evidence = _quiet_evidence(lop, a)
    if (evidence["max_unit_discharge_m2_s"] > policy.max_unit_discharge_m2_s
            or evidence["max_excess_depth_m"] > policy.max_excess_depth_m or evidence["excess_volume_m3"] > policy.excess_volume_m3
            or evidence["outlet_discharge_m3_s"] > policy.outlet_discharge_m3_s):
        raise LifecycleError(f"the immediate reset boundary's current faces/excess/outlet are not quiet under the receipt's policy: "
                             f"{evidence}; refused")
    if evidence != dict(control.quiet_evidence):
        raise LifecycleError("the immediate reset boundary's current quiet evidence differs from the receipt's saved evidence; refused")


def _restore_checkpoint_lifecycle(lop: LifecycleOperator, cap: Any, record: Any, *, role: str, arrays: tuple,
                                  storm_op: Any = None, event: Any = None, estate: Any = None, prep: Any = None,
                                  gop: Any = None, tstate: Any = None, _phase_kinds: Any = None, daily: Any = None) -> LifecycleState:
    """Owner-controlled restoration of the CURRENT accepted lifecycle state from a verified checkpoint. `daily` (V3 only) is the already STRICTLY decoded
    `DailyCursor`, supplied at CONSTRUCTION so the recomputed scalar digest (which includes the cursor) must equal the saved one; no cursor is ever attached later. `arrays` is the exact
    decoded `(surface_m, qx, qy, soil, history, bed)` tuple (a boundary payload is minted from it; the storm / interevent
    payloads are the genuine, already restored `estate` / `tstate`). Nothing here replays an entry, closure or reset: counters,
    lifetime ledger, watermark and bounded receipts are the saved accepted values, re-validated by the ordinary `_check_state`,
    the recomputed scalar digest and the role-aware `audit_lifecycle`. The reset receipt's J/intake hashes are compared with the
    current arrays ONLY while it still describes the immediate control boundary (later infiltration legitimately changes J)."""
    from maple_syrup import checkpoint_capability as cc

    _check_lop(lop)
    objs = (*arrays, storm_op, event, estate, prep, gop, tstate)
    try:
        cc.require(cap, "lifecycle", record, objs, lop_sha256=lop.sha256)
        tree = cc.decode_tree(record)
    except cc.CheckpointCapabilityError as exc:
        raise LifecycleError(f"checkpoint lifecycle record refused: {exc}") from None
    if type(tree) is not dict or set(tree) != _LIFECYCLE_RECORD_KEYS:
        raise LifecycleError("checkpoint lifecycle record keys mismatch (closed key set)")
    if role not in ("boundary", "storm", "interevent") or tree["role"] != role or tree["lop_sha256"] != lop.sha256:
        raise LifecycleError("checkpoint role/operator identity differs from the expected one; refused")
    if type(tree["t_s"]) is not float or tree["t_s"] < 0.0:
        raise LifecycleError("lifecycle t_s must be an exact non-negative float")
    for key in ("revision", "episode", "reset_count", "request_sequence"):
        if type(tree[key]) is not int:
            raise LifecycleError(f"lifecycle {key} must be an exact integer")
        _count(tree[key], key)
    sop = tree["storm_operator_sha256"]
    if sop is not None:
        _hex64(sop, "storm_operator_sha256")
    lifetime = _typed(LifetimeLedger, tree["lifetime"], "lifetime")
    mark = _typed(Watermark, tree["mark"], "mark")
    if _phase_kinds is None:
        _phase_kinds = _V1_PHASE_KINDS
    elif _phase_kinds is not _V2_PHASE_KINDS:  # closed selector: only the two module constants, by identity
        raise LifecycleError("unknown phase-kind vocabulary; refused")
    phase = _phase_from(tree["phase"], _phase_kinds)
    control = _reset_from(tree["control"], lop)
    surface, qx, qy, soil, history, bed = arrays
    if role == "boundary":
        if any(o is not None for o in (storm_op, event, estate, prep, gop, tstate)):
            raise LifecycleError("a boundary restores from its arrays only")
        payload, binding = BoundaryPayload(_SEAL, bed=bed, history=history, qx_m2_s=qx, qy_m2_s=qy, soil=soil, surface_m=surface,
                                           t_s=tree["t_s"]), None
    elif role == "storm":
        if storm_op is None or event is None or estate is None or prep is not None or gop is not None or tstate is not None:
            raise LifecycleError("a storm restores from its storm operator, event and event state only")
        payload, binding = estate, _Binding(op=storm_op, event=event)
    else:
        if storm_op is None or prep is None or gop is None or tstate is None or event is not None or estate is not None:
            raise LifecycleError("an interevent restores from its operator, interevent owners and general state only")
        payload, binding = tstate, _Binding(op=storm_op, prep=prep, gop=gop)
    state = LifecycleState(
        _SEAL, role=role, t_s=tree["t_s"], revision=tree["revision"], episode=tree["episode"], reset_count=tree["reset_count"],
        request_sequence=tree["request_sequence"], storm_operator_sha256=sop, lop_sha256=lop.sha256, binding=binding,
        last_control_receipt=control, last_phase_receipt=phase, lifetime=lifetime, mark=mark, payload=payload, daily=daily)
    if state.digest != _hex64(tree["digest"], "digest"):
        raise LifecycleError("the saved lifecycle digest differs from its recomputed digest; refused")
    if mark.n_steps > lifetime.n_hydro_steps or mark.n_requests > lifetime.n_t_requests:
        raise LifecycleError("the accounted watermark exceeds the lifetime operation counts; refused")
    if control is not None and (control.reset_count_after > state.reset_count or control.revision_after > state.revision
                                or control.t_s > state.t_s or control.episode > state.episode):
        raise LifecycleError("the reset receipt is inconsistent with the lifecycle counters/clock; refused")
    if phase is not None and (phase.episode > state.episode or phase.t_end_s > state.t_s):
        raise LifecycleError("the phase receipt is inconsistent with the lifecycle counters/clock; refused")
    if role != "storm" and phase is not None and phase.episode != state.episode:
        raise LifecycleError("a boundary/interevent phase receipt must name the current episode; refused")
    if role == "boundary" and ((phase is not None and state.t_s != phase.t_end_s)
                               or (phase is None and state.t_s != lifetime.anchor_t_s)):
        raise LifecycleError("a closed boundary's clock must equal its last phase end (or the run start when no phase has "
                             "run): no time passes outside a phase; refused")
    if (control is not None and role == "boundary" and control.revision_after == state.revision
            and control.reset_count_after == state.reset_count):
        a = _arrays_of(state)
        if (_hash_array(a.history.wetting_m) != control.new_wetting_sha256
                or _hash_array(a.history.cumulative_intake_m) != control.lifetime_intake_sha256
                or float(a.t_s) != control.t_s or np.any(a.history.wetting_m[lop.column.active] != 0.0)):
            raise LifecycleError("the immediate reset boundary's J/intake/clock differ from its source-bound receipt; refused")
        _revalidate_immediate_reset(lop, control, a)
    audit_lifecycle(lop, state)  # role-aware whole audit (admitted): metadata, every-cell inventories, lifetime closure
    return state


# --- CUDA (native9f) checkpoint owner seams (G1; additive siblings, the CPU definitions above are untouched) ---------------------
# Private, same-package trusted seams for a later explicit CUDA lifecycle checkpoint codec. They read no payload, open no file, admit no cold stage and
# never relabel a CPU/native39 lifecycle: the profile comes only from the sealed capability minted by `_prepare_cuda_lifecycle`.
@dataclass(frozen=True, slots=True)
class CudaCheckpointOwnerRecord:
    """Read-only metadata of a genuine native9f CUDA lifecycle operator (no array, no copy, no private capability object, not a MAPLE class)."""

    profile: str
    general_profile: str
    boundary_capable: bool
    device_id: Any
    host_available_bytes_fn: Any  # the caller's own callable, the SAME object given at preparation (never wrapped, never truth-tested)
    host_reserve_bytes: int  # the caller's declared reserve (the wrapper's retained leaves are separate)
    wrapper_retained_host_bytes: int
    source_package_sha256: str
    lop_sha256: str
    n_layers: int
    n_cells: int
    n_classes: int
    shape: tuple


def _cuda9f_lop(lop: Any) -> wlc.CudaLifecycleCapability:
    """Fence of every CUDA checkpoint seam: pure type/identity checks first, then the owner's `_check_lop` (current-device gate, cached source hot check, Dense class)."""
    if type(lop) is not LifecycleOperator or getattr(lop, "_seal", None) is not _SEAL:
        raise LifecycleError(f"lop must be a genuine LifecycleOperator, got {type(lop).__name__}")
    cap = lop._cuda
    if cap is None:
        raise LifecycleError("a CPU lifecycle has no CUDA checkpoint seam (the CPU checkpoint codecs are unchanged); refused")
    if type(cap) is not wlc.CudaLifecycleCapability or cap.seal is not _SEAL:
        raise LifecycleError("the CUDA lifecycle capability record is missing or foreign; refused")
    if cap.profile not in wlc.CUDA9F_PROFILES:
        raise LifecycleError("the CUDA checkpoint seams exist only for the native9f CUDA lifecycle profiles (not native39 or any other label); refused")
    _check_lop(lop)
    return cap


def _cuda_checkpoint_owner(lop: LifecycleOperator) -> CudaCheckpointOwnerRecord:
    """Read-only accessor for the native9f CUDA lifecycle's retained checkpoint-relevant metadata (current-device gated, no hash, no download)."""
    cap = _cuda9f_lop(lop)
    rec = _cuda_source_record_bytes(cap.profile)
    return CudaCheckpointOwnerRecord(
        profile=cap.profile, general_profile=wlc.general_profile_of(cap.profile), boundary_capable=bool(cap.boundary_capable), device_id=cap.device_id,
        host_available_bytes_fn=cap.host_available_bytes_fn, host_reserve_bytes=int(cap.host_reserve_bytes),
        wrapper_retained_host_bytes=wlc.wrapper_retained_host_bytes(lop.n_layers, source_record_bytes=rec),
        source_package_sha256=str(cap.source.package_sha256), lop_sha256=lop.sha256, n_layers=int(lop.n_layers), n_cells=int(lop.n_cells),
        n_classes=int(lop.n_classes), shape=tuple(int(v) for v in lop.shape))


def _cuda_checkpoint_role_fence(cap: wlc.CudaLifecycleCapability, role: Any, phase: Any = None, control: Any = None) -> None:
    if role not in ("boundary", "interevent"):
        raise LifecycleError(f"the CUDA lifecycle checkpoint covers only a closed 'boundary' or an accepted 'interevent', got {role!r} "
                             "(an incomplete storm/recession is not a public accepted boundary); refused")
    if not cap.boundary_capable and (control is not None or getattr(phase, "kind", None) == "boundary_origin"):
        raise LifecycleError("a boundary-origin entry or a controlled reset exists only under the explicit CUDA9F_BOUNDARY_PROFILE; refused")


def _checkpoint_state_record_cuda(lop: LifecycleOperator, state: LifecycleState) -> dict[str, Any]:
    """CUDA sibling of `checkpoint_state_record` (same bounded scalar/ledger/watermark/phase/control encoding; no array, payload or past state)."""
    _refuse_daily_checkpoint(state)
    return _cuda_record_body(lop, state)


def _cuda_record_body(lop: LifecycleOperator, state: LifecycleState) -> dict[str, Any]:
    from maple_syrup import checkpoint_capability as cc

    cap = _cuda9f_lop(lop)
    _check_state(lop, state)
    _cuda_checkpoint_role_fence(cap, state.role, state.last_phase_receipt, state.last_control_receipt)
    body = {name: getattr(state, name) for name in _STATE_SCALARS}
    body["digest"] = state.digest
    body["lifetime"] = asdict(state.lifetime)
    body["mark"] = asdict(state.mark)
    body["phase"] = None if state.last_phase_receipt is None else state.last_phase_receipt.as_dict()
    body["control"] = None if state.last_control_receipt is None else state.last_control_receipt.as_dict()
    return cc.encode_tree(body)


def _revalidate_immediate_reset_cuda(lop: LifecycleOperator, control: ResetReceipt, a: SimpleNamespace, parent: LifecycleState) -> None:
    """CUDA twin of `_revalidate_immediate_reset`: the authoritative CUDA reset plan is admitted first, the exact-dry and quiet evidence are the runtime reset's
    own device reductions (`interevent_entry.quiet_evidence` on the same duck operator) and the saved evidence must be EQUAL (no tolerance, no alternate rule)."""
    from maple_syrup import interevent_entry as ie

    _admit(lop, lop.memory.reset_bytes, "reset", parent=parent)  # the CUDA reset plan (host then device), BEFORE any read
    try:
        policy = ev.LayeredQuietPolicy(**dict(control.quiet_policy))
        policy.validated()
    except (TypeError, ev.LayeredEventError, ls.LayeredStormError) as exc:
        raise LifecycleError(f"the reset receipt's quiet policy is invalid: {exc}") from None
    _validate_inventories(lop, a)  # finite/bounded faces and surface BEFORE the evidence reductions
    xp = lop.xp
    active = lop.column.active.reshape(lop.shape)
    dry = ls._packet(xp, [xp.any(xp.where(active, a.surface_m, 0.0) != 0.0)])
    if dry[0] != 0.0:
        raise LifecycleError("the immediate reset boundary's active surface is not exactly dry (the reset requires H == 0 on every "
                             "active cell); refused")
    duck = SimpleNamespace(xp=xp, topology=lop.topology, column=lop.column, shape=lop.shape, dx_m=lop.topology.dx_m)
    try:
        evidence = ie.quiet_evidence(duck, a.surface_m, a.qx, a.qy)
    except ls.LayeredStormError as exc:
        raise LifecycleError(f"the immediate reset boundary's quiet evidence is refused: {exc}") from None
    if (evidence["max_unit_discharge_m2_s"] > policy.max_unit_discharge_m2_s
            or evidence["max_excess_depth_m"] > policy.max_excess_depth_m or evidence["excess_volume_m3"] > policy.excess_volume_m3
            or evidence["outlet_discharge_m3_s"] > policy.outlet_discharge_m3_s):
        raise LifecycleError(f"the immediate reset boundary's current faces/excess/outlet are not quiet under the receipt's policy: "
                             f"{evidence}; refused")
    if evidence != dict(control.quiet_evidence):
        raise LifecycleError("the immediate reset boundary's current quiet evidence differs from the receipt's saved evidence; refused")


def _restore_checkpoint_lifecycle_cuda(lop: LifecycleOperator, cap: Any, record: Any, *, role: str, arrays: tuple, storm_op: Any = None,
                                       prep: Any = None, gop: Any = None, tstate: Any = None, _phase_kinds: Any = None, daily: Any = None) -> LifecycleState:
    """CUDA sibling of `_restore_checkpoint_lifecycle` for a closed 'boundary' or an accepted 'interevent' of the native9f CUDA lifecycle (the storm role is refused).
    Nothing replays an entry, closure or reset: counters, ledgers, watermark, clock and bounded receipts are the saved accepted values, re-validated by the ordinary
    `_check_state`, the recomputed scalar digest and the whole `audit_lifecycle`. The immediate-reset recheck is device-aware (streamed hashes, device packets, the
    runtime reset's own quiet evidence) and strict."""
    from maple_syrup import checkpoint_capability as cc

    lcap = _cuda9f_lop(lop)
    _cuda_checkpoint_role_fence(lcap, role)
    if type(arrays) is not tuple or len(arrays) != 6:
        raise LifecycleError("arrays must be the decoded (surface, qx, qy, soil, history, bed) tuple; refused")
    surface, qx, qy, soil, history, bed = arrays
    if (bed is None) != (lop.context is None):
        raise LifecycleError("the restored bed presence differs from the lifecycle's sealed native binding; refused")
    if bed is not None and (type(bed) is not BedState or type(bed.voxel_column) is not VoxelColumnState):
        raise LifecycleError("the restored bed must be a genuine native Dense BedState (no Windowed or foreign bed); refused")
    if role == "boundary":
        if any(o is not None for o in (storm_op, prep, gop, tstate)):
            raise LifecycleError("a boundary restores from its arrays only")
    else:
        if storm_op is None or prep is None or gop is None or tstate is None:
            raise LifecycleError("an interevent restores from its operator, interevent owners and general state only")
        if getattr(getattr(gop, "cuda", None), "profile", None) != wlc.general_profile_of(lcap.profile):
            raise LifecycleError("the bound general owner's CUDA profile is not the one this lifecycle's profile requires (no cross-family binding); refused")
    objs = (*arrays, storm_op, None, None, prep, gop, tstate)  # the same object tuple shape as the CPU capability (no event/estate)
    try:
        cc.require(cap, "lifecycle", record, objs, lop_sha256=lop.sha256)
        tree = cc.decode_tree(record)
    except cc.CheckpointCapabilityError as exc:
        raise LifecycleError(f"checkpoint lifecycle record refused: {exc}") from None
    if type(tree) is not dict or set(tree) != _LIFECYCLE_RECORD_KEYS:
        raise LifecycleError("checkpoint lifecycle record keys mismatch (closed key set)")
    if tree["role"] != role or tree["lop_sha256"] != lop.sha256:
        raise LifecycleError("checkpoint role/operator identity differs from the expected one; refused")
    if type(tree["t_s"]) is not float or tree["t_s"] < 0.0:
        raise LifecycleError("lifecycle t_s must be an exact non-negative float")
    for key in ("revision", "episode", "reset_count", "request_sequence"):
        if type(tree[key]) is not int:
            raise LifecycleError(f"lifecycle {key} must be an exact integer")
        _count(tree[key], key)
    sop = tree["storm_operator_sha256"]
    if sop is not None:
        _hex64(sop, "storm_operator_sha256")
    lifetime = _typed(LifetimeLedger, tree["lifetime"], "lifetime")
    mark = _typed(Watermark, tree["mark"], "mark")
    if _phase_kinds is None:
        _phase_kinds = _V1_PHASE_KINDS
    elif _phase_kinds is not _V2_PHASE_KINDS:  # closed selector: only the two module constants, by identity
        raise LifecycleError("unknown phase-kind vocabulary; refused")
    phase = _phase_from(tree["phase"], _phase_kinds)
    control = _reset_from(tree["control"], lop)
    _cuda_checkpoint_role_fence(lcap, role, phase, control)
    if role == "boundary":
        payload, binding = BoundaryPayload(_SEAL, bed=bed, history=history, qx_m2_s=qx, qy_m2_s=qy, soil=soil, surface_m=surface,
                                           t_s=tree["t_s"]), None
    else:
        payload, binding = tstate, _Binding(op=storm_op, prep=prep, gop=gop)
    state = LifecycleState(
        _SEAL, role=role, t_s=tree["t_s"], revision=tree["revision"], episode=tree["episode"], reset_count=tree["reset_count"],
        request_sequence=tree["request_sequence"], storm_operator_sha256=sop, lop_sha256=lop.sha256, binding=binding,
        last_control_receipt=control, last_phase_receipt=phase, lifetime=lifetime, mark=mark, payload=payload, daily=daily)
    if state.digest != _hex64(tree["digest"], "digest"):
        raise LifecycleError("the saved lifecycle digest differs from its recomputed digest; refused")
    if mark.n_steps > lifetime.n_hydro_steps or mark.n_requests > lifetime.n_t_requests:
        raise LifecycleError("the accounted watermark exceeds the lifetime operation counts; refused")
    if control is not None and (control.reset_count_after > state.reset_count or control.revision_after > state.revision
                                or control.t_s > state.t_s or control.episode > state.episode):
        raise LifecycleError("the reset receipt is inconsistent with the lifecycle counters/clock; refused")
    if phase is not None and (phase.episode > state.episode or phase.t_end_s > state.t_s):
        raise LifecycleError("the phase receipt is inconsistent with the lifecycle counters/clock; refused")
    if phase is not None and phase.episode != state.episode:
        raise LifecycleError("a boundary/interevent phase receipt must name the current episode; refused")
    if role == "boundary" and ((phase is not None and state.t_s != phase.t_end_s)
                               or (phase is None and state.t_s != lifetime.anchor_t_s)):
        raise LifecycleError("a closed boundary's clock must equal its last phase end (or the run start when no phase has "
                             "run): no time passes outside a phase; refused")
    if (control is not None and role == "boundary" and control.revision_after == state.revision
            and control.reset_count_after == state.reset_count):
        a = _arrays_of(state)
        xp = lop.xp
        j_nonzero = ls._packet(xp, [xp.any(xp.where(lop.column.active, a.history.wetting_m, 0.0) != 0.0)])
        if (_hash_any(a.history.wetting_m) != control.new_wetting_sha256
                or _hash_any(a.history.cumulative_intake_m) != control.lifetime_intake_sha256
                or float(a.t_s) != control.t_s or j_nonzero[0] != 0.0):
            raise LifecycleError("the immediate reset boundary's J/intake/clock differ from its source-bound receipt; refused")
        _revalidate_immediate_reset_cuda(lop, control, a, state)
    audit_lifecycle(lop, state)  # role-aware whole audit (admitted): metadata, every-cell inventories, lifetime closure
    return state


# --- G5b: APPEND-ONLY seams for a genuine COMPLETED storm (trusted same-package use by a future V2 CUDA codec; nothing above this line is edited) -----------------------------
# Boundary/interevent delegate to the accepted G1 siblings byte-for-byte. The storm role is allowed ONLY here and ONLY for a storm whose payload is genuinely `completed`; an incomplete storm, a CPU/native39 lifecycle,
# a forged state or any other role is refused. Nothing replays begin/entry/step/reset/close and no completed flag, source kind, clock or anchor is fabricated.
def _checkpoint_state_record_cuda_v2(lop: LifecycleOperator, state: LifecycleState) -> dict[str, Any]:
    """O4: bounded scalar record of a genuine accepted lifecycle state for the V2 CUDA codec: closed boundary / accepted interevent (identical to the G1 record) or a COMPLETED storm."""
    from maple_syrup import checkpoint_capability as cc

    if type(state) is not LifecycleState:
        raise LifecycleError("state must be a genuine accepted LifecycleState")
    _refuse_daily_checkpoint(state)
    if state.role in ("boundary", "interevent"):
        return _checkpoint_state_record_cuda(lop, state)
    return _cuda_storm_record(lop, state, require_completed=True)


def _checkpoint_state_record_cuda_v3(lop: LifecycleOperator, state: LifecycleState) -> dict[str, Any]:
    """The V3 (Phase 1.5 DAILY) CUDA lifecycle record: the same closed record as V1/V2 for a state of a daily lineage (a state without a cursor is refused); a storm may be PARTIAL
    (the codec's storm record carries the progress; completion is never fabricated or required)."""
    if type(state) is not LifecycleState:
        raise LifecycleError("state must be a genuine accepted LifecycleState")
    if state.daily is None:
        raise LifecycleError("the V3 CUDA checkpoint records only states of a daily lineage (a cursor); refused")
    if state.role in ("boundary", "interevent"):
        return _cuda_record_body(lop, state)
    return _cuda_storm_record(lop, state, require_completed=False)


def _cuda_storm_record(lop: LifecycleOperator, state: LifecycleState, *, require_completed: bool) -> dict[str, Any]:
    from maple_syrup import checkpoint_capability as cc

    if state.role != "storm":
        raise LifecycleError(f"the CUDA lifecycle checkpoint covers only a closed 'boundary', an accepted 'interevent' or a 'storm', got {state.role!r}; refused")
    cap = _cuda9f_lop(lop)
    _check_state(lop, state)
    b = state.binding
    if type(b) is not _Binding or b.op is None or b.event is None or b.prep is not None or b.gop is not None:
        raise LifecycleError("a storm checkpoint needs exactly its storm operator and event binding; refused")
    if type(state.payload) is not ev.LayeredEventState or (require_completed and state.payload.completed is not True):
        raise LifecycleError("only a COMPLETED storm is checkpointed by this codec version (an incomplete rain/recession storm is not a public accepted boundary); refused")
    if not cap.boundary_capable and (state.last_control_receipt is not None or getattr(state.last_phase_receipt, "kind", None) == "boundary_origin"):
        raise LifecycleError("a boundary-origin entry or a controlled reset exists only under the explicit CUDA9F_BOUNDARY_PROFILE; refused")
    body = {name: getattr(state, name) for name in _STATE_SCALARS}
    body["digest"] = state.digest
    body["lifetime"] = asdict(state.lifetime)
    body["mark"] = asdict(state.mark)
    body["phase"] = None if state.last_phase_receipt is None else state.last_phase_receipt.as_dict()
    body["control"] = None if state.last_control_receipt is None else state.last_control_receipt.as_dict()
    return cc.encode_tree(body)


def _restore_checkpoint_lifecycle_cuda_v2(lop: LifecycleOperator, cap: Any, record: Any, *, role: str, arrays: tuple, storm_op: Any = None, event: Any = None, estate: Any = None,
                                          prep: Any = None, gop: Any = None, tstate: Any = None, _phase_kinds: Any = None, daily: Any = None,
                                          _partial_ok: bool = False) -> LifecycleState:
    """O5: trusted restoration of the CURRENT accepted state for the V2 CUDA codec. boundary/interevent delegate to the accepted G1 restorer (no event/estate allowed); a COMPLETED storm is restored from its
    already re-prepared real operator, event and `LayeredEventState` (bound to the exact decoded arrays by identity through the call-local capability); nothing is replayed or fabricated."""
    from maple_syrup import checkpoint_capability as cc

    if role in ("boundary", "interevent"):
        if event is not None or estate is not None:
            raise LifecycleError("a boundary/interevent restores without a storm event or event state; refused")
        return _restore_checkpoint_lifecycle_cuda(lop, cap, record, role=role, arrays=arrays, storm_op=storm_op, prep=prep, gop=gop, tstate=tstate, _phase_kinds=_phase_kinds,
                                                  daily=daily)
    if role != "storm":
        raise LifecycleError("the CUDA lifecycle checkpoint covers only 'boundary', 'interevent' and a COMPLETED 'storm'; refused")
    lcap = _cuda9f_lop(lop)
    if type(arrays) is not tuple or len(arrays) != 6:
        raise LifecycleError("arrays must be the decoded (surface, qx, qy, soil, history, bed) tuple; refused")
    surface, qx, qy, soil, history, bed = arrays
    if type(bed) is not BedState or type(bed.voxel_column) is not VoxelColumnState or lop.context is None:
        raise LifecycleError("the restored storm needs a genuine native Dense BedState (no Windowed or foreign bed); refused")
    if storm_op is None or event is None or estate is None or prep is not None or gop is not None or tstate is not None:
        raise LifecycleError("a storm restores from its storm operator, event and event state only")
    if type(estate) is not ev.LayeredEventState or (estate.completed is not True and not _partial_ok):
        raise LifecycleError("only a COMPLETED storm is restored by this codec version (completed is never fabricated or defaulted); refused")
    if _partial_ok and type(estate.completed) is not bool:
        raise LifecycleError("the storm completed flag must be an exact bool; refused")
    if getattr(event, "op", None) is not storm_op:
        raise LifecycleError("the event was not prepared on this storm operator; refused")
    w = estate.water
    if not (w.surface_m is surface and w.qx_m2_s is qx and w.qy_m2_s is qy and w.soil is soil and w.history is history and w.bed is bed):
        raise LifecycleError("the event state's water is not the decoded arrays tuple (identity); refused")
    objs = (*arrays, storm_op, event, estate, None, None, None)
    try:
        cc.require(cap, "lifecycle", record, objs, lop_sha256=lop.sha256)
        tree = cc.decode_tree(record)
    except cc.CheckpointCapabilityError as exc:
        raise LifecycleError(f"checkpoint lifecycle record refused: {exc}") from None
    if type(tree) is not dict or set(tree) != _LIFECYCLE_RECORD_KEYS:
        raise LifecycleError("checkpoint lifecycle record keys mismatch (closed key set)")
    if tree["role"] != "storm" or tree["lop_sha256"] != lop.sha256:
        raise LifecycleError("checkpoint role/operator identity differs from the expected one; refused")
    if type(tree["t_s"]) is not float or tree["t_s"] < 0.0:
        raise LifecycleError("lifecycle t_s must be an exact non-negative float")
    for key in ("revision", "episode", "reset_count", "request_sequence"):
        if type(tree[key]) is not int:
            raise LifecycleError(f"lifecycle {key} must be an exact integer")
        _count(tree[key], key)
    sop = tree["storm_operator_sha256"]
    _hex64(sop, "storm_operator_sha256")  # a storm always carries its operator identity
    if sop != storm_op.sha256:
        raise LifecycleError("the saved storm-operator identity differs from the re-prepared storm operator; refused")
    lifetime = _typed(LifetimeLedger, tree["lifetime"], "lifetime")
    mark = _typed(Watermark, tree["mark"], "mark")
    if _phase_kinds is None:
        _phase_kinds = _V1_PHASE_KINDS
    elif _phase_kinds is not _V2_PHASE_KINDS:
        raise LifecycleError("unknown phase-kind vocabulary; refused")
    phase = _phase_from(tree["phase"], _phase_kinds)
    control = _reset_from(tree["control"], lop)
    if not lcap.boundary_capable and (control is not None or getattr(phase, "kind", None) == "boundary_origin"):
        raise LifecycleError("a boundary-origin entry or a controlled reset exists only under the explicit CUDA9F_BOUNDARY_PROFILE; refused")
    state = LifecycleState(
        _SEAL, role="storm", t_s=tree["t_s"], revision=tree["revision"], episode=tree["episode"], reset_count=tree["reset_count"], request_sequence=tree["request_sequence"],
        storm_operator_sha256=sop, lop_sha256=lop.sha256, binding=_Binding(op=storm_op, event=event), last_control_receipt=control, last_phase_receipt=phase, lifetime=lifetime,
        mark=mark, payload=estate, daily=daily)
    if state.digest != _hex64(tree["digest"], "digest"):
        raise LifecycleError("the saved lifecycle digest differs from its recomputed digest; refused")
    if mark.n_steps > lifetime.n_hydro_steps or mark.n_requests > lifetime.n_t_requests:
        raise LifecycleError("the accounted watermark exceeds the lifetime operation counts; refused")
    if control is not None and (control.reset_count_after > state.reset_count or control.revision_after > state.revision or control.t_s > state.t_s or control.episode > state.episode):
        raise LifecycleError("the reset receipt is inconsistent with the lifecycle counters/clock; refused")
    if phase is not None and (phase.episode > state.episode or phase.t_end_s > state.t_s):
        raise LifecycleError("the phase receipt is inconsistent with the lifecycle counters/clock; refused")
    audit_lifecycle(lop, state)  # role-aware whole audit (admitted): metadata, every-cell inventories, lifetime closure
    try:
        ev.audit_event(event, estate)  # the genuine anchored event audit incl. the completed instantaneous quiet gates on the restored arrays
    except (ev.LayeredEventError, ls.LayeredStormError) as exc:
        raise LifecycleError(f"the restored completed storm failed its genuine event audit: {exc}") from None
    return state
