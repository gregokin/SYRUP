"""USABLE ACTUAL-MAPLE COUPLED POND / SPLASH EVENT (opt-in; dense bed first). Documentation: docs/splash_and_overflow/pond_event.md.

One bounded event over a MAPLE `BedState` with the local-inertial pond hydraulics (`pond_hydraulics`), the wet MAHLERAN laws
(`sediment_physics`), optional direct rain splash on DRY raining cells (`splash`), the multi-face characteristic transport
(`pond_transport`) and a native MAPLE commit / geometry gate (`pond_bed`). It is a NEW opt-in backend: the accepted method-5
event, `complete_event`, `checkpoint` and every default are untouched. Nothing here was run by its author (file-only tools);
the root executes `tests/pond_event`.

PUBLIC API (all methods pure: a state is never modified, an exception leaves every input unchanged)
    PondEvent(context, column, sediment, schedule, topology, ...)  -- the event object (static config + a runtime cache)
    .initial_state(bed, soil_water_m)                      -> PondEventState
    .step(state, rain_rate_m_per_s, dt_s)                  -> PondCoupledStep   (ONE accepted coupled physical step)
    .evolve(state, end_s, stop_when_quiet=False)           -> PondEvolveResult  (rain-knot split, bounded adaptive retry)
    .quiet_metrics(state) / .audit(state)                  -> dict              (quiet gates; water / sediment / geometry audit)
    .finalize(state)                                       -> PondDryHandoff    (explicit safe terminal settlement + dry reset)
    .complete(state, max_end_s)                            -> PondDryHandoff    (evolve until quiet, else PondEventIncomplete)

ACCEPTED COUPLED STEP, in order (everything is a local value until the end; nothing is published on any failure)
 1. `step_once` of the pond hydraulics (donor limiter OFF) at the CURRENT held topology: depth, soil, qx, qy.
 2. Wet laws on the POST-hydraulic depth `h1` and the erosion velocity (volume-weighted outward cell speed of the face
    velocities, `face_speed_diagnostics`) from the HOLDINGS at the start of the step. Dry splash demand (when `splash` is
    given) on cells with rain > 0 and `h1 == 0` exactly; the wet law owns every `h1 > 0` cell: no double pickup.
 3. Dry splash: native MAPLE exchange (`apply_splash_exchange`: pickup cap -> redistribution of the ACTUAL pickup ->
    native deposit/export). The mobile pool is unchanged by construction; the phase is only canonicalised where the pool is
    exactly 0 (no mass is created or erased).
 4. Wet transaction: native MAPLE pickup of the wet demand -> `pond_transport_step` on the PRE-pickup pool, its phase and
    the ACTUAL pickup (signed actual face water volumes) -> native deposit/export; the phase is reconciled to the ACTUAL pool
    with the existing `reconcile_phase` rules (`sediment_event._reconcile_phase_inputs`, the shared bound check).
 5. Geometry gate and native accounting commit (below). 6. Ledgers, quiet monitor, counters; a new immutable state.

SLOPES (three different quantities, never conflated)
 - TOPOGRAPHIC slope / aspect for the wet raindrop law and for the splash rays come from the HELD real bed
   (`pond_bed.steepest_d4_descent`: steepest strictly downward D4 neighbour among active cells and across explicit outlet
   faces; flat -> S = 0, aspect 1). They change only with a geometry refresh. The metadata graph of the hydraulics is never read.
   Consequence (law, not a choice): the Quansah raindrop law is proportional to `(100 S)^c`, so flats give exactly 0 raindrop
   detachment (wet and dry); a dry splash needs sloped dry raining cells.
 - FLOW SHEAR: the legacy `u* = sqrt(g h S)` is a uniform-flow estimate and is wrong on flats / unsteady flow. With
   `shear_slope = "friction"` (default) the equivalent friction slope `S_f = f u^2 / (8 g h)` (h > 0, else 0) is supplied
   to the wet law so that `u* = sqrt(f / 8) u` exactly (Darcy-Weisbach `tau = rho f u^2 / 8`); `f` is the cell friction of
   the topology. `"topographic"` keeps the legacy estimate.
 - STREAM POWER `rho g h v S`: `stream_power_slope = "friction"` (default) uses the same `S_f` (power `rho f u^3 / 8`, the
   energy dissipation of the flow); `"topographic"` uses the bed slope (zero stream power on flats -> no concentrated or
   diffuse transport capacity there). No floor, no minimum slope.
 The Froude numbers (max face Froude, number above `froude_reference`) are reported per step and accumulated; nothing is clipped.

GEOMETRY GATE (net elevation only). Every accepted step, BEFORE any commit, `pond_bed.gate_displacement_m` evaluates
max over ACTIVE cells |actual current MAPLE elevation - held source| (per-cell signed, one counted scalar read, device
reductions). It decides ONLY the geometry:
 - `>= pending_elevation_trigger_m` (MAPLE's value, default 0.00075 m, `>=`) forces a genuine native commit (reason
   `net_elevation`) and THEN the topology refresh from the committed elevation (stable initial datum + actual committed delta,
   fixed ring/mask; the single full-surface transfer). qx, qy, soil, depth, phase and mobile are carried unchanged.
 - MAPLE's own triggers and the forced final commit commit the bed (accounting) and NEVER reroute. Time, report, checkpoint,
   force and reset never refresh. Per-cell gains and losses cancel; different cells never cancel.
A refresh failure (for example an outlet that no longer drops to the ring) raises before anything is published.

COMPLETION. Quiet = rain off, then for `hold_s` of CONTINUOUS accepted steps: per-step max unit flux, outlet discharge and
the (conservative, summed per-step) max depth change below the policy limits, and the EXCESS water
`clip(eta - escape, 0, h)` (0 on closed components; the physical h is never altered) under the CompletionPolicy
depth / surface-volume thresholds (1e-8 m, 1e-6 m3; physical quiet thresholds, not conservation tolerances). The raw
count of cells with eta > escape is diagnostic only, so an asymptotic recession of a tiny film at an outlet can finish
while genuine moving over-crest water cannot. Below-crest ponded water stays accounted until the explicit reset. Mobile load below the MAPLE significance threshold (unless the explicit
`deposit_mobile_at_flow_end` option). Soil may still drain (as in `CompletionPolicy`); a slowly draining cell is not quiet
and never labelled terminal. `finalize` re-checks everything, settles the pool through genuine MAPLE deposition (refused
if anything stays mobile), makes the forced final commit (geometry gate still applies), audits closure and performs the
explicit dry reset, recording the removed surface AND soil water. Anything unmet raises `PondEventIncomplete` (a policy that
refuses is preferred to an invented completion).

STATE. `PondEventState` is a frozen host/device ARRAY record with no solver, namespace module or callable fields: hydraulic
state (t, depth, soil, qx, qy, adaptive cap), the MAPLE `BedState`, transport memory, phase arrays, held topology (a
host-only `PondTopology`), held source (device array) with the stable `PondDatum`, ledgers, counters, quiet monitor and the
config / forcing identities. The solver, `PhaseState`, physics grid, splash grid and face network are a RUNTIME rebuilt from
it (`PondEvent._runtime`); a checkpoint module can serialise the state and rebuild with the same config. State and history
are bounded (no per-step arrays; the commit log is capped).

RESIDENT-WINDOW BED (opt-in: `PondEventControl(bed_storage="windowed")`, documentation docs/pond_windowed/core.md). The wet
transaction and the dry splash exchange run through MAPLE's paired windowed transaction (pickup -> transport -> deposit in one
frame; an older MAPLE without it is an unsupported dependency, never downgraded); commits are MAPLE's staged windowed commit
with trusted height packets; the geometry gate reads only the height packets (no deep read, no densification). Deep storage
publication is separate from physics: `deep_publication="step"` publishes after every accepted step (default), `"boundary"`
carries MAPLE's valid unpublished journal through accepted steps and publishes at a high-water mark, at the return of
`evolve`, in `finalize`, in `publish_storage`, and in the single flush before a recognised append-cap retry of the same
physical step (`evolve`). A public `step` may return a state carrying that pending journal; it is valid, but not restartable
until published. Dense mode is untouched.

LIMITS. Dense bed, or an actual windowed bed (see above). Avalanching disabled. Hydraulics are the experimental
local-inertial ALTERNATIVE (not method-5/6); its velocity is unqualified for erosion and reported, not clipped. The pond
transport is the array implementation (backend "numba" is hydraulics only; "cuda" runs hydraulics, physics, transport and
the explicit CUDA splash on the device). Splash is instantaneous redistribution with explicit supplied inverse distances;
no calibrated default. Wind is excluded. Wet depth change inside a step is not attenuated for splash; the reference
decides the dry/wet switch on the pre-hydrology depth, this module on the post-hydrology depth (documented difference).
"""

from __future__ import annotations

import hashlib
import math
import weakref
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from maple_syrup.characteristic_transport import DEFAULT_COURANT_MAX, DEFAULT_N_BINS, PhaseState, empty_phase_state
from maple_syrup.conservation import reservoir_bound_kg
from maple_syrup.event_backend import host_packet
from maple_syrup.experimental_hydrology import (
    HydraulicControl,
    HydraulicState,
    HydraulicStepRejected,
    advance_time,
    check_dt_cap,
    check_state_time,
)
from maple_syrup.experimental_storm import ExperimentalControl
from maple_syrup.infiltration import ColumnParameters
from maple_syrup.pond_bed import (
    NativeCommit,
    PondBedError,
    bed_committed_at,
    committed_displacement_m,
    gate_displacement_m,
    native_commit,
    pending_journal_records,
    physical_full_source,
    pickup_transport_deposit,
    require_paired_entry,
    require_supported_bed,
    reroute_threshold_m,
    steepest_d4_descent,
)
from maple_syrup.pond_hydraulics import (
    PondHydraulics,
    PondLedger,
    PondStep,
    build_pond_hydraulics,
    face_speed_diagnostics,
    rebind_pond,
    step_once,
)
from maple_syrup.pond_network import PondTopology, build_pond_topology, update_pond_topology
from maple_syrup.pond_transport import (
    FaceNetwork,
    face_network_from_pond_topology,
    operation_count,
    pond_transport_step,
    validate_phase,
)
from maple_syrup.rainfall import RainfallSchedule
from maple_syrup.sediment_bed import (
    BedContext,
    BedIntegrationError,
    BedState,
    apply_bed_demand,
    bed_inventory,
    check_routing_source,
    deep_storage_record,
    is_carried_append_cap_refusal,
    is_windowed_bed,
    journal_recognizer_source,
    publish_bed_storage,
    require_typed_journal_capacity_error,
    resolved_journal_cap,
    validate_deep_storage,
    with_water,
)
from maple_syrup.sediment_physics import (
    GRAVITY_M_S2,
    SedimentPhysicsParameters,
    dry_raindrop_pickup_demand,
    physics_grid,
    sediment_physics_step,
)
from maple_syrup.sediment_transport import TransportStepRejected, water_demand_from_transport
from maple_syrup.splash import SplashParameters, apply_splash_exchange, splash_grid

__all__ = [
    "SCHEMA",
    "PondCompletionPolicy",
    "PondCoupledStep",
    "PondDatum",
    "PondDryHandoff",
    "PondEvent",
    "PondEventControl",
    "PondEventError",
    "PondEventIncomplete",
    "PondEventState",
    "PondEvolveResult",
    "PondQuiet",
    "PondStepRecord",
]

SCHEMA = "maple_syrup.pond_event.v1"
SLOPE_CHOICES = ("friction", "topographic")
SPLASH_IMPLEMENTATIONS = ("array", "cuda", "auto")
BED_STORAGES = ("dense", "windowed")
DEEP_PUBLICATIONS = ("step", "boundary")
_REJECTIONS_KEPT = 100


class PondEventError(RuntimeError):
    """Invalid request, state or a failed check of the pond event. Raised before anything is published."""


class PondEventIncomplete(PondEventError):
    """A limit or an unmet criterion is not physical completion. No dry state is returned and no reset has run."""


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise PondEventError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _positive(value: Any, name: str) -> float:
    v = _real(value, name)
    if not (math.isfinite(v) and v > 0.0):
        raise PondEventError(f"{name} must be finite and > 0, got {value!r}")
    return v


def _host(array: Any) -> np.ndarray:
    """Host float64 copy of an array through MAPLE's counted `to_host`; host scalars (NumPy reductions return
    scalars, not arrays) are converted directly."""
    from maple.core.backend import is_array, to_host

    if not is_array(array):
        return np.asarray(array, dtype=np.float64)
    return np.asarray(to_host(array), dtype=np.float64)


def _digest(*parts: Any) -> str:
    h = hashlib.sha256(SCHEMA.encode())
    for part in parts:
        if isinstance(part, np.ndarray):
            h.update(f"{part.dtype}{part.shape}".encode())
            h.update(np.ascontiguousarray(part).tobytes())
        else:
            h.update(repr(part).encode())
    return h.hexdigest()


# --- control / policy ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PondEventControl:
    """Numerical options. `time` (existing `ExperimentalControl`) bounds the adaptive retry: `max_dt_s`, `min_dt_s`,
    `max_retries` per step and `max_steps` accepted steps per `evolve` call. `backend`: hydraulics "numpy" / "numba" /
    "cuda" (the namespace of `column`, the sediment parameters and the bed must match). `sediment_courant_max` and
    `max_transport_substeps` bound the transport substeps (a request above the bound is a `TransportStepRejected` that
    halves dt). Slope choices: see the module docstring."""

    backend: str = "numpy"
    time: ExperimentalControl = field(default_factory=ExperimentalControl)
    # NEW pond default: the implicit quadratic friction treatment (the lagged update gives Fr ~ 185 at first wetting).
    # The global `HydraulicControl()` stays "lagged"; `PondEventControl(hydraulic=HydraulicControl())` selects the
    # original qualified lagged variant explicitly.
    hydraulic: HydraulicControl = field(
        default_factory=lambda: HydraulicControl(friction_treatment="implicit_quadratic"))
    phase_bins: int = DEFAULT_N_BINS
    sediment_courant_max: float = DEFAULT_COURANT_MAX
    max_transport_substeps: int = 1024
    splash_implementation: str = "array"
    shear_slope: str = "friction"
    stream_power_slope: str = "friction"
    froude_reference: float = 0.5
    max_commit_log: int = 100
    # Resident-window bed options (defaults LAST; `repr=False` so every dense configuration digest is byte-identical to
    # the earlier one; a NON-default value is appended to the digest explicitly, see `_control_identity`).
    #  - `bed_storage`: "dense" (default) or "windowed" (an actual MAPLE `WindowedVoxelColumnState` bed; declared, never
    #    inferred, so a dense event can never silently accept a windowed context option).
    #  - `deep_publication`: WHEN MAPLE's pending deep journal becomes a new deep-store version. "step" (default) =
    #    after every accepted step (historical dense-compatible durability). "boundary" = the journal is CARRIED through
    #    accepted steps and published only at `deep_publication_high_water_records`, at the return of `evolve`, in
    #    `finalize`, in the explicit `publish_storage`, and in the single flush before a recognised append-cap retry.
    #    Physical state, commits and clocks are identical; only deep-file durability (versions) differs.
    bed_storage: str = field(default="dense", repr=False)
    deep_publication: str = field(default="step", repr=False)
    deep_publication_high_water_records: int | None = field(default=None, repr=False)

    def validated(self) -> PondEventControl:
        if self.backend not in ("numpy", "numba", "cuda"):
            raise PondEventError(f"backend must be numpy, numba or cuda, got {self.backend!r}")
        self.time.validated()
        self.hydraulic.validated()
        if self.hydraulic.limiter != "off":
            raise PondEventError("the donor limiter changes the dynamics and is refused (limiter 'off')")
        for name in ("phase_bins", "max_transport_substeps", "max_commit_log"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
                raise PondEventError(f"{name} must be a positive int, got {v!r}")
        cmax = _real(self.sediment_courant_max, "sediment_courant_max")
        if not (0.0 < cmax <= DEFAULT_COURANT_MAX):
            raise PondEventError(f"sediment_courant_max must lie in (0, {DEFAULT_COURANT_MAX}]")
        if self.splash_implementation not in SPLASH_IMPLEMENTATIONS:
            raise PondEventError(f"splash_implementation must be one of {SPLASH_IMPLEMENTATIONS}")
        for name in ("shear_slope", "stream_power_slope"):
            if getattr(self, name) not in SLOPE_CHOICES:
                raise PondEventError(f"{name} must be one of {SLOPE_CHOICES}")
        _positive(self.froude_reference, "froude_reference")
        if self.bed_storage not in BED_STORAGES:
            raise PondEventError(f"bed_storage must be one of {BED_STORAGES}, got {self.bed_storage!r}")
        if self.deep_publication not in DEEP_PUBLICATIONS:
            raise PondEventError(f"deep_publication must be one of {DEEP_PUBLICATIONS}, got {self.deep_publication!r}")
        hw = self.deep_publication_high_water_records
        if self.deep_publication == "boundary":
            if self.bed_storage != "windowed":
                raise PondEventError("deep_publication 'boundary' needs bed_storage 'windowed'")
            if isinstance(hw, bool) or not isinstance(hw, int) or hw < 1:
                raise PondEventError("deep_publication 'boundary' needs deep_publication_high_water_records, a positive int")
        elif hw is not None:
            raise PondEventError("deep_publication_high_water_records is an option of deep_publication 'boundary'")
        return self


@dataclass(frozen=True)
class PondCompletionPolicy:
    """Quiet gates (defaults mirror `CompletionPolicy`). `max_depth_change_m` bounds the SUM over the hold of the per-step
    max |depth change| (a conservative upper bound of any cell's net change, no per-cell history is stored).
    `mobile_per_cell_class_kg` None uses MAPLE's physical significance threshold (the mass resolution, NOT the FP bound)."""

    hold_s: float = 60.0
    max_depth_change_m: float = 1e-8
    max_unit_discharge_m2_s: float = 1e-10
    outlet_discharge_m3_s: float = 1e-9
    mobile_per_cell_class_kg: float | None = None
    deposit_mobile_at_flow_end: bool = False
    # Physical quiet thresholds (the existing CompletionPolicy defaults, NOT conservation tolerances) applied to the
    # MOVING / EXCESS water: excess depth = clip(eta - escape, 0, h) on outlet-connected cells (0 on closed components).
    max_depth_m: float = 1e-8
    surface_volume_m3: float = 1e-6

    def validated(self) -> PondCompletionPolicy:
        for name in ("hold_s", "max_depth_change_m", "max_unit_discharge_m2_s", "outlet_discharge_m3_s",
                     "max_depth_m", "surface_volume_m3"):
            _positive(getattr(self, name), name)
        if self.mobile_per_cell_class_kg is not None:
            _positive(self.mobile_per_cell_class_kg, "mobile_per_cell_class_kg")
        if not isinstance(self.deposit_mobile_at_flow_end, bool):
            raise PondEventError("deposit_mobile_at_flow_end must be bool")
        return self


# --- state ---------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PondDatum:
    """Stable terrain evidence of the event (host, immutable): the initial FULL physical bed `(ny+2, nx+2)` and the MAPLE
    committed elevation it was bound to. The held physical source is always `initial_full + (committed - initial_committed)`
    on active cells, ring and inactive cells fixed."""

    initial_full_m: np.ndarray
    initial_committed_m: np.ndarray

    def __post_init__(self) -> None:
        for a in (self.initial_full_m, self.initial_committed_m):
            a.flags.writeable = False  # frozen against accidental mutation


@dataclass(frozen=True)
class PondQuiet:
    since_s: float | None = None
    last_t_s: float | None = None
    window_depth_change_m: float = 0.0
    ready: bool = False


@dataclass(frozen=True)
class PondCounters:
    n_accepted_steps: int = 0
    n_rejected_attempts: int = 0
    n_commits: int = 0
    n_maple_trigger_commits: int = 0
    n_net_elevation_commits: int = 0
    n_final_commits: int = 0
    n_refreshes: int = 0
    n_splash_exchanges: int = 0
    max_gate_displacement_m: float = 0.0
    max_face_froude: float = 0.0
    n_faces_above_froude_reference: int = 0
    # largest transport substep count actually EXECUTED by an accepted step (0 iff none): a host diagnostic, no history
    max_transport_substeps_used: int = 0
    # Resident-window storage counters (all 0 for a dense bed): deep-store versions this event published (every
    # publication: per step, high-water, return, finalize, explicit, flush), the subset that were the single flush before
    # a recognised append-cap retry, and the largest pending journal (records) carried by any accepted state.
    n_storage_publications: int = 0
    n_journal_flushes: int = 0
    max_carried_records: int = 0
    commit_log: tuple = ()


@dataclass(frozen=True, eq=False)
class WaterLedger:
    """Cumulative accepted-step water ledger (m, m3; arrays in the namespace). Same quantities as `PondLedger`, which
    evaluates the closure. `reset_*` hold the surface / soil depth removed by the explicit dry reset (None before)."""

    t0_s: float
    t_end_s: float
    depth0: Any
    soil0: Any
    rain: Any
    intake: Any
    sat_return: Any
    drain: Any
    pos_x: Any
    pos_y: Any
    neg_x: Any
    neg_y: Any
    net_x: Any
    net_y: Any
    export: Any
    n_steps: int
    reset_depth: Any = None
    reset_soil: Any = None


@dataclass(frozen=True, eq=False)
class SedimentLedger:
    """Cumulative accepted-step sediment ledger. `*_by_class` `(nc,)` kg (device); cell / face arrays are cumulative
    totals, not history. `inventory0` / `mobile0` are the host class totals at the start (bed inventory, mobile pool)."""

    inventory0: np.ndarray
    mobile0: np.ndarray
    wet_requested: Any
    wet_actual: Any
    wet_shortfall: Any
    splash_requested: Any
    splash_actual: Any
    deposited_wet: Any
    deposited_splash: Any
    export_wet: Any
    export_splash: Any
    deposited_terminal: Any
    deposit_unmet: Any
    pickup_cell: Any
    deposition_cell: Any
    x_gross: Any
    y_gross: Any
    x_net: Any
    y_net: Any
    numerical_residual_kg: Any
    phase_residual_max_kg: Any
    n_phase_canonicalized: Any
    n_phase_remnants: Any
    n_native_calls: int


@dataclass(frozen=True, eq=False)
class PondEventState:
    hydraulic: HydraulicState
    bed: BedState
    sediment_velocity_m_s: Any
    phase_fraction: Any
    phase_position_m: Any
    topology: PondTopology
    held_source_m: Any
    datum: PondDatum
    water: WaterLedger
    sediment: SedimentLedger
    counters: PondCounters
    quiet: PondQuiet
    config_sha256: str
    forcing_sha256: str
    status: str = "wet"
    schema: str = SCHEMA

    @property
    def t_s(self) -> float:
        return self.hydraulic.t_s


@dataclass(frozen=True, eq=False)
class PondStepRecord:
    """Diagnostics of ONE accepted step (not retained by the state). `by_class` holds `(nc,)` device vectors of the step."""

    t0_s: float
    t1_s: float
    dt_s: float
    n_substeps: int
    n_transport_rejections: int
    froude: dict
    metrics: dict
    commit: dict | None
    refreshed: bool
    splash_exchange: bool
    by_class: dict
    phase: dict
    quiet: PondQuiet
    # per-step cell/class mobile balance (max residual / max MAPLE bound, kg) and the four native numerical residuals
    balance: dict = field(default_factory=dict)
    numerical_residuals: dict = field(default_factory=dict)
    # windowed beds: {"pending_records": records the RETURNED state carries unpublished, "published": whether this step
    # published its journal ('step' policy)}; empty for a dense bed
    storage: dict = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class PondCoupledStep:
    state: PondEventState
    record: PondStepRecord


@dataclass(frozen=True, eq=False)
class PondEvolveResult:
    """`state` is the last ACCEPTED state. `stop_reason` None = the target time was reached; otherwise "max_retries",
    "min_dt", "max_steps", "time_resolution" or "quiet" (stop_when_quiet). A bound stop is INCOMPLETE: nothing was
    continued, hidden or reset."""

    state: PondEventState
    requested_end_s: float
    elapsed_s: float
    reached_target: bool
    stop_reason: str | None
    quiet_ready: bool
    n_accepted_steps: int
    n_rejected_attempts: int
    rejections: tuple
    min_accepted_dt_s: float
    max_accepted_dt_s: float
    last_record: PondStepRecord | None
    # windowed beds: publications this call made, by reason (`high_water`, `flush`, `return`) and the largest pending
    # journal carried by an accepted state of the call; empty for a dense bed
    storage: dict = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class PondDryHandoff:
    dry_state: PondEventState
    pre_reset_surface_m: Any
    pre_reset_soil_m: Any
    terminal_deposition_kg: Any
    accounting: dict
    status: str = "complete"


@dataclass(frozen=True, eq=False)
class _Runtime:
    """Derived, rebuildable objects of one held topology (never part of a state)."""

    topology: PondTopology
    pond: PondHydraulics
    network: FaceNetwork
    grid: Any
    splash_grid: Any
    active: Any
    friction: Any
    z_held: Any
    escape: Any
    outlet_x: Any
    outlet_y: Any


def _canonical_where_empty(xp: Any, fraction: Any, position: Any, mobile: Any) -> tuple[Any, Any, Any]:
    """Canonical phase (fraction e_0, position 0) where the ACTUAL pool is exactly 0; elsewhere untouched. Returns the
    arrays and the number of canonicalised cell/classes. No mass is changed."""
    from maple.core.backend import freeze

    empty = (mobile == 0.0)[..., None]
    canonical = xp.zeros((1, 1, 1, fraction.shape[-1]), dtype=np.float64)
    canonical[..., 0] = 1.0
    changed = xp.sum(empty[..., 0] & ((fraction != canonical).any(axis=-1) | (position != 0.0).any(axis=-1)))
    return freeze(xp.where(empty, canonical, fraction)), freeze(xp.where(empty, 0.0, position)), changed


# --- the event -------------------------------------------------------------------------------------
def cell_reduction_bound(n_cells: int, *nonnegative_class_sums: np.ndarray) -> np.ndarray:
    """Order-independent absolute bound of the cell reductions that produced the given class sums (non-negative terms):
    `sequential_bound_scale_factor(n_cells) * sum(values)`. Pure, diagnostic."""
    from maple.core.backend.reductions import sequential_bound_scale_factor

    total = np.zeros_like(np.asarray(nonnegative_class_sums[0], dtype=np.float64))
    for values in nonnegative_class_sums:
        total = total + np.asarray(values, dtype=np.float64)
    return sequential_bound_scale_factor(int(n_cells)) * total


class PondEvent:
    """Static configuration plus a runtime cache. See the module docstring.

    `topology` is the initial `PondTopology` (built from the full physical bed, explicit outlets; closed components only
    if explicitly allowed there). `column` / `sediment` / `bed` live in ONE namespace (`column.xp`). `vegetation_cover_fraction`
    and `rain_scale` are optional host `(ny, nx)` arrays (default: zero cover, 1 on active cells). `splash` (optional) turns
    on dry rain splash with the explicit per-class inverse distances."""

    def __init__(self, context: BedContext, column: ColumnParameters, sediment: SedimentPhysicsParameters,
                 schedule: RainfallSchedule, topology: PondTopology, *, control: PondEventControl | None = None,
                 policy: PondCompletionPolicy | None = None, splash: SplashParameters | None = None,
                 vegetation_cover_fraction: np.ndarray | None = None, rain_scale: np.ndarray | None = None):
        if not isinstance(topology, PondTopology):
            raise PondEventError("topology must be a PondTopology")
        if not isinstance(schedule, RainfallSchedule):
            raise PondEventError("schedule must be a RainfallSchedule")
        if not isinstance(column, ColumnParameters) or not isinstance(sediment, SedimentPhysicsParameters):
            raise PondEventError("column and sediment must be ColumnParameters / SedimentPhysicsParameters")
        self.control = (PondEventControl() if control is None else control).validated()
        self.policy = (PondCompletionPolicy() if policy is None else policy).validated()
        xp = column.xp
        if sediment.xp is not xp or (self.control.backend == "cuda") != (xp.__name__ == "cupy"):
            raise PondEventError("column, sediment parameters and backend must share one namespace "
                                 "(backend 'cuda' <=> CuPy; no fallback)")
        nc = len(context.grain_classes.classes)
        if sediment.n_classes != nc:
            raise PondEventError(f"sediment has {sediment.n_classes} classes, the MAPLE case {nc}")
        if splash is not None:
            if not isinstance(splash, SplashParameters) or splash.n_classes != nc or splash.xp is not xp:
                raise PondEventError("splash must be SplashParameters of the same classes and namespace")
        shape = topology.shape
        if (context.geometry.ny, context.geometry.nx) != shape or context.geometry.dx_m != topology.dx_m:
            raise PondEventError("bed context geometry does not match the topology (shape, dx)")
        require_supported_bed_context(context, self.control)
        self.windowed = self.control.bed_storage == "windowed"
        if self.windowed:
            require_paired_entry()  # unsupported MAPLE dependency: refused at construction, never downgraded
        # exact physical class checks (as the old _check_event_inputs): diameters in order and the particle density
        diameters = np.array([c.diameter_m for c in context.grain_classes.classes], dtype=np.float64)
        if not np.array_equal(diameters, _host(sediment.diameter_m)):
            raise PondEventError("sediment parameter diameters (values and ORDER) differ from the MAPLE grain classes")
        if {float(c.particle_density_kg_m3) for c in context.grain_classes.classes} != {
                float(sediment.particle_density_kg_m3)}:
            raise PondEventError("sediment particle density differs from the MAPLE grain classes")
        active = np.asarray(topology.active)
        veg = np.zeros(shape) if vegetation_cover_fraction is None else np.array(vegetation_cover_fraction, np.float64)
        scale = np.where(active, 1.0, 0.0) if rain_scale is None else np.where(active, np.array(rain_scale, np.float64), 0.0)
        for name, a in (("vegetation_cover_fraction", veg), ("rain_scale", scale)):
            if a.shape != shape or not np.isfinite(a).all() or (a < 0).any():
                raise PondEventError(f"{name} must be a finite >= 0 array of shape {shape}")
        if (veg > 1.0).any():
            raise PondEventError("vegetation_cover_fraction must be <= 1")
        self.context, self.column, self.sediment, self.schedule, self.splash = context, column, sediment, schedule, splash
        self.xp, self.nc, self.shape, self._initial_topology = xp, nc, shape, topology
        self.veg, self.rain_scale = xp.asarray(veg), xp.asarray(scale)
        self.forcing_sha256 = _digest(schedule.edges_s, schedule.intensity_mm_per_h, scale)
        col = [self.column.model] + [_host(getattr(column, n)) for n in (
            "ksat_m_per_s", "suction_m", "drainage_parameter", "theta_sat", "soil_thickness_m")]
        self.config_sha256 = _digest(
            topology.identity_sha256, *col, repr(sediment.summary()), None if splash is None else repr(splash.summary()),
            veg, self.forcing_sha256, repr(self.control), repr(self.policy), context.mass_resolution_kg,
            *_control_identity(self.control, context),
            # the FULL native context: class ids / order / diameters / densities / aggregate flags, geometry (dx, dy, dz,
            # bulk density, active layer, boundaries), commit and avalanche specs, depth rule, initial availability
            tuple((c.class_id, c.diameter_m, c.particle_density_kg_m3, c.is_aggregate)
                  for c in context.grain_classes.classes),
            repr(context.geometry), repr(context.topographic_commit_spec), repr(context.avalanching_spec),
            context.depth_update_rule, _host(context.initial_available_fraction))
        self._cache: _Runtime | None = None
        self._blessed: weakref.WeakSet = weakref.WeakSet()  # states this event built or fully validated

    # -- runtime --------------------------------------------------------------------------------------
    def _runtime(self, topology: PondTopology, hydraulic: HydraulicState | None) -> _Runtime:
        """Runtime of `topology`: the cached one, or rebuilt (rebinding the previous solver when the identity is the same;
        the carried hydraulic state, when given, is validated against the new faces either way)."""
        xp, ctl = self.xp, self.control
        cached = self._cache
        if cached is not None and cached.topology.topology_sha256 == topology.topology_sha256:
            return cached
        if cached is not None and hydraulic is not None and cached.topology.identity_sha256 == topology.identity_sha256:
            pond, _ = rebind_pond(cached.pond, topology, hydraulic)
            network = cached.network
        else:
            pond = build_pond_hydraulics(topology, self.column, backend=ctl.backend, control=ctl.hydraulic)
            if hydraulic is not None:
                pond.solver.validate_state(hydraulic)
            network = face_network_from_pond_topology(topology, xp=xp)
        slope, aspect = steepest_d4_descent(topology.source_z_full, np.asarray(topology.active), topology.dx_m,
                                            topology.outlets)
        active = np.asarray(topology.active)
        grid = physics_grid(slope, active, topology.dx_m * topology.dx_m, xp=xp)
        sgrid = None
        if self.splash is not None:
            sgrid = splash_grid(slope, aspect, active, topology.dx_m, open_x_faces=network.fx_type_host == 2,
                                open_y_faces=network.fy_type_host == 2, xp=xp)
        runtime = _Runtime(
            topology=topology, pond=pond, network=network, grid=grid, splash_grid=sgrid, active=xp.asarray(active),
            friction=xp.asarray(np.where(active, topology.friction, 0.0)),
            z_held=xp.asarray(np.array(topology.source_bed_m)), escape=xp.asarray(np.array(topology.escape_level_m)),
            outlet_x=network.fx_type == 2, outlet_y=network.fy_type == 2)
        self._cache = runtime
        return runtime

    def _check_state(self, state: Any) -> _Runtime:
        """Cheap path for a state this event built (or already fully validated): digests and the cached runtime. Any other
        state object (restart, `dataclasses.replace`, a hand-built state) gets the FULL validation once: source proof
        (the held source must reproduce the topology source EXACTLY from the frozen datum; one host transfer of this
        source object), metadata/namespace/shape/dtype, clock, bed depth == solver depth, inactive-cell convention, phase.
        No mass is repaired."""
        if not isinstance(state, PondEventState) or state.schema != SCHEMA:
            raise PondEventError("state must be a PondEventState of this schema")
        if state.status not in ("wet", "dry"):
            raise PondEventError(f"unknown state status {state.status!r}")
        if state.config_sha256 != self.config_sha256 or state.forcing_sha256 != self.forcing_sha256:
            raise PondEventError("state was built under a different event configuration or forcing")
        if state.topology.identity_sha256 != self._initial_topology.identity_sha256:
            raise PondEventError("state topology identity differs from the event's")
        cached = self._cache
        if state in self._blessed and cached is not None and cached.topology.topology_sha256 == state.topology.topology_sha256:
            return cached
        rt = self._validate_full(state)
        self._blessed.add(state)
        return rt

    def _validate_full(self, state: PondEventState) -> _Runtime:
        from maple.core.backend import (
            DeferredChecks,
            MixedArrayNamespaceError,
            array_namespace,
            finite_flag,
            is_array,
            negative_flag,
            true_flag,
        )
        from maple_syrup.sediment_transport import TransportError

        xp, (ny, nx), nc = self.xp, self.shape, self.nc
        topo, hyd, datum = state.topology, state.hydraulic, state.datum
        if topo.shape != self.shape:
            raise PondEventError("state topology shape differs from the event's")
        self._check_bed(state.bed)
        try:
            check_routing_source(state.held_source_m, self.shape, xp)
        except Exception as exc:
            raise PondEventError(f"held source: {exc}") from None
        if (not isinstance(datum, PondDatum) or datum.initial_full_m.shape != (ny + 2, nx + 2)
                or datum.initial_committed_m.shape != self.shape):
            raise PondEventError("datum evidence has the wrong type or shape")
        expected = physical_full_source(datum.initial_full_m, datum.initial_committed_m, _host(state.held_source_m),
                                        np.asarray(topo.active))
        if not np.array_equal(expected, topo.source_z_full):
            raise PondEventError("held source and topology source differ (source identity / generation mismatch)")
        rt = self._runtime(topo, hyd)
        rt.pond.solver.validate_state(hyd)
        nb = self.control.phase_bins
        named = {"bed.water.depth_m": (state.bed.water.depth_m, (ny, nx)),
                 "bed.water.mobile": (state.bed.water.mobile_mass_by_cell_class_kg, (ny, nx, nc)),
                 "sediment_velocity_m_s": (state.sediment_velocity_m_s, (ny, nx, nc)),
                 "phase_fraction": (state.phase_fraction, (ny, nx, nc, nb)),
                 "phase_position_m": (state.phase_position_m, (ny, nx, nc, nb))}
        for name, (a, shape) in named.items():
            if not is_array(a) or tuple(a.shape) != shape or a.dtype != np.float64:
                raise PondEventError(f"{name} must be a float64 array of shape {shape}")
        try:
            namespace = array_namespace(*(a for a, _ in named.values()), hyd.depth_m, state.held_source_m)
        except MixedArrayNamespaceError as exc:
            raise PondEventError(f"state arrays are in mixed namespaces: {exc}") from None
        if namespace is not xp:
            raise PondEventError(f"state arrays must be in the event namespace {xp.__name__!r}")
        t = check_state_time(hyd.t_s)
        if state.water.t_end_s != t or state.water.t0_s > t:
            raise PondEventError("ledger clock differs from the hydraulic clock")
        if state.quiet.last_t_s is not None and state.quiet.last_t_s > t:
            raise PondEventError("quiet monitor is ahead of the clock")
        used, accepted = state.counters.max_transport_substeps_used, state.counters.n_accepted_steps
        if (isinstance(used, bool) or not isinstance(used, (int, np.integer)) or used < 0
                or used > self.control.max_transport_substeps or (used == 0) != (accepted == 0)):
            raise PondEventError(
                f"max_transport_substeps_used = {used!r} is inconsistent (nonnegative int, <= the control cap "
                f"{self.control.max_transport_substeps}, 0 iff no accepted step; accepted = {accepted})")
        v = state.sediment_velocity_m_s
        inactive = ~rt.active
        checks = DeferredChecks()
        checks.require(finite_flag(v), "sediment_velocity_m_s must be finite")
        checks.forbid(negative_flag(v), "sediment_velocity_m_s must be >= 0")
        checks.forbid(true_flag(state.bed.water.depth_m != hyd.depth_m), "bed.water.depth_m must equal the solver depth")
        checks.forbid(true_flag(inactive & ((hyd.depth_m != 0.0) | (hyd.soil_water_m != 0.0))),
                      "water or soil on an inactive cell (the event requires inactive cells to hold none)")
        checks.forbid(true_flag(inactive[..., None] & (state.bed.water.mobile_mass_by_cell_class_kg != 0.0)),
                      "mobile mass on an inactive cell")
        try:
            checks.resolve()
        except ValueError as exc:
            raise PondEventError(str(exc)) from None
        try:
            validate_phase(rt.network, state.bed.water.mobile_mass_by_cell_class_kg,
                           PhaseState(fraction=state.phase_fraction, position_m=state.phase_position_m, dx_m=topo.dx_m,
                                      n_bins=nb, xp=xp))
        except TransportError as exc:
            raise PondEventError(f"phase state invalid: {exc}") from None
        return rt

    def _check_bed(self, bed: BedState) -> None:
        """Bed-level support checks of a bed handed to this event (initial state, full state validation): the shared
        `pond_bed.require_supported_bed` (windowed allowed only when the control declares `bed_storage="windowed"`), the
        declared storage against the bed's actual type, and for a windowed bed MAPLE's cheap column check (header,
        geometry, class ids, generation AND manifest digest, pending-journal binding) and a carried journal within the
        actual cap. One batched flag read; nothing is mutated, published or deep-read."""
        require_supported_bed(bed, self.context, allow_windowed=self.windowed)
        if is_windowed_bed(bed) != self.windowed:
            raise PondBedError(f"the bed is {'windowed' if is_windowed_bed(bed) else 'dense'} but the control declares "
                               f"bed_storage={self.control.bed_storage!r}")
        if not self.windowed:
            return
        from maple.surface.voxels.windowed_inventory import check_windowed_column_cheap

        try:
            check_windowed_column_cheap(bed.voxel_column, self.context.geometry,
                                        class_ids=tuple(c.class_id for c in self.context.grain_classes.classes))
        except (ValueError, TypeError) as exc:
            raise PondBedError(f"windowed bed: {exc}") from None
        carried, cap = pending_journal_records(bed), resolved_journal_cap(self.context)
        if carried > cap:
            raise PondBedError(f"the bed carries {carried} pending journal records, above the actual cap {cap}")

    # -- initial state ----------------------------------------------------------------------------------
    def initial_state(self, bed: BedState, soil_water_m: Any, *, t_s: float = 0.0) -> PondEventState:
        """Event state from an ACTUAL MAPLE bed (its `water.depth_m` is the initial surface depth, its mobile pool the
        initial load, which sits at the upstream face) and the retained soil water. The held source is the bed's
        current committed elevation; the topology must have been built from the same physical bed (constant datum)."""
        from maple.core.backend import array_namespace
        from maple.core.types.topographic_commit import committed_elevation_m

        context, xp = self.context, self.xp
        self._check_bed(bed)
        if array_namespace(bed.water.depth_m) is not xp:
            raise PondEventError("the bed is not in the event namespace (no host/device fallback)")
        topology = self._initial_topology
        active = np.asarray(topology.active)
        committed = committed_elevation_m(bed.committed_topography)
        c0 = _host(committed).copy()
        if c0.shape != self.shape or not np.isfinite(c0).all():
            raise PondEventError("MAPLE committed elevation does not match the topology grid")
        offset = (c0 - topology.source_bed_m)[active]
        tol = 128 * np.finfo(float).eps * max(1.0, float(np.abs(topology.source_z_full).max()), float(np.abs(c0).max()))
        if np.max(np.abs(offset - offset.flat[0])) > tol:
            raise PondEventError("the topology bed differs from the MAPLE terrain by more than a constant datum")
        from maple.core.backend import freeze

        held = freeze(xp.array(committed, copy=True))
        runtime = self._runtime(topology, None)  # reuses the cached prepared solver of this topology when clean
        hydraulic = runtime.pond.initial_state(bed.water.depth_m, soil_water_m, t_s=_real(t_s, "t_s"))
        mobile = bed.water.mobile_mass_by_cell_class_kg
        phase = empty_phase_state(self.shape, self.nc, self.control.phase_bins, topology.dx_m, xp=xp)
        inv0 = _host(bed_inventory(bed))
        mob0 = _host(mobile.sum(axis=(0, 1)))
        z = lambda *s: xp.zeros(s, dtype=np.float64)  # noqa: E731
        ny, nx = self.shape
        nc = self.nc
        water = WaterLedger(
            t0_s=hydraulic.t_s, t_end_s=hydraulic.t_s, depth0=xp.array(hydraulic.depth_m, copy=True),
            soil0=xp.array(hydraulic.soil_water_m, copy=True), rain=z(ny, nx), intake=z(ny, nx), sat_return=z(ny, nx),
            drain=z(ny, nx), pos_x=z(ny, nx + 1), pos_y=z(ny + 1, nx), neg_x=z(ny, nx + 1), neg_y=z(ny + 1, nx),
            net_x=z(ny, nx + 1), net_y=z(ny + 1, nx), export=z(), n_steps=0)
        vec = lambda: z(nc)  # noqa: E731
        sediment = SedimentLedger(
            inventory0=inv0, mobile0=mob0, wet_requested=vec(), wet_actual=vec(), wet_shortfall=vec(),
            splash_requested=vec(), splash_actual=vec(), deposited_wet=vec(), deposited_splash=vec(), export_wet=vec(),
            export_splash=vec(), deposited_terminal=vec(), deposit_unmet=vec(), pickup_cell=z(ny, nx, nc),
            deposition_cell=z(ny, nx, nc), x_gross=z(ny, nx + 1, nc), y_gross=z(ny + 1, nx, nc),
            x_net=z(ny, nx + 1, nc), y_net=z(ny + 1, nx, nc), numerical_residual_kg=z(), phase_residual_max_kg=z(),
            n_phase_canonicalized=z(), n_phase_remnants=z(), n_native_calls=0)
        state = PondEventState(
            hydraulic=hydraulic, bed=bed, sediment_velocity_m_s=z(ny, nx, nc), phase_fraction=phase.fraction,
            phase_position_m=phase.position_m, topology=topology, held_source_m=held,
            datum=PondDatum(np.array(topology.source_z_full, copy=True), c0), water=water, sediment=sediment,
            counters=PondCounters(), quiet=PondQuiet(), config_sha256=self.config_sha256,
            forcing_sha256=self.forcing_sha256)
        self._check_state(state)
        return state

    # -- quiet metrics ----------------------------------------------------------------------------------
    def _mobile_limit(self) -> float:
        p = self.policy.mobile_per_cell_class_kg
        return self.context.mass_resolution_kg if p is None else p

    def _metrics(self, rt: _Runtime, h_prev: Any, hyd: HydraulicState, mobile: Any, export_m3: Any) -> dict:
        """ONE host packet: max |depth change|, max |unit flux|, export, cells above their escape level, max mobile,
        finite flag (and the outlet discharge from the face fluxes)."""
        xp, dx = self.xp, rt.topology.dx_m
        h, qx, qy = hyd.depth_m, hyd.qx_m2_s, hyd.qy_m2_s
        eta = rt.z_held + h
        over = xp.sum(rt.active & (eta > rt.escape))  # RAW count: diagnostic only, never a quiet criterion
        # EXCESS (moving) water: clip(eta - escape, 0, h) on outlet-connected cells, 0 on closed components (escape = inf);
        # the physical depth h itself is never changed or removed
        excess = xp.where(rt.active, xp.minimum(xp.maximum(eta - rt.escape, 0.0), h), 0.0)
        finite = xp.all(xp.isfinite(h)) & xp.all(xp.isfinite(qx)) & xp.all(xp.isfinite(qy)) & xp.all(xp.isfinite(mobile))
        outlet = dx * (xp.sum(xp.where(rt.outlet_x, xp.abs(qx), 0.0)) + xp.sum(xp.where(rt.outlet_y, xp.abs(qy), 0.0)))
        packet = host_packet(xp, [xp.max(xp.abs(h - h_prev)), xp.maximum(xp.max(xp.abs(qx)), xp.max(xp.abs(qy))),
                                  export_m3, over, mobile.max(), finite, outlet, xp.max(excess),
                                  xp.sum(excess) * dx * dx])
        if packet[5] != 1.0:
            raise PondEventError("non-finite physical state after the step")
        return {"max_abs_depth_change_m": float(packet[0]), "max_unit_discharge_m2_s": float(packet[1]),
                "export_m3": float(packet[2]), "n_cells_above_escape": int(packet[3]),
                "max_mobile_cell_class_kg": float(packet[4]), "outlet_discharge_m3_s": float(packet[6]),
                "max_excess_depth_m": float(packet[7]), "excess_volume_m3": float(packet[8])}

    def _excess_ok(self, m: dict) -> bool:
        """Moving / excess water under BOTH original CompletionPolicy-style physical thresholds."""
        return (m["max_excess_depth_m"] <= self.policy.max_depth_m
                and m["excess_volume_m3"] <= self.policy.surface_volume_m3)

    def _quiet_advance(self, quiet: PondQuiet, t0: float, t1: float, dt: float, m: dict) -> PondQuiet:
        pol = self.policy
        rain_done = t0 >= self.schedule.end_s
        mobile_ok = pol.deposit_mobile_at_flow_end or m["max_mobile_cell_class_kg"] <= self._mobile_limit()
        qualifies = (rain_done and m["max_unit_discharge_m2_s"] <= pol.max_unit_discharge_m2_s
                     and m["export_m3"] / dt <= pol.outlet_discharge_m3_s and self._excess_ok(m) and mobile_ok)
        if not qualifies:
            return PondQuiet(None, t1, 0.0, False)
        since = t0 if quiet.since_s is None else quiet.since_s
        change = (0.0 if quiet.since_s is None else quiet.window_depth_change_m) + m["max_abs_depth_change_m"]
        if change > pol.max_depth_change_m:
            return PondQuiet(None, t1, 0.0, False)
        return PondQuiet(since, t1, change, bool(t1 - since >= pol.hold_s))

    def quiet_metrics(self, state: PondEventState) -> tuple[bool, dict]:
        """Instantaneous gates of the CURRENT state (rain finished, flux, outlet discharge, escape levels, mobile) plus the
        monitor (`ready` needs the continuous hold). Returns `(qualifies_now, values)`."""
        rt = self._check_state(state)
        h = state.hydraulic.depth_m
        m = self._metrics(rt, h, state.hydraulic, state.bed.water.mobile_mass_by_cell_class_kg, self.xp.zeros(()))
        pol = self.policy
        mobile_ok = pol.deposit_mobile_at_flow_end or m["max_mobile_cell_class_kg"] <= self._mobile_limit()
        ok = (state.t_s >= self.schedule.end_s and m["max_unit_discharge_m2_s"] <= pol.max_unit_discharge_m2_s
              and m["outlet_discharge_m3_s"] <= pol.outlet_discharge_m3_s and self._excess_ok(m) and mobile_ok)
        return bool(ok), {**m, "rainfall_finished": bool(state.t_s >= self.schedule.end_s),
                          "mobile_per_element_limit_kg": float(self._mobile_limit()), "quiet": state.quiet}

    # -- geometry gate / commit -----------------------------------------------------------------------------
    def _commit_and_gate(self, bed: BedState, held: Any, rt: _Runtime, topology: PondTopology, datum: PondDatum,
                         hydraulic: HydraulicState, t_s: float, *, final: bool):
        """Gate (every call) -> native commit (MAPLE trigger, net-elevation force, or final force) -> refresh only on a net
        crossing. Returns `(bed, held, rt, topology, record, refreshed, gate_m)`; `record["committed"]` tells whether a
        native commit happened. Pure (the runtime cache may hold the unpublished refreshed runtime; it is keyed by digest)."""
        from maple.core.types.topographic_commit import committed_elevation_m

        ctx, xp = self.context, self.xp
        threshold = reroute_threshold_m(ctx)
        gate = gate_displacement_m(bed, ctx, held, rt.active)
        commit: NativeCommit = native_commit(bed, ctx, t_s, force=bool(final or gate >= threshold))
        if not commit.did_commit:
            return bed, held, rt, topology, {"gate_m": gate, "committed": False}, False, gate
        trigger = commit.trigger.should_commit
        reason = "maple_trigger" if trigger else ("net_elevation" if gate >= threshold else "final_forced")
        bed = commit.state
        net = committed_displacement_m(bed, held, rt.active)
        record = {"t_s": t_s, "reason": reason, "gate_before_m": gate, "net_after_m": net, "threshold_m": threshold,
                  "committed": True, "refreshed": False, "generation": topology.generation}
        if not net >= threshold:
            return bed, held, rt, topology, record, False, gate
        committed_dev = committed_elevation_m(bed.committed_topography)
        full = physical_full_source(datum.initial_full_m, datum.initial_committed_m, _host(committed_dev),
                                    np.asarray(topology.active))  # the one full-surface transfer
        update = update_pond_topology(topology, full, threshold)
        if update.refreshed:
            new_topology = update.topology
        else:  # round-off of the datum shift disagrees with the committed-datum gate: the gate decides
            rebuilt = build_pond_topology(full, np.asarray(topology.active), np.asarray(topology.friction), topology.dx_m,
                                          topology.outlets, nodata_value=topology.nodata_value,
                                          allow_closed_components=topology.n_closed_components > 0)
            new_topology = replace(rebuilt, generation=topology.generation + 1)
        new_rt = self._runtime(new_topology, hydraulic)
        record.update({"refreshed": True, "generation": new_topology.generation})
        from maple.core.backend import freeze

        return bed, freeze(xp.array(committed_dev, copy=True)), new_rt, new_topology, record, True, gate

    # -- the accepted coupled step ----------------------------------------------------------------------------
    def step(self, state: PondEventState, rain_rate_m_per_s: float, dt_s: float, *,
             t_end_s: float | None = None) -> PondCoupledStep:
        """ONE accepted coupled physical step with a rain rate constant over `dt_s` (scalar `m/s`, times the event's
        `rain_scale`). `t_end_s` (optional) snaps the end time to a forcing edge and must equal `t + dt` to round-off.
        Raises `HydraulicStepRejected` / `TransportStepRejected` (retry the SAME state with a smaller dt) or another
        error; `state` and every input are unchanged. Pure."""
        from maple.core.backend import to_float

        from maple.surface.voxels._numerics import summation_error_bound_kg

        from maple_syrup.sediment_event import (
            PHASE_RECONCILIATION_OPERATIONS,
            _PhaseReconcileInputs,
            _reconcile_phase_inputs,
        )

        rt = self._check_state(state)
        if state.status != "wet":
            raise PondEventError("a dry / completed state cannot be stepped or replayed (refused before anything physical)")
        ctl, ctx, xp = self.control, self.context, self.xp
        dt = _positive(dt_s, "dt_s")
        rate = _real(rain_rate_m_per_s, "rain_rate_m_per_s")
        if not (math.isfinite(rate) and rate >= 0.0):
            raise PondEventError("rain_rate_m_per_s must be finite and >= 0")
        t0 = check_state_time(state.hydraulic.t_s)
        t1 = advance_time(t0, dt)
        if t_end_s is not None:
            t1 = _real(t_end_s, "t_end_s")
            if not (t1 > t0 and math.isclose(t1, t0 + dt, rel_tol=1e-12, abs_tol=0.0)):
                raise PondEventError("t_end_s must equal t + dt to round-off")
        dx = rt.topology.dx_m
        rain = self.rain_scale * rate

        # 1. hydraulics
        accepted = step_once(rt.pond, rain, state.hydraulic, dt)
        hyd1 = replace(accepted.state, t_s=t1)
        accepted = replace(accepted, state=hyd1)
        h1 = hyd1.depth_m
        diag = face_speed_diagnostics(rt.pond, accepted, froude_reference=ctl.froude_reference)
        u = diag.cell_speed_m_s

        # 2. wet laws on the post-hydraulic depth and the HOLDINGS at the start of the step
        wet = rt.active & (h1 > 0.0)
        safe_h = xp.where(wet, h1, 1.0)
        s_friction = xp.where(wet, rt.friction * u * u / (8.0 * GRAVITY_M_S2 * safe_h), 0.0)
        layer = state.bed.active_layer.mass_kg
        physics = sediment_physics_step(
            self.sediment, rt.grid, h1, u, rain, self.veg, layer, state.sediment_velocity_m_s, dt,
            shear_slope=s_friction if ctl.shear_slope == "friction" else None,
            stream_power_slope=s_friction if ctl.stream_power_slope == "friction" else None)

        bed = state.bed
        phase_fraction, phase_position = state.phase_fraction, state.phase_position_m
        nc = self.nc
        zero_c = xp.zeros((nc,), dtype=np.float64)
        sp_req = sp_act = sp_dep = sp_exp = zero_c
        sp_pick_cell = sp_dep_cell = xp.zeros_like(layer)
        sp_resid_dep = sp_resid_pick = xp.zeros(())
        ny, nx = self.shape
        sp_xg = sp_xn = xp.zeros((ny, nx + 1, nc), dtype=np.float64)
        sp_yg = sp_yn = xp.zeros((ny + 1, nx, nc), dtype=np.float64)
        n_canon = xp.zeros(())
        sp_tol = xp.zeros((ny, nx, nc), dtype=np.float64)
        did_splash = False
        mobile_old = state.bed.water.mobile_mass_by_cell_class_kg

        # 3. dry splash (rain > 0 and h1 == 0 exactly); native exchange of the ACTUAL pickup
        if self.splash is not None:
            dry = dry_raindrop_pickup_demand(self.sediment, rt.grid, h1, rain, self.veg, layer, dt)
            packet = host_packet(xp, [xp.sum(dry.requested_pickup_kg)])
            if packet[0] > 0.0:
                exchange = apply_splash_exchange(with_water(bed, ctx, h1), ctx, self.splash, rt.splash_grid,
                                                 dry.requested_pickup_kg, implementation=ctl.splash_implementation)
                bed = exchange.bed
                audit = exchange.audit
                sp_req, sp_act = audit.requested_pickup_by_class_kg, audit.actual_pickup_by_class_kg
                sp_dep, sp_exp = audit.native_deposited_by_class_kg, audit.native_export_by_class_kg
                sp_pick_cell = exchange.pickup.actual_removal_by_cell_class_kg
                sp_dep_cell = exchange.deposit.deposition_by_cell_class_kg
                sp_resid_dep = xp.asarray(abs(audit.numerical_residual_scalar_kg))
                sp_resid_pick = xp.asarray(abs(exchange.pickup.numerical_residual_scalar_kg))
                sp_xg, sp_yg = exchange.splash.x_face_gross_kg, exchange.splash.y_face_gross_kg
                sp_xn, sp_yn = exchange.splash.x_face_net_kg, exchange.splash.y_face_net_kg
                sp_tol = exchange.splash.cell_balance_tolerance_kg  # the component's own validated per-cell bound
                phase_fraction, phase_position, n_canon = _canonical_where_empty(
                    xp, phase_fraction, phase_position, bed.water.mobile_mass_by_cell_class_kg)
                did_splash = True

        # 4. wet transaction: native pickup -> pond transport of the ACTUAL pickup -> native deposit / export
        phase = PhaseState(fraction=phase_fraction, position_m=phase_position, dx_m=dx, n_bins=ctl.phase_bins, xp=xp)
        mobile_pre = bed.water.mobile_mass_by_cell_class_kg
        vx, vy = accepted.step.face_volume_m3["x"], accepted.step.face_volume_m3["y"]
        velocity = physics.sediment_velocity_m_s
        has_out = diag.cell_outflow_volume_m3 > 0.0
        v_max = to_float(xp.max(xp.where(has_out[..., None], velocity, 0.0)))
        n_sub = max(1, math.ceil(v_max * dt / (dx * ctl.sediment_courant_max)))
        holder: dict[str, Any] = {"rejections": 0}
        zeros = xp.zeros_like(mobile_pre)

        def transport(pickup):
            n = n_sub
            while True:
                if n > ctl.max_transport_substeps:
                    raise TransportStepRejected(
                        f"sediment Courant number needs {n} substeps (max_transport_substeps = "
                        f"{ctl.max_transport_substeps}) at dt = {dt} s; step rejected (retry with a smaller dt)")
                try:
                    pts = pond_transport_step(rt.network, mobile_pre, phase, pickup.actual_removal_by_cell_class_kg,
                                              velocity, physics.deposition_rate_per_m, physics.settle_mask, vx, vy, dt,
                                              n_substeps=n, courant_max=ctl.sediment_courant_max)
                    break
                except TransportStepRejected:
                    holder["rejections"] += 1
                    n *= 2
            holder["pts"], holder["n_sub"] = pts, n
            return pts.transport.mobile_after_transfer_kg, water_demand_from_transport(pts.transport, zeros)

        bed, pickup, deposit = pickup_transport_deposit(bed, ctx, h1, physics.requested_pickup_kg, transport)
        pts = holder["pts"]
        inputs = _PhaseReconcileInputs(False, pts.transport.mobile_after_transfer_kg, pts.transport.deposition_request_kg,
                                       pts.transport.export_request_kg, pts.mobile_remaining_kg, pts.phase_after)
        reconciled, reconcile = _reconcile_phase_inputs(inputs, bed.water.mobile_mass_by_cell_class_kg, xp)

        # 4b. per-step cell/class mobile balance from ACTUAL native quantities and the FULL face divergence (transport and
        # splash, exports included): dM = P_wet + P_splash - div_wet - div_splash - Dep_native_wet - Dep_native_splash.
        # MAPLE operand/operation-count bound; nothing is adjusted and no residual is used as a mass source.
        tr = pts.transport
        audit_arrays = composite_mobile_balance(
            xp, mobile_old=mobile_old, mobile_new=bed.water.mobile_mass_by_cell_class_kg,
            pick_wet=pickup.actual_removal_by_cell_class_kg, pick_splash=sp_pick_cell,
            dep_wet=deposit.deposition_by_cell_class_kg, dep_splash=sp_dep_cell,
            wet={"x_net": tr.x_face_net_kg, "y_net": tr.y_face_net_kg, "x_gross": tr.x_face_gross_kg,
                 "y_gross": tr.y_face_gross_kg, "mobile_before": mobile_pre, "mobile_after_transfer": tr.mobile_after_transfer_kg,
                 "before": pts.mobile_before_kg, "internal_in": tr.internal_transfer_in_kg,
                 "internal_out": tr.internal_transfer_out_kg, "n_substeps": holder["n_sub"], "n_bins": ctl.phase_bins},
            splash=None if not did_splash else {"x_net": sp_xn, "y_net": sp_yn, "x_gross": sp_xg, "y_gross": sp_yg,
                                                "tolerance": sp_tol})
        bal = host_packet(xp, [xp.max(audit_arrays["abs_residual"]), xp.max(audit_arrays["excess"]),
                               xp.max(audit_arrays["bound"]), xp.max(audit_arrays["components"]["splash"]),
                               xp.max(audit_arrays["components"]["transport"]), xp.max(audit_arrays["components"]["faces"]),
                               xp.max(audit_arrays["components"]["rest"])])
        if bal[1] > 0.0:
            raise PondEventError(
                f"per-step cell/class mobile balance violated: max residual {bal[0]} kg exceeds the composite MAPLE "
                f"bound (largest bound {bal[2]} kg); nothing published")
        balance = {"max_residual_kg": float(bal[0]), "max_bound_kg": float(bal[2]),
                   "component_bounds_kg": {"splash": float(bal[3]), "transport": float(bal[4]), "faces": float(bal[5]),
                                           "bookkeeping": float(bal[6])},
                   "n_operations": int(audit_arrays["n_bookkeeping_operations"])}

        # 5. geometry gate, native commit, refresh only on a net crossing
        hyd_next = replace(hyd1, next_dt_cap_s=state.hydraulic.next_dt_cap_s)
        bed, held, rt2, topology, commit, refreshed, gate = self._commit_and_gate(
            bed, state.held_source_m, rt, state.topology, state.datum, hyd_next, t1, final=False)

        # 6. ledgers, quiet monitor, counters
        metrics = self._metrics(rt2, state.hydraulic.depth_m, hyd_next, bed.water.mobile_mass_by_cell_class_kg,
                                accepted.step.export_m3)
        quiet = self._quiet_advance(state.quiet, t0, t1, dt, metrics)
        water = _water_add(xp, state.water, accepted)
        sl = state.sediment
        dep_unmet = pts.transport.deposition_request_by_class_kg - deposit.deposited_mass_by_class_kg
        sediment = replace(
            sl, wet_requested=sl.wet_requested + pickup.requested_removal_by_class_kg,
            wet_actual=sl.wet_actual + pickup.actual_removal_by_class_kg,
            wet_shortfall=sl.wet_shortfall + pickup.shortfall_by_class_kg,
            splash_requested=sl.splash_requested + sp_req, splash_actual=sl.splash_actual + sp_act,
            deposited_wet=sl.deposited_wet + deposit.deposited_mass_by_class_kg,
            deposited_splash=sl.deposited_splash + sp_dep,
            export_wet=sl.export_wet + deposit.boundary_export_by_class_kg, export_splash=sl.export_splash + sp_exp,
            deposit_unmet=sl.deposit_unmet + dep_unmet,
            pickup_cell=sl.pickup_cell + pickup.actual_removal_by_cell_class_kg + sp_pick_cell,
            deposition_cell=sl.deposition_cell + deposit.deposition_by_cell_class_kg + sp_dep_cell,
            # faces: wet transport AND the actual splash crossings (export crossings included in both)
            x_gross=(sl.x_gross + pts.transport.x_face_gross_kg) + sp_xg,
            y_gross=(sl.y_gross + pts.transport.y_face_gross_kg) + sp_yg,
            x_net=(sl.x_net + pts.transport.x_face_net_kg) + sp_xn,
            y_net=(sl.y_net + pts.transport.y_face_net_kg) + sp_yn,
            numerical_residual_kg=sl.numerical_residual_kg + abs(deposit.numerical_residual_scalar_kg)
            + abs(pickup.numerical_residual_scalar_kg) + sp_resid_dep + sp_resid_pick,
            phase_residual_max_kg=xp.maximum(sl.phase_residual_max_kg, reconcile["residual_max_kg"]),
            n_phase_canonicalized=sl.n_phase_canonicalized + reconcile["n_canonicalized"] + n_canon,
            n_phase_remnants=sl.n_phase_remnants + reconcile["n_rounding_remnants"],
            n_native_calls=sl.n_native_calls + 2 + (2 if did_splash else 0))
        # 7. resident-window storage (LAST operation of the candidate: every geometry / transport / balance / metric check
        # above has passed). 'step' publishes this step's journal as a new deep-store version; 'boundary' CARRIES the
        # valid unpublished journal in the returned state (published later by evolve / publish_storage / finalize).
        # Ordering rule: ALL host work (counters, record, state construction) is done on the UNPUBLISHED bed first; the
        # deep-store version is written only after it, as the very last operation, so no late host error can leave a
        # speculative version behind. `published` is therefore the decision, executed at the end.
        carried = pending_journal_records(bed) if self.windowed else 0
        published = bool(carried and self.control.deep_publication == "step")
        stats = diag.stats
        c = state.counters
        log = c.commit_log
        if commit.get("committed") and len(log) < ctl.max_commit_log:
            log = log + (commit,)
        reason = commit.get("reason")
        counters = replace(
            c, n_accepted_steps=c.n_accepted_steps + 1, n_commits=c.n_commits + int(bool(commit.get("committed"))),
            n_maple_trigger_commits=c.n_maple_trigger_commits + int(reason == "maple_trigger"),
            n_net_elevation_commits=c.n_net_elevation_commits + int(reason == "net_elevation"),
            n_final_commits=c.n_final_commits + int(reason == "final_forced"),
            n_refreshes=c.n_refreshes + int(refreshed), n_splash_exchanges=c.n_splash_exchanges + int(did_splash),
            max_gate_displacement_m=max(c.max_gate_displacement_m, gate),
            max_face_froude=max(c.max_face_froude, stats["max_face_froude"]),
            n_faces_above_froude_reference=c.n_faces_above_froude_reference + stats["n_faces_above_froude_reference"],
            max_transport_substeps_used=max(c.max_transport_substeps_used, int(holder["n_sub"])),
            n_storage_publications=c.n_storage_publications + int(published),
            max_carried_records=max(c.max_carried_records, carried), commit_log=log)
        new_state = replace(
            state, hydraulic=hyd_next, bed=bed, sediment_velocity_m_s=physics.sediment_velocity_m_s,
            phase_fraction=reconciled.fraction, phase_position_m=reconciled.position_m, topology=topology,
            held_source_m=held, water=water, sediment=sediment, counters=counters, quiet=quiet)
        record = PondStepRecord(
            t0_s=t0, t1_s=t1, dt_s=dt, n_substeps=holder["n_sub"], n_transport_rejections=holder["rejections"],
            froude=dict(stats), metrics=metrics, commit=commit if commit.get("committed") else None, refreshed=refreshed,
            splash_exchange=did_splash,
            by_class={"wet_requested": pickup.requested_removal_by_class_kg, "wet_actual": pickup.actual_removal_by_class_kg,
                      "splash_requested": sp_req, "splash_actual": sp_act, "deposited": deposit.deposited_mass_by_class_kg,
                      "export": deposit.boundary_export_by_class_kg, "splash_deposited": sp_dep, "splash_export": sp_exp},
            phase={"residual_max_kg": reconcile["residual_max_kg"], "n_canonicalized": reconcile["n_canonicalized"],
                   "n_rounding_remnants": reconcile["n_rounding_remnants"]},
            quiet=quiet, balance=balance,
            numerical_residuals={"pickup": abs(pickup.numerical_residual_scalar_kg),
                                 "deposit": abs(deposit.numerical_residual_scalar_kg),
                                 "splash_pickup": sp_resid_pick, "splash_deposit": sp_resid_dep},
            storage={"pending_records": 0 if published else carried, "published": published} if self.windowed else {})
        if published:  # the last operation of the candidate
            new_state = replace(new_state, bed=publish_bed_storage(bed, ctx))
        self._blessed.add(new_state)  # built here from validated pieces
        return PondCoupledStep(state=new_state, record=record)

    # -- bounded evolve ------------------------------------------------------------------------------------
    def evolve(self, state: PondEventState, end_s: float, *, stop_when_quiet: bool = False,
               max_steps: int | None = None, publish_at_return: bool = True) -> PondEvolveResult:
        """Advance to `end_s` by accepted steps of at most `control.time.max_dt_s`. The rain schedule is split EXACTLY at
        every discontinuity (`RainfallSchedule.next_edge_s` / `rate_after_m_per_s`; the rate is never averaged); a
        step that ends on an edge or on `end_s` snaps its time to it. A native `HydraulicStepRejected` or
        `TransportStepRejected` halves dt from the UNCHANGED state, bounded by `max_retries` per step, `min_dt_s` and
        `max_steps` (default `control.time.max_steps`); a bound STOPS the call (`stop_reason`, `reached_target=False`) and
        returns the last accepted state: never continued, reset or finished silently. Errors propagate."""
        ctl = self.control.time
        self._check_state(state)
        if state.status != "wet":
            raise PondEventError("a dry / completed state cannot be evolved or replayed (refused before anything physical)")
        t = check_state_time(state.t_s)
        end = _real(end_s, "end_s")
        if not (math.isfinite(end) and end > t):
            raise PondEventError(f"end_s must be finite and > the state time {t}, got {end_s!r}")
        limit = ctl.max_steps if max_steps is None else max_steps
        resumed = check_dt_cap(state.hydraulic.next_dt_cap_s)
        cap = ctl.max_dt_s if resumed is None else min(max(resumed, ctl.min_dt_s), ctl.max_dt_s)
        t_start = t
        n_acc = n_rej = n_flush = n_high_water = 0
        rejections: list[dict] = []
        stop = None
        dt_min, dt_max = math.inf, 0.0
        last = None
        while t < end and stop is None:
            if stop_when_quiet and state.quiet.ready:
                stop = "quiet"
                break
            if n_acc >= limit:
                stop = "max_steps"
                break
            piece_end = min(end, self.schedule.next_edge_s(t))
            rate = self.schedule.rate_after_m_per_s(t)
            remaining = piece_end - t
            dt = min(remaining, cap)
            snap = dt == remaining
            retries = 0
            result = None
            flushed = False  # at most ONE carried-journal flush per accepted step
            while result is None:
                if not t + dt > t:
                    stop = "time_resolution"
                    break
                flush_now = False
                try:
                    result = self.step(state, rate, dt, t_end_s=piece_end if snap else None)
                except (HydraulicStepRejected, TransportStepRejected) as exc:
                    # A genuine physical rejection (also of the redo that follows a storage flush: the capacity refusal can
                    # precede the transport) enters the ordinary bounded adaptation; the flush itself changed no dt,
                    # retry count or rejection record, and it is never repeated for this accepted step.
                    retries += 1
                    n_rej += 1
                    if len(rejections) < _REJECTIONS_KEPT:
                        rejections.append({"t_s": t, "dt_tried_s": dt, "reason": str(exc)})
                    if retries > ctl.max_retries:
                        stop = "max_retries"
                        break
                    if dt * 0.5 < ctl.min_dt_s:
                        stop = "min_dt"
                        break
                    dt, snap = dt * 0.5, False
                except ValueError as exc:
                    # Bounded flush / retry (windowed beds): ONLY the native append-cap refusal (source-qualified typed
                    # `JournalCapacityError` with consistent counts, or the pinned M94 message) raised while the PREVIOUS
                    # accepted state carries a journal, and only before any flush of this accepted step. Anything else
                    # (an unrelated ValueError, a step too large for a CLEAN journal, malformed counts) propagates. Only
                    # a scalar flag is set here: the flush and the redo run after the handler exits.
                    if (flushed or not self.windowed
                            or not is_carried_append_cap_refusal(exc, self.context, pending_journal_records(state.bed))):
                        raise
                    flush_now = True
                if flush_now:
                    # Publish the PREVIOUS validated accepted state (explicitly a prior-state flush, not a candidate
                    # publication), then redo the SAME attempt: same state clock, rate, dt and snap; rejection counters,
                    # retries and the adaptive cap are untouched.
                    state = self._publish(state, "flush")
                    n_flush += 1
                    flushed = True
            if result is None:
                break
            if retries:
                cap = dt
            elif dt >= cap:
                cap = min(ctl.max_dt_s, 2.0 * cap)
            t = result.state.t_s
            hydraulic = replace(result.state.hydraulic, next_dt_cap_s=cap)
            state = replace(result.state, hydraulic=hydraulic)
            self._blessed.add(state)  # only the numerical cap of a blessed state changed
            n_acc += 1
            dt_min, dt_max = min(dt_min, dt), max(dt_max, dt)
            last = result.record
            if self.windowed and self.control.deep_publication == "boundary":
                # high-water boundary: the accepted state carries a valid journal; publish it (storage only) when it
                # reached the mark. Hydraulic geometry, ledgers and clocks are untouched by the publication.
                if pending_journal_records(state.bed) >= self.control.deep_publication_high_water_records:
                    state = self._publish(state, "high_water")
                    n_high_water += 1
        counters = replace(state.counters, n_rejected_attempts=state.counters.n_rejected_attempts + n_rej)
        state = replace(state, counters=counters)
        self._blessed.add(state)  # only a host counter changed
        storage: dict = {}
        if self.windowed:
            n_return = 0
            if publish_at_return and pending_journal_records(state.bed):
                state = self._publish(state, "return")  # return boundary: the returned state is restartable
                n_return = 1
            storage = {"n_high_water_publications": n_high_water, "n_flush_publications": n_flush,
                       "n_return_publications": n_return, "pending_records": pending_journal_records(state.bed)}
        return PondEvolveResult(
            state=state, requested_end_s=end, elapsed_s=float(t - t_start), reached_target=stop is None and t >= end,
            stop_reason=stop, quiet_ready=state.quiet.ready, n_accepted_steps=n_acc, n_rejected_attempts=n_rej,
            rejections=tuple(rejections), min_accepted_dt_s=dt_min if n_acc else 0.0, max_accepted_dt_s=dt_max,
            last_record=last, storage=storage)

    # -- resident-window storage publication (no physics) -----------------------------------------------------
    def _publish(self, state: PondEventState, reason: str) -> PondEventState:
        """Publish the CARRIED journal of a VALIDATED accepted `state` as a new deep-store version (MAPLE's
        `publish_pending_column` through `publish_bed_storage`: the context's encoding and cap options, old versions
        retained). Storage only: hydraulic state, topology and held source, ledgers, quiet monitor, clocks and every
        counter are unchanged except `n_storage_publications` (and `n_journal_flushes` when `reason == "flush"`, the
        publication of a PRIOR accepted state that precedes the single retry of a capacity-refused step)."""
        c = state.counters
        # host bookkeeping FIRST (a malformed publicly supplied counter fails here, before any store write)
        counters = replace(c, n_storage_publications=c.n_storage_publications + 1,
                           n_journal_flushes=c.n_journal_flushes + int(reason == "flush"))
        published = replace(state, bed=publish_bed_storage(state.bed, self.context), counters=counters)
        self._blessed.add(published)  # only the storage binding of a blessed state changed
        return published

    def publish_storage(self, state: PondEventState) -> PondEventState:
        """Explicit NO-PHYSICS publication of a state's carried journal (the checkpoint / handoff boundary). Pure; returns
        the SAME object when there is nothing to publish (dense bed, or an already published windowed bed). The returned
        state is bitwise the input except the bed's deep-store binding and the storage counters; a failure leaves the
        input and every previously referenced store version unchanged."""
        self._check_state(state)
        if not self.windowed or not pending_journal_records(state.bed):
            return state
        return self._publish(state, "explicit")

    # -- audit ------------------------------------------------------------------------------------------------
    def _mobile_identity_bound(self, state: PondEventState, mobile: np.ndarray, vec: dict, picked: np.ndarray,
                               deposited: np.ndarray, export: np.ndarray, n_calls: int) -> tuple[np.ndarray, dict]:
        """Per-class bound (host, kg) of the CUMULATIVE class mobile identity
        `mobile - (mobile0 + picked - deposited - export) = 0`, from the arithmetic that actually produced its operands.
        Diagnostic only; nothing here changes a state. Terms (all `(nc,)`, MAPLE policies, genuine operands):

        1. CELL REDUCTIONS: the class vectors are sums over `ny*nx` cells of non-negative terms. Some are explicit pairwise
           MAPLE helpers, but the event's own `mobile0` / `mobile` use a raw `xp.sum(axis=(0, 1))` on a non-contiguous
           leading-axis layout, whose summation order neither NumPy nor CuPy promises to be a tree. The rigorous bound valid
           for ANY order is the sequential one, `sequential_bound_scale_factor(ny*nx) * sum|x|` (an upper bound for the
           native pairwise sums as well); over the steps it is the factor times the cumulative value, for `mobile0`,
           `mobile` (once each) and `picked`, `deposited`, `export`. (`pairwise_bound_scale_factor` is used only in the
           splash / transport component policies below, which are built on the explicit pairwise helper.)
        2. LEDGER ACCUMULATION: the class totals are added sequentially over the `n_steps` accepted steps (+1 terminal add
           +1 for the first addend): `sequential_bound_scale_factor(n_steps + 2) * (picked + deposited + export)`.
        3. NATIVE BOOKKEEPING: the existing `reservoir_bound_kg(n_native_calls, ...)` (cell-wise pickup / deposit / export
           arithmetic of the native calls) at the largest class operand.
        4. SPLASH COMPONENT: the splash class budget identity `pickup - deposition - export` is validated with
           `summation_error_bound_kg(_CLOSURE_OPS, max cell scale) + 2 pw (before + dep + exp) + 8 eps before`, where the cell
           scale is `P + Dep + the four gross face crossings` (the long rays' large cancelling operands). Summed over the
           steps, `sum_k max_i s_ki <= sum_i S_i` with `S_i` the CUMULATIVE per-cell scale from the ledger arrays
           (`pickup_cell + deposition_cell + the four cumulative gross faces`, a superset that also holds the wet terms), so
           the first term is `summation_error_bound_kg(_CLOSURE_OPS, sum_i S_i)`.
        5. WET TRANSPORT COMPONENT: the transport class budget `summation_error_bound_kg(operation_count(n_sub, B),
           max(before, after, in, out)) + 2 pw (before + after)` per step with `n_sub <= counters.max_transport_substeps_used`
           (the largest count ACTUALLY executed by an accepted step, a host integer; at least 1), `before, after <= mobile0 + picked` (the pool cannot exceed what was ever injected),
           `in, out <= ` the class gross face total. Over `K` steps the scale sum is at most
           `K * 2 (mobile0 + picked) + 2 G` with `G` the cumulative gross face total.
        No multiplier, floor, mass resolution or percentage; with zero genuine operands every term is exactly zero. The
        global class bed closure is a separate, independent check (it is not substituted for this one)."""
        from maple.core.backend.reductions import pairwise_bound_scale_factor, sequential_bound_scale_factor
        from maple.surface.voxels._numerics import summation_error_bound_kg

        from maple_syrup.splash import _CLOSURE_OPS

        sl = state.sediment
        ny, nx = self.shape
        pw = pairwise_bound_scale_factor(ny * nx)  # ONLY for the splash/transport policies (explicit pairwise helper)
        eps = float(np.finfo(np.float64).eps)
        k_steps = max(int(state.water.n_steps), 1)
        mobile0 = np.asarray(sl.mobile0, dtype=np.float64)
        reductions = cell_reduction_bound(ny * nx, mobile0, mobile, picked, deposited, export)
        accumulation = sequential_bound_scale_factor(k_steps + 2) * (picked + deposited + export)
        bookkeeping = reservoir_bound_kg(max(n_calls, 1), mobile0, mobile, picked, deposited, export)
        xg, yg = _host(sl.x_gross), _host(sl.y_gross)
        gross = xg.sum(axis=(0, 1)) + yg.sum(axis=(0, 1))
        cell_scale = (_host(sl.pickup_cell) + _host(sl.deposition_cell) + xg[:, :-1] + xg[:, 1:] + yg[:-1] + yg[1:])
        scale_sum = cell_scale.sum(axis=(0, 1))
        splash = np.zeros_like(picked)
        if sl.n_native_calls and np.any(vec["splash_actual"] > 0.0):
            splash = (summation_error_bound_kg(_CLOSURE_OPS, scale_sum)
                      + 2.0 * pw * (vec["splash_actual"] + vec["deposited_splash"] + vec["export_splash"])
                      + 8.0 * eps * vec["splash_actual"])
        injected = mobile0 + picked
        # the realised maximum is an operation-count upper bound for EVERY accepted step (not the configured cap)
        n_tr = operation_count(max(int(state.counters.max_transport_substeps_used), 1), self.control.phase_bins)
        transport = (summation_error_bound_kg(n_tr, k_steps * 2.0 * injected + 2.0 * gross)
                     + 2.0 * pw * k_steps * 2.0 * injected)
        total = reductions + accumulation + bookkeeping + splash + transport
        terms = {"cell_reductions": reductions.tolist(), "ledger_accumulation": accumulation.tolist(),
                 "native_bookkeeping": float(bookkeeping), "splash_component": splash.tolist(),
                 "transport_component": transport.tolist()}
        return total, terms

    def audit(self, state: PondEventState) -> dict:
        """Host audit of the accepted-step ledgers (bounded reads): water closure (the existing `PondLedger` closure with the
        dry-reset removal), global class sediment closure against MAPLE's own inventory with the MAPLE bound, the
        mobile-pool identity, gross / net face totals, actual pickup / deposit / export (splash included), geometry
        generation, commits and the current gate displacement."""
        rt = self._check_state(state)
        water = state.water
        closure = _pond_ledger(rt.pond, water).closure(state.hydraulic)
        area = rt.topology.dx_m ** 2
        removed = 0.0
        surface_cell, soil_cell = closure.surface_cell_residual_m3, closure.soil_cell_residual_m3
        if water.reset_depth is not None:
            rd, rs = _host(water.reset_depth), _host(water.reset_soil)
            removed = float(area * (rd.sum() + rs.sum()))
            surface_cell, soil_cell = surface_cell + area * rd, soil_cell + area * rs
        global_residual = closure.global_residual_m3 + removed
        max_cell = float(max(np.abs(surface_cell).max(), np.abs(soil_cell).max()))
        water_closes = (abs(global_residual) <= closure.global_bound_m3 and max_cell <= closure.cell_bound_m3
                        and closure.inactive_unchanged)
        sl = state.sediment
        inv = _host(bed_inventory(state.bed))
        mobile = _host(state.bed.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1)))
        vec = {k: _host(getattr(sl, k)) for k in (
            "wet_requested", "wet_actual", "wet_shortfall", "splash_requested", "splash_actual", "deposited_wet",
            "deposited_splash", "export_wet", "export_splash", "deposited_terminal", "deposit_unmet")}
        total0, total1 = sl.inventory0 + sl.mobile0, inv + mobile
        export = vec["export_wet"] + vec["export_splash"]
        n_calls = max(sl.n_native_calls, 1)
        closure_resid = total0 - (total1 + export)
        closure_bound = reservoir_bound_kg(n_calls, total0, total1, export)
        picked = vec["wet_actual"] + vec["splash_actual"]
        deposited = vec["deposited_wet"] + vec["deposited_splash"] + vec["deposited_terminal"]
        mobile_resid = mobile - (sl.mobile0 + picked - deposited - export)
        mobile_bound, mobile_terms = self._mobile_identity_bound(state, mobile, vec, picked, deposited, export, n_calls)
        sediment_closes = bool(np.all(np.abs(closure_resid) <= closure_bound) and np.all(np.abs(mobile_resid) <= mobile_bound))
        gate = gate_displacement_m(state.bed, self.context, state.held_source_m, rt.active)
        report = {
            "schema": SCHEMA, "t_s": state.t_s, "status": state.status,
            "water": {"closes": bool(water_closes), "global_residual_m3": global_residual,
                      "global_bound_m3": closure.global_bound_m3, "max_cell_residual_m3": max_cell,
                      "cell_bound_m3": closure.cell_bound_m3, "rain_m3": closure.rain_m3, "intake_m3": closure.intake_m3,
                      "saturation_return_m3": closure.saturation_return_m3, "drainage_m3": closure.drainage_m3,
                      "export_m3": closure.export_m3, "surface_change_m3": closure.surface_change_m3,
                      "soil_change_m3": closure.soil_change_m3, "reset_removed_m3": removed,
                      "inactive_unchanged": closure.inactive_unchanged, "n_steps": closure.n_steps,
                      "face_gross_m3": {a: float((closure.face_pos_m3[a] + closure.face_neg_m3[a]).sum()) for a in "xy"},
                      "face_net_m3": {a: float(closure.face_net_m3[a].sum()) for a in "xy"}},
            "sediment": {"closes": sediment_closes, "class_closure_residual_kg": closure_resid.tolist(),
                         "class_closure_bound_kg": float(closure_bound), "mobile_identity_residual_kg": mobile_resid.tolist(),
                         "mobile_identity_bound_kg": mobile_bound.tolist(),
                         "mobile_identity_bound_terms_kg": mobile_terms,
                         "inventory_kg": inv.tolist(), "mobile_kg": mobile.tolist(),
                         **{k + "_kg": v.tolist() for k, v in vec.items()},
                         "face_gross_kg": {"x": _host(sl.x_gross).sum(axis=(0, 1)).tolist(),
                                           "y": _host(sl.y_gross).sum(axis=(0, 1)).tolist()},
                         "face_net_kg": {"x": _host(sl.x_net).sum(axis=(0, 1)).tolist(),
                                         "y": _host(sl.y_net).sum(axis=(0, 1)).tolist()},
                         "pickup_by_cell_total_kg": _host(sl.pickup_cell).sum(axis=(0, 1)).tolist(),
                         "deposition_by_cell_total_kg": _host(sl.deposition_cell).sum(axis=(0, 1)).tolist(),
                         "numerical_residual_scalar_abs_kg": float(_host(sl.numerical_residual_kg)),
                         "phase": {"residual_max_kg": float(_host(sl.phase_residual_max_kg)),
                                   "n_canonicalized": int(_host(sl.n_phase_canonicalized)),
                                   "n_rounding_remnants": int(_host(sl.n_phase_remnants))},
                         "n_native_calls": sl.n_native_calls},
            "geometry": {"generation": state.topology.generation, "source_sha256": state.topology.source_sha256,
                         "topology_sha256": state.topology.topology_sha256, "gate_displacement_m": gate,
                         "threshold_m": reroute_threshold_m(self.context), "comparison": ">=",
                         "n_closed_components": state.topology.n_closed_components},
            "counters": {k: getattr(state.counters, k) for k in PondCounters.__dataclass_fields__ if k != "commit_log"},
            "commit_log": list(state.counters.commit_log), "quiet": state.quiet,
        }
        if self.windowed:  # key present only for a resident-window bed (a dense audit is unchanged)
            report["storage"] = {
                "bed_storage": "windowed", "deep_publication": self.control.deep_publication,
                "high_water_records": self.control.deep_publication_high_water_records,
                "pending_journal_records": pending_journal_records(state.bed),
                "journal_cap_records": resolved_journal_cap(self.context),
                "deep_store_version_id": int(state.bed.voxel_column.deep_store.version_id),
                "n_storage_publications": state.counters.n_storage_publications,
                "n_journal_flushes": state.counters.n_journal_flushes,
                "max_carried_records": state.counters.max_carried_records}
        return report

    # -- completion -----------------------------------------------------------------------------------------------
    def finalize(self, state: PondEventState) -> PondDryHandoff:
        """Explicit safe terminal settlement and dry reset (module docstring). Pure: every failure raises
        `PondEventIncomplete` / `PondEventError` with the wet input unchanged."""
        from maple.core.backend import to_bool
        from maple.surface.topographic_commit.commit import ledger_is_empty
        from maple.water import WaterProcessDemand

        rt = self._check_state(state)
        xp, ctx = self.xp, self.context
        if state.status != "wet":
            raise PondEventIncomplete("only a wet state can be finalised")
        qualifies, before = self.quiet_metrics(state)
        q = state.quiet
        if not (q.ready and q.last_t_s == state.t_s and q.since_s is not None
                and state.t_s - q.since_s >= self.policy.hold_s and qualifies):
            raise PondEventIncomplete("terminal reset requires rain off, no genuine flow / overcrest and a completed "
                                      f"continuous quiet interval (metrics: {before})")
        pre = self.audit(state)
        if not (pre["water"]["closes"] and pre["sediment"]["closes"]):
            raise PondEventError(f"pre-terminal budgets do not close: {pre['water']['closes']}, {pre['sediment']['closes']}")
        bed = state.bed
        pool = bed.water.mobile_mass_by_cell_class_kg
        deposited = xp.zeros_like(pool)
        terminal_class = xp.zeros((self.nc,), dtype=np.float64)
        n_calls = 0
        n_flush = n_publish = 0
        if to_bool(xp.any(pool)):
            flush_now = False
            try:
                bed, transfer = apply_bed_demand(bed, ctx, WaterProcessDemand(xp.zeros_like(pool), pool))
            except ValueError as exc:
                # the same bounded rule as `evolve`: ONLY a recognised append-cap refusal against a CARRIED journal, once
                if not self.windowed or not is_carried_append_cap_refusal(exc, ctx, pending_journal_records(bed)):
                    raise
                flush_now = True
            if flush_now:  # outside the handler: publish the prior state's journal, redo the same terminal deposition
                bed = publish_bed_storage(bed, ctx)
                n_flush = n_publish = 1
                bed, transfer = apply_bed_demand(bed, ctx, WaterProcessDemand(xp.zeros_like(pool), pool))
            if to_bool(xp.any(bed.water.mobile_mass_by_cell_class_kg != 0)):
                raise PondEventError("MAPLE refused terminal mobile deposition; no inventory is erased")
            deposited, terminal_class, n_calls = transfer.deposition_by_cell_class_kg, transfer.deposited_mass_by_class_kg, 1
        held, topology, rt2, refreshed, commit = state.held_source_m, state.topology, rt, False, None
        if not bed_committed_at(bed, ctx, state.t_s):
            bed, held, rt2, topology, commit, refreshed, _ = self._commit_and_gate(
                bed, held, rt, topology, state.datum, state.hydraulic, state.t_s, final=True)
        if not ledger_is_empty(bed.ledger):
            raise PondEventError("terminal MAPLE ledger is not fully empty")
        sl = state.sediment
        sediment = replace(sl, deposited_terminal=sl.deposited_terminal + terminal_class,
                           deposition_cell=sl.deposition_cell + deposited, n_native_calls=sl.n_native_calls + n_calls)
        c = state.counters
        counters = replace(c, n_commits=c.n_commits + int(commit is not None and commit["committed"]),
                           n_final_commits=c.n_final_commits + int(commit is not None and commit.get("reason") == "final_forced"),
                           n_net_elevation_commits=c.n_net_elevation_commits + int(commit is not None and commit.get("reason") == "net_elevation"),
                           n_refreshes=c.n_refreshes + int(refreshed),
                           n_storage_publications=c.n_storage_publications + n_publish,
                           n_journal_flushes=c.n_journal_flushes + n_flush,
                           max_carried_records=max(c.max_carried_records, pending_journal_records(state.bed),
                                                   pending_journal_records(bed)),
                           commit_log=c.commit_log + ((commit,) if commit and commit["committed"]
                                                      and len(c.commit_log) < self.control.max_commit_log else ()))
        phase_fraction, phase_position, _ = _canonical_where_empty(xp, state.phase_fraction, state.phase_position_m,
                                                                   bed.water.mobile_mass_by_cell_class_kg)
        settled = replace(state, bed=bed, held_source_m=held, topology=topology, sediment=sediment, counters=counters,
                          phase_fraction=phase_fraction, phase_position_m=phase_position)
        after_ok, after = self.quiet_metrics(settled)
        if not after_ok:
            raise PondEventIncomplete(f"terminal terrain change invalidated completion; input remains wet ({after})")
        post = self.audit(settled)
        if not (post["water"]["closes"] and post["sediment"]["closes"]):
            raise PondEventError("budgets fail after the terminal settlement")
        # explicit external dry reset: depth and soil removed and recorded; nothing mobile remains
        h, soil = settled.hydraulic.depth_m, settled.hydraulic.soil_water_m
        area = rt2.topology.dx_m ** 2
        removed = host_packet(xp, [h.sum() * area, soil.sum() * area])
        dry_hydraulic = replace(settled.hydraulic, depth_m=xp.zeros_like(h), soil_water_m=xp.zeros_like(soil),
                                qx_m2_s=xp.zeros_like(settled.hydraulic.qx_m2_s),
                                qy_m2_s=xp.zeros_like(settled.hydraulic.qy_m2_s))
        dry_bed = with_water(bed, ctx, xp.zeros_like(h))
        # All host work and audits run on the UNPUBLISHED dry bed; the final publication is the last operation, so a failed
        # check never leaves a speculative deep version. (The final audit's "storage" block is completed after the
        # successful publication below, so the returned receipt is truthful.)
        publishing = bool(pending_journal_records(dry_bed))
        water =replace(settled.water, reset_depth=xp.array(h, copy=True), reset_soil=xp.array(soil, copy=True))
        dry_counters = replace(settled.counters,
                               n_storage_publications=settled.counters.n_storage_publications + int(publishing))
        dry = replace(settled, hydraulic=dry_hydraulic, bed=dry_bed, water=water, counters=dry_counters,
                      sediment_velocity_m_s=xp.zeros_like(settled.sediment_velocity_m_s), quiet=PondQuiet(), status="dry")
        final_audit = self.audit(dry)
        if not (final_audit["water"]["closes"] and final_audit["sediment"]["closes"]):
            raise PondEventError("combined storm and dry-reset budgets fail")
        accounting = {
            "policy": "explicit external dry-again removal of surface AND soil water; no evapotranspiration simulated",
            "surface_removed_m3": float(removed[0]), "soil_removed_m3": float(removed[1]),
            "completion_before_terminal": before, "completion_after_terminal": after,
            "terminal_deposition_by_class_kg": _host(terminal_class).tolist(),
            "pre_terminal_audit": pre, "final_audit": final_audit,
            "final_commit": commit, "geometry_refreshed_at_final": refreshed}
        if publishing:  # a completed handoff never carries an unpersisted deep journal; last operation
            dry = replace(dry, bed=publish_bed_storage(dry.bed, ctx))
            if pending_journal_records(dry.bed):
                raise PondEventError("the completed handoff still carries an unpublished deep journal")
            self._blessed.add(dry)  # only the storage binding of an audited state changed
            # Truthful receipt WITHOUT moving the expensive / refusable checks after the write: the audit above was taken
            # on the unpublished bed; the publication changes exactly two audited storage facts (everything else in the
            # block, the counters included, was already computed for the published state), so they are updated only now,
            # after the publication succeeded. `final_audit["storage"] == audit(dry)["storage"]`.
            final_audit["storage"]["pending_journal_records"] = pending_journal_records(dry.bed)
            final_audit["storage"]["deep_store_version_id"] = int(dry.bed.voxel_column.deep_store.version_id)
        return PondDryHandoff(dry_state=dry, pre_reset_surface_m=h, pre_reset_soil_m=soil,
                              terminal_deposition_kg=deposited, accounting=accounting)

    def complete(self, state: PondEventState, max_end_s: float) -> PondDryHandoff:
        """Evolve until the quiet monitor is ready and finalise, else `PondEventIncomplete` (nothing reset)."""
        result = self.evolve(state, max_end_s, stop_when_quiet=True)
        if result.quiet_ready:
            return self.finalize(result.state)
        raise PondEventIncomplete(
            f"no quiet completion by t = {result.state.t_s} s (stop_reason = {result.stop_reason!r}); "
            "water and sediment were not reset")


def _control_identity(control: PondEventControl, context: BedContext) -> tuple:
    """Digest parts of the resident-window options: EMPTY for every dense default (so every dense digest is unchanged)."""
    if control.bed_storage == "dense":
        return ()
    return (("windowed", control.deep_publication, control.deep_publication_high_water_records,
             context.window_max_journal_records, deep_storage_record(context)),)


def require_supported_bed_context(context: BedContext, control: PondEventControl | None = None) -> None:
    """Context-only part of `pond_bed.require_supported_bed` (constructor time, no bed yet), plus the windowed policy
    checks that need no state: all raised BEFORE any event array exists.

    Dense (the default control): windowed journal / deep-storage options are refused. Windowed: the journal cap and the
    deep-chunk encoding options are validated against the bound MAPLE source; for `deep_publication == "boundary"` the
    high-water mark must lie below the ACTUAL journal cap (`resolved_journal_cap`) and the bound MAPLE package must be a
    source whose append-cap refusal the recognizer knows (M94 pinned message or the accepted typed
    `JournalCapacityError`, checked structurally)."""
    if bool(getattr(context.avalanching_spec, "enabled", False)):
        raise PondBedError("avalanching must be explicitly disabled for the pond event")
    if context.depth_update_rule != "constant_depth":
        raise PondBedError("the pond event requires constant_depth")
    if context.record_exchange_history is not False:
        raise PondBedError("record_exchange_history is not supported by the pond event")
    windowed = control is not None and control.bed_storage == "windowed"
    if not windowed:
        if getattr(context, "window_max_journal_records", None) is not None:
            raise PondBedError("windowed journal options are not supported (dense bed only)")
        try:
            validate_deep_storage(context)
        except BedIntegrationError as exc:
            raise PondBedError(str(exc)) from None
        if context.deep_chunk_encoding != "full" or context.deep_overlay_max_bytes is not None:
            raise PondBedError("windowed deep-storage options are not supported (dense bed only)")
        return
    cap = context.window_max_journal_records
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise PondBedError(f"window_max_journal_records must be None or a positive int, got {cap!r}")
    try:
        validate_deep_storage(context)
        if control.deep_publication == "boundary":
            journal_cap = resolved_journal_cap(context)
            if not control.deep_publication_high_water_records < journal_cap:
                raise PondBedError(f"deep_publication_high_water_records {control.deep_publication_high_water_records} "
                                   f"must be below the actual MAPLE journal cap {journal_cap}")
            recognizer = journal_recognizer_source()
            if recognizer is None:
                raise PondBedError("deep_publication 'boundary' recognizes only the pinned M94 append-cap refusal or the "
                                   "accepted typed JournalCapacityError source; the bound MAPLE package is neither")
            if recognizer == "typed_journal_capacity_error":
                require_typed_journal_capacity_error()
    except BedIntegrationError as exc:
        if isinstance(exc, PondBedError):
            raise
        raise PondBedError(str(exc)) from None


# --- composite mobile balance (pure error propagation) -------------------------------------------------
DIVERGENCE_FLOPS = 3  # (x[c+1] - x[c]) + (y[r+1] - y[r]): three floating operations per stage
COMBINATION_FLOPS = 7  # P_w + P_s, div_w + div_s, their difference, Dep_w + Dep_s, the difference; Mnew - Mold; residual


def _divergence(xn: Any, yn: Any) -> Any:
    return (xn[:, 1:] - xn[:, :-1]) + (yn[1:] - yn[:-1])


def _face_scale(xp: Any, stage: dict) -> Any:
    """Largest INDIVIDUAL face operand (gross crossings and nets of the four faces) of each cell and class: these are the
    operands of the divergence subtraction, however much they cancel in the net."""
    parts = []
    for key in ("x_gross", "x_net"):
        a = xp.abs(stage[key])
        parts.append(xp.maximum(a[:, :-1], a[:, 1:]))
    for key in ("y_gross", "y_net"):
        a = xp.abs(stage[key])
        parts.append(xp.maximum(a[:-1], a[1:]))
    return xp.maximum(xp.maximum(parts[0], parts[1]), xp.maximum(parts[2], parts[3]))


def composite_mobile_balance(xp: Any, *, mobile_old: Any, mobile_new: Any, pick_wet: Any, pick_splash: Any,
                             dep_wet: Any, dep_splash: Any, wet: dict | None = None, splash: dict | None = None,
                             native_operations: int | None = None) -> dict:
    """Per cell/class mobile balance of one coupled step with a COMPOSITE MAPLE error bound. Pure; arrays in `xp`; no state.

    Identity audited: `dM = (P_w + P_s) - (div_w + div_s) - (Dep_w + Dep_s)` with `dM = mobile_new - mobile_old` (actual
    native quantities, `div` the full face-net divergence including export crossings). `residual = dM - expected`.
    `wet` (None = no wet transport) holds the transport faces (`x_net, y_net, x_gross, y_gross`), its `mobile_before`,
    `before` (= M + P), `mobile_after_transfer`, `internal_in/out`, `n_substeps`, `n_bins`. `splash` (None = no splash)
    holds the splash faces and its validated per-cell `tolerance` (`SplashResult.cell_balance_tolerance_kg`).

    Bound (derivation). The audited residual is the sum of independent identity defects, each bounded by its own
    component policy, plus the rounding of the audit's own arithmetic:
      1. splash identity `Dep_req - P - netinflow`: the component's own validated `cell_balance_tolerance_kg` (its operands,
         the long ray fluxes, are inside that bound, so cancelling large face values are accounted there);
      2. transport identity `T - (M + P) = In - Out`: recomputed exactly as the operator checks it,
         `summation_error_bound_kg(operation_count(n_sub, B), max(M + P, T, In, Out))`;
      3. FACE FORMATION: the audit forms `div` from the individual face nets, and the transport accumulates those faces
         separately from its internal In/Out totals. Every INDIVIDUAL face operand (gross and net, before cancelling) is
         the scale; operations: 3 per stage for the divergence, plus for the wet stage the actual accumulation adds
         (`n_sub` face accumulations + <= 4 In + <= 4 Out + 1 export adds per substep -> `10 n_sub`);
      4. BOOKKEEPING: the 7 combination operations and the native pickup / deposit / reconcile roundings
         (`2 * PHASE_RECONCILIATION_OPERATIONS`, as the existing reconciliation) at the largest cell operand.
    `bound = sum` of the four; each term is `MAPLE summation_error_bound_kg(n_i, scale_i)`. No multiplier, floor, mass
    resolution or percentage; zero operands give a zero bound, so a 1e-300 kg imbalance with zero operands is refused.
    Returns arrays `residual`, `abs_residual`, `bound`, `excess = |residual| - bound`, the per-component bound arrays
    and `n_bookkeeping_operations`."""
    from maple.surface.voxels._numerics import summation_error_bound_kg

    from maple_syrup.sediment_event import PHASE_RECONCILIATION_OPERATIONS

    native = 2 * PHASE_RECONCILIATION_OPERATIONS if native_operations is None else int(native_operations)
    zeros = xp.zeros_like(mobile_new)
    div_w = _divergence(wet["x_net"], wet["y_net"]) if wet is not None else zeros
    div_s = _divergence(splash["x_net"], splash["y_net"]) if splash is not None else zeros
    expected = (((pick_wet + pick_splash) - (div_w + div_s)) - (dep_wet + dep_splash))
    residual = (mobile_new - mobile_old) - expected
    cell_scale = xp.maximum(xp.maximum(xp.maximum(mobile_new, mobile_old), xp.maximum(pick_wet + pick_splash,
                                                                                        dep_wet + dep_splash)),
                            xp.maximum(xp.abs(div_w), xp.abs(div_s)))
    if wet is not None:
        cell_scale = xp.maximum(cell_scale, wet["mobile_before"])
    face_scale, n_faces = zeros, 0
    transport_bound = splash_bound = zeros
    if splash is not None:
        face_scale = xp.maximum(face_scale, _face_scale(xp, splash))
        n_faces += DIVERGENCE_FLOPS
        splash_bound = splash["tolerance"]
    if wet is not None:
        n_sub = int(wet["n_substeps"])
        face_scale = xp.maximum(face_scale, _face_scale(xp, wet))
        n_faces += DIVERGENCE_FLOPS + 10 * n_sub
        t_scale = xp.maximum(xp.maximum(wet["before"], wet["mobile_after_transfer"]),
                             xp.maximum(wet["internal_in"], wet["internal_out"]))
        transport_bound = summation_error_bound_kg(operation_count(n_sub, int(wet["n_bins"])), t_scale)
    faces_bound = summation_error_bound_kg(n_faces, face_scale) if n_faces else zeros
    rest_ops = COMBINATION_FLOPS + native
    rest_bound = summation_error_bound_kg(rest_ops, cell_scale)
    bound = ((splash_bound + transport_bound) + faces_bound) + rest_bound
    abs_residual = xp.abs(residual)
    return {"residual": residual, "abs_residual": abs_residual, "bound": bound, "excess": abs_residual - bound,
            "components": {"splash": splash_bound, "transport": transport_bound, "faces": faces_bound,
                           "rest": rest_bound}, "n_bookkeeping_operations": rest_ops}


# --- water ledger helpers ---------------------------------------------------------------------------
def _pond_ledger(pond: PondHydraulics, wl: WaterLedger) -> PondLedger:
    led = PondLedger(pond, HydraulicState(wl.t0_s, wl.depth0, wl.soil0))
    led.depth0, led.soil0 = wl.depth0, wl.soil0
    led.rain, led.intake, led.sat_return, led.drain = wl.rain, wl.intake, wl.sat_return, wl.drain
    led.pos, led.neg, led.net = ({"x": wl.pos_x, "y": wl.pos_y}, {"x": wl.neg_x, "y": wl.neg_y},
                                 {"x": wl.net_x, "y": wl.net_y})
    led.export, led.n_steps, led.t_end = wl.export, wl.n_steps, wl.t_end_s
    return led


def _water_add(xp: Any, wl: WaterLedger, accepted: PondStep) -> WaterLedger:
    """Direct immutable addition in exactly `PondLedger.add`'s operand order (no constructor, no anchor copies, no zero
    grids); `PondLedger` is only built for the audit closure."""
    col, step = accepted.step.column, accepted.step
    vx, vy = step.face_volume_m3["x"], step.face_volume_m3["y"]
    return replace(
        wl, t_end_s=accepted.state.t_s, rain=wl.rain + col.rain_m, intake=wl.intake + col.intake_m,
        sat_return=wl.sat_return + col.saturation_return_m, drain=wl.drain + col.drainage_m,
        pos_x=wl.pos_x + xp.maximum(vx, 0.0), neg_x=wl.neg_x + xp.maximum(-vx, 0.0), net_x=wl.net_x + vx,
        pos_y=wl.pos_y + xp.maximum(vy, 0.0), neg_y=wl.neg_y + xp.maximum(-vy, 0.0), net_y=wl.net_y + vy,
        export=wl.export + step.export_m3, n_steps=wl.n_steps + 1)
