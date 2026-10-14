"""Phase 5b: one coupled rainfall / infiltration / routing / wet-sediment
event on the ACTUAL MAPLE bed (pure helpers; the Plot 1 runner is
sediment_experiment.py).

State ownership
---------------
One authority for sediment and water: `sediment_bed.BedState` holds the
MAPLE voxel column, active layer, availability, ledger, committed
topography and `WaterState` (depth + water-borne mobile mass). SYRUP owns
the hydraulic depth, soil water and discharge (`storm.StormState`), the
transport memory (sediment velocity, `(ny, nx, nc)`) and the routing
graph / transport network / physics grid derived from the committed
terrain. No copy of the bed or of the terrain evolves on its own: every
mass change goes through `apply_water_process_demand` and every terrain
change through `commit_topography` (both composed by `sediment_bed`).

One accepted step of length `dt` (all local until published)
--------------------------------------------------------------
1. `storm.coupled_step`: accepted Phase 3 column + Phase 4b method-5
   routing on the ORIGINAL storm state (may raise `RoutingStepRejected`).
2. `sediment_physics_step` on the routed depth and velocity, the step's
   rain rate, the prescribed vegetation cover, the CURRENT active-layer
   holdings (pre-pickup) and the velocity memory -> pickup demand over
   `dt`, sediment velocity, deposition rate `1/L`, settle mask.
3. Publish the solver depth into the MAPLE water state; PICKUP call:
   `apply_water_process_demand` with the demand only (MAPLE caps by
   availability and holdings, refills the active layer from the column
   and moves the ACTUAL removal into the same cell's mobile pool).
4. `transport_step` on the actual post-pickup pool (exact `exp(-v dt/L)`
   reaction half-steps around upwind advection; substeps chosen from the
   sediment Courant number, see below). Newly picked-up mass is
   transported in the same step (no source lag).
5. Publish `WaterState(depth, T(M))`; DEPOSIT/EXPORT call:
   `apply_water_process_demand` with the transport's deposition and
   export requests and face crossings (removal zero). Dry / no-capacity
   cells request their whole pool, so mobile mass reaching a dry cell
   returns to the MAPLE bed; wet residual mobile mass is retained.
6. Commit: `sediment_bed.commit_bed` evaluates MAPLE's own triggers on the
   ledger. When a commit fires, MAPLE publishes the terrain under
   `constant_depth` (depth array-equal, displaced volume exactly 0). A graph
   failure (new pit, lost receiver) raises before anything is published.
   CW-B104 NET-ELEVATION-ONLY REROUTING: the routing graph is built from a
   HELD hydraulic routing source (`SedimentEventState.
   routing_source_elevation_m`, the committed elevation of the last actual
   routing build; set at initialization). After every accepted step the
   exact current surface (`sediment_bed.current_routing_surface_m`, MAPLE's
   commit arithmetic) is compared with it: max over active cells of |current
   - source| (signed within a cell, never cancelling across cells), against
   the actual `pending_elevation_trigger_m` with MAPLE's `>=`. Reaching it
   forces a genuine MAPLE commit if MAPLE's triggers did not fire; the
   routing is refreshed iff the POST-commit committed elevation still
   reaches it. Only then are the graph, transport network and physics grid
   rebuilt, the discharge re-initialised from the NEW conveyance and the
   UNCHANGED depth (`storm.initial_state`), the CSR phase rebound and the
   CUDA context re-prepared, and the source replaced. Every other commit
   (elapsed time, gross turnover, the final forced commit) is an accounting
   / geometry publication that leaves graph, network, grid, discharge,
   phase binding, CUDA context and source untouched.
7. Only now are time, the storm state, the bed, the velocity memory, the
   graph objects and every accumulator published.

Retry policy: `RoutingStepRejected` (water Courant / negative RHS) and
`TransportStepRejected` (sediment Courant beyond `max_transport_substeps`
substeps) both discard the whole unpublished attempt and halve `dt` from
the SAME state, bounded by `max_retries` and the retry floor `min_dt_s`
(a forced slice to a forcing / reporting boundary may be shorter; halving
below the floor is refused). Sediment substeps are chosen from the
maximum sediment velocity, `ceil(v_max dt / (dx courant_max))`, and
doubled on an FP-boundary rejection. Every other exception propagates
with nothing published.

Accounting (device-resident, bounded)
--------------------------------------
Gross pickup, deposition and export are accumulated per cell and class
from MAPLE's ground-truth results (never from requests); requested,
availability-refused, holdings-refused and numerically-residual amounts
are accumulated per class; the transport operator's own per-class budget
residual and tolerance are summed; the ledger's per-process totals are
captured from every commit before MAPLE resets them; true per-step peaks
(mobile mass, export rate, outlet discharge, depth, velocity) are tracked
independently of the reporting cadence; the water hydrograph keeps the
Phase 4 columns and a sediment hydrograph adds per-class mobile, export
and cumulative pickup / deposition rows. The global closure
`bed_initial + mobile_initial = bed_final + mobile_final + export` per
class is derived from `bed_inventory` (MAPLE's helper) at the two ends
and the accumulated actual export -- independently of the ledger.

Not here: splash, ecology, nutrients, an event-stop criterion, the dry
reset, wind alternation, restart. The run ends at the configured time.

Backends (CW-A1, conservative CUDA hydrology)
----------------------------------------------
Every loop operation is a namespace operation (`xp` of the graph). The
hydraulic sweep is selected by `StormControl.implementation`:

* `"array"`: the generic reference on EITHER namespace (NumPy or CuPy).
* `"numba"`: the compiled host sweep; refused for a CuPy event state.
* `"cuda"`: the prepared CUDA hydrology of `hydrology_cuda` (column +
  routing + reductions on the device, bisection or Newton); requires a
  CuPy graph / column / state and is refused for a NumPy state (it is not
  a CPU fallback and makes no host/device transfer of the state).

The wet laws (`sediment_physics_step`) and the characteristic transport
(`implementation="array"`) are namespace-generic and run on the CURRENT
active-layer holdings in either namespace; the compiled numba transport is
host-only, so `transport_implementation="numba"` is refused with CUDA
hydraulics (`SedimentEventControl.validated()`), and `"auto"` resolves to
`"array"` for both the array and the CUDA sweeps. Mixed pairings are
refused by `_check_backend_binding` BEFORE any hydraulic or MAPLE work.

A CUDA context (`hydrology_cuda.CudaHydrologyContext`) owns validated
device copies of the graph conveyance / column statics and is bound to the
exact graph and column objects. `evolve_sediment_event` keeps it in a
caller-owned `CudaHydraulics` lifecycle: prepared ONCE before the first
step (a counted download of the static graph for validation and upload of
the owned copies), reused unchanged by every attempt and retry of the same
graph (a retry is the same accepted state), and REBUILT for the refreshed
graph of every REROUTING commit before that commit is published (a preparation
failure leaves the commit unpublished). No preparation and no full-grid
static transfer happens per step; per attempted step the device traffic is
the one counted 144-byte packet read of `cuda_step_with_packet` plus the
MAPLE / SYRUP validation flag reads, which are counted by
`read_transfer_counters`. `sediment_coupled_step` called directly with a
CUDA control and no `hydraulics` prepares a context ONCE PER INVOCATION
(the documented convenience form, as `storm.coupled_step`). There is no
module-level context cache: a `CudaHydraulics` object belongs to its
caller, is bound by identity to the graph/column it prepared from, and is
re-prepared (counted) when handed a different graph.

Rerouting commits rebuild the routing graph on the host from one counted
transfer of the committed elevation (`sediment_bed.routing_graph_from_elevation`).
Preparation counts, costs and commit-time transfers are reported through
`CudaHydraulics.summary()`, not through new result fields (the checkpoint
registry encodes every `SedimentEventResult` field; adding one would
break older bundles). Current state (CW-B73 documentation correction):
`complete_event`, host checkpoints (same-backend restart only) and opt-in pit
storage run with CuPy states and are qualified on SMALL full cases; the
resident-window bed runner exists for large beds, but an actual original-size
Chastre GPU step has not yet fit the capped device memory, and no full large
(Chastre-scale) storm, restart or correspondence is established.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import Any

import numpy as np

from maple_syrup.characteristic_sparse import (
    SparseCharacteristicStep,
    SparsePhaseState,
    _sparse_characteristic_accounting_step,
    _SparseAccountingStep,
    _TransportAccountingStep,
    canonical_sparse_phase_state,
    empty_sparse_phase_state,
    rebind_sparse_phase_state,
    reconcile_sparse_rows,
    sparse_characteristic_step,
    validate_sparse_phase_state,
)
from maple_syrup.characteristic_transport import (
    DEFAULT_COURANT_MAX as CHARACTERISTIC_COURANT_MAX,
)
from maple_syrup.characteristic_transport import (
    DEFAULT_N_BINS,
    MAX_N_BINS,
    CharacteristicStep,
    PhaseState,
    characteristic_step,
    empty_phase_state,
    validate_phase_state,
)
from maple_syrup.infiltration import ColumnParameters
from maple_syrup.rainfall import RainfallField, RainfallSchedule
from maple_syrup.routing import RoutingGraph, RoutingStepRejected
from maple_syrup.sediment_bed import (
    BedContext,
    BedIntegrationError,
    BedState,
    TerrainReference,
    apply_bed_demand,
    apply_bed_pickup_transport_deposit,
    bed_already_committed,
    bed_by_cell_class,
    bed_inventory,
    bed_voxel_count,
    bound_maple_package_digest,
    check_graph_matches_routing_source,
    check_routing_source,
    commit_bed,
    current_routing_surface_m,
    JOURNAL_RECOGNIZER_MAPLE_DIGEST,
    is_carried_append_cap_refusal,
    is_windowed_bed,
    journal_recognizer_source,
    net_routing_displacement_m,
    paired_transaction_entry,
    pending_journal_records,
    publish_bed_storage,
    reroute_threshold_m,
    require_typed_journal_capacity_error,
    resolved_journal_cap,
    validate_deep_storage,
    with_water,
)
from maple_syrup.sediment_physics import (
    REGIME_CODES,
    PhysicsGrid,
    SedimentPhysicsParameters,
    SedimentPhysicsStep,
    physics_grid_from_graph,
    sediment_physics_step,
)
from maple_syrup.sediment_transport import (
    TransportError,
    TransportNetwork,
    TransportStep,
    TransportStepRejected,
    transport_network,
    transport_step,
    water_demand_from_transport,
)
from maple_syrup.sediment_transport import _water_demand_from_accounting_transport
from maple_syrup.storm import (
    HYDROGRAPH_COLUMNS,
    CoupledStep,
    StormControl,
    StormError,
    StormState,
    coupled_step,
    initial_state,
    plan_boundaries,
)
from maple_syrup.storm import _validate_state as validate_storm_state

__all__ = [
    "EXCHANGE_HISTORY_KEYS",
    "PHASE_RECONCILIATION_OPERATIONS",
    "PHASE_STORAGES",
    "SEDIMENT_SCALAR_COLUMNS",
    "SEDIMENT_VECTOR_COLUMNS",
    "TRANSPORT_IMPLEMENTATIONS",
    "TRANSPORT_SCHEMES",
    "CudaHydraulics",
    "SedimentCoupledStep",
    "SedimentEventControl",
    "SedimentEventError",
    "SedimentEventResult",
    "SedimentEventState",
    "canonical_phase_for",
    "copy_exchange_history",
    "evolve_sediment_event",
    "exchange_history_totals",
    "fold_commit_exchange_history",
    "initial_event_state",
    "morphology_summary",
    "new_exchange_history",
    "phase_for_network",
    "reconcile_phase",
    "sediment_coupled_step",
    "sediment_hydrograph_columns",
    "validate_exchange_history",
]

# Lateral transport schemes: the Phase 7b characteristic kernel with bounded
# phase bins (corrected default) and the Phase 5 well-mixed upwind operator,
# retained unchanged as an explicitly named comparison scheme.
TRANSPORT_SCHEMES = ("characteristic", "upwind")
# "auto" follows the hydraulic sweep implementation (`StormControl.implementation`). "cuda" is the sparse CSR
# device operator and is accepted ONLY with `phase_storage="csr"` on a CuPy state (dense storage has no CUDA kernel;
# the dense path on CuPy is the namespace-generic "array" kernel).
TRANSPORT_IMPLEMENTATIONS = ("auto", "array", "numba", "cuda")
# Phase STORAGE of the characteristic scheme (CW-B5). "dense": the accepted `(ny, nx, nc, B)` `PhaseState` and the
# dense kernels (default; every existing result, control and bundle is unchanged). "csr": the accepted sparse
# nonempty-bin `SparsePhaseState` and the CSR operator (`characteristic_sparse` / `characteristic_cuda_csr`): SAME
# physics, B, dynamic classes, LOCAL current-law speed/distance (per-cell v and r = 1/L each step) and conservative
# semantics; only the storage differs. The
# two are never converted into each other on an event path; a mismatched state/control pairing is refused.
PHASE_STORAGES = ("dense", "csr")
# CW-B102: when the actual MAPLE pending deep journal is published (see `SedimentEventControl.deep_publication`).
DEEP_PUBLICATIONS = ("step", "boundary")
# Roundings allowed between the kernel's predicted remaining pool and the
# ACTUAL MAPLE pool after the deposit/export call (T - Dep - E on both sides,
# MAPLE's own accounting of the applied requests): a declared operation count
# for `summation_error_bound_kg`, never a physical mass floor.
PHASE_RECONCILIATION_OPERATIONS = 16

# Sediment hydrograph: scalar columns, then per-class blocks (each block
# has nc columns, suffixed `_c1..c<nc>` in canonical class order).
SEDIMENT_SCALAR_COLUMNS = (
    "t_s",
    "mobile_kg",
    "cumulative_pickup_kg",
    "cumulative_deposition_kg",
    "cumulative_export_kg",
    "export_rate_kg_s",  # last accepted step's actual export / dt
    "commit_count",
    "graph_changes",
    "max_transport_substeps",
)
SEDIMENT_VECTOR_COLUMNS = (
    "mobile",
    "export_rate",
    "cumulative_pickup",
    "cumulative_deposition",
    "cumulative_export",
)


def sediment_hydrograph_columns(n_classes: int) -> tuple[str, ...]:
    names = list(SEDIMENT_SCALAR_COLUMNS)
    for block in SEDIMENT_VECTOR_COLUMNS:
        unit = "kg_s" if block == "export_rate" else "kg"
        names += [f"{block}_c{k + 1}_{unit}" for k in range(n_classes)]
    return tuple(names)


class SedimentEventError(RuntimeError):
    """Configuration, validation, guard or consistency failure of the
    coupled sediment event. Nothing is published and no input array is
    modified."""


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SedimentEventError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _strict_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise SedimentEventError(f"{name} must be an int >= {minimum}, got {value!r}")
    return int(value)


# --- control -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class SedimentEventControl:
    """Hydraulic control (`storm`) plus the sediment / commit controls.

    `transport_scheme` selects the lateral operator: `"characteristic"`
    (Phase 7b bounded phase bins, `phase_bins` position bins per cell and
    class, `transport_implementation` `"auto"` = follow the hydraulic sweep,
    or `"array"` / `"numba"`) or `"upwind"` (the unchanged Phase 5 operator;
    no phase state is allocated). `sediment_courant_max` bounds
    `v dt / (n_substeps dx)` for the transport operator (the characteristic
    kernel needs <= 0.5 for one face crossing per packet per substep;
    upwind positivity needs <= 1); `max_transport_substeps` caps the substep
    count before the whole step is halved instead. `commit` evaluates
    MAPLE's commit triggers after every accepted step; `force_final_commit`
    commits any pending bed change at the configured end time (real time,
    no fabricated elapsed time) so the returned terrain corresponds to the
    returned bed.
    """

    storm: StormControl = field(default_factory=StormControl)
    sediment_courant_max: float = CHARACTERISTIC_COURANT_MAX
    max_transport_substeps: int = 64
    commit: bool = True
    force_final_commit: bool = True
    max_commit_log: int = 256
    max_rejections_recorded: int = 100
    transport_scheme: str = "characteristic"
    phase_bins: int = DEFAULT_N_BINS
    transport_implementation: str = "auto"
    # CW-B5 (defaults LAST, so every existing construction, identity and bundle is unchanged): the phase STORAGE of the
    # characteristic scheme (`PHASE_STORAGES`) and the optional predeclared CSR capacity handed to the sparse operator
    # (`max_nnz`; None = no cap on the host, and on CUDA the operator's own large-case rule applies: see
    # `characteristic_cuda_csr.DEFAULT_UNBOUNDED_NNZ_BYTES`). Neither ever truncates mass or reduces B / classes.
    phase_storage: str = "dense"
    phase_max_nnz: int | None = None
    # CW-D2 opt-in: accumulate the ACTUAL per-cell/class MAPLE ledger touch counts and gross physical activity
    # through every commit (receipts captured by MAPLE after the avalanche fold-in, before the ledger clear) and
    # keep them on the result (`SedimentEventResult.exchange_history`). Requires the same flag on the
    # `BedContext` and a MAPLE dependency whose commit accepts `record_exchange_history`. Default False: no
    # accumulator, no new result value, no change to any published array.
    record_exchange_history: bool = False
    # CW-B102 opt-in (defaults LAST and omitted from identities / bundles at their defaults): WHEN the actual MAPLE
    # pending deep journal of a resident-window bed is published as a new deep-store version. "step" (default) =
    # every accepted step and commit, exactly as before. "boundary" = only at report-boundary rows, before return,
    # when the carried journal reaches `deep_publication_high_water_records`, and in the bounded flush before a
    # recognized carried-journal append-cap retry. Physical state, commits and their times are identical; only
    # deep-file durability (versions / generations / lineage) differs. Context-dependent checks (windowed CSR bed,
    # actual journal cap, MAPLE binding, callbacks) run at event entry.
    deep_publication: str = "step"
    deep_publication_high_water_records: int | None = None

    def validated(self) -> SedimentEventControl:
        if not isinstance(self.storm, StormControl):
            raise SedimentEventError("storm must be a StormControl")
        self.storm.validated()
        if self.transport_scheme not in TRANSPORT_SCHEMES:
            raise SedimentEventError(f"transport_scheme must be one of {TRANSPORT_SCHEMES}, got "
                                     f"{self.transport_scheme!r}")
        if self.transport_implementation not in TRANSPORT_IMPLEMENTATIONS:
            raise SedimentEventError(f"transport_implementation must be one of {TRANSPORT_IMPLEMENTATIONS}, got "
                                     f"{self.transport_implementation!r}")
        if self.phase_storage not in PHASE_STORAGES:
            raise SedimentEventError(f"phase_storage must be one of {PHASE_STORAGES}, got {self.phase_storage!r}")
        if self.phase_storage == "csr" and self.transport_scheme != "characteristic":
            raise SedimentEventError("phase_storage 'csr' is a storage of the characteristic scheme's phase; the upwind "
                                     "scheme allocates no phase and refuses the option")
        if self.phase_storage == "csr" and self.transport_implementation == "array":
            raise SedimentEventError("phase_storage 'csr' has no array transport kernel: use transport_implementation "
                                     "'auto' (numba on a NumPy state, cuda on a CuPy state), 'numba' or 'cuda'")
        if self.phase_storage != "csr" and self.transport_implementation == "cuda":
            raise SedimentEventError("transport_implementation 'cuda' is the sparse CSR device operator and needs "
                                     "phase_storage 'csr'; the dense phase on CuPy uses transport_implementation 'array'")
        if self.phase_max_nnz is not None:
            if self.phase_storage != "csr":
                raise SedimentEventError("phase_max_nnz is a CSR capacity option; it needs phase_storage 'csr'")
            _strict_int(self.phase_max_nnz, "phase_max_nnz", 0)
        if self.storm.implementation == "cuda" and self.transport_implementation == "numba":
            # The CUDA hydrology needs a CuPy event state; the compiled transport kernel runs on host NumPy only.
            # No hidden transfer either way: refuse the pairing before any physics.
            raise SedimentEventError("storm implementation 'cuda' (CuPy event state) cannot be paired with "
                                     "transport_implementation 'numba' (host-only compiled kernel); use 'array' "
                                     "or 'auto' (which resolves to 'array' under CUDA); no fallback")
        _strict_int(self.phase_bins, "phase_bins")
        if int(self.phase_bins) > MAX_N_BINS:
            raise SedimentEventError(f"phase_bins must be <= {MAX_N_BINS}, got {self.phase_bins!r}")
        cr = _real(self.sediment_courant_max, "sediment_courant_max")
        limit = CHARACTERISTIC_COURANT_MAX if self.transport_scheme == "characteristic" else 1.0
        if not (0.0 < cr <= limit):
            raise SedimentEventError(f"sediment_courant_max must lie in (0, {limit}] for the "
                                     f"{self.transport_scheme} scheme, got {cr!r}")
        _strict_int(self.max_transport_substeps, "max_transport_substeps")
        _strict_int(self.max_commit_log, "max_commit_log", 0)
        _strict_int(self.max_rejections_recorded, "max_rejections_recorded", 0)
        for name in ("commit", "force_final_commit", "record_exchange_history"):
            if not isinstance(getattr(self, name), bool):
                raise SedimentEventError(f"{name} must be bool")
        if self.record_exchange_history and not self.commit:
            raise SedimentEventError("record_exchange_history needs actual MAPLE commits (commit=True); the history "
                                     "is captured at commits")
        if self.deep_publication not in DEEP_PUBLICATIONS:
            raise SedimentEventError(f"deep_publication must be one of {DEEP_PUBLICATIONS}, got "
                                     f"{self.deep_publication!r}")
        if self.deep_publication == "boundary":
            if self.deep_publication_high_water_records is None:
                raise SedimentEventError("deep_publication 'boundary' needs deep_publication_high_water_records")
            _strict_int(self.deep_publication_high_water_records, "deep_publication_high_water_records")
            if not self.csr:
                raise SedimentEventError("deep_publication 'boundary' needs the characteristic CSR transport "
                                         "(phase_storage 'csr') of the resident-window event")
        elif self.deep_publication_high_water_records is not None:
            raise SedimentEventError("deep_publication_high_water_records is a 'boundary' option")
        return self

    @property
    def characteristic(self) -> bool:
        return self.transport_scheme == "characteristic"

    @property
    def csr(self) -> bool:
        """True when the characteristic phase is stored as the sparse nonempty-bin CSR state."""
        return self.characteristic and self.phase_storage == "csr"

    @property
    def cuda(self) -> bool:
        """True when the hydraulic sweep is the prepared CUDA hydrology."""
        return self.storm.implementation == "cuda"

    def resolved_transport_implementation(self) -> str:
        """Dense storage: `"array"` or `"numba"` (`"auto"` follows `storm.implementation`: `"numba"` only for the
        numba sweep; the array AND the CUDA sweeps resolve to the namespace-generic array kernel). CSR storage:
        `"numba"` (host) or `"cuda"` (device); `"auto"` resolves to `"cuda"` under the CUDA sweep and to `"numba"`
        otherwise (the CSR operator has no array kernel)."""
        if self.transport_implementation == "auto":
            if self.phase_storage == "csr":
                return "cuda" if self.storm.implementation == "cuda" else "numba"
            return "numba" if self.storm.implementation == "numba" else "array"
        return self.transport_implementation

    def backend_record(self) -> dict[str, Any]:
        """Host record of the resolved hydraulic / transport pairing (for runner summaries)."""
        transport = self.resolved_transport_implementation() if self.characteristic else None
        record = {"hydraulics": self.storm.implementation,
                  "root_solver": self.storm.root_solver,
                  "transport_implementation": transport,
                  "required_namespace": "cupy" if self.cuda or transport == "cuda" else (
                      "numpy" if self.storm.implementation == "numba"
                      or (self.storm.implementation == "array" and self.storm.root_solver == "newton")
                      or (self.characteristic and transport == "numba")
                      else "numpy_or_cupy"),
                  "rule": "cuda needs a CuPy state and the array transport; numba (sweep or transport) needs a NumPy "
                          "state; array runs on either namespace; mixed pairings are refused before any work"}
        if self.csr:  # keys added only for the CSR storage so every dense record is unchanged
            record["phase_storage"] = "csr"
            record["rule"] = ("csr: the sparse operator runs as numba on a NumPy state or cuda on a CuPy state, never "
                              "converted; dense: " + record["rule"])
        return record

    def transport_record(self) -> dict[str, Any]:
        record = {"scheme": self.transport_scheme,
                  "phase_bins": int(self.phase_bins) if self.characteristic else None,
                  "implementation": self.resolved_transport_implementation() if self.characteristic else None,
                  "sediment_courant_max": float(self.sediment_courant_max),
                  "pickup_position": "upstream face (x = 0), legacy flow_distrib convention" if self.characteristic
                  else "well-mixed cell (Phase 5 upwind operator)"}
        if self.csr:  # keys added only for the CSR storage so every dense record (and its hash) is unchanged
            record["phase_storage"] = "csr"
            record["phase_max_nnz"] = None if self.phase_max_nnz is None else int(self.phase_max_nnz)
            record["storage_rule"] = ("nonempty-bin CSR of the same B-bin partition; all classes dynamic; no dense "
                                      "(ny, nx, nc, B) array on the event path; mass authority stays MAPLE's pool")
        return record


# --- state ----------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SedimentEventState:
    """Everything an accepted step publishes. `storm.depth_m` is the
    solver depth and equals `bed.water.depth_m` after every accepted step;
    `graph`, `network` and `grid` are derived from the HELD hydraulic
    routing source `routing_source_elevation_m` (CW-B104: the committed
    elevation of the last actual routing build, which may deliberately lag
    `bed.committed_topography` by less than the net-elevation threshold)."""

    storm: StormState
    bed: BedState
    sediment_velocity_m_s: Any  # (ny, nx, nc) transport memory
    graph: RoutingGraph
    network: TransportNetwork
    grid: PhysicsGrid
    terrain: TerrainReference
    # Sub-cell position partition of `bed.water.mobile_mass_by_cell_class_kg`
    # (characteristic scheme: the dense `PhaseState` or the sparse CSR
    # `SparsePhaseState`, by the control's `phase_storage`) or None (upwind
    # scheme). Never a second inventory.
    phase: Any = None
    # CW-B104 (default LAST; required by every event entry, None is refused, never guessed): the `(ny, nx)` float64
    # committed elevation, in the graph namespace, from which `graph` was built (initialization or the last actual
    # reroute). A geometry reference only: never mass, never a second terrain authority.
    routing_source_elevation_m: Any = None

    @property
    def t_s(self) -> float:
        return self.storm.t_s


def canonical_phase_for(state: SedimentEventState) -> Any:
    """Canonical empty phase of the same storage/shape/bins/namespace as
    `state.phase` (None for the upwind scheme): the partition of an empty
    mobile pool. CSR: the EMPTY state (`row_ptr` zeros, `nnz` 0) bound to the
    state's CURRENT network; no dense array."""
    phase = state.phase
    if phase is None:
        return None
    if isinstance(phase, SparsePhaseState):
        return empty_sparse_phase_state(state.network, phase.n_classes, phase.n_bins)
    return empty_phase_state(phase.shape, phase.n_classes, phase.n_bins, phase.dx_m, xp=phase.xp)


def phase_for_network(phase: Any, network: TransportNetwork) -> Any:
    """The phase to publish with a REFRESHED network (commit / reroute): the dense phase is carried over as is
    (its positions are distances along each cell's current outflow direction, the documented steering); the CSR
    phase is EXPLICITLY rebound to the new graph digest and slot maps (`rebind_sparse_phase_state`: same shape,
    dx and active set required, bins / fractions / positions carried over unchanged -- the same steering)."""
    if isinstance(phase, SparsePhaseState):
        return rebind_sparse_phase_state(phase, network)
    return phase


def _check_backend_binding(xp: ModuleType, control: SedimentEventControl) -> None:
    """Refuse a hydraulic / transport implementation that cannot run on the
    event state's namespace `xp`, BEFORE any kernel or MAPLE call. No
    host/device transfer of the state is ever made: CUDA is not a CPU
    fallback and numba is not a device fallback."""
    name = getattr(xp, "__name__", repr(xp))
    storm = control.storm
    if storm.implementation == "cuda" and xp is np:
        raise SedimentEventError("storm implementation 'cuda' needs a CuPy event state (graph, column, bed and "
                                 f"arrays built with the cupy backend); the event state lives in {name!r} and no "
                                 "host-to-device transfer is made (CUDA is not a CPU fallback); refusing before any "
                                 "kernel or MAPLE call")
    if storm.implementation == "numba" and xp is not np:
        raise SedimentEventError("storm implementation 'numba' runs on host NumPy arrays only; the event state lives "
                                 f"in {name!r} and no device-to-host transfer is made (use 'cuda' or 'array'); "
                                 "refusing before any kernel or MAPLE call")
    if storm.implementation == "array" and storm.root_solver == "newton" and xp is not np:
        raise SedimentEventError("root_solver 'newton' with the array sweep runs on host NumPy graphs only (the CUDA "
                                 f"Newton needs implementation 'cuda'); the event state lives in {name!r}; refusing "
                                 "before any kernel or MAPLE call")
    if control.characteristic and control.resolved_transport_implementation() == "numba" and xp is not np:
        raise SedimentEventError("transport_implementation 'numba' runs on host NumPy arrays only; the event state "
                                 f"lives in {name!r}; use 'array'; refusing before any kernel or MAPLE call")
    if control.characteristic and control.resolved_transport_implementation() == "cuda" and xp is np:
        raise SedimentEventError("transport_implementation 'cuda' (sparse CSR device operator) needs a CuPy event "
                                 f"state; the event state lives in {name!r} and no host-to-device transfer is made; "
                                 "refusing before any kernel or MAPLE call")


def _validate_any_phase(phase: Any, mobile_kg: Any, network: TransportNetwork) -> None:
    """Dispatch the partition validation by the TYPED phase state (dense or CSR); nothing is densified."""
    if isinstance(phase, SparsePhaseState):
        validate_sparse_phase_state(phase, mobile_kg, network)
    else:
        validate_phase_state(phase, mobile_kg, network)


def _bind_phase_to_control(state: SedimentEventState, control: SedimentEventControl) -> SedimentEventControl:
    """Cheap binding checks every public entry runs BEFORE any work: the
    control validates (scheme, bins, implementation, storage, Courant), the
    implementations can run on the state's namespace, the phase is present
    exactly for the characteristic scheme, its TYPE is the control's storage
    (a dense state under a CSR control or a CSR state under a dense control
    is refused, never converted), its bin count, class count and cell size
    are the control's and the network's. No phase-array validation here (the
    kernel validates the arrays before returning any result; the drivers
    validate the partition separately)."""
    if not isinstance(control, SedimentEventControl):
        raise SedimentEventError("control must be a SedimentEventControl")
    control = control.validated()
    _check_backend_binding(state.graph.xp, control)
    phase = state.phase
    if control.characteristic:
        if phase is None:
            raise SedimentEventError("the characteristic transport scheme needs a phase state on the event state "
                                     "(build it with initial_event_state(..., control=control))")
        expected = SparsePhaseState if control.csr else PhaseState
        if not isinstance(phase, expected):
            raise SedimentEventError(f"event state phase is a {type(phase).__name__} but the control's phase_storage "
                                     f"{control.phase_storage!r} needs a {expected.__name__}; storages are never "
                                     "converted implicitly (build the state with the same control)")
        if phase.n_bins != int(control.phase_bins):
            raise SedimentEventError(f"phase state has {phase.n_bins} bins but the control asks for "
                                     f"{control.phase_bins}; refusing to reinterpret the partition")
        if phase.dx_m != state.network.dx_m or phase.xp is not state.graph.xp:
            raise SedimentEventError("event state phase does not belong to the event's network (cell size or namespace)")
        if control.csr and (tuple(phase.shape) != tuple(state.graph.shape)
                            or phase.network_sha256 != state.network.graph_input_sha256):
            raise SedimentEventError("CSR phase state is bound to a different routing graph than the event's network "
                                     "(rebind explicitly after a routing refresh)")
    elif phase is not None:
        raise SedimentEventError("the upwind transport scheme carries no phase state; the event state has one")
    return control


def _check_phase_against_control(state: SedimentEventState, control: SedimentEventControl | None) -> None:
    """The phase state must match the control's scheme, storage and bin count,
    and partition the actual MAPLE mobile mass. Raises before any mutation."""
    phase = state.phase
    if control is None:
        if phase is not None:
            _validate_any_phase(phase, state.bed.water.mobile_mass_by_cell_class_kg, state.network)
        return
    control = _bind_phase_to_control(state, control)
    if control.characteristic:
        _validate_any_phase(phase, state.bed.water.mobile_mass_by_cell_class_kg, state.network)


def _check_event_inputs(state: SedimentEventState, context: BedContext, column: ColumnParameters,
                        sediment: SedimentPhysicsParameters, root_tolerance_m: float,
                        control: SedimentEventControl | None = None) -> None:
    from maple.core.backend import (
        DeferredChecks,
        MixedArrayNamespaceError,
        array_namespace,
        finite_flag,
        is_array,
        negative_flag,
        to_host,
        true_flag,
    )

    graph = state.graph
    xp, shape = graph.xp, graph.shape
    g = context.geometry
    nc = len(context.grain_classes.classes)
    if (g.ny, g.nx) != shape or g.dx_m != g.dy_m or g.dx_m != graph.dx_m:
        raise SedimentEventError("bed geometry, routing graph shape and spacing must agree (square cells)")
    if context.depth_update_rule != "constant_depth":
        raise SedimentEventError("the event requires the constant_depth ownership rule (SYRUP owns depth)")
    if not isinstance(state.network, TransportNetwork) or state.network.graph_input_sha256 != graph.input_sha256:
        raise SedimentEventError("transport network does not belong to the current routing graph")
    if not isinstance(state.grid, PhysicsGrid) or state.grid.shape != shape or state.grid.xp is not xp:
        raise SedimentEventError("physics grid does not match the routing graph")
    # CW-C2: the physics grid's terminal-storage mask (which forces the settle of terminal cells) must be exactly
    # the current network's (which the network validated against the current graph). Host metadata first (a
    # strict grid and network both carry None), then one batched flag read below for the mask contents.
    grid_terminal = getattr(state.grid, "terminal", None)
    network_terminal = getattr(state.network, "terminal_flat", None)
    if (grid_terminal is None) != (network_terminal is None):
        raise SedimentEventError("physics grid terminal-storage mask does not match the transport network of the "
                                 "current routing graph (one carries terminal cells, the other does not)")
    if grid_terminal is not None and (not is_array(grid_terminal) or tuple(grid_terminal.shape) != shape
                                      or grid_terminal.dtype != np.bool_):
        raise SedimentEventError("physics grid terminal-storage mask must be a bool (ny, nx) array")
    if sediment.n_classes != nc or sediment.xp is not xp:
        raise SedimentEventError("sediment parameters must have the bed's class count and the graph namespace")
    diameters = np.array([c.diameter_m for c in context.grain_classes.classes], dtype=np.float64)
    # `diameter_m` is the parameters' host copy by contract; the explicit `to_host` keeps this static comparison on
    # the host for any namespace (a NumPy array passes through unchanged).
    if not np.array_equal(diameters, to_host(sediment.diameter_m)):
        raise SedimentEventError("sediment parameter diameters differ from the MAPLE grain classes")
    densities = {float(c.particle_density_kg_m3) for c in context.grain_classes.classes}
    if densities != {float(sediment.particle_density_kg_m3)}:
        raise SedimentEventError("sediment particle density differs from the MAPLE grain classes")
    if state.terrain.initial_committed_elevation_m.shape != shape or state.terrain.dx_m != graph.dx_m:
        raise SedimentEventError("terrain reference does not match the routing graph")
    try:
        check_routing_source(state.routing_source_elevation_m, shape, xp)
    except BedIntegrationError as exc:
        raise SedimentEventError(str(exc)) from None
    validate_storm_state(graph, state.storm, root_tolerance_m, column)
    v = state.sediment_velocity_m_s
    if not is_array(v) or tuple(v.shape) != (*shape, nc) or v.dtype != np.float64:
        raise SedimentEventError(f"sediment_velocity_m_s must be float64 {(*shape, nc)}")
    try:
        namespace = array_namespace(v, state.bed.water.depth_m, state.bed.active_layer.mass_kg, graph.conveyance)
    except MixedArrayNamespaceError as exc:
        raise SedimentEventError(f"event arrays are in mixed namespaces: {exc}") from None
    if namespace is not xp:
        raise SedimentEventError(f"event arrays must be in the graph namespace {xp.__name__!r}")
    checks = DeferredChecks()
    checks.require(finite_flag(v), "sediment_velocity_m_s must be finite")
    checks.forbid(negative_flag(v), "sediment_velocity_m_s must be >= 0")
    checks.forbid(true_flag(state.bed.water.depth_m != state.storm.depth_m),
                  "bed.water.depth_m must equal the solver depth storm.depth_m")
    active3 = graph.active_flat.reshape(shape)[..., None]
    checks.forbid(true_flag(~active3 & (state.bed.water.mobile_mass_by_cell_class_kg != 0.0)),
                  "mobile mass on an inactive cell")
    if grid_terminal is not None:
        try:
            array_namespace(grid_terminal, network_terminal)
        except MixedArrayNamespaceError as exc:
            raise SedimentEventError(f"terminal-storage masks are in mixed namespaces: {exc}") from None
        checks.forbid(true_flag(grid_terminal != network_terminal.reshape(shape)),
                      "physics grid terminal-storage mask differs from the transport network's terminal cells")
    try:
        checks.resolve()
    except ValueError as exc:
        raise SedimentEventError(str(exc)) from None
    try:
        _check_phase_against_control(state, control)
    except TransportError as exc:
        raise SedimentEventError(f"phase state invalid: {exc}") from None


def initial_event_state(graph: RoutingGraph, terrain: TerrainReference, bed: BedState, context: BedContext,
                        soil_water_m: Any, *, t_s: float = 0.0,
                        control: SedimentEventControl | None = None) -> SedimentEventState:
    """Event state at `t_s` from a routing graph, its terrain reference,
    a MAPLE bed (its `water.depth_m` is the initial solver depth, its
    mobile mass the initial load) and the retained soil water. Discharge
    is initialised from the depth (`storm.initial_state`); the transport
    memory starts at zero (no prior sediment motion). The phase state
    follows `control` (default control: characteristic scheme with
    `DEFAULT_N_BINS`): any initial mobile load sits at the upstream face
    (canonical partition); the upwind scheme allocates no phase.

    CW-B104 INITIALIZATION: the hydraulic routing source is the bed's current
    MAPLE committed elevation (a fresh array in the bed namespace), and
    `graph` must be exactly the graph `routing_graph_from_elevation(terrain,
    source)` builds (one host rebuild, compared on the graph's host evidence;
    a mismatch is refused, nothing is refreshed)."""
    from maple.core.types.topographic_commit import committed_elevation_m

    control = (SedimentEventControl() if control is None else control).validated()
    source = committed_elevation_m(bed.committed_topography)
    try:
        check_routing_source(source, graph.shape, graph.xp)
        check_graph_matches_routing_source(graph, terrain, source)
    except BedIntegrationError as exc:
        raise SedimentEventError(f"initial routing: {exc}") from None
    nc = len(context.grain_classes.classes)
    storm = initial_state(graph, bed.water.depth_m, soil_water_m, t_s=t_s)
    velocity = graph.xp.zeros((*graph.shape, nc), dtype=np.float64)
    network = transport_network(graph)
    if not control.characteristic:
        phase = None
    elif control.csr:
        # CSR: the canonical partition of the initial pool built DIRECTLY from the mobile mask (one stored bin at the
        # upstream face per positive cell/class, nothing elsewhere); no (ny, nx, nc, B) array anywhere.
        phase = canonical_sparse_phase_state(network, bed.water.mobile_mass_by_cell_class_kg, int(control.phase_bins))
    else:
        phase = empty_phase_state(graph.shape, nc, int(control.phase_bins), graph.dx_m, xp=graph.xp)
    return SedimentEventState(storm=storm, bed=bed, sediment_velocity_m_s=velocity, graph=graph,
                              network=network, grid=physics_grid_from_graph(graph),
                              terrain=terrain, phase=phase, routing_source_elevation_m=source)


# --- CUDA hydrology lifecycle ----------------------------------------------------------------------------
def _hydrology_cuda() -> ModuleType:
    """Lazy import: nothing here needs CuPy or the CUDA kernels unless a CUDA control is used."""
    from maple_syrup import hydrology_cuda

    return hydrology_cuda


def _prepare_cuda_context(graph: RoutingGraph, column: ColumnParameters, *, mode: str) -> Any:
    """`hydrology_cuda.prepare_cuda_hydrology` (thin wrapper so tests can substitute a fake)."""
    return _hydrology_cuda().prepare_cuda_hydrology(graph, column, mode=mode)


def _cuda_prepared_step(context: Any, rain_rate_m_per_s: Any, storm: StormState, dt_s: float,
                        control: StormControl) -> CoupledStep:
    """`hydrology_cuda.prepared_coupled_step` on a prepared context (thin wrapper, see above)."""
    return _hydrology_cuda().prepared_coupled_step(context, rain_rate_m_per_s, storm, dt_s, control)


def _load_cuda_newton_kernels() -> None:
    _hydrology_cuda().load_newton_kernels()


CUDA_MODES = ("auto", "fused", "split")
_HYDRAULICS_REASONS = ("initial", "commit", "rebound", "step")


@dataclass(frozen=True, eq=False)
class _PreparedHydraulics:
    """A freshly prepared context and its record, not yet adopted by the lifecycle."""

    context: Any
    record: dict[str, Any]


class CudaHydraulics:
    """Caller-owned lifecycle of the prepared CUDA hydrology context of one
    coupled event (module docstring, "Backends").

    Holds at most ONE `hydrology_cuda.CudaHydrologyContext`, bound by object
    identity (weak references) and array metadata to the exact graph and
    column parameters it was prepared from (`CudaHydrologyContext.is_bound_to`).
    `bind` returns the held context when it is bound to the given graph /
    column and otherwise prepares a new one (counted, logged with its
    reason); `prepare` / `adopt` split that into a two-phase form so the
    event loop can prepare the context of a refreshed graph BEFORE
    publishing the commit and adopt it only once the commit is published.
    `record_commit_transfers` accumulates the counted host/device traffic of
    commits (the committed-elevation download of the host graph rebuild).

    Not a cache keyed by anything global: one object per caller, no sharing
    across events, nothing persisted (a context is reconstructible from the
    graph and the column and is never written to a checkpoint). Mutable,
    single-threaded. `summary()` is the host record a runner reports.
    """

    def __init__(self, *, mode: str = "auto", max_log: int = 256) -> None:
        if mode not in CUDA_MODES:
            raise SedimentEventError(f"mode must be one of {CUDA_MODES}, got {mode!r}")
        self.mode = mode
        self.max_log = _strict_int(max_log, "max_log", 0)
        self._context: Any = None
        self.n_preparations = 0
        self.n_binding_hits = 0  # bind() calls served by the held context without preparation
        self.preparation_wall_s = 0.0
        self.kernel_load_s = 0.0
        self.preparation_host_to_device_bytes = 0
        self.preparation_device_to_host_bytes = 0
        self.n_commits_recorded = 0
        self.commit_host_to_device_bytes = 0
        self.commit_device_to_host_bytes = 0
        self.commit_device_to_host = 0
        self.commit_scalar_reads = 0
        self.log: list[dict[str, Any]] = []

    @property
    def context(self) -> Any:
        """The held context (None before the first preparation)."""
        return self._context

    def is_bound_to(self, graph: RoutingGraph, column: ColumnParameters) -> bool:
        return self._context is not None and bool(self._context.is_bound_to(graph, column))

    def prepare(self, graph: RoutingGraph, column: ColumnParameters, *, t_s: float,
                reason: str) -> _PreparedHydraulics | None:
        """Prepare a context for `graph` / `column` unless the held one is
        already bound to them (then None). Nothing is adopted here; a
        preparation failure (`CudaUnavailableError`,
        `CudaHydrologyPreparationError`, `RoutingError`) propagates with the
        lifecycle unchanged."""
        if reason not in _HYDRAULICS_REASONS:
            raise SedimentEventError(f"reason must be one of {_HYDRAULICS_REASONS}, got {reason!r}")
        if self.is_bound_to(graph, column):
            return None
        context = _prepare_cuda_context(graph, column, mode=self.mode)
        record = {
            "t_s": float(t_s), "reason": reason, "n_preparation": self.n_preparations + 1,
            "graph_input_sha256": str(getattr(context, "graph_input_sha256", "")),
            "preparation_s": float(getattr(context, "preparation_s", 0.0)),
            "kernel_load_s": float(getattr(context, "kernel_load_s", 0.0)),
            "host_to_device_bytes": int(getattr(context, "host_to_device_bytes", 0)),
            "device_to_host_bytes": int(getattr(context, "device_to_host_bytes", 0)),
            "static_bytes": int(getattr(context, "static_bytes", 0)),
            "mode": getattr(context, "mode", None),
            "replaced_context": self._context is not None,
        }
        return _PreparedHydraulics(context=context, record=record)

    def adopt(self, prepared: _PreparedHydraulics) -> Any:
        """Install a prepared context (accumulating its counted costs) and return it."""
        if not isinstance(prepared, _PreparedHydraulics):
            raise SedimentEventError("adopt needs the value returned by prepare()")
        record = prepared.record
        self._context = prepared.context
        self.n_preparations += 1
        self.preparation_wall_s += record["preparation_s"]
        self.kernel_load_s += record["kernel_load_s"]
        self.preparation_host_to_device_bytes += record["host_to_device_bytes"]
        self.preparation_device_to_host_bytes += record["device_to_host_bytes"]
        if len(self.log) < self.max_log:
            self.log.append(dict(record))
        return self._context

    def bind(self, graph: RoutingGraph, column: ColumnParameters, *, t_s: float = 0.0, reason: str = "rebound") -> Any:
        """The context bound to `graph` / `column`: the held one when bound,
        otherwise a newly prepared and adopted one (counted)."""
        prepared = self.prepare(graph, column, t_s=t_s, reason=reason)
        if prepared is None:
            self.n_binding_hits += 1
            return self._context
        return self.adopt(prepared)

    def record_commit_transfers(self, delta: Any) -> None:
        """Accumulate the counted transfers of one commit (a `TransferCounters` delta)."""
        self.n_commits_recorded += 1
        self.commit_host_to_device_bytes += int(delta.host_to_device_bytes)
        self.commit_device_to_host_bytes += int(delta.device_to_host_bytes)
        self.commit_device_to_host += int(delta.device_to_host)
        self.commit_scalar_reads += int(delta.scalar_reads)

    def summary(self) -> dict[str, Any]:
        ctx = self._context
        return {
            "mode": self.mode,
            "n_preparations": self.n_preparations,
            "n_binding_hits": self.n_binding_hits,
            "preparation_wall_s": self.preparation_wall_s,
            "kernel_load_s": self.kernel_load_s,
            "preparation_host_to_device_bytes": self.preparation_host_to_device_bytes,
            "preparation_device_to_host_bytes": self.preparation_device_to_host_bytes,
            "commits": {"n_recorded": self.n_commits_recorded,
                        "host_to_device_bytes": self.commit_host_to_device_bytes,
                        "device_to_host_bytes": self.commit_device_to_host_bytes,
                        "device_to_host_transfers": self.commit_device_to_host,
                        "scalar_reads": self.commit_scalar_reads,
                        "note": "counted traffic of every actual commit_bed (rerouting or not): the host validation "
                                "reads inside MAPLE's commit, the net-displacement scalar read and, on a reroute, "
                                "the graph refresh's committed-elevation download; the per-step net-elevation gate "
                                "is outside this window; the context re-preparation of a refreshed graph is "
                                "accounted under preparation_*"},
            "context": None if ctx is None else ctx.summary(),
            "log": list(self.log),
            "log_truncated": len(self.log) < self.n_preparations,
            "rule": "one context per graph/column binding: prepared before the first step, reused by every attempt "
                    "and retry of the same graph and across non-rerouting commits, re-prepared for the refreshed "
                    "graph of every net-elevation reroute before the commit is published; no per-step preparation "
                    "or static transfer; nothing persisted",
        }


def _cuda_hydraulic_step(state: SedimentEventState, column: ColumnParameters, rain_rate_m_per_s: Any, dt: float,
                         control: SedimentEventControl, hydraulics: CudaHydraulics | None) -> CoupledStep:
    """Step 1 on the prepared CUDA hydrology. With a lifecycle the context
    bound to the state's graph is used (re-prepared and counted only if the
    lifecycle holds none for this graph); without one a context is prepared
    for THIS call only (convenience form: one counted static download and
    upload per invocation, documented, never used by the event loop)."""
    if hydraulics is None:
        context = _prepare_cuda_context(state.graph, column, mode="auto")
    else:
        context = hydraulics.bind(state.graph, column, t_s=state.t_s, reason="step")
    return _cuda_prepared_step(context, rain_rate_m_per_s, state.storm, dt, control.storm)


# --- one coupled attempt ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SedimentCoupledStep:
    """One accepted (not yet committed / published) coupled attempt."""

    dt_s: float
    storm: CoupledStep
    physics: SedimentPhysicsStep
    pickup: Any  # MAPLE WaterProcessResult of the pickup call
    transport: TransportStep
    deposit: Any  # MAPLE WaterProcessResult of the deposit / export call
    bed: BedState  # after both MAPLE calls; water = (routed depth, mobile after deposit / export)
    n_substeps: int
    n_transport_rejections: int
    characteristic: Any = None  # the kernel result: CharacteristicStep (dense) or SparseCharacteristicStep (csr)
    phase: Any = None  # partition of `bed.water` mobile mass, reconciled (PhaseState or SparsePhaseState)
    phase_reconciliation: dict[str, Any] | None = None  # 0-d namespace values (see `reconcile_phase`)


# --- CW-B20: accounting records of the PRIVATE event-loop step ------------------------------------------------------
# The event loop (`evolve_sediment_event`) reads only a few fields of the physics, pickup, transport and deposit
# results. Its private step (`_sediment_accounting_step`) returns these typed records instead of the complete
# results, so the complete physics result, the MAPLE pickup result and the post-pickup bed / ledger it carries are
# not retained through the deposit call, reconciliation and commit. Every field is the SAME array object the full
# result holds (no copy, no recomputation), except the two raindrop / flow class sums, which are the loop's original
# expression `.sum(axis=(0, 1))` evaluated on the same arrays. The records hold no bed, ledger or MAPLE state fields
# (no facade); the bed is the one `SedimentAccountingStep.bed` the full step would also return.
_PICKUP_ACCOUNTING_FIELDS = ("actual_removal_by_cell_class_kg", "requested_removal_by_class_kg",
                             "actual_removal_by_class_kg", "availability_shortfall_by_class_kg",
                             "shortfall_by_class_kg", "numerical_residual_by_class_kg",
                             "numerical_residual_scalar_kg")
_DEPOSIT_ACCOUNTING_FIELDS = ("deposition_by_cell_class_kg", "deposited_mass_by_class_kg",
                              "boundary_export_by_class_kg", "numerical_residual_by_class_kg",
                              "numerical_residual_scalar_kg")
_TRANSPORT_ACCOUNTING_FIELDS = ("export_request_kg", "deposition_request_by_class_kg", "export_request_by_class_kg",
                                "budget_residual_by_class_kg", "budget_tolerance_by_class_kg", "max_courant",
                                "max_decay_exponent", "max_cell_balance_residual_kg")


@dataclass(frozen=True, eq=False)
class PhysicsAccounting:
    """The physics fields the event loop reads: the new velocity memory, the raindrop / flow requested pickup by
    class (`raindrop_pickup_kg.sum(axis=(0, 1))`, `flow_pickup_kg.sum(axis=(0, 1))`) and the regime counts."""

    sediment_velocity_m_s: Any
    raindrop_pickup_by_class_kg: Any
    flow_pickup_by_class_kg: Any
    regime_counts: dict[str, Any]

    @classmethod
    def of(cls, physics: SedimentPhysicsStep) -> PhysicsAccounting:
        return cls(sediment_velocity_m_s=physics.sediment_velocity_m_s,
                   raindrop_pickup_by_class_kg=physics.raindrop_pickup_kg.sum(axis=(0, 1)),
                   flow_pickup_by_class_kg=physics.flow_pickup_kg.sum(axis=(0, 1)),
                   regime_counts=physics.regime_counts)


@dataclass(frozen=True, eq=False)
class PickupAccounting:
    """The MAPLE pickup-result fields the loop reads (same objects): the actual removal per cell/class (also the
    transport kernel's injected packet) and the class / scalar reductions MAPLE computed."""

    actual_removal_by_cell_class_kg: Any
    requested_removal_by_class_kg: Any
    actual_removal_by_class_kg: Any
    availability_shortfall_by_class_kg: Any
    shortfall_by_class_kg: Any
    numerical_residual_by_class_kg: Any
    numerical_residual_scalar_kg: Any

    @classmethod
    def of(cls, result: Any) -> PickupAccounting:
        return cls(**{name: getattr(result, name) for name in _PICKUP_ACCOUNTING_FIELDS})


@dataclass(frozen=True, eq=False)
class DepositAccounting:
    """The MAPLE deposit/export-result fields the loop reads (same objects)."""

    deposition_by_cell_class_kg: Any
    deposited_mass_by_class_kg: Any
    boundary_export_by_class_kg: Any
    numerical_residual_by_class_kg: Any
    numerical_residual_scalar_kg: Any

    @classmethod
    def of(cls, result: Any) -> DepositAccounting:
        return cls(**{name: getattr(result, name) for name in _DEPOSIT_ACCOUNTING_FIELDS})


@dataclass(frozen=True, eq=False)
class TransportAccounting:
    """The transport-step fields the loop reads (same objects)."""

    export_request_kg: Any
    deposition_request_by_class_kg: Any
    export_request_by_class_kg: Any
    budget_residual_by_class_kg: Any
    budget_tolerance_by_class_kg: Any
    max_courant: Any
    max_decay_exponent: Any
    max_cell_balance_residual_kg: Any

    @classmethod
    def of(cls, transport: TransportStep) -> TransportAccounting:
        return cls(**{name: getattr(transport, name) for name in _TRANSPORT_ACCOUNTING_FIELDS})


@dataclass(frozen=True, eq=False)
class SedimentAccountingStep:
    """One accepted (not yet committed / published) attempt of the PRIVATE event-loop step: the same storm, bed,
    phase and reconciliation objects as `SedimentCoupledStep`, with the four accounting records in place of the
    complete physics / pickup / transport / deposit results. Never returned by the public step."""

    dt_s: float
    storm: CoupledStep
    physics: PhysicsAccounting
    pickup: PickupAccounting
    transport: TransportAccounting
    deposit: DepositAccounting
    bed: BedState
    n_substeps: int
    n_transport_rejections: int
    phase: Any = None
    phase_reconciliation: dict[str, Any] | None = None


_STEP_RETENTION = ("full", "accounting")


@dataclass(frozen=True, eq=False)
class _PhaseReconcileInputs:
    """CW-B65 PRIVATE: exactly the objects reconciliation reads from a kernel result (the SAME arrays, no copy): the
    transport's post-transfer pool and deposition / export requests, the kernel's remaining pool and its phase. The
    accounting step holds this record instead of the complete `SparseCharacteristicStep` / `CharacteristicStep`, so
    the unused transport diagnostics can be released before the deposit call. Never exported or returned."""

    sparse: bool
    mobile_after_transfer_kg: Any
    deposition_request_kg: Any
    export_request_kg: Any
    mobile_remaining_kg: Any
    phase_after: Any

    @classmethod
    def of(cls, step: Any) -> _PhaseReconcileInputs:
        transport = step.transport
        return cls(sparse=isinstance(step, (SparseCharacteristicStep, _SparseAccountingStep)),
                   mobile_after_transfer_kg=transport.mobile_after_transfer_kg,
                   deposition_request_kg=transport.deposition_request_kg,
                   export_request_kg=transport.export_request_kg,
                   mobile_remaining_kg=step.mobile_remaining_kg, phase_after=step.phase_after)


# --- CW-B71: the post-pickup bed WITHOUT its water (accounting characteristic modes only) ------------------------------
_BED_WATER_FIELD = "water"


@dataclass(frozen=True, eq=False)
class _PostPickupBed:
    """CW-B71 PRIVATE: every `BedState` field EXCEPT `water`, the SAME objects as the actual post-pickup bed MAPLE
    returned (no copy): voxel column with its unpublished pending journal, active layer, ledger, committed topography,
    availability, and any field a future `BedState` adds (carried by name from `dataclasses.fields(BedState)`, never
    silently dropped; the rebuild refuses a field-set change). It holds NO water array: the characteristic transports
    read the PRE-pickup accepted pool and the actual MAPLE pickup, and the post-pickup MAPLE water is replaced
    wholesale by `(route depth, T)` before the deposit call, so the accounting core keeps only this record during the
    transport. Never exported, returned or published; dies with the core frame."""

    fields: tuple  # ((name, value), ...) in `BedState` field order, `water` excluded

    @classmethod
    def of(cls, bed: BedState) -> _PostPickupBed:
        if type(bed) is not BedState:
            raise SedimentEventError("the post-pickup record needs the actual BedState returned by the pickup call")
        names = tuple(f.name for f in dataclasses.fields(BedState))
        if _BED_WATER_FIELD not in names:
            raise SedimentEventError("BedState has no water field; the post-pickup record cannot be built")
        return cls(tuple((name, getattr(bed, name)) for name in names if name != _BED_WATER_FIELD))


# CW-B76: the exact `BedState` field table the paired transaction rebuilds (apply_bed_demand's `replace`).
_PAIRED_BED_FIELDS = ("voxel_column", "active_layer", "water", "ledger", "committed_topography",
                      "sediment_availability")


def _postpickup_release_eligible(bed: Any) -> bool:
    """CW-B71 correction 2: the cheap, explicit eligibility rule for the post-pickup water release. ONLY the exact
    standard `BedState` whose dataclass shape is plain -- `dataclasses.fields(BedState)` covers every entry of
    `__dataclass_fields__` (so no ClassVar / InitVar pseudo-fields) and every field is `init=True` -- is optimized
    through `_PostPickupBed` / `_bed_with_transport_water`. Any other bed (a valid subclass, an extended dataclass
    subtype, an unusual shape) keeps the ORIGINAL bed and the ORIGINAL `with_water` rebuild (the path upwind and the
    public full form use), so its constructor, type, namespace walk and error order are exactly the original ones."""
    if type(bed) is not BedState:
        return False
    fields = dataclasses.fields(BedState)
    return len(fields) == len(BedState.__dataclass_fields__) and all(f.init for f in fields)


def _bed_with_transport_water(post: _PostPickupBed, context: BedContext, depth_m: Any, mobile_kg: Any) -> BedState:
    """CW-B71 PRIVATE: `with_water(post_pickup_bed, context, depth_m, mobile_kg)` (sediment_bed.py) on the record --
    the SAME operations in the same order: build the new `WaterState(depth_m, mobile_kg)`, resolve the operand namespace
    of the bed fields, the new water and the initial available fraction, `validate_water_state` the new water, then the
    `BedState` that `replace(bed, water=water)` builds (the same field objects, the new water). The only difference:
    the namespace resolution visits the record's non-water fields and the NEW water, not the discarded post-pickup
    water arrays (outputs of the actual MAPLE pickup call, already validated by MAPLE's own result validation).
    Observer seam: an observer wrapping `with_water` sees this rebuild under this name instead."""
    from maple.core.backend import resolve_operand_namespace
    from maple.core.types.water import WaterState
    from maple.water import validate_water_state

    names = tuple(f.name for f in dataclasses.fields(BedState))
    values = dict(post.fields)
    if tuple(n for n in names if n != _BED_WATER_FIELD) != tuple(values):
        raise SedimentEventError("BedState fields changed since the post-pickup record was built")
    water = WaterState(depth_m, mobile_kg)
    resolve_operand_namespace(tuple(values.values()), water, context.initial_available_fraction)
    validate_water_state(water, context.geometry.ny, context.geometry.nx, len(context.grain_classes.classes))
    return BedState(**{name: (water if name == _BED_WATER_FIELD else values[name]) for name in names})


def _reconciliation_bound_check(inputs: _PhaseReconcileInputs, actual: Any, xp: ModuleType):
    """The shared refusal of `reconcile_phase` for both storages: `|actual - predicted|` within the MAPLE
    summation bound for `PHASE_RECONCILIATION_OPERATIONS` roundings. Returns `(predicted, residual)`.
    (CW-B65: reads the private inputs record; the same arrays and the same operations as before.)"""
    from maple.core.backend import DeferredChecks, true_flag
    from maple.surface.voxels._numerics import summation_error_bound_kg

    predicted = inputs.mobile_remaining_kg
    scale = xp.maximum(xp.maximum(inputs.mobile_after_transfer_kg, inputs.deposition_request_kg),
                       xp.maximum(xp.maximum(inputs.export_request_kg, predicted), actual))
    bound = summation_error_bound_kg(PHASE_RECONCILIATION_OPERATIONS, scale)
    residual = actual - predicted
    checks = DeferredChecks()
    checks.forbid(true_flag(xp.abs(residual) > bound),
                  "actual MAPLE mobile pool after the deposit/export call differs from the transport kernel's "
                  "remaining pool beyond the declared FP64 bound (a refused or altered request); nothing published")
    try:
        checks.resolve()
    except ValueError as exc:
        raise SedimentEventError(str(exc)) from None
    return predicted, residual


def _reconcile_sparse_phase(inputs: _PhaseReconcileInputs, actual_mobile_kg: Any,
                            xp: ModuleType) -> tuple[SparsePhaseState, dict[str, Any]]:
    """CSR counterpart of the dense reconciliation, the SAME bound and the same two rules on ROWS: a cell/class
    whose ACTUAL pool is exactly 0 loses its stored bins (the row becomes empty, `nnz` decreases); a cell/class
    whose actual pool is positive while the kernel's remaining pool is 0 (a rounding remnant, within the bound by
    the check above) receives the canonical bin 0 at the upstream face and is COUNTED. Rows that need neither keep
    their bins and fractions unchanged (MAPLE's mass is authoritative, fractions are a distribution). The two counts
    are read ONCE (one counted host read); the row surgery runs only when a count is nonzero."""
    from maple.core.backend import to_host

    predicted, residual = _reconciliation_bound_check(inputs, actual_mobile_kg, xp)
    actual = actual_mobile_kg
    phase = inputs.phase_after
    canon = (actual == 0.0) & (predicted > 0.0)
    remnant = (actual > 0.0) & (predicted == 0.0)
    counts = np.asarray(to_host(xp.stack([xp.sum(canon), xp.sum(remnant)])), dtype=np.int64)
    n_canon, n_remnant = int(counts[0]), int(counts[1])
    if n_canon or n_remnant:
        n = int(actual.shape[0]) * int(actual.shape[1])
        nc = int(actual.shape[2])
        rows_canon = canon.reshape(n, nc)[phase.cell_of_slot].reshape(-1)
        rows_remnant = remnant.reshape(n, nc)[phase.cell_of_slot].reshape(-1)
        phase = reconcile_sparse_rows(phase, rows_canon, rows_remnant)
    diagnostics = {
        "residual_max_kg": xp.max(xp.abs(residual)),
        "n_canonicalized": np.int64(n_canon),
        "n_rounding_remnants": np.int64(n_remnant),
    }
    return phase, diagnostics


def reconcile_phase(step: Any, actual_mobile_kg: Any, xp: ModuleType) -> tuple[Any, dict[str, Any]]:
    """Bind the kernel's phase to the ACTUAL MAPLE pool after the
    deposit/export call. Dispatches on the typed kernel result: a dense
    `CharacteristicStep` keeps the accepted dense arithmetic below; a
    `SparseCharacteristicStep` goes through `_reconcile_sparse_phase` (same
    bound, same rules on rows, no dense array).

    The kernel predicted `M_pred = T - Dep - E` per cell/class; MAPLE applied
    the same requests and published `M_actual`. Both are the same arithmetic
    up to round-off, so `|M_actual - M_pred|` must lie within the MAPLE
    summation bound for `PHASE_RECONCILIATION_OPERATIONS` roundings at the
    largest operand of that cell/class; a larger difference (a refused or
    altered request) is refused before anything is published. Within the
    bound the fractions remain the partition (MAPLE's mass is authoritative,
    fractions are a distribution) except that a cell whose ACTUAL pool is
    exactly 0 is canonicalised (a tiny predicted remnant with no actual mass
    keeps no position), and a cell whose actual pool is positive while the
    kernel's is 0 keeps the kernel's canonical partition (all mass at the
    upstream face) and is COUNTED, never silently invented. Returns the
    reconciled phase and 0-d diagnostics: `residual_max_kg`,
    `n_canonicalized`, `n_rounding_remnants`."""
    if isinstance(step, SparseCharacteristicStep):
        return _reconcile_sparse_phase(_PhaseReconcileInputs.of(step), actual_mobile_kg, xp)
    if not isinstance(step, CharacteristicStep):
        raise SedimentEventError("reconcile_phase needs a CharacteristicStep or a SparseCharacteristicStep")
    return _reconcile_dense_phase(_PhaseReconcileInputs.of(step), actual_mobile_kg, xp)


def _reconcile_phase_inputs(inputs: _PhaseReconcileInputs, actual_mobile_kg: Any,
                            xp: ModuleType) -> tuple[Any, dict[str, Any]]:
    """CW-B65 PRIVATE: the body of `reconcile_phase` on an already-captured inputs record (the accounting step, after
    it released the complete kernel result). Same dispatch outcome, bound, rules and diagnostics."""
    if inputs.sparse:
        return _reconcile_sparse_phase(inputs, actual_mobile_kg, xp)
    return _reconcile_dense_phase(inputs, actual_mobile_kg, xp)


def _reconcile_dense_phase(inputs: _PhaseReconcileInputs, actual_mobile_kg: Any,
                           xp: ModuleType) -> tuple[PhaseState, dict[str, Any]]:
    """The dense reconciliation of `reconcile_phase` (unchanged operations), on the private inputs record."""
    from maple.core.backend import freeze

    predicted, residual = _reconciliation_bound_check(inputs, actual_mobile_kg, xp)
    actual = actual_mobile_kg
    phase = inputs.phase_after
    zero_actual = actual == 0.0
    canonical = xp.zeros((1, 1, 1, phase.n_bins), dtype=np.float64)
    canonical[..., 0] = 1.0
    fraction = xp.where(zero_actual[..., None], canonical, phase.fraction)
    position = xp.where(zero_actual[..., None], 0.0, phase.position_m)
    diagnostics = {
        "residual_max_kg": xp.max(xp.abs(residual)),
        "n_canonicalized": xp.sum(zero_actual & (predicted > 0.0)),
        "n_rounding_remnants": xp.sum((actual > 0.0) & (predicted == 0.0)),
    }
    reconciled = PhaseState(fraction=freeze(fraction), position_m=freeze(position), dx_m=phase.dx_m,
                            n_bins=phase.n_bins, xp=phase.xp)
    return reconciled, diagnostics


def sediment_coupled_step(
    state: SedimentEventState,
    context: BedContext,
    column: ColumnParameters,
    rain_rate_m_per_s: Any,
    vegetation_cover_fraction: Any,
    sediment: SedimentPhysicsParameters,
    dt_s: float,
    control: SedimentEventControl,
    *,
    zero_demand: Any = None,
    hydraulics: CudaHydraulics | None = None,
) -> SedimentCoupledStep:
    """Steps 1-5 of the module docstring on the ORIGINAL `state`. Pure:
    every MAPLE call returns new objects and nothing is published; raises
    `RoutingStepRejected` / `TransportStepRejected` when the attempt must
    be retried with a smaller `dt`, and propagates every other failure.

    `hydraulics` (CUDA control only) is the caller's `CudaHydraulics`
    lifecycle holding the prepared context of `state.graph` / `column`;
    without it a CUDA control prepares a context for this call alone
    (convenience form, one counted static transfer per invocation).

    Returns the COMPLETE `SedimentCoupledStep` (every physics, pickup,
    transport, deposit and kernel result), fully inspectable. The event
    loop uses the private accounting form of the same core (CW-B20)."""
    return _sediment_coupled_core(state, context, column, rain_rate_m_per_s, vegetation_cover_fraction, sediment,
                                  dt_s, control, zero_demand=zero_demand, hydraulics=hydraulics, retain="full")


def _sediment_accounting_step(
    state: SedimentEventState,
    context: BedContext,
    column: ColumnParameters,
    rain_rate_m_per_s: Any,
    vegetation_cover_fraction: Any,
    sediment: SedimentPhysicsParameters,
    dt_s: float,
    control: SedimentEventControl,
    *,
    zero_demand: Any = None,
    hydraulics: CudaHydraulics | None = None,
) -> SedimentAccountingStep:
    """PRIVATE event-loop form of `sediment_coupled_step` (CW-B20): the same core, the same kernel calls, the same
    arithmetic in the same order and every validation, returning `SedimentAccountingStep`. Only object lifetimes
    differ: the complete physics result is reduced to what pickup, transport and the loop need before the pickup
    call; the MAPLE pickup result is reduced to its accounting fields right after it returns; the post-pickup bed
    is released when the deposit call returns; the transport and kernel results are released after
    reconciliation."""
    return _sediment_coupled_core(state, context, column, rain_rate_m_per_s, vegetation_cover_fraction, sediment,
                                  dt_s, control, zero_demand=zero_demand, hydraulics=hydraulics,
                                  retain="accounting")


def _transport_with_substep_retry(state: SedimentEventState, control: SedimentEventControl, velocity: Any,
                                  deposition_rate: Any, settle_mask: Any, actual_pickup: Any, dt: float,
                                  xp: ModuleType, upwind_bed: Any, *,
                                  retain: str = "full") -> tuple[Any, Any, int, int]:
    """CW-B76: step 4 of `_sediment_coupled_core` (B73 lines 1353-1388) moved VERBATIM so the two-call path and the
    paired MAPLE transaction run the same operator, Courant substep count, retry, cap and messages. `upwind_bed` is
    the post-pickup bed the upwind branch reads (`bed` in the original); the characteristic branches never read it.
    Returns `(transport, characteristic, n_sub, rejections)`.

    CW-B93: `retain="accounting"` (passed ONLY by the paired MAPLE transaction callback) selects, for the CUDA CSR
    operator only, the private accounting retention of the SAME step (`_SparseAccountingStep` /
    `_TransportAccountingStep`); every other operator, and the default, are unchanged."""
    if retain not in _STEP_RETENTION:
        raise SedimentEventError(f"retain must be one of {_STEP_RETENTION}, got {retain!r}")
    from maple.core.backend import to_float

    v_max = to_float(xp.max(velocity))
    dx = state.network.dx_m
    n_sub = max(1, math.ceil(v_max * dt / (dx * control.sediment_courant_max)))
    rejections = 0
    characteristic = None
    while True:
        if n_sub > control.max_transport_substeps:
            raise TransportStepRejected(
                f"sediment Courant number needs {n_sub} substeps (max_transport_substeps = "
                f"{control.max_transport_substeps}) at dt = {dt} s; step rejected (retry with a smaller dt)")
        try:
            if control.csr:
                # Sparse CSR storage: the accepted CSR operator (numba on the host, cuda on the device) on the
                # PRE-pickup pool and its CSR phase, ACTUAL MAPLE pickup as the new packet; same contract.
                implementation = control.resolved_transport_implementation()
                if retain == "accounting" and implementation == "cuda":
                    # CW-B93: the same CUDA CSR core, private accounting retention (same arguments, cap, messages)
                    characteristic = _sparse_characteristic_accounting_step(
                        state.network, state.bed.water.mobile_mass_by_cell_class_kg, state.phase,
                        actual_pickup, velocity, deposition_rate,
                        settle_mask, dt, n_substeps=n_sub, courant_max=control.sediment_courant_max,
                        max_nnz=control.phase_max_nnz)
                else:
                    characteristic = sparse_characteristic_step(
                        state.network, state.bed.water.mobile_mass_by_cell_class_kg, state.phase,
                        actual_pickup, velocity, deposition_rate,
                        settle_mask, dt, n_substeps=n_sub, courant_max=control.sediment_courant_max,
                        implementation=implementation, max_nnz=control.phase_max_nnz)
                transport = characteristic.transport
            elif control.characteristic:
                # PRE-pickup pool and its phase, ACTUAL MAPLE pickup as the new packet.
                characteristic = characteristic_step(
                    state.network, state.bed.water.mobile_mass_by_cell_class_kg, state.phase,
                    actual_pickup, velocity, deposition_rate,
                    settle_mask, dt, n_substeps=n_sub, courant_max=control.sediment_courant_max,
                    implementation=control.resolved_transport_implementation())
                transport = characteristic.transport
            else:
                transport = transport_step(state.network, upwind_bed.water.mobile_mass_by_cell_class_kg, velocity,
                                           deposition_rate, settle_mask, dt,
                                           courant_max=control.sediment_courant_max, n_substeps=n_sub)
            break
        except TransportStepRejected:
            rejections += 1
            n_sub *= 2
    return transport, characteristic, n_sub, rejections


def _sediment_coupled_core(
    state: SedimentEventState,
    context: BedContext,
    column: ColumnParameters,
    rain_rate_m_per_s: Any,
    vegetation_cover_fraction: Any,
    sediment: SedimentPhysicsParameters,
    dt_s: float,
    control: SedimentEventControl,
    *,
    zero_demand: Any,
    hydraulics: CudaHydraulics | None,
    retain: str,
) -> SedimentCoupledStep | SedimentAccountingStep:
    """The ONE implementation of steps 1-5 (see `sediment_coupled_step`). `retain="full"` returns the complete
    `SedimentCoupledStep`; `retain="accounting"` returns `SedimentAccountingStep` and drops each complete result
    after its last use (lifetimes only; no value, order, check or kernel differs)."""
    # CW-B76: `to_float` moved with step 4 (its only use) to `_transport_with_substep_retry`
    from maple.water import WaterProcessDemand

    if retain not in _STEP_RETENTION:
        raise SedimentEventError(f"retain must be one of {_STEP_RETENTION}, got {retain!r}")
    full = retain == "full"
    if not isinstance(state, SedimentEventState):
        raise SedimentEventError("state must be a SedimentEventState")
    # Public entry: validate the control, its namespace binding and its binding
    # to the state's phase (scheme, presence, bin count, cell size) BEFORE any
    # hydraulic or MAPLE work; a direct caller cannot run with a mismatched or
    # invalid control.
    control = _bind_phase_to_control(state, control)
    if hydraulics is not None and not isinstance(hydraulics, CudaHydraulics):
        raise SedimentEventError("hydraulics must be a CudaHydraulics lifecycle (or None)")
    if hydraulics is not None and not control.cuda:
        raise SedimentEventError("hydraulics is only meaningful with storm implementation 'cuda'")
    xp = state.graph.xp
    nc = sediment.n_classes
    dt = float(dt_s)
    zeros = xp.zeros((*state.graph.shape, nc), dtype=np.float64) if zero_demand is None else zero_demand

    # 1. water: column + routing (accepted Phase 4 kernels, or the prepared CUDA hydrology).
    if control.cuda:
        storm = _cuda_hydraulic_step(state, column, rain_rate_m_per_s, dt, control, hydraulics)
    else:
        storm = coupled_step(state.graph, column, rain_rate_m_per_s, state.storm, dt, control.storm)
    route = storm.route
    # 2. wet laws on the routed depth / velocity and the CURRENT holdings.
    physics = sediment_physics_step(sediment, state.grid, route.depth_m, route.velocity_m_s, rain_rate_m_per_s,
                                    vegetation_cover_fraction, state.bed.active_layer.mass_kg,
                                    state.sediment_velocity_m_s, dt)
    # Operands of the later steps (the same objects). The accounting form keeps only these and the loop's
    # accounting record; the complete physics result is released BEFORE the pickup call.
    requested_pickup = physics.requested_pickup_kg
    velocity = physics.sediment_velocity_m_s
    deposition_rate = physics.deposition_rate_per_m
    settle_mask = physics.settle_mask
    kept_physics = physics if full else PhysicsAccounting.of(physics)
    del physics
    # CW-B76 (accounting form, windowed bed, characteristic CSR, exact plain `BedState` only): the pickup, the
    # transport and the deposit run through MAPLE's PRIVATE paired windowed transaction, which keeps the post-pickup
    # MAPLE state private and adopts its storage for the deposit. Every other combination (full form, upwind, dense,
    # non-windowed, subclass/extended bed, a MAPLE without the module) takes the unchanged two-call path below.
    paired_entry = None
    if (not full and control.csr and is_windowed_bed(state.bed) and _postpickup_release_eligible(state.bed)
            and tuple(f.name for f in dataclasses.fields(BedState)) == _PAIRED_BED_FIELDS):
        paired_entry = paired_transaction_entry()
    if paired_entry is not None:
        from maple.core.backend import resolve_operand_namespace
        from maple.core.types.water import WaterState
        from maple.water import validate_water_state

        bed0 = with_water(state.bed, context, route.depth_m)  # line "3." below, identical
        holder = [WaterProcessDemand(requested_pickup, zeros)]  # single-use handoff, popped by MAPLE
        del requested_pickup
        operands = {"deposition_rate": deposition_rate, "settle_mask": settle_mask}
        del deposition_rate, settle_mask
        captured = []

        def between(view):
            # step 4 on the PRE-pickup pool and the ACTUAL pickup (the same operator, substeps, retry and cap)
            transport, characteristic, n_sub_, rejections_ = _transport_with_substep_retry(
                state, control, velocity, operands["deposition_rate"], operands["settle_mask"],
                view.actual_removal_by_cell_class_kg, dt, xp, None, retain="accounting")
            operands.clear()  # the rate and mask die at their original last use
            mobile = transport.mobile_after_transfer_kg
            # the deposit water's checks in the original order: namespace walk first (as `with_water` /
            # `_bed_with_transport_water`), then the value check; MAPLE then resolves and validates it again over
            # its private post-pickup state (as `apply_bed_demand`'s check did), so no value check is lost
            water = WaterState(route.depth_m, mobile)
            resolve_operand_namespace(bed0, water, context.initial_available_fraction)
            validate_water_state(water, context.geometry.ny, context.geometry.nx,
                                 len(context.grain_classes.classes))
            del water
            # all four faces, to MAPLE's validation (CW-B93: the private CUDA CSR record through the same constructor)
            demand = (_water_demand_from_accounting_transport(transport, zeros)
                      if isinstance(transport, _TransportAccountingStep)
                      else water_demand_from_transport(transport, zeros))
            captured.append((TransportAccounting.of(transport),
                             None if characteristic is None else _PhaseReconcileInputs.of(characteristic),
                             n_sub_, rejections_))
            del transport, characteristic
            return route.depth_m, mobile, demand

        view, bed, deposit = apply_bed_pickup_transport_deposit(bed0, context, holder, between, entry=paired_entry)
        del bed0, holder, between, operands
        kept_pickup = PickupAccounting(**{name: getattr(view, name) for name in _PICKUP_ACCOUNTING_FIELDS})
        del view
        kept_transport, reconcile_inputs, n_sub, rejections = captured.pop()
        kept_deposit = DepositAccounting.of(deposit)
        del deposit
        phase = reconciliation = None
        if reconcile_inputs is not None:
            # 5b. bind the phase to the ACTUAL pool MAPLE published (refuses beyond the FP bound).
            phase, reconciliation = _reconcile_phase_inputs(reconcile_inputs, bed.water.mobile_mass_by_cell_class_kg, xp)
        del reconcile_inputs
        return SedimentAccountingStep(dt_s=dt, storm=storm, physics=kept_physics, pickup=kept_pickup,
                                      transport=kept_transport, deposit=kept_deposit,
                                      bed=bed, n_substeps=n_sub, n_transport_rejections=rejections, phase=phase,
                                      phase_reconciliation=reconciliation)
    # 3. publish the solver depth; actual pickup through MAPLE.
    bed = with_water(state.bed, context, route.depth_m)
    bed, pickup = apply_bed_demand(bed, context, WaterProcessDemand(requested_pickup, zeros))
    del requested_pickup
    actual_pickup = pickup.actual_removal_by_cell_class_kg
    # The accounting form keeps the pickup accounting fields only: the MAPLE pickup result is released here, so the
    # post-pickup bed / ledger live only through `bed`, which the deposit call below replaces.
    kept_pickup = pickup if full else PickupAccounting.of(pickup)
    del pickup
    # CW-B71 (accounting form, characteristic modes -- CSR and dense -- only): the transports below read the PRE-pickup
    # accepted pool `state.bed.water` and the actual MAPLE pickup, never the post-pickup MAPLE water, which is replaced
    # wholesale before the deposit call. Keep the post-pickup bed's NON-water fields (same objects) and release the
    # local bed, so the post-pickup mobile pool is not held through the transport. Upwind (reads the post-pickup pool)
    # and the public full form (keeps the complete pickup result) are unchanged.
    # Correction 2: only the exact plain `BedState` (cheap eligibility check BEFORE any record capture); every other
    # bed keeps the original path below.
    post_pickup = None
    if not full and (control.csr or control.characteristic) and _postpickup_release_eligible(bed):
        post_pickup = _PostPickupBed.of(bed)
        del bed
    # 4. transport of the ACTUAL post-pickup pool; substeps from the sediment Courant number.
    # CW-B76: the original statements live in `_transport_with_substep_retry` (shared with the paired callback);
    # `bed` is passed only where the original upwind branch could read it (it is not bound after the B71 release).
    transport, characteristic, n_sub, rejections = _transport_with_substep_retry(
        state, control, velocity, deposition_rate, settle_mask, actual_pickup, dt, xp,
        None if post_pickup is not None else bed)
    del deposition_rate, settle_mask, actual_pickup  # transport consumed them (the records keep their own refs)
    # 5. publish T(M); actual deposition / export through MAPLE. Rebinding `bed` releases the post-pickup bed when
    #    the deposit call returns (nothing else holds it in the accounting form).
    if full:
        bed = with_water(bed, context, route.depth_m, transport.mobile_after_transfer_kg)
        bed, deposit = apply_bed_demand(bed, context, water_demand_from_transport(transport, zeros))
        phase = reconciliation = None
        if characteristic is not None:
            # 5b. bind the phase to the ACTUAL pool MAPLE published (refuses beyond the FP bound).
            phase, reconciliation = reconcile_phase(characteristic, bed.water.mobile_mass_by_cell_class_kg, xp)
        return SedimentCoupledStep(dt_s=dt, storm=storm, physics=kept_physics, pickup=kept_pickup,
                                   transport=transport, deposit=deposit, bed=bed, n_substeps=n_sub,
                                   n_transport_rejections=rejections, characteristic=characteristic, phase=phase,
                                   phase_reconciliation=reconciliation)
    # CW-B65 (accounting form only; measured B64 refusal inside the deposit call): the SAME calls in the same order,
    # but the loop's accounting records, the deposit demand (which carries all four faces to MAPLE's validation) and
    # the reconciliation inputs are captured FIRST (the same objects the return-time records would read), then the
    # complete kernel result and transport step are released before the deposit call; the complete deposit result
    # and the demand (faces) are released before reconciliation, which runs on the private inputs record.
    if post_pickup is None:  # upwind accounting, or a bed not eligible for the release: unchanged original path
        bed = with_water(bed, context, route.depth_m, transport.mobile_after_transfer_kg)
    else:  # CW-B71: the same validated rebuild on the post-pickup non-water record; the record dies here
        bed = _bed_with_transport_water(post_pickup, context, route.depth_m, transport.mobile_after_transfer_kg)
        del post_pickup
    demand = water_demand_from_transport(transport, zeros)
    kept_transport = TransportAccounting.of(transport)
    reconcile_inputs = None if characteristic is None else _PhaseReconcileInputs.of(characteristic)
    del transport, characteristic
    bed, deposit = apply_bed_demand(bed, context, demand)
    kept_deposit = DepositAccounting.of(deposit)
    del deposit, demand
    phase = reconciliation = None
    if reconcile_inputs is not None:
        # 5b. bind the phase to the ACTUAL pool MAPLE published (refuses beyond the FP bound).
        phase, reconciliation = _reconcile_phase_inputs(reconcile_inputs, bed.water.mobile_mass_by_cell_class_kg, xp)
    del reconcile_inputs
    return SedimentAccountingStep(dt_s=dt, storm=storm, physics=kept_physics, pickup=kept_pickup,
                                  transport=kept_transport, deposit=kept_deposit,
                                  bed=bed, n_substeps=n_sub, n_transport_rejections=rejections, phase=phase,
                                  phase_reconciliation=reconciliation)


# --- commit and reroute ------------------------------------------------------------------------------
#: CW-B104 reason of an actual MAPLE commit: MAPLE's own triggers fired; the net-elevation displacement from the held
#: routing source reached the threshold between MAPLE triggers (a genuine extra commit made only so the routing can be
#: refreshed from authoritative committed geometry); the caller's forced final commit.
COMMIT_REASONS = ("maple_trigger", "net_elevation", "final_forced")
#: CW-B104 hydraulic routing-refresh contract (run / checkpoint identity): a source written under any other contract
#: (e.g. the C1 refresh at every commit) cannot claim compatibility with it.
ROUTING_REFRESH_CONTRACT = "cw-b104/net-elevation-only/1"


def routing_refresh_record(context: BedContext) -> dict[str, Any]:
    """CW-B104 host record of the routing-refresh contract and its actual threshold (for identities / summaries)."""
    return {"contract": ROUTING_REFRESH_CONTRACT, "trigger": "net per-cell elevation displacement only",
            "measure": "max over active cells of |exact committed surface - held routing source| (within-cell gains "
                       "and losses cancel, different cells never cancel)",
            "threshold_m": reroute_threshold_m(context), "threshold_source": "MAPLE pending_elevation_trigger_m",
            "comparison": ">=",
            "never_reroutes": "elapsed time, gross turnover, report / checkpoint / publication, ordinary or final "
                              "forced commits on their own"}


@dataclass(frozen=True, eq=False)
class _Commit:
    bed: BedState
    graph: RoutingGraph
    network: TransportNetwork | None  # None when the commit did not reroute (the caller keeps its own)
    grid: PhysicsGrid | None
    storm: StormState
    record: dict[str, Any]
    maple_result: Any
    rerouted: bool = True
    routing_source_m: Any = None  # the (possibly new) held routing source


def routing_gate_m(bed: BedState, context: BedContext, graph: RoutingGraph, routing_source_m: Any) -> float:
    """CW-B104 per-accepted-step net-elevation gate: max over the graph's active cells of |exact current committed
    surface - held routing source| (m), from `sediment_bed.current_routing_surface_m` (MAPLE's exact height API with
    the commit's own arithmetic). Its `(ny, nx)` temporaries are released before returning; ONE counted scalar read."""
    surface = current_routing_surface_m(bed, context)
    return net_routing_displacement_m(surface, routing_source_m, graph.active_flat.reshape(graph.shape))


def _commit_and_reroute(bed: BedState, context: BedContext, graph: RoutingGraph, terrain: TerrainReference,
                        storm: StormState, t_s: float, *, force: bool, routing_source_m: Any,
                        net_due: bool = False, gate_m: float | None = None) -> _Commit | None:
    """MAPLE commit through `sediment_bed.commit_bed` (forced by the caller's final policy `force`, or by the
    net-elevation gate `net_due`); on a commit verify the depth policy and the water volume. CW-B104: the routing is
    refreshed only when `commit_bed` reports the post-commit net displacement from `routing_source_m` reaching the
    threshold; then network / grid are rebuilt from the refreshed graph and the discharge re-initialised from the
    new conveyance and the unchanged depth. A non-rerouting commit returns the caller's graph and storm (discharge)
    unchanged and no network / grid. Nothing is published here."""
    from maple.core.backend import to_float, to_host, to_int

    def xp_any(array, axis):
        return graph.xp.any(array, axis=axis)

    outcome = commit_bed(bed, context, graph, terrain, t_s, force=bool(force or net_due),
                         routing_source_m=routing_source_m)
    if not outcome.did_commit:
        return None
    depth_update = None if outcome.water_result is None else outcome.water_result.depth_update
    if (depth_update is None or depth_update.rule != "constant_depth" or depth_update.displaced_volume_m3 != 0.0
            or depth_update.clamp_added_volume_m3 != 0.0 or depth_update.clamp_cell_count != 0):
        raise SedimentEventError(f"commit at t = {t_s} s changed the water volume or depth policy: {depth_update}")
    result = outcome.maple_result
    trigger = outcome.trigger
    reason = "final_forced" if force else ("maple_trigger" if trigger.should_commit else "net_elevation")
    record = {
        "t_s": t_s,
        "forced": bool(force),
        "commit_count": int(outcome.state.committed_topography.commit_count),
        "elapsed_time_triggered": bool(trigger.elapsed_time_triggered),
        "pending_elevation_triggered": bool(trigger.pending_elevation_triggered),
        "active_layer_turnover_triggered": bool(trigger.active_layer_turnover_triggered),
        "max_pending_elevation_change_m": float(trigger.max_pending_elevation_change_m),
        "max_abs_committed_elevation_change_m": to_float(graph.xp.max(graph.xp.abs(result.committed_elevation_change_m))),
    }
    tail = {
        "avalanche_applied": bool(result.avalanche_applied),
        "depth_unchanged": True,
        "displaced_water_volume_m3": float(depth_update.displaced_volume_m3),
        "pre_commit_pending_mass_by_class_kg": [float(v) for v in
                                                to_host(result.pre_commit_pending_mass_by_class_kg).tolist()],
        "commit_stage_log": list(result.commit_stage_log),
        # CW-B73: the ACTUAL per-commit terminal-storage counts `commit_bed` already computed (host ints: before,
        # after, created, removed); carried, not recomputed. All zero under the strict (no pit storage) policy.
        "terminal_storage_change": dict(outcome.terminal_storage_change),
        # CW-B104: why MAPLE committed, whether the routing was refreshed, the gate value before the commit and the
        # post-commit displacement that decided the refresh (m), against the actual threshold (comparison >=).
        "commit_reason": reason,
        "rerouted": bool(outcome.rerouted),
        "net_displacement_before_commit_m": None if gate_m is None else float(gate_m),
        "net_displacement_m": float(outcome.net_displacement_m),
        "reroute_threshold_m": reroute_threshold_m(context),
    }
    if not outcome.rerouted:
        # Accounting / geometry publication only: the graph, its network / grid, the discharge (q), the phase binding
        # and the routing source are the caller's, unchanged; no geometry diagnostic is refreshed or fabricated.
        record.update({"rerouted_cells": 0, "phase_steered_cells": 0, "outlets_changed": 0,
                       "n_outlets": to_int(np.sum(to_host(graph.outlet))), "graph_changed": False,
                       "graph_rebound": False, "graph_input_sha256": graph.input_sha256, **tail})
        return _Commit(bed=outcome.state, graph=graph, network=None, grid=None, storm=storm, record=record,
                       maple_result=result, rerouted=False, routing_source_m=routing_source_m)
    new_graph = outcome.graph
    if new_graph.shape != graph.shape or new_graph.dx_m != graph.dx_m or new_graph.xp is not graph.xp:
        raise SedimentEventError("refreshed routing graph changed shape, spacing or namespace")
    # The graph's aspect / slope / active / outlet are its small host evidence arrays (RoutingGraph contract);
    # the explicit `to_host` keeps every comparison below on the host for any namespace without an implicit
    # conversion (a NumPy array passes through unchanged, so the host path is exactly as before).
    old_active, new_active = to_host(graph.active), to_host(new_graph.active)
    old_aspect, new_aspect = to_host(graph.aspect), to_host(new_graph.aspect)
    old_slope, new_slope = to_host(graph.slope), to_host(new_graph.slope)
    old_outlet, new_outlet = to_host(graph.outlet), to_host(new_graph.outlet)
    if not np.array_equal(new_active, old_active):
        raise SedimentEventError("refreshed routing graph changed the active cells")
    # The exporting boundary ring and the active set are fixed; WHICH interior
    # cells drain into the ring (the outlet set) may legitimately change with
    # the terrain. `build_routing_graph` has already refused pits, flats and
    # receivers outside the active set or the ring, so a valid reroute is
    # accepted and reported; the transport network is rebuilt on the new
    # outlets below and the caller refreshes its outlet-based diagnostics.
    # `graph_changed` means a PHYSICAL change (slope or receiver); a differing
    # `input_sha256` alone (`graph_rebound`) can come from the refresh omitting
    # the initial graph's nodata metadata and is only the binding.
    geometry_changed = not (np.array_equal(new_aspect, old_aspect) and np.array_equal(new_slope, old_slope))
    # Phase steering (characteristic scheme): sub-cell positions are distances
    # along the cell's CURRENT outflow direction. A rerouted cell keeps its
    # fractions and positions, i.e. the within-cell progress is carried over
    # to the new receiver direction (the same cell size); the mass is never
    # reset to the upstream face. Counted here as a documented approximation.
    # ONE counted download of the (ny, nx) "carries mobile mass" flags of the committed state.
    rerouted = new_aspect != old_aspect
    steered = rerouted & to_host(xp_any(outcome.state.water.mobile_mass_by_cell_class_kg > 0.0, axis=-1))
    # Discharge from the NEW conveyance and the UNCHANGED depth (kinematic
    # adjustment, no volume change); validates depth / soil water again.
    new_storm = initial_state(new_graph, outcome.state.water.depth_m, storm.soil_water_m, t_s=t_s)
    record.update({
        "rerouted_cells": to_int(np.sum(rerouted)),  # host reductions of the host evidence arrays
        "phase_steered_cells": to_int(np.sum(steered)),
        "outlets_changed": to_int(np.sum(new_outlet != old_outlet)),
        "n_outlets": to_int(np.sum(new_outlet)),
        "graph_changed": geometry_changed,
        "graph_rebound": new_graph.input_sha256 != graph.input_sha256,
        "graph_input_sha256": new_graph.input_sha256,
        **tail,
    })
    return _Commit(bed=outcome.state, graph=new_graph, network=transport_network(new_graph),
                   grid=physics_grid_from_graph(new_graph), storm=new_storm, record=record, maple_result=result,
                   rerouted=True, routing_source_m=outcome.routing_source_m)


# --- result ----------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SedimentEventResult:
    """Everything `evolve_sediment_event` accumulated. Grids and 0-d
    values live in the graph namespace; `hydrograph` has the Phase 4
    `HYDROGRAPH_COLUMNS`, `sediment_hydrograph` the columns of
    `sediment_hydrograph_columns(nc)`. `by_class` maps names to `(nc,)`
    device arrays; `commit_log` is bounded by `max_commit_log`."""

    state: SedimentEventState
    hydrograph: Any
    sediment_hydrograph: Any
    sediment_columns: tuple[str, ...]
    boundaries: np.ndarray
    # water
    cumulative_rain_m: Any
    cumulative_intake_m: Any
    cumulative_saturation_return_m: Any
    cumulative_drainage_m: Any
    cumulative_export_m3: Any
    peak_depth_m: Any
    peak_velocity_m_s: Any
    last_velocity_m_s: Any
    peak_outlet_discharge_m3_s: Any
    time_of_peak_outlet_s: Any
    max_courant_old: Any
    max_courant_new: Any
    max_routing_cell_balance_residual_m: Any
    max_constitutive_residual_m: Any
    cell_steps_no_runon: Any
    cell_steps_partial_runon: Any
    cell_steps_complete_runon: Any
    # sediment, per cell and class (actual MAPLE ground truth)
    cumulative_pickup_kg: Any
    cumulative_deposition_kg: Any
    cumulative_export_request_kg: Any  # nonzero only at outlets; equals actual export when nothing was refused
    by_class: dict[str, Any]
    initial_bed_by_cell_class_kg: Any  # (ny, nx, nc): voxel.sum(axis=2) + active at the start
    initial_bed_inventory_kg: Any
    initial_mobile_kg: Any
    numerical_residual_scalar_abs_kg: Any
    peak_mobile_kg: Any
    time_of_peak_mobile_s: Any
    peak_export_rate_kg_s: Any
    time_of_peak_export_s: Any
    max_sediment_courant: Any
    max_decay_exponent: Any
    max_transport_cell_residual_kg: Any
    max_transport_substeps_used: int
    n_transport_rejections: int
    regime_cell_steps: dict[str, Any]
    # commits
    ledger_process_totals_reset_kg: Any  # (n_process, nc): sum of pre-commit totals over commits
    committed_net_bed_change_kg: Any  # (nc,): sum of pre-commit pending mass over commits
    n_commits: int
    n_forced_commits: int
    n_graph_changes: int
    rerouted_cells_total: int
    commit_log: tuple[dict[str, Any], ...]
    # phase reconciliation (characteristic scheme; zeros for upwind)
    max_phase_reconciliation_residual_kg: Any  # 0-d: largest |actual - predicted| remaining pool over the event
    n_phase_canonicalized: int  # cell/class/steps with a predicted remnant but an actual pool of exactly 0
    n_phase_rounding_remnants: int  # cell/class/steps with an actual pool > 0 but a predicted pool of 0
    phase_steered_cells_total: int  # rerouted cells carrying mobile mass, summed over commits
    # steps
    n_accepted_steps: int
    n_rejected_attempts: int
    rejections: tuple[dict[str, Any], ...]
    min_accepted_dt_s: float
    max_accepted_dt_s: float
    n_maple_water_calls: int
    first_step_wall_s: float
    first_step_cpu_s: float
    remaining_wall_s: float
    remaining_cpu_s: float
    commit_wall_s: float
    n_inventory_terms: int
    # CW-D2 (opt-in; None by default): the event-lifetime exchange history as a plain dict of arrays in the
    # graph namespace plus host scalars -- `EXCHANGE_HISTORY_KEYS`. The committed part sums every recorded
    # commit's pre-clear receipt; the live pending ledger is NOT included here and is added exactly once by
    # `exchange_history_totals`. Never an inventory; never read by any physics.
    exchange_history: dict[str, Any] | None = None
    # CW-B102: operational deep-publication record of THIS evolve call only (policy "boundary"; None under the
    # default "step"). Per-execution segment: a continuation starts a new segment; never a cumulative total.
    deep_publication: dict[str, Any] | None = None
    # CW-B104 net-elevation-only rerouting (defaults LAST; cumulative across a continuation). `n_commits` /
    # `n_forced_commits` (the caller's final forced commits) / `n_graph_changes` keep their meaning; a reroute is a
    # commit whose post-commit net displacement from the held routing source reached the threshold; `n_net_commits`
    # counts the genuine extra commits made only because the gate reached it between MAPLE triggers. One gate
    # evaluation (one scalar read) per accepted step with commits enabled. The routing source is the entry state's
    # (`initial_event_state` initialization or a restored one) until the first reroute (`routing_source_t_s`).
    n_reroutes: int = 0
    n_net_commits: int = 0
    n_routing_gate_evaluations: int = 0
    max_unrouted_net_displacement_m: float = 0.0
    routing_source_origin: str = "entry"
    routing_source_t_s: float | None = None

    def bed_change_by_cell_class_kg(self) -> Any:
        """ACTUAL final minus initial MAPLE bed inventory per cell and class
        (`voxel.sum(axis=2) + active`, robust to the voxel count): the
        morphological ground truth, which includes every bed process MAPLE
        ran (water exchange, avalanching inside commits, sub-resolution
        placement). Contrast `cumulative_deposition_kg -
        cumulative_pickup_kg`, the WATER class exchange only."""
        bed = self.state.bed
        return bed_by_cell_class(bed) - self.initial_bed_by_cell_class_kg

    def closure(self) -> dict[str, Any]:
        """Per-class sediment closure from ACTUAL inventories (host):
        `bed_final + mobile_final + export_actual - (bed_initial +
        mobile_initial)` against MAPLE’s shared summation-error policy,
        scaled by real reservoir operands and actual accounting calls.
        Reported numerical residuals remain diagnostics, not lost mass. Also the water-independent request
        reconciliation (requested pickup = actual + refused + residual)."""
        from maple.core.backend import to_host

        bc = {k: np.asarray(to_host(v), dtype=np.float64) for k, v in self.by_class.items()}
        initial_bed = np.asarray(to_host(self.initial_bed_inventory_kg), dtype=np.float64)
        initial_mobile = np.asarray(to_host(self.initial_mobile_kg), dtype=np.float64)
        final_bed = np.asarray(to_host(bed_inventory(self.state.bed)), dtype=np.float64)
        final_mobile = np.asarray(to_host(self.state.bed.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1))),
                                  dtype=np.float64)
        export = bc["export_actual"]
        residual = (final_bed + final_mobile + export) - (initial_bed + initial_mobile)
        scalar = float(to_host(self.numerical_residual_scalar_abs_kg))
        from maple.surface.voxels._numerics import summation_error_bound_kg

        from maple_syrup.conservation import reservoir_bound_kg

        bound = reservoir_bound_kg(self.n_maple_water_calls, initial_bed, final_bed,
                                   initial_mobile, final_mobile, export)
        tolerance = np.full_like(initial_bed, bound)
        request = bc["requested_pickup"] - (bc["actual_pickup"] + bc["availability_shortfall"]
                                            + bc["holdings_shortfall"] + np.abs(bc["pickup_numerical_residual"]))
        # Same request identity as MAPLE water.validation, with the actual
        # number of accumulated calls and cells for this event.
        n_cells = self.state.graph.shape[0] * self.state.graph.shape[1]
        request_tol = np.full_like(initial_bed, summation_error_bound_kg(
            4 * n_cells * max(self.n_maple_water_calls, 1),
            max(float(np.max(bc["requested_pickup"])), 1.0)))
        return {
            "initial_bed_kg": initial_bed.tolist(),
            "initial_mobile_kg": initial_mobile.tolist(),
            "final_bed_kg": final_bed.tolist(),
            "final_mobile_kg": final_mobile.tolist(),
            "export_actual_kg": export.tolist(),
            "net_bed_change_kg": (final_bed - initial_bed).tolist(),
            "residual_kg": residual.tolist(),
            "tolerance_kg": tolerance.tolist(),
            "closed": bool(np.all(np.abs(residual) <= tolerance)),
            "request_reconciliation_residual_kg": request.tolist(),
            "request_reconciliation_tolerance_kg": request_tol.tolist(),
            "request_reconciled": bool(np.all(np.abs(request) <= request_tol)),
            "numerical_residual_scalar_abs_kg": scalar,
            "tolerance_rule": "MAPLE summation_error_bound_kg(max(n_accounting_calls,1), "
                              "largest genuine bed/mobile/boundary operand); no physical-resolution "
                              "floor or numerical-residual allowance",
        }


# --- exchange history (CW-D2) ------------------------------------------------------------------------------------
EXCHANGE_HISTORY_KEYS = ("reference_inventory_kg", "committed_n_physical_touches",
                         "committed_gross_physical_activity_kg", "n_commits_recorded", "receipt")
_RECEIPT_KEYS = frozenset({"captured_after_stage", "captured_before_stage", "touch_semantics", "gross_semantics",
                           "namespace", "touches_dtype", "gross_dtype", "shape"})


def new_exchange_history(xp: ModuleType, shape: tuple[int, int], n_classes: int,
                         reference_inventory_kg: Any) -> dict[str, Any]:
    """An empty event-lifetime history bound to the event's reconciliation reference (the initial bed per cell
    and class, an independent copy). Arrays live in the event namespace `xp`; `receipt` is None until the first
    recorded commit; nothing is inferred from steps or domain size."""
    reference = xp.array(reference_inventory_kg, dtype=np.float64, copy=True)
    if tuple(reference.shape) != (*shape, n_classes):
        raise SedimentEventError("exchange history reference inventory must be (ny, nx, nc)")
    return {"reference_inventory_kg": reference,
            "committed_n_physical_touches": xp.zeros((*shape, n_classes), dtype=np.int64),
            "committed_gross_physical_activity_kg": xp.zeros((*shape, n_classes), dtype=np.float64),
            "n_commits_recorded": 0, "receipt": None}


def copy_exchange_history(history: dict[str, Any], xp: ModuleType) -> dict[str, Any]:
    """An independently owned copy (new arrays) of a history in namespace `xp`."""
    return {"reference_inventory_kg": xp.array(history["reference_inventory_kg"], dtype=np.float64, copy=True),
            "committed_n_physical_touches": xp.array(history["committed_n_physical_touches"], dtype=np.int64, copy=True),
            "committed_gross_physical_activity_kg": xp.array(history["committed_gross_physical_activity_kg"],
                                                             dtype=np.float64, copy=True),
            "n_commits_recorded": int(history["n_commits_recorded"]),
            "receipt": None if history["receipt"] is None else dict(history["receipt"])}


def _maple_receipt_contract() -> dict[str, str]:
    """MAPLE's ACTUAL capture contract: the stage names and the touch / gross semantics strings exported by the
    bound dependency. A dependency without them has no contract and is refused explicitly; nothing is guessed."""
    try:
        from maple.surface.topographic_commit.commit import (
            EXCHANGE_HISTORY_CAPTURE_AFTER_STAGE,
            EXCHANGE_HISTORY_CAPTURE_BEFORE_STAGE,
            EXCHANGE_HISTORY_GROSS_SEMANTICS,
            EXCHANGE_HISTORY_TOUCH_SEMANTICS,
        )
    except ImportError as exc:
        raise SedimentEventError("the bound MAPLE dependency exports no exchange history capture contract (stage and "
                                 "semantics constants); refusing to interpret any receipt") from exc
    return {"captured_after_stage": EXCHANGE_HISTORY_CAPTURE_AFTER_STAGE,
            "captured_before_stage": EXCHANGE_HISTORY_CAPTURE_BEFORE_STAGE,
            "touch_semantics": EXCHANGE_HISTORY_TOUCH_SEMANTICS,
            "gross_semantics": EXCHANGE_HISTORY_GROSS_SEMANTICS}


def _check_receipt_contract(receipt: Any, xp: ModuleType, shape3: tuple[int, ...], what: str) -> None:
    """A receipt is accepted only if its keys, its four contract strings (both stage names AND both semantics
    strings, compared to MAPLE's actual constants), its namespace, dtypes and shape all match. Host-only."""
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise SedimentEventError(f"{what}: MAPLE exchange history receipt has unexpected keys")
    for key, expected in _maple_receipt_contract().items():
        if receipt[key] != expected:
            raise SedimentEventError(f"{what}: receipt {key!r} differs from MAPLE's actual capture contract; refusing "
                                     "(the counter semantics are MAPLE's, never relabelled here)")
    if (receipt["namespace"] != getattr(xp, "__name__", None) or receipt["touches_dtype"] != "int64"
            or receipt["gross_dtype"] != "float64" or list(receipt["shape"]) != list(shape3)):
        raise SedimentEventError(f"{what}: receipt namespace, dtypes or shape disagree with the event")


def _checked_sum(xp: ModuleType, current_touches: Any, current_gross: Any, touches: Any, gross: Any,
                 what: str) -> tuple[Any, Any]:
    """The running history plus one increment (a commit receipt or the live pending ledger), as NEW arrays, with
    one batched counted flag read (commit cadence) covering: the increment is finite and nonnegative, the current
    history is finite and nonnegative, the summed gross activity is finite (an FP64 overflow is refused, never
    published) and the int64 counter did not wrap. Inputs are never modified."""
    from maple.core.backend import DeferredChecks, finite_flag, negative_flag, true_flag

    if touches.dtype.kind not in "iu" or gross.dtype != np.float64:
        raise SedimentEventError(f"{what}: touches must be an integer array and gross activity float64")
    if current_touches.dtype != np.int64 or current_gross.dtype != np.float64:
        raise SedimentEventError(f"{what}: the recorded history must hold int64 touches and float64 gross activity")
    if (tuple(touches.shape) != tuple(current_touches.shape) or tuple(gross.shape) != tuple(current_touches.shape)
            or tuple(current_gross.shape) != tuple(current_touches.shape)):
        raise SedimentEventError(f"{what}: receipt shape differs from the event history shape")
    increment = touches.astype(np.int64)
    new_touches = current_touches + increment  # int64 wraps silently on arrays: detected below, never kept
    new_gross = current_gross + gross
    checks = DeferredChecks()
    checks.require(finite_flag(gross), f"{what}: receipt gross activity must be finite")
    checks.forbid(negative_flag(gross), f"{what}: receipt gross activity must be >= 0")
    checks.forbid(negative_flag(increment), f"{what}: receipt touch counts must be >= 0")
    checks.require(finite_flag(current_gross), f"{what}: the recorded history gross activity is not finite")
    checks.forbid(negative_flag(current_gross), f"{what}: the recorded history gross activity is negative")
    checks.forbid(negative_flag(current_touches), f"{what}: the recorded history touch counter is negative")
    checks.require(finite_flag(new_gross), f"{what}: the summed gross activity overflowed FP64; refusing to publish")
    checks.forbid(true_flag((new_touches < current_touches) | (new_touches < increment)),
                  f"{what}: the int64 touch counter overflowed; refusing to publish")
    try:
        checks.resolve()
    except ValueError as exc:
        raise SedimentEventError(str(exc)) from None
    return new_touches, new_gross


def fold_commit_exchange_history(history: dict[str, Any], maple_result: Any, xp: ModuleType) -> dict[str, Any]:
    """The history plus ONE MAPLE commit's pre-clear receipt (`CommitResult.pre_commit_n_physical_touches`,
    `pre_commit_gross_physical_activity_kg`, `exchange_history_receipt`): new dict, new arrays, inputs
    untouched. A commit without the receipt, a mixed namespace, a changed receipt, a bad sign / dtype / shape or
    a counter overflow is refused before anything is returned, so a caller publishes nothing on failure."""
    from maple.core.backend import MixedArrayNamespaceError, array_namespace

    touches = getattr(maple_result, "pre_commit_n_physical_touches", None)
    gross = getattr(maple_result, "pre_commit_gross_physical_activity_kg", None)
    receipt = getattr(maple_result, "exchange_history_receipt", None)
    if touches is None or gross is None or receipt is None:
        raise SedimentEventError("MAPLE commit returned no exchange history receipt; the recorded history cannot be "
                                 "continued (the dependency or the bed context did not record it)")
    current_t = history["committed_n_physical_touches"]
    current_g = history["committed_gross_physical_activity_kg"]
    try:
        namespace = array_namespace(touches, gross, current_t, current_g)
    except MixedArrayNamespaceError as exc:
        raise SedimentEventError(f"exchange history receipt in a mixed namespace: {exc}") from None
    if namespace is not xp:
        raise SedimentEventError("exchange history receipt is not in the event namespace")
    # The FIRST receipt and every later one must satisfy MAPLE's actual contract (stage names and semantics
    # strings), and every later one must equal the first.
    _check_receipt_contract(receipt, xp, tuple(current_t.shape), "commit exchange history")
    if history["receipt"] is not None and history["receipt"] != receipt:
        raise SedimentEventError("MAPLE exchange history receipt changed between commits of one event; refusing")
    new_touches, new_gross = _checked_sum(xp, current_t, current_g, touches, gross, "commit exchange history")
    return {"reference_inventory_kg": history["reference_inventory_kg"],
            "committed_n_physical_touches": new_touches,
            "committed_gross_physical_activity_kg": new_gross,
            "n_commits_recorded": int(history["n_commits_recorded"]) + 1,
            "receipt": dict(receipt)}


def exchange_history_totals(result: SedimentEventResult) -> dict[str, Any]:
    """Event-lifetime totals: the committed receipts plus the LIVE pending ledger, added exactly once here (the
    pending ledger holds the exchanges since the last commit, which no receipt has captured yet). Arrays are new
    and live in the event namespace. `history_complete` is True only when every commit of the result was
    recorded; `pending_ledger_empty` tells whether the live part is zero (a completed handoff)."""
    from maple.core.backend import to_bool
    from maple.core.types.sediment_ledger import sum_over_process_axis
    from maple.coupling.sediment_ledger.diagnostics import (
        gross_physical_activity_by_process_cell_class_kg,
    )

    history = result.exchange_history
    if history is None:
        raise SedimentEventError("the event did not record its exchange history (record_exchange_history was off); "
                                 "no totals can be claimed")
    xp = result.state.graph.xp
    ledger = result.state.bed.ledger
    pending_touches = ledger.n_physical_touches
    rows = gross_physical_activity_by_process_cell_class_kg(ledger)
    try:
        from maple.core.types.sediment_ledger import sum_over_present_process_axis
    except ImportError:
        # Older dense MAPLE dependencies keep the original six-channel contract.
        pending_gross = sum_over_process_axis(rows)
    else:
        pending_gross = sum_over_present_process_axis(ledger, rows)
    committed_t = history["committed_n_physical_touches"]
    committed_g = history["committed_gross_physical_activity_kg"]
    # Same batched guards as a commit fold: the summed totals are finite and the counter did not wrap, or nothing
    # is returned.
    total_touches, total_gross = _checked_sum(xp, committed_t, committed_g, pending_touches, pending_gross,
                                              "pending ledger exchange history")
    pending_empty = not to_bool(xp.any(pending_touches != 0)) and not to_bool(xp.any(pending_gross != 0.0))
    return {
        "arrays": {
            "reference_inventory_kg": xp.array(history["reference_inventory_kg"], dtype=np.float64, copy=True),
            "n_physical_touches": total_touches,
            "gross_physical_activity_kg": total_gross,
            "committed_n_physical_touches": xp.array(committed_t, dtype=np.int64, copy=True),
            "committed_gross_physical_activity_kg": xp.array(committed_g, dtype=np.float64, copy=True),
            "pending_n_physical_touches": xp.array(pending_touches, dtype=np.int64, copy=True),
            "pending_gross_physical_activity_kg": xp.array(pending_gross, dtype=np.float64, copy=True),
        },
        "n_commits_recorded": int(history["n_commits_recorded"]),
        "history_complete": int(history["n_commits_recorded"]) == int(result.n_commits),
        "pending_ledger_empty": bool(pending_empty),
        "includes_pending_ledger": True,
        "receipt": None if history["receipt"] is None else dict(history["receipt"]),
    }


def validate_exchange_history(history: Any, xp: ModuleType, shape3: tuple[int, int, int], n_commits: int) -> None:
    """Structural truth of a recorded history (checkpoint / continuation): exact keys, arrays of the namespace
    with the declared shapes and dtypes, finite and nonnegative, an integer commit count equal to the result's
    commit count (every commit recorded), and a receipt consistent with MAPLE's capture contract."""
    from maple.core.backend import array_namespace, is_array, to_bool

    if not isinstance(history, dict) or set(history) != set(EXCHANGE_HISTORY_KEYS):
        raise SedimentEventError("exchange history keys mismatch")
    for name, dtype in (("reference_inventory_kg", np.float64), ("committed_n_physical_touches", np.int64),
                        ("committed_gross_physical_activity_kg", np.float64)):
        value = history[name]
        if (not is_array(value) or array_namespace(value) is not xp or tuple(value.shape) != tuple(shape3)
                or value.dtype != dtype):
            raise SedimentEventError(f"invalid exchange history array {name}")
        if to_bool(xp.any(value < 0)) or (dtype == np.float64 and not to_bool(xp.all(xp.isfinite(value)))):
            raise SedimentEventError(f"invalid exchange history values in {name}")
    count = history["n_commits_recorded"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise SedimentEventError("exchange history commit count must be a nonnegative int")
    if count != int(n_commits):
        raise SedimentEventError(f"exchange history covers {count} commits but the event made {n_commits}; a partial "
                                 "history is refused, never treated as complete")
    receipt = history["receipt"]
    if count == 0:
        if receipt is not None:
            raise SedimentEventError("exchange history has a receipt without any recorded commit")
        return
    # Stage names AND semantics strings against MAPLE's actual constants (plus namespace, dtypes, shape).
    _check_receipt_contract(receipt, xp, tuple(shape3), "exchange history")


def morphology_summary(bed_change_kg: Any, water_exchange_kg: Any, *, bulk_density_kg_m3: float,
                       cell_area_m2: float) -> dict[str, Any]:
    """Morphological reporting from the ACTUAL per-cell/class bed change
    (`SedimentEventResult.bed_change_by_cell_class_kg()`), kept separate
    from the water CLASS exchange (`cumulative_deposition - cumulative_pickup`).

    Net erosion / deposition sum the classes PER CELL before taking the
    negative / positive parts, so a balanced compositional exchange (one
    class in, another out, same cell) is zero morphology but nonzero
    `class_sorting_exchange_kg`. `non_water_bed_change_abs_kg` is the part of
    the actual bed change that the water exchange does not explain (MAPLE
    avalanching inside commits, sub-resolution placement); it is measured
    from independent inventories, never re-derived from the exchange."""
    from maple.core.backend import to_host

    change = np.asarray(to_host(bed_change_kg), dtype=np.float64)
    exchange = np.asarray(to_host(water_exchange_kg), dtype=np.float64)
    if change.shape != exchange.shape or change.ndim != 3:
        raise SedimentEventError("bed change and water exchange must both be (ny, nx, nc)")
    cell = change.sum(axis=-1)
    exchange_cell = exchange.sum(axis=-1)
    non_water = change - exchange
    per_m = float(bulk_density_kg_m3) * float(cell_area_m2)
    dz = cell / per_m
    return {
        "basis": "actual final - initial MAPLE bed inventory per cell and class (voxel.sum(axis=2) + active); "
                 "classes summed per cell before splitting erosion / deposition; water class exchange reported "
                 "separately",
        "net_erosion_kg": float(-cell[cell < 0.0].sum()),
        "net_deposition_kg": float(cell[cell > 0.0].sum()),
        "net_bed_change_kg": float(cell.sum()),
        "n_cells_lowered": int(np.sum(cell < 0.0)),
        "n_cells_raised": int(np.sum(cell > 0.0)),
        "elevation_change_equivalent_m": {"min": float(dz.min()), "max": float(dz.max()), "mean": float(dz.mean()),
                                          "rule": "cell mass change / (bulk density x cell area)"},
        "class_bed_change_kg": change.sum(axis=(0, 1)).tolist(),
        "class_water_exchange_kg": exchange.sum(axis=(0, 1)).tolist(),
        "water_exchange_net_erosion_kg": float(-exchange_cell[exchange_cell < 0.0].sum()),
        "water_exchange_net_deposition_kg": float(exchange_cell[exchange_cell > 0.0].sum()),
        # mass swapped between classes within cells without changing the cell mass
        "class_sorting_exchange_kg": float(0.5 * (np.abs(exchange).sum(axis=-1) - np.abs(exchange_cell)).sum()),
        "non_water_bed_change_abs_kg": float(np.abs(non_water).sum()),
        "non_water_bed_change_max_abs_kg": float(np.abs(non_water).max()) if non_water.size else 0.0,
    }


def _row(xp, scalars, vectors):
    parts = [xp.stack([xp.asarray(v, dtype=np.float64) for v in scalars])]
    parts += [xp.asarray(v, dtype=np.float64).reshape(-1) for v in vectors]
    return xp.concatenate(parts)


# --- evolution -------------------------------------------------------------------------------------------
def evolve_sediment_event(
    state: SedimentEventState,
    context: BedContext,
    column: ColumnParameters,
    field: RainfallField,
    schedule: RainfallSchedule,
    vegetation_cover_fraction: Any,
    sediment: SedimentPhysicsParameters,
    end_s: float,
    control: SedimentEventControl,
    *,
    report_every_s: float,
    max_report_rows: int = 100_000,
    continuation: SedimentEventResult | None = None,
    planned_boundaries: np.ndarray | None = None,
    on_accepted_step=None,
    on_boundary=None,
    hydraulics: CudaHydraulics | None = None,
    _staged_accepted_observer=None,
) -> SedimentEventResult:
    """Advance the event from `state.t_s` to `end_s` (module docstring).
    Raises `SedimentEventError` on validation / guard failures; the two
    recoverable rejections are retried until the guards trip; every other
    exception propagates untouched with nothing published.

    `hydraulics` (CUDA control only; refused otherwise) is the caller-owned
    `CudaHydraulics` lifecycle: its context is prepared ONCE before the first
    step unless already bound to `state.graph` / `column`, reused by every
    attempt and retry, and re-prepared for the refreshed graph of every
    commit before the commit is published. A CUDA control without a
    lifecycle gets a private one (its counts are then not visible to the
    caller; pass one to report them).

    CW-B102 (`control.deep_publication == "boundary"`, opt-in): the actual MAPLE pending deep journal is carried
    across accepted steps and staged commits and published only at report-boundary rows (before the forced commit and
    `on_boundary`), before return, at `deep_publication_high_water_records`, and in ONE bounded flush of the previous
    accepted state before redoing the same attempt after a recognized carried-journal append-cap refusal. The public
    `on_accepted_step` hook (which receives published states) is refused under that policy; the PRIVATE
    `_staged_accepted_observer` (used by `complete_event` for its quiet monitor) receives the possibly staged
    accepted state and must read only resident water / storm / graph fields and never retain it."""
    from maple.core.backend import (
        freeze,
        read_transfer_counters,
        to_bool,
        to_float,
        to_int,
    )

    control = control.validated()
    storm_control = control.storm
    if hydraulics is not None and not isinstance(hydraulics, CudaHydraulics):
        raise SedimentEventError("hydraulics must be a CudaHydraulics lifecycle (or None)")
    if hydraulics is not None and not control.cuda:
        raise SedimentEventError("hydraulics is only meaningful with storm implementation 'cuda'")
    # CW-B102 / B102C1: the pure callback and publication-policy combination checks run FIRST, before
    # `_check_event_inputs` (which builds host / device validation arrays): an invalid combination is refused before
    # any array allocation or physics. The default 'step' path performs no new check except the private observer's.
    boundary_publication = control.deep_publication == "boundary"
    if _staged_accepted_observer is not None:
        if on_accepted_step is not None:
            raise SedimentEventError("pass either the public on_accepted_step hook or the private staged observer, "
                                     "not both")
        if not callable(_staged_accepted_observer):
            raise SedimentEventError("the private staged accepted-step observer must be callable")
    if boundary_publication:
        if on_accepted_step is not None:
            raise SedimentEventError("the public on_accepted_step hook receives published states and is refused with "
                                     "deep_publication 'boundary' (use the default 'step' policy for public hooks)")
        if on_boundary is not None and not callable(on_boundary):
            raise SedimentEventError("on_boundary must be callable")
    _check_event_inputs(state, context, column, sediment, storm_control.root_tolerance_m, control)
    # CW-B104: the routing graph must be the graph of the HELD routing source (one host rebuild here at entry, never
    # per step) and the threshold is the actual MAPLE pending_elevation_trigger_m; refused before any event array.
    try:
        check_graph_matches_routing_source(state.graph, state.terrain, state.routing_source_elevation_m)
        reroute_threshold = reroute_threshold_m(context)
    except BedIntegrationError as exc:
        raise SedimentEventError(str(exc)) from None
    if boundary_publication:
        # State / context-dependent policy checks need the types `_check_event_inputs` has just validated; they read
        # attributes only (the MAPLE digest reads the package files once per process), before any event array.
        if not is_windowed_bed(state.bed):
            raise SedimentEventError("deep_publication 'boundary' needs a resident-window MAPLE bed")
        journal_cap = resolved_journal_cap(context)
        if not control.deep_publication_high_water_records < journal_cap:
            raise SedimentEventError(f"deep_publication_high_water_records {control.deep_publication_high_water_records}"
                                     f" must be below the actual MAPLE journal cap {journal_cap}")
        # CW-B108: source-qualified recognizer -- the pinned M94 message path, or the accepted typed source's
        # JournalCapacityError (checked structurally here); any other MAPLE source is refused before any array.
        recognizer = journal_recognizer_source()
        if recognizer is None:
            raise SedimentEventError("deep_publication 'boundary' recognizes only the pinned M94 append-cap refusal or "
                                     "the accepted typed JournalCapacityError source; the bound MAPLE package is "
                                     "neither")
        if recognizer == "typed_journal_capacity_error":
            try:
                require_typed_journal_capacity_error()
            except BedIntegrationError as exc:
                raise SedimentEventError(str(exc)) from None
    # CW-B108: the deep-chunk storage option (default 'full') against this bed and the bound MAPLE source, read-only and
    # before any event array; an unsupported / cross-encoding combination is refused here, not at a publication.
    try:
        validate_deep_storage(context, state.bed)
    except BedIntegrationError as exc:
        raise SedimentEventError(str(exc)) from None
    graph = state.graph
    xp = graph.xp
    shape = graph.shape
    nc = sediment.n_classes
    area = graph.dx_m * graph.dx_m
    if not isinstance(field, RainfallField) or field.shape != shape or field.xp is not xp:
        raise SedimentEventError("rainfall field must match the graph shape and namespace")
    end = _real(end_s, "end_s")
    boundaries = plan_boundaries(schedule, state.t_s, end, report_every_s, max_report_rows=max_report_rows)
    if planned_boundaries is not None:
        candidate = np.asarray(planned_boundaries)
        if (candidate.dtype != np.float64 or candidate.ndim != 1 or candidate.size == 0
                or not np.isfinite(candidate).all() or candidate[0] <= state.t_s
                or np.any(np.diff(candidate) <= 0) or candidate[-1] != end):
            raise SedimentEventError("invalid explicit continuation boundaries")
        # Every remaining rainfall edge is mandatory, irrespective of reporting.
        knots = schedule.edges_s[(schedule.edges_s > state.t_s) & (schedule.edges_s < end)]
        if not np.isin(knots, candidate).all():
            raise SedimentEventError("continuation boundaries omit rainfall changes")
        boundaries = candidate.copy()
    if continuation is not None and continuation.state is not state:
        raise SedimentEventError("continuation must own the supplied accepted state")
    prefix_rows = 0 if continuation is None else len(continuation.boundaries)
    if prefix_rows + len(boundaries) > max_report_rows:
        raise SedimentEventError("cumulative reporting rows exceed max_report_rows")
    columns = sediment_hydrograph_columns(nc)
    recording = control.record_exchange_history
    if recording != bool(getattr(context, "record_exchange_history", False)):
        raise SedimentEventError("control.record_exchange_history and context.record_exchange_history must agree "
                                 "(the control accumulates what the bed context asks MAPLE to capture); refusing "
                                 "before any step")
    if recording:
        from maple_syrup.sediment_bed import maple_supports_exchange_history

        if not maple_supports_exchange_history():
            raise SedimentEventError("the bound MAPLE dependency cannot record the exchange history (commit_topography "
                                     "has no record_exchange_history keyword); refusing before any step")
    if control.cuda:
        # Prepare ONCE before any time is advanced (counted static download for validation, owned uploads,
        # kernel load); a lifecycle already bound to this graph / column is reused unchanged.
        if hydraulics is None:
            hydraulics = CudaHydraulics()
        hydraulics.bind(graph, column, t_s=state.t_s, reason="initial")
        if storm_control.root_solver == "newton":
            _load_cuda_newton_kernels()  # compile/load now, not inside the first timed step

    def zeros(*extra):
        return xp.zeros((*shape, *extra), dtype=np.float64)

    def scalar(value=0.0):
        return xp.asarray(value, dtype=np.float64)

    def vector():
        return xp.zeros((nc,), dtype=np.float64)

    active = graph.active_flat.reshape(shape)
    outlet = graph.outlet_flat.reshape(shape)
    zero_demand = freeze(zeros(nc))  # read-only on NumPy; MAPLE never writes into a demand array
    # water accumulators (Phase 4 columns)
    cum_rain, cum_intake, cum_return, cum_drain = zeros(), zeros(), zeros(), zeros()
    peak_depth = xp.array(state.storm.depth_m, copy=True)
    last_velocity = xp.where(active, xp.sqrt(state.storm.depth_m) * graph.conveyance.reshape(shape), 0.0)
    peak_velocity = xp.array(last_velocity, copy=True)
    outlet_q = graph.dx_m * xp.sum(xp.where(outlet, state.storm.discharge_m2_s, 0.0))
    peak_q, peak_t = scalar(outlet_q), scalar(state.t_s)
    cum_export = scalar()
    max_balance, max_constitutive, max_cr_old, max_cr_new = scalar(), scalar(), scalar(), scalar()
    cells_no, cells_partial, cells_complete = (xp.zeros((), dtype=np.int64) for _ in range(3))
    # sediment accumulators
    cum_pickup, cum_deposition, cum_export_request = zeros(nc), zeros(nc), zeros(nc)
    by_class = {name: vector() for name in (
        "requested_pickup", "raindrop_pickup_requested", "flow_pickup_requested", "actual_pickup",
        "availability_shortfall", "holdings_shortfall", "pickup_numerical_residual", "deposition_requested",
        "deposition_actual", "deposition_unmet", "export_requested", "export_actual", "export_unmet",
        "transport_budget_residual", "transport_budget_tolerance", "deposit_numerical_residual")}
    residual_scalar = scalar()
    mobile_by_class = state.bed.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1))
    initial_bed = bed_inventory(state.bed)
    initial_bed_cells = bed_by_cell_class(state.bed)
    initial_mobile = xp.array(mobile_by_class, copy=True)
    # CW-D2: the history is bound to the event's reconciliation reference (the initial bed per cell/class).
    history = new_exchange_history(xp, shape, nc, initial_bed_cells) if recording else None
    peak_mobile, peak_mobile_t = scalar(xp.sum(mobile_by_class)), scalar(state.t_s)
    export_rate = vector()
    peak_export, peak_export_t = scalar(), scalar(state.t_s)
    max_sed_courant, max_decay, max_cell_residual = scalar(), scalar(), scalar()
    regime_steps = {name: xp.zeros((), dtype=np.int64) for name in REGIME_CODES}
    ledger_reset = xp.zeros(state.bed.ledger.process_totals_kg.shape, dtype=np.float64)
    committed_net = vector()
    hydrograph = xp.zeros((prefix_rows + boundaries.size, len(HYDROGRAPH_COLUMNS)), dtype=np.float64)
    sediment_hydrograph = xp.zeros((prefix_rows + boundaries.size, len(columns)), dtype=np.float64)
    rate = xp.empty(shape, dtype=np.float64)  # owned scratch

    n_steps = n_rejected = n_calls = n_transport_rejections = 0
    max_substeps = 0
    n_commits = n_forced = n_graph_changes = rerouted_total = 0
    n_reroutes = n_net_commits = n_gate = 0
    max_unrouted = 0.0
    source_origin, source_t = "entry", None
    max_phase_residual = scalar()
    n_phase_canonicalized = n_phase_remnants = phase_steered_total = 0
    commit_log: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    dt_min, dt_max = math.inf, 0.0
    commit_wall = 0.0
    t = state.t_s
    first_wall = first_cpu = 0.0
    wall0, cpu0 = time.perf_counter(), time.process_time()

    prior_wall = prior_cpu = 0.0
    if continuation is not None:
        prev = continuation
        cum_rain = xp.array(prev.cumulative_rain_m, copy=True)
        cum_intake = xp.array(prev.cumulative_intake_m, copy=True)
        cum_return = xp.array(prev.cumulative_saturation_return_m, copy=True)
        cum_drain = xp.array(prev.cumulative_drainage_m, copy=True)
        cum_export = xp.array(prev.cumulative_export_m3, copy=True)
        peak_depth = xp.array(prev.peak_depth_m, copy=True)
        peak_velocity = xp.array(prev.peak_velocity_m_s, copy=True)
        last_velocity = xp.array(prev.last_velocity_m_s, copy=True)
        peak_q = xp.array(prev.peak_outlet_discharge_m3_s, copy=True)
        peak_t = xp.array(prev.time_of_peak_outlet_s, copy=True)
        max_balance = xp.array(prev.max_routing_cell_balance_residual_m, copy=True)
        max_constitutive = xp.array(prev.max_constitutive_residual_m, copy=True)
        max_cr_old = xp.array(prev.max_courant_old, copy=True)
        max_cr_new = xp.array(prev.max_courant_new, copy=True)
        cells_no = xp.array(prev.cell_steps_no_runon, copy=True)
        cells_partial = xp.array(prev.cell_steps_partial_runon, copy=True)
        cells_complete = xp.array(prev.cell_steps_complete_runon, copy=True)
        cum_pickup = xp.array(prev.cumulative_pickup_kg, copy=True)
        cum_deposition = xp.array(prev.cumulative_deposition_kg, copy=True)
        cum_export_request = xp.array(prev.cumulative_export_request_kg, copy=True)
        residual_scalar = xp.array(prev.numerical_residual_scalar_abs_kg, copy=True)
        initial_bed = xp.array(prev.initial_bed_inventory_kg, copy=True)
        initial_bed_cells = xp.array(prev.initial_bed_by_cell_class_kg, copy=True)
        initial_mobile = xp.array(prev.initial_mobile_kg, copy=True)
        peak_mobile = xp.array(prev.peak_mobile_kg, copy=True)
        peak_mobile_t = xp.array(prev.time_of_peak_mobile_s, copy=True)
        peak_export = xp.array(prev.peak_export_rate_kg_s, copy=True)
        peak_export_t = xp.array(prev.time_of_peak_export_s, copy=True)
        max_sed_courant = xp.array(prev.max_sediment_courant, copy=True)
        max_decay = xp.array(prev.max_decay_exponent, copy=True)
        max_cell_residual = xp.array(prev.max_transport_cell_residual_kg, copy=True)
        ledger_reset = xp.array(prev.ledger_process_totals_reset_kg, copy=True)
        committed_net = xp.array(prev.committed_net_bed_change_kg, copy=True)
        n_steps = prev.n_accepted_steps
        n_rejected = prev.n_rejected_attempts
        n_calls = prev.n_maple_water_calls
        n_transport_rejections = prev.n_transport_rejections
        max_substeps = prev.max_transport_substeps_used
        n_commits = prev.n_commits
        n_forced = prev.n_forced_commits
        n_graph_changes = prev.n_graph_changes
        n_reroutes, n_net_commits, n_gate = prev.n_reroutes, prev.n_net_commits, prev.n_routing_gate_evaluations
        max_unrouted = prev.max_unrouted_net_displacement_m
        source_origin, source_t = prev.routing_source_origin, prev.routing_source_t_s
        rerouted_total = prev.rerouted_cells_total
        max_phase_residual = xp.array(prev.max_phase_reconciliation_residual_kg, copy=True)
        n_phase_canonicalized = prev.n_phase_canonicalized
        n_phase_remnants = prev.n_phase_rounding_remnants
        phase_steered_total = prev.phase_steered_cells_total
        dt_min = prev.min_accepted_dt_s
        dt_max = prev.max_accepted_dt_s
        commit_wall = prev.commit_wall_s
        first_wall = prev.first_step_wall_s
        first_cpu = prev.first_step_cpu_s
        by_class = {k: xp.array(v, copy=True) for k, v in prev.by_class.items()}
        regime_steps = {k: xp.array(v, copy=True) for k, v in prev.regime_cell_steps.items()}
        commit_log, rejections = list(prev.commit_log), list(prev.rejections)
        hydrograph[:prefix_rows] = prev.hydrograph
        sediment_hydrograph[:prefix_rows] = prev.sediment_hydrograph
        export_rate = by_class["export_actual"] * 0.0
        prior_wall, prior_cpu = prev.remaining_wall_s, prev.remaining_cpu_s
        if recording:
            # A continuation can carry the history only if it recorded EVERY commit so far; a checkpoint written
            # without the option (or before the option existed) has no history to claim.
            if prev.exchange_history is None or int(prev.exchange_history["n_commits_recorded"]) != prev.n_commits:
                raise SedimentEventError("the continuation has no complete exchange history (recorded without "
                                         "record_exchange_history, or a pre-history checkpoint); the history cannot "
                                         "be claimed from here")
            validate_exchange_history(prev.exchange_history, xp, (*shape, nc), prev.n_commits)
            history = copy_exchange_history(prev.exchange_history, xp)
        elif prev.exchange_history is not None:
            raise SedimentEventError("the continuation carries an exchange history but the control does not record "
                                     "one; refusing to drop it silently")
    all_boundaries = np.concatenate((np.empty(0) if continuation is None else continuation.boundaries, boundaries))
    # CW-B30 correction 1: last use of the continuation. Every restart array, scalar, control, history and boundary
    # has been copied or validated above; the one later read (`n_inventory_terms`, an int of the frozen result) is
    # taken here. This frame then drops its references to the prior result, so it is not pinned for the loop.
    # A caller that keeps its own reference keeps it; nothing is mutated.
    continuation_inventory_terms = 0 if continuation is None else continuation.n_inventory_terms
    if continuation is not None:
        del prev
    del continuation
    stopped = False
    rows_used = prefix_rows
    # CW-B102: per-execution-segment operational record (never cumulative across a restart; not physics).
    publication = None if not boundary_publication else {
        "policy": "boundary", "scope": "this evolve call only (per-execution segment); a continuation starts anew",
        "high_water_records": int(control.deep_publication_high_water_records),
        "journal_cap_records": int(resolved_journal_cap(context)), "n_publications": 0,
        "n_boundary_publications": 0, "n_high_water_publications": 0, "n_return_publications": 0,
        "n_journal_flushes": 0, "max_carried_records": pending_journal_records(state.bed)}
    if publication is not None and recognizer != "m94_pinned_message":
        # CW-B108: the recognizer actually in use, recorded only for the non-M94 source (M94 records are unchanged).
        publication["append_cap_recognizer"] = recognizer

    def publish_carried(kind):
        """Publish the CURRENT accepted state's carried MAPLE journal (a validated accepted state only; never a
        candidate). No-op when nothing is carried."""
        nonlocal state
        if pending_journal_records(state.bed) == 0:
            return
        state = replace(state, bed=publish_bed_storage(state.bed, context))
        publication["n_publications"] += 1
        publication["n_journal_flushes" if kind == "flush" else f"n_{kind}_publications"] += 1

    def pack_result():
        remaining_wall = prior_wall + time.perf_counter() - wall0
        remaining_cpu = prior_cpu + time.process_time() - cpu0
        nz = bed_voxel_count(state.bed)
        return SedimentEventResult(
            state=state, hydrograph=hydrograph[:rows_used], sediment_hydrograph=sediment_hydrograph[:rows_used], sediment_columns=columns,
            boundaries=all_boundaries[:rows_used].copy(),
            cumulative_rain_m=cum_rain, cumulative_intake_m=cum_intake, cumulative_saturation_return_m=cum_return,
            cumulative_drainage_m=cum_drain, cumulative_export_m3=cum_export,
            peak_depth_m=peak_depth, peak_velocity_m_s=peak_velocity, last_velocity_m_s=last_velocity,
            peak_outlet_discharge_m3_s=peak_q, time_of_peak_outlet_s=peak_t,
            max_courant_old=max_cr_old, max_courant_new=max_cr_new,
            max_routing_cell_balance_residual_m=max_balance, max_constitutive_residual_m=max_constitutive,
            cell_steps_no_runon=cells_no, cell_steps_partial_runon=cells_partial, cell_steps_complete_runon=cells_complete,
            cumulative_pickup_kg=cum_pickup, cumulative_deposition_kg=cum_deposition,
            cumulative_export_request_kg=cum_export_request, by_class=by_class,
            initial_bed_by_cell_class_kg=initial_bed_cells,
            initial_bed_inventory_kg=initial_bed, initial_mobile_kg=initial_mobile,
            numerical_residual_scalar_abs_kg=residual_scalar,
            peak_mobile_kg=peak_mobile, time_of_peak_mobile_s=peak_mobile_t,
            peak_export_rate_kg_s=peak_export, time_of_peak_export_s=peak_export_t,
            max_sediment_courant=max_sed_courant, max_decay_exponent=max_decay,
            max_transport_cell_residual_kg=max_cell_residual, max_transport_substeps_used=max_substeps,
            n_transport_rejections=n_transport_rejections, regime_cell_steps=regime_steps,
            ledger_process_totals_reset_kg=ledger_reset, committed_net_bed_change_kg=committed_net,
            n_commits=n_commits, n_forced_commits=n_forced, n_graph_changes=n_graph_changes,
            rerouted_cells_total=rerouted_total, commit_log=tuple(commit_log),
            max_phase_reconciliation_residual_kg=max_phase_residual, n_phase_canonicalized=n_phase_canonicalized,
            n_phase_rounding_remnants=n_phase_remnants, phase_steered_cells_total=phase_steered_total,
            n_accepted_steps=n_steps, n_rejected_attempts=n_rejected, rejections=tuple(rejections),
            min_accepted_dt_s=dt_min, max_accepted_dt_s=dt_max, n_maple_water_calls=n_calls,
            first_step_wall_s=first_wall, first_step_cpu_s=first_cpu,
            remaining_wall_s=remaining_wall, remaining_cpu_s=remaining_cpu, commit_wall_s=commit_wall,
            n_inventory_terms=max(shape[0] * shape[1] * (nz + 1), continuation_inventory_terms),
            exchange_history=history,
            deep_publication=None if publication is None else dict(publication),
            n_reroutes=n_reroutes, n_net_commits=n_net_commits, n_routing_gate_evaluations=n_gate,
            max_unrouted_net_displacement_m=max_unrouted, routing_source_origin=source_origin,
            routing_source_t_s=source_t,
        )

    def commit_and_reroute(bed, graph_, storm_, t_, *, force):
        """CW-B104: the per-accepted-step net-elevation gate (exact current surface vs the held routing source, one
        scalar read, outside the commit's transfer window), then `_commit_and_reroute` (forced by `force` or by the
        gate). Under CUDA the counted transfers of EVERY actual commit are recorded; the context of a REFRESHED graph
        is prepared BEFORE the commit is published (a preparation failure leaves nothing published) and never for a
        non-rerouting commit. Returns (commit or None, prepared context or None)."""
        nonlocal n_gate, max_unrouted
        gate = routing_gate_m(bed, context, graph_, state.routing_source_elevation_m)
        n_gate += 1
        before = read_transfer_counters() if hydraulics is not None else None
        commit = _commit_and_reroute(bed, context, graph_, state.terrain, storm_, t_, force=force,
                                     routing_source_m=state.routing_source_elevation_m,
                                     net_due=gate >= reroute_threshold, gate_m=gate)
        prepared = None
        if hydraulics is not None and commit is not None:
            hydraulics.record_commit_transfers(read_transfer_counters().delta(before))
            if commit.rerouted:
                prepared = hydraulics.prepare(commit.graph, column, t_s=t_, reason="commit")
        if commit is None:
            max_unrouted = max(max_unrouted, gate)
        elif not commit.rerouted:
            max_unrouted = max(max_unrouted, gate, commit.record["net_displacement_m"])
        return commit, prepared

    def prepared_commit_values(commit: _Commit, phase):
        folded = None if history is None else fold_commit_exchange_history(history, commit.maple_result, xp)
        # CW-B104: the CSR phase is rebound only to a REFRESHED network; a non-rerouting commit keeps it as is.
        return folded, phase_for_network(phase, commit.network) if commit.rerouted else phase

    def publish_commit(commit: _Commit, prepared=None, *, values=None) -> None:
        nonlocal state, n_commits, n_forced, n_graph_changes, rerouted_total, ledger_reset, committed_net
        nonlocal phase_steered_total, history, n_reroutes, n_net_commits, source_origin, source_t
        # CW-D2: fold this commit's pre-clear receipt FIRST (it raises on a missing / malformed receipt or a
        # counter overflow), so a refused history publishes nothing.
        folded, committed_phase = prepared_commit_values(commit, state.phase) if values is None else values
        # CW-B102: under 'boundary' the committed bed keeps MAPLE's staged journal (published at the row end).
        published_bed = commit.bed if boundary_publication else publish_bed_storage(commit.bed, context)
        if commit.rerouted:
            # The phase is kept as is: fractions and positions are carried over to
            # the rerouted cells' new outflow directions (documented steering). The
            # CSR phase is explicitly rebound to the refreshed network's digest and
            # slot maps (same active set, dx and shape are required by the rebind).
            state = replace(state, bed=published_bed, graph=commit.graph, network=commit.network, grid=commit.grid,
                            storm=commit.storm, phase=committed_phase, routing_source_elevation_m=commit.routing_source_m)
        else:
            # CW-B104 accounting commit: only the bed changes; graph / network / grid / storm (q) / phase binding /
            # routing source are the accepted ones, unchanged.
            state = replace(state, bed=published_bed)
        history = folded
        if prepared is not None:  # the refreshed graph's CUDA context, prepared before publication
            hydraulics.adopt(prepared)
        n_commits += 1
        n_forced += int(commit.record["forced"])
        n_net_commits += int(commit.record["commit_reason"] == "net_elevation")
        n_graph_changes += int(commit.record["graph_changed"])
        rerouted_total += commit.record["rerouted_cells"]
        phase_steered_total += commit.record["phase_steered_cells"]
        if commit.rerouted:
            n_reroutes += 1
            source_origin, source_t = "net_reroute", float(commit.record["t_s"])
        ledger_reset = ledger_reset + commit.maple_result.pre_commit_process_totals_kg
        committed_net = committed_net + commit.maple_result.pre_commit_pending_mass_by_class_kg
        if len(commit_log) < control.max_commit_log:
            commit_log.append(commit.record)

    def refresh_hydraulics() -> None:
        """Hydraulic diagnostics of the CURRENT accepted graph and state
        after a commit re-initialised the discharge (instantaneous kinematic
        view, no water volume added): velocity k sqrt(h), outlet discharge
        from the re-initialised q, the outlet mask of the refreshed graph.
        Peaks are taken over BOTH the pre-commit routed values and these
        post-commit values; the integrated export stays the routing's."""
        nonlocal last_velocity, outlet_q, peak_q, peak_t, outlet
        g_ = state.graph
        outlet = g_.outlet_flat.reshape(shape)
        last_velocity = xp.where(active, xp.sqrt(state.storm.depth_m) * g_.conveyance.reshape(shape), 0.0)
        outlet_q = g_.dx_m * xp.sum(xp.where(outlet, state.storm.discharge_m2_s, 0.0))
        higher = outlet_q > peak_q
        peak_t = xp.where(higher, scalar(t), peak_t)
        peak_q = xp.where(higher, outlet_q, peak_q)
        xp.maximum(peak_velocity, last_velocity, out=peak_velocity)

    for local_row, boundary in enumerate(boundaries.tolist()):
        row = prefix_rows + local_row
        t_row = t
        span = boundary - t_row
        n_sub = max(1, math.ceil(span / storm_control.max_dt_s))
        while span / n_sub > storm_control.max_dt_s:
            n_sub += 1
        for k in range(1, n_sub + 1):
            target = boundary if k == n_sub else t_row + k * (span / n_sub)
            if not target > t:
                raise SedimentEventError(f"floating time does not advance: planned substep target {target} s is "
                                         f"not after t = {t} s (time resolution too coarse for max_dt_s = "
                                         f"{storm_control.max_dt_s})")
            while t < target:
                dt = target - t
                snap = True
                retries = 0
                flushed = False  # CW-B102C1: at most ONE carried-journal flush per accepted step
                while True:
                    flush_now = False
                    if n_steps >= storm_control.max_steps:
                        raise SedimentEventError(f"max_steps = {storm_control.max_steps} reached at t = {t} s "
                                                 f"before end_s = {end} s")
                    if not t + dt > t:
                        raise SedimentEventError(f"floating time does not advance: t = {t} s, dt = {dt} s")
                    field.apply(schedule.rate_after_m_per_s(t), out=rate)
                    try:
                        attempt = _sediment_accounting_step(state, context, column, rate,
                                                            vegetation_cover_fraction, sediment, dt, control,
                                                            zero_demand=zero_demand, hydraulics=hydraulics)
                    except (RoutingStepRejected, TransportStepRejected) as exc:
                        if flushed:
                            # CW-B102C1: ANY exception of the post-flush redo propagates unchanged (no rejection
                            # bookkeeping, no dt halving, no further attempt).
                            raise
                        retries += 1
                        n_rejected += 1
                        if len(rejections) < control.max_rejections_recorded:
                            rejections.append({"t_s": t, "dt_tried_s": dt, "kind": type(exc).__name__,
                                               "reason": str(exc)})
                        if retries > storm_control.max_retries:
                            raise SedimentEventError(f"step at t = {t} s rejected {retries} times (max_retries = "
                                                     f"{storm_control.max_retries}); last dt {dt} s: {exc}") from exc
                        halved = dt * 0.5
                        if halved < storm_control.min_dt_s:
                            raise SedimentEventError(f"halving dt {dt} s to {halved} s would fall below the retry "
                                                     f"floor min_dt_s = {storm_control.min_dt_s} s at t = {t} s "
                                                     f"after {retries} rejection(s): {exc}") from exc
                        dt = halved
                        snap = False
                        continue
                    except ValueError as exc:  # after the two rejections (both ValueError subclasses)
                        # CW-B102 bounded flush / redo ('boundary' only): ONLY M94's exact append-cap refusal raised
                        # while the PREVIOUS accepted state carries a journal, and only before any flush of this
                        # step. CW-B102C1: only a scalar flag is set here; the flush and the redo run AFTER the
                        # handler has exited, so neither the exception nor its traceback (the failed attempt's
                        # frames and arrays) is alive during publication or the redo. Every other error, and any
                        # exception of the redo, propagates unchanged.
                        if (flushed or not boundary_publication
                                or not is_carried_append_cap_refusal(exc, context, pending_journal_records(state.bed))):
                            raise
                        flush_now = True
                    if flush_now:
                        # Outside the handler: publish the PREVIOUS validated accepted state, then redo the SAME
                        # attempt (same t, dt, snap); retries / rejections / dt untouched.
                        publish_carried("flush")
                        flushed = True
                        continue
                    break
                t_new = target if snap else t + dt
                # CW-B100 (lifetime only): the attempt has passed every retry and check above, and nothing below
                # reads the PREVIOUS accepted storm / bed / sediment velocity / phase again (a commit or reroute
                # failure propagates out of this function, which then publishes and returns nothing). This private
                # local is therefore rebound to the PENDING accepted state -- the same fields step 7 published,
                # with the solver bed `attempt.bed` (still unpublished) -- BEFORE the commit, so the previous step's
                # bed (ledger, window, active layer, water, availability), velocity, phase and storm are released
                # before MAPLE allocates the commit's fresh ledger. graph / network / grid / terrain are the same
                # objects the commit reads. No array is copied, recomputed or mutated, no caller object is touched,
                # and the published values are unchanged (the no-commit path publishes the deep storage below, the
                # commit path through `publish_commit`).
                storm_state = replace(attempt.storm.state, t_s=t_new)
                state = replace(state, storm=storm_state, bed=attempt.bed,
                                sediment_velocity_m_s=attempt.physics.sediment_velocity_m_s,
                                phase=attempt.phase if control.characteristic else None)
                # 6. MAPLE commit triggers on the post-step ledger; reroute on commit. Nothing
                #    below publishes until every check passed.
                commit = prepared = None
                if control.commit:
                    wall_c = time.perf_counter()
                    commit, prepared = commit_and_reroute(attempt.bed, state.graph, storm_state, t_new, force=False)
                    commit_wall += time.perf_counter() - wall_c
                # Prevalidate phase/history before ANY immutable deep version is published.
                commit_values = None if commit is None else prepared_commit_values(commit, attempt.phase)
                # CW-B102: under 'boundary' the accepted bed keeps MAPLE's staged journal (carried, not published).
                accepted_bed = (attempt.bed if commit is not None or boundary_publication
                                else publish_bed_storage(attempt.bed, context))
                # 7. publish (storm / velocity / phase were bound above; the bed is the published one).
                t = t_new
                state = replace(state, bed=accepted_bed)
                if attempt.phase_reconciliation is not None:
                    rec = attempt.phase_reconciliation
                    max_phase_residual = xp.maximum(max_phase_residual, rec["residual_max_kg"])
                    n_phase_canonicalized += to_int(rec["n_canonicalized"])  # counted scalar reads
                    n_phase_remnants += to_int(rec["n_rounding_remnants"])
                col, route = attempt.storm.column, attempt.storm.route
                pickup, deposit, transport, physics = attempt.pickup, attempt.deposit, attempt.transport, attempt.physics
                cum_rain += col.rain_m
                cum_intake += col.intake_m
                cum_return += col.saturation_return_m
                cum_drain += col.drainage_m
                cum_export = cum_export + route.export_m3  # integrated face volume of the accepted routing step
                # pre-commit routed diagnostics (the water that actually moved this step)
                outlet_q = route.outlet_discharge_m3_s
                higher = outlet_q > peak_q
                peak_t = xp.where(higher, scalar(t), peak_t)
                peak_q = xp.where(higher, outlet_q, peak_q)
                xp.maximum(peak_depth, route.depth_m, out=peak_depth)
                xp.maximum(peak_velocity, route.velocity_m_s, out=peak_velocity)
                last_velocity = route.velocity_m_s
                if commit is not None:
                    rerouted_now = commit.rerouted
                    publish_commit(commit, prepared, values=commit_values)
                    if rerouted_now:  # CW-B104: only a refreshed graph / re-initialised q changes these diagnostics
                        refresh_hydraulics()  # post-commit graph / q: diagnostics and peaks follow the accepted state
                if boundary_publication:
                    carried = pending_journal_records(state.bed)
                    publication["max_carried_records"] = max(publication["max_carried_records"], carried)
                    if carried >= control.deep_publication_high_water_records:
                        publish_carried("high_water")
                max_balance = xp.maximum(max_balance, route.max_cell_balance_residual_m)
                max_constitutive = xp.maximum(max_constitutive, route.max_constitutive_residual_m)
                max_cr_old = xp.maximum(max_cr_old, route.max_courant_old)
                max_cr_new = xp.maximum(max_cr_new, route.max_courant_new)
                cells_no = cells_no + attempt.storm.n_no_runon
                cells_partial = cells_partial + attempt.storm.n_partial_runon
                cells_complete = cells_complete + attempt.storm.n_complete_runon
                # sediment ground truth
                cum_pickup += pickup.actual_removal_by_cell_class_kg
                cum_deposition += deposit.deposition_by_cell_class_kg
                cum_export_request += transport.export_request_kg
                by_class["requested_pickup"] += pickup.requested_removal_by_class_kg
                by_class["raindrop_pickup_requested"] += physics.raindrop_pickup_by_class_kg
                by_class["flow_pickup_requested"] += physics.flow_pickup_by_class_kg
                by_class["actual_pickup"] += pickup.actual_removal_by_class_kg
                by_class["availability_shortfall"] += pickup.availability_shortfall_by_class_kg
                by_class["holdings_shortfall"] += pickup.shortfall_by_class_kg
                by_class["pickup_numerical_residual"] += pickup.numerical_residual_by_class_kg
                by_class["deposition_requested"] += transport.deposition_request_by_class_kg
                by_class["deposition_actual"] += deposit.deposited_mass_by_class_kg
                by_class["deposition_unmet"] += transport.deposition_request_by_class_kg - deposit.deposited_mass_by_class_kg
                by_class["export_requested"] += transport.export_request_by_class_kg
                by_class["export_actual"] += deposit.boundary_export_by_class_kg
                by_class["export_unmet"] += transport.export_request_by_class_kg - deposit.boundary_export_by_class_kg
                by_class["transport_budget_residual"] += transport.budget_residual_by_class_kg
                by_class["transport_budget_tolerance"] += transport.budget_tolerance_by_class_kg
                by_class["deposit_numerical_residual"] += deposit.numerical_residual_by_class_kg
                residual_scalar = residual_scalar + abs(pickup.numerical_residual_scalar_kg) \
                    + abs(deposit.numerical_residual_scalar_kg)
                mobile_by_class = state.bed.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1))
                mobile_total = xp.sum(mobile_by_class)
                higher = mobile_total > peak_mobile
                peak_mobile_t = xp.where(higher, scalar(t), peak_mobile_t)
                peak_mobile = xp.where(higher, mobile_total, peak_mobile)
                export_rate = deposit.boundary_export_by_class_kg / dt
                export_total_rate = xp.sum(export_rate)
                higher = export_total_rate > peak_export
                peak_export_t = xp.where(higher, scalar(t), peak_export_t)
                peak_export = xp.where(higher, export_total_rate, peak_export)
                max_sed_courant = xp.maximum(max_sed_courant, transport.max_courant)
                max_decay = xp.maximum(max_decay, transport.max_decay_exponent)
                max_cell_residual = xp.maximum(max_cell_residual, transport.max_cell_balance_residual_kg)
                for name, count in physics.regime_counts.items():
                    regime_steps[name] = regime_steps[name] + count
                max_substeps = max(max_substeps, attempt.n_substeps)
                n_transport_rejections += attempt.n_transport_rejections
                n_calls += 2
                n_steps += 1
                dt_min, dt_max = min(dt_min, dt), max(dt_max, dt)
                if n_steps == 1:
                    first_wall, first_cpu = time.perf_counter() - wall0, time.process_time() - cpu0
                    wall0, cpu0 = time.perf_counter(), time.process_time()
                observer = on_accepted_step if on_accepted_step is not None else _staged_accepted_observer
                if observer is not None and observer(state):
                    stopped = True
                # These containers retain the pickup bed/ledger and transport
                # scratch, even though only the accepted state and accumulated
                # diagnostics are needed now. Release them before evaluating
                # the next full attempt (its RHS would otherwise keep this
                # iteration's objects alive throughout the new allocation).
                del attempt, pickup, deposit, transport, physics, col, route
                del commit, prepared, commit_values, accepted_bed
                if stopped:
                    break
            if stopped:
                break
        # Forced final commit at the configured end (real time), only when the
        # ledger records physical activity since the last commit and MAPLE does
        # not already consider this exact state committed (no no-op commits).
        if boundary_publication:  # CW-B102: published before the forced commit's already-committed check
            publish_carried("boundary")
        if (local_row == boundaries.size - 1 or stopped) and control.force_final_commit:
            bed = state.bed
            ledger = bed.ledger
            activity = to_bool(xp.any(ledger.pending_bed_mass_change_kg != 0.0)
                               | xp.any(ledger.n_physical_touches != 0))
            already = not activity or bed_already_committed(bed, context, t)
            if not already:
                wall_c = time.perf_counter()
                commit, prepared = commit_and_reroute(bed, state.graph, state.storm, t, force=True)
                commit_wall += time.perf_counter() - wall_c
                if commit is not None:
                    rerouted_now = commit.rerouted
                    publish_commit(commit, prepared)
                    if rerouted_now:
                        refresh_hydraulics()
        if boundary_publication:  # a forced commit above keeps the staged journal: publish before on_boundary
            publish_carried("boundary")
        storm_state = state.storm
        hydrograph[row] = xp.stack([
            scalar(t),
            area * xp.sum(cum_rain), area * xp.sum(cum_intake), area * xp.sum(cum_return),
            area * xp.sum(cum_drain), scalar(cum_export),
            area * xp.sum(storm_state.depth_m), area * xp.sum(storm_state.soil_water_m),
            scalar(outlet_q),
            xp.max(storm_state.depth_m), xp.max(last_velocity),
            scalar(float(n_steps)), scalar(float(n_rejected)),
            scalar(dt_min if n_steps else 0.0),
            scalar(max_balance), scalar(max_constitutive),
        ])
        mobile_by_class = state.bed.water.mobile_mass_by_cell_class_kg.sum(axis=(0, 1))
        sediment_hydrograph[row] = _row(xp, [
            scalar(t), xp.sum(mobile_by_class), xp.sum(by_class["actual_pickup"]),
            xp.sum(by_class["deposition_actual"]), xp.sum(by_class["export_actual"]), xp.sum(export_rate),
            scalar(float(state.bed.committed_topography.commit_count)), scalar(float(n_graph_changes)),
            scalar(float(max_substeps)),
        ], [mobile_by_class, export_rate, by_class["actual_pickup"], by_class["deposition_actual"],
            by_class["export_actual"]])
        rows_used = row + 1
        all_boundaries[row] = t
        if stopped or (on_boundary is not None and on_boundary(pack_result())):
            break
    # final consistency: solver depth is the MAPLE depth; discharge consistent with the final graph
    if not to_float(xp.max(xp.abs(state.bed.water.depth_m - state.storm.depth_m))) == 0.0:
        raise SedimentEventError("internal error: final MAPLE water depth differs from the solver depth")
    validate_storm_state(state.graph, state.storm, storm_control.root_tolerance_m, column)
    if state.network.graph_input_sha256 != state.graph.input_sha256:
        raise SedimentEventError("internal error: transport network does not belong to the final graph")
    try:
        _check_phase_against_control(state, control)
    except TransportError as exc:
        raise SedimentEventError(f"internal error: final phase state invalid: {exc}") from None
    if boundary_publication:  # CW-B102: every returned state is published
        publish_carried("return")
    return pack_result()


def water_channel_index() -> int:
    """MAPLE's ledger process index of the water channel (for reports)."""
    from maple.core.types.sediment_ledger import process_index

    return process_index("water_erosion_deposition")


def event_error_types() -> tuple[type[BaseException], ...]:
    """Exception types a runner treats as a refused / failed event."""
    return (SedimentEventError, StormError, BedIntegrationError)
