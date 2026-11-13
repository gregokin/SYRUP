"""E4b: explicit continuous FLOATING-layer vertical water inside the native Pond sediment step (CPU: NumPy and Numba owners; Dense bed).

Opt-in only. `PondEvent(..., floating_vertical=FloatingPondVertical(...))` swaps ONLY the vertical (soil) stage of the existing coupled step; the surface router, wet laws, splash, native pickup/transport/deposit,
net-elevation commit gate, ledgers, quiet monitor and publication are the ONE old body (`PondEvent.step`). A default `PondEvent` (no vertical) is untouched: same repr, config digest and numerics.

Per hydraulic step the vertical is the SAME stage order the fixed-bed layered storm uses (`layered_storm.layered_step`, steps 2-5), on the floating bodies, no new physics:
  pool = H0 + rain dt;  floating preferential (optional) on the pool;  floating Green-Ampt matrix on the remainder (`standing_before = H0`, onset = the actual rain);
  floating gravity step with ZERO added water and ZERO potential evaporation (drainage/redistribution/saturation return, soil clock advances);  pool + saturation return -> `CpuSurfaceRouter` / `NumbaSurfaceRouter`.
It is composed with the genuine surface router by `compose_hydraulic_step(SurfaceStep, ColumnStep)`: a real `infiltration.ColumnStep` of the floating fluxes; no `ColumnParameters`, no zero-K bucket.

State. `HydraulicState.soil_water_m` is ONLY the derived `(ny, nx)` sum over layers of the floating inventory `W (L, N)`; the authority is the stage payload (column view, operators, persistent Green-Ampt history, `W`,
canonical depth `D`) found by the identity of that derived array. A step consumes the payload of its input state and registers the payload of its output, so a rejected or failed trial leaves the accepted
payload untouched (the registry is append-only and weak: payloads die with their states). A state not issued by this stage is refused; the sum is re-checked bit for bit at every step.

Touched-column depth (`after_exchange`). The sediment exchange changes the actual bed. After the real MAPLE pickup/transport/deposit (and splash) and BEFORE the commit gate, quiet monitor and any publication,
the columns with a GROSS exchange (`pickup + deposition > 0` in any class, net-zero turnover included) are re-read with the native canonical depth `D = voxel.sum/(rho A) + active.sum/(rho A)` (the SAME compact reader
as the cold refresh, native NumPy order, no epsilon, no resident floor). Only a column whose `D` actually differs gets a new official geometry view of the prepared column (`floating_soil_layers.floating_geometry_view`,
O(N), every map but the depth shared, a minted generation identity, never a content digest) and only a column whose slab thicknesses or strict bottom flag differ (`native_floating_soil._geometry_changed`, the E3c rule)
runs the zero-time FC cascade/saturation return of the ONE D1 body (`_reconcile_changed`, bounded by the touched set in batches of `batch_cells`): a raw depth change below H alone drains nothing. Returned water joins
the actual local surface depth once and the bottom export leaves once (both enter the water ledger as saturation return / drainage, so the soil and surface closures stay exact); it is routed by the next rain-0 or
rain steps through the ordinary surface stage and the ordinary quiet monitor sees the new depth change (a fresh hold). The Green-Ampt history is SHARED bit for bit through the view (nothing reset, rescaled, copied or hashed).
Properties are held during the event; the between-event texture refresh stays the cold `native_floating_soil` transaction.

Limits (honest). CPU (NumPy/Numba) Dense bed only: a windowed bed (its pending-journal depth semantics belong to the separate base investigation) and CUDA are refused with a typed error, never downgraded; the GPU definitions
live in `tests/phase2_floating_pond/test_gpu_definitions.py` and are unrun. `PondEvent.finalize` (the dry reset) is refused for a floating vertical; the continuing retained route is the only one. No restart/daily/output contract.
"""
from __future__ import annotations

import contextlib
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import floating_matrix_infiltration as fmi
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs
from maple_syrup.experimental_hydrology import (
    CpuSurfaceRouter,
    HydraulicControl,
    HydraulicState,
    compose_hydraulic_step,
    validate_hydraulic_state,
)
from maple_syrup.infiltration import ColumnStep
from maple_syrup.pond_hydraulics import (
    PondHydraulics,
    PondHydraulicsError,
    metadata_routing_graph,
)
from maple_syrup.pond_network import PondTopology
from maple_syrup.surface_routing import check_state_time

COMPONENT_TAG = "floating-pond-vertical-v1"
SUPPORTED_BACKENDS = ("numpy", "numba", "cuda")


class FloatingPondError(RuntimeError):
    """Typed refusal of the floating Pond vertical (never a silent downgrade)."""


@dataclass(frozen=True, eq=False)
class _Payload:
    column: Any
    matrix_op: Any
    history: Any
    pref_op: Any
    state: Any  # FloatingSoilState
    depth_m: np.ndarray  # (N,) canonical depth of the actual bed this column view was built from (sealed)


@dataclass(frozen=True, eq=False)
class HeldTemplate:
    """The HELD law context of a vertical (R1a): the prepared root column / matrix operator / optional preferential operator the vertical was built on, the depth that column was prepared with, the layer-0 conductivity record and
    the durable owner facts. It carries NO water and NO history: it is the template a resumed current payload is built on, never a second water budget."""

    column: Any
    matrix_op: Any
    preferential: Any
    depth0_m: np.ndarray
    ks_m_s: Any
    config_sha256: str
    generation: int
    source_witness: str


@dataclass
class FloatingPondStats:
    """Counters of the hot path (host ints; never part of a state)."""

    steps: int = 0
    after_exchange_calls: int = 0
    touched_columns: int = 0
    depth_changed_columns: int = 0
    reconciled_columns: int = 0
    views_built: int = 0
    surface_return_m3: float = 0.0
    bottom_export_m3: float = 0.0
    peak_registry: int = 0


@dataclass(frozen=True, eq=False)
class EventEndBundle:
    """Read-only references (no copies) to the current payload and surface of a completed event; see `FloatingPondVertical.event_end_bundle`."""

    payload: Any
    depth_m: Any
    qx_m2_s: Any
    qy_m2_s: Any
    bed: Any
    held: Any
    capability: Any


class FloatingPondVertical:
    """The explicit floating vertical stage (see the module docstring). `owner` is the CURRENT native floating owner (`native_floating_soil.NativeFloatingSoil`, numpy/numba); `preferential` an optional
    `FloatingPreferentialOperator` prepared on `owner.column`; `available_bytes_fn` is REQUIRED (no unlimited reader) and is read before every composed admission."""

    _opened = False  # EB3: True after `detach_initial` (the accepted event open): no initial payload API and no owner
    _root_facts: Any = None
    _active_ref: Any = None
    _boundary_bed: Any = None

    def __init__(self, owner: Any, *, available_bytes_fn: Callable[[], int], preferential: Any = None, reserve_bytes: int = 0, batch_cells: int = nfs.DEFAULT_BATCH_CELLS,
                 memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None):
        if type(owner) is not nfs.NativeFloatingSoil or not owner.is_current:
            raise FloatingPondError("owner must be the CURRENT native floating owner (a stale or foreign owner is refused)")
        if owner.column.backend not in SUPPORTED_BACKENDS:
            raise FloatingPondError(f"the floating Pond vertical serves the {SUPPORTED_BACKENDS} owners; {owner.column.backend!r} needs hardware qualification and is refused")
        if not callable(available_bytes_fn):
            raise FloatingPondError("available_bytes_fn must be callable (the composed admission needs a real reader)")
        if preferential is not None and (type(preferential) is not fpi.FloatingPreferentialOperator or preferential.column is not owner.column):
            raise FloatingPondError("preferential must be a FloatingPreferentialOperator prepared on the owner's column")
        self.config = owner.config
        self.backend = owner.column.backend
        self._provisional: tuple | None = None  # R1a resume candidate under validation: initialised BEFORE the CPU/CUDA branches (the CUDA constructor returns early), so `_payload` never sees a missing attribute
        if self.backend != "cuda" and (memory_budget_bytes is not None or host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None):
            raise FloatingPondError("memory_budget_bytes / host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to the cuda backend (a meaningless CPU argument is refused)")
        self.available_bytes_fn = available_bytes_fn
        if self.backend == "cuda":
            # two separate pools (device: available_bytes_fn / reserve_bytes / memory_budget_bytes; host: host_*), real default readers when None, reserves counted ONCE by `fcu.admit`
            quota = {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": fsl._count(reserve_bytes, "reserve_bytes"), "available_bytes_fn": available_bytes_fn,
                     "host_memory_budget_bytes": host_memory_budget_bytes, "host_reserve_bytes": host_reserve_bytes, "host_available_bytes_fn": host_available_bytes_fn}
            self.n_layers, self.n_cells, self.area = owner.column.n_layers, owner.column.n_cells, float(owner.column.cell_area_m2)
            self.stats = FloatingPondStats()
            self.shape = None
            from maple_syrup import floating_pond_cuda as fpc_

            fpc_.init_cuda(self, owner, preferential, quota, batch_cells)
            c = owner.column
            self.identity = "|".join((COMPONENT_TAG, c.sha256, owner.matrix_op.sha256, "" if preferential is None else preferential.sha256, self.backend))
            self._registry = {}
            self._initial = _Payload(c, owner.matrix_op, owner.matrix_history, preferential, owner.state, self._initial_depth)
            return
        self.reserve_bytes = fsl._count(reserve_bytes, "reserve_bytes")
        self.batch_cells = fsl._count(batch_cells, "batch_cells", minimum=1)
        c = owner.column
        self.n_layers, self.n_cells, self.area = c.n_layers, c.n_cells, float(c.cell_area_m2)
        L, N = self.n_layers, self.n_cells
        need = self._set_cpu_models(c, preferential)
        self._read_and_admit(need, "the floating Pond vertical (retained stage buffers, live payloads and the operation peak)")
        if preferential is not None and (type(preferential) is not fpi.FloatingPreferentialOperator or preferential.column is not owner.column):
            raise FloatingPondError("preferential must be a FloatingPreferentialOperator prepared on the owner's column")
        self._owner = owner
        self.stats = FloatingPondStats()
        self.shape: tuple[int, int] | None = None
        self.xp = np
        self.identity = "|".join((COMPONENT_TAG, c.sha256, owner.matrix_op.sha256, "" if preferential is None else preferential.sha256, self.backend))
        self._registry: dict[int, tuple[Any, _Payload]] = {}
        self._provisional: tuple | None = None  # (id, weakref, payload) of a resume candidate under validation (R1a); never part of the registry
        depth0 = np.array(owner.depth_m, dtype=np.float64, copy=True)
        depth0.flags.writeable = False
        self._initial = _Payload(c, owner.matrix_op, owner.matrix_history, preferential, owner.state, depth0)
        self._zero_added = np.zeros((L, N))
        self._zero_pet = np.zeros(N)
        self._zero_added.flags.writeable = False
        self._zero_pet.flags.writeable = False
        self._depth_scratch = np.zeros(N)

    @classmethod
    def from_held_context(cls, context: Any, *, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0, batch_cells: int = nfs.DEFAULT_BATCH_CELLS, memory_budget_bytes: Any = None,
                          host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingPondVertical:
        """Vertical (CPU or CUDA, by the context's own backend) from a genuine WATER-FREE H0 `HeldContext` (`/1` CPU, `/2` CUDA): no `NativeFloatingSoil`, no initial W/J/I, no bed read, no property provider. The vertical has NO
        initial payload: `initial_state`, `initial_soil_water` and `check_initial` are typed refusals; its current payload can come only from a verified restore (`floating_resume` for CPU owners, `floating_resume_cuda` for CUDA
        owners: a prepared record, `resume_scope`, `adopt_resumed`). The context is referenced (immutable by convention) and is neither file provenance nor a bed proof. A CPU context strictly rejects every CUDA-only
        argument; a CUDA context needs the current device and two separate pools (device / host readers and reserves, counted once). Admission precedes every owned array."""
        from maple_syrup import floating_held_context as fhc

        fhc.require_held_context(context)  # exact class + private minted registry BEFORE any read
        backend = context.column.backend
        if backend not in SUPPORTED_BACKENDS:
            raise FloatingPondError(f"from_held_context serves the {SUPPORTED_BACKENDS} contexts; {backend!r} is refused before any transfer")
        if not callable(available_bytes_fn):
            raise FloatingPondError("available_bytes_fn must be callable (the composed admission needs a real reader)")
        if backend != "cuda" and (memory_budget_bytes is not None or host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None):
            raise FloatingPondError("memory_budget_bytes / host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to a CUDA context (a meaningless CPU argument is refused)")
        if backend == "cuda" and context.backend != "cuda":
            raise FloatingPondError("the context is not a CUDA held context")
        c, pref = context.column, context.preferential
        self = object.__new__(cls)
        self.config = context.config
        self.backend = c.backend
        self._provisional = None
        self.available_bytes_fn = available_bytes_fn
        self._owner = None
        self._held = context
        self._initial = None
        self._registry = {}
        self.stats = FloatingPondStats()
        self.shape = None
        self.n_layers, self.n_cells, self.area = c.n_layers, c.n_cells, float(c.cell_area_m2)
        self.identity = "|".join((COMPONENT_TAG, c.sha256, context.matrix_op.sha256, "" if pref is None else pref.sha256, self.backend))
        if backend == "cuda":  # all shared attributes are set ABOVE the branch; the CUDA setup is the water-free half of the owner constructor
            quota = {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": fsl._count(reserve_bytes, "reserve_bytes"), "available_bytes_fn": available_bytes_fn,
                     "host_memory_budget_bytes": host_memory_budget_bytes, "host_reserve_bytes": fsl._count(host_reserve_bytes, "host_reserve_bytes"), "host_available_bytes_fn": host_available_bytes_fn}
            from maple_syrup import floating_pond_cuda as fpc_

            fpc_.init_cuda_held(self, context, quota, batch_cells)
            return self
        self.reserve_bytes = fsl._count(reserve_bytes, "reserve_bytes")
        self.batch_cells = fsl._count(batch_cells, "batch_cells", minimum=1)
        self.n_layers, self.n_cells, self.area = c.n_layers, c.n_cells, float(c.cell_area_m2)
        self.last_exchange_required = 0
        self._set_cpu_models(c, pref)
        self.retained_bytes = (self.n_layers * self.n_cells + 2 * self.n_cells) * 8  # zero rows + scratch; no sealed initial depth copy (D0 is referenced) and no initial payload
        need = self.retained_bytes + 3 * self.payload_bytes + max(self.stage_step_bytes, self.exchange_bytes) + self.reserve_bytes
        self.construction_bytes = need
        self._read_and_admit(need, "the held-context floating Pond vertical (retained stage buffers, live payloads and the operation peak)")
        self.xp = np
        L, N = self.n_layers, self.n_cells
        self._zero_added = np.zeros((L, N))
        self._zero_pet = np.zeros(N)
        self._zero_added.flags.writeable = False
        self._zero_pet.flags.writeable = False
        self._depth_scratch = np.zeros(N)
        return self


    def _no_initial(self, what: str) -> None:
        if self._initial is None or self._opened:
            raise FloatingPondError(f"{what} is unavailable on a held-context vertical: it has no initial payload (its current water exists only through a verified restore adoption)")

    def _set_cpu_models(self, c: Any, preferential: Any) -> int:
        """The composed CPU memory model (metadata only; shared by the owner constructor and `from_held_context`): sets the byte attributes and returns the construction need."""
        L, N = self.n_layers, self.n_cells
        be = self.backend
        # ---- composed memory model (metadata only), admitted from the REAL reader BEFORE any owned array exists (module docstring: "Memory")
        mm = fmi.estimate_floating_matrix_memory(L, N, backend=be, operation="step")
        pm = None if preferential is None else fpi.estimate_floating_preferential_memory(L, N, backend=be, operation="step", n_species=preferential.n_species)
        cm = int(c.estimate.step_bytes)
        B = min(self.batch_cells, N)
        est_b = fsl.estimate_floating_memory(L, B, layouts=dict(c.layouts), diffusion_mode=c.diffusion_mode, backend=be)
        payload = (L + 2) * N * 8  # one live payload: W (L, N), canonical depth N, the derived sum N
        self.retained_bytes = (L * N + 3 * N) * 8 + N * 8  # zero_added, zero_pet, depth scratch, sealed initial depth copy
        self.payload_bytes = payload
        # one step: the pool/rain/h0 rows and the ColumnStep fields (about 8 N), the three stage results alive at once (their water copies and rows) and the largest transient scratch, plus the surface result (added at bind)
        self.stage_step_bytes = 8 * N * 8 + mm.outputs_bytes + (0 if pm is None else pm.outputs_bytes) + max(mm.scratch_bytes, 0 if pm is None else pm.scratch_bytes) + cm + 2 * payload
        # one hot exchange: indices/touched/depth rows (about 8 N), new depth map + view maps, the staged water and surface/export rows (L N + 2 N), the continued-state validation and one compact old/new sub-column pair
        self.exchange_bytes = 12 * N * 8 + 2 * L * N * 8 + cm + 2 * est_b.retained_bytes + est_b.step_bytes + payload
        self.surface_step_bytes = 0
        self.last_exchange_required = 0
        self.step_bytes = self.stage_step_bytes + self.reserve_bytes
        need = self.retained_bytes + 3 * payload + max(self.stage_step_bytes, self.exchange_bytes) + self.reserve_bytes
        self.construction_bytes = need
        return need

    def _read_and_admit(self, need: int, what: str) -> None:
        if self.backend == "cuda":
            from maple_syrup import floating_pond_cuda as fpc_

            fpc_.admit(self, need, what)  # current device first; device and host pools separately
            return
        avail = self.available_bytes_fn()
        if isinstance(avail, (bool, np.bool_)) or not isinstance(avail, (int, np.integer)) or int(avail) < 0:
            raise FloatingPondError(f"the available-bytes reader must return a non-Boolean integer >= 0, got {avail!r}")
        if need > int(avail):
            raise nfs.NativeFloatingMemoryError(f"{what} needs an estimated {need} B but only {int(avail)} B are reported available; nothing was allocated")

    # -- payload registry ------------------------------------------------------------------------------------------
    def _register(self, array: np.ndarray, payload: _Payload) -> None:
        key = id(array)

        def drop(_ref: Any, key: int = key, registry: dict = self._registry) -> None:
            registry.pop(key, None)

        self._registry[key] = (weakref.ref(array, drop), payload)
        self.stats.peak_registry = max(self.stats.peak_registry, len(self._registry))

    def _payload(self, soil_water_m: Any) -> _Payload:
        entry = self._registry.get(id(soil_water_m))
        prov = self._provisional
        if entry is None and prov is not None and prov[0] == id(soil_water_m) and prov[1]() is soil_water_m:
            return prov[2]
        if entry is None or entry[0]() is not soil_water_m:
            raise FloatingPondError("the soil_water_m of this state was not issued by this floating stage (the authority is the stage's floating inventory; a foreign or stale state is refused)")
        return entry[1]

    def stage_kwargs(self) -> dict:
        """Admission keywords of the component stage calls: CPU `available_bytes_fn`/`reserve_bytes`; CUDA the full two-pool quota (device and host, real defaults when None)."""
        if self.backend == "cuda":
            return dict(self.quota)
        return {"available_bytes_fn": self.available_bytes_fn, "reserve_bytes": self.reserve_bytes}

    def _active_host(self) -> np.ndarray:
        """The host active-mask pattern: the owner's own map on CPU, the sealed cold host record on CUDA (never a hot download)."""
        if self.backend == "cuda":
            return self._host_active
        if self._active_ref is not None:
            return self._active_ref
        return self._owner.column._maps["active"] if self._owner is not None else self._held.active

    def _derived(self, state: Any) -> np.ndarray:
        return state.water_m.sum(axis=0).reshape(self.shape)

    def _admit_step(self) -> None:
        self._read_and_admit(self.step_bytes, "the composed floating Pond step")

    # -- the hydraulic solver this stage plugs into the Pond ------------------------------------------------------------
    def build_hydraulics(self, topology: PondTopology, *, control: HydraulicControl, available_bytes_fn: Callable[[], int] | None = None) -> PondHydraulics:
        if topology.shape[0] * topology.shape[1] != self.n_cells:
            raise FloatingPondError("the topology does not match the owner's cell count")
        if not np.array_equal(np.asarray(topology.active).reshape(-1), self._active_host()):
            raise FloatingPondError("the topology's active mask differs from the owner's active-mask pattern")
        self.shape = (int(topology.shape[0]), int(topology.shape[1]))
        solver = _FloatingSolver(self, topology, control)
        graph = metadata_routing_graph(topology, np)
        return PondHydraulics(topology=topology, solver=solver, backend=self.backend, params=None, control=control, graph=graph, graph_kind="metadata")

    # -- hot touched-column depth ----------------------------------------------------------------------------------------
    def after_exchange(self, bed: Any, context: Any, hyd: HydraulicState, touched: np.ndarray, active: np.ndarray) -> tuple[HydraulicState, dict]:
        """Stage the actual depth of the GROSS-touched columns after the real MAPLE exchange (module docstring). `touched` and `active` are host `(ny, nx)` bool maps (the gross exchange map and the topology's active mask). Pure: returns the successor state (the same object when nothing
        changed) and a host receipt dict; the accepted payload of `hyd` is never mutated."""
        if self.backend == "cuda":
            from maple_syrup import floating_pond_cuda as fpc_

            return fpc_.after_exchange_cuda(self, bed, context, hyd, touched, active)
        self.stats.after_exchange_calls += 1
        payload = self._payload(hyd.soil_water_m)
        active = np.asarray(active, dtype=bool)
        nb = nfs.NativeBed(bed.voxel_column, bed.active_layer, context.geometry, context.grain_classes, context.mass_resolution_kg, active)
        windowed = nb.representation == "windowed"
        # metadata only. Dense: the compact reader's own peak (live with the exchange arrays). Windowed: NO deep read and no voxel-depth scan -- the native prefix packets decode one whole-grid NumPy-order total and the
        # active-layer thickness (about four (ny, nx) float64 arrays alive together)
        compact = 4 * self.n_cells * 8 if windowed else int(nfs._estimate(None, nb, self.config, self.batch_cells, 0)["compact_bytes"])
        self.last_exchange_required = self.exchange_bytes + compact + self.reserve_bytes
        self._read_and_admit(self.last_exchange_required, "the floating touched-column exchange stage (stage arrays + compact reader)")
        idx = np.flatnonzero((touched & active).reshape(-1)).astype(np.int64)
        receipt = {"n_touched": int(idx.size), "n_depth_changed": 0, "n_reconciled": 0, "surface_return_m3": 0.0, "bottom_export_m3": 0.0}
        self.stats.touched_columns += int(idx.size)
        if idx.size == 0:
            return hyd, {**receipt, "surface_return_m": None, "bottom_export_m": None}
        if windowed:
            # HOT native Windowed depth: the genuine native prefix packets (`windowed_committed_surface_elevation_m`: NumPy-order decoded column total / rho / area + the native active-layer thickness; bitwise the Dense
            # NumPy diagnostic, no resident floor, no copied decoder). `trusted_packet=True` is valid only along the native producer chain: the cold full-validated initial bed (check_initial) and the Pond's own
            # native descendants; the packet's source/base/store-version binding is still checked by the native call. An arbitrary imported pending bed never reaches here (refused at the cold boundary).
            from maple.surface.topographic_commit.windowed_commit import (
                windowed_committed_surface_elevation_m,
            )

            surface = windowed_committed_surface_elevation_m(bed.voxel_column, bed.active_layer, context.geometry, trusted_packet=True)
            d_new = np.array(np.asarray(surface).reshape(-1)[idx], dtype=np.float64, copy=True)
            del surface
        else:
            d_new = nfs.read_canonical_depth(nb, self.config, idx, self._depth_scratch, batch_cells=self.batch_cells, available_bytes_fn=self.available_bytes_fn, reserve_bytes=self.reserve_bytes)
        d_old = payload.depth_m[idx]
        moved = d_new != d_old
        receipt["n_depth_changed"] = int(moved.sum())
        self.stats.depth_changed_columns += receipt["n_depth_changed"]
        if not moved.any():
            return hyd, {**receipt, "surface_return_m": None, "bottom_export_m": None}
        ridx, rnew = idx[moved], d_new[moved]
        depth_full = np.array(payload.depth_m, copy=True)
        depth_full[ridx] = rnew
        view = fsl.floating_geometry_view(payload.column, depth_full, available_bytes_fn=self.available_bytes_fn, reserve_bytes=self.reserve_bytes)
        self.stats.views_built += 1
        hydraulic = nfs._geometry_changed(self.config, d_old[moved], rnew)
        cidx = ridx[hydraulic]
        water = payload.state.water_m
        surf = np.zeros(self.n_cells)
        exp_ = np.zeros(self.n_cells)
        if cidx.size:
            m = payload.column._maps
            water, surf, exp_, rec = nfs._reconcile_changed(self.config, m, payload.depth_m, payload.state.water_m, m["theta_field_capacity"], m["theta_saturation"], depth_full, cidx, self.area,
                                                             self.batch_cells, payload.state.t_s)
            receipt["n_reconciled"] = int(cidx.size)
            receipt["surface_return_m3"], receipt["bottom_export_m3"] = rec["surface_return_m3"], rec["bottom_export_m3"]
            self.stats.reconciled_columns += int(cidx.size)
            self.stats.surface_return_m3 += rec["surface_return_m3"]
            self.stats.bottom_export_m3 += rec["bottom_export_m3"]
        state = fsl.floating_continued_state(view, water, payload.state.t_s)
        op, hist = fmi.rebind_matrix_geometry_view(payload.matrix_op, view, payload.history)
        pref = None if payload.pref_op is None else fpi.rebind_preferential_geometry_view(payload.pref_op, view)
        depth_full.flags.writeable = False
        new = _Payload(view, op, hist, pref, state, depth_full)
        soil = self._derived(state)
        ny, nx = self.shape
        depth = np.array(hyd.depth_m, copy=True)
        if cidx.size:
            depth = depth + surf.reshape(ny, nx)  # the returned water joins the actual local surface ONCE
        self._register(soil, new)
        successor = HydraulicState(hyd.t_s, depth, soil, hyd.qx_m2_s, hyd.qy_m2_s, hyd.next_dt_cap_s)
        return successor, {**receipt, "surface_return_m": surf.reshape(ny, nx), "bottom_export_m": exp_.reshape(ny, nx)}

    def bind_shape(self, shape: tuple[int, int]) -> None:
        """Bind the `(ny, nx)` grid of the owner's flat cells (called by the `PondEvent` constructor; a second, different shape is refused)."""
        shape = (int(shape[0]), int(shape[1]))
        if shape[0] * shape[1] != self.n_cells or (self.shape is not None and self.shape != shape):
            raise FloatingPondError(f"the grid {shape} does not match the owner's {self.n_cells} cells (or another grid is already bound)")
        if self.shape is None:
            from maple_syrup.surface_routing import estimate_surface_memory

            sm = estimate_surface_memory("local_inertial", shape, backend="numpy", limiter="off", reserve_bytes=0)
            self._read_and_admit(self.retained_bytes + 3 * self.payload_bytes + self.stage_step_bytes + sm.step_allocation_bytes + self.reserve_bytes, "the floating Pond step with the surface result")
            self.surface_step_bytes = int(sm.step_allocation_bytes)
            self.step_bytes = self.stage_step_bytes + self.surface_step_bytes + self.reserve_bytes
        self.shape = shape

    def check_initial(self, bed: Any, context: Any, t_s: float) -> None:
        """Cold initialization handoff (ONE source scan): the owner must still be CURRENT, the event clock must equal the owner's physical clock exactly (nothing is snapped or advanced), and the ACTUAL bed
        (voxel/active values, active-mask pattern, geometry, ordered grain classes, rho, area, mass resolution) must have the native source witness this owner was accepted against."""
        self._no_initial("check_initial")
        if self._boundary_bed is not None:  # EB3: a vertical built from an accepted boundary: the ONE current bed is the boundary's own (reference identity: an execution epoch, not a content hash)
            bb = self._boundary_bed
            if self.backend == "cuda" and not getattr(self, "_preflighted", False):
                from maple_syrup import floating_pond_cuda as fpc_

                fpc_.preflight_depth(self, bed)  # metadata-only native-order qualification of the CURRENT bed extents (a Dense V may have grown since the cold open)
                self._preflighted = True
            if bed.voxel_column is not bb.voxel_column or bed.active_layer is not bb.active_layer or float(t_s) != float(self._initial.state.t_s):
                raise FloatingPondError("the bed/clock handed to the event is not the accepted boundary's current bed and clock (a modified bed needs its own adoption: later slice)")
            return
        if not self._owner.is_current:
            raise FloatingPondError("the native floating owner is no longer current (superseded by a refresh); a new vertical is needed")
        if self.backend == "cuda" and not getattr(self, "_preflighted", False):
            from maple_syrup import floating_pond_cuda as fpc_

            fpc_.preflight_depth(self, bed)  # cold, once, before any physical mutation or witness scan (Dense and Windowed)
            self._preflighted = True
        t_own = float(self._initial.state.t_s)
        if float(t_s) != t_own:
            raise FloatingPondError(f"the event initial clock {t_s!r} differs from the owner's physical clock {t_own!r}; no clock is snapped or advanced (a mismatch is refused before any work)")
        nb = nfs.NativeBed(bed.voxel_column, bed.active_layer, context.geometry, context.grain_classes, context.mass_resolution_kg, np.array(self._active_host().reshape(self.shape)))
        if self.backend == "cuda":  # the cold source scan is a HOST allocation: the host pool (explicit host reader, else the real default host reader) and the host reserve, never the device reader
            from maple_syrup import spatial_soil_layers as sp_

            reader = self.quota["host_available_bytes_fn"] or sp_.default_host_available_bytes
            witness = nfs.native_source_witness(nb, self.config, batch_cells=self.batch_cells, available_bytes_fn=reader, reserve_bytes=fsl._count(self.quota["host_reserve_bytes"], "host_reserve_bytes"))
        else:
            witness = nfs.native_source_witness(nb, self.config, batch_cells=self.batch_cells, available_bytes_fn=self.available_bytes_fn, reserve_bytes=self.reserve_bytes)
        if witness != self._owner.binding.source_witness:
            raise FloatingPondError("the bed is not the native source this owner was accepted against (cold witness mismatch: values, active-mask pattern, geometry or grain descriptor differ)")

    def current_water(self, soil_water_m: Any) -> np.ndarray:
        """Detached host copy of the floating inventory `(L, N)` behind a derived state (diagnostics/tests)."""
        w = self._payload(soil_water_m).state.water_m
        if self.backend == "cuda":
            from maple.core.backend import to_host

            return np.array(to_host(w), copy=True)  # a counted diagnostic download (never a hot path)
        return np.array(w, copy=True)

    def current_payload(self, soil_water_m: Any) -> _Payload:
        return self._payload(soil_water_m)

    def held_template(self) -> HeldTemplate:
        """The HELD template of this vertical (sealed references; see `HeldTemplate`). Read-only: no payload is created or registered."""
        if self._root_facts is not None:
            return self._root_facts
        if self._owner is None:
            h = self._held
            return HeldTemplate(h.column, h.matrix_op, h.preferential, h.depth0_m, h.ks_m_s, h.config.sha256, int(h.generation), h.source_witness)
        o = self._owner
        return HeldTemplate(self._initial.column, self._initial.matrix_op, self._initial.pref_op, self._initial.depth_m, o.ks_m_s, o.config.sha256, int(o.binding.generation), o.binding.source_witness)

    def _resume_module(self) -> Any:
        """The module owning the prepared-resume record of this vertical's backend: `floating_resume` (CPU host arrays) or `floating_resume_cuda` (device arrays). The two records never substitute for one another."""
        if self.backend == "cuda":
            from maple_syrup import floating_resume_cuda as frc

            return frc
        from maple_syrup import floating_resume as fr

        return fr


    @contextlib.contextmanager
    def resume_scope(self, prepared: Any):
        """PROVISIONAL resolution of a prepared resume payload for the duration of the candidate validation (event `_check_state` + `audit`): the registry is NOT touched, nothing outlives the block and nothing relies on garbage collection.
        Requires a genuinely minted, unconsumed prepared record bound to THIS vertical (`floating_resume`)."""
        fr = self._resume_module()
        parts = fr.prepared_parts(prepared, self)
        if self._provisional is not None:
            raise FloatingPondError("a resume scope is already open on this vertical")
        self._provisional = (id(parts.soil), weakref.ref(parts.soil), _Payload(parts.column, parts.matrix_op, parts.history, parts.preferential, parts.state, parts.depth_m))
        try:
            yield
        finally:
            self._provisional = None


    def adopt_resumed(self, prepared: Any) -> None:
        """The ONE registration of a resumed payload (LAST step of `floating_resume.restore_floating_pond`, after the candidate state passed the event's full validation and audits). The only accepted argument is a genuinely minted, unconsumed
        prepared-resume record bound to THIS vertical, whose parts `floating_resume.prepared_parts` re-verifies exactly against the capture and the held template (foreign root/law/history/state/geometry are refused); free keyword parts are not accepted."""
        fr = self._resume_module()
        parts = fr.prepared_parts(prepared, self)  # every refusal precedes the registration
        self._register(parts.soil, _Payload(parts.column, parts.matrix_op, parts.history, parts.preferential, parts.state, parts.depth_m))
        fr.mark_consumed(prepared)


    def event_end_bundle(self, cap: Any, state: Any, event: Any) -> EventEndBundle:
        """READ-ONLY references to the actual CURRENT payload of a genuinely COMPLETED event (`floating_completion`): W/J/I/D/law (`current_payload`), the surface depth/face fluxes, the bed and the held template. Verifies the exact capability / state /
        event binding and that `event` runs THIS vertical; creates no owner, water, registry entry or copy, consumes nothing and does not adopt the payload (adoption repeats the full physical checks later)."""
        from maple_syrup import floating_completion as fcm

        if getattr(event, "vertical", None) is not self:
            raise fcm.CompletionError("the event does not run this floating vertical")
        fcm.verify_completion(cap, state, event=event)
        h = state.hydraulic
        return EventEndBundle(self.current_payload(h.soil_water_m), h.depth_m, h.qx_m2_s, h.qy_m2_s, state.bed, self.held_template(), cap)

    def detach_initial(self) -> None:
        """The OWNING hook of the accepted event open (`floating_event_lifecycle.open_floating_event`, called LAST after the opener's full validation and the owner's retirement): the vertical forgets the original native owner and
        its initial water. The root facts (held root column / matrix operator / preferential operator / D0 / Ks / config sha / generation / cold source anchor) stay as a plain `HeldTemplate`; the registered CURRENT initial payload
        (the ONE water) stays in the registry until its soil array is released; no reference to W0 remains here. `initial_state` / `initial_soil_water` / `check_initial` become typed refusals."""
        if self._owner is None or self._initial is None or self._opened:
            raise FloatingPondError("detach_initial serves a vertical that still holds its native owner")
        o, i = self._owner, self._initial
        self._root_facts = HeldTemplate(i.column, i.matrix_op, i.pref_op, i.depth_m, o.ks_m_s, o.config.sha256, int(o.binding.generation), o.binding.source_witness)
        self._active_ref = o.column._maps["active"]
        self._initial = _Payload(i.column, i.matrix_op, None, i.pref_op, None, i.depth_m)  # root facts only: no water, no history
        self._owner = None
        self._opened = True

    def release_boundary_initial(self) -> None:
        """OWNING hook of an accepted next-occurrence open (`floating_event_lifecycle.begin_occurrence`, after the actual `continued_state` succeeded): the vertical forgets the boundary's initial W/J/I and the boundary bed it was built on
        (same water-free root-facts strategy as `detach_initial`); the registered CURRENT payload (the ONE water) stays in the registry. Pure reference assignments: nothing allocates, reads or raises after the validation."""
        if self._boundary_bed is None or self._initial is None or self._opened:
            raise FloatingPondError("release_boundary_initial serves a boundary vertical that still holds its initial payload")
        i = self._initial
        self._initial = _Payload(i.column, i.matrix_op, None, i.pref_op, None, i.depth_m)
        self._boundary_bed = None
        self._opened = True

    @classmethod
    def from_boundary(cls, root: Any, *, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0, batch_cells: int = nfs.DEFAULT_BATCH_CELLS, memory_budget_bytes: Any = None,
                      host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingPondVertical:
        """CPU vertical for the NEXT native occurrence from an accepted boundary root (`floating_event_lifecycle.BoundaryRoot`: the prepared root column / operators of the refreshed property generation, the CURRENT W / J / I / D
        payload and the boundary's current bed). No `NativeFloatingSoil`: the initial payload is the boundary's current inventory (the ONE water) and the bed handed to the event must be the boundary's own bed."""
        from maple_syrup import floating_event_lifecycle as lc

        if type(root) is not lc.BoundaryRoot or root not in lc._ROOTS:
            raise FloatingPondError("root must be a genuine BoundaryRoot minted by the event lifecycle")
        if root.column.backend not in SUPPORTED_BACKENDS:
            raise FloatingPondError(f"from_boundary serves the {SUPPORTED_BACKENDS} verticals; {root.column.backend!r} is refused before any transfer")
        if not callable(available_bytes_fn):
            raise FloatingPondError("available_bytes_fn must be callable (the composed admission needs a real reader)")
        if root.column.backend != "cuda" and (memory_budget_bytes is not None or host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None):
            raise FloatingPondError("memory_budget_bytes / host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to a CUDA boundary (a meaningless CPU argument is refused)")
        c, pref = root.column, root.pref_op
        self = object.__new__(cls)
        self.config, self.backend, self._provisional = root.config, c.backend, None
        self.available_bytes_fn = available_bytes_fn
        self._owner, self._held, self._opened = None, None, False
        self._root_facts = HeldTemplate(c, root.matrix_op, pref, root.depth0_m, root.ks_m_s, root.config.sha256, int(root.generation), root.source_witness)
        self._active_ref = None if c.backend == "cuda" else c._maps["active"]
        self._boundary_bed = root.bed
        self._registry = {}
        self.stats = FloatingPondStats()
        self.shape = None
        self.identity = "|".join((COMPONENT_TAG, c.sha256, root.matrix_op.sha256, "" if pref is None else pref.sha256, self.backend))
        self.n_layers, self.n_cells, self.area = c.n_layers, c.n_cells, float(c.cell_area_m2)  # the shared metadata BEFORE the CUDA branch (as the owner / held-context constructors: `_set_models` does not set them)
        if c.backend == "cuda":  # the water-free CUDA setup (device gate FIRST, one two-pool admission before any owned array); only the payload below is boundary-specific
            quota = {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": fsl._count(reserve_bytes, "reserve_bytes"), "available_bytes_fn": available_bytes_fn,
                     "host_memory_budget_bytes": host_memory_budget_bytes, "host_reserve_bytes": fsl._count(host_reserve_bytes, "host_reserve_bytes"), "host_available_bytes_fn": host_available_bytes_fn}
            from maple_syrup import floating_pond_cuda as fpc_

            fpc_.init_cuda_boundary(self, root, quota, batch_cells)
            self._initial = _Payload(c, root.matrix_op, root.history, pref, root.water_state, root.depth_m)
            return self
        self.reserve_bytes = fsl._count(reserve_bytes, "reserve_bytes")
        self.batch_cells = fsl._count(batch_cells, "batch_cells", minimum=1)
        self.n_layers, self.n_cells, self.area = c.n_layers, c.n_cells, float(c.cell_area_m2)
        self.last_exchange_required = 0
        self._set_cpu_models(c, pref)
        self.retained_bytes = (self.n_layers * self.n_cells + 2 * self.n_cells) * 8
        need = self.retained_bytes + 3 * self.payload_bytes + max(self.stage_step_bytes, self.exchange_bytes) + self.reserve_bytes
        self.construction_bytes = need
        self._read_and_admit(need, "the boundary floating Pond vertical (retained stage buffers, live payloads and the operation peak)")
        self.xp = np
        L, N = self.n_layers, self.n_cells
        self._zero_added, self._zero_pet, self._depth_scratch = np.zeros((L, N)), np.zeros(N), np.zeros(N)
        self._zero_added.flags.writeable = False
        self._zero_pet.flags.writeable = False
        self._initial = _Payload(c, root.matrix_op, root.history, pref, root.water_state, root.depth_m)
        return self

    def initial_soil_water(self) -> np.ndarray:
        """The derived `(ny, nx)` sum of the owner's accepted floating inventory (what `PondEvent.initial_state` expects)."""
        self._no_initial("initial_soil_water")
        if self.shape is None:
            raise FloatingPondError("bind the stage to a topology (build the PondEvent) before asking for the derived soil water")
        return self._derived(self._initial.state)


class _FloatingSolver:
    """Pond solver: the genuine surface router (CPU: NumPy/Numba; CUDA: `CudaSurfaceRouter`) + the floating vertical stage (API of `PondHydraulics.solver`)."""

    implementation = "floating"

    def __init__(self, vertical: FloatingPondVertical, topology: PondTopology, control: HydraulicControl):
        self.floating_vertical = vertical
        self.vertical = vertical
        self.xp = vertical.xp if vertical.backend == "cuda" else np
        self.topology, self.control = topology, control
        self.shape = tuple(topology.shape)
        self.method = "local_inertial"
        graph = metadata_routing_graph(topology, self.xp)
        if vertical.backend == "cuda":
            from maple_syrup.experimental_cuda import CudaSurfaceRouter

            # the router's own admission reads the DEVICE reader of the caller (default: the context's device driver reading); the vertical's composed admission already counted its result
            self.surface = CudaSurfaceRouter("local_inertial", graph, geometry=topology.geometry, control=control, available_bytes_fn=vertical.available_bytes_fn)
            self.geometry = topology.geometry
        else:
            if vertical.backend == "numba":
                from maple_syrup.experimental_numba import NumbaSurfaceRouter

                router_cls = NumbaSurfaceRouter
            else:
                router_cls = CpuSurfaceRouter
            self.surface = router_cls("local_inertial", graph, geometry=topology.geometry, control=control, available_bytes_fn=vertical.available_bytes_fn)
            self.geometry = self.surface.geometry

    def _equal(self, a: Any, b: Any) -> bool:
        if self.vertical.backend == "cuda":
            from maple_syrup import floating_pond_cuda as fpc_

            return bool(fpc_.scalar(self.xp.array_equal(a, b).astype(np.int64)))  # ONE counted 8-byte read: the derived-sum check
        return bool(np.array_equal(a, b))

    def initial_state(self, depth_m: Any, soil_water_m: Any, *, t_s: float = 0.0) -> HydraulicState:
        v, xp = self.vertical, self.xp
        for name, a in (("depth_m", depth_m), ("soil_water_m", soil_water_m)):
            if type(a) is not xp.ndarray or a.dtype != np.float64 or tuple(a.shape) != self.shape:
                raise FloatingPondError(f"{name} must be a {xp.__name__} float64 array with shape {self.shape}")
        if not self._equal(soil_water_m, v.initial_soil_water()):
            raise FloatingPondError("soil_water_m must be the derived sum of the owner's accepted floating inventory (it is never a second authority)")
        v._no_initial("initial_state")
        ny, nx = self.shape
        if float(t_s) != float(v._initial.state.t_s):
            raise FloatingPondError(f"the event initial clock {t_s!r} differs from the owner's physical clock {v._initial.state.t_s!r}; refused (nothing is snapped or advanced)")
        soil = v._derived(v._initial.state)
        v._register(soil, v._initial)
        state = HydraulicState(check_state_time(t_s, "t_s"), xp.array(depth_m, copy=True), soil, xp.zeros((ny, nx + 1)), xp.zeros((ny + 1, nx)))
        self.validate_state(state)
        return state

    def validate_state(self, state: HydraulicState) -> None:
        if self.vertical.backend == "cuda":
            from maple_syrup.experimental_hydrology import surface_state_of

            self.surface.validate_state(surface_state_of(state))  # current device, fingerprints, shapes, dtypes and the face arrays of the genuine router
        else:
            validate_hydraulic_state(np, state, method="local_inertial", shape=self.shape, faces=(self.geometry.fx_type, self.geometry.fx_sign, self.geometry.fy_type, self.geometry.fy_sign))
        payload = self.vertical._payload(state.soil_water_m)
        if not self._equal(state.soil_water_m, self.vertical._derived(payload.state)):
            raise FloatingPondError("soil_water_m is not the derived layer sum of the floating inventory it was issued for")

    def describe(self) -> dict[str, Any]:
        return {"component": COMPONENT_TAG, "method": "local_inertial", "implementation": self.surface.implementation, "vertical": "floating layers (matrix + optional preferential + gravity)",
                "identity": self.vertical.identity, "surface": self.surface.describe()}

    def step(self, rain_rate_m_per_s: Any, state: HydraulicState, dt_s: float):
        return self._step(rain_rate_m_per_s, state, dt_s, None)

    def step_day(self, state: HydraulicState, dt_s: float, evaporation: Any, pet_terrain_m_s: Any, cover: Any):
        """ONE wet-INTERevent fragment (rain 0): the SAME accepted stage order as `step` (preferential -> matrix(H0, zero rain) -> evaporation staging on the REMAINING pool (surface E + post-cap soil share) -> ONE FSL gravity/diffusion
        step with that share -> the saturation return rejoins the post-E pool -> the accepted surface router). `evaporation` is a `floating_evaporation` operator of this vertical's column. Returns the composed hydraulic step; the
        fragment's E/drainage terms are left in `self.last_day` (host scalars/arrays) ONLY for an accepted step. Pure: a router rejection (`HydraulicStepRejected`) registers nothing."""
        ny, nx = self.shape
        zero = self.vertical._zero_pet  # the vertical's OWN read-only exact-(N,) float64 zero row of this backend (no per-fragment allocation); the rain row is only READ by `_step` (pool = rain dt + h0, the matrix onset descriptor)
        return self._step(zero.reshape(ny, nx), state, dt_s, (evaporation, pet_terrain_m_s, cover))

    def _step(self, rain_rate_m_per_s: Any, state: HydraulicState, dt_s: float, day: Any):
        v, xp = self.vertical, self.xp
        self.validate_state(state)
        payload = v._payload(state.soil_water_m)
        dt = float(dt_s)
        if not dt > 0.0:
            raise FloatingPondError(f"dt_s must be finite and > 0, got {dt_s!r}")
        t0 = check_state_time(state.t_s, "state.t_s")
        v._admit_step()
        ny, nx = self.shape
        N = v.n_cells
        kw = v.stage_kwargs()
        rain = rain_rate_m_per_s.reshape(N)  # arrays of the stage's namespace (no host coercion of a device array)
        h0 = state.depth_m.reshape(N)
        pool = rain * dt + h0
        soil_state = payload.state
        if payload.pref_op is not None:
            p = fpi.floating_preferential_step(payload.pref_op, pool, soil_state, dt, **kw)
            pool, soil_state, pref_in = p.surface_water_m, p.state, p.intake_total_m
        else:
            pref_in = xp.zeros(N)
        m = fmi.floating_matrix_step(payload.matrix_op, pool, h0, rain, soil_state, payload.history, dt, **kw)
        if day is None:
            c = fsl.floating_step(payload.column, m.state, dt, added_water_m=v._zero_added, potential_evaporation_m_s=v._zero_pet, **kw)  # E/T are exactly zero during rain/recession
            post = m.surface_pool_m + c.surface_return_m
        else:  # the approved wet-interevent day: E on the remaining pool and the post-cap soil share, then ONE FSL step; the return rejoins the POST-E pool
            from maple_syrup import floating_evaporation as fe_

            e = fe_.stage_floating_evaporation(day[0], pet_terrain_m_s=day[1], cover=day[2], surface_water_m=m.surface_pool_m, dt_s=dt, **kw)
            c = fsl.floating_step(payload.column, m.state, dt, added_water_m=v._zero_added, **kw, **e.native_step_kwargs())
            post = e.surface_water_after_m + c.surface_return_m
        if v.backend == "cuda":
            from maple_syrup.experimental_hydrology import SurfaceState

            surface = self.surface.step(SurfaceState(t0, post.reshape(ny, nx), state.qx_m2_s, state.qy_m2_s), dt)  # genuine CudaSurfaceRouter, the ACCEPTED start time
            if surface.state.t_s != t0 + dt:
                raise FloatingPondError(f"clock mismatch: surface end {surface.state.t_s!r}, required {t0 + dt!r}")
        else:
            surface = self.surface._lateral(post.reshape(ny, nx), state.qx_m2_s, state.qy_m2_s, dt, t0 + dt, check_depth=True)
        soil = c.state.water_m.sum(axis=0).reshape(ny, nx)
        column = ColumnStep(dt, post.reshape(ny, nx), soil, (rain * dt).reshape(ny, nx), (pref_in + m.intake_m).reshape(ny, nx), c.surface_return_m.reshape(ny, nx), c.drainage_m.reshape(ny, nx))
        v._register(soil, _Payload(payload.column, payload.matrix_op, m.history, payload.pref_op, c.state, payload.depth_m))
        v.stats.steps += 1
        if day is not None:  # only an ACCEPTED fragment leaves its terms (the router rejection above raised before this point)
            if v.backend == "cuda":  # SCALAR terms only: the E ledger and FSL budget host packets and the router's export scalar; no per-fragment (N,) device-to-host array
                self.last_day = {"surface_evaporation_m3": float(e.ledger.evaporation_m3), "soil_budget": c.budget, "export_m3": float(surface.export_m3)}
            else:
                self.last_day = {"surface_evaporation_m": np.asarray(e.surface_evaporation_m), "soil_budget": c.budget, "drainage_m": np.asarray(c.drainage_m), "export_m3": float(surface.export_m3),
                                 "surface_return_m": np.asarray(c.surface_return_m)}
        return compose_hydraulic_step(surface, column)

    def rebuilt(self, topology: PondTopology) -> _FloatingSolver:
        return _FloatingSolver(self.vertical, topology, self.control)


def rebuild_floating_pond(pond: PondHydraulics, new_topology: PondTopology) -> PondHydraulics:
    """Pond rebind for the floating vertical after a net-elevation refresh of the SAME identity: a fresh surface router on the refreshed faces, the SAME stage and payloads."""
    if pond.params is not None or not isinstance(pond.solver, _FloatingSolver):
        raise PondHydraulicsError("not a floating pond")
    solver = pond.solver.rebuilt(new_topology)
    return PondHydraulics(topology=new_topology, solver=solver, backend=pond.backend, params=None, control=pond.control, graph=pond.graph, graph_kind=pond.graph_kind)
