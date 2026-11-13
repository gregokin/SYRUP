"""Phase 2 D1: FLOATING porous soil column — typed owner/state, geometry, ONE physical step (NumPy reference on the unchanged shared body), pulse step, and staged zero-time capacity reconciliation.

NOT a legacy `SoilColumn`/`SoilLayers` and NOT a `SpatialSoilColumn`: a new, genuinely typed owner (`FloatingSoilColumn`) with its own state (`FloatingSoilState`). Every legacy/spatial/native entry point
refuses it by type; no storm, lifecycle, restart, LEAF or MASY adapter exists here, and nothing in this module reads a bed (the actual soil depth `D` is a caller input).

Geometry (user decisions 2026-11-13: surface-following fixed nominal layers over a flat, impermeable full-grid bottom)
----------------------------------------------------------------------------------------------------------------------
Shared nominal edges `e_0 = 0 < e_1 < ... < e_L = H` (O(L)) and an actual per-cell soil depth `D_c >= 0` (O(N), a caller input derived from the actual mass-derived column height, never from a world datum).
The porous thickness of layer `l` in cell `c` is `h_lc = max(0, min(D_c, e_{l+1}) - e_l)`; for `D_c >= e_{l+1}` this is exactly the historical `e_{l+1} - e_l` expression, so a full-soil layer has the identical
bits as the nominal `dz` of `soil_layers`/`SpatialSoilColumn`. No `(L, N)` thickness/capacity/conductance map is retained: rows are derived lazily from the RAW maps (`O(N)` per row). Capacities are `theta * h`
(`res_m`, `fc_m`, `sat_m`, tolerance `coef * sat_m` with the same `coef` as `soil_layers`), so a partial layer holds `theta h` and a zero (rock) slot has capacity 0 and state water exactly 0. The lower boundary is per cell:
`bottom_open_c = (configured "open") AND (D_c > H)` (strict: `D_c == H` is physically closed rock; an open bottom is export BELOW the modelled profile and only above rock). No rock map, parameter or pore water exists.

Physical step (documented arithmetic; ONE shared body)
-------------------------------------------------------
`floating_step` runs the UNCHANGED `subsurface._reference_step_body` (stage order: addition with saturation return, gravity cascade (FC triggers movement, saturation limits storage), evaporation from layer 0,
sequential pair relaxation, output/closure checks) through a PRIVATE typed reader whose indexed rows are derived from `h`: `dz[k] = h if h > 0 else 1.0` (a finite stand-in, never used where `h = 0` because the water there is
exactly 0 and the exchange is limited to 0), `res_m/fc_m/sat_m = theta * h`, `cap_tol = coef (theta_sat h)`, `pair_inv = 1/dz_k + 1/dz_{k+1}`, pair separation `sep = sep_nominal[k]` (the historical centre difference, so full-soil pairs are
bit-identical) when both layers are fully inside the soil (`D >= e_{k+2}`) else `(h_k + h_{k+1}) / 2` (a stand-in 1.0 where both are 0), conductances `G_down = ((d K) h_k) / sep`, `G_up = ((d K) h_{k+1}) / sep` (donor-depth law with the actual donor depth) or
`D_k / sep` (constant mode). An interface touching a zero slot cannot exchange: the existing limits `min(amount, donor available above residual, receiver room)` are exactly 0 there. Layer-0 evaporation on an all-rock cell
has `theta0 = 0 <= residual` so the multiplier and the demand are 0 (actual E = 0, unmet 0), like a dry surface; no rock-specific loss law exists. The body's Python lower-boundary flag is per CALL, so the NumPy reference makes one pass per boundary kind present
(closed, open) and merges per cell (documented cost: a mixed domain holds two result sets transiently); the Numba/CUDA kernels take the flag per cell. Inactive cells keep every water entry bit for bit and are excluded from conservation.

Inactive placeholder policy (published; constructors and step agree): an inactive cell's water entries must be finite and >= 0; NO capacity or rock law applies to them; they are preserved exactly by the step and by reconciliation, never counted in the
active water budget, and never converted or zeroed on adoption.

States: `floating_initial_state` (new production initialisation) requires, on every ACTIVE cell, `0 <= W < sat_m` strictly on porous layers and `W == 0` on rock slots (a wholly-rock column is legal); `floating_continued_state` (transient/restart equality)
allows `0 <= W <= sat_m + tol` (zero slots exactly 0). No epsilon, no clipping.

Staged reconciliation (`reconcile_floating`, a PURE zero-time candidate; no added water, E, T, diffusion, rain, time or fake dt)
-----------------------------------------------------------------------------------------------------------------------------
Accepted old water `W` (valid for the old column) against a newly proposed column (same L, N, active mask and nominal edges; new properties/depth). `W` may exceed the NEW capacity only inside this stage. Per active cell: (1) the existing gravity cascade on the NEW
field capacity (closed/rock bottom: `forward = min(excess, deeper room)` with the room measured once on the input water, bottom layer forwards nothing; open bottom above rock: the whole excess moves and the last layer's excess is EXPORTED once);
(2) then the saturation excess of each layer returns to the SURFACE (an internal transfer). No upward soil fill, no clipping, and nothing leaves silently. The receipt reports percolation `(L-1, N)`, surface return, bottom export, the old/new column identities and the water digest, with the MAPLE
native closure bound; reconciling to an identical column moves nothing. Component only: duplicate-generation authority, changed-column selection and native adoption belong to the later integration (E).

Backends: "numpy" (reference, this module), "numba" (argument binding onto the shared compiled body, `floating_soil_layers_numba`) and "cuda" (D1b: resident device owner/state, `floating_soil_layers_cuda` + the shared rendered source; AUTHOR-WRITTEN, host-verified only, NOT run on a device).
CUDA conventions: per-cell inputs and states are CuPy arrays of ONE device (no implicit transfer), the O(L) edges are a host array uploaded once; `memory_budget_bytes`/`reserve_bytes`/`available_bytes_fn` bound the DEVICE side and `host_memory_budget_bytes`/`host_reserve_bytes`/`host_available_bytes_fn` the host side; the current device is gated BEFORE any reader, copy or accounting; the hot step reads one 72-byte packet and the hot reconcile one 56-byte packet (capability identity, no content hash).
"""

from __future__ import annotations

import contextlib
import hashlib
import secrets
import json
import math
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = [
    "BACKENDS",
    "COMPONENT_TAG",
    "FloatingBackendUnavailableError",
    "FloatingMemoryError",
    "FloatingSoilColumn",
    "FloatingSoilError",
    "FloatingSoilState",
    "FloatingStep",
    "INACTIVE_POLICY",
    "ReconcileCandidate",
    "ReconcileReceipt",
    "SCHEMA",
    "estimate_floating_memory",
    "floating_continued_state",
    "floating_initial_state",
    "floating_moisture",
    "floating_step",
    "floating_step_with_recharge",
    "floating_storage_m3",
    "porous_thickness",
    "prepare_floating_column",
    "reconcile_floating",
]

SCHEMA = "maple_syrup.floating_soil_layers/1"
COMPONENT_TAG = "floating_porous_column_pure"
BACKENDS = ("numpy", "numba", "cuda")
LAYOUT_SCALAR, LAYOUT_SHARED, LAYOUT_CELL = sp.LAYOUT_SCALAR, sp.LAYOUT_SHARED, sp.LAYOUT_CELL
INACTIVE_POLICY = "inactive_water_entries_finite_nonnegative_no_capacity_law_preserved_exactly_excluded_from_conservation"
FLOATING_ROW_DOUBLES = 16  # declared live (N,) float64 rows of the lazily derived readers/merges beyond the historical NumPy scratch (checked by tracemalloc)
DEFAULT_HASH_CHUNK_BYTES = 1 << 20
_MODULE_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()


class FloatingSoilError(ss.SubsurfaceError):
    """Invalid explicit input (metadata or value), invalid state or unrepresentable derived geometry. Nothing was published."""


class FloatingMemoryError(ss.SubsurfaceMemoryError):
    """The modelled requirement plus reserve exceeds the quota or the reported availability. Nothing was allocated."""


class FloatingBackendUnavailableError(ss.SubsurfaceBackendUnavailableError):
    """The requested backend is unavailable or not implemented for this component. There is no fallback."""


def _count(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise FloatingSoilError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)):
        raise FloatingSoilError(f"{name} must be a finite real host scalar, got {value!r}")
    return float(value)


# ---- inputs ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _array(name: str, value: Any, shapes: dict[str, tuple], *, scalar_ok: bool, ns_type: Any = np.ndarray) -> tuple[str, Any]:
    if sp._is_real_scalar(value):
        if not scalar_ok:
            raise FloatingSoilError(f"{name} must be an array of shape {list(shapes.values())}, a scalar is not accepted")
        return LAYOUT_SCALAR, np.full(1, float(value), dtype=np.float64)
    if isinstance(value, (bool, np.bool_)) or type(value) is not ns_type:
        raise FloatingSoilError(f"{name} must be exactly a {'cupy' if ns_type is not np.ndarray else 'numpy'}.ndarray of the backend's namespace (no list, masked array or implicit transfer), got {type(value).__module__}.{type(value).__name__}")
    if value.dtype != np.float64 or not value.flags.c_contiguous:
        raise FloatingSoilError(f"{name} must be a C-contiguous float64 array")
    shape = tuple(int(n) for n in value.shape)
    for kind, want in shapes.items():
        if shape == want:
            return kind, value
    raise FloatingSoilError(f"{name} must have shape {list(shapes.values())}{' or be a real scalar' if scalar_ok else ''}, got {shape}")


def _row(arr: np.ndarray, kind: str, k: int) -> Any:
    return arr[0] if kind == LAYOUT_SCALAR else arr[k]


# ---- memory model -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FloatingMemoryEstimate:
    backend: str
    n_layers: int
    n_cells: int
    retained_bytes: int
    step_bytes: int  # ONE step: the historical estimate of the shared body (`subsurface.estimate_step_memory`) plus the floating rows/merge/sanitised-water allowance
    mixed_boundary_bytes: int  # the second result set of a mixed closed/open NumPy pass (included in step_bytes)
    required_bytes: int
    terms: Any
    scope: str = "modelled host bytes of this component's own arrays; inputs, interpreter/JIT, allocator pool and whole-process memory are excluded; a reading is not a reservation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", sp._freeze(dict(self.terms)))


def estimate_floating_memory(n_layers: int, n_cells: int, *, layouts: dict, diffusion_mode: str = "constant", backend: str = "numpy", with_recharge: bool = False, reserve_bytes: int = 0) -> FloatingMemoryEstimate:
    """Pure arithmetic. Retained: the three theta maps and the interface map by layout, `(L+1,)` edges + `(L,)` dz + `(L-1,)` nominal separation, `D` (N doubles), mask (N bytes) and bottom-open flags (N bytes). One step: `ss.estimate_step_memory`
    (host, the shared body's inputs/outputs/scratch) + `FLOATING_ROW_DOUBLES` rows + one sanitised `(L, N)` water copy + a second result set (outputs) for a mixed closed/open domain."""
    L, N = _count(n_layers, "n_layers", minimum=1), _count(n_cells, "n_cells", minimum=1)
    if backend not in BACKENDS:
        raise FloatingBackendUnavailableError(f"backend must be one of {BACKENDS}, got {backend!r}")
    mode = ss._check_diffusion_mode(diffusion_mode)
    f8 = 8
    if backend == "cuda":  # device payload model (floating_soil_layers_cuda.operation_terms); the host side (packets, cold hash buffer) is listed separately in `terms`
        from maple_syrup import floating_soil_layers_cuda as fcu

        ret = fcu.retained_terms(L, N, layouts)
        op = fcu.operation_terms("step_with_recharge" if with_recharge else "step", L, N)
        rec = fcu.operation_terms("reconcile", L, N)
        step_device = max(op["device"], rec["device"])
        reserve = _count(reserve_bytes, "reserve_bytes")
        terms = {"retained_device": ret["total"], "step_outputs": op["outputs"], "step_scratch": op["scratch"], "reconcile_outputs": rec["outputs"], "reconcile_scratch": rec["scratch"],
                 "step_host": op["host"], "reconcile_host": rec["host"], "cold_hash_host_buffer": fcu.HASH_CHUNK_BYTES, "shared_per_block": op["shared"],
                 "pool_rounding_and_context": "excluded (declared)"}
        return FloatingMemoryEstimate(backend, L, N, ret["total"], step_device, rec["outputs"], ret["total"] + step_device + reserve, terms)
    def kind_bytes(kind: str, rows: int) -> int:
        return 0 if rows <= 0 else {LAYOUT_SCALAR: 8, LAYOUT_SHARED: 8 * rows, LAYOUT_CELL: 8 * rows * N}[kind]

    retained = sum(kind_bytes(layouts[k], L) for k in ("theta_residual", "theta_field_capacity", "theta_saturation")) + kind_bytes(layouts["interface"], L - 1)
    retained += (L + 1 + L + max(L - 1, 1)) * f8 + N * f8 + 2 * N
    step = ss.estimate_step_memory(L, N, backend=backend, diffusion_mode=mode, with_recharge=with_recharge)
    outputs = (L * N + 2 * (L - 1) * N + 4 * N + ss.ROW_COUNT * N) * f8
    extra = FLOATING_ROW_DOUBLES * N * f8 + L * N * f8 + outputs
    total = step.required_bytes + extra
    terms = {"retained": retained, "shared_step": step.required_bytes, "floating_rows": FLOATING_ROW_DOUBLES * N * f8, "sanitised_water": L * N * f8, "mixed_boundary_second_result": outputs}
    return FloatingMemoryEstimate(backend, L, N, retained, total, outputs, retained + total + _count(reserve_bytes, "reserve_bytes"), terms)


def _admit(required: int, budget: int | None, reader: Callable[[], int] | None, what: str) -> None:
    if budget is not None and required > budget:
        raise FloatingMemoryError(f"{what} needs an estimated {required} B but the declared memory_budget_bytes is {budget}; nothing was allocated")
    avail = (reader or sp.default_host_available_bytes)()
    if isinstance(avail, (bool, np.bool_)) or not isinstance(avail, (int, np.integer)) or int(avail) < 0:
        raise FloatingSoilError(f"the available-bytes reader must return a non-Boolean integer >= 0, got {avail!r}")
    if required > int(avail):
        raise FloatingMemoryError(f"{what} needs an estimated {required} B but only {int(avail)} B are reported available; nothing was allocated")


# ---- the typed owner ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
_CONSTRUCT = threading.local()


@contextlib.contextmanager
def _constructing():
    _CONSTRUCT.allowed = True
    try:
        yield
    finally:
        _CONSTRUCT.allowed = False


@dataclass(frozen=True, eq=False)
class FloatingSoilColumn:
    """Sealed owner of the RAW maps of ONE floating porous column (module docstring). Created only by `prepare_floating_column`; read-only arrays; never a legacy/spatial column."""

    backend: str
    n_layers: int
    n_cells: int
    n_active: int
    cell_area_m2: float
    diffusion_mode: str
    bottom_drainage: str  # the CONFIGURED lower boundary ("open" applies only above rock)
    diffusion_coefficient: float | None
    theta_evaporation_stress: float
    evaporation_factor: float
    gravity_drainage: bool  # always True in this production component
    layouts: Any
    sha256: str
    estimate: FloatingMemoryEstimate
    device_id: int | None
    mask_sha256: str | None  # cold identity of the active mask / the nominal edges: lets a hot CUDA reconcile bind "same mask, same edges" without any transfer
    edges_sha256: str | None
    _maps: Any
    component_tag: str = COMPONENT_TAG
    inactive_policy: str = INACTIVE_POLICY
    identity_kind: str = "content_sha256"  # "content_sha256" (prepared: a digest of every map) | "geometry_view_token" (hot view: `sha256` digests the parent identity and a minted generation token, NEVER the map contents)
    parent_sha256: str | None = None  # the prepared column a geometry view shares every map but the depth with

    def __post_init__(self) -> None:
        if not getattr(_CONSTRUCT, "allowed", False):
            raise FloatingSoilError("a FloatingSoilColumn is created only by prepare_floating_column (no direct construction, replace() or copy)")
        object.__setattr__(self, "layouts", sp._freeze(dict(self.layouts)))
        object.__setattr__(self, "_maps", MappingProxyType(dict(self._maps)))

    def __copy__(self):
        raise FloatingSoilError("a FloatingSoilColumn owns private maps and is not copyable")

    def __deepcopy__(self, memo):
        raise FloatingSoilError("a FloatingSoilColumn owns private maps and is not copyable")

    def __reduce_ex__(self, protocol):
        raise FloatingSoilError("a FloatingSoilColumn is not serialisable")

    @property
    def total_depth_m(self) -> float:
        return float(self._maps["edges"][-1])

    def summary(self) -> dict:
        return {"component_tag": self.component_tag, "schema": SCHEMA, "backend": self.backend, "n_layers": self.n_layers, "n_cells": self.n_cells, "n_active": self.n_active, "bottom_drainage": self.bottom_drainage,
                "diffusion_mode": self.diffusion_mode, "layouts": dict(self.layouts), "nominal_depth_m": self.total_depth_m, "sha256": self.sha256, "inactive_policy": self.inactive_policy,
                "retained_bytes": self.estimate.retained_bytes}


@dataclass(frozen=True, eq=False)
class FloatingSoilState:
    """`water_m` `(L, N)` float64 (authoritative water inventory of the ACTUAL porous soil; exactly 0 in rock slots) and the finite time. Minted only by the state constructors and the step/reconcile functions."""

    water_m: Any
    t_s: float
    column_sha256: str
    token: str = ""  # a minted generation/capability token (never a content hash): binds this exact accepted instance for hot reconcile receipts


def _new_state(column: Any, water: Any, t_s: float) -> "FloatingSoilState":
    return FloatingSoilState(water, float(t_s), column.sha256, secrets.token_hex(16))


@dataclass(frozen=True, eq=False)
class FloatingStep:
    state: FloatingSoilState
    dt_s: float
    surface_return_m: Any
    drainage_m: Any  # export below the modelled profile (open bottom above rock only)
    evaporation_m: Any
    evaporation_unmet_m: Any
    percolation_m: Any
    diffusion_m: Any
    budget: ss.StepBudget


@dataclass(frozen=True)
class ReconcileReceipt:
    old_column_sha256: str
    new_column_sha256: str
    water_sha256: str | None  # CPU: the full content digest of the old water; CUDA: None (a hot receipt never downloads water)
    t_s: float
    storage_before_m3: float
    storage_after_m3: float
    surface_return_m3: float
    bottom_export_m3: float
    residual_m3: float
    bound_m3: float
    n_terms: int
    moved_percolation_m3: float
    identity_kind: str = "content_sha256"  # "content_sha256" (CPU, cold) | "capability_token" (CUDA, hot): the tokens below are minted generation identities, NEVER physical content hashes
    old_state_token: str = ""
    new_state_token: str = ""


@dataclass(frozen=True, eq=False)
class ReconcileCandidate:
    """PURE zero-time candidate: nothing is adopted; repeating the call gives an equal candidate and mutates nothing."""

    state: FloatingSoilState
    percolation_m: Any  # (L-1, N)
    surface_return_m: Any  # (N,)
    bottom_export_m: Any  # (N,)
    receipt: ReconcileReceipt


# ---- preparation ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def prepare_floating_column(nominal_depth_edges_m: Any, *, theta_residual: Any, theta_field_capacity: Any, theta_saturation: Any, diffusivity_m2_s: Any = None, interface_conductivity_m_s: Any = None,
                            diffusion_coefficient: Any = None, theta_evaporation_stress: Any, evaporation_factor: Any = 1.0, diffusion_mode: str = "constant", bottom_drainage: str = "open",
                            soil_depth_m: Any, active_mask: Any, cell_area_m2: Any, backend: str = "numpy", memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                            available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                            host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingSoilColumn:
    """Validate every explicit input (metadata first, then values, ALL before anything is published) and own read-only copies. `soil_depth_m` `(N,)` is the actual porous soil depth per cell (finite, >= 0; 0 = wholly rock).
    Gravity drainage is always on (not a parameter). Interface inputs stay explicit (no `Ks/2`, `K_geom` or averaging); modes as in `soil_layers`."""
    if backend not in BACKENDS:
        raise FloatingBackendUnavailableError(f"backend must be one of {BACKENDS} (there is no fallback between backends), got {backend!r}")
    mode, bottom = ss._check_diffusion_mode(diffusion_mode), ss._check_bottom_drainage(bottom_drainage)
    if backend != "cuda" and (host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None):
        raise FloatingSoilError("host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to the cuda backend")
    if backend == "cuda":
        return _prepare_cuda(nominal_depth_edges_m, theta_residual, theta_field_capacity, theta_saturation, diffusivity_m2_s, interface_conductivity_m_s, diffusion_coefficient, theta_evaporation_stress,
                             evaporation_factor, mode, bottom, soil_depth_m, active_mask, cell_area_m2,
                             dict(memory_budget_bytes=memory_budget_bytes, reserve_bytes=reserve_bytes, available_bytes_fn=available_bytes_fn, host_memory_budget_bytes=host_memory_budget_bytes,
                                  host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn))
    if backend == "numba":
        from maple_syrup import floating_soil_layers_numba as fnb

        fnb.require_numba()
    if type(nominal_depth_edges_m) is not np.ndarray or nominal_depth_edges_m.ndim != 1 or nominal_depth_edges_m.shape[0] < 2 or nominal_depth_edges_m.dtype != np.float64 or not nominal_depth_edges_m.flags.c_contiguous:
        raise FloatingSoilError("nominal_depth_edges_m must be a C-contiguous float64 numpy array of shape (L+1,) with L >= 1 (shared by every cell)")
    L = int(nominal_depth_edges_m.shape[0]) - 1
    if type(soil_depth_m) is not np.ndarray or soil_depth_m.ndim != 1 or soil_depth_m.dtype != np.float64 or not soil_depth_m.flags.c_contiguous or soil_depth_m.shape[0] < 1:
        raise FloatingSoilError("soil_depth_m must be a C-contiguous float64 numpy array of shape (N,) with N >= 1")
    N = int(soil_depth_m.shape[0])
    if type(active_mask) is not np.ndarray or active_mask.dtype != np.bool_ or active_mask.shape != (N,) or not active_mask.flags.c_contiguous:
        raise FloatingSoilError(f"active_mask must be a C-contiguous bool numpy array of shape ({N},)")
    area = _real(cell_area_m2, "cell_area_m2")
    if not area > 0.0:
        raise FloatingSoilError("cell_area_m2 must be > 0")
    stress, factor = _real(theta_evaporation_stress, "theta_evaporation_stress"), _real(evaporation_factor, "evaporation_factor")
    if factor < 0.0:
        raise FloatingSoilError("evaporation_factor must be >= 0")
    maps, kinds = {}, {}
    for key, value in (("theta_residual", theta_residual), ("theta_field_capacity", theta_field_capacity), ("theta_saturation", theta_saturation)):
        kinds[key], maps[key] = _array(key, value, {LAYOUT_SHARED: (L,), LAYOUT_CELL: (L, N)}, scalar_ok=True)
    if mode == "constant":
        if diffusivity_m2_s is None or interface_conductivity_m_s is not None or diffusion_coefficient is not None:
            raise FloatingSoilError("diffusion_mode='constant' requires diffusivity_m2_s and refuses the donor inputs")
        iname, ivalue, dcoef = "diffusivity_m2_s", diffusivity_m2_s, None
    else:
        if diffusivity_m2_s is not None or interface_conductivity_m_s is None or diffusion_coefficient is None:
            raise FloatingSoilError("diffusion_mode='tietjen_donor_depth' requires interface_conductivity_m_s AND diffusion_coefficient and refuses diffusivity_m2_s")
        iname, ivalue, dcoef = "interface_conductivity_m_s", interface_conductivity_m_s, _real(diffusion_coefficient, "diffusion_coefficient")
        if dcoef < 0.0:
            raise FloatingSoilError("diffusion_coefficient must be >= 0")
    if L > 1:
        kinds["interface"], maps["interface"] = _array(iname, ivalue, {LAYOUT_SHARED: (L - 1,), LAYOUT_CELL: (L - 1, N)}, scalar_ok=True)
    else:
        v = 0.0 if not sp._is_real_scalar(ivalue) else float(ivalue)
        if not (math.isfinite(v) and v >= 0.0):
            raise FloatingSoilError(f"{iname} must be finite and >= 0")
        kinds["interface"], maps["interface"] = LAYOUT_SCALAR, np.full(1, v, dtype=np.float64)
    layouts = dict(kinds)
    est = estimate_floating_memory(L, N, layouts=layouts, diffusion_mode=mode, backend=backend, reserve_bytes=_count(reserve_bytes, "reserve_bytes"))
    budget = None if memory_budget_bytes is None else _count(memory_budget_bytes, "memory_budget_bytes")
    _admit(est.retained_bytes + _count(reserve_bytes, "reserve_bytes"), budget, available_bytes_fn, "floating column preparation")
    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    edges = np.array(nominal_depth_edges_m, dtype=np.float64, copy=True)
    dz = edges[1:] - edges[:-1]
    if not (np.all(np.isfinite(edges)) and edges[0] == 0.0 and np.all(np.isfinite(dz)) and np.all(dz > 0.0)):
        raise FloatingSoilError("nominal depth edges must be finite, start at exactly 0 and be strictly increasing with finite positive widths")
    centre = edges[:-1] + dz / 2.0
    sep = centre[1:] - centre[:-1] if L > 1 else np.zeros(1)
    if L > 1 and not (np.all(np.isfinite(sep)) and np.all(sep > 0.0)):
        raise FloatingSoilError("nominal centre separations must be finite and > 0")
    depth = np.array(soil_depth_m, dtype=np.float64, copy=True)
    if not (np.all(np.isfinite(depth)) and np.all(depth >= 0.0)):
        raise FloatingSoilError("soil_depth_m must be finite and >= 0 on every cell (0 is a wholly-rock column)")
    H = float(edges[-1])
    own = {n: np.array(a, dtype=np.float64, copy=True) for n, a in maps.items()}
    maps2 = {"edges": edges, "dz": dz, "sep_nominal": sep, "soil_depth": depth, "active": np.array(active_mask, dtype=np.bool_, copy=True), **own}
    maps2["bottom_open"] = (bottom == "open") & (depth > H)  # strict: D == H is closed rock
    _validate_values(maps2, kinds, L, N, mode, dcoef, stress, coef)
    for a in maps2.values():
        a.flags.writeable = False
    digest = _identity(maps2, kinds, L, N, mode, bottom, dcoef, stress, factor, area)
    with _constructing():
        return FloatingSoilColumn(backend, L, N, int(maps2["active"].sum()), area, mode, bottom, dcoef, stress, factor, True, layouts, digest, est, None, hashlib.sha256(maps2["active"].tobytes()).hexdigest(), hashlib.sha256(maps2["edges"].tobytes()).hexdigest(), maps2)


def _prepare_cuda(edges_in, tr, tf, ts, diffusivity, conductivity, dcoef_in, stress_in, factor_in, mode, bottom, depth_in, mask_in, area_in, quota: dict) -> FloatingSoilColumn:
    """CUDA owner factory: device gate FIRST, metadata (no value read), host validation of the O(L) edges, admission (device and host separately), device-to-device copies, ONE counted upload of the O(L) edges/separations, the resident
    validation kernel (ONE 32-byte packet) and a COLD streamed identity hash; nothing is published on any failure."""
    from maple_syrup import floating_soil_layers_cuda as fcu

    try:
        cp = fcu._rc().require_cuda()
    except Exception as exc:  # noqa: BLE001
        raise FloatingBackendUnavailableError(f"backend 'cuda' needs CuPy and a device ({type(exc).__name__}: {exc}); there is no fallback") from exc
    dev = fcu.gate_device(cp, None)
    ns = cp.ndarray
    if type(edges_in) is not np.ndarray or edges_in.ndim != 1 or edges_in.shape[0] < 2 or edges_in.dtype != np.float64 or not edges_in.flags.c_contiguous:
        raise FloatingSoilError("nominal_depth_edges_m must be a C-contiguous float64 HOST numpy array of shape (L+1,) with L >= 1 (shared metadata, uploaded once)")
    L = int(edges_in.shape[0]) - 1
    if type(depth_in) is not ns or depth_in.ndim != 1 or depth_in.dtype != np.float64 or not depth_in.flags.c_contiguous or depth_in.shape[0] < 1:
        raise FloatingSoilError("soil_depth_m must be a C-contiguous float64 cupy array of shape (N,) with N >= 1")
    N = int(depth_in.shape[0])
    if type(mask_in) is not ns or mask_in.dtype != np.bool_ or mask_in.shape != (N,) or not mask_in.flags.c_contiguous:
        raise FloatingSoilError(f"active_mask must be a C-contiguous bool cupy array of shape ({N},)")
    area = _real(area_in, "cell_area_m2")
    if not area > 0.0:
        raise FloatingSoilError("cell_area_m2 must be > 0")
    stress, factor = _real(stress_in, "theta_evaporation_stress"), _real(factor_in, "evaporation_factor")
    if factor < 0.0:
        raise FloatingSoilError("evaporation_factor must be >= 0")
    kinds, raw = {}, {}
    for key, value in (("theta_residual", tr), ("theta_field_capacity", tf), ("theta_saturation", ts)):
        kinds[key], raw[key] = _array(key, value, {LAYOUT_SHARED: (L,), LAYOUT_CELL: (L, N)}, scalar_ok=True, ns_type=ns)
    if mode == "constant":
        if diffusivity is None or conductivity is not None or dcoef_in is not None:
            raise FloatingSoilError("diffusion_mode='constant' requires diffusivity_m2_s and refuses the donor inputs")
        iname, ivalue, dcoef = "diffusivity_m2_s", diffusivity, None
    else:
        if diffusivity is not None or conductivity is None or dcoef_in is None:
            raise FloatingSoilError("diffusion_mode='tietjen_donor_depth' requires interface_conductivity_m_s AND diffusion_coefficient and refuses diffusivity_m2_s")
        iname, ivalue, dcoef = "interface_conductivity_m_s", conductivity, _real(dcoef_in, "diffusion_coefficient")
        if dcoef < 0.0:
            raise FloatingSoilError("diffusion_coefficient must be >= 0")
    if L > 1:
        kinds["interface"], raw["interface"] = _array(iname, ivalue, {LAYOUT_SHARED: (L - 1,), LAYOUT_CELL: (L - 1, N)}, scalar_ok=True, ns_type=ns)
    else:
        v = 0.0 if not sp._is_real_scalar(ivalue) else float(ivalue)
        if not (math.isfinite(v) and v >= 0.0):
            raise FloatingSoilError(f"{iname} must be finite and >= 0")
        kinds["interface"], raw["interface"] = LAYOUT_SCALAR, np.full(1, v, dtype=np.float64)
    for name, a in [("soil_depth_m", depth_in), ("active_mask", mask_in)] + [(k, a) for k, a in raw.items() if type(a) is ns]:
        if int(a.device.id) != dev:
            raise FloatingSoilError(f"{name} lives on device {int(a.device.id)} but the current device is {dev}; no implicit transfer")
    # the O(L) shared edges: host validation (cold)
    edges = np.array(edges_in, dtype=np.float64, copy=True)
    dz = edges[1:] - edges[:-1]
    if not (np.all(np.isfinite(edges)) and edges[0] == 0.0 and np.all(np.isfinite(dz)) and np.all(dz > 0.0)):
        raise FloatingSoilError("nominal depth edges must be finite, start at exactly 0 and be strictly increasing with finite positive widths")
    centre = edges[:-1] + dz / 2.0
    sep = centre[1:] - centre[:-1] if L > 1 else np.zeros(1)
    if L > 1 and not (np.all(np.isfinite(sep)) and np.all(sep > 0.0)):
        raise FloatingSoilError("nominal centre separations must be finite and > 0")
    layouts = dict(kinds)
    est = estimate_floating_memory(L, N, layouts=layouts, diffusion_mode=mode, backend="cuda", reserve_bytes=_count(quota.get("reserve_bytes", 0), "reserve_bytes"))
    prep = fcu.operation_terms("prepare", L, N)
    fcu.admit(cp, dev, est.retained_bytes + prep["scratch"], prep["host"], quota, "floating column preparation")
    from maple.core.backend import to_device

    owned = {}
    for name, a in raw.items():
        owned[name] = cp.array(a, dtype=cp.float64, copy=True) if type(a) is ns else to_device(np.ascontiguousarray(a), cp)
    owned["soil_depth"] = cp.array(depth_in, dtype=cp.float64, copy=True)
    owned["active"] = cp.array(mask_in, dtype=cp.bool_, copy=True)
    owned["edges"] = to_device(edges, cp)
    owned["sep_nominal"] = to_device(np.ascontiguousarray(np.asarray(sep, dtype=np.float64)), cp)
    owned["bottom_open"] = cp.zeros(N, dtype=cp.uint8)
    report = fcu.prepare_device(cp, owned, layouts, L, N, directional=mode != "constant", dcoef=0.0 if dcoef is None else dcoef, stress=stress, config_open=bottom == "open")
    fcu.raise_flags(report, fcu.FLAG_TEXT, "floating column preparation")
    header = f"{SCHEMA}\0cuda;mode={mode};bottom={bottom};d={dcoef!r};stress={stress!r};factor={factor!r};area={area!r};L={L};N={N};kinds={sorted(kinds.items())}\0"
    digest = fcu.stream_hash(cp, [(n, owned[n]) for n in sorted(owned)], header.encode())
    with _constructing():
        return FloatingSoilColumn("cuda", L, N, None, area, mode, bottom, dcoef, stress, factor, True, layouts, digest, est, dev, fcu.stream_hash(cp, [("active", owned["active"])], b"mask"), fcu.stream_hash(cp, [("edges", owned["edges"])], b"edges"), owned)


def _validate_values(m: dict, kinds: dict, L: int, N: int, mode: str, dcoef: float | None, stress: float, coef: float) -> None:
    """Row-wise (O(N) temporaries) value checks of the raw maps, then the derived quantities of every POROUS layer/pair of every cell (rock slots are not constrained: they carry no property)."""
    edges, depth = m["edges"], m["soil_depth"]
    with np.errstate(all="ignore"):
        for k in range(L):
            r, f, s = (_row(m[n], kinds[n], k) for n in ("theta_residual", "theta_field_capacity", "theta_saturation"))
            r, f, s = np.broadcast_to(r, (N,)), np.broadcast_to(f, (N,)), np.broadcast_to(s, (N,))
            if not (np.all(np.isfinite(r)) and np.all(np.isfinite(f)) and np.all(np.isfinite(s))):
                raise FloatingSoilError(f"layer {k}: theta inputs must be finite")
            if not np.all((r >= 0.0) & (f >= r) & (s >= f) & (s <= 1.0) & (s > r)):
                raise FloatingSoilError(f"layer {k}: theta values must satisfy 0 <= residual <= field capacity <= saturation <= 1 with residual < saturation")
            if k == 0 and not np.all(stress > r):
                raise FloatingSoilError("theta_evaporation_stress must exceed the layer-0 residual in every cell")
            if k < L - 1:
                v = np.broadcast_to(_row(m["interface"], kinds["interface"], k), (N,))
                if not np.all(np.isfinite(v) & (v >= 0.0)):
                    raise FloatingSoilError(f"interface {k}: the interface input must be finite and >= 0")
            h = np.maximum(np.minimum(depth, edges[k + 1]) - edges[k], 0.0)
            pos = h > 0.0
            sat_m = s * h
            tol = coef * sat_m
            if not np.all(~pos | (np.isfinite(sat_m) & (sat_m > 0.0) & np.isfinite(tol) & (sat_m + tol > sat_m))):
                raise FloatingSoilError(f"layer {k}: a porous slab is too thin or its capacity is not representable/resolvable (capacity tolerance); nothing is clamped")
            if k < L - 1:
                h2 = np.maximum(np.minimum(depth, edges[k + 2]) - edges[k + 1], 0.0)
                both = pos & (h2 > 0.0)
                inv = 1.0 / np.where(pos, h, 1.0) + 1.0 / np.where(h2 > 0.0, h2, 1.0)
                sep = np.where(depth >= edges[k + 2], m["sep_nominal"][k], (h + h2) / 2.0)
                if mode == "constant":
                    g = v / np.where(sep > 0.0, sep, 1.0)
                    ok = np.isfinite(g) & np.isfinite(g * inv)
                else:
                    dd, du = (dcoef * v) * np.where(pos, h, 1.0), (dcoef * v) * np.where(h2 > 0.0, h2, 1.0)
                    g, gu = dd / np.where(sep > 0.0, sep, 1.0), du / np.where(sep > 0.0, sep, 1.0)
                    ok = np.isfinite(dd) & np.isfinite(du) & np.isfinite(g) & np.isfinite(gu) & np.isfinite(g * inv) & np.isfinite(gu * inv)
                if not np.all(~both | (np.isfinite(inv) & np.isfinite(sep) & (sep > 0.0) & ok)):
                    raise FloatingSoilError(f"interface {k}: a derived pair quantity is not finite/representable (layers too thin or inputs too extreme)")


def floating_geometry_view(parent: FloatingSoilColumn, soil_depth_m: Any, *, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0, memory_budget_bytes: Any = None,
                           host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingSoilColumn:
    """HOT official geometry view of a prepared column (NumPy/Numba host `soil_depth_m` ndarray; CUDA: an exact CuPy float64 `(N,)` array of the owner's device): every map except the actual porous depth `soil_depth` `(N,)` and
    the strict bottom flag derived from it is SHARED with `parent` (no `(L, N)` copy, no map hash); O(N) work. `sha256` identifies this exact view (digest of the parent identity and a minted token;
    `identity_kind == "geometry_view_token"`): it is a generation identity, never a physical content hash, and a view of a view binds the ORIGINAL prepared column (no retained parent chain). Properties are held; the caller
    (an event owner) reconciles water for exactly the columns whose slabs or bottom flag changed. CUDA: the current device is checked before any reader/allocation/copy; the owned O(N) payload and the validation packet are
    admitted (device and host separately; `available_bytes_fn` is the DEVICE reader there; the reserves are counted once) before any copy; the existing resident validation kernel checks the new depth and writes the
    flags (ONE 32-byte packet); nothing is downloaded and nothing is published on a refusal. The host-only quota arguments apply to the cuda backend."""
    _check_column(parent)
    _gate(parent)  # CUDA: the current device before any reader, allocation or copy
    if parent.backend != "cuda" and (memory_budget_bytes is not None or host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None):
        raise FloatingSoilError("memory_budget_bytes / host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to the cuda backend")
    if parent.backend == "cuda":
        from maple_syrup import floating_soil_layers_cuda as fcu

        quota = dict(memory_budget_bytes=memory_budget_bytes, reserve_bytes=reserve_bytes, available_bytes_fn=available_bytes_fn, host_memory_budget_bytes=host_memory_budget_bytes,
                     host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn)
        maps = fcu.geometry_view_maps(fcu._rc().require_cuda(), parent, soil_depth_m, quota)
        root = parent.sha256 if parent.parent_sha256 is None else parent.parent_sha256
        digest = hashlib.sha256(f"{SCHEMA}\0geometry_view\0{root}\0{secrets.token_hex(16)}".encode()).hexdigest()
        with _constructing():
            return FloatingSoilColumn(parent.backend, parent.n_layers, parent.n_cells, parent.n_active, parent.cell_area_m2, parent.diffusion_mode, parent.bottom_drainage, parent.diffusion_coefficient,
                                      parent.theta_evaporation_stress, parent.evaporation_factor, True, parent.layouts, digest, parent.estimate, parent.device_id, parent.mask_sha256, parent.edges_sha256, maps,
                                      parent.component_tag, parent.inactive_policy, "geometry_view_token", root)
    # ALWAYS admit the owned O(N) maps (depth copy 8 N, bottom flag N) and the finite/sign validation scratch (two N-bool masks) BEFORE any copy or value check: the caller's reader, else the REAL default host reader
    # (never an invented unlimited one); the reserve is validated whatever the branch
    need = parent.n_cells * (8 + 1 + 2) + _count(reserve_bytes, "reserve_bytes")
    _admit(need, None, available_bytes_fn, "floating geometry view")
    if type(soil_depth_m) is not np.ndarray or soil_depth_m.dtype != np.float64 or soil_depth_m.shape != (parent.n_cells,) or not soil_depth_m.flags.c_contiguous:
        raise FloatingSoilError(f"soil_depth_m must be a C-contiguous float64 numpy array of shape ({parent.n_cells},)")
    if not (bool(np.isfinite(soil_depth_m).all()) and not bool((soil_depth_m < 0.0).any())):
        raise FloatingSoilError("soil_depth_m must be finite and >= 0 on every cell")
    depth = np.array(soil_depth_m, dtype=np.float64, copy=True)
    depth.flags.writeable = False
    maps = dict(parent._maps)
    maps["soil_depth"] = depth
    bottom_open = (parent.bottom_drainage == "open") & (depth > float(parent._maps["edges"][-1]))
    bottom_open.flags.writeable = False
    maps["bottom_open"] = bottom_open
    root = parent.sha256 if parent.parent_sha256 is None else parent.parent_sha256
    digest = hashlib.sha256(f"{SCHEMA}\0geometry_view\0{root}\0{secrets.token_hex(16)}".encode()).hexdigest()
    with _constructing():
        return FloatingSoilColumn(parent.backend, parent.n_layers, parent.n_cells, parent.n_active, parent.cell_area_m2, parent.diffusion_mode, parent.bottom_drainage, parent.diffusion_coefficient,
                                  parent.theta_evaporation_stress, parent.evaporation_factor, True, parent.layouts, digest, parent.estimate, parent.device_id, parent.mask_sha256, parent.edges_sha256, maps,
                                  parent.component_tag, parent.inactive_policy, "geometry_view_token", root)


def _identity(maps: dict, kinds: dict, L: int, N: int, mode: str, bottom: str, dcoef: Any, stress: float, factor: float, area: float) -> str:
    h = hashlib.sha256()
    h.update(f"{SCHEMA}\0mode={mode};bottom={bottom};d={dcoef!r};stress={stress!r};factor={factor!r};area={area!r};L={L};N={N};kinds={sorted(kinds.items())}\0".encode())
    for name in sorted(maps):
        a = maps[name]
        h.update(f"{name}|{a.dtype}|{a.shape}\0".encode())
        flat = a.reshape(-1)
        step = max(1, DEFAULT_HASH_CHUNK_BYTES // a.itemsize)
        for i in range(0, flat.size, step):
            h.update(memoryview(np.ascontiguousarray(flat[i:i + step])).cast("B"))
    return h.hexdigest()


# ---- geometry readers ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _check_column(column: Any) -> None:
    if type(column) is not FloatingSoilColumn:
        raise FloatingSoilError(f"column must be a FloatingSoilColumn (legacy SoilColumn and SpatialSoilColumn are refused), got {type(column).__name__}")


def porous_thickness(column: FloatingSoilColumn, layer: int) -> np.ndarray:
    """A fresh `(N,)` row `h = max(0, min(D, e_{l+1}) - e_l)`; no `(L, N)` map exists."""
    _check_column(column)
    e, d = column._maps["edges"], column._maps["soil_depth"]
    return np.maximum(np.minimum(d, e[layer + 1]) - e[layer], 0.0)


class _Rows:
    __slots__ = ("fn",)

    def __init__(self, fn: Callable[[int], Any]) -> None:
        self.fn = fn

    def __getitem__(self, k: int) -> Any:
        return self.fn(int(k))


class _FloatingParams:
    """PRIVATE typed reader for the UNCHANGED shared body: lazy `(N,)` rows derived from `h` and the raw maps (module docstring); `closed` is the body's Python lower-boundary flag for this pass."""

    def __init__(self, column: FloatingSoilColumn, closed: bool) -> None:
        m, kinds = column._maps, column.layouts
        coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
        L, e, depth = column.n_layers, m["edges"], m["soil_depth"]
        tr, tf, ts, iface = m["theta_residual"], m["theta_field_capacity"], m["theta_saturation"], m["interface"]
        ktr, ktf, kts, kif = kinds["theta_residual"], kinds["theta_field_capacity"], kinds["theta_saturation"], kinds["interface"]
        dcoef = column.diffusion_coefficient

        def h(k: int) -> np.ndarray:
            return np.maximum(np.minimum(depth, e[k + 1]) - e[k], 0.0)

        def dzs(k: int) -> np.ndarray:
            x = h(k)
            return np.where(x > 0.0, x, 1.0)

        def sep(k: int) -> np.ndarray:
            s = np.where(depth >= e[k + 2], m["sep_nominal"][k], (h(k) + h(k + 1)) / 2.0)
            return np.where(s > 0.0, s, 1.0)

        self.n_layers, self.n_cells = L, column.n_cells
        self.dz = _Rows(dzs)
        self.res_m = _Rows(lambda k: _row(tr, ktr, k) * h(k))
        self.fc_m = _Rows(lambda k: _row(tf, ktf, k) * h(k))
        self.sat_m = _Rows(lambda k: _row(ts, kts, k) * h(k))
        self.cap_tol = _Rows(lambda k: coef * (_row(ts, kts, k) * h(k)))
        self.pair_inv = _Rows(lambda k: 1.0 / dzs(k) + 1.0 / dzs(k + 1))
        if column.diffusion_mode == "constant":
            self.cond_down = _Rows(lambda k: _row(iface, kif, k) / sep(k))
            self.cond_up = self.cond_down
        else:
            self.cond_down = _Rows(lambda k: ((dcoef * _row(iface, kif, k)) * dzs(k)) / sep(k))
            self.cond_up = _Rows(lambda k: ((dcoef * _row(iface, kif, k)) * dzs(k + 1)) / sep(k))
        self.directional = column.diffusion_mode != "constant"
        self.closed_bottom = bool(closed)
        self.gravity_drainage = True
        self.res_theta0 = _row(tr, ktr, 0)
        self.stress, self.factor = column.theta_evaporation_stress, column.evaporation_factor


# ---- states ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _ns_type(column: FloatingSoilColumn) -> Any:
    return type(column._maps["active"])  # np.ndarray, or the cupy.ndarray type of the owner's device maps (no CuPy probe)


def _check_water(column: FloatingSoilColumn, water: Any, name: str) -> Any:
    if type(water) is not _ns_type(column) or water.dtype != np.float64 or not water.flags.c_contiguous or water.shape != (column.n_layers, column.n_cells):
        raise FloatingSoilError(f"{name} must be a C-contiguous float64 array of the column's namespace ({'cupy' if column.backend == 'cuda' else 'numpy'}) of shape {(column.n_layers, column.n_cells)}")
    if column.backend == "cuda" and int(water.device.id) != column.device_id:
        raise FloatingSoilError(f"{name} lives on device {int(water.device.id)} but the column's device is {column.device_id}; no implicit transfer")
    return water


def _check_water_values(column: FloatingSoilColumn, w: np.ndarray, initial: bool, *, who: str = "") -> None:
    """Value checks of a host water array against ITS owner (the state policies of the module docstring); raises `FloatingSoilError` with the first offending layer. `who` prefixes the message (e.g. an old reconciliation parent)."""
    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    active, ts, kts = column._maps["active"], column._maps["theta_saturation"], column.layouts["theta_saturation"]
    with np.errstate(all="ignore"):
        for k in range(column.n_layers):
            wl = w[k]
            if not np.all(np.isfinite(wl) & (wl >= 0.0)):
                raise FloatingSoilError(f"{who}layer {k}: water must be finite and >= 0 on every cell (inactive placeholders included)")
            h = porous_thickness(column, k)
            sat = _row(ts, kts, k) * h
            tol = coef * sat
            if initial:
                bad = active & np.where(h > 0.0, ~(wl < sat), wl != 0.0)
                why = "the initial state requires 0 <= W < saturation on porous layers (strictly unsaturated) and W == 0 on rock slots"
            else:
                bad = active & ~(wl <= sat + tol)
                why = "water exceeds the porous capacity (rock slots must hold exactly 0)"
            if bad.any():
                raise FloatingSoilError(f"{who}layer {k}: {why} ({int(bad.sum())} active cell(s), first {int(np.flatnonzero(bad)[0])})")


def _state(column: FloatingSoilColumn, water: Any, t_s: Any, *, initial: bool, quota: dict | None = None) -> FloatingSoilState:
    _check_column(column)
    if column.backend == "cuda":
        return _state_cuda(column, water, t_s, initial, quota or {})
    if quota and any(v for v in quota.values()):
        raise FloatingSoilError("quota arguments of this call apply only to the cuda backend")
    w = _check_water(column, water, "water_m")
    t = _real(t_s, "t_s")
    _check_water_values(column, w, initial)
    return _new_state(column, np.array(w, dtype=np.float64, copy=True), t)


def _state_cuda(column: FloatingSoilColumn, water: Any, t_s: Any, initial: bool, quota: dict) -> FloatingSoilState:
    from maple_syrup import floating_soil_layers_cuda as fcu

    cp = fcu._rc().require_cuda()
    fcu.gate_device(cp, column.device_id)  # the current device FIRST, before any metadata read of device data, reader or copy
    _check_water(column, water, "water_m")
    t = _real(t_s, "t_s")
    op = fcu.operation_terms("state_copy", column.n_layers, column.n_cells)
    chk = fcu.operation_terms("state_check", column.n_layers, column.n_cells)
    fcu.admit(cp, column.device_id, op["outputs"] + chk["scratch"], chk["host"], quota, "floating state construction")
    copy = cp.array(water, dtype=cp.float64, copy=True)  # the state owns a PRIVATE device copy (device-to-device; no host traffic)
    fcu.raise_flags(fcu.state_check(cp, column, copy, initial=initial), fcu.STATE_FLAG_TEXT, "floating state")
    return _new_state(column, copy, t)


def floating_initial_state(column: FloatingSoilColumn, water_m: Any, t_s: Any = 0.0, *, memory_budget_bytes: Any = None, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None,
                           host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingSoilState:
    """New-production initial state: strictly unsaturated actual porous soil, exactly 0 in rock slots (a wholly-rock column is legal); inactive placeholders per the published policy. CUDA: the quota arguments bound the device/host side of the copy and the 32-byte check."""
    return _state(column, water_m, t_s, initial=True, quota=dict(memory_budget_bytes=memory_budget_bytes, reserve_bytes=reserve_bytes, available_bytes_fn=available_bytes_fn,
                                                                 host_memory_budget_bytes=host_memory_budget_bytes, host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn))


def floating_continued_state(column: FloatingSoilColumn, water_m: Any, t_s: Any = 0.0, *, memory_budget_bytes: Any = None, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None,
                             host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingSoilState:
    """Transient/restart-equality state: `0 <= W <= sat_m + tol` (saturated is legal), rock slots exactly 0."""
    return _state(column, water_m, t_s, initial=False, quota=dict(memory_budget_bytes=memory_budget_bytes, reserve_bytes=reserve_bytes, available_bytes_fn=available_bytes_fn,
                                                                  host_memory_budget_bytes=host_memory_budget_bytes, host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn))


def _check_state(column: FloatingSoilColumn, state: Any) -> float:
    _check_column(column)
    if type(state) is not FloatingSoilState:
        raise FloatingSoilError(f"state must be a FloatingSoilState, got {type(state).__name__}")
    if state.column_sha256 != column.sha256 or type(state.token) is not str:
        raise FloatingSoilError("the state is bound to a different floating column (use reconcile_floating to move water between columns)")
    _check_water(column, state.water_m, "state.water_m")
    return _real(state.t_s, "state.t_s")


def _cuda_query(column: FloatingSoilColumn, state: Any, what: str, quota: dict | None) -> Any:
    from maple_syrup import floating_soil_layers_cuda as fcu

    cp = fcu._rc().require_cuda()
    fcu.gate_device(cp, column.device_id)  # the current device FIRST
    _check_state(column, state)
    op = fcu.operation_terms(what, column.n_layers, column.n_cells)
    fcu.admit(cp, column.device_id, op["device"] if what != "moisture" else op["outputs"], op["host"], quota or {}, f"floating {what}")
    return fcu.moisture(cp, column, state.water_m) if what == "moisture" else fcu.storage_depth(cp, column, state.water_m)


def floating_moisture(column: FloatingSoilColumn, state: FloatingSoilState, *, memory_budget_bytes: Any = None, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None,
                      host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> Any:
    """Volumetric water content `W / h` of the actual porous soil, exactly 0 where `h = 0` (no division by zero), as a fresh `(L, N)` array (device array on CUDA)."""
    quota = dict(memory_budget_bytes=memory_budget_bytes, reserve_bytes=reserve_bytes, available_bytes_fn=available_bytes_fn, host_memory_budget_bytes=host_memory_budget_bytes,
                 host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn)
    if column.backend == "cuda":
        return _cuda_query(column, state, "moisture", quota)
    _check_state(column, state)
    out = np.zeros_like(state.water_m)
    for k in range(column.n_layers):
        h = porous_thickness(column, k)
        np.divide(state.water_m[k], h, out=out[k], where=h > 0.0)
    return out


def floating_storage_m3(column: FloatingSoilColumn, state: FloatingSoilState, *, memory_budget_bytes: Any = None, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None,
                        host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> float:
    """Water volume over ACTIVE cells only (inactive placeholders are excluded). CUDA: a fixed block tree + one 8-byte read."""
    if column.backend == "cuda":
        quota = dict(memory_budget_bytes=memory_budget_bytes, reserve_bytes=reserve_bytes, available_bytes_fn=available_bytes_fn, host_memory_budget_bytes=host_memory_budget_bytes,
                     host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn)
        return _cuda_query(column, state, "storage", quota) * column.cell_area_m2
    _check_state(column, state)
    tot = 0.0
    for k in range(column.n_layers):
        tot += float(np.sum(state.water_m[k][column._maps["active"]]))
    return tot * column.cell_area_m2


# ---- the step ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _check_vec(column: FloatingSoilColumn, name: str, a: Any, shape: tuple) -> None:
    if type(a) is not _ns_type(column) or a.dtype != np.float64 or not a.flags.c_contiguous or tuple(a.shape) != shape:
        raise FloatingSoilError(f"{name} must be a C-contiguous float64 array of the column's namespace of shape {shape}")
    if column.backend == "cuda" and int(a.device.id) != column.device_id:
        raise FloatingSoilError(f"{name} lives on device {int(a.device.id)} but the column's device is {column.device_id}; no implicit transfer")


def _merge(a: ss.StepArrays, b: ss.StepArrays, open_mask: np.ndarray) -> ss.StepArrays:
    """Per-cell selection of the open-pass result where `open_mask`, else the closed-pass result (cells are independent in the body)."""
    pick = lambda x, y: np.where(open_mask, y, x)  # noqa: E731
    return ss.StepArrays(pick(a.water_m, b.water_m), pick(a.percolation_m, b.percolation_m), pick(a.diffusion_m, b.diffusion_m), pick(a.surface_return_m, b.surface_return_m),
                         pick(a.drainage_m, b.drainage_m), pick(a.evaporation_m, b.evaporation_m), pick(a.evaporation_unmet_m, b.evaporation_unmet_m), pick(a.rows, b.rows))


def _numpy_arrays(column: FloatingSoilColumn, water: np.ndarray, added: Any, pet: np.ndarray, dt: float, share: Any, recharge: float | None) -> ss.StepArrays:
    m = column._maps
    active, bottom_open = m["active"], m["bottom_open"] & m["active"]
    inactive = ~active
    w_in = water
    if inactive.any() and bool(np.any(water[:, inactive])):  # published inactive policy: no capacity law on placeholders -> the body sees zeros there; the originals are restored below
        w_in = water.copy()
        w_in[:, inactive] = 0.0
    want_open = column.bottom_drainage == "open" and bool(bottom_open.any())
    want_closed = column.bottom_drainage == "closed" or bool((active & ~bottom_open).any()) or not want_open
    res_closed = res_open = None
    if want_closed:
        res_closed = ss._reference_step_body(_FloatingParams(column, True), active, w_in, added, pet, dt, share, recharge)
    if want_open:
        res_open = ss._reference_step_body(_FloatingParams(column, False), active, w_in, added, pet, dt, share, recharge)
    if res_open is None:
        out = res_closed
    elif res_closed is None:
        out = res_open
    else:
        out = _merge(res_closed, res_open, bottom_open)
    if w_in is not water:
        out.water_m[:, inactive] = water[:, inactive]
    return out


def _run(column: FloatingSoilColumn, state: FloatingSoilState, dt: float, added: Any, pet: np.ndarray, share: Any, pulse: float | None) -> ss.StepArrays:
    if column.backend == "numba":
        from maple_syrup import floating_soil_layers_numba as fnb

        return fnb.step(column, state.water_m, added, pet, dt, share, pulse)
    return _numpy_arrays(column, state.water_m, added, pet, dt, share, pulse)


def _finish(column: FloatingSoilColumn, arrays: ss.StepArrays, dt: float, t1: float, recharge: bool) -> FloatingStep:
    packet = ss._read_packet(np, arrays)
    budget = ss._resolve(column, packet, recharge=recharge)  # raises the typed step/fit errors of the shared body (SubsurfaceFitFailure for a pulse that does not fit)
    return FloatingStep(_new_state(column, arrays.water_m, t1), dt, arrays.surface_return_m, arrays.drainage_m, arrays.evaporation_m, arrays.evaporation_unmet_m, arrays.percolation_m, arrays.diffusion_m, budget)


def _admit_step(column: FloatingSoilColumn, *, recharge: bool, share: bool, quota: dict) -> None:
    """CPU: the modelled host step bytes against the budget/reading. CUDA: handled by `_step_cuda` (device gate first, device and host separately)."""
    est = estimate_floating_memory(column.n_layers, column.n_cells, layouts=column.layouts, diffusion_mode=column.diffusion_mode, backend=column.backend, with_recharge=recharge, reserve_bytes=_count(quota["reserve_bytes"], "reserve_bytes"))
    budget = quota["memory_budget_bytes"]
    _admit(est.step_bytes + _count(quota["reserve_bytes"], "reserve_bytes"), None if budget is None else _count(budget, "memory_budget_bytes"), quota["available_bytes_fn"], "floating step")


def _quota(memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, column: FloatingSoilColumn) -> dict:
    if column.backend != "cuda" and (host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None):
        raise FloatingSoilError("host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to the cuda backend")
    return dict(memory_budget_bytes=memory_budget_bytes, reserve_bytes=reserve_bytes, available_bytes_fn=available_bytes_fn, host_memory_budget_bytes=host_memory_budget_bytes,
                host_reserve_bytes=host_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn)


def _step_cuda(column: FloatingSoilColumn, state: FloatingSoilState, dt: float, t1: float, added: Any, pet: Any, share: Any, pulse: float | None, quota: dict) -> FloatingStep:
    from maple_syrup import floating_soil_layers_cuda as fcu

    cp = fcu._rc().require_cuda()
    op = fcu.operation_terms("step" if pulse is None else "step_with_recharge", column.n_layers, column.n_cells)
    fcu.admit(cp, column.device_id, op["device"], op["host"], quota, "floating step")
    arrays, packet = fcu.run_step(cp, column, state.water_m, added, pet, share, dt, pulse)
    budget = ss._resolve(column, packet, recharge=pulse is not None)  # the existing closure bound and the typed step/fit errors (SubsurfaceFitFailure for a pulse that does not fit)
    return FloatingStep(_new_state(column, arrays.water_m, t1), dt, arrays.surface_return_m, arrays.drainage_m, arrays.evaporation_m, arrays.evaporation_unmet_m, arrays.percolation_m, arrays.diffusion_m, budget)


def _gate(column: Any) -> None:
    """CUDA: the CURRENT device is checked before anything else (state reads, readers, copies, accounting)."""
    if type(column) is FloatingSoilColumn and column.backend == "cuda":
        from maple_syrup import floating_soil_layers_cuda as fcu

        fcu.gate_device(fcu._rc().require_cuda(), column.device_id)


def floating_step(column: FloatingSoilColumn, state: FloatingSoilState, dt_s: Any, *, added_water_m: Any, potential_evaporation_m_s: Any, evaporation_share: Any = None, memory_budget_bytes: Any = None,
                  reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                  host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingStep:
    """One pure step (module docstring); validation and admission precede any numerical work; a refused candidate publishes nothing and the inputs are never written."""
    _gate(column)
    t0 = _check_state(column, state)
    dt = ss._check_dt(dt_s)
    t1 = t0 + dt
    if not math.isfinite(t1):
        raise FloatingSoilError("the candidate time t + dt is not finite")
    _check_vec(column, "added_water_m", added_water_m, (column.n_layers, column.n_cells))
    _check_vec(column, "potential_evaporation_m_s", potential_evaporation_m_s, (column.n_cells,))
    if evaporation_share is not None:
        _check_vec(column, "evaporation_share", evaporation_share, (column.n_cells,))
    quota = _quota(memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, column)
    if column.backend == "cuda":
        return _step_cuda(column, state, dt, t1, added_water_m, potential_evaporation_m_s, evaporation_share, None, quota)
    _admit_step(column, recharge=False, share=evaporation_share is not None, quota=quota)
    return _finish(column, _run(column, state, dt, added_water_m, potential_evaporation_m_s, evaporation_share, None), dt, t1, False)


def floating_step_with_recharge(column: FloatingSoilColumn, state: FloatingSoilState, dt_s: Any, *, recharge_depth_m: Any, potential_evaporation_m_s: Any, evaporation_share: Any = None,
                                memory_budget_bytes: Any = None, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None,
                                host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingStep:
    """Phase 1.5 pulse-aware daily step with the SAME fit/fallback semantics as `subsurface.step_with_recharge` (`SubsurfaceFitFailure` for a pulse that does not fit; nothing clipped; `dt_s > 0`)."""
    _gate(column)
    t0 = _check_state(column, state)
    dt = ss._check_dt(dt_s)
    if not dt > 0.0:
        raise FloatingSoilError("the pulse-aware daily step requires dt_s > 0 (no zero-time rain step)")
    pulse = _real(recharge_depth_m, "recharge_depth_m")
    if pulse < 0.0:
        raise FloatingSoilError(f"recharge_depth_m must be >= 0, got {pulse!r}")
    t1 = t0 + dt
    if not math.isfinite(t1) or not t1 > t0:
        raise FloatingSoilError("the positive interval dt does not advance the native clock (t + dt == t) or is not finite")
    _check_vec(column, "potential_evaporation_m_s", potential_evaporation_m_s, (column.n_cells,))
    if evaporation_share is not None:
        _check_vec(column, "evaporation_share", evaporation_share, (column.n_cells,))
    quota = _quota(memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, column)
    if column.backend == "cuda":
        return _step_cuda(column, state, dt, t1, None, potential_evaporation_m_s, evaporation_share, pulse, quota)
    _admit_step(column, recharge=True, share=evaporation_share is not None, quota=quota)
    return _finish(column, _run(column, state, dt, None, potential_evaporation_m_s, evaporation_share, pulse), dt, t1, True)


# ---- staged zero-time reconciliation ------------------------------------------------------------------------------------------------------------------------------------------------------
def _water_digest(water: np.ndarray, t_s: float) -> str:
    h = hashlib.sha256(f"{water.shape}|{t_s!r}\0".encode())
    flat = water.reshape(-1)
    step = max(1, DEFAULT_HASH_CHUNK_BYTES // 8)
    for i in range(0, flat.size, step):
        h.update(memoryview(flat[i:i + step]).cast("B"))
    return h.hexdigest()


def _reconcile_cuda(old_column: FloatingSoilColumn, old_state: FloatingSoilState, new_column: FloatingSoilColumn, quota: dict) -> ReconcileCandidate:
    """HOT resident reconciliation: ONE 56-byte packet, NO content hash and no download of water or properties. The identity is a CAPABILITY/generation identity: the minted state tokens, the cold column identities and the cold mask/edges hashes
    (never a physical content digest); native adoption (slice E) binds generation/once-only."""
    from maple_syrup import floating_soil_layers_cuda as fcu

    cp = fcu._rc().require_cuda()
    _gate(old_column)
    _gate(new_column)  # both columns must live on the CURRENT device, checked before the state is read
    _check_column(new_column)
    _check_area(old_column, new_column)  # before any reader, allocation or kernel
    t = _check_state(old_column, old_state)
    if new_column.device_id != old_column.device_id or new_column.backend != "cuda" or old_column.backend != "cuda":
        raise FloatingSoilError("reconciliation needs both columns on the same device and backend (no implicit transfer or backend change)")
    if (old_column.n_layers, old_column.n_cells) != (new_column.n_layers, new_column.n_cells) or old_column.mask_sha256 != new_column.mask_sha256 or old_column.edges_sha256 != new_column.edges_sha256:
        raise FloatingSoilError("reconciliation needs the same layer count, cell count, active mask and nominal edges in the old and new columns")
    L, N = new_column.n_layers, new_column.n_cells
    op = fcu.operation_terms("reconcile", L, N)
    fcu.admit(cp, new_column.device_id, op["device"], op["host"], quota, "floating reconciliation")
    w, perc, surf, exp_, totals, orf, flagged = fcu.run_reconcile(cp, new_column, old_column, old_state.water_m)
    if orf & ((1 << 20) | (1 << 21)):
        raise FloatingSoilError(f"the OLD state violates its own owner ({flagged} cell(s); {'non-finite or negative water' if orf & (1 << 20) else 'water above the old porous capacity / in a rock slot'}); reconciliation needs a valid accepted parent, "
                                "no state is minted")
    if flagged:
        bit = (orf & -orf).bit_length() - 1
        raise FloatingSoilError(f"the reconciled candidate is not a valid continued state of the new column ({flagged} cell(s); first failure bit {bit}); no state is minted")
    area = new_column.cell_area_m2
    before, after, s_m3, e_m3, moved_m3 = (v * area for v in totals)
    residual = (after - before) + (s_m3 + e_m3)
    bound = volume_roundoff_bound_m3(4 * L * N + 4, max(abs(before), abs(after), abs(s_m3), abs(e_m3)))
    if not (math.isfinite(residual) and abs(residual) <= bound):
        raise FloatingSoilError(f"reconciliation closure violated: residual {residual!r} m3 exceeds the FP64 bound {bound!r} m3; no state is minted")
    new_state = _new_state(new_column, w, t)
    receipt = ReconcileReceipt(old_column.sha256, new_column.sha256, None, t, before, after, s_m3, e_m3, residual, bound, 4 * L * N + 4, moved_m3, "capability_token", old_state.token, new_state.token)
    return ReconcileCandidate(new_state, perc, surf, exp_, receipt)


SELECTED_RECONCILE_TAG = "selected_zero_time_reconcile/v1"


@dataclass(frozen=True)
class SelectedReconcileReceipt:
    """Receipt of a SELECTED zero-time reconciliation. It proves ONLY the selected operation (`n_selected` cells of the actual domain went through the new-capacity cascade/saturation body; every other cell was copied bit for bit with zero
    flux); it does NOT prove that the selection covers every cell whose geometry/properties changed (`proves_complete_geometry_refresh` is always False: the owner that chose the selection proves completeness)."""

    operation: str
    old_column_sha256: str
    new_column_sha256: str
    t_s: float
    n_selected: int
    n_cells: int
    storage_before_m3: float  # selected cells
    storage_after_m3: float
    surface_return_m3: float
    bottom_export_m3: float
    residual_m3: float
    bound_m3: float
    n_terms: int
    moved_percolation_m3: float
    selection_identity: str  # a minted per-call token, NEVER a content digest of the mask
    identity_kind: str
    old_state_token: str
    new_state_token: str
    proves_complete_geometry_refresh: bool = False


@dataclass(frozen=True, eq=False)
class SelectedReconcileCandidate:
    state: FloatingSoilState  # a valid continued state of the ACTUAL new column over its FULL actual domain
    percolation_m: Any  # (L-1, N), zero outside the selection
    surface_return_m: Any  # (N,), zero outside the selection
    bottom_export_m: Any  # (N,), zero outside the selection
    receipt: SelectedReconcileReceipt


def _check_selected_columns(old_column: FloatingSoilColumn, new_column: FloatingSoilColumn) -> None:
    if (old_column.n_layers, old_column.n_cells) != (new_column.n_layers, new_column.n_cells) or old_column.mask_sha256 != new_column.mask_sha256 or old_column.edges_sha256 != new_column.edges_sha256:
        raise FloatingSoilError("reconciliation needs the same layer count, cell count, active mask and nominal edges in the old and new columns")
    if new_column.backend != old_column.backend or new_column.device_id != old_column.device_id:
        raise FloatingSoilError("reconciliation needs the same backend and device in the old and new columns (no implicit transfer or backend change)")
    root_o = old_column.sha256 if old_column.parent_sha256 is None else old_column.parent_sha256
    root_n = new_column.sha256 if new_column.parent_sha256 is None else new_column.parent_sha256
    if root_o != root_n:
        raise FloatingSoilError("the selected reconciliation needs two columns of the SAME prepared root (static maps shared); a foreign root is refused")


def reconcile_floating_selected(old_column: FloatingSoilColumn, old_state: FloatingSoilState, new_column: FloatingSoilColumn, selected_mask: Any, *, memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                                available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                                host_available_bytes_fn: Callable[[], int] | None = None) -> SelectedReconcileCandidate:
    """PURE zero-time SELECTED reconciliation: the accepted `old_state` of `old_column` against `new_column` (an official geometry view of the same prepared root), applying the unchanged D1 new-capacity gravity cascade and saturation
    return ONLY to the cells of `selected_mask` (a bool `(N,)` array of the owner's namespace that must be a SUBSET of the column's actual active cells). The mask is an OPERATION selector: the columns' actual active maps are not
    changed and no smaller fake column exists. Every unselected cell keeps its water bit for bit and has zero percolation/return/export; the returned state is validated as a continued state of the actual new column over its
    full domain, so an unselected cell that is invalid under the new geometry is a typed refusal. No time passes and no E/T/diffusion/rain/drain term is added. The receipt (`SelectedReconcileReceipt`) names the operation and does not
    claim that the selection is a complete geometry refresh. The old full `reconcile_floating` is unchanged."""
    _check_column(old_column)
    if old_column.backend == "cuda":
        return _reconcile_selected_cuda(old_column, old_state, new_column, selected_mask, _quota(memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, old_column))
    _check_column(new_column)
    _check_area(old_column, new_column)
    _quota(memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, old_column)
    t = _check_state(old_column, old_state)
    _check_selected_columns(old_column, new_column)
    N, L = new_column.n_cells, new_column.n_layers
    if type(selected_mask) is not np.ndarray or selected_mask.dtype != np.bool_ or selected_mask.shape != (N,) or not selected_mask.flags.c_contiguous:
        raise FloatingSoilError(f"selected_mask must be a C-contiguous bool numpy array of shape ({N},)")
    est = estimate_floating_memory(L, N, layouts=new_column.layouts, diffusion_mode=new_column.diffusion_mode, backend=new_column.backend, reserve_bytes=_count(reserve_bytes, "reserve_bytes"))
    _admit(est.step_bytes + 2 * N + _count(reserve_bytes, "reserve_bytes"), None if memory_budget_bytes is None else _count(memory_budget_bytes, "memory_budget_bytes"), available_bytes_fn, "floating selected reconciliation")
    if bool((selected_mask & ~new_column._maps["active"]).any()):
        raise FloatingSoilError("selected_mask must be a subset of the column's actual active cells")
    _check_water_values(old_column, old_state.water_m, False, who="the OLD state violates its own owner: ")
    w, perc, surf, exp_ = _reconcile_numpy(new_column, old_state.water_m, selected_mask)
    n_sel = int(selected_mask.sum())
    area = new_column.cell_area_m2
    before = sum(float(np.sum(old_state.water_m[k][selected_mask])) for k in range(L)) * area
    after = sum(float(np.sum(w[k][selected_mask])) for k in range(L)) * area
    s_m3, e_m3 = float(np.sum(surf[selected_mask])) * area, float(np.sum(exp_[selected_mask])) * area
    residual = (after - before) + (s_m3 + e_m3)
    n_terms = 4 * L * n_sel + 4
    bound = volume_roundoff_bound_m3(n_terms, max(abs(before), abs(after), abs(s_m3), abs(e_m3)))
    if not (math.isfinite(residual) and abs(residual) <= bound):
        raise FloatingSoilError(f"selected reconciliation closure violated: residual {residual!r} m3 exceeds the FP64 bound {bound!r} m3; no state is minted")
    new_state = _state(new_column, w, t, initial=False)  # the FULL actual domain of the new column is validated
    moved = float(sum(np.sum(np.abs(perc[k][selected_mask])) for k in range(L - 1))) * area if L > 1 else 0.0
    receipt = SelectedReconcileReceipt(SELECTED_RECONCILE_TAG, old_column.sha256, new_column.sha256, t, n_sel, N, before, after, s_m3, e_m3, residual, bound, n_terms, moved, "selection_token:" + secrets.token_hex(8),
                                       "cpu_no_content_hash", old_state.token, new_state.token)
    return SelectedReconcileCandidate(new_state, perc, surf, exp_, receipt)


def _reconcile_selected_cuda(old_column: FloatingSoilColumn, old_state: FloatingSoilState, new_column: FloatingSoilColumn, selected_mask: Any, quota: dict) -> SelectedReconcileCandidate:
    """CUDA selected operation: the SAME resident reconcile kernel/ABI with the OPERATION mask passed as its active pointer. Order: current device (both columns) -> metadata/type/root checks -> admission (the reconcile outputs and scratch
    plus two N-byte selection temporaries; device and host separately) -> selection subset/count (two counted 8-byte reads) -> ONE 32-byte full-OLD-domain state check of the accepted parent -> kernel + ONE 56-byte packet -> old-state refusal -> ONE 32-byte full-domain state check of the output
    against the actual new column -> receipt (capability tokens, never a content hash). Nothing is downloaded but those packets."""
    from maple_syrup import floating_soil_layers_cuda as fcu

    cp = fcu._rc().require_cuda()
    _gate(old_column)
    _gate(new_column)
    _check_column(new_column)
    _check_area(old_column, new_column)
    t = _check_state(old_column, old_state)
    _check_selected_columns(old_column, new_column)
    L, N = new_column.n_layers, new_column.n_cells
    if type(selected_mask) is not cp.ndarray or selected_mask.dtype != np.bool_ or selected_mask.shape != (N,) or not selected_mask.flags.c_contiguous or int(selected_mask.device.id) != new_column.device_id:
        raise FloatingSoilError(f"selected_mask must be a C-contiguous bool cupy array of shape ({N},) on the owner's device")
    op = fcu.operation_terms("reconcile", L, N)
    chk = fcu.operation_terms("state_check", L, N)
    fcu.admit(cp, new_column.device_id, op["device"] + chk["scratch"] + 2 * N, max(op["host"], chk["host"]), quota, "floating selected reconciliation")
    n_sel, n_out = fcu.selection_counts(cp, new_column, selected_mask)
    if n_out:
        raise FloatingSoilError(f"selected_mask must be a subset of the column's actual active cells ({n_out} cell(s) outside)")
    # the OLD state must be valid for its OWN owner over its FULL actual domain: the reconcile kernel's old-capacity check runs only on the OPERATION-selected cells (its active pointer is the selector), so the
    # unselected cells are validated here by the existing resident state check (ONE 32-byte packet) BEFORE the selected kernel; a violation is refused with nothing computed or published
    old_report = fcu.state_check(cp, old_column, old_state.water_m, initial=False)
    if old_report[1]:
        try:
            fcu.raise_flags(old_report, fcu.STATE_FLAG_TEXT, "the OLD state violates its own owner over its actual domain (a valid accepted parent is required; the selection does not relax this)")
        except FloatingSoilError as exc:
            raise FloatingSoilError(str(exc)) from None
    w, perc, surf, exp_, totals, orf, flagged = fcu.run_reconcile(cp, new_column, old_column, old_state.water_m, active=selected_mask)
    if orf & ((1 << 20) | (1 << 21)):
        raise FloatingSoilError(f"the OLD state violates its own owner ({flagged} cell(s); {'non-finite or negative water' if orf & (1 << 20) else 'water above the old porous capacity / in a rock slot'}); a valid accepted parent is required, "
                                "no state is minted")
    if flagged:
        bit = (orf & -orf).bit_length() - 1
        raise FloatingSoilError(f"the selected candidate is not a valid continued state of the new column ({flagged} cell(s); first failure bit {bit}); no state is minted")
    fcu.raise_flags(fcu.state_check(cp, new_column, w, initial=False), fcu.STATE_FLAG_TEXT, "selected reconciliation output (an unselected cell is invalid under the new geometry)")
    area = new_column.cell_area_m2
    before, after, s_m3, e_m3, moved_m3 = (v * area for v in totals)
    residual = (after - before) + (s_m3 + e_m3)
    n_terms = 4 * L * n_sel + 4
    bound = volume_roundoff_bound_m3(n_terms, max(abs(before), abs(after), abs(s_m3), abs(e_m3)))
    if not (math.isfinite(residual) and abs(residual) <= bound):
        raise FloatingSoilError(f"selected reconciliation closure violated: residual {residual!r} m3 exceeds the FP64 bound {bound!r} m3; no state is minted")
    new_state = _new_state(new_column, w, t)
    receipt = SelectedReconcileReceipt(SELECTED_RECONCILE_TAG, old_column.sha256, new_column.sha256, t, n_sel, N, before, after, s_m3, e_m3, residual, bound, n_terms, moved_m3, "selection_token:" + secrets.token_hex(8),
                                       "capability_token", old_state.token, new_state.token)
    return SelectedReconcileCandidate(new_state, perc, surf, exp_, receipt)


def _check_area(old_column: FloatingSoilColumn, new_column: FloatingSoilColumn) -> None:
    """The SAME physical columns must carry the identical cell area: water volume is conserved, never rescaled to hide a mismatch."""
    if old_column.cell_area_m2 != new_column.cell_area_m2:
        raise FloatingSoilError(f"reconciliation needs the identical cell_area_m2 in the old and new columns (got {old_column.cell_area_m2!r} and {new_column.cell_area_m2!r}); inventories are never rescaled")


def reconcile_floating(old_column: FloatingSoilColumn, old_state: FloatingSoilState, new_column: FloatingSoilColumn, *, memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                       available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                       host_available_bytes_fn: Callable[[], int] | None = None) -> ReconcileCandidate:
    """PURE zero-time staged reconciliation of the accepted `old_state` against the proposed `new_column` (module docstring). The old state must be valid for `old_column`; the new column must share L, N, the active mask and the nominal edges."""
    _check_column(old_column)
    if old_column.backend == "cuda":
        return _reconcile_cuda(old_column, old_state, new_column, _quota(memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, old_column))
    _check_column(new_column)
    _check_area(old_column, new_column)  # before any quota reader, allocation or kernel
    _quota(memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn, old_column)  # refuses host_* quotas on the CPU backends
    t = _check_state(old_column, old_state)
    if (old_column.n_layers, old_column.n_cells) != (new_column.n_layers, new_column.n_cells) or not np.array_equal(old_column._maps["active"], new_column._maps["active"]) or not np.array_equal(old_column._maps["edges"], new_column._maps["edges"]):
        raise FloatingSoilError("reconciliation needs the same layer count, cell count, active mask and nominal edges in the old and new columns")
    if new_column.backend != old_column.backend:
        raise FloatingSoilError("reconciliation needs the same backend in the old and new columns (no implicit backend change)")
    L, N = new_column.n_layers, new_column.n_cells
    est = estimate_floating_memory(L, N, layouts=new_column.layouts, diffusion_mode=new_column.diffusion_mode, backend=new_column.backend, reserve_bytes=_count(reserve_bytes, "reserve_bytes"))
    _admit(est.step_bytes + _count(reserve_bytes, "reserve_bytes"), None if memory_budget_bytes is None else _count(memory_budget_bytes, "memory_budget_bytes"), available_bytes_fn, "floating reconciliation")
    # The old-owner value check allocates O(N) row scratch (already inside the modelled `step_bytes`), so it runs only AFTER the metadata-only admission succeeded and before any physics/output allocation. Exceeding the NEW capacity is legal
    # only inside the stage; the parent must be valid for ITS owner.
    _check_water_values(old_column, old_state.water_m, False, who="the OLD state violates its own owner: ")  # exceeding the NEW capacity is legal only inside the stage; the parent must be valid for ITS owner
    if new_column.backend == "numba":
        from maple_syrup import floating_soil_layers_numba as fnb

        w, perc, surf, exp_ = fnb.reconcile(new_column, old_state.water_m)
    else:
        w, perc, surf, exp_ = _reconcile_numpy(new_column, old_state.water_m)
    active = new_column._maps["active"]
    area = new_column.cell_area_m2
    before = sum(float(np.sum(old_state.water_m[k][active])) for k in range(L)) * area
    after = sum(float(np.sum(w[k][active])) for k in range(L)) * area
    s_m3, e_m3 = float(np.sum(surf[active])) * area, float(np.sum(exp_[active])) * area
    residual = (after - before) + (s_m3 + e_m3)
    n_terms = 4 * L * N + 4
    bound = volume_roundoff_bound_m3(n_terms, max(abs(before), abs(after), abs(s_m3), abs(e_m3)))
    if not (math.isfinite(residual) and abs(residual) <= bound):
        raise FloatingSoilError(f"reconciliation closure violated: residual {residual!r} m3 exceeds the FP64 bound {bound!r} m3; no state is minted")
    new_state = _state(new_column, w, t, initial=False)  # a candidate is a valid CONTINUED state of the new column (rock slots 0, within capacity) or it is not minted
    moved = float(sum(np.sum(np.abs(perc[k][active])) for k in range(L - 1))) * area if L > 1 else 0.0
    receipt = ReconcileReceipt(old_column.sha256, new_column.sha256, _water_digest(old_state.water_m, t), t, before, after, s_m3, e_m3, residual, bound, n_terms, moved, "content_sha256", old_state.token,
                               new_state.token)
    return ReconcileCandidate(new_state, perc, surf, exp_, receipt)


def _reconcile_numpy(column: FloatingSoilColumn, water: np.ndarray, operation_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-layer vectorised stage: gravity cascade on the NEW field capacity (same expression structure as the step's cascade), then saturation return to the surface. Inactive cells are copied unchanged with zero fluxes."""
    m = column._maps
    L, N = column.n_layers, column.n_cells
    active, closed = (m["active"] if operation_mask is None else operation_mask), ~m["bottom_open"]  # D3: an explicit OPERATION mask replaces the domain only for the selected operation (the column's actual active map is untouched)
    tf, ts = m["theta_field_capacity"], m["theta_saturation"]
    ktf, kts = column.layouts["theta_field_capacity"], column.layouts["theta_saturation"]
    w = np.array(water, dtype=np.float64, copy=True)
    perc = np.zeros((max(L - 1, 0), N))
    n_if = L - 1
    fc = lambda k: _row(tf, ktf, k) * porous_thickness(column, k)  # noqa: E731
    sat = lambda k: _row(ts, kts, k) * porous_thickness(column, k)  # noqa: E731
    with np.errstate(all="ignore"):
        room = np.zeros(N)
        cap_rows = np.zeros((max(n_if, 0), N))
        for k in range(n_if - 1, -1, -1):  # deeper room measured once on the INPUT water (as the step does), accumulated bottom-up
            term = sat(k + 1) - w[k + 1]
            term = np.where(term < 0.0, 0.0, term)
            room = room + term
            cap_rows[k] = room
        carry = np.zeros(N)
        for k in range(L):
            wk = w[k] + carry
            ex = wk - fc(k)
            pos = ex > 0.0
            if k < n_if:
                cl_fwd = np.where(ex < cap_rows[k], ex, cap_rows[k])
            else:
                cl_fwd = np.zeros(N)
            fwd = np.where(closed, cl_fwd, ex)
            fwd = np.where(pos & active, fwd, 0.0)
            w[k] = np.where(active, wk - fwd, water[k])
            carry = fwd
            if k < n_if:
                perc[k] = carry
        export = np.where(active & ~closed, carry, 0.0)
        surface = np.zeros(N)
        for k in range(L):
            ex = w[k] - sat(k)
            pos = active & (ex > 0.0)
            w[k] = np.where(pos, w[k] - ex, w[k])
            surface = np.where(pos, surface + ex, surface)
    return w, perc, surface, export
