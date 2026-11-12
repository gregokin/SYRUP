"""Surface-only lateral hydraulics seam (Phase 1 surface stage, task phase1_surface_stage_20261110).

This module is the SHARED INTERFACE of the extracted surface operation: the purpose-defined state/result records, the
error classes, the strict input contracts, the memory estimate/admission helpers and the backend selector. It holds NO
physics: the NumPy reference arithmetic lives in `experimental_hydrology` (`CpuSurfaceRouter`), the compiled CPU form in
`experimental_numba` (`NumbaSurfaceRouter`) and the CUDA form in `experimental_cuda` (`CudaSurfaceRouter`). Those three
modules import this one; this one imports none of them at module level (the selector imports lazily), so there is no
cycle and importing this module loads neither Numba nor CuPy.

WHAT THE OPERATION IS. One pure lateral step of either EXPERIMENTAL hydraulic alternative (explicit D4 kinematic wave or
uniform-grid local inertia, equations in `experimental_hydrology`) applied to a surface depth that has ALREADY been through
whatever vertical surface/soil exchange the caller accepted for the same interval:

    router = build_surface_router("local_inertial", graph, backend="numpy", geometry=geometry, control=HydraulicControl())
    state  = router.initial_state(depth_after_vertical_exchange, t_s=t_accepted)       # or SurfaceState(t, h, qx, qy)
    result = router.step(state, dt)      # SurfaceStep: routed depth + momentum at t + dt, diagnostics; raises before a result

Inputs: the post-vertical surface depth `(ny, nx)` float64, the existing signed face momentum `qx (ny, nx+1)` / `qy (ny+1, nx)`
for local inertia (None for the explicit method), the accepted START time of the interval and the positive `dt`. Output: the
routed depth and momentum at the physical END time `t + dt` (ONE time advance for the whole vertical+lateral interval: a
vertical stage composed with this one must leave `t` untouched), the end-of-step cell velocity, the stage face depths, the
signed face volumes and unit fluxes, the export through the open outlet faces, the storage change / budget residual, the
CFL quantity and the donor-limiter activity. Nothing else: no rain, infiltration, evaporation, transpiration, growth,
sediment, bed mutation, reroute, checkpoint, soil bucket or `ColumnParameters`/`ColumnStep`/`HydraulicState`.

CONTRACTS (every backend). Pure: fresh outputs, no input written, raise before a result exists. `HydraulicStepRejected` is
the ONLY recoverable class (CFL bound or, without the limiter, a negative depth): retry the SAME state with a smaller dt.
Strict inputs: exact `numpy.ndarray` / `cupy.ndarray` (no subclass, masked array, list or other namespace; no conversion,
no transfer), float64, exact shapes, finite depth >= 0 on EVERY cell (inactive included), finite face momentum that is zero
on closed faces and never points into the domain on an open outlet face. Inactive cells keep their depth and exchange
nothing. The original equations, operation order, controls (`HydraulicControl`: cfl_max, limiter, friction_treatment),
default friction/limiter, legacy outlet faces, rejection classes/precedence and `routing.BALANCE_RTOL` tolerances are those of
the historical solvers, which now delegate their lateral stage to the same code (their public API and bucket output are
unchanged; see `experimental_hydrology.compose_hydraulic_step`).

MEMORY. `estimate_surface_memory` is the declared upper bound of what a router OWNS (static preparation) and what ONE step
ALLOCATES (outputs + temporaries + packet), per backend and method, plus the caller's reserve; `required_bytes` is compared
with `available_bytes_fn()` BEFORE preparation allocates and BEFORE every step allocates. See `MEMORY_SCOPE` for what the
bound deliberately excludes. The historical solvers pass `UNLIMITED_ADMISSION` (their old API had no admission; none is
retrofitted).

Qualification: this is an EXTRACTION of existing experimental hydraulics into a surface-only operation with its own
contracts and tests; it is not a new physics model, not a full storm/pond composition, not GPU hardware qualification and
not a performance claim. `experimental_hydrology.QUALIFICATION_STATUS` still applies to the hydraulics themselves.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

import numpy as np

__all__ = [
    "INPUT_DEPTH_BITS",
    "MEMORY_SCOPE",
    "SURFACE_BACKENDS",
    "SURFACE_METHODS",
    "SURFACE_PACKET_BYTES",
    "UNLIMITED_ADMISSION",
    "ExperimentalGeometryError",
    "ExperimentalHydrologyError",
    "HydraulicStepRejected",
    "SurfaceMemory",
    "SurfaceMemoryError",
    "SurfaceState",
    "SurfaceStep",
    "admit_bytes",
    "advance_time",
    "build_surface_router",
    "check_dt_cap",
    "check_positive_dt",
    "check_state_time",
    "check_surface_structure",
    "default_device_available_bytes_fn",
    "default_host_available_bytes",
    "effective_free_device_bytes",
    "estimate_surface_memory",
    "validate_surface_state",
]

SURFACE_METHODS = ("explicit", "local_inertial")
SURFACE_BACKENDS = ("numpy", "numba", "cuda")
# Flag bit (per method) raised by a router when the INPUT surface depth is non-finite or negative anywhere (inactive cells
# included). Disjoint from `experimental_hydrology.EXPLICIT_BITS` / `LOCAL_BITS`; resolved FIRST by `resolve_flags`. The
# historical solvers mask it out: their depth is the accepted column's output, validated by the column stage.
INPUT_DEPTH_BITS = {"explicit": 1 << 4, "local_inertial": 1 << 7}
SURFACE_PACKET_BYTES = 16 * 8  # the CUDA lateral packet (`experimental_cuda.PACKET_WORDS` words); asserted there
_F8 = 8


class ExperimentalHydrologyError(ValueError):
    """Invalid request, state, geometry or a failed check of an experimental solver/router. Raised before a result exists; no
    caller array is modified."""


class HydraulicStepRejected(ExperimentalHydrologyError):
    """The step is valid but too large for this state (CFL bound or, without the limiter, a negative depth): retry the SAME
    state with a smaller dt. Every other error is deliberately not this class."""


class ExperimentalGeometryError(ExperimentalHydrologyError):
    """Unsupported or invalid geometry/boundary specification. Raised while the geometry is built."""


class SurfaceMemoryError(MemoryError):
    """The declared working-array requirement plus the reserves exceeds the reported available bytes; NOTHING was allocated.
    A reading is not a reservation and the budget is never raised silently."""


# --- scalar checks ---------------------------------------------------------------------------------------------------
def _real(value: Any, name: str, error: type[Exception] = ExperimentalHydrologyError) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise error(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def check_state_time(t_s: Any, name: str = "state.t_s") -> float:
    """A real (bool, str and None are refused, never coerced) finite number >= 0."""
    t = _real(t_s, name)
    if not (math.isfinite(t) and t >= 0.0):
        raise ExperimentalHydrologyError(f"{name} must be finite and >= 0, got {t!r}")
    return t


def check_dt_cap(value: Any, name: str = "state.next_dt_cap_s") -> float | None:
    """None (a fresh event) or a real (bool, str, bytes, arrays are refused, never coerced), finite, strictly positive step cap in
    s; returns it as a float."""
    if value is None:
        return None
    cap = _real(value, name)
    if not (math.isfinite(cap) and cap > 0.0):
        raise ExperimentalHydrologyError(f"{name} must be None or finite and > 0, got {cap!r}")
    return cap


def check_positive_dt(dt_s: Any, name: str = "dt_s") -> float:
    """A real (bool, str, None are refused) finite number > 0: the surface operation has no dt = 0 identity (there is no
    vertical stage here to give one a meaning)."""
    dt = _real(dt_s, name)
    if not (math.isfinite(dt) and dt > 0.0):
        raise ExperimentalHydrologyError(f"{name} must be finite and > 0, got {dt!r}")
    return dt


def advance_time(t_s: float, dt_s: float) -> float:
    """`t + dt` for a validated `t >= 0` and `dt > 0`; refuses an overflowing end time and a step that does not advance
    floating time (`t + dt == t`)."""
    t_new = t_s + dt_s
    if not math.isfinite(t_new):
        raise ExperimentalHydrologyError(f"state.t_s + dt_s overflows FP64 (t = {t_s}, dt = {dt_s})")
    if not t_new > t_s:
        raise ExperimentalHydrologyError(f"dt_s = {dt_s} does not advance floating time at t = {t_s}")
    return t_new


# --- state and result ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SurfaceState:
    """Surface water of one accepted instant: `t_s` (s, the accepted START time of the next step), the surface depth `depth_m`
    `(ny, nx)` float64 AFTER the caller's vertical exchange, and for the local-inertial method the signed face fluxes
    `qx_m2_s (ny, nx+1)` (positive east) and `qy_m2_s (ny+1, nx)` (positive north). The explicit method has no momentum: both
    must be None. Arrays are the router's namespace (host NumPy or CuPy on the router's device). No soil water, no
    continuation metadata: the router never reads anything else."""

    t_s: float
    depth_m: Any
    qx_m2_s: Any = None
    qy_m2_s: Any = None


@dataclass(frozen=True, eq=False)
class SurfaceStep:
    """One accepted surface step. Arrays in the router's namespace (fresh), scalars host numbers on the CPU / 0-d device views
    of the fresh packet on CUDA. The fields mirror the historical `HydraulicStep` minus its bucket (`column`, soil water):

    state              `SurfaceState` at the physical END time `t_s + dt_s` (routed depth; local inertia: the new face fluxes)
    face_volume_m3     explicit {"out": (ny, nx)} volume that left each cell this step (>= 0); local inertia {"x", "y"} SIGNED
                       volumes dt dx q' that crossed each face
    used_flux_m2_s     the unit fluxes those volumes were made of (explicit k h_c^(3/2); local inertia the new face state)
    velocity_m_s       instantaneous END-of-step cell speed (explicit q(h_new)/h_new; local inertia face-averaged flux / h_new;
                       0 when dry). RECONSTRUCTED, unbounded near drying cells, NOT qualified for erosion (see
                       `experimental_hydrology.HydraulicStep`)
    face_flow_depth_m  the stage depth each flux was computed from: explicit {"out": the input depth itself, no copy};
                       local inertia {"x", "y"} face depths
    export_m3          volume through the open outlet faces during the step (face volumes, not instantaneous q)
    outlet_discharge_m3_s  instantaneous end-of-step outlet discharge
    storage_change_m3 / budget_residual_m3  global balance operands (storage change + export; checked against BALANCE_RTOL)
    max_cfl / cfl_kind  the CFL quantity (admissible iff <= cfl_max) and its name
    max_cell_balance_residual_m, limited_cells, limited_volume_m3  per-cell balance and donor-limiter diagnostics
    """

    method: str
    implementation: str
    dt_s: float
    state: SurfaceState
    velocity_m_s: Any
    face_volume_m3: dict
    used_flux_m2_s: dict
    export_m3: Any
    outlet_discharge_m3_s: Any
    storage_change_m3: Any
    budget_residual_m3: Any
    max_cfl: Any
    max_cell_balance_residual_m: Any
    limited_cells: Any
    limited_volume_m3: Any
    cfl_kind: str
    face_flow_depth_m: Any = None


# --- input contracts (xp-generic: NumPy or CuPy arrays) -------------------------------------------------------------
def _exact_array(xp: ModuleType, name: str, array: Any, expected: tuple[int, ...]) -> None:
    if type(array) is not xp.ndarray:
        raise ExperimentalHydrologyError(f"{name} must be an exact {xp.__name__}.ndarray (no subclass, masked array, list or "
                                         f"other namespace; nothing is converted or transferred), got {type(array).__name__}")
    if tuple(array.shape) != tuple(expected) or array.dtype != np.float64:
        raise ExperimentalHydrologyError(f"{name} must be float64 with shape {tuple(expected)}, got {array.dtype} "
                                         f"{tuple(array.shape)}")


def check_surface_structure(xp: ModuleType, state: Any, *, method: str, shape: tuple[int, int]) -> float:
    """Type, dtype and shape contract of a `SurfaceState` in namespace `xp` (no value is read): returns the validated time.
    Raises `ExperimentalHydrologyError`."""
    if method not in SURFACE_METHODS:
        raise ExperimentalHydrologyError(f"method must be one of {SURFACE_METHODS}, got {method!r}")
    if not isinstance(state, SurfaceState):
        raise ExperimentalHydrologyError(f"state must be a SurfaceState, got {type(state).__name__}")
    ny, nx = (int(v) for v in shape)
    t = check_state_time(state.t_s, "state.t_s")
    _exact_array(xp, "state.depth_m", state.depth_m, (ny, nx))
    if method == "local_inertial":
        _exact_array(xp, "state.qx_m2_s", state.qx_m2_s, (ny, nx + 1))
        _exact_array(xp, "state.qy_m2_s", state.qy_m2_s, (ny + 1, nx))
    elif state.qx_m2_s is not None or state.qy_m2_s is not None:
        raise ExperimentalHydrologyError("the explicit method has no momentum: qx_m2_s and qy_m2_s must be None")
    return t


def validate_surface_state(xp: ModuleType, state: Any, *, method: str, shape: tuple[int, int],
                           faces: tuple | None = None) -> None:
    """Structure AND values of a `SurfaceState`: finite depth >= 0 everywhere (inactive cells included); for local inertia
    finite face fluxes, zero on closed faces, never into the domain on an open face. `faces` = `(fx_type, fx_sign, fy_type,
    fy_sign)` in `xp`. One batched flag read on a device. Raises `ExperimentalHydrologyError`."""
    from maple.core.backend import DeferredChecks, finite_flag, negative_flag, true_flag

    check_surface_structure(xp, state, method=method, shape=shape)
    checks = DeferredChecks()
    checks.require(finite_flag(state.depth_m), "state.depth_m must be finite everywhere (inactive cells included)")
    checks.forbid(negative_flag(state.depth_m), "state.depth_m must be >= 0 everywhere (inactive cells included)")
    if method == "local_inertial":
        if faces is None:
            raise ExperimentalHydrologyError("the local-inertial state validation needs the face tables")
        fx_type, fx_sign, fy_type, fy_sign = faces
        qx, qy = state.qx_m2_s, state.qy_m2_s
        checks.require(finite_flag(qx), "state.qx_m2_s must be finite everywhere")
        checks.require(finite_flag(qy), "state.qy_m2_s must be finite everywhere")
        checks.forbid(true_flag((fx_type == 0) & (qx != 0.0)), "state.qx_m2_s must be 0 on closed faces")
        checks.forbid(true_flag((fy_type == 0) & (qy != 0.0)), "state.qy_m2_s must be 0 on closed faces")
        checks.forbid(true_flag((fx_type == 2) & (qx * fx_sign < 0.0)), "state.qx_m2_s points into the domain on an open face")
        checks.forbid(true_flag((fy_type == 2) & (qy * fy_sign < 0.0)), "state.qy_m2_s points into the domain on an open face")
    try:
        checks.resolve()
    except ValueError as exc:
        raise ExperimentalHydrologyError(str(exc)) from None


# --- memory estimate and admission -----------------------------------------------------------------------------------
MEMORY_SCOPE = (
    "Declared upper bound, per backend and method, of (a) the static arrays a surface router OWNS after preparation, (b) the "
    "transient scratch of preparation itself (donor-table build, contiguous staging copies), (c) one `initial_state` (depth "
    "copy, zero face arrays) and (d) the arrays ONE step allocates: its returned outputs, its transient temporaries/operand "
    "rows, a fixed Python-object allowance and the lateral packet (CUDA: 128 B device + 128 B host copy). Admission on the "
    "admitted side (host for numpy/numba, device for cuda) against `available_bytes_fn()`: `preparation_required_bytes` "
    "BEFORE preparation allocates anything, `initialization_required_bytes` before every `initial_state` copies, "
    "`required_bytes` (static + step + reserve) before every step allocates. INCLUDED for information but NOT admitted (caller-owned, already resident): the input state arrays "
    "(`input_bytes`), the caller's RoutingGraph/LocalInertialGeometry (`geometry_bytes`, referenced by the NumPy router, "
    "copied by the compiled/device routers into `static_bytes`). EXCLUDED: Numba JIT compilation memory and dispatcher "
    "caches, NVRTC/CUDA module and context memory, CuPy memory-pool granularity/fragmentation and pool limits, Python object "
    "overhead, the transient host copies made while a CUDA context validates the graph (`preparation_host_bytes`, declared "
    "separately), any composed vertical stage, sediment stage or driver arrays, and whole-process or allocator-pool "
    "guarantees: a reading is not a reservation."
)

# NumPy reference, explicit (`experimental_hydrology.explicit_surface_core`), per cell: outputs q_used, out_vol, h_new,
# velocity; temporaries sqrt_h, cfl_cell, the donor gather index (4 int64 = 4 doubles), total/inflow, q_inst, balance, scale
# and up to three simultaneously live operand chains, plus two bool masks (1 B each).
NUMPY_EXPLICIT_OUTPUT_DOUBLES_PER_CELL = 4
NUMPY_EXPLICIT_SCRATCH_DOUBLES_PER_CELL = 13
NUMPY_EXPLICIT_SCRATCH_BYTES_PER_CELL = 2
# NumPy reference, local inertia (`experimental_hydrology.local_inertial_surface_core`): outputs per face qx/qy, hfx/hfy, the
# signed face volumes (3 doubles per face) and h_new, speed per cell (2 doubles). Temporaries: NumPy frees each intermediate
# as soon as it is unreferenced, so the peak is the largest PHASE, not the sum: the x-face update holds the four zero-padded
# cell/eta operands plus ~6 live intermediates of ONE axis (~10 doubles x the x faces, i.e. ~5 per face of both axes), the
# limiter/continuity phase holds the retained face outputs plus phi/scaled copies and the cell chains. Declared bound: 12
# doubles per face of BOTH axes and 12 per cell (eta, div, safe, ux, uy, speed, balance, scale, the limiter's out/out_v/avail/
# phi and the live operand chains) plus bool masks; measured by tracemalloc at 180x200 (test_surface_memory): 0.44-0.57 of it.
NUMPY_LOCAL_OUTPUT_DOUBLES_PER_FACE = 3
NUMPY_LOCAL_OUTPUT_DOUBLES_PER_CELL = 2
NUMPY_LOCAL_SCRATCH_DOUBLES_PER_FACE = 12
NUMPY_LOCAL_SCRATCH_DOUBLES_PER_CELL = 12
NUMPY_LOCAL_SCRATCH_BYTES_PER_FACE = 3
NUMPY_LOCAL_SCRATCH_BYTES_PER_CELL = 2
# Numba (`experimental_numba.NumbaSurfaceRouter`): explicit outputs q_used, out_vol, h_new, velocity (4) + scratch rows (4) +
# one conditional contiguous copy of the depth (1); local: per face qx, hf, face volume outputs (3) + outward operand row and
# conditional copy (2); per cell h_new, speed (2) + scratch rows (4) + conditional depth copy (1). Static: explicit active
# (1 B) + outlet (1 B) + k (8) + donors (32) per cell; local active (1 B) + z (8) per cell and type (1) + zmax (8) + fric (8)
# + kb (8) + sign (1) per face.
NUMBA_EXPLICIT_OUTPUT_DOUBLES_PER_CELL = 4
NUMBA_EXPLICIT_SCRATCH_DOUBLES_PER_CELL = 5
NUMBA_LOCAL_OUTPUT_DOUBLES_PER_FACE = 3
NUMBA_LOCAL_SCRATCH_DOUBLES_PER_FACE = 2
NUMBA_LOCAL_OUTPUT_DOUBLES_PER_CELL = 2
NUMBA_LOCAL_SCRATCH_DOUBLES_PER_CELL = 5
# CUDA (`experimental_cuda.CudaSurfaceRouter`): explicit device rows f_out, q_used, h_new, vel (outputs 4) + cfl_row, q_inst,
# bal_row, sc_row (scratch 4); local per face q_new, hf (outputs 2) + face volume (1) + fflag (1 B scratch); per cell h_new,
# speed (2) + bal_row, sc_row, phi, outv_row (4); plus the 128 B packet on the device and its host copy.
CUDA_EXPLICIT_OUTPUT_DOUBLES_PER_CELL = 4
CUDA_EXPLICIT_SCRATCH_DOUBLES_PER_CELL = 4
CUDA_LOCAL_OUTPUT_DOUBLES_PER_FACE = 3
CUDA_LOCAL_SCRATCH_BYTES_PER_FACE = 1
CUDA_LOCAL_OUTPUT_DOUBLES_PER_CELL = 2
CUDA_LOCAL_SCRATCH_DOUBLES_PER_CELL = 4
# Python-level objects of one step (the SurfaceStep/SurfaceState records, dicts, array headers/views, scalar boxes): a fixed
# allowance measured on the host by tracemalloc (a few KB), counted in every backend's scratch so the declared per-step
# allocation is a bound of EVERYTHING one step allocates on the host-visible side.
STEP_OBJECT_BYTES = 16 * 1024
# Host donor-table build (`donor_cells_by_cell`): row/col/own index arrays and the per-slot neighbour/mask temporaries, at
# most ~8 int64/bool arrays of n live at once; declared as 10 doubles per cell of transient preparation scratch.
PREPARATION_DONOR_SCRATCH_DOUBLES_PER_CELL = 10
# Geometry-vs-graph validation of the local-inertial routers (`np.array_equal` on the active mask, then on the friction):
# one transient N-cell boolean mask at a time; declared as 2 bytes per cell of preparation scratch (both masks).
VALIDATION_SCRATCH_BYTES_PER_CELL = 2
STATIC_EXPLICIT_BYTES_PER_CELL = 1 + 1 + 8 + 32  # owned copies: active, outlet, conveyance, donors (compiled/device routers)
STATIC_LOCAL_BYTES_PER_CELL = 1 + 8  # active, z
STATIC_LOCAL_BYTES_PER_FACE = 1 + 8 + 8 + 8 + 1  # type, zmax, fric, kb, sign
GEOMETRY_BYTES_PER_CELL = 8 + 1 + 8  # LocalInertialGeometry z, active, friction (caller-owned host arrays)


@dataclass(frozen=True)
class SurfaceMemory:
    """Backend/method-specific upper bound of the bytes a surface router owns and one step allocates (see MEMORY_SCOPE)."""

    method: str
    backend: str
    shape: tuple[int, int]
    n_cells: int
    n_faces: int  # 0 for the explicit method; ny (nx+1) + (ny+1) nx for local inertia
    limiter: str
    input_bytes: int  # caller's state arrays (not admitted)
    geometry_bytes: int  # caller's LocalInertialGeometry arrays referenced by the NumPy router (not admitted; 0 otherwise)
    static_bytes: int  # router-owned static arrays on the admitted side
    output_bytes: int  # fresh outputs of one step
    scratch_bytes: int  # transient temporaries / operand rows / packet of one step
    preparation_host_bytes: int  # transient host copies while a CUDA context validates and uploads (not admitted)
    reserve_bytes: int  # caller-specified
    host_bytes: int
    device_bytes: int
    step_allocation_bytes: int  # output_bytes + scratch_bytes
    required_bytes: int  # static + step allocation + reserve on the admitted side (admitted before every step)
    total_bytes: int  # host + device + reserve (inputs and geometry included)
    initialization_bytes: int  # one `initial_state`: the depth copy and the zero face arrays, on the admitted side
    preparation_scratch_bytes: int  # transient allocations of preparation on the admitted side (donor-table build, copies)
    initialization_required_bytes: int  # static + initialization + reserve (admitted before every `initial_state`)
    preparation_required_bytes: int  # static + max(preparation scratch, step allocation, initialization) + reserve
    scope: str = MEMORY_SCOPE

    def as_dict(self) -> dict[str, Any]:
        return {"method": self.method, "backend": self.backend, "shape": list(self.shape), "n_cells": self.n_cells,
                "n_faces": self.n_faces, "limiter": self.limiter, "input_bytes": self.input_bytes,
                "geometry_bytes": self.geometry_bytes, "static_bytes": self.static_bytes, "output_bytes": self.output_bytes,
                "scratch_bytes": self.scratch_bytes, "preparation_host_bytes": self.preparation_host_bytes,
                "reserve_bytes": self.reserve_bytes, "host_bytes": self.host_bytes, "device_bytes": self.device_bytes,
                "step_allocation_bytes": self.step_allocation_bytes, "required_bytes": self.required_bytes,
                "total_bytes": self.total_bytes, "initialization_bytes": self.initialization_bytes,
                "preparation_scratch_bytes": self.preparation_scratch_bytes,
                "initialization_required_bytes": self.initialization_required_bytes,
                "preparation_required_bytes": self.preparation_required_bytes, "scope": self.scope}


def _checked_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ExperimentalHydrologyError(f"{name} must be a non-Boolean integer, got {value!r}")
    if int(value) < 0:
        raise ExperimentalHydrologyError(f"{name} must be >= 0, got {value!r}")
    return int(value)


def estimate_surface_memory(method: str, shape: Any, *, backend: str, limiter: str = "off",
                            reserve_bytes: int = 0) -> SurfaceMemory:
    """Pure arithmetic (see MEMORY_SCOPE and the per-backend constants above). `limiter` only matters for the local-inertial
    method; the explicit method refuses "donor" as the routers do."""
    if method not in SURFACE_METHODS:
        raise ExperimentalHydrologyError(f"method must be one of {SURFACE_METHODS}, got {method!r}")
    if backend not in SURFACE_BACKENDS:
        raise ExperimentalHydrologyError(f"backend must be one of {SURFACE_BACKENDS}, got {backend!r}")
    if limiter not in ("off", "donor"):
        raise ExperimentalHydrologyError(f"limiter must be 'off' or 'donor', got {limiter!r}")
    if method == "explicit" and limiter != "off":
        raise ExperimentalHydrologyError("the donor limiter belongs to the local-inertial method")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ExperimentalHydrologyError(f"shape must be (ny, nx), got {shape!r}")
    ny, nx = (_checked_int(v, "shape entry") for v in shape)
    if ny < 1 or nx < 1:
        raise ExperimentalHydrologyError(f"shape must be positive, got {shape!r}")
    reserve = _checked_int(reserve_bytes, "reserve_bytes")
    n = ny * nx
    n_faces = ny * (nx + 1) + (ny + 1) * nx if method == "local_inertial" else 0
    inputs = n * _F8 + n_faces * _F8
    geometry = GEOMETRY_BYTES_PER_CELL * n + STATIC_LOCAL_BYTES_PER_FACE * n_faces if method == "local_inertial" else 0
    prep_host = 0
    if method == "explicit":
        static_owned = STATIC_EXPLICIT_BYTES_PER_CELL * n
        if backend == "numpy":
            static = 4 * 8 * n  # the host donor table (4, n) int64; active/outlet/k are views of the caller's graph
            outputs = NUMPY_EXPLICIT_OUTPUT_DOUBLES_PER_CELL * n * _F8
            scratch = NUMPY_EXPLICIT_SCRATCH_DOUBLES_PER_CELL * n * _F8 + NUMPY_EXPLICIT_SCRATCH_BYTES_PER_CELL * n
        elif backend == "numba":
            static = static_owned
            outputs = NUMBA_EXPLICIT_OUTPUT_DOUBLES_PER_CELL * n * _F8
            scratch = NUMBA_EXPLICIT_SCRATCH_DOUBLES_PER_CELL * n * _F8 + 2 * _F8
        else:
            static = static_owned
            outputs = CUDA_EXPLICIT_OUTPUT_DOUBLES_PER_CELL * n * _F8
            scratch = CUDA_EXPLICIT_SCRATCH_DOUBLES_PER_CELL * n * _F8 + SURFACE_PACKET_BYTES
    else:
        static_owned = STATIC_LOCAL_BYTES_PER_CELL * n + STATIC_LOCAL_BYTES_PER_FACE * n_faces
        if backend == "numpy":
            static = 0  # the router references the caller's geometry and graph arrays (no owned copy)
            outputs = (NUMPY_LOCAL_OUTPUT_DOUBLES_PER_FACE * n_faces + NUMPY_LOCAL_OUTPUT_DOUBLES_PER_CELL * n) * _F8
            scratch = ((NUMPY_LOCAL_SCRATCH_DOUBLES_PER_FACE * n_faces + NUMPY_LOCAL_SCRATCH_DOUBLES_PER_CELL * n) * _F8
                       + NUMPY_LOCAL_SCRATCH_BYTES_PER_FACE * n_faces + NUMPY_LOCAL_SCRATCH_BYTES_PER_CELL * n)
        elif backend == "numba":
            static = static_owned
            outputs = (NUMBA_LOCAL_OUTPUT_DOUBLES_PER_FACE * n_faces + NUMBA_LOCAL_OUTPUT_DOUBLES_PER_CELL * n) * _F8
            scratch = (NUMBA_LOCAL_SCRATCH_DOUBLES_PER_FACE * n_faces + NUMBA_LOCAL_SCRATCH_DOUBLES_PER_CELL * n) * _F8 + 4 * _F8
        else:
            static = static_owned
            outputs = (CUDA_LOCAL_OUTPUT_DOUBLES_PER_FACE * n_faces + CUDA_LOCAL_OUTPUT_DOUBLES_PER_CELL * n) * _F8
            scratch = (CUDA_LOCAL_SCRATCH_DOUBLES_PER_CELL * n * _F8 + CUDA_LOCAL_SCRATCH_BYTES_PER_FACE * n_faces
                       + SURFACE_PACKET_BYTES)
    scratch += STEP_OBJECT_BYTES
    initialization = (n + n_faces) * _F8  # the depth copy and (local inertia) the two zero face arrays of `initial_state`
    donor_build = PREPARATION_DONOR_SCRATCH_DOUBLES_PER_CELL * n * _F8 if method == "explicit" else 0
    validation = VALIDATION_SCRATCH_BYTES_PER_CELL * n if method == "local_inertial" else 0  # host comparison masks
    if backend == "numpy":
        prep_scratch = donor_build + validation  # the host donor table is built directly; the geometry is the caller's
    elif backend == "numba":
        prep_scratch = static_owned + donor_build + validation  # contiguous staging copies before the owned copies, plus the build
    else:
        prep_scratch = 0  # device side: the owned uploads ARE the static bytes; host staging is `preparation_host_bytes`
    if backend == "cuda":
        # transient host copies at preparation: the downloaded graph runtime arrays (conveyance, active, outlet; level order,
        # conveyance_lo, donor tables for the active cells) and the host staging of every owned array before upload
        prep_host = (8 + 1 + 1) * n + (8 + 8 + 32 + 4) * n + static_owned + donor_build + validation + SURFACE_PACKET_BYTES
        host = prep_host + SURFACE_PACKET_BYTES + STEP_OBJECT_BYTES
        device = inputs + static + outputs + scratch - STEP_OBJECT_BYTES
    else:
        host = inputs + geometry + static + outputs + scratch
        device = 0
    step_allocation = outputs + scratch
    required = static + step_allocation + reserve
    return SurfaceMemory(
        method=method, backend=backend, shape=(ny, nx), n_cells=n, n_faces=n_faces, limiter=limiter, input_bytes=inputs,
        geometry_bytes=geometry, static_bytes=static, output_bytes=outputs, scratch_bytes=scratch,
        preparation_host_bytes=prep_host, reserve_bytes=reserve, host_bytes=host, device_bytes=device,
        step_allocation_bytes=step_allocation, required_bytes=required, total_bytes=host + device + reserve,
        initialization_bytes=initialization, preparation_scratch_bytes=prep_scratch,
        initialization_required_bytes=static + initialization + reserve,
        preparation_required_bytes=static + max(prep_scratch, step_allocation, initialization) + reserve)


def default_host_available_bytes() -> int:
    """Available physical host memory reported by the OS (`sysconf`); a platform without it is a setup error (inject
    `available_bytes_fn`). A reading, not a reservation."""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError) as exc:
        raise ExperimentalHydrologyError(
            f"the host cannot report available memory ({type(exc).__name__}: {exc}); pass available_bytes_fn") from exc
    if pages < 0 or page <= 0:
        raise ExperimentalHydrologyError("the host reported an invalid available-memory figure; pass available_bytes_fn")
    return int(pages) * int(page)


def effective_free_device_bytes(driver_free_bytes: int, pool_free_bytes: int, pool_limit_bytes: int,
                                pool_used_bytes: int) -> int:
    """`driver_free + pool_free`, limited to `pool_limit - pool_used` when the pool has a positive limit; never negative. The
    same four-number arithmetic as `storage_admission.effective_free_device_bytes` and `subsurface_cuda`, kept local so this
    module imports neither the bed-storage layer nor the soil column."""
    for name, value in (("driver_free_bytes", driver_free_bytes), ("pool_free_bytes", pool_free_bytes),
                        ("pool_limit_bytes", pool_limit_bytes), ("pool_used_bytes", pool_used_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExperimentalHydrologyError(f"{name} must be a nonnegative int, got {value!r}")
    reachable = driver_free_bytes + pool_free_bytes
    if pool_limit_bytes > 0:
        reachable = min(reachable, pool_limit_bytes - pool_used_bytes)
    return max(0, reachable)


def default_device_available_bytes_fn(cp: Any, device_id: int) -> Callable[[], int]:
    """Default admission reading for a device router: driver free memory plus the default pool's free bytes under the pool
    limit if one is set. Refuses (before reading) when the current device is not the router's device; never changes the
    device or any limit."""

    def query() -> int:
        try:
            current = int(cp.cuda.Device().id)
        except Exception as exc:  # reported, never guessed
            raise ExperimentalHydrologyError(f"CUDA device query failed ({type(exc).__name__}: {exc})") from exc
        if current != device_id:
            raise ExperimentalHydrologyError(f"the surface router is bound to device {device_id} but the current device is "
                                             f"{current}; select the router's device or inject available_bytes_fn")
        pool = cp.get_default_memory_pool()
        return effective_free_device_bytes(int(cp.cuda.Device().mem_info[0]), int(pool.free_bytes()),
                                           int(pool.get_limit()), int(pool.used_bytes()))

    return query


def UNLIMITED_ADMISSION() -> int:  # noqa: N802 - a named policy, used as a function value
    """The HISTORICAL policy: the pre-existing public solvers never had memory admission and none is retrofitted on them; they
    pass this reading (`sys.maxsize`) so every estimate is admitted. New callers should pass a real query (the default)."""
    return sys.maxsize


def admit_bytes(estimate: SurfaceMemory, available_bytes_fn: Callable[[], int], what: str, need: int | None = None) -> int:
    """Compare `need` (default `estimate.required_bytes`) with the reported available bytes and raise `SurfaceMemoryError`
    BEFORE anything is allocated. Returns the reading."""
    available = available_bytes_fn()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise ExperimentalHydrologyError(f"available_bytes_fn must return a non-Boolean integer >= 0, got {available!r}")
    available = int(available)
    need = estimate.required_bytes if need is None else int(need)
    if need > available:
        raise SurfaceMemoryError(
            f"surface {what} ({estimate.method}, {estimate.backend}) on {estimate.shape[0]}x{estimate.shape[1]} cells needs an "
            f"estimated {need} B on its admitted side (reserve {estimate.reserve_bytes} B included) but only {available} B are "
            f"reported available; nothing was allocated. Scope: {MEMORY_SCOPE}")
    return available


# --- selector --------------------------------------------------------------------------------------------------------
def build_surface_router(method: str, graph: Any, *, backend: str = "numpy", geometry: Any = None, control: Any = None,
                         reserve_bytes: int = 0, available_bytes_fn: Callable[[], int] | None = None, context: Any = None):
    """The surface router of `backend` ("numpy" | "numba" | "cuda") for `method` on `graph` (a `RoutingGraph` in the backend's
    namespace) with the local-inertial `geometry` (`LocalInertialGeometry`, host) and `control` (`HydraulicControl`). Imports
    the backend module lazily; a missing Numba/CuPy raises that module's explicit error, never a fallback. `context` (CUDA only)
    is an already prepared `CudaSurfaceContext` bound to exactly this graph/geometry."""
    if backend not in SURFACE_BACKENDS:
        raise ExperimentalHydrologyError(f"backend must be one of {SURFACE_BACKENDS}, got {backend!r}")
    if backend == "numpy":
        if context is not None:
            raise ExperimentalHydrologyError("only the cuda backend takes a prepared context")
        from maple_syrup.experimental_hydrology import CpuSurfaceRouter

        return CpuSurfaceRouter(method, graph, geometry=geometry, control=control, reserve_bytes=reserve_bytes,
                                available_bytes_fn=available_bytes_fn)
    if backend == "numba":
        if context is not None:
            raise ExperimentalHydrologyError("only the cuda backend takes a prepared context")
        from maple_syrup.experimental_numba import NumbaSurfaceRouter

        return NumbaSurfaceRouter(method, graph, geometry=geometry, control=control, reserve_bytes=reserve_bytes,
                                 available_bytes_fn=available_bytes_fn)
    from maple_syrup.experimental_cuda import CudaSurfaceRouter

    return CudaSurfaceRouter(method, graph, geometry=geometry, control=control, context=context, reserve_bytes=reserve_bytes,
                             available_bytes_fn=available_bytes_fn)
