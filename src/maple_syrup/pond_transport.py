"""Multi-face characteristic mobile-sediment transport for the pond (local-inertial) backend.

Pure operator, no bed access. It generalises the single-receiver `characteristic_transport` operator to cells
that can send mass through up to four D4 faces whose water-flux sign may change from step to step. It is the
lateral operator `T` of the later pond event wrapper:

    native pickup cap -> THIS OPERATOR on the ACTUAL pickup -> native deposit/export -> phase reconciliation

The native MAPLE active/voxel bed stays the only inventory. This module neither mutates nor creates mass: it
returns requests (`Dep`, `E`) and the pool `T = remaining + Dep + E` that MAPLE validates, exactly as
`sediment_transport.TransportStep` documents. Nothing here was run by its author (file-only tools).

Network (`FaceNetwork`)
-----------------------
Static, validated, frozen, built once from the host active mask plus either explicit exterior open-face
booleans (`face_network`), typed geometry face arrays (`face_network_from_geometry`, codes of
`LocalInertialGeometry`: 0 closed, 1 interior, 2 exterior outlet) or a `PondTopology`-like object
(`face_network_from_pond_topology`, thin duck-typed adapter reading `.active`, `.dx_m`, `.geometry.fx_type/
fy_type/fx_sign/fy_sign`; no import from the provisional reference tree). Faces follow MAPLE's layout: x faces
`(ny, nx + 1)` positive EAST, y faces `(ny + 1, nx)` positive NORTH, row 0 = south. Rules:

- interior faces are open only between two ACTIVE cells; nothing crosses an inactive cell; interior holes
  (inactive cells) stay closed; a typed interior face of type 0 between active cells is a wall;
- export leaves only through a designated outlet (an exterior face of an active cell); no implied open sides;
- periodic boundaries and external sediment inflow are explicitly refused (`periodic=True`,
  `external_inflow=True`, `external_inflow_kg` at the step) -- they are outside the initial scope.

Per cell the four directions are `0 = east (+col), 1 = north (+row), 2 = west, 3 = south`.

Water versus sediment
---------------------
Each accepted physical step supplies the SIGNED ACTUAL water volumes through every face (`water_volume_x_m3`
`(ny, nx+1)`, `water_volume_y_m3` `(ny+1, nx)`, m3, + east/north). Only their sign and ratio are used, as a
splitting rule: the outward volume of a cell through face d is `max(sign_d * volume, 0)`; the cell's outflow
weights are those volumes normalised to sum 1 (scaled by the largest first, so a 1e-300 or 1e300 magnitude
cannot overflow; a weight that underflows to 0 is a genuine ratio below ~1e-308, the mass goes to the other
faces). The sediment velocity `v` [m/s] and the exponential inverse distance `r = 1/L` [1/m] are separate
per-cell/class inputs: virtual sediment velocity is NOT the water velocity. Non-finite volumes, nonzero volume
on a closed face and water flowing INTO the domain through an outlet are refused before anything is computed.

Transport law (same as `characteristic_transport`, per substep `dt`, `Courant v dt / dx <= 0.5`)
------------------------------------------------------------------------------------------
Mobile mass is a set of packets at distance `x in [0, dx)` from the cell's upstream face. ACTUAL pickup is a new
packet at `x = 0` in the first substep. A packet moves at its class `v`; over a travelled distance `s` it keeps
`exp(-r s)` and deposits `m (-expm1(-r s))` in the cell traversed. It reaches the downstream face after
`tau = (dx - x) / v`; with the Courant limit it crosses AT MOST ONE face per substep. At the crossing the
survivor is split over the cell's current positive-outflow faces by water-volume weight (`survivor * w_d`); the
part for an outlet is an export request of the source cell (held at the actual export edge cell), the part for
an interior face arrives in that neighbour, advances for the remaining `dt - tau` at the RECEIVER's `v` and `r`,
may deposit on the way, may settle completely if the receiver's `settle_mask` says so, and otherwise starts a
new packet at `x = v_receiver (dt - tau)`. Eventual survival of mass to a face is `exp(-dx / L)` whatever
`dx / L` is; the wrongly corrected well-mixed `L / (L + dx)` transmission is not used. With exactly one
outflow face of weight 1 per cell the operator performs the same arithmetic as `characteristic_step` (tested).

Choices to be aware of (approximations, not hidden clips)
---------------------------------------------------------
- FLOW-BLOCKING: a cell with no positive outflow face for the step has no advection progress: its effective
  `v` is 0 (no travel, hence no hazard deposition `v r`), its packets and phase are preserved exactly, a pickup
  stays at `x = 0`, and mass arriving there stays at `x = 0`. It deposits only through an explicit
  `settle_mask`; it NEVER exports and nothing is deleted. The alternative (moving mass against the water or
  along the terrain) would invent transport the actual water does not do. The velocity of such a cell is
  therefore not Courant-checked; every cell WITH outflow is checked and violations raise
  `TransportStepRejected`, never a clip. `blocked_cell` reports which cells were treated this way.
- SCALAR PROGRESS: `x` is one scalar distance from "the" upstream face. Mass that enters through one face and
  leaves through another (turning flow, e.g. entering from the east and leaving north) keeps its scalar
  progress `v_r (dt - tau)` rather than a geometric path; the distance to the face it leaves through is always
  `dx - x`. Each cell has the same face-to-face length `dx`.
- VOLUME WEIGHTING: the split uses the step's water volumes for the whole step (weights frozen across the
  substeps), not instantaneous per-substep flux, and not the sediment concentration. A flux that reverses
  within the step is represented only by its net volume sign.
- SUBSTEP ORDER: weights are fixed per call; changing flow directions between calls need no state beyond the
  phase positions (scalar `x` carries over).

Merging (unchanged from `characteristic_transport`)
---------------------------------------------------
Candidates landing in one cell/class/bin are merged to one hazard-preserving logarithmic mean
`x = L log(sum m exp(x/L) / sum m)` (`expm1`/`log1p` shifted by the largest position when `r (xmax - xmin) <=
NARROW_SPREAD`, else a log-sum-exp shifted by the largest `log m + r x`; mass-weighted mean for `r = 0`).
Only round-off containment inside the constituents' range is applied; no mass is adjusted.

Bookkeeping contracts
---------------------
Returns the unchanged `TransportStep` (`T = remaining + Dep + E`, MAPLE face layout including export
crossings, gross and net) plus `phase_after` on the REMAINING pool. Two distinct accountings:

- PRE-NATIVE POOL: `T` still contains the export request `E` until native MAPLE removes it, so per cell
  `T - (M + P) = In_internal - Out_internal` (`internal_transfer_in/out`, `divergence_kg`, which exclude
  export). Total class `sum T = sum (M + P)` before any bed request.
- FULL FACE CROSSINGS: the x/y face arrays include the export crossings. Their net divergence
  `D = (x_net[c+1] - x_net[c]) + (y_net[r+1] - y_net[r]) = Out_internal + E - In_internal`, hence
  `D = (M + P) - (T - E)`, NOT `(M + P) - T`; after native removal of `Dep` and `E`,
  `remaining = (M + P) - D - Dep`. At a cell without export `D` equals `(M + P) - T`.

Residuals are FP64 round-off judged by MAPLE's
`summation_error_bound_kg` with the declared `operation_count`; they are reported, never clipped or hidden, and
the tolerance is not widened for mass.

Backends: `implementation="array"` is namespace generic (NumPy or CuPy): loops only over the 4 directions and
the substeps, no Python loop over cells, no host/device transfer per step. Memory is `n nc 5 (B + 1)` slots
times a few arrays. The Numba/CUDA kernels of the single-receiver operator are NOT provided here; asking for
them raises an explicit `TransportError`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from maple.surface.voxels._numerics import summation_error_bound_kg

from maple_syrup.characteristic_transport import (
    _BELOW_ANY_LOG_WEIGHT,
    _BELOW_ANY_POSITION,
    DEFAULT_COURANT_MAX,
    MAX_SUBSTEPS,
    NARROW_SPREAD,
    PhaseState,
    _phase_checks,
    _real,
    _require,
    _strict_int,
    bin_index,
    empty_phase_state,
    fraction_sum_tolerance,
)
from maple_syrup.sediment_transport import (
    TransportError,
    TransportStep,
    TransportStepRejected,
)

__all__ = [
    "DIRECTIONS",
    "IMPLEMENTATIONS",
    "FaceNetwork",
    "PondTransportStep",
    "empty_phase_state",
    "face_network",
    "face_network_from_geometry",
    "face_network_from_pond_topology",
    "operation_count",
    "outflow_weights",
    "pond_transport_step",
    "validate_phase",
]

_EPS = float(np.finfo(np.float64).eps)
IMPLEMENTATIONS = ("array",)
# (d_row, d_col) of directions 0..3: east, north, west, south.
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
_DIR_SIGN = (1.0, 1.0, -1.0, -1.0)  # sign of the outward normal in the x/y face convention
_DIR_IS_X = (True, False, True, False)
_COURANT_REJECTION = "pond sediment Courant number v dt / (n_substeps dx) exceeds"
_SCHEMA = b"maple_syrup.pond_transport.face_network.v1"


# --- network ------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class FaceNetwork:
    """Frozen multi-face D4 network, placed once in `xp`.

    `open_dir`, `outlet_dir` `(n, 4)` bool: the face of the cell in that direction is open (interior or outlet) /
    an exterior outlet; `neighbor` `(n, 4)` int64: flat index of the interior neighbour (the cell itself where
    the face is closed or an outlet); `face_row`, `face_col` `(n, 4)` int64: nominal index of the face in the
    x (east/west) or y (north/south) face array, always in range (closed faces carry exactly zero flux);
    `fx_type`, `fy_type` int8 and `fx_sign`, `fy_sign` float64 (outward sign on outlets, else 0) in face layout;
    `outlet_cell_flat` `(n,)`: the cell has an outlet; `active_flat` `(n,)`. Host copies and a digest bind it.
    """

    shape: tuple[int, int]
    dx_m: float
    n_cells: int
    n_active: int
    active_flat: Any
    open_dir: Any
    outlet_dir: Any
    neighbor: Any
    face_row: Any
    face_col: Any
    fx_type: Any
    fy_type: Any
    fx_sign: Any
    fy_sign: Any
    outlet_cell_flat: Any
    active_host: np.ndarray
    fx_type_host: np.ndarray
    fy_type_host: np.ndarray
    n_outlet_faces: int
    input_sha256: str
    xp: ModuleType


def _host(array: Any, name: str, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    if type(array) is not np.ndarray or array.dtype != np.dtype(dtype) or tuple(array.shape) != shape:
        got = f"{type(array).__name__} {getattr(array, 'dtype', '')} {getattr(array, 'shape', '')}"
        raise TransportError(f"{name} must be a host {np.dtype(dtype)} numpy array of shape {shape}, got {got}")
    return array


def _scope(periodic: Any, external_inflow: Any) -> None:
    if periodic is not False:
        raise TransportError("periodic boundaries are not supported in the initial pond transport scope")
    if external_inflow is not False:
        raise TransportError("external sediment inflow is not supported in the initial pond transport scope")


def _build(active: Any, fx_type: Any, fy_type: Any, dx_m: Any, fx_sign: Any, fy_sign: Any,
           xp: ModuleType | None) -> FaceNetwork:
    from maple.core.backend import freeze, to_device

    namespace = np if xp is None else xp
    if type(active) is not np.ndarray or active.ndim != 2 or active.dtype != np.bool_ or min(active.shape) < 1:
        raise TransportError("active must be a host bool (ny, nx) numpy array with ny, nx >= 1")
    ny, nx = active.shape
    if not active.any():
        raise TransportError("no active cell")
    dx = _real(dx_m, "dx_m")
    if not (math.isfinite(dx) and dx > 0.0):
        raise TransportError(f"dx_m must be finite and > 0, got {dx_m!r}")
    fx = _host(fx_type, "fx_type", np.int8, (ny, nx + 1))
    fy = _host(fy_type, "fy_type", np.int8, (ny + 1, nx))
    for name, f in (("fx_type", fx), ("fy_type", fy)):
        if not np.isin(f, (0, 1, 2)).all():
            raise TransportError(f"{name} codes must be 0 (closed), 1 (interior) or 2 (outlet)")

    # Face rules. Interior position = not on the outer ring of faces.
    both_x = np.zeros((ny, nx + 1), dtype=bool)
    both_x[:, 1:-1] = active[:, :-1] & active[:, 1:]
    adjacent_x = np.zeros((ny, nx + 1), dtype=bool)
    adjacent_x[:, 0], adjacent_x[:, nx] = active[:, 0], active[:, nx - 1]
    both_y = np.zeros((ny + 1, nx), dtype=bool)
    both_y[1:-1, :] = active[:-1, :] & active[1:, :]
    adjacent_y = np.zeros((ny + 1, nx), dtype=bool)
    adjacent_y[0, :], adjacent_y[ny, :] = active[0, :], active[ny - 1, :]
    for name, f, both, adjacent in (("x", fx, both_x, adjacent_x), ("y", fy, both_y, adjacent_y)):
        if np.any((f == 1) & ~both):
            raise TransportError(f"an interior {name} face must join two ACTIVE interior cells (no flux across "
                                 "inactive cells; interior holes are closed)")
        if np.any((f == 2) & ~adjacent):
            raise TransportError(f"an {name} outlet must be an EXTERIOR face of an ACTIVE cell")
    sign_x = np.zeros((ny, nx + 1), dtype=np.int8)
    sign_x[:, 0], sign_x[:, nx] = -1, 1
    sign_y = np.zeros((ny + 1, nx), dtype=np.int8)
    sign_y[0, :], sign_y[ny, :] = -1, 1
    sign_x, sign_y = np.where(fx == 2, sign_x, 0).astype(np.int8), np.where(fy == 2, sign_y, 0).astype(np.int8)
    for name, given, expected in (("fx_sign", fx_sign, sign_x), ("fy_sign", fy_sign, sign_y)):
        if given is not None and not (type(given) is np.ndarray and given.shape == expected.shape
                                      and np.array_equal(given, expected)):
            raise TransportError(f"{name} does not equal the outward sign of the designated outlet faces")

    n = ny * nx
    idx = np.arange(n, dtype=np.int64)
    r, c = np.divmod(idx, nx)
    ftype = np.stack([fx[r, c + 1], fy[r + 1, c], fx[r, c], fy[r, c]], axis=1)
    neighbor = np.stack([idx + 1, idx + nx, idx - 1, idx - nx], axis=1)
    neighbor = np.where(ftype == 1, neighbor, idx[:, None])
    face_row = np.stack([r, r + 1, r, r], axis=1)
    face_col = np.stack([c + 1, c, c, c], axis=1)
    flat_active = active.reshape(-1)
    if np.any((ftype != 0) & ~flat_active[:, None]):
        raise TransportError("internal error: an open face belongs to an inactive cell")
    digest = hashlib.sha256(_SCHEMA)
    for part in (active, fx, fy):
        digest.update(f"{part.dtype}{part.shape}".encode())
        digest.update(np.ascontiguousarray(part).tobytes())
    digest.update(repr(dx).encode())

    def runtime(a, dtype):
        return freeze(to_device(np.ascontiguousarray(np.asarray(a, dtype=dtype)), namespace))

    return FaceNetwork(
        shape=(ny, nx), dx_m=dx, n_cells=n, n_active=int(flat_active.sum()),
        active_flat=runtime(flat_active, np.bool_), open_dir=runtime(ftype != 0, np.bool_),
        outlet_dir=runtime(ftype == 2, np.bool_), neighbor=runtime(neighbor, np.int64),
        face_row=runtime(face_row, np.int64), face_col=runtime(face_col, np.int64),
        fx_type=runtime(fx, np.int8), fy_type=runtime(fy, np.int8),
        fx_sign=runtime(sign_x, np.float64), fy_sign=runtime(sign_y, np.float64),
        outlet_cell_flat=runtime((ftype == 2).any(axis=1), np.bool_),
        active_host=freeze(active.copy()), fx_type_host=freeze(fx.copy()), fy_type_host=freeze(fy.copy()),
        n_outlet_faces=int(np.count_nonzero(fx == 2) + np.count_nonzero(fy == 2)),
        input_sha256=digest.hexdigest(), xp=namespace)


def face_network(active: np.ndarray, dx_m: float, outlet_x: np.ndarray | None = None,
                 outlet_y: np.ndarray | None = None, *, periodic: bool = False, external_inflow: bool = False,
                 xp: ModuleType | None = None) -> FaceNetwork:
    """Network from the host `active (ny, nx)` mask and explicit exterior open-face booleans `outlet_x
    (ny, nx + 1)`, `outlet_y (ny + 1, nx)` (None = none). Interior faces between two active cells are open in
    both directions. A True entry anywhere but on the exterior ring of faces, or next to an inactive cell, is
    refused. No outlet at all is a legitimate closed domain."""
    _scope(periodic, external_inflow)
    if type(active) is not np.ndarray or active.ndim != 2 or active.dtype != np.bool_ or min(active.shape) < 1:
        raise TransportError("active must be a host bool (ny, nx) numpy array with ny, nx >= 1")
    ny, nx = active.shape
    ox = np.zeros((ny, nx + 1), dtype=bool) if outlet_x is None else _host(outlet_x, "outlet_x", np.bool_, (ny, nx + 1))
    oy = np.zeros((ny + 1, nx), dtype=bool) if outlet_y is None else _host(outlet_y, "outlet_y", np.bool_, (ny + 1, nx))
    fx = np.zeros((ny, nx + 1), dtype=np.int8)
    fx[:, 1:-1] = (active[:, :-1] & active[:, 1:])
    fy = np.zeros((ny + 1, nx), dtype=np.int8)
    fy[1:-1, :] = (active[:-1, :] & active[1:, :])
    fx[ox], fy[oy] = 2, 2
    return _build(active, fx, fy, dx_m, None, None, xp)


def face_network_from_geometry(active: np.ndarray, fx_type: np.ndarray, fy_type: np.ndarray, dx_m: float, *,
                               fx_sign: np.ndarray | None = None, fy_sign: np.ndarray | None = None,
                               periodic: bool = False, external_inflow: bool = False,
                               xp: ModuleType | None = None) -> FaceNetwork:
    """Network from typed face arrays (`LocalInertialGeometry` codes: 0 closed, 1 interior active-active,
    2 exterior outlet). Optional `fx_sign`/`fy_sign` are cross-checked against the outward direction."""
    _scope(periodic, external_inflow)
    return _build(active, fx_type, fy_type, dx_m, fx_sign, fy_sign, xp)


def face_network_from_pond_topology(topology: Any, *, xp: ModuleType | None = None) -> FaceNetwork:
    """Thin adapter for a `PondTopology`-like object (`.active`, `.dx_m`, `.geometry` with `fx_type`, `fy_type`,
    `fx_sign`, `fy_sign`). The geometry's open faces are the explicit outlets; its interior faces the static
    bidirectional topology. Nothing else of the topology (escape levels, elevations) is read or retained."""
    try:
        geometry = topology.geometry
        active, dx = topology.active, topology.dx_m
        parts = (geometry.fx_type, geometry.fy_type, geometry.fx_sign, geometry.fy_sign)
    except AttributeError as exc:
        raise TransportError(f"topology lacks the attributes the adapter reads: {exc}") from None
    if not np.array_equal(np.asarray(geometry.active), np.asarray(active)):
        raise TransportError("topology.active differs from topology.geometry.active")
    return _build(np.array(active, copy=True), np.array(parts[0], copy=True), np.array(parts[1], copy=True), dx,
                  np.array(parts[2], copy=True), np.array(parts[3], copy=True), xp)


# --- water volumes -> outflow weights ----------------------------------------------------------------
def _volume_checks(network: FaceNetwork, vx: Any, vy: Any, checks: Any) -> None:
    from maple.core.backend import finite_flag, true_flag

    if not isinstance(network, FaceNetwork):
        raise TransportError("network must be a FaceNetwork")
    xp = network.xp
    ny, nx = network.shape
    _require({"water_volume_x_m3": vx}, (ny, nx + 1), xp, np.float64)
    _require({"water_volume_y_m3": vy}, (ny + 1, nx), xp, np.float64)
    checks.require(finite_flag(vx), "water_volume_x_m3 must be finite everywhere")
    checks.require(finite_flag(vy), "water_volume_y_m3 must be finite everywhere")
    checks.forbid(true_flag((network.fx_type == 0) & (vx != 0.0)), "water_volume_x_m3 must be 0 on closed faces")
    checks.forbid(true_flag((network.fy_type == 0) & (vy != 0.0)), "water_volume_y_m3 must be 0 on closed faces")
    # Only actual outlet faces (sign +-1) are multiplied: a non-finite volume elsewhere is reported by the
    # finite check above and must not raise an inf * 0 warning here.
    checks.forbid(true_flag((network.fx_type == 2) & (xp.where(network.fx_sign != 0.0, vx, 0.0) * network.fx_sign < 0.0)),
                  "water flows INTO the domain through an x outlet face")
    checks.forbid(true_flag((network.fy_type == 2) & (xp.where(network.fy_sign != 0.0, vy, 0.0) * network.fy_sign < 0.0)),
                  "water flows INTO the domain through a y outlet face")


def _weights(network: FaceNetwork, vx: Any, vy: Any) -> tuple[Any, Any]:
    """`(weights (n, 4), has_outflow (n,))`: positive outward volumes normalised to sum 1 per cell (scaled by the
    largest first, so no overflow); cells without a positive outward volume have all weights 0."""
    xp = network.xp
    outward = xp.stack([(vx if _DIR_IS_X[d] else vy)[network.face_row[:, d], network.face_col[:, d]] * _DIR_SIGN[d]
                        for d in range(4)], axis=1)
    outward = xp.where(outward > 0.0, outward, 0.0)
    peak = outward.max(axis=1)
    has = peak > 0.0
    scaled = outward / xp.where(has, peak, 1.0)[:, None]
    total = scaled.sum(axis=1)
    return scaled / xp.where(has, total, 1.0)[:, None], has


def outflow_weights(network: FaceNetwork, water_volume_x_m3: Any, water_volume_y_m3: Any) -> Any:
    """Validated outflow weights `(ny, nx, 4)` (direction order `DIRECTIONS`), each active cell with outflow
    summing to 1, others 0. Raises `TransportError` for non-finite, closed-face or outlet-inflow volumes."""
    from maple.core.backend import DeferredChecks

    checks = DeferredChecks()
    _volume_checks(network, water_volume_x_m3, water_volume_y_m3, checks)
    try:
        checks.resolve()
    except ValueError as exc:
        raise TransportError(str(exc)) from None
    weight, _ = _weights(network, water_volume_x_m3, water_volume_y_m3)
    ny, nx = network.shape
    return weight.reshape(ny, nx, 4)


def operation_count(n_substeps: int, n_bins: int) -> int:
    """Roundings per cell/class element feeding the mass identities. Per substep and slot: the product, decay
    (`expm1`/`exp` and product), assignment and sums into the cell total (about 12, as in the single-receiver
    operator) plus, for each of the four directions, the weight product, the receiver decay, the arrival products
    and the scatters into the receiver totals (about 12 each) -> 60 per slot (`B + 1` slots), plus the endpoint
    combinations. A declared count for `summation_error_bound_kg`, not a physical floor."""
    return int(n_substeps) * 60 * (int(n_bins) + 1) + 8


def validate_phase(network: FaceNetwork, mobile_kg: Any, phase: PhaseState) -> None:
    """Raise `TransportError` unless `mobile_kg` is a valid `(ny, nx, nc)` FP64 mobile field (finite, >= 0, 0 on
    inactive cells) and `phase` a valid partition of it (one batched flag read)."""
    from maple.core.backend import DeferredChecks, finite_flag, is_array, negative_flag, true_flag

    if not isinstance(network, FaceNetwork):
        raise TransportError("network must be a FaceNetwork")
    ny, nx = network.shape
    if not is_array(mobile_kg) or len(mobile_kg.shape) != 3 or mobile_kg.shape[-1] < 1:
        raise TransportError("mobile_kg must be a (ny, nx, nc) array")
    _require({"mobile_kg": mobile_kg}, (ny, nx, int(mobile_kg.shape[-1])), network.xp, np.float64)
    checks = DeferredChecks()
    checks.require(finite_flag(mobile_kg), "mobile_kg must be finite everywhere")
    checks.forbid(negative_flag(mobile_kg), "mobile_kg must be >= 0 everywhere")
    inactive = (~network.active_flat).reshape(ny, nx)[..., None]
    checks.forbid(true_flag(inactive & (mobile_kg != 0.0)), "mobile_kg must be 0 on inactive cells")
    _phase_checks(phase, mobile_kg, network, checks)
    try:
        checks.resolve()
    except ValueError as exc:
        raise TransportError(str(exc)) from None


# --- result -------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PondTransportStep:
    """`transport` is the unchanged `TransportStep` contract (`mobile_before_by_class_kg` is `sum (M + P)`).
    `phase_after` partitions `mobile_remaining_kg = T - Dep - E`. `mobile_before_kg = M + P`;
    `arrival_deposition_kg` is the part of the deposition booked in a cell by mass that entered it this step;
    `crossing_kg = Out + E` per source cell; `face_crossing_by_direction_kg` `(ny, nx, nc, 4)` is what each cell
    sent through each face (direction order `DIRECTIONS`, outlets included); `outflow_weight` `(ny, nx, 4)`;
    `blocked_cell` `(ny, nx)` bool: active cells without outflow this step (no advection progress)."""

    transport: TransportStep
    phase_after: PhaseState
    mobile_before_kg: Any
    mobile_remaining_kg: Any
    arrival_deposition_kg: Any
    crossing_kg: Any
    face_crossing_by_direction_kg: Any
    outflow_weight: Any
    blocked_cell: Any
    n_bins: int
    implementation: str


# --- array substep ---------------------------------------------------------------------------
def _substep_array(xp: ModuleType, net: FaceNetwork, W: Any, X: Any, P: Any, V: Any, R: Any, S: Any, wint: Any,
                   wexit: Any, dx: float, dt: float, nb: int) -> tuple:
    """One substep on masses `W (n, nc, B)`, positions `X`, pickup packet `P (n, nc)` at `x = 0`. `V` is the
    EFFECTIVE velocity (0 without outflow). `wint`/`wexit` `(n, 4)` split weights to interior/outlet faces.
    Generalises `characteristic_transport._substep_array` to four receivers."""
    from maple.core.backend import scatter_add, scatter_max, scatter_min

    n, nc = V.shape
    k0 = nb + 1
    Wc = xp.concatenate([W, P[..., None]], axis=-1)
    Xc = xp.concatenate([X, xp.zeros((n, nc, 1), dtype=np.float64)], axis=-1)
    S3, V3, R3 = S[..., None], V[..., None], R[..., None]
    travel = V3 * dt
    cross = (Xc + travel >= dx) & ~S3  # same expression as the non-crossing position below
    travel_src = xp.where(cross, dx - Xc, travel)
    lost = xp.where(S3, 0.0, Wc * (-xp.expm1(-R3 * travel_src)))
    survivor = xp.where(S3, 0.0, Wc * xp.exp(-R3 * travel_src))
    dep_decay = lost.sum(axis=-1)
    settled = xp.where(S3, Wc, 0.0).sum(axis=-1)
    crossing = xp.where(cross, survivor, 0.0)
    positive_v = V3 > 0.0
    # tau = (dx - x) / v <= dt mathematically; FP may exceed dt by round-off, so the remainder is floored at 0.
    remaining = xp.where(cross & positive_v, dt - travel_src / xp.where(positive_v, V3, 1.0), 0.0)
    remaining = xp.maximum(remaining, 0.0)

    self_cell = xp.broadcast_to(xp.arange(n, dtype=np.int64)[:, None, None], (n, nc, k0))
    masses = [xp.where(cross, 0.0, survivor)]
    positions = [Xc + travel]
    cells = [self_cell]
    export = xp.zeros((n, nc), dtype=np.float64)
    out_internal = xp.zeros((n, nc), dtype=np.float64)
    in_internal = xp.zeros((n, nc), dtype=np.float64)
    arrival_decay = xp.zeros((n, nc), dtype=np.float64)
    arrival_settled = xp.zeros((n, nc), dtype=np.float64)
    sent = []
    for d in range(4):
        receiver = net.neighbor[:, d]
        amount = crossing * wint[:, d][:, None, None]
        amount_exit = crossing * wexit[:, d][:, None, None]
        to_cell, to_exit = amount.sum(axis=-1), amount_exit.sum(axis=-1)
        export = export + to_exit
        out_internal = out_internal + to_cell
        S_r3 = S[receiver][..., None]
        R_r3 = R[receiver][..., None]
        distance = V[receiver][..., None] * remaining  # <= v_r dt <= courant_max dx < dx
        moving = xp.where(S_r3, 0.0, amount)
        lost_arrival = moving * (-xp.expm1(-R_r3 * distance))
        settled_arrival = xp.where(S_r3, amount, 0.0)
        scatter_add(in_internal, receiver, to_cell)
        scatter_add(arrival_decay, receiver, lost_arrival.sum(axis=-1))
        scatter_add(arrival_settled, receiver, settled_arrival.sum(axis=-1))
        sent.append(to_cell + to_exit)
        masses.append(moving * xp.exp(-R_r3 * distance))
        positions.append(distance)
        cells.append(xp.broadcast_to(receiver[:, None, None], (n, nc, k0)))

    dest_mass = xp.concatenate(masses, axis=-1)
    dest_x = xp.concatenate(positions, axis=-1)
    dest_cell = xp.concatenate(cells, axis=-1)
    has = dest_mass > 0.0
    dest_bin = bin_index(dest_x, dx, nb, xp)
    cls = xp.arange(nc, dtype=np.int64)[None, :, None]
    key = (dest_cell * nc + cls) * nb + dest_bin

    size = n * nc * nb
    mass_new = xp.zeros(size, dtype=np.float64)
    scatter_add(mass_new, key, dest_mass)
    xmax = xp.full(size, _BELOW_ANY_POSITION, dtype=np.float64)
    scatter_max(xmax, key, xp.where(has, dest_x, _BELOW_ANY_POSITION))
    above = 2.0 * dx
    xmin = xp.full(size, above, dtype=np.float64)
    scatter_min(xmin, key, xp.where(has, dest_x, above))
    R_bin = xp.broadcast_to(R[:, :, None], (n, nc, nb)).reshape(-1)
    r_cand = R_bin[key]  # the DESTINATION cell's r
    safe_cand = xp.where(has, dest_mass, 1.0)
    # NARROW form (shift by the largest position, expm1/log1p, accurate as r -> 0) and WIDE form (shift by the
    # largest log-weight, positive sum) selected per bin, as in characteristic_transport.
    weight = xp.where(has, xp.expm1(r_cand * (dest_x - xmax[key])), 0.0)
    dsum = xp.zeros(size, dtype=np.float64)
    scatter_add(dsum, key, dest_mass * weight)
    log_weight = xp.where(has, xp.log(safe_cand) + r_cand * dest_x, _BELOW_ANY_LOG_WEIGHT)
    smax = xp.full(size, _BELOW_ANY_LOG_WEIGHT, dtype=np.float64)
    scatter_max(smax, key, log_weight)
    esum = xp.zeros(size, dtype=np.float64)
    scatter_add(esum, key, xp.where(has, xp.exp(log_weight - smax[key]), 0.0))
    msum = xp.zeros(size, dtype=np.float64)
    scatter_add(msum, key, dest_mass * dest_x)
    occupied = mass_new > 0.0
    safe_mass = xp.where(occupied, mass_new, 1.0)
    safe_r = xp.where(R_bin > 0.0, R_bin, 1.0)
    narrow = xmax + xp.log1p(dsum / safe_mass) / safe_r
    wide = (smax + xp.log(xp.where(occupied, esum, 1.0)) - xp.log(safe_mass)) / safe_r
    x_log = xp.where(R_bin * (xmax - xmin) <= NARROW_SPREAD, narrow, wide)
    x_rep = xp.where(R_bin > 0.0, x_log, msum / safe_mass)
    x_rep = xp.minimum(xp.maximum(x_rep, xmin), xmax)  # round-off containment only
    x_rep = xp.where(occupied, x_rep, 0.0)
    return (mass_new.reshape(n, nc, nb), x_rep.reshape(n, nc, nb), dep_decay, settled, export, out_internal,
            in_internal, arrival_decay, arrival_settled, xp.stack(sent, axis=-1))


# --- step -----------------------------------------------------------------------------------
def pond_transport_step(
    network: FaceNetwork,
    mobile_before_kg: Any,
    phase: PhaseState,
    pickup_kg: Any,
    sediment_velocity_m_s: Any,
    deposition_rate_per_m: Any,
    settle_mask: Any,
    water_volume_x_m3: Any,
    water_volume_y_m3: Any,
    dt_s: float,
    *,
    n_substeps: int = 1,
    courant_max: float = DEFAULT_COURANT_MAX,
    implementation: str = "array",
    external_inflow_kg: Any = None,
) -> PondTransportStep:
    """Apply the multi-face characteristic operator for `dt_s` (module docstring).

    `mobile_before_kg` `(ny, nx, nc)` is the PRE-pickup MAPLE mobile mass that `phase` partitions; `pickup_kg`
    the ACTUAL class pickup of this step (never a demand), injected once at `x = 0` in the first substep.
    `sediment_velocity_m_s` (>= 0, m/s), `deposition_rate_per_m` (>= 0, 1/m, 0 = no deposition), FP64 and
    `settle_mask` bool, all `(ny, nx, nc)`. `water_volume_*_m3` are the signed actual face volumes of the
    accepted physical step. Pure: raises `TransportError` (or the recoverable `TransportStepRejected` for the
    Courant limit) before anything is returned; no input is modified."""
    from maple.core.backend import (
        DeferredChecks,
        errstate,
        finite_flag,
        freeze,
        negative_flag,
        pairwise_bound_scale_factor,
        pairwise_sum_over_leading_axes,
        scatter_add,
        true_flag,
    )

    if not isinstance(network, FaceNetwork):
        raise TransportError("network must be a FaceNetwork (use face_network*)")
    if external_inflow_kg is not None:
        raise TransportError("external sediment inflow is not supported in the initial pond transport scope")
    if implementation not in IMPLEMENTATIONS:
        raise TransportError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}; the "
                             "compiled Numba/CUDA kernels are not provided for the multi-face operator")
    xp = network.xp
    ny, nx = network.shape
    n = network.n_cells
    dt = _real(dt_s, "dt_s")
    if not (math.isfinite(dt) and dt > 0.0):
        raise TransportError(f"dt_s must be finite and > 0, got {dt_s!r}")
    cr_max = _real(courant_max, "courant_max")
    if not (0.0 < cr_max <= DEFAULT_COURANT_MAX):
        raise TransportError(f"courant_max must lie in (0, {DEFAULT_COURANT_MAX}] (one face crossing per packet "
                             f"per substep), got {courant_max!r}")
    n_sub = _strict_int(n_substeps, "n_substeps", 1, MAX_SUBSTEPS)
    shape = getattr(mobile_before_kg, "shape", None)
    nc = int(shape[-1]) if shape is not None and len(shape) == 3 else 0
    if nc < 1:
        raise TransportError(f"mobile_before_kg must be (ny, nx, nc) with nc >= 1, got shape {shape}")
    shape3 = (ny, nx, nc)
    floats = {"mobile_before_kg": mobile_before_kg, "pickup_kg": pickup_kg,
              "sediment_velocity_m_s": sediment_velocity_m_s, "deposition_rate_per_m": deposition_rate_per_m}
    _require(floats, shape3, xp, np.float64)
    _require({"settle_mask": settle_mask}, shape3, xp, np.bool_)

    checks = DeferredChecks()
    for name, array in floats.items():
        checks.require(finite_flag(array), f"{name} must be finite everywhere")
        checks.forbid(negative_flag(array), f"{name} must be >= 0 everywhere")
    _volume_checks(network, water_volume_x_m3, water_volume_y_m3, checks)
    _phase_checks(phase, mobile_before_kg, network, checks)
    nb = phase.n_bins
    active = network.active_flat
    inactive2 = (~active)[:, None]
    M = mobile_before_kg.reshape(n, nc)
    P = pickup_kg.reshape(n, nc)
    V_in = sediment_velocity_m_s.reshape(n, nc)
    R = deposition_rate_per_m.reshape(n, nc)
    S = settle_mask.reshape(n, nc)
    F = phase.fraction.reshape(n, nc, nb)
    X = phase.position_m.reshape(n, nc, nb)
    checks.forbid(true_flag(inactive2 & (M != 0.0)), "mobile_before_kg must be 0 on inactive cells")
    checks.forbid(true_flag(inactive2 & (P != 0.0)), "pickup_kg must be 0 on inactive cells")
    checks.forbid(true_flag(inactive2 & (V_in != 0.0)), "sediment_velocity_m_s must be 0 on inactive cells")

    dx = network.dx_m
    dt_sub = dt / n_sub
    with errstate(xp=xp, all="ignore"):
        weight, has_out = _weights(network, water_volume_x_m3, water_volume_y_m3)
        # Flow-blocking: no outflow face -> no advection progress (effective velocity 0); see module docstring.
        V = xp.where(has_out[:, None], V_in, 0.0)
        a = V * (dt_sub / dx)
        max_courant = xp.max(xp.where(active[:, None], a, 0.0))
        checks.require(finite_flag(a), "pond sediment Courant number overflowed FP64")
        checks.forbid(true_flag(active[:, None] & (a > cr_max)),
                      f"{_COURANT_REJECTION} courant_max = {cr_max}; step rejected (retry with a smaller dt or "
                      "more substeps)")
        exponent = (V * dt_sub) * R  # v dt <= courant_max dx is bounded, so only an extreme r can overflow
        max_exponent = xp.max(xp.where(active[:, None], exponent, 0.0))
        checks.require(finite_flag(exponent), "deposition hazard v r dt overflowed FP64")
    try:
        checks.resolve()
    except ValueError as exc:
        message = str(exc)
        raise (TransportStepRejected if message.startswith(_COURANT_REJECTION) else TransportError)(message) from None

    with errstate(xp=xp, all="ignore"):
        wint = xp.where(network.outlet_dir, 0.0, weight)
        wexit = xp.where(network.outlet_dir, weight, 0.0)
        W = M[..., None] * F
        Xw = X
        dep_decay = xp.zeros((n, nc), dtype=np.float64)
        settled = xp.zeros((n, nc), dtype=np.float64)
        export = xp.zeros((n, nc), dtype=np.float64)
        out_internal = xp.zeros((n, nc), dtype=np.float64)
        in_internal = xp.zeros((n, nc), dtype=np.float64)
        arrival = xp.zeros((n, nc), dtype=np.float64)
        sent = xp.zeros((n, nc, 4), dtype=np.float64)
        no_pickup = xp.zeros((n, nc), dtype=np.float64)
        for k in range(n_sub):
            packet = P if k == 0 else no_pickup  # pickup injected ONCE
            (W, Xw, d_decay, d_settled, d_export, d_out, d_in, d_arr_decay, d_arr_settled,
             d_sent) = _substep_array(xp, network, W, Xw, packet, V, R, S, wint, wexit, dx, dt_sub, nb)
            dep_decay += d_decay + d_arr_decay
            settled += d_settled + d_arr_settled
            export += d_export
            out_internal += d_out
            in_internal += d_in
            arrival += d_arr_decay + d_arr_settled
            sent += d_sent

        remaining = W.sum(axis=-1)
        deposition = dep_decay + settled
        T = (remaining + deposition) + export
        before = M + P
        divergence = in_internal - out_internal
        cell_balance = (T - before) - divergence
        cell_scale = xp.maximum(xp.maximum(before, T), xp.maximum(in_internal, out_internal))
        n_ops = operation_count(n_sub, nb)
        cell_tol = summation_error_bound_kg(n_ops, cell_scale)

        before_total = pairwise_sum_over_leading_axes(before)
        after_total = pairwise_sum_over_leading_axes(T)
        dep_total = pairwise_sum_over_leading_axes(deposition)
        exp_total = pairwise_sum_over_leading_axes(export)
        in_total = pairwise_sum_over_leading_axes(in_internal)
        out_total = pairwise_sum_over_leading_axes(out_internal)
        residual = after_total - before_total
        pairwise = pairwise_bound_scale_factor(n)
        tolerance = summation_error_bound_kg(
            n_ops, xp.maximum(xp.maximum(before_total, after_total), xp.maximum(in_total, out_total))) \
            + 2.0 * pairwise * (before_total + after_total)

        occupied = remaining > 0.0
        canonical = xp.zeros((1, 1, nb), dtype=np.float64)
        canonical[..., 0] = 1.0
        fraction_new = xp.where(occupied[..., None], W / xp.where(occupied, remaining, 1.0)[..., None], canonical)
        position_new = xp.where(W > 0.0, Xw, 0.0)

        # Face crossings in MAPLE layout; each direction scatters onto its nominal face (closed faces get 0).
        crossing = out_internal + export
        x_gross = xp.zeros((ny, nx + 1, nc), dtype=np.float64)
        y_gross = xp.zeros((ny + 1, nx, nc), dtype=np.float64)
        x_net = xp.zeros((ny, nx + 1, nc), dtype=np.float64)
        y_net = xp.zeros((ny + 1, nx, nc), dtype=np.float64)
        for d in range(4):
            gross, net = (x_gross, x_net) if _DIR_IS_X[d] else (y_gross, y_net)
            index = (network.face_row[:, d], network.face_col[:, d])
            scatter_add(gross, index, sent[:, :, d])
            scatter_add(net, index, sent[:, :, d] * _DIR_SIGN[d])

    checks = DeferredChecks()
    produced = (("mobile_after_transfer", T), ("deposition_request", deposition), ("export_request", export),
                ("internal_transfer_in", in_internal), ("internal_transfer_out", out_internal),
                ("x_face_gross", x_gross), ("y_face_gross", y_gross), ("x_face_net", x_net),
                ("y_face_net", y_net), ("decay_deposition", dep_decay), ("settled", settled),
                ("phase fraction", fraction_new), ("phase position", position_new), ("remaining mass", W),
                ("direction crossing", sent))
    for name, array in produced:
        checks.require(finite_flag(array), f"pond transport produced non-finite {name}")
    for name, array in produced:
        if name not in ("x_face_net", "y_face_net"):
            checks.forbid(negative_flag(array), f"pond transport produced negative {name}")
    checks.forbid(true_flag(export > T), "export request exceeds the post-transfer pool")
    checks.forbid(true_flag(deposition > T), "deposition request exceeds the post-transfer pool")
    checks.forbid(true_flag((deposition + export) > T * (1.0 + 4.0 * _EPS)),
                  "deposition plus export requests exceed the post-transfer pool")
    checks.forbid(true_flag(xp.abs(cell_balance) > cell_tol),
                  "per-cell mobile balance T - (M + P) = in - out violated beyond FP64 tolerance")
    checks.require(finite_flag(residual), "per-class budget residual is non-finite")
    checks.forbid(true_flag(xp.abs(residual) > tolerance),
                  "per-class mobile mass not conserved by the transfer beyond the declared FP64 bound")
    checks.forbid(true_flag(xp.abs(fraction_new.sum(axis=-1) - 1.0) > fraction_sum_tolerance(nb)),
                  "phase fractions after the step do not sum to 1 within the declared bound")
    checks.forbid(true_flag(position_new >= dx), "phase position after the step reached dx")
    bins = xp.arange(nb, dtype=np.int64).reshape(1, 1, nb)
    checks.forbid(true_flag((W > 0.0) & (bin_index(position_new, dx, nb, xp) != bins)),
                  "phase position after the step left its bin")
    checks.forbid(true_flag(export[~network.outlet_cell_flat] != 0.0), "export requested at a cell without outlet")
    try:
        checks.resolve()
    except ValueError as exc:
        raise TransportError(str(exc)) from None

    def grid(array):
        return array.reshape(shape3)

    transport = TransportStep(
        dt_s=dt, n_substeps=n_sub,
        mobile_after_transfer_kg=grid(T), deposition_request_kg=grid(deposition), export_request_kg=grid(export),
        decay_deposition_kg=grid(dep_decay), settled_kg=grid(settled),
        internal_transfer_in_kg=grid(in_internal), internal_transfer_out_kg=grid(out_internal),
        divergence_kg=grid(divergence),
        x_face_gross_kg=x_gross, y_face_gross_kg=y_gross, x_face_net_kg=x_net, y_face_net_kg=y_net,
        mobile_before_by_class_kg=before_total, mobile_after_by_class_kg=after_total,
        deposition_request_by_class_kg=dep_total, export_request_by_class_kg=exp_total,
        budget_residual_by_class_kg=residual, budget_tolerance_by_class_kg=tolerance,
        max_courant=max_courant, max_decay_exponent=max_exponent,
        max_cell_balance_residual_kg=xp.max(xp.abs(cell_balance)),
    )
    phase_after = PhaseState(fraction=freeze(fraction_new.reshape(ny, nx, nc, nb)),
                             position_m=freeze(position_new.reshape(ny, nx, nc, nb)),
                             dx_m=dx, n_bins=nb, xp=xp)
    return PondTransportStep(
        transport=transport, phase_after=phase_after, mobile_before_kg=grid(before),
        mobile_remaining_kg=grid(remaining), arrival_deposition_kg=grid(arrival), crossing_kg=grid(crossing),
        face_crossing_by_direction_kg=sent.reshape(ny, nx, nc, 4), outflow_weight=weight.reshape(ny, nx, 4),
        blocked_cell=(active & ~has_out).reshape(ny, nx), n_bins=nb, implementation=implementation)
