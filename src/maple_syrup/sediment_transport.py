"""Conservative Eulerian mobile-sediment transport on the D4 graph (Phase 5a).

This is the lateral operator `T` of the Phase 1 interface contract
(docs/phase1/interface_contract.md section 4.1): it moves class-resolved
mobile mass between cells of a `maple_syrup.routing.RoutingGraph` and
returns the deposition and export it wants MAPLE to apply. It never
touches the bed: `apply_water_process_demand` does, with
`water_demand_from_transport` building the demand.

Reference physics, MAHLERAN 1.2.3 `Subroutines_Sediment/flow_distrib.for`
(read-only): detached mass is spread along the downslope D4 path with an
exponential step-length distribution of mean `L` (44-46, 102-116), and
the remainder travels in `d_soil` at the virtual velocity `v_soil`
(`route_sediment_xml.f90` 211-299). A particle moving at `v` with an
exponential travel distance of mean `L` has a deposition HAZARD `v / L`
per second: mean lifetime `L / v`, mean distance `L`. That continuum law,

    dW/dt = -(v / L) W - div(v W),

is what this operator discretizes. The legacy's instantaneous walk (the
whole pattern deposited in the pickup step), its non-depositing pool, its
`nmax_*` / ring / 1e-19 cut-offs, the `d_soil < 0` clip (220-222,
289-291) and the zeroing of dry pools (225-229) are NOT copied.

Scheme, per substep `dt` with `v`, `r = 1 / L` and `settle_mask` held
fixed (Strang splitting: exact reaction half-step, conservative explicit
upwind advection, exact reaction half-step):

    settle:  Dep += W,  W = 0                    where settle_mask (dry / no capacity)
    react:   Dep += W (1 - s),  W *= s,  s = exp(-v r dt / 2)
    advect:  a = v dt / dx (Courant, <= courant_max <= 1)
             cross = a W  -> receiver r(i), or export request E at an outlet
             W' = (1 - a) W + sum_{j -> i} cross_j
    react:   Dep += W' (1 - s),  W' *= s           (arrivals decay at their new cell's rate)
    settle:  Dep += W', W' = 0                    where settle_mask (dry arrivals)
    T(M) = W' + Dep + E                           (requests stay in the pool of their cell)

What this gives (tested in tests/phase5/test_sediment_transport.py):

- The deposition TIMESCALE is exact: with uniform `v r` and no boundary
  export (reaction only, i.e. nothing reaches an outlet during the step)
  every substep multiplies the in-domain pool by exactly `exp(-v r dt)`
  whatever the Courant number; export at outlets removes mass from the
  in-domain pool in addition, so a 1 kg pulse with `L = 0.01 m`, `v = 0.001 m/s`
  retains `exp(-10)` kg after 100 s (the previous crossing-only scheme
  retained 0.82 kg: its hazard was `(v/dx)(1 - exp(-dx/L))`, wrong for
  `L << dx`). The mean deposition time is `L / v`.
- Advection timing: mean residence per cell is `dx / v` (geometric
  residence), so the mean arrival time along a path is the physical
  `sum dx / v_i`; the spread is upwind numerical diffusion, zero at
  `a = 1`.
- The SPATIAL deposition pattern is a discretization: it converges to the
  legacy exponential-in-distance bins as `dx -> 0` (first order in `dx`,
  the upwind residence being exponential rather than fixed) and the whole
  step converges to the continuous-time generator as `dt -> 0`. Exact
  coarse-cell bins are NOT claimed; for `L << dx` mass deposits in its
  source cell except an upwind leak: the fraction of a pool that ever
  leaves its cell is `a s_h / (1 - (1 - a) s_h^2)` with
  `s_h = exp(-v r dt / 2)`, which vanishes with `L / dx` (0.0197 for
  the reproducer above).
- Conservative and non-negative by construction: every operation is a
  product by a factor in [0, 1] or a sum of such products, so `W >= 0`,
  `Dep + E <= T(M)` per cell, `sum T(M) = sum M` per class up to FP64
  rounding (checked against a declared bound and reported, never
  absorbed), and `T(M) - M = In - Out` per cell.
- Dry or no-capacity cells settle their whole pool (including arrivals of
  the same step) through the returned deposition request; nothing is
  erased or clipped; a negative, non-finite or inactive-cell mobile input
  is refused before any result exists.

Courant: `transport_step` validates its OWN Courant number
`v dt / (n_substeps dx) <= courant_max` on every active cell and class
and raises `TransportStepRejected` (recoverable) otherwise. It is NOT
implied by an accepted water step: the physics module's recession memory
decays the previous sediment velocity without capping it at the new
water velocity (legacy behaviour retained), and the water step checks
the OLD-flux Courant number while the new velocity can be larger. The
caller must retry the whole unpublished coupled step with a smaller dt or
more substeps.

All arrays stay in the network's NumPy/CuPy namespace; the loop is over
substeps, never cells; one batched flag read per call. Face crossings use
MAPLE's canonical face layout (`x` faces `(ny, nx + 1, nc)`, index c is
the face WEST of column c, `+x` east; `y` faces `(ny + 1, nx, nc)`, index
r is the face SOUTH of row r, `+y` north) and include the export crossing
at outlets, so `T(M) - M = face divergence + export request`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

from maple_syrup.routing import ASPECT_STEPS, EXPORT, RoutingGraph

__all__ = [
    "CELL_BALANCE_RTOL",
    "DEFAULT_COURANT_MAX",
    "MAX_SUBSTEPS",
    "TransportError",
    "TransportNetwork",
    "TransportStep",
    "TransportStepRejected",
    "reaction_survival",
    "transport_network",
    "transport_step",
    "water_demand_from_transport",
]

_EPS = float(np.finfo(np.float64).eps)
# Per cell and substep: settle, two reaction half-steps (2 ops each), cross,
# stay, in, sum -> about ten roundings of quantities no larger than the
# local scale.
CELL_BALANCE_RTOL = 32.0 * _EPS
DEFAULT_COURANT_MAX = 1.0
MAX_SUBSTEPS = 1_000_000


class TransportError(ValueError):
    """Invalid network, inputs or failed balance. Raised before any result
    is returned; no caller-owned array is modified."""


class TransportStepRejected(TransportError):
    """The Courant number `v dt / (n_substeps dx)` exceeds `courant_max` on
    some active cell/class. Retry the SAME state with a smaller dt or more
    substeps; every other `TransportError` is not recoverable that way."""


_COURANT_REJECTION = "sediment Courant number v dt / (n_substeps dx) exceeds"


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TransportError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def reaction_survival(sediment_velocity_m_s: Any, deposition_rate_per_m: Any, dt_s: float) -> Any:
    """Fraction of a mobile pool still mobile after `dt_s` under the
    deposition hazard `v r` (`r = 1 / L`): `exp(-v r dt)`. Exact for
    constant coefficients; the per-second rate `v / L` follows from an
    exponential travel distance of mean `L` covered at speed `v`."""
    from maple.core.backend import array_namespace, is_array

    operands = [a for a in (sediment_velocity_m_s, deposition_rate_per_m) if is_array(a)]
    xp = array_namespace(*operands) if operands else np
    return xp.exp(-(sediment_velocity_m_s * deposition_rate_per_m) * _real(dt_s, "dt_s"))


# --- network ------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class TransportNetwork:
    """Flat D4 connectivity of a `RoutingGraph` for the transport operator,
    placed once in the graph namespace `xp`.

    `receiver_index` `(n,)` int64: receiver flat index for internal cells,
    the cell itself elsewhere (outlets, inactive; they never send);
    `outlet_flat`, `active_flat` `(n,)` bool. Face scatter tables: cells
    whose out-face is an x face (`cells_x`, scatter index `(rows_x,
    faces_x)` into `(ny, nx + 1, nc)`, `sign_x` +1 east) and a y face
    (`cells_y`, scatter index `(faces_y, cols_y)` into `(ny + 1, nx, nc)`,
    `sign_y` +1 north). `graph_input_sha256` binds the graph.
    """

    shape: tuple[int, int]
    dx_m: float
    n_cells: int
    n_active: int
    receiver_index: Any
    outlet_flat: Any
    active_flat: Any
    cells_x: Any
    rows_x: Any
    faces_x: Any
    sign_x: Any
    cells_y: Any
    faces_y: Any
    cols_y: Any
    sign_y: Any
    receiver_host: np.ndarray
    outlet_host: np.ndarray
    active_host: np.ndarray
    graph_input_sha256: str
    xp: ModuleType


def transport_network(graph: RoutingGraph) -> TransportNetwork:
    """Build the transport connectivity from a routing graph (host work
    once; runtime arrays placed with `to_device`)."""
    from maple.core.backend import freeze, to_device

    if not isinstance(graph, RoutingGraph):
        raise TransportError("graph must be a maple_syrup.routing.RoutingGraph")
    ny, nx = graph.shape
    n = ny * nx
    receiver = np.asarray(graph.receiver, dtype=np.int64).reshape(-1)
    active = np.asarray(graph.active, dtype=np.bool_).reshape(-1)
    outlet = np.asarray(graph.outlet, dtype=np.bool_).reshape(-1)
    aspect = np.asarray(graph.aspect, dtype=np.int64).reshape(-1)
    internal = active & ~outlet
    if np.any(internal & (receiver < 0)):
        raise TransportError("internal error: an active non-outlet cell has no receiver")
    if np.any(outlet & (receiver != EXPORT)):
        raise TransportError("internal error: an outlet cell does not export")
    self_index = np.arange(n, dtype=np.int64)
    receiver_index = np.where(internal, receiver, self_index)

    rows, cols = np.divmod(self_index, nx)
    # Out-face of every active cell from its aspect (routing.ASPECT_STEPS:
    # 1 = N (+row), 2 = E (+col), 3 = S, 4 = W). x faces: index c is the
    # face WEST of cell (r, c); y faces: index r is the face SOUTH of (r, c).
    step_r = np.array([0] + [ASPECT_STEPS[k][0] for k in (1, 2, 3, 4)])[aspect]
    step_c = np.array([0] + [ASPECT_STEPS[k][1] for k in (1, 2, 3, 4)])[aspect]
    is_x = active & (step_c != 0)
    is_y = active & (step_r != 0)
    cells_x = self_index[is_x]
    faces_x = np.where(step_c[is_x] > 0, cols[is_x] + 1, cols[is_x])
    sign_x = np.where(step_c[is_x] > 0, 1.0, -1.0)
    cells_y = self_index[is_y]
    faces_y = np.where(step_r[is_y] > 0, rows[is_y] + 1, rows[is_y])
    sign_y = np.where(step_r[is_y] > 0, 1.0, -1.0)
    if faces_x.size and (faces_x.min() < 0 or faces_x.max() > nx):
        raise TransportError("internal error: x face index out of range")
    if faces_y.size and (faces_y.min() < 0 or faces_y.max() > ny):
        raise TransportError("internal error: y face index out of range")

    xp = graph.xp

    def runtime(a, dtype):
        return freeze(to_device(np.ascontiguousarray(np.asarray(a, dtype=dtype)), xp))

    return TransportNetwork(
        shape=(ny, nx), dx_m=graph.dx_m, n_cells=n, n_active=int(active.sum()),
        receiver_index=runtime(receiver_index, np.int64),
        outlet_flat=runtime(outlet, np.bool_), active_flat=runtime(active, np.bool_),
        cells_x=runtime(cells_x, np.int64), rows_x=runtime(rows[is_x], np.int64),
        faces_x=runtime(faces_x, np.int64), sign_x=runtime(sign_x, np.float64),
        cells_y=runtime(cells_y, np.int64), faces_y=runtime(faces_y, np.int64),
        cols_y=runtime(cols[is_y], np.int64), sign_y=runtime(sign_y, np.float64),
        receiver_host=freeze(receiver.reshape(ny, nx).copy()),
        outlet_host=freeze(outlet.reshape(ny, nx).copy()),
        active_host=freeze(active.reshape(ny, nx).copy()),
        graph_input_sha256=graph.input_sha256, xp=xp,
    )


# --- step ---------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class TransportStep:
    """Result of `transport_step`. Class arrays `(ny, nx, nc)` FP64 in the
    network namespace; per-class vectors `(nc,)`; scalars 0-d there.

    mobile_after_transfer_kg   T(M): the pool to publish into
                               `WaterState.mobile_mass_by_cell_class_kg`
                               BEFORE `apply_water_process_demand`; it still
                               contains the deposition and export requests
    deposition_request_kg      to request from MAPLE (<= T(M) per cell)
    export_request_kg          nonzero only at outlets (<= T(M) - deposition)
    decay_deposition_kg / settled_kg   the two parts of the deposition
                               (exponential hazard v/L; settle mask)
    internal_transfer_in_kg / _out_kg     mass arriving / leaving through
                               internal faces; `divergence_kg = in - out`
                               and `T(M) - M = divergence_kg`
    x_face_gross_kg (ny, nx+1, nc), y_face_gross_kg (ny+1, nx, nc),
    x_face_net_kg, y_face_net_kg   MAPLE face layout incl. export crossings
    *_by_class_kg, budget_residual_by_class_kg, budget_tolerance_by_class_kg
    max_courant                max v dt / (n_substeps dx) over active cells
    max_decay_exponent         max v r dt / n_substeps (reaction is exact
                               for any value; reported for the splitting
                               error estimate)
    max_cell_balance_residual_kg
    """

    dt_s: float
    n_substeps: int
    mobile_after_transfer_kg: Any
    deposition_request_kg: Any
    export_request_kg: Any
    decay_deposition_kg: Any
    settled_kg: Any
    internal_transfer_in_kg: Any
    internal_transfer_out_kg: Any
    divergence_kg: Any
    x_face_gross_kg: Any
    y_face_gross_kg: Any
    x_face_net_kg: Any
    y_face_net_kg: Any
    mobile_before_by_class_kg: Any
    mobile_after_by_class_kg: Any
    deposition_request_by_class_kg: Any
    export_request_by_class_kg: Any
    budget_residual_by_class_kg: Any
    budget_tolerance_by_class_kg: Any
    max_courant: Any
    max_decay_exponent: Any
    max_cell_balance_residual_kg: Any


def _require(named: dict[str, Any], shape: tuple[int, ...], xp: ModuleType, dtype) -> None:
    from maple.core.backend import MixedArrayNamespaceError, array_namespace, is_array

    for name, array in named.items():
        if not is_array(array):
            raise TransportError(f"{name} must be a NumPy/CuPy array, got {type(array).__name__}")
    try:
        namespace = array_namespace(*named.values())
    except MixedArrayNamespaceError as exc:
        raise TransportError(str(exc)) from None
    if namespace is not xp:
        raise TransportError(f"arrays must be in the network namespace {xp.__name__!r}, got {namespace.__name__!r}")
    for name, array in named.items():
        if tuple(array.shape) != shape:
            raise TransportError(f"{name} shape {tuple(array.shape)} != {shape}")
        if array.dtype != dtype:
            raise TransportError(f"{name} must be {np.dtype(dtype)}, got {array.dtype}")


def transport_step(
    network: TransportNetwork,
    mobile_kg: Any,
    sediment_velocity_m_s: Any,
    deposition_rate_per_m: Any,
    settle_mask: Any,
    dt_s: float,
    *,
    courant_max: float = DEFAULT_COURANT_MAX,
    n_substeps: int = 1,
) -> TransportStep:
    """Apply the lateral operator for `dt_s` (see the module docstring).

    `mobile_kg`, `sediment_velocity_m_s` (>= 0), `deposition_rate_per_m`
    (>= 0, `1 / L`; 0 = no deposition law) are `(ny, nx, nc)` FP64;
    `settle_mask` is `(ny, nx, nc)` bool. Mobile mass and velocity must
    be 0 on inactive cells. Pure: on any failure raises `TransportError`
    (or `TransportStepRejected` for the Courant limit) before returning
    anything."""
    from maple.core.backend import (
        DeferredChecks,
        errstate,
        finite_flag,
        negative_flag,
        pairwise_bound_scale_factor,
        pairwise_sum_over_leading_axes,
        scatter_add,
        true_flag,
    )

    if not isinstance(network, TransportNetwork):
        raise TransportError("network must be a TransportNetwork (use transport_network)")
    xp = network.xp
    ny, nx = network.shape
    n = network.n_cells
    dt = _real(dt_s, "dt_s")
    if not (math.isfinite(dt) and dt > 0.0):
        raise TransportError(f"dt_s must be finite and > 0, got {dt_s!r}")
    cr_max = _real(courant_max, "courant_max")
    if not (0.0 < cr_max <= 1.0):
        raise TransportError(f"courant_max must lie in (0, 1] (explicit upwind positivity), got {courant_max!r}")
    if isinstance(n_substeps, bool) or not isinstance(n_substeps, (int, np.integer)) \
            or not (1 <= int(n_substeps) <= MAX_SUBSTEPS):
        raise TransportError(f"n_substeps must be an int in [1, {MAX_SUBSTEPS}], got {n_substeps!r}")
    n_sub = int(n_substeps)
    nc = int(mobile_kg.shape[-1]) if hasattr(mobile_kg, "shape") and len(mobile_kg.shape) == 3 else 0
    if nc < 1:
        raise TransportError(f"mobile_kg must be (ny, nx, nc) with nc >= 1, got shape {getattr(mobile_kg, 'shape', None)}")
    shape3 = (ny, nx, nc)
    floats = {"mobile_kg": mobile_kg, "sediment_velocity_m_s": sediment_velocity_m_s,
              "deposition_rate_per_m": deposition_rate_per_m}
    _require(floats, shape3, xp, np.float64)
    _require({"settle_mask": settle_mask}, shape3, xp, np.bool_)

    checks = DeferredChecks()
    for name, array in floats.items():
        checks.require(finite_flag(array), f"{name} must be finite everywhere")
        checks.forbid(negative_flag(array), f"{name} must be >= 0 everywhere")
    active = network.active_flat
    outlet = network.outlet_flat
    inactive3 = (~active)[:, None]
    M = mobile_kg.reshape(n, nc)
    V = sediment_velocity_m_s.reshape(n, nc)
    R = deposition_rate_per_m.reshape(n, nc)
    S = settle_mask.reshape(n, nc)
    checks.forbid(true_flag(inactive3 & (M != 0.0)), "mobile_kg must be 0 on inactive cells")
    checks.forbid(true_flag(inactive3 & (V != 0.0)), "sediment_velocity_m_s must be 0 on inactive cells")

    dx = network.dx_m
    dt_sub = dt / n_sub
    with errstate(xp=xp, all="ignore"):
        a = V * (dt_sub / dx)
        max_courant = xp.max(xp.where(active[:, None], a, 0.0))
        checks.require(finite_flag(a), "sediment Courant number overflowed FP64")
        checks.forbid(true_flag(active[:, None] & (a > cr_max)),
                      f"{_COURANT_REJECTION} courant_max = {cr_max}; step rejected (retry with a smaller dt "
                      "or more substeps)")
        exponent = (V * R) * dt_sub  # v r dt: exact reaction, no restriction
        max_exponent = xp.max(xp.where(active[:, None], exponent, 0.0))
        checks.require(finite_flag(exponent), "deposition hazard v r dt overflowed FP64")
        s_half = xp.exp(-0.5 * exponent)
        s_half = xp.where(S, 1.0, s_half)  # settled cells settle everything; no separate decay

        W = xp.array(M, copy=True)
        dep_decay = xp.zeros((n, nc), dtype=np.float64)
        settled = xp.zeros((n, nc), dtype=np.float64)
        export = xp.zeros((n, nc), dtype=np.float64)
        out_internal = xp.zeros((n, nc), dtype=np.float64)
        in_internal = xp.zeros((n, nc), dtype=np.float64)
        receiver = network.receiver_index
        for _ in range(n_sub):
            # settle (dry / no capacity): the whole pool is a deposition request
            settled += xp.where(S, W, 0.0)
            W = xp.where(S, 0.0, W)
            # reaction half-step: exact exponential hazard v / L
            kept = W * s_half
            dep_decay += W - kept
            W = kept
            # conservative explicit upwind advection along the D4 receivers
            cross = a * W
            stay = W - cross
            exported = xp.where(outlet[:, None], cross, 0.0)
            internal = xp.where(outlet[:, None], 0.0, cross)
            export += exported
            inflow = xp.zeros((n, nc), dtype=np.float64)
            scatter_add(inflow, receiver, internal)
            out_internal += internal
            in_internal += inflow
            W = stay + inflow
            # reaction half-step on the advected pool (arrivals decay at their new cell's hazard)
            kept = W * s_half
            dep_decay += W - kept
            W = kept
        settled += xp.where(S, W, 0.0)  # dry arrivals of the last substep
        W = xp.where(S, 0.0, W)

        deposition = dep_decay + settled
        T = (W + deposition) + export
        divergence = in_internal - out_internal
        cell_balance = (T - M) - divergence
        cell_scale = M + in_internal + out_internal
        cell_tol = CELL_BALANCE_RTOL * (n_sub + 1) * cell_scale

        # Per-class budgets with pairwise sums and a declared bound.
        before = pairwise_sum_over_leading_axes(M)
        after = pairwise_sum_over_leading_axes(T)
        dep_total = pairwise_sum_over_leading_axes(deposition)
        exp_total = pairwise_sum_over_leading_axes(export)
        in_total = pairwise_sum_over_leading_axes(in_internal)
        out_total = pairwise_sum_over_leading_axes(out_internal)
        residual = after - before
        pairwise = pairwise_bound_scale_factor(n)
        tolerance = CELL_BALANCE_RTOL * (n_sub + 1) * (before + in_total + out_total) \
            + 2.0 * pairwise * (before + after)

        # Face crossings (internal + export) in MAPLE layout: x faces indexed
        # (row, face column), y faces indexed (face row, column).
        crossing = out_internal + export
        x_gross = xp.zeros((ny, nx + 1, nc), dtype=np.float64)
        y_gross = xp.zeros((ny + 1, nx, nc), dtype=np.float64)
        x_net = xp.zeros((ny, nx + 1, nc), dtype=np.float64)
        y_net = xp.zeros((ny + 1, nx, nc), dtype=np.float64)
        vx = crossing[network.cells_x]
        vy = crossing[network.cells_y]
        scatter_add(x_gross, (network.rows_x, network.faces_x), vx)
        scatter_add(x_net, (network.rows_x, network.faces_x), vx * network.sign_x[:, None])
        scatter_add(y_gross, (network.faces_y, network.cols_y), vy)
        scatter_add(y_net, (network.faces_y, network.cols_y), vy * network.sign_y[:, None])

    for name, array in (("mobile_after_transfer", T), ("deposition_request", deposition),
                        ("export_request", export), ("internal_transfer_in", in_internal),
                        ("internal_transfer_out", out_internal), ("x_face_gross", x_gross),
                        ("y_face_gross", y_gross), ("x_face_net", x_net), ("y_face_net", y_net),
                        ("decay_deposition", dep_decay), ("settled", settled)):
        checks.require(finite_flag(array), f"transport produced non-finite {name}")
    for name, array in (("mobile_after_transfer", T), ("deposition_request", deposition),
                        ("export_request", export), ("internal_transfer_in", in_internal),
                        ("internal_transfer_out", out_internal), ("x_face_gross", x_gross),
                        ("y_face_gross", y_gross), ("decay_deposition", dep_decay), ("settled", settled)):
        checks.forbid(negative_flag(array), f"transport produced negative {name}")
    checks.forbid(true_flag(export > T), "export request exceeds the post-transfer pool")
    checks.forbid(true_flag(deposition > T), "deposition request exceeds the post-transfer pool")
    checks.forbid(true_flag((deposition + export) > T * (1.0 + 4.0 * _EPS)),
                  "deposition plus export requests exceed the post-transfer pool")
    checks.forbid(true_flag(xp.abs(cell_balance) > cell_tol), "per-cell mobile balance T - M = in - out violated "
                                                               "beyond FP64 tolerance")
    checks.require(finite_flag(residual), "per-class budget residual is non-finite")
    checks.forbid(true_flag(xp.abs(residual) > tolerance),
                  "per-class mobile mass not conserved by the transfer beyond the declared FP64 bound")
    try:
        checks.resolve()
    except ValueError as exc:
        message = str(exc)
        raise (TransportStepRejected if message.startswith(_COURANT_REJECTION) else TransportError)(message) from None

    def grid(array):
        return array.reshape(shape3)

    return TransportStep(
        dt_s=dt, n_substeps=n_sub,
        mobile_after_transfer_kg=grid(T), deposition_request_kg=grid(deposition), export_request_kg=grid(export),
        decay_deposition_kg=grid(dep_decay), settled_kg=grid(settled),
        internal_transfer_in_kg=grid(in_internal), internal_transfer_out_kg=grid(out_internal),
        divergence_kg=grid(divergence),
        x_face_gross_kg=x_gross, y_face_gross_kg=y_gross, x_face_net_kg=x_net, y_face_net_kg=y_net,
        mobile_before_by_class_kg=before, mobile_after_by_class_kg=after,
        deposition_request_by_class_kg=dep_total, export_request_by_class_kg=exp_total,
        budget_residual_by_class_kg=residual, budget_tolerance_by_class_kg=tolerance,
        max_courant=max_courant, max_decay_exponent=max_exponent,
        max_cell_balance_residual_kg=xp.max(xp.abs(cell_balance)),
    )


def water_demand_from_transport(step: TransportStep, requested_pickup_kg: Any, *, hydraulic_diagnostics: Any = None):
    """`maple.water.interfaces.WaterProcessDemand` for one accounting step:
    removal = the physics pickup demand (or zeros when pickup was applied
    in a separate MAPLE call), deposition = the transport deposition
    request, boundary export = the export request, face flux = the
    transport crossings. Apply it with `apply_water_process_demand(...,
    water=WaterState(depth, T(M)), ...)` after publishing
    `step.mobile_after_transfer_kg` as the mobile pool (interface contract
    section 4.1, steps 2-4)."""
    from maple.aeolian.flux.face_flux import FaceFluxResult
    from maple.core.backend import (
        MixedArrayNamespaceError,
        array_namespace,
        finite_flag,
        is_array,
        negative_flag,
        read_flags,
    )
    from maple.water.interfaces import WaterProcessDemand

    if not isinstance(step, TransportStep):
        raise TransportError("step must be a TransportStep")
    pool = step.mobile_after_transfer_kg
    if not is_array(requested_pickup_kg) or tuple(requested_pickup_kg.shape) != tuple(pool.shape) \
            or requested_pickup_kg.dtype != np.float64:
        raise TransportError(f"requested_pickup_kg must be float64 with shape {tuple(pool.shape)}")
    try:
        array_namespace(requested_pickup_kg, pool)
    except MixedArrayNamespaceError as exc:
        raise TransportError(str(exc)) from None
    finite, negative = read_flags([finite_flag(requested_pickup_kg), negative_flag(requested_pickup_kg)])
    if not finite:
        raise TransportError("requested_pickup_kg must be finite everywhere")
    if negative:
        raise TransportError("requested_pickup_kg must be >= 0 everywhere")
    return WaterProcessDemand(
        requested_removal_by_cell_class_kg=requested_pickup_kg,
        requested_deposition_by_cell_class_kg=step.deposition_request_kg,
        hydraulic_diagnostics=hydraulic_diagnostics,
        face_flux=FaceFluxResult(
            x_face_crossing_mass_kg=step.x_face_gross_kg,
            y_face_crossing_mass_kg=step.y_face_gross_kg,
            x_face_net_crossing_mass_kg=step.x_face_net_kg,
            y_face_net_crossing_mass_kg=step.y_face_net_kg,
        ),
        boundary_export_by_cell_class_kg=step.export_request_kg,
    )
